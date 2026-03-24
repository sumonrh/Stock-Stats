import React, { useState, useEffect, useMemo } from 'react';
import { ComposedChart, Scatter, Line, BarChart, Bar, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend, Label } from 'recharts';
import ReactApexChart from 'react-apexcharts';
import Plot from 'react-plotly.js';
import { runTfjsPipeline, loadModelFromStorage, loadModelFromFiles, downloadModelFiles, evaluateLstmOnData, evaluateIntradayModel, runIntradayTfjsPipeline, saveModelToServer, runMaxExcursionPipeline, runSensitivityAnalysis, predictMaxExcursion } from './utils/tfjsEngine';
import { findBestFitRegression, getExpectedCumulativeVolumePercentage } from './utils/mathUtils';
import MarketRegime from './MarketRegime';
import QuantBacktestTab from './components/QuantBacktestTab';
import BacktestSimulationTab from './components/BacktestSimulationTab';

// --- CONFIGURATION ---
const INITIAL_TICKERS = ['VICR', 'RKLB', 'PL', 'ASTS', 'SEDG', 'MU', 'IREN', 'BE', 'LITE', 'OKLO', 'QBTS', 'WDC', 'EOSE', 'INTC', 'COHR', 'FIX', 'AU', 'VSCO', 'WPM'];

const BINS = [
  { label: '< 0.5x', min: 0, max: 0.5 },
  { label: '0.5x - 1.0x', min: 0.5, max: 1.0 },
  { label: '1.0x - 1.5x', min: 1.0, max: 1.5 },
  { label: '1.5x - 2.0x', min: 1.5, max: 2.0 },
  { label: '2.0x - 3.0x', min: 2.0, max: 3.0 },
  { label: '3.0x - 5.0x', min: 3.0, max: 5.0 },
  { label: '> 5.0x', min: 5.0, max: Infinity }
];

// --- MATH HELPERS ---
const getMedian = (arr) => {
  if (!arr || arr.length === 0) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

const getMean = (arr) => {
  if (!arr || arr.length === 0) return null;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
};

// Helper for Normal Distribution PDF
const getNormalDistribution = (x, mean, stdDev) => {
  if (stdDev === 0) return 0;
  const variance = stdDev * stdDev;
  const factor = 1 / Math.sqrt(2 * Math.PI * variance);
  const exponent = -Math.pow(x - mean, 2) / (2 * variance);
  return factor * Math.exp(exponent);
};

// --- DATA PROCESSING ---
// Applies the specific Abs Max Excursion from open / ADR Dollar Math
const processTickerData = (data, ticker, rvolPeriod, adrPeriod) => {
  const result = [];
  const maxDaysNeeded = Math.max(rvolPeriod, adrPeriod);

  for (let i = 0; i < data.length; i++) {
    let current = { ...data[i], ticker };

    // Absolute maximum excursion from open (either high or low)
    const highExcursion = Math.abs(current.high - current.open);
    const lowExcursion = Math.abs(current.low - current.open);
    current.maxAbsoluteExcursion = Math.max(highExcursion, lowExcursion);

    // Calculate rolling averages (shifted by 1 day)
    if (i >= maxDaysNeeded) {
      let volSum = 0;
      let adrDollarSum = 0;

      for (let j = 1; j <= rvolPeriod; j++) {
        volSum += data[i - j].volume;
      }
      for (let j = 1; j <= adrPeriod; j++) {
        adrDollarSum += (data[i - j].high - data[i - j].low); // Summing historical dollar ranges
      }

      current.avgVol = volSum / rvolPeriod;
      current.adrDollar = adrDollarSum / adrPeriod;

      if (current.adrDollar > 0 && current.avgVol > 0) {
        current.rVol = current.volume / current.avgVol;

        // NEW MATH: Max Absolute Excursion from open / (ADR in dollar value)
        current.maxExcursionAdr = current.maxAbsoluteExcursion / current.adrDollar;
        current.dateStr = typeof current.date === 'string' ? current.date.split('T')[0] : current.date.toISOString().split('T')[0];
        current.rvolPeriod = rvolPeriod;
        current.adrPeriod = adrPeriod;

        // Filter out absolute anomalies for cleaner regression/charting
        if (current.rVol < 25 && current.maxExcursionAdr < 15) {
          result.push(current);
        }
      }
    }
  }
  return result;
};

// --- DATA PROCESSING WITH RS ---
const processTickerDataWithRs = (data, ticker, rvolPeriod, adrPeriod, spyData) => {
  const result = [];
  const maxDaysNeeded = Math.max(rvolPeriod, adrPeriod, 252); // Ensure we have enough for RS weighting

  // Build a fast lookup for SPY closes by date
  const spyMap = {};
  for (const s of spyData) {
    const dStr = typeof s.date === 'string' ? s.date.split('T')[0] : s.date.toISOString().split('T')[0];
    spyMap[dStr] = s.close;
  }

  for (let i = 0; i < data.length; i++) {
    let current = { ...data[i], ticker };

    // Absolute maximum excursion from open (either high or low)
    const highExcursion = Math.abs(current.high - current.open);
    const lowExcursion = Math.abs(current.low - current.open);
    current.maxAbsoluteExcursion = Math.max(highExcursion, lowExcursion);

    // Calculate rolling averages and RS
    if (i >= maxDaysNeeded) {
      const getStockRet = (days) => {
        const oldPrice = data[i - days].close;
        return oldPrice ? ((data[i].close - oldPrice) / oldPrice) * 100 : 0;
      };

      const getSpyRet = (days) => {
        const dateNow = typeof data[i].date === 'string' ? data[i].date.split('T')[0] : data[i].date.toISOString().split('T')[0];
        const dateOld = typeof data[i - days].date === 'string' ? data[i - days].date.split('T')[0] : data[i - days].date.toISOString().split('T')[0];
        const spyNow = spyMap[dateNow];
        const spyOld = spyMap[dateOld];
        return (spyNow && spyOld) ? ((spyNow - spyOld) / spyOld) * 100 : 0;
      };

      const stockWeights = (1 + getStockRet(63) / 100) * 0.4 + (1 + getStockRet(126) / 100) * 0.2 + (1 + getStockRet(189) / 100) * 0.2 + (1 + getStockRet(252) / 100) * 0.2;
      const spyWeights = (1 + getSpyRet(63) / 100) * 0.4 + (1 + getSpyRet(126) / 100) * 0.2 + (1 + getSpyRet(189) / 100) * 0.2 + (1 + getSpyRet(252) / 100) * 0.2;

      current.rsRating = spyWeights > 0 ? stockWeights / spyWeights : 1.0;

      let volSum = 0;
      let adrDollarSum = 0;

      for (let j = 1; j <= rvolPeriod; j++) {
        volSum += data[i - j].volume;
      }
      for (let j = 1; j <= adrPeriod; j++) {
        adrDollarSum += (data[i - j].high - data[i - j].low);
      }

      current.avgVol = volSum / rvolPeriod;
      current.adrDollar = adrDollarSum / adrPeriod;

      if (current.adrDollar > 0 && current.avgVol > 0) {
        current.rVol = current.volume / current.avgVol;
        current.maxExcursionAdr = current.maxAbsoluteExcursion / current.adrDollar;
        current.dateStr = typeof current.date === 'string' ? current.date.split('T')[0] : current.date.toISOString().split('T')[0];

        if (current.rVol < 25 && current.maxExcursionAdr < 15) {
          result.push(current);
        }
      }
    }
  }
  return result;
};



// --- MAIN COMPONENT ---
export default function App() {
  const [rawMarketData, setRawMarketData] = useState({});
  const [loading, setLoading] = useState(true);
  const [availableTickers, setAvailableTickers] = useState(INITIAL_TICKERS);
  const [searchInput, setSearchInput] = useState('');
  const [selectedTickerFilter, setSelectedTickerFilter] = useState(INITIAL_TICKERS[0]);
  const [rvolPeriod, setRvolPeriod] = useState(50);
  const [adrPeriod, setAdrPeriod] = useState(20);
  const [customRvolThreshold, setCustomRvolThreshold] = useState('1.5');

  // AI Training State
  const [isTrainingAi, setIsTrainingAi] = useState(false);
  const [aiTrainingEpoch, setAiTrainingEpoch] = useState(0);
  const [aiTrainingLoss, setAiTrainingLoss] = useState(0);
  const [aiTotalEpochs, setAiTotalEpochs] = useState(50);
  const [aiResults, setAiResults] = useState(null);

  // Manual AI Prediction State
  const [manualAiInput, setManualAiInput] = useState('2.0');
  const [manualAiResult, setManualAiResult] = useState(null);

  // --- INTRADAY NEW STATE ---
  const [activeTab, setActiveTab] = useState('daily'); // 'daily' | 'intraday' | 'quantBacktest' | 'backtest'
  const [intradayFiles, setIntradayFiles] = useState([]);
  const [intradayData, setIntradayData] = useState({}); // { ticker: data }
  const [selectedIntradayTicker, setSelectedIntradayTicker] = useState('ALL');
  const [minutesToUse, setMinutesToUse] = useState(30);

  const [isIntradayLoading, setIsIntradayLoading] = useState(false);
  const [isIntradayTraining, setIsIntradayTraining] = useState(false);
  const [intradayTrainEpoch, setIntradayTrainEpoch] = useState(0);
  const [intradayTrainLoss, setIntradayTrainLoss] = useState(0);
  const [intradayAiResults, setIntradayAiResults] = useState(null);

  const [manualIntradayInputStr, setManualIntradayInputStr] = useState('500000, 200000, 150000');
  const [manualIntradayResult, setManualIntradayResult] = useState(null);
  const [intradaySaveStatus, setIntradaySaveStatus] = useState(null);

  // --- MAX EXCURSION PREDICTOR STATE ---
  const [isMaxExcTraining, setIsMaxExcTraining] = useState(false);
  const [maxExcTrainEpoch, setMaxExcTrainEpoch] = useState(0);
  const [maxExcTotalEpochs, setMaxExcTotalEpochs] = useState(80);
  const [maxExcTrainLoss, setMaxExcTrainLoss] = useState(0);
  const [maxExcResults, setMaxExcResults] = useState(null);
  const [maxExcSensitivity, setMaxExcSensitivity] = useState(null);
  const [isRunningSensitivity, setIsRunningSensitivity] = useState(false);
  const [selectedMaxExcTicker, setSelectedMaxExcTicker] = useState('ALL');
  const [maxExcMinutes, setMaxExcMinutes] = useState(5);
  const [maxExcDailyFeatures, setMaxExcDailyFeatures] = useState({});
  const [spyData, setSpyData] = useState([]);
  const [manualMaxExcInput, setManualMaxExcInput] = useState({ open: '', volume: '' });
  const [maxExcPrediction, setMaxExcPrediction] = useState(null);
  const [enabledFeatures, setEnabledFeatures] = useState(
    new Array(15).fill(true) // one boolean per feature, all on by default
  );
  const [maxExcSaveStatus, setMaxExcSaveStatus] = useState(null);
  const [predTicker, setPredTicker] = useState('');
  const [predChartData, setPredChartData] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  // --------------------------

  // Initial Load
  useEffect(() => {
    loadInitialData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadInitialData = async () => {
    setLoading(true);

    // Fetch available tickers from cache
    let cachedTickers = [];
    try {
      const cacheRes = await fetch('/api/data-loader/cache');
      if (cacheRes.ok) {
        cachedTickers = await cacheRes.json();
      }
    } catch (e) { console.error("Failed to load cached tickers", e); }

    let tickersToUse = cachedTickers.length > 0 ? cachedTickers : INITIAL_TICKERS;
    setAvailableTickers(tickersToUse);
    setSelectedTickerFilter(tickersToUse[0] || 'ALL');

    // Only load the very first ticker on initial startup to improve load speeds
    const firstTicker = tickersToUse[0];
    const newRawData = {};
    if (firstTicker) {
      newRawData[firstTicker] = await fetchRawTickerData(firstTicker);
    }

    const spyRaw = await fetchRawTickerData('SPY');
    setSpyData(spyRaw);

    setRawMarketData(newRawData);
    setLoading(false);

    // Auto-load model from IndexedDB on startup
    const cached = await loadModelFromStorage();
    if (cached) {
      // Create empty/mock result just to show it's loaded, 
      // or evaluate it immediately if we had processed data.
      // Easiest is to set a "pending" evaluation:
      setAiResults({
        ...cached,
        finalLoss: 0,
        predictionsMap: [],
        mse: 0
      });
      // A dedicated effect below will spot this and re-evaluate the predictions!
    }

    fetchIntradayFiles();
  };

  const fetchIntradayFiles = async () => {
    try {
      const res = await fetch('/api/intraday-files');
      if (res.ok) {
        const files = await res.json();
        setIntradayFiles(files);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchIntradayData = async (ticker) => {
    try {
      const res = await fetch(`/api/intraday-data?ticker=${ticker}`);
      if (res.ok) {
        const data = await res.json();
        return data.map(day => ({ ...day, ticker }));
      }
    } catch (e) {
      console.error(e);
    }
    return [];
  };

  const ensureIntradayData = async (ticker) => {
    setIsIntradayLoading(true);
    let currentData = { ...intradayData };
    if (ticker === 'ALL') {
      for (const f of intradayFiles) {
        if (!currentData[f.ticker]) {
          currentData[f.ticker] = await fetchIntradayData(f.ticker);
        }
      }
    } else {
      if (!currentData[ticker]) {
        currentData[ticker] = await fetchIntradayData(ticker);
      }
    }
    setIntradayData(currentData);
    setIsIntradayLoading(false);
    return currentData;
  };

  const volumeProfileDisplayData = useMemo(() => {
    if (!intradayData || Object.keys(intradayData).length === 0) return [];

    let keysToUse = [];
    if (selectedIntradayTicker === 'ALL') {
      keysToUse = Object.keys(intradayData);
    } else {
      if (intradayData[selectedIntradayTicker]) {
        keysToUse = [selectedIntradayTicker];
      }
    }

    const timeMap = {};
    for (const key of keysToUse) {
      const days = intradayData[key];
      if (!days) continue;
      for (const day of days) {
        if (!day.bars || day.totalVolume <= 0) continue;
        for (const bar of day.bars) {
          if (!timeMap[bar.time]) {
            timeMap[bar.time] = { time: bar.time, totalFraction: 0, count: 0 };
          }
          timeMap[bar.time].totalFraction += (bar.volume / day.totalVolume);
          timeMap[bar.time].count += 1;
        }
      }
    }

    const profile = Object.values(timeMap).map(t => ({
      time: t.time.substring(0, 5),
      avgFraction: t.count > 0 ? t.totalFraction / t.count : 0
    }));

    profile.sort((a, b) => a.time.localeCompare(b.time));

    const sumFrac = profile.reduce((acc, p) => acc + p.avgFraction, 0);
    if (sumFrac > 0) {
      profile.forEach(p => p.avgFraction = p.avgFraction / sumFrac);
    }

    // Now compute prediction based on user INTRADAY ACTUAL VOLUME input
    let base = profile.map(p => ({ ...p, predictedVolume: null, predictedVisual: null }));
    if (!manualIntradayInputStr) return { chart: base, mathProjectedVol: null };

    const parts = manualIntradayInputStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n >= 0);
    if (parts.length === 0) return { chart: base, mathProjectedVol: null };

    let inputSum = 0;
    let baseFracSum = 0;
    for (let i = 0; i < parts.length; i++) {
      if (base[i]) {
        base[i].predictedVolume = parts[i];
        base[i].predictedVisual = parts[i];
        inputSum += parts[i];
        baseFracSum += base[i].avgFraction;
      } else break;
    }

    let mathProjectedVol = null;
    if (baseFracSum > 0) {
      mathProjectedVol = inputSum / baseFracSum;
    }

    if (parts.length < base.length && parts.length > 0 && mathProjectedVol > 0) {
      for (let i = parts.length; i < base.length; i++) {
        base[i].predictedVolume = base[i].avgFraction * mathProjectedVol;
        base[i].predictedVisual = base[i].predictedVolume;
      }
    }

    // Assign scaled average representation across all bars so chart comparisons are visible
    const visualScale = mathProjectedVol > 0 ? mathProjectedVol : (inputSum > 0 ? inputSum * 10 : 1000000);
    base.forEach(b => {
      b.scaledAvgVolume = b.avgFraction * visualScale;
    });

    return { chart: base, mathProjectedVol };
  }, [intradayData, selectedIntradayTicker, manualIntradayInputStr]);

  // The Fetch wrapper: Hits backend directly, throws error if backend fails
  const fetchRawTickerData = async (ticker) => {
    try {
      // Backend integration point
      const response = await fetch(`/api/yahoo-finance2?ticker=${ticker}`);
      if (!response.ok) throw new Error("Backend not available");
      let rawData = await response.json();
      rawData = rawData.map(d => ({ ...d, date: new Date(d.date) }));
      return rawData;
    } catch (err) {
      console.error(`Backend fetch failed for ${ticker}. Ensure the backend server is running.`);
      // Return empty array to prevent app crash, 
      // but do NOT fallback to generated mock data.
      return [];
    }
  };

  const handleAddCustomTicker = async (e) => {
    e.preventDefault();
    const symbol = searchInput.trim().toUpperCase();
    if (!symbol) return;

    if (availableTickers.includes(symbol)) {
      setSelectedTickerFilter(symbol);
      setSearchInput('');
      return;
    }

    setLoading(true);

    // Append to CSV in Quant backtester stock data
    try {
      await fetch('/api/data-loader/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers: [symbol] })
      });
    } catch (e) {
      console.error("Failed to append CSV", e);
    }

    const rawData = await fetchRawTickerData(symbol);

    setAvailableTickers(prev => prev.includes(symbol) ? prev : [...prev, symbol]);
    setRawMarketData(prev => ({ ...prev, [symbol]: rawData }));
    setSelectedTickerFilter(symbol);
    setSearchInput('');
    setLoading(false);
  };

  // Process data when raw data or periods change
  const getProcessedData = (marketData) => {
    let allProcessed = [];
    for (const [ticker, rawData] of Object.entries(marketData)) {
      if (spyData && spyData.length > 0) {
        allProcessed = allProcessed.concat(processTickerDataWithRs(rawData, ticker, rvolPeriod, adrPeriod, spyData));
      } else {
        allProcessed = allProcessed.concat(processTickerData(rawData, ticker, rvolPeriod, adrPeriod));
      }
    }
    return allProcessed;
  };

  const processedData = useMemo(() => getProcessedData(rawMarketData), [rawMarketData, rvolPeriod, adrPeriod]);

  // Filter and sort data for Recharts (Sorting by X axis is crucial for Line charts)
  const { chartData, historicalRegression } = useMemo(() => {
    const filtered = selectedTickerFilter === 'ALL'
      ? processedData
      : processedData.filter(d => d.ticker === selectedTickerFilter);

    // Sort ascending by RVol
    const sortedChartData = filtered.sort((a, b) => a.rVol - b.rVol);

    // Calculate Best Fit Regression
    const points = sortedChartData.map(d => ({ x: d.rVol, y: d.maxExcursionAdr }));
    const regression = findBestFitRegression(points, 4, 2, false);

    return { chartData: sortedChartData, historicalRegression: regression };
  }, [processedData, selectedTickerFilter]);

  // Whenever chartData changes, if we have a loaded model, re-evaluate the current filtered data so predictions map properly
  useEffect(() => {
    if (aiResults?.model && aiResults?.preparedData && chartData?.length > 10) {
      const evaluate = async () => {
        // Prevent infinite loops by only re-evaluating if predictionsMap length doesn't match chartData length (rough heuristic)
        // Or better, just only auto-evaluate if the user changes the ticker/periods.
        // Actually, just calling evaluateLstmOnData is very fast (no training).
        const result = await evaluateLstmOnData(aiResults.model, aiResults.preparedData, chartData);
        if (result && result.predictionsMap && result.predictionsMap.length !== aiResults.predictionsMap?.length) {
          setAiResults(result);
        }
      };
      evaluate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartData]);


  // AI Regression Tracker
  const aiRegression = useMemo(() => {
    if (!aiResults || !aiResults.predictionsMap || aiResults.predictionsMap.length < 3) return null;
    const points = aiResults.predictionsMap.map(d => ({ x: d.rVol, y: d.lstmPredictedExc }));
    return findBestFitRegression(points, 4);
  }, [aiResults]);

  // Generate Statistical Summary Table Data
  const summaryStats = useMemo(() => {
    const totalDataPoints = chartData.length;

    return BINS.map(bin => {
      const bucketData = chartData.filter(d => d.rVol >= bin.min && d.rVol < bin.max);
      const excursions = bucketData.map(d => d.maxExcursionAdr);

      const medianExc = getMedian(excursions);
      const meanExc = getMean(excursions);

      const rvolProbability = totalDataPoints > 0 ? (bucketData.length / totalDataPoints) * 100 : 0;

      // Calculate the empirical probability of hitting at least this median excursion across the entire dataset
      const overallExcursionsGreater = medianExc != null ? chartData.filter(d => d.maxExcursionAdr >= medianExc).length : 0;
      const medianExcProbability = totalDataPoints > 0 && medianExc != null ? (overallExcursionsGreater / totalDataPoints) * 100 : 0;

      return {
        label: bin.label,
        sampleSize: bucketData.length,
        rvolProbability: rvolProbability.toFixed(2),
        medianExcursion: medianExc?.toFixed(2),
        medianExcursionProb: medianExcProbability.toFixed(2),
        meanExcursion: meanExc?.toFixed(2),
        maxExcursion: excursions.length ? Math.max(...excursions).toFixed(2) : undefined
      };
    });
  }, [chartData]);

  // Calculate RVol Distribution for Bell Curve
  const handleTrainAI = async () => {
    setIsTrainingAi(true);
    setAiTrainingEpoch(0);
    setAiResults(null);
    setManualAiResult(null); // Clear previous
    try {
      const results = await runTfjsPipeline(chartData, aiTotalEpochs, (epoch, total, loss) => {
        setAiTrainingEpoch(epoch);
        setAiTotalEpochs(total);
        setAiTrainingLoss(loss);
      });
      setAiResults(results);
    } catch (err) {
      console.error("AI Training Error", err);
    }
    setIsTrainingAi(false);
  };

  const handleDownloadModel = async () => {
    if (!aiResults || !aiResults.model || !aiResults.preparedData) return;
    await downloadModelFiles(aiResults.model, aiResults.preparedData);
  };

  const handleUploadModel = async (e) => {
    const files = e.target.files;
    if (files.length !== 3) {
      alert("Please select exactly 3 files: model.json, weights.bin, and stock-lstm-meta.json");
      return;
    }

    let modelFile = null;
    let weightsFile = null;
    let metaFile = null;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.name.endsWith('json') && file.name.includes('meta')) metaFile = file;
      else if (file.name.endsWith('json') && file.name.includes('model')) modelFile = file;
      else if (file.name.endsWith('bin')) weightsFile = file;
    }

    if (!modelFile || !weightsFile || !metaFile) {
      alert("Missing one of the required files. Make sure to select model.json, weights.bin, AND stock-lstm-meta.json correctly.");
      return;
    }

    try {
      const result = await loadModelFromFiles(modelFile, weightsFile, metaFile);
      if (result) {
        // Re-eval on current data
        const evaluation = await evaluateLstmOnData(result.model, result.preparedData, chartData);
        setAiResults(evaluation || {
          ...result, finalLoss: 0, mse: 0, predictionsMap: []
        });
        alert("Model successfully loaded and applied!");
      }
    } catch (err) {
      console.error("Failed to load custom files:", err);
      alert("Failed to load model from the provided files.");
    }
  };

  const handleManualPrediction = async () => {
    if (!aiResults || !aiResults.model || !aiResults.preparedData) return;
    const inputVal = parseFloat(manualAiInput);
    if (isNaN(inputVal)) return;

    // Normalizing the manual input based on the exact same constraints the model was built on
    const { excMin, excMax, rvolMin, rvolMax, lastSequence } = aiResults.preparedData;

    // We construct a sequence where the "past" is the last known sequence of the dataset, 
    // but the "today" trigger is the user's manual input normalized.
    // If the input is outside the bounds of what the model was trained on, normalize might extrapolate, which is fine for ML.
    let normInputRvol = (inputVal - rvolMin) / (rvolMax - rvolMin);
    // cap it functionally so it doesn't cause infinity
    if (rvolMax - rvolMin === 0) normInputRvol = 0;

    // Construct exactly as training: [pastRVol, pastExc, todaysRVol]
    // We clone the base 'lastSequence' used in training to simulate the rolling history
    const simulatedSequence = lastSequence.map((step, idx) => {
      if (idx === lastSequence.length - 1) {
        return [step[0], step[1], normInputRvol];
      }
      return [step[0], step[1], 0];
    });

    try {
      // dynamic import for tf to avoid making App.jsx fully dependent on tfjs tree statically
      const tf = await import('@tensorflow/tfjs');
      const numFeatures = 3;
      const seqLength = simulatedSequence.length;
      const flatSeq = new Float32Array(seqLength * numFeatures);
      for (let i = 0; i < seqLength; i++) {
        flatSeq[i * numFeatures] = simulatedSequence[i][0];
        flatSeq[i * numFeatures + 1] = simulatedSequence[i][1];
        flatSeq[i * numFeatures + 2] = simulatedSequence[i][2];
      }
      const inputTensor = tf.tensor3d(flatSeq, [1, seqLength, numFeatures]);
      const predTensor = aiResults.model.predict(inputTensor);
      const predVal = (await predTensor.data())[0];

      // Denormalize
      const finalExcursion = (predVal * (excMax - excMin)) + excMin;
      setManualAiResult(finalExcursion > 0 ? finalExcursion : 0); // floor at 0

      inputTensor.dispose();
      predTensor.dispose();
    } catch (err) {
      console.error("Manual inference failed", err);
    }
  };

  // Intraday Handlers
  const handleTrainIntraday = async () => {
    try {
      setIsIntradayLoading(true);
      const currentIntraday = await ensureIntradayData(selectedIntradayTicker);

      const tickersToEnsure = selectedIntradayTicker === 'ALL' ? intradayFiles.map(f => f.ticker) : [selectedIntradayTicker];
      const updatedRawData = { ...rawMarketData };

      for (const ticker of tickersToEnsure) {
        if (!updatedRawData[ticker]) {
          updatedRawData[ticker] = await fetchRawTickerData(ticker);
        }
      }
      setRawMarketData(updatedRawData);

      setIsIntradayTraining(true);
      setIntradayAiResults(null);
      setManualIntradayResult(null);

      let intradayListToUse = [];
      if (selectedIntradayTicker === 'ALL') {
        for (const key of Object.keys(currentIntraday)) {
          if (currentIntraday[key]) intradayListToUse = intradayListToUse.concat(currentIntraday[key]);
        }
      } else {
        intradayListToUse = currentIntraday[selectedIntradayTicker] || [];
      }

      const dailyDataToUse = getProcessedData(updatedRawData);

      const results = await runIntradayTfjsPipeline(
        intradayListToUse,
        dailyDataToUse,
        minutesToUse,
        50,
        (epoch, total, loss) => {
          setIntradayTrainEpoch(epoch);
          setIntradayTrainLoss(loss);
        }
      );

      if (!results) {
        alert("No valid overlapping date ranges found between Intraday CSVs and Daily Market Data. Ensure both datasets cover the same time periods.");
      } else {
        setIntradayAiResults(results);
      }
    } catch (err) {
      console.error("Intraday AI Training Error", err);
      alert("Failed to train intraday model: " + err.message);
    } finally {
      setIsIntradayLoading(false);
      setIsIntradayTraining(false);
    }
  };

  const handlePredictIntraday = async () => {
    if (!intradayAiResults || !intradayAiResults.model) {
      alert("Please click 'Train Prediction Model' first so the AI can analyze your data before it makes predictions!");
      return;
    }

    const parts = manualIntradayInputStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    const expectedBars = Math.floor(minutesToUse / 5);

    if (parts.length !== expectedBars) {
      alert(`The AI Model was trained on the first ${minutesToUse} minutes of the day (${expectedBars} bars).\nPlease enter exactly ${expectedBars} comma-separated actual volume values!`);
      return;
    }

    const fallbacks = processedData.filter(d => true);

    // Find the right historical Average Volume to map fraction predictions into real shares
    const isIndividualStock = selectedIntradayTicker !== 'ALL';
    const tickerToFind = isIndividualStock ? selectedIntradayTicker : (selectedTickerFilter !== 'ALL' ? selectedTickerFilter : null);

    let avgVolFallback = 1000000;
    let rsFallback = 1.0;
    if (tickerToFind) {
      const match = processedData.filter(d => d.ticker === tickerToFind);
      if (match.length > 0) {
        avgVolFallback = match[match.length - 1].avgVol;
        rsFallback = match[match.length - 1].rsRating || 1.0;
      }
    } else {
      if (fallbacks.length > 0) {
        avgVolFallback = fallbacks[fallbacks.length - 1].avgVol;
        rsFallback = fallbacks[fallbacks.length - 1].rsRating || 1.0;
      }
    }

    try {
      const res = await evaluateIntradayModel(intradayAiResults.model, intradayAiResults.preparedData, parts, avgVolFallback, rsFallback);
      setManualIntradayResult({
        rvol: res,
        predictedVolume: res * avgVolFallback,
        showRvol: !!tickerToFind
      });
    } catch (e) {
      console.error(e);
      alert("Error evaluating model.");
    }
  };

  // --- MAX EXCURSION HANDLERS ---
  const fetchDailyFeatures = async (ticker) => {
    try {
      const res = await fetch(`/api/intraday-features?ticker=${ticker}`);
      if (res.ok) return await res.json();
    } catch (e) {
      console.error(e);
    }
    return [];
  };

  const handleTrainMaxExcursion = async () => {
    const currentIntraday = await ensureIntradayData(selectedMaxExcTicker);

    const tickersToUse = selectedMaxExcTicker === 'ALL' ? intradayFiles.map(f => f.ticker) : [selectedMaxExcTicker];

    // Ensure daily data and features for all tickers
    setIsMaxExcTraining(true);
    setMaxExcResults(null);
    setMaxExcSensitivity(null);
    setMaxExcPrediction(null);

    try {
      // Fetch daily features for each ticker
      const updatedFeatures = { ...maxExcDailyFeatures };
      const updatedRawData = { ...rawMarketData };
      for (const ticker of tickersToUse) {
        if (!updatedFeatures[ticker]) {
          updatedFeatures[ticker] = await fetchDailyFeatures(ticker);
        }
        if (!updatedRawData[ticker]) {
          updatedRawData[ticker] = await fetchRawTickerData(ticker);
        }
      }
      setMaxExcDailyFeatures(updatedFeatures);
      setRawMarketData(updatedRawData);

      // Collect all intraday data
      let allIntraday = [];
      let allFeatures = [];
      for (const ticker of tickersToUse) {
        const days = currentIntraday[ticker] || [];
        allIntraday = allIntraday.concat(days.map(d => ({ ...d, ticker })));
        allFeatures = allFeatures.concat(updatedFeatures[ticker] || []);
      }

      const results = await runMaxExcursionPipeline(
        allIntraday,
        allFeatures,
        maxExcMinutes,
        maxExcTotalEpochs,
        (epoch, total, loss) => {
          setMaxExcTrainEpoch(epoch);
          setMaxExcTotalEpochs(total);
          setMaxExcTrainLoss(loss);
        },
        enabledFeatures
      );
      setMaxExcResults(results);
    } catch (err) {
      console.error('Max Excursion Training Error', err);
    }
    setIsMaxExcTraining(false);
  };


  const handleRunSensitivity = async () => {
    if (!maxExcResults || !maxExcResults.model || !maxExcResults.preparedData) return;
    setIsRunningSensitivity(true);
    try {
      const results = await runSensitivityAnalysis(maxExcResults.model, maxExcResults.preparedData);
      setMaxExcSensitivity(results);
    } catch (err) {
      console.error('Sensitivity Analysis Error', err);
    }
    setIsRunningSensitivity(false);
  };

  const handleManualMaxExcPrediction = async () => {
    if (!maxExcResults || !maxExcResults.model || !maxExcResults.preparedData) {
      alert('Please train the model first!');
      return;
    }
    const openPrice = parseFloat(manualMaxExcInput.open);
    const volume = parseFloat(manualMaxExcInput.volume);
    if (isNaN(openPrice) || isNaN(volume) || openPrice <= 0) {
      alert('Please enter valid open price and volume.');
      return;
    }

    // Get the latest features for the selected ticker
    const ticker = selectedMaxExcTicker !== 'ALL' ? selectedMaxExcTicker : (intradayFiles.length > 0 ? intradayFiles[0].ticker : null);
    if (!ticker) return;

    const features = maxExcDailyFeatures[ticker];
    if (!features || features.length === 0) return;
    const latestFeat = features[features.length - 1];

    const avgVol = latestFeat.avgVol50 || 1000000;
    const expectedPct = getExpectedCumulativeVolumePercentage(maxExcMinutes);
    const projectedDayVol = volume / expectedPct;
    const projectedRVol = projectedDayVol / avgVol;
    const firstBarVolRatio = volume / avgVol;

    const featureVector = [
      projectedRVol,
      latestFeat.prevCloseChange || 0,
      firstBarVolRatio,
      0.3, // placeholder early range ratio
      0,   // placeholder pct above open
      latestFeat.upDownRatio20 || 1, // 20-Day Up/Down Ratio
      openPrice > 0 ? latestFeat.adr20 / openPrice : 0,
      openPrice > 0 ? latestFeat.atr14 / openPrice : 0,
      latestFeat.atrDistEma10 || 0,
      latestFeat.atrDistEma20 || 0,
      latestFeat.atrDistEma50 || 0,
      latestFeat.vixOpen || 0,
      latestFeat.vixPctChange || 0,
      latestFeat.vixSma200 || 0,
      latestFeat.vixDistSma200 || 0
    ];

    try {
      const exc = await predictMaxExcursion(
        maxExcResults.model,
        maxExcResults.preparedData.normParams,
        featureVector,
        maxExcResults.preparedData.activeIndices
      );
      const adr = latestFeat.adr20 || 1;
      setMaxExcPrediction({
        excursionAdr: exc,
        predictedHigh: openPrice + exc * adr,
        predictedLow: openPrice - exc * adr,
        adr,
        projectedRVol
      });
    } catch (err) {
      console.error(err);
    }
  };

  const handleChartPrediction = async () => {
    if (!maxExcResults?.model || !maxExcResults?.preparedData) {
      alert('Please train the model first!');
      return;
    }
    const ticker = predTicker.trim().toUpperCase();
    if (!ticker) return;

    setIsPredicting(true);
    setPredChartData(null);

    try {
      // 1. Fetch daily OHLCV from Yahoo Finance
      const dailyRes = await fetch(`/api/yahoo-finance2?ticker=${ticker}`);
      const dailyData = await dailyRes.json();
      if (!dailyData || dailyData.length === 0) throw new Error('No daily data for ' + ticker);

      // 2. Fetch intraday 5-min bars
      const intraRes = await fetch(`/api/intraday-data?ticker=${ticker}`);
      const intraData = intraRes.ok ? await intraRes.json() : [];

      // 3. Fetch daily features (ADR, ATR, EMAs)
      const featRes = await fetch(`/api/intraday-features?ticker=${ticker}`);
      const featData = featRes.ok ? await featRes.json() : [];

      // Build date lookups
      const intraByDate = {};
      for (const day of intraData) {
        intraByDate[day.date] = day;
      }
      const featByDate = {};
      for (const feat of featData) {
        featByDate[feat.date] = feat;
      }

      const { normParams, activeIndices } = maxExcResults.preparedData;
      const maxBars = Math.floor(maxExcMinutes / 5);

      // Last 90 trading days
      const last90 = dailyData.slice(-90);

      const candlestick = [];
      const predHighDots = [];
      const predLowDots = [];
      let predictedCount = 0;

      for (const day of last90) {
        const dateStr = new Date(day.date).toISOString().split('T')[0];
        const ts = new Date(dateStr).getTime();

        candlestick.push({ x: ts, y: [day.open, day.high, day.low, day.close] });

        // Run prediction if we have intraday + features for this day
        const intra = intraByDate[dateStr];
        const feat = featByDate[dateStr];

        if (intra && feat && intra.bars && intra.bars.length >= maxBars &&
          feat.adr20 > 0 && feat.avgVol50 > 0 && intra.dayOpen > 0) {
          const earlyBars = intra.bars.slice(0, maxBars);
          const earlyVolSum = earlyBars.reduce((s, b) => s + b.volume, 0);
          const expectedPct = getExpectedCumulativeVolumePercentage(maxExcMinutes);
          const projectedDayVol = earlyVolSum / expectedPct;
          const projectedRVol = projectedDayVol / feat.avgVol50;
          const firstBarVolRatio = earlyBars[0].volume / feat.avgVol50;

          let earlyHigh = -Infinity, earlyLow = Infinity;
          for (const b of earlyBars) {
            if (b.high > earlyHigh) earlyHigh = b.high;
            if (b.low < earlyLow) earlyLow = b.low;
          }
          const earlyRangeOverAdr = feat.adr20 > 0 ? (earlyHigh - earlyLow) / feat.adr20 : 0;

          const lastEarlyClose = earlyBars[earlyBars.length - 1].close;
          const pctAboveOpen = intra.dayOpen > 0 ? ((lastEarlyClose - intra.dayOpen) / intra.dayOpen) * 100 : 0;

          const upDownRatio20 = feat.upDownRatio20 || 1;

          const normAdr = intra.dayOpen > 0 ? feat.adr20 / intra.dayOpen : 0;
          const normAtr = intra.dayOpen > 0 ? feat.atr14 / intra.dayOpen : 0;

          const featureVector = [
            projectedRVol, feat.prevCloseChange || 0, firstBarVolRatio,
            earlyRangeOverAdr, pctAboveOpen, upDownRatio20,
            normAdr, normAtr,
            feat.atrDistEma10 || 0, feat.atrDistEma20 || 0, feat.atrDistEma50 || 0,
            feat.vixOpen || 0, feat.vixPctChange || 0,
            feat.vixSma200 || 0, feat.vixDistSma200 || 0
          ];

          const exc = await predictMaxExcursion(maxExcResults.model, normParams, featureVector, activeIndices);
          const predHigh = intra.dayOpen + exc * feat.adr20;
          const predLow = intra.dayOpen - exc * feat.adr20;

          predHighDots.push({ x: ts, y: parseFloat(predHigh.toFixed(2)) });
          predLowDots.push({ x: ts, y: parseFloat(predLow.toFixed(2)) });
          predictedCount++;
        } else {
          // Push null values so ApexCharts keeps X-axis synchronization perfect
          predHighDots.push({ x: ts, y: null });
          predLowDots.push({ x: ts, y: null });
        }
      }

      setPredChartData({ candlestick, predHighDots, predLowDots, ticker, predictedCount, totalDays: last90.length });
    } catch (err) {
      console.error('Chart prediction error:', err);
      alert('Error: ' + err.message);
    }
    setIsPredicting(false);
  };

  // Generate Advanced Statistical Analysis
  const advancedStats = useMemo(() => {
    const thresholds = [1.5, 2.0, 2.5, 3.0];
    const totalDataPoints = chartData.length;

    if (totalDataPoints === 0) return [];

    return thresholds.map(threshold => {
      // Find days that met the RVol threshold
      const daysMeetingRvol = chartData.filter(d => d.rVol >= threshold);
      const denominator = daysMeetingRvol.length;

      // From those days, find how many ALSO hit High Excursion (>= 1.5)
      const daysMeetingBoth = daysMeetingRvol.filter(d => d.maxExcursionAdr >= 1.5);
      const numerator = daysMeetingBoth.length;

      // Calculate the conditional probability
      const probability = denominator > 0 ? (numerator / denominator) * 100 : 0;

      // Calculate overall frequency of this setup playing out relative to all days
      const overallFrequency = (numerator / totalDataPoints) * 100;

      // Calculate average excursion for this specific RVol group
      const avgExcursion = denominator > 0 ? daysMeetingRvol.reduce((sum, d) => sum + d.maxExcursionAdr, 0) / denominator : 0;

      return {
        threshold: threshold.toFixed(1),
        totalMatchingDays: denominator,
        highExcursionDays: numerator,
        probability: probability.toFixed(2),
        overallFrequency: overallFrequency.toFixed(2),
        avgExcursion: avgExcursion.toFixed(2)
      };
    });
  }, [chartData]);

  const customStat = useMemo(() => {
    const threshold = parseFloat(customRvolThreshold);
    const totalDataPoints = chartData.length;
    if (isNaN(threshold) || totalDataPoints === 0) return null;

    const daysMeetingRvol = chartData.filter(d => d.rVol >= threshold);
    const denominator = daysMeetingRvol.length;

    const daysMeetingBoth = daysMeetingRvol.filter(d => d.maxExcursionAdr >= 1.5);
    const numerator = daysMeetingBoth.length;

    const probability = denominator > 0 ? (numerator / denominator) * 100 : 0;
    const overallFrequency = (numerator / totalDataPoints) * 100;
    const avgExcursion = denominator > 0 ? daysMeetingRvol.reduce((sum, d) => sum + d.maxExcursionAdr, 0) / denominator : 0;

    return {
      threshold: customRvolThreshold,
      totalMatchingDays: denominator,
      highExcursionDays: numerator,
      probability: probability.toFixed(2),
      overallFrequency: overallFrequency.toFixed(2),
      avgExcursion: avgExcursion.toFixed(2),
    };
  }, [chartData, customRvolThreshold]);

  const aiRegressionResult = useMemo(() => {
    if (!intradayAiResults || !intradayAiResults.predictionsMap) return null;
    const points = intradayAiResults.predictionsMap.map(p => ({ x: p.actualVol, y: p.predictedVol }));
    const fit = findBestFitRegression(points, 4, 1, false);
    if (!fit) return null;

    // Create smoothly sorted line points
    const maxX = Math.max(...points.map(p => p.x));
    const step = maxX / 50;
    const lineData = [];
    for (let i = 0; i <= 50; i++) {
      let x = i * step;
      lineData.push({
        actualVol: x,
        regressionVol: fit.predict(x)
      });
    }
    return { fit, lineData };
  }, [intradayAiResults]);

  // Handle Loading State

  // Calculate RVol Distribution for Bell Curve
  const { rvolDistributionData, rvolStats } = useMemo(() => {
    if (!chartData || chartData.length === 0) return { rvolDistributionData: [], rvolStats: null };

    const rvols = chartData.map(d => d.rVol).filter(v => v != null && isFinite(v));
    if (rvols.length === 0) return { rvolDistributionData: [], rvolStats: null };

    const mean = getMean(rvols);
    const variance = rvols.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (rvols.length > 1 ? rvols.length - 1 : 1);
    const stdDev = Math.sqrt(variance);

    // Create histogram bins
    const minRvol = Math.floor(Math.min(...rvols) * 10) / 10;
    const maxRvol = Math.ceil(Math.max(...rvols) * 10) / 10;

    // Define bins
    const binCount = 40; // Higher granularity for smooth curve
    const step = Math.max(0.1, (maxRvol - minRvol) / binCount);

    const bins = [];
    for (let i = 0; i <= binCount; i++) {
      const binMin = minRvol + i * step;
      bins.push({
        rVolBin: parseFloat(binMin.toFixed(2)),
        count: 0,
      });
    }

    rvols.forEach(v => {
      const binIndex = Math.min(bins.length - 1, Math.floor((v - minRvol) / step));
      if (bins[binIndex]) {
        bins[binIndex].count += 1;
      }
    });

    // Scale normal distribution curve to match histogram height
    const maxCount = Math.max(...bins.map(b => b.count));
    const maxPdf = getNormalDistribution(mean, mean, stdDev);
    const scalingFactor = maxPdf > 0 ? maxCount / maxPdf : 1;

    const distributionData = bins.map(b => {
      const pdf = getNormalDistribution(b.rVolBin, mean, stdDev);
      return {
        ...b,
        bellCurve: parseFloat((pdf * scalingFactor).toFixed(2)), // Scaled probability
        probability: pdf * step // Bin probability (area under curve for this bin)
      };
    });

    return {
      rvolDistributionData: distributionData,
      rvolStats: { mean, stdDev }
    };
  }, [chartData]);

  // Calculate Excursion Distribution for Bell Curve
  const { excDistributionData, excStats } = useMemo(() => {
    if (!chartData || chartData.length === 0) return { excDistributionData: [], excStats: null };

    // Using maxExcursionAdr here instead of rVol
    const exc = chartData.map(d => d.maxExcursionAdr).filter(v => v != null && isFinite(v));
    if (exc.length === 0) return { excDistributionData: [], excStats: null };

    const mean = getMean(exc);
    const variance = exc.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (exc.length > 1 ? exc.length - 1 : 1);
    const stdDev = Math.sqrt(variance);

    const minExc = Math.floor(Math.min(...exc) * 10) / 10;
    const maxExc = Math.ceil(Math.max(...exc) * 10) / 10;

    const binCount = 40;
    const step = Math.max(0.1, (maxExc - minExc) / binCount);

    const bins = [];
    for (let i = 0; i <= binCount; i++) {
      const binMin = minExc + i * step;
      bins.push({
        excBin: parseFloat(binMin.toFixed(2)),
        count: 0,
      });
    }

    exc.forEach(v => {
      const binIndex = Math.min(bins.length - 1, Math.floor((v - minExc) / step));
      if (bins[binIndex]) {
        bins[binIndex].count += 1;
      }
    });

    const maxCount = Math.max(...bins.map(b => b.count));
    const maxPdf = getNormalDistribution(mean, mean, stdDev);
    const scalingFactor = maxPdf > 0 ? maxCount / maxPdf : 1;

    const distributionData = bins.map(b => {
      const pdf = getNormalDistribution(b.excBin, mean, stdDev);
      return {
        ...b,
        bellCurve: parseFloat((pdf * scalingFactor).toFixed(2)),
        probability: pdf * step
      };
    });

    return {
      excDistributionData: distributionData,
      excStats: { mean, stdDev }
    };
  }, [chartData]);

  // Combined Marginal Plotly Mapping
  const marginalPlotState = useMemo(() => {
    if (!chartData || chartData.length === 0) return null;

    const x = chartData.map(d => d.rVol);
    const y = chartData.map(d => d.maxExcursionAdr);

    const rvolStep = rvolDistributionData.length > 1 ? rvolDistributionData[1].rVolBin - rvolDistributionData[0].rVolBin : 0.1;
    const excStep = excDistributionData.length > 1 ? excDistributionData[1].excBin - excDistributionData[0].excBin : 0.1;

    const text = chartData.map(d => {
      const rvolPdf = rvolStats ? getNormalDistribution(d.rVol, rvolStats.mean, rvolStats.stdDev) : 0;
      const excPdf = excStats ? getNormalDistribution(d.maxExcursionAdr, excStats.mean, excStats.stdDev) : 0;

      const rvolProb = (Math.max(0, rvolPdf * rvolStep) * 100).toFixed(2);
      const excProb = (Math.max(0, excPdf * excStep) * 100).toFixed(2);

      return `<b>${d.ticker}</b> - ${d.dateStr}<br>RVol: ${d.rVol.toFixed(2)}x<br>Abs Excursion: ${d.maxExcursionAdr.toFixed(2)}x ADR<br>RVol Prob Density: ${rvolProb}%<br>Exc Prob Density: ${excProb}%`;
    });

    const traces = [];

    // Main Scatter points
    traces.push({
      x: x,
      y: y,
      mode: 'markers',
      type: 'scatter',
      name: 'Data Points',
      marker: { color: '#3b82f6', opacity: 0.6, size: 6, line: { color: '#2563eb', width: 1 } },
      text: text,
      hoverinfo: 'text',
      xaxis: 'x',
      yaxis: 'y'
    });

    // Historical Best Fit Line
    if (historicalRegression) {
      // Calculate smooth smooth plot across the domain
      const xMin = Math.min(...x);
      const xMax = Math.max(...x);
      const smoothX = [];
      const smoothY = [];
      if (isFinite(xMin) && isFinite(xMax) && xMax > xMin) {
        const step = (xMax - xMin) / 100;
        for (let i = xMin; i <= xMax; i += step) {
          smoothX.push(i);
          smoothY.push(historicalRegression.predict(i));
        }
      }

      traces.push({
        x: smoothX, y: smoothY,
        mode: 'lines', type: 'scatter', name: `Historical Trend (${historicalRegression.type})`,
        line: { color: '#ef4444', width: 2 }, hoverinfo: 'skip'
      });
    }

    // AI Best Fit Line
    if (aiRegression) {
      const rx = aiResults.predictionsMap.map(d => d.rVol);
      const xMin = Math.min(...rx);
      const xMax = Math.max(...rx);
      const aiSmoothX = [];
      const aiSmoothY = [];
      if (isFinite(xMin) && isFinite(xMax) && xMax > xMin) {
        const step = (xMax - xMin) / 100;
        for (let i = xMin; i <= xMax; i += step) {
          aiSmoothX.push(i);
          aiSmoothY.push(aiRegression.predict(i));
        }
      }

      traces.push({
        x: aiSmoothX, y: aiSmoothY,
        mode: 'lines', type: 'scatter', name: `AI Trend (${aiRegression.type})`,
        line: { color: '#a855f7', width: 2, dash: 'dash' }, hoverinfo: 'skip'
      });
    }

    // LSTM AI Predictor Trace
    if (aiResults && aiResults.predictionsMap) {
      traces.push({
        x: aiResults.predictionsMap.map(d => d.rVol),
        y: aiResults.predictionsMap.map(d => d.lstmPredictedExc),
        mode: 'markers',
        type: 'scatter',
        name: 'AI Prediction',
        marker: { color: '#9333ea', symbol: 'star-diamond', size: 10, opacity: 0.9, line: { color: '#ffffff', width: 1 } },
        text: aiResults.predictionsMap.map(d => `<b>LSTM Predicted</b><br>Historical RVol: ${d.rVol.toFixed(2)}x<br>Predicted Exc: ${d.lstmPredictedExc.toFixed(2)}x ADR`),
        hoverinfo: 'text',
        xaxis: 'x',
        yaxis: 'y'
      });
    }

    // Manual AI Input Trace Marker
    if (manualAiResult !== null) {
      traces.push({
        x: [parseFloat(manualAiInput)],
        y: [manualAiResult],
        mode: 'markers',
        type: 'scatter',
        name: 'Your Manual Input',
        marker: { color: '#f59e0b', symbol: 'star', size: 16, line: { color: '#ffffff', width: 2 } },
        text: [`<b>Manual Input Prediction</b><br>RVol Input: ${manualAiInput}x<br>Predicted Exc: ${manualAiResult.toFixed(2)}x`],
        hoverinfo: 'text',
        xaxis: 'x',
        yaxis: 'y'
      });
    }

    // Top Marginal: RVol Bell Curve
    if (rvolDistributionData && rvolDistributionData.length > 0) {
      traces.push({
        x: rvolDistributionData.map(d => d.rVolBin),
        y: rvolDistributionData.map(d => d.bellCurve),
        mode: 'lines',
        type: 'scatter',
        name: 'RVol Distribution',
        line: { color: '#8b5cf6', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(139, 92, 246, 0.2)',
        xaxis: 'x',
        yaxis: 'y2',
        hoverinfo: 'skip'
      });
    }

    // Right Marginal: Excursion Bell Curve
    if (excDistributionData && excDistributionData.length > 0) {
      traces.push({
        y: excDistributionData.map(d => d.excBin),
        x: excDistributionData.map(d => d.bellCurve),
        mode: 'lines',
        type: 'scatter',
        name: 'Excursion Distribution',
        line: { color: '#ef4444', width: 2 },
        fill: 'tozerox',
        fillcolor: 'rgba(239, 68, 68, 0.2)',
        xaxis: 'x2',
        yaxis: 'y',
        hoverinfo: 'skip'
      });
    }

    const layout = {
      autosize: true,
      showlegend: false,
      margin: { l: 60, r: 20, t: 20, b: 60 },
      xaxis: {
        domain: [0, 0.85],
        showgrid: true,
        gridcolor: '#e2e8f0',
        showline: true,
        linewidth: 2,
        linecolor: '#94a3b8',
        mirror: true,
        title: {
          text: 'Relative Volume (RVol) x',
          font: { size: 13, color: '#475569' }
        },
        zeroline: true,
        zerolinecolor: '#cbd5e1',
        tickfont: { size: 12, color: '#64748b' },
        dtick: 0.5,
        rangemode: 'nonnegative'
      },
      yaxis: {
        domain: [0, 0.85],
        range: [0, Math.max(2.0, Math.max(...y) * 1.05)],
        showgrid: true,
        gridcolor: '#e2e8f0',
        showline: true,
        linewidth: 2,
        linecolor: '#94a3b8',
        mirror: true,
        title: {
          text: 'Abs Max Excursion / ADR x',
          font: { size: 13, color: '#475569' }
        },
        zeroline: true,
        zerolinecolor: '#cbd5e1',
        tickfont: { size: 12, color: '#64748b' },
        dtick: 0.5,
        rangemode: 'nonnegative'
      },
      xaxis2: {
        domain: [0.85, 1],
        showline: false,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        rangemode: 'nonnegative'
      },
      yaxis2: {
        domain: [0.85, 1],
        showline: false,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        rangemode: 'nonnegative'
      },
      shapes: [
        {
          type: 'rect',
          xref: 'x',
          yref: 'y',
          x0: 1.5, // Profitable Zone RVol threshold
          y0: 1.5, // Profitable Zone Excursion threshold
          x1: Math.max(1.5, Math.max(...x) * 1.05), // Dynamic upper bound based precisely on actual data points
          y1: Math.max(1.5, Math.max(...y) * 1.05),
          fillcolor: 'rgba(34, 197, 94, 0.1)', // Light green shade
          line: {
            width: 1.5,
            color: 'rgba(34, 197, 94, 0.6)',
            dash: 'dot'
          },
          layer: 'below'
        }
      ],
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      hovermode: 'closest'
    };

    return { traces, layout };
  }, [chartData, historicalRegression, aiRegression, rvolDistributionData, excDistributionData, rvolStats, excStats, manualAiResult, manualAiInput, aiResults]);

  // Selected Ticker Data for Candlestick Chart
  const apexChartState = useMemo(() => {
    if (selectedTickerFilter === 'ALL' || !rawMarketData[selectedTickerFilter]) return null;

    // sorted chronological for apexcharts
    const raw = [...rawMarketData[selectedTickerFilter]].sort((a, b) => new Date(a.date) - new Date(b.date));

    // EMA helper
    const calcEma = (period) => {
      const k = 2 / (period + 1);
      let emaArr = [];
      let currentEma = raw[0]?.close || 0;
      for (let i = 0; i < raw.length; i++) {
        const hClose = raw[i].close || 0;
        if (i !== 0) {
          currentEma = (hClose * k) + (currentEma * (1 - k));
        }
        const time = typeof raw[i].date === 'string' ? raw[i].date.split('T')[0] : raw[i].date.toISOString().split('T')[0];
        emaArr.push({ x: time, y: parseFloat(currentEma.toFixed(2)), timestamp: new Date(time).getTime() });
      }
      return emaArr;
    };

    const ema10Full = calcEma(10);
    const ema20Full = calcEma(20);
    const ema50Full = calcEma(50);

    const maxDate = new Date(raw[raw.length - 1]?.date || Date.now());
    const minDate = new Date(maxDate);
    minDate.setMonth(minDate.getMonth() - 6);
    const minTime = minDate.getTime();

    const candleData = [];
    const volumeData = [];
    raw.forEach((d, i) => {
      const time = typeof d.date === 'string' ? d.date.split('T')[0] : d.date.toISOString().split('T')[0];
      const timeMs = new Date(time).getTime();

      if (timeMs >= minTime) {
        const o = d.open || 0;
        const h = d.high || 0;
        const l = d.low || 0;
        const c = d.close || 0;

        let color = '#cbd5e1'; // default slate-300
        if (c > o) color = '#22c55e'; // green
        else if (c < o) color = '#ef4444'; // red
        else if (i > 0) {
          const prevC = raw[i - 1].close || 0;
          color = c >= prevC ? '#22c55e' : '#ef4444';
        }

        candleData.push({
          x: time,
          y: [
            parseFloat(o.toFixed(2)),
            parseFloat(h.toFixed(2)),
            parseFloat(l.toFixed(2)),
            parseFloat(c.toFixed(2))
          ]
        });
        volumeData.push({
          x: time,
          y: d.volume,
          fillColor: color
        });
      }
    });

    const filterEma = (arr) => arr.filter(item => item.timestamp >= minTime).map(item => ({ x: item.x, y: item.y }));

    const priceSeries = [
      { name: 'Candle', type: 'candlestick', data: candleData },
      { name: '10 EMA', type: 'line', data: filterEma(ema10Full) },
      { name: '20 EMA', type: 'line', data: filterEma(ema20Full) },
      { name: '50 EMA', type: 'line', data: filterEma(ema50Full) }
    ];

    const volumeSeries = [
      { name: 'Volume', type: 'bar', data: volumeData }
    ];

    const priceOptions = {
      chart: {
        type: 'line',
        id: 'price-chart',
        group: 'stock-sync',
        toolbar: { show: true },
        background: 'transparent',
        zoom: { autoScaleYaxis: true },
        animations: { enabled: false }
      },
      xaxis: {
        type: 'category',
        labels: { show: false },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tickAmount: 10
      },
      yaxis: {
        decimalsInFloat: 2,
        labels: {
          style: { colors: '#64748b' },
          formatter: (v) => {
            if (v == null || isNaN(v)) return '';
            return `$${Number(v).toFixed(2)}`;
          }
        }
      },
      stroke: { width: [1, 2, 2, 2], curve: 'smooth' },
      colors: ['#000000', '#0ea5e9', '#d946ef', '#f59e0b'],
      plotOptions: {
        candlestick: {
          colors: { upward: '#22c55e', downward: '#ef4444' }
        },
      },
      legend: { position: 'top', labels: { colors: '#64748b' } },
      tooltip: { shared: false },
      dataLabels: { enabled: false }
    };

    const volumeOptions = {
      chart: {
        type: 'bar',
        id: 'volume-chart',
        group: 'stock-sync',
        toolbar: { show: false },
        background: 'transparent',
        brush: { enabled: false },
        animations: { enabled: false }
      },
      xaxis: {
        type: 'category',
        labels: { style: { colors: '#64748b' } },
        tickAmount: 10
      },
      yaxis: {
        decimalsInFloat: 0,
        labels: {
          style: { colors: '#64748b' },
          formatter: (v) => {
            if (v == null) return '';
            const num = Number(v);
            if (isNaN(num)) return String(v);
            return num >= 1000000 ? `${(num / 1000000).toFixed(1)}M` : (num >= 1000 ? `${(num / 1000).toFixed(1)}K` : num.toFixed(0));
          }
        }
      },
      legend: { show: false },
      tooltip: { shared: false },
      dataLabels: { enabled: false }
    };

    return { priceOptions, priceSeries, volumeOptions, volumeSeries };
  }, [selectedTickerFilter, rawMarketData]);

  // Custom Tooltip for Chart
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      // Find the scatter payload (it contains all custom data)
      const scatterPayload = payload.find(p => p.dataKey === 'maxExcursionAdr')?.payload;
      if (!scatterPayload) return null;

      return (
        <div className="bg-white p-4 border border-slate-200 shadow-lg rounded-md text-sm">
          <p className="font-bold text-slate-800 border-b pb-2 mb-2">{scatterPayload.ticker} - {scatterPayload.dateStr}</p>
          <p className="text-slate-600">RVol: <span className="font-semibold text-slate-900">{scatterPayload.rVol.toFixed(2)}x</span></p>
          <p className="text-slate-600">Abs Max Excursion: <span className="font-semibold text-slate-900">{scatterPayload.maxExcursionAdr.toFixed(2)}x ADR</span></p>
          <p className="text-slate-500 text-xs mt-2">Abs Max Excursion from Open: ${(scatterPayload.maxAbsoluteExcursion).toFixed(2)}</p>
          <p className="text-slate-500 text-xs">{scatterPayload.adrPeriod}-Day ADR: ${(scatterPayload.adrDollar).toFixed(2)}</p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-50">
        <div className="text-lg font-semibold text-slate-600 animate-pulse text-center">
          <p>Processing Market Data...</p>
          <p className="text-sm font-normal text-slate-400 mt-2">Fetching history and calculating statistical relationships.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8 font-sans text-slate-800">
      <div className="max-w-6xl mx-auto space-y-6">

        {/* Header & Tabs */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Stock Stats & ML Predictors</h1>
            <p className="text-sm text-slate-500 mt-1">
              Data-driven edge leveraging regressions and neural networks.
            </p>
          </div>

          <div className="flex border-b border-slate-200 gap-2">
            <button
              onClick={() => setActiveTab('daily')}
              className={`px-4 py-2 font-semibold text-sm ${activeTab === 'daily' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Daily RVol vs Excursion
            </button>
            <button
              onClick={() => setActiveTab('intraday')}
              className={`px-4 py-2 font-semibold text-sm ${activeTab === 'intraday' ? 'border-b-2 border-purple-600 text-purple-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Intraday Volume Predictor
            </button>
            <button
              onClick={() => setActiveTab('maxExcursion')}
              className={`px-4 py-2 font-semibold text-sm ${activeTab === 'maxExcursion' ? 'border-b-2 border-emerald-600 text-emerald-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Max Excursion Predictor
            </button>
            <button
              onClick={() => setActiveTab('quantBacktest')}
              className={`px-4 py-2 font-semibold text-sm ${activeTab === 'quantBacktest' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Quant Backtester
            </button>
            <button
              onClick={() => setActiveTab('backtest')}
              className={`px-4 py-2 font-semibold text-sm ${activeTab === 'backtest' ? 'border-b-2 border-orange-600 text-orange-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Day Trade Simulation
            </button>
            <button
              onClick={() => setActiveTab('regime')}
              className={`px-4 py-2 font-semibold text-sm ${activeTab === 'regime' ? 'border-b-2 border-orange-600 text-orange-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Market Regime
            </button>
          </div>
        </div>

        {activeTab === 'quantBacktest' && (
          <QuantBacktestTab availableTickers={availableTickers} rawMarketData={rawMarketData} />
        )}

        {activeTab === 'backtest' && (
          <BacktestSimulationTab />
        )}

        {/* Existing Content wrapped in activeTab === 'daily' */}
        {activeTab === 'daily' && (
          <div className="space-y-6">
            {/* Header & Controls */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-6">
              <div>
                <h2 className="text-xl font-bold text-slate-800">Historical RVol Excursion Dashboard</h2>
                <p className="text-sm text-slate-500 mt-1">
                  Analyzing `Max Abs(High-Open, Low-Open) / ADR$` mathematically regressed against RVol.
                </p>
              </div>

              <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 bg-slate-100 p-4 rounded-lg">
                <form onSubmit={handleAddCustomTicker} className="flex flex-col sm:flex-row items-center gap-3 w-full lg:w-auto">
                  <label className="text-sm font-semibold text-slate-700 whitespace-nowrap">Analyze Ticker:</label>
                  <div className="flex w-full sm:w-auto">
                    <input
                      type="text"
                      list="ticker-list"
                      value={searchInput}
                      onChange={e => setSearchInput(e.target.value)}
                      placeholder="e.g. NVDA"
                      className="bg-white border border-slate-300 text-slate-900 text-sm rounded-l-md focus:ring-blue-500 focus:border-blue-500 block w-full p-2 outline-none uppercase"
                    />
                    <datalist id="ticker-list">
                      <option value="ALL">ALL TICKERS</option>
                      {availableTickers.map(t => <option key={t} value={t} />)}
                    </datalist>
                    <button
                      type="submit"
                      className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-r-md transition-colors text-sm whitespace-nowrap"
                    >
                      Add Data
                    </button>
                  </div>
                  <select
                    value={selectedTickerFilter}
                    onChange={async (e) => {
                      const val = e.target.value;
                      // If switching to a ticker (or 'ALL') that doesn't have data fully loaded yet, we should fetch it.
                      // (Primarily for 'ALL' or other untouched tickers from INITIAL_TICKERS)
                      if (val === 'ALL') {
                        setLoading(true);
                        const newRawData = { ...rawMarketData };
                        for (const ticker of availableTickers) {
                          if (!newRawData[ticker]) {
                            newRawData[ticker] = await fetchRawTickerData(ticker);
                          }
                        }
                        setRawMarketData(newRawData);
                        setSelectedTickerFilter(val);
                        setLoading(false);
                      } else {
                        if (!rawMarketData[val]) {
                          setLoading(true);
                          const data = await fetchRawTickerData(val);
                          setRawMarketData(prev => ({ ...prev, [val]: data }));
                          setLoading(false);
                        }
                        setSelectedTickerFilter(val);
                      }
                    }}
                    className="bg-white border border-slate-300 text-slate-900 text-sm rounded-md focus:ring-blue-500 focus:border-blue-500 p-2 cursor-pointer outline-none w-full sm:w-auto"
                  >
                    <option value="ALL">View All Data</option>
                    {availableTickers.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </form>

                <div className="flex flex-col sm:flex-row items-center gap-4 w-full lg:w-auto border-t lg:border-t-0 lg:border-l border-slate-300 pt-4 lg:pt-0 lg:pl-4">
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-semibold text-slate-700 whitespace-nowrap">RVol Period:</label>
                    <select
                      value={rvolPeriod}
                      onChange={e => setRvolPeriod(Number(e.target.value))}
                      className="bg-white border border-slate-300 text-slate-900 text-sm rounded-md p-2 outline-none cursor-pointer"
                    >
                      <option value={10}>10 Days</option>
                      <option value={20}>20 Days</option>
                      <option value={50}>50 Days</option>
                    </select>
                  </div>

                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-semibold text-slate-700 whitespace-nowrap">ADR Period:</label>
                    <select
                      value={adrPeriod}
                      onChange={e => setAdrPeriod(Number(e.target.value))}
                      className="bg-white border border-slate-300 text-slate-900 text-sm rounded-md p-2 outline-none cursor-pointer"
                    >
                      <option value={10}>10 Days</option>
                      <option value={20}>20 Days</option>
                      <option value={50}>50 Days</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Predictor Controls */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-lg font-bold text-slate-800">Neural Network Predictor (LSTM)</h2>
                  <p className="text-xs text-slate-500">Train an AI model directly in your browser to predict Excursion based on sequence history. Saves locally to accelerate future reloads.</p>
                </div>

                <div className="flex flex-col sm:flex-row gap-2">
                  <button
                    onClick={handleTrainAI}
                    disabled={isTrainingAi || chartData.length < 10}
                    className={`py-2 px-4 rounded-md font-semibold text-white transition-colors text-sm ${isTrainingAi ? 'bg-slate-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700'}`}
                  >
                    {isTrainingAi ? `Training... Epoch ${aiTrainingEpoch}/${aiTotalEpochs}` : 'Train AI'}
                  </button>
                  {aiResults && !isTrainingAi && (
                    <>
                      <button
                        onClick={handleDownloadModel}
                        className="py-2 px-4 rounded-md font-semibold text-purple-700 bg-purple-100 hover:bg-purple-200 transition-colors text-sm border border-purple-300"
                        title="Export Model to PC downloads folder"
                      >
                        Export Model
                      </button>
                      <button
                        onClick={async () => {
                          const res = await saveModelToServer(aiResults.model, aiResults.preparedData, 'daily');
                          if (res.success) alert("Model saved directly to project folder!");
                          else alert("Error saving: " + res.error);
                        }}
                        className="py-2 px-4 rounded-md font-semibold text-white bg-indigo-600 hover:bg-indigo-700 transition-colors text-sm border border-indigo-700"
                        title="Save directly to Node.js backend"
                      >
                        Save Model to App
                      </button>
                    </>
                  )}

                  <div className="relative">
                    <label className="py-2 px-4 rounded-md font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 transition-colors text-sm border border-slate-300 cursor-pointer block text-center" title="Import previously downloaded model files">
                      Import
                      <input type="file" multiple accept=".json,.bin" className="hidden" onChange={handleUploadModel} />
                    </label>
                  </div>
                </div>
              </div>

              {isTrainingAi && (
                <div className="w-full bg-slate-200 rounded-full h-2 mb-1">
                  <div className="bg-purple-600 h-2 rounded-full transition-all duration-300" style={{ width: `${(aiTrainingEpoch / aiTotalEpochs) * 100}%` }}></div>
                  <p className="text-xs text-slate-500 mt-2 text-right">Current Loss: {aiTrainingLoss.toFixed(4)}</p>
                </div>
              )}

              {aiResults && !isTrainingAi && (
                <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg flex flex-col gap-4">
                  <div className="flex flex-col xl:flex-row justify-between xl:items-center gap-4 border-b border-purple-200/50 pb-4">
                    <div>
                      <h3 className="font-bold text-purple-900">Training Complete</h3>
                      <p className="text-xs text-purple-700">LSTM Final Loss: {aiResults.finalLoss.toFixed(4)}</p>
                      <p className="text-xs text-purple-600 mt-1 font-medium">The AI predictions are now plotted as Purple Diamonds on the chart below.</p>
                    </div>
                    <div className="flex gap-4">
                      <div className="bg-white p-3 rounded shadow-sm flex flex-col items-center min-w-[120px]">
                        <span className="text-xs text-slate-500 font-bold uppercase text-center">Historical Fit (R²)</span>
                        <span className="text-lg font-mono text-slate-800">{historicalRegression ? historicalRegression.r2.toFixed(4) : 'N/A'}</span>
                      </div>
                      <div className="bg-white p-3 rounded shadow-sm flex flex-col items-center min-w-[120px] border-b-2 border-purple-500 relative flex-shrink-0">
                        <span className="text-xs text-slate-500 font-bold uppercase text-center">AI Fit (R²)</span>
                        <span className="text-lg font-mono text-purple-700 font-semibold">{aiRegression ? aiRegression.r2.toFixed(4) : 'N/A'}</span>
                        {historicalRegression && aiRegression && aiRegression.r2 > historicalRegression.r2 && (
                          <div className="absolute -top-2 -right-2 bg-green-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow">WINS</div>
                        )}
                      </div>
                      <div className="bg-white p-3 rounded shadow-sm flex flex-col items-center min-w-[120px]">
                        <span className="text-xs text-slate-500 font-bold uppercase text-center">Mean Abs Error</span>
                        <span className="text-lg font-mono text-emerald-600 font-semibold">{aiResults && aiResults.mae !== undefined ? aiResults.mae.toFixed(4) : 'N/A'}</span>
                      </div>
                    </div>
                  </div>

                  {/* Manual Inference Predictor Panel */}
                  <div className="pt-2 flex flex-col sm:flex-row items-start sm:items-center gap-4">
                    <div className="flex-1">
                      <p className="text-sm font-bold text-purple-900">Manual AI Prediction</p>
                      <p className="text-xs text-purple-700">Ask the trained Neural Network directly. Input a hypothetical RVol scenario for today to see its prediction.</p>
                    </div>
                    <div className="flex items-center shadow-sm rounded-md overflow-hidden border border-purple-300 w-full sm:w-auto">
                      <div className="bg-purple-100 text-purple-800 px-3 py-2 text-xs font-bold border-r border-purple-200 uppercase tracking-wide flex-shrink-0">
                        Input RVol
                      </div>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        value={manualAiInput}
                        onChange={e => setManualAiInput(e.target.value)}
                        className="w-24 px-3 py-2 outline-none font-bold text-slate-800 flex-shrink-0"
                      />
                      <button
                        onClick={handleManualPrediction}
                        className="bg-purple-600 hover:bg-purple-700 text-white transition-colors px-4 py-2 text-sm font-bold flex-shrink-0 border-l border-purple-700"
                      >
                        Predict
                      </button>
                    </div>
                    {manualAiResult !== null && (
                      <div className="bg-white border-2 border-amber-400 p-2 rounded-md shadow-sm ml-auto text-center flex-shrink-0 min-w-[120px]">
                        <span className="block text-[10px] uppercase font-bold text-slate-500">LSTM Output</span>
                        <span className="block text-lg font-black text-slate-800">{manualAiResult.toFixed(2)}x <span className="text-xs text-slate-500 font-normal">ADR</span></span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Main Visualization */}
            {marginalPlotState && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <div className="mb-4 flex flex-col sm:flex-row sm:justify-between sm:items-end">
                  <div>
                    <h2 className="text-lg font-bold text-slate-800">Scatter Plot with Probability Distributions</h2>
                    <p className="text-xs text-slate-500">Y-Axis: Absolute Max Excursion Multiple. X-Axis: Relative Volume.</p>
                  </div>
                  {historicalRegression && (
                    <div className="mt-2 sm:mt-0 bg-blue-50 border border-blue-200 text-blue-800 text-xs px-3 py-1 rounded-full font-semibold">
                      <span className="mr-2">Historical Fit ({historicalRegression.type}):</span>
                      <span className="font-mono">{historicalRegression.equation}</span>
                    </div>
                  )}
                  {aiRegression && (
                    <div className="mt-2 sm:mt-0 bg-purple-50 border border-purple-200 text-purple-800 text-xs px-3 py-1 rounded-full font-semibold ml-2">
                      <span className="mr-2">AI Fit ({aiRegression.type}):</span>
                      <span className="font-mono">{aiRegression.equation}</span>
                    </div>
                  )}
                </div>

                <div className="h-[600px] w-full">
                  <Plot
                    data={marginalPlotState.traces}
                    layout={marginalPlotState.layout}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                    config={{ displayModeBar: false }}
                  />
                </div>
              </div>
            )}


            {/* Advanced Statistical Analysis: High Excursion Probabilities */}
            {advancedStats.length > 0 && (
              <div className="bg-white border border-slate-200 rounded-xl shadow-sm mt-6 overflow-hidden">
                <div className="bg-slate-50 border-b border-slate-200 p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                  <div>
                    <h3 className="text-md font-bold text-slate-800">Advanced High-Excursion Probabilities</h3>
                    <p className="text-xs text-slate-500 mt-1">Historically analyzing the percentage of days that push a <strong>High Excursion (&ge; 1.5 ADR)</strong> once an RVol threshold is breached.</p>
                  </div>
                  <div className="bg-white border border-green-200 rounded-lg shadow-sm p-3 flex items-center space-x-3 whitespace-nowrap">
                    <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0">
                      <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Total Data in Green Zone</div>
                      <div className="text-xl font-bold text-green-700">
                        {advancedStats.find(s => s.threshold === '1.5')?.overallFrequency || '0.00'}%
                      </div>
                    </div>
                  </div>
                </div>
                <div className="p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {advancedStats.map((stat, idx) => (
                    <div key={idx} className="bg-slate-50 rounded-lg p-4 border border-slate-100 flex flex-col justify-between">
                      <div>
                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">RVol threshold</div>
                        <div className="text-lg font-bold text-slate-800 bg-white border border-slate-200 inline-block px-2 py-1 rounded shadow-sm">
                          &ge; {stat.threshold}x
                        </div>
                      </div>

                      <div className="mt-4">
                        <div className="flex justify-between items-end mb-1">
                          <span className="text-sm font-medium text-slate-700">Green Zone Rate:</span>
                          <span className="text-xl font-bold text-green-600">{stat.probability}%</span>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-1.5 mb-3">
                          <div className="bg-green-500 h-1.5 rounded-full" style={{ width: `${stat.probability}%` }}></div>
                        </div>

                        <div className="flex justify-between text-xs text-slate-500 mb-1">
                          <span>Total Signal Days:</span>
                          <span className="font-semibold text-slate-700">{stat.totalMatchingDays}</span>
                        </div>
                        <div className="flex justify-between text-xs text-slate-500 mb-1">
                          <span>High Excursion Hits:</span>
                          <span className="font-semibold text-green-700">{stat.highExcursionDays}</span>
                        </div>
                        <div className="flex justify-between text-xs text-slate-500 mb-1">
                          <span>Avg. Excursion for Group:</span>
                          <span className="font-semibold text-blue-600">{stat.avgExcursion}x</span>
                        </div>
                        <div className="flex justify-between text-xs text-slate-500 pt-2 mt-2 border-t border-slate-200">
                          <span>Overall Frequency:</span>
                          <span className="font-semibold">{stat.overallFrequency}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Custom RVol Input Block */}
                <div className="bg-blue-50/50 border-t border-slate-200 p-4 sm:p-6">
                  <div className="flex flex-col lg:flex-row gap-6 items-start lg:items-center justify-between">
                    <div className="flex-1">
                      <h4 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                        <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path></svg>
                        Custom RVol Green Zone Calculator
                      </h4>
                      <p className="text-xs text-slate-600 mt-1">
                        Enter a specific minimum Relative Volume (RVol) across your datasets to calculate its historical probability of reaching a <span className="font-semibold text-green-700">1.5x+ ADR</span> Excursion.
                      </p>
                    </div>

                    <div className="flex items-stretch bg-white border border-blue-200 shadow-sm rounded-lg overflow-hidden w-full lg:w-auto">
                      <div className="px-4 py-3 bg-slate-50 border-r border-blue-100 flex items-center justify-center">
                        <span className="text-xs font-bold text-slate-500 tracking-wider">RVOL &ge;</span>
                      </div>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        placeholder="e.g. 1.5"
                        className="w-24 px-3 py-2 outline-none text-slate-800 font-bold focus:bg-blue-50 transition-colors"
                        value={customRvolThreshold}
                        onChange={(e) => setCustomRvolThreshold(e.target.value)}
                      />
                      <div className="flex-1 px-4 py-3 bg-blue-600 text-white flex items-center justify-between gap-4 min-w-[140px]">
                        <div className="flex flex-col">
                          <span className="text-[10px] uppercase font-semibold text-blue-200 tracking-wider leading-none mb-1">Green Zone Rate</span>
                          <span className="text-2xl font-bold leading-none">{customStat ? customStat.probability : '0.00'}%</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {customStat && !isNaN(parseFloat(customRvolThreshold)) && (
                    <div className="mt-4 pt-4 border-t border-blue-200/50 flex flex-wrap gap-x-6 gap-y-2 text-xs">
                      <div className="flex flex-col">
                        <span className="text-slate-500">Total Signal Days</span>
                        <span className="font-semibold text-slate-800">{customStat.totalMatchingDays}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-slate-500">High Excursion Hits</span>
                        <span className="font-semibold text-green-700">{customStat.highExcursionDays}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-slate-500">Avg. Group Excursion</span>
                        <span className="font-semibold text-blue-700">{customStat.avgExcursion}x</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-slate-500">Overall Frequency</span>
                        <span className="font-semibold text-slate-800">{customStat.overallFrequency}%</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}        {/* Statistical Summary Table Moved Below Chart */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mt-6">
              <div className="p-6 border-b border-slate-200">
                <h2 className="text-lg font-bold text-slate-800">Intraday Excursion by RVol Bucket</h2>
                <p className="text-xs text-slate-500">Grouped analysis showing the mathematical expansion from the day's open to the high or low.</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-slate-600">
                  <thead className="text-xs text-slate-700 uppercase bg-slate-50 border-b border-slate-200">
                    <tr>
                      <th scope="col" className="px-6 py-4 font-bold">RVol Bucket</th>
                      <th scope="col" className="px-6 py-4 text-center">Sample Size</th>
                      <th scope="col" className="px-6 py-4 text-center">RVol Prob.</th>
                      <th scope="col" className="px-6 py-4 text-right whitespace-nowrap">Median Max Exc.</th>
                      <th scope="col" className="px-6 py-4 text-right whitespace-nowrap">&ge; Median Exc Prob.</th>
                      <th scope="col" className="px-6 py-4 text-right whitespace-nowrap">Mean Max Exc.</th>
                      <th scope="col" className="px-6 py-4 text-right font-bold text-slate-800 whitespace-nowrap">Absolute Max Exc.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summaryStats.map((row, idx) => (
                      <tr key={idx} className="bg-white border-b hover:bg-slate-50 transition-colors">
                        <td className="px-6 py-4 font-semibold text-slate-900 whitespace-nowrap">
                          {row.label}
                        </td>
                        <td className="px-6 py-4 text-center">
                          <span className="bg-slate-100 text-slate-700 py-1 px-3 rounded-full text-xs font-medium">
                            {row.sampleSize}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center text-slate-500 font-medium">
                          {row.sampleSize > 0 ? `${row.rvolProbability}%` : '-'}
                        </td>
                        <td className="px-6 py-4 text-right font-medium">{row.medianExcursion ? `${row.medianExcursion}x` : '-'}</td>
                        <td className="px-6 py-4 text-right text-slate-500 font-medium whitespace-nowrap" title={`Probability of any day having an excursion of >= ${row.medianExcursion}x`}>
                          {row.medianExcursionProb && row.sampleSize > 0 ? `${row.medianExcursionProb}%` : '-'}
                        </td>
                        <td className="px-6 py-4 text-right">{row.meanExcursion ? `${row.meanExcursion}x` : '-'}</td>
                        <td className="px-6 py-4 text-right text-red-600 font-bold">{row.maxExcursion ? `${row.maxExcursion}x` : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Selected Ticker Stock Chart */}
            {selectedTickerFilter !== 'ALL' && apexChartState && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mt-6">
                <div className="mb-4">
                  <h2 className="text-lg font-bold text-slate-800">{selectedTickerFilter} Historical Chart (Last 6 Months)</h2>
                  <p className="text-xs text-slate-500">
                    Daily Candlesticks, Moving Averages (10, 20, 50 EMA), and Volume
                  </p>
                </div>
                <div className="w-full flex flex-col">
                  <ReactApexChart
                    options={apexChartState.priceOptions}
                    series={apexChartState.priceSeries}
                    type="line"
                    height={600}
                  />
                  <ReactApexChart
                    options={apexChartState.volumeOptions}
                    series={apexChartState.volumeSeries}
                    type="bar"
                    height={160}
                  />
                </div>
              </div>
            )}


          </div>
        )}

        {/* Intraday Tab Content */}
        {activeTab === 'intraday' && (
          <div className="flex flex-col gap-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-6">
              <div>
                <h2 className="text-xl font-bold text-slate-800">Intraday LSTM Network</h2>
                <p className="text-sm text-slate-500">Train an AI to predict full-day RVol based on the first few minutes of volume after market open.</p>
              </div>

              <div className="flex flex-col md:flex-row gap-4 items-end bg-slate-100 p-4 rounded-lg border border-slate-200">
                <div className="flex flex-col w-full md:w-1/3">
                  <label className="text-sm font-semibold text-slate-700 mb-2">Analysis Scope (Ticker):</label>
                  <select
                    value={selectedIntradayTicker}
                    onChange={e => setSelectedIntradayTicker(e.target.value)}
                    className="bg-white border border-slate-300 text-slate-900 text-sm rounded-md focus:ring-purple-500 focus:border-purple-500 p-2 cursor-pointer outline-none w-full"
                  >
                    <option value="ALL">All Tickers ({intradayFiles.length} files)</option>
                    {intradayFiles.map(f => <option key={f.ticker} value={f.ticker}>{f.ticker} ({f.startDate} to {f.endDate})</option>)}
                  </select>
                  {selectedIntradayTicker !== 'ALL' && intradayFiles.find(f => f.ticker === selectedIntradayTicker) && (
                    <p className="text-xs text-slate-500 mt-2">
                      Data from {intradayFiles.find(f => f.ticker === selectedIntradayTicker).startDate} to {intradayFiles.find(f => f.ticker === selectedIntradayTicker).endDate}
                    </p>
                  )}
                </div>

                <div className="flex flex-col w-full md:w-1/3">
                  <label className="text-sm font-semibold text-slate-700 mb-2 flex justify-between">
                    <span>Minutes after open:</span>
                    <span className="text-purple-600 bg-purple-100 px-2 py-0.5 rounded font-bold">{minutesToUse} mins</span>
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="120"
                    step="5"
                    value={minutesToUse}
                    onChange={e => setMinutesToUse(Number(e.target.value))}
                    className="w-full h-2 bg-slate-300 rounded-lg appearance-none cursor-pointer mt-2"
                  />
                  <p className="text-xs text-slate-500 mt-2 text-center">Using first {Math.floor(minutesToUse / 5)}x 5-min bars</p>
                </div>

                <div className="w-full md:w-auto ml-auto">
                  <button
                    onClick={handleTrainIntraday}
                    disabled={isIntradayTraining || isIntradayLoading}
                    className={`py-2 px-6 rounded-md font-bold text-white transition-colors w-full md:w-auto whitespace-nowrap shadow-sm ${isIntradayTraining || isIntradayLoading ? 'bg-slate-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700'}`}
                  >
                    {isIntradayLoading ? 'Loading Data...' : isIntradayTraining ? `Training... Epoch ${intradayTrainEpoch}/50` : 'Train Prediction Model'}
                  </button>
                  {isIntradayTraining && (
                    <div className="w-full bg-slate-200 rounded-full h-1.5 mt-2">
                      <div className="bg-purple-600 h-1.5 rounded-full transition-all duration-300" style={{ width: `${(intradayTrainEpoch / 50) * 100}%` }}></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Volume Profile visualization */}
            {volumeProfileDisplayData.chart && volumeProfileDisplayData.chart.length > 0 && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-4">
                <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-4">
                  <div>
                    <h3 className="text-lg font-bold text-slate-800">Unified Prediction: U-Shape vs AI Expected Volume</h3>
                    <p className="text-sm text-slate-500">Input exact actual bar volumes for the first {minutesToUse} minutes to see algorithmic divergence.</p>
                  </div>
                  <div className="flex flex-col sm:flex-row items-stretch shadow-sm rounded-md overflow-hidden border border-purple-300 w-full xl:w-auto flex-shrink-0">
                    <div className="bg-purple-100 text-purple-800 px-3 py-2 text-xs font-bold border-r border-purple-200 uppercase tracking-wide flex items-center justify-center whitespace-nowrap">
                      Actual Volumes
                    </div>
                    <input
                      type="text"
                      value={manualIntradayInputStr}
                      onChange={e => setManualIntradayInputStr(e.target.value)}
                      placeholder="e.g. 500000, 200000..."
                      className="w-full xl:w-64 px-3 py-2 outline-none font-medium text-sm text-slate-800 border-b sm:border-b-0 sm:border-r border-purple-200"
                    />
                    <button
                      onClick={handlePredictIntraday}
                      className="bg-purple-600 hover:bg-purple-700 text-white font-bold px-6 py-2 text-sm transition-colors whitespace-nowrap"
                    >
                      AI Predict RVol
                    </button>
                  </div>
                </div>

                {/* Advanced Prediction Comparison Box */}
                {(volumeProfileDisplayData.mathProjectedVol > 0 || manualIntradayResult !== null) && (
                  <div className="flex flex-col md:flex-row gap-4 my-2">
                    {volumeProfileDisplayData.mathProjectedVol > 0 && (
                      <div className="bg-indigo-50 border border-indigo-200 p-4 rounded-lg flex-1">
                        <span className="block text-xs font-bold text-indigo-700 uppercase mb-1">Mathematical U-Shape Projection</span>
                        <div className="text-2xl font-black text-indigo-900 border-b border-indigo-200 pb-1 mb-1">
                          {Math.floor(volumeProfileDisplayData.mathProjectedVol).toLocaleString()} <span className="text-sm font-semibold text-indigo-600">Total Shares</span>
                        </div>
                        <p className="text-xs text-indigo-800">Assumes today obeys identical relative participation proportions to historical market averages.</p>
                      </div>
                    )}
                    {manualIntradayResult !== null && (
                      <div className="bg-green-50 border border-green-200 p-4 rounded-lg flex-1">
                        <span className="block text-xs font-bold text-green-700 uppercase mb-1">Neural Network AI Projection</span>
                        <div className="text-2xl font-black text-green-700 border-b border-green-200 pb-1 mb-1 space-x-2">
                          <span>{Math.floor(manualIntradayResult.predictedVolume).toLocaleString()}</span>
                          <span className="text-sm font-semibold text-green-600">Total Shares</span>
                          {manualIntradayResult.showRvol && (
                            <span className="text-lg text-green-800 font-bold bg-green-200 px-2 py-0.5 rounded-md ml-2">{manualIntradayResult.rvol.toFixed(2)}x RVol</span>
                          )}
                        </div>
                        <p className="text-xs text-green-800">Model recognizes volume velocity sequence and predicts statistically probable momentum exhaustion.</p>
                      </div>
                    )}
                  </div>
                )}

                <div className="h-[250px] w-full mt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={volumeProfileDisplayData.chart} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                      <XAxis
                        dataKey="time"
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        minTickGap={30}
                      >
                        <Label value="Time of Day" offset={-15} position="insideBottom" style={{ fill: '#64748b', fontWeight: 'bold' }} />
                      </XAxis>
                      <YAxis
                        tickFormatter={(val) => (val > 1000000 ? `${(val / 1000000).toFixed(1)}M` : val > 1000 ? `${(val / 1000).toFixed(0)}k` : val)}
                        tick={{ fontSize: 11, fill: '#64748b' }}
                      >
                        <Label value="Projected Volume" angle={-90} position="insideLeft" style={{ fill: '#64748b', fontWeight: 'bold' }} />
                      </YAxis>
                      <RechartsTooltip
                        cursor={{ fill: '#f1f5f9' }}
                        contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                        formatter={(value, name) => {
                          if (name === 'Average Historical Pace') return [Math.round(value).toLocaleString(), name];
                          if (name === 'Projected Trace Overlay') return [Math.round(value).toLocaleString(), name];
                          return [Math.round(value).toLocaleString(), name];
                        }}
                      />
                      <Legend verticalAlign="top" height={36} />
                      <Bar dataKey="scaledAvgVolume" fill="#cbd5e1" radius={[2, 2, 0, 0]} name="Average Historical Pace" />
                      <Line type="monotone" dataKey="predictedVisual" stroke="#a855f7" strokeWidth={3} dot={false} activeDot={{ r: 6 }} name="Projected Trace Overlay" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {intradayAiResults && !isIntradayTraining && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-6">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-100 pb-4">
                  <div>
                    <h3 className="text-lg font-bold text-slate-800">Prediction Model Evaluation</h3>
                    <p className="text-sm text-slate-500">Cross-referencing Expected Volume against Actual End-Of-Day Volume</p>
                    {aiRegressionResult && aiRegressionResult.fit && (
                      <p className="text-xs text-amber-600 font-semibold mt-1">
                        ↳ Best Fit: {aiRegressionResult.fit.type} Regression | R² = {aiRegressionResult.fit.r2.toFixed(3)} | {aiRegressionResult.fit.equation}
                      </p>
                    )}
                  </div>
                  <div className="mt-4 md:mt-0 flex flex-wrap items-center gap-4">
                    <button
                      onClick={async () => {
                        setIntradaySaveStatus("Saving...");
                        try {
                          const res = await saveModelToServer(intradayAiResults.model, intradayAiResults.preparedData, 'intraday');
                          if (res.success) {
                            setIntradaySaveStatus("Saved to Intraday Models!");
                            setTimeout(() => setIntradaySaveStatus(null), 3000);
                          } else {
                            setIntradaySaveStatus("Save failed!");
                          }
                        } catch (err) {
                          setIntradaySaveStatus("Save failed!");
                        }
                      }}
                      disabled={!!intradaySaveStatus}
                      className="py-2 px-4 rounded-md font-bold text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 transition-colors text-sm shadow-sm"
                      title="Save Model to Node.js project folder"
                    >
                      {intradaySaveStatus || "Save Model to App"}
                    </button>
                    <div className="bg-purple-50 p-3 rounded-lg border border-purple-100 text-center min-w-[120px]">
                      <span className="block text-xs text-purple-700 font-bold uppercase mb-1">Final Loss (MSE)</span>
                      <span className="block text-lg font-mono font-bold text-purple-900">{intradayAiResults.finalLoss.toFixed(4)}</span>
                    </div>
                    <div className="bg-indigo-50 p-3 rounded-lg border border-indigo-100 text-center min-w-[120px]">
                      <span className="block text-xs text-indigo-700 font-bold uppercase mb-1">R-Squared (R²)</span>
                      <span className="block text-lg font-mono font-bold text-indigo-900">{intradayAiResults.rSquared !== undefined ? intradayAiResults.rSquared.toFixed(3) : 'N/A'}</span>
                    </div>
                    <div className="bg-emerald-50 p-3 rounded-lg border border-emerald-100 text-center min-w-[120px]">
                      <span className="block text-xs text-emerald-700 font-bold uppercase mb-1">Mean Abs Error</span>
                      <span className="block text-lg font-mono font-bold text-emerald-900">
                        {intradayAiResults.mae !== undefined ?
                          (intradayAiResults.mae >= 1000000 ? `${(intradayAiResults.mae / 1000000).toFixed(2)}M` : `${(intradayAiResults.mae / 1000).toFixed(0)}k`)
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-lg border border-blue-100 text-center min-w-[120px]">
                      <span className="block text-xs text-blue-700 font-bold uppercase mb-1">Standard Dev</span>
                      <span className="block text-lg font-mono font-bold text-blue-900">
                        {intradayAiResults.stdDev !== undefined ?
                          (intradayAiResults.stdDev >= 1000000 ? `±${(intradayAiResults.stdDev / 1000000).toFixed(2)}M` : `±${(intradayAiResults.stdDev / 1000).toFixed(0)}k`)
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center min-w-[120px]">
                      <span className="block text-xs text-slate-500 font-bold uppercase mb-1">Data Points</span>
                      <span className="block text-lg font-mono font-bold text-slate-800">{intradayAiResults.predictionsMap.length}</span>
                    </div>
                  </div>
                </div>

                {/* Scatter Plot for Predicted vs Actual Volume */}
                <div className="h-[500px] w-full mt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={intradayAiResults.predictionsMap} margin={{ top: 20, right: 30, bottom: 20, left: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis
                        dataKey="actualVol"
                        type="number"
                        name="Actual EOD Volume"
                        tickFormatter={(val) => (val >= 1000000 ? `${(val / 1000000).toFixed(1)}M` : `${(val / 1000).toFixed(0)}k`)}
                      >
                        <Label value="Actual End-of-Day Total Shares" offset={-10} position="insideBottom" style={{ fill: '#64748b', fontWeight: 'bold' }} />
                      </XAxis>
                      <YAxis
                        dataKey="predictedVol"
                        type="number"
                        name="AI Predicted EOD Volume"
                        tickFormatter={(val) => (val >= 1000000 ? `${(val / 1000000).toFixed(1)}M` : `${(val / 1000).toFixed(0)}k`)}
                      >
                        <Label value="AI Predicted EOD Shares" angle={-90} position="insideLeft" offset={-15} style={{ fill: '#64748b', fontWeight: 'bold' }} />
                      </YAxis>
                      <RechartsTooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            // Find the scatter point payload instead of the interpolated line point
                            const dotPayload = payload.find(p => p.payload && p.payload.ticker);
                            if (!dotPayload) return null;

                            const d = dotPayload.payload;
                            return (
                              <div className="bg-white p-3 border border-slate-200 shadow-md rounded-lg text-sm min-w-[180px]">
                                <p className="font-bold text-slate-800 mb-2 border-b pb-1">
                                  {d.ticker} | {d.dateStr}
                                </p>
                                <div className="flex justify-between mb-1">
                                  <span className="text-slate-500">Actual Shares:</span>
                                  <span className="font-semibold">{Math.floor(d.actualVol).toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-purple-600 font-medium">AI Predicted:</span>
                                  <span className="font-bold text-purple-700">{Math.floor(d.predictedVol).toLocaleString()}</span>
                                </div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter name="Predictions" fill="#9333ea" opacity={0.6} />

                      {aiRegressionResult && aiRegressionResult.fit && (
                        <Line
                          data={aiRegressionResult.lineData}
                          dataKey="regressionVol"
                          type="basis"
                          stroke="#f59e0b"
                          strokeWidth={3}
                          dot={false}
                          activeDot={false}
                          name={`Best Fit (${aiRegressionResult.fit.type})`}
                        />
                      )}

                      {/* A perfect prediction line y=x */}
                      <Line

                        data={[
                          { actualVol: 0, predictedVol: 0 },
                          {
                            actualVol: intradayAiResults.predictionsMap.length > 0 ? Math.max(...intradayAiResults.predictionsMap.map(d => d.actualVol)) : 0,
                            predictedVol: intradayAiResults.predictionsMap.length > 0 ? Math.max(...intradayAiResults.predictionsMap.map(d => d.actualVol)) : 0
                          }
                        ]}
                        dataKey="predictedVol"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                        activeDot={false}
                        name="Perfect Prediction Line"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Max Excursion Tab Content */}
        {activeTab === 'maxExcursion' && (
          <div className="flex flex-col gap-6">
            {/* Controls */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-6">
              <div>
                <h2 className="text-xl font-bold text-slate-800">Max Excursion Predictor</h2>
                <p className="text-sm text-slate-500">Predict a stock's intraday High & Low from the first minutes of trading data using 11 technical features.</p>
              </div>

              <div className="flex flex-col md:flex-row gap-4 items-end bg-emerald-50 p-4 rounded-lg border border-emerald-200">
                <div className="flex flex-col w-full md:w-1/4">
                  <label className="text-sm font-semibold text-slate-700 mb-2">Ticker Scope:</label>
                  <select
                    value={selectedMaxExcTicker}
                    onChange={e => setSelectedMaxExcTicker(e.target.value)}
                    className="bg-white border border-slate-300 text-slate-900 text-sm rounded-md focus:ring-emerald-500 focus:border-emerald-500 p-2 cursor-pointer outline-none w-full"
                  >
                    <option value="ALL">All Tickers ({intradayFiles.length} files)</option>
                    {intradayFiles.map(f => <option key={f.ticker} value={f.ticker}>{f.ticker}</option>)}
                  </select>
                </div>

                <div className="flex flex-col w-full md:w-1/4">
                  <label className="text-sm font-semibold text-slate-700 mb-2 flex justify-between">
                    <span>Minutes after open:</span>
                    <span className="text-emerald-600 bg-emerald-100 px-2 py-0.5 rounded font-bold">{maxExcMinutes} min</span>
                  </label>
                  <input
                    type="range" min="5" max="30" step="5"
                    value={maxExcMinutes}
                    onChange={e => setMaxExcMinutes(Number(e.target.value))}
                    className="w-full h-2 bg-slate-300 rounded-lg appearance-none cursor-pointer mt-2"
                  />
                </div>

                <div className="flex flex-col w-full md:w-1/4">
                  <label className="text-sm font-semibold text-slate-700 mb-2">Epochs:</label>
                  <input
                    type="number" min="10" max="300" step="10"
                    value={maxExcTotalEpochs}
                    onChange={e => setMaxExcTotalEpochs(Number(e.target.value))}
                    className="bg-white border border-slate-300 text-sm rounded-md p-2 outline-none w-full"
                  />
                </div>

                <div className="w-full md:w-auto ml-auto">
                  <button
                    onClick={handleTrainMaxExcursion}
                    disabled={isMaxExcTraining}
                    className={`py-2 px-6 rounded-md font-bold text-white transition-colors w-full md:w-auto whitespace-nowrap shadow-sm ${isMaxExcTraining ? 'bg-slate-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700'}`}
                  >
                    {isMaxExcTraining ? `Training... ${maxExcTrainEpoch}/${maxExcTotalEpochs}` : 'Train Model'}
                  </button>
                  {isMaxExcTraining && (
                    <div className="w-full bg-slate-200 rounded-full h-1.5 mt-2">
                      <div className="bg-emerald-600 h-1.5 rounded-full transition-all duration-300" style={{ width: `${(maxExcTrainEpoch / maxExcTotalEpochs) * 100}%` }}></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Live Prediction Chart */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-4">
              <div>
                <h3 className="text-lg font-bold text-slate-800">Live Prediction Chart</h3>
                <p className="text-sm text-slate-500">Enter a ticker to see predicted High/Low overlaid on the last 90 days of price data.</p>
              </div>
              <div className="flex flex-col sm:flex-row gap-4 items-end">
                <div className="flex flex-col flex-1">
                  <label className="text-xs font-semibold text-slate-600 mb-1">Ticker Symbol</label>
                  <select
                    value={predTicker}
                    onChange={e => setPredTicker(e.target.value)}
                    className="border border-slate-300 bg-white rounded-md p-2 text-sm outline-none focus:ring-emerald-500 focus:border-emerald-500 w-full"
                  >
                    <option value="" disabled>Select a ticker</option>
                    {intradayFiles.map(f => (
                      <option key={f.ticker} value={f.ticker}>{f.ticker}</option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={handleChartPrediction}
                  disabled={isPredicting || !predTicker.trim()}
                  className={`px-6 py-2 rounded-md font-bold text-white text-sm transition-colors whitespace-nowrap shadow-sm ${isPredicting ? 'bg-slate-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700'}`}
                >
                  {isPredicting ? 'Loading...' : 'Predict High / Low'}
                </button>
              </div>

              {predChartData && (
                <div className="mt-2">
                  <div className="flex items-center gap-4 mb-3 text-xs">
                    <span className="font-bold text-slate-700">{predChartData.ticker} — Last {predChartData.totalDays} days</span>
                    <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-emerald-500"></span> Predicted High ({predChartData.predHighDots.length} days)</span>
                    <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-rose-500"></span> Predicted Low ({predChartData.predLowDots.length} days)</span>
                  </div>
                  <ReactApexChart
                    type="candlestick"
                    height={420}
                    series={[
                      { name: 'Price', type: 'candlestick', data: predChartData.candlestick },
                      { name: 'Predicted High', type: 'line', data: predChartData.predHighDots },
                      { name: 'Predicted Low', type: 'line', data: predChartData.predLowDots }
                    ]}
                    options={{
                      chart: {
                        type: 'candlestick',
                        toolbar: { show: true, tools: { download: true, zoom: true, pan: true, reset: true } },
                        background: '#fff'
                      },
                      title: { text: undefined },
                      xaxis: {
                        type: 'datetime',
                        labels: { datetimeFormatter: { month: "MMM 'yy", day: 'dd MMM' } }
                      },
                      yaxis: {
                        tooltip: { enabled: true },
                        labels: { formatter: v => '$' + v.toFixed(2) }
                      },
                      plotOptions: {
                        candlestick: {
                          colors: { upward: '#22c55e', downward: '#ef4444' },
                          wick: { useFillColor: true }
                        }
                      },
                      stroke: {
                        width: [1, 0, 0]
                      },
                      markers: {
                        size: [0, 7, 7],
                        colors: [undefined, '#10b981', '#f43f5e'],
                        strokeColors: [undefined, '#059669', '#e11d48'],
                        strokeWidth: 2,
                        hover: { sizeOffset: 2 }
                      },
                      legend: {
                        show: true,
                        position: 'top',
                        labels: { colors: '#475569' },
                        markers: { fillColors: ['#64748b', '#10b981', '#f43f5e'] }
                      },
                      tooltip: {
                        shared: false,
                        custom: function ({ seriesIndex, dataPointIndex, w }) {
                          const s = w.config.series[seriesIndex];
                          const point = s.data[dataPointIndex];
                          if (!point) return '';
                          const date = new Date(point.x).toLocaleDateString();
                          if (seriesIndex === 0) {
                            const [o, h, l, c] = point.y;
                            return `<div style="padding:8px;font-size:12px"><b>${date}</b><br/>O: $${o.toFixed(2)} H: $${h.toFixed(2)}<br/>L: $${l.toFixed(2)} C: $${c.toFixed(2)}</div>`;
                          } else {
                            const label = seriesIndex === 1 ? 'Predicted High' : 'Predicted Low';
                            const color = seriesIndex === 1 ? '#10b981' : '#f43f5e';
                            return `<div style="padding:8px;font-size:12px"><b>${date}</b><br/><span style="color:${color};font-weight:bold">${label}: $${point.y.toFixed(2)}</span></div>`;
                          }
                        }
                      },
                      grid: { borderColor: '#e2e8f0', strokeDashArray: 3 }
                    }}
                  />
                </div>
              )}
            </div>

            {/* Training Results */}
            {maxExcResults && !isMaxExcTraining && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-6">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-100 pb-4">
                  <div>
                    <h3 className="text-lg font-bold text-slate-800">Backtesting Results</h3>
                    <p className="text-sm text-slate-500">Predicted vs Actual Max Excursion (in ADR multiples)</p>
                  </div>
                  <div className="mt-4 md:mt-0 flex flex-wrap items-center gap-4">
                    <button
                      onClick={async () => {
                        setMaxExcSaveStatus('Saving...');
                        try {
                          // Only save essential metadata, not the full training arrays
                          const metadataToSave = {
                            normParams: maxExcResults.preparedData.normParams,
                            featureNames: maxExcResults.preparedData.featureNames,
                            allFeatureNames: maxExcResults.preparedData.allFeatureNames,
                            featureMask: maxExcResults.preparedData.featureMask,
                            activeIndices: maxExcResults.preparedData.activeIndices,
                            rSquared: maxExcResults.rSquared,
                            mse: maxExcResults.mse,
                            stdDev: maxExcResults.stdDev,
                            dataPoints: maxExcResults.predictionsMap.length,
                            trainedAt: new Date().toISOString()
                          };
                          const res = await saveModelToServer(maxExcResults.model, metadataToSave, 'maxExcursion');
                          if (res.success) {
                            setMaxExcSaveStatus('Saved to max-excursion/');
                            setTimeout(() => setMaxExcSaveStatus(null), 3000);
                          } else {
                            console.error('Save error:', res.error);
                            setMaxExcSaveStatus('Save failed: ' + (res.error || 'unknown'));
                            setTimeout(() => setMaxExcSaveStatus(null), 5000);
                          }
                        } catch (err) {
                          console.error('Save exception:', err);
                          setMaxExcSaveStatus('Save failed: ' + err.message);
                          setTimeout(() => setMaxExcSaveStatus(null), 5000);
                        }
                      }}
                      disabled={!!maxExcSaveStatus}
                      className="py-2 px-4 rounded-md font-bold text-white bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-400 transition-colors text-sm shadow-sm"
                      title="Save to public/models/max-excursion/"
                    >
                      {maxExcSaveStatus || 'Save Model to App'}
                    </button>
                    <div className="bg-emerald-50 p-3 rounded-lg border border-emerald-100 text-center min-w-[120px]">
                      <span className="block text-xs text-emerald-700 font-bold uppercase mb-1">R²</span>
                      <span className="block text-lg font-mono font-bold text-emerald-900">{maxExcResults.rSquared.toFixed(3)}</span>
                    </div>
                    <div className="bg-teal-50 p-3 rounded-lg border border-teal-100 text-center min-w-[120px]">
                      <span className="block text-xs text-teal-700 font-bold uppercase mb-1">MAE</span>
                      <span className="block text-lg font-mono font-bold text-teal-900">{maxExcResults.mae?.toFixed(4) || 'N/A'}</span>
                    </div>
                    <div className="bg-amber-50 p-3 rounded-lg border border-amber-100 text-center min-w-[120px]">
                      <span className="block text-xs text-amber-700 font-bold uppercase mb-1">MSE</span>
                      <span className="block text-lg font-mono font-bold text-amber-900">{maxExcResults.mse.toFixed(4)}</span>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-lg border border-blue-100 text-center min-w-[120px]">
                      <span className="block text-xs text-blue-700 font-bold uppercase mb-1">Std Dev</span>
                      <span className="block text-lg font-mono font-bold text-blue-900">±{maxExcResults.stdDev.toFixed(3)}</span>
                    </div>
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center min-w-[120px]">
                      <span className="block text-xs text-slate-500 font-bold uppercase mb-1">Data Points</span>
                      <span className="block text-lg font-mono font-bold text-slate-800">{maxExcResults.predictionsMap.length}</span>
                    </div>
                  </div>
                </div>

                {/* Scatter Plot */}
                <div className="h-[450px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={maxExcResults.predictionsMap} margin={{ top: 20, right: 30, bottom: 20, left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="actualMaxExc" type="number" name="Actual Max Excursion">
                        <Label value="Actual Max Excursion (xADR)" offset={-10} position="insideBottom" style={{ fill: '#64748b', fontWeight: 'bold' }} />
                      </XAxis>
                      <YAxis dataKey="predictedMaxExc" type="number" name="Predicted">
                        <Label value="Predicted Max Excursion (xADR)" angle={-90} position="insideLeft" offset={-5} style={{ fill: '#64748b', fontWeight: 'bold', fontSize: 12 }} />
                      </YAxis>
                      <RechartsTooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const dotPayload = payload.find(p => p.payload && p.payload.ticker);
                            if (!dotPayload) return null;
                            const d = dotPayload.payload;
                            return (
                              <div className="bg-white p-3 border border-slate-200 shadow-md rounded-lg text-sm min-w-[200px]">
                                <p className="font-bold text-slate-800 mb-2 border-b pb-1">{d.ticker} | {d.date}</p>
                                <div className="flex justify-between mb-1"><span className="text-slate-500">Actual:</span><span className="font-semibold">{d.actualMaxExc.toFixed(2)}x ADR</span></div>
                                <div className="flex justify-between mb-1"><span className="text-emerald-600">Predicted:</span><span className="font-bold text-emerald-700">{d.predictedMaxExc.toFixed(2)}x ADR</span></div>
                                <div className="flex justify-between mb-1"><span className="text-slate-500">Open:</span><span>${d.dayOpen.toFixed(2)}</span></div>
                                <div className="flex justify-between mb-1"><span className="text-green-600">Pred High:</span><span className="font-semibold">${d.predictedHigh.toFixed(2)}</span></div>
                                <div className="flex justify-between"><span className="text-red-600">Pred Low:</span><span className="font-semibold">${d.predictedLow.toFixed(2)}</span></div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter name="Predictions" fill="#10b981" opacity={0.6} />
                      <Line
                        data={[
                          { actualMaxExc: 0, predictedMaxExc: 0 },
                          {
                            actualMaxExc: maxExcResults.predictionsMap.length > 0 ? Math.max(...maxExcResults.predictionsMap.map(d => d.actualMaxExc)) : 3,
                            predictedMaxExc: maxExcResults.predictionsMap.length > 0 ? Math.max(...maxExcResults.predictionsMap.map(d => d.actualMaxExc)) : 3
                          }
                        ]}
                        dataKey="predictedMaxExc"
                        stroke="#10b981"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                        activeDot={false}
                        name="Perfect Prediction"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                {/* Model Features Used */}
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mt-2 mb-4">
                  <h4 className="text-sm font-bold text-slate-700 mb-2">Features Extracted Live Before Prediction:</h4>
                  <div className="flex flex-wrap gap-2">
                    {maxExcResults.preparedData.featureNames.map((feat, i) => (
                      <span key={i} className="px-2 py-1 bg-white border border-slate-300 shadow-sm rounded text-xs text-slate-700 font-medium">
                        {feat}
                      </span>
                    ))}
                  </div>
                  <p className="text-xs text-slate-500 mt-3 italic text-center">
                    Note: The above chart displays STRICTLY out-of-sample backtesting results. The model was trained on the chronological first 80% of dates, and tested exclusively on the remaining unseen 20% to prevent overfitting or look-ahead bias.
                  </p>
                </div>

                {/* Per-Ticker Breakdown */}
                {maxExcResults.perTickerStats && maxExcResults.perTickerStats.length > 1 && (
                  <div>
                    <h4 className="text-md font-bold text-slate-700 mb-3">Per-Ticker Accuracy (MAE, sorted best → worst)</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left text-slate-600">
                        <thead className="text-xs text-slate-500 uppercase bg-slate-50">
                          <tr>
                            <th className="px-4 py-2">Ticker</th>
                            <th className="px-4 py-2">Data Points</th>
                            <th className="px-4 py-2">MAE (xADR)</th>
                            <th className="px-4 py-2">Accuracy</th>
                          </tr>
                        </thead>
                        <tbody>
                          {maxExcResults.perTickerStats.map(t => (
                            <tr key={t.ticker} className="border-b border-slate-100 hover:bg-slate-50">
                              <td className="px-4 py-2 font-semibold">{t.ticker}</td>
                              <td className="px-4 py-2">{t.count}</td>
                              <td className="px-4 py-2">{t.mae.toFixed(3)}</td>
                              <td className="px-4 py-2">
                                <div className="w-full bg-slate-200 rounded-full h-2">
                                  <div className={`h-2 rounded-full ${t.mae < 0.3 ? 'bg-emerald-500' : t.mae < 0.5 ? 'bg-amber-500' : 'bg-red-500'}`} style={{ width: `${Math.max(5, Math.min(100, (1 - t.mae) * 100))}%` }}></div>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Sensitivity Analysis */}
            {maxExcResults && !isMaxExcTraining && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-bold text-slate-800">Sensitivity Analysis</h3>
                    <p className="text-sm text-slate-500">Permutation importance: measures R² drop when each feature is shuffled.</p>
                  </div>
                  <button
                    onClick={handleRunSensitivity}
                    disabled={isRunningSensitivity}
                    className={`py-2 px-6 rounded-md font-bold text-white transition-colors text-sm shadow-sm ${isRunningSensitivity ? 'bg-slate-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}`}
                  >
                    {isRunningSensitivity ? 'Analyzing...' : 'Run Analysis'}
                  </button>
                </div>

                {/* Linear Regression Formula Display */}
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                  <h4 className="text-sm font-bold text-slate-700 mb-2">Linear Regression Proxy Formula:</h4>
                  {maxExcResults.linearFormula ? (
                    <>
                      <div className="bg-white p-3 rounded border border-slate-200 overflow-x-auto shadow-inner text-sm font-mono text-slate-800 break-words whitespace-pre-wrap">
                        {maxExcResults.linearFormula}
                      </div>
                      <p className="text-xs text-slate-500 mt-2 italic">
                        Note: This static formula minimizes squares via generic linear regression to find a baseline 'envelope'.
                      </p>
                    </>
                  ) : (
                    <div className="text-sm text-slate-400 italic py-2">
                      (Training regression baseline... Please wait or retrain to refresh indices.)
                    </div>
                  )}
                </div>

                {maxExcSensitivity && maxExcSensitivity.length > 0 && (
                  <div className="space-y-2 mt-2">
                    {maxExcSensitivity.map((feat, idx) => {
                      const allNames = maxExcResults.preparedData.allFeatureNames || ['Projected RVol', 'Gap %', 'First Bar Vol / Avg Vol', 'First 5-min Range / ADR', '% Above Open after 5 min', '20-Day Up/Down Ratio', '20-day ADR', 'ATR(14)', 'ATR Dist from 10 EMA', 'ATR Dist from 20 EMA', 'ATR Dist from 50 EMA', 'VIX Open', 'VIX % Change', 'VIX 200 SMA', 'VIX Dist from 200 SMA'];
                      const origIdx = allNames.indexOf(feat.featureName);
                      return (
                        <div key={idx} className="flex items-center gap-3">
                          <input
                            type="checkbox"
                            checked={origIdx >= 0 ? enabledFeatures[origIdx] : true}
                            onChange={() => {
                              if (origIdx >= 0) {
                                setEnabledFeatures(prev => {
                                  const next = [...prev];
                                  next[origIdx] = !next[origIdx];
                                  return next;
                                });
                              }
                            }}
                            className="w-4 h-4 accent-emerald-600 cursor-pointer flex-shrink-0"
                          />
                          <span className={`text-sm font-medium w-48 text-right truncate ${origIdx >= 0 && !enabledFeatures[origIdx] ? 'text-slate-400 line-through' : 'text-slate-700'}`} title={feat.featureName}>{feat.featureName}</span>
                          <div className="flex-1 bg-slate-100 rounded-full h-6 relative overflow-hidden">
                            <div
                              className={`h-6 rounded-full transition-all duration-500 ${feat.correlationSign === 'positive' ? 'bg-emerald-500' : 'bg-rose-500'} ${origIdx >= 0 && !enabledFeatures[origIdx] ? 'opacity-30' : ''}`}
                              style={{ width: `${Math.min(100, Math.max(2, Math.abs(feat.importanceScore) * 500))}%` }}
                            />
                            <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-slate-800">
                              {(feat.importanceScore * 100).toFixed(2)}% | {feat.correlationSign === 'positive' ? '+' : '−'}
                            </span>
                          </div>
                        </div>
                      );
                    })}

                    <div className="flex flex-col md:flex-row items-center justify-between mt-4 pt-4 border-t border-slate-200 gap-4">
                      <p className="text-xs text-slate-400 text-center md:text-left">Green = positive correlation, Red = negative. Uncheck low-impact features, then retrain.</p>
                      <button
                        onClick={handleTrainMaxExcursion}
                        disabled={isMaxExcTraining || enabledFeatures.filter(Boolean).length < 2}
                        className={`py-2 px-6 rounded-md font-bold text-white transition-colors text-sm shadow-sm ${isMaxExcTraining ? 'bg-slate-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700'}`}
                      >
                        {isMaxExcTraining ? 'Retraining...' : `Retrain with ${enabledFeatures.filter(Boolean).length} Features`}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'regime' && (
          <MarketRegime />
        )}

      </div>
    </div>
  );
}
