import React, { useState, useEffect, useMemo } from 'react';
import { ComposedChart, Scatter, Line, Bar, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';
import ReactApexChart from 'react-apexcharts';
import Plot from 'react-plotly.js';
import { runTfjsPipeline, loadModelFromStorage, loadModelFromFiles, downloadModelFiles, evaluateLstmOnData } from './utils/tfjsEngine';
import { findBestFitRegression } from './utils/mathUtils';

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

  // Initial Load
  useEffect(() => {
    loadInitialData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadInitialData = async () => {
    setLoading(true);

    // Only load the very first ticker on initial startup to improve load speeds
    const firstTicker = INITIAL_TICKERS[0];
    const newRawData = {};
    newRawData[firstTicker] = await fetchRawTickerData(firstTicker);

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
  };



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
    const rawData = await fetchRawTickerData(symbol);

    setAvailableTickers(prev => prev.includes(symbol) ? prev : [...prev, symbol]);
    setRawMarketData(prev => ({ ...prev, [symbol]: rawData }));
    setSelectedTickerFilter(symbol);
    setSearchInput('');
    setLoading(false);
  };

  // Process data when raw data or periods change
  const processedData = useMemo(() => {
    let allProcessed = [];
    for (const [ticker, rawData] of Object.entries(rawMarketData)) {
      allProcessed = allProcessed.concat(processTickerData(rawData, ticker, rvolPeriod, adrPeriod));
    }
    return allProcessed;
  }, [rawMarketData, rvolPeriod, adrPeriod]);

  // Filter and sort data for Recharts (Sorting by X axis is crucial for Line charts)
  const { chartData, historicalRegression } = useMemo(() => {
    const filtered = selectedTickerFilter === 'ALL'
      ? processedData
      : processedData.filter(d => d.ticker === selectedTickerFilter);

    // Sort ascending by RVol
    const sortedChartData = filtered.sort((a, b) => a.rVol - b.rVol);

    // Calculate Best Fit Regression
    const points = sortedChartData.map(d => ({ x: d.rVol, y: d.maxExcursionAdr }));
    const regression = findBestFitRegression(points, 4);

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
      const inputTensor = tf.tensor3d([simulatedSequence]);
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

        {/* Header & Controls */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-6">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Intraday Excursion vs Volume Dashboard</h1>
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
                  className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-r-md transition-colors text-sm"
                >
                  Fetch
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
                <button
                  onClick={handleDownloadModel}
                  className="py-2 px-4 rounded-md font-semibold text-purple-700 bg-purple-100 hover:bg-purple-200 transition-colors text-sm border border-purple-300"
                  title="Export Model to PC"
                >
                  Export Model
                </button>
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
    </div>
  );
}
