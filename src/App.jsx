import React, { useState, useEffect, useMemo } from 'react';
import { ComposedChart, Scatter, Line, Bar, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';
import ReactApexChart from 'react-apexcharts';
import Plot from 'react-plotly.js';

// --- CONFIGURATION ---
const INITIAL_TICKERS = ['QQQ', 'VICR', 'RKLB', 'PL', 'ASTS', 'SEDG', 'MU', 'IREN', 'BE', 'LITE', 'OKLO', 'QBTS', 'WDC', 'EOSE', 'INTC'];

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

// Calculates Least Squares Linear Regression: y = mx + b and standard deviation
const calculateRegression = (data) => {
  if (!data || data.length < 2) return null;

  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  let n = 0;

  data.forEach(d => {
    const x = d.rVol;
    const y = d.maxExcursionAdr;
    if (x != null && y != null && isFinite(x) && isFinite(y)) {
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
      n++;
    }
  });

  if (n < 2) return null;

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Calculate standard deviation of the residuals
  let sumSquaredResiduals = 0;
  data.forEach(d => {
    const x = d.rVol;
    const y = d.maxExcursionAdr;
    if (x != null && y != null && isFinite(x) && isFinite(y)) {
      const predictedY = slope * x + intercept;
      const residual = y - predictedY;
      sumSquaredResiduals += Math.pow(residual, 2);
    }
  });

  // standard deviation of the error (regression standard error)
  const stdDev = Math.sqrt(sumSquaredResiduals / (n > 2 ? n - 2 : 1));

  return {
    slope,
    intercept,
    stdDev,
    equation: `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`
  };
};

// --- DATA SIMULATION & PROCESSING ---
// Simulates 1 year of realistic daily stock data as a fallback
const generateMockData = (ticker) => {
  const data = [];
  let currentPrice = 20 + Math.random() * 80;
  let baseVolume = 1000000 + Math.random() * 5000000;
  const now = new Date();

  for (let i = 400; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);

    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const normalVolatility = 0.03;
    let changePct = (Math.random() - 0.5) * normalVolatility * 2;
    let volume = baseVolume * (0.6 + Math.random() * 0.8);

    const isCatalyst = Math.random() > 0.92;
    if (isCatalyst) {
      volume *= (2 + Math.random() * 4);
      changePct = (Math.random() - 0.5) * normalVolatility * 8;
    }

    const open = currentPrice;
    const close = open * (1 + changePct);

    const maxOC = Math.max(open, close);
    const minOC = Math.min(open, close);
    const high = maxOC * (1 + Math.random() * 0.015);
    const low = minOC * (1 - Math.random() * 0.015);

    data.push({ date, open, high, low, close, volume });
    currentPrice = close;
  }
  return data;
};

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
  };

  // The Fetch wrapper: Attempts to hit backend, falls back to simulator if offline/in-browser
  const fetchRawTickerData = async (ticker) => {
    let rawData;
    try {
      // Backend integration point
      const response = await fetch(`/api/yahoo-finance2?ticker=${ticker}`);
      if (!response.ok) throw new Error("Backend not available");
      rawData = await response.json();
      rawData = rawData.map(d => ({ ...d, date: new Date(d.date) }));
    } catch {
      // Fallback for missing backend
      console.log(`Backend fetch failed for ${ticker}, utilizing realistic data simulator.`);
      rawData = generateMockData(ticker);
    }
    return rawData;
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
  const { chartData, regressionObj } = useMemo(() => {
    const filtered = selectedTickerFilter === 'ALL'
      ? processedData
      : processedData.filter(d => d.ticker === selectedTickerFilter);

    // Calculate Regression and SD
    const regression = calculateRegression(filtered);

    // Sort ascending by RVol and attach the regression lines (mean and SD bands)
    const sortedChartData = filtered
      .sort((a, b) => a.rVol - b.rVol)
      .map(point => {
        let regressionY = null,
          regressionYPlus1SD = null, regressionYMinus1SD = null,
          regressionYPlus2SD = null, regressionYMinus2SD = null;

        if (regression) {
          regressionY = (regression.slope * point.rVol) + regression.intercept;
          regressionYPlus1SD = regressionY + regression.stdDev;
          regressionYMinus1SD = regressionY - regression.stdDev;
          regressionYPlus2SD = regressionY + 2 * regression.stdDev;
          regressionYMinus2SD = regressionY - 2 * regression.stdDev;
        }

        return {
          ...point,
          regressionY,
          regressionYPlus1SD,
          regressionYMinus1SD,
          regressionYPlus2SD,
          regressionYMinus2SD,
          equation: regression?.equation
        };
      });

    return { chartData: sortedChartData, regressionObj: regression };
  }, [processedData, selectedTickerFilter]);

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

    // Regression Lines
    if (regressionObj) {
      const sortedChartData = [...chartData].sort((a, b) => a.rVol - b.rVol);
      const rx = sortedChartData.map(d => d.rVol);

      traces.push({
        x: rx, y: sortedChartData.map(d => d.regressionY),
        mode: 'lines', type: 'scatter', name: 'Trendline', line: { color: '#ef4444', width: 2 }, hoverinfo: 'skip'
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
  }, [chartData, regressionObj, rvolDistributionData, excDistributionData, rvolStats, excStats]);

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
        emaArr.push({ x: new Date(time).getTime(), y: parseFloat(currentEma.toFixed(2)) });
      }
      return emaArr;
    };

    const candleData = [];
    const volumeData = [];
    raw.forEach(d => {
      const time = typeof d.date === 'string' ? d.date.split('T')[0] : d.date.toISOString().split('T')[0];
      const timeMs = new Date(time).getTime();
      const o = d.open || 0;
      const h = d.high || 0;
      const l = d.low || 0;
      const c = d.close || 0;
      candleData.push({
        x: timeMs,
        y: [
          parseFloat(o.toFixed(2)),
          parseFloat(h.toFixed(2)),
          parseFloat(l.toFixed(2)),
          parseFloat(c.toFixed(2))
        ]
      });
      volumeData.push({
        x: timeMs,
        y: d.volume
      });
    });

    const priceSeries = [
      { name: 'Candle', type: 'candlestick', data: candleData },
      { name: '10 EMA', type: 'line', data: calcEma(10) },
      { name: '20 EMA', type: 'line', data: calcEma(20) },
      { name: '50 EMA', type: 'line', data: calcEma(50) }
    ];

    const volumeSeries = [
      { name: 'Volume', type: 'bar', data: volumeData }
    ];

    const maxDate = new Date(raw[raw.length - 1]?.date || Date.now());
    const minDate = new Date(maxDate);
    minDate.setMonth(minDate.getMonth() - 6);

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
        type: 'datetime',
        min: minDate.getTime(),
        max: maxDate.getTime(),
        labels: { show: false },
        axisBorder: { show: false },
        axisTicks: { show: false }
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
        type: 'datetime',
        min: minDate.getTime(),
        max: maxDate.getTime(),
        labels: { style: { colors: '#64748b' } }
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
      colors: ['#cbd5e1'],
      legend: { show: false },
      tooltip: { shared: false },
      dataLabels: { enabled: false }
    };

    return { priceOptions, priceSeries, volumeOptions, volumeSeries };
  }, [selectedTickerFilter, rawMarketData]);

  const regressionEquation = regressionObj ? regressionObj.equation : null;
  const regressionStdDev = regressionObj && regressionObj.stdDev ? regressionObj.stdDev.toFixed(4) : null;

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

        {/* Main Visualization */}
        {marginalPlotState && (
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <div className="mb-4 flex flex-col sm:flex-row sm:justify-between sm:items-end">
              <div>
                <h2 className="text-lg font-bold text-slate-800">Scatter Plot with Probability Distributions</h2>
                <p className="text-xs text-slate-500">Y-Axis: Absolute Max Excursion Multiple. X-Axis: Relative Volume.</p>
              </div>
              {regressionEquation && (
                <div className="mt-2 sm:mt-0 bg-blue-50 border border-blue-200 text-blue-800 text-xs px-3 py-1 rounded-full font-semibold">
                  <span className="font-mono">Relationship: {regressionEquation}</span>
                  <span className="ml-2 text-blue-600 block sm:inline">| SD: ±{regressionStdDev}</span>
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

        {/* Statistical Summary Table */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
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

        {/* Advanced Statistical Analysis: High Excursion Probabilities */}
        {advancedStats.length > 0 && (
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm mt-6 overflow-hidden">
            <div className="bg-slate-50 border-b border-slate-200 p-4">
              <h3 className="text-md font-bold text-slate-800">Advanced High-Excursion Probabilities</h3>
              <p className="text-xs text-slate-500 mt-1">Historically analyzing the percentage of days that push a <strong>High Excursion (&ge; 1.5 ADR)</strong> once an RVol threshold is breached.</p>
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
                      <span className="text-sm font-medium text-slate-700">Win Rate:</span>
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
          </div>
        )}        {/* Selected Ticker Stock Chart */}
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
