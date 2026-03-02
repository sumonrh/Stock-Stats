import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Bar } from 'recharts';
import { Activity, TrendingUp, TrendingDown, Minus, Layers, RefreshCw, AlertCircle } from 'lucide-react';

// --- MATH & STATS UTILITIES ---
const mean = (arr) => arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
const stdDev = (arr, avg) => {
    if (arr.length < 2) return 0;
    const m = avg ?? mean(arr);
    return Math.sqrt(arr.reduce((sq, n) => sq + Math.pow(n - m, 2), 0) / (arr.length - 1));
};

const calculateSMA = (data, period, key) => {
    return data.map((_, i, arr) => {
        const start = Math.max(0, i - period + 1);
        const slice = arr.slice(start, i + 1).map(d => typeof d === 'number' ? d : d[key]);
        const validSlice = slice.filter(v => v !== null && v !== undefined);
        return validSlice.length > 0 ? mean(validSlice) : null;
    });
};

const calculateATR = (data, period) => {
    const tr = data.map((d, i, arr) => {
        if (i === 0) return (d.vixHigh || d.vix) - (d.vixLow || d.vix);
        const prevClose = arr[i - 1].vix;
        const high = d.vixHigh || d.vix;
        const low = d.vixLow || d.vix;
        return Math.max(
            high - low,
            Math.abs(high - prevClose),
            Math.abs(low - prevClose)
        );
    });
    return calculateSMA(tr, period);
};

// Simple K-Means used as a GMM approximation for browser performance
const kMeans1D = (data, k = 3, maxIter = 50) => {
    if (!data || data.length === 0) return [];
    const validData = data.filter(d => !isNaN(d));
    if (validData.length === 0) return new Array(data.length).fill(0);

    let centroids = [
        Math.min(...validData),
        mean(validData),
        Math.max(...validData)
    ];
    let assignments = new Array(data.length).fill(0);

    for (let iter = 0; iter < maxIter; iter++) {
        let changed = false;
        for (let i = 0; i < data.length; i++) {
            if (isNaN(data[i])) continue;
            let minDist = Infinity;
            let bestK = 0;
            for (let j = 0; j < k; j++) {
                const dist = Math.abs(data[i] - centroids[j]);
                if (dist < minDist) { minDist = dist; bestK = j; }
            }
            if (assignments[i] !== bestK) {
                assignments[i] = bestK;
                changed = true;
            }
        }
        if (!changed) break;
        for (let j = 0; j < k; j++) {
            const cluster = data.filter((d, i) => assignments[i] === j && !isNaN(d));
            if (cluster.length > 0) centroids[j] = mean(cluster);
        }
    }

    const sortedIndices = centroids.map((c, i) => ({ c, i })).sort((a, b) => a.c - b.c).map(x => x.i);
    const mapArr = [];
    sortedIndices.forEach((oldIdx, newIdx) => { mapArr[oldIdx] = newIdx; });

    return assignments.map(a => mapArr[a]);
};

export default function MarketRegime() {
    const [data, setData] = useState([]);
    const [activeModel, setActiveModel] = useState('ensemble');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [refreshCount, setRefreshCount] = useState(0);

    const fetchData = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);

            const res = await fetch('/api/market-regime');
            if (!res.ok) throw new Error('Backend returned an error. Make sure the server is running.');
            const rawData = await res.json();

            if (!rawData || rawData.length === 0) {
                throw new Error('No data returned from the server.');
            }

            const returns = rawData.map(d => d.ret);
            const sma200Vix = calculateSMA(rawData, 200, 'vix');
            const atr14Vix = calculateATR(rawData, 14);

            const vol14 = returns.map((_, i) => {
                if (i < 14) return 0;
                return stdDev(returns.slice(i - 14, i));
            });

            const enrichedData = rawData.map((d, i) => ({
                ...d,
                sma200Vix: sma200Vix[i],
                atr14Vix: atr14Vix[i],
                vol14: vol14[i]
            }));

            setData(enrichedData);
            setLoading(false);
        } catch (err) {
            console.error("Error fetching market regime data:", err);
            setError(err.message || "Failed to fetch market data. Please check your connection.");
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
    }, [fetchData, refreshCount]);

    const processedData = useMemo(() => {
        if (data.length === 0) return [];
        const result = [...data];

        // MODEL 1: VIX Model (200 SMA + ATR14)
        for (let i = 0; i < result.length; i++) {
            const d = result[i];
            if (!d.sma200Vix || !d.atr14Vix) {
                d.vixRegime = 1;
                continue;
            }
            const upperBand = d.sma200Vix + 0.5 * d.atr14Vix;
            const lowerBand = d.sma200Vix - 0.5 * d.atr14Vix;
            if (d.vix > upperBand) d.vixRegime = 2;
            else if (d.vix < lowerBand) d.vixRegime = 0;
            else d.vixRegime = 1;
        }

        // MODEL 2: GMM (K-Means approx)
        const volData = result.map(d => d.vol14);
        const gmmRegimes = kMeans1D(volData, 3);
        for (let i = 0; i < result.length; i++) { result[i].gmmRegime = gmmRegimes[i]; }

        // MODEL 3: HMM (Viterbi approx)
        const hmmRegimes = [...gmmRegimes];
        for (let i = 1; i < hmmRegimes.length - 1; i++) {
            if (hmmRegimes[i - 1] === hmmRegimes[i + 1] && hmmRegimes[i] !== hmmRegimes[i - 1]) {
                hmmRegimes[i] = hmmRegimes[i - 1];
            }
        }
        for (let i = 0; i < result.length; i++) { result[i].hmmRegime = hmmRegimes[i]; }

        // MODEL 4: PELT (Windowed Variance)
        const peltRegimes = new Array(result.length).fill(1);
        const windowSize = 20;
        for (let i = windowSize; i < result.length; i++) {
            const window = result.slice(i - windowSize, i).map(d => d.ret);
            const currentVol = stdDev(window);
            if (currentVol > 0.015) peltRegimes[i] = 2;
            else if (currentVol < 0.008) peltRegimes[i] = 0;
            else peltRegimes[i] = 1;
        }
        for (let i = 0; i < result.length; i++) { result[i].peltRegime = peltRegimes[i]; }

        // MODEL 5: Ensemble (VIX, HMM, PELT)
        for (let i = 0; i < result.length; i++) {
            const votes = [result[i].vixRegime, result[i].hmmRegime, result[i].peltRegime];
            const counts = { 0: 0, 1: 0, 2: 0 };
            votes.forEach(v => counts[v]++);
            let winner = 0;
            let maxCount = 0;
            for (const [regime, count] of Object.entries(counts)) {
                if (count > maxCount) { maxCount = count; winner = parseInt(regime); }
            }
            result[i].ensembleRegime = winner;
        }

        const maxPrice = Math.max(...result.map(d => d.price)) * 1.1;
        return result.map(d => {
            let activeRegime;
            switch (activeModel) {
                case 'vix': activeRegime = d.vixRegime; break;
                case 'gmm': activeRegime = d.gmmRegime; break;
                case 'hmm': activeRegime = d.hmmRegime; break;
                case 'pelt': activeRegime = d.peltRegime; break;
                case 'ensemble': activeRegime = d.ensembleRegime; break;
                default: activeRegime = d.ensembleRegime;
            }
            return {
                ...d,
                activeRegime,
                bgBull: activeRegime === 0 ? maxPrice : 0,
                bgChoppy: activeRegime === 1 ? maxPrice : 0,
                bgBear: activeRegime === 2 ? maxPrice : 0,
            };
        });
    }, [data, activeModel]);

    const stats = useMemo(() => {
        if (!processedData.length) return { regimeText: '-', colorClass: 'text-slate-500', icon: <Minus className="w-5 h-5 mr-2 text-slate-500" /> };
        const latest = processedData[processedData.length - 1];
        if (latest.activeRegime === 0) return { regimeText: 'Bull Market', colorClass: 'text-emerald-500', icon: <TrendingUp className="w-5 h-5 mr-2 text-emerald-500" /> };
        if (latest.activeRegime === 2) return { regimeText: 'Bear Market', colorClass: 'text-red-500', icon: <TrendingDown className="w-5 h-5 mr-2 text-red-500" /> };
        return { regimeText: 'Choppy / Sideways', colorClass: 'text-amber-500', icon: <Minus className="w-5 h-5 mr-2 text-amber-500" /> };
    }, [processedData]);

    // --- LOADING STATE ---
    if (loading) return (
        <div className="flex flex-col items-center justify-center py-24 gap-5">
            <div className="relative">
                <RefreshCw className="w-10 h-10 animate-spin text-blue-500" />
                <Activity className="w-5 h-5 text-blue-300 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
            </div>
            <div className="text-center">
                <p className="text-base font-semibold text-slate-700">Analyzing Market Context</p>
                <p className="text-xs text-slate-400">Pulling SPY & VIX data via Yahoo Finance...</p>
            </div>
        </div>
    );

    // --- ERROR STATE ---
    if (error) return (
        <div className="flex flex-col gap-5 items-center justify-center py-24 text-center">
            <div className="p-3 bg-red-50 rounded-full border border-red-200">
                <AlertCircle className="w-10 h-10 text-red-500" />
            </div>
            <div className="space-y-1">
                <h2 className="text-lg font-bold text-slate-800">Data Fetching Failed</h2>
                <p className="max-w-md text-slate-500 text-sm leading-relaxed">{error}</p>
            </div>
            <button
                onClick={() => setRefreshCount(c => c + 1)}
                className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg shadow transition-all active:scale-95 text-sm"
            >
                Retry Connection
            </button>
        </div>
    );

    // --- MAIN RENDER ---
    return (
        <div className="space-y-6">

            {/* HEADER */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-5">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-indigo-50 to-blue-100 rounded-xl border border-indigo-200/60">
                        <Activity className="w-7 h-7 text-indigo-600" />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-slate-800 tracking-tight">Market Regime Intelligence</h2>
                        <p className="text-slate-500 text-xs font-medium mt-0.5">Statistical volatility & structural break analysis</p>
                    </div>
                </div>

                <div className="flex flex-wrap items-center gap-3 w-full lg:w-auto">
                    <div className="flex flex-col gap-1 flex-grow lg:flex-grow-0">
                        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Analytical Model</label>
                        <select
                            className="bg-slate-50 border border-slate-200 text-slate-700 rounded-lg px-3 py-2.5 outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-400 transition-all text-sm font-semibold cursor-pointer"
                            value={activeModel}
                            onChange={(e) => setActiveModel(e.target.value)}
                        >
                            <option value="ensemble">3-Expert Voting Ensemble</option>
                            <option value="vix">VIX Structural (200SMA/ATR)</option>
                            <option value="hmm">Gaussian HMM (Viterbi)</option>
                            <option value="pelt">PELT Variance Shift</option>
                            <option value="gmm">Gaussian Mixture (GMM)</option>
                        </select>
                    </div>
                    <button
                        onClick={() => setRefreshCount(c => c + 1)}
                        className="lg:mt-5 p-2.5 bg-slate-100 border border-slate-200 rounded-lg hover:bg-slate-200 text-slate-500 hover:text-slate-700 transition-all active:scale-95 group"
                        title="Reload Market Data"
                    >
                        <RefreshCw className="w-4 h-4 group-active:rotate-180 transition-transform duration-500" />
                    </button>
                </div>
            </div>

            {/* METRICS */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm relative overflow-hidden">
                    <div className={`absolute top-0 right-0 w-20 h-20 blur-3xl opacity-20 ${stats.colorClass.includes('emerald') ? 'bg-emerald-400' : stats.colorClass.includes('red') ? 'bg-red-400' : 'bg-amber-400'}`}></div>
                    <h3 className="text-slate-400 text-[10px] font-bold uppercase tracking-widest mb-2">Current Inference</h3>
                    <div className={`text-xl font-bold flex items-center tracking-tight ${stats.colorClass}`}>
                        {stats.icon}
                        {stats.regimeText}
                    </div>
                </div>
                <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                    <h3 className="text-slate-400 text-[10px] font-bold uppercase tracking-widest mb-2">Price Level (SPY)</h3>
                    <div className="text-xl font-bold text-slate-800 flex items-baseline gap-2 tracking-tight">
                        ${processedData[processedData.length - 1]?.price.toFixed(2)}
                        <span className="text-[10px] font-semibold text-slate-400">ETF</span>
                    </div>
                </div>
                <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm sm:col-span-2 lg:col-span-1">
                    <h3 className="text-slate-400 text-[10px] font-bold uppercase tracking-widest mb-2">Volatility Baseline</h3>
                    <div className="text-xl font-bold text-slate-800 flex items-center gap-3 tracking-tight">
                        {processedData[processedData.length - 1]?.vix.toFixed(2)}
                        <div className="px-2 py-0.5 bg-slate-100 rounded text-slate-500 text-[9px] font-bold border border-slate-200 uppercase tracking-tight">
                            SMA200: {processedData[processedData.length - 1]?.sma200Vix?.toFixed(1) || 'N/A'}
                        </div>
                    </div>
                </div>
            </div>

            {/* CHART CONTAINER */}
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
                    <h3 className="text-lg font-bold flex items-center gap-2 text-slate-800 tracking-tight">
                        <Layers className="w-5 h-5 text-blue-500" />
                        Regime Temporal Overlay
                    </h3>
                    <div className="flex gap-1.5 p-1 bg-slate-100 rounded-lg border border-slate-200">
                        <div className="flex items-center gap-1.5 px-3 py-1.5">
                            <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                            <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Bull</span>
                        </div>
                        <div className="flex items-center gap-1.5 px-3 py-1.5">
                            <div className="w-2 h-2 rounded-full bg-amber-500"></div>
                            <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Choppy</span>
                        </div>
                        <div className="flex items-center gap-1.5 px-3 py-1.5">
                            <div className="w-2 h-2 rounded-full bg-red-500"></div>
                            <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Bear</span>
                        </div>
                    </div>
                </div>

                <div className="h-[480px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={processedData} margin={{ top: 0, right: 0, left: -10, bottom: 0 }}>
                            <defs>
                                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} opacity={0.7} />
                            <XAxis
                                dataKey="day"
                                stroke="#94a3b8"
                                tick={{ fontSize: 9, fontWeight: 700, fill: '#94a3b8' }}
                                minTickGap={80}
                                tickMargin={12}
                                axisLine={false}
                            />
                            <YAxis
                                yAxisId="price"
                                orientation="right"
                                stroke="#94a3b8"
                                tick={{ fontSize: 9, fontWeight: 700, fill: '#94a3b8' }}
                                domain={['auto', 'auto']}
                                axisLine={false}
                                tickMargin={8}
                            />
                            <RechartsTooltip
                                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px', boxShadow: '0 10px 30px rgba(0,0,0,0.08)', padding: '10px 14px' }}
                                itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
                                labelStyle={{ color: '#94a3b8', fontSize: '10px', marginBottom: '4px', fontWeight: 'bold', letterSpacing: '0.05em' }}
                                formatter={(value, name) => {
                                    if (name === 'price') return [`$${value.toFixed(2)}`, 'SPY Close'];
                                    return null;
                                }}
                            />

                            <Bar yAxisId="price" dataKey="bgBull" fill="#22c55e" opacity={0.35} isAnimationActive={false} barSize={20} />
                            <Bar yAxisId="price" dataKey="bgChoppy" fill="#eab308" opacity={0.35} isAnimationActive={false} barSize={20} />
                            <Bar yAxisId="price" dataKey="bgBear" fill="#ef4444" opacity={0.35} isAnimationActive={false} barSize={20} />

                            <Line
                                yAxisId="price"
                                type="monotone"
                                dataKey="price"
                                stroke="#3b82f6"
                                strokeWidth={2.5}
                                dot={false}
                                activeDot={{ r: 5, fill: '#fff', stroke: '#3b82f6', strokeWidth: 2 }}
                                isAnimationActive={false}
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>

                <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
                    <div className="bg-slate-50 p-5 rounded-xl border border-slate-200">
                        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-1.5">
                            <div className="w-1.5 h-1.5 rounded-full bg-blue-500"></div>
                            Analytical Strategy
                        </h4>
                        <div className="text-xs text-slate-500 leading-relaxed font-medium space-y-2">
                            {activeModel === 'ensemble' && <p>The <span className="text-slate-800 font-bold">Expert Ensemble</span> implements a high-conviction majority voting filter. By requiring agreement across VIX structure, HMM hidden states, and PELT variance shifts, it effectively removes "regime jitter" and identifies meaningful market phase transitions.</p>}
                            {activeModel === 'vix' && <p>The <span className="text-slate-800 font-bold">VIX Structural Model</span> utilizes a dynamically anchored 200-day baseline. By measuring deviation in units of ATR, it recognizes when market fear has broken out of its long-term volatility corridor, signaling structural stress or complacency.</p>}
                            {activeModel === 'hmm' && <p>The <span className="text-slate-800 font-bold">Gaussian HMM</span> models the market as a Markov process where current volatility states are hidden. The Viterbi approximation finds the most probable sequence of regimes, prioritizing persistence and reducing sensitivity to intraday return spikes.</p>}
                            {activeModel === 'pelt' && <p>The <span className="text-slate-800 font-bold">PELT Break Analysis</span> monitors the standard deviation of return distributions over rolling windows. It identifies "changepoints" where the distribution characteristics shift significantly, signaling a potential change in the underlying regime.</p>}
                            {activeModel === 'gmm' && <p>The <span className="text-slate-800 font-bold">Gaussian Mixture Model</span> clusters 14-day rolling volatility into three distinct Gaussian distributions. It is the most responsive model but lacks temporal memory, making it prone to state-flipping in high-noise environments.</p>}
                        </div>
                    </div>

                    <div className="flex flex-col gap-3">
                        <div className="p-4 bg-blue-50 border border-blue-200/60 rounded-xl">
                            <h5 className="text-[10px] font-bold text-blue-600 uppercase tracking-widest mb-1.5">Technical Note</h5>
                            <p className="text-[11px] text-slate-500 leading-normal">
                                Calculations are performed in-browser using 500+ data points for SMA200 warm-up. Regime boundaries are dynamic and recalibrate as new historical data is ingested.
                            </p>
                        </div>
                        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl flex items-center justify-between">
                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Market Sample</span>
                            <span className="text-xs font-bold text-slate-600 italic">2 Year Historical Lookback</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
