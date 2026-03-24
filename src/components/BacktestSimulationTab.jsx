import React, { useState, useEffect, useRef } from 'react';
import { createChart, AreaSeries, CandlestickSeries, HistogramSeries } from 'lightweight-charts';
import './BacktestSimulationTab.css';

const BacktestSimulationTab = () => {
    // ---- State ----
    const [stocks, setStocks] = useState([]);
    const [selectedStocks, setSelectedStocks] = useState([]);
    const [config, setConfig] = useState({
        minRR: 1.1,
        minPotential: 3.0,
        riskPerTrade: 0.3,
        portfolioValue: 40000,
        tradeManagement: 'fixed',
        stopBuffer: 0.10,
        vwapFilter: true,
        openingRangeMinutes: 5,
        minRVol: 3.0,
        minProjVol: 5,
        marginEnabled: false,
        ptRatio: 0.7,
        trendFilter: 'none',
        maxDist10: '',
        maxDist20: '',
        maxDist50: ''
    });

    const [fetchTicker, setFetchTicker] = useState('');
    const [fetchDays, setFetchDays] = useState(100);
    const [polygonKey, setPolygonKey] = useState(localStorage.getItem('polygon_api_key') || '');
    const [isFetching, setIsFetching] = useState(false);


    const [results, setResults] = useState(null);
    const [trades, setTrades] = useState([]);
    const [progress, setProgress] = useState({ message: 'Disconnected', percent: 0 });
    const [isRunning, setIsRunning] = useState(false);
    const [activeTicker, setActiveTicker] = useState(null);
    const [chartType, setChartType] = useState('daily');

    // Refs
    const equityChartRef = useRef(null);
    const equitySeriesRef = useRef(null);
    const equityContainerRef = useRef(null);
    const stockChartRef = useRef(null);
    const stockPriceSeriesRef = useRef(null);
    const stockVolSeriesRef = useRef(null);
    const stockContainerRef = useRef(null);
    const abortControllerRef = useRef(null);

    // Initial Load & Chart Creation
    useEffect(() => {
        loadAvailableStocks();
        
        // --- Create Charts ---
        // We MUST check if children already exist or if we need to dispose old one (HMR safety)
        if (equityContainerRef.current) {
            if (equityChartRef.current) equityChartRef.current.remove();
            
            const chart = createChart(equityContainerRef.current, {
                width: equityContainerRef.current.clientWidth || 600,
                height: 300,
                layout: { background: { color: 'transparent' }, textColor: '#8b949e' },
                grid: { vertLines: { color: 'rgba(255,255,255,0.04)' }, horzLines: { color: 'rgba(255,255,255,0.04)' } },
                rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
                timeScale: { borderColor: 'rgba(255,255,255,0.1)' }
            });
            const series = chart.addSeries(AreaSeries, {
                lineColor: '#58a6ff', topColor: 'rgba(88, 166, 255, 0.4)', bottomColor: 'rgba(88, 166, 255, 0)', lineWidth: 2
            });
            equityChartRef.current = chart;
            equitySeriesRef.current = series;
        }

        if (stockContainerRef.current) {
            if (stockChartRef.current) stockChartRef.current.remove();
            
            const chart = createChart(stockContainerRef.current, {
                width: stockContainerRef.current.clientWidth || 300,
                height: 350,
                layout: { background: { color: 'transparent' }, textColor: '#8b949e' },
                grid: { vertLines: { color: 'rgba(255,255,255,0.04)' }, horzLines: { color: 'rgba(255,255,255,0.04)' } },
                rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
                timeScale: { borderColor: 'rgba(255,255,255,0.1)', timeVisible: true }
            });
            const priceSeries = chart.addSeries(CandlestickSeries, {
                upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            });
            const volSeries = chart.addSeries(HistogramSeries, {
                color: '#26a69a', priceFormat: { type: 'volume' }, priceScaleId: ''
            });
            volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
            
            stockChartRef.current = chart;
            stockPriceSeriesRef.current = priceSeries;
            stockVolSeriesRef.current = volSeries;
        }

        const handleResize = () => {
            if (equityChartRef.current && equityContainerRef.current) {
                equityChartRef.current.applyOptions({ width: equityContainerRef.current.clientWidth });
            }
            if (stockChartRef.current && stockContainerRef.current) {
                stockChartRef.current.applyOptions({ width: stockContainerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (abortControllerRef.current) abortControllerRef.current.abort();
            // We usually don't remove here during HMR if we want to keep it, 
            // but for reliability we will, since we recreate anyway.
            if (equityChartRef.current) {
                equityChartRef.current.remove();
                equityChartRef.current = null;
            }
            if (stockChartRef.current) {
                stockChartRef.current.remove();
                stockChartRef.current = null;
            }
        };
    }, []);

    // Load Stock Data on Ticker Change
    useEffect(() => {
        if (activeTicker && stockPriceSeriesRef.current) {
            loadStockChartData(activeTicker);
        }
    }, [activeTicker, chartType]);

    const loadStockChartData = async (ticker) => {
        try {
            const res = await fetch(`/api/backtest-simulation/chart-intraday/${ticker}?type=${chartType}`);
            const data = await res.json();
            if (data.price && stockPriceSeriesRef.current) stockPriceSeriesRef.current.setData(data.price);
            if (data.volume && stockVolSeriesRef.current) stockVolSeriesRef.current.setData(data.volume);
            if (stockChartRef.current) stockChartRef.current.timeScale().fitContent();
        } catch (e) {
            console.error("Error loading chart data:", e);
        }
    };

    const loadAvailableStocks = async () => {
        try {
            const res = await fetch(`/api/backtest-simulation/stocks?t=${Date.now()}`);
            const data = await res.json();
            setStocks(data.stocks || []);
        } catch (e) {
            console.error("Error loading stocks:", e);
        }
    };

    const handleFetchTicker = async () => {
        if (!fetchTicker) {
            alert("Ticker is required!");
            return;
        }
        setIsFetching(true);
        if (polygonKey) localStorage.setItem('polygon_api_key', polygonKey);

        try {
            const res = await fetch('/api/backtest-simulation/fetch-ticker', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ticker: fetchTicker.toUpperCase(),
                    apiKey: polygonKey,
                    days: fetchDays
                })
            });
            const data = await res.json();
            if (data.success) {
                alert(`Successfully fetched/appended ${data.added} bars for ${data.ticker}. Total: ${data.count}`);
                loadAvailableStocks();
            } else {
                throw new Error(data.error || "Fetch failed");
            }
        } catch (e) {
            alert("Error: " + e.message);
        } finally {
            setIsFetching(false);
        }
    };


    const runBacktest = async (isAll = false) => {
        if (isRunning) {
            abortControllerRef.current?.abort();
            setIsRunning(false);
            return;
        }
        const tickers = isAll ? stocks : selectedStocks;
        if (tickers.length === 0) {
            alert("No stocks selected!");
            return;
        }
        setIsRunning(true);
        setTrades([]);
        setResults(null);
        if (equitySeriesRef.current) equitySeriesRef.current.setData([]);
        abortControllerRef.current = new AbortController();

        try {
            const requestBody = { ...config, stocks: tickers };
            console.log('[Backtest] Sending config:', JSON.stringify(requestBody, null, 2));
            const res = await fetch('/api/backtest-simulation/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
                signal: abortControllerRef.current.signal
            });
            if (!res.body) throw new Error("No body");
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (value) {
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // save incomplete line 
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        try {
                            const msg = JSON.parse(line);
                            handleMessage(msg);
                        } catch (e) {
                            console.error('Error parsing stream line:', line, e);
                        }
                    }
                }
                if (done) break;
            }
            if (buffer.trim()) {
                try {
                    const msg = JSON.parse(buffer);
                    handleMessage(msg);
                } catch (e) { }
            }
        } catch (e) {
            if (e.name !== 'AbortError') setProgress({ message: `Error: ${e.message}`, percent: 0 });
        } finally {
            setIsRunning(false);
        }
    };

    const handleMessage = (msg) => {
        if (msg.type === 'progress') {
            setProgress({ message: msg.message, percent: msg.percent });
        } else if (msg.type === 'trade') {
            setTrades(prev => [...prev, msg.trade]);
            if (equitySeriesRef.current && msg.equity) {
                try {
                    equitySeriesRef.current.update({ time: msg.trade.date, value: msg.equity });
                } catch (err) { }
            }
        } else if (msg.type === 'summary') {
            setResults(msg.summary);
            
            if (msg.summary.allTrades) {
                setTrades(msg.summary.allTrades);
            }
            
            if (equitySeriesRef.current && msg.summary.equityCurve) {
                try {
                    const curveData = msg.summary.equityCurve
                        .filter(pt => pt.date !== 'start' && pt.date.includes('-'))
                        .map(pt => ({ time: pt.date, value: pt.equity }))
                        .sort((a, b) => a.time.localeCompare(b.time)); // explicitly sort to prevent LightWeight Charts fatal crash
                        
                    // deduplicate exact times if needed (Lightweight charts crashes on exact duplicates sometimes if not the last item)
                    const uniqueCurve = [];
                    for(const pt of curveData) {
                        if (uniqueCurve.length === 0 || uniqueCurve[uniqueCurve.length - 1].time !== pt.time) {
                            uniqueCurve.push(pt);
                        } else {
                            uniqueCurve[uniqueCurve.length - 1].value = pt.value; // override
                        }
                    }

                    equitySeriesRef.current.setData(uniqueCurve);
                    if (equityChartRef.current) equityChartRef.current.timeScale().fitContent();
                } catch (err) {
                    console.error("Chart Render Error:", err);
                }
            }
        }
    };

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        let newValue;
        if (type === 'checkbox') {
            newValue = checked;
        } else if (type === 'number') {
            newValue = value === '' ? '' : parseFloat(value);
        } else {
            // select, text, etc — always keep as string
            newValue = value;
        }
        setConfig(prev => ({ ...prev, [name]: newValue }));
    };

    const handleStockSelect = (e) => {
        const values = Array.from(e.target.selectedOptions, option => option.value);
        setSelectedStocks(values);
        if (values.length === 1) setActiveTicker(values[0]);
        else setActiveTicker(null);
    };

    const handleSelectAll = () => {
        setSelectedStocks([...stocks]);
    };


    const renderStats = () => {
        if (!results) return null;
        const stats = [
            { label: 'Total Trades', value: results.totalTrades || 0 },
            { label: 'Win Rate', value: `${(results.winRate || 0).toFixed(1)}%` },
            { label: 'Total P/L', value: `$${(results.totalPL || 0).toFixed(2)}`, color: (results.totalPL || 0) >= 0 ? 'text-green' : 'text-red' },
            { label: 'Exp Value (R)', value: (results.expectedValue || 0).toFixed(2) },
            { label: 'Profit Factor', value: (results.profitFactor || 0).toFixed(2) },
            { label: 'Sharpe Ratio', value: (results.sharpeRatio || 0).toFixed(2) },
            { label: 'Max Drawdown', value: `$${(results.maxDrawdown || 0).toFixed(2)}` }
        ];
        return (
            <div className="bt-stats-grid">
                {stats.map((s, i) => (
                    <div className="bt-stat-card" key={i}>
                        <div className="label">{s.label}</div>
                        <div className={`value ${s.color || ''}`}>{s.value}</div>
                    </div>
                ))}
            </div>
        );
    };

    return (
        <div id="backtest-view">
            <div className="bt-config-panel">
                <div className="bt-config-col">
                    <h3>⚙ Settings</h3>
                    <div className="bt-field">
                        <label>Min R:R</label>
                        <input type="number" name="minRR" value={config.minRR} onChange={handleInputChange} step="0.1" />
                    </div>
                    <div className="bt-field">
                        <label>Min Potential %</label>
                        <input type="number" name="minPotential" value={config.minPotential} onChange={handleInputChange} step="0.1" />
                    </div>
                    <div className="bt-field">
                        <label>Risk Per Trade %</label>
                        <input type="number" name="riskPerTrade" value={config.riskPerTrade} onChange={handleInputChange} step="0.05" />
                    </div>
                    <div className="bt-field">
                        <label>Portfolio Start ($)</label>
                        <input type="number" name="portfolioValue" value={config.portfolioValue} onChange={handleInputChange} />
                    </div>
                    <div className="bt-field">
                        <label>Trend Filter</label>
                        <select name="trendFilter" value={config.trendFilter} onChange={handleInputChange}>
                            <option value="none">None</option>
                            <option value="uptrend">Uptrend (10&gt;20&gt;50&gt;200)</option>
                            <option value="downtrend">Downtrend (10&lt;20&lt;50&lt;200)</option>

                        </select>
                    </div>
                </div>


                <div className="bt-config-col">
                    <h3>🚀 Strategy</h3>
                    <div className="bt-field">
                        <label>Trade Management</label>
                        <select name="tradeManagement" value={config.tradeManagement} onChange={handleInputChange}>
                            <option value="fixed">Fixed Stop/Target</option>
                            <option value="trailing">Trailing Stop (0.5R)</option>
                            <option value="active">Active Management (Scale out at 1R)</option>
                        </select>
                    </div>
                    <div className="bt-field">
                        <label>Min RVol</label>
                        <input type="number" name="minRVol" value={config.minRVol} onChange={handleInputChange} step="0.1" />
                    </div>
                    <div className="bt-field">
                        <label>Min Proj Vol (M)</label>
                        <input type="number" name="minProjVol" value={config.minProjVol} onChange={handleInputChange} />
                    </div>
                    <div className="bt-field">
                        <label>Max Dist 10EMA (ATR)</label>
                        <input type="number" name="maxDist10" value={config.maxDist10} onChange={handleInputChange} step="0.1" placeholder="None" />
                    </div>
                    <div className="bt-field">
                        <label>Max Dist 20EMA (ATR)</label>
                        <input type="number" name="maxDist20" value={config.maxDist20} onChange={handleInputChange} step="0.1" placeholder="None" />
                    </div>
                    <div className="bt-field">
                        <label>Max Dist 50EMA (ATR)</label>
                        <input type="number" name="maxDist50" value={config.maxDist50} onChange={handleInputChange} step="0.1" placeholder="None" />
                    </div>
                    <div className="bt-field bt-checkbox-row">
                        <input type="checkbox" name="vwapFilter" checked={config.vwapFilter} onChange={handleInputChange} />
                        <label>VWAP Filter</label>
                    </div>
                </div>


                <div className="bt-config-col">
                    <h3>📁 Data Management</h3>
                    <div className="bt-field">
                        <label>Ticker</label>
                        <input type="text" value={fetchTicker} onChange={(e) => setFetchTicker(e.target.value)} placeholder="e.g. NVDA" />
                    </div>
                    {/* API key is hardcoded in .env — no need to show it */}
                    <div className="bt-field">
                        <label>Days</label>
                        <input type="number" value={fetchDays} onChange={(e) => setFetchDays(parseInt(e.target.value))} />
                    </div>
                    <button className="bt-btn bt-btn-secondary" onClick={handleFetchTicker} disabled={isFetching}>
                        {isFetching ? '⌛ Fetching...' : '📥 Download / Append Ticker'}
                    </button>
                    
                    <hr style={{ margin: '15px 0', border: 'none', borderTop: '1px solid #30363d' }} />
                    
                    <h3>📊 Select Tickers</h3>
                    <select multiple id="bt-stock-select" value={selectedStocks} onChange={handleStockSelect}>
                        {stocks.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <div className="bt-action-btns">
                        <button className="bt-btn bt-btn-secondary" onClick={handleSelectAll}>✔ Select All</button>
                        <button className={`bt-btn ${isRunning ? 'bt-btn-danger' : 'bt-btn-primary'}`} onClick={() => runBacktest(false)}>
                            {isRunning ? '⏹ Stop' : '▶ Run Backtest'}
                        </button>
                    </div>

                </div>

            </div>

            <div className="bt-status-panel">
                <div className="bt-progress-wrapper">
                    <div id="bt-progress-bar-container">
                        <div id="bt-progress-bar" style={{ width: `${progress.percent}%` }}></div>
                    </div>
                    <span id="bt-progress-label">{progress.message}</span>
                </div>
                <div style={{ fontSize: '11px', color: '#8b949e', marginTop: '6px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    <span style={{ color: config.trendFilter !== 'none' ? '#58a6ff' : '#555' }}>
                        🎯 Trend: <strong>{config.trendFilter === 'none' ? 'None' : config.trendFilter === 'uptrend' ? '📈 Uptrend' : '📉 Downtrend'}</strong>
                    </span>
                    <span style={{ color: config.vwapFilter ? '#58a6ff' : '#555' }}>
                        📊 VWAP: <strong>{config.vwapFilter ? 'ON' : 'OFF'}</strong>
                    </span>
                    <span>⚡ RVol ≥ <strong>{config.minRVol || 0}</strong></span>
                    <span>📦 Vol ≥ <strong>{config.minProjVol || 0}M</strong></span>
                    {config.maxDist10 ? <span>📏 10EMA ≤ <strong>{config.maxDist10}</strong>ATR</span> : null}
                    {config.maxDist20 ? <span>📏 20EMA ≤ <strong>{config.maxDist20}</strong>ATR</span> : null}
                    {config.maxDist50 ? <span>📏 50EMA ≤ <strong>{config.maxDist50}</strong>ATR</span> : null}
                </div>
            </div>

            {renderStats()}

            <div className="bt-charts-container">
                <div id="bt-equity-chart" ref={equityContainerRef}>
                    {!results && !isRunning && <div className="bt-chart-placeholder">Run a backtest to see the equity curve</div>}
                </div>
                <div className="bt-chart-panel">
                    <div className="bt-chart-header">
                        <div id="bt-chart-ticker-badge">{activeTicker || '---'}</div>
                        <div className="bt-chart-controls">
                            <button className={`bt-chart-btn ${chartType === 'daily' ? 'active' : ''}`} onClick={() => setChartType('daily')}>D</button>
                            <button className={`bt-chart-btn ${chartType === '5min' ? 'active' : ''}`} onClick={() => setChartType('5min')}>5m</button>
                        </div>
                    </div>
                    <div id="bt-stock-chart" ref={stockContainerRef}>
                        {!activeTicker && <div className="bt-chart-placeholder">Select one ticker to see chart</div>}
                    </div>
                </div>
            </div>

            <div className="bt-trades-table-container">
                <table className="bt-table">
                    <thead>
                        <tr>
                            <th>Date</th><th>Ticker</th><th>Side</th><th>Entry</th><th>Exit</th><th>P/L $</th><th>R</th><th>Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades.slice().reverse().map((t, i) => (
                            <tr key={i} className={t.plDollar >= 0 ? 'bt-win-row' : 'bt-loss-row'}>
                                <td>{t.date}</td><td><strong>{t.ticker}</strong></td><td className={t.side === 'LONG' ? 'text-green' : 'text-red'}>{t.side}</td>
                                <td className="num">${t.entryPrice?.toFixed(2)}</td><td className="num">${t.exitPrice?.toFixed(2)}</td>
                                <td className={`num ${t.plDollar >= 0 ? 'text-green' : 'text-red'}`}>${t.plDollar?.toFixed(2)}</td>
                                <td className="num">{t.rMultiple?.toFixed(2)}R</td><td>{t.exitReason}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default BacktestSimulationTab;
