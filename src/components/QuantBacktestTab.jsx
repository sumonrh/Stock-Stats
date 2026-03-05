import React, { useState, useEffect, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ZAxis } from 'recharts';
import QuantModelTrainer from './QuantModelTrainer';

const BINS = [
    { label: '< 20', min: 0, max: 20 },
    { label: '20 - 40', min: 20, max: 40 },
    { label: '40 - 60', min: 40, max: 60 },
    { label: '60 - 80', min: 60, max: 80 },
    { label: '> 80', min: 80, max: 101 }
];

export default function QuantBacktestTab({ availableTickers, rawMarketData }) {
    const [backtestTickers, setBacktestTickers] = useState(['NVDA', 'TSLA', 'PLTR', 'AAPL', 'MSFT']);
    const [newTickerInput, setNewTickerInput] = useState('');
    const [backtestData, setBacktestData] = useState([]);
    const [loading, setLoading] = useState(false);

    const [xAxisMetric, setXAxisMetric] = useState('quantScore');
    const [yAxisMetric, setYAxisMetric] = useState('ret1W');
    const [colorMetric, setColorMetric] = useState('vcp');

    const handleAddTicker = (e) => {
        e.preventDefault();
        const symbol = newTickerInput.trim().toUpperCase();
        if (symbol && !backtestTickers.includes(symbol)) {
            setBacktestTickers(prev => [...prev, symbol]);
        }
        setNewTickerInput('');
    };

    const handleRemoveTicker = (ticker) => {
        setBacktestTickers(prev => prev.filter(t => t !== ticker));
    };

    const runBacktest = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/quant-backtest/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers: backtestTickers })
            });
            const data = await res.json();
            if (res.ok) {
                setBacktestData(data);
            } else {
                alert("Error: " + data.error);
            }
        } catch (e) {
            console.error(e);
            alert("Failed to run backtest");
        }
        setLoading(false);
    };

    const scatterData = useMemo(() => {
        return backtestData.map(d => ({
            x: d[xAxisMetric],
            y: d[yAxisMetric],
            z: d[colorMetric],
            ticker: d.ticker,
            date: d.date
        }));
    }, [backtestData, xAxisMetric, yAxisMetric, colorMetric]);

    return (
        <div className="p-4 bg-gray-900 min-h-screen text-white">
            <h2 className="text-2xl font-bold mb-4 text-emerald-400">Quant Score Parameters Backtest</h2>

            <div className="flex flex-col md:flex-row gap-6">
                {/* Left Panel: Controls */}
                <div className="w-full md:w-1/3 bg-gray-800 p-4 rounded-lg shadow-lg">
                    <h3 className="text-xl font-semibold mb-2">Backtest Universe</h3>
                    <form onSubmit={handleAddTicker} className="flex gap-2 mb-4">
                        <input
                            type="text"
                            placeholder="Add Ticker..."
                            className="flex-1 bg-gray-700 text-white rounded px-3 py-2 border border-gray-600 focus:outline-none focus:border-emerald-500"
                            value={newTickerInput}
                            onChange={(e) => setNewTickerInput(e.target.value)}
                        />
                        <button type="submit" className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-white font-medium">Add</button>
                    </form>
                    <div className="flex flex-wrap gap-2 mb-4">
                        {backtestTickers.map(ticker => (
                            <span key={ticker} className="bg-gray-700 px-3 py-1 rounded-full text-sm flex items-center gap-2">
                                {ticker}
                                <button onClick={() => handleRemoveTicker(ticker)} className="text-red-400 hover:text-red-300 font-bold">&times;</button>
                            </span>
                        ))}
                    </div>
                    <button
                        onClick={runBacktest}
                        disabled={loading || backtestTickers.length === 0}
                        className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-600 px-4 py-3 rounded text-white font-bold text-lg mb-6 shadow-md transition-colors"
                    >
                        {loading ? 'Crunching 4 Years of Data...' : 'Run Historical Backtest'}
                    </button>

                    <h3 className="text-xl font-semibold mb-2 mt-6">Chart Controls</h3>
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">X-Axis (Parameter)</label>
                            <select value={xAxisMetric} onChange={e => setXAxisMetric(e.target.value)} className="w-full bg-gray-700 rounded px-2 py-1.5 border border-gray-600">
                                <option value="quantScore">Quant Score (0-100)</option>
                                <option value="rsDelta">RS Delta (1-Day Slope %)</option>
                                <option value="vcp">VCP Status (0=None, 3=High)</option>
                                <option value="rVol">RVol (50-Day Avg)</option>
                                <option value="priceChangeOverAdr">Daily Move / ADR ($ Ratio)</option>
                                <option value="episodicPivotPower">Episodic Pivot Power (RVol * Move/ADR)</option>
                                <option value="ema10DistAtr">Distance to 10EMA (ATR)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Y-Axis (Future Return)</label>
                            <select value={yAxisMetric} onChange={e => setYAxisMetric(e.target.value)} className="w-full bg-gray-700 rounded px-2 py-1.5 border border-gray-600">
                                <option value="ret1W">1 Week Forward Return (%)</option>
                                <option value="ret2W">2 Week Forward Return (%)</option>
                                <option value="ret1M">1 Month Forward Return (%)</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Right Panel: Charts */}
                <div className="w-full md:w-2/3 bg-gray-800 p-4 rounded-lg shadow-lg flex flex-col items-center justify-center min-h-[500px]">
                    {backtestData.length === 0 ? (
                        <div className="text-gray-400 text-center">
                            <p className="text-xl mb-2">No data yet.</p>
                            <p>Add some tickers and run the backtest to see the mapping of Quant Score parameters to future returns.</p>
                        </div>
                    ) : (
                        <div className="w-full h-full min-h-[500px]">
                            <h3 className="text-center font-bold text-lg mb-2">{xAxisMetric} vs {yAxisMetric}</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                                    <XAxis type="number" dataKey="x" name={xAxisMetric} stroke="#a0aec0" domain={['auto', 'auto']} />
                                    <YAxis type="number" dataKey="y" name={yAxisMetric} stroke="#a0aec0" domain={['auto', 'auto']} />
                                    <ZAxis type="number" dataKey="z" range={[20, 100]} />
                                    <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                                        if (active && payload && payload.length) {
                                            const data = payload[0].payload;
                                            return (
                                                <div className="bg-gray-700 p-3 rounded shadow-lg border border-gray-600">
                                                    <p className="font-bold text-white">{data.ticker} ({data.date})</p>
                                                    <p className="text-emerald-400">{xAxisMetric}: {data.x}</p>
                                                    <p className="text-blue-400">{yAxisMetric}: {data.y}%</p>
                                                    <p className="text-yellow-400">{colorMetric}: {data.z}</p>
                                                </div>
                                            );
                                        }
                                        return null;
                                    }} />
                                    <Scatter name="Backtest Data" data={scatterData} fill="#10b981" fillOpacity={0.6} />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
            </div>

            {/* Neural Network Model Trainer */}
            {backtestData.length > 0 && <QuantModelTrainer backtestData={backtestData} />}

            {/* Bottom Panel: Data Table */}
            {backtestData.length > 0 && (
                <div className="mt-6 bg-gray-800 p-4 rounded-lg shadow-lg overflow-x-auto">
                    <h3 className="text-xl font-semibold mb-4">Historical Parameter Data ({backtestData.length} records)</h3>
                    <div className="max-h-96 overflow-y-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs text-gray-400 uppercase bg-gray-700 sticky top-0">
                                <tr>
                                    <th className="px-4 py-3">Date</th>
                                    <th className="px-4 py-3">Ticker</th>
                                    <th className="px-4 py-3">Quant Score</th>
                                    <th className="px-4 py-3">RS Delta</th>
                                    <th className="px-4 py-3">VCP</th>
                                    <th className="px-4 py-3">RVol</th>
                                    <th className="px-4 py-3">Move/ADR</th>
                                    <th className="px-4 py-3">EP Power</th>
                                    <th className="px-4 py-3">Dist 10EMA</th>
                                    <th className="px-4 py-3 text-right">1W Ret</th>
                                    <th className="px-4 py-3 text-right">2W Ret</th>
                                    <th className="px-4 py-3 text-right">1M Ret</th>
                                </tr>
                            </thead>
                            <tbody>
                                {backtestData.slice(-100).reverse().map((row, idx) => (
                                    <tr key={idx} className="border-b border-gray-700 hover:bg-gray-700">
                                        <td className="px-4 py-2">{row.date}</td>
                                        <td className="px-4 py-2 font-bold">{row.ticker}</td>
                                        <td className="px-4 py-2 text-emerald-400">{row.quantScore}</td>
                                        <td className="px-4 py-2">{row.rsDelta}</td>
                                        <td className="px-4 py-2">{row.vcp}</td>
                                        <td className="px-4 py-2">{row.rVol}</td>
                                        <td className="px-4 py-2">{row.priceChangeOverAdr}</td>
                                        <td className="px-4 py-2 font-semibold text-purple-400">{row.episodicPivotPower}</td>
                                        <td className="px-4 py-2">{row.ema10DistAtr}</td>
                                        <td className={`px-4 py-2 text-right ${row.ret1W >= 0 ? 'text-green-400' : 'text-red-400'}`}>{row.ret1W}%</td>
                                        <td className={`px-4 py-2 text-right ${row.ret2W >= 0 ? 'text-green-400' : 'text-red-400'}`}>{row.ret2W}%</td>
                                        <td className={`px-4 py-2 text-right ${row.ret1M >= 0 ? 'text-green-400' : 'text-red-400'}`}>{row.ret1M}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <p className="text-gray-400 text-xs mt-2 text-center">* Showing last 100 rows</p>
                </div>
            )}
        </div>
    );
}
