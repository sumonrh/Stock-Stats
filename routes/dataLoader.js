import express from 'express';
import yahooFinance from 'yahoo-finance2';
import fs from 'fs';
import path from 'path';
import { QuantScorer } from './quantScorer.js';

const router = express.Router();

const DATA_DIR = path.join(process.cwd(), 'Quant backtest stock data');
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Helper to append/save to CSV
const appendToCSV = (ticker, quotes) => {
    const filePath = path.join(DATA_DIR, `${ticker.toUpperCase()}.csv`);
    let existingDates = new Set();
    let isNewFile = true;

    if (fs.existsSync(filePath)) {
        isNewFile = false;
        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.split('\n');
        for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
                const parts = lines[i].split(',');
                existingDates.add(parts[0]);
            }
        }
    }

    let csvContent = isNewFile ? "Date,Open,High,Low,Close,Volume\n" : "";
    let appendedCount = 0;

    for (const q of quotes) {
        const dateStr = typeof q.date === 'string' ? q.date.split('T')[0] : q.date.toISOString().split('T')[0];
        if (!existingDates.has(dateStr)) {
            csvContent += `${dateStr},${q.open},${q.high},${q.low},${q.close},${q.volume}\n`;
            appendedCount++;
        }
    }

    if (csvContent) {
        if (isNewFile) {
            fs.writeFileSync(filePath, csvContent);
        } else {
            fs.appendFileSync(filePath, csvContent);
        }
    }

    return appendedCount;
};

// Endpoint to list all currently cached standard CSV tickers
router.get('/cache', async (req, res) => {
    try {
        if (!fs.existsSync(DATA_DIR)) {
            return res.json([]);
        }
        const files = fs.readdirSync(DATA_DIR);
        const tickers = files
            .filter(f => f.endsWith('.csv'))
            .map(f => f.replace('.csv', ''));
        res.json(tickers);
    } catch (e) {
        console.error("Error reading cache:", e);
        res.status(500).json({ error: 'Failed to read cache' });
    }
});

// Moving averages and basic stats
const getMean = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

router.post('/load', async (req, res) => {
    try {
        const { tickers } = req.body;
        if (!tickers || !Array.isArray(tickers) || tickers.length === 0) {
            return res.status(400).json({ error: 'Tickers array is required' });
        }

        const yf = new yahooFinance();
        // Fetch Market Data (SPY for RS, VIX/VXV for regime)
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(endDate.getFullYear() - 4); // get 4 years for 252 days RS and Backtest history

        const [spyData, vixData, vxvData] = await Promise.all([
            yf.chart('SPY', { period1: startDate, period2: endDate, interval: '1d' }).catch(() => null),
            yf.chart('^VIX', { period1: startDate, period2: endDate, interval: '1d' }).catch(() => null),
            yf.chart('^VXV', { period1: startDate, period2: endDate, interval: '1d' }).catch(() => null)
        ]);

        const spyMap = {};
        if (spyData && spyData.quotes) {
            spyData.quotes.forEach(q => {
                const dList = typeof q.date === 'string' ? q.date.split('T')[0] : q.date.toISOString().split('T')[0];
                spyMap[dList] = q.close;
            });
        }

        const vxvMap = {};
        if (vxvData && vxvData.quotes) {
            vxvData.quotes.forEach(q => {
                if (q.date && q.close != null) {
                    const dateStr = typeof q.date === 'string' ? q.date.split('T')[0] : q.date.toISOString().split('T')[0];
                    vxvMap[dateStr] = q.close;
                }
            });
        }

        const vixMap = {};
        if (vixData && vixData.quotes) {
            vixData.quotes.forEach((q, i) => {
                if (q.date && q.close != null) {
                    const dateStr = typeof q.date === 'string' ? q.date.split('T')[0] : q.date.toISOString().split('T')[0];
                    const prevClose = i > 0 ? vixData.quotes[i - 1].close : q.open;
                    vixMap[dateStr] = { price: q.close, vxvPrice: vxvMap[dateStr] || q.close, previousClose: prevClose };
                }
            });
        }

        const results = [];

        for (const tic of tickers) {
            const ticker = tic.toUpperCase();
            try {
                const chart = await yf.chart(ticker, { period1: startDate, period2: endDate, interval: '1d' });
                if (!chart || !chart.quotes || chart.quotes.length === 0) continue;

                const validQuotes = chart.quotes.filter(d => d.open != null && d.close != null);

                // Save to CSV
                const newRows = appendToCSV(ticker, validQuotes);

                // Calculate Watchlist Metrics using recent data
                if (validQuotes.length < 50) continue;

                const latest = validQuotes[validQuotes.length - 1];
                const prev = validQuotes[validQuotes.length - 2];
                const currentPrice = latest.close;

                const percentChange = ((currentPrice - prev.close) / prev.close) * 100;

                const closes = validQuotes.map(q => q.close);
                const volumes = validQuotes.map(q => q.volume);
                const highs = validQuotes.map(q => q.high);
                const lows = validQuotes.map(q => q.low);

                const calcEMA = (vals, p) => {
                    const emas = new Array(vals.length).fill(null);
                    if (vals.length < p) return emas;
                    const k = 2 / (p + 1);
                    emas[p - 1] = getMean(vals.slice(0, p));
                    for (let i = p; i < vals.length; i++) emas[i] = (vals[i] * k) + (emas[i - 1] * (1 - k));
                    return emas;
                };

                const calcSMA = (vals, p) => {
                    const smas = new Array(vals.length).fill(null);
                    for (let i = p - 1; i < vals.length; i++) {
                        smas[i] = getMean(vals.slice(i - p + 1, i + 1));
                    }
                    return smas;
                };

                const ema10Array = calcEMA(closes, 10);
                const ema20Array = calcEMA(closes, 20);
                const ema50Array = calcEMA(closes, 50);
                const ema200Array = calcEMA(closes, 200);

                const ema10 = ema10Array[validQuotes.length - 1];
                const ema20 = ema20Array[validQuotes.length - 1];
                const ema50 = ema50Array[validQuotes.length - 1];
                const ema200 = ema200Array[validQuotes.length - 1];
                const ema10Prev5 = ema10Array[Math.max(0, validQuotes.length - 6)] || ema10;
                const ema20Prev5 = ema20Array[Math.max(0, validQuotes.length - 6)] || ema20;
                const ema50Prev5 = ema50Array[Math.max(0, validQuotes.length - 6)] || ema50;

                // ATR Calculation
                const trs = new Array(validQuotes.length).fill(0);
                trs[0] = highs[0] - lows[0];
                for (let i = 1; i < validQuotes.length; i++) {
                    trs[i] = Math.max(
                        highs[i] - lows[i],
                        Math.abs(highs[i] - closes[i - 1]),
                        Math.abs(lows[i] - closes[i - 1])
                    );
                }
                const atrs = new Array(validQuotes.length).fill(null);
                if (validQuotes.length > 14) {
                    atrs[13] = getMean(trs.slice(0, 14));
                    for (let i = 14; i < validQuotes.length; i++) {
                        atrs[i] = (atrs[i - 1] * 13 + trs[i]) / 14;
                    }
                }
                const atr = atrs[validQuotes.length - 1] || 0.01;

                const ema10Dist = (currentPrice - ema10) / atr;
                const ema20Dist = (currentPrice - ema20) / atr;
                const ema50Dist = (currentPrice - ema50) / atr;

                const avgVol50 = getMean(volumes.slice(volumes.length - 50));
                const rVol = avgVol50 > 0 ? latest.volume / avgVol50 : 1.0;

                // Calculate RS (Relative Strength)
                const getRet = (arr, days) => arr.length > days ? (arr[arr.length - 1] - arr[arr.length - 1 - days]) / arr[arr.length - 1 - days] : 0;
                const getSpyRet = (days) => {
                    const idx = validQuotes.length - 1 - days;
                    if (idx < 0) return 0;
                    const oldDateStr = typeof validQuotes[idx].date === 'string' ? validQuotes[idx].date.split('T')[0] : validQuotes[idx].date.toISOString().split('T')[0];
                    const newDateStr = typeof latest.date === 'string' ? latest.date.split('T')[0] : latest.date.toISOString().split('T')[0];
                    const spyOld = spyMap[oldDateStr];
                    const spyNew = spyMap[newDateStr];
                    if (spyOld && spyNew) return (spyNew - spyOld) / spyOld;
                    return 0;
                };

                const sw = (1 + getSpyRet(63)) * 0.4 + (1 + getSpyRet(126)) * 0.2 + (1 + getSpyRet(189)) * 0.2 + (1 + getSpyRet(252)) * 0.2;
                const tRet63 = getRet(closes, 63);
                const tRet126 = getRet(closes, 126);
                const tRet189 = getRet(closes, 189);
                const tRet252 = getRet(closes, 252);
                const stw = (1 + tRet63) * 0.4 + (1 + tRet126) * 0.2 + (1 + tRet189) * 0.2 + (1 + tRet252) * 0.2;

                const rsRating = sw > 0 ? (stw / sw) : 1;

                // RS Delta Line Slope (1 day)
                const getLineSlope = (idx) => {
                    const dateToday = typeof validQuotes[idx].date === 'string' ? validQuotes[idx].date.split('T')[0] : validQuotes[idx].date.toISOString().split('T')[0];
                    const datePrev = typeof validQuotes[idx - 1].date === 'string' ? validQuotes[idx - 1].date.split('T')[0] : validQuotes[idx - 1].date.toISOString().split('T')[0];
                    const rsToday = validQuotes[idx].close / (spyMap[dateToday] || 1);
                    const rsPrev = validQuotes[idx - 1].close / (spyMap[datePrev] || 1);
                    return rsPrev > 0 ? (rsToday - rsPrev) / rsPrev : 0;
                };
                let rsDelta = 0;
                if (validQuotes.length >= 2) {
                    rsDelta = getLineSlope(validQuotes.length - 1);
                }

                // Squeeze Logic (VCP)
                let vcp = "No";
                const sma20 = calcSMA(closes, 20);
                const stdevs = new Array(closes.length).fill(null);
                for (let i = 19; i < closes.length; i++) {
                    const slice = closes.slice(i - 19, i + 1);
                    const mean = sma20[i];
                    const variance = getMean(slice.map(v => Math.pow(v - mean, 2)));
                    stdevs[i] = Math.sqrt(variance);
                }
                const lastIdx = validQuotes.length - 1;
                const bbBasis = sma20[lastIdx];
                const stdev = stdevs[lastIdx];
                const trSma20 = calcSMA(trs, 20);
                const devKC = trSma20[lastIdx];

                if (bbBasis !== null && stdev !== null && devKC !== null) {
                    const bbUpper = bbBasis + (2.0 * stdev);
                    const bbLower = bbBasis - (2.0 * stdev);

                    const kcUppers = [1.0, 1.5, 2.0].map(m => bbBasis + (devKC * m));
                    const kcLowers = [1.0, 1.5, 2.0].map(m => bbBasis - (devKC * m));

                    if (bbUpper > kcUppers[2] || bbLower < kcLowers[2]) vcp = 'No';
                    else if (bbUpper <= kcUppers[0] && bbLower >= kcLowers[0]) vcp = 'High';
                    else if (bbUpper <= kcUppers[1] && bbLower >= kcLowers[1]) vcp = 'Medium';
                    else vcp = 'Low';
                }

                // Dummy VWAP for daily (usually requires intraday)
                const vwap = (latest.high + latest.low + latest.close) / 3;

                // UD Ratio (last 20 days)
                let upVol = 0, downVol = 0;
                for (let j = Math.max(1, validQuotes.length - 20); j < validQuotes.length; j++) {
                    if (closes[j] > closes[j - 1]) upVol += volumes[j];
                    else if (closes[j] < closes[j - 1]) downVol += volumes[j];
                }
                const udRatio = downVol > 0 ? upVol / downVol : 5.0;

                // Percent ADR (High/Low ratio mean over 20 days)
                let ratioSum = 0;
                let ratioCount = 0;
                for (let j = Math.max(0, validQuotes.length - 20); j < validQuotes.length; j++) {
                    const l = lows[j] > 0 ? lows[j] : 1;
                    ratioSum += (highs[j] / l);
                    ratioCount++;
                }
                const avgRatio = ratioCount > 0 ? (ratioSum / ratioCount) : 1;
                const percentADR = (avgRatio - 1) * 100;

                const stockObj = {
                    price: currentPrice,
                    high: latest.high,
                    low: latest.low,
                    percentChange: percentChange,
                    rsRating: rsRating,
                    udRatio: udRatio,
                    percentADR: percentADR,
                    atr: atr,
                    distanceFrom10EMA: ema10Dist,
                    distanceFrom20EMA: ema20Dist,
                    distanceFrom50EMA: ema50Dist,
                    ema10: ema10,
                    ema20: ema20,
                    ema50: ema50,
                    ema200: ema200 || 0,
                    rsLineSlope: rsDelta,
                    ema10Prev5: ema10Prev5,
                    ema20Prev5: ema20Prev5,
                    ema50Prev5: ema50Prev5,
                };

                // Spy Change for today
                const prevSpyStr = typeof prev.date === 'string' ? prev.date.split('T')[0] : prev.date.toISOString().split('T')[0];
                const latestSpyStr = typeof latest.date === 'string' ? latest.date.split('T')[0] : latest.date.toISOString().split('T')[0];
                const prevSpyObj = spyMap[prevSpyStr];
                const latestSpyObj = spyMap[latestSpyStr];
                const spyChange = prevSpyObj && latestSpyObj ? ((latestSpyObj - prevSpyObj) / prevSpyObj) * 100 : 0;

                const vixObj = vixMap[latestSpyStr];

                const quantScore = QuantScorer.calculateScore(stockObj, vixObj, spyChange);

                results.push({
                    ticker,
                    score: Math.round(quantScore),
                    price: currentPrice.toFixed(2),
                    percentChange: percentChange.toFixed(2),
                    rVol: rVol.toFixed(2),
                    pctPred: "0.00", // placeholder
                    rs: rsRating.toFixed(2),
                    rsDelta: (rsDelta * 100).toFixed(2),
                    e10: ema10Dist.toFixed(2),
                    e20: ema20Dist.toFixed(2),
                    e50: ema50Dist.toFixed(2),
                    projVol: (latest.volume / 1000000).toFixed(1),
                    vwap: vwap.toFixed(2),
                    potential: "0.00%", // placeholder
                    sl: (currentPrice * 0.95).toFixed(2), // dummy stop loss
                    pt: (currentPrice * 1.1).toFixed(2), // dummy profit target
                    rr: "2.00",
                    vcp: vcp,
                    newRowsAppended: newRows
                });

            } catch (e) {
                console.error(`Error loading ${tic}:`, e);
            }
        }

        res.json(results);

    } catch (e) {
        console.error("Error in DataLoader API:", e);
        res.status(500).json({ error: 'Internal server error' });
    }
});

export default router;
