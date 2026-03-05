import express from 'express';
import yahooFinance from 'yahoo-finance2';
import fs from 'fs';
import path from 'path';

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
        // Fetch SPY for RS
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(endDate.getFullYear() - 4); // get 4 years for 252 days RS and Backtest history

        const spyData = await yf.chart('SPY', { period1: startDate, period2: endDate, interval: '1d' }).catch(() => null);
        const spyMap = {};
        if (spyData && spyData.quotes) {
            spyData.quotes.forEach(q => {
                const dList = typeof q.date === 'string' ? q.date.split('T')[0] : q.date.toISOString().split('T')[0];
                spyMap[dList] = q.close;
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

                const calcEMA = (vals, p) => {
                    const k = 2 / (p + 1);
                    let ema = getMean(vals.slice(0, p));
                    for (let i = p; i < vals.length; i++) ema = (vals[i] * k) + (ema * (1 - k));
                    return ema;
                };

                const ema10 = calcEMA(closes, 10);
                const ema20 = calcEMA(closes, 20);
                const ema50 = calcEMA(closes, 50);

                const ema10Dist = ((currentPrice - ema10) / ema10) * 100;
                const ema20Dist = ((currentPrice - ema20) / ema20) * 100;
                const ema50Dist = ((currentPrice - ema50) / ema50) * 100;

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

                // Simple Quant Score Approximation
                let quantScore = 50 + (percentChange > 0 ? 5 : -5);
                quantScore += (rsRating - 1.0) * 20;
                quantScore = Math.max(0, Math.min(100, quantScore));

                // Dummy VWAP for daily (usually requires intraday)
                const vwap = (latest.high + latest.low + latest.close) / 3;

                results.push({
                    ticker,
                    score: Math.round(quantScore),
                    price: currentPrice.toFixed(2),
                    percentChange: percentChange.toFixed(2),
                    rVol: rVol.toFixed(2),
                    pctPred: "0.00", // placeholder
                    rs: rsRating.toFixed(2),
                    rsDelta: "0.00", // placeholder for slope
                    e10: ema10Dist.toFixed(2),
                    e20: ema20Dist.toFixed(2),
                    e50: ema50Dist.toFixed(2),
                    projVol: (latest.volume / 1000000).toFixed(1),
                    vwap: vwap.toFixed(2),
                    potential: "0.00%", // placeholder
                    sl: (currentPrice * 0.95).toFixed(2), // dummy stop loss
                    pt: (currentPrice * 1.1).toFixed(2), // dummy profit target
                    rr: "2.00",
                    vcp: "No",
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
