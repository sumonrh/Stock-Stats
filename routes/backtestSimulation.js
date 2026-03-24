import express from 'express';
import fs from 'fs';
import path from 'path';
import { runBacktest, getAvailableStocks, saveBacktestResults, optimizeBacktest } from '../lib/backtest-engine.js';
import { fetchAndAppendPolygonData } from '../lib/backtest-data-fetcher.js';


const router = express.Router();

// GET all available stocks for backtesting
router.get('/stocks', (req, res) => {
    try {
        const stocks = getAvailableStocks();
        res.json({ stocks });
    } catch (error) {
        console.error("Error getting available stocks:", error);
        res.status(500).json({ error: 'Failed to get available stocks' });
    }
});

// POST run backtest (streams progress)
router.post('/run', async (req, res) => {
    try {
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Transfer-Encoding', 'chunked');
        res.setHeader('Connection', 'keep-alive');

        await runBacktest(req.body, res);
        res.end();
    } catch (error) {
        console.error("Backtest run error:", error);
        if (!res.headersSent) {
            res.status(500).json({ error: 'Backtest failed' });
        } else {
            res.write(JSON.stringify({ type: 'error', message: error.message }) + '\n');
            res.end();
        }
    }
});

// POST save backtest results
router.post('/save', async (req, res) => {
    try {
        const { results, saveTarget } = req.body;
        const result = await saveBacktestResults(results, saveTarget);
        res.json(result);
    } catch (error) {
        console.error("Error saving backtest results:", error);
        res.status(500).json({ error: 'Failed to save results' });
    }
});

// POST optimize backtest (streams progress)
router.post('/optimize', async (req, res) => {
    try {
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Transfer-Encoding', 'chunked');
        res.setHeader('Connection', 'keep-alive');

        await optimizeBacktest(req.body, res);
        res.end();
    } catch (error) {
        console.error("Backtest optimization error:", error);
        if (!res.headersSent) {
            res.status(500).json({ error: 'Optimization failed' });
        } else {
            res.write(JSON.stringify({ type: 'error', message: error.message }) + '\n');
            res.end();
        }
    }
});

// GET intraday chart data for a ticker
router.get('/chart-intraday/:ticker', (req, res) => {
    try {
        const ticker = req.params.ticker;
        const daysToLoad = parseInt(req.query.days) || 5;
        const intradayDir = path.join(process.cwd(), 'Intraday Stock Price');
        const filePath = path.join(intradayDir, `${ticker}_intraday_5min.csv`);

        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'No data found for ticker' });
        }

        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.trim().split('\n');
        
        const data = [];
        const dailyGroups = {};

        // Group by day for VWAP calculation
        for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split(',');
            if (parts.length < 6) continue;
            const timestamp = parts[0].trim();
            const date = timestamp.split(' ')[0];
            if (!dailyGroups[date]) dailyGroups[date] = [];
            dailyGroups[date].push({
                timestamp,
                date,
                open: parseFloat(parts[1]),
                high: parseFloat(parts[2]),
                low: parseFloat(parts[3]),
                close: parseFloat(parts[4]),
                volume: parseFloat(parts[5])
            });
        }

        const sortedDates = Object.keys(dailyGroups).sort().slice(-daysToLoad);
        
        for (const date of sortedDates) {
            const bars = dailyGroups[date];
            let cumTPV = 0;
            let cumVol = 0;
            
            for (const bar of bars) {
                // VWAP reset daily
                const tp = (bar.high + bar.low + bar.close) / 3;
                cumTPV += tp * bar.volume;
                cumVol += bar.volume;
                
                // LightweightCharts expects seconds timestamp
                const time = Math.floor(new Date(bar.timestamp).getTime() / 1000);
                
                data.push({
                    time,
                    date: bar.date,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume,
                    vwap: cumVol > 0 ? cumTPV / cumVol : bar.close
                });
            }
        }

        res.json(data);
    } catch (error) {
        console.error("Error reading intraday chart data:", error);
        res.status(500).json({ error: 'Failed to process chart data' });
    }
});

// POST fetch ticker data using Polygon API (Task 1)
router.post('/fetch-ticker', async (req, res) => {
    try {
        const { ticker, apiKey, days, isDaily } = req.body;
        
        const finalApiKey = apiKey || process.env.POLYGON_API_KEY;
        
        if (!finalApiKey) {
            return res.status(400).json({ error: 'Polygon API Key is required. Set POLYGON_API_KEY env var or provide it in the request.' });
        }
        
        if (!ticker) {
            return res.status(400).json({ error: 'Ticker is required.' });
        }

        const result = await fetchAndAppendPolygonData(ticker, finalApiKey, { days, isDaily });
        res.json(result);
    } catch (error) {
        console.error("Error fetching Polygon data:", error);
        res.status(500).json({ error: error.message || 'Failed to fetch ticker data' });
    }
});


export default router;
