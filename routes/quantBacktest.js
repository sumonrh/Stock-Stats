import express from 'express';
import yahooFinance from 'yahoo-finance2';
import { QuantScorer } from './quantScorer.js';

// We need a subset of logic from stock-scorer's calculateMetrics to compute over a history array.
// For simplicity, we'll re-implement the rolling metric logic directly here specifically for 1d charts.

const router = express.Router();

// Helper functions (similar to lib.js from stock-scorer)
const getMean = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

const formatNumber = (num, digits = 2) => {
    if (num === null || num === undefined || !isFinite(num)) {
        return null;
    }
    return Number(num.toFixed(digits));
};

const calculateEMA = (values, period) => {
    const emas = new Array(values.length).fill(null);
    if (values.length < period) return emas;
    const k = 2 / (period + 1);
    emas[period - 1] = getMean(values.slice(0, period));
    for (let i = period; i < values.length; i++) {
        emas[i] = (values[i] * k) + (emas[i - 1] * (1 - k));
    }
    return emas;
};

const calculateSMA = (values, period) => {
    const smas = new Array(values.length).fill(null);
    for (let i = period - 1; i < values.length; i++) {
        smas[i] = getMean(values.slice(i - period + 1, i + 1));
    }
    return smas;
};

router.post('/run', async (req, res) => {
    try {
        const { tickers } = req.body;
        if (!tickers || !Array.isArray(tickers) || tickers.length === 0) {
            return res.status(400).json({ error: 'Tickers array is required' });
        }

        console.log(`[Quant Backtest] Running backtest for ${tickers.length} tickers...`);

        // Need 4 years of data
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(endDate.getFullYear() - 4);

        const queryOptions = { period1: startDate, period2: endDate, interval: '1d' };

        // 1. Fetch SPY for RS Rating and VIX for weights
        const yf = new yahooFinance();
        const [spyData, vixData, vxvData] = await Promise.all([
            yf.chart('SPY', queryOptions).catch(() => null),
            yf.chart('^VIX', queryOptions).catch(() => null),
            yf.chart('^VXV', queryOptions).catch(() => null)
        ]);

        if (!spyData || !spyData.quotes || spyData.quotes.length === 0) {
            return res.status(500).json({ error: 'Failed to fetch SPY baseline data' });
        }

        const spyMap = {};
        spyData.quotes.forEach(q => {
            if (q.date && q.close != null) {
                const dateStr = typeof q.date === 'string' ? q.date.split('T')[0] : q.date.toISOString().split('T')[0];
                spyMap[dateStr] = q.close;
            }
        });

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

        const allResults = [];

        // 2. Fetch and process each ticker
        for (const ticker of tickers) {
            try {
                const chart = await yf.chart(ticker, queryOptions);
                if (!chart || !chart.quotes || chart.quotes.length < 252) continue; // Need at least 1 year of data

                const quotes = chart.quotes
                    .map(d => ({
                        dateStr: typeof d.date === 'string' ? d.date.split('T')[0] : d.date.toISOString().split('T')[0],
                        open: d.open,
                        high: d.high,
                        low: d.low,
                        close: d.close,
                        volume: d.volume
                    }))
                    .filter(d => d.open != null && d.close != null);

                if (quotes.length < 252) continue;

                const closes = quotes.map(q => q.close);
                const highs = quotes.map(q => q.high);
                const lows = quotes.map(q => q.low);
                const volumes = quotes.map(q => q.volume);

                const ema10 = calculateEMA(closes, 10);
                const ema20 = calculateEMA(closes, 20);
                const ema50 = calculateEMA(closes, 50);
                const ema200 = calculateEMA(closes, 200);

                // For Squeeze
                const sma20 = calculateSMA(closes, 20);

                // ATR(14) calculation
                const atrs = new Array(quotes.length).fill(null);
                const trs = new Array(quotes.length).fill(0);
                trs[0] = highs[0] - lows[0];
                for (let i = 1; i < quotes.length; i++) {
                    trs[i] = Math.max(
                        highs[i] - lows[i],
                        Math.abs(highs[i] - closes[i - 1]),
                        Math.abs(lows[i] - closes[i - 1])
                    );
                }
                atrs[13] = getMean(trs.slice(0, 14));
                for (let i = 14; i < quotes.length; i++) {
                    atrs[i] = (atrs[i - 1] * 13 + trs[i]) / 14;
                }

                // Standard deviation 20
                const std20 = new Array(quotes.length).fill(null);
                for (let i = 19; i < quotes.length; i++) {
                    const slice = closes.slice(i - 19, i + 1);
                    const mean = sma20[i];
                    const variance = getMean(slice.map(v => Math.pow(v - mean, 2)));
                    std20[i] = Math.sqrt(variance);
                }

                // Evaluate over valid history (skip first 252 days to allow for 12-month RS Rating)
                for (let i = 252; i < quotes.length; i++) { // Loop until the end, handle fwd returns inside
                    const q = quotes[i];

                    // --- Forward Returns ---
                    const target1D = (i + 1 < quotes.length) ? quotes[i + 1].close : null;
                    const target1W = (i + 5 < quotes.length) ? quotes[i + 5].close : null;
                    const target2W = (i + 10 < quotes.length) ? quotes[i + 10].close : null;
                    const target1M = (i + 21 < quotes.length) ? quotes[i + 21].close : null;

                    const ret1D = (target1D !== null && q.close) ? ((target1D - q.close) / q.close) * 100 : null;
                    const ret1W = (target1W !== null && q.close) ? ((target1W - q.close) / q.close) * 100 : null;
                    const ret2W = (target2W !== null && q.close) ? ((target2W - q.close) / q.close) * 100 : null;
                    const ret1M = (target1M !== null && q.close) ? ((target1M - q.close) / q.close) * 100 : null;
                    
                    if (ret1D === null && ret1W === null && ret2W === null && ret1M === null) {
                        continue;
                    }

                    // --- Features ---
                    const currentPrice = q.close;
                    const prevClose = quotes[i - 1].close;
                    const percentChange = ((currentPrice - prevClose) / prevClose) * 100;

                    const atr = atrs[i] || 0.01;
                    const dist10 = (currentPrice - ema10[i]) / atr;
                    const dist20 = (currentPrice - ema20[i]) / atr;
                    const dist50 = (currentPrice - ema50[i]) / atr;

                    let adrDollarSum = 0;
                    for (let j = 0; j < 20; j++) {
                        adrDollarSum += (highs[i - j] - lows[i - j]);
                    }
                    const adrDollarAverage = adrDollarSum / 20;

                    // The percent change relative to the stock's average daily movement
                    const isRedDay = currentPrice < q.open;
                    const referencePrice = isRedDay ? q.high : q.low;
                    // For red days, currentPrice - high will be negative. For green/flat days, currentPrice - low will be positive.
                    const priceDollarChange = currentPrice - referencePrice;
                    const priceChangeOverAdr = adrDollarAverage > 0 ? (priceDollarChange / adrDollarAverage) : 0;

                    // RVol (50 days)
                    const avgVol50 = getMean(volumes.slice(i - 49, i + 1));
                    const currentVol = volumes[i];
                    const rVol = avgVol50 > 0 ? currentVol / avgVol50 : 1.0;

                    // Episodic Pivot Power (Volume * Move Severity)
                    const episodicPivotPower = rVol * priceChangeOverAdr;

                    // Squeeze (VCP)
                    let vcp = 0; // 0 = None, 1 = Low, 2 = Mid, 3 = High
                    if (sma20[i] && std20[i] && ema20[i] && atrs[i]) {
                        const bbUpper = sma20[i] + (2 * std20[i]);
                        const bbLower = sma20[i] - (2 * std20[i]);
                        const devKC = (trs[i] + trs[i - 1] + trs[i - 2]) / 3; // simplified KC TR
                        const kcUpper1 = ema20[i] + (1.0 * atrs[i]);
                        const kcLower1 = ema20[i] - (1.0 * atrs[i]);
                        const kcUpper15 = ema20[i] + (1.5 * atrs[i]);
                        const kcLower15 = ema20[i] - (1.5 * atrs[i]);
                        const kcUpper2 = ema20[i] + (2.0 * atrs[i]);
                        const kcLower2 = ema20[i] - (2.0 * atrs[i]);

                        if (bbUpper <= kcUpper1 && bbLower >= kcLower1) vcp = 3; // High
                        else if (bbUpper <= kcUpper15 && bbLower >= kcLower15) vcp = 2; // Mid
                        else if (bbUpper <= kcUpper2 && bbLower >= kcLower2) vcp = 1; // Low
                    }

                    // RS Rating
                    const getSpyRet = (days) => {
                        const dateOld = quotes[i - days].dateStr;
                        const spyOld = spyMap[dateOld] || spyMap[quotes[i].dateStr];
                        const spyNow = spyMap[q.dateStr] || spyMap[quotes[i].dateStr];
                        return spyOld ? ((spyNow - spyOld) / spyOld) * 100 : 0;
                    };
                    const getStockRet = (days) => {
                        const oldPrice = quotes[i - days].close;
                        return oldPrice ? ((currentPrice - oldPrice) / oldPrice) * 100 : 0;
                    };

                    const spyWeights = (1 + getSpyRet(63) / 100) * 0.4 + (1 + getSpyRet(126) / 100) * 0.2 + (1 + getSpyRet(189) / 100) * 0.2 + (1 + getSpyRet(252) / 100) * 0.2;
                    const stockWeights = (1 + getStockRet(63) / 100) * 0.4 + (1 + getStockRet(126) / 100) * 0.2 + (1 + getStockRet(189) / 100) * 0.2 + (1 + getStockRet(252) / 100) * 0.2;

                    const rsRating = spyWeights > 0 ? stockWeights / spyWeights : 1.0;

                    // RS Delta Line Slope (1 day)
                    const getLineSlope = (idx) => {
                        const dateToday = quotes[idx].dateStr;
                        const datePrev = quotes[idx - 1].dateStr;
                        const rsToday = quotes[idx].close / (spyMap[dateToday] || 1);
                        const rsPrev = quotes[idx - 1].close / (spyMap[datePrev] || 1);
                        return rsPrev > 0 ? (rsToday - rsPrev) / rsPrev : 0;
                    };
                    const rsDelta = getLineSlope(i);

                    // UD Ratio (last 20 days)
                    let upVol = 0, downVol = 0;
                    for (let j = Math.max(1, i - 19); j <= i; j++) {
                        if (closes[j] > closes[j - 1]) upVol += volumes[j];
                        else if (closes[j] < closes[j - 1]) downVol += volumes[j];
                    }
                    const udRatio = downVol > 0 ? upVol / downVol : 5.0;

                    // Percent ADR
                    let ratioSum = 0;
                    let ratioCount = 0;
                    for (let j = Math.max(0, i - 19); j <= i; j++) {
                        const l = lows[j] > 0 ? lows[j] : 1;
                        ratioSum += (highs[j] / l);
                        ratioCount++;
                    }
                    const avgRatio = ratioCount > 0 ? (ratioSum / ratioCount) : 1;
                    const percentADR = (avgRatio - 1) * 100;

                    const stockObj = {
                        price: currentPrice,
                        high: q.high,
                        low: q.low,
                        percentChange,
                        rsRating,
                        udRatio,
                        percentADR,
                        atr,
                        distanceFrom10EMA: dist10,
                        distanceFrom20EMA: dist20,
                        distanceFrom50EMA: dist50,
                        ema10: ema10[i],
                        ema20: ema20[i],
                        ema50: ema50[i],
                        ema200: ema200[i] || 0,
                        rsLineSlope: rsDelta,
                        ema10Prev5: ema10[Math.max(0, i - 5)] || ema10[i],
                        ema20Prev5: ema20[Math.max(0, i - 5)] || ema20[i],
                        ema50Prev5: ema50[Math.max(0, i - 5)] || ema50[i],
                    };

                    const spyPrevItem = spyMap[quotes[i - 1].dateStr];
                    const spyItem = spyMap[q.dateStr];
                    const spyChg = (spyPrevItem && spyItem) ? ((spyItem - spyPrevItem) / spyPrevItem) * 100 : 0;

                    const vixItem = vixMap[q.dateStr];

                    const quantScore = QuantScorer.calculateScore(stockObj, vixItem, spyChg);

                    allResults.push({
                        date: q.dateStr,
                        ticker,
                        quantScore: formatNumber(quantScore, 0),
                        rsDelta: formatNumber(rsDelta * 100),
                        rs: formatNumber(rsRating),
                        vcp,
                        rVol: formatNumber(rVol),
                        priceChangeOverAdr: formatNumber(priceChangeOverAdr),
                        episodicPivotPower: formatNumber(episodicPivotPower),
                        ret1D: formatNumber(ret1D),
                        ret1W: formatNumber(ret1W),
                        ret2W: formatNumber(ret2W),
                        ret1M: formatNumber(ret1M),
                        ema10DistAtr: formatNumber(dist10),
                        ema20DistAtr: formatNumber(dist20)
                    });
                }
            } catch (err) {
                console.error(`Error processing ${ticker}:`, err.message);
            }
        }

        res.json(allResults);

    } catch (e) {
        console.error("Quant Backtest Error:", e);
        res.status(500).json({ error: e.stack || e.message || 'Internal Server Error' });
    }
});

export default router;
