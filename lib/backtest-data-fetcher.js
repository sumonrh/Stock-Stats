import fs from 'fs';
import path from 'path';
import yahooFinance from 'yahoo-finance2';

const DATA_DIR = path.join(process.cwd(), 'Intraday Stock Price');


// Ensure data dir exists
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}

/**
 * Fetch data from Polygon.io and merge with existing CSV
 */
export async function fetchAndAppendPolygonData(ticker, apiKey, options = {}) {
    const { 
        days = 100, 
        isDaily = false,
        from = null,
        to = null
    } = options;

    const multiplier = isDaily ? 1 : 5;
    const timespan = isDaily ? 'day' : 'minute';
    
    // Calculate dates
    const endDate = to || new Date().toISOString().split('T')[0];
    let startDate = from;
    if (!startDate) {
        const d = new Date();
        d.setDate(d.getDate() - days);
        startDate = d.toISOString().split('T')[0];
    }

    const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/${multiplier}/${timespan}/${startDate}/${endDate}?adjusted=true&sort=asc&limit=50000&apiKey=${apiKey}`;

    console.log(`Fetching Polygon data for ${ticker}: ${url}`);
    
    const response = await fetch(url);
    const result = await response.json();

    if (!result.results || result.results.length === 0) {
        throw new Error(result.error || result.status || "No data found from Polygon");
    }

    // Convert Polygon results to our CSV format
    // Format: timestamp, open, high, low, close, volume (only Regular Trading Hours)
    const newData = [];
    for (const r of result.results) {
        const dateObj = new Date(r.t);
        // Convert strict EST
        const dateString = dateObj.toLocaleString('sv-SE', { timeZone: 'America/New_York' }); 
        // e.g. "2024-05-10 16:00:00"
        
        if (isDaily) {
            newData.push({
                timestamp: dateString,
                open: r.o, high: r.h, low: r.l, close: r.c, volume: r.v
            });
            continue;
        }

        // Intraday Filtering: Only standard market hours (09:30:00 to 16:00:00)
        const timePart = dateString.split(' ')[1]; // "HH:MM:SS"
        if (timePart >= "09:30:00" && timePart < "16:00:00") {
            newData.push({
                timestamp: dateString,
                open: r.o,
                high: r.h,
                low: r.l,
                close: r.c,
                volume: r.v
            });
        }
    }

    const fileName = `${ticker}_intraday_5min.csv`;
    const filePath = path.join(DATA_DIR, fileName);

    let finalData = newData;

    if (fs.existsSync(filePath)) {
        // Read existing data
        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.trim().split('\n');
        if (lines.length > 1) {
            const existingData = lines.slice(1).map(line => {
                const parts = line.split(',');
                return {
                    timestamp: parts[0],
                    open: parseFloat(parts[1]),
                    high: parseFloat(parts[2]),
                    low: parseFloat(parts[3]),
                    close: parseFloat(parts[4]),
                    volume: parseFloat(parts[5])
                };
            });

            // Merge and deduplicate by timestamp
            const map = new Map();
            existingData.forEach(d => map.set(d.timestamp, d));
            newData.forEach(d => map.set(d.timestamp, d));
            
            finalData = Array.from(map.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
        }
    }

    // Write to CSV
    const header = "timestamp,open,high,low,close,volume\n";
    const csvContent = header + finalData.map(d => `${d.timestamp},${d.open},${d.high},${d.low},${d.close},${d.volume}`).join('\n');
    
    fs.writeFileSync(filePath, csvContent);

    return { 
        success: true, 
        ticker, 
        count: finalData.length, 
        added: newData.length,
        startDate: finalData[0].timestamp.split(' ')[0],
        endDate: finalData[finalData.length - 1].timestamp.split(' ')[0]
    };
}

/**
 * Fetch Daily data for EMAs using Yahoo Finance
 */
export async function fetchDailyMetrics(ticker) {
    try {
        const yf = new yahooFinance();
        // Set suppressNotices to avoid polluting console with deprecation warnings (only if available)
        if (yf.setGlobalConfig) {
            yf.setGlobalConfig({ suppressNotices: ['ripHistorical'] });
        }
        
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(startDate.getFullYear() - 3); // 3 years for proper 200 EMA warmup

        console.log(`[DailyMetrics] Requesting 3 years for ${ticker} via yf.chart()...`);
        const result = await yf.chart(ticker, {
            period1: startDate,
            period2: endDate,
            interval: '1d'
        });

        if (!result || !result.quotes || result.quotes.length === 0) {
            console.error(`[DailyMetrics] No daily data found for ${ticker}`);
            return null;
        }

        const quotes = result.quotes.filter(q => q.close != null).map(q => ({
            date: q.date.toISOString().split('T')[0],
            open: q.open,
            high: q.high,
            low: q.low,
            close: q.close,
            volume: q.volume
        }));

        // Calculate Indicators
        const closes = quotes.map(q => q.close);
        const highs = quotes.map(q => q.high);
        const lows = quotes.map(q => q.low);

        const ema10 = calculateEMA(closes, 10);
        const ema20 = calculateEMA(closes, 20);
        const ema50 = calculateEMA(closes, 50);
        const ema200 = calculateEMA(closes, 200);
        const atr14 = calculateATR(highs, lows, closes, 14);

        // Map back to dates — only include dates where all EMAs are valid (200-EMA needs 200 bars to warm up)
        const metricsByDate = {};
        for (let i = 0; i < quotes.length; i++) {
            const e200 = ema200[i];
            if (e200 === null) continue; // skip warmup period
            metricsByDate[quotes[i].date] = {
                price: quotes[i].close,
                ema10: ema10[i],
                ema20: ema20[i],
                ema50: ema50[i],
                ema200: e200,
                atr14: atr14[i]
            };
        }

        console.log(`[DailyMetrics] ${ticker}: ${Object.keys(metricsByDate).length} valid days (first: ${Object.keys(metricsByDate)[0]})`);
        return metricsByDate;
    } catch (e) {
        console.error(`Error fetching daily metrics for ${ticker}:`, e);
        return null;
    }
}

// EMA helper — uses proper SMA seed for the first `period` bars
function calculateEMA(values, period) {
    if (values.length < period) {
        // Not enough data — return best-effort values
        return values.map((_, i) => {
            const slice = values.slice(0, i + 1);
            return slice.reduce((a, b) => a + b, 0) / slice.length;
        });
    }

    const k = 2 / (period + 1);
    // Seed with SMA of first `period` values
    const seed = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
    const result = new Array(period - 1).fill(null); // null for warmup period
    result.push(seed);

    for (let i = period; i < values.length; i++) {
        result.push((values[i] * k) + (result[result.length - 1] * (1 - k)));
    }
    return result;
}

// ATR helper
function calculateATR(highs, lows, closes, period) {
    const tr = [highs[0] - lows[0]];
    for (let i = 1; i < highs.length; i++) {
        const h_l = highs[i] - lows[i];
        const h_pc = Math.abs(highs[i] - closes[i - 1]);
        const l_pc = Math.abs(lows[i] - closes[i - 1]);
        tr.push(Math.max(h_l, h_pc, l_pc));
    }

    // 14 SMA of TR
    const atr = new Array(tr.length).fill(null);
    for (let i = period - 1; i < tr.length; i++) {
        const slice = tr.slice(i - period + 1, i + 1);
        atr[i] = slice.reduce((a, b) => a + b, 0) / period;
    }
    // For values before period, just use first valid ATR
    const firstValid = atr[period - 1];
    for (let i = 0; i < period - 1; i++) atr[i] = firstValid;

    return atr;
}
