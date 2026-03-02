import express from 'express';
import cors from 'cors';
import yahooFinance from 'yahoo-finance2';
import multer from 'multer';
import fs from 'fs';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 3002;

// Setup model saving directory
const MODELS_DIR = path.join(process.cwd(), 'public', 'models');
if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
}

// Setup multer storage engine
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const isMaxExc = req.url.includes('max-excursion');
        if (isMaxExc) {
            const dir = path.join(MODELS_DIR, 'max-excursion');
            if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
            cb(null, dir);
        } else {
            cb(null, MODELS_DIR);
        }
    },
    filename: (req, file, cb) => {
        // Guarantee specific names for consistency
        const isIntra = req.url.includes('intraday');
        const isMaxExc = req.url.includes('max-excursion');
        if (isMaxExc) {
            if (file.fieldname === 'modelJson') cb(null, 'max-excursion-model.json');
            else if (file.fieldname === 'modelWeights') cb(null, 'max-excursion-model.weights.bin');
            else cb(null, file.originalname);
        } else if (file.fieldname === 'modelJson') cb(null, isIntra ? 'intraday-model.json' : 'stock-lstm-model.json');
        else if (file.fieldname === 'modelWeights') cb(null, isIntra ? 'intraday-model.weights.bin' : 'stock-lstm-model.weights.bin');
        else cb(null, file.originalname);
    }
});
const upload = multer({
    storage,
    limits: {
        fieldSize: 50 * 1024 * 1024, // 50MB for metadata JSON text field
        fileSize: 50 * 1024 * 1024   // 50MB for weight bin files
    }
});


app.use(cors());
app.use(express.json());

app.get('/api/yahoo-finance2', async (req, res) => {
    try {
        const { ticker } = req.query;
        if (!ticker) {
            return res.status(400).json({ error: 'Ticker symbol is required' });
        }

        const yf = new yahooFinance();

        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(startDate.getFullYear() - 2);

        const queryOptions = {
            period1: startDate,
            period2: endDate,
            interval: '1d',
        };

        const result = await yf.chart(ticker, queryOptions);

        let formattedData = [];
        if (result && result.quotes) {
            formattedData = result.quotes.map(d => ({
                date: typeof d.date === 'string' ? d.date : d.date.toISOString(),
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close,
                volume: d.volume
            })).filter(d => d.open !== null && d.close !== null);
        }

        res.json(formattedData);
    } catch (error) {
        console.error(`Error fetching data for ${req.query.ticker}:`, error);
        res.status(500).json({ error: 'Failed to fetch historical data' });
    }
});

app.post('/api/save-model', upload.fields([
    { name: 'modelJson', maxCount: 1 },
    { name: 'modelWeights', maxCount: 1 }
]), (req, res) => {
    try {
        if (req.body.metadata) {
            const metaPath = path.join(MODELS_DIR, 'stock-lstm-meta.json');
            fs.writeFileSync(metaPath, req.body.metadata);
        }
        res.json({ success: true, message: "Model successfully saved to project folder!" });
    } catch (e) {
        console.error("Error saving model files", e);
        res.status(500).json({ success: false, error: "Failed to write model to disk" });
    }
});

app.post('/api/save-intraday-model', upload.fields([
    { name: 'modelJson', maxCount: 1 },
    { name: 'modelWeights', maxCount: 1 }
]), (req, res) => {
    try {
        if (req.body.metadata) {
            const metaPath = path.join(MODELS_DIR, 'intraday-meta.json');
            fs.writeFileSync(metaPath, req.body.metadata);
        }
        res.json({ success: true, message: "Intraday Model successfully saved to project folder!" });
    } catch (e) {
        console.error("Error saving intraday model files", e);
        res.status(500).json({ success: false, error: "Failed to write intraday model to disk" });
    }
});

app.post('/api/save-max-excursion-model', upload.fields([
    { name: 'modelJson', maxCount: 1 },
    { name: 'modelWeights', maxCount: 1 }
]), (req, res) => {
    try {
        const dir = path.join(MODELS_DIR, 'max-excursion');
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        if (req.body.metadata) {
            const metaPath = path.join(dir, 'max-excursion-meta.json');
            fs.writeFileSync(metaPath, req.body.metadata);
        }
        res.json({ success: true, message: "Max Excursion Model saved to public/models/max-excursion/" });
    } catch (e) {
        console.error("Error saving max excursion model files", e);
        res.status(500).json({ success: false, error: "Failed to write max excursion model to disk" });
    }
});

// --- INTRADAY API ENDPOINTS ---
const INTRADAY_DIR = path.join(process.cwd(), 'Intraday Stock Price');

app.get('/api/intraday-files', (req, res) => {
    try {
        if (!fs.existsSync(INTRADAY_DIR)) {
            return res.json([]);
        }
        const files = fs.readdirSync(INTRADAY_DIR).filter(f => f.endsWith('.csv'));
        const result = files.map(file => {
            const ticker = file.split('_')[0];
            const content = fs.readFileSync(path.join(INTRADAY_DIR, file), 'utf-8');
            const lines = content.trim().split('\n');
            if (lines.length <= 1) return null;

            const firstLine = lines[1].split(',');
            const lastLine = lines[lines.length - 1].split(',');

            const startDate = firstLine[0] ? firstLine[0].split(' ')[0] : 'Unknown';
            const endDate = lastLine[0] ? lastLine[0].split(' ')[0] : 'Unknown';

            return {
                ticker,
                startDate,
                endDate,
                dataPoints: lines.length - 1
            };
        }).filter(Boolean);

        res.json(result);
    } catch (error) {
        console.error("Error reading intraday files:", error);
        res.status(500).json({ error: 'Failed to read intraday files' });
    }
});

app.get('/api/intraday-data', (req, res) => {
    try {
        const { ticker } = req.query;
        if (!ticker) return res.status(400).json({ error: 'Ticker symbol is required' });

        const filePath = path.join(INTRADAY_DIR, `${ticker}_intraday_5min.csv`);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'Data not found for ticker' });
        }

        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.trim().split('\n');

        const dailyData = {};

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const parts = line.split(',');
            if (parts.length < 6) continue;

            const timestampStr = parts[0]; // "2025-11-18 09:30:00-05:00"
            const open = parseFloat(parts[1]) || 0;
            const high = parseFloat(parts[2]) || 0;
            const low = parseFloat(parts[3]) || 0;
            const close = parseFloat(parts[4]) || 0;
            const volume = parseFloat(parts[5]) || 0;

            const spaceIdx = timestampStr.indexOf(' ');
            if (spaceIdx === -1) continue;

            const datePart = timestampStr.substring(0, spaceIdx);
            const timePart = timestampStr.substring(spaceIdx + 1);

            if (!dailyData[datePart]) {
                dailyData[datePart] = {
                    date: datePart,
                    totalVolume: 0,
                    dayOpen: open,
                    dayHigh: high,
                    dayLow: low,
                    dayClose: close,
                    bars: []
                };
            }

            // Track day-level OHLC
            if (high > dailyData[datePart].dayHigh) dailyData[datePart].dayHigh = high;
            if (low < dailyData[datePart].dayLow) dailyData[datePart].dayLow = low;
            dailyData[datePart].dayClose = close;

            dailyData[datePart].totalVolume += volume;
            dailyData[datePart].bars.push({
                timestamp: timestampStr,
                time: timePart,
                open, high, low, close,
                volume: volume
            });
        }

        // Convert to sorted array
        const result = Object.values(dailyData).sort((a, b) => new Date(a.date) - new Date(b.date));

        res.json(result);
    } catch (error) {
        console.error(`Error processing intraday data for ${req.query.ticker}:`, error);
        res.status(500).json({ error: 'Failed to process intraday data' });
    }
});

// --- INTRADAY FEATURES ENDPOINT ---
// Computes daily technical indicators by cross-referencing Yahoo Finance daily data
app.get('/api/intraday-features', async (req, res) => {
    try {
        const { ticker } = req.query;
        if (!ticker) return res.status(400).json({ error: 'Ticker symbol is required' });

        const yf = new yahooFinance();
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(startDate.getFullYear() - 2);

        const result = await yf.chart(ticker, {
            period1: startDate,
            period2: endDate,
            interval: '1d',
        });

        if (!result || !result.quotes || result.quotes.length < 50) {
            return res.json([]);
        }

        const quotes = result.quotes
            .map(d => ({
                date: typeof d.date === 'string' ? d.date.split('T')[0] : d.date.toISOString().split('T')[0],
                open: d.open, high: d.high, low: d.low, close: d.close, volume: d.volume
            }))
            .filter(d => d.open !== null && d.close !== null);

        // Helper: EMA calculation
        const calcEma = (values, period) => {
            const emas = [];
            const k = 2 / (period + 1);
            emas[0] = values[0];
            for (let i = 1; i < values.length; i++) {
                emas[i] = values[i] * k + emas[i - 1] * (1 - k);
            }
            return emas;
        };

        const closes = quotes.map(q => q.close);
        const ema10 = calcEma(closes, 10);
        const ema20 = calcEma(closes, 20);
        const ema50 = calcEma(closes, 50);

        const features = [];
        for (let i = 50; i < quotes.length; i++) {
            const q = quotes[i];
            const prevQ = quotes[i - 1];

            // ADR(20): average of (high - low) over past 20 days
            let adrSum = 0;
            for (let j = 1; j <= 20; j++) adrSum += (quotes[i - j].high - quotes[i - j].low);
            const adr20 = adrSum / 20;

            // ATR(14): average of true range over past 14 days
            let atrSum = 0;
            for (let j = 1; j <= 14; j++) {
                const prev = quotes[i - j];
                const curr = quotes[i - j + 1];
                const tr = Math.max(
                    curr.high - curr.low,
                    Math.abs(curr.high - prev.close),
                    Math.abs(curr.low - prev.close)
                );
                atrSum += tr;
            }
            const atr14 = atrSum / 14;

            // 50-day average volume
            let volSum = 0;
            for (let j = 1; j <= 50; j++) volSum += quotes[i - j].volume;
            const avgVol50 = volSum / 50;

            // % change from prev day
            const prevCloseChange = prevQ.close > 0 ? ((q.open - prevQ.close) / prevQ.close) * 100 : 0;

            // ATR distance from EMAs (in ATR units)
            const atrDistEma10 = atr14 > 0 ? (q.open - ema10[i]) / atr14 : 0;
            const atrDistEma20 = atr14 > 0 ? (q.open - ema20[i]) / atr14 : 0;
            const atrDistEma50 = atr14 > 0 ? (q.open - ema50[i]) / atr14 : 0;

            features.push({
                date: q.date,
                open: q.open,
                prevClose: prevQ.close,
                prevCloseChange,
                adr20,
                atr14,
                avgVol50,
                ema10: ema10[i],
                ema20: ema20[i],
                ema50: ema50[i],
                atrDistEma10,
                atrDistEma20,
                atrDistEma50
            });
        }

        res.json(features);
    } catch (error) {
        console.error(`Error computing features for ${req.query.ticker}:`, error);
        res.status(500).json({ error: 'Failed to compute intraday features' });
    }
});

// --- MARKET REGIME ENDPOINT ---
app.get('/api/market-regime', async (req, res) => {
    try {
        const yf = new yahooFinance();
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(startDate.getFullYear() - 3);

        const queryOptions = {
            period1: startDate,
            period2: endDate,
            interval: '1d',
        };

        const [spyResult, vixResult] = await Promise.all([
            yf.chart('SPY', queryOptions),
            yf.chart('^VIX', queryOptions),
        ]);

        const formatQuotes = (result) => {
            if (!result || !result.quotes) return [];
            return result.quotes
                .map(d => ({
                    date: typeof d.date === 'string' ? d.date.split('T')[0] : d.date.toISOString().split('T')[0],
                    open: d.open,
                    high: d.high,
                    low: d.low,
                    close: d.close,
                    volume: d.volume
                }))
                .filter(d => d.close !== null);
        };

        const spyQuotes = formatQuotes(spyResult);
        const vixQuotes = formatQuotes(vixResult);

        // Build VIX lookup by date
        const vixMap = {};
        for (const q of vixQuotes) {
            vixMap[q.date] = q;
        }

        // Align by date
        const aligned = [];
        let prevPrice = null;
        for (const spy of spyQuotes) {
            const vix = vixMap[spy.date];
            if (!vix) continue;

            const ret = prevPrice !== null ? (spy.close - prevPrice) / prevPrice : 0;

            aligned.push({
                day: spy.date,
                price: spy.close,
                vix: vix.close,
                vixHigh: vix.high || vix.close,
                vixLow: vix.low || vix.close,
                ret
            });

            prevPrice = spy.close;
        }

        res.json(aligned);
    } catch (error) {
        console.error('Error fetching market regime data:', error);
        res.status(500).json({ error: 'Failed to fetch market regime data' });
    }
});

app.listen(PORT, () => {
    console.log(`Backend server is running on http://localhost:${PORT}`);
});
