import express from 'express';
import cors from 'cors';
import yahooFinance from 'yahoo-finance2';
import multer from 'multer';
import fs from 'fs';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 3001;

// Setup model saving directory
const MODELS_DIR = path.join(process.cwd(), 'public', 'models');
if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
}

// Setup multer storage engine
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, MODELS_DIR);
    },
    filename: (req, file, cb) => {
        // Guarantee specific names for consistency
        if (file.fieldname === 'modelJson') cb(null, 'stock-lstm-model.json');
        else if (file.fieldname === 'modelWeights') cb(null, 'stock-lstm-model.weights.bin');
        else cb(null, file.originalname);
    }
});
const upload = multer({ storage });


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

app.listen(PORT, () => {
    console.log(`Backend server is running on http://localhost:${PORT}`);
});
