import express from 'express';
import cors from 'cors';
import yahooFinance from 'yahoo-finance2';

const app = express();
const PORT = process.env.PORT || 3001;

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

app.listen(PORT, () => {
    console.log(`Backend server is running on http://localhost:${PORT}`);
});
