# Stock Stats & ML Predictors

## Overview

**Stock Stats & ML Predictors** is a data-driven trading analytics platform that combines statistical analysis with neural network predictions to help traders quantify a stock's expected intraday price behavior. The app provides three specialized analysis tabs, each targeting a different aspect of stock price/volume analysis.

Built with React + Vite on the frontend and Node.js/Express on the backend, the app fetches live market data via `yahoo-finance2` and processes intraday 5-minute OHLCV data from local CSV files for 20 actively traded stocks.

---

## Three Analysis Tabs

### Tab 1: Daily RVol vs Excursion

**Purpose:** Analyze the historical statistical relationship between a stock's **Relative Volume (RVol)** and its **Maximum Intraday Price Excursion** (the absolute maximum move from the daily open, expressed as a multiple of ADR).

**Key Features:**
- **Interactive Scatter Plot** — Each dot represents one trading day. X-axis = RVol (volume relative to 50-day average), Y-axis = max excursion from open in ADR multiples.
- **Best-Fit Regression Lines** — Automatically tests Linear, Exponential, Quadratic, and Quartic regressions and selects the model that maximizes R².
- **Probability Distribution Overlays** — Marginal density curves (Plotly.js) show the probability distribution of RVol events and excursion events independently.
- **Profitable Zone Highlighting** — Highlights the quadrant where RVol > 1.5x AND excursion > 1.0 ADR — the "sweet spot" for momentum trades.
- **Statistical Summary Table** — Aggregates data by RVol bucket ranges (0–1x, 1–2x, 2–3x, 3x+) showing average excursion, green zone rate, and sample size.
- **LSTM Neural Network** — Train a TensorFlow.js LSTM model directly in the browser to predict max excursion from historical feature sequences. Supports model export/import and saving to the project folder.
- **Dynamic Controls** — Switch tickers, adjust RVol period (10/20/50 day), ADR period, and filter datasets in real time.

**Use Case:** "Given today's RVol for NVDA is 2.5x, what's the statistically expected max move from open?"

---

### Tab 2: Intraday Volume Predictor

**Purpose:** Predict a stock's **total end-of-day volume** using only the first N minutes (configurable, default 30 min) of trading activity after market open.

**Key Features:**
- **LSTM Neural Network** — Trains on sequences of 5-minute bar volumes (normalized by 50-day average volume) to predict the full day's RVol.
- **U-Shape Volume Profile** — Displays the typical intraday volume distribution curve (the well-known U-shape: high volume at open, taper midday, spike at close) and projects today's expected volume at each interval.
- **Dual Prediction Engine:**
  - *Mathematical U-Shape Projection:* Assumes today follows the historical average participation proportions.
  - *Neural Network AI Projection:* The LSTM recognizes volume velocity patterns and predicts momentum exhaustion.
- **Manual Input** — Enter actual 5-minute bar volumes as they come in throughout the day to get continuously updated predictions.
- **Model Evaluation** — Scatter plot of predicted vs actual end-of-day volume across all training data, with best-fit regression line and R² metric.
- **Model Persistence** — Save trained models to the project's `public/models/` folder for reuse across sessions.

**Use Case:** "It's 9:35 AM, and AAPL has traded 500K shares in the first bar. What's the projected total volume for the day?"

---

### Tab 3: Max Excursion Predictor

**Purpose:** Predict a stock's **intraday High and Low prices** from the first 5 minutes of trading data using 11 technical features and a Dense neural network.

**Key Features:**
- **Two-Stage Prediction Pipeline:**
  1. Project total day volume from early bars → derive projected RVol.
  2. Combine projected RVol with 10 additional technical features → predict max excursion as a multiple of ADR → convert to dollar High/Low.
- **11 Technical Features:**
  | Category | Feature |
  |----------|---------|
  | Volume | Projected RVol (from first bars) |
  | Volume | First bar volume / avg volume |
  | Volume | Up/Down volume ratio (early) |
  | Price Action | % Change from previous day close |
  | Price Action | First 5-min price range / ADR |
  | Price Action | % above open after 5 min |
  | Volatility | 20-day ADR (normalized) |
  | Volatility | ATR(14) (normalized) |
  | Moving Averages | ATR distance from 10 EMA |
  | Moving Averages | ATR distance from 20 EMA |
  | Moving Averages | ATR distance from 50 EMA |
- **Live Prediction** — Enter today's open price and first bar volume to get predicted High, Low, and max excursion in ADR multiples.
- **Backtesting Results** — Scatter plot of predicted vs actual max excursion across all training data, with R², MSE, and standard deviation metrics.
- **Per-Ticker Accuracy Table** — MAE breakdown for each of the 20 stocks, sorted best to worst.
- **Sensitivity Analysis** — Permutation-based feature importance: shuffles each feature independently and measures R² drop. Features are ranked with color-coded bars (green = positive correlation, red = negative).
- **Actionable Feature Selection** — Toggle individual features on/off using checkboxes, then retrain with only the selected features to improve generalization.
- **Model Persistence** — Save trained models to `public/models/max-excursion/` folder.

**Use Case:** "TSLA opened at $245 with 800K first-bar volume. Based on today's ADR, ATR, and EMA positioning, what are the predicted High and Low for the day?"

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Vite, TailwindCSS |
| Charting | Plotly.js, Recharts, ApexCharts |
| Machine Learning | TensorFlow.js (LSTM + Dense NN, runs in browser) |
| Backend | Node.js, Express |
| Data Source | Yahoo Finance (`yahoo-finance2`), local intraday CSVs |

## Project Structure

```
├── server.js                  # Express API: YF data, intraday data, model saving
├── src/
│   ├── App.jsx                # Main React component with 3-tab UI
│   ├── utils/
│   │   ├── tfjsEngine.js      # TensorFlow.js model creation, training, prediction
│   │   └── mathUtils.js        # Regression calculators (linear, exponential, polynomial)
│   └── index.css              # Global styles
├── public/models/             # Saved trained models
│   ├── stock-lstm-model.*     # Daily RVol vs Excursion LSTM
│   ├── intraday-model.*       # Intraday volume prediction LSTM
│   └── max-excursion/         # Max excursion prediction Dense NN
├── Intraday Stock Price/      # 20 CSV files with 5-min OHLCV data
└── package.json
```

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/sumonrh/Stock-Stats.git
   cd Stock-Stats
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the app (backend + frontend concurrently):
   ```bash
   npm run dev
   ```
4. Open your browser at `http://localhost:5173`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/yahoo-finance2?ticker=NVDA` | GET | Fetch 2 years of daily OHLCV data |
| `/api/intraday-files` | GET | List available intraday CSV files |
| `/api/intraday-data?ticker=ASTS` | GET | Get 5-min OHLCV bars grouped by day |
| `/api/intraday-features?ticker=ASTS` | GET | Compute daily technical indicators (ADR, ATR, EMAs) |
| `/api/save-model` | POST | Save daily LSTM model to project folder |
| `/api/save-intraday-model` | POST | Save intraday volume model |
| `/api/save-max-excursion-model` | POST | Save max excursion model to subfolder |

## License

See [LICENSE](LICENSE) for details.
