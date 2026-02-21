# Stock Stats Analyzer

## Overview
Stock Stats Analyzer is a powerful, interactive web application designed to help traders and data analysts visualize the statistical relationship between a stock's **Relative Volume (RVol)** and its **Intraday Price Excursion** (the absolute maximum move from the daily open). 

By analyzing historical candlestick data and projecting the spread of both volume surges and price volatility, this dashboard helps traders identify high-probability setups and quantify how much "stretch" a stock might exhibit on a given day.

## Key Features
* **Real-Time Data Fetching:** Seamlessly pulls live and historical market data using `yahoo-finance2` via a custom backend Express API.
* **Interactive Scatter Plot with Regressions:** Visualizes historical days for a ticker on an X/Y axes (RVol vs Abs Max Excursion / ADR), complete with standard deviation distribution bands to calculate statistical mean trajectories. 
* **Combined Marginal Probability Charting:** Utilizes advanced `Plotly.js` charting to map the scatter plot alongside marginal distribution "bell curves" that compute the specific Probability Density of an RVol event and the Probability Density of an Excursion event.
* **Profitable Zone Highlighting:** Specifically isolates the "Profitable Zone" mapping out historical precedents of simultaneous high RVol (>1.5x) and high excursions (>1.0 ADR) designed to help quantify outlier expansion trading setups.
* **Custom Inputs:** Dynamically switch tickers, dynamically adjust ADR history periods (10-day, 20-day, 50-day), and filter individual datasets without needing to reload the tool.
* **Data Transparency:** Mouse-hover interactive tooltips offer fully transparent insights into the underlying values, and an embedded summary table aggregates findings by predefined volume bucket groups.

## Technical Stack
* **Frontend:** React, Vite, TailwindCSS
* **Charting:** Plotly.js (`react-plotly.js`), Recharts (for distribution bar plots), ApexCharts (for traditional Candlestick/Volume charts).
* **Backend:** Node.js, Express
* **Data Source:** Yahoo Finance (`yahoo-finance2` library).

## How to Run Locally
1. Clone the repository.
2. Run `npm install` to install all necessary dependencies.
3. Run `npm run backend` (or `node server.js`) to spin up the local Express API fetching data.
4. Run `npm run dev` to launch the React frontend locally.
5. Navigate to the port listed by Vite (usually `http://localhost:5173`).
