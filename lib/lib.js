import YahooFinance from 'yahoo-finance2';
const yahooFinance = new YahooFinance();

// --- CONFIGURATION ---
export const ETFS = [
    'ITA', 'ROBO', 'PEJ', 'BLOK', 'TAN', 'CIBR', 'IGV', 'ARKG', 'KWEB',
    'XLE', 'SMH', 'XLV', 'XLF', 'FDN', 'XLY', 'XLB', 'UFO', 'XRT', 'XBI',
    'ITB', 'MSOS', 'IYT', 'XLP', 'IYZ', 'NLR', 'XME', 'GDX', 'JETS', 'PBW',
    'MEME', 'BOTT', 'COPX'
];

export const SCORING_CONFIG = {
    MIN_PRICE: 5,
    MIN_PERCENT_CHANGE: -20,
    MAX_PERCENT_CHANGE: 20,
    MAX_UD_RATIO: 5,
    MIN_EMA_DISTANCE: -10,
    MAX_EMA_DISTANCE: 10,
    MAX_VIX_SPIKE: 0.50,
    UNDERCUT_TOLERANCE: -0.5,
};

export const BASE_WEIGHTS = {
    dailyPerformance: 0.07,
    strength: 0.11,
    accumulation: 0.14,
    pullback: 0.24,
    risk: 0.17,
    rsLineMomentum: 0.27,
};

// Standardized U-shaped volume profile (Cumulative)
export const CUMULATIVE_VOLUME_PROFILE = [
    0.008, 0.016, 0.024, 0.032, 0.040, 
    0.047, 0.054, 0.061, 0.068, 0.075, 
    0.081, 0.087, 0.093, 0.099, 0.105, 
    0.130, 0.153, 0.173, 0.191, 0.208, 0.224,
    0.239, 0.253, 0.266, 0.279, 0.291, 0.303, 0.315, 0.327,
    0.339, 0.350, 0.361, 0.372, 0.383, 0.394, 0.405, 0.416, 0.427, 0.438,
    0.449, 0.460, 0.471, 0.482, 0.493, 0.504, 0.515, 0.526, 0.537, 0.548,
    0.559, 0.570, 0.581, 0.592, 0.603, 0.614, 0.625, 0.636, 0.647, 0.658,
    0.669, 0.680, 0.691, 0.702, 0.713, 0.724, 0.735, 0.746, 0.757, 0.768,
    0.779, 0.790, 0.801, 0.812, 0.823, 0.834, 0.845, 0.856, 0.867, 0.878,
    0.889, 0.900, 0.912, 0.924, 0.936, 0.948, 0.962, 0.978, 1.000, 1.000
];

export class PandasLite {
    static mean(arr) {
        if (!arr.length) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }
    static std(arr) {
        if (arr.length < 2) return 0;
        const avg = this.mean(arr);
        const squareDiffs = arr.map(v => Math.pow(v - avg, 2));
        return Math.sqrt(this.mean(squareDiffs));
    }
    static ema(values, span) {
        if (!values.length) return [];
        const k = 2 / (span + 1);
        let ema = values[0];
        const result = [ema];
        for (let i = 1; i < values.length; i++) {
            ema = (values[i] * k) + (ema * (1 - k));
            result.push(ema);
        }
        return result;
    }
    static rollingMean(values, windowSize) {
        const result = [];
        for (let i = 0; i < values.length; i++) {
            if (i < windowSize - 1) { result.push(null); continue; }
            const slice = values.slice(i - windowSize + 1, i + 1);
            result.push(this.mean(slice));
        }
        return result;
    }
    static rollingStd(values, windowSize) {
        const result = [];
        for (let i = 0; i < values.length; i++) {
            if (i < windowSize - 1) { result.push(null); continue; }
            const slice = values.slice(i - windowSize + 1, i + 1);
            result.push(this.std(slice));
        }
        return result;
    }
    static calculateATR(highs, lows, closes, period = 14) {
        const tr = [];
        for (let i = 0; i < highs.length; i++) {
            if (i === 0) tr.push(highs[i] - lows[i]);
            else {
                const hl = highs[i] - lows[i];
                const h_pc = Math.abs(highs[i] - closes[i - 1]);
                const l_pc = Math.abs(lows[i] - closes[i - 1]);
                tr.push(Math.max(hl, h_pc, l_pc));
            }
        }
        return this.rollingMean(tr, period);
    }
    static calculateSMAValues(values, period) {
        const sma = new Array(values.length).fill(null);
        if (values.length < period) return sma;
        for (let i = period - 1; i < values.length; i++) {
            const slice = values.slice(i - period + 1, i + 1);
            sma[i] = slice.reduce((sum, val) => sum + val, 0) / period;
        }
        return sma;
    }
    static calculateStdevValues(values, period) {
        const stdevs = new Array(values.length).fill(null);
        if (values.length < period) return stdevs;
        for (let i = period - 1; i < values.length; i++) {
            const slice = values.slice(i - period + 1, i + 1);
            const mean = slice.reduce((sum, val) => sum + val, 0) / period;
            const sqDiffs = slice.map(val => Math.pow(val - mean, 2));
            const variance = sqDiffs.reduce((sum, val) => sum + val, 0) / period;
            stdevs[i] = Math.sqrt(variance);
        }
        return stdevs;
    }
    static getSqueezeStatus(bbUpper, bbLower, kcUppers, kcLowers) {
        if (bbUpper > kcUppers[2] || bbLower < kcLowers[2]) return 'No';
        if (bbUpper <= kcUppers[0] && bbLower >= kcLowers[0]) return 'High';
        if (bbUpper <= kcUppers[1] && bbLower >= kcLowers[1]) return 'Medium';
        return 'Low';
    }
}

export function detectVixRegime(vixQuotes) {
    if (!vixQuotes || vixQuotes.length < 200) return { vixRegimeNum: 1, sma200: null, atr14: null, upperBand: null, lowerBand: null };
    const closes = vixQuotes.map(q => q.close);
    const sma200 = closes.slice(-200).reduce((a, b) => a + b, 0) / 200;
    const recent = vixQuotes.slice(-15);
    const tr = [];
    for (let i = 1; i < recent.length; i++) {
        tr.push(Math.max(recent[i].high - recent[i].low, Math.abs(recent[i].high - recent[i - 1].close), Math.abs(recent[i].low - recent[i - 1].close)));
    }
    const atr14 = tr.length > 0 ? tr.reduce((a, b) => a + b, 0) / tr.length : 0;
    return { vixRegimeNum: closes[closes.length - 1] > (sma200 + 0.5 * atr14) ? 2 : (closes[closes.length - 1] < (sma200 - 0.5 * atr14) ? 0 : 1), sma200, atr14, upperBand: sma200 + 0.5 * atr14, lowerBand: sma200 - 0.5 * atr14 };
}

export class MarketRegime {
    static clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
    static evaluate(vixLevel, vixChange, atrZScore = 0, vixRegimeNum = 1, vixSafeguardEnabled = true) {
        if (vixSafeguardEnabled) {
            if (vixLevel > 35) return { status: 'PANIC', multiplier: 0.5, reason: 'VIX > 35' };
            if (vixChange > 15) return { status: 'PANIC', multiplier: 0.5, reason: 'VIX Spike' };
        }
        const mult = vixRegimeNum === 0 ? 1.25 : (vixRegimeNum === 2 ? 0.5 : 1.0);
        return { status: vixRegimeNum === 0 ? 'BULL' : (vixRegimeNum === 2 ? 'BEAR' : 'NORMAL'), multiplier: mult };
    }
}

export class QuantScorer {
    static sigmoidNormalize(value, minVal, maxVal, steepness = 1.0, asymmetric = false) {
        if (maxVal === minVal) return value === minVal ? 0.5 : (value > maxVal ? 1.0 : 0.0);
        const linear = (Math.max(minVal, Math.min(maxVal, value)) - minVal) / (maxVal - minVal);
        return 1 / (1 + Math.exp(-(asymmetric && linear < 0.5 ? steepness * 1.5 : steepness) * (linear - 0.5) * 10));
    }
    static normalizeWeights(weights) {
        let total = 0; for (const k in weights) total += Math.max(0, weights[k]);
        const norm = {}; for (const k in weights) norm[k] = total === 0 ? 1 / Object.keys(weights).length : Math.max(0, weights[k]) / total;
        return norm;
    }
    static getAdjustedWeights(vixData) {
        const weights = { ...BASE_WEIGHTS };
        if (!vixData || vixData.price <= 0) return this.normalizeWeights(weights);
        const vvRatio = vixData.vxvPrice ? vixData.price / vixData.vxvPrice : 1.0;
        const vixChg = (vixData.price - vixData.previousClose) / vixData.previousClose;
        if (vvRatio > 1.0 || vixChg > 0.10) { weights.risk += 0.15; weights.pullback += 0.10; weights.dailyPerformance -= 0.20; }
        return this.normalizeWeights(weights);
    }
    static calculateScore(stock, vixData, spyChange) {
        const weights = this.getAdjustedWeights(vixData);
        const alpha = stock.percentChange - spyChange;
        const dailyPerf = this.sigmoidNormalize(alpha, -3, 3, 2.0, true);
        const strength = this.sigmoidNormalize(stock.rsRating, 0.5, 1.5, 2.0);
        const accum = this.sigmoidNormalize(stock.udRatio, 0.7, 2.5, 1.5);
        const pullback = (this.sigmoidNormalize(stock.distanceFrom10EMA, -2, 2) + this.sigmoidNormalize(stock.distanceFrom20EMA, -3, 3)) / 2;
        const composite = (dailyPerf * weights.dailyPerformance) + (strength * weights.strength) + (accum * weights.accumulation) + (pullback * weights.pullback);
        return Math.round(composite * 100);
    }
}

export class IntradayPredictor {
    static clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
    static getProjectedRelativeVolume(curVol, avgVol, mins) {
        if (avgVol === 0) return 0;
        if (mins >= 390) return curVol / avgVol;
        const pct = CUMULATIVE_VOLUME_PROFILE[mins < 15 ? Math.floor(mins) : 15 + Math.floor((mins - 15) / 5)] || 1;
        return (curVol / pct) / avgVol;
    }
    static predict(input) {
        const { currentPrice, prevClose, openPrice, relativeVolume, percentADR, minsSinceOpen } = input;
        const retSoFar = (currentPrice - prevClose) / prevClose;
        const projectedRange = (percentADR / 100) * (1 + Math.log10(relativeVolume || 1));
        const remaining = (390 - (minsSinceOpen || 0)) / 390;
        const pred = retSoFar + (projectedRange * remaining * 0.5 * (currentPrice > openPrice ? 1 : -1));
        return { predictedEodChange: Number((pred * 100).toFixed(2)) };
    }
}

export async function fetchMarketContext() {
    const twoYearsAgo = new Date(Date.now() - 730 * 86400000);
    const spyData = await yahooFinance.chart('SPY', { period1: twoYearsAgo, interval: '1d' });
    const vixData = await yahooFinance.chart('^VIX', { period1: new Date(Date.now() - 400 * 86400000), interval: '1d' });
    const currentVix = vixData.quotes[vixData.quotes.length - 1];
    return { spyData: spyData.quotes, vixContext: { price: currentVix.close, previousClose: vixData.quotes[vixData.quotes.length - 2].close, vixRegimeNum: detectVixRegime(vixData.quotes).vixRegimeNum } };
}

export function calculateMetrics(ticker, quotes, context) {
    if (!quotes || quotes.length < 50) return null;
    const closes = quotes.map(q => q.close);
    const cp = closes[closes.length - 1], pc = closes[closes.length - 2];
    const relVol = IntradayPredictor.getProjectedRelativeVolume(quotes[quotes.length-1].volume, PandasLite.mean(quotes.slice(-50).map(q=>q.volume)), 390);
    const prediction = IntradayPredictor.predict({ currentPrice: cp, prevClose: pc, openPrice: quotes[quotes.length-1].open, relativeVolume: relVol, percentADR: 5, minsSinceOpen: 390 });
    return { ticker, price: cp, percentChange: ((cp - pc) / pc) * 100, RVol: relVol, '%Pred': prediction.predictedEodChange };
}

export async function processBatch(tickers, context) {
    const results = [];
    for (const t of tickers) {
        try {
            const chart = await yahooFinance.chart(t, { period1: new Date(Date.now() - 730 * 86400000), interval: '1d' });
            const m = calculateMetrics(t, chart.quotes, context);
            if (m) results.push(m);
        } catch (e) {}
    }
    return results;
}
