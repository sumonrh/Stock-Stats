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
    dailyPerformance: 0.05,
    strength: 0.1,
    accumulation: 0.1,
    pullback: 0.4,
    risk: 0.1,
    rsLineMomentum: 0.25,
};

export class QuantScorer {
    static sigmoidNormalize(value, minVal, maxVal, steepness = 1.0, asymmetric = false) {
        if (maxVal === minVal) {
            return value === minVal ? 0.5 : (value > maxVal ? 1.0 : 0.0);
        }
        const clamped = Math.max(minVal, Math.min(maxVal, value));
        const linear = (clamped - minVal) / (maxVal - minVal);
        let adjSteepness = steepness;
        if (asymmetric && linear < 0.5) adjSteepness *= 1.5;
        return 1 / (1 + Math.exp(-adjSteepness * (linear - 0.5) * 10));
    }

    static normalizeWeights(weights) {
        let total = 0;
        for (const k in weights) total += Math.max(0, weights[k]);
        if (total === 0) total = 1;
        const normalized = {};
        for (const k in weights) normalized[k] = Math.max(0, weights[k]) / total;
        return normalized;
    }

    static getAdjustedWeights(vixData) {
        const weights = { ...BASE_WEIGHTS };
        if (!vixData || vixData.price <= 0) return this.normalizeWeights(weights);

        const vixPrice = vixData.price;
        const vxvPrice = vixData.vxvPrice || vixPrice;
        const prevClose = vixData.previousClose || vixPrice;

        const vvRatio = vxvPrice > 0 ? vixPrice / vxvPrice : 1.0;
        const vixDayChange = Math.min(SCORING_CONFIG.MAX_VIX_SPIKE, (vixPrice - prevClose) / prevClose);

        if (vvRatio > 1.0 || vixDayChange > 0.10) {
            const stress = Math.max(vvRatio, 1 + vixDayChange);
            weights.risk += 0.15 * stress;
            weights.pullback += 0.10;
            weights.dailyPerformance -= 0.20;
            weights.rsLineMomentum += 0.10;
        } else if (vvRatio < 0.85) {
            weights.strength += 0.10;
        }
        return this.normalizeWeights(weights);
    }

    static calculateSlope(current, previous) {
        if (previous === 0) return 0;
        return (current - previous) / previous;
    }

    static getPullbackSubScore(distance, idealMax, slope) {
        let baseScore = 0;
        if (distance >= SCORING_CONFIG.UNDERCUT_TOLERANCE && distance <= idealMax) {
            baseScore = 1.0;
        } else if (distance > idealMax) {
            const excess = distance - idealMax;
            baseScore = Math.max(0, 1 - (excess / (idealMax * 2)));
        } else {
            const severity = Math.abs(distance - SCORING_CONFIG.UNDERCUT_TOLERANCE);
            baseScore = Math.max(0, 1 - Math.pow(severity, 2) / 2);
        }

        let slopeMultiplier = 1.0;
        if (slope > 0.005) slopeMultiplier = 1.2;
        else if (slope < 0) slopeMultiplier = 0.5;
        else if (slope <= 0.001) slopeMultiplier = 0.8;

        return baseScore * slopeMultiplier;
    }

    static calculateScore(stock, vixData, spyChange) {
        // --- Input Validation & Sanitization (Sync with provided algorithm) ---
        const stockCopy = { ...stock };
        const requiredProps = ['price', 'high', 'low', 'percentChange', 'rsRating', 'udRatio', 'percentADR', 'atr', 'distanceFrom10EMA', 'distanceFrom20EMA', 'distanceFrom50EMA', 'ema10', 'ema20', 'ema50', 'rsLineSlope', 'ema10Prev5', 'ema20Prev5', 'ema50Prev5'];

        for (const prop of requiredProps) {
            const val = stockCopy[prop];
            if (val == null || !Number.isFinite(val)) {
                stockCopy[prop] = (prop === 'rsRating' || prop === 'udRatio') ? 1.0 : (prop === 'price' ? SCORING_CONFIG.MIN_PRICE : 0);
            }
        }

        // Explicit ATR Validations
        if (stockCopy.atr <= 0) stockCopy.atr = 0.0001;

        // Input Clamping
        stockCopy.percentChange = Math.max(SCORING_CONFIG.MIN_PERCENT_CHANGE, Math.min(SCORING_CONFIG.MAX_PERCENT_CHANGE, stockCopy.percentChange));
        stockCopy.udRatio = Math.max(0, Math.min(SCORING_CONFIG.MAX_UD_RATIO, stockCopy.udRatio));
        stockCopy.distanceFrom10EMA = Math.max(SCORING_CONFIG.MIN_EMA_DISTANCE, Math.min(SCORING_CONFIG.MAX_EMA_DISTANCE, stockCopy.distanceFrom10EMA));
        stockCopy.distanceFrom20EMA = Math.max(SCORING_CONFIG.MIN_EMA_DISTANCE, Math.min(SCORING_CONFIG.MAX_EMA_DISTANCE, stockCopy.distanceFrom20EMA));
        stockCopy.distanceFrom50EMA = Math.max(SCORING_CONFIG.MIN_EMA_DISTANCE, Math.min(SCORING_CONFIG.MAX_EMA_DISTANCE, stockCopy.distanceFrom50EMA));


        const weights = this.getAdjustedWeights(vixData);

        // Daily Perf
        const relativeAlpha = stockCopy.percentChange - spyChange;
        let dailyPerfScore = this.sigmoidNormalize(relativeAlpha, -3, 3, 2.0, true);
        if (spyChange < -1.5 && stockCopy.percentChange > 0) dailyPerfScore = Math.min(1.0, dailyPerfScore + 0.15);

        // Strength
        const strengthScore = this.sigmoidNormalize(stockCopy.rsRating, 0.5, 1.5, 2.0, true);

        // Accumulation
        const accumulationScore = this.sigmoidNormalize(stockCopy.udRatio, 0.7, 2.5, 1.5, true);

        // Pullback
        const slope10 = this.calculateSlope(stockCopy.ema10, stockCopy.ema10Prev5);
        const slope20 = this.calculateSlope(stockCopy.ema20, stockCopy.ema20Prev5);
        const slope50 = this.calculateSlope(stockCopy.ema50, stockCopy.ema50Prev5);

        const dist10Score = this.getPullbackSubScore(stockCopy.distanceFrom10EMA, 1.5, slope10);
        const dist20Score = this.getPullbackSubScore(stockCopy.distanceFrom20EMA, 2.0, slope20);
        const dist50Score = this.getPullbackSubScore(stockCopy.distanceFrom50EMA, 4.0, slope50);

        let rawPullback = (dist10Score * 0.4) + (dist20Score * 0.4) + (dist50Score * 0.2);

        // Trend Checks
        const isEma200Valid = stockCopy.ema200 !== undefined && stockCopy.ema200 > 0;
        const bullishAlignments = [
            stockCopy.price > stockCopy.ema50,
            stockCopy.ema10 > stockCopy.ema20,
            stockCopy.ema20 > stockCopy.ema50,
            isEma200Valid ? stockCopy.ema50 > stockCopy.ema200 : false
        ];
        let bullish = bullishAlignments.every(Boolean);

        const atr = stockCopy.atr;
        const emaSep = Math.abs(stockCopy.ema10 - stockCopy.ema20) / atr;
        const isCoiled = emaSep < 0.5;
        const isBouncing = stockCopy.distanceFrom10EMA >= 0;
        const hasHighRs = this.sigmoidNormalize(stockCopy.rsLineSlope * 100, 0, 15) > 0.8;

        if (bullish && isCoiled && isBouncing && hasHighRs) rawPullback += 0.25;

        let pullbackScore = Math.max(0, Math.min(1.0, rawPullback));


        // Risk
        const dists = [stockCopy.distanceFrom10EMA, stockCopy.distanceFrom20EMA, stockCopy.distanceFrom50EMA];
        const supports = dists.filter(d => d >= 0);
        const resistances = dists.filter(d => d < 0);
        let riskScore = 0;
        const riskWindow = bullish ? 4.0 : 3.0;

        if (supports.length > 0) {
            const nearestBelow = Math.min(...supports);
            riskScore = 1 - this.sigmoidNormalize(nearestBelow, 0, riskWindow, 1.5);
        } else {
            const nearestAbove = resistances.length ? Math.max(...resistances) : -5;
            riskScore = 0.2 * this.sigmoidNormalize(nearestAbove, -5, 0, 1.0);
        }
        riskScore = Math.max(0, Math.min(1.0, riskScore));

        // RS Momentum (Updated for Matching TradeGenius-V8: 1-Day Slope, 0-15 Range)
        const rsSlopePct = stockCopy.rsLineSlope * 100;
        const rsMult = (spyChange <= 0 && rsSlopePct > 0) ? 1.25 : 1.0;
        const rsMomScore = Math.min(1.0, this.sigmoidNormalize(rsSlopePct, 0, 15, 1.0) * rsMult);

        // Composite
        let composite =
            (dailyPerfScore * weights.dailyPerformance) +
            (strengthScore * weights.strength) +
            (accumulationScore * weights.accumulation) +
            (pullbackScore * weights.pullback) +
            (riskScore * weights.risk) +
            (rsMomScore * weights.rsLineMomentum);

        // Penalties
        let adrPenalty = 1.0;
        if (stockCopy.percentADR > 20) adrPenalty = 0.85;
        else if (stockCopy.percentADR < 1.5) adrPenalty = 0.9;

        let pricePenalty = 1.0;
        if (stockCopy.price < SCORING_CONFIG.MIN_PRICE) {
            if (stockCopy.price >= 1) pricePenalty = 0.7 + 0.3 * ((stockCopy.price - 1) / (SCORING_CONFIG.MIN_PRICE - 1));
            else pricePenalty = 0.6;
        }

        return Math.max(0, Math.min(100, Math.round(composite * adrPenalty * pricePenalty * 100)));
    }
}
