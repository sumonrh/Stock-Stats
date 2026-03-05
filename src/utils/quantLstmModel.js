import * as tf from '@tensorflow/tfjs';

// Initialize WebGL/WASM backend for browser multi-core / GPU limits
let backendInitialized = false;
export const initTfBackend = async () => {
    if (backendInitialized) return;
    try {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log("TFJS WebGL backend enabled for fast GPU acceleration.");
    } catch (e) {
        try {
            await tf.setBackend('wasm'); // use WebAssembly if WebGL fails
            await tf.ready();
            console.log("TFJS WASM backend enabled for multi-core CPU computation.");
        } catch (err) {
            console.log("Using default TFJS CPU backend.");
        }
    }
    backendInitialized = true;
};

/**
 * Prepares the backtest dataset for LSTM training
 */
export const prepareQuantDataForLstm = (backtestData, targetFeature = 'ret1W', sequenceLength = 5) => {
    // 1. Sort chronological grouping by ticker
    const dataByTicker = {};
    backtestData.forEach(row => {
        if (!dataByTicker[row.ticker]) dataByTicker[row.ticker] = [];
        dataByTicker[row.ticker].push(row);
    });

    for (let ticker in dataByTicker) {
        dataByTicker[ticker].sort((a, b) => new Date(a.date) - new Date(b.date));
    }

    const featureKeys = [
        'rsDelta',
        'vcp',
        'rVol',
        'priceChangeOverAdr',
        'episodicPivotPower',
        'ema10DistAtr',
        'ema20DistAtr'
    ];

    // 2. Compute Z-Score Normalization Stats (Mean and StdDev)
    // Z-Score is vastly superior to MinMax here because episodic pivots/RVol can have massive outliers
    // which compress normal signals into near-zero variance.
    const stats = {};
    featureKeys.forEach(key => {
        const values = backtestData.map(d => d[key] != null ? d[key] : 0);
        const sum = values.reduce((a, b) => a + b, 0);
        const mean = sum / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance) || 1; // avoid divide by zero
        stats[key] = { mean, std };
    });

    const targetValues = backtestData.map(d => d[targetFeature] != null ? d[targetFeature] : 0);
    const targetSum = targetValues.reduce((a, b) => a + b, 0);
    const targetMean = targetSum / targetValues.length;
    const targetVariance = targetValues.reduce((a, b) => a + Math.pow(b - targetMean, 2), 0) / targetValues.length;
    const targetStd = Math.sqrt(targetVariance) || 1;
    stats.target = { mean: targetMean, std: targetStd };

    const normalize = (val, mean, std) => {
        // Robust scaling: clip extreme outliers (+/- 5 standard deviations) to prevent model explosion
        const z = (val - mean) / std;
        return Math.max(-5, Math.min(5, z));
    };

    const sequences = [];
    const targets = [];

    // 3. Create Rolling Sequences per Ticker
    for (let ticker in dataByTicker) {
        const rows = dataByTicker[ticker];
        if (rows.length < sequenceLength) continue;

        for (let i = sequenceLength - 1; i < rows.length; i++) {
            const seq = [];
            // FIX: Include day `i` in the features to remove the 1-day look-ahead/lag bias!
            // We want features up to day `i` to predict forward return from day `i`
            for (let j = sequenceLength - 1; j >= 0; j--) {
                const stepRow = rows[i - j];
                const featureVector = featureKeys.map(key => normalize(stepRow[key], stats[key].mean, stats[key].std));
                seq.push(featureVector);
            }
            sequences.push(seq);

            // Normalize Target
            targets.push(normalize(rows[i][targetFeature], stats.target.mean, stats.target.std));
        }
    }

    return { sequences, targets, stats, featureKeys };
};

export const buildQuantLstmModel = (sequenceLength, numFeatures) => {
    const model = tf.sequential();

    // 1st LSTM Layer (Single layer is preferred in browser to prevent thread freezing)
    model.add(tf.layers.lstm({
        units: 64,
        inputShape: [sequenceLength, numFeatures],
        returnSequences: false
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));

    // Dense Layers
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 1 })); // Linear output

    // Compile with Mean Squared Error
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError'
    });

    return model;
};

/**
 * Full Pipeline: Prepare, Compile, Train
 */
export const trainQuantModel = async (backtestData, targetFeature, sequenceLength, epochs, onEpochEndMsg) => {
    if (!backtestData || backtestData.length === 0) throw new Error("No data provided");

    // Pre-init webGL acceleration
    await initTfBackend();

    const { sequences, targets, stats, featureKeys } = prepareQuantDataForLstm(backtestData, targetFeature, sequenceLength);

    if (sequences.length === 0) throw new Error("Not enough sequential data to train");

    const xs = tf.tensor3d(sequences);
    const ys = tf.tensor2d(targets, [targets.length, 1]);

    const model = buildQuantLstmModel(sequenceLength, featureKeys.length);

    // Early Stopping Configuration
    let bestLoss = Infinity;
    let patienceCounter = 0;
    const patience = 12; // Adjusted for deeper net
    let finalEpoch = 0;

    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: 64, // Larger batch size to utilize GPU/CPU parallelism better
        validationSplit: 0.2,
        yieldEvery: 'epoch', // Prevents browser from freezing by yielding to the main thread
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                finalEpoch = epoch + 1;
                const currentLoss = logs.val_loss !== undefined ? logs.val_loss : logs.loss;

                if (onEpochEndMsg) onEpochEndMsg(epoch + 1, epochs, currentLoss);

                if (currentLoss < bestLoss) {
                    bestLoss = currentLoss;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                }

                if (patienceCounter >= patience) {
                    console.log(`Early stopping triggered at epoch ${epoch + 1}`);
                    model.stopTraining = true;
                }
            }
        }
    });

    // Run custom sensitivity analysis after training
    const sensitivity = await runSensitivityAnalysis(model, xs, ys, featureKeys);

    xs.dispose();
    ys.dispose();

    return {
        model,
        stats,
        featureKeys,
        targetFeature,
        sequenceLength,
        sensitivity,
        finalEpoch
    };
};

/**
 * Live Prediction using trained model
 */
export const predictFutureReturn = (model, stats, featureKeys, recentHistory) => {
    const sequenceLength = recentHistory.length;

    const normalize = (val, mean, std) => {
        const z = (val - mean) / std;
        return Math.max(-5, Math.min(5, z));
    };

    const seq = [];
    for (let i = 0; i < sequenceLength; i++) {
        const stepRow = recentHistory[i];
        const featureVector = featureKeys.map(key => normalize(stepRow[key] || 0, stats[key].mean, stats[key].std));
        seq.push(featureVector);
    }

    const inputTensor = tf.tensor3d([seq]);
    const predictionTensor = model.predict(inputTensor);
    const normalizedPrediction = predictionTensor.dataSync()[0];

    inputTensor.dispose();
    predictionTensor.dispose();

    // Denormalize the output (Reverse Z-score)
    const finalPredictedReturn = (normalizedPrediction * stats.target.std) + stats.target.mean;

    return finalPredictedReturn;
};

/**
 * Runs sensitivity analysis on the trained model
 */
export const runSensitivityAnalysis = async (model, xs, ys, featureKeys) => {
    const numSamples = xs.shape[0];
    const seqLength = xs.shape[1];
    const numFeatures = xs.shape[2];

    // Baseline Error
    const baselinePreds = model.predict(xs);
    const baselineError = tf.losses.absoluteDifference(ys, baselinePreds).dataSync()[0];
    baselinePreds.dispose();

    const analysis = [];

    // Test each feature
    for (let f = 0; f < numFeatures; f++) {
        const xsData = await xs.data();
        const maskedData = new Float32Array(xsData.length);

        // Copy data but mask (zero out) the specific feature
        for (let i = 0; i < xsData.length; i++) {
            const featureIndex = i % numFeatures;
            if (featureIndex === f) {
                // IMPORTANT FIX: Z-Score mean is 0. So to neutralize a feature exactly to its historical average, set it to 0.
                maskedData[i] = 0;
            } else {
                maskedData[i] = xsData[i];
            }
        }

        const maskedXs = tf.tensor3d(maskedData, [numSamples, seqLength, numFeatures]);
        const maskedPreds = model.predict(maskedXs);
        const maskedError = tf.losses.absoluteDifference(ys, maskedPreds).dataSync()[0];

        // Calculate Error Increase (higher = feature is more important)
        const maeIncrease = maskedError - baselineError;

        analysis.push({
            feature: featureKeys[f],
            impactScore: Math.max(0, maeIncrease) // Floor at 0 if no impact
        });

        maskedXs.dispose();
        maskedPreds.dispose();
    }

    // Normalize impact scores to percentages summing to 100
    const totalImpact = analysis.reduce((sum, item) => sum + item.impactScore, 0);
    analysis.forEach(item => {
        item.weight = totalImpact > 0 ? (item.impactScore / totalImpact) * 100 : 0;
    });

    // Sort by largest impact
    analysis.sort((a, b) => b.weight - a.weight);
    return analysis;
};
