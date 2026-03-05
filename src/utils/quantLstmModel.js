import * as tf from '@tensorflow/tfjs';

/**
 * Prepares the backtest dataset for LSTM training
 * Features: rsDelta, rVol, percentADR, ema10DistAtr, ema20DistAtr, quantScore (optional, or we use it later)
 * Target: ret1W or ret1M
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
        'rsDelta',              // slope %
        'vcp',                  // 0 to 3
        'rVol',                 // typically 0 to 10
        'priceChangeOverAdr',   // EPS proxy, move size / ADR
        'episodicPivotPower',   // EPS power, RVol * move size
        'ema10DistAtr',         // typically -5 to 5
        'ema20DistAtr'          // typically -5 to 5
    ];

    // 2. Global Min/Max Normalization Stats
    const stats = {};
    featureKeys.forEach(key => {
        const values = backtestData.map(d => d[key]);
        stats[key] = {
            min: Math.min(...values),
            max: Math.max(...values)
        };
    });

    // Also get stats for target to denormalize later
    const targetValues = backtestData.map(d => d[targetFeature]);
    stats.target = {
        min: Math.min(...targetValues),
        max: Math.max(...targetValues)
    };

    const normalize = (val, min, max) => {
        if (max === min) return 0.5;
        return (val - min) / (max - min);
    };

    const sequences = [];
    const targets = [];

    // 3. Create Rolling Sequences per Ticker
    for (let ticker in dataByTicker) {
        const rows = dataByTicker[ticker];
        if (rows.length <= sequenceLength) continue;

        for (let i = sequenceLength; i < rows.length; i++) {
            const seq = [];
            for (let j = sequenceLength; j > 0; j--) {
                const stepRow = rows[i - j];
                const featureVector = featureKeys.map(key => normalize(stepRow[key], stats[key].min, stats[key].max));
                seq.push(featureVector);
            }
            sequences.push(seq);

            // Normalize Target
            targets.push(normalize(rows[i][targetFeature], stats.target.min, stats.target.max));
        }
    }

    return { sequences, targets, stats, featureKeys };
};

/**
 * Builds and Compiles the LSTM Model
 */
export const buildQuantLstmModel = (sequenceLength, numFeatures) => {
    const model = tf.sequential();

    // LSTM Layer
    model.add(tf.layers.lstm({
        units: 64, // Increased capacity
        inputShape: [sequenceLength, numFeatures],
        returnSequences: false
    }));

    // Regularization
    model.add(tf.layers.dropout({ rate: 0.2 }));

    // Dense Layers for prediction
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 1 })); // Linear activation for regression

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

    const { sequences, targets, stats, featureKeys } = prepareQuantDataForLstm(backtestData, targetFeature, sequenceLength);

    if (sequences.length === 0) throw new Error("Not enough sequential data to train");

    const xs = tf.tensor3d(sequences);
    const ys = tf.tensor2d(targets, [targets.length, 1]);

    const model = buildQuantLstmModel(sequenceLength, featureKeys.length);

    // Early Stopping Configuration
    let bestLoss = Infinity;
    let patienceCounter = 0;
    const patience = 10; // Stop if val_loss doesn't improve for 10 epochs
    let finalEpoch = 0;

    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: 32,
        validationSplit: 0.2, // Use 20% for testing
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
 * @param {tf.Sequential} model 
 * @param {Object} stats The normalization stats output from training
 * @param {Array} featureKeys Ordered list of feature names used during training
 * @param {Array<Object>} recentHistory The last N days of data objects (length must equal sequenceLength)
 */
export const predictFutureReturn = (model, stats, featureKeys, recentHistory) => {
    const sequenceLength = recentHistory.length;

    const normalize = (val, min, max) => {
        if (max === min) return 0.5;
        return (val - min) / (max - min);
    };

    const seq = [];
    for (let i = 0; i < sequenceLength; i++) {
        const stepRow = recentHistory[i];
        const featureVector = featureKeys.map(key => normalize(stepRow[key] || 0, stats[key].min, stats[key].max));
        seq.push(featureVector);
    }

    const inputTensor = tf.tensor3d([seq]);
    const predictionTensor = model.predict(inputTensor);
    const normalizedPrediction = predictionTensor.dataSync()[0];

    inputTensor.dispose();
    predictionTensor.dispose();

    // Denormalize the output
    const targetMin = stats.target.min;
    const targetMax = stats.target.max;
    const finalPredictedReturn = (normalizedPrediction * (targetMax - targetMin)) + targetMin;

    return finalPredictedReturn;
};

/**
 * Runs sensitivity analysis on the trained model by systematically zeroing out features
 * to determine their relative impact on Mean Absolute Error (MAE)
 */
export const runSensitivityAnalysis = async (model, xs, ys, featureKeys) => {
    const numSamples = xs.shape[0];
    const seqLength = xs.shape[1];
    const numFeatures = xs.shape[2];

    // Baseline MAE
    const baselinePreds = model.predict(xs);
    const baselineMae = tf.losses.absoluteDifference(ys, baselinePreds).dataSync()[0];
    baselinePreds.dispose();

    const analysis = [];

    // Test each feature
    for (let f = 0; f < numFeatures; f++) {
        // Create a copy of the tensor data
        const xsData = await xs.data();
        const maskedData = new Float32Array(xsData.length);

        // Copy data but mask (zero out) the specific feature
        for (let i = 0; i < xsData.length; i++) {
            // Calculate which feature this index belongs to
            // i is a flat index mapping to [sample, seqStep, feature]
            const featureIndex = i % numFeatures;
            if (featureIndex === f) {
                maskedData[i] = 0.5; // Set to the normalized "neutral/zero" point (which is 0.5 based on our normalization) instead of strictly 0
            } else {
                maskedData[i] = xsData[i];
            }
        }

        const maskedXs = tf.tensor3d(maskedData, [numSamples, seqLength, numFeatures]);

        // Predict with masked feature
        const maskedPreds = model.predict(maskedXs);
        const maskedMae = tf.losses.absoluteDifference(ys, maskedPreds).dataSync()[0];

        // Calculate Error Increase (higher = feature is more important)
        const maeIncrease = maskedMae - baselineMae;

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
