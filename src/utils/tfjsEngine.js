import * as tf from '@tensorflow/tfjs';
import { getExpectedCumulativeVolumePercentage } from './mathUtils';

// Configuration
const SEQ_LENGTH = 5; // Use past 5 days to predict today

/**
 * Normalizes an array of numbers to the range [0, 1] using Min-Max scaling.
 * Returns the normalized data, along with min and max for denormalization.
 */
const normalize = (data) => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    if (max - min === 0) return { data: data.map(() => 0), min, max };
    const normalized = data.map((val) => (val - min) / (max - min));
    return { data: normalized, min, max };
};

/**
 * Denormalizes a single value.
 */
const denormalizeValue = (val, min, max) => {
    return val * (max - min) + min;
};

/**
 * Prepares sequences from the flat array of processed data points.
 * Sorts chronological (oldest to newest).
 */
export const prepareTfjsData = (tickerData) => {
    if (!tickerData || tickerData.length <= SEQ_LENGTH) {
        return null;
    }

    // Ensure chronological order
    const sorted = [...tickerData].sort((a, b) => new Date(a.date) - new Date(b.date));

    const rawRVols = sorted.map(d => d.rVol);
    const rawExcursions = sorted.map(d => d.maxExcursionAdr);

    const { data: normRVols } = normalize(rawRVols);
    const { data: normExcursions, min: excMin, max: excMax } = normalize(rawExcursions);

    const inputs = [];
    const labels = [];
    const validDates = [];
    const sourcePoints = []; // Keep reference to original point to map later

    // Build sequences
    for (let i = SEQ_LENGTH; i < sorted.length; i++) {
        const sequence = [];
        for (let j = i - SEQ_LENGTH; j < i; j++) {
            // Historical features: past RVol and past Excursions
            sequence.push([normRVols[j], normExcursions[j]]);
        }
        // Also include today's rVol as that is our trigger mechanism 
        // We append it to the sequence as another time step or feature?
        // Let's just put it in the last step of the sequence as a 3rd feature, and 0 for past days.
        // Actually, simpler: just feed the past 5 days, and let the LSTM figure it out.
        // Wait, the prompt says predicting better than linear regression of TODAY's RVol. 
        // If the LSTM doesn't see TODAY's RVol, it might be at a huge disadvantage.
        // So let's add today's RVol to the final step of the sequence.

        // Feature shape: [past_RVol, past_Exc, todays_rVol_mask]
        const seqWithToday = sequence.map((step, idx) => {
            if (idx === SEQ_LENGTH - 1) {
                return [step[0], step[1], normRVols[i]]; // At the last step, inject today's RVol
            } else {
                return [step[0], step[1], 0];
            }
        });

        inputs.push(seqWithToday);
        labels.push(normExcursions[i]);
        validDates.push(sorted[i].dateStr);
        sourcePoints.push(sorted[i]);
    }

    return {
        inputs, // Shape: [batch, SEQ_LENGTH, 3]
        labels, // Shape: [batch, 1]
        validDates,
        sourcePoints,
        excMin,
        excMax,
        rvolMin: normRVols.length > 0 ? Math.min(...rawRVols) : 0,
        rvolMax: normRVols.length > 0 ? Math.max(...rawRVols) : 1,
        // Save the very last historical sequence [pastRVol, pastExc] so we have a base to append 'today's' manual input to for inference without retraining
        lastSequence: inputs.length > 0 ? inputs[inputs.length - 1].map(step => [step[0], step[1]]) : []
    };
};

/**
 * Constructs the LSTM Model
 */
export const createLstmModel = (inputShape) => {
    const model = tf.sequential();

    // LSTM Layer
    model.add(tf.layers.lstm({
        units: 32,
        returnSequences: false,
        inputShape: inputShape // e.g. [SEQ_LENGTH, 3]
    }));

    // Dropout for regularization
    model.add(tf.layers.dropout({ rate: 0.2 }));

    // Hidden Dense
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

    // Linear output for regression
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    return model;
};

/**
 * Trains the model and yields progress so the UI doesn't freeze.
 */
export async function* trainModelGenerator(model, xs, ys, epochs = 50) {
    let currentEpoch = 0;
    let currentLoss = 0;

    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                currentEpoch = epoch + 1;
                currentLoss = logs.loss;
            }
        },
        // yieldEvery is required to not hang the browser
        yieldEvery: 'epoch'
    });

    // Final yield when complete
    yield { epoch: currentEpoch, loss: currentLoss, status: 'complete' };
}

/**
 * Computes the Mean Squared Error against standard scale data
 */
export const calculateMSE = (predictions, actuals) => {
    if (predictions.length === 0 || predictions.length !== actuals.length) return null;
    let sumSquared = 0;
    for (let i = 0; i < predictions.length; i++) {
        sumSquared += Math.pow(predictions[i] - actuals[i], 2);
    }
    return sumSquared / predictions.length;
};

/**
 * Wrapper to run full train and predict pipeline.
 */
export const runTfjsPipeline = async (tickerData, epochs = 50, onProgress = null) => {
    const prepared = prepareTfjsData(tickerData);
    if (!prepared) return null;

    const xs = tf.tensor3d(prepared.inputs);
    const ys = tf.tensor2d(prepared.labels, [prepared.labels.length, 1]);

    const model = createLstmModel([SEQ_LENGTH, 3]);

    // Train with non-blocking generator loop
    let finalLoss = 0;
    let currentEpoch = 0;

    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                currentEpoch = epoch + 1;
                finalLoss = logs.loss;
                if (onProgress) {
                    onProgress(currentEpoch, epochs, logs.loss);
                }
                await tf.nextFrame(); // Let the UI render
            }
        }
    });

    // Predict
    const predsTensor = model.predict(xs);
    const predsArray = await predsTensor.data();

    // Denormalize predictions
    const denormPreds = Array.from(predsArray).map(v =>
        denormalizeValue(v, prepared.excMin, prepared.excMax)
    );

    // Denormalize actuals for MSE
    const actuals = prepared.labels.map(v => denormalizeValue(v, prepared.excMin, prepared.excMax));

    const mse = calculateMSE(denormPreds, actuals);

    // Cleanup tensors to prevent memory leaks in the browser
    xs.dispose();
    ys.dispose();
    predsTensor.dispose();
    // We keep the model in memory if we wanted to predict live, 
    // but for this dashboard we return the mapped predictions to graph statically.

    // Zip the original dates/points with their LSTM prediction so App.jsx can graph them easily
    const predictionsMap = prepared.sourcePoints.map((pt, idx) => ({
        ...pt,
        lstmPredictedExc: denormPreds[idx]
    }));

    // Auto-save the freshly trained model to IndexedDB AND the Project Folder automatically!
    await saveModelToStorage(model, prepared);

    return {
        mse,
        predictionsMap,
        finalLoss,
        model, // return it in case we optionally want to run inference elsewhere
        preparedData: prepared // return min max boundaries for future manual inference mapping
    };
};

/**
 * Runs inference only using an already trained model on newly fetched/processed data.
 */
export const evaluateLstmOnData = async (model, preparedDataMeta, tickerData) => {
    // We recreate the prepared sequences but we MUST use the originally trained min/max 
    // to normalize the new data exactly the way the model expects.
    if (!tickerData || tickerData.length <= SEQ_LENGTH) return null;

    const sorted = [...tickerData].sort((a, b) => new Date(a.date) - new Date(b.date));
    const rawRVols = sorted.map(d => d.rVol);
    const rawExcursions = sorted.map(d => d.maxExcursionAdr);

    // Normalize using existing meta boundaries!
    const normRVols = rawRVols.map(v => (v - preparedDataMeta.rvolMin) / (preparedDataMeta.rvolMax - preparedDataMeta.rvolMin));
    const normExcursions = rawExcursions.map(v => (v - preparedDataMeta.excMin) / (preparedDataMeta.excMax - preparedDataMeta.excMin));

    const inputs = [];
    for (let i = SEQ_LENGTH; i < sorted.length; i++) {
        const sequence = [];
        for (let j = i - SEQ_LENGTH; j < i; j++) {
            sequence.push([normRVols[j], normExcursions[j]]);
        }
        const seqWithToday = sequence.map((step, idx) => {
            if (idx === SEQ_LENGTH - 1) return [step[0], step[1], normRVols[i]];
            return [step[0], step[1], 0];
        });
        inputs.push(seqWithToday);
    }

    // Also update lastSequence
    preparedDataMeta.lastSequence = inputs.length > 0 ? inputs[inputs.length - 1].map(step => [step[0], step[1]]) : preparedDataMeta.lastSequence;

    const xs = tf.tensor3d(inputs);
    const predsTensor = model.predict(xs);
    const predsArray = await predsTensor.data();

    const denormPreds = Array.from(predsArray).map(v => denormalizeValue(v, preparedDataMeta.excMin, preparedDataMeta.excMax));

    xs.dispose();
    predsTensor.dispose();

    // Map predictions to the slice of points that were passed (offset by SEQ_LENGTH)
    const validPoints = sorted.slice(SEQ_LENGTH);
    const predictionsMap = validPoints.map((pt, idx) => ({
        ...pt,
        lstmPredictedExc: denormPreds[idx] || 0
    }));

    // If finalLoss is missing since it wasn't just trained, we can mock it
    return {
        mse: 0,
        predictionsMap,
        finalLoss: 0,
        model,
        preparedData: preparedDataMeta
    };
};

/**
 * Saves a trained model and its normalization metadata to the browser's IndexedDB.
 * AND sends it to the backend server to save it permanently in public/models/
 */
export const saveModelToStorage = async (model, preparedData) => {
    try {
        // 1. Save locally to browser IndexedDB
        await model.save('indexeddb://stock-lstm-model');
        localStorage.setItem('stock-lstm-meta', JSON.stringify(preparedData));

        // 2. Fetch the newly saved model from IndexedDB as a Blob so we can send it to our backend
        // (TensorFlow.js doesn't easily return binary buffers directly from model.save('localstorage'), 
        //  but we can use model.save(tf.io.withSaveHandler(...)) for a custom route to backend)
        await model.save(tf.io.withSaveHandler(async (artifacts) => {
            const formData = new FormData();

            // Appends model.json
            const modelTopologyAndWeightManifest = {
                modelTopology: artifacts.modelTopology,
                format: artifacts.format,
                generatedBy: artifacts.generatedBy,
                convertedBy: artifacts.convertedBy,
                weightsManifest: artifacts.weightsManifest
            };
            const jsonBlob = new Blob([JSON.stringify(modelTopologyAndWeightManifest)], { type: 'application/json' });
            formData.append('modelJson', jsonBlob, 'stock-lstm-model.json');

            // Appends model.weights.bin
            const weightData = artifacts.weightData;
            const weightBlob = new Blob([weightData], { type: 'application/octet-stream' });
            formData.append('modelWeights', weightBlob, 'stock-lstm-model.weights.bin');

            // Appends metadata
            formData.append('metadata', JSON.stringify(preparedData, null, 2));

            // POST to backend securely
            const response = await fetch('/api/save-model', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                console.error("Backend failed to save the model project files.");
            } else {
                console.log("Model successfully backed up inside the project folder: public/models/");
            }
            return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyBytes: jsonBlob.size, weightDataBytes: weightBlob.size } };
        }));

        return true;
    } catch (e) {
        console.error("Failed to save model to indexeddb or backend", e);
        return false;
    }
};

/**
 * Attempts to load the existing saved model from IndexedDB.
 */
export const loadModelFromStorage = async () => {
    try {
        const metaStr = localStorage.getItem('stock-lstm-meta');
        if (!metaStr) return null;

        const preparedData = JSON.parse(metaStr);
        // Ensure tf is fully loaded before doing this
        const model = await tf.loadLayersModel('indexeddb://stock-lstm-model');
        return { model, preparedData };
    } catch (e) {
        console.warn("No valid model found in indexeddb, training will be required.");
        return null;
    }
};

/**
 * Downloads the model's weights & biases (model.json + .bin) and its metadata (JSON)
 * directly to the user's hard drive so they can be ported to other apps.
 */
export const downloadModelFiles = async (model, preparedData) => {
    try {
        await model.save('downloads://stock-lstm-model');

        // Also trigger download of metadata json
        const metadataStr = JSON.stringify(preparedData, null, 2);
        const blob = new Blob([metadataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'stock-lstm-meta.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (e) {
        console.error("Error downloading model files", e);
    }
};

/**
 * Loads the model directly from manually uploaded local files.
 * Expects exactly: model.json, weights.bin, and stock-lstm-meta.json
 */
export const loadModelFromFiles = async (modelFile, weightsFile, metaFile) => {
    try {
        const metaText = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = e => reject(e);
            reader.readAsText(metaFile);
        });
        const preparedData = JSON.parse(metaText);
        const model = await tf.loadLayersModel(tf.io.browserFiles([modelFile, weightsFile]));

        // After loading a custom file, let's cache it globally to IndexedDB
        await saveModelToStorage(model, preparedData);

        return { model, preparedData };
    } catch (e) {
        console.error("Error loading model from files", e);
        throw e;
    }
};

/**
 * Saves the model directly to the local node server project folder instead of downloading to browser.
 */
export const saveModelToServer = async (model, preparedData, modelType = 'daily') => {
    try {
        await model.save(tf.io.withSaveHandler(async (artifacts) => {
            const formData = new FormData();

            const modelJsonBlob = new Blob([JSON.stringify({
                modelTopology: artifacts.modelTopology,
                format: artifacts.format,
                generatedBy: artifacts.generatedBy,
                convertedBy: artifacts.convertedBy,
                weightsManifest: artifacts.weightSpecs ? [{
                    paths: ['weights.bin'],
                    weights: artifacts.weightSpecs
                }] : []
            })], { type: 'application/json' });

            formData.append('modelJson', modelJsonBlob, 'model.json');

            const weightsBlob = new Blob([artifacts.weightData], { type: 'application/octet-stream' });
            formData.append('modelWeights', weightsBlob, 'weights.bin');

            formData.append('metadata', JSON.stringify(preparedData));

            let endpoint = '/api/save-model';
            if (modelType === 'intraday') endpoint = '/api/save-intraday-model';
            else if (modelType === 'maxExcursion') endpoint = '/api/save-max-excursion-model';

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || "Failed to save model to server");
            }
            return { modelArtifactsInfo: artifacts };
        }));
        return { success: true };
    } catch (e) {
        console.error("Error saving model to server:", e);
        return { success: false, error: e.message };
    }
};

// --- INTRADAY RVol PREDICTION LSTM ---

/**
 * Prepares sequences from the flat array of processed data points.
 * intradayDataByDate: from backend /api/intraday-data
 * processedYfData: chartData inside App.jsx containing existing RVol features
 */
export const prepareIntradayData = (intradayDataByDate, processedYfData, minutesToUse = 30) => {
    const maxBars = Math.floor(minutesToUse / 5);

    // Create a lookup for YF data by compound ticker & date string
    const yfMap = {};
    for (const d of processedYfData) {
        if (d.dateStr && d.avgVol > 0 && d.ticker) {
            yfMap[`${d.ticker}_${d.dateStr}`] = d;
        }
    }

    const inputs = [];
    const labels = [];
    const validDates = [];
    const sourcePoints = [];

    for (const day of intradayDataByDate) {
        if (!day.ticker) continue; // Safety check
        const yfDay = yfMap[`${day.ticker}_${day.date}`];
        if (!yfDay) continue;

        if (day.bars.length < maxBars) continue;

        const avgVol = yfDay.avgVol;
        const actualRvol = yfDay.rVol;

        const sequence = [];
        for (let i = 0; i < maxBars; i++) {
            // Feature: Volume of this 5-min bar relative to the 50-day Daily Average Volume
            sequence.push([day.bars[i].volume / avgVol]);
        }

        inputs.push(sequence);
        labels.push(actualRvol);
        validDates.push(day.date);
        sourcePoints.push({
            dateStr: day.date,
            actualRvol: actualRvol,
            barsUtilized: maxBars,
            avgVol: avgVol,
            ticker: day.ticker
        });
    }

    if (inputs.length === 0) return null;

    if (inputs.length === 0) return null;

    // Remove global min-max normalization, since volume/avgVol is natively self-normalized
    // and scaling globally crushes volatility across multi-stock arrays!

    return {
        inputs: inputs,
        labels: labels,
        validDates,
        sourcePoints,
        rawRVols: labels
    };
};

export const createIntradayLstmModel = (sequenceLength) => {
    const model = tf.sequential();
    model.add(tf.layers.lstm({
        units: 16,
        returnSequences: false,
        inputShape: [sequenceLength, 1]
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    return model;
};

export const runIntradayTfjsPipeline = async (intradayDataByDate, processedYfData, minutesToUse = 30, epochs = 50, onProgress = null) => {
    const prepared = prepareIntradayData(intradayDataByDate, processedYfData, minutesToUse);
    if (!prepared) return null;

    const sequenceLength = Math.floor(minutesToUse / 5);
    const xs = tf.tensor3d(prepared.inputs);
    const ys = tf.tensor2d(prepared.labels, [prepared.labels.length, 1]);

    const model = createIntradayLstmModel(sequenceLength);

    let finalLoss = 0;
    let currentEpoch = 0;

    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: 16,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                currentEpoch = epoch + 1;
                finalLoss = logs.loss;
                if (onProgress) {
                    onProgress(currentEpoch, epochs, logs.loss);
                }
                await tf.nextFrame();
            }
        }
    });

    const predsTensor = model.predict(xs);
    const predsArray = await predsTensor.data();

    // Predictions are raw native RVol estimates
    const rawPreds = Array.from(predsArray);
    const actuals = prepared.rawRVols;

    // calculate MSE
    let sumSquared = 0;
    for (let i = 0; i < rawPreds.length; i++) {
        sumSquared += Math.pow(rawPreds[i] - actuals[i], 2);
    }
    const mse = sumSquared / rawPreds.length;

    xs.dispose();
    ys.dispose();
    predsTensor.dispose();

    const predictionsMap = prepared.sourcePoints.map((pt, idx) => ({
        ...pt,
        predictedRvol: rawPreds[idx],
        actualVol: pt.actualRvol * pt.avgVol,
        predictedVol: rawPreds[idx] * pt.avgVol
    }));

    // Calculate R^2 and Standard Deviation for Absolute Volumes
    let meanActualVol = 0;
    predictionsMap.forEach(p => meanActualVol += p.actualVol);
    meanActualVol /= predictionsMap.length;

    let ssTot = 0;
    let ssRes = 0;
    let errors = [];

    predictionsMap.forEach(p => {
        ssTot += Math.pow(p.actualVol - meanActualVol, 2);
        const err = p.predictedVol - p.actualVol;
        ssRes += Math.pow(err, 2);
        errors.push(err);
    });

    const rSquared = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;

    const meanError = errors.reduce((sum, e) => sum + e, 0) / errors.length;
    const squaredDiffs = errors.map(e => Math.pow(e - meanError, 2));
    const stdDev = Math.sqrt(squaredDiffs.reduce((sum, sq) => sum + sq, 0) / errors.length);

    return {
        mse,
        rSquared,
        stdDev,
        predictionsMap,
        finalLoss,
        model,
        preparedData: prepared
    };
};

export const evaluateIntradayModel = async (model, preparedDataMeta, manualVolumes, avgVol) => {
    let sequence = manualVolumes.map(vol => [vol / avgVol]);

    // We expect inputShape: [1, sequenceLength, 1]
    const xs = tf.tensor3d([sequence]);
    const predsTensor = model.predict(xs);
    const predsArray = await predsTensor.data();

    let rawPred = predsArray[0];

    xs.dispose();
    predsTensor.dispose();

    return rawPred > 0 ? rawPred : 0;
};

// --- MAX EXCURSION PREDICTION PIPELINE ---

const MAX_EXC_FEATURE_NAMES = [
    'Projected RVol',
    '% Change (Prev Day)',
    'First Bar Vol / Avg Vol',
    'First 5-min Range / ADR',
    '% Above Open after 5 min',
    'Up/Down Vol Ratio (early)',
    '20-day ADR',
    'ATR(14)',
    'ATR Dist from 10 EMA',
    'ATR Dist from 20 EMA',
    'ATR Dist from 50 EMA'
];

/**
 * Prepares feature vectors for max excursion prediction.
 * @param {Array} intradayDataByDate - From /api/intraday-data (with OHLCV bars)
 * @param {Array} dailyFeatures - From /api/intraday-features
 * @param {number} minutesToUse - How many minutes of early data to use (default 5)
 * @returns {Object|null} { inputs, labels, featureNames, sourcePoints, normParams }
 */
export const prepareMaxExcursionData = (intradayDataByDate, dailyFeatures, minutesToUse = 5, featureMask = null) => {
    const maxBars = Math.floor(minutesToUse / 5);

    // Build lookup for daily features by date
    const featMap = {};
    for (const f of dailyFeatures) {
        featMap[f.date] = f;
    }

    const inputs = [];
    const labels = [];
    const sourcePoints = [];

    for (const day of intradayDataByDate) {
        const feat = featMap[day.date];
        if (!feat) continue;
        if (!day.bars || day.bars.length < maxBars) continue;
        if (!day.dayOpen || !day.dayHigh || !day.dayLow) continue;
        if (feat.adr20 <= 0 || feat.avgVol50 <= 0) continue;

        // --- Extract early-session features from first N bars ---
        const earlyBars = day.bars.slice(0, maxBars);

        // 1. Projected RVol using U-shaped cumulative volume profile
        const earlyVolSum = earlyBars.reduce((s, b) => s + b.volume, 0);
        const expectedPct = getExpectedCumulativeVolumePercentage(minutesToUse);
        const projectedDayVol = earlyVolSum / expectedPct;
        const projectedRVol = projectedDayVol / feat.avgVol50;

        // 2. % Change prev day
        const prevCloseChange = feat.prevCloseChange || 0;

        // 3. First bar volume / avg vol
        const firstBarVolRatio = earlyBars[0].volume / feat.avgVol50;

        // 4. First 5-min range / ADR
        let earlyHigh = -Infinity, earlyLow = Infinity;
        for (const b of earlyBars) {
            if (b.high > earlyHigh) earlyHigh = b.high;
            if (b.low < earlyLow) earlyLow = b.low;
        }
        const earlyRangeOverAdr = feat.adr20 > 0 ? (earlyHigh - earlyLow) / feat.adr20 : 0;

        // 5. % above open after first N bars
        const lastEarlyClose = earlyBars[earlyBars.length - 1].close;
        const pctAboveOpen = day.dayOpen > 0 ? ((lastEarlyClose - day.dayOpen) / day.dayOpen) * 100 : 0;

        // 6. Up/Down volume ratio in early bars
        let upVol = 0, downVol = 0;
        for (const b of earlyBars) {
            if (b.close >= b.open) upVol += b.volume;
            else downVol += b.volume;
        }
        const upDownRatio = (downVol > 0) ? upVol / downVol : (upVol > 0 ? 10 : 1);

        // 7. Normalized ADR (as fraction of price)
        const normAdr = day.dayOpen > 0 ? feat.adr20 / day.dayOpen : 0;

        // 8. Normalized ATR (as fraction of price)
        const normAtr = day.dayOpen > 0 ? feat.atr14 / day.dayOpen : 0;

        // 9-11. ATR distances from EMAs
        const atrDistEma10 = feat.atrDistEma10 || 0;
        const atrDistEma20 = feat.atrDistEma20 || 0;
        const atrDistEma50 = feat.atrDistEma50 || 0;

        const featureVector = [
            projectedRVol,
            prevCloseChange,
            firstBarVolRatio,
            earlyRangeOverAdr,
            pctAboveOpen,
            upDownRatio,
            normAdr,
            normAtr,
            atrDistEma10,
            atrDistEma20,
            atrDistEma50
        ];

        // --- Label: actual max excursion from open as multiple of ADR ---
        const highExcursion = Math.abs(day.dayHigh - day.dayOpen);
        const lowExcursion = Math.abs(day.dayLow - day.dayOpen);
        const maxExcursion = Math.max(highExcursion, lowExcursion);
        const maxExcursionAdr = feat.adr20 > 0 ? maxExcursion / feat.adr20 : 0;

        // Filter outliers
        if (maxExcursionAdr > 10 || projectedRVol > 20) continue;

        inputs.push(featureVector);
        labels.push(maxExcursionAdr);
        sourcePoints.push({
            date: day.date,
            ticker: day.ticker || '',
            dayOpen: day.dayOpen,
            dayHigh: day.dayHigh,
            dayLow: day.dayLow,
            dayClose: day.dayClose,
            actualMaxExc: maxExcursionAdr,
            adr20: feat.adr20,
            projectedRVol
        });
    }

    if (inputs.length < 10) return null;

    // Apply feature mask — drop disabled features from every vector
    const activeMask = featureMask || MAX_EXC_FEATURE_NAMES.map(() => true);
    const activeIndices = activeMask.map((v, i) => v ? i : -1).filter(i => i >= 0);
    const filteredInputs = inputs.map(row => activeIndices.map(i => row[i]));
    const filteredNames = activeIndices.map(i => MAX_EXC_FEATURE_NAMES[i]);

    // Normalize each feature column independently
    const numFeatures = filteredInputs[0].length;
    const featureMins = new Array(numFeatures);
    const featureMaxs = new Array(numFeatures);

    for (let f = 0; f < numFeatures; f++) {
        const col = filteredInputs.map(row => row[f]);
        featureMins[f] = Math.min(...col);
        featureMaxs[f] = Math.max(...col);
    }

    const normalizedInputs = filteredInputs.map(row =>
        row.map((val, f) => {
            const range = featureMaxs[f] - featureMins[f];
            return range > 0 ? (val - featureMins[f]) / range : 0;
        })
    );

    // Normalize labels
    const labelMin = Math.min(...labels);
    const labelMax = Math.max(...labels);
    const normalizedLabels = labels.map(v => {
        const range = labelMax - labelMin;
        return range > 0 ? (v - labelMin) / range : 0;
    });

    return {
        inputs: normalizedInputs,
        rawInputs: filteredInputs,
        labels: normalizedLabels,
        rawLabels: labels,
        featureNames: filteredNames,
        allFeatureNames: MAX_EXC_FEATURE_NAMES,
        featureMask: activeMask,
        activeIndices,
        sourcePoints,
        normParams: { featureMins, featureMaxs, labelMin, labelMax }
    };
};

/**
 * Creates a Dense neural network for max excursion prediction.
 */
export const createMaxExcursionModel = (numFeatures) => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [numFeatures] }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
        optimizer: tf.train.adam(0.005),
        loss: 'meanSquaredError'
    });

    return model;
};

/**
 * Full train → predict → evaluate pipeline for max excursion.
 */
export const runMaxExcursionPipeline = async (intradayData, dailyFeatures, minutesToUse = 5, epochs = 80, onProgress = null, featureMask = null) => {
    const prepared = prepareMaxExcursionData(intradayData, dailyFeatures, minutesToUse, featureMask);
    if (!prepared) return null;

    const numFeatures = prepared.inputs[0].length;
    const xs = tf.tensor2d(prepared.inputs);
    const ys = tf.tensor2d(prepared.labels, [prepared.labels.length, 1]);

    const model = createMaxExcursionModel(numFeatures);

    let finalLoss = 0;

    await model.fit(xs, ys, {
        epochs,
        batchSize: 32,
        shuffle: true,
        validationSplit: 0.15,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                finalLoss = logs.loss;
                if (onProgress) onProgress(epoch + 1, epochs, logs.loss);
                await tf.nextFrame();
            }
        }
    });

    // Predict
    const predsTensor = model.predict(xs);
    const predsArray = await predsTensor.data();

    // Denormalize predictions and actuals
    const { labelMin, labelMax } = prepared.normParams;
    const denormPreds = Array.from(predsArray).map(v => v * (labelMax - labelMin) + labelMin);
    const actuals = prepared.rawLabels;

    // Calculate R² and MSE
    const meanActual = actuals.reduce((s, v) => s + v, 0) / actuals.length;
    let ssTot = 0, ssRes = 0, sumSqErr = 0;
    const errors = [];
    for (let i = 0; i < actuals.length; i++) {
        ssTot += Math.pow(actuals[i] - meanActual, 2);
        const err = denormPreds[i] - actuals[i];
        ssRes += Math.pow(err, 2);
        sumSqErr += Math.pow(err, 2);
        errors.push(err);
    }
    const rSquared = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
    const mse = sumSqErr / actuals.length;

    const meanError = errors.reduce((s, e) => s + e, 0) / errors.length;
    const stdDev = Math.sqrt(errors.reduce((s, e) => s + Math.pow(e - meanError, 2), 0) / errors.length);

    // Build predictions map
    const predictionsMap = prepared.sourcePoints.map((pt, idx) => ({
        ...pt,
        predictedMaxExc: denormPreds[idx],
        predictedHigh: pt.dayOpen + denormPreds[idx] * pt.adr20,
        predictedLow: pt.dayOpen - denormPreds[idx] * pt.adr20
    }));

    // Per-ticker breakdown
    const tickerMap = {};
    for (const p of predictionsMap) {
        if (!tickerMap[p.ticker]) tickerMap[p.ticker] = { errors: [], count: 0 };
        tickerMap[p.ticker].errors.push(Math.abs(p.predictedMaxExc - p.actualMaxExc));
        tickerMap[p.ticker].count++;
    }
    const perTickerStats = Object.entries(tickerMap).map(([ticker, data]) => ({
        ticker,
        count: data.count,
        mae: data.errors.reduce((s, e) => s + e, 0) / data.errors.length
    })).sort((a, b) => a.mae - b.mae);

    xs.dispose();
    ys.dispose();
    predsTensor.dispose();

    return {
        mse,
        rSquared,
        stdDev,
        finalLoss,
        predictionsMap,
        perTickerStats,
        model,
        preparedData: prepared
    };
};

/**
 * Permutation-based sensitivity analysis.
 * Shuffles each feature independently and measures R² drop.
 */
export const runSensitivityAnalysis = async (model, preparedData) => {
    if (!preparedData || !preparedData.inputs || preparedData.inputs.length < 10) return [];

    const { inputs, rawLabels, normParams, featureNames } = preparedData;
    const { labelMin, labelMax } = normParams;

    // Get baseline R²
    const baseXs = tf.tensor2d(inputs);
    const basePreds = await model.predict(baseXs).data();
    baseXs.dispose();

    const actuals = rawLabels;
    const baseDenorm = Array.from(basePreds).map(v => v * (labelMax - labelMin) + labelMin);
    const meanActual = actuals.reduce((s, v) => s + v, 0) / actuals.length;

    let baseSsTot = 0, baseSsRes = 0;
    for (let i = 0; i < actuals.length; i++) {
        baseSsTot += Math.pow(actuals[i] - meanActual, 2);
        baseSsRes += Math.pow(baseDenorm[i] - actuals[i], 2);
    }
    const baseR2 = baseSsTot > 0 ? 1 - (baseSsRes / baseSsTot) : 0;

    const results = [];
    const numFeatures = inputs[0].length;

    for (let f = 0; f < numFeatures; f++) {
        // Create shuffled copy of inputs
        const shuffled = inputs.map(row => [...row]);
        const colValues = shuffled.map(row => row[f]);

        // Fisher-Yates shuffle
        for (let i = colValues.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [colValues[i], colValues[j]] = [colValues[j], colValues[i]];
        }
        shuffled.forEach((row, idx) => { row[f] = colValues[idx]; });

        const shuffXs = tf.tensor2d(shuffled);
        const shuffPreds = await model.predict(shuffXs).data();
        shuffXs.dispose();

        const shuffDenorm = Array.from(shuffPreds).map(v => v * (labelMax - labelMin) + labelMin);
        let shuffSsRes = 0;
        for (let i = 0; i < actuals.length; i++) {
            shuffSsRes += Math.pow(shuffDenorm[i] - actuals[i], 2);
        }
        const shuffR2 = baseSsTot > 0 ? 1 - (shuffSsRes / baseSsTot) : 0;

        const importance = baseR2 - shuffR2; // Positive = feature helps

        // Compute correlation sign from raw inputs
        const rawCol = preparedData.rawInputs.map(row => row[f]);
        const meanFeat = rawCol.reduce((s, v) => s + v, 0) / rawCol.length;
        let covSum = 0;
        for (let i = 0; i < rawCol.length; i++) {
            covSum += (rawCol[i] - meanFeat) * (actuals[i] - meanActual);
        }

        results.push({
            featureName: featureNames[f] || `Feature ${f}`,
            importanceScore: importance,
            correlationSign: covSum > 0 ? 'positive' : 'negative'
        });
    }

    // Sort by importance descending
    results.sort((a, b) => b.importanceScore - a.importanceScore);
    return results;
};

/**
 * Manual prediction for max excursion given today's inputs.
 */
export const predictMaxExcursion = async (model, normParams, featureVector, activeIndices = null) => {
    const { featureMins, featureMaxs, labelMin, labelMax } = normParams;

    // Filter to only active features if mask was used during training
    const filtered = activeIndices ? activeIndices.map(i => featureVector[i]) : featureVector;

    // Normalize the feature vector using trained params
    const normalized = filtered.map((val, f) => {
        const range = featureMaxs[f] - featureMins[f];
        return range > 0 ? (val - featureMins[f]) / range : 0;
    });

    const xs = tf.tensor2d([normalized]);
    const predTensor = model.predict(xs);
    const predVal = (await predTensor.data())[0];

    xs.dispose();
    predTensor.dispose();

    // Denormalize
    const predictedExc = predVal * (labelMax - labelMin) + labelMin;
    return Math.max(0, predictedExc);
};

