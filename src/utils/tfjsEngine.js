import * as tf from '@tensorflow/tfjs';

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

    return {
        mse,
        predictionsMap,
        finalLoss,
        model, // return it in case we optionally want to run inference elsewhere
        preparedData: prepared // return min max boundaries for future manual inference mapping
    };
};

