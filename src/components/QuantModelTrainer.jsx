import React, { useState } from 'react';
import { trainQuantModel, predictFutureReturn } from '../utils/quantLstmModel';

export default function QuantModelTrainer({ backtestData, onApplyAiFormula }) {
    const [training, setTraining] = useState(false);
    const [targetFeature, setTargetFeature] = useState('ret1W');
    const [epochs, setEpochs] = useState(50);
    const [sequenceLength, setSequenceLength] = useState(5);
    const [currentEpoch, setCurrentEpoch] = useState(0);
    const [loss, setLoss] = useState(null);

    const [modelData, setModelData] = useState(null);

    const [manualInput, setManualInput] = useState({
        rsDelta: 1.5,
        vcp: 3,
        rVol: 2.5,
        priceChangeOverAdr: 1.5,
        episodicPivotPower: 3.75,
        ema10DistAtr: 1.2,
        ema20DistAtr: 2.5
    });
    const [predictionResult, setPredictionResult] = useState(null);

    // Derived Equation from sensitivity
    const generateEquation = (sensitivity) => {
        if (!sensitivity || sensitivity.length === 0) return '';
        const eqParts = sensitivity.map(item => `(${item.weight.toFixed(1)} * ${item.feature})`);
        return `Quant Score = ${eqParts.join(' + ')}`;
    };

    const handleTrain = async () => {
        if (!backtestData || backtestData.length < sequenceLength * 2) {
            alert("Not enough backtest data available. Please run the backtest first.");
            return;
        }

        setTraining(true);
        setCurrentEpoch(0);
        setLoss(null);
        setModelData(null);
        setPredictionResult(null);

        try {
            // Dynamic import for tf
            const tf = await import('@tensorflow/tfjs');

            // Re-format backtest data to simple array of objects
            // The trainQuantModel handles sorting by ticker inherently
            const result = await trainQuantModel(
                backtestData,
                targetFeature,
                parseInt(sequenceLength),
                parseInt(epochs),
                (ep, tot, currentLoss) => {
                    setCurrentEpoch(ep);
                    setLoss(currentLoss.toFixed(4));
                });

            setModelData(result);
        } catch (e) {
            console.error(e);
            alert("Failed to train model: " + e.message);
        }
        setTraining(false);
    };

    const handlePredict = () => {
        if (!modelData || !modelData.model) return;

        // Since the model needs a "sequence" (e.g. 5 days of history),
        // for manual entry we'll just duplicate the single input 5 times 
        // to represent a "steady state" of these parameters over the past week.
        const mockHistory = [];
        for (let i = 0; i < modelData.sequenceLength; i++) {
            mockHistory.push(manualInput);
        }

        try {
            const pred = predictFutureReturn(
                modelData.model,
                modelData.stats,
                modelData.featureKeys,
                mockHistory
            );
            setPredictionResult(pred);
        } catch (e) {
            console.error(e);
            alert("Prediction failed");
        }
    };

    return (
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mt-6 w-full text-white">
            <h3 className="text-xl font-bold mb-4 text-emerald-400">LSTM Predictive Model</h3>

            <div className="flex flex-col md:flex-row gap-8">
                {/* Training Controls */}
                <div className="w-full md:w-1/2 space-y-4">
                    <h4 className="font-semibold text-lg border-b border-gray-700 pb-2">Train Model</h4>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Target Predictor</label>
                            <select
                                value={targetFeature}
                                onChange={e => setTargetFeature(e.target.value)}
                                className="w-full bg-gray-700 rounded px-2 py-2 border border-gray-600 focus:outline-none focus:border-emerald-500"
                                disabled={training}
                            >
                                <option value="ret1W">1-Week Forward Return (%)</option>
                                <option value="ret2W">2-Week Forward Return (%)</option>
                                <option value="ret1M">1-Month Forward Return (%)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Epochs</label>
                            <input
                                type="number"
                                value={epochs}
                                onChange={e => setEpochs(e.target.value)}
                                className="w-full bg-gray-700 rounded px-2 py-2 border border-gray-600"
                                disabled={training}
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Sequence Length (Days)</label>
                            <input
                                type="number"
                                value={sequenceLength}
                                onChange={e => setSequenceLength(e.target.value)}
                                className="w-full bg-gray-700 rounded px-2 py-2 border border-gray-600"
                                disabled={training}
                            />
                        </div>
                    </div>

                    <button
                        onClick={handleTrain}
                        disabled={training || !backtestData || backtestData.length === 0}
                        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-3 rounded text-white font-bold shadow-md transition-colors mt-4"
                    >
                        {training ? `Training... Epoch ${currentEpoch}/${epochs}` : 'Build & Train LSTM Model'}
                    </button>

                    {loss && (
                        <div className="bg-gray-900 border border-emerald-900 p-3 rounded mt-2 text-center text-sm text-emerald-300">
                            Training completed at Epoch {modelData?.finalEpoch || epochs}.<br />
                            Final Validation Loss: <span className="font-bold">{loss}</span>
                        </div>
                    )}

                    {/* Model Metrics */}
                    {modelData && typeof modelData.mae !== 'undefined' && (
                        <div className="bg-gray-700 p-4 rounded mt-4 border border-gray-600 flex justify-between items-center">
                            <div>
                                <p className="text-xs text-gray-400">Mean Abs Error (MAE)</p>
                                <p className="text-lg font-bold text-emerald-400">{modelData.mae.toFixed(2)}%</p>
                            </div>
                            <div className="border-l border-gray-600 pl-4">
                                <p className="text-xs text-gray-400">Accuracy (R²)</p>
                                <p className="text-lg font-bold text-blue-400">{typeof modelData.rSquared === 'number' ? (modelData.rSquared * 100).toFixed(2) + '%' : 'N/A'}</p>
                            </div>
                        </div>
                    )}

                    {/* Sensitivity Analysis Box */}
                    {modelData && modelData.sensitivity && (
                        <div className="bg-gray-700 p-4 rounded mt-4 border border-gray-600">
                            <h5 className="font-bold text-emerald-400 mb-2">Parameter Sensitivity Analysis</h5>
                            <p className="text-xs text-gray-400 mb-3">Relative impact of each feature on predicting accuracy (determined by individually masking features during validation).</p>

                            <div className="space-y-2 mb-4">
                                {modelData.sensitivity.map((item, idx) => (
                                    <div key={item.feature} className="flex items-center text-sm">
                                        <div className="w-32 truncate text-gray-300">{item.feature}</div>
                                        <div className="flex-1 bg-gray-600 h-2 rounded ml-2 overflow-hidden">
                                            <div className="bg-emerald-500 h-full" style={{ width: `${item.weight}%` }}></div>
                                        </div>
                                        <div className="w-12 text-right ml-2 font-mono text-emerald-300">{item.weight.toFixed(1)}%</div>
                                    </div>
                                ))}
                            </div>

                            <div className="bg-gray-800 p-3 rounded border border-gray-600">
                                <h6 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Optimum Equation Formula</h6>
                                <p className="font-mono text-sm text-yellow-300 break-words mb-2">
                                    {generateEquation(modelData.sensitivity)}
                                </p>
                                <button
                                    onClick={() => onApplyAiFormula(modelData.sensitivity, modelData.stats)}
                                    className="text-xs bg-emerald-600 hover:bg-emerald-500 px-2 py-1 rounded font-bold transition-colors"
                                >
                                    Apply Weights as AI Score
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Prediction Output */}
                <div className="w-full md:w-1/2 space-y-4">
                    <h4 className="font-semibold text-lg border-b border-gray-700 pb-2">Manual Inference</h4>
                    <p className="text-sm text-gray-400">
                        Enter current parameter values to predict the {targetFeature === 'ret1W' ? '1-Week' : targetFeature === 'ret2W' ? '2-Week' : '1-Month'} forward percentage return using the trained logic.
                    </p>

                    <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                        {Object.keys(manualInput).map(key => (
                            <div key={key}>
                                <label className="block text-xs text-gray-400 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={manualInput[key]}
                                    onChange={e => setManualInput({ ...manualInput, [key]: parseFloat(e.target.value) })}
                                    className="w-full bg-gray-700 rounded px-2 py-1 text-sm border border-gray-600"
                                    disabled={!modelData}
                                />
                            </div>
                        ))}
                    </div>

                    <button
                        onClick={handlePredict}
                        disabled={!modelData}
                        className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-600 px-4 py-3 rounded text-white font-bold shadow-md transition-colors mt-4"
                    >
                        Predict Return
                    </button>

                    {predictionResult !== null && (
                        <div className={`p-4 rounded text-center text-2xl font-black mt-2 ${predictionResult >= 0 ? 'bg-green-900/40 text-green-400 border border-green-800' : 'bg-red-900/40 text-red-400 border border-red-800'}`}>
                            {predictionResult > 0 ? '+' : ''}{predictionResult.toFixed(2)}%
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
