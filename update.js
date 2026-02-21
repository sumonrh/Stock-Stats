const fs = require('fs');
const file = 'c:/Users/sumon/Documents/App Development/Stock Stats/src/App.jsx';
let content = fs.readFileSync(file, 'utf8');
content = content.replace(
  /\{\/\* Chart Visualization \*\/\}[\s\S]*?\{\/\* Statistical Summary Table \*\/\}/,
  \{/* Chart Visualization */}
        {marginalPlotState && (
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <div className="mb-4 flex flex-col sm:flex-row sm:justify-between sm:items-end">
              <div>
                <h2 className="text-lg font-bold text-slate-800">Scatter Plot with Probability Distributions</h2>
                <p className="text-xs text-slate-500">Y-Axis: Absolute Max Excursion Multiple. X-Axis: Relative Volume.</p>
              </div>
              {regressionEquation && (
                <div className="mt-2 sm:mt-0 bg-blue-50 border border-blue-200 text-blue-800 text-xs px-3 py-1 rounded-full font-semibold">
                  <span className="font-mono">Relationship: {regressionEquation}</span>
                  <span className="ml-2 text-blue-600 block sm:inline">| SD: ±{regressionStdDev}</span>
                </div>
              )}
            </div>

            <div className="h-[500px] w-full">
              <Plot
                data={marginalPlotState.traces}
                layout={marginalPlotState.layout}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </div>
          </div>
        )}

        {/* Statistical Summary Table */}\
);
content = content.replace(
  /\{\/\* Probabilistic Distribution Bell Curve \*\/\}[\s\S]*?\{\/\* Selected Ticker Stock Chart \*\/\}/,
  \{/* Selected Ticker Stock Chart */}\
);
fs.writeFileSync(file, content);
console.log('App.jsx modified successfully!');
