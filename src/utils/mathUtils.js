/**
 * mathUtils.js - Advanced Regression Calculators
 */

/**
 * Calculates R-Squared given actual y values and predicted y values.
 */
const calculateRSquared = (actualY, predictedY) => {
    if (actualY.length !== predictedY.length || actualY.length < 2) return 0;
    const meanY = actualY.reduce((a, b) => a + b, 0) / actualY.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < actualY.length; i++) {
        ssTot += Math.pow(actualY[i] - meanY, 2);
        ssRes += Math.pow(actualY[i] - predictedY[i], 2);
    }
    if (ssTot === 0) return 0;
    const r2 = 1 - (ssRes / ssTot);
    return Math.max(0, r2); // Cap at 0 instead of negative for completely invalid models
};

/**
 * Linear Regression: y = mx + b
 */
const calcLinear = (data) => {
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    let n = data.length;

    data.forEach(d => {
        sumX += d.x;
        sumY += d.y;
        sumXY += d.x * d.y;
        sumXX += d.x * d.x;
    });

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const actualY = data.map(d => d.y);
    const predictedY = data.map(d => slope * d.x + intercept);
    const r2 = calculateRSquared(actualY, predictedY);

    return {
        type: 'Linear',
        equation: `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`,
        r2,
        predict: (x) => slope * x + intercept
    };
};

/**
 * Exponential Regression: y = a * e^(bx)
 * Solved via linear regression on ln(y) = ln(a) + bx
 */
const calcExponential = (data) => {
    let validData = data.filter(d => d.y > 0);
    if (validData.length < 2) return { r2: -1 }; // Cannot do exponential on <= 0

    let sumX = 0, sumLnY = 0, sumXLnY = 0, sumXX = 0;
    let n = validData.length;

    validData.forEach(d => {
        const lnY = Math.log(d.y);
        sumX += d.x;
        sumLnY += lnY;
        sumXLnY += d.x * lnY;
        sumXX += d.x * d.x;
    });

    const b = (n * sumXLnY - sumX * sumLnY) / (n * sumXX - sumX * sumX);
    const lnA = (sumLnY - b * sumX) / n;
    const a = Math.exp(lnA);

    const actualY = data.map(d => d.y);
    const predictedY = data.map(d => {
        if (d.y <= 0) return 0; // Model technically fails here but for dataset prediction accuracy we compare
        return a * Math.exp(b * d.x);
    });

    const r2 = calculateRSquared(actualY, predictedY);

    return {
        type: 'Exponential',
        equation: `y = ${a.toFixed(4)}e^(${b.toFixed(4)}x)`,
        r2,
        predict: (x) => a * Math.exp(b * x)
    };
};

/**
 * Gaussian Elimination to solve system of linear equations (Ax = B)
 */
const gaussianElimination = (matrix, vector) => {
    let n = matrix.length;
    let A = matrix.map((row, i) => [...row, vector[i]]);

    for (let i = 0; i < n; i++) {
        // Find pivot
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
                maxRow = k;
            }
        }

        // Swap rows
        let temp = A[i];
        A[i] = A[maxRow];
        A[maxRow] = temp;

        // Check for singular matrix
        if (Math.abs(A[i][i]) < 1e-10) {
            return null;
        }

        // Eliminate below
        for (let k = i + 1; k < n; k++) {
            let factor = A[k][i] / A[i][i];
            for (let j = i; j <= n; j++) {
                A[k][j] -= factor * A[i][j];
            }
        }
    }

    // Back substitution
    let x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        let sum = 0;
        for (let j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (A[i][n] - sum) / A[i][i];
    }

    return x;
};

/**
 * Generic Polynomial Regression of specified order
 */
const calcPolynomial = (data, order) => {
    let n = data.length;
    if (n <= order) return { r2: -1 }; // Need more points than parameters

    let sumX = new Array(2 * order + 1).fill(0);
    let sumXY = new Array(order + 1).fill(0);

    data.forEach(d => {
        let xPower = 1;
        for (let i = 0; i <= 2 * order; i++) {
            sumX[i] += xPower;
            if (i <= order) {
                sumXY[i] += xPower * d.y;
            }
            xPower *= d.x;
        }
    });

    let matrix = [];
    for (let i = 0; i <= order; i++) {
        let row = [];
        for (let j = 0; j <= order; j++) {
            row.push(sumX[i + j]);
        }
        matrix.push(row);
    }

    let coeffs = gaussianElimination(matrix, sumXY);
    if (!coeffs) return { r2: -1 }; // Singular

    const actualY = data.map(d => d.y);
    const predictedY = data.map(d => {
        let y = 0;
        let xPower = 1;
        for (let i = 0; i <= order; i++) {
            y += coeffs[i] * xPower;
            xPower *= d.x;
        }
        return y;
    });

    const r2 = calculateRSquared(actualY, predictedY);

    let eqParts = [];
    for (let i = order; i >= 0; i--) {
        let c = coeffs[i];
        if (Math.abs(c) < 1e-10) continue;

        let termStr = "";
        let absC = Math.abs(c);
        let absCStr = (absC > 0 && absC < 0.0001) ? absC.toExponential(4) : absC.toFixed(4);

        if (i === 0) {
            termStr = absCStr;
        } else if (i === 1) {
            termStr = `${absCStr}x`;
        } else {
            termStr = `${absCStr}x^${i}`;
        }

        if (eqParts.length === 0) {
            eqParts.push(c < 0 ? `-${termStr}` : termStr);
        } else {
            eqParts.push(c < 0 ? `- ${termStr}` : `+ ${termStr}`);
        }
    }

    const equation = "y = " + (eqParts.length > 0 ? eqParts.join(" ") : "0");

    let type = `Polynomial (${order}º)`;
    if (order === 2) type = 'Quadratic';
    if (order === 3) type = 'Cubic';
    if (order === 4) type = 'Quartic';

    return {
        type,
        equation,
        r2,
        predict: (x) => {
            let y = 0;
            let xPower = 1;
            for (let i = 0; i <= order; i++) {
                y += coeffs[i] * xPower;
                xPower *= x;
            }
            return y;
        }
    };
};

/**
 * Evaluates regression types (Linear, Exponential, Polynomials) and returns the one with the highest R-Squared value.
 * @param {Array} points - Array of {x, y}
 * @param {number} maxPolyOrder - The maximum polynomial order to test up to (default 2)
 */
export const findBestFitRegression = (points, maxPolyOrder = 2, minPolyOrder = 1, includeExponential = true) => {
    if (!points || points.length < 3) return null;

    // Filter valid numeric points
    const data = points.filter(p => p.x != null && p.y != null && isFinite(p.x) && isFinite(p.y));
    if (data.length < 3) return null;

    let best = null;

    if (includeExponential) {
        best = calcExponential(data);
    }

    // Test polynomial models from minPolyOrder up to maxPolyOrder
    for (let order = minPolyOrder; order <= maxPolyOrder; order++) {
        // Or if order is 1 we can just use calcPolynomial(data, 1) and that acts as linear
        const poly = calcPolynomial(data, order);
        if (!best || (poly && poly.r2 > best.r2)) {
            best = poly;
        }
    }

    // Default to a model even if r2 is negative (e.g. completely linear flat)
    if (!best || best.r2 === -1) {
        return calcPolynomial(data, minPolyOrder);
    }

    return best;
};
