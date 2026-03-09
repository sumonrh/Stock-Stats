import * as tf from '@tensorflow/tfjs';
import fs from 'fs';

let output = '';

try {
    const data = [[[1], [2]], [[2], [3], [4]]];
    tf.tensor3d(data);
    output += 'TEST 1 (ragged length): OK\n';
} catch (e) {
    output += 'TEST 1 error: ' + e.message + '\n';
}

try {
    const data = [[[1], [2]], [[2], [3]]];
    data[1][0] = undefined;
    tf.tensor3d(data);
    output += 'TEST 2 (undefined inside): OK\n';
} catch (e) {
    output += 'TEST 2 error: ' + e.message + '\n';
}

try {
    const data = [[[1], [2]], [[2], [NaN]]];
    tf.tensor3d(data);
    output += 'TEST 3 (NaN inside): OK\n';
} catch (e) {
    output += 'TEST 3 error: ' + e.message + '\n';
}

fs.writeFileSync('out2.txt', output);
