import * as tf from '@tensorflow/tfjs';
import fs from 'fs';

const oldLog = console.log;
console.log = function () { }; // mute banner

try {
    const data = [[[1], [2], [3]], [[2], [3], [4]]];
    tf.tensor3d(data, [2, 3, 1]);
    fs.writeFileSync('out.txt', 'TEST 1: OK\n', { flag: 'a' });
} catch (e) {
    fs.writeFileSync('out.txt', 'TEST 1 error: ' + e.message + '\n', { flag: 'a' });
}

try {
    const data2 = [[[1], [2], [3]], [[2], [3]]];
    tf.tensor3d(data2);
    fs.writeFileSync('out.txt', 'TEST 2: OK\n', { flag: 'a' });
} catch (e) {
    fs.writeFileSync('out.txt', 'TEST 2 error: ' + e.message + '\n', { flag: 'a' });
}

try {
    const y = [1, 2];
    tf.tensor2d(y, [2, 1]);
    fs.writeFileSync('out.txt', 'TEST 3: OK\n', { flag: 'a' });
} catch (e) {
    fs.writeFileSync('out.txt', 'TEST 3 error: ' + e.message + '\n', { flag: 'a' });
}
