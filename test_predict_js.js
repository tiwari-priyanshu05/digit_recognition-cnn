import * as tf from '@tensorflow/tfjs';
import fs from 'fs';
import path from 'path';

console.log("Running TensorFlow.js forward pass test in Node...");

async function test() {
  // 1. Load weights and metadata
  const metaPath = path.resolve('public/model_metadata.json');
  const binPath = path.resolve('public/model_weights.bin');

  const metadata = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  const binBuffer = fs.readFileSync(binPath);
  // Convert Node Buffer to ArrayBuffer/Float32Array
  const weightsFloat32 = new Float32Array(binBuffer.buffer, binBuffer.byteOffset, binBuffer.byteLength / 4);

  // 2. Define model topology
  const tfModel = tf.sequential();
  
  tfModel.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 32,
    activation: 'relu',
    name: 'conv2d'
  }));

  tfModel.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
    name: 'max_pooling2d'
  }));

  tfModel.add(tf.layers.flatten({ name: 'flatten' }));

  tfModel.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    name: 'dense'
  }));

  tfModel.add(tf.layers.dense({
    units: 10,
    activation: 'softmax',
    name: 'dense_1'
  }));

  // 3. Extract and load weights
  let offset = 0;
  const getTensor = (meta) => {
    const slice = weightsFloat32.subarray(offset, offset + meta.size);
    offset += meta.size;
    return tf.tensor(slice, meta.shape, 'float32');
  };

  const convKernel = getTensor(metadata.conv2d_kernel);
  const convBias = getTensor(metadata.conv2d_bias);
  const denseKernel = getTensor(metadata.dense_kernel);
  const denseBias = getTensor(metadata.dense_bias);
  const outputKernel = getTensor(metadata.dense_1_kernel);
  const outputBias = getTensor(metadata.dense_1_bias);

  tfModel.setWeights([convKernel, convBias, denseKernel, denseBias, outputKernel, outputBias]);
  
  tfModel.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy'
  });

  // 4. Create deterministic diagonal input (1, 28, 28, 1)
  const inputPixels = new Float32Array(28 * 28);
  for (let i = 0; i < 28; i++) {
    inputPixels[i * 28 + i] = 1.0;
  }
  const inputTensor = tf.tensor4d(inputPixels, [1, 28, 28, 1]);

  // 5. Test intermediate activations
  const convLayer = tfModel.layers[0];
  const convModel = tf.model({ inputs: tfModel.inputs, outputs: convLayer.output });
  const convOut = convModel.predict(inputTensor);
  console.log("Conv2D output sum:", convOut.sum().dataSync()[0]);

  const poolLayer = tfModel.layers[1];
  const poolModel = tf.model({ inputs: tfModel.inputs, outputs: poolLayer.output });
  const poolOut = poolModel.predict(inputTensor);
  console.log("MaxPooling2D output sum:", poolOut.sum().dataSync()[0]);

  const denseLayer = tfModel.layers[3];
  const denseModel = tf.model({ inputs: tfModel.inputs, outputs: denseLayer.output });
  const denseOut = denseModel.predict(inputTensor);
  console.log("Dense output sum:", denseOut.sum().dataSync()[0]);

  // 6. Complete prediction
  const output = tfModel.predict(inputTensor);
  const probs = output.squeeze().arraySync();
  const predictedDigit = output.argMax(-1).dataSync()[0];

  console.log("\nTFJS Forward Pass Probabilities:");
  for (let digit = 0; digit < 10; digit++) {
    console.log(`Digit ${digit}: ${probs[digit].toFixed(6)}`);
  }
  console.log("Predicted Digit:", predictedDigit);
}

test();
