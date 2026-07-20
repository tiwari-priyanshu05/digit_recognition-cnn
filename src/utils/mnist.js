// MNIST Dataset Loader Helper for TensorFlow.js
// Adapted from official TFJS MNIST examples.

const NUM_DATASET_ELEMENTS = 65000;
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

export class MnistData {
  constructor() {
    this.shuffledIndices = [];
    this.datasetImages = null;
    this.datasetLabels = null;
    this.trainIndices = [];
    this.testIndices = [];
  }

  async load(maxElements = 5000) {
    // Requesting maxElements (e.g. 5,000) limits the memory and keeps download fast
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    const imgLoaded = new Promise((resolve, reject) => {
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        // We only draw up to the elements we want to train/test on to save memory
        const numElements = Math.min(NUM_DATASET_ELEMENTS, maxElements);
        canvas.width = img.width;
        canvas.height = numElements;
        ctx.drawImage(img, 0, 0);

        const imgData = ctx.getImageData(0, 0, canvas.width, numElements);
        this.datasetImages = new Float32Array(numElements * IMAGE_SIZE);

        for (let i = 0; i < numElements; i++) {
          const startIdx = i * IMAGE_SIZE;
          for (let j = 0; j < IMAGE_SIZE; j++) {
            // Read grayscale values from red channel
            this.datasetImages[startIdx + j] = imgData.data[(startIdx + j) * 4] / 255.0;
          }
        }
        resolve();
      };
      img.onerror = () => reject(new Error('Failed to load MNIST sprite sheet.'));
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [_, labelsResponse] = await Promise.all([imgLoaded, labelsRequest]);
    
    const labelBuffer = await labelsResponse.arrayBuffer();
    const numElements = Math.min(NUM_DATASET_ELEMENTS, maxElements);
    this.datasetLabels = new Uint8Array(labelBuffer).slice(0, numElements * NUM_CLASSES);

    // Create shuffled indices
    this.shuffledIndices = Array.from({ length: numElements }, (_, i) => i);
    tf.util.shuffle(this.shuffledIndices);

    // Split into 80% train, 20% test
    const numTrain = Math.floor(numElements * 0.8);
    this.trainIndices = this.shuffledIndices.slice(0, numTrain);
    this.testIndices = this.shuffledIndices.slice(numTrain);
  }

  // Get next batch of training images and labels
  nextTrainBatch(batchSize) {
    return this.nextBatch(batchSize, this.datasetImages, this.datasetLabels, this.trainIndices);
  }

  // Get next batch of test images and labels
  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, this.datasetImages, this.datasetLabels, this.testIndices);
  }

  nextBatch(batchSize, dataImages, dataLabels, indexArray) {
    const card = indexArray.length;
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = indexArray[Math.floor(Math.random() * card)];
      
      const imageOffset = idx * IMAGE_SIZE;
      batchImagesArray.set(
        dataImages.subarray(imageOffset, imageOffset + IMAGE_SIZE),
        i * IMAGE_SIZE
      );

      const labelOffset = idx * NUM_CLASSES;
      batchLabelsArray.set(
        dataLabels.subarray(labelOffset, labelOffset + NUM_CLASSES),
        i * NUM_CLASSES
      );
    }

    const xs = tf.tensor3d(batchImagesArray, [batchSize, 28, 28]).expandDims(-1);
    const ys = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, ys };
  }
}
