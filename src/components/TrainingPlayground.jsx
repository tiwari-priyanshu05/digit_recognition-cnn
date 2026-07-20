import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { MnistData } from '../utils/mnist';
import { Play, Pause, RefreshCw, Award, Sliders, Database } from 'lucide-react';
import confetti from 'canvas-confetti';

const TrainingPlayground = ({ onModelUpdate }) => {
  const [dataLoader, setDataLoader] = useState(null);
  const [datasetStatus, setDatasetStatus] = useState('unloaded'); // unloaded, loading, loaded, error
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, training, paused, complete
  
  // Hyperparameters
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(128);
  const [learningRate, setLearningRate] = useState(0.005);
  
  // Progress state
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentBatch, setCurrentBatch] = useState(0);
  const [totalBatches, setTotalBatches] = useState(0);
  const [loss, setLoss] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  
  // Validation set predictions
  const [validationDigits, setValidationDigits] = useState([]);
  const [validationPredictions, setValidationPredictions] = useState([]);

  const lossHistoryRef = useRef([]);
  const accHistoryRef = useRef([]);
  const chartCanvasRef = useRef(null);
  const modelRef = useRef(null);
  const isTrainingRef = useRef(false);

  // Load the MNIST Dataset
  const handleLoadDataset = async () => {
    setDatasetStatus('loading');
    try {
      const loader = new MnistData();
      await loader.load(4000); // 4000 elements keeps download small and fast
      setDataLoader(loader);
      setDatasetStatus('loaded');
      
      // Load 10 random validation digits to display
      const { xs, ys } = loader.nextTestBatch(10);
      const imagesData = xs.arraySync();
      const labelsData = ys.argMax(-1).arraySync();
      
      const validationData = [];
      for (let i = 0; i < 10; i++) {
        // Flatten image
        const imgArray = [];
        for (let r = 0; r < 28; r++) {
          for (let c = 0; c < 28; c++) {
            imgArray.push(imagesData[i][r][c][0]);
          }
        }
        validationData.push({
          pixels: imgArray,
          label: labelsData[i]
        });
      }
      setValidationDigits(validationData);
      
      // Initial empty predictions
      setValidationPredictions(new Array(10).fill(null));
      
      xs.dispose();
      ys.dispose();
    } catch (err) {
      console.error(err);
      setDatasetStatus('error');
    }
  };

  // Build a fresh CNN architecture in TF.js
  const createModel = () => {
    const model = tf.sequential();
    
    // Conv2D
    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 3,
      filters: 16,
      activation: 'relu'
    }));
    
    // MaxPooling
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    
    // Flatten
    model.add(tf.layers.flatten());
    
    // Dense Layer
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    
    // Output Layer (10 digits, softmax)
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    
    // Compile
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    return model;
  };

  // Run validation set predictions live during training
  const runValidation = (modelInstance) => {
    if (validationDigits.length === 0 || !modelInstance) return;
    
    tf.tidy(() => {
      // Stack validation arrays to shape [10, 28, 28, 1]
      const tensors = validationDigits.map(d => tf.tensor(d.pixels, [28, 28, 1]));
      const inputBatch = tf.stack(tensors);
      const prediction = modelInstance.predict(inputBatch);
      const predictedLabels = prediction.argMax(-1).arraySync();
      setValidationPredictions(predictedLabels);
    });
  };

  // Draw the loss/accuracy curves on custom canvas graph
  const drawChart = () => {
    const canvas = chartCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const w = (canvas.width = canvas.parentElement.clientWidth);
    const h = (canvas.height = 180);

    ctx.clearRect(0, 0, w, h);
    
    const lossHistory = lossHistoryRef.current;
    const accHistory = accHistoryRef.current;
    if (lossHistory.length === 0) return;

    // Draw Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (h / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const padding = 10;
    const graphW = w - padding * 2;
    const graphH = h - padding * 2;
    const maxLen = Math.max(lossHistory.length, 2);

    // Draw Loss Curve (Neon Purple)
    const maxLoss = Math.max(...lossHistory, 1.0);
    ctx.beginPath();
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.5;
    lossHistory.forEach((val, idx) => {
      const x = padding + (idx / (maxLen - 1)) * graphW;
      const y = h - padding - (val / maxLoss) * graphH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw Accuracy Curve (Neon Cyan)
    ctx.beginPath();
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2.5;
    accHistory.forEach((val, idx) => {
      const x = padding + (idx / (maxLen - 1)) * graphW;
      const y = h - padding - val * graphH; // accuracy goes 0 to 1
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  // Start/Resume training
  const handleTrainModel = async () => {
    if (trainingStatus === 'training') return;
    
    if (trainingStatus === 'idle' || trainingStatus === 'complete') {
      // Create fresh model
      const model = createModel();
      modelRef.current = model;
      lossHistoryRef.current = [];
      accHistoryRef.current = [];
      setLoss(0);
      setAccuracy(0);
    }
    
    setTrainingStatus('training');
    isTrainingRef.current = true;
    
    const model = modelRef.current;
    const numTrainElements = dataLoader.trainIndices.length;
    const batchesPerEpoch = Math.ceil(numTrainElements / batchSize);
    setTotalBatches(batchesPerEpoch);

    // Custom training loop so we can pause/resume and yield UI thread updates
    for (let epoch = currentEpoch; epoch < epochs; epoch++) {
      if (!isTrainingRef.current) break;
      setCurrentEpoch(epoch + 1);

      for (let batch = 0; batch < batchesPerEpoch; batch++) {
        if (!isTrainingRef.current) break;
        setCurrentBatch(batch + 1);

        // Fetch training batch
        const { xs, ys } = dataLoader.nextTrainBatch(batchSize);

        // Train batch
        const history = await model.fit(xs, ys, {
          epochs: 1,
          batchSize: batchSize,
          verbose: 0
        });

        const currentLoss = history.history.loss[0];
        const currentAcc = history.history.acc[0];

        setLoss(currentLoss);
        setAccuracy(currentAcc);

        // Record history for graphs
        lossHistoryRef.current.push(currentLoss);
        accHistoryRef.current.push(currentAcc);
        
        // Draw charts & run validation digits test
        drawChart();
        if ((batch + 1) % 5 === 0) {
          runValidation(model);
        }

        // Clean memory
        xs.dispose();
        ys.dispose();

        // Yield to allow UI rendering
        await tf.nextFrame();
      }
    }

    if (isTrainingRef.current) {
      setTrainingStatus('complete');
      isTrainingRef.current = false;
      runValidation(model);
      
      // Update the parent component with our newly trained model weights!
      onModelUpdate(model);
      
      // Trigger celebrate confetti
      confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 }
      });
    }
  };

  const handlePauseTraining = () => {
    isTrainingRef.current = false;
    setTrainingStatus('paused');
  };

  const handleResetTraining = () => {
    isTrainingRef.current = false;
    setTrainingStatus('idle');
    setCurrentEpoch(0);
    setCurrentBatch(0);
    setLoss(0);
    setAccuracy(0);
    lossHistoryRef.current = [];
    accHistoryRef.current = [];
    setValidationPredictions(new Array(10).fill(null));
    drawChart();
  };

  useEffect(() => {
    // Redraw chart on resize
    const handleResize = () => drawChart();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      isTrainingRef.current = false;
    };
  }, []);

  return (
    <div className="flex flex-col gap-8 w-full max-w-4xl">
      
      {/* Step 1: Loading Dataset */}
      {datasetStatus !== 'loaded' ? (
        <div className="flex flex-col items-center justify-center p-12 bg-glassBg border border-glassBorder rounded-3xl backdrop-blur-md shadow-glass text-center gap-6">
          <Database size={40} className="text-neonPurple animate-pulse-slow" />
          <div className="space-y-2">
            <h3 className="text-lg font-bold text-white/90">Load MNIST Dataset</h3>
            <p className="text-sm text-white/50 max-w-md leading-relaxed">
              We need to download a lightweight 5,000-digit sprite sheet database of actual human handwriting from Google Cloud servers to train our neural network directly on your device.
            </p>
          </div>
          
          <button
            onClick={handleLoadDataset}
            disabled={datasetStatus === 'loading'}
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-neonPurple text-white font-semibold shadow-neon-purple hover:bg-neonPurple/90 active:scale-95 transition-all disabled:opacity-50"
          >
            {datasetStatus === 'loading' ? (
              <>
                <RefreshCw className="animate-spin" size={18} />
                <span>Downloading Sprite Sheet...</span>
              </>
            ) : (
              <>
                <Database size={18} />
                <span>Fetch MNIST Dataset (4.5 MB)</span>
              </>
            )}
          </button>
        </div>
      ) : (
        /* Training Control Board */
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Column 1: Config & Progress */}
          <div className="bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass flex flex-col gap-6">
            <div className="flex items-center gap-2 border-b border-white/10 pb-3">
              <Sliders size={18} className="text-neonPurple" />
              <h3 className="text-sm font-semibold text-white/95 uppercase tracking-wider">Hyperparameters</h3>
            </div>

            {/* Slider Configs */}
            <div className="space-y-4">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-white/60">Training Epochs</span>
                  <span className="font-semibold text-neonPurple">{epochs}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={epochs}
                  disabled={trainingStatus === 'training'}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="w-full accent-neonPurple bg-white/10 rounded-lg cursor-pointer h-1.5 disabled:opacity-30"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-white/60">Batch Size</span>
                  <span className="font-semibold text-neonPurple">{batchSize}</span>
                </div>
                <select
                  value={batchSize}
                  disabled={trainingStatus === 'training'}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="w-full bg-black/60 border border-glassBorder rounded-xl px-3 py-2 text-xs text-white/80 focus:border-neonPurple focus:outline-none disabled:opacity-30"
                >
                  <option value="64">64 digits</option>
                  <option value="128">128 digits</option>
                  <option value="256">256 digits</option>
                </select>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-white/60">Learning Rate (α)</span>
                  <span className="font-semibold text-neonPurple">{learningRate}</span>
                </div>
                <select
                  value={learningRate}
                  disabled={trainingStatus === 'training'}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full bg-black/60 border border-glassBorder rounded-xl px-3 py-2 text-xs text-white/80 focus:border-neonPurple focus:outline-none disabled:opacity-30"
                >
                  <option value="0.001">0.001 (Slow / Precise)</option>
                  <option value="0.005">0.005 (Recommended)</option>
                  <option value="0.01">0.01 (Fast)</option>
                  <option value="0.05">0.05 (Aggressive)</option>
                </select>
              </div>
            </div>

            {/* Play controls */}
            <div className="flex gap-3 border-t border-white/5 pt-5">
              {trainingStatus === 'training' ? (
                <button
                  onClick={handlePauseTraining}
                  className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-yellow-500/10 hover:bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 font-semibold text-sm transition-all active:scale-95"
                >
                  <Pause size={16} />
                  <span>Pause</span>
                </button>
              ) : (
                <button
                  onClick={handleTrainModel}
                  className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-neonPurple text-white font-semibold text-sm shadow-neon-purple hover:bg-neonPurple/90 active:scale-95 transition-all"
                >
                  <Play size={16} />
                  <span>{trainingStatus === 'paused' ? 'Resume' : 'Train CNN'}</span>
                </button>
              )}

              <button
                onClick={handleResetTraining}
                disabled={trainingStatus === 'idle'}
                className="p-3 rounded-xl bg-white/5 hover:bg-white/10 text-white/80 border border-glassBorder disabled:opacity-30 disabled:pointer-events-none transition-all active:scale-95"
                title="Reset Training"
              >
                <RefreshCw size={16} />
              </button>
            </div>

            {/* Progress Metrics Panel */}
            <div className="bg-black/40 border border-glassBorder rounded-2xl p-4 space-y-3 font-mono">
              <div className="flex justify-between text-xs">
                <span className="text-white/40">Epoch:</span>
                <span className="text-white/90">{currentEpoch} / {epochs}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-white/40">Batch:</span>
                <span className="text-white/90">{currentBatch} / {totalBatches}</span>
              </div>
              <div className="flex justify-between text-xs border-t border-white/5 pt-2">
                <span className="text-white/40">Batch Loss:</span>
                <span className="text-neonPurple font-bold">{loss.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-white/40">Batch Acc:</span>
                <span className="text-neonCyan font-bold">{(accuracy * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Column 2: Loss & Accuracy Graphs + Validation Predictions */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            
            {/* Live Chart Panel */}
            <div className="bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">Live Accuracy & Loss Curves</h4>
                <div className="flex gap-4 text-[10px]">
                  <span className="flex items-center gap-1.5 text-neonCyan">
                    <span className="w-2.5 h-1 rounded bg-neonCyan block"></span> Accuracy
                  </span>
                  <span className="flex items-center gap-1.5 text-neonPurple">
                    <span className="w-2.5 h-1 rounded bg-neonPurple block"></span> Loss
                  </span>
                </div>
              </div>
              
              <div className="relative rounded-2xl overflow-hidden bg-black/50 border border-glassBorder p-2 h-[180px] flex items-center justify-center">
                <canvas ref={chartCanvasRef} className="block w-full h-full" />
                {lossHistoryRef.current.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-white/30">
                    Charts will draw live once training begins
                  </div>
                )}
              </div>
            </div>

            {/* Validation Digits Performance Grid */}
            <div className="bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass flex flex-col gap-4">
              <div className="flex items-center gap-2 pb-2 border-b border-white/5">
                <Award size={16} className="text-neonCyan" />
                <h4 className="text-xs font-semibold text-white/95 uppercase tracking-wider">Real-time Validation Testing (10 random digits)</h4>
              </div>

              <div className="grid grid-cols-5 sm:grid-cols-10 gap-3">
                {validationDigits.map((digit, idx) => {
                  const pred = validationPredictions[idx];
                  const isCorrect = pred === digit.label;
                  
                  return (
                    <div
                      key={idx}
                      className={`flex flex-col items-center p-2 rounded-xl border transition-all duration-300 ${
                        pred === null
                          ? 'border-glassBorder bg-black/20'
                          : isCorrect
                            ? 'border-neonGreen/40 bg-neonGreen/5 shadow-[0_0_10px_rgba(16,185,129,0.1)]'
                            : 'border-red-500/40 bg-red-500/5'
                      }`}
                    >
                      {/* Digit canvas image */}
                      <div className="rounded overflow-hidden bg-black p-0.5 border border-white/10 mb-2">
                        <canvas
                          width={28}
                          height={28}
                          className="block w-8 h-8 image-render-pixelated grayscale invert"
                          ref={(canvas) => {
                            if (!canvas) return;
                            const ctx = canvas.getContext('2d');
                            const imgData = ctx.createImageData(28, 28);
                            const data = imgData.data;
                            for (let i = 0; i < 28 * 28; i++) {
                              const val = digit.pixels[i] * 255;
                              data[i * 4] = val;
                              data[i * 4 + 1] = val;
                              data[i * 4 + 2] = val;
                              data[i * 4 + 3] = 255;
                            }
                            ctx.putImageData(imgData, 0, 0);
                          }}
                        />
                      </div>
                      
                      {/* True Label */}
                      <span className="text-[10px] text-white/45">Label: <strong className="text-white/80">{digit.label}</strong></span>
                      
                      {/* Predicted Label */}
                      <span className={`text-xs font-bold mt-0.5 ${
                        pred === null
                          ? 'text-white/30'
                          : isCorrect
                            ? 'text-neonGreen'
                            : 'text-red-400'
                      }`}>
                        Pred: {pred !== null ? pred : '?'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingPlayground;
