import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import Background3D from './components/Background3D';
import DrawingCanvas from './components/DrawingCanvas';
import ImageUploader from './components/ImageUploader';
import CnnVisualizer from './components/CnnVisualizer';
import TrainingPlayground from './components/TrainingPlayground';
import { PenTool, Upload, BrainCircuit, RefreshCw, BarChart2, ShieldAlert, Cpu } from 'lucide-react';

const App = () => {
  const [activeTab, setActiveTab] = useState('predict'); // predict, train
  const [inputMode, setInputMode] = useState('draw'); // draw, upload
  
  // Model states
  const [model, setModel] = useState(null);
  const [modelStatus, setModelStatus] = useState('loading'); // loading, loaded, failed
  const [modelSource, setModelSource] = useState('none'); // keras, browser, none
  
  // Classification states
  const [inputPixels, setInputPixels] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [predictedDigit, setPredictedDigit] = useState(null);
  const [confidence, setConfidence] = useState(null);

  // Load the pre-trained Keras weights converted in Python
  const loadKerasModel = async () => {
    setModelStatus('loading');
    try {
      // 1. Fetch metadata and raw binary weight files
      const metaRes = await fetch('/model_metadata.json');
      if (!metaRes.ok) throw new Error('Metadata not found');
      const metadata = await metaRes.json();

      const binRes = await fetch('/model_weights.bin');
      if (!binRes.ok) throw new Error('Weights bin not found');
      const binBuffer = await binRes.arrayBuffer();
      const weightsData = new Float32Array(binBuffer);

      // 2. Build the model topology matching the H5 structure
      const tfModel = tf.sequential();
      
      // Conv2D (32 filters, 3x3 kernel, relu activation)
      tfModel.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 32,
        activation: 'relu',
        name: 'conv2d'
      }));

      // MaxPooling2D (2x2 pool)
      tfModel.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
        name: 'max_pooling2d'
      }));

      // Flatten
      tfModel.add(tf.layers.flatten({ name: 'flatten' }));

      // Dense Fully Connected (128 units, relu)
      tfModel.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        name: 'dense'
      }));

      // Dense Output Layer (10 units, softmax)
      tfModel.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
        name: 'dense_1'
      }));

      // 3. Reconstruct tensors from flat Float32 arrays based on metadata sizes
      let offset = 0;
      const getTensor = (meta) => {
        const slice = weightsData.subarray(offset, offset + meta.size);
        offset += meta.size;
        return tf.tensor(slice, meta.shape, 'float32');
      };

      const convKernel = getTensor(metadata.conv2d_kernel);
      const convBias = getTensor(metadata.conv2d_bias);
      const denseKernel = getTensor(metadata.dense_kernel);
      const denseBias = getTensor(metadata.dense_bias);
      const outputKernel = getTensor(metadata.dense_1_kernel);
      const outputBias = getTensor(metadata.dense_1_bias);

      // 4. Load weights onto layers
      tfModel.setWeights([convKernel, convBias, denseKernel, denseBias, outputKernel, outputBias]);
      
      // Compile dummy to initialize weights inside GPU memory
      tfModel.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy'
      });

      setModel(tfModel);
      setModelSource('keras');
      setModelStatus('loaded');
      console.log("Keras model loaded successfully in TensorFlow.js");
    } catch (err) {
      console.error("Local Keras weights load failed:", err);
      setModelStatus('failed');
    }
  };

  useEffect(() => {
    loadKerasModel();
  }, []);

  // Run on-screen digit prediction
  const handleInference = (pixels) => {
    if (!model) return;
    setInputPixels(pixels);

    tf.tidy(() => {
      // Shape input array into standard [1, 28, 28, 1] Keras batch format
      const inputTensor = tf.tensor4d(pixels, [1, 28, 28, 1]);
      const outputTensor = model.predict(inputTensor);
      
      const probs = outputTensor.squeeze().arraySync();
      const digit = outputTensor.argMax(-1).dataSync()[0];
      
      setPredictions(probs);
      setPredictedDigit(digit);
      setConfidence(probs[digit]);
    });
  };

  const handleClearInference = () => {
    setInputPixels(null);
    setPredictions(null);
    setPredictedDigit(null);
    setConfidence(null);
  };

  // Called when a user trains a fresh model in the training playground tab
  const handleBrowserModelUpdate = (trainedModel) => {
    setModel(trainedModel);
    setModelSource('browser');
    setModelStatus('loaded');
    // Clear prediction states
    handleClearInference();
  };

  return (
    <div className="min-h-screen pb-12 relative overflow-hidden text-slate-100 selection:bg-neonPurple/30 selection:text-white">
      {/* 3D Moving Particle Network Background */}
      <Background3D />

      {/* Glistening Header Bar */}
      <header className="sticky top-0 z-50 w-full bg-black/30 backdrop-blur-xl border-b border-white/5 py-4 px-6 md:px-12 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-2xl bg-gradient-to-tr from-neonPurple to-neonCyan text-white shadow-neon-purple animate-pulse">
            <Cpu size={24} />
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
              ML Digit Recognition Playground
            </h1>
            <p className="text-[10px] text-white/40 uppercase tracking-widest font-bold">MNIST Convolutional Neural Network</p>
          </div>
        </div>

        {/* Model Status Indicator */}
        <div className="flex items-center gap-4">
          {modelStatus === 'loading' && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-glassBorder text-xs text-white/70 animate-pulse">
              <RefreshCw size={14} className="animate-spin text-neonCyan" />
              <span>Loading Model Weights...</span>
            </div>
          )}
          {modelStatus === 'loaded' && (
            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-green-500/10 border border-green-500/30 text-xs text-green-400 font-semibold shadow-[0_0_10px_rgba(34,197,94,0.05)]">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-ping"></span>
              <span>Model Loaded: </span>
              <strong className="uppercase bg-green-500/20 text-green-300 px-1.5 py-0.5 rounded text-[9px]">
                {modelSource === 'keras' ? 'Keras H5' : 'Browser Trained'}
              </strong>
            </div>
          )}
          {modelStatus === 'failed' && (
            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-red-500/10 border border-red-500/30 text-xs text-red-400 font-semibold">
              <ShieldAlert size={14} />
              <span>Weights load failed. Train a new one below!</span>
            </div>
          )}
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-6xl mx-auto px-4 md:px-8 mt-8 flex flex-col gap-8">
        
        {/* Navigation Tabs */}
        <div className="flex justify-center">
          <div className="flex bg-black/60 p-1.5 rounded-2xl border border-glassBorder backdrop-blur-md shadow-glass">
            <button
              onClick={() => setActiveTab('predict')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold tracking-wide transition-all duration-300 ${
                activeTab === 'predict' 
                  ? 'bg-neonCyan text-black shadow-neon-cyan' 
                  : 'text-white/60 hover:text-white/95 hover:bg-white/5'
              }`}
            >
              <PenTool size={16} />
              <span>Testing Playground</span>
            </button>
            <button
              onClick={() => setActiveTab('train')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold tracking-wide transition-all duration-300 ${
                activeTab === 'train' 
                  ? 'bg-neonPurple text-white shadow-neon-purple' 
                  : 'text-white/60 hover:text-white/95 hover:bg-white/5'
              }`}
            >
              <BrainCircuit size={16} />
              <span>Training Playground</span>
            </button>
          </div>
        </div>

        {/* Tab 1: Inference & Predictions */}
        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
            
            {/* Input Panel (Draw or Upload) */}
            <div className="bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass flex flex-col gap-6 items-center">
              <div className="flex justify-between items-center w-full border-b border-white/10 pb-4">
                <h2 className="text-sm font-black uppercase tracking-wider text-white/90">Input Interface</h2>
                
                {/* Mode Selector */}
                <div className="flex bg-black/50 p-1 rounded-xl border border-glassBorder">
                  <button
                    onClick={() => { setInputMode('draw'); handleClearInference(); }}
                    className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      inputMode === 'draw' 
                        ? 'bg-neonCyan/20 text-neonCyan border border-neonCyan/30' 
                        : 'text-white/50 hover:text-white'
                    }`}
                  >
                    <PenTool size={12} />
                    <span>Canvas</span>
                  </button>
                  <button
                    onClick={() => { setInputMode('upload'); handleClearInference(); }}
                    className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      inputMode === 'upload' 
                        ? 'bg-neonPurple/20 text-neonPurple border border-neonPurple/30' 
                        : 'text-white/50 hover:text-white'
                    }`}
                  >
                    <Upload size={12} />
                    <span>Upload Image</span>
                  </button>
                </div>
              </div>

              {/* Render Drawing Canvas or Image Uploader */}
              {inputMode === 'draw' ? (
                <DrawingCanvas onDraw={handleInference} onClear={handleClearInference} />
              ) : (
                <ImageUploader onUpload={handleInference} onClear={handleClearInference} />
              )}
            </div>

            {/* Output Panel: Prediction Bars & Visualizer */}
            <div className="flex flex-col gap-8">
              
              {/* Prediction Card */}
              <div className="bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass flex flex-col md:flex-row gap-6 items-center justify-between">
                
                {/* Glow Digit Circle */}
                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-white/50">Predicted Digit</span>
                  <div 
                    className={`w-28 h-28 rounded-full flex items-center justify-center border font-black text-5xl transition-all duration-500 ${
                      predictedDigit !== null 
                        ? inputMode === 'draw'
                          ? 'border-neonCyan bg-neonCyan/5 text-neonCyan shadow-neon-cyan scale-105 animate-pulse'
                          : 'border-neonPurple bg-neonPurple/5 text-neonPurple shadow-neon-purple scale-105 animate-pulse'
                        : 'border-white/10 bg-black/40 text-white/20'
                    }`}
                  >
                    {predictedDigit !== null ? predictedDigit : '?'}
                  </div>
                  {confidence !== null && (
                    <span className={`text-xs font-bold ${inputMode === 'draw' ? 'text-neonCyan' : 'text-neonPurple'}`}>
                      Confidence: {(confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>

                {/* Probability Horizontal Bars */}
                <div className="flex-1 w-full space-y-2 border-l border-white/5 pl-0 md:pl-6">
                  <div className="flex items-center gap-1.5 mb-2">
                    <BarChart2 size={14} className="text-white/50" />
                    <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">Confidence Scores (0-9)</span>
                  </div>
                  {Array.from({ length: 10 }).map((_, digit) => {
                    const prob = predictions ? predictions[digit] : 0;
                    const isWinner = predictedDigit === digit;
                    
                    return (
                      <div key={digit} className="flex items-center gap-3 text-xs">
                        <span className={`w-3 font-bold ${isWinner ? (inputMode === 'draw' ? 'text-neonCyan' : 'text-neonPurple') : 'text-white/40'}`}>
                          {digit}
                        </span>
                        <div className="flex-1 h-2 bg-black/60 rounded-full overflow-hidden border border-white/5">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              isWinner 
                                ? inputMode === 'draw'
                                  ? 'bg-gradient-to-r from-neonCyan to-cyan-400 shadow-neon-cyan'
                                  : 'bg-gradient-to-r from-neonPurple to-fuchsia-400 shadow-neon-purple'
                                : 'bg-white/10'
                            }`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className={`w-10 text-right font-mono text-[10px] ${isWinner ? 'font-bold' : 'text-white/40'}`}>
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* CNN Activation Feature-Map Grid Visualizer */}
              <CnnVisualizer model={model} inputPixels={inputPixels} />

            </div>

          </div>
        )}

        {/* Tab 2: Browser Training Simulator */}
        {activeTab === 'train' && (
          <div className="flex justify-center">
            <TrainingPlayground onModelUpdate={handleBrowserModelUpdate} />
          </div>
        )}

      </main>

      {/* Footer Details */}
      <footer className="mt-16 text-center text-[10px] text-white/30 space-y-1">
        <p>Built with Tailwind CSS, React, and TensorFlow.js. Model weights exported from Python Keras CNN.</p>
        <p>&copy; 2026 Digit Recognition ML Concept. Fully Client-Side Neural Network Inference.</p>
      </footer>
    </div>
  );
};

export default App;
