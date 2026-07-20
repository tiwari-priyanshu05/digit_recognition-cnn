import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Layers, HelpCircle, Activity } from 'lucide-react';

const CnnVisualizer = ({ model, inputPixels }) => {
  const [activations, setActivations] = useState(null);
  const [denseActivations, setDenseActivations] = useState(null);
  const [selectedFilter, setSelectedFilter] = useState(null);
  const canvasRefs = useRef([]);

  useEffect(() => {
    if (!model || !inputPixels) {
      setActivations(null);
      setDenseActivations(null);
      return;
    }

    tf.tidy(() => {
      try {
        // Reshape input pixels to [1, 28, 28, 1] tensor
        const inputTensor = tf.tensor4d(inputPixels, [1, 28, 28, 1]);

        // 1. Get Conv2D Layer Activations
        // The first layer is the Conv2D layer (index 0)
        const convLayer = model.layers[0];
        const convModel = tf.model({ inputs: model.inputs, outputs: convLayer.output });
        const convResult = convModel.predict(inputTensor); // Shape: [1, 26, 26, 32]
        
        // Convert to JS arrays
        const convData = convResult.squeeze().arraySync(); // Shape: [26, 26, 32]
        
        // Rearrange to an array of 32 filter outputs [32, 26, 26]
        const filtersArray = [];
        for (let f = 0; f < 32; f++) {
          const filterMap = new Float32Array(26 * 26);
          for (let y = 0; y < 26; y++) {
            for (let x = 0; x < 26; x++) {
              filterMap[y * 26 + x] = convData[y][x][f];
            }
          }
          filtersArray.push(filterMap);
        }
        setActivations(filtersArray);

        // 2. Get Dense Layer Activations
        // The dense layer is layer index 3 (after Conv2D, MaxPool, Flatten)
        // Let's find the layer with name starting with 'dense' or index 3
        let denseLayer = model.layers.find(l => l.name.startsWith('dense') && !l.name.includes('dense_1'));
        if (!denseLayer) {
          // Fallback to index 3
          denseLayer = model.layers[3];
        }

        if (denseLayer) {
          const denseModel = tf.model({ inputs: model.inputs, outputs: denseLayer.output });
          const denseResult = denseModel.predict(inputTensor); // Shape: [1, 128]
          const denseData = denseResult.squeeze().arraySync(); // Array of 128 elements
          setDenseActivations(denseData);
        }
      } catch (err) {
        console.error("Error extracting activations:", err);
      }
    });
  }, [model, inputPixels]);

  // Render individual activation maps on their small canvases
  useEffect(() => {
    if (!activations) return;
    
    activations.forEach((filterData, idx) => {
      const canvas = canvasRefs.current[idx];
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(26, 26);
      const data = imgData.data;

      // Find min and max for scaling
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < filterData.length; i++) {
        if (filterData[i] < min) min = filterData[i];
        if (filterData[i] > max) max = filterData[i];
      }
      const range = max - min || 1;

      // Draw as a beautiful neon cyan activation map
      for (let i = 0; i < 26 * 26; i++) {
        const val = ((filterData[i] - min) / range) * 255;
        // Cyan color mapping: R=0, G=cyan-gradient, B=cyan-gradient
        data[i * 4] = Math.max(0, val * 0.1);              // Red
        data[i * 4 + 1] = Math.min(255, val * 0.7 + 20);   // Green
        data[i * 4 + 2] = Math.min(255, val * 0.9 + 50);   // Blue
        data[i * 4 + 3] = val > 10 ? Math.min(255, val + 50) : 25; // Alpha (more active = more opaque)
      }
      
      ctx.putImageData(imgData, 0, 0);
    });
  }, [activations]);

  if (!model) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-glassBg border border-glassBorder rounded-3xl backdrop-blur-md shadow-glass w-full max-w-xl h-64 text-center">
        <Activity size={32} className="text-neonPurple animate-pulse-slow mb-3" />
        <p className="text-sm text-white/50">Model is not loaded. Draw a digit first or wait for the Keras weights to load.</p>
      </div>
    );
  }

  if (!inputPixels) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-glassBg border border-glassBorder rounded-3xl backdrop-blur-md shadow-glass w-full max-w-xl h-64 text-center">
        <Layers size={32} className="text-neonCyan animate-bounce mb-3" />
        <p className="text-sm text-white/50">Draw or upload a digit to visualize the neural network's layers and filters live!</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8 w-full max-w-4xl">
      {/* Visualizer Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Column 1: Conv2D Feature Maps (32 filters) */}
        <div className="lg:col-span-2 bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div className="flex items-center gap-2">
              <Layers size={18} className="text-neonCyan" />
              <h3 className="text-sm font-semibold text-white/95 uppercase tracking-wider">Convolutional Layer (32 Filters)</h3>
            </div>
            <span className="text-[10px] bg-neonCyan/20 text-neonCyan border border-neonCyan/30 px-2 py-0.5 rounded font-mono">26x26x32</span>
          </div>

          <p className="text-xs text-white/50 leading-relaxed">
            The CNN runs 32 convolutional filters. Each canvas below shows the activation of a specific filter, highlighting curves, edges, and shapes extracted from your drawing.
          </p>

          {/* Filters Grid */}
          <div className="grid grid-cols-8 gap-2 bg-black/40 p-3 rounded-2xl border border-glassBorder">
            {activations && activations.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedFilter(selectedFilter === idx ? null : idx)}
                className={`relative group rounded p-0.5 border transition-all duration-300 ${
                  selectedFilter === idx 
                    ? 'border-neonCyan bg-neonCyan/10 shadow-neon-cyan scale-105' 
                    : 'border-white/5 bg-black hover:border-neonCyan/30'
                }`}
                title={`Filter ${idx + 1}`}
              >
                <canvas
                  ref={el => canvasRefs.current[idx] = el}
                  width={26}
                  height={26}
                  className="block w-full h-auto aspect-square object-cover interpolation-pixelated"
                />
                <span className="absolute bottom-0 right-0 text-[7px] bg-black/80 px-0.5 text-white/60 font-mono">
                  #{idx}
                </span>
              </button>
            ))}
          </div>

          {/* Filter Zoomed View */}
          {selectedFilter !== null && activations && (
            <div className="flex items-center gap-4 bg-neonCyan/5 border border-neonCyan/20 rounded-2xl p-4 animate-fadeIn">
              <div className="p-1 rounded-xl bg-black border border-neonCyan/30">
                <canvas
                  id="zoomed-filter-canvas"
                  width={26}
                  height={26}
                  className="block w-16 h-16 aspect-square image-render-pixelated"
                  ref={(canvas) => {
                    if (!canvas) return;
                    const ctx = canvas.getContext('2d');
                    const imgData = ctx.createImageData(26, 26);
                    const data = imgData.data;
                    const filterData = activations[selectedFilter];
                    let min = Infinity, max = -Infinity;
                    for (let i = 0; i < filterData.length; i++) {
                      if (filterData[i] < min) min = filterData[i];
                      if (filterData[i] > max) max = filterData[i];
                    }
                    const range = max - min || 1;
                    for (let i = 0; i < 26 * 26; i++) {
                      const val = ((filterData[i] - min) / range) * 255;
                      data[i * 4] = Math.max(0, val * 0.1);
                      data[i * 4 + 1] = Math.min(255, val * 0.8 + 20);
                      data[i * 4 + 2] = Math.min(255, val + 50);
                      data[i * 4 + 3] = 255;
                    }
                    ctx.putImageData(imgData, 0, 0);
                  }}
                />
              </div>
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-neonCyan">Feature Map #{selectedFilter}</h4>
                <p className="text-[10px] text-white/50 leading-normal max-w-sm">
                  This filter acts as a feature extractor. Bright cyan areas indicate local patterns like diagonals or curves that match the filter's weights.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Column 2: Dense Layer & Activations */}
        <div className="bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div className="flex items-center gap-2">
              <Layers size={18} className="text-neonPurple" />
              <h3 className="text-sm font-semibold text-white/95 uppercase tracking-wider">Dense Layer (128 Nodes)</h3>
            </div>
            <span className="text-[10px] bg-neonPurple/20 text-neonPurple border border-neonPurple/30 px-2 py-0.5 rounded font-mono">128</span>
          </div>

          <p className="text-xs text-white/50 leading-relaxed">
            The flattened features pass into a dense fully-connected layer. Below is a representation of the 128 nodes, glowing based on their real-time activations.
          </p>

          {/* Dense Nodes Grid */}
          <div className="grid grid-cols-8 gap-2 bg-black/40 p-4 rounded-2xl border border-glassBorder h-full min-h-[160px] justify-items-center items-center">
            {denseActivations ? (
              denseActivations.map((activationVal, idx) => {
                // ReLU makes activations >= 0
                // Normalize for visual glow intensity
                const maxVal = Math.max(...denseActivations, 0.001);
                const intensity = Math.min(1.0, Math.max(0, activationVal / maxVal));
                
                return (
                  <div
                    key={idx}
                    className="relative group w-3 h-3 rounded-full transition-all duration-300"
                    style={{
                      backgroundColor: `rgba(168, 85, 247, ${intensity * 0.8 + 0.15})`,
                      boxShadow: intensity > 0.3 ? `0 0 ${intensity * 10}px rgba(168, 85, 247, ${intensity})` : 'none',
                      border: '1px solid rgba(255, 255, 255, 0.05)'
                    }}
                    title={`Dense Node #${idx}: ${activationVal.toFixed(3)}`}
                  />
                );
              })
            ) : (
              <div className="col-span-8 text-[10px] text-white/30 text-center">Loading activations...</div>
            )}
          </div>
          
          <div className="flex items-center gap-2 text-[10px] text-white/40 border-t border-white/5 pt-2">
            <HelpCircle size={12} className="text-neonPurple" />
            <span>Bright purple dots represent heavily fired nodes.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CnnVisualizer;
