import React, { useRef, useState, useEffect } from 'react';
import { Trash2, RotateCcw, Paintbrush, Grid } from 'lucide-react';

const DrawingCanvas = ({ onDraw, onClear }) => {
  const canvasRef = useRef(null);
  const previewRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(16);
  const [showGrid, setShowGrid] = useState(true);
  const historyRef = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Set black background initially
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    saveState();
  }, []);

  // Save current canvas state to history for undo
  const saveState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    historyRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (historyRef.current.length > 20) {
      historyRef.current.shift();
    }
  };

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    
    // Support touch events
    if (e.touches && e.touches.length > 0) {
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top
      };
    }
    
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    const coords = getCoordinates(e);
    if (!coords) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff'; // White ink on black background
    setIsDrawing(true);

    // Draw single dot on click
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    preprocessAndPredict();
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const coords = getCoordinates(e);
    if (!coords) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    preprocessAndPredict();
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      saveState();
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    historyRef.current = [];
    saveState();

    // Clear preview canvas
    const previewCanvas = previewRef.current;
    if (previewCanvas) {
      const pCtx = previewCanvas.getContext('2d');
      pCtx.fillStyle = '#000000';
      pCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    }
    onClear();
  };

  const undo = () => {
    if (historyRef.current.length <= 1) {
      clearCanvas();
      return;
    }
    
    // Pop current state
    historyRef.current.pop();
    // Retrieve previous state
    const prevState = historyRef.current[historyRef.current.length - 1];
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(prevState, 0, 0);
    
    preprocessAndPredict();
  };

  // Preprocess canvas image to 28x28 grayscale array
  const preprocessAndPredict = () => {
    const canvas = canvasRef.current;
    const previewCanvas = previewRef.current;
    if (!canvas || !previewCanvas) return;

    const ctx = canvas.getContext('2d');
    const pCtx = previewCanvas.getContext('2d');

    // Find bounding box and Center of Mass
    const bounds = getDigitCenterOfMass(canvas);
    if (!bounds) {
      // Nothing drawn yet
      onClear();
      return;
    }

    // Create temporary image from canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tCtx = tempCanvas.getContext('2d');

    tCtx.fillStyle = '#000000';
    tCtx.fillRect(0, 0, 28, 28);

    // Calculate dimensions to center the digit
    const digitW = bounds.maxX - bounds.minX + 1;
    const digitH = bounds.maxY - bounds.minY + 1;
    const maxDim = Math.max(digitW, digitH);
    
    // Scale factor to fit 20x20 area inside 28x28 (standard MNIST preprocessing)
    const scale = 20 / maxDim;
    const drawW = digitW * scale;
    const drawH = digitH * scale;
    
    // Align using Center of Mass: Center of Mass should end up at (14, 14)
    const dx = 14 - (bounds.comX - bounds.minX) * scale;
    const dy = 14 - (bounds.comY - bounds.minY) * scale;

    // Draw bounding box contents scaled and centered by Center of Mass
    tCtx.drawImage(
      canvas,
      bounds.minX, bounds.minY, digitW, digitH, // Source
      dx, dy, drawW, drawH                     // Destination
    );

    // Get 28x28 pixel data
    const imgData = tCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;

    // Convert RGBA to 28x28 flat array [0, 1]
    let pixels = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      pixels[i] = data[i * 4] / 255.0;
    }

    // Apply a soft blur to mimic the anti-aliased MNIST dataset format
    pixels = applySoftBlur(pixels);

    // Render the blurred image to the visual 28x28 preview canvas
    const previewImgData = pCtx.createImageData(28, 28);
    const pData = previewImgData.data;
    for (let i = 0; i < 28 * 28; i++) {
      const val = Math.min(255, Math.max(0, pixels[i] * 255));
      pData[i * 4] = val;
      pData[i * 4 + 1] = val;
      pData[i * 4 + 2] = val;
      pData[i * 4 + 3] = 255;
    }
    
    // Draw the 28x28 image onto the 140x140 visual preview canvas using a temporary canvas
    const tempCanvasBlurred = document.createElement('canvas');
    tempCanvasBlurred.width = 28;
    tempCanvasBlurred.height = 28;
    const tbCtx = tempCanvasBlurred.getContext('2d');
    tbCtx.putImageData(previewImgData, 0, 0);

    pCtx.fillStyle = '#000000';
    pCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    pCtx.drawImage(tempCanvasBlurred, 0, 0, previewCanvas.width, previewCanvas.height);

    onDraw(pixels);
  };

  // Helper to calculate both bounding box and Center of Mass (Centroid)
  const getDigitCenterOfMass = (canvas) => {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let sumX = 0, sumY = 0, totalWeight = 0;
    let minX = width, minY = height, maxX = -1, maxY = -1;
    let hasPixels = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const val = data[idx]; // Red channel of white stroke
        if (val > 15) {
          hasPixels = true;
          sumX += x * val;
          sumY += y * val;
          totalWeight += val;
          
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!hasPixels || totalWeight === 0) return null;

    // Add some padding to bounding box
    const padding = 10;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(width - 1, maxX + padding);
    maxY = Math.min(height - 1, maxY + padding);

    return {
      comX: sumX / totalWeight,
      comY: sumY / totalWeight,
      minX, minY, maxX, maxY
    };
  };

  // 3x3 soft box blur to match MNIST's anti-aliased image characteristics
  const applySoftBlur = (pixels) => {
    const blurred = new Float32Array(28 * 28);
    const kernel = [
      1/16, 2/16, 1/16,
      2/16, 4/16, 2/16,
      1/16, 2/16, 1/16
    ];

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        let sum = 0;
        let weightSum = 0;
        
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const ny = y + ky;
            const nx = x + kx;
            if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
              const pixelVal = pixels[ny * 28 + nx];
              const weight = kernel[(ky + 1) * 3 + (kx + 1)];
              sum += pixelVal * weight;
              weightSum += weight;
            }
          }
        }
        
        let blurredVal = sum / weightSum;
        // Increase contrast slightly on blurred values to maintain brightness
        if (blurredVal > 0.05) {
          blurredVal = Math.min(1.0, blurredVal * 1.15);
        }
        blurred[y * 28 + x] = blurredVal;
      }
    }
    return blurred;
  };

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-xl">
      {/* Tool panel */}
      <div className="flex items-center justify-between w-full bg-glassBg border border-glassBorder rounded-2xl px-6 py-4 backdrop-blur-md shadow-glass">
        <div className="flex items-center gap-4">
          <button
            onClick={clearCanvas}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 transition-all duration-300 active:scale-95"
            title="Clear Canvas"
          >
            <Trash2 size={18} />
            <span className="hidden sm:inline text-sm font-medium">Clear</span>
          </button>

          <button
            onClick={undo}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-white/80 border border-glassBorder transition-all duration-300 active:scale-95"
            title="Undo Last Stroke"
          >
            <RotateCcw size={18} />
            <span className="hidden sm:inline text-sm font-medium">Undo</span>
          </button>
        </div>

        <div className="flex items-center gap-6">
          {/* Brush Size Slider */}
          <div className="flex items-center gap-3">
            <Paintbrush size={16} className="text-neonCyan" />
            <input
              type="range"
              min="8"
              max="32"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              className="w-24 accent-neonCyan bg-white/10 rounded-lg cursor-pointer h-1.5"
            />
            <span className="text-xs font-semibold text-neonCyan w-4 text-center">{brushSize}</span>
          </div>

          {/* Grid Toggle */}
          <button
            onClick={() => setShowGrid(!showGrid)}
            className={`p-2 rounded-xl transition-all duration-300 border ${
              showGrid 
                ? 'bg-neonCyan/20 text-neonCyan border-neonCyan/40' 
                : 'bg-white/5 text-white/50 border-glassBorder hover:text-white/80'
            }`}
            title="Toggle Drawing Helper Grid"
          >
            <Grid size={18} />
          </button>
        </div>
      </div>

      {/* Main interactive area */}
      <div className="flex flex-col md:flex-row items-center gap-8 w-full">
        {/* Draw Canvas Container */}
        <div className="relative group rounded-3xl overflow-hidden border border-glassBorder shadow-glass bg-black p-1 transition-all duration-500 hover:border-neonCyan/50">
          {showGrid && (
            <div className="absolute inset-0 pointer-events-none grid grid-cols-4 grid-rows-4 opacity-15 border border-dashed border-white/50">
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-r border-b border-dashed border-white/50"></div>
              <div className="border-b border-dashed border-white/50"></div>
            </div>
          )}
          
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
            className="block cursor-crosshair rounded-2xl bg-black touch-none"
          />
        </div>

        {/* AI Preprocessor Preview Container */}
        <div className="flex flex-col items-center gap-3 bg-glassBg border border-glassBorder rounded-3xl p-6 backdrop-blur-md shadow-glass w-full max-w-[200px]">
          <span className="text-xs font-semibold uppercase tracking-wider text-white/50">AI Input Preview</span>
          <div className="rounded-xl overflow-hidden border border-glassBorder bg-black p-0.5 shadow-inner">
            <canvas
              ref={previewRef}
              width={140}
              height={140}
              className="block bg-black"
            />
          </div>
          <span className="text-[10px] text-white/40 text-center leading-normal">
            Centering and resizing to <strong className="text-neonCyan">28x28 grayscale</strong> pixels matching MNIST database standard.
          </span>
        </div>
      </div>
    </div>
  );
};

export default DrawingCanvas;
