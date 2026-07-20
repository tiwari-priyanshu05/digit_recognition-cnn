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

    // Create temporary image from canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tCtx = tempCanvas.getContext('2d');

    // Draw the main canvas contents resized to 28x28
    // To mimic MNIST better, we compute the bounding box of the drawn digit and center it with padding
    const bounds = getDigitBoundingBox(canvas);
    if (!bounds) {
      // Nothing drawn yet
      onClear();
      return;
    }

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
    
    const dx = (28 - drawW) / 2;
    const dy = (28 - drawH) / 2;

    // Draw bounding box contents scaled and centered
    tCtx.drawImage(
      canvas,
      bounds.minX, bounds.minY, digitW, digitH, // Source
      dx, dy, drawW, drawH                     // Destination
    );

    // Update the visual 28x28 preview canvas
    pCtx.drawImage(tempCanvas, 0, 0, previewCanvas.width, previewCanvas.height);

    // Get 28x28 pixel data
    const imgData = tCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;

    // Convert RGBA to 28x28 flat array [0, 1]
    const pixels = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      // Since it's white on black, we can take red, green, or blue channel (all equal)
      pixels[i] = data[i * 4] / 255.0;
    }

    onDraw(pixels);
  };

  // Helper to find the bounding box of the drawn digit (non-black pixels)
  const getDigitBoundingBox = (canvas) => {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let minX = width, minY = height, maxX = -1, maxY = -1;
    let hasPixels = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        // Check if pixel red value is above threshold (non-black)
        if (data[idx] > 20) {
          hasPixels = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!hasPixels) return null;

    // Add some padding to bounding box
    const padding = 10;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(width - 1, maxX + padding);
    maxY = Math.min(height - 1, maxY + padding);

    return { minX, minY, maxX, maxY };
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
