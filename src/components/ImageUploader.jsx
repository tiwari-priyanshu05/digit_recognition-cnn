import React, { useRef, useState } from 'react';
import { Upload, FileImage, Image as ImageIcon } from 'lucide-react';

const ImageUploader = ({ onUpload, onClear }) => {
  const [dragActive, setDragActive] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const previewRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG, JPG, JPEG).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target.result);
      
      const img = new Image();
      img.onload = () => {
        preprocessImage(img);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  };

  const preprocessImage = (img) => {
    const previewCanvas = previewRef.current;
    if (!previewCanvas) return;
    const pCtx = previewCanvas.getContext('2d');

    // Create a 28x28 offscreen canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tCtx = tempCanvas.getContext('2d');

    // Draw uploaded image onto offscreen canvas (automatically resizes it to 28x28)
    // To preserve aspect ratio and center the digit, we crop/center it
    const minDim = Math.min(img.width, img.height);
    const sx = (img.width - minDim) / 2;
    const sy = (img.height - minDim) / 2;

    // Draw cropped center square
    tCtx.drawImage(
      img,
      sx, sy, minDim, minDim, // Source
      4, 4, 20, 20            // Destination (centered 20x20 in 28x28)
    );

    // Get pixel data to analyze background and invert if necessary
    const imgData = tCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;

    // Analyze corner pixels to detect if background is white/light
    // We average the red channel of the 4 corners
    const cornerPixels = [
      data[0], // top-left
      data[(28 - 1) * 4], // top-right
      data[(28 * 27) * 4], // bottom-left
      data[(28 * 28 - 1) * 4] // bottom-right
    ];
    const avgCornerBrightness = cornerPixels.reduce((a, b) => a + b, 0) / 4;
    const shouldInvert = avgCornerBrightness > 120; // It's a light background!

    const pixels = new Float32Array(28 * 28);

    // Convert to grayscale and invert if background is light
    for (let i = 0; i < 28 * 28; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      
      // Calculate luminosity grayscale
      let grayscale = 0.299 * r + 0.587 * g + 0.114 * b;
      
      if (shouldInvert) {
        grayscale = 255 - grayscale; // Invert to white digit on black background
      }

      // Apply a threshold or contrast stretch to remove noise
      if (grayscale < 35) grayscale = 0; // threshold background noise

      // Save processed value
      pixels[i] = grayscale / 255.0;

      // Update image data for the visual preview canvas
      data[i * 4] = grayscale;
      data[i * 4 + 1] = grayscale;
      data[i * 4 + 2] = grayscale;
      data[i * 4 + 3] = 255; // opaque
    }

    // Write preprocessed pixels back to temp canvas
    tCtx.putImageData(imgData, 0, 0);

    // Draw the 28x28 temp canvas onto our larger visual preview canvas
    pCtx.fillStyle = '#000000';
    pCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    pCtx.drawImage(tempCanvas, 0, 0, previewCanvas.width, previewCanvas.height);

    // Trigger classification callback
    onUpload(pixels);
  };

  const clearUpload = () => {
    setImagePreview(null);
    const previewCanvas = previewRef.current;
    if (previewCanvas) {
      const pCtx = previewCanvas.getContext('2d');
      pCtx.fillStyle = '#000000';
      pCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    }
    onClear();
  };

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-xl">
      {/* Upload area */}
      <div className="flex flex-col md:flex-row items-center gap-8 w-full">
        {/* Upload Drop Zone */}
        <div className="w-full max-w-[280px]">
          <input
            type="file"
            id="input-file-upload"
            className="hidden"
            accept="image/*"
            onChange={handleChange}
          />
          
          <label
            htmlFor="input-file-upload"
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`flex flex-col items-center justify-center w-[280px] h-[280px] rounded-3xl border-2 border-dashed cursor-pointer transition-all duration-300 ${
              dragActive 
                ? 'border-neonPurple bg-neonPurple/10 scale-[0.98]' 
                : imagePreview 
                  ? 'border-glassBorder bg-black/40 hover:border-neonPurple/40' 
                  : 'border-glassBorder bg-glassBg hover:bg-white/5 hover:border-glassBorder/60'
            } backdrop-blur-md shadow-glass`}
          >
            {imagePreview ? (
              <div className="relative w-full h-full p-4 flex flex-col items-center justify-center gap-3">
                <img
                  src={imagePreview}
                  alt="Original upload"
                  className="max-w-full max-h-[180px] rounded-xl object-contain shadow-md"
                />
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    clearUpload();
                  }}
                  className="px-4 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 text-xs font-medium transition-colors"
                >
                  Remove Image
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center p-6 text-center gap-3">
                <div className="p-4 rounded-full bg-neonPurple/10 text-neonPurple border border-neonPurple/20 animate-pulse-slow">
                  <Upload size={28} />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-white/90">Upload digit image</p>
                  <p className="text-xs text-white/50">Drag and drop here, or click to browse</p>
                </div>
                <p className="text-[10px] text-white/30 max-w-[200px] leading-relaxed">
                  Supports JPEG, PNG. Grayscale/color, light/dark backgrounds are automatically corrected.
                </p>
              </div>
            )}
          </label>
        </div>

        {/* Preprocessor Preview */}
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
            Normalizing image into centered <strong className="text-neonPurple">28x28 grayscale</strong> pixels with background correction.
          </span>
        </div>
      </div>
    </div>
  );
};

export default ImageUploader;
