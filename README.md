# 🧠 MNIST CNN Digit Recognition ML Playground

An interactive, high-fidelity machine learning playground built with **React**, **Tailwind CSS v4**, and **TensorFlow.js**. 

This web application loads a pre-trained Python Keras CNN model (`mnist_cnn.h5`) completely client-side in the browser using binary float weights. It features interactive drawing and image uploading tools, real-time convolutional activation maps showing what features the network is detecting, and an on-device training playground where users can load the MNIST dataset and train a fresh model live with real-time graphs and validation tracking.

---

## ✨ Key Features

*   **⚡ 100% Client-Side Inference**: No heavy Python backends, API request delays, or serverless cold starts. Model evaluation runs at 60fps directly in the user's browser.
*   **🎨 Interactive Drawing Canvas**: Draw numbers with a custom brush size, a helper grid, and full undo/clear capabilities.
*   **📤 Smart Image Uploader**: Drag and drop handwritten images. The app automatically detects background brightness and inverts light backgrounds (like black pen on white paper) to match the white-on-black MNIST format.
*   **🔎 Real-Time CNN Layer Activations**: Captures the intermediate activations of the **Conv2D layer (32 filters)** to display exactly what features (edges, curves, diagonals) each filter is extracting. Renders a glowing node matrix of the **128 Dense Layer** units.
*   **🏫 On-device Browser Training**: Fetch 4,000 samples of the MNIST dataset sprite sheet from Google APIs and train a CNN live in the browser.
    *   **Live Charts**: Renders loss (purple) and accuracy (cyan) curves updating batch-by-batch on a custom canvas graph.
    *   **Live Test Grid**: Displays 10 random validation digits that dynamically update color (from red to green) as the network learns.
*   **🌌 Premium 3D Aesthetics**: Floating 3D perspective particle network background which rotates on X/Y axes and connects nodes with glowing alpha lines that track and warp away from mouse cursor movements.

---

## 📁 Repository Structure

```
├── public/
│   ├── model_metadata.json   # CNN weight shapes and offsets manifest
│   ├── model_weights.bin     # Raw float32 binary weights from keras .h5
│   └── favicon.svg
├── src/
│   ├── components/
│   │   ├── Background3D.jsx       # 3D Particle Canvas Background
│   │   ├── DrawingCanvas.jsx      # Draw canvas & MNIST Center-of-Mass alignment
│   │   ├── ImageUploader.jsx      # Upload panel with auto-brightness check
│   │   ├── CnnVisualizer.jsx      # Conv2D intermediate activation maps grid
│   │   └── TrainingPlayground.jsx # Live browser training playground & graphs
│   ├── utils/
│   │   └── mnist.js               # Sprite dataset downloader and batch compiler
│   ├── App.jsx                    # Coordinate tab navigation and TFJS inference
│   ├── main.jsx
│   └── index.css                  # Tailwind CSS v4 directives & theme configurations
├── vercel.json                    # Caching configurations for production deployment
├── vite.config.js                 # Vite bundler with tailwind integration
├── package.json
└── README.md
```

---

## 🛠️ Local Development Setup

To run the application on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tiwari-priyanshu05/digit_recognition-cnn.git
   cd digit_recognition-cnn
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Run the Vite development server**:
   ```bash
   npm run dev
   ```
   Open your browser to the local port (usually `http://localhost:5173`).

4. **Compile a production build**:
   ```bash
   npm run build
   ```

---

## 🌐 Deployment to Vercel

The project is fully configured for deployment on Vercel:

1. Go to [Vercel](https://vercel.com) and sign in using your GitHub account.
2. Click **Add New** -> **Project**.
3. Select and import your `digit_recognition-cnn` repository.
4. Vercel will automatically detect Vite. Leave the default configurations as they are and click **Deploy**.
5. Your application will be live on a secure HTTPS URL in under a minute!

---

## 🧠 Technical Details of the Network

The pre-trained model has the following architecture, matching `mnist_cnn.h5`:

1.  **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation. Outputs a `(26, 26, 32)` activation map.
2.  **MaxPooling2D Layer**: 2x2 pool size, 2x2 strides. Downsamples output to `(13, 13, 32)`.
3.  **Flatten Layer**: Flattens features in channels-last order to a `5408` dimensional vector.
4.  **Dense Layer**: 128 hidden nodes, ReLU activation.
5.  **Dense Output Layer**: 10 output nodes (digits 0-9), Softmax activation.
