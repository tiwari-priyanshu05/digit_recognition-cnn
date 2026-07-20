import h5py
import numpy as np

print("Running pure-NumPy Keras CNN forward pass test...")

# Create deterministic diagonal input (28x28x1)
img = np.zeros((1, 28, 28, 1), dtype=np.float32)
for i in range(28):
    img[0, i, i, 0] = 1.0

with h5py.File("mnist_cnn.h5", "r") as f:
    # Read weights
    conv_k = np.array(f['model_weights/conv2d/sequential/conv2d/kernel'])
    conv_b = np.array(f['model_weights/conv2d/sequential/conv2d/bias'])
    dense_k = np.array(f['model_weights/dense/sequential/dense/kernel'])
    dense_b = np.array(f['model_weights/dense/sequential/dense/bias'])
    output_k = np.array(f['model_weights/dense_1/sequential/dense_1/kernel'])
    output_b = np.array(f['model_weights/dense_1/sequential/dense_1/bias'])

# 1. Conv2D Layer
# input: (1, 28, 28, 1)
# kernel: (3, 3, 1, 32), bias: (32,)
# output: (1, 26, 26, 32)
conv_out = np.zeros((1, 26, 26, 32), dtype=np.float32)
for r in range(26):
    for c in range(26):
        for f in range(32):
            # dot product of 3x3 patch
            patch = img[0, r:r+3, c:c+3, 0]
            kernel_slice = conv_k[:, :, 0, f]
            val = np.sum(patch * kernel_slice) + conv_b[f]
            conv_out[0, r, c, f] = max(val, 0.0) # ReLU

print("Conv2D output sum:", np.sum(conv_out))

# 2. MaxPooling2D Layer
# input: (1, 26, 26, 32)
# pool size: 2x2, stride: 2x2
# output: (1, 13, 13, 32)
pool_out = np.zeros((1, 13, 13, 32), dtype=np.float32)
for r in range(13):
    for c in range(13):
        for f in range(32):
            patch = conv_out[0, r*2:r*2+2, c*2:c*2+2, f]
            pool_out[0, r, c, f] = np.max(patch)

print("MaxPooling2D output sum:", np.sum(pool_out))

# 3. Flatten Layer
# input: (1, 13, 13, 32)
# output: (1, 5408)
flatten_out = pool_out.reshape(1, -1)
print("Flatten output shape:", flatten_out.shape)

# 4. Dense Layer
# input: (1, 5408)
# weights: (5408, 128), bias: (128,)
# output: (1, 128)
dense_out = np.dot(flatten_out, dense_k) + dense_b
dense_out = np.maximum(dense_out, 0.0) # ReLU
print("Dense output sum:", np.sum(dense_out))

# 5. Dense Output Layer
# input: (1, 128)
# weights: (128, 10), bias: (10,)
# output: (1, 10)
logits = np.dot(dense_out, output_k) + output_b

# Softmax
exp_logits = np.exp(logits - np.max(logits)) # stable softmax
probs = exp_logits / np.sum(exp_logits)

print("\nNumPy Forward Pass Probabilities:")
for digit, prob in enumerate(probs[0]):
    print(f"Digit {digit}: {prob:.6f}")
print("Predicted Digit:", np.argmax(probs[0]))
