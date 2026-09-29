"""
Week-5: Day 22 Project: Modular Multi-Layer Neural Network Engine with Multi-Class Softmax & Momentum
-------------------------------------------------------------------------------------------------------
Topics Covered:
- T1: Modular Deep Learning Architecture (Dynamic Layer & Parameter Stacking)
- T2: Multi-Class Softmax Activation & Categorical Cross-Entropy (CCE) Loss
- T3: Dynamic Forward & Backward Propagation Across Arbitrary N-Layers
- T4: Momentum-Based Stochastic Gradient Descent (SGD) Optimization
- T5: Multi-Class Decision Thresholding, Evaluation Metrics & Single-Sample Inference

Platform: Google Colab / Python
Libraries: NumPy, Pandas
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Input Data Generation (Synthetic Customer Tier Classification - 60 Records)
# Features: [Normalized Monthly Spend, App Usage Hours, Support Tickets, Account Age Rank]
# Multi-Class Targets: Tier 0 = Basic, Tier 1 = Pro, Tier 2 = Enterprise
# -------------------------------------------------------------------
np.random.seed(42)

X_raw = np.array([
    [0.15,  2, 0, 1], [0.85, 35, 4, 3], [0.95, 80, 8, 4], [0.10,  1, 0, 1], [0.78, 28, 3, 3],
    [0.98, 92, 9, 4], [0.20,  3, 0, 1], [0.82, 30, 4, 3], [0.92, 75, 7, 4], [0.18,  2, 0, 1],
    [0.88, 33, 3, 3], [0.99, 88, 8, 4], [0.12,  1, 0, 1], [0.80, 29, 3, 3], [0.96, 82, 8, 4],
    [0.22,  4, 0, 1], [0.84, 32, 4, 3], [0.91, 78, 7, 4], [0.19,  2, 0, 1], [0.86, 34, 3, 3],
    [0.94, 85, 8, 4], [0.14,  1, 0, 1], [0.81, 31, 3, 3], [0.97, 90, 9, 4], [0.25,  5, 0, 1],
    [0.83, 33, 4, 3], [0.93, 81, 7, 4], [0.16,  2, 0, 1], [0.87, 36, 3, 3], [0.90, 76, 7, 4],
    [0.21,  3, 0, 1], [0.79, 27, 3, 3], [0.98, 89, 8, 4], [0.11,  1, 0, 1], [0.85, 35, 4, 3],
    [0.95, 84, 8, 4], [0.23,  4, 0, 1], [0.82, 32, 3, 3], [0.92, 79, 7, 4], [0.17,  2, 0, 1],
    [0.89, 37, 4, 3], [0.99, 91, 9, 4], [0.13,  1, 0, 1], [0.80, 30, 3, 3], [0.94, 83, 8, 4],
    [0.24,  4, 0, 1], [0.86, 34, 4, 3], [0.96, 86, 8, 4], [0.18,  2, 0, 1], [0.88, 35, 3, 3],
    [0.91, 77, 7, 4], [0.15,  1, 0, 1], [0.83, 31, 3, 3], [0.97, 88, 8, 4], [0.20,  3, 0, 1],
    [0.85, 33, 4, 3], [0.93, 80, 7, 4], [0.12,  1, 0, 1], [0.81, 29, 3, 3], [0.95, 85, 8, 4]
])

# Normalize feature scales
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0) + 1e-8
X_data = (X_raw - X_mean) / X_std

# Target class labels (0 = Basic, 1 = Pro, 2 = Enterprise)
y_raw = np.array([
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2
])

# One-hot encoding targets (60, 3)
num_classes = 3
y_data = np.zeros((y_raw.size, num_classes))
y_data[np.arange(y_raw.size), y_raw] = 1.0

"""
Day 22 Tasks & Individual Expected Outputs
--------------------------------------------

TASK 1 — Multi-Class Activation Engine (T2): Implement ReLU, its derivative, and a numerically stabilized Softmax activation function.

Expected Output:
[TASK 1 SUCCESS] Multi-Class Activation Engine Verified:
ReLU(-2.5) = 0.0000 | ReLU'(2.5) = 1.0000
Softmax Probabilities Sum: 1.0000 | Sample Output: [0.0900, 0.2447, 0.6652]


TASK 2 — Modular Layer Architecture Engine (T1): Construct a self-contained Dense Layer module tracking weights, biases, activations, and velocity vectors for Momentum optimization.

Expected Output:
[TASK 2 SUCCESS] Dense Layer Initialized:
Layer 1 (He Init) Weights Shape: (4, 16) | Biases Shape: (1, 16)
Layer 2 (Xavier Init) Weights Shape: (16, 3) | Biases Shape: (1, 3)


TASK 3 — Arbitrary N-Layer Modular Forward Pass (T3): Construct an N-Layer network wrapper executing dynamic forward operations through hidden ReLU layers to a final Softmax output layer.

Expected Output:
[TASK 3 SUCCESS] Dynamic Forward Propagation Complete:
Output Prediction Matrix Shape: (60, 3)
Sample Class Probabilities (First 3): [[0.3168, 0.4078, 0.2754], [0.3802, 0.3642, 0.2556], [0.4158, 0.3448, 0.2394]]


TASK 4 — Categorical Cross-Entropy (CCE) Loss Engine (T2): Calculate Categorical Cross-Entropy Loss with log-clipping (1e-15) for multi-class targets.

Expected Output:
[TASK 4 SUCCESS] Initial CCE Loss Engine Computed:
Initial Un-optimized Loss: 1.3421


TASK 5 — Generalized Dynamic Backpropagation Engine (T3): Compute automated backpropagation gradients backwards across arbitrary hidden layers using unified Softmax + CCE derivative (dZ = A - Y).

Expected Output:
[TASK 5 SUCCESS] Backpropagation Engine Execution:
Output Layer Gradient dW Shape: (8, 3) | Hidden Layer 1 Gradient dW Shape: (4, 16)
Layer 1 Gradient Norm: 0.0812 | Layer 3 Gradient Norm: 0.1542


TASK 6 — Momentum SGD Optimization Engine (T4): Apply velocity-based SGD updates (v = beta * v + (1 - beta) * dW, W = W - lr * v) to accelerate convergence.

Expected Output:
[TASK 6 SUCCESS] Momentum SGD Optimizer Applied:
Layer 1 Velocity Norm: 0.008120
Layer 1 Weight Delta Norm: 0.000812
Velocity Integration Confirmed: True


TASK 7 — End-to-End Modular Training Execution (T1-T4): Train an architecture [4, 16, 8, 3] for 120 epochs using learning rate lr = 0.1 and momentum beta = 0.9.

Expected Output:
[TASK 7 SUCCESS] Modular Neural Network Training Progress:
Epoch 001/120 | CCE Loss: 1.3421
Epoch 030/120 | CCE Loss: 0.5124
Epoch 060/120 | CCE Loss: 0.2145
Epoch 090/120 | CCE Loss: 0.0982
Epoch 120/120 | CCE Loss: 0.0451


TASK 8 — Argmax Multi-Class Decision Thresholding (T5): Convert raw Softmax probability distributions into discrete multi-class target predictions using argmax.

Expected Output:
[TASK 8 SUCCESS] Argmax Classification Thresholding Applied:
Discrete Class Predictions Shape: (60,)
Predicted Class Distribution -> Basic (0): 20 | Pro (1): 20 | Enterprise (2): 20


TASK 9 — Multi-Class Model Evaluation Engine (T5): Compute categorical accuracy score comparing predicted classes against ground truth.

Expected Output:
[TASK 9 SUCCESS] Categorical Accuracy Evaluated:
Training Classification Accuracy: 100.00%
Correct Classifications: 60 / 60 Samples


TASK 10 — Multi-Class Real-Time Single-Sample Inference Engine (T5): Build a real-time prediction pipeline that accepts new raw sample vectors and returns confidence scores and assigned class labels.

Expected Output:
[TASK 10 SUCCESS] Single-Sample Inference Execution:
Raw Sample Input: [[0.95, 80.0, 8.0, 4.0]]
Normalized Vector: [[1.6214, 1.6841, 1.2514, 1.3412]]
Class Probabilities -> Basic: 0.0012 | Pro: 0.0215 | Enterprise: 0.9773
Assigned Tier Label: ENTERPRISE


Overall Output Summary
-------------------------
========== WEEK 5 DAY 22: MODULAR DEEP LEARNING ENGINE COMPLETE ==========

Architecture Parameters:
- Total Network Layers       : 3 Parameterized Layers
- Layer Dimensions          : [4, 16, 8, 3]
- Hidden Activation         : ReLU (He Initialized)
- Output Activation         : Softmax (Xavier Initialized)
- Dataset Records           : 60 Samples (3 Classes)

Training Initialization:
- Loss Function             : Categorical Cross-Entropy (CCE)
- Optimizer                 : Momentum SGD (Beta = 0.9)
- Initial Loss (Epoch 1)    : 1.3421
- Learning Rate (Alpha)     : 0.1

Optimization Progress:
- Epoch 30 Loss             : 0.5124
- Epoch 60 Loss             : 0.2145
- Epoch 90 Loss             : 0.0982
- Epoch 120 Loss            : 0.0451

Evaluation Metrics:
- Final Training Loss       : 0.0451
- Categorical Accuracy      : 100.00%
- Decision Engine           : Multi-Class Argmax

Engine Outcome: Modular N-Layer Engine with Momentum SGD built and verified successfully.
"""