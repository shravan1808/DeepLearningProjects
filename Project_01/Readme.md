"""
Week-5: Day 21 Project: Custom Deep Neural Network Engine from Scratch
----------------------------------------------------------------------
Topics Covered:
- T1: What is a Neural Network (Multi-Layer Perceptron Architecture)
- T2: Weights, Bias, and Activation Functions (ReLU & Sigmoid)
- T3: Forward Pass (Matrix Multiplication & Layer Propagation)
- T4: Backpropagation & Gradient Descent (Chain Rule & SGD Updates)
- T5: Loss Functions (Binary Cross-Entropy Loss Calculation)

Platform: Google Colab / Python
Libraries: NumPy, Pandas
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Input Data Generation (Synthetic Customer Churn Metrics - 50 Records)
# Features: [Normalized Monthly Spend, App Sessions, Support Tickets, Account Tier Rank]
# -------------------------------------------------------------------
np.random.seed(42)

X_data = np.array([
    [0.45, 12, 1, 2], [0.89, 35, 4, 3], [0.25, 2, 0, 1], [0.98, 85, 8, 4], [0.78, 28, 3, 3],
    [0.35, 8, 1, 2],  [0.20, 3, 0, 1],  [0.95, 38, 5, 3], [0.99, 92, 9, 4], [0.40, 10, 1, 2],
    [0.22, 2, 0, 1],  [0.85, 31, 3, 3], [0.42, 11, 1, 2], [0.92, 78, 7, 4], [0.28, 3, 0, 1],
    [0.92, 36, 4, 3], [0.38, 7, 1, 2],  [0.99, 88, 8, 4], [0.21, 2, 0, 1],  [0.88, 33, 3, 3],
    [0.41, 9, 1, 2],  [0.24, 3, 0, 1],  [0.96, 40, 4, 3], [0.97, 75, 7, 4], [0.39, 8, 1, 2],
    [0.18, 1, 0, 1],  [0.90, 34, 3, 3], [0.99, 90, 9, 4], [0.26, 4, 0, 1],  [0.84, 32, 3, 3],
    [0.43, 10, 1, 2], [0.37, 8, 1, 2],  [0.98, 81, 8, 4], [0.19, 1, 0, 1],  [0.99, 42, 5, 3],
    [0.44, 9, 1, 2],  [0.99, 89, 8, 4], [0.23, 3, 0, 1],  [0.91, 35, 4, 3], [0.36, 11, 1, 2],
    [0.27, 2, 0, 1],  [0.93, 37, 4, 3], [0.96, 80, 7, 4], [0.40, 12, 1, 2], [0.18, 1, 0, 1],
    [0.94, 38, 4, 3], [0.99, 86, 8, 4], [0.33, 7, 1, 2],  [0.87, 31, 3, 3], [0.38, 9, 1, 2]
])

# Normalize feature scales for numerical stability
X_data = (X_data - X_data.mean(axis=0)) / (X_data.std(axis=0) + 1e-8)

# Binary Target Labels (0 = Retained, 1 = Churned)
y_data = np.array([
    [0], [1], [0], [1], [1], [0], [0], [1], [1], [0],
    [0], [1], [0], [1], [0], [1], [0], [1], [0], [1],
    [0], [0], [1], [1], [0], [0], [1], [1], [0], [1],
    [0], [0], [1], [0], [1], [0], [1], [0], [1], [0],
    [0], [1], [1], [0], [0], [1], [1], [0], [1], [0]
])

"""
Day 21 Tasks & Individual Expected Outputs
--------------------------------------------

TASK 1 — Activation Engine (T2): Implement ReLU and Sigmoid activation functions along with their first derivatives.

Expected Output:
[TASK 1 SUCCESS] Activation Engine Verified:
ReLU(-2.5) = 0.0000 | ReLU'(2.5) = 1.0000
Sigmoid(0.0) = 0.5000 | Sigmoid' Derivative Verified


TASK 2 — Network Initialization (T1, T2): Build a 2-Layer Neural Network initialization setup (Input Size: 4, Hidden Neurons: 8, Output Neurons: 1) with small random weights and zero-initialized biases.

Expected Output:
[TASK 2 SUCCESS] Network Architecture Initialized:
W1 Shape: (4, 8) | b1 Shape: (1, 8)
W2 Shape: (8, 1) | b2 Shape: (1, 1)


TASK 3 — Deep Forward Pass (T3): Implement the complete forward propagation step (Z1 = X*W1 + b1 -> A1 = ReLU(Z1) -> Z2 = A1*W2 + b2 -> A2 = Sigmoid(Z2)).

Expected Output:
[TASK 3 SUCCESS] Forward Propagation Complete:
A2 Predictions Matrix Shape: (50, 1)
Sample Probabilities (First 3): [[0.5002], [0.5001], [0.4998]]
Min Output Probability: 0.4995 | Max Output Probability: 0.5008


TASK 4 — Binary Cross-Entropy Loss Engine (T5): Calculate Binary Cross-Entropy Loss between predictions (A2) and targets (y) with numerical clipping (1e-15) to prevent log(0).

Expected Output:
[TASK 4 SUCCESS] Initial BCE Loss Engine Computed:
Initial Un-optimized Loss: 0.6931 (Matches theoretical log(2) random guess baseline)


TASK 5 — Backpropagation & Derivative Chain Rule Engine (T4): Compute exact partial derivatives layer-by-layer (dZ2 = A2 - y, dW2 = A1^T * dZ2 / N, dA1 = dZ2 * W2^T, dZ1 = dA1 * ReLU'(Z1), dW1 = X^T * dZ1 / N).

Expected Output:
[TASK 5 SUCCESS] Backpropagation Gradients Computed:
dW1 Gradient Shape: (4, 8) | db1 Gradient Shape: (1, 8)
dW2 Gradient Shape: (8, 1) | db2 Gradient Shape: (1, 1)
dW2 Norm: 0.0412 | dW1 Norm: 0.0125


TASK 6 — Stochastic Gradient Descent (SGD) Optimizer (T4): Apply learning rate updates (W = W - alpha * dW, b = b - alpha * db) across parameters.

Expected Output:
[TASK 6 SUCCESS] SGD Parameter Update Applied:
W2 Delta Norm (Change): 0.004120
b2 Delta Norm (Change): 0.001042
Weight Drift Confirmed: True


TASK 7 — End-to-End Training Loop Execution (T1-T5): Train the custom network for 100 epochs with learning rate alpha = 0.1, logging loss every 20 epochs.

Expected Output:
[TASK 7 SUCCESS] Training Loop Progress:
Epoch 001/100 | BCE Loss: 0.6931
Epoch 020/100 | BCE Loss: 0.5412
Epoch 040/100 | BCE Loss: 0.3891
Epoch 060/100 | BCE Loss: 0.2415
Epoch 080/100 | BCE Loss: 0.1420
Epoch 100/100 | BCE Loss: 0.0815


TASK 8 — Boundary & Class Inference Engine (T1, T3): Convert final sigmoid probability outputs into binary classifications using a 0.5 decision threshold.

Expected Output:
[TASK 8 SUCCESS] Binary Classification Thresholding Applied:
Binary Predictions Shape: (50, 1)
Class Distribution -> Retained (0): 29 | Churned (1): 21


TASK 9 — Model Evaluation & Classification Accuracy (T5): Calculate accuracy score by comparing final binary predictions against true ground truth labels.

Expected Output:
[TASK 9 SUCCESS] Classification Accuracy Evaluated:
Training Accuracy: 100.00%
Correct Matches: 50 / 50 Records


TASK 10 — Single-Sample Inference Engine (T3): Construct an execution function to take a single new customer record and output the raw probabilities and final risk classification.

Expected Output:
[TASK 10 SUCCESS] Single-Sample Inference Execution:
Raw Inputs: [[0.95, 80.0, 7.0, 4.0]]
Normalized Input Vector: [[1.6214, 1.6841, 1.2514, 1.3412]]
Predicted Churn Probability: 0.9412
Assigned Class: CHURN


Overall Output Summary
-------------------------
========== WEEK 5 DAY 21: NEURAL NETWORK ENGINE FROM SCRATCH ==========

Architecture Parameters:
- Input Features           : 4
- Hidden Layer Neurons     : 8 (Activation: ReLU)
- Output Neurons           : 1 (Activation: Sigmoid)
- Dataset Records          : 50 Samples

Training Initialization:
- Loss Function            : Binary Cross-Entropy (BCE)
- Optimizer                : Vanilla Gradient Descent (SGD)
- Initial Loss (Epoch 1)   : 0.6931
- Learning Rate (Alpha)    : 0.1

Optimization Progress:
- Epoch 20 Loss            : 0.5412
- Epoch 40 Loss            : 0.3891
- Epoch 60 Loss            : 0.2415
- Epoch 80 Loss            : 0.1420
- Epoch 100 Loss           : 0.0815

Evaluation Metrics:
- Final Training Loss      : 0.0815
- Classification Accuracy  : 100.00%
- Decision Threshold       : 0.50

Engine Outcome: Custom Neural Network trained successfully using pure NumPy forward and backward passes.
"""