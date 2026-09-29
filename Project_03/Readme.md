"""
Week-5: Day 23 Project: End-to-End PyTorch Deep Learning Workflow
-----------------------------------------------------------------
Topics Covered:
- T1: 17.10a PyTorch Tensors (GPU/CPU Device Allocation, Reshaping, Slicing)
- T2: 17.10b Autograd Engine (requires_grad, Computational Graphs, backward())
- T3: 17.10c Deep Training Loop (zero_grad -> forward -> loss -> backward -> step)
- T4: 17.10d DataLoader & Datasets (Batching, Shuffling, Parallel Feeding)
- T5: 17.10e Checkpointing Engine (Model State Dictionary Saving, Loading & Resuming)

Platform: Google Colab / Python / PyTorch
Libraries: PyTorch (torch), NumPy, Pandas
"""

# ===================================================================
# PROBLEM STATEMENT
# ===================================================================
"""
Problem Statement:
------------------
Transitioning from manual calculus/matrix operations in NumPy to PyTorch's high-level deep learning abstractions. You are required to build a production-ready, modular PyTorch multi-class classification workflow for customer tier assignment (Basic, Pro, Enterprise).

Key Technical Deliverables:
1. PyTorch Tensor Management: Allocate tensors to appropriate hardware devices (CPU/CUDA), re-shape tensor dimensions, and extract tensor slices.
2. Autograd Mechanics: Demonstrate automatic differentiation using explicit computational graphs and compute analytical gradients via .backward().
3. Custom Dataset & DataLoader Pipeline: Construct a PyTorch Dataset subclass (implementing __len__ and __getitem__) and wrap it in a DataLoader to handle batching and shuffling.
4. Sequential Neural Architecture: Assembly of an nn.Module multi-layer classifier using nn.Sequential, linear transformations (nn.Linear), and non-linear activations (nn.ReLU).
5. Explicit Training Loop & Optimization: Implement the line-by-line step sequence (zero_grad -> forward -> CrossEntropyLoss -> backward -> step) using Adam optimizer.
6. Model Checkpointing & Restoration: Serialize model parameters, optimizer state, epoch counts, and loss metrics to disk using torch.save(), and restore state via torch.load() for uninterrupted training resumption.
7. Batch Evaluation & Single-Sample Inference: Calculate categorical accuracy using torch.no_grad() and perform real-time probability inference on unseen customer records.
"""

# -------------------------------------------------------------------
# Input Data Generation (Synthetic Customer Tier Data - 60 Records)
# Features: [Spend, App Usage, Support Tickets, Account Age]
# Classes: 0 = Basic, 1 = Pro, 2 = Enterprise
# -------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
], dtype=np.float32)

y_raw = np.array([
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2
], dtype=np.int64)

# Feature Scaling
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0) + 1e-8
X_norm = (X_raw - X_mean) / X_std

"""
Day 23 Tasks & Individual Expected Outputs
--------------------------------------------

TASK 1 — PyTorch Tensor Fundamentals (17.10a): Convert raw NumPy arrays to PyTorch Tensors, handle CPU/GPU hardware device allocation, perform shape reshaping, and extract tensor slices.

Expected Output:
[TASK 1 SUCCESS] Tensor Operations Verified:
Tensor Device: cpu | Tensor Shape: torch.Size([60, 4])
Reshaped Tensor Shape: torch.Size([15, 16]) | Sample Tensor Slice: tensor([-1.2415, -0.7412])


TASK 2 — Autograd Computational Graph Engine (17.10b): Demonstrate automatic differentiation mechanics by setting requires_grad=True, defining a mathematical function, and executing backward() to extract gradients.

Expected Output:
[TASK 2 SUCCESS] Autograd Engine Verified:
Scalar Function Loss Output: tensor(13.0000, grad_fn=<SumBackward0>)
Computed Analytical Gradient (dL/dx): tensor([4., 6.])


TASK 3 — Custom Dataset & DataLoader Pipeline (17.10d): Construct a PyTorch Dataset subclass (implementing __len__ and __getitem__) and wrap it inside a DataLoader configured with batch_size=16 and shuffle=True.

Expected Output:
[TASK 3 SUCCESS] PyTorch DataLoader Active:
Total Dataset Samples: 60 | Configured Batch Size: 16
First Batch Features Shape: torch.Size([16, 4]) | First Batch Labels Shape: torch.Size([16])


TASK 4 — PyTorch nn.Module Multi-Class Classifier (17.10a, 17.10c): Build a deep neural network module subclassing nn.Module with stacked nn.Linear and nn.ReLU layers using nn.Sequential.

Expected Output:
[TASK 4 SUCCESS] PyTorch Neural Network Architecture Assembled:
Model Device Placement: cpu
Sequential Architecture: Sequential(
  (0): Linear(in_features=4, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=8, bias=True)
  (3): ReLU()
  (4): Linear(in_features=8, out_features=3, bias=True)
)


TASK 5 — Line-by-Line Training Step Engine (17.10c): Implement a single explicit training step demonstrating optimizer.zero_grad(), forward propagation, nn.CrossEntropyLoss calculation, loss.backward(), and optimizer.step().

Expected Output:
[TASK 5 SUCCESS] Single PyTorch Step Execution Complete:
Step Pre-Loss: 1.1024 | Post-Step Loss: 1.0851
Gradient Calculated for Output Layer: True


TASK 6 — Multi-Epoch Training Optimization Loop (17.10c): Train the PyTorch model for 100 epochs using the Adam optimizer, aggregating batch losses and logging total loss progress.

Expected Output:
[TASK 6 SUCCESS] PyTorch Training Loop Progress:
Epoch 001/100 | CrossEntropyLoss: 1.1024
Epoch 025/100 | CrossEntropyLoss: 0.6512
Epoch 050/100 | CrossEntropyLoss: 0.3214
Epoch 075/100 | CrossEntropyLoss: 0.1425
Epoch 100/100 | CrossEntropyLoss: 0.0612


TASK 7 — Model Checkpointing & Disk Serialization (17.10e): Save model state_dict, optimizer state_dict, current epoch, and final loss value to disk (.pt file) using torch.save().

Expected Output:
[TASK 7 SUCCESS] Checkpoint Serialized to Disk:
Checkpoint File Saved: 'model_checkpoint.pt'
Saved Epoch State: 100 | Saved Loss Value: 0.0612


TASK 8 — Model State Restoration Engine (17.10e): Instantiation of a new un-trained model instance, loading state_dict from disk using torch.load(), and verifying identical predictions between original and restored models.

Expected Output:
[TASK 8 SUCCESS] Model State Dictionary Successfully Restored:
Pre-Restoration Predictions Match Post-Restoration: True
Model Ready for Interrupted Training Resumption.


TASK 9 — Batch Evaluation Engine (17.10c, 17.10d): Compute categorical classification accuracy across DataLoader batches in evaluation mode (model.eval()) with gradient tracking disabled (torch.no_grad()).

Expected Output:
[TASK 9 SUCCESS] Batch Model Evaluation Complete:
Total Dataset Accuracy: 100.00%
Correct Matches: 60 / 60 Samples


TASK 10 — Real-Time Single-Sample Inference Engine (17.10a): Build a prediction pipeline that accepts a new raw single customer feature record, applies normalization, passes it through the model, and outputs class probability distributions and assigned label.

Expected Output:
[TASK 10 SUCCESS] Single-Sample Inference Output:
Input Sample Tensor: tensor([[1.6214, 1.6841, 1.2514, 1.3412]])
Predicted Probabilities Tensor: tensor([[0.0012, 0.0215, 0.9773]])
Assigned Output Class: 2


Overall Output Summary
-------------------------
========== WEEK 5 DAY 23: PYTORCH WORKFLOW & CHECKPOINTING ENGINE ==========

Architecture Parameters:
- Total Model Layers        : 3 Linear Layers (4 -> 16 -> 8 -> 3)
- Activation Functions      : ReLU (Hidden), Implicit Softmax (Output)
- Dataset Records           : 60 Samples (3 Classes)
- Target Device             : CPU / CUDA

Training & Optimization:
- Loss Function             : PyTorch nn.CrossEntropyLoss
- Optimizer                 : torch.optim.Adam (lr=0.01)
- Data Feeding Engine       : DataLoader (Batch Size: 16, Shuffle: True)
- Initial Loss (Epoch 1)    : 1.1024
- Final Loss (Epoch 100)    : 0.0612

Evaluation & Persistence:
- Model Evaluation Accuracy : 100.00%
- Checkpoint State File     : 'model_checkpoint.pt' (Saved & Verified)
- Inference Execution       : Tensor Single-Sample Pipeline Active

Engine Outcome: Complete PyTorch Pipeline (Tensors, Autograd, DataLoader, Training Loop & Checkpointing) verified successfully.
"""