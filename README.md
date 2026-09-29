# Deep Learning Projects — Project 1 to Capstone

A practical Deep Learning implementation series using Python, NumPy, Pandas, and PyTorch.

## Project Roadmap

| Project | Focus | Status |
|---|---|---|
| Project 1 | Custom Deep Neural Network Engine from Scratch | Completed |
| Project 2 | Multi-Layer Neural Network with Softmax & Momentum | Completed |
| Project 3 | End-to-End PyTorch Deep Learning Workflow | Completed |
| Project 4 | PyTorch Workflow & Checkpointing | Skipped — duplicate of Project 3 |
| Project 5 | Fine-Tuning & Transfer Learning Engine | Completed |
| Project 6 | Regularization, LR Scheduling & Gradient Clipping | Completed |
| Capstone | End-to-End Regularized Deep Learning Pipeline | Completed |

---

# Project 1 — Custom Deep Neural Network Engine from Scratch

## Objective

Build a small neural-network engine using NumPy and Pandas to understand the mechanics of neural-network training.

## Architecture

```text
Input: 4 features
      ↓
Dense Layer: 8 neurons
      ↓
ReLU
      ↓
Dense Layer: 1 neuron
      ↓
Sigmoid
      ↓
Binary Classification
```

Architecture: `4 → 8 → 1`

## Dataset

Synthetic customer-churn style data with:

- Normalized Monthly Spend
- App Sessions
- Support Tickets
- Account Tier Rank

Target: binary classification.

## Concepts Implemented

- Weights and biases
- Forward propagation
- ReLU
- Sigmoid
- Binary Cross-Entropy
- Backpropagation
- Chain rule
- Gradient calculation
- Gradient descent
- Learning rate
- Training loop
- Parameter updates
- Single-sample inference
- Preprocessing consistency between training and inference

## Training Flow

```text
Input → Forward Pass → Prediction → Loss
      → Backpropagation → Gradients
      → Gradient Descent → Updated Parameters
      → Repeat
```

---

# Project 2 — Modular Multi-Layer Neural Network Engine

## Objective

Build a modular multi-class neural-network engine supporting Softmax, Cross-Entropy, dynamic layers, initialization strategies, and Momentum.

## Architecture

```text
4 → 16 → 8 → 3
```

```text
Input
 ↓
Dense 16
 ↓
ReLU
 ↓
Dense 8
 ↓
ReLU
 ↓
Dense 3
 ↓
Softmax
 ↓
3-Class Prediction
```

## Dataset

Synthetic customer-tier classification:

- Basic
- Pro
- Enterprise

## Concepts Implemented

- Modular Dense layers
- Dynamic forward propagation
- Dynamic backward propagation
- Multi-class classification
- One-hot encoding
- Numerically stable Softmax
- Categorical Cross-Entropy
- He/Xavier initialization
- Momentum optimization
- Argmax class prediction
- Single-sample inference

## Result

Reference run:

```text
Final Loss: ~0.0330
Accuracy: 100%
```

Exact numerical values can vary with initialization and execution state.

---

# Project 3 — End-to-End PyTorch Deep Learning Workflow

## Objective

Translate neural-network mechanics into a complete PyTorch workflow.

## Architecture

```text
4 → 16 → 8 → 3
```

```text
Input
 ↓
Linear(4,16)
 ↓
ReLU
 ↓
Linear(16,8)
 ↓
ReLU
 ↓
Linear(8,3)
 ↓
Raw Logits
```

CrossEntropyLoss was used, so Softmax was not placed inside the model.

## Dataset

60 synthetic customer records with:

- Spend
- App Usage
- Support Tickets
- Account Age

Classes:

- Basic
- Pro
- Enterprise

## Tasks Covered

### T1 — PyTorch Tensors

- Tensor creation
- CPU/CUDA device allocation
- dtype
- reshaping
- slicing

### T2 — Autograd

Implemented:

```python
requires_grad=True
loss.backward()
x.grad
```

### T3 — Dataset & DataLoader

Custom Dataset using:

- `__len__`
- `__getitem__`

DataLoader:

```text
batch_size = 16
shuffle = True
```

### T4 — PyTorch Model

Used:

- `nn.Module`
- `nn.Sequential`
- `nn.Linear`
- `nn.ReLU`

### T5 — Training Step

```text
optimizer.zero_grad()
        ↓
model(inputs)
        ↓
loss_fn(predictions, targets)
        ↓
loss.backward()
        ↓
optimizer.step()
```

### T6 — Training Loop

100 epochs using Adam.

Reference losses:

```text
Epoch 1   : 1.1024
Epoch 25  : 0.6512
Epoch 50  : 0.3214
Epoch 75  : 0.1425
Epoch 100 : 0.0612
```

### T7 — Checkpointing

Saved:

- model state dictionary
- optimizer state dictionary
- epoch
- loss

File:

```text
model_checkpoint.pt
```

### T8 — Model Restoration

Loaded the saved model state into a new model and verified matching predictions.

### T9 — Evaluation

Training-set accuracy:

```text
100%
60 / 60
```

### T10 — Single-Sample Inference

```text
New Customer
 ↓
Tensor
 ↓
Model
 ↓
Logits
 ↓
Softmax
 ↓
Probabilities
 ↓
Predicted Class
```

## Key Learning

```text
Tensor
 ↓
Dataset
 ↓
DataLoader
 ↓
Model
 ↓
Loss
 ↓
Autograd
 ↓
Optimizer
 ↓
Training
 ↓
Checkpoint
 ↓
Evaluation
 ↓
Inference
```

---

# Project 4 — PyTorch Workflow & Checkpointing

## Status

**Skipped intentionally.**

Project 4 was effectively a duplicate of Project 3.

It repeated:

- the same dataset structure
- the same `4 → 16 → 8 → 3` architecture
- tensors
- autograd
- Dataset/DataLoader
- training loop
- checkpointing
- evaluation
- inference

The project statement also did not provide a new dataset that justified repeating the implementation.

Therefore Project 4 was skipped to avoid redundant work.

---

# Project 5 — Fine-Tuning & Transfer Learning Engine

## Objective

Build a modular transfer-learning and fine-tuning pipeline using a simulated pretrained backbone.

## Architecture

### Backbone

```text
512 → 256 → 128
```

with:

- Linear
- BatchNorm
- ReLU

### Classification Head

```text
128
 ↓
Dropout(0.3)
 ↓
64
 ↓
ReLU
 ↓
3
```

Complete flow:

```text
Input 512
    ↓
Backbone Block 1
512 → 256
    ↓
Backbone Block 2
256 → 128
    ↓
Dropout
    ↓
Linear 128 → 64
    ↓
ReLU
    ↓
Linear 64 → 3
```

## Concepts Implemented

### Transfer Learning

Reuse learned representations and attach a new task-specific head.

### Freezing

```python
parameter.requires_grad = False
```

### Feature Extraction

Phase 1:

```text
Backbone → Frozen
Head     → Trainable
```

### Selective Unfreezing

Phase 2:

```text
Block 1 → Frozen
Block 2 → Trainable
Head    → Trainable
```

### Differential Learning Rates

```text
Block 2 → 0.0001
Head    → 0.001
```

### BatchNorm

Used `nn.BatchNorm1d` and studied its training/evaluation behavior.

### Dropout

Used:

```python
nn.Dropout(0.3)
```

### Two-Phase Training

```text
Phase 1
Backbone frozen
      ↓
Train head
      ↓
Phase 2
Unfreeze upper backbone block
      ↓
Fine-tune backbone + head
```

## Important Note

The backbone was simulated using linear blocks. It was not an actual pretrained ResNet, VGG, EfficientNet, or ViT.

---

# Project 6 — Regularization, LR Scheduling & Gradient Clipping Engine

## Objective

Build a PyTorch pipeline using regularization, learning-rate scheduling, gradient clipping, validation, and dynamic best-model checkpointing.

## Architecture

```text
256
 ↓
Linear 256 → 128
 ↓
BatchNorm
 ↓
ReLU
 ↓
Dropout(0.4)
 ↓
Linear 128 → 64
 ↓
BatchNorm
 ↓
ReLU
 ↓
Dropout(0.4)
 ↓
Linear 64 → 3
```

## Concepts Introduced

### L2 Regularization / Weight Decay

Implemented:

```python
weight_decay=1e-4
```

Conceptually:

```text
Total Objective
=
Data Loss + Regularization Penalty
```

### Learning-Rate Scheduling

Used:

```python
CosineAnnealingLR
```

with:

```text
Initial LR = 0.005
Minimum LR = 0.00001
T_max = 20
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

Purpose: prevent excessively large gradients and improve training stability.

### Validation

Tracked:

- training loss
- validation loss
- validation accuracy

### Best-Model Checkpointing

Saved the model whenever validation loss reached a new minimum.

## Training Order

```text
Forward
 ↓
Loss
 ↓
Backward
 ↓
Gradient Clipping
 ↓
Optimizer Step
 ↓
Scheduler Step
```

---

# Week 5 Capstone — End-to-End Regularized Deep Learning Pipeline

## Objective

Combine the major PyTorch concepts from the previous projects into one end-to-end pipeline.

## Dataset

Synthetic numerical dataset:

```text
Samples  : 120
Features : 256
Classes  : 3
Batch    : 16
```

Custom Dataset:

```python
class SyntheticDataset(Dataset):
    ...
```

## Model

```text
256 → 128 → 64 → 3
```

with:

- BatchNorm1d
- ReLU
- Dropout(0.4)

## Optimizer

```text
Adam
Learning Rate = 0.005
Weight Decay = 0.0001
```

## Loss

```text
CrossEntropyLoss
```

## Training

20 epochs with:

- training loss
- validation loss
- validation accuracy
- gradient clipping
- learning-rate scheduling

## Scheduler

```text
CosineAnnealingLR
T_max = 20
eta_min = 1e-5
```

## Gradient Clipping

```text
max_norm = 1.0
```

## Validation Split

```text
Training   = 96 samples
Validation = 24 samples
```

## Best Checkpoint

Saved to:

```text
best_model.pt
```

with:

- `model_state_dict`
- epoch
- best validation loss

## Architecture Preview

Implemented an evaluator-oriented:

```python
architecture_preview()
```

with six pipeline stages:

```text
dataset_1
model_2
optimizer_3
scheduler_4
trainer_5
checkpoint_6
```

## Actual Training Observation

The practice dataset used randomly generated features and labels:

```python
X = torch.randn(120,256)
y = torch.randint(0,3,(120,))
```

Observed:

```text
Training Loss
1.1442 → 0.0510

Validation Loss
1.0895 → 1.4896
```

Best validation loss:

```text
1.0854058861732483
```

This behavior is expected for random features and random labels because the model can fit training examples without finding a meaningful relationship that generalizes to validation data.

---

# Overall Knowledge Acquired

## Neural Network Fundamentals

- Neurons
- Weights
- Biases
- Activation functions
- Forward propagation
- Loss functions
- Backpropagation
- Gradients
- Gradient descent
- Learning rate
- Parameters vs hyperparameters

## Mathematical Foundations

- Matrix multiplication
- Chain rule
- Gradient calculation
- Stable Softmax
- Cross-Entropy
- Binary Cross-Entropy
- He/Xavier initialization
- Momentum

## PyTorch

- Tensors
- Tensor shapes
- dtype
- CPU/CUDA
- `reshape`
- `view`
- `flatten`
- `unsqueeze`
- `squeeze`
- `permute`
- `transpose`
- `cat`
- `stack`
- Autograd
- Computational graphs
- `nn.Module`
- `nn.Sequential`
- `nn.Linear`
- `nn.ReLU`
- Dataset
- DataLoader
- CrossEntropyLoss
- Adam
- AdamW concepts
- `train()`
- `eval()`
- `no_grad()`

## Training Engineering

- Training loops
- Batch training
- Epochs
- Validation
- Accuracy
- Checkpointing
- State dictionaries
- Optimizer state
- Model restoration
- Single-sample inference

## Regularization

- Dropout
- BatchNorm
- L2 regularization
- Weight decay
- Gradient clipping
- Learning-rate scheduling
- Cosine Annealing
- Best-model checkpointing

## Transfer Learning

- Backbone
- Classification head
- Freezing
- Feature extraction
- Selective unfreezing
- Fine-tuning
- Differential learning rates
- Optimizer parameter groups

---

# Current Deep Learning Position

The completed projects have primarily covered the **general Deep Learning training engine and feed-forward neural networks**.

```text
                    DEEP LEARNING
                         │
          ┌──────────────┴──────────────┐
          │                             │
    Training Engine               Architectures
          │                             │
   ┌──────┼──────┐             ┌────────┼────────┐
   │      │      │             │        │        │
Forward Backprop Optimizer    MLP      CNN      RNN
   │      │      │                       │
Loss  Gradients  Update                  ↓
   └── Regularization              Computer Vision
       Scheduling
       Checkpoints
       Transfer Learning
```

## Completed

```text
MLP / Dense Networks       ✓
PyTorch Core               ✓
Training & Backprop        ✓
Regularization             ✓
Transfer Learning          ✓
Fine-Tuning                ✓
Checkpointing              ✓
```

## Next Major Area

# CNN — Convolutional Neural Networks

The next project will build a small CNN from scratch using PyTorch.

Planned concepts:

- Image tensors
- Channels
- Convolution
- Filters / kernels
- Feature maps
- Stride
- Padding
- Pooling
- CNN architecture
- Flattening
- Image preprocessing
- Data augmentation
- CNN training
- Evaluation
- Single-image inference

---

# Learning Method

Each project follows:

```text
Concept
 ↓
Mathematical Understanding
 ↓
Small Implementation
 ↓
Run & Verify
 ↓
Debug
 ↓
Integrate Into Project
 ↓
Evaluate
```

The emphasis is on understanding why each component exists rather than hard-coding reference outputs.

Reference numerical values can vary because of random initialization, data generation, shuffling, and execution state.

---

## Technology Stack

- Python
- NumPy
- Pandas
- PyTorch
- Jupyter Notebook / Google Colab
- Matplotlib
- scikit-learn concepts
- CPU / CUDA awareness

---

## Next Phase

**CNN — Image Classification from Scratch**

The CNN project will extend the existing PyTorch foundation into computer vision and will be built incrementally rather than using a ready-made pretrained architecture.
