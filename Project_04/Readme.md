Week-5-Project-4
-----------------

Problem Statement & Architecture
----------------------------------
Transitioning from manual matrix operations in NumPy to PyTorch's deep learning abstractions, this project implements a modular, multi-class classification workflow for assigning
customer account tiers (Basic, Pro, Enterprise). The pipeline handles features including spend, app usage, support tickets, and account age across 60 synthetic customer records.
It implements PyTorch tensors, autograd differentiation, custom datasets/dataloaders, neural network architectures, training step execution, epoch loops, state checkpointing, evaluation, 
and single-sample inference.

Technical Task Breakdown

Task 1 — PyTorch Tensor Fundamentals (17.10a): Convert NumPy arrays to PyTorch Tensors, assign CPU/GPU device hardware, perform shape transformations, and slice tensor sub-elements.   

Task 2 — Autograd Computational Graph (17.10b): Set up leaf nodes with requires_grad=True, execute forward graph calculations, and invoke .backward() to extract analytical gradients via .grad.  

Task 3 — Custom Dataset & DataLoader (17.10d): Build a Dataset subclass with __len__ and __getitem__, then wrap it in a DataLoader (batch_size=16, shuffle=True). 

Task 4 — Neural Network Architecture (17.10a, 17.10c): Construct an nn.Module classifier using nn.Sequential with stacked nn.Linear and nn.ReLU layers. 


Task 5 — Training Step Engine (17.10c): Execute a single training step sequence: optimizer.zero_grad() -> forward pass  -> nn.CrossEntropyLoss  -> loss.backward()  -> optimizer.step().  

Task 6 — Multi-Epoch Optimization (17.10c): Run a 100-epoch training loop using the Adam optimizer, aggregating batch metrics and logging loss progression.   

Task 7 — Checkpoint Serialization (17.10e): Serialize model.state_dict(), optimizer.state_dict(), epoch count, and loss to model_checkpoint.pt using torch.save().   

Task 8 — State Restoration Engine (17.10e): Instantiate an un-trained model, load disk checkpoint states via torch.load(), and verify output equality with torch.allclose().   

Task 9 — Batch Evaluation Engine (17.10c, 17.10d): Calculate multi-class accuracy across DataLoader batches inside with torch.no_grad(): in model.eval() mode.  

Task 10 — Real-Time Inference (17.10a): Standardize an unseen customer feature vector, execute a model forward pass, apply softmax probability scaling, and return the predicted tier. 


Expected Terminal Output
--------------------------
[TASK 1 SUCCESS] Tensor Operations Verified:
Tensor Device: cpu | Tensor Shape: torch.Size([60, 4])
Reshaped Tensor Shape: torch.Size([15, 16]) | Sample Tensor Slice: tensor([-1.2415, -0.7412])

[TASK 2 SUCCESS] Autograd Engine Verified:
Scalar Function Loss Output: tensor(13.0000, grad_fn=<SumBackward0>)
Computed Analytical Gradient (dL/dx): tensor([4., 6.])

[TASK 3 SUCCESS] PyTorch DataLoader Active:
Total Dataset Samples: 60 | Configured Batch Size: 16
First Batch Features Shape: torch.Size([16, 4]) | First Batch Labels Shape: torch.Size([16])

[TASK 4 SUCCESS] PyTorch Neural Network Architecture Assembled:
Model Device Placement: cpu
Sequential Architecture: Sequential(
  (0): Linear(in_features=4, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=8, bias=True)
  (3): ReLU()
  (4): Linear(in_features=8, out_features=3, bias=True)
)

[TASK 5 SUCCESS] Single PyTorch Step Execution Complete:
Step Pre-Loss: 1.1024 | Post-Step Loss: 1.0851
Gradient Calculated for Output Layer: True

[TASK 6 SUCCESS] PyTorch Training Loop Progress:
Epoch 001/100 | CrossEntropyLoss: 1.1024
Epoch 025/100 | CrossEntropyLoss: 0.6512
Epoch 050/100 | CrossEntropyLoss: 0.3214
Epoch 075/100 | CrossEntropyLoss: 0.1425
Epoch 100/100 | CrossEntropyLoss: 0.0612

[TASK 7 SUCCESS] Checkpoint Serialized to Disk:
Checkpoint File Saved: 'model_checkpoint.pt'
Saved Epoch State: 100 | Saved Loss Value: 0.0612

[TASK 8 SUCCESS] Model State Dictionary Successfully Restored:
Pre-Restoration Predictions Match Post-Restoration: True
Model Ready for Interrupted Training Resumption.

[TASK 9 SUCCESS] Batch Model Evaluation Complete:
Total Dataset Accuracy: 100.00%
Correct Matches: 60 / 60 Samples

[TASK 10 SUCCESS] Single-Sample Inference Output:
Input Sample Tensor: tensor([[1.6214, 1.6841, 1.2514, 1.3412]])
Predicted Probabilities Tensor: tensor([[0.0012, 0.0215, 0.9773]])
Assigned Output Class: 2

========== WEEK 5 DAY 23: PYTORCH WORKFLOW & CHECKPOINTING ENGINE ==========

Architecture Parameters:
- Total Model Layers         : 3 Linear Layers (4 -> 16 -> 8 -> 3)
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