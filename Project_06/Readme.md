Week 5 – Day 25 Project: Model Regularization, Learning Rate Schedulers & Gradient Clipping Engine

Problem Statement
-------------------
Training deep neural networks requires techniques to prevent overfitting, stabilize training dynamics, and accelerate convergence. This project implements a PyTorch Fine-Tuning & Regularization Engine that incorporates advanced training techniques, including learning rate scheduling, gradient clipping, L2 regularization (weight decay), and dynamic model checkpointing.

The engine handles:
1. Model Architecture with Regularization: Instantiation of a multi-layer classifier incorporating Dropout and Batch Normalization.

2. Optimizer Configuration with L2 Regularization: Configuring AdamW/Adam with explicit weight decay (10^-4) to enforce weight regularization.

3. Learning Rate Scheduler Integration: Initializing a learning rate scheduler (CosineAnnealingLR or StepLR) to adjust step sizes dynamically during training.

4. Gradient Clipping Execution: Applying norm-based gradient clipping (torch.nn.utils.clip_grad_norm_) during backward passes to prevent exploding gradients.

5. Evaluation, Best Model Tracking & Serialization: Monitoring validation loss across training epochs, saving only the best model checkpoint (best_model.pt) based on lowest loss, and generating performance summaries.

Task 1 — Model Architecture & Dropout Configuration
Build a classification model (RegularizedClassifier) inheriting from nn.Module.
It must include linear layers transitioning from input dimension 256 ->128 ->64 ->3 target classes, integrated with nn.BatchNorm1d, nn.ReLU(), and nn.Dropout(p=0.4).

Task 2 — Optimizer & Weight Decay Setup
Instantiate torch.optim.Adam (or AdamW) with a primary learning rate of 0.005 and weight decay (weight_decay=1e-4) for L2 regularization. Define nn.CrossEntropyLoss as the optimization criterion.

Task 3 — Learning Rate Scheduler Initialization
Configure a learning rate scheduler using torch.optim.lr_scheduler.CosineAnnealingLR (with T_max=20, eta_min=1e-5) or StepLR (with step_size=5, gamma=0.5). Implement a method or step call to update learning rates after each training epoch.

Task 4 — Training Loop with Gradient Clipping & Scheduler Step
Train the model for 20 epochs. In each epoch, compute backward passes, execute gradient clipping using torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0), step the optimizer, and update the learning rate scheduler (scheduler.step()). Track loss across epochs.

Task 5 — Evaluation & Best Checkpoint Persistence
Evaluate loss after each epoch. Implement a dynamic checkpointing mechanism that saves the model state, current epoch, and best loss to best_model.pt via torch.save() whenever a new minimum loss is achieved.


Expected Outputs:
------------------
[TASK 1 & 2 SUCCESS] Regularized Classifier Initialized:
Dropout Probability: 0.4
Optimizer Weight Decay: 0.0001

[TASK 3 SUCCESS] Learning Rate Scheduler Initialized:
Initial LR: 0.005000 | Min/Final Target LR: 0.000010

[TASK 4 & 5 SUCCESS] Training Loop Completed with Gradient Clipping:
Epoch [1/20] - Loss: 1.0542 | LR: 0.004969 | Checkpoint Saved
Epoch [10/20] - Loss: 0.6120 | LR: 0.002505
Epoch [20/20] - Loss: 0.3845 | LR: 0.000010

[SUMMARY & PERSISTENCE SUCCESS] Regularized Pipeline Verification:
- Initial Epoch Loss      : 1.0542
- Best Validation Loss    : 0.3845
- Final Learning Rate     : 0.000010
- Saved Checkpoint File   : 'best_model.pt'