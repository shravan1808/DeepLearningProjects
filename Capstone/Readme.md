# Week-5 Capstone Project: End-to-End Regularized Deep Learning Pipeline

## Problem Statement

Build an end-to-end PyTorch deep learning pipeline for a multi-class classification task. The system must process synthetic numerical features using a custom dataset wrapper, feed them through a regularized deep neural network, optimize model parameters using L2 weight decay, clip gradients during backward passes, apply dynamic learning rate scheduling, track performance metrics, and save the best checkpoint to disk.

---

## Deliverables & Tasks

* **Task 1 — Custom Dataset & DataLoader Pipeline:** 
Implement `SyntheticDataset` inheriting from `torch.utils.data.Dataset`. Pass `X` and `y` tensors, implement `__len__` and `__getitem__`, and wrap inside a `DataLoader`.

* **Task 2 — Deep Neural Network Architecture:** 
Build `RegularizedClassifier` (`nn.Module`). Map dimensions $256 \rightarrow 128 \rightarrow 64 \rightarrow 3$ using `nn.Sequential`, incorporating `nn.BatchNorm1d`, `nn.ReLU()`, and `nn.Dropout(p=0.4)`.

* **Task 3 — Optimization & Loss Engine:** Instantiate `torch.optim.Adam` with `lr=0.005` and L2 weight decay `weight_decay=1e-4`. Use `nn.CrossEntropyLoss()`.

* **Task 4 — Evaluation & Metric Tracking:** Implement epoch-level training and validation loops, tracking classification loss and accuracy across training batches.

* **Task 5 — Regularization, LR Scheduler & Checkpointing:** 
Integrate `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)`. Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` during training, and save the best model to `best_model.pt` via `torch.save()`. Wrap execution inside `architecture_preview()` returning pipeline stage tracking dictionaries (`dataset_1`, `model_2`, `optimizer_3`, `scheduler_4`, `trainer_5`, `checkpoint_6`).



Expected Outputs:
------------------
[TASK 1 SUCCESS] Dataset & DataLoader Pipeline Initialized:
Features Shape: torch.Size([120, 256]) | Target Classes: 3
DataLoader Configured with Batch Size: 16

[TASK 2 & 3 SUCCESS] Regularized Classifier & Optimizer Initialized:
Architecture: 256 -> 128 -> 64 -> 3
Dropout Probability: 0.4
Optimizer: Adam (LR: 0.005, Weight Decay: 0.0001)

[TASK 5 SUCCESS] Learning Rate Scheduler Initialized:
Initial LR: 0.005000 | Min Target LR: 0.000010

[TASK 4 & 5 SUCCESS] Training, Evaluation & Gradient Clipping Execution:
Epoch [01/20] - Loss: 1.0542 | Acc: 41.67% | LR: 0.004969 | Checkpoint Saved
Epoch [10/20] - Loss: 0.6120 | Acc: 78.33% | LR: 0.002505
Epoch [20/20] - Loss: 0.3845 | Acc: 91.67% | LR: 0.000010

[CAPSTONE SUMMARY] Pipeline Execution Completed:
- Initial Epoch Loss      : 1.0542
- Best Validation Loss    : 0.3845
- Final Learning Rate     : 0.000010
- Saved Checkpoint File   : 'best_model.pt'