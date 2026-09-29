
## Fine-Tuning & Transfer Learning Engine (Week 5 Day 24-Project-5)

Problem Statement:
-------------------

Modern deep learning workflows often leverage pre-trained vision/sequence backbones rather than training networks from scratch. This project implements an end-to-end modular **PyTorch Transfer Learning Pipeline** using simulated 512-dimensional deep feature maps across a 3-class target dataset (e.g., medical image severity classification).

The pipeline handles:

1. Pre-trained backbone instantiation and initial layer freezing
 (`requires_grad = False`).
2. Custom task-specific classification head attachment (Dropout, Linear, ReLU).
3. Phase 1 training (Feature Extractor Mode with frozen backbone).
4. Selective upper-layer unfreezing (`requires_grad = True`).
5. Phase 2 fine-tuning with differential learning rates (10^-4 for backbone vs 10^-3 for custom head).
6. Evaluation and model serialization.

---

## Tasks

Task 1 — Pre-trained Backbone Architecture & Layer Freezing:
Build a pre-trained feature extractor (`PretrainedBackbone`) containing stacked linear blocks with Batch Normalization and ReLU activations. Implement a method `freeze_backbone()` that iterates over all parameters and sets `requires_grad = False`.

Task 2 — Custom Classification Head Attachment: 
Build a complete model (`TransferLearningClassifier`) that wraps the backbone and appends a custom task-specific classification head using `nn.Sequential` with `nn.Dropout(0.3)`, `nn.Linear(128, 64)`, `nn.ReLU()`, and `nn.Linear(64, 3)`.

Task 3 — Phase 1: Feature Extractor Optimization Loop: 
Instantiate the classifier, invoke `freeze_backbone()`, and train only the classification head parameters (`model.classifier_head.parameters()`) using `torch.optim.Adam` (learning rate = `0.01`) for 10 epochs. Compute and aggregate batch loss metrics.


Task 4 — Selective Unfreezing & Differential Learning Rates:
Implement `unfreeze_upper_layers()` to set `requires_grad = True` only
for `backbone.block2` while keeping `block1` frozen. Construct an optimizer passing parameter dictionary groups with custom learning rates: $10^{-4}$ for the upper backbone and $10^{-3}$ for the custom head.

Task 5 — Phase 2: Fine-Tuning Loop & Model Checkpoint Export: 
Fine-tune the network for 15 epochs using differential learning rates. Evaluate overall model accuracy inside `with torch.no_grad():` in `model.eval()` mode across the `DataLoader`. Export the fine-tuned model state dictionary, loss, and accuracy to `finetuned_model.pt` via `torch.save()`.



Expected outputs task wise:
----------------------------
[TASK 1 & 2 SUCCESS] Transfer Learning Network Initialized:
Backbone Param 1 requires_grad: False
Classifier Head Param 1 requires_grad: True

[TASK 3 SUCCESS] Phase 1 (Feature Extractor Training) Complete:
Phase 1 Final Loss: 0.9842

[TASK 4 SUCCESS] Unfrozen Upper Layers with Differential Learning Rates:
Block1 Freezing Status: False
Block2 Freezing Status: True

[TASK 5 SUCCESS] Phase 2 (Fine-Tuning) Complete:
Phase 2 Fine-Tuned Loss: 0.5213

[SUMMARY & PERSISTENCE SUCCESS] Transfer Learning Pipeline Verification:
- Phase 1 Loss (Feature Extraction) : 0.9842
- Phase 2 Loss (Fine-Tuned Model)   : 0.5213
- Final Fine-Tuned Accuracy         : 86.00%
- Saved Checkpoint File             : 'finetuned_model.pt'
---
