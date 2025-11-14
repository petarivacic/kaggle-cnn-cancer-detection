# 1. Challenge Overview

This project addresses the Kaggle _Histopathologic Cancer Detection_ challenge. The task is to classify small 96×96 RGB histopathology patches extracted from lymph node tissue and determine whether the **central 32×32 region** contains metastatic cancer. Because each patch is a tiny crop from a much larger whole-slide image, the model must learn to recognize subtle textural and morphological cues rather than global anatomical structure.

From a clinical perspective, automating this process reduces pathologist workload and shortens diagnostic turnaround times. More importantly, the cost of a false negative (missed metastasis) is extremely high in real screening workflows. For this reason, we evaluate models not only by accuracy but also by **ROC AUC**, which captures how well predicted probabilities rank malignant versus benign tissue patches.

**Dataset composition:**

- **Training set:** 220,025 labeled 96×96×3 tiles
- **Test set:** 57,000 unlabeled tiles (Kaggle leaderboard)
- **Local validation:** 20% split from the training set to support hyperparameter tuning and model comparison

# 2. Exploratory Data Analysis

The dataset is derived from the PatchCamelyon (PCam) benchmark, which consists of small 96×96 RGB patches extracted from whole-slide histopathology images of lymph node tissue. Each patch is labeled as metastatic (1) or non-metastatic (0), turning metastasis detection into a binary image classification task that can be trained on a single GPU.

Although the original PCam dataset is close to balanced, the subset used in this project is **moderately imbalanced**, with substantially more benign tiles than metastatic ones (~130,907 benign vs ~89,116 metastatic). This imbalance does not require heavy class reweighting, but it does influence evaluation: accuracy alone becomes less informative, and metrics such as **ROC-AUC** provide a more reliable measure of ranking quality. The imbalance also motivates the use of data augmentation and regularization to help the model generalize across the less frequent metastatic class.

**Qualitative observations:**

- **Negative patches (0)**
  Often contain more homogeneous tissue structure, with regular glandular or stromal patterns and relatively uniform nuclei distribution. There may be large regions of background or fat with low cellular density.

- **Positive patches (1)**
  Tend to exhibit clusters of densely packed, irregular nuclei with heterogeneous staining. Nuclear shapes and textures are more variable, reflecting the chaotic growth typical of metastatic tissue.

There is also noticeable variability in staining intensity and color balance across samples, reflecting expected differences in H&E staining procedures, tissue thickness, and scanner characteristics. This variation further motivates augmentation (rotations, flips, shifts, zooms) to improve robustness.

A histogram-based inspection shows that benign tissue often exhibits more uniform brightness, whereas metastatic tissue displays a bimodal distribution with peaks around ~120 and ~220, consistent with the heterogeneous nature of cancerous regions.

For training and validation, the labeled dataset is split into a training portion and a held-out 20% local validation set. The Kaggle test set remains unlabeled. The validation set is used to compare model configurations using ROC-AUC as the primary metric, supplemented by accuracy. ROC-AUC is particularly suitable because false negatives carry heavy clinical cost, making ranking quality more informative than raw accuracy.

# 3. Model Architecture

We evaluate two convolutional neural network (CNN) architectures: a compact **baseline model** and a more expressive **improved model** featuring channel attention and deeper convolutional blocks. Both networks operate on 96×96 RGB patches and output a single sigmoid probability for metastatic presence.

## 3.1 Baseline CNN

The baseline architecture follows a classic convolutional design suitable for small medical-image patches. It consists of **three convolutional blocks**, each increasing in depth, followed by a fully connected classifier.

**Each block includes:**

- Conv2D with filters **32 → 64 → 128**
- Batch Normalization
- ReLU activation
- MaxPooling2D (2×2)
- Dropout

After the final pooling layer, the feature maps (8×8×128) are **flattened** and fed into a fully connected classification head:

- Dense(256)
- Batch Normalization, ReLU and Dropout
- Final Dense(1) with sigmoid activation

The model contains approximately **2.39 million parameters**, most of them in the dense layer. Despite its simplicity, it trains reliably and provides a meaningful performance baseline.

## 3.2 Improved CNN with Channel Attention

The improved architecture aims to capture richer morphological patterns by increasing representational capacity and incorporating **squeeze-and-excitation (SE) modules**, which adaptively reweight channels based on global context.

The model consists of **three stages**, each containing:

- Two Conv2D layers (instead of one)
- Higher filter counts: **48 → 96 → 192**
- Batch Normalization and ReLU activation
- **Squeeze-and-Excitation block:**

  - GlobalAveragePooling
  - Small two-layer MLP (Dense → ReLU → Dense → sigmoid)
  - Channel-wise attention reweighting

- MaxPooling2D
- Dropout

After the final stage:

- Global Average Pooling over the 12×12×384 feature map
- Dense(256) with BatchNorm and Dropout
- Final Dense(1) sigmoid output

This design uses ~**1.42 million parameters**, making it _more parameter-efficient_ than the baseline despite being deeper and significantly more expressive.

## 3.3 Architectural Rationale

The baseline offers a simple reference point, while the improved network shifts representational power into the convolutional pathway, where it is more effective for extracting the local textural features characteristic of metastatic tissue.

**Motivations for the improved design:**

- Deeper feature extraction via double-conv blocks
- Channel attention that highlights clinically relevant structures
- Balanced parameter distribution (fewer dense-layer parameters)
- Better inductive bias through global average pooling

These enhancements lead to substantially better validation performance while reducing total parameter count.

# 4. Results & Analysis

We compare the baseline CNN and the improved CNN across several learning rates and dropout configurations. Below is a summary of how the models performed.

### **Learning rate: 1e-4**

- **baseline_0.0001:** accuracy ~0.74, ROC-AUC ~0.91
- **improved_0.0001:** accuracy ~0.87, ROC-AUC ~0.94

Even with a small learning rate, the improved network provides clear gains in both accuracy and probability ranking.

### **Learning rate: 5e-4**

- **baseline_0.0005:** accuracy ~0.83, ROC-AUC ~0.93, val_loss ~0.40
- **improved_0.0005:** accuracy ~0.893, ROC-AUC ~0.957, val_loss ~0.27

This is the best-performing configuration overall. The improved CNN outperforms the baseline in every metric and shows significantly better calibration.

### **Learning rate: 1e-3**

- **baseline_0.001:** accuracy ~0.762, ROC-AUC ~0.934, val_loss ~0.62
- **improved_0.001:** accuracy ~0.874, ROC-AUC ~0.944, val_loss ~0.30

The higher LR destabilizes the baseline but the improved model remains robust.

## **Summary of Findings**

Across all learning rates:

- The improved CNN **dominates** the baseline CNN.
- The best model is **improved_0.0005**, achieving ROC-AUC ≈ **0.96**.
- Moderate dropout improves generalization.
- Very small LRs slow convergence; very large ones destabilize shallow models.

Given that top Kaggle solutions often achieve ROC-AUC in the mid-0.95+ range using heavy architectures and ensembling, achieving ~0.96 with a custom lightweight CNN is a strong outcome for a course project.

# 5. Conclusion

This project tackles metastatic cancer detection from histopathology patches using the PCam dataset. Two CNN architectures were implemented and compared: a simple baseline model and an improved model with deeper convolutional blocks, progressive channel expansion and squeeze-and-excitation attention mechanisms.

After performing exploratory analysis and setting up augmentation and regularization strategies, both models were trained under multiple learning rates and dropout settings. The improved architecture consistently outperformed the baseline across all configurations. The best model achieved validation ROC-AUC of ~0.96 and accuracy of ~0.89, surpassing the baseline range of 0.91–0.94 ROC-AUC.

These results show that:

- Thoughtful architectural upgrades can yield strong performance without relying on transfer learning or ensembling.
- Regularization and careful hyperparameter tuning are essential in medical imaging tasks.
- There is still room for improvement using deeper backbones (ResNet, DenseNet, EfficientNet), longer training, or optimization strategies such as cosine annealing or warm restarts.

Overall, the project demonstrates that convolutional neural networks can effectively learn discriminative features for metastatic cancer detection from compact histopathology patches, and that careful model design is key for clinical decision-support systems.
