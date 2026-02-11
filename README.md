# NeurIPS 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_papers.txt

## 1. What is Flagged in Uncertainty Quantification?  Latent Density Models for Uncertainty Categorization

- [ ] What is Flagged in Uncertainty Quantification?  Latent Density Models for Uncertainty Categorization | https://neurips.cc/virtual/2023/poster/69865

- **Link**: https://neurips.cc/virtual/2023/poster/69865

<details>
<summary><strong>Abstract</strong></summary>

Uncertainty quantification (UQ) is essential for creating trustworthy machine learning models. Recent years have seen a steep rise in UQ methods that can flag suspicious examples, however, it is often unclear what exactly these methods identify. In this work, we propose a framework for categorizing uncertain examples flagged by UQ methods. We introduce the confusion density matrix---a kernel-based approximation of the misclassification density---and use this to categorize suspicious examples identified by a given uncertainty method into three classes: out-of-distribution (OOD) examples, boundary (Bnd) examples, and examples in regions of high in-distribution misclassification (IDM). Through extensive experiments, we show that our framework provides a new and distinct perspective for assessing differences between uncertainty quantification methods, thereby forming a valuable assessment benchmark.

</details>

---

## 2. Don’t blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy

- [ ] Don’t blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy | https://neurips.cc/virtual/2023/poster/69866

- **Link**: https://neurips.cc/virtual/2023/poster/69866

<details>
<summary><strong>Abstract</strong></summary>

Common explanations for shortcut learning assume that the shortcut improves prediction only under the training distribution. Thus, models trained in the typical way by minimizing log-loss using gradient descent, which we call default-ERM, should utilize the shortcut. However, even when the stable feature determines the label in the training distribution and the shortcut does not provide any additional information, like in perception tasks, default-ERM exhibits shortcut learning. Why are such solutions preferred when the loss can be driven to zero when using the stable feature alone? By studying a linear perception task, we show that default-ERM’s preference for maximizing the margin, even without overparameterization, leads to models that depend more on the shortcut than the stable feature. This insight suggests that default-ERM’s implicit inductive bias towards max-margin may be unsuitable for perception tasks. Instead, we consider inductive biases toward uniform margins. We show that uniform margins guarantee sole dependence on the perfect stable feature in the linear perception task and suggest alternative loss functions, termed margin control (MARG-CTRL), that encourage uniform-margin solutions. MARG-CTRL techniques mitigate shortcut learning on a variety of vision and language tasks, showing that changing inductive biases can remove the need for complicated shortcut-mitigating methods in perception tasks.

</details>

---
