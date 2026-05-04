```yaml
tags: [explainable-ai, shap, deep-learning, image-classification, bias-detection, python, production-ml]
```

# Uncovering Model Bias with SHAP: A Case Study on Image Classification

![Explainable AI for Deep Learning Models](../images/explainable-ai-for-deep-learning-models.jpg)

---

## TL;DR

- **SHAP values can uncover model bias and hidden decision patterns in deep learning image classifiers.**
- **We’ll walk through a real-world pipeline, with code, for pixel-level attribution and model bias auditing.**
- **Includes architectural diagrams, production best practices, and actionable lessons from implementing explainability at scale.**

---

## Introduction: Why Explainability for Image Models Matters—Now

Deep learning image classifiers are everywhere—from medical diagnostics to retail surveillance. But as these models grow in complexity, so does the challenge of trusting their predictions, especially when it comes to **bias and fairness**. Black-box models can encode social, demographic, or data-driven biases—sometimes with serious consequences.

**Explainable AI (XAI)** is no longer a research toy; it’s a **production necessity**. Regulators demand audit trails. Users (and teams) need to know *why* the model did what it did. Among the arsenal of XAI tools, **SHAP (SHapley Additive exPlanations)** stands out for its strong theoretical foundation and flexible tooling for deep learning.

This article is a hands-on, code-driven case study: **How can you use SHAP to uncover and mitigate model bias in image classifiers?** I’ll show you, based on what’s worked (and what’s failed) in real production deployments.

---

## Technical Deep Dive: SHAP on Image Classifiers

### 1. Theoretical Primer

**SHAP** computes Shapley values—concepts borrowed from cooperative game theory—to fairly attribute the “payout” (the model’s prediction) to each feature (pixels, superpixels, or regions). For images:
- **Pixel-level attribution**: Which pixels most increased or decreased the probability for a given class?
- **Superpixel attribution**: Grouped pixels for more robust, less noisy explanations.

The **DeepExplainer** module in SHAP leverages connections with DeepLIFT and integrates seamlessly with both PyTorch and TensorFlow.

---

### 2. Setup: A Case Study with CIFAR-10 and ResNet

Let’s get practical. Assume you have a ResNet (e.g., ResNet18) trained on CIFAR-10. We’ll use SHAP to probe for bias in “cat vs. dog” images.

**Environment prerequisites:**
```bash
pip install torch torchvision shap matplotlib
```

**Import libraries & load model:**

```python
import torch
import torchvision
import shap
import matplotlib.pyplot as plt

# Load pretrained ResNet18 for CIFAR-10 (use torch.hub for demo)
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Transform for CIFAR-10 images
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])

# Sample images
from torchvision.datasets import CIFAR10
dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
dog_images = [img for img, label in dataset if label == 5][:5]  # Class 5: Dog
cat_images = [img for img, label in dataset if label == 3][:5]  # Class 3: Cat
```

---

### 3. Computing SHAP Values

DeepExplainer requires a *background dataset*, typically a small random sample of the images (e.g., 100), to estimate “missingness.” Here’s how you compute and visualize SHAP values for a single image:

```python
import numpy as np

background = torch.stack([img for img, _ in [dataset[i] for i in range(100)]])
test_images = torch.stack([dog_images[0], cat_images[0]])  # Example

# DeepExplainer expects model to output logits (pre-softmax)
explainer = shap.DeepExplainer(model, background)

# Pass test images through the explainer
shap_values = explainer.shap_values(test_images)

# Visualize the SHAP attributions
shap.image_plot(shap_values, test_images.numpy())
plt.show()
```

**What this yields:**  
A side-by-side visualization: the original image, and an overlay heatmap showing per-pixel importance (red=increasing probability, blue=decreasing).

---

### 4. Interpreting SHAP Visualizations for Bias

Suppose your SHAP overlays consistently highlight *background pixels* (e.g., a specific colored carpet in “dog” images). That’s a **bias signal**: the model is relying on contextual clues, not just the animal. This is how spurious correlations sneak in.

**In production, I have seen:**
- Skin lesion classifiers using ruler marks as “malignant” signals, not actual lesion features.
- Retail classifiers focusing on shelf tags, not products themselves.

**Your SHAP heatmap should primarily light up the object of interest.** If it doesn’t, you have a bias or shortcut in your dataset/model.

---

## Architecture Pattern: Batch SHAP Auditing Pipeline

A typical production setup for XAI-driven model audits, which I’ve implemented for large-scale image pipelines, looks like this:

```
[Input Images]
      |
      v
[Model Inference (TorchServe/TF Serving)]
      |
      v
[SHAP Value Computation (Batch Worker Pool)]
      |
      v
[Visualization Storage (S3, GCS)]
      |
      v
[Bias Detection Dashboard (Grafana, Streamlit, Custom UI)]
```

- **Batch processing**: Run SHAP explanations nightly on fresh data slices (per class, per region, per demographic).
- **Dashboards/Alerting**: Data scientists review heatmaps; bias triggers raise tickets for re-training or data collection.
- **Storage**: Store SHAP overlays alongside prediction logs for compliance.

For **on-demand explanations** (e.g., for clinicians):
- Expose a REST API endpoint (FastAPI/Flask) that runs inference + SHAP for a single image, and returns the overlay as a PNG/JSON.

---

## Production Lessons Learned

**From real, often painful, deployments:**

- **SHAP for images is compute-heavy.** On large CNNs, explanations may take 10–30 seconds per image even with CUDA. Batch processing and GPU autoscaling are essential.
- **Background dataset selection is critical.** A non-representative background can yield misleading attributions. Use a random sample from the full inference set, not just the train set.
- **Superpixels > raw pixels for end users.** Raw pixel attributions are noisy. For user-facing explanations, aggregate SHAP to superpixels (e.g., via SLIC segmentation) for clearer overlays.
- **Bias diagnosis is iterative.** The first SHAP heatmap hints at issues; the fix may require multiple cycles—data cleaning, augmentation, retraining.
- **Visualization is not enough.** Automate detection: e.g., compute the fraction of attribution mass outside the object bounding box as a metric for “spurious focus.”
- **Compliance Integration:** Store SHAP overlays and logs with predictions—regulators increasingly ask for “reason codes” even for image models.

---

## Key Takeaways

- **SHAP brings transparency** to deep image models—if used thoughtfully.
- **Pixel-level and superpixel attributions** quickly surface bias and shortcut learning in production pipelines.
- **Auditing explainability must be automated** and made part of your MLOps stack, not a one-off Jupyter Notebook task.
- **Design for scale**: GPU batch workers, summary dashboards, and alerting on bias signals are essential for real-world deployments.

---

## Further Reading & Resources

- **Official SHAP documentation:** [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
- **SHAP DeepExplainer for PyTorch:** [https://github.com/slundberg/shap#deep-explainer](https://github.com/slundberg/shap#deep-explainer)
- **“A Unified Approach to Interpreting Model Predictions” (SHAP paper):** [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
- **Case Study – Bias in Skin Lesion Classification:** [Goyal et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32443148/)
- **Superpixel segmentation for XAI:** [scikit-image SLIC docs](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#sphx-glr-auto-examples-segmentation-plot-segmentations-py)

---

*By Sheikh Muhammad Qasim | ML Architect*