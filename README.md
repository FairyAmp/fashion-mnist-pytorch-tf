# Fashion-MNIST: PyTorch & TensorFlow Implementations

An exploration and implementation of image classification on the **Fashion-MNIST** dataset using both **PyTorch** and **TensorFlow / Keras**. 

This repository demonstrates model construction, training, and performance evaluation across PyTorch and TensorFlow frameworks.

---

## 📌 Project Overview

The project itself was originally part of a project done for ECS 170 at UC Davis. I ended up coding it in both:
* **PyTorch**
* **TensorFlow**

Consequently, the goal of this project evolved into a personal project to compare model implementation workflows and performance on the Fashion-MNIST benchmark dataset using the two distinct frameworks.

### About the Dataset
Fashion-MNIST consists of 70,000 grayscale images (28x28 pixels) spanning 10 distinct clothing categories (60,000 training images and 10,000 test images).

**Categories:**
`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`

---

## 📌 Maintenance Note

> **Note:** This codebase was originally written ~3 years ago. While the core deep learning architecture and logic remain sound, some dependencies or API calls (e.g., legacy `torch` export interfaces or earlier `tf.keras` syntax) may trigger deprecation warnings in newer framework releases.
>
> There may also be some bugs in the current TensorFlow version.
> **Planned Updates / Roadmap:**
> - [ ] Refactor PyTorch training loop to use current `torch.compile` and modern `torch.export` standards.
> - [ ] Update TensorFlow dependencies and test compatibility against latest Keras versions.
> - [ ] Check for and fix any possible bugs in the TensorFlow Implementation
> - [ ] Add `requirements.txt` / `environment.yml` with pinned legacy versions for reproducible environments.

## 🛠️ Project Structure

```text
.
├── pytorch/         # PyTorch dataset loading, model definitions, & training scripts
├── tensorflow/      # TensorFlow/Keras scripts & evaluation
└── README.md
