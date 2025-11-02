# Image-Classification

This project implements **image classification using a Vision Transformer (ViT) model** trained on a custom dataset. It can classify images into different food categories (Food-41 dataset) using a pre-trained and fine-tuned ViT model.

---

## Project Structure
```
D:\GenAi\Image-Classification\
│
├── vit_food41/                # Folder containing ViT model files
│   ├── vit_food41_full_model.pth
│   ├── vit_food41_weights.pth
│              
│
├── app.py                    # Folder for test images or dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Features
- Image classification using **Vision Transformer (ViT)**.
- Support for loading heavy pre-trained models.
- Predict on single images or batch of images.
- Optional GPU acceleration for faster inference.

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/ibraheem8887/Image-Classification.git
```

2. Navigate to the project folder:
```bash
cd D:\GenAi\Image-Classification\vit_food41
```

3. Create and activate a Python virtual environment (optional but recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

---




---

## Notes
- **Heavy model files** (`.pth`) are tracked using **Git LFS**.
- If you don’t want to push large files to GitHub, you can host them externally and download in your code.
- GPU is optional but recommended for faster predictions.

---

## Dependencies
- Python 3.9+
- torch
- torchvision
- transformers
- Pillow
- tqdm

Install all dependencies using:
```bash
pip install -r requirements.txt
```

