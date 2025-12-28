🕵️ SHERLOCK: Structured Hierarchical Efficient Resolution-aware Learning with Optimized Class Knowledge
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-red.svg
https://img.shields.io/badge/arXiv-Paper-b31b1b.svg

Official implementation of SHERLOCK, a comprehensive framework for efficient dermoscopic skin lesion classification that achieves clinical-grade accuracy with mobile-friendly computational efficiency. SHERLOCK addresses three critical deployment challenges through:

RAAT: Resolution-Aware Attention Transfer (96.6% FLOPs reduction)

DHL: Dermoscopic Hierarchical Learning (+8.2% melanoma recall)

Integrated System: 90.1% accuracy at only 0.28 GFLOPs

📊 Key Results
Metric  SHERLOCK  Baseline  Improvement
Accuracy  90.1%  84.2%  +5.9 pp
Macro-F1  85.8%  78.4%  +7.4 pp
Melanoma Recall  91.8%  81.3%  +10.5 pp
FLOPs  0.28 G  8.35 G  -96.7%
Mobile Latency  89.4 ms  387.6 ms  -76.9%
🚀 Quick Start
Installation
bash
# Clone repository
git clone https://github.com/your-org/SHERLOCK.git
cd SHERLOCK

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Requirements
Python 3.8+

PyTorch 2.0+

torchvision

scikit-learn

matplotlib

opencv-python

albumentations

Data Preparation
Download HAM10000 dataset from ISC2017 Challenge

Organize the data structure:

text
data/
├── HAM10000/
│   ├── images/
│   │   ├── ISIC_0024306.jpg
│   │   └── ...
│   └── HAM10000_metadata.csv
Preprocess and split data:

bash
python scripts/preprocess.py --data_dir data/HAM10000 --output_dir data/processed
🏗 Model Architecture
SHERLOCK consists of three integrated components:

1. Multi-Model Baseline
Systematic evaluation of 6 architectures (MobileNetV3, EfficientNet, Xception, ResNet50, VGG16/19) to establish performance-efficiency Pareto frontiers.

2. Resolution-Aware Attention Transfer (RAAT)
python
from models.raat import RAAT

# Initialize RAAT with teacher and student models
teacher = XceptionTeacher(resolution=224)
student = EfficientNetStudent(resolution=160)

raat = RAAT(teacher=teacher, student=student, beta=0.3)
raat.train(train_loader, val_loader, epochs=25)
3. Dermoscopic Hierarchical Learning (DHL)
python
from models.dhl import DHL

# Initialize DHL with shared backbone and dual heads
model = DHL(backbone='efficientnet-b3', lambda_param=0.5)
model.train(train_loader, val_loader, epochs=20)
📁 Repository Structure
text
SHERLOCK/
├── models/
│   ├── baseline.py      # Multi-model baseline implementations
│   ├── raat.py          # Resolution-Aware Attention Transfer
│   └── dhl.py           # Dermoscopic Hierarchical Learning
├── scripts/
│   ├── preprocess.py    # Data preprocessing and augmentation
│   ├── train.py         # Training scripts
│   └── evaluate.py      # Evaluation and metrics
├── configs/
│   ├── baseline.yaml    # Baseline model configurations
│   ├── raat.yaml        # RAAT training configurations
│   └── dhl.yaml         # DHL training configurations
├── data/
│   └── processed/       # Preprocessed datasets
├── notebooks/
│   └── analysis.ipynb   # Results analysis and visualization
├── requirements.txt
├── train_baseline.py
├── train_raat.py
├── train_dhl.py
└── README.md
🎯 Training
Baseline Models
bash
python train_baseline.py \
  --model efficientnet-b3 \
  --data_dir data/processed \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4
RAAT Training
bash
python train_raat.py \
  --teacher xception \
  --student efficientnet-b0 \
  --teacher_res 224 \
  --student_res 160 \
  --beta 0.3 \
  --data_dir data/processed \
  --epochs 25
DHL Training
bash
python train_dhl.py \
  --backbone efficientnet-b3 \
  --lambda 0.5 \
  --data_dir data/processed \
  --epochs 20
📈 Evaluation
bash
# Evaluate baseline model
python scripts/evaluate.py \
  --model_path checkpoints/baseline_efficientnet-b3.pth \
  --data_dir data/processed/test
