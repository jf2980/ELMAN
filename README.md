# ELMAN
📂 Project StructureELMAN/
│── configs/ # Training configuration files (e.g., epochs, dataset paths, layers)
│── datas/ # Data loading scripts
│── def_loss/ # Loss function implementations
│── experiment/ # Saved model checkpoints and logs
│── models/ # Model architecture definitions
│── utils/ # Utility functions (metrics, saving, image processing, etc.)
│── dataset/
├── Benchmark/ # Validation & test datasets (Set5, Set14, BSD100, Urban100, Manga109...)
├── DIV2K/ # Training data
├── HR/ # High-resolution images
├── LR_bicubic/ # Low-resolution inputs

⚙️ Environment
GPU: RTX 4090
Python: 3.8
PyTorch stack:
torch == 2.0.0
torchaudio == 2.0.0
torchvision == 0.15.0
🧩 Model Architecture
The proposed ELMAN consists of:

Head:

3×3 CNN for shallow feature extraction.
Body (main module):

Shift-Conv: Local receptive field enhancement.
GMMSA: Global Multi-Scale Self-Attention.
Includes REB (Residual Enhanced Block) with deformable convolutions.
Tail:

PixelShuffle for upsampling.
Pipeline:

🚀 Training
Prepare your dataset in the following format:
dataset/
├── Benchmark/ # test/validation sets
├── DIV2K/
├── HR/
├── LR_bicubic/
Run training:
Ex:
python train.py --config ./configs/X3.yaml
python demo.py
--data_dir ./dataset/Benchmark/Set5
--weights ./experiment/model_x3.pth
--output ./results/set5_x3
