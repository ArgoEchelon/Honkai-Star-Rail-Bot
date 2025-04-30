# Honkai Star Rail Reinforcement Learning

This repository contains the implementation of a Deep Q-Network (DQN) reinforcement learning system designed to automate gameplay in Honkai Star Rail (HSR), a turn-based RPG with complex combat mechanics.

## Project Overview

This project explores the application of reinforcement learning to automate strategic decision-making in HSR's combat system. The implementation combines computer vision for game state recognition with a DQN architecture to learn effective gameplay strategies without direct API access to the game.

### Key Features

- **Computer Vision Integration**: Uses template matching and OCR to recognize game state from screen captures
- **Deep Q-Network Implementation**: A PyTorch-based DQN with experience replay and target networks
- **Character-Specific Reward Engineering**: Detailed reward function designed around HSR's combat mechanics
- **Training and Evaluation Framework**: Comprehensive system for agent training and performance analysis

## Repository Structure

- `DQN.py` - Implementation of the Deep Q-Network agent and replay buffer
- `Ikuso.py` - Custom Gymnasium environment that interfaces with HSR through computer vision
- `xdding.py` - Training and evaluation framework
- `assets/` - Template images used for game state recognition
- `models/` - Directory for saved model checkpoints
- `requirements.txt` - Required Python dependencies

## Requirements

### Hardware Requirements

- CPU: Intel Core i7 or equivalent (quad-core recommended)
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU with CUDA support recommended for faster training
- Display: 2560x1440 or higher resolution monitor

### Software Requirements

- Python 3.8 or higher
- Honkai Star Rail game client 
- Tesseract OCR engine installed

### Python Dependencies

```
torch>=1.10.0
opencv-python>=4.5.3
gymnasium>=0.28.1
pyautogui>=0.9.53
mss>=6.1.0
pytesseract>=0.3.8
numpy>=1.21.0
matplotlib>=3.4.3
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ArgoEchelon/Honkai-Star-Rail-Bot
   cd honkai-star-rail-rl
   ```

2. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Setup

1. Launch Honkai Star Rail in Fullscreen mode at 2560x1440
2. Ensure the game window is fully visible and not obstructed by other windows
3. Configure the in-game settings:
   - Set combat speed to 2x
   - Set graphics settings to 2560x1440 

### Training

To start training the agent:

```
python Training.py --episodes 10 --render True --save_freq 1
```

Parameters:
- `--episodes`: Number of training episodes
- `--render`: Whether to render visual feedback (True/False)
- `--save_freq`: How often to save model checkpoints (in episodes)
- `--log_freq`: How often to log performance metrics (in episodes)
- `--max_steps`: Maximum steps per episode
- `--save_dir`: Directory to save models

### Evaluation

To evaluate a trained agent:

```
python Training.py --mode evaluate --model_path models/best_model.pt --eval_episodes 10
```

Parameters:
- `--mode`: Set to "evaluate" for evaluation mode
- `--model_path`: Path to the saved model checkpoint
- `--eval_episodes`: Number of evaluation episodes to run

## Acknowledgements

- This project uses PyTorch, OpenCV, and other open-source libraries
- Honkai Star Rail is a game developed by HoyoVerse