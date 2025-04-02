# scalable-checkers-engine
A scalable, adaptive checkers engine that extends the traditional 8×8 board to support different board sizes. This project combines classical game-playing algorithms such as minimax with alpha–beta pruning and iterative deepening, together with modern machine learning techniques using neural networks. The hybrid approach enables enhanced evaluation and strategic move selection on boards of various dimensions.

Overview
This project implements a checkers game engine designed to address the limitations of traditional 8×8 checkers by allowing flexible board sizes. The engine leverages:

Classical AI Techniques: An enhanced minimax algorithm with alpha–beta pruning and iterative deepening.

Neural Network Evaluation: A convolutional neural network (CNN) model with residual blocks to provide adaptive board evaluation.

Training and Self-play: A training pipeline that uses self-play and replay buffers to refine the neural network’s evaluation function.

Tournament Mode: Multiple opponent strategies and tournament logic to assess performance across different board sizes.

The resulting engine demonstrates improved scalability, deeper search capabilities, and robust move selection by integrating handcrafted heuristics with modern ML-based evaluation.

Features
Scalable Board Support: Play checkers on boards of varying sizes (e.g., 6×6, 8×8, 10×10, etc.).

Adaptive Evaluation Function: Combines classical heuristics and neural network predictions using adaptive weighting.

Efficient Search: Implements iterative deepening minimax with alpha–beta pruning, along with quiescence search to handle tactical positions.

Neural Network Architecture: A custom CNN with multiple residual blocks designed for board state evaluation.

Training Pipeline: A self-play training module with data augmentation and replay buffer management.

Graphical User Interface: A simple Tkinter-based UI for interactive play.

Tournament Mode: Automated tournaments against various opponent strategies (minimax, random, aggressive, defensive, hybrid).

Installation
Clone the Repository:
git clone https://github.com/yourusername/scalable-checkers-engine.git
cd scalable-checkers-engine
Create a Virtual Environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install Required Packages:
pip install -r requirements.txt
Note: Ensure you have PyTorch installed with CUDA support if available, as well as other packages like matplotlib, numpy, and tkinter (usually included with Python).

File Structure
main.py – The entry point for running the game with UI dialogs.

checkers.py – Core implementation of the checkers board, pieces, and valid moves.

minimax.py – Implementation of the iterative deepening minimax search with transposition table and quiescence search.

neural_network.py – Defines the CNN-based model used for board evaluation.

train.py – Training module for the neural network, including replay buffer management and data augmentation.

tournament.py – Logic for automated tournaments against different opponent strategies.

ui.py – Graphical user interface implementation using Tkinter.

analysis.py – Utility scripts for performance and feature analysis.

evaluation.py – Evaluation functions that combine heuristic and neural network predictions.

Usage
Running the Game
To launch the game with the interactive UI:
python main.py
You will be prompted with a startup dialog where you can set game mode (Player vs AI or AI vs AI), board size, AI difficulty, and training options.

Training the Neural Network
If you wish to train the evaluation model via self-play, set the appropriate option in the startup dialog or run:
python train.py
Training parameters (number of epochs, games per epoch, learning rate, etc.) can be adjusted within the code or via provided UI dialogs.

Running Tournament Mode
To run an automated tournament testing different board sizes and opponent strategies, execute:
python tournament.py
Results will be displayed in the terminal and optionally plotted using matplotlib.
