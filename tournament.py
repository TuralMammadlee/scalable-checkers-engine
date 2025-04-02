import sys
import os
import tkinter as tk
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkers import Board
from minimax import iterative_deepening
from ui import UI
from neural_network import NeuralNetworkModel, ResidualBlock
from train import train, select_move

TRAIN_MODEL = False
USE_TRAINED = False
GAME_MODE = "pva"
BOARD_SIZE = 8
AI_DIFFICULTY = "medium"
DEBUG_MODE = True
REPLAY_BUFFER_FILE = "replay_buffer_8x8.pkl"


def engine(board, color, model):
    
    piece_count = board.black_left + board.white_left
    
    #  search depths
    if piece_count > 16:   # early
        depth = 3
    elif piece_count > 8:  # mid
        depth = 4
    else:                  # late
        depth = 5
    
    # Reduced time limit to 3 seconds
    best = iterative_deepening(board, max_depth=depth, color=color, model=model, time_limit=3.0)
    if best:
        return (best["piece"], best["move"], best["skipped"])
    return None

# ---------------------------------------------------------------------
#  opponents:
#    a) Minimax with shallow depth
#    b) Random
#    c) Hybrid with shallow depth
#    d) Aggressive
#    e) Defensive
# ---------------------------------------------------------------------
def minimax_move_fn(board, color, model=None, depth=2):
    
    piece_count = board.black_left + board.white_left
    
    # Slightly lowered baseline depth
    if piece_count > 16:   # early game
        d = depth
    elif piece_count > 8:  # mid game
        d = depth + 1
    else:                  # end game
        d = depth + 2
    
    # Slightly reduced time limit
    time_limit = 4.0 if d <= 4 else 5.0
    best = iterative_deepening(board, max_depth=d, color=color, model=model, time_limit=time_limit)
    if best:
        return (best.get("piece"), best.get("move"), best.get("skipped"))
    return None

def random_move_fn(board, color, model=None):
    valid_moves = []
    for piece in board.get_all_pieces(color):
        piece_valid_moves = board.get_valid_moves(piece)
        for move, skipped in piece_valid_moves.items():
            valid_moves.append((piece, move, skipped))
    if not valid_moves:
        return None
    return random.choice(valid_moves)

def aggressive_move_fn(board, color, model=None):
    valid_moves = []
    capture_moves = []
    forward_moves = []
    for piece in board.get_all_pieces(color):
        piece_valid_moves = board.get_valid_moves(piece)
        for move, skipped in piece_valid_moves.items():
            if skipped:
                capture_moves.append((piece, move, skipped))
            elif (color == "white" and move[0] > piece.row) or (color == "black" and move[0] < piece.row):
                forward_moves.append((piece, move, skipped))
            else:
                valid_moves.append((piece, move, skipped))
    if capture_moves:
        return random.choice(capture_moves)  
    elif forward_moves:
        return random.choice(forward_moves)
    elif valid_moves:
        return random.choice(valid_moves)
    return None

def defensive_move_fn(board, color, model=None):
    valid_moves = []
    safe_moves = []
    for piece in board.get_all_pieces(color):
        piece_valid_moves = board.get_valid_moves(piece)
        for move, skipped in piece_valid_moves.items():
            if skipped:
                # Always consider captures
                valid_moves.append((piece, move, skipped))
                continue
            # Check if the resulting position is safe from immediate capture
            safe = True
            temp_board = board.copy()
            temp_piece = None
            for p in temp_board.get_all_pieces(color):
                if p.row == piece.row and p.col == piece.col:
                    temp_piece = p
                    break
            if temp_piece:
                temp_board.move(temp_piece, move[0], move[1])
                opp_color = "black" if color == "white" else "white"
                for opp_piece in temp_board.get_all_pieces(opp_color):
                    opp_moves = temp_board.get_valid_moves(opp_piece)
                    for _, skipped_after in opp_moves.items():
                        if skipped_after and any(p.row == move[0] and p.col == move[1] for p in skipped_after):
                            safe = False
                            break
                    if not safe:
                        break
            if safe:
                safe_moves.append((piece, move, skipped))
            else:
                valid_moves.append((piece, move, skipped))
    if safe_moves:
        return random.choice(safe_moves)
    elif valid_moves:
        return random.choice(valid_moves)
    return None

def hybrid_move_fn(board, color, model, depth=2):
    """Stronger hybrid that combines the neural network with deeper minimax."""
    piece_count = board.black_left + board.white_left
    
    if piece_count > 16:
        d = depth
    elif piece_count > 8:
        d = depth + 1
    else:
        d = depth + 2
    
    time_limit = 4.0 if d <= 4 else 6.0
    best = iterative_deepening(board, max_depth=d, color=color, model=model, time_limit=time_limit)
    if best:
        return (best["piece"], best["move"], best["skipped"])
    return None


# ---------------------------------------------------------------------
# 3) Tournament logic:  AI  vs  opponents
# ---------------------------------------------------------------------
def simulate_game_with_stats(move_fn_white, move_fn_black, board_size=8, max_moves=150):
    board = Board(rows=board_size, cols=board_size)
    current_player = "white"
    move_count = 0
    white_captures = 0
    black_captures = 0
    no_progress_count = 0
    max_no_progress = 30
    timeout = False
    start_time = time.time()
    max_game_time = 60  # e.g. it is an instance for testing 
    
    while move_count < max_moves:
        if time.time() - start_time > max_game_time:
            timeout = True
            break
        if board.winner() is not None:
            break
        move_fn = move_fn_white if current_player == "white" else move_fn_black
        move = move_fn(board, current_player)
        if move is None:
            # no valid moves => break
            break
        piece, move_coords, skipped = move
        board.move(piece, move_coords[0], move_coords[1])
        if skipped:
            board.remove(skipped)
            if current_player == "white":
                white_captures += len(skipped)
            else:
                black_captures += len(skipped)
            no_progress_count = 0
        else:
            no_progress_count += 1
        
        current_player = "black" if current_player == "white" else "white"
        move_count += 1
        if no_progress_count >= max_no_progress:
            break
    
    winner = board.winner()
    if winner is None:
        # Always declare a winner based on piece count
        if board.white_left > board.black_left:
            winner = "white"
        elif board.black_left > board.white_left:
            winner = "black"
        else:
            # If equal pieces, use king count as tiebreaker
            if board.white_kings > board.black_kings:
                winner = "white"
            elif board.black_kings > board.white_kings:
                winner = "black"
            else:
                # If still tied, give advantage to the player who made fewer moves
                winner = "black" if move_count % 2 == 0 else "white"
    
    return {
        "winner": winner,
        "moves": move_count,
        "white_captures": white_captures,
        "black_captures": black_captures,
        "white_kings": board.white_kings,
        "black_kings": board.black_kings,
        "final_white_pieces": board.white_left,
        "final_black_pieces": board.black_left,
        "timeout": timeout
    }

def create_resized_model(original_model, new_size):
    """Create a new model with the same architecture but resized for a different board size.
    The network uses adaptive pooling to maintain fixed dimensions for FC layers."""
    resized_model = NeuralNetworkModel(board_rows=new_size, board_cols=new_size)
    
    try:
        state_dict = original_model.model.state_dict()
        new_state_dict = {}
        
        # The network architecture is board-size independent
        for key, value in state_dict.items():
            if 'conv' in key:
                if len(value.shape) == 4:  # Conv2d weights
                    #  re-initialize
                    new_state_dict[key] = torch.nn.Parameter(
                        torch.zeros(value.shape)
                    )
                    torch.nn.init.xavier_uniform_(new_state_dict[key])
                elif len(value.shape) == 1:  # Bias
                    new_state_dict[key] = torch.nn.Parameter(torch.zeros(value.shape))
            elif 'bn' in key:
                if len(value.shape) == 1:
                    new_state_dict[key] = torch.nn.Parameter(torch.zeros(value.shape))
            elif 'fc' in key:
                if len(value.shape) == 2:
                    new_state_dict[key] = torch.nn.Parameter(
                        torch.zeros(value.shape)
                    )
                    torch.nn.init.xavier_uniform_(new_state_dict[key])
                elif len(value.shape) == 1:
                    new_state_dict[key] = torch.nn.Parameter(torch.zeros(value.shape))
            else:
                new_state_dict[key] = value
        
        missing_keys, unexpected_keys = resized_model.model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully created new model for {new_size}x{new_size} board")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
    except Exception as e:
        print(f"Warning: Could not resize model for {new_size}x{new_size} board: {e}")
        print("Using randomly initialized weights instead")
    
    return resized_model

def tournament_mode():
    # Load  main model
    main_model = NeuralNetworkModel(board_rows=BOARD_SIZE, board_cols=BOARD_SIZE)
    if USE_TRAINED and os.path.exists("model.pth"):
        main_model.load_model("model.pth")
        print("\n=== Tournament Starting ===")
        print("✓ Loaded trained model from model.pth")
    else:
        print("\n WARNING: Using random weights (not trained)! Expect weaker or random play.\n")
    
    # Define opponents 
    opponents = {
        "Minimax": lambda board, color: minimax_move_fn(board, color, None, depth=3),
        "Random": lambda board, color: random_move_fn(board, color),
        "Hybrid": lambda board, color: hybrid_move_fn(board, color, main_model, depth=3),
        "Aggressive": lambda board, color: aggressive_move_fn(board, color),
        "Defensive": lambda board, color: defensive_move_fn(board, color),
    }
    
    # Test different board sizes
    board_sizes = [6, 8, 10, 12, 14, 16, 18]
    
    print("\n=== Configuration ===")
    print(f"• Number of board sizes to test: {len(board_sizes)}")
    print(f"• Board sizes: {board_sizes}")
    print(f"• Number of opponents: {len(opponents)}")
    print(f"• Games per matchup: 10")
    print("• Using iterative deepening minimax with neural network evaluation\n")
    
    # Store results for each board size
    size_results = {}
    for size in board_sizes:
        print(f"\n Testing on {size}x{size} board:")
        print("=" * 40)
        size_results[size] = {
            "Win Rate (%)": {},
            "Draw Rate (%)": {},
            "Avg Game Length": {},
            "Capture Ratio": {},
            "King Ratio": {},
            "Decisiveness": {}
        }
        
        # Create a resized model for this board size
        print(f"• Resizing model for {size}x{size} board...")
        resized_model = create_resized_model(main_model, size)
    
    for opp_name, opp_move_fn in opponents.items():
            print(f"\n Testing against: {opp_name}")
            print("-" * 30)
            
            wins = 0
            losses = 0
            draws = 0
            total_moves = 0
            own_captures = 0
            opp_captures = 0
            own_kings = 0
            opp_kings = 0
            decisiveness_scores = []
            timeouts = 0
            
            for game_i in range(10):  # increased from 5 to 10
                print(f"  Game {game_i + 1}/10: ", end="", flush=True)
                
                if game_i % 2 == 0:
                    game_result = simulate_game_with_stats(
                        lambda b, c: engine(b, c, resized_model),  # Using  actual engine
                    opp_move_fn,
                        board_size=size,
                        max_moves=150
                )
                    
                    if game_result["winner"] == "white":
                        wins += 1
                        print("Won ✓")
                    elif game_result["winner"] == "black":
                        losses += 1
                        print("Lost ✗")
                    else:
                        draws += 1
                        print("Draw =")
                    
                    total_moves += game_result["moves"]
                    own_captures += game_result["white_captures"]
                    opp_captures += game_result["black_captures"]
                    own_kings += game_result["white_kings"]
                    opp_kings += game_result["black_kings"]
                    piece_diff = game_result["final_white_pieces"] - game_result["final_black_pieces"]
                    decisiveness_scores.append(piece_diff)
                    
            else:
                    game_result = simulate_game_with_stats(
                    opp_move_fn,
                        lambda b, c: engine(b, c, resized_model),  # Using  actual engine
                        board_size=size,
                        max_moves=150
                )
                    
                    if game_result["winner"] == "black":
                     wins += 1
                     print("Won ✓")
                    elif game_result["winner"] == "white":
                        losses += 1
                        print("Lost ✗")
                    else:
                        draws += 1
                        print("Draw =")
                    
                    total_moves += game_result["moves"]
                    own_captures += game_result["black_captures"]
                    opp_captures += game_result["white_captures"]
                    own_kings += game_result["black_kings"]
                    opp_kings += game_result["white_kings"]
                    piece_diff = game_result["final_black_pieces"] - game_result["final_white_pieces"]
                    decisiveness_scores.append(piece_diff)
                
                    if game_result.get("timeout", False):
                        timeouts += 1
            
            total_games = wins + losses + draws
            if total_games == 0:
                total_games = 1
            
            win_rate = (wins / total_games) * 100
            draw_rate = (draws / total_games) * 100
            avg_moves = total_moves / total_games
            capture_ratio = own_captures / max(1, opp_captures)
            king_ratio = own_kings / max(1, opp_kings)
            avg_decisiveness = sum(decisiveness_scores) / len(decisiveness_scores)
            
            size_results[size]["Win Rate (%)"][opp_name] = win_rate
            size_results[size]["Draw Rate (%)"][opp_name] = draw_rate
            size_results[size]["Avg Game Length"][opp_name] = avg_moves
            size_results[size]["Capture Ratio"][opp_name] = capture_ratio
            size_results[size]["King Ratio"][opp_name] = king_ratio
            size_results[size]["Decisiveness"][opp_name] = avg_decisiveness
            
            # Print detailed results for this opponent
            print(f"\n  Results vs {opp_name}:")
            print(f"  • Wins: {wins}/{total_games} ({win_rate:.1f}%)")
            print(f"  • Draws: {draws}/{total_games} ({draw_rate:.1f}%)")
            print(f"  • Average game length: {avg_moves:.1f} moves")
            print(f"  • Capture ratio: {capture_ratio:.2f}")
            print(f"  • King ratio: {king_ratio:.2f}")
            if timeouts > 0:
                print(f"  • Timeouts: {timeouts}")
            print("-" * 30)
    
    # Plot results with heatmap
    plot_tournament_results_with_heatmap(size_results)

def plot_tournament_results_with_heatmap(size_results):
    """Plot tournament results with a heatmap for win rates across board sizes"""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Tournament Results: Your AI vs Various Opponents", fontsize=14)
    
    # 1. Win Rate Heatmap
    ax1 = fig.add_subplot(221)
    board_sizes = sorted(size_results.keys())
    opponents = list(size_results[board_sizes[0]]["Win Rate (%)"].keys())
    
    heatmap_data = np.zeros((len(opponents), len(board_sizes)))
    for i, opp in enumerate(opponents):
        for j, size in enumerate(board_sizes):
            heatmap_data[i, j] = size_results[size]["Win Rate (%)"][opp]
    
    im = ax1.imshow(heatmap_data, aspect='auto', cmap='RdYlGn')
    ax1.set_xticks(np.arange(len(board_sizes)))
    ax1.set_yticks(np.arange(len(opponents)))
    ax1.set_xticklabels([f"{size}x{size}" for size in board_sizes])
    ax1.set_yticklabels(opponents)
    plt.colorbar(im, ax=ax1, label='Win Rate (%)')
    ax1.set_title("Win Rate Heatmap Across Board Sizes")
    
    # Add text annotations
    for i in range(len(opponents)):
        for j in range(len(board_sizes)):
            ax1.text(j, i, f"{heatmap_data[i, j]:.1f}%", ha="center", va="center", color="black")
    
    # 2. Average Win Rate Bar Chart
    ax2 = fig.add_subplot(222)
    avg_win_rates = [np.mean([size_results[size]["Win Rate (%)"][opp] for size in board_sizes]) for opp in opponents]
    y_pos = np.arange(len(opponents))
    colors = ['#5cb85c' if r >= 50 else '#d9534f' for r in avg_win_rates]
    bars = ax2.barh(y_pos, avg_win_rates, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(opponents)
    ax2.invert_yaxis()
    ax2.set_xlabel("Average Win Rate (%)")
    ax2.set_title("Average Win Rate Across All Board Sizes")
    ax2.axvline(x=50, color='gray', linestyle='--')
    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax2.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center')
    
    # 3. Average Game Length
    ax3 = fig.add_subplot(223)
    avg_moves = [size_results[8]["Avg Game Length"][opp] for opp in opponents]
    bars = ax3.bar(opponents, avg_moves, color='lightblue')
    ax3.set_title("Average Game Length (moves)")
    ax3.set_xticklabels(opponents, rotation=45, ha='right')
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h+0.3, f"{h:.1f}", ha='center')
    
    # 4. Efficiency Ratios
    ax4 = fig.add_subplot(224)
    capture_ratios = [size_results[8]["Capture Ratio"][opp] for opp in opponents]
    king_ratios = [size_results[8]["King Ratio"][opp] for opp in opponents]
    x = np.arange(len(opponents))
    width = 0.35
    bars1 = ax4.bar(x - width/2, capture_ratios, width, label='Capture Ratio', color='salmon')
    bars2 = ax4.bar(x + width/2, king_ratios, width, label='King Ratio', color='plum')
    ax4.set_title("Efficiency Ratios (higher is better)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(opponents, rotation=45, ha='right')
    ax4.axhline(y=1.0, color='gray', linestyle='--')
    ax4.legend()
    for bar in bars1:
        val = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., val+0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        val = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., val+0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('tournament_results.png', dpi=300)
    plt.show()

# ---------------------- GUI / main code for demonstration ---------------------- #

def show_startup_dialog(root):
    pass

def main():
    global USE_TRAINED
    USE_TRAINED = True  # set to True if  have a model.pth
    tournament_mode()

if __name__ == "__main__":
    main()

class CheckersCNN(nn.Module):
    def __init__(self, board_rows, board_cols, num_residual_blocks=3):
        """
    
        - Both policy and value heads use adaptive pooling to collapse spatial dimensions
        - Policy head: 64 -> 32 channels -> adaptive pool -> FC layers
        - Value head: 64 -> 32 channels -> adaptive pool -> FC layers
        """
        super(CheckersCNN, self).__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        
        # Input: 9 channels (board state)
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(num_residual_blocks)])
        
        # Policy head: 64 -> 32 channels -> adaptive pool -> FC layers
        self.conv_policy = nn.Conv2d(64, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.adaptive_pool_policy = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_policy1 = nn.Linear(32, 128)
        self.fc_policy2 = nn.Linear(128, board_rows * board_cols)  # One output per board position
        
        # Value head: 64 -> 32 channels -> adaptive pool -> FC layers
        self.conv_value = nn.Conv2d(64, 32, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(32)
        self.adaptive_pool_value = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_value1 = nn.Linear(32, 128)
        self.fc_value2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Common trunk
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy branch
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = self.adaptive_pool_policy(policy)  # shape: (batch, 32, 1, 1)
        policy = policy.view(policy.size(0), -1)    # shape: (batch, 32)
        policy = self.dropout(policy)
        policy = F.relu(self.fc_policy1(policy))
        policy = self.fc_policy2(policy)            # shape: (batch, board_rows * board_cols)
        
        # Value branch
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = self.adaptive_pool_value(value)     # shape: (batch, 32, 1, 1)
        value = value.view(value.size(0), -1)       # shape: (batch, 32)
        value = self.dropout(value)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))   # shape: (batch, 1)
        
        return value, policy
