#!/usr/bin/env python3
import random
import math
import copy
import matplotlib.pyplot as plt

from checkers import Board
from evaluation import get_adaptive_weights, count_moves

# Increase default font size for clarity
plt.rcParams.update({'font.size': 12})

def simulate_random_game(board_size, max_moves=150):
    """
    Simulates a random game on a board of the given size for up to max_moves.
    Returns the final board state and a history of moves.
    Each move record classifies the move type (capture, non-capture, promotion, central, defensive).
    """
    board = Board(rows=board_size, cols=board_size)
    move_history = []
    current_player = 'white'
    for _ in range(max_moves):
        valid_moves = []
        pieces = board.get_all_pieces(current_player)
        for piece in pieces:
            moves = board.get_valid_moves(piece)
            for dest, skipped in moves.items():
                valid_moves.append((piece, dest, skipped))
        if not valid_moves:
            break

        piece, dest, skipped = random.choice(valid_moves)
        start_pos = (piece.row, piece.col)
        is_capture = bool(skipped)
        is_promotion = False
        if not piece.king:
            if current_player == 'white' and dest[0] == board.rows - 1:
                is_promotion = True
            elif current_player == 'black' and dest[0] == 0:
                is_promotion = True

        # Define central region as the middle 50% of rows and columns.
        central_row_min = int(board.rows * 0.25)
        central_row_max = int(board.rows * 0.75)
        central_col_min = int(board.cols * 0.25)
        central_col_max = int(board.cols * 0.75)
        is_central = (central_row_min <= dest[0] < central_row_max) and (central_col_min <= dest[1] < central_col_max)

        # A defensive move retreats relative to the forward direction.
        is_defensive = False
        if current_player == 'white' and dest[0] < start_pos[0]:
            is_defensive = True
        elif current_player == 'black' and dest[0] > start_pos[0]:
            is_defensive = True

        move_record = {
            "player": current_player,
            "start": start_pos,
            "dest": dest,
            "skipped": skipped,
            "capture": is_capture,
            "promotion": is_promotion,
            "central": is_central,
            "defensive": is_defensive
        }
        move_history.append(move_record)

        board.move(piece, dest[0], dest[1])
        if skipped:
            board.remove(skipped)
        current_player = 'black' if current_player == 'white' else 'white'

    return board, move_history

def frequency_distribution_analysis(board_size, num_games=100):
    """
    Runs multiple simulated games on a board of a given size and records
    how often each move type occurs.
    Returns both raw counts and percentages.
    """
    counters = {
        "capture": 0,
        "non_capture": 0,
        "promotion": 0,
        "central": 0,
        "defensive": 0,
        "total_moves": 0
    }
    for _ in range(num_games):
        _, move_history = simulate_random_game(board_size, max_moves=150)
        for move in move_history:
            counters["total_moves"] += 1
            if move["capture"]:
                counters["capture"] += 1
            else:
                counters["non_capture"] += 1
            if move["promotion"]:
                counters["promotion"] += 1
            if move["central"]:
                counters["central"] += 1
            if move["defensive"]:
                counters["defensive"] += 1

    percentages = {}
    if counters["total_moves"] > 0:
        for k, v in counters.items():
            if k != "total_moves":
                percentages[k] = (v / counters["total_moves"]) * 100

    return counters, percentages

def compute_feature_metrics(board, color):
    """
    Computes individual evaluation feature components for the given board state.
    These include material balance, mobility, positional control, king safety penalty,
    promotion bonus, structure bonus, and the adaptive weights.
    """
    opponent = 'black' if color == 'white' else 'white'
    total_pieces = board.black_left + board.white_left
    initial_rows = (board.rows - 2) // 2
    max_pieces = 2 * (initial_rows * (board.cols // 2))
    game_stage = 1.0 - (total_pieces / max_pieces) if max_pieces > 0 else 0

    king_value_base = 1.75
    king_value_endgame = 2.5
    king_value = king_value_base + (king_value_endgame - king_value_base) * game_stage

    black_piece_value = 1.0
    black_king_value = king_value
    white_piece_value = 1.0
    white_king_value = king_value

    black_value = board.black_left * black_piece_value + board.black_kings * black_king_value
    white_value = board.white_left * white_piece_value + board.white_kings * white_king_value

    # Material balance
    material_heuristic = black_value - white_value if color == 'black' else white_value - black_value

    # Mobility
    mobility_self = count_moves(board, color)
    mobility_opp = count_moves(board, opponent)
    mobility_factor = mobility_self - mobility_opp

    # Positional control
    center_weight = 0.3 * (1 - game_stage)
    center_row = board.rows / 2
    center_col = board.cols / 2
    pos_control = 0
    for piece in board.get_all_pieces(color):
        pos_control += 1.0 / (abs(piece.row - center_row) + abs(piece.col - center_col) + 1)
    for piece in board.get_all_pieces(opponent):
        pos_control -= 1.0 / (abs(piece.row - center_row) + abs(piece.col - center_col) + 1)
    combined_heuristic = material_heuristic + 0.1 * mobility_factor + center_weight * pos_control

    # King safety penalty
    king_safety_penalty = 0
    king_safety_weight = 0.1 * (1 - game_stage)
    for piece in board.get_all_pieces(color):
        if piece.king:
            # Penalize kings on the board edge
            if piece.row in [0, board.rows - 1] or piece.col in [0, board.cols - 1]:
                king_safety_penalty += king_safety_weight
    combined_heuristic -= king_safety_penalty

    # Promotion bonus
    promotion_bonus = 0
    for piece in board.get_all_pieces(color):
        if not piece.king:
            if color == 'white':
                promotion_bonus += 0.05 * (board.rows - 1 - piece.row) / (board.rows - 1)
            else:
                promotion_bonus += 0.05 * piece.row / (board.rows - 1)
    combined_heuristic += promotion_bonus

    # Structure bonus (friendly adjacency)
    structure_bonus = 0
    for piece in board.get_all_pieces(color):
        if not piece.king:
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = piece.row + dr, piece.col + dc
                if board.in_bounds(nr, nc):
                    neighbor = board.get_piece(nr, nc)
                    if neighbor != 0 and neighbor.color == color:
                        structure_bonus += 0.02
    combined_heuristic += structure_bonus

    # Adaptive weights
    heuristic_weight, nn_weight = get_adaptive_weights(board)

    return {
       "game_stage": game_stage,
       "material_heuristic": material_heuristic,
       "mobility_factor": mobility_factor,
       "pos_control": pos_control,
       "king_safety_penalty": king_safety_penalty,
       "promotion_bonus": promotion_bonus,
       "structure_bonus": structure_bonus,
       "combined_heuristic": combined_heuristic,
       "heuristic_weight": heuristic_weight,
       "nn_weight": nn_weight
    }

def feature_weight_analysis(board_size, num_samples=100, color='white'):
    """
    For a given range ofboard sizes, this function generates several board states by
    simulating a short random game and computes the evaluation feature metrics.
    It then averages these metrics over all samples.
    """
    metrics_sum = {}
    count = 0
    keys = [
        "game_stage",
        "material_heuristic",
        "mobility_factor",
        "pos_control",
        "king_safety_penalty",
        "promotion_bonus",
        "structure_bonus",
        "combined_heuristic",
        "heuristic_weight",
        "nn_weight"
    ]
    for k in keys:
        metrics_sum[k] = 0.0

    for _ in range(num_samples):
        moves_to_make = random.randint(5, 15)
        board, _ = simulate_random_game(board_size, max_moves=moves_to_make)
        metrics = compute_feature_metrics(board, color)
        for k in keys:
            metrics_sum[k] += metrics[k]
        count += 1

    averages = {k: metrics_sum[k] / count for k in keys}
    return averages

def visualize_frequency(freq_results):
    """
    Creates two subplots:
      1) A line plot of each move type's percentage vs. board size.
      2) A bar chart of total moves vs. board size.
    """
    board_sizes = sorted(freq_results.keys())
    move_types = ["capture", "non_capture", "promotion", "central", "defensive"]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # 1) Line plot for move type percentages
    for move in move_types:
        percentages = [freq_results[size]["percentages"].get(move, 0) for size in board_sizes]
        ax1.plot(board_sizes, percentages, marker='o', label=move)

    # Remove x-axis label, but keep tick labels
    ax1.set_xticks(board_sizes)
    ax1.set_xticklabels(board_sizes)  # show numeric ticks
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Frequency Distribution of Move Types")
    ax1.legend()
    ax1.grid(True)

    # 2) Bar chart for total moves
    total_moves = [freq_results[size]["counts"]["total_moves"] for size in board_sizes]
    ax2.bar(board_sizes, total_moves, color='skyblue')
    ax2.set_xticks(board_sizes)
    ax2.set_xticklabels(board_sizes)
    ax2.set_ylabel("Total Moves")
    ax2.set_title("Total Moves per Board Size")
    ax2.grid(axis='y')

    plt.tight_layout()
    plt.show()

def visualize_features(feature_results):
    """
    Plots each feature as a separate subplot, arranged in a grid.
    """
    # Use consistent ordering of board sizes
    board_sizes = sorted(feature_results.keys())

    # Extract the list of features from the first board size
    feature_keys = list(feature_results[board_sizes[0]].keys())

    # Decide how to arrange subplots: e.g., 2 columns, enough rows for all features
    num_features = len(feature_keys)
    cols = 2
    rows = (num_features + cols - 1) // cols

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(14, rows * 3))

    # In case there's only one row or one column, handle indexing carefully
    if rows == 1 and cols == 1:
        axs = [[axs]]
    elif rows == 1:
        axs = [axs]

    for i, key in enumerate(feature_keys):
        row = i // cols
        col = i % cols
        ax = axs[row][col] if rows > 1 else axs[col]

        values = [feature_results[size][key] for size in board_sizes]
        ax.plot(board_sizes, values, marker='o')
        # Remove x-axis label text
        ax.set_xticks(board_sizes)
        ax.set_xticklabels(board_sizes)
        ax.set_ylabel("Value")
        ax.set_title(key)
        ax.grid(True)

    # Hide any unused subplots if the number of features is not exactly rows*cols
    total_plots = rows * cols
    if num_features < total_plots:
        for j in range(num_features, total_plots):
            row = j // cols
            col = j % cols
            axs[row][col].set_visible(False)

    plt.tight_layout()
    plt.suptitle("Feature Weight Analysis", fontsize=16, y=1.03)
    plt.show()

def main():
    # Analyze board sizes from 6x6 to 18x18 (even sizes)
    board_sizes = list(range(6, 20, 2))
    num_games = 100      # Number of games for frequency analysis
    num_samples = 100    # Number of board samples for feature weight analysis

    freq_results = {}
    feature_results = {}

    print("Frequency Distribution Analysis:")
    for size in board_sizes:
        counts, percentages = frequency_distribution_analysis(size, num_games=num_games)
        freq_results[size] = {"counts": counts, "percentages": percentages}
        print(f"\nBoard Size: {size}x{size}")
        print("Total Moves:", counts["total_moves"])
        for k, v in percentages.items():
            print(f"  {k}: {v:.2f}%")

    print("\nFeature Weight Analysis (averaged over random board states):")
    for size in board_sizes:
        averages = feature_weight_analysis(size, num_samples=num_samples, color='white')
        feature_results[size] = averages
        print(f"\nBoard Size: {size}x{size}")
        for key, value in averages.items():
            print(f"  {key}: {value:.4f}")

    # Visualize the collected results
    visualize_frequency(freq_results)
    visualize_features(feature_results)

if __name__ == "__main__":
    main()
