import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import os
import gc
import threading
import random
from checkers import Board
from minimax import minimax, initialize_zobrist, transposition_table
from neural_network import NeuralNetworkModel
import torch

def monitor_resource_usage(stop_event, result_dict, lock):
    """Thread function to continuously monitor CPU usage and total memory"""
    process = psutil.Process(os.getpid())
    cpu_samples = []
    memory_samples = []

    # Prime CPU percent measurement
    process.cpu_percent(interval=None)

    # Continuously sample until stop event is set
    while not stop_event.is_set():
        cpu_percent = process.cpu_percent(interval=0.5)
        cpu_samples.append(cpu_percent)

        # Measure total memory (not delta)
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_samples.append(current_memory)

    # Calculate averages
    cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    memory_avg = sum(memory_samples) / len(memory_samples) if memory_samples else 0

    with lock:
        result_dict['cpu_avg'] = cpu_avg
        result_dict['memory_avg'] = memory_avg


def measure_resources_over_games(board_sizes=[6, 8, 10, 12, 14, 16, 18], num_games=10, fixed_depth=3):
    """Measure resources across different board sizes over multiple games"""
    transposition_sizes = []
    cpu_usage = []
    ram_usage = []
    
    # Force initial cleanup to get to a clean baseline
    for _ in range(3):  # Multiple GC passes to ensure cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.5)

    print(f"Starting resource measurements across {num_games} games for each board size...")

    # Capture base memory before any board-specific allocations
    process = psutil.Process(os.getpid())
    base_memory = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"Base memory usage: {base_memory:.1f} MB")

    for board_size in board_sizes:
        print(f"\nTesting board size {board_size}x{board_size} over {num_games} games...")

        # Seed RNGs for reproducibility
        random.seed(board_size)
        np.random.seed(board_size)
        torch.manual_seed(board_size)

        # Clean up before starting this board size
        for _ in range(2):  # Multiple GC passes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.5)

        # Initialize model on appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralNetworkModel(board_rows=board_size, board_cols=board_size)

        # Initialize Zobrist hashing and clear TT
        initialize_zobrist(board_size, board_size)
        transposition_table.clear()

        # Start resource monitoring thread
        stop_monitoring = threading.Event()
        resource_results = {}
        lock = threading.Lock()
        monitor_thread = threading.Thread(
            target=monitor_resource_usage,
            args=(stop_monitoring, resource_results, lock)
        )
        monitor_thread.start()

        # Play games
        for game_num in range(num_games):
            print(f"  Playing game {game_num+1}/{num_games}...")

            board = Board(rows=board_size, cols=board_size)
            current_player = 'white'
            for _ in range(random.randint(1, 5)):
                pieces = board.get_all_pieces(current_player)
                if not pieces:
                    break
                piece = random.choice(pieces)
                moves = board.get_valid_moves(piece)
                if not moves:
                    break
                move, skipped = random.choice(list(moves.items()))
                board.move(piece, move[0], move[1])
                if skipped:
                    board.remove(skipped)
                current_player = 'black' if current_player == 'white' else 'white'

            # Count total pieces on the board
            white_pieces = len(board.get_all_pieces('white'))
            black_pieces = len(board.get_all_pieces('black'))
            total_pieces = white_pieces + black_pieces
            
            # Calculate adaptive depth based on number of pieces
            max_possible_pieces = board_size * (board_size//2 - 1)  # Approximate max pieces
            piece_ratio = total_pieces / max_possible_pieces
            
            if piece_ratio < 0.3:  # Less than 30% of pieces remain
                adaptive_depth = 5  # Go deeper
            elif piece_ratio < 0.6:  # Between 30% and 60% of pieces remain
                adaptive_depth = 4  # Medium depth
            else:  # More than 60% of pieces remain
                adaptive_depth = 3  # Default depth
                
            print(f"    Using adaptive depth: {adaptive_depth} (remaining pieces: {total_pieces})")

            _evaluation, _move = minimax(
                board, adaptive_depth, -float('inf'), float('inf'),
                True, current_player, model
            )
            del board

        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()

        # Record averages
        cpu_avg = resource_results.get('cpu_avg', 0)
        memory_avg = resource_results.get('memory_avg', 0)
        cpu_usage.append(cpu_avg)
        ram_usage.append(memory_avg)
        tt_size = len(transposition_table.cache)
        transposition_sizes.append(tt_size)

        print(f"  Completed {num_games} games:")
        print(f"  Transposition table entries: {tt_size}")
        print(f"  Average CPU Usage: {cpu_avg:.1f}%, Average RAM Usage: {memory_avg:.1f} MB")

        # Clean up model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'board_sizes': board_sizes,
        'transposition_sizes': transposition_sizes,
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'num_games': num_games,
        'fixed_depth': 'adaptive',  # Update this to reflect adaptive depth
        'base_memory': base_memory
    }

def plot_memory_usage(data):
    """Create a memory usage chart with green bars and values at the top"""
    board_sizes = data['board_sizes']
    memory_values = data['ram_usage']  # Total memory values
    base_memory = data['base_memory']  # Base memory from measurement
    depth_info = data['fixed_depth']  # Could be "adaptive" now
    
    # Subtract base memory to get the memory usage due to the board
    adjusted_values = [value - base_memory for value in memory_values]
    
    # Convert board sizes to string labels for x-axis
    board_labels = [f"{size}x{size}" for size in board_sizes]
    
    # Create the figure with white background
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Create the bar chart
    bars = plt.bar(board_labels, adjusted_values, color='#32CD32')
    
    # Set the title and labels
    plt.title(f'Memory Usage Above Base (Depth: {depth_info})', fontsize=14)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    
    # Set y-axis limits based on the actual data
    max_val = max(adjusted_values) if adjusted_values else 100
    plt.ylim(0, max_val * 1.1)  # Add 10% headroom
    
    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='-', alpha=0.2)
    
    # Add memory value labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Display memory values at the top of each bar
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=10)
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add ticks on the left side only
    ax.tick_params(axis='y', which='both', length=0)
    
    # Remove x-axis line but keep ticks
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Save and display
    plt.tight_layout()
    plt.savefig('memory_usage.png', dpi=300)
    plt.show()

def plot_transposition_table(data):
    """Create a chart showing transposition table entries for different board sizes"""
    board_sizes = data['board_sizes']
    tt_entries = data['transposition_sizes']
    depth_info = data['fixed_depth']  # Could be "adaptive" now
    
    # Convert board sizes to string labels for x-axis
    board_labels = [f"{size}x{size}" for size in board_sizes]
    
    # Create the figure with white background
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Create the bar chart with the purple color
    bars = plt.bar(board_labels, tt_entries, color='#9b59b6')  # Purple
    
    # Set the title and labels
    plt.title(f'Transposition Table Size (Depth: {depth_info})', fontsize=14)
    plt.ylabel('Number of Entries', fontsize=12)
    
    # Set y-axis limits based on data
    max_val = max(tt_entries) if tt_entries else 10
    plt.ylim(0, max_val * 1.2)  # Add 20% headroom
    
    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='-', alpha=0.2)
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Display values at the top of each bar
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{height}',
                ha='center', va='bottom', fontsize=10)
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add ticks on the left side only
    ax.tick_params(axis='y', which='both', length=0)
    
    # Remove x-axis line but keep ticks
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Save and display
    plt.tight_layout()
    plt.savefig('transposition_table_size.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Measuring transposition table and resource usage across board sizes with adaptive depth...")
    data = measure_resources_over_games(
        board_sizes=[6, 8, 10, 12, 14, 16, 18],
        num_games=10
    )
    print("Plotting results...")
    plot_memory_usage(data)
    plot_transposition_table(data)
    print("Done!")
