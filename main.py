import sys
import tkinter as tk
import random
import queue
import os
from checkers import Board
from minimax import iterative_deepening
from ui import UI
from neural_network import NeuralNetworkModel
from train import train

# Global configuration variables (set via the UI dialogs)
TRAIN_MODEL = False
USE_TRAINED = False
GAME_MODE = "pva"  # "pva" for Player vs AI, "ava" for AI vs AI
BOARD_SIZE = 8     # Game board size for playing
AI_DIFFICULTY = "medium"
DEBUG_MODE = True
REPLAY_BUFFER_FILE = "replay_buffer_8x8.pkl"

# Training-specific parameters (set via training options dialog)
TRAIN_BOARD_SIZE = 8
NUM_EPOCHS = 20
GAMES_PER_EPOCH = 200
LR = 0.001
TRAINING_MOVE_MODE = "cnn"  # "cnn" or "random"

# Create a queue for UI callbacks
ui_queue = queue.Queue()
global_ui = None  # Will hold our UI instance

def process_ui_queue():
    while not ui_queue.empty():
        callback = ui_queue.get()
        try:
            callback()
        except Exception as e:
            print("Error in UI callback:", e)
    if global_ui is not None:
        global_ui.root.after(50, process_ui_queue)
    else:
        print("global_ui is None in process_ui_queue!")

def show_startup_dialog(root):
    dialog = tk.Toplevel(root)
    dialog.title("Checkers AI - Startup Options")
    dialog.grab_set()
    dialog.geometry("400x550")
    dialog.configure(bg="#f0f0f0")
    
    title_label = tk.Label(dialog, text="Checkers AI", font=("Arial", 16, "bold"), bg="#f0f0f0")
    title_label.pack(padx=10, pady=10)
    
    train_frame = tk.LabelFrame(dialog, text="Training Options", padx=10, pady=10, bg="#f0f0f0")
    train_frame.pack(padx=10, pady=5, fill="x")
    train_var = tk.StringVar(value="no")
    tk.Radiobutton(train_frame, text="Train a new model", variable=train_var, value="yes", bg="#f0f0f0").pack(anchor="w")
    tk.Radiobutton(train_frame, text="Skip training", variable=train_var, value="no", bg="#f0f0f0").pack(anchor="w")
    
    use_trained_var = tk.StringVar(value="yes")
    tk.Label(train_frame, text="Use saved trained model:", bg="#f0f0f0").pack(anchor="w", pady=(10, 0))
    tk.Radiobutton(train_frame, text="Yes (recommended)", variable=use_trained_var, value="yes", bg="#f0f0f0").pack(anchor="w")
    tk.Radiobutton(train_frame, text="No (use random initial weights)", variable=use_trained_var, value="no", bg="#f0f0f0").pack(anchor="w")
    
    buffer_frame = tk.LabelFrame(dialog, text="Replay Buffer File", padx=10, pady=10, bg="#f0f0f0")
    buffer_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(buffer_frame, text="Enter replay buffer file name:", bg="#f0f0f0").pack(anchor="w")
    replay_buffer_entry = tk.Entry(buffer_frame)
    replay_buffer_entry.insert(0, "replay_buffer_8x8.pkl")
    replay_buffer_entry.pack(padx=10, pady=5, fill="x")
    
    mode_frame = tk.LabelFrame(dialog, text="Game Mode", padx=10, pady=10, bg="#f0f0f0")
    mode_frame.pack(padx=10, pady=5, fill="x")
    mode_var = tk.StringVar(value="pva")
    tk.Radiobutton(mode_frame, text="Player vs AI", variable=mode_var, value="pva", bg="#f0f0f0").pack(anchor="w")
    tk.Radiobutton(mode_frame, text="AI vs AI", variable=mode_var, value="ava", bg="#f0f0f0").pack(anchor="w")
    
    difficulty_frame = tk.LabelFrame(dialog, text="AI Difficulty", padx=10, pady=10, bg="#f0f0f0")
    difficulty_frame.pack(padx=10, pady=5, fill="x")
    difficulty_var = tk.StringVar(value="medium")
    tk.Radiobutton(difficulty_frame, text="Easy (depth 2)", variable=difficulty_var, value="easy", bg="#f0f0f0").pack(anchor="w")
    tk.Radiobutton(difficulty_frame, text="Medium (depth 4)", variable=difficulty_var, value="medium", bg="#f0f0f0").pack(anchor="w")
    tk.Radiobutton(difficulty_frame, text="Hard (depth 6)", variable=difficulty_var, value="hard", bg="#f0f0f0").pack(anchor="w")
    
    debug_frame = tk.LabelFrame(dialog, text="Debug Options", padx=10, pady=10, bg="#f0f0f0")
    debug_frame.pack(padx=10, pady=5, fill="x")
    debug_var = tk.BooleanVar(value=True)
    tk.Checkbutton(debug_frame, text="Enable debug output", variable=debug_var, bg="#f0f0f0").pack(anchor="w")
    
    board_frame = tk.LabelFrame(dialog, text="Board Size for Game", padx=10, pady=10, bg="#f0f0f0")
    board_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(board_frame, text="Enter board size (even, min 6):", bg="#f0f0f0").pack(anchor="w")
    board_size_entry = tk.Entry(board_frame)
    board_size_entry.insert(0, "6")
    board_size_entry.pack(padx=10, pady=5, fill="x")
    
    def on_start():
        nonlocal train_var, use_trained_var, mode_var, difficulty_var, debug_var, board_size_entry, replay_buffer_entry
        global TRAIN_MODEL, USE_TRAINED, GAME_MODE, BOARD_SIZE, AI_DIFFICULTY, DEBUG_MODE, REPLAY_BUFFER_FILE
        TRAIN_MODEL = (train_var.get() == "yes")
        USE_TRAINED = (use_trained_var.get() == "yes")
        GAME_MODE = mode_var.get()
        AI_DIFFICULTY = difficulty_var.get()
        DEBUG_MODE = debug_var.get()
        try:
            size = int(board_size_entry.get())
            if size < 6 or size % 2 != 0:
                size = 6
            BOARD_SIZE = size
        except:
            BOARD_SIZE = 6
        default_buffer_name = f"replay_buffer_{BOARD_SIZE}x{BOARD_SIZE}.pkl"
        entered_name = replay_buffer_entry.get().strip()
        REPLAY_BUFFER_FILE = entered_name if entered_name else default_buffer_name
        dialog.destroy()
    
    start_button = tk.Button(dialog, text="Start Game", command=on_start, 
                             bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                             activebackground="#3e8e41", activeforeground="white")
    start_button.pack(pady=20, ipadx=20, ipady=5)
    
    version_label = tk.Label(dialog, text="Version 2.1", font=("Arial", 8), fg="gray", bg="#f0f0f0")
    version_label.pack(side="bottom", pady=5)
    
    root.wait_window(dialog)

def show_training_options_dialog(root):
    dialog = tk.Toplevel(root)
    dialog.title("Training Options")
    dialog.grab_set()
    dialog.geometry("400x450")
    dialog.configure(bg="#f0f0f0")
    
    title_label = tk.Label(dialog, text="Training Configuration", font=("Arial", 16, "bold"), bg="#f0f0f0")
    title_label.pack(padx=10, pady=10)
    
    board_size_frame = tk.LabelFrame(dialog, text="Training Board Size", padx=10, pady=10, bg="#f0f0f0")
    board_size_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(board_size_frame, text="Enter training board size (even, min 6):", bg="#f0f0f0").pack(anchor="w")
    training_board_size_entry = tk.Entry(board_size_frame)
    training_board_size_entry.insert(0, str(BOARD_SIZE))
    training_board_size_entry.pack(padx=10, pady=5, fill="x")
    
    epochs_frame = tk.LabelFrame(dialog, text="Number of Epochs", padx=10, pady=10, bg="#f0f0f0")
    epochs_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(epochs_frame, text="Enter number of epochs:", bg="#f0f0f0").pack(anchor="w")
    epochs_entry = tk.Entry(epochs_frame)
    epochs_entry.insert(0, "20")
    epochs_entry.pack(padx=10, pady=5, fill="x")
    
    games_frame = tk.LabelFrame(dialog, text="Games per Epoch", padx=10, pady=10, bg="#f0f0f0")
    games_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(games_frame, text="Enter number of self-play games per epoch:", bg="#f0f0f0").pack(anchor="w")
    games_entry = tk.Entry(games_frame)
    games_entry.insert(0, "200")
    games_entry.pack(padx=10, pady=5, fill="x")
    
    lr_frame = tk.LabelFrame(dialog, text="Learning Rate", padx=10, pady=10, bg="#f0f0f0")
    lr_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(lr_frame, text="Enter learning rate (e.g., 0.001):", bg="#f0f0f0").pack(anchor="w")
    lr_entry = tk.Entry(lr_frame)
    lr_entry.insert(0, "0.001")
    lr_entry.pack(padx=10, pady=5, fill="x")
    
    move_mode_frame = tk.LabelFrame(dialog, text="Training Move Mode", padx=10, pady=10, bg="#f0f0f0")
    move_mode_frame.pack(padx=10, pady=5, fill="x")
    move_mode_var = tk.StringVar(value="cnn")
    tk.Radiobutton(move_mode_frame, text="CNN-based (cnn)", variable=move_mode_var, value="cnn", bg="#f0f0f0").pack(anchor="w")
    tk.Radiobutton(move_mode_frame, text="Random", variable=move_mode_var, value="random", bg="#f0f0f0").pack(anchor="w")
    
    def on_train_start():
        nonlocal training_board_size_entry, epochs_entry, games_entry, lr_entry, move_mode_var
        global TRAIN_BOARD_SIZE, NUM_EPOCHS, GAMES_PER_EPOCH, LR, TRAINING_MOVE_MODE
        try:
            t_size = int(training_board_size_entry.get())
            if t_size < 6 or t_size % 2 != 0:
                t_size = BOARD_SIZE
            TRAIN_BOARD_SIZE = t_size
        except:
            TRAIN_BOARD_SIZE = BOARD_SIZE
        try:
            NUM_EPOCHS = int(epochs_entry.get())
        except:
            NUM_EPOCHS = 20
        try:
            GAMES_PER_EPOCH = int(games_entry.get())
        except:
            GAMES_PER_EPOCH = 200
        try:
            LR = float(lr_entry.get())
        except:
            LR = 0.001
        TRAINING_MOVE_MODE = move_mode_var.get()
        dialog.destroy()
    
    start_button = tk.Button(dialog, text="Start Training", command=on_train_start, 
                             bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                             activebackground="#3e8e41", activeforeground="white")
    start_button.pack(pady=20, ipadx=20, ipady=5)
    
    dialog.wait_window(dialog)
# Difficulty map from difficulty string -> search depth
DIFFICULTY_MAP = {
    "easy": 2,
    "medium": 4,
    "hard": 6
}

# Now our main function
def main():
    global global_ui
    root = tk.Tk()
    root.withdraw()  # hide main window during dialogs

    # Show startup dialog
    show_startup_dialog(root)
    
    if TRAIN_MODEL:
        show_training_options_dialog(root)
        print("Starting training...")
        model, best_model = train(num_epochs=NUM_EPOCHS, games_per_epoch=GAMES_PER_EPOCH,
                                  training_move_mode=TRAINING_MOVE_MODE, lr=LR,
                                  replay_buffer_file=REPLAY_BUFFER_FILE, board_size=TRAIN_BOARD_SIZE)
        print("Training complete. Exiting program after displaying plots.")
        sys.exit()  # exit after training
    else:
        # Create a board and pass the same root to UI
        board = Board(rows=BOARD_SIZE, cols=BOARD_SIZE)
        global_ui = UI(board, width=600, height=600, root=root)
        
        # Create AI model for making moves (for pva or ava)
        ai_model = NeuralNetworkModel(board_rows=BOARD_SIZE, board_cols=BOARD_SIZE)
        if USE_TRAINED and os.path.exists("best_model.pth"):
            ai_model.load_model("best_model.pth")
        
        # Define a variable to track current player (assume player is white)
        current_player = "white"
        selected_piece = None
        valid_moves_for_piece = {}

        def check_winner():
            winner = board.winner()
            if winner is not None:
                global_ui.set_status(f"Game Over! Winner: {winner}")
                return True
            return False

        def apply_move(piece, dest_r, dest_c, skipped):
            board.move(piece, dest_r, dest_c)
            if skipped:
                board.remove(skipped)
            global_ui.update()

        def do_ai_move():
            nonlocal current_player
            if check_winner():
                return
            # Use iterative_deepening for the current player's move
            move_info = iterative_deepening(
                board,
                max_depth=DIFFICULTY_MAP.get(AI_DIFFICULTY, 4),
                color=current_player,
                model=ai_model,
                time_limit=5.0
            )
            if move_info:
                piece_info = move_info.get("piece")
                move_coords = move_info.get("move")
                skipped = move_info.get("skipped")
                if piece_info and move_coords:
                    piece = board.get_piece(piece_info.row, piece_info.col)
                    if piece:
                        apply_move(piece, move_coords[0], move_coords[1], skipped)
            if not check_winner():
                current_player = "white" if current_player == "black" else "black"

        def handle_player_click(row, col):
            nonlocal selected_piece, valid_moves_for_piece, current_player
            if check_winner():
                return
            if current_player == "white":
                if selected_piece is None:
                    piece = board.get_piece(row, col)
                    if piece and piece.color == "white":
                        selected_piece = piece
                        valid_moves_for_piece = board.get_valid_moves(piece)
                        global_ui.clear_highlights()
                        global_ui.highlight_moves(valid_moves_for_piece.keys())
                else:
                    if (row, col) in valid_moves_for_piece:
                        apply_move(selected_piece, row, col, valid_moves_for_piece[(row, col)])
                        selected_piece = None
                        valid_moves_for_piece = {}
                        global_ui.clear_highlights()
                        if not check_winner():
                            current_player = "black"
                            do_ai_move()
                    else:
                        selected_piece = None
                        valid_moves_for_piece = {}
                        global_ui.clear_highlights()

        def ai_vs_ai_loop():
            nonlocal current_player
            if check_winner():
                return
            move_info = iterative_deepening(
                board,
                max_depth=DIFFICULTY_MAP.get(AI_DIFFICULTY, 4),
                color=current_player,
                model=ai_model,
                time_limit=5.0
            )
            if move_info:
                piece_info = move_info.get("piece")
                move_coords = move_info.get("move")
                skipped = move_info.get("skipped")
                if piece_info and move_coords:
                    piece = board.get_piece(piece_info.row, piece_info.col)
                    if piece:
                        apply_move(piece, move_coords[0], move_coords[1], skipped)
            if not check_winner():
                current_player = "white" if current_player == "black" else "black"
                root.after(500, ai_vs_ai_loop)

        # Start the appropriate mode
        if GAME_MODE == "pva":
            global_ui.set_move_callback(handle_player_click)
        elif GAME_MODE == "ava":
            ai_vs_ai_loop()
        
        process_ui_queue()
        root.deiconify()
        root.mainloop()

if __name__ == '__main__':
    main()
