import random
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
import pickle
import os
import math
from checkers import Board
from neural_network import NeuralNetworkModel
from evaluation import evaluate_board
import matplotlib.pyplot as plt

# Default training parameters (can be overridden via UI)
DEFAULT_NUM_EPOCHS = 10
DEFAULT_GAMES_PER_EPOCH = 200
DEFAULT_TRAINING_MOVE_MODE = "cnn"  # "cnn" or "random"
DEFAULT_LR = 0.001

# Threshold for updating the best model (if current win rate > this value and > previous best)
BEST_MODEL_UPDATE_THRESHOLD = 0.55

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer saved to {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    self.buffer = pickle.load(f)
                if not self.buffer:
                    raise EOFError("Buffer is empty")
                self.position = len(self.buffer) % self.capacity
                print(f"Loaded {len(self.buffer)} transitions from {filename}")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: {filename} is empty or corrupted. Creating new replay buffer.")
                self.buffer = []
                self.position = 0
                self.save(filename)
            return True
        else:
            print(f"{filename} does not exist. Creating new replay buffer.")
            self.buffer = []
            self.position = 0
            self.save(filename)
            return True

def augment_board(board, color, policy=None):
    # Helper functions for board transformations
    def horizontal_flip_board(original_board):
        new_board = [[0 for _ in range(board.cols)] for _ in range(board.rows)]
        for r in range(board.rows):
            # Identify playable (dark) columns in this row.
            playable_cols = [c for c in range(board.cols) if (r + c) % 2 != 0]
            for c in range(board.cols):
                if (r + c) % 2 != 0:
                    # Find the index in the playable columns list.
                    idx = playable_cols.index(c)
                    # New column is the corresponding index from the reversed playable columns.
                    new_col = playable_cols[len(playable_cols) - 1 - idx]
                    new_board[r][c] = original_board[r][new_col]
                else:
                    # Non-playable squares are copied as-is.
                    new_board[r][c] = original_board[r][c]
        return new_board

    def vertical_flip_board(original_board):
        new_board = [[0 for _ in range(board.cols)] for _ in range(board.rows)]
        for c in range(board.cols):
            # Identify playable (dark) rows for this column.
            playable_rows = [r for r in range(board.rows) if (r + c) % 2 != 0]
            for r in range(board.rows):
                if (r + c) % 2 != 0:
                    idx = playable_rows.index(r)
                    new_row = playable_rows[len(playable_rows) - 1 - idx]
                    new_board[r][c] = original_board[new_row][c]
                else:
                    new_board[r][c] = original_board[r][c]
        return new_board

    def rotate_180_board(original_board):
        # 180° rotation can be done by applying both vertical and horizontal flips.
        return horizontal_flip_board(vertical_flip_board(original_board))

    # Helper functions for policy remapping (assuming policy is a flat list of length board.rows * board.cols)
    def flip_policy_horizontal(original_policy):
        new_policy = original_policy.copy()
        for r in range(board.rows):
            playable_cols = [c for c in range(board.cols) if (r + c) % 2 != 0]
            indices = [r * board.cols + c for c in playable_cols]
            # Reverse the policy values for these indices.
            for orig, new in zip(indices, reversed(indices)):
                new_policy[orig] = original_policy[new]
        return new_policy

    def flip_policy_vertical(original_policy):
        new_policy = original_policy.copy()
        for c in range(board.cols):
            playable_rows = [r for r in range(board.rows) if (r + c) % 2 != 0]
            indices = [r * board.cols + c for r in playable_rows]
            for orig, new in zip(indices, reversed(indices)):
                new_policy[orig] = original_policy[new]
        return new_policy

    def rotate_policy_180(original_policy):
        # 180° rotation: apply vertical then horizontal flips on the policy.
        return flip_policy_horizontal(flip_policy_vertical(original_policy))

    augmented = []
    if policy is not None:
        augmented.append((board, policy))
    else:
        augmented.append(board)

    # Vertical flip
    v_flip = board.copy()
    v_flip.board = vertical_flip_board(board.board)
    if policy is not None:
        v_policy = flip_policy_vertical(policy)
        augmented.append((v_flip, v_policy))
    else:
        augmented.append(v_flip)

    # 180° rotation
    rot_180 = board.copy()
    rot_180.board = rotate_180_board(board.board)
    if policy is not None:
        r_policy = rotate_policy_180(policy)
        augmented.append((rot_180, r_policy))
    else:
        augmented.append(rot_180)

    return augmented


def select_move(board, color, model, temperature=1.0, move_mode="cnn"):
    """Select a move using the neural network with temperature scaling"""
    try:
        # Get policy predictions from the model
        policy = model.predict_move(board, color)
        
        # Get valid moves
        valid_moves = []
        for piece in board.get_all_pieces(color):
            piece_valid_moves = board.get_valid_moves(piece)
            for move, skipped in piece_valid_moves.items():
                valid_moves.append((piece, move, skipped))
        
        if not valid_moves:
            return None
        
        # Convert policy to numpy array and handle NaN values
        policy_np = policy.cpu().numpy()
        policy_np = np.nan_to_num(policy_np, nan=-float('inf'))
        
        # Apply temperature scaling
        policy_np = policy_np / temperature
        
        # Get indices of valid moves
        valid_indices = [move[0] * board.cols + move[1] for _, move, _ in valid_moves]
        
        # Extract scores for valid moves
        valid_scores = policy_np[valid_indices]
        
        # Handle any remaining NaN values
        valid_scores = np.nan_to_num(valid_scores, nan=-float('inf'))
        
        # Normalize scores to probabilities
        exp_scores = np.exp(valid_scores - np.max(valid_scores))  # Subtract max for numerical stability
        sum_exp = np.sum(exp_scores)
        
        if sum_exp == 0 or np.isnan(sum_exp):
            # If all scores are -inf or NaN, use uniform distribution
            probs = np.ones(len(valid_moves)) / len(valid_moves)
        else:
            probs = exp_scores / sum_exp
        
        # Ensure probabilities sum to 1 and contain no NaN values
        probs = np.nan_to_num(probs, nan=0.0)
        probs = probs / np.sum(probs)
        
        # Select move based on probabilities
        chosen_index = np.random.choice(len(valid_moves), p=probs)
        return valid_moves[chosen_index]
        
    except Exception as e:
        print(f"Error in select_move: {e}")
        # Fallback to random move if there's an error
        valid_moves = []
        for piece in board.get_all_pieces(color):
            piece_valid_moves = board.get_valid_moves(piece)
            for move, skipped in piece_valid_moves.items():
                valid_moves.append((piece, move, skipped))
        if valid_moves:
            return random.choice(valid_moves)
        return None

def simulate_game(model, best_model=None, temperature=1.0, move_mode="cnn", board_size=8):
    board = Board(rows=board_size, cols=board_size)
    game_transitions = [] # Renamed from transitions to avoid conflict in loop
    move_history = []
    player_turn = 'white'
    no_progress_count = 0
    while board.winner() is None and no_progress_count < 40:
        current_model = model if player_turn == 'white' else best_model if best_model is not None else model
        state = copy.deepcopy(board) # State *before* the move
        move = select_move(board, player_turn, current_model, temperature, move_mode)
        if move is None:
            # If no move possible, game ends, determine winner based on whose turn it is
            winner = 'black' if player_turn == 'white' else 'white'
            break
        
        piece, action, skipped = move
        move_history.append((player_turn, piece, action, skipped))
        
        # Create policy target for this move
        policy_target = torch.zeros(board_size * board_size)
        move_idx = action[0] * board_size + action[1]
        if 0 <= move_idx < len(policy_target):
            policy_target[move_idx] = 1.0

        # --- Apply Augmentation Here --- 
        augmented_experiences = augment_board(state, player_turn, policy_target)

        # Store original and augmented transitions (with reward=0 for now)
        # Note: next_state is not determined yet, will be filled after move execution
        for aug_state, aug_policy in augmented_experiences:
             # Store state, turn, reward (initially 0), policy_target. Placeholder for next_state and done.
             game_transitions.append([aug_state, player_turn, 0, None, False, aug_policy]) 
        
        # Execute the move on the main board
        pieces_before = board.black_left + board.white_left
        kings_before = board.black_kings + board.white_kings
        board.move(piece, action[0], action[1])
        if skipped:
            board.remove(skipped)
            
        # --- Update next_state for all transitions generated in this step --- 
        current_next_state = copy.deepcopy(board)
        for i in range(len(game_transitions) - len(augmented_experiences), len(game_transitions)):
            game_transitions[i][3] = current_next_state # Update the next_state placeholder
        
        pieces_after = board.black_left + board.white_left
        kings_after = board.black_kings + board.white_kings
        if pieces_after < pieces_before or kings_after > kings_before:
            no_progress_count = 0
        else:
            no_progress_count += 1
            
        player_turn = 'black' if player_turn == 'white' else 'white'
        
    # If loop finished without break, determine winner from board state
    if 'winner' not in locals():
        winner = board.winner()

    # Propagate final game outcome reward to each transition
    if winner == "white":
        final_reward = 1
    elif winner == "black":
        final_reward = -1
    else:
        final_reward = 0 # Draw

    # Update reward for each transition, considering player perspective
    final_transitions = []
    for state, turn, _, next_state, done, policy_target in game_transitions:
        perspective_reward = final_reward if turn == 'white' else -final_reward
        # Mark the last state-action as done (though 'done' isn't heavily used later here)
        is_done = (next_state is None) # Or check if next_state itself represents a terminal state if needed
        final_transitions.append((state, turn, perspective_reward, next_state, is_done, policy_target))

    return final_transitions, winner, move_history

def sample_validation_data(replay_buffer, sample_size=100):
    if len(replay_buffer) < sample_size:
        sample_size = len(replay_buffer)
    return random.sample(replay_buffer.buffer, sample_size)

def validate_model(model, validation_data):
    model.model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    for state, player_turn, reward, next_state, done, policy_target in validation_data:
        input_tensor = model.board_to_tensor(state, player_turn)
        pred_value, pred_policy_logits = model.model(input_tensor)
        pred_value = pred_value.squeeze(-1)
        target_value = torch.tensor([reward], dtype=torch.float32, device=pred_value.device)
        value_loss = model.value_criterion(pred_value, target_value)
        target_policy_index = torch.argmax(policy_target).unsqueeze(0).to(pred_policy_logits.device, dtype=torch.long)
        policy_loss = model.policy_criterion(pred_policy_logits, target_policy_index)
        loss = value_loss + policy_loss
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()
        predicted_index = torch.argmax(pred_policy_logits).item()
        if predicted_index == target_policy_index.item():
            correct += 1
        total += 1
    avg_policy_loss = total_policy_loss / total if total > 0 else 0
    avg_value_loss = total_value_loss / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    model.model.train()
    return avg_policy_loss, avg_value_loss, avg_loss, accuracy

def online_evaluation(model, best_model, num_games=10, board_size=8, temperature=1.0, move_mode="cnn"):
    wins = 0
    for i in range(num_games):
        transitions, winner, move_history = simulate_game(model, best_model=best_model,
                                                          temperature=temperature, move_mode=move_mode,
                                                          board_size=board_size)
        if winner == "white":
            wins += 1
    win_rate = wins / num_games
    return win_rate

def train(num_epochs=DEFAULT_NUM_EPOCHS, games_per_epoch=DEFAULT_GAMES_PER_EPOCH,
          training_move_mode=DEFAULT_TRAINING_MOVE_MODE, lr=DEFAULT_LR,
          replay_buffer_file='replay_buffer_8x8.pkl', board_size=8):
    model = NeuralNetworkModel(board_rows=board_size, board_cols=board_size, lr=lr)
    best_model = NeuralNetworkModel(board_rows=board_size, board_cols=board_size, lr=lr)
    replay_buffer = ReplayBuffer(capacity=10000)
    replay_buffer.load(replay_buffer_file)

    # For tracking metrics and best model performance
    epoch_policy_losses = []
    epoch_value_losses = []
    epoch_total_losses = []
    epoch_accuracies = []
    epoch_win_rates = []
    epoch_training_losses = []  # To track average training loss per epoch
    epoch_mse = []  # To track Mean Squared Error per epoch
    epoch_mae = []  # To track Mean Absolute Error per epoch
    best_win_rate = 0.0

    # Number of training update steps per epoch (this is a hyperparameter)
    batch_size = 32
    updates_per_epoch = 100

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # Self-play simulation
        for game in tqdm(range(games_per_epoch)):
            transitions, winner, move_history = simulate_game(
                model, 
                best_model=best_model,
                temperature=1.0,
                move_mode=training_move_mode,
                board_size=board_size
            )
            for transition in transitions:
                replay_buffer.add(transition)
        # Perform training updates
        training_losses = []
        for _ in range(updates_per_epoch):
            batch = replay_buffer.sample(batch_size)
            boards = [t[0] for t in batch]
            colors = [t[1] for t in batch]
            target_values = [t[2] for t in batch]
            policy_targets = [t[5] for t in batch]
            loss = model.update_batch(boards, colors, target_values, policy_targets)
            training_losses.append(loss)
        avg_train_loss = sum(training_losses) / len(training_losses) if training_losses else 0
        print(f"Training loss: {avg_train_loss:.4f}")
        epoch_training_losses.append(avg_train_loss)

        # Offline validation
        val_data = sample_validation_data(replay_buffer, sample_size=100)
        avg_policy_loss, avg_value_loss, avg_loss, accuracy = validate_model(model, val_data)
        # --- Additional Validation Metrics (MSE and MAE) ---
        total_squared_error = 0.0
        total_absolute_error = 0.0
        total_samples = 0
        for state, player_turn, reward, next_state, done, policy_target in val_data:
            input_tensor = model.board_to_tensor(state, player_turn)
            pred_value, _ = model.model(input_tensor)
            pred_value = pred_value.squeeze(-1)
            error = pred_value.item() - reward
            total_squared_error += error ** 2
            total_absolute_error += abs(error)
            total_samples += 1
        mse = total_squared_error / total_samples if total_samples > 0 else 0
        mae = total_absolute_error / total_samples if total_samples > 0 else 0

        print(f"Validation - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Total Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        print(f"Additional Value Prediction Errors - MSE: {mse:.4f}, MAE: {mae:.4f}")
        epoch_policy_losses.append(avg_policy_loss)
        epoch_value_losses.append(avg_value_loss)
        epoch_total_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        epoch_mse.append(mse)
        epoch_mae.append(mae)

        # Online evaluation
        win_rate = online_evaluation(model, best_model=best_model, num_games=10, board_size=board_size)
        print(f"Online evaluation win rate (current vs. best): {win_rate*100:.2f}%")
        epoch_win_rates.append(win_rate)
        if win_rate > BEST_MODEL_UPDATE_THRESHOLD and win_rate > best_win_rate:
            best_win_rate = win_rate
            best_model.model.load_state_dict(model.model.state_dict())
            torch.save(best_model.model.state_dict(), "best_model.pth")
            print(f"*** Best model updated at epoch {epoch+1} with win rate: {win_rate*100:.2f}% ***")
        replay_buffer.save(replay_buffer_file)

    epochs_range = range(1, len(epoch_policy_losses)+1)
    
    # Plot offline validation policy loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_policy_losses, label="Policy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Policy Loss")
    plt.title("Offline Validation Policy Loss Over Epochs")
    plt.legend()
    plt.show()

    # Plot offline validation value loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_value_losses, label="Value Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value Loss")
    plt.title("Offline Validation Value Loss Over Epochs")
    plt.legend()
    plt.show()

    # Plot offline validation total loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_total_losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Offline Validation Total Loss Over Epochs")
    plt.legend()
    plt.show()

    # Plot offline validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, [a*100 for a in epoch_accuracies], label="Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Offline Validation Accuracy Over Epochs")
    plt.legend()
    plt.show()

    # Plot online evaluation win rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, [w*100 for w in epoch_win_rates], label="Online Win Rate (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Win Rate (%)")
    plt.title("Online Evaluation Win Rate Over Epochs")
    plt.legend()
    plt.show()

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_training_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

    # Plot Mean Squared Error (MSE)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_mse, label="MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Offline Validation MSE Over Epochs")
    plt.legend()
    plt.show()

    # Plot Mean Absolute Error (MAE)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_mae, label="MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title("Offline Validation MAE Over Epochs")
    plt.legend()
    plt.show()

    print("\nTraining complete.")
    print(f"Final trained model win rate: {epoch_win_rates[-1]*100:.2f}%")
    print(f"Best model win rate achieved: {best_win_rate*100:.2f}% (saved as 'best_model.pth')")
    return model, best_model

if __name__ == '__main__':
    trained_model, best_model = train()
