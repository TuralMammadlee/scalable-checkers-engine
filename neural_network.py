import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import time
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class CheckersCNN(nn.Module):
    def __init__(self, board_rows, board_cols, num_residual_blocks=3):
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
        self.fc_policy2 = nn.Linear(128, 64)  # Fixed size intermediate layer
        
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
        policy = self.fc_policy2(policy)            # shape: (batch, 64)
        
        # Reshape policy to match board size
        policy = policy.view(-1, 8, 8)  # Reshape to 8x8 grid
        policy = F.interpolate(policy.unsqueeze(1), 
                             size=(self.board_rows, self.board_cols), 
                             mode='bilinear', 
                             align_corners=False)
        policy = policy.squeeze(1).view(-1, self.board_rows * self.board_cols)
        
        # Value branch
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = self.adaptive_pool_value(value)     # shape: (batch, 32, 1, 1)
        value = value.view(value.size(0), -1)       # shape: (batch, 32)
        value = self.dropout(value)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))   # shape: (batch, 1)
        
        return value, policy

class NeuralNetworkModel:
    def __init__(self, board_rows=8, board_cols=8, lr=0.001, num_residual_blocks=3):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.model = CheckersCNN(board_rows, board_cols, num_residual_blocks=num_residual_blocks)
        # Set device to CUDA if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.value_criterion = nn.MSELoss()
        self.policy_criterion = nn.CrossEntropyLoss()
        self.model.eval()
        self.prediction_timeout = 1.0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    def __call__(self, x):
        return self.model(x)
    
    def board_to_tensor(self, board, color):
        try:
            if board is None or not hasattr(board, 'board'):
                return torch.zeros((1, 9, self.board_rows, self.board_cols), dtype=torch.float32, device=self.device)
            
            # Initialize 9 channels with the correct board dimensions
            channels = [[[0 for _ in range(self.board_cols)] for _ in range(self.board_rows)] for _ in range(9)]
            
            # Channels 0-3: pieces (non-king and king for each color)
            for r in range(min(board.rows, self.board_rows)):
                for c in range(min(board.cols, self.board_cols)):
                    cell = board.board[r][c]
                    if cell != 0:
                        if cell.color == 'black':
                            if cell.king:
                                channels[1][r][c] = 1.0
                            else:
                                channels[0][r][c] = 1.0
                        else:
                            if cell.king:
                                channels[3][r][c] = 1.0
                            else:
                                channels[2][r][c] = 1.0
            
            # Channel 4: Current player's valid moves
            for piece in board.get_all_pieces(color):
                valid_moves = board.get_valid_moves(piece)
                for move in valid_moves:
                    if 0 <= move[0] < self.board_rows and 0 <= move[1] < self.board_cols:
                        channels[4][move[0]][move[1]] = 1.0
            
            # Channel 5: Current player's capture moves
            for piece in board.get_all_pieces(color):
                captures = {}
                board._get_captures(piece, piece.row, piece.col, [], captures)
                for move in captures:
                    if 0 <= move[0] < self.board_rows and 0 <= move[1] < self.board_cols:
                        channels[5][move[0]][move[1]] = 1.0
            
            # Channel 6: Opponent's valid moves
            opponent = 'black' if color == 'white' else 'white'
            for piece in board.get_all_pieces(opponent):
                valid_moves = board.get_valid_moves(piece)
                for move in valid_moves:
                    if 0 <= move[0] < self.board_rows and 0 <= move[1] < self.board_cols:
                        channels[6][move[0]][move[1]] = 1.0
            
            # Channel 7: King safety for current player's kings
            for piece in board.get_all_pieces(color):
                if piece.king:
                    dist = min(piece.row, board.rows - 1 - piece.row, piece.col, board.cols - 1 - piece.col)
                    norm_dist = dist / (min(board.rows, board.cols) / 2)
                    if 0 <= piece.row < self.board_rows and 0 <= piece.col < self.board_cols:
                        channels[7][piece.row][piece.col] = norm_dist
            
            # Channel 8: Piece density (friendly pieces in 3x3 neighborhood)
            density = [[0 for _ in range(self.board_cols)] for _ in range(self.board_rows)]
            friendly_pieces = board.get_all_pieces(color)
            for r in range(self.board_rows):
                for c in range(self.board_cols):
                    count = 0
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < board.rows and 0 <= nc < board.cols:
                                p = board.get_piece(nr, nc)
                                if p != 0 and p.color == color:
                                    count += 1
                    density[r][c] = count / 9.0
                    channels[8][r][c] = density[r][c]
            
            tensor = torch.tensor(channels, dtype=torch.float32, device=self.device)
            tensor = tensor.unsqueeze(0)
            
            if color == 'black':
                temp = tensor[:, 0].clone()
                tensor[:, 0] = tensor[:, 2]
                tensor[:, 2] = temp
            
            return tensor
        except Exception as e:
            print(f"Error in board_to_tensor: {e}")
            traceback.print_exc()
            return torch.zeros((1, 9, self.board_rows, self.board_cols), dtype=torch.float32, device=self.device)

    
    def get_policy_mask(self, board, color):
        """Create a mask for valid moves in the policy output"""
        try:
            # Initialize mask with zeros for all possible moves
            mask = torch.zeros(self.board_rows * self.board_cols, dtype=torch.bool, device=self.device)
            
            # Get all pieces for the current color
            pieces = board.get_all_pieces(color)
            
            # For each piece, get its valid moves and set the corresponding mask indices
            for piece in pieces:
                valid_moves = board.get_valid_moves(piece)
                for move, skipped in valid_moves.items():
                    # Convert move coordinates to policy index
                    if 0 <= move[0] < self.board_rows and 0 <= move[1] < self.board_cols:
                        idx = move[0] * self.board_cols + move[1]
                        if idx < len(mask):  # Ensure index is within bounds
                            mask[idx] = True
            
            return mask
        except Exception as e:
            print(f"Error in get_policy_mask: {e}")
            traceback.print_exc()
            return torch.ones(self.board_rows * self.board_cols, dtype=torch.bool, device=self.device)

    def predict(self, board, color):
        try:
            start_time = time.time()
            x = self.board_to_tensor(board, color)
            with torch.no_grad():
                if time.time() - start_time > self.prediction_timeout:
                    return 0.0
                value, _ = self.model(x)
                clamped_value = torch.clamp(value, -1.0, 1.0)
                if color == 'black':
                    clamped_value = -clamped_value
            return clamped_value.item()
        except Exception as e:
            print(f"Error in predict: {e}")
            traceback.print_exc()
            return 0.0
    
    def predict_move(self, board, color):
        try:
            start_time = time.time()
            x = self.board_to_tensor(board, color)
            with torch.no_grad():
                if time.time() - start_time > self.prediction_timeout:
                    return torch.ones(self.board_rows * board.cols, device=self.device)
                _, policy = self.model(x)
            # Policy masking: set logits for invalid moves to -1e9
            mask = self.get_policy_mask(board, color)
            policy = policy.squeeze(0)
            policy_masked = policy.clone()
            policy_masked[~mask] = -1e9
            return policy_masked
        except Exception as e:
            print(f"Error in predict_move: {e}")
            traceback.print_exc()
            return torch.ones(self.board_rows * board.cols, device=self.device)
    
    def update(self, board, color, target_value, policy_target=None):
        try:
            self.model.train()
            x = self.board_to_tensor(board, color)
            target_value_tensor = torch.tensor([[target_value]], dtype=torch.float32, device=self.device)
            self.optimizer.zero_grad()
            value_output, policy_output = self.model(x)
            value_loss = self.value_criterion(value_output, target_value_tensor)
            if policy_target is not None:
                policy_target = torch.tensor(policy_target, dtype=torch.long, device=self.device)
                policy_loss = self.policy_criterion(policy_output, policy_target)
                loss = value_loss + policy_loss
            else:
                loss = value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.eval()
            return loss.item()
        except Exception as e:
            print(f"Error in update: {e}")
            traceback.print_exc()
            return 0.0
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        try:
            state_dict = torch.load(filename, map_location=self.device)
            # Use non-strict loading to accommodate architecture changes
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"Model loaded from {filename}. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print("Error loading saved model. Using random initial weights.", e)
    
    def update_batch(self, boards, colors, target_values, policy_targets=None):
        try:
            self.model.train()
            tensors = []
            for board, color in zip(boards, colors):
                tensors.append(self.board_to_tensor(board, color))
            x = torch.cat(tensors, dim=0)
            target_value_tensor = torch.tensor(target_values, dtype=torch.float32, device=self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            value_output, policy_output = self.model(x)
            value_loss = self.value_criterion(value_output, target_value_tensor)
            if policy_targets is not None:
                target_indices = []
                for pt in policy_targets:
                    if isinstance(pt, list):
                        target_indices.append(max(range(len(pt)), key=lambda i: pt[i]))
                    elif isinstance(pt, torch.Tensor):
                        target_indices.append(int(pt.argmax().item()))
                    else:
                        target_indices.append(pt)
                policy_target_tensor = torch.tensor(target_indices, dtype=torch.long, device=self.device)
                policy_loss = self.policy_criterion(policy_output, policy_target_tensor)
                loss = value_loss + policy_loss
            else:
                loss = value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.eval()
            return loss.item()
        except Exception as e:
            print(f"Error in update_batch: {e}")
            traceback.print_exc()
            return 0.0
