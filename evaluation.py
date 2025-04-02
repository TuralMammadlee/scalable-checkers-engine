import math
from neural_network import NeuralNetworkModel

HEURISTIC_INIT_WEIGHT = 0.7
NN_INIT_WEIGHT = 0.3

def get_adaptive_weights(board, use_nn_only=False):
    total_pieces = board.black_left + board.white_left
    initial_rows = (board.rows - 2) // 2
    max_pieces = 2 * (initial_rows * (board.cols // 2))
    game_progression = total_pieces / max_pieces if max_pieces > 0 else 0
    if use_nn_only:
        return 0.0, 1.0
    transition_point = 0.5
    transition_speed = 10.0
    factor = 1 / (1 + math.exp(transition_speed * (game_progression - transition_point)))
    nn_weight = HEURISTIC_INIT_WEIGHT + (1 - HEURISTIC_INIT_WEIGHT) * factor
    heuristic_weight = 1 - nn_weight
    return heuristic_weight, nn_weight

def count_moves(board, color):
    total_moves = 0
    for piece in board.get_all_pieces(color):
        moves = board.get_valid_moves(piece)
        total_moves += len(moves)
    return total_moves

def evaluate_board(board, color, model=None, use_nn_only=False):
    # If NN-only flag and model provided, use NN evaluation directly.
    if use_nn_only and model is not None:
        return model.predict(board, color)
        
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

    material_heuristic = black_value - white_value if color == 'black' else white_value - black_value

    mobility_self = count_moves(board, color)
    mobility_opp = count_moves(board, opponent)
    mobility_factor = mobility_self - mobility_opp

    center_weight = 0.3 * (1 - game_stage)
    center_row = board.rows / 2
    center_col = board.cols / 2
    pos_control = 0
    for piece in board.get_all_pieces(color):
        pos_control += 1.0 / (abs(piece.row - center_row) + abs(piece.col - center_col) + 1)
    for piece in board.get_all_pieces(opponent):
        pos_control -= 1.0 / (abs(piece.row - center_row) + abs(piece.col - center_col) + 1)

    combined_heuristic = (
        material_heuristic + 
        0.1 * mobility_factor + 
        center_weight * pos_control
    )

    king_safety_penalty = 0
    king_safety_weight = 0.1 * (1 - game_stage)
    for piece in board.get_all_pieces(color):
        if piece.king:
            if piece.row == 0 or piece.row == board.rows - 1 or piece.col == 0 or piece.col == board.cols - 1:
                king_safety_penalty += king_safety_weight
    combined_heuristic -= king_safety_penalty

    promotion_bonus = 0
    for piece in board.get_all_pieces(color):
        if not piece.king:
            if color == 'white':
                promotion_bonus += 0.05 * (board.rows - 1 - piece.row) / (board.rows - 1)
            else:
                promotion_bonus += 0.05 * piece.row / (board.rows - 1)
    combined_heuristic += promotion_bonus

    structure_bonus = 0
    for piece in board.get_all_pieces(color):
        if not piece.king:
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor_row, neighbor_col = piece.row + dr, piece.col + dc
                if board.in_bounds(neighbor_row, neighbor_col):
                    neighbor = board.get_piece(neighbor_row, neighbor_col)
                    if neighbor != 0 and neighbor.color == color:
                        structure_bonus += 0.02
    combined_heuristic += structure_bonus

    nn_eval = model.predict(board, color) if model is not None else 0

    heuristic_weight, nn_weight = get_adaptive_weights(board)
    return heuristic_weight * combined_heuristic + nn_weight * nn_eval
