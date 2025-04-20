import math
import time
import random
import numpy as np
import torch
import traceback
import logging
from evaluation import evaluate_board
from checkers import Board
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("minimax_errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("minimax")

# LRU Cache for transposition table
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def __contains__(self, key):
        return key in self.cache
    
    def clear(self):
        self.cache.clear()

transposition_table = LRUCache(capacity=1000000)
node_expansions = 0
ZOBRIST_TABLE = {}

def initialize_zobrist(rows, cols):
    global ZOBRIST_TABLE
    ZOBRIST_TABLE = {}
    piece_types = ['black', 'black_king', 'white', 'white_king']
    for r in range(rows):
        for c in range(cols):
            for pt in piece_types:
                ZOBRIST_TABLE[(r, c, pt)] = random.getrandbits(64)

def board_hash(board):
    try:
        expected_keys = board.rows * board.cols * 4
        if not ZOBRIST_TABLE or len(ZOBRIST_TABLE) != expected_keys:
            initialize_zobrist(board.rows, board.cols)
        h = 0
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if piece != 0:
                    pt = piece.color if not piece.king else piece.color + "_king"
                    h ^= ZOBRIST_TABLE[(r, c, pt)]
        return h
    except Exception as e:
        logger.error(f"Error in board_hash: {e}", exc_info=True)
        return random.getrandbits(64)

def is_time_up(start_time, time_limit):
    return time.time() - start_time > time_limit * 0.95

def quiescence_search(board, alpha, beta, maximizing_player, color, model, depth=2, start_time=None, time_limit=float('inf')):
    global node_expansions
    node_expansions += 1
    if start_time and is_time_up(start_time, time_limit):
        return evaluate_board(board, color, model=model), None
    stand_pat = evaluate_board(board, color, model=model)
    if depth == 0:
        return stand_pat, None
    try:
        if maximizing_player:
            max_eval = stand_pat
            best_move = None
            capture_moves = []
            for piece in board.get_all_pieces(color):
                captures = {}
                board._get_captures(piece, piece.row, piece.col, [], captures)
                for move, skipped in captures.items():
                    if skipped:
                        temp_board = board.copy()
                        temp_piece = None
                        for p in temp_board.get_all_pieces(color):
                            if p.row == piece.row and p.col == piece.col:
                                temp_piece = p
                                break
                        if temp_piece is None:
                            continue
                        temp_board.move(temp_piece, move[0], move[1])
                        if skipped:
                            temp_board.remove(skipped)
                        capture_moves.append({'board': temp_board, 'piece': piece, 'move': move, 'skipped': skipped})
            if not capture_moves:
                return stand_pat, None
            for move in capture_moves:
                evaluation, _ = quiescence_search(move['board'], alpha, beta, False, color, model, depth - 1, start_time, time_limit)
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                alpha = max(alpha, max_eval)
                if beta <= alpha or (start_time and is_time_up(start_time, time_limit)):
                    break
            return max_eval, best_move
        else:
            min_eval = stand_pat
            best_move = None
            opponent = 'black' if color == 'white' else 'white'
            capture_moves = []
            for piece in board.get_all_pieces(opponent):
                captures = {}
                board._get_captures(piece, piece.row, piece.col, [], captures)
                for move, skipped in captures.items():
                    if skipped:
                        temp_board = board.copy()
                        temp_piece = None
                        for p in temp_board.get_all_pieces(opponent):
                            if p.row == piece.row and p.col == piece.col:
                                temp_piece = p
                                break
                        if temp_piece is None:
                            continue
                        temp_board.move(temp_piece, move[0], move[1])
                        if skipped:
                            temp_board.remove(skipped)
                        capture_moves.append({'board': temp_board, 'piece': piece, 'move': move, 'skipped': skipped})
            if not capture_moves:
                return stand_pat, None
            for move in capture_moves:
                evaluation, _ = quiescence_search(move['board'], alpha, beta, True, color, model, depth - 1, start_time, time_limit)
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                beta = min(beta, min_eval)
                if beta <= alpha or (start_time and is_time_up(start_time, time_limit)):
                    break
            return min_eval, best_move
    except Exception as e:
        logger.error(f"Error in quiescence_search: {e}", exc_info=True)
        return evaluate_board(board, color, model=model), None

def minimax(board, depth, alpha, beta, maximizing_player, color, model, start_time=None, time_limit=float('inf'), use_quiescence=True):
    global node_expansions
    node_expansions += 1
    if node_expansions > 1000000 * depth:
        return evaluate_board(board, color, model=model), None
    if start_time and is_time_up(start_time, time_limit):
        return evaluate_board(board, color, model=model), None
    try:
        board_key = board_hash(board)
        cache_entry = transposition_table.get(board_key)
        if cache_entry and cache_entry.get('depth', 0) >= depth:
            flag = cache_entry.get('flag', 'exact')
            value = cache_entry.get('value', 0)
            if flag == 'exact':
                return value, cache_entry.get('best_move')
            elif flag == 'lower_bound' and value > alpha:
                alpha = value
            elif flag == 'upper_bound' and value < beta:
                beta = value
            if alpha >= beta:
                return value, cache_entry.get('best_move')
        winner = board.winner()
        if depth == 0 or winner is not None:
            if winner == color:
                return 1000, None
            elif winner is not None:
                return -1000, None
            elif depth == 0:
                if use_quiescence and depth == 0:
                    return quiescence_search(board, alpha, beta, maximizing_player, color, model, start_time=start_time, time_limit=time_limit)
                else:
                    return evaluate_board(board, color, model=model), None
        best_move = None
        original_alpha = alpha
        original_beta = beta
        if maximizing_player:
            max_eval = -math.inf
            moves = get_all_moves(board, color)
            if not moves:
                return -1000, None
            moves = order_moves(board, moves, color, model)
            for move in moves:
                # In recursive calls, the current turn switches to the opponent.
                evaluation, _ = minimax(move['board'], depth - 1, alpha, beta, False, color, model, start_time, time_limit, use_quiescence)
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                alpha = max(alpha, evaluation)
                if beta <= alpha or (start_time and is_time_up(start_time, time_limit)):
                    break
            if math.isnan(max_eval) or math.isinf(max_eval):
                max_eval = evaluate_board(board, color, model=model)
            flag = 'exact'
            if max_eval <= original_alpha:
                flag = 'upper_bound'
            elif max_eval >= beta:
                flag = 'lower_bound'
            if not (start_time and is_time_up(start_time, time_limit)):
                transposition_table.put(board_key, {
                    'value': max_eval, 
                    'depth': depth, 
                    'flag': flag,
                    'best_move': best_move
                })
            return max_eval, best_move
        else:
            min_eval = math.inf
            opponent = 'black' if color == 'white' else 'white'
            moves = get_all_moves(board, opponent)
            if not moves:
                return 1000, None
            moves = order_moves(board, moves, opponent, model)
            for move in moves:
                evaluation, _ = minimax(move['board'], depth - 1, alpha, beta, True, color, model, start_time, time_limit, use_quiescence)
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha or (start_time and is_time_up(start_time, time_limit)):
                    break
            if math.isnan(min_eval) or math.isinf(min_eval):
                min_eval = evaluate_board(board, color, model=model)
            flag = 'exact'
            if min_eval <= original_alpha:
                flag = 'upper_bound'
            elif min_eval >= beta:
                flag = 'lower_bound'
            if not (start_time and is_time_up(start_time, time_limit)):
                transposition_table.put(board_key, {
                    'value': min_eval, 
                    'depth': depth, 
                    'flag': flag,
                    'best_move': best_move
                })
            return min_eval, best_move
    except Exception as e:
        logger.error(f"Error in minimax: {e}", exc_info=True)
        return evaluate_board(board, color, model=model), None

def iterative_deepening(board, max_depth, color, model, time_limit=5.0):
    global node_expansions
    node_expansions = 0
    best_move = None
    try:
        start_time = time.time()
        transposition_table.clear()
        
        for depth in range(1, max_depth + 1):
            logger.info(f"Starting search at depth {depth}")
            evaluation, move = minimax(board, depth, -math.inf, math.inf, True, color, model, start_time, time_limit)
            
            if is_time_up(start_time, time_limit):
                logger.info(f"Time limit reached during depth {depth} search")
                break
                
            if move is not None:
                best_move = move
                logger.info(f"Depth {depth} complete - Best eval: {evaluation:.3f}")
            else:
                logger.warning(f"No valid move found at depth {depth}")
                break
                
            if abs(evaluation) > 900:  # Near-terminal state, no need to search deeper
                logger.info(f"Terminal value detected at depth {depth}: {evaluation:.3f}")
                break
                
        end_time = time.time()
        search_time = end_time - start_time
        logger.info(f"Search completed in {search_time:.3f}s. Nodes expanded: {node_expansions}")
        logger.info(f"Nodes per second: {node_expansions / search_time if search_time > 0 else 0:.1f}")
        
        if best_move is None:
            logger.warning("No valid move found during iterative deepening")
            # Try a fallback single depth search
            evaluation, best_move = minimax(board, 1, -math.inf, math.inf, True, color, model, None, float('inf'))
            if best_move is None:
                # Emergency fallback: get any valid move
                logger.error("Emergency fallback: searching for any valid move")
                moves = get_all_moves(board, color)
                if moves:
                    best_move = moves[0]
                    
        return best_move
    except Exception as e:
        logger.error(f"Error in iterative_deepening: {e}", exc_info=True)
        # Emergency fallback
        try:
            moves = get_all_moves(board, color)
            if moves:
                logger.info("Using fallback move selection after error")
                return moves[0]
        except Exception as nested_e:
            logger.error(f"Fallback move selection failed: {nested_e}", exc_info=True)
        return None

def get_all_moves(board, color):
    try:
        moves = []
        for piece in board.get_all_pieces(color):
            valid_moves = board.get_valid_moves(piece)
            for move, skipped in valid_moves.items():
                temp_board = board.copy()
                temp_piece = None
                for p in temp_board.get_all_pieces(color):
                    if p.row == piece.row and p.col == piece.col:
                        temp_piece = p
                        break
                if temp_piece is None:
                    logger.warning(f"Could not find piece at {piece.row},{piece.col} for color {color}")
                    continue
                temp_board.move(temp_piece, move[0], move[1])
                if skipped:
                    temp_board.remove(skipped)
                moves.append({'board': temp_board, 'piece': piece, 'move': move, 'skipped': skipped})
        
        if not moves:
            logger.warning(f"No legal moves found for {color}")
            # Verify that the board state is valid
            black_pieces = len(board.get_all_pieces('black'))
            white_pieces = len(board.get_all_pieces('white'))
            logger.info(f"Board state: {black_pieces} black pieces, {white_pieces} white pieces")
            if black_pieces == 0 or white_pieces == 0:
                logger.info("Terminal state detected: one side has no pieces")
            
        return moves
    except Exception as e:
        logger.error(f"Error in get_all_moves: {e}", exc_info=True)
        return []

def order_moves(board, moves, color, model):
    # Get policy prediction for the current board
    policy_scores_tensor = model.predict_move(board, color)
    policy_scores = policy_scores_tensor.cpu().numpy()
    policy_scores = np.nan_to_num(policy_scores, nan=-float('inf')) # Handle potential NaNs

    scored_moves = []
    opponent = 'black' if color == 'white' else 'white'
    center_row = board.rows / 2
    center_col = board.cols / 2

    for move in moves:
        heuristic_score = 0 # Renamed 'score' to 'heuristic_score'
        piece = move.get('piece')
        dest = move.get('move')
        if piece is None or dest is None:
            continue

        # --- Calculate Heuristic Score (same as before) ---
        if move.get('skipped'):
            for skipped_piece in move.get('skipped'):
                if skipped_piece.king:
                    heuristic_score += 1000
                else:
                    heuristic_score += 500
        # Simplified capture check - heuristic only, not exact
        # if board.get_piece(dest[0], dest[1]) != 0: 
        #     heuristic_score += 1000 # Potential capture (heuristic)
        if color == 'white' and dest[0] == board.rows - 1 and not piece.king:
            heuristic_score += 500 # Kinging move
        elif color == 'black' and dest[0] == 0 and not piece.king:
            heuristic_score += 500 # Kinging move
        center_dist = abs(dest[0] - center_row) + abs(dest[1] - center_col)
        heuristic_score += 50 / (center_dist + 1) # Center control
        board_key = board_hash(board)
        cache_entry = transposition_table.get(board_key)
        # Check if the 'best_move' from cache matches the current move structure
        if cache_entry and cache_entry.get('best_move'):
            cached_move = cache_entry.get('best_move')
            # Compare relevant parts (piece position, destination)
            if cached_move.get('piece') and cached_move.get('move'):
                 if (cached_move['piece'].row == piece.row and
                     cached_move['piece'].col == piece.col and
                     cached_move['move'] == dest):
                    heuristic_score += 2000 # TT Hit Bonus

        # --- Get Policy Score --- 
        policy_idx = dest[0] * board.cols + dest[1]
        policy_score = -float('inf') # Default if index is out of bounds
        if 0 <= policy_idx < len(policy_scores):
             policy_score = policy_scores[policy_idx]

        # Store move with both policy score and heuristic score
        scored_moves.append((move, policy_score, heuristic_score))

    # Sort primarily by policy score (descending), then by heuristic score (descending)
    # Higher scores are better regardless of color in this combined sort
    scored_moves.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return [move for move, policy_score, heuristic_score in scored_moves]
