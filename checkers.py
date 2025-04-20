import copy

class Piece:
    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color  # 'black' or 'white'
        self.king = False

    def make_king(self):
        self.king = True

    def __repr__(self):
        return f"{self.color[0].upper()}{'K' if self.king else 'P'}"

class Board:
    def __init__(self, rows=8, cols=8):
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.black_left = 0
        self.white_left = 0
        self.black_kings = 0
        self.white_kings = 0
        self.create_board()
    
    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def create_board(self):
        # Compute the number of rows to fill based on board size.
        rows_to_fill = (self.rows - 2) // 2
        for row in range(self.rows):
            for col in range(self.cols):
                if (row + col) % 2 != 0:
                    if row < rows_to_fill:
                        self.board[row][col] = Piece(row, col, 'white')
                        self.white_left += 1
                    elif row >= self.rows - rows_to_fill:
                        self.board[row][col] = Piece(row, col, 'black')
                        self.black_left += 1

    def move(self, piece, row, col):
        self.board[piece.row][piece.col] = 0
        piece.row = row
        piece.col = col
        self.board[row][col] = piece
        # Crown kings upon reaching the opposite end.
        if row == 0 and piece.color == 'black' and not piece.king:
            piece.make_king()
            self.black_kings += 1
        elif row == self.rows - 1 and piece.color == 'white' and not piece.king:
            piece.make_king()
            self.white_kings += 1

    def get_piece(self, row, col):
        if self.in_bounds(row, col):
            return self.board[row][col]
        return None
    
    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if piece:
                if piece.color == 'black':
                    self.black_left -= 1
                    if piece.king:
                        self.black_kings -= 1
                else:
                    self.white_left -= 1
                    if piece.king:
                        self.white_kings -= 1

    def winner(self):
        if self.black_left <= 0:
            return 'white'
        elif self.white_left <= 0:
            return 'black'
        return None

    def get_all_pieces(self, color):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces

    def get_valid_moves(self, piece):
        try:
            # First, check if any piece of this color can capture.
            color = piece.color
            any_captures_exist = False
            for p in self.get_all_pieces(color):
                test_captures = {}
                self._get_captures(p, p.row, p.col, [], test_captures)
                if test_captures:
                    any_captures_exist = True
                    break

            # Then compute moves for this piece.
            captures = {}
            self._get_captures(piece, piece.row, piece.col, [], captures)
            if captures:
                return captures

            if any_captures_exist:
                return {}  # This piece can't move if another capture exists.

            moves = {}
            directions = []
            # Fix: White pieces move down (+1) and black pieces move up (-1)
            if piece.color == 'white' or piece.king:
                directions.append(1)
            if piece.color == 'black' or piece.king:
                directions.append(-1)
            
            for d in directions:
                for col_dir in [-1, 1]:
                    new_row = piece.row + d
                    new_col = piece.col + col_dir
                    if self.in_bounds(new_row, new_col) and self.get_piece(new_row, new_col) == 0:
                        moves[(new_row, new_col)] = []
            return moves
        except Exception as e:
            print("Error in get_valid_moves:", e)
            return {}

    def _get_captures(self, piece, row, col, current_skipped, captures, visited=None):
        if visited is None:
            visited = set() # Use set for faster lookups

        # Prevent revisiting the *same square* within the *same jump sequence*
        current_path_key = (row, col)
        if current_path_key in visited:
             return captures # Already visited this square in this path

        visited.add(current_path_key)

        directions = []
        if piece.color == 'white' or piece.king:
            directions.append(1)
        if piece.color == 'black' or piece.king:
            directions.append(-1)

        found_further_jump = False # Track if any recursive call found moves

        for d in directions:
            for col_dir in [-1, 1]:
                mid_row, mid_col = row + d, col + col_dir
                end_row, end_col = row + 2 * d, col + 2 * col_dir

                if self.in_bounds(mid_row, mid_col) and self.in_bounds(end_row, end_col):
                    mid_piece = self.get_piece(mid_row, mid_col)
                    end_cell = self.get_piece(end_row, end_col)

                    # Check if this jump is valid
                    if mid_piece != 0 and mid_piece.color != piece.color and \
                       mid_piece not in current_skipped and end_cell == 0:

                        new_skipped = current_skipped + [mid_piece]
                        new_visited = visited.copy() # Pass a copy down recursion

                        # Recursively explore further jumps from the landing square
                        # Note: We pass the *original* captures dict to accumulate results
                        self._get_captures(piece, end_row, end_col, new_skipped, captures, new_visited)
                        
                        #  mark that we initiated a recursive search from this path.
                        # The actual results (final landing spots and skips) will be added 
                        # to the 'captures' dict by the deepest successful calls in the recursion.
                        found_further_jump = True


        
        if not found_further_jump and current_skipped: # Must have skipped at least one piece to be a capture end
             final_pos = (row, col) # The current (row, col) is the final landing spot
             # Update if this path is longer than an existing one to the same square
             if final_pos not in captures or len(current_skipped) > len(captures[final_pos]):
                 captures[final_pos] = current_skipped

        return captures

    def copy(self):
        return copy.deepcopy(self)
        
    def index_to_coord(self, index):
        row = index // self.cols
        col = index % self.cols
        return (row, col)
        
    def coord_to_index(self, row, col):
        return row * self.cols + col
