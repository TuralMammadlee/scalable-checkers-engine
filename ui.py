import tkinter as tk

class UI:
    def __init__(self, board, width=600, height=600, root=None):
        self.board = board
        self.width = width
        self.height = height
        self.square_size = min(width, height) // board.rows
        # Use provided root, or create a new one if not provided.
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
        self.root.title("Checkers AI")
        self.canvas = tk.Canvas(self.root, width=width, height=height)
        self.canvas.pack()
        self.status_label = tk.Label(self.root, text="Welcome", font=("Arial", 14))
        self.status_label.pack()
        self.move_callback = None
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()
        self.draw_pieces()
    
    def set_move_callback(self, callback):
        self.move_callback = callback
    
    def set_status(self, text):
        self.status_label.config(text=text)
    
    def update(self):
        self.canvas.delete("all")
        self.draw_board()
        self.draw_pieces()
        self.root.update()
    
    def draw_board(self):
        # Use default checkers board colors: dark squares as saddle brown and light squares as wheat.
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                x1 = c * self.square_size
                y1 = r * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                if (r + c) % 2 == 0:
                    color = "#8B4513"  # saddle brown
                else:
                    color = "#F5DEB3"  # wheat
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
    
    def draw_pieces(self):
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                piece = self.board.get_piece(r, c)
                if piece != 0:
                    x = c * self.square_size + self.square_size // 2
                    y = r * self.square_size + self.square_size // 2
                    radius = self.square_size // 2 - 5
                    # Use black for black pieces and white for white pieces.
                    fill_color = "black" if piece.color == "black" else "white"
                    self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill_color)
                    if piece.king:
                        font_size = max(12, self.square_size // 2)
                        self.canvas.create_text(x, y, text="♕", font=("Arial", font_size), fill="gold")
    
    def highlight_moves(self, moves):
        for move in moves:
            r, c = move
            x1 = c * self.square_size
            y1 = r * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=3, tag="highlight")
    
    def clear_highlights(self):
        self.canvas.delete("highlight")
    
    def on_click(self, event):
        x, y = event.x, event.y
        col = int(x // self.square_size)
        row = int(y // self.square_size)
        tolerance = 5
        if x % self.square_size < tolerance:
            col = max(0, col)
        if y % self.square_size < tolerance:
            row = max(0, row)
        if self.move_callback:
            self.move_callback(row, col)
