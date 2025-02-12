import numpy as np
import time
from collections import deque

# Define the board dimensions
BOARD_HEIGHT = 7
BOARD_WIDTH =7 

class PuzzlePiece:
    def __init__(self, name, shape):
        self.name = name
        self.small_matrix = np.array(shape, dtype=int)  # Use numpy for easy transformations
        self.orientations = self.generate_orientations()

    def generate_orientations(self):
        """Generate all unique orientations of the piece (rotations and flips)."""
        orientations = []

        def add_unique_orientation(matrix):
            if not any(np.array_equal(matrix, ori) for ori in orientations):
                orientations.append(matrix)

        # Original and rotations
        for k in range(4):
            rotated = np.rot90(self.small_matrix, k)
            add_unique_orientation(rotated)

        # Flips and their rotations
        for flip_func in [np.fliplr, np.flipud]:
            flipped = flip_func(self.small_matrix)
            add_unique_orientation(flipped)
            for k in range(1, 4):
                rotated_flipped = np.rot90(flipped, k)
                add_unique_orientation(rotated_flipped)

        return orientations

    def place_on_board(self, board, orientation, top_left_row, top_left_col):
        """Attempt to place the piece on the board at the specified position."""
        shape = np.array(orientation)
        shape_height, shape_width = shape.shape

        if top_left_row + shape_height > BOARD_HEIGHT or top_left_col + shape_width > BOARD_WIDTH:
            return False  # Out of bounds

        # Check for overlap
        sub_board = board[top_left_row:top_left_row + shape_height, top_left_col:top_left_col + shape_width]
        # Create a mask where matrix1 has zeros
        board_mask = (sub_board != 0)
        shape_mask = (shape != 0)

        if np.any(board_mask & shape_mask):
            return False  # Overlap detected

        # Place the piece
        board[top_left_row:top_left_row + shape_height, top_left_col:top_left_col + shape_width] += shape
        return True

    def remove_from_board(self, board, orientation, top_left_row, top_left_col):
        """Remove the piece from the board."""
        shape = np.array(orientation)
        shape_height, shape_width = shape.shape
        board[top_left_row:top_left_row + shape_height, top_left_col:top_left_col + shape_width] -= shape

# Define the puzzle pieces
pieces = [
    PuzzlePiece("Piece1", [[1, 1, 1, 1], [1, 0, 0, 0]]),
    PuzzlePiece("Piece2", [[2, 2, 0], [0, 2, 2]]),
    PuzzlePiece("Piece3", [[3, 3, 0], [3, 3, 3]]),
    PuzzlePiece("Piece4", [[4, 0, 4], [4, 4, 4]]),
    PuzzlePiece("Piece5", [[5, 5, 5, 5, 5], [5, 0, 0, 0, 0]]),
    PuzzlePiece("Piece6", [[6, 6, 6, 6, 6]]),
    PuzzlePiece("Piece7", [[7, 7, 7, 7, 7, 7]]),
    PuzzlePiece("Piece8", [[8, 8, 8, 8, 8, 8]]),
    PuzzlePiece("Piece9", [[9, 9, 9, 9, 9, 9, 9]]),
]

def bfs(pieces):
    """Perform BFS to find a valid arrangement."""
    initial_state = (np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int), 0)  # (board, piece_index)
    queue = deque([initial_state])

    while queue:
        board, piece_index = queue.popleft()

        if piece_index == len(pieces):
            # All pieces placed successfully
            return board

        piece = pieces[piece_index]

        for orientation in piece.orientations:
            for row in range(BOARD_HEIGHT):
                for col in range(BOARD_WIDTH):
                    new_board = board.copy()
                    if piece.place_on_board(new_board, orientation, row, col):
                        queue.append((new_board, piece_index + 1))

    return None

# Record the start time
start_time = time.time()


# Start BFS
solution = bfs(pieces)
if solution is not None:
    print("*** Solution Found ***")
    print(solution)
else:
    print("No solution found")

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
