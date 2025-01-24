# -*- coding: utf-8 -*-
"""
@author: Kelvin Yue

History:
    1. v0.0, Kelvin, 2024-12: original
    2. v1.0, Ham Nguyen, 2025-01-23: add TQDM to show progress

$ python ./dfs.py
Depth-First Search: 100%|████████████████████████████████████████████████████████████▋| 120490232/121000000 [5:21:43<01:21, 6241.91 trials/s]
Execution time: 19303.4391 seconds

"""

import numpy as np
import time
from tqdm import tqdm

__VERSION__ = "1.0"

# Define the board dimensions
BOARD_HEIGHT = 7
BOARD_WIDTH = 6

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

def dfs(board, pieces, piece_index, pbar):
    """Perform DFS to find a valid arrangement."""
    if piece_index == len(pieces):
        # All pieces placed successfully
        return True

    piece = pieces[piece_index]

    # print(board)

    for orientation in piece.orientations:
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                if piece.place_on_board(board, orientation, row, col):
                    pbar.update(1)
                    # Recursive step
                    if dfs(board, pieces, piece_index + 1, pbar):
                        print(board)
                        return True

                    # Backtrack
                    piece.remove_from_board(board, orientation, row, col)

    return False

# Record the start time
start_time = time.time()


# Initialize the board
board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

n = int(1.21e8)  # Adjust the number as needed
with tqdm(total=n, desc="Depth-First Search", mininterval=10, unit=' trials') as pbar:
    # Start DFS
    if dfs(board, pieces, 0, pbar):
        print("*** Solution Found ***")
        print(board)
    else:
        print("No solution found")

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
