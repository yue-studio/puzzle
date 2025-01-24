"""
@author: Kelvin Yue

History:
    1. v0.0, Kelvin, 2024-12: original
    2. v1.0, Ham Nguyen, 2025-01-24: add TQDM to show progress

% python ./puzzle.py
Permutation Search:  96%|█████████████████████████████████████████████████████████▉  | 2026480/2100000 [1:39:49<03:49, 319.69 permutations/s]
board is filled
*** Solution Found ***
Permutation Search:  97%|█████████████████████████████████████████████████████████▉  | 2026855/2100000 [1:39:51<03:36, 338.32 permutations/s]
Execution time: 5991.0269 seconds

"""

import numpy as np
import time
from itertools import permutations
from tqdm import tqdm

__VERSION__ = "1.0"

# Define the board dimensions
BOARD_HEIGHT = 7 
BOARD_WIDTH = 7 

class PuzzlePiece:
    def __init__(self, name, shape):
        self.name = name
        self.small_matrix = np.array(shape, dtype=int)  # Use numpy for easy transformations
        self.large_rows = BOARD_HEIGHT
        self.large_cols = BOARD_WIDTH
        self.orientations = self.all_placements_with_orientations()
        self.placed = False

    def place_matrix(self, top_left_row, top_left_col):
        small_rows, small_cols = self.small_matrix.shape
        large_matrix = np.zeros((self.large_rows, self.large_cols), dtype=int)
        large_matrix[top_left_row:top_left_row+small_rows, top_left_col:top_left_col+small_cols] = self.small_matrix
        return large_matrix.tolist()

    def all_placements_for_orientation(self):
        small_rows, small_cols = self.small_matrix.shape
        possible_placements = []

        for i in range(self.large_rows - small_rows + 1):
            for j in range(self.large_cols - small_cols + 1):
                possible_placements.append(self.place_matrix(i, j))

        return possible_placements

    def all_placements_with_orientations(self):
        def add_unique_orientation(matrix, orientation_set):
            """Add unique orientation to the set if not already present."""
            if not any(np.array_equal(matrix, ori) for ori in orientation_set):
               orientation_set.append(matrix)

    # Initialize with the original matrix
        orientations = [self.small_matrix]

    # Generate rotations of the original matrix
        for k in range(1, 4):  # Rotations: 90, 180, 270 degrees
            rotated = np.rot90(self.small_matrix, k)
            add_unique_orientation(rotated, orientations)

    # Generate flips and their rotations
        for flip_func in [np.fliplr, np.flipud]:
            flipped = flip_func(self.small_matrix)
            add_unique_orientation(flipped, orientations)

            for k in range(1, 4):  # Rotations of the flipped matrix
                rotated_flipped = np.rot90(flipped, k)
                add_unique_orientation(rotated_flipped, orientations)

    # Compute placements for each unique orientation
        all_possible_placements = []
        for orientation in orientations:
            self.small_matrix = orientation
            all_possible_placements.extend(self.all_placements_for_orientation())

        return all_possible_placements


# Define the puzzle pieces
#pieces = [
#    PuzzlePiece("Piece1", [[1, 1, 1, 1], [1, 0, 0], [1, 0, 0]]),  # L-shaped piece
#    PuzzlePiece("Piece2", [[2, 2], [2, 2]]),  # Square piece
#    PuzzlePiece("Piece3", [[3, 3, 3]]),  # I-shaped piece
#    PuzzlePiece("Piece4", [[4, 4, 4, 4]]),  # long I-shaped piece
#    PuzzlePiece("Piece5", [[5, 5, 5, 5, 5], [5, 0 ,0, 0]]),  # Long L-shaped piece
#    PuzzlePiece("Piece6", [[6, 6, 6, 6, 6]]),  # long I-shaped piece
#    PuzzlePiece("Piece7", [[7, 7, 7, 7, 7, 7]]),  # long I-shaped piece
#]

# Define the puzzle pieces
pieces = [
    PuzzlePiece("Piece1", [[1, 1, 1, 1], [1, 0, 0, 0]]),  # L-shaped piece
    PuzzlePiece("Piece2", [[2, 2, 0], [0, 2, 2]]),  # Z piece
    PuzzlePiece("Piece3", [[3, 3, 0], [3, 3, 3]]),  # block-shaped piece
    PuzzlePiece("Piece4", [[4, 0, 4], [4, 4, 4]]),  # U-shaped piece
    PuzzlePiece("Piece5", [[5, 5, 5, 5, 5], [5, 0 ,0, 0, 0]]),  # Long L-shaped piece
    PuzzlePiece("Piece6", [[6, 6, 6, 6, 6]]),  # long I-shaped piece
    PuzzlePiece("Piece7", [[7, 7, 7, 7, 7, 7]]),  # long I-shaped piece
    PuzzlePiece("Piece8", [[8, 8, 8, 8, 8, 8]]),  # long I-shaped piece
    PuzzlePiece("Piece9", [[9, 9, 9, 9, 9, 9, 9]]),  # long I-shaped piece
    PuzzlePiece("Piece9", [[10, 10, 10, 10, 10, 10, 10, 10]]),  # long I-shaped piece
]

# Function to check if a piece arrangement is valid
def is_valid_arrangement(arrangement, pbar):
    """
    Check if the arrangement satisfies the puzzle constraints:
    - Pieces fit within the board.
    - No overlaps occur.
    - Target cells align with required values (month, date, day).
    """
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

    for piece_index, piece in enumerate(arrangement):
        piece.placed = False
        for orientation in piece.orientations:
            shape = np.array(orientation)
            shape_height, shape_width = shape.shape

            if piece.placed:
                continue
            # else:
                # print(piece.name, piece.idx)
                # print(repr(shape))

            # Attempt to place the piece on the board
            for row in range(BOARD_HEIGHT - shape_height + 1):
                for col in range(BOARD_WIDTH - shape_width + 1):
                    # Check for overlap
                    sub_board = board[row:row + shape_height, col:col + shape_width]
                    # Create a mask where matrix1 has zeros
                    board_mask = (sub_board != 0)
                    shape_mask = (shape != 0)

                    if np.any(board_mask & shape_mask):
                      continue

                    # Place the piece on the board
                    board[row:row + shape_height, col:col + shape_width] += shape
                    piece.placed = True

                    # print("Board :")
                    # print(repr(board))

    pbar.update(1)

    if np.all(board):
       print("\nboard is filled")
       #print(repr(board))
       return True
    else:
       #print(repr(board))
       return False

start_time = time.time()

found = False

n = int(2.1e6)  # Adjust the number as needed
with tqdm(total=n, desc="Permutation Search", mininterval=5, unit=' permutations') as pbar:
    # Try all permutations of the pieces
    for perm in permutations(pieces):
        if (is_valid_arrangement(perm, pbar)):
           print("*** Solution Found ***")
           found = True
           break

if not found:
       print("No solution found")

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
