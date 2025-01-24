import numpy as np
import time
from itertools import permutations

# Define the board dimensions
BOARD_HEIGHT = 8
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
pieces = [
    PuzzlePiece("Piece1", [[1, 1, 1], [1, 0, 0]]),  # L-shaped piece
    PuzzlePiece("Piece2", [[2, 2, 0], [0, 2, 2]]),  # Z piece
    PuzzlePiece("Piece3", [[3, 3, 3, 0], [3, 3, 3, 3]]),  # block-shaped piece
    PuzzlePiece("Piece4", [[4, 0, 4], [4, 4, 4]]),  # U-shaped piece
    PuzzlePiece("Piece5", [[5, 5, 5, 5], [5, 0 ,0, 0]]),  # Long L-shaped piece
    PuzzlePiece("Piece6", [[6, 6, 6, 6]]),  # long I-shaped piece
    PuzzlePiece("Piece7", [[7, 0, 0], [7, 0, 0], [7, 7, 7]]),  # long L-shaped piece 2
    PuzzlePiece("Piece8", [[8, 8, 8], [0, 8, 0], [0, 8, 0]]),  # T-shaped piece
    PuzzlePiece("Piece9", [[9, 9, 9, 0], [0, 0, 9, 9]]),  # long Z-shaped piece
    PuzzlePiece("Piece10", [[10, 10, 0], [0, 10, 0], [0, 10, 10]]),  # big Z-shaped piece
]

# function to find the smallest connected cells
def min_connected_cells(matrix):
    rows, cols = matrix.shape
    visited = np.zeros((rows, cols), dtype=bool)

    def is_valid(i, j):
        return 0 <= i < rows and 0 <= j < cols and not visited[i, j] and matrix[i, j] == 0

    def dfs(i, j):
        stack = [(i, j)]
        connected_cells = 0
        while stack:
            x, y = stack.pop()
            if not visited[x, y]:
                visited[x, y] = True
                connected_cells += 1
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if is_valid(x + dx, y + dy):
                        stack.append((x + dx, y + dy))
        return connected_cells

    min_connected_cells = float('inf')
    found = False
    for i in range(rows):
        for j in range(cols):
            if is_valid(i, j):
                size = dfs(i, j)
                min_connected_cells = min(min_connected_cells, size)
                found = True

    return min_connected_cells if found else 0

# Function to check if a piece arrangement is valid
def is_valid_arrangement(arrangement):
    """
    Check if the arrangement satisfies the puzzle constraints:
    - Pieces fit within the board.
    - No overlaps occur.
    - Target cells align with required values (month, date, day).
    """
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
    
    # mask out based on the board shap
    
    board[0, 6] = 99
    board[1, 6] = 99
    board[7, 0] = 99
    board[7, 1] = 99
    board[7, 2] = 99
    board[7, 3] = 99
    
    # mask out the date (1/23/2025 Thurs)
    
    board[0, 0] = 88
    board[5, 1] = 88
    board[7, 4] = 88
    

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
                    # Create a mask where matrix has zeros
                    board_mask = (sub_board != 0)
                    shape_mask = (shape != 0)

                    #print("mask: ")
                    #print(board_mask)
                    #print(shape_mask)

                    if np.any(board_mask & shape_mask):
                      # print("skipping")
                      continue

                    min_size = min_connected_cells(sub_board)

                    if (min_size < 5):
                        # print("skipping")
                        continue

                    # Place the piece on the board
                    board[row:row + shape_height, col:col + shape_width] += shape
                    piece.placed = True

                    # print("Board :")
                    # print(repr(board))

    if np.all(board):
       print("board is filled")
       print(repr(board))
       return True
    else:
       # print(repr(board))
       return False

start_time = time.time()

found = False 
# Try all permutations of the pieces
for perm in permutations(pieces):
    if (is_valid_arrangement(perm)):
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
