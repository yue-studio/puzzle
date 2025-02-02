import numpy as np
import time
from itertools import permutations

# Define the board dimensions
BOARD_HEIGHT = 8
BOARD_WIDTH = 7

class PuzzlePiece:
    def __init__(self, name, shape):
        self.name = name
        self.base_shape = np.array(shape, dtype=int)  
        self.orientations = self.generate_unique_orientations()

    def generate_unique_orientations(self):
        """Generates all unique rotations and flips of the shape."""
        unique_orientations = set()
        for k in range(4):  # 0°, 90°, 180°, 270°
            rotated = np.rot90(self.base_shape, k)
            unique_orientations.add(tuple(map(tuple, rotated)))  
            flipped_lr = np.fliplr(rotated)
            flipped_ud = np.flipud(rotated)
            unique_orientations.add(tuple(map(tuple, flipped_lr)))
            unique_orientations.add(tuple(map(tuple, flipped_ud)))
        return [np.array(ori) for ori in unique_orientations]

# Define the puzzle pieces
pieces = [
    PuzzlePiece("Piece1", [[2, 2, 0], [0, 2, 2]]),
    PuzzlePiece("Piece2", [[1, 1, 1], [1, 0, 0]]),
    PuzzlePiece("Piece3", [[3, 3, 3, 0], [3, 3, 3, 3]]),
    PuzzlePiece("Piece4", [[4, 0, 4], [4, 4, 4]]),
    PuzzlePiece("Piece5", [[5, 5, 5, 5], [5, 0 ,0, 0]]),
    PuzzlePiece("Piece6", [[6, 6, 6, 6]]),
    PuzzlePiece("Piece7", [[7, 0, 0], [7, 0, 0], [7, 7, 7]]),
    PuzzlePiece("Piece8", [[8, 8, 8], [0, 8, 0], [0, 8, 0]]),
    PuzzlePiece("Piece9", [[9, 9, 9, 0], [0, 0, 9, 9]]),
    PuzzlePiece("Piece10", [[10, 10, 0], [0, 10, 0], [0, 10, 10]]),
]

# Initialize Board with Masked Cells
base_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
base_board[0, 6] = base_board[1, 6] = 99  # Masked out positions
base_board[7, 0:4] = 99  # Another blocked area
base_board[0, 0] = base_board[5, 1] = base_board[7, 4] = 88  # Date/Day cells


def min_connected_cells(matrix):
    """Finds the smallest region of connected zeros (empty spaces)."""
    rows, cols = matrix.shape
    visited = np.zeros((rows, cols), dtype=bool)

    def dfs(i, j):
        stack = [(i, j)]
        count = 0
        while stack:
            x, y = stack.pop()
            if visited[x, y]: continue
            visited[x, y] = True
            count += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and matrix[nx, ny] == 0 and not visited[nx, ny]:
                    stack.append((nx, ny))
        return count

    min_size = float('inf')
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0 and not visited[i, j]:
                min_size = min(min_size, dfs(i, j))

    return min_size if min_size != float('inf') else 0


def is_valid_arrangement(arrangement):
    """Checks if the puzzle arrangement is valid on the board."""
    board = base_board.copy()

    for piece in arrangement:
        placed = False
        for orientation in piece.orientations:
            shape = np.array(orientation)
            shape_height, shape_width = shape.shape

            if placed:
                break  # Skip other orientations once placed

            for row in range(BOARD_HEIGHT - shape_height + 1):
                for col in range(BOARD_WIDTH - shape_width + 1):
                    sub_board = board[row:row + shape_height, col:col + shape_width]

                    # Fast overlap check
                    if np.any(np.logical_and(sub_board != 0, shape != 0)):
                        continue

                    # Ensure the board doesn't form small isolated empty spaces
                    if min_connected_cells(sub_board) < 5:
                        continue  

                    # Place the piece
                    board[row:row + shape_height, col:col + shape_width] += shape
                    placed = True
                    print(board)
                    break

        if not placed:  
            return False  # If any piece couldn't be placed, arrangement is invalid

    return np.all(board != 0)  # Return True only if board is fully filled


# Main Execution
start_time = time.time()
found = False

for perm in permutations(pieces):
    if is_valid_arrangement(perm):
        print("*** Solution Found ***")
        print(base_board)
        found = True
        break

if not found:
    print("No solution found")

print(f"Execution time: {time.time() - start_time:.4f} seconds")

