import numpy as np
import time
from itertools import permutations
#from multiprocessing import Pool
import multiprocessing
import os
import logging

# Define the board dimensions
BOARD_HEIGHT = 8
BOARD_WIDTH = 7 

# Configure global logging
def setup_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not logger.hasHandlers():  # Prevent duplicate handlers
        logger.addHandler(handler)

def log_message(message):
    logger = multiprocessing.get_logger()
    logger.debug(message)

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

    def bfs(start_i, start_j):
        queue = [(start_i, start_j)]
        visited[start_i, start_j] = True
        count = 1

        while queue:
            i, j = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < rows and 0 <= nj < cols and not visited[ni, nj] and matrix[ni, nj] == 0:
                    visited[ni, nj] = True
                    queue.append((ni, nj))
                    count += 1

        return count

    min_size = float('inf')
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0 and not visited[i, j]:
                min_size = min(min_size, bfs(i, j))

    return min_size if min_size != float('inf') else 0

def is_valid_arrangement(arrangement):
    """Checks if the puzzle arrangement is valid on the board."""
    board = base_board.copy()
    pid = os.getpid()

    for piece in arrangement:
        placed = False
        for orientation in piece.orientations:
            shape = np.array(orientation)
            shape_height, shape_width = shape.shape

            log_message(f"Trying:\n{shape}")

            if placed:
                log_message(f"skipping shape used")
                break  # Skip other orientations once placed

            for row in range(BOARD_HEIGHT - shape_height + 1):
                for col in range(BOARD_WIDTH - shape_width + 1):
                    sub_board = board[row:row + shape_height, col:col + shape_width]

                    # Fast overlap check
                    if np.any(np.logical_and(sub_board != 0, shape != 0)):
                        continue

                    # Ensure the board doesn't form small isolated empty spaces
                    if min_connected_cells(sub_board) < 5:
                        log_message(f"skipping small space")
                        continue  

                    # Place the piece
                    board[row:row + shape_height, col:col + shape_width] += shape
                    placed = True
                    log_message(f"Current Board:\n{board}")
                    break
 
                if placed:
                   break

        if not placed:  
            return False  # If any piece couldn't be placed, arrangement is invalid

    return np.all(board != 0)  # Return True only if board is fully filled


# Main Execution
#start_time = time.time()
#found = False

# for perm in permutations(pieces):
#    if is_valid_arrangement(perm):
#        print("*** Solution Found ***")
#        print(base_board)
#        found = True
#        break

#if not found:
#    print("No solution found")

#print(f"Execution time: {time.time() - start_time:.4f} seconds")


def check_permutation(perm):
    setup_logger()  # Setup logging for multiprocessing
    return is_valid_arrangement(perm)

if __name__ == "__main__":
    setup_logger()  # Setup logging for multiprocessing

    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:  # Adjust based on CPU cores
        results = pool.map(check_permutation, permutations(pieces))
        
        if any(results):
            print("*** Solution Found ***")
        else:
            print("No solution found")


    print(f"Execution time: {time.time() - start_time:.4f} seconds")
