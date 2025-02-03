import numpy as np
import argparse
from multiprocessing import Pool, Manager

# Define board dimensions
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

def can_place(board, piece_matrix, row, col):
    """Check if the piece can be placed at (row, col) without overlapping the board mask."""
    piece_height, piece_width = piece_matrix.shape
    
    if row + piece_height > BOARD_HEIGHT or col + piece_width > BOARD_WIDTH:
        return False  # Out of bounds
    
    sub_board = board[row:row + piece_height, col:col + piece_width]
    return not np.any(sub_board & piece_matrix) and not np.any(sub_board == 99) and not np.any(sub_board == 88)  # Ensure no overlap with pieces or mask

def place_piece(board, piece_matrix, row, col, debug):
    """Place a piece on the board."""
    if debug:
        print(f"Placing piece at ({row}, {col})")
    piece_height, piece_width = piece_matrix.shape
    board[row:row + piece_height, col:col + piece_width] += piece_matrix
    if debug:
        print("Board after placement:")
        print(board)

def remove_piece(board, piece_matrix, row, col, debug):
    """Remove a piece from the board."""
    if debug:
        print(f"Removing piece from ({row}, {col})")
    piece_height, piece_width = piece_matrix.shape
    board[row:row + piece_height, col:col + piece_width] -= piece_matrix
    if debug:
        print("Board after removal:")
        print(board)

def solve_puzzle(board, pieces, index=0, debug=False):
    """Backtracking solver to place all pieces on the board."""
    if index == len(pieces):
        print("All pieces placed successfully!")
        print(board)
        return True  # All pieces placed successfully
    
    piece = pieces[index]
    if debug:
        print(f"Trying to place piece {index}...")
    for orientation in piece.orientations:
        piece_matrix = np.array(orientation)
        
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                if can_place(board, piece_matrix, row, col):
                    if debug:
                        print(f"Piece {index} fits at ({row}, {col})")
                    place_piece(board, piece_matrix, row, col, debug)
                    
                    if solve_puzzle(board, pieces, index + 1, debug):
                        return True  # If successful, return immediately
                    
                    remove_piece(board, piece_matrix, row, col, debug)  # Backtrack
    
    if debug:
        print(f"No valid placement found for piece {index}, backtracking...")
    return False  # No valid placement found for this piece

def parallel_solve(args):
    board, pieces, debug, solution_board = args
    if solve_puzzle(board, pieces, debug=debug):
        solution_board[:] = board  # Store the solution board
        return True
    return False

# Initialize board
board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

# Mask out specific board areas efficiently
masked_positions = [(0, 6), (1, 6), (7, 0), (7, 1), (7, 2), (7, 3)]
date_positions = [(0, 0), (5, 1), (7, 4)]

for row, col in masked_positions:
    board[row, col] = 99
for row, col in date_positions:
    board[row, col] = 88

def find_solution_parallel(pieces, debug):
    print("Starting parallel solution search...")
    with Manager() as manager:
        solution_board = manager.list(np.copy(board))  # Shared memory for solution board
        with Pool(processes=4) as pool:  # Adjust based on CPU cores
            results = pool.map(parallel_solve, [(np.copy(board), pieces, debug, solution_board) for _ in range(4)])
            
            if any(results):
                print("Solution Found!")
                print(np.array(solution_board))
            else:
                print("No solution possible.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()
    
    # Solve the puzzle in parallel with optional debug prints
    find_solution_parallel(pieces, args.debug)

