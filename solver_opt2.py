#!/usr/bin/env python3
"""
Optimized Calendar Puzzle Solver using Donald Knuth's Algorithm X (DLX).
"""

import logging
import time
import argparse
import datetime
from typing import List, Tuple, Dict, Any, Iterator, Set
from dataclasses import dataclass
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate, scale
from tqdm import tqdm
from visualizer import plot_solution

# --- Data Structures ---

@dataclass(frozen=True)
class Position:
    """Immutable position representation"""
    x: float
    y: float
    rotation: int
    mirrored: bool = False

@dataclass
class PartDefinition:
    """Immutable part definition"""
    name: str
    polygon: Polygon
    color: str
    can_mirror: bool

    def __post_init__(self):
        self._rotations = self._compute_distinct_rotations()

    def _compute_distinct_rotations(self) -> List[int]:
        rotations, polys = [0], [self.polygon]
        for r in range(90, 360, 90):
            test = rotate(self.polygon, r)
            if not any(test.equals(p) for p in polys):
                rotations.append(r)
                polys.append(test)
        return rotations

    @property
    def distinct_rotations(self) -> List[int]:
        return self._rotations

    def get_polygon_at_position(self, pos: Position) -> Polygon:
        p = self.polygon
        if pos.mirrored: p = scale(p, -1, 1)
        if pos.rotation != 0: p = rotate(p, pos.rotation)
        if pos.x != 0 or pos.y != 0: p = translate(p, pos.x, pos.y)
        return p

# --- DLX (Algorithm X) Implementation ---

class DLX:
    """Implementation of Donald Knuth's Algorithm X with Dancing Links."""

    @dataclass(eq=False)
    class Node:
        L: 'DLX.Node'; R: 'DLX.Node'; U: 'DLX.Node'; D: 'DLX.Node'
        C: 'DLX.Column'
        row_data: Any = None

    @dataclass(eq=False)
    class Column(Node):
        size: int = 0
        name: str = ""

    def __init__(self, columns: set, rows: Dict[Any, List[Any]], max_solutions: int):
        self.header = self.Column(L=None, R=None, U=None, D=None, C=None)
        self.header.L = self.header.R = self.header
        self.solutions = []
        self.max_solutions = max_solutions
        self._build_links(columns, rows)

    def _build_links(self, columns: set, rows: Dict[Any, List[Any]]):
        column_nodes = {}
        for col_name in sorted(list(columns), key=str):
            new_col = self.Column(L=self.header.L, R=self.header, U=None, D=None, C=None, name=col_name)
            new_col.U = new_col.D = new_col
            self.header.L.R = new_col
            self.header.L = new_col
            column_nodes[col_name] = new_col

        for row_data, row_cols in rows.items():
            first_node = None
            for col_name in row_cols:
                if col_name not in column_nodes: continue
                col = column_nodes[col_name]
                col.size += 1
                new_node = self.Node(L=None, R=None, U=col.U, D=col, C=col, row_data=row_data)
                col.U.D = new_node
                col.U = new_node
                if first_node is None:
                    first_node = new_node
                    first_node.L = first_node.R = first_node
                else:
                    new_node.L = first_node.L
                    new_node.R = first_node
                    first_node.L.R = new_node
                    first_node.L = new_node

    def solve(self) -> Iterator[List[Any]]:
        solution_stack = []
        self._search(solution_stack)
        return self.solutions

    def _search(self, solution_stack):
        if self.header.R is self.header:
            self.solutions.append(list(solution_stack))
            return len(self.solutions) >= self.max_solutions

        c = self._choose_column()
        self._cover(c)

        r = c.D
        while r is not c:
            solution_stack.append(r.row_data)
            j = r.R
            while j is not r:
                self._cover(j.C)
                j = j.R
            
            if self._search(solution_stack):
                return True

            solution_stack.pop()
            j = r.L
            while j is not r:
                self._uncover(j.C)
                j = j.L
            r = r.D

        self._uncover(c)
        return False

    def _choose_column(self) -> Column:
        s, c = float('inf'), None
        j = self.header.R
        while j is not self.header:
            if j.size < s:
                s, c = j.size, j
            j = j.R
        return c

    def _cover(self, c: Column):
        c.R.L, c.L.R = c.L, c.R
        i = c.D
        while i is not c:
            j = i.R
            while j is not i:
                j.D.U, j.U.D = j.U, j.D
                j.C.size -= 1
                j = j.R
            i = i.D

    def _uncover(self, c: Column):
        i = c.U
        while i is not c:
            j = i.L
            while j is not i:
                j.C.size += 1
                j.D.U, j.U.D = j, j
                j = j.L
            i = i.U
        c.R.L, c.L.R = c, c

# --- Main Solver Class ---

class OptimizedSolver:
    """Puzzle solver using DLX for the exact cover problem."""

    def __init__(self, parts: List[PartDefinition], debug: bool = False):
        self.parts = parts
        self.part_map = {p.name: p for p in parts}
        self.debug = debug
        self.logger = logging.getLogger('OptimizedSolver')
        if self.debug:
            self.logger.info(f"Initialized solver with {len(self.parts)} parts.")

    def solve(self, target: Polygon, max_solutions: int = 1, show_progress: bool = True) -> List[Dict]:
        self.start_time = time.time()
        
        if self.debug:
            self.logger.info(f"Building exact cover matrix for target area {target.area:.2f}...")

        all_cols, placements = self._build_exact_cover_matrix(target, show_progress)
        
        if not placements:
            self.logger.info("No valid placements found. No solutions possible.")
            return []

        if self.debug:
            self.logger.info(f"Matrix built in {time.time() - self.start_time:.2f}s. "
                           f"{len(placements)} rows (placements), {len(all_cols)} columns (constraints).")
            self.logger.info("Starting DLX solver...")

        dlx_solver = DLX(all_cols, placements, max_solutions)
        raw_solutions = dlx_solver.solve()

        solutions = self._format_solutions(raw_solutions)

        end_time = time.time()
        self.logger.info(f"Solved in {end_time - self.start_time:.2f}s. Found {len(solutions)} solutions.")
        return solutions

    def _build_exact_cover_matrix(self, target: Polygon, show_progress: bool) -> Tuple[Dict, Dict]:
        bounds = target.bounds
        min_x, min_y, max_x, max_y = map(round, bounds)

        # Primary columns: one for each part
        primary_cols = {part.name for part in self.parts}
        
        # Secondary columns: one for each grid cell in the target
        secondary_cols = set()
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if target.contains(Point(x + 0.5, y + 0.5)):
                    secondary_cols.add((x, y))

        all_cols = primary_cols.union(secondary_cols)
        placements = {}

        # Create a generator for all possible placements to use with tqdm
        placement_generator = self._generate_all_placements(target, secondary_cols)
        
        # Estimate total placements for progress bar
        total_placements = sum(len(p.distinct_rotations) * (2 if p.can_mirror else 1) for p in self.parts) * len(secondary_cols)

        pbar = tqdm(placement_generator, total=total_placements, desc="Building Matrix", disable=not show_progress)

        for part, pos, poly in pbar:
            cells = self._get_covered_cells(poly, secondary_cols)
            if cells is not None:
                row_key = (part.name, pos)
                placements[row_key] = [part.name] + cells
        
        return all_cols, placements

    def _generate_all_placements(self, target: Polygon, grid_cells: Set[Tuple[int, int]]) -> Iterator[Tuple[PartDefinition, Position, Polygon]]:
        for part in self.parts:
            for mirrored in ([False, True] if part.can_mirror else [False]):
                base_poly = scale(part.polygon, -1, 1) if mirrored else part.polygon
                for rotation in part.distinct_rotations:
                    rotated_poly = rotate(base_poly, rotation)
                    for x, y in grid_cells:
                        pos = Position(x, y, rotation, mirrored)
                        # This is a simplified translation check; the real check is containment
                        test_poly = translate(rotated_poly, x, y)
                        if target.contains(test_poly.centroid):
                            yield part, pos, test_poly

    def _get_covered_cells(self, poly: Polygon, grid_cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        covered = []
        for x_cell, y_cell in grid_cells:
            cell_center = Point(x_cell + 0.5, y_cell + 0.5)
            if poly.contains(cell_center):
                covered.append((x_cell, y_cell))
        
        # Ensure the area matches the number of cells covered
        if abs(poly.area - len(covered)) > 1e-9:
            return None # Invalid placement that doesn't align with the grid
            
        return covered

    def _format_solutions(self, raw_solutions: List[List[Tuple[str, Position]]]) -> List[Dict[str, Position]]:
        solutions = []
        for raw_solution in raw_solutions:
            solution_map = {}
            for part_name, position in raw_solution:
                solution_map[part_name] = position
            solutions.append(solution_map)
        return solutions

# --- Puzzle Definition and Execution ---

def create_calendar_target(weekday: int, day: int, month: int) -> Polygon:
    def square(x, y): return Polygon([(x,y), (x+1,y), (x+1,y+1), (x,y+1)])
    outer = Polygon([(0,0), (4,0), (4,-1), (7,-1), (7,5), (6,5), (6,7), (0,7)])
    mx, my = (month - 1) % 6, 6 - (month - 1) // 6
    dx, dy = (day - 1) % 7, 4 - (day - 1) // 7
    w_pos = {0:(4,0), 1:(5,0), 2:(6,0), 3:(4,-1), 4:(5,-1), 5:(6,-1), 6:(3,0)}
    wx, wy = w_pos[weekday]
    return outer.difference(square(mx,my)).difference(square(dx,dy)).difference(square(wx,wy))

def create_puzzle_parts() -> List[PartDefinition]:
    return [
        PartDefinition('A', Polygon([(0,0),(4,0),(4,1),(0,1)]), 'red', False),
        PartDefinition('B', Polygon([(0,0),(4,0),(4,2),(3,2),(3,1),(0,1)]), 'green', True),
        PartDefinition('C', Polygon([(0,0),(3,0),(3,2),(2,2),(2,1),(0,1)]), 'blue', True),
        PartDefinition('D', Polygon([(0,0),(3,0),(3,1),(2,1),(2,3),(1,3),(1,1),(0,1)]), 'purple', True),
        PartDefinition('E', Polygon([(0,0),(2,0),(2,2),(3,2),(3,3),(1,3),(1,1),(0,1)]), 'yellow', True),
        PartDefinition('F', Polygon([(0,0),(2,0),(2,1),(3,1),(3,2),(0,2)]), 'orange', True),
        PartDefinition('G', Polygon([(0,0),(3,0),(3,3),(2,3),(2,1),(0,1)]), 'pink', False),
        PartDefinition('H', Polygon([(0,0),(3,0),(3,2),(2,2),(2,1),(1,1),(1,2),(0,2)]), 'cyan', False),
        PartDefinition('I', Polygon([(0,0),(3,0),(3,1),(4,1),(4,2),(2,2),(2,1),(0,1)]), 'brown', True),
        PartDefinition('J', Polygon([(0,0),(2,0),(2,1),(3,1),(3,2),(1,2),(1,1),(0,1)]), 'gray', True)
    ]

def solve_calendar_puzzle(weekday: int, day: int, month: int, max_solutions: int = 1, 
                         show_progress: bool = True, debug: bool = False) -> List[Dict]:
    if debug: logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
    
    target = create_calendar_target(weekday, day, month)
    parts = create_puzzle_parts()
    solver = OptimizedSolver(parts, debug=debug)
    return solver.solve(target, max_solutions, show_progress)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Solve Calendar Puzzle using Algorithm X (DLX).')
    parser.add_argument('--weekday', type=int, default=0, choices=range(7), help='Weekday (0=Mon..6=Sun)')
    parser.add_argument('--day', type=int, default=1, choices=range(1, 32), help='Day of month (1-31)')
    parser.add_argument('--month', type=int, default=1, choices=range(1, 13), help='Month (1-12)')
    parser.add_argument('--max-solutions', type=int, default=1, help='Max solutions to find')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--plot', action='store_true', help='Plot the first solution found')
    parser.add_argument('--today', action='store_true', help="Use today's date as the target")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.today:
        today = datetime.date.today()
        args.weekday = today.weekday()
        args.day = today.day
        args.month = today.month
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"Solving for: {weekday_names[args.weekday]}, {month_names[args.month-1]} {args.day}")
    
    start_time = time.time()
    try:
        solutions = solve_calendar_puzzle(
            weekday=args.weekday, day=args.day, month=args.month,
            max_solutions=args.max_solutions, show_progress=not args.no_progress, debug=args.debug
        )
        duration = time.time() - start_time
        if solutions:
            print(f"\n✅ Found {len(solutions)} solution(s) in {duration:.2f} seconds!")
            for i, sol in enumerate(solutions):
                print(f"\nSolution {i+1}:")
                for name, pos in sorted(sol.items()):
                    print(f"  {name}: x={pos.x:.0f}, y={pos.y:.0f}, rot={pos.rotation}°, mirror={pos.mirrored}")
            
            if args.plot:
                print("\n🎨 Plotting first solution...")
                plot_solution(args.weekday, args.day, args.month, solutions[0])

        else:
            print(f"\n❌ No solutions found in {duration:.2f} seconds.")
    except KeyboardInterrupt:
        print(f"\n⏹️ Interrupted after {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug: import traceback; traceback.print_exc()