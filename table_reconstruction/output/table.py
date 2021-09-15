from typing import List, Tuple

from .cell import Cell
from .element import Element


class Table(Element):
    """ """

    def __init__(
        self,
        coordinate: Tuple[int, int, int, int],
        col_numb: int,
        row_numb: int,
        cells: List[Cell] = [],
    ):
        """The constructor of objects of class Table, this function accepts column
        number, row number, table coordinates and information of cells defined within it

        Args:
            coordinate (List[int, int, int, int]): A list that contains 4 integer values
                defined as x_min, x_max, y_min, y_max respectively
            col_numb (int): Number of columns detected in the table
            row_numb (int): Number of rows detected in the table
            cells (List[Cell], optional): List of objects of class Cell containing
                information about detected cells in the table. Defaults to [].
        """
        self.coordinate = coordinate
        self.col_numb = col_numb
        self.row_numb = row_numb
        self.cells = cells
