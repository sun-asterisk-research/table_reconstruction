from typing import Tuple

from .element import Element


class Cell(Element):
    """ """

    def __init__(
        self,
        coordinate: Tuple[int, int, int, int],
        col_index: int,
        row_index: int,
        col_span: int,
        row_span: int,
    ):
        """The constructor of objects of class Table, this function accepts column
        number, row number, table coordinates and information of cells defined within it
        To handle the cases where cells are merged together, the cells to be extracted
        are defined as connected regions surrounded by straight lines.

        Args:
            coordinate (List[int, int, int, int]): A list that contains 4 integer values
                ​​defined as x_min, x_max, y_min, y_max respectively
            col_index (int): The ordinal number of the column in which this cell starts
            row_index (int): The ordinal number of the row in which this cell starts
            col_numb (int): The number of units that the cell spans in cols
            row_numb (int): The number of units that the cell spans in rows
        """
        self.coordinate = coordinate
        self.row_index = row_index
        self.col_index = col_index
        self.col_span = col_span
        self.row_span = row_span
