from typing import Tuple

"""
The constructor of objects of class Element, this function takes 4 integer
values ​​defined as x_min, x_max, y_min, y_max respectively.
These values ​​are used to determine 2 points: top-left point (x_min, y_min) and
bottom-right point (x_max, y_max)
For example:
        (x_min, y_min)──────┐
            │               │
            │               │
            │               │
            └─────────(x_max, y_max)
"""


class Element:
    def __init__(self, coordinate: Tuple[int, int, int, int]):
        """The constructor of objects of class Element, this function takes 4 integer
        values ​​defined as x_min, x_max, y_min, y_max respectively.
        These values ​​are used to determine 2 points: top-left point (x_min, y_min) and
        bottom-right point (x_max, y_max)

        Args:
            coordinate (List[int, int, int, int]): A list that contains 4 integer
                values ​​defined as x_min, x_max, y_min, y_max respectively

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        self.coordinate = coordinate

    @property
    def x_min(self):
        return self.coord[0]

    @x_min.setter
    def x_min(self, value):
        self.coordinate[0] = value

    @property
    def x_max(self):
        return self.coord[1]

    @x_max.setter
    def x_max(self, value):
        self.coordinate[1] = value

    @property
    def y_min(self):
        return self.coordinate[2]

    @y_min.setter
    def y_min(self, value):
        self.coordinate[2] = value

    @property
    def y_max(self):
        return self.coordinate[3]

    @y_max.setter
    def y_max(self, value):
        self.coordinate[3] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.coordinate})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.coordinate})"
