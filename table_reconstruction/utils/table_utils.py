from typing import Dict, List, Tuple

import numpy as np


class DirectedGraph(object):
    def __init__(self, nb_vertices: int):
        """Initialize DirectedGraph class

        Args:
            nb_vertices (int): the number of vertices
        """
        self.nb_vertices: int = nb_vertices
        self.adj: List[List] = [[] for i in range(nb_vertices)]
        self.undirected_vertex: List[List] = [[] for i in range(nb_vertices)]
        self.span_cell_ids: List = []

    def add_edge(self, id_a: int, id_b: int):
        """Add edge to graph

        Args:
            id_a (int): Index of cell a
            id_b (int): Index of cell b
        """
        self.adj[id_a].append(id_b)
        self.undirected_vertex[id_a].append(id_b)
        self.undirected_vertex[id_b].append(id_a)

        if self.check_span_cell(id_a):
            self.span_cell_ids.append(id_a)

        if self.check_span_cell(id_b):
            self.span_cell_ids.append(id_b)

        if len(self.adj[id_a]) > 1 and id_a not in self.span_cell_ids:
            self.span_cell_ids.append(id_a)

    def add_edges(self, couple_ids: List):
        """Add index of vertices (cell) to graph

        Args:
            couple_ids (List): the list index of couple vertice indexes
        """
        for couple_id in couple_ids:
            id_a, id_b = couple_id
            self.add_edge(id_a, id_b)

        for span_cell_id in self.span_cell_ids:
            if len(self.adj[span_cell_id]) == 1:
                self.span_cell_ids.append(self.adj[span_cell_id][0])

    def check_span_cell(self, vertex_id: int) -> bool:
        """Check whether cell is span cell or not

        Args:
            vertex_id (int): the index of cell

        Returns:
            Bool: if cell is span cell, return True, otherwise return False
        """
        if len(self.undirected_vertex[vertex_id]) > 2:
            if vertex_id not in self.span_cell_ids:
                return True

        return False

    def dfs(self, v: int, dp: List, vis: List):
        """Use Depth First Search algorithm to search longest path from specific vertex.

        Args:
            v (int): the index of specific cell
            dp (List): distances to all vertices
            vis (List): visited status
        """
        vis[v] = True

        # traverse for all its children
        for i in range(len(self.adj[v])):
            # if not visited:
            if not vis[self.adj[v][i]]:
                self.dfs(self.adj[v][i], dp, vis)

            # store the max of the paths
            dp[v] = max(dp[v], 1 + dp[self.adj[v][i]])

    def findLongestPath(self) -> int:
        """Function that returns the longest path in graph

        Returns:
            int: the maximum distance
        """
        # Dp array: dp[i] be the length of the longest path starting from the node i.
        dp = [0] * self.nb_vertices

        # Visited array to know if the node
        # has been visited previously or not
        vis = [False] * (self.nb_vertices)

        # Call DFS for every unvisited vertex
        for i in range(self.nb_vertices):
            if not vis[i] and len(self.adj[i]) > 0:
                self.dfs(i, dp, vis)

        return max(dp) + 1


def convertId2DocxCoord(cell_id: int, nb_col: int) -> Tuple[int, int]:
    """Find the XY coordinate of a know point

    Args:
        cell_id (int): the index of cell
        nb_col (int): the number columns of table

    Returns:
        tuple: the XY coordinate corresponding to the index of cell
    """
    x = cell_id % nb_col
    y = cell_id // nb_col

    return x, y


def convertSpanCell2DocxCoord(
    cells: List[List],
    fake_cells: List[List],
    span_cell_ids: List,
    nb_col: int,
    thresh: int = 5,
) -> List[Dict]:
    """Find the XY coordinate of span cells

    Args:
        cells (List[List]): the coordinate of cells
        fake_cells (List[List]): the coordinate of fake cells
        span_cell_ids (List): the index of span cells
        nb_col (int): number columns in table
        thresh (int, optional): threshold value to group the same x, y coordinate.

    Returns:
        List[Dict]: the XY coordinate of span cells
    """
    fake_x1_coords = np.array(fake_cells)[:, 0]
    fake_x2_coords = np.array(fake_cells)[:, 1]
    fake_y1_coords = np.array(fake_cells)[:, 2]
    fake_y2_coords = np.array(fake_cells)[:, 3]
    id_range = np.arange(len(fake_cells))
    docx_coords = []

    for idx in span_cell_ids:
        docx_coor = dict()
        x1, x2, y1, y2 = cells[idx]

        x_mask = (x1 - thresh <= fake_x1_coords) & (fake_x2_coords <= x2 + thresh)
        y_mask = (y1 - thresh <= fake_y1_coords) & (fake_y2_coords <= y2 + thresh)
        coord_mask = x_mask & y_mask

        filter_ids = id_range[coord_mask]
        x_start_idx = 10000
        x_end_idx = -1
        y_start_idx = 10000
        y_end_idx = -1

        for cell_id in filter_ids:
            x, y = convertId2DocxCoord(cell_id, nb_col)
            x_start_idx = min(x_start_idx, x)
            x_end_idx = max(x_end_idx, x)
            y_start_idx = min(y_start_idx, y)
            y_end_idx = max(y_end_idx, y)

        docx_coor["id"] = idx
        docx_coor["x"] = [x_start_idx, x_end_idx]
        docx_coor["y"] = [y_start_idx, y_end_idx]
        docx_coords.append(docx_coor)

    return docx_coords
