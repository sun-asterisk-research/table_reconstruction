from typing import List, Tuple, Union

import numpy as np
from shapely.geometry import LineString

from .lines_utils import is_line


def get_intersection_points(
    horizontal_lines: Union[List[List], np.ndarray],
    vertical_lines: Union[List[List], np.ndarray],
    tab_coord: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """This a function which find the coordinate (x, y) of intersection points

    Args:
        horizontal_lines (List[List]): The coordinate of horizontal lines
        vertical_lines (List[List]): The coordinate of vertical lines
        tab_coord (List): The coordinate of table

    Returns:
        Tuple[np.ndarray, np.ndarray]: intersection and fake intersection points
    """

    intersect_points = []
    fake_intersect_points = []
    tab_x1, tab_y1, tab_x2, tab_y2 = tab_coord

    for h_coor in horizontal_lines:
        h_x1, h_y1, h_x2, h_y2 = h_coor
        fake_h_x1 = min(h_x1, tab_x1)
        fake_h_x2 = max(h_x2, tab_x2)

        h_line = LineString([(h_x1, h_y1), (h_x2, h_y2)])
        fake_h_line = LineString([(fake_h_x1, h_y1), (fake_h_x2, h_y2)])

        for v_coor in vertical_lines:
            v_x1, v_y1, v_x2, v_y2 = v_coor
            fake_v_y1 = min(v_y1, tab_y1)
            fake_v_y2 = max(v_y2, tab_y2)

            fake_v_line = LineString([(v_x1, fake_v_y1), (v_x2, fake_v_y2)])
            v_line = LineString([(v_x1, v_y1), (v_x2, v_y2)])

            intersect_point = h_line.intersection(v_line)
            fake_intersect_point = fake_h_line.intersection(fake_v_line)

            if len(list(intersect_point.coords)) != 0:
                intersect_points.append(list(intersect_point.coords))

            if len(list(fake_intersect_point.coords)) != 0:
                fake_intersect_points.append(list(fake_intersect_point.coords))

    final_intersect_points = np.array(intersect_points)
    final_intersect_points = np.squeeze(final_intersect_points, axis=1)
    final_fake_intersect_points = np.array(fake_intersect_points)
    final_fake_intersect_points = np.squeeze(final_fake_intersect_points, axis=1)

    return final_intersect_points, final_fake_intersect_points


def is_cell_existed(
    cell_coord: List[List], thresh: int = 15, *lines: Tuple[List, ...]
) -> bool:
    """This is a function to check whether the coordinate is
    the coordinate of an existing cell or not.

    Args:
        cell_coord (List[List]): The coordinate of cell
        thresh (int): The threshold value to group line which has same x, y coordinate

    Returns:
        Bool: returns True if the coordinate is the coordinate of an existing cell,
        otherwise returns False
    """
    x1, y1, x2, y2 = cell_coord
    h_lines, v_lines = lines[0][0]

    left_status = is_line([x1, y1, x1, y2], np.array(v_lines), axis=1, ths=thresh)
    if left_status is False:
        return False

    right_status = is_line([x2, y1, x2, y2], np.array(v_lines), axis=1, ths=thresh)
    if right_status is False:
        return False

    top_status = is_line([x1, y1, x2, y1], np.array(h_lines), axis=0, ths=thresh)
    if top_status is False:
        return False

    bottom_status = is_line([x1, y2, x2, y2], np.array(h_lines), axis=0, ths=thresh)
    if bottom_status is False:
        return False

    return True


def get_bottom_right_corner(
    pred_point: Tuple, points: np.ndarray, ths: int = 5
) -> Tuple:
    """This is a function which find the coordinates of bottom right point of
    a cell by coordinate of top left point

    Args:
        pred_point (Tuple): The top left point has form (x, y)
        points (np.ndarray): The list of intersection points has form [[x, y]]
        ths (int, optional): The threshold to find the coordinate of point
        on y-axis which is nearest to top left point. Defaults to 5.

    Returns:
        Tuple: The coordinate of bottom right point has form [x, y]
    """
    dup_pred_point = np.array(list(pred_point) * len(points)).reshape(len(points), -1)
    minus_arr = abs(dup_pred_point - points)

    nearest_point_metric = np.sum(minus_arr, axis=1)

    # Dont find intersection point
    if np.all(nearest_point_metric > ths):
        return None, None
    else:
        index = np.argmin(nearest_point_metric)
        bottom_right_vertices = points[index]

        return (bottom_right_vertices[0], bottom_right_vertices[1])


def calculate_cell_coordinate(
    points: np.ndarray, fake_flag: bool, ths: int = 15, *lines: List
) -> List[List]:
    """This is a function which find the coordinate of cells in table

    Args:
        points (List[List]): The list of the coordinate of intersection points
        fake_flag (Bool): if True, this method extract fake the coordinate of points,
        otherwise find the real coordinate of points
        ths (int): The threshold value to find the coordinate of point o
        n y-axis which is nearest to top left point. Defaults to 5.
    Returns:
        List[List]: The coordinate of cells.
    """
    cells = []
    x_coords = np.array(points[:, 0])
    y_coords = np.array(points[:, 1])

    for point in points:
        x1, y1 = point

        # define the nearest right and bottom vertices
        y_coord_mask = (x_coords > x1 - ths) & (x_coords < x1 + ths) & (y_coords > y1)
        filter_y_coords = sorted(y_coords[y_coord_mask])

        # define the nearest right and bottom vertices
        x_coord_mask = (y_coords > y1 - ths) & (y_coords < y1 + ths) & (x_coords > x1)
        filter_x_coords = sorted(x_coords[x_coord_mask])

        if len(filter_y_coords) > 0 and len(filter_x_coords) > 0:
            status = 0
            for pred_x2 in filter_x_coords:
                for pred_y2 in filter_y_coords:
                    point = (pred_x2, pred_y2)
                    x2, y2 = get_bottom_right_corner(point, points.copy(), ths)
                    if x2 and y2:
                        if fake_flag:
                            status = 1
                            break
                        else:
                            if is_cell_existed([x1, y1, x2, y2], ths, lines):
                                status = 1
                                break
                if status == 1:
                    break
        else:
            x2, y2 = None, None

        if x2 and y2 and status == 1:
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            cells.append([x1, x2, y1, y2])

    return cells


def sort_cell(cells: np.ndarray, ths: int = 5) -> List[List]:
    """Sort cells from left to right and top to bottom

    Args:
        cells (List[List]): The coordinate of cells.
        ths (int, optional): The threshold value to group cells has same y coordinate.

    Returns:
        List[List]: The sorted coordinate of cells
    """
    sorted_cells = []
    y1_coords = cells[:, 2]
    uni_y1_coords = np.unique(y1_coords)

    for y1_coord in uni_y1_coords:
        # take cells which have same y coordinate
        y_mask = (y1_coords < y1_coord + ths) & (y1_coords > y1_coord - ths)

        # sorted recorresponding to x axis
        same_y_cells = sorted(cells[y_mask], key=lambda x: x[0])
        for cell in same_y_cells:
            x1, x2, y1, y2 = cell
            if [x1, x2, y1, y2] not in sorted_cells:
                sorted_cells.append([x1, x2, y1, y2])

    return sorted_cells


def predict_relation(cells: List[List]) -> Tuple[List, List]:
    """Extract relationship value between cells.

    Args:
        cells (List[List]): The sorted coordinate of cells
    Returns:
        Tuple[List, List]: tuple contain id couples corrsponding to relationship value
    """

    hor_couple_ids = []
    ver_couple_ids = []
    nb_cells = len(cells)
    list_x1_coords = np.array(cells)[:, 0]
    list_y1_coords = np.array(cells)[:, 2]
    ids = np.arange(nb_cells)

    for id_a, cell in enumerate(cells):
        x1_a, x2_a, y1_a, y2_a = cell

        # vertical
        for id_b in ids[list_y1_coords == y2_a]:
            if id_b == id_a:
                continue
            x1_b, x2_b, _, _ = cells[id_b]
            sorted_x = sorted([x1_a, x2_a, x1_b, x2_b])
            if abs(sorted_x[1] - sorted_x[2]) != 0:
                if (sorted_x[1] >= x1_a) and (sorted_x[2] <= x2_a):
                    ver_couple_ids.append([id_a, id_b])

        # horizontal
        for id_b in ids[list_x1_coords == x2_a]:
            if id_b == id_a:
                continue
            _, _, y1_b, y2_b = cells[id_b]

            sorted_y = sorted([y1_a, y2_a, y1_b, y2_b])
            if abs(sorted_y[1] - sorted_y[2]) != 0:
                if (sorted_y[1] >= y1_a) and (sorted_y[2] <= y2_a):
                    hor_couple_ids.append([id_a, id_b])

    return hor_couple_ids, ver_couple_ids
