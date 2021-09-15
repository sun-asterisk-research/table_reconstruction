from shapely.geometry import LineString
from table_reconstruction.utils.lines_utils import is_line
import numpy as np

def get_intersection_points(horizontal_lines:list, vertical_lines:list, tab_coord:list) -> tuple:
    """This a function which find the coordinate (x, y) of intersection points

    Args:
        horizontal_lines (list): The coordinate of horizontal lines has form [(x_min, y_min. x_max, y_max)]
        vertical_lines (list): The coordinate of vertical lines has form [(x_min, y_min. x_max, y_max)]
        tab_coord (list): The coordinate of table has form [x_min, y_min, x_max, y_max]

    Returns:
        tuple: The tuple contains intersection points and fake intersection points
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
    

    intersect_points = np.array(intersect_points)
    intersect_points = np.squeeze(intersect_points, axis=1)
    fake_intersect_points = np.array(fake_intersect_points)
    fake_intersect_points = np.squeeze(fake_intersect_points, axis=1)

    return intersect_points, fake_intersect_points


def is_cell_existed(cell_coord: list, thresh: float, *lines) -> bool:   
    """This is a function to check whether the coordinate is the coordinate of an existing cell or not.

    Args:
        cell_coord (list): The coordinate of cell has form [x_min, y_min, x_max, y_max]
        thresh (float): [description]

    Returns:
        bool: This method returns True if the coordinate is the coordinate of an existing cell, otherwise returns False
    """
    x1, y1, x2, y2 = cell_coord    
    h_lines, v_lines = lines[0][0]

    left_status = is_line([x1, y1, x1, y2], v_lines, axis=1, thresh=thresh)
    if left_status is False:
        return False

    right_status = is_line([x2, y1, x2, y2], v_lines, axis=1, thresh=thresh)
    if right_status is False:
        return False

    top_status = is_line([x1, y1, x2, y1], h_lines, axis=0, thresh=thresh)
    if top_status is False:
        return False

    bottom_status = is_line([x1, y2, x2, y2], h_lines, axis=0, thresh=thresh)
    if bottom_status is False:
        return False

    return True

def get_bottom_right_corner(pred_point: tuple, points: list, ths=5) -> tuple: 
    """This is a function which find the coordinates of bottom right point of a cell by coordinate of top left point

    Args:
        pred_point (tuple): The top left point has form (x, y)
        points (list): The list of intersection points has form [[x, y]]
        ths (int, optional): The threshold value to find the coordinate of point on y-axis which is nearest to top left point. Defaults to 5.

    Returns:
        tuple: The coordinate of bottom right point has form [x, y]
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


def calculate_cell_coordinate(points: list, fake_flag: bool, thresh: int, *lines) -> list:
    """This is a function which find the coordinate of cells in table

    Args:
        points (list): The list of the coordinate of intersection points has form [(x_min, y_min, x_max, y_max)]
        fake_flag (bool): if True, this method extract fake the coordinate of points, otherwise find the real coordinate of points
        thresh (int): The threshold value to find the coordinate of point on y-axis which is nearest to top left point. Defaults to 5.
    Returns:
        [List]: The coordinate of cells.
    """
    cells = []
    list_x_coords = np.array(points[:, 0])
    list_y_coords = np.array(points[:, 1])

    for point in points:
        x1, y1 = point
        
        # define the nearest right and bottom vertices
        y_coord_mask = (list_x_coords > x1 - thresh) & (list_x_coords < x1 + thresh) & (list_y_coords > y1)
        filter_y_coords = sorted(list_y_coords[y_coord_mask])

        # define the nearest right and bottom vertices
        x_coord_mask = (list_y_coords > y1 - thresh) & (list_y_coords < y1 + thresh) & (list_x_coords > x1)
        filter_x_coords = sorted(list_x_coords[x_coord_mask])

        if len(filter_y_coords) > 0 and len(filter_x_coords) > 0:
            status = 0
            for pred_x2 in filter_x_coords:
                for pred_y2 in filter_y_coords:
                    x2, y2 = get_bottom_right_corner((pred_x2, pred_y2), points.copy(), thresh)
                    if x2 and y2:
                        if fake_flag:
                            status = 1
                            break
                        else:
                            if is_cell_existed([x1, y1, x2, y2], thresh, lines):
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


def sort_cell(cells: list, ths=5) -> list:
    """Sort cells from left to right and top to bottom

    Args:
        cells (list): The coordinate of cells.
        ths (int, optional): The threshold value to group cells which has same y coordinate. Defaults to 5.

    Returns:
        list: The sorted coordinate of cells
    """
    sorted_cells = []
    cells = np.array(cells)
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


def predict_relation(cells: list):
    """This is a function which extract relationship value between cells.

    Args:
        cells (list): The sorted coordinate of cells
    Returns:
        (list, list): two list contain id couples corrsponding to relationship value
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
            if abs(sorted_x[1] - sorted_x[2]) != 0 and (sorted_x[1] >= x1_a) and (sorted_x[2] <= x2_a):
                ver_couple_ids.append([id_a, id_b])
    
        # horizontal
        for id_b in ids[list_x1_coords == x2_a]:
            if id_b == id_a:
                continue
            _, _, y1_b, y2_b = cells[id_b]
            
            sorted_y = sorted([y1_a, y2_a, y1_b, y2_b])
            if abs(sorted_y[1] - sorted_y[2]) != 0 and (sorted_y[1] >= y1_a) and (sorted_y[2] <= y2_a):
                hor_couple_ids.append([id_a, id_b])
    
    return hor_couple_ids, ver_couple_ids