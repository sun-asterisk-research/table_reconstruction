from typing import List, Tuple

import cv2
import numpy as np
from scipy.spatial import distance as dist
from skimage import measure

from .mask_utils import get_hor_lines_mask, get_ver_lines_mask


def get_table_line(binimg: np.ndarray, axis: int, lineW: int = 5) -> List[List]:
    """Extract the coordinate of lines from table binary image

    Args:
        binimg (np.ndarray): Table binary image
        axis (int): if 0, extracted line is horizontal lines,
        otherwise extracted line is vertical lines.
        lineW (int, optional): The minimum line width. Defaults to 5.

    Returns:
        List[List]: The coordinate of extracted line.
    """
    labels = measure.label(binimg > 0, connectivity=2)
    regions = measure.regionprops(labels)

    lineboxes = []
    if axis == 1:
        for line in regions:
            if line.bbox[2] - line.bbox[0] > lineW:
                lineboxes.append(min_area_rect(line.coords))
    else:
        for line in regions:
            if line.bbox[3] - line.bbox[1] > lineW:
                lineboxes.append(min_area_rect(line.coords))

    return lineboxes


def min_area_rect(coords: np.ndarray) -> List:
    """Get coordinate of line

    Args:
        coords (ndarray): Coordinate list (row, col) of the region. has shape (N, 2)

    Returns:
        List: The coordinate of line
    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()

    box = image_location_sort_box(box)

    x1, y1, x2, y2, x3, y3, x4, y4 = box
    _, w, h, _, _ = solve(box)

    if w < h:
        xmin = (x1 + x2) / 2
        xmax = (x3 + x4) / 2
        ymin = (y1 + y2) / 2
        ymax = (y3 + y4) / 2
    else:
        xmin = (x1 + x4) / 2
        xmax = (x2 + x3) / 2
        ymin = (y1 + y4) / 2
        ymax = (y2 + y3) / 2

    return [xmin, ymin, xmax, ymax]


def solve(box: List) -> Tuple[int, int, int, int, int]:
    """Caculate angle, width, height, the center coordinate of box

    Args:
        box (List): the coordinate of region

    Returns:
        Tuple[int, int, int, int, int]: (angle, width, height, center x, center y)
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
    ) / 2
    h = (
        np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
    ) / 2

    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)

    return angle, w, h, cx, cy


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Extract top left. top right, bottom left, bottom right of region

    Args:
        pts (np.ndarray[Tuple]): The coordinate of points

    Returns:
        np.ndarray: The coordinate of points.
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def image_location_sort_box(box: List) -> List:
    """Sort and extract the coordinate of points

    Args:
        box (List): the coordinate of region

    Returns:
        List: the sorted coordinate of region
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    arr_pts = np.array(pts, dtype="float32")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(arr_pts)

    return [x1, y1, x2, y2, x3, y3, x4, y4]


def get_lines_coordinate(line_mask: np.ndarray, axis: int, ths: int = 30) -> np.ndarray:
    """Extract coordinate of line from  binary image

    Args:
        line_mask (np.ndarray): the line binary image
        axis (int): if axis=0, line_mask is horizontal lines binary image,
        otherwise line_mask is vertical lines binary image.
        ths (int, optional): The threshold value to ignore noise edge.

    Returns:
        np.ndarray: the coordinate of lines.
    """

    boxes = get_table_line(line_mask, axis=axis, lineW=ths)

    lines_coordinate = []
    for coor in boxes:
        x1, y1, x2, y2 = coor
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if axis == 0 and (y2 - y1) > ths:
            continue
        elif axis == 1 and (x2 - x1) > ths:
            continue

        lines_coordinate.append([x1, y1, x2, y2])

    return np.array(lines_coordinate)


def get_table_coordinate(hor_lines: np.ndarray, ver_lines: np.ndarray) -> List:
    """Extract the coordinate of table in image

    Args:
        hor_lines_coord (np.ndarray): The coordinate of horizontal lines
        ver_lines_coord (np.ndarray): The coordinate of vertical lines

    Returns:
        List: The coordinat of table has form (xmin, ymin, xmax, ymax)
    """

    tab_x1 = min(min(hor_lines[:, 0]), min(ver_lines[:, 0]))
    tab_y1 = min(min(hor_lines[:, 1]), min(ver_lines[:, 1]))
    tab_x2 = max(max(hor_lines[:, 2]), max(ver_lines[:, 2]))
    tab_y2 = max(max(hor_lines[:, 3]), max(ver_lines[:, 3]))

    return [tab_x1, tab_y1, tab_x2, tab_y2]


def remove_noise(
    hor_lines: np.ndarray,
    ver_lines: np.ndarray,
    ths: int = 15,
    noise_edge_ths: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove noise edge from image

    Args:
        hor_lines (np.ndarray): The coordinate of horizontal lines
        ver_lines (np.ndarray): The coordinate of vertical lines
        ths (int, optional): Threshold value to group lines which has same coordinate.
        noise_edge_ths (float, optional): Threshold value check whether
        the line is noise edge or not.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The coordinate of horizontal and vertical lines.
    """
    hor_mask = np.array([True] * len(hor_lines))
    ver_mask = np.array([True] * len(ver_lines))
    hor_x1 = hor_lines[:, 0]
    hor_x2 = hor_lines[:, 2]
    hor_y = hor_lines[:, 1]

    ver_x = ver_lines[:, 0]
    ver_y1 = ver_lines[:, 1]
    ver_y2 = ver_lines[:, 3]

    max_hor_length = max(hor_x2 - hor_x1)
    max_ver_length = max(ver_y2 - ver_y1)

    if abs(min(hor_x1) - min(ver_x)) > ths:
        if min(ver_x) > min(hor_x1):
            mask_index = (hor_x1 > min(hor_x1) - ths) & (hor_x1 < min(hor_x1) + ths)
            hor_mask[mask_index] = [False] * len(hor_mask[mask_index])
        else:
            mask_index = (ver_x > min(ver_x) - ths) & (ver_x < min(ver_x) + ths)
            ver_mask[mask_index] = [False] * len(ver_mask[mask_index])

    if abs(max(hor_x2) - max(ver_x)) > ths:
        if max(hor_x2) > max(ver_x):
            mask_index = (hor_x2 > max(hor_x2) - ths) & (hor_x2 < max(hor_x2) + ths)
            hor_mask[mask_index] = [False] * len(hor_mask[mask_index])
        else:
            mask_index = (ver_x > max(ver_x) - ths) & (ver_x < max(ver_x) + ths)
            ver_mask[mask_index] = [False] * len(ver_mask[mask_index])

    if abs(min(hor_y) - min(ver_y1)) > ths:
        if min(hor_y) > min(ver_y1):
            mask_index = (ver_y1 > min(ver_y1) - ths) & (ver_y1 < min(ver_y1) + ths)
            ver_mask[mask_index] = [False] * len(ver_mask[mask_index])
        else:
            mask_index = (hor_y > min(hor_y) - ths) & (hor_y < min(hor_y) + ths)
            hor_mask[mask_index] = [False] * len(hor_mask[mask_index])

    if abs(max(hor_y) - max(ver_y2)) > ths:
        if max(hor_y) > max(ver_y2):
            mask_index = (hor_y > max(hor_y) - ths) & (hor_y < max(hor_y) + ths)
            hor_mask[mask_index] = [False] * len(hor_mask[mask_index])
        else:
            mask_index = (ver_y2 > max(ver_y2) - ths) & (ver_y2 < max(ver_y2) + ths)
            ver_mask[mask_index] = [False] * len(ver_mask[mask_index])

    for i, stat in enumerate(hor_mask):
        if not stat:
            x1, _, x2, _ = hor_lines[i]
            if (x2 - x1) / max_hor_length > noise_edge_ths:
                hor_mask[i] = True

    for i, stat in enumerate(ver_mask):
        if stat is False:
            _, y1, _, y2, _ = ver_lines[i]
            if (y2 - y1) / max_ver_length > noise_edge_ths:
                ver_mask[i] = True

    return hor_lines[hor_mask], ver_lines[ver_mask]


def get_coordinates(
    mask: np.ndarray, ths: int = 5, kernel_len: int = 10
) -> Tuple[List, np.ndarray, np.ndarray]:
    """This function extract the coordinate of table, horizontal and vertical lines.

    Args:
        mask (np.darray): A binary table image
        ths (int, optional): Threshold value to ignore the lines
        has not same y coordinate for horizontal lines or x coordinate
        for vertical lines. Defaults to 5.
        kernel_len (int, optional): The size of kernel is applied.

    Raises:
        ValueError: will be raised if the number of detected lines is not enough to
            rebuild the table

    Returns:
        Tuple[List, np.ndarray, np.ndarray]: Tuple contain the coordinate of
        table, vertical and horizontal lines.
    """

    # get horizontal lines mask image
    horizontal_lines_mask = get_hor_lines_mask(mask, kernel_len)

    # get vertical lines mask image
    vertical_lines_mask = get_ver_lines_mask(mask, kernel_len)

    # get coordinate of horizontal and vertical lines
    hor_lines = get_lines_coordinate(horizontal_lines_mask, axis=0, ths=ths)
    ver_lines = get_lines_coordinate(vertical_lines_mask, axis=1, ths=ths)

    if len(hor_lines.shape) != 2 or len(ver_lines.shape) != 2:
        raise ValueError("Empty line coords array")
    # remove noise edge
    hor_lines, ver_lines = remove_noise(hor_lines, ver_lines, ths)

    # get coordinate of table
    tab_x1, tab_y1, tab_x2, tab_y2 = get_table_coordinate(hor_lines, ver_lines)

    # preserve sure that all table has 4 borders
    new_ver_lines = []
    new_hor_lines = []
    for e in ver_lines:
        x1, y1, x2, y2 = e

        # dont add left and right border
        if abs(x1 - tab_x1) >= ths and abs(x2 - tab_x2) >= ths:
            new_ver_lines.append([x1, y1, x2, y2])

    for e in hor_lines:
        x1, y1, x2, y2 = e

        # dont add top and bottom border
        if abs(y1 - tab_y1) >= ths and abs(y2 - tab_y2) >= ths:
            new_hor_lines.append([x1, y1, x2, y2])

    # add top, bottom ,left, right border
    new_ver_lines.append([tab_x1, tab_y1, tab_x1, tab_y2])
    new_ver_lines.append([tab_x2, tab_y1, tab_x2, tab_y2])
    new_hor_lines.append([tab_x1, tab_y1, tab_x2, tab_y1])
    new_hor_lines.append([tab_x1, tab_y2, tab_x2, tab_y2])

    # normalize
    final_hor_lines = normalize_v1(new_hor_lines, axis=0, ths=ths)
    final_ver_lines = normalize_v1(new_ver_lines, axis=1, ths=ths)
    final_hor_lines, final_ver_lines = normalize_v2(final_ver_lines, final_hor_lines)

    return [tab_x1, tab_y1, tab_x2, tab_y2], final_ver_lines, final_hor_lines


def normalize_v1(lines: List[List], axis: int, ths: int = 10) -> np.ndarray:
    """Normalize the coordinate of vertical lines or horizontal lines

    Args:
        lines (List[List]): The coordinate of horizontal lines or vertical lines.
        axis (int): If 0, lines is horizontal lines, otherwise vertical lines.
        ths (int, optional): Threshold value to group the lines
        has same x or y coordinate.

    Returns:
        np.ndarray: The normalized coordinate of lines.
    """
    filter_lines = np.array(lines.copy())
    id_range = np.arange(len(filter_lines))
    x1_coords = filter_lines.copy()[:, 0]
    y1_coords = filter_lines.copy()[:, 1]
    x2_coords = filter_lines.copy()[:, 2]
    y2_coords = filter_lines.copy()[:, 3]

    # horizontal
    if axis == 0:
        # equalize x1
        for v in np.unique(x1_coords):
            x1_mask = (v - ths < x1_coords) & (x1_coords < v + ths)
            update_coord = np.min(x1_coords[x1_mask])

            filter_lines[id_range[x1_mask], 0] = update_coord

        # equalize x2
        for v in np.unique(x2_coords):
            x2_mask = (v - ths < x2_coords) & (x2_coords < v + ths)
            update_coord = np.max(x2_coords[x2_mask])

            filter_lines[id_range[x2_mask], 2] = update_coord

        # equalize y
        concat_y = np.concatenate((y1_coords, y2_coords))
        for v in np.unique(concat_y):
            y1_mask = (v - ths < y1_coords) & (y1_coords < v + ths)
            y2_mask = (v - ths < y2_coords) & (y2_coords < v + ths)

            filter_y = np.concatenate((y1_coords[y1_mask], y2_coords[y2_mask]))
            update_coord = int(np.max(filter_y))

            filter_lines[id_range[y1_mask], 1] = update_coord
            filter_lines[id_range[y2_mask], 3] = update_coord
    else:
        # vertical
        # equalize y1
        for v in np.unique(y1_coords):
            y1_mask = (v - ths < y1_coords) & (y1_coords < v + ths)
            update_coord = np.min(y1_coords[y1_mask])

            filter_lines[id_range[y1_mask], 1] = update_coord

        # equalize y2
        for v in np.unique(y2_coords):
            y2_mask = (v - ths < y2_coords) & (y2_coords < v + ths)
            update_coord = np.max(y2_coords[y2_mask])

            filter_lines[id_range[y2_mask], 3] = update_coord

        # equalize x
        concat_x = np.concatenate((x1_coords, x2_coords))
        for v in np.unique(concat_x):
            x1_mask = (v - ths < x1_coords) & (x1_coords < v + ths)
            x2_mask = (v - ths < x2_coords) & (x2_coords < v + ths)
            filter_x = np.concatenate((x1_coords[x1_mask], x2_coords[x2_mask]))
            update_coord = int(np.max(filter_x))

            filter_lines[id_range[x1_mask], 0] = update_coord
            filter_lines[id_range[x2_mask], 2] = update_coord

    return filter_lines


def normalize_v2(
    ver_lines: np.ndarray, hor_lines: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the coordinate between vertical lines and horizontal lines

    Args:
        ver_lines_coord (np.ndarray): The coordinate of vertical lines
        hor_lines_coord (np.ndarray): The coordinate of horizontal lines

    Returns:
        Tuple(np.ndarray, np.ndarray): the normalized coordinate of
        horizontal and vertical lines
    """
    # normalize x1
    ver_x1 = ver_lines[:, 0].copy()
    hor_x1 = hor_lines[:, 0].copy()

    for i, x1 in enumerate(hor_x1):
        concat_coor = sorted(np.concatenate(([x1], ver_x1)))
        tgt_idx = np.argwhere(concat_coor == x1)[0][0]

        if abs(concat_coor[tgt_idx - 1] - x1) > abs(concat_coor[tgt_idx + 1] - x1):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]

        hor_lines[i, 0] = update_coord

    # normalize x2
    ver_x2 = ver_lines[:, 2].copy()
    hor_x2 = hor_lines[:, 2].copy()

    for i, x2 in enumerate(hor_x2):
        concat_coor = sorted(np.concatenate(([x2], ver_x2)))
        tgt_idx = np.argwhere(concat_coor == x2)[0][0]

        if abs(concat_coor[tgt_idx - 1] - x2) > abs(concat_coor[tgt_idx + 1] - x2):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]

        hor_lines[i, 2] = update_coord

    # normalize y1
    ver_y1 = ver_lines[:, 1].copy()
    hor_y1 = hor_lines[:, 1].copy()

    for i, y1 in enumerate(ver_y1):
        concat_coor = sorted(np.concatenate(([y1], hor_y1)))
        tgt_idx = np.argwhere(concat_coor == y1)[0][0]

        if abs(concat_coor[tgt_idx - 1] - y1) > abs(concat_coor[tgt_idx + 1] - y1):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]

        ver_lines[i, 1] = update_coord

    # normalize y2
    ver_y2 = ver_lines[:, 3].copy()
    hor_y2 = hor_lines[:, 3].copy()

    for i, y2 in enumerate(ver_y2):
        concat_coor = sorted(np.concatenate(([y2], hor_y2)))
        tgt_idx = np.argwhere(concat_coor == y2)[0][0]

        if abs(concat_coor[tgt_idx - 1] - y2) > abs(concat_coor[tgt_idx + 1] - y2):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]

        ver_lines[i, 3] = update_coord

    return hor_lines, ver_lines


def is_line(line: List, lines: np.ndarray, axis: int, ths: float) -> bool:
    """Check whether the coordinate is the coordinate of an existing line or not.

    Args:
        line (List): The coordinate of line
        lines (np.ndarray): The coordinate of lines
        axis (int): If axis == 0 lines is horizontal line, otherwise vertical lines.
        thresh (float): The threshold value to group line which has same x, y coordinate

    Returns:
        bool: returns True if the coordinate is coordinate of an existing line,
        otherwise returns False
    """
    x1, y1, x2, y2 = line

    # horizontal
    if axis == 0:
        y1_coord_list = lines[:, 1]
        lines_mask = (y1_coord_list > y1 - ths) & (y1_coord_list < y1 + ths)
        sub_h_lines = lines[lines_mask]

        for coor in sub_h_lines:
            line_x1, line_y1, line_x2, line_y2 = coor

            if line_x1 - ths <= x1 < x2 <= line_x2 + ths:
                return True
    # vertical
    elif axis == 1:
        x1_coord_list = lines[:, 0]
        lines_mask = (x1_coord_list > x1 - ths) & (x1_coord_list < x1 + ths)
        sub_v_lines = lines[lines_mask]

        for coor in sub_v_lines:
            line_x1, line_y1, line_x2, line_y2 = coor

            if line_y1 - ths <= y1 < y2 <= line_y2 + ths:
                return True

    return False
