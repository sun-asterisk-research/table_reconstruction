from skimage import measure
from scipy.spatial import distance as dist
import numpy as np
from table_reconstruction.utils.mask_utils import get_horizontal_lines_mask, get_vertical_lines_mask
import cv2


def get_table_line(binimg:np.array, axis=0,lineW=5):
    """Extract the coordinate of lines from table binary image

    Args:
        binimg (np.array): Table binary image
        axis (int, optional): if 0, extracted line is horizontal lines, otherwise extracted line is vertical lines. Defaults to 0.
        lineW (int, optional): The minimum line width. Defaults to 5.

    Returns:
        [list]: The coordinate of extracted line.
    """
    labels = measure.label(binimg > 0, connectivity = 2)
    regions = measure.regionprops(labels)

    if axis == 1:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[2] - line.bbox[0] > lineW ]
    else:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[3] - line.bbox[1] > lineW ]


    return lineboxes

def minAreaRect(coords):
    """Get coordinate of line

    Args:
        coords (ndarray): Coordinate list (row, col) of the region. has shape (N, 2)

    Returns:
        [list]: The coordinate of line
    """
    rect= cv2.minAreaRect(coords[:,::-1])
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

def solve(box):
    """Caculate angle, width, height, the center coordinate of box

    Args:
        box (list): the coordinate of region

    Returns:
        tuple: (angle, width, height, center x, center y)
    """
    x1, y1, x2, y2, x3, y3, x4, y4= box[:8]
    cx = (x1 + x3 + x2 + x4)/4.0
    cy = (y1 + y3 + y4 + y2)/4.0
    w = (np.sqrt((x2 - x1)**2 + (y2 - y1)**2) + np.sqrt((x3 - x4)**2 + (y3 - y4)**2))/2
    h = (np.sqrt((x2 - x3)**2 + (y2 - y3)**2) + np.sqrt((x1 - x4)**2 + (y1 - y4)**2))/2

    sinA = (h * (x1 - cx) -w * (y1 - cy)) * 1.0/(h * h + w * w) * 2
    angle = np.arcsin(sinA)

    return angle, w, h, cx, cy

def _order_points(pts):
    """Extract top left. top right, bottom left, bottom right of region

    Args:
        pts (np.ndarray): The coordinate of points

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

def image_location_sort_box(box):
    """Sort and extract the coordinate of points

    Args:
        box (list): the coordinate of region

    Returns:
        [list]: the sorted coordinate of region
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1),(x2, y2),(x3, y3),(x4, y4)
    pts = np.array(pts, dtype="float32")
    (x1, y1),(x2, y2),(x3, y3),(x4, y4) = _order_points(pts)

    return [x1, y1, x2, y2, x3, y3, x4, y4]


def get_lines_coordinate(line_mask:np.array, axis:int, ths=30):
    """Extract coordinate of line from  binary image

    Args:
        line_mask (np.array): the line binary image
        axis (int): if axis=0, line_mask is horizontal lines binary image, otherwise line_mask is vertical lines binary image.
        ths (int, optional): The threshold value to ignore noise edge. Defaults to 30.

    Returns:
        [list]: the coordinate of lines.
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

def get_table_coordinate(hor_lines_coord:list, ver_lines_coord:list):
    """Extract the coordinate of table in image

    Args:
        hor_lines_coord (list): The coordinate of horizontal lines
        ver_lines_coord (list): The coordinate of vertical lines

    Returns:
        [list]: The coordinat of table has form (xmin, ymin, xmax, ymax)
    """
    hor_lines_coord = np.array(hor_lines_coord)
    ver_lines_coord = np.array(ver_lines_coord)

    tab_x1 = min(min(hor_lines_coord[:, 0]), min(ver_lines_coord[:, 0]))
    tab_y1 = min(min(hor_lines_coord[:, 1]), min(ver_lines_coord[:, 1]))
    tab_x2 = max(max(hor_lines_coord[:, 2]), max(ver_lines_coord[:, 2]))
    tab_y2 = max(max(hor_lines_coord[:, 3]), max(ver_lines_coord[:, 3]))

    return tab_x1, tab_y1, tab_x2, tab_y2

def remove_noise(hor_lines_coord:list, ver_lines_coord:list, ths=15, noise_edge_ths=0.5):
    """Remove noise edge from image

    Args:
        hor_lines_coord (list): The coordinate of horizontal lines
        ver_lines_coord (list): The coordinate of vertical lines
        ths (int, optional): The threshold value to group lines which has same coordinate. Defaults to 15.
        noise_edge_ths (float, optional): The threshold value to check whether the line is noise edge or not. Defaults to 0.5.

    Returns:
        [tuple]: The coordinate of horizontal and vertical lines.
    """
    hor_mask = np.array([True] * len(hor_lines_coord))
    ver_mask = np.array([True] * len(ver_lines_coord))
    hor_x1 = hor_lines_coord[:, 0]
    hor_x2 = hor_lines_coord[:, 2]
    hor_y = hor_lines_coord[:, 1]

    ver_x = ver_lines_coord[:, 0]
    ver_y1 = ver_lines_coord[:, 1]
    ver_y2 = ver_lines_coord[:, 3]

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
            x1, _, x2, _ = hor_lines_coord[i]
            if (x2 - x1) / max_hor_length > noise_edge_ths:
                hor_mask[i] = True

    for i, stat in enumerate(ver_mask):
        if stat is False:
            _, y1, _, y2, _ = ver_lines_coord[i]
            if (y2 - y1) / max_ver_length > noise_edge_ths:
                ver_mask[i] = True

    return hor_lines_coord[hor_mask], ver_lines_coord[ver_mask]


def get_coordinates(mask: np.darray, thresh=5, kernel_len=10):
    """This function extract the coordinate of table, the coordinate of horizontal and vertical lines.

    Args:
        mask (np.darray): A binary table image
        thresh (int, optional): Threshold value to ignore the lines which has not same y coordinate for horizontal lines or x coordinate 
                                for vertical lines. Defaults to 5.
        kernel_len (int, optional): The size of kernel is applied in method cv2.getStructuringElement. Defaults to 10.

    Returns:
        [tuple]: Tuple contain the coordinate of table, the coordinate of vertical and horizontal lines.
    """

    # get horizontal lines mask image
    horizontal_lines_mask = get_horizontal_lines_mask(mask, kernel_len)

    # get vertical lines mask image
    vertical_lines_mask = get_vertical_lines_mask(mask, kernel_len)

    # get coordinate of horizontal and vertical lines
    hor_lines_coord = get_lines_coordinate(horizontal_lines_mask, axis=0, ths=thresh)
    ver_lines_coord = get_lines_coordinate(vertical_lines_mask, axis=1, ths=thresh)

    # remove noise edge
    hor_lines_coord, ver_lines_coord = remove_noise(hor_lines_coord, ver_lines_coord, thresh)

    # get coordinate of table
    tab_x1, tab_y1, tab_x2, tab_y2 = get_table_coordinate(hor_lines_coord, ver_lines_coord)


    # preserve sure that all table has 4 borders
    new_ver_lines_coord = []
    new_hor_lines_coord = []
    for e in ver_lines_coord:
        x1, y1, x2, y2 = e

        # dont add left and right border
        if abs(x1 - tab_x1) >= thresh and abs(x2 - tab_x2) >= thresh:
            new_ver_lines_coord.append([x1, y1, x2, y2])

    for e in hor_lines_coord:
        x1, y1, x2, y2 = e

        # dont add top and bottom border
        if abs(y1 - tab_y1) >= thresh and abs(y2 - tab_y2) >= thresh:
            new_hor_lines_coord.append([x1, y1, x2, y2])

    # add top, bottom ,left, right border
    new_ver_lines_coord.append([tab_x1, tab_y1, tab_x1, tab_y2])
    new_ver_lines_coord.append([tab_x2, tab_y1, tab_x2, tab_y2])
    new_hor_lines_coord.append([tab_x1, tab_y1, tab_x2, tab_y1])
    new_hor_lines_coord.append([tab_x1, tab_y2, tab_x2, tab_y2])

    # normalize
    new_hor_lines_coord = normalize_v1(new_hor_lines_coord, axis=0, thresh=thresh)
    new_ver_lines_coord = normalize_v1(new_ver_lines_coord, axis=1, thresh=thresh)
    new_hor_lines_coord, new_ver_lines_coord = normalize_v2(new_ver_lines_coord, new_hor_lines_coord)

    return [tab_x1, tab_y1, tab_x2, tab_y2], new_ver_lines_coord, new_hor_lines_coord


def normalize_v1(lines:list, axis:int, thresh=10):
    """Normalize the coordinate of vertical lines or horizontal lines

    Args:
        lines (list): The coordinate of horizontal lines or vertical lines.
        axis ([type]): if 0, lines is horizontal lines, otherwise lines is vertical lines.
        thresh (int, optional): The threshold value to group the lines which has same x or y coordinate. Defaults to 10.

    Returns:
        list: The normalized coordinate of lines.
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
            x1_mask = (v - thresh < x1_coords) & (x1_coords < v + thresh)
            update_coord = np.min(x1_coords[x1_mask])

            filter_lines[id_range[x1_mask], 0] = update_coord

        # equalize x2
        for v in np.unique(x2_coords):
            x2_mask = (v - thresh < x2_coords) & (x2_coords < v + thresh)
            update_coord = np.max(x2_coords[x2_mask])

            filter_lines[id_range[x2_mask], 2] = update_coord

        # equalize y
        concat_y = np.concatenate((y1_coords, y2_coords))
        for v in np.unique(concat_y):
            y1_mask = (v - thresh < y1_coords) & (y1_coords < v + thresh)
            y2_mask = (v - thresh < y2_coords) & (y2_coords < v + thresh)

            filter_y = np.concatenate((y1_coords[y1_mask], y2_coords[y2_mask]))
            update_coord = int(np.max(filter_y))

            filter_lines[id_range[y1_mask], 1] = filter_lines[id_range[y2_mask], 3] = update_coord
    else: # vertical
        # equalize y1
        for v in np.unique(y1_coords):
            y1_mask = (v - thresh < y1_coords) & (y1_coords < v + thresh)
            update_coord = np.min(y1_coords[y1_mask])

            filter_lines[id_range[y1_mask], 1] = update_coord

        # equalize y2
        for v in np.unique(y2_coords):
            y2_mask = (v - thresh < y2_coords) & (y2_coords < v + thresh)
            update_coord = np.max(y2_coords[y2_mask])

            filter_lines[id_range[y2_mask], 3] = update_coord

        # equalize x
        concat_x = np.concatenate((x1_coords, x2_coords))
        for v in np.unique(concat_x):
            x1_mask = (v - thresh < x1_coords) & (x1_coords < v + thresh)
            x2_mask = (v - thresh < x2_coords) & (x2_coords < v + thresh)
            filter_x = np.concatenate((x1_coords[x1_mask], x2_coords[x2_mask]))
            update_coord = int(np.max(filter_x))

            filter_lines[id_range[x1_mask], 0] = filter_lines[id_range[x2_mask], 2] = update_coord

    return filter_lines


def normalize_v2(ver_lines_coord:list, hor_lines_coord:list):
    """ Normalize the coordinate between vertical lines and horizontal lines

    Args:
        ver_lines_coord (list): The coordinate of vertical lines
        hor_lines_coord (list): The coordinate of horizontal lines

    Returns:
        tuple: the normalized coordinate of horizontal and vertical lines
    """
    ver_lines_coord = np.array(ver_lines_coord)
    hor_lines_coord = np.array(hor_lines_coord)

    # normalize x1
    ver_x1 = ver_lines_coord[:, 0].copy()
    hor_x1 = hor_lines_coord[:, 0].copy()

    for i, x1 in enumerate(hor_x1):
        concat_coor = sorted(np.concatenate(([x1], ver_x1)))
        tgt_idx = np.argwhere(concat_coor == x1)[0][0]

        if abs(concat_coor[tgt_idx - 1] - x1) > abs(concat_coor[tgt_idx + 1] - x1):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]

        hor_lines_coord[i, 0] = update_coord

    # normalize x2
    ver_x2 = ver_lines_coord[:, 2].copy()
    hor_x2 = hor_lines_coord[:, 2].copy()

    for i, x2 in enumerate(hor_x2):
        concat_coor = sorted(np.concatenate(([x2], ver_x2)))
        tgt_idx = np.argwhere(concat_coor == x2)[0][0]

        if abs(concat_coor[tgt_idx - 1] - x2) > abs(concat_coor[tgt_idx + 1] - x2):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]

        hor_lines_coord[i, 2] = update_coord

    # normalize y1
    ver_y1 = ver_lines_coord[:, 1].copy()
    hor_y1 = hor_lines_coord[:, 1].copy()

    for i, y1 in enumerate(ver_y1):
        concat_coor = sorted(np.concatenate(([y1], hor_y1)))
        tgt_idx = np.argwhere(concat_coor == y1)[0][0]

        if abs(concat_coor[tgt_idx - 1] - y1) > abs(concat_coor[tgt_idx + 1] - y1):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]

        ver_lines_coord[i, 1] = update_coord

    # normalize y2
    ver_y2 = ver_lines_coord[:, 3].copy()
    hor_y2 = hor_lines_coord[:, 3].copy()

    for i, y2 in enumerate(ver_y2):
        concat_coor = sorted(np.concatenate(([y2], hor_y2)))
        tgt_idx = np.argwhere(concat_coor == y2)[0][0]

        if abs(concat_coor[tgt_idx - 1] - y2) > abs(concat_coor[tgt_idx + 1] - y2):
            update_coord = concat_coor[tgt_idx + 1]
        else:
            update_coord = concat_coor[tgt_idx - 1]


        ver_lines_coord[i, 3] = update_coord

    return hor_lines_coord, ver_lines_coord

def is_line(line:list, lines:list, axis:int, thresh:int):
    """This is a function to check whether the coordinate is the coordinate of an existing line or not.

    Args:
        line (list): The coordinate of line
        lines (list): The coordinate of lines
        axis (int): If axis == 0 lines is the checking line is horizontal line, otherwise vertical lines.
        thresh (float): The threshold value to group line which has same x, y coordinate

    Returns:
        bool: This method returns True if the coordinate is the coordinate of an existing line, otherwise returns False
    """
    x1, y1, x2, y2 = line
    lines = np.array(lines)

    if axis == 0: # horizontal
        y1_coord_list = lines[:, 1]
        lines_mask = (y1_coord_list > y1 - thresh) & (y1_coord_list < y1 + thresh)
        sub_h_lines = lines[lines_mask]

        for coor in sub_h_lines:
            line_x1, line_y1, line_x2, line_y2 = coor

            if line_x1 - thresh <= x1 < x2 <= line_x2 + thresh:
                return True
    elif axis == 1: # vertical
        x1_coord_list = lines[:, 0]
        lines_mask = (x1_coord_list > x1 - thresh) & (x1_coord_list < x1 + thresh)
        sub_v_lines = lines[lines_mask]

        for coor in sub_v_lines:
            line_x1, line_y1, line_x2, line_y2 = coor

            if line_y1 - thresh <= y1 < y2 <= line_y2 + thresh:
                return True

    return False