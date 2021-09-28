from typing import List, Union

import numpy as np
import torch
from numpy import array, ndarray
from PIL.Image import Image
from pkg_resources import DistributionNotFound, get_distribution

from .line_segmentation.line_segment import LineSegmentation  # noqa: F401
from .output.cell import Cell
from .output.table import Table
from .table_detection.detector import TableDetector
from .utils.cell_utils import (
    calculate_cell_coordinate,
    get_intersection_points,
    predict_relation,
    sort_cell,
)
from .utils.lines_utils import get_coordinates
from .utils.mask_utils import normalize
from .utils.table_utils import DirectedGraph, convertSpanCell2DocxCoord

__version__ = None
try:
    __version__ = get_distribution("table_reconstruction").version
except DistributionNotFound:
    __version__ == "0.0.0"  # package is not installed
    pass


class TableExtraction:
    """ """

    def __init__(
        self,
        device: torch.device,
        line_segment_weight_path: str = None,
        table_detection_weight_path: str = None,
        normalize_thresh: int = 15,
    ) -> None:
        """[summary]

        Args:
            device (torch.device)
            line_segment_weight_path (str, optional):  Defaults to None.
            table_detection_weight_path (str, optional):  Defaults to None.
            normalize_thresh (int, optional): Normalize threshold used after receive
                result from line segmentation model. Defaults to 15.
        """
        self.table_detection_model = TableDetector(
            table_detection_weight_path, device=device.type
        )
        self.line_segmentation_model = LineSegmentation(
            line_segment_weight_path, device=device
        )
        self.normalize_thresh = normalize_thresh

    def extract(self, image: Union[ndarray, Image]) -> List[Table]:
        """Extract table from image

        Args:
            image (Union[ndarray, Image]): Input image

        Raises:
            ValueError: will be raised if the input image is not np.ndarray or PIL.Image

        Returns:
            List[Table]: list of extracted tables
        """
        if isinstance(image, Image):
            image = array(image)
        elif not isinstance(image, ndarray):
            raise ValueError(("Input image must be Numpy array or PIL Image"))

        table_regions = self.table_detection_model.predict([image])

        tables = []
        for region in table_regions[0]:
            x_min, y_min, x_max, y_max = region

            img = image[y_min:y_max, x_min:x_max]

            h, w, _ = img.shape
            padding_img = np.ones((h + 10, w + 10, 3), dtype=np.uint8) * 255
            padding_img[5 : h + 5, 5 : w + 5, :] = img

            mask = self.line_segmentation_model.predict(padding_img)
            mask = normalize(img, mask_img=mask)
            mask = np.array(mask[5 : h + 5, 5 : w + 5])
            try:
                (
                    tab_coord,
                    vertical_lines_coord,
                    horizontal_lines_coord,
                ) = get_coordinates(mask, ths=self.normalize_thresh)
            except Exception as e:
                print(str(e))
                continue

            intersect_points, fake_intersect_points = get_intersection_points(
                horizontal_lines_coord, vertical_lines_coord, tab_coord
            )

            cells = calculate_cell_coordinate(
                intersect_points.copy(),
                False,
                self.normalize_thresh,
                [horizontal_lines_coord, vertical_lines_coord],
            )

            fake_cells = calculate_cell_coordinate(
                fake_intersect_points.copy(), True, self.normalize_thresh
            )

            if len(cells) <= 1:
                continue
            cells = sort_cell(cells=np.array(cells))
            fake_cells = sort_cell(cells=np.array(fake_cells))

            hor_couple_ids, ver_couple_ids = predict_relation(cells)

            H_Graph = DirectedGraph(len(cells))
            H_Graph.add_edges(hor_couple_ids)
            nb_col = H_Graph.findLongestPath()

            V_Graph = DirectedGraph(len(cells))
            V_Graph.add_edges(ver_couple_ids)
            nb_row = V_Graph.findLongestPath()

            span_list = convertSpanCell2DocxCoord(
                cells, fake_cells, list(range(len(cells))), nb_col
            )

            cells_list = [
                Cell(
                    (c_x_min, c_x_max, c_y_min, c_y_max),
                    col_index=span_info["y"][0],
                    row_index=span_info["x"][0],
                    col_span=span_info["y"][1],
                    row_span=span_info["x"][1],
                )
                for span_info, (c_x_min, c_x_max, c_y_min, c_y_max) in zip(
                    span_list, cells
                )
            ]

            tables.append(
                Table(
                    coordinate=(x_min, x_max, y_min, y_max),
                    col_numb=nb_col,
                    row_numb=nb_row,
                    cells=cells_list,
                )
            )
        return tables
