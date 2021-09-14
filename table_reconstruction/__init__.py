import os
from typing import List, Union

from numpy import array, ndarray
from PIL.Image import Image
from pkg_resources import DistributionNotFound, get_distribution

from table_reconstruction.output.table import Table

__version__ = None
try:
    __version__ = get_distribution("table_reconstruction").version
except DistributionNotFound:
    __version__ == "0.0.0"  # package is not installed
    pass


class TableExtraction:
    def __init__(self, line_segment_weight_path: str) -> None:
        if not os.path.exists(line_segment_weight_path):
            raise ValueError(
                "Could not find weight files at {}".format(line_segment_weight_path)
            )

        raise NotImplementedError("Line Segmentation model was not defined")

    def extract(self, image: Union[ndarray, Image]) -> List[Table]:
        if isinstance(image, Image):
            image = array(image)
        elif not isinstance(image, ndarray):
            raise ValueError(("Input image must be Numpy array or PIL Image"))

        raise NotImplementedError("Extracting methods were not defined")
