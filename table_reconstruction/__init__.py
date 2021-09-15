from typing import List, Union

from numpy import array, ndarray
from PIL.Image import Image
from pkg_resources import DistributionNotFound, get_distribution

from .output.table import Table

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
        line_segment_weight_path: str = None,
        table_detection_weight_path: str = None,
    ) -> None:
        """

        Args:
            line_segment_weight_path (str, optional): Path to exported weight file of
                Line segmentation model. Defaults to None.
            table_detection_weight_path (str, optional):  Path to exported weight file
            of Table detection model. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError("Required models was not defined")

    def extract(self, image: Union[ndarray, Image]) -> List[Table]:
        """Extract tables from image

        Args:
            image (Union[ndarray, Image]): [description]

        Raises:
            ValueError: Will be raised when input image is not Numpy array or PIL Image
            NotImplementedError: [description]

        Returns:
            List[Table]: [description]
        """
        if isinstance(image, Image):
            image = array(image)
        elif not isinstance(image, ndarray):
            raise ValueError(("Input image must be Numpy array or PIL Image"))

        raise NotImplementedError("Extracting methods were not defined")
