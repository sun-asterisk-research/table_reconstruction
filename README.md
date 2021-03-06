# Table Reconstruction

`table-reconstruction` is a tool used to detect table spaces and reconstruct the information in them using DL models.

To provide the above feature, Table reconstruction works based on several components as follows:

- A table detection model is developed based on Yolov5
- A line segmentation model is built based on Unet
- Additional modules are used in the information extraction process, especially a directed graph is used to extract information related to the merged cells.

## Before start

Due to the requirements of the used libraries, table-reconstruction requires version 3.7 or higher.

Currently, this package works well with most popular operating systems including Windows, Linux/GNU and MacOS. its system requirements will be mainly based on the requirements of Pytorch version 1.9.1, please check more [here](https://pytorch.org/get-started/locally/)

Note that although not exactly measured, the processing of this library uses a RAM amount of about 235.9 MiB (for the example provided [here](https://github.com/sun-asterisk-research/table_reconstruction/blob/master/example/example.ipynb)) when using the CPU device and about 1000MiB VRAM when used with GPU. In general, the amount of resources used is still quite large and they will be gradually reduced by optimizing the models used in the next versions.

Finally, because it does not require too much computing power, this library is only too demanding on CPU when most devices can use this package without any problems. The processing time with measured in the example provided above has a value of 13.4 s . wall time

## Installation

Table Reconstruction is published on [PyPI](https://pypi.org/project/table-reconstruction/) and can be installed from there:

```bash
pip install table-reconstruction
```

You can also install this package manually with the following command:

```bash
python setup.py install
```

## Basic usage

You can easily use this library by using the following statements:

```python
import torch
from table_reconstruction import TableExtraction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
extraction = TableExtraction(device=device)


image = ... # Accept Numpy ndarray and PIL image
tables = extraction.extract(image)
```

We also provide a simple Jupyter notebook which can be used to illustrate the results obtained after processing, please check it out [here](https://github.com/sun-asterisk-research/table_reconstruction/blob/master/example/example.ipynb)

## Documentation

Documentation will be available soon.

## Get in touch

- Report bugs, suggest features or view the source code on [GitHub](https://github.com/sun-asterisk-research/table_reconstruction).
