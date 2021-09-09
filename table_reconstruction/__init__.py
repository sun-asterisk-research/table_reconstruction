from pkg_resources import get_distribution, DistributionNotFound
__version__ = None
try:
    __version__ = get_distribution("table_reconstruction").version
except DistributionNotFound:
    __version__ == "0.0.0"  # package is not installed
    pass
