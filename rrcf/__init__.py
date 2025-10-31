from newrrcf.newrrcf import *
from newrrcf.shingle import shingle
from importlib.metadata import version, PackageNotFoundError

__version__: str

try:
    __version__ = version("newrrcf")
except PackageNotFoundError:
    __version__ = "unknown"
