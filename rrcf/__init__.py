from rrcf.rrcf import *
from rrcf.shingle import shingle
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rrcf")
except PackageNotFoundError:
    __version__ = "unknown"
