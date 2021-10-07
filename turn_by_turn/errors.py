"""
Errors
------

Errors that can be raised during the handling of turn-by-turn data files.
"""


class DataTypeError(Exception):
    """Raised when an unsupported or invalid datatype is given in I/O operations."""

    def __init__(self, datatype: str):
        super().__init__(f"Provided datatype {datatype} is not supported by this package")


class ExclusiveArgumentsError(Exception):
    """Raised when two incompatible arguments are provided to a function."""

    def __init__(self, *args):
        super().__init__(f"Only one of the following should be provided: {args}")


class HDF5VersionError(Exception):
    """Raised when the wrong HDF5 format version is used during read operations."""


class PTCFormatError(IOError):
    """Raised when a wrong format is detected in a PTC output file."""
