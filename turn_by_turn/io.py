"""
IO
--

This module contains high-level I/O functions to read and write turn-by-turn data objects in different formats.

There are two main entry points for users:

1. ``read_tbt``: Reads turn-by-turn data from disk (file-based). Use this when you have a measurement file on disk and want to load it into a ``TbtData`` object. The file format is detected or specified by the ``datatype`` argument.

2. ``convert_to_tbt``: Converts in-memory data (such as a pandas DataFrame, tfs DataFrame, or xtrack.Line) to a ``TbtData`` object. Use this when your data is already loaded in memory and you want to standardize it for further processing or writing.

Writing Data
============

The single entry point for writing is ``write_tbt``. This function writes a ``TbtData`` object to disk, typically in the LHC SDDS format (default), but other formats are supported via the ``datatype`` argument. The output file extension and format are determined by the ``datatype`` you specify.

- If you specify ``datatype='lhc'``, ``'sps'``, or ``'ascii'``, the output will be in SDDS format and the file extension will be set to ``.sdds`` if not already present (for compatibility with downstream tools).
- If you specify ``datatype='madng'``, the output will be in MAD-NG TFS format (extension ``.tfs`` is recommended).
- Other supported datatypes (see ``WRITERS``) will use their respective formats and conventions.
- If you provide the ``noise`` argument, random noise will be added to the data before writing. The ``seed`` argument can be used for reproducibility.
- The ``datatype`` argument controls both the output format and any additional options passed to the underlying writer.
- The interface is extensible: new formats can be added by implementing a module with a ``write_tbt`` function and adding it to ``TBT_MODULES`` and ``WRITERS``.

Example::

    from turn_by_turn.io import write_tbt
    write_tbt("output.sdds", tbt_data)  # writes in SDDS format by default
    write_tbt("output.tfs", tbt_data, datatype="madng")  # writes in MAD-NG TFS format
    write_tbt("output.sdds", tbt_data, noise=0.01, seed=42)  # add noise before writing

While data can be loaded from the formats of different machines/codes (each format getting its own reader module), writing functionality is at the moment always done in the ``LHC``'s **SDDS** format by default, unless another supported format is specified. The interface is designed to be future-proof and easy to extend for new formats.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from turn_by_turn import (
    ascii,
    doros,
    esrf,
    iota,
    lhc,
    madng,
    ptc,
    sps,
    trackone,
    xtrack_line,
)

from turn_by_turn.ascii import write_ascii
from turn_by_turn.errors import DataTypeError
from turn_by_turn.structures import TbtData
from turn_by_turn.utils import add_noise_to_tbt

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pandas import DataFrame
    from xtrack import Line

TBT_MODULES = dict(
    lhc=lhc,
    doros=doros,
    doros_positions=doros,
    doros_oscillations=doros,
    sps=sps,
    iota=iota,
    esrf=esrf,
    ptc=ptc,
    trackone=trackone,
    ascii=ascii,
    madng=madng,
    xtrack=xtrack_line,
)

# Modules supporting in-memory conversion to TbtData (not file readers)
TBT_CONVERTERS = ("madng", "xtrack")

# implemented writers
WRITERS = ("lhc", "sps", "doros", "doros_positions", "doros_oscillations", "ascii", "madng")

write_lhc_ascii = write_ascii  # Backwards compatibility <0.4
     
def read_tbt(file_path: str | Path, datatype: str = "lhc") -> TbtData:
    """
    Calls the appropriate loader for the provided matrices type and returns a ``TbtData`` object of the
    loaded matrices.

    Args:
        file_path (Union[str, Path]): path to a file containing TbtData.
        datatype (str): type of matrices in the file, determines the reader to use. Case-insensitive,
            defaults to ``lhc``.

    Returns:
        A ``TbtData`` object with the loaded matrices.
    """
    file_path = Path(file_path)
    LOGGER.info(f"Loading turn-by-turn matrices from '{file_path}'")
    try:
        module = TBT_MODULES[datatype.lower()]
    except KeyError as error:
        LOGGER.exception(
            f"Unsupported datatype '{datatype}' was provided, should be one of {list(TBT_MODULES.keys())}"
        )
        raise DataTypeError(datatype) from error
    else:
        return module.read_tbt(file_path, **additional_args(datatype))

# Note: I don't specify tfs.TfsDataFrame as this inherits from pandas.DataFrame
def convert_to_tbt(file_data: DataFrame | Line, datatype: str = 'xtrack') -> TbtData:
    """
    Convert a pandas or tfs DataFrame (MAD-NG) or a Table (XTrack) to a TbtData object.
    Args:
        file_data (Union[DataFrame, Table]): The data to convert.
        datatype (str): The type of the data, either 'xtrack' or 'madng'. Defaults to 'xtrack'.
    Returns:
        TbtData: The converted TbtData object.
    """
    if datatype.lower() not in TBT_CONVERTERS:
        raise DataTypeError(f"Only {','.join(TBT_CONVERTERS)} converters are implemented for now.")
    
    module = TBT_MODULES[datatype.lower()]
    return module.convert_to_tbt(file_data) # No additional arguments as no doros.


def write_tbt(output_path: str | Path, tbt_data: TbtData, noise: float = None, seed: int = None, datatype: str = "lhc") -> None:
    """
    Write a ``TbtData`` object's data to file, in the ``LHC``'s **SDDS** format.

    Args:
        output_path (Union[str, Path]): path to a the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
        noise (float): optional noise to add to the data.
        seed (int): A given seed to initialise the RNG if one chooses to add noise. This is useful
            to ensure the exact same RNG state across operations. Defaults to `None`, which means
            any new RNG operation in noise addition will pull fresh entropy from the OS.
        datatype (str): type of matrices in the file, determines the reader to use. Case-insensitive,
            defaults to ``lhc``.
    """
    output_path = Path(output_path)
    if datatype.lower() not in WRITERS:
        raise DataTypeError(f"Only {','.join(WRITERS)} writers are implemented for now.")

    if datatype.lower() in ("lhc", "sps", "ascii") and  output_path.suffix != ".sdds":  
        # I would like to remove this, but I'm afraid of compatibility issues with omc3 (jdilly, 2024) 
        output_path = output_path.with_name(f"{output_path.name}.sdds")

    # If the datatype is not in the list of writers, we raise an error. Therefore the datatype
    # must be in the TBT_MODULES dictionary -> No need for a try-except block here.
    try:
        module = TBT_MODULES[datatype.lower()]
    except KeyError:
        raise DataTypeError(f"Invalid datatype: {datatype}. Ensure it is one of {', '.join(TBT_MODULES)}.")
    else:
        if noise is not None:
            tbt_data = add_noise_to_tbt(tbt_data, noise=noise, seed=seed)
        return module.write_tbt(output_path, tbt_data, **additional_args(datatype))


def additional_args(datatype: str) -> dict[str, Any]:
    """ Additional parameters to be added to the reader/writer function. 
        
    Args:
        datatype (str): Type of the data.
    """
    if datatype.lower() == "doros_oscillations":
        return dict(data_type=doros.DataKeys.OSCILLATIONS)

    if datatype.lower() == "doros_positions":
        return dict(data_type=doros.DataKeys.POSITIONS)

    return dict()
