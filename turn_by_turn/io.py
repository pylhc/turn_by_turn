"""
IO
--

This module contains high-level I/O functions to read and write turn-by-turn data objects in different
formats. While data can be loaded from the formats of different machines / codes, each format getting its
own reader module, writing functionality is at the moment always done in the ``LHC``'s **SDDS** format.
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

if TYPE_CHECKING:
    from pandas import DataFrame
    from xtrack import Line

LOGGER = logging.getLogger(__name__)

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

def load_tbt_data(
    tbt_input: str | Path | Line | DataFrame, 
    datatype: str = "lhc"
) -> TbtData:
    """
    Get a TbtData object from various input types. Explicitly does not infer the datatype from the input. 

    Args:
        tbt_input (str | Path | Line | DataFrame): 
            The input data object or path to a file.
        datatype (str): 
            Defaults to "lhc".

    Returns:
        TbtData: The resulting TbtData object.
    
    Raises:
        DataTypeError: If the datatype is None and the input type cannot be inferred.
        TypeError: If the input type is not supported for the given datatype.
    """
    datatype = datatype.lower()

    try:
        module = TBT_MODULES[datatype]
    except KeyError as error:
        LOGGER.exception(
            f"Unsupported datatype '{datatype}'. Supported types: {list(TBT_MODULES.keys())}"
        )
        raise DataTypeError(datatype) from error

    if isinstance(tbt_input, (str, Path)):
        return module.read_tbt(Path(tbt_input), **additional_args(datatype))
    return module.convert_to_tbt(tbt_input)

        
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
