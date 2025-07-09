"""
IO
--

This module contains high-level I/O functions to read and write turn-by-turn data objects to and from different formats.

Reading Data
============
Since version ``0.9.0`` of the package, data can be loaded either from file or from in-memory structures exclusive to certain codes (for some tracking simulation in *MAD-NG* or *xtrack*).
Two different APIs are provided for these use cases.

1. **To read from file**, use the ``read_tbt`` function (exported as ``read`` at the package's level). The file format is detected or specified by the ``datatype`` parameter.
2. **To load in-memory data**, use the ``convert_to_tbt`` function (exported as ``convert`` at the package's level). This is valid for tracking simulation results from e.g. *xtrack* or sent back by *MAD-NG*.

In both cases, the returned value is a structured ``TbtData`` object.

Writing Data
============
The single entry point for writing to disk is the ``write_tbt`` function (exported as ``write`` at the package's level). This writes a ``TbtData`` object to disk, typically in the LHC SDDS format (by default). The output file extension and format are determined by the ``datatype`` argument.

The following cases arise:
- If ``datatype`` is set to ``lhc``, ``sps`` or ``ascii``, the output will be in SDDS format and the file extension will be set to ``.sdds`` if not already present.
- If ``datatype`` is set to ``madng``, the output will be in a TFS file (extension ``.tfs`` is recommended).
- Other supported datatypes (see ``WRITERS``) will use their respective formats and conventions if implemented.

The ``datatype`` parameter controls both the output format and any additional options passed to the underlying writer.
Should the ``noise`` parameter be used, random noise will be added to the data before writing. A ``seed`` can be provided for reproducibility.

Example::

    from turn_by_turn import write
    write("output.sdds", tbt_data)  # writes in SDDS format by default
    write("output.tfs", tbt_data, datatype="madng")  # writes a TFS file in MAD-NG's tracking results format
    write("output.sdds", tbt_data, noise=0.01, seed=42)  # reproducibly adds noise before writing

While data can be loaded from the formats of different machines/codes (each through its own reader module), writing functionality is at the moment always done in the ``LHC``'s **SDDS** format by default, unless another supported format is specified. The interface is designed to be future-proof and easy to extend for new formats.


Supported Modules and Limitations
=================================

The following table summarizes which modules support disk reading and in-memory conversion, and any important limitations:

+----------------+---------------------+-----------------------+----------------------------------------------------------+
| Module         | Disk Reading        | In-Memory Conversion  | Notes / Limitations                                      |
+================+=====================+=======================+==========================================================+
| lhc            | Yes (SDDS, ASCII)   | No                    | Reads LHC SDDS and legacy ASCII files.                   |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| sps            | Yes (SDDS, ASCII)   | No                    | Reads SPS SDDS and legacy ASCII files.                   |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| doros          | Yes (HDF5)          | No                    | Reads DOROS HDF5 files.                                  |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| madng          | Yes (TFS)           | Yes                   | In-memory: only via pandas/tfs DataFrame.                |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| xtrack         | No                  | Yes                   | Only in-memory via xtrack.Line.                          |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| ptc            | Yes (trackone)      | No                    | Reads MAD-X PTC trackone files.                          |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| esrf           | Yes (Matlab .mat)   | No                    | Experimental/untested.                                   |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| iota           | Yes (HDF5)          | No                    | Reads IOTA HDF5 files.                                   |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| ascii          | Yes (legacy ASCII)  | No                    | For legacy ASCII files only.                             |
+----------------+---------------------+-----------------------+----------------------------------------------------------+
| trackone       | Yes (MAD-X)         | No                    | Reads MAD-X trackone files.                              |
+----------------+---------------------+-----------------------+----------------------------------------------------------+

- Only ``madng`` and ``xtrack`` support in-memory conversion.
- Most modules are for disk reading only.
- Some modules (e.g., ``esrf``) are experimental or have limited support.

API
===
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from turn_by_turn import (
    ascii,  # noqa: A004
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
from turn_by_turn.utils import add_noise_to_tbt

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pandas import DataFrame
    from xtrack import Line

    from turn_by_turn.structures import TbtData

TBT_MODULES = {
    "lhc": lhc,
    "doros": doros,
    "doros_positions": doros,
    "doros_oscillations": doros,
    "sps": sps,
    "iota": iota,
    "esrf": esrf,
    "ptc": ptc,
    "trackone": trackone,
    "ascii": ascii,
    "madng": madng,
    "xtrack": xtrack_line,
}

# Modules supporting in-memory conversion to TbtData (not file readers)
TBT_CONVERTERS = ("madng", "xtrack")

# implemented writers
WRITERS = (
    "lhc",
    "sps",
    "doros",
    "doros_positions",
    "doros_oscillations",
    "ascii",
    "madng",
)

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
def convert_to_tbt(file_data: DataFrame | Line, datatype: str = "xtrack") -> TbtData:
    """
    Convert a pandas or tfs DataFrame (MAD-NG) or a Line (XTrack) to a TbtData object.
    Args:
        file_data (Union[DataFrame, xt.Line]): The data to convert.
        datatype (str): The type of the data, either 'xtrack' or 'madng'. Defaults to 'xtrack'.
    Returns:
        TbtData: The converted TbtData object.
    """
    if datatype.lower() not in TBT_CONVERTERS:
        raise DataTypeError(
            f"Only {','.join(TBT_CONVERTERS)} converters are implemented for now."
        )

    module = TBT_MODULES[datatype.lower()]
    return module.convert_to_tbt(file_data)  # No additional arguments as no doros.


def write_tbt(
    output_path: str | Path,
    tbt_data: TbtData,
    noise: float = None,
    seed: int = None,
    datatype: str = "lhc",
) -> None:
    """
    Write a ``TbtData`` object's data to file, in the ``LHC``'s **SDDS** format.

    Args:
        output_path (Union[str, Path]): path to a the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
        noise (float): optional noise to add to the data.
        seed (int): A given seed to initialise the RNG if one chooses to add noise. This is useful
            to ensure the exact same RNG state across operations. Defaults to ``None``, which means
            any new RNG operation in noise addition will pull fresh entropy from the OS.
        datatype (str): type of matrices in the file, determines the reader to use. Case-insensitive,
            defaults to ``lhc``.
    """
    output_path = Path(output_path)
    if datatype.lower() not in WRITERS:
        raise DataTypeError(
            f"Only {','.join(WRITERS)} writers are implemented for now."
        )

    if datatype.lower() in ("lhc", "sps", "ascii") and output_path.suffix != ".sdds":
        # I would like to remove this, but I'm afraid of compatibility issues with omc3 (jdilly, 2024)
        output_path = output_path.with_name(f"{output_path.name}.sdds")

    # If the datatype is not in the list of writers, we raise an error. Therefore the datatype
    # must be in the TBT_MODULES dictionary -> No need for a try-except block here.
    try:
        module = TBT_MODULES[datatype.lower()]
    except KeyError:
        raise DataTypeError(
            f"Invalid datatype: {datatype}. Ensure it is one of {', '.join(TBT_MODULES)}."
        )
    else:
        if noise is not None:
            tbt_data = add_noise_to_tbt(tbt_data, noise=noise, seed=seed)
        return module.write_tbt(output_path, tbt_data, **additional_args(datatype))


def additional_args(datatype: str) -> dict[str, Any]:
    """Additional parameters to be added to the reader/writer function.

    Args:
        datatype (str): Type of the data.
    """
    if datatype.lower() == "doros_oscillations":
        return {"data_type": doros.DataKeys.OSCILLATIONS}

    if datatype.lower() == "doros_positions":
        return {"data_type": doros.DataKeys.POSITIONS}

    return {}
