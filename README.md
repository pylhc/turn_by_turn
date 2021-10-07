# Turn-By-Turn

[![Cron Testing](https://github.com/pylhc/turn_by_turn/workflows/Cron%20Testing/badge.svg)](https://github.com/pylhc/turn_by_turn/actions?query=workflow%3A%22Cron+Testing%22)
[![Code Climate coverage](https://img.shields.io/codeclimate/coverage/pylhc/turn_by_turn.svg?style=popout)](https://codeclimate.com/github/pylhc/turn_by_turn)
[![Code Climate maintainability (percentage)](https://img.shields.io/codeclimate/maintainability-percentage/pylhc/turn_by_turn.svg?style=popout)](https://codeclimate.com/github/pylhc/turn_by_turn)
[![GitHub last commit](https://img.shields.io/github/last-commit/pylhc/turn_by_turn.svg?style=popout)](https://github.com/pylhc/turn_by_turn/)
[![GitHub release](https://img.shields.io/github/release/pylhc/turn_by_turn.svg?style=popout)](https://github.com/pylhc/turn_by_turn/)

[comment]: <> (TODO: uncomment below once we give turn_by_turn its zenodo releases)
[comment]: <> ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.5070986.svg&#41;]&#40;https://doi.org/10.5281/zenodo.5070986&#41;)

This package provides reading functionality for turn-by-turn BPM measurements data from different particle accelerators.
It also provides writing functionality in the `LHC`'s own SDDS format, through our `sdds` package.
Files are read into a custom-made `TbtData` dataclass encompassing the relevant information.

## Installing

Installation is easily done via `pip`:
```
pip install turn_by_turn
```

## Example Usage

 The package is imported as `turn_by_turn`, and exports top-level functions for reading and writing:
```python
import turn_by_turn as tbt

# Loading a file is simple and returns a custom dataclass named TbtData
data: tbt.TbtData = tbt.read("Beam2@BunchTurn@2018_12_02@20_08_49_739.sdds", datatype="lhc")

# Easily access relevant information from the loaded data: transverse data, measurement date, 
# number of turns, bunches and IDs of the recorded bunches
first_bunch_transverse_positions: tbt.TransverseData = data.matrices[0]
measurement_date = data.date  # a datetime.datetime object

# Transverse positions are recorded as pandas DataFrames
first_bunch_x = first_bunch_transverse_positions.X.copy()
first_bunch_y = first_bunch_transverse_positions.Y.copy()

# Do any operations with these as you usually do with pandas
first_bunch_mean_x = first_bunch_x.mean()

# Average over all bunches/particles at all used BPMs from the measurement
averaged_tbt: tbt.TbtData = tbt.utils.generate_average_tbtdata(data)

# Writing out to disk (in the LHC's SDDS format) is simple too, potentially with added noise
tbt.write("path_to_output.sdds", averaged_tbt, noise=1e-5)
```

See the [API documentation](https://pylhc.github.io/turn_by_turn/) for details.

## License

This project is licensed under the `MIT License` - see the [LICENSE](LICENSE) file for details.
