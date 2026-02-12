# Changelog

All notable changes to **turn_by_turn** will be documented in this file.

#### v1.2.0 – 2026-01-28

This minor release adds the reading support of the SuperKEKB file format for the HER and LER rings,
through the `superkekb` module.

#### v1.1.0 – 2025-10-29

This release removes the experimental `esrf` module from the package, as it was both unused and out-of-date to the ESRF data format.

#### v1.0.0 – 2025-10-27

This is the first major release of `turn_by_turn`, marking the transition from a pre-1.0 version to a stable API.
This release includes introduces some small breaking changes to the API, mainly the removal of the `date` attribute from the `TbtData` dataclass, as it was not consistently populated across all datatypes and readers.
Instead a new attribute `meta` has been added
which is a dictionary to hold any additional metadata that might be relevant for a specific datatype or reader in the future,
or can be used to store user-defined metadata,
but the entries should not be relied upon to be present across all datatypes or readers.

Some common meta-entries are:

- `date`: The date and time of the measurement, if available from the file.
- `file`: The path to the file the data was loaded from.
- `source_datatype`: The datatype the data was loaded from, e.g. `lhc`, `sps`, `doros`, etc.
- `comment`: Any comment on the measurement.

**Changed:**

- Removed the `date` attribute from the `TbtData` dataclass.
- Reordered the parameters of the `TbtData` dataclass to have `matrices`, `nturns` first, as required attributes, then optinally `bunch_ids`,  and `meta`.
- Added a `meta` attribute to the `TbtData` dataclass to hold additional metadata as a dictionary.
- Updated all readers to populate the `meta` attribute with relevant metadata where available.
- Restructured the `iota` module. This should be mostly transparent to the user, unless they were using internal functions or classes from the `iota` module directly.

#### v0.9.1 – 2025-07-10

This patch release fixes the reading of SPS files after the technical stop in 2025, during which the format seems to have been changed. The array `MonPlanes` in the SDDS file, which before contained `1` if the BPM was vertical and `0` if it was horizontal switched to using `3` for vertical and `0` for horizontal BPMs.
The current implementation now tries to determine the BPM plane not from this array, but from the ‘.H’ and ‘.V’ at the end of the BPM name in the file. Only if this ending is not present — you are able to deactivate it in the writer as this ending is also not present in the SPS model — it will first be checked if `3`s are present in the array and then the new format used, otherwise it will be checked if `0`s are in the array and then the new format used.
Otherwise the reader will raise an informative error.
If you only have vertical BPMs in the old format or only horizontal BPMs in the new format (i.e. your `MonPlanes` array will consist only of `1`s) this will also cause the reader to not be able to determine the format and raise an error.

#### v0.9.0 – 2025-07-09

Release `0.9.0` adds functionality for the loading of tracking simulation data from an `xtrack.Line`. A specific tracking setup and the use of `ParticleMonitor`s is necessary.
This version introduces a new top-level function, `convert`, to handle data that already lives in-memory: the result of an `xtrack.Line` tracking and potentially data from `MAD_NG`, for now.

**Added:**

- A new module, `turn_by_turn.xtrack_line`, to handle loading data from an `xtrack.Line` after tracking. See the documentation for details.
- A new function, `turn_by_turn.convert`, similar to `turn_by_turn.read` but to handle in-memory data.

#### v0.8.0 – 2025-01-08

In release 0.8.0:
Added support for converting MAD-NG tracking results into turn-by-turn (“TBT”) form.

#### v0.7.2 – 2024-10-11

This patch release enables the capability to read also the oscillation data from DOROS and the means to switch between positions and oscillations data.

**Added:**

- `doros_oscillations`: Read/write data into the oscillations attributes of the doros-hdf5 file.
- `doros_positions`: Read/write data into the positions attributes of the doros-hdf5 file.
- The original `doros` datatype defaults now to `oscillations`.

#### v0.7.1 – 2024-10-02

In this patch release, the DOROS reader has been updated to handle files that have more entries on the root level than BPMs and `METADATA`.

**Changed:**

- Identifying BPMs in DOROS data by having the `"nbOrbitSamplesRead"` entry.

#### v0.7.0 – 2024-08-20

In this release, a reader and writer for DOROS BPMs in `hdf5` format has been added.

**Changed:**

- Added DOROS `hdf5` reader/writer
- Clean-up of the Documentation

#### v0.6.0 – 2023-12-01

Release `0.6.0` adds to the SPS-module the possibility to remove the trailing planes (.H/.V) from the BPM names upon reading, and adding them back on writing. Both are enabled by default.
This allows compatibility with the MAD-X models.

**Added:**

- sps-reader: `remove_trailing_bpm_plane` removes the trailing plane-suffixes (.H/.V) from the BPM names, if present
- sps-writer: `add_trailing_bpm_plane` adds plane-suffixes (.H/.V) to the BPM names, if not already present
Fixed:
- ascii-reader: returns `TbtData`-object instead of the individual parts for one.

#### v0.5.0 – 2023-06-05

Release `0.5.0` adds functionality for the loading of tracking simulation data in the `trackone` module.
Important: With this release the minimum supported Python version is upped to 3.8

**Added:**

- A new class, `TrackingData`, was added to `turn_by_turn.structures` which is similar to `TransverseData` but holds all 8 dimensions (`X`, `PX`, `Y`, `PY`, `T`, `PT`, `S`, `E`).
- The `read_tbt` function in `turn_by_turn.trackone` has a new boolean argument, `is_tracking_data`, to specify data should be loaded with this new class. Default behavior is unchanged.
- The `numpy_to_tbt` function in `turn_by_turn.utils`, which handles the loading, has a `dtype` argument to specify the datatype to load into. Default behavior is unchanged.
- The `generate_average_tbtdata` function in `turn_by_turn.utils` handles the new class.
Fixed:
- The `fieldnames` method in `TransverseData` and `TrackingData` is now a `classmethod` and is properly called.

#### v0.4.2 – 2022-09-21

A patch release, that now allows the ASCII module to be accessed directly from the main read/write functionality.

#### v0.4.1 – 2022-09-21

This is a bugfix release.

**Fixed:**

- Less strict checking for ASCII-File format (only a `#` in the first line is now required)
