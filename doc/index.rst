Welcome to turn_by_turn' documentation!
=======================================

``turn_by_turn`` is a library for reading and writing from various particle accelerator measurement data formats used at `CERN <https://home.cern/>`_.

It provides a custom dataclass ``TbtData`` to do so, with attributes corresponding to the relevant measurements information.

How to Use turn_by_turn
=======================

There are two main ways to create a ``TbtData`` object:

1. **Reading from file (disk):**
   Use ``read_tbt`` to load measurement data from a file on disk. This is the standard entry point for working with measurement files in supported formats.

2. **In-memory conversion:**
   Use ``convert_to_tbt`` to convert data that is already loaded in memory (such as a pandas DataFrame, tfs DataFrame, or xtrack.Line) into a ``TbtData`` object. This is useful for workflows where you generate or manipulate data in Python before standardizing it.

Both methods produce a ``TbtData`` object, which can then be used for further analysis or written out to supported formats.

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
- For writing, see the next section.

Writing Data
============

To write a ``TbtData`` object to disk, use the ``write_tbt`` function. This function supports writing in the LHC SDDS format by default, as well as other supported formats depending on the ``datatype`` argument. The output format is determined by the ``datatype`` you specify, but for most workflows, SDDS is the standard output.

Example::

   from turn_by_turn.io import write_tbt
   write_tbt("output.sdds", tbt_data)

Package Reference
=================

.. toctree::
   :caption: Modules
   :maxdepth: 1
   :glob:

   modules/*

.. toctree::
   :caption: Readers
   :maxdepth: 1
   :glob:

   readers/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

