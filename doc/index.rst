Welcome to turn_by_turn' documentation!
=======================================

``turn_by_turn`` is a library for reading and writing from various particle accelerator measurement data formats used at `CERN <https://home.cern/>`_.

It provides a custom dataclass ``TbtData`` to do so, with attributes corresponding to the relevant measurements information.

How to Use turn_by_turn
-----------------------

There are two main ways to create a ``TbtData`` object:

1. **Reading from file (disk):**
   Use ``read_tbt`` to load measurement data from a file on disk. This is the standard entry point for working with measurement files in supported formats.

2. **In-memory conversion:**
   Use ``convert_to_tbt`` to convert data that is already loaded in memory (such as a pandas DataFrame, tfs DataFrame, or xtrack.Line) into a ``TbtData`` object. This is useful for workflows where you generate or manipulate data in Python before standardizing it.

Both methods produce a ``TbtData`` object, which can then be used for further analysis or written out to supported formats.

Supported Modules and Limitations
---------------------------------

Different modules support different file formats and workflows (disk reading vs. in-memory conversion). For a detailed table of which modules support which features, and any important limitations, see the documentation for the :mod:`turn_by_turn.io` module.

- Only ``madng`` and ``xtrack`` support in-memory conversion.
- Most modules are for disk reading only.
- For writing, see the next section.

Writing Data
------------

To write a ``TbtData`` object to disk, use the ``write_tbt`` function. This function supports writing in the LHC SDDS format by default, as well as other supported formats depending on the ``datatype`` argument. The output format is determined by the ``datatype`` you specify, but for most workflows, SDDS is the standard output.

Example::

   from turn_by_turn.io import write_tbt
   write_tbt("output.sdds", tbt_data)

Package Reference
-----------------

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
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
