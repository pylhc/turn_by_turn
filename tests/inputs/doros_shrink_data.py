"""
Quick script to remove some data from the DOROS files, to make them smaller for the tests.
"""

import shutil
import subprocess
from pathlib import Path

import h5py

from turn_by_turn.doros import DataKeys

file_in = "/afs/cern.ch/work/j/jdilly/dorostest/DOROS-2024-09-29_01_37_13_522358-NO_USER.h5"
file_out = "./tests/inputs/test_doros.h5"

# Values need to coincide with the values in the test_doros.py
N_BPMS = 3
NTURNS = 50000

# Copy file to temporary location and "delete" unneccessary data (only removes references to data)
file_temp = file_out + ".tmp"
shutil.copyfile(file_in, file_temp)

data_keys_list = [DataKeys.get_data_keys(datatype) for datatype in DataKeys.types()]

with h5py.File(file_temp, "r+", track_order=True) as hdf_file:
    bpms = [name for name in hdf_file["/"] if data_keys_list[0].n_samples in hdf_file[f"/{name}"]]

    for bpm in bpms[:N_BPMS]:
        for data_keys in data_keys_list:
            del hdf_file[bpm][data_keys.n_samples]
            hdf_file[bpm].create_dataset(data_keys.n_samples, data=[NTURNS])
            data_x = hdf_file[bpm][data_keys.data["X"]][:NTURNS]
            data_y = hdf_file[bpm][data_keys.data["Y"]][:NTURNS]

            del hdf_file[bpm][data_keys.data["X"]]
            del hdf_file[bpm][data_keys.data["Y"]]
            hdf_file[bpm].create_dataset(data_keys.data["X"], data=data_x)
            hdf_file[bpm].create_dataset(data_keys.data["Y"], data=data_y)

    for bpm in bpms[N_BPMS:]:
        del hdf_file[bpm]

# Repack file (which actually removes the data in the file)
subprocess.run(["h5repack", file_temp, file_out])
Path(file_temp).unlink()
