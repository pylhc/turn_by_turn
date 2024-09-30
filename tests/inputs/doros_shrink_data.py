""" 
Quick script to remove some data from the DOROS files, to make them smaller for the tests.
"""
from pathlib import Path
import shutil
import subprocess
import h5py

from turn_by_turn.doros import DEFAULT_OSCILLATION_DATA, N_ORBIT_SAMPLES, N_OSCILLATION_SAMPLES, OSCILLATIONS, POSITIONS

file_in = "/afs/cern.ch/work/j/jdilly/dorostest/DOROS-2024-09-29_01_37_13_522358-NO_USER.h5"
file_out = "./tests/inputs/test_doros_2024-09-29.h5"

# Values need to coincide with the values in the test_doros.py
N_BPMS= 3
NTURNS = 50000

# Copy file to temporary location and "delete" unneccessary data (only removes references to data)
file_temp = file_out + ".tmp"
shutil.copyfile(file_in, file_temp)
with h5py.File(file_temp, "r+", track_order=True) as hdf_file:
    bpms = [name for name in hdf_file["/"].keys() if N_ORBIT_SAMPLES in hdf_file[f"/{name}"].keys()]

    for bpm in bpms[:N_BPMS]:
        del hdf_file[bpm][N_ORBIT_SAMPLES]
        hdf_file[bpm].create_dataset(N_ORBIT_SAMPLES, data=[NTURNS])
        data_x = hdf_file[bpm][POSITIONS["X"]][:NTURNS]
        data_y = hdf_file[bpm][POSITIONS["Y"]][:NTURNS]

        del hdf_file[bpm][POSITIONS["X"]]
        del hdf_file[bpm][POSITIONS["Y"]]
        hdf_file[bpm].create_dataset(POSITIONS["X"], data=data_x)
        hdf_file[bpm].create_dataset(POSITIONS["Y"], data=data_y)

        del hdf_file[bpm][N_OSCILLATION_SAMPLES]
        del hdf_file[bpm][OSCILLATIONS["X"]]
        del hdf_file[bpm][OSCILLATIONS["Y"]]
        hdf_file[bpm].create_dataset(N_OSCILLATION_SAMPLES, data=0)
        hdf_file[bpm].create_dataset(OSCILLATIONS["X"], data=[DEFAULT_OSCILLATION_DATA])
        hdf_file[bpm].create_dataset(OSCILLATIONS["Y"], data=[DEFAULT_OSCILLATION_DATA])
    
    for bpm in bpms[N_BPMS:]:
        del hdf_file[bpm]

# Repack file (which actually removes the data in the file)
subprocess.run(["h5repack", file_temp, file_out])
Path(file_temp).unlink()