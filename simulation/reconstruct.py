import h5py

from data_io import LoadSimulationResults

# filename = 'simulation_archive_2025-08-26_03-57-30_3x_full_history.hdf5'
filename = 'simulation_archive_2025-08-26_03-59-07_3x_sensor_results.hdf5'

with h5py.File('./training_data/'+filename, 'r') as hdf5_file:
    print(hdf5_file.keys())
    for i in range(len(hdf5_file.keys())):
        simloader = LoadSimulationResults(hdf5_file=hdf5_file)
        results = simloader.load_raw_results(simulation_id=i)
        print(results)