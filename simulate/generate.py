import numpy as np
import h5py
from datetime import datetime
from tqdm import tqdm

from simulate import Simulate
from setup import GenerateDriver, GenerateSensor
from data_io import SaveSimulationResults



if __name__ == '__main__':
    dims = 2

    if dims == 2:
        gridsize = (256, 256)

    gridstep = 8

    duration = 5
    timestep = 0.01

    wavespeed = 330

    save_type = 'sensor_results'
    number_of_runs = 3

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"simulation_archive_{timestamp}_{number_of_runs}x_{save_type}.hdf5"


    with h5py.File('./training_data/'+filename, 'w') as hdf5_file:
        simsaver = SaveSimulationResults(hdf5_file=hdf5_file)
        hdf5_file.attrs['save_type'] = save_type
        for run_number in tqdm(range(number_of_runs), desc='generating data'):
            simulation = Simulate(gridsize=gridsize, gridstep=gridstep, duration=duration, timestep=timestep,
                                  wavespeed=wavespeed)
            driver_gen = GenerateDriver(gridsize=gridsize)
            sensor_gen = GenerateSensor(gridsize=gridsize)

            number_of_speakers = np.random.randint(low=1, high=6)
            for i in range(number_of_speakers):
                driver = driver_gen.get_random_cosine(detailed=False)
                simulation.add_driver(driver=driver)

            sensor = sensor_gen.get_random_basic(detailed=False)
            simulation.add_sensor(sensor=sensor)

            simulation.check_stability()

            history = simulation.run()

            simsaver.save_results(simulation_object=simulation, simulation_id=run_number, save_type='full_history')



