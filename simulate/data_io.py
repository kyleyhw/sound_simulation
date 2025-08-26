from simulate import Simulate
from waveforms import waveform_registry
from setup import Driver, Sensor

class SaveSimulationResults:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

        self._saver_registry = {
            'full_history': self._save_full_history,
            'sensor_results': self._save_sensor_results
        }

    def _save_history(self, simulation_object, sim_group):
        sim_group.create_dataset('history', data=simulation_object.history, compression="gzip")
        return

    def _save_params(self, simulation_object, sim_group):
        params = simulation_object.get_params()
        for key, value in params.items():
            if isinstance(value, tuple):
                value = str(value)
            sim_group.attrs[key] = value

        return

    def _save_drivers(self, simulation_object, sim_group):
        driver_group = sim_group.create_group('drivers')
        for i, driver in enumerate(simulation_object.drivers):
            driver_subgroup = driver_group.create_group(f'driver_{i}')
            # Save the waveform's class name
            waveform_class_name = driver.waveform.__class__.__name__
            driver_subgroup.attrs['waveform_class'] = waveform_class_name

            # Save the waveform's parameters (kwargs)
            for key, value in driver.waveform.__dict__.items():
                driver_subgroup.attrs[key] = value

        return

    def _save_sensors(self, simulation_object, sim_group):
        sensor_group = sim_group.create_group('sensors')
        for i, sensor in enumerate(simulation_object.sensors):
            sensor_subgroup = sensor_group.create_group(f'sensor_{i}')
            sensor_subgroup.attrs['location'] = sensor.location
            sensor_subgroup.create_dataset('timeseries', data=sensor.timeseries, compression="gzip")
        return

    def _save_full_history(self, simulation_object: Simulate, simulation_id, verbose=False):
        """Saves a simulation into an already open HDF5 file object."""
        sim_group = self.hdf5_file.create_group(f'simulation_{simulation_id:04d}')

        self._save_history(simulation_object=simulation_object, sim_group=sim_group)
        self._save_params(simulation_object=simulation_object, sim_group=sim_group)
        self._save_drivers(simulation_object=simulation_object, sim_group=sim_group)

        if verbose:
            print(f'saved full history for {simulation_id}')

        return

    def _save_sensor_results(self, simulation_object: Simulate, simulation_id, verbose=False):
        """Saves a simulation into an already open HDF5 file object."""
        sim_group = self.hdf5_file.create_group(f'simulation_{simulation_id:04d}')

        self._save_sensors(simulation_object=simulation_object, sim_group=sim_group)
        self._save_params(simulation_object=simulation_object, sim_group=sim_group)
        self._save_drivers(simulation_object=simulation_object, sim_group=sim_group)

        if verbose:
            print(f'saved full history for {simulation_id}')

        return

    def save_results(self, simulation_object, simulation_id, save_type='full_history', **kwargs):
        saver_func = self._saver_registry.get(save_type)
        if not saver_func:
            raise ValueError(f"Unknown save_type: '{save_type}'")

        saver_func(simulation_object, simulation_id, **kwargs)


# In your data_io.py file
import h5py


class LoadSimulationResults:
    """Loads raw data from an HDF5 archive without reconstructing objects."""

    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

        # 1. Read the file-level attribute once during initialization
        self.save_type = self.hdf5_file.attrs.get('save_type')
        if not self.save_type:
            raise ValueError("HDF5 file is missing the required 'save_type' attribute.")

        # The loader registry remains the same
        self._loader_registry = {
            'full_history': self._load_full_history_raw,
            'sensor_results': self._load_sensor_results_raw,
        }

    def _load_full_history_raw(self, sim_group):
        """Loads raw data for a full history simulation."""
        data = {
            'type': 'full_history',
            'params': dict(sim_group.attrs),
            'history': sim_group['history'][:],
            'drivers': []
        }
        if 'drivers' in sim_group:
            for name in sim_group['drivers']:
                data['drivers'].append(dict(sim_group['drivers'][name].attrs))
        return data

    def _load_sensor_results_raw(self, sim_group):
        """Loads raw data for a sensor results simulation."""
        data = {
            'type': 'sensor_results',
            'params': dict(sim_group.attrs),
            'sensors': [],
            'drivers': []
        }
        if 'sensors' in sim_group:
            for name in sim_group['sensors']:
                sensor_group = sim_group['sensors'][name]
                sensor_data = dict(sensor_group.attrs)
                sensor_data['timeseries'] = sensor_group['timeseries'][:]
                data['sensors'].append(sensor_data)
        if 'drivers' in sim_group:
            for name in sim_group['drivers']:
                data['drivers'].append(dict(sim_group['drivers'][name].attrs))
        return data

    def load_raw_results(self, simulation_id):
        """Loads the raw data for a single simulation."""
        sim_group_name = f'simulation_{simulation_id:04d}'
        sim_group = self.hdf5_file[sim_group_name]

        # 2. Use the save_type that was read from the file attribute
        loader_func = self._loader_registry.get(self.save_type)

        if loader_func:
            return loader_func(sim_group)
        else:
            raise ValueError(f"Unknown save_type: '{self.save_type}' found in file attributes.")
