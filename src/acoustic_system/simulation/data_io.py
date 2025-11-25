from .simulate import Simulate
from .waveforms import waveform_registry
from .setup import Driver, Sensor

class SaveSimulationResults:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def _save_history(self, history, sim_group):
        sim_group.create_dataset('history', data=history, compression="gzip")

    def _save_params(self, params, sim_group):
        for key, value in params.items():
            if isinstance(value, tuple):
                value = str(value)
            sim_group.attrs[key] = value

    def _save_drivers(self, drivers, sim_group):
        driver_group = sim_group.create_group('drivers')
        for i, driver in enumerate(drivers):
            driver_subgroup = driver_group.create_group(f'driver_{i}')
            waveform_class_name = driver.waveform.__class__.__name__
            driver_subgroup.attrs['waveform_class'] = waveform_class_name
            for key, value in driver.waveform.__dict__.items():
                driver_subgroup.attrs[key] = value

    def _save_sensors(self, sensors, sim_group):
        sensor_group = sim_group.create_group('sensors')
        for i, sensor in enumerate(sensors):
            sensor_subgroup = sensor_group.create_group(f'sensor_{i}')
            sensor_subgroup.attrs['position'] = sensor.position
            sensor_subgroup.create_dataset('timeseries', data=sensor.timeseries, compression="gzip")

    def save_full_history(self, simulation_id, history, params, drivers, verbose=False):
        sim_group = self.hdf5_file.create_group(f'simulation_{simulation_id:04d}')
        self._save_history(history, sim_group)
        self._save_params(params, sim_group)
        self._save_drivers(drivers, sim_group)
        if verbose:
            print(f'saved full history for {simulation_id}')

    def save_sensor_results(self, simulation_id, sensors, params, drivers, verbose=False):
        sim_group = self.hdf5_file.create_group(f'simulation_{simulation_id:04d}')
        self._save_sensors(sensors, sim_group)
        self._save_params(params, sim_group)
        self._save_drivers(drivers, sim_group)
        if verbose:
            print(f'saved sensor results for {simulation_id}')


class LoadSimulationResults:
    """Loads raw data from an HDF5 archive without reconstructing objects."""

    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.save_type = self.hdf5_file.attrs.get('save_type')
        if not self.save_type:
            raise ValueError("HDF5 file is missing the required 'save_type' attribute.")
        self._loader_registry = {
            'full_history': self._load_full_history_raw,
            'sensor_results': self._load_sensor_results_raw,
        }

    def _load_full_history_raw(self, sim_group):
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
        sim_group_name = f'simulation_{simulation_id:04d}'
        sim_group = self.hdf5_file[sim_group_name]
        loader_func = self._loader_registry.get(self.save_type)
        if loader_func:
            return loader_func(sim_group)
        else:
            raise ValueError(f"Unknown save_type: '{self.save_type}' found in file attributes.")