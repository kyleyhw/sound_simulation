from simulate import Simulate
from waveforms import waveform_registry
from setup import Driver

def save_full_simulation_results(simulation_object: Simulate, hdf5_file, simulation_id):
    """Saves a simulation into an already open HDF5 file object."""
    sim_group = hdf5_file.create_group(f'simulation_{simulation_id:04d}')
    sim_group.create_dataset('history', data=simulation_object.history, compression="gzip")

    params = simulation_object.get_params()
    for key, value in params.items():
        if isinstance(value, tuple):
            value = str(value)
        sim_group.attrs[key] = value

    driver_group = sim_group.create_group('drivers')
    for i, driver in enumerate(simulation_object.drivers):
        driver_subgroup = driver_group.create_group(f'driver_{i}')
        # Save the waveform's class name
        waveform_class_name = driver.waveform.__class__.__name__
        driver_subgroup.attrs['waveform_class'] = waveform_class_name

        # Save the waveform's parameters (kwargs)
        for key, value in driver.waveform.__dict__.items():
            driver_subgroup.attrs[key] = value


def load_simulation_from_archive(archive_file, sim_id):
    """Loads a full simulation object, including drivers, from the archive."""
    sim_group = archive_file[sim_id]

    loaded_drivers = []
    if 'drivers' in sim_group:
        for driver_name in sim_group['drivers']:
            driver_subgroup = sim_group['drivers'][driver_name]

            # 1. Get the class name and look it up in the registry
            waveform_class_name = driver_subgroup.attrs['waveform_class']
            WaveformClass = waveform_registry[waveform_class_name]

            # 2. Recreate the kwargs from ALL other saved attributes
            kwargs = {key: val for key, val in driver_subgroup.attrs.items()
                      if key not in ['location', 'waveform_class']}

            # 3. Create the waveform and driver objects
            waveform_obj = WaveformClass(**kwargs)
            location = tuple(eval(driver_subgroup.attrs['location']))
            loaded_drivers.append(Driver(location=location, waveform=waveform_obj))

    # ... you would then load the main params and history and
    #     return a fully reconstructed Simulate object
    # For now, we'll just return the drivers to show it works
    return loaded_drivers