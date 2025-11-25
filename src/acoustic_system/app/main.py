import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import numpy as np

# Adjusting the path to allow imports from the simulation package
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.setup import Driver, Sensor
from acoustic_system.simulation.waveforms import sinusoidal_packet, blackman_harris_window

# --- Simulation State Management ---
class SimulationManager:
    """Manages the state and execution of the FDTD simulation."""
    def __init__(self, sio):
        self.sio = sio
        self.simulation = None
        self.task = None
        self.is_running = False

    def setup_simulation(self, config):
        """Initializes the simulation with a given configuration."""
        grid_shape = tuple(config.get('grid_shape', (100, 100)))
        
        # Example driver and sensor setup
        driver_pos = tuple(config.get('driver_pos', (grid_shape[0] // 4, grid_shape[1] // 4)))
        
        duration = 500
        sample_rate = 2000
        frequency = 20
        
        waveform = sinusoidal_packet(duration, sample_rate, frequency,-1)
        # waveform = blackman_harris_window(duration)
        
        driver = Driver(position=driver_pos, waveform=waveform)
        self.simulation = Simulate(grid_shape=grid_shape, drivers=[driver])
        print("Simulation configured:", config)

    async def run_simulation_loop(self):
        """The main loop for running the simulation and emitting updates."""
        self.is_running = True
        print("Starting simulation loop...")
        while self.is_running:
            self.simulation.step()
            
            # Downsample the grid for visualization to reduce data transfer size
            scale = 4
            max_val = np.max(np.abs(self.simulation.p))
            
            # Convert to list for JSON serialization before emitting
            grid_data = self.simulation.p[::scale, ::scale].tolist()
            
            await self.sio.emit('simulation_update', {'grid': grid_data, 'max_val': max_val})
            await asyncio.sleep(0.01) # Control the update rate

    def start(self):
        """Starts the simulation background task."""
        if not self.is_running and self.simulation:
            self.task = self.sio.start_background_task(self.run_simulation_loop)
        else:
            print("Simulation is already running or not configured.")

    def stop(self):
        """Stops the simulation background task."""
        self.is_running = False
        if self.task:
            print("Stopping simulation loop...")
            # self.task.cancel() # Optionally cancel the task
            self.task = None

# --- FastAPI and Socket.IO Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

sim_manager = SimulationManager(sio)

@app.get("/")
async def read_root():
    return {"message": "Acoustic Simulation Server is running"}

# --- Socket.IO Event Handlers ---
@sio.event
async def connect(sid, environ):
    print(f"Socket connected: {sid}")
    # On connection, set up a default simulation
    default_config = {'grid_shape': [200, 200], 'driver_pos': [50, 50]}
    sim_manager.setup_simulation(default_config)
    await sio.emit('message', to=sid, data={'data': 'Welcome! Simulation is configured.'})

@sio.event
async def disconnect(sid):
    print(f"Socket disconnected: {sid}")
    # Ensure the simulation stops if the client disconnects
    sim_manager.stop()

@sio.event
async def start_simulation(sid, data):
    """Starts the simulation."""
    print(f"Received start command from {sid} with data: {data}")
    sim_manager.start()

@sio.event
async def stop_simulation(sid, data):
    """Stops the simulation."""
    print(f"Received stop command from {sid} with data: {data}")
    sim_manager.stop()

@sio.event
async def update_config(sid, config):
    """Updates the simulation configuration."""
    if not sim_manager.is_running:
        print(f"Updating config from {sid}: {config}")
        sim_manager.setup_simulation(config)
    else:
        print("Cannot update config while simulation is running.")

