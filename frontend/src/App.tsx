import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

const socket = io('http://127.0.0.1:8000'); // Connect to the Python backend

// --- Main App Component ---
function App() {
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [isRunning, setIsRunning] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Effect for handling socket connection events
  useEffect(() => {
    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));

    // Listen for simulation updates from the backend
    socket.on('simulation_update', (data: { grid: number[][], max_val: number }) => {
      drawGrid(data.grid, data.max_val);
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('simulation_update');
    };
  }, []);

  // --- Rendering Logic ---
  const drawGrid = (grid: number[][], maxVal: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const gridHeight = grid.length;
    if (gridHeight === 0) return;
    const gridWidth = grid[0].length;
    if (gridWidth === 0) return;

    const cellWidth = canvas.width / gridWidth;
    const cellHeight = canvas.height / gridHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const absMax = Math.max(0.001, maxVal);

    for (let y = 0; y < gridHeight; y++) {
      for (let x = 0; x < gridWidth; x++) {
        const value = grid[y][x];
        const normalizedValue = value / absMax; // Normalize to [-1, 1]

        let r=0, g=0, b=0;
        if (normalizedValue > 0) {
            r = 255 * normalizedValue;
            g = 255 * normalizedValue;
            b = 255;
        } else {
            r = 255;
            g = 255 * (1 + normalizedValue);
            b = 255 * (1 + normalizedValue);
        }

        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
      }
    }
  };

  // --- UI Event Handlers ---
  const handleStart = () => {
    console.log('Sending start command');
    socket.emit('start_simulation', {});
    setIsRunning(true);
  };

  const handleStop = () => {
    console.log('Sending stop command');
    socket.emit('stop_simulation', {});
    setIsRunning(false);
  };

  const handleReset = () => {
    console.log('Sending reset/config command');
    handleStop(); // Stop first
    // This could be expanded to send new config data from UI inputs
    const defaultConfig = {'grid_shape': [200, 200], 'driver_pos': [50, 50]};
    socket.emit('update_config', defaultConfig);
  };

  return (
    <div className="App">
      <header>
        <h1>Acoustic Simulation</h1>
        <p>Connection Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      </header>
      <div className="controls">
        <button onClick={handleStart} disabled={isRunning}>Start</button>
        <button onClick={handleStop} disabled={!isRunning}>Stop</button>
        <button onClick={handleReset} disabled={isRunning}>Reset</button>
      </div>
      <div className="canvas-container">
        <canvas 
          ref={canvasRef} 
          width={800} 
          height={800} 
          style={{ border: '1px solid white' }}
        />
      </div>
    </div>
  );
}

export default App;
