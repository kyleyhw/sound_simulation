# Project Plan: Real-Time Acoustic Control System

## 1. Project Vision

The ultimate goal is to develop a closed-loop acoustic control system capable of sensing an environment's acoustic properties and actively manipulating its sound field. This includes "virtual headphones" (directed audio) and directed noise cancellation.

## 2. Guiding Principles

*   **Intermediate Deliverables:** We will prioritize the creation of tangible, usable tools at each stage of the project. This allows for continuous validation and experimentation.
*   **Cross-Platform Compatibility:** All user-facing components, such as interactive UIs, will be browser-based to ensure they run on various operating systems.
*   **Consumer Hardware Constraint:** The final system must be computationally efficient enough to run in real-time on standard consumer hardware (laptops, phones).

## 3. Phased Development Plan

Work on each task or phase will only commence with explicit user permission.

---

### Phase 1: Interactive Browser-Based Simulation UI
**Objective:** Create the first major deliverable: a web-based user interface to interactively define, run, and visualize simulations in real-time.

*   `[ ]` **Task 1.1: Technology Stack Selection & Setup**
    *   `[ ]` 1.1.1: **Backend:** Establish a Python-based web server using a lightweight framework (e.g., FastAPI or Flask) to manage and run the simulation.
    *   `[ ]` 1.1.2: **Communication:** Implement a WebSocket protocol for low-latency, bidirectional communication between the browser frontend and the simulation backend.
    *   `[ ]` 1.1.3: **Frontend:** Set up a modern JavaScript framework (e.g., React, Vue, or Svelte) to build the interactive UI components.
*   `[ ]` **Task 1.2: Backend Development for Simulation Control**
    *   `[ ]` 1.2.1: Develop a WebSocket endpoint to receive commands from the UI (e.g., `start_simulation`, `add_obstacle`, `place_driver`).
    *   `[ ]` 1.2.2: Refactor the core FDTD simulation logic (`Simulate` class) to run in a separate, non-blocking thread and be controlled by the web server.
    *   `[ ]` 1.2.3: Implement logic to periodically stream the simulation state (e.g., the pressure grid) to the UI for visualization.
*   `[ ]` **Task 1.3: Frontend Development for Interactive UI**
    *   `[ ]` 1.3.1: Create a primary UI component: an interactive canvas to render the simulation grid received from the backend.
    *   `[ ]` 1.3.2: Implement client-side logic for drawing boundaries/obstacles and placing drivers/sensors with mouse interactions.
    *   `[ ]` 1.3.3: Develop UI controls (e.g., buttons, sliders) to start, stop, reset the simulation, and adjust parameters.
*   `[ ]` **Task 1.4: Enable Live Interaction (Advanced)**
    *   `[ ]` 1.4.1: Modify the core simulation engine to allow for the addition or removal of objects (obstacles, drivers) *during* a run, without requiring a full reset.
    *   `[ ]` 1.4.2: Connect the UI and backend to support these live modifications, providing a truly dynamic interaction experience.

---

### Phase 2: Advanced Sensing (Developing the "Ear")
**Objective:** Achieve high-fidelity, real-time room mapping within the simulation environment, using the UI from Phase 1 for testing.

*   `[ ]` **Task 2.1 (Active Sensing): Integrate & Model Complex Audio Sources**
    *   `[ ]` 2.1.1: Modify simulation to use audio files as sound sources and generate a corresponding dataset.
    *   `[ ]` 2.1.2: Design and implement a dual-input CNN that accepts a microphone recording and a source audio reference.
    *   `[ ]` 2.1.3: Train and validate the active sensing model.
*   `[ ]` **Task 2.2 (Passive Sensing): Research & Model Blind Deconvolution**
    *   `[ ]` 2.2.1: Research and implement a model architecture suitable for blind deconvolution (e.g., Autoencoder, RNN).
    *   `[ ]` 2.2.2: Train and validate the passive model, comparing its performance to the active model.

---

### Phase 3: Beamforming Simulation (Developing the "Mouth")
**Objective:** Create the algorithms and simulation environment for directed audio using a speaker array.

*   `[ ]` **Task 3.1: Implement Speaker Array and Control Algorithm**
    *   `[ ]` 3.1.1: Modify the simulation to support multiple, independently controlled drivers.
    *   `[ ]` 3.1.2: Develop a `control/beamformer.py` module with a foundational beamforming algorithm.
*   `[ ]` **Task 3.2: Validate Beamforming in Simulation**
    *   `[ ]` 3.2.1: Create test scenarios using the interactive UI to place a speaker array and target zones.
    *   `[ ]` 3.2.2: Verify the creation of localized sound and silence zones.

---

### Phase 4: Closed-Loop System Integration
**Objective:** Connect the "Ear" and "Mouth" in a real-time feedback loop within the simulation.

*   `[ ]` **Task 4.1: Develop Integrated Simulation Environment**
    *   `[ ]` 4.1.1: Create a master script or enhance the web backend to manage the full sense-model-act loop.
*   `[ ]` **Task 4.2: Connect Inference to Control**
    *   `[ ]` 4.2.1: Feed the microphone data into the trained inference model (from Phase 2).
    *   `[ ]` 4.2.2: Use the model's output (inferred RIR) as the input for the beamformer (from Phase 3).
*   `[ ]` **Task 4.3: Validate Real-Time Adaptation**
    *   `[ ]` 4.3.1: Use the UI to create dynamic scenarios (e.g., move a target or an obstacle).
    *   `[ ]` 4.3.2: Verify that the system adapts the directed audio beam in response to the changes.