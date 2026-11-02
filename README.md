# PEPG_simulation_thin_diffuser
MATLAB simulation of wavefront shaping with Parameter-Exploring Policy Gradients (PEPG). The algorithm optimizes the phase of a spatial light modulator (SLM) to produce a target camera intensity pattern through a random phase screen, without measuring a transmission matrix.
Requirements
- MATLAB with Parallel Computing Toolbox and a compatible GPU
- PEPG_opt_perturbed_GPU.m in the MATLAB path
Usage
1. Set the parameters at the top of PEPG_run.m.
2. Choose target_type: 'gaussian', 'checker', or 'image' (provide target_file for an image).
3. Run PEPG_run.m.
The script displays the target, simulated camera intensity, SLM phase, and optimization loss during training.
Results
Each run creates a timestamped pepg_* folder containing the optimized intensity and phase images, MATLAB figures, and a .mat file with the optimization history and parameters.

Note: The optical system is modeled as fixed thin phase screens followed by Fourier propagation; it does not simulate a full multimode fiber or a dynamically changing material.


## Acknowledgments

This project adapts parts of the optimization code from [ASkalli/learning_strategies](https://github.com/ASkalli/learning_strategies),
