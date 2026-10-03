# Solver Codebase Directory

This directory (`codes/`) contains the Fortran 90 implementation of a 3D compressible turbulent channel flow Direct Numerical Simulation (DNS) solver with active opposition flow control. It also includes statistical post-processing routines, data analysis utilities, and GNUplot configuration scripts.

---

## Directory Structure & File Summary

### 1. Main Solver & Core Simulation Modules (`.f90`)
The main solver uses a modular Fortran 90 layout with MPI domain decomposition and 2nd-order Runge-Kutta (RK2) time stepping.

| File | Description |
| :--- | :--- |
| [`main_v2.0_08_oc+v_st_3.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/main_v2.0_08_oc+v_st_3.f90) | **Main Entry Point (`program chanCompr`)**: Driver for the simulation loop. Coordinates RK2 sub-stepping, MPI boundary exchanges, active flow control calls (`OC()`), and statistical data output. |
| [`mod_comdata.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_comdata.f90) | **Mesh Initialization & Global Data**: Reads runtime parameters from `../param.dat` (`readParam`), generates the 3D grid with tanh wall refinement (`setupMesh`), and computes finite-difference stencil coefficients and metric Jacobians. |
| [`mod_var_dec.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_var_dec.f90) | **Variable Declarations**: Sets grid size defaults (`nP = [192, 130, 160]`), physical flow parameters ($\gamma, \text{Pr}, \text{Re}, \text{Ma}$), state arrays (`var`, `solV`), coordinate metrics, and statistical accumulation buffers. |
| [`mod_bc.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_bc.f90) | **Boundary Conditions**: Applies non-reflective, no-slip, isothermal/adiabatic wall boundary conditions (`bc_update_2D`, `bc_update_3D`) and computes wall derivatives. |
| [`mod_control.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_control.f90) | **Active Opposition Flow Control (`OC()`)**: Implements active wall blowing and suction driven by sensed velocity fluctuations at a target distance $y^+$ from the wall to reduce drag. |
| [`mod_conv.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_conv.f90) | **Convective Flux Discretization**: Calculates convective terms using 4th-order central differences with adaptive upwinding based on the local Peclet number (`Pe`). |
| [`mod_diff.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_diff.f90) | **Diffusive Flux Discretization**: Computes viscous shear stress tensor components, heat fluxes, and viscous dissipation rates in 2D and 3D flux forms. |
| [`mod_in_out.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_in_out.f90) | **File I/O & Checkpoints**: Handles reading initial condition fields (`read_input_files`), writing restart files (`write_output_files`), and checking for solution divergence (`check_blown`). |
| [`mod_my_mpi.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_my_mpi.f90) | **MPI Parallel Architecture**: Manages 3D Cartesian domain partitioning, process rank mappings, non-blocking halo exchanges between neighbors, and MPI topology setup. |
| [`mod_stats.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/mod_stats.f90) | **On-the-fly Statistics**: Computes runtime statistics including Reynolds stresses, Favre averages, TKE budget terms, skin friction ($C_f$), wall shear stress ($\tau_w$), and active control power input (`power_OC`). |
| [`comdata.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/comdata.f90) | **Shared Data Definitions**: Helper module declaring shared flow variables and grid parameters. |

---

### 2. Post-Processing & Statistical Analysis Tools (`.f90`)
Standalone routines used to process saved flow snapshots and calculate detailed turbulence statistics.

| File | Description |
| :--- | :--- |
| [`avgTauWall.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avgTauWall.f90) | Averages wall shear stress ($\tau_w$) over time and homogeneous planes. |
| [`avgTauWallT.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avgTauWallT.f90) | Calculates time-averaged wall temperature and wall thermal flux profiles. |
| [`avg_Cf.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_Cf.f90) | Computes skin friction coefficient ($C_f$) distributions along the wall. |
| [`avg_Cf_rho.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_Cf_rho.f90) | Calculates density-scaled skin friction coefficient profiles ($C_{f,\rho}$). |
| [`avg_FIK.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_FIK.f90) | Performs Fukagata-Iwamoto-Kasagi (FIK) identity decomposition to break down total drag into laminar, turbulent Reynolds stress, and control contributions. |
| [`avg_Favre.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_Favre.f90) | Computes density-weighted Favre averages ($\tilde{u}, \tilde{v}, \tilde{w}, \tilde{T}$) for compressible turbulence analysis. |
| [`avg_Omega.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_Omega.f90) | Calculates vorticity, enstrophy, and $\Omega$-criterion vortex identification statistics across the domain. |
| [`avg_Omega_haf.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_Omega_haf.f90) | Calculates half-channel vorticity and vortex identification statistical averages. |
| [`avg_TKE.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_TKE.f90) | Calculates Turbulent Kinetic Energy (TKE) profiles, Reynolds stress components ($u'u', v'v', w'w', u'v'$), and energy transport terms. |
| [`avg_v1.7.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_v1.7.f90) | General utility for computing mean velocity, density, pressure, and temperature profiles. |
| [`avg_v1.7_TKE_ALL.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_v1.7_TKE_ALL.f90) | Evaluates the full TKE budget terms (production, dissipation, turbulent diffusion, and pressure strain). |
| [`avg_VD.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_VD.f90) | Computes viscous dissipation rate distribution profiles across the boundary layer. |
| [`avg_local.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_local.f90) | Extracts localized flow field statistics at specified probe coordinates. |
| [`avg_tauWall_Lxt.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_tauWall_Lxt.f90) | Evaluates space-time correlations of wall shear stress ($L_{xt}$). |
| [`avg_xPlane.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_xPlane.f90) | Calculates plane-averaged flow quantities at fixed streamwise ($x$) positions. |
| [`avg_xtCorr.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_xtCorr.f90) | Computes two-point space-time velocity correlation functions ($R_{uu}, R_{vv}, R_{ww}$). |
| [`avg_xtCorrALL.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/avg_xtCorrALL.f90) | Evaluates two-point space-time correlations across velocity, pressure, and temperature fluctuations. |
| [`combine.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/combine.f90) | Merges sub-domain partitioned MPI output files into consolidated full-domain snapshot data files. |
| [`fluc.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/fluc.f90) | Computes instantaneous velocity, density, and pressure fluctuation components ($u', v', w', \rho', p'$). |
| [`fluc_prime.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/fluc_prime.f90) | Normalizes fluctuation variables using wall units or friction velocity ($u_\tau$). |
| [`fluc_rho_tau.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/fluc_rho_tau.f90) | Evaluates correlations between density and shear stress fluctuations near wall boundaries. |
| [`prime.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/prime.f90) | Isolates turbulent perturbation signals from instantaneous state snapshots. |
| [`qInvariant.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/qInvariant.f90) | Calculates the second invariant of the velocity gradient tensor ($Q$-criterion) for 3D coherent vortex visualization. |
| [`qInvariant_Omega.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/qInvariant_Omega.f90) | Computes both $Q$-criterion and $\Omega$-criterion vortex identification metrics. |

---

### 3. Plotting Scripts & GNUplot Configurations (`.txt`)
GNUplot command scripts for automated plotting and validation against benchmark DNS data (such as Coleman, Kim & Moser / CKM).

| File | Description |
| :--- | :--- |
| [`cmdGNUplot.txt`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/cmdGNUplot.txt) | Plots mean density, velocity, temperature, and pressure profiles vs. wall distance ($z$), comparing results against CKM DNS benchmark data. |
| [`cmdGNUplot2.txt`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/cmdGNUplot2.txt) | Plots RMS velocity fluctuations ($u'_{rms}, v'_{rms}, w'_{rms}$), wall shear stress history, and mass flow rate. |
| [`cmdGNUplot_TKE.txt`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/cmdGNUplot_TKE.txt) | Plots Turbulent Kinetic Energy (TKE) budget profiles (production, dissipation, pressure strain, diffusion). |
| [`cmdGNUplot_TKE_Fav.txt`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/cmdGNUplot_TKE_Fav.txt) | Plots Favre-averaged TKE budget components across the channel height. |

---

### 4. Data Input & Control Files (`.dat`)
Control configurations specifying sampling parameters and plot formatting.

| File | Description |
| :--- | :--- |
| [`AoNDT.dat`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/AoNDT.dat) | Sampling count parameters for general statistical averaging routines. |
| [`AoNDT_Cf.dat`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/AoNDT_Cf.dat) | Sampling configurations for skin friction ($C_f$) averaging. |
| [`AoNDT_Favre.dat`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/AoNDT_Favre.dat) | Sampling configurations for Favre-averaged statistics. |
| [`AoNDT_Omega.dat`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/AoNDT_Omega.dat) | Sampling parameters for vorticity and $\Omega$-criterion averaging. |
| [`AoNDT_v1.7.dat`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/AoNDT_v1.7.dat) | Sampling control file for `avg_v1.7.f90`. |
| [`AoNDT_v1.7_TKE_ALL.dat`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/AoNDT_v1.7_TKE_ALL.dat) | Sampling parameters for full TKE budget averaging. |
| [`plotInfo.dat`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/plotInfo.dat) | Axis bounds, label configurations, and scaling factors for plot outputs. |

---

### 5. Legacy Code (`old/`)
Subdirectory containing older iterations of the solver.

| File | Description |
| :--- | :--- |
| [`old/main_v1.9_unc.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/old/main_v1.9_unc.f90) | Legacy main entry point solver (v1.9) for baseline uncontrolled compressible channel flow. |
| [`old/old_osc_main_v1.9_unc.f90`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/old/old_osc_main_v1.9_unc.f90) | Legacy main solver (v1.9) incorporating spanwise wall oscillation boundary conditions. |
| [`old/main_v1.9_unc.o`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/old/main_v1.9_unc.o) | Compiled object file for legacy solver v1.9. |
| [`old/main_v1.9_unc.x`](file:///c:/Users/Amber/Downloads/CUDA%20solver/codes/old/main_v1.9_unc.x) | Binary executable for legacy solver v1.9. |

---

### 6. Binary Executables, Objects & Intermediate Files
Compiled binaries, module interfaces, and object files generated during compilation.

* **Binary Executables (`.x`)**: `main_v2.0_08_oc+v_st_3.x`, `avgTauWall.x`, `avgTauWallT.x`, `avg_Cf.x`, `avg_Cf_rho.x`, `avg_FIK.x`, `avg_Favre.x`, `avg_Omega.x`, `avg_Omega_haf.x`, `avg_TKE.x`, `avg_VD.x`, `avg_local.x`, `avg_tauWall_Lxt.x`, `avg_v1.7.x`, `avg_v1.7_TKE_ALL.x`, `avg_xPlane.x`, `avg_xtCorr.x`, `avg_xtCorrALL.x`, `combine.x`, `fluc.x`, `fluc_prime.x`, `fluc_rho_tau.x`, `prime.x`, `qInvariant.x`, `qInvariant_Omega.x`.
* **Object Files (`.o`)**: `main_v2.0_08_oc+v_st_3.o`, `mod_bc.o`, `mod_comdata.o`, `mod_control.o`, `mod_conv.o`, `mod_diff.o`, `mod_in_out.o`, `mod_my_mpi.o`, `mod_stats.o`, `mod_var_dec.o`, `comdata.o`, `avg_v1.7.o`, `qInvariant.o`.
* **Module Interfaces (`.mod`)**: `mod_bc.mod`, `mod_comdata.mod`, `mod_control.mod`, `mod_conv.mod`, `mod_diff.mod`, `mod_in_out.mod`, `mod_my_mpi.mod`, `mod_stats.mod`, `mod_var_dec.mod`, `comdata.mod`.
* **Swap Files**: `.avg_v1.7.x.swp` (Vim swap file).

