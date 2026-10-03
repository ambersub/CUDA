# Comprehensive CFD Solver Benchmark & Performance Data Report

**Case Study**: 3D Compressible Supersonic Wall-Bounded Turbulent Channel Flow Direct Numerical Simulation (DNS)  
**Flow Parameters**: $\text{Re} = 3000$, $\text{Ma} = 1.5$, Mesh: $160 \times 130 \times 130 = 2,704,000$ cells  
**Hardware**: 4 CPU Cores (Fortran MPI) vs 1 NVIDIA GPU (Python Numba CUDA)

---

## 1. Executive Summary & Dynamic Time Differences

### Flow Solver Execution Time Comparison Across Step Scales

| Step Count Scale | Fortran 90 MPI Solver (4 CPU Cores) | Python Numba CUDA Solver (1 GPU) | Time Saved on GPU ($\Delta T$) | Time Reduction (%) | GPU Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Step** | **0.4753 s** | **0.3102 s** | **0.1651 s** | **34.73%** | **1.532×** |
| **10 Steps** | **4.753 s** | **3.102 s** | **1.651 s** | **34.73%** | **1.532×** |
| **50 Steps** | **23.765 s** | **15.510 s** | **8.255 s** | **34.73%** | **1.532×** |
| **100 Steps** | **47.530 s** | **31.020 s** | **16.510 s** | **34.73%** | **1.532×** |
| **500 Steps** | **237.650 s** *(3m 57.65s)* | **155.100 s** *(2m 35.10s)* | **82.550 s** *(1m 22.55s)* | **34.73%** | **1.532×** |
| **1,000 Steps** | **475.300 s** *(7m 55.30s)* | **310.200 s** *(5m 10.20s)* | **165.100 s** *(2m 45.10s)* | **34.73%** | **1.532×** |

### Time Saved for 1,000 Time Steps
$$\text{Fortran 90 MPI Time (1,000 steps)} = 1000 \times 0.4753\text{ s} = \mathbf{475.30\text{ seconds}}\quad (7\text{ min } 55.3\text{ sec})$$
$$\text{Python CUDA GPU Time (1,000 steps)} = 1000 \times 0.3102\text{ s} = \mathbf{310.20\text{ seconds}}\quad (5\text{ min } 10.2\text{ sec})$$
$$\mathbf{\Delta T_{1,000} = 475.30\text{ s} - 310.20\text{ s} = 165.10\text{ seconds saved on GPU}\quad (2\text{ min } 45.10\text{ sec faster})}$$

---

## 2. Post-Processing Analysis Timings & Time Differences

Re-compiled Fortran 90 binaries (`gfortran 16.1.0` MinGW-w64 UCRT) vs Vectorized CUDA / Python processors executing live on full $2,704,000$ cell domain fields:

| Analysis Operation | Fortran 90 Binary (`gfortran`) | CUDA / Python Processor | Time Saved ($\Delta T$) | Time Reduction (%) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vortex $Q$-Criterion (`qInvariant`)** | **35.948 s** | **7.977 s** | **27.971 s saved** | **77.81%** | **4.506×** |
| **Fluctuation Fields (`fluc`)** | **63.807 s** | **7.148 s** | **56.659 s saved** | **88.79%** | **8.926×** |
| **Full 5 Datasets Batch Processor** | -- | **7.225 s** | -- | -- | -- |

---

## 3. Comprehensive Data Summary: Input Datasets

Summary of quantitative physical properties extracted from all 5 raw datasets in `input/`:

| Dataset File | File Size | Dimensions / Rows | Key Quantities Extracted | Numerical Values |
| :--- | :--- | :--- | :--- | :--- |
| [`input/3D.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input/3D.dat) | $527\text{ MB}$ | $160 \times 130 \times 130$ ($2,704,000$ cells) | Bulk Density ($\rho_b$)<br>Bulk Velocity ($U_b$)<br>Wall Shear Stress ($\tau_w$)<br>Skin Friction ($C_f$) | $\rho_b = 1.054516$<br>$U_b = 0.834480$<br>$\tau_w = 3.5767 \times 10^{-3}$<br>$C_f = 9.7416 \times 10^{-3}$ |
| [`input/Prime3D.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input/Prime3D.dat) | $527\text{ MB}$ | $160 \times 130 \times 130$ ($2,704,000$ cells) | Peak Streamwise Turbulence ($u'_{rms}$)<br>Max TKE ($k_{max}$) | $u'_{rms, max} = 0.182934$ at $z = 1.9070$<br>$k_{max} = 0.018071$ |
| [`input/PrimeUV.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input/PrimeUV.dat) | $127\text{ MB}$ | $2,704,000$ rows | Streamwise $u'$ mean & std<br>Wall-normal $v'$ mean & std<br>Cross Covariance $\langle u'v' \rangle$ | $\bar{u}' = 6.4462 \times 10^{-3}$, $\sigma_u = 0.107919$<br>$\bar{v}' = -1.6547 \times 10^{-4}$, $\sigma_v = 0.034020$<br>$\langle u'v' \rangle = 2.3228 \times 10^{-4}$ |
| [`input/RMSu2v2w2t2uvvt.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input/RMSu2v2w2t2uvvt.dat) | $15\text{ KB}$ | $119$ wall-normal points | Peak $u'_{rms}$<br>Peak $v'_{rms}$<br>Peak $w'_{rms}$<br>Peak $T'_{rms}$ | $0.415054$ at $z = 0.0913$<br>$0.258128$<br>$0.223537$<br>$0.234725$ |
| [`input/rhouvwtpmu.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input/rhouvwtpmu.dat) | $15\text{ KB}$ | $119$ wall-normal points | Friction Velocity ($u_\tau$)<br>Viscous Scale ($\nu_w$)<br>Log-Law Fit Parameters | $u_\tau = 0.054624$<br>$\nu_w = 2.4602 \times 10^{-4}$<br>$u^+ = \frac{1}{0.302}\ln z^+ + 4.56$ |

---

## 4. Field Accuracy & Error Metrics: Fortran vs CUDA

Field-by-field verification metrics between Fortran output (`output_fortran/3D_fortran.dat`) and CUDA output (`output_cuda/3D_cuda.dat`) across all $2,704,000$ points:

| Variable Symbol | Variable Name | Fortran Mean | CUDA Mean | Mean Absolute Error ($L_1$) | Relative $L_2$ Error | Max Error ($L_\infty$) | Pearson Correlation ($r$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rho` | Density | $1.0545305$ | $1.0545228$ | $9.712948 \times 10^{-3}$ | **$0.009211$** ($0.92\%$) | $0.135474$ | **$0.996567$** |
| `u` | Streamwise Velocity | $0.8344915$ | $0.8345463$ | $3.615284 \times 10^{-2}$ | **$0.043323$** ($4.33\%$) | $0.393946$ | **$0.994887$** |
| `v` | Spanwise Velocity | $-1.243997 \times 10^{-3}$ | $-1.241848 \times 10^{-3}$ | $7.563512 \times 10^{-4}$ | $0.608004$ | $0.207830$ | $0.815007$ |
| `w` | Wall-Normal Velocity | $-8.407741 \times 10^{-5}$ | $-1.288841 \times 10^{-4}$ | $5.565983 \times 10^{-5}$ | $0.661962$ | $0.207257$ | $0.781301$ |
| `e` | Total Energy | $1.8471873$ | $1.8472607$ | $5.231502 \times 10^{-2}$ | **$0.028321$** ($2.83\%$) | $0.510693$ | **$0.992615$** |
| `T` | Temperature | $1.3077164$ | $1.3078864$ | $8.241192 \times 10^{-3}$ | **$0.006302$** ($0.63\%$) | $0.135735$ | **$0.997733$** |
| `p` | Pressure | $0.4332223$ | $0.4332907$ | $3.542150 \times 10^{-3}$ | **$0.008176$** ($0.82\%$) | $0.059372$ | $0.770550$ |

### Extracted 1D Profile Comparison
- **Mean Streamwise Velocity $\bar{u}(z)$ Profile Relative Error**: **$0.05868\%$** ($5.8677 \times 10^{-4}$)
- **Mean Density $\bar{\rho}(z)$ Profile Relative Error**: **$0.01151\%$** ($1.1508 \times 10^{-4}$)

---

## 5. Execution Commands Cheatsheet

```powershell
# 1. Process all 5 input datasets in input/
python process_input_files.py --input-dir input --output-dir output_processed

# 2. Re-compile Fortran codes from scratch, delete old outputs, and calculate live
python recompile_and_run_fortran.py

# 3. Run dynamic GPU vs Fortran MPI solver time benchmark
python benchmark_solvers.py --steps 50

# 4. Run CUDA GPU solver for 100 RK2 time steps
python "cuda code.py" --steps 100 --output output_cuda/3D_cuda.dat

# 5. Compare Fortran and CUDA solution field outputs and error metrics
python compare_fortran_cuda.py --fortran-dir output_fortran --cuda-dir output_cuda --output-dir output_comparison
```
