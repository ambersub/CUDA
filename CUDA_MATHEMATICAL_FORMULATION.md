# Mathematical Formulation & Code Implementation Trace of the CUDA CFD Solver

This document provides a comprehensive, rigorous mathematical description of the **3D Compressible Supersonic Turbulent Channel Flow Direct Numerical Simulation (DNS)** solver. For every governing equation, numerical stencil, thermodynamic relation, and boundary law, this document states the **exact file path, function/kernel name, source lines, variables, and GPU execution context**.

---

## Table of Contents
1. [Governing Navier-Stokes System & State Vectors](#1-governing-navier-stokes-system--state-vectors)
2. [Thermodynamic Relations, EOS & Fluid Properties](#2-thermodynamic-relations-eos--fluid-properties)
3. [Computational Domain, Grid Stretching & Metric Jacobians](#3-computational-domain-grid-stretching--metric-jacobians)
4. [Inviscid Convective Flux Discretization & Adaptive Peclet Limiter](#4-inviscid-convective-flux-discretization--adaptive-peclet-limiter)
5. [Viscous Diffusive Flux Discretization & Dissipation](#5-viscous-diffusive-flux-discretization--dissipation)
6. [Driving Source Term & Body Forcing](#6-driving-source-term--body-forcing)
7. [Explicit Two-Stage Runge-Kutta (RK2) Time Stepping](#7-explicit-two-stage-runge-kutta-rk2-time-stepping)
8. [Boundary Conditions & Active Opposition Flow Control (OC)](#8-boundary-conditions--active-opposition-flow-control-oc)
9. [Numerical Stability & Solution Sanity Check](#9-numerical-stability--solution-sanity-check)
10. [Diagnostic Turbulence Statistics & Post-Processing Mathematics](#10-diagnostic-turbulence-statistics--post-processing-mathematics)
11. [CUDA Architecture Execution Hierarchy & Kernel Mapping](#11-cuda-architecture-execution-hierarchy--kernel-mapping)

---

## 1. Governing Navier-Stokes System & State Vectors

The solver integrates the unsteady, three-dimensional, compressible Navier-Stokes equations in conservative divergence form:

$$\frac{\partial \mathbf{q}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} + \frac{\partial \mathbf{H}}{\partial z} = \frac{\partial \mathbf{F}_D}{\partial x} + \frac{\partial \mathbf{G}_D}{\partial y} + \frac{\partial \mathbf{H}_D}{\partial z} + \mathbf{S}$$

### 1.1. Conserved State Vector ($\mathbf{q}$)
The conserved state vector per unit volume $\mathbf{q} \in \mathbb{R}^5$ is:

$$\mathbf{q} = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ q_4 \end{bmatrix} = \begin{bmatrix} \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho E \end{bmatrix}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L89-L96)
> - **Lines**: Lines 89–96 (constants `NQ = 5`), lines 326–332 (`read_combined_input`), lines 351–357 (`demo_initial_condition`)
> - **GPU Array**: `self.d_q` and `self.d_q_stage` (shape: `(nx, ny, nz, 5)`, `float64`) allocated at lines 1033–1034
> - **Hardware Context**: Allocated in GPU global VRAM; persists across all RK2 time steps to eliminate host-device data transfers.

---

### 1.2. Primitive & Working Variables Recovery ($\mathbf{v}$)
The primitive fluid state vector $\mathbf{v} \in \mathbb{R}^7$ represents the physical observables:

$$\mathbf{v} = [v_0, v_1, v_2, v_3, v_4, v_5, v_6]^T = [\rho, u, v, w, E, T, P]^T$$

The primitive components are extracted from the conserved vector $\mathbf{q}$ via:

$$\rho = \max(q_0, 10^{-14})$$
$$u = \frac{q_1}{\rho}, \quad v = \frac{q_2}{\rho}, \quad w = \frac{q_3}{\rho}, \quad E = \frac{q_4}{\rho}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L360-L394)
> - **Function / Kernel**: `@cuda.jit def primary_variables_kernel(q, var, mu, kt, ggM, gamma, mach)`
> - **Lines**: Lines 375–382 (`rho = max(q[i,j,k,0], 1.0e-14)`, `u = q[i,j,k,1]/rho`, etc.)
> - **GPU Array**: `self.d_var` (shape: `(nx, ny, nz, 7)`, `float64`) allocated at line 1039
> - **Launch Configuration**: Launched with 3D thread blocks `tpb3 = (8, 8, 8)` inside `update_primitive()` (line 1074).

---

## 2. Thermodynamic Relations, EOS & Fluid Properties

### 2.1. Equation of State (EOS) & Temperature Inversion
The flow is modeled as an ideal gas with constant specific heat ratio $\gamma = 1.4$.

1. **Specific Kinetic Energy ($k$)**:
   $$k = \frac{1}{2} (u^2 + v^2 + w^2)$$

2. **Static Temperature ($T$)**:
   $$T = E - \Gamma_{gM} \cdot k$$
   where $\Gamma_{gM}$ is the Mach compressibility parameter:
   $$\Gamma_{gM} \equiv \gamma (\gamma - 1) \text{Ma}^2$$

3. **Static Pressure ($P$)**:
   $$P = \frac{\rho T}{\gamma \text{Ma}^2}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L145-L147) and [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L383-L393)
> - **Config Constant**: Property `ggM` defined at lines 145–147: `return GAMMA * (GAMMA - 1.0) * self.mach * self.mach`
> - **Kernel Implementation**: Lines 383–386 in `primary_variables_kernel`:
>   ```python
>   kinetic = 0.5 * (u * u + v * v + w * w)
>   T = E - ggM * kinetic
>   P = rho * T / (gamma * mach * mach)
>   ```
> - **Wall Density Inversion**: Used in `opposition_and_wall_bc_kernel` at lines 956–967 to reconstruct $\rho$ from wall pressure:
>   $$\rho_{\text{wall}} = \frac{\gamma \text{Ma}^2 P_{\text{wall}}}{T_{\text{wall}}}$$

---

### 2.2. Power-Law Dynamic Viscosity & Thermal Conductivity
The dynamic molecular viscosity $\mu$ is evaluated using a non-dimensional power-law model with exponent $n = 0.7$:

$$\mu(T) = \left[ \max(T, 10^{-12}) \right]^{0.7}$$

The thermal conductivity $k_t$ is proportional to viscosity under constant Prandtl number $\text{Pr} = 0.71$:

$$k_t(T) = \mu(T), \quad \text{Pr}^{-1} = \frac{1}{0.71} \approx 1.40845$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L82-L83) and [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L395-L397)
> - **Constants**: `GAMMA = 1.4`, `INV_PR = 1.0 / 0.71` (lines 82–83)
> - **Kernel Lines**: Lines 395–397 in `primary_variables_kernel`:
>   ```python
>   mu[i, j, k] = max(T, 1.0e-12) ** 0.7
>   kt[i, j, k] = mu[i, j, k]
>   ```
> - **GPU Arrays**: `self.d_mu` and `self.d_kt` (shape: `(nx, ny, nz)`, `float64`) allocated at lines 1040–1041.

---

## 3. Computational Domain, Grid Stretching & Metric Jacobians

### 3.1. Physical Coordinates & Domain Discretization
The computational channel domain spans:
* Streamwise: $x \in [0, 4\pi] \approx [0, 12.56637]$, grid spacing $\Delta x = \frac{4\pi}{N_x - 1}$
* Spanwise: $y \in [0, \frac{4\pi}{3}] \approx [0, 4.18879]$, grid spacing $\Delta y = \frac{4\pi / 3}{N_y - 1}$
* Wall-Normal: $z \in [0, 2.0]$ (channel half-width $h = 1.0$)

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L161-L170)
> - **Function**: `def make_mesh(cfg: SolverConfig)` (lines 161–170)
> - **Variables**: `x1` ($x$-grid), `x2` ($y$-grid), `xi` ($\xi$-grid), `dx`, `dy`, `dxi`.

---

### 3.2. Hyperbolic Tangent Mesh Stretching
To concentrate grid points near the walls at $z=0$ and $z=2$, computational coordinates $\xi \in [0, 2]$ are transformed via:

$$z(\xi) \equiv x_3(\xi) = 1 + \frac{\tanh\left[\frac{1}{2} a (\xi - 1)\right]}{\tanh\left(\frac{1}{2} a\right)}, \quad a = \text{EP} = 5.0$$

with analytical boundary conditions $z(0) = 0.0$ and $z(2) = 2.0$.

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L87) and [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L171-L173)
> - **Lines**: Line 87 (`EP = 5.0`), lines 171–173:
>   ```python
>   x3 = 1.0 + np.tanh(0.5 * EP * (xi - 1.0)) / np.tanh(0.5 * EP)
>   x3[0] = 0.0
>   ```

---

### 3.3. Metric Transformation & Analytical Jacobian ($\mathcal{J}$)
Chain-rule transformation maps the non-uniform physical space $z$ into uniform computational space $\xi$:

$$\frac{\partial (\cdot)}{\partial z} = \frac{1}{\mathcal{J}(\xi)} \frac{\partial (\cdot)}{\partial \xi}$$

where the analytical Jacobian metric $\mathcal{J}(\xi) = \frac{dz}{d\xi}$ is:

$$\mathcal{J}(\xi) = \frac{a \left(1 - \tanh^2\left[\frac{1}{2} a (\xi - 1)\right]\right)}{2 \tanh\left(\frac{1}{2} a\right)}$$

The inverse factor $\mathcal{J}^{-1}[k] = 1 / \mathcal{J}(\xi_k)$ is pre-computed and stored on the device.

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L174-L181)
> - **Lines**: Lines 174–181 in `make_mesh`:
>   ```python
>   jac = 0.5 * EP * (1.0 - np.tanh(0.5 * EP * (xi - 1.0)) ** 2) / np.tanh(0.5 * EP)
>   inv_jac = np.zeros_like(jac)
>   inv_jac[1:] = 1.0 / jac[1:]
>   ```
> - **GPU Array**: `self.d_inv_jac` transferred to device at line 1052 and passed into `convective_kernel`, `diffusive_kernel`, and `rhs_and_rk2_kernel`.

---

### 3.4. Non-Uniform One-Sided Boundary Stencils (3rd Order)
At the bottom wall ($z=0, k=0$) and top wall ($z=2, k=N_z-1$), one-sided 4-point derivative stencils are calculated using Lagrange polynomial weights for non-uniform grid intervals $d_{ij} = z_i - z_j$:

$$\left.\frac{\partial \phi}{\partial z}\right|_{z_0} = \sum_{m=0}^3 c_m^{\text{low}} \phi_m, \quad \left.\frac{\partial \phi}{\partial z}\right|_{z_{\text{top}}} = \sum_{m=0}^3 c_m^{\text{high}} \phi_{N_z-1-m}$$

The lower boundary coefficients are:
$$c_0^{\text{low}} = \frac{d_{31} d_{21} + d_{31} d_{41} + d_{21} d_{41}}{(-d_{21})(-d_{31})(-d_{41})}, \quad c_1^{\text{low}} = \frac{d_{41} d_{31}}{d_{21}(-d_{32})(-d_{42})}$$
$$c_2^{\text{low}} = \frac{d_{41} d_{21}}{d_{31} d_{32}(-d_{43})}, \quad c_3^{\text{low}} = \frac{d_{31} d_{21}}{d_{41} d_{42} d_{43}}$$

where:
$$d_{21} = z_1 - z_0, \quad d_{31} = z_2 - z_0, \quad d_{32} = z_2 - z_1, \quad d_{41} = z_3 - z_0, \quad d_{42} = z_3 - z_1, \quad d_{43} = z_3 - z_2$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L184-L216)
> - **Functions**: `coeff_low_3(z)` (lines 184–199) and `coeff_high_3(z)` (lines 200–213)
> - **GPU Arrays**: `self.d_low3` and `self.d_high3` (lines 1053–1054)
> - **Application**: Consumed inside `diffusive_kernel` at lines 717–765 for wall-normal derivatives of velocity components and temperature.

---

## 4. Inviscid Convective Flux Discretization & Adaptive Peclet Limiter

The convective flux vector components are defined as:

$$\mathbf{F} = u \mathbf{q}, \quad \mathbf{G} = v \mathbf{q}, \quad \mathbf{H} = w \mathbf{q}$$

### 4.1. 4th-Order Central Finite-Difference Stencil
In smooth flow regions, derivatives are evaluated via the 4th-order central difference stencil:

$$\left(\frac{\partial (u \mathbf{q})}{\partial x}\right)_i = \frac{-\phi_{i+2} + 8\phi_{i+1} - 8\phi_{i-1} + \phi_{i-2}}{12 \Delta x} + \mathcal{O}(\Delta x^4)$$

where periodic indices wrap around via modular arithmetic:
$$i_{\pm 1} = (i \pm 1) \pmod{N_x}, \quad i_{\pm 2} = (i \pm 2) \pmod{N_x}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L429-L438) and [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L507-L513)
> - **Kernel**: `convective_kernel`
> - **Lines**:
>   - Index wrapping ($x, y$): lines 429–438
>   - $x$-direction central flux: lines 507–513
>   - $y$-direction central flux: lines 577–583
>   - $z$-direction central flux: lines 652–658

---

### 4.2. Local Grid Peclet Numbers
Numerical dispersion is controlled by monitoring the directional grid Peclet number:

$$\text{Pe}_x = \frac{\Delta x |\rho u|}{\mu_{ijk}}, \quad \text{Pe}_y = \frac{\Delta y |\rho v|}{\mu_{ijk}}, \quad \text{Pe}_z = \frac{\Delta \xi |\rho w| \mathcal{J}^{-1}[k]}{\mu_{ijk}}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L445)
> - **Lines**: Line 445 (`pe_x`), line 515 (`pe_y`), line 585 (`pe_z`) in `convective_kernel`:
>   ```python
>   pe = dx * abs(q[i, j, k, 1]) / max(mu[i, j, k], 1.0e-14)
>   ```
> - **Threshold Limits**: Default $\text{Pe}_x = 2.0, \text{Pe}_y = 1.0, \text{Pe}_z = 0.1$ parsed from [`param.dat`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/param.dat#L16-L18).

---

### 4.3. Directional 3rd-Order Biased Upwind Formulation
When $\text{Pe} > \text{Pe}_{\text{limit}}$ or at physical boundary planes, the solver switches to a biased upwind stencil evaluated at interface convective velocities:

$$a = \frac{1}{2}(u_i + u_{i+1}), \quad b = \frac{1}{2}(u_i + u_{i-1})$$

The flux derivative is selected based on interface velocity signs:

$$\Phi = \begin{cases}
\frac{a(6q_i + 3q_{i+1} - q_{i-1}) - b(6q_{i-1} + 3q_i - q_{i-2})}{8}, & a \ge 0, b \ge 0 \\
\frac{a(6q_{i+1} + 3q_i - q_{i+2}) - b(6q_i + 3q_{i-1} - q_{i+1})}{8}, & a < 0, b < 0 \\
\frac{a(6q_{i+1} + 3q_i - q_{i+2}) - b(6q_{i-1} + 3q_i - q_{i-2})}{8}, & a < 0, b \ge 0 \\
\frac{a(6q_i + 3q_{i+1} - q_{i-1}) - b(6q_i + 3q_{i-1} - q_{i+1})}{8}, & a \ge 0, b < 0
\end{cases}$$

$$\left(\frac{\partial (u \mathbf{q})}{\partial x}\right)_i = \frac{\Phi}{\Delta x}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L449-L506)
> - **Lines**:
>   - $x$-direction upwinding: lines 449–506
>   - $y$-direction upwinding: lines 519–576
>   - $z$-direction upwinding: lines 595–651
> - **Output Arrays**: `self.d_F`, `self.d_G`, `self.d_H` (shape: `(nx, ny, nz, 5)`, `float64`) allocated at lines 1043–1045.

---

## 5. Viscous Diffusive Flux Discretization & Dissipation

### 5.1. Velocity Gradient Tensor & Dilatation ($\Theta$)
Gradients are computed via 2nd-order central differences:

$$\frac{\partial u}{\partial x} = \frac{u_{i+1} - u_{i-1}}{2\Delta x}, \quad \frac{\partial u}{\partial y} = \frac{u_{j+1} - u_{j-1}}{2\Delta y}, \quad \frac{\partial u}{\partial z} = \mathcal{J}^{-1}[k] \frac{u_{k+1} - u_{k-1}}{2\Delta \xi}$$

Near walls ($k=0$ and $k=N_z-1$), wall-normal derivatives use the 3rd-order one-sided weights from Section 3.4.

The volumetric dilatation (fluid divergence) is:
$$\Theta \equiv \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L704-L780)
> - **Kernel**: `diffusive_kernel`
> - **Lines**: Lines 704–715 (interior gradients), lines 717–765 (wall one-sided stencils), line 780 (`divg = dudx + dvdy + dwdz`).

---

### 5.2. Viscous Stress Tensor Components ($\tau_{ij}$)
Under Stokes' hypothesis (bulk viscosity ratio $\lambda = -\frac{2}{3}\mu$), the non-dimensional viscous stress components are:

$$\tau_{xx} = \frac{\mu}{\text{Re}} \left( 2 \frac{\partial u}{\partial x} - \frac{2}{3} \Theta \right), \quad \tau_{yy} = \frac{\mu}{\text{Re}} \left( 2 \frac{\partial v}{\partial y} - \frac{2}{3} \Theta \right), \quad \tau_{zz} = \frac{\mu}{\text{Re}} \left( 2 \frac{\partial w}{\partial z} - \frac{2}{3} \Theta \right)$$

$$\tau_{xy} = \frac{\mu}{\text{Re}} \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right), \quad \tau_{xz} = \frac{\mu}{\text{Re}} \left( \frac{\partial u}{\partial z} + \frac{\partial w}{\partial x} \right), \quad \tau_{yz} = \frac{\mu}{\text{Re}} \left( \frac{\partial v}{\partial z} + \frac{\partial w}{\partial y} \right)$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L801-L818)
> - **Kernel**: `diffusive_kernel`
> - **Lines**: Embedded directly into flux definitions at lines 802–804 (`fD`), 808–810 (`gD`), and 814–816 (`hD`).

---

### 5.3. Directional Viscous Dissipation Rates ($\Phi_x, \Phi_y, \Phi_z$)
The mechanical energy dissipation rates per coordinate direction are:

$$\Phi_x = \Gamma_{gM} \frac{\mu}{\text{Re}} \left[ \frac{2}{3} u \Theta - 2 u \frac{\partial u}{\partial x} - v \left(\frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}\right) - w \left(\frac{\partial w}{\partial x} + \frac{\partial u}{\partial z}\right) \right]$$

$$\Phi_y = \Gamma_{gM} \frac{\mu}{\text{Re}} \left[ -u \left(\frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}\right) + \frac{2}{3} v \Theta - 2 v \frac{\partial v}{\partial y} - w \left(\frac{\partial w}{\partial y} + \frac{\partial v}{\partial z}\right) \right]$$

$$\Phi_z = \Gamma_{gM} \frac{\mu}{\text{Re}} \left[ -u \left(\frac{\partial w}{\partial x} + \frac{\partial u}{\partial z}\right) - v \left(\frac{\partial w}{\partial y} + \frac{\partial v}{\partial z}\right) + \frac{2}{3} w \Theta - 2 w \frac{\partial w}{\partial z} \right]$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L782-L799)
> - **Kernel**: `diffusive_kernel`
> - **Lines**: Lines 782–787 (`diss_f`), lines 788–793 (`diss_g`), lines 794–799 (`diss_h`).

---

### 5.4. Full Diffusive Flux Vectors ($\mathbf{F}_D, \mathbf{G}_D, \mathbf{H}_D$)
The 5-component viscous diffusive fluxes evaluated across the domain are:

$$\mathbf{F}_D = \begin{bmatrix} 0 \\ P - \tau_{xx} \\ -\tau_{xy} \\ -\tau_{xz} \\ -\frac{\gamma k_t}{\text{Re} \cdot \text{Pr}} \frac{\partial T}{\partial x} + \Gamma_{gM} u P + \Phi_x \end{bmatrix}, \quad \mathbf{G}_D = \begin{bmatrix} 0 \\ -\tau_{yx} \\ P - \tau_{yy} \\ -\tau_{yz} \\ -\frac{\gamma k_t}{\text{Re} \cdot \text{Pr}} \frac{\partial T}{\partial y} + \Gamma_{gM} v P + \Phi_y \end{bmatrix}, \quad \mathbf{H}_D = \begin{bmatrix} 0 \\ -\tau_{zx} \\ -\tau_{zy} \\ P - \tau_{zz} \\ -\frac{\gamma k_t}{\text{Re} \cdot \text{Pr}} \frac{\partial T}{\partial z} + \Gamma_{gM} w P + \Phi_z \end{bmatrix}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L801-L818)
> - **Kernel**: `diffusive_kernel`
> - **Output Arrays**: `self.d_fD`, `self.d_gD`, `self.d_hD` (shape: `(nx, ny, nz, 5)`, `float64`) allocated at lines 1047–1049.

---

## 6. Driving Source Term & Body Forcing

In fully developed turbulent channel flow, the mean flow is sustained against viscous wall shear stress by a constant body force $f_{\text{body}}$:

$$f_{\text{body}} = \frac{1}{2} (\tau_{\text{lower}} - \tau_{\text{upper}})$$

The resulting source vector $\mathbf{S}$ added to the governing equations is:

$$\mathbf{S} = [S_0, S_1, S_2, S_3, S_4]^T = [0, f_{\text{body}}, 0, 0, \Gamma_{gM} \cdot u \cdot f_{\text{body}}]^T$$

where $S_4$ models the mechanical work rate performed on the fluid by the body force.

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L867-L888)
> - **Kernel**: `rhs_and_rk2_kernel`
> - **Lines**:
>   - Body force definition: line 867 (`body = 0.5 * (tau_lower - tau_upper)`)
>   - Momentum injection ($S_1$): lines 884–885 (`if c == 1: source = body`)
>   - Energy injection ($S_4$): lines 886–887 (`elif c == 4: source = ggM * var[i, j, k, 1] * body`)

---

## 7. Explicit Two-Stage Runge-Kutta (RK2) Time Stepping

### 7.1. Spatial Residual Calculation ($\mathbf{R}$)
The spatial residual vector $\mathbf{R}(\mathbf{q})$ combines source terms, convective derivatives, and the divergence of diffusive fluxes:

$$\mathbf{R}(\mathbf{q}) = \mathbf{S} - \left(\mathbf{F} + \mathbf{G} + \mathbf{H}\right) - \left( \frac{\partial \mathbf{F}_D}{\partial x} + \frac{\partial \mathbf{G}_D}{\partial y} + \frac{\partial \mathbf{H}_D}{\partial z} \right)$$

where the diffusive divergence is evaluated via 2nd-order central differences:
$$\frac{\partial \mathbf{F}_D}{\partial x} = \frac{\mathbf{F}_{D, i+1} - \mathbf{F}_{D, i-1}}{2 \Delta x}, \quad \frac{\partial \mathbf{G}_D}{\partial y} = \frac{\mathbf{G}_{D, j+1} - \mathbf{G}_{D, j-1}}{2 \Delta y}, \quad \frac{\partial \mathbf{H}_D}{\partial z} = \mathcal{J}^{-1}[k] \frac{\mathbf{H}_{D, k+1} - \mathbf{H}_{D, k-1}}{2 \Delta \xi}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L870-L889)
> - **Kernel**: `rhs_and_rk2_kernel`
> - **Lines**: Lines 870–879 (`div_diff`), line 881 (`conv = F + G + H`), line 889 (`R = source - (conv + div_diff)`).

---

### 7.2. Two-Stage Predictor-Corrector Algorithm
For timestep $\Delta t = 1.0 \times 10^{-4}$, the solution is advanced in two stages:

1. **Predictor Stage (Iteration 1)**:
   $$\mathbf{q}_{\text{stage}} = \mathbf{q}^{(n)}$$
   $$\mathbf{q}^{(1)} = \mathbf{q}^{(n)} + \frac{1}{2} \Delta t \mathbf{R}\left(\mathbf{q}^{(n)}\right)$$

2. **Corrector Stage (Iteration 2)**:
   $$\mathbf{q}^{(n+1)} = \mathbf{q}_{\text{stage}} + \Delta t \mathbf{R}\left(\mathbf{q}^{(1)}\right)$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L891-L895) and [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L1174-L1200)
> - **Kernel**: `rhs_and_rk2_kernel` lines 891–895:
>   ```python
>   if iteration == 1:
>       q_stage[i, j, k, c] = q[i, j, k, c]
>       q[i, j, k, c] = q[i, j, k, c] + 0.5 * dt * R
>   else:
>       q[i, j, k, c] = q_stage[i, j, k, c] + dt * R
>   ```
> - **Host Driver Loop**: Orchestrated inside `CUDASolver.step()` at lines 1191–1197.

---

## 8. Boundary Conditions & Active Opposition Flow Control (OC)

Boundary enforcement and opposition control are isolated in a specialized 2D CUDA kernel (`opposition_and_wall_bc_kernel`) with thread blocks `tpb2 = (16, 8)` to eliminate execution divergence in interior threads.

### 8.1. Active Opposition Control (OC) Formulation
Opposition control reads the instantaneous wall-normal velocity from a detection sensor plane located at index $k_d = 33$ ($z^+ \approx 10\text{--}30$) and applies equal and opposite blowing/suction at the walls:

$$w_{\text{anti, bot}} = -\frac{(\rho w)_{i, j, k_d}}{\max(\rho_{i, j, 0}, 10^{-14})}$$

$$w_{\text{anti, top}} = -\frac{(\rho w)_{i, j, N_z - 1 - k_d}}{\max(\rho_{i, j, N_z - 1}, 10^{-14})}$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L128) and [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L930-L946)
> - **Config**: `opposition_distance: int = 33` (line 128)
> - **Kernel Lines**: Lines 930–946 in `opposition_and_wall_bc_kernel`:
>   ```python
>   src_b = max(0, kd)
>   src_t = min(nz - 1, nz - 1 - kd)
>   var[i, j, 0, 3] = -q[i, j, src_b, 3] / max(var[i, j, 0, 0], 1.0e-14)
>   var[i, j, nz - 1, 3] = -q[i, j, src_t, 3] / max(var[i, j, nz - 1, 0], 1.0e-14)
>   ```

---

### 8.2. Wall Surface Conditions ($k=0$ and $k=N_z-1$)
At each wall boundary point:
1. **No-Slip Condition**: Streamwise velocity is zero: $u_{\text{wall}} = 0$.
2. **Spanwise Actuation**: Supports optional traveling wave actuation:
   $$v_{\text{wall}}(x, t) = A_v \sin(\kappa x - \omega t)$$
   *(where $A_v = 0$ for unactuated channel flow)*.
3. **Wall-Normal Velocity**: Enforced after applying opposition flux: $w_{\text{wall}} = 0$.
4. **Isothermal Wall**: Static temperature is held constant: $T_{\text{wall}} = 1.0$.
5. **Wall Density Recovery**: From zero normal pressure gradient $\frac{\partial P}{\partial z} \approx 0$:
   $$\rho_{\text{wall}} = \frac{\gamma \text{Ma}^2 P_{\text{wall}}}{T_{\text{wall}}}$$
6. **Conserved Variable Reconstruction**:
   $$E_{\text{wall}} = T_{\text{wall}} + \frac{1}{2} \Gamma_{gM} \left(u_{\text{wall}}^2 + v_{\text{wall}}^2 + w_{\text{wall}}^2\right)$$
   $$\mathbf{q}_{\text{wall}} = [\rho_{\text{wall}}, \, \rho_{\text{wall}} u_{\text{wall}}, \, \rho_{\text{wall}} v_{\text{wall}}, \, \rho_{\text{wall}} w_{\text{wall}}, \, \rho_{\text{wall}} E_{\text{wall}}]^T$$

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L947-L975)
> - **Kernel**: `opposition_and_wall_bc_kernel`
> - **Lines**: Lines 947–975:
>   ```python
>   act = amp_v * math.sin(kappa * x1[i] - omega * time_value) if enable_oc else 0.0
>   for k in (0, nz - 1):
>       var[i, j, k, 1] = 0.0
>       var[i, j, k, 2] = act
>       var[i, j, k, 3] = 0.0
>       var[i, j, k, 5] = 1.0
>       var[i, j, k, 0] = gamma * mach * mach * var[i, j, k, 6] / var[i, j, k, 5]
>       var[i, j, k, 4] = var[i, j, k, 5] + 0.5 * ggM * (var[i,j,k,1]**2 + var[i,j,k,2]**2 + var[i,j,k,3]**2)
>   ```

---

## 9. Numerical Stability & Solution Sanity Check

To catch floating-point blowups (such as negative density or temperature caused by excessive timesteps):

$$\text{Valid Solution Criterion}: \quad \rho_{i,j,k} > 0 \quad \text{and} \quad T_{i,j,k} > 0 \quad \forall (i, j, k)$$

If violated, atomic operations flag the host to trigger a `FloatingPointError`.

> **Where it is implemented in the codebase:**
> - **File**: [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L978-L985) and [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L1165-L1173)
> - **Kernel**: `@cuda.jit def finite_check_kernel(var, flag)` (lines 978–985)
> - **Host Handler**: `CUDASolver.check_blown()` (lines 1165–1173), executed every 10 timesteps.

---

## 10. Diagnostic Turbulence Statistics & Post-Processing Mathematics

### 10.1. Planar Reynolds Decomposition
Instantaneous 3D state fields are decomposed into planar-averaged mean profiles and turbulent fluctuating fields:

$$\bar{\phi}(z) = \frac{1}{N_x N_y} \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} \phi[i, j, k]$$
$$\phi'(x, y, z, t) = \phi(x, y, z, t) - \bar{\phi}(z)$$

> **Where it is implemented in the codebase:**
> - **File**: [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py#L195-L198)
> - **Lines**: Lines 195–198 in `compute_and_save_new_results()`:
>   ```python
>   var_prime = var_h - ref_mean_bcast
>   sim_mean = np.mean(var_h, axis=(0, 1))
>   ```

---

### 10.2. Root-Mean-Square (RMS) Fluctuations, Reynolds Stresses & TKE
1. **RMS Fluctuation Profiles**:
   $$\phi_{\text{rms}}(z) = \sqrt{\frac{1}{N_x N_y} \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} (\phi'[i, j, k])^2}$$

2. **Turbulent Kinetic Energy (TKE)**:
   $$k_{\text{TKE}}(z) = \frac{1}{2} \left( u_{\text{rms}}^2(z) + v_{\text{rms}}^2(z) + w_{\text{rms}}^2(z) \right)$$

3. **Reynolds Normal & Shear Stresses**:
   $$\langle u'u' \rangle = \frac{1}{N_x N_y} \sum (u')^2, \quad \langle v'v' \rangle = \frac{1}{N_x N_y} \sum (v')^2, \quad \langle w'w' \rangle = \frac{1}{N_x N_y} \sum (w')^2$$
   $$\tau_{12}(z) = -\overline{\rho u' v'}(z)$$

> **Where it is implemented in the codebase:**
> - **File**: [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py#L200-L210)
> - **Lines**: Lines 200–210 in `compute_and_save_new_results()`:
>   ```python
>   sim_rms = np.sqrt(np.mean(var_prime ** 2, axis=(0, 1)))
>   uu = np.mean(var_prime[:, :, :, 1] ** 2, axis=(0, 1))
>   vv = np.mean(var_prime[:, :, :, 2] ** 2, axis=(0, 1))
>   ww = np.mean(var_prime[:, :, :, 3] ** 2, axis=(0, 1))
>   tke = 0.5 * (uu + vv + ww)
>   ```
> - **Output Checkpoints**: Saved to [`output_cuda/rms.dat`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/output_cuda) and [`output_cuda/turb_KE.dat`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/output_cuda).

---

### 10.3. Vortex Core Identification: 3D $Q$-Criterion
Vortical structures are identified using the second invariant $Q$ of the velocity gradient tensor $\nabla \mathbf{u} = \mathbf{S} + \mathbf{\Omega}$:

$$Q = \frac{1}{2} \left( ||\mathbf{\Omega}||_F^2 - ||\mathbf{S}||_F^2 \right) = \frac{1}{2} \left( \mathcal{R}^2 - \mathcal{S}^2 \right)$$

where:
$$\mathcal{R}^2 = 2 \left[ \left(\frac{1}{2}\left(\frac{\partial u}{\partial y} - \frac{\partial v}{\partial x}\right)\right)^2 + \left(\frac{1}{2}\left(\frac{\partial u}{\partial z} - \frac{\partial w}{\partial x}\right)\right)^2 + \left(\frac{1}{2}\left(\frac{\partial v}{\partial z} - \frac{\partial w}{\partial y}\right)\right)^2 \right]$$

$$\mathcal{S}^2 = \left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial v}{\partial y}\right)^2 + \left(\frac{\partial w}{\partial z}\right)^2 + 2\left(\frac{1}{2}\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right)\right)^2 + 2\left(\frac{1}{2}\left(\frac{\partial u}{\partial z} + \frac{\partial w}{\partial x}\right)\right)^2 + 2\left(\frac{1}{2}\left(\frac{\partial v}{\partial z} + \frac{\partial w}{\partial y}\right)\right)^2$$

> **Where it is implemented in the codebase:**
> - **File**: [`python/compare_processing_times.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compare_processing_times.py#L135-L170)
> - **Function**: `run_python_cuda_qinvariant()`
> - **Lines**: Lines 160–167:
>   ```python
>   dudx, dudy, dudz = np.gradient(u, x, y, z, axis=(2, 1, 0))
>   ...
>   S2 = dudx**2 + dvdy**2 + dwdz**2 + 2.0*(0.5*(dudy + dvdx))**2 + ...
>   R2 = 2.0*(0.5*(dudy - dvdx))**2 + ...
>   Q = 0.5 * (R2 - S2)
>   ```

---

### 10.4. Field Verification Error Metrics (Fortran vs. CUDA)
To rigorously establish precision between the Fortran MPI and Python CUDA solvers across all $M = N_x \times N_y \times N_z$ mesh cells:

1. **Mean Absolute Error ($L_1$)**:
   $$L_1 = \frac{1}{M} \sum_{m=1}^M |\phi_{\text{Fortran}}[m] - \phi_{\text{CUDA}}[m]|$$

2. **Root-Mean-Square Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{M} \sum_{m=1}^M (\phi_{\text{Fortran}}[m] - \phi_{\text{CUDA}}[m])^2}$$

3. **Relative $L_2$ Error Norm**:
   $$\text{Rel } L_2 = \frac{||\phi_{\text{Fortran}} - \phi_{\text{CUDA}}||_2}{||\phi_{\text{Fortran}}||_2} = \frac{\sqrt{\sum (\phi_{\text{F}} - \phi_{\text{C}})^2}}{\sqrt{\sum \phi_{\text{F}}^2}}$$

4. **Maximum Error ($L_\infty$)**:
   $$L_\infty = \max_{1 \le m \le M} |\phi_{\text{Fortran}}[m] - \phi_{\text{CUDA}}[m]|$$

5. **Pearson Correlation Coefficient ($r$)**:
   $$r = \frac{\sum (\phi_{\text{F}} - \bar{\phi}_{\text{F}})(\phi_{\text{C}} - \bar{\phi}_{\text{C}})}{\sqrt{\sum (\phi_{\text{F}} - \bar{\phi}_{\text{F}})^2 \sum (\phi_{\text{C}} - \bar{\phi}_{\text{C}})^2}}$$

> **Where it is implemented in the codebase:**
> - **File**: [`python/compare_fortran_cuda.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compare_fortran_cuda.py#L122-L140)
> - **Function**: `compute_metrics(q_fort, q_cuda)` (lines 122–140)
> - **Also in**: [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py#L221-L239) inside `compute_and_save_new_results()`.

---

## 11. CUDA Architecture Execution Hierarchy & Kernel Mapping

The table below summarizes the mathematical-to-kernel mapping executed on the GPU:

| Kernel / Function Name | Mathematical Operations | Thread Block Config | Source Code Location |
| :--- | :--- | :--- | :--- |
| `make_mesh()` | Grid generation, Tanh stretching $z(\xi)$, metric Jacobian $\mathcal{J}(\xi)$, 3rd-order one-sided Lagrange weights | CPU (Pre-run) | [`cuda code.py` L161–219](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L161-L219) |
| `primary_variables_kernel` | Primitive variable extraction, ideal gas EOS, temperature inversion, power-law viscosity $\mu(T) = T^{0.7}$ | `(8, 8, 8)` (512 threads) | [`cuda code.py` L360–398](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L360-L398) |
| `convective_kernel` | 4th-order central convective fluxes, local directional grid Peclet calculations ($\text{Pe}_x, \text{Pe}_y, \text{Pe}_z$), 3rd-order directional upwind limiter | `(8, 8, 8)` (512 threads) | [`cuda code.py` L399–658](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L399-L658) |
| `diffusive_kernel` | Velocity gradient tensor $\nabla \mathbf{u}$, volumetric dilatation $\Theta$, viscous stress tensor $\tau_{ij}$, directional dissipation rates $\Phi_x, \Phi_y, \Phi_z$, viscous fluxes $\mathbf{F}_D, \mathbf{G}_D, \mathbf{H}_D$ | `(8, 8, 8)` (512 threads) | [`cuda code.py` L660–819](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L660-L819) |
| `rhs_and_rk2_kernel` | Viscous flux divergence, driving body force source term $f_{\text{body}}$, explicit RK2 Stage 1 predictor & Stage 2 corrector | `(8, 8, 8)` (512 threads) | [`cuda code.py` L820–896](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L820-L896) |
| `opposition_and_wall_bc_kernel` | Sensor reading at $k_d = 33$, dynamic wall blowing/suction opposition $w_{\text{anti}}$, isothermal wall $T_{\text{wall}} = 1.0$, wall density recovery from zero pressure gradient Neumann condition | `(16, 8)` (128 threads) | [`cuda code.py` L898–976](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L898-L976) |
| `finite_check_kernel` | Solution positivity verification ($\rho > 0, T > 0$), NaN/Inf detection via CUDA atomic max | `(8, 8, 8)` (512 threads) | [`cuda code.py` L978–986](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L978-L986) |
| `save_solution()` | Solution re-ordering and ASCII Tecplot checkpoint export (`3D_cuda.dat`) | CPU (Post-run) | [`cuda code.py` L1207–1244](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L1207-L1244) |
| `run_python_cuda_qinvariant()` | 3D Second Invariant ($Q$-criterion) vortex identification using 3D finite difference gradients | Vectorized CPU/GPU | [`python/compare_processing_times.py` L135–171](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compare_processing_times.py#L135-L171) |
| `compute_and_save_new_results()` | Reynolds decomposition, RMS profiles, TKE, Reynolds stresses, and error metrics | Vectorized | [`python/benchmark_solvers.py` L148–240](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py#L148-L240) |
