# Fortran MPI vs Python CUDA Solver Output Comparison Report

- **Fortran Source Checkpoint**: `3D_fortran.dat`
- **CUDA Source Checkpoint**: `3D_cuda.dat`
- **Overall Verification Status**: `TOLERANCE_EXCEEDED`
- **Target Tolerance Threshold**: `0.001`

## Field Metric Summary Table

| Variable | Fortran Mean | CUDA Mean | Rel L2 Error | L_inf Error | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rho` | 1.054530e+00 | 1.054523e+00 | 9.210858e-03 | 1.354736e-01 | 0.996567 |
| `u` | 8.344915e-01 | 8.345463e-01 | 4.332331e-02 | 3.939462e-01 | 0.994887 |
| `v` | -1.243997e-03 | -1.241848e-03 | 6.080035e-01 | 2.078304e-01 | 0.815007 |
| `w` | -8.407741e-05 | -1.288841e-04 | 6.619621e-01 | 2.072572e-01 | 0.781301 |
| `e` | 1.847187e+00 | 1.847261e+00 | 2.832127e-02 | 5.106926e-01 | 0.992615 |
| `T` | 1.307716e+00 | 1.307886e+00 | 6.301878e-03 | 1.357351e-01 | 0.997733 |
| `p` | 4.332223e-01 | 4.332907e-01 | 8.175935e-03 | 5.937248e-02 | 0.770550 |
