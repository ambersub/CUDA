program fortran_ref_parity
  ! Minimal single-domain Fortran 90 reference implementation mirroring
  ! codes/main_v1.9_unc.f90 (cosine, 2nd-order walls) and codes1/mod_*.f90
  ! (tanh, 3rd-order walls, opposition control, wall actuation) for
  ! automated numerical parity testing of Phase 1 CUDA kernels.
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nx = 16, ny = 16, nz = 80
  integer :: bn(3) = [2, 2, 2], en(3) = [15, 15, 79]
  integer :: bs(3) = [1, 1, 1], es(3) = [16, 16, 80]

  real(dp), parameter :: gama = 1.4d0, invPr = 1.0d0 / 0.71d0
  real(dp), parameter :: one = 1.0d0, two = 2.0d0, zer = 0.0d0, haf = 0.5d0
  real(dp), parameter :: two3rd = 2.0d0 / 3.0d0, one12th = 1.0d0 / 12.0d0, Ep = 5.0d0

  integer :: mesh_mode, wall_order, oc_flag, istat, nsteps, y_d, defer_source
  real(dp) :: Re, Mac, dt, PexVal, PeyVal, PezVal, Kappa, Omega, Amp_v
  character(len=512) :: out_file

  real(dp) :: Pi, dx, dy, dXi, invdx, invdy, invdXi, invRe, ggM, invdt
  real(dp) :: x1(nx), x2(ny), Xi(nz), x3(nz), jac(nz), invJac(nz)
  real(dp) :: f1d2c0, f1d2c1, f1d2c2, b1d2c0, b1d2c1, b1d2c2
  real(dp) :: f2d2c0, f2d2c1, f2d2c2, f2d2c3, b2d2c0, b2d2c1, b2d2c2, b2d2c3
  real(dp) :: f1d3c0, f1d3c1, f1d3c2, f1d3c3, b1d3c0, b1d3c1, b1d3c2, b1d3c3
  real(dp) :: df21, df31, df32, df41, df42, df43, db12, db13, db23, db14, db24, db34

  real(dp) :: solV(0:nx+1, 0:ny+1, 0:nz+1, 5, 2)
  real(dp) :: var(nx, ny, nz, 7)
  real(dp) :: fD(nx, ny, nz, 5), gD(nx, ny, nz, 5), hD(nx, ny, nz, 5)
  real(dp) :: srcV(nx, ny, nz, 5), F(nx, ny, nz, 5, 2), G(nx, ny, nz, 5, 2), H(nx, ny, nz, 5, 2), R(nx, ny, nz, 5)
  real(dp) :: mu(nx, ny, nz), kt(nx, ny, nz), rhow(nx, ny, nz), stat1Spc1(nz)
  real(dp) :: tauxzL, tauxzU, time
  integer :: i, j, k, iter, nstep, ios
  character(len=512) :: arg

  if (command_argument_count() < 12) then
     stop "Usage: parity mesh_mode wall_order oc_flag amp_v kappa omega y_d istat nsteps defer_src re out_file"
  endif
  call get_command_argument(1, arg); read(arg, *) mesh_mode
  call get_command_argument(2, arg); read(arg, *) wall_order
  call get_command_argument(3, arg); read(arg, *) oc_flag
  call get_command_argument(4, arg); read(arg, *) Amp_v
  call get_command_argument(5, arg); read(arg, *) Kappa
  call get_command_argument(6, arg); read(arg, *) Omega
  call get_command_argument(7, arg); read(arg, *) y_d
  call get_command_argument(8, arg); read(arg, *) istat
  call get_command_argument(9, arg); read(arg, *) nsteps
  call get_command_argument(10, arg); read(arg, *) defer_source
  call get_command_argument(11, arg); read(arg, *) Re
  call get_command_argument(12, out_file)

  Mac = 1.5d0
  dt = 1.0d-4
  PexVal = 2.0d0
  PeyVal = 1.0d0
  PezVal = 0.1d0
  invdt = 1.0d0 / dt
  invRe = 1.0d0 / Re
  ggM = gama * (gama - 1.0d0) * Mac**two
  Pi = 4.0d0 * datan(1.0d0)

  x1(1) = 0.0d0; x2(1) = 0.0d0; Xi(1) = 0.0d0
  x1(nx) = 4.0d0 * Pi
  x2(ny) = 4.0d0 * Pi / 3.0d0
  dx = x1(nx) / dble(nx - 1)
  dy = x2(ny) / dble(ny - 1)
  dXi = 2.0d0 / dble(nz - 1)
  invdx = 1.0d0 / dx
  invdy = 1.0d0 / dy
  invdXi = 1.0d0 / dXi

  do i = 1, nx - 1
     x1(i + 1) = x1(i) + dx
  enddo
  do j = 1, ny - 1
     x2(j + 1) = x2(j) + dy
  enddo
  do k = 1, nz - 1
     Xi(k + 1) = Xi(k) + dXi
  enddo

  invJac = 0.0d0
  if (mesh_mode == 0) then
     do k = 1, nz
        x3(k) = 1.0d0 - dcos(0.5d0 * Xi(k) * Pi)
        jac(k) = 0.5d0 * Pi * dsin(0.5d0 * Xi(k) * Pi)
     enddo
     invJac(2:nz-1) = 1.0d0 / jac(2:nz-1)
  else
     do k = 1, nz
        x3(k) = 1.0d0 + dtanh(Ep * 0.5d0 * (Xi(k) - 1.0d0)) / dtanh(Ep * 0.5d0)
        jac(k) = 0.5d0 * Ep * (1.0d0 - (dtanh(0.5d0 * Ep * (Xi(k) - 1.0d0)))**2) / dtanh(Ep * 0.5d0)
     enddo
     x3(1) = 0.0d0
     invJac(2:nz) = 1.0d0 / jac(2:nz)
  endif

  call compute_coeffs()

  solV = 0.0d0
  var = 0.0d0
  rhow = 0.0d0
  stat1Spc1 = 1.0d0
  tauxzL = 0.0d0
  tauxzU = 0.0d0
  do k = 1, nz
     do j = 1, ny
        do i = 1, nx
           var(i,j,k,1) = 1.0d0 + 0.05d0 * dsin(0.5d0 * x1(i)) * dcos(1.5d0 * x2(j)) * dsin(Pi * x3(k) * 0.5d0)
           var(i,j,k,2) = (x3(k) * (2.0d0 - x3(k))) * (1.0d0 + 0.08d0 * dsin(x1(i)) * dcos(1.5d0 * x2(j)))
           var(i,j,k,3) = 0.04d0 * dsin(0.5d0 * Pi * x3(k)) * dsin(x1(i) + 1.5d0 * x2(j))
           var(i,j,k,4) = 0.03d0 * dsin(Pi * x3(k)) * dcos(x1(i)) * dsin(1.5d0 * x2(j))
           var(i,j,k,6) = 1.0d0 + 0.04d0 * dsin(0.5d0 * Pi * x3(k))
           var(i,j,k,5) = var(i,j,k,6) + ggM * haf * (var(i,j,k,2)**2 + var(i,j,k,3)**2 + var(i,j,k,4)**2)
           var(i,j,k,7) = var(i,j,k,1) * var(i,j,k,6) / (gama * Mac**two)
           solV(i,j,k,1,1) = var(i,j,k,1)
           solV(i,j,k,2:5,1) = var(i,j,k,1) * var(i,j,k,2:5)
           mu(i,j,k) = var(i,j,k,6)**0.7d0
           kt(i,j,k) = var(i,j,k,6)**0.7d0
        enddo
     enddo
  enddo

  call sync_all_halos()
  if (defer_source == 0) call compute_tau_and_rhomean()

  time = 0.0d0
  do nstep = 1, nsteps
     time = time + dt
     do iter = 1, 2
        call calc_conv()
        call calc_diff()
        call sync_fhd_halos()
        call calc_rhs_rk2(iter)
        call sync_solv_halos()
        call find_prim()
        call sync_var_halos()
        if (oc_flag == 1) then
           call apply_oc()
           call sync_xyline()
        else
           do j = bn(2), en(2)
              do i = bn(1), en(1)
                 var(i,j,1,4) = 0.0d0; solV(i,j,1,4,1) = 0.0d0
                 var(i,j,nz,4) = 0.0d0; solV(i,j,nz,4,1) = 0.0d0
              enddo
           enddo
        endif
        call apply_wall_bc()
        call sync_xyline()
     enddo
     if (mod(nstep, istat) == 0) call compute_tau_and_rhomean()
  enddo

  open(unit=20, file=trim(out_file), form="unformatted", access="stream", status="replace", iostat=ios)
  if (ios /= 0) stop "Failed to open output file"
  write(20) tauxzL, tauxzU, time
  do k = 1, nz
     do j = 1, ny
        do i = 1, nx
           write(20) var(i, j, k, 1:7)
        enddo
     enddo
  enddo
  close(20)

contains

  subroutine compute_coeffs()
    real(dp) :: dz1, dzN, alpha, tmpR1
    dz1 = x3(2) - x3(1)
    alpha = (x3(3) - x3(2)) / dz1
    tmpR1 = dz1 * alpha * (alpha + one)
    f1d2c0 = (one - (alpha + one)**two) / tmpR1
    f1d2c1 = ((alpha + one)**two) / tmpR1
    f1d2c2 = -one / tmpR1

    df21 = x3(2) - x3(1); df31 = x3(3) - x3(1); df32 = x3(3) - x3(2)
    df41 = x3(4) - x3(1); df42 = x3(4) - x3(2); df43 = x3(4) - x3(3)

    f1d3c0 = (df31*df21 + df31*df41 + df21*df41) / ((-df21)*(-df31)*(-df41))
    f1d3c1 = (df41*df31) / (df21*(-df32)*(-df42))
    f1d3c2 = (df41*df21) / (df31*df32*(-df43))
    f1d3c3 = (df31*df21) / (df41*df42*df43)

    f2d2c0 = (2.0d0*(-df21-df31-df41)) / ((-df21)*(-df31)*(-df41))
    f2d2c1 = (2.0d0*(-df31-df41)) / (df21*(-df32)*(-df42))
    f2d2c2 = (2.0d0*(-df21-df41)) / (df31*df32*(-df43))
    f2d2c3 = (2.0d0*(-df21-df31)) / (df41*df42*df43)

    dzN = x3(nz) - x3(nz-1)
    alpha = (x3(nz-1) - x3(nz-2)) / dzN
    tmpR1 = dzN * alpha * (alpha + one)
    b1d2c0 = ((alpha + one)**two - one) / tmpR1
    b1d2c1 = -((alpha + one)**two) / tmpR1
    b1d2c2 = one / tmpR1

    db12 = x3(nz) - x3(nz-1); db13 = x3(nz) - x3(nz-2); db23 = x3(nz-1) - x3(nz-2)
    db14 = x3(nz) - x3(nz-3); db24 = x3(nz-1) - x3(nz-3); db34 = x3(nz-2) - x3(nz-3)

    b1d3c0 = (db13*db12 + db13*db14 + db12*db14) / (db12*db13*db14)
    b1d3c1 = (db14*db13) / ((-db12)*db23*db24)
    b1d3c2 = (db14*db12) / ((-db13)*(-db23)*db34)
    b1d3c3 = (db13*db12) / ((-db14)*(-db24)*(-db34))

    b2d2c0 = (2.0d0*(db12 + db13 + db14)) / (db12*db13*db14)
    b2d2c1 = (2.0d0*(db13 + db14)) / ((-db12)*db23*db24)
    b2d2c2 = (2.0d0*(db12 + db14)) / ((-db13)*(-db23)*db34)
    b2d2c3 = (2.0d0*(db12 + db13)) / ((-db14)*(-db24)*(-db34))
  end subroutine compute_coeffs

  subroutine sync_solv_halos()
    integer :: kk
    do kk = bn(3), en(3)
       solV(0:1, bn(2):en(2), kk, :, 1) = solV(en(1)-1:en(1), bn(2):en(2), kk, :, 1)
       solV(en(1)+1:en(1)+2, bn(2):en(2), kk, :, 1) = solV(bn(1):bn(1)+1, bn(2):en(2), kk, :, 1)
       solV(0:nx+1, 0:1, kk, :, 1) = solV(0:nx+1, en(2)-1:en(2), kk, :, 1)
       solV(0:nx+1, en(2)+1:en(2)+2, kk, :, 1) = solV(0:nx+1, bn(2):bn(2)+1, kk, :, 1)
    enddo
  end subroutine sync_solv_halos

  subroutine sync_var_halos()
    integer :: kk
    do kk = bn(3), en(3)
       var(bn(1)-1, bn(2):en(2), kk, :) = var(en(1), bn(2):en(2), kk, :)
       var(en(1)+1, bn(2):en(2), kk, :) = var(bn(1), bn(2):en(2), kk, :)
       var(:, bn(2)-1, kk, :) = var(:, en(2), kk, :)
       var(:, en(2)+1, kk, :) = var(:, bn(2), kk, :)
       mu(bn(1)-1, bn(2):en(2), kk) = mu(en(1), bn(2):en(2), kk)
       mu(en(1)+1, bn(2):en(2), kk) = mu(bn(1), bn(2):en(2), kk)
       mu(:, bn(2)-1, kk) = mu(:, en(2), kk)
       mu(:, en(2)+1, kk) = mu(:, bn(2), kk)
       kt(bn(1)-1, bn(2):en(2), kk) = kt(en(1), bn(2):en(2), kk)
       kt(en(1)+1, bn(2):en(2), kk) = kt(bn(1), bn(2):en(2), kk)
       kt(:, bn(2)-1, kk) = kt(:, en(2), kk)
       kt(:, en(2)+1, kk) = kt(:, bn(2), kk)
    enddo
  end subroutine sync_var_halos

  subroutine sync_xyline()
    integer :: kw
    do kw = 1, nz, nz - 1
       var(bn(1)-1, bn(2):en(2), kw, :) = var(en(1), bn(2):en(2), kw, :)
       var(en(1)+1, bn(2):en(2), kw, :) = var(bn(1), bn(2):en(2), kw, :)
       var(:, bn(2)-1, kw, :) = var(:, en(2), kw, :)
       var(:, en(2)+1, kw, :) = var(:, bn(2), kw, :)

       solV(bn(1)-1, bn(2):en(2), kw, :, 1) = solV(en(1), bn(2):en(2), kw, :, 1)
       solV(en(1)+1, bn(2):en(2), kw, :, 1) = solV(bn(1), bn(2):en(2), kw, :, 1)
       solV(1:nx, bn(2)-1, kw, :, 1) = solV(1:nx, en(2), kw, :, 1)
       solV(1:nx, en(2)+1, kw, :, 1) = solV(1:nx, bn(2), kw, :, 1)

       mu(bn(1)-1, bn(2):en(2), kw) = mu(en(1), bn(2):en(2), kw)
       mu(en(1)+1, bn(2):en(2), kw) = mu(bn(1), bn(2):en(2), kw)
       mu(:, bn(2)-1, kw) = mu(:, en(2), kw)
       mu(:, en(2)+1, kw) = mu(:, bn(2), kw)
       kt(bn(1)-1, bn(2):en(2), kw) = kt(en(1), bn(2):en(2), kw)
       kt(en(1)+1, bn(2):en(2), kw) = kt(bn(1), bn(2):en(2), kw)
       kt(:, bn(2)-1, kw) = kt(:, en(2), kw)
       kt(:, en(2)+1, kw) = kt(:, bn(2), kw)
    enddo
  end subroutine sync_xyline

  subroutine sync_fhd_halos()
    integer :: kk
    do kk = bn(3), en(3)
       fD(bn(1)-1, bn(2):en(2), kk, :) = fD(en(1), bn(2):en(2), kk, :)
       fD(en(1)+1, bn(2):en(2), kk, :) = fD(bn(1), bn(2):en(2), kk, :)
       gD(bn(1):en(1), bn(2)-1, kk, :) = gD(bn(1):en(1), en(2), kk, :)
       gD(bn(1):en(1), en(2)+1, kk, :) = gD(bn(1):en(1), bn(2), kk, :)
    enddo
  end subroutine sync_fhd_halos

  subroutine sync_all_halos()
    call sync_solv_halos()
    call sync_var_halos()
    call sync_xyline()
  end subroutine sync_all_halos

  subroutine calc_conv()
    real(dp) :: Pe, tmpR1, tmpR2
    do k = bn(3), en(3)
       do j = bn(2), en(2)
          do i = bn(1), en(1)
             Pe = Re * abs(solV(i,j,k,2,1)) * dx / mu(i,j,k)
             if ((Pe > PexVal) .or. (i == bn(1)) .or. (i == en(1))) then
                tmpR1 = (var(i,j,k,2) + var(i+1,j,k,2)) * haf
                tmpR2 = (var(i,j,k,2) + var(i-1,j,k,2)) * haf
                if (tmpR1 >= 0.0d0 .and. tmpR2 >= 0.0d0) then
                   F(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i+1,j,k,1:5,1)-solV(i-1,j,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i-1,j,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i-2,j,k,1:5,1))) * invdx / 8.0d0
                elseif (tmpR1 < 0.0d0 .and. tmpR2 < 0.0d0) then
                   F(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i+1,j,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i+2,j,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i-1,j,k,1:5,1)-solV(i+1,j,k,1:5,1))) * invdx / 8.0d0
                elseif (tmpR1 < 0.0d0 .and. tmpR2 >= 0.0d0) then
                   F(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i+1,j,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i+2,j,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i-1,j,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i-2,j,k,1:5,1))) * invdx / 8.0d0
                else
                   F(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i+1,j,k,1:5,1)-solV(i-1,j,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i-1,j,k,1:5,1)-solV(i+1,j,k,1:5,1))) * invdx / 8.0d0
                endif
             else
                F(i,j,k,1:5,1) = (-var(i+2,j,k,2)*solV(i+2,j,k,1:5,1) + 8.0d0*var(i+1,j,k,2)*solV(i+1,j,k,1:5,1) - &
                                  8.0d0*var(i-1,j,k,2)*solV(i-1,j,k,1:5,1) + var(i-2,j,k,2)*solV(i-2,j,k,1:5,1)) * invdx * one12th
             endif

             Pe = Re * abs(solV(i,j,k,3,1)) * dy / mu(i,j,k)
             if ((Pe > PeyVal) .or. (j == bn(2)) .or. (j == en(2))) then
                tmpR1 = (var(i,j,k,3) + var(i,j+1,k,3)) * haf
                tmpR2 = (var(i,j,k,3) + var(i,j-1,k,3)) * haf
                if (tmpR1 >= 0.0d0 .and. tmpR2 >= 0.0d0) then
                   G(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j+1,k,1:5,1)-solV(i,j-1,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j-1,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j-2,k,1:5,1))) * invdy / 8.0d0
                elseif (tmpR1 < 0.0d0 .and. tmpR2 < 0.0d0) then
                   G(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j+1,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j+2,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j-1,k,1:5,1)-solV(i,j+1,k,1:5,1))) * invdy / 8.0d0
                elseif (tmpR1 < 0.0d0 .and. tmpR2 >= 0.0d0) then
                   G(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j+1,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j+2,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j-1,k,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j-2,k,1:5,1))) * invdy / 8.0d0
                else
                   G(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j+1,k,1:5,1)-solV(i,j-1,k,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j-1,k,1:5,1)-solV(i,j+1,k,1:5,1))) * invdy / 8.0d0
                endif
             else
                G(i,j,k,1:5,1) = (-var(i,j+2,k,3)*solV(i,j+2,k,1:5,1) + 8.0d0*var(i,j+1,k,3)*solV(i,j+1,k,1:5,1) - &
                                  8.0d0*var(i,j-1,k,3)*solV(i,j-1,k,1:5,1) + var(i,j-2,k,3)*solV(i,j-2,k,1:5,1)) * invdy * one12th
             endif

             Pe = Re * abs(solV(i,j,k,4,1)) * dXi * jac(k) / mu(i,j,k)
             if (k < 3 .or. k > nz - 2) then
                H(i,j,k,1:5,1) = (var(i,j,k+1,4)*solV(i,j,k+1,1:5,1) - var(i,j,k-1,4)*solV(i,j,k-1,1:5,1)) * haf * invdXi * invJac(k)
             elseif (Pe > PezVal) then
                tmpR1 = (var(i,j,k,4) + var(i,j,k+1,4)) * haf
                tmpR2 = (var(i,j,k,4) + var(i,j,k-1,4)) * haf
                if (tmpR1 >= 0.0d0 .and. tmpR2 >= 0.0d0) then
                   H(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j,k+1,1:5,1)-solV(i,j,k-1,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k-1,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j,k-2,1:5,1))) * invdXi * invJac(k) / 8.0d0
                elseif (tmpR1 < 0.0d0 .and. tmpR2 < 0.0d0) then
                   H(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k+1,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j,k+2,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j,k-1,1:5,1)-solV(i,j,k+1,1:5,1))) * invdXi * invJac(k) / 8.0d0
                elseif (tmpR1 < 0.0d0 .and. tmpR2 >= 0.0d0) then
                   H(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k+1,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j,k+2,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k-1,1:5,1)+3.0d0*solV(i,j,k,1:5,1)-solV(i,j,k-2,1:5,1))) * invdXi * invJac(k) / 8.0d0
                else
                   H(i,j,k,1:5,1) = (tmpR1*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j,k+1,1:5,1)-solV(i,j,k-1,1:5,1)) - &
                                     tmpR2*(6.0d0*solV(i,j,k,1:5,1)+3.0d0*solV(i,j,k-1,1:5,1)-solV(i,j,k+1,1:5,1))) * invdXi * invJac(k) / 8.0d0
                endif
             else
                H(i,j,k,1:5,1) = (-var(i,j,k+2,4)*solV(i,j,k+2,1:5,1) + 8.0d0*var(i,j,k+1,4)*solV(i,j,k+1,1:5,1) - &
                                  8.0d0*var(i,j,k-1,4)*solV(i,j,k-1,1:5,1) + var(i,j,k-2,4)*solV(i,j,k-2,1:5,1)) * invdXi * invJac(k) * one12th
             endif
          enddo
       enddo
    enddo
  end subroutine calc_conv

  subroutine calc_diff()
    real(dp) :: Ut, Vt, Wt, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, dTdx, dTdy, dTdz
    real(dp) :: divg, muT, ktT, PressT, dissF, dissG, dissH
    do k = bs(3), es(3)
       do j = bn(2), en(2)
          do i = bn(1), en(1)
             Ut = var(i,j,k,2); Vt = var(i,j,k,3); Wt = var(i,j,k,4)
             dudx = (var(i+1,j,k,2) - var(i-1,j,k,2)) * haf * invdx
             dudy = (var(i,j+1,k,2) - var(i,j-1,k,2)) * haf * invdy
             dvdx = (var(i+1,j,k,3) - var(i-1,j,k,3)) * haf * invdx
             dvdy = (var(i,j+1,k,3) - var(i,j-1,k,3)) * haf * invdy
             dwdx = (var(i+1,j,k,4) - var(i-1,j,k,4)) * haf * invdx
             dwdy = (var(i,j+1,k,4) - var(i,j-1,k,4)) * haf * invdy
             dTdx = (var(i+1,j,k,6) - var(i-1,j,k,6)) * haf * invdx
             dTdy = (var(i,j+1,k,6) - var(i,j-1,k,6)) * haf * invdy
             if (k == 1) then
                if (wall_order == 2) then
                   dudz = var(i,j,1,2)*f1d2c0 + var(i,j,2,2)*f1d2c1 + var(i,j,3,2)*f1d2c2
                   dvdz = var(i,j,1,3)*f1d2c0 + var(i,j,2,3)*f1d2c1 + var(i,j,3,3)*f1d2c2
                   dwdz = var(i,j,1,4)*f1d2c0 + var(i,j,2,4)*f1d2c1 + var(i,j,3,4)*f1d2c2
                   dTdz = var(i,j,1,6)*f1d2c0 + var(i,j,2,6)*f1d2c1 + var(i,j,3,6)*f1d2c2
                else
                   dudz = var(i,j,1,2)*f1d3c0 + var(i,j,2,2)*f1d3c1 + var(i,j,3,2)*f1d3c2 + var(i,j,4,2)*f1d3c3
                   dvdz = var(i,j,1,3)*f1d3c0 + var(i,j,2,3)*f1d3c1 + var(i,j,3,3)*f1d3c2 + var(i,j,4,3)*f1d3c3
                   dwdz = var(i,j,1,4)*f1d3c0 + var(i,j,2,4)*f1d3c1 + var(i,j,3,4)*f1d3c2 + var(i,j,4,4)*f1d3c3
                   dTdz = var(i,j,1,6)*f1d3c0 + var(i,j,2,6)*f1d3c1 + var(i,j,3,6)*f1d3c2 + var(i,j,4,6)*f1d3c3
                endif
             elseif (k == nz) then
                if (wall_order == 2) then
                   dudz = var(i,j,nz,2)*b1d2c0 + var(i,j,nz-1,2)*b1d2c1 + var(i,j,nz-2,2)*b1d2c2
                   dvdz = var(i,j,nz,3)*b1d2c0 + var(i,j,nz-1,3)*b1d2c1 + var(i,j,nz-2,3)*b1d2c2
                   dwdz = var(i,j,nz,4)*b1d2c0 + var(i,j,nz-1,4)*b1d2c1 + var(i,j,nz-2,4)*b1d2c2
                   dTdz = var(i,j,nz,6)*b1d2c0 + var(i,j,nz-1,6)*b1d2c1 + var(i,j,nz-2,6)*b1d2c2
                else
                   dudz = var(i,j,nz,2)*b1d3c0 + var(i,j,nz-1,2)*b1d3c1 + var(i,j,nz-2,2)*b1d3c2 + var(i,j,nz-3,2)*b1d3c3
                   dvdz = var(i,j,nz,3)*b1d3c0 + var(i,j,nz-1,3)*b1d3c1 + var(i,j,nz-2,3)*b1d3c2 + var(i,j,nz-3,3)*b1d3c3
                   dwdz = var(i,j,nz,4)*b1d3c0 + var(i,j,nz-1,4)*b1d3c1 + var(i,j,nz-2,4)*b1d3c2 + var(i,j,nz-3,4)*b1d3c3
                   dTdz = var(i,j,nz,6)*b1d3c0 + var(i,j,nz-1,6)*b1d3c1 + var(i,j,nz-2,6)*b1d3c2 + var(i,j,nz-3,6)*b1d3c3
                endif
             else
                dudz = (var(i,j,k+1,2) - var(i,j,k-1,2)) * haf * invdXi * invJac(k)
                dvdz = (var(i,j,k+1,3) - var(i,j,k-1,3)) * haf * invdXi * invJac(k)
                dwdz = (var(i,j,k+1,4) - var(i,j,k-1,4)) * haf * invdXi * invJac(k)
                dTdz = (var(i,j,k+1,6) - var(i,j,k-1,6)) * haf * invdXi * invJac(k)
             endif
             divg = dudx + dvdy + dwdz
             muT = mu(i,j,k); ktT = kt(i,j,k); PressT = var(i,j,k,7)
             dissF = ggM*invRe*(two3rd*muT*Ut*divg - two*muT*Ut*dudx - muT*Vt*(dvdx+dudy) - muT*Wt*(dwdx+dudz))
             dissG = ggM*invRe*(-muT*Ut*(dvdx+dudy) + two3rd*muT*Vt*divg - two*muT*Vt*dvdy - muT*Wt*(dwdy+dvdz))
             dissH = ggM*invRe*(-muT*Ut*(dwdx+dudz) - muT*Vt*(dwdy+dvdz) + two3rd*muT*Wt*divg - two*muT*Wt*dwdz)
             fD(i,j,k,1) = zer
             fD(i,j,k,2) = PressT + two3rd*muT*invRe*divg - two*muT*invRe*dudx
             fD(i,j,k,3) = -muT*invRe*(dvdx + dudy)
             fD(i,j,k,4) = -muT*invRe*(dwdx + dudz)
             fD(i,j,k,5) = -gama*ktT*invRe*invPr*dTdx + ggM*Ut*PressT + dissF
             gD(i,j,k,1) = zer
             gD(i,j,k,2) = -muT*invRe*(dvdx + dudy)
             gD(i,j,k,3) = PressT + two3rd*muT*invRe*divg - two*muT*invRe*dvdy
             gD(i,j,k,4) = -muT*invRe*(dwdy + dvdz)
             gD(i,j,k,5) = -gama*ktT*invRe*invPr*dTdy + ggM*Vt*PressT + dissG
             hD(i,j,k,1) = zer
             hD(i,j,k,2) = -muT*invRe*(dwdx + dudz)
             hD(i,j,k,3) = -muT*invRe*(dwdy + dvdz)
             hD(i,j,k,4) = PressT + two3rd*muT*invRe*divg - two*muT*invRe*dwdz
             hD(i,j,k,5) = -gama*ktT*invRe*invPr*dTdz + ggM*Wt*PressT + dissH
          enddo
       enddo
    enddo
  end subroutine calc_diff

  subroutine calc_rhs_rk2(iter_idx)
    integer, intent(in) :: iter_idx
    do k = bn(3), en(3)
       do j = bn(2), en(2)
          do i = bn(1), en(1)
             F(i,j,k,:,2) = (fD(i+1,j,k,:) - fD(i-1,j,k,:)) * haf * invdx
             G(i,j,k,:,2) = (gD(i,j+1,k,:) - gD(i,j-1,k,:)) * haf * invdy
             H(i,j,k,:,2) = (hD(i,j,k+1,:) - hD(i,j,k-1,:)) * haf * invdXi * invJac(k)
          enddo
       enddo
    enddo
    do k = bn(3), en(3)
       do j = bn(2), en(2)
          do i = bn(1), en(1)
             srcV(i,j,k,1) = zer
             srcV(i,j,k,2) = haf * (tauxzL - tauxzU) * stat1Spc1(k)
             srcV(i,j,k,3:4) = zer
             srcV(i,j,k,5) = ggM * var(i,j,k,2) * srcV(i,j,k,2)
             R(i,j,k,:) = srcV(i,j,k,:) - (F(i,j,k,:,1) + F(i,j,k,:,2) + G(i,j,k,:,1) + G(i,j,k,:,2) + H(i,j,k,:,1) + H(i,j,k,:,2))
             if (iter_idx == 1) then
                solV(i,j,k,:,2) = solV(i,j,k,:,1)
                solV(i,j,k,:,1) = solV(i,j,k,:,1) + haf * dt * R(i,j,k,:)
             else
                solV(i,j,k,:,1) = solV(i,j,k,:,2) + dt * R(i,j,k,:)
             endif
          enddo
       enddo
    enddo
  end subroutine calc_rhs_rk2

  subroutine find_prim()
    do k = bn(3), en(3)
       do j = bn(2), en(2)
          do i = bn(1), en(1)
             var(i,j,k,1) = solV(i,j,k,1,1)
             var(i,j,k,2:5) = solV(i,j,k,2:5,1) / solV(i,j,k,1,1)
             var(i,j,k,6) = var(i,j,k,5) - ggM * haf * sum(var(i,j,k,2:4)**two)
             var(i,j,k,7) = var(i,j,k,1) * var(i,j,k,6) / (gama * Mac**two)
             mu(i,j,k) = var(i,j,k,6)**0.7d0
             kt(i,j,k) = mu(i,j,k)
          enddo
       enddo
    enddo
  end subroutine find_prim

  subroutine apply_oc()
    do j = bn(2), en(2)
       do i = bn(1), en(1)
          var(i,j,1,4) = -(solV(i,j,y_d,4,1)) / var(i,j,1,1)
          solV(i,j,1,4,1) = -(solV(i,j,y_d,4,1))
          var(i,j,nz,4) = -(solV(i,j,nz-y_d+1,4,1)) / var(i,j,nz,1)
          solV(i,j,nz,4,1) = -(solV(i,j,nz-y_d+1,4,1))
       enddo
    enddo
  end subroutine apply_oc

  subroutine apply_wall_bc()
    real(dp) :: dudxdz, dudzdx, drhouwdx, d2wdx2, dvdydz, dvdzdy, drhovwdy, d2wdy2
    real(dp) :: mud2wdz2, dwdzdmudz, drhow2dz, drhowdt, lhs
    mu(:,:,1) = var(:,:,1,6)**0.7d0
    mu(:,:,nz) = var(:,:,nz,6)**0.7d0
    do j = bn(2), en(2)
       do i = bn(1), en(1)
          if (wall_order == 2) then
             dudxdz = (mu(i+1,j,1)*(f1d2c0*var(i+1,j,1,2)+f1d2c1*var(i+1,j,2,2)+f1d2c2*var(i+1,j,3,2)) - &
                       mu(i-1,j,1)*(f1d2c0*var(i-1,j,1,2)+f1d2c1*var(i-1,j,2,2)+f1d2c2*var(i-1,j,3,2)))*haf*invdx
             dudzdx = ((f1d2c0*mu(i,j,1)*var(i+1,j,1,2)+f1d2c1*mu(i,j,2)*var(i+1,j,2,2)+f1d2c2*mu(i,j,3)*var(i+1,j,3,2)) - &
                       (f1d2c0*mu(i,j,1)*var(i-1,j,1,2)+f1d2c1*mu(i,j,2)*var(i-1,j,2,2)+f1d2c2*mu(i,j,3)*var(i-1,j,3,2)))*haf*invdx
             dvdydz = (mu(i,j+1,1)*(f1d2c0*var(i,j+1,1,3)+f1d2c1*var(i,j+1,2,3)+f1d2c2*var(i,j+1,3,3)) - &
                       mu(i,j-1,1)*(f1d2c0*var(i,j-1,1,3)+f1d2c1*var(i,j-1,2,3)+f1d2c2*var(i,j-1,3,3)))*haf*invdy
             dvdzdy = ((f1d2c0*mu(i,j,1)*var(i,j+1,1,3)+f1d2c1*mu(i,j,2)*var(i,j+1,2,3)+f1d2c2*mu(i,j,3)*var(i,j+1,3,3)) - &
                       (f1d2c0*mu(i,j,1)*var(i,j-1,1,3)+f1d2c1*mu(i,j,2)*var(i,j-1,2,3)+f1d2c2*mu(i,j,3)*var(i,j-1,3,3)))*haf*invdy
             mud2wdz2 = mu(i,j,1)*(f2d2c0*var(i,j,1,4)+f2d2c1*var(i,j,2,4)+f2d2c2*var(i,j,3,4)+f2d2c3*var(i,j,4,4))
             dwdzdmudz = (f1d2c0*var(i,j,1,4)+f1d2c1*var(i,j,2,4)+f1d2c2*var(i,j,3,4)) * (f1d2c0*mu(i,j,1)+f1d2c1*mu(i,j,2)+f1d2c2*mu(i,j,3))
             drhow2dz = f1d2c0*var(i,j,1,1)*var(i,j,1,4)**two + f1d2c1*var(i,j,2,1)*var(i,j,2,4)**two + f1d2c2*var(i,j,3,1)*var(i,j,3,4)**two
             drhouwdx = (var(i+1,j,1,1)*var(i+1,j,1,2)*var(i+1,j,1,4) - var(i-1,j,1,1)*var(i-1,j,1,2)*var(i-1,j,1,4))*haf*invdx
             d2wdx2 = mu(i,j,1)*(var(i+1,j,1,4)+var(i-1,j,1,4)-2*var(i,j,1,4))*invdx*invdx
             drhovwdy = (var(i,j+1,1,1)*var(i,j+1,1,3)*var(i,j+1,1,4) - var(i,j-1,1,1)*var(i,j-1,1,3)*var(i,j-1,1,4))*haf*invdy
             d2wdy2 = mu(i,j,1)*(var(i,j+1,1,4)+var(i,j-1,1,4)-2*var(i,j,1,4))*invdy*invdy
             drhowdt = ((var(i,j,1,1)*var(i,j,1,4)) - rhow(i,j,1))*invdt
             lhs = invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz))) + drhow2dz + drhouwdx + drhovwdy + drhowdt
             var(i,j,1,7) = -(lhs + f1d2c1*var(i,j,2,7) + f1d2c2*var(i,j,3,7)) / f1d2c0

             dudxdz = (mu(i+1,j,nz)*(b1d2c0*var(i+1,j,nz,2)+b1d2c1*var(i+1,j,nz-1,2)+b1d2c2*var(i+1,j,nz-2,2)) - &
                       mu(i-1,j,nz)*(b1d2c0*var(i-1,j,nz,2)+b1d2c1*var(i-1,j,nz-1,2)+b1d2c2*var(i-1,j,nz-2,2)))*haf*invdx
             dudzdx = ((b1d2c0*mu(i,j,nz)*var(i+1,j,nz,2)+b1d2c1*mu(i,j,nz-1)*var(i+1,j,nz-1,2)+b1d2c2*mu(i,j,nz-2)*var(i+1,j,nz-2,2)) - &
                       (b1d2c0*mu(i,j,nz)*var(i-1,j,nz,2)+b1d2c1*mu(i,j,nz-1)*var(i-1,j,nz-1,2)+b1d2c2*mu(i,j,nz-2)*var(i-1,j,nz-2,2)))*haf*invdx
             dvdydz = (mu(i,j+1,nz)*(b1d2c0*var(i,j+1,nz,3)+b1d2c1*var(i,j+1,nz-1,3)+b1d2c2*var(i,j+1,nz-2,3)) - &
                       mu(i,j-1,nz)*(b1d2c0*var(i,j-1,nz,3)+b1d2c1*var(i,j-1,nz-1,3)+b1d2c2*var(i,j-1,nz-2,3)))*haf*invdy
             dvdzdy = ((b1d2c0*mu(i,j,nz)*var(i,j+1,nz,3)+b1d2c1*mu(i,j,nz-1)*var(i,j+1,nz-1,3)+b1d2c2*mu(i,j,nz-2)*var(i,j+1,nz-2,3)) - &
                       (b1d2c0*mu(i,j,nz)*var(i,j-1,nz,3)+b1d2c1*mu(i,j,nz-1)*var(i,j-1,nz-1,3)+b1d2c2*mu(i,j,nz-2)*var(i,j-1,nz-2,3)))*haf*invdy
             mud2wdz2 = mu(i,j,nz)*(b2d2c0*var(i,j,nz,4)+b2d2c1*var(i,j,nz-1,4)+b2d2c2*var(i,j,nz-2,4)+b2d2c3*var(i,j,nz-3,4))
             dwdzdmudz = (b1d2c0*var(i,j,nz,4)+b1d2c1*var(i,j,nz-1,4)+b1d2c2*var(i,j,nz-2,4)) * (b1d2c0*mu(i,j,nz)+b1d2c1*mu(i,j,nz-1)+b1d2c2*mu(i,j,nz-2))
             drhow2dz = b1d2c0*var(i,j,nz,1)*var(i,j,nz,4)**two + b1d2c1*var(i,j,nz-1,1)*var(i,j,nz-1,4)**two + b1d2c2*var(i,j,nz-2,1)*var(i,j,nz-2,4)**two
             drhouwdx = (var(i+1,j,nz,1)*var(i+1,j,nz,2)*var(i+1,j,nz,4) - var(i-1,j,nz,1)*var(i-1,j,nz,2)*var(i-1,j,nz,4))*haf*invdx
             d2wdx2 = mu(i,j,nz)*(var(i+1,j,nz,4)+var(i-1,j,nz,4)-2*var(i,j,nz,4))*invdx*invdx
             drhovwdy = (var(i,j+1,nz,1)*var(i,j+1,nz,3)*var(i,j+1,nz,4) - var(i,j-1,nz,1)*var(i,j-1,nz,3)*var(i,j-1,nz,4))*haf*invdy
             d2wdy2 = mu(i,j,nz)*(var(i,j+1,nz,4)+var(i,j-1,nz,4)-2*var(i,j,nz,4))*invdy*invdy
             drhowdt = ((var(i,j,nz,1)*var(i,j,nz,4)) - rhow(i,j,nz))*invdt
             lhs = invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz))) + drhow2dz + drhouwdx + drhovwdy + drhowdt
             var(i,j,nz,7) = -(lhs + b1d2c1*var(i,j,nz-1,7) + b1d2c2*var(i,j,nz-2,7)) / b1d2c0
          else
             dudxdz = (mu(i+1,j,1)*(f1d3c0*var(i+1,j,1,2)+f1d3c1*var(i+1,j,2,2)+f1d3c2*var(i+1,j,3,2)+f1d3c3*var(i+1,j,4,2)) - &
                       mu(i-1,j,1)*(f1d3c0*var(i-1,j,1,2)+f1d3c1*var(i-1,j,2,2)+f1d3c2*var(i-1,j,3,2)+f1d3c3*var(i-1,j,4,2)))*haf*invdx
             dudzdx = ((f1d3c0*mu(i,j,1)*var(i+1,j,1,2)+f1d3c1*mu(i,j,2)*var(i+1,j,2,2)+f1d3c2*mu(i,j,3)*var(i+1,j,3,2)+f1d3c3*mu(i,j,4)*var(i+1,j,4,2)) - &
                       (f1d3c0*mu(i,j,1)*var(i-1,j,1,2)+f1d3c1*mu(i,j,2)*var(i-1,j,2,2)+f1d3c2*mu(i,j,3)*var(i-1,j,3,2)+f1d3c3*mu(i,j,4)*var(i-1,j,4,2)))*haf*invdx
             drhouwdx = (var(i+1,j,1,1)*var(i+1,j,1,2)*var(i+1,j,1,4) - var(i-1,j,1,1)*var(i-1,j,1,2)*var(i-1,j,1,4))*haf*invdx
             d2wdx2 = mu(i,j,1)*(var(i+1,j,1,4)+var(i-1,j,1,4)-2*var(i,j,1,4))*invdx*invdx
             dvdydz = (mu(i,j+1,1)*(f1d3c0*var(i,j+1,1,3)+f1d3c1*var(i,j+1,2,3)+f1d3c2*var(i,j+1,3,3)+f1d3c3*var(i,j+1,4,3)) - &
                       mu(i,j-1,1)*(f1d3c0*var(i,j-1,1,3)+f1d3c1*var(i,j-1,2,3)+f1d3c2*var(i,j-1,3,3)+f1d3c3*var(i,j-1,4,3)))*haf*invdy
             dvdzdy = ((f1d3c0*mu(i,j,1)*var(i,j+1,1,3)+f1d3c1*mu(i,j,2)*var(i,j+1,2,3)+f1d3c2*mu(i,j,3)*var(i,j+1,3,3)+f1d3c3*mu(i,j,4)*var(i,j+1,4,3)) - &
                       (f1d3c0*mu(i,j,1)*var(i,j-1,1,3)+f1d3c1*mu(i,j,2)*var(i,j-1,2,3)+f1d3c2*mu(i,j,3)*var(i,j-1,3,3)+f1d3c3*mu(i,j,4)*var(i,j-1,4,3)))*haf*invdy
             drhovwdy = (var(i,j+1,1,1)*var(i,j+1,1,3)*var(i,j+1,1,4) - var(i,j-1,1,1)*var(i,j-1,1,3)*var(i,j-1,1,4))*haf*invdy
             d2wdy2 = mu(i,j,1)*(var(i,j+1,1,4)+var(i,j-1,1,4)-2*var(i,j,1,4))*invdy*invdy
             mud2wdz2 = (mu(i,j,2)*(var(i,j,3,4)-var(i,j,2,4))/df32 - mu(i,j,1)*(var(i,j,2,4)-var(i,j,1,4))/df21)/df21
             dwdzdmudz = 0.0d0
             drhow2dz = f1d3c0*var(i,j,1,1)*var(i,j,1,4)**two + f1d3c1*var(i,j,2,1)*var(i,j,2,4)**two + f1d3c2*var(i,j,3,1)*var(i,j,3,4)**two + f1d3c3*var(i,j,4,1)*var(i,j,4,4)**two
             drhowdt = ((var(i,j,1,1)*var(i,j,1,4)) - rhow(i,j,1))*invdt
             lhs = invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz))) + drhow2dz + drhouwdx + drhovwdy + drhowdt
             var(i,j,1,7) = -(lhs + f1d3c1*var(i,j,2,7) + f1d3c2*var(i,j,3,7) + f1d3c3*var(i,j,4,7)) / f1d3c0

             dudxdz = (mu(i+1,j,nz)*(b1d3c0*var(i+1,j,nz,2)+b1d3c1*var(i+1,j,nz-1,2)+b1d3c2*var(i+1,j,nz-2,2)+b1d3c3*var(i+1,j,nz-3,2)) - &
                       mu(i-1,j,nz)*(b1d3c0*var(i-1,j,nz,2)+b1d3c1*var(i-1,j,nz-1,2)+b1d3c2*var(i-1,j,nz-2,2)+b1d3c3*var(i-1,j,nz-3,2)))*haf*invdx
             dudzdx = ((b1d3c0*mu(i,j,nz)*var(i+1,j,nz,2)+b1d3c1*mu(i,j,nz-1)*var(i+1,j,nz-1,2)+b1d3c2*mu(i,j,nz-2)*var(i+1,j,nz-2,2)+b1d3c3*mu(i,j,nz-3)*var(i+1,j,nz-3,2)) - &
                       (b1d3c0*mu(i,j,nz)*var(i-1,j,nz,2)+b1d3c1*mu(i,j,nz-1)*var(i-1,j,nz-1,2)+b1d3c2*mu(i,j,nz-2)*var(i-1,j,nz-2,2)+b1d3c3*mu(i,j,nz-3)*var(i-1,j,nz-3,2)))*haf*invdx
             drhouwdx = (var(i+1,j,nz,1)*var(i+1,j,nz,2)*var(i+1,j,nz,4) - var(i-1,j,nz,1)*var(i-1,j,nz,2)*var(i-1,j,nz,4))*haf*invdx
             d2wdx2 = mu(i,j,nz)*(var(i+1,j,nz,4)+var(i-1,j,nz,4)-2*var(i,j,nz,4))*invdx*invdx
             dvdydz = (mu(i,j+1,nz)*(b1d3c0*var(i,j+1,nz,3)+b1d3c1*var(i,j+1,nz-1,3)+b1d3c2*var(i,j+1,nz-2,3)+b1d3c3*var(i,j+1,nz-3,3)) - &
                       mu(i,j-1,nz)*(b1d3c0*var(i,j-1,nz,3)+b1d3c1*var(i,j-1,nz-1,3)+b1d3c2*var(i,j-1,nz-2,3)+b1d3c3*var(i,j-1,nz-3,3)))*haf*invdy
             dvdzdy = ((b1d3c0*mu(i,j,nz)*var(i,j+1,nz,3)+b1d3c1*mu(i,j,nz-1)*var(i,j+1,nz-1,3)+b1d3c2*mu(i,j,nz-2)*var(i,j+1,nz-2,3)+b1d3c3*mu(i,j,nz-3)*var(i,j+1,nz-3,3)) - &
                       (b1d3c0*mu(i,j,nz)*var(i,j-1,nz,3)+b1d3c1*mu(i,j,nz-1)*var(i,j-1,nz-1,3)+b1d3c2*mu(i,j,nz-2)*var(i,j-1,nz-2,3)+b1d3c3*mu(i,j,nz-3)*var(i,j-1,nz-3,3)))*haf*invdy
             drhovwdy = (var(i,j+1,nz,1)*var(i,j+1,nz,3)*var(i,j+1,nz,4) - var(i,j-1,nz,1)*var(i,j-1,nz,3)*var(i,j-1,nz,4))*haf*invdy
             d2wdy2 = mu(i,j,nz)*(var(i,j+1,nz,4)+var(i,j-1,nz,4)-2*var(i,j,nz,4))*invdy*invdy
             mud2wdz2 = (mu(i,j,nz-1)*(var(i,j,nz-2,4)-var(i,j,nz-1,4))/db23 - mu(i,j,nz)*(var(i,j,nz-1,4)-var(i,j,nz,4))/db12)/db12
             dwdzdmudz = 0.0d0
             drhow2dz = b1d3c0*var(i,j,nz,1)*var(i,j,nz,4)**two + b1d3c1*var(i,j,nz-1,1)*var(i,j,nz-1,4)**two + b1d3c2*var(i,j,nz-2,1)*var(i,j,nz-2,4)**two + b1d3c3*var(i,j,nz-3,1)*var(i,j,nz-3,4)**two
             drhowdt = ((var(i,j,nz,1)*var(i,j,nz,4)) - rhow(i,j,nz))*invdt
             lhs = invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz))) + drhow2dz + drhouwdx + drhovwdy + drhowdt
             var(i,j,nz,7) = -(lhs + b1d3c1*var(i,j,nz-1,7) + b1d3c2*var(i,j,nz-2,7) + b1d3c3*var(i,j,nz-3,7)) / b1d3c0
          endif
       enddo
    enddo
    kt(:,:,1) = 1.0d0; kt(:,:,nz) = 1.0d0
    rhow(:,:,1) = var(:,:,1,1) * var(:,:,1,4)
    rhow(:,:,nz) = var(:,:,nz,1) * var(:,:,nz,4)
    do j = bn(2), en(2)
       do i = bn(1), en(1)
          var(i,j,1,2) = 0.0d0
          var(i,j,1,3) = Amp_v * dsin(Kappa * x1(i) - Omega * time)
          var(i,j,1,6) = 1.0d0
          var(i,j,1,5) = var(i,j,1,6) + ggM * haf * sum(var(i,j,1,2:4)**two)
          var(i,j,1,1) = gama * Mac**two * var(i,j,1,7) / var(i,j,1,6)
          solV(i,j,1,1,1) = var(i,j,1,1)
          solV(i,j,1,2:3,1) = var(i,j,1,1) * var(i,j,1,2:3)
          solV(i,j,1,5,1) = var(i,j,1,1) * var(i,j,1,5)
          mu(i,j,1) = 1.0d0

          var(i,j,nz,2) = 0.0d0
          var(i,j,nz,3) = Amp_v * dsin(Kappa * x1(i) - Omega * time)
          var(i,j,nz,6) = 1.0d0
          var(i,j,nz,5) = var(i,j,nz,6) + ggM * haf * sum(var(i,j,nz,2:4)**two)
          var(i,j,nz,1) = gama * Mac**two * var(i,j,nz,7) / var(i,j,nz,6)
          solV(i,j,nz,1,1) = var(i,j,nz,1)
          solV(i,j,nz,2:3,1) = var(i,j,nz,1) * var(i,j,nz,2:3)
          solV(i,j,nz,5,1) = var(i,j,nz,1) * var(i,j,nz,5)
          mu(i,j,nz) = 1.0d0
       enddo
    enddo
  end subroutine apply_wall_bc

  subroutine compute_tau_and_rhomean()
    real(dp) :: dwdx_v, dudz_v, s_l, s_u, rsum
    s_l = 0.0d0; s_u = 0.0d0
    do j = bn(2), en(2)
       do i = bn(1), en(1)
          dwdx_v = mu(i,j,1) * (var(i+1,j,1,4) - var(i-1,j,1,4)) * haf * invdx
          dudz_v = mu(i,j,1) * (f1d2c0*var(i,j,1,2) + f1d2c1*var(i,j,2,2) + f1d2c2*var(i,j,3,2))
          s_l = s_l + (dwdx_v + dudz_v) * invRe
          dwdx_v = mu(i,j,nz) * (var(i+1,j,nz,4) - var(i-1,j,nz,4)) * haf * invdx
          dudz_v = mu(i,j,nz) * (b1d2c0*var(i,j,nz,2) + b1d2c1*var(i,j,nz-1,2) + b1d2c2*var(i,j,nz-2,2))
          s_u = s_u + (dwdx_v + dudz_v) * invRe
       enddo
    enddo
    tauxzL = s_l / dble((nx - 2) * (ny - 2))
    tauxzU = s_u / dble((nx - 2) * (ny - 2))
    do k = 1, nz
       rsum = 0.0d0
       do j = 1, ny
          do i = 1, nx
             rsum = rsum + var(i,j,k,1)
          enddo
       enddo
       stat1Spc1(k) = rsum / dble(nx * ny)
    enddo
  end subroutine compute_tau_and_rhomean

end program fortran_ref_parity
