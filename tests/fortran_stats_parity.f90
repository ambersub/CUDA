program fortran_stats_parity
  ! Standalone Fortran 90 reference implementing codes1/mod_stats.f90 formulas
  ! (Diagnostic_files, PowSpec, tauWall_write, write_stats, TKE_stats,
  !  Omega_stats, Cf_stats, Favre) for Phase 2 numerical parity verification.
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer :: nx, ny, nz
  integer :: bn(3), en(3)
  integer :: planes(5), k_fft(3)
  real(dp) :: dx, dy, dXi, invdx, invdy, invdXi, invRe, Re, two3rd
  real(dp) :: low2(3), high2(3), Ayz, t_val
  real(dp), allocatable :: x1(:), x2(:), x3(:), invJac(:)
  real(dp), allocatable :: var(:,:,:,:), mu(:,:,:)

  real(dp) :: ke, tauxzL, tauxzU, fft_vals(3)
  real(dp) :: rhobulk(5), ubulk(5), massflow(5)
  real(dp), allocatable :: stat1Spc(:,:), stat2Spc(:,:), corrSpc(:,:)
  real(dp), allocatable :: fav1(:,:), fav2(:,:), fav_corr(:,:)
  real(dp), allocatable :: stat1SpcT(:,:), stat2SpcT(:,:), corrSpcT(:,:)
  real(dp), allocatable :: omega1(:,:), omega2(:,:), cf1(:,:), cf2(:,:), cf_corr(:,:)

  character(len=512) :: in_file, out_file
  integer :: ios, i, j, k, c, p, ip1, im1, jp1, jm1
  real(dp) :: nXY, nXY_2, rho_m
  real(dp) :: u_c, v_c, w_c, p_c, mu_c, dudx, dvdx, dwdx, dudy, dvdy, dwdy
  real(dp) :: dudz, dvdz, dwdz, dudz_f, dvdz_f, tauxz, tauyz, tauzz
  real(dp) :: ox, oy, oz, rho_f, var_f(7)
  real(dp) :: locT(8), locCorrT(12), s_loc(7), c_loc(9), triple_uvw(3)
  real(dp) :: dz_step, dA, r_s, u_s, m_s, var_w, var_t

  if (command_argument_count() < 2) then
     stop "Usage: fortran_stats_parity <in_bin> <out_bin>"
  endif
  call get_command_argument(1, in_file)
  call get_command_argument(2, out_file)

  two3rd = 2.0d0 / 3.0d0

  open(unit=10, file=trim(in_file), form="unformatted", access="stream", status="old", action="read", iostat=ios)
  if (ios /= 0) stop "Failed to open input binary"
  read(10) nx, ny, nz
  read(10) dx, dy, dXi, Re, t_val
  read(10) planes
  allocate(x1(nx), x2(ny), x3(nz), invJac(nz), var(nx, ny, nz, 7), mu(nx, ny, nz))
  read(10) x1
  read(10) x2
  read(10) x3
  read(10) invJac
  read(10) low2
  read(10) high2
  read(10) var
  read(10) mu
  close(10)

  bn = (/ 2, 2, 2 /)
  en = (/ nx - 1, ny - 1, nz - 1 /)
  invdx = 1.0d0 / dx
  invdy = 1.0d0 / dy
  invdXi = 1.0d0 / dXi
  invRe = 1.0d0 / Re
  nXY = dble(nx * ny)
  nXY_2 = dble((nx - 2) * (ny - 2))
  Ayz = x3(nz) * x2(ny)

  ! 1. Diagnostic_files: KE and plane-integrated bulk quantities
  ke = 0.0d0
  do k = 1, nz
     do j = 1, ny
        do i = 1, nx
           ke = ke + var(i,j,k,2)**2 + var(i,j,k,3)**2 + var(i,j,k,4)**2
        enddo
     enddo
  enddo
  ke = ke / (nXY * dble(nz))

  do p = 1, 5
     i = planes(p)
     r_s = 0.0d0; u_s = 0.0d0; m_s = 0.0d0
     do k = 1, nz - 1
        dz_step = x3(k + 1) - x3(k)
        dA = dy * dz_step
        do j = 1, ny - 1
           r_s = r_s + 0.25d0 * dA * (var(i,j,k,1) + var(i,j+1,k,1) + var(i,j,k+1,1) + var(i,j+1,k+1,1))
           u_s = u_s + 0.25d0 * dA * (var(i,j,k,2) + var(i,j+1,k,2) + var(i,j,k+1,2) + var(i,j+1,k+1,2))
           m_s = m_s + 0.25d0 * dA * ( &
                var(i,j,k,1)*var(i,j,k,2) + var(i,j+1,k,1)*var(i,j+1,k,2) + &
                var(i,j,k+1,1)*var(i,j,k+1,2) + var(i,j+1,k+1,1)*var(i,j+1,k+1,2))
        enddo
     enddo
     rhobulk(p) = r_s / Ayz
     ubulk(p) = u_s / Ayz
     massflow(p) = m_s / Ayz
  enddo

  ! 2. PowSpec: var(bn(1)+5, bn(2)+5, (16, 36, 52), 2) (clamped safely if small)
  k_fft = (/ min(16, nz), min(36, nz), min(52, nz) /)
  fft_vals(1) = var(min(bn(1)+5, nx-1), min(bn(2)+5, ny-1), k_fft(1), 2)
  fft_vals(2) = var(min(bn(1)+5, nx-1), min(bn(2)+5, ny-1), k_fft(2), 2)
  fft_vals(3) = var(min(bn(1)+5, nx-1), min(bn(2)+5, ny-1), k_fft(3), 2)

  ! 3. tauWall_write (with dwdx term on interior bn..en)
  tauxzL = 0.0d0
  tauxzU = 0.0d0
  do j = bn(2), en(2)
     do i = bn(1), en(1)
        im1 = i - 1; ip1 = i + 1
        if (i == bn(1)) im1 = en(1)
        if (i == en(1)) ip1 = bn(1)
        dwdx = 0.5d0 * (var(ip1,j,1,4) - var(im1,j,1,4)) * invdx
        dudz = low2(1)*var(i,j,1,2) + low2(2)*var(i,j,2,2) + low2(3)*var(i,j,3,2)
        tauxzL = tauxzL + mu(i,j,1) * (dwdx + dudz) * invRe

        dwdx = 0.5d0 * (var(ip1,j,nz,4) - var(im1,j,nz,4)) * invdx
        dudz = high2(1)*var(i,j,nz,2) + high2(2)*var(i,j,nz-1,2) + high2(3)*var(i,j,nz-2,2)
        tauxzU = tauxzU + mu(i,j,nz) * (dwdx + dudz) * invRe
     enddo
  enddo
  tauxzL = tauxzL / nXY_2
  tauxzU = tauxzU / nXY_2

  ! 4. Spatial moments (1..nx, 1..ny) & Favre moments (mod_stats.f90 stats/Favre)
  allocate(stat1Spc(nz, 7), stat2Spc(nz, 7), corrSpc(nz, 9))
  allocate(fav1(nz, 7), fav2(nz, 7), fav_corr(nz, 9))
  stat1Spc = 0.0d0; stat2Spc = 0.0d0; corrSpc = 0.0d0
  fav1 = 0.0d0; fav2 = 0.0d0; fav_corr = 0.0d0

  do k = 1, nz
     triple_uvw = 0.0d0
     do j = 1, ny
        do i = 1, nx
           var_f(1:7) = var(i,j,k,1:7)
           rho_f = var_f(1)
           stat1Spc(k, :) = stat1Spc(k, :) + var_f(:)
           stat2Spc(k, :) = stat2Spc(k, :) + var_f(:)**2
           triple_uvw(1) = triple_uvw(1) + var_f(2)**2 * var_f(4)
           triple_uvw(2) = triple_uvw(2) + var_f(3)**2 * var_f(4)
           triple_uvw(3) = triple_uvw(3) + var_f(4)**3

           c_loc(1) = var_f(1) * var_f(4)
           c_loc(2) = var_f(1) * var_f(6)
           c_loc(3) = var_f(2) * var_f(3)
           c_loc(4) = var_f(2) * var_f(4)
           c_loc(5) = var_f(3) * var_f(4)
           c_loc(6) = var_f(3) * var_f(6)
           c_loc(7) = var_f(4) * var_f(6)
           c_loc(8) = var_f(4) * var_f(7)
           c_loc(9) = var_f(6) * var_f(7)
           corrSpc(k, :) = corrSpc(k, :) + c_loc(:)

           fav1(k, :) = fav1(k, :) + rho_f * var_f(:)
           fav2(k, :) = fav2(k, :) + (rho_f * var_f(:))**2
           fav_corr(k, :) = fav_corr(k, :) + rho_f * c_loc(:)
        enddo
     enddo
     stat1Spc(k, :) = stat1Spc(k, :) / nXY
     stat2Spc(k, :) = stat2Spc(k, :) / nXY
     corrSpc(k, :)  = corrSpc(k, :) / nXY
     rho_m = stat1Spc(k, 1)
     fav1(k, :) = fav1(k, :) / (nXY * rho_m)
     fav2(k, :) = fav2(k, :) / (nXY * rho_m * rho_m)
     fav_corr(k, :) = fav_corr(k, :) / (nXY * rho_m)
  enddo

  ! 5. TKE_stats, Omega_stats, Cf_stats on interior bn..en
  allocate(stat1SpcT(nz, 8), stat2SpcT(nz, 8), corrSpcT(nz, 12))
  allocate(omega1(nz, 3), omega2(nz, 3), cf1(nz, 7), cf2(nz, 7), cf_corr(nz, 2))
  stat1SpcT = 0.0d0; stat2SpcT = 0.0d0; corrSpcT = 0.0d0
  omega1 = 0.0d0; omega2 = 0.0d0; cf1 = 0.0d0; cf2 = 0.0d0; cf_corr = 0.0d0

  do k = 1, nz
     triple_uvw = 0.0d0
     do j = 1, ny
        do i = 1, nx
           triple_uvw(1) = triple_uvw(1) + var(i,j,k,2)**2 * var(i,j,k,4)
           triple_uvw(2) = triple_uvw(2) + var(i,j,k,3)**2 * var(i,j,k,4)
           triple_uvw(3) = triple_uvw(3) + var(i,j,k,4)**3
        enddo
     enddo
     triple_uvw = 0.5d0 * stat1Spc(k, 1) * (triple_uvw / nXY)

     dz_step = x3(min(k+1, nz)) - x3(max(k, 1))
     if (k == nz) dz_step = x3(nz) - x3(nz - 1)

     do j = bn(2), en(2)
        jm1 = j - 1; jp1 = j + 1
        if (j == bn(2)) jm1 = en(2)
        if (j == en(2)) jp1 = bn(2)
        do i = bn(1), en(1)
           im1 = i - 1; ip1 = i + 1
           if (i == bn(1)) im1 = en(1)
           if (i == en(1)) ip1 = bn(1)

           u_c = var(i,j,k,2); v_c = var(i,j,k,3); w_c = var(i,j,k,4)
           p_c = var(i,j,k,7); mu_c = mu(i,j,k)

           dudx = 0.5d0 * (var(ip1,j,k,2) - var(im1,j,k,2)) * invdx
           dvdx = 0.5d0 * (var(ip1,j,k,3) - var(im1,j,k,3)) * invdx
           dwdx = 0.5d0 * (var(ip1,j,k,4) - var(im1,j,k,4)) * invdx

           dudy = 0.5d0 * (var(i,jp1,k,2) - var(i,jm1,k,2)) * invdy
           dvdy = 0.5d0 * (var(i,jp1,k,3) - var(i,jm1,k,3)) * invdy
           dwdy = 0.5d0 * (var(i,jp1,k,4) - var(i,jm1,k,4)) * invdy

           if (k == 1) then
              dudz = low2(1)*var(i,j,1,2) + low2(2)*var(i,j,2,2) + low2(3)*var(i,j,3,2)
              dvdz = low2(1)*var(i,j,1,3) + low2(2)*var(i,j,2,3) + low2(3)*var(i,j,3,3)
              dwdz = low2(1)*var(i,j,1,4) + low2(2)*var(i,j,2,4) + low2(3)*var(i,j,3,4)
           elseif (k == nz) then
              dudz = high2(1)*var(i,j,nz,2) + high2(2)*var(i,j,nz-1,2) + high2(3)*var(i,j,nz-2,2)
              dvdz = high2(1)*var(i,j,nz,3) + high2(2)*var(i,j,nz-1,3) + high2(3)*var(i,j,nz-2,3)
              dwdz = high2(1)*var(i,j,nz,4) + high2(2)*var(i,j,nz-1,4) + high2(3)*var(i,j,nz-2,4)
           else
              dudz = 0.5d0 * (var(i,j,k+1,2) - var(i,j,k-1,2)) * invdXi * invJac(k)
              dvdz = 0.5d0 * (var(i,j,k+1,3) - var(i,j,k-1,3)) * invdXi * invJac(k)
              dwdz = 0.5d0 * (var(i,j,k+1,4) - var(i,j,k-1,4)) * invdXi * invJac(k)
           endif

           tauxz = mu_c * (dwdx + dudz) * invRe
           tauyz = mu_c * (dwdy + dvdz) * invRe
           tauzz = mu_c * (2.0d0*dwdz - two3rd*(dudx + dvdy + dwdz)) * invRe

           locT = (/ dudx, dudz, tauxz, dvdz, dwdz, dvdy, tauyz, tauzz /)
           stat1SpcT(k, :) = stat1SpcT(k, :) + locT(:)
           stat2SpcT(k, :) = stat2SpcT(k, :) + locT(:)**2

           corrSpcT(k, 1)  = corrSpcT(k, 1)  + p_c * dudx
           corrSpcT(k, 2)  = corrSpcT(k, 2)  + u_c * tauxz
           corrSpcT(k, 3)  = corrSpcT(k, 3)  + tauxz * dudz
           corrSpcT(k, 5)  = corrSpcT(k, 5)  + p_c * dvdy
           corrSpcT(k, 6)  = corrSpcT(k, 6)  + v_c * tauyz
           corrSpcT(k, 7)  = corrSpcT(k, 7)  + tauyz * dvdz
           corrSpcT(k, 9)  = corrSpcT(k, 9)  + p_c * dwdz
           corrSpcT(k, 10) = corrSpcT(k, 10) + w_c * tauzz
           corrSpcT(k, 11) = corrSpcT(k, 11) + tauzz * dwdz

           if (k < nz) then
              dudz_f = (var(i,j,k+1,2) - var(i,j,k,2)) / dz_step
              dvdz_f = (var(i,j,k+1,3) - var(i,j,k,3)) / dz_step
           else
              dudz_f = (var(i,j,nz,2) - var(i,j,nz-1,2)) / dz_step
              dvdz_f = (var(i,j,nz,3) - var(i,j,nz-1,3)) / dz_step
           endif
           ox = dwdy - dvdz_f
           oy = dwdx - dudz_f
           oz = dvdx - dudy
           omega1(k, 1) = omega1(k, 1) + ox
           omega1(k, 2) = omega1(k, 2) + oy
           omega1(k, 3) = omega1(k, 3) + oz
           omega2(k, 1) = omega2(k, 1) + ox*ox
           omega2(k, 2) = omega2(k, 2) + oy*oy
           omega2(k, 3) = omega2(k, 3) + oz*oz
           cf_corr(k, 1) = cf_corr(k, 1) + w_c * (-0.5d0 * oy)
           cf_corr(k, 2) = cf_corr(k, 2) + v_c * (0.5d0 * oz)
        enddo
     enddo

     stat1SpcT(k, :) = stat1SpcT(k, :) / nXY_2
     stat2SpcT(k, :) = stat2SpcT(k, :) / nXY_2
     corrSpcT(k, :)  = corrSpcT(k, :) / nXY_2
     corrSpcT(k, 4)  = triple_uvw(1)
     corrSpcT(k, 8)  = triple_uvw(2)
     corrSpcT(k, 12) = triple_uvw(3)

     omega1(k, :) = omega1(k, :) / nXY_2
     omega2(k, :) = omega2(k, :) / nXY_2
     cf1(k, 1:4) = stat1Spc(k, 1:4)
     cf2(k, 1:4) = stat2Spc(k, 1:4)
     cf1(k, 5) = 0.5d0 * omega1(k, 1)
     cf1(k, 6) = -0.5d0 * omega1(k, 2)
     cf1(k, 7) = 0.5d0 * omega1(k, 3)
     cf2(k, 5:7) = 0.25d0 * omega2(k, 1:3)
     cf_corr(k, :) = cf_corr(k, :) / nXY_2
  enddo

  open(unit=20, file=trim(out_file), form="unformatted", access="stream", status="replace", iostat=ios)
  if (ios /= 0) stop "Failed to open output binary"
  write(20) ke, rhobulk, ubulk, massflow, fft_vals, tauxzL, tauxzU
  write(20) stat1Spc, stat2Spc, corrSpc
  write(20) fav1, fav2, fav_corr
  write(20) stat1SpcT, stat2SpcT, corrSpcT
  write(20) omega1, omega2, cf1, cf2, cf_corr
  close(20)

end program fortran_stats_parity
