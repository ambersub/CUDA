program chanCompr
!=======================================================================
use mod_control

implicit none

call readParam() 										!located in mod_comdata
call setupMesh()										!located in mod_comdata
call Coefficients()									!located in mod_comdata
call initialize_mpi()								!located in mod_my_mpi
call print_param()									!located in mod_comdata
call read_input_files()							!located in mod_in_out
call initial_solnV()								!located in mod_comdata
call transfer_2p_solV()							!located in mod_my_mpi
call transfer_1p_var()							!located in mod_my_mpi
!========================Main time loop (starts)========================
DO WHILE (nstep < maxstep)
	if(master .and. mod(nstep,100)==0) call cpu_time(tStart)
	nstep = nstep+1
	time  = time + dt
	!========================RK2 Sub-steps (starts)=======================
	do iter=1,maxiter
		call calc_convective_terms()		!located in mod_conv
		!call calc_diffusive_terms_2D()	!located in mod_diff
		call calc_diffusive_terms_3D()	!located in mod_diff
		call transfer_FHD()							!located in mod_my_mpi
		call calc_FHD_and_srcV()				!located in mod_diff
		call transfer_2p_solV()					!located in mod_my_mpi
		call find_primary_variable()		!located in mod_comdata--Finding Primitive Variables
		call transfer_1p_var()					!located in mod_my_mpi
		call OC()												!located in mod_control
		!call bc_update_2D()						!Pressure BC
		call bc_update_3D()							!Pressure BC
		call transfer_xyline()					!located in mod_my_mpi
	enddo
	!=========================RK2 Sub-steps (ends)========================
	call check_blown()								!located in mod_in_out
	if(mod(nstep,1000).eq.0) then
		call write_output_files()				!located in mod_stats
	endif
	!==================Write output files (ends)==========================
	if (mod(nstep,10).eq.0) then
		call Diagnostic_files()					!located in mod_stats
		call PowSpec()
	endif
	!==================Spatial Statistics (starts)========================
	if(mod(nstep,istat)==0) then
		call tauWall_write()						!located in mod_stats
		!call write_stats()							!located in mod_stats
		call TKE_stats()								!located in mod_stats--Open one stats at a time
		!call NU_stats()								!located in mod_stats--create NU folfer in output and check write loop
		!call write_stats_wMax()
		call Omega_stats()
		!call tauWall_Lxt()
		call Cf_stats()
		call Favre()
	endif
	if(master .and. mod(nstep,100)==0) then
	!====================Spatial Statistics (ends)========================
		call cpu_time(tEnd)
!		write(6,114) nstep,time," | CPU Time:",tEnd-tStart,"sec for 10 iters | ", dateTime(), var(5,5,1,7), var(5,5,y_d,7), var(5,5,1,4), var(5,5,y_d,4)
		write(6,114) nstep,time," | CPU Time:",tEnd-tStart,"sec for 10 iters | ", dateTime(), var(5,5,1,7), var(5,5,y_d,7), var(5,5,1,3), var(5,5,y_d,3), var(5,5,1,4), var(5,5,y_d,4)
		write(6,'(A)')"-----------------------------------------------------------------------------------"
	endif
	114  format(I10,2X,F10.5,A,F6.3,2(1X,A),6(F10.5))
!	114  format(I10,2X,F10.5,A,F6.3,2(1X,A),4(F10.5))
	!114  format(I10,2X,F10.5,A,F6.3,2(1X,A))
END DO
!========================Main time loop (ends)==========================
call finalize_mpi()									!located in mod_my_mpi
!=======================================================================
if(master) write(*,*) "Compressible Channel Flow Solver exited at :",dateTime()
!=======================================================================
end program chanCompr