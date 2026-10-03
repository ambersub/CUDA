module mod_in_out

use mod_my_mpi

implicit none

contains

subroutine read_input_files()

	implicit none

	select case(resOp)
	case(1)
		!======================Read Combined Input File=====================
		open(unit=10,file=inputPath)
			read (10,100)time,nstep
			read (10,*)
			read (10,*)

		100	format(8X,F20.10,I9,1X)

		do k=1,nP(3)
			read(10,*)
			do j=1,nP(2)
				read(10,*)
				do i=1,nP(1)
					read(10,101,iostat=st) x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3),var(i,j,k,4)	&
																															,var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
					if(st/=0)	then
						write(6,'(A,4(1X,I4))') "Error in reading combined input file at",i,j,k,st
						stop
					endif
				end do
			end do
		end do

		close(10)
		101	format(3(1X,F10.6),7(1X,E22.15))
	case(2)
		!======================Read Component Input File====================
		WRITE(filename,'(a,i3.3,a)')"../output/3D",id+1,".dat"
		OPEN (UNIT=10,FILE=filename)

		read(10,100) time,nstep
		read(10,*)
		read(10,*)

		do k=bs(3),es(3)
			read(10,*)
			do j=bs(2),es(2)
				read(10,*)
				do i=bs(1),es(1)
					read(10,101,iostat=st) x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3),var(i,j,k,4)	&
																															,var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
					if(st/=0)	then
						write(6,'(A,5(1X,I4))') "Error in reading component input file at",id,i,j,k,st
						write(6,'(A)') "Change in MPI process topology since last run could possibly be a reason"
						stop
					endif

				end do
			end do
		enddo
		close(10)
			!-----------------------------------------------------------------------
	end select

	if(master) write(6,'(A)') "Input file successfully read"

end subroutine read_input_files

subroutine check_blown()

	implicit none
	!=====================Check for unphysical values=====================
	do k=bs(3),es(3)
		do j=bs(2),es(2)
			do i=bs(1),es(1)

				if (var(i,j,k,1) < 0.0d0 .or. var(i,j,k,6) < 0.0d0) then
					write(6,102) nstep,id,i,j,k,var(i,j,k,1),var(i,j,k,6)
					isBlown=.true.
				endif

			enddo
		enddo
	enddo

	if(isBlown) then

		WRITE(filename,'(a,i3.3,a)')"../debug/3D",id+1,".dat"
		OPEN (UNIT=10,FILE=filename)

		write(10,103) 'TITLE ="',time,nstep,'"'
		write(10,*) 'Variables ="x","y","z","Rho","U","V","W","E","T","P"'
		write(10,104) 'ZONE k=',es(3)-bs(3)+1,',j=',es(2)-bs(2)+1,',i=',es(1)-bs(1)+1,',DATAPACKING="POINT"'

		do k=bs(3),es(3)
			write(10,*)
			do j=bs(2),es(2)
				write(10,*)
				do i=bs(1),es(1)
					write(10,116) x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3),var(i,j,k,4)	&
																											,var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
				end do
			end do
		enddo
		close(10)
		stop
	endif

	102 format(I10,4(I4),2(F8.4))
	116 format(3(1X,F7.4),7(1X,E12.5))
	103	format(A,F20.10,I9,A)
	104	format(A,I3,A,I3,A,I3,A)

end subroutine check_blown

subroutine write_output_files()

	!==================Write output files (starts)========================
		if(mod(nstep/1000,2).eq.0) then
			WRITE(filename,'(a,i3.3,a)')"../output/3D",id+1,".dat"
		else
			WRITE(filename,'(a,i3.3,a)')"../output/3D",id+1,"_.dat"
		endif

		OPEN (UNIT=10,FILE=filename)

		write(10,103) 'TITLE ="',time,nstep,'"'
		write(10,*) 'Variables ="x","y","z","Rho","U","V","W","E","T","P"'
		write(10,104) 'ZONE k=',es(3)-bs(3)+1,',j=',es(2)-bs(2)+1,',i=',es(1)-bs(1)+1,',DATAPACKING="POINT"'

		do k=bs(3),es(3)
			write(10,*)
			do j=bs(2),es(2)
				write(10,*)
				do i=bs(1),es(1)
					write(10,101) x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3),var(i,j,k,4)	&
																											,var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
				end do
			end do
		enddo
		close(10)
		103	format(A,F20.10,I9,A)
		104	format(A,I3,A,I3,A,I3,A)
		101	format(3(1X,F10.6),7(1X,E22.15))

		!-------------------------------------------------------------------
		call mpi_gather(bs(1),3,mpi_integer,bb(1,0),3,mpi_integer,0,mpi_comm_world,ierr)
		call mpi_gather(es(1),3,mpi_integer,ee(1,0),3,mpi_integer,0,mpi_comm_world,ierr)
		!-----------------------Write Domain Info---------------------------
		if(master) then
			open(unit=10,file="../output/domInfo.txt")
			write(10,105) nP(1),nP(2),nP(3)
			write(10,106) nproc
			do i=0,nproc-1
				write(10,107) bb(1,i),ee(1,i),bb(2,i),ee(2,i),bb(3,i),ee(3,i)
			enddo
			close(10)
			105 format(3(i4))
			106 format(i3)
			107 format(6(i4,1x))
		endif

end subroutine write_output_files

end module mod_in_out