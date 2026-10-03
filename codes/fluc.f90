program fluc
implicit none
!=======================================================================
DOUBLE PRECISION, dimension(:,:),allocatable    :: stat1
double precision, allocatable, dimension(:,:,:,:) :: var
DOUBLE PRECISION, dimension(:),allocatable      :: x1,x2,x3
DOUBLE PRECISION  	:: time, dummy_z
integer      				:: i,j,k,nstep,nP(3),st
!=======================================================================
open (unit=10,file="../input/3D.dat",status="old",iostat=st)
if (st /= 0) then
	write(0,*) "Error opening ../input/3D.dat: st=", st
	stop
endif

read (10,*)
read (10,*)
read (10,*)
read (10,*)
read (10,*)
nP(3) = 160
nP(2) = 130
nP(1) = 130
!-----------------------------------------------------------------------
allocate( stat1(nP(3),7))
allocate( var(nP(1),nP(2),nP(3),7) )
allocate(x1(nP(1)),x2(nP(2)),x3(nP(3)))
!-----------------------------------------------------------------------
do k=1,nP(3)
	do j=1,nP(2)
		do i=1,nP(1)
			read(10,*,iostat=st) x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3)	&
																								,var(i,j,k,4),var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
																								 
			if(st/=0)	then
				print *, "Error reading input file at i=", i, "j=", j, "k=", k, "st=", st
				stop
			endif	
		end do 
	end do
end do
write(0,*)"Instantaneous solution file read successfully"

close(10)
!-----------------------------------------------------------------------
open (unit=10,file="../output/mean.dat",status="old")

do k=1,nP(3)
	read (10,*,iostat=st) dummy_z, stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4)	&
																,stat1(k,5),stat1(k,6),stat1(k,7)
	
	if(st/=0)then
		write(0,*)"Problem in reading mean.dat at k=",k," st=",st
		stop
	endif
	
enddo
write(0,*)"File mean.dat read successfully"

close(10)
!-----------------------------------------------------------------------
! Find the fluctuating field
do k=1,nP(3)
	do j=1,nP(2)
		do i=1,nP(1)
			var(i,j,k,:)	=var(i,j,k,:) - stat1(k,:)
		enddo
	enddo
enddo
!-----------------------------------------------------------------------
open (unit=10,file="../output/fluc.dat")

write(10,'(A,F20.10,I9,A)')'TITLE ="',time,nstep,'"'
write(10,'(A)')'Variables ="x","y","z","rho_prime","U_prime","V_prime","W_prime","E_prime","T_prime","P_prime"' 
write(10,'(3(A,I3),A)')'ZONE k=',nP(3),',j=',nP(2),',i=',nP(1),',DATAPACKING="POINT"'

do k=1,nP(3)
	write(10,*)
	do j=1,nP(2)
		write(10,*)
		do i=1,nP(1)
			write(10,'(3(1X,F10.6),7(1X,E22.15))') x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3)	&
																								 ,var(i,j,k,4),var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
		end do
	end do
end do

close(10)
write(6,*)"File fluc.dat created"
!-----------------------------------------------------------------------
print*, "Program executed successfully"
end program fluc
