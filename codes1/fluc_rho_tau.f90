program fluc
implicit none
!=======================================================================
DOUBLE PRECISION, dimension(:,:),allocatable    :: stat1,u
double precision, allocatable, dimension(:,:,:,:) :: var,u_1,u_2,var_2
DOUBLE PRECISION, dimension(:),allocatable      :: x1,x2,x3
DOUBLE PRECISION  	:: time,tau=0.0038d0,nXY,Re_tau=193d0
integer      				:: i,j,k,nstep,nP(3),st
!=======================================================================
open (unit=10,file="../input/3D.dat",status="old")

read (10,'(8X,F20.10,I9,1X)')time,nstep
read (10,*)
read (10,'(7X,I3,3X,I3,3X,I6)')nP(3),nP(2),nP(1)
!-----------------------------------------------------------------------
allocate( stat1(nP(3),7))
allocate( var(nP(1),nP(2),nP(3),7) )
!allocate( u_1(nP(1),nP(2),nP(3),3) )
!allocate( u_2(nP(1),nP(2),nP(3),3) )
allocate( u(nP(3),3) )
allocate( var_2(nP(1),nP(2),nP(3),7) )

allocate(x1(nP(1)),x2(nP(2)),x3(nP(3)))
!-----------------------------------------------------------------------
nXY = nP(1)*nP(2)

do k=1,nP(3)
	read(10,*)
	do j=1,nP(2)
		read(10,*)
		do i=1,nP(1)
			read(10,'(3(1X,F10.6),7(1X,E22.15))',iostat=st) x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3)	&
																													,var(i,j,k,4),var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
																								 
			if(st/=0)	then
				write(6,'(A,4(1X,I4))') "Error in reading input file at",i,j,k,st
				stop
			endif	
		end do 
	end do
end do
write(6,*)"Instantaneous solution file read successfully"

close(10)
!-----------------------------------------------------------------------
open (unit=10,file="../output/mean.dat",status="old")
	read(10,*)
	read(10,*)

do k=1,nP(3)
	read (10,'(23X,7(1X,F22.18))',iostat=st)stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4)	&
																										,stat1(k,5),stat1(k,6),stat1(k,7)
	
	if(st/=0)then
		print*,"Problem in reading mean.dat at k=",k
		stop
	endif
	
enddo
write(6,*)"File mean.dat read successfully"

close(10)
!-----------------------------------------------------------------------
var_2 = var

! Find the fluctuating field
do k=1,nP(3)
	do j=1,nP(2)
		do i=1,nP(1)
			var(i,j,k,:)	=var(i,j,k,:) - stat1(k,:)
		enddo
	enddo
enddo


do k=1,nP(3)
			u(k,1) = (sum(var_2(:,:,k,1)*var_2(:,:,k,2))/(stat1(k,1)*nXY))!**2
			u(k,2) = (sum(var_2(:,:,k,1)*var_2(:,:,k,3))/(stat1(k,1)*nXY))!**2
			u(k,3) = (sum(var_2(:,:,k,1)*var_2(:,:,k,4))/(stat1(k,1)*nXY))!**2
enddo
 
do k=1,nP(3)
	do j=1,nP(2)
		do i=1,nP(1)
			var(i,j,k,2) = sqrt(stat1(k,1))*(var_2(i,j,k,2) - u(k,1))/sqrt(tau)
			var(i,j,k,3) = sqrt(stat1(k,1))*(var_2(i,j,k,3) - u(k,2))/sqrt(tau)
			var(i,j,k,4) = sqrt(stat1(k,1))*(var_2(i,j,k,4) - u(k,3))/sqrt(tau)
		enddo
	enddo
enddo

!-----------------------------------------------------------------------
open (unit=10,file="../output/fluc_rho_tau.dat")

write(10,'(A,F20.10,I9,A)')'TITLE ="',time,nstep,'"'
write(10,'(A)')"Variables =x,y,z,rho',u',v',w',e',t',p',U_mean,V_mean,W_mean "
write(10,'(3(A,I3),A)')'ZONE k=',nP(3),',j=',nP(2),',i=',nP(1),',DATAPACKING="POINT"'

do k=1,nP(3)
	write(10,*)
	do j=1,nP(2)
		write(10,*)
		do i=1,nP(1)
			write(10,'(3(1X,F12.6),10(1X,E22.15))') x1(i)*Re_tau,x2(j)*Re_tau,x3(k)*Re_tau,var(i,j,k,1),var(i,j,k,2),var(i,j,k,3)	&
																								 ,var(i,j,k,4),var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)	&
																								 ,var_2(i,j,k,2),var_2(i,j,k,3),var_2(i,j,k,4)
		end do
	end do
end do

close(10)
write(6,*)"File fluc_rho_tau.dat created"
!-----------------------------------------------------------------------
open (unit=10,file="../output/u_pp.dat")

do k=1,nP(3)
			write(10,'(4(1X,F10.6))') x3(k),u(k,1),u(k,2),u(k,3)
end do

close(10)
write(6,*)"File u_pp.dat created"
!-----------------------------------------------------------------------
print*, "Program executed successfully"
end program fluc
