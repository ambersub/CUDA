program combine
!=======================================================================
implicit none

DOUBLE PRECISION																::time,nstep
double precision,dimension(:,:),allocatable			::stat1
DOUBLE PRECISION,dimension(:,:,:,:),allocatable	::var,prime
DOUBLE PRECISION,dimension(:)    ,allocatable		::x1,x2,x3
INTEGER																					::i,j,k,nX,nY,nZ,p

!====================Read Domain Info===================================
open(unit=12,file='../input/3D.dat')
read(12,*) 
read(12,*)
read(12,104)nZ,nY,nX
close(12)
write(6,*)"Grid Points read successfully"
write(6,'(3(1X,I3))')nX,nY,nZ
!-----------------------------------------------------------------------
104	format(7X,I3,3X,I3,3X,I3)

allocate(var(nX,nY,nZ,7))
allocate(x1(nX),x2(nY),x3(nZ))
allocate(stat1(nZ,7))
allocate(prime(nX,nY,nZ,7))

!=====================read combined file===============================
open(unit=12,file='../input/3D.dat')
read(12,*)
read(12,*)
read(12,*)
do k=1,nZ
	read(12,*)
	do j=1,nY
		read(12,*)
		do i=1,nX
			read(12,101)x1(i),x2(j),x3(k),var(i,j,k,1),var(i,j,k,2),var(i,j,k,3),var(i,j,k,4)	&
																								,var(i,j,k,5),var(i,j,k,6),var(i,j,k,7)
		enddo
	enddo
enddo
close(12)

write(6,*)"File 3D.dat read"
101	format(3(1X,F10.6),7(1X,E22.15))

!-----------------------------------------------------------------------

open (unit=10,file="../output/mean.dat")
do k=1,nZ
	read (10,100)x3(k),stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4),stat1(k,5),stat1(k,6),stat1(k,7)
enddo
close(10)

write(6,*)"File mean.dat read"
100 format(8(1X,F22.18)) 

do k=1,nZ
	do j=1,nY
		do i=1,nX
			do p=1,7
				prime(i,j,k,p) = var(i,j,k,p) - stat1(k,p)
			enddo
		enddo
	enddo
enddo

open(unit=12,file='../input/Prime3D.dat')

write(12,114)'ZONE k=',nZ,',j=',nY,',i=',nX,',DATAPACKING=POINT'
do k=1,nZ
	write(12,*)
	do j=1,nY
		write(12,*)
		do i=1,nX
			write(12,101) x1(i),x2(j),x3(k),prime(i,j,k,1),prime(i,j,k,2),prime(i,j,k,3),prime(i,j,k,4)	&
																								    ,prime(i,j,k,5),prime(i,j,k,6),prime(i,j,k,7)
		enddo
	enddo
enddo
close(12)
114	format(A,I3,A,I3,A,I3,A)

write(6,*)"Prime3D.dat created"

open(unit=12,file='../input/PrimeUV.dat')
do k=12,25
	do j=1,nY
		do i=1,nX
			write(12,201) prime(i,j,k,2),prime(i,j,k,4)
		enddo
	enddo
enddo
close(12)
201	format(2(1X,E22.15))

write(6,*)"PrimeUV.dat created"
end program combine