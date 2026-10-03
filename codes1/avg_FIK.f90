program average
!Reads Time1 history of spatially avg data and computes temporal avg

!=======================================================================
use mod_comdata
IMPLICIT NONE

character(len=20)	::str1,str2
integer					::cnt,fstat,is=1,ie,istep=1,alc,cnts(100),cnte(100)
integer					::comS,comE,flag,omit(100),red(100),opt,r1
double precision::ts(100),te(100),d1,d2,d3,d4,fgtime
double precision,allocatable,dimension(:)::aTime1,Time1,cf1,cf2,cf3,cf1_2,cf2_2,cf3_2

DOUBLE PRECISION,allocatable,dimension(:,:,:)	::Astat1Spc,Astat2Spc,AcorrSpc
DOUBLE PRECISION,dimension(nP(3),7) 					:: stat1,stat2,rms
DOUBLE PRECISION,dimension(nP(3),9) 					:: corr
DOUBLE PRECISION,dimension(nP(3)) 						:: mu_mean,mu_rms
!-----------------------------------------------------------------------
allocate(cf1(nP(3)))
allocate(cf2(nP(3)))
allocate(cf3(nP(3)))
allocate(cf1_2(nP(3)))
allocate(cf2_2(nP(3)))
allocate(cf3_2(nP(3)))
!-----------------------------------------------------------------------
open (unit=10,file="../output/mean.dat")
read(10,*)
read(10,*)

do k=1,nP(3)
	read (10,101)x3(k),stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4),stat1(k,5),stat1(k,6),stat1(k,7)
enddo
close(10)

write(6,*)"File mean.dat read"
!-----------------------------------------------------------------------
open (unit=10,file="../output/rms.dat")
read(10,*)
read(10,*)
do k=1,nP(3)
 	read (10,101)x3(k),rms(k,1),rms(k,2),rms(k,3),rms(k,4),rms(k,5),rms(k,6),rms(k,7)
 enddo
 close(10)
101 format(8(1X,F22.15)) 

 write(6,*)"File rms.dat read"
!-----------------------------------------------------------------------
open (unit=10,file="../output/corr.dat")
read(10,*)
read(10,*)
do k=1,nP(3)
	read (10,105)x3(k),corr(k,1),corr(k,2),corr(k,3),corr(k,4),corr(k,5),corr(k,6),corr(k,7),corr(k,8),corr(k,9)
enddo
close(10)

write(6,*)"File corr.dat read"
!100 format(3(1X,F22.15)) 
105 format(10(1X,F10.5))
!----------------------------------------
mu_mean(:)	= stat1(:,6)**0.7d0
mu_rms(:)		=   rms(:,6)**0.7d0
!------------------- CF2 OR CT ---------------------
do k=1,80
	if (k==1) then
		cf1(k) 	 = (1-x3(k))*corr(k,4)*(x3(k+1)-x3(k))
		cf1_2(k) = (1-x3(k))*corr(k,4)*(x3(k+1)-x3(k))
		else
		cf1(k) 	 = (1-x3(k))*corr(k,4)*(x3(k+1)-x3(k)) + cf1(k-1)
		cf1_2(k) = (1-x3(k))*corr(k,4)*(x3(k+1)-x3(k)) 
	endif
enddo

!-----------------------------------------------------------------------

do k=1,80
	if (k==1) then
		cf2(k) 	 = (1-x3(k))*(mu_mean(k)/mu_mean(1)-1.0d0)*((stat1(k+1,2)-stat1(k,2))/(x3(k+1)-x3(k)))*(x3(k+1)-x3(k))
		cf2_2(k) = (1-x3(k))*(mu_mean(k)/mu_mean(1) -1.0d0)*((stat1(k+1,2)-stat1(k,2))/(x3(k+1)-x3(k)))*(x3(k+1)-x3(k))
		else
		cf2(k) 	 = (1-x3(k))*(mu_mean(k)/mu_mean(1) -1.0d0)*((stat1(k+1,2)-stat1(k-1,2))/(x3(k+1)-x3(k-1)))*(x3(k+1)-x3(k)) + cf2(k-1)
		cf2_2(k) = (1-x3(k))*(mu_mean(k)/mu_mean(1) -1.0d0)*((stat1(k+1,2)-stat1(k-1,2))/(x3(k+1)-x3(k-1)))*(x3(k+1)-x3(k))
	endif
enddo

do k=1,80
	if (k==1) then
		cf3(k) 	 = (1-x3(k))*(mu_rms(k))*((rms(k+1,2)-rms(k,2))/(x3(k+1)-x3(k)))*(x3(k+1)-x3(k))
		cf3_2(k) = (1-x3(k))*(mu_rms(k))*((rms(k+1,2)-rms(k,2))/(x3(k+1)-x3(k)))*(x3(k+1)-x3(k))
		else
		cf3(k)	 = (1-x3(k))*(mu_rms(k))*((rms(k+1,2)-rms(k-1,2))/(x3(k+1)-x3(k-1)))*(x3(k+1)-x3(k)) + cf3(k-1)
		cf3_2(k) = (1-x3(k))*(mu_rms(k))*((rms(k+1,2)-rms(k-1,2))/(x3(k+1)-x3(k-1)))*(x3(k+1)-x3(k))
	endif
enddo
!-----------------------------------------------------------------------
open (unit=10,file="../output/FIK.dat")
do k=1,nP(3)/2
	write (10,100)x3(k), -6.0d0*cf1(k), 6.0d0*cf2(k)/3000, 6.0d0*cf3(k)/3000, -6.0d0*cf1_2(k), 6.0d0*cf2_2(k)/3000, 6.0d0*cf3_2(k)/3000
enddo
close(10)

write(6,*)"File FIK.dat created"
100 format(7(1X,F22.15))
!-----------------------------------------------------------------------
open (unit=10,file="../output/mu.dat")
do k=1,nP(3)/2
	write (10,200)x3(k), mu_mean(k), mu_rms(k)
enddo
close(10)

write(6,*)"File FIK.dat created"
200 format(3(1X,F22.15))
write(*,*)"Program executed successfully"

end program
