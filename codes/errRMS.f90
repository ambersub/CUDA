program average
!Reads time history of spatially avg data and computes temporal avg
!Developer: Parvez Ahmad
!=======================================================================
IMPLICIT NONE
integer,parameter,dimension(3)					::nP			=[130,130,160]

character(len=20)	::str1,str2
integer					::cnt,fstat,i,is=1,ie,istep=1,alc,j,k,cnts(100),cnte(100)
integer					::comS,comE,flag,omit(100),red(100),opt,p,r,Nk
double precision::ts(100),te(100),d1,d2,d3,d4,fgtime
double precision,allocatable,dimension(:)::aTime,time
double precision 	:: dXi,dx,dy,Pi,x1(nP(1)),x2(nP(2)),invdx,invdy,invdXi
double precision :: x3(nP(3)),jac(nP(3)),jac2(nP(3)),Xi(nP(3)),invJac(nP(3)),z(119),CKM(119,6),CKM_new(160,6),err(160,6)


DOUBLE PRECISION,allocatable,dimension(:,:,:)	::stat1Spc,stat2Spc,corrSpc
DOUBLE PRECISION,dimension(nP(3),7) 					:: stat1,stat2,rms
DOUBLE PRECISION,dimension(nP(3),9) 					:: corr,covar
!-----------------------------------------------------------------------
!-------------------------------------------------------

!open (unit=10,file="../output/mean.dat")
!do k=1,nP(3)
!	read (10,100)x3(k),stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4),stat1(k,5),stat1(k,6),stat1(k,7)
!enddo
!close(10)

!write(6,*)"File mean.dat created"
100 format(8(1X,F22.18)) 
!-----------------------------------------------------------------------

open (unit=10,file="../output/rms.dat")
do k=1,nP(3)
	read (10,100)x3(k),rms(k,1),rms(k,2),rms(k,3),rms(k,4),rms(k,5),rms(k,6),rms(k,7)
enddo
close(10)

write(6,*)"File rms.dat read"
!-----------------------------------------------------------------------
open (unit=10,file="../output/corr.dat")
do k=1,nP(3)
	read (10,101)x3(k),covar(k,1),covar(k,2),covar(k,3),covar(k,4),covar(k,5),covar(k,6),covar(k,7),covar(k,8),covar(k,9)
enddo
close(10)

write(6,*)"File corr.dat read"
101 format(10(1X,F10.5))
!-----------------------------------------------------------------------
open (unit=10,file="rms_160.dat")
do k=1,nP(3)
	read (10,201)x3(k),CKM_new(k,1),CKM_new(k,2),CKM_new(k,3),CKM_new(k,4),CKM_new(k,5),CKM_new(k,6)
enddo
close(10)

write(6,*)"File rms_160.dat read"
201 format(E17.10,6(6X,E12.5))

err(:,1) = 100*(CKM_new(:,1) -   rms(:,2))/CKM_new(:,1)			!uRMS
err(:,2) = 100*(CKM_new(:,2) -   rms(:,3))/CKM_new(:,2)			!vRMS
err(:,3) = 100*(CKM_new(:,3) -   rms(:,4))/CKM_new(:,3)			!wRMS
err(:,4) = 100*(CKM_new(:,4) -   rms(:,6))/CKM_new(:,4)			!tRMS
err(:,5) = 100*(CKM_new(:,5) - covar(:,4))/CKM_new(:,5)  		!uw
err(:,6) = 100*(CKM_new(:,6) - covar(:,7))/CKM_new(:,6)	  	!wt

open (unit=10,file="err.dat")
do k=2,nP(3)-1
	write (10,201)x3(k),err(k,1),err(k,2),err(k,3),err(k,4),err(k,5),err(k,6)
enddo
close(10)

write(6,*)"File err.dat created"
!201 format(E17.10,6(6X,E12.5))
write(*,*)"Program executed successfully"

end