program average
!Reads Time1 history of spatially avg data and computes temporal avg

!=======================================================================
use mod_comdata
IMPLICIT NONE

character(len=20)	::str1,str2
integer					::cnt,fstat,is=1,ie,istep=1,alc,cnts(100),cnte(100)
integer					::comS,comE,flag,omit(100),red(100),opt,r1
double precision::ts(100),te(100),d1,d2,d3,d4,fgtime,rho_ratio
double precision,allocatable,dimension(:)::aTime1,Time1,VD,TL,cf3,VD_2,TL_2,cf3_2

DOUBLE PRECISION,allocatable,dimension(:,:,:)	::Astat1Spc,Astat2Spc,AcorrSpc
DOUBLE PRECISION,dimension(nP(3),7) 					:: stat1,stat2,rms
DOUBLE PRECISION,dimension(nP(3),9) 					:: corr
DOUBLE PRECISION,dimension(nP(3)) 						:: mu_mean,mu_rms

double precision															:: u_tau=0.059d0,Re_tau=192.93d0,tauWall=0.0038d0
DOUBLE PRECISION,dimension(nP(3)) 						:: u_local,z_local,z_plus,Re_local
DOUBLE PRECISION,dimension(nP(3),6) 					:: tau_unc,tau_plus,tau_local
!-----------------------------------------------------------------------
!allocate(VD(nP(3)))
!allocate(TL(nP(3)))
!allocate(cf3(nP(3)))
!allocate(VD_2(nP(3)))
!allocate(TL_2(nP(3)))
!allocate(cf3_2(nP(3)))
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
!---------------- In order of rhow,rhot,uv,uw,vw,vt,tw,pw,pt------------
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

z_plus(:)		=	x3(:)*Re_tau
z_local(:) 	= z_plus(:)*(sqrt(stat1(:,1)/stat1(1,1)))/(mu_mean(:)/mu_mean(1))

Re_local(:) = Re_tau*(sqrt(stat1(:,1)/stat1(1,1)))/(mu_mean(:)/mu_mean(1))

u_local(:)	= sqrt(tauWall/stat1(:,1))

write(6,*)"                                                                        "
write(6,*)"########################################################################"
write(6,*)"Check average code for rho in correlations and comment/uncomment in code"
write(6,*)"########################################################################"
write(6,*)"                                                                        "


tau_unc(:,1)			= stat1(:,1)*rms(:,2)**2/tauWall
tau_unc(:,2)			= stat1(:,1)*rms(:,3)**2/tauWall
tau_unc(:,3)			= stat1(:,1)*rms(:,4)**2/tauWall
tau_unc(:,4)			= corr(:,3)/tauWall
tau_unc(:,5)			= corr(:,4)/tauWall
tau_unc(:,6)			= corr(:,5)/tauWall
!tau_unc(:,4)			= stat1(:,1)*corr(:,3)/tauWall
!tau_unc(:,5)			= stat1(:,1)*corr(:,4)/tauWall
!tau_unc(:,6)			= stat1(:,1)*corr(:,5)/tauWall

tau_plus(:,1)			= stat1(:,1)*rms(:,2)**2/u_tau**2
tau_plus(:,2)			= stat1(:,1)*rms(:,3)**2/u_tau**2
tau_plus(:,3)			= stat1(:,1)*rms(:,4)**2/u_tau**2
tau_plus(:,4)			= corr(:,3)/u_tau**2
tau_plus(:,5)			= corr(:,4)/u_tau**2
tau_plus(:,6)			= corr(:,5)/u_tau**2
!tau_plus(:,4)			= stat1(:,1)*corr(:,3)/u_tau**2
!tau_plus(:,5)			= stat1(:,1)*corr(:,4)/u_tau**2
!tau_plus(:,6)			= stat1(:,1)*corr(:,5)/u_tau**2

tau_local(:,1)		= stat1(:,1)*rms(:,2)**2/u_local(:)**2
tau_local(:,2)		= stat1(:,1)*rms(:,3)**2/u_local(:)**2
tau_local(:,3)		= stat1(:,1)*rms(:,4)**2/u_local(:)**2
tau_local(:,4)		= corr(:,3)/u_local(:)**2
tau_local(:,5)		= corr(:,4)/u_local(:)**2
tau_local(:,6)		= corr(:,5)/u_local(:)**2
!tau_local(:,4)		= stat1(:,1)*corr(:,3)/u_local(:)**2
!tau_local(:,5)		= stat1(:,1)*corr(:,4)/u_local(:)**2
!tau_local(:,6)		= stat1(:,1)*corr(:,5)/u_local(:)**2

!----------------------------------------------------
!open (unit=10,file="../output/local_units.dat")

!do k=1,nP(3)/2
!	write (10,100) z_plus(k),z_local(k),u_local(k),Re_local(k)
!enddo
!close(10)

!write(6,*)"File local_units.dat created"
!100 format(7(1X,F22.15))

open (unit=10,file="../output/RSS_unc.dat")
do k=1,nP(3)/2
	write (10,200)z_plus(k),z_local(k),u_local(k),Re_local(k),tau_unc(k,1),tau_unc(k,2),tau_unc(k,3),tau_unc(k,4),tau_unc(k,5),tau_unc(k,6)
enddo
close(10)

write(6,*)"File RSS_unc.dat created"
200 format(10(1X,F22.15))
!-----------------------------------------------------------------------

open (unit=10,file="../output/RSS_plus.dat")
do k=1,nP(3)/2
	write (10,200)z_plus(k),z_local(k),u_local(k),Re_local(k),tau_plus(k,1),tau_plus(k,2),tau_plus(k,3),tau_plus(k,4),tau_plus(k,5),tau_plus(k,6)
enddo
close(10)

write(6,*)"File RSS_plus.dat created"
!-----------------------------------------------------------------------

open (unit=10,file="../output/RSS_local.dat")
do k=1,nP(3)/2
	write (10,200)z_plus(k),z_local(k),u_local(k),Re_local(k),tau_local(k,1),tau_local(k,2),tau_local(k,3),tau_local(k,4),tau_local(k,5),tau_local(k,6)
enddo
close(10)

write(6,*)"File RSS_local.dat created"
!-----------------------------------------------------------------------

write(*,*)"Program executed successfully"

end program
