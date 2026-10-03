program average
!Reads Time1 history of spatially avg data and computes temporal avg
!Developer: Parvez Ahmad
!=======================================================================
use mod_comdata
IMPLICIT NONE

character(len=20)	::str1,str2
integer					::cnt,fstat,is=1,ie,istep=1,alc,cnts(100),cnte(100)
integer					::comS,comE,flag,omit(100),red(100),opt,r1
double precision::ts(100),te(100),d1,d2,d3,d4,fgtime,Ma=1.5
double precision,allocatable,dimension(:)::aTime1,Time1

DOUBLE PRECISION,allocatable,dimension(:,:,:)	::Astat1Spc,Astat2Spc,AcorrSpc
DOUBLE PRECISION,dimension(nP(3),7) 					::stat1,stat2,rms
DOUBLE PRECISION,dimension(nP(3),9) 					::corr,covar
!----------------------------------TKE---------------------------------
DOUBLE PRECISION,allocatable,dimension(:,:,:)	::Astat1SpcT,Astat2SpcT,AcorrSpcT
DOUBLE PRECISION,dimension(nP(3),8) 					::stat1T,stat2T,rmsT
DOUBLE PRECISION,dimension(nP(3),12) 					::corrT,covarT
double precision															::Pk(nP(3)),Pik(nP(3)),Tk(nP(3)),Dk(nP(3)),Epk(nP(3))

double precision,dimension(nP(3))							::TKE, MACH, Tur_Mach
!-----------------------------------------------------------------------
call get_command_argument(1,str1)
call get_command_argument(2,str2)
if(command_argument_count()==1) read(str1,*) is

omit(:)=0
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/moment1st.dat",status="old")
do
	read(12,*,iostat=fstat)
	if(fstat/=0) exit
	cnt=cnt+1
enddo
write(*,180)" Total no of lines in the file",cnt
close(12)

if(command_argument_count()==2) then
	read(str1,*) is
	read(str2,*) cnt
	cnt=cnt+1
endif	
	
!---------------Obtains first Column of Data----------------------------
allocate(Time1(0:cnt))
open(unit=12,file="../output/moment1st.dat",status="old")
do i=1,cnt
	read(12,111,iostat=fstat) Time1(i)
enddo
close(12)
150 format(f4.1,1x)
!------------------Finds out multiple intervals if present--------------
k=1
red(:)=1
!opt=1
comS=1
comE=1
Time1(is-1)=0
ts(k)=Time1(is)
cnts(k)=is
do i=is,cnt
	if(Time1(i) .lt. Time1(i-1)) then
		te(k)=Time1(i-2)
		cnte(k)=i-2
		k=k+1
		ts(k)=Time1(i)
		cnts(k)=i
	endif
enddo
te(k)=Time1(cnt-1)
cnte(k)=cnt-1

if(k > 1) then
	write(*,'(A,i6)')" File contains multiple intervals starting from line",is
	do i=1,k
	write(*,130) i,")",ts(i)," to ",te(i)," | line",cnts(i)," to ",cnte(i)
	enddo
	130 format(i2,2(a,f15.7),2(a,i10))
	
	write(*,'(a)',advance='no')" Choose interval/type 100 to combine: "
	read(*,*) opt
	comS=opt
	comE=opt
	
	if(opt==100) then
		write(*,'(a)',advance='no')" Enter index of starting interval: "
		read(*,*) comS
		write(*,'(a)',advance='no')" Enter index of ending interval: "
		read(*,*) comE
	
!-------------Skipping duplicates Time1 strands--------------------------	
		j=comS-1
		flag=1
		fgtime=0.0d0
		
		do i=cnts(comS),cnte(comE)
		
			if(Time1(i) .lt. Time1(i-1) .and. Time1(i) .le. fgtime) red(j)=red(j)+1
			
			if(Time1(i) .lt. Time1(i-1) .and. i .gt. cnts(comS) .and. Time1(i) .gt. fgtime) then
				flag=0
				fgtime=Time1(i-1)
				j=j+1
			endif
			if(Time1(i) .gt. fgtime) flag=1
			if(flag==0) omit(j)=omit(j)+1
		enddo
		write(*,*)"Total duplicate strands ignored ",sum(omit)
	endif
	
endif
!-----------------------------------------------------------------------
alc=ceiling(dble(cnte(comE)-cnts(comS)-sum(omit)+1)/istep)
write(*,180)" Total lines averaged",alc
180 format(a,i10)
!------------------------------------------------
allocate(Astat1Spc(nP(3),7,alc))
allocate(Astat2Spc(nP(3),7,alc))
allocate(AcorrSpc (nP(3),9,alc))
allocate( aTime1(alc))

!-----------------------------------------------------------------------
call setupMesh()
!-----------------------------------------------------------------------
open(unit=12,file="../output/moment1st.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		read(12,112,advance="no",iostat=fstat)Astat1Spc(k,1,i),Astat1Spc(k,2,i),Astat1Spc(k,3,i),Astat1Spc(k,4,i)	&
																												 ,Astat1Spc(k,5,i),Astat1Spc(k,6,i),Astat1Spc(k,7,i)
		if(fstat/=0) then
			write(*,*)"Error reading moment1st.dat at",i,k
			stop
		endif
	enddo
	read (12,*)
	
	do j=1,istep-1
		read(12,*,iostat=fstat)
	enddo
	
	if(aTime1(i) .eq. te(r1)) then
		do j=1,omit(p)
			read(12,*,iostat=fstat)
		enddo
		r1=r1+red(p)
		p=p+1
	endif
	
enddo
close(12)

write(6,*)"File moment1st.dat successfully read"

111	format(1(1X,F15.7))
112	format(7(1X,F15.7))
!-----------------------------------------------------------------------
open(unit=12,file="../output/moment2nd.dat",status="old")

do i=1,cnts(comS)-1
	read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no") aTime1(i)
	do k=1,nP(3)
		read(12,112,advance="no",iostat=fstat) Astat2Spc(k,1,i),Astat2Spc(k,2,i),Astat2Spc(k,3,i),Astat2Spc(k,4,i)	&
																													,Astat2Spc(k,5,i),Astat2Spc(k,6,i),Astat2Spc(k,7,i)
		if(fstat/=0) then
			write(*,*)"Error reading moment2nd.dat at",i,k
			stop
		endif
	enddo
	read (12, *)
	
	do j=1,istep-1
		read(12,*,iostat=fstat)
	enddo
		
	if(aTime1(i) .eq. te(r1)) then
		do j=1,omit(p)
			read(12,*,iostat=fstat)
		enddo
		r1=r1+red(p)
		p=p+1
	endif	
		
enddo
close(12)

write(6,*)"File moment2nd.dat successfully read"
!-----------------------------------------------------------------------
open(unit=12,file="../output/momentJoint.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		read(12,113,advance="no",iostat=fstat) AcorrSpc(k,1,i),AcorrSpc(k,2,i),AcorrSpc(k,3,i)	&
																					,AcorrSpc(k,4,i),AcorrSpc(k,5,i),AcorrSpc(k,6,i)	&
																					,AcorrSpc(k,7,i),AcorrSpc(k,8,i),AcorrSpc(k,9,i)
		if(fstat/=0) then
			write(*,*)"Error reading momentJoint.dat at",i,k
			stop
		endif
	enddo
	read (12, *)
	
	do j=1,istep-1
		read(12,*,iostat=fstat)
	enddo
	
	if(aTime1(i) .eq. te(r1)) then
		do j=1,omit(p)
			read(12,*,iostat=fstat)
		enddo
		r1=r1+red(p)
		p=p+1
	endif
	
enddo
close(12)

write(6,*)"File momentJoint.dat successfully read"
113	format(9(1X,F15.7))
!-----------------------------------------------------------------------
stat1 = sum(Astat1Spc,3)/alc
!-----------------------------------------------------------------------
MACH(:) = stat1(:,2)*Ma/sqrt(stat1(:,6))
!-----------------------------------------------------------------------
open (unit=10,file="../output/mean.dat")
write(10,*) 'Variables ="z","Rho","U","V","W","E","T","P","Mach No."'
write(10,104) 'ZONE k=',nP(3),',DATAPACKING="POINT"'

do k=1,nP(3)
	write (10,100)x3(k),stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4),stat1(k,5),stat1(k,6),stat1(k,7),MACH(k)
enddo
close(10)

write(6,*)"File mean.dat created"
100 format(9(1X,F22.18)) 
!-----------------------------------------------------------------------
!<U'V'> = <UV>-<U><V>

stat2 = sum(Astat2Spc,3)/alc
rms = sqrt(abs(stat2 - stat1**two))
!-----------------------------------------------------------------------
TKE(:) = (rms(:,2)**2 + rms(:,3)**2 + rms(:,4)**2)/2.0d0
Tur_Mach(:)= sqrt( rms(:,2)**two + rms(:,3)**two + rms(:,4)**two)*Ma/sqrt(stat1(:,6))
!-----------------------------------------------------------------------
open (unit=10,file="../output/rms.dat")

write(10,*) 'Variables ="z","Rho","U","V","W","E","T","P","TKE","TMa"'
write(10,104) 'ZONE k=',nP(3),',DATAPACKING="POINT"'

do k=1,nP(3)
	write (10,102)x3(k),rms(k,1),rms(k,2),rms(k,3),rms(k,4),rms(k,5),rms(k,6),rms(k,7),TKE(k),Tur_Mach(k)
enddo
close(10)

write(6,*)"File rms.dat created"
102 format(10(1X,F22.18)) 
!-----------------------------------------------------------------------

corr = sum(AcorrSpc,3)/alc
!In order of rhow,rhot,uv,uw,vw,vt,tw,pw,pt
covar(:,1) = corr(:,1) - stat1(:,1)*stat1(:,4)
covar(:,2) = corr(:,2) - stat1(:,1)*stat1(:,6)
covar(:,3) = corr(:,3) - stat1(:,2)*stat1(:,3)
covar(:,4) = corr(:,4) - stat1(:,2)*stat1(:,4)
covar(:,5) = corr(:,5) - stat1(:,3)*stat1(:,4)
covar(:,6) = corr(:,6) - stat1(:,3)*stat1(:,6)
covar(:,7) = corr(:,7) - stat1(:,6)*stat1(:,4)
covar(:,8) = corr(:,8) - stat1(:,7)*stat1(:,4)
covar(:,9) = corr(:,9) - stat1(:,7)*stat1(:,6)

open (unit=10,file="../output/corr.dat")
write(10,*) 'Variables ="y","z","rhow","rhot","uv","uw","vw","vt","tw","pw","pt"'
write(10,104) 'ZONE k=',nP(3),',DATAPACKING="POINT"'
do k=1,nP(3)
	write (10,105)x3(k),covar(k,1),covar(k,2),covar(k,3)*stat1(k,1),covar(k,4)*stat1(k,1),covar(k,5)*stat1(k,1),covar(k,6),covar(k,7),covar(k,8),covar(k,9)
enddo
close(10)

write(6,*)"File corr.dat created"
105 format(10(1X,F10.5))
104	format(A,I3,A)

open(unit=10,file="AoNDT_v1.7.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time ",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"
!-----------------------------------------------------------------------
deallocate(aTime1)
deallocate(Time1)
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/moment1stT.dat",status="old")
do
	read(12,*,iostat=fstat)
	if(fstat/=0) exit
	cnt=cnt+1
enddo
write(*,180)" Total no of lines in the file",cnt
close(12)

if(command_argument_count()==2) then
	read(str1,*) is
	read(str2,*) cnt
	cnt=cnt+1
endif	
	
!---------------Obtains first Column of Data----------------------------
allocate(Time1(0:cnt))
open(unit=12,file="../output/moment1stT.dat",status="old")
do i=1,cnt
	read(12,111,iostat=fstat) Time1(i)
enddo
close(12)
!150 format(f4.1,1x)
!------------------Finds out multiple intervals if present--------------
k=1
red(:)=1
!opt=1
comS=1
comE=1
Time1(is-1)=0
ts(k)=Time1(is)
cnts(k)=is
do i=is,cnt
	if(Time1(i) .lt. Time1(i-1)) then
		te(k)=Time1(i-2)
		cnte(k)=i-2
		k=k+1
		ts(k)=Time1(i)
		cnts(k)=i
	endif
enddo
te(k)=Time1(cnt-1)
cnte(k)=cnt-1

if(k > 1) then
	write(*,'(A,i6)')" File contains multiple intervals starting from line",is
	do i=1,k
	write(*,130) i,")",ts(i)," to ",te(i)," | line",cnts(i)," to ",cnte(i)
	enddo
	!130 format(i2,2(a,f15.7),2(a,i10))
	
	write(*,'(a)',advance='no')" Choose interval/type 100 to combine: "
	read(*,*) opt
	comS=opt
	comE=opt
	
	if(opt==100) then
		write(*,'(a)',advance='no')" Enter index of starting interval: "
		read(*,*) comS
		write(*,'(a)',advance='no')" Enter index of ending interval: "
		read(*,*) comE
	
!-------------Skipping duplicates Time1 strands--------------------------	
		j=comS-1
		flag=1
		fgtime=0.0d0
		
		do i=cnts(comS),cnte(comE)
		
			if(Time1(i) .lt. Time1(i-1) .and. Time1(i) .le. fgtime) red(j)=red(j)+1
			
			if(Time1(i) .lt. Time1(i-1) .and. i .gt. cnts(comS) .and. Time1(i) .gt. fgtime) then
				flag=0
				fgtime=Time1(i-1)
				j=j+1
			endif
			if(Time1(i) .gt. fgtime) flag=1
			if(flag==0) omit(j)=omit(j)+1
		enddo
		write(*,*)"Total duplicate strands ignored ",sum(omit)
	endif
	
endif
!-----------------------------------------------------------------------
alc=ceiling(dble(cnte(comE)-cnts(comS)-sum(omit)+1)/istep)
write(*,180)" Total lines averaged",alc
!180 format(a,i10)
!------------------------------------------------
!--------------------TKE-------------------------
allocate(Astat1SpcT(nP(3),8,alc))
allocate(Astat2SpcT(nP(3),8,alc))
allocate(AcorrSpcT (nP(3),12,alc))
allocate( aTime1(alc))
!-----------------------------------------------------------------------

open(unit=12,file="../output/moment1stT.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		read(12,115,advance="no",iostat=fstat)Astat1SpcT(k,1,i),Astat1SpcT(k,2,i),Astat1SpcT(k,3,i),Astat1SpcT(k,4,i)	&
																				 ,Astat1SpcT(k,5,i),Astat1SpcT(k,6,i),Astat1SpcT(k,7,i),Astat1SpcT(k,8,i)
		if(fstat/=0) then
			write(*,*)"Error reading moment1stT.dat at",i,k
			stop
		endif
	enddo
	read (12,*)
	
	do j=1,istep-1
		read(12,*,iostat=fstat)
	enddo
	
	if(aTime1(i) .eq. te(r1)) then
		do j=1,omit(p)
			read(12,*,iostat=fstat)
		enddo
		r1=r1+red(p)
		p=p+1
	endif
	
enddo
close(12)

write(6,*)"File moment1stT.dat successfully read"

115	format(8(1X,F15.7))
!-----------------------------------------------------------------------
open(unit=12,file="../output/moment2ndT.dat",status="old")

do i=1,cnts(comS)-1
	read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no") aTime1(i)
	do k=1,nP(3)
		read(12,115,advance="no",iostat=fstat) Astat2SpcT(k,1,i),Astat2SpcT(k,2,i),Astat2SpcT(k,3,i),Astat2SpcT(k,4,i)	&
																					,Astat2SpcT(k,5,i),Astat2SpcT(k,6,i),Astat2SpcT(k,7,i),Astat2SpcT(k,8,i)
		if(fstat/=0) then
			write(*,*)"Error reading moment2ndT.dat at",i,k
			stop
		endif
	enddo
	read (12, *)
	
	do j=1,istep-1
		read(12,*,iostat=fstat)
	enddo
		
	if(aTime1(i) .eq. te(r1)) then
		do j=1,omit(p)
			read(12,*,iostat=fstat)
		enddo
		r1=r1+red(p)
		p=p+1
	endif	
		
enddo
close(12)

write(6,*)"File moment2ndT.dat successfully read"
!-----------------------------------------------------------------------
open(unit=12,file="../output/momentJointT.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		read(12,116,advance="no",iostat=fstat) AcorrSpcT(k,1,i),AcorrSpcT(k,2,i),AcorrSpcT(k,3,i),AcorrSpcT(k,4,i)	&
																					,AcorrSpcT(k,5,i),AcorrSpcT(k,6,i),AcorrSpcT(k,7,i),AcorrSpcT(k,8,i)	&
																					,AcorrSpcT(k,9,i),AcorrSpcT(k,10,i),AcorrSpcT(k,11,i),AcorrSpcT(k,12,i)
		if(fstat/=0) then
			write(*,*)"Error reading momentJointT.dat at",i,k
			stop
		endif
	enddo
	read (12, *)
	
	do j=1,istep-1
		read(12,*,iostat=fstat)
	enddo
	
	if(aTime1(i) .eq. te(r1)) then
		do j=1,omit(p)
			read(12,*,iostat=fstat)
		enddo
		r1=r1+red(p)
		p=p+1
	endif
	
enddo
close(12)

write(6,*)"File momentJointT.dat successfully read"
116	format(12(1X,F15.7))
!---------------------------------TKE-----------------------------------
stat1T = sum(Astat1SpcT,3)/alc

open (unit=10,file="../output/TKE/meanT.dat")
do k=1,nP(3)
	write (10,200)x3(k),stat1T(k,1),stat1T(k,2),stat1T(k,3),stat1T(k,4),stat1T(k,5),stat1T(k,6),stat1T(k,7),stat1T(k,8)
enddo
close(10)

write(6,*)"File meanT.dat created"
200 format(9(1X,F22.18)) 
!-----------------------------------------------------------------------
!<U'V'> = <UV>-<U><V>

stat2T = sum(Astat2SpcT,3)/alc
rmsT = sqrt(stat2T - stat1T**two)

open (unit=10,file="../output/TKE/rmsT.dat")
do k=1,nP(3)
	write (10,200)x3(k),rmsT(k,1),rmsT(k,2),rmsT(k,3),rmsT(k,4),rmsT(k,5),rmsT(k,6),rmsT(k,7),rmsT(k,8)
enddo
close(10)

write(6,*)"File rmsT.dat created"
!-----------------------------------------------------------------------

corrT = sum(AcorrSpcT,3)/alc
!In order of pdudx,tauxz_u,tauxz_dudz,rho_u_u_w
covarT(:,1) = corrT(:,1) - stat1 (:,7)*stat1T(:,1)
covarT(:,2) = corrT(:,2) - stat1 (:,2)*stat1T(:,3)
covarT(:,3) = corrT(:,3) - stat1T(:,2)*stat1T(:,3)
covarT(:,4)	=	corrT(:,4) - (stat1(:,1)*stat1(:,2)**2*stat1(:,4) + stat1(:,1)*stat1(:,4)*rms(:,2)**2 + 2.0d0*stat1(:,1)*stat1(:,2)*covar(:,4))*haf
!In order of pdvdy,tauyz_v,tauyz_dvdz,rho_v_v_w
covarT(:,5) = corrT(:,5) - stat1 (:,7)*stat1T(:,6)
covarT(:,6) = corrT(:,6) - stat1 (:,3)*stat1T(:,7)
covarT(:,7) = corrT(:,7) - stat1T(:,4)*stat1T(:,7)
covarT(:,8)	=	corrT(:,8) - (stat1(:,1)*stat1(:,3)**2*stat1(:,4) + stat1(:,1)*stat1(:,4)*rms(:,3)**2 + 2.0d0*stat1(:,1)*stat1(:,3)*covar(:,5))*haf
!In order of pdwdz,tauwz_w,tauzz_dwdz,rho_w_w_w
covarT(:,	9) = corrT(:, 9) - stat1 (:,7)*stat1T(:,5)
covarT(:,10) = corrT(:,10) - stat1 (:,4)*stat1T(:,8)
covarT(:,11) = corrT(:,11) - stat1T(:,5)*stat1T(:,8)
covarT(:,12) = corrT(:,12) - (stat1(:,1)*stat1(:,4)**3 + 3.0d0*stat1(:,1)*stat1(:,4)*rms(:,4)**2)*haf

open (unit=10,file="../output/TKE/corrT.dat")
do k=1,nP(3)
	write (10,201)x3(k),-covarT(k,1),covarT(k,2),-covarT(k,3),-covarT(k,4),-covarT(k,5),covarT(k,6),-covarT(k,7),-covarT(k,8)	&
										 ,-covarT(k,9),covarT(k,10),-covarT(k,11),-covarT(k,12)
enddo
close(10)
201 format(13(1X,F15.5))
write(6,*)"File corrT.dat created"
!-------------------------------Pk---------------------------------------
do k=1,nP(3)
	if (k.eq.1) then
		Pk(1) = stat1(1,1)*covar(1,4) *((stat1(2,2) - stat1(1,2)) / (x3(2) - x3(1))) +	&
						stat1(1,1)*covar(1,5) *((stat1(2,3) - stat1(1,3)) / (x3(2) - x3(1))) +	&
						stat1(1,1)*rms(1,4)**2*((stat1(2,4) - stat1(1,4)) / (x3(2) - x3(1)))
		elseif (k.eq.nP(3)) then
			Pk(nP(3)) = stat1(nP(3),1)*covar(nP(3),4) *((stat1(nP(3),2) - stat1(nP(3)-1,2)) / (x3(nP(3)) - x3(nP(3)-1))) +	&
									stat1(nP(3),1)*covar(nP(3),5) *((stat1(nP(3),3) - stat1(nP(3)-1,3)) / (x3(nP(3)) - x3(nP(3)-1))) +	&
									stat1(nP(3),1)*rms(nP(3),4)**2*((stat1(nP(3),4) - stat1(nP(3)-1,4)) / (x3(nP(3)) - x3(nP(3)-1)))
		else
			Pk(k) = stat1(k,1)*covar(k,4) *((stat1(k+1,2) - stat1(k-1,2)) / (x3(k+1) - x3(k-1)))  +	&
							stat1(k,1)*covar(k,5) *((stat1(k+1,3) - stat1(k-1,3)) / (x3(k+1) - x3(k-1)))	+	&
							stat1(k,1)*rms(k,4)**2*((stat1(k+1,4) - stat1(k-1,4)) / (x3(k+1) - x3(k-1)))
	endif
enddo
!-------------------------------Pik--------------------------------------

Pik(:)	= covarT(:,1) + covarT(:,5) + covarT(:,9)
!-------------------------------Tk---------------------------------------
do k=1,nP(3)
	if (k.eq.nP(3)) then
		Tk(k) = (		((covarT(	 k,4) + covarT(	 k,8) + covarT(	 k,12)) + covar(	k,8))	&
							- ((covarT(k-1,4) + covarT(k-1,8) + covarT(k-1,12)) + covar(k-1,8))	&
						)/(x3(k) - x3(k-1))
	else
		Tk(k) = (		((covarT(k+1 ,4) + covarT(k+1,8) + covarT(k+1,12)) + covar(k+1,8))	&
							- ((covarT(k	 ,4) + covarT(k	 ,8) + covarT(k	 ,12)) + covar(k	 ,8))	&
						)/(x3(k+1) - x3(k))
	endif
enddo
!-------------------------------Dk---------------------------------------
do k=1,nP(3)
	if (k.eq.nP(3)) then
		Dk(k) = (		(covarT(k	 ,2) + covarT(k	 ,6) + covarT(k	 ,10))	&
							- (covarT(k-1,2) + covarT(k-1,6) + covarT(k-1,10))	&
						)/(x3(k) - x3(k-1))
	else
		Dk(k) = (		(covarT(k+1,2) + covarT(k+1,6) + covarT(k+1,10))	&
							- (covarT(k	 ,2) + covarT(k	 ,6) + covarT(k	 ,10))	&
						)/(x3(k+1) - x3(k))
	endif
enddo
!------------------------------Epk---------------------------------------
Epk(:)	= covarT(:,3) + covarT(:,7) + covarT(:,11)

!-----------------------------------------------------------------------
open (unit=10,file="../output/TKE/TKE.dat")
write(10,*) 'Variables ="y","Pk","Pik","Tk","Dk","Epk"'
write(10,104) 'ZONE k=',nP(3),',DATAPACKING="POINT"'

do k=1,nP(3)
	write (10,202)x3(k),-Pk(k),-Pik(k),-Tk(k),Dk(k),-Epk(k)
enddo
close(10)

write(6,*)"File TKE.dat created"
202 format(6(1X,F15.7))
!-----------------------------------------------------------------------

open(unit=10,file="AoNDT_v1.7_TKE_ALL.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time ",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"
end program