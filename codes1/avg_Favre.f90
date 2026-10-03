program average
!Reads Time1 history of spatially avg data and computes temporal avg
!Developer: Parvez Ahmad
!=======================================================================
use mod_comdata
IMPLICIT NONE

character(len=20)	::str1,str2
integer					::cnt,fstat,is=1,ie,istep=1,alc,cnts(100),cnte(100)
integer					::comS,comE,flag,omit(100),red(100),opt,r1
double precision::ts(100),te(100),d1,d2,d3,d4,fgtime
double precision,allocatable,dimension(:)::aTime1,Time1

DOUBLE PRECISION,allocatable,dimension(:,:,:)	::Astat1Spc,Astat2Spc,AcorrSpc
DOUBLE PRECISION,dimension(nP(3),7) 					:: stat1,stat2,rms
DOUBLE PRECISION,dimension(nP(3),9) 					:: corr,covar

DOUBLE PRECISION,allocatable,dimension(:,:,:)	::Astat1Spc_J
DOUBLE PRECISION,dimension(nP(2),7) 					:: stat1_J

!-----------------------------------------------------------------------
call get_command_argument(1,str1)
call get_command_argument(2,str2)
if(command_argument_count()==1) read(str1,*) is

omit(:)=0
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/Favre1st.dat",status="old")
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
open(unit=12,file="../output/Favre1st.dat",status="old")
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

allocate(Astat1Spc_J(nP(2),7,alc))
!-----------------------------------------------------------------------
call setupMesh()
!-----------------------------------------------------------------------
open(unit=12,file="../output/Favre1st.dat",status="old")

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
			write(*,*)"Error reading Favre1st.dat at",i,k
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

write(6,*)"File Favre1st.dat successfully read"

111	format(1(1X,F15.7))
112	format(7(1X,F15.7))
!-----------------------------------------------------------------------
open(unit=12,file="../output/Favre2nd.dat",status="old")

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
			write(*,*)"Error reading Favre2nd.dat at",i,k
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

write(6,*)"File Favre2nd.dat successfully read"
!-----------------------------------------------------------------------
open(unit=12,file="../output/FavreJoint.dat",status="old")

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
			write(*,*)"Error reading FavreJoint.dat at",i,k
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

write(6,*)"File FavreJoint.dat successfully read"
113	format(9(1X,F15.7))
!-----------------------------------------------------------------------
stat1 = sum(Astat1Spc,3)/alc

open (unit=10,file="../output/Favre_mean.dat")
do k=1,nP(3)
	write (10,100)x3(k),stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4),stat1(k,5),stat1(k,6),stat1(k,7)
enddo
close(10)

write(6,*)"File Favre_mean.dat created"
100 format(8(1X,F22.18)) 
!-----------------------------------------------------------------------
!<U'V'> = <UV>-<U><V>

stat2 = sum(Astat2Spc,3)/alc
rms = sqrt(stat2 - stat1**two)

open (unit=10,file="../output/Favre_rms.dat")
do k=1,nP(3)
	write (10,100)x3(k),rms(k,1),rms(k,2),rms(k,3),rms(k,4),rms(k,5),rms(k,6),rms(k,7)
enddo
close(10)

write(6,*)"File Favre_rms.dat created"
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

open (unit=10,file="../output/Favre_corr.dat")
do k=1,nP(3)
	write (10,101)x3(k),covar(k,1),covar(k,2),covar(k,3),covar(k,4),covar(k,5),covar(k,6),covar(k,7),covar(k,8),covar(k,9)
enddo
close(10)

write(6,*)"File Favre_corr.dat created"
101 format(10(1X,F10.5))

!-----------------------------------------------------------------------
open(unit=10,file="AoNDT_Favre.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time1",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"

end program