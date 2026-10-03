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
double precision,allocatable,dimension(:)::aTime1,Time1,cf1,cf2

DOUBLE PRECISION,allocatable,dimension(:,:,:)	::Astat1Spc,Astat2Spc,AcorrSpc
DOUBLE PRECISION,dimension(nP(3),7) 					:: stat1,stat2,rms,stat1_
DOUBLE PRECISION,dimension(nP(3),2) 					:: corr,covar

!-----------------------------------------------------------------------
call get_command_argument(1,str1)
call get_command_argument(2,str2)
if(command_argument_count()==1) read(str1,*) is

omit(:)=0
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/Cf1st.dat",status="old")
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
open(unit=12,file="../output/Cf1st.dat",status="old")
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
allocate(AcorrSpc (nP(3),2,alc))
allocate( aTime1(alc))
allocate(cf1(nP(3)))
allocate(cf2(nP(3)))

!-----------------------------------------------------------------------
call setupMesh()
!-----------------------------------------------------------------------
open(unit=12,file="../output/Cf1st.dat",status="old")

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
			write(*,*)"Error reading Cf1st.dat at",i,k
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

write(6,*)"File Cf1st.dat successfully read"

111	format(1(1X,F15.7))
112	format(7(1X,F15.7))
!-----------------------------------------------------------------------
open(unit=12,file="../output/Cf2nd.dat",status="old")

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
			write(*,*)"Error reading Cf2nd.dat at",i,k
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

write(6,*)"File Cf2nd.dat successfully read"
!-----------------------------------------------------------------------
open(unit=12,file="../output/CfJoint.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		read(12,113,advance="no",iostat=fstat) AcorrSpc(k,1,i),AcorrSpc(k,2,i)
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

write(6,*)"File CfJoint.dat successfully read"
113	format(2(1X,F15.7))
!-----------------------------------------------------------------------
stat1 = sum(Astat1Spc,3)/alc

!<U'V'> = <UV>-<U><V>

stat2 = sum(Astat2Spc,3)/alc
rms = sqrt(abs(stat2 - stat1**two))

corr = sum(AcorrSpc,3)/alc

!In order of rho, U, V ,W, OMEGA_X, OMEGA_Y, OMEGA_Z

!w*OMEGA_Y, -v*OMEGA_Z

covar(:,1) = corr(:,1) - stat1(:,4)*stat1(:,6)
covar(:,2) = corr(:,2) - stat1(:,3)*stat1(:,7)
!-----------------------------------------------------------------------
open (unit=10,file="../output/mean.dat",status="old")
	read(10,*)
	read(10,*)

do k=1,nP(3)
	read (10,'(23X,7(1X,F22.18))',iostat=st)stat1_(k,1),stat1_(k,2),stat1_(k,3),stat1_(k,4)	&
																										 ,stat1_(k,5),stat1_(k,6),stat1_(k,7)
	
	if(st/=0)then
		print*,"Problem in reading mean.dat at k=",k
		stop
	endif
	
enddo
write(6,*)"File mean.dat read successfully"

close(10)
!-----------------------------------------------------------------------
do k=1,80
	if (k==1) then
		cf1(k) = 2.0d0*(1-x3(k))*covar(k,1)*stat1_(k,1)
		else
			cf1(k) = 2.0d0*(1-x3(k))*covar(k,1)*stat1_(k,1)
	endif
enddo

do k=1,80
	if (k==1) then
		cf2(k) = 2.0d0*(1-x3(k))*covar(k,2)*stat1_(k,1)
		else
			cf2(k) = 2.0d0*(1-x3(k))*covar(k,2)*stat1_(k,1)
	endif
enddo

open (unit=10,file="../output/Cf.dat")
do k=1,nP(3)/2
	write (10,100)x3(k),cf1(k),cf2(k)
enddo
close(10)

write(6,*)"File Cf.dat created"
100 format(3(1X,F22.15)) 

open (unit=10,file="../output/Cf_mean.dat")
do k=1,nP(3)/2
	write (10,101)x3(k),stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4),stat1(k,5),stat1(k,6),stat1(k,7)
enddo
close(10)

write(6,*)"File Cf_mean.dat created"
!-----------------------------------------------------------------------
 open (unit=10,file="../output/Cf_rms.dat")
 do k=1,nP(3)
 	write (10,101)x3(k),rms(k,1),rms(k,2),rms(k,3),rms(k,4),rms(k,5),rms(k,6),rms(k,7)
 enddo
 close(10)
101 format(8(1X,F22.15)) 

 write(6,*)"File Cf_rms.dat created"
 !-----------------------------------------------------------------------

open (unit=10,file="../output/Cf_corr.dat")
do k=1,nP(3)/2
	write (10,100)x3(k),covar(k,1),-covar(k,2)
enddo
close(10)

write(6,*)"File Cf.dat created"
!100 format(3(1X,F22.15)) 
!-----------------------------------------------------------------------
open(unit=10,file="AoNDT_Cf.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time1",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"

end program