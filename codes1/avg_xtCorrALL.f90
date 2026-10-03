program average
!Reads Time1 history of spatially avg data and computes temporal avg
!Developer: Moghees Ahmad
!=======================================================================
use mod_comdata
IMPLICIT NONE

character(len=20)	::str1,str2
integer					::cnt,fstat,is=1,ie,istep=1,alc,cnts(100),cnte(100)
integer					::comS,comE,flag,omit(100),red(100),opt,r1
double precision::ts(100),te(100),d1,d2,d3,d4,fgtime
double precision,allocatable,dimension(:)::aTime1,Time1

DOUBLE PRECISION,allocatable,dimension(:,:,:,:)	::Astat1Spc,Astat2Spc,AcorrSpc
DOUBLE PRECISION,dimension(nP(2),nP(3),7) 			::stat1,stat2,rms
DOUBLE PRECISION,dimension(nP(2),nP(3),2)				::corr,covar
DOUBLE PRECISION,dimension(nP(2),nP(3))					::Pk
!-----------------------------------------------------------------------
call get_command_argument(1,str1)
call get_command_argument(2,str2)
if(command_argument_count()==1) read(str1,*) is

omit(:)=0
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/momentxP.dat",status="old")
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
open(unit=12,file="../output/momentxP.dat",status="old")
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
allocate(aTime1(alc))
allocate(Astat1Spc(nP(2),nP(3),7,alc))
!-----------------------------------------------------------------------
call setupMesh()
!-----------------------------------------------------------------------
open(unit=12,file="../output/momentxP.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		do j=1,nP(2)
			read(12,112,advance="no",iostat=fstat)Astat1Spc(j,k,1,i),Astat1Spc(j,k,2,i),Astat1Spc(j,k,3,i),Astat1Spc(j,k,4,i)	&
																															,Astat1Spc(j,k,5,i),Astat1Spc(j,k,6,i),Astat1Spc(j,k,7,i)
			if(fstat/=0) then
				write(*,*)"Error reading momentxP.dat at",i,j,k
				stop
			endif
		enddo
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

write(6,*)"File momentxP.dat successfully read"

111	format(1(1X,F15.7))
112	format(7(1X,F15.7))
113	format(2(1X,F15.7))
!-----------------------------------------------------------------------
stat1 = sum(Astat1Spc,4)/alc

open (unit=10,file="../output/meanxt.dat")

write(10,*) 'Variables ="y","z","Rho","U","V","W","E","T","P"'
write(10,104) 'ZONE k=',nP(3),',j=',nP(2),',DATAPACKING="POINT"'

do k=1,nP(3)
	write(10,*)
		do j=1,nP(2)
			write (10,100)x2(j),x3(k),stat1(j,k,1),stat1(j,k,2),stat1(j,k,3),stat1(j,k,4),stat1(j,k,5),stat1(j,k,6),stat1(j,k,7)
		enddo
enddo

close(10)

write(6,*)"File meanxt.dat created"
100 format(9(1X,F22.18))
104	format(A,I3,A,I3,A)
!-----------------------------------------------------------------------
open(unit=10,file="AoNDTmean_xt.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time1",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"
deallocate(aTime1)
deallocate(Time1)
!#######################################################################
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/momentxP2.dat",status="old")
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
open(unit=12,file="../output/momentxP2.dat",status="old")
do i=1,cnt
	read(12,111,iostat=fstat) Time1(i)
enddo
close(12)
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
!------------------------------------------------
allocate(aTime1(alc))
allocate(Astat2Spc(nP(2),nP(3),7,alc))
!-----------------------------------------------------------------------

open(unit=12,file="../output/momentxP2.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		do j=1,nP(2)
			read(12,112,advance="no",iostat=fstat)Astat2Spc(j,k,1,i),Astat2Spc(j,k,2,i),Astat2Spc(j,k,3,i),Astat2Spc(j,k,4,i)	&
																															,Astat2Spc(j,k,5,i),Astat2Spc(j,k,6,i),Astat2Spc(j,k,7,i)
			if(fstat/=0) then
				write(*,*)"Error reading momentxP2.dat at",i,j,k
				stop
			endif
		enddo
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

write(6,*)"File momentxP2.dat successfully read"
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
stat2 = sum(Astat2Spc,4)/alc
rms = sqrt(abs(stat2 - stat1**two))

open (unit=10,file="../output/rmsxt.dat")

write(10,*) 'Variables ="y","z","Rho","U","V","W","E","T","P"'
write(10,104) 'ZONE k=',nP(3),',j=',nP(2),',DATAPACKING="POINT"'

do k=1,nP(3)
	write(10,*)
		do j=1,nP(2)
			write (10,100)x2(j),x3(k),rms(j,k,1),rms(j,k,2),rms(j,k,3),rms(j,k,4),rms(j,k,5),rms(j,k,6),rms(j,k,7)
		enddo
enddo
close(10)

write(6,*)"File rmsxt.dat created"

open(unit=10,file="AoNDTrms_xt.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time1",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"
deallocate(aTime1)
deallocate(Time1)
!#######################################################################
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/momentxPJoint.dat",status="old")
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
open(unit=12,file="../output/momentxPJoint.dat",status="old")
do i=1,cnt
	read(12,111,iostat=fstat) Time1(i)
enddo
close(12)
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
!------------------------------------------------
allocate(aTime1(alc))
allocate( AcorrSpc(nP(2),nP(3),2,alc))
!-----------------------------------------------------------------------

open(unit=12,file="../output/momentxPJoint.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(3)
		do j=1,nP(2)
			read(12,113,advance="no",iostat=fstat)AcorrSpc(j,k,1,i),AcorrSpc(j,k,2,i)
			if(fstat/=0) then
				write(*,*)"Error reading momentxPJoint.dat at",i,j,k
				stop
			endif
		enddo
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

write(6,*)"File momentxPJoint.dat successfully read"
!-----------------------------------------------------------------------
corr = sum(AcorrSpc,4)/alc
!In order of uw,tw

covar(:,:,1) = corr(:,:,4) - stat1(:,:,2)*stat1(:,:,4)
covar(:,:,2) = corr(:,:,7) - stat1(:,:,6)*stat1(:,:,4)
!-------------------------------Pk---------------------------------------
do k=1,nP(3)
	do j=1,nP(2)
		if (k.eq.1) then
			Pk(j,1) = stat1(j,1,1)*covar(j,1,1) *((stat1(j,2,2) - stat1(j,1,2)) / (x3(2) - x3(1)))
			elseif (k.eq.nP(3)) then
				Pk(j,nP(3)) = stat1(j,nP(3),1)*covar(j,nP(3),1) *((stat1(j,nP(3),2) - stat1(j,nP(3)-1,2)) / (x3(nP(3)) - x3(nP(3)-1)))
			else
				Pk(j,k) = stat1(j,k,1)*covar(j,k,1) *((stat1(j,k+1,2) - stat1(j,k-1,2)) / (x3(k+1) - x3(k-1)))
			endif
	enddo
enddo

open (unit=10,file="../output/corrxt.dat")

write(10,*) 'Variables ="y","z","vw","wt","Pk"'
write(10,104) 'ZONE k=',nP(3),',j=',nP(2),',DATAPACKING="POINT"'

do k=1,nP(3)
	write(10,*)
		do j=1,nP(2)
			write (10,101)x2(j),x3(k),covar(j,k,1),covar(j,k,2),Pk(j,k)
		enddo
enddo
close(10)

write(6,*)"File corrxt.dat created"
101 format(5(1X,F10.5))
!-----------------------------------------------------------------------

open(unit=10,file="AoNDTcorr_xt.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time1",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"

end program