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

DOUBLE PRECISION,allocatable,dimension(:,:)	::Astat1Spc,Astat2Spc
DOUBLE PRECISION,dimension(nP(2)) 			  	:: stat1,stat2,rms

!-----------------------------------------------------------------------
call get_command_argument(1,str1)
call get_command_argument(2,str2)
if(command_argument_count()==1) read(str1,*) is

omit(:)=0
!--------------Counts no of lines in a file-----------------------------
cnt=0
open(unit=12,file="../output/tauWall_Lxt1.dat",status="old")
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
open(unit=12,file="../output/tauWall_Lxt1.dat",status="old")
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
allocate(Astat1Spc(nP(2),alc))
allocate(Astat2Spc(nP(2),alc))
allocate( aTime1(alc))

!-----------------------------------------------------------------------
call setupMesh()
!-----------------------------------------------------------------------
open(unit=12,file="../output/tauWall_Lxt1.dat",status="old")

do i=1,cnts(comS)-1
		read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no")aTime1(i)
	do k=1,nP(2)
		read(12,112,advance="no",iostat=fstat)Astat1Spc(k,i)
		if(fstat/=0) then
			write(*,*)"Error reading tauWall_Lxt1.dat at",i,k
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

write(6,*)"File tauWall_Lxt1.dat successfully read"

111	format(1(1X,F15.7))
112	format((1X,F15.7))
!-----------------------------------------------------------------------
open(unit=12,file="../output/tauWall_Lxt2.dat",status="old")

do i=1,cnts(comS)-1
	read(12,*)
enddo

p=comS
r1=comS
do i=1,alc

	read(12,111,advance="no") aTime1(i)
	do k=1,nP(2)
		read(12,112,advance="no",iostat=fstat) Astat2Spc(k,i)
		if(fstat/=0) then
			write(*,*)"Error reading tauwall_Lxt2.dat at",i,k
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

write(6,*)"File tauwall_Lxt2.dat successfully read"
!-----------------------------------------------------------------------
stat1 = sum(Astat1Spc,2)/alc

open (unit=10,file="../output/tauLxt_mean.dat")
do k=1,nP(2)
	write (10,100)x2(k)/x2(nP(2)),stat1(k)/.0037
enddo
close(10)

write(6,*)"File tauLxt_mean.dat created"
100 format(8(1X,F22.18)) 
!-----------------------------------------------------------------------
!<U'V'> = <UV>-<U><V>

stat2 = sum(Astat2Spc,2)/alc
rms = sqrt(abs(stat2 - stat1**two))

open (unit=10,file="../output/tauLxt_rms.dat")
do k=1,nP(2)
	write (10,100)x2(k)/x2(nP(2)),rms(k)
enddo
close(10)

write(6,*)"File tauLxt_rms.dat created"
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
open(unit=10,file="AoNDT_tauwall_Lxt.dat")
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time",ts(comS)," to ",te(comE)
write(10,'(A)')"-----------------------------------------------------------"
close(10)
!-----------------------------------------------------------------------
write(*,*)"Program executed successfully"

end program