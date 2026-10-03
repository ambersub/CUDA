program avgTauWall

implicit none

integer::st,k,cnt,max
double precision,allocatable,dimension(:):: time,tauxzL,tauxzU
double precision :: sum
!-----------------------------------------------------------------------
open(unit=10,file="../output/tauWallAvg_.dat",status="old")
cnt=0
do
	read(10,*,iostat=st)
	if(st/=0) exit
	cnt=cnt+1
enddo
cnt=cnt-1
close(10)

write(6,*) "No of lines read :",cnt
!-----------------------------------------------------------------------
allocate(time(cnt))
allocate(tauxzL(cnt))
allocate(tauxzU(cnt))
!-----------------------------------------------------------------------
!open(unit=10,file="../output/tauWall.dat",status="old")
!do k=1,cnt
!	read(10,'(3(1X,F15.7))',iostat=st) time(k), tauxzL(k), tauxzU(k)
!enddo
!if(st/=0) stop
!close(10)

sum =0.0d0

!write(6,*) "File tauWall.dat successfully read"
!-----------------------------------------------------------------------
open(unit=10,file="../output/tauWallAvg_.dat",status="unknown")
do k=1,cnt
	read(10,'(3(1X,F15.7))') time(k), tauxzL(k), tauxzU(k)!, (tauxzL(k)-tauxzU(k))/2.0
	sum = sum + ((tauxzL(k)-tauxzU(k))/2.0)
enddo
close(10)
write(6,*) "File tauWallAvg_.dat successfully read"

!-----------------------------------------------------------------------
open(unit=10,file="../output/tauWallAvgT.dat",status="unknown")
!do k=1,cnt
write(10,'(2(A,F10.5))')"Averaged b/w nondimensional Time ",time(1)," to ",time(cnt)
	write(10,'(I10, (1X,F15.7))') cnt, sum/cnt
!enddo
close(10)

write(6,*) "File tauWallAvgT.dat created"
write(6,*) sum/cnt

end program avgTauWall
