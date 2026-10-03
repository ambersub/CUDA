module mod_control 

use mod_my_mpi
use mod_conv
use mod_diff
use mod_stats
use mod_bc
use mod_in_out

implicit none

contains

subroutine OC()

implicit none

		! Boundary Conditions
		do j=bn(2),en(2)
			do i=bn(1),en(1)

				if(bottom) then
!					var(i,j,1,3) 	 				= -(solV(i,j,y_d,3,1))/var(i,j,1,1)   			   ! Opposition control at Z+ = 10
					var(i,j,1,4) 	 				= -(solV(i,j,y_d,4,1))/var(i,j,1,1)   			   ! Opposition control at Z+ = 10

!					solV(i,j,1,3,1)				= -(solV(i,j,y_d,3,1))
					solV(i,j,1,4,1)				= -(solV(i,j,y_d,4,1))

				endif

				if(top) then
!					var(i,j,nP(3),3)			= -(solV(i,j,nP(3)-y_d+1,3,1))/var(i,j,nP(3),1)  	   ! Opposition control at Z+ = 10
					var(i,j,nP(3),4)			= -(solV(i,j,nP(3)-y_d+1,4,1))/var(i,j,nP(3),1)  	   ! Opposition control at Z+ = 10

!					solV(i,j,nP(3),3,1) 	= -(solV(i,j,nP(3)-y_d+1,3,1))
					solV(i,j,nP(3),4,1) 	= -(solV(i,j,nP(3)-y_d+1,4,1))

				endif

			enddo
		enddo

		
		call transfer_xyline()					!located in mod_my_mpi
end subroutine OC


end module mod_control
