module mod_conv

use mod_my_mpi

implicit none

contains

subroutine calc_convective_terms()
implicit none

	do k=bn(3),en(3)
		do j=bn(2),en(2)
			do i=bn(1),en(1)

				Pe = Re*abs(solV(i,j,k,2,1))*dx/mu(i,j,k)

				if((Pe .gt. PexVal) .or. ( i .eq. bn(1)) .or. (i .eq. en(1))) then
					tmpR1  = ( var(i,j,k,2)+var(i+1,j,k,2) )*haf
					tmpR2  = ( var(i,j,k,2)+var(i-1,j,k,2) )*haf

					if(tmpR1 .ge. 0.0d0 .and. tmpR2 .ge. 0.0d0) then
						F(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i  ,j,k,1:5,1) + 3.0d0*solV (i+1,j,k,1:5,1) - 1.0d0*solV (i-1,j,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i-1,j,k,1:5,1) + 3.0d0*solV (i  ,j,k,1:5,1) - 1.0d0*solV (i-2,j,k,1:5,1) )		&
														)*invdx/8.0d0
					endif

					if(tmpR1 .lt. 0.0d0 .and. tmpR2 .lt. 0.0d0) then
						F(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i+1,j,k,1:5,1) + 3.0d0*solV (i  ,j,k,1:5,1) - 1.0d0*solV (i+2,j,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i  ,j,k,1:5,1) + 3.0d0*solV (i-1,j,k,1:5,1) - 1.0d0*solV (i+1,j,k,1:5,1) ) 		&
														)*invdx/8.0d0
					endif

					if(tmpR1 .lt. 0.0d0 .and. tmpR2 .ge. 0.0d0) then
						F(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i+1,j,k,1:5,1) + 3.0d0*solV (i,j,k,1:5,1) - 1.0d0*solV (i+2,j,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i-1,j,k,1:5,1) + 3.0d0*solV (i,j,k,1:5,1) - 1.0d0*solV (i-2,j,k,1:5,1) ) 		&
														)*invdx/8.0d0
					endif

					if(tmpR1 .ge. 0.0d0 .and. tmpR2 .lt. 0.0d0) then
						F(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j,k,1:5,1) + 3.0d0*solV (i+1,j,k,1:5,1) - 1.0d0*solV (i-1,j,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j,k,1:5,1) + 3.0d0*solV (i-1,j,k,1:5,1) - 1.0d0*solV (i+1,j,k,1:5,1) ) 		&
														)*invdx/8.0d0
					endif

				else   ! Fourth order Central Differencing
					F(i,j,k,1:5,1) =   (-        var(i+2,j,k,2)*solV (i+2,j,k,1:5,1)  &
															+  8.0d0*var(i+1,j,k,2)*solV (i+1,j,k,1:5,1)  &
															-  8.0d0*var(i-1,j,k,2)*solV (i-1,j,k,1:5,1)  &
															+        var(i-2,j,k,2)*solV (i-2,j,k,1:5,1)  &
														)*invdx*one12th
				endif
				!-------------------------------------------------------------
				Pe = Re*abs(solV(i,j,k,3,1))*dy/mu(i,j,k)

				if((Pe .gt. PeyVal) .or. (j .eq. bn(2)) .or. (j .eq. en(2))) then
					tmpR1  = ( var(i,j,k,3)+var(i,j+1,k,3) )*haf
					tmpR2  = ( var(i,j,k,3)+var(i,j-1,k,3) )*haf

					if(tmpR1.ge.0.0d0.and.tmpR2.ge.0.0d0) then
						G(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j  ,k,1:5,1) + 3.0d0*solV (i,j+1,k,1:5,1) - 1.0d0*solV (i,j-1,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j-1,k,1:5,1) + 3.0d0*solV (i,j  ,k,1:5,1) - 1.0d0*solV (i,j-2,k,1:5,1) ) 		&
														)*invdy/8.0d0
					endif

					if(tmpR1.lt.0.0d0.and.tmpR2.lt.0.0d0) then
						G(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j+1,k,1:5,1) + 3.0d0*solV (i,j  ,k,1:5,1) - 1.0d0*solV (i,j+2,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j  ,k,1:5,1) + 3.0d0*solV (i,j-1,k,1:5,1) - 1.0d0*solV (i,j+1,k,1:5,1) ) 		&
														)*invdy/8.0d0
					endif

					if(tmpR1.lt.0.0d0.and.tmpR2.ge.0.0d0) then
						G(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j+1,k,1:5,1) + 3.0d0*solV (i,j,k,1:5,1) - 1.0d0*solV (i,j+2,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j-1,k,1:5,1) + 3.0d0*solV (i,j,k,1:5,1) - 1.0d0*solV (i,j-2,k,1:5,1) ) 		&
														)*invdy/8.0d0
					endif

					if(tmpR1.ge.0.0d0.and.tmpR2.lt.0.0d0) then
						G(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j,k,1:5,1) + 3.0d0*solV (i,j+1,k,1:5,1) - 1.0d0*solV (i,j-1,k,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j,k,1:5,1) + 3.0d0*solV (i,j-1,k,1:5,1) - 1.0d0*solV (i,j+1,k,1:5,1) ) 		&
														)*invdy/8.0d0
					endif

				else   ! Fourth order Central Differencing
					G(i,j,k,1:5,1) = 	 (-        var(i,j+2,k,3)*solV (i,j+2,k,1:5,1)  &
															+  8.0d0*var(i,j+1,k,3)*solV (i,j+1,k,1:5,1)  &
															-  8.0d0*var(i,j-1,k,3)*solV (i,j-1,k,1:5,1)  &
															+        var(i,j-2,k,3)*solV (i,j-2,k,1:5,1)  &
														)*invdy*one12th

				endif
				!-------------------------------------------------------------
				Pe = Re*abs(solV(i,j,k,4,1))*dXi*Jac(k)/mu(i,j,k)

				if (k .lt. 3 .or. k .gt. (nP(3)-2)) then
					
					call H_central_2D()
					!call H_upwind_1D()
					!call H_mixed_4D()

				elseif((Pe .gt. PezVal) .or. (k .eq. bn(3)) .or. (k .eq. en(3))) then

					tmpR1  = ( var(i,j,k,4) + var(i,j,k+1,4) )*haf
					tmpR2  = ( var(i,j,k,4) + var(i,j,k-1,4) )*haf

					if(tmpR1.ge.0.0d0.and.tmpR2.ge.0.0d0) then
						H(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j,k,1:5,1  ) + 3.0d0*solV (i,j,k+1,1:5,1) - 1.0d0*solV (i,j,k-1,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j,k-1,1:5,1) + 3.0d0*solV (i,j,k,1:5,1  ) - 1.0d0*solV (i,j,k-2,1:5,1) )		&
														)*invdXi*invJac(k)/8.0d0
					endif

					if(tmpR1 .lt. 0.0d0 .and. tmpR2 .lt. 0.0d0) then
						H(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j,k+1,1:5,1) + 3.0d0*solV (i,j,k  ,1:5,1) - 1.0d0*solV (i,j,k+2,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j,k  ,1:5,1) + 3.0d0*solV (i,j,k-1,1:5,1) - 1.0d0*solV (i,j,k+1,1:5,1) ) 		&
														)*invdXi*invJac(k)/8.0d0
					endif

					if(tmpR1.lt.0.0d0.and.tmpR2.ge.0.0d0) then
						H(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j,k+1,1:5,1) + 3.0d0*solV (i,j,k,1:5,1) - 1.0d0*solV (i,j,k+2,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j,k-1,1:5,1) + 3.0d0*solV (i,j,k,1:5,1) - 1.0d0*solV (i,j,k-2,1:5,1) ) 		&
														)*invdXi*invJac(k)/8.0d0
					endif

					if(tmpR1.ge.0.0d0.and.tmpR2.lt.0.0d0) then
						H(i,j,k,1:5,1) =(	tmpR1*( 6.0d0*solV (i,j,k,1:5,1) + 3.0d0*solV (i,j,k+1,1:5,1) - 1.0d0*solV (i,j,k-1,1:5,1) ) -	&
															tmpR2*( 6.0d0*solV (i,j,k,1:5,1) + 3.0d0*solV (i,j,k-1,1:5,1) - 1.0d0*solV (i,j,k+1,1:5,1) ) 	&
														)*invdXi*invJac(k)/8.0d0
					endif

				else   ! Fourth order Central Differencing
					H(i,j,k,1:5,1) =	( -				var(i,j,k+2,4)*solV (i,j,k+2,1:5,1)	&
															+	8.0d0*var(i,j,k+1,4)*solV (i,j,k+1,1:5,1)	&
															-	8.0d0*var(i,j,k-1,4)*solV (i,j,k-1,1:5,1)	&
															+				var(i,j,k-2,4)*solV (i,j,k-2,1:5,1)	&
														)*invdXi*invJac(k)*one12th
				endif

			enddo
		enddo
	enddo

end subroutine calc_convective_terms

subroutine H_central_2D()
	implicit none
	H(i,j,k,1:5,1) = ( var(i,j,k+1,4)*solV (i,j,k+1,1:5,1)-var(i,j,k-1,4)*solV (i,j,k-1,1:5,1) )*invdXi*invJac(k)*haf
end subroutine H_central_2D

subroutine H_upwind_1D()
	implicit none

	tmpR1  = ( var(i,j,k,4) + var(i,j,k+1,4) )*haf
	tmpR2  = ( var(i,j,k,4) + var(i,j,k-1,4) )*haf

	if(tmpR1.ge.0.0d0.and.tmpR2.ge.0.0d0) then
		H(i,j,k,1:5,1) =(	tmpR1* solV (i,j,k  ,1:5,1) -	tmpR2* solV (i,j,k-1,1:5,1) )/dz
	endif

	if(tmpR1 .lt. 0.0d0 .and. tmpR2 .lt. 0.0d0) then
		H(i,j,k,1:5,1) =(	tmpR1* solV (i,j,k+1,1:5,1) -	tmpR2* solV (i,j,k	,1:5,1) )/dz
	endif

	if(tmpR1.lt.0.0d0.and.tmpR2.ge.0.0d0) then
		H(i,j,k,1:5,1) =(	tmpR1* solV (i,j,k+1,1:5,1) - tmpR2* solV (i,j,k-1,1:5,1) )/dz
	endif

	if(tmpR1.ge.0.0d0.and.tmpR2.lt.0.0d0) then
		H(i,j,k,1:5,1) =(	tmpR1* solV (i,j,k	,1:5,1) -	tmpR2* solV (i,j,k	,1:5,1) )/dz
	endif

end subroutine H_upwind_1D

subroutine H_mixed_4D()
	implicit none

	if (k .eq. 2) then
		H(i,j,k,1:5,1) =	( 	1.0d0*var(i,j,k+3,4)*solV (i,j,k+3,1:5,1)	&
												-	6.0d0*var(i,j,k+2,4)*solV (i,j,k+2,1:5,1)	&
												+18.0d0*var(i,j,k+1,4)*solV (i,j,k+1,1:5,1)	&
												-10.0d0*var(i,j,k  ,4)*solV (i,j,k  ,1:5,1)	&
												-	3.0d0*var(i,j,k-1,4)*solV (i,j,k-1,1:5,1)	&
											)*invdXi*invJac(k)*one12th
	else
		H(i,j,k,1:5,1) =	( -	1.0d0*var(i,j,k-3,4)*solV (i,j,k-3,1:5,1)	&
												+	6.0d0*var(i,j,k-2,4)*solV (i,j,k-2,1:5,1)	&
												-18.0d0*var(i,j,k-1,4)*solV (i,j,k-1,1:5,1)	&
												+10.0d0*var(i,j,k  ,4)*solV (i,j,k  ,1:5,1)	&
												+	3.0d0*var(i,j,k+1,4)*solV (i,j,k+1,1:5,1)	&
											)*invdXi*invJac(k)*one12th
	endif			

end subroutine H_mixed_4D

end module mod_conv