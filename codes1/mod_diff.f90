module mod_diff

use mod_my_mpi

implicit none

contains

subroutine calc_diffusive_terms_2D()

	implicit none
		!===========Discretization of Diffusive terms in Flux Form==========
	do k=bs(3),es(3)
		do j=bn(2),en(2)
			do i=bn(1),en(1)

				Ut	=var(i,j,k,2)
				Vt	=var(i,j,k,3)
				Wt	=var(i,j,k,4)
				!-------------------------------------------------------------
				dudx	=(var(i+1,j,k,2)-var(i-1,j,k,2))*haf*invdx
				dudy	=(var(i,j+1,k,2)-var(i,j-1,k,2))*haf*invdy
				!-------------------------------------------------------------
				dvdx	=(var(i+1,j,k,3)-var(i-1,j,k,3))*haf*invdx
				dvdy	=(var(i,j+1,k,3)-var(i,j-1,k,3))*haf*invdy
				!-------------------------------------------------------------
				dwdx	=(var(i+1,j,k,4)-var(i-1,j,k,4))*haf*invdx
				dwdy	=(var(i,j+1,k,4)-var(i,j-1,k,4))*haf*invdy
				!-------------------------------------------------------------
				dTdx	=(var(i+1,j,k,6)-var(i-1,j,k,6))*haf*invdx
				dTdy	=(var(i,j+1,k,6)-var(i,j-1,k,6))*haf*invdy
				!-------------------------------------------------------------
				if(k==1) then
					dudz = (	var(i,j,1,2)*f1d2c0 + &
										var(i,j,2,2)*f1d2c1 + &
										var(i,j,3,2)*f1d2c2 	&
								)
					dvdz = (	var(i,j,1,3)*f1d2c0 + &
										var(i,j,2,3)*f1d2c1 + &
										var(i,j,3,3)*f1d2c2 	&
								)
					dwdz = (	var(i,j,1,4)*f1d2c0 + &
										var(i,j,2,4)*f1d2c1 + &
										var(i,j,3,4)*f1d2c2 	&
								)
					dTdz = (	var(i,j,1,6)*f1d2c0 + &
										var(i,j,2,6)*f1d2c1 + &
										var(i,j,3,6)*f1d2c2 	&
								)
				elseif(k==nP(3)) then
					dudz = (	var(i,j,nP(3)	 ,2)*b1d2c0 + &
										var(i,j,nP(3)-1,2)*b1d2c1 + &
										var(i,j,nP(3)-2,2)*b1d2c2 	&
								)
					dvdz = (	var(i,j,nP(3)	 ,3)*b1d2c0 + &
										var(i,j,nP(3)-1,3)*b1d2c1 + &
										var(i,j,nP(3)-2,3)*b1d2c2 	&
								)
					dwdz = (	var(i,j,nP(3)	 ,4)*b1d2c0 + &
										var(i,j,nP(3)-1,4)*b1d2c1 + &
										var(i,j,nP(3)-2,4)*b1d2c2 	&
								)
					dTdz = (	var(i,j,nP(3)	 ,6)*b1d2c0 + &
										var(i,j,nP(3)-1,6)*b1d2c1 + &
										var(i,j,nP(3)-2,6)*b1d2c2 	&
								)
				else
					dudz  =(var(i,j,k+1,2)-var(i,j,k-1,2))*haf*invdXi*invJac(k)
					dvdz	=(var(i,j,k+1,3)-var(i,j,k-1,3))*haf*invdXi*invJac(k)
					dwdz	=(var(i,j,k+1,4)-var(i,j,k-1,4))*haf*invdXi*invJac(k)
					dTdz	=(var(i,j,k+1,6)-var(i,j,k-1,6))*haf*invdXi*invJac(k)
				endif
				!-------------------------------------------------------------
				divg	=dudx + dvdy + dwdz
				muT		=mu(i,j,k)
				ktT		=kt(i,j,k)
				PressT=var(i,j,k,7)

				dissF	=ggM*invRe*(  two3rd*muT*Ut*divg - two*muT*Ut*dudx - muT*Vt*(dvdx + dudy) - muT*Wt*(dwdx + dudz) )
				dissG	=ggM*invRe*( -muT*Ut*(dvdx + dudy) + two3rd*muT*Vt*divg - two*muT*Vt*dvdy - muT*Wt*(dwdy + dvdz) )
				dissH	=ggM*invRe*( -muT*Ut*(dwdx + dudz) - muT*Vt*(dwdy + dvdz) + two3rd*muT*Wt*divg - two*muT*Wt*dwdz )

				!-------------------------------------------------------------
				fD(i,j,k,1)	=	zer

				fD(i,j,k,2)	=	PressT + two3rd*muT*invRe*divg - two*muT*invRe*dudx

				fD(i,j,k,3)	=	-muT*invRe*( dvdx + dudy )

				fD(i,j,k,4)	=	-muT*invRe*( dwdx + dudz )

				fD(i,j,k,5)	=	-gama*ktT*invRe*invPr*dTdx + ggM*Ut*PressT + dissF

				!-------------------------------------------------------------
				gD(i,j,k,1)	=	zer

				gD(i,j,k,2)	=	-muT*invRe*(dvdx + dudy)

				gD(i,j,k,3)	=	PressT + two3rd*muT*invRe*divg - two*muT*invRe*dvdy

				gD(i,j,k,4)	=	-muT*invRe*(dwdy + dvdz)

				gD(i,j,k,5)	=	-gama*ktT*invRe*invPr*dTdy + ggM*Vt*PressT + dissG

				!-------------------------------------------------------------
				hD(i,j,k,1)	=	zer

				hD(i,j,k,2)	=	-muT*invRe*(dwdx + dudz)

				hD(i,j,k,3)	=	-muT*invRe*(dwdy + dvdz)

				hD(i,j,k,4)	=	PressT + two3rd*muT*invRe*divg - two*muT*invRe*dwdz

				hD(i,j,k,5)	=	-gama*ktT*invRe*invPr*dTdz + ggM*Wt*PressT + dissH
				!-------------------------------------------------------------
			enddo
		enddo
	enddo

end subroutine calc_diffusive_terms_2D

subroutine calc_diffusive_terms_3D()
	implicit none
	
	do k=bs(3),es(3)
		do j=bn(2),en(2)
			do i=bn(1),en(1)

				Ut	=var(i,j,k,2)
				Vt	=var(i,j,k,3)
				Wt	=var(i,j,k,4)
				!-------------------------------------------------------------
				dudx	=(var(i+1,j,k,2)-var(i-1,j,k,2))*haf*invdx
				dudy	=(var(i,j+1,k,2)-var(i,j-1,k,2))*haf*invdy
				!-------------------------------------------------------------
				dvdx	=(var(i+1,j,k,3)-var(i-1,j,k,3))*haf*invdx
				dvdy	=(var(i,j+1,k,3)-var(i,j-1,k,3))*haf*invdy
				!-------------------------------------------------------------
				dwdx	=(var(i+1,j,k,4)-var(i-1,j,k,4))*haf*invdx
				dwdy	=(var(i,j+1,k,4)-var(i,j-1,k,4))*haf*invdy
				!-------------------------------------------------------------
				dTdx	=(var(i+1,j,k,6)-var(i-1,j,k,6))*haf*invdx
				dTdy	=(var(i,j+1,k,6)-var(i,j-1,k,6))*haf*invdy
				!-------------------------------------------------------------
				if(k==1) then
					dudz = (	var(i,j,1,2)*f1d3c0 + &
										var(i,j,2,2)*f1d3c1 + &
										var(i,j,3,2)*f1d3c2 +	&
										var(i,j,4,2)*f1d3c3   &
								 )
					dvdz = (	var(i,j,1,3)*f1d3c0 + &
										var(i,j,2,3)*f1d3c1 + &
										var(i,j,3,3)*f1d3c2 +	&
										var(i,j,4,3)*f1d3c3   &
								 )
					dwdz = (	var(i,j,1,4)*f1d3c0 + &
										var(i,j,2,4)*f1d3c1 + &
										var(i,j,3,4)*f1d3c2 +	&
										var(i,j,4,4)*f1d3c3   &
								 )
					dTdz = (	var(i,j,1,6)*f1d3c0 + &
										var(i,j,2,6)*f1d3c1 + &
										var(i,j,3,6)*f1d3c2 +	&
										var(i,j,4,6)*f1d3c3   &
								 )
				elseif(k==nP(3)) then
					dudz = (	var(i,j,nP(3)	 ,2)*b1d3c0 + &
										var(i,j,nP(3)-1,2)*b1d3c1 + &
										var(i,j,nP(3)-2,2)*b1d3c2 +	&
										var(i,j,nP(3)-3,2)*b1d3c3   &
								 )
					dvdz = (	var(i,j,nP(3)	 ,3)*b1d3c0 + &
										var(i,j,nP(3)-1,3)*b1d3c1 + &
										var(i,j,nP(3)-2,3)*b1d3c2 +	&
										var(i,j,nP(3)-3,3)*b1d3c3   &
								 )
					dwdz = (	var(i,j,nP(3)	 ,4)*b1d3c0 + &
										var(i,j,nP(3)-1,4)*b1d3c1 + &
										var(i,j,nP(3)-2,4)*b1d3c2 +	&
										var(i,j,nP(3)-3,4)*b1d3c3   &
								 )
					dTdz = (	var(i,j,nP(3)	 ,6)*b1d3c0 + &
										var(i,j,nP(3)-1,6)*b1d3c1 + &
										var(i,j,nP(3)-2,6)*b1d3c2 +	&
										var(i,j,nP(3)-3,6)*b1d3c3   &
								 )
				else
					dudz  =(var(i,j,k+1,2)-var(i,j,k-1,2))*haf*invdXi*invJac(k)
					dvdz	=(var(i,j,k+1,3)-var(i,j,k-1,3))*haf*invdXi*invJac(k)
					dwdz	=(var(i,j,k+1,4)-var(i,j,k-1,4))*haf*invdXi*invJac(k)
					dTdz	=(var(i,j,k+1,6)-var(i,j,k-1,6))*haf*invdXi*invJac(k)
				endif
				!-------------------------------------------------------------
				divg	=dudx + dvdy + dwdz
				muT		=mu(i,j,k)
				ktT		=kt(i,j,k)
				PressT=var(i,j,k,7)

				dissF	=ggM*invRe*(  two3rd*muT*Ut*divg - two*muT*Ut*dudx - muT*Vt*(dvdx + dudy) - muT*Wt*(dwdx + dudz) )
				dissG	=ggM*invRe*( -muT*Ut*(dvdx + dudy) + two3rd*muT*Vt*divg - two*muT*Vt*dvdy - muT*Wt*(dwdy + dvdz) )
				dissH	=ggM*invRe*( -muT*Ut*(dwdx + dudz) - muT*Vt*(dwdy + dvdz) + two3rd*muT*Wt*divg - two*muT*Wt*dwdz )

				!-------------------------------------------------------------
				fD(i,j,k,1)	=	zer

				fD(i,j,k,2)	=	PressT + two3rd*muT*invRe*divg - two*muT*invRe*dudx

				fD(i,j,k,3)	=	-muT*invRe*( dvdx + dudy )

				fD(i,j,k,4)	=	-muT*invRe*( dwdx + dudz )

				fD(i,j,k,5)	=	-gama*ktT*invRe*invPr*dTdx + ggM*Ut*PressT + dissF

				!-------------------------------------------------------------
				gD(i,j,k,1)	=	zer

				gD(i,j,k,2)	=	-muT*invRe*(dvdx + dudy)

				gD(i,j,k,3)	=	PressT + two3rd*muT*invRe*divg - two*muT*invRe*dvdy

				gD(i,j,k,4)	=	-muT*invRe*(dwdy + dvdz)

				gD(i,j,k,5)	=	-gama*ktT*invRe*invPr*dTdy + ggM*Vt*PressT + dissG

				!-------------------------------------------------------------
				hD(i,j,k,1)	=	zer

				hD(i,j,k,2)	=	-muT*invRe*(dwdx + dudz)

				hD(i,j,k,3)	=	-muT*invRe*(dwdy + dvdz)

				hD(i,j,k,4)	=	PressT + two3rd*muT*invRe*divg - two*muT*invRe*dwdz

				hD(i,j,k,5)	=	-gama*ktT*invRe*invPr*dTdz + ggM*Wt*PressT + dissH
				!-------------------------------------------------------------
			enddo
		enddo
	enddo

end subroutine calc_diffusive_terms_3D

subroutine calc_FHD_and_srcV()
implicit none

	do k=bn(3),en(3)
		do j=bn(2),en(2)
			do i=bn(1),en(1)
				F(i,j,k,:,2) = ( fD(i+1,j,k,:) - fD(i-1,j,k,:) )*haf*invdx

				G(i,j,k,:,2) = ( gD(i,j+1,k,:) - gD(i,j-1,k,:) )*haf*invdy

				H(i,j,k,:,2) = ( hD(i,j,k+1,:) - hD(i,j,k-1,:) )*haf*invdXi*invJac(k)
			enddo
		enddo
	enddo
	!---------------------------------------------------------------
	do k=bn(3),en(3)
		do j=bn(2),en(2)
			do i=bn(1),en(1)
				srcV(i,j,k,1) 	= zer
				srcV(i,j,k,2) 	= haf*(tauxzL-tauxzU)*stat1Spc(k,1)
				srcV(i,j,k,3:4) = zer
				srcV(i,j,k,5) 	= ggM*var(i,j,k,2)*haf*(tauxzL-tauxzU)*stat1Spc(k,1)
				!!srcV(i,j,k,3)   = -(Amp_ls*Pi	  * dsin(Beta*x2(j)) * dsin(Pi*(x3(k)-1)))*stat1Spc(k,1)
				!srcV(i,j,k,4)   = -(Amp_ls*Beta * dcos(Beta*x2(j)) * (1.0d0 + dcos(Pi*(x3(k)-1))))*stat1Spc(k,1)
				!srcV(i,j,k,5) 	= ggM*(var(i,j,k,2)*srcV(i,j,k,2) + var(i,j,k,3)*srcV(i,j,k,3) + var(i,j,k,4)*srcV(i,j,k,4))
			
				R(i,j,k,:,1) = srcV(i,j,k,:) - (F(i,j,k,:,1) + F(i,j,k,:,2) + G(i,j,k,:,1) + G(i,j,k,:,2) + H(i,j,k,:,1) + H(i,j,k,:,2))

				if (iter.eq.1) then
					solV(i,j,k,:,2)	= solV(i,j,k,:,1)
					solV(i,j,k,:,1)	= solV(i,j,k,:,1) + haf*dt*R(i,j,k,:,1)
				endif

				if (iter.eq.2) then
					solV(i,j,k,:,1) = solV(i,j,k,:,2) + dt*( R(i,j,k,:,1) )
				endif

			enddo
		enddo
	enddo

end subroutine calc_FHD_and_srcV

end module mod_diff