module mod_bc

use mod_my_mpi

implicit none

contains
subroutine bc_update_2D()
implicit none

	mu(bn(1)-1:en(1)+1,bn(2)-1:en(2)+1,bs(3):es(3)) = var(bn(1)-1:en(1)+1,bn(2)-1:en(2)+1,bs(3):es(3),6)**0.7d0

	do j=bn(2),en(2)
		do i=bn(1),en(1)

			if(bottom) then
				dudxdz		= (	(		mu(i+1,j,1)*(		f1d2c0*var(i+1,j,1,2)	&
																				+ f1d2c1*var(i+1,j,2,2)	&
																				+ f1d2c2*var(i+1,j,3,2)	&
																			)													&
											) -																				&
											(		mu(i-1,j,1)*(		f1d2c0*var(i-1,j,1,2)	&
																				+ f1d2c1*var(i-1,j,2,2)	&
																				+ f1d2c2*var(i-1,j,3,2)	&
																			)													&
											)																					&
										) *haf*invdx
				dudzdx		= (	(		f1d2c0*mu(i,j,1)*var(i+1,j,1,2)	&
												+ f1d2c1*mu(i,j,2)*var(i+1,j,2,2)	&
												+ f1d2c2*mu(i,j,3)*var(i+1,j,3,2)	&
											) -																	&
											(		f1d2c0*mu(i,j,1)*var(i-1,j,1,2)	&
												+ f1d2c1*mu(i,j,2)*var(i-1,j,2,2)	&
												+ f1d2c2*mu(i,j,3)*var(i-1,j,3,2)	&
											)																		&
										) *haf*invdx
				drhouwdx	=	(			var(i+1,j,1,1)*var(i+1,j,1,2)*var(i+1,j,1,4)	&
												-	var(i-1,j,1,1)*var(i-1,j,1,2)*var(i-1,j,1,4)	&
										)*haf*invdx
				d2wdx2		=	(			mu(i,j,1)*(			var(i+1,j,1,4)	&
																			+		var(i-1,j,1,4)	&
																			-	2*var(i	 ,j,1,4)	&
																		)											&
										)*invdx*invdx
				dvdydz		= (	(		mu(i,j+1,1)*(		f1d2c0*var(i,j+1,1,3)	&
																				+ f1d2c1*var(i,j+1,2,3)	&
																				+ f1d2c2*var(i,j+1,3,3)	&
																			)													&
											) -																				&
											(		mu(i,j-1,1)*(		f1d2c0*var(i,j-1,1,3)	&
																				+ f1d2c1*var(i,j-1,2,3)	&
																				+ f1d2c2*var(i,j-1,3,3)	&
																			)													&
											)																					&
										) *haf*invdy
				dvdzdy		= (	(		f1d2c0*mu(i,j,1)*var(i,j+1,1,3)	&
												+ f1d2c1*mu(i,j,2)*var(i,j+1,2,3)	&
												+ f1d2c2*mu(i,j,3)*var(i,j+1,3,3)	&
											) -																	&
											(		f1d2c0*mu(i,j,1)*var(i,j-1,1,3)	&
												+ f1d2c1*mu(i,j,2)*var(i,j-1,2,3)	&
												+ f1d2c2*mu(i,j,3)*var(i,j-1,3,3)	&
											)																		&
										) *haf*invdy
				drhovwdy	=	(		var(i,j+1,1,1)*var(i,j+1,1,3)*var(i,j+1,1,4)	&
											-	var(i,j-1,1,1)*var(i,j-1,1,3)*var(i,j-1,1,4)	&
										)*haf*invdy
				d2wdy2		=	(			mu(i,j,1)*(			var(i,j+1,1,4)	&
																			+		var(i,j-1,1,4)	&
																			-	2*var(i,j  ,1,4)	&
																		)											&
										)*invdy*invdy
				mud2wdz2	= (		mu(i,j,2)*(var(i,j,3,4) - var(i,j,2,4))/(df32) 	&
											- mu(i,j,1)*(var(i,j,2,4) - var(i,j,1,4))/(df21)	&
										)/df21				
				!mud2wdz2	= mu(i,j,1)*(		f2d2c0*var(i,j,1,4)	&
				!												+ f2d2c1*var(i,j,2,4)	&
				!												+ f2d2c2*var(i,j,3,4)	&
				!												+ f2d2c3*var(i,j,4,4)	&
				!											)
				!dwdzdmudz	=	(			f1d2c0*var(i,j,1,4)	&
				!								+	f1d2c1*var(i,j,2,4)	&
				!								+	f1d2c2*var(i,j,3,4)	&
				!						)*												&
				!						(			f1d2c0*mu(i,j,1)		&
				!								+ f1d2c1*mu(i,j,2)		&
				!								+ f1d2c2*mu(i,j,3)		&
				!						)
				dwdzdmudz = 0.0d0
				drhow2dz	= 	f1d2c0*var(i,j,1,1)*var(i,j,1,4)**two	&
										+	f1d2c1*var(i,j,2,1)*var(i,j,2,4)**two	&
										+	f1d2c2*var(i,j,3,1)*var(i,j,3,4)**two
				drhowdt			=	((var(i,j,1,1)*var(i,j,1,4)) - rhow(i,j,1))*invdt

				lhs				= invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz))) &
													+ drhow2dz + drhouwdx + drhovwdy + drhowdt

				var(i,j,1,7) = -(lhs + f1d2c1*var(i,j,2,7) + f1d2c2*var(i,j,3,7) ) / f1d2c0
			endif

			if(top) then
				dudxdz		= (	(		mu(i+1,j,nP(3))*(		b1d2c0*var(i+1,j,nP(3)  ,2)	&
																						+ b1d2c1*var(i+1,j,nP(3)-1,2)	&
																						+ b1d2c2*var(i+1,j,nP(3)-2,2)	&
																					)																&
											)	-																									&
											(		mu(i-1,j,nP(3))*(		b1d2c0*var(i-1,j,nP(3)  ,2)	&
																						+ b1d2c1*var(i-1,j,nP(3)-1,2)	&
																						+ b1d2c2*var(i-1,j,nP(3)-2,2)	&
																					)																&
											)																										&
										) *haf*invdx
				dudzdx		= (	(		b1d2c0*mu(i,j,nP(3)  )*var(i+1,j,nP(3)  ,2)	&
												+ b1d2c1*mu(i,j,nP(3)-1)*var(i+1,j,nP(3)-1,2)	&
												+ b1d2c2*mu(i,j,nP(3)-2)*var(i+1,j,nP(3)-2,2)	&
											)	-																							&
											(		b1d2c0*mu(i,j,nP(3)  )*var(i-1,j,nP(3)  ,2)	&
												+ b1d2c1*mu(i,j,nP(3)-1)*var(i-1,j,nP(3)-1,2)	&
												+ b1d2c2*mu(i,j,nP(3)-2)*var(i-1,j,nP(3)-2,2)	&
											)																								&
										) *haf*invdx
				drhouwdx	=	(			var(i+1,j,nP(3),1)*var(i+1,j,nP(3),2)*var(i+1,j,nP(3),4)	&
												-	var(i-1,j,nP(3),1)*var(i-1,j,nP(3),2)*var(i-1,j,nP(3),4)	&
										)*haf*invdx
				d2wdx2		=	(		mu(i,j,nP(3))*(			var(i+1,j,nP(3),4)	&
																				+		var(i-1,j,nP(3),4)	&
																				-	2*var(i	 ,j,nP(3),4)	&
																			)													&
										)*invdx*invdx
				dvdydz		= (	(		mu(i,j+1,nP(3))*(		b1d2c0*var(i,j+1,nP(3)  ,3)	&
																						+	b1d2c1*var(i,j+1,nP(3)-1,3)	&
																						+ b1d2c2*var(i,j+1,nP(3)-2,3)	&
																					)																&
											)	-																									&
											(		mu(i,j-1,nP(3))*(		b1d2c0*var(i,j-1,nP(3)  ,3)	&
																						+ b1d2c1*var(i,j-1,nP(3)-1,3)	&
																						+ b1d2c2*var(i,j-1,nP(3)-2,3)	&
																					)																&
											)																										&
										)	*haf*invdy
				dvdzdy		= (	(		b1d2c0*mu(i,j,nP(3)  )*var(i,j+1,nP(3)  ,3)	&
												+	b1d2c1*mu(i,j,nP(3)-1)*var(i,j+1,nP(3)-1,3)	&
												+ b1d2c2*mu(i,j,nP(3)-2)*var(i,j+1,nP(3)-2,3)	&
											)	-																							&
											(		b1d2c0*mu(i,j,nP(3)  )*var(i,j-1,nP(3)  ,3)	&
												+ b1d2c1*mu(i,j,nP(3)-1)*var(i,j-1,nP(3)-1,3)	&
												+ b1d2c2*mu(i,j,nP(3)-2)*var(i,j-1,nP(3)-2,3)	&
											)																								&
										)	*haf*invdy
				drhovwdy	=	(		var(i,j+1,nP(3),1)*var(i,j+1,nP(3),3)*var(i,j+1,nP(3),4)	&
											-	var(i,j-1,nP(3),1)*var(i,j-1,nP(3),3)*var(i,j-1,nP(3),4)	&
										)*haf*invdy
				d2wdy2		=	(		mu(i,j,nP(3))*(			var(i,j+1,nP(3),4)	&
																				+		var(i,j-1,nP(3),4)	&
																				-	2*var(i,j  ,nP(3),4)	&
																			)													&
										)*invdy*invdy
				mud2wdz2	= (		mu(i,j,nP(3)-1)*(var(i,j,nP(3)-2,4) - var(i,j,nP(3)-1,4))/(db23) 	&
											- mu(i,j,nP(3)	)*(var(i,j,nP(3)-1,4) - var(i,j,nP(3)	 ,4))/(db12)	&
										)/db12				
!				mud2wdz2	= 	mu(i,j,nP(3))*(		b2d2c0*var(i,j,nP(3)  ,4)	&
!																			+	b2d2c1*var(i,j,nP(3)-1,4)	&
!																			+	b2d2c2*var(i,j,nP(3)-2,4)	&
!																			+ b2d2c3*var(i,j,nP(3)-3,4)	&
!																		)
!				dwdzdmudz	=	(		b1d2c0*var(i,j,nP(3)  ,4)	&
!											+	b1d2c1*var(i,j,nP(3)-1,4)	&
!											+	b1d2c2*var(i,j,nP(3)-2,4)	&
!										)	*														&
!										(		b1d2c0*mu(i,j,nP(3)  )		&
!											+ b1d2c1*mu(i,j,nP(3)-1)		&
!											+ b1d2c2*mu(i,j,nP(3)-2)		&
!										)

				drhow2dz	= 	b1d2c0*var(i,j,nP(3)  ,1)*var(i,j,nP(3)  ,4)**two	&
										+	b1d2c1*var(i,j,nP(3)-1,1)*var(i,j,nP(3)-1,4)**two	&
										+	b1d2c2*var(i,j,nP(3)-2,1)*var(i,j,nP(3)-2,4)**two
				drhowdt		=	((var(i,j,nP(3),1)*var(i,j,nP(3),4)) - rhow(i,j,nP(3)))*invdt

				lhs				= invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz)))	&
													+ drhow2dz + drhouwdx + drhovwdy + drhowdt

				var(i,j,nP(3),7) = -(lhs + b1d2c1*var(i,j,nP(3)-1,7) + b1d2c2*var(i,j,nP(3)-2,7) ) / b1d2c0
				!-------------------------------------------------------------
			endif

		enddo
	enddo

	kt(bs(1):es(1),bs(2):es(2),bs(3):es(3)) = var(bs(1):es(1),bs(2):es(2),bs(3):es(3),6)**0.7d0

	if (bottom) 		rhow(:,:,		 1) 	= var(:,:,		1,1)*var(:,:,		 1,4)
	if (top) 				rhow(:,:,nP(3)) 	= var(:,:,nP(3),1)*var(:,:,nP(3),4)
	!-------------------------------------------------------------------
	! Boundary Conditions
	do j=bn(2),en(2)
		do i=bn(1),en(1)

			if(bottom) then
				var(i,j,1,2) 	 				= 0.0d0
				var(i,j,1,3)		 	 		= 0.0d0
				var(i,j,1,4) 	 				= 0.0d0
				var(i,j,1,6)		 	 		= 1.0d0
				var(i,j,1,5) 	 				= var(i,j,1,6) +	ggM*haf*sum(var(i,j,1,2:4)**two)
				var(i,j,1,1)	 				= gama*Mac**two*var(i,j,1,7) / var(i,j,1,6)
				solV(i,j,1,1,1)				= var(i,j,1,1)
				solV(i,j,1,2:5,1)			= var(i,j,1,1)*var(i,j,1,2:5)
			endif

			if(top) then
				var(i,j,nP(3),2)			= 0.0d0
				var(i,j,nP(3),3)			= 0.0d0
				var(i,j,nP(3),4)			= 0.0d0
				var(i,j,nP(3),6) 			= 1.0d0
				var(i,j,nP(3),5) 			= var(i,j,nP(3),6) +	ggM*haf*sum(var(i,j,nP(3),2:4)**2)
				var(i,j,nP(3),1) 			= gama*Mac**two*var(i,j,nP(3),7) / var(i,j,nP(3),6)
				solV(i,j,nP(3),1,1) 	= var(i,j,nP(3),1)
				solV(i,j,nP(3),2:5,1) = var(i,j,nP(3),1)*var(i,j,nP(3),2:5)
			endif

		enddo
	enddo

end subroutine bc_update_2D

subroutine bc_update_3D()
	implicit none
	mu(bn(1)-1:en(1)+1,bn(2)-1:en(2)+1,bs(3):es(3)) = var(bn(1)-1:en(1)+1,bn(2)-1:en(2)+1,bs(3):es(3),6)**0.7d0

	do j=bn(2),en(2)
		do i=bn(1),en(1)
			!-------------------------Pressure BC---------------------------
			!Simplifying z-momentum equation assuming impervious wall
			if(bottom) then
				dudxdz		= (	(		mu(i+1,j,1)*(		f1d3c0*var(i+1,j,1,2)	&
																				+ f1d3c1*var(i+1,j,2,2)	&
																				+ f1d3c2*var(i+1,j,3,2)	&
																				+ f1d3c3*var(i+1,j,4,2)	&
																			)													&
											) -																				&
											(		mu(i-1,j,1)*(		f1d3c0*var(i-1,j,1,2)	&
																				+ f1d3c1*var(i-1,j,2,2)	&
																				+ f1d3c2*var(i-1,j,3,2)	&
																				+ f1d3c3*var(i-1,j,4,2)	&
																			)													&
											)																					&
										) *haf*invdx
				dudzdx		= (	(		f1d3c0*mu(i,j,1)*var(i+1,j,1,2)	&
												+ f1d3c1*mu(i,j,2)*var(i+1,j,2,2)	&
												+ f1d3c2*mu(i,j,3)*var(i+1,j,3,2)	&
												+ f1d3c3*mu(i,j,4)*var(i+1,j,4,2)	&
											) -																	&
											(		f1d3c0*mu(i,j,1)*var(i-1,j,1,2)	&
												+ f1d3c1*mu(i,j,2)*var(i-1,j,2,2)	&
												+ f1d3c2*mu(i,j,3)*var(i-1,j,3,2)	&
												+ f1d3c3*mu(i,j,4)*var(i-1,j,4,2)	&
											)																		&
										) *haf*invdx
				drhouwdx	=	(			var(i+1,j,1,1)*var(i+1,j,1,2)*var(i+1,j,1,4)	&
												-	var(i-1,j,1,1)*var(i-1,j,1,2)*var(i-1,j,1,4)	&
										)*haf*invdx
				d2wdx2		=	(			mu(i,j,1)*(			var(i+1,j,1,4)	&
																			+		var(i-1,j,1,4)	&
																			-	2*var(i	 ,j,1,4)	&
																		)											&
										)*invdx*invdx
				dvdydz		= (	(		mu(i,j+1,1)*(		f1d3c0*var(i,j+1,1,3)	&
																				+ f1d3c1*var(i,j+1,2,3)	&
																				+ f1d3c2*var(i,j+1,3,3)	&
																				+ f1d3c3*var(i,j+1,4,3)	&
																			)													&
											) -																				&
											(		mu(i,j-1,1)*(		f1d3c0*var(i,j-1,1,3)	&
																				+ f1d3c1*var(i,j-1,2,3)	&
																				+ f1d3c2*var(i,j-1,3,3)	&
																				+ f1d3c3*var(i,j-1,4,3)	&
																			)													&
											)																					&
										) *haf*invdy
				dvdzdy		= (	(		f1d3c0*mu(i,j,1)*var(i,j+1,1,3)	&
												+ f1d3c1*mu(i,j,2)*var(i,j+1,2,3)	&
												+ f1d3c2*mu(i,j,3)*var(i,j+1,3,3)	&
												+ f1d3c3*mu(i,j,4)*var(i,j+1,4,3)	&
											) -																	&
											(		f1d3c0*mu(i,j,1)*var(i,j-1,1,3)	&
												+ f1d3c1*mu(i,j,2)*var(i,j-1,2,3)	&
												+ f1d3c2*mu(i,j,3)*var(i,j-1,3,3)	&
												+ f1d3c3*mu(i,j,4)*var(i,j-1,4,3)	&
											)																		&
										) *haf*invdy
				drhovwdy	=	(		var(i,j+1,1,1)*var(i,j+1,1,3)*var(i,j+1,1,4)	&
											-	var(i,j-1,1,1)*var(i,j-1,1,3)*var(i,j-1,1,4)	&
										)*haf*invdy
				d2wdy2		=	(			mu(i,j,1)*(			var(i,j+1,1,4)	&
																			+		var(i,j-1,1,4)	&
																			-	2*var(i,j  ,1,4)	&
																		)											&
										)*invdy*invdy					
				mud2wdz2	= (		mu(i,j,2)*(var(i,j,3,4) - var(i,j,2,4))/(df32) 	&
											- mu(i,j,1)*(var(i,j,2,4) - var(i,j,1,4))/(df21)	&
										)/df21
				!mud2wdz2	= mu(i,j,1)*(		f2d3c0*var(i,j,1,4)	&
				!												+ f2d3c1*var(i,j,2,4)	&
				!												+ f2d3c2*var(i,j,3,4)	&
				!												+ f2d3c3*var(i,j,4,4)	&
				!												+ f2d3c4*var(i,j,5,4)	&
				!											)
				!dwdzdmudz	=	(			f1d3c0*var(i,j,1,4)	&
				!								+	f1d3c1*var(i,j,2,4)	&
				!								+	f1d3c2*var(i,j,3,4)	&
				!								+	f1d3c3*var(i,j,4,4)	&
				!						)*												&
				!						(			f1d3c0*mu(i,j,1)		&
				!								+ f1d3c1*mu(i,j,2)		&
				!								+ f1d3c2*mu(i,j,3)		&
				!								+ f1d3c3*mu(i,j,4)		&
				!						)
				dwdzdmudz = 0.0d0
				drhow2dz	= 	f1d3c0*var(i,j,1,1)*var(i,j,1,4)**two	&
										+	f1d3c1*var(i,j,2,1)*var(i,j,2,4)**two	&
										+	f1d3c2*var(i,j,3,1)*var(i,j,3,4)**two &
										+	f1d3c3*var(i,j,4,1)*var(i,j,4,4)**two
				drhowdt			=	((var(i,j,1,1)*var(i,j,1,4)) - rhow(i,j,1))*invdt

				lhs				= invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz))) &
													 + drhow2dz + drhouwdx + drhovwdy + drhowdt

				var(i,j,1,7) = -(lhs + f1d3c1*var(i,j,2,7) + f1d3c2*var(i,j,3,7) + f1d3c3*var(i,j,4,7)) / f1d3c0
			endif

			if(top) then
				dudxdz		= (	(		mu(i+1,j,nP(3))*(		b1d3c0*var(i+1,j,nP(3)  ,2)	&
																						+ b1d3c1*var(i+1,j,nP(3)-1,2)	&
																						+ b1d3c2*var(i+1,j,nP(3)-2,2)	&
																						+ b1d3c3*var(i+1,j,nP(3)-3,2)	&
																					)																&
											)	-																									&
											(		mu(i-1,j,nP(3))*(		b1d3c0*var(i-1,j,nP(3)  ,2)	&
																						+ b1d3c1*var(i-1,j,nP(3)-1,2)	&
																						+ b1d3c2*var(i-1,j,nP(3)-2,2)	&
																						+ b1d3c3*var(i-1,j,nP(3)-3,2)	&
																					)																&
											)																										&
										) *haf*invdx
				dudzdx		= (	(		b1d3c0*mu(i,j,nP(3)  )*var(i+1,j,nP(3)  ,2)	&
												+ b1d3c1*mu(i,j,nP(3)-1)*var(i+1,j,nP(3)-1,2)	&
												+ b1d3c2*mu(i,j,nP(3)-2)*var(i+1,j,nP(3)-2,2)	&
												+ b1d3c3*mu(i,j,nP(3)-3)*var(i+1,j,nP(3)-3,2)	&
											)	-																							&
											(		b1d3c0*mu(i,j,nP(3)  )*var(i-1,j,nP(3)  ,2)	&
												+ b1d3c1*mu(i,j,nP(3)-1)*var(i-1,j,nP(3)-1,2)	&
												+ b1d3c2*mu(i,j,nP(3)-2)*var(i-1,j,nP(3)-2,2)	&
												+ b1d3c3*mu(i,j,nP(3)-3)*var(i-1,j,nP(3)-3,2)	&
											)																								&
										) *haf*invdx
				drhouwdx	=	(			var(i+1,j,nP(3),1)*var(i+1,j,nP(3),2)*var(i+1,j,nP(3),4)	&
												-	var(i-1,j,nP(3),1)*var(i-1,j,nP(3),2)*var(i-1,j,nP(3),4)	&
										)*haf*invdx
				d2wdx2		=	(		mu(i,j,nP(3))*(			var(i+1,j,nP(3),4)	&
																				+		var(i-1,j,nP(3),4)	&
																				-	2*var(i	 ,j,nP(3),4)	&
																			)													&
										)*invdx*invdx
				dvdydz		= (	(		mu(i,j+1,nP(3))*(		b1d3c0*var(i,j+1,nP(3)  ,3)	&
																						+	b1d3c1*var(i,j+1,nP(3)-1,3)	&
																						+ b1d3c2*var(i,j+1,nP(3)-2,3)	&
																						+ b1d3c3*var(i,j+1,nP(3)-3,3)	&
																					)																&
											)	-																									&
											(		mu(i,j-1,nP(3))*(		b1d3c0*var(i,j-1,nP(3)  ,3)	&
																						+ b1d3c1*var(i,j-1,nP(3)-1,3)	&
																						+ b1d3c2*var(i,j-1,nP(3)-2,3)	&
																						+ b1d3c3*var(i,j-1,nP(3)-3,3)	&
																					)																&
											)																										&
										)	*haf*invdy
				dvdzdy		= (	(		b1d3c0*mu(i,j,nP(3)  )*var(i,j+1,nP(3)  ,3)	&
												+	b1d3c1*mu(i,j,nP(3)-1)*var(i,j+1,nP(3)-1,3)	&
												+ b1d3c2*mu(i,j,nP(3)-2)*var(i,j+1,nP(3)-2,3)	&
												+ b1d3c3*mu(i,j,nP(3)-3)*var(i,j+1,nP(3)-3,3)	&
											)	-																							&
											(		b1d3c0*mu(i,j,nP(3)  )*var(i,j-1,nP(3)  ,3)	&
												+ b1d3c1*mu(i,j,nP(3)-1)*var(i,j-1,nP(3)-1,3)	&
												+ b1d3c2*mu(i,j,nP(3)-2)*var(i,j-1,nP(3)-2,3)	&
												+ b1d3c3*mu(i,j,nP(3)-3)*var(i,j-1,nP(3)-3,3)	&
											)																								&
										)	*haf*invdy
				drhovwdy	=	(		var(i,j+1,nP(3),1)*var(i,j+1,nP(3),3)*var(i,j+1,nP(3),4)	&
											-	var(i,j-1,nP(3),1)*var(i,j-1,nP(3),3)*var(i,j-1,nP(3),4)	&
										)*haf*invdy
				d2wdy2		=	(		mu(i,j,nP(3))*(			var(i,j+1,nP(3),4)	&
																				+		var(i,j-1,nP(3),4)	&
																				-	2*var(i,j  ,nP(3),4)	&
																			)													&
										)*invdy*invdy
				mud2wdz2	= (		mu(i,j,nP(3)-1)*(var(i,j,nP(3)-2,4) - var(i,j,nP(3)-1,4))/(db23) 	&
											- mu(i,j,nP(3)	)*(var(i,j,nP(3)-1,4) - var(i,j,nP(3)	 ,4))/(db12)	&
										)/db12			
				!mud2wdz2	= 	mu(i,j,nP(3))*(		b2d3c0*var(i,j,nP(3)  ,4)	&
				!															+	b2d3c1*var(i,j,nP(3)-1,4)	&
				!															+	b2d3c2*var(i,j,nP(3)-2,4)	&
				!															+ b2d3c3*var(i,j,nP(3)-3,4)	&
				!															+ b2d3c4*var(i,j,nP(3)-4,4)	&
				!														)
				!dwdzdmudz	=	(		b1d3c0*var(i,j,nP(3)  ,4)	&
				!							+	b1d3c1*var(i,j,nP(3)-1,4)	&
				!							+	b1d3c2*var(i,j,nP(3)-2,4)	&
				!							+	b1d3c3*var(i,j,nP(3)-3,4)	&
				!						)	*														&
				!						(		b1d3c0*mu(i,j,nP(3)  )		&
				!							+ b1d3c1*mu(i,j,nP(3)-1)		&
				!							+ b1d3c2*mu(i,j,nP(3)-2)		&
				!							+ b1d3c3*mu(i,j,nP(3)-3)		&
				!						)
				dwdzdmudz = 0.0d0
				drhow2dz	= 	b1d3c0*var(i,j,nP(3)  ,1)*var(i,j,nP(3)  ,4)**two	&
										+	b1d3c1*var(i,j,nP(3)-1,1)*var(i,j,nP(3)-1,4)**two	&
										+	b1d3c2*var(i,j,nP(3)-2,1)*var(i,j,nP(3)-2,4)**two	&
										+	b1d3c3*var(i,j,nP(3)-3,1)*var(i,j,nP(3)-3,4)**two
				drhowdt		=	((var(i,j,nP(3),1)*var(i,j,nP(3),4)) - rhow(i,j,nP(3)))*invdt

				lhs				= invRe*(-dudxdz -dvdydz -d2wdx2 -d2wdy2 + two3rd*(dudzdx + dvdzdy - two*(mud2wdz2 + dwdzdmudz)))	&
													 + drhow2dz + drhouwdx + drhovwdy + drhowdt

				var(i,j,nP(3),7) = -(lhs + b1d3c1*var(i,j,nP(3)-1,7) + b1d3c2*var(i,j,nP(3)-2,7) + b1d3c3*var(i,j,nP(3)-3,7) ) / b1d3c0
				!-------------------------------------------------------------
			endif

		enddo
	enddo

	kt(bs(1):es(1),bs(2):es(2),bs(3):es(3)) = var(bs(1):es(1),bs(2):es(2),bs(3):es(3),6)**0.7d0

	if (bottom) 		rhow(:,:,		 1) 	= var(:,:,		1,1)*var(:,:,		 1,4)
	if (top) 				rhow(:,:,nP(3)) 	= var(:,:,nP(3),1)*var(:,:,nP(3),4)
	!-------------------------------------------------------------------
	! Boundary Conditions
	do j=bn(2),en(2)
		do i=bn(1),en(1)

			if(bottom) then
				var(i,j,1,2) 	 				= 0.0d0
!				var(i,j,1,3)		 	 		= 0.0d0
				var(i,j,1,3) 	 				= Amp_v*(dsin (Kappa*x1(i) - Omega*time)) ! A+ =2% w sin wave oscillation, omega=-0.5, k=1, on bottom wall
				!var(i,j,1,4) 	 				= 0.0d0
				var(i,j,1,6)		 	 		= 1.0d0
				var(i,j,1,5) 	 				= var(i,j,1,6) +	ggM*haf*sum(var(i,j,1,2:4)**two)
				var(i,j,1,1)	 				= gama*Mac**two*var(i,j,1,7) / var(i,j,1,6)
				solV(i,j,1,1,1)				= var(i,j,1,1)
!				solV(i,j,1,2:5,1)			= var(i,j,1,1)*var(i,j,1,2:5)
				solV(i,j,1,2:3,1)			= var(i,j,1,1)*var(i,j,1,2:3)
				solV(i,j,1,5,1)			= var(i,j,1,1)*var(i,j,1,5)
			endif

			if(top) then
				var(i,j,nP(3),2)			= 0.0d0
!				var(i,j,nP(3),3)			= 0.0d0
				var(i,j,nP(3),3)			= Amp_v*(dsin (Kappa*x1(i) - Omega*time))
				!var(i,j,nP(3),4)			= 0.0d0
				var(i,j,nP(3),6) 			= 1.0d0
				var(i,j,nP(3),5) 			= var(i,j,nP(3),6) +	ggM*haf*sum(var(i,j,nP(3),2:4)**2)
				var(i,j,nP(3),1) 			= gama*Mac**two*var(i,j,nP(3),7) / var(i,j,nP(3),6)
				solV(i,j,nP(3),1,1) 	= var(i,j,nP(3),1)
!				solV(i,j,nP(3),2:5,1) = var(i,j,nP(3),1)*var(i,j,nP(3),2:5)
				solV(i,j,nP(3),2:3,1) = var(i,j,nP(3),1)*var(i,j,nP(3),2:3)
				solV(i,j,nP(3),5,1) = var(i,j,nP(3),1)*var(i,j,nP(3),5)
			endif

		enddo
	enddo

	end subroutine bc_update_3D

end module mod_bc