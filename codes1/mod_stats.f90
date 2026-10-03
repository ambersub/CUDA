module mod_stats

use mod_my_mpi
implicit none

double precision:: inv_cnt_RUU
integer :: cnt_RUU=0


contains		
!######################################################################
	subroutine power_OC()
	implicit none
		double precision:: powL,powU
!-------------------------------------------------------------------
		tmpR1=0.0d0
		if(bottom) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					! dwdx = mu(i,j,1) * (var(i+1,j,1,4) - var(i-1,j,1,4)) *haf*invdx !zero for impervious bottom wall

					dudz = abs(var(i,j,1,7)*var(i,j,1,4))	+ abs(var(i,j,1,1)*var(i,j,1,4)**3)*haf

					tmpR1 = tmpR1 + dudz
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,powL,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		powL= powL/dble((nP(1)-2)*(nP(2)-2))

		call mpi_bcast(powL,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		tmpR1=0.0d0
		if(top) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					! dwdx = mu(i,j,nP(3)) * (var(i+1,j,nP(3),4) - var(i-1,j,nP(3),4)) *haf*invdx !zero for impervious top wall
					dudz = abs(var(i,j,nP(3),7)*var(i,j,nP(3),4))	+ abs(var(i,j,nP(3),1)*var(i,j,nP(3),4)**3)*haf

					tmpR1 = tmpR1 + dudz
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,powU,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		powU= powU/dble(((nP(1)-2)*(nP(2)-2)))

		call mpi_bcast(powU,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		if(master) then
			open(10,file="../output/power_OC.dat",access="append")
			write(10,110)time, powL, powU, (powL + powU)*haf
			close(10)
		endif
		110	format(4(1X,F15.7))

	end subroutine power_OC
!######################################################################
	subroutine power_STWSV()
	implicit none
		double precision:: powL,powU
!-------------------------------------------------------------------
		tmpR1=0.0d0
		if(bottom) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					! dwdx = mu(i,j,1) * (var(i+1,j,1,4) - var(i-1,j,1,4)) *haf*invdx !zero for impervious bottom wall

					dudz = mu(i,j,1)*	(		f1d2c0*var(i,j,1,3)	&
															+ f1d2c1*var(i,j,2,3) &
															+ f1d2c2*var(i,j,3,3)	&
														)

					tmpR1 = tmpR1 + var(i,j,1,3)*( dudz)*invRe
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,powL,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		powL= powL/dble((nP(1)-2)*(nP(2)-2))

		call mpi_bcast(powL,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		tmpR1=0.0d0
		if(top) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					! dwdx = mu(i,j,nP(3)) * (var(i+1,j,nP(3),4) - var(i-1,j,nP(3),4)) *haf*invdx !zero for impervious top wall

					dudz = mu(i,j,nP(3)) * (		b1d2c0*var(i,j,nP(3)  ,3)	&
																		+ b1d2c1*var(i,j,nP(3)-1,3) &
																		+	b1d2c2*var(i,j,nP(3)-2,3)	&
																	)

					tmpR1 = tmpR1 + var(i,j,nP(3),3)*( dudz)*invRe
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,powU,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		powU= powU/dble(((nP(1)-2)*(nP(2)-2)))

		call mpi_bcast(powU,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		if(master) then
			open(10,file="../output/power_STWSV.dat",access="append")
			write(10,110)time, powL, powU, (abs(powL) + abs(powU))
			close(10)
		endif
		110	format(4(1X,F15.7))

	end subroutine power_STWSV
		!######################################################################
	subroutine power_uc()
	implicit none
		double precision:: powL,powU
!-------------------------------------------------------------------
		tmpR1=0.0d0
		if(bottom) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					! dwdx = mu(i,j,1) * (var(i+1,j,1,4) - var(i-1,j,1,4)) *haf*invdx !zero for impervious bottom wall

					dudz = mu(i,j,1)*	(		f1d2c0*var(i,j,1,2)	&
															+ f1d2c1*var(i,j,2,2) &
															+ f1d2c2*var(i,j,3,2)	&
														)

					tmpR1 = tmpR1 + ( dudz)*invRe
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,powL,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		powL= powL/dble((nP(1)-2)*(nP(2)-2))

		call mpi_bcast(powL,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		tmpR1=0.0d0
		if(top) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					! dwdx = mu(i,j,nP(3)) * (var(i+1,j,nP(3),4) - var(i-1,j,nP(3),4)) *haf*invdx !zero for impervious top wall

					dudz = mu(i,j,nP(3)) * (		b1d2c0*var(i,j,nP(3)  ,2)	&
																		+ b1d2c1*var(i,j,nP(3)-1,2) &
																		+	b1d2c2*var(i,j,nP(3)-2,2)	&
																	)

					tmpR1 = tmpR1 + ( dudz)*invRe
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,powU,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		powU= powU/dble(((nP(1)-2)*(nP(2)-2)))

		call mpi_bcast(powU,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		if(master) then
			open(10,file="../output/power.dat",access="append")
			write(10,110)time, powL, powU, (powL+powU)/2.0d0
			close(10)
		endif
		110	format(4(1X,F15.7))

	end subroutine power_uc

			!######################################################################

	subroutine Favre()
	implicit none
		double precision,dimension(:,:), allocatable :: Fav_stat1Spc, Fav_stat2Spc, Fav_corrSpc
		nXY=nP(1)*nP(2)

		allocate(Fav_stat1Spc(nP(3),7))
		allocate(Fav_stat2Spc(nP(3),7))
		allocate(Fav_corrSpc(nP(3),9))

		do k=1,nP(3)
			loc=0.0d0

			if(k >= bs(3) .and. k <= es(3)) then
				do j=bs(2),es(2)
					do i=bs(1),es(1)
						loc(i,j,1:7)	= var(i,j,k,1:7)
					enddo
				enddo
			endif

				call mpi_reduce(sum(loc(:,:,1))/nXY,stat1Spc(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_bcast(stat1Spc(1,1),nP(3),mpi_double_precision,0,mpi_comm_world,ierr)

			do p=1,7
				call mpi_reduce(sum( loc(:,:,p)*loc(:,:,1))/(nXY*stat1Spc(k,1)),Fav_stat1Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum((loc(:,:,p)*loc(:,:,1))**2)/(nXY*stat1Spc(k,1)**2),Fav_stat2Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,1)*loc(:,:,4))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,1)*loc(:,:,6))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,2)*loc(:,:,3))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,2)*loc(:,:,4))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,3)*loc(:,:,4))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,3)*loc(:,:,6))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,6)*loc(:,:,4))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,7)*loc(:,:,4))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,7)*loc(:,:,6))/(nXY*stat1Spc(k,1)),Fav_corrSpc(k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		enddo


		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/Favre1st.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") Fav_stat1Spc(k,1),Fav_stat1Spc(k,2),Fav_stat1Spc(k,3),Fav_stat1Spc(k,4)	&
																										,Fav_stat1Spc(k,5),Fav_stat1Spc(k,6),Fav_stat1Spc(k,7)
			enddo
			write (10, *)
			close(10)
			111	format(1(1X,F15.7))
			112	format(7(1X,F15.7))
			!-----------------------------------------------------------------
			open(unit=10,file="../output/Favre2nd.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") Fav_stat2Spc(k,1),Fav_stat2Spc(k,2),Fav_stat2Spc(k,3),Fav_stat2Spc(k,4)	&
																										,Fav_stat2Spc(k,5),Fav_stat2Spc(k,6),Fav_stat2Spc(k,7)
			enddo
			write (10, *)
			close(10)
			!-----------------------------------------------------------------
			open(unit=10,file="../output/FavreJoint.dat",access="append")
			write(10,111,advance="no")time
			do k=1,nP(3)
				write(10,113,advance="no") Fav_corrSpc(k,1),Fav_corrSpc(k,2),Fav_corrSpc(k,3)	&
																	,Fav_corrSpc(k,4),Fav_corrSpc(k,5),Fav_corrSpc(k,6)	&
																	,Fav_corrSpc(k,7),Fav_corrSpc(k,8),Fav_corrSpc(k,9)
			enddo
			write (10,*)
			close(10)
			113	format(9(1X,F15.7))
			!-----------------------------------------------------------------
		endif

		deallocate(Fav_stat1Spc)
		deallocate(Fav_stat2Spc)
		deallocate(Fav_corrSpc)

	end subroutine Favre

		!######################################################################
		subroutine vel_corr()

			implicit none

			double precision,dimension(:,:,:,:),allocatable :: RUUcorr,RUUcorr_,var_new
			double precision,dimension(:,:),allocatable 		:: Ufluc
			integer :: nXY_2, ds, ids,nXY_,bses3
			integer,dimension(:),allocatable 		 ::siz3,disp,bses,bsm
			integer,dimension(:,:),allocatable 	 ::siz_3,disp_3

			
			nXY_2		= (nP(1)-2)*(nP(2)-2)
			nXY_	 	=	(nP(1))*(nP(2))

			allocate(RUUcorr(2,3,ds,nP(3)))
			allocate(RUUcorr_(2,3,ds,nP(3)))
			allocate(Ufluc(2,3))
			allocate(var_new(nP(1),nP(2),nP(3),7))
			allocate(siz3(0:nproc-1))
			allocate(disp(0:nproc-1))
			allocate(bses(3))
			allocate( siz_3(3,0:nproc-1))
			allocate(disp_3(3,0:nproc-1))
			allocate(bsm(3))

			cnt_RUU = cnt_RUU + 1

			inv_cnt_RUU = 1.0d0/dble(cnt_RUU)

			bses3 = es(3)-bs(3)+1

			bses(:)  = es(:)-bs(:)+1
			bsm(:) 	 = bs(:)-1 
		!	disp(0) = 0

		!do k = 1, proc-1
		!	disp(k) = disp(k-1) + siz(k-1)
		!enddo

  !call mpi_gatherv(t(1,istart(my_rank)),siz(my_rank),oneplane,tnew(1,1),siz,disp,oneplane,0,mpi_comm_world,ierr)
  call mpi_gather(bses3,1,mpi_integer,siz3(0),1,mpi_integer,0,mpi_comm_world,ierr)
  call mpi_gather(bs(3)-1,1,mpi_integer,disp(0),1,mpi_integer,0,mpi_comm_world,ierr)

	!do p=1,3
  	call mpi_gather(bses(1),3,mpi_integer,siz_3(1,0),3,mpi_integer,0,mpi_comm_world,ierr)
 		call mpi_gather(bsm(1),3,mpi_integer,disp_3(1,0),3,mpi_integer,0,mpi_comm_world,ierr)
	!enddo

    !call mpi_bcast(siz_3,1,mpi_integer,0,mpi_comm_world,ierr)
		!call mpi_bcast(disp_3,1,mpi_integer,0,mpi_comm_world,ierr)

  !call mpi_allgather(var(bs(1),bs(2),bs(3),1),siz_3(1,0),xy1p7v,var_new(1,1,1,1),3,xy1p7v,0,mpi_comm_world,ierr)
  call mpi_gatherv(var(bs(1),bs(2),bs(3),1),siz_3(1,id),xy1p7v,var_new(1,1,1,1),siz_3,disp_3,xy1p7v,0,mpi_comm_world,ierr)
	print *, 'done'
  !call mpi_gatherv(RUUcorr_(1,1,1,1),siz(3),xy1p7v,var_new(1,1,1,1),siz(3),ee(3,3),xy1p7v,0,mpi_comm_world,ierr)

				!do k=bn(3),en(3)
				!	do ds=1,nP(1)/2+1
				!		do j=2,nP(2)-1
				!			do i=2,nP(1)-1
				!				ids=ds
				!				if(i+ds.ge.nP(1))	ids = ds - nP(1)+2
				!				do p = 1,3
				!					Ufluc(1,p) 				= var(i,j,k,p+1)			- stat1Spc(k,p+1)*inv_cnt_RUU
				!					Ufluc(2,p) 				= var(i+ids,j,k,p+1)	-	stat1Spc(k,p+1)*inv_cnt_RUU
				!					RUUcorr(1,p,ds,k) = RUUcorr(1,p,ds,k) 	+ Ufluc(1,p)*Ufluc(2,p)/nXY_2
				!				end do
				!			end do
				!		end do
				!	end do
				!end do

				!do k=bn(3),en(3)
				!	do ds=1,nP(2)/2+1
				!		do j=2,nP(2)-1
				!			do i=2,nP(1)-1
				!				ids=ds
				!				if(j+ds.ge.nP(2)) ids = ds - nP(2)+2
				!				do  p = 1,3
				!					Ufluc(1,p)				=	var(i,j,k,p+1)    	-	stat1Spc(k,p+1)*inv_cnt_RUU
				!					Ufluc(2,p)				=	var(i,j+ids,k,p+1)	-	stat1Spc(k,p+1)*inv_cnt_RUU
				!					RUUcorr(2,p,ds,k) = RUUcorr(2,p,ds,k)   + Ufluc(2,p)*Ufluc(2,p)/nXY_2
				!				end do
				!			end do
				!		end do
				!	end do
				!end do

			!if(mod(nstat,4)==0) then
			!do k=bn(3),en(3)
			!	do ds=1,nP(2)/2+1
			!		do p=1,4
						!call mpi_reduce(sum(RUUcorr(1,p,ds,k))/cnt_RUU,RUUcorr_(1,p,ds,k),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
						!call mpi_reduce(sum(RUUcorr(2,p,ds,k))/cnt_RUU,RUUcorr_(2,p,ds,k),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)

						!call mpi_reduce(sum(tmp(:,:,p)**2)/nXY_,stat2(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			!		enddo
			!	enddo
			!enddo



		!if(master) then
			
		!		!write(filname,'(a,i3.3,a)')'corr',my_rank+1,'.tp'
		!	open(unit=20,file="../output/vel_corr/vel_corr_x.dat",access="append")
		!	!open(unit=20,file=filname,status='unknown')
		!	 write(20,*)cnt_RUU,nP(1),nP(2),nP(3)

		!		do k=2,nP(3)-1
		!			do ds=1,nP(1)/2+1
		!				write(20,12)x3(k),RUUcorr_(1,1,ds,k),RUUcorr_(1,2,ds,k),RUUcorr_(1,3,ds,k),RUUcorr_(1,4,ds,k)
		!			end do
		!		end do

		!	open(unit=20,file="../output/vel_corr/vel_corr_y.dat",access="append")
		!	!open(unit=20,file=filname,status='unknown')
		!	 write(20,*)cnt_RUU,nP(1),nP(2),nP(3)
		!		do k=2,nP(3)-1
		!			do ds=1,nP(2)/2+1
		!				write(20,12)x3(k),RUUcorr_(2,1,ds,k),RUUcorr_(2,2,ds,k),RUUcorr_(2,3,ds,k),RUUcorr_(2,4,ds,k)
		!			end do
		!		end do
		!	close(20)
		!	endif
		!if(master) then
		!	open(unit=10,file="../output/siz3.txt")
		!	write(10,105) nP(1),nP(2),nP(3)
		!	write(10,106) nproc
		!	do i=0,nproc-1
		!		write(10,107) siz3(i),disp(i)
		!	enddo
		!	close(10)
		!	105 format(3(i4))
		!	106 format(i3)
		!	107 format(2(i4,1x))
		!endif
		if(master) then
			open(unit=10,file="../output/siz3.txt")
			write(10,105) nP(1),nP(2),nP(3)
			write(10,106) nproc
			do i=0,nproc-1
				write(10,107) siz_3(1,i),siz_3(2,i),siz_3(3,i),disp_3(1,i),disp_3(2,i),disp_3(3,i)
			enddo
			close(10)
			105 format(3(i4))
			106 format(i3)
			107 format(6(i4,1x))
		endif
		stop
			12  format(5(1x,F15.7))
			13  format(3(1X,I5))

		end subroutine vel_corr
		!######################################################################

	subroutine Cf_stats()
	implicit none
		double precision,dimension(:,:,:),allocatable :: omega,tmp
		double precision,dimension(:,:)  ,allocatable :: stat1,stat2,corr
		integer :: nXY_2,nXY_

		allocate(stat1(nP(3),7))
		allocate(stat2(nP(3),7))
		allocate(corr(nP(3),2))
		allocate(omega(nP(1),nP(2),3))
		allocate(tmp(nP(1),nP(2),7))

		nXY_2=(nP(1)-2)*(nP(2)-2)
		nXY_	 =(nP(1))*(nP(2))

		do k=1,nP(3)
			omega=0.0d0
			tmp=0.d0

			if(k >= bs(3) .and. k <= es(3)) then
				do j=bs(2),es(2)
					do i=bs(1),es(1)
						tmp(i,j,1:7)	= var(i,j,k,1:7)
					enddo
				enddo

				do j=bn(2),en(2)
					do i=bn(1),en(1)
						if (k.eq.nP(3)) then
							dudz = (var(i,j,k,2) - var(i,j,k-1,2))/(x3(k)-x3(k-1))
							dvdz = (var(i,j,k,3) - var(i,j,k-1,3))/(x3(k)-x3(k-1))
						else
							dudz = (var(i,j,k+1,2) - var(i,j,k,2))/(x3(k+1)-x3(k))
							dvdz = (var(i,j,k+1,3) - var(i,j,k,3))/(x3(k+1)-x3(k))
						endif
						dvdx = (var(i+1,j,k,3) - var(i-1,j,k,3)) *haf*invdx
						dwdx = (var(i+1,j,k,4) - var(i-1,j,k,4)) *haf*invdx

						dudy = (var(i,j+1,k,2) - var(i,j-1,k,2)) *haf*invdy
						dwdy = (var(i,j+1,k,4) - var(i,j-1,k,4)) *haf*invdy

						omega(i,j,1) =  (dwdy - dvdz)*haf
						omega(i,j,2) =  -(dwdx - dudz)*haf
						omega(i,j,3) =  (dvdx - dudy)*haf
					enddo
				enddo
			endif
			!In order of rho, U, V ,W, OMEGA_X, OMEGA_Y, OMEGA_Z

			do p=1,4
				call mpi_reduce(sum(tmp(:,:,p)   )/nXY_,stat1(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(tmp(:,:,p)**2)/nXY_,stat2(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

			do p=1,3
				call mpi_reduce(sum(omega(:,:,p)   )/nXY_2,stat1(k,p+4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(omega(:,:,p)**2)/nXY_2,stat2(k,p+4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo
			!w*OMEGA_Y, -v*OMEGA_Z

				call mpi_reduce(sum(tmp(:,:,4)*omega(:,:,2))/nXY_2,corr(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(tmp(:,:,3)*omega(:,:,3))/nXY_2,corr(k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		enddo

		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/Cf1st.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat1(k,1),stat1(k,2),stat1(k,3),stat1(k,4),stat1(k,5),stat1(k,6),stat1(k,7)
			enddo
			write (10, *)
			close(10)
			111	format(1(1X,F15.7))
			112	format(7(1X,F15.7))
			!-----------------------------------------------------------------
			open(unit=10,file="../output/Cf2nd.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat2(k,1),stat2(k,2),stat2(k,3),stat2(k,4),stat2(k,5),stat2(k,6),stat2(k,7)
			enddo
			write (10, *)
			close(10)
			!-----------------------------------------------------------------
			 open(unit=10,file="../output/CfJoint.dat",access="append")
			 write(10,111,advance="no") time
			 do k=1,nP(3)
			 	write(10,113,advance="no") corr(k,1),corr(k,2)
			 enddo
			 write (10, *)
			 close(10)
			113	format(2(1X,F15.7))
			!-----------------------------------------------------------------
		endif

		deallocate(stat1)
		deallocate(stat2)
		deallocate(corr)
		deallocate(tmp)
		deallocate(omega)	
	
	end subroutine Cf_stats

	!######################################################################

	!######################################################################

	subroutine stats_xz()
		implicit none
		
		double precision,dimension(:,:,:),allocatable :: stat1,stat2,corr
		double precision,dimension(:,:)  ,allocatable :: temp1

		allocate(stat1(nP(1),nP(3),7))
		allocate(stat2(nP(1),nP(3),7))
		allocate( corr(nP(1),nP(3),9))

		allocate(temp1(nP(2),7))
		
		do k=1,nP(3)
			do i=1,nP(1)
				temp1=0.0d0

				if(i >= bs(1) .and. i <= es(1) .and. (k >= bs(3) .and. k <= es(3))) then
					do j=bs(2),es(2)
							temp1(j,1:7)	= var(i,j,k,1:7)
					enddo
				endif

				do p=1,7
					call mpi_reduce(sum(temp1(:,p)   )/nP(2),stat1(i,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,p)**2)/nP(2),stat2(i,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				enddo

				call mpi_reduce(sum(temp1(:,1)*temp1(:,4))/nP(2),corr(i,k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,1)*temp1(:,6))/nP(2),corr(i,k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,2)*temp1(:,3))/nP(2),corr(i,k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,2)*temp1(:,4))/nP(2),corr(i,k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,3)*temp1(:,4))/nP(2),corr(i,k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,3)*temp1(:,6))/nP(2),corr(i,k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,6)*temp1(:,4))/nP(2),corr(i,k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,7)*temp1(:,4))/nP(2),corr(i,k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(temp1(:,7)*temp1(:,6))/nP(2),corr(i,k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)

			enddo
		enddo	

		if (master) then

			open(unit=10,file="../output/xz/moment1st_xz.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				do i=1,nP(1)
					write(10,112,advance="no") stat1(i,k,1),stat1(i,k,2),stat1(i,k,3),stat1(i,k,4),stat1(i,k,5),stat1(i,k,6),stat1(i,k,7)
				enddo
			enddo
			write (10, *)
			close(10)
			111	format(1(1X,F15.7))
			112	format(7(1X,F15.7))
			!-----------------------------------------------------------------
			open(unit=10,file="../output/xz/moment2nd_xz.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				do i=1,nP(1)
					write(10,112,advance="no") stat2(i,k,1),stat2(i,k,2),stat2(i,k,3),stat2(i,k,4),stat2(i,k,5),stat2(i,k,6),stat2(i,k,7)
				enddo
			enddo
			write (10, *)
			close(10)
			!-----------------------------------------------------------------
			open(unit=10,file="../output/xz/momentJoint_xz.dat",access="append")
			write(10,111,advance="no")time
			do k=1,nP(3)
				do i=1,nP(1)
					write(10,113,advance="no") corr(i,k,1),corr(i,k,2),corr(i,k,3),corr(i,k,4),corr(i,k,5),corr(i,k,6),corr(i,k,7),corr(i,k,8),corr(i,k,9)
				enddo
			enddo
			write (10,*)
			close(10)
			113	format(9(1X,F15.7))
			endif

		deallocate(stat1)
		deallocate(stat2)
		deallocate(corr)

		deallocate(temp1)

	end subroutine stats_xz

	!######################################################################
	subroutine slip()
	implicit none

		if(master) then
			open(10,file="../output/uv_slip.dat",access="append")
			write(10,110)time, stat1Spc(1,2), stat1Spc(1,3)
			close(10)
		endif
		110	format(3(1X,F15.7))

	end subroutine slip

	!######################################################################

	subroutine PowSpec()
	implicit none

	double precision,dimension(:),allocatable	:: Spectra
	allocate(Spectra(np(3)))

		do k=1,nP(3)
		tmpR2 = 0.0d0

			if ((bs(1) .eq. 1) .and. (bs(2) .eq. 1) .and. (k >= bs(3) .and. k <= es(3)))  then
				tmpR2 = var(bn(1)+5,bn(2)+5,k,2)
			endif

			call mpi_reduce((tmpR2),Spectra(k),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
	
		enddo

		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/FFT.dat",access="append")
					write(10,112) time,Spectra(16),Spectra(36),Spectra(52)
			close(10)
			
			112	format(4(1X,F15.7))
		endif

		deallocate(Spectra)

	end subroutine PowSpec

	!######################################################################

	subroutine TKE_stats_xt()
	implicit none

	double precision,dimension(:,:,:),allocatable :: stat1,stat2,corr
	double precision,dimension(:,:,:),allocatable :: stats1,stats2,corrs
	double precision,dimension(:,:)  ,allocatable :: temp1,temp2
	integer :: nX_2

	allocate(stat1(nP(2),nP(3),7))
	allocate(stat2(nP(2),nP(3),7))
	allocate( corr(nP(2),nP(3),9))

	allocate(stats1(nP(2),nP(3),8))
	allocate(stats2(nP(2),nP(3),8))
	allocate( corrs(nP(2),nP(3),12))

	allocate(temp1(nP(1),7))
	allocate(temp2(nP(1),8))

			nXY=nP(1)*nP(2)
			nXY_2=(nP(1)-2)*(nP(2)-2)
			nX_2=(nP(1)-2)

			do k=1,nP(3)
				do j=1,nP(2)

					temp1	=0.0d0
					temp2 =0.0d0
					if ( (j >= bs(2) .and. j <= es(2)) .and. (k >= bs(3) .and. k <= es(3)) ) then

							do i=bs(1),es(1)
								temp1(i,1:7)	= var(i,j,k,1:7)
							enddo
					endif

					do p=1,7
						call mpi_reduce(sum(temp1(:,p)   )/nP(1),stat1(j,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
						call mpi_reduce(sum(temp1(:,p)**2)/nP(1),stat2(j,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					enddo

					call mpi_reduce(sum(temp1(:,1)*temp1(:,4))/nP(1),corr(j,k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,1)*temp1(:,6))/nP(1),corr(j,k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,2)*temp1(:,3))/nP(1),corr(j,k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,2)*temp1(:,4))/nP(1),corr(j,k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,3)*temp1(:,4))/nP(1),corr(j,k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,3)*temp1(:,6))/nP(1),corr(j,k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,6)*temp1(:,4))/nP(1),corr(j,k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,7)*temp1(:,4))/nP(1),corr(j,k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(temp1(:,7)*temp1(:,6))/nP(1),corr(j,k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)

					!---------------------------TKE STATS-----------------------------
					if ( (j >= bs(2) .and. j <= es(2)) .and. (k >= bs(3) .and. k <= es(3)) ) then
							do i=bn(1),en(1)
								temp2(i,1)		= (var(i+1,j,k,2) - var(i-1,j,k,2))*haf*invdx									!dudx
								dwdx 					= (var(i+1,j,k,4) - var(i-1,j,k,4))*haf*invdx 								!dwdx
								if (k.eq.nP(3)) then
									temp2(i,2) = (		b1d2c0*var(i,j,k  ,2)	&																	!dudz
																	+ b1d2c1*var(i,j,k-1,2) &
																	+	b1d2c2*var(i,j,k-2,2)	&
															)
									temp2(i,4) = (		b1d2c0*var(i,j,k  ,3)	&																	!dvdz
																	+ b1d2c1*var(i,j,k-1,3) &
																	+	b1d2c2*var(i,j,k-2,3)	&
															)															
									temp2(i,5) = (		b1d2c0*var(i,j,k  ,4)	&																	!dwdz
																	+ b1d2c1*var(i,j,k-1,4) &
																	+	b1d2c2*var(i,j,k-2,4)	&
															)
									elseif (k.eq.1) then
										temp2(i,2) = (		f1d2c0*var(i,j,k	,2)	&																!dudz
																		+ f1d2c1*var(i,j,k+1,2) &
																		+ f1d2c2*var(i,j,k+2,2)	&
																)
										temp2(i,4) = (		f1d2c0*var(i,j,k	,3)	&																!dvdz
																		+ f1d2c1*var(i,j,k+1,3) &
																		+ f1d2c2*var(i,j,k+2,3)	&
																) 
										temp2(i,5) = (		f1d2c0*var(i,j,k	,4)	&																!dwdz
																		+ f1d2c1*var(i,j,k+1,4) &
																		+ f1d2c2*var(i,j,k+2,4)	&
																)
									else
										temp2(i,2) = (var(i,j,k+1,2)	- var(i,j,k-1,2))*haf*invdXi*invJac(k)		!dudz
										temp2(i,4) = (var(i,j,k+1,3)	- var(i,j,k-1,3))*haf*invdXi*invJac(k)		!dvdz
										temp2(i,5) = (var(i,j,k+1,4)	- var(i,j,k-1,4))*haf*invdXi*invJac(k)		!dwdz
								endif
								if (j.eq.1) then
									dwdy 				= (var(i,j+1,k,4) - var(i,j,k,4))*invdy 											!dwdy
									temp2(i,6)	=	(var(i,j+1,k,3) - var(i,j,k,3))*invdy												!dvdy
									elseif (j.eq.nP(2))	then
										dwdy 				= (var(i,j,k,4) - var(i,j-1,k,4))*invdy 										!dwdy
										temp2(i,6)	=	(var(i,j,k,3) - var(i,j-1,k,3))*invdy											!dvdy
									else
										dwdy 				= (var(i,j+1,k,4) - var(i,j-1,k,4))*haf*invdy 							!dwdy
										temp2(i,6)	=	(var(i,j+1,k,3) - var(i,j-1,k,3))*haf*invdy								!dvdy
								endif
									temp2(i,3)	= mu(i,j,k)*(dwdx + temp2(i,2))*invRe													!tauxz
									temp2(i,7)	=	mu(i,j,k)*(dwdy + temp2(i,4))*invRe													!tauyz
									temp2(i,8)	=	mu(i,j,k)*(two*temp2(i,5) - two3rd*(temp2(i,1) + temp2(i,6)	+ temp2(i,5)))*invRe	!tauzz
							enddo
					endif

					do p=1,8
						call mpi_reduce(sum(temp2(:,p)   )/nX_2,stats1(j,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
						call mpi_reduce(sum(temp2(:,p)**2)/nX_2,stats2(j,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					enddo

					call mpi_reduce(sum(temp1(:,7)*temp2(:,1))/nX_2,corrs(j,k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!pdudx
					call mpi_reduce(sum(temp1(:,2)*temp2(:,3))/nX_2,corrs(j,k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauxz_u
					call mpi_reduce(sum(temp2(:,3)*temp2(:,2))/nX_2,corrs(j,k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauxz_dudz
					call mpi_reduce(sum((stat1(j,k,1)*temp1(:,2)**2*temp1(:,4))*haf)/nP(1),corrs(j,k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)	!rho_u_u_w

					call mpi_reduce(sum(temp1(:,7)*temp2(:,6))/nX_2,corrs(j,k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!pdvdy
					call mpi_reduce(sum(temp1(:,3)*temp2(:,7))/nX_2,corrs(j,k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauyz_v
					call mpi_reduce(sum(temp2(:,4)*temp2(:,7))/nX_2,corrs(j,k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauyz_dvdz
					call mpi_reduce(sum((stat1(j,k,1)*temp1(:,3)**2*temp1(:,4))*haf)/nP(1),corrs(j,k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)	!rho_v_v_w

					call mpi_reduce(sum(temp1(:,7)*temp2(:,5))/nX_2,corrs(j,k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!pdwdz
					call mpi_reduce(sum(temp1(:,4)*temp2(:,8))/nX_2,corrs(j,k,10),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauzz_w
					call mpi_reduce(sum(temp2(:,5)*temp2(:,8))/nX_2,corrs(j,k,11),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauzz_dwdz
					call mpi_reduce(sum((stat1(j,k,1)*temp1(:,4)**3)*haf)/nP(1),corrs(j,k,12),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)	!rho_w_w_w
				enddo
			enddo

			if(master) then
				!-----------------------------------------------------------------
				open(unit=10,file="../output/TKE_xt_1st.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					do j=1,nP(2)
						write(10,112,advance="no") stat1(j,k,1),stat1(j,k,2),stat1(j,k,3),stat1(j,k,4),stat1(j,k,5),stat1(j,k,6),stat1(j,k,7)
					enddo
				enddo
				write (10, *)
				close(10)
				111	format(1(1X,F15.7))
				112	format(7(1X,F15.7))
				!-----------------------------------------------------------------
				open(unit=10,file="../output/TKE_xt_2nd.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					do j=1,nP(2)
						write(10,112,advance="no") stat2(j,k,1),stat2(j,k,2),stat2(j,k,3),stat2(j,k,4),stat2(j,k,5),stat2(j,k,6),stat2(j,k,7)
					enddo
				enddo
				write (10, *)
				close(10)
				!-----------------------------------------------------------------
				open(unit=10,file="../output/TKE_xt_Joint.dat",access="append")
				write(10,111,advance="no")time
				do k=1,nP(3)
					do j=1,nP(2)
						write(10,113,advance="no") corr(j,k,1),corr(j,k,2),corr(j,k,3),corr(j,k,4),corr(j,k,5),corr(j,k,6),corr(j,k,7),corr(j,k,8),corr(j,k,9)
					enddo
				enddo
				write (10,*)
				close(10)
				113	format(9(1X,F15.7))
				!-----------------------------------------------------------------
				open(unit=10,file="../output/TKE_xt_1stT.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					do j=1,nP(2)
						write(10,115,advance="no") stats1(j,k,1),stats1(j,k,2),stats1(j,k,3),stats1(j,k,4)	&
																			,stats1(j,k,5),stats1(j,k,6),stats1(j,k,7),stats1(j,k,8)
					enddo
				enddo
				write (10, *)
				close(10)
				115	format(8(1X,F15.7))
				!-----------------------------------------------------------------
				open(unit=10,file="../output/TKE_xt_2ndT.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					do j=1,nP(2)
						write(10,115,advance="no") stats2(j,k,1),stats2(j,k,2),stats2(j,k,3),stats2(j,k,4)	&
																			,stats2(j,k,5),stats2(j,k,6),stats2(j,k,7),stats2(j,k,8)
					enddo
				enddo
				write (10, *)
				close(10)
				
				!-----------------------------------------------------------------
				open(unit=10,file="../output/TKE_xt_JointT.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					do j=1,nP(2)
						write(10,216,advance="no") corrs(j,k,1),corrs(j,k,2),corrs(j,k,3),corrs(j,k,4)	&
																			,corrs(j,k,5),corrs(j,k,6),corrs(j,k,7),corrs(j,k,8)	&
																			,corrs(j,k,9),corrs(j,k,10),corrs(j,k,11),corrs(j,k,12)
					enddo
				enddo
				write (10,*)
				close(10)
				216	format(12(1X,F15.7))	
			endif

	deallocate(stat1)
	deallocate(stat2)
	deallocate(corr)

	deallocate(stats1)
	deallocate(stats2)
	deallocate(corrs)
	
	deallocate(temp1)
	deallocate(temp2)

	end subroutine TKE_stats_xt

	!######################################################################

	!######################################################################

	subroutine Omega_stats()
	implicit none

	double precision,dimension(nP(3),3) 			:: stat1,stat2
	double precision,dimension(nP(1),nP(2),3) :: omega !x,y,z
		nXY_2=(nP(1)-2)*(nP(2)-2)

		do k=1,nP(3)
			omega=0.0d0

			if(k >= bs(3) .and. k <= es(3)) then
				do j=bn(2),en(2)
					do i=bn(1),en(1)
						if (k.eq.nP(3)) then
							dudz = (var(i,j,k,2) - var(i,j,k-1,2))/(x3(k)-x3(k-1))
							dvdz = (var(i,j,k,3) - var(i,j,k-1,3))/(x3(k)-x3(k-1))
						else
							dudz = (var(i,j,k+1,2) - var(i,j,k,2))/(x3(k+1)-x3(k))
							dvdz = (var(i,j,k+1,3) - var(i,j,k,3))/(x3(k+1)-x3(k))
						endif
						dvdx = (var(i+1,j,k,3) - var(i-1,j,k,3)) *haf*invdx
						dwdx = (var(i+1,j,k,4) - var(i-1,j,k,4)) *haf*invdx

						dudy = (var(i,j+1,k,2) - var(i,j-1,k,2)) *haf*invdy
						dwdy = (var(i,j+1,k,4) - var(i,j-1,k,4)) *haf*invdy

						omega(i,j,1) = (dwdy - dvdz)
						omega(i,j,2) = (dwdx - dudz)
						omega(i,j,3) = (dvdx - dudy)
					enddo
				enddo
			endif

			do p=1,3
				call mpi_reduce(sum(omega(:,:,p)   )/nXY_2,stat1(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(omega(:,:,p)**2)/nXY_2,stat2(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

		enddo

		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/Omega1st.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat1(k,1),stat1(k,2),stat1(k,3)
			enddo
			write (10, *)
			close(10)
			111	format(1(1X,F15.7))
			112	format(3(1X,F15.7))
			!-----------------------------------------------------------------
			open(unit=10,file="../output/Omega2nd.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat2(k,1),stat2(k,2),stat2(k,3)
			enddo
			write (10, *)
			close(10)
			!-----------------------------------------------------------------
		endif
	end subroutine Omega_stats

	!######################################################################
	!######################################################################
	subroutine tauWall_Lxt()
	implicit none

	double precision,dimension(nP(2)) :: stat1,stat2
	double precision,dimension(nP(1)) :: stat

		!-------------------------------------------------------------------
			do j=1,nP(2)
				stat = 0.0d0

				if( (j >= bs(2) .and. j <= es(2)) .and. (bs(3) .eq. 1)) then
					do i=bn(1),en(1)
						dwdx = mu(i,j,1) * (var(i+1,j,1,4) - var(i-1,j,1,4)) *haf*invdx !zero for impervious bottom wall

						dudz = mu(i,j,1)*	(		f1d2c0*var(i,j,1,2)	&
																+ f1d2c1*var(i,j,2,2) &
																+ f1d2c2*var(i,j,3,2)	&
															)

						stat(i) =  (dwdx + dudz)*invRe
					enddo
				endif

				call mpi_reduce(sum(stat(:))/(nP(1)-2),stat1(j),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(stat(:)**2)/(nP(1)-2),stat2(j),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)

			enddo
		!-------------------------------------------------------------------
		if(master) then
					open(unit=10,file="../output/tauWall_Lxt1.dat",access="append")
			write(10,111,advance="no") time
				do j=1,nP(2)
					write(10,112,advance="no") stat1(j)
				enddo
			write (10, *)
			close(10)

			open(unit=10,file="../output/tauWall_Lxt2.dat",access="append")
			write(10,111,advance="no") time
				do j=1,nP(2)
					write(10,112,advance="no") stat2(j)
				enddo
			write (10, *)
			close(10)
		endif
		112	format(1X,F15.7)
		111	format(1(1X,F15.7))

	end subroutine tauWall_Lxt
	!######################################################################
	subroutine write_stats_xAvg()
	implicit none
	
	allocate(Lstat1Spc(nP(2),nP(3),7))
	allocate(Lstat2Spc(nP(2),nP(3),7))
	allocate( LcorrSpc(nP(2),nP(3),2))
	allocate(local(nP(1),7))

			do k=1,nP(3)
				do j=1,nP(2)
					local=0.0d0

					if( (j >= bs(2) .and. j <= es(2)) .and. (k >= bs(3) .and. k <= es(3)) ) then
								do i=bs(1),es(1)
									local(i,1:7)	= var(i,j,k,1:7)
								enddo
					endif
					
					do p=1,7
						call mpi_reduce(sum(local(:,p)   )/nP(1),Lstat1Spc(j,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
						call mpi_reduce(sum(local(:,p)**2)/nP(1),Lstat2Spc(j,k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					enddo

					call mpi_reduce(sum(local(:,2)*local(:,4))/nP(1),LcorrSpc(j,k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(local(:,6)*local(:,4))/nP(1),LcorrSpc(j,k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
								
				enddo
			enddo

		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/momentxP.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				do j=1,nP(2)
					write(10,112,advance="no") Lstat1Spc(j,k,1),Lstat1Spc(j,k,2),Lstat1Spc(j,k,3),Lstat1Spc(j,k,4)	&
																		,Lstat1Spc(j,k,5),Lstat1Spc(j,k,6),Lstat1Spc(j,k,7)
				enddo
			enddo
			write (10, *)
			close(10)
			
			111	format(1(1X,F15.7))
			112	format(7(1X,F15.7))
			113	format(2(1X,F15.7))

			open(unit=10,file="../output/momentxP2.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				do j=1,nP(2)
					write(10,112,advance="no") Lstat2Spc(j,k,1),Lstat2Spc(j,k,2),Lstat2Spc(j,k,3),Lstat2Spc(j,k,4)	&
																		,Lstat2Spc(j,k,5),Lstat2Spc(j,k,6),Lstat2Spc(j,k,7)
				enddo
			enddo
			write (10, *)
			close(10)

			open(unit=10,file="../output/momentxPJoint.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				do j=1,nP(2)
					write(10,113,advance="no") LcorrSpc(j,k,1),LcorrSpc(j,k,2)
				enddo
			enddo
			write (10, *)
			close(10)			
			!-----------------------------------------------------------------
		endif

		deallocate(Lstat1Spc)
		deallocate(Lstat2Spc)
		deallocate(LcorrSpc)
		deallocate(local)

	end subroutine write_stats_xAvg
	!######################################################################
	subroutine write_stats_wMax()

	implicit none

		allocate(stat1SpcJ(nP(2),7))

			do m=1,nP(2)
				loc=0.0d0

				if( m >= bs(2) .and. m <= es(2) ) then
					if( ( bs(3) .eq. 80 .or. es(3) .eq. 80 ) .or. ( bs(3) .le. 80  .and. es(3) .ge. 80 ) ) then
							do i=bs(1),es(1)
								loc(i,m,1:7)	= var(i,m,80,1:7)
							enddo
					endif
				endif
				
					do p=1,7
						call mpi_reduce(sum(loc(:,m,p))/nP(1),stat1SpcJ(m,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					enddo
			enddo
			
		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/moment80.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(2)
				write(10,112,advance="no") stat1SpcJ(k,1),stat1SpcJ(k,2),stat1SpcJ(k,3),stat1SpcJ(k,4)	&
																								 ,stat1SpcJ(k,5),stat1SpcJ(k,6),stat1SpcJ(k,7)
			enddo
			write (10, *)
			close(10)
			111	format(1(1X,F15.7))
			112	format(7(1X,F15.7))
			!-----------------------------------------------------------------
		endif

		deallocate(stat1SpcJ)
	end subroutine write_stats_wMax

	!######################################################################

	subroutine write_stats()
	implicit none
		nXY=nP(1)*nP(2)

		do k=1,nP(3)
			loc=0.0d0

			if(k >= bs(3) .and. k <= es(3)) then
				do j=bs(2),es(2)
					do i=bs(1),es(1)
						loc(i,j,1:7)	= var(i,j,k,1:7)
					enddo
				enddo
			endif

			do p=1,7
				call mpi_reduce(sum(loc(:,:,p)   )/nXY,stat1Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,p)**2)/nXY,stat2Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,4))/nXY,corrSpc(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,6))/nXY,corrSpc(k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,2)*loc(:,:,3))/nXY,corrSpc(k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,2)*loc(:,:,4))/nXY,corrSpc(k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,3)*loc(:,:,4))/nXY,corrSpc(k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,3)*loc(:,:,6))/nXY,corrSpc(k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,6)*loc(:,:,4))/nXY,corrSpc(k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,7)*loc(:,:,4))/nXY,corrSpc(k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,7)*loc(:,:,6))/nXY,corrSpc(k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		enddo

		call mpi_bcast(stat1Spc(1,1),nP(3),mpi_double_precision,0,mpi_comm_world,ierr)

		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/moment1st.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat1Spc(k,1),stat1Spc(k,2),stat1Spc(k,3),stat1Spc(k,4)	&
																								,stat1Spc(k,5),stat1Spc(k,6),stat1Spc(k,7)
			enddo
			write (10, *)
			close(10)
			111	format(1(1X,F15.7))
			112	format(7(1X,F15.7))
			!-----------------------------------------------------------------
			open(unit=10,file="../output/moment2nd.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat2Spc(k,1),stat2Spc(k,2),stat2Spc(k,3),stat2Spc(k,4)	&
																								,stat2Spc(k,5),stat2Spc(k,6),stat2Spc(k,7)
			enddo
			write (10, *)
			close(10)
			!-----------------------------------------------------------------
			open(unit=10,file="../output/momentJoint.dat",access="append")
			write(10,111,advance="no")time
			do k=1,nP(3)
				write(10,113,advance="no") corrSpc(k,1),corrSpc(k,2),corrSpc(k,3)	&
																	,corrSpc(k,4),corrSpc(k,5),corrSpc(k,6)	&
																	,corrSpc(k,7),corrSpc(k,8),corrSpc(k,9)
			enddo
			write (10,*)
			close(10)
			113	format(9(1X,F15.7))
			!-----------------------------------------------------------------
		endif
	end subroutine write_stats

	!######################################################################

	subroutine TKE_stats()
	implicit none

			nXY=nP(1)*nP(2)
			nXY_2=(nP(1)-2)*(nP(2)-2)

			do k=1,nP(3)
				loc	=0.0d0
				locT=0.0d0
				if(k >= bs(3) .and. k <= es(3)) then
					do j=bs(2),es(2)
						do i=bs(1),es(1)
							loc(i,j,1:7)	= var(i,j,k,1:7)
						enddo
					enddo
				endif

				do p=1,7
					call mpi_reduce(sum(loc(:,:,p)   )/nXY,stat1Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(:,:,p)**2)/nXY,stat2Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				enddo

				call mpi_reduce(sum(loc(:,:,1)*loc(:,:,4))/nXY,corrSpc(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,1)*loc(:,:,6))/nXY,corrSpc(k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,2)*loc(:,:,3))/nXY,corrSpc(k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,2)*loc(:,:,4))/nXY,corrSpc(k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,3)*loc(:,:,4))/nXY,corrSpc(k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,3)*loc(:,:,6))/nXY,corrSpc(k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,6)*loc(:,:,4))/nXY,corrSpc(k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,7)*loc(:,:,4))/nXY,corrSpc(k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,7)*loc(:,:,6))/nXY,corrSpc(k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)

				!---------------------------TKE STATS-----------------------------
				if(k >= bs(3) .and. k <= es(3)) then
					do j=bn(2),en(2)
						do i=bn(1),en(1)
							locT(i,j,1)		= (var(i+1,j,k,2) - var(i-1,j,k,2))*haf*invdx									!dudx
							dwdx 					= (var(i+1,j,k,4) - var(i-1,j,k,4))*haf*invdx 								!dwdx
							dwdy 					= (var(i,j+1,k,4) - var(i,j-1,k,4))*haf*invdy 								!dwdy
							if (k.eq.nP(3)) then
									locT(i,j,2) = (		b1d2c0*var(i,j,k  ,2)	&																!dudz
																	+ b1d2c1*var(i,j,k-1,2) &
																	+	b1d2c2*var(i,j,k-2,2)	&
																)
									locT(i,j,4) = (		b1d2c0*var(i,j,k  ,3)	&																!dvdz
																	+ b1d2c1*var(i,j,k-1,3) &
																	+	b1d2c2*var(i,j,k-2,3)	&
																)															
									locT(i,j,5) = (		b1d2c0*var(i,j,k  ,4)	&																!dwdz
																	+ b1d2c1*var(i,j,k-1,4) &
																	+	b1d2c2*var(i,j,k-2,4)	&
																)
								elseif (k.eq.1) then
									locT(i,j,2) = (		f1d2c0*var(i,j,k	,2)	&																!dudz
																	+ f1d2c1*var(i,j,k+1,2) &
																	+ f1d2c2*var(i,j,k+2,2)	&
																)
									locT(i,j,4) = (		f1d2c0*var(i,j,k	,3)	&																!dvdz
																	+ f1d2c1*var(i,j,k+1,3) &
																	+ f1d2c2*var(i,j,k+2,3)	&
																)
									locT(i,j,5) = (		f1d2c0*var(i,j,k	,4)	&																!dwdz
																	+ f1d2c1*var(i,j,k+1,4) &
																	+ f1d2c2*var(i,j,k+2,4)	&
																)
								else
									locT(i,j,2) = (var(i,j,k+1,2)	- var(i,j,k-1,2))*haf*invdXi*invJac(k)		!dudz
									locT(i,j,4) = (var(i,j,k+1,3)	- var(i,j,k-1,3))*haf*invdXi*invJac(k)		!dvdz
									locT(i,j,5) = (var(i,j,k+1,4)	- var(i,j,k-1,4))*haf*invdXi*invJac(k)		!dwdz
							endif
								locT(i,j,6)	=	(var(i,j+1,k,3) - var(i,j-1,k,3))*haf*invdy									!dvdy
								locT(i,j,3)	= mu(i,j,k)*(dwdx + locT(i,j,2))*invRe												!tauxz
								locT(i,j,7)	=	mu(i,j,k)*(dwdy + locT(i,j,4))*invRe												!tauyz
								locT(i,j,8)	=	mu(i,j,k)*(two*locT(i,j,5) - two3rd*(	locT(i,j,1) + locT(i,j,6)	+ locT(i,j,5)))*invRe	!tauzz
						enddo
					enddo
				endif

				do p=1,8
					call mpi_reduce(sum(locT(:,:,p))/nXY_2,stat1SpcT(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(locT(:,:,p)**2)/nXY_2,stat2SpcT(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				enddo

				call mpi_reduce(sum(loc (:,:,7)*locT(:,:,1))/nXY_2,corrSpcT(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!pdudx
				call mpi_reduce(sum(loc (:,:,2)*locT(:,:,3))/nXY_2,corrSpcT(k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauxz_u
				call mpi_reduce(sum(locT(:,:,3)*locT(:,:,2))/nXY_2,corrSpcT(k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauxz_dudz
				call mpi_reduce(sum((stat1Spc(k,1)*loc(:,:,2)**2*loc(:,:,4))*haf)/nXY,corrSpcT(k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)	!rho_u_u_w

				call mpi_reduce(sum(loc (:,:,7)*locT(:,:,6))/nXY_2,corrSpcT(k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!pdvdy
				call mpi_reduce(sum(loc (:,:,3)*locT(:,:,7))/nXY_2,corrSpcT(k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauyz_v
				call mpi_reduce(sum(locT(:,:,4)*locT(:,:,7))/nXY_2,corrSpcT(k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauyz_dvdz
				call mpi_reduce(sum((stat1Spc(k,1)*loc(:,:,3)**2*loc(:,:,4))*haf)/nXY,corrSpcT(k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)	!rho_v_v_w

				call mpi_reduce(sum(loc (:,:,7)*locT(:,:,5))/nXY_2,corrSpcT(k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!pdwdz
				call mpi_reduce(sum(loc (:,:,4)*locT(:,:,8))/nXY_2,corrSpcT(k,10),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauzz_w
				call mpi_reduce(sum(locT(:,:,5)*locT(:,:,8))/nXY_2,corrSpcT(k,11),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)							!tauzz_dwdz
				call mpi_reduce(sum((stat1Spc(k,1)*loc(:,:,4)**3)*haf)/nXY,corrSpcT(k,12),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)						!rho_w_w_w
				enddo

			call mpi_bcast(stat1Spc(1,1),nP(3),mpi_double_precision,0,mpi_comm_world,ierr)

			if(master) then
				!-----------------------------------------------------------------
				open(unit=10,file="../output/moment1st.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					write(10,112,advance="no") stat1Spc(k,1),stat1Spc(k,2),stat1Spc(k,3),stat1Spc(k,4)	&
																									,stat1Spc(k,5),stat1Spc(k,6),stat1Spc(k,7)
				enddo
				write (10, *)
				close(10)
				111	format(1(1X,F15.7))
				112	format(7(1X,F15.7))
				!-----------------------------------------------------------------
				open(unit=10,file="../output/moment2nd.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					write(10,112,advance="no") stat2Spc(k,1),stat2Spc(k,2),stat2Spc(k,3),stat2Spc(k,4)	&
																									,stat2Spc(k,5),stat2Spc(k,6),stat2Spc(k,7)
				enddo
				write (10, *)
				close(10)
				!-----------------------------------------------------------------
				open(unit=10,file="../output/momentJoint.dat",access="append")
				write(10,111,advance="no")time
				do k=1,nP(3)
					write(10,113,advance="no") corrSpc(k,1),corrSpc(k,2),corrSpc(k,3)	&
																		,corrSpc(k,4),corrSpc(k,5),corrSpc(k,6)	&
																		,corrSpc(k,7),corrSpc(k,8),corrSpc(k,9)
				enddo
				write (10,*)
				close(10)
				113	format(9(1X,F15.7))
				!-----------------------------------------------------------------
				open(unit=10,file="../output/moment1stT.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					write(10,115,advance="no") stat1SpcT(k,1),stat1SpcT(k,2),stat1SpcT(k,3),stat1SpcT(k,4)	&
																		,stat1SpcT(k,5),stat1SpcT(k,6),stat1SpcT(k,7),stat1SpcT(k,8)
				enddo
				write (10, *)
				close(10)
				115	format(8(1X,F15.7))
				!-----------------------------------------------------------------
				open(unit=10,file="../output/moment2ndT.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					write(10,115,advance="no") stat2SpcT(k,1),stat2SpcT(k,2),stat2SpcT(k,3),stat2SpcT(k,4)	&
																		,stat2SpcT(k,5),stat2SpcT(k,6),stat2SpcT(k,7),stat2SpcT(k,8)
				enddo
				write (10, *)
				close(10)
				
				!-----------------------------------------------------------------
				open(unit=10,file="../output/momentJointT.dat",access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					write(10,216,advance="no") corrSpcT(k,1),corrSpcT(k,2),corrSpcT(k,3),corrSpcT(k,4)	&
																		,corrSpcT(k,5),corrSpcT(k,6),corrSpcT(k,7),corrSpcT(k,8)	&
																		,corrSpcT(k,9),corrSpcT(k,10),corrSpcT(k,11),corrSpcT(k,12)
				enddo
				write (10,*)
				close(10)
				216	format(12(1X,F15.7))	
			endif
	end subroutine TKE_stats

	!######################################################################

	subroutine NU_stats()
		implicit none
		nXY=nP(1)*nP(2)
		nXY_haf=nXY/2

		do k=1,nP(3)
			loc=0.0d0

			if(k >= bs(3) .and. k <= es(3) .and. bs(1) .lt. 66) then
				do j=bs(2),es(2)
					do i=bs(1),65
						loc(i,j,1:7)	= var(i,j,k,1:7)
					enddo
				enddo
			endif

			do p=1,7
				call mpi_reduce(sum(loc(:,:,p)   )/nXY_haf,stat1Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				call mpi_reduce(sum(loc(:,:,p)**2)/nXY_haf,stat2Spc(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,4))/nXY_haf,corrSpc(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,1)*loc(:,:,6))/nXY_haf,corrSpc(k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,2)*loc(:,:,3))/nXY_haf,corrSpc(k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,2)*loc(:,:,4))/nXY_haf,corrSpc(k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,3)*loc(:,:,4))/nXY_haf,corrSpc(k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,3)*loc(:,:,6))/nXY_haf,corrSpc(k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,6)*loc(:,:,4))/nXY_haf,corrSpc(k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,7)*loc(:,:,4))/nXY_haf,corrSpc(k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			call mpi_reduce(sum(loc(:,:,7)*loc(:,:,6))/nXY_haf,corrSpc(k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		enddo

		call mpi_bcast(stat1Spc(1,1),nP(3),mpi_double_precision,0,mpi_comm_world,ierr)

		if(master) then
			!-----------------------------------------------------------------
			open(unit=10,file="../output/moment1st.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat1Spc(k,1),stat1Spc(k,2),stat1Spc(k,3),stat1Spc(k,4)	&
																								,stat1Spc(k,5),stat1Spc(k,6),stat1Spc(k,7)
			enddo
			write (10, *)
			close(10)
			111	format(1(1X,F15.7))
			112	format(7(1X,F15.7))
			!-----------------------------------------------------------------
			open(unit=10,file="../output/moment2nd.dat",access="append")
			write(10,111,advance="no") time
			do k=1,nP(3)
				write(10,112,advance="no") stat2Spc(k,1),stat2Spc(k,2),stat2Spc(k,3),stat2Spc(k,4)	&
																								,stat2Spc(k,5),stat2Spc(k,6),stat2Spc(k,7)
			enddo
			write (10, *)
			close(10)
			!-----------------------------------------------------------------
			open(unit=10,file="../output/momentJoint.dat",access="append")
			write(10,111,advance="no")time
			do k=1,nP(3)
				write(10,113,advance="no") corrSpc(k,1),corrSpc(k,2),corrSpc(k,3)	&
																	,corrSpc(k,4),corrSpc(k,5),corrSpc(k,6)	&
																	,corrSpc(k,7),corrSpc(k,8),corrSpc(k,9)
			enddo
			write (10,*)
			close(10)
			113	format(9(1X,F15.7))

		endif
		!------------------------------NU STATS-----------------------------
		do q = 70,120,10
			loc=0.0d0

			if (bs(1) .le. q .and. es(1) .ge. q) then

				do k=1,nP(3)

					loc=0.0d0

					if(k >= bs(3) .and. k <= es(3)) then
						do j=bs(2),es(2)
								loc(q,j,1:7)	= var(q,j,k,1:7)
						enddo
					endif

					do p=1,7
						call mpi_reduce(sum(loc(q,:,p)   )/nP(2),stat1SpcNU(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
						call mpi_reduce(sum(loc(q,:,p)**2)/nP(2),stat2SpcNU(k,p),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					enddo

					call mpi_reduce(sum(loc(q,:,1)*loc(q,:,4))/nP(2),corrSpcNU(k,1),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,1)*loc(q,:,6))/nP(2),corrSpcNU(k,2),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,2)*loc(q,:,3))/nP(2),corrSpcNU(k,3),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,2)*loc(q,:,4))/nP(2),corrSpcNU(k,4),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,3)*loc(q,:,4))/nP(2),corrSpcNU(k,5),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,3)*loc(q,:,6))/nP(2),corrSpcNU(k,6),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,6)*loc(q,:,4))/nP(2),corrSpcNU(k,7),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,7)*loc(q,:,4))/nP(2),corrSpcNU(k,8),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
					call mpi_reduce(sum(loc(q,:,7)*loc(q,:,6))/nP(2),corrSpcNU(k,9),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
				enddo
			endif

			!CALL mpi_barrier(mpi_comm_world,ierr)

			if (master) then
				!-----------------------------------------------------------------
					WRITE(filename1,'(a,i3.3,a)')"../output/NU/moment1st",q,".dat"
					WRITE(filename2,'(a,i3.3,a)')"../output/NU/moment2nd",q,".dat"
					WRITE(filename3,'(a,i3.3,a)')"../output/NU/momentJoint",q,".dat"

				open(unit=10,file=filename1,access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					write(10,112,advance="no") stat1SpcNU(k,1),stat1SpcNU(k,2),stat1SpcNU(k,3),stat1SpcNU(k,4)	&
																										,stat1SpcNU(k,5),stat1SpcNU(k,6),stat1SpcNU(k,7)
				enddo
				write (10, *)
				close(10)
				!-----------------------------------------------------------------
				open(unit=10,file=filename2,access="append")
				write(10,111,advance="no") time
				do k=1,nP(3)
					write(10,112,advance="no") stat2SpcNU(k,1),stat2SpcNU(k,2),stat2SpcNU(k,3),stat2SpcNU(k,4)	&
																										,stat2SpcNU(k,5),stat2SpcNU(k,6),stat2SpcNU(k,7)
				enddo
				write (10, *)
				close(10)
				!-----------------------------------------------------------------
				open(unit=10,file=filename3,access="append")
				write(10,111,advance="no")time
				do k=1,nP(3)
					write(10,113,advance="no") corrSpcNU(k,1),corrSpcNU(k,2),corrSpcNU(k,3)	&
																		,corrSpcNU(k,4),corrSpcNU(k,5),corrSpcNU(k,6)	&
																		,corrSpcNU(k,7),corrSpcNU(k,8),corrSpcNU(k,9)
				enddo
				write (10,*)
				close(10)
			endif
		enddo
	end subroutine NU_stats

	!######################################################################

	subroutine tauWall_write()
	implicit none
		!-------------------------------------------------------------------
		tmpR1=0.0d0
		if(bottom) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					dwdx = mu(i,j,1) * (var(i+1,j,1,4) - var(i-1,j,1,4)) *haf*invdx !zero for impervious bottom wall

					dudz = mu(i,j,1)*	(		f1d2c0*var(i,j,1,2)	&
															+ f1d2c1*var(i,j,2,2) &
															+ f1d2c2*var(i,j,3,2)	&
														)

					tmpR1 = tmpR1 + (dwdx + dudz)*invRe
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,tauxzL,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		tauxzL= tauxzL/dble((nP(1)-2)*(nP(2)-2))

		call mpi_bcast(tauxzL,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		tmpR1=0.0d0
		if(top) then

			do j=bn(2),en(2)
				do i=bn(1),en(1)
					dwdx = mu(i,j,nP(3)) * (var(i+1,j,nP(3),4) - var(i-1,j,nP(3),4)) *haf*invdx !zero for impervious top wall

					dudz = mu(i,j,nP(3)) * (		b1d2c0*var(i,j,nP(3)  ,2)	&
																		+ b1d2c1*var(i,j,nP(3)-1,2) &
																		+	b1d2c2*var(i,j,nP(3)-2,2)	&
																	)

					tmpR1 = tmpR1 + (dwdx + dudz)*invRe
				enddo
			enddo

		endif

		CALL MPI_Reduce(tmpR1,tauxzU,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
		tauxzU= tauxzU/dble(((nP(1)-2)*(nP(2)-2)))

		call mpi_bcast(tauxzU,1,mpi_double_precision,0,mpi_comm_world,ierr)
		!-------------------------------------------------------------------

		if(master) then
			open(10,file="../output/tauWall.dat",access="append")
			write(10,110)time, tauxzL, tauxzU
			close(10)
		endif
		110	format(3(1X,F15.7))

	end subroutine tauWall_write

	!######################################################################

	subroutine Diagnostic_files()
		implicit none

			!------------------------Write Kinetic Energy-----------------------
			tmpR1=0.0d0
			tmpR2	=0.0d0
			do k=bs(3),es(3)
				do j=bs(2),es(2)
					do i=bs(1),es(1)
						tmpR1 = tmpR1 + sum(var(i,j,k,2:4)**two)
					enddo
				enddo
			enddo

			CALL MPI_Reduce(tmpR1,tmpR2,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)

			if(master) then
				tmpR2 = tmpR2/product(nP)
				open(10,file="../output/KE.dat",access="append")
				write(10,108) time,tmpR2
				close(10)
			endif
			108	format(2(1X,F15.7))
			!------------------------Write Bulk Density-------------------------
			Ayz = x3(nP(3))*x2(nP(2))

			tmpA1	=0.0d0
			do i=1,5
				tmpR1	=0.0d0

				if(plane(i) > bn(1) .and. plane(i) < en(1)) then
					do k=bs(3),en(3)
						do j=bs(2),en(2)
							dAyz = dy*(x3(k+1)-x3(k))
							tmpR1 =tmpR1 + quar*(	var(plane(i),j  ,k  ,1) +	&
																		var(plane(i),j+1,k  ,1) +	&
																		var(plane(i),j  ,k+1,1) +	&
																		var(plane(i),j+1,k+1,1)		&
																	)*dAyz
						enddo
					enddo
				endif

				CALL MPI_Reduce(tmpR1,tmpA1(i),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

			if(master) then
				tmpA1=tmpA1/Ayz
				open (unit=10,file="../output/rhobulk.dat",access="append")
				write(10,109)time,tmpA1(1),tmpA1(2),tmpA1(3),tmpA1(4),tmpA1(5)
				close(10)
			endif
			109	format(6(1X,F15.7))
			!------------------------Write Bulk Velocity------------------------
			tmpA1	=0.0d0
			do i=1,5
				tmpR1	=0.0d0

				if(plane(i) > bn(1) .and. plane(i) < en(1)) then
					do k=bs(3),en(3)
						do j=bs(2),en(2)
							dAyz = dy*(x3(k+1)-x3(k))
							tmpR1 =tmpR1 + quar*(	var(plane(i),j  ,k  ,2) +	&
																		var(plane(i),j+1,k  ,2) +	&
																		var(plane(i),j  ,k+1,2) +	&
																		var(plane(i),j+1,k+1,2)		&
																	)*dAyz
						enddo
					enddo
				endif

				CALL MPI_Reduce(tmpR1,tmpA1(i),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

			if(master) then
				tmpA1=tmpA1/Ayz
				open (unit=10,file="../output/Ubulk.dat",access="append")
				write(10,109)time,tmpA1(1),tmpA1(2),tmpA1(3),tmpA1(4),tmpA1(5)
				close(10)
			endif

			!--------------------------Write Massflow---------------------------
			tmpA1	=0.0d0
			do i=1,5
				tmpR1	=0.0d0

				if(plane(i) > bn(1) .and. plane(i) < en(1)) then
					do k=bs(3),en(3)
						do j=bs(2),en(2)
							dAyz = dy*(x3(k+1)-x3(k))
							tmpR1 =tmpR1 + quar*(	var(plane(i),j  ,k  ,1)*var(plane(i),j  ,k  ,2) +	&
																		var(plane(i),j+1,k  ,1)*var(plane(i),j+1,k  ,2) + &
																		var(plane(i),j  ,k+1,1)*var(plane(i),j  ,k+1,2) +	&
																		var(plane(i),j+1,k+1,1)*var(plane(i),j+1,k+1,2)		&
																	)*dAyz
						enddo
					enddo
				endif

				CALL MPI_Reduce(tmpR1,tmpA1(i),1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
			enddo

			if(master) then
				tmpA1=tmpA1/Ayz
				open (unit=10,file="../output/massflow.dat",access="append")
				write(10,109)time,tmpA1(1),tmpA1(2),tmpA1(3),tmpA1(4),tmpA1(5)
				close(10)
			endif

		!====================Write Diagnostic Files (ends)====================
	end subroutine Diagnostic_files

end module mod_stats