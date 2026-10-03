module mod_my_mpi

use mod_comdata

implicit none

contains

subroutine initialize_mpi()

	implicit none
	!==============================MPI======================================
	CALL mpi_init(ierr)
	CALL mpi_comm_rank(mpi_comm_world,id,ierr)
	CALL mpi_comm_size(mpi_comm_world,nproc,ierr)

	call mpi_cart_create(mpi_comm_world,3,topo,period,.false.,comm3d,ierr)
	call mpi_cart_get(comm3d,3,myDim,myperiod,mycoord,ierr)

	call mpi_cart_shift(comm3d,0,1,src(1),des(1),ierr)
	call mpi_cart_shift(comm3d,1,1,src(2),des(2),ierr)
	call mpi_cart_shift(comm3d,2,1,src(3),des(3),ierr)

	allocate(tArr(0:maxval(myDim)-1,3))
	allocate(bb(3,0:nproc-1),ee(3,0:nproc-1))

	master=(id==0)
	east	=(mycoord(1)==myDim(1)-1)
	west	=(mycoord(1)==0)
	north	=(mycoord(2)==myDim(2)-1)
	south	=(mycoord(2)==0)
	top		=(mycoord(3)==myDim(3)-1)
	bottom=(mycoord(3)==0)

	do k=1,3

		tmpI1=nP(k)/myDim(k)
		tmpI2=myDim(k)-mod(nP(k),myDim(k))

		tArr(0,1)=1
		do i=0,myDim(k)-1
			if(i==tmpI2) tmpI1=tmpI1+1
			tArr(i,2)=tArr(i,1)+tmpI1-1
			tArr(i,3)=tArr(i,2)-tArr(i,1)+1
			if(i==myDim(k)-1) exit
			tArr(i+1,1)=tArr(i,2)+1
		enddo

		do i=0,myDim(k)-1
			if(i==mycoord(k)) then
				bs(k)=tArr(i,1)
				es(k)=tArr(i,2)
				siz(k)=tArr(i,3)
			endif
		enddo

		bn(k)=bs(k)
		en(k)=es(k)

		if((west .and. k==1) .or. (south 	.and. k==2) .or. (bottom .and. k==3)) then
			bn(k)=bs(k)+1
			siz(k)=siz(k)-1
		endif

		if((east .and. k==1) .or. (north 	.and. k==2) .or. (top 	 .and. k==3))	then
			en(k)=es(k)-1
			siz(k)=siz(k)-1
		endif

	enddo
	!-----------------------------------------------------------------------
	CALL mpi_type_extent (mpi_double_precision,sizDP,ierr)
	CALL mpi_type_vector (siz(2), 1, nP(1), mpi_double_precision,liney ,ierr)
	CALL mpi_type_contiguous(siz(1),mpi_double_precision,linex,ierr)

	CALL mpi_type_commit(liney,ierr)
	CALL mpi_type_commit(linex,ierr)
	
	CALL mpi_type_hvector(siz(3), 1			, nP(1)*nP(2)*sizDP	,liney								,yz1p,ierr)
	CALL mpi_type_vector (siz(3), siz(1), nP(1)*nP(2)				, mpi_double_precision,xz1p,ierr)
	CALL mpi_type_vector (siz(2), siz(1), nP(1)							, mpi_double_precision,xy1p,ierr)

	CALL mpi_type_commit(yz1p,ierr)
	CALL mpi_type_commit(xz1p,ierr)
	CALL mpi_type_commit(xy1p,ierr)

	CALL mpi_type_hvector(5, 1 , product(nP)*sizDP , yz1p , yz1p5v ,ierr)
	CALL mpi_type_hvector(5, 1 , product(nP)*sizDP , xz1p , xz1p5v ,ierr)
	CALL mpi_type_hvector(5, 1 , product(nP)*sizDP , xy1p , xy1p5v ,ierr)

	CALL mpi_type_commit(yz1p5v,ierr)
	CALL mpi_type_commit(xz1p5v,ierr)
	CALL mpi_type_commit(xy1p5v,ierr)

	CALL mpi_type_hvector(7, 1 , product(nP)*sizDP , yz1p , yz1p7v ,ierr)
	CALL mpi_type_hvector(7, 1 , product(nP)*sizDP , xz1p , xz1p7v ,ierr)
	CALL mpi_type_hvector(7, 1 , product(nP)*sizDP , xy1p , xy1p7v ,ierr)


	CALL mpi_type_commit(yz1p7v,ierr)
	CALL mpi_type_commit(xz1p7v,ierr)
	CALL mpi_type_commit(xy1p7v,ierr)

	CALL mpi_type_hvector(7, 1 , product(nP)*sizDP , liney , liney7v ,ierr)
	CALL mpi_type_hvector(7, 1 , product(nP)*sizDP , linex , linex7v ,ierr)

	CALL mpi_type_commit(liney7v,ierr)
	CALL mpi_type_commit(linex7v,ierr)
	!-----------------------------------------------------------------------
	CALL mpi_type_vector (siz(2),			 1, nP(1)+2, mpi_double_precision,lineyN ,ierr)

	CALL mpi_type_hvector(siz(3),			 1, (nP(1)+2)*(nP(2)+2)*sizDP,lineyN,yzN1p,ierr)
	CALL mpi_type_vector (siz(3), siz(1), (nP(1)+2)*(nP(2)+2), mpi_double_precision,xzN1p ,ierr)
	CALL mpi_type_vector (siz(2), siz(1), nP(1)+2, mpi_double_precision,xyN1p ,ierr)

	CALL mpi_type_hvector(2, 1, sizDP      								, yzN1p,yzN2p,ierr)
	CALL mpi_type_hvector(2, 1, (nP(1)+2)*sizDP   				, xzN1p,xzN2p,ierr)
	CALL mpi_type_hvector(2, 1, (nP(1)+2)*(nP(2)+2)*sizDP	, xyN1p,xyN2p,ierr)

	CALL mpi_type_hvector(5, 1 , product(nP+2)*sizDP , yzN2p , yz2p5v ,ierr)
	CALL mpi_type_hvector(5, 1 , product(nP+2)*sizDP , xzN2p , xz2p5v ,ierr)
	CALL mpi_type_hvector(5, 1 , product(nP+2)*sizDP , xyN2p , xy2p5v ,ierr)

	CALL mpi_type_commit(yz2p5v,ierr)
	CALL mpi_type_commit(xz2p5v,ierr)
	CALL mpi_type_commit(xy2p5v,ierr)

	CALL mpi_barrier(mpi_comm_world,ierr)
	!-----------------------------MPI---------------------------------------
		
end subroutine initialize_mpi

subroutine transfer_2p_solV()
	implicit none
	call mpi_sendrecv(solV(en(1)-1,bn(2),bn(3),1,1),1,yz2p5v,des(1),50,solV(bn(1)-2,bn(2),bn(3),1,1),1,yz2p5v,src(1),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(solV(bn(1)  ,bn(2),bn(3),1,1),1,yz2p5v,src(1),50,solV(en(1)+1,bn(2),bn(3),1,1),1,yz2p5v,des(1),50,mpi_comm_world,STATUS,ierr)
	
	call mpi_sendrecv(solV(bn(1),en(2)-1,bn(3),1,1),1,xz2p5v,des(2),50,solV(bn(1),bn(2)-2,bn(3),1,1),1,xz2p5v,src(2),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(solV(bn(1),bn(2)  ,bn(3),1,1),1,xz2p5v,src(2),50,solV(bn(1),en(2)+1,bn(3),1,1),1,xz2p5v,des(2),50,mpi_comm_world,STATUS,ierr)
	
	call mpi_sendrecv(solV(bn(1),bn(2),en(3)-1,1,1),1,xy2p5v,des(3),50,solV(bn(1),bn(2),bn(3)-2,1,1),1,xy2p5v,src(3),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(solV(bn(1),bn(2),bn(3)  ,1,1),1,xy2p5v,src(3),50,solV(bn(1),bn(2),en(3)+1,1,1),1,xy2p5v,des(3),50,mpi_comm_world,STATUS,ierr)
endsubroutine transfer_2p_solV

subroutine transfer_1p_var()
	implicit none
	call mpi_sendrecv(var(en(1),bn(2),bn(3),1),1,yz1p7v,des(1),50,var(bn(1)-1,bn(2),bn(3),1),1,yz1p7v,src(1),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(var(bn(1),bn(2),bn(3),1),1,yz1p7v,src(1),50,var(en(1)+1,bn(2),bn(3),1),1,yz1p7v,des(1),50,mpi_comm_world,STATUS,ierr)
	
	call mpi_sendrecv(var(bn(1),en(2),bn(3),1),1,xz1p7v,des(2),50,var(bn(1),bn(2)-1,bn(3),1),1,xz1p7v,src(2),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(var(bn(1),bn(2),bn(3),1),1,xz1p7v,src(2),50,var(bn(1),en(2)+1,bn(3),1),1,xz1p7v,des(2),50,mpi_comm_world,STATUS,ierr)
	
	call mpi_sendrecv(var(bn(1),bn(2),en(3),1),1,xy1p7v,des(3),50,var(bn(1),bn(2),bn(3)-1,1),1,xy1p7v,src(3),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(var(bn(1),bn(2),bn(3),1),1,xy1p7v,src(3),50,var(bn(1),bn(2),en(3)+1,1),1,xy1p7v,des(3),50,mpi_comm_world,STATUS,ierr)
endsubroutine transfer_1p_var

subroutine transfer_FHD()
	implicit none
	call mpi_sendrecv(fD(en(1),bn(2),bn(3),1),1,yz1p5v,des(1),50,fD(bn(1)-1,bn(2),bn(3),1),1,yz1p5v,src(1),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(fD(bn(1),bn(2),bn(3),1),1,yz1p5v,src(1),50,fD(en(1)+1,bn(2),bn(3),1),1,yz1p5v,des(1),50,mpi_comm_world,STATUS,ierr)

	call mpi_sendrecv(gD(bn(1),en(2),bn(3),1),1,xz1p5v,des(2),50,gD(bn(1),bn(2)-1,bn(3),1),1,xz1p5v,src(2),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(gD(bn(1),bn(2),bn(3),1),1,xz1p5v,src(2),50,gD(bn(1),en(2)+1,bn(3),1),1,xz1p5v,des(2),50,mpi_comm_world,STATUS,ierr)

	call mpi_sendrecv(hD(bn(1),bn(2),en(3),1),1,xy1p5v,des(3),50,hD(bn(1),bn(2),bn(3)-1,1),1,xy1p5v,src(3),50,mpi_comm_world,STATUS,ierr)
	call mpi_sendrecv(hD(bn(1),bn(2),bn(3),1),1,xy1p5v,src(3),50,hD(bn(1),bn(2),en(3)+1,1),1,xy1p5v,des(3),50,mpi_comm_world,STATUS,ierr)
endsubroutine transfer_FHD

subroutine transfer_xyline()
	implicit none
	if (bottom) then
		call mpi_sendrecv(var(en(1),bn(2),1,1),1,liney7v,des(1),50,var(bn(1)-1,bn(2),1,1),1,liney7v,src(1),50,mpi_comm_world,STATUS,ierr)
		call mpi_sendrecv(var(bn(1),bn(2),1,1),1,liney7v,src(1),50,var(en(1)+1,bn(2),1,1),1,liney7v,des(1),50,mpi_comm_world,STATUS,ierr)
		
		call mpi_sendrecv(var(bn(1),en(2),1,1),1,linex7v,des(2),50,var(bn(1),bn(2)-1,1,1),1,linex7v,src(2),50,mpi_comm_world,STATUS,ierr)
		call mpi_sendrecv(var(bn(1),bn(2),1,1),1,linex7v,src(2),50,var(bn(1),en(2)+1,1,1),1,linex7v,des(2),50,mpi_comm_world,STATUS,ierr)
	endif
	if (top) then
		call mpi_sendrecv(var(en(1),bn(2),nP(3),1),1,liney7v,des(1),50,var(bn(1)-1,bn(2),nP(3),1),1,liney7v,src(1),50,mpi_comm_world,STATUS,ierr)
		call mpi_sendrecv(var(bn(1),bn(2),nP(3),1),1,liney7v,src(1),50,var(en(1)+1,bn(2),nP(3),1),1,liney7v,des(1),50,mpi_comm_world,STATUS,ierr)
		
		call mpi_sendrecv(var(bn(1),en(2),nP(3),1),1,linex7v,des(2),50,var(bn(1),bn(2)-1,nP(3),1),1,linex7v,src(2),50,mpi_comm_world,STATUS,ierr)
		call mpi_sendrecv(var(bn(1),bn(2),nP(3),1),1,linex7v,src(2),50,var(bn(1),en(2)+1,nP(3),1),1,linex7v,des(2),50,mpi_comm_world,STATUS,ierr)
	endif
endsubroutine transfer_xyline

subroutine finalize_mpi()
	implicit none
	CALL mpi_type_free(yz1p,ierr)
	CALL mpi_type_free(xz1p,ierr)
	CALL mpi_type_free(xy1p,ierr)
	
	CALL mpi_type_free(yz1p5v,ierr)
	CALL mpi_type_free(xz1p5v,ierr)
	CALL mpi_type_free(xy1p5v,ierr)
	
	CALL mpi_type_free(yz2p5v,ierr)
	CALL mpi_type_free(xz2p5v,ierr)
	CALL mpi_type_free(xy2p5v,ierr)
	
	CALL mpi_type_free(yz1p7v,ierr)
	CALL mpi_type_free(xz1p7v,ierr)
	CALL mpi_type_free(xy1p7v,ierr)
	
	CALL mpi_finalize(ierr)
endsubroutine finalize_mpi

end module mod_my_mpi