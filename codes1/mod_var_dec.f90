module mod_var_dec
use mpi
implicit none
save

!========================PARAMETERS DECLARATION=========================
integer,parameter,dimension(3)					::nP			=[192,130,160]
logical,parameter,dimension(3)					::period	=[.true.,.true.,.false.]
INTEGER, PARAMETER											::maxiter	=2,&
																					maxstep	=int(5E7)
DOUBLE PRECISION, PARAMETER							::gama		=1.4d0,&
																					invPr		=1.0d0/0.71d0,&
																					one			=1.0d0,&
																					two			=2.0d0,&
																					zer			=0.0d0,&
																					quar		=0.25d0,&
																					haf			=0.5d0,&
																					two3rd	=2.0d0/3.0d0,&
																					one12th	=1.0d0/12.0d0,&																	
																					threeby2=3.0d0/2.0d0,&
																					Ep 			=5.0d0

!-----------------------------------------------------------------------
integer::resOp,topo(3),istat,debugOp
double precision:: Re,invRe,Mac,dt,PexVal,PeyVal,PezVal,ggM,invdt
character(len=10)::descp_
character(len=100)::inputPath_
character(len=:),allocatable::descp,inputPath
double precision 	:: dXi,dx,dy,Pi,x1(nP(1)),x2(nP(2)),invdx,invdy,invdXi
double precision,dimension(nP(3)) :: x3,jac,jac2,Xi,invJac
!=======================Variable Declaration Starts====================
character(len=30)	  	:: filename
logical								:: isBlown=.false.
integer								:: i,j,k,m,p,nstep,iter,st,nXY
!integer,dimension(5)  :: plane =[5,10,70,100,125]
integer,dimension(5)  :: plane =[15,50,80,130,175]
double precision 			:: tStart, tEnd, tauxzL, tauxzU
double precision 			:: time, Pe, flux,dAyz,Ayz
!---------------------------COEFFICIENTS--------------------------------
double precision			:: f1d2c0, f1d2c1, f1d2c2, b1d2c0, b1d2c1, b1d2c2
double precision 			:: f2d1c0, f2d1c1, f2d1c2, b2d1c0, b2d1c1, b2d1c2
double precision 			:: dz1,dzN,alpha,dz2,dz,dzN_1
double precision			:: f2d2c0, f2d2c1, f2d2c2, f2d2c3
double precision			:: b2d2c0, b2d2c1, b2d2c2, b2d2c3
double precision			:: f1d3c0,f1d3c1,f1d3c2,f1d3c3,b1d3c0,b1d3c1,b1d3c2,b1d3c3
double precision			:: f2d3c0,f2d3c1,f2d3c2,f2d3c3,f2d3c4,b2d3c0,b2d3c1,b2d3c2,b2d3c3,b2d3c4
double precision 			:: df21,df31,df32,df41,df42,df43,df51,df52,df53,df54
double precision 			:: db12,db13,db23,db14,db24,db34,db15,db25,db35,db45
!--------------------------Temporary Variables--------------------------
integer								::tmpI1,tmpI2,tmpI3,tmpI4,tmpI5
double precision			::tmpR1,tmpR2,tmpR3,tmpR4,tmpA1(5),loc(nP(1),nP(2),7)
double precision			::Ut,Vt,Wt,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz
double precision			::dTdx,dTdy,dTdz,divg,muT,ktT,PressT,dissF,dissG,dissH
double precision			::dudxdz,dvdydz,mud2wdz2,drhow2dz,lhs,dudzdx,dvdzdy,dwdzdmudz
double precision			::drhouwdx,d2wdx2,drhovwdy,d2wdy2,drhowdt
integer,allocatable,dimension(:,:) 	::tArr,bb,ee
!------------------------------STATS Variables---------------------------
double precision,dimension(nP(3),7) :: stat1Spc	! In order of rho,u,v,w,e,t,p
double precision,dimension(nP(3),7) :: stat2Spc ! In order of rho,u,v,w,e,t,p
double precision,dimension(nP(3),9) :: corrSpc 	! In order of rhow,rhot,uv,uw,vw,vt,tw,pw,pt
!------------------------------Main Vairables---------------------------
double precision,dimension(0:nP(1)+1,0:nP(2)+1,0:nP(3)+1,5,2)	:: solV
double precision,dimension(nP(1),nP(2),nP(3),7)								:: var ! In order of rho,U,V,W,E,T,P
double precision,dimension(nP(1),nP(2),nP(3),5)								:: fD,gD,hD
double precision,dimension(nP(1),nP(2),nP(3),5)								:: srcV
double precision,dimension(nP(1),nP(2),nP(3),5,2)							:: F,G,H,R
double precision,dimension(nP(1),nP(2),nP(3))									:: mu,kt,rhow
!------------------------MPI Variables----------------------------------
integer,dimension(3)		:: bs,es,bn,en,siz,des,src,myDim,myCoord
integer		:: liney,yz1p, xz1p, xy1p, yz1p5v, xz1p5v, xy1p5v, yz1p7v, xz1p7v, xy1p7v
integer		:: lineyN, yzN1p, xzN1p, xyN1p, yzN2p, xzN2p, xyN2p, yz2p5v, xz2p5v, xy2p5v
integer		:: comm3d,nproc, id, ierr, sizDP, STATUS(mpi_status_size)
logical 	:: myperiod(3), bottom, top, east, west, north, south, master
integer		:: linex, linex7v, liney7v
!-------------------------Debug Variables-------------------------------
double precision,dimension(nP(3))			:: pl
!--------------------------------TKE------------------------------------
double precision			::locT(nP(1),nP(2),8)
integer								::nXY_2
double precision,dimension(nP(3),8) :: stat1SpcT! In order of dudx,dudz,tauxz,dvdz,dwdz,dvdy,tauyz,tauzz
double precision,dimension(nP(3),8) :: stat2SpcT! In order of dudx,dudz,tauxz,dvdz,dwdz,dvdy,tauyz,tauzz
double precision,dimension(nP(3),12):: corrSpcT ! In order of pdudx,tauxz_u,tauxz_dudz,rho_u_u_w, pdvdy,tauyz_v,tauyz_dvdz,rho_v_v_w, pdwdz,tauzz_w,tauzz_dwdz,rho_w^3
!-------------------------------NU STATS--------------------------------
double precision,dimension(nP(3),7) :: stat1SpcNU	! In order of rho,u,v,w,e,t,p
double precision,dimension(nP(3),7) :: stat2SpcNU ! In order of rho,u,v,w,e,t,p
double precision,dimension(nP(3),9) :: corrSpcNU 	! In order of rhow,rhot,uv,uw,vw,vt,tw,pw,pt
character(len=50)	  	:: filename1,filename2,filename3
integer		:: q,nXY_haf

double precision,dimension(:,:),allocatable :: stat1SpcJ	! In order of rho,u,v,w,e,t,p
!----------------------------x-Plane Avg--------------------------------
double precision,allocatable,dimension(:,:,:) :: Lstat1Spc	! In order of rho,u,v,w,e,t,p
double precision,allocatable,dimension(:,:,:) :: Lstat2Spc  ! In order of rho,u,v,w,e,t,p
double precision,allocatable,dimension(:,:,:) :: LcorrSpc 	! In order of rhow,rhot,uv,uw,vw,vt,tw,pw,pt
double precision,allocatable,dimension(:,:)   :: local				
!----------------------------CONTROL PARAMETERS-------------------------
DOUBLE PRECISION			::Beta,Amp_ls
double precision			::Kappa,C,Omega,Amp_v
!========================Variable Declaration Ends======================
integer :: y_d = 33

end module mod_var_dec