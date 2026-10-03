module mod_comdata

use mod_var_dec

implicit none

contains

subroutine readParam()
implicit none
open(unit=10,file="../param.dat",status="old")

read(10,*)
read(10,*)
read(10,'(48X,A)') descp_
read(10,'(48X,i1)') resOp
read(10,*)
read(10,'(48X,i2)') topo(1)
read(10,'(48X,i2)') topo(2)
read(10,'(48X,i2)') topo(3)
read(10,*)
read(10,*)
read(10,'(48X,F4.0)') Re
read(10,'(48X,f4.2)') Mac
read(10,'(48X,f6.4)') dt
read(10,'(48X,i9)') istat
read(10,*)
read(10,'(48X,f3.1)') PexVal
read(10,'(48X,f3.1)') PeyVal
read(10,'(48X,f4.2)') PezVal
read(10,*)
read(10,'(48X,i1)') debugOp
read(10,'(48X,A)') inputPath_
read(10,*)
read(10,*)
read(10,*)
read(10,*)
read(10,'(48X,F7.4)') Kappa
read(10,'(48X,f7.4)') Omega
read(10,'(48X,f4.2)') Amp_v
read(10,*)
read(10,'(48X,f5.2)') Beta
read(10,'(48X,f9.6)') Amp_ls

close(10)

invdt	=	1.0d0/dt
invRe = 1.0d0/Re
ggM = gama*(gama-1)*Mac**two

allocate(character(len=len(trim(descp_))) :: descp)
allocate(character(len=len(trim(inputPath_ ))) :: inputPath )

descp=trim(descp_)
inputPath =trim(inputPath_)

end subroutine readParam


subroutine setupMesh()
implicit none
integer					:: i,j,k

Pi=4.d0*datan(1.0d0)

x1(1) = 0.0d0
x2(1) = 0.0d0
Xi(1) =0.0d0
x3(1) =0.0d0

x1(nP(1)) = 4.0d0*Pi
x2(nP(2)) = 4.0d0*Pi/3.0d0
x3(nP(3)) = 2.0d0

dx = x1(nP(1))/dble(nP(1)-1)
dy = x2(nP(2))/dble(nP(2)-1)
dXi= x3(nP(3))/dble(nP(3)-1)

invdx = 1.0d0/dx
invdy = 1.0d0/dy
invdXi= 1.0d0/dXi

do i=1,nP(1)-1
	x1(i+1) =x1(i) + dx    
enddo

do j=1,nP(2)-1
	x2(j+1) =x2(j) + dy    
enddo

do k=1,nP(3)-1
	Xi(k+1) =Xi(k) + dXi
enddo

do k=1,nP(3)
	x3(k)		=	1.0d0 + dtanh(Ep*0.50d0*(Xi(k)-1)) / (dtanh(Ep*0.50d0))
	jac(k)	= 0.50d0*Ep * (1-(dtanh(0.50d0*Ep*(Xi(k)-1)))**2) / (dtanh(Ep*0.50d0))
	jac2(k)	= -0.50d0*Ep**2 * (1/(dcosh(0.50d0*Ep*(Xi(k)-1)))**2) * (dtanh(Ep*0.50d0*(Xi(k)-1))) / (dtanh(Ep*0.50d0))
enddo
invJac(2:nP(3))	= 1.0d0/jac(2:nP(3))
x3(1) =0.0d0

if(debugOp==1) then
	OPEN (UNIT=10,FILE='Mesh.dat')

		write(10,*) 'TITLE ="Mesh"'
		write(10,*) 'Variables ="x","y","z"'
		write(10,104) 'ZONE k=',nP(3),',j=',nP(2),',i=',nP(1),',DATAPACKING="POINT"'

		do k=1,nP(3)
			write(10,*)
			do j=1,nP(2)
				write(10,*)
				do i=1,nP(1)
					write(10,117) x1(i),x2(j),x3(k)
				end do
			end do
		enddo
		close(10)	

endif
104	format(3(A,I3),A)
117 format(3(F6.4))

end subroutine setupMesh

subroutine Coefficients()
implicit none
!-----------------------------------------------------------------------
dz1		= x3(2)-x3(1)
dz2 	=	x3(3)-x3(2)
dz		=	(dz1 + dz2)/two
alpha	= (x3(3)-x3(2))/dz1

tmpR1	= dz1*alpha*(alpha + one)
f1d2c0	= (one - (alpha + one)**two)/tmpR1
f1d2c1	= ((alpha + one)**two)/tmpR1
f1d2c2	= -one / tmpR1

tmpR1	= haf*dz1**two*alpha*(alpha + one)
f2d1c0	= alpha/tmpR1
f2d1c1	= -(alpha + one)/tmpR1
f2d1c2	= one / tmpR1

df21 = x3(2)-x3(1)

df31 = x3(3)-x3(1)
df32 = x3(3)-x3(2)   

df41 = x3(4)-x3(1)
df42 = x3(4)-x3(2)
df43 = x3(4)-x3(3)

f1d3c0=  ( df31*df21 + df31*df41 + df21*df41 )/( (-df21)*(-df31)*(-df41) )        
f1d3c1=  ( df41*df31	       								 )/(   df21 *(-df32)*(-df42) )
f1d3c2=  ( df41*df21  					             )/(   df31 *  df32 *(-df43) )
f1d3c3=  ( df31*df21  					             )/(   df41 *  df42 *	 df43  )

f2d2c0 =  ( 2.0d0*(-df21-df31-df41) )/( (-df21)*(-df31)*(-df41) )        
f2d2c1 =  ( 2.0d0*(-df31-df41)      )/(   df21 *(-df32)*(-df42) )
f2d2c2 =  ( 2.0d0*(-df21-df41)      )/(   df31 *  df32 *(-df43) )
f2d2c3 =  ( 2.0d0*(-df21-df31)      )/(   df41 *  df42 *  df43  )

f2d3c0=  ( 2.0d0*(df21*(df31+df41+df51) + df31*(df41+df51) + df41*df51))/( (-df21)*(-df31)*(-df41)*(-df51) )
f2d3c1=  ( 2.0d0*(df31*df41 			 			+ df31*df51        + df41*df51))/(   df21 *(-df32)*(-df42)*(-df52) )
f2d3c2=  ( 2.0d0*(df21*df41							+ df21*df51				 + df41*df51))/(   df31 *  df32 *(-df43)*(-df53) )
f2d3c3=  ( 2.0d0*(df21*df31							+ df21*df51				 + df31*df51))/(   df41 *  df42 *  df43 *(-df54) )
f2d3c4=  ( 2.0d0*(df21*df31							+ df21*df41				 + df31*df41))/(   df51 *  df52 *  df53 *  df54  )

!-----------------------------------------------------------------------
dzN		= x3(nP(3))-x3(nP(3)-1)
dzN_1	=	x3(nP(3)-1)-x3(nP(3)-2)
alpha	= (x3(nP(3)-1)-x3(nP(3)-2))/dzN

tmpR1	= dzN*alpha*(alpha + one)
b1d2c0	= ((alpha + one)**two - one)/tmpR1
b1d2c1	= -((alpha + one)**two)/tmpR1
b1d2c2	= one / tmpR1

tmpR1	= haf*dzN**two*alpha*(alpha + one)
b2d1c0	= alpha/tmpR1
b2d1c1	= -(alpha + one)/tmpR1
b2d1c2	= one / tmpR1

db12 =x3(nP(3))  -x3(nP(3)-1) 

db13 =x3(nP(3))  -x3(nP(3)-2)
db23 =x3(nP(3)-1)-x3(nP(3)-2)

db14 =x3(nP(3))  -x3(nP(3)-3)
db24 =x3(nP(3)-1)-x3(nP(3)-3)    
db34 =x3(nP(3)-2)-x3(nP(3)-3)

b1d3c0=  ( db13*db12 + db13*db14 + db12*db14 )/(   db12 *	 db13 *  db14 	)        
b1d3c1=  ( db14*db13	       								 )/( (-db12)*	 db23 *  db24 	)
b1d3c2=  ( db14*db12  					             )/( (-db13)*(-db23)*	 db34 	)
b1d3c3=  ( db13*db12  					             )/( (-db14)*(-db24)*(-db34)  )

b2d2c0 =  ( 2.0d0*(db12 + db13 + db14	))/(   db12 *	 db13 *  db14  )        
b2d2c1 =  ( 2.0d0*(db13 + db14        ))/( (-db12)*  db23 *  db24  )
b2d2c2 =  ( 2.0d0*(db12 + db14        ))/( (-db13)*(-db23)*	 db34  )
b2d2c3 =  ( 2.0d0*(db12 + db13        ))/( (-db14)*(-db24)*(-db34) )

b2d3c0=  ( 2.0d0*(db12*(db13+db14+db15) + db13*(db14+db15) + db14*db15))/( 	 db12 *  db13 *  db14 *  db15  )
b2d3c1=  ( 2.0d0*(db13*db14 			 			+ db13*db15        + db14*db15))/( (-db12)*  db23 *  db24 *  db25  )
b2d3c2=  ( 2.0d0*(db12*db14							+ db12*db15				 + db14*db15))/( (-db13)*(-db23)*  db34 *  db35  )
b2d3c3=  ( 2.0d0*(db12*db13							+ db12*db15				 + db13*db15))/( (-db14)*(-db24)*(-db34)*  db45  )
b2d3c4=  ( 2.0d0*(db12*db13							+ db12*db14				 + db13*db14))/( (-db15)*(-db25)*(-db35)*(-db45) )
!-----------------------------------------------------------------------
end subroutine Coefficients

function dateTime()

implicit none
character(len=30)::dateTime
character(len=3):: ampm
integer:: d,h,m,n,s,y,mm,values(8)
character(len=3), parameter, dimension(12) :: &
month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

call date_and_time(values=values)

y=values(1)
m=values(2)
d=values(3)
h=values(5)
n=values(6)
s=values(7)
mm=values(8)

if(h<12) then
	ampm='AM'
elseif(h==12) then
	if(n==0 .and. s==0) then
		ampm='Noon'
	else
		ampm='PM'
	endif
else
	h=h-12
	if(h<12) then
		ampm='PM'
	elseif(h==12) then
		if(n==0 .and. s==0) then
			ampm='Midnight'
		else
			ampm='AM'
		endif
	endif
endif

write(dateTime,'(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)')&
d,trim(month(m)),y,h,':',n,':',s,'.',mm,trim(ampm)
end function dateTime

subroutine print_param()
	implicit none
	if(master) then
		write(6,'(A)')"==================================================================================="
		write(6,'(2A)')"Compressible Channel Flow Solver started at :",dateTime()
		write(6,'(A)')"==================================================================================="
		write(6,'(A)')"Parameters read from param.dat"
		write(6,'(A)')"-----------------------------------------------------------------------------------"
		write(6,'(3A)')"Case Description                : ",descp
		write(6,'(A,3(I3,1X))')"Process Topology        : ",topo(1),topo(2),topo(3)
		write(6,'(A,3(I3,1X))')"Domain                  : ",nP(1),nP(2),nP(3)
		write(6,'(A,F10.4)'   )"Reynolds Number         : ",Re
		write(6,'(A,F10.2)'   )"Mach Number             : ",Mac
		write(6,'(A,F10.8)'   )"Timestep                : ",dt
		write(6,'(A,I10)'     )"Freq of stat calculation: ",istat
		write(6,'(A,3(F10.4,1X))')"Peclet in x-direction: ",PexVal,PeyVal,PezVal
		write(6,'(A)')"==================================================================================="
	endif
end subroutine print_param

subroutine initial_solnV()
	implicit none
	do k=bs(3),es(3)
		do j=bs(2),es(2)
			do i=bs(1),es(1)
				solV(i,j,k,1,1)		=	var(i,j,k,1)
				solV(i,j,k,2:5,1)	=	var(i,j,k,1) * var(i,j,k,2:5)
	
				mu(i,j,k)					= var(i,j,k,6)**0.7d0
				kt(i,j,k)					= var(i,j,k,6)**0.7d0
			enddo
		enddo
	enddo
endsubroutine initial_solnV

subroutine find_primary_variable()
	implicit none
	do k=bn(3),en(3)
		do j=bn(2),en(2)
			do i=bn(1),en(1)
				var(i,j,k,1  )	= solV(i,j,k,1,1)
				var(i,j,k,2:5)	= solV(i,j,k,2:5,1) / solV(i,j,k,1,1)
				var(i,j,k,6)		= var(i,j,k,5) - ggM*haf*sum(var(i,j,k,2:4)**two)
				var(i,j,k,7)		= var(i,j,k,1)*var(i,j,k,6) / (gama*Mac**two)
			enddo
		enddo
	enddo
endsubroutine find_primary_variable

end module mod_comdata