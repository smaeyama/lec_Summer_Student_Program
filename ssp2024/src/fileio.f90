MODULE fileio
!------------------------------------------------------------------------------!
!
!  Set input/output files
!
!------------------------------------------------------------------------------!
  use parameters
  use netcdf
  implicit none
  private

  public  opxy, fileio_phiinxy_nc, fileio_phiinkxky_nc

  integer, parameter :: opxy = 100

 CONTAINS

SUBROUTINE fileio_phiinxy_nc(path,time,xx,yy,dns,omg,phi)

  character(*), intent(in) :: path
  real(kind=DP), intent(in) :: time
  real(kind=DP), intent(in) :: xx(0:nx-1)
  real(kind=DP), intent(in) :: yy(0:ny-1)
  real(kind=DP), dimension(0:nx-1,0:ny-1), intent(in) :: dns, omg, phi

  integer(kind=4) :: ncid, dimids(1:3), ierr_nf90
  integer(kind=4) :: varid_xx, varid_yy, varid_tt, varid_dns, varid_omg, varid_phi
  integer(kind=4) :: start_var(1:3), count_var(1:3)

    != Create file
    ierr_nf90=nf90_create(path=path, cmode=NF90_NETCDF4, ncid=ncid)
    call check_nf90err(ierr_nf90, "nf90_create")

    != Define dimensions in file
    ierr_nf90=nf90_def_dim(ncid=ncid, name="x", len=int(nx,kind=4), dimid=dimids(1))
    ierr_nf90=nf90_def_dim(ncid=ncid, name="y", len=int(ny,kind=4), dimid=dimids(2))
    ierr_nf90=nf90_def_dim(ncid=ncid, name="t", len=int(1, kind=4), dimid=dimids(3))
    call check_nf90err(ierr_nf90, "nf90_def_dim")
  
    != Define variables in file
    ierr_nf90=nf90_def_var(ncid=ncid, name="x",   xtype=NF90_DOUBLE, dimids=dimids(1),   varid=varid_xx)
    ierr_nf90=nf90_def_var(ncid=ncid, name="y",   xtype=NF90_DOUBLE, dimids=dimids(2),   varid=varid_yy)
    ierr_nf90=nf90_def_var(ncid=ncid, name="t",   xtype=NF90_DOUBLE, dimids=dimids(3),   varid=varid_tt)
    ierr_nf90=nf90_def_var(ncid=ncid, name="dns", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varid_dns)
    ierr_nf90=nf90_def_var(ncid=ncid, name="omg", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varid_omg)
    ierr_nf90=nf90_def_var(ncid=ncid, name="phi", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varid_phi)
    call check_nf90err(ierr_nf90, "nf90_def_var")

    != End of definition of file
    ierr_nf90=nf90_enddef(ncid=ncid)
    call check_nf90err(ierr_nf90, "nf90_enddef")

    != Write variables: static coordinates x and y
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_xx, values=xx(0:nx-1))
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_yy, values=yy(0:ny-1))
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_tt, values=(/time/))
    call check_nf90err(ierr_nf90, "nf90_putvar")

    !%%% Time step loop %%%
    start_var(:) = (/ 1,1,1 /) 
    count_var(:) = (/ nx,ny,1 /)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_dns, values=dns, start=start_var, count=count_var)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_omg, values=omg, start=start_var, count=count_var)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_phi, values=phi, start=start_var, count=count_var)
    call check_nf90err(ierr_nf90, "nf90_putvar")

    != Close file
    ierr_nf90=nf90_close(ncid)
    call check_nf90err(ierr_nf90, "nf90_close")

END SUBROUTINE fileio_phiinxy_nc


SUBROUTINE fileio_phiinkxky_nc(path,time,kx,ky,dns,omg,phi)

  character(*), intent(in) :: path
  real(kind=DP), intent(in) :: time
  real(kind=DP), intent(in) :: kx(-nkx:nkx)
  real(kind=DP), intent(in) :: ky(0:nky)
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(in) :: dns, omg, phi

  integer(kind=4) :: ncid, dimids(1:3), ierr_nf90
  integer(kind=4) :: varid_kx, varid_ky, varid_tt, varids(1:6)
  integer(kind=4) :: start_var(1:3), count_var(1:3)

    != Create file
    ierr_nf90=nf90_create(path=path, cmode=NF90_NETCDF4, ncid=ncid)
    call check_nf90err(ierr_nf90, "nf90_create")

    != Define dimensions in file
    ierr_nf90=nf90_def_dim(ncid=ncid, name="kx", len=int(2*nkx+1,kind=4), dimid=dimids(1))
    ierr_nf90=nf90_def_dim(ncid=ncid, name="ky", len=int(nky+1,kind=4), dimid=dimids(2))
    ierr_nf90=nf90_def_dim(ncid=ncid, name="t", len=int(1, kind=4), dimid=dimids(3))
    call check_nf90err(ierr_nf90, "nf90_def_dim")
  
    != Define variables in file
    ierr_nf90=nf90_def_var(ncid=ncid, name="kx",  xtype=NF90_DOUBLE, dimids=dimids(1),   varid=varid_kx)
    ierr_nf90=nf90_def_var(ncid=ncid, name="ky",  xtype=NF90_DOUBLE, dimids=dimids(2),   varid=varid_ky)
    ierr_nf90=nf90_def_var(ncid=ncid, name="t",   xtype=NF90_DOUBLE, dimids=dimids(3),   varid=varid_tt)
    ierr_nf90=nf90_def_var(ncid=ncid, name="redns", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varids(1))
    ierr_nf90=nf90_def_var(ncid=ncid, name="imdns", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varids(2))
    ierr_nf90=nf90_def_var(ncid=ncid, name="reomg", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varids(3))
    ierr_nf90=nf90_def_var(ncid=ncid, name="imomg", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varids(4))
    ierr_nf90=nf90_def_var(ncid=ncid, name="rephi", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varids(5))
    ierr_nf90=nf90_def_var(ncid=ncid, name="imphi", xtype=NF90_DOUBLE, dimids=dimids(1:3), varid=varids(6))
    call check_nf90err(ierr_nf90, "nf90_def_var")

    != End of definition of file
    ierr_nf90=nf90_enddef(ncid=ncid)
    call check_nf90err(ierr_nf90, "nf90_enddef")

    != Write variables: static coordinates x and y
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_kx, values=kx(-nkx:nkx))
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_ky, values=ky(0:nky))
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varid_tt, values=(/time/))
    call check_nf90err(ierr_nf90, "nf90_putvar")

    !%%% Time step loop %%%
    start_var(:) = (/ 1,1,1 /) 
    count_var(:) = (/ 2*nkx+1,nky+1,1 /)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varids(1), values=dble(dns),  start=start_var, count=count_var)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varids(2), values=aimag(dns), start=start_var, count=count_var)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varids(3), values=dble(omg),  start=start_var, count=count_var)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varids(4), values=aimag(omg), start=start_var, count=count_var)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varids(5), values=dble(phi),  start=start_var, count=count_var)
    ierr_nf90=nf90_put_var(ncid=ncid, varid=varids(6), values=aimag(phi), start=start_var, count=count_var)
    call check_nf90err(ierr_nf90, "nf90_putvar")

    != Close file
    ierr_nf90=nf90_close(ncid)
    call check_nf90err(ierr_nf90, "nf90_close")

END SUBROUTINE fileio_phiinkxky_nc


SUBROUTINE check_nf90err(werr, comment)
!--------------------------------------
!  Check error message of nf90
  integer(kind=4), intent(in) :: werr
  character(len=*), intent(in) :: comment

  if(werr /= nf90_noerr) then
    write(*,*) comment//" "//trim(nf90_strerror(werr))
    stop
  end if

END SUBROUTINE check_nf90err


END MODULE fileio
