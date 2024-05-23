MODULE geometry
!-------------------------------------------------------------------------------
!
!  Module for geometry
!
!    "set_gometry"
!
!-------------------------------------------------------------------------------
  use parameters
  implicit none
  private

  public :: geometry_init, geometry_finalize, dx, dy, dkx, dky, xx, yy, kx, ky, ksq, fct_poisson

  real(kind=DP) :: dx, dy, dkx, dky
  real(kind=DP), dimension(:), allocatable :: xx
  real(kind=DP), dimension(:), allocatable :: yy
  real(kind=DP), dimension(:), allocatable :: kx
  real(kind=DP), dimension(:), allocatable :: ky
  real(kind=DP), dimension(:,:), allocatable :: ksq, fct_poisson

 CONTAINS

SUBROUTINE geometry_init

  integer :: ix, iy, mx, my

    allocate(xx(0:nx-1))
    allocate(yy(0:ny-1))
    allocate(kx(-nkx:nkx))
    allocate(ky(0:nky))
    allocate(ksq(-nkx:nkx,0:nky))
    allocate(fct_poisson(-nkx:nkx,0:nky))

    dx = 2._DP * lx / real(nx, kind=DP)
    dy = 2._DP * ly / real(ny, kind=DP)
    dkx = 2._DP * pi / (2._DP * lx)
    dky = 2._DP * pi / (2._DP * ly)
    do ix = 0, nx-1
      xx(ix) = -lx + dx * real(ix, kind=DP)
    end do
    do iy = 0, ny-1
      yy(iy) = -ly + dy * real(iy, kind=DP)
    end do
    do mx = -nkx, nkx
      kx(mx) = dkx * real(mx, kind=DP)
    end do
    do my = 0, nky
      ky(my) = dky * real(my, kind=DP)
    end do
    do my = 0, nky
      do mx = -nkx, nkx
        ksq(mx,my) = (kx(mx)**2 + ky(my)**2)
        if (mx == 0 .and. my == 0) then
          fct_poisson(mx,my) = 0._DP
        else
          fct_poisson(mx,my) = - 1._DP / ksq(mx,my)
        end if
      end do
    end do

END SUBROUTINE geometry_init

SUBROUTINE geometry_finalize
    deallocate(xx)
    deallocate(yy)
    deallocate(kx)
    deallocate(ky)
    deallocate(ksq)
    deallocate(fct_poisson)
END SUBROUTINE geometry_finalize


END MODULE geometry
