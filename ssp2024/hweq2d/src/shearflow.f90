MODULE shearflow
!-------------------------------------------------------------------------------
!
!  Module for shearflow
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : kx, ky, xx, yy, dkx
  implicit none
  private

  public :: shearflow_init, shearflow_finalize, &
            shearflow_update_labframe,  &
            shearflow_dataremap_xy, shearflow_aliasing, &
            yy_labframe, kx_labframe, ksq_labframe, fct_poisson_labframe

  real(kind=DP), dimension(:,:), allocatable :: yy_labframe
  real(kind=DP), dimension(:,:), allocatable :: kx_labframe, ksq_labframe, &
                                                fct_poisson_labframe
  real(kind=DP), dimension(:), allocatable :: mx_aliasing_left, mx_aliasing_right

 CONTAINS


SUBROUTINE shearflow_init

  integer :: ix, iy, mx, my

    allocate(yy_labframe(0:nx-1,0:ny-1))
    allocate(kx_labframe(-nkx:nkx,0:nky))
    allocate(ksq_labframe(-nkx:nkx,0:nky))
    allocate(fct_poisson_labframe(-nkx:nkx,0:nky))
    allocate(mx_aliasing_left(0:nky))
    allocate(mx_aliasing_right(0:nky))

    do iy = 0, ny-1
      do ix = 0, nx-1
        yy_labframe(ix,iy) = yy(iy)
      end do
    end do
    do my = 0, nky
      do mx = -nkx, nkx
        kx_labframe(mx,my) = kx(mx)
      end do
    end do
    do my = 0, nky
      do mx = -nkx, nkx
        ksq_labframe(mx,my) = (kx_labframe(mx,my)**2 + ky(my)**2)
        if (mx == 0 .and. my == 0) then
          fct_poisson_labframe(mx,my) = 0._DP
        else
          fct_poisson_labframe(mx,my) = - 1._DP / ksq_labframe(mx,my)
        end if
      end do
    end do

    if (gammae /= 0.d0) then
      do my = 0, nky
        mx_aliasing_left(my) = -nkx+0.5*my-1
        mx_aliasing_right(my) = nkx-0.5*my+1
      end do
    else
      mx_aliasing_left(:) = -nkx-1
      mx_aliasing_right(:) = nkx+1
    end if

END SUBROUTINE shearflow_init


SUBROUTINE shearflow_update_labframe(dt_shear)

  real(kind=DP), intent(in) :: dt_shear
  integer :: ix, iy, mx, my

    do iy = 0, ny-1
      do ix = 0, nx-1
        yy_labframe(ix,iy) = yy_labframe(ix,iy) + xx(ix) * gammae * dt_shear
      end do
    end do
    do my = 0, nky
      do mx = -nkx, nkx
        kx_labframe(mx,my) = kx_labframe(mx,my) - ky(my) * gammae * dt_shear
      end do
    end do
    do my = 0, nky
      do mx = -nkx, nkx
        ksq_labframe(mx,my) = (kx_labframe(mx,my)**2 + ky(my)**2)
        if (mx == 0 .and. my == 0) then
          fct_poisson_labframe(mx,my) = 0._DP
        else
          fct_poisson_labframe(mx,my) = - 1._DP / ksq_labframe(mx,my)
        end if
      end do
    end do

    do my = 0, nky
      mx_aliasing_left(my) = mx_aliasing_left(my) + ky(my) * gammae * dt_shear * (lx/pi)
      mx_aliasing_right(my) = mx_aliasing_right(my) + ky(my) * gammae * dt_shear * (lx/pi)
    end do

END SUBROUTINE shearflow_update_labframe


SUBROUTINE shearflow_dataremap_xy(ff,phi,qq)

  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(inout) :: ff, qq
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(inout) :: phi
  integer :: ix, iy, mx, my

    do my = 0, nky
      if (my == 0) then
        ff(-nkx:nkx,my,:) = ff(-nkx:nkx,my,:)
        phi(-nkx:nkx,my) = phi(-nkx:nkx,my)
        qq(-nkx:nkx,my,:) = qq(-nkx:nkx,my,:)
      else if (my <= 2*nkx) then
        ff(-nkx:nkx-my,my,:) = ff(-nkx+my:nkx,my,:)
        ff(nkx-my:nkx,my,:) = (0._DP, 0._DP)
        phi(-nkx:nkx-my,my) = phi(-nkx+my:nkx,my)
        phi(nkx-my:nkx,my) = (0._DP, 0._DP)
        qq(-nkx:nkx-my,my,:) = qq(-nkx+my:nkx,my,:)
        qq(nkx-my:nkx,my,:) = (0._DP, 0._DP)
      else
        ff(:,my,:) = (0._DP, 0._DP)
        phi(:,my) = (0._DP, 0._DP)
        qq(:,my,:) = (0._DP, 0._DP)
      end if
    end do

    do iy = 0, ny-1
      do ix = 0, nx-1
        yy_labframe(ix,iy) = yy_labframe(ix,iy) - xx(ix) * ly / lx
      end do
    end do
    do my = 0, nky
      do mx = -nkx, nkx
        kx_labframe(mx,my) = kx_labframe(mx,my) + ky(my) * ly / lx
      end do
    end do
    do my = 0, nky
      do mx = -nkx, nkx
        ksq_labframe(mx,my) = (kx_labframe(mx,my)**2 + ky(my)**2)
        if (mx == 0 .and. my == 0) then
          fct_poisson_labframe(mx,my) = 0._DP
        else
          fct_poisson_labframe(mx,my) = - 1._DP / ksq_labframe(mx,my)
        end if
      end do
    end do

    do my = 0, nky
      mx_aliasing_left(my) = mx_aliasing_left(my) - ky(my) * ly / lx * (lx/pi)
      mx_aliasing_right(my) = mx_aliasing_right(my) - ky(my) * ly / lx * (lx/pi)
    end do

END SUBROUTINE shearflow_dataremap_xy


SUBROUTINE shearflow_aliasing(ff,phi,qq)

  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(inout) :: ff,qq
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(inout) :: phi
  integer :: my, mxl(0:nky), mxr(0:nky)

    !kxmax = dkx * nkx
    !do my = 0, nky
    !  do mx = -nkx, nkx
    !    if (abs(kx_labframe(mx,my)) > kxmax) then
    !      ff(mx,my,:) = (0._DP, 0._DP)
    !      phi(mx,my) = (0._DP, 0._DP)
    !      qq(mx,my,:) = (0._DP, 0._DP)
    !    end if
    !  end do
    !end do

    mxl(:) = int(mx_aliasing_left(:)) - 1
    mxr(:) = int(mx_aliasing_right(:)) + 1
    do my = 0, nky
      !write(*,*) mxl(my), mx_aliasing_left(my), mxr(my), mx_aliasing_right(my)
      if (my <= nkx) then
        if (-nkx <= mxl(my)) then
          ff(-nkx:mxl(my),my,:) = (0._DP, 0._DP)
          phi(-nkx:mxl(my),my) = (0._DP, 0._DP)
          qq(-nkx:mxl(my),my,:) = (0._DP, 0._DP)
        end if
        if (mxr(my) <= nkx) then
          ff(mxr(my):nkx,my,:) = (0._DP, 0._DP)
          phi(mxr(my):nkx,my) = (0._DP, 0._DP)
          qq(mxr(my):nkx,my,:) = (0._DP, 0._DP)
        end if
      else
          ff(:,my,:) = (0._DP, 0._DP)
          phi(:,my) = (0._DP, 0._DP)
          qq(:,my,:) = (0._DP, 0._DP)
      end if
    end do

END SUBROUTINE shearflow_aliasing

SUBROUTINE shearflow_finalize
    deallocate(yy_labframe)
    deallocate(kx_labframe)
    deallocate(ksq_labframe)
    deallocate(fct_poisson_labframe)
    deallocate(mx_aliasing_left)
    deallocate(mx_aliasing_right)
END SUBROUTINE shearflow_finalize


END MODULE shearflow
