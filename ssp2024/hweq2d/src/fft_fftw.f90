MODULE fft
!-------------------------------------------------------------------------------
!
!    FFT module using fftw3
!
!-------------------------------------------------------------------------------
  use parameters
  implicit none
  include "fftw3.f"
  private

  public   fft_pre, fft_backward_xy, fft_forward_xy

  integer(kind=DP), save :: plan_backward_xy, plan_forward_xy

 CONTAINS

SUBROUTINE fft_pre
!--------------------------------------
!  Initialization of FFT

  complex(kind=DP), dimension(0:nx/2,0:ny-1) :: wwkk
  real(kind=DP), dimension(0:nx-1,0:ny-1) :: wwxy
!$  integer :: ierr, omp_get_max_threads

!$OMP parallel
!$  write(*,*) omp_get_max_threads()
!$OMP end parallel
!$  call dfftw_init_threads(ierr)
!$  call dfftw_plan_with_nthreads(omp_get_max_threads())
    call dfftw_plan_dft_c2r_2d(plan_backward_xy,  &
                                nx, ny,            &
                                wwkk, wwxy,        &
                                FFTW_ESTIMATE)
    call dfftw_plan_dft_r2c_2d(plan_forward_xy,  &
                                nx, ny,           &
                                wwxy, wwkk,       &
                                FFTW_ESTIMATE)
  

END SUBROUTINE fft_pre
  

SUBROUTINE fft_backward_xy (ww, wwxy)
!--------------------------------------
!  Execution of FFT
  
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(in) :: ww
  real(kind=DP), dimension(0:nx-1,0:ny-1), intent(out)    :: wwxy

  complex(kind=DP), dimension(0:nx/2,0:ny-1) :: wwkk
  integer :: mx, my
  
    wwkk(:,:) = (0._DP, 0._DP)

!$OMP parallel default(none) &
!$OMP private(mx,my) shared(wwkk,ww,nkx,nky,ny)
!$OMP do
    do my = 0, nky
      do mx = 0, nkx
        wwkk(mx,my) = ww(mx,my)
      end do
    end do
!$OMP end do nowait
!$OMP do
    do my = 1, nky
      do mx = -nkx, 0
        wwkk(-mx,ny-my) = conjg(ww(mx,my)) ! = ww(-mx,-my)
      end do
    end do
!$OMP end do nowait
!$OMP end parallel
  
    call dfftw_execute_dft_c2r(plan_backward_xy, wwkk, wwxy)


END SUBROUTINE fft_backward_xy


SUBROUTINE fft_forward_xy (wwxy, ww)
!--------------------------------------
!  Execution of FFT
  
  real(kind=DP), dimension(0:nx-1,0:ny-1), intent(in)      :: wwxy
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(out) :: ww

  complex(kind=DP), dimension(0:nx/2,0:ny-1) :: wwkk
  real(kind=DP) :: ceff_norm
  integer :: mx, my
  
    call dfftw_execute_dft_r2c(plan_forward_xy, wwxy, wwkk)

    ceff_norm = 1._DP / real(nx * ny, kind=DP)

!$OMP parallel default(none) &
!$OMP private(mx,my) shared(ceff_norm,wwkk,ww,nkx,nky,ny)
!$OMP do
    do my = 0, nky
      do mx = 0, nkx
        ww(mx,my) = ceff_norm * wwkk(mx,my)
      end do
    end do
!$OMP end do nowait
!$OMP do
    do my = 1, nky
      do mx = -nkx, -1
        ww(mx,my) = ceff_norm * conjg(wwkk(-mx,ny-my))
      end do
    end do
!$OMP end do nowait
!$OMP end parallel
    my = 0
      do mx = 1, nkx
        ww(-mx,-my) = conjg(ww(mx,my))
      end do
    ww(0,0) = (0._DP, 0._DP)


END SUBROUTINE fft_forward_xy


END MODULE fft
