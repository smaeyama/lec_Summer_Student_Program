MODULE hweq2d
  use parameters
  implicit none
  private

  public  hweq2d_simulation

  !complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), save :: qq
  complex(kind=DP), dimension(:,:,:), allocatable, save :: qq

 CONTAINS

SUBROUTINE hweq2d_simulation
!-------------------------------------------------------------------------------
!
!    Solve 2D Hasegawa-Wakatani eqs.,
!          ddns/dt + {phi,dns} + dphi/dy = - ca * (dns-phi) - nu * \nabla^4 dns
!          domg/dt + {phi,omg} = - ca * (dns-phi) - nu * \nabla^4 omg
!          omg = \nabla^2 phi
!
!    Doubly periodic boundary in x and y
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : geometry_init, geometry_finalize
  use fft, only : fft_pre
  use clock, only : clock_sta, clock_end, elt
  use shearflow, only : shearflow_init, shearflow_dataremap_xy, shearflow_finalize
  implicit none
  real(kind=DP) :: time = 0._DP
  !real(kind=DP), dimension(0:nx-1,0:ny-1) :: dns, omg, phi
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1) :: ff ! ff(:,:,0) = dns
                                                        ! ff(:,:,1) = omg
  complex(kind=DP), dimension(-nkx:nkx,0:nky) :: phi
  real(kind=DP) :: time_out, time_remap, dt_remap
  integer :: itime

                                                            call clock_sta(1)
                                                            call clock_sta(2)
!- initialize -!
    itime = 0
    allocate(qq(-nkx:nkx,0:nky,0:1))
    call geometry_init
    call shearflow_init
    call fft_pre
    call initial_condition(time, ff, phi)
!--------------!
                                                            call clock_end(2)
                                                            call clock_sta(4)
    write(*,*) 'time = ', time
    call out_phiinxy(time, ff, phi)
    time_out = dt_out - eps
    if (gammae /= 0.d0) then
      dt_remap = ly / lx / gammae
      time_remap = 0.5d0 * dt_remap - eps
    end if
                                                            call clock_end(4)

!= time-evolution =!
    do itime = 1, litime

                                                            call clock_sta(3)
    call rkg4(time, ff, phi)
    time = time + dt
                                                            call clock_end(3)

                                                            call clock_sta(4)
!- output variables -!
    if (time > time_out) then
      write(*,*) 'time = ', time
      call out_phiinxy(time, ff, phi)
      time_out = time_out + dt_out
    end if
!--------------------!
    if (gammae /= 0.d0) then
      if (time > time_remap) then
        call shearflow_dataremap_xy(ff,phi,qq)
        time_remap = time_remap + dt_remap
      end if
    end if
                                                            call clock_end(4)

    if (elt(2) + elt(3) + elt(4) > elt_limit) then
      write(*, *) 'Elapsed time limit is close.'
      exit
    end if
    if (time > time_limit) then
      write(*, *) 'Simulation time limit is close.'
      exit
    end if

    end do
!==================!

                                                            call clock_end(1)
    call shearflow_finalize
    call geometry_finalize
    deallocate(qq)

    write(*, *) '### Elapsed time ###'
    write(*, *) '#  Total = ', elt(1)
    write(*, *) '#   Init = ', elt(2)
    write(*, *) '#    RKG = ', elt(3)
    write(*, *) '# Output = ', elt(4)
    write(*, *) '#  Other = ', elt(1) - (elt(2) + elt(3) + elt(4))
    write(*, *) 'End program.'


END SUBROUTINE hweq2d_simulation


SUBROUTINE initial_condition(time, ff, phi)
!-------------------------------------------------------------------------------
!
!    Set initial condition
!
!-------------------------------------------------------------------------------
  use parameters
  use shearflow, only : ksq_labframe
  implicit none
  real(kind=DP), intent(out) :: time
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(out) :: ff
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(out) :: phi

  real(kind=DP), dimension(-nkx:nkx,0:nky) :: rnum
  integer :: mx, my, iv

    time = 0._DP
    ff(:,:,:) = (0._DP, 0._DP)
    phi(:,:) = (0._DP, 0._DP)

    iv = 0
      call random_number(rnum)
      do my = 0, nky
        do mx = -nkx, nkx
          ff(mx,my,iv) = init_ampl / (1._DP + ksq_labframe(mx,my)) * exp(ci * 2._DP * pi * rnum(mx,my))
        end do
      end do
    ff(:,:,1) = - ff(:,:,0) * ksq_labframe(:,:)
    call field_solver(ff, phi)

  !- reality condition -
    do iv = 0, 1
      my = 0
        do mx = 1, nkx
          ff(-mx,-my,iv) = conjg(ff(mx,my,iv))
        end do
      ff(0,0,iv) = (0._DP, 0._DP)
    end do
    my = 0
      do mx = 1, nkx
        phi(-mx,-my) = conjg(phi(mx,my))
      end do
    phi(0,0) = (0._DP, 0._DP)

END SUBROUTINE initial_condition


SUBROUTINE out_phiinxy(time, ff, phi)
!-------------------------------------------------------------------------------
!
!    Output variables
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : xx, yy, kx, ky
  use fileio, only : opxy, fileio_phiinxy_nc, fileio_phiinkxky_nc
  use fft, only : fft_backward_xy
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(in) :: ff
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(in) :: phi

  complex(kind=DP), dimension(-nkx:nkx,0:nky) :: wc2
  real(kind=DP), dimension(0:nx-1,0:ny-1) :: dns_xy, omg_xy, phi_xy
  character(len=8) :: ctime
  integer :: ix, iy, mx, my

    write(ctime,'(i8.8)') nint(time / dt_out)

    !call fft_backward_xy(ff(:,:,0), dns_xy)
    !call fft_backward_xy(ff(:,:,1), omg_xy)
    !call fft_backward_xy(phi, phi_xy)
    do my = 0, nky
      do mx = -nkx, nkx
        wc2(mx,my) = exp(-ci*kx(mx)*lx) * ff(mx,my,0)
      end do
    end do
    call fft_backward_xy(wc2, dns_xy)
    do my = 0, nky
      do mx = -nkx, nkx
        wc2(mx,my) = exp(-ci*kx(mx)*lx) * ff(mx,my,1)
      end do
    end do
    call fft_backward_xy(wc2, omg_xy)
    do my = 0, nky
      do mx = -nkx, nkx
        wc2(mx,my) = exp(-ci*kx(mx)*lx) * phi(mx,my)
      end do
    end do
    call fft_backward_xy(wc2, phi_xy)

    if (trim(flag_datatype) == "ascii") then ! ASCII data output

      open(opxy, file="./data/phiinxy_t"//ctime//".dat")
        write(opxy, *) "# Time = ", time
        write(opxy, "(99a17)") "#              xx", "yy", "dns", "omg", "phi"
        do iy = 0, ny-1
          do ix = 0, nx-1
            write(opxy, "(99e17.7e3)") xx(ix), yy(iy), dns_xy(ix,iy), omg_xy(ix,iy), phi_xy(ix,iy)
          end do
          write(opxy, *)
        end do
      close(opxy)

    else if (trim(flag_datatype) == "binary") then ! BINARY data output

      open(unit=opxy, file="./data/phiinxy_t"//ctime//".dat", status="replace", &
                       action="write", form="unformatted", access="stream", convert="LITTLE_ENDIAN")
        do iy = 0, ny-1
          do ix = 0, nx-1
            write(opxy) xx(ix), yy(iy), dns_xy(ix,iy), omg_xy(ix,iy), phi_xy(ix,iy)
          end do
        end do
      close(opxy)

    else if (trim(flag_datatype) == "netcdf") then ! BINARY data output

      call fileio_phiinxy_nc("./data/phiinxy_t"//ctime//".nc",time,xx,yy,dns_xy,omg_xy,phi_xy)
      call fileio_phiinkxky_nc("./data/phiinkxky_t"//ctime//".nc",time,kx,ky,ff(:,:,0),ff(:,:,1),phi)

    else

      write(*,*) "Wrong flag_datatype: ", flag_datatype
      stop

    end if

END SUBROUTINE out_phiinxy


SUBROUTINE time_diff(time, ff, phi, dfdt)
!-------------------------------------------------------------------------------
!
!    Calculate time-differential term
!
!          ddns/dt + {phi,dns} + dphi/dy = - ca * (dns-phi) - nu * \nabla^4 dns
!          domg/dt + {phi,omg} = - ca * (dns-phi) - nu * \nabla^4 omg
!          omg = \nabla^2 phi
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : ky
  use shearflow, only : ksq_labframe
  use fft, only : fft_backward_xy, fft_forward_xy
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(in) :: ff !=dns,omg
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(in) :: phi
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(out) :: dfdt

  complex(kind=DP), dimension(-nkx:nkx,0:nky) :: pb_phidns, pb_phiomg
  real(kind=DP), dimension(0:nky) :: wca
  integer :: mx, my

    if (trim(flag_calctype) == "linear") then ! Linear
      pb_phidns(:,:) = (0._DP, 0._DP)
      pb_phiomg(:,:) = (0._DP, 0._DP)
    else if (trim(flag_calctype) == "nonlinear") then ! Nonlinear
      call calc_pb(phi, ff(:,:,0), pb_phidns) ! = {phi,dns}
      call calc_pb(phi, ff(:,:,1), pb_phiomg) ! = {phi,omg}
    else
      write(*,*) "Wrong flag_calctype: ", flag_calctype
      stop
    end if

    if (trim(flag_adiabaticity) == "constant") then ! C*(phi-n) where kz=const.
!$OMP parallel do default(none) private(mx,my) shared(dfdt,pb_phidns,pb_phiomg,ky,ksq_labframe,ff,phi,eta,ca,nu,nkx,nky)
      do my = 0, nky
        do mx = -nkx, nkx
          dfdt(mx,my,0) = - pb_phidns(mx,my)                &
                          - ci * eta * ky(my) * phi(mx,my)  &
                          - ca * (ff(mx,my,0) - phi(mx,my)) &
                          - nu * ksq_labframe(mx,my)**2 * ff(mx,my,0)
          dfdt(mx,my,1) = - pb_phiomg(mx,my)                &
                          - ca * (ff(mx,my,0) - phi(mx,my)) &
                          - nu * ksq_labframe(mx,my)**2 * ff(mx,my,1)
        end do
      end do
    else if (trim(flag_adiabaticity) == "kysquare") then ! C*ky**2*(phi-n) where kz~ky
!$OMP parallel do default(none) private(mx,my) shared(dfdt,pb_phidns,pb_phiomg,ky,ksq_labframe,ff,phi,eta,ca,nu,nkx,nky)
      do my = 0, nky
        do mx = -nkx, nkx
          dfdt(mx,my,0) = - pb_phidns(mx,my)                &
                          - ci * eta * ky(my) * phi(mx,my)  &
                          - ca * ky(my)**2 * (ff(mx,my,0) - phi(mx,my)) &
                          - nu * ksq_labframe(mx,my)**2 * ff(mx,my,0)
          dfdt(mx,my,1) = - pb_phiomg(mx,my)                &
                          - ca * ky(my)**2 * (ff(mx,my,0) - phi(mx,my)) &
                          - nu * ksq_labframe(mx,my)**2 * ff(mx,my,1)
        end do
      end do
    else if (trim(flag_adiabaticity) == "modified") then ! C*(phi(ky/=0)-n(ky/=0))
      wca(:) = ca
      wca(0) = 0._DP
!$OMP parallel do default(none) private(mx,my) shared(dfdt,pb_phidns,pb_phiomg,ky,ksq_labframe,ff,phi,eta,ca,nu,nkx,nky,wca)
      do my = 0, nky
        do mx = -nkx, nkx
          dfdt(mx,my,0) = - pb_phidns(mx,my)                &
                          - ci * eta * ky(my) * phi(mx,my)  &
                          - wca(my) * (ff(mx,my,0) - phi(mx,my)) &
                          - nu * ksq_labframe(mx,my)**2 * ff(mx,my,0)
          dfdt(mx,my,1) = - pb_phiomg(mx,my)                &
                          - wca(my) * (ff(mx,my,0) - phi(mx,my)) &
                          - nu * ksq_labframe(mx,my)**2 * ff(mx,my,1)
        end do
      end do
    else
      write(*,*) "Wrong flag_adiabaticity: ", flag_adiabaticity
      stop
    end if

!%%% Modified Hasegawa-Wakatani for zonal flows %%%
!   my = 0
!     do mx = -nkx, nkx
!       dfdt(mx,my,0) = dfdt(mx,my,0) + ca * (ff(mx,my,0) - phi(mx,my))
!       dfdt(mx,my,1) = dfdt(mx,my,1) + ca * (ff(mx,my,0) - phi(mx,my))
!     end do
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                    !%%% For debug %%%
                                    !write(99998,"(99g17.7e3)") ff(0,3,:), phi(0,3), &
                                    !    dfdt(0,3,:), pb_phidns(0,3), pb_phiomg(0,3)
                                    !%%%%%%%%%%%%%%%%%

END SUBROUTINE time_diff
      

SUBROUTINE calc_pb(ff, gg, pb)
!-------------------------------------------------------------------------------
!
!    Calculate Poisson bracket
!
!        {f,g} = (df/dx)(dg/dy) - (df/dy)(dg/dx)
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : ky
  use shearflow, only : kx_labframe
  use fft, only : fft_backward_xy, fft_forward_xy
  implicit none

  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(in) :: ff, gg
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(out) :: pb

  complex(kind=DP), dimension(-nkx:nkx,0:nky) :: ikxf, ikyf, ikxg, ikyg
  real(kind=DP), dimension(0:nx-1,0:ny-1) :: dfdx, dfdy, dgdx, dgdy, wkxy
  integer :: mx, my, ix, iy

!$OMP parallel do default(none) private(mx,my) shared(ff,gg,kx_labframe,ky,ikxf,ikyf,ikxg,ikyg,nkx,nky)
    do my = 0, nky
      do mx = -nkx, nkx
        !ikxf(mx,my) = ci * kx(mx) * ff(mx,my)
        ikxf(mx,my) = ci * kx_labframe(mx,my) * ff(mx,my)
        ikyf(mx,my) = ci * ky(my) * ff(mx,my)
        !ikxg(mx,my) = ci * kx(mx) * gg(mx,my)
        ikxg(mx,my) = ci * kx_labframe(mx,my) * gg(mx,my)
        ikyg(mx,my) = ci * ky(my) * gg(mx,my)
      end do
    end do
    call fft_backward_xy(ikxf, dfdx)
    call fft_backward_xy(ikyf, dfdy)
    call fft_backward_xy(ikxg, dgdx)
    call fft_backward_xy(ikyg, dgdy)

!$OMP parallel do default(none) private(ix,iy) shared(wkxy,dfdx,dgdy,dfdy,dgdx,nx,ny)
    do iy = 0, ny-1
      do ix = 0, nx-1
        wkxy(ix,iy) = dfdx(ix,iy) * dgdy(ix,iy) - dfdy(ix,iy) * dgdx(ix,iy)
      end do
    end do
    call fft_forward_xy(wkxy, pb)

END SUBROUTINE calc_pb


SUBROUTINE field_solver(ff, phi)
!-------------------------------------------------------------------------------
!
!    Solve field eqs.,   omg = \nabla^2 phi
!
!-------------------------------------------------------------------------------
  use parameters
  use shearflow, only : fct_poisson_labframe
  implicit none
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(in) :: ff
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(out) :: phi

  integer :: mx, my

!$OMP parallel do default(none) private(mx,my) shared(ff,phi,fct_poisson_labframe,nkx,nky)
    do my = 0, nky
      do mx = -nkx, nkx
        phi(mx,my) = fct_poisson_labframe(mx,my) * ff(mx,my,1)
      end do
    end do

END SUBROUTINE field_solver


SUBROUTINE reality_condition(ff, phi, qq)
!-------------------------------------------------------------------------------
!
!    Condition of complex ff, phi so that ff_xy, phi_xy are real.
!
!-------------------------------------------------------------------------------
  use parameters
  implicit none
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(inout) :: ff, qq
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(inout) :: phi

  integer :: mx, my, iv

    do iv = 0, 1
      my = 0
        do mx = 1, nkx
          ff(-mx,-my,iv) = conjg(ff(mx,my,iv))
          qq(-mx,-my,iv) = conjg(qq(mx,my,iv))
        end do
    end do
    ff(0,0,:) = (0._DP, 0._DP)
    qq(0,0,:) = (0._DP, 0._DP)

    my = 0
      do mx = 1, nkx
        phi(-mx,-my) = conjg(phi(mx,my))
      end do
    phi(0,0) = (0._DP, 0._DP)


END SUBROUTINE reality_condition


SUBROUTINE rkg4(time, ff, phi)
!-------------------------------------------------------------------------------
!
!    Time evolution by Runge_Kutta_Gill
!
!-------------------------------------------------------------------------------
  use parameters
  use shearflow, only : shearflow_update_labframe, shearflow_aliasing
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), intent(inout) :: ff
  complex(kind=DP), dimension(-nkx:nkx,0:nky), intent(inout) :: phi

  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1) :: dfdt
  !complex(kind=DP), dimension(-nkx:nkx,0:nky,0:1), save :: qq
  complex(kind=DP) :: k, r, s
  integer :: mx, my, iv

!- initialize -!
    k = (0._DP, 0._DP)
    r = (0._DP, 0._DP)
    s = (0._DP, 0._DP)
    if (time < eps) then
      do my = 0, nky
        do mx = -nkx, nkx
          qq(mx,my,:) = (0._DP, 0._DP)
        end do
      end do
    end if
!--------------!

    call time_diff(time, ff, phi, dfdt)
!$OMP parallel do default(none) private(mx,my,k,r,s) shared(dt,dfdt,ff,qq,nkx,nky) collapse(2)
    do iv = 0, 1
      do my = 0, nky
        do mx = -nkx, nkx
          k = dt * dfdt(mx,my,iv)
          r = 0.5_DP * (k - 2._DP * qq(mx,my,iv))
          s = ff(mx,my,iv)
          ff(mx,my,iv) = s + r
          r = ff(mx,my,iv) - s ! care for round off
          qq(mx,my,iv) = qq(mx,my,iv) + 3._DP * r - 0.5_DP * k
        end do
      end do
    end do
    call field_solver(ff, phi)
    call reality_condition(ff, phi, qq)
    call shearflow_aliasing(ff, phi, qq)

    call shearflow_update_labframe(0.5_DP * dt)
    call time_diff(time + 0.5_DP * dt, ff, phi, dfdt)
!$OMP parallel do default(none) private(mx,my,k,r,s) shared(dt,dfdt,ff,qq,nkx,nky) collapse(2)
    do iv = 0, 1
      do my = 0, nky
        do mx = -nkx, nkx
          k = dt * dfdt(mx,my,iv)
          r = (1._DP - sqrt(0.5_DP)) * (k - qq(mx,my,iv))
          s = ff(mx,my,iv)
          ff(mx,my,iv) = s + r
          r = ff(mx,my,iv) - s ! care for round off
          qq(mx,my,iv) = qq(mx,my,iv) + 3._DP * r - (1._DP - sqrt(0.5_DP)) * k
        end do
      end do
    end do
    call field_solver(ff, phi)
    call reality_condition(ff, phi, qq)
    call shearflow_aliasing(ff, phi, qq)

    call time_diff(time + 0.5_DP * dt, ff, phi, dfdt)
!$OMP parallel do default(none) private(mx,my,k,r,s) shared(dt,dfdt,ff,qq,nkx,nky) collapse(2)
    do iv = 0, 1
      do my = 0, nky
        do mx = -nkx, nkx
          k = dt * dfdt(mx,my,iv)
          r = (1._DP + sqrt(0.5_DP)) * (k - qq(mx,my,iv))
          s = ff(mx,my,iv)
          ff(mx,my,iv) = s + r
          r = ff(mx,my,iv) - s ! care for round off
          qq(mx,my,iv) = qq(mx,my,iv) + 3._DP * r - (1._DP + sqrt(0.5_DP)) * k
        end do
      end do
    end do
    call field_solver(ff, phi)
    call reality_condition(ff, phi, qq)
    call shearflow_aliasing(ff, phi, qq)

    call shearflow_update_labframe(0.5_DP * dt)
    call time_diff(time + dt, ff, phi, dfdt)
!$OMP parallel do default(none) private(mx,my,k,r,s) shared(dt,dfdt,ff,qq,nkx,nky) collapse(2)
    do iv = 0, 1
      do my = 0, nky
        do mx = -nkx, nkx
          k = dt * dfdt(mx,my,iv)
          r = 1._DP/6._DP * (k - 2._DP * qq(mx,my,iv))
          s = ff(mx,my,iv)
          ff(mx,my,iv) = s + r
          r = ff(mx,my,iv) - s ! care for round off
          qq(mx,my,iv) = qq(mx,my,iv) + 3._DP * r - 0.5_DP * k
        end do
      end do
    end do
    call field_solver(ff, phi)
    call reality_condition(ff, phi, qq)
    call shearflow_aliasing(ff, phi, qq)


END SUBROUTINE rkg4


END MODULE hweq2d
