MODULE parameters
!-------------------------------------------------------------------------------
!
!    Set parameters
!
!-------------------------------------------------------------------------------
  implicit none

!- constants for fortran -!
  integer, parameter :: DP = selected_real_kind(14)
  real(kind=DP), parameter :: pi = 3.141592653589793_DP
  real(kind=DP), parameter :: eps = 0.000000001_DP
  complex(kind=DP), parameter :: ci = (0._DP, 1._DP)

!- constants for numerical calculation -!
  integer       :: nx          ! Grid number in x direction
  integer       :: ny          ! Grid number in y direction
  integer       :: nkx         ! Mode number in kx(+-nkx)
  integer       :: nky         ! Mode number in ky(+-nky)

!- constants for numerical calculation -!
  integer       :: litime      ! Iteration limit of time
  real(kind=DP) :: elt_limit   ! Elapese time limit [sec]
  real(kind=DP) :: time_limit  ! Simulation time limit
  real(kind=DP) :: lx          ! Box size -lx <= xx < lx
  real(kind=DP) :: ly          ! Box size -ly <= yy < ly
  real(kind=DP) :: dt_max      ! Maximum time-step size
  real(kind=DP) :: dt_out      ! Time step for output
  real(kind=DP) :: dt          ! Time-step size

!- physical parameters -!
  real(kind=DP) :: ca          ! Adiabaticity parameter
  real(kind=DP) :: nu          ! Viscosity
  real(kind=DP) :: eta         ! Density gradient
  real(kind=DP) :: gammae      ! ExB shearing rate
  real(kind=DP) :: init_ampl   ! Initial amplitude of fluctuations

!- flags for computation -!
  character(len=9) :: flag_adiabaticity != "constant" ! C*(phi-n) where kz=const.
                                        != "kysquare" ! C*ky**2*(phi-n) where kz~ky

  character(len=9) :: flag_calctype     != "linear"    ! Linear
                                        != "nonlinear" ! Nonlinear

  character(len=9) :: flag_datatype     != "ascii"  ! ASCII data output
                                        != "binary" ! BINARY data output

 CONTAINS

SUBROUTINE parameters_set
  namelist/numer/ nx, ny, litime, elt_limit, time_limit, &
                  lx, ly, dt_max, dt_out, dt
  namelist/physp/ ca, nu, eta, gammae, init_ampl
  namelist/flags/ flag_adiabaticity, flag_calctype, flag_datatype

!  character(256) :: env_string
!    call getenv ('fu05', env_string)
!    open(10,file=env_string)
    open(10,file="./param.namelist",status="old",action="read")
      read(10,nml=numer)
      read(10,nml=physp)
      read(10,nml=flags)
    close(10)
    nkx = int((nx-2)/3)
    nky = int((ny-2)/3)
    write(*,*) "# nx = ", nx
    write(*,*) "# ny = ", ny
    write(*,*) "# lx = ", lx
    write(*,*) "# ly = ", ly
    write(*,*) "# ca = ", ca
    write(*,*) "# nu = ", nu
    write(*,*) "# eta = ", eta
    write(*,*) "# gammae = ", gammae

END SUBROUTINE parameters_set

END MODULE parameters
