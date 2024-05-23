PROGRAM main
  use parameters
  use clock, only : clock_init
  use hweq2d, only : hweq2d_simulation
    call parameters_set
    call clock_init
    call hweq2d_simulation
END PROGRAM main
