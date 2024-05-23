#
# Makefile
#

## Intel fortran @ p-srv1 ###
##    NOTE: you need to add the following path on the terminal for execution:
##    export LD_LIBRARY_PATH=/usr/local/pken/netcdf-fortran-4.5.3-ifort/build/lib:$LD_LIBRARY_PATH
#FC = ifort
#FFLAGS = -warn all -fpp -std #-traceback \
#         -check bounds -check uninit -fpe0 # These options may cause slow down
#FFLAGS += -O3 -xHOST -r8 -no-prec-div --shared-intel -mcmodel=large
#INC =
#LIB = -mkl
#OPENMP = #-openmp
##NF_DIR=/usr/local/pken/netcdf-fortran-4.5.3-ifort/build
#NF_DIR=/usr/local/pken/netcdf-fortran-4.5.3-ifort/build
#INC += -I/$(NF_DIR)/include
#LIB += -L/$(NF_DIR)/lib -lnetcdff

### GNU fortran ###
#FC = gfortran
#FFLAGS = -Wall -Wextra -pedantic -fbacktrace \
#         -fbounds-check -Wuninitialized -ffpe-trap=invalid,zero,overflow 
#FFLAGS += -mcmodel=medium -m64 -march=native -mtune=native -O3 -ffast-math
##FFTW_DIR=`spack location -i fftw`            # For spack
##NF_DIR=`spack location -i netcdf-fortran`    # For spack
##FFTW_DIR=/usr/                               # For apt
##NF_DIR=$(FFTW_DIR)                           # For apt
#FFTW_DIR=$(HOME)/mylib/Miniconda3/envs/pken2/ # For conda
#NF_DIR=$(FFTW_DIR)                            # For conda
#INC = -I/$(FFTW_DIR)/include
#LIB = -L/$(FFTW_DIR)/lib -lfftw3 -lm
#INC += -I/$(NF_DIR)/include
#LIB += -L/$(NF_DIR)/lib -lnetcdff
#OPENMP = #-fopenmp
FC = gfortran
FFLAGS = -Wall -Wextra -pedantic -fbacktrace \
         -fbounds-check -Wuninitialized -ffpe-trap=invalid,zero,overflow 
FFLAGS += -mcmodel=medium -m64 -march=native -mtune=native -O3 -ffast-math
INC = -I/usr/include
LIB = -L/usr/lib -lfftw3 -lm -lnetcdff

DIR = ./src/

### create hweq2d.exe ###
hweq2d:	$(DIR)/parameters.f90\
	$(DIR)/clock.f90     \
	$(DIR)/fileio.f90    \
	$(DIR)/fft_fftw.f90  \
	$(DIR)/geometry.f90  \
	$(DIR)/shearflow.f90 \
	$(DIR)/hweq2d.f90    \
	$(DIR)/main.f90

	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/parameters.f90
	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/clock.f90
	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/fileio.f90 $(INC)
	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/fft_fftw.f90 $(INC)
	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/geometry.f90
	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/shearflow.f90
	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/hweq2d.f90
	$(FC) $(FFLAGS) $(OPENMP) -c $(DIR)/main.f90

	$(FC) $(FFLAGS) $(OPENMP) *.o -o hweq2d.exe $(LIB) 
	rm -f *.o *.mod *_genmod.f90


### clean up files ###
clean:
	rm -f *.o *.mod *_genmod.f90 *.exe plot_*.gn ./png/* sub.q.o* sub.q.e* 
	find ./data/ -type f -name *.dat | xargs rm -f
	find ./data/ -type f -name *.nc  | xargs rm -f

git:
	rm -f *.o *.mod *_genmod.f90 *.exe plot_*.gn ./png/* sub.q.o* sub.q.e* 
	find ./data/ -type f -name *.dat | xargs rm -f
	find ./data/ -type f -name *.nc  | xargs rm -f
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace diag/*.ipynb
	jupyter nbconvert --to python diag/*.ipynb
	rm -rf diag/.ipynb_checkpoints/
