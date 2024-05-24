#
# Makefile
#



### Intel fortran ###
FC = ifx
FFLAGS = -warn all -fpp -std #-traceback \
         -check bounds -check uninit -fpe0 # These options may cause slow down
FFLAGS += -O3 -xHOST -r8 -mcmodel=large
INC = 
LIB = -qmkl
NF_DIR = /home/smaeyama/mylib/netcdf-fortran-4.6.1-ifx
INC += -I/$(NF_DIR)/include
LIB += -L/$(NF_DIR)/lib -lnetcdff
OPENMP = -qopenmp

### GNU fortran ###
#FC = gfortran
#FFLAGS = -Wall -Wextra -pedantic -fbacktrace \
#         -fbounds-check -Wuninitialized -ffpe-trap=invalid,zero,overflow 
#FFLAGS += -mcmodel=medium -m64 -march=native -mtune=native -O3 -ffast-math
#INC = -I/usr/include
#LIB = -L/usr/lib -lfftw3 -lm -lnetcdff -lfftw3_threads
# #OPENMP = -fopenmp
# #LIB += -lfftw3_threads

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
	find ./diag/data_netcdf/ -type f -name *.nc  | xargs rm -f
	find ./diag/png_*/ -type f -name *.png  | xargs rm -f
