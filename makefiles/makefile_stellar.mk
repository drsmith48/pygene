
DEBUG = yes

GENE_DIR= $(HOME)/gene
GENECODE_DIR = $(GENE_DIR)/genecode

PETSC_DIR = $(GENE_DIR)/packages/petsc-3.17.1
SLEPC_DIR = $(GENE_DIR)/packages/slepc-3.17.1

ifeq ($(DEBUG), yes)
	PETSC_ARCH = impi_2021.3.1_debug
	PETSC_OPT_FLAGS =
	EXEC = gene_$(MACHINE)
	WITH_DEBUGGING = 1
else
	PETSC_ARCH = impi_2021.3.1_opt
	PETSC_OPT_FLAGS = --COPTFLAGS=-Ofast --CXXOPTFLAGS=-Ofast --FOPTFLAGS=-Ofast
	EXEC = gene_$(MACHINE)_opt
	WITH_DEBUGGING = 0
endif

.PHONY: petsc slepc gene

gene: slepc
	#rm -f $(GENECODE_DIR)/bin/*.mk
	cd $(GENECODE_DIR) && $(MAKE) PETSC_ARCH=$(PETSC_ARCH) DEBUG=$(DEBUG) EXEC=$(EXEC) clean
	cd $(GENECODE_DIR) && $(MAKE) PETSC_ARCH=$(PETSC_ARCH) DEBUG=$(DEBUG) EXEC=$(EXEC) -j16
	cp -f $(GENECODE_DIR)/bin/stellar.mk $(GENECODE_DIR)/makefiles/stellar
	@echo "For minimum testing:"
	@echo "salloc -n8 -t10  # launch interactive session"
	@echo "cd $(GENECODE_DIR)/testsuite && ./testsuite -e 2  # run GENE test suite"

slepc: petsc
	cd $(SLEPC_DIR) && ./configure --with-clean
	$(MAKE) SLEPC_DIR=$(SLEPC_DIR) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C $(SLEPC_DIR) clean
	$(MAKE) SLEPC_DIR=$(SLEPC_DIR) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C $(SLEPC_DIR) -j16

petsc:
	cd $(PETSC_DIR) && ./configure \
		PETSC_ARCH=$(PETSC_ARCH) \
		--with-clean \
		--with-cc=mpiicc \
		--with-cxx=mpiicpc \
		--with-fc=mpiifort \
		--with-scalar-type=complex \
		--with-debugging=$(WITH_DEBUGGING) \
		--with-blaslapack-dir=$(MKLROOT) \
		--with-scalapack \
		--with-hdf5-dir=$(HDF5_ROOT) \
		--with-fftw-dir=$(FFTW_PATH) \
		$(PETSC_OPT_FLAGS)
	$(MAKE) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C $(PETSC_DIR) clean
	$(MAKE) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C $(PETSC_DIR) -j16 all
