CUDA_PATH     ?= /nfs/gce/software/custom/linux-ubuntu18.04-x86_64/cuda/10.2
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin

ifeq ($(OS_SIZE), 32)
	CUDA_LIB_PATH ?= $(CUDA_PATH)/lib
else
	CUDA_LIB_PATH ?= $(CUDA_PATH)/lib64
endif

NVCC ?= $(CUDA_BIN_PATH)/nvcc
CC := gcc

GENCODE_SM70  := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS := $(GENCODE_SM70)

LDFLAGS    := -lcudart -lrt -lcurand -lm
CFLAGS     := -O3 -lineinfo -Xptxas="-dlcm=ca -v"
SOURCE_DIR  = ./src
TEST_DIR    = ./test
CUH_EXA    := src/cicfs.cuh src/cicf.cuh src/cnrm2.cuh src/cdot.cuh \
              src/ccopy.cuh src/cmid.cuh src/cscal.cuh src/cgpstep.cuh \
              src/ccauchy.cuh src/cbreakpt.cuh src/cssyax.cuh src/caxpy.cuh \
		      src/csol.cuh src/cgpnorm.cuh src/cprsrch.cuh src/ctrqsol.cuh \
		      src/ctrpcg.cuh src/cspcg.cuh src/creorder.cuh src/ctron.cuh \
		      src/cdriver.cuh
CUH_TEST   := test/gputest_utilities.cuh test/gputest_icf.cuh \
			  test/gputest_icfs.cuh test/gputest_copy.cuh test/gputest_dot.cuh \
			  test/gputest_scal.cuh test/gputest_nrm2.cuh test/gputest_mid.cuh \
			  test/gputest_gpstep.cuh test/gputest_axpy.cuh test/gputest_ssyax.cuh \
			  test/gputest_breakpt.cuh test/gputest_cauchy.cuh \
			  test/gputest_prsrch.cuh test/gputest_trpcg.cuh test/gputest_trqsol.cuh \
			  test/gputest_spcg.cuh test/gputest_tron.cuh test/gputest_gpnorm.cuh \
			  test/gputest_driver.cuh
CUH_ACOPF  := src/admm_auglag.cuh src/admm_generator.cuh src/admm_bus.cuh \
              src/consts.cuh src/network.cuh src/parse_mat.cuh src/print.cuh \
			  src/admittance.cuh
HPP_ACOPF  := src/admm_auglag.hpp src/admm_generator.hpp src/admm_bus.hpp \
              src/consts.hpp src/network.hpp src/parse_mat.hpp src/print.hpp \
			  src/admittance.hpp src/gputest_utilities.hpp

INCLUDES   := -I$(SOURCE_DIR) -I$(TEST_DIR)

all: test/gputest src/ExaTron src/ExaTronCPU
test/gputest: test/gputest.cu $(CUH_EXA) $(CUH_TEST)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(GENCODE_FLAGS) test/gputest.cu $(LDFLAGS) -o $@
src/ExaTron: src/acopf_admm.cu $(CUH_EXA) $(CUH_ACOPF)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(GENCODE_FLAGS) src/acopf_admm.cu $(LDFLAGS) -o $@
src/ExaTronCPU: src/acopf_admm.cpp $(HPP_ACOPF)
	$(CC) -O3 $(INCLUDES) src/acopf_admm.cpp -lm -o $@

clean:
	rm -rf *.o test/gputest src/ExaTron src/ExaTronCPU
