# Location of the CUDA Toolkit
NVCC := $(CUDA_PATH)/bin/nvcc
CCFLAGS := -O2
EXTRA_NVCCFLAGS := --cudart=shared 
CXXFLAGS := -std=c++11
build: vectorAdd

vectorAdd.o:vectorAdd.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) $(CXXFLAGS) -o $@ -c $<

vectorAdd: vectorAdd.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./vectorAdd

clean:
	rm -f vectorAdd *.o
