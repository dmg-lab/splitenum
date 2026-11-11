CXX        = g++
DELTA     ?= 8
MAX_DELTA ?= $(DELTA)
LIBISOTOPY?= libs/libisotopy
ARCH      ?= -march=native

# MAX_DELTA sizes the fixed arrays inside Isotopy::Graph and is part of its ABI,
# so this project and libisotopy must be built with the same value.  Building at
# 8 instead of the library default of 20 shrinks Graph from ~17KB to ~2KB.
GPU_MAX_DELTA ?= $(MAX_DELTA)
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -flto=auto $(ARCH) -fopenmp -MMD -MP \
           -DMAX_DELTA=$(MAX_DELTA) -DISOTOPY_GPU_MAX_DELTA=$(GPU_MAX_DELTA) \
           -I$(LIBISOTOPY)/include -I$(LIBISOTOPY)/gpu -Isrc
LDEXTRA ?=
LDFLAGS  = -L$(LIBISOTOPY) -lisotopy -Wl,-rpath,'$$ORIGIN/$(LIBISOTOPY)' $(LDEXTRA)

SRC  = src/splits.cpp src/compat.cpp src/store.cpp src/chunkio.cpp \
       src/symmetry.cpp src/refine.cpp src/walk.cpp src/jobs.cpp \
       src/gpuclassify.cpp src/pool.cpp src/master.cpp src/net.cpp src/tcp.cpp src/enumerate.cpp src/cli.cpp
OBJ  = $(SRC:src/%.cpp=obj/%.o)
MAIN = obj/main.o
TARGET = splitenum
MPICXX ?= mpicxx
MPI_TARGET = splitenum-mpi

TEST_SRC = tests/test_splitenum.cpp tests/test_gpurefine_warp.cpp
TEST_BIN = bin/test_splitenum

BENCH_BIN = bin/bench

TOOL_BIN = bin/chunkstat bin/pcomcheck bin/spanprobe bin/witnesses bin/peelaudit bin/statepeek

NVCC ?= nvcc
NVCC_CCBIN ?=
NVCC_EXTRA ?=
CUDA_ARCHS ?= 75
CUDA_LIBDIR ?= $(dir $(shell command -v $(NVCC) 2>/dev/null))../lib
NVCC_GENCODE = $(foreach a,$(CUDA_ARCHS),-gencode arch=compute_$(a),code=sm_$(a)) \
               -gencode arch=compute_$(lastword $(CUDA_ARCHS)),code=compute_$(lastword $(CUDA_ARCHS))
NVCCFLAGS = $(NVCC_GENCODE) -std=c++17 -O3 -DMAX_DELTA=$(MAX_DELTA) \
            -DISOTOPY_GPU_MAX_DELTA=$(GPU_MAX_DELTA) -I$(LIBISOTOPY)/include \
            -I$(LIBISOTOPY)/gpu -Isrc \
            $(if $(NVCC_CCBIN),-ccbin $(NVCC_CCBIN),) $(NVCC_EXTRA)
GPU_TARGET = splitenum-gpu
GPU_OBJ = $(filter-out obj/gpuclassify.o,$(OBJ)) obj/gpuclassify_cuda.o \
          obj/gpu_device.o

.PHONY: all libisotopy tools test bench mpi mpi-test gpu clean distclean

all: $(TARGET)

tools: $(TOOL_BIN)

libisotopy: $(LIBISOTOPY)/libisotopy.so

$(LIBISOTOPY)/libisotopy.so:
	$(MAKE) -C $(LIBISOTOPY) libisotopy.so ARCH= MAX_DELTA=$(MAX_DELTA)

obj:
	mkdir -p obj

bin:
	mkdir -p bin

obj/%.o: src/%.cpp | obj $(LIBISOTOPY)/libisotopy.so
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJ:.o=.d) $(MAIN:.o=.d) $(wildcard bin/*.d)

$(TARGET): $(OBJ) $(MAIN) | $(LIBISOTOPY)/libisotopy.so
	$(CXX) $(CXXFLAGS) $(OBJ) $(MAIN) -o $@ $(LDFLAGS)

$(TEST_BIN): $(TEST_SRC) src/gpurefine.h $(OBJ) | bin $(LIBISOTOPY)/libisotopy.so
	$(CXX) $(CXXFLAGS) -I$(LIBISOTOPY)/tests $(TEST_SRC) $(OBJ) -o $@ \
	    -L$(LIBISOTOPY) -lisotopy -Wl,-rpath,'$$ORIGIN/../$(LIBISOTOPY)' $(LDEXTRA)

bin/%: tools/%.cpp $(OBJ) | bin $(LIBISOTOPY)/libisotopy.so
	$(CXX) $(CXXFLAGS) $< $(OBJ) -o $@ \
	    -L$(LIBISOTOPY) -lisotopy -Wl,-rpath,'$$ORIGIN/../$(LIBISOTOPY)' $(LDEXTRA)

$(BENCH_BIN): bench/bench.cpp $(OBJ) | bin $(LIBISOTOPY)/libisotopy.so
	$(CXX) $(CXXFLAGS) $< $(OBJ) -o $@ \
	    -L$(LIBISOTOPY) -lisotopy -Wl,-rpath,'$$ORIGIN/../$(LIBISOTOPY)' $(LDEXTRA)

mpi: $(MPI_TARGET)

$(MPI_TARGET): src/mpi_main.cpp $(OBJ) | $(LIBISOTOPY)/libisotopy.so
	$(MPICXX) $(CXXFLAGS) -MF obj/mpi_main.d src/mpi_main.cpp $(OBJ) -o $@ $(LDFLAGS)

gpu: $(GPU_TARGET)

obj/gpuclassify_cuda.o: src/gpuclassify.cpp | obj $(LIBISOTOPY)/libisotopy.so
	$(CXX) $(CXXFLAGS) -DSPLITENUM_CUDA -c $< -o $@

obj/gpu_device.o: src/gpu_device.cu src/gpudevice.h src/gpurefine.h | obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_TARGET): $(GPU_OBJ) $(MAIN) | $(LIBISOTOPY)/libisotopy.so
	$(CXX) $(CXXFLAGS) $(GPU_OBJ) $(MAIN) -o $@ $(LDFLAGS) \
	    -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart

mpi-test: $(MPI_TARGET) $(TARGET)
	rm -rf bin/mpitest && mkdir -p bin/mpitest
	./$(TARGET) --delta 5 --threads 2 --out bin/mpitest/chunks > /dev/null
	mpirun -n 3 ./$(MPI_TARGET) --delta 5 --threads 2 --budget 500 \
	    --slice-seconds 0.05 --checkpoint-seconds 0 --out bin/mpitest/mpi > /dev/null
	sort bin/mpitest/chunks/types.csv > bin/mpitest/a.csv
	sort bin/mpitest/mpi/types.csv > bin/mpitest/b.csv
	cmp bin/mpitest/a.csv bin/mpitest/b.csv
	@echo "mpi-test: types.csv identical"

bench: $(BENCH_BIN)
	./$(BENCH_BIN) $(BENCH_ARGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf obj bin $(TARGET) $(MPI_TARGET) $(GPU_TARGET)

distclean: clean
	$(MAKE) -C $(LIBISOTOPY) clean || true
	rm -rf out
