CXX = g++
NVCC = nvcc
ARCH = sm_61

# Common Flags
COMMON_FLAGS = -std=c++17
LDFLAGS = 
LIBS = 

# Debug Flags
DEBUG_FLAGS = -g -DDEBUG
CUDA_DEBUG_FLAGS = -G

# Release Flags
RELEASE_FLAGS = -O3 -DNDEBUG

# Dependency Flags
DEPFLAGS = -MMD -MP

# Object Files and Dependencies
OBJECTS = graph.o mmio.o
DEPS = $(OBJECTS:.o=.d)

# Targets
TARGETS = gpubfs gpubfs_blockactive gpubfs_ultrafinegrained 

# Default build set to Release
BUILD = release

# Conditional Flags based on Build Type
ifeq ($(BUILD),debug)
  CXXFLAGS = $(COMMON_FLAGS) $(DEBUG_FLAGS)
  NVCCFLAGS = -arch=$(ARCH) $(COMMON_FLAGS) $(CUDA_DEBUG_FLAGS)
else
  CXXFLAGS = $(COMMON_FLAGS) $(RELEASE_FLAGS)
  NVCCFLAGS = -arch=$(ARCH) $(COMMON_FLAGS) $(RELEASE_FLAGS)
endif

.PHONY: all clean debug release

# Default target
all: $(TARGETS)

debug:
	$(MAKE) BUILD=debug

release:
	$(MAKE) BUILD=release

gpubfs: gpubfs.cu $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ $(LIBS) -o $@

gpubfs_blockactive: gpubfs_blockactive.cu $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ $(LIBS) -o $@

gpubfs_ultrafinegrained: gpubfs_ultrafinegrained.cu $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) $(OBJECTS) $(DEPS)

-include $(DEPS)
