# Build with: make            (all programs)
#             make optH2      (only the training)
# Eigen 3 is needed, change EIGEN if it is installed somewhere else
# (on macOS with Homebrew it is usually /opt/homebrew/include/eigen3)

CXX      = mpic++
EIGEN   ?= /usr/include/eigen3
CXXFLAGS = -std=c++17 -O3 -I$(EIGEN)

PROGRAMS = optH2 Energy diagH2kron Wfplotter

all: $(PROGRAMS)

%: %.cc $(wildcard src/*.cc src/*.hh)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
