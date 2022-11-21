OS := $(shell uname)

$(info $$OS is $(OS)) # 'Darwin' for MacOS

ifdef LONG
INTT = -DLONG
endif

ifdef EDGELONG
INTE = -DEDGELONG
endif

ifdef PD
PD = -DPD
endif

ifdef BYTE
CODE = -DBYTE
else ifdef NIBBLE
CODE = -DNIBBLE
else
CODE = -DBYTERLE
endif

ifdef LOWMEM
MEM = -DLOWMEM
endif

#compilers
ifdef CILK
PCC = g++
PCFLAGS = -std=c++14 -fcilkplus -lcilkrts -O3 -DCILK $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
PLFLAGS = -fcilkplus -lcilkrts

else ifdef MKLROOT
PCC = icpc
PCFLAGS = -std=c++14 -O3 -DCILKP $(INTT) $(INTE) $(CODE) $(PD) $(MEM)

else ifdef OPENMP
PCC = g++
PCFLAGS = -std=c++14 -fopenmp -march=native -O3 -DOPENMP $(INTT) $(INTE) $(CODE) $(PD) $(MEM)

else ifeq ($(OS),Darwin)
# Run MacOS commands
# Aliasing g++ to /opt/llvm doesn't work in CLion
PCC = /opt/homebrew/opt/llvm/bin/clang++
# PCFLAGS = -std=c++14 -march=native -O3 -fopenmp $(INTT) $(INTE) $(CODE) $(PD) $(MEM) # Serial
PCFLAGS = -std=c++14 -march=native -O3 -fopenmp -DOPENMP $(INTT) $(INTE) $(CODE) $(PD) $(MEM) # Parallel
else
# check for Linux and run other commands
PCC = g++ # Add these for parallel run -fopenmp -DOPENMP. To Debug, add -g
PCFLAGS = -std=c++14 -march=native -O3 -fopenmp -DOPENMP $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
endif

COMMON= ligra.h graph.h compressedVertex.h vertex.h utils.h IO.h parallel.h gettime.h index_map.h maybe.h sequence.h edgeMap_utils.h binary_search.h quickSort.h blockRadixSort.h transpose.h parseCommandLine.h byte.h byteRLE.h nibble.h byte-pd.h byteRLE-pd.h nibble-pd.h vertexSubset.h encoder.C decoder.C

ALL= encoder decoder BFS BC BellmanFord Components Components-Shortcut Radii PageRank PageRankDelta GraphEncoderEmbedding WeighedGraphEncoderEmbedding BFSCC BFS-Bitvector KCore MIS Triangle CF

all: $(ALL)

% : %.C $(COMMON)
	$(PCC) $(PCFLAGS) -o $@ $<

$(COMMON):
	ln -s ../ligra/$@ .

.PHONY : clean

clean :
	rm -f *.o $(ALL)

cleansrc :
	rm -f *.o $(ALL)
	rm $(COMMON)
