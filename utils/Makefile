OS := $(shell uname)

$(info $$OS is $(OS)) # 'Darwin' for MacOS

ifdef LONG
INTT = -DLONG
endif

ifdef EDGELONG
INTE = -DEDGELONG
endif

#compilers
ifdef CILK
PCC = g++
PCFLAGS = -std=c++14 -fcilkplus -lcilkrts -O3 -DCILK $(INTT) $(INTE)
PLFLAGS = -fcilkplus -lcilkrts

else ifdef MKLROOT
PCC = icpc
PCFLAGS = -std=c++14 -O3 -DCILKP $(INTT) $(INTE)

else ifdef OPENMP
PCC = g++
PCFLAGS = -std=c++14 -fopenmp -march=native -O3 -DOPENMP $(INTT) $(INTE)



else ifeq ($(OS),Darwin) 
# MacOS
# Aliasing g++ to /opt/llvm doesn't work in CLion
PCC = /opt/homebrew/opt/llvm/bin/clang++
# PCFLAGS = -std=c++14 -march=native -O3 $(INTT) $(INTE) $(CODE) $(PD) $(MEM) # Serial
PCFLAGS = -std=c++14 -march=native -O3 -fopenmp -DOPENMP $(INTT) $(INTE) $(CODE) $(PD) $(MEM) # Parallel
else
PCC = g++ #Linux
PCFLAGS = -std=c++14 -fopenmp -march=native -O3 -DOPENMP $(INTT) $(INTE) # Parallel
# PCFLAGS = -std=c++14 -march=native -O3 $(INTT) $(INTE) # Serial
endif

COMMON = utils.h parseCommandLine.h parallel.h quickSort.h blockRadixSort.h transpose.h
LOCAL_COMMON = graphIO.h
GENERATORS = rMatGraph gridGraph randLocalGraph SNAPtoAdj intWeightSNAPtoAdj adjGraphAddWeights adjToBinary communityToHyperAdj hyperAdjToBinary adjHypergraphAddWeights randHypergraph KONECTtoHyperAdj KONECTtoClique communityToClique communityToMESH KONECTtoMESH

.PHONY: all clean
all: $(GENERATORS)

$(COMMON):
	ln -s ../ligra/$@ .

% : %.C $(COMMON) $(LOCAL_COMMON)
	$(PCC) $(PCFLAGS) -o $@ $<

clean :
	rm -f *.o $(GENERATORS)

cleansrc :
	make -s clean
	rm -f $(COMMON)
