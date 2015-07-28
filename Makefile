include $(FSLCONFDIR)/default.mk

PROJNAME=fdt

DBGFLAGS=-g -O0
USRCXXFLAGS=-std=c++11 -MMD -MP
GCCBUGFLAGS=-pthread -std=c++11

# LIB_OPENCL and INC_OPENCL should be shell variables pointing to, for example,
# /usr/local/cuda/lib64 and /usr/local/cuda/include, for CUDA 7.0
# the first should point to libOpenCL.so and the second to the directory with
# cl.hpp, etc

USRINCFLAGS = -I${INC_NEWMAT} -I${INC_NEWRAN} -I${INC_CPROB} -I${INC_PROB} -I${INC_BOOST} -I${INC_ZLIB} -I${INC_OPENCL}
USRLDFLAGS = -L${LIB_NEWMAT} -L${LIB_NEWRAN} -L${LIB_CPROB} -L${LIB_PROB} -L${LIB_ZLIB} -L${LIB_OPENCL}

DLIBS =	-lwarpfns -lbasisfield -lfslvtkio -lmeshclass -lnewimage -lutils -lmiscmaths -lnewmat -lnewran -lfslio -lfirst_lib -lniftiio -lznz -lcprob -lutils -lprob -lm -lz -lOpenCL

OCLPTX=oclptx
OCLPTXOBJ=main.o oclenv.o oclptxhandler.o threading.o samplemanager.o oclptxOptions.o particlegen.o

RNGTEST=rng_test
RNGTESTOBJ=rng_test.o oclenv.o

FIFOTEST=fifo_test
FIFOTESTOBJ=fifo_test.o

XFILES=${OCLPTX}

all: ${OCLPTX}

${OCLPTX}: ${OCLPTXOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS} ${GCCBUGFLAGS}

${RNGTEST}: ${RNGTESTOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

${FIFOTEST}: ${FIFOTESTOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

.PHONY: lint
lint:
	bash -c 'python cpplint.py --extensions=cc,h,cl --filter=-whitespace/braces `find ./ -name \*.h -o -name \*.cc -o -name \*.cl` > lint 2>&1'

-include $(OCLPTXOBJ:%.o=%.d)
