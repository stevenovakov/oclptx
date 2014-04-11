include $(FSLCONFDIR)/default.mk

PROJNAME=fdt

DBGFLAGS=-g -O0
USRCXXFLAGS=-std=c++0x -MMD -MP

# TODO: Move LIB_OPENCL and INC_OPENCL into systemvars.mk
# test b/w AMD/Nvidia hardware?
LIB_OPENCL=/usr/lib64/nvidia
#LIB_OPENCL=/opt/AMD-APP-SDK-v2.8-RC-lnx64/lib/x86_64
INC_OPENCL=/usr/local/cuda-5.5/include
#INC_OPENCL=/opt/AMD-APP-SDK-v2.8-RC-lnx64/include

USRINCFLAGS = -I${INC_NEWMAT} -I${INC_NEWRAN} -I${INC_CPROB} -I${INC_PROB} -I${INC_BOOST} -I${INC_ZLIB} -I${INC_OPENCL}
USRLDFLAGS = -L${LIB_NEWMAT} -L${LIB_NEWRAN} -L${LIB_CPROB} -L${LIB_PROB} -L${LIB_ZLIB} -L${LIB_OPENCL}

DLIBS =	-lwarpfns -lbasisfield -lfslvtkio -lmeshclass -lnewimage -lutils -lmiscmaths -lnewmat -lnewran -lfslio -lfirst_lib -lniftiio -lznz -lcprob -lutils -lprob -lm -lz -lOpenCL

OCLPTX=oclptx
OCLPTXOBJ=main.o oclenv.o oclptxhandler.o threading.o samplemanager.o oclptxOptions.o

RNGTEST=rng_test
RNGTESTOBJ=rng_test.o oclenv.o

FIFOTEST=fifo_test
FIFOTESTOBJ=fifo_test.o

XFILES=${OCLPTX}

all: ${OCLPTX}

${OCLPTX}: ${OCLPTXOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

${RNGTEST}: ${RNGTESTOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

${FIFOTEST}: ${FIFOTESTOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

.PHONY: lint
lint:
	bash -c 'python cpplint.py --extensions=cc,h,cl --filter=-whitespace/braces `find ./ -name \*.h -o -name \*.cc -o -name \*.cl` > lint 2>&1'

-include $(OCLPTXOBJ:%.o=%.d)
-include $(FIFOTESTOBJ:%.o=%.d)
