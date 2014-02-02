include $(FSLCONFDIR)/default.mk

PROJNAME = fdt

DBGFLAGS=-g
USRCXXFLAGS=-std=c++0x

# TODO: Move LIB_OPENCL and INC_OPENCL into systemvars.mk
LIB_OPENCL=/usr/lib64/nvidia
INC_OPENCL=/usr/local/cuda-5.5/include

USRINCFLAGS = -I${INC_NEWMAT} -I${INC_NEWRAN} -I${INC_CPROB} -I${INC_PROB} -I${INC_BOOST} -I${INC_ZLIB} -I${INC_OPENCL} -I..
USRLDFLAGS = -L${LIB_NEWMAT} -L${LIB_NEWRAN} -L${LIB_CPROB} -L${LIB_PROB} -L${LIB_ZLIB} -L${LIB_OPENCL}

DLIBS =  -lwarpfns -lbasisfield -lfslsurface  -lfslvtkio -lmeshclass -lnewimage -lutils -lmiscmaths -lnewmat -lnewran -lfslio -lgiftiio -lexpat -lfirst_lib -lniftiio -lznz -lcprob -lutils -lprob -lm -lz -lOpenCL

OCLPTX=oclptx
OCLPTXOBJ=main.o gpu.o threading.o

GPUTEST=gpu_test
GPUTESTOBJ=gpu_test.o gpu.o

FIFOTEST=fifo_test
FIFOTESTOBJ=fifo_test.o

XFILES=${OCLPTX}

all: ${OCLPTX}

${OCLPTX}: ${OCLPTXOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

${GPUTEST}: ${GPUTESTOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

${FIFOTEST}: ${FIFOTESTOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}


lint: *.cc *.h *.cl
	bash -c 'python cpplint.py --extensions=cc,h,cl --filter=-whitespace/braces $^ > lint 2>&1'
