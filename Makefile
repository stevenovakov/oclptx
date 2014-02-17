# NOTE
# Based off of Makefile of ptx2
#
# -std=c++0x is required for using:
#		<thread>	<chrono>	<mutex>
#

include $(FSLCONFDIR)/default.mk

PROJNAME =fdt

DBGFLAGS=-g

# TODO: Move LIB_OPENCL and INC_OPENCL into systemvars.mk
# test b/w AMD/Nvidia hardware?
#LIB_OPENCL=/usr/lib64/nvidia
LIB_OPENCL=/opt/AMD-APP-SDK-v2.8-RC-lnx64/lib/x86_64
#INC_OPENCL=/usr/local/cuda-5.5/include
INC_OPENCL=/opt/AMD-APP-SDK-v2.8-RC-lnx64/include
CPP11 = -std=c++0x

USRINCFLAGS = -I${INC_NEWMAT} -I${INC_NEWRAN} -I${INC_CPROB} -I${INC_PROB} -I${INC_BOOST} -I${INC_ZLIB} -I${INC_OPENCL} ${CPP11}
USRLDFLAGS = -L${LIB_NEWMAT} -L${LIB_NEWRAN} -L${LIB_CPROB} -L${LIB_PROB} -L${LIB_ZLIB} -L${LIB_OPENCL}

DLIBS =	-lwarpfns -lbasisfield -lfslsurface	-lfslvtkio -lmeshclass -lnewimage -lutils -lmiscmaths -lnewmat -lnewran -lfslio -lgiftiio -lexpat -lfirst_lib -lniftiio -lznz -lcprob -lutils -lprob -lm -lz -lOpenCL

OCLPTX=oclptx
OCLPTXOBJ=oclptx.o oclenv.o oclptxhandler.o samplemanager.o oclptxOptions.o

XFILES=${OCLPTX}

all: ${OCLPTX}

${OCLPTX}: ${OCLPTXOBJ}
				${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}

lint: *.cc *.h
				bash -c 'python cpplint.py --extensions=cc,h --filter=-whitespace/braces $^ > lint 2>&1'

