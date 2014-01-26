include $(FSLCONFDIR)/default.mk

PROJNAME = fdt

# TODO: Move LIB_OPENCL and INC_OPENCL into systemvars.mk
LIB_OPENCL=/usr/lib64/nvidia
INC_OPENCL=/usr/include/nvidia

USRINCFLAGS = -I${INC_NEWMAT} -I${INC_NEWRAN} -I${INC_CPROB} -I${INC_PROB} -I${INC_BOOST} -I${INC_ZLIB} -I${INC_OPENCL}
USRLDFLAGS = -L${LIB_NEWMAT} -L${LIB_NEWRAN} -L${LIB_CPROB} -L${LIB_PROB} -L${LIB_ZLIB} -L${LIB_OPENCL}

DLIBS =  -lwarpfns -lbasisfield -lfslsurface  -lfslvtkio -lmeshclass -lnewimage -lutils -lmiscmaths -lnewmat -lnewran -lfslio -lgiftiio -lexpat -lfirst_lib -lniftiio -lznz -lcprob -lutils -lprob -lm -lz -lOpenCL

OCLPTX=oclptx
OCLPTXOBJ=oclptx.o oclptxhandler.o samplemanager.o

XFILES=${OCLPTX}

all: ${OCLPTX}

${OCLPTX}: ${OCLPTXOBJ}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $^ ${DLIBS}
