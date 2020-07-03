PETSC_DIR=/home/soumya/Downloads/petsc/petsc-3.8.3

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

main: main.o  chkopts
	-${CLINKER} -o main main.o  ${PETSC_KSP_LIB}
	${RM} main.o
