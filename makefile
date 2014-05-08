#Makefile to run and test fit tensor functions

CFLAGS = -g 

memcheck = valgrind --tool=memcheck --leak-check=yes --track-origins=yes 

cuda-leak: cuda-test
	cuda-memcheck --leak-check full ./cuda_test

leaks: test
	${memcheck} ./structure_test
	${memcheck} ./opt_test

run-cuda: cuda-test
	./cuda_test

run: structure_test opt_test
	./structure_test
	./opt_test

cuda-test: cuda_util.o
	gcc ${CFLAGS} -o cuda_test cuda_test.c structure_util.o opt_util.o cuda_util.o BatchedSolver/solve.o -L/usr/local/cuda/lib64 -lgsl -lgslcblas -lm -lcuda -lcudart -lcublas -lcunit 

cuda_util.o: structure_util.o opt_util.o 
	nvcc ${CFLAGS} -G -c  -O0 cuda_util.cu

opt_test: opt_util.o
	gcc ${CFLAGS} -o opt_test opt_test.c opt_util.o structure_util.o -lm -lgsl -lgslcblas -lcunit
	
opt_util.o: opt_util.c
	gcc ${CFLAGS} -o opt_util.o -c opt_util.c

structure_test: structure_util.o
	gcc ${CFLAGS} -o structure_test structure_unit_test.c structure_util.o -lm -lgsl -lgslcblas -lcunit

structure_util.o: structure_util.c
	gcc ${CFLAGS} -o structure_util.o -c structure_util.c

clean: 
	rm *.o
