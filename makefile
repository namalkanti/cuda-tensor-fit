#Makefile to run and test fit tensor functions

CFLAGS = -g 

memcheck = valgrind --tool=memcheck --leak-check=yes --track-origins=yes 

cuda-leak: cuda-test
	cuda-memcheck --leak-check full ./cuda_test

opt-leak: opt-test
	${memcheck} ./opt_test

leaks: test
	${memcheck} ./main_test

run-cuda: cuda-test
	./cuda_test

run: test
	./structure_test
	./opt_test

cuda-test: cuda_util.o
	gcc ${CFLAGS} -o cuda_test cuda_test.c fit_tensor_util.o fit_tensor.o cuda_util.o -L/usr/local/cuda/lib64 -lgsl -lgslcblas -lm -lcuda -lcudart -lcublas -lcunit 

opt-test: fit_tensor_util.o
	gcc ${CFLAGS} -o opt_test opt_test.c fit_tensor_util.o -lgsl -lgslcblas -lm -lcunit

test: fit_tensor.o
	gcc ${CFLAGS} -o main_test fit_unit_test.c fit_tensor.o fit_tensor_util.o -lgsl -lgslcblas -lm -lcunit

cuda_util.o: fit_tensor.o 
	nvcc ${CFLAGS} -c cuda_util.cu

opt_test: opt_util.o
	gcc ${CFLAGS} -o opt_test opt_test.c opt_util.o structure_util.o -lm -lgsl -lgslcblas -lcunit
	
opt_util.o: opt_util.c
	gcc ${CFLAGS} -o opt_util.o -c opt_util.c

structure_test:structure_util.o
	gcc ${CFLAGS} -o structure_test structure_unit_test.c structure_util.o -lm -lgsl -lgslcblas -lcunit

structure_util.o: structure_util.c
	gcc ${CFLAGS} -o structure_util.o -c structure_util.c

clean: 
	rm *.o
