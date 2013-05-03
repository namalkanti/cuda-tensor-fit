#Makefile to run and test fit tensor functions

headers = data.h fit_tensor_util.h fit_tensor_opt.h

compile_tests = gcc -I/usr/lib/sagemath/local/include -o fit_tensor_tests fit_unit_test.c -lcunit -lm -lgsl -lgslcblas 

memcheck = valgrind --tool=memcheck --leak-check=yes --track-origins=yes ./fit_tensor_tests

leaks-mp: tests-mp
	${memcheck}

leaks: tests
	${memcheck}

run-mp: tests-mp
	./fit_tensor_tests

run: tests
	./fit_tensor_tests

debug-mp: ${headers}
	${compile_tests} -fopenmp -g
	
debug: ${headers}
	${compile_tests} -g

tests-mp: ${headers}
	${compile_tests} -fopenmp 

tests: ${headers} 
	${compile_tests}

clean: 
	rm fit_tensor_tests
