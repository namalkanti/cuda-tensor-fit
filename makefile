#Makefile to run and test fit tensor functions

run: tests
	valgrind --tool=memcheck --leak-check=yes ./fit_tensor_tests

tests: fit_tensor.h 
	gcc -g -I/usr/lib/sagemath/local/include -o fit_tensor_tests fit_unit_test.c -lcunit -lm

clean:
	rm fit_tensor_tests
