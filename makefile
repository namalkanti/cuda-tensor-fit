#Makefile to run and test fit tensor functions

run: tests
	valgrind --tool=memcheck --leak-check=yes --track-origins=yes ./fit_tensor_tests

tests:  
	gcc -g -I/usr/lib/sagemath/local/include -o fit_tensor_tests fit_unit_test.c -lcunit -lm -lgsl -lgslcblas

clean:
	rm fit_tensor_tests
