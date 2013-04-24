#Makefile to run and test fit tensor functions

test: unit_test.c
	./fit_tensor_tests

unit_test.c: 
	gcc -g -o fit_tensor_tests fit_unit_test.c -lcunit -lm

clean:
	rm fit_tensor_tests
