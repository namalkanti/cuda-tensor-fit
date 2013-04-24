#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Emulates numpy's maximum function and log function combined for efficiency.
//Iterates through array and if value is less than min signal, replaces with minimum value.
//Also takes logarithm of every value.
void cutoff_log(double* signal, double min_signal, size_t n){
    int i;
    for (i = 0; i < n; i++){
        if (signal[i] < min_signal){
            signal[i] = min_signal;
        }
        signal[i] = log(signal[i]);
    }
}
