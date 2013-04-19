#include <stdio.h>
#include <stdlib.h>
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <math.h>
#include "data.h"

//Initalization stub for compare test suite
int init_compare(void){
    return 0;
}

//Cleanup stub for compare test suite
int clean_compare(void){
    return 0;
}

//Tests for array compare function
void test_compare_array(void){
   double test1[] = {0, 0, 0, 0, 0}; 
   double test2[] = {0, 0, 0, 0, 0};
   double test3[] = {1, 2, 3, 4, 5};
   double test4[] = {1, 2, 3, 4, 5};
   double test5[] = {10, 23, 45, 234, 455};
   double test6[] = {10, 23, 45, 234, 455};
   CU_ASSERT(arr_compare(test1, test2, 5, .00001) == true);
   CU_ASSERT(arr_compare(test2, test3, 5, .00001) == false);
   CU_ASSERT(arr_compare(test5, test6, 5, .00001) == true);
   CU_ASSERT(arr_compare(test1, test6, 5, .00001) == false);
   CU_ASSERT(arr_compare(test3, test5, 5, .00001) == false);
}

//Tests for compare tensor functions
void test_compare_tensors(void){
    double sig4a[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 1};
    double val4a[] = {3.769504672651095*pow(10, -5), 4.999999999999999*pow(10, -10), 4.999999999999999*pow(10, -10)};
    double vec4a[] = {{0.8383792245225715, 0.4482168235823326, 0.3101966391608646}, {-0.24173661893711018, -0.20432347278508128, 0.9485859610672609}, {0.48855264092397055, -0.8702606491889705, -0.06295013518179123}};
    tensor tensor4a = {val4a, vec4a};

    double sig5a[] = {1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0};
    double val5a[] = {2.2462636541968677*pow(10, -5), 1.197421120795504*pow(10, -5), 4.999999999999999*pow(10, -10)};
    double vec5a[][] = {{0.8512136823637526, 0.45060875596465305, 0.26904835254044707}, {0.3834153608107982, -0.8839931657006418, 0.2674859698916625}, {-0.35836842501965277, 0.12453044624727543, 0.9252373965131264}};
    tensor tensor5a = {val5a, vec5a};
    
    CU_ASSERT(compare_tensors(&tensor4, &tensor4a, .00001) == true);
    CU_ASSERT(compare_tensors(&tensor5, &tensor5a, .00001) == true);
    CU_ASSERT(compare_tensors(&tensor4, &tensor5a, .00001) == false);
    CU_ASSERT(compare_tensors(&tensor5, &tensor4a, .00001) == false);

}

int main{
    CU_pSuite compare_suite = NULL;
    CU_pSuite opt_suite = NULL;

    if (CUE_SUCCESS != CU_initilize_registry())
        return CU_get_error();

    compare_suite = CU_add_suite("Compare Suite", init_compare, clean_compare);
    if (NULL == compare_suite){
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}
