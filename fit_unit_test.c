#include <stdio.h>
#include <stdlib.h>
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include "fit_tensor_util.h"

//Initalization stub for utility test suite
int init_utility(void){
    return 0;
}

//Cleanup stub for utility test suite
int clean_utility(void){
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

//Tests matrix comparison function
void test_compare_matrix(void){
    double mat1_data[] = {0, 0, 0, 0};
    matrix mat1 = {mat1_data, 2, 2};
    double mat2_data[] = {0, 0, 0, 0};
    matrix mat2 = {mat2_data, 2, 2};
    double mat3_data[] = {0, 0, 0, 0};
    matrix mat3 = {mat3_data, 1, 4};
    double mat4_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    matrix mat4 = {mat4_data, 3, 3};
    double mat5_data[] = {3, 2, 6, 4, 4, 3, 88, 8, 9};
    matrix mat5 = {mat5_data, 3, 3};
    double mat6_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    matrix mat6 = {mat6_data, 3, 3};
    CU_ASSERT(mat_compare(&mat1, &mat3, .00001) == false);
    CU_ASSERT(mat_compare(&mat1, &mat2, .00001) == true);
    CU_ASSERT(mat_compare(&mat4, &mat5, .00001) == false);
    CU_ASSERT(mat_compare(&mat4, &mat6, .00001) == true);
}

//Tests columnar eigen compare function
void test_columnar_eig_compare(void){
    double mat1_data[] = {0, -1, 2, 3, -4, 5, 6, -7, 8};
    matrix mat1 = {mat1_data, 3, 3};
    double mat2_data[] = {0, -1, 2, 3, -4, 5, 6, -7, 8};
    matrix mat2 = {mat2_data, 3, 3};
    double mat3_data[] = {0, -1, -2, 3, -4, 5, 6, -7, 8};
    matrix mat3 = {mat3_data, 3, 3};
    double mat4_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    matrix mat4 = {mat4_data, 3, 3};
    double mat5_data[] = {3, 2, 6, 4, 4, 3, 88, 8, 9};
    matrix mat5 = {mat5_data, 3, 3};
    double mat6_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    matrix mat6 = {mat6_data, 3, 3};
    CU_ASSERT(columnar_eig_compare(&mat1, &mat2, .00001) == true);
    CU_ASSERT(columnar_eig_compare(&mat1, &mat3, .00001) == false);
    CU_ASSERT(columnar_eig_compare(&mat1, &mat4, .00001) == true);
    CU_ASSERT(columnar_eig_compare(&mat4, &mat5, .00001) == false);
    CU_ASSERT(columnar_eig_compare(&mat4, &mat6, .00001) == true);
}

//Tests for compare tensor functions
void test_compare_tensors(void){
    double sig4a[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 1};
    double val4a[] = {0.0000376950467265109499890, 0.0000000004999999999999999, 0.0000000004999999999999999};
    double vec4a_data[] = {0.8383792245225715, 0.4482168235823326, 0.3101966391608646, -0.24173661893711018, -0.20432347278508128, 0.9485859610672609, 0.48855264092397055, -0.8702606491889705, -0.06295013518179123};
    matrix vec4a = {vec4a_data, 3, 3};
    tensor tensor4a = {val4a, &vec4a};

    double sig5a[] = {1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0};
    double val5a[] = {0.0000224626365419686772299, 0.0000119742112079550399738, 0.0000000004999999999999999};
    double vec5a_data[] = {0.8512136823637526, 0.45060875596465305, 0.26904835254044707, 0.3834153608107982, -0.8839931657006418, 0.2674859698916625, -0.35836842501965277, 0.12453044624727543, 0.9252373965131264};
    matrix vec5a = {vec5a_data, 3, 3};
    tensor tensor5a = {val5a, &vec5a};
    
    CU_ASSERT(compare_tensors(&tensor4, &tensor4a, .00001) == true);
    CU_ASSERT(compare_tensors(&tensor5, &tensor5a, .00001) == true);
    CU_ASSERT(compare_tensors(&tensor4, &tensor5a, .00001) == false);
    CU_ASSERT(compare_tensors(&tensor5, &tensor4a, .00001) == false);

}

//Test to convert to a gsl matrix and back
void test_gsl_matrix_convert(void){
    double test1_data[] = {3, 4, 5, 6, 2, 4, 54, 6, 7, 34, 534, 3};
    matrix test1 = {test1_data, 2, 6};
    matrix test1b = {test1_data, 6, 2};
    double test2_data[] = {7, 6, 4, 87, 3, 756, 34, 76, 83, 45, 34, 22, 65, 76};
    matrix test2 = {test2_data, 2, 7};
    matrix test2b = {test2_data, 7, 2};
    double test3_data[] = {23, 3, 5, 6, 2, 3, 3, 4, 5};
    matrix test3 = {test3_data, 3, 3};
    gsl_matrix* gsl1 = to_gsl(&test1);
    gsl_matrix* gsl1b = to_gsl(&test1b);
    gsl_matrix* gsl2 = to_gsl(&test2);
    gsl_matrix* gsl2b = to_gsl(&test2b);
    gsl_matrix* gsl3 = to_gsl(&test3);
    matrix* return1 = to_matrix(gsl1);
    matrix* return1b = to_matrix(gsl1b);
    matrix* return2 = to_matrix(gsl2);
    matrix* return2b = to_matrix(gsl2b);
    matrix* return3 = to_matrix(gsl3);
    CU_ASSERT(mat_compare(&test1, return1, .000001) == true);
    CU_ASSERT(mat_compare(&test1b, return1b, .00001) == true);
    CU_ASSERT(mat_compare(&test2, return2, .000001) == true);
    CU_ASSERT(mat_compare(&test2b, return2b, .000001) == true);
    CU_ASSERT(mat_compare(&test3, return3, .0000001) == true);
    CU_ASSERT(mat_compare(&test1, return2, .000000001) == false);
    CU_ASSERT(mat_compare(&test1, return3, .000000001) == false);
    free_matrix(return1);
    free_matrix(return1b);
    free_matrix(return2);
    free_matrix(return2b);
    free_matrix(return3);
    gsl_matrix_free(gsl1);
    gsl_matrix_free(gsl1b);
    gsl_matrix_free(gsl2);
    gsl_matrix_free(gsl2b);
    gsl_matrix_free(gsl3);
}

//Test for cutoff and log function
void test_cutoff_log(void){
    double min_value = M_E;
    double test1[] = {0, 2, 3, 4, 1, 23, 3, 5, 6, 43, 5};
    size_t len1 = sizeof(test1)/sizeof(test1[0]);
    double result1[] = {1.0,
     1.0,
     1.0986122886681098,
     1.3862943611198906,
     1.0,
     3.1354942159291497,
     1.0986122886681098,
     1.6094379124341003,
     1.791759469228055,
     3.7612001156935624,
     1.6094379124341003};
    double test2[] = {1, 5, 2, 1, 4, 1.3, 4.5, 1.2, 3, 41, 9};
    size_t len2 = sizeof(test2)/sizeof(test2[0]);
    double result2[] = {1.0,
     1.6094379124341003,
     1.0,
     1.0,
     1.3862943611198906,
     1.0,
     1.5040773967762742,
     1.0,
     1.0986122886681098,
     3.713572066704308,
     2.1972245773362196};
    cutoff_log(test1, min_value, len1);
    cutoff_log(test2, min_value, len2);
    CU_ASSERT(arr_compare(test1, result1, len1, .0000001) == true);
    CU_ASSERT(arr_compare(test2, result2, len2, .00000001) == true);
}

//Test function for exponentation
void test_exp_array(void){
    double test1[] = {0, 0, 0, 0};
    double results1[] = {1, 1, 1, 1};
    size_t size1 = sizeof(test1)/sizeof(test1[0]);
    double test2[] = {1, 2, 3, 4, 14, 15, 15, 3, 12, 13};
    double results2[] = {2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 1202604.2841647768, 3269017.3724721107, 3269017.3724721107, 20.085536923187668, 162754.79141900392, 442413.3920089205};
    size_t size2 = sizeof(test2)/sizeof(test2[0]);
    double* return1 = exp_array(test1, size1);
    double* return2 = exp_array(test2, size2);
    CU_ASSERT(arr_compare(results1, return1, size1, .0000001) == true);
    CU_ASSERT(arr_compare(results2, return2, size2, .0000001) == true);
    free(return1);
    free(return2);
}

//Test function to form lower triangular tensor
void test_tensor_lower_triangular(void){
    double test1[] = {0, 1, 2, 3, 4, 5};
    double test2[] = {6, 7, 8, 9, 10, 11}; 
    double expected_data1[] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
    double expected_data2[] = {6, 7, 9, 7, 8, 10, 9, 10, 11};
    matrix expected1 = {expected_data1, 3, 3};
    matrix expected2 = {expected_data2, 3, 3};
    matrix* return1 = tensor_lower_triangular(test1);
    matrix* return2 = tensor_lower_triangular(test2);
    CU_ASSERT(mat_compare(&expected1, return1, .00001) == true);
    CU_ASSERT(mat_compare(&expected1, return2, .000001) == false);
    CU_ASSERT(mat_compare(&expected2, return2, .000001) == true);
    free_matrix(return1);
    free_matrix(return2);
}

//Test function to scale matrices
void test_matrix_scale(void){
    double test_vector[] = {0, 1, 2, 3, 4, 5};
    double test1_data[] = {6, 7, 8, 9, 10, 11};
    double test2_data[] = {12, 13, 14, 15, 16, 17, 0 , 1, 2, 3, 4, 5};
    matrix test_mat1 = {test1_data, 1, 6};
    matrix test_mat2 = {test1_data, 6, 1};
    matrix test_mat3 = {test2_data, 2, 6};
    matrix test_mat3t = {test2_data, 6, 2};
    double result1_data[] = {0, 7, 16, 27, 40, 55};
    matrix result_mat1 = {result1_data, 1, 6};
    matrix result_mat2 = {result1_data, 6, 1};
    double result2_data[] = {0, 13, 28, 45, 64, 85, 0, 1, 4, 9, 16, 25};
    matrix result_mat3 = {result2_data, 2, 6};
    double result2t_data[] = {0, 0, 14, 15, 32, 34, 0, 3, 8, 12, 20, 25};
    matrix result_mat3t = {result2t_data, 6, 2};
    matrix* return1 = matrix_scale(&test_mat1, test_vector, 0);
    matrix* return2 = matrix_scale(&test_mat2, test_vector, 1);
    matrix* return3 = matrix_scale(&test_mat3, test_vector, 0);
    matrix* return3t = matrix_scale(&test_mat3t, test_vector, 1);
    CU_ASSERT(mat_compare(&result_mat1, return1, .000001) == true);
    CU_ASSERT(mat_compare(&result_mat2, return2, .000001) == true);
    CU_ASSERT(mat_compare(&result_mat3, return3, .000001) == true);
    CU_ASSERT(mat_compare(&result_mat3t, return3t, .000001) == true);
    CU_ASSERT(mat_compare(&result_mat1, return2, .000001) == false);
    CU_ASSERT(mat_compare(&result_mat3, return2, .000001) == false);
    free_matrix(return1);
    free_matrix(return2);
    free_matrix(return3);
    free_matrix(return3t);
}

//Testing function for dot wrapper
void test_matrix_dot(void){
    double test1_data[] = {1, 2, 3, 4};
    double test2_data[] = {5, 6, 7, 8, 9, 10};
    matrix test1 = {test1_data, 2, 2};
    matrix test2 = {test2_data, 2, 3};
    matrix test2b = {test2_data, 3, 2};
    double result12_data[] ={ 21, 24, 27, 47, 54, 61};
    matrix result12 = {result12_data, 2, 3};
    double result21_data[] = {23, 34, 31, 46, 39, 58};
    matrix result21 = {result21_data, 3, 2};
    double result11_data[] = { 7, 10, 15, 22};
    matrix result11 = {result11_data, 2, 2};
    matrix* return1 = matrix_dot(&test1, &test1); 
    matrix* return2 = matrix_dot(&test1, &test2); 
    matrix* return3 = matrix_dot(&test2b, &test1); 
    CU_ASSERT(mat_compare(&result11, return1, .000001) == true);
    CU_ASSERT(mat_compare(&result12, return2, .000001) == true);
    CU_ASSERT(mat_compare(&result21, return3, .000001) == true);
    CU_ASSERT(mat_compare(&result21, return2, .000001) == false);
    free_matrix(return1);
    free_matrix(return2);
    free_matrix(return3);
}

//Testing function for tensor decomposition
void test_decompose_tensor(void){
    const double min_diffusitivity = 0;
    double test1_data[] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
    double test2_data[] = {5, 6, 8, 6, 7, 9, 8, 9, 10};
    matrix test1 = {test1_data, 3, 3};
    matrix test2 = {test2_data, 3, 3};
    double vals1[] = {8.82572109571665, 0, 0};
    double vecs1_data[] = {-0.32779753,  0.60627778,  0.72455229,
                             -0.51295123, -0.75825246,  0.40241053,
                             -0.79336613,  0.23975081, -0.55954422};
    matrix vecs1 = {vecs1_data, 3, 3};
    tensor tensor1 = {vals1, &vecs1};
    double vals2[] = {23.12613876, 0, 0};
    double vecs2_data[] = {-0.48266957,  0.6524927,   0.58419463,
                             -0.55684965, -0.74348717,  0.37033133,
                             -0.6759797,   0.14656091, -0.72219896};
    matrix vecs2 = {vecs2_data, 3, 3};
    tensor tensor2 = {vals2, &vecs2};
    tensor* return1 = decompose_tensor(&test1, min_diffusitivity);
    tensor* return2 = decompose_tensor(&test2, min_diffusitivity);
    CU_ASSERT(compare_tensors(&tensor1, return1, .00001) == true);
    CU_ASSERT(compare_tensors(&tensor2, return2, .00001) == true);
    CU_ASSERT(compare_tensors(&tensor1, return2, .00001) == false);
    free_tensor(return1);
    free_tensor(return2);
}

//Init stub for opt tests
int init_opt(void){
    return 0;
}

//Clean stuf for tests
int clean_opt(void){
    return 0;
}


//Main test function
int main(){
    CU_pSuite utility_suite = NULL;
    CU_pSuite opt_suite = NULL;
    CU_pSuite cuda_suite = NULL;

    if (CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();

    utility_suite = CU_add_suite("Utility Suite", init_utility, clean_utility);
    if (NULL == utility_suite){
        CU_cleanup_registry();
        return CU_get_error();
    }

    if ((NULL == CU_add_test(utility_suite, "Array comparison test", test_compare_array)) || 
            (NULL == CU_add_test(utility_suite, "Matrix comparison test", test_compare_matrix)) ||
            (NULL == CU_add_test(utility_suite, "Columnar Eig Compare test", test_columnar_eig_compare)) ||
            (NULL == CU_add_test(utility_suite, "Tensor comparison test", test_compare_tensors)) ||
            (NULL == CU_add_test(utility_suite, "Cutoff and logarithm test", test_cutoff_log)) ||
            (NULL == CU_add_test(utility_suite, "Array exp test", test_exp_array)) ||
            (NULL == CU_add_test(utility_suite, "Tensor lower triangular test", test_tensor_lower_triangular)) ||
            (NULL == CU_add_test(utility_suite, "Matrix scale test", test_matrix_scale)) ||
            (NULL == CU_add_test(utility_suite, "GSL conversion functions", test_gsl_matrix_convert)) ||
            (NULL == CU_add_test(utility_suite, "Matrix dot test", test_matrix_dot)) ||
            (NULL == CU_add_test(utility_suite, "Decompose tensor test", test_decompose_tensor))){
        CU_cleanup_registry();
        return CU_get_error();
    }

    /*opt_suite = CU_add_suite("Optimization Suite", init_opt, clean_opt);
    if (NULL == opt_suite){
        CU_cleanup_registry();
        return CU_get_error();
    }*/

    /*if (){
        CU_cleanup_registry();
        return CU_get_error();
    }*/

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}
