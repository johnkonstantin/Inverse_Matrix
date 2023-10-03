#ifndef PRACT4_1_MATRIX_OP_H
#define PRACT4_1_MATRIX_OP_H

#include <iostream>
#include <cstring>
#include <iomanip>
#include <cmath>

void mult_matrix(const float* A, const float* B, float* C, size_t N);
void sum_matrix(const float* A, float* B, size_t N);
void mult_matrix_number(float* A, float num, size_t N);
float* trans_matrix(const float* A, size_t N);
float* reverse_matrix(const float* A, size_t N, size_t M);
void print_matrix(const float* A, size_t N);
#endif