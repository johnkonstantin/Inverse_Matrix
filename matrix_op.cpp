#include "matrix_op.h"

void mult_matrix(const float* A, const float* B, float* C, size_t N) {
	memset(C, 0, N * N * sizeof(float));
	for (size_t i = 0; i < N; ++i) {
		for (size_t k = 0; k < N; ++k) {
			for (size_t j = 0; j < N; ++j) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

void sum_matrix(const float* A, float* B, size_t N) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			B[i * N + j] += A[i * N + j];
		}
	}
}

void print_matrix(const float* A, size_t N) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			if (A[i * N + j] < 1e-01) {
				std::cout << 0 << " ";
			}
			else if (A[i * N + j] > 8e-01) {
				std::cout << 1 << " ";
			}
			else {
				std::cout << std::setprecision(2) << A[i * N + j] << " ";
			}
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void mult_matrix_number(float* A, float num, size_t N) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			A[i * N + j] *= num;
		}
	}
}

float* trans_matrix(const float* A, size_t N) {
	float* res = new float[N * N];
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			res[i * N + j] = A[j * N + i];
		}
	}
	return res;
}

float* reverse_matrix(const float* A, size_t N, size_t M) {
	float* B = trans_matrix(A, N);
	float A_1 = 0;
	float A_INF = 0;
	for (size_t i = 0; i < N; ++i) {
		float t1 = 0;
		float tINF = 0;
		for (size_t j = 0; j < N; ++j) {
			t1 += fabsf(B[i * N + j]);
			tINF += fabsf(A[i * N + j]);
		}
		if (t1 > A_1) {
			A_1 = t1;
		}
		if (tINF > A_INF) {
			A_INF = tINF;
		}
	}
	mult_matrix_number(B, 1 / (A_1 * A_INF), N);
	float* BA = new float[N * N];
	mult_matrix(B, A, BA, N);
	mult_matrix_number(BA, -1.0, N);
	float* R = new float[N * N];
	memset(R, 0, N * N * sizeof(float));
	for (size_t i = 0; i < N; ++i) {
		R[i * N + i] = 1.0;
	}
	sum_matrix(BA, R, N);
	float* RES = new float[N * N];
	memset(RES, 0, N * N * sizeof(float));
	for (size_t i = 0; i < N; ++i) {
		RES[i * N + i] = 1.0;
	}
	float* R_n = new float[N * N];
	memset(R_n, 0, N * N * sizeof(float));
	for (size_t i = 0; i < N; ++i) {
		R_n[i * N + i] = 1.0;
	}
	for (size_t i = 1; i < M; ++i) {
		float* t = new float[N * N];
		mult_matrix(R_n, R, t, N);
		sum_matrix(t, RES, N);
		delete[] R_n;
		R_n = t;
	}
	delete[] R_n;
	float* A_R = new float[N * N];
	mult_matrix(RES, B, A_R, N);
	delete[] B;
	delete[] RES;
	delete[] BA;
	delete[] R;
	return A_R;
}