#include <iostream>
#include "matrix_op.h"

int main(int argc, char** argv) {
	size_t N;
	size_t M;
	if (argc != 3) {
		N = 2048;
		M = 10;
	}
	else {
		N = atoll(argv[1]);
		M = atoll(argv[2]);
		if (N == 0 or M == 0) {
			N = 2048;
			M = 10;
		}
	}
	float* A = new float[N * N];
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			A[i * N + j] = rand() % 1024;
		}
	}
	clock_t start, end;
	start = clock();
	float* A_R = reverse_matrix(A, N, M);
	end = clock();
	float time = ((float)end - start) / CLOCKS_PER_SEC;
	std::cout << "Time: " << time << std::endl;
	float* check = new float[N * N];
	mult_matrix(A, A_R, check, N);
	if (N <= 12) {
		print_matrix(check, N);
	}
	else {
		for (size_t i = 0; i < N; ++i) {
			std::cout << check[i * N + i] << " ";
		}
	}
	delete[] check;
	delete[] A;
	delete[] A_R;
	return 0;
}