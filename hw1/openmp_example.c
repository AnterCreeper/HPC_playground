#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#pragma GCC target("avx2")

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <time.h>

static double timestamp;

#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp) {
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year  = wtm.wYear - 1900;
    tm.tm_mon   = wtm.wMonth - 1;
    tm.tm_mday  = wtm.wDay;
    tm.tm_hour  = wtm.wHour;
    tm.tm_min   = wtm.wMinute;
    tm.tm_sec   = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

double get_time() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

void start_perf() {
    timestamp=get_time();
}

void end_perf(const char* str) {
    timestamp=get_time()-timestamp;
    printf("PERF: %s times cost: %.6lfs\n", str, timestamp);
}

#define CACHE_LINE_SIZE 64
#define VECTOR_SIZE (256/8/sizeof(int32_t))

void matrix_reduction(int32_t* result_matrix, int32_t* mid_matrix, int32_t* matrix) {
#pragma omp parallel num_threads(8)
	{
	int n = omp_get_thread_num();
	__m256i* src = (__m256i*)&matrix[n*800*6400];
	__m256i* dst = (__m256i*)&mid_matrix[n*400*6400];
	for(int i = 0; i < 400*6400/VECTOR_SIZE; i++) {
		__m256i a = src[i];
		__m256i b = src[i+400*6400/VECTOR_SIZE];
		__m256i c = _mm256_add_epi32(a, b);
		_mm256_stream_si256(&dst[i], c);
	}
	}
#pragma omp parallel num_threads(16)
	{
	int n = omp_get_thread_num();
	__m256i* src0 = (__m256i*)&mid_matrix[n*400];
	__m256i* dst0 = (__m256i*)&result_matrix[n*200];
	for(int i = 0; i < 3200; i++) {
		__m256i* src = &src0[i*6400/VECTOR_SIZE];
		__m256i* dst = &dst0[i*3200/VECTOR_SIZE];
		for(int j = 0; j < 200/VECTOR_SIZE; j++) {
			__m256i a = src[j];
			__m256i b = src[j+200/VECTOR_SIZE];
			__m256i c = _mm256_add_epi32(a, b);
			_mm256_stream_si256(&dst[j], c);
		}
	}
	}
	return;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];

    int rows = 6400;
    int cols = 6400;
    int32_t *matrix = (int32_t *)aligned_alloc(CACHE_LINE_SIZE, rows * cols * sizeof(int32_t));
    int32_t *mid_matrix = (int32_t *)aligned_alloc(CACHE_LINE_SIZE, rows * cols / 2 * sizeof(int32_t));
    int32_t *result_matrix = (int32_t *)aligned_alloc(CACHE_LINE_SIZE, rows / 2 * cols / 2 * sizeof(int32_t));

    FILE *file = fopen(input_file, "rb");
    if (file != NULL) {
        fread(matrix, sizeof(int32_t), rows * cols, file);
        fclose(file);
        printf("Matrix has been read from random_matrix.bin\n");
    } else {
        printf("Error opening file for reading\n");
    }

    start_perf();
    matrix_reduction(result_matrix, mid_matrix, matrix);
    end_perf("openMP or MPI");

    file = fopen("result_matrix.bin", "wb");
    if (file != NULL) {
        fwrite(result_matrix, sizeof(int32_t), rows / 2 * cols / 2, file);
        fclose(file);
        printf("Matrix has been written to result_matrix.bin\n");
    } else {
        printf("Error opening file for writing\n");
    }

    free(matrix);
    free(result_matrix);

    return 0;
}
