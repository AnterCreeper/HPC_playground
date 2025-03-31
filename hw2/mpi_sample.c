#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <immintrin.h>
#pragma GCC target("avx2")

int32_t *matrix;
int32_t *result_matrix;
int rows = 6400;
int cols = 6400;

__attribute__((aligned(64))) int32_t wsrc0[400];
__attribute__((aligned(64))) int32_t wsrc1[400];
__attribute__((aligned(64))) int32_t wdst[200];

void matrix_reduction(__m256i* result_matrix, __m256i* matrix[2]) {
	for(int i = 0; i < 25; i++) {
		__m256i a, b, c, d;
		a = matrix[0][i];
		b = matrix[1][i];
		c = matrix[0][i+25];
		d = matrix[1][i+25];
		a = _mm256_add_epi32(a, b);
		c = _mm256_add_epi32(c, d);
		d = _mm256_add_epi32(a, c);
		_mm256_stream_si256(&result_matrix[i], d);
	}
	return;
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;

    // 初始化 MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // 检查是否提供了文件路径参数
        if (argc < 2) {
            printf("Usage: %s <input_file>\n", argv[0]);
            return 1;  // 退出程序，返回错误码
        }
        
        // 从命令行参数中获取输入文件路径
        const char *input_file = argv[1];

        // 为矩阵分配内存
        matrix = (int32_t *)malloc(rows * cols * sizeof(int32_t));
        result_matrix = (int32_t *)malloc(rows / 2 * cols / 2 * sizeof(int32_t));

        // 从二进制文件中读取矩阵数据
        FILE *file = fopen(input_file, "rb");
        if (file != NULL) {
            fread(matrix, sizeof(int32_t), rows * cols, file);
            fclose(file);
            printf("Matrix has been read from random_matrix.bin\n");
        } else {
            printf("Error opening file for reading\n");
        }
    }

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // 记录开始时间
    start_time = MPI_Wtime();

	if (rank == 0) {
        MPI_Request request0[32];
        MPI_Request request1[32];
        int32_t* dst[32];
        memset(dst, 0, sizeof(dst));

        int proc = 1;
		for(int i = 0; i < 8; i++) {
			for(int j = 0; j < 400; j++) {
				for(int k = 0; k < 16; k++) {
                    int32_t* src0 = &matrix[i*800*6400+j*6400+k*400];
                    int32_t* src1 = &matrix[i*800*6400+j*6400+k*400+400*6400];
                    if(dst[proc]) {
                        MPI_Wait(&request0[proc], MPI_STATUS_IGNORE);
                        MPI_Wait(&request1[proc], MPI_STATUS_IGNORE);
                        MPI_Recv(dst[proc], 200, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    MPI_Isend(src0, 400, MPI_INT, proc, 0, MPI_COMM_WORLD, &request0[proc]);
                    MPI_Isend(src1, 400, MPI_INT, proc, 1, MPI_COMM_WORLD, &request1[proc]);
                    dst[proc] = &result_matrix[(i*400+j)*3200+k*200];
                    proc = proc == size - 1 ? 1 : proc + 1;
                }
			}
		}
		for(int proc = 1; proc < size; proc++) {
            if(dst[proc]) {
                MPI_Wait(&request0[proc], MPI_STATUS_IGNORE);
                MPI_Wait(&request1[proc], MPI_STATUS_IGNORE);
                MPI_Recv(dst[proc], 200, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        MPI_Request request;
        int start = rank - 1;
        int stripe = size - 1;
        for(int i = start; i < 51200; i = i + stripe) {
            __m256i* wsrc[2] = {(__m256i*)wsrc0, (__m256i*)wsrc1};
            MPI_Recv(wsrc0, 400, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wsrc1, 400, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            matrix_reduction((__m256i*)&wdst, wsrc);
            MPI_Send(wdst, 200, MPI_INT, 0, 2, MPI_COMM_WORLD);
        }
    }

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // 记录结束时间
    end_time = MPI_Wtime();

    // 计算并输出总时间（以 rank 0 为准）
    if (rank == 0) {
        double total_time = end_time - start_time;
        printf("Total time: %f seconds\n", total_time);
        // 将矩阵写入二进制文件
        FILE *file = fopen("result_matrix.bin", "wb");
        if (file != NULL) {
            fwrite(result_matrix, sizeof(int32_t), rows / 2 * cols / 2, file);
            fclose(file);
            printf("Matrix has been written to result_matrix.bin\n");
        } else {
            printf("Error opening file for writing\n");
        }
        // 释放内存
        free(matrix);
        free(result_matrix);
    }
    MPI_Finalize();

    return 0;
}
