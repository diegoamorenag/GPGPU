#include "mmio.h"
#include <cub/cub.cuh>
#include <thrust/copy.h> 
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>


#define WARP_PER_BLOCK 32
#define WARP_SIZE 32
#define CUDA_CHK(call) print_cuda_state(call);
#define MAX(A,B)        (((A)>(B))?(A):(B))
#define MIN(A,B)        (((A)<(B))?(A):(B))

static inline void print_cuda_state(cudaError_t code){
   if (code != cudaSuccess) printf("\ncuda error: %s\n", cudaGetErrorString(code));
}

struct Trans_niveles {
    int* levels;
    int* rowptr;

    Trans_niveles(int* levels, int* rowptr) : levels(levels), rowptr(rowptr) {}

    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        int level = levels[i] - 1;
        int nnz_row = rowptr[i + 1] - rowptr[i] - 1;
        int vector_size;

        if (nnz_row == 0)
            vector_size = 6;
        else if (nnz_row == 1)
            vector_size = 0;
        else if (nnz_row <= 2)
            vector_size = 1;
        else if (nnz_row <= 4)
            vector_size = 2;
        else if (nnz_row <= 8)
            vector_size = 3;
        else if (nnz_row <= 16)
            vector_size = 4;
        else
            vector_size = 5;

        return 7 * level + vector_size;
    }
};

struct Trans_mapear {
    int* itr2;
    int* iorder;

    Trans_mapear(int* itr2, int* iorder) : itr2(itr2), iorder(iorder) {}

    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        int r = itr2[iorder[i]] % 7;
        int nnz_row = (r < 0) ? r + 7 : r;
        return (nnz_row == 6) ? 0 : pow(2, nnz_row);
    }
};

struct Trans_asignar_warps {
    int* ivectsAux;

    Trans_asignar_warps(int* ivectsAux) : ivectsAux(ivectsAux) {}

    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        if (ivectsAux[i] != 0) {
            int r = i % 7;
            int nnz_row = (r < 0) ? r + 7 : r;

            if (nnz_row == 6) {
                int a = ivectsAux[i] / 32;
                if (ivectsAux[i] % 32 != 0) a++;
                return a;
            } else if (nnz_row == 5) {
                return ivectsAux[i];
            } else {
                int count = ivectsAux[i] * pow(2, nnz_row + 1);
                int a = count / 32;
                if (count % 32 != 0) a++;
                return a;
            }
        }
        return 0;
    }
};


__global__ void kernel_analysis_L(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	unsigned int* niveles) {
	extern volatile __shared__ int s_mem[];

	if(threadIdx.x==0&&blockIdx.x==0) printf("%i\n", WARP_PER_BLOCK);
	int* s_is_solved = (int*)&s_mem[0];
	int* s_info = (int*)&s_is_solved[WARP_PER_BLOCK];

	int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
	int local_warp_id = threadIdx.x / WARP_SIZE;

	int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

	if (wrp >= n) return;

	int row = row_ptr[wrp];
	int start_row = blockIdx.x * WARP_PER_BLOCK;
	int nxt_row = row_ptr[wrp + 1];

	int my_level = 0;
	if (lne == 0) {
		s_is_solved[local_warp_id] = 0;
		s_info[local_warp_id] = 0;
	}

	__syncthreads();

	int off = row + lne;
	int colidx = col_idx[off];
	int myvar = 0;

	while (off < nxt_row - 1)
	{
		colidx = col_idx[off];
		if (!myvar)
		{
			if (colidx > start_row) {
				myvar = s_is_solved[colidx - start_row];

				if (myvar) {
					my_level = max(my_level, s_info[colidx - start_row]);
				}
			} else
			{
				myvar = is_solved[colidx];

				if (myvar) {
					my_level = max(my_level, niveles[colidx]);
				}
			}
		}

		if (__all_sync(__activemask(), myvar)) {

			off += WARP_SIZE;
			//           colidx = col_idx[off];
			myvar = 0;
		}
	}
	__syncwarp();
	
	for (int i = 16; i >= 1; i /= 2) {
		my_level = max(my_level, __shfl_down_sync(__activemask(), my_level, i));
	}

	if (lne == 0) {

		s_info[local_warp_id] = 1 + my_level;
		s_is_solved[local_warp_id] = 1;
		niveles[wrp] = 1 + my_level;

		__threadfence();

		is_solved[wrp] = 1;
	}
}

    int* RowPtrL_d, *ColIdxL_d;
    VALUE_TYPE* Val_d;


int ordenar_filas(int* RowPtrL, int* ColIdxL, VALUE_TYPE* Val, int n, int* iorder) {
auto start = std::chrono::high_resolution_clock::now();

    int* levels = (int*)malloc(n * sizeof(int));
    int* row_order = (int*)malloc(n * sizeof(int));

    int* d_unsolved;
    unsigned int* d_levels;
    
    CUDA_CHK(cudaMalloc((void**)&d_levels, n * sizeof(unsigned int)));
    CUDA_CHK(cudaMalloc((void**)&d_unsolved, n * sizeof(int)));
    
    int threads_per_block = WARP_PER_BLOCK * WARP_SIZE;
    int blocks_per_grid = ceil((double)n * WARP_SIZE / (double)(threads_per_block));

    CUDA_CHK(cudaMemset(d_levels, 0, n * sizeof(unsigned int)));
    CUDA_CHK(cudaMemset(d_unsolved, 0, n * sizeof(int)));

    int shared_mem_size = WARP_PER_BLOCK * (2 * sizeof(int));

    auto start_kernel = std::chrono::high_resolution_clock::now();
    kernel_analysis_L<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(RowPtrL, ColIdxL, d_unsolved, n, d_levels);
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_duration = end_kernel - start_kernel;
    std::cout << "Tiempo de ejecucion del kernel: " << kernel_duration.count() << " segundos." << std::endl;

    auto start_copy = std::chrono::high_resolution_clock::now();
    CUDA_CHK(cudaMemcpy(levels, d_levels, n * sizeof(int), cudaMemcpyDeviceToHost));
    auto end_copy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> copy_duration = end_copy - start_copy;
    std::cout << "Tiempo de copia: " << copy_duration.count() << " segundos." << std::endl;

    int* d_input = nullptr;
    int* d_output = nullptr;
    int* max_level_holder = new int[1];

    CUDA_CHK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_output, 1 * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_input, levels, n * sizeof(int), cudaMemcpyHostToDevice));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    auto start_reduce = std::chrono::high_resolution_clock::now();
    CUDA_CHK(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output, n));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHK(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output, n));
    auto end_reduce = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> reduce_duration = end_reduce - start_reduce;
    std::cout << "Tiempo del reduce: " << reduce_duration.count() << " segundos." << std::endl;

    CUDA_CHK(cudaMemcpy(max_level_holder, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    int max_levels = max_level_holder[0];

    int* host_RowPtrL = (int*)malloc((n + 1) * sizeof(int));
    CUDA_CHK(cudaMemcpy(host_RowPtrL, RowPtrL, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    int* vector_counts = (int*)calloc(7 * max_levels, sizeof(int));
    int* vector_sizes = (int*)calloc(n, sizeof(int));

    int* idx = (int*)malloc(n * sizeof(int));
    int* idx2 = (int*)malloc(7 * max_levels * sizeof(int));
    for (int i = 0; i < n; i++) {
        idx[i] = i;
        idx2[i] = i;
    }
    for (int i = n; i < 7 * max_levels; i++) {
        idx2[i] = i;
    }

    Trans_niveles classify(levels, host_RowPtrL);
    auto transformed_idx = cub::TransformInputIterator<int, Trans_niveles, int*>(idx, classify);

    int* d_transformed_idx;
    int* d_vector_counts;
    int num_bins = 7 * max_levels + 1;
    float lower_bound = 0;
    float upper_bound = 7 * max_levels;

    int* transformed_idx_copy = new int[n * sizeof(int)];
    thrust::copy(transformed_idx, transformed_idx + n, transformed_idx_copy);

    CUDA_CHK(cudaMalloc(&d_transformed_idx, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_vector_counts, 7 * max_levels * sizeof(int)));

    CUDA_CHK(cudaMemcpy(d_transformed_idx, transformed_idx_copy, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemset(d_vector_counts, 0, 7 * max_levels * sizeof(int)));

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    auto start_histogram = std::chrono::high_resolution_clock::now();
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_transformed_idx, d_vector_counts, num_bins, lower_bound, upper_bound, n);
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_transformed_idx, d_vector_counts, num_bins, lower_bound, upper_bound, n);
    auto end_histogram = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> histogram_duration = end_histogram - start_histogram;
    std::cout << "Tiempo del histograma: " << histogram_duration.count() << " segundos." << std::endl;

    CUDA_CHK(cudaMemcpy(vector_counts, d_vector_counts, 7 * max_levels * sizeof(int), cudaMemcpyDeviceToHost));

    int* vector_counts_copy = new int[n * sizeof(int)];
    thrust::copy(vector_counts, vector_counts + 7 * max_levels, vector_counts_copy);

    int length = 7 * max_levels;

    d_input = nullptr;
    d_output = nullptr;

    CUDA_CHK(cudaMalloc(&d_input, length * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_output, length * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_input, vector_counts, length * sizeof(int), cudaMemcpyHostToDevice));

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    auto start_scan = std::chrono::high_resolution_clock::now();
    CUDA_CHK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, length));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, length));
    auto end_scan = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> scan_duration = end_scan - start_scan;
    std::cout << "Tiempo del scan: " << scan_duration.count() << " segundos." << std::endl;

    CUDA_CHK(cudaMemcpy(vector_counts, d_output, 7 * max_levels * sizeof(int), cudaMemcpyDeviceToHost));

    int* d_keys_in = nullptr;
    int* d_keys_out = nullptr;
    int* d_values_in = nullptr;
    int* d_values_out = nullptr;

    CUDA_CHK(cudaMalloc(&d_keys_in, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_keys_out, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_values_in, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_values_out, n * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_keys_in, transformed_idx_copy, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_values_in, idx, n * sizeof(int), cudaMemcpyHostToDevice));

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    auto start_sort = std::chrono::high_resolution_clock::now();
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
    auto end_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sort_duration = end_sort - start_sort;
    std::cout << "Tiempo del sort: " << sort_duration.count() << " segundos." << std::endl;

    CUDA_CHK(cudaMemcpy(row_order, d_values_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    Trans_mapear map_power(transformed_idx_copy, row_order);
    cub::TransformInputIterator<int, Trans_mapear, int*> mapped_idx(idx, map_power);
    thrust::copy(mapped_idx, mapped_idx + n, vector_sizes);

    Trans_asignar_warps assign_warps(vector_counts_copy);
    cub::TransformInputIterator<int, Trans_asignar_warps, int*> transformed_idx2(idx2, assign_warps);

    int* transformed_idx2_copy = new int[n * sizeof(int)];
    thrust::copy(transformed_idx2, transformed_idx2 + 7 * max_levels, transformed_idx2_copy);

    int* d_input2;
    int* d_output2;

    CUDA_CHK(cudaMalloc(&d_input2, 7 * max_levels * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_output2, sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_input2, transformed_idx2_copy, 7 * max_levels * sizeof(int), cudaMemcpyHostToDevice));

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    auto start_reduce_sum = std::chrono::high_resolution_clock::now();
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input2, d_output2, 7 * max_levels);
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input2, d_output2, 7 * max_levels);
    CUDA_CHK(cudaDeviceSynchronize());
    auto end_reduce_sum = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> reduce_sum_duration = end_reduce_sum - start_reduce_sum;
    std::cout << "Tiempo del reduce: " << reduce_sum_duration.count() << " segundos." << std::endl;

    int num_warps[1];
    CUDA_CHK(cudaMemcpy(num_warps, d_output2, sizeof(int), cudaMemcpyDeviceToHost));

    int result = num_warps[0];

    CUDA_CHK(cudaFree(d_levels));
    CUDA_CHK(cudaFree(d_unsolved));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end - start;
    std::cout << "Total: " << total_duration.count() << " segundos." << std::endl;

    free(row_order);

    return result;
}

int main(int argc, char** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char* precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char*)"32-bit Single Precision";
    } else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char*)"64-bit Double Precision";
    } else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);


    int m, n, nnzA;
    int* csrRowPtrA;
    int* csrColIdxA;
    VALUE_TYPE* csrValA;

    int argi = 1;

    char* filename;
    if (argc > argi)
    {
        filename = argv[argi];
        argi++;
    }

    printf("-------------- %s --------------\n", filename);

    int ret_code;
    MM_typecode matcode;
    FILE* f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_complex(matcode))
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        return -3;
    }

    char* pch, * pch1;
    pch = strtok(filename, "/");
    while (pch != NULL) {
        pch1 = pch;
        pch = strtok(NULL, "/");
    }

    pch = strtok(pch1, ".");


    if (mm_is_pattern(matcode)) { isPattern = 1; }
    if (mm_is_real(matcode)) { isReal = 1;  }
    if (mm_is_integer(matcode)) { isInteger = 1; }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;


    if (n != m)
    {
        printf("Matrix is not square.\n");
        return -5;
    }

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric = 1;
        printf("input matrix is symmetric = true\n");
    } else
    {
        printf("input matrix is symmetric = false\n");
    }

    int* csrRowPtrA_counter = (int*)malloc((m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    int* csrRowIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    int* csrColIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE* csrValA_tmp = (VALUE_TYPE*)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;
        int returnvalue;

        if (isReal)
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        } else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int*)malloc((m + 1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    csrColIdxA = (int*)malloc(nnzA * sizeof(int));
    csrValA = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            } else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    } else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }
 
    printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);

    // extract L with the unit-lower triangular sparsity structure of A
    int nnzL = 0;
    int* csrRowPtrL_tmp = (int*)malloc((m + 1) * sizeof(int));
    int* csrColIdxL_tmp = (int*)malloc(nnzA * sizeof(int));
    VALUE_TYPE* csrValL_tmp = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++)
        {
            if (csrColIdxA[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = csrColIdxA[j];
                csrValL_tmp[nnz_pointer] = 1.0; //csrValA[j];
                nnz_pointer++;
            } else
            {
                break;
            }
        }

        csrColIdxL_tmp[nnz_pointer] = i;
        csrValL_tmp[nnz_pointer] = 1.0;
        nnz_pointer++;

        csrRowPtrL_tmp[i + 1] = nnz_pointer;
    }

    nnzL = csrRowPtrL_tmp[m];
    printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzL);

    csrColIdxL_tmp = (int*)realloc(csrColIdxL_tmp, sizeof(int) * nnzL);
    csrValL_tmp = (VALUE_TYPE*)realloc(csrValL_tmp, sizeof(VALUE_TYPE) * nnzL);

    printf("---------------------------------------------------------------------------------------------\n");

    int* RowPtrL_d, *ColIdxL_d;
    VALUE_TYPE* Val_d;

    cudaMalloc((void**)&RowPtrL_d, (n + 1) * sizeof(int));
    cudaMalloc((void**)&ColIdxL_d, nnzL * sizeof(int));
    cudaMalloc((void**)&Val_d, nnzL * sizeof(VALUE_TYPE));
  
    cudaMemcpy(RowPtrL_d, csrRowPtrL_tmp, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ColIdxL_d, csrColIdxL_tmp, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Val_d, csrValL_tmp, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    int * iorder  = (int *) calloc(n,sizeof(int));

    int nwarps = ordenar_filas(RowPtrL_d, ColIdxL_d, Val_d, n, iorder);
    printf("Number of warps: %i\n",nwarps);
    for(int i =0; i<n && i<20;i++)
        printf("Iorder[%i] = %i\n",i,iorder[i]);

    printf("Bye!\n");

    // done!
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);

    return 0;
}
