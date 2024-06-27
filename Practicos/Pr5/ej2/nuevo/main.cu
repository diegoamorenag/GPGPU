#include "mmio.h"
#define WARP_PER_BLOCK 32
#define WARP_SIZE 32
#define CUDA_CHK(call) print_cuda_state(call);
#define MAX(A,B)        (((A)>(B))?(A):(B))
#define MIN(A,B)        (((A)<(B))?(A):(B))

#include <thrust/copy.h> 
#include <cub/cub.cuh>

static inline void print_cuda_state(cudaError_t code){

   if (code != cudaSuccess) printf("\ncuda error: %s\n", cudaGetErrorString(code));
   
}

__global__ void kernel_analysis_L(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	unsigned int* levels) {
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
					my_level = max(my_level, levels[colidx]);
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
		levels[wrp] = 1 + my_level;

		__threadfence();

		is_solved[wrp] = 1;
	}
}

    int* RowPtrL_d, *ColIdxL_d;
    VALUE_TYPE* Val_d;

/* Transformaciones que usamos */

//calcular el nviel
struct TransNivel {
    int* levels;
    int* rowptr;
    TransNivel(int* levels, int* rowptr) : levels(levels), rowptr(rowptr) {}

    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        int size;
        int nnz_row = rowptr[i+1]-rowptr[i]-1;
        int lev = levels[i]-1;
        if (nnz_row == 0){
            size = 6;
        } else if (nnz_row == 1) {
            size = 0;
        } else if (nnz_row <= 2) {
            size = 1;
        } else if (nnz_row <= 4) {
            size = 2;
        } else if (nnz_row <= 8) {
            size = 3;
        } else if (nnz_row <= 16) {
            size = 4;
        }
        else size = 5;
        return 7*lev+size;
    }
};

//mappearlo a como pide la letra
struct TransMap {
    int* iorder;
    int* iter;
    TransMap(int* iter, int* iorder) : iter(iter), iorder(iorder) {}

    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        int r = iter[iorder[i]] % 7;
        int nnz_row;
        if (r < 0) {
            nnz_row = r + 7;
        } else {
            nnz_row = r;
        }

        if (nnz_row == 6) {
            return 0;
        }
        return pow(2,nnz_row);
    }
};

//asignar a los warps
struct TransAsignWarp {
    int* ivects;
    TransAsignWarp(int* ivects) : ivects(ivects) {}

    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        if (ivects[i] != 0) {
            int r = i % 7;
            int nnz_row = (r < 0) ? r + 7 : r;

            if (nnz_row == 6) {
                int a = ivects[i] / 32;
                if (ivects[i] % 32 != 0) {
                    a++;
                }
                return a;
            } else if (nnz_row == 5) {
                return ivects[i];
            } else {
                int cant_ncv = ivects[i] * pow(2, nnz_row + 1);
                int a = cant_ncv / 32;
                if (cant_ncv % 32 != 0) {
                    a++;
                }
                return a;
            }
        }
        return 0;
    }
};


int ordenar_filas(int* RowPtrL, int* ColIdxL, VALUE_TYPE* Val, int n, int* iorder) {
    auto start_total = std::chrono::high_resolution_clock::now();
    int* levels = (int*)malloc(n * sizeof(int));
    unsigned int* d_levels;
    int* d_is_solved;

    CUDA_CHK(cudaMalloc((void**)&d_levels, n * sizeof(unsigned int)));
    CUDA_CHK(cudaMalloc((void**)&d_is_solved, n * sizeof(int)));

    int thread_count = WARP_PER_BLOCK * WARP_SIZE;
    int grid = ceil((double)n * WARP_SIZE / (double)(thread_count));

    CUDA_CHK(cudaMemset(d_is_solved, 0, n * sizeof(int)));
    CUDA_CHK(cudaMemset(d_levels, 0, n * sizeof(unsigned int)));
    
    auto start_kernel_analysis = std::chrono::high_resolution_clock::now();
    kernel_analysis_L<<<grid, thread_count, WARP_PER_BLOCK * (2 * sizeof(int))>>>(RowPtrL, ColIdxL, d_is_solved, n, d_levels);
    auto end_kernel_analysis = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_analysis_duration = end_kernel_analysis - start_kernel_analysis;
    std::cout << "Tiempo de kernel: " << kernel_analysis_duration.count() << " segundos" << std::endl;


    CUDA_CHK(cudaMemcpy(levels, d_levels, n * sizeof(int), cudaMemcpyDeviceToHost));

    /*Paralelice a partir de aquí*/
    auto parallel_time = std::chrono::high_resolution_clock::now();


    int* d_input = nullptr;
    int* d_output = nullptr;

    int* nLevsArr = new int[1];

    CUDA_CHK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_output, 1 * sizeof(int)));
    
    CUDA_CHK(cudaMemcpy(d_input, levels, n * sizeof(int), cudaMemcpyHostToDevice));
    
    auto start_device_reduce = std::chrono::high_resolution_clock::now();
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHK(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output, n));
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHK(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output, n));
    CUDA_CHK(cudaMemcpy(nLevsArr, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    auto end_device_reduce = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> device_reduce_duration = end_device_reduce - start_device_reduce;
    std::cout << "Tiempo de reduccion: " << device_reduce_duration.count() << " segundos" << std::endl;

    int nLevs = nLevsArr[0];
    int* RowPtrL_h = (int*)malloc((n + 1) * sizeof(int));
    CUDA_CHK(cudaMemcpy(RowPtrL_h, RowPtrL, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    int* ivects = (int*)calloc(7 * nLevs, sizeof(int));
    int* ivect_size = (int*)calloc(n, sizeof(int));
    int* index = (int*)malloc(n * sizeof(int));
    int* index2 = (int*)malloc(7 * nLevs * sizeof(int));

    for (int i = 0; i < n; i++) {
        index[i] = i;
        index2[i] = i;
    }

    for (int i = n; i < 7 * nLevs; i++) {
        index2[i] = i;
    }

    TransNivel transform(levels, RowPtrL_h);
    auto itr = cub::TransformInputIterator<int, TransNivel, int*>(index, transform);
    int* d_itr;
    int* d_ivects;
    int num_levels = 7 * nLevs + 1;
    float lower_level = 0;
    float upper_level = 7 * nLevs;
    int* itr2 = new int[n * sizeof(int)];

    thrust::copy(itr, itr + n, itr2);

    CUDA_CHK(cudaMalloc(&d_itr, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_ivects, 7 * nLevs * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_itr, itr2, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemset(d_ivects, 0, 7 * nLevs * sizeof(int)));

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_itr, d_ivects, num_levels, lower_level, upper_level, n);
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_itr, d_ivects, num_levels, lower_level, upper_level, n);
    CUDA_CHK(cudaMemcpy(ivects, d_ivects, 7 * nLevs * sizeof(int), cudaMemcpyDeviceToHost));

    int* ivectsAux = new int[n * sizeof(int)];
    thrust::copy(ivects, ivects + 7 * nLevs, ivectsAux);

    int length = 7 * nLevs;
    CUDA_CHK(cudaMalloc(&d_input, length * sizeof(int))); //ver si hay que declara explicitamente
    CUDA_CHK(cudaMalloc(&d_output, length * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_input, ivects, length * sizeof(int), cudaMemcpyHostToDevice));

    auto start_scan = std::chrono::high_resolution_clock::now();
    temp_storage_bytes = 0;
    d_temp_storage = nullptr;
    CUDA_CHK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, length));
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, length));
    auto end_scan = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> scan_duration = end_scan - start_scan;
    std::cout << "Tiempo de escaneo: " << scan_duration.count() << " segundos" << std::endl;


    CUDA_CHK(cudaMemcpy(ivects, d_output, 7 * nLevs * sizeof(int), cudaMemcpyDeviceToHost));

    int* d_keys_in = nullptr;
    int* d_keys_out = nullptr;
    int* d_values_in = nullptr;
    int* d_values_out = nullptr;

    CUDA_CHK(cudaMalloc(&d_keys_in, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_keys_out, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_values_in, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_values_out, n * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_keys_in, itr2, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_values_in, index, n * sizeof(int), cudaMemcpyHostToDevice));

    auto start_sort = std::chrono::high_resolution_clock::now();
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
    auto end_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sort_duration = end_sort - start_sort;
    std::cout << "Tiempo para ordenar: " << sort_duration.count() << " segundos" << std::endl;

    CUDA_CHK(cudaMemcpy(iorder, d_values_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    TransMap transform2(itr2, iorder);
    cub::TransformInputIterator<int, TransMap, int*> itr3(index, transform2);
    thrust::copy(itr3, itr3 + n, ivect_size);

    TransAsignWarp transform3(ivectsAux);
    cub::TransformInputIterator<int, TransAsignWarp, int*> itr4(index2, transform3);

    int* itr4aux = new int[n * sizeof(int)];
    thrust::copy(itr4, itr4 + 7 * nLevs, itr4aux);

    int num = 7 * nLevs;
    int* d_in;
    int* d_out;

    CUDA_CHK(cudaMalloc(&d_in, 7 * nLevs * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_in, itr4aux, 7 * nLevs * sizeof(int), cudaMemcpyHostToDevice));

    auto start_reduce_sum = std::chrono::high_resolution_clock::now();
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num);
    CUDA_CHK(cudaDeviceSynchronize());
    auto end_reduce_sum = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> reduce_sum_duration = end_reduce_sum - start_reduce_sum;
    std::cout << "Tiempo de reduccion de suma: " << reduce_sum_duration.count() << " segundos" << std::endl;

    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num);
    CUDA_CHK(cudaDeviceSynchronize());

    int n_warps[1];
    CUDA_CHK(cudaMemcpy(n_warps, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    int sol = n_warps[0];

    /*Termine aquí*/
    auto end_parallel_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallel_time_dur = end_parallel_time - parallel_time;
    std::cout << "Tiempo de parte paralela: " << parallel_time_dur.count()*100 << " segundos" << std::endl;



    CUDA_CHK(cudaFree(d_levels));
    CUDA_CHK(cudaFree(d_is_solved));

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total - start_total;
    std::cout << "Tiempo total de ejecucion: " << total_duration.count() << " segundos" << std::endl;

    return sol;
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



    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE* f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
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

    int nwarps = ordenar_filas(RowPtrL_d,ColIdxL_d,Val_d,n,iorder);

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