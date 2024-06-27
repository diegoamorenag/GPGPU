#include "mmio.h"
#include <cub/cub.cuh>
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

struct TransformarNiveles {
    int* rowPtr;
    int* levels;

    TransformarNiveles( int* levels, int* rowPtr) : rowPtr(rowPtr), levels(levels) {}
    __host__ __device__ __forceinline__
    int operator()(const int  & i ) const {
        int  lev   = levels [i]-1;
        int filaNNZ = rowPtr [i+1]-rowPtr [ i ]-1;
        int tamanioArray;

        if (filaNNZ  == 0)
            tamanioArray  = 6;
        else if (filaNNZ ==  1)
            tamanioArray =  0;
        else if (filaNNZ <= 2)
            tamanioArray = 1; 
        else if ( filaNNZ <= 4)
            tamanioArray = 2 ; 
        else if (filaNNZ <= 8)
            tamanioArray = 3;
        else if ( filaNNZ <=  16)
            tamanioArray = 4 ;
        else tamanioArray = 5; 
        return 7* lev + tamanioArray;
    }
};

struct TransformarTamanio {
    int* orden;
    int* tamFila;

    TransformarTamanio(int* orden, int* tamFila) : tamFila(tamFila), orden(orden) {}
    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        int r =  tamFila[ orden[i]]  % 7; 
        int filaNNZ =  (r < 0) ?  r + 7 : r ;
        return ( filaNNZ == 6)? 0 : pow(2,filaNNZ);;
    }
};

struct CalcularWarps {
    int* vectorI;
    CalcularWarps(int* vectorI) : vectorI(vectorI) {}

    __host__ __device__ __forceinline__
    int operator()(const int &i) const {
        if (vectorI [i] != 0) {
            int r  = i  % 7;
            int  filaNNZ = (r < 0) ? r + 7 : r;
            if ( filaNNZ == 5){
                return vectorI[i];
            } else if (filaNNZ == 6) {
                int a = vectorI[i] / 32;
                 if (vectorI[ i] % 32  != 0)  a++;
                return a; 
            } else {
                int cant  = vectorI[i] *  pow( 2 , filaNNZ + 1);
                int a = cant / 32 ;
                if (cant  % 32  != 0) a++;
                return a;}
        }
        return 0;}
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

int ordenar_filas( int* RowPtrL, int* ColIdxL, VALUE_TYPE * Val, int n, int* iorder){
    int * niveles;
    niveles = (int*) malloc(n * sizeof(int));

    unsigned int * d_niveles;
    int * d_is_solved;
    
    CUDA_CHK( cudaMalloc((void**) &(d_niveles) , n * sizeof(unsigned int)) )
    CUDA_CHK( cudaMalloc((void**) &(d_is_solved) , n * sizeof(int)) )
    
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;

    int grid = ceil ((double)n*WARP_SIZE / (double)(num_threads));

    CUDA_CHK( cudaMemset(d_is_solved, 0, n * sizeof(int)) )
    CUDA_CHK( cudaMemset(d_niveles, 0, n * sizeof(unsigned int)) )

    kernel_analysis_L<<< grid , num_threads, WARP_PER_BLOCK * (2*sizeof(int)) >>>( RowPtrL, 
                                                                                   ColIdxL, 
                                                                                   d_is_solved, 
                                                                                   n, 
                                                                                   d_niveles);

    CUDA_CHK( cudaMemcpy(niveles, d_niveles, n * sizeof(int), cudaMemcpyDeviceToHost) )

    printf("------------------------------------------DEBUG: 1---------------------------------------------\n");
    for (int  y = 0;  y < n; ++ y) {
        std::cout << "niveles[" <<  y << "]: " << niveles[ y] << std::endl;
    }
    printf("------------------------------------------DEBUG: 2---------------------------------------------\n");
    int* device_output = nullptr;
    int* device_input = nullptr;
    int* nLevsArr = new int[1];

    CUDA_CHK(cudaMalloc(&device_output, 1 * sizeof(int)));
    CUDA_CHK(cudaMalloc(&device_input, n * sizeof(int)));
    CUDA_CHK(cudaMemcpy(device_input, niveles, n * sizeof(int), cudaMemcpyHostToDevice));

    size_t tmp_bytes = 0;
    void* tmp_storage = nullptr;
    CUDA_CHK( cub::DeviceReduce::Max( tmp_storage, tmp_bytes, device_input, device_output, n)); 
    cudaMalloc( &tmp_storage, tmp_bytes);
    CUDA_CHK( cub::DeviceReduce::Max(tmp_storage, tmp_bytes, device_input, device_output, n));
    CUDA_CHK( cudaMemcpy( nLevsArr, device_output, sizeof(int), cudaMemcpyDeviceToHost));
    int nLevs = nLevsArr[0],* RowPtrL_h = (int *) malloc( (n+1) * sizeof(int) );
    CUDA_CHK( cudaMemcpy(RowPtrL_h, RowPtrL, (n+1) * sizeof(int), cudaMemcpyDeviceToHost) );

    int * ivects = (int *) calloc( 7*nLevs, sizeof(int) ), * ivect_size  = (int *) calloc(n,sizeof(int));
    for (int  y = 0;  y < n+1;  y++) {
        std::cout << "RowPtrL_h[" << y << "]: " << RowPtrL_h[y] << std::endl;
    }
    printf("------------------------------------------DEBUG: 3---------------------------------------------\n");

    int* x= (int*)malloc(n*sizeof(int));
    int*  y=(int*)malloc(7*nLevs*sizeof(int));

    for (int i = 0; i < n; i++){
        x[i] = i;
         y[i] = i;
    }
    for (int i = n; i < 7*nLevs; i++){
         y[i] = i;
    }    

    TransformarNiveles transform(niveles, RowPtrL_h);
    auto itr = cub::TransformInputIterator<int, TransformarNiveles, int*>(x, transform);

    int* ditr,*d_ivects, num_levels = 7*nLevs + 1,* itr2 = new int[n * sizeof(int)];
    float lowlevel = 0,uplevel = 7*nLevs; 
    thrust::copy(itr, itr + n, itr2);

    CUDA_CHK(cudaMalloc(&ditr, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_ivects, 7*nLevs * sizeof(int)));

    CUDA_CHK(cudaMemcpy(ditr, itr2, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemset(d_ivects, 0, 7*nLevs * sizeof(int)));
    tmp_storage = nullptr;
    tmp_bytes = 0;

    cub::DeviceHistogram::HistogramEven(
    tmp_storage, tmp_bytes,
    ditr, d_ivects, num_levels,
    lowlevel, uplevel, n);

    // Allocate temporary storage
    CUDA_CHK(cudaMalloc(&tmp_storage, tmp_bytes));

    // Compute histograms
    cub::DeviceHistogram::HistogramEven(
    tmp_storage, tmp_bytes,
    ditr, d_ivects, num_levels,
    lowlevel, uplevel, n);

    CUDA_CHK( cudaMemcpy(ivects, d_ivects, 7*nLevs * sizeof(int), cudaMemcpyDeviceToHost) )

    int* ivectsAux = new int[n * sizeof(int)];
    thrust::copy(ivects, ivects + 7*nLevs, ivectsAux);

    for (int  y = 0;  y < 7*nLevs;  y++) {
        printf("itr2[%d]: %d\n",  y, itr2[ y]);
    }
    for (int  y = 0;  y < 7*nLevs;  y++) {
        printf("ivects[%d]: %d\n",  y, ivects[ y]);
    }
    printf("------------------------------------------DEBUG: 4---------------------------------------------\n");

    int lar = 7*nLevs;

    device_input = nullptr;
    device_output = nullptr;

    CUDA_CHK(cudaMalloc(&device_input, lar * sizeof(int)));
    CUDA_CHK(cudaMalloc(&device_output, lar * sizeof(int)));
    CUDA_CHK(cudaMemcpy(device_input, ivects, lar * sizeof(int), cudaMemcpyHostToDevice));

    tmp_storage = nullptr;
    tmp_bytes = 0;
    CUDA_CHK(cub::DeviceScan::ExclusiveSum(tmp_storage, tmp_bytes, device_input, device_output, lar));  // GPUassert: invalid device function example.cu
    cudaMalloc(&tmp_storage, tmp_bytes);
    CUDA_CHK(cub::DeviceScan::ExclusiveSum(tmp_storage, tmp_bytes, device_input, device_output, lar));

    CUDA_CHK(cudaMemcpy(ivects, device_output, 7*nLevs * sizeof(int), cudaMemcpyDeviceToHost));

    for (int  y = 0;  y < 7*nLevs;  y++) {
        printf("ivects[%d]: %d\n",  y, ivects[ y]);
    }
    printf("------------------------------------------DEBUG: 5---------------------------------------------\n");
    
    int  *d_keys_in = nullptr;
    int  *d_keys_out = nullptr;
    int  *d_values_in = nullptr;
    int  *d_values_out = nullptr;

    CUDA_CHK(cudaMalloc(&d_keys_in, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_keys_out, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_values_in, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_values_out, n * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_keys_in, itr2, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_values_in, x, n * sizeof(int), cudaMemcpyHostToDevice));

    tmp_storage = nullptr;
    tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(tmp_storage, tmp_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, n);

    CUDA_CHK(cudaMalloc(&tmp_storage, tmp_bytes));

    cub::DeviceRadixSort::SortPairs(tmp_storage, tmp_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, n);

    CUDA_CHK(cudaMemcpy(iorder, d_values_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    TransformarTamanio transform2(iorder, itr2);
    cub::TransformInputIterator<int, TransformarTamanio, int*> itr3(x, transform2);    
    thrust::copy(itr3, itr3 + n, ivect_size);

    for (int  y = 0;  y < n;  y++) {
        printf("iorder[%d]: %d\n",  y, iorder[ y]);
    }
    for (int  y = 0;  y < n;  y++) {
        printf("ivect_size[%d]: %d\n",  y, ivect_size[ y]);
    }
    for (int  y = 0;  y < 7*nLevs;  y++) {
        printf("ivects[%d]: %d\n",  y, ivects[ y]);
    }
    printf("------------------------------------------DEBUG: 6---------------------------------------------\n");

    int ii = 1;
    int filas_warp = 1;

    CalcularWarps transform3(ivectsAux);
    cub::TransformInputIterator<int, CalcularWarps, int*> itr4( y, transform3);

    int* itr4aux = new int[n * sizeof(int)];
    thrust::copy(itr4, itr4 + 7*nLevs, itr4aux);

    int num = 7*nLevs;
    int *d_in;          // [8, 6, 7, 5, 3, 0, 9]
    int *d_out;         // [-]

    CUDA_CHK(cudaMalloc(&d_in, 7*nLevs * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_in, itr4aux, 7*nLevs * sizeof(int), cudaMemcpyHostToDevice));

    // Determine temporary device storage requirements
    tmp_storage = nullptr;
    tmp_bytes = 0;
    cub::DeviceReduce::Sum(tmp_storage, tmp_bytes, d_in, d_out, num);
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMalloc(&tmp_storage, tmp_bytes));

    cub::DeviceReduce::Sum(tmp_storage, tmp_bytes, d_in, d_out, num);
    CUDA_CHK(cudaDeviceSynchronize());

    int n_warps[1];
    CUDA_CHK(cudaMemcpy(n_warps, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    for (int  y = 0;  y < 7*nLevs;  y++) {
        printf("itr4[%d]: %d\n",  y, itr4[ y]);
    }
    printf("n_warps[%d]: %d\n", 0, n_warps[0]);
    printf("------------------------------------------DEBUG: 6---------------------------------------------\n");
    
    int sol = n_warps[0];
    CUDA_CHK( cudaFree(d_niveles) ) 
    CUDA_CHK( cudaFree(d_is_solved) ) 

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

    /* find device_output size of sparse matrix .... */
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

    /* NOTE: when reading device_input doubles, ANSI C requires the use of the "l"  */
    /*   specifier as device_input "%lg", "%lf", "%le", otherwise errors will occur */
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
