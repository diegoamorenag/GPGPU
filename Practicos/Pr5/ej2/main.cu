#include "mmio.h"
#define WARP_PER_BLOCK 32
#define WARP_SIZE 32
#define CUDA_CHK(call) print_cuda_state(call);
#define MAX(A,B)        (((A)>(B))?(A):(B))
#define MIN(A,B)        (((A)<(B))?(A):(B))

static inline void print_cuda_state(cudaError_t code){

   if (code != cudaSuccess) printf("\ncuda error: %s\n", cudaGetErrorString(code));
   
}

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
    // Variables CUDA
    unsigned int *d_niveles;
    int *d_is_solved;

    // Allocate memory and initialize data
    cudaMalloc((void**)&d_niveles, n * sizeof(unsigned int));
    cudaMalloc((void**)&d_is_solved, n * sizeof(int));
    cudaMemset(d_is_solved, 0, n * sizeof(int));
    cudaMemset(d_niveles, 0, n * sizeof(unsigned int));

    // Setup kernel execution parameters
    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    // Run the level analysis kernel
    kernel_analysis_L<<<num_blocks, num_threads>>>(RowPtrL, ColIdxL, d_is_solved, n, d_niveles);

    // Copy levels back to host (for demonstration purposes, normally you'd keep this on device for further processing)
    int* niveles = new int[n];
    cudaMemcpy(niveles, d_niveles, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Thrust device vector for sorting and assigning
    thrust::device_vector<int> d_levels(d_niveles, d_niveles + n);
    thrust::device_vector<int> d_iorder(n);
    thrust::sequence(thrust::device, d_iorder.begin(), d_iorder.end());

    // Sort by levels using Thrust
    thrust::sort_by_key(d_levels.begin(), d_levels.end(), d_iorder.begin());

    // Copy order back to host
    thrust::copy(d_iorder.begin(), d_iorder.end(), iorder);

    // Clean up
    cudaFree(d_niveles);
    cudaFree(d_is_solved);
    delete[] niveles;

    return 0;  // Change this to return the number of warps if needed
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
