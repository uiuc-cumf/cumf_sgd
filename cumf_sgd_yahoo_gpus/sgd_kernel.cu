
#include "sgd.h"
#include "sgd_kernel.h"
#include <curand.h>
#include <curand_kernel.h>

#include <cstring>

#include <unistd.h>

#include <iomanip>

using namespace std;

__device__ unsigned int update_count;

extern __global__ void init_rand_state(curandState*state, int size);
extern __global__ void init_block_lock(bool*row, bool*col, int b);


#include "sgd_k128_kernel_hogwild_warp32.h"
 

__global__ void init_rand_state(curandState*state, int size)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < size)curand_init(clock() + tid,tid,0,&state[tid]);
}


__global__ void transform_half(half *gpu_half_feature, float *gpu_float_feature, long long vec_size)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int number_threads = gridDim.x*blockDim.x;

    for(long long i = tid;i < vec_size;i += number_threads)
    {
        gpu_float_feature[i] = __half2float(gpu_half_feature[i]); 
    }
}

void transform_feature_vector(short *half_feature, float *float_feature, int m, int grid, long long seg, int k)
{

    printf("transform_feature_vector ...\n");
    half *gpu_half_feature;
    float *gpu_float_feature;

    cudaMalloc(&gpu_half_feature, sizeof(half)*seg*k);
    cudaMalloc(&gpu_float_feature, sizeof(float)*seg*k);
    gpuErr(cudaPeekAtLastError());

    for(int i = 0;i < grid;i++)
    {
        printf("\tgrid:%d\n", i);
        cudaMemcpy(gpu_half_feature, half_feature + i*seg*k, sizeof(half)*seg*k, cudaMemcpyHostToDevice);
        gpuErr(cudaPeekAtLastError());

        int num_blocks = (seg*k+255)/256;
        if(num_blocks > 8*24)num_blocks = 8*24;

        transform_half<<<num_blocks,256>>>(gpu_half_feature, gpu_float_feature, seg*k);
        
        gpuErr(cudaPeekAtLastError());
        cudaMemcpy(float_feature + i*seg*k, gpu_float_feature, sizeof(float)*seg*k, cudaMemcpyDeviceToHost);
        gpuErr(cudaPeekAtLastError());
    }

    cudaFree(gpu_half_feature);
    cudaFree(gpu_float_feature);
    gpuErr(cudaPeekAtLastError());
}

__global__ void pre_transform_print(half *p, half *q)
{
    long long m_index, n_index;

    m_index = 0;
    printf("m:%lld\n", m_index);
    for(long long i = m_index*128; i < m_index*128 + 128; i++)
    {
        printf("%.6f ", __half2float(p[i]));
    }
    printf("\n");

    m_index = 1;
    printf("m:%lld\n", m_index);
    for(long long i = m_index*128; i < m_index*128 + 128; i++)
    {
        printf("%.6f ", __half2float(p[i]));
    }
    printf("\n");

    m_index = 2;
    printf("m:%lld\n", m_index);
    for(long long i = m_index*128; i < m_index*128 + 128; i++)
    {
        printf("%.6f ", __half2float(p[i]));
    }
    printf("\n");


    // n
    n_index = 0;
    printf("n:%lld\n", n_index);
    for(long long i = n_index*128; i < n_index*128 + 128; i++)
    {
        printf("%.6f ", __half2float(q[i]));
    }
    printf("\n");

    n_index = 1;
    printf("n:%lld\n", n_index);
    for(long long i = n_index*128; i < n_index*128 + 128; i++)
    {
        printf("%.6f ", __half2float(q[i]));
    }
    printf("\n");

    n_index = 2;
    printf("n:%lld\n", n_index);
    for(long long i = n_index*128; i < n_index*128 + 128; i++)
    {
        printf("%.6f ", __half2float(q[i]));
    }
    printf("\n");

}


#include "set.h"

struct pthread_argument
{
    int local_ite;
    int iter;
    mf_model*model;
};


void *kernel_thread0(void *argument)
{
    pthread_argument *my_argument = (pthread_argument*)argument;

    int local_ite  = my_argument->local_ite;
    int iter       = my_argument->iter;
    mf_model*model = my_argument->model;

    int gpu_id = 0;

    //prepare device & stream.
    cudaSetDevice(gpu_id);
    cudaStream_t stream_com, stream_mem_d2h, stream_mem_h2d;
    cudaStreamCreate(&stream_com);
    cudaStreamCreate(&stream_mem_d2h);
    cudaStreamCreate(&stream_mem_h2d);


    int p_index;
    int q_index[4];

    if(local_ite == 0)
    {
        p_index = 0;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }
    else if(local_ite == 1)
    {
        p_index = 0;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }
    else if(local_ite == 2)
    {
        p_index = 2;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }
    else if(local_ite == 3)
    {
        p_index = 2;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }
    else if(local_ite == 4)
    {
        p_index = 4;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }
    else if(local_ite == 5)
    {
        p_index = 4;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }
    else if(local_ite == 6)
    {
        p_index = 6;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }
    else if(local_ite == 7)
    {
        p_index = 6;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }

    
    //transfer p_index from cpu to GPU
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_p,
                    model->halfp + p_index*model->u_seg*model->k,
                    sizeof(half)*model->u_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);

    //transfer q_index[0] to gpu_q[0]
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[0],
                    model->halfq + q_index[0]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d
                    );

    cudaDeviceSynchronize();
        
    //compute p_index-q_index[0], transfer q_index[1] to gpu_q[1]
    int grid_id = p_index*8 + q_index[0];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[0],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[1],
                    model->halfq + q_index[1]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);

    cudaDeviceSynchronize();
    
    //compute p_index-q_index[1], transfer q_index[2] to gpu_q[2], transfer q_index[0] back to cpu.
    grid_id = p_index*8 + q_index[1];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[1],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[2],
                    model->halfq + q_index[2]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);
    cudaMemcpyAsync(model->halfq + q_index[0]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[0],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);
    cudaDeviceSynchronize();

    //compute p_index-q_index[2], transfer p_index[3] to gpu_q[3], transfer q_index[1] back to cpu. 
    grid_id = p_index*8 + q_index[2];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[2],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[3],
                    model->halfq + q_index[3]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);
    cudaMemcpyAsync(model->halfq + q_index[1]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[1],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);
    cudaDeviceSynchronize();

    //compute p_index-q_index[3], transfer q_index[2] back to cpu. 
    grid_id = p_index*8 + q_index[3];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[3],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );
    cudaMemcpyAsync(model->halfq + q_index[2]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[2],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);
    cudaDeviceSynchronize();

    //transfer q_index[3] back to cpu. 
    cudaMemcpyAsync(model->halfq + q_index[3]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[3],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);

    //transfer p_index back to cpu
     cudaMemcpyAsync(model->halfp + p_index*model->u_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_p,
                    sizeof(half)*model->u_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);

     cudaDeviceSynchronize();

}

void *kernel_thread1(void *argument)
{
    pthread_argument *my_argument = (pthread_argument*)argument;

    int local_ite  = my_argument->local_ite;
    int iter       = my_argument->iter;
    mf_model*model = my_argument->model;

    int gpu_id = 1;

    //prepare device & stream.
    cudaSetDevice(gpu_id);
    cudaStream_t stream_com, stream_mem_d2h, stream_mem_h2d;
    cudaStreamCreate(&stream_com);
    cudaStreamCreate(&stream_mem_d2h);
    cudaStreamCreate(&stream_mem_h2d);

    int p_index;
    int q_index[4];

    if(local_ite == 0)
    {
        p_index = 1;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }
    else if(local_ite == 1)
    {
        p_index = 1;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }
    else if(local_ite == 2)
    {
        p_index = 3;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }
    else if(local_ite == 3)
    {
        p_index = 3;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }
    else if(local_ite == 4)
    {
        p_index = 5;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }
    else if(local_ite == 5)
    {
        p_index = 5;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }
    else if(local_ite == 6)
    {
        p_index = 7;

        q_index[0] = 4;
        q_index[1] = 5;
        q_index[2] = 6;
        q_index[3] = 7;
    }
    else if(local_ite == 7)
    {
        p_index = 7;

        q_index[0] = 0;
        q_index[1] = 1;
        q_index[2] = 2;
        q_index[3] = 3;
    }

    
    //transfer p_index from cpu to GPU
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_p,
                    model->halfp + p_index*model->u_seg*model->k,
                    sizeof(half)*model->u_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);

    //transfer q_index[0] to gpu_q[0]
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[0],
                    model->halfq + q_index[0]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d
                    );
    cudaDeviceSynchronize();

    //compute p_index-q_index[0], transfer q_index[1] to gpu_q[1]
    int grid_id = p_index*8 + q_index[0];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[0],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );

    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[1],
                    model->halfq + q_index[1]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);

    cudaDeviceSynchronize();

    //compute p_index-q_index[1], transfer q_index[2] to gpu_q[2], transfer q_index[0] back to cpu.
    grid_id = p_index*8 + q_index[1];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[1],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[2],
                    model->halfq + q_index[2]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);
    cudaMemcpyAsync(model->halfq + q_index[0]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[0],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);
    cudaDeviceSynchronize();

    //compute p_index-q_index[2], transfer p_index[3] to gpu_q[3], transfer q_index[1] back to cpu. 
    grid_id = p_index*8 + q_index[2];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[2],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );
    cudaMemcpyAsync(model->gpuPtrs[gpu_id].gpu_q[3],
                    model->halfq + q_index[3]*model->v_seg*model->k,
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyHostToDevice,
                    stream_mem_h2d);
    cudaMemcpyAsync(model->halfq + q_index[1]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[1],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);
    cudaDeviceSynchronize();

    //compute p_index-q_index[3], transfer q_index[2] back to cpu. 
    grid_id = p_index*8 + q_index[3];
    sgd_k128_kernel_hogwild_warp32_lrate<<<model->para.num_blocks, 32, 0, stream_com>>>(
                                        model->gpuPtrs[gpu_id].R2D[grid_id],
                                        model->gpuPtrs[gpu_id].gridSize[grid_id],
                                        model->gpuPtrs[gpu_id].gpu_p,
                                        model->gpuPtrs[gpu_id].gpu_q[3],
                                        model->gpuPtrs[gpu_id].rand_state,
                                        model->gpuPtrs[gpu_id].gpu_dynamic_rate,
                                        model->u_seg,
                                        model->v_seg,
                                        model->k,
                                        1,
                                        iter,
                                        model->max_update_count_per_block,
                                        model->update_count_per_block[grid_id],
                                        model->update_vector_size,
                                        model->para.lambda_p,
                                        model->para.lambda_q,
                                        NULL,
                                        1,
                                        1,
                                        0,
                                        0 
    );
    cudaMemcpyAsync(model->halfq + q_index[2]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[2],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);
    cudaDeviceSynchronize();

    //transfer q_index[3] back to cpu. 
    cudaMemcpyAsync(model->halfq + q_index[3]*model->v_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_q[3],
                    sizeof(half)*model->v_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);

    //transfer p_index back to cpu
     cudaMemcpyAsync(model->halfp + p_index*model->u_seg*model->k,
                    model->gpuPtrs[gpu_id].gpu_p,
                    sizeof(half)*model->u_seg*model->k,
                    cudaMemcpyDeviceToHost,
                    stream_mem_d2h);

     cudaDeviceSynchronize();

}


void sgd_update_k128(Parameter para, mf_model *model, mf_problem *prob, float scale)
{

    printf("calling sgd_update_k128() ...\n");
    //generate the random state for the hogwild scheduling policy.
    curandState *rand_state;
    cudaMalloc(&rand_state, sizeof(curandState)*para.num_blocks);
    gpuErr(cudaPeekAtLastError());

    init_rand_state<<<((para.num_blocks+255)/256),256>>>(rand_state,para.num_blocks);
    gpuErr(cudaPeekAtLastError());

    //generate the dynamic learning rate
    float dynamic_rate[1024];
    float alpha = para.alpha;
    float beta  = para.beta;
    float lrate = para.lrate;

    printf("alpha:%.4f\n", alpha);
    printf("beta :%.4f\n", beta);
    printf("lrate:%.4f\n", lrate);

    for(int i = 0;i < (para.num_iters + 4);i++)
    {
        double tmp_rate = alpha/(1 + beta*pow(i, 1.5));

        if(tmp_rate < para.lrate) tmp_rate = tmp_rate + para.lrate;
        dynamic_rate[i] = tmp_rate;
        printf("i:%4d, rate:%.8f\n",i, dynamic_rate[i]);
    }
    float *gpu_dynamic_rate;
    cudaMalloc((void**)&gpu_dynamic_rate, sizeof(float)*1024);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(gpu_dynamic_rate, dynamic_rate, sizeof(float)*1024, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    fflush(stdout);

    //malloc a problem grid on GPU
    if(prob->x_grid*prob->y_grid == 1)
    {
        cudaMalloc(&(prob->gpuR), sizeof(mf_node)*prob->maxGridSize);
        prob->cur_u_id = -1;
        prob->cur_v_id = -1;
    }
    else
    {
        cudaMalloc(&(prob->gpuRptrs[0]), sizeof(mf_node)*prob->maxGridSize);
        cudaMalloc(&(prob->gpuRptrs[1]), sizeof(mf_node)*prob->maxGridSize);
        prob->cur_global_x_id[0] = -1;
        prob->cur_global_x_id[1] = -1;
        prob->cur_global_y_id[0] = -1;
        prob->cur_global_y_id[1] = -1;
    }

    //malloc feature vectors on GPU
    if(prob->x_grid*prob->y_grid == 1)
    {
        cudaMalloc(&model->gpuHalfp, sizeof(half)*model->u_seg*model->k);
        cudaMalloc(&model->gpuHalfq, sizeof(half)*model->v_seg*model->k);
        model->cur_u_id = -1;
        model->cur_v_id = -1;
    }
    else
    {
        cudaMalloc(&model->gpuHalfPptrs[0], sizeof(half)*model->u_seg*model->k);
        cudaMalloc(&model->gpuHalfPptrs[1], sizeof(half)*model->u_seg*model->k);

        cudaMalloc(&model->gpuHalfQptrs[0], sizeof(half)*model->v_seg*model->k);
        cudaMalloc(&model->gpuHalfQptrs[1], sizeof(half)*model->v_seg*model->k);

        model->cur_global_x_id[0] = -1;
        model->cur_global_x_id[1] = -1;
        model->cur_global_y_id[0] = -1;
        model->cur_global_y_id[1] = -1;
    }   

    //set update count & block_err
    int update_vector_size = 128;
    int *update_count_per_block = new int[prob->ux*prob->vy]();
    int max_update_count_per_block = -1;
    for(int cur_grid_id = 0;cur_grid_id < prob->ux*prob->vy; cur_grid_id ++)
    {
        update_count_per_block[cur_grid_id] = (ceil)(1.0*prob->gridSize[cur_grid_id]/(para.num_blocks*update_vector_size));   
        if(max_update_count_per_block < update_count_per_block[cur_grid_id])
        {
            max_update_count_per_block = update_count_per_block[cur_grid_id];
        }
    }
    int count_size = para.num_iters*prob->ux*prob->vy*para.num_blocks*max_update_count_per_block + 1;
    printf("update_vector_size        : %d\n", update_vector_size);
    printf("max_update_count_per_block: %d\n",max_update_count_per_block);

    double *gpu_iter_err = NULL;

    #ifdef PRINTITE
        double *iter_err = new double[count_size];
        for(int i = 0;i < count_size; i++)iter_err[i] = 0.0;
        cudaMalloc(&gpu_iter_err,sizeof(double)*count_size);
        gpuErr(cudaPeekAtLastError());
        cudaMemcpy(gpu_iter_err, iter_err, sizeof(double)*count_size, cudaMemcpyHostToDevice);
        gpuErr(cudaPeekAtLastError());
    #endif

    //run the update kernel
    if(prob->u_grid*prob->v_grid == 1)
    {
        cudaMemcpy(prob->gpuR, prob->R2D[0], sizeof(mf_node)*prob->gridSize[0], cudaMemcpyHostToDevice);
        cudaMemcpy(model->gpuHalfp, model->halfp, sizeof(half)*model->u_seg*model->k, cudaMemcpyHostToDevice);
        cudaMemcpy(model->gpuHalfq, model->halfq, sizeof(half)*model->v_seg*model->k, cudaMemcpyHostToDevice);

        clock_t start = clock();

        sgd_k128_kernel_hogwild_warp32_lrate<<<para.num_blocks,32>>>(
                                                     prob->gpuR,
                                                     prob->gridSize[0],
                                                     model->gpuHalfp,
                                                     model->gpuHalfq,
                                                     rand_state,
                                                     gpu_dynamic_rate,
                                                     model->u_seg,
                                                     model->v_seg,
                                                     model->k,
                                                     para.num_iters,
                                                     0,
                                                     max_update_count_per_block,
                                                     update_count_per_block[0],
                                                     update_vector_size,
                                                     para.lambda_p,
                                                     para.lambda_q,
                                                     gpu_iter_err,
                                                     prob->u_grid,
                                                     prob->v_grid,
                                                     0,
                                                     0);
        cudaDeviceSynchronize();
        printf("time elapsed:%.8fs\n",(clock()-start)/double(CLOCKS_PER_SEC));
        cudaMemcpy(model->halfp, model->gpuHalfp, sizeof(half)*model->u_seg*model->k, cudaMemcpyDeviceToHost);
        cudaMemcpy(model->halfq, model->gpuHalfq, sizeof(half)*model->v_seg*model->k, cudaMemcpyDeviceToHost);
    }
    else if(prob->x_grid*prob->y_grid == 1)
    {
        clock_t start = clock();

        pthread_t threads[2];

        if(prob->u_grid == 8 && prob->v_grid == 8)
        {
            //prepare gpu
            model->gpuPtrs = new mf_gpu[2];

            for(int gpu_ite = 0; gpu_ite < 2; gpu_ite ++)
            {
                model->gpuPtrs[gpu_ite].maxGridSize = prob->maxGridSize;
                model->gpuPtrs[gpu_ite].gridSize = prob->gridSize;
            }

            //prepare R
            for(int gpu_ite = 0; gpu_ite < 2; gpu_ite ++)
            {
                cudaSetDevice(gpu_ite);
                model->gpuPtrs[gpu_ite].R2D = new mf_node*[64];
                for(int grid_id = 0; grid_id < 64; grid_id ++)
                {
                    cudaMalloc(&(model->gpuPtrs[gpu_ite].R2D[grid_id]), sizeof(mf_node)*prob->maxGridSize);
                    cudaMemcpy(model->gpuPtrs[gpu_ite].R2D[grid_id], prob->R2D[grid_id], sizeof(mf_node)*prob->gridSize[grid_id], cudaMemcpyHostToDevice);
                }
            }
            //prepare p
            for(int gpu_ite = 0; gpu_ite < 2; gpu_ite ++)
            {
                cudaSetDevice(gpu_ite);
                cudaMalloc(&(model->gpuPtrs[gpu_ite].gpu_p), sizeof(half)*model->u_seg*model->k);
            }
            //prepare q
            for(int gpu_ite = 0; gpu_ite < 2; gpu_ite ++)
            {
                cudaSetDevice(gpu_ite);
                model->gpuPtrs[gpu_ite].gpu_q = new half*[4];
                cudaMalloc(&(model->gpuPtrs[gpu_ite].gpu_q[0]), sizeof(half)*model->v_seg*model->k);
                cudaMalloc(&(model->gpuPtrs[gpu_ite].gpu_q[1]), sizeof(half)*model->v_seg*model->k);
                cudaMalloc(&(model->gpuPtrs[gpu_ite].gpu_q[2]), sizeof(half)*model->v_seg*model->k);
                cudaMalloc(&(model->gpuPtrs[gpu_ite].gpu_q[3]), sizeof(half)*model->v_seg*model->k);
            }

            //prepare randstate
            for(int gpu_ite = 0; gpu_ite < 2; gpu_ite ++)
            {
                cudaSetDevice(gpu_ite);
                cudaMalloc(&(model->gpuPtrs[gpu_ite].rand_state), sizeof(curandState)*para.num_blocks);
                init_rand_state<<<(para.num_blocks + 255)/256, 256>>>(model->gpuPtrs[gpu_ite].rand_state, para.num_blocks);
            }

            model->para = para;

            //learning rate
            for(int gpu_ite = 0; gpu_ite < 2; gpu_ite ++)
            {
                cudaSetDevice(gpu_ite);
                cudaMalloc(&(model->gpuPtrs[gpu_ite].gpu_dynamic_rate), sizeof(float)*1024);
                cudaMemcpy(model->gpuPtrs[gpu_ite].gpu_dynamic_rate, dynamic_rate, sizeof(float)*1024, cudaMemcpyHostToDevice);
            }

            model->max_update_count_per_block = max_update_count_per_block;
            model->update_count_per_block = update_count_per_block;
            model->update_vector_size = update_vector_size;

            pthread_argument arg_list[2];


            struct timespec begin, end;
            double elapsed;

            clock_gettime(CLOCK_MONOTONIC, &begin);
            //run
            for(int iter = 0; iter < para.num_iters; iter ++)
            {
                
                for(int local_ite = 0; local_ite < 8; local_ite ++)
                {
                      
                    arg_list[0].local_ite = local_ite;
                    arg_list[0].iter = iter;
                    arg_list[0].model = model;
                    pthread_create(&(threads[0]), NULL, kernel_thread0, (void*)&(arg_list[0]));
                    
pthread_join(threads[0], NULL);
                    arg_list[1].local_ite = local_ite;
                    arg_list[1].iter = iter;
                    arg_list[1].model = model;
                    pthread_create(&(threads[1]), NULL, kernel_thread1, (void*)&(arg_list[1]));
                    
                    //pthread_join(threads[0], NULL);
                    pthread_join(threads[1], NULL);
                }
                
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            elapsed = end.tv_sec - begin.tv_sec;
            elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;

            printf("computation time:%.8f\n", elapsed);

        }
        else
        {

            //random shuffle
            vector<int> u_id_vec(prob->u_grid, 0);
            vector<int> v_id_vec(prob->v_grid, 0);
            vector<int> uv_id_vec(prob->u_grid*prob->v_grid, 0);

            for(int i = 0;i < prob->u_grid;i++)u_id_vec[i] = i;
            for(int i = 0;i < prob->v_grid;i++)v_id_vec[i] = i;
            for(int i = 0;i < prob->u_grid*prob->v_grid; i++)uv_id_vec[i] = i;

            for(int iter = 0;iter < para.num_iters; iter ++)
            {
                printf("iter: %d\n", iter);
                clock_t ite_start = clock();

                //random_shuffle(uv_id_vec.begin(), uv_id_vec.end());
                set_order(uv_id_vec);

                for(int u_ite = 0;u_ite < prob->u_grid; u_ite ++)
                {

                    for(int v_ite = 0;v_ite < prob->v_grid; v_ite ++)
                    {

                        int tmp_index = u_ite*prob->v_grid + v_ite;
                        int cur_grid_id = uv_id_vec[tmp_index];
                        int cur_u_id = cur_grid_id/prob->v_grid;
                        int cur_v_id = cur_grid_id%prob->v_grid;

                        printf("\tgrid id: %d\n",cur_grid_id);
                        //transfer problem grid to gpu.
                        if(prob->cur_u_id != cur_u_id || prob->cur_v_id != cur_v_id)
                        {
                            cudaMemcpy(prob->gpuR, prob->R2D[cur_grid_id], sizeof(mf_node)*prob->gridSize[cur_grid_id], cudaMemcpyHostToDevice);
                        }
                        gpuErr(cudaPeekAtLastError());
                        prob->cur_u_id = cur_u_id;
                        prob->cur_v_id = cur_v_id;

                        //transfer p grid to gpu
                        if(model->cur_u_id == -1)
                        {
                            short *p_tmp = model->halfp + model->u_seg*model->k*cur_u_id; 
                            cudaMemcpy(model->gpuHalfp, p_tmp, sizeof(half)*model->u_seg*model->k, cudaMemcpyHostToDevice);
                            gpuErr(cudaPeekAtLastError());
                        }
                        else if(model->cur_u_id != cur_u_id)
                        {
                            short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_u_id;
                            cudaMemcpy(p_tmp, model->gpuHalfp, sizeof(half)*model->u_seg*model->k, cudaMemcpyDeviceToHost);
                            gpuErr(cudaPeekAtLastError());

                            p_tmp = model->halfp + model->u_seg*model->k*cur_u_id;
                            cudaMemcpy(model->gpuHalfp, p_tmp, sizeof(half)*model->u_seg*model->k, cudaMemcpyHostToDevice);
                            gpuErr(cudaPeekAtLastError());
                        }
                        model->cur_u_id = cur_u_id;
                        gpuErr(cudaPeekAtLastError());

                        //transfer q grid to gpu
                        if(model->cur_v_id == -1)
                        {
                            short *q_tmp = model->halfq + model->v_seg*model->k*cur_v_id;
                            cudaMemcpy(model->gpuHalfq, q_tmp, sizeof(half)*model->v_seg*model->k, cudaMemcpyHostToDevice);
                            gpuErr(cudaPeekAtLastError());
                        }
                        else if(model->cur_v_id != cur_v_id)
                        {
                            short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_v_id;
                            cudaMemcpy(q_tmp, model->gpuHalfq, sizeof(half)*model->v_seg*model->k, cudaMemcpyDeviceToHost);
                            gpuErr(cudaPeekAtLastError());

                            q_tmp = model->halfq + model->v_seg*model->k*cur_v_id;
                            cudaMemcpy(model->gpuHalfq, q_tmp, sizeof(half)*model->v_seg*model->k, cudaMemcpyHostToDevice);
                            gpuErr(cudaPeekAtLastError());
                        }
                        model->cur_v_id = cur_v_id;
                        gpuErr(cudaPeekAtLastError());

                        //call the kernel
                        sgd_k128_kernel_hogwild_warp32_lrate<<<para.num_blocks,32>>>(
                                                         prob->gpuR,
                                                         prob->gridSize[cur_grid_id],
                                                         model->gpuHalfp,
                                                         model->gpuHalfq,
                                                         rand_state,
                                                         gpu_dynamic_rate,
                                                         model->u_seg,
                                                         model->v_seg,
                                                         model->k,
                                                         1,
                                                         iter,
                                                         max_update_count_per_block,
                                                         update_count_per_block[cur_grid_id],
                                                         update_vector_size,
                                                         para.lambda_p,
                                                         para.lambda_q,
                                                         gpu_iter_err,
                                                         prob->u_grid,
                                                         prob->v_grid,
                                                         cur_u_id,
                                                         cur_v_id);
                        gpuErr(cudaPeekAtLastError());
                    }
                }
                cudaDeviceSynchronize();
                printf("\ttime elapsed:%.8fs\n",(clock()-ite_start)/double(CLOCKS_PER_SEC));

            }
            cudaDeviceSynchronize();
            printf("time elapsed:%.8fs\n",(clock()-start)/double(CLOCKS_PER_SEC));
        
            printf("%d,%d\n", model->cur_u_id, model->cur_v_id);
            //transfer p back to CPU
            if(model->cur_u_id >= 0)
            {
                short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_u_id;
                cudaMemcpy(p_tmp, model->gpuHalfp, sizeof(half)*model->u_seg*model->k, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
            //transfer q back to CPU
            if(model->cur_v_id >= 0)
            {
                short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_v_id;
                cudaMemcpy(q_tmp, model->gpuHalfq, sizeof(half)*model->v_seg*model->k, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
        }
    }
    else
    {
        clock_t start = clock();
        
        //scheduling info
        int *global_x_list = new int[prob->x_grid*prob->y_grid];
        int *global_y_list = new int[prob->x_grid*prob->y_grid];
        int *global_id_list = new int[prob->x_grid*prob->y_grid];

        //create stream
        cudaStream_t stream_com, stream_mem_d2h, stream_mem_h2d;
        cudaStreamCreate(&stream_com);
        cudaStreamCreate(&stream_mem_d2h);
        cudaStreamCreate(&stream_mem_h2d);

        //random shuffle
        vector<int> u_id_vec(prob->u_grid, 0);
        vector<int> v_id_vec(prob->v_grid, 0);
        for(int i = 0;i < prob->u_grid;i++)u_id_vec[i] = i;
        for(int i = 0;i < prob->v_grid;i++)v_id_vec[i] = i;

        vector<int> x_id_vec(prob->x_grid, 0);
        vector<int> y_id_vec(prob->y_grid, 0);
        for(int i = 0;i < prob->x_grid;i++)x_id_vec[i] = i;
        for(int i = 0;i < prob->y_grid;i++)y_id_vec[i] = i;

        //fully random
        vector<int> uv_id_vec(prob->u_grid*prob->v_grid, 0);
        for(int i = 0;i < prob->u_grid*prob->v_grid; i++)uv_id_vec[i] = i;
        vector<int> xy_id_vec(prob->x_grid*prob->y_grid, 0);
        for(int i = 0;i < prob->x_grid*prob->y_grid; i++)xy_id_vec[i] = i;


        for(int iter = 0;iter < para.num_iters; iter ++)
        {
            fflush(stdout);
            printf("iter: %d\n", iter);
            clock_t ite_start = clock();

            random_shuffle(uv_id_vec.begin(), uv_id_vec.end());

            random_shuffle(u_id_vec.begin(), u_id_vec.end());
            for(int u_ite = 0;u_ite < prob->u_grid; u_ite ++)
            {
                random_shuffle(v_id_vec.begin(), v_id_vec.begin());
                for(int v_ite = 0;v_ite < prob->v_grid; v_ite ++)
                {
                    

                    //no random
                    //int cur_u_id = u_ite;
                    //int cur_v_id = v_ite;
                        
                    //partial random
                    //int cur_u_id = u_id_vec[u_ite];
                    //int cur_v_id = v_id_vec[v_ite];   

                    //fully random
                    int tmp_uv_id = u_ite*prob->v_grid + v_ite;
                    int cur_u_id = uv_id_vec[tmp_uv_id]/prob->v_grid;
                    int cur_v_id = uv_id_vec[tmp_uv_id]%prob->v_grid;

                    //set information
                    random_shuffle(x_id_vec.begin(), x_id_vec.end());
                    random_shuffle(xy_id_vec.begin(), xy_id_vec.end());

                    for(int local_x_ite = 0;local_x_ite < prob->x_grid;local_x_ite ++)
                    {
                        random_shuffle(y_id_vec.begin(),y_id_vec.end());
                        for(int local_y_ite = 0;local_y_ite < prob->y_grid;local_y_ite ++)
                        {
                            //no random
                            //int cur_x_id = local_x_ite;
                            //int cur_y_id = local_y_ite;

                            //partial random
                            //int cur_x_id = x_id_vec[local_x_ite];
                            //int cur_y_id = y_id_vec[local_y_ite];

                            //fully random
                            int tmp_xy_id = local_x_ite*prob->y_grid + local_y_ite;
                            int cur_x_id = xy_id_vec[tmp_xy_id]/prob->y_grid;
                            int cur_y_id = xy_id_vec[tmp_xy_id]%prob->y_grid;

                            int local_id = cur_x_id*prob->y_grid + cur_y_id;

                            int global_x = cur_u_id*prob->x_grid + cur_x_id;
                            int global_y = cur_v_id*prob->y_grid + cur_y_id;
                            int global_id = global_x*prob->vy + global_y;

                            global_x_list[local_id] = global_x;
                            global_y_list[local_id] = global_y;
                            global_id_list[local_id] = global_id;

                            //printf("\tu_ite:%2d, v_ite:%2d, global_x:%2d, global_y:%2d, global_id:%2d\n", u_ite, v_ite, global_x_list[local_id], global_y_list[local_id], global_id_list[local_id]);
                        }
                    }

                    //run
                    for(int i = -1;i < prob->x_grid*prob->y_grid;i++)
                    {
                        //compute
                        if(i >= 0)
                        {
                            
                            sgd_k128_kernel_hogwild_warp32_lrate<<<para.num_blocks,32, 0, stream_com>>>(
                                                            prob->gpuRptrs[i%2],
                                                            prob->gridSize[global_id_list[i]],
                                                            model->gpuHalfPptrs[i%2],
                                                            model->gpuHalfQptrs[i%2],
                                                            rand_state,
                                                            gpu_dynamic_rate,
                                                            model->u_seg,
                                                            model->v_seg,
                                                            model->k,
                                                            1,
                                                            iter,
                                                            max_update_count_per_block,
                                                            update_count_per_block[global_id_list[i]],
                                                            update_vector_size,
                                                            para.lambda_p,
                                                            para.lambda_q,
                                                            gpu_iter_err,
                                                            prob->ux,
                                                            prob->vy,
                                                            global_x_list[i],
                                                            global_y_list[i]);
                            
                        }

                        //memcpy for the next block
                        if(i != (prob->x_grid*prob->y_grid - 1))
                        {
                            int next_global_x = global_x_list[i+1];
                            int next_global_y = global_y_list[i+1];
                            int next_global_id = global_id_list[i+1];

                            //transfer problem grid to gpu
                            if(prob->cur_global_x_id[(i+1)%2] !=  next_global_x || prob->cur_global_y_id[(i+1)%2] != next_global_y)
                            {
                                cudaMemcpyAsync(prob->gpuRptrs[(i+1)%2], 
                                                prob->R2D[next_global_id], 
                                                sizeof(mf_node)*prob->gridSize[next_global_id],
                                                cudaMemcpyHostToDevice,
                                                stream_mem_h2d);
                            }

                            //transfer feature p
                            if(model->cur_global_x_id[(i+1)%2] == -1)
                            {
                                if(model->cur_global_x_id[(i+2)%2] == next_global_x)
                                {
                                    model->cur_global_x_id[(i+2)%2] = -1;
                                    model->cur_global_x_id[(i+1)%2] = next_global_x;

                                    half *tmp_ptr = model->gpuHalfPptrs[(i+1)%2];
                                    model->gpuHalfPptrs[(i+1)%2] = model->gpuHalfPptrs[(i+2)%2];
                                    model->gpuHalfPptrs[(i+2)%2] = tmp_ptr;
                                }
                                else
                                {
                                    short *p_tmp = model->halfp + model->u_seg*model->k*next_global_x;
                                    cudaMemcpyAsync(model->gpuHalfPptrs[(i+1)%2],
                                                    p_tmp,    
                                                    sizeof(half)*model->u_seg*model->k,
                                                    cudaMemcpyHostToDevice,
                                                    stream_mem_h2d);
                                    model->cur_global_x_id[(i+1)%2] = next_global_x;
                                }
                            }
                            else if(model->cur_global_x_id[(i+1)%2] != next_global_x)
                            {
                                if(model->cur_global_x_id[(i+2)%2] == -1)
                                {
                                    //swap value
                                    int tmp = model->cur_global_x_id[(i+1)%2];
                                    model->cur_global_x_id[(i+1)%2] = next_global_x;
                                    model->cur_global_x_id[(i+2)%2] = tmp;

                                    //swap pointer
                                    half *p_tmp = model->gpuHalfPptrs[(i+1)%2];
                                    model->gpuHalfPptrs[(i+1)%2] = model->gpuHalfPptrs[(i+2)%2];
                                    model->gpuHalfPptrs[(i+2)%2] = p_tmp;

                                    //transfer
                                    short *p_tmp_trans = model->halfp + model->u_seg*model->k*next_global_x;
                                    cudaMemcpyAsync(model->gpuHalfPptrs[(i+1)%2],
                                                    p_tmp_trans,    
                                                    sizeof(half)*model->u_seg*model->k,
                                                    cudaMemcpyHostToDevice,
                                                    stream_mem_h2d);
                                    model->cur_global_x_id[(i+1)%2] = next_global_x;
                                }
                                else if(model->cur_global_x_id[(i+2)%2] == next_global_x)
                                {
                                    //swap value
                                    int tmp = model->cur_global_x_id[(i+1)%2];
                                    model->cur_global_x_id[(i+1)%2] = next_global_x;
                                    model->cur_global_x_id[(i+2)%2] = tmp;

                                    //swap pointer
                                    half *p_tmp = model->gpuHalfPptrs[(i+1)%2];
                                    model->gpuHalfPptrs[(i+1)%2] = model->gpuHalfPptrs[(i+2)%2];
                                    model->gpuHalfPptrs[(i+2)%2] = p_tmp;
                                }
                                else
                                {
                                    short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_global_x_id[(i+1)%2];
                                    cudaMemcpyAsync(p_tmp,
                                                    model->gpuHalfPptrs[(i+1)%2],
                                                    sizeof(half)*model->u_seg*model->k,
                                                    cudaMemcpyDeviceToHost,
                                                    stream_mem_d2h);

                                    p_tmp = model->halfp + model->u_seg*model->k*next_global_x;
                                    cudaMemcpyAsync(model->gpuHalfPptrs[(i+1)%2],
                                                    p_tmp,
                                                    sizeof(half)*model->u_seg*model->k,
                                                    cudaMemcpyHostToDevice,
                                                    stream_mem_h2d);

                                    model->cur_global_x_id[(i+1)%2] = next_global_x;
                                }
                            }
                            
                            //transfer feature q
                            if(model->cur_global_y_id[(i+1)%2] == -1)
                            {
                                if(model->cur_global_y_id[(i+2)%2] == next_global_y)
                                {
                                    model->cur_global_y_id[(i+2)%2] = -1;
                                    model->cur_global_y_id[(i+1)%2] = next_global_y;

                                    half *tmp_ptr = model->gpuHalfQptrs[(i+1)%2];
                                    model->gpuHalfQptrs[(i+1)%2] = model->gpuHalfQptrs[(i+2)%2];
                                    model->gpuHalfQptrs[(i+2)%2] = tmp_ptr;
                                }
                                else
                                {
                                    short *q_tmp = model->halfq + model->v_seg*model->k*next_global_y;
                                    cudaMemcpyAsync(model->gpuHalfQptrs[(i+1)%2],
                                                    q_tmp,
                                                    sizeof(half)*model->v_seg*model->k,
                                                    cudaMemcpyHostToDevice,
                                                    stream_mem_h2d);
                                    model->cur_global_y_id[(i+1)%2] = next_global_y;
                                }
                            }
                            else if(model->cur_global_y_id[(i+1)%2] != next_global_y)
                            {
                                if(model->cur_global_y_id[(i+2)%2] == -1)
                                {
                                    //swap value
                                    int tmp = model->cur_global_y_id[(i+1)%2];
                                    model->cur_global_y_id[(i+1)%2] = model->cur_global_y_id[(i+2)%2];
                                    model->cur_global_y_id[(i+2)%2] = tmp;

                                    //swap pointer
                                    half *q_tmp = model->gpuHalfQptrs[(i+1)%2];
                                    model->gpuHalfQptrs[(i+1)%2] = model->gpuHalfQptrs[(i+2)%2];
                                    model->gpuHalfQptrs[(i+2)%2] = q_tmp;

                                    short *q_tmp_trans = model->halfq + model->v_seg*model->k*next_global_y;
                                    cudaMemcpyAsync(model->gpuHalfQptrs[(i+1)%2],
                                                    q_tmp_trans,
                                                    sizeof(half)*model->v_seg*model->k,
                                                    cudaMemcpyHostToDevice,
                                                    stream_mem_h2d);
                                    model->cur_global_y_id[(i+1)%2] = next_global_y;
                                }
                                else if(model->cur_global_y_id[(i+2)%2] == next_global_y)
                                {
                                    //swap value
                                    int tmp = model->cur_global_y_id[(i+1)%2];
                                    model->cur_global_y_id[(i+1)%2] = model->cur_global_y_id[(i+2)%2];
                                    model->cur_global_y_id[(i+2)%2] = tmp;

                                    //swap pointer
                                    half *q_tmp = model->gpuHalfQptrs[(i+1)%2];
                                    model->gpuHalfQptrs[(i+1)%2] = model->gpuHalfQptrs[(i+2)%2];
                                    model->gpuHalfQptrs[(i+2)%2] = q_tmp;
                                }
                                else
                                {
                                    short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_global_y_id[(i+1)%2];
                                    cudaMemcpyAsync(q_tmp,
                                                    model->gpuHalfQptrs[(i+1)%2],
                                                    sizeof(half)*model->v_seg*model->k,
                                                    cudaMemcpyDeviceToHost,
                                                    stream_mem_d2h);

                                    q_tmp = model->halfq + model->v_seg*model->k*next_global_y;
                                    cudaMemcpyAsync(model->gpuHalfQptrs[(i+1)%2],
                                                    q_tmp,
                                                    sizeof(half)*model->v_seg*model->k,
                                                    cudaMemcpyHostToDevice,
                                                    stream_mem_h2d);
                                    model->cur_global_y_id[(i+1)%2] = next_global_y;
                                }
                            }
                        }
                        cudaDeviceSynchronize();
                    }   

                }
            }
            
            cudaDeviceSynchronize();
            printf("\ttime elapsed:%.8fs\n",(clock()-ite_start)/double(CLOCKS_PER_SEC));
        }
        cudaDeviceSynchronize();
        printf("time elapsed:%.8fs\n",(clock()-start)/double(CLOCKS_PER_SEC));
    
        //transfer p back
        if(model->cur_global_x_id[0] != -1)
        {
            short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_global_x_id[0];
            cudaMemcpy(p_tmp, model->gpuHalfPptrs[0], sizeof(half)*model->u_seg*model->k, cudaMemcpyDeviceToHost);
        }
        if(model->cur_global_x_id[1] != -1)
        {
            short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_global_x_id[1];
            cudaMemcpy(p_tmp, model->gpuHalfPptrs[1], sizeof(half)*model->u_seg*model->k, cudaMemcpyDeviceToHost);
        }

        //transfer q back
        if(model->cur_global_y_id[0] != -1)
        {
            short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_global_y_id[0];
            cudaMemcpy(q_tmp, model->gpuHalfQptrs[0], sizeof(half)*model->v_seg*model->k, cudaMemcpyDeviceToHost);
        }
        if(model->cur_global_y_id[1] != -1)
        {
            short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_global_y_id[1];
            cudaMemcpy(q_tmp, model->gpuHalfQptrs[1], sizeof(half)*model->v_seg*model->k, cudaMemcpyDeviceToHost);
        }
    }   

    if(prob->x_grid*prob->y_grid == 1)
    {
        cudaFree(model->gpuHalfp);
        cudaFree(model->gpuHalfq);
        cudaFree(prob->gpuR);
    }
    else
    {
        cudaFree(model->gpuHalfPptrs[0]);
        cudaFree(model->gpuHalfPptrs[1]);
        cudaFree(model->gpuHalfQptrs[0]);
        cudaFree(model->gpuHalfQptrs[1]);
        cudaFree(prob->gpuRptrs[0]);
        cudaFree(prob->gpuRptrs[1]);
    }

    gpuErr(cudaPeekAtLastError());

    printf("finished\n");


    #ifdef PRINTITE
        //print the per-iteration error
        cudaMemcpy(iter_err, gpu_iter_err, sizeof(double)*count_size, cudaMemcpyDeviceToHost);
    
        printf("train RMSE\n");
        for(int i = 0;i < para.num_iters; i++)
        {
            int update = prob->ux*prob->vy*para.num_blocks*max_update_count_per_block;

            double total_err = 0.0;
            for(int j = 0;j < update; j++)
            {
                total_err += iter_err[i*update + j];
            }
            SGDRate rmse = sqrt(total_err/prob->nnz)*scale;
            cout.width(4);
            cout << i;
            cout.width(13);
            cout << fixed << setprecision(4) << rmse << endl;
        }
        printf("\n\n");
        delete(iter_err);
        cudaFree(gpu_iter_err);
    #endif


    //transform halfp & halfq to floatp & floatq.
    cudaDeviceSynchronize();
    transform_feature_vector(model->halfp, model->floatp, model->m, model->ux, model->u_seg, model->k);
    transform_feature_vector(model->halfq, model->floatq, model->n, model->vy, model->v_seg, model->k);
    
    cudaFree(gpu_dynamic_rate);
    cudaFree(rand_state);
}