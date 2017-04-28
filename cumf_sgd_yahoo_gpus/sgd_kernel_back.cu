
#include "sgd.h"
#include "sgd_kernel.h"
#include <curand.h>
#include <curand_kernel.h>

#include <cstring>

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

    printf("alpha:%.4f\n", alpha);
    printf("beta :%.4f\n", beta);

    for(int i = 0;i < (para.num_iters + 4);i++)
    {
        dynamic_rate[i] = alpha/(1 + beta*pow(i+1, 1.5));
        printf("i:%4d, rate:%.8f\n",i, dynamic_rate[i]);
    }
    float *gpu_dynamic_rate;
    cudaMalloc((void**)&gpu_dynamic_rate, sizeof(float)*1024);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(gpu_dynamic_rate, dynamic_rate, sizeof(float)*1024, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

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
    int update_vector_size = 256;
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

    double *iter_err = new double[count_size];
    for(int i = 0;i < count_size; i++)iter_err[i] = 0.0;
    double *gpu_iter_err;
    cudaMalloc(&gpu_iter_err,sizeof(double)*count_size);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(gpu_iter_err, iter_err, sizeof(double)*count_size, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

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
        for(int iter = 0;iter < para.num_iters; iter ++)
        {
            printf("iter: %d\n", iter);
            clock_t ite_start = clock();

            for(int u_ite = 0;u_ite < prob->u_grid; u_ite ++)
            {
                for(int v_ite = 0;v_ite < prob->v_grid; v_ite ++)
                {
                    
                    int cur_u_id = u_ite;
                    int cur_v_id = v_ite;

                    int cur_grid_id = cur_u_id*prob->v_grid + cur_v_id;
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
    else
    {
        clock_t start = clock();
        
        //scheduling info
        int *global_x_list = new int[prob->x_grid*prob->y_grid];
        int *global_y_list = new int[prob->x_grid*prob->y_grid];
        int *global_id_list = new int[prob->x_grid*prob->y_grid];

        //create stream
        cudaStream_t stream_com, stream_mem;
        cudaStreamCreate(&stream_com);
        cudaStreamCreate(&stream_mem);

        for(int iter = 0;iter < para.num_iters; iter ++)
        {
            printf("iter: %d\n", iter);
            clock_t ite_start = clock();

            //set information
            for(int u_ite = 0;u_ite < prob->u_grid; u_ite ++)
            {
                for(int v_ite = 0;v_ite < prob->v_grid; v_ite ++)
                {
                    int cur_u_id = u_ite;
                    int cur_v_id = v_ite;
                    
                    for(int local_x_ite = 0;local_x_ite < prob->x_grid;local_x_ite ++)
                    {
                        for(int local_y_ite = 0;local_y_ite < prob->y_grid;local_y_ite ++)
                        {
                            int local_id = local_x_ite*prob->y_grid + local_y_ite;

                            int global_x = cur_u_id*prob->x_grid + local_x_ite;
                            int global_y = cur_v_id*prob->y_grid + local_y_ite;
                            int global_id = global_x*prob->vy + global_y;

                            global_x_list[local_id] = global_x;
                            global_y_list[local_id] = global_y;
                            global_id_list[local_id] = global_id;

                            printf("\tu_ite:%d, v_ite:%d, global_x:%d, global_y:%d, global_id:%d\n", u_ite, v_ite, global_x_list[local_id], global_y_list[local_id], global_id_list[local_id]);
                        }
                    }

                }
            }



            //run
            for(int i = -1;i < prob->x_grid*prob->y_grid;i++)
            {
                //compute
                if(i >= 0)
                {
                    /*
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
                    */

                }

                //memcpy for the next block
                if(i != (prob->x_grid*prob->y_grid - 1))
                {
                    int next_global_x = global_x_list[i+1];
                    int next_global_y = global_y_list[i+1];
                    int next_global_id = global_id_list[i+1];
                    //transfer problem grid



                }

                cudaDeviceSynchronize();
            }


            
            cudaDeviceSynchronize();
            printf("\ttime elapsed:%.8fs\n",(clock()-ite_start)/double(CLOCKS_PER_SEC));
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
    //print the per-iteration error
    cudaMemcpy(iter_err, gpu_iter_err, sizeof(double)*count_size, cudaMemcpyDeviceToHost);
    
    printf("train RMSE\n");
    for(int i = 0;i < para.num_iters; i++)
    {
        int update = prob->ux*prob->vy*para.num_blocks*max_update_count_per_block;

        double total_err = 0.0;
        for(int j = 0;j < update; j++)
        {
            total_err += iter_err[i*update+ j];
        }
        SGDRate rmse = sqrt(total_err/prob->nnz)*scale;
        cout.width(4);
        cout << i;
        cout.width(13);
        cout << fixed << setprecision(4) << rmse << endl;
    }
    printf("\n\n");
    //transform halfp & halfq to floatp & floatq.


    cudaDeviceSynchronize();

    transform_feature_vector(model->halfp, model->floatp, model->m, model->ux, model->u_seg, model->k);
    transform_feature_vector(model->halfq, model->floatq, model->n, model->vy, model->v_seg, model->k);

    delete(iter_err);
    cudaFree(gpu_iter_err);
    cudaFree(gpu_dynamic_rate);
    cudaFree(rand_state);

}