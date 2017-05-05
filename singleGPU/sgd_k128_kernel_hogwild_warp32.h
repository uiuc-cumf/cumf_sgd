__global__ void sgd_k128_kernel_hogwild_warp32_lrate(
    const mf_node *R, long long nnz, half *p, half *q, curandState *state,
    float *dynamic_rate, long long u_seg, long long v_seg, int k, int num_iters,
    int current_iter, int update_count_per_block, int update_count_this_block,
    int update_vector_size, float lambda_p, float lambda_q,
    double *gpu_iter_err, int u_grid, int v_grid, int u_id, int v_id) {
  // persistant thread, k = blockDim.x = 128
  for (int ite = current_iter; ite < current_iter + num_iters; ite++) {
    // MDS
    float lrate = __ldg(&dynamic_rate[ite]);
    for (int update_ite = 0; update_ite < update_count_this_block;
         update_ite++) {
      int lane_id = threadIdx.x % 32;
      int local_wid = threadIdx.x / 32;
      int wid = 4 * blockIdx.x + local_wid;

      // select a random sample with the start_id
      long long start_id = 0;
      if (lane_id == 0) {
        long long origin = (long long)(curand_uniform(&state[wid]) * nnz);
        start_id = origin % nnz;
      }
      start_id = __shfl(start_id, 0);

      // update the conitunous update_vector_size samples starting from
      // start_id, update_vector_size = batch size
      for (int i = 0; i < update_vector_size; i++) {
        int offset = (start_id + i) % nnz;

        // stored in COO format
        float ruv = __ldg(&R[offset].rate);
        int u = __ldg(&R[offset].u);
        int v = __ldg(&R[offset].v);

        int base_p = u * k;
        int base_q = v * k;

        // read p and q into registers
        float p1 = __half2float(p[base_p + lane_id]);
        float q1 = __half2float(q[base_q + lane_id]);

        float p2 = __half2float(p[base_p + lane_id + 32]);
        float q2 = __half2float(q[base_q + lane_id + 32]);

        float p3 = __half2float(p[base_p + lane_id + 64]);
        float q3 = __half2float(q[base_q + lane_id + 64]);

        float p4 = __half2float(p[base_p + lane_id + 96]);
        float q4 = __half2float(q[base_q + lane_id + 96]);

        // get dot product p x q
        float pq = p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4;
        pq += __shfl_down(pq, 16);
        pq += __shfl_down(pq, 8);
        pq += __shfl_down(pq, 4);
        pq += __shfl_down(pq, 2);
        pq += __shfl_down(pq, 1);
        pq = __shfl(pq, 0);

        // error
        float err = ruv - pq;

        // update p and q
        p[base_p + lane_id + 0] =
            __float2half(p1 + lrate * (err * q1 - lambda_p * p1));
        q[base_q + lane_id + 0] =
            __float2half(q1 + lrate * (err * p1 - lambda_q * q1));

        p[base_p + lane_id + 32] =
            __float2half(p2 + lrate * (err * q2 - lambda_p * p2));
        q[base_q + lane_id + 32] =
            __float2half(q2 + lrate * (err * p2 - lambda_q * q2));

        p[base_p + lane_id + 64] =
            __float2half(p3 + lrate * (err * q3 - lambda_p * p3));
        q[base_q + lane_id + 64] =
            __float2half(q3 + lrate * (err * p3 - lambda_q * q3));

        p[base_p + lane_id + 96] =
            __float2half(p4 + lrate * (err * q4 - lambda_p * p4));
        q[base_q + lane_id + 96] =
            __float2half(q4 + lrate * (err * p4 - lambda_q * q4));
      }
    }
  }
}

// buggy
__global__ void sgd_k128_kernel_hogwild_warp32_mom(
    const mf_node *R, long long nnz, half *p, half *q, curandState *state,
    long long u_seg, long long v_seg, int k, int num_iters, int current_iter,
    int update_count_per_block, int update_count_this_block,
    int update_vector_size, float lambda_p, float lambda_q,
    double *gpu_iter_err, int u_grid, int v_grid, int u_id, int v_id, float *vu,
    float *vv) {
  // persistant thread, k = blockDim.x = 128
  float lrate = 0.05;
  for (int ite = current_iter; ite < current_iter + num_iters; ite++) {
    for (int update_ite = 0; update_ite < update_count_this_block;
         update_ite++) {
      int lane_id = threadIdx.x % 32;
      int local_wid = threadIdx.x / 32;
      int wid = 4 * blockIdx.x + local_wid;

      // select a random sample with the start_id
      long long start_id = 0;
      if (lane_id == 0) {
        long long origin = (long long)(curand_uniform(&state[wid]) * nnz);
        start_id = origin % nnz;
      }
      start_id = __shfl(start_id, 0);

      // update the conitunous update_vector_size samples starting from
      // start_id, update_vector_size = batch size
      for (int i = 0; i < update_vector_size; i++) {
        int offset = (start_id + i) % nnz;

        // stored in COO format
        float ruv = __ldg(&R[offset].rate);
        int u = __ldg(&R[offset].u);
        int v = __ldg(&R[offset].v);

        int base_p = u * k;
        int base_q = v * k;

        // read p and q into registers
        float p1 = __half2float(p[base_p + lane_id]);
        float q1 = __half2float(q[base_q + lane_id]);

        float p2 = __half2float(p[base_p + lane_id + 32]);
        float q2 = __half2float(q[base_q + lane_id + 32]);

        float p3 = __half2float(p[base_p + lane_id + 64]);
        float q3 = __half2float(q[base_q + lane_id + 64]);

        float p4 = __half2float(p[base_p + lane_id + 96]);
        float q4 = __half2float(q[base_q + lane_id + 96]);

        // get dot product p x q
        float pq = p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4;
        pq += __shfl_down(pq, 16);
        pq += __shfl_down(pq, 8);
        pq += __shfl_down(pq, 4);
        pq += __shfl_down(pq, 2);
        pq += __shfl_down(pq, 1);
        pq = __shfl(pq, 0);

        // error
        float err = ruv - pq;

        // calculate gradient gu and hv
        float p1_gu = -err * q1 + lambda_p * p1;
        float p2_gu = -err * q2 + lambda_p * p2;
        float p3_gu = -err * q3 + lambda_p * p3;
        float p4_gu = -err * q4 + lambda_p * p4;

        float q1_hv = -err * p1 + lambda_q * q1;
        float q2_hv = -err * p2 + lambda_q * q2;
        float q3_hv = -err * p3 + lambda_q * q3;
        float q4_hv = -err * p4 + lambda_q * q4;

        // MOMENTUM
        float vp1 = vu[base_p + lane_id];
        float vp2 = vu[base_p + lane_id + 32];
        float vp3 = vu[base_p + lane_id + 64];
        float vp4 = vu[base_p + lane_id + 96];
        float vq1 = vv[base_q + lane_id];
        float vq2 = vv[base_q + lane_id + 32];
        float vq3 = vv[base_q + lane_id + 64];
        float vq4 = vv[base_q + lane_id + 96];

        vp1 = 0.9 * vp1 + lrate * p1_gu;
        vp2 = 0.9 * vp2 + lrate * p2_gu;
        vp3 = 0.9 * vp3 + lrate * p3_gu;
        vp4 = 0.9 * vp4 + lrate * p4_gu;

        vq1 = 0.9 * vq1 + lrate * q1_hv;
        vq2 = 0.9 * vq2 + lrate * q2_hv;
        vq3 = 0.9 * vq3 + lrate * q3_hv;
        vq4 = 0.9 * vq4 + lrate * q4_hv;

        // update p and q
        p[base_p + lane_id + 0] = __float2half(p1 - vp1);
        q[base_q + lane_id + 0] = __float2half(q1 - vq1);

        p[base_p + lane_id + 32] = __float2half(p2 - vp2);
        q[base_q + lane_id + 32] = __float2half(q2 - vq2);

        p[base_p + lane_id + 64] = __float2half(p3 - vp3);
        q[base_q + lane_id + 64] = __float2half(q3 - vq3);

        p[base_p + lane_id + 96] = __float2half(p4 - vp4);
        q[base_q + lane_id + 96] = __float2half(q4 - vq4);

        vu[base_p + lane_id + 0] = vp1;
        vu[base_p + lane_id + 32] = vp2;
        vu[base_p + lane_id + 64] = vp3;
        vu[base_p + lane_id + 96] = vp4;

        vv[base_q + lane_id + 0] = vq1;
        vv[base_q + lane_id + 32] = vq2;
        vv[base_q + lane_id + 64] = vq3;
        vv[base_q + lane_id + 96] = vq4;
      }
    }
  }
}

#define LR 0.1

__global__ void sgd_k128_kernel_hogwild_warp32_rpcs(
    const mf_node *R, long long nnz, half *p, half *q, curandState *state,
    long long u_seg, long long v_seg, int k, int num_iters, int current_iter,
    int update_count_per_block, int update_count_this_block,
    int update_vector_size, float lambda_p, float lambda_q,
    double *gpu_iter_err, int u_grid, int v_grid, int u_id, int v_id, float *gu,
    float *hv) {
  // persistant thread, k = blockDim.x = 128
  for (int ite = current_iter; ite < current_iter + num_iters; ite++) {
    for (int update_ite = 0; update_ite < update_count_this_block;
         update_ite++) {
      int lane_id = threadIdx.x % 32;
      int local_wid = threadIdx.x / 32;
      int wid = 4 * blockIdx.x + local_wid;

      // select a random sample with the start_id
      long long start_id = 0;
      if (lane_id == 0) {
        long long origin = (long long)(curand_uniform(&state[wid]) * nnz);
        start_id = origin % nnz;
      }
      start_id = __shfl(start_id, 0);

      // update the conitunous update_vector_size samples starting from
      // start_id, update_vector_size = batch size
      for (int i = 0; i < update_vector_size; i++) {
        int offset = (start_id + i) % nnz;

        // stored in COO format
        float ruv = __ldg(&R[offset].rate);
        int u = __ldg(&R[offset].u);
        int v = __ldg(&R[offset].v);

        int base_p = u * k;
        int base_q = v * k;
        if (threadIdx.x == 0 && blockIdx.x ==0 && i == 0 && update_ite== 0 && ite < 5) {
          printf("gu == %f and %d ",gu[0]);
        }
        // RPCS
        float p_lrate = LR * rsqrt(gu[u]);
        float q_lrate = LR * rsqrt(hv[v]);

        // read p and q into registers
        float p1 = __half2float(p[base_p + lane_id]);
        float q1 = __half2float(q[base_q + lane_id]);

        float p2 = __half2float(p[base_p + lane_id + 32]);
        float q2 = __half2float(q[base_q + lane_id + 32]);

        float p3 = __half2float(p[base_p + lane_id + 64]);
        float q3 = __half2float(q[base_q + lane_id + 64]);

        float p4 = __half2float(p[base_p + lane_id + 96]);
        float q4 = __half2float(q[base_q + lane_id + 96]);

        // get dot product p x q
        float pq = p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4;
        pq += __shfl_down(pq, 16);
        pq += __shfl_down(pq, 8);
        pq += __shfl_down(pq, 4);
        pq += __shfl_down(pq, 2);
        pq += __shfl_down(pq, 1);
        pq = __shfl(pq, 0);

        // error
        float err = ruv - pq;

        // calculate gradient gu and hv
        float p1_gu = -err * q1 + lambda_p * p1;
        float p2_gu = -err * q2 + lambda_p * p2;
        float p3_gu = -err * q3 + lambda_p * p3;
        float p4_gu = -err * q4 + lambda_p * p4;

        float q1_hv = -err * p1 + lambda_q * q1;
        float q2_hv = -err * p2 + lambda_q * q2;
        float q3_hv = -err * p3 + lambda_q * q3;
        float q4_hv = -err * p4 + lambda_q * q4;

        // update p and q
        p[base_p + lane_id + 0] = __float2half(p1 - p_lrate * p1_gu);
        q[base_q + lane_id + 0] = __float2half(q1 - q_lrate * q1_hv);

        p[base_p + lane_id + 32] = __float2half(p2 - p_lrate * p2_gu);
        q[base_q + lane_id + 32] = __float2half(q2 - q_lrate * q2_hv);

        p[base_p + lane_id + 64] = __float2half(p3 - p_lrate * p3_gu);
        q[base_q + lane_id + 64] = __float2half(q3 - q_lrate * q3_hv);

        p[base_p + lane_id + 96] = __float2half(p4 - p_lrate * p4_gu);
        q[base_q + lane_id + 96] = __float2half(q4 - q_lrate * q4_hv);

        // get dot product gu x gu
        float gugu =
            p1_gu * p1_gu + p2_gu * p2_gu + p3_gu * p3_gu + p4_gu * p4_gu;
        gugu += __shfl_down(gugu, 16);
        gugu += __shfl_down(gugu, 8);
        gugu += __shfl_down(gugu, 4);
        gugu += __shfl_down(gugu, 2);
        gugu += __shfl_down(gugu, 1);

        // lane 0 updates gu
        if (lane_id == 0) {
          gu[u] += gugu / k;
        }

        // get dot product hv x hv
        float hvhv =
            q1_hv * q1_hv + q2_hv * q2_hv + q3_hv * q3_hv + q4_hv * q4_hv;
        hvhv += __shfl_down(hvhv, 16);
        hvhv += __shfl_down(hvhv, 8);
        hvhv += __shfl_down(hvhv, 4);
        hvhv += __shfl_down(hvhv, 2);
        hvhv += __shfl_down(hvhv, 1);

        // lane 0 updates hv
        if (lane_id == 0) {
          hv[v] += hvhv / k;
        }
      }
    }
  }
}

__global__ void sgd_k128_kernel_hogwild_warp32_rpcs_fast(
    const mf_node *R, long long nnz, half *p, half *q, curandState *state,
    long long u_seg, long long v_seg, int k, int num_iters, int current_iter,
    int update_count_per_block, int update_count_this_block,
    int update_vector_size, float lambda_p, float lambda_q,
    double *gpu_iter_err, int u_grid, int v_grid, int u_id, int v_id, float *gu,
    float *hv, int *gu_cnt, int *hv_cnt) {
  // persistant thread, k = blockDim.x = 128
  for (int ite = current_iter; ite < current_iter + num_iters; ite++) {
    for (int update_ite = 0; update_ite < update_count_this_block;
         update_ite++) {
      int lane_id = threadIdx.x % 32;
      int local_wid = threadIdx.x / 32;
      int wid = 4 * blockIdx.x + local_wid;

      // select a random sample with the start_id
      long long start_id = 0;
      if (lane_id == 0) {
        long long origin = (long long)(curand_uniform(&state[wid]) * nnz);
        start_id = origin % nnz;
      }
      start_id = __shfl(start_id, 0);

      // update the conitunous update_vector_size samples starting from
      // start_id, update_vector_size = batch size
      for (int i = 0; i < update_vector_size; i++) {
        int offset = (start_id + i) % nnz;

        // stored in COO format
        float ruv = __ldg(&R[offset].rate);
        int u = __ldg(&R[offset].u);
        int v = __ldg(&R[offset].v);

        int base_p = u * k;
        int base_q = v * k;

        // RPCS
        float p_lrate = LR * rsqrt(gu[u]);
        float q_lrate = LR * rsqrt(hv[v]);

        // read p and q into registers
        float p1 = __half2float(p[base_p + lane_id]);
        float q1 = __half2float(q[base_q + lane_id]);

        float p2 = __half2float(p[base_p + lane_id + 32]);
        float q2 = __half2float(q[base_q + lane_id + 32]);

        float p3 = __half2float(p[base_p + lane_id + 64]);
        float q3 = __half2float(q[base_q + lane_id + 64]);

        float p4 = __half2float(p[base_p + lane_id + 96]);
        float q4 = __half2float(q[base_q + lane_id + 96]);

        // get dot product p x q
        float pq = p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4;
        pq += __shfl_down(pq, 16);
        pq += __shfl_down(pq, 8);
        pq += __shfl_down(pq, 4);
        pq += __shfl_down(pq, 2);
        pq += __shfl_down(pq, 1);
        pq = __shfl(pq, 0);

        // error
        float err = ruv - pq;

        // calculate gradient gu and hv
        float p1_gu = -err * q1 + lambda_p * p1;
        float p2_gu = -err * q2 + lambda_p * p2;
        float p3_gu = -err * q3 + lambda_p * p3;
        float p4_gu = -err * q4 + lambda_p * p4;

        float q1_hv = -err * p1 + lambda_q * q1;
        float q2_hv = -err * p2 + lambda_q * q2;
        float q3_hv = -err * p3 + lambda_q * q3;
        float q4_hv = -err * p4 + lambda_q * q4;

        // update p and q
        p[base_p + lane_id + 0] = __float2half(p1 - p_lrate * p1_gu);
        q[base_q + lane_id + 0] = __float2half(q1 - q_lrate * q1_hv);

        p[base_p + lane_id + 32] = __float2half(p2 - p_lrate * p2_gu);
        q[base_q + lane_id + 32] = __float2half(q2 - q_lrate * q2_hv);

        p[base_p + lane_id + 64] = __float2half(p3 - p_lrate * p3_gu);
        q[base_q + lane_id + 64] = __float2half(q3 - q_lrate * q3_hv);

        p[base_p + lane_id + 96] = __float2half(p4 - p_lrate * p4_gu);
        q[base_q + lane_id + 96] = __float2half(q4 - q_lrate * q4_hv);

        // get dot product gu x gu
        float gugu =
            p1_gu * p1_gu + p2_gu * p2_gu + p3_gu * p3_gu + p4_gu * p4_gu;
        gugu += __shfl_down(gugu, 16);
        gugu += __shfl_down(gugu, 8);
        gugu += __shfl_down(gugu, 4);
        gugu += __shfl_down(gugu, 2);
        gugu += __shfl_down(gugu, 1);

        // lane 0 updates gu
        if (lane_id == 0) {
          if (gu_cnt[u] > 4) {
            gu[u] += gugu / k;
          } else {
            gu_cnt[u]++;
          }
        }

        // get dot product hv x hv
        float hvhv =
            q1_hv * q1_hv + q2_hv * q2_hv + q3_hv * q3_hv + q4_hv * q4_hv;
        hvhv += __shfl_down(hvhv, 16);
        hvhv += __shfl_down(hvhv, 8);
        hvhv += __shfl_down(hvhv, 4);
        hvhv += __shfl_down(hvhv, 2);
        hvhv += __shfl_down(hvhv, 1);

        // lane 0 updates hv
        if (lane_id == 0) {
          if (hv_cnt[v] > 4) {
            hv[v] += hvhv / k;
          } else {
            hv_cnt[v]++;
          }
        }
      }
    }
  }
}

__global__ void sgd_k128_kernel_hogwild_warp32_rpcs_fast_tl(
    const mf_node *R, long long nnz, half *p, half *q, curandState *state,
    long long u_seg, long long v_seg, int k, int num_iters, int current_iter,
    int update_count_per_block, int update_count_this_block,
    int update_vector_size, float lambda_p, float lambda_q,
    double *gpu_iter_err, int u_grid, int v_grid, int u_id, int v_id, float *gu,
    float *hv, int *gu_cnt, int *hv_cnt, float *gu_f, float *hv_f) {
  // persistant thread, k = blockDim.x = 128
  for (int ite = current_iter; ite < current_iter + num_iters; ite++) {
    for (int update_ite = 0; update_ite < update_count_this_block;
         update_ite++) {
      int lane_id = threadIdx.x % 32;
      int local_wid = threadIdx.x / 32;
      int wid = 4 * blockIdx.x + local_wid;

      // select a random sample with the start_id
      long long start_id = 0;
      if (lane_id == 0) {
        long long origin = (long long)(curand_uniform(&state[wid]) * nnz);
        start_id = origin % nnz;
      }
      start_id = __shfl(start_id, 0);

      // update the conitunous update_vector_size samples starting from
      // start_id, update_vector_size = batch size
      for (int i = 0; i < update_vector_size; i++) {
        int offset = (start_id + i) % nnz;

        // stored in COO format
        float ruv = __ldg(&R[offset].rate);
        int u = __ldg(&R[offset].u);
        int v = __ldg(&R[offset].v);

        int base_p = u * k;
        int base_q = v * k;
        
        // RPCS
        float p_lrate = LR * rsqrt(gu[u]);
        float q_lrate = LR * rsqrt(hv[v]);
        float p_lrate_f = LR * rsqrt(gu_f[u]);
        float q_lrate_f = LR * rsqrt(hv_f[v]);

        // read p and q into registers
        float p1 = __half2float(p[base_p + lane_id]);
        float q1 = __half2float(q[base_q + lane_id]);

        float p2 = __half2float(p[base_p + lane_id + 32]);
        float q2 = __half2float(q[base_q + lane_id + 32]);

        float p3 = __half2float(p[base_p + lane_id + 64]);
        float q3 = __half2float(q[base_q + lane_id + 64]);

        float p4 = __half2float(p[base_p + lane_id + 96]);
        float q4 = __half2float(q[base_q + lane_id + 96]);

        // get dot product p x q
        float pq = p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4;
        pq += __shfl_down(pq, 16);
        pq += __shfl_down(pq, 8);
        pq += __shfl_down(pq, 4);
        pq += __shfl_down(pq, 2);
        pq += __shfl_down(pq, 1);
        pq = __shfl(pq, 0);

        // error
        float err = ruv - pq;

        // calculate gradient gu and hv
        float p1_gu = -err * q1 + lambda_p * p1;
        float p2_gu = -err * q2 + lambda_p * p2;
        float p3_gu = -err * q3 + lambda_p * p3;
        float p4_gu = -err * q4 + lambda_p * p4;

        float q1_hv = -err * p1 + lambda_q * q1;
        float q2_hv = -err * p2 + lambda_q * q2;
        float q3_hv = -err * p3 + lambda_q * q3;
        float q4_hv = -err * p4 + lambda_q * q4;

        // update p and q
        p[base_p + lane_id + 0] = __float2half(p1 - p_lrate * p1_gu);
        q[base_q + lane_id + 0] = __float2half(q1 - q_lrate * q1_hv);

        p[base_p + lane_id + 32] = __float2half(p2 - p_lrate_f * p2_gu);
        q[base_q + lane_id + 32] = __float2half(q2 - q_lrate_f * q2_hv);

        p[base_p + lane_id + 64] = __float2half(p3 - p_lrate_f * p3_gu);
        q[base_q + lane_id + 64] = __float2half(q3 - q_lrate_f * q3_hv);

        p[base_p + lane_id + 96] = __float2half(p4 - p_lrate_f * p4_gu);
        q[base_q + lane_id + 96] = __float2half(q4 - q_lrate_f * q4_hv);

        // get dot product gu x gu
        float gugu = p1_gu * p1_gu;
        gugu += __shfl_down(gugu, 16);
        gugu += __shfl_down(gugu, 8);
        gugu += __shfl_down(gugu, 4);
        gugu += __shfl_down(gugu, 2);
        gugu += __shfl_down(gugu, 1);

        // lane 0 updates gu
        if (lane_id == 0) {
          gu[u] += gugu / 32;
        }

        gugu = p2_gu * p2_gu + p3_gu * p3_gu + p4_gu * p4_gu;
        gugu += __shfl_down(gugu, 16);
        gugu += __shfl_down(gugu, 8);
        gugu += __shfl_down(gugu, 4);
        gugu += __shfl_down(gugu, 2);
        gugu += __shfl_down(gugu, 1);

        // lane 0 updates gu_f
        if (lane_id == 0) {
          if (gu_cnt[u] > 2) {
            gu_f[u] += gugu / 96;
          } else {
            gu_cnt[u]++;
          }
        }

        // get dot product hv x hv
        float hvhv = q1_hv * q1_hv;
        hvhv += __shfl_down(hvhv, 16);
        hvhv += __shfl_down(hvhv, 8);
        hvhv += __shfl_down(hvhv, 4);
        hvhv += __shfl_down(hvhv, 2);
        hvhv += __shfl_down(hvhv, 1);

        // lane 0 updates gu
        if (lane_id == 0) {
          hv[v] += hvhv / 32;
        }

        hvhv = q2_hv * q2_hv + q3_hv * q3_hv + q4_hv * q4_hv;
        hvhv += __shfl_down(hvhv, 16);
        hvhv += __shfl_down(hvhv, 8);
        hvhv += __shfl_down(hvhv, 4);
        hvhv += __shfl_down(hvhv, 2);
        hvhv += __shfl_down(hvhv, 1);

        // lane 0 updates hv
        if (lane_id == 0) {
          if (hv_cnt[v] > 2) {
            hv[v] += hvhv / 96;
          } else {
            hv_cnt[v]++;
          }
        }
      }
    }
  }
}
