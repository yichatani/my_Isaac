// #pragma once
// #include "cpu/vision.h"

// #ifdef WITH_CUDA
// #include "cuda/vision.h"

// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAEvent.h>

// extern THCState *state;
// #endif



// int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
// {

//     // TODO check dimensions
//     long batch, ref_nb, query_nb, dim, k;
//     batch = ref.size(0);
//     dim = ref.size(1);
//     k = idx.size(1);
//     ref_nb = ref.size(2);
//     query_nb = query.size(2);

//     float *ref_dev = ref.data<float>();
//     float *query_dev = query.data<float>();
//     long *idx_dev = idx.data<long>();




//   if (ref.type().is_cuda()) {
// #ifdef WITH_CUDA
//     // TODO raise error if not compiled with CUDA
//     float *dist_dev = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float));

//     for (int b = 0; b < batch; b++)
//     {
//     // knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
//     //   dist_dev, idx_dev + b * k * query_nb, THCState_getCurrentStream(state));
//       knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
//       dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
//     }
//     c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("error in knn: %s\n", cudaGetErrorString(err));
//         throw std::runtime_error("aborting");
//     }
//     return 1;
// #else
//     AT_ERROR("Not compiled with GPU support");
// #endif
//   }


//     float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
//     long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
//     for (int b = 0; b < batch; b++) {
//     knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
//       dist_dev, idx_dev + b * k * query_nb, ind_buf);
//     }

//     free(dist_dev);
//     free(ind_buf);

//     return 1;

// }

#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    // 检查输入张量的维度
    TORCH_CHECK(ref.dim() == 3, "ref tensor must be 3-dimensional");
    TORCH_CHECK(query.dim() == 3, "query tensor must be 3-dimensional");
    TORCH_CHECK(idx.dim() == 2, "idx tensor must be 2-dimensional");

    // 获取张量维度
    long batch = ref.size(0);
    long dim = ref.size(1);
    long ref_nb = ref.size(2);
    long query_nb = query.size(2);
    long k = idx.size(1);

    // 获取数据指针
    float* ref_dev = ref.data_ptr<float>();
    float* query_dev = query.data_ptr<float>();
    long* idx_dev = idx.data_ptr<long>();

    // CUDA 分支
    if (ref.device().is_cuda()) {
#ifdef WITH_CUDA
        // 分配临时内存
        float* dist_dev = static_cast<float*>(
            c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float))
        );

        for (int b = 0; b < batch; b++) {
            knn_device(
                ref_dev + b * dim * ref_nb,
                ref_nb,
                query_dev + b * dim * query_nb,
                query_nb,
                dim,
                k,
                dist_dev,
                idx_dev + b * k * query_nb,
                c10::cuda::getCurrentCUDAStream()
            );
        }

        // 释放临时内存
        c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);

        // 检查 CUDA 错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in knn: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error in knn");
        }

        return 1;
#else
        TORCH_CHECK(false, "Not compiled with GPU support");
#endif
    }

    // CPU 分支
    std::vector<float> dist_dev(ref_nb * query_nb);
    std::vector<long> ind_buf(ref_nb);

    for (int b = 0; b < batch; b++) {
        knn_cpu(
            ref_dev + b * dim * ref_nb,
            ref_nb,
            query_dev + b * dim * query_nb,
            query_nb,
            dim,
            k,
            dist_dev.data(),
            idx_dev + b * k * query_nb,
            ind_buf.data()
        );
    }

    return 1;
}

