#ifndef ASTAR_CUH
#define ASTAR_CUH

#include <cuda_runtime.h>
#include <limits>
#include "heap.cuh"
#include "common.cuh"

template<typename Game>
__global__ void extract_expand(typename Game::Heap* heaps_dev,
                               Arc<typename Game::State>* s_dev,
                               Arc<typename Game::State>* m_dev,
                               typename Game::Node t) {
    auto global_index = blockIdx.x * blockDim.x + threadIdx.x;
    auto block_index = blockIdx.x;
    auto index = threadIdx.x;

    extern __shared__ typename Game::StatePtr buf[];

    auto& heap = heaps_dev[global_index];

    if (auto q = heap.pop()) {
        if (q->node == t) {
            // push candidate
            buf[index] = std::move(q);
        } else {
            // expand the state list
            Game::expand(s_dev, q, t);
        }
    }

    __syncthreads();

    // get the best target state
    auto i = blockDim.x;
    while (i > 1) {
        i >>= 1;
        if (index < i) {
            auto& a = buf[index];
            auto& b = buf[index + i];
            if (a && b) {
                // a <- min(a, b)
                if (b->f < a->f) a = std::move(b);
            } else if (a == nullptr) a = std::move(b);
        }
    }
    if (index == 0) m_dev[block_index] = std::move(buf[0]);
}


#endif // !ASTAR_CUH
