#ifndef ASTAR_CUH
#define ASTAR_CUH

#include <cuda_runtime.h>
#include <limits>
#include "heap.cuh"
#include "common.cuh"

template<typename Game>
__global__ void extract_expand(typename Game::Heap* heaps_dev,
                               typename Game::StatePtr* s_dev,
                               typename Game::StatePtr* m_dev,
                               typename Game::Node t) {
    extern __shared__ typename Game::StatePtr buf[];

    auto global_index = blockIdx.x * blockDim.x + threadIdx.x;
    auto block_index = blockIdx.x;
    auto index = threadIdx.x;

    if (global_index >= num_heaps) return;
    auto& heap = heaps_dev[global_index];

    if (auto q = heap.pop()) {
        if (q->node == t) {
            // push candidate
            buf[index] = std::move(q);
        } else {
            // expand the state list
            Game::expand(s_dev, q, t);
            // s_dev[index] = std::move(q);
        }
    }

    __syncthreads();

    // get the best target state
    for (auto i = blockDim.x; i > 1;) {
        i >>= 1;
        if (index < i) {
            auto& a = buf[index];
            auto& b = buf[index + i];
            // a <- min(a, b)
            if ((a && b && b->f < a->f) || !a) a = std::move(b);
        }
        __syncthreads();
    }

    if (index == 0) m_dev[block_index] = std::move(buf[0]);

    __syncthreads();

    if (global_index == 0) {
        for (auto i = 1u; i < gridDim.x; ++i) {
            auto& a = m_dev[0];
            auto& b = m_dev[i];
            if ((a && b && b->f < a->f) || !a) a = std::move(b);
        }
    }
}

template<typename Game>
__global__ void compare_heap_best(typename Game::Heap* heaps_dev,
                                  typename Game::StatePtr* m_dev,
                                  bool* found_dev) {
    extern __shared__ typename Game::StatePtr buf[];

    auto global_index = blockIdx.x * blockDim.x + threadIdx.x;
    auto block_index = blockIdx.x;
    auto index = threadIdx.x;
    if (global_index >= num_heaps) return;
    auto& heap = heaps_dev[global_index];

    buf[index] = heap.data()[0];

    for (auto i = blockDim.x; i > 1;) {
        i >>= 1;
        if (index < i) {
            auto& a = buf[index];
            auto& b = buf[index + i];
            // a <- min(a, b)
            if ((a && b && b->f < a->f) || !a) a = std::move(b);
        }
        __syncthreads();
    }

    if (index == 0) {
        auto b = std::move(buf[0]);
        found_dev[block_index] = m_dev[0] && ((b && m_dev[0]->f <= b->f) || !b);
    }

    __syncthreads();

    if (global_index == 0) {
        for (auto i = 1u; i < gridDim.x; ++i) {
            found_dev[0] = found_dev[0] || found_dev[i];
        }
    }
}

template<typename Game>
__global__ void remove_duplication(typename Game::Hashtable* hashtable_dev) {

}


#endif // !ASTAR_CUH
