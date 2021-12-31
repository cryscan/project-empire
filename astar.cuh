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

    auto index = threadIdx.x;
    if (index >= num_heaps) return;
    auto& heap = heaps_dev[index];

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
    typename Game::StatePtr current_best;
    if (index == 0) {
        current_best = std::move(buf[0]);
        *m_dev = current_best;
    }
}

template<typename Game>
__global__ void compare_heap_best(typename Game::Heap* heaps_dev,
                                  typename Game::StatePtr* m_dev,
                                  bool* found_dev) {
    extern __shared__ typename Game::StatePtr buf[];

    auto index = threadIdx.x;
    if (index >= num_heaps) return;
    auto& heap = heaps_dev[index];

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

    typename Game::StatePtr heap_best;
    if (index == 0) {
        heap_best = std::move(buf[0]);
        *found_dev = *m_dev && heap_best && (*m_dev)->f <= heap_best->f;
    }
}


#endif // !ASTAR_CUH
