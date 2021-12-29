#ifndef ASTAR_CUH
#define ASTAR_CUH

#include <cuda_runtime.h>
#include <limits>
#include "heap.cuh"
#include "common.cuh"

constexpr size_t num_threads = 1024;

template<typename Node, typename Value, typename ExpandFunc>
__global__ void extract_expand(Heap<Node, Value>* queues_dev,
                               Arc<State<Node, Value>>* s_dev,
                               Arc<State<Node, Value>>* m_dev,
                               const Node& t,
                               ExpandFunc expand) {
    auto index = threadIdx.x;
    __shared__ Arc<State<Node, Value>> buf[num_threads];

    auto& queue = queues_dev[index];

    if (auto q = queue.pop()) {
        if (q->node == t) {
            // push candidate
            buf[index] = std::move(q);
        } else {
            // expand the state list
            expand(s_dev, q, t);
        }
    }

    __syncthreads();

    // get the best target state
    auto i = num_threads;
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
    return;
}


#endif // !ASTAR_CUH
