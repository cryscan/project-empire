#ifndef ASTAR_CUH
#define ASTAR_CUH

#include <cuda_runtime.h>
#include <limits>
#include "heap.cuh"
#include "common.cuh"

template<typename Game>
__global__ void
init_heaps(typename Game::Heap* heaps_dev,
           typename Game::Node* start_dev,
           typename Game::Node* target_dev,
           size_t index = 0) {
    typename Game::State state;
    state.node = *start_dev;
    state.g = 0;
    state.f = Game::heuristic(*start_dev, *target_dev);
    heaps_dev[index].push(make_arc<typename Game::State>(state));
}

template<typename Game>
__global__ void extract_expand(typename Game::Heap* heaps_dev,
                               typename Game::StatePtr* s_dev,
                               typename Game::StatePtr* m_dev,
                               typename Game::Node* target_dev) {
    extern __shared__ typename Game::StatePtr buf[];

    auto global_index = blockIdx.x * blockDim.x + threadIdx.x;
    auto block_index = blockIdx.x;
    auto index = threadIdx.x;

    if (global_index >= num_heaps) return;
    auto& heap = heaps_dev[global_index];

    if (auto q = heap.pop()) {
        if (q->node == *target_dev) {
            // push candidate
            buf[index] = std::move(q);
        } else {
            // expand the state list
            Game::expand(s_dev, q, *target_dev);
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
__global__ void remove_duplication(typename Game::Hashtable* hashtable_dev,
                                   typename Game::StatePtr* s_dev,
                                   typename Game::StatePtr* t_dev) {
    auto& hashtable = *hashtable_dev;
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (auto& ptr = s_dev[index]) {
        auto state = *ptr;
        auto result = hashtable.find(state.node);

        // not found or is superior to
        if (!result || result->g >= state.g) {
            t_dev[index] = std::move(ptr);
        }
    }
}

template<class Game>
__global__ void reinsert(typename Game::Hashtable* hashtable_dev,
                         typename Game::Heap* heaps_dev,
                         typename Game::StatePtr* t_dev,
                         typename Game::Node* target_dev) {
    auto& hashtable = *hashtable_dev;

    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_heaps) return;
    auto& heap = heaps_dev[index];

    for (auto i = index; i < num_expanded_states; i += num_heaps) {
        if (auto& ptr = t_dev[i]) {
            auto state = *ptr;
            hashtable.insert(state.node, ptr);
            ptr->f = state.g + Game::heuristic(state.node, *target_dev);
            heap.push(ptr);
        }
    }
}


#endif // !ASTAR_CUH
