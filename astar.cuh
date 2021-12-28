#ifndef ASTAR_CUH
#define ASTAR_CUH

#include <cuda_runtime.h>
#include <limits>
#include "heap.h"
#include "common.h"

namespace Directions
{
    enum Direction : size_t
    {
        UP,
        RIGHT,
        DOWN,
        LEFT,
        MAX_NEW_STATE_RATIO
    };

    static const size_t All[] = { UP, RIGHT, DOWN, LEFT };
}

constexpr size_t THREAD_NUM = 1;

template <typename Node, typename Value>
__device__ Arc<State<Node, Value>> expand(const Arc<State<Node, Value>>& origin, 
                                          Directions::Direction direction) {
    Arc<State<Node, Value>> output = make_arc<Node, Value>();
    // temp test value
    output->f = 10;
    output->prev = origin;
    return output;
}

template <typename Node, typename Value>
__device__ void extract_expand(Heap<Node, Value>& PQ, 
                               Arc<State<Node, Value>>& destination,
                               Arc<State<Node, Value>>* S) {
    
    if (PQ.get_size() == 0) return;
    
    Arc<State<Node, Value>> q = PQ.pop();
    if (q->node == destination->node) {
        if (destination->f == std::numeric_limits<Node>::max()
            || q->f < destination->f) {
            destination = q;
        }
        return;
    }

    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    HANDLE_RESULT(cudaMalloc(&S, Directions::Direction::MAX_NEW_STATE_RATIO * THREAD_NUM))

    for (const Directions::Direction d : Directions::All) {
        S[index + d * THREAD_NUM] = expand(q, d);
    }
    return;
}



#endif // !ASTAR_CUH
