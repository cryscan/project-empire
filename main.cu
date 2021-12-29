#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"
#include "heap.cuh"
#include "hashtable.cuh"
#include "astar.cuh"

using NodeType = uint64_t;
using ValueType = unsigned int;
using HeapType = Heap<NodeType, ValueType>;
using StateType = State<NodeType, ValueType>;
using HashtableType = Hashtable<NodeType, ValueType>;

enum Direction {
    UP,
    RIGHT,
    DOWN,
    LEFT,
};

constexpr size_t max_expansion = 4;

__device__ ValueType heuristic(NodeType s, NodeType t) {
    return 0;
}

// Write a specific expand function for the sliding pad nodes!
// Can write other expand functions later, but with the same signature.
__device__ Arc<StateType> expand(const Arc<StateType>& state, NodeType t, Direction direction) {
    /*  Board
     *      0   1   2   3
     *      4   5   6   7
     *      8   9   10  11
     *      12  13  14  15
     *
     *  Node
     *      15  14  13  12  11  10  9   8   7   6   5   4   3   2   1   0
     */
    StateType current = *state;
    NodeType filter = 0xf;
    int x, y;
    for (int i = 0; i < 16; i++) {
        if ((current.node & filter) == 0) {
            x = i / 4;
            y = i % 4;
            break;
        }
        filter <<= 4;
    }

    if (direction == UP && x > 0) {
        // select the number on the upper row
        auto selected = current.node & (filter >> 16);
        StateType next;
        next.node = (current.node | (selected << 16)) ^ selected;
        next.g = current.g + 1;
        next.f = next.g + heuristic(next.node, t);
        next.prev = state;
        return make_arc<StateType>(next);
    }

    if (direction == DOWN && x < 3) {
        auto selected = current.node & (filter << 16);
        StateType next;
        next.node = (current.node | (selected >> 16)) ^ selected;
        next.g = current.g + 1;
        next.f = next.g + heuristic(next.node, t);
        next.prev = state;
        return make_arc<StateType>(next);
    }

    if (direction == LEFT && y > 0) {
        auto selected = current.node & (filter >> 4);
        StateType next;
        next.node = (current.node | (selected << 4)) ^ selected;
        next.g = current.g + 1;
        next.f = next.g + heuristic(next.node, t);
        next.prev = state;
        return make_arc<StateType>(next);
    }

    if (direction == RIGHT && y < 3) {
        auto selected = current.node & (filter << 4);
        StateType next;
        next.node = (current.node | (selected >> 4)) ^ selected;
        next.g = current.g + 1;
        next.f = next.g + heuristic(next.node, t);
        next.prev = state;
        return make_arc<StateType>(next);
    }

//    switch (direction) {
//        case UP:
//            if (x > 0) {
//                uint64_t selected = state->node & (filter << 16);
//                return (state->node | (selected >> 16)) ^ selected;
//            }
//            break;
//        case DOWN:
//            if (x < 3) {
//                uint64_t selected = state->node & (filter >> 16);
//                return (state->node | (selected << 16)) ^ selected;
//            }
//            break;
//        case LEFT:
//            if (y > 0) {
//                uint64_t selected = state->node & (filter << 4);
//                return (state->node | (selected >> 4)) ^ selected;
//            }
//            break;
//        case RIGHT:
//            if (y < 3) {
//                uint64_t selected = state->node & (filter >> 4);
//                return (state->node | (selected << 4)) ^ selected;
//            }
//            break;
//    }

    return {};
}

__device__ void expand(Arc<StateType>* s_dev, const Arc<StateType>& state, NodeType t) {
    auto index = threadIdx.x;
    for (auto d: {UP, RIGHT, DOWN, LEFT}) {
        s_dev[index * max_expansion + d] = expand(state, t, d);
    }
}

__global__ void test(HeapType* heap_dev, unsigned* buf) {
    auto s1 = make_arc<StateType>();
    auto s2 = make_arc<StateType>();
    auto s3 = make_arc<StateType>();
    auto s4 = make_arc<StateType>();
    auto s5 = make_arc<StateType>();

    s1->f = 2;
    s2->f = 1;
    s3->f = 5;
    s4->f = 4;
    s5->f = 2;

    heap_dev->push(s1);
    heap_dev->push(s2);
    heap_dev->push(s3);
    heap_dev->push(s4);
    heap_dev->push(s5);

    buf[0] = heap_dev->pop()->f;
    buf[1] = heap_dev->pop()->f;
    buf[2] = heap_dev->pop()->f;
    buf[3] = heap_dev->pop()->f;
    buf[4] = heap_dev->pop()->f;
}


__global__ void test_hash(HashtableType* table_dev) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    auto key = index;
    auto value = make_arc<StateType>();
    value->node = key;
    table_dev->insert(key, value);
}

__global__ void test_hash_find(HashtableType* table_dev, uint64_t* buf_dev, bool* bool_dev) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    auto key = index;

    Arc<StateType> result;
    if (index == 42) key = 1000;
    bool_dev[index] = table_dev->find(key, result);
    if (result) buf_dev[index] = result->node;
}


int main(int argc, char** argv) {
    /*
     * HeapType heap(1024);

    HeapType* heap_dev;
    HANDLE_RESULT(cudaMalloc(&heap_dev, sizeof(HeapType)))
    HANDLE_RESULT(cudaMemcpy(heap_dev, &heap, sizeof(HeapType), cudaMemcpyHostToDevice))

    unsigned* buf_dev;
    HANDLE_RESULT(cudaMalloc(&buf_dev, 5 * sizeof(unsigned)))

    test<< <1, 1> >>(heap_dev, buf_dev);

    unsigned buf[5];
    HANDLE_RESULT(cudaMemcpy(buf, buf_dev, 5 * sizeof(unsigned), cudaMemcpyDeviceToHost))
     */


    /*constexpr size_t thread_count = 1024;
    constexpr size_t table_size = 1024 * 1024;

    HashtableType table(table_size);

    HashtableType* table_dev;
    HANDLE_RESULT(cudaMalloc(&table_dev, sizeof(HashtableType)))
    HANDLE_RESULT(cudaMemcpy(table_dev, &table, sizeof(HashtableType), cudaMemcpyHostToDevice))

    uint64_t* buf_dev;
    HANDLE_RESULT(cudaMalloc(&buf_dev, thread_count * sizeof(uint64_t)));

    bool* bool_dev;
    HANDLE_RESULT(cudaMalloc(&bool_dev, thread_count * sizeof(bool)));

    test_hash<<<1, thread_count>>>(table_dev);

    cudaDeviceSynchronize();

    test_hash_find<<<1, thread_count>>>(table_dev, buf_dev, bool_dev);

    uint64_t buf[thread_count];
    HANDLE_RESULT(cudaMemcpy(buf, buf_dev, thread_count * sizeof(uint64_t), cudaMemcpyDeviceToHost))

    bool bool_buf[thread_count];
    HANDLE_RESULT(cudaMemcpy(bool_buf, bool_dev, thread_count * sizeof(bool), cudaMemcpyDeviceToHost))

    std::cout << "elements: \n";
    for (auto element: buf) {
        std::cout << element << '\n';
    }

    std::cout << "finds: \n";
    for (auto element: bool_buf) {
        std::cout << element << '\n';
    }*/


    /*constexpr uint64_t HEAP_CAPACITY = 1024;
    HeapType h(HEAP_CAPACITY);
    HeapType* h_dev;
    HANDLE_RESULT(cudaMalloc(&h_dev, sizeof(HeapType)))
    HANDLE_RESULT(cudaMemcpy(h_dev, &h, sizeof(HeapType), cudaMemcpyHostToDevice))


    uint64_t nodesInS[Directions::Direction::NUM_DIRECTIONS];
    uint64_t* nodesInS_dev;
    HANDLE_RESULT(cudaMalloc(&nodesInS_dev, Directions::Direction::NUM_DIRECTIONS * sizeof(uint64_t)))

    unsigned valueInDest;
    unsigned* valueInDest_dev;
    HANDLE_RESULT(cudaMalloc(&valueInDest_dev, sizeof(unsigned)));*/




    return 0;
}
