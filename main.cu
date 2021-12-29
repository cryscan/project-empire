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
__device__ Arc<StateType> expand_direction(const Arc<StateType>& state, NodeType t, Direction direction) {
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

    return {};
}

struct SlidingPadExpandFunc {
    __device__ void operator()(Arc<StateType>* s_dev, const Arc<StateType>& state, NodeType t) {
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        for (auto d: {UP, RIGHT, DOWN, LEFT}) {
            s_dev[index * max_expansion + d] = expand_direction(state, t, d);
        }
    }
};

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

__global__ void init_heaps(HeapType* heaps_dev, NodeType s, NodeType t) {
    StateType state;
    state.node = s;
    state.g = 0;
    state.f = heuristic(s, t);
    heaps_dev[0].push(make_arc<StateType>(state));
}

__global__ void extract_nodes(Arc<StateType>* s_dev, NodeType* nodes_dev) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    nodes_dev[index] = s_dev[index] ? s_dev[index]->node : 0;
}

int main(int argc, char** argv) {
    std::vector<HeapType> heaps(num_threads);

    HeapType* heaps_dev;
    HANDLE_RESULT(cudaMalloc(&heaps_dev, num_threads * sizeof(HeapType)))
    HANDLE_RESULT(cudaMemcpy(heaps_dev, heaps.data(), num_threads * sizeof(HeapType), cudaMemcpyHostToDevice))

    Arc<StateType>* s_dev;
    HANDLE_RESULT(cudaMalloc(&s_dev, num_threads * max_expansion * sizeof(Arc<StateType>)))
    HANDLE_RESULT(cudaMemset(s_dev, 0, num_threads * max_expansion * sizeof(Arc<StateType>)))

    Arc<StateType>* m_dev;
    HANDLE_RESULT(cudaMalloc(&m_dev, sizeof(Arc<StateType>)))

    // test section
    // test function here
    NodeType s = 0xfedcba9876543210;
    NodeType t = 0x0123456789abcdef;

    init_heaps<<<1, 1>>>(heaps_dev, s, t);
    extract_expand<<<1, num_threads, num_threads * sizeof(Arc<StateType>)>>>(
            heaps_dev,
            s_dev,
            m_dev,
            t,
            SlidingPadExpandFunc());

    NodeType nodes_cpu[num_threads * max_expansion];
    NodeType* nodes_dev;
    HANDLE_RESULT(cudaMalloc(&nodes_dev, num_threads * max_expansion * sizeof(NodeType)))

    // extract nodes from pointers
    extract_nodes<<<max_expansion, num_threads>>>(s_dev, nodes_dev);

    HANDLE_RESULT(
            cudaMemcpy(nodes_cpu, nodes_dev, num_threads * max_expansion * sizeof(NodeType), cudaMemcpyDeviceToHost))

    HANDLE_RESULT(cudaFree(heaps_dev))
    HANDLE_RESULT(cudaFree(s_dev))
    HANDLE_RESULT(cudaFree(m_dev))

    HANDLE_RESULT(cudaFree(nodes_dev))

    // test <<< 1, 1 >>>(heap_dev, buf_dev);

    /*
    constexpr size_t thread_count = 1024;
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
