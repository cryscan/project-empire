#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"
#include "heap.h"
#include "hashtable.cuh"
#include "Astar.cuh"

using HeapType = Heap<uint64_t, unsigned>;
using StateType = State<uint64_t, unsigned>;
using TableType = Hashtable<uint64_t, unsigned>;

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


__global__ void test_hash(TableType* table_dev) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    auto key = index;
    auto value = make_arc<StateType>();
    value->node = key;
    table_dev->insert(key, value);
}

__global__ void test_hash_find(TableType* table_dev, uint64_t* buf_dev, bool* bool_dev) {
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

    TableType table(table_size);

    TableType* table_dev;
    HANDLE_RESULT(cudaMalloc(&table_dev, sizeof(TableType)))
    HANDLE_RESULT(cudaMemcpy(table_dev, &table, sizeof(TableType), cudaMemcpyHostToDevice))

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


    uint64_t nodesInS[Directions::Direction::MAX_NEW_STATE_RATIO];
    uint64_t* nodesInS_dev;
    HANDLE_RESULT(cudaMalloc(&nodesInS_dev, Directions::Direction::MAX_NEW_STATE_RATIO * sizeof(uint64_t)))

    unsigned valueInDest;
    unsigned* valueInDest_dev;
    HANDLE_RESULT(cudaMalloc(&valueInDest_dev, sizeof(unsigned)));*/




    return 0;
}
