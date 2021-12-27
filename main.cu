#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"
#include "heap.h"
#include "hashtable.cuh"

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


__global__ void test_hash(TableType* table_dev, unsigned* buf_dev, bool* bool_dev) {
    size_t gindex = threadIdx.x + blockIdx.x * blockDim.x;
    buf_dev[gindex] = gindex;
    // if (gindex > 0) return;
    uint64_t v1 = gindex;
    Arc<StateType> s1 = make_arc<StateType>();
    s1->node = v1;
    Arc<StateType> out;
    if (gindex == 0 || gindex == 32)  table_dev->insert(v1, s1);
    // cudaDeviceSynchronize();
    // bool_dev[gindex] = 1;
    // table_dev->find(10, out);
    // bool_dev[gindex] = table_dev->find(v1, out);
    //if (bool_dev[gindex])
    //{
    //    buf_dev[gindex] = out->node;
    //}

}

int main(int argc, char** argv) {
    /*HeapType heap(1024);

    HeapType* heap_dev;
    HANDLE_RESULT(cudaMalloc(&heap_dev, sizeof(HeapType)))
    HANDLE_RESULT(cudaMemcpy(heap_dev, &heap, sizeof(HeapType), cudaMemcpyHostToDevice))

    unsigned* buf_dev;
    HANDLE_RESULT(cudaMalloc(&buf_dev, 5 * sizeof(unsigned)))

    test<< <1, 1> >>(heap_dev, buf_dev);

    unsigned buf[5];
    HANDLE_RESULT(cudaMemcpy(buf, buf_dev, 5 * sizeof(unsigned), cudaMemcpyDeviceToHost))*/
    constexpr size_t thread_count = 64;

    TableType table(thread_count);
    
    TableType* table_dev;
    HANDLE_RESULT(cudaMalloc(&table_dev, sizeof(TableType)))
    HANDLE_RESULT(cudaMemcpy(table_dev, &table, sizeof(TableType), cudaMemcpyHostToDevice))

    unsigned* buf_dev;
    HANDLE_RESULT(cudaMalloc(&buf_dev, thread_count * sizeof(unsigned)));

    bool* bool_dev;
    HANDLE_RESULT(cudaMalloc(&bool_dev, thread_count * sizeof(bool)));

    test_hash << <1, thread_count>> > (table_dev, buf_dev, bool_dev);

    unsigned buf[thread_count];
    HANDLE_RESULT(cudaMemcpy(buf, buf_dev, thread_count * sizeof(unsigned), cudaMemcpyDeviceToHost))

    bool bool_buf[thread_count];
    HANDLE_RESULT(cudaMemcpy(bool_buf, bool_dev, thread_count * sizeof(bool), cudaMemcpyDeviceToHost))

    std::cout << "elements: \n";
    for (auto element : buf)
    {
        std::cout << element << '\n';
    }
    
    std::cout << "finds: \n";
    for (auto element : bool_buf)
    {
        std::cout << element << '\n';
    }
    return 0;
}
