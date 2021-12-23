#include <iostream>
#include <cuda_runtime.h>

#include "common.h"
#include "heap.h"

using HeapType = Heap<uint64_t, unsigned>;
using StateType = State<uint64_t, unsigned>;

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

int main(int argc, char** argv) {
    HeapType heap(1024);

    HeapType* heap_dev;
    HANDLE_RESULT(cudaMalloc(&heap_dev, sizeof(HeapType)))
    HANDLE_RESULT(cudaMemcpy(heap_dev, &heap, sizeof(HeapType), cudaMemcpyHostToDevice))

    unsigned* buf_dev;
    HANDLE_RESULT(cudaMalloc(&buf_dev, 5 * sizeof(unsigned)))

    test<<<1, 1>>>(heap_dev, buf_dev);

    unsigned buf[5];
    HANDLE_RESULT(cudaMemcpy(buf, buf_dev, 5 * sizeof(unsigned), cudaMemcpyDeviceToHost))

    return 0;
}
