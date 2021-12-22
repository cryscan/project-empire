#include <iostream>
#include <cuda_runtime.h>

#include "common.h"
#include "heap.h"

struct Test {
    int a;
    int b;
};

__global__ void test(void* heap_dev) {
    auto ptr = make_arc<Test>();
    ptr->a = 3;

    Heap<Test, int> heap(heap_dev, 1024);
}

int main(int argc, char** argv) {
    void* heap_dev;
    HANDLE_RESULT(cudaMalloc(&heap_dev, 1024 * sizeof(void*)))

    test<<<1, 64>>>(heap_dev);

    return 0;
}
