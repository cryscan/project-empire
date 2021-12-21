#include <iostream>
#include <cuda_runtime.h>

#include "common.h"
#include "heap.h"

struct Test {
    int a;
    int b;
};

__global__ void test() {
    Arc<Test> ptr(new Test);
    ptr->a = 3;
}

int main(int argc, char** argv) {
    test<<<1, 1024>>>();

    return 0;
}
