//
// Created by lepet on 12/21/2021.
//

#ifndef PROJECT_EMPIRE_HEAP_H
#define PROJECT_EMPIRE_HEAP_H

template<typename Node, typename Value>
struct Heap {
public:
    using Ptr = Arc<State<Node, Value>>;

    __device__ Heap(void* heap, size_t capacity) :
            heap(reinterpret_cast<Ptr*>(heap)),
            capacity(capacity),
            size(0) {}

    __device__ void insert() {}

private:
    Arc<State<Node, Value>>* heap;
    size_t capacity, size;
};

#endif //PROJECT_EMPIRE_HEAP_H
