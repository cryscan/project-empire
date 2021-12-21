//
// Created by lepet on 12/21/2021.
//

#ifndef PROJECT_EMPIRE_COMMON_H
#define PROJECT_EMPIRE_COMMON_H

template<typename Node, typename Value>
struct State;

template<typename T>
class Arc {
    T* ptr;
    int* ref_count;

public:
    __device__ explicit Arc(T* ptr) : ptr(ptr) {
        ref_count = new int(1);
    }

    __device__ ~Arc() {
        int old_ref_count = atomicAdd(ref_count, -1);
        if (old_ref_count - 1 <= 0) {
            delete ref_count;
            delete ptr;
        }
    }

    __device__ T* operator->() const { return ptr; }

    __device__ T& operator*() const { return *ptr; }
};

template<typename Node, typename Value>
struct State {
    Node node;
    Value g, f;
};

#endif //PROJECT_EMPIRE_COMMON_H
