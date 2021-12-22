//
// Created by lepet on 12/21/2021.
//

#ifndef PROJECT_EMPIRE_COMMON_H
#define PROJECT_EMPIRE_COMMON_H

#define HANDLE_RESULT(expr) {cudaError_t _asdf__err; if ((_asdf__err = expr) != cudaSuccess) { printf("cuda call failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_asdf__err)); exit(1);}}

template<typename Node, typename Value>
struct State;

template<typename T>
class Arc {
    T* ptr;
    unsigned int* ref_count;

public:
    __device__ Arc() : ptr(nullptr), ref_count(nullptr) {}

    __device__ explicit Arc(T* ptr) : ptr(ptr), ref_count(nullptr) {
        if (ptr) ref_count = new unsigned int(1);
    }

    __device__ ~Arc() {
        if (ptr && atomicSub(ref_count, 1) == 1) {
            delete ref_count;
            delete ptr;
        }
    }

    __device__ Arc(const Arc<T>& other) : ptr(other.ptr), ref_count(other.ref_count) {
        if (ptr) atomicAdd(ref_count, 1);
    }

    __device__ Arc& operator=(const Arc<T>& other) {
        if (&other != this) {
            if (ptr && atomicSub(ref_count, 1) == 1) {
                delete ref_count;
                delete ptr;
            }

            ptr = other.ptr;
            ref_count = other.ref_count;
            if (ptr) atomicAdd(ref_count, 1);
        }

        return *this;
    }

    __device__ T* operator->() const { return ptr; }

    __device__ T& operator*() const { return *ptr; }

    template<typename Ptr>
    __device__ bool operator==(const Ptr& other) const { return ptr == other; }

    __device__ bool operator==(const Arc<T>& other) const { return ptr == other.ptr; }
};

template<typename T, typename ...U>
__device__ Arc<T> make_arc(U&& ...u) {
    return Arc<T>(new T(u...));
}

template<typename Node, typename Value>
struct State {
    Node node;
    Value g, f;
};

#endif //PROJECT_EMPIRE_COMMON_H
