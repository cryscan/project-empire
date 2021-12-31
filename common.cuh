//
// Created by lepet on 12/21/2021.
//

#ifndef PROJECT_EMPIRE_COMMON_CUH
#define PROJECT_EMPIRE_COMMON_CUH

#include <type_traits>

#define HANDLE_RESULT(expr) {cudaError_t _asdf__err; if ((_asdf__err = expr) != cudaSuccess) { printf("cuda call failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_asdf__err)); exit(1);}}

constexpr size_t num_heaps = 1024;
constexpr size_t max_expansion = 4;
constexpr size_t num_expanded_states = num_heaps * max_expansion;

template<typename Node, typename Value>
struct State;

template<typename T>
class Arc {
    T* ptr;
    int* ref_count;

    __device__ void decrease_and_free() {
        if (ptr && ref_count && (atomicSub(ref_count, 1) == 1)) {
            delete ref_count;
            delete ptr;
        }
    }

    __device__ void increase_ref_count() {
        if (ptr && ref_count) atomicAdd(ref_count, 1);
    }

public:
    __device__ Arc() : ptr(nullptr), ref_count(nullptr) {}

    __device__ explicit Arc(T* ptr) : ptr(ptr), ref_count(nullptr) {
        if (ptr) ref_count = new int(1);
    }

    __device__ ~Arc() {
        decrease_and_free();
    }

    __device__ Arc(const Arc& other) : ptr(other.ptr), ref_count(other.ref_count) {
        increase_ref_count();
    }

    __device__ Arc& operator=(const Arc& other) {
        if (&other != this) {
            decrease_and_free();

            ptr = other.ptr;
            ref_count = other.ref_count;
            increase_ref_count();
        }

        return *this;
    }

    __device__ Arc(Arc&& other) noexcept: ptr(other.ptr), ref_count(other.ref_count) {
        other.ptr = nullptr;
        other.ref_count = nullptr;
    }

    __device__ Arc& operator=(Arc&& other) noexcept {
        if (&other != this) {
            decrease_and_free();

            ptr = other.ptr;
            ref_count = other.ref_count;
            other.ptr = nullptr;
            other.ref_count = nullptr;
        }

        return *this;
    }

    __device__ T* operator->() const { return ptr; }

    __device__ T& operator*() const { return *ptr; }

    template<typename U, typename=std::enable_if_t<std::is_pointer_v<U> || std::is_null_pointer_v<U>>>
    __device__ bool operator==(const U& other) const { return ptr == other; }

    __device__ bool operator==(const Arc& other) const { return ptr == other.ptr; }

    template<typename U>
    __device__ bool operator!=(const U& other) const { return !this->operator==(other); }

    __device__  explicit operator bool() const { return *this != nullptr; }
};

template<typename T, typename ...U>
__device__ Arc<T> make_arc(U&& ...u) {
    return Arc<T>(new T(u...));
}

template<typename T>
__device__ void swap(Arc<T>& a, Arc<T>& b) {
    using std::move;
    Arc<T> temp = move(a);
    a = move(b);
    b = move(temp);
}

template<typename Node, typename Value>
struct State {
    Node node;
    Value g;
    Value f;
    Arc<State<Node, Value>> prev;
};

#endif //PROJECT_EMPIRE_COMMON_CUH
