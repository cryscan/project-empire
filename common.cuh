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

constexpr size_t num_iterations = 6144;
constexpr size_t heap_size = 4 * num_iterations;
constexpr size_t hashtable_size = num_iterations * num_iterations;
constexpr size_t pool_size = num_expanded_states * num_iterations;

constexpr size_t solution_size = 100;

template<typename Node, typename Value>
struct State;

template<typename T>
class Arc {
    T* ptr;
    // int* ref_count;

    __device__ friend void swap(Arc& a, Arc& b) {
        T* temp_ptr = a.ptr;
        a.ptr = b.ptr;
        b.ptr = temp_ptr;
    }

    __device__ void decrease_and_free() {
        // if (ptr && ref_count && (atomicSub(ref_count, 1) == 1)) {
        // delete ref_count;
        // delete ptr;
        // }
    }

    __device__ void increase_ref_count() {
        // if (ptr && ref_count) atomicAdd(ref_count, 1);
    }

public:
    __device__ Arc() : ptr(nullptr) {}

    __device__ Arc(nullptr_t) : ptr(nullptr) {}

    __device__ explicit Arc(T* ptr) : ptr(ptr) {}

    __device__ ~Arc() {
        decrease_and_free();
    }

    __device__ Arc(const Arc& other) : ptr(other.ptr) {
        increase_ref_count();
    }

    __device__ Arc& operator=(const Arc& other) {
        if (&other != this) {
            Arc temp(ptr);
            ptr = other.ptr;
            increase_ref_count();
        }

        return *this;
    }

    __device__ Arc& operator=(nullptr_t) {
        Arc temp(ptr);
        ptr = nullptr;
        return *this;
    }

    __device__ Arc(Arc&& other) noexcept: ptr(other.ptr) {
        other.ptr = nullptr;
    }

    __device__ Arc& operator=(Arc&& other) noexcept {
        if (&other != this) {
            Arc temp(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
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

template<typename Node, typename Value>
struct State {
    Node node;
    Value g;
    Value f;
    Arc<State<Node, Value>> prev;
};

template<typename Node, typename Value>
struct SerializedState {
    Node node;
    Value g;
    Value f;

    SerializedState() = default;

    __device__ explicit SerializedState(const State<Node, Value>& state) : node(state.node), g(state.g), f(state.f) {}
};

template<typename T, size_t CAPACITY>
class Pool {
public:
    explicit Pool() : size(0), capacity(CAPACITY), memory(nullptr) {
        HANDLE_RESULT(cudaMalloc(&memory, capacity * sizeof(T)))
        HANDLE_RESULT(cudaMemset(memory, 0, capacity * sizeof(T)))
    }

    __device__ Arc<T> allocate() {
        assert(size < capacity);

        auto old = atomicAdd(&size, 1);
        return Arc<T>(memory + old);
    }

    __device__ Arc<T> allocate(const T& t) {
        assert(size < capacity);

        auto old = atomicAdd(&size, 1);
        auto ptr = Arc<T>(memory + old);
        *ptr = t;
        return ptr;
    }

private:
    T* memory;
    size_t size, capacity;
};

template<typename T, size_t CAPACITY>
Pool<T, CAPACITY>* make_pool() {
    using PoolType = Pool<T, CAPACITY>;
    PoolType pool;

    PoolType* dev;
    HANDLE_RESULT(cudaMalloc(&dev, sizeof(PoolType)))
    HANDLE_RESULT(cudaMemcpy(dev, &pool, sizeof(PoolType), cudaMemcpyHostToDevice))
    return dev;
}

#endif //PROJECT_EMPIRE_COMMON_CUH
