#ifndef PROJECT_EMPIRE_HASHTABLE_CUH
#define PROJECT_EMPIRE_HASHTABLE_CUH

// #include "hashtable.cu"
#include "common.cuh"
#include <cstdlib>
#include <device_atomic_functions.h>

template<typename Node, typename Value>
class Hashtable {
public:
    using StatePtr = Arc<State<Node, Value>>;

    explicit Hashtable(size_t capacity) : capacity(capacity), states(nullptr), locks(nullptr) {
        HANDLE_RESULT(cudaMalloc(&states, capacity * sizeof(StatePtr)))
        HANDLE_RESULT(cudaMemset(states, 0, capacity * sizeof(StatePtr)))
        HANDLE_RESULT(cudaMalloc(&locks, capacity * sizeof(int)))
        HANDLE_RESULT(cudaMemset(locks, 0, capacity * sizeof(int)))
    }

    ~Hashtable() {
        HANDLE_RESULT(cudaFree(states))
        HANDLE_RESULT(cudaFree(locks))
    }

    __device__ void insert(Node& key, const StatePtr& state_ptr) {
        size_t slot = hash(key, threadIdx.x % 32);
        while (atomicExch(&locks[slot], 1) == 1);
        states[slot] = state_ptr;
        atomicExch(&locks[slot], 0);
    }

    __device__ StatePtr find(const Node& key) const {
        // step1: hash
        size_t slot = hash(key, threadIdx.x % 32);

        // step2: check initial & check node == key
        auto ptr = states[slot];
        if (ptr == nullptr) return {};
        if (ptr->node != key) return {};

        // step3: if found, update output
        return ptr;
    }

private:
    size_t capacity;
    StatePtr* states;
    int* locks;

    __device__ size_t hash(Node key, unsigned int index) const {
        size_t hashed = 0;
        {
            uint32_t temp = key;
            temp ^= temp >> 16;
            temp *= 0x85ebca6b;
            temp ^= temp >> 13;
            temp *= 0xc2b2ae35;
            temp ^= temp >> 16;
            hashed += temp;
        }
        {
            uint32_t temp = key >> 32;
            temp ^= temp >> 16;
            temp *= 0x85ebca6b;
            temp ^= temp >> 13;
            temp *= 0xc2b2ae35;
            temp ^= temp >> 16;
            hashed += (size_t) temp << 32;
        }

        // make sure threads in one wrap are mapped into different slots
        hashed <<= 5;
        hashed += index;
        return hashed % capacity;
    }
};


#endif // !PROJECT_EMPIRE_HASHTABLE_CUH
