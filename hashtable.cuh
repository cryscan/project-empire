#ifndef PROJECT_EMPIRE_HASHTABLE_CUH
#define PROJECT_EMPIRE_HASHTABLE_CUH

// #include "hashtable.cu"
#include "common.cuh"
#include <cstdlib>
#include <device_atomic_functions.h>

template<typename Node, typename Value, size_t CAPACITY>
class Hashtable {
public:
    using StatePtr = Arc<State<Node, Value>>;

    explicit Hashtable() : capacity(CAPACITY), states(nullptr), locks(nullptr) {
        HANDLE_RESULT(cudaMalloc(&states, capacity * sizeof(StatePtr)))
        HANDLE_RESULT(cudaMemset(states, 0, capacity * sizeof(StatePtr)))
        HANDLE_RESULT(cudaMalloc(&locks, capacity * sizeof(int)))
        HANDLE_RESULT(cudaMemset(locks, 0, capacity * sizeof(int)))
    }

    __device__ void insert(Node& key, const StatePtr& state) {
        size_t slot = hash(key);
        for (auto i = 0u; i < 32; ++i) {
            if (i != threadIdx.x % 32) continue;

            while (atomicExch(&locks[slot], 1) == 1);
            states[slot] = state;
            atomicExch(&locks[slot], 0);
        }
    }

    __device__ StatePtr find(const Node& key) const {
        // step1: hash
        size_t slot = hash(key);

        // step2: check
        auto ptr = states[slot];
        if (ptr == nullptr || ptr->node != key) return nullptr;

        // step3: if found, update output
        return ptr;
    }

private:
    size_t capacity;
    StatePtr* states;
    int* locks;

    __device__ size_t hash(Node key) const {
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

        return hashed % capacity;
    }
};

template<typename Node, typename Value, size_t COUNT, size_t CAPACITY>
Hashtable<Node, Value, CAPACITY>* make_hashtable() {
    using HashtableType = Hashtable<Node, Value, CAPACITY>;
    HashtableType hashtable[COUNT];

    HashtableType* dev;
    HANDLE_RESULT(cudaMalloc(&dev, COUNT * sizeof(HashtableType)))
    HANDLE_RESULT(cudaMemcpy(dev, hashtable, COUNT * sizeof(HashtableType), cudaMemcpyHostToDevice))
    return dev;
}


#endif // !PROJECT_EMPIRE_HASHTABLE_CUH
