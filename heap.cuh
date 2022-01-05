//
// Created by lepet on 12/21/2021.
//

#ifndef PROJECT_EMPIRE_HEAP_CUH
#define PROJECT_EMPIRE_HEAP_CUH

#include <cassert>

template<typename Node, typename Value, size_t CAPACITY>
class Heap {
public:
    using StatePtr = Arc<State<Node, Value>>;

    explicit Heap() : size(0), capacity(CAPACITY) {
        HANDLE_RESULT(cudaMalloc(&states, capacity * sizeof(StatePtr)))
        HANDLE_RESULT(cudaMemset(states, 0, capacity * sizeof(StatePtr)))
    }

    __device__ void push(const StatePtr& state) {
        assert(size <= capacity);

        states[size] = state;
        auto current = size;
        while (current > 0 && comp(states[current], states[parent(current)])) {
            swap(states[current], states[parent(current)]);
            current = parent(current);
        }
        ++size;
    }

    __device__ StatePtr pop() {
        if (size == 0) return nullptr;

        StatePtr result = std::move(states[0]);
        states[0] = std::move(states[size - 1]);
        --size;

        size_t current = 0;
        while (current < size) {
            auto smallest = current;

            auto child = left_child(current);
            if (child < size && comp(states[child], states[smallest]))
                smallest = child;

            child = right_child(current);
            if (child < size && comp(states[child], states[smallest]))
                smallest = child;

            if (smallest == current) break;
            swap(states[current], states[smallest]);
            current = smallest;
        }
        return result;
    }

    __device__ StatePtr* data() const { return states; }

private:
    StatePtr* states;
    size_t size, capacity;

    __device__ bool comp(const StatePtr& a, const StatePtr& b) const {
        return a->f < b->f;
    }

    __device__ size_t parent(size_t index) { return (index - 1) / 2; }

    __device__ size_t left_child(size_t index) { return index * 2 + 1; }

    __device__ size_t right_child(size_t index) { return index * 2 + 2; }
};

template<typename Node, typename Value, size_t COUNT, size_t CAPACITY>
Heap<Node, Value, CAPACITY>* make_heaps() {
    using HeapType = Heap<Node, Value, CAPACITY>;
    HeapType heaps[COUNT];

    HeapType* dev;
    HANDLE_RESULT(cudaMalloc(&dev, COUNT * sizeof(HeapType)))
    HANDLE_RESULT(cudaMemcpy(dev, heaps, COUNT * sizeof(HeapType), cudaMemcpyHostToDevice))
    return dev;
}

#endif //PROJECT_EMPIRE_HEAP_CUH
