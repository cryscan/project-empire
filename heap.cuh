//
// Created by lepet on 12/21/2021.
//

#ifndef PROJECT_EMPIRE_HEAP_CUH
#define PROJECT_EMPIRE_HEAP_CUH

#include <cassert>

template<typename Node, typename Value>
class Heap {
public:
    using StatePtr = Arc<State<Node, Value>>;

    explicit Heap(size_t capacity = heap_size) : size(0), capacity(capacity) {
        HANDLE_RESULT(cudaMalloc(&states, capacity * sizeof(StatePtr)))
        HANDLE_RESULT(cudaMemset(states, 0, capacity * sizeof(StatePtr)))
    }

    ~Heap() {
        HANDLE_RESULT(cudaFree(states))
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
        if (a->f < b->f) return true;
        else if (a->f > b->f) return false;
        else return a->g > b->g;
    }

    __device__ size_t parent(size_t index) { return (index - 1) / 2; }

    __device__ size_t left_child(size_t index) { return index * 2 + 1; }

    __device__ size_t right_child(size_t index) { return index * 2 + 2; }
};

#endif //PROJECT_EMPIRE_HEAP_CUH
