//
// Created by lepet on 12/21/2021.
//

#ifndef PROJECT_EMPIRE_HEAP_H
#define PROJECT_EMPIRE_HEAP_H

template<typename Node, typename Value>
class Heap {
public:
    using StatePtr = Arc<State<Node, Value>>;

    explicit Heap(size_t capacity) :
            capacity(capacity),
            size(0) {
        HANDLE_RESULT(cudaMalloc(&states, capacity * sizeof(StatePtr)))
        HANDLE_RESULT(cudaMemset(states, 0, capacity * sizeof(StatePtr)))
    }

    ~Heap() {
        HANDLE_RESULT(cudaFree(states))
    }

    __device__ void push(StatePtr state) {
        using std::move;

        states[size] = move(state);
        auto current = size;
        while (current > 0 && *states[current] < *states[parent(current)]) {
            swap(states[current], states[parent(current)]);
            current = parent(current);
        }
        ++size;
    }

    __device__ StatePtr pop() {
        using std::move;

        if (size == 0) return StatePtr();

        auto result = move(states[0]);
        states[0] = move(states[size - 1]);
        --size;

        size_t current = 0;
        while (current < size) {
            auto smallest = current;

            auto child = left_child(current);
            if (child < size && *states[child] < *states[smallest])
                smallest = child;

            child = right_child(current);
            if (child < size && *states[child] < *states[smallest])
                smallest = child;

            if (smallest == current) break;
            swap(states[current], states[smallest]);
            current = smallest;
        }
        return result;
    }

private:
    StatePtr* states;
    size_t capacity, size;

    __device__ size_t parent(size_t index) { return (index - 1) / 2; }

    __device__ size_t left_child(size_t index) { return index * 2 + 1; }

    __device__ size_t right_child(size_t index) { return index * 2 + 2; }
};

#endif //PROJECT_EMPIRE_HEAP_H
