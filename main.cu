#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common.cuh"
#include "heap.cuh"
#include "hashtable.cuh"
#include "astar.cuh"

struct SlidingPad {
    using Node = uint64_t;
    using Value = unsigned int;
    using Heap = Heap<Node, Value>;
    using State = State<Node, Value>;
    using StatePtr = Arc<State>;
    using SerializedState = SerializedState<Node, Value>;
    using Hashtable = Hashtable<Node, Value>;

    enum Direction {
        UP = 0,
        RIGHT,
        DOWN,
        LEFT,
    };

    static __device__ Value heuristic(Node s, Node t) {
        Node filter = 0xf;
        Value result = 0;
        for (int i = 0; i < 16; ++i, filter <<= 4) {
            auto x = (s & filter) >> (4 * i);
            auto y = (t & filter) >> (4 * i);
            result += x > y ? x - y : y - x;
        }
        return result;
    }

    static __device__ StatePtr expand_direction(const StatePtr& state, Node t, Direction direction) {
        /*  Board
         *      0   1   2   3
         *      4   5   6   7
         *      8   9   10  11
         *      12  13  14  15
         *
         *  Node
         *      15  14  13  12  11  10  9   8   7   6   5   4   3   2   1   0
         */
        State current = *state;
        Node filter = 0xf;
        int x, y;
        for (int i = 0; i < 16; ++i, filter <<= 4) {
            if ((current.node & filter) == 0) {
                x = i / 4;
                y = i % 4;
                break;
            }
        }

        if (direction == UP && x > 0) {
            auto selected = current.node & (filter >> 16);
            State next;
            next.node = (current.node | (selected << 16)) ^ selected;
            next.g = current.g + 1;
            // next.f = next.g + heuristic(next.node, t);
            next.prev = state;
            return make_arc<State>(next);
        }

        if (direction == DOWN && x < 3) {
            auto selected = current.node & (filter << 16);
            State next;
            next.node = (current.node | (selected >> 16)) ^ selected;
            next.g = current.g + 1;
            // next.f = next.g + heuristic(next.node, t);
            next.prev = state;
            return make_arc<State>(next);
        }

        if (direction == LEFT && y > 0) {
            auto selected = current.node & (filter >> 4);
            State next;
            next.node = (current.node | (selected << 4)) ^ selected;
            next.g = current.g + 1;
            // next.f = next.g + heuristic(next.node, t);
            next.prev = state;
            return make_arc<State>(next);
        }

        if (direction == RIGHT && y < 3) {
            auto selected = current.node & (filter << 4);
            State next;
            next.node = (current.node | (selected >> 4)) ^ selected;
            next.g = current.g + 1;
            // next.f = next.g + heuristic(next.node, t);
            next.prev = state;
            return make_arc<State>(next);
        }

        return {};
    }

    static __device__ void expand(StatePtr* s_dev, const StatePtr& state, Node t) {
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        for (auto d: {UP, RIGHT, DOWN, LEFT}) {
            s_dev[index * max_expansion + d] = expand_direction(state, t, d);
        }
    }
};

/*
__global__ void test(Heap* heap_dev, unsigned* buf) {
    auto s1 = make_arc<State>();
    auto s2 = make_arc<State>();
    auto s3 = make_arc<State>();
    auto s4 = make_arc<State>();
    auto s5 = make_arc<State>();

    s1->f = 2;
    s2->f = 1;
    s3->f = 5;
    s4->f = 4;
    s5->f = 2;

    heap_dev->push(s1);
    heap_dev->push(s2);
    heap_dev->push(s3);
    heap_dev->push(s4);
    heap_dev->push(s5);

    buf[0] = heap_dev->pop()->f;
    buf[1] = heap_dev->pop()->f;
    buf[2] = heap_dev->pop()->f;
    buf[3] = heap_dev->pop()->f;
    buf[4] = heap_dev->pop()->f;
}

__global__ void test_hash(HashtableType* table_dev) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    auto key = index;
    auto value = make_arc<State>();
    value->node = key;
    table_dev->insert(key, value);
}

__global__ void test_hash_find(HashtableType* table_dev, uint64_t* buf_dev, bool* bool_dev) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    auto key = index;

    Arc<State> result;
    if (index == 42) key = 1000;
    bool_dev[index] = table_dev->find(key, result);
    if (result) buf_dev[index] = result->node;
}
 */

template<typename Game>
__global__ void extract_states(typename Game::StatePtr* s_dev, typename Game::SerializedState* states_dev) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (auto& ptr = s_dev[index]) {
        states_dev[index] = typename Game::SerializedState(*ptr);
    }
}

template<typename Game>
__global__ void extract_chain(typename Game::StatePtr* m_dev, typename Game::SerializedState* solution_dev) {
    auto ptr = *m_dev;
    for (auto i = 0u; ptr; ++i) {
        auto state = *ptr;
        solution_dev[i] = typename Game::SerializedState(state);
        ptr = state.prev;
    }
}

int main(int argc, char** argv) {
    using Game = SlidingPad;

    std::vector<Game::Heap> heaps(num_heaps);
    Game::Hashtable hashtable(1024 * 1024);

    Game::Heap* heaps_dev;
    HANDLE_RESULT(cudaMalloc(&heaps_dev, num_heaps * sizeof(Game::Heap)))
    HANDLE_RESULT(cudaMemcpy(heaps_dev, heaps.data(), num_heaps * sizeof(Game::Heap), cudaMemcpyHostToDevice))

    Game::Hashtable* hashtable_dev;
    HANDLE_RESULT(cudaMalloc(&hashtable_dev, sizeof(Game::Hashtable)))
    HANDLE_RESULT(cudaMemcpy(hashtable_dev, &hashtable, sizeof(Game::Hashtable), cudaMemcpyHostToDevice))

    Game::StatePtr* s_dev;
    HANDLE_RESULT(cudaMalloc(&s_dev, num_expanded_states * sizeof(Game::StatePtr)))
    HANDLE_RESULT(cudaMemset(s_dev, 0, num_expanded_states * sizeof(Game::StatePtr)))

    Game::StatePtr* t_dev;
    HANDLE_RESULT(cudaMalloc(&t_dev, num_expanded_states * sizeof(Game::StatePtr)))
    HANDLE_RESULT(cudaMemset(t_dev, 0, num_expanded_states * sizeof(Game::StatePtr)))

    Game::StatePtr* m_dev;
    HANDLE_RESULT(cudaMalloc(&m_dev, sizeof(Game::StatePtr)))
    HANDLE_RESULT(cudaMemset(m_dev, 0, sizeof(Game::StatePtr)))

    bool found;
    bool* found_dev;
    HANDLE_RESULT(cudaMalloc(&found_dev, sizeof(bool)))

    Game::Node start = 0xfedcba9876543210;
    Game::Node target = 0x0123456789abcdef;
    // Game::Node target = 0xF0DCBEA976583214;

    Game::Node* start_dev;
    HANDLE_RESULT(cudaMalloc(&start_dev, sizeof(Game::Node)))
    HANDLE_RESULT(cudaMemcpy(start_dev, &start, sizeof(Game::Node), cudaMemcpyHostToDevice))

    Game::Node* target_dev;
    HANDLE_RESULT(cudaMalloc(&target_dev, sizeof(Game::Node)))
    HANDLE_RESULT(cudaMemcpy(target_dev, &target, sizeof(Game::Node), cudaMemcpyHostToDevice))

    init_heaps<Game><<<1, 1>>>(heaps_dev, start_dev, target_dev);
    HANDLE_RESULT(cudaGetLastError())

    for (int i = 0; i < 1024; ++i) {
        std::cout << "Iteration " << i << '\n';

        extract_expand<Game><<<num_heaps / 1024, num_heaps, num_heaps * sizeof(Game::StatePtr)>>>(
                heaps_dev,
                s_dev,
                m_dev,
                target_dev);
        HANDLE_RESULT(cudaGetLastError())

        compare_heap_best<Game><<<num_heaps / 1024, num_heaps, num_heaps * sizeof(Game::StatePtr)>>>(
                heaps_dev,
                m_dev,
                found_dev);
        HANDLE_RESULT(cudaGetLastError())

        HANDLE_RESULT(cudaMemcpy(&found, found_dev, sizeof(bool), cudaMemcpyDeviceToHost))
        if (found) break;

        remove_duplication<Game><<<max_expansion, num_heaps>>>(hashtable_dev, s_dev, t_dev);
        HANDLE_RESULT(cudaGetLastError())

        reinsert<Game><<<num_heaps / 1024, num_heaps>>>(hashtable_dev, heaps_dev, t_dev, target_dev);
        HANDLE_RESULT(cudaGetLastError())
    }

    Game::SerializedState solution[1024];
    Game::SerializedState* solution_dev;
    HANDLE_RESULT(cudaMalloc(&solution_dev, 1024 * sizeof(Game::SerializedState)))
    HANDLE_RESULT(cudaMemset(solution_dev, 0, 1024 * sizeof(Game::SerializedState)))

    extract_chain<Game><<<1, 1>>>(m_dev, solution_dev);

    HANDLE_RESULT(cudaMemcpy(solution, solution_dev, 1024 * sizeof(Game::SerializedState), cudaMemcpyDeviceToHost))

    for (auto x: solution) {
        if (x.node == 0) break;
        std::cout << x.node << ' ' << x.g << ' ' << x.f << std::endl;
    }

    /*
    std::vector<Game::SerializedState> s_states(num_expanded_states);
    Game::SerializedState* s_states_dev;
    HANDLE_RESULT(cudaMalloc(&s_states_dev, num_expanded_states * sizeof(Game::SerializedState)))

    std::vector<Game::SerializedState> t_states(num_expanded_states);
    Game::SerializedState* t_states_dev;
    HANDLE_RESULT(cudaMalloc(&t_states_dev, num_expanded_states * sizeof(Game::SerializedState)))

    // extract nodes from pointers
    extract_states<Game><<<max_expansion, num_heaps>>>(s_dev, s_states_dev);
    extract_states<Game><<<max_expansion, num_heaps>>>(t_dev, t_states_dev);

    HANDLE_RESULT(cudaMemcpy(
            s_states.data(),
            s_states_dev,
            num_expanded_states * sizeof(Game::SerializedState),
            cudaMemcpyDeviceToHost))

    HANDLE_RESULT(cudaMemcpy(
            t_states.data(),
            t_states_dev,
            num_expanded_states * sizeof(Game::SerializedState),
            cudaMemcpyDeviceToHost))

    // HANDLE_RESULT(cudaFree(heaps_dev))
    // HANDLE_RESULT(cudaFree(s_dev))
    // HANDLE_RESULT(cudaFree(t_dev))
    // HANDLE_RESULT(cudaFree(m_dev))
    // HANDLE_RESULT(cudaFree(found_dev))

    // test <<< 1, 1 >>>(heap_dev, buf_dev);

    /*
    constexpr size_t thread_count = 1024;
    constexpr size_t table_size = 1024 * 1024;

    HashtableType table(table_size);

    HashtableType* table_dev;
    HANDLE_RESULT(cudaMalloc(&table_dev, sizeof(HashtableType)))
    HANDLE_RESULT(cudaMemcpy(table_dev, &table, sizeof(HashtableType), cudaMemcpyHostToDevice))

    uint64_t* buf_dev;
    HANDLE_RESULT(cudaMalloc(&buf_dev, thread_count * sizeof(uint64_t)));

    bool* bool_dev;
    HANDLE_RESULT(cudaMalloc(&bool_dev, thread_count * sizeof(bool)));

    test_hash<<<1, thread_count>>>(table_dev);

    cudaDeviceSynchronize();

    test_hash_find<<<1, thread_count>>>(table_dev, buf_dev, bool_dev);

    uint64_t buf[thread_count];
    HANDLE_RESULT(cudaMemcpy(buf, buf_dev, thread_count * sizeof(uint64_t), cudaMemcpyDeviceToHost))

    bool bool_buf[thread_count];
    HANDLE_RESULT(cudaMemcpy(bool_buf, bool_dev, thread_count * sizeof(bool), cudaMemcpyDeviceToHost))

    std::cout << "elements: \n";
    for (auto element: buf) {
        std::cout << element << '\n';
    }

    std::cout << "finds: \n";
    for (auto element: bool_buf) {
        std::cout << element << '\n';
    }*/


    /*constexpr uint64_t HEAP_CAPACITY = 1024;
    Heap h(HEAP_CAPACITY);
    Heap* h_dev;
    HANDLE_RESULT(cudaMalloc(&h_dev, sizeof(Heap)))
    HANDLE_RESULT(cudaMemcpy(h_dev, &h, sizeof(Heap), cudaMemcpyHostToDevice))


    uint64_t nodesInS[Directions::Direction::NUM_DIRECTIONS];
    uint64_t* nodesInS_dev;
    HANDLE_RESULT(cudaMalloc(&nodesInS_dev, Directions::Direction::NUM_DIRECTIONS * sizeof(uint64_t)))

    unsigned valueInDest;
    unsigned* valueInDest_dev;
    HANDLE_RESULT(cudaMalloc(&valueInDest_dev, sizeof(unsigned)));*/

    return 0;
}
