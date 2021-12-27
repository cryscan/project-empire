#ifndef PROJECT_EMPIRE_HASHTABLE_CU
#define PROJECT_EMPIRE_HASHTABLE_CU

#include <cuda_runtime.h>
#include <iostream>
#include "hashtable.cuh"

template <typename Node, typename Value>
using StatePtr = Arc<State<Node, Value>>;

template <typename Node, typename Value>
__device__ void Hashtable::insert(const Node& node, const StatePtr& state_ptr) {
    // step1: hash
    size_t slot = hash(node);

    // step2: copy Value to hashtable & make it atomic
    StatePtr target = states + slot;
    atomicExch(target, state_ptr->);
}

template <typename Node, typename Value>
__device__ bool Hashtable<Node, Value>::find(const Node& key, StatePtr& value_ptr) {
    // step1: hash
    size_t slot = hash(node);

    // step2: check initial & check node == key
    StatePtr target = states + slot;
    if (target == 0 || target->node != key) return false;

    // step3: if found, update value_ptr
    value_ptr = target;
    return true;
}

template <typename Node, typename Value>
__device__ size_t Hashtable<Node, Value>::hash(Node node) {
    if (Node == uint32_t) {
        node ^= node >> 16;
        node *= 0x85ebca6b;
        node ^= node >> 13;
        node *= 0xc2b2ae35;
        node ^= node >> 16;
    }
    return k & (capacity - 1);
}



#endif // !PROJECT_EMPIRE_HASHTABLE_CU

