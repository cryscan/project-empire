#ifndef PROJECT_EMPIRE_HASHTABLE_CUH
#define PROJECT_EMPIRE_HASHTABLE_CUH

// #include "hashtable.cu"
#include "common.h"
#include <cstdlib>
#include <device_atomic_functions.h>

template <typename Node, typename Value>
class Hashtable {
public:
	using StatePtr = Arc<State<Node, Value>>;

	explicit Hashtable(size_t capacity) :
		capacity(capacity) {
		HANDLE_RESULT(cudaMalloc(&states, capacity * sizeof(StatePtr)))
		HANDLE_RESULT(cudaMemset(states, 0, capacity * sizeof(StatePtr)))
		HANDLE_RESULT(cudaMalloc(&locks, capacity * sizeof(int)))
		HANDLE_RESULT(cudaMemset(locks, 0, capacity * sizeof(int)))
	}

	~Hashtable() {
		HANDLE_RESULT(cudaFree(states))
		HANDLE_RESULT(cudaFree(locks))
	}

	__device__ void insert(Node& key, StatePtr& state_ptr) {
		// step1: hash
		size_t slot = hash(key);
		// for (int i = 0; i < 32; i++) {
			// if (threadIdx.x % 32 == i) {
		// if old is 1, keep going; if old is 0, quit (ie execute)
		while (atomicExch(&locks[slot], 1) == 1);
		// do here
		atomicExch(&locks[slot], 0);
			//critical section
		// states[slot] = state_ptr;
				
			// }
		// }
		return;
		// step2: copy Value to hashtable & make it atomic
		// StatePtr* target = (states + slot);
		// atomicExch(target, state_ptr);
		// state_ptr = *states;
		// StatePtr test = state_ptr;
		// atomicExch(&states[slot], state_ptr);
	}
	__device__ bool find(const Node& key, StatePtr& output) {
		// step1: hash
		size_t slot = hash(key);

		// step2: check initial & check node == key
		// StatePtr* target = states + slot;
		
		if ((states+slot) == nullptr) return false;
		if (!(states[slot]->node == key)) return false;

		// step3: if found, update output
		output = states[slot];
		
		return true;
	}
	

private:
	size_t capacity;
	StatePtr* states;
	int* locks;

	// template <typename T>
	__device__ size_t hash(Node node) {
		// if (Node == uint32_t) {
			//node ^= node >> 16;
			//node *= 0x85ebca6b;
			//node ^= node >> 13;
			//node *= 0xc2b2ae35;
			//node ^= node >> 16;
		// }
		// return node & (capacity - 1);
		return 0;

	}
};


#endif // !PROJECT_EMPIRE_HASHTABLE_CUH
