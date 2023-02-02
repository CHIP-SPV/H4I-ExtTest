#include <iostream>
#include <cstdlib>

template<typename T>
T* get_array(size_t sz) {
  T* memory = (T*)malloc(sz * sizeof(T));
  return memory;
}

template<typename T>
T* get_random_array(size_t sz) {
  T* memory = get_array<T>(sz);

  if (memory != nullptr) {
    srand((unsigned int) time(NULL));
    for(size_t i=0; i<sz; ++i) {
      memory[i] = rand() * 1.0;
    }
  }
  return memory;
}


