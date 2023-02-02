#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include <iostream>
#include "Util.h"

template<typename T>
int find_idx_max_element(T* ptr, int size) {
  auto id = std::distance(ptr, std::max_element(ptr, ptr+size));
  return id;
}

void hipblasamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result){
  hipblasIsamax(handle, n, x, 1, result);
}

void hipblasamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result){
  hipblasIdamax(handle, n, x, 1, result);
}

template<typename T>
bool amax_test(size_t no_of_elements) {
  hipblasStatus_t blasStatus = HIPBLAS_STATUS_SUCCESS;
  // cretae blas handle
  hipblasHandle_t handle;
  blasStatus = hipblasCreate(&handle);
  if (blasStatus != HIPBLAS_STATUS_SUCCESS){
    std::cout<<"create blas handle failed\n";
    return false;
  }

  auto noOfele = no_of_elements;
  auto size = noOfele * sizeof(T);
  T* dev_ptr = nullptr;
  hipMalloc(&dev_ptr, size);
  T *host_ptr = get_random_array<T>(noOfele);

  hipMemcpy(dev_ptr, host_ptr, size, hipMemcpyHostToDevice);

  // Find max element using GPU
  int result = 0;
  bool test_result = true;
  hipblasamax(handle, noOfele, dev_ptr, 1, &result);

  // Find max element using CPU
  auto cpu_idx = find_idx_max_element<T>(host_ptr, noOfele);

  // result
  if (cpu_idx != result) {
    //std::cout<<"result :"<<result<<"  | cpu_idx :"<<cpu_idx<<std::endl;
    test_result = false;
  }

  // cleanup
  hipFree(dev_ptr);
  free(host_ptr);
  // destroy blas handle
  blasStatus = hipblasDestroy(handle);
  if (blasStatus != HIPBLAS_STATUS_SUCCESS){
    std::cout<<"destroy blas handle failed\n";
    test_result = false;
  }
  return test_result;
}

int amax() {
  std::cout<<"------------------------------------------------------\n";
  size_t no_of_elements = 1000000;
  if (!amax_test<float>(no_of_elements)) {
    std::cout<<"AMAX test failed for data type 'float' \n";
  } else {
    std::cout<<"AMAX test passed for data type 'float' \n";
  }
  if (!amax_test<double>(no_of_elements)) {
    std::cout<<"AMAX test failed for data type 'double' \n";
  } else {
    std::cout<<"AMAX test passed for data type 'double' \n";
  }
  std::cout<<"------------------------------------------------------\n";
  return 0;
}