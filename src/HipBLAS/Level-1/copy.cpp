#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include <iostream>
#include "Util.h"

void hipblascopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy){
    hipblasScopy(handle, n, x, 1, y, 1);
}

void hipblascopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy){
    hipblasDcopy(handle, n, x, 1, y, 1);
}

template<typename T>
bool copy_test(size_t no_of_elements) {
  hipblasStatus_t blasStatus = HIPBLAS_STATUS_SUCCESS;
  // cretae blas handle
  hipblasHandle_t handle;
  blasStatus = hipblasCreate(&handle);
  if (blasStatus != HIPBLAS_STATUS_SUCCESS){
    std::cout<<"create blas handle failed\n";
    return false;
  }

  // allocate host and device memory
  auto noOfele = no_of_elements;
  auto size = noOfele * sizeof(T);
  T* dev_ptr_x = nullptr; hipMalloc(&dev_ptr_x, size);
  T* dev_ptr_y = nullptr; hipMalloc(&dev_ptr_y, size);
  T *host_ptr_1 = get_random_array<T>(noOfele);

  // write values into device memory
  hipMemcpy(dev_ptr_x, host_ptr_1, size, hipMemcpyHostToDevice);

  hipblascopy(handle, noOfele, dev_ptr_x, 1, dev_ptr_y, 1);

  // result
  bool test_result = true;
  T *host_ptr_2 = get_array<T>(noOfele);
  hipMemcpy(host_ptr_2, dev_ptr_y, size, hipMemcpyDefault);

  for (int i=0; i< noOfele; ++i) {
    if (host_ptr_2[i] != host_ptr_1[i]) {
      test_result = false;
      break;
    }
  }

  // cleanup
  hipFree(dev_ptr_x);
  hipFree(dev_ptr_y);
  free(host_ptr_1);
  free(host_ptr_2);
  // destroy blas handle
  blasStatus = hipblasDestroy(handle);
  if (blasStatus != HIPBLAS_STATUS_SUCCESS){
    std::cout<<"destroy blas handle failed\n";
    test_result = false;
  }

  return test_result;
}

int copy() {
  std::cout<<"------------------------------------------------------\n";
  size_t no_of_elements = 1000000;
  if (!copy_test<float>(no_of_elements)) {
    std::cout<<"COPY test failed for data type 'float' \n";
  } else {
    std::cout<<"COPY test passed for data type 'float' \n";
  }
  if (!copy_test<double>(no_of_elements)) {
    std::cout<<"COPY test failed for data type 'double' \n";
  } else {
    std::cout<<"COPY test passed for data type 'double' \n";
  }
  std::cout<<"------------------------------------------------------\n";
  return 0;
}