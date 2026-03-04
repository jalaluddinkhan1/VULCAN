# ─── Tests ───────────────────────────────────────────────────────────────────
# GoogleTest unit tests for VULCAN components

# Test: Tensor operations
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE vulcan GTest::gtest_main)
add_test(NAME test_tensor COMMAND test_tensor)

# Test: Vector Add kernel (baseline verification)
add_executable(test_vector_add test_vector_add.cu)
target_link_libraries(test_vector_add PRIVATE vulcan GTest::gtest_main)
set_target_properties(test_vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_vector_add COMMAND test_vector_add)

# Test: Core CUDA kernels
add_executable(test_kernels test_kernels.cu)
target_link_libraries(test_kernels PRIVATE vulcan GTest::gtest_main)
set_target_properties(test_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_kernels COMMAND test_kernels)

# Test: Attention + Element-wise kernels
add_executable(test_attention test_attention.cu)
target_link_libraries(test_attention PRIVATE vulcan GTest::gtest_main)
set_target_properties(test_attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_attention COMMAND test_attention)

# Test: Memory manager + KV Cache
add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE vulcan GTest::gtest_main)
add_test(NAME test_memory COMMAND test_memory)

# Test: Quantization + Fused kernels
add_executable(test_quantization test_quantization.cu)
target_link_libraries(test_quantization PRIVATE vulcan GTest::gtest_main)
set_target_properties(test_quantization PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_quantization COMMAND test_quantization)
