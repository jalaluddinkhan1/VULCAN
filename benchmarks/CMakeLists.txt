# ─── Benchmarks ──────────────────────────────────────────────────────────────

add_executable(bench_inference bench_inference.cpp)
target_link_libraries(bench_inference PRIVATE vulcan)

add_executable(bench_memory bench_memory.cpp)
target_link_libraries(bench_memory PRIVATE vulcan)
