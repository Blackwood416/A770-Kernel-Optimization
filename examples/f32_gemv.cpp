#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace sycl;

constexpr size_t M = 4096;
constexpr size_t N = 4096;
constexpr size_t SUB = 16;
constexpr size_t SG_PER_WG = 32;
constexpr size_t VEC = 16;

void fill_f32(float* p, size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

size_t parse_size(const char* arg, const char* name, size_t fallback) {
  const std::string s(arg);
  if (s.rfind(name, 0) != 0) return fallback;
  return static_cast<size_t>(std::strtoull(s.c_str() + std::strlen(name), nullptr, 10));
}

int main(int argc, char** argv) {
  size_t warmup = 20;
  size_t samples = 20;
  size_t batch = 100;
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--json") {
      json = true;
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg.rfind("--warmup=", 0) == 0) {
      warmup = parse_size(argv[i] + 9, "", 20);
    } else if (arg == "--samples" && i + 1 < argc) {
      samples = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg.rfind("--samples=", 0) == 0) {
      samples = parse_size(argv[i] + 10, "", 20);
    } else if (arg == "--batch" && i + 1 < argc) {
      batch = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg.rfind("--batch=", 0) == 0) {
      batch = parse_size(argv[i] + 8, "", 100);
    }
  }

  try {
    device dev(gpu_selector_v);
    context ctx(dev);
    queue q(ctx, dev, {property::queue::in_order{}, property::queue::enable_profiling{}});
    std::printf("device: %s driver: %s\n",
                dev.get_info<info::device::name>().c_str(),
                dev.get_info<info::device::driver_version>().c_str());

    std::vector<float> a_host(M * N);
    std::vector<float> x_host(N);
    std::vector<float> y_host(M);
    std::vector<float> y_zero(M, 0.0f);
    fill_f32(a_host.data(), a_host.size(), 11);
    fill_f32(x_host.data(), x_host.size(), 12);

    float* A = aligned_alloc_device<float>(64, M * N, q);
    float* X = aligned_alloc_device<float>(64, N, q);
    float* Y = aligned_alloc_device<float>(64, M, q);
    q.memcpy(A, a_host.data(), M * N * sizeof(float));
    q.memcpy(X, x_host.data(), N * sizeof(float));
    q.memcpy(Y, y_zero.data(), M * sizeof(float));
    q.wait();

    auto launch = [&]() -> event {
      auto e = q.submit([&](handler& h) {
        h.parallel_for(
            nd_range<2>(range<2>(SUB, M), range<2>(SUB, SG_PER_WG)),
            [=](nd_item<2> it) {
              auto sg = it.get_sub_group();
              const size_t lane = sg.get_local_linear_id();
              const size_t sg_id = sg.get_group_linear_id();
              const size_t sg_size = sg.get_local_range()[0];
              const size_t per_lane = (N / VEC) / sg_size;
              const size_t row = it.get_group(1) * SG_PER_WG + sg_id;
              vec<float, VEC> acc(0.0f);
              for (size_t k = 0; k < per_lane; ++k) {
                const size_t col = (lane * per_lane + k) * VEC;
                acc += *reinterpret_cast<const vec<float, VEC>*>(A + row * N + col) *
                       *reinterpret_cast<const vec<float, VEC>*>(X + col);
              }
              float sum = 0.0f;
              for (size_t e = 0; e < VEC; ++e) sum += acc[e];
              const float s = reduce_over_group(sg, sum, plus<float>());
              if (lane == 0) Y[row] = s;
            });
      });
      e.wait();
      return e;
    };

    for (size_t i = 0; i < warmup; ++i) launch();

    for (size_t s = 0; s < samples; ++s) {
      double dev_sum = 0.0;
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (size_t b = 0; b < batch; ++b) {
        const auto e = launch();
        const auto start =
            e.get_profiling_info<info::event_profiling::command_start>();
        const auto end =
            e.get_profiling_info<info::event_profiling::command_end>();
        dev_sum += static_cast<double>(end - start) * 1e-3;
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      const double wall_us =
          std::chrono::duration<double, std::micro>(t1 - t0).count() /
          static_cast<double>(batch);
      const double device_us = dev_sum / static_cast<double>(batch);
      if (json) {
        std::printf(
            "{\"sample\":%zu,\"device_us\":%.4f,\"wall_us\":%.4f,"
            "\"pipeline_us\":%.4f}\n",
            s, device_us, wall_us, wall_us);
      } else {
        std::printf("sample %zu: device %.4f us wall %.4f us\n", s,
                    device_us, wall_us);
      }
    }

    launch();
    q.memcpy(y_host.data(), Y, M * sizeof(float));
    q.wait();

    double max_abs = 0.0;
    size_t errors = 0;
    for (size_t row = 0; row < M; ++row) {
      double want = 0.0;
      for (size_t k = 0; k < N; ++k) {
        want += static_cast<double>(a_host[row * N + k]) *
                static_cast<double>(x_host[k]);
      }
      const double got = y_host[row];
      const double diff = std::fabs(got - want);
      max_abs = std::max(max_abs, diff);
      const double bound = 1e-3 * std::max(1.0, std::fabs(want)) + 1e-3;
      if (!(diff <= bound)) ++errors;
    }
    std::printf("errors: %zu/%zu max_abs: %.6g\n", errors, M, max_abs);

    sycl::free(A, q);
    sycl::free(X, q);
    sycl::free(Y, q);
    return errors == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
