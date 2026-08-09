#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sycl;

constexpr size_t M = 4096;
constexpr size_t K = 4096;
constexpr size_t N = 1;

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

    dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
    dnnl::stream stream = dnnl::sycl_interop::make_stream(eng, q);

    auto src_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::ab);
    auto wei_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::ba);
    auto dst_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::ab);
    dnnl::matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
    dnnl::matmul prim(pd);
    std::printf("implementation: %s\n", pd.impl_info_str());

    std::vector<float> a_host(M * K);
    std::vector<float> x_host(K);
    std::vector<float> y_host(M);
    std::vector<float> y_zero(M, 0.0f);
    fill_f32(a_host.data(), a_host.size(), 21);
    fill_f32(x_host.data(), x_host.size(), 22);

    float* A = aligned_alloc_device<float>(64, M * K, q);
    float* X = aligned_alloc_device<float>(64, K, q);
    float* Y = aligned_alloc_device<float>(64, M, q);
    q.memcpy(A, a_host.data(), M * K * sizeof(float));
    q.memcpy(X, x_host.data(), K * sizeof(float));
    q.memcpy(Y, y_zero.data(), M * sizeof(float));
    q.wait();

    auto src_mem = dnnl::sycl_interop::make_memory(
        src_md, eng, dnnl::sycl_interop::memory_kind::usm, A);
    auto wei_mem = dnnl::sycl_interop::make_memory(
        wei_md, eng, dnnl::sycl_interop::memory_kind::usm, X);
    auto dst_mem = dnnl::sycl_interop::make_memory(
        dst_md, eng, dnnl::sycl_interop::memory_kind::usm, Y);
    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, wei_mem},
        {DNNL_ARG_DST, dst_mem},
    };

    auto launch = [&]() -> event {
      auto e = dnnl::sycl_interop::execute(prim, stream, args);
      e.wait();
      return e;
    };

    for (size_t i = 0; i < warmup; ++i) launch();

    for (size_t s = 0; s < samples; ++s) {
      double dev_sum = 0.0;
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (size_t b = 0; b < batch; ++b) {
        const auto e = launch();
        double dev_us = 0.0;
        try {
          const auto start =
              e.get_profiling_info<info::event_profiling::command_start>();
          const auto end =
              e.get_profiling_info<info::event_profiling::command_end>();
          dev_us = static_cast<double>(end - start) * 1e-3;
        } catch (const std::exception&) {
          dev_us = 0.0;
        }
        dev_sum += dev_us;
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
      for (size_t k = 0; k < K; ++k) {
        want += static_cast<double>(a_host[row * K + k]) *
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
