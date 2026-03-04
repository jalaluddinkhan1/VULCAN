/// @file pybind_vulcan.cpp
/// @brief pybind11 Python bindings for VULCAN inference engine.
///
/// Exposes the core VULCAN classes to Python:
///   import vulcan
///   engine = vulcan.Engine()
///   engine.load_model("model.vulcan", config)
///   tokens = engine.generate([1, 2, 3], gen_config)


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "vulcan/engine.h"
#include "vulcan/tensor.h"
#include "vulcan/model.h"
#include "vulcan/sampler.h"
#include "cuda/utils.h"

namespace py = pybind11;

PYBIND11_MODULE(vulcan, m) {
    m.doc() = "VULCAN — Custom C++/CUDA LLM Inference Engine";

    // ── Device Enum ─────────────────────────────────────────────────
    py::enum_<vulcan::Device>(m, "Device")
        .value("CPU", vulcan::Device::CPU)
        .value("CUDA", vulcan::Device::CUDA)
        .export_values();

    // ── ModelConfig ─────────────────────────────────────────────────
    py::class_<vulcan::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("hidden_dim", &vulcan::ModelConfig::hidden_dim)
        .def_readwrite("num_layers", &vulcan::ModelConfig::num_layers)
        .def_readwrite("num_heads", &vulcan::ModelConfig::num_heads)
        .def_readwrite("num_kv_heads", &vulcan::ModelConfig::num_kv_heads)
        .def_readwrite("vocab_size", &vulcan::ModelConfig::vocab_size)
        .def_readwrite("intermediate", &vulcan::ModelConfig::intermediate)
        .def_readwrite("norm_eps", &vulcan::ModelConfig::norm_eps)
        .def_readwrite("rope_theta", &vulcan::ModelConfig::rope_theta)
        .def_readwrite("max_seq_len", &vulcan::ModelConfig::max_seq_len)
        .def("__repr__", [](const vulcan::ModelConfig& c) {
            return "<ModelConfig hidden=" + std::to_string(c.hidden_dim) +
                   " layers=" + std::to_string(c.num_layers) +
                   " heads=" + std::to_string(c.num_heads) +
                   " vocab=" + std::to_string(c.vocab_size) + ">";
        });

    // ── GenerationConfig ────────────────────────────────────────────
    py::class_<vulcan::GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_tokens", &vulcan::GenerationConfig::max_tokens)
        .def_readwrite("max_seq_len", &vulcan::GenerationConfig::max_seq_len)
        .def_readwrite("temperature", &vulcan::GenerationConfig::temperature)
        .def_readwrite("top_p", &vulcan::GenerationConfig::top_p)
        .def_readwrite("top_k", &vulcan::GenerationConfig::top_k)
        .def_readwrite("greedy", &vulcan::GenerationConfig::greedy)
        .def_readwrite("eos_token_id", &vulcan::GenerationConfig::eos_token_id)
        .def("__repr__", [](const vulcan::GenerationConfig& c) {
            return "<GenerationConfig max_tokens=" + std::to_string(c.max_tokens) +
                   " temp=" + std::to_string(c.temperature) +
                   " top_p=" + std::to_string(c.top_p) + ">";
        });

    // ── Engine ──────────────────────────────────────────────────────
    py::class_<vulcan::Engine>(m, "Engine")
        .def(py::init<>())
        .def("load_model", &vulcan::Engine::load_model,
             py::arg("path"), py::arg("config"),
             "Load model weights from a VULCAN binary file.")
        .def("forward", &vulcan::Engine::forward,
             py::arg("input_ids"),
             "Run full forward pass. Returns logits for last position.")
        .def("forward_one", &vulcan::Engine::forward_one,
             py::arg("token_id"), py::arg("pos"),
             "Run single-token decode step (uses KV cache).")
        .def("generate", &vulcan::Engine::generate,
             py::arg("prompt_ids"),
             py::arg("gen_config") = vulcan::GenerationConfig{},
             "Generate tokens autoregressively with KV cache.")
        .def("reset_cache", &vulcan::Engine::reset_cache,
             "Reset KV cache for new generation.")
        .def("is_ready", &vulcan::Engine::is_ready,
             "Check if model is loaded.")
        .def("cache_memory_usage", &vulcan::Engine::cache_memory_usage,
             "Get KV cache memory usage in bytes.")
        .def("__repr__", [](const vulcan::Engine& e) {
            return std::string("<vulcan.Engine ready=") +
                   (e.is_ready() ? "True" : "False") + ">";
        });

    // ── Tensor ──────────────────────────────────────────────────────
    py::class_<vulcan::Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, vulcan::Device>(),
             py::arg("shape"), py::arg("device") = vulcan::Device::CPU)
        .def("shape", &vulcan::Tensor::shape,
             py::return_value_policy::reference_internal)
        .def("ndim", &vulcan::Tensor::ndim)
        .def("numel", &vulcan::Tensor::numel)
        .def("nbytes", &vulcan::Tensor::nbytes)
        .def("device", &vulcan::Tensor::device)
        .def("is_valid", &vulcan::Tensor::is_valid)
        .def("to_numpy", [](const vulcan::Tensor& t) {
            // Copy tensor data to a numpy array
            std::vector<float> data(t.numel());
            if (t.device() == vulcan::Device::CPU) {
                std::memcpy(data.data(), t.data(), t.nbytes());
            } else {
                t.to_host(data.data());
            }
            // Convert shape
            std::vector<py::ssize_t> np_shape;
            for (int s : t.shape()) np_shape.push_back(s);
            return py::array_t<float>(np_shape, data.data());
        }, "Convert tensor to numpy array (copies to CPU).")
        .def("__repr__", [](const vulcan::Tensor& t) {
            std::string s = "<vulcan.Tensor shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "] device=";
            s += (t.device() == vulcan::Device::CUDA ? "CUDA" : "CPU");
            s += ">";
            return s;
        });

    // ── Utility Functions ───────────────────────────────────────────
    m.def("print_device_info", &vulcan::cuda::print_device_info,
          "Print GPU device information.");

    m.def("get_memory_info", []() {
        size_t free_bytes, total_bytes;
        vulcan::cuda::get_memory_info(free_bytes, total_bytes);
        return py::make_tuple(free_bytes, total_bytes);
    }, "Get (free_bytes, total_bytes) of GPU VRAM.");

    // ── Version ─────────────────────────────────────────────────────
    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "VULCAN Team";
}
