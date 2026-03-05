/// @file vulcan_bindings.cpp
/// @brief pybind11 Python bindings for the VULCAN inference engine.
///
/// Exposes:
///   vulcan.ModelConfig      — model architecture parameters
///   vulcan.GenerationConfig — generation / sampling parameters
///   vulcan.Engine           — main inference engine
///
/// Usage (Python):
///   import vulcan
///   engine = vulcan.Engine()
///   engine.load_model("model_q4.vulcan", vulcan.ModelConfig())
///   tokens = engine.generate([1, 15043, 29892], vulcan.GenerationConfig())

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vulcan/engine.h"
#include "vulcan/model.h"
#include "vulcan/sampler.h"

namespace py = pybind11;

PYBIND11_MODULE(vulcan, m) {
    m.doc() = "VULCAN — Custom C++/CUDA LLM Inference Engine";

    // ── ModelConfig ──────────────────────────────────────────────────────────
    py::class_<vulcan::ModelConfig>(m, "ModelConfig",
        "Architecture configuration for the transformer model.\n"
        "Defaults match Llama-2-7B.")
        .def(py::init<>())
        .def_readwrite("vocab_size",   &vulcan::ModelConfig::vocab_size,
                       "Vocabulary size (default: 32000)")
        .def_readwrite("hidden_dim",   &vulcan::ModelConfig::hidden_dim,
                       "Hidden dimension d_model (default: 4096)")
        .def_readwrite("num_heads",    &vulcan::ModelConfig::num_heads,
                       "Number of attention heads (default: 32)")
        .def_readwrite("num_kv_heads", &vulcan::ModelConfig::num_kv_heads,
                       "Number of KV heads for GQA (default: 32)")
        .def_readwrite("num_layers",   &vulcan::ModelConfig::num_layers,
                       "Number of transformer layers (default: 32)")
        .def_readwrite("max_seq_len",  &vulcan::ModelConfig::max_seq_len,
                       "Maximum sequence length (default: 4096)")
        .def_readwrite("intermediate", &vulcan::ModelConfig::intermediate,
                       "MLP intermediate dimension (default: 11008)")
        .def_readwrite("norm_eps",     &vulcan::ModelConfig::norm_eps,
                       "RMSNorm epsilon (default: 1e-5)")
        .def_readwrite("rope_theta",   &vulcan::ModelConfig::rope_theta,
                       "RoPE base frequency — Llama-2: 10000, Llama-3: 500000")
        .def("__repr__", [](const vulcan::ModelConfig& c) {
            return "<ModelConfig vocab=" + std::to_string(c.vocab_size) +
                   " hidden=" + std::to_string(c.hidden_dim) +
                   " layers=" + std::to_string(c.num_layers) +
                   " heads=" + std::to_string(c.num_heads) + ">";
        });

    // ── GenerationConfig ─────────────────────────────────────────────────────
    py::class_<vulcan::GenerationConfig>(m, "GenerationConfig",
        "Parameters controlling text generation.")
        .def(py::init<>())
        .def_readwrite("max_tokens",   &vulcan::GenerationConfig::max_tokens,
                       "Maximum tokens to generate (default: 256)")
        .def_readwrite("max_seq_len",  &vulcan::GenerationConfig::max_seq_len,
                       "Maximum total sequence length (default: 2048)")
        .def_readwrite("temperature",  &vulcan::GenerationConfig::temperature,
                       "Sampling temperature — 1.0 = neutral (default: 1.0)")
        .def_readwrite("top_p",        &vulcan::GenerationConfig::top_p,
                       "Nucleus sampling threshold (default: 0.9)")
        .def_readwrite("top_k",        &vulcan::GenerationConfig::top_k,
                       "Top-K sampling count (default: 40)")
        .def_readwrite("greedy",       &vulcan::GenerationConfig::greedy,
                       "If True, always pick argmax (default: False)")
        .def_readwrite("eos_token_id", &vulcan::GenerationConfig::eos_token_id,
                       "End-of-sequence token ID (default: 2)")
        .def("__repr__", [](const vulcan::GenerationConfig& g) {
            return "<GenerationConfig max_tokens=" + std::to_string(g.max_tokens) +
                   " temp=" + std::to_string(g.temperature) +
                   " top_p=" + std::to_string(g.top_p) +
                   " top_k=" + std::to_string(g.top_k) +
                   " greedy=" + (g.greedy ? "True" : "False") + ">";
        });

    // ── Engine ───────────────────────────────────────────────────────────────
    py::class_<vulcan::Engine>(m, "Engine",
        "Main VULCAN inference engine with KV-cached generation.\n\n"
        "Example::\n\n"
        "    engine = vulcan.Engine()\n"
        "    cfg = vulcan.ModelConfig()\n"
        "    engine.load_model('model_q4.vulcan', cfg)\n"
        "    tokens = engine.generate([1, 15043], vulcan.GenerationConfig())\n")
        .def(py::init<>())

        .def("load_model",
             &vulcan::Engine::load_model,
             py::arg("path"), py::arg("config"),
             "Load model weights from a VULCAN binary file.\n\n"
             "Args:\n"
             "    path:   Path to the .vulcan weight file\n"
             "    config: ModelConfig matching the model architecture\n\n"
             "Returns:\n"
             "    True on success, False on failure")

        .def("generate",
             &vulcan::Engine::generate,
             py::arg("prompt_ids"),
             py::arg("config") = vulcan::GenerationConfig{},
             py::call_guard<py::gil_scoped_release>(),
             "Generate tokens using KV-cached incremental decode.\n\n"
             "Args:\n"
             "    prompt_ids: List of integer token IDs (the prompt)\n"
             "    config:     GenerationConfig controlling sampling\n\n"
             "Returns:\n"
             "    List of generated token IDs (including the prompt)")

        .def("forward",
             &vulcan::Engine::forward,
             py::arg("input_ids"),
             py::call_guard<py::gil_scoped_release>(),
             "Prefill forward pass — returns logits for the last token.\n\n"
             "Args:\n"
             "    input_ids: Full prompt token ID list\n\n"
             "Returns:\n"
             "    List of floats [vocab_size] — raw logits")

        .def("forward_one",
             &vulcan::Engine::forward_one,
             py::arg("token_id"), py::arg("pos"),
             py::call_guard<py::gil_scoped_release>(),
             "Single-token decode step. Requires prior forward() call.\n\n"
             "Args:\n"
             "    token_id: The new token to process\n"
             "    pos:      Its position in the full sequence\n\n"
             "Returns:\n"
             "    List of floats [vocab_size] — raw logits")

        .def("reset_cache",
             &vulcan::Engine::reset_cache,
             "Reset the KV cache. Call before starting a new conversation.")

        .def("is_ready",
             &vulcan::Engine::is_ready,
             "Returns True if the model is loaded and ready to generate.")

        .def("cache_memory_usage",
             &vulcan::Engine::cache_memory_usage,
             "Returns the KV cache GPU memory usage in bytes.");
}
