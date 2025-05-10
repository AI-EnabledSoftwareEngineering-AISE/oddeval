# FODSE: Framing ODDs as Domain Specifications for Evaluation

**Domain**: Visual Dataset Evaluation for AI-enabled Systems  
**Focus**: Automated Driving Systems (ADS) and Safety-Critical Applications  
**Technologies**: CLIP, BLIP2, IPEX, RAG, Intel® Tiber™ AI Cloud

---

## 🚀 Summary

FODSE is a semi-automated framework for refining **Operational Design Domains (ODDs)** into structured specifications and evaluating the coverage of visual datasets against these specifications. Developed under the NIU–Intel® oneAPI initiative, FODSE bridges the gap between **domain knowledge** and **multimodal AI models**, enabling more interpretable and safety-aware visual understanding in AI-enabled systems (AIS).

**Key highlights:**

- Up to **30% improvement** in inference speed and memory efficiency using Intel® technologies.
- Integration of **prompt learning** and **Retrieval-Augmented Generation (RAG)**.
- Support for multimodal models such as **CLIP** and **BLIP2**.
- Structured evaluation of dataset coverage based on domain-relevant attributes.

---

## 🧩 Problem Statement

Ensuring the safety of Automated Driving Systems (ADS) depends on verifying whether training and testing datasets fully represent their intended ODDs—covering variables such as road types, weather, time of day, and presence of vulnerable road users. Public datasets frequently lack this coverage, leading to blind spots and unreliable model performance.

**FODSE** addresses this issue by:

- Refining ODDs into structured, queryable specifications.
- Evaluating datasets for underrepresented conditions and edge cases.
- Highlighting safety-critical dataset gaps.

---

## ⚙️ Approach and Technical Contributions

### Framework Overview

FODSE is a **requirements-driven framework** that integrates multimodal AI with domain specifications. It includes:

- **RAG Integration**: Extracts ODD concepts from documentation.
- **Prompt Learning**: Guides CLIP/BLIP2 models to detect ODD-related features.
- **Coverage Engine**: Automatically assesses dataset coverage across structured attributes.

---

## 🔧 Intel-Accelerated Implementation

This project was built and tested on **Intel® Tiber™ AI Cloud** using **Intel® Max Series GPUs** and **Intel® Extension for PyTorch (IPEX)**.


Key goals of this project included:

- Benchmarking Intel's software-hardware stack for multimodal AI pipelines
- Applying IPEX optimizations to models and optimizers
- Supporting XPU architecture for dynamic device assignment
- Enhancing throughput and reducing memory consumption
- Supporting mixed-precision computation for model efficiency

---

## Device Management

```python
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f"Your device is set to: {device}")
```

---

## Model Precision Handling

```python
if cfg.TRAINER.PROMPTSRC.PREC == "fp16":
    clip_model.half()
    self.dtype = torch.float16
else:
    clip_model.float()
    self.dtype = torch.float32
```

---

## Optimizations with IPEX

```python
import intel_extension_for_pytorch as ipex

# Optimize student model
self.model, self.optim = ipex.optimize(
    self.model,
    optimizer=self.optim,
    auto_kernel_selection=False
)
```

---

## Device and Data Type Transfer After Optimization

```python
# Apply correct device and precision after optimization
self.model.to(device, dtype=self.dtype)
```

---

## Teacher Model Adaptations for PromptKD

```python
# Load and prepare teacher model
clip_model_teacher = load_clip_to_cpu_teacher(cfg)
self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)
self.model_teacher.to(device)
self.model_teacher.eval()

# Optimize teacher model with IPEX
self.model_teacher, _ = ipex.optimize(
    self.model_teacher,
    optimizer=self.optim,
    auto_kernel_selection=False
)
```

---

## Performance Summary

| Metric                                 | Result                         |
|----------------------------------------|--------------------------------|
| Inference Throughput                   | +20–30%                        |
| Memory Efficiency                      | Improved; larger batch sizes   |
| Training Speed (PromptSRC, PromptKD)   | Accelerated                    |
| Integration                            | Seamless XPU & precision aware |

---

## Repository Structure

```plaintext
📁 PromptsLearniong/
│   ├── 📁 configs/           # Configuration files
│   ├── 📁 datasets/         # Dataset handling and processing
│   ├── 📁 docs/            # Documentation
│   ├── 📁 images/          # Image assets
│   ├── 📁 interpret_prompts/ # Prompt interpretation utilities
│   ├── 📁 lpclip/          # CLIP model implementations
│   ├── 📁 scripts/         # Utility scripts
│   ├── 📁 teacher_model/   # Teacher model implementations
│   ├── 📁 trainers/        # Training utilities
│   ├── 📄 train.py         # Main training script
│   └── 📄 requirements.txt  # Project dependencies

📄 roadway_user_prompts.md   # Road user-related prompts
📄 lane_markings_prompts.md  # Lane marking-related prompts
📄 RAG-llama.ipynb          # RAG implementation with LLaMA
📄 survay.ipynb             # Survey analysis notebook
📄 test_all_blip_user.py    # BLIP model testing script
📄 result_blip.ipynb        # BLIP results analysis
```

---

### Performance Optimization
- Use `torch.xpu` for device management
- Enable mixed precision training with IPEX
- Utilize Intel® oneAPI DPC++/C++ Compiler for custom operations
- Enable Intel® MKL optimizations for numerical computations

---

## Use Case

This framework is optimized for **high-performance multimodal dataset evaluation**, especially in **safety-critical applications** like **autonomous driving**. It demonstrates how Intel's ecosystem can boost throughput, reduce memory overhead, and streamline distillation-based learning strategies.

---

## Acknowledgments

This work was supported by **Intel OneAPI Center of Excellence**.

Special thanks to **David Demarle** and the **Intel Rendering Team (OSPRay/OSPRay Studio)** for infrastructure and mentorship that enabled successful integration of IPEX on **Intel® Tiber™ AI Cloud**.
