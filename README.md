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

- Benchmarking Intel’s software-hardware stack for multimodal AI pipelines
- Applying IPEX optimizations to models and optimizers
- Supporting XPU architecture for dynamic device assignment
- Enhancing throughput and reducing memory consumption
- Supporting mixed-precision computation for model efficiency

### Device Management

```python
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f"Your device is set to: {device}")
```

### Model Precision Handling
```python
if cfg.TRAINER.PROMPTSRC.PREC == "fp16":
    clip_model.half()
    self.dtype = torch.float16
else:
    clip_model.float()
    self.dtype = torch.float32
```

### Optimizations with IPEX
```python
import intel_extension_for_pytorch as ipex
self.model, self.optim = ipex.optimize(self.model, optimizer=self.optim, auto_kernel_selection=False)
```

### Device and Data Type Transfer After Optimization
```python
self.model.to(device, dtype=self.dtype)
```

### Teacher Model Adaptations for PromptKD
```python
clip_model_teacher = load_clip_to_cpu_teacher(cfg)
self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)
self.model_teacher.to(device)
self.model_teacher.eval()
```
