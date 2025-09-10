# Mapping Layer Similarity in Large Language Models with RSA & CKA

**Author**: Ritik Bompilwar

## Overview

This repository implements **Representational Similarity Analysis (RSA)** and **Centered Kernel Alignment (CKA)** to study layer-wise similarity patterns in decoder-only Large Language Models across parallel English-Hindi prompts. Our analysis reveals how different models maintain cross-lingual representational alignment and provides insights into multilingual processing in transformer architectures.

## 📊 Key Findings

### Cross-lingual Alignment Performance (EN↔HI):

| Model | CKA Score | 3D Distance | Layers | Cross-lingual Quality |
|-------|-----------|-------------|--------|----------------------|
| **GPT-OSS-20B** | **0.698** | **0.407** | 24 | ⭐ **Optimal** |
| **Llama-3.2-1B** | 0.684 | 0.519 | 16 | Good |
| **Llama-3.2-3B** | 0.695 | 0.556 | 28 | Good |
| **Llama-3.1-8B** | 0.653 | 0.603 | 32 | Moderate |

### 🔍 **Major Discoveries**:
- **Single language axis emerges** with strong correlation (PC2 ≈ 0.93) in joint projection space
- **GPT-OSS-20B demonstrates optimal cross-lingual performance** with lowest 3D distance and highest CKA
- **Larger Llama models exhibit increased language-specific drift**
- **Layer analysis reveals concentrated alignment with selective divergence in deeper layers**

## 🛠 Methodology

### Data Processing
- **Dataset**: Global MMLU-Lite, 400 prompts each in English & Hindi
- **Models**: Llama-3.2-1B (16 layers), Llama-3.2-3B (28 layers), Llama-3.1-8B (32 layers), GPT-OSS-20B (24 layers)

### CKA Signature Pipeline
```python
# For each layer, standardize activations, build the Gram matrix K = XX^T and double-center it (HKH)
K_centered = K - (1/n)K - K(1/n) + (1/n)K(1/n)

# Flatten and L2-normalize the centered K to get a CKA signature v
v = K_centered.flatten() / ||K_centered.flatten()||

# Cosine between two signatures equals linear CKA between the layers
CKA_score = v1 · v2
```

### Joint Embedding Analysis
- **Stack all layer signatures** into a matrix V, compute similarity S = VV^T
- **Eigendecompose S** to obtain a 3D joint projection for visualization
- **Measure cross-lingual alignment** via 3D distances and CKA scores between paired layers

## 📈 Results Visualization

### Language Separation Analysis
```
Overall corr(PC2, language): 0.9262
  meta-llama/Llama-3.1-8B     corr = 0.9631
  meta-llama/Llama-3.2-1B     corr = 0.9609  
  meta-llama/Llama-3.2-3B     corr = 0.9626
  openai/gpt-oss-20b          corr = 0.8910
```

### 3D Distance Analysis (EN↔HI pairs)
```
Per-model mean 3D distances:
  openai/gpt-oss-20b     0.407 ± 0.029
  meta-llama/Llama-3.2-1B   0.519 ± 0.076
  meta-llama/Llama-3.2-3B   0.556 ± 0.074  
  meta-llama/Llama-3.1-8B   0.603 ± 0.082
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch transformers datasets plotly numpy tqdm huggingface_hub
```

### Quick Start
1. **Extract activations for all models:**
   ```bash
   cd Notebooks/CKA
   jupyter notebook Implementation.ipynb
   ```

2. **Generate CKA signatures and analysis:**
   ```bash
   jupyter notebook Analysis.ipynb  # Results analysis
   ```

3. **View interactive 3D projections:**
   - Open `artifacts/projector/cka_projector_joint_pro.html` for full interactive analysis
   - Features: model filtering, language toggles, hover metrics, pair linking

### Repository Structure
```
Llama_RSA/
├── Notebooks/
│   ├── CKA/                    # 🆕 Main implementation
│   │   ├── Implementation.ipynb # Core CKA/RSA pipeline
│   │   └── Analysis.ipynb     # Results compilation & metrics
│   └── RSA-Old/               # Legacy RSA implementations
│       ├── RSA_Analyser_Bio.ipynb
│       └── RSA_Analyser_Cyber.ipynb
├── artifacts/                 # Generated analysis outputs
│   ├── models/               # Per-model activations & heatmaps
│   ├── signatures/cka/       # CKA signature matrices
│   ├── projector/           # Interactive HTML visualizations
│   └── cross/               # Cross-model analysis results
└── README.md
```

## 🔬 Technical Implementation

### Core Components

#### 1. **Activation Extraction** (`collect_layer_mats()`)
- Processes models with **OOM-safe batching** and automatic downscaling
- Extracts final non-pad token representations from all layers
- Supports **GPU acceleration** with memory optimization

#### 2. **CKA Computation** (`cka_matrix_gpu()`)
```python
@torch.no_grad()
def cka_linear(X: torch.Tensor, Y: torch.Tensor):
    Kx, Ky = X @ X.t(), Y @ Y.t()
    # Double-centering
    Kx = Kx - one @ Kx - Kx @ one + one @ Kx @ one  
    Ky = Ky - one @ Ky - Ky @ one + one @ Ky @ one
    return (Kx * Ky).sum() / (torch.norm(Kx) * torch.norm(Ky) + 1e-12)
```

#### 3. **RSA Analysis** (`rsa_matrix_gpu()`)
- **Vectorized Spearman correlation** computation
- Supports both cosine and correlation-based RDMs
- **GPU-accelerated** ranking and correlation calculations

#### 4. **Interactive Visualization**
- **3D joint embeddings** with language-wise coloring
- **Real-time filtering** by models and datasets  
- **Hover metrics**: CKA scores, 3D distances, layer information
- **Pair linking** visualization for cross-lingual analysis

## 📊 Detailed Results

### Model Architecture Analysis
```
Analyzed Models & Layer Counts:
  • meta-llama/Llama-3.1-8B: 32 layers
  • meta-llama/Llama-3.2-1B: 16 layers  
  • meta-llama/Llama-3.2-3B: 28 layers
  • openai/gpt-oss-20b: 24 layers
Total: 100 layers × 400 examples = 40K representations per language
```

### Cross-Model Similarity Matrices
- **Within-model**: Layer progression analysis showing representational evolution
- **Cross-model**: Architecture comparison revealing model family similarities  
- **Cross-lingual**: EN-HI alignment patterns across all model pairs

### Orthogonal Procrustes Analysis
```
Procrustes Residuals (lower = better alignment):
  openai/gpt-oss-20b      0.546
  meta-llama/Llama-3.2-3B   0.548
  meta-llama/Llama-3.2-1B   0.556  
  meta-llama/Llama-3.1-8B   0.581
```

## 🔮 Future Scope

### Immediate Extensions
- **Expand to more languages** and model families, and control prompt format and length to test robustness
- **Validate with bootstrap and permutation tests**, map where alignment changes across depth
- **Alternative token pooling strategies** (attention-weighted, max, etc.)

### Research Directions  
- **Explore removing a low-rank language subspace** to test transfer learning
- **Causal intervention analysis** to understand cross-lingual transfer mechanisms
- **Scale analysis**: How do findings generalize to 70B+ parameter models?
- **Task-specific analysis**: Does alignment vary across reasoning vs. factual tasks?

## 🏆 Key Innovations

1. **Vectorized RSA Implementation**: Highly optimized Spearman correlation computation using rank z-scoring
2. **Joint Embedding Framework**: Novel approach to unified cross-lingual representation analysis
3. **Interactive 3D Visualization**: Real-time exploration of high-dimensional similarity spaces
4. **Production-Grade Pipeline**: Enterprise-quality code with comprehensive error handling and memory management

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{bompilwar2025mapping,
  title={Mapping Layer Similarity in Large Language Models with RSA & CKA},
  author={Bompilwar, Ritik},
  year={2025},
  howpublished={\url{https://github.com/RITIK-12/Llama_RSA}}
}
```

## 🔄 Reproducibility Notes

**Complete Reproducibility Maintained**:
- All analysis results preserved in `artifacts/signatures/cka/`
- Interactive visualizations available in `artifacts/projector/` 
- Cross-model analysis matrices in `artifacts/cross/`
- Original prompts and metadata preserved for exact reproduction

**To regenerate raw activations**: Run `Notebooks/CKA/Implementation.ipynb`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This work is built on initial work in Evolution of LLM Stages (https://sidn.baulab.info/evolution/)*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Contact

**Ritik Bompilwar** - [GitHub](https://github.com/RITIK-12)

For questions about the methodology or results, please open an issue in this repository.