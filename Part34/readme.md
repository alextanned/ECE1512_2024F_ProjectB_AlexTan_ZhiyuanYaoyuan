## part 34
# VILA: Pre-training Strategies for Vision-Language Models


---

## Table of Contents

- [Introduction](#introduction)
- [Key Components](#key-components)
- [Techniques and Contributions](#techniques-and-contributions)
- [Optimization](#optimization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Introduction

VILA (Visual-Language Alignment) is a vision-language model (VLM) designed to enhance the integration of visual and linguistic data. This project explores VILA's architectural design, training strategies, and optimization methods, specifically focusing on:

- Vision Transformer (ViT) for image encoding
- LLaMA-2 large language model for text generation
- Linear projector for modality alignment

For more information, check the [full report](https://github.com/alextanned/ECE1512_2024F_ProjectB_AlexTan_ZhiyuanYaoyuan).

---

## Key Components

1. **Visual Encoder**:  
   Vision Transformer (ViT) divides images into patches and encodes them into visual tokens.

2. **Language Model**:  
   LLaMA-2 processes text input and generates contextual responses.

3. **Projector**:  
   A linear layer aligns visual tokens with text tokens.

---

## Techniques and Contributions

- **Interleaved Image-Text Pre-training**:  
  Improves modality alignment using datasets like MMC4.

- **Fine-Tuning the LLM**:  
  Enhances in-context learning by unfreezing the language model during training.

- **Joint Instruction Fine-Tuning**:  
  Supports both visual-language and text-only tasks to maintain robust text generation.

---

## Optimization

To address efficiency bottlenecks, the project proposes **Channel Pruning** to reduce computational overhead while maintaining accuracy. This technique focuses on:

- Reducing Floating Point Operations (FLOPs)
- Minimizing memory usage
- Maintaining model performance post-pruning

---

## Results

### Key Benchmarks

| **Model**    | **VQAv2** | **GQA** | **TextVQA** | **MMBench** | **MM-Vet** |
|--------------|------------|---------|-------------|-------------|------------|
| **VILA-7B** | 79.9%      | 62.3%   | 57.8%       | 68.9%       | 34.9%      |
| **VILA-13B**| 80.8%      | 63.3%   | 60.6%       | 70.3%       | 38.8%      |

### Optimization Results

- **FLOPs Reduction**: 3.98M → 1.09M  
- **Accuracy**: Maintained at 98.29% on MNIST  
- **Memory Usage**: Slight decrease post-pruning

---

## Installation

   ```bash
   git clone https://github.com/alextanned/ECE1512_2024F_ProjectB_AlexTan_ZhiyuanYaoyuan.git
   cd ECE1512_2024F_ProjectB_AlexTan_ZhiyuanYaoyuan
