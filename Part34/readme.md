## part 34
# VILA: Pre-training Strategies for Vision-Language Models


---

## Table of Contents

- [Introduction](#introduction)
- [Key Components](#key-components)
- [Techniques and Contributions](#techniques-and-contributions)
- [Optimization](#optimization)
- [Results](#results)
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

### Channel Pruning Optimization Results

- **FLOPs Reduction**: 3.98M → 1.09M  
- **Accuracy**: Maintained at 98.29% on MNIST  
- **Memory Usage**: Slight decrease post-pruning

### Run
-Run the Part34.ipynb to see Channel Pruning results

---

## References

1. **Radford, A., et al.**  
   *Learning Transferable Visual Models from Natural Language Supervision.*  
   In: Proceedings of the International Conference on Machine Learning (ICML), 2021.  
   [Paper Link](https://arxiv.org/abs/2103.00020)

2. **Touvron, H., et al.**  
   *LLaMA: Open and Efficient Foundation Language Models.*  
   arXiv preprint, 2023.  
   [Paper Link](https://arxiv.org/abs/2302.13971)

3. **Lin, J., et al.**  
   *VILA: On Pre-training for Visual Language Models.*  
   arXiv preprint, 2023.  
   [Paper Link](https://arxiv.org/abs/2312.07533)

4. **Schuhmann, C., et al.**  
   *LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models.*  
   Advances in Neural Information Processing Systems (NeurIPS), 2022.  
   [Paper Link](https://arxiv.org/abs/2210.08402)

5. **Hudson, D. A., & Manning, C. D.**  
   *GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.*  
   In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.  
   [Paper Link](https://arxiv.org/abs/1902.09506)

6. **He, Y., Zhang, X., & Sun, J.**  
   *Channel Pruning for Accelerating Very Deep Neural Networks.*  
   In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.  
   [Paper Link](https://arxiv.org/abs/1707.06168)

7. **Goyal, Y., et al.**  
   *Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering.*  
   In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.  
   [Paper Link](https://arxiv.org/abs/1612.00837)

