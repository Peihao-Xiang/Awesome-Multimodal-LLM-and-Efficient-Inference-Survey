# AAAI 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_aaai2023_papers.csv

## 1. Efficient End-to-End Video Question Answering with Pyramidal Multimodal Transformer

- [ ] Efficient End-to-End Video Question Answering with Pyramidal Multimodal Transformer | https://ojs.aaai.org/index.php/AAAI/article/view/25296

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25296

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a new method for end-to-end Video Question Answering (VideoQA), aside from the current popularity of using large-scale pre-training with huge feature extractors. We achieve this with a pyramidal multimodal transformer (PMT) model, which simply incorporates a learnable word embedding layer, a few convolutional and transformer layers. We use the anisotropic pyramid to fulfill video-language interactions across different spatio-temporal scales. In addition to the canonical pyramid, which includes both bottom-up and top-down pathways with lateral connections, novel strategies are proposed to decompose the visual feature stream into spatial and temporal sub-streams at different scales and implement their interactions with the linguistic semantics while preserving the integrity of local and global semantics. We demonstrate better or on-par performances with high computational efficiency against state-of-the-art methods on five VideoQA benchmarks. Our ablation study shows the scalability of our model that achieves competitive results for text-to-video retrieval by leveraging feature extractors with reusable pre-trained weights, and also the effectiveness of the pyramid. Code available at: https://github.com/Trunpm/PMT-AAAI23.

</details>

---

