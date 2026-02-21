# 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_2023_papers.csv

## 1. Efficient End-to-End Video Question Answering with Pyramidal Multimodal Transformer

- [x] Efficient End-to-End Video Question Answering with Pyramidal Multimodal Transformer | https://ojs.aaai.org/index.php/AAAI/article/view/25296

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25296

- **Conference**: AAAI

- **Year**: 2023

- **Type**: Lightweight architecture

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a new method for end-to-end Video Question Answering (VideoQA), aside from the current popularity of using large-scale pre-training with huge feature extractors. We achieve this with a pyramidal multimodal transformer (PMT) model, which simply incorporates a learnable word embedding layer, a few convolutional and transformer layers. We use the anisotropic pyramid to fulfill video-language interactions across different spatio-temporal scales. In addition to the canonical pyramid, which includes both bottom-up and top-down pathways with lateral connections, novel strategies are proposed to decompose the visual feature stream into spatial and temporal sub-streams at different scales and implement their interactions with the linguistic semantics while preserving the integrity of local and global semantics. We demonstrate better or on-par performances with high computational efficiency against state-of-the-art methods on five VideoQA benchmarks. Our ablation study shows the scalability of our model that achieves competitive results for text-to-video retrieval by leveraging feature extractors with reusable pre-trained weights, and also the effectiveness of the pyramid. Code available at: https://github.com/Trunpm/PMT-AAAI23.

</details>

---

## 2. CocaCLIP: Exploring Distillation of Fully-Connected Knowledge Interaction Graph for Lightweight Text-Image Retrieval

- [x] CocaCLIP: Exploring Distillation of Fully-Connected Knowledge Interaction Graph for Lightweight Text-Image Retrieval | https://aclanthology.org/2023.acl-industry.8/

- **Link**: https://aclanthology.org/2023.acl-industry.8/

- **Conference**: ACL

- **Year**: 2023

- **Type**: Model Distillation

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained text-image models with dual-encoder architectures (such as CLIP) are typically adopted for various vision-language applications, including text-image retrieval. However, these models are still less practical on edge devices or for real-time situations, due to the substantial indexing and inference time and the large consumption of computational resources. Although knowledge distillation techniques have been widely utilized for uni-modal model compression, how to expand them to the situation when the numbers of modalities and teachers/students are doubled has been rarely studied. In this paper, we conduct comprehensive experiments on this topic and propose the fully-Connected knowledge interaction graph (Coca) technique for cross-modal pre-training distillation. Based on our findings, the resulting CocaCLIP achieves SOTA performances on the widely-used Flickr30K and MSCOCO benchmarks under the lightweight setting. An industry application of our method on an e-commercial platform further demonstrates the significant effectiveness of CocaCLIP.

</details>

---

## 3. PuMer: Pruning and Merging Tokens for Efficient Vision Language Models

- [x] PuMer: Pruning and Merging Tokens for Efficient Vision Language Models | https://aclanthology.org/2023.acl-long.721/

- **Link**: https://aclanthology.org/2023.acl-long.721/

- **Conference**: ACL

- **Year**: 2023

- **Type**: Token Pruning & Merging

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision language (VL) models use Transformers to perform cross-modal interactions between the input text and image. These cross-modal interactions are computationally expensive and memory-intensive due to the quadratic complexity of processing the input image and text. We present PuMer: a token reduction framework that uses text-informed Pruning and modality-aware Merging strategies to progressively reduce the tokens of input image and text, improving model inference speed and reducing memory footprint. PuMer learns to keep salient image tokens related to the input text and merges similar textual and visual tokens by adding lightweight token reducer modules at several cross-modal layers in the VL model. Training PuMer is mostly the same as finetuning the original VL model but faster. Our evaluation for two vision language models on four downstream VL tasks shows PuMer increases inference throughput by up to 2x and reduces memory footprint by over 50% while incurring less than a 1% accuracy drop.

</details>

---

## 4. Fusion or Defusion? Flexible Vision-and-Language Pre-Training

- [x] Fusion or Defusion? Flexible Vision-and-Language Pre-Training | https://aclanthology.org/2023.findings-acl.316/

- **Link**: https://aclanthology.org/2023.findings-acl.316/

- **Conference**: ACL

- **Year**: 2023

- **Type**: Dynamic routing

<details>
<summary><strong>Abstract</strong></summary>

Existing approaches in the vision-and-language pre-training (VLP) paradigm mainly deploy either fusion-based encoders or dual-encoders, failing to achieve both effectiveness and efficiency in downstream multimodal tasks. In this paper, we build a flexible VLP model by incorporating cross-modal fusions into a dual-encoder architecture, where the introduced fusion modules can be easily decoupled from the dual encoder so as to switch the model to a fusion-free one. To better absorb cross-modal features from the fusion modules, we design a cross-modal knowledge transfer strategy along with other comprehensive pre-training tasks to guide the training process, which can further strengthen both the fusion-based and fusion-free representation learning. Extensive experiments conducted on various downstream vision-language tasks show that our proposed model is well-equipped with effectiveness as well as efficiency, demonstrating a superior performance compared with other strong VLP models.

</details>

---

## 5. EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning

- [x] EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning | https://aclanthology.org/2023.findings-acl.873/

- **Link**: https://aclanthology.org/2023.findings-acl.873/

- **Conference**: ACL

- **Year**: 2023

- **Type**: Model Distillation & Pruning

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have achieved impressive results in a range of vision-language tasks. However, popular VLMs usually consist of hundreds of millions of parameters which brings challenges for fine-tuning and deployment in real-world applications due to space, memory, and latency constraints. In this work, we introduce a distilling then pruning framework to compress large vision-language models into smaller, faster, and more accurate ones. We first shrink the size ofa pre-trained large VLM and apply knowledge distillation in the vision-language pre-training stage to obtain a task-agnostic compact VLM. Then we propose a modal-adaptive pruning algorithm to automatically infer the importance of vision and language modalities for different downstream tasks and adaptively remove redundant structures and neurons in different encoders with controllable target sparsity. We apply our framework to train EfficientVLM, a fast and accurate vision-language model consisting of 6 vision layers, 3 text layers, and 3 cross-modal fusion layers, accounting for only 93 million parameters in total, which is 44.3% of the teacher model. EfficientVLM retains 98.4% performance of the teacher model and accelerates its inference speed by 2.2×. EfficientVLM achieves a large absolute improvement over previous SoTA efficient VLMs of similar sizes by a large margin on various vision-language tasks, including VQAv2 (+4.9%), NLVR2 (+5.6%), ITR (R@1 on TR +17.2%, on IR + 15.6% ) and COCO caption generation (CIDEr +6.5), demonstrating a large potential on training lightweight VLMs.

</details>

---

## 6. CLIPPING: Distilling CLIP-Based Models With a Student Base for Video-Language Retrieval

- [x] CLIPPING: Distilling CLIP-Based Models With a Student Base for Video-Language Retrieval | https://cvpr.thecvf.com/virtual/2023/poster/20979

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/20979

- **Conference**: CVPR

- **Year**: 2023

- **Type**: Distilling

<details>
<summary><strong>Abstract</strong></summary>

Pre-training a vison-language model and then fine-tuning it on downstream tasks have become a popular paradigm. However, pre-trained vison-language models with the Transformer architecture usually take long inference time. Knowledge distillation has been an efficient technique to transfer the capability of a large model to a small one while maintaining the accuracy, which has achieved remarkable success in natural language processing. However, it faces many problems when applying KD to the multi-modality applications. In this paper, we propose a novel knowledge distillation method, named CLIPPING, where the plentiful knowledge of a large teacher model that has been fine-tuned for video-language tasks with the powerful pre-trained CLIP can be effectively transferred to a small student only at the fine-tuning stage. Especially, a new layer-wise alignment with the student as the base is proposed for knowledge distillation of the intermediate layers in CLIPPING, which enables the student’s layers to be the bases of the teacher, and thus allows the student to fully absorb the knowledge of the teacher. CLIPPING with MobileViT-v2 as the vison encoder without any vison-language pre-training achieves 88.1%-95.3% of the performance of its teacher on three video-language retrieval benchmarks, with its vison encoder being 19.5x smaller. CLIPPING also significantly outperforms a state-of-the-art small baseline (ALL-in-one-B) on the MSR-VTT dataset, obtaining relatively 7.4% performance gain, with 29% fewer parameters and 86.9% fewer flops. Moreover, CLIPPING is comparable or even superior to many large pre-training models.

</details>

---

## 7. You Need Multiple Exiting: Dynamic Early Exiting for Accelerating Unified Vision Language Model

- [x] You Need Multiple Exiting: Dynamic Early Exiting for Accelerating Unified Vision Language Model | https://cvpr.thecvf.com/virtual/2023/poster/21102

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21102

- **Conference**: CVPR

- **Year**: 2023

- **Type**: Early Exiting

<details>
<summary><strong>Abstract</strong></summary>

Large-scale transformer models bring significant improvements for various downstream vision language tasks with a unified architecture. The performance improvements come with increasing model size, resulting in slow inference speed and increased cost for severing. While some certain predictions benefit from the full complexity of the large-scale model, not all of input need the same amount of computation to conduct, potentially leading to computation resource waste. To handle this challenge, early exiting is proposed to adaptively allocate computational power in term of input complexity to improve inference efficiency. The existing early exiting strategies usually adopt output confidence based on intermediate layers as a proxy of input complexity to incur the decision of skipping following layers. However, such strategies cannot apply to encoder in the widely-used unified architecture with both encoder and decoder due to difficulty of output confidence estimation in the encoder. It is suboptimal in term of saving computation power to ignore the early exiting in encoder component. To handle this challenge, we propose a novel early exiting strategy for unified visual language models, which allows dynamically skip the layers in encoder and decoder simultaneously in term of input layer-wise similarities with multiple times of early exiting, namely MuE. By decomposing the image and text modalities in the encoder, MuE is flexible and can skip different layers in term of modalities, advancing the inference efficiency while minimizing performance drop. Experiments on the SNLI-VE and MS COCO datasets show that the proposed approach MuE can reduce inference time by up to 50% and 40% while maintaining 99% and 96% performance respectively.

</details>

---

## 8. Side Adapter Network for Open-Vocabulary Semantic Segmentation

- [x] Side Adapter Network for Open-Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2023/poster/21298

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21298

- **Conference**: CVPR

- **Year**: 2023

- **Type**: Efficient Adapter

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a new framework for open-vocabulary semantic segmentation with the pre-trained vision-language model, named SAN. Our approach models the semantic segmentation task as a region recognition problem. A side network is attached to a frozen CLIP model with two branches: one for predicting mask proposals, and the other for predicting attention bias which is applied in the CLIP model to recognize the class of masks. This decoupled design has the benefit CLIP in recognizing the class of mask proposals. Since the attached side network can reuse CLIP features, it can be very light. In addition, the entire network can be trained end-to-end, allowing the side network to be adapted to the frozen CLIP model, which makes the predicted mask proposals CLIP-aware. Our approach is fast, accurate, and only adds a few additional trainable parameters. We evaluate our approach on multiple semantic segmentation benchmarks. Our method significantly outperforms other counterparts, with up to 18 times fewer trainable parameters and 19 times faster inference speed. We hope our approach will serve as a solid baseline and help ease future research in open-vocabulary semantic segmentation.

</details>

---
## 9. All in One: Exploring Unified Video-Language Pre-Training

- [x] All in One: Exploring Unified Video-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/22225

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22225

- **Conference**: CVPR

- **Year**: 2023

- **Type**: Lightweight Archecture

<details>
<summary><strong>Abstract</strong></summary>

Mainstream Video-Language Pre-training models consist of three parts, a video encoder, a text encoder, and a video-text fusion Transformer. They pursue better performance via utilizing heavier unimodal encoders or multimodal fusion Transformers, resulting in increased parameters with lower efficiency in downstream tasks. In this work, we for the first time introduce an end-to-end video-language model, namely all-in-one Transformer, that embeds raw video and textual signals into joint representations using a unified backbone architecture. We argue that the unique temporal information of video data turns out to be a key barrier hindering the design of a modality-agnostic Transformer. To overcome the challenge, we introduce a novel and effective token rolling operation to encode temporal representations from video clips in a non-parametric manner. The careful design enables the representation learning of both video-text multimodal inputs and unimodal inputs using a unified backbone model. Our pre-trained all-in-one Transformer is transferred to various downstream video-text tasks after fine-tuning, including text-video retrieval, video-question answering, multiple choice and visual commonsense reasoning. State-of-the-art performances with the minimal model FLOPs on nine datasets demonstrate the superiority of our method compared to the competitive counterparts.

</details>

---

## 10. Position-Guided Text Prompt for Vision-Language Pre-Training

- [x] Position-Guided Text Prompt for Vision-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/22379

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22379

- **Conference**: CVPR

- **Year**: 2023

- **Type**: Simplified Architecture

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pre-Training (VLP) has shown promising capabilities to align image and text pairs, facilitating a broad variety of cross-modal learning tasks. However, we observe that VLP models often lack the visual grounding/localization capability which is critical for many downstream tasks such as visual reasoning. In this work, we propose a novel Position-guided Text Prompt (PTP) paradigm to enhance the visual grounding ability of cross-modal models trained with VLP. Specifically, in the VLP phase, PTP divides the image into NxN blocks, and identifies the objects in each block through the widely used object detector in VLP. It then reformulates the visual grounding task into a fill-in-the-blank problem given a PTP by encouraging the model to predict the objects in the given blocks or regress the blocks of a given object, e.g. filling “P” or “O” in a PTP “The block P has a O”. This mechanism improves the visual grounding capability of VLP models and thus helps them better handle various downstream tasks. By introducing PTP into several state-of-the-art VLP frameworks, we observe consistently significant improvements across representative cross-modal learning model architectures and several benchmarks, e.g. zero-shot Flickr30K Retrieval (+4.8 in average recall@1) for ViLT baseline, and COCO Captioning (+5.3 in CIDEr) for SOTA BLIP baseline. Moreover, PTP achieves comparable results with object-detector based methods, and much faster inference speed since PTP discards its object detector for inference while the later cannot. Our code and pre-trained weight will be released.

</details>

---

## 11. FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks

- [x] FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks | https://cvpr.thecvf.com/virtual/2023/poster/23158

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23158

- **Conference**: CVPR

- **Year**: 2023

- **Type**: Efficient Adapter

<details>
<summary><strong>Abstract</strong></summary>

In the fashion domain, there exists a variety of vision-and-language (V+L) tasks, including cross-modal retrieval, text-guided image retrieval, multi-modal classification, and image captioning. They differ drastically in each individual input/output format and dataset size. It has been common to design a task-specific model and fine-tune it independently from a pre-trained V+L model (e.g., CLIP). This results in parameter inefficiency and inability to exploit inter-task relatedness. To address such issues, we propose a novel FAshion-focused Multi-task Efficient learning method for Vision-and-Language tasks (FAME-ViL) in this work. Compared with existing approaches, FAME-ViL applies a single model for multiple heterogeneous fashion tasks, therefore being much more parameter-efficient. It is enabled by two novel components: (1) a task-versatile architecture with cross-attention adapters and task-specific adapters integrated into a unified V+L model, and (2) a stable and effective multi-task training strategy that supports learning from heterogeneous data and prevents negative transfer. Extensive experiments on four fashion tasks show that our FAME-ViL can save 61.5% of parameters over alternatives, while significantly outperforming the conventional independently trained single-task models. Code is available at https://github.com/BrandonHanx/FAME-ViL.

</details>

---

## 12. A Suite of Generative Tasks for Multi-Level Multimodal Webpage Understanding

- [x] A Suite of Generative Tasks for Multi-Level Multimodal Webpage Understanding | https://aclanthology.org/2023.emnlp-main.119/

- **Link**: https://aclanthology.org/2023.emnlp-main.119/

- **Conference**: EMNLP

- **Year**: 2023

- **Type**: Sparse Attention

<details>
<summary><strong>Abstract</strong></summary>

Webpages have been a rich, scalable resource for vision-language and language only tasks. Yet only pieces of webpages are kept in existing datasets: image-caption pairs, long text articles, or raw HTML, never all in one place. Webpage tasks have resultingly received little attention and structured image-text data left underused. To study multimodal webpage understanding, we introduce the Wikipedia Webpage suite (WikiWeb2M) containing 2M pages with all of the associated image, text, and structure data. We verify its utility on three generative tasks: page description generation, section summarization, and contextual image captioning. We design a novel attention mechanism Prefix Global, which selects the most relevant image and text content as global tokens to attend to the rest of the webpage for context. By using page structure to separate such tokens, it performs better than full attention with lower computational complexity. Extensive experiments show that the new data in WikiWeb2M improves task performance compared to prior work.

</details>

---

## 13. Compressing and Debiasing Vision-Language Pre-Trained Models for Visual Question Answering

- [x] Compressing and Debiasing Vision-Language Pre-Trained Models for Visual Question Answering | https://aclanthology.org/2023.emnlp-main.34/

- **Link**: https://aclanthology.org/2023.emnlp-main.34/

- **Conference**: EMNLP

- **Year**: 2023

- **Type**: Sparse Subnetwork Searching

<details>
<summary><strong>Abstract</strong></summary>

Despite the excellent performance of vision-language pre-trained models (VLPs) on conventional VQA task, they still suffer from two problems: First, VLPs tend to rely on language biases in datasets and fail to generalize to out-of-distribution (OOD) data. Second, they are inefficient in terms of memory footprint and computation. Although promising progress has been made in both problems, most existing works tackle them independently. To facilitate the application of VLP to VQA tasks, it is imperative to jointly study VLP compression and OOD robustness, which, however, has not yet been explored. This paper investigates whether a VLP can be compressed and debiased simultaneously by searching sparse and robust subnetworks. To this end, we systematically study the design of a training and compression pipeline to search the subnetworks, as well as the assignment of sparsity to different modality-specific modules. Our experiments involve 2 VLPs, 2 compression methods, 4 training methods, 2 datasets and a range of sparsity levels. Our results show that there indeed exist sparse and robust subnetworks, which are competitive with the debiased full VLP and clearly outperform the debiasing SoTAs with fewer parameters on OOD datasets VQA-CP v2 and VQA-VS. The codes can be found at https://github.com/PhoebusSi/Compress-Robust-VQA.

</details>

---

## 14. TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding

- [x] TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding | https://aclanthology.org/2023.findings-emnlp.66/

- **Link**: https://aclanthology.org/2023.findings-emnlp.66/

- **Conference**: EMNLP

- **Year**: 2023

- **Type**: Token Merging

<details>
<summary><strong>Abstract</strong></summary>

Large-scale video-language pre-training has made remarkable strides in advancing video-language understanding tasks. However, the heavy computational burden of video encoding remains a formidable efficiency bottleneck, particularly for long-form videos. These videos contain massive visual tokens due to their inherent 3D properties and spatiotemporal redundancy, making it challenging to capture complex temporal and spatial relationships. To tackle this issue, we propose an efficient method called TEmporal-Spatial Token Aggregation (TESTA). TESTA condenses video semantics by adaptively aggregating similar frames, as well as similar patches within each frame. TESTA can reduce the number of visual tokens by 75% and thus accelerate video encoding. Building upon TESTA, we introduce a pre-trained video-language model equipped with a divided space-time token aggregation module in each video encoder block. We evaluate our model on five datasets for paragraph-to-video retrieval and long-form VideoQA tasks. Experimental results show that TESTA improves computing efficiency by 1.7 times, and achieves significant performance gains from its scalability in processing longer input frames, e.g., +13.7 R@1 on QuerYD and +6.5 R@1 on Condensed Movie.

</details>

---

## 15. Scaling Vision-Language Models with Sparse Mixture of Experts

- [x] Scaling Vision-Language Models with Sparse Mixture of Experts | https://aclanthology.org/2023.findings-emnlp.758/

- **Link**: https://aclanthology.org/2023.findings-emnlp.758/

- **Conference**: EMNLP

- **Year**: 2023

- **Type**: Sparse MoE

<details>
<summary><strong>Abstract</strong></summary>

The field of natural language processing (NLP) has made significant strides in recent years, particularly in the development of large-scale vision-language models (VLMs). These models aim to bridge the gap between text and visual information, enabling a more comprehensive understanding of multimedia data. However, as these models become larger and more complex, they also become more challenging to train and deploy. One approach to addressing this challenge is the use of sparsely-gated mixture-of-experts (MoE) techniques, which divide the model into smaller, specialized sub-models that can jointly solve a task. In this paper, we explore the effectiveness of MoE in scaling vision-language models, demonstrating its potential to achieve state-of-the-art performance on a range of benchmarks over dense models of equivalent computational cost. Our research offers valuable insights into stabilizing the training of MoE models, understanding the impact of MoE on model interpretability, and balancing the trade-offs between compute performance when scaling VLMs. We hope our work will inspire further research into the use of MoE for scaling large-scale vision-language models and other multimodal machine learning applications.

</details>

---

## 16. Global Knowledge Calibration for Fast Open-Vocabulary Segmentation

- [x] Global Knowledge Calibration for Fast Open-Vocabulary Segmentation | https://openaccess.thecvf.com/content/ICCV2023/html/Han_Global_Knowledge_Calibration_for_Fast_Open-Vocabulary_Segmentation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Han_Global_Knowledge_Calibration_for_Fast_Open-Vocabulary_Segmentation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Model Distillation

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in pre-trained vision-language models, such as CLIP, have enabled the segmentation of arbitrary concepts solely from textual inputs, a process commonly referred to as open-vocabulary semantic segmentation (OVS). However, existing OVS techniques confront a fundamental challenge: the trained classifier tends to overfit on the base classes observed during training, resulting in suboptimal generalization performance to unseen classes. To mitigate this issue, recent studies have proposed the use of an additional frozen pre-trained CLIP for classification. Nonetheless, this approach incurs heavy computational overheads as the CLIP vision encoder must be repeatedly forward-passed for each mask, rendering it impractical for real-world applications. To address this challenge, our objective is to develop a fast OVS model that can perform comparably or better without the extra computational burden of the CLIP image encoder during inference. To this end, we propose a core idea of preserving the generalizable representation when fine-tuning on known classes. Specifically, we introduce a text diversification strategy that generates a set of synonyms for each training category, which prevents the learned representation from collapsing onto specific known category names. Additionally, we employ a text-guided knowledge distillation method to preserve the generalizable knowledge of CLIP. Extensive experiments demonstrate that our proposed model achieves robust generalization performance across various datasets. Furthermore, we perform a preliminary exploration of open-vocabulary video segmentation and present a benchmark that can facilitate future open-vocabulary research in the video domain.

</details>

---

## 17. BUS: Efficient and Effective Vision-Language Pre-Training with Bottom-Up Patch Summarization.

- [x] BUS: Efficient and Effective Vision-Language Pre-Training with Bottom-Up Patch Summarization. | https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_BUS_Efficient_and_Effective_Vision-Language_Pre-Training_with_Bottom-Up_Patch_Summarization._ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_BUS_Efficient_and_Effective_Vision-Language_Pre-Training_with_Bottom-Up_Patch_Summarization._ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Token Pruning

<details>
<summary><strong>Abstract</strong></summary>

Vision Transformer (ViT) based Vision-Language Pretraining (VLP) models recently demonstrated impressive performance in various tasks. However, the lengthy visual token sequences used in these models can lead to inefficient and ineffective performance. Existing methods to address these issues lack textual guidance and may overlook crucial visual information related to the text, leading to the introduction of irrelevant information during cross-modal fusion and additional computational cost. In this paper, we propose a Bottom-Up Patch Summarization approach named BUS which is inspired by the Document Summarization Task in NLP to learn a concise visual summary of lengthy visual token sequences, guided by textual semantics. We introduce a Text-Semantic Aware Patch Selector (TAPS) in the ViT backbone to perform a coarse-grained selective visual summarization to over-determine the text-relevant patches, and a light Summarization Decoder to perform fine-grained abstractive summarization based on the selected patches, resulting in a further condensed representation sequence that highlights text-relevant visual semantic information. Such bottom-up process is both efficient and effective with higher performing. We evaluate our approach on various VL understanding and generation tasks and show competitive or better downstream task performance while boosting the efficiency by 50%. Additionally, our model achieves well-designed SOTA downstream task performance by increasing input image resolution without increasing computational costs compared to baselines.

</details>

---

## 18. Distilling Large Vision-Language Model with Out-of-Distribution Generalizability

- [x] Distilling Large Vision-Language Model with Out-of-Distribution Generalizability | https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Model Distillation

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models have achieved outstanding performance, but their size and computational requirements make their deployment on resource-constrained devices and time-sensitive tasks impractical. Model distillation, the process of creating smaller, faster models that maintain the performance of larger models, is a promising direction towards the solution. This paper investigates the distillation of visual representations in large teacher vision-language models into lightweight student models using a small- or mid-scale dataset. Notably, this study focuses on open-vocabulary out-of-distribution (OOD) generalization, a challenging problem that has been overlooked in previous model distillation literature. We propose two principles from vision and language modality perspectives to enhance student's OOD generalization: (1) by better imitating teacher's visual representation space, and carefully promoting better coherence in vision-language alignment with the teacher; (2) by enriching the teacher's language representations with informative and finegrained semantic attributes to effectively distinguish between different labels. We propose several metrics and conduct extensive experiments to investigate their techniques. The results demonstrate significant improvements in zero-shot and few-shot student performance on open-vocabulary out-of-distribution classification, highlighting the effectiveness of our proposed approaches.

</details>

---

## 19. Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection

- [x] Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection | https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Model Distillation

<details>
<summary><strong>Abstract</strong></summary>

Current methods for open-vocabulary object detection (OVOD) rely on a pre-trained vision-language model (VLM) to acquire the recognition ability. In this paper, we propose a simple yet effective framework to Distill the Knowledge from the VLM to a DETR-like detector, termed DK-DETR. Specifically, we present two ingenious distillation schemes named semantic knowledge distillation (SKD) and relational knowledge distillation (RKD). To utilize the rich knowledge from the VLM systematically, SKD transfers the semantic knowledge explicitly, while RKD exploits implicit relationship information between objects. Furthermore, a distillation branch including a group of auxiliary queries is added to the detector to mitigate the negative effect on base categories. Equipped with SKD and RKD on the distillation branch, DK-DETR improves the detection performance of novel categories significantly and avoids disturbing the detection of base categories. Extensive experiments on LVIS and COCO datasets show that DK-DETR surpasses existing OVOD methods under the setting that the base-category supervision is solely available. The code and models are available at https://github.com/hikvision-research/opera.

</details>

---

## 20. SMAUG: Sparse Masked Autoencoder for Efficient Video-Language Pre-Training

- [x] SMAUG: Sparse Masked Autoencoder for Efficient Video-Language Pre-Training | https://openaccess.thecvf.com/content/ICCV2023/html/Lin_SMAUG_Sparse_Masked_Autoencoder_for_Efficient_Video-Language_Pre-Training_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lin_SMAUG_Sparse_Masked_Autoencoder_for_Efficient_Video-Language_Pre-Training_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Token Sparsification

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-training is crucial for learning powerful multi-modal representation. However, it typically requires a massive amount of computation. In this paper, we develop SMAUG, an efficient pre-training framework for video-language models. The foundation component in SMAUG is masked autoencoders. Different from prior works which only mask textual inputs, our masking strategy considers both visual and textual modalities, providing a better cross-modal alignment and saving more pre-training costs. On top of that, we introduce a space-time token sparsification module, which leverages context information to further select only "important" spatial regions and temporal frames for pre-training. Coupling all these designs allows our method to enjoy both competitive performances on text-to-video retrieval and video question answering tasks, and much less pre-training costs by 1.9x or more. For example, our SMAUG only needs  50 NVIDIA A6000 GPU hours for pre-training to attain competitive performances on these two video-language tasks across six popular benchmarks.

</details>

---

## 21. Spectrum-guided Multi-granularity Referring Video Object Segmentation

- [x] Spectrum-guided Multi-granularity Referring Video Object Segmentation | https://openaccess.thecvf.com/content/ICCV2023/html/Miao_Spectrum-guided_Multi-granularity_Referring_Video_Object_Segmentation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Miao_Spectrum-guided_Multi-granularity_Referring_Video_Object_Segmentation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Simplified Architecture

<details>
<summary><strong>Abstract</strong></summary>

Current referring video object segmentation (R-VOS) techniques extract conditional kernels from encoded (low-resolution) vision-language features to segment the decoded high-resolution features. We discovered that this causes significant feature drift, which the segmentation kernels struggle to perceive during the forward computation. This negatively affects the ability of segmentation kernels. To address the drift problem, we propose a Spectrum-guided Multi-granularity (SgMg) approach, which performs direct segmentation on the encoded features and employs visual details to further optimize the masks. In addition, we propose Spectrum-guided Cross-modal Fusion (SCF) to perform intra-frame global interactions in the spectral domain for effective multimodal representation. Finally, we extend SgMg to perform multi-object R-VOS, a new paradigm that enables simultaneous segmentation of multiple referred objects in a video. This not only makes R-VOS faster, but also more practical. Extensive experiments show that SgMg achieves state-of-the-art performance on four video benchmark datasets, outperforming the nearest competitor by 2.8% points on Ref-YouTube-VOS. Our extended SgMg enables multi-object R-VOS, runs about 3 times faster while maintaining satisfactory performance. Code is available at https://github.com/bo-miao/SgMg.

</details>

---

## 22. EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone

- [x] EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone | https://openaccess.thecvf.com/content/ICCV2023/html/Pramanick_EgoVLPv2_Egocentric_Video-Language_Pre-training_with_Fusion_in_the_Backbone_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Pramanick_EgoVLPv2_Egocentric_Video-Language_Pre-training_with_Fusion_in_the_Backbone_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Efficient Fusion

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-training (VLP) has become increasingly important due to its ability to generalize to various vision and language tasks. However, existing egocentric VLP frameworks utilize separate video and language encoders and learn task-specific cross-modal information only during fine-tuning, limiting the development of a unified system. In this work, we introduce the second generation of egocentric video-language pre-training (EgoVLPv2), a significant improvement from the previous generation, by incorporating cross-modal fusion directly into the video and language backbones. EgoVLPv2 learns strong video-text representation during pre-training and reuses the cross-modal attention modules to support different downstream tasks in a flexible and efficient manner, reducing fine-tuning costs. Moreover, our proposed fusion in the backbone strategy is more lightweight and compute-efficient than stacking additional fusion-specific layers. Extensive experiments on a wide range of VL tasks demonstrate the effectiveness of EgoVLPv2 by achieving consistent state-of-the-art performance over strong baselines across all downstream.

</details>

---

## 23. DIME-FM : DIstilling Multimodal and Efficient Foundation Models

- [x] DIME-FM : DIstilling Multimodal and Efficient Foundation Models | https://openaccess.thecvf.com/content/ICCV2023/html/Sun_DIME-FM__DIstilling_Multimodal_and_Efficient_Foundation_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Sun_DIME-FM__DIstilling_Multimodal_and_Efficient_Foundation_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Model Distillation

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Foundation Models (VLFM), such as CLIP, ALIGN and Florence, are trained on large private datasets of image-caption pairs and achieve superior transferability and robustness on downstream tasks, but they are difficult to use in many practical applications due to their large size, high latency and fixed architectures. Unfortunately, recent works show training a small custom VLFM for resource-limited applications is currently very difficult using public and smaller-scale data. In this paper, we introduce a new distillation mechanism (DIME-FM) that allows us to transfer the knowledge contained in large VLFMs to smaller, customized foundation models using a relatively small amount of inexpensive, unpaired images and sentences. We transfer the knowledge from the pre-trained CLIP-ViT-L/14 model to a ViT-B/32 model, with only 40M public images and 28.4M unpaired public sentences. The resulting model "Distill-ViT-B/32" rivals the CLIP-ViT-B/32 model pre-trained on its private WiT dataset (400M image-text pairs): Distill-ViT-B/32 achieves similar results in terms of zero-shot and linear-probing performance on both ImageNet and the ELEVATER (20 image classification tasks) benchmarks. It also displays comparable robustness when evaluated on five datasets with natural distribution shifts from ImageNet.

</details>

---

## 24. Attentive Mask CLIP

- [x] Attentive Mask CLIP | https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Attentive_Mask_CLIP_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Attentive_Mask_CLIP_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Token Pruning

<details>
<summary><strong>Abstract</strong></summary>

In vision-language modeling, image token removal is an efficient augmentation technique to reduce the cost of encoding image features. The CLIP-style models, however, have been found to be negatively impacted by this technique. We hypothesize that removing a large portion of image tokens may inadvertently destroy the semantic information associated to a given text description, resulting in misaligned paired data in CLIP training. To address this issue, we propose an attentive token removal approach, which retains a small number of tokens that have a strong semantic correlation to the corresponding text description. The correlation scores are dynamically evaluated through an EMA-updated vision encoder. Our method, termed attentive mask CLIP, outperforms original CLIP and CLIP variant with random token removal while saving the training time. In addition, our approach also enables efficient multi-view contrastive learning. Experimentally, by training ViT-B on YFCC-15M dataset, our approach achieves 43.9% top-1 accuracy on ImageNet-1K zero-shot classification, 62.7/42.1 and 38.0/23.2 I2T/T2I retrieval accuracy on Flickr30K and MS COCO, outperforming SLIP by +1.1%,+5.5/+0.9, and +4.4/+1.3, respectively, while being 2.30x faster. An efficient version of our approach runs 1.16x faster than the plain CLIP model, while achieving significant gains of +5.3%, +11.3/+8.0, and +9.5/+4.9 on these benchmarks, respectively.

</details>

---

## 25. Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-Trained Vision-Language Models

- [x] Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-Trained Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Regularized_Mask_Tuning_Uncovering_Hidden_Knowledge_in_Pre-Trained_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Regularized_Mask_Tuning_Uncovering_Hidden_Knowledge_in_Pre-Trained_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

- **Type**: Sparse Routing/Activation

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning and adapter tuning have shown great potential in transferring pre-trained vision-language models (VLMs) to various downstream tasks. In this work, we design a new type of tuning method, termed as regularized mask tuning, which masks the network parameters through a learnable selection. Inspired by neural pathways, we argue that the knowledge required by a downstream task already exists in the pre-trained weights but just gets concealed in the upstream pre-training stage. To bring the useful knowledge back into light, we first identify a set of parameters that are important to a given downstream task, then attach a binary mask to each parameter, and finally optimize these masks on the downstream data with the parameters frozen. When updating the mask, we introduce a novel gradient dropout strategy to regularize the parameter selection, in order to prevent the model from forgetting old knowledge and overfitting the downstream data. Experimental results on 11 datasets demonstrate the consistent superiority of our method over previous alternatives. It is noteworthy that we manage to deliver 18.73% performance improvement compared to the zero-shot CLIP via masking an average of only 2.56% parameters. Furthermore, our method is synergistic with most existing parameter-efficient tuning methods and can boost the performance on top of them. Code will be made publicly available.

</details>

---

## 26. UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers

- [x] UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers | https://icml.cc/virtual/2023/poster/23979

- **Link**: https://icml.cc/virtual/2023/poster/23979

- **Conference**: ICML

- **Year**: 2023

- **Type**: Model Pruning

<details>
<summary><strong>Abstract</strong></summary>

Real-world data contains a vast amount of multimodal information, among which vision and language are the two most representative modalities. Moreover, increasingly heavier models, e.g., Transformers, have attracted the attention of researchers to model compression. However, how to compress multimodal models, especially vison-language Transformers, is still under-explored. This paper proposes the Unified and Progressive Pruning (UPop) as a universal vison-language Transformer compression framework, which incorporates 1) unifiedly searching multimodal subnets in a continuous optimization space from the original model, which enables automatic assignment of pruning ratios among compressible modalities and structures; 2) progressively searching and retraining the subnet, which maintains convergence between the search and retrain to attain higher compression ratios. Experiments on various tasks, datasets, and model architectures demonstrate the effectiveness and versatility of the proposed UPop framework. The code is available at https://github.com/sdc17/UPop.

</details>

---

## 27. Distilling Internet-Scale Vision-Language Models into Embodied Agents

- [x] Distilling Internet-Scale Vision-Language Models into Embodied Agents | https://icml.cc/virtual/2023/poster/24664

- **Link**: https://icml.cc/virtual/2023/poster/24664

- **Conference**: ICML

- **Year**: 2023

- **Type**: Model Distillation

<details>
<summary><strong>Abstract</strong></summary>

Instruction-following agents must ground language into their observation and action spaces. Learning to ground language is challenging, typically requiring domain-specific engineering or large quantities of human interaction data. To address this challenge, we propose using pretrained vision-language models (VLMs) to supervise embodied agents. We combine ideas from model distillation and hindsight experience replay (HER), using a VLM to retroactively generate language describing the agent's behavior. Simple prompting allows us to control the supervision signal, teaching an agent to interact with novel objects based on their names (e.g., planes) or their features (e.g., colors) in a 3D rendered environment. Fewshot prompting lets us teach abstract category membership, including pre-existing categories (food vs toys) and ad-hoc ones (arbitrary preferences over objects). Our work outlines a new and effective way to use internet-scale VLMs, repurposing the generic language grounding acquired by such models to teach task-relevant groundings to embodied agents.

</details>

---

## 28. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

- [x] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | https://icml.cc/virtual/2023/poster/25182

- **Link**: https://icml.cc/virtual/2023/poster/25182

- **Conference**: ICML

- **Year**: 2023

- **Type**: Efficient Adapter

<details>
<summary><strong>Abstract</strong></summary>

The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes BLIP-2, a generic and efficient pre-training strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pre-trained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various vision-language tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model's emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions.

</details>

---

## 29. Retrieval-Augmented Multimodal Language Modeling

- [x] Retrieval-Augmented Multimodal Language Modeling | https://icml.cc/virtual/2023/poster/25248

- **Link**: https://icml.cc/virtual/2023/poster/25248

- **Conference**: ICML

- **Year**: 2023

- **Type**: Memory Mechanism

<details>
<summary><strong>Abstract</strong></summary>

Recent multimodal models such as DALL-E and CM3 have achieved remarkable progress in text-to-image and image-to-text generation. However, these models store all their knowledge (e.g., the appearance of the Eiffel Tower) in the model parameters, requiring increasingly larger models and training data to capture more knowledge. To integrate knowledge in a more scalable and modular way, we propose a retrieval-augmented multimodal model, which enables a base multimodal model (generator) to refer to relevant text and images fetched by a retriever from external memory (e.g., documents on the web). Specifically, for the retriever, we use a pretrained CLIP, and for the generator, we train a CM3 Transformer on the LAION dataset. Our resulting model, named Retrieval-Augmented CM3 (RA-CM3), is the first multimodal model that can retrieve and generate both text and images. We show that RA-CM3 significantly outperforms baseline multimodal models such as DALL-E and CM3 on both image and caption generation tasks (12 FID and 17 CIDEr improvements on MS-COCO), while requiring much less compute for training (<30% of DALL-E). Moreover, we show that RA-CM3 exhibits novel capabilities such as faithful image generation and multimodal in-context learning (e.g., image generation from demonstrations).

</details>

---

## 30. Stable and low-precision training for large-scale vision-language models

- [x] Stable and low-precision training for large-scale vision-language models | https://neurips.cc/virtual/2023/poster/70245

- **Link**: https://neurips.cc/virtual/2023/poster/70245

- **Conference**: NeurIPS

- **Year**: 2023

- **Type**: Model Quantization

<details>
<summary><strong>Abstract</strong></summary>

We introduce new methods for 1) accelerating and 2) stabilizing training for large language-vision models. 1) For acceleration, we introduce SwitchBack, a linear layer for int8 quantized training which provides a speed-up of 13-25% while matching the performance of bfloat16 training within 0.1 percentage points for the 1B parameter CLIP ViT-Huge---the largest int8 training to date. Our main focus is int8 as GPU support for float8 is rare, though we also analyze float8 training through simulation. While SwitchBack proves effective for float8, we show that standard techniques are also successful if the network is trained and initialized so that large feature magnitudes are discouraged, which we accomplish via layer-scale initialized with zeros. 2) For stability, we analyze loss spikes and find they consistently occur 1-8 iterations after the squared gradients become under-estimated by their AdamW second moment estimator. As a result, we recommend an AdamW-Adafactor hybrid which avoids loss spikes when training a CLIP ViT-Huge model and outperforms gradient clipping at the scales we test.

</details>

---

## 31. Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models

- [x] Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models | https://neurips.cc/virtual/2023/poster/70716

- **Link**: https://neurips.cc/virtual/2023/poster/70716

- **Conference**: NeurIPS

- **Year**: 2023

- **Type**: Model Distillation

<details>
<summary><strong>Abstract</strong></summary>

We propose a conceptually simple and lightweight framework for improving the robustness of vision models through the combination of knowledge distillation and data augmentation. We address the conjecture that larger models do not make for better teachers by showing strong gains in out-of-distribution robustness when distilling from pretrained foundation models. Following this finding, we propose Discrete Adversarial Distillation (DAD), which leverages a robust teacher to generate adversarial examples and a VQGAN to discretize them, creating more informative samples than standard data augmentation techniques. We provide a theoretical framework for the use of a robust teacher in the knowledge distillation with data augmentation setting and demonstrate strong gains in out-of-distribution robustness and clean accuracy across different student architectures. Notably, our method adds minor computational overhead compared to similar techniques and can be easily combined with other data augmentations for further improvements.

</details>

---

## 32. Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models

- [x] Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models | https://neurips.cc/virtual/2023/poster/71562

- **Link**: https://neurips.cc/virtual/2023/poster/71562

- **Conference**: NeurIPS

- **Year**: 2023

- **Type**: Dynamic Skippping

<details>
<summary><strong>Abstract</strong></summary>

With ever increasing parameters and computation, vision-language pre-trained (VLP) models exhibit prohibitive expenditure in downstream task adaption. Recent endeavors mainly focus on parameter efficient transfer learning (PETL) for VLP models by only updating a small number of parameters. However, excessive computational overhead still plagues the application of VLPs. In this paper, we aim at parameter and computation efficient transfer learning (PCETL) for VLP models. In particular, PCETL not only needs to limit the number of trainable parameters in VLP models, but also to reduce the computational redundancy during inference, thus enabling a more efficient transfer. To approach this target, we propose a novel dynamic architecture skipping (DAS) approach towards effective PCETL. Instead of directly optimizing the intrinsic architectures of VLP models, DAS first observes the significances of their modules to downstream tasks via a reinforcement learning (RL) based process, and then skips the redundant ones with lightweight networks, i.e. adapters, according to the obtained rewards. In this case, the VLP model can well maintain the scale of trainable parameters while speeding up its inference on downstream tasks. To validate DAS, we apply it to two representative VLP models, namely ViLT and METER, and conduct extensive experiments on a bunch of VL tasks. The experimental results not only show the great advantages of DAS in reducing computational complexity, e.g. -11.97%  FLOPs of METER on VQA2.0, but also confirm its competitiveness against existing PETL methods in terms of parameter scale and performance. Our source code is given in our appendix.

</details>

---
