# AAAI 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_aaai2025_papers.csv

## 1. ChemVLM: Exploring the Power of Multimodal Large Language Models in Chemistry Area

- [ ] ChemVLM: Exploring the Power of Multimodal Large Language Models in Chemistry Area | https://ojs.aaai.org/index.php/AAAI/article/view/32020

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32020

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have achieved remarkable success and have been applied across various scientific fields, including chemistry. However, many chemical tasks require the processing of visual information, which cannot be successfully handled by existing chemical LLMs. This brings a growing need for models capable of integrating multimodal information in the chemical domain. In this paper, we introduce ChemVLM, an open-source chemical multimodal large language model specifically designed for chemical applications. ChemVLM is trained on a carefully curated bilingual multimodal dataset that enhances its ability to understand both textual and visual chemical information, including molecular structures, reactions, and chemistry examination questions. We develop three datasets for comprehensive evaluation, tailored to Chemical Optical Character Recognition (OCR), Multimodal Chemical Reasoning (MMCR), and Multimodal Molecule Understanding tasks. We benchmark ChemVLM against a range of open-source and proprietary multimodal large language models on various tasks. Experimental results demonstrate that ChemVLM achieves competitive performance across all evaluated tasks.

</details>

---

## 2. Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation

- [ ] Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation | https://ojs.aaai.org/index.php/AAAI/article/view/32078

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32078

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding and predicting the popularity of online User-Generated Content (UGC) is critical for various social and recommendation systems. Existing efforts have focused on extracting predictive features and using pre-trained deep models to learn and fuse multimodal UGC representations. However, the dissemination of social UGCs is not an isolated process in social network; rather, it is influenced by contextual relevant UGCs and various exogenous factors, including social ties, trends, user interests, and platform algorithms. In this work, we propose a retrieval-based framework to enhance the popularity prediction of multimodal UGCs. Our framework extends beyond a simple semantic retrieval, incorporating a meta retrieval strategy that queries a diverse set of relevant UGCs by considering multimodal content semantics, and metadata from user and post. Moreover, to eliminate irrelevant and noisy UGCs in retrieval, we introduce a new measure called Relative Retrieval Contribution to Prediction (RRCP), which selectively refines the retrieved UGCs. We then aggregate the contextual UGC knowledge using vision-language graph neural networks, and fuse them with an RRCP-Attention-based prediction network. Extensive experiments on three large-scale social media datasets demonstrate significant improvements ranging from 26.68% to 48.19% across all metrics compared to strong baselines.

</details>

---

## 3. ISR-DPO: Aligning Large Multimodal Models for Videos by Iterative Self-Retrospective DPO

- [ ] ISR-DPO: Aligning Large Multimodal Models for Videos by Iterative Self-Retrospective DPO | https://ojs.aaai.org/index.php/AAAI/article/view/32166

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32166

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Iterative self-improvement, a concept extending beyond personal growth, has found powerful applications in machine learning, particularly in transforming weak models into strong ones. While recent advances in natural language processing have shown its efficacy through iterative preference optimization, applying this approach to Video Large Multimodal Models (VLMMs) remains challenging due to modality misalignment. VLMMs struggle with this misalignment during iterative preference modeling, as the self-judge model often prioritizes linguistic knowledge over visual information.
Additionally, iterative preference optimization can lead to visually hallucinated verbose responses due to length bias within the self-rewarding cycle. To address these issues, we propose Iterative Self-Retrospective Direct Preference Optimization (ISR-DPO), a method that uses self-retrospection to enhance preference modeling. This approach enhances the self-judge’s focus on informative video regions, resulting in more visually grounded preferences. In extensive empirical evaluations across diverse video question answering benchmarks, the ISR-DPO significantly outperforms the state of the art. We are committed to open-sourcing our code, models, and datasets to encourage further investigation.

</details>

---

## 4. AGFSync: Leveraging AI-Generated Feedback for Preference Optimization in Text-to-Image Generation

- [ ] AGFSync: Leveraging AI-Generated Feedback for Preference Optimization in Text-to-Image Generation | https://ojs.aaai.org/index.php/AAAI/article/view/32168

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32168

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-Image (T2I) diffusion models have achieved remarkable success in image generation. Despite their progress, challenges remain in both prompt-following ability, image quality and lack of high-quality datasets, which are essential for refining these models. As acquiring labeled data is costly, we introduce AGFSync, a framework that enhances T2I diffusion models through Direct Preference Optimization (DPO) in a fully AI-driven approach. AGFSync utilizes Vision-Language Models (VLM) to assess image quality across style, coherence, and aesthetics, generating feedback data within an AI-driven loop. By applying AGFSync to leading T2I models such as SD v1.4, v1.5, and SDXL-base, our extensive experiments on the TIFA dataset demonstrate notable improvements in VQA scores, aesthetic evaluations, and performance on the HPS v2 benchmark, consistently outperforming the base models. AGFSync's method of refining T2I diffusion models paves the way for scalable alignment techniques.

</details>

---

## 5. HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models

- [ ] HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32171

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32171

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-resolution Vision-Language Models (VLMs) are widely used in multimodal tasks to enhance accuracy by preserving detailed image information. However, these models often generate an excessive number of visual tokens due to the need to encode multiple partitions of a high-resolution image input. Processing such a large number of visual tokens poses significant computational challenges, particularly for resource-constrained commodity GPUs. To address this challenge, we propose High-Resolution Early Dropping (HiRED), a plug-and-play token-dropping method designed to operate within a fixed token budget. HiRED leverages the attention of CLS token in the vision transformer (ViT) to assess the visual content of the image partitions and allocate an optimal token budget for each partition accordingly. The most informative visual tokens from each partition within the allocated budget are then selected and passed to the subsequent Large Language Model (LLM). We showed that HiRED achieves superior accuracy and performance, compared to existing token-dropping methods. Empirically, HiRED-20% (i.e., a 20% token budget) on LLaVA-Next-7B achieves a 4.7x increase in token generation throughput, reduces response latency by 78%, and saves 14% of GPU memory for single inference on an NVIDIA TESLA P40 (24 GB). For larger batch sizes (e.g., 4), HiRED-20% prevents out-of-memory errors by cutting memory usage by 30%, while preserving throughput and latency benefits.

</details>

---

## 6. Dynamic Adapter with Semantics Disentangling for Cross-lingual Cross-modal Retrieval

- [ ] Dynamic Adapter with Semantics Disentangling for Cross-lingual Cross-modal Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/32186

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32186

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing cross-modal retrieval methods typically rely on large-scale vision-language pair data. This makes it challenging to efficiently develop a cross-modal retrieval model for under-resourced languages of interest. Therefore, Cross-lingual Cross-modal Retrieval (CCR), which aims to align vision and the low-resource language (the target language) without using any human-labeled target-language data, has gained increasing attention. As a general parameter-efficient way, a common solution is to utilize adapter modules to transfer the vision-language alignment ability of Vision-Language Pretraining (VLP) models from a source language to a target language. However, these adapters are usually static once learned, making it difficult to adapt to target-language captions with varied expressions. To alleviate it, we propose Dynamic Adapter with Semantics Disentangling (DASD), whose parameters are dynamically generated conditioned on the characteristics of the input captions. Considering that the semantics and expression styles of the input caption largely influence how to encode it, we propose a semantic disentangling module to extract the semantic-related and semantic-agnostic features from the input, ensuring that generated adapters are well-suited to the characteristics of input caption. Extensive experiments on two image-text datasets and one video-text dataset demonstrate the effectiveness of our model for cross-lingual cross-modal retrieval, as well as its good compatibility with various VLP models.

</details>

---

## 7. ObjVariantEnsemble: Advancing Point Cloud LLM Evaluation in Challenging Scenes with Subtly Distinguished Objects

- [ ] ObjVariantEnsemble: Advancing Point Cloud LLM Evaluation in Challenging Scenes with Subtly Distinguished Objects | https://ojs.aaai.org/index.php/AAAI/article/view/32190

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32190

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D scene understanding is an important task, and there has been a recent surge of research interest in aligning 3D representations of point clouds with text to empower embodied AI. However, due to the lack of comprehensive 3D benchmarks, the capabilities of 3D models in real-world scenes, particularly those that are challenging with subtly distinguished objects, remain insufficiently investigated. To facilitate a more thorough evaluation of 3D models' capabilities, we propose a scheme, ObjVariantEnsemble, to systematically introduce more scenes with specified object classes, colors, shapes, quantities, and spatial relationships to meet model evaluation needs. More importantly, we intentionally construct scenes with similar objects to a certain degree and design an LLM-VLM-cooperated annotator to capture key distinctions as annotations. The resultant benchmark can better challenge 3D models, reveal their shortcomings in understanding, and potentially aid in the further development of 3D models.

</details>

---

## 8. Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos

- [ ] Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos | https://ojs.aaai.org/index.php/AAAI/article/view/32214

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32214

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper considers the problem of Multi-Hop Video Question Answering (MH-VidQA) in long-form egocentric videos. This task not only requires to answer visual questions, but also to localize multiple relevant time intervals within the video as visual evidences. We develop an automated pipeline to create multi-hop question-answering pairs with associated temporal evidence, enabling to construct a large-scale dataset for instruction-tuning. To monitor the progress of this new task, we further curate a high-quality benchmark, MULTIHOP-EGOQA, with careful manual verification and refinement. Experimental results reveal that existing multimodal systems exhibit inadequate multi-hop grounding and reasoning abilities, resulting in unsatisfactory performance. We then propose a novel architecture, termed as Grounding Scattered Evidence with Large Language Model (GeLM), that enhances multi-modal large language models by incorporating a grounding module to retrieve temporal evidence from videos using flexible grounding tokens. Trained on our visual instruction-tuning data, GeLM demonstrates improved multi-hop grounding and reasoning capabilities, setting a baseline for this new task. Furthermore, when trained on third-person view videos, the same architecture also achieves state-of-the-art performance on the single-hop VidQA benchmark, ActivityNet-RTL, demonstrating its effectiveness.

</details>

---

## 9. Attribution Analysis Meets Model Editing: Advancing Knowledge Correction in Vision Language Models with VisEdit

- [ ] Attribution Analysis Meets Model Editing: Advancing Knowledge Correction in Vision Language Models with VisEdit | https://ojs.aaai.org/index.php/AAAI/article/view/32215

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32215

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Model editing aims to correct outdated or erroneous knowledge in large models without costly retraining. Recent research discovered that the mid-layer representation of the subject's final token in a prompt has a strong influence on factual predictions, and developed Large Language Model (LLM) editing techniques based on this observation. However, for Vision-LLMs (VLLMs), how visual representations impact the predictions from a decoder-only language model remains largely unexplored. To the best of our knowledge, model editing for VLLMs has not been extensively studied in the literature. In this work, we employ the contribution allocation and noise perturbation methods to measure the contributions of visual representations for token predictions. Our attribution analysis shows that visual representations in mid-to-later layers that are highly relevant to the prompt contribute significantly to predictions. Based on these insights, we propose *VisEdit*, a novel model editor for VLLMs that effectively corrects knowledge by editing intermediate visual representations in regions important to the edit prompt. We evaluated *VisEdit* using multiple VLLM backbones and public VLLM editing benchmark datasets. The results show the superiority of *VisEdit* over the strong baselines adapted from existing state-of-the-art editors for LLMs.

</details>

---

## 10. Motion Prior Knowledge Learning with Homogeneous Language Descriptions for Moving Infrared Small Target Detection

- [ ] Motion Prior Knowledge Learning with Homogeneous Language Descriptions for Moving Infrared Small Target Detection | https://ojs.aaai.org/index.php/AAAI/article/view/32217

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32217

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Different from traditional object detection, pure vision is not enough to infrared small target detection, due to small target size and weak background contrast. For promoting detection performance, more target representations are needed. Currently, motion representations have been proved to be one of the most potential feature kinds for infrared small target detection. Existing methods have an obvious weakness, that besides vision features, they could only capture coarse motion representations from temporal domain. With vision features, fine motion representations could be more effective to enhance detection performance. To overcome this weakness, inspired by prevalent vision-language models, we propose the first vision-language framework with motion prior knowledge learning (MoPKL). Breaking through traditional pure-vision modality, it utilizes homogeneous language descriptions, formatted for moving targets, to directionally guide vision channel learning motion prior knowledge. With the facilitation of motion-vision alignment and motion-relation mining, the motion of infrared small targets is further refined by graph attention, to generate more fine motion representations. The extensive experiments on datasets ITSDT-15K and IRDST show that our framework is effective. It could often obviously outperform other methods.

</details>

---

## 11. H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving

- [ ] H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving | https://ojs.aaai.org/index.php/AAAI/article/view/32220

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32220

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the prevalence of Multimodal Large Language Models(MLLMs), autonomous driving has encountered new opportunities and challenges. 
In particular, multi-modal video understanding is critical to interactively analyze what will happen in the procedure of autonomous driving.
However, videos in such a dynamical scene that often contains complex spatial-temporal movements,
which restricts the generalization capacity of the existing MLLMs in this field.
To bridge the gap, we propose a novel Hierarchical Mamba Adaptation (H-MBA) framework to fit the complicated motion changes in autonomous driving videos.
Specifically, our H-MBA consists of two distinct modules,
including Context Mamba (C-Mamba) and Query Mamba (Q-Mamba).
First, C-Mamba contains various types of structure state space models,
which can effectively capture multi-granularity video context for different temporal resolution.
Second, Q-Mamba flexibly transforms the current frame as the learnable query, 
and attentively select multi-granularity video context into query.
Consequently, it can adaptively integrate all the video contexts of multi-scale temporal resolutions to enhance video understanding.
Via a plug-and-play paradigm in MLLMs,
our H-MBA shows the remarkable performance on multi-modal video tasks in autonomous driving,
e.g., for risk object detection, it outperforms the previous SOTA method with 5.5% mIoU improvement.

</details>

---

## 12. Comprehensive Multi-Modal Prototypes Are Simple and Effective Classifiers for Vast-Vocabulary Object Detection

- [ ] Comprehensive Multi-Modal Prototypes Are Simple and Effective Classifiers for Vast-Vocabulary Object Detection | https://ojs.aaai.org/index.php/AAAI/article/view/32232

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32232

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Enabling models to recognize vast open-world categories has been a longstanding pursuit in object detection. By leveraging the generalization capabilities of vision-language models, current open-world detectors can recognize a broader range of vocabularies, despite being trained on limited categories. However, when the scale of the category vocabularies during training expands to a real-world level, previous classifiers aligned with coarse class names significantly reduce the recognition performance of these detectors. In this paper, we introduce Prova, a multi-modal prototype classifier for vast-vocabulary object detection. Prova extracts comprehensive multi-modal prototypes as initialization of alignment classifiers to tackle the vast-vocabulary object recognition failure problem. On V3Det, this simple method greatly enhances the performance among one-stage, two-stage, and DETR-based detectors with only additional projection layers in both supervised and open-vocabulary settings. In particular, Prova improves Faster R-CNN, FCOS, and DINO by 3.3, 6.2, and 2.9 AP respectively in the supervised setting of V3Det. For the open-vocabulary setting, Prova achieves a new state-of-the-art performance with 32.8 base AP and 11.0 novel AP, which is of 2.6 and 4.3 gain over the previous methods.

</details>

---

## 13. LOMA: Language-assisted Semantic Occupancy Network via Triplane Mamba

- [ ] LOMA: Language-assisted Semantic Occupancy Network via Triplane Mamba | https://ojs.aaai.org/index.php/AAAI/article/view/32264

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32264

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-based 3D occupancy prediction has become a popular research task due to its versatility and affordability. Nowadays, conventional methods usually project the image-based vision features to 3D space and learn the geometric information through the attention mechanism, enabling the 3D semantic occupancy prediction. However, these works usually face two main challenges: 1) Limited geometric information. Due to the lack of geometric information in the image itself, it is challenging to directly predict 3D space information, especially in large-scale outdoor scenes. 2) Local restricted interaction. Due to the quadratic complexity of the attention mechanism, they often use modified local attention to fuse features, resulting in a restricted fusion. To address these problems, in this paper, we propose a language-assisted 3D semantic occupancy prediction network, named LOMA. In the proposed vision-language framework, we first introduce a VL-aware Scene Generator (VSG) module to generate the 3D language feature of the scene. By leveraging the vision-language model, this module provides implicit geometric knowledge and explicit semantic information from the language. Furthermore, we present a Tri-plane Fusion Mamba (TFM) block to efficiently fuse the 3D language feature and 3D vision feature. The proposed module not only fuses the two features with global modeling but also avoids too much computation costs. Experiments on the SemanticKITTI and SSCBench-KITTI360 datasets show that our algorithm achieves new state-of-the-art performances in both geometric and semantic completion tasks. Our code will be open soon.

</details>

---

## 14. VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering

- [ ] VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering | https://ojs.aaai.org/index.php/AAAI/article/view/32301

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32301

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Albeit progress has been made in Composed Image Retrieval (CIR), we empirically find that a certain percentage of failure retrieval results are not consistent with their relative captions. To address this issue, this work provides a Visual Question Answering (VQA) perspective to boost the performance of CIR. The resulting VQA4CIR is a post-processing approach and can be directly plugged into existing CIR methods. Given the top-C retrieved images by a CIR method, VQA4CIR aims to decrease the adverse effect of the failure retrieval results being inconsistent with the relative caption. To find the retrieved images inconsistent with the relative caption, we resort to the "QA generation → VQA" self-verification pipeline. For QA generation, we suggest fine-tuning LLM (e.g., LLaMA) to generate several pairs of questions and answers from each relative caption. We then fine-tune LVLM (e.g., LLaVA) to obtain the VQA model. By feeding the retrieved image and question to the VQA model, one can find the images inconsistent with relative caption when the answer by VQA is inconsistent with the answer in the QA pair. Consequently, the CIR performance can be boosted by modifying the ranks of inconsistently retrieved images. Experimental results show that our proposed method outperforms state-of-the-art CIR methods on the CIRR and Fashion-IQ datasets.

</details>

---

## 15. PoseLLaVA: Pose Centric Multimodal LLM for Fine-Grained 3D Pose Manipulation

- [ ] PoseLLaVA: Pose Centric Multimodal LLM for Fine-Grained 3D Pose Manipulation | https://ojs.aaai.org/index.php/AAAI/article/view/32302

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32302

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Manipulating human poses based on natural language is an emerging research field that has traditionally focused on coarse commands such as “walking” or “dancing.” However, fine-grained pose manipulation, like instructing “put both hands in front of the stomach,” remains underexplored. In this paper, we introduce PoseLLaVA, a pioneering model that integrates SMPL-based pose representations into the multimodal LLaVA framework. Through a novel pose encoder decoder mechanism, PoseLLaVA achieves precise alignment between pose, textual, and visual modalities, enabling detailed control over pose manipulation tasks. PoseLLaVA excels in three key tasks: pose estimation, generation, and adjustment, all driven by detailed language instructions. We further introduce a fine-grained pose adjustment dataset PosePart, where each sample contains an initial pose and a target pose, along with specific instructions for adjustments, mimicking the guidance a human instructor might provide. Extensive evaluations across these tasks demonstrate significant improvements over existing methods, including metrics such as MPJPE and PA-MPJPE, which measure SMPL reconstruction errors, and Recall rates, which assess feature alignment across modalities. Specifically, PoseLLaVA reduces MPJPE errors by more than 20% compared to state-of-the-art methods in pose adjustment and generation tasks. Additionally, we demonstrate the feasibility of combining PoseLLaVA with generative models, such as diffusion, for pose image editing, highlighting its potential applications in language-controlled pose manipulation.

</details>

---

## 16. AIM: Let Any Multimodal Large Language Models Embrace Efficient In-Context Learning

- [ ] AIM: Let Any Multimodal Large Language Models Embrace Efficient In-Context Learning | https://ojs.aaai.org/index.php/AAAI/article/view/32316

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32316

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In-context learning (ICL) advances Large Language Models (LLMs) exhibiting emergent ability on downstream tasks without updating billions of parameters. However, in the area of multimodal Large Language Models (MLLMs), two problems hinder the application of multimodal ICL: (1) Most primary MLLMs are only trained on single-image datasets, making them unable to read extra multimodal demonstrations. (2) With the demonstrations increasing, thousands of visual tokens highly challenge hardware and degrade ICL performance. During preliminary explorations, we discovered that the inner LLM focuses more on the linguistic modality within multimodal demonstrations during generation. Therefore, we propose a general and lightweight framework AIM to tackle the mentioned problems through Aggregating Image information of Multimodal demonstrations to the latent space of the corresponding textual labels. After aggregation, AIM substitutes each demonstration with generated fused virtual tokens whose length is reduced to the same as its texts. Except for shortening input length, AIM further upgrades MLLMs pre-trained on image-text pairs to support multimodal ICL, as images from demonstrations are disregarded. Furthermore, benefiting from aggregating different demonstrations independently, AIM configures Demonstration Bank (DB) to avoid repeated aggregation, which significantly boosts model efficiency. We build AIM upon QWen-VL and LLaVA-Next, and AIM is comprehensively evaluated on image caption, VQA, and hateful speech detection. Outstanding results reveal that AIM provides an efficient and effective solution in upgrading MLLMs for multimodal ICL.

</details>

---

## 17. TC-LLaVA: Rethinking the Transfer of LLava from Image to Video Understanding with Temporal Considerations

- [ ] TC-LLaVA: Rethinking the Transfer of LLava from Image to Video Understanding with Temporal Considerations | https://ojs.aaai.org/index.php/AAAI/article/view/32317

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32317

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have significantly improved performance across various image-language applications. Recently, there has been a growing interest in adapting image pre-trained MLLMs for video-related tasks. However, most efforts concentrate on enhancing the vision encoder and projector components, while the core part, Large Language Models (LLMs), remains comparatively under-explored. In this paper, we propose two strategies to enhance the model's capability in video understanding tasks by improving inter-layer attention computation in LLMs. Specifically, the first approach focuses on the enhancement of Rotary Position Embedding (RoPE) with Temporal-Aware Dual RoPE, which introduces temporal position information to strengthen the MLLM's temporal modeling capabilities while preserving the relative position relationships of both visual and text tokens. The second approach involves enhancing the Attention Mask with the Frame-wise Block Causal Attention Mask, a simple yet effective method that broadens visual token interactions within and across video frames while maintaining the causal inference mechanism. Based on these proposed methods, we adapt LLaVA for video understanding tasks, naming it Temporal-Considered LLaVA (TC-LLaVA). Our TC-LLaVA achieves new state-of-the-art performance across various video understanding benchmarks with only supervised fine-tuning (SFT) on video-related datasets.

</details>

---

## 18. Queryable Prototype Multiple Instance Learning with Vision-Language Models for Incremental Whole Slide Image Classification

- [ ] Queryable Prototype Multiple Instance Learning with Vision-Language Models for Incremental Whole Slide Image Classification | https://ojs.aaai.org/index.php/AAAI/article/view/32325

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32325

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Whole Slide Image (WSI) classification has very significant applications in clinical pathology, e.g., tumor identification and cancer diagnosis. Currently, most research attention is focused on Multiple Instance Learning (MIL) using static datasets. One of the most obvious weaknesses of these methods is that they cannot efficiently preserve and utilize previously learned knowledge. With any new data arriving, classification models are required to be re-trained on both previous and current new data. To overcome this shortcoming and break through traditional vision modality, this paper proposes the first Vision-Language-based framework with Queryable Prototype Multiple Instance Learning (QPMIL-VL) specially designed for incremental WSI classification. This framework mainly consists of two information processing branches: one is for generating bag-level features by prototype-guided aggregation of instance features, while the other is for enhancing class features through a combination of class ensemble, tunable vector and class similarity loss. The experiments on four public WSI datasets demonstrate that our QPMIL-VL framework is effective for incremental WSI classification and often significantly outperforms other compared methods, achieving state-of-the-art (SOTA) performance.

</details>

---

## 19. LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies

- [ ] LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies | https://ojs.aaai.org/index.php/AAAI/article/view/32342

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32342

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating Natural Language Explanations (NLEs) for model predictions on medical images, particularly those depicting thoracic pathologies, remains a critical and challenging task. Existing methodologies often struggle due to general models' insufficient domain-specific medical knowledge and privacy concerns associated with retrieval-based augmentation techniques. To address these issues, we propose a novel Vision-Language framework augmented with a Knowledge Graph (KG)-based datastore, which enhances the model's understanding by incorporating additional domain-specific medical knowledge essential for generating accurate and informative NLEs. Our framework employs a KG-based retrieval mechanism that not only improves the precision of the generated explanations but also preserves data privacy by avoiding direct data retrieval. The KG datastore is designed as a plug-and-play module, allowing for seamless integration with various model architectures. We introduce and evaluate three distinct frameworks within this paradigm: KG-LLaVA, which integrates the pre-trained LLaVA model with KG-RAG; Med-XPT, a custom framework combining MedCLIP, a transformer-based projector, and GPT-2; and Bio-LLaVA, which adapts LLaVA by incorporating the Bio-ViT-L vision model. These frameworks are validated on the MIMIC-NLE dataset, where they achieve state-of-the-art results, underscoring the effectiveness of KG augmentation in generating high-quality NLEs for thoracic pathologies.

</details>

---

## 20. DME-Driver: Integrating Human Decision Logic and 3D Scene Perception in Autonomous Driving

- [ ] DME-Driver: Integrating Human Decision Logic and 3D Scene Perception in Autonomous Driving | https://ojs.aaai.org/index.php/AAAI/article/view/32346

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32346

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

There are two crucial aspects of reliable autonomous driving systems: the reasoning behind decision-making and the precision of environmental perception. This paper introduces DME-Driver, a new autonomous driving system that enhances performance and robustness by fully leveraging the two crucial aspects. This system comprises two main models. The first, the Decision Maker, is responsible for providing logical driving instructions. The second, the Executor, receives these instructions and generates precise control signals for the vehicles. To ensure explainable and reliable driving decisions, we build the Decision-Maker based on a large vision language model. This model follows the logic employed by experienced human drivers and simulates making decisions in a safe and reasonable manner. On the other hand, the generation of accurate control signals relies on precise and detailed environmental perception, where 3D scene perception models excel. Therefore, a planning-oriented perception model is employed as the Executor. It translates the logical decisions made by the Decision-Maker into accurate control signals for the self-driving cars. To effectively train the proposed system, a new dataset named Human-driver Behavior and Decision-making (HBD) dataset has been collected. This dataset encompasses a diverse range of human driver behaviors and their underlying motivations. By leveraging this dataset, our system achieves high-precision planning accuracy through a logical thinking process.

</details>

---

## 21. V2Xum-LLM: Cross-Modal Video Summarization with Temporal Prompt Instruction Tuning

- [ ] V2Xum-LLM: Cross-Modal Video Summarization with Temporal Prompt Instruction Tuning | https://ojs.aaai.org/index.php/AAAI/article/view/32374

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32374

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video summarization aims to create short, accurate, and cohesive summaries of longer videos. Despite the existence of various video summarization datasets, a notable limitation is their limited amount of source videos, which hampers the effective training of advanced large vision-language models (VLMs). Additionally, most existing datasets are created for video-to-video summarization, overlooking the contemporary need for multimodal video content summarization. Recent efforts have been made to expand from unimodal to multimodal video summarization, categorizing the task into three sub-tasks based on the summary's modality: video-to-video (V2V), video-to-text (V2T), and a combination of video and text summarization (V2VT). However, the textual summaries in previous multimodal datasets are inadequate. 
To address these issues, we introduce Instruct-V2Xum, a cross-modal video summarization dataset featuring 30,000 diverse videos sourced from YouTube, with lengths ranging from 40 to 940 seconds and an average summarization ratio of 16.39%. Each video summary in Instruct-V2Xum is paired with a textual summary that references specific frame indexes, facilitating the generation of aligned video and textual summaries.
In addition, we propose a new video summarization framework named V2Xum-LLM. V2Xum-LLM, specifically V2Xum-LLaMA in this study, is the first framework that unifies different video summarization tasks into one large language model's (LLM) text decoder and achieves task-controllable video summarization with temporal prompts and task instructions. Experiments show that V2Xum-LLaMA outperforms strong baseline models on multiple video summarization tasks. 
Furthermore, we propose an enhanced evaluation metric for V2V and V2VT summarization tasks.

</details>

---

## 22. EvoChart: A Benchmark and a Self-Training Approach Towards Real-World Chart Understanding

- [ ] EvoChart: A Benchmark and a Self-Training Approach Towards Real-World Chart Understanding | https://ojs.aaai.org/index.php/AAAI/article/view/32383

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32383

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chart understanding enables automated data analysis for humans, which requires models to achieve highly accurate visual comprehension. While existing Visual Language Models (VLMs) have shown progress in chart understanding, the lack of high-quality training data and comprehensive evaluation benchmarks hinders VLM chart comprehension. In this paper, we introduce EvoChart, a novel self-training method for generating synthetic chart data to enhance VLMs' capabilities in real-world chart comprehension. We also propose EvoChart-QA, a noval benchmark for measuring models' chart comprehension abilities in real-world scenarios. Specifically, EvoChart is a unique self-training data synthesis approach that simultaneously produces high-quality training corpus and a high-performance chart understanding model. EvoChart-QA consists of 650 distinct real-world charts collected from 140 different websites and 1,250 expert-curated questions that focus on chart understanding. Experimental results on various open-source and proprietary VLMs tested on EvoChart-QA demonstrate that even the best proprietary model, GPT-4o, achieves only 49.8% accuracy. Moreover, the EvoChart method significantly boosts the performance of open-source VLMs on real-world chart understanding tasks, achieving 54.2% accuracy on EvoChart-QA.

</details>

---

## 23. SLIP: Spoof-Aware One-Class Face Anti-Spoofing with Language Image Pretraining

- [ ] SLIP: Spoof-Aware One-Class Face Anti-Spoofing with Language Image Pretraining | https://ojs.aaai.org/index.php/AAAI/article/view/32385

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32385

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Face anti-spoofing (FAS) plays a pivotal role in ensuring the security and reliability of face recognition systems. With advancements in vision-language pretrained (VLP) models, recent two-class FAS techniques have leveraged the advantages of using VLP guidance, while this potential remains unexplored in one-class FAS methods. The one-class FAS focuses on learning intrinsic liveness features solely from live training images to differentiate between live and spoof faces. However, the lack of spoof training data can lead one-class FAS models to inadvertently incorporate domain information irrelevant to the live/spoof distinction (\eg, facial content), causing performance degradation when tested with a new application domain. To address this issue, we propose a novel framework called Spoof-aware one-class face anti-spoofing with Language Image Pretraining (SLIP). Given that live faces should ideally not be obscured by any spoof-attack-related objects (\eg, paper, or masks) and are assumed to yield zero spoof cue maps, we first propose an effective language-guided spoof cue map estimation to enhance one-class FAS models by simulating whether the underlying faces are covered by attack-related objects and generating corresponding nonzero spoof cue maps. Next, we introduce a novel prompt-driven liveness feature disentanglement to alleviate live/spoof-irrelative domain variations by disentangling live/spoof-relevant and domain-dependent information. Finally, we design an effective augmentation strategy by fusing latent features from live images and spoof prompts to generate spoof-like image features and thus diversify latent spoof features to facilitate the learning of one-class FAS. Our extensive experiments and ablation studies support that SLIP consistently outperforms previous one-class FAS methods.

</details>

---

## 24. ZoRI: Towards Discriminative Zero-Shot Remote Sensing Instance Segmentation

- [ ] ZoRI: Towards Discriminative Zero-Shot Remote Sensing Instance Segmentation | https://ojs.aaai.org/index.php/AAAI/article/view/32388

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32388

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Instance segmentation algorithms in remote sensing are typically based on conventional methods, limiting their application to seen scenarios and closed-set predictions. In this work, we propose a novel task called zero-shot remote sensing instance segmentation, aimed at identifying aerial objects that are absent from training data. Challenges arise when classifying aerial categories with high inter-class similarity and intra-class variance. Besides, the domain gap between vision-language models’ pretraining datasets and remote sensing datasets hinders the zero-shot capabilities of the pretrained model when it is directly applied to remote sensing images. To address these challenges, we propose a Zero-Shot Remote Sensing Instance Segmentation framework, dubbed ZoRI. Our approach features a discrimination-enhanced classifier that uses refined textual embeddings to increase the awareness of class disparities. Instead of direct fine-tuning, we propose a knowledge-maintained adaptation strategy that decouples semantic-related information to preserve vision-language alignment while adjusting features to capture remote sensing domain-specific visual cues. Additionally, we introduce a prior-injected prediction with cache bank of aerial visual prototypes to supplement the semantic richness of text embeddings and seamlessly integrate aerial representations, adapting to the remote sensing domain. We establish new experimental protocols and benchmarks, and extensive experiments demonstrate that ZoRI achieves the state-of-art performance on the zero-shot remote sensing instance segmentation task.

</details>

---

## 25. Towards a Multimodal Large Language Model with Pixel-Level Insight for Biomedicine

- [ ] Towards a Multimodal Large Language Model with Pixel-Level Insight for Biomedicine | https://ojs.aaai.org/index.php/AAAI/article/view/32394

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32394

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, Multimodal Large Language Models (MLLM) have achieved notable advancements, demonstrating the feasibility of developing an intelligent biomedical assistant. However, current biomedical MLLMs predominantly focus on image-level understanding and restrict interactions to textual commands, thus limiting their capability boundaries and the flexibility of usage. In this paper, we introduce a novel end-to-end multimodal large language model for the biomedical domain, named MedPLIB, which possesses pixel-level understanding. Excitingly, it supports visual question answering (VQA), arbitrary pixel-level prompts (points, bounding boxes, and free-form shapes), and pixel-level grounding. We propose a novel Mixture-of-Experts (MoE) multi-stage training strategy, which divides MoE into separate training phases for a visual-language expert model and a pixel-grounding expert model, followed by fine-tuning using MoE. This strategy effectively coordinates multitask learning while maintaining the computational cost at inference equivalent to that of a single expert model. To advance the research of biomedical MLLMs, we introduce the Medical Complex Vision Question Answering Dataset (MeCoVQA), which comprises an array of 8 modalities for complex medical imaging question answering and image region understanding. Experimental results indicate that MedPLIB has achieved state-of-the-art outcomes across multiple medical visual language tasks. More importantly, in zero-shot evaluations for the pixel grounding task, MedPLIB leads the best small and large models by margins of 19.7 and 15.6 respectively on the mDice metric.

</details>

---

## 26. Medical MLLM Is Vulnerable: Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models

- [ ] Medical MLLM Is Vulnerable: Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32396

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32396

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Security concerns related to Large Language Models (LLMs) have been extensively explored; however, the safety implications for Multimodal Large Language Models (MLLMs), particularly in medical contexts (MedMLLMs), remain inadequately addressed. This paper investigates the security vulnerabilities of MedMLLMs, focusing on their deployment in clinical environments where the accuracy and relevance of question-and-answer interactions are crucial for addressing complex medical challenges. We introduce and redefine two attack types: mismatched malicious attack (2M-attack) and optimized mismatched malicious attack (O2M-attack), by integrating existing clinical data with atypical natural phenomena. Using the comprehensive 3MAD dataset that we developed, which spans a diverse range of medical imaging modalities and adverse medical scenarios, we performed an in-depth analysis and proposed the MCM optimization method. This approach significantly improves the attack success rate against MedMLLMs. Our evaluations, which include white-box attacks on LLaVA-Med and transfer (black-box) attacks on four other SOTA models, reveal that even MedMLLMs designed with advanced security mechanisms remain vulnerable to breaches. This study highlights the critical need for robust security measures to enhance the safety and reliability of open-source MedMLLMs, especially in light of the potential impact of jailbreak attacks and other malicious exploits in clinical applications. Warning: Medical jailbreaking may generate content that includes unverified diagnoses and treatment recommendations. Always consult professional medical advice.

</details>

---

## 27. What Kind of Visual Tokens Do We Need? Training-Free Visual Token Pruning for Multi-Modal Large Language Models from the Perspective of Graph

- [ ] What Kind of Visual Tokens Do We Need? Training-Free Visual Token Pruning for Multi-Modal Large Language Models from the Perspective of Graph | https://ojs.aaai.org/index.php/AAAI/article/view/32427

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32427

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent Multimodal Large Language Models(MLLMs) often use a large number of visual tokens to compensate their visual shortcoming, leading to excessive computation and obvious visual redundancy. In this paper, we investigate what kind of visual tokens are needed for MLLMs, and reveal that both foreground and background tokens are critical for MLLMs given the varying difficulties of examples. Based on this observation, we propose a graph-based method towards training-free visual token pruning, termed G-Prune. In particular, G-Prune regards visual tokens as nodes, and construct their connections based on their semantic similarities. Afterwards, the information flow is propagated via weighted links, and the most important tokens after iterations are kept for MLLMs, which can be front or background. To validate G-Prune, we apply it to a recent MLLM called LLaVA-NeXT, and conduct extensive experiments on a set of benchmarks. The experiment results show that G-Prune can greatly reduce computation overhead while retaining high performance on both coarse- and fine-grained tasks. For instance, G-Prune can reduce 63.57% FLOPs of LLaVA-NeXT on VQA2.0 and TextVQA with only 0.95% and 2.34% accuracy drops, respectively.

</details>

---

## 28. LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction

- [ ] LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction | https://ojs.aaai.org/index.php/AAAI/article/view/32433

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32433

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Logical image understanding involves interpreting and reasoning about the relationships and consistency within an image's visual content. This capability is essential in applications such as industrial inspection, where logical anomaly detection is critical for maintaining high-quality standards and minimizing costly recalls. Previous research in anomaly detection (AD) has relied on prior knowledge for designing algorithms, which often requires extensive manual annotations, significant computing power, and large amounts of data for training. Autoregressive, multimodal Vision Language Models (AVLMs) offer a promising alternative due to their exceptional performance in visual reasoning across various domains. Despite this, their application to logical AD remains unexplored. In this work, we investigate using AVLMs for logical AD and demonstrate that they are well-suited to the task. Combining AVLMs with format embedding and a logic reasoner, we achieve SOTA performance on public benchmarks, MVTec LOCO AD, with an AUROC of 86.0% and an F1-max of 83.7% along with explanations of the anomalies. This significantly outperforms the existing SOTA method by 18.1% in AUROC and 4.6% in F1-max score.

</details>

---

## 29. NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization

- [ ] NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization | https://ojs.aaai.org/index.php/AAAI/article/view/32439

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32439

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Compositional generalization is crucial for artificial intelligence agents to solve complex vision-language reasoning tasks. Neuro-symbolic approaches have demonstrated promise in capturing compositional structures, but they face critical challenges: (a) reliance on predefined predicates for symbolic representations that limit adaptability, (b) difficulty in extracting predicates from raw data, and (c) using non-differentiable operations for combining primitive concepts. To address these issues, we propose NeSyCoCo, a neuro-symbolic framework that leverages large language models (LLMs) to generate symbolic representations and map them to differentiable neural computations. NeSyCoCo introduces three innovations: (a) augmenting natural language inputs with dependency structures to enhance the alignment with symbolic representations, (b) employing distributed word representations to link diverse, linguistically motivated logical predicates to neural modules, and (c) using the soft composition of normalized predicate scores to align symbolic and differentiable reasoning. Our framework achieves state-of-the-art results on the ReaSCAN and CLEVR-CoGenT compositional generalization benchmarks and demonstrates robust performance with novel concepts in the CLEVR-SYN benchmark.

</details>

---

## 30. Learning to Prompt with Text Only Supervision for Vision-Language Models

- [ ] Learning to Prompt with Text Only Supervision for Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32444

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32444

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Foundational vision-language models like CLIP are emerging as a promising paradigm in vision due to their excellent generalization. However, adapting these models for downstream tasks while maintaining their generalization remains challenging. In literature, one branch of methods adapts CLIP by learning prompts using images. While effective, these methods often rely on image-label data, which is not always practical, and struggle to generalize to new datasets due to overfitting on few-shot source data. Another approach explores training-free methods by generating class captions from large language models (LLMs) and performing prompt ensembling, but these methods often produce static, class-specific prompts that cannot be transferred to new classes and incur additional costs by generating LLM descriptions for each class separately.
In this work, we aim to combine the strengths of both approaches by learning prompts using only text data derived from LLMs. As supervised training of prompts in the image-free setup is non-trivial, we develop a language-only efficient training approach that enables prompts to distill rich contextual knowledge from LLM data. Furthermore, by mapping the LLM contextual text data within the learned prompts, our approach enables zero-shot transfer of prompts to new classes and datasets, potentially reducing the LLM prompt engineering cost. To the best of our knowledge, this is the first work that learns generalized and transferable prompts for image tasks using only text data. We perform evaluations on 4 benchmarks, where ProText improves over ensembling methods while being competitive with those using labeled images.

</details>

---

## 31. COLUMBUS: Evaluating COgnitive Lateral Understanding Through Multiple-Choice reBUSes

- [ ] COLUMBUS: Evaluating COgnitive Lateral Understanding Through Multiple-Choice reBUSes | https://ojs.aaai.org/index.php/AAAI/article/view/32464

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32464

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While visual question-answering (VQA) benchmarks have catalyzed the development of reasoning techniques, they have focused on vertical thinking. Effective problem-solving also necessitates lateral thinking, which remains understudied in AI and has not been used to test visual perception systems. To bridge this gap, we formulate visual lateral thinking as a multiple-choice question-answering task and describe a three-step taxonomy-driven methodology for instantiating task examples. Then, we develop COLUMBUS, a synthetic benchmark that applies the task pipeline to create QA sets with text and icon rebus puzzles based on publicly available collections of compounds and common phrases. COLUMBUS comprises over 1,000 puzzles, each with four answer candidates. While the SotA vision language models (VLMs) achieve decent performance, our evaluation demonstrates a substantial gap between humans and models. VLMs benefit from human-curated descriptions but struggle to self-generate such representations at the right level of abstraction.

</details>

---

## 32. Mind the Uncertainty in Human Disagreement: Evaluating Discrepancies Between Model Predictions and Human Responses in VQA

- [ ] Mind the Uncertainty in Human Disagreement: Evaluating Discrepancies Between Model Predictions and Human Responses in VQA | https://ojs.aaai.org/index.php/AAAI/article/view/32468

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32468

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models struggle to accurately predict responses provided by multiple human annotators, particularly when those responses exhibit high uncertainty. In this study, we focus on a Visual Question Answering (VQA) task and comprehensively evaluate how well the output of the state-of-the-art vision-language model correlates with the distribution of human responses. To do so, we categorize our samples based on their levels (low, medium, high) of human uncertainty in disagreement (HUD) and employ, not only accuracy, but also three new human-correlated metrics for the first time in VQA, to investigate the impact of HUD. We also verify the effect of common calibration and human calibration (Baan et al. 2022) on the alignment of models and humans. Our results show that even BEiT3, currently the best model for this task, struggles to capture the multi-label distribution inherent in diverse human responses. Additionally, we observe that the commonly used accuracy-oriented calibration technique adversely affects BEiT3’s ability to capture HUD, further widening the gap between model predictions and human distributions. In contrast, we show the benefits of calibrating models towards human distributions for VQA, to better align model confidence with human uncertainty. Our findings highlight that for VQA, the alignment between human responses and model predictions is understudied and is an important target for future studies.

</details>

---

## 33. Progressive Multi-granular Alignments for Grounded Reasoning in Large Vision-Language Models

- [ ] Progressive Multi-granular Alignments for Grounded Reasoning in Large Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32471

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32471

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing Large Vision-Language Models (LVLMs) excel at matching concepts across multi-modal inputs but struggle with compositional concepts and high-level relationships between entities. This paper introduces Progressive multi-granular Vision-Language alignments (PromViL), a novel framework to enhance LVLMs' ability in performing grounded compositional visual reasoning tasks. Our approach constructs a hierarchical structure of multi-modal alignments, ranging from simple to complex concepts. By progressively aligning textual descriptions with corresponding visual regions, our model learns to leverage contextual information from lower levels to inform higher-level reasoning. To facilitate this learning process, we introduce a data generation process that creates a novel dataset derived from Visual Genome, providing a wide range of nested compositional vision-language pairs. Experimental results demonstrate that our PromViL framework significantly outperforms baselines on various visual grounding and compositional question answering tasks.

</details>

---

## 34. VEGAS: Towards Visually Explainable and Grounded Artificial Social Intelligence

- [ ] VEGAS: Towards Visually Explainable and Grounded Artificial Social Intelligence | https://ojs.aaai.org/index.php/AAAI/article/view/32497

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32497

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Social Intelligence Queries (Social-IQ) serve as the primary multimodal benchmark for evaluating a  model’s social intelligence level. While impressive multiple-choice question (MCQ) accuracy is  achieved by current solutions, increasing evidence shows that they are largely, and in some cases entirely, dependent on language modality, overlooking visual context. Additionally, the closed-set nature further prevents the exploration of whether and to what extent the reasoning path behind selection is correct. To address these limitations, we propose the Visually Explainable and Grounded Artificial Social Intelligence (VEGAS) model. As a generative multimodal model, VEGAS leverages open-ended answering to provide explainable responses, which enhances the clarity and evaluation of reasoning paths. To enable visually grounded answering, we propose a novel sampling strategy to provide the model with more relevant visual frames. We then enhance the model’s interpretation of these frames through Generalist Instruction Fine-Tuning (GIFT), which aims to: i) learn multimodal language transformations for fundamental emotional social traits, and ii) establish multimodal joint reasoning capabilities. Extensive experiments, comprising modality ablation, open-ended assessments, and supervised MCQ evaluations, consistently show that VEGAS effectively utilizes visual information in reasoning to produce correct and also credible answers. We expect this work to offer a new perspective on Social-IQ and advance the development of human-like social AI.

</details>

---

## 35. Multimodal Hypothetical Summary for Retrieval-based Multi-image Question Answering

- [ ] Multimodal Hypothetical Summary for Retrieval-based Multi-image Question Answering | https://ojs.aaai.org/index.php/AAAI/article/view/32513

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32513

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-based multi-image question answering (QA) task involves retrieving multiple question-related images and synthesizing these images to generate an answer. Conventional "retrieve-then-answer" pipelines often suffer from cascading errors because the training objective of QA fails to optimize the retrieval stage. To address this issue, we propose a novel method to effectively introduce and reference retrieved information into the QA. Given the image set to be retrieved, we employ a multimodal large language model (visual perspective) and a large language model (textual perspective) to obtain multimodal hypothetical summary in question-form and description-form. By combining visual and textual perspectives, MHyS captures image content more specifically and replaces real images in retrieval, which eliminates the modality gap by transforming into text-to-text retrieval and helps improve retrieval. To more advantageously introduce retrieval with QA, we employ contrastive learning to align queries (questions) with MHyS. Moreover, we propose a coarse-to-fine strategy for calculating both sentence-level and word-level similarity scores, to further enhance retrieval and filter out irrelevant details. Our approach achieves a 3.7% absolute improvement over state-of-the-art methods on RETVQA and a 14.5% improvement over CLIP.  Comprehensive experiments and detailed ablation studies demonstrate the superiority of our method.

</details>

---

## 36. DigitalLLaVA: Incorporating Digital Cognition Capability for Physical World Comprehension in Multimodal LLMs

- [ ] DigitalLLaVA: Incorporating Digital Cognition Capability for Physical World Comprehension in Multimodal LLMs | https://ojs.aaai.org/index.php/AAAI/article/view/32522

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32522

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown remarkable cognitive capabilities in various cross-modal tasks.However, existing MLLMs struggle with tasks that require physical digital cognition, such as accurately reading an electric meter or pressure gauge. This limitation significantly reduces their effectiveness in practical applications like industrial monitoring and home energy management, where digital sensors are not feasible. For humans, physical digits are artificially defined quantities presented on specific carriers, which require training to recognize. As existing MLLMs are only pre-trained in the manner of object recognition, they fail to comprehend the relationship between digital carriers and their reading. To this end, referring to human behavior, we propose a novel DigitalLLaVA method to explicitly inject digital cognitive abilities into MLLMs in a two-step manner. In the first step, to improve the MLLM's understanding of physical digit carriers, we propose a digit carrier mapping method. This step utilizes object-level text-image pairs to enhance the model's comprehension of objects containing physical digits. For the second step, unlike previous methods that rely on sequential digital prediction or digit regression, we propose a 32 bit floating point simulation approach that treats digit prediction as a whole. Using digit-level text-image pairs, we train three float heads to predict 32-bit floating-point numbers using 0/1 binary classification. This step significantly reduces the search space, making the prediction process more robust and straightforward. Being simple but effective, our method can identify very precise metrics (i.e., accurate to ±0.001) and provide floating-point results, showing its applicability in digital carrier domains.

</details>

---

## 37. Generative Planning with 3D-Vision Language Pre-training for End-to-End Autonomous Driving

- [ ] Generative Planning with 3D-Vision Language Pre-training for End-to-End Autonomous Driving | https://ojs.aaai.org/index.php/AAAI/article/view/32524

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32524

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autonomous driving is a challenging task that requires perceiving and understanding the surrounding environment for safe trajectory planning. While existing vision-based end-to-end models have achieved promising results, these methods are still facing the challenges of vision understanding, decision reasoning and scene generalization. To solve these issues, a generative planning with 3D-vision language pre-training model named GPVL is proposed for end-to-end autonomous driving. The proposed paradigm has two significant aspects. On one hand, a 3D-vision language pre-training module is designed to bridge the gap between visual perception and linguistic understanding in the bird's eye view. On the other hand, a cross-modal language model is introduced to generate reasonable planning with perception and navigation information in an auto-regressive manner. Experiments on the challenging nuScenes dataset demonstrate that the proposed scheme achieves excellent performances compared with state-of-the-art methods. Besides, the proposed GPVL presents strong generalization ability and real-time potential when handling high-level commands in various scenarios. It is believed that the effective, robust and efficient performance of GPVL is crucial for the practical application of future autonomous driving systems.

</details>

---

## 38. Exploring the Potential of Large Vision-Language Models for Unsupervised Text-Based Person Retrieval

- [ ] Exploring the Potential of Large Vision-Language Models for Unsupervised Text-Based Person Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/32543

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32543

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The aim of text-based person retrieval is to identify pedestrians using natural language descriptions within a large-scale image gallery. Traditional methods rely heavily on manually annotated image-text pairs, which are resource-intensive to obtain. With the emergence of Large Vision-Language Models (LVLMs), the advanced capabilities of contemporary models in image understanding have led to the generation of highly accurate captions. Therefore, this paper explores the potential of employing Large Vision-Language Models for unsupervised text-based pedestrian image retrieval and proposes a Multi-grained Uncertainty Modeling and Alignment framework (MUMA). Initially, multiple Large Vision-Language Models are employed to generate diverse and hierarchically structured pedestrian descriptions across different styles and granularities. However, the generated captions inevitably introduce noise. To address this issue, an uncertainty-guided sample filtration module is proposed to estimate and filter out unreliable image-text pairs. Additionally, to simulate the diversity of styles and granularities in captions, a multi-grained uncertainty modeling approach is applied to model the distributions of captions, with each caption represented as a multivariate Gaussian distribution. Finally, a multi-level consistency distillation loss is employed to integrate and align the multi-grained captions, aiming to transfer knowledge across different granularities. Experimental evaluations conducted on three widely-used datasets demonstrate the significant advancements achieved by our approach.

</details>

---

## 39. Standing on the Shoulders of Giants: Reprogramming Visual-Language Model for General Deepfake Detection

- [ ] Standing on the Shoulders of Giants: Reprogramming Visual-Language Model for General Deepfake Detection | https://ojs.aaai.org/index.php/AAAI/article/view/32559

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32559

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The proliferation of deepfake faces poses huge potential negative impacts on our daily lives. Despite substantial advancements in deepfake detection over these years, the generalizability of existing methods against forgeries from unseen datasets or created by emerging generative models remains constrained. In this paper, inspired by the zero-shot advantages of Vision-Language Models (VLMs), we propose a novel approach that repurposes a well-trained VLM for general deepfake detection. Motivated by the model reprogramming paradigm that manipulates the model prediction via input perturbations, our method can reprogram a pre-trained VLM model (e.g., CLIP) solely based on manipulating its input without tuning the inner parameters. First, learnable visual perturbations are used to refine feature extraction for deepfake detection. Then, we exploit information of face embedding to create sample-level adaptative text prompts, improving the performance. Extensive experiments on several popular benchmark datasets demonstrate that (1) the cross dataset and cross-manipulation performances of deepfake detection can be significantly and consistently improved (e.g., over 88% AUC in cross-dataset setting from FF++ to Wild-Deepfake); (2) the superior performances are achieved with fewer trainable parameters, making it a promising approach for real-world applications.

</details>

---

## 40. Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference

- [ ] Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference | https://ojs.aaai.org/index.php/AAAI/article/view/32567

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32567

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) demand considerable computations for inference due to the extensive parameters and the additional input tokens needed for visual information representation. Herein, we introduce Visual Tokens Withdrawal (VTW), a plug-and-play module to boost MLLMs for rapid inference. Our approach is inspired by two intriguing phenomena we have observed: (1) the attention sink phenomenon that is prevalent in LLMs also persists in MLLMs, suggesting that initial tokens and nearest tokens receive the majority of attention, while middle vision tokens garner minimal attention in deep layers; (2) the presence of information migration, which implies that visual information is transferred to subsequent text tokens within the first few layers of MLLMs. As per our findings, we conclude that vision tokens are unnecessary in the deep layers of MLLMs. Thus, we strategically withdraw them at a certain layer, enabling only text tokens to engage in subsequent layers. To pinpoint the ideal layer for VTW, we initially analyze a limited set of tiny datasets and choose the first layer that meets the Kullback-Leibler divergence criterion. Our VTW approach can cut computational overhead by over 40% across diverse multimodal tasks while maintaining performance.

</details>

---

## 41. Making Large Vision Language Models to Be Good Few-Shot Learners

- [ ] Making Large Vision Language Models to Be Good Few-Shot Learners | https://ojs.aaai.org/index.php/AAAI/article/view/32576

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32576

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Few-shot classification (FSC) is a fundamental yet challenging task in computer vision that involves recognizing novel classes from limited data. While previous methods have focused on enhancing visual features or incorporating additional modalities, Large Vision Language Models (LVLMs) offer a promising alternative due to their rich knowledge and strong visual perception. However, LVLMs risk learning specific response formats rather than effectively extracting useful information from support data in FSC. In this paper, we investigate LVLMs' performance in FSC and identify key issues such as insufficient learning and the presence of severe position biases. To tackle above challenges, we adopt the meta-learning strategy to teach models ``learn to learn". By constructing a rich set of meta-tasks for instruction fine-tuning, LVLMs enhance the ability to extract information from few-shot support data for classification. Additionally, we further boost LVLM's few-shot learning capabilities through label augmentation (LA) and candidate selection (CS) in the fine-tuning and inference stages, respectively. LA is implemented via a character perturbation strategy to ensure the model focuses on support information. CS leverages attribute descriptions to filter out unreliable candidates and simplify the task. Extensive experiments demonstrate that our approach achieves superior performance on both general and fine-grained datasets. Furthermore, our candidate selection strategy has been proven beneficial for training-free LVLMs.

</details>

---

## 42. Union Is Strength! Unite the Power of LLMs and MLLMs for Chart Question Answering

- [ ] Union Is Strength! Unite the Power of LLMs and MLLMs for Chart Question Answering | https://ojs.aaai.org/index.php/AAAI/article/view/32584

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32584

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chart Question Answering (CQA) requires models to perform chart perception and reasoning. Recent studies driven by Large Language Models (LLMs) have dominated CQA. These include employing more cognitively capable LLMs for indirectly reasoning over transformed charts, i.e., tables, and directly perceiving charts utilizing Multimodal Large Language Models (MLLMs) with a wider perceptual range. Yet, they often encounter bottlenecks due to the limitation of the receptive field of LLMs and the fragility of the complex reasoning of some MLLMs. To unite the strengths of LLMs and MLLMs to complement each other's limitations, we propose Synergy, a framework that unites the power of both models for CQA. Synergy first unites the chart with a table as the augmented perceptual signal. Next, it unites LLMs and MLLMs, scheduling the former to decompose a question into subquestions and the latter to answer these by perceiving the chart. Lastly, it operates LLMs to summarize the subquestion-answer pairs to refine the final answer. Extensive experimental results on popular CharQA and PlotQA benchmarks reveal that, with the power of union, Synergy outperforms strong competitors and achieves superior boosts over naive MLLMs by uniting them with a smaller LLM.

</details>

---

## 43. Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation

- [ ] Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation | https://ojs.aaai.org/index.php/AAAI/article/view/32594

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32594

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary Scene Graph Generation (OV-SGG) overcomes the limitations of the closed-set assumption by aligning visual relationship representations with open-vocabulary textual representations. This enables the identification of novel visual relationships, making it applicable to real-world scenarios with diverse relationships.  However, existing OV-SGG methods are constrained by fixed text representations, limiting diversity and accuracy in image-text alignment.  To address these challenges, we propose the Relation-Aware Hierarchical Prompting (RAHP) framework, which enhances text representation by integrating subject-object and region-specific relation information.  Our approach utilizes entity clustering to address the complexity of relation triplet categories, enabling the effective integration of subject-object information. Additionally, we utilize a large language model (LLM) to generate detailed region-aware prompts, capturing fine-grained visual interactions and improving alignment between visual and textual modalities.  RAHP also introduces a dynamic selection mechanism within Vision-Language Models (VLMs), which adaptively selects relevant text prompts based on the visual content, reducing noise from irrelevant prompts.  Extensive experiments on the Visual Genome and Open Images v6 datasets demonstrate that our framework consistently achieves state-of-the-art performance, demonstrating its effectiveness in addressing the challenges of open-vocabulary scene graph generation.

</details>

---

## 44. Unveiling the Knowledge of CLIP for Training-Free Open-Vocabulary Semantic Segmentation

- [ ] Unveiling the Knowledge of CLIP for Training-Free Open-Vocabulary Semantic Segmentation | https://ojs.aaai.org/index.php/AAAI/article/view/32602

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32602

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Training-free open-vocabulary semantic segmentation aims to explore the potential of frozen vision-language models (VLM)  for segmentation tasks. Recent works reform the inference process of CLIP and utilize the features from the final layer to reconstruct dense representations for segmentation, demonstrating promising performance. However, the final layer tends to prioritize global components over local representations, leading to suboptimal robustness and effectiveness of existing methods. In this paper, we propose CLIPSeg, a novel training-free framework that fully exploits the diverse knowledge across layers in CLIP for dense predictions. Our study unveils two key discoveries: Firstly, the features in the middle layers exhibit high locality awareness and feature coherence compared to the final layer, based on which we propose the coherence enhanced residual attention module that generates semantic-aware attention. Secondly, despite not being directly aligned with the text, the deep layers capture valid local semantics that complement those in the final layer. Leveraging this insight, we introduce the deep semantic integration module to boost the patch semantics in the final block. Experiments conducted on 9 segmentation benchmarks with various CLIP models demonstrate that CLIPSeg consistently outperforms all training-free methods by substantial margins, e.g., a 7.8 % improvement in average mIoU for CLIP with a ViT-L backbone, and competes with learning-based counterparts in generalizing to novel concepts in an efficient way.

</details>

---

## 45. DoGA: Enhancing Grounded Object Detection via Grouped Pre-Training with Attributes

- [ ] DoGA: Enhancing Grounded Object Detection via Grouped Pre-Training with Attributes | https://ojs.aaai.org/index.php/AAAI/article/view/32603

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32603

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language pre-training have significantly enhanced the model capabilities on grounded object detection. However, these studies often pre-train with coarse-grained text prompts, such as plain category names and brief grounded phrases. This limitation curtails the model's capacity for fine-grained linguistic comprehension and leads to a significant decline in performance when faced with detailed descriptions or contextual information. To tackle these problems, we develop DoGA: Detect objects with Grouped Attributes, which employs commonly apparent attributes to bridge different granular semantics and uses specific attributes to identify the object discrepancy. Our DoGA incorporates three principle components: 1) Generation of attribute-based prompts, consisting of linguistic definitions enriched with common-sense visible attributes and hard negative notations deriving from the image-specific attribute features; 2) Paralleled entity fusion and optimization, designed to manage long attribute-based descriptions and negative concepts efficiently; and 3) Prompt-wise grouped training to accommodate model to perform many-to-many assignments, facilitating simultaneous training and inferring with multiple attribute-based synonyms. Extensive experiments demonstrate that training with synonymous attribute-based prompts allows DoGA to generalize multi-granular prompts and surpass previous state-of-the-art approaches, yielding 50.2 on the COCO and 38.0 on the LVIS benchmarks under the zero-short setting. We will make our code publicly available upon acceptance.

</details>

---

## 46. Asymmetric Visual Semantic Embedding Framework for Efficient Vision-Language Alignment

- [ ] Asymmetric Visual Semantic Embedding Framework for Efficient Vision-Language Alignment | https://ojs.aaai.org/index.php/AAAI/article/view/32605

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32605

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning visual semantic similarity is a critical challenge in bridging the gap between images and texts. However, there exist inherent variations between vision and language data, such as information density, i.e., images can contain textual information from multiple different views, which makes it difficult to compute the similarity between these two modalities accurately and efficiently. In this paper, we propose a novel framework called Asymmetric Visual Semantic Embedding (AVSE) to dynamically select features from various regions of images tailored to different textual inputs for similarity calculation.
To capture information from different views in the image, we design a radial bias sampling module to sample image patches and obtain image features from various views, Furthermore, AVSE introduces a novel module for efficient computation of visual semantic similarity between asymmetric image and text embeddings.
 Central to this module is the presumption of foundational semantic units within the embeddings, denoted as ``meta-semantic embeddings." It segments all embeddings into meta-semantic embeddings with the same dimension and calculates visual semantic similarity by finding the optimal match of meta-semantic embeddings of two modalities. 
Our proposed AVSE model is extensively evaluated on the large-scale MS-COCO and Flickr30K datasets, demonstrating its superiority over recent state-of-the-art methods.

</details>

---

## 47. CLIP-PCQA: Exploring Subjective-Aligned Vision-Language Modeling for Point Cloud Quality Assessment

- [ ] CLIP-PCQA: Exploring Subjective-Aligned Vision-Language Modeling for Point Cloud Quality Assessment | https://ojs.aaai.org/index.php/AAAI/article/view/32607

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32607

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, No-Reference Point Cloud Quality Assessment (NR-PCQA) research has achieved significant progress. However, existing methods mostly seek a direct mapping function from visual data to the Mean Opinion Score (MOS), which is contradictory to the mechanism of practical subjective evaluation. To address this, we propose a novel language-driven PCQA method named CLIP-PCQA. Considering that human beings prefer to describe visual quality using discrete quality descriptions (e.g., "excellent" and "poor") rather than specific scores, we adopt a retrieval-based mapping strategy to simulate the process of subjective assessment. More specifically, based on the philosophy of CLIP, we calculate the cosine similarity between the visual features and multiple textual features corresponding to different quality descriptions, in which process an effective contrastive loss and learnable prompts are introduced to enhance the feature extraction. Meanwhile, given the personal limitations and bias in subjective experiments, we further covert the feature similarities into probabilities and consider the Opinion Score Distribution (OSD) rather than a single MOS as the final target. Experimental results show that our CLIP-PCQA outperforms other State-Of-The-Art (SOTA) approaches.

</details>

---

## 48. DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval

- [ ] DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/32608

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32608

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-based person retrieval (TPR) has gained significant attention as a fine-grained and challenging task that closely aligns with practical applications. Tailoring CLIP to person domain is now a emerging research topic due to the abundant knowledge of vision-language pretraining, but challenges still remain during fine-tuning: (i) Previous full-model fine-tuning in TPR is computationally expensive and prone to overfitting.(ii) Existing parameter-efficient transfer learning (PETL) for TPR lacks of fine-grained feature extraction. To address these issues, we propose Domain-Aware Mixture-of-Adapters (DM-Adapter), which unifies Mixture-of-Experts (MOE) and PETL to enhance fine-grained feature representations while maintaining efficiency. Specifically, Sparse Mixture-of-Adapters is designed in parallel to MLP layers in both vision and language branches, where different experts specialize in distinct aspects of person knowledge to handle features more finely. To promote the router to exploit domain information effectively and alleviate the routing imbalance, Domain-Aware Router is then developed by building a novel gating function and injecting learnable domain-aware prompts. Extensive experiments show that our DM-Adapter achieves state-of-the-art performance, outperforming previous methods by a significant margin.

</details>

---

## 49. Advancing Comprehensive Aesthetic Insight with Multi-Scale Text-Guided Self-Supervised Learning

- [ ] Advancing Comprehensive Aesthetic Insight with Multi-Scale Text-Guided Self-Supervised Learning | https://ojs.aaai.org/index.php/AAAI/article/view/32613

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32613

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image Aesthetic Assessment (IAA) is a vital and intricate task that entails analyzing and assessing an image's aesthetic values, and identifying its highlights and areas for improvement. Traditional methods of IAA often concentrate on a single aesthetic task and suffer from inadequate labeled datasets, thus impairing in-depth aesthetic comprehension. Despite efforts to overcome this challenge through the application of Multi-modal Large Language Models (MLLMs), such models remain underdeveloped for IAA purposes. To address this, we propose a comprehensive aesthetic MLLM capable of nuanced aesthetic insight. Central to our approach is an innovative multi-scale text-guided self-supervised learning technique. This technique features a multi-scale feature alignment module and capitalizes on a wealth of unlabeled data in a self-supervised manner to structurally and functionally enhance aesthetic ability. The empirical evidence indicates that accompanied with extensive instruct-tuning, our model sets new state-of-the-art benchmarks across multiple tasks, including aesthetic scoring, aesthetic commenting, and personalized image aesthetic assessment. Remarkably, it also demonstrates zero-shot learning capabilities in the emerging task of aesthetic suggesting. Furthermore, for personalized image aesthetic assessment, we harness the potential of in-context learning and showcase its inherent advantages.

</details>

---

## 50. Can LVLMs Obtain a Driver’s License? A Benchmark Towards Reliable AGI for Autonomous Driving

- [ ] Can LVLMs Obtain a Driver’s License? A Benchmark Towards Reliable AGI for Autonomous Driving | https://ojs.aaai.org/index.php/AAAI/article/view/32623

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32623

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have recently garnered significant attention, with many efforts aimed at harnessing their general knowledge to enhance the interpretability and robustness of autonomous driving models. However, LVLMs typically rely on large, general-purpose datasets and lack the specialized expertise required for professional and safe driving. Existing vision-language driving datasets focus primarily on scene understanding and decision-making, without providing explicit guidance on traffic rules and driving skills, which are critical aspects directly related to driving safety. To bridge this gap, we propose IDKB, a large-scale dataset containing over one million data items collected from various countries, including driving handbooks, theory test data, and simulated road test data. Much like the process of obtaining a driver's license, IDKB encompasses nearly all the explicit knowledge needed for driving from theory to practice. In particular, we conducted comprehensive tests on 15 LVLMs using IDKB to assess their reliability in the context of autonomous driving and provided extensive analysis. We also fine-tuned popular models, achieving notable performance improvements, which further validate the significance of our dataset.

</details>

---

## 51. Revisiting Change Captioning from Self-supervised Global-Part Alignment

- [ ] Revisiting Change Captioning from Self-supervised Global-Part Alignment | https://ojs.aaai.org/index.php/AAAI/article/view/32629

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32629

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The goal of image change captioning is to capture the content differences between two images and describe them in natural language. The key is how to learn stable content changes from noise such as viewpoint and image structure. However, current work mostly focuses on identifying changes, and the influence of global noise leads to unstable recognition of global features. In order to tackle this problem, we propose a Self-supervised Global-Part Alignment (SSGPA) network and revisit the image change captioning task by enhancing the construction process of overall image global features, enabling the model to integrate global changes such as viewpoint into local changes, and to detect and describe changes in the image through alignment.  Concretely, we first design a Global-Part Transport Alignment mechanism to enhance global features and learn stable content changes through a self-supervised method of optimal transport. Further, we design a Change Fusion Adapter with pre-trained vision-language model to enhance the similar parts features of paired images, thereby enhancing global features, and expanding content changes. Extensive experiments show our method achieves the state-of-the-art results on four datasets.

</details>

---

## 52. Aligning and Prompting Anything for Zero-Shot Generalized Anomaly Detection

- [ ] Aligning and Prompting Anything for Zero-Shot Generalized Anomaly Detection | https://ojs.aaai.org/index.php/AAAI/article/view/32637

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32637

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot generalized anomaly detection (ZGAD) plays a critical role in industrial automation and health screening. Recent studies have shown that ZGAD methods built on visual-language models (VLMs) like CLIP have excellent cross-domain detection performance. Different from other computer vision tasks, ZGAD needs to jointly optimize both image-level anomaly classification and pixel-level anomaly segmentation tasks for determining whether an image contains anomalies and detecting anomalous parts of an image, respectively, this leads to different granularity of the tasks. However, existing methods ignore this problem, processing these two tasks with one set of broad text prompts used to describe the whole image. This limits CLIP to align textual features with pixel-level visual features and impairs anomaly segmentation performance. Therefore, for precise visual-text alignment, in this paper we propose a novel fine-grained text prompts generation strategy. We then apply the broad text prompts and the generated fine-grained text prompts for visual-textual alignment in classification and segmentation tasks, respectively, accurately capturing normal and anomalous instances in images. We also introduce the Text Prompt Shunt (TPS) model, which performs joint learning by reconstruction the complementary and dependency relationships between the two tasks to enhance anomaly detection performance. This enables our method to focus on fine-grained segmentation of anomalous targets while ensuring accurate anomaly classification, and achieve pixel-level comprehensible CLIP for the first time in the ZGAD task. Extensive experiments on 13 real-world anomaly detection datasets demonstrate that TPS achieves superior ZGAD performance across highly diverse datasets from industrial and medical domains.

</details>

---

## 53. Does VLM Classification Benefit from LLM Description Semantics?

- [ ] Does VLM Classification Benefit from LLM Description Semantics? | https://ojs.aaai.org/index.php/AAAI/article/view/32638

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32638

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Accurately describing images with text is a foundation of explainable AI. Vision-Language Models (VLMs) like CLIP have recently addressed this by aligning images and texts in a shared embedding space, expressing semantic similarities between vision and language embeddings. VLM classification can be improved with descriptions generated by Large Language Models (LLMs). However, it is difficult to determine the contribution of actual description semantics, as the performance gain may also stem from a semantic-agnostic ensembling effect, where multiple modified text prompts act as a noisy test-time augmentation for the original one. 
We propose an alternative evaluation scenario to decide if a performance boost of LLM-generated descriptions is caused by such a noise augmentation effect or rather by genuine description semantics. The proposed scenario avoids noisy test-time augmentation and ensures that genuine, distinctive descriptions cause the performance boost. Furthermore, we propose a training-free method for selecting discriminative descriptions that work independently of classname-ensembling effects. Our approach identifies descriptions that effectively differentiate classes within a local CLIP label neighborhood, improving classification accuracy across seven datasets. Additionally, we provide insights into the explainability of description-based image classification using VLMs.

</details>

---

## 54. CAKE: Category Aware Knowledge Extraction for Open-Vocabulary Object Detection

- [ ] CAKE: Category Aware Knowledge Extraction for Open-Vocabulary Object Detection | https://ojs.aaai.org/index.php/AAAI/article/view/32639

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32639

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open vocabulary object detection (OVOD) task aims to detect objects of novel categories beyond the base categories in the training set. To this end, the detector needs to access image-text pairs containing rich semantic information or the visual language pre-trained model (VLM) learned on them. Recent OVOD methods rely on knowledge distillation from VLMs. However, there are two main problems in current methods: (1) Current knowledge distillation frameworks fail to take advantage of the global category information of VLMs and thus fail to learn category-specific knowledge. (2) Due to the overfitting phenomenon of base categories during training, current OVOD networks generally have the problem of suppressing novel categories as background. To address these two problems, we propose a Category Aware Knowledge Extraction framework (CAKE), which consists of a Category-Specific Knowledge Distillation branch (CSKD) and a Category Generalization Region Proposal Network (CG-RPN). CSKD can more fully extract category-strong related information through category-specific distillation, and it is also conducive to filtering the exclusion problem between individuals of the same category; in this process, the model constructs a category-specific feature set to maintain high-quality category features. CG-RPN leverages the guidance of feature set to adjust the confidence scores of region proposals, thereby mining proposals that potentially contain novel categories of objects.
Extensive experiments show that our method can plug and play well with many existing methods and significantly improve their detection performance. Moreover, our CAKE framework can reach the-state-of-the-art performance on OV-COCO and OV-LVIS datasets.

</details>

---

## 55. Instruct Where the Model Fails: Generative Data Augmentation via Guided Self-contrastive Fine-tuning

- [ ] Instruct Where the Model Fails: Generative Data Augmentation via Guided Self-contrastive Fine-tuning | https://ojs.aaai.org/index.php/AAAI/article/view/32640

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32640

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Data augmentation is expected to bring about unseen features of training set, enhancing the model’s ability to generalize in situations where data is limited. Generative image models trained on large web-crawled datasets such as LAION are known to produce images with stereotypes and imperceptible bias when used to augment training data, owing to dataset misalignment and the generator’s ignorance of the downstream model. We improve downstream task awareness in generated images by proposing a task-aware fine-tuning strategy that actively detects failures of downstream task in the target model to fine-tune the generation process between epochs. The dynamic fine-tuning strategy is achieved by (1) inspecting misalignment between generated data and original data via VLM captioners and (2) adjusts both prompts and diffusion model so that the strategy dynamically guides the generator by focusing on the detected bias of VLM. This is done via re-captioning the overfitted data as well as finetuning the diffusion trajectory in a contrastive manner. To co-operate with the VLM captioner, the contrastive fine-tuning process dynamically adjusts different parts of the diffusion trajectory based on detected misalignment, thus shifting the the generated distribution away from making the downstream model overfit. Our experiments on few-shot class incremental learning show that our instruction-guided finetuning strategy consistently assists the downstream model with higher classification accuracy compared to generative data augmentation baselines such as Stable Diffusion and GPT-4o, and state-of-the-art non-generative strategies.

</details>

---

## 56. Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models

- [ ] Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32651

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32651

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models have revitalized the image generation domain, playing crucial roles in both academic research and artistic expression. With the emergence of new diffusion models, assessing the performance of text-to-image models has become increasingly important.
Current metrics focus on directly matching the input text with the generated image, but due to cross-modal information asymmetry, this leads to unreliable or incomplete assessment results. Motivated by this, we introduce the Image Regeneration task in this study to assess text-to-image models by tasking the T2I model with generating an image according to the reference image.
We use GPT4V to bridge the gap between the reference image and the text input for the T2I model, allowing T2I models to understand image content.
This evaluation process is simplified as comparisons between the generated image and the reference image are straightforward. Two regeneration datasets spanning content-diverse and style-diverse evaluation dataset are introduced to evaluate the leading diffusion models currently available.
Additionally, we present ImageRepainter framework to enhance the quality of generated images by improving content comprehension via MLLM guided iterative generation and revision.
Our comprehensive experiments have showcased the effectiveness of this framework in assessing the generative capabilities of models. By leveraging MLLM, we have demonstrated that a robust T2M can produce images more closely resembling the reference image.

</details>

---

## 57. Black-Box Test-Time Prompt Tuning for Vision-Language Models

- [ ] Black-Box Test-Time Prompt Tuning for Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32652

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32652

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time prompt tuning (TPT) aims to adjust the vision-language models (e.g., CLIP) with learnable prompts during the inference phase. However, previous works overlooked that pre-trained models as a service (MaaS) have become a noticeable trend due to their commercial usage and potential risk of misuse. In the context of MaaS, users can only design prompts in inputs and query the black-box vision-language models through inference APIs, rendering the previous paradigm of utilizing gradient for prompt tuning is infeasible. In this paper, we propose black-box test-time prompt tuning (B²TPT), a novel framework that addresses the challenge of optimizing prompts without gradients in an unsupervised manner. Specifically, B²TPT designs a consistent or confident (CoC) pseudo-labeling strategy to generate high-quality pseudo-labels from the outputs. Subsequently, we propose to optimize low-dimensional intrinsic prompts using a derivative-free evolution algorithm and to project them onto the original text and vision prompts. This strategy addresses the gradient-free challenge while reducing complexity. Extensive experiments across 15 datasets demonstrate the superiority of B²TPT. The results show that B²TPT not only outperforms CLIP's zero-shot inference at test time, but also surpasses other gradient-based TPT methods.

</details>

---

## 58. EvdCLIP: Improving Vision-Language Retrieval with Entity Visual Descriptions from Large Language Models

- [ ] EvdCLIP: Improving Vision-Language Retrieval with Entity Visual Descriptions from Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32655

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32655

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language retrieval (VLR) has attracted significant attention in both academia and industry, which involves using text (or images) as queries to retrieve corresponding images (or text). However, existing methods often neglect the rich visual semantics knowledge of entities, thus leading to incorrect retrieval results. To address this problem, we propose the Entity Visual Description enhanced CLIP (EvdCLIP), designed to leverage the visual knowledge of entities to enrich queries. Specifically, since humans recognize entities through visual cues, we employ a large language model (LLM) to generate Entity Visual Descriptions (EVDs) as alignment cues to complement textual data. These EVDs are then integrated into raw queries to create visually-rich, EVD-enhanced queries. Furthermore, recognizing that EVD-enhanced queries may introduce noise or low-quality expansions, we develop a novel, trainable EVD-aware Rewriter (EaRW) for vision-language retrieval tasks. EaRW utilizes EVD knowledge and the generative capabilities of the language model to effectively rewrite queries. With our specialized training strategy, EaRW can generate high-quality and low-noise EVD-enhanced queries. Extensive quantitative and qualitative experiments on image-text retrieval benchmarks validate the superiority of EvdCLIP on vision-language retrieval tasks.

</details>

---

## 59. Extract Free Dense Misalignment from CLIP

- [ ] Extract Free Dense Misalignment from CLIP | https://ojs.aaai.org/index.php/AAAI/article/view/32660

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32660

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language generative models still frequently produce outputs misaligned with their inputs, evidenced by object hallucination in captioning and prompt misalignment in the text-to-image generation model. Recent studies have explored methods for identifying misaligned elements, aiming not only to enhance interpretability but also to improve model performance. However, current approaches primarily rely on large foundation models in a zero-shot manner or fine-tuned models with human annotations, which limits scalability due to significant computational costs. This work proposes a novel approach, dubbed CLIP4DM, for detecting dense misalignments from pre-trained CLIP, specifically focusing on pinpointing misaligned words between image and text. We carefully revamp the gradient-based attribution computation method, enabling negative gradient of individual text tokens to indicate misalignment. We also propose F-CLIPScore, which aggregates misaligned attributions with a global alignment score. We evaluate our method on various dense misalignment detection benchmarks, covering various image and text domains and misalignment types. Our method demonstrates state-of-the-art performance among zero-shot models and competitive performance with fine-tuned models while maintaining superior efficiency. Our qualitative examples show that our method has a unique strength to detect entity-level objects, intangible objects, and attributes that can not be easily detected for existing works. We conduct ablation studies and analyses to highlight the strengths and limitations of our approach.

</details>

---

## 60. Multi-Scale Contrastive Learning for Video Temporal Grounding

- [ ] Multi-Scale Contrastive Learning for Video Temporal Grounding | https://ojs.aaai.org/index.php/AAAI/article/view/32666

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32666

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Temporal grounding, which localizes video moments related to a natural language query, is a core problem of vision-language learning and video understanding. To encode video moments of varying lengths, recent methods employ a multi-level structure known as a feature pyramid. In this structure, lower levels concentrate on short-range video moments, while higher levels address long-range moments. Because higher levels experience downsampling to accommodate increasing moment length, their capacity to capture information is reduced and consequently leads to degraded information in moment representations. To resolve this problem, we propose a contrastive learning framework to capture salient semantics among video moments. Our key methodology is to leverage samples from the feature space emanating from multiple stages of the video encoder itself requiring neither data augmentation nor online memory banks to obtain positive and negative samples. To enable such an extension, we introduce a sampling process to draw multiple video moments corresponding to a common query. Subsequently, by utilizing these moments' representations across video encoder layers, we instantiate a novel form of multi-scale and cross-scale contrastive learning that links local short-range video moments with global long-range video moments. Extensive experiments demonstrate the effectiveness of our framework for not only long-form but also short-form video grounding.

</details>

---

## 61. DuSSS: Dual Semantic Similarity-Supervised Vision-Language Model for Semi-Supervised Medical Image Segmentation

- [ ] DuSSS: Dual Semantic Similarity-Supervised Vision-Language Model for Semi-Supervised Medical Image Segmentation | https://ojs.aaai.org/index.php/AAAI/article/view/32674

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32674

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Semi-supervised medical image segmentation (SSMIS) uses consistency learning to regularize model training, which alleviates the burden of pixel-wise manual annotations. However, it often suffers from error supervision from low-quality pseudo labels. Vision-Language Model (VLM) has great potential to enhance pseudo labels by introducing text prompt guided multimodal supervision information. It nevertheless faces the cross-modal problem: the obtained messages tend to correspond to multiple targets. To address aforementioned problems, we propose a Dual Semantic Similarity-Supervised VLM (DuSSS) for SSMIS. Specifically, 1) a Dual Contrastive Learning (DCL) is designed to improve cross-modal semantic consistency by capturing intrinsic representations within each modality and semantic correlations across modalities. 2) To encourage the learning of multiple semantic correspondences, a Semantic Similarity-Supervision strategy (SSS) is proposed and injected into each contrastive learning process in DCL, supervising semantic similarity via the distribution-based uncertainty levels. Furthermore, a novel VLM-based SSMIS network is designed to compensate for the quality deficiencies of pseudo-labels. It utilizes the pretrained VLM to generate text prompt guided supervision information, refining the pseudo label for better consistency regularization. Experimental results demonstrate that our DuSSS achieves outstanding performance with Dice of 82.52%, 74.61% and 78.03% on three public datasets (QaTa-COV19, BM-Seg and MoNuSeg).

</details>

---

## 62. VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis

- [ ] VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis | https://ojs.aaai.org/index.php/AAAI/article/view/32683

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32683

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper develops a Versatile and Honest vision language Model (VHM) for remote sensing image analysis. VHM is built on a large-scale remote sensing image-text dataset with rich-content captions (VersaD), and an honest instruction dataset comprising both factual and deceptive questions (HnstD). Unlike prevailing remote sensing image-text datasets, in which image captions focus on a few prominent objects and their relationships, VersaD captions provide detailed information about image properties, object attributes, and the overall scene. This comprehensive captioning enables VHM to thoroughly understand remote sensing images and perform diverse remote sensing tasks. Moreover, different from existing remote sensing instruction datasets that only include factual questions, HnstD contains additional deceptive questions stemming from the non-existence of objects. This feature prevents VHM from producing affirmative answers to nonsense queries, thereby ensuring its honesty. In our experiments, VHM significantly outperforms various vision language models on common tasks of scene classification, visual question answering, and visual grounding. Additionally, VHM achieves competent performance on several unexplored tasks, such as building vectorizing, multi-label classification and honest question answering.

</details>

---

## 63. ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models

- [ ] ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32689

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32689

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucinations in Multimodal Large Language Models (MLLMs) where generated responses fail to accurately reflect the given image pose a significant challenge to their reliability. To address this, we introduce ConVis, a novel training-free contrastive decoding method. ConVis leverages a text-to-image (T2I) generation model to semantically reconstruct the given image from hallucinated captions. By comparing the contrasting probability distributions produced by the original and reconstructed images, ConVis enables MLLMs to capture visual contrastive signals that penalize hallucination generation. Notably, this method operates purely within the decoding process, eliminating the need for additional data or model updates. Our extensive experiments on five popular benchmarks demonstrate that ConVis effectively reduces hallucinations across various MLLMs, highlighting its potential to enhance model reliability.

</details>

---

## 64. Eve: Efficient Multimodal Vision Language Models with Elastic Visual Experts

- [ ] Eve: Efficient Multimodal Vision Language Models with Elastic Visual Experts | https://ojs.aaai.org/index.php/AAAI/article/view/32718

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32718

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal vision language models (VLMs) have made significant progress with the support of continuously increasing model sizes and data volumes. Running VLMs on edge devices has become a challenge for their widespread application. There are several efficient VLM efforts, but they often sacrifice linguistic capabilities to enhance multimodal abilities, or require extensive training. To address this quandary,  we introduce the innovative framework of Efficient Vision Language Models with Elastic Visual Experts (Eve). By strategically incorporating adaptable visual expertise at multiple stages of training, Eve strikes a balance between preserving linguistic abilities and augmenting multimodal capabilities. This balanced approach results in a versatile model with only 1.8B parameters that delivers significant improvements in both multimodal and linguistic tasks. Notably, in configurations below 3B parameters, Eve distinctly outperforms in language benchmarks and achieves state-of-the-art results in VLM Benchmarks. Additionally, its multimodal accuracy outstrips that of the larger 7B LLaVA-1.5 model.

</details>

---

## 65. MM-CamObj: A Comprehensive Multimodal Dataset for Camouflaged Object Scenarios

- [ ] MM-CamObj: A Comprehensive Multimodal Dataset for Camouflaged Object Scenarios | https://ojs.aaai.org/index.php/AAAI/article/view/32723

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32723

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large visual-language models (LVLMs) have achieved great success in multiple applications. However, they still encounter challenges in complex scenes, especially those involving camouflaged objects. This is primarily due to the lack of samples related to camouflaged scenes in the training dataset. To mitigate this issue, we construct the MM-CamObj dataset for the first time, comprising two subsets: CamObj-Align and CamObj-Instruct. Specifically, CamObj-Align contains 11,363 image-text pairs, and it is designed for VL alignment and injecting rich knowledge of camouflaged scenes into LVLMs. CamObj-Instruct is collected for fine-tuning the LVLMs with improved instruction-following capabilities, and it includes 11,363 images and 68,849 conversations with diverse instructions. Based on the MM-CamObj dataset, we propose the CamObj-Llava, an LVLM specifically designed for addressing tasks in camouflaged scenes. To facilitate our model's effective acquisition of knowledge about camouflaged objects and scenes, we introduce a curriculum learning strategy with six distinct modes. Additionally, we construct the CamObj-Bench to evaluate the existing LVLMs' capabilities of understanding, recognition, localization and count in camouflage scenes. This benchmark includes 600 images and 7 tasks, with a total of 9,449 questions. Extensive experiments are conducted on the CamObj-Bench with CamObj-Llava, 8 existing open-source and 3 closed-source LVLMs. Surprisingly, the results indicate that our model achieves a 25.84% improvement in 4 out of 7 tasks compared to GPT-4o.

</details>

---

## 66. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment

- [ ] Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment | https://ojs.aaai.org/index.php/AAAI/article/view/32734

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32734

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on image classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-Steal), the first stealing attack against medical MLLMs. ADA-Steal relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.

</details>

---

## 67. ProcTag: Process Tagging for Assessing the Efficacy of Document Instruction Data

- [ ] ProcTag: Process Tagging for Assessing the Efficacy of Document Instruction Data | https://ojs.aaai.org/index.php/AAAI/article/view/32735

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32735

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, large language models (LLMs) and multimodal large language models (MLLMs) have demonstrated promising results on document visual question answering (VQA) task, particularly after training on document instruction datasets. An effective evaluation method for document instruction data is crucial in constructing instruction data with high efficacy, which, in turn, facilitates the training of LLMs and MLLMs for document VQA. However, most existing evaluation methods for instruction data are limited to the textual content of the instructions themselves, thereby hindering the effective assessment of document instruction datasets and constraining their construction. In this paper, we propose ProcTag, a data-oriented method that assesses the efficacy of document instruction data. ProcTag innovatively performs tagging on the execution process of instructions rather than the instruction text itself. By leveraging the diversity and complexity of these tags to assess the efficacy of the given dataset, ProcTag enables selective sampling or filtering of document instructions. Furthermore, DocLayPrompt, a novel semi-structured layout-aware document prompting strategy, is proposed for effectively representing documents. Experiments demonstrate that sampling existing open-sourced and generated document VQA/instruction datasets with ProcTag significantly outperforms current methods for evaluating instruction data. Impressively, with ProcTag-based sampling in the generated document datasets, only 30.5 percent of the document instructions are required to achieve 100 percent efficacy compared to the complete dataset.

</details>

---

## 68. ResMaster: Mastering High-Resolution Image Generation via Structural and Fine-Grained Guidance

- [ ] ResMaster: Mastering High-Resolution Image Generation via Structural and Fine-Grained Guidance | https://ojs.aaai.org/index.php/AAAI/article/view/32739

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32739

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models excel at producing high-quality images; however, scaling to higher resolutions, such as 4K, often results in structural distortions, and repetitive patterns. To this end, we introduce ResMaster, a novel, training-free method that empowers resolution-limited diffusion models to generate high-quality images beyond resolution restrictions. Specifically, ResMaster leverages a low-resolution reference image created by a pre-trained diffusion model to provide structural and fine-grained guidance for crafting high-resolution images on a patch-by-patch basis. To ensure a coherent structure, ResMaster meticulously aligns the low-frequency components of high-resolution patches with the low-resolution reference at each denoising step. For fine-grained guidance, tailored image prompts based on the low-resolution reference and enriched textual prompts produced by a vision-language model are incorporated. This approach could significantly mitigate local pattern distortions and improve detail refinement. Extensive experiments validate that ResMaster sets a new benchmark for high-resolution image generation.

</details>

---

## 69. SKI Models: Skeleton Induced Vision-Language Embeddings for Understanding Activities of Daily Living

- [ ] SKI Models: Skeleton Induced Vision-Language Embeddings for Understanding Activities of Daily Living | https://ojs.aaai.org/index.php/AAAI/article/view/32744

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32744

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The introduction of vision-language models like CLIP has enabled the development of foundational video models capable of generalizing to unseen videos and human actions. However, these models are typically trained on web videos, which often fail to capture the challenges present in Activities of Daily Living (ADL) videos. Existing works address ADL-specific challenges, such as similar appearances, subtle motion patterns, and multiple viewpoints, by combining 3D skeletons and RGB videos. However, these approaches are not integrated with language, limiting their ability to generalize to unseen action classes. In this paper, we introduce SKI models, which integrate 3D skeletons into the vision-language embedding space. SKI models leverage a skeleton-language model, SkeletonCLIP, to infuse skeleton information into Vision Language Models (VLMs) and Large Vision Language Models (LVLMs) through collaborative training. Notably, SKI models do not require skeleton data during inference, enhancing their robustness for real-world applications. The effectiveness of SKI models is validated on three popular ADL datasets for zero-shot action recognition and video caption generation tasks.

</details>

---

## 70. Leveraging Large Vision-Language Model as User Intent-Aware Encoder for Composed Image Retrieval

- [ ] Leveraging Large Vision-Language Model as User Intent-Aware Encoder for Composed Image Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/32768

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32768

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Composed Image Retrieval (CIR) aims to retrieve target images from candidate set using a hybrid-modality query consisting of a reference image and a relative caption that describes the user intent. Recent studies attempt to utilize Vision-Language Pre-training Models (VLPMs) with various fusion strategies for addressing the task. However, these methods typically fail to simultaneously meet two key requirements of CIR: comprehensively extracting visual information and faithfully following the user intent. In this work, we propose CIR-LVLM, a novel framework that leverages the large vision-language model (LVLM) as the powerful user intent-aware encoder to better meet these requirements. Our motivation is to explore the advanced reasoning and instruction-following capabilities of LVLM for accurately understanding and responding the user intent. Furthermore, we design a novel hybrid intent instruction module to provide explicit intent guidance at two levels: (1) The task prompt clarifies the task requirement and assists the model in discerning user intent at the task level. (2) The instance-specific soft prompt, which is adaptively selected from the learnable prompt pool, enables the model to better comprehend the user intent at the instance level compared to a universal prompt for all instances. CIR-LVLM achieves state-of-the-art performance across three prominent benchmarks with acceptable inference efficiency. We believe this study provides fundamental insights into CIR-related fields.

</details>

---

## 71. Beyond Human Data: Aligning Multimodal Large Language Models by Iterative Self-Evolution

- [ ] Beyond Human Data: Aligning Multimodal Large Language Models by Iterative Self-Evolution | https://ojs.aaai.org/index.php/AAAI/article/view/32774

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32774

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human preference alignment can significantly enhance the capabilities of Multimodal Large Language Models (MLLMs). However, collecting high-quality preference data remains costly. One promising solution is the self-evolution strategy, where models are iteratively trained on data they generate. Current multimodal self-evolution techniques, nevertheless, still need human- or GPT-annotated data. Some methods even require extra models or ground truth answers to construct preference data. To overcome these limitations, we propose a novel multimodal self-evolution framework that empowers the model to autonomously generate high-quality questions and answers using only unannotated images. First, in the question generation phase, we implement an image-driven self-questioning mechanism. This approach allows the model to create questions and evaluate their relevance and answerability based on the image content. If a question is deemed irrelevant or unanswerable, the model regenerates it to ensure alignment with the image. This process establishes a solid foundation for subsequent answer generation and optimization. Second, while generating answers, we design an answer self-enhancement technique to boost the discriminative power of answers. We begin by captioning the images and then use the descriptions to enhance the generated answers. Additionally, we utilize corrupted images to generate rejected answers, thereby forming distinct preference pairs for effective optimization. Finally, in the optimization step, we incorporate an image content alignment loss function alongside the Direct Preference Optimization (DPO) loss to mitigate hallucinations. This function maximizes the likelihood of the above generated descriptions in order to constrain the model's attention to the image content. As a result, model can generate more accurate and reliable outputs. Experiments demonstrate that our framework is competitively compared with previous methods that utilize external information, paving the way for more efficient and scalable MLLMs.

</details>

---

## 72. ALLVB: All-in-One Long Video Understanding Benchmark

- [ ] ALLVB: All-in-One Long Video Understanding Benchmark | https://ojs.aaai.org/index.php/AAAI/article/view/32775

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32775

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

From image to video understanding, the capabilities of Multi-modal LLMs (MLLMs) are increasingly powerful. However, most existing video understanding benchmarks are relatively short, which makes them inadequate for effectively evaluating the long-sequence modeling capabilities of MLLMs. This highlights the urgent need for a comprehensive and integrated long video understanding benchmark to assess the ability of MLLMs thoroughly. To this end, we propose ALLVB (ALL-in-One Long Video Understanding Benchmark). ALLVB's main contributions include: 1) It integrates 9 major video understanding tasks. These tasks are converted into video QA formats, allowing a single benchmark to evaluate 9 different video understanding capabilities of MLLMs, highlighting the versatility, comprehensiveness, and challenging nature of ALLVB. 2) A fully automated annotation pipeline using GPT-4o is designed, requiring only human quality control, which facilitates the maintenance and expansion of the benchmark. 3) It contains 1,376 videos across 16 categories, averaging nearly 2 hours each, with a total of 252k QAs. To the best of our knowledge, it is the largest long video understanding benchmark in terms of the number of videos, average duration, and number of QAs. We have tested various mainstream MLLMs on ALLVB, and the results indicate that even the most advanced commercial models have significant room for improvement. This reflects the benchmark's challenging nature and demonstrates the substantial potential for development in long video understanding.

</details>

---

## 73. MUSE: Mamba Is Efficient Multi-scale Learner for Text-video Retrieval

- [ ] MUSE: Mamba Is Efficient Multi-scale Learner for Text-video Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/32778

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32778

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-Video Retrieval (TVR) aims to align and associate relevant video content with corresponding natural language queries. Most existing TVR methods are based on large-scale pre-trained vision-language models (e.g., CLIP). However, due to CLIP's inherent plain structure, few TVR methods explore the multi-scale representations which offer richer contextual information for a more thorough understanding. To this end, we propose MUSE, a multi-scale mamba with linear computational complexity for efficient cross-resolution modeling. Specifically, the multi-scale representations are generated by applying a feature pyramid on the last single-scale feature map. Then, we employ the Mamba structure as an efficient multi-scale learner to jointly learn scale-wise representations. Furthermore, we conduct comprehensive studies to investigate different model structures and designs. Extensive results on three popular benchmarks have validated the superiority of MUSE.

</details>

---

## 74. Empowering LLMs with Pseudo-Untrimmed Videos for Audio-Visual Temporal Understanding

- [ ] Empowering LLMs with Pseudo-Untrimmed Videos for Audio-Visual Temporal Understanding | https://ojs.aaai.org/index.php/AAAI/article/view/32784

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32784

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have demonstrated remarkable capabilities in natural language and multimodal domains. By fine-tuning multimodal LLMs with temporal annotations from well-annotated datasets, e.g., dense video captioning datasets, their temporal understanding capacity in video-language tasks can be obtained. However, there is a notable lack of untrimmed audio-visual video datasets with precise temporal annotations for events. This deficiency hinders LLMs from learning the alignment between time, audio-visual events, and text tokens, thus impairing their ability to localize audio-visual events in videos temporally.
To address this gap, we introduce PU-VALOR, a comprehensive audio-visual dataset comprising over 114,081 pseudo-untrimmed videos with detailed temporal annotations. PU-VALOR is derived from the large-scale but coarse-annotated audio-visual dataset VALOR, through a subtle method involving event-based video clustering, random temporal scaling, and permutation.
By fine-tuning a multimodal LLM on PU-VALOR, we developed AVicuna, a model capable of aligning audio-visual events with temporal intervals and corresponding text tokens. AVicuna excels in temporal localization and time-aware dialogue capabilities.
Our experiments demonstrate that AVicuna effectively handles temporal understanding in audio-visual videos and achieves state-of-the-art performance on open-ended video QA, audio-visual QA, and audio-visual event dense localization tasks.

</details>

---

## 75. CaRDiff: Video Salient Object Ranking Chain of Thought Reasoning for Saliency Prediction with Diffusion

- [ ] CaRDiff: Video Salient Object Ranking Chain of Thought Reasoning for Saliency Prediction with Diffusion | https://ojs.aaai.org/index.php/AAAI/article/view/32785

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32785

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video saliency prediction aims to identify the regions in a video that attract human attention and gaze, driven by bottom-up features from the video and top-down processes like memory and cognition. Among these top-down influences, language plays a crucial role in guiding attention by shaping how visual information is interpreted.
Existing methods primarily focus on modeling perceptual information while neglecting the reasoning process facilitated by language, where ranking cues are crucial outcomes of this process and practical guidance for saliency prediction.
In this paper, we propose CaRDiff (Caption, Rank, and generate with Diffusion), a framework that imitates the process by integrating multimodal large language model (MLLM), a grounding module, and a diffusion model, to enhance video saliency prediction. Specifically, we introduce a novel prompting method VSOR-CoT (Video Slient Object Ranking Chain of Thought), which utilizes an MLLM with a grounding module to caption video content and infer salient objects along with their rankings and positions. This process derives ranking maps that can be sufficiently leveraged by the diffusion model to accurately decode the saliency maps for the given video.
Extensive experiments showcase the effectiveness of VSOR-CoT in improving the performance of video saliency prediction.
The proposed CaRDiff performs better than state-of-the-art models on the MVS dataset and demonstrates cross-dataset capabilities on the DHF1k dataset through zero-shot evaluation.

</details>

---

## 76. Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection

- [ ] Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection | https://ojs.aaai.org/index.php/AAAI/article/view/32790

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32790

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Few-shot anomaly detection (FSAD) aims to detect unseen anomaly regions with the guidance of very few normal support images from the same class. Existing FSAD methods usually find anomalies by directly designing complex text prompts to align them with visual features under the prevailing large vision-language model paradigm. However, these methods, almost always, neglect intrinsic contextual information in visual features, e.g., the interaction relationships between different vision layers, which is an important clue for detecting anomalies comprehensively. To this end, we propose a kernel-aware graph prompt learning framework, termed as KAG-prompt, by reasoning the cross-layer relations among visual features for FSAD. Specifically, a kernel-aware hierarchical graph is built by taking the different layer features focusing on anomalous regions of different sizes as nodes, meanwhile, the relationships between arbitrary pairs of nodes stand for the edges of the graph. By message passing over this graph, KAG-prompt can capture cross-layer contextual information, thus leading to more accurate anomaly prediction. Moreover, to integrate the information of multiple important anomaly signals in the prediction map, we propose a novel image-level scoring method based on multi-level information fusion. Extensive experiments on MVTecAD and VisA datasets show that KAG-prompt achieves state-of-the-art FSAD results for image-level/pixel-level anomaly detection.

</details>

---

## 77. VOILA: Complexity-Aware Universal Segmentation of CT Images by Voxel Interacting with Language

- [ ] VOILA: Complexity-Aware Universal Segmentation of CT Images by Voxel Interacting with Language | https://ojs.aaai.org/index.php/AAAI/article/view/32805

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32805

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Satisfactory progress has been achieved recently in universal segmentation of CT images. Following the success of vision-language methods, there is a growing trend towards utilizing text prompts and contrastive learning to develop universal segmentation models. However, there exists a significant imbalance in information density between 3D images and text prompts. Moreover, the standard fully connected layer segmentation approach faces significant challenges with handling multiple classes and exhibits poor generalizability. To address these challenges, we propose VOxel Interacting with LAnguage method (VOILA) for universal CT image segmentation. Initially, we align voxels and language into a shared representation space and classify voxels based on cosine similarity. Subsequently, we develop the Voxel-Language Interaction framework to mitigate the impact of class imbalance caused by foreground-background discrepancies and variations in target volumes. Furthermore, a Complexity-Aware Sampling method is proposed to focus on region hard to segment, achieved by generating pseudo heatmaps from a trainable Gaussian mixture distribution. Our results indicate the proposed VOILA is capable to achieve improved performance with reduced parameters and computational cost during training. Furthermore, it demonstrates significant generalizability across diverse datasets without additional fine-tuning.

</details>

---

## 78. TextToucher: Fine-Grained Text-to-Touch Generation

- [ ] TextToucher: Fine-Grained Text-to-Touch Generation | https://ojs.aaai.org/index.php/AAAI/article/view/32802

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32802

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Tactile sensation plays a crucial role in the development of multi-modal large models and embodied intelligence. To collect tactile data with minimal cost as possible, a series of studies have attempted to generate tactile images by vision-to-touch image translation. However, compared to text modality, visual modality-driven tactile generation cannot accurately depict human tactile sensation. In this work, we analyze the characteristics of tactile images in detail from two granularities: object-level (tactile texture, tactile shape), and sensor-level (gel status). We model these granularities of information through text descriptions and propose a fine-grained Text-to-Touch generation method (TextToucher) to generate high-quality tactile samples. Specifically, we introduce a multimodal large language model to build the text sentences about object-level tactile information and employ a set of learnable text prompts to represent the sensor-level tactile information. To better guide the tactile generation process with the built text information, we fuse the dual grains of text information and explore various dual-grain text conditioning methods within the diffusion transformer architecture. Furthermore, we propose a Contrastive Text-Touch Pre-training (CTTP) metric to precisely evaluate the quality of text-driven generated tactile data. Extensive experiments demonstrate the superiority of our TextToucher method.

</details>

---

## 79. Overcoming Heterogeneous Data in Federated Medical Vision-Language Pre-training: A Triple-Embedding Model Selector Approach

- [ ] Overcoming Heterogeneous Data in Federated Medical Vision-Language Pre-training: A Triple-Embedding Model Selector Approach | https://ojs.aaai.org/index.php/AAAI/article/view/32807

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32807

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The scarcity data of medical field brings the collaborative training in medical vision-language pre-training (VLP) cross different clients. Therefore, the collaborative training in medical VLP faces two challenges: First, the medical data requires privacy, thus can not directly shared across different clients. Second, medical data distribution across institutes is typically heterogeneous, hindering local model alignment and representation capabilities. To simultaneously overcome these two challenges, we propose the framework called personalized model selector with fused multimodal information (PMS-FM). The contribution of PMS-FM is two-fold: 1) PMS-FM uses embeddings to represent information in different formats, allowing for the fusion of multimodal data. 2) PMS-FM adapts to personalized data distributions by training multiple models. A model selector then identifies and selects the best-performing model for each individual client. Extensive experiments with multiple real-world medical datasets demonstrate the superb performance of PMS-FM over existing federated learning methods on different zero-shot classification tasks.

</details>

---

## 80. ParGo: Bridging Vision-Language with Partial and Global Views

- [ ] ParGo: Bridging Vision-Language with Partial and Global Views | https://ojs.aaai.org/index.php/AAAI/article/view/32806

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32806

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This work presents ParGo, a novel Partial-Global projector designed to connect the vision and language modalities for Multimodal Large Language Models (MLLMs). Unlike previous works that rely on global attention-based projectors, our ParGo bridges the representation gap between the separately pre-trained vision encoders and the LLMs by integrating global and partial views, which alleviates the overemphasis on prominent regions. To facilitate the effective training of ParGo, we collect a large-scale detail-captioned image-text dataset named ParGoCap-1M-PT, consisting of 1 million images paired with high-quality captions. Extensive experiments on several MLLM benchmarks demonstrate the effectiveness of our ParGo, highlighting its superiority in aligning vision and language modalities. Compared to conventional Q-Former projector, our ParGo achieves an improvement of 259.96 in MME benchmark. Furthermore, our experiments reveal that ParGo significantly outperforms other projectors, particularly in tasks that emphasize detail perception ability.

</details>

---

## 81. VLScene: Vision-Language Guidance Distillation for Camera-Based 3D Semantic Scene Completion

- [ ] VLScene: Vision-Language Guidance Distillation for Camera-Based 3D Semantic Scene Completion | https://ojs.aaai.org/index.php/AAAI/article/view/32841

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32841

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Camera-based 3D semantic scene completion (SSC) provides dense geometric and semantic perception for autonomous driving. However, images provide limited information making the model susceptible to geometric ambiguity caused by occlusion and perspective distortion. Existing methods often lack explicit semantic modeling between objects, limiting their perception of 3D semantic context. To address these challenges, we propose a novel method VLScene: Vision-Language Guidance Distillation for Camera-based 3D Semantic Scene Completion. The key insight is to use the vision-language model to introduce high-level semantic priors to provide the object spatial context required for 3D scene understanding. Specifically, we design a vision-language guidance distillation process to enhance image features, which can effectively capture semantic knowledge from the surrounding environment and improve spatial context reasoning. In addition, we introduce a geometric-semantic sparse awareness mechanism to propagate geometric structures in the neighborhood and enhance semantic information through contextual sparse interactions. Experimental results demonstrate that VLScene achieves rank-1st performance on challenging benchmarks—SemanticKITTI and SSCBench-KITTI-360, yielding remarkably mIoU scores of 17.52 and 19.10, respectively.

</details>

---

## 82. CAD-GPT: Synthesising CAD Construction Sequence with Spatial Reasoning-Enhanced Multimodal LLMs

- [ ] CAD-GPT: Synthesising CAD Construction Sequence with Spatial Reasoning-Enhanced Multimodal LLMs | https://ojs.aaai.org/index.php/AAAI/article/view/32849

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32849

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Computer-aided design (CAD) significantly enhances the efficiency, accuracy, and innovation of design processes by enabling precise 2D and 3D modeling, extensive analysis, and optimization. Existing methods for creating CAD models rely on latent vectors or point clouds, which are difficult to obtain, and storage costs are substantial. Recent advances in Multimodal Large Language Models (MLLMs) have inspired researchers to use natural language instructions and images for CAD model construction. However, these models still struggle with inferring accurate 3D spatial location and orientation, leading to inaccuracies in determining the spatial 3D starting points and extrusion directions for constructing geometries. This work introduces CAD-GPT, a CAD synthesis method with spatial reasoning-enhanced MLLM that takes either a single image or a textual description as input. To achieve precise spatial inference, our approach introduces a 3D Modeling Spatial Mechanism. This method maps 3D spatial positions and 3D sketch plane rotation angles into a 1D linguistic feature space using a specialized spatial unfolding mechanism, while discretizing 2D sketch coordinates into an appropriate planar space to enable precise determination of spatial starting position, sketch orientation, and 2D sketch coordinate translations. Extensive experiments demonstrate that CAD-GPT consistently outperforms existing state-of-the-art methods in CAD model synthesis, both quantitatively and qualitatively.

</details>

---

## 83. Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models

- [ ] Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32852

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32852

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have experienced significant advancements recently, but still struggle to recognize and interpret intricate details in high-resolution (HR) images effectively. While state-of-the-art (SOTA) MLLMs claim to process images at 4K resolution, existing MLLM benchmarks only support up to 2K, leaving the capabilities of SOTA models on true HR images largely untested. Furthermore, existing methods for enhancing HR image perception in MLLMs rely on computationally expensive visual instruction tuning. To address these limitations, we introduce HR-Bench, the first deliberately designed benchmark to rigorously evaluate MLLM performance on 4K & 8K images. Through extensive experiments, we demonstrate that while downsampling HR images leads to vision information loss, leveraging complementary modalities, e.g., text, can effectively compensate for this loss. Building upon this insight, we propose Divide, Conquer and Combine, a novel training-free framework for enhancing MLLM perception of HR images. Our method follows a three-staged approach: 1) Divide: recursively partitioning the HR image into patches and merging similar patches to minimize computational overhead, 2) Conquer: leveraging the MLLM to generate accurate textual descriptions for each image patch, and 3) Combine: utilizing the generated text descriptions to enhance the MLLM's understanding of the overall HR image. Extensive experiments show that: 1) the SOTA MLLM achieves 63% accuracy, which is markedly lower than the 87% accuracy achieved by humans on HR-Bench; 2) our method brings consistent and significant improvements (a relative increase of +6% on HR-Bench and +8% on general multimodal benchmarks).

</details>

---

## 84. From 2D CAD Drawings to 3D Parametric Models: A Vision-Language Approach

- [ ] From 2D CAD Drawings to 3D Parametric Models: A Vision-Language Approach | https://ojs.aaai.org/index.php/AAAI/article/view/32858

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32858

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present CAD2Program, a new method for reconstructing 3D parametric models from 2D CAD drawings. Our proposed method is inspired by recent successes in vision-language models (VLMs), and departs from traditional methods which rely on task-specific data representations and/or algorithms. Specifically, on the input side, we simply treat the 2D CAD drawing as a raster image, regardless of its original format, and encode the image with a standard ViT model. We show that such an encoding scheme achieves competitive performance against existing methods that operate on vector-graphics inputs, while imposing substantially fewer restrictions on the 2D drawings. On the output side, our method auto-regressively predicts a general-purpose language describing 3D parametric models in text form. Compared to other sequence modeling methods for CAD which use domain-specific sequence representations with fixed-size slots, our text-based representation is more flexible, and can be easily extended to arbitrary geometric entities and semantic or functional properties. Experimental results on a large-scale dataset of cabinet models demonstrate the effectiveness of our method.

</details>

---

## 85. GCD: Advancing Vision-Language Models for Incremental Object Detection via Global Alignment and Correspondence Distillation

- [ ] GCD: Advancing Vision-Language Models for Incremental Object Detection via Global Alignment and Correspondence Distillation | https://ojs.aaai.org/index.php/AAAI/article/view/32864

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32864

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Incremental object detection (IOD) is a challenging task that requires detection models to continuously learn from newly arriving data. This work focuses on incremental learning for vision-language detectors (VLDs), an under explored domain. Existing research typically adopts a local alignment paradigm to avoid label conflicts, where different tasks are learned separately without interaction. However, we reveal that this practice fails to effectively preserve the semantic structure. Specifically, aligned relationships between objects and texts would collapse when handling novel categories, ultimately leading to catastrophic forgetting. Though knowledge distillation (KD) is a common approach for tackling this, traditional KD performs poorly when directly applied to VLDs, as for different phases, a natural knowledge gap exists in both encoding and decoding processes. To address above issues, we propose a novel method called Global alignment and Correspondence Distillation (GCD). Differently, we first integrate knowledge across phases within the same embedding space to construct global semantic structure. We then enable effective knowledge distillation in VLDs through a semantic correspondence mechanism, ensuring consistent proposal generation and decoding. On the top of that, we distill teacher model’s informative predictions and topological relationships to maintain stable local semantic structure. Extensive experiments on COCO 2017 demonstrate that our method significantly outperforms existing approaches, achieving new state-of-the-art in various IOD scenarios.

</details>

---

## 86. RefDetector: A Simple Yet Effective Matching-based Method for Referring Expression Comprehension

- [ ] RefDetector: A Simple Yet Effective Matching-based Method for Referring Expression Comprehension | https://ojs.aaai.org/index.php/AAAI/article/view/32866

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32866

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the rapid and substantial advancements in object detection, it continues to face limitations imposed by pre-defined category sets. Current methods for visual grounding primarily focus on how to better leverage the visual backbone to generate text-tailored visual features, which may require adjusting the parameters of the entire model. Besides, some early methods, \ie, matching-based method, build upon and extend the functionality of existing object detectors by enabling them to localize an object based on free-form linguistic expressions, which have good application potential. However, the untapped potential of the matching-based approach has not been fully realized due to inadequate exploration. In this paper, we first analyze the limitations that exist in the current matching-based method (\ie, mismatch problem and complicated fusion mechanisms), and then present a simple yet effective matching-based method, namely RefDetector. To tackle the above issues, we devise a simple heuristic rule to generate proposals with improved referent recall. Additionally, we introduce a straightforward vision-language interaction module that eliminates the need for intricate manually-designed mechanisms. Moreover, we have explored the visual grounding based on the modern detector DETR, and achieved significant performance improvement. Extensive experiments on three REC benchmark datasets, \ie, RefCOCO, RefCOCO+, and RefCOCOg validate the effectiveness of the proposed method.

</details>

---

## 87. Enhancing Fine-Grained Vision-Language Pretraining with Negative Augmented Samples

- [ ] Enhancing Fine-Grained Vision-Language Pretraining with Negative Augmented Samples | https://ojs.aaai.org/index.php/AAAI/article/view/32869

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32869

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing Vision-Language Pretraining (VLP) methods have achieved remarkable improvements across a variety of vision-language tasks, confirming their effectiveness in capturing coarse-grained semantic correlations.  However, their capability for fine-grained understanding, which is critical for many nuanced vision-language applications, remains limited.  Prevailing VLP models often overlook the intricate distinctions in expressing different modal features and typically depend on the similarity of holistic features for cross-modal interactions.  Moreover, these models directly align and integrate features from different modalities, focusing more on coarse-grained general representations, thus failing to capture the nuanced differences necessary for tasks demanding a more detailed perception. In response to these limitations, we introduce Negative Augmented Samples(NAS), a refined vision-language pretraining model that innovatively incorporates NAS to specifically address the challenge of fine-grained understanding.  NAS utilizes a Visual Dictionary(VD) as a semantic bridge between visual and linguistic domains.  Additionally, it employs a Negative Visual Augmentation(NVA) method based on the VD to generate challenging negative image samples. These samples deviate from positive samples exclusively at the token level, thereby necessitating that the model discerns the subtle disparities between positive and negative samples with greater precision.  Comprehensive experiments validate the efficacy of NAS components and underscore its potential to enhance fine-grained vision-language comprehension.

</details>

---

## 88. Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature

- [ ] Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature | https://ojs.aaai.org/index.php/AAAI/article/view/32870

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32870

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As deep neural networks (DNNs) are widely applied in the physical world, many researches are focusing on physical-world adversarial examples (PAEs), which introduce perturbations to inputs and cause the model's incorrect outputs. However, existing PAEs face two challenges: unsatisfactory attack performance (i.e., poor transferability and insufficient robustness to environment conditions), and difficulty in balancing attack effectiveness with stealthiness, where better attack effectiveness often makes PAEs more perceptible.

In this paper, we explore a novel perturbation-based method to overcome the challenges. For the first challenge, we introduce a strategy Deceptive RF injection based on robust features (RFs) that are predictive, robust to perturbations, and consistent across different models. Specifically, it improves the transferability and robustness of PAEs by covering RFs of other classes onto the predictive features in clean images. For the second challenge, we introduce another strategy Adversarial Semantic Pattern Minimization, which removes most perturbations and retains only essential adversarial patterns in AEs. Based on the two strategies, we design our method Robust Feature Coverage Attack (RFCoA), comprising Robust Feature Disentanglement and Adversarial Feature Fusion. In the first stage, we extract target class RFs in feature space. In the second stage, we use attention-based feature fusion to overlay these RFs onto predictive features of clean images and remove unnecessary perturbations. Experiments show our method's superior transferability, robustness, and stealthiness compared to existing state-of-the-art methods.  Additionally, our method's effectiveness can extend to Large Vision-Language Models (LVLMs), indicating its potential applicability to more complex tasks.

</details>

---

## 89. IteRPrimE: Zero-shot Referring Image Segmentation with Iterative Grad-CAM Refinement and Primary Word Emphasis

- [ ] IteRPrimE: Zero-shot Referring Image Segmentation with Iterative Grad-CAM Refinement and Primary Word Emphasis | https://ojs.aaai.org/index.php/AAAI/article/view/32880

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32880

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot Referring Image Segmentation (RIS) identifies the instance mask that best aligns with a specified referring expression without training and fine-tuning, significantly reducing the labor-intensive annotation process. Despite achieving commendable results, previous CLIP-based models have a critical drawback: the models exhibit a notable reduction in their capacity to discern relative spatial relationships of objects. This is because they generate all possible masks on an image and evaluate each masked region for similarity to the given expression, often resulting in decreased sensitivity to direct positional clues in text inputs. Moreover, most methods have weak abilities to manage relationships between primary words and their contexts, causing confusion and reduced accuracy in identifying the correct target region. To address these challenges, we propose IteRPrimE (Iterative Grad-CAM Refinement and Primary word Emphasis), which leverages a saliency heatmap through Grad-CAM from a Vision-Language Pre-trained (VLP) model for image-text matching. An iterative Grad-CAM refinement strategy is introduced to progressively enhance the model's focus on the target region and overcome positional insensitivity, creating a self-correcting effect. Additionally, we design the Primary Word Emphasis module to help the model handle complex semantic relations, enhancing its ability to attend to the intended object. Extensive experiments conducted on the RefCOCO/+/g, and PhraseCut benchmarks demonstrate that IteRPrimE outperforms previous SOTA zero-shot methods, particularly excelling in out-of-domain scenarios.

</details>

---

## 90. CVLUE: A New Benchmark Dataset for Chinese Vision-Language Understanding Evaluation

- [ ] CVLUE: A New Benchmark Dataset for Chinese Vision-Language Understanding Evaluation | https://ojs.aaai.org/index.php/AAAI/article/view/32884

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32884

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the rapid development of Chinese vision-language models (VLMs), most existing Chinese vision-language (VL) datasets are constructed on Western-centric images from existing English VL datasets. The cultural bias in the images makes these datasets unsuitable for evaluating VLMs in Chinese culture. To remedy this issue, we present a new Chinese Vision-Language Understanding Evaluation (CVLUE) benchmark dataset, where the selection of object categories and images is entirely driven by Chinese native speakers, ensuring that the source images are representative of Chinese culture. The benchmark contains four distinct VL tasks ranging from image-text retrieval to visual question answering, visual grounding and visual dialogue. We present a detailed statistical analysis of CVLUE and provide a baseline performance analysis with several open-source multilingual VLMs on CVLUE and its English counterparts to reveal their performance gap between English and Chinese. Our in-depth category-level analysis reveals a lack of Chinese cultural knowledge in existing VLMs. We also find that fine-tuning on Chinese culture-related VL datasets effectively enhances VLMs' understanding of Chinese culture.

</details>

---

## 91. ICM-Assistant: Instruction-tuning Multimodal Large Language Models for Rule-based Explainable Image Content Moderation

- [ ] ICM-Assistant: Instruction-tuning Multimodal Large Language Models for Rule-based Explainable Image Content Moderation | https://ojs.aaai.org/index.php/AAAI/article/view/32908

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32908

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Controversial contents largely inundate the Internet, infringing various cultural norms and child protection standards. Traditional Image Content Moderation (ICM) models fall short in producing precise moderation decisions for diverse standards, while recent multimodal large language models (MLLMs), when adopted to general rule-based ICM, often produce classification and explanation results that are inconsistent with human moderators. Aiming at flexible, explainable, and accurate ICM, we design a novel rule-based dataset generation pipeline, decomposing concise human-defined rules and leveraging well-designed multi-stage prompts to enrich short explicit image annotations. Our ICM-Instruct dataset includes detailed moderation explanation and moderation Q-A pairs. Built upon it, we create our ICM-Assistant model in the framework of rule-based ICM, making it readily applicable in real practice. Our ICM-Assistant model demonstrates exceptional performance and flexibility. Specifically, it significantly outperforms existing approaches on various sources, improving both the moderation classification (36.8% on average) and moderation explanation quality (26.6% on average) consistently over existing MLLMs. 
Caution: Content includes offensive language or images.

</details>

---

## 92. Combating Multimodal LLM Hallucination via Bottom-Up Holistic Reasoning

- [ ] Combating Multimodal LLM Hallucination via Bottom-Up Holistic Reasoning | https://ojs.aaai.org/index.php/AAAI/article/view/32913

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32913

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have shown unprecedented capabilities in advancing various vision-language tasks. However, MLLMs face significant challenges with hallucinations, and misleading outputs that do not align with the input data. While existing efforts are paid to combat MLLM hallucinations, several pivotal challenges are still unsolved. First, while current approaches aggressively focus on addressing errors at the perception level, another important type at the cognition level requiring factual commonsense can be overlooked. In addition, existing methods might fall short in finding a more effective way to represent visual input, which is yet a key bottleneck that triggers visual hallucinations. Moreover, MLLMs can frequently be misled by faulty textual inputs and cause hallucinations, while unfortunately, this type of issue has long been overlooked by existing studies. Inspired by human intuition in handling hallucinations, this paper introduces a novel bottom-up reasoning framework. Our framework systematically addresses potential issues in both visual and textual inputs by verifying and integrating perception-level information with cognition-level commonsense knowledge, ensuring more reliable outputs. Extensive experiments demonstrate significant improvements in multiple hallucination benchmarks after integrating MLLMs with the proposed framework. In-depth analyses reveal the great potential of our methods in addressing perception- and cognition-level hallucinations.

</details>

---

## 93. Unified Knowledge Maintenance Pruning and Progressive Recovery with Weight Recalling for Large Vision-Language Models

- [ ] Unified Knowledge Maintenance Pruning and Progressive Recovery with Weight Recalling for Large Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32923

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32923

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Model (LVLM), leveraging Large Language Model (LLM) as the cognitive core, has recently become one of the most representative multimodal model paradigms.
However, with the expansion of unimodal branches, \emph{i.e.} visual encoder and LLM, the storage and computational burdens intensify, posing challenges for deployment.
Structured pruning has proved promising in compressing large models by trimming a large portion of insignificant network structures.
Nevertheless, most of them are predominantly designed for LLMs, either relying on unitary importance metrics that fail to deal with modality-wise imbalances or adopting generic pruning and recovery paradigms that overlook the unique calibration status and capability requirements of large models, leading to substantial performance degradation.
To address these issues, we propose a novel structured pruning approach for LVLMs, dubbed Unified Knowledge Maintenance Pruning and Progressive Recovery with Weight Recalling (UKMP). Specifically, we design a Unified Knowledge Maintenance Importance (UKMI) metric, which simultaneously considers balancing the block-wise and modality-wise importance by adaptive normalization, optimizing the importance estimation by refining gradient-based criteria, and maintaining the knowledge capacity of LVLMs by using the angle distribution information entropy. Moreover, we develop a LoRA-based Progressive Distillation (LPD) method that recalls the pruned weights and performs progressive distillation for comprehensive recovery.
Extensive experimental results across various vision-language tasks demonstrate the effectiveness of our approach, comparing to the state-of-the-art structured pruning methods.

</details>

---

## 94. TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning

- [ ] TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning | https://ojs.aaai.org/index.php/AAAI/article/view/32942

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32942

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the efficiency of prompt learning in transferring vision-language models (VLMs) to downstream tasks, existing methods mainly learn the prompts in a coarse-grained manner where the learned prompt vectors are shared across all categories. Consequently, the tailored prompts often fail to discern class-specific visual concepts, thereby hindering the transferred performance for classes that share similar or complex visual attributes. Recent advances mitigate this challenge by leveraging external knowledge from Large Language Models (LLMs) to furnish class descriptions, yet incurring notable inference costs. In this paper, we introduce TextRefiner, a plug-and-play method to refine the text prompts of existing methods by leveraging the internal knowledge of VLMs. Particularly, TextRefiner builds a novel local cache module to encapsulate fine-grained visual concepts derived from local tokens within the image branch. By aggregating and aligning the cached visual descriptions with the original output of the text branch, TextRefiner can efficiently refine and enrich the learned prompts from existing methods without relying on any external expertise. For example, it improves the performance of CoOp from 71.66% to 76.96% on 11 benchmarks, surpassing CoCoOp which introduced instance-wise feature for text prompts. Equipped with TextRefiner, PromptKD achieves state-of-the-art performance while keep inference efficient.

</details>

---

## 95. Expand VSR Benchmark for VLLM to Expertize in Spatial Rules

- [ ] Expand VSR Benchmark for VLLM to Expertize in Spatial Rules | https://ojs.aaai.org/index.php/AAAI/article/view/32945

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32945

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Distinguishing spatial relations is a basic part of human cognition which requires fine-grained perception on cross-instance.  
Although benchmarks like MME, MMBench and SEED  comprehensively have evaluated various capabilities which already include visual spatial reasoning(VSR).
There is still a lack of sufficient quantity and quality evaluation and optimization datasets for Vision Large Language Models(VLLMs) specifically targeting visual positional reasoning. 
To handle this, we first diagnosed current VLLMs with the VSR dataset and proposed a unified test set.
We found current VLLMs to exhibit a contradiction of over-sensitivity to language instructions and under-sensitivity to visual positional information.
By expanding the original benchmark from two aspects of tunning data and model structure, we mitigated this phenomenon. 
To our knowledge, we expanded spatially positioned image data controllably using diffusion models for the first time and integrated original visual encoding(CLIP) with other 3 powerful visual encoders(SigLIP, SAM and DINO).
After conducting combination experiments on scaling data and models, we obtained a VLLM VSR Expert(VSRE) that not only generalizes better to different instructions but also accurately distinguishes differences in visual positional information. 
VSRE achieved over a 27% increase in accuracy on the VSR test set. 
It becomes a performant VLLM on the position reasoning of both the VSR dataset and relevant subsets of other evaluation benchmarks. 
We hope it will accelerate advancements in VLLM on VSR learning.

</details>

---

## 96. Attention-Driven GUI Grounding: Leveraging Pretrained Multimodal Large Language Models Without Fine-Tuning

- [ ] Attention-Driven GUI Grounding: Leveraging Pretrained Multimodal Large Language Models Without Fine-Tuning | https://ojs.aaai.org/index.php/AAAI/article/view/32957

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32957

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have generated significant interest in their ability to autonomously interact with and interpret Graphical User Interfaces (GUIs). A major challenge in these systems is grounding—accurately identifying critical GUI components such as text or icons based on a GUI image and a corresponding text query. Traditionally, this task has relied on fine-tuning MLLMs with specialized training data to predict component locations directly. However, in this paper, we propose a novel Tuning-free Attention-driven Grounding (TAG) method that leverages the inherent attention patterns in pretrained MLLMs to accomplish this task without the need for additional fine-tuning. Our method involves identifying and aggregating attention maps from specific tokens within a carefully constructed query prompt. Applied to MiniCPM-Llama3-V 2.5, a state-of-the-art MLLM, our tuning-free approach achieves performance comparable to tuning-based methods, with notable success in text localization. Additionally, we demonstrate that our attention map-based grounding technique significantly outperforms direct localization predictions from MiniCPM-Llama3-V 2.5, highlighting the potential of using attention maps from pretrained MLLMs and paving the way for future innovations in this domain.

</details>

---

## 97. CLIP-driven View-aware Prompt Learning for Unsupervised Vehicle Re-identification

- [ ] CLIP-driven View-aware Prompt Learning for Unsupervised Vehicle Re-identification | https://ojs.aaai.org/index.php/AAAI/article/view/32962

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32962

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the emergence of vision-language pre-trained models, such as CLIP, some textual prompts have been gradually introduced recently into re-identification (Re-ID) tasks to obtain considerably robust multimodal information. However, most textual descriptions based on vehicle Re-ID tasks only contain identity index words without specific words to describe vehicle view information, thereby resulting in difficulty to be widely applied in vehicle Re-ID tasks with view variations. This case inspires us to propose a CLIP-driven view-aware prompt learning framework for unsupervised vehicle Re-ID. We first design a learnable textual prompt template called view-aware context optimization (ViewCoOp) based on dynamic multi-view word embeddings, which can fully obtain the proportion and position encoding of each view in the whole vehicle body region. Subsequently, a cross-modal mutual graph is constructed to explore the connections between inter-modal and intra-modal. Each sample is treated as a graph node, which extracts textual features based on ViewCoOp and the visual features of images. Moreover, leveraging the inter-cluster and intra-cluster correlation in the bimodal clustering results in the determination of connectivity between graph node pairs. Lastly, the proposed cross-modal mutual graph method utilizes supervised information from the bimodal gap to directly fine-tune the image encoder of CLIP for downstream unsupervised vehicle Re-ID tasks. Extensive experiments verify that the proposed method is capable of effectively obtaining cross-modal description ability from multiple views.

</details>

---

## 98. Zero-shot Video Moment Retrieval via Off-the-shelf Multimodal Large Language Models

- [ ] Zero-shot Video Moment Retrieval via Off-the-shelf Multimodal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32971

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32971

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The target of video moment retrieval (VMR) is predicting temporal spans within a video that semantically match a given linguistic query. Existing VMR methods based on multimodal large language models (MLLMs) overly rely on expensive high-quality datasets and time-consuming fine-tuning. Although some recent studies introduce a zero-shot setting to avoid fine-tuning, they overlook inherent language bias in the query, leading to erroneous localization. To tackle the aforementioned challenges, this paper proposes Moment-GPT, a tuning-free pipeline for zero-shot VMR utilizing frozen MLLMs. Specifically, we first employ LLaMA-3 to correct and rephrase the query to mitigate language bias. Subsequently, we design a span generator combined with MiniGPT-v2 to produce candidate spans adaptively. Finally, to leverage the video comprehension capabilities of MLLMs, we apply Video-ChatGPT and span scorer to select the most appropriate spans. Our proposed method substantially outperforms the state-of-the-art MLLM-based and zero-shot models on several public datasets, including QVHighlights, ActivityNet-Captions, and Charades-STA.

</details>

---

## 99. FATE: Feature-Adapted Parameter Tuning for Vision-Language Models

- [ ] FATE: Feature-Adapted Parameter Tuning for Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/32975

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32975

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Following the recent popularity of vision language models, several attempts, e.g., parameter-efficient fine-tuning (PEFT), have been made to extend them to different downstream tasks. Previous PEFT works motivate their methods from the view of introducing new parameters for adaptation but still need to learn this part of weight from scratch, i.e., random initialization. In this paper, we present a novel strategy that incorporates the potential of prompts, e.g., vision features, to facilitate the initial parameter space adapting to new scenarios. We introduce a Feature-Adapted parameTer Efficient tuning paradigm for vision-language models, dubbed as FATE, which injects informative features from the vision encoder into language encoder's parameters space. Specifically, we extract vision features from the last layer of CLIP's vision encoder and, after projection, treat them as parameters for fine-tuning each layer of CLIP's language encoder. By adjusting these feature-adapted parameters, we can directly enable communication between the vision and language branches, facilitating CLIP's adaptation to different scenarios. Experimental results show that FATE exhibits superior generalization performance on 11 datasets with a very small amount of extra parameters and computation.

</details>

---

## 100. FLAME: Learning to Navigate with Multimodal LLM in Urban Environments

- [ ] FLAME: Learning to Navigate with Multimodal LLM in Urban Environments | https://ojs.aaai.org/index.php/AAAI/article/view/32974

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32974

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have demonstrated potential in Vision-and-Language Navigation (VLN) tasks, yet current applications face challenges. While LLMs excel in general conversation scenarios, they struggle with specialized navigation tasks, yielding suboptimal performance compared to specialized VLN models. We introduce FLAME (FLAMingo-Architected Embodied Agent), a novel Multimodal LLM-based agent and architecture designed for urban VLN tasks that efficiently handles multiple observations. Our approach implements a three-phase tuning technique for effective adaptation to navigation tasks, including single perception tuning for street view description, multiple perception tuning for route summarization, and end-to-end training on VLN datasets. The augmented datasets are synthesized automatically. Experimental results demonstrate FLAME's superiority over existing methods, surpassing state-of-the-art methods by a 7.3% increase in task completion on Touchdown dataset. This work showcases the potential of Multimodal LLMs (MLLMs) in complex navigation tasks, representing an advancement towards applications of MLLMs in the field of embodied intelligence.

</details>

---

## 101. TG-LLaVA: Text Guided LLaVA via Learnable Latent Embeddings

- [ ] TG-LLaVA: Text Guided LLaVA via Learnable Latent Embeddings | https://ojs.aaai.org/index.php/AAAI/article/view/32982

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/32982

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Currently, inspired by the success of vision-language models (VLMs), an increasing number of researchers are focusing on improving VLMs and have achieved promising results. However, most existing methods concentrate on optimizing the connector and enhancing the language model component, while neglecting improvements to the vision encoder itself. In contrast, we propose Text Guided LLaVA (TG-LLaVA) in this paper, which optimizes VLMs by guiding the vision encoder with text, offering a new and orthogonal optimization direction. Specifically, inspired by the purpose-driven logic inherent in human behavior, we use learnable latent embeddings as a bridge to analyze textual instruction and add the analysis results to the vision encoder as guidance, refining it. Subsequently, another set of latent embeddings extracts additional detailed text-guided information from high-resolution local patches as auxiliary information. Finally, with the guidance of text, the vision encoder can extract text-related features, similar to how humans focus on the most relevant parts of an image when considering a question. This results in generating better answers. Experiments on various datasets validate the effectiveness of the proposed method. Remarkably, without the need for additional training data, our proposed method can bring more benefits to the baseline (LLaVA-1.5) compared with other concurrent methods. Furthermore, the proposed method consistently brings improvement in different settings.

</details>

---

## 102. LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding

- [ ] LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding | https://ojs.aaai.org/index.php/AAAI/article/view/33001

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33001

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have shown promise in instruction following and image understanding. While these models are powerful, they have not yet been developed to comprehend the more challenging 3D geometric and physical scenes, especially when it comes to the sparse outdoor LiDAR data. In this paper, we introduce LiDAR-LLM, which takes raw LiDAR data as input and harnesses the remarkable reasoning capabilities of LLMs to gain a comprehensive understanding of outdoor 3D scenes. The central insight of our LiDAR-LLM is the reformulation of 3D outdoor scene cognition as a language modeling problem, encompassing tasks such as 3D captioning, 3D grounding, 3D question answering, etc. Specifically, due to the scarcity of 3D LiDAR-text pairing data, we introduce a three-stage training strategy and generate relevant datasets, progressively aligning the 3D modality with the language embedding of LLM. Furthermore, we design a Position-Aware Transformer (PAT) to connect the 3D encoder with the LLM, which effectively bridges the modality gap and enhances the LLM's spatial orientation comprehension of visual features. Our experiments demonstrate that LiDAR-LLM effectively comprehends a wide range of instructions related to 3D scenes, achieving a 40.9 BLEU-1 score on the 3D captioning dataset, a Grounded Captioning accuracy of 63.1%, and a BEV mIoU of 14.3%.

</details>

---

## 103. Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation

- [ ] Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation | https://ojs.aaai.org/index.php/AAAI/article/view/33022

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33022

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, deep learning based methods have revolutionized remote sensing image segmentation. However, these methods usually rely on a predefined semantic class set, thus needing additional image annotation and model training when adapting to new classes. More importantly, they are unable to segment arbitrary semantic classes. In this work, we introduce Open-Vocabulary Remote Sensing Image Semantic Segmentation (OVRSISS), which aims to segment arbitrary semantic classes in remote sensing images. To address the lack of OVRSISS datasets, we develop LandDiscover50K, a comprehensive dataset of 51,846 images covering 40 diverse semantic classes. In addition, we propose a novel framework named GSNet that integrates domain priors from special remote sensing models and versatile capabilities of general vision-language models. Technically, GSNet consists of a Dual-Stream Image Encoder (DSIE), a Query-Guided Feature Fusion (QGFF), and a Residual Information Preservation Decoder (RIPD). DSIE first captures comprehensive features from both special models and general models in dual streams. Then, with the guidance of variable vocabularies, QGFF integrates specialist and generalist features, enabling them to complement each other. Finally, RIPD is proposed to aggregate multi-source features for more accurate mask predictions. Experiments show that our method outperforms other methods by a large margin, and our proposed LandDiscover50K improves the performance of OVRSISS methods. The dataset and method will be publicly available.

</details>

---

## 104. CLIMB-ReID: A Hybrid CLIP-Mamba Framework for Person Re-Identification

- [ ] CLIMB-ReID: A Hybrid CLIP-Mamba Framework for Person Re-Identification | https://ojs.aaai.org/index.php/AAAI/article/view/33039

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33039

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Person Re-IDentification (ReID) aims to identify specific persons from non-overlapping cameras. Recently, some works have suggested using large-scale pre-trained vision-language models like CLIP to boost ReID performance. Unfortunately, existing methods still struggle to address two key issues simultaneously: efficiently transferring the knowledge learned from CLIP and comprehensively extracting the context information from images or videos. To address these issues, we introduce CLIMB-ReID, a pioneering hybrid framework that synergizes the impressive power of CLIP with the remarkable computational efficiency of Mamba. Specifically, we first propose a novel Multi-Memory Collaboration (MMC) strategy to transfer CLIP's knowledge in a parameter-free and prompt-free form. Then, we design a Multi-Temporal Mamba (MTM) to capture multi-granular spatiotemporal information in videos. Finally, with Importance-aware Reorder Mamba (IRM), information from various scales is combined to produce robust sequence features. Extensive experiments show that our proposed method outperforms other state-of-the-art methods on both image and video person ReID benchmarks.

</details>

---

## 105. Cross-Lingual Text-Rich Visual Comprehension: An Information Theory Perspective

- [ ] Cross-Lingual Text-Rich Visual Comprehension: An Information Theory Perspective | https://ojs.aaai.org/index.php/AAAI/article/view/33049

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33049

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent Large Vision-Language Models (LVLMs) have shown promising reasoning capabilities on text-rich images from charts, tables, and documents.
However, the abundant text within such images may increase the model's sensitivity to language. 
This raises the need to evaluate LVLM performance on cross-lingual text-rich visual inputs, where the language in the image differs from the language of the instructions.
To address this, we introduce XT-VQA (Cross-Lingual Text-Rich Visual Question Answering), a benchmark designed to assess how LVLMs handle language inconsistency between image text and questions.
XT-VQA integrates five existing text-rich VQA datasets and a newly collected dataset, XPaperQA, covering diverse scenarios that require faithful recognition and comprehension of visual information despite language inconsistency. Our evaluation of prominent LVLMs on XT-VQA reveals a significant drop in performance for cross-lingual scenarios, even for models with multilingual capabilities. A mutual information analysis suggests that this performance gap stems from cross-lingual questions failing to adequately activate relevant visual information. To mitigate this issue, we propose MVCL-MI (Maximization of Vision-Language Cross-Lingual Mutual Information), where a visual-text cross-lingual alignment is built by maximizing mutual information between the model's outputs and visual information. This is achieved by distilling knowledge from monolingual to cross-lingual settings through KL divergence minimization, where monolingual output logits serve as a teacher. Experimental results on the XT-VQA demonstrate that MVCL-MI effectively reduces the visual-text cross-lingual performance disparity while preserving the inherent capabilities of LVLMs, shedding new light on the potential practice for improving LVLMs.

</details>

---

## 106. World Knowledge-Enhanced Reasoning Using Instruction-Guided Interactor in Autonomous Driving

- [ ] World Knowledge-Enhanced Reasoning Using Instruction-Guided Interactor in Autonomous Driving | https://ojs.aaai.org/index.php/AAAI/article/view/33067

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33067

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Multi-modal Large Language Models (MLLMs) with extensive world knowledge have revitalized autonomous driving, particularly in reasoning tasks within perceivable regions. However, when faced with perception-limited areas (dynamic or static occlusion regions), MLLMs struggle to effectively integrate perception ability with world knowledge for reasoning. These perception-limited regions can conceal crucial safety information, especially for vulnerable road users. In this paper, we propose a framework, which aims to improve autonomous driving performance under perception-limited conditions by enhancing the integration of perception capabilities and world knowledge. Specifically, we propose a plug-and-play instruction-guided interaction module that bridges modality gaps and significantly reduces the input sequence length, allowing it to adapt effectively to multi-view video inputs. Furthermore, to better integrate world knowledge with driving-related tasks, we have collected and refined a large-scale multi-modal dataset that includes 2 million natural language QA pairs, 1.7 million grounding task data. To evaluate the model’s utilization of world knowledge, we introduce an object-level risk assessment dataset comprising 200K QA pairs, where the questions necessitate multi-step reasoning leveraging world knowledge for resolution. Extensive experiments validate the effectiveness of our proposed method.

</details>

---

## 107. Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models

- [ ] Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/33073

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33073

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Face Anti-Spoofing (FAS) is essential for ensuring the security and reliability of facial recognition systems. Most existing FAS methods are formulated as binary classification tasks, providing confidence scores without interpretation. They exhibit limited generalization in out-of-domain scenarios, such as new environments or unseen spoofing types. In this work, we introduce a multimodal large language model (MLLM) framework for FAS, termed Interpretable Face Anti-Spoofing (I-FAS), which transforms the FAS task into an interpretable visual question answering (VQA) paradigm. Specifically, we propose a Spoof-aware Captioning and Filtering (SCF) strategy to generate high-quality captions for FAS images, enriching the model's supervision with natural language interpretations. To mitigate the impact of noisy captions during training, we develop a Lopsided Language Model (L-LM) loss function that separates loss calculations for judgment and interpretation, prioritizing the optimization of the former. Furthermore, to enhance the model's perception of global visual features, we design a Globally Aware Connector (GAC) to align multi-level visual representations with the language model. Extensive experiments on standard and newly devised One to Eleven cross-domain benchmarks, comprising 12 public datasets, demonstrate that our method significantly outperforms state-of-the-art methods.

</details>

---

## 108. DocKylin: A Large Multimodal Model for Visual Document Understanding with Efficient Visual Slimming

- [ ] DocKylin: A Large Multimodal Model for Visual Document Understanding with Efficient Visual Slimming | https://ojs.aaai.org/index.php/AAAI/article/view/33076

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33076

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current multimodal large language models (MLLMs) face significant challenges in visual document understanding (VDU) tasks due to the high resolution, dense text, and complex layouts typical of document images. These characteristics demand a high level of detail perception ability from MLLMs. While increasing input resolution improves detail perception capability, it also leads to longer sequences of visual tokens, increasing computational costs and straining the models' ability to handle long contexts. To address these challenges, we introduce DocKylin, a document-centric MLLM that performs visual content slimming at both the pixel and token levels, thereby reducing token sequence length in VDU scenarios. We introduce an Adaptive Pixel Slimming (APS) preprocessing module to perform pixel-level slimming, increasing the proportion of informative pixels. Moreover, we propose a novel Dynamic Token Slimming (DTS) module to conduct token-level slimming, filtering essential tokens and removing others to adaptively create a more compact visual sequence. Experiments demonstrate DocKylin's promising performance across various VDU benchmarks and the effectiveness of each component.

</details>

---

## 109. Enhancing Multimodal Large Language Models Complex Reason via Similarity Computation

- [ ] Enhancing Multimodal Large Language Models Complex Reason via Similarity Computation | https://ojs.aaai.org/index.php/AAAI/article/view/33107

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33107

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models have experienced rapid growth, and numerous different models have emerged. The interpretability of LVLMs remains an under-explored area. Especially when faced with more complex tasks such as chain-of-thought reasoning, its internal mechanisms still resemble a black box that is difficult to decipher. By studying the interaction and information flow between images and text, we noticed that in models such as LLaVA1.5,  image tokens that are semantically related to text are more likely to have information flow convergence in the LLM decoding layer, and these image tokens receive higher attention scores. However, those image tokens that are less relevant to the text do not have information flow convergence, and they only get very small attention scores. To efficiently utilize the image information, we propose a new image token reduction method, Simignore, which aims to improve the complex reasoning ability of LVLMs by computing the similarity between image and text embeddings and ignoring image tokens that are irrelevant and unimportant to the text. Through extensive experiments, we demonstrate the effectiveness of our method for complex reasoning tasks.

</details>

---

## 110. Track the Answer: Extending TextVQA from Image to Video with Spatio-Temporal Clues

- [ ] Track the Answer: Extending TextVQA from Image to Video with Spatio-Temporal Clues | https://ojs.aaai.org/index.php/AAAI/article/view/33115

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33115

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video text-based visual question answering (TextVQA) is a practical task that aims to answer questions by jointly reasoning textual and visual information in a given video. Inspired by the development of TextVQA in image domain, existing Video TextVQA approaches leverage a language model (e.g. T5) to process text-rich multiple frames and generate answers auto-regressively. Nevertheless, the spatio-temporal relationships among visual entities (including scene text and objects) will be disrupted and models are susceptible to interference from unrelated information, resulting in irrational reasoning and inaccurate answering. To tackle these challenges, we propose the TEA (stands for "Track the Answer'') method that better extends the generative TextVQA framework from image to video. TEA recovers the spatio-temporal relationships in a complementary way and incorporates OCR-aware clues to enhance the quality of reasoning questions. Extensive experiments on several public Video TextVQA datasets validate the effectiveness and generalization of our framework. TEA outperforms existing TextVQA methods, video-language pretraining methods and video large language models by great margins. The code will be publicly released.

</details>

---

## 111. Zero-Shot Learning in Industrial Scenarios: New Large-Scale Benchmark, Challenges and Baseline

- [ ] Zero-Shot Learning in Industrial Scenarios: New Large-Scale Benchmark, Challenges and Baseline | https://ojs.aaai.org/index.php/AAAI/article/view/33124

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33124

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Visual Language Models (LVLMs) have achieved remarkable success in vision tasks. However, the significant differences between industrial and natural scenes make applying LVLMs challenging. Existing LVLMs rely on user-provided prompts to segment objects. This often leads to suboptimal performance due to the inclusion of irrelevant pixels. In addition, the scarcity of data also makes the application of LVLMs in industrial scenarios remain unexplored. To fill this gap, this paper proposes an open industrial dataset and a Refined Text-Visual Prompt (RTVP) for zero-shot industrial defect detection. First, this paper constructs the Multi-Modal Industrial Open Dataset (MMIO) containing 80K+ samples. MMIO contains diverse industrial categories, including 6 super categories and 18 subcategories. MMIO is the first large-scale multi-scenes pre-training dataset for industrial zero-shot learning, and provides valuable training data for open models in future industrial scenarios. Based on MMIO, this paper provides a RTVP specifically for industrial zero-shot tasks. RTVP has two significant advantages: First, this paper designs an expert-guided large model domain adaptation mechanism and designs an industrial zero-shot method based on Mobile-SAM, which enhances the generalization ability of large models in industrial scenarios. Second, RTVP automatically generates visual prompts directly from images and considers text-visual prompt interactions ignored by previous LVLM, improving visual and textual content understanding. RTVP achieves SOTA with 42.2% and 24.7% AP in zero-shot and closed scenes of MMIO.

</details>

---

## 112. Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference

- [ ] Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference | https://ojs.aaai.org/index.php/AAAI/article/view/33131

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33131

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, applying multi-modal large language models (MLLMs) in various fields has achieved remarkable success. However, as the foundation model for many downstream tasks, MLLMs comprise the well-known Transformer network, which has a less efficient quadratic computation complexity. In this study, we introduce Cobra, a multi-modal large-scale language model built upon a state-space model, which has demonstrated significant potential in efficiently handling long sequences with fast inference and linear scalability concerning sequence length. Specifically, Cobra involves replacing Transformer-based backbone models (e.g., LLaMA or Phi) with pre-trained Mamba language models. We then empirically explore effective strategies for aligning visual and textual modalities and integrating various pre-trained Mamba model variants with visual encoders. Experiments across various multi-modal benchmarks demonstrate that: (i) Cobra performs 3× ∼ 4× faster than the most computationally efficient state-of-the-art methods, e.g., LLaVA-Phi and MobileVLM v2. Additionally, its performance is significantly enhanced thanks to the implementation of linear sequential modeling. (ii) Cobra fine-tunes a small parameter (∼48% of model parameters), leading to a significant improvement in overall performance compared to LLaVA.

</details>

---

## 113. Hierarchical Cross-Modal Alignment for Open-Vocabulary 3D Object Detection

- [ ] Hierarchical Cross-Modal Alignment for Open-Vocabulary 3D Object Detection | https://ojs.aaai.org/index.php/AAAI/article/view/33140

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33140

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary 3D object detection (OV-3DOD) aims at localizing and classifying novel objects beyond closed sets. The recent success of vision-language models (VLMs) has demonstrated their remarkable capabilities to understand open vocabularies. Existing works that leverage VLMs for 3D object detection (3DOD) generally resort to representations that lose the rich scene context required for 3D perception. To address this problem, we propose in this paper a hierarchical framework, named HCMA, to simultaneously learn local object and global scene information for OV-3DOD. Specifically, we first design a Hierarchical Data Integration (HDI) approach to obtain coarse-to-fine 3D-image-text data, which is fed into a VLM to extract object-centric knowledge. To facilitate the association of feature hierarchies, we then propose an Interactive Cross-Modal Alignment (ICMA) strategy to establish effective intra-level and inter-level feature connections. To better align features across different levels, we further propose an Object-Focusing Context Adjustment (OFCA) module to refine multi-level features by emphasizing object-related features. Extensive experiments demonstrate that the proposed method outperforms SOTA methods on the existing OV-3DOD benchmarks. It also achieves promising OV-3DOD results even without any 3D annotations.

</details>

---

## 114. Position-Aware Guided Point Cloud Completion with CLIP Model

- [ ] Position-Aware Guided Point Cloud Completion with CLIP Model | https://ojs.aaai.org/index.php/AAAI/article/view/33166

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33166

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Point cloud completion aims to recover partial geometric and topological shapes caused by equipment defects or limited viewpoints. Current methods either solely rely on the 3D coordinates of the point cloud to complete it or incorporate additional images with well-calibrated intrinsic parameters to guide the geometric estimation of the missing parts. Although these methods have achieved excellent performance by directly predicting the location of complete points, the extracted features lack fine-grained information regarding the location of the missing area. To address this issue, we propose a rapid and efficient method to expand an unimodal framework into a multimodal framework. This approach incorporates a position-aware module designed to enhance the spatial information of the missing parts through a weighted map learning mechanism. In addition, we establish a Point-Text-Image triplet corpus PCI-TI and MVP-TI based on the existing unimodal point cloud completion dataset and use the pre-trained vision-language model CLIP to provide richer detail information for 3D shapes, thereby enhancing performance. Extensive quantitative and qualitative experiments demonstrate that our method outperforms state-of-the-art point cloud completion methods.

</details>

---

## 115. Low-Light Image Enhancement via Generative Perceptual Priors

- [ ] Low-Light Image Enhancement via Generative Perceptual Priors | https://ojs.aaai.org/index.php/AAAI/article/view/33168

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33168

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although significant progress has been made in enhancing visibility, retrieving texture details, and mitigating noise in Low-Light (LL) images, the challenge persists in applying current Low-Light Image Enhancement (LLIE) methods to real-world scenarios, primarily due to the diverse illumination conditions encountered. Furthermore, the quest for generating enhancements that are visually realistic and attractive remains an underexplored realm. In response to these challenges, we present a novel LLIE framework with the guidance of Generative Perceptual Priors (GPP-LLIE) derived from vision-language models (VLMs). Specifically, we first propose a pipeline that guides VLMs to assess multiple visual attributes of the LL image and quantify the assessment to output the global and local perceptual priors. Subsequently, to incorporate these generative perceptual priors to benefit LLIE, we introduce a transformer-based backbone in the diffusion process, and develop a new layer normalization (GPP-LN) and an attention mechanism (LPP-Attn) guided by global and local perceptual priors. Extensive experiments demonstrate that our model outperforms current SOTA methods on paired LL datasets and exhibits superior generalization on real-world data.

</details>

---

## 116. A Comprehensive Overhaul of Multimodal Assistant with Small Language Models

- [ ] A Comprehensive Overhaul of Multimodal Assistant with Small Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/33194

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33194

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have showcased impressive skills in tasks related to visual understanding and reasoning. Yet, their widespread application faces obstacles due to the high computational demands during both the training and inference phases, restricting their use to a limited audience within the research and user communities. In this paper, we investigate the design aspects of Multimodal Small Language Models (MSLMs) and propose an efficient multimodal assistant named Mipha, which is designed to create synergy among various aspects: visual representation, language models, and optimization strategies. We show that without increasing the volume of training data, our Mipha-3B outperforms the state-of-the-art large MLLMs, especially LLaVA-1.5-13B, on multiple benchmarks. Through detailed discussion, we provide insights and guidelines for developing strong MSLMs that rival the capabilities of MLLMs.

</details>

---

## 117. ST3: Accelerating Multimodal Large Language Model by Spatial-Temporal Visual Token Trimming

- [ ] ST3: Accelerating Multimodal Large Language Model by Spatial-Temporal Visual Token Trimming | https://ojs.aaai.org/index.php/AAAI/article/view/33201

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33201

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) enhance their perceptual capabilities by integrating visual and textual information. However, processing the massive number of visual tokens incurs a significant computational cost. Existing analysis of the MLLM attention mechanisms remains shallow, leading to coarse-grain token pruning strategies that fail to effectively balance speed and accuracy. In this paper, we conduct a comprehensive investigation of MLLM attention mechanisms with LLaVA. We find that numerous visual tokens and partial attention computations are redundant during the decoding process. Based on this insight, we propose Spatial-Temporal Visual Token Trimming (ST3), a framework designed to accelerate MLLM inference without retraining. ST3 consists of two primary components: 1) Progressive Visual Token Pruning (PVTP), which eliminates inattentive visual tokens across layers, and 2)  Visual Token Annealing (VTA), which dynamically reduces the number of visual tokens in each layer as the generated tokens grow. Together, these techniques deliver around 2x faster inference with only about 30% KV cache memory compared to the original LLaVA, while maintaining consistent performance across various datasets. Crucially, ST3 can be seamlessly integrated into existing pre-trained MLLMs, providing a plug-and-play solution for efficient inference.

</details>

---

## 118. Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation

- [ ] Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation | https://ojs.aaai.org/index.php/AAAI/article/view/33426

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33426

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Large Language Models (LLMs) have demonstrated significant potential in the field of Recommendation Systems (RSs).  Most existing studies have focused on converting user behavior logs into textual prompts and leveraging techniques such as prompt tuning to enable LLMs for recommendation tasks. Meanwhile, research interest has recently grown in multimodal recommendation systems that integrate data from images, text, and other sources using modality fusion techniques.  This introduces new challenges to the existing LLM-based recommendation paradigm which relies solely on text modality information. Moreover, although Multimodal Large Language Models (MLLMs) capable of processing multi-modal inputs have emerged, how to equip MLLMs with multi-modal recommendation capabilities remains largely unexplored. To this end, in this paper, we propose the Multimodal Large Language Model-enhanced Sequential Multimodal Recommendation (MLLM-MSR) model. To capture the dynamic user preference, we design a two-stage user preference summarization method. Specifically, we first utilize an MLLM-based item-summarizer to extract image feature given an item and convert the image into text. Then, we employ a recurrent user preference summarization generation paradigm to capture the dynamic changes in user preferences based on an LLM-based user-summarizer. Finally, to enable the MLLM for multi-modal recommendation task, we propose to fine-tune a MLLM-based recommender using Supervised Fine-Tuning (SFT) techniques. Extensive evaluations across various datasets validate the effectiveness of MLLM-MSR, showcasing its superior ability to capture and adapt to the evolving dynamics of user preferences.

</details>

---

## 119. Debiased Multimodal Understanding for Human Language Sequences

- [ ] Debiased Multimodal Understanding for Human Language Sequences | https://ojs.aaai.org/index.php/AAAI/article/view/33583

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33583

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human multimodal language understanding (MLU) is an indispensable component of expression analysis (e.g., sentiment or humor) from heterogeneous modalities, including visual postures, linguistic contents, and acoustic behaviours. Existing works invariably focus on designing sophisticated structures or fusion strategies to achieve impressive improvements. Unfortunately, they all suffer from the subject variation problem due to data distribution discrepancies among subjects. Concretely, MLU models are easily misled by distinct subjects with different expression customs and characteristics in the training data to learn subject-specific spurious correlations, limiting performance and generalizability across new subjects. Motivated by this observation, we introduce a recapitulative causal graph to formulate the MLU procedure and analyze the confounding effect of subjects. Then, we propose SuCI, a simple yet effective causal intervention module to disentangle the impact of subjects acting as unobserved confounders and achieve model training via true causal effects. As a plug-and-play component, SuCI can be widely applied to most methods that seek unbiased predictions. Comprehensive experiments on several MLU benchmarks clearly show the effectiveness of the proposed module.

</details>

---

## 120. Enhancing Multi-Robot Semantic Navigation Through Multimodal Chain-of-Thought Score Collaboration

- [ ] Enhancing Multi-Robot Semantic Navigation Through Multimodal Chain-of-Thought Score Collaboration | https://ojs.aaai.org/index.php/AAAI/article/view/33607

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33607

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding how humans cooperatively utilize semantic knowledge to explore unfamiliar environments and decide on navigation directions is critical for house service multi-robot systems. Previous methods primarily focused on single-robot centralized planning strategies, which severely limited exploration efficiency. Recent research has considered decentralized planning strategies for multiple robots, assigning separate planning models to each robot, but these approaches often overlook communication costs. In this work, we propose Multimodal Chain-of-Thought Co-Navigation (MCoCoNav), a modular approach that utilizes multimodal Chain-of-Thought to plan collaborative semantic navigation for multiple robots. MCoCoNav combines visual perception with Vision Language Models (VLMs) to evaluate exploration value through probabilistic scoring, thus reducing time costs and achieving stable outputs. Additionally, a global semantic map is used as a communication bridge, minimizing communication overhead while integrating observational results. Guided by scores that reflect exploration trends, robots utilize this map to assess whether to explore new frontier points or revisit history nodes. Experiments on HM3D_v0.2 and MP3D demonstrate the effectiveness of our approach.

</details>

---

## 121. A Similarity Paradigm Through Textual Regularization Without Forgetting

- [ ] A Similarity Paradigm Through Textual Regularization Without Forgetting | https://ojs.aaai.org/index.php/AAAI/article/view/33768

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33768

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has emerged as a promising method for adapting pre-trained visual-language models (VLMs) to a range of downstream tasks. While optimizing the context can be effective for improving performance on specific tasks, it can often lead to poor generalization performance on unseen classes or datasets sampled from different distributions. It may be attributed to the fact that textual prompts tend to overfit downstream data distributions, leading to the forgetting of generalized knowledge derived from hand-crafted prompts.
In this paper, we propose a novel method called Similarity Paradigm with Textual Regularization  (SPTR) for prompt learning without forgetting. SPTR is a two-pronged design based on hand-crafted prompts that is an inseparable framework.  1) To avoid forgetting general textual knowledge, we introduce the optimal transport as a textual regularization to finely ensure approximation with hand-crafted features and tuning textual features. 2) In order to continuously unleash the general ability of multiple hand-crafted prompts, we propose a similarity paradigm for natural alignment score and adversarial alignment score to improve model robustness for generalization. Both modules share a common objective in addressing generalization issues, aiming to maximize the generalization capability derived from multiple hand-crafted prompts.
Four representative tasks
(i.e.,  non-generalization few-shot learning, base-to-novel generalization, cross-dataset generalization,  domain generalization) across  11 datasets demonstrate that SPTR outperforms existing prompt learning methods.

</details>

---

## 122. A Wander Through the Multimodal Landscape: Efficient Transfer Learning via Low-rank Sequence Multimodal Adapter

- [ ] A Wander Through the Multimodal Landscape: Efficient Transfer Learning via Low-rank Sequence Multimodal Adapter | https://ojs.aaai.org/index.php/AAAI/article/view/33868

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33868

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Efficient transfer learning methods such as adapter-based methods have shown great success in unimodal models and vision-language models. However, existing methods have two main challenges in fine-tuning multimodal models. Firstly, they are designed for vision-language tasks and fail to extend to situations where there are more than two modalities. Secondly, they exhibit limited exploitation of interactions between modalities and lack efficiency. To address these issues, in this paper, we propose the loW-rank sequence multimodal adapter (Wander). We first use the outer product to fuse the information from different modalities in an element-wise way effectively. For efficiency, we use CP decomposition to factorize tensors into rank-one components and achieve substantial parameter reduction. Furthermore,  we implement a token-level low-rank decomposition to extract more fine-grained features and sequence relationships between modalities. With these designs, Wander enables token-level interactions between sequences of different modalities in a parameter-efficient way. We conduct extensive experiments on datasets with different numbers of modalities, where Wander outperforms state-of-the-art efficient transfer learning methods consistently. The results fully demonstrate the effectiveness, efficiency and universality of Wander.

</details>

---

## 123. MARS: Mixture of Auto-Regressive Models for Fine-grained Text-to-image Synthesis

- [ ] MARS: Mixture of Auto-Regressive Models for Fine-grained Text-to-image Synthesis | https://ojs.aaai.org/index.php/AAAI/article/view/33882

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33882

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Auto-regressive models have made significant progress in the realm of text-to-image synthesis, yet devising an appropriate model architecture and training strategy to achieve a satisfactory level remains an important avenue of exploration. In this work, we introduce MARS, a novel framework for T2I generation that incorporates a specially designed Semantic Vision-Language Integration Expert (SemVIE). This innovative component integrates pre-trained LLMs by independently processing linguistic and visual information—freezing the textual component while fine-tuning the visual component. This methodology preserves the NLP capabilities of LLMs while imbuing them with exceptional visual understanding. Building upon the powerful base of the pre-trained Qwen-7B, MARS stands out with its bilingual generative capabilities corresponding to both English and Chinese language prompts and the capacity for joint image and text generation.  The flexibility of this framework lends itself to migration towards any-to-any task adaptability. Furthermore, MARS employs a multi-stage training strategy that first establishes robust image-text alignment through complementary bidirectional tasks and subsequently concentrates on refining the T2I generation process, significantly augmenting text-image synchrony and the granularity of image details.  Notably, MARS requires only 9% of the GPU days needed by SD1.5, yet it achieves remarkable results across a variety of benchmarks, illustrating the training efficiency and the potential for swift deployment in various applications.

</details>

---

## 124. Target Semantics Clustering via Text Representations for Robust Universal Domain Adaptation

- [ ] Target Semantics Clustering via Text Representations for Robust Universal Domain Adaptation | https://ojs.aaai.org/index.php/AAAI/article/view/33883

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33883

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Universal Domain Adaptation (UniDA) focuses on transferring source domain knowledge to the target domain under both domain shift and unknown category shift. Its main challenge lies in identifying common class samples and aligning them. Current methods typically obtain target domain semantics centers from an unconstrained continuous image representation space. Due to domain shift and the unknown number of clusters, these centers often result in complex and less robust alignment algorithm. In this paper, based on vision-language models, we search for semantic centers in a semantically meaningful and discrete text representation space. The constrained space ensures almost no domain bias and appropriate semantic granularity for these centers, enabling a simple and robust adaptation algorithm. Specifically, we propose TArget Semantics Clustering (TASC) via Text Representations, which leverages information maximization as a unified objective and involves two stages. First, with the frozen encoders, a greedy search-based framework is used to search for an optimal set of text embeddings to represent target semantics. Second, with the search results fixed, encoders are refined based on gradient descent, simultaneously achieving robust domain alignment and private class clustering. Additionally, we propose Universal Maximum Similarity (UniMS), a scoring function tailored for detecting open-set samples in UniDA. Experimentally, we evaluate the universality of UniDA algorithms under four category shift scenarios. Extensive experiments on four benchmarks demonstrate the effectiveness and robustness of our method, which has achieved state-of-the-art performance.

</details>

---

## 125. Enhance Vision-Language Alignment with Noise

- [ ] Enhance Vision-Language Alignment with Noise | https://ojs.aaai.org/index.php/AAAI/article/view/33918

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33918

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the advancement of pre-trained vision-language (VL) models, enhancing the alignment between visual and linguistic modalities in downstream tasks has emerged as a critical challenge. Different from existing fine-tuning methods that add extra modules to these two modalities, we investigate whether the frozen model can be fine-tuned by customized noise. Our approach is motivated by the scientific study of beneficial noise, namely Positive-incentive Noise (Pi-noise) , which quantitatively analyzes the impact of noise. It therefore implies a new scheme to learn beneficial noise distribution that can be employed to fine-tune VL models. Focusing on few-shot classification tasks based on CLIP, we reformulate the inference process of CLIP and apply variational inference, demonstrating how to generate Pi-noise towards visual and linguistic modalities. Then, we propose Positive-incentive Noise Injector (PiNI), which can fine-tune CLIP via injecting noise into both visual and text encoders. Since the proposed method can learn the distribution of beneficial noise, we can obtain more diverse embeddings of vision and language to better align these two modalities for specific downstream tasks within limited computational resources. We evaluate different noise incorporation approaches and network architectures of PiNI. The evaluation across 11 datasets demonstrates its effectiveness.

</details>

---

## 126. Super-Class Guided Transformer for Zero-Shot Attribute Classification

- [ ] Super-Class Guided Transformer for Zero-Shot Attribute Classification | https://ojs.aaai.org/index.php/AAAI/article/view/33971

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/33971

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Attribute classification is crucial for identifying specific characteristics within image regions.
Vision-Language Models (VLMs) have been effective in zero-shot tasks by leveraging their general knowledge from large-scale datasets.
Recent studies demonstrate that transformer-based models with class-wise queries can effectively address zero-shot multi-label classification.
However, poor utilization of the relationship between seen and unseen attributes makes the model lack generalizability.
Additionally, attribute classification generally involves many attributes, making maintaining the model’s scalability difficult.
To address these issues, we propose Super-class guided transFormer (SugaFormer), a novel framework that leverages super-classes to enhance scalability and generalizability for zero-shot attribute classification.
SugaFormer employs Super-class Query Initialization (SQI) to reduce the number of queries, utilizing common semantic information from super-classes, and incorporates Multi-context Decoding (MD) to handle diverse visual cues.
To strengthen generalizability, we introduce two knowledge transfer strategies that utilize VLMs. 
During training, Super-class guided Consistency Regularization (SCR) aligns model’s features with VLMs using super-class guided prompts, and during inference, Zero-shot Retrieval-based Score Enhancement (ZRSE) refines predictions for unseen attributes.
Extensive experiments demonstrate that SugaFormer achieves state-of-the-art performance across three widely-used attribute classification benchmarks under zero-shot, and cross-dataset transfer settings.

</details>

---

## 127. Self-Prompting Analogical Reasoning for UAV Object Detection

- [ ] Self-Prompting Analogical Reasoning for UAV Object Detection | https://ojs.aaai.org/index.php/AAAI/article/view/34026

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34026

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unmanned Aerial Vehicle Object Detection (UAVOD) presents unique challenges due to varying altitudes, dynamic backgrounds, and the small size of objects. Traditional detection methods often struggle with these challenges, as they typically rely on visual feature only and fail to extract the semantic relations between the objects. To address these limitations, we propose a novel approach named Self-Prompting Analogical Reasoning (SPAR). Our method utilizes the vision-language model (CLIP) to generate context-aware prompts based on image feature, providing rich semantic information that guides analogical reasoning. SPAR includes two main modules: self-prompting and analogical reasoning. Self-prompting module based on learnable description and CLIP-text encoder generates context-aware prompt by combining specific image feature; then an objectness prompt score map is produced by computing the similarity between pixel-level features and context-aware prompt. With this score map, multi-scale image features are enhanced and pixel-level features are chosen for graph construction. While for analogical reasoning module, graph nodes consists of category-level prompt nodes and pixel-level image feature nodes. Analogical inference is based graph convolution. Under the guidance of category-level nodes, different-scale object features  have been enhanced, which helps achieve more accurate detection of challenging objects. Extensive experiments illustrate  that SPAR outperforms traditional methods, offering a more robust and accurate solution for UAVOD.

</details>

---

## 128. MoLE:Decoding by Mixture of Layer Experts Alleviates Hallucination in Large Vision-Language Models

- [ ] MoLE:Decoding by Mixture of Layer Experts Alleviates Hallucination in Large Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34056

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34056

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision-Language Models (LVLMs) highlight their ability to integrate and process multi-modal information. However, hallucinations—where generated content is inconsistent with input vision and instructions—remain a challenge. In this paper, we analyze LVLMs' layer-wise decoding and identify that hallucinations can arise during the reasoning and factual information injection process. Additionally, as the number of generated tokens increases, the forgetting of the original prompt may also lead to hallucinations.To address this, we propose a training-free decoding method called Mixture of Layer Experts (MoLE). MoLE leverages a heuristic gating mechanism to dynamically select multiple layers of LVLMs as expert layers: the Final Expert, the Second Opinion expert, and the Prompt Retention Expert. By the cooperation of each expert, MoLE enhances the robustness and faithfulness of the generation process. Our extensive experiments demonstrate that MoLE significantly reduces hallucinations, outperforming the current state-of-the-art decoding techniques across three mainstream LVLMs and two established hallucination benchmarks. Moreover, our method reveals the potential of LVLMs to independently produce more reliable and accurate outputs.

</details>

---

## 129. KPL: Training-Free Medical Knowledge Mining of Vision-Language Models

- [ ] KPL: Training-Free Medical Knowledge Mining of Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34075

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34075

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Language Models such as CLIP excel in image recognition due to extensive image-text pre-training. However, applying the CLIP inference in zero-shot classification, particularly for medical image diagnosis, faces challenges due to: 1) the inadequacy of representing image classes solely with single category names; 2) the modal gap between the visual and text spaces generated by CLIP encoders. Despite attempts to enrich disease descriptions with large language models, the lack of class-specific knowledge often leads to poor performance. In addition, empirical evidence suggests that existing proxy learning methods for zero-shot image classification on natural image datasets exhibit instability when applied to medical datasets. 
To tackle these challenges, we introduce the Knowledge Proxy Learning (KPL) to mine knowledge from CLIP. 
KPL is designed to leverage CLIP's multimodal understandings for medical image classification through Text Proxy Optimization and Multimodal Proxy Learning.
Specifically, KPL retrieves image-relevant knowledge descriptions from the constructed knowledge-enhanced base to enrich semantic text proxies.
It then harnesses input images and these descriptions, encoded via CLIP, to stably generate multimodal proxies that boost the zero-shot classification performance.
Extensive experiments conducted on both medical and natural image datasets demonstrate that KPL enables effective zero-shot image classification, outperforming all baselines. These findings highlight the great potential in this paradigm of mining knowledge from CLIP for medical image classification and broader areas.

</details>

---

## 130. Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning

- [ ] Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning | https://ojs.aaai.org/index.php/AAAI/article/view/34123

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34123

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Guiding the policy of multi-agent reinforcement learning to align with human common sense is a difficult problem, largely due to the complexity of modeling common sense as a reward, especially in complex and long-horizon multi-agent tasks. Recent works have shown the effectiveness of reward shaping, such as potential-based rewards, to enhance policy alignment. The existing works, however, primarily rely on experts to design rule-based rewards, which are often labor-intensive and lack a high-level semantic understanding of common sense. To solve this problem, we propose a hierarchical vision-based reward shaping method. At the bottom layer, a visual-language model (VLM) serves as a generic potential function, guiding the policy to align with human common sense through its intrinsic semantic understanding. To help the policy adapts to uncertainty and changes in long-horizon tasks, the top layer features an adaptive skill selection module based on a visual large language model (vLLM). The module uses instructions, video replays, and training records to dynamically select suitable potential function from a pre-designed pool. Besides, our method is theoretically proven to preserve the optimal policy. Extensive experiments conducted in the Google Research Football environment demonstrate that our method not only achieves a higher win rate but also effectively aligns the policy with human common sense.

</details>

---

## 131. Spurious Feature Eraser: Stabilizing Test-Time Adaptation for Vision-Language Foundation Model

- [ ] Spurious Feature Eraser: Stabilizing Test-Time Adaptation for Vision-Language Foundation Model | https://ojs.aaai.org/index.php/AAAI/article/view/34124

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34124

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language foundation models have exhibited remarkable success across a multitude of downstream tasks due to their scalability on extensive image-text paired data. However, these models also display significant limitations when applied to downstream tasks, such as fine-grained image classification, as a result of ``decision shortcuts'' that hinder their generalization capabilities. In this work, we find that the CLIP model possesses a rich set of features, encompassing both desired invariant causal features and undesired decision shortcuts. Moreover, the underperformance of CLIP on downstream tasks originates from its inability to effectively utilize pre-trained features in accordance with specific task requirements. To address this challenge, we propose a simple yet effective method, Spurious Feature Eraser (SEraser), to alleviate the decision shortcuts by erasing the spurious features. Specifically, we introduce a test-time prompt tuning paradigm that optimizes a learnable prompt, thereby compelling the model to exploit invariant features while disregarding decision shortcuts during the inference phase. The proposed method effectively alleviates excessive dependence on potentially misleading spurious information. We conduct comparative analysis of the proposed method against various approaches which validates the significant superiority.

</details>

---

## 132. ComprehendEdit: A Comprehensive Dataset and Evaluation Framework for Multimodal Knowledge Editing

- [ ] ComprehendEdit: A Comprehensive Dataset and Evaluation Framework for Multimodal Knowledge Editing | https://ojs.aaai.org/index.php/AAAI/article/view/34127

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34127

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal language models (MLLMs) have revolutionized natural language processing and visual understanding, but often contain outdated or inaccurate information. Current multimodal knowledge editing evaluations are limited in scope and potentially biased, focusing on narrow tasks and failing to assess the impact on in-domain samples. To address these issues, we introduce ComprehendEdit, a comprehensive benchmark comprising eight diverse tasks from multiple datasets. We propose two novel metrics: Knowledge Generalization Index (KGI) and Knowledge Preservation Index (KPI), which evaluate editing effects on in-domain samples without relying on AI-synthetic samples. Based on insights from our framework, we establish Hierarchical In-Context Editing (HICE), a baseline method employing a two-stage approach that balances performance across all metrics. This study provides a more comprehensive evaluation framework for multimodal knowledge editing, reveals unique challenges in this field, and offers a baseline method demonstrating improved performance. Our work opens new perspectives for future research and provides a foundation for developing more robust and effective editing techniques for MLLMs.

</details>

---

## 133. Assessing Modality Bias in Video Question Answering Benchmarks with Multimodal Large Language Models

- [ ] Assessing Modality Bias in Video Question Answering Benchmarks with Multimodal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34183

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34183

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) can simultaneously process visual, textual, and auditory data, capturing insights that complement human analysis. 
However, existing video question-answering (VidQA) benchmarks and datasets often exhibit a bias toward a single modality, despite the goal of requiring advanced reasoning skills that integrate diverse modalities to answer the queries.

In this work, we introduce the modality importance score (MIS) to identify such bias. It is designed to assess which modality embeds the necessary information to answer the question. 
Additionally, we propose an innovative method using state-of-the-art MLLMs to estimate the modality importance, which can serve as a proxy for human judgments of modality perception.
With this MIS, we demonstrate the presence of unimodal bias and the scarcity of genuinely multimodal questions in existing datasets. 
We further validate the modality importance score with multiple ablation studies to evaluate the performance of MLLMs on permuted feature sets. 
Our results indicate that current models do not effectively integrate information due to modality imbalance in existing datasets. 
Our proposed MLLM-derived MIS can guide the curation of modality-balanced datasets that advance multimodal learning and enhance MLLMs' capabilities to understand and utilize synergistic relations across modalities.

</details>

---

## 134. FedPIA – Permuting and Integrating Adapters Leveraging Wasserstein Barycenters for Finetuning Foundation Models in Multi-Modal Federated Learning

- [ ] FedPIA – Permuting and Integrating Adapters Leveraging Wasserstein Barycenters for Finetuning Foundation Models in Multi-Modal Federated Learning | https://ojs.aaai.org/index.php/AAAI/article/view/34228

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34228

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (VLMs), possessing millions or billions of parameters, typically require large text and image datasets for effective fine-tuning. However, collecting data from various sites, especially in healthcare, is challenging due to strict privacy regulations. An alternative is to fine-tune these foundation models on end-user devices, such as in medical clinics and hospitals, without sending data to a server. These local clients typically have limited computing power and small datasets, which are not enough for fully fine-tuning large VLMs on their own. A naive solution to these scenarios is to leverage parameter-efficient fine-tuning (PEFT) strategies such as adapters and apply federated learning (FL) algorithms to combine the learned adapter weights, thereby respecting the resource limitations and data privacy of the clients. However, this approach does not fully leverage the knowledge from multiple adapters trained on diverse data distributions and for diverse tasks. The adapters are adversely impacted by data heterogeneity and task heterogeneity across clients resulting in sub-optimal convergence. To this end, we propose a novel framework called FedPIA that improves upon the naive combinations of FL and PEFT by introducing Permutation and Integration of the local Adapters in the server and global Adapters in the clients exploiting Wasserstein barycenters for improved blending of client-specific and client-agnostic knowledge. This layerwise permutation helps to bridge the gap in the parameter space of local and global adapters before integration. We conduct over 2000 client-level experiments utilizing 48 medical image datasets across five different medical vision-language FL task settings encompassing visual question answering as well as image and report-based multi-label disease detection. Our experiments involving diverse client settings, ten different modalities, and two VLM backbones demonstrate that FedPIA consistently outperforms the state-of-the-art PEFT-FL baselines.

</details>

---

## 135. Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models

- [ ] Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34366

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34366

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in Multimodal Large Language Models (MLLMs) often use large image tokens to compensate the visual shortcoming of MLLMs, which not only exhibits obvious redundancy but also greatly exacerbates the already high computation. Token pruning is an effective solution for speeding up MLLMs, but  when and how to drop tokens still remains a challenge.  In this paper, we propose a novel and training-free approach for the effective visual token pruning of MLLMs, termed FitPrune, which can quickly produce a complete pruning recipe for MLLMs according to a pre-defined budget.  Specifically, FitPrune considers token pruning as a statistical problem of MLLM and its objective is to find out an optimal pruning scheme that can minimize the divergence of the attention distributions before and after pruning. In practice, FitPrune can be quickly accomplished based on the attention statistics from a small batch of inference data, avoiding the expensive trials of MLLMs. According to the pruning recipe, an MLLM can directly remove the redundant visual tokens of different examples during inference.  To validate FitPrune, we apply it to a set of recent MLLMs, including LLaVA-1.5, LLaVA-HR and LLaVA-NEXT, and conduct extensive experiments on a set of benchmarks. The experimental results show that our FitPrune can not only reduce the computational complexity to a large extent, while retaining high performance, e.g., -54.9% FLOPs for LLaVA-NEXT with only 0.5% accuracy drop. Notably, the pruning recipe can be obtained in about 5 minutes.

</details>

---

## 136. Exploring the Better Multimodal Synergy Strategy for Vision-Language Models

- [ ] Exploring the Better Multimodal Synergy Strategy for Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34372

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34372

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language models (VLMs) have shown great potential in enhancing open-world visual concept comprehension. Recent researches focus on an optimum multimodal collaboration strategy that significantly advances CLIP-based few-shot tasks. However, existing prompt-based solutions suffer from unidirectional information flow and increased parameters since they explicitly condition the vision prompts on textual prompts across different transformer layers using non-shareable coupling functions. To address this issue, we propose a Dual-shared mechanism based on LoRA (DsRA) that addresses VLM adaptation in low-data regimes. The proposed DsRA enjoys several merits. First, we design an inter-modal shared coefficient that focuses on capturing visual and textual shared patterns, ensuring effective mutual synergy between image and text features. Second, an intra-modal shared matrix is proposed to achieve efficient parameter fine-tuning by combining the different coefficients to generate layer-wise adapters placed in encoder layers. Our extensive experiments demonstrate that DsRA improves the generalizability under few-shot classification, base-to-new generalization, and domain generalization settings. Our code will be released soon.

</details>

---

## 137. BiMAC: Bidirectional Multimodal Alignment in Contrastive Learning

- [ ] BiMAC: Bidirectional Multimodal Alignment in Contrastive Learning | https://ojs.aaai.org/index.php/AAAI/article/view/34384

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34384

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Achieving robust performance in vision-language tasks requires strong multimodal alignment, where textual and visual data interact seamlessly. Existing frameworks often combine contrastive learning with image captioning to unify visual and textual representations. However, reliance on global representations and unidirectional information flow from images to text limits their ability to reconstruct visual content accurately from textual descriptions. To address this limitation, we propose BiMAC, a novel framework that enables bidirectional interactions between images and text at both global and local levels. BiMAC employs advanced components to simultaneously reconstruct visual content from textual cues and generate textual descriptions guided by visual features. By integrating a text-region alignment mechanism, BiMAC identifies and selects relevant image patches for precise cross-modal interaction, reducing information noise and enhancing mapping accuracy. BiMAC achieves state-of-the-art performance across diverse vision-language tasks, including image-text retrieval, captioning, and classification.

</details>

---

## 138. A-VL: Adaptive Attention for Large Vision-Language Models

- [ ] A-VL: Adaptive Attention for Large Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34403

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34403

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Large Vision-Language Model (LVLM) integrates computer vision and natural language processing techniques, offering substantial application potential. However, these models demand extensive resources during inference. Adaptive attention techniques can dynamically reduce computational redundancy and thus improve efficiency. Although current adaptive attention methods significantly reduce the memory requirements of Transformer-based language models, they are not tailored for LVLMs. We observe that LVLMs generate responses from both remote image tokens and local text tokens, and different modalities have different attention patterns. This observation inspires us to manage the attention for each modality separately. Specifically, for visual input, we store the cache of potentially useful information but only compute the most critical parts. For language input, we care more about local information. Based on our observation and analysis of vision-language attention patterns, we develop A-VL, a plug-and-play adaptive attention tailored for LVLM inference. Extensive evaluations on three vision-language tasks and five datasets show the effectiveness of our designs. Our approach A-VL outperforms existing adaptive attention methods in reducing memory usage and computational load without compromising performance.

</details>

---

## 139. Mitigating Hallucinations in Large Vision-Language Models by Adaptively Constraining Information Flow

- [ ] Mitigating Hallucinations in Large Vision-Language Models by Adaptively Constraining Information Flow | https://ojs.aaai.org/index.php/AAAI/article/view/34512

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34512

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models show tremendous potential in understanding visual information through human languages. However, they are prone to suffer from object hallucination, i.e., the generated image descriptions contain objects that do not exist in the image. In this paper, we reveal that object hallucination can be attributed to overconfidence in irrelevant visual features when soft visual tokens map to the LLM's word embedding space. Specifically, by figuring out the semantic similarity between visual tokens and LLM's word embedding, we observe that the smoothness of similarity distribution strongly correlates with the emergence of object hallucinations. To mitigate hallucinations, we propose using the Variational Information Bottleneck (VIB) to alleviate overconfidence by introducing stochastic noise, facilitating the constraining of irrelevant information. Furthermore, we propose an entropy-based noise-controlling strategy to enable the injected noise to be adaptively constrained regarding the smoothness of the similarity distribution. We adapt the proposed AdaVIB across distinct model architectures. Experimental results demonstrate that the proposed AdaVIB mitigates object hallucinations by effectively alleviating the overconfidence in irrelevant visual features, with consistent improvements on two object hallucination benchmarks.

</details>

---

## 140. Affordances-Oriented Planning Using Foundation Models for Continuous Vision-Language Navigation

- [ ] Affordances-Oriented Planning Using Foundation Models for Continuous Vision-Language Navigation | https://ojs.aaai.org/index.php/AAAI/article/view/34526

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34526

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

LLM-based agents have demonstrated impressive zero-shot performance in vision-language navigation (VLN) task. However, existing LLM-based methods often focus only on solving high-level task planning by selecting nodes in predefined navigation graphs for movements, overlooking low-level control in navigation scenarios. To bridge this gap, we propose AO-Planner, a novel Affordances-Oriented Planner for continuous VLN task. Our AO-Planner integrates various foundation models to achieve affordances-oriented low-level motion planning and high-level decision-making, both performed in a zero-shot setting. Specifically, we employ a Visual Affordances Prompting (VAP) approach, where the visible ground is segmented by SAM to provide navigational affordances, based on which the LLM selects potential candidate waypoints and plans low-level paths towards selected waypoints. We further propose a high-level PathAgent which marks planned paths into the image input and reasons the most probable path by comprehending all environmental information. Finally, we convert the selected path into 3D coordinates using camera intrinsic parameters and depth information, avoiding challenging 3D predictions for LLMs. Experiments on the challenging R2R-CE and RxR-CE datasets show that AO-Planner achieves state-of-the-art zero-shot performance (8.8% improvement on SPL). Our method can also serve as a data annotator to obtain pseudo-labels, distilling its waypoint prediction ability into a learning-based predictor. This new predictor does not require any waypoint data from the simulator and achieves 47% SR competing with supervised methods. We establish an effective connection between LLM and 3D world, presenting novel prospects for employing foundation models in low-level motion control.

</details>

---

## 141. CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models

- [ ] CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34538

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34538

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have recently demonstrated amazing success in multi-modal tasks, including advancements in Multi-modal Chain-of-Thought (MCoT) reasoning. Despite these successes, current benchmarks still follow a traditional paradigm with multi-modal input and text-modal output, which leads to significant drawbacks such as missing visual operations and vague expressions.
Motivated by this, we introduce a novel Chain of Multi-modal Thought (CoMT) benchmark to address these limitations. Different from the traditional MCoT benchmark, CoMT requires both multi-modal input and multi-modal reasoning output, aiming to mimic human-like reasoning that inherently integrates visual operation. Specifically, CoMT consists of four categories: (1) Visual Creation, (2) Visual Deletion, (3) Visual Update, and (4) Visual Selection to comprehensively explore complex visual operations and concise expression in real scenarios.
We evaluate various LVLMs and strategies on CoMT, revealing some key insights into the capabilities and limitations of the current approaches. We hope that CoMT can inspire more research on introducing multi-modal generation into the reasoning process.

</details>

---

## 142. FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts

- [ ] FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts | https://ojs.aaai.org/index.php/AAAI/article/view/34568

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34568

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) signify a groundbreaking paradigm shift within the Artificial Intelligence (AI) community, extending beyond the capabilities of Large Language Models (LLMs) by assimilating additional modalities (e.g., images).
Despite this advancement, the safety of LVLMs remains adequately underexplored, with a potential overreliance on the safety assurances purported by their underlying LLMs. 
In this paper, we propose FigStep, a straightforward yet effective black-box jailbreak algorithm against LVLMs.
Instead of feeding textual harmful instructions directly, FigStep converts the prohibited content into images through typography to bypass the safety alignment.
The experimental results indicate that FigStep can achieve an average attack success rate of 82.50% on six promising open-source LVLMs.
Not merely to demonstrate the efficacy of FigStep, we conduct comprehensive ablation studies and analyze the distribution of the semantic embeddings to uncover that the reason behind the success of FigStep is the deficiency of safety alignment for visual embeddings. 
Moreover, we compare FigStep with five text-only jailbreaks and four image-based jailbreaks to demonstrate the superiority of FigStep, i.e., negligible attack costs and better attack performance.
Above all, our work reveals that current LVLMs are vulnerable to jailbreak attacks, which highlights the necessity of novel cross-modality safety alignment techniques.

</details>

---

## 143. DEQA: Descriptions Enhanced Question-Answering Framework for Multimodal Aspect-Based Sentiment Analysis

- [ ] DEQA: Descriptions Enhanced Question-Answering Framework for Multimodal Aspect-Based Sentiment Analysis | https://ojs.aaai.org/index.php/AAAI/article/view/34572

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34572

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal aspect-based sentiment analysis (MABSA) integrates text and images to perform fine-grained sentiment analysis on specific aspects, enhancing the understanding of user opinions in various applications. Existing methods use modality alignment for information interaction and fusion between images and text, but an inherent gap between these two modalities necessitates a more direct bridging mechanism to effectively connect image understanding with text content. For this, we propose the Descriptions Enhanced Question-Answering Framework (DEQA), which generates descriptions of images using GPT-4, leveraging the multimodal large language model to provide more direct semantic context of images. In DEQA, to help the model better understand the task's purpose, we frame MABSA as a multi-turn question-answering problem to add semantic guidance and hints. We input text, image, and description into separate experts in various combinations, allowing each expert to focus on different features and thereby improving the comprehensive utilization of input information. By integrating these expert outputs within a multi-turn question-answering format, we employ a multi-expert ensemble decision-making approach to produce the final prediction results. Experimental results on two widely-used datasets demonstrate that our method achieves state-of-the-art performance. Furthermore, our framework substantially outperforms GPT-4o and other multimodal large language models, showcasing its superior effectiveness in multimodal sentiment analysis.

</details>

---

## 144. LRM-LLaVA: Overcoming the Modality Gap of Multilingual Large Language-Vision Model for Low-Resource Languages

- [ ] LRM-LLaVA: Overcoming the Modality Gap of Multilingual Large Language-Vision Model for Low-Resource Languages | https://ojs.aaai.org/index.php/AAAI/article/view/34623

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34623

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multilingual large language-vision models (LVLMs), which understand and generate both text and images across multiple languages, have achieved remarkable performance on English-centric multimodal generation tasks. However, their performance on non-English tasks has been underwhelming. One major challenge with multilingual LVLMs is the modality gap between visual inputs and multilingual textual inputs/outputs due to the lack of high-quality multilingual training data. In this paper, we propose LRM-LLaVA, a multilingual large language-vision model designed for low-resource languages to overcome the modality gap. It is composed of four components: a visual encoder, a multilingual large language model, a vision-text representation projector, and a cross-modal regularizer. Both the projector and regularizer aim at reducing the modality gap and improving multilingual performance. To train LRM-LLaVA, we employ a two-stage training strategy including pre-training and instruction fine-tuning. Meanwhile, we construct a multilingual visual question answering dataset based on English open-source datasets and adopt multiple task instructions. To evaluate the performance of LVLMs across various languages, we construct four multilingual benchmarks for 10 languages, based on English open-source benchmarks. Experimental results show that LRM-LLaVA achieves competitive performance compared to other multilingual LVLMs of similar parameters.

</details>

---

## 145. Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines

- [ ] Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines | https://ojs.aaai.org/index.php/AAAI/article/view/34653

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34653

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-augmented generation (RAG) has emerged to address the knowledge-intensive visual question answering (VQA) task. Current methods mainly employ separate retrieval and generation modules to acquire external knowledge and generate answers, respectively. We propose ReAuSE, an alternative to the previous RAG model for the knowledge-based VQA task, which seamlessly integrates knowledge retriever into the generative multi-modal large language model, serving as a built-in search engine. Specifically, our model functions both as a generative retriever and an accurate answer generator. It not only helps retrieve documents from the knowledge base by producing identifier for each document, but it also answers visual questions based on the retrieved documents. Furthermore, we also propose a reinforced retrieval calibration module from relevance feedback to improve retrieval performance and align with the preferences for accurate answer generation. Extensive experiments on two representative OKVQA and A-OKVQA datasets demonstrate significant improvements ranging from 2.9% to 9.6% across all evaluation metrics when compared to strong baselines.

</details>

---

## 146. GNS: Solving Plane Geometry Problems by Neural-Symbolic Reasoning with Multi-Modal LLMs

- [ ] GNS: Solving Plane Geometry Problems by Neural-Symbolic Reasoning with Multi-Modal LLMs | https://ojs.aaai.org/index.php/AAAI/article/view/34679

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34679

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the outstanding capabilities of Large Language Models (LLMs), 
solving math word problems (MWP) has greatly progressed, achieving higher performance on several benchmark datasets. 
However, it is more challenging to solve plane geometry problems (PGPs) due to the necessity of understanding, reasoning and computation on two modality data including both geometry diagrams and textual questions, where Multi-Modal Large Language Models (MLLMs) have not been extensively explored.
Previous works simply regarded a plane geometry problem as multi-modal QA task, which ignored the importance of explicit parsing geometric elements from problems. 
To tackle this limitation, we propose to solve plane Geometry problems by Neural-Symbolic reasoning with MLLMs (GNS). 
We first leverage an MLLM to understand PGPs through knowledge prediction and symbolic parsing, next perform mathematical reasoning to obtain solutions, last adopt a symbolic solver to compute answers. 
Correspondingly, we introduce the largest PGPs dataset GNS-260K with multiple annotations including symbolic parsing, understanding, reasoning and computation. 
In experiments, our Phi3-Vision-based MLLM wins the first place on the PGPs solving task of MathVista benchmark, outperforming GPT-4o, Gemini Ultra and other much larger MLLMs.
While LLaVA-13B-based MLLM markedly exceeded other close-source and open-source MLLMs on the MathVerse benchmark and also achieved the new SOTA on GeoQA dataset.

</details>

---

## 147. VERO: Verification and Zero-Shot Feedback Acquisition for Few-Shot Multimodal Aspect-Level Sentiment Classification

- [ ] VERO: Verification and Zero-Shot Feedback Acquisition for Few-Shot Multimodal Aspect-Level Sentiment Classification | https://ojs.aaai.org/index.php/AAAI/article/view/34707

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34707

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Deep learning approaches for multimodal aspect-level sentiment classification (MALSC) often require extensive data, which is costly and time-consuming to obtain. To mitigate this, current methods typically fine-tune small-scale pretrained models like BERT and BART with few-shot examples. While these models have shown success, Large Vision-Language Models (LVLMs) offer significant advantages due to their greater capacity and ability to understand nuanced language in both zero-shot and few-shot settings. However, there is limited work on fine-tuning LVLMs for MALSC. A major challenge lies in selecting few-shot examples that effectively capture the underlying patterns in data for these LVLMs. To bridge this research gap, we propose an acquisition function designed to select challenging samples for the few-shot learning of LVLMs for MALSC. We compare our approach, Verification and ZERO-shot feedback acquisition (VERO), with diverse acquisition functions for few-shot learning in MALSC. Our experiments show that VERO outperforms prior methods, achieving an F1 score improvement of up to 6.07% on MALSC benchmark datasets.

</details>

---

## 148. RoVRM: A Robust Visual Reward Model Optimized via Auxiliary Textual Preference Data

- [ ] RoVRM: A Robust Visual Reward Model Optimized via Auxiliary Textual Preference Data | https://ojs.aaai.org/index.php/AAAI/article/view/34721

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34721

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) often fail to align with human preferences, leading to issues like generating misleading content without proper visual context (also known as hallucination). A promising solution to this problem is using human-preference alignment techniques, such as best-of-n sampling and reinforcement learning. However, these techniques face the difficulty arising from the scarcity of visual preference data, which is required to train a visual reward model (VRM). In this work, we continue the line of research. We present a Robust Visual Reward Model (RoVRM) which improves human-preference alignment for LVLMs. RoVRM leverages auxiliary textual preference data through a three-phase progressive training and optimal transport-based preference data selection to effectively mitigate the scarcity of visual preference data. We experiment with RoVRM on the commonly used vision-language tasks based on the LLaVA-1.5-7B and -13B models. Experimental results demonstrate that RoVRM consistently outperforms traditional VRMs.  Furthermore, our three-phase progressive training and preference data selection approaches can yield consistent performance gains over ranking-based alignment techniques, such as direct preference optimization.

</details>

---

## 149. Detecting and Mitigating Hallucination in Large Vision Language Models via Fine-Grained AI Feedback

- [ ] Detecting and Mitigating Hallucination in Large Vision Language Models via Fine-Grained AI Feedback | https://ojs.aaai.org/index.php/AAAI/article/view/34744

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34744

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapidly developing Large Vision Language Models (LVLMs) still face the hallucination phenomena where the generated responses do not align with the given contexts, significantly restricting the usages of LVLMs. Most previous work detects and mitigates hallucination at the coarse-grained level or requires expensive annotation (e.g., labeling by human experts or proprietary models). To address these issues, we propose detecting and mitigating hallucinations in LVLMs via fine-grained AI feedback. The basic idea is that we generate a small-size sentence-level hallucination annotation dataset by proprietary models, whereby we train a detection model which can perform sentence-level hallucination detection. Then, we propose a detect-then-rewrite pipeline to automatically construct preference dataset for hallucination mitigation training. Furthermore, we propose differentiating the severity of hallucinations, and introducing a Hallucination Severity-Aware Direct Preference Optimization (HSA-DPO) which prioritizes the mitigation of critical hallucination in LVLMs by incorporating the severity of hallucinations into preference learning. Extensive experiments on hallucination detection and mitigation benchmarks demonstrate that our method sets a new state-of-the-art in hallucination detection on MHaluBench, surpassing GPT-4V and Gemini, and reduces the hallucination rate by 36.1% on AMBER and 76.3% on Object HalBench compared to the base model.

</details>

---

## 150. Math-PUMA: Progressive Upward Multimodal Alignment to Enhance Mathematical Reasoning

- [ ] Math-PUMA: Progressive Upward Multimodal Alignment to Enhance Mathematical Reasoning | https://ojs.aaai.org/index.php/AAAI/article/view/34815

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34815

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) excel in solving text-based mathematical problems, but they struggle with mathematical diagrams since they are primarily trained on natural scene images. For humans, visual aids generally enhance problem-solving, but MLLMs perform worse as information shifts from textual to visual modality. This decline is mainly due to their shortcomings in aligning images and text.
To tackle aforementioned challenges, we propose Math-PUMA, a methodology focused on Progressive Upward Multimodal Alignment. This approach is designed to improve the mathematical reasoning skills of MLLMs through a  three-stage training process, with the second stage being the critical alignment stage.
We first enhance the language model's mathematical reasoning capabilities with extensive set of textual mathematical problems. We then construct a multimodal dataset with varying degrees of textual and visual information, creating data pairs by presenting each problem in at least two forms. By leveraging the Kullback-Leibler (KL) divergence of next-token prediction distributions to align visual and textual modalities, consistent problem-solving abilities are ensured. Finally, we utilize multimodal instruction tuning for MLLMs with high-quality multimodal data. 
Experimental results on multiple mathematical reasoning benchmarks demonstrate that the MLLMs trained with Math-PUMA surpass most open-source MLLMs. Our approach effectively narrows the performance gap for problems presented in different modalities.

</details>

---

## 151. RUNA: Object-Level Out-of-Distribution Detection via Regional Uncertainty Alignment of Multimodal Representations

- [ ] RUNA: Object-Level Out-of-Distribution Detection via Regional Uncertainty Alignment of Multimodal Representations | https://ojs.aaai.org/index.php/AAAI/article/view/34841

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34841

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Enabling object detectors to recognize out-of-distribution (OOD) objects is vital for building reliable systems. A primary obstacle stems from the fact that models frequently do not receive supervisory signals from unfamiliar data, leading to overly confident predictions regarding OOD objects. Despite previous progress that estimates OOD uncertainty based on the detection model and in-distribution (ID) samples, we explore using pre-trained vision-language representations for object-level OOD detection. We first discuss the limitations of applying image-level CLIP-based OOD detection methods to object-level scenarios. Building upon these insights, we propose RUNA, a novel framework that leverages a dual encoder architecture to capture rich contextual information and employs a regional uncertainty alignment mechanism to distinguish ID from OOD objects effectively. We introduce a few-shot fine-tuning approach that aligns region-level semantic representations to further improve the model's capability to discriminate between similar objects. Our experiments show that RUNA substantially surpasses state-of-the-art methods in object-level OOD detection, particularly in challenging scenarios with diverse and complex object instances.

</details>

---

## 152. Internal Activation Revision: Safeguarding Vision Language Models Without Parameter Update

- [ ] Internal Activation Revision: Safeguarding Vision Language Models Without Parameter Update | https://ojs.aaai.org/index.php/AAAI/article/view/34954

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34954

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Warning: This paper contains offensive content that may disturb some readers. Vision-language models (VLMs) demonstrate strong multimodal capabilities but have been found to be more susceptible to generating harmful content compared to their backbone large language models (LLMs). Our investigation reveals that the integration of images significantly shifts the model's internal activations during the forward pass, diverging from those triggered by textual input. Moreover, the safety alignments of LLMs embedded within VLMs are not sufficiently robust to handle the activations discrepancies, making the models vulnerable to even the simplest jailbreaking attacks. To address this issue, we propose an internal activation revision approach that efficiently revises activations during generation, steering the model toward safer outputs. Our framework incorporates revisions at both the layer and head levels, offering control over the model's generation at varying levels of granularity. In addition, we explore three strategies for constructing positive and negative samples and two approaches for extracting revision vectors, resulting in different variants of our method. Comprehensive experiments demonstrate that the internal activation revision method significantly improves the safety of widely used VLMs, reducing attack success rates by an average of 48.94%, 34.34%, 43.92%, and 52.98% on SafeBench, Safe-Unsafe, Unsafe, and MM-SafetyBench, respectively, while minimally impacting model helpfulness.

</details>

---

## 153. Retention Score: Quantifying Jailbreak Risks for Vision Language Models

- [ ] Retention Score: Quantifying Jailbreak Risks for Vision Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34956

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34956

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of Vision-Language Models (VLMs) is significant advancement in integrating computer vision with Large Language Models (LLMs) to enhance multi-modal machine learning capabilities. However, this progress has made VLMs vulnerable to advanced adversarial attacks, raising concerns about reliability. Objective of this paper is to assess resilience of VLMs against jailbreak attacks that can compromise model safety compliance and result in harmful outputs. To evaluate VLM's ability to maintain robustness against adversarial input perturbations, we propose novel metric called \textbf{Retention Score}. Retention Score is multi-modal evaluation metric that includes Retention-I and Retention-T scores for quantifying jailbreak risks in visual and textual components of VLMs. Our process involves generating synthetic image-text pairs using conditional diffusion model. These pairs are then predicted for toxicity score by VLM alongside toxicity judgment classifier. By calculating margin in toxicity scores, we can quantify robustness of VLM in attack-agnostic manner. Our work has four main contributions. First, we prove that Retention Score can serve as certified robustness metric. Second, we demonstrate that most VLMs with visual components are less robust against jailbreak attacks than corresponding plain VLMs. Additionally, we evaluate black-box VLM APIs and find that security settings in Google Gemini significantly affect score and robustness. Moreover, robustness of GPT4V is similar to medium settings of Gemini. Finally, our approach offers time-efficient alternative to existing adversarial attack methods and provides consistent model robustness rankings when evaluated on VLMs including MiniGPT-4, InstructBLIP, and LLaVA.

</details>

---

## 154. MMJ-Bench: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models

- [ ] MMJ-Bench: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/34983

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/34983

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Vision-Language Models (VLMs), have shown exceptional performance in many real-world tasks. However, VLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model’s safety alignment to elicit harmful responses. The threat of jailbreak attacks on VLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that VLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce MMJ-Bench, a unified pipeline for evaluating jailbreak attacks and defense techniques for VLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA VLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for VLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.

</details>

---

## 155. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection

- [ ] PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection | https://ojs.aaai.org/index.php/AAAI/article/view/35003

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35003

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

</details>

---

## 156. Leveraging Computer Vision and Visual LLMs for Cost-Effective and Consistent Street Food Safety Assessment in Kolkata India

- [ ] Leveraging Computer Vision and Visual LLMs for Cost-Effective and Consistent Street Food Safety Assessment in Kolkata India | https://ojs.aaai.org/index.php/AAAI/article/view/35008

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35008

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Ensuring street food safety in developing countries is crucial due to the high prevalence of foodborne illnesses.  Traditional methods of food safety assessments face challenges such as resource constraints, logistical issues, and subjective biases influenced by surveyors personal lived experiences, particularly when interacting with local communities. For instance, a local food safety inspector may inadvertently overrate the quality of infrastructure due to prior familiarity or past purchases, thereby compromising objective assessment.  This subjectivity highlights the necessity for technologies that reduce human biases and enhance the accuracy of survey data across various domains.
This paper proposes a novel approach based on a combination of Computer Vision and a lightweight Visual Large Language Model (VLLM) to automate the detection and analysis of critical food safety infrastructure in street food vendor environments at a field experiment in Kolkata, India. The system utilises a three-stage object extraction pipeline from the video to identify, extract and select unique representations of critical elements such as hand-washing stations, dishwashing areas, garbage bins, and water tanks. These four infrastructure items are crucial for maintaining safe food practices, irrespective of the specific methods employed by the vendors. A VLLM then analyses the extracted representations to assess compliance with food safety standards. Notably, over half of the pipeline can be processed using a user's smartphone, significantly reducing government server workload. By leveraging this decentralised approach, the proposed system decreases the analysis cost by many orders of magnitude compared to alternatives like ChatGPT or Claude 3.5. Additionally, processing data on local government servers provides better privacy and security than cloud platforms, addressing critical ethical considerations. This automated approach significantly improves efficiency, consistency, and scalability, providing a robust solution to enhance public health outcomes in developing regions.

</details>

---

## 157. Enhancing Vision-Language Models with Morphological and Taxonomic Knowledge: Towards Coral Recognition for Ocean Health

- [ ] Enhancing Vision-Language Models with Morphological and Taxonomic Knowledge: Towards Coral Recognition for Ocean Health | https://ojs.aaai.org/index.php/AAAI/article/view/35023

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35023

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Coral reefs play a crucial role in marine ecosystems, offering a nutrient-rich environment and safe shelter for numerous marine species. Automated coral image recognition aids in monitoring ocean health at a scale without experts' manual effort. Recently, large vision-language models like CLIP have greatly enhanced zero-shot and low-shot classification capabilities for various visual tasks. However, these models struggle with fine-grained coral-related tasks due to a lack of specific knowledge. To bridge this gap, we compile a fine-grained coral image dataset consisting of 16,659 images with taxonomy labels (from Kingdom to Species), accompanied by morphology-specific text descriptions for each species. Based on the dataset, we propose CORAL-Adapter, integrating two complementary kinds of coral-specific knowledge (biological taxonomy and coral morphology) with general knowledge learned by CLIP. CORAL-Adapter is a simple yet powerful extension of CLIP with only a few parameter updates and can be used as a plug-and-play module with various CLIP-based methods.  We show improvements in accuracy across diverse coral recognition tasks, e.g., recognizing corals unseen during training that are prone to bleaching or originate from different oceans.

</details>

---

## 158. UrbanVLP: Multi-Granularity Vision-Language Pretraining for Urban Socioeconomic Indicator Prediction

- [ ] UrbanVLP: Multi-Granularity Vision-Language Pretraining for Urban Socioeconomic Indicator Prediction | https://ojs.aaai.org/index.php/AAAI/article/view/35024

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35024

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Urban socioeconomic indicator prediction aims to infer various metrics related to sustainable development in diverse urban landscapes using data-driven methods. However, prevalent pretrained models, particularly those reliant on satellite imagery, face dual challenges. Firstly, concentrating solely on macro-level patterns from satellite data may introduce bias, lacking nuanced details at micro levels, such as architectural details at a place. Secondly, the text generated by the precursor work UrbanCLIP, which fully utilizes the extensive knowledge of LLMs, frequently exhibits issues such as hallucination and homogenization, resulting in a lack of reliable quality. In response to these issues, we devise a novel framework entitled UrbanVLP based on Vision-Language Pretraining. Our UrbanVLP seamlessly integrates multi-granularity information from both macro (satellite) and micro (street-view) levels, overcoming the limitations of prior pretrained models. 
Moreover, it introduces automatic text generation and calibration, providing a robust guarantee for producing high-quality text descriptions of urban imagery. Rigorous experiments conducted across six socioeconomic indicator prediction tasks underscore its superior performance.

</details>

---

## 159. Open-World Multimodal Understanding and Generation with Efficiently Finetuned Foundation Models

- [ ] Open-World Multimodal Understanding and Generation with Efficiently Finetuned Foundation Models | https://ojs.aaai.org/index.php/AAAI/article/view/35101

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35101

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the astonishing ability of different pretrained foundation models (e.g., large language models (LLMs), vision-language models, diffusion models), today’s AI research and development tendency has been revolutionized. In this talk, I will answer two questions: Q1: How can we efficiently train or fine-tune foundation models? Q2: How can we build strong open-world multimodal understanding and generation models with these pretrained foundation models?

</details>

---

## 160. Advancements in AI for Reasoning with Complex Data

- [ ] Advancements in AI for Reasoning with Complex Data | https://ojs.aaai.org/index.php/AAAI/article/view/35106

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35106

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Artificial intelligence has made remarkable progress in reasoning over complex, structured, multimodal, and multilingual data, addressing critical challenges in domains such as finance and healthcare. This abstract underscores key advancements in tabular reasoning, temporal analysis, and structured multimodal reasoning.

Key contributions include the development of TempTabQA, a benchmark for temporal question answering, along with novel methods for enhancing temporal reasoning in large language models (LLMs). Additionally, a framework for evaluating mathematical reasoning in financial documents has been introduced, establishing robust techniques for interpreting time-sensitive and quantitative data. Building on these foundations, we have developed hybrid SQL-text adaptive reasoning models (H-STAR) and knowledge-aware reasoning techniques for semi-structured tables (MMTabQA), enabling precise and efficient handling of complex queries.

In the vision-language domain, our contributions include advancements in spatial reasoning for geographic data (MAPWise), methods to improve robustness in chart interpretation (FlowVQA), and evaluations of LLMs’ ability to understand visual data, such as charts. Furthermore, we have addressed challenges in multilingual and cross-modal robustness through innovations such as multilingual table synchronization (InfoSync), concurrent robustness evaluations across languages and modalities, and numerical reasoning in tabular data.

Our work aims to enhance reasoning on dynamically evolving data using hybrid LLM-SQL queries, symbolic query generation, and multi-table retrieval techniques. We also plan to tackle challenges in interpreting hierarchical table structures, analyzing multiple complex chart types, and exploring diverse map types, while advancing real-world multimodal data analysis. Additionally, we plan to improve table generation in both closed/open-book scenarios and refine evaluation frameworks for structured tasks. These advancements demonstrate the potential of AI in tackling complex, multimodal data and delivering impactful real-world solutions.

</details>

---

## 161. From Large Language Models to Large Action Models: Reasoning and Planning with Physical World Knowledge

- [ ] From Large Language Models to Large Action Models: Reasoning and Planning with Physical World Knowledge | https://ojs.aaai.org/index.php/AAAI/article/view/35109

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35109

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Large Language Models excel in language processing, Large Agent Models are designed to interact with the environment. This transition poses significant challenges in understanding lower-level visual details, and long-horizon reasoning for effective goal interpretation and decision-making. Despite the impressive performance of LLMs/VLMs on various benchmarks, these models perceive images as bags of words (semantic concepts). In detail, they use semantic understanding as a shortcut but lack the ability to recognize geometric structures or solve spatial problems such as mazes. To interact with the physical world, we focus on two dimensions: (1) From high-level semantic to low-level geometric understanding: We introduce a low-level visual description language that serves as geometric tokens, allowing the abstraction of multimodal low-level geometric structures. (2) From fast-thinking to slow-thinking: We propose to quantify long-horizon reasoning by incorporating Markov Decision Process (MDP) based decision-making. The key difference between language models and agent models lies in their decision-making capabilities. This fundamental difference necessitates a shift in how we approach the development of large agent models, focusing on both geometric understanding and long-term planning to create more capable embodied AI agents.

</details>

---

## 162. Scalable Vision-Language Understanding and Generation

- [ ] Scalable Vision-Language Understanding and Generation | https://ojs.aaai.org/index.php/AAAI/article/view/35130

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35130

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language models have shown remarkable potential, yet creating scalable systems that can effectively understand and generate across modalities remains challenging. This talk will present our contributions to advancing scalable vision-language systems, focusing on three key themes: (1) efficient vision-language understanding, including our work on temporal perceiving video-language pre-training and knowledge-enhanced zero-shot retrieval; (2) scalable generation frameworks, encompassing our innovations in zero-shot captioning and co-speech gesture generation; and (3) practical applications and deployments of these technologies. We will discuss how these advances have enabled both better performance and improved efficiency in real-world scenarios, and explore future directions for scalable multimodal systems.

</details>

---

## 163. An Application-Agnostic Automatic Target Recognition System Using Vision Language Models

- [ ] An Application-Agnostic Automatic Target Recognition System Using Vision Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/35154

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35154

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a novel Automatic Target Recognition (ATR) system using open-vocabulary object detection and classification models. A primary advantage of this approach is that target classes can be defined just before runtime by a non-technical end user, using either a few natural language text descriptions of the target, or a few image exemplars, or both. Nuances in the desired targets can be expressed in natural language, which is useful for unique targets with little or no training data. We also implemented a novel combination of several techniques to improve performance, such as leveraging the additional information in the sequence of overlapping frames to perform tubelet identification (i.e., sequential bounding box matching), bounding box re-scoring, and tubelet linking. Additionally, we developed a technique to visualize the aggregate output of many overlapping frames as a mosaic of the area scanned during the aerial surveillance or reconnaissance, and a kernel density estimate (or heatmap) of the detected targets. We initially applied this ATR system to the use case of detecting and clearing unexploded ordinance on airfield runways and we are currently extending our research to other real-world applications.

</details>

---

## 164. Stress-Testing of Multimodal Models in Medical Image-Based Report Generation

- [ ] Stress-Testing of Multimodal Models in Medical Image-Based Report Generation | https://ojs.aaai.org/index.php/AAAI/article/view/35203

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35203

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal models, namely vision-language models, present unique possibilities through the seamless integration of different information mediums for data generation. These models mostly act as a black-box, making them lack transparency and explicability. Reliable results require accountable and trustworthy Artificial Intelligence (AI), namely when in use for critical tasks, such as the automatic generation of medical imaging reports for healthcare diagnosis. By exploring stress-testing techniques, multimodal generative models can become more transparent by disclosing their shortcomings, further supporting their responsible usage in the medical field.

</details>

---

## 165. Leveraging Textual Memory and Key Frame Reasoning for Full Video Understanding Using Off-the-Shelf LLMs and VLMs (Student Abstract)

- [ ] Leveraging Textual Memory and Key Frame Reasoning for Full Video Understanding Using Off-the-Shelf LLMs and VLMs (Student Abstract) | https://ojs.aaai.org/index.php/AAAI/article/view/35248

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35248

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

To address the limitations of current Large-scale Video-Language Models (LVLMs) in fine-grained understanding and long-term temporal memory, we propose a novel video understanding approach that integrates a Vision Language Model (VLM) and a Large Language Model (LLM) with a textual memory mechanism to ensure continuity and contextual coherence. In addition, we introduce a novel evaluation metric, VAD-Score (Video Automated Description Score), to assess precision, recall, and F1 scores for events, subjects, and objects. Our approach delivers competitive results on a diverse set of videos from the DREAM-1K dataset, spanning categories such as live-action, animation, shorts, stock, and YouTube, with a focus on fine-grained comprehension.

</details>

---

## 166. Multimodal Commonsense Knowledge Distillation for Visual Question Answering (Student Abstract)

- [ ] Multimodal Commonsense Knowledge Distillation for Visual Question Answering (Student Abstract) | https://ojs.aaai.org/index.php/AAAI/article/view/35320

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35320

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing Multimodal Large Language Models (MLLMs) and Visual Language Pretrained Models (VLPMs) have shown remarkable performances in general Visual Question Answering (VQA). However, these models struggle with VQA questions that require external commonsense knowledge due to the challenges in generating high-quality prompts and the high computational costs of fine-tuning. In this work, we propose a novel graph-based multimodal commonsense knowledge distillation framework that constructs a unified relational graph over commonsense knowledge, visual objects and questions through a Graph Convolutional Network (GCN) following a teacher-student environment. This proposed framework is flexible with any type of teacher and student models without further fine-tuning, and has achieved competitive performances on the ScienceQA dataset. The code is in https://github.com/adlnlp/MCKDVQA.

</details>

---

## 167. Utilizing Vision-Language Models for Detection of Leaf-Based Diseases in Tomatoes

- [ ] Utilizing Vision-Language Models for Detection of Leaf-Based Diseases in Tomatoes | https://ojs.aaai.org/index.php/AAAI/article/view/35327

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35327

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Leaf based diseases in tomatoes such as early blight, late blight, and septoria leaf spot, pose a significant threat to global food security and have substantial economic impacts. Early detection of these diseases is crucial for improving crop yields. This paper explores the use of vision-language models (VLMs) for detecting tomato leaf diseases by fine-tuning a pre-trained model on a large dataset of tomato leaf images with corresponding disease annotations. This approach enhances disease detection accuracy and enables multi-modal learning, real-time monitoring, and automated diagnosis, offering promising applications in precision farming and food production.

</details>

---

## 168. Falcon Medical Visual Question Answering

- [ ] Falcon Medical Visual Question Answering | https://ojs.aaai.org/index.php/AAAI/article/view/35346

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35346

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) bridge the gap between visual and textual data, enabling multimodal tasks like Visual Question Answering (VQA). Leveraging this capability, Medical VQA systems have the potential to transform clinical decision-making by allowing healthcare providers to query medical images—such as X-rays, MRIs, and CT scans—and receive rapid, informed responses, thereby speeding up diagnoses and treatment planning. In this work, we introduce Falcon Med-VQA, a generative VQA system meticulously designed to interpret visual and textual medical data and generate free-form answers to medical questions. By leveraging a vision language model and a dynamic model selection mechanism, Falcon Med-VQA ensures relevance and precision in its responses. The system is equipped with an intuitive user interface that displays top answers with Confidence Scores (CF), enhances explainability through medical terminology extraction, and offers attention map visualizations for improved interpretability. Our experiments demonstrate that Falcon Med-VQA achieves comparable performance against specialized models and outperforms recent generative approaches in a key benchmark.

</details>

---

## 169. StarVector: Generating Scalable Vector Graphics Code from Images and Text

- [ ] StarVector: Generating Scalable Vector Graphics Code from Images and Text | https://ojs.aaai.org/index.php/AAAI/article/view/35369

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35369

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scalable Vector Graphics (SVG) have become integral to modern image rendering applications due to their infinite scalability and versatility, especially in graphic design and web development. SVGs are essentially long strings of code that adhere to a structured syntax with validity constraints. With the rise of large language models, which excel at generating code in various languages, we aim to generate SVG code in a similar way. Our findings show that a vision-language model can be conditioned to produce valid SVG code that closely resembles input images, effectively enabling vectorization. Additionally, we harness the rich SVG syntax, encompassing all possible primitives—such as lines, paths, polygons, text, and effects like color gradients—that previous methods often missed. We briefly explain how the StarVector model operates, primarily leveraging a vision-language transformer architecture to generate SVG code. We also detail our training and inference procedures. Finally, we provide an interactive demo that allows users to input an image and generate its SVG code autoregressively, featuring real-time rendering that visually demonstrates the SVG generation process.

</details>

---

## 170. SWIFT: A Scalable Lightweight Infrastructure for Fine-Tuning

- [ ] SWIFT: A Scalable Lightweight Infrastructure for Fine-Tuning | https://ojs.aaai.org/index.php/AAAI/article/view/35383

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35383

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent development in Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) have achieved superior performance and generalization capabilities, covered extensive areas of traditional tasks. However, existing large model training frameworks support only a limited number of models and techniques, particularly lacking in support for new models, which makes fine-tuning LLMs challenging for most developers. Therefore, we develop SWIFT, a customizable one-stop infrastructure for large models. With support of over 350+ LLMs and 80+ MLLMs, SWIFT stands as the open-source framework that provide the most comprehensive support for fine-tuning large models. In particular, it is the first training framework that provides systematic support for MLLMs. Moreover, SWIFT integrates post-training processes such as inference, evaluation, and quantization, to facilitate fast adoptions of large models in various application scenarios, offering helpful utilities like benchmark comparisons among different training techniques.

</details>

---

## 171. Federated Weakly Supervised Video Anomaly Detection with Multimodal Prompt

- [ ] Federated Weakly Supervised Video Anomaly Detection with Multimodal Prompt | https://ojs.aaai.org/index.php/AAAI/article/view/35398

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35398

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video anomaly detection (VAD) aims at locating the abnormal events in videos. Recently, the Weakly Supervised VAD has made great progress, which only requires video-level annotations when training. In practical applications, different institutions may have different types of abnormal videos. However, the abnormal videos cannot be circulated on the internet due to privacy protection. To train a more generalized anomaly detector that can identify various anomalies, it is reasonable to introduce federated learning into WSVAD. In this paper, we propose Global and Local Context-driven Federated Learning, a new paradigm for privacy protected weakly supervised video anomaly detection. Specifically, we utilize the vision-language association of CLIP to detect whether the video frame is abnormal. Instead of leveraging handcrafted text prompts for CLIP, we propose a text prompt generator. The generated prompt is simultaneously influenced by text and visual. On the one hand, the text provides global context related to anomaly, which improves the model's ability of generalization. On the other hand, the visual provides personalized local context because different clients may have videos with different types of anomalies or scenes. The generated prompt ensures global generalization while processing personalized data from different clients. Extensive experiments show that the proposed method achieves remarkable performance.

</details>

---

## 172. IAA: Inner-Adaptor Architecture Empowers Frozen Large Language Model with Multimodal Capabilities

- [ ] IAA: Inner-Adaptor Architecture Empowers Frozen Large Language Model with Multimodal Capabilities | https://ojs.aaai.org/index.php/AAAI/article/view/35400

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35400

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the field of multimodal large language models (MLLMs), common methods typically involve unfreezing the language model during training to foster profound visual understanding. However, the fine-tuning of such models with vision-language data often leads to a diminution of their natural language processing (NLP) capabilities. To avoid this performance degradation, a straightforward solution is to freeze the language model while developing multimodal competencies. Unfortunately, previous works have not attained satisfactory outcomes. Building on the strategy of freezing the language model, we conduct thorough structural exploration and introduce the Inner-Adaptor Architecture (IAA). Specifically, the architecture incorporates multiple multimodal adaptors at varying depths within the large language model to facilitate direct interaction with the inherently text-oriented transformer layers, thereby enabling the frozen language model to acquire multimodal capabilities. Unlike previous approaches of freezing language models that require large-scale aligned data, our proposed architecture is able to achieve superior performance on small-scale datasets. We conduct extensive experiments to improve the general multimodal capabilities and visual grounding abilities of the MLLM. Our approach remarkably outperforms previous state-of-the-art methods across various vision-language benchmarks without sacrificing performance on NLP tasks. Code and models will be released.

</details>

---

## 173. Pre-Trained Vision-Language Models as Noisy Partial Annotators

- [ ] Pre-Trained Vision-Language Models as Noisy Partial Annotators | https://ojs.aaai.org/index.php/AAAI/article/view/35417

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35417

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In noisy partial label learning, each training sample is associated with a set of candidate labels, and the ground-truth label may be contained within this set. With the emergence of powerful pre-trained vision-language models, e.g. CLIP, it is natural to consider using these models to automatically label training samples instead of relying on laborious manual annotation. In this paper, we investigate the pipeline of learning with CLIP annotated noisy partial labels and propose a novel collaborative consistency regularization method, in which we simultaneously train two neural networks, which collaboratively purify training labels for each other, called Co-Pseudo-Labeling, and perform consistency regularization between label and representation levels. For instance-dependent noise that embodies the underlying patterns of the pre-trained model, our method employs multiple mechanisms to avoid overfitting to noisy annotations, effectively mines information from potentially noisy sample set while iteratively optimizing both representations and pseudo-labels during the training process. Comparison experiments with various kinds of annotations and weakly supervised methods, as well as other pre-trained model application methods demonstrates the effectiveness of method and the feasibility of incorporating weakly supervised learning into the distillation of pre-trained models.

</details>

---

## 174. Beyond Accuracy: On the Effects of Fine-Tuning Towards Vision-Language Model’s Prediction Rationality

- [ ] Beyond Accuracy: On the Effects of Fine-Tuning Towards Vision-Language Model’s Prediction Rationality | https://ojs.aaai.org/index.php/AAAI/article/view/35421

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35421

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs), such as CLIP, have already seen widespread applications. Researchers actively engage in further fine-tuning VLMs in safety-critical domains. In these domains, prediction rationality is crucial: the prediction should be correct and based on valid evidence. Yet, for VLMs, the impact of fine-tuning on prediction rationality is seldomly investigated. To study this problem, we proposed two new metrics called Prediction Trustworthiness and Inference Reliability. We conducted extensive experiments on various settings and observed some interesting phenomena. On the one hand, we found that the well-adopted fine-tuning methods led to more correct predictions based on invalid evidence. This potentially undermines the trustworthiness of correct predictions from fine-tuned VLMs. On the other hand, having identified valid evidence of target objects, fine-tuned VLMs were more likely to make correct predictions. Moreover, the findings are also consistent under distributional shifts and across various experimental settings. We hope our research offer fresh insights to VLM fine-tuning.

</details>

---

## 175. Pilot: Building the Federated Multimodal Instruction Tuning Framework

- [ ] Pilot: Building the Federated Multimodal Instruction Tuning Framework | https://ojs.aaai.org/index.php/AAAI/article/view/35476

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35476

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we explore a novel federated multimodal instruction tuning task(FedMIT), which is significant for collaboratively fine-tuning MLLMs on different types of multimodal instruction data on distributed devices. To solve the new task, we propose a federated multimodal instruction tuning framework(Pilot). Our framework integrates two-stage of ``adapter on adapter” into the connector of the vision encoder and the LLM. In stage 1, we extract task-specific features and client-specific features from visual information. In stage 2, we build the cross-task Mixture-of-Adapters(CT-MoA) module to perform cross-task interaction. Each client can not only capture personalized information of local data and learn task-related multimodal information, but also learn general knowledge from other tasks. In addition, we introduce an adaptive parameter aggregation strategy for text training parameters, which optimizes parameter aggregation by calculating weights based on the euclidean distance between parameters, so that parameter aggregation can benefit from positive effects to the greatest extent while effectively reducing negative effects. Our framework can collaboratively exploit distributed data from different local clients to learn cross-task knowledge without being affected by the task heterogeneity during instruction tuning. The effectiveness of our method is verified in two different cross-task scenarios.

</details>

---

## 176. Explanation Bottleneck Models

- [ ] Explanation Bottleneck Models | https://ojs.aaai.org/index.php/AAAI/article/view/35495

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35495

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent concept-based interpretable models have succeeded in providing meaningful explanations by pre-defined concept sets. However, the dependency on the pre-defined concepts restricts the application because of the limited number of concepts for explanations. This paper proposes a novel interpretable deep neural network called explanation bottleneck models (XBMs). XBMs generate a text explanation from the input without pre-defined concepts and then predict a final task prediction based on the generated explanation by leveraging pre-trained vision-language encoder-decoder models. To achieve both the target task performance and the explanation quality, we train XBMs through the target task loss with the regularization penalizing the explanation decoder via the distillation from the frozen pre-trained decoder. Our experiments, including a comparison to state-of-the-art concept bottleneck models, confirm that XBMs provide accurate and fluent natural language explanations without pre-defined concept sets.

</details>

---

## 177. CLIP-CID: Efficient CLIP Distillation via Cluster-Instance Discrimination

- [ ] CLIP-CID: Efficient CLIP Distillation via Cluster-Instance Discrimination | https://ojs.aaai.org/index.php/AAAI/article/view/35505

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/35505

- **Conference**: AAAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) has achieved excellent performance over a wide range of tasks. However, the effectiveness of CLIP heavily relies on a substantial corpus of pre-training data, resulting in notable consumption of computational resources. Although knowledge distillation has been widely applied in single modality models, how to efficiently expand knowledge distillation to vision-language foundation models with extensive data remains relatively unexplored. In this paper, we introduce CLIP-CID, a novel distillation mechanism that effectively transfers knowledge from a large vision-language foundation model to a smaller model. We initially propose a simple but efficient image semantic balance method to reduce transfer learning bias and improve distillation efficiency. This method filters out 43.7% of image-text pairs from the LAION400M while maintaining superior performance. After that, we leverage cluster-instance discrimination to facilitate knowledge transfer from the teacher model to the student model, thereby empowering the student model to acquire a holistic semantic comprehension of the pre-training data. Experimental results demonstrate that CLIP-CID achieves state-of-the-art performance on various downstream tasks including linear probe and zero-shot classification.

</details>

---

