# ICML 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_icml2025_papers.csv

## 1. CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging

- [ ] CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging | https://icml.cc/virtual/2025/poster/43445

- **Link**: https://icml.cc/virtual/2025/poster/43445

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-task model merging offers a promising paradigm for integrating multiple expert models into a unified system without additional training. Existing state-of-the-art techniques, such as Task Arithmetic and its variants, merge models by accumulating task vectors—defined as the parameter differences between pre-trained and fine-tuned models. However, task vector accumulation is often hindered by knowledge conflicts, where conflicting components across different task vectors can lead to performance degradation during the merging process. To address this challenge, we propose Conflict-Aware Task Merging (CAT Merging) , a novel training-free framework that selectively trims conflict-prone components from the task vectors. CAT Merging introduces several parameter-specific strategies, including projection for linear weights and masking for scaling and shifting parameters in normalization layers. Extensive experiments on vision and vision-language tasks demonstrate that CAT Merging effectively suppresses knowledge conflicts, achieving average accuracy improvements of up to 4.7% (ViT-B/32) and 2.0% (ViT-L/14) over state-of-the-art methods.

</details>

---

## 2. DAMA: Data- and Model-aware Alignment of Multi-modal LLMs

- [ ] DAMA: Data- and Model-aware Alignment of Multi-modal LLMs | https://icml.cc/virtual/2025/poster/43449

- **Link**: https://icml.cc/virtual/2025/poster/43449

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Direct Preference Optimization (DPO) has shown effectiveness in aligning multi-modal large language models (MLLM) with human preferences. However, existing methods exhibit an imbalanced responsiveness to the data of varying hardness, tending to overfit on the easy-to-distinguish data while underfitting on the hard-to-distinguish data. In this paper, we propose Data- and Model-aware DPO (DAMA) to dynamically adjust the optimization process from two key aspects: (1) a data-aware strategy that incorporates data hardness, and (2) a model-aware strategy that integrates real-time model responses. By combining the two strategies, DAMA enables the model to effectively adapt to data with varying levels of hardness.Extensive experiments on five benchmarks demonstrate that DAMA not only significantly enhances the trustworthiness, but also improves the effectiveness over general tasks. For instance, on the Object HalBench, our DAMA-7B reduces response-level and mentioned-level hallucination by 90.0% and 95.3%, respectively, surpassing the performance of GPT-4V.

</details>

---

## 3. Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models

- [ ] Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/43450

- **Link**: https://icml.cc/virtual/2025/poster/43450

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Detailed image captioning is essential for tasks like data generation and aiding visually impaired individuals. High-quality captions require a balance between precision and recall, which remains challenging for current multimodal large language models (MLLMs). In this work, we hypothesize that this limitation stems from weakening and increasingly noisy visual attention as responses lengthen. To address this issue, we propose SPARC (Selective Progressive Attention ReCalibration), a training-free method that enhances the contribution of visual tokens during decoding. SPARC is founded on three key observations: (1) increasing the influence of all visual tokens reduces recall; thus, SPARC selectively amplifies visual tokens; (2) as captions lengthen, visual attention becomes noisier, so SPARC identifies critical visual tokens by leveraging attention differences across time steps; (3) as visual attention gradually weakens, SPARC reinforces it to preserve its influence. Our experiments, incorporating both automated and human evaluations, demonstrate that existing methods improve the precision of MLLMs at the cost of recall. In contrast, our proposed method enhances both precision and recall with minimal computational overhead.

</details>

---

## 4. Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning

- [ ] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning | https://icml.cc/virtual/2025/poster/43454

- **Link**: https://icml.cc/virtual/2025/poster/43454

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Continual multimodal instruction tuning is crucial for adapting Multimodal Large Language Models (MLLMs) to evolving tasks. However, most existing methods adopt a fixed architecture, struggling with adapting to new tasks due to static model capacity. We propose to evolve the architecture under parameter budgets for dynamic task adaptation, which remains unexplored and imposes two challenges: 1) task architecture conflict, where different tasks require varying layer-wise adaptations, and 2) modality imbalance, where different tasks rely unevenly on modalities, leading to unbalanced updates. To address these challenges, we propose a novel Dynamic Mixture of Curriculum LoRA Experts (D-MoLE) method, which automatically evolves MLLM's architecture with controlled parameter budgets to continually adapt to new tasks while retaining previously learned knowledge. Specifically, we propose a dynamic layer-wise expert allocator, which automatically allocates LoRA experts across layers to resolve architecture conflicts, and routes instructions layer-wisely to facilitate knowledge sharing among experts. Then, we propose a gradient-based inter-modal continual curriculum, which adjusts the update ratio of each module in MLLM based on the difficulty of each modality within the task to alleviate the modality imbalance problem. Extensive experiments show that D-MoLE significantly outperforms state-of-the-art baselines, achieving a 15 percent average improvement over the best baseline. To the best of our knowledge, this is the first study of continual learning for MLLMs from an architectural perspective.

</details>

---

## 5. Surrogate Prompt Learning: Towards Efficient and Diverse Prompt Learning for Vision-Language Models

- [ ] Surrogate Prompt Learning: Towards Efficient and Diverse Prompt Learning for Vision-Language Models | https://icml.cc/virtual/2025/poster/43460

- **Link**: https://icml.cc/virtual/2025/poster/43460

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning is a cutting-edge parameter-efficient fine-tuning technique for pre-trained vision-language models (VLMs). Instead of learning a single text prompt, recent works have revealed that learning diverse text prompts can effectively boost the performances on downstream tasks, as the diverse prompted text features can comprehensively depict the visual concepts from different perspectives. However, diverse prompt learning demands enormous computational resources. This efficiency issue still remains unexplored. To achieve efficient and diverse prompt learning, this paper proposes a novel \textbf{Surrogate Prompt Learning (SurPL)} framework. Instead of learning diverse text prompts, SurPL directly generates the desired prompted text features via a lightweight \textbf{Surrogate Feature Generator (SFG)}, thereby avoiding the complex gradient computation procedure of conventional diverse prompt learning. Concretely, based on a basic prompted text feature, SFG can directly and efficiently generate diverse prompted features according to different pre-defined conditional signals. Extensive experiments indicate the effectiveness of the surrogate prompted text features, and show compelling performances and efficiency of SurPL on various benchmarks.

</details>

---

## 6. CtrlSynth: Controllable Image Text Synthesis for Data-Efficient Multimodal Learning

- [ ] CtrlSynth: Controllable Image Text Synthesis for Data-Efficient Multimodal Learning | https://icml.cc/virtual/2025/poster/43494

- **Link**: https://icml.cc/virtual/2025/poster/43494

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pretraining robust vision or multimodal foundation models (e.g., CLIP) relies on large-scale datasets that may be noisy, potentially misaligned, and have long-tail distributions. Previous works have shown promising results in augmenting datasets by generating synthetic samples. However, they only support domain-specific ad hoc use cases (e.g., either image or text only, but not both), and are limited in data diversity due to a lack of fine-grained control over the synthesis process. In this paper, we design a controllable image-text synthesis pipeline, CtrlSynth, for data-efficient and robust multimodal learning. The key idea is to decompose the visual semantics of an image into basic elements, apply user-specified control policies (e.g., remove, add, or replace operations), and recompose them to synthesize images or texts. The decompose and recompose feature in CtrlSynth allows users to control data synthesis in a fine-grained manner by defining customized control policies to manipulate the basic elements. CtrlSynth leverages the capabilities of pretrained foundation models such as large language models or diffusion models to reason and recompose basic elements such that synthetic samples are natural and composed in diverse ways. CtrlSynth is a closed-loop, training-free, and modular framework, making it easy to support different pretrained models. With extensive experiments on 31 datasets spanning different vision and vision-language tasks, we show that CtrlSynth substantially improves zero-shot classification, image-text retrieval, and compositional reasoning performance of CLIP models.

</details>

---

## 7. EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping

- [ ] EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping | https://icml.cc/virtual/2025/poster/43523

- **Link**: https://icml.cc/virtual/2025/poster/43523

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal language modeling has enabled breakthroughs for representation learning, yet remains unexplored in the realm of functional brain data for clinical phenotyping. This paper pioneers EEG-language models (ELMs) trained on clinical reports and 15000 EEGs. We propose to combine multimodal alignment in this novel domain with timeseries cropping and text segmentation, enabling an extension based on multiple instance learning to alleviate misalignment between irrelevant EEG or text segments. Our multimodal models significantly improve over EEG-only models across four clinical evaluations and for the first time enable zero-shot classification as well as retrieval of both neural signals and reports. In sum, these results highlight the potential of ELMs, representing significant progress for clinical applications.

</details>

---

## 8. Divide and Conquer: Exploring Language-centric Tree Reasoning for Video Question-Answering

- [ ] Divide and Conquer: Exploring Language-centric Tree Reasoning for Video Question-Answering | https://icml.cc/virtual/2025/poster/43530

- **Link**: https://icml.cc/virtual/2025/poster/43530

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video Question-Answering (VideoQA) remains challenging in achieving advanced cognitive reasoning due to the uncontrollable and opaque reasoning processes in existing Multimodal Large Language Models (MLLMs). To address this issue, we propose a novel Language-centric Tree Reasoning (LTR) framework that targets on enhancing the reasoning ability of models. In detail, it recursively divides the original question into logically manageable parts and conquers them piece by piece, enhancing the reasoning capabilities and interpretability of existing MLLMs. Specifically, in the first stage, the LTR focuses on language to recursively generate a language-centric logical tree, which gradually breaks down the complex cognitive question into simple perceptual ones and plans the reasoning path through a RAG-based few-shot approach. In the second stage, with the aid of video content, the LTR performs bottom-up logical reasoning within the tree to derive the final answer along with the traceable reasoning path. Experiments across 11 VideoQA benchmarks demonstrate that our LTR framework significantly improves both accuracy and interpretability compared to state-of-the-art MLLMs. To our knowledge, this is the first work to implement a language-centric logical tree to guide MLLM reasoning in VideoQA, paving the way for language-centric video understanding from perception to cognition.

</details>

---

## 9. FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation

- [ ] FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation | https://icml.cc/virtual/2025/poster/43552

- **Link**: https://icml.cc/virtual/2025/poster/43552

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Federated Prompt Learning (FPL) adapts pre-trained Vision-Language Models (VLMs) to federated learning through prompt tuning, leveraging their transferable representations and strong generalization capabilities. Traditional methods often require uniform prompt lengths for federated aggregation, limiting adaptability to clients with diverse prompt lengths and distribution biases. In this paper, we propose Fed erated P rompt Learning for H eterogeneous Client A daptation (FedPHA), a novel framework that combines a fixed-length global prompt for efficient aggregation with local prompts of varying lengths to capture client-specific data characteristics. Additionally, FedPHA designs Singular Value Decomposition (SVD) based projection and bidirectional alignment to disentangle global conflicts arising from client heterogeneity, ensuring that personalized client tasks effectively utilize non-harmful global knowledge. This approach ensures that global knowledge improves model generalization while local knowledge preserves local optimization. Experimental results validate the effectiveness of FedPHA in achieving a balance between global and personalized knowledge in federated learning scenarios.

</details>

---

## 10. AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models

- [ ] AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/43565

- **Link**: https://icml.cc/virtual/2025/poster/43565

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of multimodal large language models (MLLMs) advances multimodal emotion recognition (MER) to the next level—from naive discriminative tasks to complex emotion understanding with advanced video understanding abilities and natural language description. However, the current community suffers from a lack of large-scale datasets with intensive, descriptive emotion annotations, as well as a multimodal-centric framework to maximize the potential of MLLMs for emotion understanding. To address this, we establish a new benchmark for MLLM-based emotion understanding with a novel dataset (MER-Caption) and a new model (AffectGPT). Utilizing our model-based crowd-sourcing data collection strategy, we construct the largest descriptive emotion dataset to date (by far), featuring over 2K fine-grained emotion categories across 115K samples. We also introduce the AffectGPT model, designed with pre-fusion operations to enhance multimodal integration. Finally, we present MER-UniBench, a unified benchmark with evaluation metrics tailored for typical MER tasks and the free-form, natural language output style of MLLMs. Extensive experimental results show AffectGPT's robust performance across various MER tasks. We have released both the code and the dataset to advance research and development in emotion understanding: https://github.com/zeroQiaoba/AffectGPT.

</details>

---

## 11. Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models

- [ ] Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models | https://icml.cc/virtual/2025/poster/43589

- **Link**: https://icml.cc/virtual/2025/poster/43589

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With advancements in visual language models (VLMs) and large language models (LLMs), video anomaly detection (VAD) has progressed beyond binary classification to fine-grained categorization and multidimensional analysis. However, existing methods focus mainly on coarse-grained detection, lacking anomaly explanations. To address these challenges, we propose Ex-VAD, an Explainable Fine-grained Video Anomaly Detection approach that combines fine-grained classification with detailed explanations of anomalies. First, we use a VLM to extract frame-level captions, and an LLM converts them to video-level explanations, enhancing the model's explainability. Second, integrating textual explanations of anomalies with visual information greatly enhances the model's anomaly detection capability. Finally, we apply label-enhanced alignment to optimize feature fusion, enabling precise fine-grained detection. Extensive experimental results on the UCF-Crime and XD-Violence datasets demonstrate that Ex-VAD significantly outperforms existing State-of-The-Art methods.

</details>

---

## 12. From Thousands to Billions: 3D Visual Language Grounding via Render-Supervised Distillation from 2D VLMs

- [ ] From Thousands to Billions: 3D Visual Language Grounding via Render-Supervised Distillation from 2D VLMs | https://icml.cc/virtual/2025/poster/43636

- **Link**: https://icml.cc/virtual/2025/poster/43636

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D vision-language grounding faces a fundamental data bottleneck: while 2D models train on billions of images, 3D models have access to only thousands of labeled scenes--a six-order-of-magnitude gap that severely limits performance. We introduce \textbf{\emph{LIFT-GS}}, a practical distillation technique that overcomes this limitation by using differentiable rendering to bridge 3D and 2D supervision. LIFT-GS predicts 3D Gaussian representations from point clouds and uses them to render predicted language-conditioned 3D masks into 2D views, enabling supervision from 2D foundation models (SAM, CLIP, LLaMA) without requiring any 3D annotations. This render-supervised formulation enables end-to-end training of complete encoder-decoder architectures and is inherently model-agnostic.  LIFT-GS achieves state-of-the-art results with 25.7\% mAP on open-vocabulary instance segmentation (vs. 20.2\% prior SOTA) and consistent 10-30\% improvements on referential grounding tasks. Remarkably, pretraining effectively multiplies fine-tuning datasets by 2×, demonstrating strong scaling properties that suggest 3D VLG currently operates in a severely data-scarce regime. Project page: \url{https://liftgs.github.io}.

</details>

---

## 13. Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance

- [ ] Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance | https://icml.cc/virtual/2025/poster/43644

- **Link**: https://icml.cc/virtual/2025/poster/43644

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of Large Vision-Language Models (LVLMs) has increasingly highlighted the critical issue of their tendency to hallucinate non-existing objects in the images. To address this issue, previous works focused on using specially curated datasets or powerful LLMs to rectify the outputs of LVLMs. However, these approaches require either costly training or fine-tuning, or API access to proprietary LLMs for post-generation correction. In response to these limitations, we propose Mitigating hallucinAtion via image-gRounded guIdaNcE (MARINE), a framework that is both training-free and API-free. MARINE effectively and efficiently reduces object hallucinations during inference by introducing image-grounded guidance to LVLMs. This is achieved by leveraging open-source vision models to extract object-level information, thereby enhancing the precision of LVLM-generated content. Our framework's flexibility further allows for the integration of multiple vision models, enabling more reliable and robust object-level guidance. Through comprehensive evaluations across 5 popular LVLMs with diverse evaluation metrics and benchmarks, we demonstrate the effectiveness of MARINE, which even outperforms existing fine-tuning-based methods. Remarkably, it reduces hallucinations consistently in GPT-4V-assisted evaluation while maintaining the detailedness of LVLMs' generations. We release our code at https://github.com/Linxi-ZHAO/MARINE.

</details>

---

## 14. Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models

- [ ] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models | https://icml.cc/virtual/2025/poster/43674

- **Link**: https://icml.cc/virtual/2025/poster/43674

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-Modal Large Language Models (MLLMs) have exhibited remarkable performance on various vision-language tasks such as Visual Question Answering (VQA). Despite accumulating evidence of privacy concerns associated with task-relevant content, it remains unclear whether MLLMs inadvertently memorize private content that is entirely irrelevant to the training tasks. In this paper, we investigate how randomly generated task-irrelevant private content can become spuriously correlated with downstream objectives due to partial mini-batch training dynamics, thus causing inadvertent memorization. Concretely, we randomly generate task-irrelevant watermarks into VQA fine-tuning images at varying probabilities and propose a novel probing framework to determine whether MLLMs have inadvertently encoded such content. Our experiments reveal that MLLMs exhibit notably different training behaviors in partial mini-batch settings with task-irrelevant watermarks embedded. Furthermore, through layer-wise probing, we demonstrate that MLLMs trigger distinct representational patterns when encountering previously seen task-irrelevant knowledge, even if this knowledge does not influence their output during prompting. Our code is available at https://github.com/illusionhi/ProbingPrivacy.

</details>

---

## 15. Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection

- [ ] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection | https://icml.cc/virtual/2025/poster/43687

- **Link**: https://icml.cc/virtual/2025/poster/43687

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in understanding multimodal data, but their potential remains underexplored for deepfake detection due to the misalignment of their knowledge and forensics patterns. To this end, we present a novel framework that unlocks LVLMs' potential capabilities for deepfake detection. Our framework includes a Knowledge-guided Forgery Detector (KFD), a Forgery Prompt Learner (FPL), and a Large Language Model (LLM). The KFD is used to calculate correlations between image features and pristine/deepfake image description embeddings, enabling forgery classification and localization. The outputs of the KFD are subsequently processed by the Forgery Prompt Learner to construct fine-grained forgery prompt embeddings. These embeddings, along with visual and question prompt embeddings, are fed into the LLM to generate textual detection responses. Extensive experiments on multiple benchmarks, including FF++, CDF2, DFD, DFDCP, DFDC, and DF40, demonstrate that our scheme surpasses state-of-the-art methods in generalization performance, while also supporting multi-turn dialogue capabilities.

</details>

---

## 16. Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark

- [ ] Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark | https://icml.cc/virtual/2025/poster/43702

- **Link**: https://icml.cc/virtual/2025/poster/43702

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The ability to organically reason over and with both text and images is a pillar of human intelligence, yet the ability of Multimodal Large Language Models (MLLMs) to perform such multimodal reasoning remains under-explored. Existing benchmarks often emphasize text-dominant reasoning or rely on shallow visual cues, failing to adequately assess integrated visual and textual reasoning. We introduce EMMA (Enhanced MultiModal reAsoning), a benchmark targeting organic multimodal reasoning across mathematics, physics, chemistry, and coding. EMMA tasks demand advanced cross-modal reasoning that cannot be addressed by reasoning independently in each modality, offering an enhanced test suite for MLLMs' reasoning capabilities. Our evaluation of state-of-the-art MLLMs on EMMA reveals significant limitations in handling complex multimodal and multi-step reasoning tasks, even with advanced techniques like Chain-of-Thought prompting and test-time compute scaling underperforming. These findings underscore the need for improved multimodal architectures and training paradigms to close the gap between human and model reasoning in multimodality.

</details>

---

## 17. DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts

- [ ] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts | https://icml.cc/virtual/2025/poster/43719

- **Link**: https://icml.cc/virtual/2025/poster/43719

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The proliferation of disinformation demands reliable and scalable fact-checking solutions. We present D ynamic E vidence-based FA ct-checking with M ultimodal E xperts (DEFAME), a modular, zero-shot MLLM pipeline for open-domain, text-image claim verification. DEFAME operates in a six-stage process, dynamically selecting the tools and search depth to extract and evaluate textual and visual evidence. Unlike prior approaches that are text-only, lack explainability, or rely solely on parametric knowledge, DEFAME performs end-to-end verification, accounting for images in claims and evidence while generating structured, multimodal reports. Evaluation on the popular benchmarks VERITE, AVeriTeC, and MOCHEG shows that DEFAME surpasses all previous methods, establishing itself as the new general state-of-the-art fact-checking system for uni- and multimodal fact-checking. Moreover, we introduce a new multimodal benchmark, ClaimReview2024+, featuring claims after the knowledge cutoff of GPT-4o, avoiding data leakage. Here, DEFAME drastically outperforms the GPT-4o baselines, showing temporal generalizability and the potential for real-time fact-checking.

</details>

---

## 18. Proposer-Agent-Evaluator (PAE): Autonomous Skill Discovery For Foundation Model Internet Agents

- [ ] Proposer-Agent-Evaluator (PAE): Autonomous Skill Discovery For Foundation Model Internet Agents | https://icml.cc/virtual/2025/poster/43739

- **Link**: https://icml.cc/virtual/2025/poster/43739

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

A generalist foundation model agent needs to have a large and diverse skill repertoire, such as finding directions between two travel locations and buying specific items from the Internet. If each skill needs to be specified manually through a fixed set of human-annotated instructions, the agent’s skill repertoire will necessarily be limited due to the scalability of human-annotated instructions. In this work, we address this challenge by proposing Proposer-Agent-Evaluator (PAE), an effective learning system that enables foundation model agents to autonomously discover and practice skills in the wild. After a context-aware task proposer generates instructions based on website information, the agent policy attempts those tasks in the real world with resulting trajectories evaluated by an autonomous VLM-based success evaluator. The success evaluation serves as the reward signal for the agent to refine its policies through RL. We validate PAE on challenging vision-based web navigation, using both real-world and selfhosted websites from WebVoyager and WebArena. Our results show that PAE significantly improves the zero-shot generalization capability of VLM Internet agents (around 50% relative improvement)to both unseen tasks and websites.

</details>

---

## 19. GMAIL: Generative Modality Alignment for generated Image Learning

- [ ] GMAIL: Generative Modality Alignment for generated Image Learning | https://icml.cc/virtual/2025/poster/43745

- **Link**: https://icml.cc/virtual/2025/poster/43745

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generative models have made it possible to synthesize highly realistic images, potentially providing an abundant data source for training machine learning models. Despite the advantages of these synthesizable data sources, the indiscriminate use of generated images as real images for training can even cause mode collapse due to modality discrepancies between real and synthetic domains. In this paper, we propose a novel framework for discriminative use of generated images, coined \textit{GMAIL}, that explicitly treats generated images as a separate modality from real images. Instead of indiscriminately replacing real images with generated ones in the pixel space, our approach bridges the two distinct modalities in the same latent space through a multi-modal learning approach. To be specific, we first fine-tune a model exclusively on generated images using a cross-modality alignment loss and then employ this aligned model to further train various vision-language models with generated images. By aligning the two modalities, our approach effectively leverages the benefits of recent advances in generative models, thereby boosting the effectiveness of generated image learning across a range of vision-language tasks. Our framework can be easily incorporated with various vision-language models, and we demonstrate its efficacy throughout extensive experiments. For example, our framework significantly improves performance on image captioning, zero-shot image retrieval, zero-shot image classification, and long caption retrieval tasks. It also shows positive generated data scaling trends and notable enhancements in the captioning performance of the large multimodal model, LLaVA.

</details>

---

## 20. LADA: Scalable Label-Specific CLIP Adapter for Continual Learning

- [ ] LADA: Scalable Label-Specific CLIP Adapter for Continual Learning | https://icml.cc/virtual/2025/poster/43751

- **Link**: https://icml.cc/virtual/2025/poster/43751

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Continual learning with vision-language models like CLIP offers a pathway toward scalable machine learning systems by leveraging its transferable representations. Existing CLIP-based methods adapt the pre-trained image encoder by adding multiple sets of learnable parameters, with each task using a partial set of parameters. This requires selecting the expected parameters for input images during inference, which is prone to error that degrades performance. To address this problem, we introduce LADA ( L abel-specific ADA pter). Instead of partitioning parameters across tasks, LADA appends lightweight, label-specific memory units to the frozen CLIP image encoder, enabling discriminative feature generation by aggregating task-agnostic knowledge. To prevent catastrophic forgetting, LADA employs feature distillation for seen classes, preventing their features from being interfered with by new classes. Positioned after the image encoder, LADA prevents gradient flow to the frozen CLIP parameters, ensuring efficient training. Extensive results show that LADA achieves state-of-the-art performance in continual learning settings.  The implementation code is available at https://github.com/MaolinLuo/LADA .

</details>

---

## 21. Visual and Domain Knowledge for Professional-level Graph-of-Thought Medical Reasoning

- [ ] Visual and Domain Knowledge for Professional-level Graph-of-Thought Medical Reasoning | https://icml.cc/virtual/2025/poster/43761

- **Link**: https://icml.cc/virtual/2025/poster/43761

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical Visual Question Answering (MVQA) requires AI models to answer questions related to medical images, offering significant potential to assist medical professionals in evaluating and diagnosing diseases, thereby improving early interventions. However, existing MVQA datasets primarily focus on basic questions regarding visual perception and pattern recognition, without addressing the more complex questions that are critical in clinical diagnosis and decision-making. This paper introduces a new benchmark designed for professional-level medical reasoning, simulating the decision-making process. We achieve this by collecting MRI and clinical data related to Hypoxic-Ischemic Encephalopathy, enriched with expert annotations and insights. Building on this data, we generate clinical question-answer pairs and MRI interpretations to enable comprehensive diagnosis, interpretation, and prediction of neurocognitive outcomes. Our evaluation of current large vision-language models (LVLMs) shows limited performance on this benchmark, highlighting both the challenges of the task and the importance of this benchmark for advancing medical AI. Furthermore, we propose a novel ``Clinical Graph of Thoughts" model, which integrates domain-specific medical knowledge and clinical reasoning processes with the interpretive abilities of LVLMs. The model demonstrates promising results, achieving around 15\% absolute gain on the most important neurocognitive outcome task, while the benchmark still reveals substantial opportunities for further research innovation.

</details>

---

## 22. SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning

- [ ] SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning | https://icml.cc/virtual/2025/poster/43771

- **Link**: https://icml.cc/virtual/2025/poster/43771

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Continual Instruction Tuning (MCIT) aims to enable Multimodal Large Language Models (MLLMs) to incrementally learn new tasks without catastrophic forgetting, thus adapting to evolving requirements. In this paper, we explore the forgetting caused by such incremental training, categorizing it into superficial forgetting and essential forgetting. Superficial forgetting refers to cases where the model’s knowledge may not be genuinely lost, but its responses to previous tasks deviate from expected formats due to the influence of subsequent tasks’ answer styles, making the results unusable. On the other hand, essential forgetting refers to situations where the model provides correctly formatted but factually inaccurate answers, indicating a true loss of knowledge. Assessing essential forgetting necessitates addressing superficial forgetting first, as severe superficial forgetting can conceal the model’s knowledge state. Hence, we first introduce the Answer Style Diversification (ASD) paradigm, which defines a standardized process for data style transformations across different tasks, unifying their training sets into similarly diversified styles to prevent superficial forgetting caused by style shifts. Building on this, we propose RegLoRA to mitigate essential forgetting. RegLoRA stabilizes key parameters where prior knowledge is primarily stored by applying regularization to LoRA’s weight update matrices, enabling the model to retain existing competencies while remaining adaptable to new tasks. Experimental results demonstrate that our overall method, SEFE, achieves state-of-the-art performance.

</details>

---

## 23. Double-Filter: Efficient Fine-tuning of Pre-trained Vision-Language Models via Patch&Layer Filtering

- [ ] Double-Filter: Efficient Fine-tuning of Pre-trained Vision-Language Models via Patch&Layer Filtering | https://icml.cc/virtual/2025/poster/43782

- **Link**: https://icml.cc/virtual/2025/poster/43782

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present a novel approach, termed Double-Filter,to “slim down” the fine-tuning process of vision-language pre-trained (VLP) models via filtering redundancies in feature inputs and architectural components. We enhance the fine-tuning process using two approaches. First, we develop a new patch selection method incorporating image patch filtering through background and foreground separation, followed by a refined patch selection process. Second, we design a genetic algorithm to eliminate redundant fine-grained architecture layers, improving the efficiency and effectiveness of the model. The former makes patch selection semantics more comprehensive, improving inference efficiency while ensuring semantic representation. The latter’s fine-grained layer filter removes architectural redundancy to the extent possible and mitigates the impact on performance. Experimental results demonstrate that the proposed Double-Filter achieves superior efficiency of model fine-tuning and maintains competitive performance compared with the advanced efficient fine-tuning methods on three downstream tasks, VQA, NLVR and Retrieval. In addition, it has been proven to be effective under METER and ViLT VLP models.

</details>

---

## 24. Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models

- [ ] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models | https://icml.cc/virtual/2025/poster/43778

- **Link**: https://icml.cc/virtual/2025/poster/43778

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Confidence calibration is critical for the safe deployment of machine learning models in the real world. However, such issue in vision-language models like CLIP, particularly after fine-tuning, has not been fully addressed. In this work, we demonstrate that existing prompt tuning methods usually lead to a trade-off of calibration between base and new classes: the cross-entropy loss used in standard fine-tuning (e.g., CoOp) causes overconfidence in new classes by increasing textual label divergence, whereas regularization-based tuning (e.g., KgCoOp) maintains the confidence level but results in underconfidence in base classes due to the improved accuracy. Inspired by the observations, we introduce Dynamic Outlier Regularization (DOR) to ensure the confidence calibration on both base and new classes after fine-tuning. In particular, we propose to minimize the feature deviation of novel textual labels (instead of base classes) sampled from a large vocabulary. In effect, DOR prevents the increase in textual divergence for new labels while easing restrictions on base classes. Extensive experiments demonstrate that DOR can enhance the calibration performance of current fine-tuning methods on base and new classes.

</details>

---

## 25. Contrastive Localized Language-Image Pre-Training

- [ ] Contrastive Localized Language-Image Pre-Training | https://icml.cc/virtual/2025/poster/43845

- **Link**: https://icml.cc/virtual/2025/poster/43845

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

CLIP has been a celebrated method for training vision encoders to generate image/text representations facilitating various applications. Recently, it has been widely adopted as the vision backbone of multimodal large language models (MLLMs). The success of CLIP relies on aligning web-crawled noisy text annotations at image levels. However, such criteria may be insufficient for downstream tasks in need of fine-grained vision representations, especially when understanding region-level is demanding for MLLMs. We improve the localization capability of CLIP with several advances. Our proposed pre-training method, Contrastive Localized Language-Image Pre-training (CLOC), complements CLIP with region-text contrastive loss and modules. We formulate a new concept, promptable embeddings, of which the encoder produces image embeddings easy to transform into region representations given spatial hints. To support large-scale pre-training, we design a visually-enriched and spatially-localized captioning framework to effectively generate region-text labels. By scaling up to billions of annotated images, CLOC enables high-quality regional embeddings for recognition and retrieval tasks, and can be a drop-in replacement of CLIP to enhance MLLMs, especially on referring and grounding tasks.

</details>

---

## 26. Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM

- [ ] Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM | https://icml.cc/virtual/2025/poster/43854

- **Link**: https://icml.cc/virtual/2025/poster/43854

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The GPT-4o's excellent duplex speech interaction ability has given users an impressive experience. Researchers have recently proposed several multimodal LLMs to achieve user-agent speech-to-speech conversations. In this paper, we propose a novel speech-text multimodal LLM architecture called Freeze-Omni, and our main contribution is that the speech input and output modalities can be easily connected to a textual LLM while keeping the LLM's parameters frozen throughout the training process. We effectively ensure that the intelligence of the Freeze-Omni in the speech modality is at the same level as that in the text modality of its backbone LLM while achieving low latency in the end-to-end spoken response. In addition, we also designed a method to achieve duplex dialogue ability through multitask training, giving Freeze-Omni a more natural style of dialogue ability between users and agents. In summary, Freeze-Omni holds great potential to conduct speech-to-speech dialogue based on a multimodal LLM under the condition of a frozen LLM, avoiding the catastrophic forgetting problem caused by limited data and training resources.

</details>

---

## 27. Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs?

- [ ] Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs? | https://icml.cc/virtual/2025/poster/43878

- **Link**: https://icml.cc/virtual/2025/poster/43878

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) are impressive at visual question answering and image captioning. But they underperform on multi-step visual reasoning---even compared to LLMs on the same tasks presented in text form---giving rise to perceptions of modality imbalance or brittleness . Towards a systematic study of such issues, we introduce a synthetic framework for assessing the ability of VLMs to perform algorithmic visual reasoning, comprising three tasks: Table Readout, Grid Navigation, and Visual Analogy. Each has two levels of difficulty, SIMPLE and HARD, and even the SIMPLE versions are difficult for frontier VLMs. We propose strategies for training on the SIMPLE version of tasks that improve performance on the corresponding HARD task, i.e., simple-to-hard (S2H) generalization. This controlled setup, where each task also has an equivalent text-only version, allows a quantification of the modality imbalance and how it is impacted by training strategy. We show that 1) explicit image-to-text conversion is important in promoting S2H generalization on images, by transferring reasoning from text; 2) conversion can be internalized at test time. We also report results of mechanistic study of this phenomenon. We identify measures of gradient alignment that can identify training strategies that promote better S2H generalization. Ablations highlight the importance of chain-of-thought.

</details>

---

## 28. DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs

- [ ] DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs | https://icml.cc/virtual/2025/poster/43884

- **Link**: https://icml.cc/virtual/2025/poster/43884

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the success of distillation in large language models (LLMs), most prior work applies identical loss functions to both teacher- and student-generated data. These strategies overlook the synergy between loss formulations and data types, leading to a suboptimal performance boost in student models. To address this, we propose DistiLLM-2, a contrastive approach that simultaneously increases the likelihood of teacher responses and decreases that of student responses by harnessing this synergy. Our extensive experiments show that DistiLLM-2 not only builds high-performing student models across a wide range of tasks, including instruction-following and code generation, but also supports diverse applications, such as preference alignment and vision-language extensions. These findings highlight the potential of a contrastive approach to enhance the efficacy of LLM distillation by effectively aligning teacher and student models across varied data types.

</details>

---

## 29. OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via  Omniverse Computation Balance

- [ ] OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via  Omniverse Computation Balance | https://icml.cc/virtual/2025/poster/43963

- **Link**: https://icml.cc/virtual/2025/poster/43963

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language instruction-tuning models have recently achieved significant performance improvements. In this work, we discover that large-scale 3D parallel training on those models leads to an imbalanced computation load across different devices. The vision and language parts are inherently heterogeneous:  their data distribution and model architecture differ significantly, which affects distributed training efficiency. To address this issue, we rebalance the computational load from data, model, and memory perspectives, achieving more balanced computation across devices.  Specifically, for the data, instances are grouped into new balanced mini-batches within and across devices. A search-based method is employed for the model to achieve a more balanced partitioning. For memory optimization, we adaptively adjust the re-computation strategy for each partition to utilize the available memory fully. These three perspectives are not independent but are closely connected, forming an omniverse balanced training framework. Extensive experiments are conducted to validate the effectiveness of our method. Compared with the open-source training code of InternVL-Chat, training time is reduced greatly, achieving about 1.8$\times$ speed-up. Our method's efficacy and generalizability are further validated across various models and datasets. Codes will be released at https://github.com/ModelTC/OmniBal.

</details>

---

## 30. DIS-CO: Discovering Copyrighted Content in VLMs Training Data

- [ ] DIS-CO: Discovering Copyrighted Content in VLMs Training Data | https://icml.cc/virtual/2025/poster/43970

- **Link**: https://icml.cc/virtual/2025/poster/43970

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

How can we verify whether copyrighted content was used to train a large vision-language model (VLM) without direct access to its training data? Motivated by the hypothesis that a VLM is able to recognize images from its training corpus, we propose DIS-CO, a novel approach to infer the inclusion of copyrighted content during the model's development. By repeatedly querying a VLM with specific frames from targeted copyrighted material, DIS-CO extracts the content's identity through free-form text completions. To assess its effectiveness, we introduce MovieTection, a benchmark comprising 14,000 frames paired with detailed captions, drawn from films released both before and after a model’s training cutoff. Our results show that DIS-CO significantly improves detection performance, nearly doubling the average AUC of the best prior method on models with logits available. Our findings also highlight a broader concern: all tested models appear to have been exposed to some extent to copyrighted content. We provide the code in the supplementary materials.

</details>

---

## 31. SPEX: Scaling Feature Interaction Explanations for LLMs

- [ ] SPEX: Scaling Feature Interaction Explanations for LLMs | https://icml.cc/virtual/2025/poster/44009

- **Link**: https://icml.cc/virtual/2025/poster/44009

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have revolutionized machine learning due to their ability to capture complex interactions between input features. Popular post-hoc explanation methods like SHAP provide *marginal* feature attributions, while their extensions to interaction importances only scale to small input lengths ($\approx 20$). We propose *Spectral Explainer* (SPEX), a model-agnostic interaction attribution algorithm that efficiently scales to large input lengths ($\approx 1000)$. SPEX exploits underlying natural sparsity among interactions—common in real-world data—and applies a sparse Fourier transform using a channel decoding algorithm to efficiently identify important interactions. We perform experiments across three difficult long-context datasets that require LLMs to utilize interactions between inputs to complete the task. For large inputs, SPEX outperforms marginal attribution methods by up to 20\% in terms of faithfully reconstructing LLM outputs. Further, SPEX successfully identifies key features and interactions that strongly influence model output. For one of our datasets, *HotpotQA*, SPEX provides interactions that align with human annotations. Finally, we use our model-agnostic approach to generate explanations to demonstrate abstract reasoning in closed-source  LLMs (*GPT-4o mini*) and  compositional reasoning in vision-language models.

</details>

---

## 32. Empowering World Models with Reflection for Embodied Video Prediction

- [ ] Empowering World Models with Reflection for Embodied Video Prediction | https://icml.cc/virtual/2025/poster/44044

- **Link**: https://icml.cc/virtual/2025/poster/44044

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video generation models have made significant progress in simulating future states, showcasing their potential as world simulators in embodied scenarios. However, existing models often lack robust understanding, limiting their ability to perform multi-step predictions or handle Out-of-Distribution (OOD) scenarios.  To address this challenge, we propose the Reflection of Generation (RoG), a set of intermediate reasoning strategies designed to enhance video prediction.  It leverages the complementary strengths of pre-trained vision-language and video generation models, enabling them to function as a world model in embodied scenarios. To support RoG, we introduce Embodied Video Anticipation Benchmark(EVA-Bench), a comprehensive benchmark that evaluates embodied world models across diverse tasks and scenarios, utilizing both in-domain and OOD datasets. Building on this foundation, we devise a world model, Embodied Video Anticipator (EVA), that follows a multistage training paradigm to generate high-fidelity video frames and apply an autoregressive strategy to enable adaptive generalization for longer video sequences. Extensive experiments demonstrate the efficacy of EVA in various downstream tasks like video generation and robotics, thereby paving the way for large-scale pre-trained models in real-world video prediction applications. The video demos are available at https://sites.google.com/view/icml-eva.

</details>

---

## 33. Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models

- [ ] Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models | https://icml.cc/virtual/2025/poster/44046

- **Link**: https://icml.cc/virtual/2025/poster/44046

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The improved semantic understanding of vision-language pretrained (VLP) models has made it increasingly difficult to protect publicly posted images from being exploited by search engines and other similar tools. In this context, this paper seeks to protect users' privacy by implementing defenses at the image compression stage to prevent exploitation. Specifically, we propose a flexible coding method, termed Privacy-Shielded Image Compression (PSIC), that can produce bitstreams with multiple decoding options. By default, the bitstream is decoded to preserve satisfactory perceptual quality while preventing interpretation by VLP models. Our method also retains the original image compression functionality. With a customizable input condition, the proposed scheme can reconstruct the image that preserves its full semantic information. A Conditional Latent Trigger Generation (CLTG) module is proposed to produce bias information based on customizable conditions to guide the decoding process into different reconstructed versions, and an Uncertainty-Aware Encryption-Oriented (UAEO) optimization function is designed to leverage the soft labels inferred from the target VLP model's uncertainty on the training data. This paper further incorporates an adaptive multi-objective optimization strategy to obtain improved encrypting performance and perceptual quality simultaneously within a unified training process. The proposed scheme is plug-and-play and can be seamlessly integrated into most existing Learned Image Compression (LIC) models. Extensive experiments across multiple downstream tasks have demonstrated the effectiveness of our design.

</details>

---

## 34. Unifying Specialized Visual Encoders for Video Language Models

- [ ] Unifying Specialized Visual Encoders for Video Language Models | https://icml.cc/virtual/2025/poster/44055

- **Link**: https://icml.cc/virtual/2025/poster/44055

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision backbones have yielded powerful and diverse visual and video encoders. Yet, current Video Large Language Models encode visual inputs using an encoder from a single backbone family, limiting the amount and type of visual information they can process. We propose MERV, a Multi-Encoder Video Representation, which utilizes multiple encoders for a comprehensive video representation. To optimize heterogeneous features from a broad spectrum of encoders and ensure efficient and coherent feature integration, MERV first aligns encoder features spatio-temporally, then projects them into a unified structure, and finally fuses them through cross-attention. Under fair comparison, MERV achieves up to 4.62% higher accuracy than its base model, while introducing minimal extra parameters and training faster than equivalent single-encoder methods after parallelizing visual processing. Qualitative analysis shows MERV successfully captures and integrates domain knowledge from each encoder, opening new possibilities for scaling enhanced video understanding.

</details>

---

## 35. Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging

- [ ] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging | https://icml.cc/virtual/2025/poster/44093

- **Link**: https://icml.cc/virtual/2025/poster/44093

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) combine visual perception with the general capabilities, such as reasoning, of Large Language Models (LLMs). However, the mechanisms by which these two abilities can be combined and contribute remain poorly understood.In this work, we explore to compose perception and reasoning through model merging that connects parameters of different models.  Unlike previous works that often focus on merging models of the same kind, we propose merging models across modalities , enabling the incorporation of the reasoning capabilities of LLMs into VLMs. Through extensive experiments, we demonstrate that model merging offers a successful pathway to transfer reasoning abilities from LLMs to VLMs in a training-free manner.Moreover, we utilize the merged models to understand the internal mechanism of perception and reasoning and how merging affects it. We find that perception capabilities are predominantly encoded in the early layers of the model, whereas reasoning is largely facilitated by the middle-to-late layers. After merging, we observe that all layers begin to contribute to reasoning, whereas the distribution of perception abilities across layers remains largely unchanged. These observations shed light on the potential of model merging as a tool for multimodal integration and interpretation.

</details>

---

## 36. SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models

- [ ] SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models | https://icml.cc/virtual/2025/poster/44108

- **Link**: https://icml.cc/virtual/2025/poster/44108

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional autonomous driving systems often struggle to connect high-level reasoning with low-level control, leading to suboptimal and sometimes unsafe behaviors. Recent advances in multimodal large language models (MLLMs), which process both visual and textual data, offer an opportunity to unify perception and reasoning. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge.To address this, we propose SafeAuto, a framework that enhances MLLM-based autonomous driving by incorporating both unstructured and structured knowledge. First, we introduce a Position-Dependent Cross-Entropy (PDCE) loss to improve low-level control signal predictions when values are represented as text. Second, to explicitly integrate safety knowledge, we develop a reasoning component that translates traffic rules into first-order logic (e.g., "red light => stop") and embeds them into a probabilistic graphical model (e.g., Markov Logic Network) to verify predicted actions using recognized environmental attributes.Additionally, our Multimodal Retrieval-Augmented Generation (RAG) model leverages video, control signals, and environmental attributes to learn from past driving experiences. Integrating PDCE, MLN, and Multimodal RAG, SafeAuto outperforms existing baselines across multiple datasets, enabling more accurate, reliable, and safer autonomous driving. The code is available at https://github.com/AI-secure/SafeAuto.

</details>

---

## 37. Primitive Vision: Improving Diagram Understanding in MLLMs

- [ ] Primitive Vision: Improving Diagram Understanding in MLLMs | https://icml.cc/virtual/2025/poster/44142

- **Link**: https://icml.cc/virtual/2025/poster/44142

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mathematical diagrams have a distinctive structure. Standard feature transforms designed for natural images (e.g., CLIP) fail to process them effectively, limiting their utility in multimodal large language models (MLLMs). Current efforts to improve MLLMs have primarily focused on scaling mathematical visual instruction datasets and strengthening LLM backbones, yet fine‐grained visual recognition errors remain unaddressed. Our systematic evaluation on the visual grounding capabilities of state‐of‐the‐art MLLMs highlights that fine‐grained visual understanding remains a crucial bottleneck in visual mathematical reasoning (GPT-4o exhibits a 70% grounding error rate, and correcting these errors improves reasoning accuracy by 12%). We thus propose a novel approach featuring a geometrically‐grounded vision encoder and a feature router that dynamically selects between hierarchical visual feature maps. Our model accurately recognizes visual primitives and generates precise visual prompts aligned with the language model’s reasoning needs. In experiments, PRIMITIVE-Qwen2.5-7B outperforms other 7B models by 12% on MathVerse and is on par with GPT-4V on MathVista. Our findings highlight the need for better fine‐grained visual integration in MLLMs. Code is available at github.com/AI4Math-ShanZhang/SVE-Math.

</details>

---

## 38. MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention

- [ ] MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention | https://icml.cc/virtual/2025/poster/44144

- **Link**: https://icml.cc/virtual/2025/poster/44144

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The integration of long-context capabilities with visual understanding unlocks unprecedented potential for Vision Language Models (VLMs). However, the quadratic attention complexity during the pre-filling phase remains a significant obstacle to real-world deployment. To overcome this limitation, we introduce MMInference (Multimodality Million tokens Inference), a dynamic sparse attention method that accelerates the prefilling stage for long-context multi-modal inputs. First, our analysis reveals that the temporal and spatial locality of video input leads to a unique sparse pattern, the Grid pattern. Simultaneously, VLMs exhibit markedly different sparse distributions across different modalities. We introduce a permutation-based method to leverage the unique Grid pattern and handle modality boundary issues. By offline search the optimal sparse patterns for each head, MMInference constructs the sparse distribution dynamically based on the input. We also provide optimized GPU kernels for efficient sparse computations. Notably, MMInference integrates seamlessly into existing VLM pipelines without any model modifications or fine-tuning. Experiments on multi-modal benchmarks-including Video QA, Captioning, VisionNIAH, and Mixed-Modality NIAH-with state-of-the-art long-context VLMs (LongVila, LlavaVideo, VideoChat-Flash, Qwen2.5-VL) show that MMInference accelerates the pre-filling stage by up to 8.3x at 1M tokens while maintaining accuracy. Our code is available at https://ama.ms/MMInference.

</details>

---

## 39. FeatSharp: Your Vision Model Features, Sharper

- [ ] FeatSharp: Your Vision Model Features, Sharper | https://icml.cc/virtual/2025/poster/44186

- **Link**: https://icml.cc/virtual/2025/poster/44186

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The feature maps of vision encoders are fundamental to myriad modern AI tasks, ranging from core perception algorithms (e.g. semantic segmentation, object detection, depth perception, etc.) to modern multimodal understanding in vision-language models (VLMs). Currently, in computer vision, the frontier of general purpose vision backbones is Vision Transformers (ViT), typically trained using contrastive loss (e.g. CLIP). A key problem with most off-the-shelf ViTs, particularly CLIP, is that these models are inflexibly low resolution. Most run at $224 \times 224$px, while the "high-resolution" versions are around $378-448$px, but still inflexible. We introduce a novel method to coherently and cheaply upsample the feature maps of low-resolution vision encoders while picking up on fine-grained details that would otherwise be lost due to resolution. We demonstrate the effectiveness of this approach on core perception tasks as well as within agglomerative model training using RADIO as a way of providing richer targets for distillation. Code available at https://github.com/NVlabs/FeatSharp

</details>

---

## 40. Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

- [ ] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models | https://icml.cc/virtual/2025/poster/44202

- **Link**: https://icml.cc/virtual/2025/poster/44202

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. Intricate instructions (e.g., "Could you make me a vegetarian sandwich?" or "I don't like that one") require not just the ability to physically perform the individual steps, but the ability to situate complex commands and feedback in the physical world. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands ("pick up the cup"), our system can reason through complex prompts and incorporate situated feedback during task execution ("that's not trash"). We evaluate our system across three robotic platforms, including single-arm, dual-arm, and dual-arm mobile robots, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping.Videos are available at https://www.pi.website/research/hirobot

</details>

---

## 41. LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models

- [ ] LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/44230

- **Link**: https://icml.cc/virtual/2025/poster/44230

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Cross-attention is commonly adopted in multimodal large language models (MLLMs) for integrating visual information into the language backbone. However, in applications with large visual inputs, such as video understanding, processing a large number of visual tokens in cross-attention layers leads to high memory demands and often necessitates distributed computation across multiple GPUs. Existing distributed attention mechanisms face significant communication overheads, making cross-attention layers a critical bottleneck for efficient training and inference of MLLMs. To address this, we propose LV-XAttn, a distributed, exact cross-attention mechanism with minimal communication overhead. We observe that in applications involving large visual inputs, the size of the query block is typically much smaller than that of the key-value blocks.  Thus, in LV-XAttn we keep the large key-value blocks locally on each GPU and exchange smaller query blocks across GPUs. We also introduce an efficient activation recomputation technique to support longer visual context. We theoretically analyze the communication benefits of LV-XAttn and show that it can achieve speedups for a wide range of models. Our evaluations with Llama 3-V, mPLUG-Owl3 and OpenFlamingo models find that LV-XAttn achieves up to 10.62$\times$ end-to-end speedup compared to existing approaches.

</details>

---

## 42. Explainable Concept Generation through Vision-Language Preference Learning for Understanding Neural Networks' Internal Representations

- [ ] Explainable Concept Generation through Vision-Language Preference Learning for Understanding Neural Networks' Internal Representations | https://icml.cc/virtual/2025/poster/44233

- **Link**: https://icml.cc/virtual/2025/poster/44233

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding the inner representation of a neural network helps users improve models. Concept-based methods have become a popular choice for explaining deep neural networks post-hoc because, unlike most other explainable AI techniques, they can be used to test high-level visual "concepts" that are not directly related to feature attributes. For instance, the concept of "stripes" is important to classify an image as a zebra. Concept-based explanation methods, however, require practitioners to guess and manually collect multiple candidate concept image sets, making the process labor-intensive and prone to overlooking important concepts. Addressing this limitation, in this paper, we frame concept image set creation as an image generation problem. However, since naively using a standard generative model does not result in meaningful concepts, we devise a reinforcement learning-based preference optimization (RLPO) algorithm that fine-tunes a vision-language generative model from approximate textual descriptions of concepts. Through a series of experiments, we demonstrate our method's ability to efficiently and reliably articulate diverse concepts that are otherwise challenging to craft manually.

</details>

---

## 43. Enhancing Rating-Based Reinforcement Learning to Effectively Leverage Feedback from Large Vision-Language Models

- [ ] Enhancing Rating-Based Reinforcement Learning to Effectively Leverage Feedback from Large Vision-Language Models | https://icml.cc/virtual/2025/poster/44273

- **Link**: https://icml.cc/virtual/2025/poster/44273

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Designing effective reward functions remains a fundamental challenge in reinforcement learning (RL), as it often requires extensive human effort and domain expertise. While RL from human feedback has been successful in aligning agents with human intent, acquiring high-quality feedback is costly and labor-intensive, limiting its scalability. Recent advancements in foundation models present a promising alternative--leveraging AI-generated feedback to reduce reliance on human supervision in reward learning. Building on this paradigm, we introduce ERL-VLM, an enhanced rating-based RL method that effectively learns reward functions from AI feedback. Unlike prior methods that rely on pairwise comparisons, ERL-VLM queries large vision-language models (VLMs) for absolute ratings of individual trajectories, enabling more expressive feedback and improved sample efficiency. Additionally, we propose key enhancements to rating-based RL, addressing instability issues caused by data imbalance and noisy labels. Through extensive experiments across both low-level and high-level control tasks, we demonstrate that ERL-VLM significantly outperforms existing VLM-based reward generation methods. Our results demonstrate the potential of AI feedback for scaling RL with minimal human intervention, paving the way for more autonomous and efficient reward learning.

</details>

---

## 44. Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas

- [ ] Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas | https://icml.cc/virtual/2025/poster/44272

- **Link**: https://icml.cc/virtual/2025/poster/44272

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision Language Models (VLMs) have long struggled with spatial reasoning tasks. Surprisingly, even simple spatial reasoning tasks, such as recognizing “under” or “behind” relationships between only two objects, pose significant challenges for current VLMs. We believe it is crucial to use the lens of mechanism interpretability, opening up the model and diving into model’s internal states to examine the interactions between image and text tokens during spatial reasoning. Our analysis of attention behaviors reveals significant differences in how VLMs allocate attention to image versus text. By tracing the areas of images that receive the highest attention scores throughout intermediate layers, we observe a notable pattern: errors often coincide with attention being misdirected towards irrelevant objects within the image. Moreover, such attention patterns exhibit substantial differences between familiar (e.g., “on the left side of ”) and unfamiliar (e.g.,“in front of ”) spatial relationships. Motivated by these findings, we propose ADAPTVIS based on inference-time confidence scores to sharpen the attention on highly relevant regions when the model exhibits high confidence, while smoothing and broadening the attention window to consider a wider context when confidence is lower. This training-free decoding method shows significant improvement (e.g., up to a 50 absolute point improvement) on spatial reasoning benchmarks such as WhatsUp and VSR with negligible additional cost.

</details>

---

## 45. Testing the Limits of Fine-Tuning for Improving Visual Cognition in Vision Language Models

- [ ] Testing the Limits of Fine-Tuning for Improving Visual Cognition in Vision Language Models | https://icml.cc/virtual/2025/poster/44300

- **Link**: https://icml.cc/virtual/2025/poster/44300

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision language models still fall short of human visual cognition. In an effort to improve visual cognition and align models with human behavior, we introduce visual stimuli and human judgments on visual cognition tasks, allowing us to systematically evaluate performance across cognitive domains under a consistent environment. We fine-tune models on ground truth data for intuitive physics and causal reasoning and find that this improves model performance in the respective fine-tuning domain. Furthermore, it can improve model alignment with human behavior. However, we find that task-specific fine-tuning does not contribute to robust human-like generalization to data with other visual characteristics or to tasks in other cognitive domains.

</details>

---

## 46. Gradient Inversion of Multimodal Models

- [ ] Gradient Inversion of Multimodal Models | https://icml.cc/virtual/2025/poster/44322

- **Link**: https://icml.cc/virtual/2025/poster/44322

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Federated learning (FL) enables privacy-preserving distributed machine learning by sharing gradients instead of raw data. However, FL remains vulnerable to gradient inversion attacks, in which shared gradients can reveal sensitive training data. Prior research has mainly concentrated on unimodal tasks, particularly image classification, examining the reconstruction of single-modality data, and analyzing privacy vulnerabilities in these relatively simple scenarios. As multimodal models are increasingly used to address complex vision-language tasks, it becomes essential to assess the privacy risks inherent in these architectures. In this paper, we explore gradient inversion attacks targeting multimodal vision-language Document Visual Question Answering (DQA) models and propose GI-DQA, a novel method that reconstructs private document content from gradients. Through extensive evaluation on state-of-the-art DQA models, our approach exposes critical privacy vulnerabilities and highlights the urgent need for robust defenses to secure multimodal FL systems.

</details>

---

## 47. Subobject-level Image Tokenization

- [ ] Subobject-level Image Tokenization | https://icml.cc/virtual/2025/poster/44334

- **Link**: https://icml.cc/virtual/2025/poster/44334

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Patch-based image tokenization ignores the morphology of the visual world, limiting effective and efficient learning of image understanding. Inspired by subword tokenization, we introduce subobject-level adaptive token segmentation and explore several approaches, including superpixel, SAM, and a proposed Efficient and PanOptiC (EPOC) image tokenizer. Our EPOC combines boundary detection--a simple task that can be handled well by a compact model--with watershed segmentation, which inherently guarantees no pixels are left unsegmented. Intrinsic evaluations across 5 datasets demonstrate that EPOC's segmentation aligns well with human annotations of both object- and part-level visual morphology, producing more monosemantic tokens and offering substantial efficiency advantages. For extrinsic evaluation, we designed a token embedding that handles arbitrary-shaped tokens, and trained VLMs with different tokenizers on 4 datasets of object recognition and detailed captioning. The results reveal that subobject tokenization enables faster convergence and better generalization while using fewer visual tokens.

</details>

---

## 48. Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach

- [ ] Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach | https://icml.cc/virtual/2025/poster/44373

- **Link**: https://icml.cc/virtual/2025/poster/44373

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have shown promising capabilities but struggle under distribution shifts, where evaluation data differ from instruction tuning distributions. Although previous works have provided empirical evaluations, we argue that establishing a formal framework that can characterize and quantify the risk of MLLMs is necessary to ensure the safe and reliable application of MLLMs in the real world. By taking an information-theoretic perspective, we propose the first theoretical framework that enables the quantification of the maximum risk of MLLMs under distribution shifts. Central to our framework is the introduction of Effective Mutual Information (EMI), a principled metric that quantifies the relevance between input queries and model responses. We derive an upper bound for the EMI difference between in-distribution (ID) and out-of-distribution (OOD) data, connecting it to visual and textual distributional discrepancies.Extensive experiments on real benchmark datasets, spanning 61 shift scenarios, empirically validate our theoretical insights.

</details>

---

## 49. ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics

- [ ] ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics | https://icml.cc/virtual/2025/poster/44449

- **Link**: https://icml.cc/virtual/2025/poster/44449

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reinforcement learning (RL) has demonstrated compelling performance in robotic tasks, but its success often hinges on the design of complex, ad hoc reward functions. Researchers have explored how Large Language Models (LLMs) could enable non-expert users to specify reward functions more easily. However, LLMs struggle to balance the importance of different features, generalize poorly to out-of-distribution robotic tasks, and cannot represent the problem properly with only text-based descriptions. To address these challenges, we propose ELEMENTAL (intEractive LEarning froM dEmoNstraTion And Language), a novel framework that combines natural language guidance with visual user demonstrations to align robot behavior with user intentions better. By incorporating visual inputs, ELEMENTAL overcomes the limitations of text-only task specifications, while leveraging inverse reinforcement learning (IRL) to balance feature weights and match the demonstrated behaviors optimally. ELEMENTAL also introduces an iterative feedback-loop through self-reflection to improve feature, reward, and policy learning. Our experiment results demonstrate that ELEMENTAL outperforms prior work by 42.3% on task success, and achieves 41.3% better generalization in out-of-distribution tasks, highlighting its robustness in LfD.

</details>

---

## 50. GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models

- [ ] GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models | https://icml.cc/virtual/2025/poster/44465

- **Link**: https://icml.cc/virtual/2025/poster/44465

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in test-time adaptation (TTA) for Vision-Language Models (VLMs) have garnered increasing attention, particularly through the use of multiple augmented views of a single image to boost zero-shot generalization. Unfortunately, existing methods fail to strike a satisfactory balance between performance and efficiency, either due to excessive overhead of tuning text prompts or unstable benefits from handcrafted, training-free visual feature enhancement. In this paper, we present Global-Spatial Bias Learner (GS-Bias), an efficient and effective TTA paradigm that incorporates two learnable biases during TTA, unfolded as the global bias and spatial bias. Particularly, the global bias captures the global semantic features of a test image by learning consistency across augmented views, while spatial bias learns the semantic coherence between regions in the image’s spatial visual representation. It is worth highlighting that these two sets of biases are directly added to the logits outputed by the pretrained VLMs, which circumvent the full backpropagation through VLM that hinders the efficiency of existing TTA methods. This endows GS-Bias with extremely high efficiency while achieving state-of-the-art performance on 15 benchmark datasets. For example, it achieves a 2.23% improvement over TPT in cross-dataset generalization and a 2.72% improvement in domain generalization, while requiring only 6.5% of TPT's memory usage on ImageNet.

</details>

---

## 51. BDC-CLIP: Brownian Distance Covariance for Adapting CLIP to Action Recognition

- [ ] BDC-CLIP: Brownian Distance Covariance for Adapting CLIP to Action Recognition | https://icml.cc/virtual/2025/poster/44508

- **Link**: https://icml.cc/virtual/2025/poster/44508

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Bridging contrastive language-image pre-training (CLIP) to video action recognition has attracted growing interest. Human actions are inherently rich in spatial and temporal contexts, involving dynamic interactions among people, objects, and the environment. Accurately recognizing actions requires effectively capturing these fine-grained elements and modeling their relationships with language. However, most existing methods rely on cosine similarity--practically equivalent to the Pearson correlation coefficient--between global tokens for video-language alignment. As a result, they have limited capacity to model complex dependencies and tend to overlook local tokens that encode critical spatio-temporal cues. To overcome these limitations, we propose BDC-CLIP, a novel framework that leverages Brownian Distance Covariance (BDC) to align visual and textual representations. Our method can capture complex relationships--both linear and nonlinear--between all visual and textual tokens, enabling fine-grained modeling in space, time, and language. BDC-CLIP achieves state-of-the-art performance across zero-shot, few-shot, base-to-novel, and fully supervised action recognition settings, demonstrating its effectiveness and broad applicability.

</details>

---

## 52. Textural or Textual: How Vision-Language Models Read Text in Images

- [ ] Textural or Textual: How Vision-Language Models Read Text in Images | https://icml.cc/virtual/2025/poster/44522

- **Link**: https://icml.cc/virtual/2025/poster/44522

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Typographic attacks are often attributed to the ability of multimodal pre-trained models to fuse textual semantics into visual representations, yet the mechanisms and locus of such interference remain unclear. We examine whether such models genuinely encode textual semantics or primarily rely on texture-based visual features. To disentangle orthographic form from meaning, we introduce the ToT dataset, which includes controlled word pairs that either share semantics with distinct appearances (synonyms) or share appearance with differing semantics (paronyms). A layer-wise analysis of Intrinsic Dimension (ID) reveals that early layers exhibit competing dynamics between orthographic and semantic representations. In later layers, semantic accuracy increases as ID decreases, but this improvement largely stems from orthographic disambiguation. Notably, clear semantic differentiation emerges only in the final block, challenging the common assumption that semantic understanding is progressively constructed across depth. These findings reveal how current vision-language models construct text representations through texture-dependent processes, prompting a reconsideration of the gap between visual perception and semantic understanding. The code is available at: https://github.com/Ovsia/Textural-or-Textual

</details>

---

## 53. LEMoN: Label Error Detection using Multimodal Neighbors

- [ ] LEMoN: Label Error Detection using Multimodal Neighbors | https://icml.cc/virtual/2025/poster/44525

- **Link**: https://icml.cc/virtual/2025/poster/44525

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large repositories of image-caption pairs are essential for the development of vision-language models. However, these datasets are often extracted from noisy data scraped from the web, and contain many mislabeled instances. In order to improve the reliability of downstream models, it is important to identify and filter images with incorrect captions. However, beyond filtering based on image-caption embedding similarity, no prior works have proposed other methods to filter noisy multimodal data, or concretely assessed the impact of noisy captioning data on downstream training. In this work, we propose, theoretically justify, and empirically validate LEMoN, a method to identify label errors in image-caption datasets. Our method leverages the multimodal neighborhood of image-caption pairs in the latent space of contrastively pretrained multimodal models to automatically identify label errors. Through empirical evaluations across eight datasets and twelve baselines, we find that LEMoN outperforms the baselines by over 3% in label error detection, and that training on datasets filtered using our method improves downstream captioning performance by more than 2 BLEU points over noisy training.

</details>

---

## 54. MATS: An Audio Language Model under Text-only Supervision

- [ ] MATS: An Audio Language Model under Text-only Supervision | https://icml.cc/virtual/2025/poster/44538

- **Link**: https://icml.cc/virtual/2025/poster/44538

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large audio-language models (LALMs), built upon powerful Large Language Models (LLMs), have exhibited remarkable audio comprehension and reasoning capabilities. However, the training of LALMs demands a large corpus of audio-language pairs, which requires substantial costs in both data collection and training resources. In this paper, we propose MATS , an audio-language multimodal LLM designed to handle M ultiple A udio task using solely T ext-only S upervision. By leveraging pre-trained audio-language alignment models such as CLAP, we develop a text-only training strategy that projects the shared  audio-language latent space into LLM latent space, endowing the LLM with audio comprehension capabilities without relying on audio data during training. To further bridge the modality gap between audio and language embeddings within CLAP, we propose the S trongly-rel a ted n oisy t ext with a udio ( Santa ) mechanism. Santa maps audio embeddings into CLAP language embedding space while preserving essential information from the audio input. Extensive experiments demonstrate that MATS, despite being trained exclusively on text data, achieves competitive performance compared to recent LALMs trained on large-scale audio-language pairs. The code is publicly available in https://github.com/wangwen-banban/MATS

</details>

---

## 55. Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models

- [ ] Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/44566

- **Link**: https://icml.cc/virtual/2025/poster/44566

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite their impressive capabilities, Multimodal Large Language Models (MLLMs) are prone to hallucinations, i.e., the generated content that is nonsensical or unfaithful to input sources.Unlike in LLMs, hallucinations in MLLMs often stem from the sensitivity of text decoder to visual tokens, leading to a phenomenon akin to "amnesia" about visual information.To address this issue, we propose MemVR, a novel decoding paradigm inspired by common cognition: when the memory of an image seen the moment before is forgotten, people will look at it again for factual answers. Following this principle, we treat visual tokens as supplementary evidence, re-injecting them into the MLLM through Feed Forward Network (FFN) as “key-value memory” at the middle trigger layer. This look-twice mechanism occurs when the model exhibits high uncertainty during inference, effectively enhancing factual alignment. Comprehensive experimental evaluations demonstrate that MemVR significantly mitigates hallucination across various MLLMs and excels in general benchmarks without incurring additional time overhead.

</details>

---

## 56. CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing

- [ ] CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing | https://icml.cc/virtual/2025/poster/44580

- **Link**: https://icml.cc/virtual/2025/poster/44580

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Computer Aided Design (CAD) is indispensable across various industries. \emph{Text-based CAD editing}, which automates the modification of CAD models based on textual instructions, holds great potential but remains underexplored.Existing methods primarily focus on design variation generation or text-based CAD generation, either lacking support for text-based control or neglecting existing CAD models as constraints.We introduce \emph{CAD-Editor}, the first framework for text-based CAD editing. To address the challenge of demanding triplet data with accurate correspondence for training, we propose an automated data synthesis pipeline. This pipeline utilizes design variation models to generate pairs of original and edited CAD models and employs Large Vision-Language Models (LVLMs) to summarize their differences into editing instructions.To tackle the composite nature of text-based CAD editing, we propose a locate-then-infill framework that decomposes the task into two focused sub-tasks: locating regions requiring modification and infilling these regions with appropriate edits. Large Language Models (LLMs) serve as the backbone for both sub-tasks, leveraging their capabilities in natural language understanding and CAD knowledge.Experiments show that CAD-Editor achieves superior performance both quantitatively and qualitatively.

</details>

---

## 57. MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization

- [ ] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization | https://icml.cc/virtual/2025/poster/44599

- **Link**: https://icml.cc/virtual/2025/poster/44599

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of Large Vision-Language Models (LVLMs) has propelled their application in the medical field. However, Medical LVLMs (Med-LVLMs) encounter factuality challenges due to modality misalignment, where the models prioritize textual knowledge over visual input, leading to hallucinations that contradict information in medical images. Previous attempts to enhance modality alignment in Med-LVLMs through preference optimization have inadequately addressed clinical relevance in preference data, making these samples easily distinguishable and reducing alignment effectiveness. In response, we propose MMedPO, a novel multimodal medical preference optimization approach that considers the clinical relevance of preference samples to enhance Med-LVLM alignment. MMedPO curates multimodal preference data by introducing two types of dispreference: (1) plausible hallucinations injected through target Med-LVLMs or GPT-4o to produce medically inaccurate responses, and (2) lesion region neglect achieved through local lesion-noising, disrupting visual understanding of critical areas. We then calculate clinical relevance for each sample based on scores from multiple Med-LLMs and visual tools, enabling effective alignment. Our experiments demonstrate that MMedPO significantly enhances factual accuracy in Med-LVLMs, achieving substantial improvements over existing preference optimization methods by 14.2% and 51.7% on the Med-VQA and report generation tasks, respectively. Our code are available in https://github.com/aiming-lab/MMedPO}{https://github.com/aiming-lab/MMedPO.

</details>

---

## 58. Modularized Self-Reflected Video Reasoner for Multimodal LLM with Application to Video Question Answering

- [ ] Modularized Self-Reflected Video Reasoner for Multimodal LLM with Application to Video Question Answering | https://icml.cc/virtual/2025/poster/44609

- **Link**: https://icml.cc/virtual/2025/poster/44609

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (Multimodal LLMs) have shown their strength in Video Question Answering (VideoQA). However, due to the black-box nature of end-to-end training strategies, existing approaches based on Multimodal LLMs suffer from the lack of interpretability for VideoQA: they can neither present reasoning paths nor indicate where the answers are derived from the video. To address this issue, we propose MSR-ViR ( M odularized S elf- R eflected Vi deo R easoner), which for the first time integrates modular networks to Multimodal LLMs, capable of providing VideoQA with explicit reasoning paths for more interpretability. Specifically, a MoST-Grounding (Modularized Spatial-Temporal Grounding) network is proposed to decompose complex questions via tree-structured policies, localizing relevant temporal and spatial segments within videos through step-by-step reasoning. The proposed MoST-Grounding network provides explicit visually grounded information for Multimodal LLMs with clear reasoning paths, thus enhancing interpretability for the predicted answers. To further improve the reasoning quality, we design an Alternate Self-reflection Training Strategy to jointly optimize policy generation and Multimodal LLMs. Experiments on real-world datasets demonstrate the superiority of our proposed MSR-ViR framework in video understanding, reasoning transparency, and providing explicit localization evidence for answers.

</details>

---

## 59. TRUST-VLM: Thorough Red-Teaming for Uncovering Safety Threats in Vision-Language Models

- [ ] TRUST-VLM: Thorough Red-Teaming for Uncovering Safety Threats in Vision-Language Models | https://icml.cc/virtual/2025/poster/44631

- **Link**: https://icml.cc/virtual/2025/poster/44631

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have become a cornerstone in multi-modal artificial intelligence, enabling seamless integration of visual and textual information for tasks such as image captioning, visual question answering, and cross-modal retrieval. Despite their impressive capabilities, these models often exhibit inherent vulnerabilities that can lead to safety failures in critical applications. Red-teaming is an important approach to identify and test system's vulnerabilities, but how to conduct red-teaming for contemporary VLMs is an unexplored area. In this paper, we propose a novel multi-modal red-teaming approach, TRUST-VLM, to enhance both the attack success rate and the diversity of successful test cases for VLMs. Specifically, TRUST-VLM is built upon the in-context learning to adversarially test a VLM on both image and text inputs. Furthermore, we involve feedback from the target VLM to improve the efficiency of test case generation. Extensive experiments show that TRUST-VLM not only outperforms traditional red-teaming techniques in generating diverse and effective adversarial cases but also provides actionable insights for model improvement. These findings highlight the importance of advanced red-teaming strategies in ensuring the reliability of VLMs.

</details>

---

## 60. Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation

- [ ] Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation | https://icml.cc/virtual/2025/poster/44642

- **Link**: https://icml.cc/virtual/2025/poster/44642

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-video (T2V) models like Sora have made significant strides in visualizing complex prompts, which is increasingly viewed as a promising path towards constructing the universal world simulator. Cognitive psychologists believe that the foundation for achieving this goal is the ability to understand intuitive physics. However, the capacity of these models to accurately represent intuitive physics remains largely unexplored. To bridge this gap, we introduce PhyGenBench, a comprehensive \textbf{Phy}sics \textbf{Gen}eration \textbf{Ben}chmark designed to evaluate physical commonsense correctness in T2V generation. PhyGenBench comprises 160 carefully crafted prompts across 27 distinct physical laws, spanning four fundamental domains, which could comprehensively assesses models' understanding of physical commonsense. Alongside PhyGenBench, we propose a novel evaluation framework called PhyGenEval. This framework employs a hierarchical evaluation structure utilizing appropriate advanced vision-language models and large language models to assess physical commonsense. Through PhyGenBench and PhyGenEval, we can conduct large-scale automated assessments of T2V models' understanding of physical commonsense, which align closely with human feedback. Our evaluation results and in-depth analysis demonstrate that current models struggle to generate videos that comply with physical commonsense. Moreover, simply scaling up models or employing prompt engineering techniques is insufficient to fully address the challenges presented by PhyGenBench (e.g., dynamic scenarios). We hope this study will inspire the community to prioritize the learning of physical commonsense in these models beyond entertainment applications. We will release the data and codes at https://github.com/OpenGVLab/PhyGenBench

</details>

---

## 61. CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization

- [ ] CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization | https://icml.cc/virtual/2025/poster/44709

- **Link**: https://icml.cc/virtual/2025/poster/44709

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning, which adapts vision-language models by freezing model parameters and opti- mizing only the prompt, has proven effective for task-specific adaptations. The core challenge in prompt tuning is improving specialization for a specific task and generalization for unseen domains. However, frozen encoders often produce misaligned features, leading to confusion between classes and limiting specialization. To overcome this issue, we propose a confusion-aware loss (CoA-loss) that improves specialization by refining the decision boundaries between confusing classes. Additionally, we mathematically demonstrate that a mixture model can enhance generalization without compromising specialization. This is achieved using confidence-aware weights (CoA- weights), which adjust the weights of each prediction in the mixture model based on its confidence within the class domains. Extensive experiments show that CoCoA-Mix, a mixture model with CoA-loss and CoA-weights, outperforms state-of-the-art methods by enhancing specialization and generalization. Our code is publicly available at https://github.com/url-kaist/CoCoA-Mix

</details>

---

## 62. Vision-Language Model Selection and Reuse for Downstream Adaptation

- [ ] Vision-Language Model Selection and Reuse for Downstream Adaptation | https://icml.cc/virtual/2025/poster/44712

- **Link**: https://icml.cc/virtual/2025/poster/44712

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained Vision-Language Models (VLMs) are becoming increasingly popular across various visual tasks, and several open-sourced VLM variants have been released. However, selecting the best-performing pre-trained VLM for a specific downstream task is challenging since no single VLM can achieve promising performance on all downstream tasks, and evaluating all available VLMs is impossible due to time and data limitations. To address this problem, this paper proposes a novel paradigm to select and reuse VLM for downstream tasks, called M odel L abel L earning ( MLL ). The proposal contains three key modules: model labeling , which assigns labels to each VLM to describe their specialty and utility; model selection , which matches the requirements of the target task with model labels; and model reuse , which applies selected VLMs to the target task in an ensemble manner. The proposal is highly computationally efficient and growable since the model labeling process is completed target task independent and the ability could grow with the number of candidate VLMs. We also introduce a new benchmark for evaluating VLM selection methods, including 49 VLMs and 17 target task datasets. Experimental results clearly demonstrate the effectiveness of the proposed method for selecting and reusing VLMs.

</details>

---

## 63. Be Confident: Uncovering Overfitting in MLLM Multi-Task Tuning

- [ ] Be Confident: Uncovering Overfitting in MLLM Multi-Task Tuning | https://icml.cc/virtual/2025/poster/44726

- **Link**: https://icml.cc/virtual/2025/poster/44726

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning Multimodal Large Language Models (MLLMs) in multi-task learning scenarios has emerged as an effective strategy for achieving cross-domain specialization. However, multi-task fine-tuning frequently induces performance degradation on open-response datasets. We posit that free-form answer generation primarily depends on language priors, and strengthening the integration of visual behavioral cues is critical for enhancing prediction robustness. In this work, we propose Noise Resilient Confidence Alignment to address the challenge of open-response overfitting during multi-task fine-tuning. Our approach prioritizes maintaining consistent prediction patterns in MLLMs across varying visual input qualities. To achieve this, we employ Gaussian perturbations to synthesize distorted visual inputs and enforce token prediction confidence alignment towards the normal visual branch. By explicitly linking confidence calibration to visual robustness, this method reduces over-reliance on language priors. We conduct extensive empirical evaluations across diverse multi-task downstream settings via popular MLLM architectures. The comprehensive experiment demonstrates the effectiveness of our method, showcasing its ability to alleviate open-response overfitting while maintaining satisfying multi-task fine-tuning performance.

</details>

---

## 64. Probing Visual Language Priors in VLMs

- [ ] Probing Visual Language Priors in VLMs | https://icml.cc/virtual/2025/poster/44730

- **Link**: https://icml.cc/virtual/2025/poster/44730

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) may over-rely on visual language priors from their training data rather than true visual reasoning. To investigate this, we introduce ViLP, a benchmark featuring deliberately out-of-distribution images synthesized via image generation models and out-of-distribution Q\&A pairs. Each question in ViLP is coupled with three potential answers and three corresponding images: one that can be resolved by text priors alone and two that demand visual reasoning. Although humans achieve near-perfect accuracy, modern VLMs falter; for instance, GPT-4o achieves only 66.17\% on ViLP. To alleviate this, we propose a self-improving framework in which models generate new VQA data and then apply pixel-level and semantic corruptions to form ``good-bad" image pairs for self-training. Our proposed training objective, Image-DPO, compels VLMs to focus more on the actual visual inputs, and we demonstrate its effectiveness in LLaVA-v1.5 and Cambrian. Project Page: \href{https://vilp-team.github.io/}{ViLP}.

</details>

---

## 65. Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting

- [ ] Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting | https://icml.cc/virtual/2025/poster/44762

- **Link**: https://icml.cc/virtual/2025/poster/44762

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in time series forecasting have explored augmenting models with text or vision modalities to improve accuracy. While text provides contextual understanding, it often lacks fine-grained temporal details. Conversely, vision captures intricate temporal patterns but lacks semantic context, limiting the complementary potential of these modalities. To address this, we propose Time-VLM, a novel multimodal framework that leverages pre-trained Vision-Language Models (VLMs) to bridge temporal, visual, and textual modalities for enhanced forecasting. Our framework comprises three key components: (1) a Retrieval-Augmented Learner, which extracts enriched temporal features through memory bank interactions; (2) a Vision-Augmented Learner, which encodes time series as informative images; and (3) a Text-Augmented Learner, which generates contextual textual descriptions. These components collaborate with frozen pre-trained VLMs to produce multimodal embeddings, which are then fused with temporal features for final prediction. Extensive experiments demonstrate that Time-VLM achieves superior performance, particularly in few-shot and zero-shot scenarios, thereby establishing a new direction for multimodal time series forecasting. Code is available at https://github.com/CityMind-Lab/ICML25-TimeVLM.

</details>

---

## 66. $\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation

- [ ] $\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation | https://icml.cc/virtual/2025/poster/44785

- **Link**: https://icml.cc/virtual/2025/poster/44785

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current video-language models struggle with long-video understanding due to limited context lengths and reliance on sparse frame subsampling, which often leads to information loss. In this paper, we introduce $\infty$-Video, which is able to process arbitrarily long videos through a continuous-time long-term memory (LTM) consolidation mechanism. Our framework augments video Q-formers by making them able to process unbounded video contexts efficiently and without requiring additional training. Through continuous attention, our approach dynamically allocates higher granularity to the most relevant video segments, forming "sticky" memories which evolve over time. Experiments with Video-LLaMA and VideoChat2 demonstrate improved performance in video question-answering tasks, showcasing the potential of continuous-time LTM mechanisms to enable scalable and training-free comprehension of long videos.

</details>

---

## 67. Compositional Condition Question Answering in Tabular Understanding

- [ ] Compositional Condition Question Answering in Tabular Understanding | https://icml.cc/virtual/2025/poster/44789

- **Link**: https://icml.cc/virtual/2025/poster/44789

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) for tabular understanding have made significant progress in tasks such as financial report analysis and public data tests. However, our comprehensive analysis shows that these models are still limited in certain simple scenarios, particularly when handling compositional conditions in QA. Further investigation reveals that the poor performance can be attributed to two main challenges: the visual encoder's inability to accurately recognize the content of a row, and the model's tendency to overlook conditions in the question.To address these, we introduce a new Compositional Condition Tabular Understanding method, called {\sc CoCoTab}. Specifically, to capture the structural relationships within tables, we enhance the visual encoder with additional row and column patches. Moreover, we introduce the conditional tokens between the visual patches and query embeddings, ensuring the model focuses on relevant parts of the table according to the conditions specified in the query.Additionally, we also introduce the Massive Multimodal Tabular Understanding (MMTU) benchmark, which comprehensively assesses the full capabilities of MLLMs in tabular understanding. Our proposed method achieves state-of-the-art performance on both existing tabular understanding benchmarks and MMTU.Our code can be available at \url{https://github.com/LAMDA-Tabular/MMTU}.

</details>

---

## 68. Fine-Grained Captioning of Long Videos through Scene Graph Consolidation

- [ ] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation | https://icml.cc/virtual/2025/poster/44795

- **Link**: https://icml.cc/virtual/2025/poster/44795

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language models have led to impressive progress in caption generation for images and short video clips. However, these models remain constrained by their limited temporal receptive fields, making it difficult to producecoherent and comprehensive captions for long videos. While several methods have been proposed to aggregate information across video segments, they often rely on supervised fine-tuning or incur significant computational overhead. To address these challenges, we introduce a novel framework for long video captioning based on graph consolidation. Our approach first generates segment-level captions, corresponding to individual frames or short video intervals, using off-the-shelf visual captioning models. These captions are then parsed into individual scene graphs, which are subsequently consolidated into a unified graph representation that preserves both holistic context and fine-grained details throughout the video. A lightweight graph-to-text decoder then produces the final video-level caption. This framework effectively extends the temporal understanding capabilities of existing models without requiring any additional fine-tuning on long video datasets. Experimental results show that our method significantly outperforms existing LLM-based consolidation approaches, achieving strong zero-shot performance while substantially reducing computational costs.

</details>

---

## 69. ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding

- [ ] ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding | https://icml.cc/virtual/2025/poster/44816

- **Link**: https://icml.cc/virtual/2025/poster/44816

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Structured image understanding, such as interpreting tables and charts, requires strategically refocusing across various structures and texts within an image, forming a reasoning sequence to arrive at the final answer. However, current multimodal large language models (LLMs) lack this multihop selective attention capability. In this work, we introduce ReFocus, a simple yet effective framework that equips multimodal LLMs with the ability to generate ``visual thoughts'' by performing visual editing on the input image through code, shifting and refining their visual focuses. Specifically, ReFocus enables multimodal LLMs to generate Python codes to call tools and modify the input image, sequentially drawing boxes, highlighting sections, and masking out areas, thereby enhancing the visual reasoning process. We experiment upon a wide range of structured image understanding tasks involving tables and charts. ReFocus largely improves performance on all tasks over GPT-4o without visual editing, yielding an average gain of 11.0% on table tasks and 6.8% on chart tasks. We present an in-depth analysis of the effects of different visual edits, and reasons why ReFocus can improve the performance without introducing additional information. Further, we collect a 14k training set using ReFocus, and prove that such visual chain-of-thought with intermediate information offers a better supervision than standard VQA data, reaching a 8.0% average gain over the same model trained with QA pairs and 2.6% over CoT.

</details>

---

## 70. Do Vision-Language Models Really Understand Visual Language?

- [ ] Do Vision-Language Models Really Understand Visual Language? | https://icml.cc/virtual/2025/poster/44858

- **Link**: https://icml.cc/virtual/2025/poster/44858

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual language is a system of communication that conveys information through symbols, shapes, and spatial arrangements. Diagrams are a typical example of a visual language depicting complex concepts and their relationships in the form of an image. The symbolic nature of diagrams presents significant challenges for building models capable of understanding them. Yet, recent studies seem to suggest that Large Vision-Language Models (LVLMs) can even tackle complex reasoning tasks involving diagrams. In this paper, we investigate this phenomenon by developing a comprehensive test suite to evaluate the diagram comprehension capability of LVLMs. Our test suite uses a variety of questions focused on concept entities and their relationships over a set of synthetic as well as real diagrams across several domains to evaluate the recognition and reasoning abilities of models. Our evaluation of six LVLMs shows that while these models can accurately identify and reason about entities, their ability to understand relationships is notably limited. Further testing reveals that the decent performance on diagram understanding largely stems from leveraging their background knowledge as shortcuts to identify and reason about the relational information. Thus, we conclude that LVLMs have a limited capability for genuine diagram understanding, and their impressive performance in diagram reasoning is an illusion emanating from other confounding factors, such as the background knowledge in the models.

</details>

---

## 71. ERICT: Enhancing Robustness by Identifying Concept Tokens in Zero-Shot Vision Language Models

- [ ] ERICT: Enhancing Robustness by Identifying Concept Tokens in Zero-Shot Vision Language Models | https://icml.cc/virtual/2025/poster/44869

- **Link**: https://icml.cc/virtual/2025/poster/44869

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have revolutionized the field of machine learning, demonstrating exceptional performance across a wide range of tasks. However, their robustness remains vulnerable to the spurious-correlation problem. Existing works often involve fine-tuning the model with labeled data or relying on large language models (LLMs) to generate more complex prompts. Although effective to some extent, these methods introduce new challenges, including additional computational costs and dependence on the quality of prompts without fully utilizing the vision modality. To address these limitations, we propose a novel method named ERICT to Enhance model Robustness by Identifying Concept Tokens. ERICT mitigates spurious correlation directly in the inference stage and comprises two key steps: (1) Identify concept tokens capturing invariant features through auxiliary prompts to generate a token-level mask. (2) Apply the mask to the attention weights of the CLS token in the vision encoder to help the model focus on the relevant image region. Extensive experiments show that ERICT significantly improves the overall performance including that of the worst group, and achieves new state-of-the-art results.

</details>

---

## 72. SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models

- [ ] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models | https://icml.cc/virtual/2025/poster/44870

- **Link**: https://icml.cc/virtual/2025/poster/44870

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Exploration is a cornerstone of reinforcement learning (RL). Intrinsic motivation attempts to decouple exploration from external, task-based rewards. However, established approaches to intrinsic motivation that follow general principles such as information gain, often only uncover low-level interactions. In contrast, children’s play suggests that they engage in meaningful high-level behavior by imitating or interacting with their caregivers. Recent work has focused on using foundation models to inject these semantic biases into exploration. However, these methods often rely on unrealistic assumptions, such as language-embedded environments or access to high-level actions. We propose SEmaNtically Sensible ExploratIon (SENSEI), a framework to equip model-based RL agents with an intrinsic motivation for semantically meaningful behavior. SENSEI distills a reward signal of interestingness from Vision Language Model (VLM) annotations, enabling an agent to predict these rewards through a world model. Using model-based RL, SENSEI trains an exploration policy that jointly maximizes semantic rewards and uncertainty. We show that in both robotic and video game-like simulations SENSEI discovers a variety of meaningful behaviors from image observations and low-level actions. SENSEI provides a general tool for learning from foundation model feedback, a crucial research direction, as VLMs become more powerful.

</details>

---

## 73. Parrot: Multilingual Visual Instruction Tuning

- [ ] Parrot: Multilingual Visual Instruction Tuning | https://icml.cc/virtual/2025/poster/44886

- **Link**: https://icml.cc/virtual/2025/poster/44886

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of Multimodal Large Language Models (MLLMs), such as GPT-4, marks a significant step toward artificial general intelligence. Existing methods typically align vision encoders with LLMs via supervised fine-tuning (SFT), but this often deteriorates their ability to handle multiple languages as training progresses. We empirically observe that imbalanced SFT datasets, largely English-centric, degrade performance on non-English languages due to the failure in multilingual token alignment. To address this, we propose Parrot, a novel approach that leverages textual guidance for visual token alignment at the language level. Parrot conditions visual tokens on diverse language inputs and uses Mixture-of-Experts (MoE) to align multilingual tokens. By computing cross-attention between initial visual features and textual embeddings, we select the most relevant experts, converting visual tokens into language-specific representations. Additionally, we introduce the Massive Multilingual Multimodal Benchmark (MMMB), a new benchmark comprising 6 languages, 15 categories, and 12,000 questions, to assess multilingual capabilities. Parrot achieves state-of-the-art performance on both the multilingual benchmarks and a wide range of multimodal tasks. Code and dataset are available at: \url{https://github.com/AIDC-AI/Parrot}.

</details>

---

## 74. Perception in Reflection

- [ ] Perception in Reflection | https://icml.cc/virtual/2025/poster/44894

- **Link**: https://icml.cc/virtual/2025/poster/44894

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a perception in reflection paradigm designed to transcend the limitations of current large vision-language models (LVLMs), which are expected yet often fail to achieve perfect perception initially. Specifically, we propose Reflective Perception (RePer), a dual-model reflection mechanism that systematically alternates between policy and critic models, enables iterative refinement of visual perception. This framework is powered by Reflective Perceptual Learning (RPL), which reinforces intrinsic reflective capabilities through a methodically constructed visual reflection dataset and reflective unlikelihood training Comprehensive experimental evaluation demonstrates RePer's quantifiable improvements in image understanding, captioning precision, and hallucination reduction. Notably, RePer achieves strong alignment between model attention patterns and human visual focus, while RPL optimizes fine-grained and free-form preference alignment. These advancements establish perception in reflection as a robust paradigm for future multimodal agents, particularly in tasks requiring complex reasoning and multi-step manipulation. Project Page: https://weiyana.github.io/Perception-in-Reflection

</details>

---

## 75. LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models

- [ ] LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models | https://icml.cc/virtual/2025/poster/44918

- **Link**: https://icml.cc/virtual/2025/poster/44918

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces Llavaguard, a suite of VLM-based vision safeguards that address the critical need for reliable tools in the era of large-scale data and models. To this end, we establish a novel open framework, describing a customizable safety taxonomy, data preprocessing, augmentation, and training setup. For teaching a VLM safeguard on safety, we further create a multimodal safety dataset with high-quality human expert annotations, where each image is labeled with a safety rating, category, and rationale. We also employ advanced augmentations to support context-specific assessments. The resulting Llavaguard models, ranging from 0.5B to 7B, serve as a versatile tool for evaluating the safety compliance of visual content against flexible policies. In comprehensive experiments, Llavaguard outperforms both state-of-the-art safeguards and VLMs in accuracy and in flexibly handling different policies. Additionally, we demonstrate Llavaguard's performance in two real-world applications: large-scale dataset annotation and moderation of text-to-image models. We make our entire framework, including the dataset, model weights, and training code, publicly available at https://ml-research.github.io/human-centered-genai/projects/llavaguard.

</details>

---

## 76. LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding

- [ ] LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding | https://icml.cc/virtual/2025/poster/44939

- **Link**: https://icml.cc/virtual/2025/poster/44939

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown promising progress in understanding and analyzing video content. However, processing long videos remains a significant challenge constrained by LLM's context size. To address this limitation, we propose \textbf{LongVU}, a spatiotemporal adaptive compression mechanism that reduces the number of video tokens while preserving visual details of long videos. Our idea is based on leveraging cross-modal query and inter-frame dependencies to adaptively reduce temporal and spatial redundancy in videos. Specifically, we leverage DINOv2 features to remove redundant frames that exhibit high similarity. Then we utilize text-guided cross-modal query for selective frame feature reduction. Further, we perform spatial token reduction across frames based on their temporal dependencies. Our adaptive compression strategy effectively processes a large number of frames with little visual information loss within given context length. Our LongVU consistently surpass existing methods across a variety of video understanding benchmarks, especially on hour-long video understanding tasks such as VideoMME and MLVU. Given a light-weight LLM, our LongVU also scales effectively into a smaller size with state-of-the-art video understanding performance.

</details>

---

## 77. FOCoOp: Enhancing Out-of-Distribution Robustness in Federated Prompt Learning for Vision-Language Models

- [ ] FOCoOp: Enhancing Out-of-Distribution Robustness in Federated Prompt Learning for Vision-Language Models | https://icml.cc/virtual/2025/poster/44973

- **Link**: https://icml.cc/virtual/2025/poster/44973

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Federated prompt learning (FPL) for vision-language models is a powerful approach to collaboratively adapt models across distributed clients while preserving data privacy. However, existing FPL approaches suffer from a trade-off between performance and robustness, particularly in out-of-distribution (OOD) shifts, limiting their reliability in real-world scenarios. The inherent in-distribution (ID) data heterogeneity among different clients makes it more challenging to maintain this trade-off. To fill this gap, we introduce a Federated OOD-aware Context Optimization (FOCoOp) framework, which captures diverse distributions among clients using ID global prompts, local prompts, and OOD prompts. Specifically, FOCoOp leverages three sets of prompts to create both class-level and distribution-level separations, which adapt to OOD shifts through bi-level distributionally robust optimization. Additionally, FOCoOp improves the discrimination consistency among clients, i.e., calibrating global prompts, seemly OOD prompts, and OOD prompts by Semi-unbalanced optimal transport. The extensive experiments on real-world datasets demonstrate that FOCoOp effectively captures decentralized heterogeneous distributions and enhances robustness of different OOD shifts. The project is available at GitHub.

</details>

---

## 78. Retrieval-Augmented Perception: High-resolution Image Perception Meets Visual RAG

- [ ] Retrieval-Augmented Perception: High-resolution Image Perception Meets Visual RAG | https://icml.cc/virtual/2025/poster/44979

- **Link**: https://icml.cc/virtual/2025/poster/44979

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs).  To drive progress beyond the limits of heuristic methods, this paper advances HR perception capabilities of MLLMs by harnessing cutting-edge long-context techniques such as retrieval-augmented generation (RAG). Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose Retrieval-Augmented Perception (RAP), a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed Spatial-Awareness Layout. To accommodate different tasks, the proposed Retrieved-Exploration Search (RE-Search) dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of RAP, with LLaVA-v1.5-13B achieving a 43\% improvement on $V^*$ Bench and 19\% on HR-Bench. Code is available at https://github.com/DreamMr/RAP.

</details>

---

## 79. Open-Det: An Efficient Learning Framework for Open-Ended Detection

- [ ] Open-Det: An Efficient Learning Framework for Open-Ended Detection | https://icml.cc/virtual/2025/poster/45000

- **Link**: https://icml.cc/virtual/2025/poster/45000

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-Ended object Detection (OED) is a novel and challenging task that detects objects and generates their category names in a free-form manner, without requiring additional vocabularies during inference. However, the existing OED models, such as GenerateU, require large-scale datasets for training, suffer from slow convergence, and exhibit limited performance. To address these issues, we present a novel and efficient Open-Det framework, consisting of four collaborative parts. Specifically, Open-Det accelerates model training in both the bounding box and object name generation process by reconstructing the Object Detector and the Object Name Generator. To bridge the semantic gap between Vision and Language modalities, we propose a Vision-Language Aligner with V-to-L and L-to-V alignment mechanisms, incorporating with the Prompts Distiller to transfer knowledge from the VLM into VL-prompts, enabling accurate object name generation for the LLM. In addition, we design a Masked Alignment Loss to eliminate contradictory supervision and introduce a Joint Loss to enhance classification, resulting in more efficient training. Compared to GenerateU, Open-Det, using only 1.5% of the training data (0.077M vs. 5.077M), 20.8% of the training epochs (31 vs. 149), and fewer GPU resources (4 V100 vs. 16 A100), achieves even higher performance (+1.0% in APr). The source codes are available at: https://github.com/Med-Process/Open-Det.

</details>

---

## 80. HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation

- [ ] HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation | https://icml.cc/virtual/2025/poster/45007

- **Link**: https://icml.cc/virtual/2025/poster/45007

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present HealthGPT , a powerful Medical Large Vision-Language Model (Med-LVLM) that integrates medical visual comprehension and generation capabilities within a unified autoregressive paradigm. Our bootstrapping philosophy is to progressively adapt heterogeneous comprehension and generation knowledge to pre-trained Large Language Models (LLMs). This is achieved through a novel heterogeneous low-rank adaptation (H-LoRA) technique, which is complemented by a tailored hierarchical visual perception (HVP) approach and a three-stage learning strategy (TLS) . To effectively learn the HealthGPT, we devise a comprehensive medical domain-specific comprehension and generation dataset called VL-Health . Experimental results demonstrate exceptional performance and scalability of HealthGPT in medical visual unified tasks. Our project can be accessed at https://github.com/DCDmllm/HealthGPT.

</details>

---

## 81. Improving Zero-Shot Adversarial Robustness in Vision-Language Models by Closed-form Alignment of Adversarial Path Simplices

- [ ] Improving Zero-Shot Adversarial Robustness in Vision-Language Models by Closed-form Alignment of Adversarial Path Simplices | https://icml.cc/virtual/2025/poster/45018

- **Link**: https://icml.cc/virtual/2025/poster/45018

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) such as CLIP excel at zero-shot classification due to large-scale pre-training but are vulnerable to adversarial examples. Adversarial fine-tuning robustifies zero-shot models by aligning prediction scores of individual adversaries with their clean counterparts, which typically overlooks intermediate adversarial samples along the adversarial trajectory crossing the decision boundary. Such intermediate adversaries and their vicinity produce informative representations capturing the decision boundary in detail. They can be improved by sampling adversarial candidates from simplices formed by joining two consecutive vertices on the adversarial trajectory and their clean counterpart. However, sampling simplices for adversaries is very costly. To train robust VLM, we overcome these limitations by Taylor expansion and formulating an upper-bound of alignment loss that depends on the Jacobian/Hessian obtained at clean samples. As regions between clean and intermediate adversarial samples capture a larger decision landscape, we robustify VLM by plausible adversaries from simplices by our closed-form formulation equivalent to infinite uniform sampling of the simplex. We obtain state-of-the-art robustness across 15 datasets and diverse vision-language tasks.

</details>

---

## 82. Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale

- [ ] Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale | https://icml.cc/virtual/2025/poster/45035

- **Link**: https://icml.cc/virtual/2025/poster/45035

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) show potential as computer agents, enhancing productivity and software accessibility in multi-modal tasks. However, measuring agent performance in sufficiently realistic and complex environments becomes increasingly challenging as: (i) most benchmarks are limited to specific modalities/domains (e.g., text-only, web navigation, Q&A) and (ii) full benchmark evaluations are slow (on order of magnitude of multiple hours/days) given the multi-step sequential nature of tasks.To address these challenges, we introduce Windows Agent Arena: a general environment focusing exclusively on the Windows operating system (OS) where agents can operate freely within a real OS to use the same applications and tools available to human users when performing tasks.We create 150+ diverse tasks across representative domains that require agentic abilities in planning, screen understanding, and tool usage.Our benchmark is scalable and can be seamlessly parallelized for a full benchmark evaluation in as little as $20$ minutes.Our work not only speeds up the development and evaluation cycle of multi-modal agents, but also highlights and analyzes existing shortfalls in the agentic  abilities of several multimodal LLMs as agents within the Windows computing environment---with the best achieving only a 19.5\% success rate compared to a human success rate of 74.5\%.

</details>

---

## 83. On Path to Multimodal Generalist: General-Level and General-Bench

- [ ] On Path to Multimodal Generalist: General-Level and General-Bench | https://icml.cc/virtual/2025/poster/45047

- **Link**: https://icml.cc/virtual/2025/poster/45047

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Multimodal Large Language Model (MLLM) is currently experiencing rapid growth, driven by the advanced capabilities of language-based LLMs. Unlike their specialist predecessors, existing MLLMs are evolving towards a Multimodal Generalist paradigm. Initially limited to understanding multiple modalities, these models have advanced to not only comprehend but also generate across modalities. Their capabilities have expanded from coarse-grained to fine-grained multimodal understanding and from supporting singular modalities to accommodating a wide array of or even arbitrary modalities. To assess the capabilities of various MLLMs, a diverse array of benchmark test sets has been proposed. This leads to a critical question: Can we simply assume that higher performance across tasks indicates a stronger MLLM capability, bringing us closer to human-level AI? We argue that the answer is not as straightforward as it seems. In this project, we introduce an evaluation framework to delineate the capabilities and behaviors of current multimodal generalists. This framework, named General-Level , establishes 5-scale levels of MLLM performance and generality, offering a methodology to compare MLLMs and gauge the progress of existing systems towards more robust multimodal generalists and, ultimately, towards AGI (Artificial General Intelligence). Central to our framework is the use of Synergy as the evaluative criterion, categorizing capabilities based on whether MLLMs preserve synergy across comprehension and generation, as well as across multimodal interactions.To evaluate the comprehensive abilities of various generalists, we present a massive multimodal benchmark, General-Bench , which encompasses a broader spectrum of skills, modalities, formats, and capabilities, including over 700 tasks and 325,800 instances. The evaluation results that involve over 100 existing state-of-the-art MLLMs uncover the capability rankings of generalists, highlighting the challenges in reaching genuine AI. We expect this project to pave the way for future research on next-generation multimodal foundation models, providing a robust infrastructure to accelerate the realization of AGI.Project Page: https://generalist.top/,Leaderboard: https://generalist.top/leaderboard/,Benchmark: https://huggingface.co/General-Level/.

</details>

---

## 84. DiffusionVLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression

- [ ] DiffusionVLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression | https://icml.cc/virtual/2025/poster/45061

- **Link**: https://icml.cc/virtual/2025/poster/45061

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present DiffusionVLA, a novel framework that integrates autoregressive reasoning with diffusion policies to address the limitations of existing methods: while autoregressive Vision-Language-Action (VLA) models lack precise and robust action generation, diffusion-based policies inherently lack reasoning capabilities. Central to our approach is autoregressive reasoning — a task decomposition and explanation process enabled by a pre-trained VLM — to guide diffusion-based action policies. To tightly couple reasoning with action generation, we introduce a reasoning injection module that directly embeds self-generated reasoning phrases into the policy learning process. The framework is simple, flexible, and efficient, enabling seamless deployment across diverse robotic platforms.We conduct extensive experiments using multiple real robots to validate the effectiveness of DiVLA. Our tests include a challenging factory sorting task, where DiVLA successfully categorizes objects, including those not seen during training. The reasoning injection module enhances interpretability, enabling explicit failure diagnosis by visualizing the model’s decision process. Additionally, we test DiVLA on a zero-shot bin-picking task, achieving \textbf{63.7\% accuracy on 102 previously unseen objects}. Our method demonstrates robustness to visual changes, such as distractors and new backgrounds, and easily adapts to new embodiments. Furthermore, DiVLA can follow novel instructions and retain conversational ability. Notably, DiVLA is data-efficient and fast at inference; our smallest DiVLA-2B runs 82Hz on a single A6000 GPU. Finally, we scale the model from 2B to 72B parameters, showcasing improved generalization capabilities with increased model size.

</details>

---

## 85. UP-VLA:  A Unified Understanding and Prediction Model for Embodied Agent

- [ ] UP-VLA:  A Unified Understanding and Prediction Model for Embodied Agent | https://icml.cc/virtual/2025/poster/45080

- **Link**: https://icml.cc/virtual/2025/poster/45080

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language-Action (VLA) models have leveraged pre-trained Vision-Language Models (VLMs) to improve the generalization capabilities.VLMs, typically pre-trained on vision-language understanding tasks, provide rich semantic knowledge and reasoning abilities. However, prior research has shown that VLMs often focus onhigh-level semantic content and neglect low-level features, limiting their ability to capture detailed spatial information and understand physical dynamics.These aspects, which are crucial for embodied control tasks, remain underexplored in existing pre-training paradigms.In this paper, we investigate the training paradigm for VLAs, and introduce \textbf{UP-VLA}, a \textbf{U}nified VLA model training with both multi-modal \textbf{U}nderstanding and future \textbf{P}rediction objectives, enhancing both high-level semantic comprehension and low-level spatial understanding. Experimental results show that UP-VLA achieves a 33\% improvement on the Calvin ABC-D benchmark compared to the previous state-of-the-art method. Additionally, UP-VLA demonstrates improved success rates in real-world manipulation tasks, particularly those requiring precise spatial information.

</details>

---

## 86. MM-RLHF: The Next Step Forward in Multimodal LLM Alignment

- [ ] MM-RLHF: The Next Step Forward in Multimodal LLM Alignment | https://icml.cc/virtual/2025/poster/45124

- **Link**: https://icml.cc/virtual/2025/poster/45124

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing efforts to align multimodal large language models (MLLMs) with human preferences have only achieved progress in narrow areas, such as hallucination reduction, but remain limited in practical applicability and generalizability. To this end, we introduce MM-RLHF , a dataset containing 120k fine-grained, human-annotated preference comparison pairs. This dataset represents a substantial advancement over existing resources, offering superior size, diversity, annotation granularity, and quality.  Leveraging this dataset, we propose several key innovations to improve both the quality of reward models and the efficiency of alignment algorithms. Notably, we introduce the Critique-Based Reward Model , which generates critiques of model outputs before assigning scores, offering enhanced interpretability and more informative feedback compared to traditional scalar reward mechanisms.  Additionally, we propose Dynamic Reward Scaling , a method that adjusts the loss weight of each sample according to the reward signal, thereby optimizing the use of high-quality comparison pairs.  Our approach is rigorously evaluated across 10 distinct dimensions, encompassing 27 benchmarks, with results demonstrating significant and consistent improvements in model performance (Figure.1).

</details>

---

## 87. OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction

- [ ] OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction | https://icml.cc/virtual/2025/poster/45131

- **Link**: https://icml.cc/virtual/2025/poster/45131

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language-Action (VLA) models aim to predict robotic actions based on visual observations and language instructions. Existing approaches require fine-tuning pre-trained vision-language models (VLMs) as visual and language features are independently fed into downstream policies, degrading the pre-trained semantic alignments. We propose OTTER, a novel VLA architecture that leverages these existing alignments through explicit, text-aware visual feature extraction. Instead of processing all visual features, OTTER selectively extracts and passes only task-relevant visual features that are semantically aligned with the language instruction to the policy transformer. This allows OTTER to keep the pre-trained vision-language encoders frozen. Thereby, OTTER preserves and utilizes the rich semantic understanding learned from large-scale pre-training, enabling strong zero-shot generalization capabilities. In simulation and real-world experiments, OTTER significantly outperforms existing VLA models, demonstrating strong zero-shot generalization to novel objects and environments. Video, code, checkpoints, and dataset: https://ottervla.github.io/.

</details>

---

## 88. SECOND: Mitigating Perceptual Hallucination in Vision-Language Models via Selective and Contrastive Decoding

- [ ] SECOND: Mitigating Perceptual Hallucination in Vision-Language Models via Selective and Contrastive Decoding | https://icml.cc/virtual/2025/poster/45215

- **Link**: https://icml.cc/virtual/2025/poster/45215

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant advancements in Vision-Language Models (VLMs), the performance of existing VLMs remains hindered by object hallucination, a critical challenge to achieving accurate visual understanding. To address this issue, we propose SECOND: Selective and Contrastive Decoding, a novel approach that enables VLMs to effectively leverage multi-scale visual information with an object-centric manner, closely aligning with human visual perception. SECOND progressively selects and integrates multi-scale visual information, facilitating a more precise interpretation of images. By contrasting these visual information iteratively, SECOND significantly reduces perceptual hallucinations and outperforms a wide range of benchmarks. Our theoretical analysis and experiments highlight the largely unexplored potential of multi-scale application in VLMs, showing that prioritizing and contrasting across scales outperforms existing methods.

</details>

---

## 89. Robust Multimodal Large Language Models Against Modality Conflict

- [ ] Robust Multimodal Large Language Models Against Modality Conflict | https://icml.cc/virtual/2025/poster/45224

- **Link**: https://icml.cc/virtual/2025/poster/45224

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the impressive capabilities of multimodal large language models (MLLMs) in vision-language tasks, they are prone to hallucinations in real-world scenarios. This paper investigates the hallucination phenomenon in MLLMs from the perspective of modality conflict. Unlike existing works focusing on the conflicts between model responses and inputs, we study the inherent conflicts in inputs from different modalities that place MLLMs in a dilemma and directly lead to hallucinations. We formally define the modality conflict and construct a dataset named Multimodal Modality Conflict (MMMC) to simulate this phenomenon in vision-language tasks. Three methods based on prompt engineering, supervised fine-tuning, and reinforcement learning are proposed to alleviate the hallucination caused by modality conflict. Extensive experiments are conducted on the MMMC dataset to analyze the merits and demerits of these methods. Our results show that the reinforcement learning method achieves the best performance in mitigating the hallucination under modality conflict, while the supervised fine-tuning method shows promising and stable performance. Our work sheds light on the unnoticed modality conflict that leads to hallucinations and provides more insights into the robustness of MLLMs.

</details>

---

## 90. SAE-V: Interpreting Multimodal Models for Enhanced Alignment

- [ ] SAE-V: Interpreting Multimodal Models for Enhanced Alignment | https://icml.cc/virtual/2025/poster/45246

- **Link**: https://icml.cc/virtual/2025/poster/45246

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the integration of image modality, the semantic space of multimodal large language models (MLLMs) is more complex than text-only models, making their interpretability more challenging and their alignment less stable, particularly susceptible to low-quality data, which can lead to inconsistencies between modalities, hallucinations, and biased outputs. As a result, developing interpretability methods for MLLMs is crucial for improving alignment quality and efficiency. In text-only LLMs, Sparse Autoencoders (SAEs) have gained attention for their ability to interpret latent representations. However, extending SAEs to multimodal settings presents new challenges due to modality fusion and the difficulty of isolating cross-modal representations. To address these challenges, we introduce SAE-V, a mechanistic interpretability framework that extends the SAE paradigm to MLLMs. By identifying and analyzing interpretable features along with their corresponding data, SAE-V enables fine-grained interpretation of both model behavior and data quality, facilitating a deeper understanding of cross-modal interactions and alignment dynamics. Moreover, by utilizing cross-modal feature weighting, SAE-V provides an intrinsic data filtering mechanism to enhance model alignment without requiring additional models. Specifically, when applied to the alignment process of MLLMs, SAE-V-based data filtering methods could achieve more than 110% performance with less than 50% data. Our results highlight SAE-V’s ability to enhance interpretability and alignment in MLLMs, providing insights into their internal mechanisms.

</details>

---

## 91. Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage

- [ ] Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage | https://icml.cc/virtual/2025/poster/45289

- **Link**: https://icml.cc/virtual/2025/poster/45289

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) excel at generating highly detailed captions but often produce hallucinations. Our analysis reveals that existing hallucination detection methods struggle with detailed captions. We attribute this to the increasing reliance of MLLMs on their generated text, rather than the input image, as the sequence length grows. To address this issue, we propose a multiagent approach that leverages LLM-MLLM collaboration to correct given captions. Additionally, we introduce an evaluation framework and a benchmark dataset to facilitate the systematic analysis of detailed captions. Our experiments demonstrate that the proposed evaluation method aligns better with human judgments of factuality than existing metrics. Moreover, we show that current approaches for enhancing MLLM factuality often fail in hyper-detailed image captioning tasks. In contrast, our approach significantly enhances the factual accuracy of captions, even improving those generated by GPT-4V. Finally, we highlight a limitation of VQA-centric benchmarking by demonstrating that an MLLM's performance on VQA benchmarks may not correlate with its ability to generate detailed image captions.

</details>

---

## 92. Bongard in Wonderland: Visual Puzzles that Still Make AI Go Mad?

- [ ] Bongard in Wonderland: Visual Puzzles that Still Make AI Go Mad? | https://icml.cc/virtual/2025/poster/45299

- **Link**: https://icml.cc/virtual/2025/poster/45299

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, newly developed Vision-Language Models (VLMs), such as OpenAI's o1, have emerged, seemingly demonstrating advanced reasoning capabilities across text and image modalities. However, the depth of these advances in language-guided perception and abstract reasoning remains underexplored, and it is unclear whether these models can truly live up to their ambitious promises. To assess the progress and identify shortcomings, we enter the wonderland of Bongard problems, a set of classic visual reasoning puzzles that require human-like abilities of pattern recognition and abstract reasoning. With our extensive evaluation setup, we show that while VLMs occasionally succeed in identifying discriminative concepts and solving some of the problems, they frequently falter. Surprisingly, even elementary concepts that may seem trivial to humans, such as simple spirals, pose significant challenges. Moreover, when explicitly asked to recognize ground truth concepts, they continue to falter, suggesting not only a lack of understanding of these elementary visual concepts but also an inability to generalize to unseen concepts. We compare the results of VLMs to human performance and observe that a significant gap remains between human visual reasoning capabilities and machine cognition.

</details>

---

## 93. Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin

- [ ] Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin | https://icml.cc/virtual/2025/poster/45347

- **Link**: https://icml.cc/virtual/2025/poster/45347

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adapting vision-language models (VLMs) to downstream tasks with pseudolabels has gained increasing attention. A major obstacle is that the pseudolabels generated by VLMs tend to be imbalanced, leading to inferior performance.While existing methods have explored various strategies to address this, the underlying causes of imbalance remain insufficiently investigated.To fill this gap, we delve into imbalanced pseudolabels and identify two primary contributing factors: concept mismatch and concept confusion. To mitigate these two issues, we propose a novel framework incorporating concept alignment and confusion-aware calibrated margin mechanisms. The core of our approach lies in enhancing underperforming classes and promoting balanced predictions across categories, thus mitigating imbalance. Extensive experiments on six benchmark datasets with three learning paradigms demonstrate that the proposed method effectively enhances the accuracy and balance of pseudolabels, achieving a relative improvement of 6.29\% over the SoTA method. Our code is avaliable at https://github.com/Noahwangyuchen/CAP

</details>

---

## 94. Explanatory Instructions: Towards Unified Vision Tasks Understanding and Zero-shot Generalization

- [ ] Explanatory Instructions: Towards Unified Vision Tasks Understanding and Zero-shot Generalization | https://icml.cc/virtual/2025/poster/45375

- **Link**: https://icml.cc/virtual/2025/poster/45375

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Computer Vision (CV) has yet to fully achieve the zero-shot task generalization observed in Natural Language Processing (NLP), despite following many of the milestones established in NLP, such as large transformer models, extensive pre-training, and the auto-regression paradigm, among others. In this paper, we rethink the reality that CV adopts discrete and terminological task definitions (e.g., "image segmentation"), and conjecture it is a key barrier that hampers zero-shot task generalization. Our hypothesis is that without truly understanding previously-seen tasks—due to these terminological definitions—deep models struggle to generalize to novel tasks. To verify this, we introduce Explanatory Instructions, which provide an intuitive way to define CV task objectives through detailed linguistic transformations from input images to outputs. We create a large-scale dataset comprising 12 million "image input $\to$ explanatory instruction $\to$ output" triplets, and train an auto-regressive-based vision-language model (AR-based VLM) that takes both images and explanatory instructions as input. By learning to follow these instructions, the AR-based VLM achieves instruction-level zero-shot capabilities for previously-seen tasks and demonstrates strong zero-shot generalization for unseen CV tasks.  Code and dataset will be open-sourced.

</details>

---

## 95. From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection

- [ ] From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection | https://icml.cc/virtual/2025/poster/45421

- **Link**: https://icml.cc/virtual/2025/poster/45421

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pretrained vision-language models (VLMs), e.g., CLIP, demonstrate impressive zero-shot capabilities on downstream tasks. Prior research highlights the crucial role of visual augmentation techniques, like random cropping, in alignment with fine-grained class descriptions generated by large language models (LLMs), significantly enhancing zero-shot performance by incorporating multi-view information. However, the inherent randomness of these augmentations can inevitably introduce background artifacts and cause models to overly focus on local details, compromising global semantic understanding. To address these issues, we propose an A ttention- B ased S election ( ABS ) method from local details to global context, which applies attention-guided cropping in both raw images and feature space, supplement global semantic information through strategic feature selection. Additionally, we introduce a soft matching technique to effectively filter LLM descriptions for better alignment. ABS achieves state-of-the-art performance on out-of-distribution generalization and zero-shot classification tasks. Notably, ABS is training-free and even rivals few-shot and test-time adaptation methods.

</details>

---

## 96. $\mathcal{V}ista\mathcal{DPO}$: Video Hierarchical Spatial-Temporal Direct Preference Optimization for Large Video Models

- [ ] $\mathcal{V}ista\mathcal{DPO}$: Video Hierarchical Spatial-Temporal Direct Preference Optimization for Large Video Models | https://icml.cc/virtual/2025/poster/45463

- **Link**: https://icml.cc/virtual/2025/poster/45463

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Video Models (LVMs) built upon Large Language Models (LLMs) have shown promise in video understanding but often suffer from misalignment with human intuition and video hallucination issues. To address these challenges, we introduce VistaDPO , a novel framework for Video Hierarchical Spatial-Temporal Direct Preference Optimization. VistaDPO enhances text-video preference alignment across three hierarchical levels: i) Instance Level , aligning overall video content with responses; ii) Temporal Level , aligning video temporal semantics with event descriptions; and iii) Perceptive Level , aligning spatial objects with language tokens. Given the lack of datasets for fine-grained video-language preference alignment, we construct VistaDPO-7k , a dataset of 7.2K QA pairs annotated with chosen and rejected responses, along with spatial-temporal grounding information such as timestamps, keyframes, and bounding boxes. Extensive experiments on benchmarks such as Video Hallucination, Video QA, and Captioning performance tasks demonstrate that VistaDPO significantly improves the performance of existing LVMs, effectively mitigating video-language misalignment and hallucination.

</details>

---

## 97. DS-VLM: Diffusion Supervision Vision Language Model

- [ ] DS-VLM: Diffusion Supervision Vision Language Model | https://icml.cc/virtual/2025/poster/45511

- **Link**: https://icml.cc/virtual/2025/poster/45511

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) face two critical limitations in visual representation learning: degraded supervision due to information loss during gradient propagation, and the inherent semantic sparsity of textual supervision compared to visual data. We propose the Diffusion Supervision Vision-Language Model (DS-VLM), a plug-and-play framework that introduces diffusion-based direct supervision for vision-language alignment. By reconstructing input images through a diffusion model conditioned on outputs of the visual encoder and the connector, our method establishes a short-path gradient propagation channel from pixel space to visual features. This approach simultaneously preserves high-level semantic alignment through conventional text supervision while enhancing visual feature quality via pixel-level reconstruction constraints. Extensive experiments conducted across various visual encoders and LLMs of different scales demonstrate the effectiveness of our approach.

</details>

---

## 98. Reasoning Limitations of Multimodal Large Language Models. A case study of Bongard Problems

- [ ] Reasoning Limitations of Multimodal Large Language Models. A case study of Bongard Problems | https://icml.cc/virtual/2025/poster/45515

- **Link**: https://icml.cc/virtual/2025/poster/45515

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Abstract visual reasoning (AVR) involves discovering shared concepts across images through analogy, akin to solving IQ test problems. Bongard Problems (BPs) remain a key challenge in AVR, requiring both visual reasoning and verbal description. We investigate whether multimodal large language models (MLLMs) can solve BPs by formulating a set of diverse MLLM-suited solution strategies and testing $4$ proprietary and $4$ open-access models on $3$ BP datasets featuring synthetic (classic BPs) and real-world (Bongard HOI and Bongard-OpenWorld) images. Despite some successes on real-world datasets, MLLMs struggle with synthetic BPs. To explore this gap, we introduce Bongard-RWR, a dataset representing synthetic BP concepts using real-world images. Our findings suggest that weak MLLM performance on classical BPs is not due to the domain specificity, but rather comes from their general AVR limitations. Code and dataset are available at: https://github.com/pavonism/bongard-rwr

</details>

---

## 99. ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning

- [ ] ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning | https://icml.cc/virtual/2025/poster/45523

- **Link**: https://icml.cc/virtual/2025/poster/45523

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language-Action (VLA) models have shown great potential in general robotic decision-making tasks via imitation learning. However, the variable quality of training data often constrains the performance of these models. On the other hand, offline Reinforcement Learning (RL) excels at learning robust policy models from mixed-quality data. In this paper, we introduce Reinforced robot GPT (ReinboT), a novel end-to-end VLA model that integrates the RL principle of maximizing cumulative reward. ReinboT achieves a deeper understanding of the data quality distribution by predicting dense returns that capture the nuances of manipulation tasks. The dense return prediction capability enables the robot to generate more robust decision-making actions, oriented towards maximizing future benefits. Extensive experiments show that ReinboT achieves state-of-the-art performance on the CALVIN mixed-quality dataset and exhibits superior few-shot learning and out-of-distribution generalization capabilities in real-world tasks.

</details>

---

## 100. When and How Does CLIP Enable Domain and Compositional Generalization?

- [ ] When and How Does CLIP Enable Domain and Compositional Generalization? | https://icml.cc/virtual/2025/poster/45573

- **Link**: https://icml.cc/virtual/2025/poster/45573

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The remarkable generalization performance of contrastive vision-language models like CLIP is often attributed to the diversity of their training distributions. However,  key questions remain unanswered: Can CLIP generalize to an entirely unseen domain when trained on a diverse mixture of domains (domain generalization)? Can it generalize to unseen classes within partially seen domains (compositional generalization)? What factors affect such generalization? To answer these questions, we trained CLIP models on systematically constructed training distributions with controlled domain diversity and object class exposure. Our experiments show that domain diversity is essential for both domain and compositional generalization, yet compositional generalization can be surprisingly weaker than domain generalization when the training distribution contains a suboptimal subset of the test domain. Through data-centric and mechanistic analyses, we find that successful generalization requires the learning of sufficiently shared representations in intermediate layers and circuits.

</details>

---

## 101. Preference Adaptive and Sequential Text-to-Image Generation

- [ ] Preference Adaptive and Sequential Text-to-Image Generation | https://icml.cc/virtual/2025/poster/45601

- **Link**: https://icml.cc/virtual/2025/poster/45601

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We address the problem of interactive text-to-image (T2I) generation, designing a reinforcement learning (RL) agent which iteratively improves a set of generated images for a user through a sequence of prompt expansions. Using human raters, we create a novel dataset of sequential preferences, which we leverage, together with large-scale open-source (non-sequential) datasets. We construct user-preference and user-choice models using an EM strategy and identify varying user preference types. We then leverage a large multimodal language model (LMM) and a value-based RL approach to suggest an adaptive and diverse slate of prompt expansions to the user. Our Preference Adaptive and Sequential Text-to-image Agent (PASTA) extends T2I models with adaptive multi-turn capabilities, fostering collaborative co-creation and addressing uncertainty or underspecification in a user's intent. We evaluate PASTA using human raters, showing significant improvement compared to baseline methods. We also open-source our sequential rater dataset and simulated user-rater interactions to support future research in user-centric multi-turn T2I systems.

</details>

---

## 102. Test-Time Adaptation for Online Vision-Language Navigation with Feedback-based Reinforcement Learning

- [ ] Test-Time Adaptation for Online Vision-Language Navigation with Feedback-based Reinforcement Learning | https://icml.cc/virtual/2025/poster/45655

- **Link**: https://icml.cc/virtual/2025/poster/45655

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Navigating in an unfamiliar environment during deployment poses a critical challenge for a vision-language navigation (VLN) agent. Yet, test-time adaptation (TTA) remains relatively underexplored in robotic navigation, leading us to the fundamental question: what are the key properties of TTA for online VLN? In our view, effective adaptation requires three qualities: 1) flexibility in handling different navigation outcomes, 2) interactivity with external environment, and 3) maintaining a harmony between plasticity and stability. To address this, we introduce FeedTTA, a novel TTA framework for online VLN utilizing feedback-based reinforcement learning. Specifically, FeedTTA learns by maximizing binary episodic feedback, a practical setup in which the agent receives a binary scalar after each episode that indicates the success or failure of the navigation. Additionally, we propose a gradient regularization technique that leverages the binary structure of FeedTTA to achieve a balance between plasticity and stability during adaptation. Our extensive experiments on challenging VLN benchmarks demonstrate the superior adaptability of FeedTTA, even outperforming the state-of-the-art offline training methods in REVERIE benchmark with a single stream of learning.

</details>

---

## 103. MedRAX: Medical Reasoning Agent for Chest X-ray

- [ ] MedRAX: Medical Reasoning Agent for Chest X-ray | https://icml.cc/virtual/2025/poster/45678

- **Link**: https://icml.cc/virtual/2025/poster/45678

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chest X-rays (CXRs) play an integral role in driving critical decisions in disease management and patient care. While recent innovations have led to specialized models for various CXR interpretation tasks, these solutions often operate in isolation, limiting their practical utility in clinical practice. We present MedRAX, the first versatile AI agent that seamlessly integrates state-of-the-art  CXR analysis tools and multimodal large language models into a unified framework. MedRAX dynamically leverages these models to address complex medical queries without requiring additional training. To rigorously evaluate its capabilities, we introduce ChestAgentBench, a comprehensive benchmark containing 2,500 complex medical queries across 7 diverse categories. Our experiments demonstrate that MedRAX achieves state-of-the-art performance compared to both open-source and proprietary models, representing a significant step toward the practical deployment of automated CXR interpretation systems. Data and code have been publicly available at https://github.com/bowang-lab/MedRAX

</details>

---

## 104. Enhancing Target-unspecific Tasks through a Features Matrix

- [ ] Enhancing Target-unspecific Tasks through a Features Matrix | https://icml.cc/virtual/2025/poster/45705

- **Link**: https://icml.cc/virtual/2025/poster/45705

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent developments in prompt learning of large Vision-Language Models (VLMs) have significantly improved performance in target-specific tasks. However, these prompting methods often struggle to tackle the target-unspecific or generalizable tasks effectively. It may be attributed to the fact that overfitting training causes the model to forget its general knowledge. The general knowledge has a strong promotion on target-unspecific tasks. To alleviate this issue, we propose a novel Features Matrix (FM) approach designed to enhance these models on target-unspecific tasks. Our method extracts and leverages general knowledge, shaping a Features Matrix (FM). Specifically, the FM captures the semantics of diverse inputs from a deep and fine perspective, preserving essential general knowledge, which mitigates the risk of overfitting. Representative evaluations demonstrate that: 1) the FM is compatible with existing frameworks as a generic and flexible module, and 2)  the FM significantly showcases its effectiveness in enhancing target-unspecific tasks (base-to-novel generalization, domain generalization, and cross-dataset generalization), achieving state-of-the-art performance.

</details>

---

## 105. 3D Question Answering via only 2D Vision-Language Models

- [ ] 3D Question Answering via only 2D Vision-Language Models | https://icml.cc/virtual/2025/poster/45722

- **Link**: https://icml.cc/virtual/2025/poster/45722

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have significantly advanced numerous fields. In this work, we explore how to harness their potential to address 3D scene understanding tasks, using 3D question answering (3D-QA) as a representative example. Due to the limited training data in 3D, we do not train LVLMs but infer in a zero-shot manner. Specifically, we sample 2D views from a 3D point cloud and feed them into 2D models to answer a given question. When the 2D model is chosen, e.g., LLAVA-OV, the quality of sampled views matters the most. We propose cdViews, a novel approach to automatically selecting critical and diverse Views for 3D-QA. cdViews consists of two key components: viewSelector prioritizing critical views based on their potential to provide answer-specific information, and viewNMS enhancing diversity by removing redundant views based on spatial overlap. We evaluate cdViews on the widely-used ScanQA and SQA benchmarks, demonstrating that it achieves state-of-the-art performance in 3D-QA while relying solely on 2D models without fine-tuning. These findings support our belief that 2D LVLMs are currently the most effective alternative (of the resource-intensive 3D LVLMs) for addressing 3D tasks.

</details>

---

## 106. Catch Your Emotion: Sharpening Emotion Perception in Multimodal Large Language Models

- [ ] Catch Your Emotion: Sharpening Emotion Perception in Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/45730

- **Link**: https://icml.cc/virtual/2025/poster/45730

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have achieved impressive progress in tasks such as visual question answering and visual understanding, but they still face significant challenges in emotional reasoning. Current methods to enhance emotional understanding typically rely on fine-tuning or manual annotations, which are resource-intensive and limit scalability. In this work, we focus on improving the ability of MLLMs to capture emotions during the inference phase. Specifically, MLLMs encounter two main issues: they struggle to distinguish between semantically similar emotions, leading to misclassification, and they are overwhelmed by redundant or irrelevant visual information, which distracts from key emotional cues. To address these, we propose Sharpening Emotion Perception in MLLMs (SEPM), which incorporates a Confidence-Guided Coarse-to-Fine Inference framework to refine emotion classification by guiding the model through simpler tasks. Additionally, SEPM employs Focus-on-Emotion Visual Augmentation to reduce visual redundancy by directing the attention of models to relevant emotional cues in images. Experimental results demonstrate that SEPM significantly improves MLLM performance on emotion-related tasks, providing a resource-efficient and scalable solution for emotion recognition.

</details>

---

## 107. CoMemo: LVLMs Need Image Context with Image Memory

- [ ] CoMemo: LVLMs Need Image Context with Image Memory | https://icml.cc/virtual/2025/poster/45756

- **Link**: https://icml.cc/virtual/2025/poster/45756

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision-Language Models built upon Large Language Models have established aligning visual features with LLM representations as the dominant paradigm. However, inherited LLM architectural designs introduce suboptimal characteristics for multimodal processing. First, LVLMs exhibit a bimodal distribution in attention allocation, leading to the progressive neglect of middle visual content as context expands. Second, conventional positional encoding schemes fail to preserve vital 2D structural relationships when processing dynamic high-resolution images. To address these limitations, we propose CoMemo - a dual-path architecture that combines a Co ntext image path with an image Memo ry path for visual processing, effectively alleviating visual information neglect. Additionally, we introduce RoPE-DHR, a novel positional encoding mechanism that employs thumbnail-based positional aggregation to maintain 2D spatial awareness while mitigating remote decay in extended sequences. Evaluations across seven benchmarks,including long-context comprehension, multi-image reasoning, and visual question answering, demonstrate CoMemo's superior performance compared to conventional LVLM architectures.Project page is available at https://lalbj.github.io/projects/CoMemo/ .

</details>

---

## 108. What If We Recaption Billions of Web Images with LLaMA-3?

- [ ] What If We Recaption Billions of Web Images with LLaMA-3? | https://icml.cc/virtual/2025/poster/45764

- **Link**: https://icml.cc/virtual/2025/poster/45764

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Web-crawled image-text pairs are inherently noisy. Prior studies demonstrate that semantically aligning and enriching textual descriptions of these pairs can significantly enhance model training across various vision-language tasks, particularly text-to-image generation. However, large-scale investigations in this area remain predominantly closed-source. Our paper aims to bridge this community effort, leveraging the powerful and $\textit{open-sourced}$ LLaMA-3, a GPT-4 level LLM. Our recaptioning pipeline is simple: first, we fine-tune a LLaMA-3-8B powered LLaVA-1.5 and then employ it to recaption ~1.3 billion images from the DataComp-1B dataset. Our empirical results confirm that this enhanced dataset, Recap-DataComp-1B, offers substantial benefits in training advanced vision-language models. For discriminative models like CLIP, we observe an average of 3.1% enhanced zero-shot performance cross four cross-modal retrieval tasks using a mixed set of the original and our captions. For generative models like text-to-image Diffusion Transformers, the generated images exhibit a significant improvement in alignment with users' text instructions, especially in following complex queries. Our project page is https://www.haqtu.me/Recap-Datacomp-1B/.

</details>

---

## 109. WMarkGPT: Watermarked Image Understanding via Multimodal Large Language Models

- [ ] WMarkGPT: Watermarked Image Understanding via Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/45767

- **Link**: https://icml.cc/virtual/2025/poster/45767

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Invisible watermarking is widely used to protect digital images from unauthorized use. Accurate assessment of watermarking efficacy is crucial for advancing algorithmic development. However, existing statistical metrics, such as PSNR, rely on access to original images, which are often unavailable in text-driven generative watermarking and fail to capture critical aspects of watermarking, particularly visibility. More importantly, these metrics fail to account for potential corruption of image content. To address these limitations, we propose WMarkGPT, the first multimodal large language model (MLLM) specifically designed for comprehensive watermarked image understanding, without accessing original images. WMarkGPT not only predicts watermark visibility but also generates detailed textual descriptions of its location, content, and impact on image semantics, enabling a more nuanced interpretation of watermarked images. Tackling the challenge of precise location description and understanding images with vastly different content, we construct three visual question-answering (VQA) datasets: an object location-aware dataset, a synthetic watermarking dataset, and a real watermarking dataset. We introduce a meticulously designed three-stage learning pipeline to progressively equip WMarkGPT with the necessary abilities. Extensive experiments on synthetic and real watermarking QA datasets demonstrate that WMarkGPT outperforms existing MLLMs, achieving significant improvements in visibility prediction and content description. The datasets and code are released at https://github.com/TanSongBai/WMarkGPT.

</details>

---

## 110. LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models

- [ ] LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models | https://icml.cc/virtual/2025/poster/45771

- **Link**: https://icml.cc/virtual/2025/poster/45771

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) robustness is a desired property of computer vision models. Improving model robustness requires high-quality signals from robustness benchmarks to quantify progress. While various benchmark datasets such as ImageNet-C were proposed in the ImageNet era, most ImageNet-C corruption types are no longer OOD relative to today's large, web-scraped datasets, which already contain common corruptions such as blur or JPEG compression artifacts. Consequently, these benchmarks are no longer well-suited for evaluating OOD robustness in the era of web-scale datasets. Indeed, recent models show saturating scores on ImageNet-era OOD benchmarks, indicating that it is unclear whether models trained on web-scale datasets truly become better at OOD generalization or whether they have simply been exposed to the test distortions during training. To address this, we introduce LAION-C as a benchmark alternative for ImageNet-C. LAION-C consists of six novel distortion types specifically designed to be OOD, even for web-scale datasets such as LAION. In a comprehensive evaluation of state-of-the-art models, we find that the LAION-C dataset poses significant challenges to contemporary models, including MLLMs such as Gemini and GPT-4o. We additionally conducted a psychophysical experiment to evaluate the difficulty of our corruptions for human observers, enabling a comparison of models to lab-quality human robustness data. We observe a paradigm shift in OOD generalization: from humans outperforming models, to the best models now matching or outperforming the best human observers.

</details>

---

## 111. CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models

- [ ] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models | https://icml.cc/virtual/2025/poster/45781

- **Link**: https://icml.cc/virtual/2025/poster/45781

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) excel across diverse tasks but suffer from high inference costs in time and memory. Token sparsity mitigates inefficiencies in token usage, while neuron sparsity reduces high-dimensional computations, both offering promising solutions to enhance efficiency.Recently, these two sparsity paradigms have evolved largely in parallel, fostering the prevailing assumption that they function independently. However, a fundamental yet underexplored question remains: Do they truly operate in isolation, or is there a deeper underlying interplay that has yet to be uncovered?In this paper, we conduct the first comprehensive investigation into this question. By introducing and analyzing the matching mechanism between Core Neurons and Core Tokens, we found that key neurons and tokens for inference mutually influence and reinforce each other. Building on this insight, we propose CoreMatching, a co-adaptive sparse inference framework, which leverages the synergy between token and neuron sparsity to enhance inference efficiency. Through theoretical analysis and efficiency evaluations, we demonstrate that the proposed method surpasses state-of-the-art baselines on ten image understanding tasks and three hardware devices. Notably, on the NVIDIA Titan Xp, it achieved 5$\times$ FLOPs reduction and a 10$\times$ overall speedup.Code is released at [https://github.com/wangqinsi1/2025-ICML-CoreMatching/tree/main](https://github.com/wangqinsi1/2025-ICML-CoreMatching/tree/main).

</details>

---

## 112. Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning

- [ ] Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning | https://icml.cc/virtual/2025/poster/45797

- **Link**: https://icml.cc/virtual/2025/poster/45797

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at https://github.com/langfengQ/CoSo.

</details>

---

## 113. EasyRef: Omni-Generalized Group Image Reference for Diffusion Models via Multimodal LLM

- [ ] EasyRef: Omni-Generalized Group Image Reference for Diffusion Models via Multimodal LLM | https://icml.cc/virtual/2025/poster/45830

- **Link**: https://icml.cc/virtual/2025/poster/45830

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Significant achievements in personalization of diffusion models have been witnessed. Conventional tuning-free methods mostly encode multiple reference images by averaging or concatenating their image embeddings as the injection condition, but such an image-independent operation cannot perform interaction among images to capture consistent visual elements within multiple references. Although tuning-based approaches can effectively extract consistent elements within multiple images through the training process, it necessitates test-time finetuning for each distinct image group. This paper introduces EasyRef, a plug-and-play adaption method that empowers diffusion models to condition consistent visual elements (e.g., style and human facial identity, etc.) across multiple reference images under instruction controls. To effectively exploit consistent visual elements within multiple images, we leverage the multi-image comprehension and instruction-following capabilities of the multimodal large language model (MLLM), prompting it to capture consistent visual elements based on the instruction. Besides, injecting the MLLM's representations into the diffusion process through adapters can easily generalize to unseen domains. To mitigate computational costs and enhance fine-grained detail preservation, we introduce an efficient reference aggregation strategy and a progressive training scheme. Finally, we introduce MRBench, a new multi-reference image generation benchmark. Experimental results demonstrate EasyRef surpasses both tuning-free and tuning-based methods, achieving superior aesthetic quality and robust zero-shot generalization across diverse domains.

</details>

---

## 114. Learning Invariant Causal Mechanism from Vision-Language Models

- [ ] Learning Invariant Causal Mechanism from Vision-Language Models | https://icml.cc/virtual/2025/poster/45848

- **Link**: https://icml.cc/virtual/2025/poster/45848

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pretraining (CLIP) has achieved remarkable success, but its performance can degrade when fine-tuned in out-of-distribution (OOD) scenarios. We model the prediction process using a Structural Causal Model (SCM) and show that the causal mechanism involving both invariant and variant factors in training environments differs from that in test environments. In contrast, the causal mechanism with solely invariant factors remains consistent across environments. We theoretically prove the existence of a linear mapping from CLIP embeddings to invariant factors, which can be estimated using interventional data. Additionally, we provide a condition to guarantee low OOD risk of the invariant predictor. Based on these insights, we propose the Invariant Causal Mechanism of CLIP (CLIP-ICM) framework. CLIP-ICM involves collecting interventional data, estimating a linear projection matrix, and making predictions within the invariant subspace. Experiments on several OOD datasets show that CLIP-ICM significantly improves the performance of CLIP. Our method offers a simple but powerful enhancement, boosting the reliability of CLIP in real-world applications.

</details>

---

## 115. Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration

- [ ] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration | https://icml.cc/virtual/2025/poster/45854

- **Link**: https://icml.cc/virtual/2025/poster/45854

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have manifested strong visual question answering capability. However, they still struggle with aligning the rationale and the generated answer, leading to inconsistent reasoning and incorrect responses. To this end, this paper introduces Self-Rationale Calibration (SRC) framework to iteratively calibrate the alignment between rationales and answers. SRC begins by employing a lightweight “rationale fine-tuning” approach, which modifies the model’s response format to require a rationale before deriving answer without explicit prompts. Next, SRC searches a diverse set of candidate responses from the fine-tuned LVLMs for each sample, followed by a proposed pairwise scoring strategy using a tailored scoring model, R-Scorer, to evaluate both rationale quality and factual consistency of candidates. Based on a confidence-weighted preference curation process, SRC decouples the alignment calibration into a preference fine-tuning manner, leading to significant improvements of LVLMs in perception, reasoning, and generalization across multiple benchmarks. Our results emphasize the rationale-oriented alignment in exploring the potential of LVLMs.

</details>

---

## 116. Unifying 2D and 3D Vision-Language Understanding

- [ ] Unifying 2D and 3D Vision-Language Understanding | https://icml.cc/virtual/2025/poster/45879

- **Link**: https://icml.cc/virtual/2025/poster/45879

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Progress in 3D vision-language learning has been hindered by the scarcity of large-scale 3D datasets. We introduce UniVLG, a unified architecture for 2D and 3D vision-language understanding that bridges the gap between existing 2D-centric models and the rich 3D sensory data available in embodied systems. Our approach initializes most model weights from pre-trained 2D models and trains on both 2D and 3D vision-language data. We propose a novel language-conditioned  mask decoder shared across 2D and 3D modalities to ground objects effectively in both RGB and RGB-D images, outperforming box-based approaches. To further reduce the domain gap between 2D and 3D, we incorporate 2D-to-3D lifting strategies, enabling UniVLG to utilize 2D data to enhance 3D performance. With these innovations, our model achieves state-of-the-art performance across multiple 3D vision-language grounding tasks, demonstrating the potential of transferring advances from 2D vision-language learning to the data-constrained 3D domain. Furthermore, co-training on both 2D and 3D data enhances performance across modalities without sacrificing 2D capabilities. By removing the reliance on 3D mesh reconstruction and ground-truth object proposals, UniVLG sets a new standard for realistic, embodied-aligned evaluation. Code and additional visualizations are available at https://univlg.github.io.

</details>

---

## 117. Learn from Downstream and Be Yourself in Multimodal Large Language Models Fine-Tuning

- [ ] Learn from Downstream and Be Yourself in Multimodal Large Language Models Fine-Tuning | https://icml.cc/virtual/2025/poster/45894

- **Link**: https://icml.cc/virtual/2025/poster/45894

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Model (MLLM) has demonstrated strong generalization capabilities across diverse distributions and tasks, largely due to extensive pre-training datasets. Fine-tuning MLLM has become a common practice to improve performance on specific downstream tasks. However, during fine-tuning, MLLM often faces the risk of forgetting knowledge acquired during pre-training, which can result in a decline in generalization abilities. To balance the trade-off between generalization and specialization, we propose measuring the parameter importance for both pre-trained and fine-tuning distributions, based on frozen pre-trained weight magnitude and accumulated fine-tuning gradient values. We further apply an importance-aware weight allocation strategy, selectively updating relatively important parameters for downstream tasks. We conduct empirical evaluations on both image captioning and visual question-answering tasks using various MLLM architectures. The comprehensive experimental analysis demonstrates the effectiveness of the proposed solution, highlighting the efficiency of the crucial modules in enhancing downstream specialization performance while mitigating generalization degradation in MLLM Fine-Tuning.

</details>

---

## 118. Graph4MM: Weaving Multimodal Learning with Structural Information

- [ ] Graph4MM: Weaving Multimodal Learning with Structural Information | https://icml.cc/virtual/2025/poster/45904

- **Link**: https://icml.cc/virtual/2025/poster/45904

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Real-world multimodal data usually exhibit complex structural relationships beyond traditional one-to-one mappings like image-caption pairs. Entities across modalities interact in intricate ways, with images and text forming diverse interconnections through contextual dependencies and co-references. Graphs provide powerful structural information for modeling intra-modal and inter-modal relationships. However, previous works fail to distinguish multi-hop neighbors and treat the graph as a standalone modality, which fragments the overall understanding. This limitation presents two key challenges in multimodal learning: (1) integrating structural information from multi-hop neighbors into foundational models, and (2) fusing modality-specific information in a principled manner. To address these challenges, we revisit the role of graphs in multimodal learning within the era of foundation models and propose Graph4MM, a graph-based multimodal learning framework. To be specific, we introduce Hop-Diffused Attention, which integrates multi-hop structural information into self-attention through causal masking and hop diffusion. Furthermore, we design MM-QFormer, a multi-mapping querying transformer for cross-modal fusion. Through theoretical and empirical analysis, we show that leveraging structures to integrate both intra- and inter-modal interactions improves multimodal understanding beyond treating them as a standalone modality. Experiments on both generative and discriminative tasks show that Graph4MM outperforms larger VLMs, LLMs, and multimodal graph baselines, achieving a 6.93% average improvement.

</details>

---

## 119. Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models

- [ ] Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models | https://icml.cc/virtual/2025/poster/45910

- **Link**: https://icml.cc/virtual/2025/poster/45910

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have improved significantly in their capabilities, but their complex architecture makes their safety alignment challenging. In this paper, we reveal an uneven distribution of harmful information across the intermediate layers of the image encoder and show that skipping a certain set of layers and exiting early can increase the chance of the VLM generating harmful responses. We call it as “Image enCoder Early-exiT” based vulnerability (ICET). Our experiments across three VLMs: LLaVA-1.5, LLaVA-NeXT, and Llama 3.2 show that performing early exits from the image encoder significantly increases the likelihood of generating harmful outputs. To tackle this, we propose a simple yet effective modification of the Clipped-Proximal Policy Optimization (Clip-PPO) algorithm for performing layer-wise multi-modal RLHF for VLMs. We term this as Layer-Wise PPO (L-PPO). We evaluate our L-PPO algorithm across three multi-modal datasets and show that it consistently reduces the harmfulness caused by early exits.

</details>

---

## 120. SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs

- [ ] SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs | https://icml.cc/virtual/2025/poster/45942

- **Link**: https://icml.cc/virtual/2025/poster/45942

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal retrieval-augmented generation (RAG) plays a crucial role in domains such as knowledge-based visual question answering (KB-VQA), where models should effectively integrate additional knowledge to generate a response. However, existing vision and language models (VLMs) are not inherently designed for context-augmented generation, limiting their effectiveness in such tasks. While synthetic data generation has recently gained attention for training large VLMs, its application for context-augmented generation remains underexplored. To address this gap, we introduce SKVQA, a large-scale synthetic multimodal dataset containing over 2 million visual question-answer pairs, each associated with external knowledge sources to determine the final answer. Compared to previous datasets, SKVQA exhibits 11× more unique questions, greater domain diversity, and a broader spectrum of image sources. Through human evaluations, we confirm the high quality of the generated question-answer pairs and their contextual relevance. Extensive experiments show that SKVQA serves both as a challenging benchmark for knowledge-based VQA and as an effective training resource for adapting generative multimodal models to context-augmented generation. Our results further indicate that models trained on SKVQA demonstrate enhanced generalization in both context-aware VQA and multimodal RAG settings.

</details>

---

## 121. Core Knowledge Deficits in Multi-Modal Language Models

- [ ] Core Knowledge Deficits in Multi-Modal Language Models | https://icml.cc/virtual/2025/poster/45955

- **Link**: https://icml.cc/virtual/2025/poster/45955

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Multi-modal Large Language Models (MLLMs) demonstrate impressive abilities over high-level perception and reasoning, their robustness in the wild remains limited, often falling short on tasks that are intuitive and effortless for humans. We examine the hypothesis that these deficiencies stem from the absence of core knowledge—rudimentary cognitive abilities innate to humans from early childhood. To explore the core knowledge representation in MLLMs, we introduce CoreCognition, a large-scale benchmark encompassing 12 core knowledge concepts grounded in developmental cognitive science.We evaluate 230 models with 11 different prompts, leading to a total of 2,530 data points for analysis. Our experiments uncover four key findings, collectively demonstrating core knowledge deficits in MLLMs: they consistently underperform and show reduced, or even absent, scalability on low-level abilities relative to high-level ones.Finally, we propose Concept Hacking, a novel controlled evaluation method that reveals MLLMs fail to progress toward genuine core knowledge understanding, but instead rely on shortcut learning as they scale.

</details>

---

## 122. EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents

- [ ] EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents | https://icml.cc/virtual/2025/poster/45994

- **Link**: https://icml.cc/virtual/2025/poster/45994

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Leveraging Multi-modal Large Language Models (MLLMs) to create embodied agents offers a promising avenue for tackling real-world tasks. While language-centric embodied agents have garnered substantial attention, MLLM-based embodied agents remain underexplored due to the lack of comprehensive evaluation frameworks. To bridge this gap, we introduce EmbodiedBench, an extensive benchmark designed to evaluate vision-driven embodied agents.EmbodiedBench features: (1) a diverse set of 1,128 testing tasks across four environments, ranging from high-level semantic tasks (e.g., household) to low-level tasks involving atomic actions (e.g., navigation and manipulation); and (2) six meticulously curated subsets evaluating essential agent capabilities like commonsense reasoning, complex instruction understanding, spatial awareness, visual perception, and long-term planning.Through extensive experiments, we evaluated 24 leading proprietary and open-source MLLMs within EmbodiedBench. Our findings reveal that: MLLMs excel at high-level tasks but struggle with low-level manipulation, with the best model, GPT-4o, scoring only $28.9\\%$ on average. EmbodiedBench provides a multifaceted standardized evaluation platform that not only highlights existing challenges but also offers valuable insights to advance MLLM-based embodied agents. Our code and dataset are available at [https://embodiedbench.github.io](https://embodiedbench.github.io).

</details>

---

## 123. Diffusion Instruction Tuning

- [ ] Diffusion Instruction Tuning | https://icml.cc/virtual/2025/poster/46009

- **Link**: https://icml.cc/virtual/2025/poster/46009

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Lavender , a simple supervised fine-tuning (SFT) method that boosts the performance of advanced vision-language models (VLMs) by leveraging state-of-the-art image generation models such as Stable Diffusion. Specifically, Lavender aligns the text-vision attention in the VLM transformer with the equivalent used by Stable Diffusion during SFT, instead of adapting separate encoders. This alignment enriches the model’s visual understanding and significantly boosts performance across in- and out-of-distribution tasks. Lavender requires just 0.13 million training examples---2.5\% of typical large-scale SFT datasets---and fine-tunes on standard hardware (8 GPUs) in a single day. It consistently improves state-of-the-art open-source multimodal LLMs (e.g., Llama-3.2-11B, MiniCPM-Llama3-v2.5), achieving up to 30\% gains and a 68\% boost on challenging out-of-distribution medical QA tasks. By efficiently transferring the visual expertise of image generators with minimal supervision, Lavender offers a scalable solution for more accurate vision-language systems. Code, training data, and models are available on the project page .

</details>

---

## 124. An Empirical Study on Configuring In-Context Learning Demonstrations for Unleashing MLLMs' Sentimental Perception Capability

- [ ] An Empirical Study on Configuring In-Context Learning Demonstrations for Unleashing MLLMs' Sentimental Perception Capability | https://icml.cc/virtual/2025/poster/46011

- **Link**: https://icml.cc/virtual/2025/poster/46011

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancements in Multimodal Large Language Models (MLLMs) have enabled various multimodal tasks to be addressed under a zero-shot paradigm. This paradigm sidesteps the cost of model fine-tuning, emerging as a dominant trend in practical application. Nevertheless, Multimodal Sentiment Analysis (MSA), a pivotal challenge in the quest for general artificial intelligence, fails to accommodate this convenience. The zero-shot paradigm exhibits undesirable performance on MSA, casting doubt on whether MLLMs can perceive sentiments as competent as supervised models. By extending the zero-shot paradigm to In-Context Learning (ICL) and conducting an in-depth study on configuring demonstrations, we validate that MLLMs indeed possess such capability. Specifically, three key factors that cover demonstrations' retrieval, presentation, and distribution are comprehensively investigated and optimized. A sentimental predictive bias inherent in MLLMs is also discovered and later effectively counteracted. By complementing each other, the devised strategies for three factors result in average accuracy improvements of 15.9% on six MSA datasets against the zero-shot paradigm and 11.2% against the random ICL baseline.

</details>

---

## 125. Hypo3D: Exploring Hypothetical Reasoning in 3D

- [ ] Hypo3D: Exploring Hypothetical Reasoning in 3D | https://icml.cc/virtual/2025/poster/46012

- **Link**: https://icml.cc/virtual/2025/poster/46012

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rise of vision-language foundation models marks an advancement in bridging the gap between human and machine capabilities in 3D scene reasoning. Existing 3D reasoning benchmarks assume real-time scene accessibility, which is impractical due to the high cost of frequent scene updates. To this end, we introduce Hypothetical 3D Reasoning , namely Hypo3D, a benchmark designed to evaluate models' ability to reason without access to real-time scene data. Models need to imagine the scene state based on a provided change description before reasoning. Hypo3D is formulated as a 3D Visual Question Answering (VQA) benchmark, comprising 7,727 context changes across 700 indoor scenes, resulting in 14,885 question-answer pairs. An anchor-based world frame is established for all scenes, ensuring consistent reference to a global frame for directional terms in context changes and QAs. Extensive experiments show that state-of-the-art foundation models struggle to reason effectively in hypothetically changed scenes. This reveals a substantial performance gap compared to humans, particularly in scenarios involving movement changes and directional reasoning. Even when the change is irrelevant to the question, models often incorrectly adjust their answers. The code and dataset are publicly available at: https://matchlab-imperial.github.io/Hypo3D.

</details>

---

## 126. Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger

- [ ] Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger | https://icml.cc/virtual/2025/poster/46017

- **Link**: https://icml.cc/virtual/2025/poster/46017

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision Language Models (LVLMs) have significantly improved performance in Visual Question Answering (VQA) tasks through multimodal Retrieval-Augmented Generation (RAG). However, existing methods still face challenges, such as the scarcity of knowledge with reasoning examples and erratic responses from retrieved knowledge. To address these issues, in this study, we propose a multimodal RAG framework, termed RCTS, which enhances LVLMs by constructing a Reasoning Context-enriched knowledge base and a Tree Search re-ranking method. Specifically, we introduce a self-consistent evaluation mechanism to enrich the knowledge base with intrinsic reasoning patterns.  We further propose a Monte Carlo Tree Search with Heuristic Rewards (MCTS-HR) to prioritize the most relevant examples.  This ensures that LVLMs can leverage high-quality contextual reasoning for better and more consistent responses. Extensive experiments demonstrate that our framework achieves state-of-the-art performance on multiple VQA datasets, significantly outperforming In-Context Learning (ICL) and Vanilla-RAG methods. It highlights the effectiveness of our knowledge base and re-ranking method in improving LVLMs.

</details>

---

## 127. SAN: Hypothesizing Long-Term Synaptic Development and Neural Engram Mechanism in Scalable Model's Parameter-Efficient Fine-Tuning

- [ ] SAN: Hypothesizing Long-Term Synaptic Development and Neural Engram Mechanism in Scalable Model's Parameter-Efficient Fine-Tuning | https://icml.cc/virtual/2025/poster/46049

- **Link**: https://icml.cc/virtual/2025/poster/46049

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Advances in Parameter-efficient Fine-tuning (PEFT) bridged the performance gap with Full Fine-Tuning (FFT) through sophisticated analysis of pre-trained parameter spaces. Starting from drawing insights from Neural Engrams (NE) in Biological Neural Networks (BNNs), we establish a connection between the low-rank property observed during PEFT's parameter space shifting and neurobiological mechanisms. This observation leads to our proposed method, S ynapse and N euron ( SAN ), which decomposes and propagates the scaling component from anterior feature adjustment vectors towards posterior weight matrices. Our approach is theoretically grounded in Long-Term Potentiation/Depression (LTP/D) phenomena, which govern synapse development through neurotransmitter release modulation. Extensive experiments demonstrate its effectiveness: on vision tasks across VTAB, FGVC, and GIC (25 datasets) using ViT, Swin-T and ConvNeXt architectures, SAN outperforms FFT up to 8.7% and LoRA by 3.2% ; on language tasks using Commonsense Reasoning (8 datasets) with LLaMA models (all generations), surpassing ChatGPT up to 8.5% and LoRA by 4.7% ; on vision-language tasks using Visual Instruction Tuning (7 datasets) with LLaVA models, it exceeds FFT up to 2.4% and LoRA by 1.9% . Our code and W&B log will be released

</details>

---

## 128. Learning from True-False Labels via Multi-modal Prompt Retrieving

- [ ] Learning from True-False Labels via Multi-modal Prompt Retrieving | https://icml.cc/virtual/2025/poster/46070

- **Link**: https://icml.cc/virtual/2025/poster/46070

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained V ision- L anguage M odels (VLMs) exhibit strong zero-shot classification abilities, demonstrating great potential for generating weakly supervised labels. Unfortunately, existing weakly supervised learning methods are short of ability in generating accurate labels via VLMs. In this paper, we propose a novel weakly supervised labeling setting, namely T rue- F alse L abels (TFLs) which can achieve high accuracy when generated by VLMs. The TFL indicates whether an instance belongs to the label, which is randomly and uniformly sampled from the candidate label set. Specifically, we theoretically derive a risk-consistent estimator to explore and utilize the conditional probability distribution information of TFLs. Besides, we propose a convolutional-based M ulti-modal P rompt R etrieving (MRP) method to bridge the gap between the knowledge of VLMs and target learning tasks. Experimental results demonstrate the effectiveness of the proposed TFL setting and MRP learning method. The code to reproduce the experiments is at https://github.com/Tranquilxu/TMP.

</details>

---

## 129. From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models

- [ ] From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/46073

- **Link**: https://icml.cc/virtual/2025/poster/46073

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As large language models evolve, there is growing anticipation that they will emulate human-like Theory of Mind (ToM) to assist with routine tasks. However, existing methods for evaluating machine ToM focus primarily on unimodal models and largely treat these models as black boxes, lacking an interpretative exploration of their internal mechanisms. In response, this study adopts an approach based on internal mechanisms to provide an interpretability-driven assessment of ToM in multimodal large language models (MLLMs). Specifically, we first construct a multimodal ToM test dataset, GridToM, which incorporates diverse belief testing tasks and perceptual information from multiple perspectives. Next, our analysis shows that attention heads in multimodal large models can distinguish cognitive information across perspectives, providing evidence of ToM capabilities. Furthermore, we present a lightweight, training-free approach that significantly enhances the model’s exhibited ToM by adjusting in the direction of the attention head.

</details>

---

## 130. Defending LVLMs Against Vision Attacks Through Partial-Perception Supervision

- [ ] Defending LVLMs Against Vision Attacks Through Partial-Perception Supervision | https://icml.cc/virtual/2025/poster/46083

- **Link**: https://icml.cc/virtual/2025/poster/46083

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have raised significant concerns regarding the vulnerability of Large Vision Language Models (LVLMs) to maliciously injected or perturbed input images, which can mislead their responses. Existing defense methods show that such vision attacks are sensitive to image modifications especially cropping, using majority voting across responses of modified images as corrected responses. However, these modifications often result in partial images and distort the semantics, which reduces response quality on clean images after voting. Instead of directly using responses from partial images for voting, we investigate using them to supervise (guide) the LVLM's responses to the original images at inference time. We propose a black-box, training-free method called DPS (Defense through Partial-Perception Supervision) . In this approach, the model is prompted using the responses generated by a model that perceives only a partial image.With DPS, the model can adjust its response based on partial image understanding when under attack, while confidently maintaining its original response for clean input. Empirical experiments show our method outperforms the baseline, cutting the average attack success rate by 76.3\% across six datasets on three popular models.

</details>

---

## 131. OpenworldAUC: Towards Unified Evaluation and Optimization for Open-world Prompt Tuning

- [ ] OpenworldAUC: Towards Unified Evaluation and Optimization for Open-world Prompt Tuning | https://icml.cc/virtual/2025/poster/46099

- **Link**: https://icml.cc/virtual/2025/poster/46099

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning adapts Vision-Language Models like CLIP to open-world tasks with minimal training costs. In this direction, one typical paradigm evaluates model performance **separately** on known classes (*i.e.*, base domain) and unseen classes (*i.e.*, new domain). However, real-world scenarios require models to handle inputs **without prior domain knowledge**. This practical challenge has spurred the development of **open-world prompt tuning**, which demands a unified evaluation of two stages: 1) detecting whether an input belongs to the base or new domain (**P1**), and 2) classifying the sample into its correct class (**P2**). What's more, as domain distributions are generally unknown, a proper metric should be insensitive to varying base/new sample ratios (**P3**). However, we find that current metrics, including HM, overall accuracy, and AUROC, fail to satisfy these three properties simultaneously. To bridge this gap, we propose $\mathsf{OpenworldAUC}$, a unified metric that jointly assesses detection and classification through pairwise instance comparisons. To optimize $\mathsf{OpenworldAUC}$ effectively, we introduce **Gated Mixture-of-Prompts (GMoP)**, which employs domain-specific prompts and a gating mechanism to dynamically balance detection and classification. Theoretical guarantees ensure generalization of GMoP under practical conditions. Experiments on 15 benchmarks in open-world scenarios show GMoP achieves SOTA performance on $\mathsf{OpenworldAUC}$  and other metrics.

</details>

---

## 132. RBench: Graduate-level Multi-disciplinary Benchmarks for LLM & MLLM Complex Reasoning Evaluation

- [ ] RBench: Graduate-level Multi-disciplinary Benchmarks for LLM & MLLM Complex Reasoning Evaluation | https://icml.cc/virtual/2025/poster/46102

- **Link**: https://icml.cc/virtual/2025/poster/46102

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reasoning stands as a cornerstone of intelligence, enabling the synthesis of existing knowledge to solve complex problems. Despite remarkable progress, existing reasoning benchmarks often fail to rigorously evaluate the nuanced reasoning capabilities required for complex, real-world problemsolving, particularly in multi-disciplinary and multimodal contexts. In this paper, we introduce a graduate-level, multi-disciplinary, EnglishChinese benchmark, dubbed as Reasoning Bench (RBench), for assessing the reasoning capability of both language and multimodal models. RBench spans 1,094 questions across 108 subjects for language model evaluation and 665 questions across 83 subjects for multimodal model testing. These questions are meticulously curated to ensure rigorous difficulty calibration, subject balance, and cross-linguistic alignment, enabling the assessment to be an Olympiad-level multidisciplinary benchmark. We evaluate many models such as o1, GPT-4o, DeepSeek-R1, etc. Experimental results indicate that advanced models perform poorly on complex reasoning, especially multimodal reasoning. Even the top-performing model OpenAI o1 achieves only 53.2% accuracy on our multimodal evaluation. Data and code are made publicly available athttps://evalmodels.github.io/rbench/

</details>

---

## 133. Generative Data Mining with Longtail-Guided Diffusion

- [ ] Generative Data Mining with Longtail-Guided Diffusion | https://icml.cc/virtual/2025/poster/46120

- **Link**: https://icml.cc/virtual/2025/poster/46120

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

It is difficult to anticipate the myriad challenges that a predictive model will encounter once deployed. Common practice entails a reactive, cyclical approach: model deployment, data mining, and retraining. We instead develop a proactive longtail discovery process by imagining additional data during training. In particular, we develop general model-based longtail signals, including a differentiable, single forward pass formulation of epistemic uncertainty that does not impact model parameters or predictive performance but can flag rare or hard inputs. We leverage these signals as guidance to generate additional training data from a latent diffusion model in a process we call Longtail Guidance (LTG). Crucially, we can perform LTG without retraining the diffusion model or the predictive model, and we do not need to expose the predictive model to intermediate diffusion states. Data generated by LTG exhibit semantically meaningful variation, yield significant generalization improvements on numerous image classification benchmarks, and can be analyzed by a VLM to proactively discover, textually explain, and address conceptual gaps in a deployed predictive model.

</details>

---

## 134. Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models

- [ ] Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/46129

- **Link**: https://icml.cc/virtual/2025/poster/46129

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models have driven breakthroughs in visual question answering. Yet, a critical gap persists, `conceptualization'—the ability to recognize and reason about the same concept despite variations in visual form, a basic ability of human reasoning. To address this challenge, we introduce the Visual Graph Arena (VGA), a dataset featuring six graph-based tasks designed to evaluate and improve AI systems’ capacity for visual abstraction. VGA uses diverse graph layouts (e.g., Kamada-Kawai vs. planar) to test reasoning independent of visual form. Experiments with state-of-the-art vision models and multimodal LLMs reveal a striking divide: humans achieved near-perfect accuracy across tasks, while models totally failed on isomorphism detection and showed limited success in path/cycle tasks. We further identify behavioral anomalies suggesting pseudo-intelligent pattern matching rather than genuine understanding. These findings underscore fundamental limitations in current AI models for visual understanding. By isolating the challenge of representation-invariant reasoning, the VGA provides a framework to drive progress toward human-like conceptualization in AI visual models. The Visual Graph Arena is available at: \href{https://vga.csail.mit.edu/}{vga.csail.mit.edu}.

</details>

---

## 135. Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems

- [ ] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems | https://icml.cc/virtual/2025/poster/46140

- **Link**: https://icml.cc/virtual/2025/poster/46140

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Sortformer is an encoder-based speaker diarization model designed for supervising speaker tagging in speech-to-text models. Instead of relying solely on permutation invariant loss (PIL), Sortformer introduces Sort Loss to resolve the permutation problem, either independently or in tandem with PIL. In addition, we propose a streamlined multi-speaker speech-to-text architecture that leverages Sortformer for speaker supervision, embedding speaker labels into the encoder using sinusoidal kernel functions. This design addresses the speaker permutation problem through sorted objectives, effectively bridging timestamps and tokens to supervise speaker labels in the output transcriptions. Experiments demonstrate that Sort Loss can boost speaker diarization performance, and incorporating the speaker supervision from Sortformer improves multi-speaker transcription accuracy. We anticipate that the proposed Sortformer and multi-speaker architecture will enable the seamless integration of speaker tagging capabilities into foundational speech-to-text systems and multimodal large language models (LLMs), offering an easily adoptable and user-friendly mechanism to enhance their versatility and performance in speaker-aware tasks. The code and trained models are made publicly available through the NVIDIA NeMo Framework.

</details>

---

## 136. Can We Predict Performance of Large Models across Vision-Language Tasks?

- [ ] Can We Predict Performance of Large Models across Vision-Language Tasks? | https://icml.cc/virtual/2025/poster/46185

- **Link**: https://icml.cc/virtual/2025/poster/46185

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Evaluating large vision-language models (LVLMs) is very expensive, due to high computational cost and the wide variety of tasks. The good news is that if we already have some observed performance scores, we may be able to infer unknown ones. In this study, we propose a new framework for predicting unknown performance scores based on observed ones from other LVLMs or tasks. We first formulate the performance prediction as a matrix completion task. Specifically, we construct a sparse performance matrix $\boldsymbol{R}$, where each entry $R_{mn}$ represents the performance score of the $m$-th model on the $n$-th dataset. By applying probabilistic matrix factorization (PMF) with Markov chain Monte Carlo (MCMC), we can complete the performance matrix, i.e., predict unknown scores. Additionally, we estimate the uncertainty of performance prediction based on MCMC. Practitioners can evaluate their models on untested tasks with higher uncertainty first, which quickly reduces the prediction errors. We further introduce several improvements to enhance PMF for scenarios with sparse observed performance scores. Our experiments demonstrate the accuracy of PMF in predicting unknown scores, the reliability of uncertainty estimates in ordering evaluations, and the effectiveness of our enhancements for handling sparse data. Our code is available at https://github.com/Qinyu-Allen-Zhao/CrossPred-LVLM.

</details>

---

## 137. MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding

- [ ] MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding | https://icml.cc/virtual/2025/poster/46210

- **Link**: https://icml.cc/virtual/2025/poster/46210

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) recently showed strong capacity in integrating data among multiple modalities, empowered by generalizable attention architecture. Advanced methods predominantly focus on language-centric tuning while less exploring multimodal tokens mixed through attention, posing challenges in high-level tasks that require fine-grained cognition and emotion understanding. In this work, we identify the attention deficit disorder problem in multimodal learning, caused by inconsistent cross-modal attention and layer-by-layer decayed attention activation. To address this, we propose a novel attention mechanism, termed MOdular Duplex Attention (MODA), simultaneously conducting the inner-modal refinement and inter-modal interaction. MODA employs a correct-after-align strategy to effectively decouple modality alignment from cross-layer token mixing. In the alignment phase, tokens are mapped to duplex modality spaces based on the basis vectors, enabling the interaction between visual and language modality. Further, the correctness of attention scores is ensured through adaptive masked attention, which enhances the model's flexibility by allowing customizable masking patterns for different modalities. Extensive experiments on 21 benchmark datasets verify the effectiveness of MODA in perception, cognition, and emotion tasks.

</details>

---

## 138. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP

- [ ] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP | https://icml.cc/virtual/2025/poster/46255

- **Link**: https://icml.cc/virtual/2025/poster/46255

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce X-Transfer , a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as super transferability —a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through surrogate scaling , a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models.

</details>

---

## 139. SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

- [ ] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference | https://icml.cc/virtual/2025/poster/46297

- **Link**: https://icml.cc/virtual/2025/poster/46297

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In vision-language models (VLMs), visual tokens usually consume a significant amount of computational overhead, despite their sparser information density compared to text tokens. To address this, most existing methods learn a network to prune redundant visual tokens and require additional training data. Differently, we propose an efficient training-free token optimization mechanism dubbed SparseVLM without extra parameters or fine-tuning costs. Concretely, given that visual tokens complement text tokens in VLMs for linguistic reasoning, we select visual-relevant text tokens to rate the significance of vision tokens within the self-attention matrix extracted from the VLMs. Then we progressively prune irrelevant tokens. To maximize sparsity while retaining essential information, we introduce a rank-based strategy to adaptively determine the sparsification ratio for each layer, alongside a token recycling method that compresses pruned tokens into more compact representations. Experimental results show that our SparseVLM improves the efficiency of various VLMs across a range of image and video understanding tasks. In particular, when LLaVA is equipped with SparseVLM, it achieves a 54\% reduction in FLOPs, lowers CUDA time by 37\%, and maintains an accuracy rate of 97\%. Our code is available at https://github.com/Gumpest/SparseVLMs.

</details>

---

## 140. The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models Via Visual Information Steering

- [ ] The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models Via Visual Information Steering | https://icml.cc/virtual/2025/poster/46338

- **Link**: https://icml.cc/virtual/2025/poster/46338

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) can reason effectively over both textual and visual inputs, but they tend to hallucinate syntactically coherent yet visually ungrounded contents. In this paper, we investigate the internal dynamics of hallucination by examining the tokens logits rankings throughout the generation process, revealing three key patterns in how LVLMs process information: (1) gradual visual information loss -- visually grounded tokens gradually become less favored throughout generation, and (2) early excitation -- semantically meaningful tokens achieve peak activation in the layers earlier than the final layer.(3) hidden genuine information -- visually grounded tokens though not being eventually decided still retain relatively high rankings at inference.Based on these insights, we propose VISTA ( V isual I nformation S teering with T oken-logit A ugmentation), a training-free inference-time intervention framework that reduces hallucination while promoting genuine information. VISTA works by combining two complementary approaches: reinforcing visual information in activation space and leveraging early layer activations to promote semantically meaningful decoding. Compared to existing methods, VISTA requires no external supervision and is applicable to various decoding strategies. Extensive experiments show that VISTA on average reduces hallucination by about 40% on evaluated open-ended generation task, and it consistently outperforms existing methods on four benchmarks across four architectures under three decoding strategies. Code is available at https://github.com/LzVv123456/VISTA.

</details>

---

## 141. Vision-Language Models Create Cross-Modal Task Representations

- [ ] Vision-Language Models Create Cross-Modal Task Representations | https://icml.cc/virtual/2025/poster/46340

- **Link**: https://icml.cc/virtual/2025/poster/46340

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autoregressive vision-language models (VLMs) can handle many tasks within a single model, yet the representations that enable this capability remain opaque. We find that VLMs align conceptually equivalent inputs into a shared task vector, which is invariant to modality (text, image) and format (examples, instruction), and may simplify VLM processing. We measure this alignment via cross-modal transfer--the ability of a task vector derived in one modality to trigger the correct generation in another--on a range of tasks and model architectures. Although the task vector is highly compressed, we find that this single vector outperforms prompting the model with the full task information, unique to this cross-modal case. Furthermore, we show that task vectors can be transferred from a base language model to its fine-tuned vision-language counterpart, and that they can be derived solely from instructions without the need for examples. Taken together, our findings shed light on how VLMs internally process task information, and how they map different modalities into common semantic representations.

</details>

---

## 142. Imagine While Reasoning in Space: Multimodal Visualization-of-Thought

- [ ] Imagine While Reasoning in Space: Multimodal Visualization-of-Thought | https://icml.cc/virtual/2025/poster/46352

- **Link**: https://icml.cc/virtual/2025/poster/46352

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex reasoning in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Yet, it struggles in complex spatial reasoning tasks. Nonetheless, human cognition extends beyond language alone, enabling the remarkable capability to think in both words and images. Inspired by this mechanism, we propose a new reasoning paradigm, Multimodal Visualization-of-Thought (MVoT). It enables visual thinking in MLLMs by generating image visualizations of their reasoning traces. To ensure high-quality visualization, we introduce token discrepancy loss into autoregressive MLLMs. This innovation significantly improves both visual coherence and fidelity. We validate this approach through several dynamic spatial reasoning tasks. Experimental results reveal that MVoT demonstrates competitive performance across tasks. Moreover, it exhibits robust and reliable improvements in the most challenging scenarios where CoT fails. Ultimately, MVoT establishes new possibilities for complex reasoning tasks where visual thinking can effectively complement verbal reasoning.

</details>

---

## 143. PoisonedEye: Knowledge Poisoning Attack on Retrieval-Augmented Generation based Large Vision-Language Models

- [ ] PoisonedEye: Knowledge Poisoning Attack on Retrieval-Augmented Generation based Large Vision-Language Models | https://icml.cc/virtual/2025/poster/46373

- **Link**: https://icml.cc/virtual/2025/poster/46373

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Retrieval-Augmented Generation (VLRAG) systems have been widely applied to Large Vision-Language Models (LVLMs) to enhance their generation ability. However, the reliance on external multimodal knowledge databases renders VLRAG systems vulnerable to malicious poisoning attacks. In this paper, we introduce PoisonedEye, the first knowledge poisoning attack designed for VLRAG systems. Our attack successfully manipulates the response of the VLRAG system for the target query by injecting only one poison sample into the knowledge database. To construct the poison sample, we follow two key properties for the retrieval and generation process, and identify the solution by satisfying these properties. Besides, we also introduce a class query targeted poisoning attack, a more generalized strategy that extends the poisoning effect to an entire class of target queries. Extensive experiments on multiple query datasets, retrievers, and LVLMs demonstrate that our attack is highly effective in compromising VLRAG systems.

</details>

---

## 144. Efficient Multi-modal Long Context Learning for Training-free Adaptation

- [ ] Efficient Multi-modal Long Context Learning for Training-free Adaptation | https://icml.cc/virtual/2025/poster/46375

- **Link**: https://icml.cc/virtual/2025/poster/46375

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional approaches to adapting multi-modal large language models (MLLMs) to new tasks have relied heavily on fine-tuning. This paper introduces Efficient Multi-Modal Long Context Learning (EMLoC), a novel training-free alternative that embeds demonstration examples directly into the model input. EMLoC offers a more efficient, flexible, and scalable solution for task adaptation. Because extremely lengthy inputs introduce prohibitive computational and memory overhead, EMLoC contributes a chunk-wise compression mechanism combined with layer-wise adaptive pruning. It condenses long-context multimodal inputs into compact, task-specific memory representations. By adaptively pruning tokens at each layer under a Jensen-Shannon divergence constraint, our method achieves a dramatic reduction in inference complexity without sacrificing performance. This approach is the first to seamlessly integrate compression and pruning techniques for multi-modal long-context learning, offering a scalable and efficient solution for real-world applications. Extensive experiments on diverse vision-language benchmarks demonstrate that EMLoC achieves performance on par with or superior to naive long-context approaches. Our results highlight the potential of EMLoC as a groundbreaking framework for efficient and flexible adaptation of multi-modal models in resource-constrained environments. Codes are publicly available at https://github.com/Zehong-Ma/EMLoC.

</details>

---

## 145. Targeted Unlearning with Single Layer Unlearning Gradient

- [ ] Targeted Unlearning with Single Layer Unlearning Gradient | https://icml.cc/virtual/2025/poster/46379

- **Link**: https://icml.cc/virtual/2025/poster/46379

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Machine unlearning methods aim to remove sensitive or unwanted content from trained models, but typically demand extensive model updates at significant computational cost while potentially degrading model performance on both related and unrelated tasks. We propose Single Layer Unlearning Gradient (SLUG) as an efficient method to unlearn targeted information by updating a single critical layer using a one-time gradient computation. SLUG uses layer importance and gradient alignment metrics to identify the optimal layer for targeted information removal while preserving the model utility. We demonstrate the effectiveness of SLUG for CLIP, Stable Diffusion, and vision-language models (VLMs) in removing concrete (e.g., identities and objects) and abstract concepts (e.g., artistic styles). On the UnlearnCanvas benchmark, SLUG achieves comparable unlearning performance to existing methods while requiring significantly less computational resources. Our proposed approach offers a practical solution for targeted unlearning that is computationally efficient and precise. Our code is available at https://github.com/CSIPlab/SLUG

</details>

---

## 146. Interpreting CLIP with Hierarchical Sparse Autoencoders

- [ ] Interpreting CLIP with Hierarchical Sparse Autoencoders | https://icml.cc/virtual/2025/poster/46435

- **Link**: https://icml.cc/virtual/2025/poster/46435

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Sparse autoencoders (SAEs) are useful for detecting and steering interpretable features in neural networks, with particular potential for understanding complex multimodal representations. Given their ability to uncover interpretable features, SAEs are particularly valuable for analyzing vision-language models (e.g., CLIP and SigLIP), which are fundamental building blocks in modern large-scale systems yet remain challenging to interpret and control. However, current SAE methods are limited by optimizing both reconstruction quality and sparsity simultaneously, as they rely on either activation suppression or rigid sparsity constraints. To this end, we introduce Matryoshka SAE (MSAE), a new architecture that learns hierarchical representations at multiple granularities simultaneously, enabling a direct optimization of both metrics without compromise. MSAE establishes a state-of-the-art Pareto frontier between reconstruction quality and sparsity for CLIP, achieving 0.99 cosine similarity and less than 0.1 fraction of variance unexplained while maintaining 80\% sparsity. Finally, we demonstrate the utility of MSAE as a tool for interpreting and controlling CLIP by extracting over 120 semantic concepts from its representation to perform concept-based similarity search and bias analysis in downstream tasks like CelebA. We make the codebase available at https://github.com/WolodjaZ/MSAE.

</details>

---

## 147. Cowpox: Towards the Immunity of VLM-based Multi-Agent Systems

- [ ] Cowpox: Towards the Immunity of VLM-based Multi-Agent Systems | https://icml.cc/virtual/2025/poster/46436

- **Link**: https://icml.cc/virtual/2025/poster/46436

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Model (VLM) Agents are stateful, autonomous entities capable of perceiving and interacting with their environments through vision and language.Multi-agent systems comprise specialized agents who collaborate to solve a (complex) task. A core security property is robustness , stating that the system maintains its integrity during adversarial attacks. Multi-agent systems lack robustness, as a successful exploit against one agent can spread and infect other agents to undermine the entire system's integrity. We propose a defense Cowpox to provably enhance the robustness of a multi-agent system by a distributed mechanism that improves the recovery rate of agents by limiting the expected number of infections to other agents.The core idea is to generate and distribute a special cure sample that immunizes an agent against the attack before exposure. We demonstrate the effectiveness of Cowpox empirically and provide theoretical robustness guarantees.

</details>

---

## 148. ELITE: Enhanced Language-Image Toxicity Evaluation for Safety

- [ ] ELITE: Enhanced Language-Image Toxicity Evaluation for Safety | https://icml.cc/virtual/2025/poster/46445

- **Link**: https://icml.cc/virtual/2025/poster/46445

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current Vision Language Models (VLMs) remain vulnerable to malicious prompts that induce harmful outputs. Existing safety benchmarks for VLMs primarily rely on automated evaluation methods, but these methods struggle to detect implicit harmful content or produce inaccurate evaluations. Therefore, we found that existing benchmarks have low levels of harmfulness, ambiguous data, and limited diversity in image-text pair combinations. To address these issues, we propose the ELITE benchmark, a high-quality safety evaluation benchmark for VLMs, underpinned by our enhanced evaluation method, the ELITE evaluator. The ELITE evaluator explicitly incorporates a toxicity score to accurately assess harmfulness in multimodal contexts, where VLMs often provide specific, convincing, but unharmful descriptions of images. We filter out ambiguous and low-quality image-text pairs from existing benchmarks using the ELITE evaluator and generate diverse combinations of safe and unsafe image-text pairs. Our experiments demonstrate that the ELITE evaluator achieves superior alignment with human evaluations compared to prior automated methods, and the ELITE benchmark offers enhanced benchmark quality and diversity. By introducing ELITE, we pave the way for safer, more robust VLMs, contributing essential tools for evaluating and mitigating safety risks in real-world applications.

</details>

---

## 149. What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities

- [ ] What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities | https://icml.cc/virtual/2025/poster/46463

- **Link**: https://icml.cc/virtual/2025/poster/46463

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As multimodal large language models (MLLMs) advance, MLLM-based virtual agents have demonstrated remarkable performance. However, existing benchmarks face significant limitations, including uncontrollable task complexity, extensive manual annotation, and a lack of multidimensional evaluation. In response to these challenges, we introduce OmniBench, a self-generating, graph-based benchmark with an automated pipeline for synthesizing tasks of controllable complexity through subtask composition. To evaluate the diverse capabilities of virtual agents on the graph, we further present OmniEval, a multidimensional evaluation framework that includes subtask-level evaluation, graph-based metrics, and comprehensive tests across 10 capabilities. Our synthesized dataset contains 36k graph-structured tasks across 20 scenarios, achieving a 91% human acceptance rate. Training on our graph-structured data shows that it improves generalization across environments. We conduct multidimensional evaluations for virtual agents, revealing their performance across various capabilities and paving the way for future advancements. Our project is available at https://omni-bench.github.io.

</details>

---

## 150. Improving LLM Video Understanding with 16 Frames Per Second

- [ ] Improving LLM Video Understanding with 16 Frames Per Second | https://icml.cc/virtual/2025/poster/46540

- **Link**: https://icml.cc/virtual/2025/poster/46540

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human vision is dynamic and continuous. However, in video understanding with multimodal large language models (LLMs), existing methods primarily rely on static features extracted from images sampled at a fixed low frame rate of frame-per-second (FPS) $\leqslant$2, leading to critical visual information loss. In this paper, we introduce F-16, the first multimodal LLM designed for high-frame-rate video understanding. By increasing the frame rate to 16 FPS and compressing visual tokens within each 1-second clip, F-16 efficiently captures dynamic visual features while preserving key semantic information.Experimental results demonstrate that higher frame rates considerably enhance video understanding across multiple benchmarks, providing a new approach to improving video LLMs beyond scaling model size or training data. F-16 achieves state-of-the-art performance among 7-billion-parameter video LLMs on both general and fine-grained video understanding benchmarks, such as Video-MME and TemporalBench. Furthermore, F-16 excels in complex spatiotemporal tasks, including high-speed sports analysis (*e.g.*, basketball, football, gymnastics, and diving), outperforming SOTA proprietary visual models like GPT-4o and Gemini-1.5-pro.Additionally, we introduce a novel decoding method for F-16 that enables highly efficient low-frame-rate inference without requiring model retraining. We will release the source code, model checkpoints, and data at [https://github.com/bytedance/F-16](https://github.com/bytedance/F-16).

</details>

---

## 151. SlimLLM: Accurate Structured Pruning for Large Language Models

- [ ] SlimLLM: Accurate Structured Pruning for Large Language Models | https://icml.cc/virtual/2025/poster/46559

- **Link**: https://icml.cc/virtual/2025/poster/46559

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models(LLMs) have garnered significant attention and demonstrated impressive capabilities in a wide range of applications. However, due to their enormous computational costs, the deployment and application of LLMs are often severely limited. To address this issue, structured pruning is an effective solution to compress the parameters of LLMs. Determining the importance of each sub-module in LLMs and minimizing performance loss are critical issues that need to be carefully addressed in structured pruning. In this paper, we propose an effective and fast structured pruning method named SlimLLM for large language models. For channel and attention head pruning, we evaluate the importance based on the entire channel or head, rather than merely aggregating the importance of individual elements within a sub-module. This approach enables a more holistic consideration of the interdependence among elements within the sub-module. In addition, we design a simple linear regression strategy for the output matrix to quickly recover performance. We also propose layer-based importance ratio to determine the pruning ratio for each layer. Based on the LLaMA benchmark results, our SlimLLM outperforms other methods and achieves state-of-the-art performance.

</details>

---

## 152. I Think, Therefore I Diffuse: Enabling Multimodal In-Context Reasoning in Diffusion Models

- [ ] I Think, Therefore I Diffuse: Enabling Multimodal In-Context Reasoning in Diffusion Models | https://icml.cc/virtual/2025/poster/46563

- **Link**: https://icml.cc/virtual/2025/poster/46563

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents ThinkDiff, a novel alignment paradigm that empowers text-to-image diffusion models with multimodal in-context understanding and reasoning capabilities by integrating the strengths of vision-language models (VLMs). Existing multimodal diffusion finetuning methods largely focus on pixel-level reconstruction rather than in-context reasoning, and are constrained by the complexity and limited availability of reasoning-based datasets. ThinkDiff addresses these challenges by leveraging vision-language training as a proxy task, aligning VLMs with the decoder of an encoder-decoder large language model (LLM) instead of a diffusion decoder. This proxy task builds on the observation that the LLM decoder shares the same input feature space with diffusion decoders that use the corresponding LLM encoder for prompt embedding. As a result, aligning VLMs with diffusion decoders can be simplified through alignment with the LLM decoder. Without complex training and datasets, ThinkDiff effectively unleashes understanding, reasoning, and composing capabilities in diffusion models. Experiments demonstrate that ThinkDiff significantly improves accuracy from 19.2% to 46.3% on the challenging CoBSAT benchmark for multimodal in-context reasoning generation, with only 5 hours of training on 4 A100 GPUs. Additionally, ThinkDiff demonstrates exceptional performance in composing multiple images and texts into logically coherent images. Project page: https://mizhenxing.github.io/ThinkDiff.

</details>

---

## 153. Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation

- [ ] Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation | https://icml.cc/virtual/2025/poster/46649

- **Link**: https://icml.cc/virtual/2025/poster/46649

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-form video processing fundamentally challenges vision-language models (VLMs) due to the high computational costs of handling extended temporal sequences. Existing token pruning and feature merging methods often sacrifice critical temporal dependencies or dilute semantic information. We introduce differential distillation, a principled approach that systematically preserves task-relevant information while suppressing redundancy. Based on this principle, we develop ViLAMP, a hierarchical video-language model that processes hour-long videos at "mixed precision" through two key mechanisms: (1) differential keyframe selection that maximizes query relevance while maintaining temporal distinctiveness at the frame level and (2) differential feature merging that preserves query-salient features in non-keyframes at the patch level. Hence, ViLAMP retains full information in keyframes while reducing non-keyframes to their most salient features, resembling mixed-precision training. Extensive experiments demonstrate ViLAMP's superior performance across five video understanding benchmarks, particularly on long-form content. Notably, ViLAMP can process ultra-long videos (up to 10K frames) on a single NVIDIA A100 GPU, achieving substantial computational efficiency while maintaining state-of-the-art performance. Code and model are available at https://github.com/steven-ccq/ViLAMP.

</details>

---

## 154. Federated Disentangled Tuning with Textual Prior Decoupling and Visual Dynamic Adaptation

- [ ] Federated Disentangled Tuning with Textual Prior Decoupling and Visual Dynamic Adaptation | https://icml.cc/virtual/2025/poster/46665

- **Link**: https://icml.cc/virtual/2025/poster/46665

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Federated Parameter-Efficient Fine-Tuning aims to adapt Vision-Language Models for downstream tasks in distributed environments. However, data heterogeneity across participants hinders collaborative effectiveness, necessitating personalized adaptation to cover distinct data distributions. Current personalized methods suffer from two limitations. 1) Textual Property Loss: Existing methods facilitate the collaboration between decoupled prompts at the feature level, which potentially undermines the textual properties of the prompts. 2) Visual Feature Diversity: The diversity of visual features makes it challenging to leverage naive image features directly for image-text alignment in downstream tasks. In this work, we propose Federated Disentangled Tuning with Textual Prior Decoupling and Visual Dynamic Adaptation (FedDDA) to overcome the above limitations. Specifically, we encourage decoupling prompts in a way that maximizes the efficacy of prior knowledge, which is essential for maintaining a coherent linguistic context. Furthermore, we design a visual adaption model to reshape visual space to optimally align with the textual space. Extensive experiments on various image classification tasks show the effectiveness of our work in addressing data heterogeneity. The codes are released at https://github.com/MoratalYang/FedDDA.

</details>

---

## 155. Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models

- [ ] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models | https://icml.cc/virtual/2025/poster/46673

- **Link**: https://icml.cc/virtual/2025/poster/46673

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models, such as CLIP, have achieved significant success in aligning visual and textual representations, becoming essential components of many multi-modal large language models (MLLMs) like LLaVA and OpenFlamingo. However, numerous studies have identified CLIP's limited fine-grained perception as a critical drawback, leading to substantial failures in downstream MLLMs. In contrast, vision-centric foundation models like DINOv2 demonstrate remarkable capabilities in capturing fine details from images. In this work, we propose a novel kernel-based method to align CLIP's visual representation with that of DINOv2, ensuring that the resulting embeddings maintain compatibility with text embeddings while enhancing perceptual capabilities. Our alignment objective is designed for efficient stochastic optimization. Following this image-only alignment fine-tuning, the visual encoder retains compatibility with the frozen text encoder and exhibits significant improvements in zero-shot object recognition, fine-grained spatial reasoning, and localization. By integrating the aligned visual encoder, downstream MLLMs also demonstrate enhanced performance. The code and models are available at https://github.com/peterant330/KUEA.

</details>

---

## 156. CPCF: A Cross-Prompt Contrastive Framework for Referring Multimodal Large Language Models

- [ ] CPCF: A Cross-Prompt Contrastive Framework for Referring Multimodal Large Language Models | https://icml.cc/virtual/2025/poster/46688

- **Link**: https://icml.cc/virtual/2025/poster/46688

- **Conference**: ICML

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring MLLMs extend conventional multimodal large language models by allowing them to receive referring visual prompts and generate responses tailored to the indicated regions. However, these models often suffer from suboptimal performance due to incorrect responses tailored to misleading areas adjacent to or similar to the target region. This work introduces CPCF, a novel framework to address this issue and achieve superior results. CPCF contrasts outputs generated from the indicated visual prompt with those from contrastive prompts sampled from misleading regions, effectively suppressing the influence of erroneous information outside the target region on response generation. To further enhance the effectiveness and efficiency of our framework, several novel designs are proposed, including a prompt extraction network to automatically identify suitable contrastive prompts, a self-training method that leverages unlabeled data to improve training quality, and a distillation approach to reduce the additional computational overhead associated with contrastive decoding. Incorporating these novel designs, CPCF achieves state-of-the-art performance, as demonstrated by extensive experiments across multiple benchmarks. Project page: https://lanyunzhu.site/CPCF/

</details>

---

