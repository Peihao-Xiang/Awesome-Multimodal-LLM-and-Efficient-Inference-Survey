# CVPR 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_cvpr2025_papers.csv

## 1. Alignment, Mining and Fusion: Representation Alignment with Hard Negative Mining and Selective Knowledge Fusion for Medical Visual Question Answering

- [ ] Alignment, Mining and Fusion: Representation Alignment with Hard Negative Mining and Selective Knowledge Fusion for Medical Visual Question Answering | https://cvpr.thecvf.com/virtual/2025/poster/32389

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32389

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical Visual Question Answering (Med-VQA) is a challenging task that requires a deep understanding of both medical images and textual questions. Although recent works leveraging Medical Vision-Language Pre-training (Med-VLP) have shown strong performance on the Med-VQA task, there is still no unified solution for modality alignment, and the issue of hard negatives remains under-explored. Additionally, commonly used knowledge fusion techniques for Med-VQA may introduce irrelevant information. In this work, we propose a framework to address these challenges through three key contributions: (1) a unified solution for heterogeneous modality alignments across multiple levels, modalities, views, and stages, leveraging methods such as contrastive learning and optimal transport theory; (2) a hard negative mining method that employs soft labels for multi-modality alignments and enforces the hard negative pair discrimination; and (3) a Gated Cross-Attention Module for Med-VQA that integrates the answer vocabulary as prior knowledge and select relevant information from it. Our framework outperforms the previous state-of-the-art on widely used Med-VQA datasets like RAD-VQA, SLAKE, PathVQA and VQA-2019. The code will be publicly available.

</details>

---

## 2. ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation

- [ ] ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/32394

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32394

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in multimodal large language models (MLLMs) have expanded research in video understanding, primarily focusing on high-level tasks such as video captioning and question-answering. Meanwhile, a smaller body of work addresses dense, pixel-precise segmentation tasks, which typically involve category-guided or referral-based object segmentation. Although both research directions are essential for developing models with human-level video comprehension, they have largely evolved separately, with distinct benchmarks and architectures. This paper aims to unify these efforts by introducing ViCaS, a new dataset containing thousands of challenging videos, each annotated with detailed, human-written captions and temporally consistent, pixel-accurate masks for multiple objects with phrase grounding. Our benchmark evaluates models on both holistic/high-level understanding and language-guided, pixel-precise segmentation. We also present carefully validated evaluation measures and propose an effective model architecture that can tackle our benchmark. All annotations, as well as the code and model weights will be made public.

</details>

---

## 3. Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks

- [ ] Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks | https://cvpr.thecvf.com/virtual/2025/poster/32399

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32399

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have showcased remarkable performance in image and natural language understanding, such as image captioning and response generation. As the practical applications of vision-language models become increasingly widespread, their potential safety and robustness issues raise concerns that adversaries may evade the system and cause these models to generate toxic content through malicious attacks. Therefore, evaluating the robustness of open-source VLMs against adversarial attacks has garnered growing attention, with transfer-based attacks as a representative black-box attacking strategy. However, most existing transfer-based attacks neglect the importance of the semantic correlations between vision and text modalities, leading to sub-optimal adversarial example generation and attack performance. To address this issue, we present Chain of Attack (CoA), which iteratively enhances the generation of adversarial examples based on the multi-modal semantic update using a series of intermediate attacking steps, achieving superior adversarial transferability and efficiency. A unified attack success rate computing method is further proposed for automatic evasion evaluation. Extensive experiments conducted under the most realistic and high-stakes scenario, demonstrate that our attacking strategy is able to effectively mislead models to generate targeted responses using only black-box attacks without any knowledge of the victim models. The comprehensive robustness evaluation in our paper provides insight into the vulnerabilities of VLMs and offers a reference for the safety considerations of future model developments. The code will be made publically available.

</details>

---

## 4. EventGPT: Event Stream Understanding with Multimodal Large Language Models

- [ ] EventGPT: Event Stream Understanding with Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32407

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32407

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Event cameras record visual information as asynchronous pixel change streams, excelling at scene perception under unsatisfactory lighting or high-dynamic conditions. Existing multimodal large language models (MLLMs) concentrate on natural RGB images, failing in scenarios where event data fits better. In this paper, we introduce EventGPT, the first MLLM for event stream understanding, to the best of our knowledge, marking a pioneering attempt to integrate large language models (LLMs) with event stream comprehension. Our EventGPT comprises an event encoder, followed by a spatio-temporal aggregator, a linear projector, an event-language adapter, and an LLM. Firstly, RGB image-text pairs generated by GPT are leveraged to warm up the linear projector, referring to LLaVA, as the gap between natural image and language modalities is relatively smaller. Secondly, we construct a synthetic yet large dataset, N-ImageNet-Chat, consisting of event frames and corresponding texts to enable the use of the spatio-temporal aggregator and to train the event-language adapter, thereby aligning event features more closely with the language space. Finally, we gather an instruction dataset, Event-Chat, which contains extensive real-world data to fine-tune the entire model, further enhancing its generalization ability. We construct a comprehensive evaluation benchmark, and extensive experiments demonstrate that EventGPT outperforms previous state-of-the-art MLLMs in generation quality, descriptive accuracy, and reasoning capability.

</details>

---

## 5. ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models

- [ ] ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32412

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32412

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The growing popularity of Contrastive Language-Image Pretraining (CLIP) has led to its widespread application in various visual downstream tasks. To enhance CLIP's effectiveness, efficient few-shot adaptation techniques have been widely adopted. Among these approaches, training-free methods, particularly caching methods exemplified by Tip-Adapter, have gained attention for their lightweight adaptation without the need for additional fine-tuning. In this paper, we revisit Tip-Adapter from a kernel perspective, showing that caching methods function as local adapters and are connected to a well-established kernel literature. Leveraging this insight, we offer a theoretical understanding of how these methods operate and suggest multiple avenues for enhancing over the Tip-Adapter baseline. Notably, our analysis shows the importance of incorporating global information in local adapters. Therefore, we subsequently propose a global method that learns a proximal regularizer in a reproducing kernel Hilbert space (RKHS) using CLIP as a base learner. Our method, that we call ProKeR (Proximal Kernel ridge Regression), has a closed form solution and achieves state-of-the-art performance across 11 datasets in the standard few-shot adaptation benchmark.

</details>

---

## 6. TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models

- [ ] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32411

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32411

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks. TAPT is a test-time defense method that learns defensive bimodal (textual and visual) prompts to robustify the inference process of CLIP. Specifically, it is an unsupervised method that optimizes the defensive prompts for each test sample by minimizing a multi-view entropy and aligning adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9\% against AutoAttack (AA), while largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning methods across various backbones, achieving an average robustness improvement of at least 36.6\%.

</details>

---

## 7. DiscoVLA: Discrepancy Reduction in Vision, Language, and Alignment for Parameter-Efficient Video-Text Retrieval

- [ ] DiscoVLA: Discrepancy Reduction in Vision, Language, and Alignment for Parameter-Efficient Video-Text Retrieval | https://cvpr.thecvf.com/virtual/2025/poster/32425

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32425

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The parameter-efficient adaptation of the image-text pretraining model CLIP for video-text retrieval is a prominent area of research. While CLIP is focused on image-level vision-language matching, video-text retrieval demands comprehensive understanding at the video level. Three key discrepancies emerge in the transfer from image-level to video-level: vision, language, and alignment. However, existing methods mainly focus on vision while neglecting language and alignment. In this paper, we propose Discrepancy Reduction in Vision, Language, and Alignment (DiscoVLA), which simultaneously mitigates all three discrepancies. Specifically, we introduce Image-Video Features Fusion to integrate image-level and video-level features, effectively tackling both vision and language discrepancies. Additionally, we generate pseudo image captions to learn fine-grained image-level alignment. To mitigate alignment discrepancies, we propose Image-to-Video Alignment Distillation, which leverages image-level alignment knowledge to enhance video-level alignment. Extensive experiments demonstrate the superiority of our DiscoVLA. In particular, on MSRVTT with CLIP (ViT-B/16), DiscoVLA outperforms previous methods by 2.2% R@1 and 7.5% R@sum. Our code will be made available.

</details>

---

## 8. MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation

- [ ] MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation | https://cvpr.thecvf.com/virtual/2025/poster/32433

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32433

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mobile manipulation is the fundamental challenge for robotics to assist humans with diverse tasks and environments in everyday life. However, conventional mobile manipulation approaches often struggle to generalize across different tasks and environments because of the lack of large-scale training.In contrast, recent advances in vision-language-action (VLA) models have shown impressive generalization capabilities, but these foundation models are developed for fixed-base manipulation tasks.Therefore, we propose an efficient policy adaptation framework to transfer pre-trained VLA models of fix-base manipulation to mobile manipulation, so that high generalization ability across tasks and environments can be achieved in mobile manipulation policy.Specifically, we utilize pre-trained VLA models to generate waypoints of the end-effector with high generalization ability. We design motion planning objectives for the mobile base and the robot arm, which aim at maximizing the physical feasibility of the trajectory. Finally, we present an efficient bi-level objective optimization framework for trajectory generation, where the upper-level optimization predicts waypoints for base movement to enhance the manipulator policy space, and the lower-level optimization selects the optimal end-effector trajectory to complete the manipulation task. Extensive experimental results on OVMM and the real world demonstrate that our method achieves a 4.2\% higher success rate than the state-of-the-art mobile manipulation, and only requires 50 training cost for real world deployment due to the strong generalization ability in the pre-trained VLA models.

</details>

---

## 9. GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation

- [ ] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/32431

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32431

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper proposes a novel framework utilizing multi-modal large language models (MLLMs) for referring video object segmentation (RefVOS). Previous MLLM-based methods commonly struggle with the dilemma between "Ref" and "VOS": they either specialize in understanding a few key frames (global reasoning) or tracking objects on continuous frames (local reasoning), and rely on external VOS or frame selectors to mitigate the other end of the challenge. However, our framework GLUS shows that Global and Local consistency can be Unified into a single video Segmentation MLLM: a set of sparse "context frames" provides global information, while a stream of continuous "query frames" conducts local object tracking. This is further supported by jointly training the MLLM with a pre-trained VOS memory bank to simultaneously digest short-range and long-range temporal information. To improve the information efficiency within the limited context window of MLLMs, we introduce object contrastive learning to distinguish hard false-positive objects and a self-refined framework to identify crucial frames and perform propagation. By collectively integrating these insights, our GLUS delivers a simple yet effective baseline, achieving new state-of-the-art for MLLMs on the MeViS and Ref-Youtube-VOS benchmark.

</details>

---

## 10. Towards All-in-One Medical Image Re-Identification

- [ ] Towards All-in-One Medical Image Re-Identification | https://cvpr.thecvf.com/virtual/2025/poster/32452

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32452

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical image re-identification (MedReID) is under-explored so far, despite its critical applications in personalized healthcare and privacy protection.In this paper, we introduce a thorough benchmark and a unified model for this problem.First, to handle various medical modalities, we propose a novel Continuous Modality-based Parameter Adapter (ComPA). ComPA condenses medical content into a continuous modality representation and dynamically adjusts the modality-agnostic model with modality-specific parameters at runtime. This allows a single model to adaptively learn and process diverse modality data.Furthermore, we integrate medical priors into our model by aligning it with a bag of pre-trained medical foundation models, in terms of the differential features.Compared to single-image feature, modeling the inter-image difference better fits the re-identification problem, which involves discriminating multiple images.We evaluate the proposed model against 25 foundation models and 8 large multi-modal language models across 11 image datasets, demonstrating consistently superior performance.Additionally, we deploy the proposed MedReID technique to two real-world applications, i.e., history-augmented personalized diagnosis and medical privacy protection.

</details>

---

## 11. SpiritSight Agent: Advanced GUI Agent with One Look

- [ ] SpiritSight Agent: Advanced GUI Agent with One Look | https://cvpr.thecvf.com/virtual/2025/poster/32460

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32460

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interface (GUI) agents show amazing abilities in assisting human-computer interaction, automating human user's navigation on digital devices. An ideal GUI agent is expected to achieve high accuracy, low latency, and compatibility for different GUI platforms. Recent vision-based approaches have shown promise by leveraging advanced Vision Language Models (VLMs). While they generally meet the requirements of compatibility and low latency, these vision-based GUI agents tend to have low accuracy due to their limitations in element grounding. To address this issue, we propose $\textbf{SpiritSight}$, a vision-based, end-to-end GUI agent that excels in GUI navigation tasks across various GUI platforms. First, we create a multi-level, large-scale, high-quality GUI dataset called $\textbf{GUI-Lasagne}$ using scalable methods, empowering SpiritSight with robust GUI understanding and grounding capabilities. Second, we introduce the $\textbf{Universal Block Parsing (UBP)}$ method to resolve the ambiguity problem in dynamic high-resolution of visual inputs, further enhancing SpiritSight's ability to ground GUI objects. Through these efforts, SpiritSight agent outperforms other advanced methods on diverse GUI benchmarks, demonstrating its superior capability and compatibility in GUI navigation tasks. The models and code will be made available upon publications.

</details>

---

## 12. HOIGen-1M: A Large-scale Dataset for Human-Object Interaction Video Generation

- [ ] HOIGen-1M: A Large-scale Dataset for Human-Object Interaction Video Generation | https://cvpr.thecvf.com/virtual/2025/poster/32465

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32465

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-video (T2V) generation has made tremendous progress in generating complicated scenes based on texts. However, human-object interaction (HOI) often cannot be precisely generated by current T2V models due to the lack of large-scale videos with accurate captions for HOI. To address this issue, we introduce HOIGen-1M, the first large-scale dataset for HOI Generation, consisting of over one million high-quality videos collected from diverse sources. In particular, to guarantee the high quality of videos, we first design an efficient framework to automatically curate HOI videos using the powerful multimodal large language models (MLLMs), and then the videos are further cleaned by human annotators. Moreover, to obtain accurate textual captions for HOI videos, we design a novel video description method based on a Mixture-of-Multimodal-Experts (MoME) strategy that not only generates expressive captions but also eliminates the hallucination by individual MLLM. Furthermore, due to the lack of an evaluation framework for generated HOI videos, we propose two new metrics to assess the quality of generated videos in a coarse-to-fine manner. Extensive experiments reveal that current T2V models struggle to generate high-quality HOI videos and confirm that our HOIGen-1M dataset is instrumental for improving HOI video generation.

</details>

---

## 13. RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics

- [ ] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics | https://cvpr.thecvf.com/virtual/2025/poster/32478

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32478

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spatial understanding is a crucial capability for robots to make grounded decisions based on their environment. This foundational skill enables robots not only to perceive their surroundings but also to reason about and interact meaningfully within the world. In modern robotics, these capabilities are taken on by visual language models, and they face significant challenges when applied to spatial reasoning context due to their training data sources. These sources utilize general-purpose image datasets, and they often lack sophisticated spatial scene understanding capabilities. For example, the datasets do not address reference frame comprehension — spatial relationships require clear contextual understanding, whether from a ego-centric, object-centric, or world-centric perspective, which allow for effective real-world interaction. To address this issue, we introduce RoboSpatial, a large-scale spatial understanding dataset consisting of real indoor and tabletop scenes captured as 3D scans and ego-centric images, annotated with rich spatial information relevant to robotics. The dataset includes 1M images, 5K 3D scans, and 3M annotated spatial relationships, with paired 2D egocentric images and 3D scans to make it both 2D and 3D ready. Our experiments show that models trained with RoboSpatial outperform baselines on downstream tasks such as spatial affordance prediction, spatial relationship prediction, and robotics manipulation.

</details>

---

## 14. CLIP-driven Coarse-to-fine Semantic Guidance for Fine-grained Open-set Semi-supervised Learning

- [ ] CLIP-driven Coarse-to-fine Semantic Guidance for Fine-grained Open-set Semi-supervised Learning | https://cvpr.thecvf.com/virtual/2025/poster/32494

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32494

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fine-grained open-set semi-supervised learning (OSSL) investigates a practical scenario where unlabeled data may contain fine-grained out-of-distribution (OOD) samples. Due to the subtle visual differences among in-distribution (ID) samples, as well as between ID and OOD samples, it is extremely challenging to separate ID and OOD samples. Recent Vision-Language Models, such as CLIP, have shown excellent generalization capabilities. However, it tends to focus on general attributes, and thus is insufficient to distinguish the fine-grained details. To tackle the issues, in this paper, we propose a novel CLIP-driven coarse-to-fine semantic-guided framework, named CFSG-CLIP, by progressively filtering and focusing the distinctive fine-grained clues. Specifically, CFSG-CLIP comprises a coarse-guidance module and a fine-guidance module derived from the pre-trained CLIP model. In the coarse-guidance module, we design a semantic filtering strategy to initially filter out local visual features guided by cross-modality guidance. Then, in the fine-guidance module, we further design a visual-semantic injection strategy, which embeds category-related visual cues into the visual encoder to further refine the local visual features. By the designed dual-guidance framework, the local subtle cues are progressively discovered to distinct the subtle difference between ID and OOD samples. Extensive experiments demonstrates that CFSG-CLIP is able to not only improve the reliability of the fine-grained semi-supervised learning training process, but also achieves a competitive performance on multiple fine-grained datasets.

</details>

---

## 15. SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model

- [ ] SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model | https://cvpr.thecvf.com/virtual/2025/poster/32496

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32496

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D affordance segmentation aims to link human instructions to touchable regions of 3D objects for embodied manipulations. Existing efforts typically adhere to single-object, single-affordance paradigms, where each affordance type or explicit instruction strictly corresponds to a specific affordance region and are unable to handle long-horizon tasks. Such a paradigm cannot actively reason about complex user intentions that often imply sequential affordances. In this paper, we introduce the Sequential 3D Affordance Reasoning task, which extends the traditional paradigm by reasoning from cumbersome user intentions and then decomposing them into a series of segmentation maps. Toward this, we construct the first instruction-based affordance segmentation benchmark that includes reasoning over both single and sequential affordances, comprising 180K instruction-point cloud pairs. Based on the benchmark, we propose our model, SeqAfford, to unlock the 3D multi-modal large language model with additional affordance segmentation abilities, which ensures reasoning with world knowledge and fine-grained affordance grounding in a cohesive framework. We further introduce a multi-granular language-point integration module to endow 3D dense prediction. Extensive experimental evaluations show that our model excels over well-established methods and exhibits open-world generalization with sequential reasoning abilities.

</details>

---

## 16. Efficient Motion-Aware Video MLLM

- [ ] Efficient Motion-Aware Video MLLM | https://cvpr.thecvf.com/virtual/2025/poster/32495

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32495

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Most current video MLLMs rely on uniform frame sampling and image-level encoders, resulting in inefficient data processing and limited motion awareness. To address these challenges, we introduce EMA , an E fficient M otion- A ware video MLLM that utilizes compressed video structures as inputs. We propose a motion-aware GOP (Group of Pictures) encoder that fuses spatial and motion information within a GOP unit in the compressed video stream, generating compact, informative visual tokens. By integrating fewer but denser RGB frames with more but sparser motion vectors in this native slow-fast input architecture, our approach reduces redundancy and enhances motion representation. Additionally, we introduce MotionBench, a benchmark for evaluating motion understanding across four motion types: linear, curved, rotational, and contact-based. Experimental results show that EMA achieves state-of-the-art performance on both MotionBench and popular video question answering benchmarks, while reducing inference costs. Moreover, EMA demonstrates strong scalability, as evidenced by its competitive performance on long video understanding benchmarks.

</details>

---

## 17. Task-Aware Clustering for Prompting Vision-Language Models

- [ ] Task-Aware Clustering for Prompting Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32503

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32503

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has attracted widespread attention in adapting vision-language models to downstream tasks. Existing methods largely rely on optimization strategies to ensure the task-awareness of learnable prompts. Due to the scarcity of task-specific data, overfitting is prone to occur. The resulting prompts often do not generalize well or exhibit limited task-awareness. To address this issue, we propose a novel Task-Aware Clustering (TAC) framework for prompting vision-language models, which increases the task-awareness of learnable prompts by introducing task-aware pre-context. The key ingredients are as follows: (a) generating task-aware pre-context based on task-aware clustering that can preserve the backbone structure of a downstream task with only a few clustering centers, (b) enhancing the task-awareness of learnable prompts by enabling them to interact with task-aware pre-context via the well-pretrained encoders, and (c) preventing the visual task-aware pre-context from interfering the interaction between patch embeddings by masked attention mechanism. Extensive experiments are conducted on benchmark datasets, covering the base-to-novel, domain generalization, and cross-dataset transfer settings. Ablation studies validate the effectiveness of key ingredients. Comparative results show the superiority of our TAC over competitive counterparts. The code will be made publicly available.

</details>

---

## 18. CASP: Compression of Large Multimodal Models Based on Attention Sparsity

- [ ] CASP: Compression of Large Multimodal Models Based on Attention Sparsity | https://cvpr.thecvf.com/virtual/2025/poster/32507

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32507

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we propose an extreme compression technique for Large Multimodal Models (LMMs). While previous studies have explored quantization as an efficient post-training compression method for Large Language Models (LLMs), low-bit compression for multimodal models remains under-explored. The redundant nature of inputs in multimodal models results in a highly sparse attention matrix. We theoretically and experimentally demonstrate that the attention matrix's sparsity bounds the compression error of the Query and Key weight matrices. Based on this, we introduce CASP, a model compression technique for LMMs. Our approach performs a data-aware low-rank decomposition on the Query and Key weight matrix, followed by quantization across all layers based on an optimal bit allocation process. CASP is compatible with any quantization technique and enhances state-of-the-art 2-bit quantization methods (AQLM and QuIP#) by an average of 21% on image- and video-language benchmarks. The code is provided in the supplementary materials.

</details>

---

## 19. SnowMaster: Comprehensive Real-world Image Desnowing via MLLM with Multi-Model Feedback Optimization

- [ ] SnowMaster: Comprehensive Real-world Image Desnowing via MLLM with Multi-Model Feedback Optimization | https://cvpr.thecvf.com/virtual/2025/poster/32533

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32533

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Snowfall poses a significant challenge to visual data processing, requiring specialized desnowing algorithms. However, current models often struggle with generalization due to their reliance on synthetic datasets, creating a domain gap. Evaluating real snowfall images is difficult due to the lack of ground truth. To tackle these issues, we introduce a large-scale, high-quality dataset of 10,000 annotated real snow scenes, develop a dataset with 36k preference pairs based on human expert rankings, enhance multimodal large language models' perception of snowfall images using direct preference optimization (DPO), and refine desnowing models through a mean teacher semi-supervised framework with high-quality pseudo-label screening. This Framework substantially improves the generalization and performance of desnowing models on real snowfall images.

</details>

---

## 20. Realistic Test-Time Adaptation of Vision-Language Models

- [ ] Realistic Test-Time Adaptation of Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32541

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32541

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The zero-shot capabilities of Vision-Language Models (VLMs) have been widely leveraged to improve predictive performance. However, previous works on transductive or test-time adaptation (TTA) often make strong assumptions about the data distribution, such as the presence of all classes. Our work challenges these favorable deployment scenarios, and introduces a more realistic evaluation framework, including: (i) a variable number of effective classes for adaptation within a single batch, and (ii) non-i.i.d. batches of test samples in online adaptation settings. We provide comprehensive evaluations, comparisons, and ablation studies that demonstrate how current transductive or TTA methods for VLMs systematically compromise the models’ initial zero-shot robustness across various realistic scenarios, favoring performance gains under advantageous assumptions about the test samples' distributions. Furthermore, we introduce Stat${\cal A}$, a versatile method that could handle a wide range of deployment scenarios, including those with a variable number of effective classes at test time. Our approach incorporates a novel regularization term designed specifically for VLMs, which acts as a statistical anchor preserving the initial text-encoder knowledge, particularly in low-data regimes. Code will be made available.

</details>

---

## 21. Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?

- [ ] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves? | https://cvpr.thecvf.com/virtual/2025/poster/32542

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32542

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Improving semantic grounding in Vision-Language Models (VLMs) often involves collecting domain-specific training data, refining the network architectures, or modifying the training recipes. In this work, we venture into an orthogonal direction and explore self-correction in VLMs focusing on semantic grounding. We find that VLMs can correct their own semantic grounding mistakes when properly prompted and framed for the task, without any fine-tuning or even access to oracle feedback. We also introduce a self-correction framework in an iterative setting which consistently improves performance across all models investigated. Overall, we show that iterative self-correction consistently improves VLM performance in semantic grounding by up to 8.4 accuracy points across all models investigated, without requiring fine-tuning, additional architectural changes, or external data. Our exploration of self-correction also reveals that, even after several rounds of feedback, strong models like GPT-4V and GPT-4o  retain limited capability in leveraging oracle feedback, suggesting promising directions for further research.

</details>

---

## 22. MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders

- [ ] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders | https://cvpr.thecvf.com/virtual/2025/poster/32553

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32553

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual encoders are fundamental components in vision-language models (VLMs), each showcasing unique strengths derived from various pre-trained visual foundation models. To leverage the various capabilities of these encoders, recent studies incorporate multiple encoders within a single VLM, leading to a considerable increase in computational cost. In this paper, we present Mixture-of-Visual-Encoder Knowledge Distillation (MoVE-KD), a novel framework that distills the unique proficiencies of multiple vision encoders into a single, efficient encoder model. Specifically, to mitigate conflicts and retain the unique characteristics of each teacher encoder, we employ low-rank adaptation (LoRA) and mixture-of-experts (MoEs) to selectively activate specialized knowledge based on input features, enhancing both adaptability and efficiency. To regularize the KD process and enhance performance, we propose an attention-based distillation strategy that adaptively weighs the different visual encoders and emphasizes valuable visual tokens, reducing the burden of replicating comprehensive but distinct features from multiple teachers. Comprehensive experiments on popular VLMs, such as LLaVA and LLaVA-NeXT, validate the effectiveness of our method. The code will be released.

</details>

---

## 23. SpatialCLIP: Learning 3D-aware Image Representations from Spatially Discriminative Language

- [ ] SpatialCLIP: Learning 3D-aware Image Representations from Spatially Discriminative Language | https://cvpr.thecvf.com/virtual/2025/poster/32556

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32556

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) learns robust visual models through language supervision, making it a crucial visual encoding technique for various applications. However, CLIP struggles with comprehending spatial concepts in images, potentially restricting the spatial intelligence of CLIP-based AI systems. In this work, we propose SpatialCLIP, an enhanced version of CLIP with better spatial understanding capabilities. To capture the intricate 3D spatial relationships in images, we improve both "visual model" and "language supervision" of CLIP. Specifically, we design 3D-inspired ViT to replace the standard ViT in CLIP. By lifting 2D image tokens into 3D space and incorporating design insights from point cloud networks, our visual model gains greater potential for spatial perception. Meanwhile, captions with accurate and detailed spatial information are very rare. To explore better language supervision for spatial understanding, we re-caption images and perturb their spatial phrases as negative descriptions, which compels the visual model to seek spatial cues to distinguish these hard negative captions. With the enhanced visual model, we introduce SpatialLLaVA, following the same LLaVA-1.5 training protocol, to investigate the importance of visual representations for MLLM's spatial intelligence. Furthermore, we create SpatialBench, a benchmark specifically designed to evaluate CLIP and MLLM in spatial reasoning. SpatialCLIP and SpatialLLaVA achieve substantial performance improvements, demonstrating stronger capabilities in spatial perception and reasoning, while maintaining comparable results on general-purpose benchmarks.

</details>

---

## 24. GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks

- [ ] GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks | https://cvpr.thecvf.com/virtual/2025/poster/32567

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32567

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have recently shown promising advancements in sequential decision-making tasks through task-specific fine-tuning. However, common fine-tuning methods, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) techniques like Proximal Policy Optimization (PPO), present notable limitations: SFT assumes Independent and Identically Distributed (IID) data, while PPO focuses on maximizing cumulative rewards. These limitations often restrict solution diversity and hinder generalization in multi-step reasoning tasks. To address these challenges, we introduce a novel framework, GFlowVLM, a framework that fine-tune VLMs using Generative Flow Networks (GFlowNets) to promote generation of diverse solutions for complex reasoning tasks. GFlowVLM models the environment as a non-Markovian decision process, allowing it to capture long-term dependencies essential for real-world applications. It takes observations and task descriptions as inputs to prompt chain-of-thought (CoT) reasoning which subsequently guides action selection. We use task based rewards to fine-tune VLM with GFlowNets. This approach enables VLMs to outperform prior fine-tuning methods, including SFT and RL. Empirical results demonstrate the effectiveness of GFlowVLM on complex tasks such as card games (NumberLine, BlackJack) and embodied planning tasks (ALFWorld), showing enhanced training efficiency, solution diversity, and stronger generalization capabilities across both in-distribution and out-of-distribution scenarios.

</details>

---

## 25. MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs through Disentangled Spatial-Temporal Representations

- [ ] MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs through Disentangled Spatial-Temporal Representations | https://cvpr.thecvf.com/virtual/2025/poster/32568

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32568

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we tackle action-scene hallucination in Video Large Language Models (Video-LLMs), where models incorrectly predict actions based on the scene context or scenes based on observed actions. We observe that existing Video-LLMs often suffer from action-scene hallucination due to two main factors. First, existing Video-LLMs intermingle spatial and temporal features by applying an attention operation across all tokens. Second, they use the standard Rotary Position Embedding (RoPE), which causes the text tokens to overemphasize certain types of tokens depending on their sequential orders. To address these issues, we introduce MASH-VLM, Mitigating Action-Scene Hallucination in Video-LLMs through disentangled spatial-temporal representations. Our approach includes two key innovations: (1) DST-attention, a novel attention mechanism that disentangles the spatial and temporal tokens within the LLM by using masked attention to restrict direct interactions between the spatial and temporal tokens; (2) Harmonic-RoPE, which extends the dimensionality of the positional IDs, allowing the spatial and temporal tokens to maintain balanced positions relative to the text tokens. To evaluate the action-scene hallucination in Video-LLMs, we introduce the UNSCENE benchmark with 1,320 videos and 4,078 QA pairs. Extensive experiments demonstrate that MASH-VLM achieves state-of-the-art results on the UNSCENE benchmark, as well as on existing video understanding benchmarks.

</details>

---

## 26. GRAPHGPT-O: Synergistic Multimodal Comprehension and Generation on Graphs

- [ ] GRAPHGPT-O: Synergistic Multimodal Comprehension and Generation on Graphs | https://cvpr.thecvf.com/virtual/2025/poster/32574

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32574

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of Multimodal Large Language Models (MLLMs) has enabled the integration of multiple modalities, including texts and images, within the large language model (LLM) framework.However, texts and images are usually interconnected, forming a multimodal attributed graph (MMAG).It is underexplored how MLLMs can incorporate the relational information (i.e., graph structure) and semantic information (i.e., texts and images) on such graphs for multimodal comprehension and generation.In this paper, we propose GraphGPT-o, which supports omni-multimodal understanding and creation on MMAGs.We first comprehensively study linearization variants to transform semantic and structural information as input for MLLMs.Then, we propose a hierarchical aligner that enables deep graph encoding, bridging the gap between MMAGs and MLLMs.Finally, we explore the inference choices, adapting MLLM to interleaved text and image generation in graph scenarios. Extensive experiments on three datasets from different domains demonstrate the effectiveness of our proposed method.

</details>

---

## 27. CALICO: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models

- [ ] CALICO: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32572

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32572

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Large Vision-Language Models (LVLMs) advances have sparked significant progress in general-purpose vision tasks through visual instruction tuning. While some works have demonstrated the capability of LVLMs to generate segmentation masks that align phrases with natural language descriptions in a single image, they struggle with segmentation-grounded comparisons across multiple images, particularly at finer granularities such as object parts. In this paper, we introduce the new task of *part-focused semantic co-segmentation*, which seeks to identify and segment common and unique objects and parts across multiple images. To address this task, we present CALICO, the first LVLM that can segment and reason over multiple masks across images, enabling object comparison based on their constituent parts. CALICO features two novel components, a Correspondence Extraction Module, which captures semantic-rich information to identify part-level correspondences between objects, and a Correspondence Adaptation Module, which embeds this information into the LLM and facilitates multi-image understanding in a parameter-efficient manner. To support training and evaluation, we curate MIXEDPARTS, a comprehensive multi-image segmentation dataset containing $\sim$2.4M samples across $\sim$44K images with diverse object and part categories. Experimental results show CALICO, finetuned on only 0.3\% of its architecture, achieves robust performance in part-focused semantic co-segmentation. Code, models, and data are available at anon.link.

</details>

---

## 28. GroundingFace: Fine-grained Face Understanding via Pixel Grounding Multimodal Large Language Model

- [ ] GroundingFace: Fine-grained Face Understanding via Pixel Grounding Multimodal Large Language Model | https://cvpr.thecvf.com/virtual/2025/poster/32585

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32585

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Language Learning Models (MLLMs) have shown remarkable performance in image understanding, generation, and editing, with recent advancements achieving pixel-level grounding with reasoning. However, these models for common objects struggle with fine-grained face understanding. In this work, we introduce the \textbf{\textit{FacePlayGround-240K}} dataset, the first pioneering large-scale, pixel-grounded face caption and question-answer (QA) dataset, meticulously curated for alignment pretraining and instruction-tuning. We present the \textbf{\textit{GroundingFace}} framework, specifically designed to enhance fine-grained face understanding. This framework significantly augments the capabilities of existing grounding models in face part segmentation, face attribute comprehension, while preserving general scene understanding. Comprehensive experiments validate that our approach surpasses current state-of-the-art models in pixel-grounded face captioning/QA and various downstream tasks, including face captioning, referring segmentation, and zero-shot face attribute recognition.

</details>

---

## 29. Can Text-to-Video Generation help Video-Language Alignment?

- [ ] Can Text-to-Video Generation help Video-Language Alignment? | https://cvpr.thecvf.com/virtual/2025/poster/32590

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32590

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent video-language alignment models are trained on sets of videos, each with an associated positive caption and a negative caption generated by large language models. A problem with this procedure is that negative captions may introduce linguistic biases, i.e., concepts are seen only as negatives and never associated with a video. While a solution would be to collect videos for the negative captions, existing databases lack the fine-grained variations needed to cover all possible negatives. In this work, we study whether synthetic videos can help to overcome this issue. Our preliminary analysis with multiple generators shows that, while promising on some tasks, synthetic videos harm the performance of the model on others. We hypothesize this issue is linked to noise (semantic and visual) in the generated videos and develop a method, SynViTA, that accounts for those. SynViTA dynamically weights the contribution of each synthetic video based on how similar its target caption is w.r.t. the real counterpart. Moreover, a semantic consistency loss makes the model focus on fine-grained differences across captions, rather than differences in video appearance. Experiments show that, on average, SynViTA improves over existing methods on VideoCon test sets and SSv2-Temporal, SSv2-Events, and ATP-Hard benchmarks, being a first promising step for using synthetic videos when learning video-language models.

</details>

---

## 30. Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation

- [ ] Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/32595

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32595

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Foundation Models (VFMs) and Vision-Language Models (VLMs) have gained traction in Domain Generalized Semantic Segmentation (DGSS) due to their strong generalization capabilities. However, existing DGSS methods often rely exclusively on either VFMs or VLMs, overlooking their complementary strengths. VFMs (e.g., DINOv2) excel at capturing fine-grained features, while VLMs (e.g., CLIP) provide robust text alignment but struggle with coarse granularity. Despite their complementary strengths, effectively integrating VFMs and VLMs with attention mechanisms is challenging, as the increased patch tokens complicate long-sequence modeling. To address this, we propose MFuser, a novel Mamba-based fusion framework that efficiently combines the strengths of VFMs and VLMs while maintaining linear scalability in token length. MFuser consists of two key components: MVFuser, which acts as a co-adapter to jointly fine-tune the two models by capturing both sequential and spatial dynamics; and MTEnhancer, a hybrid attention-Mamba module that refines text embeddings by incorporating image priors. Our approach achieves precise feature locality and strong text alignment without incurring significant computational overhead. Extensive experiments demonstrate that MFuser significantly outperforms state-of-the-art DGSS methods, achieving 68.19 mIoU on synthetic-to-real and 71.87 mIoU on real-to-real benchmarks. The code will be released upon acceptance.

</details>

---

## 31. MM-OR: A Large Multimodal Operating Room Dataset for Semantic Understanding of High-Intensity Surgical Environments

- [ ] MM-OR: A Large Multimodal Operating Room Dataset for Semantic Understanding of High-Intensity Surgical Environments | https://cvpr.thecvf.com/virtual/2025/poster/32596

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32596

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Operating rooms (ORs) are complex, high-stakes environments requiring precise understanding of interactions among medical staff, tools, and equipment for enhancing surgical assistance, situational awareness, and patient safety. Current datasets fall short in scale, realism and do not capture the multimodal nature of OR scenes, limiting progress in OR modeling. To this end, we introduce MM-OR, a realistic and large-scale multimodal spatiotemporal OR dataset, and the first dataset to enable multimodal scene graph generation. MM-OR captures comprehensive OR scenes containing RGB-D data, detail views, audio, speech transcripts, robotic logs, and tracking data and is annotated with panoptic segmentations, semantic scene graphs, and downstream task labels. Further, we propose MM2SG, the first multimodal large vision-language model for scene graph generation, and through extensive experiments, demonstrate its ability to effectively leverage multimodal inputs. Together, MM-OR and MM2SG establish a new benchmark for holistic OR understanding, and open the path towards multimodal scene analysis in complex, high-stakes environments. We will publish all our code and dataset upon acceptance.

</details>

---

## 32. POSTA: A Go-to Framework for Customized Artistic Poster Generation

- [ ] POSTA: A Go-to Framework for Customized Artistic Poster Generation | https://cvpr.thecvf.com/virtual/2025/poster/32604

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32604

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Poster design is a critical medium for visual communication. Prior work has explored automatic poster design using deep learning techniques, but these approaches lack text accuracy, user customization, and aesthetic appeal, limiting their applicability in artistic domains such as movies and exhibitions, where both clear content delivery and visual impact are essential. To address these limitations, we present POSTA: a modular framework powered by diffusion models and multimodal large language models (MLLMs) for customized artistic poster generation. The framework consists of three modules. Background Diffusion creates a themed background based on user input. Design MLLM then generates layout and typography elements that align with and complement the background style. Finally, to enhance the poster's aesthetic appeal, ArtText Diffusion applies additional stylization to key text elements. The final result is a visually cohesive and appealing poster, with a fully modular process that allows for complete customization. To train our models, we develop the PosterArt dataset, comprising high-quality artistic posters annotated with layout, typography, and pixel-level stylized text segmentation. Our comprehensive experimental analysis demonstrates POSTA’s exceptional controllability and design diversity, outperforming existing models in both text accuracy and aesthetic quality.

</details>

---

## 33. S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation

- [ ] S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation | https://cvpr.thecvf.com/virtual/2025/poster/32619

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32619

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The  latest  advancements  in  multi-modal  large  language models  (MLLMs)  have  spurred  a  strong  renewed  interest in end-to-end motion planning approaches for autonomous driving.   Many end-to-end approaches rely on human annotations to learn intermediate perception and prediction tasks,  while purely self-supervised approaches—which directly learn from sensor inputs to generate planning trajectories without human annotations—often underperform the state of the art. We observe a key gap in the input representation space:  end-to-end approaches built on MLLMs are often  pretrained  with  reasoning  tasks  in  perspective  view space rather than the native 3D space that autonomous vehicles plan in.  To this end, we propose PaLI-Driver, based on  the  popular  PaLI  vision-language  model.    PaLI-Driver uses a novel sparse volume strategy to seamlessly transform the strong visual representation of MLLMs from perspective view to 3D space without the need to finetune the vision encoder.   This representation aggregates multiview and multi-frame visual inputs and enables better pre diction  of  planning trajectories  in  3D  space.   To  validate our  method,  we  run  experiments  on  both  nuScenes  and our in-house collected dataset X-Planning.  Results show that PaLI-Driver performs favorably against existing supervised multi-task approaches while requiring no human annotations.  It also demonstrates great scalability when pretrained on large volumes of unannotated driving logs.

</details>

---

## 34. Believing is Seeing: Unobserved Object Detection using Generative Models

- [ ] Believing is Seeing: Unobserved Object Detection using Generative Models | https://cvpr.thecvf.com/virtual/2025/poster/32635

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32635

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Can objects that are not visible in an image---but are in the vicinity of the camera---be detected? This study introduces the novel tasks of 2D, 2.5D and 3D unobserved object detection for predicting the location of nearby objects that are occluded or lie outside the image frame.  We adapt several state-of-the-art pre-trained generative models to address this task, including 2D and 3D diffusion models and vision-language models, and show that they can be used to infer the presence of objects that are not directly observed.  To benchmark this task, we propose a suite of metrics that capture different aspects of performance.  Our empirical evaluation on indoor scenes from the RealEstate10k and NYU Depth v2 datasets demonstrate results that motivate the use of generative models for the unobserved object detection task.

</details>

---

## 35. FireEdit: Fine-grained Instruction-based Image Editing via Region-aware Vision Language Model

- [ ] FireEdit: Fine-grained Instruction-based Image Editing via Region-aware Vision Language Model | https://cvpr.thecvf.com/virtual/2025/poster/32646

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32646

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Currently, instruction-based image editing methods have made significant progress by leveraging the powerful cross-modal understanding capabilities of visual language models (VLMs). However, they still face challenges in three key areas: 1) complex scenarios; 2) semantic consistency; and 3) fine-grained editing. To address these issues, we propose FireEdit, an innovative \textbf{F}ine-grained \textbf{I}nstruction-based image editing framework that exploits a REgion-aware VLM. FireEdit is designed to accurately comprehend user instructions and ensure effective control over the editing process. We employ a VLM to precisely localize the desired editing regions within complex scenes. To enhance the fine-grained visual perception capabilities of the VLM, we introduce additional region tokens that complement the holistic image features and are integrated into the user's instructions. Relying solely on the output of the Language Model (LLM) to guide the diffusion model may result in suboptimal editing outcomes.Therefore, we propose a Time-Aware Target Injection module and a Hybrid Visual Cross Attention module. The former dynamically adjusts the guidance strength at various denoising stages by integrating timestep embeddings with the text embeddings. The latter enhances visual details for image editing, thereby preserving semantic consistency between the edited result and the source image. By combining the VLM enhanced with fine-grained region tokens and the time-dependent diffusion model, FireEdit demonstrates significant advantages in comprehending editing instructions and maintaining high semantic consistency. Extensive experiments indicate that our approach surpasses the state-of-the-art instruction-based image editing methods.

</details>

---

## 36. Narrating the Video: Boosting Text-Video Retrieval via Comprehensive Utilization of Frame-Level Captions

- [ ] Narrating the Video: Boosting Text-Video Retrieval via Comprehensive Utilization of Frame-Level Captions | https://cvpr.thecvf.com/virtual/2025/poster/32652

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32652

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent text-video retrieval, the use of additional captions from vision-language models has shown promising effects on the performance. However, existing models using additional captions often have struggled to capture the rich semantics, including temporal changes, inherent in the video. In addition, incorrect information caused by generative models can lead to inaccurate retrieval. To address these issues, we propose a new framework, Narrating the Video (NarVid), which strategically leverages the comprehensive information available from frame-level captions, the narration. The proposed NarVid exploits narration in multiple ways: 1) feature enhancement through cross-modal interactions between narration and video, 2) query-aware adaptive filtering to suppress irrelevant or incorrect information, 3) dual-modal matching score by adding query-video similarity and query-narration similarity, and 4) hard-negative loss to learn discriminative features from multiple perspectives using the two similarities from different views. Experimental results demonstrate that NarVid achieves state-of-the-art performance on various benchmark datasets. The code will be available at [github]

</details>

---

## 37. VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment

- [ ] VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment | https://cvpr.thecvf.com/virtual/2025/poster/32672

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32672

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

A fundamental aspect of compositional reasoning in a video is associating people and their actions across time. Recent years have seen great progress in general-purpose vision/video models and a move towards long-video understanding. While exciting, we take a step back and ask: are today’s models good at compositional reasoning on short videos? To this end, we introduce VELOCITI, a benchmark to study Video-LLMs by disentangling and assessing the comprehension of agents, actions, and their associations across multiple events. We adopt the Video-Language Entailment setup and propose StrictVLE that requires correct classification (rather than ranking) of the positive and negative caption. We evaluate several models and observe that even the best, LLaVA-OneVision (42.5%) and GPT-4o (44.3%), are far from human accuracy at 89.6%. Results show that action understanding lags behind agents, and negative captions created using entities appearing in the video perform worse than those obtained from pure text manipulation. We also present challenges with ClassicVLE and multiple-choice (MC) evaluation, strengthening our preference for StrictVLE. Finally, we validate that our benchmark requires visual inputs of multiple frames making it ideal to study video-language compositional reasoning.

</details>

---

## 38. RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness

- [ ] RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness | https://cvpr.thecvf.com/virtual/2025/poster/32679

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32679

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional feedback learning for hallucination reduction relies on labor-intensive manual labeling or expensive proprietary models.This leaves the community without  foundational knowledge about how to build high-quality feedback with open-source MLLMs.In this work, we introduce RLAIF-V, a novel framework that aligns MLLMs in a fully open-source paradigm. RLAIF-V maximally explores open-source MLLMs from two perspectives, including high-quality feedback data generation for preference learning and self-feedback guidance for inference-time scaling.Extensive experiments on seven benchmarks in both automatic and human evaluation show that RLAIF-V substantially enhances the trustworthiness of models at both preference learning and inference time. RLAIF-V 7B reduces object hallucination by 80.7\% and overall hallucination by 33.7\%. Remarkably, RLAIF-V 12B further reveals the self-alignment potential of open-source MLLMs, where the  model can learn from feedback of itself to achieve super GPT-4V trustworthiness.

</details>

---

## 39. Unveiling Visual Perception in Language Models: An Attention Head Analysis Approach

- [ ] Unveiling Visual Perception in Language Models: An Attention Head Analysis Approach | https://cvpr.thecvf.com/virtual/2025/poster/32684

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32684

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated remarkable progress in visual understanding. This impressive leap raises a compelling question: how can language models, initially trained solely on linguistic data, effectively interpret and process visual content? This paper aims to address this question with systematic investigation across 4 model families and 4 model scales, uncovering a unique class of attention heads that focus specifically on visual content. Our analysis reveals a strong correlation between the behavior of these attention heads, the distribution of attention weights, and their concentration on visual tokens within the input. These findings enhance our understanding of how LLMs adapt to multimodal tasks, demonstrating their potential to bridge the gap between textual and visual understanding. This work paves the way for the development of AI systems capable of engaging with diverse modalities.

</details>

---

## 40. Synthetic Visual Genome

- [ ] Synthetic Visual Genome | https://cvpr.thecvf.com/virtual/2025/poster/32689

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32689

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding and reasoning over visual relationships—spatial, functional, interactional, social, etc.—are considered to be a fundamental component of human cognition.Yet, despite the major advances in visual comprehension in multimodal language models, precise reasoning over relationships remains a challenge. We introduce Robin: an MLM instruction-tuned with densely annotated relationships capable of constructing high-quality dense scene graphs at scale. To train Robin, we curate SVG, a scene graph based instruction tuning dataset containing $33K$ images and $855K$ relationships for $170K$ objects by completing the missing relations in existing scene graphs using GPT4-V and a carefully designed filtering process—combining rule-based and model-based filtering techniques to ensure high-quality. To generate more accurate and rich scene graphs at scale for any image,  we introduce SG-EDIT: a self-distillation framework where GPT-4o refines Robin's predicted scene graphs by removing unlikely relations and/or suggesting relevant ones. Results show that our Robin-3B model, despite being trained on less than $3$ million instances, outperforms similar-size models trained on over $300$ million instances on relationship understanding benchmarks, and even surpasses larger models up to 13B parameters. Notably, it achieves state-of-the-art performance in referring expression comprehension with a score of $88.2$, surpassing the previous best of $87.4$. Our results suggest that training on the refined scene graph data is crucial to maintaining high performance across diverse visual reasoning tasks.

</details>

---

## 41. What’s in the Image? A Deep-Dive into the Vision of Vision Language Models

- [ ] What’s in the Image? A Deep-Dive into the Vision of Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32693

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32693

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have recently demonstrated remarkable capabilities in comprehending complex visual content. However, the mechanisms underlying how VLMs process visual information remain largely unexplored. In this paper,  we conduct a thorough empirical analysis, focusing on the attention modules across layers, by which we reveal several key insights about how these models process visual data: (i) the internal representation of the query tokens (e.g., representations of "describe the image"), is utilized by the model to store global image information; we demonstrate that the model generates surprisingly descriptive responses solely from these tokens, without direct access to image tokens.  (ii) Cross-modal information flow is predominantly influenced by the middle layers (approximately 25% of all layers), while early and late layers contribute only marginally. (iii) Fine-grained visual attributes and object details are directly extracted from image tokens in a spatially localized manner, i.e., the generated tokens associated with a specific object or attribute attend strongly to their corresponding regions in the image.  We propose novel quantitative evaluation to validate our observations, leveraging real-world complex visual scenes. Finally, we demonstrate the potential of our findings in facilitating efficient visual processing in state-of-the-art VLMs.

</details>

---

## 42. OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts

- [ ] OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts | https://cvpr.thecvf.com/virtual/2025/poster/32692

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32692

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of multi-modal language models (MLLMs) like GPT-4o has propelled the development of Omni language models, designed to process and proactively respond to continuous streams of multi-modal data. Despite their potential, evaluating their real-world interactive capabilities in streaming video contexts remains a formidable challenge. In this work, we introduce OmniMMI, a comprehensive multi-modal interaction benchmark tailored for OmniLLMs in streaming video contexts. OmniMMI encompasses over 1,121 real-world interactive videos and 2,290 questions, addressing two critical yet underexplored challenges in existing video benchmarks: streaming video understanding and proactive reasoning, across six distinct subtasks.  Moreover, we propose a novel framework, Multi-modal Multiplexing Modeling (M4), designed to enhance real-time interactive reasoning with minimum finetuning on pre-trained MLLMs. Extensive experimental results reveal that the existing MLLMs fall short in interactive streaming understanding, particularly struggling with proactive tasks and multi-turn queries. Our proposed M4, though lightweight, demonstrates a significant improvement in handling proactive tasks and real-time interactions.

</details>

---

## 43. Synthetic Data is an Elegant GIFT for Continual Vision-Language Models

- [ ] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32691

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32691

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained Vision-Language Models (VLMs) require Continual Learning (CL) to efficiently update their knowledge and adapt to various downstream tasks without retraining from scratch. However, for VLMs, in addition to the loss of knowledge previously learned from downstream tasks, pre-training knowledge is also corrupted during continual fine-tuning. This issue is exacerbated by the unavailability of original pre-training data, leaving VLM's generalization ability degrading. In this paper, we propose GIFT, a novel continual fine-tuning approach that utilizes synthetic data to overcome catastrophic forgetting in VLMs. Taking advantage of recent advances in text-to-image synthesis, we employ a pre-trained diffusion model to recreate both pre-training and learned downstream task data. In this way, the VLM can revisit previous knowledge through distillation on matching diffusion-generated images and corresponding text prompts. Leveraging the broad distribution and high alignment between synthetic image-text pairs in VLM's feature space, we propose a contrastive distillation loss along with an image-text alignment constraint. To further combat in-distribution overfitting and enhance distillation performance with limited amount of generated data, we incorporate adaptive weight consolidation, utilizing Fisher information from these synthetic image-text pairs and achieving a better stability-plasticity balance. Extensive experiments demonstrate that our method consistently outperforms previous state-of-the-art approaches across various settings.

</details>

---

## 44. O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models

- [ ] O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32701

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32701

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time prompt tuning for vision-language models (VLMs) are getting attention due to their ability to learn with unlabeled data without fine-tuning. Although test-time prompt tuning methods for VLMs can boost accuracy, the resulting models tend to demonstrate poor calibration, which casts doubts on the reliability and trustworthiness of these models. Notably, more attention needs to be devoted to calibrating the test-time prompt tuning in vision-language models. To this end, we propose a new approach, called O-TPT that introduces orthogonality constraints on the textual features corresponding to the learnable prompts for calibrating test-time prompt tuning in VLMsTowards introducing orthogonality constraints, we make the following contributions. First, we uncover new insights behind the suboptimal calibration performance of existing methods relying on textual feature dispersion. Second, we show that imposing a simple orthogonalization of textual features is a more effective approach towards obtaining textual dispersion.We conduct extensive experiments on various datasets with different backbones and baselines. Results indicate that our method consistently outperforms the state-of-the-art in significantly reducing the overall average calibration error. Also, our method surpasses the zero-shot calibration performance on fine-grained classification tasks. Our code will be made public upon acceptance.

</details>

---

## 45. Paint by Inpaint: Learning to Add Image Objects by Removing Them First

- [ ] Paint by Inpaint: Learning to Add Image Objects by Removing Them First | https://cvpr.thecvf.com/virtual/2025/poster/32708

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32708

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image editing has advanced significantly with the introduction of text-conditioned diffusion models. Despite this progress, seamlessly adding objects to images based on textual instructions without requiring user-provided input masks remains a challenge. We address this by leveraging the insight that removing objects (Inpaint) is significantly simpler than its inverse process of adding them (Paint), attributed to the utilization of segmentation mask datasets alongside inpainting models that inpaint within these masks. Capitalizing on this realization, by implementing an automated and extensive pipeline, we curate a filtered large-scale image dataset containing pairs of images and their corresponding object-removed versions. Using these pairs, we train a diffusion model to inverse the inpainting process, effectively adding objects into images. Unlike other editing datasets, ours features natural target images instead of synthetic ones; moreover, it maintains consistency between source and target by construction. Additionally, we utilize a large Vision-Language Model to provide detailed descriptions of the removed objects and a Large Language Model to convert these descriptions into diverse, natural-language instructions. Our quantitative and qualitative results show that the trained model surpasses existing models in both object addition and general editing tasks. To propel future research, we will release the dataset alongside the trained models.

</details>

---

## 46. VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models

- [ ] VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models | https://cvpr.thecvf.com/virtual/2025/poster/32712

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32712

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce a benchmark and learning framework for advancing video-text compositionality understanding, aimed at enhancing vision-language models (VLMs) in fine-grained temporal alignment. Unlike existing benchmarks focused on static image-text compositionality or isolated single-event videos, our benchmark focuses on fine-grained video-text alignment in continuous multi-event videos. Leveraging video-text datasets with temporally localized event captions (\eg ActivityNet-Captions, YouCook2), we create challenging negative samples with subtle temporal disruptions such as reordering, action word replacements, partial captioning, and combined disruptions that comprehensively test models’ compositional sensitivity across extended, cohesive video-text sequences. To enhance model performance, we propose a hierarchical pairwise preference loss that strengthens alignment with temporally accurate pairs and progressively reduces similarity for increasingly disrupted pairs, encouraging fine-grained compositional alignment. To mitigate the limited availability of densely annotated video data, we introduce a pretraining strategy that concatenates short video-caption pairs to simulate multi-event sequences, facilitating effective compositional learning. We evaluate large multimodal models (LMMs) on our benchmark, identifying both strengths and areas for improvement in video-text compositionality. Our work provides a comprehensive framework for assessing and advancing model capabilities in achieving fine-grained, temporally coherent video-text alignment.

</details>

---

## 47. Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector

- [ ] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector | https://cvpr.thecvf.com/virtual/2025/poster/32709

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32709

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Deepfake detection is a long-established research topic crucial for combating the spread of malicious misinformation. Unlike previous methods that provide either binary classification results or textual explanations for deepfake detection, we propose a novel method that delivers both simultaneously. Our method harnesses the multi-modal learning power of the pre-trained CLIP and the unprecedented interpretability of large language models (LLMs) to enhance both the generalization and interpretability of deepfake detection. Specifically, we introduce a multi-modal face forgery detector (M2F2-Det) that employs specially designed face forgery prompt learning, integrating zero-shot learning capabilities of the pre-trained CLIP to improve generalization to unseen forgeries.Also, M2F2-Det incorporates the LLM to provide detailed explanations for detection decisions, offering strong interpretability by bridging the gap between natural language and the subtle nuances of facial forgery detection. Empirically, we evaluate M2F2-Det for both detection and sentence generation tasks, on both of which M2F2-Det achieves state-of-the-art performance, showing its effectiveness in detecting and explaining diverse and unseen forgeries. Code and models will be released upon publication.

</details>

---

## 48. ImagineFSL: Self-Supervised Pretraining Matters on Imagined Base Set for VLM-based Few-shot Learning

- [ ] ImagineFSL: Self-Supervised Pretraining Matters on Imagined Base Set for VLM-based Few-shot Learning | https://cvpr.thecvf.com/virtual/2025/poster/32717

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32717

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adapting CLIP models for few-shot recognition has recently attracted significant attention. Despite considerable progress, these adaptations remain hindered by the pervasive challenge of data scarcity. Text-to-image models, capable of generating abundant photorealistic labeled images, offer a promising solution. However, existing approaches treat synthetic images merely as complements to real images, rather than as standalone knowledge repositories stemming from distinct foundation models. To overcome this limitation, we reconceptualize synthetic images as an imagined base set , i.e., a unique, large-scale synthetic dataset encompassing diverse concepts. We introduce a novel CLIP adaptation methodology called ImagineFSL , involving pretraining on the imagined base set followed by fine-tuning on downstream few-shot tasks. We find that, compared to no pretraining, both supervised   and self-supervised pretraining are beneficial, with the latter providing better performance. Building on this finding, we propose an improved self-supervised method tailored for few-shot scenarios, enhancing the transferability of representations from synthetic to real image domains. Additionally, we present an image generation pipeline that employs chain-of-thought and in-context learning techniques, harnessing foundation models to automatically generate diverse, realistic images. Our methods are validated across eleven datasets, consistently outperforming state-of-the-art methods by substantial margins.

</details>

---

## 49. Evaluating Vision-Language Models as Evaluators in Path Planning

- [ ] Evaluating Vision-Language Models as Evaluators in Path Planning | https://cvpr.thecvf.com/virtual/2025/poster/32727

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32727

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite their promise to perform complex reasoning, large language models (LLMs) have been shown to have limited effectiveness in end-to-end planning. This has inspired an intriguing question: if these models cannot plan well, can they still contribute to the planning framework as a helpful plan evaluator? In this work, we generalize this question to consider LLMs augmented with visual understanding, i.e., Vision-Language Models (VLMs). We introduce PathEval , a novel benchmark evaluating VLMs as plan evaluators in complex path-planning scenarios. Succeeding in the benchmark requires a VLM to be able to abstract traits of optimal paths from the scenario description, demonstrate precise low-level perception on each path, and integrate this information to decide the better path.  Our analysis of state-of-the-art VLMs reveals that these models face significant challenges on the benchmark. We observe that the VLMs can precisely abstract given scenarios to identify the desired traits and exhibit mixed performance in integrating the provided information. Yet, their vision component presents a critical bottleneck, with models struggling to perceive low-level details about a path. Our experimental results show that this issue cannot be trivially addressed via end-to-end fine-tuning; rather, task-specific discriminative adaptation of these vision encoders is needed for these VLMs to become effective path evaluators.

</details>

---

## 50. SAIST: Segment Any Infrared Small Target Model Guided by Contrastive Language-Image Pretraining

- [ ] SAIST: Segment Any Infrared Small Target Model Guided by Contrastive Language-Image Pretraining | https://cvpr.thecvf.com/virtual/2025/poster/32729

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32729

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Infrared Small Target Detection (IRSTD) aims to identify low signal-to-noise ratio small targets in infrared images with complex backgrounds, which is crucial for various applications. However, existing IRSTD methods typically rely solely on image modalities for processing, which fail to fully capture contextual information, leading to limited detection accuracy and adaptability in complex environments. Inspired by vision-language models, this paper proposes a novel framework, SAIST, which integrates textual information with image modalities to enhance IRSTD performance. The framework consists of two main components: Scene Recognition Contrastive Language-Image Pretraining (SR-CLIP) and CLIP-guided Segment Anything Model (CG-SAM).  SR-CLIP generates a set of visual descriptions through object-object similarity and object-scene relevance, embedding them into learnable prompts to refine the textual description set. This reduces the domain gap between vision and language, generating precise textual and visual prompts. CG-SAM utilizes the prompts generated by SR-CLIP to accurately guide the Mask Decoder in learning prior knowledge of background features, while incorporating infrared imaging equations to improve small target recognition in complex backgrounds and significantly reduce the false alarm rate. Additionally, this paper introduces the first multimodal IRSTD dataset, MIRSTD, which contains abundant image-text pairs. Experimental results demonstrate that the proposed SAIST method outperforms existing state-of-the-art approaches. The dataset and code will be made publicly available.

</details>

---

## 51. Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation

- [ ] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/32731

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32731

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We tackle open-vocabulary 3D scene understanding by introducing a novel data generation pipeline and training framework. Our method addresses three critical requirements for effective training: precise 3D region segmentation, comprehensive textual descriptions, and sufficient dataset scale. By leveraging state-of-the-art open-vocabulary image segmentation models and region-aware Vision-Language Models (VLM), we develop an automatic pipeline that generates high-quality 3D mask-text pairs. Applying this pipeline to multiple 3D scene datasets, we create Mosaic3D-5.6M, a dataset of over 30K annotated scenes with 5.6M mask-text pairs—significantly larger than existing datasets. Building upon this data, we propose Mosaic3D, a foundation model combining a 3D encoder trained with contrastive learning and a lightweight mask decoder for open-vocabulary 3D semantic and instance segmentation. Our approach achieves state-of-the-art results on open-vocabulary 3D semantic and instance segmentation tasks including ScanNet200, Matterport3D, and ScanNet++, with ablation studies validating the effectiveness of our large-scale training data.

</details>

---

## 52. VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?

- [ ] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos? | https://cvpr.thecvf.com/virtual/2025/poster/32736

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32736

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of Multimodal Large Language Models (MLLMs) has enabled significant progress in multimodal understanding, expanding their capacity to analyze video content.However, existing evaluation benchmarks for MLLMs primarily focus on abstract video comprehension, lacking a detailed assessment of their ability to understand video compositions, the nuanced interpretation of how visual elements combine and interact within highly compiled video contexts.We introduce VidComposition, a new benchmark specifically designed to evaluate the video composition understanding capabilities of MLLMs using carefully curated compiled videos and cinematic-level annotations.VidComposition includes 982 videos with 1706 multiple-choice questions, covering various compositional aspects such as camera movement, angle, shot size, narrative structure, character actions and emotions, etc.Our comprehensive evaluation of 33 open-source and proprietary MLLMs reveals a significant performance gap between human and model capabilities. This highlights the limitations of current MLLMs in understanding complex, compiled video compositions and offers insights into areas for further improvement.Our benchmark will be publicly available for evaluating more models.

</details>

---

## 53. Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly

- [ ] Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly | https://cvpr.thecvf.com/virtual/2025/poster/32734

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32734

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

M ultimodal L arge L anguage M odels (MLLMs) have displayed remarkable performance in multimodal tasks, particularly in visual comprehension. However, we reveal that MLLMs often generate incorrect answers even when they understand the visual content. To this end, we manually construct a benchmark with 12 categories and design evaluation metrics that assess the degree of error in MLLM responses even when the visual content is seemingly understood. Based on this benchmark, we test 15 leading MLLMs and analyze the distribution of attention maps and logits of some MLLMs. Our investigation identifies two primary issues: 1) most instruction tuning datasets predominantly feature questions that ``directly" relate to the visual content, leading to a bias in MLLMs' responses to other indirect questions, and 2) MLLMs’ attention to visual tokens is notably lower than to system and question tokens. We further observe that attention scores between questions and visual tokens as well as the model's confidence in the answers are lower in response to misleading questions than to straightforward ones. To address the first challenge, we introduce a paired positive and negative data construction pipeline to diversify the dataset. For the second challenge, we propose to enhance the model's focus on visual content during decoding by refining the text and visual prompt. For the text prompt, we propose a content-guided refinement strategy that performs preliminary visual content analysis to generate structured information before answering the question. Additionally, we employ a visual attention refinement strategy that highlights question-relevant visual tokens to increase the model’s attention to visual content that aligns with the question. Extensive experiments demonstrate that these challenges can be significantly mitigated with our proposed dataset and techniques. The benchmark, training set, and code will be available.

</details>

---

## 54. Navigating the Unseen: Zero-shot Scene Graph Generation via Capsule-Based Equivariant Features

- [ ] Navigating the Unseen: Zero-shot Scene Graph Generation via Capsule-Based Equivariant Features | https://cvpr.thecvf.com/virtual/2025/poster/32752

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32752

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In scene graph generation (SGG), the accurate prediction of unseen triples is essential for its effectiveness in downstream vision-language tasks. We hypothesize that the predicates of unseen triples can be viewed as transformations of seen predicates in feature space, and the essence of the zero-shot task is to bridge the gap caused by this transformation. Traditional models, however, have difficulty addressing this challenge, which we attribute to their inability to model the predicates equivariant. To overcome this limitation, we introduce a novel framework based on capsule networks (CAPSGG). We propose a $\textbf{Three-Stream Pipeline}$ that generates modality-specific representations for predicates, while building low-level predicate capsules of these modalities. Then these capsules are aggregated into high-level predicate capsules using a $\textbf{Routing Capsule Layer}$. In addition, we introduce $\textbf{GroupLoss}$ to aggregate capsules with the same predicate label into groups. This replaces the global loss with the intra-group loss, effectively balancing the learning of predicate invariance and equivariant features, while mitigating the impact of the severe long-tail distribution of the predicate categories. Our extensive experiments demonstrate the notable superiority of our approach over state-of-the-art methods, with zero-shot indicators outperforming up to $\textbf{132.26\\%}$  on SGCls task than the T-CAR [21]. Our code will be available upon publication.

</details>

---

## 55. Post-pre-training for Modality Alignment in Vision-Language Foundation Models

- [ ] Post-pre-training for Modality Alignment in Vision-Language Foundation Models | https://cvpr.thecvf.com/virtual/2025/poster/32768

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32768

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive language image pre-training (CLIP) is an essential component of building modern vision-language foundation models. While CLIP demonstrates remarkable zero-shot performance on downstream tasks, the multi-modal feature spaces still suffer from a modality gap, which is a gap between image and text feature clusters and limits downstream task performance. Although existing works attempt to address the modality gap by modifying pre-training or fine-tuning, they struggle with heavy training costs with large datasets or degradations of zero-shot performance. This paper presents CLIP-Refine, a post-pre-training method for CLIP models at a phase between pre-training and fine-tuning. CLIP-Refine aims to align the feature space with 1 epoch training on small image-text datasets without zero-shot performance degradations. To this end, we introduce two techniques: random feature alignment (RaFA) and hybrid contrastive-distillation (HyCD). RaFA aligns the image and text features to follow a shared prior distribution by minimizing the distance to random reference vectors sampled from the prior. HyCD updates the model with hybrid soft labels generated by combining ground-truth image-text pair labels and outputs from the pre-trained CLIP model. This contributes to achieving both maintaining the past knowledge and learning new knowledge to align features. Our extensive experiments with multiple classification and retrieval tasks show that CLIP-Refine succeeds in mitigating the modality gap and improving the zero-shot performance.

</details>

---

## 56. Vision-Language Embodiment for Monocular Depth Estimation

- [ ] Vision-Language Embodiment for Monocular Depth Estimation | https://cvpr.thecvf.com/virtual/2025/poster/32784

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32784

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Depth estimation is a core problem in robotic perception and vision tasks, but 3D reconstruction from a single image presents inherent uncertainties. With the development of deep learning, current methods primarily rely on inter-image relationships to train supervised models, often overlooking intrinsic information provided by the camera itself. From the perspective of embodied intelligence, perception and understanding are not only based on external data inputs but are also closely linked to the physical environment in which the model is embedded. Following this concept, we propose a method that embeds the camera model and its physical characteristics into a deep learning model to compute Embodied Scene Depth through interactions with road environments. This approach leverages the intrinsic properties of the camera and provides robust depth priors without the need for additional equipment.By combining Embodied Scene Depth with RGB image features, the model gains a comprehensive perspective of both geometric and visual details. Additionally, we incorporate text descriptions containing environmental content and depth information as another dimension of embodied intelligence, embedding them as scale priors for scene understanding, thus enriching the model’s perception of the scene. This integration of image and language — two inherently ambiguous modalities — leverages their complementary strengths for monocular depth estimation, ensuring a more realistic understanding of scenes in diverse environments. We validated this method on outdoor datasets KITTI and CityScapes, with experimental results demonstrating that this embodied intelligence-based depth estimation method consistently enhances model performance across different scenes.

</details>

---

## 57. Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction

- [ ] Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction | https://cvpr.thecvf.com/virtual/2025/poster/32789

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32789

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Building a generalizable self-correction system as human cognition is crucial for robots to recover from failures.Despite advancements in Multimodal Large Language Models (MLLMs) that empower robots with semantic reflection ability for failure, translating this semantic reflection into "how to correct" fine-grained robotic actions remains a significant challenge.To address this gap, we build the Phoenix framework, which leverages motion instruction as a bridge to connect high-level semantic reflection with low-level robotic action correction. In this motion-based self-reflection framework,we start with a dual-process motion adjustment mechanism with MLLMs to translate the semantic reflection into coarse-grained motion instruction adjustment. To leverage this motion instruction for guiding "how to correct" fine-grained robotic actions, a multi-task motion-conditioned diffusion policy is proposed to integrate visual observations for high-frequency robotic action correction.By combining these two models, we could shift the demand for generalization capability from the low-level manipulation policy to the MLLMs-driven motion refinement model and facilitate precise, fine-grained robotic action correction.Utilizing this framework, we further develop a continual learning method to automatically improve the model's capability from interactions with dynamic environments.The experiments conducted in both the RoboMimic simulation and real-world scenarios prove the superior generalization and robustness of our framework across a variety of manipulation tasks.

</details>

---

## 58. CTRL-O: Language-Controllable Object-Centric Visual Representation Learning

- [ ] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning | https://cvpr.thecvf.com/virtual/2025/poster/32790

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32790

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Object-centric representation learning aims to decompose visual scenes into fixed-size vectors called slots'' or object files'', where each slot captures a distinct object. Current state-of-the-art object-centric models have shown remarkable success in object discovery in diverse domains including complex real-world scenes. However, these models suffer from a key limitation: they lack controllability. Specifically, current object-centric models learn representations based on their preconceived understanding of objects and parts, without allowing user input to guide which objects are represented. Introducing controllability into object-centric models could unlock a range of useful capabilities, such as the ability to extract instance-specific representations from a scene. In this work, we propose a novel approach for user-directed control over slot representations by conditioning slots on language descriptions. The proposed ConTRoLlable Object-centric representation learning approach, which we term CTRL-O, achieves targeted object-language binding in complex real-world scenes without requiring mask supervision. Next, we apply these controllable slot representations on two downstream vision language tasks: text-to-image generation and visual question answering. We find that the proposed approach enables instance-specific text-to-image generation and also achieves strong performance on visual question answering.

</details>

---

## 59. ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos

- [ ] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos | https://cvpr.thecvf.com/virtual/2025/poster/32795

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32795

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) excel at retrieving information from lengthy text, but their vision-language counterparts (VLMs) face difficulties with hour-long videos, especially for temporal grounding. Specifically, these VLMs are constrained by frame limitations, often losing essential temporal details needed for accurate event localization in extended video content. We propose ReVisionLLM, a recursive vision-language model designed to locate events in hour-long videos. Inspired by human search strategies, our model initially targets broad segments of interest, progressively revising its focus to pinpoint exact temporal boundaries. Our model can seamlessly handle videos of vastly different lengths—from minutes to hours. We also introduce a hierarchical training strategy that starts with short clips to capture distinct events and progressively extends to longer videos. To our knowledge, ReVisionLLM is the first VLM capable of temporal grounding in hour-long videos, outperforming previous state-of-the-art methods across multiple datasets by a significant margin (e.g., +2.6\% R1@0.1 on MAD). The code is available in the supplementary and will be released.

</details>

---

## 60. ResCLIP: Residual Attention for Training-free Dense Vision-language Inference

- [ ] ResCLIP: Residual Attention for Training-free Dense Vision-language Inference | https://cvpr.thecvf.com/virtual/2025/poster/32791

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32791

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While vision-language models like CLIP have shown remarkable success in open-vocabulary tasks, their application is currently confined to image-level tasks, and they still struggle with dense predictions. Recent works often attribute such deficiency in dense predictions to the self-attention layers in the final block, and have achieved commendable results by modifying the original query-key attention to self-correlation attention, (e.g., query-query and key-key attention). However, these methods overlook the cross-correlation attention (query-key) properties, which capture the rich spatial correspondence. In this paper, we reveal that the cross-correlation of the self-attention in CLIP's non-final layers also exhibits localization properties. Therefore, we propose the Residual Cross-correlation Self-attention (RCS) module, which leverages the cross-correlation self-attention from intermediate layers to remold the attention in the final block. The RCS module effectively reorganizes spatial information, unleashing the localization potential within CLIP for dense vision-language inference. Furthermore, to enhance the focus on regions of the same categories and local consistency, we propose the Semantic Feedback Refinement (SFR) module, which utilizes semantic segmentation maps to further adjust the attention scores. By integrating these two strategies, our method, termed ResCLIP, can be easily incorporated into existing approaches as a plug-and-play module, significantly boosting their performance in dense vision-language inference. Extensive experiments across multiple standard benchmarks demonstrate that our method surpasses state-of-the-art training-free methods, validating the effectiveness of the proposed approach.

</details>

---

## 61. Compositional Caching for Training-free Open-vocabulary Attribute Detection

- [ ] Compositional Caching for Training-free Open-vocabulary Attribute Detection | https://cvpr.thecvf.com/virtual/2025/poster/32802

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32802

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Attribute detection is crucial for many computer vision tasks, as it enables systems to describe properties such as color, texture, and material. Current approaches often rely on labor-intensive annotation processes which are inherently limited: objects can be described at an arbitrary level of detail (e.g., color vs. color shades), leading to ambiguities when the annotators are not instructed carefully. Furthermore, they operate within a predefined set of attributes, reducing scalability and adaptability to unforeseen downstream applications. We present Compositional Caching (ComCa), a training-free method for open-vocabulary attribute detection that overcomes these constraints. ComCa requires only the list of target attributes and objects as input, using them to populate an auxiliary cache of images by leveraging web-scale databases and Large Language Models to determine attribute-object compatibility. To account for the compositional nature of attributes, cache images receive soft attribute labels. Those are aggregated at inference time based on the similarity between the input and cache images, refining the predictions of underlying Vision-Language Models (VLMs). Importantly, our approach is model-agnostic, compatible with various VLMs. Experiments on public datasets demonstrate that ComCa significantly outperforms zero-shot and cache-based baselines, competing with recent training-based methods, proving that a carefully designed training-free approach can successfully address open-vocabulary attribute detection.

</details>

---

## 62. CoSpace: Benchmarking Continuous Space Perception Ability for Vision-Language Models

- [ ] CoSpace: Benchmarking Continuous Space Perception Ability for Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32810

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32810

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have recently witnessed significant progress in visual comprehension. As the permitting length of image context grows, VLMs can now comprehend a broader range of views and spaces. Current benchmarks provide insightful analysis of VLMs in tasks involving complex visual instructions following, multi-image understanding and spatial reasoning. However, they usually focus on spatially irrelevant images or discrete images captured from varied viewpoints. The compositional characteristic of images captured from a static viewpoint remains underestimated. We term this characteristic as $\textbf{Continuous Space Perception}$. When observing a scene from a static viewpoint while shifting orientations, it produces a series of spatially continuous images, enabling the reconstruction of the entire space. In this paper, we present CoSpace, a multi-image visual understanding benchmark designed to assess the $\textbf{Co}$ntinuous $\textbf{Space}$ perception ability for VLMs. CoSpace contains 2,918 images and 1,626 question-answer pairs, covering seven types of tasks. We conduct evaluation across 16 proprietary and open-source VLMs. Results reveal that there exist pitfalls on the continuous space perception ability for most of the evaluated models, including proprietary ones. Interestingly, we find that the main discrepancy between open-source and proprietary models lies not in accuracy but in the consistency of responses. We believe that enhancing the ability of continuous space perception is essential for VLMs to perform effectively in real-world tasks and encourage further research to advance this capability.

</details>

---

## 63. LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation

- [ ] LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/32811

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32811

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We propose a training-free method for open-vocabulary semantic segmentation using Vision-and-Language Models (VLMs). Our approach enhances the initial per-patch predictions of VLMs through label propagation, which jointly optimizes predictions by incorporating patch-to-patch relationships. Since VLMs are primarily optimized for cross-modal alignment and not for intra-modal similarity, we use a Vision Model (VM) that is observed to better captures these relationships. We address resolution limitations inherent to patch-based encoders by applying label propagation at the pixel level as a refinement step, significantly improving segmentation accuracy near class boundaries. Our method called LPOSS+, performs inference over the entire image, avoiding window-based processing and thereby capturing contextual interactions across the full image. LPOSS+ achieves state-of-the-art performance across a diverse set of datasets.

</details>

---

## 64. Progress-Aware Video Frame Captioning

- [ ] Progress-Aware Video Frame Captioning | https://cvpr.thecvf.com/virtual/2025/poster/32812

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32812

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While image captioning provides isolated descriptions for individual images, and video captioning offers one single narrative for an entire video clip, our work explores an important middle ground: progress-aware video captioning at the frame level. This novel task aims to generate temporally fine-grained captions that not only accurately describe each frame but also capture the subtle progression of actions throughout a video sequence. Despite the strong capabilities of existing leading vision language models, they often struggle to discern the nuances of frame-wise differences. To address this, we propose ProgressCaptioner, a captioning model designed to capture the fine-grained temporal dynamics within an action sequence. Alongside, we develop the FrameCap dataset to support training and the FrameCapEval benchmark to assess caption quality. The results demonstrate that ProgressCaptioner significantly surpasses leading captioning models, producing precise captions that accurately capture action progression and set a new standard for temporal precision in video captioning. Finally, we showcase practical applications of our approach, specifically in aiding keyframe selection and advancing video understanding, highlighting its broad utility.

</details>

---

## 65. VISCO: Benchmarking Fine-Grained Critique and Correction Towards Self-Improvement in Visual Reasoning

- [ ] VISCO: Benchmarking Fine-Grained Critique and Correction Towards Self-Improvement in Visual Reasoning | https://cvpr.thecvf.com/virtual/2025/poster/32821

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32821

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The ability of large vision-language models (LVLMs) to critique and correct their reasoning is an essential building block towards their self-improvement. However, a systematic analysis of such capabilities in LVLMs is still lacking. We propose VISCO, the first benchmark to extensively analyze the fine-grained critique and correction capabilities of LVLMs. Compared to existing work that uses a single scalar value to critique the entire reasoning [4], VISCO features dense and fine-grained critique, requiring LVLMs to evaluate the correctness of each step in the chain-of-thought and provide natural language explanations to support their judgments. Extensive evaluation of 24 LVLMs demonstrates that human-written critiques significantly enhance the performance after correction, showcasing the potential of the self-improvement strategy. However, the model-generated critiques are less helpful and sometimes detrimental to the performance, suggesting that critique is the crucial bottleneck. We identified three common patterns in critique failures: failure to critique visual perception, reluctance to "say no", and exaggerated assumption of error propagation. To address these issues, we propose an effective LookBack strategy that revisits the image to verify each piece of information in the initial reasoning. \ourscritic{} significantly improves critique and correction performance by up to 13.5%.

</details>

---

## 66. RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models

- [ ] RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32822

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32822

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of large language models (LLMs) has significantly enhanced the capabilities of multimodal LLMs (MLLMs) as general assistants. However, lack of user-specific knowledge still restricts their application in human's daily life. In this paper, we introduce the R etrieval A ugmented P ersonalization (RAP) framework for MLLMs' personalization. Starting from a general MLLM, we turn it into a personalized assistant in three steps. (a) Remember: We design a key-value database to store user-related information, e.g. , user's name, avatar and other attributes. (b) Retrieve: When the user initiates a conversation, RAP will retrieve relevant information from the database using a multimodal retriever. (c) Generate: The input query and retrieved concepts' information are fed into MLLMs to generate personalized, knowledge-augmented responses. Unlike previous methods, RAP allows real-time concept editing via updating the external database. To further improve generation quality and alignment with user-specific information, we design a pipeline for data collection and create a specialized dataset for personalized training of MLLMs. Based on the dataset, we train a series of MLLMs as personalized multimodal assistants. By pretraining on large-scale dataset, RAP-MLLMs can generalize to infinite visual concepts without additional finetuning. Our models demonstrate outstanding flexibility and generation quality across a variety of tasks, such as personalized image captioning, question answering and visual recognition. The code, data and models will be publicly available.

</details>

---

## 67. SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving

- [ ] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving | https://cvpr.thecvf.com/virtual/2025/poster/32830

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32830

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The integration of Vision-Language Models (VLMs) into autonomous driving systems has shown promise in addressing key challenges such as learning complexity, interpretability, and common-sense reasoning. However, existing approaches often struggle with efficient integration and real-time decision-making due to computational demands. In this paper, we introduce SOLVE, an innovative framework that synergizes VLMs with end-to-end (E2E) models to enhance autonomous vehicle planning. Our approach emphasizes knowledge sharing at the feature level through a shared visual encoder, enabling comprehensive interaction between VLM and E2E components. We propose a Trajectory Chain-of-Thought (T-CoT) paradigm, which progressively refines trajectory predictions, reducing uncertainty and improving accuracy. By employing a temporal decoupling strategy, SOLVE achieves efficient asynchronous cooperation, aligning high-quality VLM outputs with E2E real-time performance. Evaluated on the nuScenes dataset, our method demonstrates significant improvements in trajectory prediction accuracy, paving the way for more robust and interpretable autonomous driving systems.

</details>

---

## 68. Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement

- [ ] Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement | https://cvpr.thecvf.com/virtual/2025/poster/32841

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32841

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Transformers (ViTs) have been widely applied in various computer vision and vision-language tasks. To gain insights into their robustness in practical scenarios, transferable adversarial examples on ViTs have been extensively studied. A typical approach to improving adversarial transferability is by refining the surrogate model. However, existing work on ViTs has restricted their surrogate refinement to backward propagation. In this work, we instead focus on Forward Propagation Refinement (FPR) and specifically refine two key modules of ViTs: attention maps and token embeddings. For attention maps, we propose Attention Map Diversification (AMD), which diversifies certain attention maps and also implicitly imposes beneficial gradient vanishing during backward propagation. For token embeddings, we propose Momentum Token Embedding (MTE), which accumulates historical token embeddings to stabilize the forward updates in both the Attention and MLP blocks. We conduct extensive experiments with adversarial examples transferred from ViTs to various CNNs and ViTs, demonstrating that our FPR outperforms the current best (backward) surrogate refinement method by up to 7.0\% on average.We also validate its superior against popular defenses and its compatibility with other transfer methods.

</details>

---

## 69. FineLIP: Extending CLIP’s Reach via Fine-Grained Alignment with Longer Text Inputs

- [ ] FineLIP: Extending CLIP’s Reach via Fine-Grained Alignment with Longer Text Inputs | https://cvpr.thecvf.com/virtual/2025/poster/32839

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32839

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As a pioneering vision-language model, CLIP (Contrastive Language-Image Pre-training) has achieved significant success across various domains and a wide range of downstream vision-language tasks. However, the text encoders in popular CLIP models are limited to processing only 77 text tokens, which constrains their ability to effectively handle longer, detail-rich captions. Additionally, CLIP models often struggle to effectively capture detailed visual and textual information, which hampers their performance on tasks that require fine-grained analysis. To address these limitations, we present a novel approach, FineLIP, that extends the capabilities of CLIP. FineLIP enhances cross-modal text-image mapping by incorporating Fine-grained alignment with Longer text input within the CLIP-style framework. FineLIP first extends the positional embeddings to handle longer text, followed by the dynamic aggregation of local image and text tokens. The aggregated results are then used to enforce fine-grained token-to-token cross-modal alignment. We validate our model on datasets with long, detailed captions across two tasks: zero-shot cross-modal retrieval and text-to-image generation. Quantitative and qualitative experimental results demonstrate the effectiveness of FineLIP, outperforming existing state-of-the-art approaches. Furthermore, comprehensive ablation studies validate the benefits of key design elements within FineLIP.

</details>

---

## 70. ExpertAF: Expert Actionable Feedback from Video

- [ ] ExpertAF: Expert Actionable Feedback from Video | https://cvpr.thecvf.com/virtual/2025/poster/32840

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32840

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Feedback is essential for learning a new skill or improving one's current skill-level. However, current methods for skill-assessment from video only provide scores or compare demonstrations, leaving the burden of knowing what to do differently on the user. We introduce a novel method to generate actionable feedback from video of a person doing a physical activity, such as basketball or soccer.  Our method takes a video demonstration and its accompanying 3D body pose and generates (1) free-form expert commentary describing what the person is doing well and what they could improve, and (2) a visual expert demonstration that incorporates the required corrections. We show how to leverage Ego-Exo4D's videos of skilled activity and expert commentary together with a strong language model to create a weakly-supervised training dataset for this task, and we devise a multimodal video-language model to infer coaching feedback. Our method is able to reason across multi-modal input combinations to output full-spectrum, actionable coaching---expert commentary, expert video retrieval, and expert pose generation---outperforming strong vision-language models on both established metrics and human preference studies. Code and data will be publicly released.

</details>

---

## 71. FINECAPTION: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity

- [ ] FINECAPTION: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity | https://cvpr.thecvf.com/virtual/2025/poster/32855

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32855

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advent of large Vision-Language Models (VLMs) has significantly advanced multimodal tasks, enabling more sophisticated and accurate integration of visual and textual information across various applications, including image and video captioning, visual question answering, and cross-modal retrieval.Despite their superior capabilities, VLMs still struggle with fine-grained compositional image region descriptions. Specifically, they have difficulty recognizing arbitrary segmentation masks as referential inputs, interpreting compositional aspect instructions for referencing, and precisely describing the compositional aspects of a region. However, compositionality—the ability to understand and generate novel combinations of known visual and textual components—is critical for facilitating coherent reasoning and understanding across modalities in VLMs. To address this issue, we propose OpenCompositionCap, a new dataset for multi-grained region compositional image captioning that distinguishes itself from prior works by introducing the new task of compositional aspect-aware regional image captioning. To support this endeavor, we also introduce a new VLM model, FineCaption. The empirical results illustrate the effectiveness of our proposed model compared with other strong VLMs. In addition, we analyze the capabilities of current VLMs in recognizing various visual prompts for compositional region image captioning, highlighting areas for improvement in VLM design and training.

</details>

---

## 72. SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding

- [ ] SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding | https://cvpr.thecvf.com/virtual/2025/poster/32856

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32856

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The remarkable success of Large Language Models (LLMs) has extended to the multimodal domain, achieving outstanding performance in image understanding and generation. Recent efforts to develop unified Multimodal Large Language Models (MLLMs) that integrate these capabilities have shown promising results. However, existing approaches often involve complex designs in model architecture or training pipeline, increasing the difficulty of model training and scaling. In this paper, we propose SynerGen-VL, a simple yet powerful encoder-free MLLM capable of both image understanding and generation. To address challenges identified in existing encoder-free unified MLLMs, we introduce the token folding mechanism and the vision-expert-based progressive alignment pretraining strategy, effectively supporting high-resolution image understanding while reducing training complexity. After being trained on large-scale mixed image-text data with a unified next-token prediction objective, SynerGen-VL achieves or surpasses the performance of existing encoder-free unified MLLMs with comparable or smaller parameter sizes, and narrows the gap with task-specific state-of-the-art models, highlighting a promising path toward future unified MLLMs. Our code and models shall be released.

</details>

---

## 73. Vision-Language Model IP Protection via Prompt-based Learning

- [ ] Vision-Language Model IP Protection via Prompt-based Learning | https://cvpr.thecvf.com/virtual/2025/poster/32870

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32870

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) like CLIP (Contrastive Language-Image Pre-Training) have seen remarkable success in visual recognition, highlighting the increasing need to safeguard the intellectual property (IP) of well-trained models. Effective IP protection extends beyond ensuring authorized usage; it also necessitates restricting model deployment to authorized data domains, particularly when the model is fine-tuned for specific target domains. However, current IP protection methods often rely solely on the visual backbone, which may lack sufficient semantic richness. To bridge this gap, we introduce IP-CLIP, a lightweight IP protection strategy tailored to CLIP, employing a prompt-based learning approach. By leveraging the frozen visual backbone of CLIP, we extract both image style and content information, incorporating them into the learning of IP prompt. This strategy acts as a robust barrier, effectively preventing the unauthorized transfer of features from authorized domains to unauthorized ones. Additionally, we propose a style-enhancement branch that constructs feature banks for both authorized and unauthorized domains. This branch integrates self-enhanced and cross-domain features, further strengthening IP-CLIP’s capability to block features from unauthorized domains. Finally, we present new three metrics designed to better balance the performance degradation of authorized and unauthorized domains. Comprehensive experiments in various scenarios demonstrate its promising potential for application in IP protection tasks for VLMs.

</details>

---

## 74. Reasoning to Attend: Try to Understand How <SEG> Token Works

- [ ] Reasoning to Attend: Try to Understand How <SEG> Token Works | https://cvpr.thecvf.com/virtual/2025/poster/32873

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32873

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current Large Multimodal Models (LMMs) empowered tasks such as visual grounding and segmentation typically rely on $\texttt{ }$ token as a text prompting to jointly optimize the vision-language model (e.g., LLaVA) and the downstream task-specified model ($\eg$, SAM). However, we observe that little research has looked into how it works when mapping language vocabulary embedding into corresponding vision codebook space. In this work, we first visualize the similarity maps, $\aka$ pseudo images, which are obtained by computing the dot product similarity between the $\texttt{ }$ token and the image token embedings derived from the last hidden layer in both LLaVA and SAM models. Intriguingly, we have found that a striking consistency holds in terms of activation responses in the pseudo images, which reveals that what $\texttt{ }$ token  contributes to is the semantic correspondences from image-text pairs. Specifically, $\texttt{ }$ token, a placeholder expanded in text vocabulary, extensively queries within individual tokenized image patches to map the semantics of an object from text to the paired image while the Large Language Models (LLMs) is being fine tined. Upon above findings, we present READ, which facilitates LMMs' resilient $\textbf{REA}$soning capability of where to atten\textbf{D} under the guidance of highly activated points borrowed from pseudo images. Remarkably, READ features an intuitive design, Similarity as Points module (SasP), which can be seamlessly applied to existing $\texttt{ }$-like paradigms with negligible overheads in a plug-and-play fashion. Also, extensive experiments have been conducted on highly challenging reasoning segmentation dataset and widely used RefCOCO(+/g) referring segmentation dataset. To validate whether READ suffers from catastrophic forgetting of previous skills after fine-tuning, as observed in prior works ($\eg$, LISA), we further assess its generation ability  on FP-RefCOCO(+/g) dataset. All code, models will be publicly available.

</details>

---

## 75. NLPrompt: Noise-Label Prompt Learning for Vision-Language Models

- [ ] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32883

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32883

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of vision-language foundation models, such as CLIP, has revolutionized image-text representation, enabling a broad range of applications via prompt learning. Despite its promise, real-world datasets often contain noisy labels that can degrade prompt learning performance. In this paper, we demonstrate that using mean absolute error (MAE) loss in prompt learning, named PromptMAE, significantly enhances robustness against noisy labels while maintaining high accuracy. Though MAE is straightforward and recognized for its robustness, it is rarely used in noisy-label learning due to its slow convergence and poor performance outside prompt learning scenarios. To elucidate the robustness of PromptMAE, we leverage feature learning theory to show that MAE can suppress the influence of noisy samples, thereby improving the signal-to-noise ratio and enhancing overall robustness. Additionally, we introduce PromptOT, a prompt-based optimal transport data purification method to enhance the robustness further. PromptOT employs text encoder representations in vision-language models as prototypes to construct an optimal transportation matrix. This matrix effectively partitions datasets into clean and noisy subsets, allowing for the application of cross-entropy loss to the clean subset and MAE loss to the noisy subset. Our Noise-Label Prompt Learning method, named NLPrompt, offers a simple and efficient approach that leverages the expressive representation and precise alignment capabilities of vision-language models for robust prompt learning.  We validate NLPrompt through extensive experiments across various noise settings, demonstrating significant performance improvements.

</details>

---

## 76. FastVLM: Efficient Vision Encoding for Vision Language Models

- [ ] FastVLM: Efficient Vision Encoding for Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32887

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32887

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) like LLaVA encode images into tokens aligned to the word embedding space of the LLM decoder. Scaling input image resolution is essential for improving performance, especially in text-rich image understanding tasks. However, popular visual encoders such as CLIP-pretrained ViTs become inefficient at high resolutions due to the large number of tokens and high encoding latency caused by stacked self-attention layers. At different operational resolutions, the vision encoder of a VLM can be optimized along two axes: reducing encoding latency and minimizing the number of visual tokens passed to the LLM, thereby lowering overall latency. In this work, we introduce FastVLM, which achieves an optimized trade-off between resolution, latency, and accuracy by incorporating FastViTHD—a new hybrid vision encoder that outputs fewer tokens and significantly reduces encoding time while processing high-resolution images. We provide a comprehensive efficiency analysis of the interplay between image resolution, vision latency, number of visual tokens, and LLM size. In the LLaVA-1.5 setup, we achieve 3.2$\times$ improvement in overall time-to-first-token (TTFT) while maintaining similar performance on VLM benchmarks compared to prior works. On text-rich evaluations like TextVQA and DocVQA, FastVLM obtains +8.4\% and +12.5\% better accuracy than ConvLLaVA at a similar operating point of 144 visual tokens. Compared to LLaVa-OneVision at the highest resolution (1152$\times$1152), FastVLM achieves comparable performance on key benchmarks like SeedBench and MMMU, using the same LLM, but with 85$\times$ faster TTFT, 3$\times$ less vision instruction tuning data, and a vision encoder that is 3.4$\times$ smaller.

</details>

---

## 77. Data Distributional Properties As Inductive Bias for Systematic Generalization

- [ ] Data Distributional Properties As Inductive Bias for Systematic Generalization | https://cvpr.thecvf.com/virtual/2025/poster/32893

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32893

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Deep neural networks (DNNs) struggle at systematic generalization (SG). Several studies have evaluated the possibility to promote SG through the proposal of novel architectures, loss functions or training methodologies. Few studies, however, have focused on the role of training data properties in promoting SG. In this work, we investigate the impact of certain data distributional properties, as inductive biases for the SG ability of a multi-modal language model. To this end, we study three different properties. First, data diversity, instantiated as an increase in the possible values a latent property in the training distribution may take. Second, burstiness, where we probabilistically restrict the number of possible values of latent factors on particular inputs during training. Third, latent intervention, where a particular latent factor is altered randomly during training. We find that all three factors significantly enhance SG, with diversity contributing an 89\% absolute increase in accuracy in the most affected property. Through a series of experiments, we test various hypotheses to understand why these properties promote SG. Finally, we find that Normalized Mutual Information (NMI) between latent attributes in the training distribution is strongly predictive of out-of-distribution generalization. We find that a mechanism by which lower NMI induces SG is in the geometry of representations. In particular, we find that NMI induces more parallelism in neural representations (i.e., input features coded in parallel neural vectors) of the model, a property related to the capacity of reasoning by analogy.

</details>

---

## 78. ROD-MLLM: Towards More Reliable Object Detection in Multimodal Large Language Models

- [ ] ROD-MLLM: Towards More Reliable Object Detection in Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32889

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32889

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have demonstrated strong language understanding and generation capabilities, excelling in visual tasks like referring and grounding. However, due to task type limitations and dataset scarcity, existing MLLMs only ground objects present in images and cannot reject non-existent objects effectively, resulting in unreliable predictions. In this paper, we introduce ROD-MLLM, a novel MLLM for Reliable Object Detection using free-form language. We propose a query-based localization mechanism to extract low-level object features. By aligning global and object-level visual information with text space, we leverage the large language model (LLM) for high-level comprehension and final localization decisions, overcoming the language understanding limitations of normal detectors. To enhance language-based object detection, we design an automated data annotation pipeline and construct the dataset ROD. This pipeline uses the referring capabilities of existing MLLMs and chain-of-thought techniques to generate diverse expressions corresponding to zero or multiple objects, addressing the shortage of training data. Experiments across various tasks, including referring, grounding, and language-based object detection, show that ROD-MLLM achieves state-of-the-art performance among MLLMs. Notably, in language-based object detection, our model achieves a +13.7 mAP improvement over existing MLLMs and surpasses most specialized detection models, especially in scenarios requiring advanced complex language understanding.

</details>

---

## 79. Flexible Frame Selection for Efficient Video Reasoning

- [ ] Flexible Frame Selection for Efficient Video Reasoning | https://cvpr.thecvf.com/virtual/2025/poster/32897

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32897

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video-language models have shown promise for addressing a range of multimodal tasks for video understanding, such as video question-answering. However, the inherent computational challenges of processing long video data and increasing model sizes have led to standard approaches that are limited by the number of frames they can process. In this work, we propose the Flexible Frame Selector (FFS), a learnable policy model with a new flexible selection operation, that helps alleviate input context restrictions by enabling video-language models to focus on the most informative frames for the downstream multimodal task, without adding undue processing cost. Our method differentiates from prior work due to its learnability, efficiency, and flexibility. We verify the efficacy of our method on standard video-question answering and reasoning benchmarks, and observe that our model can improve base video-language model accuracy while reducing the number of downstream processed frames.

</details>

---

## 80. SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding

- [ ] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding | https://cvpr.thecvf.com/virtual/2025/poster/32903

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32903

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D Visual Grounding (3DVG) aims to locate objects in 3D scenes based on textual descriptions, essential for applications like augmented reality and robotics. Traditional 3DVG approaches rely on annotated 3D datasets and predefined object categories, limiting scalability and adaptability. To overcome these limitations, we introduce SeeGround, a zero-shot 3DVG framework leveraging 2D Vision-Language Models (VLMs) trained on large-scale 2D data. SeeGround represents 3D scenes as a hybrid of query-aligned rendered images and spatially enriched text descriptions, bridging the gap between 3D data and 2D-VLMs input formats. We propose two modules: the Perspective Adaptation Module, which dynamically selects viewpoints for query-relevant image rendering, and the Fusion Alignment Module, which integrates 2D images with 3D spatial descriptions to enhance object localization. Extensive experiments on ScanRefer and Nr3D demonstrate that our approach outperforms existing zero-shot methods by large margins. Notably, we exceed weakly supervised methods and rival some fully supervised ones, outperforming previous SOTA by 7.7\% on ScanRefer and 7.1\% on Nr3D, showcasing its effectiveness in complex 3DVG task.

</details>

---

## 81. Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification

- [ ] Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification | https://cvpr.thecvf.com/virtual/2025/poster/32928

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32928

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-image person re-identification (ReID) aims to retrieve the images of an interested person based on textual descriptions. One main challenge for this task is the high cost in manually annotating large-scale databases, which affects the generalization ability of ReID models. Recent works handle this problem by leveraging Multi-modal Large Language Models (MLLMs) to describe pedestrian images automatically. However, the captions produced by MLLMs lack diversity in description styles. To address this issue, we propose a Human Annotator Modeling (HAM) approach to enable MLLMs to mimic the description styles of thousands of human annotators. Specifically, we first extract style features from human textual descriptions and perform clustering on them. This allows us to group textual descriptions with similar styles into the same cluster. Then, we employ a prompt to represent each of these clusters and apply prompt learning to mimic the description styles of different human annotators. Furthermore, we define a style feature space and perform uniform sampling in this space to obtain more diverse clustering prototypes, which further enriches the diversity of the MLLM-generated captions. Finally, we adopt HAM to automatically annotate a massive-scale database for text-to-image ReID. Extensive experiments on this database demonstrate that it significantly improves the generalization ability of ReID models. Code of this paper will be released.

</details>

---

## 82. Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents

- [ ] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents | https://cvpr.thecvf.com/virtual/2025/poster/32935

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32935

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have sparked significant interest in developing agents capable of mobile operating system (mobile OS) navigation. We introduce MONDAY (Mobile OS Navigation Task Dataset for Agents from YouTube), a large-scale dataset of 313K annotated frames from 20K instructional videos capturing diverse real-world mobile OS navigation across multiple platforms. Models trained on MONDAY demonstrate robust cross-platform generalization capabilities, consistently outperforming models trained on existing single OS datasets while achieving 21.41\%p better performance on previously unseen mobile OS configurations. To enable continuous dataset expansion as mobile platforms evolve, we present an automated framework that leverages publicly available video content to create comprehensive task datasets without manual annotation. Our framework combines robust OCR-based scene detection (95.04% F1-score), near-perfect UI component detection (99.87% hit ratio), and novel multi-step action identification to extract reliable action sequences across diverse interface configurations. We contribute both the MONDAY dataset and our automated collection framework to facilitate future research in mobile OS navigation.

</details>

---

## 83. Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks

- [ ] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks | https://cvpr.thecvf.com/virtual/2025/poster/32941

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32941

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Dynamic image degradations, including noise, blur and lighting inconsistencies, pose significant challenges in image restoration, often due to sensor limitations or adverse environmental conditions. Existing Deep Unfolding Networks (DUNs) offer stable restoration performance but require manual selection of degradation matrices for each degradation type, limiting their adaptability across diverse scenarios.To address this issue, we propose the Vision-Language-guided Unfolding Network (VLU-Net), a unified DUN framework for handling multiple degradation types simultaneously.VLU-Net leverages a Vision-Language Model (VLM) refined on degraded image-text pairs to align image features with degradation descriptions, selecting the appropriate transform for target degradation.By integrating an automatic VLM-based gradient estimation strategy into the Proximal Gradient Descent (PGD) algorithm, VLU-Net effectively tackles complex multi-degradation restoration tasks while maintaining interpretability. Furthermore, we design a hierarchical feature unfolding structure to enhance VLU-Net framework, efficiently synthesizing degradation patterns across various levels.VLU-Net is the first all-in-one DUN framework and outperforms current leading one-by-one and all-in-one end-to-end methods by 3.74 dB on the SOTS dehazing dataset and 1.70 dB on the Rain100L deraining dataset.

</details>

---

## 84. GOAL: Global-local Object Alignment Learning

- [ ] GOAL: Global-local Object Alignment Learning | https://cvpr.thecvf.com/virtual/2025/poster/32938

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32938

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models like CLIP have shown impressive capabilities in aligning images and text, but they often struggle with lengthy, detailed text descriptions due to their training focus on concise captions. We present GOAL (Global-local Object Alignment Learning), a novel fine-tuning method that enhances CLIP's ability to handle lengthy text by leveraging both global and local semantic alignments. Our approach consists of two key components: Local Image-Sentence Matching (LISM), which identifies corresponding pairs between image segments and descriptive sentences, and Token Similarity-based Learning (TSL), which efficiently propagates local element attention through these matched pairs. Evaluating GOAL on three new benchmarks for image-lengthy text retrieval, we demonstrate significant improvements over baseline CLIP fine-tuning, establishing a simple yet effective approach for adapting CLIP to detailed textual descriptions. Through extensive experiments, we show that our method's focus on local semantic alignment alongside global context leads to more nuanced and representative embeddings, particularly beneficial for tasks requiring fine-grained understanding of lengthy text descriptions.

</details>

---

## 85. MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation

- [ ] MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation | https://cvpr.thecvf.com/virtual/2025/poster/32943

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32943

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating high-fidelity 3D content from text prompts remains a significant challenge in computer vision due to the limited size, diversity, and annotation depth of the existing datasets. To address this, we introduce MARVEL-$40$M+, an extensive dataset with $40$ million text annotations for over $8.9$ million 3D assets aggregated from seven major 3D datasets. Our contribution is a novel multi-stage annotation pipeline that integrates open-source pretrained multi-view VLMs and LLMs to automatically produce multi-level descriptions, ranging from detailed ($150$-$200$ words) to concise semantic tags ($10$-$20$ words). This structure supports both fine-grained 3D reconstruction and rapid prototyping. Furthermore, we incorporate human metadata from source datasets into our annotation pipeline to add domain-specific information in our annotation and reduce VLM hallucinations. Additionally, we develop MARVEL-FX3D, a two-stage text-to-3D pipeline. We fine-tune Stable Diffusion with our annotations and use a pretrained image-to-3D network to generate 3D textured meshes within 15s.  Extensive evaluations show that MARVEL-40M+ significantly outperforms existing datasets in annotation quality and linguistic diversity, achieving win rates of $72.41\%$ by GPT-4 and $73.40\%$ by human evaluators.

</details>

---

## 86. VLMs-Guided Representation Distillation for Efficient Vision-Based Reinforcement Learning

- [ ] VLMs-Guided Representation Distillation for Efficient Vision-Based Reinforcement Learning | https://cvpr.thecvf.com/virtual/2025/poster/32944

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32944

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-based Reinforcement Learning (VRL) attempts to establish associations between visual inputs and optimal actions through interactions with the environment. Given the high-dimensional and complex nature of visual data, it becomes essential to learn policy upon high-quality state representation. To this end, existing VRL methods primarily rely on interaction-collected data, combined with self-supervised auxiliary tasks. However, two key challenges remain: limited data samples and a lack of task-relevant semantic constraints. To tackle this, we propose \textbf{DGC}, a method that \textbf{d}istills \textbf{g}uidance from Visual Language Models (VLMs) alongside self-supervised learning into a \textbf{c}ompact VRL agent. Notably, we leverage the state representation capabilities of VLMs, rather than their decision-making abilities. Within DGC, a novel prompting-reasoning pipeline is designed to convert historical observations and actions into usable supervision signals, enabling semantic understanding within the compact visual encoder. By leveraging these distilled semantic representations, the VRL agent achieves significant improvements in the sample efficiency. Extensive experiments on the Carla benchmark demonstrate our state-of-the-art performance. The source code is available in the supplementary material.

</details>

---

## 87. Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts

- [ ] Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts | https://cvpr.thecvf.com/virtual/2025/poster/32947

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32947

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Model editing aims to correct inaccurate knowledge, update outdated information, and incorporate new data into Large Language Models (LLMs) without the need for retraining. This task poses challenges in lifelong scenarios where edits must be continuously applied for real-world applications. While some editors demonstrate strong robustness for lifelong editing in pure LLMs, Vision LLMs (VLLMs), which incorporate an additional vision modality, are not directly adaptable to existing LLM editors. In this paper, we propose LiveEdit, a lifelong vision language model edit to bridge the gap between lifelong LLM editing and VLLMs. We begin by training an editing expert generator to independently produce low-rank experts for each editing instance, with the goal of correcting the relevant responses of the VLLM. A hard filtering mechanism is developed to utilize visual semantic knowledge, thereby coarsely eliminating visually irrelevant experts for input queries during the inference stage of the post-edited model. Finally, to integrate visually relevant experts, we introduce a soft routing mechanism based on textual semantic relevance to achieve multi-expert fusion. For evaluation, we establish a benchmark for lifelong VLLM editing. Extensive experiments demonstrate that LiveEdit offers significant advantages in lifelong VLLM editing scenarios. Further experiments validate the rationality and effectiveness of each module design in LiveEdit.

</details>

---

## 88. DH-Set: Improving Vision-Language Alignment with Diverse and Hybrid Set-Embeddings Learning

- [ ] DH-Set: Improving Vision-Language Alignment with Diverse and Hybrid Set-Embeddings Learning | https://cvpr.thecvf.com/virtual/2025/poster/32951

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32951

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language (VL) alignment across image and text modalities is a challenging task due to the inherent semantic ambiguity of data with multiple possible meanings. Existing methods typically solve it by learning multiple sub-representation spaces to encode each input data as a set of embeddings, and constraining diversity between whole subspaces to capture diverse semantics for accurate VL alignment. Despite their promising outcomes, existing methods suffer two imperfections: 1) actually, specific semantics is mainly expressed by some local dimensions within the subspace. Ignoring this intrinsic property, existing diversity constraints imposed on the whole subspace may impair diverse embedding learning; 2) multiple embeddings are inevitably introduced, sacrificing computational and storage efficiency. In this paper, we propose a simple yet effective Diverse and Hybrid Set-embeddings learning framework (DH-Set), which is distinct from prior work in three aspects. DH-Set 1) devises a novel semantic importance dissecting method to focus on key local dimensions within each subspace; and thereby 2) not only imposes finer-grained diversity constraint to improve the accuracy of diverse embedding learning, 3) but also mixes key dimensions of all subspaces into the single hybrid embedding to boost inference efficiency. Extensive experiments on various benchmarks and model backbones show the superiority of DH-Set over state-of-the-art methods, achieving substantial 2.3%-14.7% rSum improvements while lowering computational and storage complexity. Codes will be released.

</details>

---

## 89. Bringing CLIP to the Clinic: Dynamic Soft Labels and Negation-Aware Learning for Medical Analysis

- [ ] Bringing CLIP to the Clinic: Dynamic Soft Labels and Negation-Aware Learning for Medical Analysis | https://cvpr.thecvf.com/virtual/2025/poster/32959

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32959

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of large-scale image-text pair datasets has significantly advanced self-supervised learning in Vision-Language Processing (VLP). However, directly applying general-domain architectures such as CLIP to medical data presents challenges, particularly in handling negations and addressing the inherent data imbalance of medical datasets. To address these issues, we propose a novel approach that integrates clinically-enhanced dynamic soft labels and medical graphical alignment, thereby improving clinical comprehension and improving the applicability of contrastive loss in medical contexts. Furthermore, we introduce negation-based hard negatives to deepen the model’s understanding of the complexities of clinical language. Our approach integrates seamlessly into any medical CLIP training pipeline and achieves state-of-the-art performance across multiple tasks, including zero-shot, fine-tuned classification and report retrieval. To further assess our model’s capacity for clinical language comprehension, we introduce CXR-Align, a benchmark uniquely designed to evaluate the understanding of negation and clinical information within chest X-ray (CXR) datasets. Experimental results demonstrate that our proposed methods are straightforward to implement and generalize effectively across contrastive learning frameworks, enhancing medical VLP capabilities and advancing clinical language understanding in medical imaging.

</details>

---

## 90. Recurrence-Enhanced Vision-and-Language Transformers for Robust Multimodal Document Retrieval

- [ ] Recurrence-Enhanced Vision-and-Language Transformers for Robust Multimodal Document Retrieval | https://cvpr.thecvf.com/virtual/2025/poster/32962

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32962

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Cross-modal retrieval is gaining increasing efficacy and interest from the research community, thanks to large-scale training, novel architectural and learning designs, and its application in LLMs and multimodal LLMs. In this paper, we move a step forward and design an approach that allows for multimodal queries -- composed of both an image and a text -- and can search within collections of multimodal documents, where images and text are interleaved. Our model, ReT, employs multi-level representations extracted from different layers of both visual and textual backbones, both at the query and document side. To allow for multi-level and cross-modal understanding and feature extraction, ReT employs a novel Transformer-based recurrent cell that integrates both textual and visual features at different layers, and leverages sigmoidal gates inspired by the classical design of LSTMs. Extensive experiments on M2KR and M-BEIR benchmarks show that Ret achieves state-of-the-art performance across diverse settings.

</details>

---

## 91. DTOS: Dynamic Time Object Sensing with Large Multimodal Model

- [ ] DTOS: Dynamic Time Object Sensing with Large Multimodal Model | https://cvpr.thecvf.com/virtual/2025/poster/32970

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32970

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing multimodal large language models (MLLM) face significant challenges in Referring Video Object Segmentation(RVOS). We identify three critical challenges: (C1) insufficient quantitative representation of textual numerical data, (C2) repetitive and degraded response templates for spatiotemporal referencing, and (C3) loss of visual information in video sampling queries lacking textual guidance. To address these, we propose a novel framework, \textbf{Dynamic Time Object Sensing (DTOS)}, specifically designed for RVOS. To tackle (C1) and (C2), we introduce specialized tokens to construct multi-answer response templates, enabling regression of event boundaries and target localization. This approach improves the accuracy of numerical regression while mitigating the issue of repetitive degradation. To address (C3), we propose a Text-guided Clip Sampler (TCS) that selects video clips aligned with user instructions, preventing visual information loss and ensuring consistent temporal resolution. TCS is also applicable to Moment Retrieval tasks, with enhanced multimodal input sequences preserving spatial details and maximizing temporal resolution. DTOS demonstrates exceptional capability in flexibly localizing multiple spatiotemporal targets based on user-provided textual instructions. Extensive experiments validate the effectiveness of our approach, with DTOS achieving state-of-the-art performance in J&F scores: an improvement of +4.36 on MeViS, +4.48 on Ref-DAVIS17, and +3.02 on Ref-YT-VOS. Additionally, our TCS demonstrates exceptional performance in Moment Retrieval. All code, models, and datasets will be made publicly available.

</details>

---

## 92. MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research

- [ ] MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research | https://cvpr.thecvf.com/virtual/2025/poster/32974

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32974

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scientific research demands sophisticated reasoning over multimodal data, a challenge especially prevalent in biology. Despite recent advances in multimodal large language models (MLLMs) for AI-assisted research, existing multimodal reasoning benchmarks target up to college level difficulty, while research-level benchmarks emphasize lower-level perception, falling short of the complex multimodal reasoning needed for scientific discovery. To bridge this gap, we introduce MicroVQA, a visual-question answering (VQA) benchmark designed to assess three reasoning capabilities vital in research workflows: expert image understanding, hypothesis generation, and experiment proposal. MicroVQA consists of 1,061 multiple-choice questions (MCQs) curated by biological experts across diverse microscopy modalities, ensuring VQA samples represent real scientific practice. We find that standard MCQ creation methods do not properly test our targeted reasoning capabilities, motivating a new two stage pipeline: an optimized LLM prompt structures question-answer pairs into MCQs; then, an agent-based `RefineBot' generates more challenging distractors. Benchmarking on state-of-the-art MLLMs reveal a peak performance of 43%; models with smaller LLMs only slightly underperform top models, suggesting that language-based reasoning is less challenging than multimodal reasoning; and tuning with scientific articles enhances performance. Expert analysis of chain-of-thought reasoning failures indicates that multimodal reasoning errors are frequent, followed by knowledge errors and overgeneralization. These insights highlight the challenges in multimodal scientific reasoning, showing MicroVQA is a valuable resource advancing AI-driven biomedical research.

</details>

---

## 93. Effective SAM Combination for Open-Vocabulary Semantic Segmentation

- [ ] Effective SAM Combination for Open-Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/32975

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32975

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation aims to assign pixel-level labels to images across an unlimited range of classes. Traditional methods address this by sequentially connecting a powerful mask proposal generator, such as the Segment Anything Model (SAM), with a pre-trained vision-language model like CLIP. But these two-stage approaches often suffer from high computational costs, memory inefficiencies. In this paper, we propose ESC-Net, a novel one-stage open-vocabulary segmentation model that leverages the SAM decoder blocks for class-agnostic segmentation within an efficient inference framework. By embedding pseudo prompts generated from image-text correlations into SAM’s promptable segmentation framework, ESC-Net achieves refined spatial aggregation for accurate mask predictions. Additionally, a Vision-Language Fusion (VLF) module enhances the final mask prediction through image and text guidance. ESC-Net achieves superior performance on standard benchmarks, including ADE20K, PASCAL-VOC, and PASCAL-Context, outperforming prior methods in both efficiency and accuracy. Comprehensive ablation studies further demonstrate its robustness across challenging conditions.

</details>

---

## 94. Hierarchical Knowledge Prompt Tuning for Multi-task Test-Time Adaptation

- [ ] Hierarchical Knowledge Prompt Tuning for Multi-task Test-Time Adaptation | https://cvpr.thecvf.com/virtual/2025/poster/32985

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32985

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation using vision-language model (such as CLIP) to quickly adjust to distributional shifts of downstream tasks has shown great potential. Despite significant progress, existing methods are still limited to single-task test-time adaptation scenarios and have not effectively explored the issue of multi-task adaptation. To address this practical problem, we propose a novel Hierarchical Knowledge Prompt Tuning (HKPT) method, which achieves joint adaptation to multiple target domains by mining more comprehensive source domain discriminative knowledge and hierarchically modeling task-specific and task-shared knowledge. Specifically, HKPT constructs a CLIP prompt distillation framework that utilizes the broader source domain knowledge of large teacher CLIP to guide prompt tuning for lightweight student CLIP from multiple views during testing. Meanwhile, HKPT establishes task-specific dual dynamic knowledge graph to capture fine-grained contextual knowledge from continuous test data. And to fully exploit the complementarity among multiple target tasks, HKPT employs an adaptive task grouping strategy for achieving inter-task knowledge sharing. Furthermore, HKPT can seamlessly transfer to basic single-task test-time adaptation scenarios while maintaining robust performance. Extensive experimental results in both multi-task and single-task test-time adaptation settings demonstrate that our HKPT significantly outperforms state-of-the-art methods.

</details>

---

## 95. DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models

- [ ] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32994

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32994

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Base-New Trade-off (BNT) problem universally exists during the optimization of CLIP-based prompt tuning, where continuous fine-tuning on base (target) classes leads to a simultaneous decrease of generalization ability on new (unseen) classes. Existing approaches attempt to regulate the prompt tuning process to balance BNT by appending constraints. However, imposed on the same target prompt, these constraints fail to fully avert the mutual exclusivity between the optimization directions for base and new. As a novel solution to this challenge, we propose the plug-and-play Dual-Prompt Collaboration (DPC) framework, the first that decoupling the optimization processes of base and new tasks at the prompt level. Specifically, we clone a learnable parallel prompt based on the backbone prompt, and introduce a variable Weighting-Decoupling framework to independently control the optimization directions of dual prompts specific to base or new tasks, thus avoiding the conflict in generalization. Meanwhile, we propose a Dynamic Hard Negative Optimizer, utilizing dual prompts to construct a more challenging optimization task on base classes for enhancement. For interpretability, we prove the feature channel invariance of the prompt vector during the optimization process, providing theoretical support for the Weighting-Decoupling of DPC. Extensive experiments on multiple backbones demonstrate that DPC can significantly improve base performance without introducing any external knowledge beyond the base classes, while maintaining generalization to new classes.

</details>

---

## 96. Taxonomy-Aware Evaluation of Vision-Language Models

- [ ] Taxonomy-Aware Evaluation of Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/32998

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32998

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

When a vision-and-language model (VLM) is prompted to identify an entity in an image, it may err on the side of caution and answer with "tree", instead of a more specific description such as "Pine tree''. Traditional binary accuracy metrics cannot differentiate between wrong predictions and insufficiently specific ones. They also do not give partial credit for close answers: "pine tree'' for a Norway Spruce should be better than "cypress'', taxonomically speaking, but string matching-based similarity measures will reject both equally.To address this shortcoming, we propose a framework for evaluating open-ended text predictions against a taxonomic hierarchy,using measures of hierarchical precision and recall to measure the level of correctness and specificity of predictions.We first show that existing text similarity measures and accuracy-based evaluation metrics do not capture taxonomic similarity well. We then develop and compare different methods to map textual VLM predictions onto a taxonomy. This allows us to compute hierarchical similarity measures between the free-form outputs and the ground truth labels.Finally, we analyze modern VLMs on fine-grained visual classification tasks based on our taxonomic evaluation. We find that models respond differently to instructions prompting for more specific answers, with GPT4V responding most specifically and others showing a trade-off between hierarchical precision and recall.

</details>

---

## 97. Stop Learning it all to Mitigate Visual Hallucination, Focus on the Hallucination Target.

- [ ] Stop Learning it all to Mitigate Visual Hallucination, Focus on the Hallucination Target. | https://cvpr.thecvf.com/virtual/2025/poster/32999

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/32999

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) frequently suffer from hallucination issues, generating information about objects that are not present in input images during vision-language tasks. These hallucinations particularly undermine model reliability in practical applications requiring accurate object identification. To address this challenge, we propose TL-DPO, a preference learning approach that mitigates hallucinations by focusing on targeted areas where they occur. To implement this, we build a dataset containing hallucinated responses, correct responses, and target information (i.e., objects present in the images and the corresponding chunk positions in responses affected by hallucinations). By applying a preference learning method restricted to these specific targets, the model can filter out irrelevant signals and focus on correcting hallucinations. This allows the model to produce more factual responses by concentrating solely on relevant information. Experimental results demonstrate that TL-DPO effectively reduces hallucinations across multiple vision hallucination tasks, improving the reliability and performance of MLLMs without diminishing overall performance.

</details>

---

## 98. Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects

- [ ] Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects | https://cvpr.thecvf.com/virtual/2025/poster/33003

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33003

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancements in text-to-3D generation necessitate robust and scalable evaluation metrics that align closely with human judgment—a need unmet by current metrics such as PSNR and CLIP, which require ground-truth data or focus only on prompt fidelity. To address this, we introduce Gen3DEval, a novel evaluation framework that leverages vision large language models (vLLMs) specifically fine-tuned for 3D object quality assessment. Gen3DEval evaluates text fidelity, appearance, and surface quality—by analyzing 3D surface normals—without requiring ground-truth comparisons, bridging the gap between automated metrics and user preferences.Compared to state-of-the-art task-agnostic models, Gen3DEval demonstrates superior performance in user-aligned evaluations, establishing itself as a comprehensive and accessible benchmark for future research in text-to-3D generation. To support and encourage further research in this field, we will release both our code and benchmark, establishing Gen3DEval as a comprehensive and accessible tool for text-to-3D evaluation.

</details>

---

## 99. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis

- [ ] Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis | https://cvpr.thecvf.com/virtual/2025/poster/33002

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33002

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the quest for artificial general intelligence, Multi-modal Large Language Models (MLLMs) have emerged as a focal point in recent advancements. However, the predominant focus remains on developing their capabilities in static image understanding. The potential of MLLMs to process sequential visual data is still insufficiently explored, highlighting the lack of a comprehensive, high-quality assessment of their performance. In this paper, we introduce Video-MME, the first-ever full-spectrum, Multi-Modal Evaluation benchmark of MLLMs in Video analysis. Our work distinguishes from existing benchmarks through four key features: 1) Diversity in video types, spanning 6 primary visual domains with 30 subfields to ensure broad scenario generalizability; 2) Duration in temporal dimension, encompassing both short-, medium-, and long-term videos, ranging from 11 seconds to 1 hour, for robust contextual dynamics; 3) Breadth in data modalities, integrating multi-modal inputs besides video frames, including subtitles and audios, to unveil the all-round capabilities of MLLMs; 4) Quality in annotations, utilizing rigorous manual labeling by expert annotators to facilitate precise and reliable model assessment. With Video-MME, we extensively evaluate various state-of-the-art MLLMs, and reveal that Gemini 1.5 Pro is the best-performing commercial model, significantly outperforming the open-source models with an average accuracy of 75\%, compared to 71.9% for GPT-4o. The results also demonstrate that Video-MME is a universal benchmark that applies to both image and video MLLMs. Further analysis indicates that subtitle and audio information could significantly enhance video understanding. Besides, a decline in MLLM performance is observed as video duration increases for all models. Our dataset along with these findings underscores the need for further improvements in handling longer sequences and multi-modal data, shedding light on future MLLM development.

</details>

---

## 100. EgoLife: Towards Egocentric Life Assistant

- [ ] EgoLife: Towards Egocentric Life Assistant | https://cvpr.thecvf.com/virtual/2025/poster/33009

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33009

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce EgoLife , a project to develop an egocentric life assistant that accompanies and enhances personal efficiency through AI-powered wearable glasses. To lay the foundation for this assistant, we conducted a comprehensive data collection study where six participants lived together for one week, continuously recording their daily activities—including discussions, shopping, cooking, socializing, and entertainment—using AI glasses for multimodal egocentric video capture, along with synchronized third-person-view video references. This effort resulted in the EgoLife Dataset , a comprehensive 300-hour egocentric, interpersonal, multiview, and multimodal daily life dataset with intensive annotation. Leveraging this dataset, we introduce EgoLifeQA, a suite of long-context, life-oriented question-answering tasks designed to provide meaningful assistance in daily life by addressing practical questions such as recalling past relevant events, monitoring health habits, and offering personalized recommendations.To address the key technical challenges of 1) developing robust visual-audio models for egocentric data, 2) enabling accurate identity recognition, and 3) facilitating long-context question answering over extensive temporal information, we introduce EgoButler , an integrated system comprising EgoGPT and EgoRAG . EgoGPT is a vision-language model trained on egocentric datasets, achieving state-of-the-art performance on egocentric video understanding. EgoRAG is a retrieval-based component that supports answering ultra-long-context questions. Our experimental studies verify their working mechanisms and reveal critical factors and bottlenecks, guiding future improvements. By releasing our datasets, models, and benchmarks, we aim to stimulate further research in egocentric AI assistants.

</details>

---

## 101. JTD-UAV: MLLM-Enhanced Joint Tracking and Description Framework for Anti-UAV Systems

- [ ] JTD-UAV: MLLM-Enhanced Joint Tracking and Description Framework for Anti-UAV Systems | https://cvpr.thecvf.com/virtual/2025/poster/33012

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33012

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unmanned Aerial Vehicles (UAVs) are widely adopted across various fields, yet they raise significant privacy and safety concerns, demanding robust monitoring solutions. Existing anti-UAV methods primarily focus on position tracking but fail to capture UAV behavior and intent. To address this, we introduce a novel task—UAV Tracking and Intent Understanding (UTIU)—which aims to track UAVs while inferring and describing their motion states and intent for a more comprehensive monitoring approach. To tackle the task, we propose JTD-UAV, the first joint tracking, and intent description framework based on large language models. Our dual-branch architecture integrates UAV tracking with Visual Question Answering (VQA), allowing simultaneous localization and behavior description. To benchmark this task, we introduce the TDUAV dataset, the largest dataset for joint UAV tracking and intent understanding, featuring 1,328 challenging video sequences, over 163K annotated thermal frames, and 3K VQA pairs. Our benchmark demonstrates the effectiveness of JTD-UAV, and both the dataset and code will be publicly available.

</details>

---

## 102. Motion-Grounded Video Reasoning: Understanding and Perceiving Motion at Pixel Level

- [ ] Motion-Grounded Video Reasoning: Understanding and Perceiving Motion at Pixel Level | https://cvpr.thecvf.com/virtual/2025/poster/33015

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33015

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce Motion-Grounded Video Reasoning, a new motionunderstanding task that requires generating visual answers (video segmentationmasks) according to the input question, and hence needs implicit spatiotemporalreasoning and grounding. This task extends existing spatiotemporal groundingwork focusing on explicit action/motion grounding, to a more general format byenabling implicit reasoning via questions. To facilitate the development of the newtask, we collect a large-scale dataset called GROUNDMORE, which comprises1,715 video clips, 249K object masks that are deliberately designed with 4 questiontypes (Causal, Sequential, Counterfactual, and Descriptive) for benchmarkingdeep and comprehensive motion reasoning abilities. GROUNDMORE uniquelyrequires models to generate visual answers, providing a more concrete and visuallyinterpretable response than plain texts. It evaluates models on both spatiotemporalgrounding and reasoning, fostering to address complex challenges in motion-relatedvideo reasoning, temporal perception, and pixel-level understanding. Furthermore,we introduce a novel baseline model named Motion-Grounded Video ReasoningAssistant (MORA). MORA incorporates the multimodal reasoning ability from theMultimodal LLM, the pixel-level perception capability from the grounding model(SAM), and the temporal perception ability from a lightweight localization head.MORA achieves respectable performance on GROUNDMORE outperforming thebest existing visual grounding baseline model by an average of 21.5% relatively.We hope this novel and challenging task will pave the way for future advancementsin robust and general motion understanding via video reasoning segmentation.

</details>

---

## 103. Omnia de EgoTempo: Benchmarking Temporal Understanding of Multi-Modal LLMs in Egocentric Videos

- [ ] Omnia de EgoTempo: Benchmarking Temporal Understanding of Multi-Modal LLMs in Egocentric Videos | https://cvpr.thecvf.com/virtual/2025/poster/33032

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33032

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding fine-grained temporal dynamics is crucial in egocentric videos, where continuous streams capture frequent, close-up interactions with objects. In this work, we bring to light that current egocentric video question-answering datasets often include questions that can be answered using only few frames or commonsense reasoning, without being necessarily grounded in the actual video. Our analysis shows that state-of-the-art Multi-Modal Large Language Models (MLLMs) on these benchmarks achieve remarkably high performance using just text or a single frame as input.To address these limitations, we introduce EgoTempo, a dataset specifically designed to evaluate temporal understanding in the egocentric domain. EgoTempo emphasizes tasks that require integrating information across the entire video, ensuring that models would need to rely on temporal patterns rather than static cues or pre-existing knowledge. Extensive experiments on EgoTempo show that current MLLMs still fall short in temporal reasoning on egocentric videos, and thus we hope EgoTempo will catalyze new research in the field and inspire models that better capture the complexity of temporal dynamics in egocentric settings.The dataset will be made publicly available upon acceptance.

</details>

---

## 104. DistinctAD: Distinctive Audio Description Generation in Contexts

- [ ] DistinctAD: Distinctive Audio Description Generation in Contexts | https://cvpr.thecvf.com/virtual/2025/poster/33034

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33034

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Audio Descriptions (ADs) aim to provide a narration of a movie in text form, describing non-dialogue-related narratives, such as characters, actions, or scene establishment. Automatic generation of ADs remains challenging due to: i) the domain gap between movie-AD data and existing data used to train vision-language models, and ii) the issue of contextual redundancy arising from highly similar neighboring visual clips in a long movie. In this work, we propose DistinctAD , a novel two-stage framework for generating ADs that emphasize distinctiveness to produce better narratives. To address the domain gap, we introduce a CLIP-AD adaptation strategy that does not require additional AD corpora, enabling more effective alignment between movie and AD modalities at both global and fine-grained levels. In Stage-II, DistinctAD incorporates two key innovations: (i) a Contextual Expectation-Maximization Attention (EMA) module that reduces redundancy by extracting common bases from consecutive video clips, and (ii) an explicit distinctive word prediction loss that filters out repeated words in the context, ensuring the prediction of unique terms specific to the current AD. Comprehensive evaluations on MAD-Eval, CMD-AD, and TV-AD benchmarks demonstrate the superiority of DistinctAD, with the model consistently outperforming baselines, particularly in Recall@k/N, highlighting its effectiveness in producing high-quality, distinctive ADs.

</details>

---

## 105. MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts

- [ ] MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts | https://cvpr.thecvf.com/virtual/2025/poster/33039

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33039

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown promising capabilities in mathematical reasoning within visual contexts across various datasets. However, most existing multimodal math benchmarks are limited to single-visual contexts, which diverges from the multi-visual scenarios commonly encountered in real-world mathematical applications. To address this gap, we introduce MV-MATH: a meticulously curated dataset of 2,009 high-quality mathematical problems. Each problem integrates multiple images interleaved with text, derived from authentic K-12 scenarios and enriched with detailed annotations. MV-MATH includes multiple-choice, free-form, and multi-step questions, covering 11 subject areas across 3 difficulty levels, and serves as a comprehensive and rigorous benchmark for assessing MLLMs’ mathematical reasoning in multi-visual contexts. Through extensive experimentation, we observe that MLLMs encounter substantial challenges in multi-visual math tasks, with a considerable performance gap relative to human capabilities on MV-MATH. Furthermore, we analyze the performance and error patterns of various models, providing insights into MLLMs' mathematical reasoning capabilities within multi-visual settings.

</details>

---

## 106. Video-Bench: Human-Aligned Video Generation Benchmark

- [ ] Video-Bench: Human-Aligned Video Generation Benchmark | https://cvpr.thecvf.com/virtual/2025/poster/33048

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33048

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video generation assessment is essential for ensuring that generative models produce visually realistic, high-quality videos while aligning with human expectations. Current video generation benchmarks fall into two main categories: traditional benchmarks, which use metrics and embeddings to evaluate generated video quality across multiple dimensions but often lack alignment with human judgments; and large language model (LLM)-based benchmarks, though capable of human-like reasoning, are constrained by a limited understanding of video quality metrics and cross-modal consistency.To address these challenges and establish a benchmark that better aligns with human preferences, this paper introduces HA-Video-Bench, a comprehensive benchmark featuring a rich prompt suite and extensive evaluation dimensions. This benchmark represents the first attempt to systematically leverage MLLMs across all dimensions relevant to video generation assessment in generative models. By incorporating few-shot scoring and chain-of-query techniques, HA-Video-Bench provides a structured, scalable approach to generated video evaluation. Experimental results demonstrate that MLLMs achieve superior alignment with human preferences across all dimensions. Moreover, in instances where our framework’s assessments diverge from human evaluations, it consistently offers more objective and accurate insights, suggesting an even greater potential advantage over traditional human judgment.

</details>

---

## 107. PromptHMR: Promptable Human Mesh Recovery

- [ ] PromptHMR: Promptable Human Mesh Recovery | https://cvpr.thecvf.com/virtual/2025/poster/33049

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33049

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human pose and shape (HPS) estimation presents challenges in diverse scenarios such as crowded scenes, person-person interactions, and single-view reconstruction. Existing approaches lack mechanisms to incorporate auxiliary ``side information" that could enhance reconstruction accuracy in such challenging scenarios. Furthermore, the most accurate methods rely on cropped person detections and cannot exploit scene context while methods that process the whole image often fail to detect people and are less accurate than methods that use crops. While recent language-based methods explore HPS reasoning through large language or vision-language models, their metric accuracy is well below the state of the art. In contrast, we present PromptHMR, a transformer-based promptable method that reformulates HPS estimation through spatial and semantic prompts. Our method processes full images to maintain scene context and accepts multiple input modalities: spatial prompts like face or body bounding boxes, and semantic prompts like language descriptions or interaction labels. PromptHMR demonstrates robust performance across challenging scenarios: estimating people from bounding boxes as small as faces in crowded scenes, improving body shape estimation through language descriptions, modeling person-person interactions, and producing temporally coherent motions in videos. Experiments on benchmarks show that PromptHMR achieves state-of-the-art performance while offering flexible prompt-based control over the HPS estimation process.

</details>

---

## 108. Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought

- [ ] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought | https://cvpr.thecvf.com/virtual/2025/poster/33063

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33063

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in multimodal large language models (MLLMs) have demonstrated remarkable capabilities in vision-language tasks, yet they often struggle with vision-centric scenarios where precise visual focus is needed for accurate reasoning. In this paper, we introduce Argus to address these limitations with a new visual attention grounding mechanism. Our approach employs object-centric grounding as visual chain-of-thought signals, enabling more effective goal-conditioned visual attention during multimodal reasoning tasks. Evaluations on diverse benchmarks demonstrate that Argus excels in both multimodal reasoning tasks and referring object grounding tasks. Extensive analysis further validates various design choices of Argus, and reveals the effectiveness of explicit language-guided visual region-of-interest engagement in MLLMs, highlighting the importance of advancing multimodal intelligence from a visual-centric perspective.

</details>

---

## 109. Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment

- [ ] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment | https://cvpr.thecvf.com/virtual/2025/poster/33068

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33068

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current multimodal large language models (MLLMs) struggle with fine-grained or precise understanding of visuals though they give comprehensive perception and reasoning in a spectrum of vision applications. Recent studies either develop tool-using or unify specific visual tasks into the autoregressive framework, often at the expense of overall multimodal performance. To address this issue and enhance MLLMs with visual tasks in a scalable fashion, we propose Task Preference Optimization (TPO), a novel method that utilizes differentiable task preferences derived from typical fine-grained visual tasks. TPO introduces learnable task tokens that establish connections between multiple task-specific heads and the MLLM. By leveraging rich visual labels during training, TPO significantly enhances the MLLM's multimodal capabilities and task-specific performance. Through multi-task co-training within TPO, we observe synergistic benefits that elevate individual task performance beyond what is achievable through single-task training methodologies. Our instantiation of this approach with VideoChat and LLaVA demonstrates an overall 14.6\% improvement in multimodal performance compared to baseline models. Additionally, MLLM-TPO demonstrates robust zero-shot capabilities across various tasks, performing comparably to state-of-the-art supervised models.

</details>

---

## 110. Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models

- [ ] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33073

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33073

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Today's most advanced vision-language models (VLMs) remain proprietary. The strongest open-weight models rely heavily on synthetic data from proprietary VLMs to achieve good performance, effectively distilling these closed VLMs into open ones. As a result, the community has been missing foundational knowledge about how to build performant VLMs from scratch. We present \textbf{Molmo}, a new family of VLMs that are state-of-the-art in their class of openness.  Our key contribution is a collection of new datasets, including a dataset of highly detailed image captions for pre-training called \textbf{PixMo}, a free-form image Q\&A dataset for fine-tuning, and an innovative 2D pointing dataset, all collected without the use of external VLMs. The success of our approach relies on careful modeling choices, a well-tuned training pipeline, and, most critically, the quality of our newly collected datasets. Our best-in-class 72B model not only outperforms others in the class of open weight and data models, but also outperforms larger proprietary models including Claude 3.5 Sonnet, and Gemini 1.5 Pro and Flash, second only to GPT-4o based on both academic benchmarks and  on a large human evaluation. Our model weights, new datasets, and source code will all be released.

</details>

---

## 111. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning

- [ ] R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning | https://cvpr.thecvf.com/virtual/2025/poster/33087

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33087

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional visual models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and is often difficult to generalize across tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in supplementary materials.

</details>

---

## 112. Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body

- [ ] Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body | https://cvpr.thecvf.com/virtual/2025/poster/33097

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33097

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent improvements in visual synthesis have significantly enhanced the depiction of generated human photos, which are pivotal due to their wide applicability and demand. Nonetheless, the existing text-to-image or text-to-video models often generate low-quality human photos that might differ considerably from real-world body structures, referred to as ``abnormal human bodies''. Such abnormalities, typically deemed unacceptable, pose considerable challenges in the detection and repair of them within human photos. These challenges require precise abnormality recognition capabilities, which entail pinpointing both the location and the abnormality type. Intuitively, Visual Language Models (VLMs) that have obtained remarkable performance on various visual tasks are quite suitable for this task. However, their performance on abnormality detection in human photos is quite poor.Hence, it is quite important to highlight this task for the research community. In this paper, we first introduce a simple yet challenging task, i.e., \textbf{F}ine-grained \textbf{H}uman-body \textbf{A}bnormality \textbf{D}etection \textbf{(FHAD)}, and construct two high-quality datasets for evaluation. Then, we propose a meticulous framework, named HumanCalibrator, which identifies and repairs abnormalities in human body structures while preserving the other content. Experiments indicate that our HumanCalibrator achieves high accuracy in abnormality detection and accomplishes an increase in visual comparisons while preserving the other visual content.

</details>

---

## 113. Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding

- [ ] Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding | https://cvpr.thecvf.com/virtual/2025/poster/33098

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33098

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual grounding seeks to localize the image region corresponding to a free-form text description. Recently, the strong multimodal capabilities of Large Vision-Language Models (LVLMs) have driven substantial improvements in visual grounding, though they inevitably require fine-tuning and additional model components to explicitly generate bounding boxes or segmentation masks. However, we discover that a few attention heads in frozen LVLMs demonstrate strong visual grounding capabilities. We refer to these heads, which consistently capture object locations related to text semantics, as localization heads. Using localization heads, we introduce a straightforward and effective training-free visual grounding framework that utilizes text-to-image attention maps from localization heads to identify the target objects. Surprisingly, only three out of thousands of attention heads are sufficient to achieve competitive localization performance compared to existing LVLM-based visual grounding methods that require fine-tuning. Our findings suggest that LVLMs can innately ground objects based on a deep comprehension of the text-image relationship, as they implicitly focus on relevant image regions to generate informative text outputs. All the source codes will be made available to the public.

</details>

---

## 114. Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning

- [ ] Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning | https://cvpr.thecvf.com/virtual/2025/poster/33105

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33105

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper proposes the first video-grounded entailment tree reasoning method for commonsense video question answering (VQA). Despite the remarkable progress of large visual-language models (VLMs), there are growing concerns that they learn spurious correlations between videos and likely answers, reinforced by their black-box nature and remaining benchmarking biases. Our method explicitly grounds VQA tasks to video fragments in four steps: entailment tree construction, video-language entailment verification, tree reasoning, and dynamic tree expansion. A vital benefit of the method is its generalizability to current video- and image-based VLMs across reasoning types.To support fair evaluation, we devise a de-biasing procedure based on large-language models that rewrite VQA benchmark answer sets to enforce model reasoning. Systematic experiments on existing and de-biased benchmarks highlight the impact of our method components across benchmarks, VLMs, and reasoning types.

</details>

---

## 115. Accelerating Multimodal Large Language Models by Searching Optimal Vision Token Reduction

- [ ] Accelerating Multimodal Large Language Models by Searching Optimal Vision Token Reduction | https://cvpr.thecvf.com/virtual/2025/poster/33104

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33104

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prevailing Multimodal Large Language Models (MLLMs) encode the input image(s) as vision tokens and feed them into the language backbone, similar to how Large Language Models (LLMs) process the text tokens. However, the number of vision tokens increases quadratically as the image resolutions, leading to huge computational costs.In this paper, we consider improving MLLM's efficiency from two scenarios, (I) Reducing computational cost without degrading the performance. (II) Improving the performance with given budgets. We start with our main finding that the ranking of each vision token sorted by attention scores is similar in each layer except the first layer. Based on it, we assume that the number of essential top vision tokens does not increase along layers. Accordingly, for Scenario I, we propose a greedy search algorithm (G-Search) to find the least number of vision tokens to keep at each layer from the shallow to the deep. Interestingly, G-Search is able to reach the optimal reduction strategy based on our assumption. For Scenario II, based on the reduction strategy from G-Search, we design a parametric sigmoid function (P-Sigmoid) to guide the reduction at each layer of the MLLM, whose parameters are optimized by Bayesian Optimization. Extensive experiments demonstrate that our approach can significantly accelerate those popular MLLMs, e.g. LLaVA, and InternVL2 models, by more than $2 \times$ without performance drops. Our approach also far outperforms other token reduction methods when budgets are limited, achieving a better trade-off between efficiency and effectiveness.

</details>

---

## 116. Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model

- [ ] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model | https://cvpr.thecvf.com/virtual/2025/poster/33110

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33110

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Periodic or quasi-periodic phenomena reveal intrinsic characteristics in various natural processes, such as weather patterns, movement behaviors, traffic flows, and biological signals. Given that these phenomena span multiple modalities, the capabilities of Multimodal Large Language Models (MLLMs) offer promising potential to effectively capture and understand their complex nature. However, current MLLMs struggle with periodic tasks due to limitations in: 1) lack of temporal modelling and 2) conflict between short and long periods. This paper introduces Period-LLM, a multimodal large language model designed to enhance the performance of periodic tasks across various modalities, and constructs a benchmark of various difficulty for evaluating the cross-modal periodic capabilities of large models. Specially, We adopt an ``Easy to Hard Generalization" paradigm, starting with relatively simple text-based tasks and progressing to more complex visual and multimodal tasks, ensuring that the model gradually builds robust periodic reasoning capabilities. Additionally, we propose a Resisting Logical Oblivion optimization strategy to maintain periodic reasoning abilities during semantic alignment. Extensive experiments demonstrate the superiority of the proposed Period-LLM over existing MLLMs in periodic tasks. The code will be available on GitHub.

</details>

---

## 117. LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models

- [ ] LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33117

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33117

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Multimodal Large Language Models (MLLMs) excel at generalizing across modalities and tasks, effectively adapting them to specific downstream tasks while simultaneously retaining both general and specialized knowledge remains challenging. Although Low-Rank Adaptation (LoRA) is widely used to efficiently acquire specialized knowledge in MLLMs, it introduces substantial harmful redundancy during visual instruction tuning, which exacerbates the forgetting of general knowledge and degrades downstream task performance.To address this issue, we propose LoRASculpt to eliminate harmful redundant parameters, thereby harmonizing general and specialized knowledge.Specifically, under theoretical guarantees, we introduce sparse updates into LoRA to discard redundant parameters effectively. Furthermore, we propose a Conflict Mitigation Regularizer to refine the update trajectory of LoRA, mitigating knowledge conflicts with the pretrained weights.Extensive experimental results demonstrate that even at very high degree of sparsity ($\le$ 5\%), our method simultaneously enhances generalization and downstream task performance. This confirms that our approach effectively mitigates the catastrophic forgetting issue and further promotes knowledge harmonization in MLLMs.

</details>

---

## 118. STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding

- [ ] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33123

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33123

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained on tremendous image-text pairs, vision-language models like CLIP have demonstrated promising zero-shot generalization across numerous image-based tasks. However, extending these capabilities to video tasks remains challenging due to limited labeled video data and high training costs. Recent video prompting methods attempt to adapt CLIP for video tasks by introducing learnable prompts, but they typically rely on a single static prompt for all video sequences, overlooking the diverse temporal dynamics and spatial variations that exist across frames. This limitation significantly hinders the model’s ability to capture essential temporal information for effective video understanding. To address this, we propose an integrated Spatial-TempOral dynamic Prompting (STOP) model which consists of two complementary modules, the intra-frame spatial prompting and inter-frame temporal prompting. Our intra-frame spatial prompts are designed to adaptively highlight discriminative regions within each frame by leveraging intra-frame attention and temporal variation, allowing the model to focus on areas with substantial temporal dynamics and capture fine-grained spatial details. Additionally, to highlight the varying importance of frames for video understanding, we further introduce inter-frame temporal prompts, dynamically inserting prompts between frames with high temporal variance as measured by frame similarity. This enables the model to prioritize key frames and enhances its capacity to understand temporal dependencies across sequences. Extensive experiments on various video benchmarks demonstrate that STOP consistently achieves superior performance against state-of-the-art methods. Our code will be released soon.

</details>

---

## 119. Scaling Vision Pre-Training to 4K Resolution

- [ ] Scaling Vision Pre-Training to 4K Resolution | https://cvpr.thecvf.com/virtual/2025/poster/33120

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33120

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-resolution perception of visual details is crucial for daily tasks. Current vision pre-training, however, is still limited to low resolutions (e.g., 384x384) due to the quadratic cost of processing larger images. We introduce PS3, for Pre-training with Scale-Selective Scaling, that scales CLIP-style vision pre-training to 4K resolution with a near-constant cost. Instead of processing entire global images, PS3 is pre-trained to selectively process local regions and contrast them with local detailed captions, allowing it to learn detailed representation at high resolution with greatly reduced computational overhead. The pre-trained PS3 is able to both encode the global low-resolution image and select local high-resolution regions to process based on their saliency or relevance to a text prompt. When applied to multi-modal LLMs (MLLMs), PS3 demonstrates performance that effectively scales with the pre-training resolution and significantly improves over baselines without high-resolution pre-training. We also find current benchmarks do not require recognizing details at 4K resolution, which motivates us to propose 4KPro, a new benchmark that evaluates visual perception at 4K resolution, on which PS3 outperforms state-of-the-art MLLMs, including a 13% improvement over GPT-4o.

</details>

---

## 120. PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset

- [ ] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset | https://cvpr.thecvf.com/virtual/2025/poster/33126

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33126

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) hallucinate, resulting in an emerging topic of visual hallucination evaluation (VHE). This paper contributes a ChatGPT-Prompted visual hallucination  evaluation Dataset (PhD) for objective VHE at a large scale.  The essence of VHE is to ask an MLLM questions about specific images to assess its susceptibility to hallucination. Depending on what to ask (objects, attributes, sentiment, etc.) and how the questions are asked, we structure PhD along two dimensions, i.e., task and mode. Five visual recognition tasks, ranging from low-level (object  / attribute recognition) to middle-level (sentiment / position recognition and counting), are considered. Besides a normal visual QA mode, which we term PhD-base, PhD also asks questions with inaccurate context (PhD-iac) or with incorrect context (PhD-icc), or with AI-generated counter common sense images (PhD-ccs). We construct PhD by a ChatGPT-assisted semi-automated pipeline, encompassing four pivotal modules: task-specific hallucinatory item (hitem) selection, hitem-embedded question generation, inaccurate / incorrect context generation, and counter-common-sense (CCS) image generation. With over 14k daily images, 750 CCS images and 102k VQA triplets in total, PhD reveals considerable variability in MLLMs' performance across various modes and tasks, offering valuable insights into the nature of hallucination. As such, PhD stands as a potent tool not only for VHE but may also play a significant role in the refinement of MLLMs.

</details>

---

## 121. DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation

- [ ] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/33140

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33140

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation aims to segment images into distinct semantic regions for both seen and unseen categories at the pixel level. Current methods utilize text embeddings from pre-trained vision-language models like CLIP but struggle with the inherent domain gap between image and text embeddings, even after extensive alignment during training. Additionally, relying solely on deep text-aligned features limits shallow-level feature guidance, which is crucial for detecting small objects and fine details, ultimately reducing segmentation accuracy.To address these limitations, we propose a dual prompting framework, DPSeg, for this task. Our approach combines dual-prompt cost volume generation, a cost volume-guided decoder, and a semantic-guided prompt refinement strategy that leverages our dual prompting scheme to mitigate alignment issues in visual prompt generation. By incorporating visual embeddings from a visual prompt encoder, our approach reduces the domain gap between text and image embeddings while providing multi-level guidance through shallow features. Extensive experiments demonstrate that our method significantly outperforms existing state-of-the-art approaches on multiple public datasets.

</details>

---

## 122. Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models?

- [ ] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models? | https://cvpr.thecvf.com/virtual/2025/poster/33133

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33133

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability.However, it is unknown what is actually learned in the high-quality dataset.Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences.Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction fine-tuning. Simply correcting the underlying data bias is enough to address the vision domain safety gap.

</details>

---

## 123. LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale

- [ ] LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale | https://cvpr.thecvf.com/virtual/2025/poster/33157

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33157

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent video large language models (Video LLMs) often depend on costly human annotations or proprietary APIs (e.g., GPT-4) to produce training data, which limits their training at scale. In this paper, we explore large-scale training for Video LLM with cheap automatic speech recognition (ASR) transcripts. Specifically, we propose a novel streaming training approach that densely interleaves the ASR words and video frames according to their timestamps. Compared to previous studies in vision-language representation with ASR, our method enables the model to learn fine-grained vision-language correlations in temporal. To support this, we introduce a series of data processing techniques on YouTube videos and closed captions (CC), resulting in 30M pre-training data samples and 1.5M for instruction tuning. Benefiting from our training paradigm, the trained model is powerful at streaming applications and can naturally support real-time video commentary. We also introduce a new benchmark focused on sports commentary and event understanding, a domain where live performance is critical. Experiments show that our model outperforms state-of-the-art models in both accuracy and latency. Additionally, our model achieves state-of-the-art or competitive results on several mainstream benchmarks, demonstrating its broad generalizability. We will release the codes, datasets, and models to facilitate further research.

</details>

---

## 124. ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models

- [ ] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33160

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33160

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucination poses a persistent challenge for multimodal large language models (MLLMs). However, existing benchmarks for evaluating hallucinations are generally static, which may overlook the potential risk of data contamination. To address this issue, we propose ODE, an open-set, dynamic protocol designed to evaluate object hallucinations in MLLMs at both the existence and attribute levels. ODE employs a graph-based structure to represent real-world object concepts, their attributes, and the distributional associations between them. This structure facilitates the extraction of concept combinations based on diverse distributional criteria, generating varied samples for structured queries that evaluate hallucinations in both generative and discriminative tasks. Through the generation of new samples, dynamic concept combinations, and varied distribution frequencies, ODE mitigates the risk of data contamination and broadens the scope of evaluation. This protocol is applicable to both general and specialized scenarios, including those with limited data. Experimental results demonstrate the effectiveness of our protocol, revealing that MLLMs exhibit higher hallucination rates when evaluated with ODE-generated samples, which indicates potential data contamination. Furthermore, these generated samples aid in analyzing hallucination patterns and fine-tuning models, offering an effective approach to mitigating hallucinations in MLLMs.

</details>

---

## 125. Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Large Model Enhancement

- [ ] Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Large Model Enhancement | https://cvpr.thecvf.com/virtual/2025/poster/33163

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33163

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) bring powerful understanding and reasoning capabilities to multimodal tasks. Meanwhile, the great need for capable aritificial intelligence on mobile devices also arises, such as the AI assistant software. Some efforts try to migrate VLMs to edge devices to expand their application scope. Simplifying the model structure is a common method, but as the model shrinks, the trade-off between performance and size becomes more and more difficult. Knowledge distillation (KD) can help models improve comprehensive capabilities without increasing size or data volume. However, most of the existing large model distillation techniques only consider applications on single-modal LLMs, or only use teachers to create new data environments for students. None of these methods take into account the distillation of the most important cross-modal alignment knowledge in VLMs. We propose a method called Align-KD to guide the student model to learn the cross-modal matching that occurs at the shallow layer. The teacher also helps student learn the projection of vision token into text embedding space based on the focus of text. Under the guidance of Align-KD, the 1.7B MobileVLM V2 model can learn rich knowledge from the 7B teacher model with light design of training loss, and achieve an average score improvement of 2.0 across 6 benchmarks under two training subsets respectively.

</details>

---

## 126. Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception

- [ ] Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception | https://cvpr.thecvf.com/virtual/2025/poster/33162

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33162

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have achieved impressive results across various multi-modal tasks. However, hallucinations, i.e., the models generating counterfactual responses, remain a challenge. Though recent studies have attempted to alleviate object perception hallucinations, they focus on the models' response generation, overlooking the task question itself. This paper discusses the vulnerability of LVLMs in solving counterfactual presupposition questions (CPQs), where the models are prone to accept the presuppositions of counterfactual objects and produce severe hallucinatory responses. To this end, we introduce “Antidote,” a unified, synthetic data-driven post-training framework for mitigating both types of hallucination above. It leverages synthetic data to incorporate factual priors into questions to achieve self-correction and decouple the mitigation process into a preference optimization problem. Furthermore, we construct “CP-Bench,” a novel benchmark to evaluate LVLMs' ability to correctly handle CPQs and produce factual responses. Applied to the LLaVA series, Antidote can simultaneously enhance performance on CP-Bench by over 50%, POPE by 1.8-3.3%, and CHAIR & SHR by 30-50%, all without relying on external supervision from stronger LVLMs or human feedback and introducing noticeable catastrophic forgetting issues.

</details>

---

## 127. MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking

- [ ] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking | https://cvpr.thecvf.com/virtual/2025/poster/33167

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33167

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The vision-language tracking task aims to perform object tracking based on various modality references. Existing Transformer-based vision-language tracking methods have made remarkable progress by leveraging the global modeling ability of self-attention. However, current approaches still face challenges in effectively exploiting the temporal information and dynamically updating reference features during tracking. Recently, the State Space Model (SSM), known as Mamba, has shown astonishing ability in efficient long-sequence modeling. Particularly, its state space evolving process demonstrates promising capabilities in memorizing multimodal temporal information with linear complexity. Witnessing its success, we propose a Mamba-based vision-language tracking model to exploit its state space evolving ability in temporal space for robust multimodal tracking, dubbed MambaVLT. In particular, our approach mainly integrates a time-evolving hybrid state space block and a selective locality enhancement block, to capture contextual information for multimodal modeling and adaptive reference feature update. Besides, we introduce a modality-selection module that dynamically adjusts the weighting between visual and language references, mitigating potential ambiguities from either reference type. Extensive experimental results show that our method performs favorably against state-of-the-art trackers across diverse benchmarks.

</details>

---

## 128. TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model

- [ ] TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model | https://cvpr.thecvf.com/virtual/2025/poster/33168

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33168

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) demand substantial computational resources during inference, largely due to the extensive visual input tokens required to represent visual information. Previous studies have observed that visual tokens often receive less attention than other tokens, such as system and instruction tokens, highlighting their lower relative importance during VLM inference and then pruning redundant visual tokens. However, previous approaches to token pruning encounter several challenges: reliance on heuristic criteria for token importance and incompatibility with FlashAttention and KV cache. To address these issues, we introduce TopV, a compatible Token Pruning with inference Time Optimization for fast and low-memory VLM, achieving efficient pruning without additional training or fine-tuning. Instead of relying on attention scores as the importance metric in the previous works, we formulate token pruning as an optimization problem, allowing us to accurately identify important visual tokens. By avoiding the need for attention scores, our approach maintains compatibility with FlashAttention. Additionally, since we only perform this pruning once during the prefilling stage, it effectively reduces KV cache size. Our optimization framework incorporates several critical components. First, given the to-be-pruned source tokens, we investigate the appropriate positions of target tokens within the VLM layer. Then, we define a visual-aware cost function considering factors such as Feature Similarity, Relative Spatial Distance, and Absolute Central Distance. Solving this optimization yields a contribution matrix that measures the importance of each source visual token in constructing target tokens, enabling effective pruning of low-importance tokens. Extensive experiments demonstrate that our method outperforms previous token pruning methods, validating the effectiveness and efficiency of our approach.

</details>

---

## 129. HiRes-LLaVA: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models

- [ ] HiRes-LLaVA: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33169

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33169

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-resolution image inputs allow Large Vision-Language Models (LVLMs) to capture finer visual details, improving comprehension. However, the increased training and computational costs associated with such inputs pose significant challenges. A common approach to mitigate these costs involves slicing the input into uniform patches using sliding windows, each aligned with the vision encoder’s input size. While efficient, this method fragments the input, disrupting the continuity of context, which negatively impacts cross-patch perception tasks. To address these limitations, we propose HiRes-LLaVA, a novel framework designed to efficiently process high-resolution inputs of any size without altering the original contextual and geometric information. HiRes-LLaVA introduces two key components: (i) a SliceRestore Adapter (SRA) that reconstructs sliced patches into their original form, enabling efficient extraction of both global and local features through down-up-sampling and convolutional layers, and (ii) a Self-Mining Sampler (SMS) that compresses visual tokens based on internal relationships, preserving original context and positional information while reducing training overhead. To assess the ability of handling context fragmentation, we construct a new benchmark, EntityGrid-QA, consisting of edge-related tasks. Extensive experiments demonstrate the superiority of HiRes-LLaVA on both existing public benchmarks and EntityGrid-QA. For example, with SRA, our method achieves a performance improvement of ∼ 12% over state-of-the-art LVLMs in addressing fragmentation issues. Additionally, our SMS outperforms other visual token downsamplers, while offering high data efficiency.

</details>

---

## 130. SynTab-LLaVA: Enhancing Multimodal Table Understanding with Decoupled Synthesis

- [ ] SynTab-LLaVA: Enhancing Multimodal Table Understanding with Decoupled Synthesis | https://cvpr.thecvf.com/virtual/2025/poster/33173

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33173

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Due to the limited scale of multimodal table understanding (MTU) data, model performance is constrained. A straightforward approach is to use multimodal large language models to obtain more samples, but this may cause hallucinations, generate incorrect sample pairs, and cost significantly. To address the above issues, we design a simple yet effective synthesis framework that consists of two independent steps: table image rendering and table question and answer (Q\&A) pairs generation. We use table codes (HTML, LaTeX, Markdown) to synthesize images and generate Q\&A pairs with large language model (LLM). This approach leverages LLM’s high concurrency and low cost to boost annotation efficiency and reduce expenses. By inputting code instead of images, LLMs can directly access the content and structure of the table, reducing hallucinations in table understanding and improving the accuracy of generated Q\&A pairs. Finally, we synthesize a large-scale MTU dataset, SynTab, containing 636K images and 1.8M samples costing within \$200 in US dollars. We further introduce a generalist tabular multimodal model, SynTab-LLaVA. This model not only effectively extracts local textual content within the table but also enables global modeling of relationships between cells. SynTab-LLaVA achieves SOTA performance on 21 out of 24 in-domain and out-of-domain benchmarks, demonstrating the effectiveness and generalization of our method. Some data is provided in the supplementary materials and will be open-sourced later.

</details>

---

## 131. Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions

- [ ] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions | https://cvpr.thecvf.com/virtual/2025/poster/33183

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33183

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Grounding 3D object affordance is a task that locates objects in 3D space where they can be manipulated, which links perception and action for embodied intelligence. For example, for an intelligent robot, it is necessary to accurately ground the affordance of an object and grasp it according to human instructions. In this paper, we introduce a novel task that grounds 3D object affordance based on language instructions, visual observations and interactions, which is inspired by cognitive science. We collect an Affordance Grounding dataset with Points, Images and Language instructions (AGPIL) to support the proposed task. In the 3D physical world, due to observation orientation, object rotation, or spatial occlusion, we can only get a partial observation of the object. So this dataset includes affordance estimations of objects from full-view, partial-view, and rotation-view perspectives. To accomplish this task, we propose LMAffordance3D, the first multi-modal, language-guided 3D affordance grounding network, which applies a vision-language model to fuse 2D and 3D spatial features with semantic features. Comprehensive experiments on AGPIL demonstrate the effectiveness and superiority of our method on this task, even in unseen experimental settings.

</details>

---

## 132. Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input

- [ ] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input | https://cvpr.thecvf.com/virtual/2025/poster/33185

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33185

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This work focuses on tracking and understanding human motion using consumer wearable devices, such as VR/AR headsets, smart glasses, cellphones, and smartwatches. These devices provide diverse, multi-modal sensor inputs, including egocentric images, and 1-3 sparse IMU sensors in varied combinations. Motion descriptions can also accompany these signals. The diverse input modalities and their intermittent availability pose challenges for consistent motion capture and understanding. In this work, we present Ego4o (o for omni), a new framework for simultaneous human motion capture and understanding from multi-modal egocentric inputs. This method maintains performance with partial inputs while achieving better results when multiple modalities are combined. First, the IMU sensor inputs, the optional egocentric image, and text description of human motion are encoded into the latent space of a motion VQ-VAE. Next, the latent vectors are sent to the VQ-VAE decoder and optimized to track human motion. When motion descriptions are unavailable, the latent vectors can be input into a multi-modal LLM to generate human motion descriptions, which can further enhance motion capture accuracy. Quantitative and qualitative evaluations demonstrate the effectiveness of our method in predicting accurate human motion and high-quality motion descriptions.

</details>

---

## 133. Conformal Prediction for Zero-Shot Models

- [ ] Conformal Prediction for Zero-Shot Models | https://cvpr.thecvf.com/virtual/2025/poster/33189

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33189

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models pre-trained at large scale have shown unprecedented adaptability and generalization to downstream tasks. Although its discriminative potential has been widely explored, its reliability and uncertainty are still overlooked. In this work, we investigate the capabilities of CLIP models under the split conformal prediction paradigm, which provides theoretical guarantees to black-box models based on a small, labeled calibration set. In contrast to the main body of literature on conformal predictors in vision classifiers, foundation models exhibit a particular characteristic: they are pre-trained on a one-time basis on an inaccessible source domain, different from the transferred task. This domain drift negatively affects the efficiency of the conformal sets and poses additional challenges. To alleviate this issue, we propose Conf-OT, a transfer learning setting that operates transductive over the combined calibration and query sets. Solving an optimal transport problem, the proposed method bridges the domain gap between pre-training and adaptation without requiring additional data splits but still maintaining coverage guarantees. We comprehensively explore this conformal prediction strategy on a broad span of 15 datasets and three popular non-conformity scores. Conf-OT provides consistent relative improvements of up to 20% on set efficiency while being $\times$15 faster than popular transductive approaches.

</details>

---

## 134. Reproducible Vision-Language Models Meet Concepts Out of Pre-Training

- [ ] Reproducible Vision-Language Models Meet Concepts Out of Pre-Training | https://cvpr.thecvf.com/virtual/2025/poster/33200

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33200

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) models as a milestone of modern multimodal intelligence, its generalization mechanism grasped massive research interests in the community. While existing studies limited in the scope of pre-training knowledge, hardly underpinned its generalization to countless open-world concepts absent from the pre-training regime. This paper dives into such Out-of-Pre-training (OOP) generalization problem from a holistic perspective. We propose LAION-Beyond benchmark to isolate the evaluation of OOP concepts from pre-training knowledge, with regards to OpenCLIP and its reproducible variants derived from LAION datasets. Empirical analysis evidences that despite image features of OOP concepts born with significant category margins, their zero-shot transfer significantly fails due to the poor image-text alignment. To this, we elaborate the ``name-tuning'' methodology with its theoretical merits in terms of OOP generalization, then propose few-shot name learning (FSNL) and zero-shot name learning (ZSNL) algorithms to achieve OOP generalization in a data-efficient manner. Their superiority have been further verified in our comprehensive experiments.

</details>

---

## 135. OpenSDI: Spotting Diffusion-Generated Images in the Open World

- [ ] OpenSDI: Spotting Diffusion-Generated Images in the Open World | https://cvpr.thecvf.com/virtual/2025/poster/33201

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33201

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper identifies OpenSDI, a challenge for spotting diffusion-generated images in open-world settings. In response to this challenge, we define a new benchmark, the OpenSDI dataset (OpenSDID), which stands out from existing datasets due to its diverse use of large vision-language models that simulate open-world diffusion-based manipulations. Another outstanding feature of OpenSDID is its inclusion of both detection and localization tasks for images manipulated globally and locally by diffusion models. To address the OpenSDI challenge, we propose a Synergizing Pretrained Models (SPM) scheme to build up a mixture of foundation models. This approach exploits a collaboration mechanism with multiple pretrained foundation models to enhance generalization in the OpenSDI context, moving beyond traditional training by synergizing multiple pretrained models through prompting and attending strategies. Building on this scheme, we introduce MaskCLIP, an SPM-based model that aligns Contrastive Language-Image Pre-Training (CLIP) with Masked Autoencoder (MAE). Extensive evaluations on OpenSDID show that MaskCLIP significantly outperforms current state-of-the-art methods for the OpenSDI challenge, achieving remarkable relative improvements of 14.23\% in IoU (14.11\% in F1) and 2.05\% in accuracy (2.38\% in F1) compared to the second-best model in detection and localization tasks, respectively.

</details>

---

## 136. Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning

- [ ] Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning | https://cvpr.thecvf.com/virtual/2025/poster/33205

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33205

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have shown remarkable advancements in multimodal reasoning tasks. However, they still often generate inaccurate or irrelevant responses due to issues like hallucinated image understandings or unrefined reasoning paths. To address these challenges, we introduce Critic-V, a novel framework inspired by the Actor-Critic paradigm to boost the reasoning capability of VLMs. This framework decouples the reasoning process and critic process by integrating two independent components: the Reasoner, which generates reasoning paths based on visual and textual inputs, and the Critic, which provides constructive critique to refine these paths.In this approach, the Reasoner generates reasoning responses according to text prompts, which can evolve iteratively as a policy based on feedback from the Critic. This interaction process was theoretically driven by a reinforcement learning framework where the Critic offers natural language critiques instead of scalar rewards, enabling more nuanced feedback to boost the Reasoner's capability on complex reasoning tasks. The Critic model is trained using Direct Preference Optimization (DPO), leveraging a preference dataset of critiques ranked by Rule-based Reward (RBR) to enhance its critic capabilities. Evaluation results show that the Critic-V framework significantly outperforms existing methods, including GPT-4V, on 5 out of 8 benchmarks, especially regarding reasoning accuracy and efficiency. Combining a dynamic text-based policy for the Reasoner and constructive feedback from the preference-optimized Critic enables a more reliable and context-sensitive multimodal reasoning process. Our approach provides a promising solution to enhance the reliability of VLMs, improving their performance in real-world reasoning-heavy multimodal applications such as autonomous driving and embodied intelligence.

</details>

---

## 137. VoCo-LLaMA: Towards Vision Compression with Large Language Models

- [ ] VoCo-LLaMA: Towards Vision Compression with Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33214

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33214

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have achieved remarkable success in various multi-modal tasks, but they are often bottlenecked by the limited context window and high computational cost of processing high-resolution image inputs and videos. Vision compression can alleviate this problem by reducing the vision token count. Previous approaches compress vision tokens with external modules and force LLMs to understand the compressed ones, leading to visual information loss. However, the LLMs' understanding paradigm of vision tokens is not fully utilised in the compression learning process. We propose VoCo-LLaMA, the first approach to compress vision tokens using LLMs. By introducing Vision Compression tokens during the vision instruction tuning phase and leveraging attention distillation, our method distill how LLMs comprehend vision tokens into their processing of VoCo tokens. VoCo-LLaMA facilitates effective vision compression and improves the computational efficiency during the inference stage. Specifically, our method can achieve a 576 times compression rate while maintaining 83.7% performance. Furthermore, through continuous training using time-series compressed token sequences of video frames, VoCo-LLaMA demonstrates the ability to understand temporal correlations, outperforming previous methods on popular video question-answering benchmarks.Our approach presents a promising way to unlock the full potential of VLMs' contextual window, enabling more scalable multi-modal applications.

</details>

---

## 138. VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks

- [ ] VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks | https://cvpr.thecvf.com/virtual/2025/poster/33217

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33217

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Deploying high-performing neural networks in resource-constrained environments poses a significant challenge due to the computational demands of large-scale models. We introduce VL2Lite, a knowledge distillation framework designed to enhance the performance of lightweight neural networks in image classification tasks by leveraging the rich representational knowledge from Vision-Language Models (VLMs). VL2Lite directly integrates multi-modal knowledge from VLMs into compact models during training, effectively compensating for the limited computational and modeling capabilities of smaller networks. By transferring high-level features and complex data representations, our approach improves the accuracy and efficiency of image classification tasks without increasing computational overhead during inference. Experimental evaluations demonstrate that VL2Lite achieves up to a 7% improvement in classification performance across various datasets. This method addresses the challenge of deploying accurate models in environments with constrained computational resources, offering a balanced solution between model complexity and operational efficiency.

</details>

---

## 139. CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models

- [ ] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models | https://cvpr.thecvf.com/virtual/2025/poster/33233

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33233

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language-action models (VLAs) have shown potential in leveraging pretrained vision-language models and diverse robot demonstrations for learning generalizable sensorimotor control. While this paradigm effectively utilizes large-scale data from both robotic and non-robotic sources, current VLAs primarily focus on direct input--output mappings, lacking the intermediate reasoning steps crucial for complex manipulation tasks. As a result, existing VLAs lack temporal planning or reasoning capabilities. In this paper, we introduce a method that incorporates explicit visual chain-of-thought (CoT) reasoning into vision-language-action models (VLAs) by predicting future image frames autoregressively as visual goals before generating a short action sequence to achieve these goals. We introduce CoT-VLA, a state-of-the-art 7B VLA that can understand and generate visual and action tokens. Our experimental results demonstrate that CoT-VLA achieves strong performance, outperforming the state-of-the-art VLA model by 17\% in real-world manipulation tasks and 6\% in simulation benchmarks.

</details>

---

## 140. SketchAgent: Language-Driven Sequential Sketch Generation

- [ ] SketchAgent: Language-Driven Sequential Sketch Generation | https://cvpr.thecvf.com/virtual/2025/poster/33235

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33235

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Sketching serves as a versatile tool for externalizing ideas, enabling rapid exploration and visual communication that spans various disciplines. While artificial systems have driven substantial advances in content creation and human-computer interaction, capturing the dynamic and abstract nature of human sketching remains challenging. In this work, we introduce SketchAgent, a language-driven, sequential sketch generation method that enables users to create, modify, and refine sketches through dynamic, conversational interactions.Our approach requires no training or fine-tuning. Instead, we leverage the sequential nature and rich prior knowledge of off-the-shelf multimodal large language models (LLMs). We present an intuitive sketching language, introduced to the model through in-context examples, enabling it to "draw" using string-based actions. These are processed into vector graphics and then rendered to create a sketch on a pixel canvas, which can be accessed again for further tasks.By drawing stroke by stroke, our agent captures the evolving, dynamic qualities intrinsic to sketching. We demonstrate that SketchAgent can generate sketches from diverse prompts, engage in dialogue-driven drawing, and collaborate meaningfully with human users.

</details>

---

## 141. EgoLM: Multi-Modal Language Model of Egocentric Motions

- [ ] EgoLM: Multi-Modal Language Model of Egocentric Motions | https://cvpr.thecvf.com/virtual/2025/poster/33242

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33242

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As wearable devices become more prevalent, understanding the user's motion is crucial for improving contextual AI systems. We introduce EgoLM, a versatile framework designed for egocentric motion understanding using multi-modal data. EgoLM integrates the rich contextual information from egocentric videos and motion sensors afforded by wearable devices. It also combines dense supervision signals from motion and language, leveraging the vast knowledge encoded in pre-trained large language models (LLMs). EgoLM models the joint distribution of egocentric motions and natural language using LLMs, conditioned on observations from egocentric videos and motion sensors. It unifies a range of motion understanding tasks, including motion narration from video or motion data, as well as motion generation from text or sparse sensor data. Unique to wearable devices, it also enables a novel task to generate text descriptions from sparse sensors. Through extensive experiments, we validate the effectiveness of EgoLM in addressing the challenges of under-constrained egocentric motion learning, and demonstrate its capability as a generalist model through a variety of applications.

</details>

---

## 142. VASparse: Towards Efficient Visual Hallucination Mitigation via Visual-Aware Token Sparsification

- [ ] VASparse: Towards Efficient Visual Hallucination Mitigation via Visual-Aware Token Sparsification | https://cvpr.thecvf.com/virtual/2025/poster/33244

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33244

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) may produce outputs that are unfaithful to reality, also known as visual hallucinations (VH), which significantly impedes their real-world usage. To alleviate VH, various decoding strategies have been proposed to enhance visual information. However, many of these methods may require secondary decoding and rollback, which significantly reduces inference speed. In this work, we propose an efficient plug-and-play decoding algorithm via Visual-Aware Sparsification (VASparse) from the perspective of token sparsity for mitigating VH. VASparse is inspired by empirical observations: (1) the sparse activation of attention in LVLMs, and (2) visual-agnostic tokens sparsification exacerbates VH. Based on these insights, we propose a novel token sparsification strategy that balances efficiency and trustworthiness.  Specifically, VASparse implements a visual-aware token selection strategy during decoding to reduce redundant tokens while preserving visual context effectively.  Additionally, we innovatively introduce a sparse-based visual contrastive decoding method to recalibrate the distribution of hallucinated outputs without the time overhead associated with secondary decoding. Subsequently, VASparse recalibrates attention scores to penalize attention sinking of LVLMs towards text tokens. Extensive experiments across four popular benchmarks confirm the effectiveness of VASparse in mitigating VH across different LVLM families without requiring additional training or post-processing. Impressively, VASparse achieves state-of-the-art performance for mitigating VH while maintaining competitive decoding speed. Code is available at https://anonymous.4open.science/r/VASparse-128C .

</details>

---

## 143. Locality-Aware Zero-Shot Human-Object Interaction Detection

- [ ] Locality-Aware Zero-Shot Human-Object Interaction Detection | https://cvpr.thecvf.com/virtual/2025/poster/33246

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33246

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent methods for zero-shot Human-Object Interaction (HOI) detection typically leverage the generalization ability of large Vision-Language Model (VLM), $\textit{i.e.,}$ CLIP, on unseen categories, showing impressive results on various zero-shot settings.However, existing methods struggle to adapt CLIP representations for human-object pairs, as CLIP tends to overlook fine-grained information necessary for distinguishing interactions.To address this issue, we devise, LAIN, a novel zero-shot HOI detection framework enhancing the locality and interaction awareness of CLIP representations.The locality awareness, which involves capturing fine-grained details and the spatial structure of individual objects, is achieved by aggregating the information and spatial priors of adjacent neighborhood patches.The interaction awareness, which involves identifying whether and how a human is interacting with an object, is achieved by capturing the interaction pattern between the human and the object.By infusing locality and interaction awareness into CLIP representation, LAIN captures detailed information about the human-object pairs.Our extensive experiments on existing benchmarks show that LAIN outperforms previous methods on various zero-shot settings, demonstrating the importance of locality and interaction awareness for effective zero-shot HOI detection.

</details>

---

## 144. ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding

- [ ] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding | https://cvpr.thecvf.com/virtual/2025/poster/33251

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33251

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present ASAP, a new framework for detecting and grounding multi-modal media manipulation (DGM4).Upon thorough examination, we observe that accurate fine-grained cross-modal semantic alignment between the image and text is vital for accurately manipulation detection and grounding. While existing DGM4 methods pay rare attention to the cross-modal alignment, hampering the accuracy of manipulation detecting to step further. To remedy this issue, this work targets to advance the semantic alignment learning to promote this task. Particularly, we utilize the off-the-shelf Multimodal Large-Language Models (MLLMs) and Large Language Models (LLMs) to construct paired image-text pairs, especially for the manipulated instances. Subsequently, a cross-modal alignment learning is performed to enhance the semantic alignment. Besides the explicit auxiliary clues, we further design a Manipulation-Guided Cross Attention (MGCA) to provide implicit guidance for augmenting the manipulation perceiving. With the grounding truth available during training, MGCA encourages the model to concentrate more on manipulated components while downplaying normal ones, enhancing the model's ability to capture manipulations. Extensive experiments are conducted on the DGM4 dataset, the results demonstrate that our model can surpass the comparison method with a clear margin.

</details>

---

## 145. Marten: Visual Question Answering with Mask Generation for Multi-modal Document Understanding

- [ ] Marten: Visual Question Answering with Mask Generation for Multi-modal Document Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33267

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33267

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) have introduced a novel dimension to document understanding, i.e., they endow large language models with visual comprehension capabilities; however, how to design a suitable image-text pre-training task for bridging the visual and language modality in document-level MLLMs remains underexplored. In this study, we introduce a novel visual-language alignment method that casts the key issue as a Visual Question Answering with Mask generation (VQAMask) task, optimizing two tasks simultaneously: VQA-based text parsing and mask generation. The former allows the model to implicitly align images and text at the semantic level. The latter introduces an additional mask generator (discarded during inference) to explicitly ensure alignment between visual texts within images and their corresponding image regions at a spatially-aware level. Together, they can prevent model hallucinations when parsing visual text and effectively promote spatially-aware feature representation learning. To support the proposed VQAMask task, we construct a comprehensive image-mask generation pipeline and provide a large-scale dataset with 6M data (MTMask6M). Subsequently, we demonstrate that introducing the proposed mask generation task yields competitive document-level understanding performance. Leveraging the proposed VQAMask, we introduce Marten, a training-efficient MLLM tailored for document-level understanding. Extensive experiments show that our Marten consistently achieves significant improvements among 8B-MLLMs in document-centric tasks. Code and datasets will be available soon.

</details>

---

## 146. Seeing the Abstract: Translating the Abstract Language for Vision Language Models

- [ ] Seeing the Abstract: Translating the Abstract Language for Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33268

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33268

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Natural language goes beyond dryly describing visual content. It contains rich abstract concepts to express feeling, creativity and properties that cannot be directly perceived. Yet, current research in Vision Language Models (VLMs) has not shed light on abstract-oriented language.Our research breaks new ground by uncovering its wide presence and under-estimated value, with extensive analysis.Particularly, we focus our investigation on the fashion domain, a highly-representative field with abstract expressions. By analyzing recent large-scale multimodal fashion datasets, we find that abstract terms have a dominant presence, rivaling the concrete ones, providing novel information, and being useful in the retrieval task. However, a critical challenge emerges: current general-purpose or fashion-specific VLMs are pre-trained with databases that lack sufficient abstract words in their text corpora, thus hindering their ability to effectively represent abstract-oriented language. We propose a training-free and model-agnostic method, Abstract-to-Concrete Translator (ACT), to shift abstract representations towards well-represented concrete ones in the VLM latent space, using pre-trained models and existing multimodal databases.On the text-to-image retrieval task, despite being training-free, ACT outperforms the fine-tuned VLMs in both same- and cross-dataset settings, exhibiting its effectiveness with a strong generalization capability. Moreover, the improvement introduced by ACT is consistent with various VLMs, making it a plug-and-play solution.Our code will be publicly available.

</details>

---

## 147. VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge

- [ ] VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge | https://cvpr.thecvf.com/virtual/2025/poster/33276

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33276

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generalist vision language models (VLMs) have made significant strides in computer vision, but they fall short in specialized fields like healthcare, where expert knowledge is essential. Current large multimodal models like Gemini and GPT-4o are insufficient for medical tasks due to their reliance on memorized internet knowledge rather than the nuanced expertise required in healthcare. Meanwhile, existing medical VLMs (e.g. Med-Gemini) often lack expert consultation as part of their design, and many rely on outdated, static datasets that were not created with modern, large deep learning models in mind. VLMs are usually trained in three stages: vision pre-training, vision-language pre-training, and instruction fine-tuning (IFT). IFT has been typically applied using a mixture of generic and healthcare data. In contrast, we propose that for medical VLMs, a fourth stage of specialized IFT is necessary, which focuses on medical data and includes information from domain expert models. Domain expert models developed for medical use are crucial because they are specifically trained for certain clinical tasks, e.g. to detect tumors and classify abnormalities through segmentation and classification, which learn fine-grained features of medical data$-$features that are often too intricate for a VLM to capture effectively. This paper introduces a new framework, VILA-M3, for medical VLMs that utilizes domain knowledge via expert models. We argue that generic VLM architectures alone are not viable for real-world clinical applications and on-demand usage of domain-specialized expert model knowledge is critical for advancing AI in healthcare. Through our experiments, we show an improved state-of-the-art (SOTA) performance with an average improvement of $\sim$9\% over the prior SOTA model Med-Gemini and $\sim$6\% over models trained on the specific tasks. Our approach emphasizes the importance of domain expertise in creating precise, reliable VLMs for medical applications.

</details>

---

## 148. Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents

- [ ] Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents | https://cvpr.thecvf.com/virtual/2025/poster/33281

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33281

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal models (LMMs) have achieved impressive progress in vision-language understanding, yet they face limitations in real-world applications requiring complex reasoning over a large number of images. Existing benchmarks for multi-image question-answering are limited in scope, each question is paired with only up to 30 images, which does not fully capture the demands of large-scale retrieval tasks encountered in the real-world usages. To reduce these gaps, we introduce two document haystack benchmarks, dubbed DocHaystack and InfoHaystack, designed to evaluate LMM performance on large-scale visual document retrieval and understanding. Additionally, we propose V-RAG, a novel, vision-centric retrieval-augmented generation (RAG) framework that leverages a suite of multimodal vision encoders, each optimized for specific strengths, and a dedicated question-document relevance module. V-RAG sets a new standard, with a 9\% and 11\% improvement in Recall@1 on the challenging DocHaystack-1000 and InfoHaystack-1000 benchmarks, respectively, compared to the previous best baseline models. Additionally, integrating V-RAG with LMMs enables them to efficiently operate across thousands of images, yielding significant improvements on our DocHaystack and InfoHaystack benchmarks. Our code and datasets will be made publicly available.

</details>

---

## 149. Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval

- [ ] Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval | https://cvpr.thecvf.com/virtual/2025/poster/33289

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33289

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Composed Image Retrieval (CIR) aims to retrieve target images that closely resemble a reference image while integrating user-specified textual modifications, thereby capturing user intent more precisely. This dual-modality approach is especially valuable in internet search and e-commerce, facilitating tasks like scene image search with object manipulation and product recommendations with attribute changes. Existing training-free zero-shot CIR (ZS-CIR) methods often employ a two-stage process: they first generate a caption for the reference image and then use Large Language Models for reasoning to obtain a target description. However, these methods suffer from missing critical visual details and limited reasoning capabilities, leading to suboptimal retrieval performance. To address these challenges, we propose a novel, training-free one-stage method, One-Stage Reflective Chain-of-Thought Reasoning for ZS-CIR (OSrCIR), which employs Multimodal Large Language Models to retain essential visual information in a single-stage reasoning process, eliminating the information loss seen in two-stage methods. Our Reflective Chain-of-Thought framework further improves interpretative accuracy by aligning manipulation intent with contextual cues from reference images. OSrCIR achieves performance gains of 1.80% to 6.44% over existing training-free methods across multiple tasks, setting new state-of-the-art results in ZS-CIR and enhancing its utility in vision-language applications. Our code is available at https://anonymous.4open.science/r/osrcir24/.

</details>

---

## 150. Docopilot: Improving Multimodal Models for Document-Level Understanding

- [ ] Docopilot: Improving Multimodal Models for Document-Level Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33306

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33306

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant progress in multimodal large language models (MLLMs), their performance on complex, multi-page document comprehension remains inadequate, largely due to the lack of high-quality, document-level datasets.While current retrieval-augmented generation (RAG) methods offer partial solutions, they suffer from issues, such as fragmented retrieval contexts, multi-stage error accumulation, and extra time costs of retrieval. In this work, we present a high-quality document-level dataset, Doc-750K, designed to support in-depth understanding of multimodal documents.This dataset includes diverse document structures, extensive cross-page dependencies, and real question-answer pairs derived from original documents.Building on the dataset, we developed a native multimodal model—Docopilot, which can accurately handle document-level dependencies without relying on RAG.Experiments demonstrate that Docopilot achieves superior coherence, accuracy, and efficiency in document understanding tasks and multi-turn interactions, setting a new baseline for document-level multimodal understanding. Data, code, and models shall be released.

</details>

---

## 151. IDEA: Inverted Text with Cooperative Deformable Aggregation for Multi-modal Object Re-Identification

- [ ] IDEA: Inverted Text with Cooperative Deformable Aggregation for Multi-modal Object Re-Identification | https://cvpr.thecvf.com/virtual/2025/poster/33307

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33307

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal object Re-IDentification (ReID) aims to retrieve specific objects by utilizing complementary information from various modalities. However, existing methods focus on fusing heterogeneous visual features, neglecting the potential benefits of text-based semantic information. To address this issue, we first construct three text-enhanced multi-modal object ReID benchmarks. To be specific, we propose a standardized multi-modal caption generation pipline for structured and concise text annotations with Multi-modal Large Language Models (MLLMs). Additionally, current methods often directly aggregate multi-modal features without selecting representative local features, leading to redundancy and high complexity. To address the above issues, we introduce IDEA, a novel feature learning framework comprising the Inverted Multi-modal Feature Extractor (IMFE) and Cooperative Deformable Aggregation (CDA). The IMFE utilizes Modal Prefixes and an InverseNet to integrate multi-modal information with semantic guidance from inverted text. The CDA adaptively generates sampling positions, enabling the model to focus on the interplay between global features and discriminative local features. With the constructed benchmarks and the proposed modules, our framework can generate more robust multi-modal features under complex scenarios. Extensive experiments on three multi-modal object ReID benchmarks demonstrate the effectiveness of our proposed method.

</details>

---

## 152. NVILA: Efficient Frontier Visual Language Models

- [ ] NVILA: Efficient Frontier Visual Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33311

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33311

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual language models (VLMs) have made significant strides in accuracy in recent years, but their efficiency has often been overlooked. This paper presents NVILA, a family of open frontier VLMs designed to optimize both efficiency and accuracy. Building upon VILA, we improve its model architecture by first scaling up spatial and temporal resolutions, and then compressing visual tokens. This "scale-then-compress" approach allows NVILA to efficiently process high-resolution images and long videos. We also conduct a systematic study to improve NVILA's efficiency throughout its entire lifecycle—from training and fine-tuning to deployment. NVILA is both efficient and accurate, reducing training costs by 4.5×, fine-tuning memory usage by 3.4×, prefilling latency by 1.8× and decoding throughput by 1.2×. It also achieves state-of-the-art results across diverse image and video benchmarks. We will release our implementation and trained models to support full reproducibility.

</details>

---

## 153. Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices

- [ ] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices | https://cvpr.thecvf.com/virtual/2025/poster/33317

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33317

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have made significant advancements in recent years, with visual features playing an increasingly critical role in enhancing model performance. However, the integration of multi-layer visual features in MLLMs remains underexplored, particularly with regard to optimal layer selection and fusion strategies. Existing methods often rely on arbitrary design choices, leading to suboptimal outcomes. In this paper, we systematically investigate two core aspects of multi-layer visual feature fusion: (1) selecting the most effective visual layers and (2) identifying the best fusion approach with the language model. Our experiments reveal that while combining visual features from multiple stages improves generalization, incorporating additional features from the same stage typically leads to diminished performance. Furthermore, we find that direct fusion of multi-layer visual features at the input stage consistently yields superior and more stable performance across various configurations.

</details>

---

## 154. Zero-shot 3D Question Answering via Voxel-based Dynamic Token Compression

- [ ] Zero-shot 3D Question Answering via Voxel-based Dynamic Token Compression | https://cvpr.thecvf.com/virtual/2025/poster/33335

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33335

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in 3D Large Multi-modal Models (3D-LMMs) have driven significant progress in 3D question answering. However, recent multi-frame Vision-Language Models (VLMs) demonstrate superior performance compared to 3D-LMMs on 3D question answering tasks, largely due to the greater scale and diversity of available 2D image data in contrast to the more limited 3D data. Multi-frame VLMs, although achieving superior performance, suffer from the difficulty of retaining all the detailed visual information in the 3D scene while limiting the number of visual tokens. Common methods such as token pooling, reduce visual token usage but often lead to information loss, impairing the model’s ability to preserve visual details essential for 3D question answering tasks. To address this, we propose voxel-based Dynamic Token Compression (DTC), which combines 3D spatial priors and visual semantics to achieve over 90% reduction in visual tokens usage for current multi-frame VLMs. Our method maintains performance comparable to state-of-the-art models on 3D question answering benchmarks including OpenEQA and ScanQA, demonstrating its effectiveness.

</details>

---

## 155. POPEN: Preference-Based Optimization and Ensemble for LVLM-Based Reasoning Segmentation

- [ ] POPEN: Preference-Based Optimization and Ensemble for LVLM-Based Reasoning Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/33340

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33340

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing LVLM-based reasoning segmentation methods often suffer from imprecise segmentation results and hallucinations in their text responses. This paper introduces POPEN, a novel framework designed to address these issues and achieve improved results. POPEN includes a preference-based optimization method to finetune the LVLM, aligning it more closely with human preferences and thereby generating better text responses and segmentation results. Additionally, POPEN introduces a preference-based ensemble method for inference, which integrates multiple outputs from the LVLM using a preference-score-based attention mechanism for refinement. To better adapt to the segmentation task, we incorporate several task-specific designs in our POPEN framework, including a new approach for collecting segmentation preference data with a curriculum learning mechanism, and a novel preference optimization loss to refine the segmentation capability of the LVLM. Experiments demonstrate that our method achieves state-of-the-art performance in reasoning segmentation, exhibiting minimal hallucination in text responses and the highest segmentation accuracy compared to previous advanced methods like LISA and PixelLM.

</details>

---

## 156. MotionBench: Benchmarking and Improving Fine-grained Video Motion Understanding for Vision Language Models

- [ ] MotionBench: Benchmarking and Improving Fine-grained Video Motion Understanding for Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33344

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33344

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, vision language models (VLMs) have made significant advancements in video understanding. However, a crucial capability — fine-grained motion comprehension — remains under-explored in current benchmarks. To address this gap, we propose MotionBench, a comprehensive evaluation benchmark designed to assess the fine-grained motion comprehension of video understanding models. MotionBench evaluates models' motion-level perception through six primary categories of motion-oriented question types and includes data collected from diverse sources, ensuring a broad representation of real-world video content.Experimental results reveal that existing VLMs perform poorly in understanding fine-grained motions.To enhance VLM's ability to perceive fine-grained motion within a limited LLM sequence length, we conduct extensive experiments reviewing VLM architectures optimized for video feature compression, and propose a novel and efficient Through-Encoder (TE) Fusion method. Experiments show that higher frame rate inputs together with TE Fusion yield improvements in motion understanding, yet there is still substantial room for enhancement. Our benchmark aims to guide and motivate the development of more capable video understanding models, emphasizing the importance of fine-grained motion comprehension.

</details>

---

## 157. MP-GUI: Modality Perception with MLLMs for GUI Understanding

- [ ] MP-GUI: Modality Perception with MLLMs for GUI Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33358

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33358

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical user interface (GUI) has become integral to modern society, making it crucial to be understood for human-centric systems. The rapid development of multi-modal large language models (MLLMs) in recent years has revealed their significant potential in GUI understanding. However, unlike natural images or documents, GUIs comprise artificially designed graphical elements arranged to convey specific semantic meanings. Current MLLMs already proficient in processing graphical and textual components suffer from hurdles in GUI understanding due to the lack of explicit spatial structure modeling. Moreover, obtaining high-quality spatial structure data is challenging due to privacy issues and noisy environments. To tackle these challenges, this paper presents MP-GUI, a specially designed MLLM for GUI understanding. MP-GUI features three precisely specialized perceivers to extract graphical, textual, and spatial modality from GUIs, with spatial structure enhancing strategy and adaptively combined via a fusion gate to meet the distinct requirements of different GUI interpretation tasks. To cope with the scarcity of high-quality data, we also introduce a pipeline for automatically collecting spatial information. Our extensive experiments demonstrate that MP-GUI achieves impressive results on numerous GUI understanding tasks even with a limited amount of generated data.

</details>

---

## 158. Provoking Multi-modal Few-Shot LVLM via Exploration-Exploitation In-Context Learning

- [ ] Provoking Multi-modal Few-Shot LVLM via Exploration-Exploitation In-Context Learning | https://cvpr.thecvf.com/virtual/2025/poster/33366

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33366

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In-context learning (ICL), a predominant trend in instruction learning, aims at enhancing the performance of large language models by providing clear task guidance and examples, improving their capability in task understanding and execution. This paper investigates ICL on Large Vision-Language Models (LVLMs) and explores the policies of multi-modal demonstration selection. Existing research efforts in ICL face significant challenges: First, they rely on pre-defined demonstrations or heuristic selecting strategies based on human intuition, which are usually inadequate for covering diverse task requirements, leading to sub-optimal solutions; Second, individually selecting each demonstration fails in modeling the interactions between them, resulting in information redundancy. Unlike these prevailing efforts, we propose a new exploration-exploitation reinforcement learning framework, which explores policies to fuse multi-modal information and adaptively select adequate demonstrations as an integrated whole. The framework allows LVLMs to optimize themselves by continually refining their demonstrations through self-exploration, enabling the ability to autonomously identify and generate the most effective selection policies for in-context learning. Experimental results verify that our approach achieves significant performance improvements on four Visual Question-Answering (VQA) datasets, demonstrating its effectiveness in enhancing the generalization capability of few-shot LVLMs. The code will be publicly available.

</details>

---

## 159. Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding

- [ ] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33367

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33367

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long video understanding poses a significant challenge for current Multi-modal Large Language Models (MLLMs). Notably, the MLLMs are constrained by their limited context lengths and the substantial costs while processing long videos. Although several existing methods attempt to reduce visual tokens, their strategies encounter severe bottleneck, restricting MLLMs' ability to perceive fine-grained visual details. In this work, we propose Video-XL, a novel approach that leverages MLLMs' inherent key-value (KV) sparsification capacity to condense the visual input. Specifically, we introduce a new special token, the Visual Summarization Token (VST), for each interval of the video, which summarizes the visual information within the interval as its associated KV. The VST module is trained by instruction fine-tuning, where two optimizing strategies are offered. 1. Curriculum learning, where VST learns to make small (easy) and large compression (hard) progressively. 2. Composite data curation, which integrates single-image, multi-image, and synthetic data to overcome the scarcity of long-video instruction data. The compression quality is further improved by dynamic compression, which customizes compression granularity based on the information density of different video intervals. Video-XL's effectiveness is verified from three aspects. First, it achieves a superior long-video understanding capability, outperforming state-of-the-art models of comparable sizes across multiple popular benchmarks. Second, it effectively preserves video information, with minimal compression loss even at 16x compression ratio. Third, it realizes outstanding cost-effectiveness, enabling high-quality processing of thousands of frames on a single A100 GPU.

</details>

---

## 160. Multi-modal Knowledge Distillation-based Human Trajectory Forecasting

- [ ] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting | https://cvpr.thecvf.com/virtual/2025/poster/33379

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33379

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pedestrian trajectory forecasting is crucial in various applications such as autonomous driving and mobile robot navigation. Their camera-based visual features enable the extraction of additional modalities (human pose, text) which enhance prediction accuracy. We focus on pedestrian motion prediction to fully utilize the rich, dynamic visual features of pedestrians. Indeed, we find that textual descriptions play a crucial role in integrating additional modalities into a unified understanding. However, online extraction of text requires an use of VLM, which may not be feasible for resource-constrained systems. To address this challenge, we propose a multi-modal knowledge distillation framework: a student model with limited modality is distilled from a teacher model trained with full range of modalities. The comprehensive knowledge of a teacher model trained with trajectory, human pose, and text is distilled into a student model using only trajectory or human pose as a sole supplement. We validate our generalizable framework with two state-of-the-art models across three datasets on both ego-view (JRDB, SIT) and BEV-view (ETH/UCY) setups. For the SIT dataset, we utilize VLM to generate captions to compensate for the lack of text annotations. Distilled student models show consistent improvement in all prediction metrics for both full and instantaneous observations.

</details>

---

## 161. VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models

- [ ] VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models | https://cvpr.thecvf.com/virtual/2025/poster/33389

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33389

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language generative reward models (VL-GenRMs) play a crucial role in aligning and evaluating multimodal AI systems, yet their own evaluation remains under-explored. Current assessment methods primarily rely on AI-annotated preference labels from traditional VL tasks, which can introduce biases and often fail to effectively challenge state-of-the-art models.To address these limitations, we introduce VL-RewardBench, a comprehensive benchmark spanning general multimodal queries, visual hallucination detection, and complex reasoning tasks.Through our AI-assisted annotation pipeline combining sample selection with human verification, we curate 1,250 high-quality examples specifically designed to probe model limitations.Comprehensive evaluation across 16 leading large vision-language models, demonstrates VL-RewardBench's effectiveness as a challenging testbed, where even GPT-4o achieves only 65.4\% accuracy, and state-of-the-art open-source models such as Qwen2-VL-72B, struggle to surpass random-guessing. Importantly, performance on VL-RewardBench strongly correlates (Pearson's r $>$ 0.9) with MMMU-Pro accuracy using Best-of-N sampling with VL-GenRMs.Analysis experiments uncover three critical insights for improving VL-GenRMs: (i) models predominantly fail at basic visual perception tasks rather than reasoning tasks; (ii) inference-time scaling benefits vary dramatically by model capacity; and (iii) training VL-GenRMs to learn to judge substantially boosts judgment capability (+14.3\% accuracy for a 7B VL-GenRM).We believe VL-RewardBench along with the experimental insights will become a valuable resource for advancing VL-GenRMs.

</details>

---

## 162. COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training

- [ ] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training | https://cvpr.thecvf.com/virtual/2025/poster/33396

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33396

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) trained with contrastive loss have achieved significant advancements in various vision and language tasks. However, the global nature of contrastive loss makes VLMs focus predominantly on foreground objects, neglecting other crucial information in the image, which limits their effectiveness in downstream tasks. To address these challenges, we propose COSMOS: CrOSs-MOdality Self-distillation for vision-language pre-training that integrates novel text-cropping strategy and cross-attention module into self-supervised learning framework. We create global and local views of images and texts (i.e., multi-modal augmentations), which are essential for self-distillation in VLMs. We further introduce a cross-attention module, enabling COSMOS to learn comprehensive cross-modal representations optimized via a cross-modality self-distillation loss. COSMOS consistently outperforms previous strong baselines on various zero-shot downstream tasks including retrieval, classification, and semantic segmentation. Additionally, it surpasses CLIP-based models trained on larger datasets in visual perception and contextual understanding tasks.

</details>

---

## 163. Single Domain Generalization for Few-Shot Counting via Universal Representation Matching

- [ ] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching | https://cvpr.thecvf.com/virtual/2025/poster/33402

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33402

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Few-shot counting estimates the number of target objects in an image using only a few annotated exemplars. However, domain shift severely hinders existing methods to generalize to unseen scenarios.  This falls into the realm of single domain generalization that remains unexplored in few-shot counting. To solve this problem, we begin by analyzing the main limitations of current methods, which typically follow a standard pipeline that extract the object prototypes from exemplars and then match them with image feature to construct the correlation map. We argue that existing methods overlook the significance of learning highly generalized prototypes. Building on this insight, we propose the first domain generalization few-shot counter, Universal Representation Matching, termed URM. Our primary contribution is the discovery that incorporating universal vision-language representations distilled from a large scale pretrained vision-language model into the correlation construction process substantially improves robustness to domain shifts without compromising in domain performance. As a result, URM achieves state-of-the-art performance on both in domain and the newly introduced domain generalization setting.

</details>

---

## 164. Instruction-based Image Manipulation by Watching How Things Move

- [ ] Instruction-based Image Manipulation by Watching How Things Move | https://cvpr.thecvf.com/virtual/2025/poster/33406

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33406

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces a novel dataset construction pipeline that samples pairs of frames from videos and uses multimodal large language models (MLLMs) to generate editing instructions for training instruction-based image manipulation models. Video frames inherently preserve the identity of subjects and scenes, ensuring consistent content preservation during editing. Additionally, video data captures diverse, natural dynamics—such as non-rigid subject motion and complex camera movements—that are difficult to model otherwise, making it an ideal source for scalable dataset construction. Using this approach, we create a new dataset to train InstructMove, a model capable of instruction-based complex manipulations that are difficult to achieve with synthetically generated datasets. Our model demonstrates state-of-the-art performance in tasks such as adjusting subject poses, rearranging elements, and altering camera perspectives.

</details>

---

## 165. The Photographer's Eye: Teaching Multimodal Large Language Models to See, and Critique Like Photographers

- [ ] The Photographer's Eye: Teaching Multimodal Large Language Models to See, and Critique Like Photographers | https://cvpr.thecvf.com/virtual/2025/poster/33405

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33405

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Photographer, curator, and former director of photography at the Museum of Modern Art (MoMA), John Szarkowski remarked in William Eggleston’s Guide , “While editing directly from life, photographers have found it too difficult to see simultaneously both the blue and the sky.” Szarkowski insightfully revealed a notable gap between general and aesthetic visual understanding: while the former emphasizes identifying factual elements in an image (the sky), the latter transcends mere object identification, viewing it instead as an aesthetic component—a pure expanse of blue, valued purely as a color block in visual aesthetics. Such distinctions between general visual understanding (detection, localization, etc.) and aesthetic perception (color, lighting, composition, etc.) pose a significant challenge for existing Multimodal Large Language Models (MLLMs) in comprehending image aesthetics, which is increasingly needed in real-world applications, from image recommendation and enhancement to generation. To fundamentally advance the aesthetic understanding of MLLMs, we introduce a novel dataset, PhotoCritique, derived from extensive discussions among professional photographers and enthusiasts, distinguished by its large scale, expertise, and diversity. Additionally, we propose a new model, PhotoEye, an MLLM featuring a language-guided multi-view vision fusion mechanism for understanding image aesthetics from multiple perspectives. Finally, we introduce PhotoBench, a comprehensive and professional benchmark for aesthetic visual understanding. Our model demonstrates significant advantages over both open-source and commercial models on existing benchmarks and PhotoBench.

</details>

---

## 166. Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language

- [ ] Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language | https://cvpr.thecvf.com/virtual/2025/poster/33408

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33408

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models can generate realistic and diverse images, potentially facilitating data availability for data-intensive perception tasks.However, leveraging these models to boost performance on downstream tasks with synthetic data poses several challenges, including aligning with real data distribution, scaling synthetic sample volumes, and ensuring their quality. To bridge these gaps, we present Auto Cherry-Picker (ACP), a novel framework that generates high-quality cross-modality training samples at scale to augment perception and multi-modal training. ACP first uses LLMs to sample descriptions and layouts based on object combinations from real data priors, eliminating the need for ground truth image captions or annotations. Next, we use an off-the-shelf controllable diffusion model to generate multiple images. Then, the generated data are refined using a comprehensively designed metric, Composite Layout and Image Score (CLIS), to ensure quality. Our customized synthetic high-quality samples boost performance in various scenarios, especially in addressing challenges associated with long-tailed distribution and imbalanced datasets. Experiment results on downstream tasks demonstrate that ACP can significantly improve the performance of existing models. In addition, we find a positive correlation between CLIS and performance gains in downstream tasks. This finding shows the potential for evaluation metrics as the role for various visual perception and MLLM tasks. Code will be available.

</details>

---

## 167. LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant

- [ ] LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant | https://cvpr.thecvf.com/virtual/2025/poster/33401

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33401

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

First-person video assistants are highly anticipated to enhance our daily life through online video dialogue. However, existing online video assistants often sacrifice assistant efficacy for real-time efficiency by processing low-frame-rate videos with coarse-grained visual features. To overcome the trade-off between efficacy and efficiency, we propose " F ast & S low Video-Language Thinker" as on LI ne vide O assista N t, LION-FS , achieving real-time, proactive, temporally accurate, and contextually precise responses. LION-FS adopts a two-stage optimization strategy: 1) Fast Path: Routing-Based Response Determination evaluates frame-by-frame whether a immediate response is necessary. To enhance responses determination accuracy and handle higher frame-rate inputs efficiently, we employ Token Aggregation Routing to dynamically fuse spatiotemporal features without increasing token numbers, while utilizing Token Dropping Routing to eliminate redundant features, and 2) Slow Path: Multi-granularity Keyframe Augmentation optimizes keyframes during response generation. To provide comprehensive and detailed responses beyond atomic actions constrained by training data, fine-grained spatial features and human-environment interaction features are extracted through multi-granular pooling. They are further integrated into a meticulously designed multimodal Thinking Template to guide more precise response generation. Comprehensive evaluations on online video tasks demonstrate that LION-FS achieves state-of-the-art efficacy and efficiency. The codes will be released soon.

</details>

---

## 168. Global-Local Tree Search in VLMs for 3D Indoor Scene Generation

- [ ] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation | https://cvpr.thecvf.com/virtual/2025/poster/33423

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33423

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (VLMs), such as GPT-4, have achieved remarkable success across various fields. However, there are few work studies on 3D indoor scene generation with VLMs. This paper considers this task as a planning problem subject to spatial and layout common sense constraints. To solve the problem with a VLM, we propose a new global-local tree search algorithm. Globally, the method places each object sequentially and explores multiple placements during each placement process, where the problem space is presented as a tree. To reduce the depth of the tree, we decompose the scene structure hierarchically, \ie room level, region level, floor object level, and supported object level. The algorithm independently generates the floor objects in different regions and supported objects placed on different floor objects. Locally, we also decompose the sub-task, the placement of each object, into multiple steps. The algorithm searches the tree of problem space. To leverage the VLM model to produce positions of objects, we discrete the top-down view space as a dense grid and fill each cell with diverse emojis to make to cells distinct. We prompt the VLM with the emoji grid and the VLM produces a reasonable location for the object by describing the position with the name of emojis. The quantitative and qualitative experiments results illustrate our approach generates more plausible 3D scenes than state-of-the-art approaches. We will release our code and model.

</details>

---

## 169. SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling

- [ ] SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling | https://cvpr.thecvf.com/virtual/2025/poster/33431

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33431

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-world interpretation aims to accurately localize and recognize all objects within images by vision-language models (VLMs). While substantial progress has been made in this task for natural images, the advancements for remote sensing (RS) images still remain limited, primarily due to these two challenges. 1) Existing RS semantic categories are limited, particularly for pixel-level interpretation datasets. 2) Distinguishing among diverse RS spatial regions solely by language space is challenging due to the dense and intricate spatial distribution in open-world RS imagery. To address the first issue, we develop a fine-grained RS interpretation dataset, Sky-SA, which contains 183,375 high-quality local image-text pairs with full-pixel manual annotations, covering 1,763 category labels, exhibiting richer semantics and higher density than previous datasets. Afterwards, to solve the second issue, we introduce the vision-centric principle for vision-language modeling. Specifically, in the pre-training stage, the visual self-supervised paradigm is incorporated into image-text alignment, reducing the degradation of general visual representation capabilities of existing paradigms. Then, we construct a visual-relevance knowledge graph across open-category texts and further develop a novel vision-centric image-text contrastive loss for fine-tuning with text prompts. This new model, denoted as SkySense-O, demonstrates impressive zero-shot capabilities on a thorough evaluation encompassing 14 datasets over 4 tasks, from recognizing to reasoning and classification to localization. Specifically, it outperforms the latest models such as SegEarth-OV, GeoRSCLIP, and VHM by a large margin, i.e., 11.95\%, 8.04\% and 3.55\% on average respectively. We will release the dataset and model to facilitate future research.

</details>

---

## 170. MBQ: Modality-Balanced Quantization for Large Vision-Language Models

- [ ] MBQ: Modality-Balanced Quantization for Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33438

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33438

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have already enabled a variety of real-world applications. The large parameter size of VLMs brings large memory and computation overhead which poses significant challenges for deployment. Post-Training Quantization (PTQ) is an effective technique to reduce the memory and computation overhead. Existing PTQ methods mainly focus on the language modality in large language models (LLMs), without considering the differences across other modalities. In this paper, we discover that there is a significant difference in sensitivity between language and vision tokens in large VLMs. Therefore, treating tokens from different modalities equally, as in existing PTQ methods, may over-emphasize the insensitive modalities, leading to significant accuracy loss. To deal with the above issue, we propose a simple yet effective method, Modality-Balanced Quantization (MBQ), for large VLMs. Specifically, MBQ incorporates the different sensitivities across modalities during the calibration process to minimize the reconstruction loss for better quantization parameters. Extensive experiments show that MBQ can significantly improve task accuracy by up to 4.4% and 11.6% under W3 and W4A8 quantization for 7B to 70B VLMs, compared to SOTA baselines. Additionally, we implement a W3 GPU kernel that fuses the dequantization and GEMV operators, achieving a 1.4x speedup on LLaVA-onevision-7B on the RTX 4090. We will release the code.

</details>

---

## 171. HyperSeg: Hybrid Segmentation Assistant with Fine-grained Visual Perceiver

- [ ] HyperSeg: Hybrid Segmentation Assistant with Fine-grained Visual Perceiver | https://cvpr.thecvf.com/virtual/2025/poster/33435

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33435

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper aims to address universal segmentation for image and video perception with the strong reasoning ability empowered by Visual Large Language Models (VLLMs). Despite significant progress in current unified segmentation methods, limitations in adaptation to both image and video scenarios, as well as the complex reasoning segmentation, make it difficult for them to handle various challenging instructions and achieve accurate understanding of fine-grained visual text correlations. We propose HyperSeg, the first VLLM-based universal segmentation model for pixel-level image and video perception, encompassing generic segmentation tasks and more complex reasoning perception tasks requiring challenging reasoning abilities and world knowledge. Besides, to fully leverage the recognition capacity of VLLMs and the fine-grained visual information, HyperSeg incorporates hybrid entity recognition and fine-grained visual perceiver modules for distinct segmentation tasks. Combined with temporal adapter, HyperSeg chieves a comprehensive understanding of space-time information. Experimental results validate the effectiveness of our insights in resolving universal image and video segmentation tasks, including the more complex reasoning perception tasks.

</details>

---

## 172. CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation

- [ ] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation | https://cvpr.thecvf.com/virtual/2025/poster/33443

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33443

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Interleaved image-text generation has emerged as a crucial multimodal task, aiming at creating sequences of interleaved visual and textual content given a query. Despite notable advancements in recent multimodal large language models (MLLMs), generating integrated image-text sequences that exhibit narrative coherence and entity and style consistency remains challenging due to poor training data quality. To address this gap, we introduce CoMM, a high-quality Coherent interleaved image-text MultiModal dataset designed to enhance the coherence, consistency, and alignment of generated multimodal content. Initially, CoMM harnesses raw data from diverse sources, focusing on instructional content and visual storytelling, establishing a foundation for coherent and consistent content. To further refine the data quality, we devise a multi-perspective filter strategy that leverages advanced pre-trained models to ensure the development of sentences, consistency of inserted images, and semantic alignment between them. Various quality evaluation metrics are designed to prove the high quality of the filtered dataset. Meanwhile, extensive few-shot experiments on various downstream tasks demonstrate CoMM's effectiveness in significantly enhancing the in-context learning capabilities of MLLMs. Moreover, we propose four new tasks to evaluate MLLMs' interleaved generation abilities, supported by a comprehensive evaluation framework. We believe CoMM opens a new avenue for advanced MLLMs with superior multimodal in-context learning and understanding ability. The dataset and codes will be released.

</details>

---

## 173. HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding

- [ ] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33455

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33455

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite advancements in multimodal large language models (MLLMs), current approaches struggle in medium-to-long video understanding due to frame and context length limitations. As a result, these models often depend on frame sampling, which risks missing key information over time and lacks task-specific relevance. To address these challenges, we introduce HierarQ , a task-aware hierarchical Q-Former based framework that sequentially processes frames to bypass the need for frame sampling, while avoiding LLM's context length limitations. We introduce a lightweight two-stream language-guided feature modulator to incorporate task awareness in video understanding, with the entity stream capturing frame-level object information within a short context and the scene stream identifying their broader interactions over longer period of time. Each stream is supported by dedicated memory banks which enables our proposed Hierar chical Q uerying transformer (HierarQ) to effectively capture short and long-term context. Extensive evaluations on 10 video benchmarks across video understanding, question answering, and captioning tasks demonstrate HierarQ’s state-of-the-art performance across most datasets, proving its robustness and efficiency for comprehensive video analysis. All code will be made available upon acceptance.

</details>

---

## 174. Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation

- [ ] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation | https://cvpr.thecvf.com/virtual/2025/poster/33463

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33463

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Uncertainty quantification in text-to-image (T2I) generative models is crucial for understanding model behavior and improving output reliability. In this paper, we are the first to quantify and evaluate the uncertainty of T2I models with respect to the prompt. Alongside adapting existing approaches designed to measure uncertainty in the image space, we also introduce Prompt-based UNCertainty Estimation for T2I models (PUNC), a novel method leveraging Large Vision-Language Models (LVLMs) to better address uncertainties arising from the semantics of the prompt and generated images. PUNC utilizes a LVLM to caption a generated image, and then compares the caption with the original prompt in the more semantically meaningful text space. PUNC also enables the disentanglement of both aleatoric and epistemic uncertainties via precision and recall, which image-space approaches are unable to do. Extensive experiments demonstrate that PUNC outperforms state-of-the-art uncertainty estimation techniques across various settings. Uncertainty quantification in text-to-image generation models can be used on various applications including bias detection, copyright protection, and OOD detection. We also introduce a comprehensive dataset of text prompts and generation pairs to foster further research in uncertainty quantification for generative models. Our findings illustrate that PUNC not only achieves competitive performance but also enables novel applications in evaluating and improving the trustworthiness of text-to-image models.

</details>

---

## 175. Perception Tokens Enhance Visual Reasoning in Multimodal Language Models

- [ ] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33465

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33465

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal language models (MLMs) continue to struggle with fundamental visual perception tasks that specialist models excel at tasks that require reasoning over 3D benefit from depth estimation; similarly, reasoning 2D over object instances benefits from object detection. Yet, MLMs can not produce intermediate depth or boxes to reason over.Finetuning MLMs on relevant data doesn't generalize well and outsourcing computation to specialized vision tools is too compute-intensive and memory-inefficient.To address this, we introduce Perception Tokens, intrinsic image representations designed to assist reasoning tasks where language is insufficient. Perception tokens act as auxiliary reasoning tokens, akin to chain-of-thought prompts in language models. For example, in a depth-related task, an MLM augmented with perception tokens can reason by generating a depth map as tokens, enabling it to solve the problem effectively.We propose Aurora, a training method that augments MLMs with perception tokens for improved reasoning over visual inputs. Aurora employs a VQVAE to transform intermediate image representations, such as depth maps or bounding boxes, into a tokenized format, which is then used in a multi-task training framework.Aurora achieves notable improvements across counting benchmarks: $+10.8\%$ on BLINK, $+11.3\%$ on CVBench, and $+8.3\%$ on SEED-Bench, outperforming finetuning approaches in generalization across datasets. It also improves on relative depth: over $+6\%$ on BLINK.With perception tokens, Aurora expands the scope of MLMs beyond language-based reasoning, paving the way for more effective visual reasoning capabilities.

</details>

---

## 176. ShowUI: One Vision-Language-Action Model for GUI Visual Agent

- [ ] ShowUI: One Vision-Language-Action Model for GUI Visual Agent | https://cvpr.thecvf.com/virtual/2025/poster/33472

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33472

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Building Graphical User Interface (GUI) assistants holds significant promise for enhancing human workflow productivity. While most agents are language-based, relying on closed-source API with text-rich meta-information (e.g., HTML or accessibility tree), they show limitations in perceiving UI visuals as humans do, highlighting the need for GUI visual agents.In this work, we develop a vision-language-action model in the digital world, namely "Our Model," which features the following innovations:1. UI-Guided Visual Token Selection to reduce computational costs by formulating screenshots as a UI-connected graph, adaptively identifying their redundant relationships and serving as the criteria for token selection during self-attention blocks.  2. Interleaved Vision-Language-Action Streaming that flexibly unifies diverse needs within GUI tasks, enabling effective management of visual-action history in navigation or pairing multi-turn query-action sequences per screenshot to enhance training efficiency.  3. Small-Scale High-Quality GUI Instruction-Following Datasets by careful data curation and employing a resampling strategy to address significant data type imbalances.  With the above components, our model, a lightweight 2B model using 256K data, achieves a strong 75.1% accuracy in zero-shot screenshot grounding. Its UI-guided token selection further reduces 33% of redundant visual tokens during training and speeds up performance by 1.4×. Navigation experiments across web, mobile, and online environments further underscore the effectiveness and potential of our model in advancing GUI visual agents.

</details>

---

## 177. DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment

- [ ] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment | https://cvpr.thecvf.com/virtual/2025/poster/33482

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33482

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Self-supervised visual foundation models produce powerful embeddings that achieve remarkable performance on a wide range of downstream tasks. However, unlike vision-language models such as CLIP, self-supervised visual features are not readily aligned with language, hindering their adoption in open-vocabulary tasks. Our method, named dino.txt , unlocks this new ability for DINOv2, a widely used self-supervised visual encoder. We build upon the LiT training strategy, which trains a text encoder to align with a frozen vision model, but leads to unsatisfactory results on dense tasks. We propose several key ingredients to improve performance on both global and dense tasks,such as concatenating the [CLS] token with the patch average to train the alignment, curating data using both text and image modalities. With these, we successfully train a CLIP-like model with only a fraction of the computational cost compared to CLIP while achieving state-of-the-art results in zero-shot classification and open-vocabulary semantic segmentation.

</details>

---

## 178. METASCENES: Towards Automated Replica Creation for Real-world 3D Scans

- [ ] METASCENES: Towards Automated Replica Creation for Real-world 3D Scans | https://cvpr.thecvf.com/virtual/2025/poster/33486

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33486

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Embodied AI (EAI) research depends on high-quality and diverse 3D scenes to enable effective skill acquisition, sim-to-real transfer, and domain generalization. Recent 3D scene datasets are still limited in scalability due to their dependence on artist-driven designs and challenges in replicating the diversity of real-world objects. To address these limitations and automate the creation of 3D simulatable scenes, we present METASCENES, a large-scale 3D scene dataset constructed from real-world scans. It features 706 scenes with 15366 objects across a wide array of types, arranged in realistic layouts, with visually accurate appearances and physical plausibility. Leveraging the recent advancements in object-level modeling, we provide each object with a curated set of candidates, ranked through human annotation for optimal replacements based on geometry, texture, and functionality. These annotations enable a novel multi-modal alignment model, SCAN2SIM, which facilitates automated and high-quality asset replacement. We further validate the utility of our dataset with two benchmarks: Micro-Scene Synthesos for small object layout generation and cross-domain vision-language navigation (VLN). Results confirm the potential of METASCENES to enhance EAI by supporting more generalizable agent learning and sim-to-real applications, introducing new possibilities for EAI research.

</details>

---

## 179. The Devil is in Temporal Token: High Quality Video Reasoning Segmentation

- [ ] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/33488

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33488

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing methods for Video Reasoning Segmentation rely heavily on a single special token to represent the object in the keyframe or the entire video, inadequately capturing spatial complexity and inter-frame motion. To overcome these challenges, we propose VRS-HQ, an end-to-end video reasoning segmentation approach that leverages Multimodal Large Language Models (MLLMs) to inject rich spatiotemporal features into hierarchical tokens. Our key innovations include a Temporal Dynamic Aggregation (TDA) and a Token-driven Keyframe Selection (TKS). Specifically, we design frame-level and temporal-level tokens that utilize MLLM’s autoregressive learning to effectively capture both local and global information. Subsequently, we apply a similarity-based weighted fusion and frame selection strategy, then utilize SAM2 to perform keyframe segmentation and propagation. To enhance keyframe localization accuracy, the TKS filters keyframes based on SAM2’s occlusion scores during inference. VRS-HQ achieves state-of-the-art performance on ReVOS, surpassing VISA  by 5.9%/12.5%/9.1% in J&F scores across the three subsets. These results highlight the strong temporal reasoning and segmentation capabilities of our method. Code and model weights will be made publicly available.

</details>

---

## 180. One-Minute Video Generation with Test-Time Training

- [ ] One-Minute Video Generation with Test-Time Training | https://cvpr.thecvf.com/virtual/2025/poster/33506

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33506

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a novel framework for generating long-form cartoon videos, specifically focusing on recreating the classic "Tom and Jerry" series. While recent advances in video generation have shown promising results for short clips, generating long videos with coherent storylines and dynamic motions remains challenging with high computation costs. We propose a hybrid framework that combines local self-attention with a Test-Time Training (TTT) based global attention mechanism, enabling our model to process and maintain consistency across significantly longer temporal context windows. We develop a new dataset curation pipeline specifically designed for long-form cartoon videos, combining human annotations for complex motion dynamics with Vision-Language Models for detailed descriptions. Our pipeline captures the exaggerated movements and dynamic camera work characteristic of "Tom and Jerry". Experiments show that our approach outperforms existing methods in generating long-form animated content with plausible motion and consistent storylines.

</details>

---

## 181. Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models

- [ ] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33505

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33505

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present an efficient encoder-free approach for video-language understanding that achieves competitive performance while significantly reducing computational overhead. Current video-language models typically rely on heavyweight image encoders (300M-1.1B parameters) or video encoders (1B-1.4B parameters), creating a substantial computational burden when processing multi-frame videos. Our method introduces a novel Spatio-Temporal Alignment Block (STAB) that directly processes video inputs without requiring pre-trained encoders while using only 45M parameters for visual processing - at least a 6.5$\times$ reduction compared to traditional approaches. The STAB architecture combines Local Spatio-Temporal Encoding for fine-grained feature extraction, efficient spatial downsampling through learned attention and separate mechanisms for modeling frame-level and video-level relationships. Our model achieves comparable or superior performance to encoder-based approaches for open-ended video question answering on standard benchmarks. The fine-grained video question-answering evaluation demonstrates our model's effectiveness, outperforming the encoder-based approaches Video-ChatGPT and Video-LLaVA in key aspects like correctness and temporal understanding. Extensive ablation studies validate our architectural choices and demonstrate the effectiveness of our spatio-temporal modeling approach while achieving 3-4$\times$ faster processing speeds than previous methods.

</details>

---

## 182. DriveGPT4-V2: Harnessing Large Language Model Capabilities for Enhanced Closed-Loop Autonomous Driving

- [ ] DriveGPT4-V2: Harnessing Large Language Model Capabilities for Enhanced Closed-Loop Autonomous Driving | https://cvpr.thecvf.com/virtual/2025/poster/33516

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33516

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) possess the ability to comprehend visual images or videos, and show impressive reasoning ability thanks to the vast amounts of pretrained knowledge, making them highly suitable for autonomous driving applications. Unlike the previous work, DriveGPT4-V1, which focused on open-loop tasks, this study explores the capabilities of LLMs in enhancing closed-loop autonomous driving. DriveGPT4-V2 processes camera images and vehicle states as input to generate low-level control signals for end-to-end vehicle operation. A high-resolution visual tokenizer (HR-VT) is employed enabling DriveGPT4-V2 to perceive the environment with an extensive range while maintaining critical details. The model architecture has been refined to improve decision prediction and inference speed. To further enhance the performance, an additional expert LLM is trained for online imitation learning. The expert LLM, sharing a similar structure with DriveGPT4-V2, can access privileged information about surrounding objects for more robust and reliable predictions. Experimental results show that DriveGPT4-V2 significantly outperforms all baselines on the challenging CARLA Longest6 benchmark. The code and data of DriveGPT4-V2 will be publicly available.

</details>

---

## 183. Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion

- [ ] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion | https://cvpr.thecvf.com/virtual/2025/poster/33518

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33518

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present Florence-VL, a new family of multimodal large language models (MLLMs) with enriched visual representations produced by Florence-2, a generative vision foundation model. Unlike the widely used CLIP-style vision transformer trained by contrastive learning, Florence-2 can capture different levels and aspects of visual features, which are more versatile to be adapted to diverse downstream tasks. We propose a novel feature-fusion architecture and an innovative training recipe that effectively integrates Florence-2's visual features into pretrained LLMs, such as Phi 3.5 and LLama 3. In particular, we propose ``depth-breath fusion (DBFusion)'' to fuse the visual features extracted from different depths and under multiple prompts. Our model training is composed of end-to-end pretraining of the whole model followed by finetuning of the projection layer and the LLM, on a carefully designed recipe of diverse open-source datasets that include high-quality image captions and instruction-tuning pairs. Our quantitative analysis and visualization of Florence-VL's visual features show its advantages over popular vision encoders on vision-language alignment, where the enriched depth and breath play important roles. Florence-VL achieves significant improvements over existing state-of-the-art MLLMs across various multi-modal and vision-centric benchmarks covering general VQA, perception, hallucination, OCR, Chart, knowledge-intensive understanding, etc. To facilitate future research, our models and the complete training recipe are open-sourced.

</details>

---

## 184. VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models

- [ ] VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33523

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33523

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of vision-language models (VLMs) has established a new paradigm in video anomaly detection (VAD): leveraging VLMs to simultaneously detect anomalies and provide comprehendible explanations for the decisions. Existing work in this direction often assumes the complex reasoning required for VAD exceeds the capabilities of pretrained VLMs. Consequently, these approaches either incorporate specialized reasoning modules during inference or rely on instruction tuning datasets through additional training to adapt VLMs for VAD. However, such strategies often incur substantial computational costs or data annotation overhead. To address these challenges in explainable VAD, we introduce a verbalized learning framework named VERA that enables VLMs to perform  VAD without model parameter modifications. Specifically, VERA automatically decomposes the complex reasoning required for VAD into reflections on simpler, more focused guiding questions capturing distinct abnormal patterns. It treats these reflective questions as learnable parameters and optimizes them through data-driven verbal interactions between learner and optimizer VLMs, using coarsely labeled training data. During inference, VERA embeds the learned questions into model prompts to guide VLMs in generating segment-level anomaly scores, which are then refined into frame-level scores via the fusion of scene and temporal contexts. Experimental results on challenging benchmarks demonstrate that the learned questions of VERA are highly adaptable, significantly improving both detection performance and explainability of VLMs for VAD.

</details>

---

## 185. SF2T: Self-supervised Fragment Finetuning of Video-LLMs for Fine-Grained Understanding

- [ ] SF2T: Self-supervised Fragment Finetuning of Video-LLMs for Fine-Grained Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33524

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33524

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video-based Large Language Models (Video-LLMs) have witnessed substantial advancements in recent years, propelled by the advancement in multi-modal LLMs. Although these models have demonstrated proficiency in providing the overall description of videos, they struggle with fine-grained understanding, particularly in aspects such as visual dynamics and video details inquiries. To tackle these shortcomings, we find that fine-tuning Video-LLMs on self-supervised fragment tasks, greatly improve their fine-grained video understanding abilities. Hence we propose two key contributions:(1) Self-Supervised Fragment Fine-Tuning (SF$^2$T), a novel effortless fine-tuning method, employs the rich inherent characteristics of videos for training, while unlocking more fine-grained understanding ability of Video-LLMs. Moreover, it relieves researchers from labor-intensive annotations and smartly circumvents the limitations of natural language, which often fails to capture the complex spatiotemporal variations in videos;(2) A novel benchmark dataset, namely FineVidBench, for rigorously assessing Video-LLMs' performance at both the scene and fragment levels, offering a comprehensive evaluation of their capabilities.We assessed multiple models and validated the effectiveness of SF$^2$T on them. Experimental results reveal that our approach improves their ability to capture and interpret spatiotemporal details.

</details>

---

## 186. BACON: Improving Clarity of Image Captions via Bag-of-Concept Graphs

- [ ] BACON: Improving Clarity of Image Captions via Bag-of-Concept Graphs | https://cvpr.thecvf.com/virtual/2025/poster/33526

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33526

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Advancements in large Vision-Language Models have brought precise, accurate image captioning, vital for advancing multi-modal image understanding and processing. Yet these captions often carry lengthy, intertwined contexts that are difficult to parse and frequently overlook essential cues, posing a great barrier for models like GroundingDINO and SDXL, which lack the strong text encoding and syntax analysis needed to fully leverage dense captions.To address this, we propose BACON, a prompting method that breaks down VLM-generated captions into disentangled, structured elements such as objects, relationships, styles, and themes. This approach not only minimizes confusion from handling complex contexts but also allows for efficient transfer into a JSON dictionary, enabling models without linguistic processing capabilities to easily access key information.We annotated 100,000 image-caption pairs using BACON with GPT-4V and trained an LLaVA captioner on this dataset, enabling it to produce BACON-style captions without relying on costly GPT-4V resources. Evaluations of overall quality, precision, and recall—as well as user studies—demonstrate that the resulting caption model consistently outperforms other state-of-the-art VLM models in generating high-quality captions.Additionally, we show that BACON-style captions exhibit better clarity when applied to various models, enabling them to accomplish previously unattainable tasks or surpass existing SOTA solutions without training. For example, BACON-style captions help groundingDINO achieve 1.51 times higher recall scores on open-vocabulary object detection tasks compared to leading methods.

</details>

---

## 187. HoVLE: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding

- [ ] HoVLE: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding | https://cvpr.thecvf.com/virtual/2025/poster/33530

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33530

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advance of Large Language Models (LLMs) has catalyzed the development of Vision-Language Models (VLMs). Monolithic VLMs, which avoid modality-specific encoders, offer a promising alternative to the compositional ones but face the challenge of inferior performance. Most existing monolithic VLMs require tuning pre-trained LLMs to acquire vision abilities, which may degrade their language capabilities. To address this dilemma, this paper presents a novel high-performance monolithic VLM named HoVLE. We note that LLMs have been shown capable of interpreting images, when image embeddings are aligned with text embeddings. The challenge for current monolithic VLMs actually lies in the lack of a holistic embedding module for both vision and language inputs. Therefore, HoVLE introduces a holistic embedding module that converts visual and textual inputs into a shared space, allowing LLMs to process images in the same way as texts. Furthermore, a multi-stage training strategy is carefully designed to empower the holistic embedding module. It is first trained to distill visual features from a pre-trained vision encoder and text embeddings from the LLM, enabling large-scale training with unpaired random images and text tokens. The whole model further undergoes next-token prediction on multi-modal data to align the embeddings. Finally, an instruction-tuning stage is incorporated. Our experiments show that HoVLE achieves performance close to leading compositional models on various benchmarks, outperforming previous monolithic models by a large margin.

</details>

---

## 188. Probabilistic Prompt Distribution Learning for Animal Pose Estimation

- [ ] Probabilistic Prompt Distribution Learning for Animal Pose Estimation | https://cvpr.thecvf.com/virtual/2025/poster/33529

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33529

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-species animal pose estimation has emerged as a challenging yet critical task, hindered by substantial visual diversity and uncertainty. This paper challenges the problem by efficient prompt learning for Vision-Language Pretrained (VLP) models, e.g. CLIP, aiming to resolve the cross-species generalization problem. At the core of the solution lies in the prompt designing, probabilistic prompt modeling and cross-modal adaptation, thereby enabling prompts to compensate for cross-modal information and effectively overcome large data variances under unbalanced data distribution. To this end, we propose a novel probabilistic prompting approach to fully explore textual descriptions, which could alleviate the diversity issues caused by long-tail property and increase the adaptability of prompts on unseen category instance. Specifically, we first introduce a set of learnable prompts and propose a diversity loss to maintain distinctiveness among prompts, thus representing diverse image attributes. Diverse textual probabilistic representations are sampled and used as the guidance for the pose estimation. Subsequently, we explore three different cross-modal fusion strategies at spatial level to alleviate the adverse impacts of visual uncertainty. Extensive experiments on multi-species animal pose benchmarks show that our method achieves the state-of-the-art performance under both supervised and zero-shot settings.

</details>

---

## 189. FLAIR: VLM with Fine-grained Language-informed Image Representations

- [ ] FLAIR: VLM with Fine-grained Language-informed Image Representations | https://cvpr.thecvf.com/virtual/2025/poster/33533

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33533

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

CLIP has shown impressive results in aligning images and text at scale. However, its ability to capture detailed visual features remains limited because CLIP matches images and texts at a global level. To address this issue, we propose FLAIR, Fine-grained Language-informed Image Representations, an approach that utilizes long and detailed image descriptions to learn localized image embeddings. By sampling diverse sub-captions that describe fine-grained details about an image, we train our vision-language model to produce not only global embeddings but also text-specific image representations. Our model introduces text-conditioned attention pooling on top of local image tokens to produce fine-grained image representations that excel at retrieving detailed image content. We achieve state-of-the-art performance on both, existing multimodal retrieval benchmarks, as well as, our newly introduced fine-grained retrieval task which evaluates vision-language models' ability to retrieve partial image content. Furthermore, our experiments demonstrate the effectiveness of FLAIR trained on 30M image-text pairs in capturing fine-grained visual information, including zero-shot semantic segmentation, outperforming models trained on billions of pairs. Code and model checkpoints will be released upon acceptance.

</details>

---

## 190. FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models

- [ ] FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models | https://cvpr.thecvf.com/virtual/2025/poster/33537

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33537

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical vision-language model models often struggle with generating accurate quantitative measurements in radiology reports, leading to hallucinations that undermine clinical reliability. We introduce FactCheXcker, a modular framework that de-hallucinates radiology report measurements by leveraging an improved query-code-update paradigm. Specifically, FactCheXcker employs specialized modules and the code generation capabilities of large language models to solve measurement queries generated based on the original report.After extracting measurable findings, the results are incorporated into an updated report. We evaluate FactCheXcker on endotracheal tube placement, which accounts for an average of 78\% of report measurements, using the MIMIC-CXR dataset and 11 medical report-generation models. Our results show that FactCheXcker significantly reduces hallucinations, improves measurement precision, and maintains the quality of the original reports. Specifically, FactCheXcker improves the performance of all 11 models and achieves an average improvement of 94.0\% in reducing measurement hallucinations measured by mean absolute error.

</details>

---

## 191. VideoGLaMM : A Large Multimodal Model for Pixel-Level Visual Grounding in Videos

- [ ] VideoGLaMM : A Large Multimodal Model for Pixel-Level Visual Grounding in Videos | https://cvpr.thecvf.com/virtual/2025/poster/33544

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33544

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fine-grained alignment between videos and text is challenging due to complex spatial and temporal dynamics in videos. Existing video-based Large Multimodal Models (LMMs) handle basic conversations but struggle with precise pixel-level grounding in videos. To address this, we introduce VideoGLaMM, a LMM designed for fine-grained pixel-level grounding in videos based on user-provided textual inputs. Our design seamlessly connects three key components: a Large Language Model, a dual vision encoder that emphasizes both spatial and temporal details, and a spatio-temporal decoder for accurate mask generation. This connection is facilitated via tunable  V→L and L→V adapters that enable close Vision-Language (VL) alignment. The architecture is trained to synchronize both spatial and temporal elements of video content with textual instructions. To enable fine-grained grounding, we curate a multimodal dataset featuring detailed visually-grounded conversations using a semiautomatic annotation pipeline, resulting in a diverse set of 38k video-QA triplets along with 83k objects and 671k masks. We evaluate VideoGLaMM on three challenging tasks: Grounded Conversation Generation, Visual Grounding, and Referring Video Segmentation. Experimental results show that our model consistently outperforms existing approaches across all three tasks.

</details>

---

## 192. HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator

- [ ] HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator | https://cvpr.thecvf.com/virtual/2025/poster/33551

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33551

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

AIGC images are prevalent across various fields, yet they frequently suffer from quality issues like artifacts and unnatural textures. Specialized models aim to predict defect region heatmaps but face two primary challenges: (1) lack of explainability, failing to provide reasons and analyses for subtle defects, and (2) inability to leverage common sense and logical reasoning, leading to poor generalization. Multimodal large language models (MLLMs) promise better comprehension and reasoning but face their own challenges: (1) difficulty in fine-grained defect localization due to the limitations in capturing tiny details; and (2) constraints in providing pixel-wise outputs necessary for precise heatmap generation.To address these challenges, we propose HEIE: a novel MLLM-Based Hierarchical Explainable image Implausibility Evaluator. We introduce the CoT-Driven Explainable Trinity Evaluator, which integrates heatmaps, scores, and explanation outputs, using CoT to decompose complex tasks into subtasks of increasing difficulty and enhance interpretability. Our Adaptive Hierarchical Implausibility Mapper synergizes low-level image features with high-level mapper tokens from LLMs, enabling precise local-to-global hierarchical heatmap predictions through an uncertainty-based adaptive token approach.Moreover, we propose a new dataset: Expl-AIGI-Eval, designed to facilitate interpretable implausibility evaluation of AIGC images. Our method demonstrates state-of-the-art performance through extensive experiments. Our dataset and code are included in the supplementary materials and will be publicly available.

</details>

---

## 193. Magma: A Foundation Model for Multimodal AI Agents

- [ ] Magma: A Foundation Model for Multimodal AI Agents | https://cvpr.thecvf.com/virtual/2025/poster/33563

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33563

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a new foundation model, called Magma, for multimodal AI agents in both the digital and physical worlds.  Magma is a significant extension of vision-language (VL) models in that the former not only retains the VL understanding ability (verbal intelligence) of the latter, but is also equipped with the ability to plan and act in the visual-spatial world (spatial intelligence) to complete agentic tasks ranging from UI navigation to robot manipulation. Magma is pre-trained on large amounts of heterogeneous VL datasets, where the actionable visual objects (e.g., clickable buttons in GUI) in images are labeled by Set of Marks (SoM) and the object movements (e.g., the trace of a robotic arm) in videos are labeled by Trace of Mark (ToM). Evaluation shows that SoM and ToM facilitate acquisition of spatial intelligence from training data. Magma creates new state-of-the-art results on UI navigation and robotic manipulation tasks, outperforming previous models that are tailored specifically to these tasks. On VL tasks, Magma also compares favorably to popular VL models that are trained on much larger datasets.

</details>

---

## 194. A Simple yet Effective Layout Token in Large Language Models for Document Understanding

- [ ] A Simple yet Effective Layout Token in Large Language Models for Document Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33577

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33577

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent methods that integrate spatial layouts with text for document understanding in large language models (LLMs) have shown promising results. A commonly used method is to represent layout information as text tokens and interleave them with text content as inputs to the LLMs. However, such a method still demonstrates limitations, as it requires additional position IDs for tokens that are used to represent layout information. Due to the constraint on max position IDs, assigning them to layout information reduces those available for text content, reducing the capacity for the model to learn from the text during training, while also introducing a large number of potentially untrained position IDs during long-context inference, which can hinder performance on document understanding tasks.To address these issues, we propose LayTokenLLM, a simple yet effective method for document understanding. LayTokenLLM represents layout information as a single token per text segment and uses a specialized positional encoding scheme. It shares position IDs between text and layout tokens, eliminating the need for additional position IDs. This design maintains the model's capacity to learn from text while mitigating long-context issues during inference. Furthermore, a novel pre-training objective called Next Interleaved Text and Layout Token Prediction (NTLP) is devised to enhance cross-modality learning between text and layout tokens. Extensive experiments show that LayTokenLLM outperforms existing layout-integrated LLMs and MLLMs of similar scales on multi-page document understanding tasks, while also achieving superior performance on most single-page tasks. Code and data will be publicly available.

</details>

---

## 195. AdaDARE-gamma: Balancing Stability and Plasticity in Multi-modal LLMs through Efficient Adaptation

- [ ] AdaDARE-gamma: Balancing Stability and Plasticity in Multi-modal LLMs through Efficient Adaptation | https://cvpr.thecvf.com/virtual/2025/poster/33585

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33585

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adapting Multi-modal Large Language Models (MLLMs) to target tasks often suffers from catastrophic forgetting, where acquiring new task-specific knowledge compromises performance on pre-trained tasks. In this paper, we introduce AdaDARE-$\gamma$, an efficient approach that alleviates catastrophic forgetting by controllably injecting new task-specific knowledge through adaptive parameter selection from fine-tuned models without requiring retraining procedures. This approach consists two key innovations: (1) an adaptive parameter selection mechanism that identifies and retains the most task-relevant parameters from fine-tuned models, and (2) a controlled task-specific information injection strategy that precisely balances the preservation of pre-trained knowledge with the acquisition of new capabilities. Theoretical analysis proves the optimality of our parameter selection strategy and establishes bounds for the task-specific information injection factor. Extensive experiments on InstructBLIP and LLaVA-1.5 across image captioning and visual question answering tasks demonstrate that AdaDARE-$\gamma$ establishes new state-of-the-art results in balancing model performance. Specifically, it maintains 98.2\% of pre-training effectiveness on original tasks while achieving 98.7\% of standard fine-tuning performance on target tasks.

</details>

---

## 196. Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning

- [ ] Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning | https://cvpr.thecvf.com/virtual/2025/poster/33587

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33587

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning vision-language models (VLMs) with large amounts of unlabeled data has recently garnered significant interest. However, a key challenge remains the lack of high-quality pseudo-labeled data. Current pseudo-labeling strategies often struggle with mismatches between semantic and visual information, leading to sub-optimal performance of unsupervised prompt learning (UPL) methods.In this paper, we introduce a simple yet effective approach called \textbf{A}ugmenting D\textbf{i}scriminative \textbf{R}ichness via Diffusions (AiR), toward learning a richer discriminating way to represent the class comprehensively and thus facilitate classification.Specifically, our approach includes a pseudo-label generation module that leverages high-fidelity synthetic samples to create an auxiliary classifier, which captures richer visual variation, bridging text-image-pair classification to a more robust image-image-pair classification. Additionally, we exploit the diversity of diffusion-based synthetic samples to enhance prompt learning, providing greater information for semantic-visual alignment.Extensive experiments on five public benchmarks, including RESISC45 and Flowers102, and across three learning paradigms-UL, SSL, and TRZSL-demonstrate that AiR achieves substantial and consistent performance improvements over state-of-the-art unsupervised prompt learning methods.

</details>

---

## 197. HD-EPIC: A Highly-Detailed Egocentric Video Dataset

- [ ] HD-EPIC: A Highly-Detailed Egocentric Video Dataset | https://cvpr.thecvf.com/virtual/2025/poster/33586

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33586

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a validation dataset of newly-collected kitchen based egocentric videos, manually annotated with highly detailed and interconnected ground-truth labels covering: recipe steps, fine-grained actions, ingredients with nutritional values, moving objects, and audio annotations. Importantly, all annotations are grounded in 3D through digital twinning of the scene, fixtures, object locations, and primed with gaze. Footage is collected from unscripted recordings in diverse home environments, making HD-EPIC the first dataset collected in-the-wild but with detailed annotations matching those in controlled lab environments. We show the potential of our highly-detailed annotations through a challenging VQA benchmark of 26K questions assessing capability to recognise recipes, ingredients, nutrition, fine-grained actions, 3D perception, object motion, and gaze direction. The powerful long-context Gemini Pro only achieves 37.0% on this benchmark, showcasing its difficulty and highlighting shortcomings in current VLMs. We additionally assess action recognition, sound recognition, and long-term video-object segmentation on HD-EPIC. HD-EPIC is 41 hours of video in 9 kitchens with digital twins of 404 kitchen fixtures, capturing 69 recipes, 59K fine-grained actions, 51K audio events, 20K object movements and 37K object masks lifted to 3D. On average, we have 263 annotations per min of our unscripted videos.

</details>

---

## 198. Semantic and Sequential Alignment for Referring Video Object Segmentation

- [ ] Semantic and Sequential Alignment for Referring Video Object Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/33595

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33595

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring video object segmentation (RVOS) aims to segment the objects within a video referred by linguistic expressions.  Existing RVOS solutions follow a "fuse then select" paradigm:  establishing semantic correlation between visual and linguistic feature, and performing frame-level query interaction to select the instance mask per frame with instance segmentation module. This paradigm overlooks the challenge of semantic gap between the linguistic descriptor and the video object as well as the underlying clutters in the video. This paper proposes a novel Semantic and Sequential Alignment (SSA) paradigm to handle these challenges. We first insert a light adapter after the vision language model (VLM) to perform the semantic alignment. Then, prior to selecting mask per frame, we exploit the trajectory-to-instance enhancement for each frame via sequential alignment.  This paradigm reuses the visual-language alignment of VLM during adaptation and tries to capture global information by ensembling trajectories. This helps understand videos and the corresponding descriptors by bridging the gap with complex activity semantics, particularly when facing occlusion or similar interference. SSA demonstrates competitive performance  while maintaining remarkably low computational costs. Code is available at https://github.com/anonymous61888/SSA.

</details>

---

## 199. ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models

- [ ] ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33610

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33610

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision Language Models (LVLMs) have achieved significant success across multi-modal tasks. However, the computational cost of processing long visual tokens can be prohibitively expensive on resource-limited devices. Previous methods have identified redundancy in visual tokens within the Large Language Model (LLM) decoder layers and have mitigated this by pruning tokens using a pre-defined or fixed ratio, thereby reducing computational overhead. Nonetheless, we observe that the impact of pruning ratio varies across different LLM layers and instances (image-prompt pairs). Therefore, it is essential to develop a layer-wise and instance-wise vision token pruning strategy to balance computational cost and model performance effectively. We propose ATP-LLaVA, a novel approach that adaptively determines instance-specific token pruning ratios for each LLM layer. Specifically, we introduce an Adaptive Token Pruning (ATP) module, which computes the importance score and pruning threshold based on input instance adaptively. The ATP module can be seamlessly integrated between any two LLM layers with negligible computational overhead. Additionally, we develop a Spatial Augmented Pruning (SAP) strategy that prunes visual tokens with both token redundancy and spatial modeling perspectives. Our approach reduces the average token count by 75% while maintaining performance, with only a minimal 1.9% degradation across seven widely used benchmarks.

</details>

---

## 200. Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves

- [ ] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves | https://cvpr.thecvf.com/virtual/2025/poster/33614

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33614

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning (PT) has long been recognized as an effective and efficient paradigm for transferring large pre-trained vision-language models (VLMs) to downstream tasks by learning a tiny set of context vectors. Nevertheless, in this work, we reveal that freezing the parameters of VLMs during learning the context vectors neither facilitates the transferability of pre-trained knowledge nor improves the memory and time efficiency significantly. Upon further investigation, we find that reducing both the length and width of the feature-gradient propagation flows of the full fine-tuning (FT) baseline is key to achieving effective and efficient knowledge transfer. Motivated by this, we propose Skip Tuning, a novel paradigm for adapting VLMs to downstream tasks. Unlike existing PT or adapter-based methods, Skip Tuning applies Layer-wise Skipping (LSkip) and Class-wise Skipping (CSkip) upon the FT baseline without introducing extra context vectors or adapter modules. Extensive experiments across a wide spectrum of benchmarks demonstrate the superior effectiveness and efficiency of our Skip Tuning over both PT and adapter-based methods. Code: https://github.com/anonymity-007/SkipT.

</details>

---

## 201. MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures

- [ ] MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures | https://cvpr.thecvf.com/virtual/2025/poster/33619

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33619

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The automated analysis of chemical literature holds promise to accelerate discovery in fields such as material science and drug development. In particular, search capabilities for chemical structures and Markush structures (chemical structure templates) within patent documents are valuable, e.g., for prior-art search. Advancements have been made in the automatic extraction of chemical structures from text and images, yet the Markush structures remain largely unexplored due to their complex multi-modal nature. In this work we present MarkushGrapher, a multi-modal approach for recognizing Markush structures in documents. Our method jointly encodes text, image, and layout information through a Vision-Text-Layout encoder and an Optical Chemical Structure Recognition vision encoder. These representations are merged and used to auto-regressively generate a sequential graph representation of the Markush structure along with a table defining its variable groups. To overcome the lack of real-world training data, we propose a synthetic data generation pipeline that produces a wide range of realistic Markush structures. Additionally, we present M2S, the first annotated benchmark of real-world Markush structures, to advance research on this challenging task. Extensive experiments demonstrate that our approach outperforms state-of-the-art chemistry-specific and general-purpose vision-language models in most evaluation settings. Code, models, and datasets will be available upon acceptance.

</details>

---

## 202. Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key

- [ ] Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key | https://cvpr.thecvf.com/virtual/2025/poster/33633

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33633

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucination remains a major challenge for Large Vision-Language Models (LVLMs). Direct Preference Optimization (DPO) has gained increasing attention as a simple solution to hallucination issues. It directly learns from constructed preference pairs that reflect the severity of hallucinations in responses to the same prompt and image. Nonetheless, different data construction methods in existing works bring notable performance variations. We identify a crucial factor here: outcomes are largely contingent on whether the constructed data aligns on-policy w.r.t the initial (reference) policy of DPO. Theoretical analysis suggests that learning from off-policy data is impeded by the presence of KL-divergence between the updated policy and the reference policy. From the perspective of dataset distribution, we systematically summarize the inherent flaws in existing algorithms that employ DPO to address hallucination issues. To alleviate the problems, we propose On-Policy Alignment (OPA)-DPO framework, which uniquely leverages expert feedback to correct hallucinated responses and aligns both the original and expert-revised responses in an on-policy manner. Notably, with only 4.8k data, OPA-DPO achieves an additional reduction in the hallucination rate of  LLaVA-1.5-7B: 13.26\% on the AMBER benchmark and 5.39\% on the Object-Hal benchmark, compared to the previous SOTA algorithm trained with 16k samples.

</details>

---

## 203. Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding

- [ ] Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33629

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33629

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Modern vision-language models (VLMs) develop patch embedding and convolution backbone within vector space, especially Euclidean ones, at the very founding. When expanding VLMs to a galaxy-scale for understanding astronomical phenomena, the integration of spherical space for planetary orbits and hyperbolic spaces for black holes raises two formidable challenges. a) The current pre-training model is confined to Euclidean space rather than a comprehensive geometric embedding. b) The predominant architecture lacks suitable backbones for anisotropic physical geometries. In this paper, we introduced Galaxy-Walker, a geometry-aware VLM, for the universe-level vision understanding tasks. We proposed the geometry prompt that generates geometry tokens by random walks across diverse spaces on a multi-scale physical graph, along with a geometry adapter that compresses and reshapes the space anisotropy in a mixture-of-experts manner. Extensive experiments demonstrate the effectiveness of our approach, with Galaxy-Walker achieving state-of-the-art performance in both galaxy property estimation (R² scores up to 0.91) and morphology classification tasks (up to +0.17 F1 improvement in challenging features), significantly outperforming both domain-specific models and general-purpose VLMs.

</details>

---

## 204. Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention

- [ ] Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention | https://cvpr.thecvf.com/virtual/2025/poster/33641

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33641

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite great success across various multimodal tasks, Large Vision-Language Models (LVLMs) often encounter object hallucinations with generated textual responses being inconsistent with the actual objects in images. We examine different LVLMs and pinpoint that one root cause of object hallucinations lies with deficient attention on discriminative image features. Specifically, LVLMs often predominantly attend to prompt-irrelevant global features instead of prompt-relevant local features, undermining their visual grounding capacity and leading to object hallucinations. We propose Assembly of Global and Local Attention (AGLA), a training-free and plug-and-play approach that mitigates hallucinations by assembling global features for response generation and local features for visual discrimination simultaneously. Specifically, we introduce an image-prompt matching scheme that captures prompt-relevant local features from images, leading to an augmented view of the input image where prompt-relevant content is highlighted while irrelevant distractions are suppressed. Hallucinations can thus be mitigated with a calibrated logit distribution that is from generative global features of the original image and discriminative local features of the augmented image. Extensive experiments show the superiority of AGLA in LVLM hallucination mitigation, demonstrating its wide applicability across both discriminative and generative tasks. Our data and code will be released.

</details>

---

## 205. Adaptive Markup Language Generation for Contextually-Grounded Visual Document Understanding

- [ ] Adaptive Markup Language Generation for Contextually-Grounded Visual Document Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33648

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33648

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Document Understanding has become essential with the increase of text-rich visual content. This field poses significant challenges due to the need for effective integration of visual perception and textual comprehension, particularly across diverse document types with complex layouts. Moreover, existing fine-tuning datasets for this domain often fall short in providing the detailed contextual information for robust understanding, leading to hallucinations and limited comprehension of spatial relationships among visual elements. To address these challenges, we propose an innovative pipeline that utilizes adaptive generation of markup languages, such as Markdown, JSON, HTML, and TiKZ, to build highly structured document representations and deliver contextually-grounded responses. We introduce two fine-grained structured datasets: DocMark-Pile, comprising approximately 3.8M pretraining data pairs for document parsing, and DocMark-Instruct, featuring 624k fine-tuning data annotations for grounded instruction following.Extensive experiments demonstrate that our proposed model significantly outperforms existing state-of-the-art MLLMs across a range of visual document understanding benchmarks, facilitating advanced reasoning and comprehension capabilities in complex visual scenarios.

</details>

---

## 206. Interleaved-Modal Chain-of-Thought

- [ ] Interleaved-Modal Chain-of-Thought | https://cvpr.thecvf.com/virtual/2025/poster/33645

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33645

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chain-of-Thought (CoT) prompting elicits large language models (LLMs) to produce a series of intermediate reasoning steps before arriving at the final answer.However, when transitioning to vision-language models (VLMs), their text-only rationales struggle to express the fine-grained associations with the original image.In this paper, we propose an image-incorporated multimodal Chain-of-Thought, named \textbf{Interleaved-modal Chain-of-Thought (ICoT)}, which generates sequential reasoning steps consisting of paired visual and textual rationales to infer the final answer.Intuitively, the novel ICoT requires VLMs to enable the generation of fine-grained interleaved-modal content, which is hard for current VLMs to fulfill.Considering that the required visual information is usually part of the input image, we propose \textbf{Attention-driven Selection (ADS)} to realize ICoT over existing VLMs.ADS intelligently inserts regions of the input image to generate the interleaved-modal reasoning steps with ignorable additional latency.ADS relies solely on the attention map of VLMs without the need for parameterization, and therefore it is a plug-and-play strategy that can be generalized to a spectrum of VLMs.We apply ADS to realize ICoT on two popular VLMs of different architectures.Extensive evaluations of three benchmarks have shown that ICoT prompting achieves substantial performance (up to 14\%) and interpretability improvements compared to existing multimodal CoT prompting methods.

</details>

---

## 207. MLVU: Benchmarking Multi-task Long Video Understanding

- [ ] MLVU: Benchmarking Multi-task Long Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33659

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33659

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The evaluation of Long Video Understanding (LVU) performance poses an important but challenging research problem. Despite previous efforts, the existing video understanding benchmarks are severely constrained by several issues, especially the insufficient lengths of videos, a lack of diversity in video types and evaluation tasks, and the inappropriateness for evaluating LVU performances. To address the above problems, we propose a new benchmark called MLVU (Multi-task Long Video Understanding Benchmark) for the comprehensive and in-depth evaluation of LVU. MLVU presents the following critical values: 1) The substantial and flexible extension of video lengths, which enables the benchmark to evaluate LVU performance across a wide range of durations. 2) The inclusion of various video genres, e.g., movies, surveillance footage, egocentric videos, cartoons, game videos, etc., which reflects the models' LVU performances in different scenarios. 3) The development of diversified evaluation tasks, which enables a comprehensive examination of MLLMs' key abilities in long-video understanding. The empirical study with 23 latest MLLMs reveals significant room for improvement in today's technique, as all existing methods struggle with most of the evaluation tasks and exhibit severe performance degradation when handling longer videos. Additionally, it suggests that factors such as context length, image-understanding ability, and the choice of LLM backbone can play critical roles in future advancements. We anticipate that MLVU will advance the research of long video understanding by providing a comprehensive and in-depth analysis of MLLMs.

</details>

---

## 208. Towards General Visual-Linguistic Face Forgery Detection

- [ ] Towards General Visual-Linguistic Face Forgery Detection | https://cvpr.thecvf.com/virtual/2025/poster/33669

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33669

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Face manipulation techniques have achieved significant advances, presenting serious challenges to security and social trust. Recent works demonstrate that leveraging multimodal models can enhance the generalization and interpretability of face forgery detection. However, existing annotation approaches, whether through human labeling or direct Multimodal Large Language Model (MLLM) generation, often suffer from hallucination issues, leading to inaccurate text descriptions, especially for high-quality forgeries. To address this, we propose Face Forgery Text Generator (FFTG), a novel annotation pipeline that generates accurate text descriptions by leveraging forgery masks for initial region and type identification, followed by a comprehensive prompting strategy to guide MLLMs in reducing hallucination. We validate our approach through fine-tuning both CLIP with a three-branch training framework combining unimodal and multimodal objectives, and MLLMs with our structured annotations. Experimental results demonstrate that our method not only achieves more accurate annotations with higher region identification accuracy, but also leads to improvements in model performance across various forgery detection benchmarks.

</details>

---

## 209. Cross-modal Information Flow in Multimodal Large Language Models

- [ ] Cross-modal Information Flow in Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33685

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33685

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The recent advancements in auto-regressive multi-modal large language models (MLLMs) have demonstrated promising progress for vision-language tasks. While there exists a variety of studies investigating the processing of linguistic information within large language models, little is currently known about the inner working mechanism of MLLMs and how linguistic and visual information interact within these models. In this study, we aim to fill this gap by examining the information flow between different modalities---language and vision---in MLLMs, focusing on visual question answering.Specifically, given an image-question pair as input, we investigate where in the model and how the visual and linguistic information are combined to generate the final prediction. Conducting experiments with a series of models from the LLaVA series, we find that there are two distinct stages in the process of integration of the two modalities. In the lower layers, the model first transfers the more general visual features of the whole image into the representations of (linguistic) question tokens. In the middle layers, it once again transfers visual information about specific objects relevant to the question to the respective token positions of the question. Finally, in the higher layers, the resulting multimodal representation is propagated to the last position of the input sequence for the final prediction.Overall, our findings provide a new and comprehensive perspective on the spatial and functional aspects of image and language processing in the MLLMs, thereby facilitating future research into multi-modal information localization and editing.

</details>

---

## 210. MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling

- [ ] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling | https://cvpr.thecvf.com/virtual/2025/poster/33687

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33687

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multi-modal large language models have propelled the development of joint probabilistic models capable of both image understanding and generation. However, we have identified that recent methods suffer from loss of image information during understanding task, due to either image discretization or diffusion denoising steps. To address this issue, we propose a novel Multi-Modal Auto-Regressive (MMAR) probabilistic modeling framework. Unlike discretization line of method, MMAR takes in continuous-valued image tokens to avoid information loss in an efficient way. Differing from diffusion-based approaches, we disentangle the diffusion process from auto-regressive backbone model by employing a light-weight diffusion head on top each auto-regressed image patch embedding. In this way, when the model transits from image generation to understanding through text generation, the backbone model's hidden representation of the image is not limited to the last denoising step. To successfully train our method, we also propose a theoretically proven technique that addresses the numerical stability issue and a training strategy that balances the generation and understanding task goals. Extensive evaluations on 18 image understanding benchmarks show that MMAR significantly outperforms most of the existing joint multi-modal models, surpassing the method that employs pre-trained CLIP vision encoder. Meanwhile, MMAR is able to generate high quality images. We also show that our method is scalable with larger data and model size.

</details>

---

## 211. MedUnifier: Unifying Vision-and-Language Pre-training on Medical Data with Vision Generation Task using Discrete Visual Representations

- [ ] MedUnifier: Unifying Vision-and-Language Pre-training on Medical Data with Vision Generation Task using Discrete Visual Representations | https://cvpr.thecvf.com/virtual/2025/poster/33705

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33705

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant progress in Vision-Language Pre-training (VLP), existing VLP approaches predominantly emphasize feature extraction and cross-modal comprehension, with limited attention to generating or transforming visual content. This misalignment constrains the model's ability to synthesize coherent and novel visual representations from textual prompts, thereby reducing the effectiveness of multi-modal learning. In this work, we propose \textbf{MedUnifier}, a unified vision-language pre-training framework tailored for medical data. MedUnifier seamlessly integrates text-grounded image generation capabilities with multi-modal learning strategies, including image-text contrastive alignment, image-text matching and image-grounded text generation. Unlike traditional methods that reply on continuous visual representations, our approach employs visual vector quantization, which not only facilitates a more cohesive learning strategy for cross-modal understanding but also enhances multi-modal generation quality by effectively leveraging discrete representations. Our framework's effectiveness is evidenced by the experiments on established benchmarks, including uni-modal tasks (supervised fine-tuning), cross-modal tasks (image-text retrieval and zero-shot image classification), and multi-modal tasks (medical report generation, image synthesis), where it achieves state-of-the-art performance across various tasks. It also offers a highly adaptable tool designed for a broad spectrum of language and vision tasks in healthcare, marking advancement toward the development of a genuinely generalizable AI model for medical contexts.

</details>

---

## 212. Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception

- [ ] Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception | https://cvpr.thecvf.com/virtual/2025/poster/33712

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33712

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-quality image captions play a crucial role in improving the performance of cross-modal applications such as text-to-image generation, text-to-video generation, and text-image retrieval. To generate long-form, high-quality captions, many recent studies have employed multimodal large language models (MLLMs). However, current MLLMs often produce captions that lack fine-grained details or suffer from hallucinations, a challenge that persists in both open-source and closed-source models. Inspired by Feature-Integration theory, which suggests that attention must focus on specific regions to integrate visual information effectively, we propose a divide-then-aggregate strategy. Our method first divides the image into semantic and spatial patches to extract fine-grained details, enhancing the model's local perception of the image. These local details are then hierarchically aggregated to generate a comprehensive global description. To address hallucinations and inconsistencies in the generated captions, we apply a semantic-level filtering process during hierarchical aggregation. This training-free pipeline can be applied to both open-source models (LLaVA-1.5, LLaVA-1.6, Mini-Gemini) and closed-source models (Claude-3.5-Sonnet, GPT-4o, GLM-4V-Plus). Extensive experiments demonstrate that our method generates more detailed, reliable captions, advancing multimodal description generation without requiring model retraining.

</details>

---

## 213. Visual Lexicon: Rich Image Features in Language Space

- [ ] Visual Lexicon: Rich Image Features in Language Space | https://cvpr.thecvf.com/virtual/2025/poster/33714

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33714

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present Visual Lexicon, an image representation that encodes visual information in the text space while retaining intricate visual details that are often challenging to convey in natural language. Unlike traditional methods that prioritize either high-level semantics (e.g., CLIP) or pixel-level reconstruction (e.g., VAE), ViLex captures both rich semantic content and fine visual details, facilitating high-quality image generation and visual scene understanding. Using a self-supervised learning pipeline, ViLex generates embeddings optimized for reconstructing input images through a frozen text-to-image (T2I) diffusion model, preserving the detailed information necessary for high-fidelity semantic level reconstruction. As visual embeddings in the text space, ViLex embeddings can be used independently as text tokens or combined with natural language tokens for zero-shot multimodal image generation. ViLex is also compatible with downstream vision-language tasks like visual question answering and referring expression segmentation, significantly enhancing performance. Experiments demonstrate that ViLex achieves higher fidelity in image reconstruction compared to text-based embeddings—even with a single token. ViLex also performs various DreamBooth tasks in a zero-shot manner without the need for fine-tuning T2I models, and serves as a powerful vision encoder, consistently enhancing vision-language model performance across 15 benchmarks compared to a strong SigLIP baseline.

</details>

---

## 214. Domain Generalization in CLIP via Learning with Diverse Text Prompts

- [ ] Domain Generalization in CLIP via Learning with Diverse Text Prompts | https://cvpr.thecvf.com/virtual/2025/poster/33717

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33717

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Domain generalization (DG) aims to train a model on source domains that can generalize well to unseen domains. Recent advances in Vision-Language Models (VLMs), such as CLIP, exhibit remarkable generalization capabilities across a wide range of data distributions, benefiting tasks like DG. However, CLIP is pre-trained by aligning images with their descriptions, which inevitably captures domain-specific details. Moreover, adapting CLIP to source domains with limited feature diversity introduces bias. These limitations hinder the model's ability to generalize across domains. In this paper, we propose a new DG approach by learning with diverse text prompts. These text prompts incorporate varied contexts to imitate different domains, enabling DG model to learn domain-invariant features. The text prompts guide DG model learning in three aspects: feature suppression, which uses these prompts to identify domain-sensitive features and suppress them; feature consistency, which ensures the model's features are robust to domain variations imitated by the diverse prompts; and feature diversification, which diversifies features based on the prompts to mitigate bias. Experimental results show that our approach improves domain generalization performance on five datasets on the DomainBed benchmark, achieving state-of-the-art results.

</details>

---

## 215. Online Video Understanding: OVBench and VideoChat-Online

- [ ] Online Video Understanding: OVBench and VideoChat-Online | https://cvpr.thecvf.com/virtual/2025/poster/33731

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33731

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown significant progress in video understanding. However, applying these models to real-world streaming scenarios, such as autonomous driving, augmented reality, and surveillance, presents unique challenges due to real-time dynamics. This paper introduces the Online Spatial-Temporal Video Understanding Benchmark, OVBench, designed specifically to evaluate models’ capacity to interpret spatiotemporal features in streaming contexts. Unlike existing offline benchmarks, OVBench integrates six core task types, each defined across three temporal contexts (past, current, and future) to capture real-time complexity, resulting in a comprehensive set of 15 subtasks based on diverse datasets. To address the unique data constraints and architectural challenges in streaming video understanding, we present our strong baseline model, VideoChat-Online. This model incorporates a hierarchical memory bank architecture that effectively balances spatial and temporal representation, achieving state-of-the-art performance across online and offline scenarios. Our approach surpasses existing state of art offline models Qwen2-VL 7B and online models Flash-VStream, by 4.19\% and 23.7\% on OVBench, respectively. The results demonstrate VideoChat-Online’s efficacy in providing real-time responses while maintaining high accuracy in spatiotemporal comprehension.

</details>

---

## 216. OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation

- [ ] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation | https://cvpr.thecvf.com/virtual/2025/poster/33736

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33736

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have made significant strides in visual understanding and generation tasks. However, generating interleaved image-text content remains a challenge, which requires integrated multimodal understanding and generation abilities. While the progress in unified models offers new solutions, existing benchmarks are insufficient for evaluating these methods due to data size and diversity limitations. To bridge this gap, we introduce OpenING, a comprehensive benchmark comprising 5,400 high-quality human-annotated instances across 56 real-world tasks. OpenING covers diverse daily scenarios such as travel guide, design, and brainstorming, offering a robust platform for challenging interleaved generation methods. In addition, we present IntJudge, a judge model for evaluating open-ended multimodal generation methods. Trained with a novel data pipeline, our IntJudge achieves an agreement rate of 82. 42% with human judgments, outperforming GPT-based evaluators by 11.34%. Extensive experiments on OpenING reveal that current interleaved generation methods still have substantial room for improvement. Key findings on interleaved image-text generation are further presented to guide the development of next-generation models. The benchmark, code and judge models will be released.

</details>

---

## 217. Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model

- [ ] Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model | https://cvpr.thecvf.com/virtual/2025/poster/33735

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33735

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generalized few-shot 3D point cloud segmentation (GFS-PCS) enables model adaptation to new classes with a few support samples while retaining base class segmentation. Existing GFS-PCS approaches focus on enhancing prototypes via interacting with support or query features but remain limited by the sparse knowledge from few-shot samples. Meanwhile, 3D vision-language models (3D VLMs), designed to generalize across open-world novel classes by aligning with language models, contain rich but noisy novel class knowledge. In this work, we introduce a GFS-PCS framework that synergizes dense but noisy pseudo-labels from 3D VLMs with precise yet sparse few-shot samples to maximize the strengths of both, named GFS-VL. Specifically, we present a prototype-guided pseudo-label selection to filter low-quality regions, followed by an adaptive infilling strategy that combines knowledge from pseudo-label contexts and few-shot samples to adaptively label the filtered, unlabeled areas. Additionally, to further utilize few-shot samples, we design a novel-base mix strategy to embed few-shot samples into training scenes, preserving essential context for improved novel class learning. Moreover, recognizing the limited diversity in current GFS-PCS benchmarks, we introduce two challenging benchmarks with diverse novel classes for comprehensive generalization evaluation. Experiments validate the effectiveness of our framework across models and datasets. Our approach and benchmarks provide a solid foundation for advancing GFS-PCS in real-world applications. The code will be released.

</details>

---

## 218. Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation

- [ ] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation | https://cvpr.thecvf.com/virtual/2025/poster/33750

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33750

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In Computational Pathology (CPath), the introduction of Vision-Language Models (VLMs) has opened new avenues for research, focusing primarily on aligning image-text pairs at a single magnification level. However, this approach might not be sufficient for tasks like cancer subtype classification, tissue phenotyping, and survival analysis due to the limited level of detail that a single-resolution image can provide. Addressing this, we propose a novel multi-resolution paradigm leveraging Whole Slide Images (WSIs) to extract histology patches at multiple resolutions and generate corresponding textual descriptions through advanced CPath VLM. This method aims to capture a broader range of information, supported by novel loss functions, enriches feature representation, improves discriminative ability, and enhances generalization across different resolutions. Pre-trained on a comprehensive TCGA dataset with 34 million image-language pairs at various resolutions, our fine-tuned model outperforms State-Of-The-Art (SOTA) counterparts across multiple datasets and tasks, demonstrating its effectiveness in CPath. The code is available on GitHub at xxx.

</details>

---

## 219. On the Zero-shot Adversarial Robustness of Vision-Language Models: A Truly Zero-shot and Training-free Approach

- [ ] On the Zero-shot Adversarial Robustness of Vision-Language Models: A Truly Zero-shot and Training-free Approach | https://cvpr.thecvf.com/virtual/2025/poster/33759

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33759

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained Vision-Language Models (VLMs) like CLIP, have demonstrated strong zero-shot generalization capabilities. Despite their effectiveness on various downstream tasks, they remain vulnerable to adversarial samples. Existing methods fine-tune VLMs to improve their performance via performing adversarial training on a certain dataset. However, this can lead to model overfitting and is not a true zero-shot scenario. In this paper, we propose a truly zero-shot and training-free approach that can significantly improve the VLM's zero-shot adversarial robustness. Specifically, we first discover that simply adding Gaussian noise greatly enhances the VLM's zero-shot performance. Then, we treat the adversarial examples with added Gaussian noise as anchors and strive to find a path in the embedding space that leads from the adversarial examples to the cleaner samples. We improve the VLMs' generalization abilities in a truly zero-shot and training-free manner compared to previous methods. Extensive experiments on 16 datasets demonstrate that our method can achieve state-of-the-art zero-shot robust performance, improving the top-1 robust accuracy by an average of $9.77\%$. The code will be publicly available.

</details>

---

## 220. BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature

- [ ] BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature | https://cvpr.thecvf.com/virtual/2025/poster/33761

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33761

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of vision-language models (VLMs) is driven by large-scale and diverse multi-modal datasets. However, progress toward generalist biomedical VLMs is limited by the lack of annotated, publicly accessible datasets across biology and medicine. Existing efforts are limited to narrow domains, missing the opportunity to leverage the full diversity of biomedical knowledge encoded in scientific literature. To address this gap, we introduce BIOMEDICA: a scalable, open-source framework to extract, annotate, and serialize the entirety of the PubMed Central Open Access subset into an easy-to-use, publicly accessible dataset. Our framework produces a comprehensive archive with over 24 million unique image-text pairs from over 6 million articles. Metadata and expert-guided annotations are additionally provided. We demonstrate the utility and accessibility of our resource by releasing BMCA-LIP, a suite of CLIP-style models continuously pre-trained on BIOMEDICA dataset via streaming (eliminating the need to download 27 TB of data locally). On average, our models achieve state-of-the-art performance across 40 tasks — spanning pathology, radiology, ophthalmology, dermatology, surgery, molecular biology, parasitology, and cell biology — excelling in zero-shot classification with 5.57% average improvement (as high as 26.93% and 17.63% gains in surgery and ophthalmology, respectively) and stronger image-text retrieval while using 10x less compute.

</details>

---

## 221. UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation

- [ ] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation | https://cvpr.thecvf.com/virtual/2025/poster/33765

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33765

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have emerged to tackle the challenges of Visual Question Answering (VQA), sparking a new research focus on conducting objective evaluations of these models. Existing evaluation mechanisms face limitations due to the significant human workload required to design Q\&A pairs for visual images, which inherently restricts the scale and scope of evaluations. Although automated MLLM-as-judge approaches attempt to reduce human workload through mutual model evaluations, they often introduce biases.To address these problems, we propose an unsupervised evaluation method— U nsupervised P eer review M LLM E valuation framework. This framework utilizes only image data, allowing models to automatically generate questions and conduct peer review assessments of answers from other models, effectively alleviating the reliance on human workload.Additionally, we introduce the vision-language scoring system to mitigate the bias issues, which focuses on three aspects: (i) response correctness; (ii) the model capability of visual understanding and reasoning; (iii) relevance of text-image matching.Experimental results demonstrate that UPME achieves a Pearson correlation of 0.944 with human evaluations on the MMstar dataset and 0.814 on the ScienceQA dataset, indicating that our UPME framework closely aligns with human-designed QA benchmarks and inherent human preferences.

</details>

---

## 222. Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection

- [ ] Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection | https://cvpr.thecvf.com/virtual/2025/poster/33773

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33773

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, a task of Single-Domain Generalized Object Detection (Single-DGOD) is proposed,  aiming to generalize a detector to multiple unknown domains never seen before during training. Due to the unavailability of target-domain data, some methods leverage the multimodal capabilities of vision-language models, using textual prompts to estimate cross-domain information, enhancing the model's generalization capability. These methods typically use a single textual prompt, often referred to as the one-step prompt method. However, when dealing with complex styles such as the combination of rain and night, we observe that the performance of the one-step prompt method tends to be relatively weak. The reason may be that many scenes incorporate not just a single style but a combination of multiple styles. The one-step prompt method may not effectively synthesize combined information involving various styles. To address this limitation, we propose a new method, i.e., Style Evolving along Chain-of-Thought, which aims to progressively integrate and expand style information along the chain of thought, enabling the continual evolution of styles. Specifically, by progressively refining style descriptions and guiding the diverse evolution of styles, this approach enables more accurate simulation of various style characteristics and helps the model gradually learn and adapt to subtle differences between styles. Additionally, it exposes the model to a broader range of style features with different data distributions, thereby enhancing its generalization capability in unseen domains. The significant performance gains over five adverse-weather scenarios and the Real to Art benchmark demonstrate the superiorities of our method.

</details>

---

## 223. Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

- [ ] Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity | https://cvpr.thecvf.com/virtual/2025/poster/33777

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33777

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

How can we enable models to comprehend video anomalies occurring over varying temporal scales and contexts?Traditional Video Anomaly Understanding (VAU) methods focus on frame-level anomaly prediction, often missing the interpretability of complex and diverse real-world anomalies. Recent multimodal approaches leverage visual and textual data but lack hierarchical annotations that capture both short-term and long-term anomalies.To address this challenge, we introduce HIVAU-70k, a large-scale benchmark for hierarchical video anomaly understanding across any granularity. We develop a semi-automated annotation engine that efficiently scales high-quality annotations by combining manual video segmentation with recursive free-text annotation using large language models (LLMs). This results in over 70,000 multi-granular annotations organized at clip-level, event-level, and video-level segments.For efficient anomaly detection in long videos, we propose the Anomaly-focused Temporal Sampler (ATS). ATS integrates an anomaly scorer with a density-aware sampler to adaptively select frames based on anomaly scores, ensuring that the multimodal LLM concentrates on anomaly-rich regions, which significantly enhances both efficiency and accuracy.Extensive experiments demonstrate that our hierarchical instruction data markedly improves anomaly comprehension. The integrated ATS and visual-language model outperform traditional methods in processing long videos.Our benchmark and model will be publicly available.

</details>

---

## 224. Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning

- [ ] Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning | https://cvpr.thecvf.com/virtual/2025/poster/33779

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33779

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Few-shot recognition (FSR) aims to train a classification model with only a few labeled examples of each concept concerned by a downstream task, where data annotation cost can be prohibitively high. We develop methods to solve FSR by leveraging a pretrained Vision-Language Model (VLM). We particularly explore retrieval-augmented learning (RAL), which retrieves data from the VLM's pretraining set to learn better models for serving downstream tasks. RAL has been widely studied in zero-shot recognition but remains under-explored in FSR. Although applying RAL to FSR may seem straightforward, we observe interesting and novel challenges and opportunities. First, somewhat surprisingly, finetuning a VLM on a large amount of retrieved data underperforms state-of-the-art zero-shot methods. This is due to the imbalanced distribution of retrieved data and its domain gaps with the few-shot examples in the downstream task. Second, more surprisingly, we find that simply finetuning a VLM solely on few-shot examples significantly outperforms previous FSR methods, and finetuning on the mix of retrieved and few-shot data yields even better results. Third, to mitigate the imbalanced distribution and domain gap issues, we propose Stage-Wise retrieval-Augmented fineTuning (SWAT), which involves end-to-end finetuning on mixed data in the first stage and retraining the classifier on the few-shot data in the second stage. Extensive experiments on nine popular benchmarks demonstrate that SWAT significantly outperforms previous methods by $>$6\% accuracy.

</details>

---

## 225. Context-Aware Multimodal Pretraining

- [ ] Context-Aware Multimodal Pretraining | https://cvpr.thecvf.com/virtual/2025/poster/33797

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33797

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large-scale multimodal representation learning successfully optimizes for zero-shot transfer at test time. Yet the standard pretraining paradigm (contrastive learning on large amounts of image-text data) does not explicitly encourage representations to support few-shot adaptation. In this work, we propose a simple, but carefully designed extension to multimodal pretraining which enables representations to accommodate additional context. Using this objective, we show that vision-language models can be trained to exhibit significantly increased few-shot adaptation: across 21 downstream tasks, we find up to four-fold improvements in test-time sample efficiency, and average few-shot adaptation gains of over 5\%, while retaining zero-shot generalization performance across model scales and training durations. In particular, equipped with simple, training-free, metric-based adaptation mechanisms, our representations surpass significantly more complex optimization-based adaptation schemes.

</details>

---

## 226. Incorporating Dense Knowledge Alignment into Unified Multimodal Representation Models

- [ ] Incorporating Dense Knowledge Alignment into Unified Multimodal Representation Models | https://cvpr.thecvf.com/virtual/2025/poster/33799

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33799

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Leveraging Large Language Models (LLMs) for text representation has achieved significant success, but the exploration of using Multimodal LLMs (MLLMs) for multimodal representation remains limited. Previous MLLM-based representation studies have primarily focused on unifying the embedding space while neglecting the importance of multimodal alignment. As a result, their cross-modal retrieval performance falls markedly behind that of the CLIP series models. To address this, in our work, we 1) construct DeKon5M, a contrastive learning dataset enriched with dense multimodal knowledge, which efficiently enhances multimodal alignment capabilities in representation tasks. 2) design a framework for training unified representation on MLLMs. Building upon this unified representation framework and the dense knowledge dataset DeKon5M, we developed the dense knowledge representation model DeKR on Qwen2VL. Through extensive quantitative and qualitative experiments, our results demonstrate that DeKR not only aligns text, image, video, and text-image combinations within a unified embedding space but also achieves cross-modal retrieval performance comparable to SoTA CLIP series models. This fully validates the effectiveness of our approach and provides new insights for multimodal representation research.

</details>

---

## 227. Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding

- [ ] Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33816

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33816

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The lack of a large-scale 3D-text corpus has led recent works to distill open-vocabulary knowledge from vision-language models (VLMs). However, these methods typically rely on a single VLM to align the feature spaces of 3D models within a common language space, which limits the potential of 3D models to leverage the diverse spatial and semantic capabilities encapsulated in various foundation models. In this paper, we propose Cross-modal and Uncertainty-aware Agglomeration for Open-vocabulary 3D Scene Understanding dubbed CUA-O3D, the first model to integrate multiple foundation models—such as CLIP, Dinov2, and Stable Diffusion—into 3D scene understanding. We further introduce a deterministic uncertainty estimation to adaptively distill and harmonize the heterogeneous 2D feature embeddings from these models.   Our method addresses two key challenges: (1) incorporating semantic priors from VLMs alongside the geometric knowledge of spatially-aware vision foundation models, and (2) using a novel deterministic uncertainty estimation to capture model-specific uncertainties across diverse semantic and geometric sensitivities, helping to reconcile heterogeneous representations during training.    Extensive experiments on ScanNetV2 and Matterport3D demonstrate that our method not only advances open-vocabulary segmentation but also achieves robust cross-domain alignment and competitive spatial perception capabilities.

</details>

---

## 228. Conical Visual Concentration for Efficient Large Vision-Language Models

- [ ] Conical Visual Concentration for Efficient Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33817

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33817

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In large vision-language models (LVLMs), images serve as inputs that carry a wealth of information. As the idiom ''A picture is worth a thousand words" implies, representing a single image in current LVLMs can require hundreds or even thousands of tokens. This results in significant computational costs, which grow quadratically as input image resolution increases, thereby severely impacting the efficiency. Previous approaches have attempted to reduce the number of image tokens either before or within the early layers of LVLMs. However, these strategies inevitably result in the loss of crucial image information. To address this challenge, we conduct an empirical study revealing that all visual tokens are necessary for LVLMs in the shallow layers, and token redundancy progressively increases in the deeper layers.To this end, we propose ViCo, a conical-style visual concentration strategy for LVLMs to boost their efficiency in both training and inference with neglectable performance loss. Specifically, we partition the LVLM into several stages and drop part of the image tokens at the end of each stage with a pre-defined ratio. The dropping is based on a lightweight similarity calculation with a negligible time overhead. Extensive experiments demonstrate that ViCo can achieve over 40\% training time reduction and 55\% inference FLOPs acceleration on leading LVLMs like LLaVA-NeXT, maintaining comparable multi-modal performance. Besides, ViCo can also serve as a plug-and-play strategy to accelerate inference in a free way, with better performance and lower inference cost than counterparts.

</details>

---

## 229. Once-Tuning-Multiple-Variants: Tuning Once and Expanded as Multiple Vision-Language Model Variants

- [ ] Once-Tuning-Multiple-Variants: Tuning Once and Expanded as Multiple Vision-Language Model Variants | https://cvpr.thecvf.com/virtual/2025/poster/33819

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33819

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language model (VLM) is one of the most important models for mono-modal tasks. Real industrial applications often meet the challenge of adapting VLMs to different scenarios, such as varying hardware platforms or performance requirements. Traditional methods involve training or fine-tuning to adapt multiple unique VLMs or using model compression techniques to create multiple compact models. These approaches are complex and resource-intensive. This paper introduces a novel paradigm called Once-Tuning-Multiple-Variants (OTMV). OTMV requires only a single tuning process to inject dynamic weight expansion capacity into the VLM with dynamic expansion capacity. This tuned VLM can then be expanded into multiple variants tailored for different scenarios in inference. The tuning mechanism of OTMV is inspired by the mathematical series expansion theorem, which helps to reduce the parameter size and memory requirements while maintaining accuracy for VLM. Experiment results show that OTMV-tuned models achieve comparable accuracy to baseline VLMs across various visual-language tasks. The experiments also demonstrate the dynamic expansion capability of OTMV-tuned VLMs, outperforming traditional model compression and adaptation techniques in terms of accuracy and efficiency.

</details>

---

## 230. Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs

- [ ] Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs | https://cvpr.thecvf.com/virtual/2025/poster/33822

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33822

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-form video understanding with Large Vision Language Models is challenged by the need to analyze temporally dispersed yet spatially concentrated key moments within limited context windows. In this work, we introduce VideoMindPalace, a new framework inspired by the ``Mind Palace", which organizes critical video moments into a topologically structured semantic graph. VideoMindPalace organizes key information through (i) hand-object tracking and interaction, (ii) clustered activity zones representing specific areas of recurring activities, and (iii) environment layout mapping, allowing natural language parsing by LLMs to provide grounded insights on spatio-temporal and 3D context. In addition, we propose the Video MindPalace  Benchmark (VMB), to assess human-like reasoning, including spatial localization, temporal reasoning, and layout-aware sequential understanding. Evaluated on VMB and established video QA datasets, including EgoSchema, NExT-QA, IntentQA, and the Active Memories Benchmark, VideoMindPalace demonstrates notable gains in spatio-temporal coherence and human-aligned reasoning, advancing long-form video analysis capabilities in VLMs.

</details>

---

## 231. From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons

- [ ] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons | https://cvpr.thecvf.com/virtual/2025/poster/33823

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33823

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We examine the capability of Multimodal Large Language Models (MLLMs) to tackle diverse domains that extend beyond the traditional language and vision tasks these models are typically trained on. Specifically, our focus lies in areas such as Embodied AI, Games, UI Control, and Planning. To this end, we introduce a process of adapting an MLLM to a Generalist Embodied Agent (GEA). GEA is a single unified model capable of grounding itself across these varied domains through a multi-embodiment action tokenizer. GEA is trained with supervised learning on a large dataset of embodied experiences and with online RL in interactive simulators. We explore the data and algorithmic choices necessary to develop such a model. Our findings reveal the importance of training with cross-domain data and online RL for building generalist agents. The final GEA model achieves strong generalization performance to unseen tasks across diverse benchmarks compared to other generalist models and benchmark-specific approaches.

</details>

---

## 232. Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks

- [ ] Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks | https://cvpr.thecvf.com/virtual/2025/poster/33840

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33840

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) can produce unintended and harmful content when exposed to adversarial attacks, particularly because their vision capabilities create new vulnerabilities. Existing defenses, such as input preprocessing, adversarial training, and response evaluation-based methods, are often impractical for real-world deployment due to their high costs. To address this challenge, we propose ASTRA, an efficient and effective defense by adaptively steering models away from adversarial feature directions to resist VLM attacks.Our key procedures involve finding transferable steering vectors representing the direction of harmful response and applying adaptive activation steering to remove these directions at inference time. To create effective steering vectors, we randomly ablate the visual tokens from the adversarial images and identify those most strongly associated with jailbreaks. These tokens are then used to construct steering vectors. During inference, we perform the adaptive steering method that involves the projection between the steering vectors and calibrated activation, resulting in little performance drops on benign inputs while strongly avoiding harmful outputs under adversarial inputs. Extensive experiments across multiple models and baselines demonstrate our state-of-the-art performance and high efficiency in mitigating jailbreak risks. Additionally, ASTRA exhibits good transferability, defending against both unseen attacks at design time (i.e., structured-based attacks) and adversarial images from diverse distributions.

</details>

---

## 233. Words or Vision: Do Vision-Language Models Have Blind Faith in Text?

- [ ] Words or Vision: Do Vision-Language Models Have Blind Faith in Text? | https://cvpr.thecvf.com/virtual/2025/poster/33847

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33847

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) excel in integrating visual and textual information for vision-centric tasks, but their handling of inconsistencies between modalities is underexplored. We investigate VLMs' modality preferences when faced with visual data and varied textual inputs in vision-centered settings.By introducing textual variations to four vision-centric tasks and evaluating ten Vision-Language Models (VLMs), we discover a \emph{``blind faith in text''} phenomenon: VLMs disproportionately trust textual data over visual data when inconsistencies arise, leading to significant performance drops under corrupted text and raising safety concerns.We analyze factors influencing this text bias, including instruction prompts, language model size, text relevance, token order, and the interplay between visual and textual certainty. While certain factors, such as scaling up the language model size, slightly mitigate text bias, others like token order can exacerbate it due to positional biases inherited from language models. To address this issue, we explore supervised fine-tuning with text augmentation and demonstrate its effectiveness in reducing text bias. Additionally, we provide a theoretical analysis suggesting that the blind faith in text phenomenon may stem from an imbalance of pure text and multi-modal data during training.Our findings highlight the need for balanced training and careful consideration of modality interactions in VLMs to enhance their robustness and reliability in handling multi-modal data inconsistencies.

</details>

---

## 234. Embodied Scene Understanding for Vision Language Models via MetaVQA

- [ ] Embodied Scene Understanding for Vision Language Models via MetaVQA | https://cvpr.thecvf.com/virtual/2025/poster/33852

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33852

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) show promise as embodied agents in many mobility applications, yet there is a lack of a generalizable platform for evaluating their spatial reasoning and embodied scene understanding. We introduce MetaVQA, a comprehensive benchmark that assesses and enhances VLMs’ understanding of spatial relationships and embodied dynamics in driving scenes through Visual-Question-Answering (VQA) and closed-loop simulation. MetaVQA collects various question-answer pairs from diverse real-world traffic scenarios through Set-of-Mark prompting and top-down view ground-truth annotations of nuScenes and Waymo datasets to ensure real-world and object-centric instructions. We demonstrate that fine-tuning VLMs on the MetaVQA dataset improves their spatial reasoning and embodied scene understanding in safety-critical simulations. Code and data will be made available.

</details>

---

## 235. Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training

- [ ] Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training | https://cvpr.thecvf.com/virtual/2025/poster/33855

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33855

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we focus on  monolithic Multimodal Large Language Models (MLLMs) that integrate visual encoding and language decoding into a single LLM.  In particular, we identify that existing pre-training strategies for monolithic MLLMs often suffer from unstable optimization or catastrophic forgetting.  To address this issue, our core idea is to embed a new visual parameter space into a pre-trained LLM, thereby stably learning visual knowledge from noisy data while freezing the LLM.  Based on this principle, we   present Mono-InternVL,  a novel monolithic MLLM  that  seamlessly integrates a set of  visual experts via  a multimodal mixture-of-experts structure.  Moreover, we propose an innovative pre-training strategy to maximize the visual capability of Mono-InternVL, namely Endogenous Visual Pre-training (EViP).  In particular, EViP is designed as a progressive learning process for  visual experts,   which aims to  fully exploit the visual knowledge  from noisy data to high-quality data.   To validate our approach, we conduct extensive experiments on 16 benchmarks.  Experimental results confirm the superior performance of Mono-InternVL than existing monolithic MLLMs on 13 of 16 multimodal benchmarks, e.g., +80 points over Emu3 on OCRBench.  Compared to the modular baseline, i.e., InternVL-1.5, Mono-InternVL  still retains comparable multimodal performance while reducing up to   67% first token latency.  Code and model will be released.

</details>

---

## 236. Parameter-efficient Fine-tuning in Hyperspherical Space for Open-vocabulary Semantic Segmentation

- [ ] Parameter-efficient Fine-tuning in Hyperspherical Space for Open-vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/33863

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33863

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation seeks to label each pixel in an image with arbitrary text descriptions. Vision-language foundation models, especially CLIP, have recently emerged as powerful tools for acquiring open-vocabulary capabilities. However, fine-tuning CLIP to equip it with pixel-level prediction ability often suffers three issues: 1) high computational cost, 2) misalignment between the two inherent modalities of CLIP, and 3) degraded generalization ability on unseen categories. To address these issues, we propose \alg, a symmetrical parameter-efficient fine-tuning (PEFT) strategy conducted in hyperspherical space for both of the two CLIP modalities. Specifically, the PEFT strategy is achieved by a series of efficient block-diagonal learnable transformation matrices and a dual cross-relation communication module among all learnable matrices. Since the PEFT strategy is conducted symmetrically to the two CLIP modalities, the misalignment between them is mitigated. Furthermore, we apply an additional constraint to PEFT on the CLIP text encoder according to the hyperspherical energy principle, i.e., minimizing hyperspherical energy during fine-tuning preserves the intrinsic structure of the original parameter space, to prevent the destruction of the generalization ability offered by the CLIP text encoder. Extensive evaluations across various benchmarks show that H-CLIP achieves new SOTA open-vocabulary semantic segmentation results while only requiring updating approximately 4% of the total parameters of CLIP.

</details>

---

## 237. Empowering Large Language Models with 3D Situation Awareness

- [ ] Empowering Large Language Models with 3D Situation Awareness | https://cvpr.thecvf.com/virtual/2025/poster/33870

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33870

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Driven by the great success of Large Language Models (LLMs) in the 2D image domain, their applications in 3D scene understanding has emerged as a new trend. A key difference between 3D and 2D is that the situation of an egocentric observer in 3D scenes can change, resulting in different descriptions (e.g., ''left" or ''right"). However, current LLM-based methods overlook the egocentric perspective and simply use datasets from a global viewpoint. To address this issue, we propose a novel approach to automatically generate a situation-aware dataset by leveraging the scanning trajectory during data collection and utilizing Vision-Language Models (VLMs) to produce high-quality captions and question-answer pairs. Furthermore, we introduce a situation grounding module to explicitly predict the position and orientation of observer's viewpoint, thereby enabling LLMs to ground situation description in 3D scenes. We evaluate our approach on several benchmarks, demonstrating that our method effectively enhances the 3D situational awareness of LLMs while significantly expanding existing datasets and reducing manual effort.

</details>

---

## 238. SLADE: Shielding against Dual Exploits in Large Vision-Language Models

- [ ] SLADE: Shielding against Dual Exploits in Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33877

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33877

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have emerged as transformative tools in multimodal tasks, seamlessly integrating pretrained vision encoders to align visual and textual modalities. Prior works have highlighted the susceptibility of LVLMs to dual exploits (gradient-based and optimization-based jailbreak attacks), which leverage the expanded attack surface introduced by the image modality. Despite advancements in enhancing robustness, existing methods fall short in their ability to defend against dual exploits while preserving fine-grained semantic details and overall semantic coherence under intense adversarial perturbations. To bridge this gap, we introduce SLADE, a novel unsupervised adversarial fine-tuning scheme that enhances the resilience of CLIP-based vision encoders. SLADE’s dual-level contrastive learning approach balances the granular and the holistic, capturing fine-grained image details without losing sight of high-level semantic coherence. Extensive experiments demonstrate that SLADE-equipped LVLMs set a new benchmark for robustness against dual exploits while preserving fine-grained semantic details of perturbed images. Notably, SLADE achieves these results without compromising the core functionalities of LVLMs, such as instruction following, or requiring the computational overhead (e.g., large batch sizes, momentum encoders) commonly associated with traditional contrastive learning methods. The code is provided in the supplementary material with this submission.

</details>

---

## 239. ChatGarment: Garment Estimation, Generation and Editing via Large Language Models

- [ ] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33886

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33886

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce ChatGarment, a novel approach that leverages large vision-language models (VLMs) to automate the estimation, generation, and editing of 3D garment sewing patterns from images or text descriptions. Unlike previous methods that often lack robustness and interactive editing capabilities, ChatGarment finetunes a VLM to produce GarmentCode, a JSON-based, language-friendly format for 2D sewing patterns, enabling both estimating and editing from images and text instructions. To optimize performance, we refine GarmentCode by expanding its support for more diverse garment types and simplifying its structure, making it more efficient for VLM finetuning. Additionally, we develop an automated data construction pipeline to generate a large-scale dataset of image-to-sewing-pattern and text-to-sewing-pattern pairs, empowering ChatGarment with strong generalization across various garment types.  Extensive evaluations demonstrate ChatGarment’s ability to accurately reconstruct, generate, and edit garments from multimodal inputs, highlighting its potential to revolutionize workflows in fashion and gaming applications.

</details>

---

## 240. EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions

- [ ] EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions | https://cvpr.thecvf.com/virtual/2025/poster/33880

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33880

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

GPT-4o, an omni-modal model that enables vocal conversations with diverse emotions and tones, marks a milestone for omni-modal foundation models. However, empowering Large Language Models to perceive and generate images, texts, and speeches end-to-end with publicly available data remains challenging for the open-source community. Existing vision-language models rely on external tools for speech processing, while speech-language models still suffer from limited or totally without vision-understanding capabilities. To address this gap, we propose the EMOVA (EMotionally Omni-present Voice Assistant), to enable Large Language Models with end-to-end speech abilities while maintaining the leading vision-language performance. With a semantic-acoustic disentangled speech tokenizer, we surprisingly notice that omni-modal alignment can further enhance vision-language and speech abilities compared with the bi-modal aligned counterparts. Moreover, a lightweight style module is introduced for the flexible speech style controls including emotions and pitches.For the first time, EMOVA achieves state-of-the-art performance on both the vision-language and speech benchmarks, and meanwhile, supporting omni-modal spoken dialogue with vivid emotions.

</details>

---

## 241. OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints

- [ ] OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints | https://cvpr.thecvf.com/virtual/2025/poster/33901

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33901

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of general robotic systems capable of manipulating in unstructured environments is a significant challenge. While Vision-Language Models(VLM)  excel in high-level commonsense reasoning, they lack the fine-grained 3D spatial understanding required for precise manipulation tasks. Fine-tuning VLM on robotic datasets to create Vision-Language-Action Models(VLA) is a potential solution, but it is hindered by high data collection costs and generalization issues. To address these challenges, we propose a novel object-centric representation that bridges the gap between VLM's high-level reasoning and the low-level precision required for manipulation. Our key insight is that an object's canonical space, defined by its functional affordances, provides a structured and semantically meaningful way to describe interaction primitives, such as points and directions. These primitives act as a bridge, translating VLM's commonsense reasoning into actionable 3D spatial constraints. In this context, we introduce a dual closed-loop, open-vocabulary robotic manipulation system: one loop for high-level planning through primitive resampling, interaction rendering and VLM checking, and another for low-level execution via 6D pose tracking. This design ensures robust, real-time control without requiring VLM fine-tuning. Extensive experiments demonstrate strong zero-shot generalization across diverse robotic manipulation tasks, highlighting the potential of this approach for automating large-scale simulation data generation.

</details>

---

## 242. BlenderGym: Benchmarking Foundational Model Systems for Graphics Editing

- [ ] BlenderGym: Benchmarking Foundational Model Systems for Graphics Editing | https://cvpr.thecvf.com/virtual/2025/poster/33899

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33899

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D graphics editing is a crucial component in applications like movie production and game design, yet it remains a time-consuming process that demands highly specialized domain expertise. Automating the process is challenging because graphical editing requires performing different tasks, each requiring distinct skill sets. Recently, multi-modal foundation models have emerged as a powerful framework for automating the editing process, but their development and evaluation are bottlenecked by the lack of a comprehensive benchmark that requires human-level perception and real-world editing complexity. In this work, we present BlenderGym, a benchmark designed to systematically evaluate foundational model systems for 3D graphics editing with tasks capturing the various aspects of 3D editing and fixed ground-truth for evaluation. We evaluate closed- and open-source VLMs with BlenderGym and observe that even the state-of-the-art VLMs struggle with tasks relatively easily for a novice Blender user. Enabled by BlenderGym, we study how inference scaling techniques impact graphics editing tasks. Notably, our findings reveal that the verifier used to guide the scaling of generation can itself be improved through scaling, complementing recent insights on scaling of LLM generation in coding and math tasks. We further show that inference compute is not uniformly effective and can be optimized by strategically distributing it between generation and verification.

</details>

---

## 243. InteractVLM: 3D Interaction Reasoning from 2D Foundational Models

- [ ] InteractVLM: 3D Interaction Reasoning from 2D Foundational Models | https://cvpr.thecvf.com/virtual/2025/poster/33902

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33902

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Estimating the 3D pose and shape of interacting humans and objects from single in-the-wild images is important for mixed reality and robotics. This is challenging due to occlusions, depth ambiguities, and widely varying object shapes. Existing work tackles these challenges by exploiting surface contact points on the body and object and using these to guide 3D reconstruction. Unfortunately, obtaining 3D contact annotations requires either expensive 3D ground truth or time-consuming manual labeling. Consequently, obtaining training data at scale is a challenge. We tackle this by developing a novel model called InteractVLM that harnesses the broad visual knowledge of large Visual-Language Models (VLMs). The problem is, however, that these large models do not directly “understand” 3D human-object contact. To address this, we exploit existing small datasets of 3D human-object interaction to fine-tune large models to understand contact. However, this is non-trivial, as such models reason “only” in 2D, while contact is inherently 3D. Thus, we introduce a novel “Render-Localize-Lift” module that: (1) embeds 3D body and object surfaces in 2D space via multi-view rendering, (2) trains a novel multi-view localization model (MV-Loc) to infer contacts in 2D, and (3) lifts these to 3D. This lets InteractVLM infer 3D contacts for both bodies and objects from a single in-the-wild image. InteractVLM outperforms existing work on contact estimation and also facilitates 3D reconstruction from an in-the-wild image. To estimate 3D human and object pose, we infer initial body and object meshes, then infer contacts on both of these via InteractVLM, and lastly exploit these in fitting human and object meshes to image evidence. Results show that our approach performs promisingly in the wild. Our code and models will be released.

</details>

---

## 244. Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models

- [ ] Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33903

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33903

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent Multi-modal Large Language Models (MLLMs) have been challenged by the computational overhead resulting from massive video frames, often alleviated through compression strategies. However, the visual content is not equally contributed to user instructions, existing strategies (\eg, average pool) inevitably lead to the loss of potentially useful information. To tackle this, we propose the \textbf{H}ybrid-level Instruction \textbf{I}njection Strategy for \textbf{C}onditional Token C\textbf{om}pression in MLLMs (HICom), utilizing the instruction as a condition to guide the compression from both local and global levels. This encourages the compression to retain the maximum amount of user-focused information while reducing visual tokens to minimize computational burden. Specifically, the instruction condition is injected into the grouped visual tokens at the local level and the learnable tokens at the global level, and we conduct the attention mechanism to complete the conditional compression. From the hybrid-level compression, the instruction-relevant visual parts are highlighted while the temporal-spatial structure is also preserved for easier understanding of LLMs. To further unleash the potential of HICom, we introduce a new conditional pre-training stage with our proposed dataset HICom-248K. Experiments show that our HICom can obtain distinguished video understanding ability with fewer tokens, increasing the performance by 3.88\% average on three multiple-choice QA benchmarks and saving 78.8\% tokens compared with the SOTA method. The code, dataset, and model will be released.

</details>

---

## 245. ECBench: Can Multi-modal Foundation Models Understand the Egocentric World? A Holistic Embodied Cognition Benchmark

- [ ] ECBench: Can Multi-modal Foundation Models Understand the Egocentric World? A Holistic Embodied Cognition Benchmark | https://cvpr.thecvf.com/virtual/2025/poster/33904

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33904

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The enhancement of generalization in robots by large vision-language models (LVLMs) is increasingly evident. Therefore, the embodied cognitive abilities of LVLMs based on egocentric videos are of great interest. However, current datasets for embodied video question answering lack comprehensive and systematic evaluation frameworks. Critical embodied cognitive issues, such as robotic self-cognition, dynamic scene perception, and hallucination, are rarely addressed.To tackle these challenges, we propose ECBench, a high-quality benchmark designed to systematically evaluate the embodied cognitive abilities of LVLMs. ECBench features a diverse range of scene video sources, open and varied question formats, and 30 dimensions of embodied cognition. To ensure quality, balance, and high visual dependence, ECBench uses class-independent meticulous human annotation and multi-round question screening strategies.Additionally, we introduce ECEval, a comprehensive evaluation system that ensures the fairness and rationality of the indicators. Utilizing ECBench, we conduct extensive evaluations of proprietary, open-source, and task-specific LVLMs. ECBench is pivotal in advancing the embodied cognitive capabilities of LVLMs, laying a solid foundation for developing reliable core models for embodied agents. All data and code will be open-sourced.

</details>

---

## 246. Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning

- [ ] Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning | https://cvpr.thecvf.com/virtual/2025/poster/33906

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33906

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-task prompt learning has emerged as a promising technique for fine-tuning pre-trained Vision-Language Models (VLMs) to various downstream tasks. However, existing methods ignore challenges caused by spurious correlations and dynamic task relationships, which may reduce the model performance. To tackle these challenges, we propose JSCPT, a novel approach for \textit{Joint Scheduling of Causal Prompts and Tasks} to enhance multi-task prompt learning. Specifically, we first design a \textit{Multi-Task Vison-Language Prompt} (MTVLP) model, which learns task-shared and task-specific vison-language prompts and selects useful prompt features via causal intervention, alleviating spurious correlations. Then, we propose the task-prompt scheduler that models inter-task affinities and assesses the causal effect of prompt features to optimize the multi-task prompt learning process. Finally, we formulate the scheduler and the multi-task prompt learning process as a bi-level optimization problem to optimize prompts and tasks adaptively. In the lower optimization, MTVLP is updated with the scheduled gradient, while in the upper optimization, the scheduler is updated with the implicit gradient. Extensive experiments show the superiority of our proposed JSCPT approach over several baselines in terms of multi-task prompt learning for pre-trained VLMs.

</details>

---

## 247. Notes-guided MLLM Reasoning: Enhancing MLLM with Knowledge and Visual Notes for Visual Question Answering

- [ ] Notes-guided MLLM Reasoning: Enhancing MLLM with Knowledge and Visual Notes for Visual Question Answering | https://cvpr.thecvf.com/virtual/2025/poster/33913

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33913

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The knowledge-based visual question answering (KB-VQA) task involves using external knowledge about the image to assist reasoning. Building on the impressive performance of multimodal large language model (MLLM), recent methods have commenced leveraging MLLM as an implicit knowledge base for reasoning. However, the direct employment of MLLM with raw external knowledge might result in reasoning errors due to misdirected knowledge information.  Additionally, MLLM may lack fine-grained perception of visual features, which can result in hallucinations during reasoning. To address these challenges, we propose Notes-guided MLLM Reasoning (NoteMR), a novel framework that guides MLLM in better reasoning by utilizing knowledge notes and visual notes. Specifically, we initially obtain explicit knowledge from an external knowledge base. Then, this explicit knowledge, combined with images, is used to assist the MLLM in generating knowledge notes. These notes are designed to filter explicit knowledge and identify relevant internal implicit knowledge within the MLLM. We then identify highly correlated regions between the images and knowledge notes, retaining them as image notes to enhance the model's fine-grained perception, thereby mitigating MLLM induced hallucinations. Finally, both notes are fed into the MLLM, enabling a more comprehensive understanding of the image-question pair and enhancing the model's reasoning capabilities. Our method achieves state-of-the-art performance on the OK-VQA and A-OKVQA datasets, demonstrating its robustness and effectiveness across diverse VQA scenarios.

</details>

---

## 248. Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Model

- [ ] Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Model | https://cvpr.thecvf.com/virtual/2025/poster/33917

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33917

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal language models (MLLMs) are increasingly being applied in real-world environments, necessitating their ability to interpret 3D spaces and comprehend temporal dynamics. Current methods often rely on specialized architectural designs or task-specific fine-tuning to achieve this. We introduce Coarse Correspondences, a simple lightweight method that enhances MLLMs’ spatial-temporal reasoning with 2D images as input, without modifying the architecture or requiring task-specific fine-tuning. Our method uses a lightweight tracking model to identify primary object correspondences between frames in a video or across different image viewpoints, and then conveys this information to MLLMs through visual prompting. We demonstrate that this simple training-free approach brings substantial gains to GPT4-V/O consistently on four benchmarks that require spatial-temporal reasoning, including +20.5% improvement on ScanQA, +9.7% on OpenEQA’s episodic memory subset, +6.0% on the long-form video benchmark EgoSchema, and +11% on the R2R navigation benchmark. Additionally, we show that Coarse Correspondences  can also enhance open-source MLLMs’ spatial reasoning (by +6.9% on ScanQA) when applied in both training and inference and that the improvement can generalize to unseen datasets such as SQA3D (+3.1%). Taken together, we show that Coarse Correspondences  effectively and efficiently boosts models’ performance on downstream tasks requiring spatial-temporal reasoning.

</details>

---

## 249. Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift

- [ ] Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift | https://cvpr.thecvf.com/virtual/2025/poster/33919

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33919

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuning enhances large vision-language models (LVLMs) but increases their vulnerability to backdoor attacks due to their open design. Unlike prior studies in static settings, this paper explores backdoor attacks in LVLM instruction tuning across mismatched training and testing domains. We introduce a new evaluation dimension, backdoor domain generalization, to assess attack robustness under visual and text domain shifts. Our findings reveal two insights: (1) backdoor generalizability improves when distinctive trigger patterns are independent of specific data domains or model architectures, and (2) triggers placed in preference over clean semantic regions significantly enhance attack generalization. Based on these insights, we propose a multimodal attribution backdoor attack (MABA) that injects domain-agnostic triggers into critical areas using attributional interpretation. Experiments with OpenFlamingo, Blip-2, and Otter show that MABA significantly boosts the attack success rate of generalization by 36.4\%, achieving a 97\% success rate at a 0.2\% poisoning rate. This study reveals limitations in current evaluations and highlights how enhanced backdoor generalizability poses a security threat to LVLMs, even without test data access.

</details>

---

## 250. Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment

- [ ] Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment | https://cvpr.thecvf.com/virtual/2025/poster/33930

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33930

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent contrastive multimodal vision-language models like CLIP have demonstrated robust open-world semantic understanding, becoming the standard image backbones for vision-language applications. However, recent findings suggest high semantic similarity between well-trained unimodal encoders, which raises a key question: Are semantically similar embedding spaces separated only by simple projection transformations? To validate this, we propose a novel framework that aligns vision and language using frozen unimodal encoders. It involves selecting semantically similar encoders in the latent space, curating a concept-rich dataset of image-caption pairs, and training simple MLP projectors. We evaluated our approach on various tasks involving both strong unimodal vision (0-shot localization) and language encoders (multi-lingual, long context) and show that simple Projectors retain unimodal capabilities in joint embedding space. Furthermore, our best model, utilizing DINOv2 and All-Roberta-Large text encoder, achieves 76(\%) accuracy on ImageNet with a 20-fold reduction in data and 65-fold reduction in compute requirements compared to multimodal alignment where models are trained from scratch. The proposed framework enhances the accessibility of multimodal model development while enabling flexible adaptation across diverse scenarios. Code and curated datasets will be released soon

</details>

---

## 251. DIV-FF: Dynamic Image-Video Feature Fields For Environment Understanding in Egocentric Videos

- [ ] DIV-FF: Dynamic Image-Video Feature Fields For Environment Understanding in Egocentric Videos | https://cvpr.thecvf.com/virtual/2025/poster/33936

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33936

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Environment understanding in egocentric videos is an important step for applications like robotics, augmented reality and assistive technologies. These videos are characterized by dynamic interactions and a strong dependence on the wearer’s engagement with the environment. Traditional approaches often focus on isolated clips or fail to integrate rich semantic and geometric information, limiting scene comprehension. We introduce Dynamic Image-Video Feature Fields (DIV-FF), a framework that decomposes the egocentric scene into persistent, dynamic, and actor-based components while integrating both image and video-language features. Our model enables detailed segmentation, captures affordances, understands the surroundings and maintains consistent understanding over time. DIV-FF outperforms state-of-the-art methods, particularly in dynamically evolving scenarios, demonstrating its potential to advance long-term, spatio-temporal scene understanding.

</details>

---

## 252. Is Your World Simulator a Good Story Presenter? A Consecutive Events-Based Benchmark for Future Long Video Generation

- [ ] Is Your World Simulator a Good Story Presenter? A Consecutive Events-Based Benchmark for Future Long Video Generation | https://cvpr.thecvf.com/virtual/2025/poster/33938

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33938

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The current state-of-the-art video generative models can produce commercial-grade videos with highly realistic details. However, they still struggle to coherently present multiple sequential events in specific short stories, which is foreseeable an essential capability for future long video generation scenarios. While existing detail-oriented benchmarks primarily focus on fine-grained metrics like aesthetic quality and spatial-temporal consistency, they fall short of evaluating models' abilities to handle event-level story presentation.To address this gap, we introduce StoryEval, a story-oriented benchmark specifically designed to assess text-to-video (T2V) models' story-completion capabilities. StoryEval features 423 prompts spanning 7 classes, each representing short stories composed of 2–4 consecutive events. We employ Vision-Language Models, such as GPT-4V and LLaVA-OV-Chat-72B, to verify the completion of each event in the generated videos, applying a unanimous voting method to enhance reliability. Our methods ensure high alignment with human evaluations, and the evaluation of 11 models reveals its challenge, with none exceeding an average story-completion rate of 50\%. StoryEval provides a new benchmark for advancing T2V models and highlights the challenges and opportunities in developing next-generation solutions for coherent story-driven video generation.

</details>

---

## 253. From Head to Tail: Towards Balanced Representation in Large Vision-Language Models through Adaptive Data Calibration

- [ ] From Head to Tail: Towards Balanced Representation in Large Vision-Language Models through Adaptive Data Calibration | https://cvpr.thecvf.com/virtual/2025/poster/33942

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33942

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have achieved significant progress in combining visual comprehension with language generation.Despite this success, the training data of LVLMs still suffers from $\textit{Long-Tail (LT)}$ problems, where the data distribution is highly imbalanced.Previous works have mainly focused on traditional VLM architectures, i.e., CLIP or ViT, and specific tasks such as recognition and classification. Nevertheless, the exploration of LVLM (e.g. LLaVA) and more general tasks (e.g. Visual Question Answering and Visual Reasoning) remains under-explored.In this paper, we first conduct an in-depth analysis of the LT issues in LVLMs and identify two core causes: the overrepresentation of head concepts and the underrepresentation of tail concepts.Based on the above observation, we propose an $\textbf{A}$daptive $\textbf{D}$ata $\textbf{R}$efinement Framework ($\textbf{ADR}$), which consists of two stages: $\textbf{D}$ata $\textbf{R}$ebalancing ($\textbf{DR}$) and $\textbf{D}$ata $\textbf{S}$ynthesis ($\textbf{DS}$).In the DR stage, we adaptively rebalance the redundant data based on entity distributions, while in the DS stage, we leverage Denoising Diffusion Probabilistic Models (DDPMs) and scarce images to supplement underrepresented portions.Through comprehensive evaluations across eleven benchmarks, our proposed ADR effectively mitigates the long-tail problem in the training data, improving the average performance of LLaVA 1.5 relatively by $4.36\%$, without increasing the training data volume. Our code and data will be publicly released.

</details>

---

## 254. LLaVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding

- [ ] LLaVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding | https://cvpr.thecvf.com/virtual/2025/poster/33958

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33958

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have shown promising results, yet existing approaches struggle to effectively handle both temporal and spatial localization simultaneously. This challenge stems from two key issues: first, incorporating spatial-temporal localization introduces a vast number of coordinate combinations, complicating the alignment of linguistic and visual coordinate representations; second, encoding fine-grained temporal and spatial information during video feature compression is inherently difficult. To address these issues, we propose LLaVA-ST, a MLLM for fine-grained spatial-temporal multimodal understanding. In LLaVA-ST, we propose Language-Aligned Positional Embedding, which embeds the textual coordinate special token into the visual space, simplifying the alignment of fine-grained spatial-temporal correspondences. Additionally, we design the Spatial-Temporal Packer, which decouples the feature compression of temporal and spatial resolutions into two distinct point-to-region attention processing streams. Furthermore, we propose ST-Align dataset with 4.3M training samples for fine-grained spatial-temporal multimodal understanding. With ST-align, we present a progressive training pipeline that aligns the visual and textual feature through sequential coarse-to-fine stages. LLaVA-ST achieves outstanding performance on 12 benchmarks requiring fine-grained temporal, spatial, or spatial-temporal interleaving multimodal understanding. Our code, data and benchmark will be released.

</details>

---

## 255. Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation

- [ ] Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/33959

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33959

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-Vocabulary Semantic Segmentation (OVSS) has advanced with recent vision-language models (VLMs), enabling segmentation beyond predefined categories through various learning schemes. Notably, training-free methods offer scalable, easily deployable solutions for handling unseen data, a key goal of OVSS. Yet, a critical issue persists: lack of object-level context consideration when segmenting complex objects in the challenging environment of OVSS based on arbitrary query prompts. This oversight limits models' ability to group semantically consistent elements within object and map them precisely to user-defined arbitrary classes. In this work, we introduce a novel approach that overcomes this limitation by incorporating object-level contextual knowledge within images. Specifically, our model enhances intra-object consistency by distilling spectral-driven features from vision foundation models into the attention mechanism of the visual encoder, enabling semantically coherent components to form a single object mask. Additionally, we refine the text embeddings with zero-shot object presence likelihood to ensure accurate alignment with the specific objects represented in the images. By leveraging object-level contextual knowledge, our proposed approach achieves state-of-the-art performance with strong generalizability across diverse datasets. All the attached source code will be made available to the public.

</details>

---

## 256. LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models

- [ ] LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33962

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33962

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-universe 3D layout generation arranges unlabeled 3D assets conditioned on language instruction. Large language models (LLMs) struggle with generating physically plausible 3D scenes and adherence to input instructions, particularly in dense scenes. We introduce LayoutVLM, a framework and scene layout representation that exploits the semantic knowledge of Vision-Language Models (VLMs) and supports differentiable optimization to ensure physical plausibility. LayoutVLM employs VLMs to generate two mutually reinforcing representations from visually marked images, and a self-consistent decoding process to improve VLMs spatial planning. Our experiments show that LayoutVLM addresses the limitations of existing LLM and constraint-based approaches, producing physically plausible 3D layouts better aligned with the semantic intent of input language instructions. We also demonstrate that fine-tuning VLMs with the proposed scene layout representation extracted from existing scene datasets can improve performance.

</details>

---

## 257. ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large Language Models

- [ ] ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33965

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33965

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive decoding strategies are widely used to mitigate object hallucinations in multimodal large language models (MLLMs). By reducing over-reliance on language priors, these strategies ensure that generated content remains closely grounded in visual inputs, producing contextually accurate outputs. Since contrastive decoding requires no additional training or external tools, it offers both computational efficiency and versatility, making it highly attractive. However, these methods present two main limitations: (1) bluntly suppressing language priors can compromise coherence and accuracy of generated content, and (2) processing contrastive inputs adds computational load, significantly slowing inference speed. To address these challenges, we propose Visual Amplification Fusion (VAF), a plug-and-play technique that enhances attention to visual signals within the model’s middle layers, where modality fusion predominantly occurs. This approach enables more effective capture of visual features, reducing the model’s bias toward language modality. Experimental results demonstrate that VAF significantly reduces hallucinations across various MLLMs without affecting inference speed, while maintaining coherence and accuracy in generated outputs.

</details>

---

## 258. Distraction is All You Need for Multimodal Large Language Model Jailbreaking

- [ ] Distraction is All You Need for Multimodal Large Language Model Jailbreaking | https://cvpr.thecvf.com/virtual/2025/poster/33971

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33971

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) bridge the gap between visual and textual data, enabling a range of advanced applications. However, complex internal interactions among visual elements and their alignment with text can introduce vulnerabilities, which may be exploited to bypass safety mechanisms. To address this, we analyze the relationship between image content and task and find that the complexity of subimages, rather than their content, is key. Building on this insight, we propose the $\textbf{Distraction Hypothesis}$, followed by a novel framework called Contrasting Subimage Distraction Jailbreaking ($\textbf{CS-DJ}$), to achieve jailbreaking by disrupting MLLMs alignment through multi-level distraction strategies. CS-DJ consists of two components: structured distraction, achieved through query decomposition that induces a distributional shift by fragmenting harmful prompts into sub-queries, and visual-enhanced distraction, realized by constructing contrasting subimages to disrupt the interactions among visual elements within the model. This dual strategy disperses the model’s attention, reducing its ability to detect and mitigate harmful content. Extensive experiments across five representative scenarios and four popular closed-source MLLMs, including $\texttt{GPT-4o-mini}$, $\texttt{GPT-4o}$, $\texttt{GPT-4V}$, and $\texttt{Gemini-1.5-Flash}$, demonstrate that CS-DJ achieves average success rates of $\textbf{52.40\%}$ for the attack success rate and $\textbf{74.10\%}$ for the ensemble attack success rate. These results reveal the potential of distraction-based approaches to exploit and bypass MLLMs' defenses, offering new insights for attack strategies.

</details>

---

## 259. STPro: Spatial and Temporal Progressive Learning for Weakly Supervised Spatio-Temporal Grounding

- [ ] STPro: Spatial and Temporal Progressive Learning for Weakly Supervised Spatio-Temporal Grounding | https://cvpr.thecvf.com/virtual/2025/poster/33976

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33976

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work we study Weakly Supervised Spatio-Temporal Video Grounding (WSTVG), a challenging task of localizing subjects spatio-temporally in videos using only textual queries and no bounding box supervision. Inspired by recent advances in vision-language foundation models, we investigate their utility for WSTVG, leveraging their zero-shot grounding capabilities. However, we find that a simple adaptation lacks essential spatio-temporal grounding abilities. To bridge this gap, we introduce Tubelet Referral Grounding (TRG), which connects textual queries to tubelets to enable spatio-temporal predictions. Despite its promise, TRG struggles with compositional action understanding and dense scene scenarios. To address these limitations, we propose STPro, a novel progressive learning framework with two key modules: (1) Sub-Action Temporal Curriculum Learning (SA-TCL), which incrementally builds compositional action understanding, and (2) Congestion-Guided Spatial Curriculum Learning (CG-SCL), which adapts the model to complex scenes by spatially increasing task difficulty. STPro achieves state-of-the-art results on three benchmark datasets, with improvements of 1.0% on VidSTG-Declarative and 3.0% on HCSTVG-v1. Code and models will be released publicly.

</details>

---

## 260. Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy

- [ ] Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy | https://cvpr.thecvf.com/virtual/2025/poster/33982

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33982

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Building an agent that can mimic human behavior patterns to accomplish various open-world tasks  is a long-term goal. To enable agents to effectively learn behavioral patterns across diverse tasks, a key challenge lies in modeling the intricate relationships among observations, actions, and language. To this end, we propose Optimus-2, a novel Minecraft agent that incorporates a Multimodal Large Language Model (MLLM) for high-level planning, alongside a Goal-Observation-Action Conditioned Policy (GOAP) for low-level control. GOAP contains (1) an Action-guided Behavior Encoder that models causal relationships between observations and actions at each timestep, then dynamically interacts with the historical observation-action sequence, consolidating it into fixed-length behavior tokens, and (2) an MLLM that aligns behavior tokens with open-ended language instructions to predict actions auto-regressively. Moreover, we introduce a high-quality Minecraft Goal-Observation-Action (MGOA) dataset, which contains 25,000 videos across 8 atomic tasks, providing about 30M goal-observation-action pairs. The automated construction method, along with the MGOA dataset, can contribute to the community's efforts in training Minecraft agents. Extensive experimental results demonstrate that Optimus-2 exhibits superior performance across atomic tasks, long-horizon tasks, and open-ended instruction tasks in Minecraft.

</details>

---

## 261. Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection

- [ ] Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection | https://cvpr.thecvf.com/virtual/2025/poster/33980

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33980

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have shown that large vision-language models (LVLMs) often suffer from the issue of object hallucinations (OH). To mitigate this issue, we introduce an efficient method that edits the model weights based on an unsafe subspace, which we call HalluSpace in this paper. With truthful and hallucinated text prompts accompanying the visual content as inputs, the HalluSpace can be identified by extracting the hallucinated embedding features and removing the truthful representations in LVLMs. By orthogonalizing the model weights, input features will be projected into the Null space of the HalluSpace to reduce OH, based on which we name our method Nullu. We reveal that HalluSpaces generally contain statistical bias and unimodal priors of the large language models (LLMs) applied to build LVLMs, which have been shown as essential causes of OH in previous studies. Therefore, null space projection suppresses the LLMs' priors to filter out the hallucinated features, resulting in contextually accurate outputs. Experiments show that our method can effectively mitigate OH across different LVLM families without extra inference costs and also show strong performance in general LVLM benchmarks. Codes will be released at \url{url}.

</details>

---

## 262. Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space

- [ ] Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space | https://cvpr.thecvf.com/virtual/2025/poster/33984

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33984

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

CLIP, a foundational vision-language model, has emerged as a powerful tool for open-vocabulary semantic segmentation. While freezing the text encoder preserves its powerful embeddings, recent studies show that fine-tuning both the text and image encoders jointly significantly enhances segmentation performance, especially for classes from open sets. In this work, we explain this phenomenon from the perspective of hierarchical alignment, since during fine-tuning, the hierarchy level of image embeddings shifts from image-level to pixel-level. We achieve this by leveraging hyperbolic space, which naturally encoders hierarchical structures. Our key observation is that, during fine-tuning, the hyperbolic radius of CLIP’s text embeddings decreases, facilitating better alignment with the pixel-level hierarchical structure of visual data. Building on this insight, we propose HyperCLIP, a novel fine-tuning strategy that adjusts the hyperbolic radius of the text embeddings through scaling transformations. By doing so, HyperCLIP equips CLIP with segmentation capability while introducing only a small number of learnable parameters. Our experiments demonstrate that HyperCLIP achieves state-of-the-art performance on open-vocabulary semantic segmentation tasks across three benchmarks, while fine-tuning only approximately 4\% of the total parameters of CLIP. More importantly, we observe that after adjustment, CLIP's text embeddings exhibit a relatively fixed hyperbolic radius across datasets, suggesting that the segmentation task has a characteristic level in hyperbolic space.

</details>

---

## 263. Anchor-Aware Similarity Cohesion in Target Frames Enables Predicting Temporal Moment Boundaries in 2D

- [ ] Anchor-Aware Similarity Cohesion in Target Frames Enables Predicting Temporal Moment Boundaries in 2D | https://cvpr.thecvf.com/virtual/2025/poster/33991

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33991

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video moment retrieval aims to locate specific moments from a video according to the query text. This task presents two main challenges: i) aligning the query and video frames at the feature level, and ii) projecting the query-aligned frame features to the start and end boundaries of the matching interval. Previous work commonly involves all frames in feature alignment, easy to cause aligning irrelevant frames with the query. Furthermore, they forcibly map visual features to interval boundaries but ignoring the information gap between them, yielding suboptimal performance. In this study, to reduce distraction from irrelevant frames, we designate an anchor frame as that with the maximum query-frame relevance measured by the established Vision-Language Model. Via similarity comparison between the anchor frame and the others, we produce a semantically compact segment around the anchor frame, which serves as a guide to align features of query and related frames. We observe that such a feature alignment will make similarity cohesive between target frames, which enables us to predict the interval boundaries by a single point detection in the 2D semantic similarity space of frames, thus well bridging the information gap between frame semantics and temporal boundaries. Experimental results across various datasets demonstrate that our approach significantly improves the alignment between queries and video frames while effectively predicting temporal moment boundaries. Especially, on QVHighlights Test and ActivityNet Captions datasets, our proposed approach achieves 3.8\% and 7.4\% respectively higher than current state-of-the-art R1@.7 performance. Codes will be released.

</details>

---

## 264. Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method

- [ ] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method | https://cvpr.thecvf.com/virtual/2025/poster/33993

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33993

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing Vision-Language Navigation (VLN) methods primarily focus on single-stage navigation, limiting their effectiveness in multi-stage and long-horizon tasks within complex and dynamic environments. To address these limitations, we propose a novel VLN task, named Long-Horizon Vision-Language Navigation (LH-VLN), which emphasizes long-term planning and decision consistency across consecutive subtasks. Furthermore, to support LH-VLN, we develop an automated data generation platform NavGen, which constructs datasets with complex task structures and improves data utility through a bidirectional, multi-granularity generation approach. To accurately evaluate complex tasks, we construct the Long-Horizon Planning and Reasoning in VLN (LHPR-VLN) benchmark consisting of 3,260 tasks with an average of 150 task steps, serving as the first dataset specifically designed for the long-horizon vision-language navigation task. Furthermore, we propose Independent Success Rate (ISR), Conditional Success Rate (CSR), and CSR weight by Ground Truth (CGT) metrics, to provide fine-grained assessments of task completion. To improve model adaptability in complex tasks, we propose a novel Multi-Granularity Dynamic Memory (MGDM) module that integrates short-term memory blurring with long-term memory retrieval to enable flexible navigation in dynamic environments. Our platform, benchmark and method supply LH-VLN with a robust data generation pipeline, comprehensive model evaluation dataset, reasonable metrics, and a novel VLN model, establishing a foundational framework for advancing LH-VLN.

</details>

---

## 265. Assessing and Learning Alignment of Unimodal Vision and Language Models

- [ ] Assessing and Learning Alignment of Unimodal Vision and Language Models | https://cvpr.thecvf.com/virtual/2025/poster/33994

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/33994

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

How well are unimodal vision and language models aligned? Although prior work have approached answering this question, their assessment methods do not directly translate to how these models are used in practical vision-language tasks. In this paper, we propose a direct assessment method, inspired by linear probing, to assess vision-language alignment. We identify that the degree of alignment of the SSL vision models depends on their SSL training objective, and we find that the clustering quality of SSL representations has a stronger impact on alignment performance than their linear separability. Next, we introduce Swift Alignment of Image and Language (SAIL), a efficient transfer learning framework that aligns pretrained unimodal vision and language models for downstream vision-language tasks. Since SAIL leverages the strengths of pretrained unimodal models, it requires significantly fewer ($\sim$6\%) paired image-text data for the multimodal alignment compared to models like CLIP which are trained from scratch. SAIL training only requires a single A100 GPU, $\sim$5 hours of training and can accommodate a batch size up to 32,768. SAIL achieves 73.4\% zero-shot accuracy on ImageNet (vs. CLIP's 72.7\%) and excels in zero-shot retrieval, complex reasoning, and semantic segmentation. Additionally, SAIL improves the language-compatibility of vision encoders that in turn enhance the performance of multimodal large language models.

</details>

---

## 266. Generative Zero-Shot Composed Image Retrieval

- [ ] Generative Zero-Shot Composed Image Retrieval | https://cvpr.thecvf.com/virtual/2025/poster/34000

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34000

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Composed Image Retrieval (CIR) is a vision-language task utilizing queries comprising images and textual descriptions to achieve precise image retrieval. This task seeks to find images that are visually similar to a reference image and incorporate specific changes or features described textually (visual delta). CIR enables a more flexible and user-specific retrieval by bridging visual data with verbal instructions. This paper introduces a novel generative method that augments Composed Image Retrieval by Composed Image Generation (CIG) to provide pseudo-target images. CIG utilizes a textual inversion network to map reference images into semantic word space, which generates pseudo-target images in combination with textual descriptions. These images serve as additional visual information, significantly improving the accuracy and relevance of retrieved images when integrated into existing retrieval frameworks. Experiments conducted across multiple CIR datasets and several baseline methods demonstrate improvements in retrieval performance, which shows the potential of our approach as an effective add-on for existing composed image retrieval.

</details>

---

## 267. VideoGEM: Training-free Action Grounding in Videos

- [ ] VideoGEM: Training-free Action Grounding in Videos | https://cvpr.thecvf.com/virtual/2025/poster/34020

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34020

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language foundation models have shown impressive capabilities across various zero-shot tasks, including training-free localization and grounding, primarily focusing on localizing objects in images. However, leveraging those capabilities to localize actions and events in videos is challenging as actions have less physical outline and are usually described by higher-level concepts.In this work, we propose VideoGEM, the first training-free action grounding method based on pretrained image- and video-language backbones. Namely, we adapt the self-self attention formulation of GEM to activity grounding. By doing so, we observe that high-level semantic concepts, such as actions, usually emerge in the higher layers of the image- and video-language models. We, therefore, propose a layer weighting in the self-attention path to prioritize higher layers. Additionally, we introduce a dynamic weighting method to automatically tune layer weights to capture each layer’s relevance to a specific prompt. Finally, we introduce a prompt decomposition, processing action, verb, and object prompts separately, resulting in a better localization of actions. We evaluate the proposed approach on three image- and video-language backbones, CLIP, OpenCLIP, and ViCLIP, and on four video grounding datasets, V-HICO, DALY, YouCook-Interactions, and GroundingYouTube, showing that the proposed training-free approach is able to outperform current trained state-of-the-art approaches for video grounding.

</details>

---

## 268. Knowledge-Aligned Counterfactual-Enhancement Diffusion Perception for Unsupervised Cross-Domain Visual Emotion Recognition

- [ ] Knowledge-Aligned Counterfactual-Enhancement Diffusion Perception for Unsupervised Cross-Domain Visual Emotion Recognition | https://cvpr.thecvf.com/virtual/2025/poster/34028

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34028

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Emotion Recognition (VER) is a critical yet challenging task aimed at inferring the emotional states of individuals based on visual cues. However, recent approaches predominantly focus on single domains, e.g., realistic images or stickers, limiting VER models' cross-domain generalizability. To address this limitation, we introduce an Unsupervised Cross-Domain Visual Emotion Recognition (UCDVER) task, which aims to generalize visual emotion recognition from the source domain (e.g., realistic images) to the low-resource target domain (e.g., stickers) in an unsupervised manner. Compared to the conventional unsupervised domain adaptation problems, UCDVER presents two key challenges: a significant emotional expression variability and an affective distribution shift. To mitigate these issues, we propose the  Knowledge-aligned Counterfactual-enhancement Diffusion Perception (KCDP) framework for UCDVER. Specifically, KCDP first leverages a vision-language model to align emotional representations in a shared knowledge space and guides diffusion models for improved visual affective perception. Furthermore, a Counterfactual-Enhanced Language-image Emotional Alignment (CLIEA) method generates high-quality pseudo-labels for the target domain. Extensive experiments demonstrate that our approach surpasses state-of-the-art models in both perceptibility and generalization, e.g., gaining 12% improvements over SOTA VER model TGCA-PVT.

</details>

---

## 269. HalLoc: Token-level Localization of Hallucinations for Vision Language Models

- [ ] HalLoc: Token-level Localization of Hallucinations for Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34044

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34044

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucinations pose a significant challenge to the reliability of large vision-language models, making their detection essential for ensuring accuracy in critical applications. Current detection methods often rely on computationally intensive models, leading to high latency and resource demands. Their definitive outcomes also fail to account for real-world scenarios where the line between hallucinated and truthful information is unclear. To address these issues, we propose HalLoc, a dataset designed for efficient, probabilistic hallucination detection. It features 150K token-level annotated samples, including hallucination types, across Visual Question Answering (VQA), instruction-following, and image captioning tasks. This dataset facilitates the development of models that detect hallucinations with graded confidence, enabling more informed user interactions. Additionally, we introduce a baseline model trained on HalLoc, offering low-overhead, concurrent hallucination detection during generation. The model can be seamlessly integrated into existing VLMs, improving reliability while preserving efficiency. The prospect of a robust plug-and-play hallucination detection module opens new avenues for enhancing the trustworthiness of vision-language models in real-world applications.

</details>

---

## 270. Personalized Preference Fine-tuning of Diffusion Models

- [ ] Personalized Preference Fine-tuning of Diffusion Models | https://cvpr.thecvf.com/virtual/2025/poster/34046

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34046

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

RLHF techniques like DPO can significantly improve the generation quality of text-to-image diffusion models. However, these methods optimize for a single reward that aligns model generation with population-level preferences, neglecting the nuances of individual users' beliefs or values. This lack of personalization limits the efficacy of these models. To bridge this gap, we introduce PPD, a multi-reward optimization objective that aligns diffusion models with personalized preferences. With PPD, a diffusion model learns the individual preferences of a population of users in a few-shot way, enabling generalization to unseen users. Specifically, our approach (1) leverages a vision-language model (VLM) to extract personal preference embeddings from a small set of pairwise preference examples, and then (2) incorporates the embeddings into diffusion models through cross attention. Conditioning on user embeddings, the text-to-image models are fine-tuned with the DPO objective, simultaneously optimizing for alignment with the preferences of multiple users. Empirical results demonstrate that our method effectively optimizes for multiple reward functions and can interpolate between them during inference. In real-world user scenarios, with as few as four preference examples from a new user, our approach achieves an average win rate of 76\% over Stable Cascade, generating images that more accurately reflect specific user preferences.

</details>

---

## 271. VisionArena: 230k Real World User-VLM Conversations with Preference Labels

- [ ] VisionArena: 230k Real World User-VLM Conversations with Preference Labels | https://cvpr.thecvf.com/virtual/2025/poster/34048

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34048

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The growing adoption and capabilities of vision-language models (VLMs) demand benchmarks that reflect real-world user interactions. We introduce VisionArena, the largest existing dataset of crowdsourced real-world conversations between users and VLMs. While most visual question-answering datasets focus on close-ended problems or synthetic scenarios, VisionArena contains a wide variety of closed and open ended problems across 230K conversations, 73K unique users, 138 languages, and 45 VLMs. VisionArena consists of VisionArena-Chat, a set of 200k single-turn and multi-turn chat logs with a VLM,  VisionArena-Battle, a set of 30K conversations between a user and 2 anonymous VLMs with preference votes, and VisionArena-Bench, an automatic benchmark consisting of 500 diverse user prompts which can be used to cheaply approximate model rankings.We analyze these datasets and highlight the types of question asked by users, the influence of style on user preference, and areas where models often fall short. We find that more open-ended questions like captioning and humor are heavily influenced by style, which causes certain models like Reka Flash and InternVL which are tuned for style to perform significantly better on these categories compared to other categories. We show that VisionArena-Chat and VisionArena-Battle can be used for post-training to align VLMs to human preferences through supervised fine-tuning. Compared to the popular instruction tuning dataset Llava-Instruct-158K, finetuning the same base model on VisionArena results in a a 17 point improvement on MMMU and a 46 point improvement on the WildVision human preference benchmark. Lastly, we show that running automatic VLM evaluation on VisionArena-Bench results in a model ranking which is largely consistent with major online preference benchmarks. We release both VisionArena and our finetuned model to further VLM development.

</details>

---

## 272. RoboGround: Robotic Manipulation with Grounded Vision-Language Priors

- [ ] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors | https://cvpr.thecvf.com/virtual/2025/poster/34049

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34049

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in robot manipulation have highlighted the potential of intermediate representations for improving policy generalization. In this work, we explore grounding masks as an effective intermediate representation, balancing two key advantages: (1) effective spatial guidance that specifies target objects and placement areas while also conveying information about object shape and size, enabling low-level policies to accurately interpret spatial information, and (2) broad generalization potential driven by large-scale vision-language models pretrained on diverse grounding datasets. We introduce RoboGround, a grounding-aware robotic policy that leverages grounding masks as an intermediate representation to guide policy networks in object manipulation tasks. To further explore and enhance generalization, we propose an automated pipeline for generating large-scale, simulated data with featuring a diverse set of objects and instructions. Extensive experiments show the value of our dataset and the effectiveness of grounding masks as intermediate guidance, significantly enhancing the generalization abilities of robot policies.

</details>

---

## 273. Object-aware Sound Source Localization via Audio-Visual Scene Understanding

- [ ] Object-aware Sound Source Localization via Audio-Visual Scene Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34050

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34050

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Sound source localization task aims to localize each region of sound-making objects in visual scenes. Recent methods, which rely on simple audio-visual correspondence, often struggle to accurately localize each object in complex environments, such as those with visually similar silent objects. To address these challenges, we propose a novel sound source localization framework, which incorporates detailed contextual information for fine-grained sound source localization. Our approach utilizes Multimodal Large Language Models (MLLMs) to generate detailed information through understanding audio-visual scenes. To effectively incorporate generated detail information, we propose two loss functions: Object-aware Contrastive Alignment (OCA) loss and Object Region Isolation (ORI) loss. By utilizing these losses, our method effectively performs precise localization through fine-grained audio-visual correspondence. Our extensive experiments on MUSIC and VGGSound datasets demonstrate significant improvements in both single- and multi-source sound localization. Our code and generated detail information will be made publicly available.

</details>

---

## 274. Adaptive Keyframe Sampling for Long Video Understanding

- [ ] Adaptive Keyframe Sampling for Long Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34055

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34055

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have enabled open-world visual understanding by injecting visual input as extra tokens into large language models (LLMs) as contexts. However, when the visual input changes from a single image to a long video, the above paradigm encounters difficulty because the vast amount of video tokens has significantly exceeded the maximal capacity of MLLMs. Therefore, existing video-based MLLMs are mostly established upon sampling a small portion of tokens from input data, which can cause key information to be lost and thus produce incorrect answers. This paper presents a simple yet effective algorithm named Adaptive Keyframe Sampling (AKS). It inserts a plug-and-play module known as keyframe selection, which aims to maximize the useful information with a fixed number of video tokens. We formulate keyframe selection as an optimization involving (1) the relevance between the keyframes and the prompt, and (2) the coverage of the keyframes over the video, and present an adaptive algorithm to approximate the best solution. Experiments on two long video understanding benchmarks validate that AKS improves video QA accuracy (beyond strong baselines) upon selecting informative keyframes. Our study reveals the importance of information pre-filtering in video-based MLLMs. Our code and models will be open-sourced.

</details>

---

## 275. Evaluating Model Perception of Color Illusions in Photorealistic Scenes

- [ ] Evaluating Model Perception of Color Illusions in Photorealistic Scenes | https://cvpr.thecvf.com/virtual/2025/poster/34064

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34064

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We study the perception of color illusions by vision-language models. Color illusion, where a person's visual system perceives color differently from actual color, is well-studied in human vision. However, it remains underexplored whether vision-language models (VLMs), trained on large-scale human data, exhibit similar perceptual biases when confronted with such color illusions. We propose an automated framework for generating color illusion images, resulting in RCID (Realistic Color Illusion Dataset), a dataset of 19,000 realistic illusion images. Our experiments show that all studied VLMs exhibit perceptual biases similar human vision. Finally, we train a model to distinguish both human perception and actual pixel differences.

</details>

---

## 276. Preserve or Modify? Context-Aware Evaluation for Balancing Preservation and Modification in Text-Guided Image Editing

- [ ] Preserve or Modify? Context-Aware Evaluation for Balancing Preservation and Modification in Text-Guided Image Editing | https://cvpr.thecvf.com/virtual/2025/poster/34063

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34063

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of vision-language and generative models has significantly advanced text-guided image editing, which seeks the preservation of core elements in the source image while implementing modifications based on the target text. However, existing metrics have a context-blindness problem, which is indiscriminately applying the same criteria on completely different contexts and biasing towards either modification or preservation. Directional CLIP similarity, the only metric that considers both source image and target text, is also biased towards modification aspects and attends to irrelevant editing regions of the image. We propose AugCLIP, a context-aware metric that adaptively coordinates preservation and modification aspects, depending on the specific context of a given source image and target text. This is done by deriving the CLIP representation of an ideally edited image, that preserves the source image with necessary modifications to align with target text. More specifically, using a multi-modal large language model, AugCLIP generates detailed textual descriptions of the source and target, then calculates a modification vector through a hyperplane in CLIP space that separates source and target attributes. Extensive experiments on five benchmark datasets, encompassing a diverse range of editing scenarios, show that AugCLIP aligns remarkably well with human evaluation standards, outperforming existing metrics. The code will be open-sourced for community use.

</details>

---

## 277. Distilling Multi-modal Large Language Models for Autonomous Driving

- [ ] Distilling Multi-modal Large Language Models for Autonomous Driving | https://cvpr.thecvf.com/virtual/2025/poster/34067

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34067

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autonomous driving demands safe motion planning, especially in critical "long-tail'' scenarios. Recent end-to-end autonomous driving systems leverage large language models (LLMs) as planners to improve generalizability to rare events. However, using LLMs at test time introduces high computational costs. To address this, we propose DiMA, an end-to-end autonomous driving system that maintains the efficiency of an LLM-free (or vision-based) planner while leveraging the world knowledge of an LLM. DiMA distills the information from a multi-modal LLM to a vision-based end-to-end planner through a set of specially designed surrogate tasks. Under a joint training strategy, a scene encoder common to both networks produces structured representations that are semantically grounded as well as aligned to the final planning objective. Notably, the LLM is optional at inference, enabling robust planning without compromising on efficiency. Training with DiMA results in a 37% reduction in the L2 trajectory error and an 80% reduction in the collision rate of the vision-based planner, as well as a 44% trajectory error reduction in long-tail scenarios. \ours also achieves state-of-the-art performance on the nuScenes planning benchmark.

</details>

---

## 278. Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection

- [ ] Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection | https://cvpr.thecvf.com/virtual/2025/poster/34073

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34073

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, vision-language models (e.g. CLIP) have demonstrated remarkable performance in zero-shot anomaly detection (ZSAD). By leveraging auxiliary data during training, these models can directly perform cross-category anomaly detection on target datasets, such as detecting defects on industrial product surfaces or identifying tumors in organ tissues. Existing approaches typically construct text prompts through either manual design or the optimization of learnable prompt vectors. However, these methods face several challenges: 1) Hand-crafted text prompts depend heavily on expert knowledge and require extensive trial and error; 2) The single-form learnable prompts is insufficient to capture the complex semantics of anomalies; and 3) The prompt space is poorly constrained, leading to suboptimal generalization performance on unseen categories. To address these issues, we propose Bayesian Prompt Flow Learning (Bayes-PFL), which models the prompt space as a learnable probability distribution from a Bayesian perspective. Specifically, a prompt flow module is designed to learn both image-specific and image-agnostic distributions, which are jointly utilized to regularize the text prompt space and enhance model's generalization on unseen categories. These learned distributions are then sampled to generate diverse text prompts, effectively covering the prompt space. Additionally, a residual cross-attention (RCA) module is introduced to better align dynamic text embeddings with fine-grained image features. Experimental results demonstrate that our method achieves state-of-the-art performance in ZSAD across 15 public industrial and medical anomaly detection datasets. Code will be released upon acceptance.

</details>

---

## 279. SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding

- [ ] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34083

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34083

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid development of Multi-modal Large Language Models (MLLMs), an increasing number of benchmarks have been established to evaluate the video understanding capabilities of these models. However, these benchmarks focus solely on standalone videos and assess only “visual elements” in videos, such as human actions and object states. In reality, contemporary videos often encompass complex and continuous narratives, typically presented as a series. To address this challenge, we propose SeriesBench, a benchmark consisting of 105 carefully curated narrative-driven series, covering 28 specialized tasks that require deep narrative understanding to solve. Specifically, we first select a diverse set of drama series spanning various genres. Then, we introduce a novel long-span narrative annotation method, combined with a full-information transformation approach to convert manual annotations into diverse task formats. To further enhance the model's capacity for detailed analysis of plot structures and character relationships within series, we propose a novel narrative reasoning framework, PC-DCoT. Extensive results on SeriesBench indicate that existing MLLMs still face significant challenges in understanding narrative-driven series, while PC-DCoT enables these MLLMs to achieve performance improvements. Overall, our SeriesBench and PC-DCoT highlight the critical necessity of advancing model capabilities for understanding narrative-driven series, guiding future MLLM development.

</details>

---

## 280. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment

- [ ] Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment | https://cvpr.thecvf.com/virtual/2025/poster/34085

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34085

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks—carefully crafted image-prompt pairs that compel the model to generate harmful content. In this work, we first highlight a critical safety gap, demonstrating that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model during decoding to defend against jailbreak attacks. Additionally, we provide a rigorous mathematical characterization of Immune, offering provable guarantees against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by $57.82$% and $16.78$% compared to the base MLLM and state-of-the-art defense strategy, respectively.

</details>

---

## 281. Self-Evolving Visual Concept Library using Vision-Language Critics

- [ ] Self-Evolving Visual Concept Library using Vision-Language Critics | https://cvpr.thecvf.com/virtual/2025/poster/34091

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34091

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We study the problem of building a visual concept library for visual recognition. Building effective visual concept libraries is challenging, as manual definition is labor-intensive, while relying solely on LLMs for concept generation can result in concepts that lack discriminative power or fail to account for the complex interactions between them. Our approach, ESCHER, takes a library learning perspective to iteratively discover and improve visual concepts. ESCHER uses a vision-language model (VLM) as a critic to iteratively refine the concept library, including accounting for interactions between concepts and how they affect downstream classifiers. By leveraging the in-context learning abilities of LLMs and the history of performance using various concepts, ESCHER dynamically improves its concept generation strategy based on the VLM critic's feedback. Finally, ESCHER does not require any human annotations, and is thus an automated plug-and-play framework. We empirically demonstrate the ability of ESCHER to learn a concept library for zero-shot, few-shot, and fine-tuning visual classification tasks. This work represents, to our knowledge, the first application of concept library learning to real-world visual tasks.

</details>

---

## 282. Vision-Language Models Do Not Understand Negation

- [ ] Vision-Language Models Do Not Understand Negation | https://cvpr.thecvf.com/virtual/2025/poster/34106

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34106

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Many practical vision-language applications require models that understand negation , e.g., when using natural language to retrieve images which contain certain objects but not others. Despite advancements in vision-language models (VLMs) through large-scale training, their ability to comprehend negation remains underexplored. This study addresses the question: how well do current VLMs understand negation? We introduce NegBench, a new benchmark designed to evaluate negation understanding across 18 task variations and 79k examples spanning image, video, and medical datasets. The benchmark consists of two core tasks designed to evaluate negation understanding in diverse multimodal settings: Retrieval with Negation and Multiple Choice Questions with Negated Captions. Our evaluation reveals that modern VLMs struggle significantly with negation, often performing at chance level. To address these shortcomings, we explore a data-centric approach wherein we finetune CLIP models on large-scale synthetic datasets containing millions of negated captions. We show that this approach can result in  a 10\% increase in recall on negated queries and a 40\% boost in accuracy on multiple-choice questions with negated captions.

</details>

---

## 283. RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete

- [ ] RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete | https://cvpr.thecvf.com/virtual/2025/poster/34105

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34105

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various multimodal contexts. However, their application in robotic scenarios, particularly for long-horizon manipulation tasks, reveals significant limitations. These limitations arise from the current MLLMs lacking three essential robotic brain capabilities: \textbf{Planning Capability}, which involves decomposing complex manipulation instructions into manageable sub-tasks; \textbf{Affordance Perception}, the ability to recognize and interpret the affordances of interactive objects; and \textbf{Trajectory Prediction}, the foresight to anticipate the complete manipulation trajectory necessary for successful execution. To enhance the robotic brain's core capabilities from abstract to concrete, we introduce \textbf{ShareRobot}, a high-quality heterogeneous dataset that labels multi-dimensional information such as task planning, object affordance, and end-effector trajectory. ShareRobot's diversity and accuracy have been meticulously refined by three human annotators. Building on this dataset, we developed \textbf{RoboBrain}, an MLLM-based model that combines robotic and general multi-modal data, utilizes a multi-stage training strategy, and incorporates long videos and high-resolution images to improve its robotic manipulation capabilities.Extensive experiments demonstrate that RoboBrain achieves state-of-the-art performance across various obotic tasks, highlighting its potential to advance robotic brain capabilities.

</details>

---

## 284. T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation

- [ ] T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation | https://cvpr.thecvf.com/virtual/2025/poster/34118

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34118

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-video (T2V) generative models have advanced significantly, yet their ability to compose different objects, attributes, actions, and motions into a video remains unexplored. Previous text-to-video benchmarks also neglect this important ability for evaluation. In this work, we conduct the first systematic study on compositional text-to-video generation. We propose T2V-CompBench, the first benchmark tailored for compositional text-to-video generation. T2V-CompBench encompasses diverse aspects of compositionality, including consistent attribute binding, dynamic attribute binding, spatial relationships, motion binding, action binding, object interactions, and generative numeracy. We further carefully design evaluation metrics of multimodal large language model (MLLM)-based, detection-based, and tracking-based metrics, which can better reflect the compositional text-to-video generation quality of seven proposed categories with 1400 text prompts. The effectiveness of the proposed metrics is verified by correlation with human evaluations. We also benchmark various text-to-video generative models and conduct in-depth analysis across different models and various compositional categories. We find that compositional text-to-video generation is highly challenging for current models, and we hope our attempt could shed light on future research in this direction.

</details>

---

## 285. AesthetiQ: Enhancing Graphic Layout Design via Aesthetic-Aware Preference Alignment of Multi-modal Large Language Models

- [ ] AesthetiQ: Enhancing Graphic Layout Design via Aesthetic-Aware Preference Alignment of Multi-modal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34123

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34123

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual layouts are essential in graphic design fields such as advertising, posters, and web interfaces. The application of generative models for content-aware layout generation has recently gained traction. However, these models fail to understand the contextual aesthetic requirements of layout design and do not align with human-like preferences, primarily treating it as a prediction task without considering the final rendered output. To overcome these problems, we offer Aesthetic-Aware Preference Alignment (AAPA), a novel technique to train a Multi-modal Large Language Model (MLLM) for layout prediction that uses MLLM's aesthetic preferences for Direct Preference Optimization over graphic layouts. We propose a data filtering protocol utilizing our layout-quality heuristics for AAPA to ensure training happens on high-quality layouts. Additionally, we introduce a novel evaluation metric that uses another MLLM to compute the win rate of the generated layout against the ground-truth layout based on aesthetics criteria. We also demonstrate the applicability of AAPA for MLLMs of varying scales (1B to 8B parameters) and LLM families (Qwen, Phi, InternLM). By conducting thorough qualitative and quantitative analyses, we verify the efficacy of our approach on two challenging benchmarks - Crello and Webui, showcasing 17%, and 16% improvement over current State-of-The-Art methods, thereby highlighting the potential of MLLMs in aesthetic-aware layout generation.

</details>

---

## 286. Robotic Visual Instruction

- [ ] Robotic Visual Instruction | https://cvpr.thecvf.com/virtual/2025/poster/34129

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34129

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, natural language has been the primary medium for human-robot interaction. However, its inherent lack of spatial precision for robotic control introduces challenges such as ambiguity and verbosity. To address these limitations, we introduce the Robotic Visual Instruction (RoVI) , a novel paradigm to guide robotic tasks through an object-centric, hand-drawn symbolic representation. RoVI effectively encodes spatial-temporal information into human-interpretable visual instructions through 2D sketches, utilizing arrows, circles, colors, and numbers to direct 3D robotic manipulation. To enable robots to understand RoVI better and generate precise actions based on RoVI, we present Visual Instruction Embodied Workflow (VIEW) , a pipeline formulated for RoVI-conditioned policies. This approach leverages Vision-Language Models (VLMs) to interpret RoVI inputs, decode spatial and temporal constraints from 2D pixel space via keypoint extraction, and then transform them into executable 3D action sequences. We additionally curate a specialized dataset of 15K instances to fine-tune small VLMs for edge deployment, enabling them to effectively learn RoVI capabilities. Our approach is rigorously validated across 11 novel tasks in both real and simulated environments, demonstrating significant generalization capability. Notably, VIEW achieves an 87.5% success rate in real-world scenarios involving unseen tasks that feature multi-step actions, with disturbances, and trajectory-following requirements. Code and Datasets in this paper will be released soon.

</details>

---

## 287. AdaMMS: Model Merging for Heterogeneous Multimodal Large Language Models with Unsupervised Coefficient Optimization

- [ ] AdaMMS: Model Merging for Heterogeneous Multimodal Large Language Models with Unsupervised Coefficient Optimization | https://cvpr.thecvf.com/virtual/2025/poster/34127

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34127

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, model merging methods have demonstrated powerful strengths in combining abilities on various tasks from multiple Large Language Models (LLMs). While previous model merging methods mainly focus on merging homogeneous models with identical architecture, they meet challenges when dealing with Multimodal Large Language Models (MLLMs) with inherent heterogeneous property, including differences in model architecture and the asymmetry in the parameter space. In this work, we propose AdaMMS, a novel model merging method tailored for heterogeneous MLLMs. Our method tackles the challenges in three steps: mapping, merging and searching. Specifically, we first design mapping function between models to apply model merging on MLLMs with different architecture. Then we apply linear interpolation on model weights to actively adapt the asymmetry in the heterogeneous MLLMs. Finally in the hyper-parameter searching step, we propose an unsupervised hyper-parameter selection method for model merging. As the first model merging method capable of merging heterogeneous MLLMs without labeled data, extensive experiments on various model combinations demonstrated that AdaMMS outperforms previous model merging methods on various vision-language benchmarks.

</details>

---

## 288. BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models

- [ ] BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34133

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34133

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in vision-language models (VLMs), such as CLIP, have demonstrated substantial success in self-supervised representation learning for vision tasks. However, effectively adapting VLMs to downstream applications remains challenging, as their accuracy often depends on time-intensive and expertise-demanding prompt engineering, while full model fine-tuning is costly. This is particularly true for biomedical images, which, unlike natural images, typically suffer from limited annotated datasets, unintuitive image contrasts, and nuanced visual features. Recent prompt learning techniques, such as Context Optimization (CoOp) intend to tackle these issues, but still fall short in generalizability. Meanwhile, explorations in prompt learning for biomedical image analysis are still highly limited. In this work, we propose BiomedCoOp, a novel prompt learning framework that enables efficient adaptation of BiomedCLIP for accurate and highly generalizable few-shot biomedical image classification. Our approach achieves effective prompt context learning by leveraging semantic consistency with average prompt ensembles from Large Language Models (LLMs) and knowledge distillation with a statistics-based prompt selection strategy. We conducted comprehensive validation of our proposed framework on 11 medical datasets across 9 modalities and 10 organs against existing state-of-the-art methods, demonstrating significant improvements in both accuracy and generalizability. The code will be publicly available upon acceptance of the submission.

</details>

---

## 289. BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices

- [ ] BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices | https://cvpr.thecvf.com/virtual/2025/poster/34136

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34136

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence and growing popularity of multimodal large language models (MLLMs) have significant potential to enhance various aspects of daily life, from improving communication to facilitating learning and problem-solving. Mobile phones, as essential daily companions, represent the most effective and accessible deployment platform for MLLMs, enabling seamless integration into everyday tasks. However, deploying MLLMs on mobile phones presents challenges due to limitations in memory size and computational capability, making it difficult to achieve smooth and real-time processing without extensive optimization. In this paper, we present BlueLM-V-3B , an algorithm and system co-design approach specifically tailored for the efficient deployment of MLLMs on mobile platforms. To be specific, we redesign the dynamic resolution scheme adopted by mainstream MLLMs and implement system optimization for hardware-aware deployment to optimize model inference on mobile phones. BlueLM-V-3B boasts the following key highlights: (1) Small Size : BlueLM-V-3B features a language model with 2.7B parameters and a vision encoder with 400M parameters. (2) Fast Speed : BlueLM-V-3B achieves a generation speed of 24.4 token/s on the MediaTek Dimensity 9300 processor with 4-bit LLM weight quantization. (3) Strong Performance : BlueLM-V-3B  has attained the highest average score of 66.1 on the OpenCompass benchmark among models with ≤ 4B parameters and surpassed a series of models with much larger parameter sizes (e.g., MiniCPM-V-2.6, InternVL2-8B).

</details>

---

## 290. SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation

- [ ] SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation | https://cvpr.thecvf.com/virtual/2025/poster/34140

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34140

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language temporal alignment is a crucial capability for human recognition and cognition in real-world scenarios. Although existing works have designed methods to capture vision-language correlations, they are limited by benchmark issues, including biased temporal distributions, imprecise annotations, and inadequate compositionally. To achieve fair evaluation and comprehensive exploration, our objective is to investigate and evaluate the ability of models to achieve alignment from a temporal perspective, specifically focusing on their capacity to synchronize visual scenarios with linguistic context in a temporally coherent manner. As a preliminary, we first present the statistical analysis of existing benchmarks and reveal the existing challenges from a decomposed perspective.To this end, we introduce $\textbf{SVLTA}$, a synthetic, large-scale, and compositional benchmark for vision-language temporal alignment derived via a well-designed and feasible control generation method within a simulation environment. The approach considers commonsense knowledge, process permutation, and constrained filtering, which generates reasonable, diverse, and balanced data distributions for diagnostic evaluations. Our experiments reveal diagnostic insights through the evaluations in temporal question answering, distributional shift sensitiveness, and temporal alignment adaptation.

</details>

---

## 291. RADIOv2.5: Improved Baselines for Agglomerative Vision Foundation Models

- [ ] RADIOv2.5: Improved Baselines for Agglomerative Vision Foundation Models | https://cvpr.thecvf.com/virtual/2025/poster/34144

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34144

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Agglomerative models have recently emerged as a powerful approach to training vision foundation models, leveraging multi-teacher distillation from existing models such as CLIP, DINO, and SAM. This strategy enables the creation of robust models more efficiently, combining the strengths of individual teachers while significantly reducing computational and resource demands. In this paper, we thoroughly analyze state-of-the-art agglomerative models, identifying critical challenges including resolution mode shifts, teacher imbalance, weak initializations, idiosyncratic teacher artifacts, and an excessive number of output tokens. To address these issues, we propose several novel solutions: multi-resolution training, mosaic augmentation, and improved balancing of teacher loss functions. Specifically, in the context of Vision Language Models, we introduce a token compression technique to maintain high-resolution information within a fixed token count. We release our top-performing models, available in multiple scales (-B, -L, and -H), alongside code and pretrained weights, to support further research and development in the community.

</details>

---

## 292. MultiVENT 2.0: A Massive Multilingual Benchmark for Event-Centric Video Retrieval

- [ ] MultiVENT 2.0: A Massive Multilingual Benchmark for Event-Centric Video Retrieval | https://cvpr.thecvf.com/virtual/2025/poster/34145

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34145

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Efficiently retrieving and synthesizing information from large-scale multimodal collections has become a critical challenge. However, existing video retrieval datasets suffer from scope limitations, primarily focusing on matching descriptive but vague queries with small collections of professionally edited, English-centric videos. To address this gap, we introduce \textbf{MultiVENT 2.0}, a large-scale, multilingual event-centric video retrieval benchmark featuring a collection of more than 218,000 news videos and over 3,900 queries targeting specific world events. These queries specifically target information found in the visual content, audio, embedded text, and text metadata of the videos, requiring systems leverage all these sources to succeed at the task. Preliminary results show that state-of-the-art vision-language models struggle significantly with this task, and while alternative approaches show promise, they are still insufficient to adequately address this problem. These findings underscore the need for more robust multimodal retrieval systems, as effective video retrieval is a crucial step towards multimodal content understanding and generation tasks.

</details>

---

## 293. Benchmarking Large Vision-Language Models via Directed Scene Graph for Comprehensive Image Captioning

- [ ] Benchmarking Large Vision-Language Models via Directed Scene Graph for Comprehensive Image Captioning | https://cvpr.thecvf.com/virtual/2025/poster/34150

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34150

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating detailed captions comprehending text-rich visual content in images has received growing attention for Large Vision-Language Models (LVLMs). However, few studies have developed benchmarks specifically tailored for detailed captions to measure their accuracy and comprehensiveness. In this paper, we introduce a detailed caption benchmark, termed as CompreCap, to evaluate the visual context from a directed scene graph view. Concretely, we first manually segment the image into semantically meaningful regions (i.e., semantic segmentation mask) according to common-object vocabulary, while also distinguishing attributes of objects within all those regions. Then directional relation labels of these objects are annotated to compose a directed scene graph that can well encode rich compositional information of the image. Based on our directed scene graph, we develop a pipeline to assess the generated detailed captions from LVLMs on multiple levels, including the object-level coverage, the accuracy of attribute descriptions, the score of key relationships, etc. Experimental results on the CompreCap dataset confirm that our evaluation method aligns closely with human evaluation scores across LVLMs. We will release the code and the dataset to support the community.

</details>

---

## 294. VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos

- [ ] VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos | https://cvpr.thecvf.com/virtual/2025/poster/34160

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34160

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-form video understanding has been a challenging task due to the high redundancy in video data and the abundance of query-irrelevant information. To tackle this challenge, we propose VideoTree, a training-free framework which builds a query-adaptive and hierarchical video representation for LLM reasoning over long-form videos. First, VideoTree extracts query-relevant information from the input video through an iterative process, progressively refining the selection of keyframes based on their relevance to the query. Furthermore, VideoTree leverages the inherent hierarchical structure of long video data, which is often overlooked by existing LLM-based methods. Specifically, we incorporate multi-granularity information into a tree-based representation, allowing VideoTree to extract query-relevant details from long videos in a coarse-to-fine manner. This enables the model to effectively handle a wide range of video queries with varying levels of detail. Finally, VideoTree aggregates the hierarchical query-relevant information within the tree structure and feeds it into an LLM reasoning model to answer the query. Our experiments show that VideoTree improves both reasoning accuracy and efficiency compared to existing methods. Specifically, VideoTree outperforms the existing training-free approaches on the popular EgoSchema and NExT-QA benchmarks with less inference time, achieving 61.1% and 75.6% accuracy on the test set without additional video-specific training. Moreover, on the long split of Video-MME benchmark (average 44 minutes), the training-free VideoTree framework achieves better performance than the strong proprietary GPT-4V model and many other MLLMs that were extensively trained on video data. Our code is provided in the supplementary materials and will be made public.

</details>

---

## 295. Non-Natural Image Understanding with Advancing Frequency-based Vision Encoders

- [ ] Non-Natural Image Understanding with Advancing Frequency-based Vision Encoders | https://cvpr.thecvf.com/virtual/2025/poster/34165

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34165

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have significantly enhanced cross-modal understanding capabilities by integrating visual encoders with textual embeddings, giving rise to multimodal large language models (MLLMs). However, these models struggle with non-natural images such as geometric and charts, particularly in fields like education and finance. Despite efforts to collect datasets and fine-tune the MLLMs, the gap with natural image understanding is still evident, and the cost of collecting large and diverse non-natural image datasets is high. To address this, we analyzed the limitations of transformer-based vision encoders(ViT) within existing MLLMs from a frequency perspective. Studies have shown that ViT models are less effective at capturing high-frequency information, impairing their ability to capture elements like points, lines, and angles in non-natural images. In response, we introduced FM-ViT, a frequency-modulated vision encoder that utilizes Fourier decomposition to extract high and low frequency components from self-attention features and re-weight them during tuning to non-natural images. In addition, we combine the features of CNN models with FM-ViT and propose EDGE, an MLLM with enhanced graphical encoders tailored for understanding non-natural images. Extensive experiments have confirmed the effectiveness of our FM-ViT and EDGE in 4 types of comprehension tasks (classification, retrieval, captioning, and question answering) on 3 types of non-natural images (geometric, charts, and functional).

</details>

---

## 296. ProAPO: Progressively Automatic Prompt Optimization for Visual Classification

- [ ] ProAPO: Progressively Automatic Prompt Optimization for Visual Classification | https://cvpr.thecvf.com/virtual/2025/poster/34163

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34163

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have made significant progress in image classification by training with large-scale paired image-text data. Their performances largely depend on the prompt quality. While recent methods show that visual descriptions generated by large language models (LLMs) enhance the generalization of VLMs, class-specific prompts may be inaccurate or lack discrimination due to the hallucination in LLMs. In this paper, we aim to find visually discriminative prompts for fine-grained categories with minimal supervision and no human-in-the-loop. An evolution-based algorithm is proposed to progressively optimize language prompts from task-specific templates to class-specific descriptions. Unlike optimizing templates, the search space shows an explosion in class-specific candidate prompts. This increases prompt generation costs, iterative times, and the overfitting problem. To this end, we first introduce several simple yet effective edit-based and evolution-based operations to generate diverse candidate prompts by one-time query of LLMs. Then, two sampling strategies are proposed to find a better initial search point and reduce traversed categories, saving iteration costs. Moreover, we apply a novel fitness score with entropy constraints to mitigate overfitting. In a challenging one-shot image classification setting, our method outperforms existing textual prompt-based methods and improves LLM-generated description methods across 13 datasets. Meanwhile, we demonstrate that our optimal prompts improve adapter-based methods and transfer effectively across different backbones. Our code is available at https://anonymous.4open.science/r/ProAPO.

</details>

---

## 297. VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection

- [ ] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection | https://cvpr.thecvf.com/virtual/2025/poster/34170

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34170

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of Large Vision Language Models (LVLMs) has significantly improved multimodal understanding, yet challenges remain in video reasoning tasks due to the scarcity of high-quality, large-scale datasets. Existing video question-answering (VideoQA) datasets often rely on costly manual annotations with insufficient granularity or automatic construction methods with redundant frame-by-frame analysis, limiting their scalability and effectiveness for complex reasoning. To address these challenges, we introduce VideoEspresso, a novel dataset that features VideoQA pairs preserving essential spatial details and temporal coherence, along with multimodal annotations of intermediate reasoning steps. Our construction pipeline employs a semantic-aware method to reduce redundancy, followed by generating QA pairs using GPT-4o. We further develop video Chain-of-Thought (CoT) annotations to enrich reasoning processes, guiding GPT-4o in extracting logical relationships from QA pairs and video content. To exploit the potential of high-quality VideoQA pairs, we propose a Hybrid LVLMs Collaboration framework, featuring a Frame Selector and a two-stage instruction fine-tuned reasoning LVLM. This framework adaptively selects core frames and performs CoT reasoning using multimodal evidence. Evaluated on our proposed benchmark with 14 tasks against 9 popular LVLMs, our method outperforms existing baselines on most tasks, demonstrating superior video reasoning capabilities.

</details>

---

## 298. Dual-Granularity Semantic Guided Sparse Routing Diffusion Model for General Pansharpening

- [ ] Dual-Granularity Semantic Guided Sparse Routing Diffusion Model for General Pansharpening | https://cvpr.thecvf.com/virtual/2025/poster/34171

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34171

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pansharpening aims at integrating complementary information from panchromatic and multispectral images. Available deep-learning based pansharpening methods typically perform exceptionally with particular satellite datasets. At the same time, it has been observed that these models also exhibit scene dependence, for example, if the majority of the training samples come from the urban scenes, the model's performance may decline in the river scene. To address the domain gap produced by varying satellite sensors and distinct scenes, we propose a dual-granularity semantic guided sparse routing diffusion model for general pansharpening. By utilizing the large Vision Language Models (VLMs) in the field of geoscience, i.e., GeoChat, we introduce the dual granularity semantics to generate dynamic sparse routing scores for adaptation of different satellite sensors and scenes. These scene-level and region-level dual-granularity semantic information serve as guidance to dynamically activating specialized experts within the diffusion model. Extensive experiments on WorldView-3, QuickBird, and GaoFen-2 datasets show the effectiveness of our proposed method. Notably, the proposed method outperforms the comparison approaches in adapting to new satellite sensors and scenes. The code will be available.

</details>

---

## 299. Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification

- [ ] Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification | https://cvpr.thecvf.com/virtual/2025/poster/34178

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34178

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-label classification is crucial for comprehensive image understanding, yet acquiring accurate annotations is challenging and costly. To address this, a recent study suggests exploiting unsupervised multi-label classification leveraging CLIP, a powerful vision-language model. Despite CLIP's proficiency, it suffers from view-dependent predictions and inherent bias, limiting its effectiveness. We propose a novel method that addresses these issues by leveraging multiple views near target objects, guided by Class Activation Mapping (CAM) of the classifier, and debiasing pseudo-labels derived from CLIP predictions. Our Classifier-guided CLIP Distillation (CCD) enables selecting multiple local views without extra labels and debiasing predictions to enhance classification performance. Experimental results validate our method's superiority over existing techniques across diverse datasets. The code will be publicly available.

</details>

---

## 300. Forensics-Bench: A Comprehensive Forgery Detection Benchmark Suite for Large Vision Language Models

- [ ] Forensics-Bench: A Comprehensive Forgery Detection Benchmark Suite for Large Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34175

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34175

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, the rapid development of AIGC has significantly boosted the diversities of fake media spread in the Internet, posing unprecedented threats to social security, politics, law, and etc.To detect the ever-increasingly **diverse** malicious fake media in the new era of AIGC, recent studies have proposed to exploit Large Vision Language Models (LVLMs) to design **robust** forgery detectors due to their impressive performance on a **wide** range of multimodal tasks.However, it still lacks a comprehensive benchmark designed to comprehensively assess LVLMs' discerning capabilities on forgery media.To fill this gap, we present Forensics-Bench, a new forgery detection evaluation benchmark suite to assess LVLMs across massive forgery detection tasks, requiring comprehensive recognition, location and reasoning capabilities on diverse forgeries.Forensics-Bench comprises $63,292$ meticulously curated multi-choicevisual questions, covering $112$ unique forgery detection types from $5$ perspectives: forgery semantics, forgery modalities, forgery tasks, forgery types and forgery models.We conduct thorough evaluations on $22$ open-sourced LVLMs and $3$ proprietary models GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet, highlighting the significant challenges of comprehensive forgery detection posed by Forensics-Bench.We anticipate that Forensics-Bench will motivate the community to advance the frontier of LVLMs, striving for all-around forgery detectors in the era of AIGC.

</details>

---

## 301. MLLM-as-a-Judge for Image Safety without Human Labeling

- [ ] MLLM-as-a-Judge for Image Safety without Human Labeling | https://cvpr.thecvf.com/virtual/2025/poster/34185

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34185

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image content safety has become a significant challenge with the rise of visual media on online platforms. Meanwhile, in the age of AI-generated content (AIGC), many image generation models are capable of producing harmful content, such as images containing sexual or violent material. Thus, it becomes crucial to identify such unsafe images based on established safety rules. Pre-trained Multimodal Large Language Models (MLLMs) offer potential in this regard, given their strong pattern recognition abilities. Existing approaches typically fine-tune MLLMs with human-labeled datasets, which however brings a series of drawbacks. First, relying on human annotators to label data following intricate and detailed guidelines is both expensive and labor-intensive. Furthermore, users of safety judgment systems may need to frequently update safety rules, making fine-tuning on human-based annotation more challenging. This raises the research question: Can we detect unsafe images by querying MLLMs in a zero-shot setting using a predefined safety constitution (a set of safety rules)? Our research showed that simply querying pre-trained MLLMs does not yield satisfactory results. This lack of effectiveness stems from factors such as the subjectivity of safety rules, the complexity of lengthy constitutions, and the inherent biases in the models. To address these challenges, we propose a MLLM-based method includes objectifying safety rules, assessing the relevance between rules and images, making quick judgments based on debiased token probabilities with logically complete yet simplified precondition chains for safety rules, and conducting more in-depth reasoning with cascaded chain-of-thought processes if necessary. Experiment results demonstrate that our method is highly effective for zero-shot image safety judgment tasks.

</details>

---

## 302. One-shot 3D Object Canonicalization based on Geometric and Semantic Consistency

- [ ] One-shot 3D Object Canonicalization based on Geometric and Semantic Consistency | https://cvpr.thecvf.com/virtual/2025/poster/34193

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34193

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D object canonicalization is a fundamental task, essential for a variety of downstream tasks. Existing methods rely on either cumbersome manual processes or priors learned from extensive, per-category training samples. Real-world datasets, however, often exhibit long-tail distributions, challenging existing learning-based methods, especially in categories with limited samples. We address this by introducing the first one-shot category-level object canonicalization framework, requiring only a single canonical model as a reference (the "prior model") for each category. To canonicalize any object, our framework first extracts semantic cues with large language models (LLMs) and vision-language models (VLMs) to establish correspondences with the prior model. We introduce a novel loss function to enforce geometric and semantic consistency, aligning object orientations precisely despite significant shape variations. Moreover, we adopt a support-plane strategy to reduce search space for initial poses and utilize a semantic relationship map to select the canonical pose from multiple hypotheses. Extensive experiments on multiple datasets demonstrate that our framework achieves state-of-the-art performance and validate key design choices. Using our framework, we create the Canonical Objaverse Dataset (COD), canonicalizing 33K samples in the Objaverse-LVIS dataset, underscoring the effectiveness of our framework on handling large-scale datasets.

</details>

---

## 303. Bayesian Test-Time Adaptation for Vision-Language Models

- [ ] Bayesian Test-Time Adaptation for Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34196

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34196

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation with pre-trained vision-language models, such as CLIP, aims to adapt the model to new, potentially out-of-distribution test data.  Existing methods calculate the similarity between visual embedding and learnable class embeddings, which are initialized by text embeddings, for zero-shot image classification. In this work, we first analyze this process based on Bayes theorem, and observe that the core factors influencing the final prediction are the likelihood and the prior. However, existing methods essentially focus on adapting class embeddings to adapt likelihood, but they often ignore the importance of prior. To address this gap, we propose a novel approach, \textbf{B}ayesian \textbf{C}lass \textbf{A}daptation (BCA), which in addition to continuously updating class embeddings to adapt likelihood, also uses the posterior of incoming samples to continuously update the prior for each class embedding. This dual updating mechanism allows the model to better adapt to distribution shifts and achieve higher prediction accuracy. Our method not only surpasses existing approaches in terms of performance metrics but also maintains superior inference rates and memory usage, making it highly efficient and practical for real-world applications.

</details>

---

## 304. Olympus: A Universal Task Router for Computer Vision Tasks

- [ ] Olympus: A Universal Task Router for Computer Vision Tasks | https://cvpr.thecvf.com/virtual/2025/poster/34212

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34212

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Olympus, a new approach that transforms Multimodal Large Language Models (MLLMs) into a unified framework capable of handling a wide array of computer vision tasks. Utilizing a controller MLLM, Olympus delegates over 20 specialized tasks across images, videos, and 3D objects to dedicated modules. This instruction-based routing enables complex workflows through chained actions without the need for training heavy generative models. Olympus easily integrates with existing MLLMs, expanding their capabilities with comparable performance. Experimental results demonstrate that Olympus achieves an average routing accuracy of 94.75% across 20 tasks and precision of 91.82% in chained action scenarios, showcasing its effectiveness as a universal task router that can solve a diverse range of computer vision tasks.

</details>

---

## 305. Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation

- [ ] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation | https://cvpr.thecvf.com/virtual/2025/poster/34226

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34226

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper tackles the problem of video question answering (VideoQA), a task that often requires multi-step reasoning and a profound understanding of spatial-temporal dynamics. While large video-language models perform well on benchmarks, they often lack explainability and spatial-temporal grounding. In this paper, we propose A gent- o f- T houghts D istillation ( AoTD ), a method that enhances models by incorporating automatically generated Chain-of-Thoughts (CoTs) into the instruction-tuning process. Specifically, we leverage an agent-based system to decompose complex questions into sub-tasks, and address them with specialized vision models, the intermediate results are then treated as reasoning chains. We also introduce a verification mechanism using a large language model (LLM) to ensure the reliability of generated CoTs. Extensive experiments demonstrate that AoTD improves the performance on multiple-choice and open-ended benchmarks.

</details>

---

## 306. Efficient Transfer Learning for Video-language Foundation Models

- [ ] Efficient Transfer Learning for Video-language Foundation Models | https://cvpr.thecvf.com/virtual/2025/poster/34228

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34228

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models provide a robust foundation for efficient transfer learning across various downstream tasks. In the field of video action recognition, mainstream approaches often introduce additional parameter modules to capture temporal information. While the increased model capacity brought by these additional parameters helps better fit the video-specific inductive biases, existing methods require learning a large number of parameters and are prone to catastrophic forgetting of the original generalizable knowledge. In this paper, we propose a simple yet effective Multi-modal Spatio-Temporal Adapter (MSTA) to improve the alignment between representations in the text and vision branches, achieving a balance between general knowledge and task-specific knowledge. Furthermore, to mitigate over-fitting and enhance generalizability, we introduce a spatio-temporal description-guided consistency constraint. This constraint involves feeding template inputs (i.e., ``a video of $\{\textbf{cls}\}$'') into the trainable language branch, while LLM-generated spatio-temporal descriptions are input into the pre-trained language branch, enforcing consistency between the outputs of the two branches. This mechanism prevents over-fitting to downstream tasks and improves the distinguishability of the trainable branch within the spatio-temporal semantic space. We evaluate the effectiveness of our approach across four tasks: zero-shot transfer, few-shot learning, base-to-novel generalization, and fully-supervised learning. Compared to many state-of-the-art methods, our MSTA achieves outstanding performance across all evaluations, while using only 2-7\% of the trainable parameters in the original model.

</details>

---

## 307. A3: Few-shot Prompt Learning of Unlearnable Examples with Cross-Modal Adversarial Feature Alignment

- [ ] A3: Few-shot Prompt Learning of Unlearnable Examples with Cross-Modal Adversarial Feature Alignment | https://cvpr.thecvf.com/virtual/2025/poster/34231

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34231

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the age of pervasive machine learning applications, protecting digital content from unauthorized use has become a pressing concern. Unlearnable examples (UEs), i.e., data modified with imperceptible perturbations to inhibit model training while preserving human usability, have emerged as a promising approach. However, existing UE methods assume unauthorized trainers have extensive exposure to UEs or that models are trained from scratch, which may not hold in practical scenarios, This paper investigates the effectiveness of UEs under the few-shot learning paradigm, pitching it against visual prompt learning (VPL) models that leverage pretrained vision-language models (VLMs), like CLIP, capable of generalizing to new classes with minimal data. To address this, we introduce an adaptive UE framework to generate unlearnable examples that specifically target the VPL process. In addition, we propose a novel UE countermeasure, A3, with cross-modal adversarial feature alignment, specifically designed to circumvent UEs under few-shot VPL. Experimental evaluations on 7 datasets show that A3 outperforms existing VPL methods, achieving up to 33% higher performance in learning from UEs. For example, in the scenario involving $\ell_\infty$-bounded EM perturbations, A3 has an average harmonic mean accuracy across 7 datasets of 82.43%, compared to CoCoOp's baseline of 65.47%. Our findings highlight the limitations of existing UEs against VPL and lay the foundation for future data protection mechanisms.

</details>

---

## 308. Cropper: Vision-Language Model for Image Cropping through In-Context Learning

- [ ] Cropper: Vision-Language Model for Image Cropping through In-Context Learning | https://cvpr.thecvf.com/virtual/2025/poster/34238

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34238

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The goal of image cropping is to identify visually appealing crops in an image. Conventional methods are trained on specific datasets and fail to adapt to new requirements. Recent breakthroughs in large vision-language models (VLMs) enable visual in-context learning without explicit training. However, downstream tasks with VLMs remain under explored. In this paper, we propose an effective approach to leverage VLMs for image cropping. First, we propose an efficient prompt retrieval mechanism for image cropping to automate the selection of in-context examples. Second, we introduce an iterative refinement strategy to iteratively enhance the predicted crops. The proposed framework, we refer to as  Cropper, is applicable to a wide range of cropping tasks, including free-form cropping, subject-aware cropping, and aspect ratio-aware cropping. Extensive experiments demonstrate that Cropper significantly outperforms state-of-the-art methods across several benchmarks.

</details>

---

## 309. MagicQuill: An Intelligent Interactive Image Editing System

- [ ] MagicQuill: An Intelligent Interactive Image Editing System | https://cvpr.thecvf.com/virtual/2025/poster/34255

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34255

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As a highly practical application, image editing encounters a variety of user demands and thus prioritizes excellent ease of use.In this paper, we unveil MagicQuill, an integrated image editing system designed to support users in swiftly actualizing their creativity.Our system starts with a streamlined yet functionally robust interface, enabling users to articulate their ideas (e.g., inserting elements, erasing objects, altering color, etc.) with just a few strokes.These interactions are then monitored by a multimodal large language model (MLLM) to anticipate user intentions in real time, bypassing the need for prompt entry.Finally, we apply the powerful diffusion prior, enhanced by a carefully learned two-branch plug-in module, to process the editing request with precise control.We will release the entire system to facilitate the community.

</details>

---

## 310. CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering

- [ ] CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering | https://cvpr.thecvf.com/virtual/2025/poster/34268

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34268

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have garnered widespread attention from researchers due to their remarkable understanding and generation capabilities in visual language tasks (e.g., visual question answering). However, the rapid pace of knowledge updates in the real world makes offline training of MLLMs costly, and when faced with non-stationary data streams, MLLMs suffer from catastrophic forgetting during learning. In this paper, we propose an MLLMs-based dual momentum Mixture-of-Experts ($\texttt{CL-MoE}$) framework for continual visual question answering. We integrate MLLMs with continual learning to utilize the rich commonsense knowledge in LLMs.We introduce a Dual-Router MoE (RMoE) to select the global and local experts using task-level and instance-level routers, to robustly assign weights to the experts most appropriate for the task. Then, we design a dynamic Momentum MoE (MMoE) to update the parameters of experts dynamically based on the relationships between the experts and tasks, so that the model can absorb new knowledge while maintaining existing knowledge. The extensive experimental results indicate that our method achieves state-of-the-art performance on 10 VQA tasks, proving the effectiveness of our approach. The codes and weights will be released on GitHub.

</details>

---

## 311. ICT: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models

- [ ] ICT: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34264

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34264

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the recent breakthroughs achieved by Large Vision Language Models (LVLMs) in understanding and responding to complex visual-textual contexts, their inherent hallucination tendencies limit their practical application in real-world scenarios that demand high levels of precision. Existing methods typically either fine-tune the LVLMs using additional data, which incurs extra costs in manual annotation and computational resources or perform comparisons at the decoding stage, which may eliminate useful language priors for reasoning while introducing inference time overhead. Therefore, we propose ICT, a lightweight, training-free method that calculates an intervention direction to shift the model's focus towards different levels of visual information, enhancing its attention to high-level and fine-grained visual details.  During the forward pass stage, the intervention is applied to the attention heads that encode the overall image information and the fine-grained object details, effectively mitigating the phenomenon of overly language priors, and thereby alleviating hallucinations. Extensive experiments demonstrate that ICT achieves strong performance with a small amount of data and generalizes well across different datasets and models. Our code will be public.

</details>

---

## 312. Conformal Prediction and MLLM aided Uncertainty Quantification in Scene Graph Generation

- [ ] Conformal Prediction and MLLM aided Uncertainty Quantification in Scene Graph Generation | https://cvpr.thecvf.com/virtual/2025/poster/34281

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34281

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scene Graph Generation (SGG) aims to represent visual scenes by identifying objects and their pairwise relationships, providing a structured understanding of image content. However, inherent challenges like long-tailed class distributions and prediction variability necessitate uncertainty quantification in SGG for its practical viability. In this paper, we introduce a novel Conformal Prediction (CP) based framework, adaptive to any existing SGG method, for quantifying their predictive uncertainty by constructing well-calibrated prediction sets over their generated scene graphs. These scene graph prediction sets are designed to achieve statistically rigorous coverage guarantees. Additionally, to ensure these prediction sets contain the most practically interpretable scene graphs, we design an effective MLLM-based post-processing strategy for selecting the most visually and semantically plausible scene graphs within these prediction sets. We show that our proposed approach can produce diverse possible scene graphs from an image, assess the reliability of SGG methods, and improve overall SGG performance.

</details>

---

## 313. Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video

- [ ] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video | https://cvpr.thecvf.com/virtual/2025/poster/34289

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34289

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a unified approach to understanding dynamic scenes from casual videos. Large pretrained vision models, such as vision-language, video depth prediction, motion tracking, and segmentation models, offer promising capabilities. However, training a single model for comprehensive 4D understanding remains challenging. We introduce Uni4D, a multi-stage optimization framework that harnesses multiple pretrained models to advance dynamic 3D modeling, including static/dynamic reconstruction, camera pose estimation, and dense 3D motion tracking. Our results show state-of-the-art performance in dynamic 4D modeling with superior visual quality. Notably, Uni4D requires no retraining or fine-tuning, highlighting the effectiveness of repurposing large visual models for 4D understanding.

</details>

---

## 314. VideoICL: Confidence-based Iterative In-context Learning for Out-of-Distribution Video Understanding

- [ ] VideoICL: Confidence-based Iterative In-context Learning for Out-of-Distribution Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34294

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34294

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in video large multimodal models (LMMs) have significantly improved their video understanding and reasoning capabilities. However, their performance drops on out-of-distribution (OOD) tasks that are underrepresented in training data. Traditional methods like fine-tuning on OOD datasets are impractical due to high computational costs. While In-context learning (ICL) with demonstration examples has shown promising generalization performance in language tasks and image-language tasks without fine-tuning, applying ICL to video-language tasks faces challenges due to the limited context length in Video LMMs, as videos require longer token lengths. To address these issues, we propose VideoICL, a novel video in-context learning framework for OOD tasks that introduces a similarity-based relevant example selection strategy and a confidence-based iterative inference approach. This allows to select the most relevant examples and rank them based on similarity, to be used for inference. If the generated response has low confidence, our framework selects new examples and performs inference again, iteratively refining the results until a high-confidence response is obtained. This approach improves OOD video understanding performance by extending effective context length without incurring high costs. The experimental results on multiple benchmarks demonstrate significant performance gains, especially in domain-specific scenarios, laying the groundwork for broader video comprehension applications.

</details>

---

## 315. Zero-Shot 4D Lidar Panoptic Segmentation

- [ ] Zero-Shot 4D Lidar Panoptic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/34293

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34293

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot 4D segmentation of arbitrary objects in Lidar is of crucial importance for embodied navigation, with applications ranging from streaming perception to semantic mapping and localization. However, the primary challenge in advancing research and developing generalized, versatile methods for spatio-temporal scene understanding in Lidar lies in the scarcity of datasets that provide the necessary diversity and scale of annotations.To overcome these challenges, we propose SAL-4D (Segment Anything in Lidar-4D), a method that utilizes multi-modal sensory robotic setups as a bridge to distill recent developments in Video Object Segmentation (VOS) in conjunction with off-the-shelf Vision-Language foundation models to Lidar.  We utilize VOS models to pseudo-label tracklets in short video sequences, annotate these tracklets with sequence-level CLIP tokens, and lift them to the 4D Lidar space using calibrated multi-modal sensory setups to distill them to our SAL-4D model. Due to temporally consistent predictions, we outperform prior art in 3D Zero-Shot Lidar Panoptic Segmentation (LPS) over 5 PQ, and unlock Zero-Shot 4D LPS.

</details>

---

## 316. Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models

- [ ] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34306

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34306

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) demonstrate enhanced capabilities and reliability by reasoning more, evolving from Chain-of-Thought prompting to product-level solutions like OpenAI o1.  Despite various efforts to improve LLM reasoning, high-quality long-chain reasoning data and optimized training pipelines still remain inadequately explored in vision-language tasks. In this paper, we present Insight-V, an early effort to 1) scalably produce long and robust reasoning data for complex multi-modal tasks, and 2) an effective training pipeline to enhance the reasoning capabilities of multi-modal large language models (MLLMs). Specifically, to create long and structured reasoning data without human labor, we design a two-step pipeline with a progressive strategy to generate sufficiently long and diverse reasoning paths and a multi-granularity assessment method to ensure data quality. We observe that directly supervising MLLMs with such long and complex reasoning data will not yield ideal reasoning ability. To tackle this problem, we design a multi-agent system consisting of a reasoning agent dedicated to performing long-chain reasoning and a summary agent trained to judge and summarize reasoning results. We further incorporate an iterative DPO algorithm to enhance the reasoning agent's generation stability and quality. Based on the popular LLaVA-NeXT model, our method shows an average improvement of 7.5% across seven challenging multi-modal benchmarks requiring visual reasoning.  We also achieve a 4.2% improvement on a stronger base MLLM, highlighting the potential to further advance state-of-the-art models. Benefiting from our multi-agent system, Insight-V can also easily maintain or improve performance on perception-focused multi-modal tasks. We will make our data and code publicly available to promote future research in this emerging field.

</details>

---

## 317. M-LLM Based Video Frame Selection for Efficient Video Understanding

- [ ] M-LLM Based Video Frame Selection for Efficient Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34303

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34303

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in \acf{mllms} show promising results in video reasoning. Popular \ac{mllm} frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an \ac{mllm}, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream \ac{mllm} may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight \ac{mllm}-based frame selection method that adaptively select frames that are more relevant to users' queries. The selected frames are then digested by a frozen downstream \acf{videollm} for visual reasoning and question answering. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a \ac{mllm}; (ii) Temporal signal, in which multiple frames selection by prompting \ac{llm} using the captions of all frame candidates. Empirical results show that the proposed \ac{mllm} video frame selector improves the performances various downstream \ac{videollm} across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.

</details>

---

## 318. PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models

- [ ] PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34313

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34313

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (VLMs) have been extended to understand both images and videos. Visual token compression is leveraged to reduce the considerable token length of visual inputs. To meet the needs of different tasks, existing high-performance models usually process images and videos separately with different token compression strategies, limiting the capabilities of combining images and videos. To this end, we extend each image into a "static" video and introduce a unified token compression strategy called Progressive Visual Token Compression (PVC), where the tokens of each frame are progressively encoded and adaptively compressed to supplement the information not extracted from previous frames. Video tokens are efficiently compressed with exploiting the inherent temporal redundancy. Images are repeated as static videos, and the spatial details can be gradually supplemented in multiple frames. PVC unifies the token compressing of images and videos. With a limited number of tokens per frame (64 tokens by default), spatial details and temporal changes can still be preserved. Experiments show that our model achieves state-of-the-art performance across various video understanding benchmarks, including long video tasks and fine-grained short video tasks. Meanwhile, our unified token compression strategy incurs no performance loss on image benchmarks, particularly in detail-sensitive tasks.

</details>

---

## 319. LOGICZSL: Exploring Logic-induced Representation for Compositional Zero-shot Learning

- [ ] LOGICZSL: Exploring Logic-induced Representation for Compositional Zero-shot Learning | https://cvpr.thecvf.com/virtual/2025/poster/34316

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34316

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Compositional zero-shot learning (CZSL) aims to recognize unseen attribute-object compositions by learning the primitive concepts ( i.e. , attribute and object) from the training set. While recent works achieve impressive results in CZSL by leveraging large vision-language models like CLIP, they ignore the rich semantic relationships between primitive concepts and their compositions. In this work, we propose LOGICZSL, a novel logic-induced learning framework to explicitly model the semantic relationships. Our logic-induced learning framework formulates the relational knowledge constructed from large language models as a set of logic rules, and grounds them onto the training data. Our logic-induced losses are complementary to the widely used CZSL losses, therefore can be employed to inject the semantic information into any existing CZSL methods. Extensive experimental results show that our method brings significant performance improvements across diverse datasets ( i.e. , CGQA, UT-Zappos50K, MIT-States) with strong CLIP-based methods and settings ( i.e. , Close World, Open World). Codes will be publicly released.

</details>

---

## 320. BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding

- [ ] BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34323

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34323

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) have shown promising progress in various video understanding tasks. However, their potential for long-form video analysis is limited by their high computational resource requirements and constrained context windows. Traditional methods, particularly uniform frame sampling, often allocate resources to irrelevant content, reducing effectiveness in real-world scenarios. This paper introduces BOLT to BOost Large VLMs without additional Training through an extensive study of frame selection strategies for large VLMs. To provide a realistic evaluation of VLMs in long-form video understanding, we first present a multi-source retrieval evaluation setting. Our findings show that uniform sampling significantly underperforms when dealing with noisy contexts, highlighting the importance of selecting the right frames. Furthermore, we introduce several frame selection strategies based on query-frame similarity and analyze their effectiveness in enhancing VLM performance without retraining. We find that inverse transform sampling with refined query descriptions yields the most substantial performance improvement, boosting accuracy on the Video-MME benchmark from 49.94% to 53.8%. Our code will be released.

</details>

---

## 321. JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration

- [ ] JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration | https://cvpr.thecvf.com/virtual/2025/poster/34333

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34333

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-centric perception systems for autonomous driving often struggle with unpredictable and coupled weather degradations in the wild. Current solutions are often limited, as they either depend on specific degradation priors or suffer from significant domain gaps. To enable robust and autonomous operation in real-world conditions, we propose JarvisIR, a VLM-powered agent that leverages the VLM (e.g., Llava-Llama3) as a controller to manage multiple expert restoration models. To further enhance system robustness, reduce hallucinations, and improve generalizability in real-world adverse weather, JarvisIR employs a novel two-stage framework consisting of supervised fine-tuning and human feedback alignment. Specifically, to address the lack of paired data in real-world scenarios, the human feedback alignment enables the VLM to be fine-tuned effectively on large-scale real-world data in an unsupervised manner. To support the training and evaluation of JarvisIR, we introduce CleanBench, a comprehensive dataset consisting of high-quality and large-scale instruction-responses pairs, including 150K synthetic entries and 80K real entries. Extensive experiments demonstrate that JarvisIR exhibits superior decision-making and restoration capabilities. Compared with existing methods, it achieves a 50\% improvement in the average of all perception metrics on CleanBench-Real. Furthermore, it effectively supports high-level tasks, such as semantic segmentation and object detection.

</details>

---

## 322. DocVLM: Make Your VLM an Efficient Reader

- [ ] DocVLM: Make Your VLM an Efficient Reader | https://cvpr.thecvf.com/virtual/2025/poster/34337

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34337

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) excel in diverse visual tasks but face challenges in document understanding, which requires fine-grained text processing. While typical visual tasks perform well with low-resolution inputs, reading-intensive applications demand high-resolution, resulting in significant computational overhead. Using OCR-extracted text in VLM prompts partially addresses this issue but underperforms compared to full-resolution counterpart, as it lacks the complete visual context needed for optimal performance.We introduce DocVLM, a method that integrates an OCR-based modality into VLMs to enhance document processing while preserving original weights. Our approach employs an OCR encoder to capture textual content and layout, compressing these into a compact set of learned queries incorporated into the VLM. Comprehensive evaluations across leading VLMs show that DocVLM significantly reduces reliance on high-resolution images for document understanding.In limited-token regimes (448$\times$448), DocVLM with 64 learned queries improves DocVQA results from 56.0% to 86.6% when integrated with InternVL2 and from 84.4% to 91.2% with Qwen2-VL. In LLaVA-OneVision, DocVLM achieves improved results while using 80% less image tokens. The reduced token usage allows processing multiple pages effectively, showing impressive zero-shot results on DUDE and state-of-the-art performance on MP-DocVQA, highlighting DocVLM’s potential for applications requiring high-performance and efficiency.

</details>

---

## 323. Overcoming Shortcut Problem in VLM for Robust Out-of-Distribution Detection

- [ ] Overcoming Shortcut Problem in VLM for Robust Out-of-Distribution Detection | https://cvpr.thecvf.com/virtual/2025/poster/34353

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34353

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs), such as CLIP, have shown remarkable capabilities in downstream tasks. However, the coupling of semantic information between the foreground and the background in images leads to significant shortcut issues that adversely affect out-of-distribution (OOD) detection abilities. When confronted with a background OOD sample, VLMs are prone to misidentifying it as in-distribution (ID) data. In this paper, we analyze the OOD problem from the perspective of shortcuts in VLMs and propose OSPCoOp which includes background decoupling and mask-guided region regularization. We first decouple images into ID-relevant and ID-irrelevant regions and utilize the latter to generate a large number of augmented OOD background samples as pseudo-OOD supervision. We then use the masks from background decoupling to adjust the model's attention, minimizing its focus on ID-irrelevant regions. To assess the model's robustness against background interference, we introduce a new OOD evaluation dataset, ImageNet-Bg, which solely consists of background images with all ID-relevant regions removed. Our method demonstrates exceptional performance in few-shot scenarios, achieving strong results even in one-shot setting, and outperforms existing methods.

</details>

---

## 324. Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events

- [ ] Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events | https://cvpr.thecvf.com/virtual/2025/poster/34354

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34354

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The commonsense reasoning capabilities of vision-language models (VLMs), especially in abductive reasoning and defeasible reasoning, remain poorly understood. Most benchmarks focus on typical visual scenarios, making it difficult to discern whether model performance stems from keen perception and reasoning skills, or reliance on pure statistical recall. We argue that by focusing on atypical events in videos, clearer insights can be gained on the core capabilities of VLMs. Explaining and understanding such out-of-distribution events requires models to extend beyond basic pattern recognition and regurgitation of their prior knowledge. To this end, we introduce BlackSwanSuite, a benchmark for evaluating VLMs' ability to reason about unexpected events through abductive and defeasible tasks. Our tasks artificially limit the amount of visual information provided to models while questioning them about hidden unexpected events, or provide new visual information that could change an existing hypothesis about the event. We curate a comprehensive benchmark suite comprising over 3,800 MCQ, 4,900 generative and 6,700 yes/no tasks, spanning 1,655 videos. After extensively evaluating various state-of-the-art VLMs, including GPT-4o and Gemini 1.5 Pro, as well as open-source VLMs such as LLaVA-Video, we find significant performance gaps of up to 32% from humans on these tasks. Our findings reveal key limitations in current VLMs, emphasizing the need for enhanced model architectures and training strategies.

</details>

---

## 325. Enhanced OoD Detection through Cross-Modal Alignment of Multi-Modal Representations

- [ ] Enhanced OoD Detection through Cross-Modal Alignment of Multi-Modal Representations | https://cvpr.thecvf.com/virtual/2025/poster/34361

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34361

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prior research on out-of-distribution detection (OoDD) has primarily focused on single-modality models. Recently, with the advent of large-scale pretrained vision-language models such as CLIP, OoDD methods utilizing such multi-modal representations through zero-shot and prompt learning strategies have emerged. However, these methods typically involve either freezing the pretrained weights or only partially tuning them, which can be suboptimal for downstream datasets. In this paper, we highlight that multi-modal fine-tuning (MMFT) can achieve notable OoDD performance. Despite some recent works demonstrating the impact of fine-tuning methods for OoDD, there remains significant potential for performance improvement. We investigate the limitation of naive fine-tuning methods, examining why they fail to fully leverage the pretrained knowledge. Our empirical analysis suggests that this issue could stem from the modality gap within in-distribution (ID) embeddings. To address this, we propose a training objective that enhances cross-modal alignment by regularizing the distances between image and text embeddings of ID data. This adjustment helps in better utilizing pretrained textual information by aligning similar semantics from different modalities (i.e., text and image) more closely in the hyperspherical representation space. We theoretically demonstrate that the proposed regularization corresponds to the maximum likelihood estimation of an energy-based model on a hypersphere. Utilizing ImageNet-1k OoD benchmark datasets, we show that our method, combined with post-hoc OoDD approaches leveraging pretrained knowledge (e.g., NegLabel), significantly outperforms existing methods, achieving state-of-the-art OoDD performance and leading ID accuracy.

</details>

---

## 326. Dual Semantic Guidance for Open Vocabulary Semantic Segmentation

- [ ] Dual Semantic Guidance for Open Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/34368

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34368

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation aims to enable models to segment arbitrary categories. Currently, though pre-trained Vision-Language Models (VLMs) like CLIP have established a robust foundation for this task by learning to match text and image representations from large-scale data, their lack of pixel-level recognition necessitates further fine-tuning. Most existing methods leverage text as a guide to achieve pixel-level recognition. However, the inherent biases in text semantic descriptions and the lack of pixel-level supervisory information make it challenging to fine-tune CLIP-based models effectively. This paper considers leveraging image-text data to simultaneously capture the semantic information contained in both image and text, thereby constructing Dual Semantic Guidance and corresponding pixel-level pseudo annotations. Particularly, the visual semantic guidance is enhanced via explicitly exploring foreground regions and minimizing the influence of background. The dual semantic guidance is then jointly utilized to fine-tune CLIP-based segmentation models, achieving decent fine-grained recognition capabilities. As the comprehensive evaluation shows, our method outperforms state-of-art results with large margins, on eight commonly used datasets with/without background.

</details>

---

## 327. Patient-Level Anatomy Meets Scanning-Level Physics: Personalized Federated Low-Dose CT Denoising Empowered by Large Language Model

- [ ] Patient-Level Anatomy Meets Scanning-Level Physics: Personalized Federated Low-Dose CT Denoising Empowered by Large Language Model | https://cvpr.thecvf.com/virtual/2025/poster/34378

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34378

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reducing radiation doses benefits patients, however, the resultant low-dose computed tomography (LDCT) images often suffer from clinically unacceptable noise and artifacts. While deep learning (DL) shows promise in LDCT reconstruction, it requires large-scale data collection from multiple clients, raising privacy concerns. Federated learning (FL) has been introduced to address these privacy concerns; however, current methods are typically tailored to specific scanning protocols, which limits their generalizability and makes them less effective for unseen protocols. To address these issues, we propose SCAN-PhysFed, a novel SCanning- and ANatomy-level personalized Physicis-Driven Federated learning paradigm for LDCT reconstruction. Since the noise distribution in LDCT data is closely tied to scanning protocols and anatomical structures being scanned, we design a dual-level physics-informed way to address these challenges. Specifically, we incorporate physical and anatomical prompts into our physics-informed hypernetworks to capture scanning- and anatomy-specific information, enabling dual-level physics-driven personalization of imaging features. These prompts are derived from the scanning protocol and the radiology report generated by a medical large language model (MLLM), respectively. Subsequently, client-specific decoders project these dual-level personalized imaging features back into the image domain. Besides, to tackle the challenge of unseen data, we introduce a novel protocol vector-quantization strategy (PVQS), which ensures consistent performance across new clients by quantifying the unseen scanning code as one of the codes in the scanning codebook. Extensive experimental results demonstrate the superior performance of SCAN-PhysFed on public datasets.

</details>

---

## 328. Language-Guided Salient Object Ranking

- [ ] Language-Guided Salient Object Ranking | https://cvpr.thecvf.com/virtual/2025/poster/34388

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34388

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Salient Object Ranking (SOR) aims to study human attention shifts across different objects in the scene. It is a challenging task, as it requires comprehension of the relations among the salient objects in the scene. However, existing works often overlook such relations or model them implicitly. In this work, we observe that when Large Vision-Language Models (LVLMs) describe a scene, they usually focus on the most salient object first, and then discuss the relations as they move on to the next (less salient) one. Based on this observation, we propose a novel Language-Guided Salient Object Ranking approach (named LG-SOR), which utilizes the internal knowledge within the LVLM-generated language descriptions, i.e., semantic relation cues and the implicit entity order cues, to facilitate saliency ranking. Specifically, we first propose a novel Text-Guided Visual Modulation (TGVM) module to incorporate semantic information in the description for saliency ranking. TGVM controls the flow of linguistic information to the visual features, suppresses noisy background image features, and enables propagation of useful textual features. We then propose a novel Text-Aware Visual Reasoning (TAVR) module to enhance model reasoning in object ranking, by explicitly learning a multimodal graph based on the entity and relation cues derived from the description. Extensive experiments demonstrate superior performances of our model on two SOR benchmarks.

</details>

---

## 329. Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data

- [ ] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data | https://cvpr.thecvf.com/virtual/2025/poster/34390

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34390

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite impressive advancements in various multimodal tasks, vision-language models (VLMs) still struggle with compositional understanding due to limited exposure to training samples that contain subtle variations within paired examples. With advances in multimodal generative models, a natural solution is to generate synthetic samples with subtle variations for training VLMs. However, generating and training on synthetic samples with subtle variations presents two challenges: difficulty in accurately creating precise variations and inconsistency in cross-modal alignment quality. To address these challenges, we propose SVD-GT (Subtle Variation Data Generation and Training), which integrates image feature injection into a text-to-image generative model to enhance the quality of synthetic variations and employs an adaptive margin loss to differentiate samples using adaptive margins, which help filter out potentially incorrect synthetic samples and focus the learning on informative hard samples. Evaluations on four compositional understanding benchmarks demonstrate that SVD-GT significantly improves the compositionality of VLMs, boosting the average accuracy of CLIP by over 8% across all benchmarks and outperforming state-of-the-art methods by 2% on three benchmarks.

</details>

---

## 330. MMRL: Multi-Modal Representation Learning for Vision-Language Models

- [ ] MMRL: Multi-Modal Representation Learning for Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34413

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34413

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained Vision-Language Models (VLMs) have become essential for transfer learning across diverse tasks. However, adapting these models with limited few-shot data often leads to overfitting, diminishing their performance on new tasks. To tackle this issue, we propose a novel Multi-Modal Representation Learning (MMRL) framework that introduces a shared, learnable, and modality-agnostic representation space. MMRL projects the space tokens to text and image representation tokens, facilitating more effective multi-modal interactions. Unlike previous approaches that solely optimize class token features, MMRL integrates representation tokens at higher layers of the encoders—where dataset-specific features are more prominent—while preserving generalized knowledge in the lower layers. During training, both representation and class features are optimized, with trainable projection layer applied to the representation tokens, whereas the class token projection layer remains frozen to retain pre-trained knowledge. Furthermore, a regularization term is introduced to align the class features and text features with the zero-shot features from the frozen VLM, thereby safeguarding the model's generalization capacity. For inference, a decoupling strategy is employed, wherein both representation and class features are utilized for base classes, while only the class features, which retain more generalized knowledge, are used for new tasks. Extensive experiments across 15 datasets demonstrate that MMRL outperforms state-of-the-art methods, achieving a balanced trade-off between task-specific adaptation and generalization.

</details>

---

## 331. Font-Agent: Enhancing Font Understanding with Large Language Models

- [ ] Font-Agent: Enhancing Font Understanding with Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34417

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34417

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of generative models has significantly advanced the font generation. However, limited explorations have been devoted into the evaluation and interpretability of graphical fonts. Especially, existing quality assessment models can only provide basic visual capabilities, such as recognizing clarity and brightness, lacking in-depth explanation.To address these limitations, we firstly constructed a large-scale multimodal dataset comprising 135,000 font-text pairs named Diversity Font Dataset (DFD). This dataset includes a wide range of generated font types and annotations including language descriptions and quality assessments, providing a strong basis for training and evaluating font analysis models.Based on the dataset, we developed a Vision Language Model (VLM)-based Font-Agent with the aim of improving font quality assessment and offering interpretative question-answering capabilities. Alongside the original visual encoder in VLM, we integrated a Edge Aware Traces (EAT) Module to capture detailed edge information of font strokes and components. Furthermore, we introduce a Dynamic Direct Preference Optimization (D-DPO) strategy to facilitate efficient model fine-tuning. Experimental outcomes showcase that Font-Agent achieves state-of-the-art performance on the the established dataset. To further assess the generalization of our algorithm, we conducted evaluation on several public datasets. The results highlight the notable advantage of Font-Agent in both assessing the quality of generated fonts and comprehending their contents.

</details>

---

## 332. F^3OCUS - Federated Finetuning of Vision-Language Foundation Models with Optimal Client Layer Updating Strategy via Multi-objective Meta-Heuristics

- [ ] F^3OCUS - Federated Finetuning of Vision-Language Foundation Models with Optimal Client Layer Updating Strategy via Multi-objective Meta-Heuristics | https://cvpr.thecvf.com/virtual/2025/poster/34425

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34425

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Effective training of large Vision-Language Models (VLMs) on resource-constrained client devices in Federated Learning (FL) requires the usage of parameter-efficient fine-tuning (PEFT) strategies. To this end, we demonstrate the impact of two factors \textit{viz.}, client-specific layer importance score that selects the most important VLM layers for fine-tuning and inter-client layer diversity score that encourages diverse layer selection across clients for optimal VLM layer selection. We first theoretically motivate and leverage the principal eigenvalue magnitude of layerwise Neural Tangent Kernels and show its effectiveness as client-specific layer importance score. Next, we propose a novel layer updating strategy dubbed \textbf{F$^3$OCUS} that jointly optimizes the layer importance and diversity factors by employing a data-free, multi-objective, meta-heuristic optimization on the server. We explore 5 different meta-heuristic algorithms and compare their effectiveness for selecting model layers and adapter layers towards PEFT-FL. Furthermore, we release a new MedVQA-FL dataset involving overall 707,962 VQA triplets and 9 modality-specific clients and utilize it to train and evaluate our method. Overall, we conduct more than 10,000 client-level experiments on 6 Vision-Language FL task settings involving 58 medical image datasets and 4 different VLM architectures of varying sizes to demonstrate the effectiveness of the proposed method.

</details>

---

## 333. Explaining in Diffusion: Explaining a Classifier with Diffusion Semantics

- [ ] Explaining in Diffusion: Explaining a Classifier with Diffusion Semantics | https://cvpr.thecvf.com/virtual/2025/poster/34429

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34429

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Classifiers are crucial to computer vision, yet their "black box" nature obscures the decision-making process, limiting the ability to trace the influence of individual features. Traditional interpretability methods, including GAN-based attribute editing, are constrained by domain and resource demands, often requiring extensive labeling and model-specific training. Text-to-image diffusion models, while promising for broader applications, lack precise semantics for classifier interpretation without extensive user input. We introduce DiffEx, a training-free framework that combines large language models (LLMs) and pre-trained diffusion models to improve classifier explainability. DiffEx leverages Vision-Language Models (VLMs) to build a comprehensive, hierarchical semantic corpus and applies a novel algorithm to rank impactful features, offering broad and fine-grained attributes that influence classifier scores. Our experiments show that DiffEx provides nuanced, interpretable insights across diverse domains, including medical diagnostics, making it versatile, scalable, and well-suited for understanding complex classifiers in critical applications.

</details>

---

## 334. Explainable Saliency: Articulating Reasoning with Contextual Prioritization

- [ ] Explainable Saliency: Articulating Reasoning with Contextual Prioritization | https://cvpr.thecvf.com/virtual/2025/poster/34435

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34435

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Deep saliency models, which predict what parts of an image capture our attention, are often like black boxes. This limits their use, especially in areas where understanding why a model makes a decision is crucial. Our research tackles this challenge by building a saliency model that can not only identify what is important in an image, but also explain its choices in a way that makes sense to humans. We achieve this by using vision-language models to reason about images and by focusing the model's attention on the most crucial information using a contextual prioritization mechanism. Unlike prior approaches that rely on fixation descriptions or soft-attention based semantic aggregation, our method directly models the reasoning steps involved in saliency prediction, generating selectively prioritized explanations clarify why specific regions are prioritized. Comprehensive evaluations demonstrate the effectiveness of our model in generating high-quality saliency maps and coherent, contextually relevant explanations. This research is a step towards more transparent and trustworthy AI systems that can help us understand and navigate the world around us.

</details>

---

## 335. EfficientLLaVA: Generalizable Auto-Pruning for Large Vision-language Models

- [ ] EfficientLLaVA: Generalizable Auto-Pruning for Large Vision-language Models | https://cvpr.thecvf.com/virtual/2025/poster/34439

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34439

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While multimodal large language models demonstrate strong performance in complex reasoning tasks, they pose significant challenges related to model complexity during deployment, especially for resource-limited devices. In this paper, we propose an automatic pruning method for large vision-language models to enhance the efficiency of multimodal reasoning. Conventional methods rely on the training data of the original model to select the proper pruning ratio for different network components. However, these methods are impractical for large vision-language models due to the unaffordable search costs caused by web-scale training corpus. In contrast, our approach only leverages a small number of samples to search for the desired pruning policy by maximizing its generalization ability on unknown training data while maintaining the model accuracy, which enables the achievement of an optimal trade-off between accuracy and efficiency for large visual language models. Specifically, we formulate the generalization gap of the pruning strategy using the structural risk minimization principle. Based on both task performance and generalization capability, we iteratively search for the optimal pruning policy within a given search space and optimize the vision projector to evolve the search space with higher upper bound of performance. We conduct extensive experiments on the ScienceQA, Vizwiz, MM-vet, and LLaVA-Bench datasets for the task of visual question answering. Using only 64 samples for pruning policy search, EfficientLLaVA achieves an accuracy of 83.05\% on ScienceQA, along with a $\times$ 1.8 speedup compared to the dense LLaVA-v1.5-7B model.

</details>

---

## 336. Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training

- [ ] Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training | https://cvpr.thecvf.com/virtual/2025/poster/34443

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34443

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In rapidly evolving field of vision-language models (VLMs), contrastive language-image pre-training (CLIP) has made significant strides, becoming foundation for various downstream tasks. However, relying on one-to-one (image, text) contrastive paradigm to learn alignment from large-scale messy web data, CLIP faces a serious myopic dilemma, resulting in biases towards monotonous short texts and shallow visual expressivity. To overcome these issues, this paper advances CLIP into one novel holistic paradigm, by updating both diverse data and alignment optimization. To obtain colorful data with low cost, we use image-to-text captioning to generate multi-texts for each image, from multiple perspectives, granularities, and hierarchies. Two gadgets are proposed to encourage textual diversity. To match such (image, multi-texts) pairs, we modify the CLIP image encoder into multi-branch, and propose multi-to-multi contrastive optimization for image-text part-to-part matching. As a result, diverse visual embeddings are learned for each image, bringing good interpretability and generalization. Extensive experiments and ablations across over ten benchmarks indicate that our holistic CLIP significantly outperforms existing myopic CLIP, including image-text retrieval, open-vocabulary classification, and dense visual tasks. Code for holistic CLIP will be released upon publication, to further promote the prosperity of VLMs.

</details>

---

## 337. A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for Accelerating Large VLMs

- [ ] A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for Accelerating Large VLMs | https://cvpr.thecvf.com/virtual/2025/poster/34456

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34456

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have shown remarkable success across various multi-modal tasks, yet large VLMs encounter significant efficiency challenges due to processing numerous visual tokens. A promising approach to accelerating large VLM inference is using partial information, such as attention maps from specific layers, to assess token importance and prune less essential tokens. However, our study reveals three key insights: (i) Partial attention information is insufficient for accurately identifying critical visual tokens, resulting in suboptimal performance, especially at low token retention ratios; (ii) Global attention information, such as the attention map aggregated across all layers, more effectively preserves essential tokens and maintains performance under aggressive pruning. However, it requires a full inference pass, which increases computational load and is therefore impractical in existing methods; and (iii) The global attention map aggregated from a small VLM closely resembles that of a large VLM, suggesting an efficient alternative. Based on these findings, we introduce \underline{\textbf{S}}mall VLM \underline{\textbf{G}}uidance for \underline{\textbf{L}}arge VLMs (\textbf{SGL}). Specifically, we employ the aggregated attention map from a small VLM guide the pruning of visual tokens in a large VLM. Additionally, we develop a small VLM early exiting mechanism to make full use of the small VLM's predictions, dynamically invoking the larger VLM only when necessary, yielding a superior trade-off between accuracy and computational cost. Extensive evaluations across 11 benchmarks demonstrate the effectiveness and generalizability of our method, achieving up to 91\% pruning ratio for visual tokens while retaining competitive performance.

</details>

---

## 338. EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues

- [ ] EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues | https://cvpr.thecvf.com/virtual/2025/poster/34460

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34460

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automated analysis of vast Earth observation data via interactive Vision-Language Models (VLMs) can unlock new opportunities for environmental monitoring, disaster response, and {resource management}. Existing generic VLMs do not perform well on Remote Sensing data, while the recent Geo-spatial VLMs remain restricted to a fixed resolution and few sensor modalities. In this paper, we introduce EarthDial, a conversational assistant specifically designed for Earth Observation (EO) data, transforming complex, multi-sensory Earth observations into interactive, natural language dialogues. EarthDial supports multi-spectral, multi-temporal, and multi-resolution imagery, enabling a wide range of remote sensing tasks, including classification, detection, captioning, question answering, visual reasoning, and visual grounding.To achieve this, we introduce an extensive instruction tuning dataset comprising over 11.11M instruction pairs covering RGB, Synthetic Aperture Radar (SAR), and multispectral modalities such as Near-Infrared (NIR) and infrared. Furthermore, EarthDial handles bi-temporal and multi-temporal sequence analysis for applications like change detection.Our extensive experimental results on 37 downstream applications demonstrate that EarthDial outperforms existing generic and domain-specific models, achieving better generalization across various EO tasks. Our codes and data will be publicly released.

</details>

---

## 339. Re-thinking Temporal Search for Long-Form Video Understanding

- [ ] Re-thinking Temporal Search for Long-Form Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34465

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34465

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Efficient understanding of long-form videos remains a significant challenge in computer vision. In this work, we revisit temporal search paradigms for long-form video understanding, studying a fundamental issue pertaining to all state-of-the-art (SOTA) long-context vision-language models (VLMs). In particular, our contributions are two-fold: **First**, we formulate temporal search as a **Long Video Haystack** problem, i.e., finding a minimal set of relevant frames (typically one to five) among tens of thousands of frames from real-world long videos given specific queries. To validate our formulation, we create **LV-Haystack**, the first benchmark containing 3,874 human-annotated instances with fine-grained evaluation metrics for assessing keyframe search quality and computational efficiency. Experimental results on LV-Haystack highlight a significant research gap in temporal search capabilities, with SOTA keyframe selection methods achieving only 2.1% temporal F1 score on the LVBench subset.**Next**, inspired by visual search in images, we re-think temporal searching and propose a lightweight keyframe searching framework, $T^*$ , which casts the expensive temporal search as a spatial search problem. $T^*$  leverages superior visual localization capabilities typically used in images and introduces an adaptive zooming-in mechanism that operates across both temporal and spatial dimensions. Our extensive experiments show that when integrated with existing methods, $T^*$  significantly improves SOTA long-form video understanding performance. Specifically, under an inference budget of 32 frames, $T^*$  improves GPT-4o's performance from 50.5% to **53.1%** and LLaVA-OneVision-72B's performance from 56.5% to **62.4%** on LongVideoBench XL subset. Our PyTorch code, benchmark dataset and models are included in the Supplementary material.

</details>

---

## 340. VladVA: Discriminative Fine-tuning of LVLMs

- [ ] VladVA: Discriminative Fine-tuning of LVLMs | https://cvpr.thecvf.com/virtual/2025/poster/34477

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34477

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastively-trained Vision-Language Models (VLMs) like CLIP have become the de facto approach for discriminative vision-language representation learning. However, these models have limited language understanding, often exhibiting a ``bag of words'' behavior. At the same time, Large Vision-Language Models (LVLMs), which combine vision encoders with LLMs, have been shown capable of detailed vision-language reasoning, yet their autoregressive nature renders them less suitable for discriminative tasks.In this work, we propose to combine "the best of both worlds": a new training approach for discriminative fine-tuning of LVLMs that results in strong discriminative and compositional capabilities. Essentially, our approach converts a generative LVLM into a discriminative one, unlocking its capability for powerful image-text discrimination combined with enhanced language understanding.Our contributions include (1) A carefully designed training/optimization framework that utilizes image-text pairs of variable length and granularity for training the model with both contrastive and next-token prediction losses. This is accompanied by ablation studies that justify the necessity of our framework's components. (2) A parameter-efficient adaptation method using a combination of soft prompting and LoRA adapters. (3) Significant improvements over state-of-the-art CLIP-like models of similar size, including standard image-text retrieval benchmarks and notable gains in compositionality.

</details>

---

## 341. Bridging Modalities: Improving Universal Multimodal Retrieval by Multimodal Large Language Models

- [ ] Bridging Modalities: Improving Universal Multimodal Retrieval by Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34475

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34475

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Universal Multimodal Retrieval (UMR) aims to enable search across various modalities using a unified model, where queries and candidates can consist of pure text or images, or a combination of both. Previous work has attempted to adopt multimodal large language models (MLLMs) to realize UMR using only text data. However, our preliminary experiments demonstrate that a more diverse multimodal training data can further unlock the potential of MLLMs. Despite its effectiveness, the existing multimodal training data is highly imbalanced in terms of modality, which motivates us to develop a training data synthesis pipeline and construct a large-scale, high-quality fused-modal training dataset. Based on the synthetic training data, we develop the General Multimodal Embedder (GME), an MLLM-based dense retriever designed for UMR. Furthermore, we construct a comprehensive UMR Benchmark (UMRB) to evaluate the effectiveness of our approach. Experimental results show that our method achieves state-of-the-art performance among existing UMR methods. Last, we provide in-depth analyses of model scaling, training strategies, and perform ablation studies on both the model and synthetic data.

</details>

---

## 342. Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding

- [ ] Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34493

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34493

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of Multimodal Large Language Models (MLLMs) has significantly impacted various multimodal tasks. However, these models face challenges in tasks that require spatial understanding within 3D environments. Efforts to enhance MLLMs, such as incorporating point cloud features, have been made, yet a considerable gap remains between the models' learned representations and the inherent complexity of 3D scenes. This discrepancy largely stems from the training of MLLMs on predominantly 2D data, which restricts their effectiveness in comprehending 3D spaces. To address this issue, in this paper, we propose a novel generalist model, i.e., Video-3D LLM, for 3D scene understanding. By treating 3D scenes as dynamic videos and incorporating 3D position encoding into these representations, our Video-3D LLM aligns video representations with real-world spatial contexts more accurately. Additionally, we have implemented a maximum coverage sampling technique to optimize the balance between computational costs and performance efficiency. Extensive experiments demonstrate that our model achieves state-of-the-art performance on several 3D scene understanding benchmarks, including ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, and SQA3D.

</details>

---

## 343. 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models

- [ ] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34514

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34514

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning 4D language fields to enable time-sensitive, open-ended language queries in dynamic scenes is essential for many real-world applications. While LangSplat successfully grounds CLIP features into 3D Gaussian representations, achieving precision and efficiency in 3D static scenes, it lacks the ability to handle dynamic 4D fields as CLIP, designed for static image-text tasks, cannot capture temporal dynamics in videos. Real-world environments are inherently dynamic, with object semantics evolving over time. Building a precise 4D language field necessitates obtaining pixel-aligned, object-wise video features, which current vision models struggle to achieve. To address these challenges, we propose 4D LangSplat, which learns 4D language fields to handle time-agnostic or time-sensitive open-vocabulary queries in dynamic scenes efficiently. 4D LangSplat bypasses learning the language field from vision features and instead learns directly from text generated from object-wise video captions via Multimodal Large Language Models (MLLMs). Specifically, we propose a multimodal object-wise video prompting method, consisting of visual and text prompts that guide MLLMs to generate detailed, temporally consistent, high-quality captions for objects throughout a video. These captions are encoded using a Large Language Model into high-quality sentence embeddings, which then serve as pixel-aligned, object-specific feature supervision, facilitating open-vocabulary text queries through shared embedding spaces. Recognizing that objects in 4D scenes exhibit smooth transitions across states, we further propose a status deformable network to model these continuous changes over time effectively. Our results across multiple benchmarks demonstrate that 4D LangSplat attains precise and efficient results for both time-sensitive and time-agnostic open-vocabulary queries.

</details>

---

## 344. Relation-Rich Visual Document Generator for Visual Information Extraction

- [ ] Relation-Rich Visual Document Generator for Visual Information Extraction | https://cvpr.thecvf.com/virtual/2025/poster/34517

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34517

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite advances in Large Language Models (LLMs) and Multimodal LLMs (MLLMs) for visual document understanding (VDU), visual information extraction (VIE) from relation-rich documents remains challenging due to the layout diversity and limited training data. While existing synthetic document generators attempt to address data scarcity, they either rely on manually designed layouts and templates, or adopt rule-based approaches that limit layout diversity. Besides, current layout generation methods focus solely on topological patterns without considering textual content, making them impractical for generating documents with complex associations between the contents and layouts. In this paper, we propose a Relation-rIch visual Document GEnerator (RIDGE) that addresses these limitations through a two-stage approach: (1) Content Generation, which leverages LLMs to generate document content using a carefully designed Hierarchical Structure Text format which captures entity categories and relationships, and (2) Content-driven Layout Generation, which learns to create diverse, plausible document layouts solely from easily available Optical Character Recognition (OCR) results, requiring no human labeling or annotations efforts. Experimental results have demonstrated that our method significantly enhances the performance of document understanding models on various VIE benchmarks.

</details>

---

## 345. GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery

- [ ] GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery | https://cvpr.thecvf.com/virtual/2025/poster/34519

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34519

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Given unlabelled datasets containing both old and new categories, generalized category discovery (GCD) aims to accurately discover new classes while correctly classifying old classes.Current GCD methods only use a single visual modality of information, resulting in poor classification of visually similar classes. As a different modality, text information can provide complementary discriminative information, which motivates us to introduce it into the GCD task.However, the lack of class names for unlabelled data makes it impractical to utilize text information.To tackle this challenging problem, in this paper, we propose a Text Embedding Synthesizer (TES) to generate pseudo text embeddings for unlabelled samples. Specifically, our TES leverages the property that CLIP can generate aligned vision-language features, converting visual embeddings into tokens of the CLIP’s text encoder to generate pseudo text embeddings. Besides, we employ a dual-branch framework, through the joint learning and instance consistency of different modality branches, visualand semantic information  mutually enhance each other,promoting the interaction and fusionof visual and text knowledge.Our method unlocks the multi-modal potentials of CLIP and outperforms the baseline methods by alarge margin on all GCD benchmarks, achieving new state-of-the-art.

</details>

---

## 346. Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation

- [ ] Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation | https://cvpr.thecvf.com/virtual/2025/poster/34522

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34522

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In robotic manipulation, task goals can be conveyed through various modalities, such as language, goal images, and goal videos. However, natural language can be ambiguous, while images or videos may offer overly detailed specifications. To address these challenges, we propose a novel approach using comprehensive multi-modal prompts that explicitly convey both low-level actions and high-level planning in a simple manner.Specifically, for each key-frame in the task sequence, our method allows for manual or automatic generation of simple and expressive 2D visual prompts overlaid on RGB images. These prompts represent the required task goals, such as the end-effector pose and the desired movement direction after contact. We develop a training strategy that enables the model to interpret these visual-language prompts and predict the corresponding contact poses and movement directions in SE(3) space.Furthermore, by sequentially executing all key-frame steps, the model can complete long-horizon tasks. This approach not only helps the model explicitly understand the task objectives but also enhances its robustness on unseen tasks by providing easily interpretable prompts.We evaluate our method in both simulated and real-world environments, demonstrating its robust manipulation capabilities.

</details>

---

## 347. SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models

- [ ] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34524

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34524

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of Vision Language Models (VLMs) has brought unprecedented advances in understanding multimodal information. The combination of textual and visual semantics in VLMs is highly complex and diverse, making the safety alignment of these models challenging. Furthermore, due to the limited study on the safety alignment of VLMs, there is a lack of large-scale, high-quality datasets. To address these limitations, we propose a Safety Preference Alignment dataset for Vision Language Models named SPA-VL. In terms of breadth, SPA-VL covers 6 harmfulness domains, 13 categories, and 53 subcategories, and contains 100,788 samples of the quadruple (question, image, chosen response, rejected response). In terms of depth, the responses are collected from 12 open-source (e.g., QwenVL) and closed-source (e.g., Gemini) VLMs to ensure diversity. The construction of preference data is fully automated, and the experimental results indicate that models trained with alignment techniques on the SPA-VL dataset exhibit substantial improvements in harmlessness and helpfulness while maintaining core capabilities. SPA-VL, as a large-scale, high-quality, and diverse dataset, represents a significant milestone in ensuring that VLMs achieve both harmlessness and helpfulness.

</details>

---

## 348. SmartCLIP: Modular Vision-language Alignment with Identification Guarantees

- [ ] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees | https://cvpr.thecvf.com/virtual/2025/poster/34525

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34525

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP)~\citep{radford2021learning} has emerged as a pivotal model in computer vision and multimodal learning, achieving state-of-the-art performance at aligning visual and textual representations through contrastive learning.However, CLIP struggles with potential information misalignment in many image-text datasets and suffers from entangled representation. On the one hand, short captions for a single image in datasets like MSCOCO may describe disjoint regions in the image, leaving the model uncertain about which visual features to retain or disregard.On the other hand, directly aligning long captions with images can lead to the retention of entangled details, preventing the model from learning disentangled, atomic concepts -- ultimately limiting its generalization on certain downstream tasks involving short prompts.In this paper, we establish theoretical conditions that enable flexible alignment between textual and visual representations across varying levels of granularity. Specifically, our framework ensures that a model can not only \emph{preserve} cross-modal semantic information in its entirety but also \emph{disentangle} visual representations to capture fine-grained textual concepts. Building on this foundation, we introduce \ours, a novel approach that identifies and aligns the most relevant visual and textual representations in a modular manner. Superior performance across various tasks demonstrates its capability to handle information misalignment and supports our identification theory.

</details>

---

## 349. VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models

- [ ] VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34531

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34531

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The recent surge in high-quality visual instruction tuning samples from closed-source vision-language models (VLMs) such as GPT-4V has accelerated the release of open-source VLMs across various model sizes. However, scaling VLMs to improve performance using larger models brings significant computational challenges, especially for deployment on resource-constrained devices like mobile platforms and robots. To address this, we propose VLsI: Verbalized Layers-to-Interactions, a new VLM family in 2B and 7B model sizes, which prioritizes efficiency without compromising accuracy. VLsI leverages a unique, layer-wise distillation process, introducing intermediate "verbalizers" that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. This approach mitigates the training instability often encountered in output imitation and goes beyond typical final-layer tuning by aligning the small VLMs’ layer-wise progression with that of the large ones. We validate VLsI across ten challenging vision-language benchmarks, achieving notable performance gains (11.0% for 2B and 17.4% for 7B) over GPT-4V without the need for model scaling, merging, or architectural changes.

</details>

---

## 350. HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation

- [ ] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation | https://cvpr.thecvf.com/virtual/2025/poster/34539

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34539

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal LLMs have advanced vision-language tasks but still struggle with understanding video scenes. To bridge this gap, Video Scene Graph Generation (VidSGG) has emerged to capture multi-object relationships across video frames. However, prior methods rely on pairwise connections, limiting their ability to handle complex multi-object interactions and reasoning. To this end, we propose Multimodal LLMs on a Scene HyperGraph (HyperGLM), promoting reasoning about multi-way interactions and higher-order relationships. Our approach uniquely integrates entity scene graphs, which capture spatial relationships between objects, with a procedural graph that models their causal transitions, forming a unified HyperGraph. Significantly, HyperGLM enables reasoning by injecting this unified HyperGraph into LLMs. Additionally, we introduce a new Video Scene Graph Reasoning (VSGR) dataset featuring 1.9M frames from third-person, egocentric, and drone views and supports five tasks: Scene Graph Generation, Scene Graph Anticipation, Video Question Answering, Video Captioning, and Relation Reasoning. Empirically, HyperGLM consistently outperforms state-of-the-art methods across five tasks, effectively modeling and reasoning complex relationships in diverse video scenes.

</details>

---

## 351. The Language of Motion: Unifying Verbal and Non-verbal Language of 3D Human Motion

- [ ] The Language of Motion: Unifying Verbal and Non-verbal Language of 3D Human Motion | https://cvpr.thecvf.com/virtual/2025/poster/34537

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34537

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human communication is inherently multimodal, involving a combination of verbal and non-verbal cues such as speech, facial expressions, and body gestures. Modeling these behaviors is essential for understanding human interaction and for creating virtual characters that can communicate naturally in applications like games, films, and virtual reality.  However, existing motion generation models are typically limited to specific input modalities—either speech, text, or motion data—and cannot fully leverage the diversity of available data. In this paper, we propose a novel framework that unifies verbal and non-verbal language using multimodal language models for human motion understanding and generation. This model is flexible in taking text, speech, and motion or any combination of them as input and output. Coupled with our novel pre-training strategy, our model not only achieves state-of-the-art performance on co-speech gesture generation but also requires much less data for training. Our unified model also unlocks an array of novel tasks such as editable gesture generation and emotion prediction from motion. We believe unifying the verbal and non-verbal language of human motion is essential for real-world applications, and language models offer a powerful approach to achieving this goal.

</details>

---

## 352. Explaining Domain Shifts in Language: Concept Erasing for Interpretable Image Classification

- [ ] Explaining Domain Shifts in Language: Concept Erasing for Interpretable Image Classification | https://cvpr.thecvf.com/virtual/2025/poster/34541

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34541

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Concept-based models are inherently interpretable, as they map black-box representations to human-understandable concepts, making the decision-making process more transparent. These models allow users to understand the reasoning behind predictions, which is crucial for high-stakes applications. However, they often introduce domain-specific concepts that contribute to the final predictions, which can undermine their generalization capabilities. In this paper, we propose a novel Language-guided Concept-Erasing (lanCE) framework. Specifically, we empirically demonstrate that pre-trained vision-language models (VLMs) can approximate distinct visual domain shifts via a large domain descriptor set. Based on these findings, we introduce a novel plug-in domain descriptor orthogonality (DDO) regularizer to mitigate the impact of these domain-specific concepts on the final predictions. To simulate a wide range of unseen visual domains, we generate a set of domain descriptors by prompting large language models (LLMs). Notably, our proposed DDO regularizer is agnostic to the design of concept-based models and thus can be widely integrated into various such models. By integrating the proposed DDO regularizer into several prevailing models and evaluating them on two standard domain generalization benchmarks and three new benchmarks introduced in this paper, we demonstrate that DDO loss can significantly improve the out-of-distribution (OOD) generalization capabilities over the previous state-of-the-art concept-based models. Codes are available in the supplementary material.

</details>

---

## 353. InsightEdit: Towards Better Instruction Following for Image Editing

- [ ] InsightEdit: Towards Better Instruction Following for Image Editing | https://cvpr.thecvf.com/virtual/2025/poster/34545

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34545

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we focus on the task of instruction-based image editing. Previous works like InstructPix2Pix, InstructDiffusion, and SmartEdit have explored end-to-end editing. However, two limitations still remain: First, existing datasets suffer from low resolution, poor background consistency, and overly simplistic instructions. Second, current approaches mainly condition on the text while the rich image information is underexplored, therefore inferior in complex instruction following and maintaining background consistency. Targeting these issues, we first curated the AdvancedEdit dataset using a novel data construction pipeline, formulating a large-scale dataset with high visual quality, complex instructions, and good background consistency. Then, to further inject the rich image information, we introduce a two-stream bridging mechanism utilizing both the textual and visual features reasoned by the powerful Multimodal Large Language Models (MLLM) to guide the image editing process more precisely. Extensive results demonstrate that our approach, InsightEdit, achieves state-of-the-art performance, excelling in complex instruction following and maintaining high background consistency with the original image.

</details>

---

## 354. Weakly Supervised Temporal Action Localization via Dual-Prior Collaborative Learning Guided by Multimodal Large Language Models

- [ ] Weakly Supervised Temporal Action Localization via Dual-Prior Collaborative Learning Guided by Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34555

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34555

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent breakthroughs in Multimodal Large Language Models (MLLMs) have gained significant recognition within the deep learning community, where the fusion of the Video Foundation Models (VFMs) and Large Language Models(LLMs) has proven instrumental in constructing robust video understanding systems, effectively surmounting constraints associated with predefined visual tasks. These sophisticated MLLMs exhibit remarkable proficiency in comprehending videos, swiftly attaining unprecedented performance levels across diverse benchmarks. However, their operation demands substantial memory and computational resources, underscoring the continued importance of traditional models in video comprehension tasks. In this paper, we introduce a novel learning paradigm termed MLLM4WTAL. This paradigm harnesses the potential of MLLM to offer temporal action key semantics and complete semantic textual cues for conventional Weakly-supervised Temporal Action Localization (WTAL) methods. MLLM4WTAL facilitates the enhancement of WTAL by leveraging MLLM guidance. It achieves this by integrating two distinct modules: Key Semantic Matching (KSM) and Complete Semantic Reconstruction (CSR). These modules work in tandem to effectively address prevalent issues like incomplete and over-complete outcomes common in WTAL methods. Rigorous experiments are conducted to validate the efficacy of our proposed approach in augmenting the performance of various heterogeneous WTAL models.

</details>

---

## 355. Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection

- [ ] Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection | https://cvpr.thecvf.com/virtual/2025/poster/34558

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34558

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automatic detection and prevention of open-set failures are crucial in closed-loop robotic systems. Recent studies often struggle to simultaneously identify unexpected failures reactively after they occur and prevent foreseeable ones proactively. To this end, we propose Code-as-Monitor (CaM), a novel paradigm leveraging the vision-language model (VLM) for both open-set reactive and proactive failure detection. The core of our method is to formulate both tasks as a unified set of spatio-temporal constraint satisfaction problems and use VLM-generated code to evaluate them for real-time monitoring. To enhance the accuracy and efficiency of monitoring, we further introduce constraint elements that abstract constraint-related entities or their parts into compact geometric elements. This approach offers greater generality, simplifies tracking, and facilitates constraint-aware visual programming by leveraging these elements as visual prompts. Experiments show that CaM achieves a 28.7% higher success rate and reduces execution time by 31.8% under severe disturbances compared to baselines across three simulators and a real-world setting. Moreover, CaM can be integrated with open-loop control policies to form closed-loop systems, enabling long-horizon tasks in cluttered scenes with dynamic environments.

</details>

---

## 356. CoLLM: A Large Language Model for Composed Image Retrieval

- [ ] CoLLM: A Large Language Model for Composed Image Retrieval | https://cvpr.thecvf.com/virtual/2025/poster/34564

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34564

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Composed Image Retrieval (CIR) is a complex task that aims to retrieve images based on a multimodal query. Typical training data consists of triplets containing a reference image, a textual description of desired modifications, and the target image, which are expensive and time-consuming to acquire. The scarcity of CIR datasets has led to zero-shot approaches utilizing synthetic triplets or leveraging vision-language models (VLMs) with ubiquitous web-crawled image-caption pairs. However, these methods have significant limitations: synthetic triplets suffer from limited scale, lack of diversity, and unnatural modification text, while image-caption pairs hinder joint embedding learning of the multimodal query due to the absence of triplet data. Moreover, existing approaches struggle with complex and nuanced modification texts that demand sophisticated fusion and understanding of vision and language modalities. We present CoLLM, a one-stop framework that effectively addresses these limitations. Our approach generates triplets on-the-fly from image-caption pairs, enabling supervised training without manual annotation. We leverage Large Language Models (LLMs) to generate joint embeddings of reference images and modification texts, facilitating deeper multimodal fusion. Additionally, we introduce Multi-Text CIR (MTCIR), a large-scale dataset comprising 3.4M samples, and refine existing CIR benchmarks (CIRR and Fashion-IQ) to enhance evaluation reliability. Experimental results demonstrate that CoLLM achieves state-of-the-art performance across multiple CIR benchmarks and settings. MTCIR yields competitive results, with up to 15\% performance improvement. Our refined benchmarks provide more reliable evaluation metrics for CIR models, contributing to the advancement of this important field.

</details>

---

## 357. PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models

- [ ] PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34568

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34568

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) respond to user-crafted text prompts and visual inputs, and are applied to numerous real-world problems.VLMs integrate visual modalities with large language models (LLMs), which are well known to be prompt-sensitive.Hence, it is crucial determining whether VLMs inherit this instability to varying prompts.We therefore investigate which prompt variations VLMs are most sensitive to and which VLMs are most agnostic to prompt variations.To this end, we introduce PARC (Prompt Analysis via Reliability and Calibration), a VLM prompt sensitivity analysis framework built on three pillars: (1) plausible prompt variations in both the language and vision domain, (2) a novel model reliability score with built-in guarantees, and (3) a calibration step that enables dataset- and prompt-spanning prompt variation analysis.Regarding prompt variations, experimental results from PARC show that VLMs mirror LLM language prompt sensitivity in the vision domain, and most destructive variations are those that change the expected answer. Regarding models, outstandingly robust VLMs among 22 evaluated models come from the InternVL2 family.We further find indications that prompt sensitivity is linked more closely to training data than to model size.Code and datasets will be released.

</details>

---

## 358. SEEN-DA: SEmantic ENtropy guided Domain-aware Attention for Domain Adaptive Object Detection

- [ ] SEEN-DA: SEmantic ENtropy guided Domain-aware Attention for Domain Adaptive Object Detection | https://cvpr.thecvf.com/virtual/2025/poster/34570

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34570

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Domain adaptive object detection (DAOD) aims to generalize detectors trained on an annotated source domain to an unlabelled target domain. Traditional works focus on aligning visual features between domains to extract domain-invariant knowledge, and recent VLM-based DAOD methods leverage semantic information provided by the textual encoder to supplement domain-specific features for each domain.However, they overlook the role of semantic information in guiding the learning of visual features that are beneficial for adaptation.To solve the problem, we propose semantic entropy to quantify the semantic information contained in visual features, and design SEmantic ENtropy guided Domain-aware Attention (SEEN-DA) to adaptively refine visual features with the semantic information of two domains.Semantic entropy reflects the importance of features based on semantic information, which can serve as attention to select discriminative visual features and suppress semantically irrelevant redundant information.Guided by semantic entropy, we introduce domain-aware attention modules into the visual encoder in SEEN-DA.It utilizes an inter-domain attention branch to extract domain-invariant features and eliminate redundant information, and an intra-domain attention branch to supplement the domain-specific semantic information discriminative on each domain.Comprehensive experiments validate the effectiveness of SEEN-DA, demonstrating significant improvements in cross-domain object detection performance.

</details>

---

## 359. Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding

- [ ] Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34575

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34575

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid growth of social media and digital photography, visually appealing images have become essential for effective communication and emotional engagement. Among the factors influencing aesthetic appeal, composition—the arrangement of visual elements within a frame—plays a crucial role. In recent years, specialized models for photographic composition have achieved impressive results across various aesthetic tasks. Meanwhile, rapidly advancing multimodal large language models (MLLMs) have excelled in several visual perception tasks. However, their ability to embed and understand compositional information remains underexplored, primarily due to the lack of suitable evaluation datasets. To address this gap, we introduce the Photographic Image Composition Dataset (PICD), a large-scale dataset consisting of 36,857 images categorized into 24 composition categories across 355 diverse scenes. We demonstrate the advantages of PICD over existing datasets in terms of data scale, composition category, label quality, and scene diversity. Building on PICD, we establish benchmarks to evaluate the composition embedding capabilities of specialized models and the compositional understanding ability of MLLMs. To enable efficient and effective evaluation, we propose a novel Composition Discrimination Accuracy (CDA) metric. Our evaluation highlights the limitations of current models and provides insights into directions for improving their ability to embed and understand composition.

</details>

---

## 360. Towards Natural Language-Based Document Image Retrieval: New Dataset and Benchmark

- [ ] Towards Natural Language-Based Document Image Retrieval: New Dataset and Benchmark | https://cvpr.thecvf.com/virtual/2025/poster/34571

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34571

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Document image retrieval (DIR) aims to retrieve document images from a gallery according to a given query. Existing DIR methods are primarily based on image queries that retrieves documents within the same coarse semantic category, e.g., newspapers or receipts. However, these methods struggle to effectively retrieve document images in real-world scenarios when using fine-grained semantics from text queries. To bridge this gap, this paper introduces a new benchmark of Natural Language-based Document Image Retrieval (NL-DIR) along with corresponding evaluation metrics. In this work, natural language descriptions serve as semantically rich queries for the DIR task. The NL-DIR dataset contains 41K authentic document images, each paired with five high-quality, fine-grained semantic queries generated and evaluated through large language models in conjunction with manual verification. We propose a two-stage retrieval method for DIR that enhances retrieval performance while optimizing both time and space efficiency. Furthermore, we perform zero-shot and fine-tuning evaluations of existing contrastive vision-language models and OCR-free visual document understanding (VDU) models on this dataset. The datasets and codes will be publicly available to facilitate research in the VDU community.

</details>

---

## 361. Is `Right' Right? Enhancing Object Orientation Understanding in Multimodal Large Language Models through Egocentric Instruction Tuning

- [ ] Is `Right' Right? Enhancing Object Orientation Understanding in Multimodal Large Language Models through Egocentric Instruction Tuning | https://cvpr.thecvf.com/virtual/2025/poster/34572

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34572

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) act as essential interfaces, connecting humans with AI technologies in multimodal applications. However, current MLLMs face challenges in accurately interpreting object orientation in images due to inconsistent orientation annotations in training data, hindering the development of a coherent orientation understanding. To overcome this, we propose egocentric instruction tuning, which aligns MLLMs' orientation understanding with the user’s perspective, based on a consistent annotation standard derived from the user’s egocentric viewpoint. We first generate egocentric instruction data that leverages MLLMs' ability to recognize object details and applies prior knowledge for orientation understanding. Using this data, we perform instruction tuning to enhance the model’s capability for accurate orientation interpretation. In addition, we introduce EgoOrientBench, a benchmark that evaluates MLLMs' orientation understanding across three tasks using images collected from diverse domains. Experimental results on this benchmark show that egocentric instruction tuning significantly improves orientation understanding without compromising overall MLLM performance. The instruction data and benchmark dataset are available on our project page at \url{https://anonymous.4open.science/r/EgocentricInstructionTuning-E189}.

</details>

---

## 362. Beyond Sight: Towards Cognitive Alignment in LVLM via Enriched Visual Knowledge

- [ ] Beyond Sight: Towards Cognitive Alignment in LVLM via Enriched Visual Knowledge | https://cvpr.thecvf.com/virtual/2025/poster/34599

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34599

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Does seeing always mean knowing? Large Vision-Language Models (LVLMs) integrate separately pre-trained vision and language components, often using CLIP-ViT as vision backbone.  However, these models frequently encounter a core issue of ``cognitive misalignment" between the vision encoder (VE) and the large language model (LLM). Specifically, the VE's representation of visual information may not fully align with LLM's cognitive framework, leading to a mismatch where visual features exceed the language model’s interpretive range.To address this, we investigate how variations in VE representations influence LVLM comprehension, especially when the LLM faces VE-Unknown data—images whose ambiguous visual representations challenge the VE’s interpretive precision. Accordingly, we construct a multi-granularity landmark dataset and systematically examine the impact of VE-Known and VE-Unknown data on interpretive abilities. Our results show that VE-Unknown data limits LVLM’s capacity for accurate understanding, while VE-Known data, rich in distinctive features, helps reduce cognitive misalignment.Building on these insights, we propose Entity-Enhanced Cognitive Alignment (EECA), a method that employs multi-granularity supervision to generate visually enriched, well-aligned tokens that not only integrate within the embedding space but also align with the LLM's cognitive framework. This alignment markedly enhances LVLM performance in landmark recognition. Our findings underscore the challenges posed by VE-Unknown data and highlight the essential role of cognitive alignment in advancing multimodal systems.

</details>

---

## 363. Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks

- [ ] Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks | https://cvpr.thecvf.com/virtual/2025/poster/34606

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34606

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present Omni-RGPT, a multimodal large language model designed to facilitate region-level comprehension for both images and videos. To achieve consistent region representation across spatio-temporal dimensions, we introduce Token Mark, a set of discretized tokens highlighting the target regions within the visual feature space. These tokens are directly embedded into spatial regions using region prompts (e.g., boxes or masks) and simultaneously incorporated into the text prompt to specify the target, establishing a direct connection between visual and language tokens. To further support robust video understanding without requiring tracklets, we introduce an auxiliary task that guides Token Mark by leveraging the consistency of the tokens, enabling stable region interpretation across video sequences.Additionally, we introduce RegVID-300k, a large-scale region-level video instruction dataset curated from diverse public video sources. Omni-RGPT achieves state-of-the-art results on visual commonsense reasoning benchmarks in image-based (VCR) and video-based (Causal-VidQA) tasks while also demonstrating strong performance in captioning and Referring Expression Comprehension (REC) tasks.

</details>

---

## 364. SocialGesture: Delving into Multi-person Gesture Understanding

- [ ] SocialGesture: Delving into Multi-person Gesture Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34623

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34623

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Previous research in human gesture recognition has largely overlooked multi-person interactions, which are crucial for understanding the social context of naturally occurring gestures. This limitation in existing datasets presents a significant challenge in aligning human gestures with other modalities like language and speech. To address this issue, we introduce SocialGesture, the first large-scale dataset specifically designed for multi-person gesture analysis. SocialGesture features a diverse range of natural scenarios and supports multiple gesture analysis tasks, including video-based recognition and temporal localization, providing a valuable resource for advancing the study of gesture during complex social interactions. Furthermore, we propose a novel visual question answering (VQA) task to benchmark vision language models' (VLMs) performance on social gesture understanding. Our findings highlight several limitations of current gesture recognition models, offering insights into future directions for improvement in this field.

</details>

---

## 365. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments

- [ ] SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments | https://cvpr.thecvf.com/virtual/2025/poster/34629

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34629

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent.Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration.The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image.This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications.Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.

</details>

---

## 366. Continual SFT Matches Multimodal RLHF with Negative Supervision

- [ ] Continual SFT Matches Multimodal RLHF with Negative Supervision | https://cvpr.thecvf.com/virtual/2025/poster/34637

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34637

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal RLHF usually happens after supervised finetuning (SFT) stage to continually improve vision-language models' (VLMs) comprehension. Conventional wisdom holds its superiority over continual SFT during this preference alignment stage. In this paper, we observe that the inherent value of multimodal RLHF lies in its negative supervision, the logit of the rejected responses. We thus propose a novel negative supervised finetuning (nSFT) approach that fully excavates these information resided. Our nSFT disentangles this negative supervision in RLHF paradigm, and continually aligns VLMs with a simple SFT loss. This is more memory efficient than multimodal RLHF where 2 (e.g., DPO) or 4 (e.g., PPO) large VLMs are strictly required. The effectiveness of nSFT is rigorously proved by comparing it with various multimodal RLHF approaches, across different dataset sources, base VLMs and evaluation metrics. Besides, fruitful of ablations are provided to support our hypothesis. We hope this paper will stimulate further research to properly align large vision language models.

</details>

---

## 367. SCAP: Transductive Test-Time Adaptation via Supportive Clique-based Attribute Prompting

- [ ] SCAP: Transductive Test-Time Adaptation via Supportive Clique-based Attribute Prompting | https://cvpr.thecvf.com/virtual/2025/poster/34641

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34641

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) exhibit promising generalization capabilities, yet face considerable challenges when adapting to domain shifts stemming from changes in data distributions.  Test-time adaptation (TTA) has thus emerged as a promising approach for enhancing VLM performance under such conditions. In practice, test data often arrives in batches, which has led to increasing interest in the transductive TTA setting. Existing TTA methods, however, are typically limited by focusing solely on individual test samples, thereby overlooking the critical cross-sample correlations within a batch. While recent ViT-based TTA methods have started to incorporate batch-level adaptation, they remain suboptimal for VLMs due to insufficient integration of the essential text modality. To bridge key gaps in TTA for VLMs, we propose a novel transductive TTA framework called Supportive Clique-based Attribute Prompting (SCAP), which effectively combines visual and textual information to enhance adaptation by generating fine-grained attribute prompts across test batches. SCAP first unsupervisedly forms supportive cliques of test samples based on visual similarity and learns an attribute prompt for each clique, capturing shared attributes critical for adaptation. For each test sample, SCAP aggregates attribute prompts from its associated cliques, providing enriched contextual information. To ensure adaptability over time, we incorporate a retention module that dynamically updates attribute prompts and their associated attributes as new data arrives. Comprehensive experiments across multiple benchmarks demonstrate that SCAP outperforms existing state-of-the-art methods, significantly advancing VLM generalization under domain shifts. The code will be released.

</details>

---

## 368. It’s a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data

- [ ] It’s a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data | https://cvpr.thecvf.com/virtual/2025/poster/34642

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34642

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The platonic representation hypothesis suggests that vision and language embeddings become more homogeneous as model and dataset sizes increase. In particular, pairwise distances within each modality become more similar. This suggests that as foundation models mature, it may become possible to match vision and language embeddings in a fully unsupervised fashion, i.e., without parallel data. We present the first study towards this prospect, and investigate conformity of existing vision and language foundation models in the context of "blind" matching. First, we formulate unsupervised matching as a quadratic assignment problem and introduce a novel heuristic that outperforms previous solvers. We also develop a technique to find optimal matching problems, for which a non-trivial match is very likely. Second, we conduct an extensive study deploying a range of vision and language models on four datasets. Our analysis reveals that for many problem instances, vision and language representations can be indeed matched without supervision. This finding opens possibility for exciting applications embedding semantic knowledge into other modalities. As a showcase, we demonstrate a proof-of-concept unsupervised classifier, which achieves non-trivial classification accuracy without any image-text annotation.

</details>

---

## 369. STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection

- [ ] STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection | https://cvpr.thecvf.com/virtual/2025/poster/34647

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34647

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Advancements in Computer-Aided Screening (CAS) systems are essential for improving the detection of security threats in X-ray baggage scans. However, current datasets are limited in representing real-world, sophisticated threats and concealment tactics, and existing approaches are constrained by a closed-set paradigm with predefined labels. To address these challenges, we introduce STCray, the first multimodal X-ray baggage security dataset, comprising 46,642 image-caption paired scans across 21 threat categories, generated using an X-ray scanner for airport security. STCray is meticulously developed with our specialized protocol that ensures domain-aware, coherent captions, that lead to the multi-modal instruction following data in X-ray baggage security. This allows us to train a domain-aware visual AI assistant named STING-BEE that supports a range of vision-language tasks, including scene comprehension, referring threat localization, visual grounding, and visual question answering (VQA), establishing novel baselines for multi-modal learning in X-ray baggage security. Further, STING-BEE shows state-of-the-art generalization in cross-domain settings. Our code, data, and pre-trained models will be made publicly available.

</details>

---

## 370. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models

- [ ] Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34656

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34656

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99\% attack success rate, causing a 77.6\% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.

</details>

---

## 371. Adaptive Parameter Selection for Tuning Vision-Language Models

- [ ] Adaptive Parameter Selection for Tuning Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34663

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34663

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) like CLIP have been widely used in various specific tasks.Parameter-efficient fine-tuning (PEFT) methods, such as prompt and adapter tuning,have become key techniques for adapting these models to specific domains.However, existing approaches rely on prior knowledgeto manually identify the locations requiring fine-tuning.Adaptively selecting which parameters in VLMs should be tuned remains unexplored. In this paper, we propose CLIP with Adaptive Selective Tuning (CLIP-AST), which can be used to automatically select critical parameters in VLMs for fine-tuning for specific tasks.It opportunely leveragesthe adaptive learning rate in the optimizer and improves model performance without extra parameter overhead. We conduct extensive experiments on 13 benchmarks, such as ImageNet, Food101, Flowers102, etc,with different settings, including few-shot learning, base-to-novel class generalization, and out-of-distribution. The results show that CLIP-AST consistently outperforms the original CLIP model as well as its variantsand achieves state-of-the-art (SOTA) performance in all cases. For example, with the 16-shot learning, CLIP-AST surpasses GraphAdapter and PromptSRC by 3.56\% and 2.20\% in average accuracy on 11 datasets, respectively.Code will be publicly available.

</details>

---

## 372. IterIS: Iterative Inference-Solving Alignment for LoRA Merging

- [ ] IterIS: Iterative Inference-Solving Alignment for LoRA Merging | https://cvpr.thecvf.com/virtual/2025/poster/34667

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34667

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Low-rank adaptations (LoRA) are widely used to fine-tune large models across various domains for specific downstream tasks. While task-specific LoRAs are often available, concerns about data privacy and intellectual property can restrict access to training data, limiting the acquisition of a multi-task model through gradient-based training. In response, LoRA merging presents an effective solution by combining multiple LoRAs into a unified adapter while maintaining data privacy. Prior works on LoRA merging primarily frame it as an optimization problem, yet these approaches face several limitations, including the rough assumption about input features utilized in optimization, massive sample requirements, and the unbalanced optimization objective. These limitations can significantly degrade performance. To address these, we propose a novel optimization-based method, named IterIS: 1) We formulate LoRA merging as an advanced optimization problem to mitigate the rough assumption. Additionally, we employ an iterative inference-solving framework in our algorithm. It can progressively refine the optimization objective for improved performance. 2) We introduce an efficient regularization term to reduce the need for massive sample requirements (requiring only 1-5\% of the unlabeled samples compared to prior methods). 3) We utilize adaptive weights in the optimization objective to mitigate potential unbalances in LoRA merging process. Our method demonstrates significant improvements over multiple baselines and state-of-the-art methods in composing tasks for text-to-image diffusion, vision-language models, and large language models. Furthermore, our layer-wise algorithm can achieve convergence with minimal steps, ensuring efficiency in both memory and computation.

</details>

---

## 373. JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation

- [ ] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation | https://cvpr.thecvf.com/virtual/2025/poster/34669

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34669

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present JanusFlow , a powerful framework that unifies image understanding and generation in a single model.JanusFlow introduces a minimalist architecture that integrates autoregressive language models with rectified flow, a state-of-the-art method in generative modeling.Our key finding demonstrates that rectified flow can be straightforwardly trained within the large language model framework, eliminating the need for complex architectural modifications.To further improve the performance of our unified model, we adopt two key strategies: (i) decoupling the understanding and generation encoders, and (ii) aligning their representations during unified training.Extensive experiments show that JaunsFlow achieves comparable or superior performance to specialized models in their respective domains, while significantly outperforming existing unified approaches.This work represents a step toward more efficient and versatile vision-language models.

</details>

---

## 374. Just Dance with pi! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection

- [ ] Just Dance with pi! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection | https://cvpr.thecvf.com/virtual/2025/poster/34670

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34670

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Weakly-supervised methods for video anomaly detection (VAD) are conventionally based merely on RGB spatio-temporal features, which continues to limit their reliability in real-world scenarios. This is due to the fact that RGB-features are not sufficiently distinctive in setting apart categories such as shoplifting from visually similar events. Therefore, towards robust complex real-world VAD, it is essential to augment RGB spatio-temporal features by additional modalities. Motivated by this, we introduce the Poly-modal Induced framework for VAD: PI-VAD (or $\pi$-VAD), a novel approach that augments RGB representations by five additional modalities. Specifically, the modalities include sensitivity to fine-grained motion (Pose), three dimensional scene and entity representation (Depth), surrounding objects (Panoptic masks), global motion (optical flow), as well as language cues (VLM). Each modality represents an axis of a polygon, streamlined to add salient cues to RGB. $\pi$-VAD includes two plug-in modules, namely Pseudo-modality Generation module and Cross Modal Induction module, which generate modality-specific prototypical representation and, thereby, induce multi-modal information into RGB cues. These modules operate by performing anomaly-aware auxiliary tasks and necessitate five modality backbones -- only during training. Notably, $\pi$-VAD achieves state-of-the-art accuracy on three prominent VAD datasets encompassing real-world scenarios, without requiring the computational overhead of five modality backbones at inference.

</details>

---

## 375. SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters

- [ ] SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters | https://cvpr.thecvf.com/virtual/2025/poster/34683

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34683

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human beings are social animals. How to equip 3D autonomous characters with similar social intelligence that can perceive, understand and interact with humans remains an open yet foundamental problem. In this paper, we introduce SOLAMI, the first end-to-end Social vision-Language-Action (VLA) Modeling framework for Immersive interaction with 3D autonomous characters. Specifically, SOLAMI builds 3D autonomous characters from three aspects: 1) Social VLA Architecture: We propose a unified social VLA framework to generate multimodal response (speech and motion) based on the user's multimodal input to drive the character for social interaction. 2) Interactive Multimodal Data: We present SynMSI, a synthetic  multimodal social interaction dataset generated by an automatic pipeline using only existing motion datasets to address the issue of data scarcity. 3) Immersive VR Interface: We develop a VR interface that enables users to immersively interact with these characters driven by various architectures. Extensive quantitative experiments and user studies demonstrate that our framework leads to more precise and natural character responses (in both speech and motion) that align with user expectations with lower latency.

</details>

---

## 376. Recover and Match: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport

- [ ] Recover and Match: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport | https://cvpr.thecvf.com/virtual/2025/poster/34686

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34686

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Identifying multiple novel classes in an image, known as open-vocabulary multi-label recognition, is a challenging task in computer vision. Recent studies explore the transfer of powerful vision-language models such as CLIP. However, these approaches face two critical challenges: (1) The local semantics of CLIP are disrupted due to its global pre-training objectives, resulting in unreliable regional predictions. (2) The matching property between image regions and candidate labels has been neglected, relying instead on naive feature aggregation such as average pooling, which leads to spurious predictions from irrelevant regions. In this paper, we present RAM (Recover And Match), a novel framework that effectively addresses the above issues. To tackle the first problem, we propose Ladder Local Adapter (LLA) to enforce the model to refocus on its surrounding local regions, recovering local semantics in a memory-friendly way. For the second issue, we propose Knowledge-Constrained Optimal Transport (KCOT) to suppress meaningless matching to non-GT labels by formulating the task as an optimal transport problem. As a result, RAM achieves state-of-the-art performance on NUS-WIDE, MS-COCO, RAPv1, PA100K, MultiScene and MLRSNet datasets from three distinct domains, and shows great potential to boost the existing methods. Code will be made available.

</details>

---

## 377. Identifying and Mitigating Position Bias of Multi-image Vision-Language Models

- [ ] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34691

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34691

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The evolution of Large Vision-Language Models (LVLMs) has progressed from single-image understanding to multi-image reasoning. Despite this advancement, our findings indicate that LVLMs struggle to robustly utilize information across multiple images, with predictions significantly affected by the alteration of image positions. To further explore this issue, we introduce Position-wise Question Answering (PQA), a meticulously designed task to quantify reasoning capabilities at each position. Our analysis reveals a pronounced position bias in LVLMs: open-source models excel in reasoning with images positioned later but underperform with those in the middle or at the beginning, while proprietary models like GPT-4o show improved comprehension for images at the beginning and end but struggle with those in the middle. Motivated by these insights, we propose SoFt Attention (SoFA), a simple, training-free approach that mitigates this bias by employing linear interpolation between inter-image causal attention and bidirectional counterparts. Experimental results demonstrate that SoFA effectively reduces position bias and significantly enhances the reasoning performance of existing LVLMs.

</details>

---

## 378. OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning

- [ ] OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning | https://cvpr.thecvf.com/virtual/2025/poster/34693

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34693

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advances in vision-language models (VLMs) have led to a growing interest in autonomous driving to leverage their strong reasoning capabilities. However, extending these capabilities from 2D to full 3D understanding is crucial for real-world applications. To address this challenge, we propose OmniDrive, a holistic vision-language dataset that aligns agent models with 3D driving tasks through counterfactual reasoning. This approach enhances decision-making by evaluating potential scenarios and their outcomes, similar to human drivers considering alternative actions. Our counterfactual-based synthetic data annotation process generates large-scale, high-quality datasets, providing denser supervision signals that bridge planning trajectories and language-based reasoning. Futher, we explore two advanced OmniDrive-Agent frameworks, namely Omni-L and Omni-Q, to assess the importance of vision-language alignment versus 3D perception, revealing critical insights into designing effective LLM-agents. Significant improvements on the DriveLM Q\&A benchmark and nuScenes open-loop planning demonstrate the effectiveness of our dataset and methods.

</details>

---

## 379. Video Language Model Pretraining with Spatio-temporal Masking

- [ ] Video Language Model Pretraining with Spatio-temporal Masking | https://cvpr.thecvf.com/virtual/2025/poster/34702

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34702

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of video-language self-supervised models based on mask learning has significantly advanced downstream video tasks. These models leverage masked reconstruction to facilitate joint learning of visual and linguistic information. However, a recent study reveals that reconstructing image features yields superior downstream performance compared to video feature reconstruction. We hypothesize that this performance gap stems from how masking strategies influence the model's attention to temporal dynamics.To validate this hypothesis, we conduct two sets of experiments demonstrating that alignment between masked object and reconstruction target is crucial for effective video-language self-supervised learning. Based on these findings, we propose a spatio-temporal masking strategy (STM) for video-language model pretraining that operates across adjacent frames, and a decoder leverages semantic information to enhance the spatio-temporal representations of masked tokens. Through the combination of masking strategy and reconstruction decoder, STM enforces the model to learn the spatio-temporal feature representation more comprehensively. Experimental results across three video understanding downstream tasks validate the superiority of our method.

</details>

---

## 380. ILIAS: Instance-Level Image retrieval At Scale

- [ ] ILIAS: Instance-Level Image retrieval At Scale | https://cvpr.thecvf.com/virtual/2025/poster/34712

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34712

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This work introduces ILIAS, a new test dataset for Instance-Level Image retrieval At Scale. It is designed to evaluate the ability of current and future foundation models and retrieval techniques to recognize particular objects. The key benefits over existing datasets include large scale, domain diversity, accurate ground truth, and a performance that is far from saturated. ILIAS includes query and positive images for 1,000 object instances, manually collected to capture challenging conditions and diverse domains. Large-scale retrieval is conducted against 100 million distractor images from YFCC100M. To avoid false negatives without extra annotation effort, we include only query objects confirmed to have emerged after 2014, i.e. the compilation date of YFCC100M. An extensive benchmarking is performed with the following observations: i) models fine-tuned on specific domains, such as landmarks or products, excel in that domain but fail on ILIAS, ii) learning a linear adaptation layer using multi-domain class supervision results in  performance improvements, especially for vision-and-language models, iii) local descriptors in retrieval re-ranking are still a key ingredient, especially in the presence of severe background clutter, iv) the text-to-image performance of the vision-language foundation models is surprisingly close to the corresponding image-to-image case.

</details>

---

## 381. Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces

- [ ] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces | https://cvpr.thecvf.com/virtual/2025/poster/34720

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34720

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce the task of predicting functional 3D scene graphs for real-world indoor environments from posed RGB-D images. Unlike traditional 3D scene graphs that focus on spatial relationships of objects, functional 3D scene graphs capture objects, interactive elements, and their functional relationships. Due to the lack of training data, we leverage foundation models, including visual language models (VLMs) and large language models (LLMs), to encode functional knowledge. We evaluate our approach on an extended SceneFun3D dataset and a newly collected dataset, FunGraph3D, both annotated with functional 3D scene graphs. Our method significantly outperforms adapted baselines, including Open3DSG and ConceptGraph, demonstrating its effectiveness in modeling complex scene functionalities. We also demonstrate downstream applications such as 3D question answering and robotic manipulation using functional 3D scene graphs.

</details>

---

## 382. UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning

- [ ] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning | https://cvpr.thecvf.com/virtual/2025/poster/34715

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34715

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Transductive few-shot learning has recently triggered wide attention in computer vision. Yet, current methods introduce key hyper-parameters, which control the pre-diction statistics of the test batches, such as the level of class balance, affecting performances significantly. Such hyper-parameters are empirically grid-searched over validation data, and their configurations may vary substantially with the target dataset and pre-training model, making such empirical searches both sub-optimal and computationally intractable. In this work, we advocate and introduce the unrolling paradigm, also referred to as “learning to optimize”, in the context of few-shot learning, thereby learning efficiently and effectively a set of optimized hyperparameters. Specifically, we unroll a generalization of the ubiquitous Expectation-Maximization (EM) optimizer into a neural network architecture, mapping each of its iterates to a layer and learning a set of key hyper-parameters over validation data. Our unrolling approach covers various statistical feature distributions and pre-training paradigms,  including recent foundational vision-language models and standard vision-only classifiers. We report comprehensive experiments, which cover a breadth of fine-grained downstream image classification tasks, showing significant gains brought by the proposed unrolled EM algorithm over iterative variants. The achieved improvements reach up to 10% and 7.5% on vision-only and vision-language benchmarks, respectively. The source code and learned parameters are available at https://anonymous.4open.science/r/UNEM .

</details>

---

## 383. T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting

- [ ] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting | https://cvpr.thecvf.com/virtual/2025/poster/34719

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34719

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot object counting aims to count instances of arbitrary object categories specified by text descriptions. Existing methods typically rely on vision-language models like CLIP, but often exhibit limited sensitivity to text prompts. We present T2ICount, a one-step diffusion-based framework that leverages rich prior knowledge and fine-grained visual understanding from pretrained diffusion models. While one-step denoising ensures efficiency, it leads to weakened text sensitivity. To address this challenge, we propose a Hierarchical Semantic Correction Module that progressively refines text-image feature alignment, and a Representational Regional Coherence Loss that provides reliable supervision signals by leveraging the cross-attention maps extracted from the denosing U-Net. Furthermore, we observe that current benchmarks mainly focus on majority objects in images, potentially masking models' text sensitivity. To address this, we contribute a challenging re-annotated subset of FSC147 for better evaluation of text-guided counting ability. Extensive experiments demonstrate that our method achieves superior performance across different benchmarks. Code will be made publicly available.

</details>

---

## 384. FaceBench: A Multi-View Multi-Level Facial Attribute VQA Dataset for Benchmarking Face Perception MLLMs

- [ ] FaceBench: A Multi-View Multi-Level Facial Attribute VQA Dataset for Benchmarking Face Perception MLLMs | https://cvpr.thecvf.com/virtual/2025/poster/34717

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34717

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in various tasks. However, effectively evaluating these MLLMs on face perception remains largely unexplored. To address this gap, we introduce FaceBench, a dataset featuring hierarchical multi-view and multi-level attributes specifically designed to assess the comprehensive face perception abilities of MLLMs. Initially, we construct a hierarchical facial attribute structure, which encompasses five views with up to three levels of attributes, totaling over 210 attributes and 700 attribute values. Based on the structure, the proposed FaceBench consists of 49,919 visual question-answering (VQA) pairs for evaluation and 23,841 pairs for fine-tuning. Moreover, we further develop a robust face perception MLLM baseline, Face-LLaVA, by multi-modal training with our proposed face instruction-tuning data. Extensive experiments on various mainstream MLLMs and Face-LLaVA are conducted to test their face perception ability, which are also comapred with human. The results reveal that, the existing MLLMs are far from satisfactory in understanding the fine-grained facial attributes, while our Face-LLaVA significantly outperforms existing open-source models with a small amount of training data and is comparable to commercial ones like GPT-4o and Gemini. The dataset will be released upon acceptance of this work.

</details>

---

## 385. Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy

- [ ] Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy | https://cvpr.thecvf.com/virtual/2025/poster/34727

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34727

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the remarkable versatility of Large Language Models (LLMs) and Multimodal LLMs (MLLMs) to generalize across both language and vision tasks, LLMs and MLLMs have shown vulnerability to jailbreaking, generating textual outputs that undermine safety, ethical, and bias standards when exposed to harmful or sensitive inputs. With the recent advancement of safety-alignment via preference-tuning from human feedback, LLMs and MLLMs have been equipped with safety guardrails to yield safe, ethical, and fair responses with regard to harmful inputs. However, despite the significance of safety-alignment, research on the vulnerabilities remains largely underexplored. In this paper, we investigate the unexplored vulnerability of the safety-alignment, examining its ability to consistently provide safety guarantees for out-of-distribution(OOD)-ifying harmful inputs that may fall outside the aligned data distribution. Our key observation is that OOD-ifying the vanilla harmful inputs highly increases the uncertainty of the model to discern the malicious intent within the input, leading to a higher chance of being jailbroken. Exploiting this vulnerability, we propose JOOD, a new Jailbreak framework via OOD-ifying inputs beyond the safety-alignment. We explore various off-the-shelf visual and textual transformation techniques for OOD-ifying the harmful inputs. Notably, we observe that even simple mixing-based techniques such as image mixup prove highly effective in increasing the uncertainty of the model, thereby facilitating the bypass of the safety-alignment. Experimental results across diverse jailbreak scenarios demonstrate that JOOD effectively jailbreaks recent proprietary LLMs and MLLMs such as GPT-4 and GPT-4V with high attack success rate, which previous attack approaches have consistently struggled to jailbreak.

</details>

---

## 386. COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation

- [ ] COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation | https://cvpr.thecvf.com/virtual/2025/poster/34730

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34730

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) face significant challenges in test-time adaptation to novel domains. While cache-based methods show promise by leveraging historical information, they struggle with both caching unreliable feature-label pairs and indiscriminately using single-class information during querying, significantly compromising adaptation accuracy. To address these limitations, we propose \textbf{COSMIC} (\underline{C}lique-\underline{O}riented \underline{S}emantic \underline{M}ulti-space \underline{I}ntegration for \underline{C}LIP), a robust test-time adaptation framework that enhances adaptability through multi-granular, cross-modal semantic caching and graph-based querying mechanisms. Our framework introduces two key innovations: \textit{Dual Semantics Graph} (DSG) and \textit{Clique Guided Hyper-class} (CGH). The Dual Semantics Graph constructs complementary semantic spaces by incorporating textual features, coarse-grained CLIP features, and fine-grained DINOv2 features to capture rich semantic relationships. Building upon these dual graphs, the Clique Guided Hyper-class component leverages structured class relationships to enhance prediction robustness through correlated class selection. Extensive experiments demonstrate COSMIC's superior performance across multiple benchmarks, achieving significant improvements over state-of-the-art methods: 15.81\% gain on out-of-distribution tasks and 5.33\% on cross-domain generation with CLIP RN-50.

</details>

---

## 387. ANNEXE: Unified Analyzing, Answering, and Pixel Grounding for Egocentric Interaction

- [ ] ANNEXE: Unified Analyzing, Answering, and Pixel Grounding for Egocentric Interaction | https://cvpr.thecvf.com/virtual/2025/poster/34752

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34752

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Egocentric interaction perception is one of the essential branches in investigating human-environment interaction, which lays the basis for developing next-generation intelligent systems. However, existing egocentric interaction understanding methods cannot yield coherent textual and pixel-level responses simultaneously according to user queries, which lacks flexibility for varying downstream application requirements. To comprehend egocentric interactions exhaustively, this paper presents a novel task named Egocentric Interaction Reasoning and pixel Grounding (Ego-IRG). Taking an egocentric image with the query as input, Ego-IRG is the first task that aims to resolve the interactions through three crucial steps: analyzing, answering, and pixel grounding, which results in fluent textual and fine-grained pixel-level responses.Another challenge is that existing datasets cannot meet the conditions for the Ego-IRG task. To address this limitation, this paper creates the Ego-IRGBench dataset based on extensive manual efforts, which includes over 20k egocentric images with 1.6 million queries and corresponding multimodal responses about interactions. Moreover, we design a unified ANNEXE model to generate text- and pixel-level outputs utilizing multimodal large language models, which enables a comprehensive interpretation of egocentric interactions.The experiments on the Ego-IRGBench exhibit the effectiveness of our ANNEXE model compared with other works.

</details>

---

## 388. DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception

- [ ] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception | https://cvpr.thecvf.com/virtual/2025/poster/34755

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34755

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Dense visual prediction tasks have been constrained by their reliance on predefined categories, limiting their applicability in real-world scenarios where visual concepts are unbounded. While Vision-Language Models (VLMs) like CLIP have shown promise in open-vocabulary tasks, their direct application to dense prediction often leads to suboptimal performance due to limitations in local feature representation. In this work, we present our observation that CLIP's image tokens struggle to effectively aggregate information from spatially or semantically related regions, resulting in features that lack local discriminability and spatial consistency. To address this issue, we propose DeCLIP, a novel framework that enhances CLIP by decoupling the self-attention module to obtain "content'' and "context'' features respectively. The "content'' features are aligned with image crop representations to improve local discriminability, while "context'' features learn to retain the spatial correlations under the guidance of vision foundation models, such as DINO. Extensive experiments demonstrate that DeCLIP significantly outperforms existing methods across multiple open-vocabulary dense prediction tasks, including object detection and semantic segmentation. Code and models will be made publicly available.

</details>

---

## 389. Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens

- [ ] Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens | https://cvpr.thecvf.com/virtual/2025/poster/34758

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34758

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent endeavors in Multimodal Large Language Models (MLLMs) aim to unify visual comprehension and generation by combining LLM and diffusion models, the state-of-the-art in each task, respectively. Existing approaches rely on spatial visual tokens, where image patches are encoded and arranged according to a spatial order (e.g., raster scan). However, we show that spatial tokens lack the recursive structure inherent to languages, hence form an impossible language for LLM to master. In this paper, we build a proper visual language by leveraging diffusion timesteps to learn discrete, recursive visual tokens. Our proposed tokens recursively compensate for the progressive attribute loss in noisy images as timesteps increase, enabling the diffusion model to reconstruct the original image at any timestep. This approach allows us to effectively integrate the strengths of LLMs in autoregressive reasoning and diffusion models in precise image generation, achieving seamless multimodal comprehension and generation within a unified framework.  Extensive experiments show that we achieve a new SOTA for multimodal comprehension and generation  simultaneously compared with other MLLMs.

</details>

---

## 390. Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding

- [ ] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding | https://cvpr.thecvf.com/virtual/2025/poster/34761

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34761

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have obtained impressive performance in visual content understanding and multi-modal reasoning. Unfortunately, these large models suffer from serious hallucination problems and tend to generate fabricated responses. Recently, several Contrastive Decoding (CD) strategies have been proposed to alleviate hallucination by introducing disturbed inputs. Although great progress has been made, these CD strategies mostly apply a one-size-fits-all approach for all input conditions. In this paper, we revisit this process through extensive experiments. Related results show that hallucination causes are hybrid and each generative step faces a unique hallucination challenge. Leveraging these meaningful insights, we introduce a simple yet effective Octopus-like framework that enables the model to adaptively identify hallucination types and create a dynamic CD workflow. Our Octopus framework not only outperforms existing methods across four benchmarks but also demonstrates excellent deployability and expansibility. Our code will be released.

</details>

---

## 391. Human-centered Interactive Learning via MLLMs for Text-to-Image Person Re-identification

- [ ] Human-centered Interactive Learning via MLLMs for Text-to-Image Person Re-identification | https://cvpr.thecvf.com/virtual/2025/poster/34768

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34768

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite remarkable advancements in text-to-image person re-identification (TIReID) facilitated by the breakthrough of cross-modal embedding models, existing methods often struggle to distinguish challenging candidate images due to intrinsic limitations, such as network architecture and data quality. To address these issues, we propose an Interactive Cross-modal Learning framework (ICL), which leverages human-centered interaction to enhance the discriminability of text queries through external multimodal knowledge. To achieve this, we propose a plug-and-play Test-time Humane-centered Interaction (TUI) module, which performs visual question answering focused on human characteristics, facilitating multi-round interactions with a multimodal large language model (MLLM) to align query intent with latent target images. Specifically, TUI refines user queries based on the MLLM responses to reduce the gap to the best-matching images, thereby boosting ranking accuracy. Additionally, to address the limitation of low-quality training texts, we introduce a novel Reorganization Data Augmentation (RDA) strategy based on information enrichment and diversity enhancement to enhance query discriminability by enriching, decomposing, and reorganizing person descriptions. Extensive experiments on four TIReID benchmarks, i.e., CUHK-PEDES, CFG-PEDES RSTPReid, RSTPReid, and UFine6926, demonstrate that our method achieves remarkable performance with substantial improvement. The code will be released publicly.

</details>

---

## 392. VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary

- [ ] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary | https://cvpr.thecvf.com/virtual/2025/poster/34773

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34773

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human daily activities can be concisely narrated as sequences of routine events (e.g., turning off an alarm) in video streams, forming an event vocabulary. Motivated by this, we introduce VLog , a novel video understanding framework that defines video narrations as a vocabulary, going beyond the typical subword vocabularies in existing generative video-language models. Built on the lightweight language model GPT-2, VLog features three key innovations:1. A Generative Retrieval Model Marrying the language model's complex reasoning capabilities with contrastive retrieval's efficient similarity search.2. A Hierarchical Vocabulary Derived from large-scale video narrations using our narration pair encoding algorithm, enabling efficient indexing of specific events (e.g., cutting a tomato) by identifying broader scenarios (e.g., kitchen) with expressive postfixes (e.g., by the left hand).3. A Vocabulary Update Strategy Leveraging generative models to extend the vocabulary for novel events encountered during inference.To validate our approach, we introduce VidCab-Eval , a development set requiring concise narrations with reasoning relationships (e.g., before and after). Experiments on EgoSchema , COIN , and HiREST further demonstrate the effectiveness of VLog , highlighting its ability to generate concise, contextually accurate, and efficient narrations. This offers a novel perspective on video understanding.

</details>

---

## 393. ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting

- [ ] ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting | https://cvpr.thecvf.com/virtual/2025/poster/34772

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34772

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have excelled in multimodal tasks, but adapting them to embodied decision-making in open-world environments presents challenges.  One critical issue is bridging the gap between discrete entities in low-level observations and the abstract concepts required for effective planning.  A common solution is building hierarchical agents, where VLMs serve as high-level reasoners that break down tasks into executable sub-tasks, typically specified using language.  However, language suffers from the inability to communicate detailed spatial information.  We propose visual-temporal context prompting, a novel communication protocol between VLMs and policy models. This protocol leverages object segmentation from past observations to guide policy-environment interactions. Using this approach, we train ROCKET-1, a low-level policy that predicts actions based on concatenated visual observations and segmentation masks, supported by real-time object tracking from SAM-2. Our method unlocks the potential of VLMs, enabling them to tackle complex tasks that demand spatial reasoning.  Experiments in Minecraft show that our approach enables agents to achieve previously unattainable tasks, with a $76$\% absolute improvement in open-world interaction performance.  Codes and demos will be released.

</details>

---

## 394. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

- [ ] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces | https://cvpr.thecvf.com/virtual/2025/poster/34778

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34778

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Humans possess the visual-spatial intelligence to remember spaces from sequential visual observations. However, can Multimodal Large Language Models (MLLMs) trained on million-scale video datasets also "think in space" from videos? We present a novel video-based visual-spatial intelligence benchmark (VSI-Bench) of over 5,000 question-answer pairs, and find that MLLMs exhibit competitive—though subhuman—visual-spatial intelligence. We probe models to express how they think in space both linguistically and visually and find that while spatial reasoning capabilities remain the primary bottleneck for MLLMs to reach higher benchmark performance, local world models and spatial awareness do emerge within these models. Notably, prevailing linguistic reasoning techniques (e.g., chain-of-thought, self-consistency, tree-of-thoughts) fail to improve performance, whereas  explicitly generating cognitive maps during question-answering enhances MLLMs' spatial distance awareness.

</details>

---

## 395. DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models

- [ ] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34775

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34775

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video large language models (VLLMs) have significantly advanced recently in processing complex video content, yet their inference efficiency remains constrained because of the high computational cost stemming from the thousands of visual tokens generated from the video inputs. We empirically observe that, unlike single image inputs, VLLMs typically attend visual tokens from different frames at different decoding iterations, making a one-shot pruning strategy prone to removing important tokens by mistake. Motivated by this, we present DyCoke, a training-free token compression method to optimize token representation and accelerate VLLMs. DyCoke incorporates a plug-and-play temporal compression module to minimize temporal redundancy by merging redundant tokens across frames, and applies dynamic KV cache reduction to prune spatially redundant tokens selectively. It ensures high-quality inference by dynamically retaining the critical tokens at each decoding step. Extensive experimental results demonstrate that DyCoke can outperform the prior SoTA counterparts, achieving 1.5$\times$ inference speedup, 1.4$\times$ memory reduction against the baseline VLLM, while still improving the performance, with no training.

</details>

---

## 396. Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models

- [ ] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34787

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34787

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-performance Multimodal Large Language Models (MLLMs) rely heavily on data quality. This study introduces a novel data synthesis method, leveraging insights from contrastive learning and image difference captioning to enhance fine-grained image recognition in MLLMs. By analyzing object differences in detailed regions between similar images, we challenge the model to identify both matching and distinct components. Specifically, our method initially create pairs of similar images that highlight object variations. After that, we introduce a Difference Area Generator for object differences identifying, followed by a Difference Captions Generator for differences describing. The outcome is a high-quality dataset of "object replacement" samples, named Img-Diff, which can be expanded as needed due to its automation. We use the generated dataset to finetune state-of-the-art (SOTA) MLLMs such as InternVL2, yielding comprehensive improvements across numerous image difference and Visual Question Answering tasks. For instance, the trained models notably surpass the SOTA models GPT-4V and Gemini on the MMVP benchmark. Additionally, we conduct thorough evaluations to confirm the dataset's diversity, quality, and robustness, presenting several insights on the synthesis of such a contrastive dataset. We release our codes and dataset to encourage further research on multimodal data synthesis and MLLMs' fundamental capabilities for image understanding.

</details>

---

## 397. Adapting Text-to-Image Generation with Feature Difference Instruction for Generic Image Restoration

- [ ] Adapting Text-to-Image Generation with Feature Difference Instruction for Generic Image Restoration | https://cvpr.thecvf.com/virtual/2025/poster/34792

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34792

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion-based Text-to-Image (T2I) models have demonstrated significant potential in image restoration. However, existing models continue to grapple with challenges such as complex training and prompt design. We introduce a new perspective for improving image restoration by injecting knowledge from pretrained vision-language models into current T2I models. We empirically show that the degradation and content representations in BLIP-2 can be linearly separated, providing promising degradation guidance for image restoration. Specifically, the Feature Difference Instruction (FDI) is first extracted by Q-Formers through a simple subtraction operation based on reference image pairs. Then, we propose a multi-scale FDI adapter to decouple the degradation style and corrupted artifacts, and inject the styleflow exclusively into specific blocks through adapter-tuning, thereby preventing noise interference and eschewing the need for cumbersome weight retraining.  In this way, we can train various task-specific adapters according to different degradations, achieving rich detail enhancement in the restoration results.  Furthermore, the proposed FDI adapters have attractive properties of practical value, such as composability and generalization ability for all-in-one and mixed-degradation restoration. Extensive experiments under various settings demonstrate that our method has promising repairing quality over 10 image restoration tasks and a wide range of other applications. Codes will be publicly available.

</details>

---

## 398. Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM

- [ ] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM | https://cvpr.thecvf.com/virtual/2025/poster/34801

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34801

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have become prominent in open-world image recognition for their strong generalization abilities. Yet, their effectiveness in practical applications is compromised by domain shifts and distributional changes, especially when test data distributions diverge from training data. Therefore, the paradigm of test-time adaptation (TTA) has emerged, enabling the use of online off-the-shelf data at test time, supporting independent sample predictions, and eliminating reliance on test annotations. Traditional TTA methods, however, often rely on costly training or optimization processes, or make unrealistic assumptions about accessing or storing historical training and test data.Instead, this study proposes FreeTTA, a training-free and universally available method that makes no assumptions, to enhance the flexibility of TTA. More importantly, FreeTTA is the first to explicitly model the test data distribution, enabling the use of intrinsic relationships among test samples to enhance predictions of individual samples without simultaneous access—a direction not previously explored. FreeTTA achieves these advantages by introducing an online EM algorithm that utilizes zero-shot predictions from VLMs as priors to iteratively compute the posterior probabilities of each online test sample and update parameters. Experiments demonstrate that FreeTTA achieves stable and significant improvements compared to state-of-the-art methods across 15 datasets in both cross-domain and out-of-distribution settings.

</details>

---

## 399. Libra-Merging: Importance-redundancy and Pruning-merging Trade-off for Acceleration Plug-in in Large Vision-Language Model

- [ ] Libra-Merging: Importance-redundancy and Pruning-merging Trade-off for Acceleration Plug-in in Large Vision-Language Model | https://cvpr.thecvf.com/virtual/2025/poster/34817

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34817

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have achieved significant progress in recent years. However, the expensive inference cost limits the realistic deployment of LVLMs. Some works find that visual tokens are redundant and compress tokens to reduce the inference cost. These works identify important non-redundant tokens as target tokens, then prune the remaining tokens (non-target tokens) or merge them into target tokens. However, target token identification faces the token importance-redundancy dilemma. Besides, token merging and pruning face a dilemma between disrupting target token information and losing non-target token information. To solve these problems, we propose a novel visual token compression scheme, named Libra-Merging. In target token identification, Libra-Merging selects the most important tokens from spatially discrete intervals, achieving a more robust token importance-redundancy trade-off than relying on a hyper-parameter. In token compression, when non-target tokens are dissimilar to target tokens, Libra-Merging does not merge them into the target tokens, thus avoiding disrupting target token information. Meanwhile, Libra-Merging condenses these non-target tokens into an information compensation token to prevent losing important non-target token information. Our method can serve as a plug-in for diverse LVLMs, and extensive experimental results demonstrate its effectiveness. The code will be publicly available.

</details>

---

## 400. Lifting the Veil on Visual Information Flow in MLLMs: Unlocking Pathways to Faster Inference

- [ ] Lifting the Veil on Visual Information Flow in MLLMs: Unlocking Pathways to Faster Inference | https://cvpr.thecvf.com/virtual/2025/poster/34818

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34818

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) improve performance on vision-language tasks by integrating visual features from pre-trained vision encoders into large language models (LLMs). However, how MLLMs process and utilize visual information remains unclear. In this paper, a shift in the dominant flow of visual information is uncovered: (1) in shallow layers, strong interactions are observed between image tokens and instruction tokens, where most visual information is injected into instruction tokens to form cross-modal semantic representations; (2) in deeper layers, image tokens primarily interact with each other, aggregating the remaining visual information to optimize semantic representations within the visual modality. Based on these insights, we propose Hierarchical Modality-Aware Pruning (HiMAP), a plug-and-play inference acceleration method that dynamically prunes image tokens at specific layers, reducing computational costs by approximately 65% without sacrificing performance. Our findings offer a new understanding of visual information processing in MLLMs and provide a state-of-the-art solution for efficient inference.

</details>

---

## 401. VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding

- [ ] VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding | https://cvpr.thecvf.com/virtual/2025/poster/34827

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34827

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have recently shown significant advancements in video understanding, excelling in content reasoning and instruction-following tasks. However, the problem of hallucination, where models generate inaccurate or misleading content, remains underexplored in the video domain. Building on the observation that the visual encoder of MLLMs often struggles to differentiate between video pairs that are visually distinct but semantically similar, we introduce VidHalluc, the largest benchmark designed to examine hallucinations in MLLMs for video understanding tasks. VidHalluc assesses hallucinations across three critical dimensions: (1) action, (2) temporal sequence, and (3) scene transition. VidHalluc consists of 5,002 videos, paired based on semantic similarity and visual differences, focusing on cases where hallucinations are most likely to occur. Through comprehensive testing, our experiments show that most MLLMs are vulnerable to hallucinations across these dimensions. Furthermore, we propose DINO-HEAL, a training-free method that reduces hallucinations by incorporating spatial saliency information from DINOv2 to reweight visual features during inference. Our results demonstrate that DINO-HEAL consistently improves performance on VidHalluc, achieving an average improvement of 3.02% in mitigating hallucinations among all tasks. Both the VidHalluc benchmark and DINO-HEAL code will be publicly released.

</details>

---

## 402. Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models

- [ ] Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34829

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34829

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) learn a shared feature space for text and images, enabling the comparison of inputs of different modalities. While prior works demonstrated that VLMs organize natural language representations into regular structures encoding composite meanings, it remains unclear if compositional patterns also emerge in the visual embedding space. In this work, we investigate compositionality in the image domain, where the analysis of compositional properties is challenged by noise and sparsity of visual data.We propose a framework, called Geodesically Decomposable Embeddings (GDE), that addresses these problems and approximates image representations with geometry-aware compositional structures in the latent space. We demonstrate that visual embeddings of pre-trained VLMs exhibit a compositional arrangement, and evaluate the effectiveness of this property in the tasks of compositional classification and group robustness. GDE achieves stronger performance in compositional classification compared to its counterpart method that assumes linear geometry of the latent space. Notably, it is particularly effective for group robustness, where we achieve higher results than task-specific solutions. Our results indicate that VLMs can automatically develop a human-like form of compositional reasoning in the visual domain, making their underlying processes more interpretable.

</details>

---

## 403. Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment

- [ ] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment | https://cvpr.thecvf.com/virtual/2025/poster/34843

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34843

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation (TTA) aims to improve the performance of source-domain pre-trained models on previously unseen, shifted target domains. Traditional TTA methods primarily adapt model weights based on target data streams, making model performance sensitive to the amount and order of target data. The recently proposed diffusion-driven TTA methods mitigate this by adapting model inputs instead of weights, where an unconditional diffusion model, trained on the source domain, transforms target-domain data into a synthetic domain that is expected to approximate the source domain. However, in this paper, we reveal that although the synthetic data in diffusion-driven TTA seems indistinguishable from the source data, it is unaligned with, or even markedly different from the latter for deep networks. To address this issue, we propose a Synthetic-Domain Alignment (SDA) framework. Our key insight is to fine-tune the source model with synthetic data to ensure better alignment. Specifically, we first employ a conditional diffusion model to generate labeled samples, creating a synthetic dataset. Subsequently, we use the aforementioned unconditional diffusion model to add noise to and denoise each sample before fine-tuning. This Mix of Diffusion (MoD) process mitigates the potential domain misalignment between the conditional and unconditional models. Extensive experiments across classifiers, segmenters, and multimodal large language models (MLLMs, \eg, LLaVA) demonstrate that SDA achieves superior domain alignment and consistently outperforms existing diffusion-driven TTA methods. Our code will be open-sourced.

</details>

---

## 404. DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models

- [ ] DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models | https://cvpr.thecvf.com/virtual/2025/poster/34849

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34849

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs) have emerged as powerful models capable of understanding various data modalities, including text, images, and videos. LMMs encode both text and visual data into tokens that are then combined and processed by an integrated Large Language Model (LLM). Including visual tokens substantially increases the total token count, often by thousands. The increased input length for LLM significantly raises the complexity of inference, resulting in high latency in LMMs. To address this issue, token pruning methods, which remove part of the visual tokens, are proposed. The existing token pruning methods either require extensive calibration and fine-tuning or rely on suboptimal importance metrics which results in increased redundancy among the retained tokens. In this paper, we first formulate token pruning as Max-Min Diversity Problem (MMDP) where the goal is to select a subset such that the diversity among the selected tokens is maximized. Then, we solve the MMDP to obtain the selected subset and prune the rest. The proposed method, DivPrune, reduces redundancy and achieves the highest diversity of the selected tokens. By ensuring high diversity, the selected tokens better represent the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments with various LMMs show that DivPrune achieves state-of-the-art accuracy over 16 image- and video-language datasets. Additionally, DivPrune reduces both the end-to-end latency and GPU memory usage for the tested models. The code is provided in the supplementary material.

</details>

---

## 405. Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution

- [ ] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution | https://cvpr.thecvf.com/virtual/2025/poster/34854

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34854

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of Multi-modal Large Language Models (MLLMs), MLLM-based Image Quality Assessment (IQA) methods have shown promising performance in linguistic quality description. However, current methods still fall short in accurately scoring image quality. In this work, we aim to leverage MLLMs to regress accurate quality scores. A key challenge is that the quality score is inherently continuous, typically modeled as a Gaussian distribution, whereas MLLMs generate discrete token outputs. This mismatch necessitates score discretization. Previous approaches discretize the mean score into a one-hot label, resulting in information loss and failing to capture inter-image relationships. We propose a distribution-based approach that discretizes the score distribution into a soft label. This method preserves the characteristics of the score distribution, achieving high accuracy and maintaining inter-image relationships. Moreover, to address dataset variation, where different IQA datasets exhibit various distributions, we introduce a fidelity loss based on Thurstone’s model. This loss captures intra-dataset relationships, facilitating co-training across multiple IQA datasets. With these designs, we develop the Di stribution-based m ulti-modal i mage Q uality A ssessment model (DimiQA). Experiments across multiple benchmarks show that DimiQA stably outperforms baselines in score regression. Also, DimiQA can predict the score distribution that closely aligns with human annotations. Codes and model weights will be released.

</details>

---

## 406. FlashSloth : Lightning Multimodal Large Language Models via Embedded Visual Compression

- [ ] FlashSloth : Lightning Multimodal Large Language Models via Embedded Visual Compression | https://cvpr.thecvf.com/virtual/2025/poster/34869

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34869

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite a big leap forward in capability, \emph{multimodal large language models} (MLLMs) tend to behave like a sloth in practical use, \emph{i.e.}, slow response and large latency. Recent efforts are devoted to building tiny MLLMs for better efficiency, but the plethora of visual tokens still used limit their actual speedup. In this paper, we propose a powerful and fast tiny MLLM called \emph{\textbf{FlashSloth}}. Different from previous efforts, FlashSloth focuses on improving the descriptive power of visual tokens in the process of compressing their redundant semantics. In particular, FlashSloth introduces embedded visual compression designs to capture both visually salient and instruction-related image information, so as to achieving superior multimodal performance with fewer visual tokens. Extensive experiments are conducted to validate the proposed FlashSloth, and a bunch of tiny but strong MLLMs are also comprehensively compared, e.g., InternVL-2, MiniCPM-V2 and Qwen2-VL. The experimental results show that compared with these advanced tiny MLLMs, our FlashSloth can greatly reduce the number of visual tokens, training memory and computation complexity while retaining high performance on various VL tasks. Our code is anonymously released at: \url{https://anonymous.4open.science/r/FlashSloth/}.

</details>

---

## 407. Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection for Visual Instruction Tuning

- [ ] Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection for Visual Instruction Tuning | https://cvpr.thecvf.com/virtual/2025/poster/34872

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34872

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual instruction tuning (VIT) for large vision-language models (LVLMs) requires training on expansive datasets of image-instruction pairs, which can be costly. Recent efforts in VIT data selection aim to select a small subset of high-quality image-instruction pairs, reducing VIT runtime while maintaining performance comparable to full-scale training. However, a major challenge often overlooked is that generating instructions from unlabeled images for VIT is highly expensive. Most existing VIT datasets rely heavily on human annotations or paid services like the GPT API, which limits users with constrained resources from creating VIT datasets for custom applications. To address this, we introduce Pre-Instruction Data Selection (PreSel), a more practical data selection paradigm that directly selects the most beneficial unlabeled images and generates instructions only for the selected images. PreSel first estimates the relative importance of each vision task within VIT datasets to derive task-wise sampling budgets. It then clusters image features within each task, selecting the most representative images with the budget. This approach reduces computational overhead for both instruction generation during VIT data formation and LVLM fine-tuning. By generating instructions for only 15% of the images, PreSel achieves performance comparable to full-data VIT on the LLaVA-1.5 and Vision-Flan datasets. Code will be made available.

</details>

---

## 408. MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation

- [ ] MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation | https://cvpr.thecvf.com/virtual/2025/poster/34879

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34879

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models have shown excellent performance in text-to-image generation. However, existing methods often suffer from performance bottlenecks when dealing with complex prompts involving multiple objects, characteristics, and relations.  Therefore, we propose a Multi-agent Collaboration-based Compositional Diffusion (MCCD) for text-to-image generation for complex scenes. Specifically, we design a multi-agent collaboration based scene parsing module that generates an agent system containing multiple agents with different tasks using MLLMs to adequately extract multiple scene elements. In addition, Hierarchical Compositional diffusion utilizes Gaussian mask and filtering to achieve the refinement of bounding box regions and highlights objects through region enhancement for accurate and high-fidelity generation of complex scenes. Comprehensive experiments demonstrate that our MCCD significantly improves the performance of the baseline models in a training-free manner, which has a large advantage in complex scene generation. The code will be open-source on github.

</details>

---

## 409. Dynamic Updates for Language Adaptation in Visual-Language Tracking

- [ ] Dynamic Updates for Language Adaptation in Visual-Language Tracking | https://cvpr.thecvf.com/virtual/2025/poster/34874

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34874

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The consistency between the semantic information provided by the multi-modal reference and the tracked object is crucial for visual-language (VL) tracking. However, existing VL tracking frameworks rely on static multi-modal references to locate dynamic objects, which can lead to semantic discrepancies and reduce the robustness of the tracker. To address this issue, we propose a novel vision-language tracking framework, named DUTrack, which captures the latest state of the target by dynamically updating multi-modal references to maintain consistency.Specifically, we introduce a Dynamic Language Update Module, which leverages a large language model to generate dynamic language descriptions for the object based on visual features and object category information. Then, we design a Dynamic Template Capture Module, which captures the regions in the image that highly match the dynamic language descriptions. Furthermore, to ensure the efficiency of description generation, we design an update strategy that assesses changes in target displacement, scale, and other factors to decide on updates. Finally, the dynamic template and language descriptions that record the latest state of the target are used to update the multi-modal references, providing more accurate reference information for subsequent inference and enhancing the robustness of the tracker.DUTrack achieves new state-of-the-art performance on four mainstream vision-language and two vision-only tracking benchmarks, including LaSOT, LaSOT_ext, TNL2K, OTB99-Lang, GOT-10K, and UAV123.

</details>

---

## 410. V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents

- [ ] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents | https://cvpr.thecvf.com/virtual/2025/poster/34885

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34885

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the recent advancement in video stylization, most existing methods struggle to render any video with complex transitions,based on an open style description of user query.To fill this gap,we introduce a generic multi-agent system for video stylization, $\textbf{V-Stylist}$, by a novel collaboration and reflection paradigm of multi-modal large language models. Specifically, our V-Stylist is a systematical workflow with three key roles: (1) $\textbf{Video Parser}$ decomposes the input video into a number of shots and generates their text prompts of key shot content.Via a concise video-to-shot prompting paradigm,it allows our V-Stylist to effectively handle videos with complex transitions. (2) $\textbf{Style Parser}$ identifies the style in the user query and progressively search the matched style model from a style tree.Via a robust tree-of-thought searching paradigm,it allows our V-Stylist to precisely specify vague style preference in the open user query.(3) $\textbf{Style Artist}$ leverages the matched model to render all the video shots into the required style.Via a novel multi-round self-reflection paradigm,it allows our V-Stylist to adaptively adjust detail control,according to the style requirement.With such a distinct design of mimicking human professionals, our V-Stylist achieves a major breakthrough over the primary challenges for effective and automatic video stylization. Moreover,  we further construct a new benchmark Text-driven Video Stylization Benchmark (TVSBench),which fills the gap to assess stylization of complex videos on open user queries. Extensive experiments show that, V-Stylist achieves the state-of-the-art,e.g.,V-Stylist surpasses FRESCO and ControlVideo by 6.05\% and 4.51\% respectively in overall average metrics, marking a significant advance in video stylization.

</details>

---

## 411. VisionZip: Longer is Better but Not Necessary in Vision Language Models

- [ ] VisionZip: Longer is Better but Not Necessary in Vision Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34888

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34888

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in vision-language models have enhanced performance by increasing the length of visual tokens, making them much longer than text tokens and significantly raising computational costs.However, we observe that the visual tokens generated by popular vision encoders, such as CLIP and SigLIP, contain significant redundancy. To address this, we introduce VisionZip, a simple yet effective method that selects a set of informative tokens for input to the language model, reducing visual token redundancy and improving efficiency while maintaining model performance. The proposed VisionZip can be widely applied to image and video understanding tasks and is well-suited for multi-turn dialogues in real-world scenarios, where previous methods tend to underperform.Experimental results show that VisionZip outperforms the previous state-of-the-art method by at least 5\% performance gains across nearly all settings.Moreover, our method significantly enhances model inference speed, improving the prefilling time by 8$\times$ and enabling the LLaVA-Next 13B model to infer faster than the LLaVA-Next 7B model while achieving better results.Furthermore, we analyze the causes of this redundancy and encourage the community to focus on extracting better visual features rather than merely increasing token length. All code and models will be publicly available.

</details>

---

## 412. StarVector: Generating Scalable Vector Graphics Code from Images and Text

- [ ] StarVector: Generating Scalable Vector Graphics Code from Images and Text | https://cvpr.thecvf.com/virtual/2025/poster/34902

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34902

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scalable Vector Graphics (SVGs) are vital for modern image rendering due to their scalability and versatility. Previous SVG generation methods have focused on curve-based vectorization, lacking semantic understanding, often producing artifacts, and struggling with SVG primitives beyond \textit{path} curves. To address these issues, we introduce StarVector, a multimodal large language model for SVG generation. It performs image vectorization by understanding image semantics and using SVG primitives for compact, precise outputs. Unlike traditional methods, StarVector works directly in the SVG code space, leveraging visual understanding to apply accurate SVG primitives. To train StarVector, we create SVG-Stack, a diverse dataset of 2M samples that enables generalization across vectorization tasks and precise use of primitives like ellipses, polygons, and text. We address challenges in SVG evaluation, showing that pixel-based metrics like MSE fail to capture the unique qualities of vector graphics. We introduce SVG-Bench, a benchmark across 10 datasets, and three tasks: image vectorization, text-driven SVG generation, and diagram generation. Using this contribution, StarVector achieves state-of-the-art performance, producing more compact and semantically rich SVGs.

</details>

---

## 413. Towards Understanding How Knowledge Evolves in Large Vision-Language Models

- [ ] Towards Understanding How Knowledge Evolves in Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/34921

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34921

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) are gradually becoming the foundation for many artificial intelligence applications. However, understanding their internal working mechanisms has continued to puzzle researchers, which in turn limits the further enhancement of their capabilities. In this paper, we seek to investigate how multimodal knowledge evolves and eventually induces natural languages in LVLMs. We design a series of novel strategies for analyzing internal knowledge within LVLMs, and delve into the evolution of multimodal knowledge from three levels, including single token probabilities, token probability distributions, and feature encodings. In this process,  we identify two key nodes in knowledge evolution: the critical layers and the mutation layers, dividing the evolution process into three stages: rapid evolution, stabilization, and mutation. Our research is the first to reveal the trajectory of knowledge evolution in LVLMs, providing a fresh perspective for understanding their underlying mechanisms.

</details>

---

## 414. VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents

- [ ] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents | https://cvpr.thecvf.com/virtual/2025/poster/34926

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34926

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We aim to develop a retrieval-augmented generation (RAG) framework capable of answering questions over a corpus of visually-rich documents presented in mixed modalities (e.g., charts, tables) and diverse formats (e.g., PDF, PPTX). In this paper, we present a new RAG framework, VDocRAG, which can directly understand varied documents and modalities in a unified image format to prevent missing information that occurs by parsing documents to obtain text. To improve the performance of VDocRAG, we propose novel self-supervised pre-training tasks that adapt large vision-language models for retrieval by compressing visual information into dense token representations while aligning them with textual content in documents. Furthermore, we introduce OpenDocVQA, the first unified collection of open-domain document visual question answering datasets, encompassing diverse document types and formats. OpenDocVQA provides a comprehensive resource for training and evaluating retrieval and question answering models on visually-rich documents in an open-domain setting. Experiments show that VDocRAG substantially outperforms conventional text-based RAG and has strong generalization capability, highlighting the potential of an effective RAG paradigm for real-world documents.

</details>

---

## 415. Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering

- [ ] Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering | https://cvpr.thecvf.com/virtual/2025/poster/34931

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34931

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal LLMs (MLLMs) are the natural extension of large language models to handle multimodal inputs, combining text and image data. They have recently garnered attention due to their capability to address complex tasks involving both modalities. However, their effectiveness is limited to the knowledge acquired during training, which restricts their practical utility. In this work, we introduce a novel method to enhance the adaptability of MLLMs by integrating external knowledge sources. Our proposed model, Reflective LLaVA (ReflectiVA), utilizes reflective tokens to dynamically determine the need for external knowledge and predict the relevance of information retrieved from an external database. Tokens are trained following a two-stage two-model training recipe. This ultimately enables the MLLM to manage external knowledge while preserving fluency and performance on tasks where external knowledge is not needed. Through our experiments, we demonstrate the efficacy of ReflectiVA for knowledge-based visual question answering, highlighting its superior performance compared to existing methods. Source code and models will be publicly released.

</details>

---

## 416. Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map

- [ ] Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map | https://cvpr.thecvf.com/virtual/2025/poster/34937

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34937

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Ensuring adherence to traffic sign regulations is essential for both human and autonomous vehicle navigation. While current online mapping solutions often prioritize the construction of the geometric and connectivity layers of HD maps, overlooking the construction of the traffic regulation layer within HD maps. Addressing this gap, we introduce MapDR, a novel dataset designed for the extraction of Driving Rules from traffic signs and their association with vectorized, locally perceived HD Maps. MapDR features over $10,000$ annotated video clips that capture the intricate correlation between traffic sign regulations and lanes. Built upon this benchmark and the newly defined task of integrating traffic regulations into online HD maps, we provide modular and end-to-end solutions: VLE-MEE and RuleVLM, offering a strong baseline for advancing autonomous driving technology. It fills a critical gap in the integration of traffic sign rules, contributing to the development of reliable autonomous driving systems.

</details>

---

## 417. COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts

- [ ] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts | https://cvpr.thecvf.com/virtual/2025/poster/34946

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34946

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current object detectors often suffer significant performance degradation in real-world applications when encountering distributional shifts, posing serious risks in high-stakes domains such as autonomous driving and medical diagnosis. Consequently, the out-of-distribution (OOD) generalization capability of object detectors has garnered increasing attention from researchers. Despite this growing interest, there remains a lack of a large-scale, comprehensive dataset and evaluation benchmark with fine-grained annotations tailored to assess the OOD generalization on more intricate tasks like object detection and grounding. To address this gap, we introduce COUNTS, a large-scale OOD dataset with object-level annotations. COUNTS encompasses 14 natural distributional shifts, over 222K samples, and more than 1,196K labeled bounding boxes. Leveraging COUNTS, we introduce two novel benchmarks: O(OD) and OODG. OODOD is designed to comprehensively evaluate the OOD generalization capabilities of object detectors by utilizing controlled distribution shifts between training and testing data. OODG, on the other hand, aims to assess the OOD generalization of grounding abilities in multimodal large language models (MLLMs). Our findings reveal that, while large models and extensive pre-training data substantially enhance performance in in-distribution (IID) scenarios, significant limitations and opportunities for improvement persist in OOD contexts for both object detectors and MLLMs. In visual grounding tasks, even the advanced GPT-4o and Gemini-1.5 only achieve 56.7% and 28.0% accuracy, respectively. We hope COUNTS facilitates advancements in the development and assessment of robust object detectors and MLLMs capable of maintaining high performance under distributional shifts.

</details>

---

## 418. Automated Generation of Challenging Multiple-Choice Questions for Vision Language Model Evaluation

- [ ] Automated Generation of Challenging Multiple-Choice Questions for Vision Language Model Evaluation | https://cvpr.thecvf.com/virtual/2025/poster/34956

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34956

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of vision language models (VLMs) demands rigorous and reliable evaluation. However, current visual question answering (VQA) benchmarks often depend on open-ended questions, making accurate evaluation difficult due to the variability in natural language responses. To address this, we introduce AutoConverter, an agentic framework that automatically converts these open-ended questions into multiple-choice format, enabling objective evaluation while reducing the costly question creation process. Our experiments demonstrate that AutoConverter can generate correct and challenging multiple-choice questions, with VLMs demonstrating consistently similar or lower accuracy on these questions compared to human-created ones. Using AutoConverter, we construct VMCBench, a benchmark created by transforming 20 existing VQA datasets into a unified multiple-choice format, totaling 9,018 questions. We comprehensively evaluate 28 state-of-the-art VLMs on VMCBench, setting a new standard for scalable, consistent, and reproducible VLM evaluation.

</details>

---

## 419. FirePlace: Geometric Refinements of LLM Common Sense Reasoning for 3D Object Placement

- [ ] FirePlace: Geometric Refinements of LLM Common Sense Reasoning for 3D Object Placement | https://cvpr.thecvf.com/virtual/2025/poster/34985

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34985

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scene generation with 3D assets presents a complex challenge, requiring both high-level semantic understanding and low-level geometric reasoning. While Multimodal Large Language Models (MLLMs) excel at semantic tasks, their application to 3D scene generation is hindered by their limited grounding on 3D geometry. In this paper, we investigate how to best work with MLLMs in an object placement task. Towards this goal, we introduce a novel framework, FirePlace, that applies existing MLLMs in (1) 3D geometric reasoning and the extraction of relevant geometric details from the 3D scene, (2) constructing and solving geometric constraints on the extracted low-level geometry, and (3) pruning for final placements that conform to common sense. By combining geometric reasoning with real-world understanding of MLLMs, our method can propose object placements that satisfy both geometric constraints as well as high-level semantic common-sense considerations. Our experiments show that these capabilities allow our method to place objects more effectively in complex scenes with intricate geometry, surpassing the quality of prior work.

</details>

---

## 420. PEACE: Empowering Geologic Map Holistic Understanding with MLLMs

- [ ] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs | https://cvpr.thecvf.com/virtual/2025/poster/34989

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34989

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Geologic map, as a fundamental diagram in geology science, provides critical insights into the structure and composition of Earth's subsurface and surface.These maps are indispensable in various fields, including disaster detection, resource exploration, and civil engineering.Despite their significance, current Multimodal Large Language Models (MLLMs) often fall short in geologic map understanding.This gap is primarily due to the challenging nature of cartographic generalization, which involves handling high-resolution map, managing multiple associated components, and requiring domain-specific knowledge.To quantify this gap, we construct GeoMap-Bench, the first-ever benchmark for evaluating MLLMs in geologic map understanding, which assesses the full-scale abilities in extracting, referring, grounding, reasoning, and analyzing.To bridge this gap, we introduce GeoMap-Agent, the inaugural agent designed for geologic map understanding,which features three modules: Hierarchical Information Extraction (HIE), Domain Knowledge Injection (DKI), and Prompt-enhanced Question Answering (PEQA).Inspired by the interdisciplinary collaboration among human scientists, an AI expert group acts as consultants, utilizing a diverse tool pool to comprehensively analyze questions.Through comprehensive experiments, GeoMap-Agent achieves an overall score of 0.811 on GeoMap-Bench, significantly outperforming 0.369 of GPT-4o.Our work, em P owering g E ologic m A p holisti C und E rstanding ( PEACE ) with MLLMs, paves the way for advanced AI applications in geology, enhancing the efficiency and accuracy of geological investigations.

</details>

---

## 421. Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis

- [ ] Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis | https://cvpr.thecvf.com/virtual/2025/poster/34996

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/34996

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing 3D vision-language (3D-VL) benchmarks fall short in evaluating 3D-VL models, creating a “mist” that obscures rigorous insights into model capabilities and 3D-VL tasks. This mist persists due to three key limitations. First, flawed test data, like ambiguous referential text in the grounding task, can yield incorrect and unreliable test results. Second, oversimplified metrics such as simply averaging accuracy per question answering (QA) pair, cannot reveal true model capability due to their vulnerability to language variations. Third, existing benchmarks isolate the grounding and QA tasks, disregarding the underlying coherence that QA should be based on solid grounding capabilities. To unveil the “mist”, we propose Beacon3D, a benchmark for 3D-VL grounding and QA tasks, delivering a perspective shift in the evaluation of 3D-VL understanding. Beacon3D features (i) high-quality test data with precise and natural language, (ii) object-centric evaluation with multiple tests per object to ensure robustness, and (iii) a novel chain-of-analysis paradigm to address language robustness and model performance coherence across grounding and QA. Our evaluation of state-of-the-art 3D-VL models on Beacon3D reveals that (i) object-centric evaluation elicits true model performance and particularly weak generalization in QA; (ii) grounding-QA coherence remains fragile in current 3D-VL models, and (iii) incorporating large language models (LLMs) to 3D-VL models, though commonly viewed as a practical technique, hinders grounding capabilities and has yet to elevate QA capabilities. We hope Beacon3D and our comprehensive analysis could benefit the 3D-VL community towards faithful developments.

</details>

---

## 422. ReWind: Understanding Long Videos with Instructed Learnable Memory

- [ ] ReWind: Understanding Long Videos with Instructed Learnable Memory | https://cvpr.thecvf.com/virtual/2025/poster/35002

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35002

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) are crucial for real-world applications that require understanding textual and visual information. However, existing VLMs face multiple challenges in processing long videos, including computational inefficiency, memory limitations, and difficulties maintaining coherent understanding across extended sequences. These issues stem partly from the quadratic scaling of self-attention w.r.t. number of tokens but also encompass broader challenges in temporal reasoning and information integration over long sequences. To address these challenges, we introduce ReWind, a novel two-stage framework for long video understanding. In the first stage, ReWind maintains a dynamic memory that stores and updates instruction-relevant visual information as the video unfolds.Memory updates leverage novel read and write mechanisms utilizing learnable queries and cross-attentions between memory contents and the input stream. This approach maintains low memory requirements as the cross-attention layers scale linearly w.r.t. number of tokens. In the second stage, the memory content guides the selection of a few relevant frames, represented at high spatial resolution, which are combined with the memory contents and fed into an LLM to generate the final answer. We empirically demonstrate ReWind's superiority in visual question answering (VQA) and temporal grounding tasks, surpassing previous methods on long video benchmarks. Notably, ReWind achieves a +13\% score gain and a +12\% accuracy improvement on the MovieChat-1K VQA dataset and an +8\% mIoU increase on Charades-STA for temporal grounding.

</details>

---

## 423. LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences

- [ ] LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences | https://cvpr.thecvf.com/virtual/2025/poster/35004

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35004

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Research on 3D Vision-Language Models (3D-VLMs) is gaining increasing attention, which is crucial for developing embodied AI within 3D scenes, such as visual navigation and embodied question answering. Due to the high density of visual features, especially in large 3D scenes, accurately locating task-relevant visual information is challenging. Existing works attempt to segment all objectsand consider their features as scene representations. However, these task-agnostic object features include much redundant information and missing details for the task-relevant area. To tackle these problems, we propose LSceneLLM, an adaptive framework that automatically identifies task-relevant areas by leveraging LLM's visual preference for different tasks, followed by a plug-and-play scene magnifier module to capture fine-grained details in focused areas. Specifically, a dense token selector examines the attention map of LLM to identify visual preferences for the instruction input. It then magnifies fine-grained details of the focusing area. An adaptive self-attention module is leveraged to fuse the coarse-grained and selected fine-grained visual information. To comprehensively evaluate the large scene understanding ability of 3D-VLMs, we further introduce a cross-room understanding benchmark, XR-Scene, which contains a series of large scene understanding tasks including XR-QA, XR-EmbodiedPlanning, and XR-SceneCaption. Experiments show that our method surpasses existing methods on both large scene understanding and existing scene understanding benchmarks. Plunging our scene magnifier module into the existing 3D-VLMs also brings significant improvement.

</details>

---

## 424. RelationField: Relate Anything in Radiance Fields

- [ ] RelationField: Relate Anything in Radiance Fields | https://cvpr.thecvf.com/virtual/2025/poster/35013

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35013

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Neural radiance fields are an emerging 3D scene representation and recently even been extended to learn features for scene understanding by distilling open-vocabulary features from vision-language models. However, current method primarily focus on object-centric representations, supporting object segmentation or detection, while understanding semantic relationships between objects remains largely unexplored. To address this gap, we propose RelationField, the first method to extract inter-object relationships directly from neural radiance fields. RelationField represents relationships between objects as pairs of rays within a neural radiance field, effectively extending its formulation to include implicit relationship queries. To teach RelationField complex, open-vocabulary relationships, relationship knowledge is distilled from multi-modal LLMs. To evaluate RelationField, we solve open-vocabulary 3D scene graph generation tasks and relationship-guided instance segmentation, achieving state-of-the-art performance in both tasks.

</details>

---

## 425. Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models

- [ ] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/35020

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35020

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-Shot Anomaly Detection (ZSAD) is an emerging AD paradigm. Unlike the traditional unsupervised AD setting that requires a large number of normal samples to train a model, ZSAD is more practical for handling data-restricted real-world scenarios. Recently, Multimodal Large Language Models (MLLMs) have shown revolutionary reasoning capabilities in various vision tasks. However, the reasoning of image abnormalities remains underexplored due to the lack of corresponding datasets and benchmarks. To facilitate research in anomaly detection and reasoning, we establish the first visual instruction tuning dataset, Anomaly-Instruct-125k, and the evaluation benchmark, VisA-D&R. Through investigation with our benchmark, we reveal that current MLLMs like GPT-4o cannot accurately detect and describe fine-grained anomalous details in images. To address this, we propose Anomaly-OneVision (Anomaly-OV), the first specialist visual assistant for ZSAD and reasoning, based on LLaVA-OneVision. Inspired by human behavior in visual inspection, Anomaly-OV leverages a Look-Twice Feature Matching (LTFM) mechanism to adaptively select and emphasize abnormal visual tokens for its LLM. Extensive experiments demonstrate that Anomaly-OV achieves significant improvements over advanced generalist models in both detection and reasoning. Furthermore, extensions to medical and 3D anomaly reasoning are provided for future study.

</details>

---

## 426. FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts

- [ ] FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts | https://cvpr.thecvf.com/virtual/2025/poster/35021

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35021

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Controllability plays a crucial role in the practical applications of 3D indoor scene synthesis. Existing works either allow rough language-based control, that is convenient but lacks fine-grained scene customization, or employ graph-based control, which offers better controllability but demands considerable knowledge for the cumbersome graph design process. To address these challenges, we present FreeScene, a user-friendly framework that enables both convenient and effective control for indoor scene synthesis. Specifically, FreeScene supports free-form user inputs including text description and/or reference images, allowing users to express versatile design intentions. The user inputs are adequately analyzed and integrated into a graph representation by a VLM-based Graph Designer. We then propose MG-DiT, a Mixed Graph Diffusion Transformer, which performs graph-aware denoising to enhance scene generation. Our MG-DiT not only excels at preserving graph structure but also offers broad applicability to various tasks, including, but not limited to, text-to-scene, graph-to-scene, and rearrangement, all within a single model. Extensive experiments demonstrate that FreeScene provides an efficient and user-friendly solution that unifies text-based and graph-based scene synthesis, outperforming state-of-the-art methods in terms of both generation quality and controllability in a range of applications.

</details>

---

## 427. Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis

- [ ] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis | https://cvpr.thecvf.com/virtual/2025/poster/35031

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35031

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models have achieved impressive success in generating photorealistic images, but challenges remain in ensuring precise semantic alignment with input prompts. Optimizing the initial noisy latent offers a more efficient alternative to modifying model architectures or prompt engineering for improving semantic alignment. A latest approach, InitNo, refines the initial noisy latent by leveraging attention maps; however, these maps capture only limited information, and the effectiveness of InitNo is highly dependent on the initial starting point, as it tends to converge on a local optimum near this point. To this end, this paper proposes leveraging the language comprehension capabilities of large vision-language models (LVLMs) to guide the optimization of the initial noisy latent, and introduces the Noise Diffusion process, which updates the noisy latent to generate semantically faithful images while preserving distribution consistency. Furthermore, we provide a theoretical analysis of the condition under which the update improves semantic faithfulness. Experimental results demonstrate the effectiveness and adaptability of our framework, consistently enhancing semantic alignment across various diffusion models.

</details>

---

## 428. Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens

- [ ] Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens | https://cvpr.thecvf.com/virtual/2025/poster/35035

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35035

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucinations in Large Vision-Language Models (LVLMs) significantly undermine their reliability, motivating researchers to explore the causes of hallucination. However, most studies primarily focus on the language aspect rather than the visual. In this paper, we address how LVLMs process visual information and whether this process causes hallucination. Firstly, we use the attention lens to identify the stages at which LVLMs handle visual data, discovering that the middle layers are crucial. Moreover, we find that these layers can be further divided into two stages: "visual information enrichment" and "semantic refinement" which respectively propagate visual data to object tokens and interpret it through text. By analyzing attention patterns during the visual information enrichment stage, we find that real tokens consistently receive higher attention weights than hallucinated ones, serving as a strong indicator of hallucination. Further examination of multi-head attention maps reveals that hallucination tokens often result from heads interacting with inconsistent objects. Based on these insights, we propose a simple inference-time method that adjusts visual attention by integrating information across various heads. Extensive experiments demonstrate that this approach effectively mitigates hallucinations in mainstream LVLMs without additional training costs. Our code will be released at: https://anonymous.4open.science/r/middle layers indicating_hallucinations-C45A.

</details>

---

## 429. Joint Vision-Language Social Bias Removal for CLIP

- [ ] Joint Vision-Language Social Bias Removal for CLIP | https://cvpr.thecvf.com/virtual/2025/poster/35050

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35050

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language (V-L) pre-trained models such as CLIP show prominent capabilities in various downstream tasks. Despite this promise, V-L models are notoriously limited by their inherent social biases. A typical demonstration is that V-L models often produce biased predictions against specific groups of people, significantly undermining their real-world applicability. Existing approaches endeavor to mitigate the social bias problem in V-L models by removing biased attribute information from model embeddings. However, after our revisiting of these methods, we find that their bias removal is frequently accompanied by greatly compromised V-L alignment capabilities. We then reveal that this performance degradation stems from the unbalanced debiasing in image and text embeddings. To address this issue, we propose a novel V-L debiasing framework to align image and text biases followed by removing them from both modalities. By doing so, our method achieves multi-modal bias mitigation while maintaining the V-L alignment in the debiased embeddings. Additionally, we advocate a new evaluation protocol that can 1) holistically quantify the model debiasing and V-L alignment ability, and 2) evaluate the generalization of social bias removal models. We believe this work will offer new insights and guidance for future studies addressing the social bias problem in CLIP.

</details>

---

## 430. BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models

- [ ] BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models | https://cvpr.thecvf.com/virtual/2025/poster/35047

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35047

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) extend large language models (LLMs) to process multi-modal information, enabling them to generate responses to image-text inputs. MLLMs have been incorporated into diverse multi-modal applications, such as autonomous driving and medical diagnosis, via plug-and-play without fine-tuning. This deployment paradigm increases the vulnerability of MLLMs to backdoor attacks. However, existing backdoor attacks against MLLMs achieve limited effectiveness and stealthiness. In this work, we propose $\textit{BadToken}$, the first token-level backdoor attack to MLLMs. BadToken introduces two novel backdoor behaviors: $\textit{Token-substitution}$ and $\textit{Token-addition}$, which enable flexible and stealthy attacks by making token-level modifications to the original output for backdoored inputs. We formulate a general optimization problem that considers the two backdoor behaviors to maximize the attack effectiveness. We evaluate BadToken on two open-source MLLMs and various tasks. Our results show that our attack maintains the model's utility while achieving high attack success rates and stealthiness. We also show the real-world threats of BadToken in two scenarios, i.e., autonomous driving and medical diagnosis. Furthermore, we consider defenses including fine-tuning and input purification. Our results highlight the threat of our attack.

</details>

---

## 431. EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering

- [ ] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering | https://cvpr.thecvf.com/virtual/2025/poster/35056

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35056

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce EgoTextVQA, a novel and rigorously constructed benchmark for egocentric QA assistance involving scene text. EgoTextVQA contains 1.5$K$ ego-view videos and 7$K$ scene-text aware questions that reflect real-user needs in outdoor driving and indoor house-keeping activities. The questions are designed to elicit identification and reasoning on scene text in an egocentric and dynamic environment. With EgoTextVQA, we comprehensively evaluate 10 prominent multimodal large language models. Currently, all models struggle, and the best results (Gemini Pro) are around 33\% accuracy, highlighting the severe deficiency of these techniques in egocentric QA assistance. Our further investigations suggest that precise temporal grounding and multi-frame reasoning, along with high resolution and auxiliary scene-text inputs, are key for better performance. With thorough analyses and heuristic suggestions, we hope EgoTextVQA can serve as a solid testbed for research in egocentric scene-text QA assistance.

</details>

---

## 432. Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages

- [ ] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages | https://cvpr.thecvf.com/virtual/2025/poster/35059

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35059

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

An old-school recipe for training a classifier is to (i) learn a good feature extractor and (ii) optimize a linear layer atop. When only a handful of samples are available per category, as in Few-Shot Adaptation (FSA), data are insufficient to fit a large number of parameters, rendering the above impractical. This is especially true with large pre-trained Vision-Language Models (VLMs), which motivated successful research at the intersection of Parameter-Efficient Fine-tuning (PEFT) and FSA. In this work, we start by analyzing the learning dynamics of PEFT techniques when trained on few-shot data from only a subset of categories, referred to as the “base” classes. We show that such dynamics naturally splits into two distinct phases: (i) task-level feature extraction and (ii) specialization to the available concepts. To accommodate this dynamic, we then depart from prompt- or adapter-based methods and tackle FSA differently. Specifically, given a fixed computational budget, we split it to (i) learn a task-specific feature extractor via PEFT and (ii) train a linear classifier on top. We call this scheme Two-Stage Few-Shot Adaptation (2SFS). Differently from established methods, our scheme enables a novel form of selective inference at a category level, i.e., at test time, only novel categories are embedded by the adapted text encoder, while embeddings of base categories are available within the classifier. Results with fixed hyperparameters across two settings, three backbones, and eleven datasets, show that 2SFS matches or surpasses the state-of-the-art, while established methods degrade significantly across settings.

</details>

---

## 433. ReasonGrounder: LVLM-Guided Hierarchical Feature Splatting for Open-Vocabulary 3D Visual Grounding and Reasoning

- [ ] ReasonGrounder: LVLM-Guided Hierarchical Feature Splatting for Open-Vocabulary 3D Visual Grounding and Reasoning | https://cvpr.thecvf.com/virtual/2025/poster/35062

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35062

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary 3D visual grounding and reasoning aim to localize objects in a scene based on implicit language descriptions, even when they are occluded. This ability is crucial for tasks such as vision-language navigation and autonomous robotics. However, current methods struggle because they rely heavily on fine-tuning with 3D annotations and mask proposals, which limits their ability to handle diverse semantics and common knowledge required for effective reasoning. To address this, we propose ReasonGrounder, an LVLM-guided framework that uses hierarchical 3D feature Gaussian fields for adaptive grouping based on physical scale, enabling open-vocabulary 3D grounding and reasoning. ReasonGrounder interprets implicit instructions using large vision-language models (LVLM) and localizes occluded objects through 3D Gaussian splatting. By incorporating 2D segmentation masks from the Segment Anything Model (SAM) and multi-view CLIP embeddings, ReasonGrounder selects Gaussian groups based on object scale, enabling accurate localization through both explicit and implicit language understanding, even in novel, occluded views. We also contribute ReasoningGD, a new dataset containing over 10K scenes and 2 million annotations for evaluating open-vocabulary 3D grounding and amodal perception under occlusion. Experiments show that ReasonGrounder significantly improves 3D grounding accuracy in real-world scenarios.

</details>

---

## 434. Improving Personalized Search with Regularized Low-Rank Parameter Updates

- [ ] Improving Personalized Search with Regularized Low-Rank Parameter Updates | https://cvpr.thecvf.com/virtual/2025/poster/35065

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35065

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Personalized vision-language retrieval seeks to recognize new concepts (e.g. "my dog Fido'') from only a few examples. This task is challenging because it requires not only learning a new concept from a few images, but also integrating the personal and general knowledge together to recognize the concept in different contexts. In this paper, we show how to effectively adapt the internal representation of a vision-language dual encoder model for personalized vision-language retrieval. We find that regularized low-rank adaption of a small set of parameters in the language encoder's final layer serves as a highly effective alternative to textual inversion for recognizing the personal concept while preserving general knowledge. Additionally, we explore strategies for combining parameters of multiple learned personal concepts, finding that parameter addition is effective. To evaluate how well general knowledge is preserved in a finetuned representation, we introduce a metric that measures image retrieval accuracy based on captions generated by a vision language model (VLM). Our approach achieves state-of-the-art accuracy on two benchmarks for personalized image retrieval with natural language queries -- DeepFashion2 and ConConChi -- outperforming the prior art by 4%-22% on personal retrievals.

</details>

---

## 435. Visual Persona: Foundation Model for Full-Body Human Customization

- [ ] Visual Persona: Foundation Model for Full-Body Human Customization | https://cvpr.thecvf.com/virtual/2025/poster/35063

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35063

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Visual Persona, a foundation model for text-to-image full-body human customization that, given a single in-the-wild human image, generates diverse images of the individual guided by text descriptions. Unlike prior methods that focus solely on preserving facial identity, our approach captures detailed full-body appearance, aligning with text descriptions for body structure and scene variations. Training this model requires large-scale paired human data, consisting of multiple images per individual with consistent full-body identities, which is notoriously difficult to obtain. To address this, we propose a data curation pipeline leveraging vision language models to evaluate full-body appearance consistency, resulting in Visual Persona-500K—a dataset of 580k paired human images across 100k unique identities. For precise appearance transfer, we introduce a transformer encoder-decoder architecture adapted to a pre-trained text-to-image diffusion model, which augments the input image into distinct body regions, encodes these regions as local appearance features, and projects them into dense identity embeddings independently to condition the diffusion model for synthesizing customized images. Visual Persona consistently surpasses existing approaches, generating high-quality, customized images from in-the-wild inputs. Extensive ablation studies validate design choices, and we demonstrate the versatility of Visual Persona across various downstream tasks. The code and pre-trained weights will be publicly available.

</details>

---

## 436. XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?

- [ ] XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery? | https://cvpr.thecvf.com/virtual/2025/poster/35068

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35068

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The astonishing breakthrough of multimodal large language models (MLLMs) has necessitated new benchmarks to quantitatively assess their capabilities, reveal their limitations, and indicate future research directions. However, this is challenging in the context of remote sensing (RS), since the imagery features ultra-high resolution that incorporates extremely complex semantic relationships. Existing benchmarks usually adopt notably smaller image sizes than real-world RS scenarios, suffer from limited annotation quality, and consider insufficient dimensions of evaluation. To address these issues, we present XLRS-Bench: a comprehensive benchmark for evaluating the perception and reasoning capabilities of MLLMs in ultra-high-resolution RS scenarios. XLRS-Bench boasts the largest average image size (8500$\times$8500) observed thus far, with all evaluation samples meticulously annotated manually, assisted by a novel semi-automatic captioner on ultra-high-resolution RS images. On top of the XLRS-Bench, 16 sub-tasks are defined to evaluate MLLMs' 6 kinds of perceptual abilities and 4 kinds of reasoning capabilities, with a primary emphasis on advanced cognitive processes that facilitate real-world decision-making and the capture of spatiotemporal changes. The results of both general and RS-focused MLLMs on XLRS-Bench indicate that further efforts are needed to enhance their performance in real RS scenarios. We will open source XLRS-Bench to support further research of developing more powerful MLLMs for RS.

</details>

---

## 437. DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation

- [ ] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation | https://cvpr.thecvf.com/virtual/2025/poster/35070

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35070

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Story visualization, the task of creating visual narratives from textual descriptions, has seen progress with text-to-image generation models. However, these models often lack effective control over character appearances and interactions, particularly in multi-character scenes. To address these limitations, we propose a new task: \textbf{customized manga generation} and introduce \textbf{DiffSensei}, an innovative framework specifically designed for generating manga with dynamic multi-character control. DiffSensei integrates a diffusion-based image generator with a multimodal large language model (MLLM) that acts as a text-compatible identity adapter. Our approach employs masked cross-attention to seamlessly incorporate character features, enabling precise layout control without direct pixel transfer. Additionally, the MLLM-based adapter adjusts character features to align with panel-specific text cues, allowing flexible adjustments in character expressions, poses, and actions. We also introduce \textbf{MangaZero}, a large-scale dataset tailored to this task, containing 43,264 manga pages and 427,147 annotated panels, supporting the visualization of varied character interactions and movements across sequential frames. Extensive experiments demonstrate that DiffSensei outperforms existing models, marking a significant advancement in manga generation by enabling text-adaptable character customization. The code, model, and dataset will be open-sourced to the community.

</details>

---

## 438. SPARC: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models

- [ ] SPARC: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/35074

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35074

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot multi-label recognition (MLR) with Vision-Language Models (VLMs) faces significant challenges without training data, model tuning, or architectural modifications. Existing approaches require prompt tuning or architectural adaptations, limiting zero-shot applicability. Our work proposes a novel solution treating VLMs as black boxes, leveraging scores without training data or ground truth. Using large language model insights on object co-occurrence, we introduce compound prompts grounded in realistic object combinations. Analysis of these prompt scores reveals VLM biases and AND''/ OR'' signal ambiguities, notably that maximum compound scores are surprisingly suboptimal compared to second-highest scores. We address these through a debiasing and score-fusion algorithm that corrects image bias and clarifies VLM response behaviors. Our method enhances other zero-shot approaches, consistently improving their results. Experiments show superior mean Average Precision (mAP) compared to methods requiring training data, achieved through refined object ranking for robust zero-shot MLR.

</details>

---

## 439. Semantic and Expressive Variations in Image Captions Across Languages

- [ ] Semantic and Expressive Variations in Image Captions Across Languages | https://cvpr.thecvf.com/virtual/2025/poster/35077

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35077

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Most vision-language models today are primarily trained on English image-text pairs, with non-English pairs often filtered out. Evidence from cross-cultural psychology suggests that this approach will bias models against perceptual modes exhibited by people who speak other (non-English) languages.We investigate semantic and expressive variation in image captions across different languages; we analyze both human-annotated datasets and model-produced captions.By analyzing captions across seven languages (English, French, German, Russian, Chinese, Japanese, Korean) in high-quality image captioning datasets (Crossmodal and Visual Genome), we find that multilingual caption sets tend to provide richer visual descriptions than monolingual (including English-only) ones; multilingual sets contain 46.0% more objects66.1% more relationships, and66.8% more attributes.We observe the same results with multilingual captions produced by LLaVA and the Google Vertex API: for example, compared to monolingual captions, they cover21.9% more objects,18.8% more relations, and20.1% more attributes.These suggest that, across a large number of samples, different languages bias people and models to focus on different visual concepts.Finally, we show that models trained on image-text data in one language perform distinctly better on that language's test set.Our work points towards the potential value of training vision models on multilingual data sources to widen the range/variation of descriptive information those models are exposed to.

</details>

---

## 440. Hyperbolic Safety-Aware Vision-Language Models

- [ ] Hyperbolic Safety-Aware Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/35085

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35085

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Addressing the retrieval of unsafe content from vision-language models such as CLIP is an important step towards real-world integration. Current efforts have relied on unlearning techniques that try to erase the model’s knowledge of unsafe concepts. While effective in reducing unwanted outputs, unlearning limits the model's capacity to discern between safe and unsafe content. In this work, we introduce a novel approach that shifts from unlearning to an awareness paradigm by leveraging the inherent hierarchical properties of the hyperbolic space. We propose to encode safe and unsafe content as an entailment hierarchy, where both are placed in different regions of hyperbolic space. Our HySAC, Hyperbolic Safety-Aware CLIP, employs entailment loss functions to model the hierarchical and asymmetrical relations between safe and unsafe image-text pairs. This modelling – ineffective in standard vision-language models due to their reliance on Euclidean embeddings – endows the model with awareness of unsafe content, enabling it to serve as both a multimodal unsafe classifier and a flexible content retriever, with the option to dynamically redirect unsafe queries toward safer alternatives or retain the original output. Extensive experiments show that our approach not only enhances safety recognition, but also establishes a more adaptable and interpretable framework for content moderation in vision-language models.

</details>

---

## 441. DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation

- [ ] DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/35099

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35099

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Few-shot semantic segmentation (FSS) aims to enable models to segment novel/unseen object classes using only a limited number of labeled examples. However, current FSS methods frequently struggle with generalization due to incomplete and biased feature representations, especially when support images do not capture the full appearance variability of the target class. To improve the FSS pipeline, we propose a novel framework that utilizes large language models (LLMs) to adapt general class semantic information to the query image. Furthermore, the framework employs dense pixel-wise matching to identify similarities between query and support images, resulting in enhanced FSS performance. Inspired by reasoning-based segmentation frameworks, our method, named DSV-LFS, introduces an additional token into the LLM vocabulary, allowing a multimodal LLM to generate a "semantic prompt" from class descriptions. In parallel, a dense matching module identifies visual similarities between the query and support images, generating a "visual prompt". These prompts are then jointly employed to guide the prompt-based decoder for accurate segmentation of the query image. Comprehensive experiments on the benchmark datasets Pascal-5i and COCO-20i demonstrate that our framework achieves state-of-the-art performance-by a significant margin-demonstrating superior generalization to novel classes and robustness across diverse scenarios.

</details>

---

## 442. Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks

- [ ] Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks | https://cvpr.thecvf.com/virtual/2025/poster/35096

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35096

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We propose a new "Unbiased through Textual Description (UTD)" video benchmark based on unbiased subsets of existing video classification and retrieval datasets to enable a more robust assessment of video understanding capabilities.Namely, we tackle the problem that current video benchmarks may suffer from different representation biases, e.g., object bias or single-frame bias, where mere recognition of objects or utilization of only a single frame is sufficient for correct prediction. We leverage VLMs and LLMs to analyze and debias video benchmarks from such representation biases. Specifically, we generate frame-wise textual descriptions of videos, filter them for specific information (e.g. only objects) and leverage them to examine representation biases across three dimensions: 1) concept bias — determining if a specific concept (e.g., objects) alone suffice for prediction; 2) temporal bias — assessing if temporal information contributes to prediction; and 3) common sense vs. dataset bias —evaluating whether zero-shot reasoning or dataset correlations contribute to prediction. Since our new toolkit allows us to analyze representation biases at scale without additional human annotation, we conduct a systematic and comprehensive analysis of representation biases in 12 popular video classification and retrieval datasets and create new object-debiased test splits for these datasets. Moreover, we benchmark 33 state-of-the-art video models on original and debiased splits and analyze biases in the models. To facilitate the future development of more robust video understanding benchmarks and models, we release: "UTD-descriptions", a dataset with our rich structured descriptions for each dataset, and "UTD-splits", a dataset of object debiased test splits.

</details>

---

## 443. SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding

- [ ] SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding | https://cvpr.thecvf.com/virtual/2025/poster/35105

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35105

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the progress made by multimodal large language models (MLLMs) in computational pathology, they remain limited by a predominant focus on patch-level analysis, missing essential contextual information at the whole-slide level. The lack of large-scale instruction datasets and the gigapixel scale of whole slide images (WSIs) pose significant developmental challenges. In this paper, we present SlideChat, the first vision-language assistant capable of understanding gigapixel whole-slide images, exhibiting excellent multimodal conversational capability and response complex instruction across diverse pathology scenarios. To support its development, we created SlideInstruction, the largest instruction-following dataset for WSIs consisting of 4.2K WSI captions and 176K VQA pairs with multiple categories. Furthermore, we propose SlideBench, a multimodal benchmark that incorporates captioning and VQA tasks to assess SlideChat's capabilities in various settings such as microscopy, diagnosis and clinical. Compared to both general and specialized MLLMs, SlideChat exhibits exceptional  capabilities, achieving state-of-the-art performance on 18 of 22 tasks. For example, it achieved an overall accuracy of 81.17% on SlideBench-VQA (TCGA), and 54.15% on SlideBench-VQA (BCNB). We will fully release SlideChat, SlideInstruction and SlideBench as open-source resources to facilitate research and development in computational pathology.

</details>

---

## 444. Ges3ViG : Incorporating Pointing Gestures into Language-Based 3D Visual Grounding for Embodied Reference Understanding

- [ ] Ges3ViG : Incorporating Pointing Gestures into Language-Based 3D Visual Grounding for Embodied Reference Understanding | https://cvpr.thecvf.com/virtual/2025/poster/35108

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35108

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3-Dimensional Embodied Reference Understanding (3D-ERU) is a complex vision-language task, crucial for supporting interactions between humans and situated AI agents. 3D-ERU combines a language description and an accompanying pointing gesture to identify the most relevant target object in a given 3D scene. While previous research has extensively explored pure language-based 3D grounding, there has been limited exploration of 3D-ERU, which also incorporates human pointing gestures. To address this gap, we introduce a data augmentation framework-- **Imputer**, and use it to curate a new challenging benchmark dataset-- **ImputeRefer** for 3D-ERU, by incorporating human pointing gestures into existing 3D scene datasets that only contain language instructions. We also propose **Ges3ViG**, a novel model for 3D-ERU that achieves a $\sim$30\% improvement in accuracy, compared to other 3D-ERU models and $\sim$9\% compared to other purely language-based 3D grounding models. Our code and data will be released publicly upon acceptance of the paper.

</details>

---

## 445. Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models

- [ ] Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models | https://cvpr.thecvf.com/virtual/2025/poster/35117

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35117

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks, particularly targeted adversarial images that manipulate the model to generate harmful content specified by the adversary. Current attack methods rely on predefined target labels to create targeted adversarial attacks, which limits their scalability and applicability for large-scale robustness evaluations. In this paper, we propose AnyAttack , a self-supervised framework that generates targeted adversarial images for VLMs without label supervision, allowing any image to serve as a target for the attack .Our framework employs the "pre-training and fine-tuning" paradigm, with the adversarial noise generator pre-trained on the large-scale LAION-400M dataset.This large-scale pre-training endows our method with powerful transferability across a wide range of VLMs.Extensive experiments on five mainstream open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) across three multimodal tasks (image-text retrieval, multimodal classification, and image captioning) demonstrate the effectiveness of our attack.Additionally, we successfully transfer AnyAttack to multiple commercial VLMs, including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT.These results reveal an unprecedented risk to VLMs, highlighting the need for effective countermeasures.

</details>

---

## 446. Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding

- [ ] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding | https://cvpr.thecvf.com/virtual/2025/poster/35123

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35123

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have significantly improved performance in visual question answering. However, they often suffer from hallucinations. In this work, hallucinations are categorized into two main types: initial hallucinations and snowball hallucinations. We argue that adequate contextual information can be extracted directly from the token interaction process. Inspired by causal inference in decoding strategy, we propose to leverage causal masks to establish information propagation between multimodal tokens. The hypothesis is that insufficient interaction between those tokens may lead the model to rely on outlier tokens, overlooking dense and rich contextual cues. Therefore, we propose to intervene in the propagation process by tackling outlier tokens to enhance in-context inference. With this goal, we present FarSight, a versatile plug-and-play decoding strategy to reduce attention interference from outlier tokens merely by optimizing the causal mask. The heart of our method is effective token propagation. We design an attention register structure within the upper triangular matrix of the causal mask, dynamically allocating attention capture attention diverted to outlier tokens. Moreover, a positional awareness encoding method with a diminishing masking rate is proposed, allowing the model to attend to further preceding tokens, especially for video sequence tasks. With extensive experiments, FarSight demonstrates significant hallucination-mitigating performance across different MLLMs on both image and video benchmarks, proving its effectiveness.

</details>

---

## 447. SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories

- [ ] SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories | https://cvpr.thecvf.com/virtual/2025/poster/35138

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35138

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While MLLMs have demonstrated adequate image understanding capabilities, they still struggle with pixel-level comprehension, limiting their practical applications. Current evaluation tasks like VQA and visual grounding remain too coarse to assess fine-grained pixel comprehension accurately. Though segmentation is foundational for pixel-level understanding, existing methods often require MLLMs to generate implicit tokens, decoded through external pixel decoders. This approach disrupts the MLLM’s text output space, potentially compromising language capabilities and reducing flexibility and extensibility, while failing to reflect the model’s intrinsic pixel-level understanding.Thus, We introduce the Human-Like Mask Annotation Task (HLMAT), a new paradigm where MLLMs mimic human annotators using interactive segmentation tools. Modeling segmentation as a multi-step Markov Decision Process, HLMAT enables MLLMs to iteratively generate text-based click points, achieving high-quality masks without architectural changes or implicit tokens. Through this setup, we develop SegAgent, a model fine-tuned on human-like annotation trajectories, which achieves performance comparable to SOTA methods and supports additional tasks like mask refinement and annotation filtering.HLMAT provides a protocol for assessing fine-grained pixel understanding in MLLMs and introduces a vision-centric, multi-step decision-making task that facilitates exploration of MLLMs’ visual reasoning abilities. Our adaptations of policy improvement method StaR and PRM guided tree search further enhance model robustness in complex segmentation tasks, laying a foundation for future advancements in fine-grained visual perception and multi-step decision-making for MLLMs.

</details>

---

## 448. SemiDAViL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation

- [ ] SemiDAViL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/35137

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35137

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Domain Adaptation (DA) and Semi-supervised Learning (SSL) converge in Semi-supervised Domain Adaptation (SSDA), where the objective is to transfer knowledge from a source domain to a target domain using a combination of limited labeled target samples and abundant unlabeled target data. Although intuitive, a simple amalgamation of DA and SSL is suboptimal in semantic segmentation due to two major reasons: (1) previous methods, while able to learn good segmentation boundaries, are prone to confuse classes with similar visual appearance due to limited supervision; and (2) skewed and imbalanced training data distribution preferring source representation learning whereas impeding from exploring limited information about tailed classes. Language guidance can serve as a pivotal semantic bridge, facilitating robust class discrimination and mitigating visual ambiguities by leveraging the rich semantic relationships encoded in pre-trained language models to enhance feature representations across domains. Therefore, we propose the first language-guided SSDA setting for semantic segmentation in this work. Specifically, we harness the semantic generalization capabilities inherent in vision-language models (VLMs) to establish a synergistic framework within the SSDA paradigm. To address the inherent class-imbalance challenges in long-tailed distributions, we introduce class-balanced segmentation loss formulations that effectively regularize the learning process. Through extensive experimentation across diverse domain adaptation scenarios, our approach demonstrates substantial performance improvements over contemporary state-of-the-art (SoTA) methodologies. Code will be released.

</details>

---

## 449. MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output

- [ ] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output | https://cvpr.thecvf.com/virtual/2025/poster/35156

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35156

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Currently, medical vision language models are widely used in medical vision question answering tasks. However, existing models are confronted with two issues: for input, the model only relies on text instructions and lacks direct understanding of visual clues in the image; for output, the model only gives text answers and lacks connection with key areas in the image. To address these issues, we propose a unified medical vision language model MIMO, with visual referring Multimodal Input and pixel grounding Multimodal Output. MIMO can not only combine visual clues and textual instructions to understand complex medical images and semantics, but can also ground medical terminologies in textual output within the image. To overcome the scarcity of relevant data in the medical field, we propose MIMOSeg, a comprehensive medical multimodal dataset including 895K samples. MIMOSeg is constructed from four different perspectives, covering basic instruction following and complex question answering with multimodal input and multimodal output. We conduct experiments on several downstream medical multimodal tasks. Extensive experimental results verify that MIMO can uniquely combine visual referring and pixel grounding capabilities, which are not available in previous models.

</details>

---

## 450. Exploring Visual Vulnerabilities via Multi-Loss Adversarial Search for Jailbreaking Vision-Language Models

- [ ] Exploring Visual Vulnerabilities via Multi-Loss Adversarial Search for Jailbreaking Vision-Language Models | https://cvpr.thecvf.com/virtual/2025/poster/35162

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35162

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite inheriting security measures from underlying language models, Vision-Language Models (VLMs) may still be vulnerable to safety alignment issues. Through empirical analysis, we uncover two critical findings: scenario-matched images can significantly amplify harmful outputs, and contrary to common assumptions in gradient-based attacks, minimal loss values do not guarantee optimal attack effectiveness. Building on these insights, we introduce MLAI (Multi-Loss Adversarial Images), a novel jailbreak framework that leverages scenario-aware image generation for semantic alignment, exploits flat minima theory for robust adversarial image selection, and employs multi-image collaborative attacks for enhanced effectiveness. Extensive experiments demonstrate MLAI's significant impact, achieving attack success rates of 77.75\% on MiniGPT-4 and 82.80\% on LLaVA-2, substantially outperforming existing methods by margins of 34.37\% and 12.77\% respectively. Furthermore, MLAI shows considerable transferability to commercial black-box VLMs, achieving up to 60.11\% success rate. Our work reveals fundamental visual vulnerabilities in current VLMs safety mechanisms and underscores the need for stronger defenses. Warning: This paper contains potentially harmful example text.

</details>

---

## 451. SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment

- [ ] SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment | https://cvpr.thecvf.com/virtual/2025/poster/35169

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35169

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Integrating large language models (LLMs) into autonomous driving has attracted significant attention with the hope of improving generalization and explainability. However, existing methods often focus on either driving or vision-language understanding but achieving both high driving performance and extensive language understanding remains challenging. In addition, the dominant approach to tackle vision-language understanding is using visual question answering. However, for autonomous driving, this is only useful if it is grounded in the action space. Otherwise, the model’s answers could be inconsistent with its behavior. Therefore, we propose a model that can handle three different tasks: (1) closed-loop driving, (2) vision-language understanding, and (3) language-action alignment. Our model SimLingo is based on a vision language model (VLM) and works using only camera, excluding expensive sensors like LiDAR. SimLingo obtains state-of-the-art performance on the widely used CARLA simulator on the Leaderboard 2.0 and the Bench2Drive benchmarks. Additionally, we achieve strong results in a wide variety of language-related tasks while maintaining high driving performance. We will release code, data and models upon acceptance.

</details>

---

## 452. Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels

- [ ] Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels | https://cvpr.thecvf.com/virtual/2025/poster/35179

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35179

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present a simple yet effective workflow for automatically scaling instruction-following data to elicit the pixel-level grounding capabilities of VLMs under complex instructions. We address five critical real-world challenges: hallucination, multi-object scenarios, reasoning, multi-granularity, and part-level reference. By distilling visual-language knowledge from a teacher model, our workflow generates instruction-response pairs that link with existing, abundant pixel-level annotations of the images, minimizing the need for human annotation. We refer to the resulting dataset as Ground-V, which captures extensive object localization knowledge and nuanced pixel-level referring expressions. Experimental results show that models of various architectures trained on Ground-V exhibit substantial improvements across diverse grounding tasks. Specifically, incorporating Ground-V during training directly achieve an average accuracy boost of 4.4% for LISA and a 7.9% for PSALM across six benchmarks on the gIoU metric. It also sets new state-of-the-art results on standard benchmarks such as RefCOCO/+/g. Notably, on gRefCOCO, we achieve an N-Acc of 83.3%, exceeding the previous state-of-the-art by more than 20%.

</details>

---

## 453. EchoTraffic: Enhancing Traffic Anomaly Understanding with Audio-Visual Insights

- [ ] EchoTraffic: Enhancing Traffic Anomaly Understanding with Audio-Visual Insights | https://cvpr.thecvf.com/virtual/2025/poster/35198

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35198

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traffic Anomaly Understanding (TAU) is essential for improving public safety and transportation efficiency by enabling timely detection and response to incidents. Beyond existing methods, which rely largely on visual data, we propose to consider audio cues, a valuable source that offers strong hints to anomaly scenarios such as crashes and honking. Our contributions are twofold. First, we compile AV-TAU, the first large-scale audio-visual dataset for TAU, providing 29,865 traffic anomaly videos and 149,325 Q&A pairs, while supporting five essential TAU tasks. Second, we develop EchoTraffic, a multimodal LLM that integrates audio and visual data for TAU, through our audio-insight frame selector and dynamic connector to effectively extract crucial audio cues for anomaly understanding with a two-phase training framework. Experimental results on AV-TAU manifest that EchoTraffic sets a new SOTA performance in TAU, outperforming the existing multimodal LLMs. Our contributions, including AV-TAU and EchoTraffic, pave a new direction for multimodal TAU. Our dataset and code will be publicly available upon publication of this work.

</details>

---

## 454. Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation

- [ ] Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation | https://cvpr.thecvf.com/virtual/2025/poster/35203

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35203

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the unprecedented progress in the field of 3D generation, current systems still often fail to produce high-quality 3D assets that are visually appealing and geometrically and semantically consistent across multiple viewpoints. To effectively assess the quality of the generated 3D data, there is a need for a reliable 3D evaluation tool. Unfortunately, existing 3D evaluation metrics often overlook the geometric quality of generated assets or merely rely on black-box multimodal large language models for coarse assessment. In this paper, we introduce Eval3D, a fine-grained, interpretable evaluation tool that can faithfully evaluate the quality of generated 3D assets based on various distinct yet complementary criteria. Our key observation is that many desired properties of 3D generation, such as semantic and geometric consistency, can be effectively captured by measuring the consistency among various foundation models and tools. We thus leverage a diverse set of models and tools as probes to evaluate the inconsistency of generated 3D assets across different aspects. Compared to prior work, Eval3D provides pixel-wise measurement, enables accurate 3D spatial feedback, and aligns more closely with human judgments. We comprehensively evaluate existing 3D generation models using Eval3D and highlight the limitations and challenges of current models.

</details>

---

## 455. IDEA-Bench: How Far are Generative Models from Professional Designing?

- [ ] IDEA-Bench: How Far are Generative Models from Professional Designing? | https://cvpr.thecvf.com/virtual/2025/poster/35207

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35207

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in image generation models enable the creation of high-quality images and targeted modifications based on textual instructions. Some models even support multimodal complex guidance and demonstrate robust task generalization capabilities. However, they still fall short of meeting the nuanced, professional demands of designers. To bridge this gap, we introduce IDEA-Bench, a comprehensive benchmark designed to advance image generation models toward applications with robust task generalization. IDEA-Bench comprises 97 professional image generation tasks and 266 specific cases, categorized into five major types based on the current capabilities of existing models. Furthermore, we provide a representative subset of 18 tasks with enhanced evaluation criteria to facilitate more nuanced and reliable evaluations using Multimodal Large Language Models (MLLMs). By assessing models' ability to comprehend and execute novel, complex tasks, IDEA-Bench paves the way toward the development of generative models with autonomous and versatile visual generation capabilities.

</details>

---

## 456. PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability

- [ ] PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability | https://cvpr.thecvf.com/virtual/2025/poster/35212

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35212

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding the environment and a robot's physical reachability is crucial for task execution. While state-of-the-art vision-language models (VLMs) excel in environmental perception, they often generate inaccurate or impractical responses in embodied visual reasoning tasks due to a lack of understanding of robotic physical reachability. To address this issue, we propose a unified representation of physical reachability across diverse robots, i.e., Space-Physical Reachability Map (S-P Map), and PhysVLM, a vision-language model that integrates this reachability information into visual reasoning. Specifically, the S-P Map abstracts a robot's physical reachability into a generalized spatial representation, independent of specific robot configurations, allowing the model to focus on reachability features rather than robot-specific parameters. Subsequently, PhysVLM extends traditional VLM architectures by incorporating an additional feature encoder to process the S-P Map, enabling the model to reason about physical reachability without compromising its general vision-language capabilities. To train and evaluate PhysVLM, we constructed a large-scale multi-robot dataset, Phys100K, and a challenging benchmark, EQA-phys, which includes tasks for six different robots in both simulated and real-world environments. Experimental results demonstrate that PhysVLM outperforms existing models, achieving a 14\% improvement over GPT-4o on EQA-phys and surpassing advanced embodied VLMs such as RoboMamba and SpatialVLM on the RoboVQA-val and OpenEQA benchmarks. Additionally, the S-P Map shows strong compatibility with various VLMs, and its integration into GPT-4o-mini yields a 7.1\% performance improvement.

</details>

---

## 457. GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill

- [ ] GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill | https://cvpr.thecvf.com/virtual/2025/poster/35222

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35222

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning open-vocabulary physical skills for simulated agents remains challenging due to the limitations of reinforcement learning approaches: manually designed rewards lack scalability, while demonstration-based methods struggle to cover arbitrary tasks. We propose GROVE, a generalized reward framework for open-vocabulary physical skill learning without manual reward design or task-specific demonstrations. GROVE uniquely combines Large Language Models (LLMs) for generating precise constraints with Vision Language Models (VLMs) for semantic evaluation. Through an iterative reward design process, VLM-based feedback guides the refinement of LLM-generated constraints, significantly enhancing the reliability of our method. Central to our approach is Pose2CLIP, a lightweight pose-to-semantic feature mapper that significantly enhances the quality and efficiency of VLM evaluation. Extensive experiments demonstrate GROVE's versatility across diverse tasks and learning paradigms. Our approach achieves 22.2% higher naturalness and 25.7% better task completion score while training 8.4 times faster than previous open-vocabulary methods, establishing a new foundation for scalable physical skill acquisition.

</details>

---

## 458. Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation

- [ ] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation | https://cvpr.thecvf.com/virtual/2025/poster/35217

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35217

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent open-vocabulary segmentation methods adopt mask generators to predict segmentation masks and leverage pre-trained vision-language models, e.g. , CLIP, to classify these masks via mask pooling.Although these approaches show promising results, it is counterintuitive that accurate masks often fail to yield accurate classification results through pooling CLIP image embeddings within the mask regions.In this paper, we reveal the performance limitations of mask pooling and introduce Mask-Adapter , a simple yet effective method to address these challenges in open-vocabulary segmentation.Compared to directly using proposal masks, our proposed Mask-Adapter extracts semantic activation maps from proposal masks, providing richer contextual information and ensuring alignment between masks and CLIP.Additionally, we propose a mask consistency loss that encourages proposal masks with similar IoUs to obtain similar CLIP embeddings to enhance models' robustness to varying predicted masks.Mask-Adapter integrates seamlessly into open-vocabulary segmentation methods based on mask pooling in a plug-and-play manner, delivering more accurate classification results. Extensive experiments across several zero-shot benchmarks demonstrate significant performance gains for the proposed Mask-Adapter on several well-established methods.Notably, Mask-Adapter also extends effectively to SAM and achieves impressive results on several open-vocabulary segmentation datasets. Code and models will be made publicly available.

</details>

---

## 459. Dual Diffusion for Unified Image Generation and Understanding

- [ ] Dual Diffusion for Unified Image Generation and Understanding | https://cvpr.thecvf.com/virtual/2025/poster/35223

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35223

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models have gained tremendous success in text-to-image generation, yet still lag behind with visual understanding tasks, an area dominated by autoregressive vision-language models.We propose a large-scale and fully end-to-end diffusion model for multi-modal understanding and generation that significantly improves on existing diffusion-based multimodal models, and is the first of its kind to support the full suite of vision-language modeling capabilities.Inspired by the multimodal diffusion transformer (MM-DiT) and recent advances in discrete diffusion language modeling, we leverage a cross-modal maximum likelihood estimation framework that simultaneously trains the conditional likelihoods of both images and text jointly under a single loss function, which is back-propagated through both branches of the diffusion transformer. The resulting model is highly flexible and capable of a wide range of tasks including image generation, captioning, and visual question answering. Our model attained competitive performance compared to recent unified image understanding and generation models, demonstrating the potential of multimodal diffusion modeling as a promising alternative to autoregressive next-token prediction models.

</details>

---

## 460. LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant

- [ ] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant | https://cvpr.thecvf.com/virtual/2025/poster/35227

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35227

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of multimodal information retrieval, increasingly complex retrieval tasks have emerged. Existing methods predominately rely on task-specific fine-tuning of vision-language models, often those trained with image-text contrastive learning. In this paper, we explore the possibility of re-purposing generative Large Multimodal Models (LMMs) for retrieval. This approach enables to unify of all retrieval tasks under the same formulation and, more importantly, allows for extrapolation towards unseen retrieval tasks without additional training. Our contributions can be summarised in the following aspects: (i) We introduce LamRA , a versatile framework designed to empower LMMs with sophisticated retrieval and reranking capabilities. (ii) For retrieval, we adopt a two-stage training strategy comprising language-only pre-training and multimodal instruction tuning to progressively enhance LMM's retrieval performance. (iii) For reranking, we employ joint training for both pointwise and listwise reranking, offering two distinct ways to further boost the retrieval performance. (iv) Extensive experimental results underscore the efficacy of our method in handling more than ten retrieval tasks, demonstrating robust performance in both supervised and zero-shot settings, including scenarios involving previously unseen retrieval tasks. The code and model will be made publicly available for reproduction.

</details>

---

## 461. Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization

- [ ] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization | https://cvpr.thecvf.com/virtual/2025/poster/35238

- **Link**: https://cvpr.thecvf.com/virtual/2025/poster/35238

- **Conference**: CVPR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) excel in various tasks, yet often struggle with modality bias, tending to rely heavily on a single modality or prior knowledge when generating responses. In this paper, we propose a debiased preference optimization dataset, RLAIF-V-Bias, and introduce a Noise-Aware Preference Optimization (NAPO) algorithm. Specifically, we first construct the dataset by introducing perturbations to reduce the informational content of certain modalities, prompting the model to overly rely on a specific modality when generating responses. To address the inevitable noise in automatically constructed data, we combine the noise-robust Mean Absolute Error (MAE) with the Binary Cross-Entropy (BCE) in Direct Preference Optimization (DPO) using a negative Box-Cox transformation and dynamically adjust the algorithm’s noise robustness based on the evaluated noise levels in the data.Extensive experiments validate our approach, demonstrating not only its effectiveness in mitigating modality bias but also its significant role in minimizing hallucinations.

</details>

---

