# NeurIPS 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_neurips2023_papers.csv

## 1. PERIA: Perceive, Reason, Imagine, Act via Holistic Language and Vision Planning for Manipulation

- [ ] PERIA: Perceive, Reason, Imagine, Act via Holistic Language and Vision Planning for Manipulation | https://neurips.cc/virtual/2024/poster/92925

- **Link**: https://neurips.cc/virtual/2024/poster/92925

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Long-horizon manipulation tasks with general instructions often implicitly encapsulate multiple sub-tasks, posing significant challenges in instruction following.While language planning is a common approach to decompose general instructions into stepwise sub-instructions, text-only guidance may lack expressiveness and lead to potential ambiguity. Considering that humans often imagine and visualize sub-instructions reasoning out before acting, the imagined subgoal images can provide more intuitive guidance and enhance the reliability of decomposition. Inspired by this, we propose PERIA ( PE rceive, R eason, I magine, A ct), a novel framework that integrates holistic language planning and vision planning for long-horizon manipulation tasks with complex instructions, leveraging both logical and intuitive aspects of task decomposition.Specifically, we first perform a lightweight multimodal alignment on the encoding side to empower the MLLM to perceive visual details and language instructions. The MLLM is then jointly instruction-tuned with a pretrained image-editing model to unlock capabilities of simultaneous reasoning of language instructions and generation of imagined subgoals. Furthermore, we introduce a consistency alignment loss to encourage coherent subgoal images and align with their corresponding instructions, mitigating potential hallucinations and semantic conflicts between the two planning manners.Comprehensive evaluations across three task domains demonstrate that PERIA, benefiting from holistic language and vision planning, significantly outperforms competitive baselines in both instruction following accuracy and task success rate on complex manipulation tasks.

</details>

---

## 2. Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation

- [ ] Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation | https://neurips.cc/virtual/2024/poster/92928

- **Link**: https://neurips.cc/virtual/2024/poster/92928

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Surgical video-language pretraining (VLP) faces unique challenges due to the knowledge domain gap and the scarcity of multi-modal data. This study aims to bridge the gap by addressing issues regarding textual information loss in surgical lecture videos and the spatial-temporal challenges of surgical VLP. To tackle these issues, we propose a hierarchical knowledge augmentation approach and a novel Procedure-Encoded Surgical Knowledge-Augmented Video-Language Pretraining (PeskaVLP) framework. The proposed knowledge augmentation approach uses large language models (LLM) to refine and enrich surgical concepts, thus providing comprehensive language supervision and reducing the risk of overfitting. The PeskaVLP framework combines language supervision with visual self-supervision, constructing hard negative samples and employing a Dynamic Time Warping (DTW) based loss function to effectively comprehend the cross-modal procedural alignment. Extensive experiments on multiple public surgical scene understanding and cross-modal retrieval datasets show that our proposed method significantly improves zero-shot transferring performance and offers a generalist visual repre- sentation for further advancements in surgical scene understanding. The source code will be available at https://github.com/CAMMA-public/PeskaVLP.

</details>

---

## 3. G2D: From Global to Dense Radiography Representation Learning via Vision-Language Pre-training

- [ ] G2D: From Global to Dense Radiography Representation Learning via Vision-Language Pre-training | https://neurips.cc/virtual/2024/poster/92931

- **Link**: https://neurips.cc/virtual/2024/poster/92931

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Medical imaging tasks require an understanding of subtle and localized visual features due to the inherently detailed and area-specific nature of pathological patterns, which are crucial for clinical diagnosis. Although recent advances in medical vision-language pre-training (VLP) enable models to learn clinically relevant visual features by leveraging both medical images and their associated radiology reports, current medical VLP methods primarily focus on aligning images with entire reports. This focus hinders the learning of dense (pixel-level) visual features and is suboptimal for dense prediction tasks (e.g., medical image segmentation).To address this challenge, we propose a novel medical VLP framework, named Global to Dense level representation learning (G2D) , which aims to learn global and dense visual features simultaneously using only image-text pairs without extra annotations. In particular, G2D designs a Pseudo Segmentation (PS) task, which enables the model to learn dense visual features during VLP. Notably, generating PS masks can be performed on the fly during VLP, which does not incur extra trainable parameters. With this simple yet effective idea, G2D achieves superior performance across 5 medical imaging tasks and 25 diseases. Particularly, in the segmentation task which requires dense visual features, G2D surpasses existing models even with just 1% of the training data for finetuning, compared to 100% used by other models . The code can be found in https://github.com/cheliu-computation/G2D-NeurIPS24/tree/main.

</details>

---

## 4. TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning

- [ ] TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning | https://neurips.cc/virtual/2024/poster/92938

- **Link**: https://neurips.cc/virtual/2024/poster/92938

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) such as CLIP have shown excellent performance for zero-shot classification. Based on CLIP, recent methods design various learnable prompts to evaluate the zero-shot generalization capability on a base-to-novel setting. This setting assumes test samples are already divided into either base or novel classes, limiting its application to realistic scenarios. In this paper, we focus on a more challenging and practical setting: generalized zero-shot learning (GZSL), i.e., testing with no information about the base/novel division. To address this challenging zero-shot problem, we introduce two unique designs that enable us to classify an image without the need of knowing whether it comes from seen or unseen classes. Firstly, most existing methods only adopt a single latent space to align visual and linguistic features, which has a limited ability to represent complex visual-linguistic patterns, especially for fine-grained tasks. Instead, we propose a dual-space feature alignment module that effectively augments the latent space with a novel attribute space induced by a well-devised attribute reservoir. In particular, the attribute reservoir consists of a static vocabulary and learnable tokens complementing each other for flexible control over feature granularity. Secondly, finetuning CLIP models (e.g., prompt learning) on seen base classes usually sacrifices the model's original generalization capability on unseen novel classes. To mitigate this issue, we present a new topology-preserving objective that can enforce feature topology structures of the combined base and novel classes to resemble the topology of CLIP. In this manner, our model will inherit the generalization ability of CLIP through maintaining the pairwise class angles in the attribute space. Extensive experiments on twelve object recognition datasets demonstrate that our model, termed Topology-Preserving Reservoir (TPR), outperforms strong baselines including both prompt learning and conventional generative-based zero-shot methods.

</details>

---

## 5. GenRL: Multimodal-foundation world models for generalization in embodied agents

- [ ] GenRL: Multimodal-foundation world models for generalization in embodied agents | https://neurips.cc/virtual/2024/poster/92947

- **Link**: https://neurips.cc/virtual/2024/poster/92947

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Learning generalist embodied agents, able to solve multitudes of tasks in different domains is a long-standing problem. Reinforcement learning (RL) is hard to scale up as it requires a complex reward design for each task. In contrast, language can specify tasks in a more natural way. Current foundation vision-language models (VLMs) generally require fine-tuning or other adaptations to be adopted in embodied contexts, due to the significant domain gap. However, the lack of multimodal data in such domains represents an obstacle to developing foundation models for embodied applications. In this work, we overcome these problems by presenting multimodal-foundation world models, able to connect and align the representation of foundation VLMs with the latent space of generative world models for RL, without any language annotations. The resulting agent learning framework, GenRL, allows one to specify tasks through vision and/or language prompts, ground them in the embodied domain’s dynamics, and learn the corresponding behaviors in imagination.As assessed through large-scale multi-task benchmarking in locomotion and manipulation domains, GenRL enables multi-task generalization from language and visual prompts. Furthermore, by introducing a data-free policy learning strategy, our approach lays the groundwork for foundational policy learning using generative world models. Website, code and data: https://mazpie.github.io/genrl/

</details>

---

## 6. InstructG2I: Synthesizing Images from Multimodal Attributed Graphs

- [ ] InstructG2I: Synthesizing Images from Multimodal Attributed Graphs | https://neurips.cc/virtual/2024/poster/92951

- **Link**: https://neurips.cc/virtual/2024/poster/92951

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we approach an overlooked yet critical task Graph2Image: generating images from multimodal attributed graphs (MMAGs). This task poses significant challenges due to the explosion in graph size, dependencies among graph entities, and the need for controllability in graph conditions. To address these challenges, we propose a graph context-conditioned diffusion model called InstructG2I. InstructG2I first exploits the graph structure and multimodal information to conduct informative neighbor sampling by combining personalized page rank and re-ranking based on vision-language features. Then, a graph QFormer encoder adaptively encodes the graph nodes into an auxiliary set of graph prompts to guide the denoising process of diffusion. Finally, we propose graph classifier-free guidance, enabling controllable generation by varying the strength of graph guidance and multiple connected edges to a node. Extensive experiments conducted on three datasets from different domains demonstrate the effectiveness and controllability of our approach. The code is available at https://github.com/PeterGriffinJin/InstructG2I.

</details>

---

## 7. XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation

- [ ] XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation | https://neurips.cc/virtual/2024/poster/92979

- **Link**: https://neurips.cc/virtual/2024/poster/92979

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing methodologies in open vocabulary 3D semantic segmentation primarily concentrate on establishing a unified feature space encompassing 3D, 2D, and textual modalities. Nevertheless, traditional techniques such as global feature alignment or vision-language model distillation tend to impose only approximate correspondence, struggling notably with delineating fine-grained segmentation boundaries. To address this gap, we propose a more meticulous mask-level alignment between 3D features and the 2D-text embedding space through a cross-modal mask reasoning framework, XMask3D. In our approach, we developed a mask generator based on the denoising UNet from a pre-trained diffusion model, leveraging its capability for precise textual control over dense pixel representations and enhancing the open-world adaptability of the generated masks. We further integrate 3D global features as implicit conditions into the pre-trained 2D denoising UNet, enabling the generation of segmentation masks with additional 3D geometry awareness. Subsequently, the generated 2D masks are employed to align mask-level 3D representations with the vision-language feature space, thereby augmenting the open vocabulary capability of 3D geometry embeddings. Finally, we fuse complementary 2D and 3D mask features, resulting in competitive performance across multiple benchmarks for 3D open vocabulary semantic segmentation. Code is available at https://github.com/wangzy22/XMask3D.

</details>

---

## 8. Toward a Stable, Fair, and Comprehensive Evaluation of Object Hallucination in Large Vision-Language Models

- [ ] Toward a Stable, Fair, and Comprehensive Evaluation of Object Hallucination in Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/93023

- **Link**: https://neurips.cc/virtual/2024/poster/93023

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Given different instructions, large vision-language models (LVLMs) exhibit different degrees of object hallucinations, posing a significant challenge to the evaluation of object hallucinations. Overcoming this challenge, existing object hallucination evaluation methods average the results obtained from a set of instructions. However, these methods fail to provide consistent evaluation across instruction sets that generate image descriptions of significantly different lengths. In this paper, we present the first systematic investigation of the effect of instructions on object hallucinations in LVLMs, with a specific focus on the role played by image description lengths. A valuable finding is that instructions indirectly affect hallucinations through the length of image descriptions. The longer the image description, the higher the object hallucination degree. Accordingly, we fit an informative length-hallucination curve, upon which a fine-grained evaluation framework named LeHaCE is introduced for evaluating object hallucinations at any given image description length. LeHaCE evaluates the object hallucination degree at a uniform image description length to mitigate the effect of description lengths, promoting stability and fairness. Moreover, LeHaCE incorporates the curve slope as an innovative hallucination evaluation metric, reflecting the extent to which the object hallucination degree is affected by the image description length, achieving a more comprehensive evaluation. Experimental results demonstrate that LeHaCE provides a more stable, fair, and comprehensive evaluation of object hallucinations in LVLMs compared to existing methods.

</details>

---

## 9. Scene Graph Generation with Role-Playing Large Language Models

- [ ] Scene Graph Generation with Role-Playing Large Language Models | https://neurips.cc/virtual/2024/poster/93060

- **Link**: https://neurips.cc/virtual/2024/poster/93060

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current approaches for open-vocabulary scene graph generation (OVSGG) use vision-language models such as CLIP and follow a standard zero-shot pipeline – computing similarity between the query image and the text embeddings for each category (i.e., text classifiers). In this work, we argue that the text classifiers adopted by existing OVSGG methods, i.e., category-/part-level prompts, are scene-agnostic as they remain unchanged across contexts. Using such fixed text classifiers not only struggles to model visual relations with high variance, but also falls short in adapting to distinct contexts. To plug these intrinsic shortcomings, we devise SDSGG, a scene-specific description based OVSGG framework where the weights of text classifiers are adaptively adjusted according to the visual content. In particular, to generate comprehensive and diverse descriptions oriented to the scene, an LLM is asked to play different roles (e.g., biologist and engineer) to analyze and discuss the descriptive features of a given scene from different views. Unlike previous efforts simply treating the generated descriptions as mutually equivalent text classifiers, SDSGG is equipped with an advanced renormalization mechanism to adjust the influence of each text classifier based on its relevance to the presented scene (this is what the term “specific” means). Furthermore, to capture the complicated interplay between subjects and objects, we propose a new lightweight module called mutual visual adapter. It refines CLIP’s ability to recognize relations by learning an interaction-aware semantic space. Extensive experiments on prevalent benchmarks show that SDSGG significantly outperforms top-leading methods.

</details>

---

## 10. RestoreAgent: Autonomous Image Restoration Agent via Multimodal Large Language Models

- [ ] RestoreAgent: Autonomous Image Restoration Agent via Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/93068

- **Link**: https://neurips.cc/virtual/2024/poster/93068

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Natural images captured by mobile devices often suffer from multiple types of degradation, such as noise, blur, and low light. Traditional image restoration methods require manual selection of specific tasks, algorithms, and execution sequences, which is time-consuming and may yield suboptimal results. All-in-one models, though capable of handling multiple tasks, typically support only a limited range and often produce overly smooth, low-fidelity outcomes due to their broad data distribution fitting. To address these challenges, we first define a new pipeline for restoring images with multiple degradations, and then introduce RestoreAgent, an intelligent image restoration system leveraging multimodal large language models. RestoreAgent autonomously assesses the type and extent of degradation in input images and performs restoration through (1) determining the appropriate restoration tasks, (2) optimizing the task sequence, (3) selecting the most suitable models, and (4) executing the restoration. Experimental results demonstrate the superior performance of RestoreAgent in handling complex degradation, surpassing human experts. Furthermore, the system’s modular design facilitates the fast integration of new tasks and models.

</details>

---

## 11. Automated Multi-level Preference for MLLMs

- [ ] Automated Multi-level Preference for MLLMs | https://neurips.cc/virtual/2024/poster/93122

- **Link**: https://neurips.cc/virtual/2024/poster/93122

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current multimodal Large Language Models (MLLMs) suffer from ''hallucination'', occasionally generating responses that are not grounded in the input images. To tackle this challenge, one promising path is to utilize reinforcement learning from human feedback (RLHF), which steers MLLMs towards learning superior responses while avoiding inferior ones. We rethink the common practice of using binary preferences ( i.e. , superior, inferior), and find that adopting multi-level preferences ( e.g. , superior, medium, inferior) is better for two benefits: 1) It narrows the gap between adjacent levels, thereby encouraging MLLMs to discern subtle differences. 2) It further integrates cross-level comparisons (beyond adjacent-level comparisons), thus providing a broader range of comparisons with hallucination examples. To verify our viewpoint, we present the Automated Multi-level Preference ( AMP ) framework for MLLMs. To facilitate this framework, we first develop an automated dataset generation pipeline that provides high-quality multi-level preference datasets without any human annotators. Furthermore, we design the Multi-level Direct Preference Optimization (MDPO) algorithm to robustly conduct complex multi-level preference learning. Additionally, we propose a new hallucination benchmark, MRHal-Bench. Extensive experiments across public hallucination and general benchmarks, as well as our MRHal-Bench, demonstrate the effectiveness of our proposed method. Code is available at https://github.com/takomc/amp.

</details>

---

## 12. Efficient LLM Scheduling by Learning to Rank

- [ ] Efficient LLM Scheduling by Learning to Rank | https://neurips.cc/virtual/2024/poster/93127

- **Link**: https://neurips.cc/virtual/2024/poster/93127

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In Large Language Model (LLM) inference, the output length of an LLM request is typically regarded as not known a priori. Consequently, most LLM serving systems employ a simple First-come-first-serve (FCFS) scheduling strategy, leading to Head-Of-Line (HOL) blocking and reduced throughput and service quality. In this paper, we reexamine this assumption -- we show that, although predicting the exact generation length of each request is infeasible, it is possible to predict the relative ranks of output lengths in a batch of requests, using learning to rank. The ranking information offers valuable guidance for scheduling requests. Building on this insight, we develop a novel scheduler for LLM inference and serving that can approximate the shortest-job-first (SJF) schedule better than existing approaches. We integrate this scheduler with the state-of-the-art LLM serving system and show significant performance improvement in several important applications: 2.8x lower latency in chatbot serving and 6.5x higher throughput in synthetic data generation. Our code is available at https://github.com/hao-ai-lab/vllm-ltr.git

</details>

---

## 13. A Sober Look at the Robustness of CLIPs to Spurious Features

- [ ] A Sober Look at the Robustness of CLIPs to Spurious Features | https://neurips.cc/virtual/2024/poster/93146

- **Link**: https://neurips.cc/virtual/2024/poster/93146

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision language models, such as CLIP, demonstrate impressive robustness to spurious features than single-modal models trained on ImageNet. However, existing test datasets are typically curated based on ImageNet-trained models, which aim to capture the spurious features inherited in ImageNet. Benchmarking CLIP models based on the ImageNet-oriented spurious features may not be sufficient to reflect the extent to which CLIP models are robust to spurious correlations within CLIP training data, e.g., LAION. To this end, we craft a new challenging dataset named CounterAnimal designed to reveal the reliance of CLIP models on realistic spurious features. Specifically, we split animal photos into groups according to the backgrounds, and then identify a pair of groups for each class where a CLIP model shows high-performance drops across the two groups. Our evaluations show that the spurious features captured by CounterAnimal are generically learned by CLIP models with different backbones and pre-train data, yet have limited influence for ImageNet models. We provide theoretical insights that the CLIP objective cannot offer additional robustness. Furthermore, we also re-evaluate strategies such as scaling up parameters and high-quality pre-trained data. We find that they still help mitigate the spurious features, providing a promising path for future developments.

</details>

---

## 14. Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection

- [ ] Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection | https://neurips.cc/virtual/2024/poster/93172

- **Link**: https://neurips.cc/virtual/2024/poster/93172

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection is crucial for deploying reliable machine learning models in open-world applications. Recent advances in CLIP-based OOD detection have shown promising results via regularizing prompt tuning with OOD features extracted from ID data. However, the irrelevant context mined from ID data can be spurious due to the inaccurate foreground-background decomposition, thus limiting the OOD detection performance. In this work, we propose a novel framework, namely, \textit{Self-Calibrated Tuning (SCT)}, to mitigate this problem for effective OOD detection with only the given few-shot ID data. Specifically, SCT introduces modulating factors respectively on the two components of the original learning objective. It adaptively directs the optimization process between the two tasks during training on data with different prediction uncertainty to calibrate the influence of OOD regularization, which is compatible with many prompt tuning based OOD detection methods. Extensive experiments and analyses have been conducted to characterize and demonstrate the effectiveness of the proposed SCT. The code is publicly available at: https://github.com/tmlr-group/SCT.

</details>

---

## 15. Unified Generative and Discriminative Training for Multi-modal Large Language Models

- [ ] Unified Generative and Discriminative Training for Multi-modal Large Language Models | https://neurips.cc/virtual/2024/poster/93174

- **Link**: https://neurips.cc/virtual/2024/poster/93174

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In recent times, Vision-Language Models (VLMs) have been trained under two predominant paradigms. Generative training has enabled Multimodal Large Language Models (MLLMs) to tackle various complex tasks, yet issues such as hallucinations and weak object discrimination persist. Discriminative training, exemplified by models like CLIP, excels in zero-shot image-text classification and retrieval, yet struggles with complex scenarios requiring fine-grained semantic differentiation. This paper addresses these challenges by proposing a unified approach that integrates the strengths of both paradigms. Considering interleaved image-text sequences as the general format of input samples, we introduce a structure-induced training strategy that imposes semantic relationships between input samples and the MLLM’s hidden state. This approach enhances the MLLM’s ability to capture global semantics and distinguish fine-grained semantics. By leveraging dynamic sequence alignment within the Dynamic Time Warping framework and integrating a novel kernel for fine-grained semantic differentiation, our method effectively balances generative and discriminative tasks. Extensive experiments demonstrate the effectiveness of our approach, achieving state-of-the-art results in multiple generative tasks, especially those requiring cognitive and discrimination abilities. Additionally, our method surpasses discriminative benchmarks in interleaved and fine-grained retrieval tasks. By employing a retrieval-augmented generation strategy, our approach further enhances performance in some generative tasks within one model, offering a promising direction for future research in vision-language modeling.

</details>

---

## 16. Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspective

- [ ] Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspective | https://neurips.cc/virtual/2024/poster/93182

- **Link**: https://neurips.cc/virtual/2024/poster/93182

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Foundational Vision-Language models such as CLIP have exhibited impressive generalization in downstream tasks. However, CLIP suffers from a two-level misalignment issue, i.e., task misalignment and data misalignment, when adapting to specific tasks. Soft prompt tuning has mitigated the task misalignment, yet the data misalignment remains a challenge. To analyze the impacts of the data misalignment, we revisit the pre-training and adaptation processes of CLIP and develop a structural causal model. We discover that while we expect to capture task-relevant information for downstream tasks accurately, the task-irrelevant knowledge impacts the prediction results and hampers the modeling of the true relationships between the images and the predicted classes. As task-irrelevant knowledge is unobservable, we leverage the front-door adjustment and propose Causality-Guided Semantic Decoupling and Classification (CDC) to mitigate the interference of task-irrelevant knowledge. Specifically, we decouple semantics contained in the data of downstream tasks and perform classification based on each semantic. Furthermore, we employ the Dempster-Shafer evidence theory to evaluate the uncertainty of each prediction generated by diverse semantics. Experiments conducted in multiple different settings have consistently demonstrated the effectiveness of CDC.

</details>

---

## 17. Homology Consistency Constrained Efficient Tuning for Vision-Language Models

- [ ] Homology Consistency Constrained Efficient Tuning for Vision-Language Models | https://neurips.cc/virtual/2024/poster/93196

- **Link**: https://neurips.cc/virtual/2024/poster/93196

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Efficient transfer learning has shown remarkable performance in tuning large-scale vision-language models (VLMs) toward downstream tasks with limited data resources. The key challenge of efficient transfer lies in adjusting image-text alignment to be task-specific while preserving pre-trained general knowledge. However, existing methods adjust image-text alignment merely on a set of observed samples, e.g., data set and external knowledge base, which cannot guarantee to keep the correspondence of general concepts between image and text latent manifolds without being disrupted and thereby a weak generalization of the adjusted alignment. In this work, we propose a Homology Consistency (HC) constraint for efficient transfer on VLMs, which explicitly constrains the correspondence of image and text latent manifolds through structural equivalence based on persistent homology in downstream tuning. Specifically, we build simplicial complex on the top of data to mimic the topology of latent manifolds, then track the persistence of the homology classes of topological features across multiple scales, and guide the directions of persistence tracks in image and text manifolds to coincide each other, with a deviating perturbation additionally. For practical application, we tailor the implementation of our proposed HC constraint for two main paradigms of adapter tuning. Extensive experiments on few-shot learning over 11 datasets and domain generalization demonstrate the effectiveness and robustness of our method.

</details>

---

## 18. AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models

- [ ] AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models | https://neurips.cc/virtual/2024/poster/93203

- **Link**: https://neurips.cc/virtual/2024/poster/93203

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent research has shown that pre-trained vision-language models are effective at identifying out-of-distribution (OOD) samples by using negative labels as guidance. However, employing consistent negative labels across different OOD datasets often results in semantic misalignments, as these text labels may not accurately reflect the actual space of OOD images. To overcome this issue, we introduce \textit{adaptive negative proxies}, which are dynamically generated during testing by exploring actual OOD images, to align more closely with the underlying OOD label space and enhance the efficacy of negative proxy guidance. Specifically, our approach utilizes a feature memory bank to selectively cache discriminative features from test images, representing the targeted OOD distribution. This facilitates the creation of proxies that can better align with specific OOD datasets. While task-adaptive proxies average features to reflect the unique characteristics of each dataset, the sample-adaptive proxies weight features based on their similarity to individual test samples, exploring detailed sample-level nuances. The final score for identifying OOD samples integrates static negative labels with our proposed adaptive proxies, effectively combining textual and visual knowledge for enhanced performance. Our method is training-free and annotation-free, and it maintains fast testing speed. Extensive experiments across various benchmarks demonstrate the effectiveness of our approach, abbreviated as AdaNeg. Notably, on the large-scale ImageNet benchmark, our AdaNeg significantly outperforms existing methods, with a 2.45\% increase in AUROC and a 6.48\% reduction in FPR95. Codes are available at \url{https://github.com/YBZh/OpenOOD-VLM}.

</details>

---

## 19. LOVA3: Learning to Visual Question Answering, Asking and Assessment

- [ ] LOVA3: Learning to Visual Question Answering, Asking and Assessment | https://neurips.cc/virtual/2024/poster/93210

- **Link**: https://neurips.cc/virtual/2024/poster/93210

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Question answering, asking, and assessment are three innate human traits crucial for understanding the world and acquiring knowledge. By enhancing these capabilities, humans can more effectively utilize data, leading to better comprehension and learning outcomes. However, current Multimodal Large Language Models (MLLMs) primarily focus on question answering, often neglecting the full potential of questioning and assessment skills. In this study, we introduce LOVA3, an innovative framework named ``Learning tO Visual Question Answering, Asking and Assessment,'' designed to equip MLLMs with these additional capabilities. Our approach involves the creation of two supplementary training tasks GenQA and EvalQA, aiming at fostering the skills of asking and assessing questions in the context of images. To develop the questioning ability, we compile a comprehensive set of multimodal foundational tasks. For assessment, we introduce a new benchmark called EvalQABench, comprising 64,000 training samples (split evenly between positive and negative samples) and 5,000 testing samples. We posit that enhancing MLLMs with the capabilities to answer, ask, and assess questions will enhance their multimodal comprehension, ultimately improving overall performance. To validate this hypothesis, we train MLLMs using the LOVA3 framework and evaluate them on a range of multimodal datasets and benchmarks. Our results demonstrate consistent performance gains, underscoring the critical role of these additional tasks in fostering comprehensive intelligence in MLLMs.

</details>

---

## 20. HAWK: Learning to Understand Open-World Video Anomalies

- [ ] HAWK: Learning to Understand Open-World Video Anomalies | https://neurips.cc/virtual/2024/poster/93219

- **Link**: https://neurips.cc/virtual/2024/poster/93219

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video Anomaly Detection (VAD) systems can autonomously monitor and identify disturbances, reducing the need for manual labor and associated costs. However, current VAD systems are often limited by their superficial semantic understanding of scenes and minimal user interaction. Additionally, the prevalent data scarcity in existing datasets restricts their applicability in open-world scenarios.In this paper, we introduce HAWK, a novel framework that leverages interactive large Visual Language Models (VLM) to interpret video anomalies precisely. Recognizing the difference in motion information between abnormal and normal videos, HAWK explicitly integrates motion modality to enhance anomaly identification. To reinforce motion attention, we construct an auxiliary consistency loss within the motion and video space, guiding the video branch to focus on the motion modality. Moreover, to improve the interpretation of motion-to-language, we establish a clear supervisory relationship between motion and its linguistic representation. Furthermore, we have annotated over 8,000 anomaly videos with language descriptions, enabling effective training across diverse open-world scenarios, and also created 8,000 question-answering pairs for users' open-world questions. The final results demonstrate that HAWK achieves SOTA performance, surpassing existing baselines in both video description generation and question-answering. Our codes/dataset/demo will be released at https://github.com/jqtangust/hawk.

</details>

---

## 21. OccamLLM: Fast and Exact Language Model Arithmetic in a Single Step

- [ ] OccamLLM: Fast and Exact Language Model Arithmetic in a Single Step | https://neurips.cc/virtual/2024/poster/93221

- **Link**: https://neurips.cc/virtual/2024/poster/93221

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite significant advancements in text generation and reasoning, Large Language Models (LLMs) still face challenges in accurately performing complex arithmetic operations. Language model systems often enable LLMs to generate code for arithmetic operations to achieve accurate calculations. However, this approach compromises speed and security, and fine-tuning risks the language model losing prior capabilities. We propose a framework that enables exact arithmetic in *a single autoregressive step*, providing faster, more secure, and more interpretable LLM systems with arithmetic capabilities. We use the hidden states of a LLM to control a symbolic architecture that performs arithmetic. Our implementation using Llama 3 with OccamNet as a symbolic model (OccamLlama) achieves 100\% accuracy on single arithmetic operations ($+,-,\times,\div,\sin{},\cos{},\log{},\exp{},\sqrt{}$), outperforming GPT 4o with and without a code interpreter. Furthermore, OccamLlama outperforms GPT 4o with and without a code interpreter on average across a range of mathematical problem solving benchmarks, demonstrating that OccamLLMs can excel in arithmetic tasks, even surpassing much larger models. Code is available at https://github.com/druidowm/OccamLLM.

</details>

---

## 22. Measuring Dejavu Memorization Efficiently

- [ ] Measuring Dejavu Memorization Efficiently | https://neurips.cc/virtual/2024/poster/93225

- **Link**: https://neurips.cc/virtual/2024/poster/93225

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent research has shown that representation learning models may accidentally memorize their training data. For example, the déjà vu method shows that for certain representation learning models and training images, it is sometimes possible to correctly predict the foreground label given only the representation of he background – better than through dataset-level correlations. However, their measurement method requires training two models – one to estimate dataset-level correlations and the other to estimate memorization. This multiple model setup becomes infeasible for large open-source models. In this work, we propose alter native simple methods to estimate dataset-level correlations, and show that these can be used to approximate an off-the-shelf model’s memorization ability without any retraining. This enables, for the first time, the measurement of memorization in pre-trained open-source image representation and vision-language models. Our results show that different ways of measuring memorization yield very similar aggregate results. We also find that open-source models typically have lower aggregate memorization than similar models trained on a subset of the data. The code is available both for vision (https://github.com/facebookresearch/DejaVuOSS) and vision language (https://github.com/facebookresearch/VLMDejaVu) models.

</details>

---

## 23. Vision-Language Navigation with Energy-Based Policy

- [ ] Vision-Language Navigation with Energy-Based Policy | https://neurips.cc/virtual/2024/poster/93232

- **Link**: https://neurips.cc/virtual/2024/poster/93232

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language navigation (VLN) requires an agent to execute actions following human instructions. Existing VLN models are optimized through expert demonstrations by supervised behavioural cloning or incorporating manual reward engineering. While straightforward, these efforts overlook the accumulation of errors in the Markov decision process, and struggle to match the distribution of the expert policy. Going beyond this, we propose an Energy-based Navigation Policy (ENP) to model the joint state-action distribution using an energy-based model. At each step, low energy values correspond to the state-action pairs that the expert is most likely to perform, and vice versa. Theoretically, the optimization objective is equivalent to minimizing the forward divergence between the occupancy measure of the expert and ours. Consequently, ENP learns to globally align with the expert policy by maximizing the likelihood of the actions and modeling the dynamics of the navigation states in a collaborative manner. With a variety of VLN architectures, ENP achieves promising performances on R2R, REVERIE, RxR, and R2R-CE, unleashing the power of existing VLN models.

</details>

---

## 24. Lumen: Unleashing Versatile Vision-Centric Capabilities of Large Multimodal Models

- [ ] Lumen: Unleashing Versatile Vision-Centric Capabilities of Large Multimodal Models | https://neurips.cc/virtual/2024/poster/93228

- **Link**: https://neurips.cc/virtual/2024/poster/93228

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Model (LMM) is a hot research topic in the computer vision area and has also demonstrated remarkable potential across multiple disciplinary fields. A recent trend is to further extend and enhance the perception capabilities of LMMs. The current methods follow the paradigm of adapting the visual task outputs to the format of the language model, which is the main component of a LMM. This adaptation leads to convenient development of such LMMs with minimal modifications, however, it overlooks the intrinsic characteristics of diverse visual tasks and hinders the learning of perception capabilities. To address this issue, we propose a novel LMM architecture named Lumen, a Large multimodal model with versatile vision-centric capability enhancement. We decouple the LMM's learning of perception capabilities into task-agnostic and task-specific stages. Lumen first promotes fine-grained vision-language concept alignment, which is the fundamental capability for various visual tasks. Thus the output of the task-agnostic stage is a shared representation for all the tasks we address in this paper. Then the task-specific decoding is carried out by flexibly routing the shared representation to lightweight task decoders with negligible training efforts. Comprehensive experimental results on a series of vision-centric and VQA benchmarks indicate that our Lumen model not only achieves or surpasses the performance of existing LMM-based approaches in a range of vision-centric tasks while maintaining general visual understanding and instruction following capabilities.

</details>

---

## 25. BendVLM: Test-Time Debiasing of Vision-Language Embeddings

- [ ] BendVLM: Test-Time Debiasing of Vision-Language Embeddings | https://neurips.cc/virtual/2024/poster/93242

- **Link**: https://neurips.cc/virtual/2024/poster/93242

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language (VL) embedding models have been shown to encode biases present in their training data, such as societal biases that prescribe negative characteristics to members of various racial and gender identities. Due to their wide-spread adoption for various tasks ranging from few-shot classification to text-guided image generation, debiasing VL models is crucial. Debiasing approaches that fine-tune the VL model often suffer from catastrophic forgetting. On the other hand, fine-tuning-free methods typically utilize a ``one-size-fits-all" approach that assumes that correlation with the spurious attribute can be explained using a single linear direction across all possible inputs. In this work, we propose a nonlinear, fine-tuning-free approach for VL embedding model debiasing that tailors the debiasing operation to each unique input. This allows for a more flexible debiasing approach. Additionally, we do not require knowledge of the set of inputs a priori to inference time, making our method more appropriate for online tasks such as retrieval and text guided image generation.

</details>

---

## 26. Learning to Reason Iteratively and Parallelly for Complex Visual Reasoning Scenarios

- [ ] Learning to Reason Iteratively and Parallelly for Complex Visual Reasoning Scenarios | https://neurips.cc/virtual/2024/poster/93246

- **Link**: https://neurips.cc/virtual/2024/poster/93246

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Complex visual reasoning and question answering (VQA) is a challenging task that requires compositional multi-step processing and higher-level reasoning capabilities beyond the immediate recognition and localization of objects and events. Here, we introduce a fully neural Iterative and Parallel Reasoning Mechanism (IPRM) that combines two distinct forms of computation -- iterative and parallel -- to better address complex VQA scenarios.  Specifically, IPRM's "iterative" computation facilitates compositional step-by-step reasoning for scenarios wherein individual operations need to be computed, stored, and recalled dynamically (e.g. when computing the query “determine the color of pen to the left of the child in red t-shirt sitting at the white table”). Meanwhile, its  "parallel'' computation allows for the simultaneous exploration of different reasoning paths and benefits more robust and efficient execution of  operations that are mutually independent (e.g. when counting individual colors for the query: "determine the maximum occurring color amongst all t-shirts'"). We design IPRM as a lightweight and fully-differentiable neural module that can be conveniently applied to both transformer and non-transformer vision-language backbones. It notably outperforms prior task-specific methods and transformer-based attention modules across various image and video VQA benchmarks testing distinct complex reasoning capabilities such as compositional spatiotemporal reasoning (AGQA), situational reasoning (STAR), multi-hop reasoning generalization (CLEVR-Humans) and causal event linking (CLEVRER-Humans). Further, IPRM's internal computations can be visualized across reasoning steps, aiding interpretability and diagnosis of its errors.

</details>

---

## 27. Octopus: A Multi-modal LLM with Parallel Recognition and Sequential Understanding

- [ ] Octopus: A Multi-modal LLM with Parallel Recognition and Sequential Understanding | https://neurips.cc/virtual/2024/poster/93251

- **Link**: https://neurips.cc/virtual/2024/poster/93251

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A mainstream of Multi-modal Large Language Models (MLLMs) have two essential functions, i.e., visual recognition (e.g., grounding) and understanding (e.g., visual question answering). Presently, all these MLLMs integrate visual recognition and understanding in a same sequential manner in the LLM head, i.e., generating the response token-by-token for both recognition and understanding. We think unifying them in the same sequential manner is not optimal for two reasons: 1) parallel recognition is more efficient than sequential recognition and is actually prevailing in deep visual recognition, and 2) the recognition results can be integrated to help high-level cognition (while the current manner does not). Such motivated, this paper proposes a novel “parallel recognition → sequential understanding” framework for MLLMs. The bottom LLM layers are utilized for parallel recognition and the recognition results are relayed into the top LLM layers for sequential understanding. Specifically, parallel recognition in the bottom LLM layers is implemented via object queries, a popular mechanism in DEtection TRansformer, which we find to harmonize well with the LLM layers. Empirical studies show our MLLM named Octopus improves accuracy on popular MLLM tasks and is up to 5× faster on visual grounding tasks.

</details>

---

## 28. MoVA: Adapting Mixture of Vision Experts to Multimodal Context

- [ ] MoVA: Adapting Mixture of Vision Experts to Multimodal Context | https://neurips.cc/virtual/2024/poster/93279

- **Link**: https://neurips.cc/virtual/2024/poster/93279

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As the key component in multimodal large language models (MLLMs), the ability of the visual encoder greatly affects MLLM's understanding on diverse image content. Although some large-scale pretrained vision encoders such as vision encoders in CLIP and DINOv2 have brought promising performance, we found that there is still no single vision encoder that can dominate various image content understanding, e.g., the CLIP vision encoder leads to outstanding results on general image understanding but poor performance on document or chart content. To alleviate the bias of CLIP vision encoder, we first delve into the inherent behavior of different pre-trained vision encoders and then propose the MoVA, a powerful and novel MLLM, adaptively routing and fusing task-specific vision experts with a coarse-to-fine mechanism. In the coarse-grained stage, we design a context-aware expert routing strategy to dynamically select the most suitable vision experts according to the user instruction, input image, and expertise of vision experts. This benefits from the powerful model function understanding ability of the large language model (LLM). In the fine-grained stage, we elaborately conduct the mixture-of-vision-expert adapter (MoV-Adapter) to extract and fuse task-specific knowledge from various experts. This coarse-to-fine paradigm effectively leverages representations from experts based on multimodal context and model expertise, further enhancing the generalization ability. We conduct extensive experiments to evaluate the effectiveness of the proposed approach. Without any bells and whistles, MoVA can achieve significant performance gains over current state-of-the-art methods in a wide range of challenging multimodal benchmarks.

</details>

---

## 29. Unveiling the Tapestry of Consistency in Large Vision-Language Models

- [ ] Unveiling the Tapestry of Consistency in Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/93307

- **Link**: https://neurips.cc/virtual/2024/poster/93307

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have recently achieved rapid progress, exhibiting great perception and reasoning abilities concerning visual information. However, when faced with prompts in different sizes of solution spaces, LVLMs fail to always give consistent answers regarding the same knowledge point. This inconsistency of answers between different solution spaces is prevalent in LVLMs and erodes trust. To this end, we provide a multi-modal benchmark ConBench, to intuitively analyze how LVLMs perform when the solution space of a prompt revolves around a knowledge point. Based on the ConBench tool, we are the first to reveal the tapestry and get the following findings: (1) In the discriminate realm, the larger the solution space of the prompt, the lower the accuracy of the answers. (2) Establish the relationship between the discriminative and generative realms: the accuracy of the discriminative question type exhibits a strong positive correlation with its Consistency with the caption. (3) Compared to open-source models, closed-source models exhibit a pronounced bias advantage in terms of Consistency. Eventually, we ameliorate the consistency of LVLMs by trigger-based diagnostic refinement, indirectly improving the performance of their caption. We hope this paper will accelerate the research community in better evaluating their models and encourage future advancements in the consistency domain.

</details>

---

## 30. TransAgent: Transfer Vision-Language Foundation Models with Heterogeneous Agent Collaboration

- [ ] TransAgent: Transfer Vision-Language Foundation Models with Heterogeneous Agent Collaboration | https://neurips.cc/virtual/2024/poster/93312

- **Link**: https://neurips.cc/virtual/2024/poster/93312

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language foundation models (such as CLIP) have recently shown their power in transfer learning, owing to large-scale image-text pre-training. However, target domain data in the downstream tasks can be highly different from the pre-training phase, which makes it hard for such a single model to generalize well. Alternatively, there exists a wide range of expert models that contain diversified vision and/or language knowledge pre-trained on different modalities, tasks, networks, and datasets. Unfortunately, these models are "isolated agents" with heterogeneous structures, and how to integrate their knowledge for generalizing CLIP-like models has not been fully explored. To bridge this gap, we propose a general and concise TransAgent framework, which transports the knowledge of the isolated agents in a unified manner, and effectively guides CLIP to generalize with multi-source knowledge distillation. With such a distinct framework, we flexibly collaborate with 11 heterogeneous agents to empower vision-language foundation models, without further cost in the inference phase. Finally, our TransAgent achieves state-of-the-art performance on 11 visual recognition datasets. Under the same low-shot setting, it outperforms the popular CoOp with around 10\% on average, and 20\% on EuroSAT which contains large domain shifts.

</details>

---

## 31. One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection

- [ ] One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection | https://neurips.cc/virtual/2024/poster/93345

- **Link**: https://neurips.cc/virtual/2024/poster/93345

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Traditional Anomaly Detection (AD) methods have predominantly relied on unsupervised learning from extensive normal data. Recent AD methods have evolved with the advent of large pre-trained vision-language models, enhancing few-shot anomaly detection capabilities. However, these latest AD methods still exhibit limitations in accuracy improvement. One contributing factor is their direct comparison of a query image's features with those of few-shot normal images. This direct comparison often leads to a loss of precision and complicates the extension of these techniques to more complex domains—an area that remains underexplored in a more refined and comprehensive manner. To address these limitations, we introduce the anomaly personalization method, which performs a personalized one-to-normal transformation of query images using an anomaly-free customized generation model, ensuring close alignment with the normal manifold. Moreover, to further enhance the stability and robustness of prediction results, we propose a triplet contrastive anomaly inference strategy, which incorporates a comprehensive comparison between the query and generated anomaly-free data pool and prompt information. Extensive evaluations across eleven datasets in three domains demonstrate our model's effectiveness compared to the latest AD methods. Additionally, our method has been proven to transfer flexibly to other AD methods, with the generated image data effectively improving the performance of other AD methods.

</details>

---

## 32. OneRef:  Unified One-tower Expression Grounding and Segmentation with Mask Referring Modeling

- [ ] OneRef:  Unified One-tower Expression Grounding and Segmentation with Mask Referring Modeling | https://neurips.cc/virtual/2024/poster/93378

- **Link**: https://neurips.cc/virtual/2024/poster/93378

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Constrained by the separate encoding of vision and language, existing grounding and referring segmentation works heavily rely on bulky Transformer-based fusion en-/decoders and a variety of early-stage interaction technologies. Simultaneously, the current mask visual language modeling (MVLM) fails to capture the nuanced referential relationship between image-text in referring tasks. In this paper, we propose OneRef , a minimalist referring framework built on the modality-shared one-tower transformer that unifies the visual and linguistic feature spaces. To modeling the referential relationship, we introduce a novel MVLM paradigm called Mask Referring Modeling ( MRefM ), which encompasses both referring-aware mask image modeling and referring-aware mask language modeling. Both modules not only reconstruct modality-related content but also cross-modal referring content. Within MRefM, we propose a referring-aware dynamic image masking strategy that is aware of the referred region rather than relying on fixed ratios or generic random masking schemes. By leveraging the unified visual language feature space and incorporating MRefM's ability to model the referential relations, our approach enables direct regression of the referring results without resorting to various complex techniques. Our method consistently surpasses existing approaches and achieves SoTA performance on both grounding and segmentation tasks, providing valuable insights for future research. Our code and models are available at https://github.com/linhuixiao/OneRef.

</details>

---

## 33. Understanding Information Storage and Transfer in Multi-Modal Large Language Models

- [ ] Understanding Information Storage and Transfer in Multi-Modal Large Language Models | https://neurips.cc/virtual/2024/poster/93402

- **Link**: https://neurips.cc/virtual/2024/poster/93402

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding the mechanisms of information storage and transfer in Transformer-based models is important for driving model understanding progress. Recent work has studied these mechanisms for Large Language Models (LLMs), revealing insights on how information is stored in a model's parameters and how information flows to and from these parameters in response to specific prompts. However, these studies have not yet been extended to Multi-modal Large Language Models (MLLMs). Given their expanding capabilities and real-world use, we start by studying one aspect of these models -- how MLLMs process information in a factual visual question answering task. We use a constraint-based formulation which views a visual question as having a set of visual or textual constraints that the model's generated answer must satisfy to be correct (e.g. What movie directed by \emph{the director in this photo} has won a \emph{Golden Globe}?). Under this setting, we contribute i) a method that extends causal information tracing from pure language to the multi-modal setting, and ii) \emph{VQA-Constraints}, a test-bed of 9.7K visual questions annotated with constraints. We use these tools to study two open-source MLLMs, LLaVa and multi-modal Phi-2. Our key findings show that these MLLMs rely on MLP and self-attention blocks in much earlier layers for information storage, compared to LLMs whose mid-layer MLPs are more important. We also show that a consistent small subset of visual tokens output by the vision encoder are responsible for transferring information from the image to these causal blocks. We validate these mechanisms by introducing MultEdit a model-editing algorithm that can correct errors and insert new long-tailed information into MLLMs by targeting these causal blocks. We will publicly release our dataset and code.

</details>

---

## 34. CLAP4CLIP: Continual Learning with Probabilistic Finetuning for Vision-Language Models

- [ ] CLAP4CLIP: Continual Learning with Probabilistic Finetuning for Vision-Language Models | https://neurips.cc/virtual/2024/poster/93449

- **Link**: https://neurips.cc/virtual/2024/poster/93449

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Continual learning (CL) aims to help deep neural networks to learn new knowledge while retaining what has been learned. Owing to their powerful generalizability,  pre-trained vision-language models such as Contrastive Language-Image Pre-training (CLIP)  have lately gained traction as practical CL candidates. However, the domain mismatch between the pre-training and the downstream CL tasks calls for finetuning of the CLIP on the latter. The deterministic nature of the existing finetuning methods makes them overlook the many possible interactions across the modalities and deems them unsafe for high-risk tasks requiring reliable uncertainty estimation. To address these, our work proposes C ontinual L e A rning with P robabilistic finetuning (CLAP) - a probabilistic modeling framework over visual-guided text features per task, thus providing more calibrated CL finetuning. Unlike recent data-hungry anti-forgetting CL techniques, CLAP alleviates forgetting by exploiting the rich pre-trained knowledge of CLIP for weight initialization and distribution regularization of task-specific parameters. Cooperating with the diverse range of existing prompting methods, CLAP can surpass the predominant deterministic finetuning approaches for CL with CLIP. We conclude with out-of-the-box applications of superior uncertainty estimation abilities of CLAP including novel data detection and exemplar selection within the existing CL setups. Our code is available at https://github.com/srvCodes/clap4clip.

</details>

---

## 35. Conjugated Semantic Pool Improves OOD Detection with Pre-trained Vision-Language Models

- [ ] Conjugated Semantic Pool Improves OOD Detection with Pre-trained Vision-Language Models | https://neurips.cc/virtual/2024/poster/93471

- **Link**: https://neurips.cc/virtual/2024/poster/93471

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A straightforward pipeline for zero-shot out-of-distribution (OOD) detection involves selecting potential OOD labels from an extensive semantic pool and then leveraging a pre-trained vision-language model to perform classification on both in-distribution (ID) and OOD labels. In this paper, we theorize that enhancing performance requires expanding the semantic pool, while increasing the expected probability of selected OOD labels being activated by OOD samples, and ensuring low mutual dependence among the activations of these OOD labels. A natural expansion manner is to adopt a larger lexicon; however, the inevitable introduction of numerous synonyms and uncommon words fails to meet the above requirements, indicating that viable expansion manners move beyond merely selecting words from a lexicon. Since OOD detection aims to correctly classify input images into ID/OOD class groups, we can "make up" OOD label candidates which are not standard class names but beneficial for the process. Observing that the original semantic pool is comprised of unmodified specific class names, we correspondingly construct a conjugated semantic pool (CSP) consisting of modified superclass names, each serving as a cluster center for samples sharing similar properties across different categories. Consistent with our established theory, expanding OOD label candidates with the CSP satisfies the requirements and outperforms existing works by 7.89% in FPR95. Codes are available in https://github.com/MengyuanChen21/NeurIPS2024-CSP.

</details>

---

## 36. Towards Safe Concept Transfer of Multi-Modal Diffusion via Causal Representation Editing

- [ ] Towards Safe Concept Transfer of Multi-Modal Diffusion via Causal Representation Editing | https://neurips.cc/virtual/2024/poster/93488

- **Link**: https://neurips.cc/virtual/2024/poster/93488

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in vision-language-to-image (VL2I) diffusion generation have made significant progress. While generating images from broad vision-language inputs holds promise, it also raises concerns about potential misuse, such as copying artistic styles without permission, which could have legal and social consequences. Therefore, it's crucial to establish governance frameworks to ensure ethical and copyright integrity, especially with widely used diffusion models. To address these issues, researchers have explored various approaches, such as dataset filtering, adversarial perturbations, machine unlearning, and inference-time refusals. However, these methods often lack either scalability or effectiveness. In response, we propose a new framework called causal representation editing (CRE), which extends representation editing from large language models (LLMs) to diffusion-based models. CRE enhances the efficiency and flexibility of safe content generation by intervening at diffusion timesteps causally linked to unsafe concepts. This allows for precise removal of harmful content while preserving acceptable content quality, demonstrating superior effectiveness, precision and scalability compared to existing methods. CRE can handle complex scenarios, including incomplete or blurred representations of unsafe concepts, offering a promising solution to challenges in managing harmful content generation in diffusion-based models.

</details>

---

## 37. Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning

- [ ] Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning | https://neurips.cc/virtual/2024/poster/93492

- **Link**: https://neurips.cc/virtual/2024/poster/93492

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Accurate emotion perception is crucial for various applications, including human-computer interaction, education, and counseling.However, traditional single-modality approaches often fail to capture the complexity of real-world emotional expressions, which are inherently multimodal. Moreover, existing Multimodal Large Language Models (MLLMs) face challenges in integrating audio and recognizing subtle facial micro-expressions. To address this, we introduce the MERR dataset, containing 28,618 coarse-grained and 4,487 fine-grained annotated samples across diverse emotional categories. This dataset enables models to learn from varied scenarios and generalize to real-world applications. Furthermore, we propose Emotion-LLaMA, a model that seamlessly integrates audio, visual, and textual inputs through emotion-specific encoders. By aligning features into a shared space and employing a modified LLaMA model with instruction tuning, Emotion-LLaMA significantly enhances both emotional recognition and reasoning capabilities. Extensive evaluations show Emotion-LLaMA outperforms other MLLMs, achieving top scores in Clue Overlap (7.83) and Label Overlap (6.25) on EMER, an F1 score of 0.9036 on MER2023-SEMI challenge, and the highest UAR (45.59) and WAR (59.37) in zero-shot evaluations on DFEW dataset.

</details>

---

## 38. Empowering Visible-Infrared Person Re-Identification with Large Foundation Models

- [ ] Empowering Visible-Infrared Person Re-Identification with Large Foundation Models | https://neurips.cc/virtual/2024/poster/93497

- **Link**: https://neurips.cc/virtual/2024/poster/93497

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visible-Infrared Person Re-identification (VI-ReID) is a challenging cross-modal retrieval task due to significant modality differences, primarily resulting from the absence of color information in the infrared modality. The development of large foundation models like Large Language Models (LLMs) and Vision Language Models (VLMs) motivates us to explore a feasible solution to empower VI-ReID with off-the-shelf large foundation models. To this end, we propose a novel Text-enhanced VI-ReID framework driven by Large Foundation Models (TVI-LFM). The core idea is to enrich the representation of the infrared modality with textual descriptions automatically generated by VLMs. Specifically, we incorporate a pre-trained VLM to extract textual features from texts generated by VLM and augmented by LLM, and incrementally fine-tune the text encoder to minimize the domain gap between generated texts and original visual modalities. Meanwhile, to enhance the infrared modality with extracted textual representations, we leverage modality alignment capabilities of VLMs and VLM-generated feature-level filters. This enables the text model to learn complementary features from the infrared modality, ensuring the semantic structural consistency between the fusion modality and the visible modality. Furthermore, we introduce modality joint learning to align features across all modalities, ensuring that textual features maintain stable semantic representation of overall pedestrian appearance during complementary information learning. Additionally, a modality ensemble retrieval strategy is proposed to leverage complementary strengths of each query modality to improve retrieval effectiveness and robustness. Extensive experiments on three expanded VI-ReID datasets demonstrate that our method significantly improves the retrieval performance, paving the way for the utilization of large foundation models in downstream multi-modal retrieval tasks.

</details>

---

## 39. Prism: A Framework for Decoupling and Assessing the Capabilities of VLMs

- [ ] Prism: A Framework for Decoupling and Assessing the Capabilities of VLMs | https://neurips.cc/virtual/2024/poster/93501

- **Link**: https://neurips.cc/virtual/2024/poster/93501

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) demonstrate remarkable proficiency in addressing a wide array of visual questions, which requires strong perception and reasoning faculties. Assessing these two competencies independently is crucial for model refinement, despite the inherent difficulty due to the intertwined nature of seeing and reasoning in existing VLMs. To tackle this issue, we present Prism, an innovative framework designed to disentangle the perception and reasoning processes involved in visual question solving. Prism comprises two distinct stages: a perception stage that utilizes a VLM to extract and articulate visual information in textual form, and a reasoning stage that formulates responses based on the extracted visual information using a Large Language Model (LLM). This modular design enables the systematic comparison and assessment of both proprietary and open-source VLM for their perception and reasoning strengths. Our analytical framework provides several valuable insights, underscoring Prism's potential as a cost-effective solution for vision-language tasks.By combining a streamlined VLM focused on perception with a powerful LLM tailored for reasoning, Prism achieves superior results in general vision-language tasks while substantially cutting down on training and operational expenses. Quantitative evaluations show that Prism, when configured with a vanilla 2B LLaVA and freely accessible GPT-3.5, delivers performance on par with VLMs $10 \times$ larger on the rigorous multimodal benchmark MMStar.

</details>

---

## 40. Recognize Any Regions

- [ ] Recognize Any Regions | https://neurips.cc/virtual/2024/poster/93502

- **Link**: https://neurips.cc/virtual/2024/poster/93502

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding the semantics of individual regions or patches of unconstrained images, such as open-world object detection, remains a critical yet challenging task in computer vision. Building on the success of powerful image-level vision-language (ViL) foundation models like CLIP, recent efforts have sought to harness their capabilities by either training a contrastive model from scratch with an extensive collection of region-label pairs or aligning the outputs of a detection model with image-level representations of region proposals. Despite notable progress, these approaches are plagued by computationally intensive training requirements, susceptibility to data noise, and deficiency in contextual information. To address these limitations, we explore the synergistic potential of off-the-shelf foundation models, leveraging their respective strengths in localization and semantics. We introduce a novel, generic, and efficient architecture, named RegionSpot, designed to integrate position-aware localization knowledge from a localization foundation model (e.g., SAM) with semantic information from a ViL model (e.g., CLIP). To fully exploit pretrained knowledge while minimizing training overhead, we keep both foundation models frozen, focusing optimization efforts solely on a lightweight attention-based knowledge integration module.Extensive experiments in open-world object recognition show that our  RegionSpot achieves significant performance gain over prior alternatives, along with substantial computational savings (e.g., training our model with 3 million data in a single day using 8 V100 GPUs). RegionSpot outperforms GLIP-L by 2.9 in mAP on LVIS val set,  with an even larger margin of 13.1 AP for more challenging and rare categories, and a 2.5 AP increase on ODinW. Furthermore, it exceeds GroundingDINO-L by 11.0 AP for rare categories on the LVIS minival set.

</details>

---

## 41. HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid

- [ ] HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid | https://neurips.cc/virtual/2024/poster/93535

- **Link**: https://neurips.cc/virtual/2024/poster/93535

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Physical Human-Scene Interaction (HSI) plays a crucial role in numerous applications.     However, existing HSI techniques are limited to specific object dynamics and privileged information, which prevents the development of more comprehensive applications.    To address this limitation, we introduce HumanVLA for general object rearrangement directed by practical vision and language.     A teacher-student framework is utilized to develop HumanVLA.    A state-based teacher policy is trained first using goal-conditioned reinforcement learning and adversarial motion prior.    Then, it is distilled into a vision-language-action model via behavior cloning.    We propose several key insights to facilitate the large-scale learning process.    To support general object rearrangement by physical humanoid, we introduce a novel Human-in-the-Room dataset encompassing various rearrangement tasks.    Through extensive experiments and analysis, we demonstrate the effectiveness of our approach.

</details>

---

## 42. PromptFix: You Prompt and We Fix the Photo

- [ ] PromptFix: You Prompt and We Fix the Photo | https://neurips.cc/virtual/2024/poster/93588

- **Link**: https://neurips.cc/virtual/2024/poster/93588

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models equipped with language models demonstrate excellent controllability in image generation tasks, allowing image processing to adhere to human instructions. However, the lack of diverse instruction-following data hampers the development of models that effectively recognize and execute user-customized instructions, particularly in low-level tasks. Moreover, the stochastic nature of the diffusion process leads to deficiencies in image generation or editing tasks that require the detailed preservation of the generated images. To address these limitations, we propose PromptFix, a comprehensive framework that enables diffusion models to follow human instructions to perform a wide variety of image-processing tasks. First, we construct a large-scale instruction-following dataset that covers comprehensive image-processing tasks, including low-level tasks, image editing, and object creation. Next, we propose a high-frequency guidance sampling method to explicitly control the denoising process and preserve high-frequency details in unprocessed areas. Finally, we design an auxiliary prompting adapter, utilizing Vision-Language Models (VLMs) to enhance text prompts and improve the model's task generalization. Experimental results show that PromptFix outperforms previous methods in various image-processing tasks. Our proposed model also achieves comparable inference efficiency with these baseline models and exhibits superior zero-shot capabilities in blind restoration and combination tasks.

</details>

---

## 43. Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE

- [ ] Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE | https://neurips.cc/virtual/2024/poster/93590

- **Link**: https://neurips.cc/virtual/2024/poster/93590

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have shown impressive capabilities as a general-purpose interface for various visual and linguistic tasks. However, building a unified MLLM for multi-task learning in the medical field remains a thorny challenge. To mitigate the tug-of-war problem of multi-modal multi-task optimization in MLLMs, recent advances primarily focus on improving the LLM components, while neglecting the connector that bridges the gap between modalities. In this paper, we introduce Uni-Med, a novel medical generalist foundation model which consists of a universal visual feature extraction module, a connector mixture-of-experts (CMoE) module, and an LLM. Benefiting from the proposed CMoE that leverages a well-designed router with a mixture of projection experts at the connector, Uni-Med achieves efficient solution to the tug-of-war problem and can perform six different medical tasks including question answering, visual question answering, report generation, referring expression comprehension, referring expression generation and image classification. To the best of our knowledge, Uni-Med is the first effort to tackle multi-task interference at the connector in MLLMs. Extensive ablation experiments validate the effectiveness of introducing CMoE under any configuration, with up to an average 8% performance gains. We further provide interpretation analysis of the tug-of-war problem from the perspective of gradient optimization and parameter statistics. Compared to previous state-of-the-art medical MLLMs, Uni-Med achieves competitive or superior evaluation metrics on diverse tasks. Code and resources are available at https://github.com/MSIIP/Uni-Med.

</details>

---

## 44. Compositional 3D-aware Video Generation with LLM Director

- [ ] Compositional 3D-aware Video Generation with LLM Director | https://neurips.cc/virtual/2024/poster/93599

- **Link**: https://neurips.cc/virtual/2024/poster/93599

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Significant progress has been made in text-to-video generation through the use of powerful generative models and large-scale internet data. However, substantial challenges remain in precisely controlling individual elements within the generated video, such as the movement and appearance of specific characters and the manipulation of viewpoints. In this work, we propose a novel paradigm that generates each element in 3D representation separately and then composites them with priors from Large Language Models (LLMs) and 2D diffusion models. Specifically, given an input textual query, our scheme consists of four stages: 1) we leverage the LLMs as the director to first decompose the complex query into several sub-queries, where each sub-query describes each element of the generated video; 2) to generate each element, pre-trained models are invoked by the LLMs to obtain the corresponding 3D representation; 3) to composite the generated 3D representations, we prompt multi-modal LLMs to produce coarse guidance on the scale, location, and trajectory of different objects; 4) to make the results adhere to natural distribution, we further leverage 2D diffusion priors and use score distillation sampling to refine the composition. Extensive experiments demonstrate that our method can generate high-fidelity videos from text with flexible control over each element.

</details>

---

## 45. Diff-eRank: A Novel Rank-Based Metric for Evaluating Large Language Models

- [ ] Diff-eRank: A Novel Rank-Based Metric for Evaluating Large Language Models | https://neurips.cc/virtual/2024/poster/93654

- **Link**: https://neurips.cc/virtual/2024/poster/93654

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have transformed natural language processing and extended their powerful capabilities to multi-modal domains. As LLMs continue to advance, it is crucial to develop diverse and appropriate metrics for their evaluation. In this paper, we introduce a novel rank-based metric, Diff-eRank, grounded in information theory and geometry principles. Diff-eRank assesses LLMs by analyzing their hidden representations, providing a quantitative measure of how efficiently they eliminate redundant information during training. We demonstrate the applicability of Diff-eRank in both single-modal (e.g., language) and multi-modal settings. For language models, our results show that Diff-eRank increases with model size and correlates well with conventional metrics such as loss and accuracy. In the multi-modal context, we propose an alignment evaluation method based on the eRank, and verify that contemporary multi-modal LLMs exhibit strong alignment performance based on our method. Our code is publicly available at https://github.com/waltonfuture/Diff-eRank.

</details>

---

## 46. VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model for Hundreds of Vision-Language Tasks

- [ ] VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model for Hundreds of Vision-Language Tasks | https://neurips.cc/virtual/2024/poster/93655

- **Link**: https://neurips.cc/virtual/2024/poster/93655

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present VisionLLM v2, an end-to-end generalist multimodal large model (MLLM) that unifies visual perception, understanding, and generation within a single framework. Unlike traditional MLLMs limited to text output, VisionLLM v2 significantly broadens its application scope. It excels not only in conventional visual question answering (VQA) but also in open-ended, cross-domain vision tasks such as object localization, pose estimation, and image generation and editing. To this end, we propose a new information transmission mechanism termed ``super link'', as a medium to connect MLLM with task-specific decoders. It not only allows flexible transmission of task information and gradient feedback between the MLLM and multiple downstream decoders but also effectively resolves training conflicts in multi-tasking scenarios. In addition, to support the diverse range of tasks, we carefully collected and combed training data from hundreds of public vision and vision-language tasks. In this way, our model can be joint-trained end-to-end on hundreds of vision language tasks and generalize to these tasks using a set of shared parameters through different user prompts, achieving performance comparable to task-specific models. We believe VisionLLM v2 will offer a new perspective on the generalization of MLLMs.

</details>

---

## 47. Membership Inference Attacks against Large Vision-Language Models

- [ ] Membership Inference Attacks against Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/93657

- **Link**: https://neurips.cc/virtual/2024/poster/93657

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLLMs) exhibit promising capabilities for processing multi-modal tasks across various application scenarios. However, their emergence also raises significant data security concerns, given the potential inclusion of sensitive information, such as private photos and medical records, in their training datasets. Detecting inappropriately used data in VLLMs remains a critical and unresolved issue, mainly due to the lack of standardized datasets and suitable methodologies. In this study, we introduce the first membership inference attack (MIA) benchmark tailored for various VLLMs to facilitate training data detection. Then, we propose a novel MIA pipeline specifically designed for token-level image detection. Lastly, we present a new metric called MaxRényi-K%, which is based on the confidence of the model output and applies to both text and image data. We believe that our work can deepen the understanding and methodology of MIAs in the context of VLLMs. Our code and datasets are available at https://github.com/LIONS-EPFL/VL-MIA.

</details>

---

## 48. Wings: Learning Multimodal LLMs without Text-only Forgetting

- [ ] Wings: Learning Multimodal LLMs without Text-only Forgetting | https://neurips.cc/virtual/2024/poster/93663

- **Link**: https://neurips.cc/virtual/2024/poster/93663

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs), initiated with a trained LLM, first align images with text and then fine-tune on multimodal mixed inputs. However, during the continued training, the MLLM catastrophically forgets the text-only instructions that the initial LLM masters. In this paper, we present Wings, a novel MLLM that excels in both text-only and multimodal instructions. By examining attention across layers of MLLM, we find that text-only forgetting is related to the attention shifts from pre-image to post-image text. From that, we construct an additional Low-Rank Residual Attention (LoRRA) block that acts as the "modality learner" to expand the learnable space and compensate for the attention shift. The complementary learners, like "wings" on either side, are connected in parallel to each layer's attention block. The LoRRA mirrors the structure of attention but utilizes low-rank connections to ensure efficiency. Initially, image and text inputs are aligned with visual learners operating alongside the main attention, balancing focus on visual elements. Later, textual learners are integrated with token-wise routing, blending the outputs of both modality learners collaboratively. Our experimental results demonstrate that Wings outperforms equally-scaled MLLMs in both text-only and visual question-answering tasks. Wings with compensation of learners addresses text-only forgetting during visual modality expansion in general MLLMs.

</details>

---

## 49. Calibrated Self-Rewarding Vision Language Models

- [ ] Calibrated Self-Rewarding Vision Language Models | https://neurips.cc/virtual/2024/poster/93685

- **Link**: https://neurips.cc/virtual/2024/poster/93685

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have made substantial progress by integrating pre-trained large language models (LLMs) and vision models through instruction tuning. Despite these advancements, LVLMs often exhibit the hallucination phenomenon, where generated text responses appear linguistically plausible but contradict the input image, indicating a misalignment between image and text pairs. This misalignment arises because the model tends to prioritize textual information over visual input, even when both the language model and visual representations are of high quality. Existing methods leverage additional models or human annotations to curate preference data and enhance modality alignment through preference optimization. These approaches are resource-intensive and may not effectively reflect the target LVLM's preferences, making the curated preferences easily distinguishable. Our work addresses these challenges by proposing the Calibrated Self-Rewarding (CSR) approach, which enables the model to self-improve by iteratively generating candidate responses, evaluating the reward for each response, and curating preference data for fine-tuning. In the reward modeling, we employ a step-wise strategy and incorporate visual constraints into the self-rewarding process to place greater emphasis on visual input. Empirical results demonstrate that CSR significantly enhances performance and reduces hallucinations across twelve benchmarks and tasks, achieving substantial improvements over existing methods by 7.62\%. Our empirical results are further supported by rigorous theoretical analysis, under mild assumptions, verifying the effectiveness of introducing visual constraints into the self-rewarding paradigm. Additionally, CSR shows compatibility with different vision-language models and the ability to incrementally improve performance through iterative fine-tuning.

</details>

---

## 50. InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD

- [ ] InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD | https://neurips.cc/virtual/2024/poster/93691

- **Link**: https://neurips.cc/virtual/2024/poster/93691

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The Large Vision-Language Model (LVLM) field has seen significant advancements, yet its progression has been hindered by challenges in comprehending fine-grained visual content due to limited resolution. Recent efforts have aimed to enhance the high-resolution understanding capabilities of LVLMs, yet they remain capped at approximately 1500 $\times$ 1500 pixels and constrained to a relatively narrow resolution range. This paper represents InternLM-XComposer2-4KHD, a groundbreaking exploration into elevating LVLM resolution capabilities up to 4K HD (3840 × 1600) and beyond. Concurrently, considering the ultra-high resolution may not be necessary in all scenarios, it supports a wide range of diverse resolutions from 336 pixels to 4K standard, significantly broadening its scope of applicability. Specifically, this research advances the patch division paradigm by introducing a novel extension: dynamic resolution with automatic patch configuration. It maintains the training image aspect ratios while automatically varying patch counts and configuring layouts based on a pre-trained Vision Transformer (ViT) (336 $\times$ 336), leading to dynamic training resolution from 336 pixels to 4K standard. Our research demonstrates that scaling training resolution up to 4K HD leads to consistent performance enhancements without hitting the ceiling of potential improvements. InternLM-XComposer2-4KHD shows superb capability that matches or even surpasses GPT-4V and Gemini Pro in 10 of the 16 benchmarks.

</details>

---

## 51. Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning

- [ ] Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning | https://neurips.cc/virtual/2024/poster/93706

- **Link**: https://neurips.cc/virtual/2024/poster/93706

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) fine-tuned on specialized visual instruction-following data have exhibited impressive language reasoning capabilities across various scenarios. However, this fine-tuning paradigm may not be able to efficiently learn optimal decision-making agents in multi-step goal-directed tasks from interactive environments. To address this challenge, we propose an algorithmic framework that fine-tunes VLMs with reinforcement learning (RL). Specifically, our framework provides a task description and then prompts the VLM to generate chain-of-thought (CoT) reasoning, enabling the VLM to efficiently explore intermediate reasoning steps that lead to the final text-based action. Next, the open-ended text output is parsed into an executable action to interact with the environment to obtain goal-directed task rewards. Finally, our framework uses these task rewards to fine-tune the entire VLM with RL. Empirically, we demonstrate that our proposed framework enhances the decision-making capabilities of VLM agents across various tasks, enabling 7b models to outperform commercial models such as GPT4-V or Gemini. Furthermore, we find that CoT reasoning is a crucial component for performance improvement, as removing the CoT reasoning results in a significant decrease in the overall performance of our method.

</details>

---

## 52. Few-Shot Adversarial Prompt Learning on Vision-Language Models

- [ ] Few-Shot Adversarial Prompt Learning on Vision-Language Models | https://neurips.cc/virtual/2024/poster/93713

- **Link**: https://neurips.cc/virtual/2024/poster/93713

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The vulnerability of deep neural networks to imperceptible adversarial perturbations has attracted widespread attention. Inspired by the success of vision-language foundation models, previous efforts achieved zero-shot adversarial robustness by aligning adversarial visual features with text supervision. However, in practice, they are still unsatisfactory due to several issues, including heavy adaptation cost, suboptimal text supervision, and uncontrolled natural generalization capacity. In this paper, to address these issues, we propose a few-shot adversarial prompt framework where adapting input sequences with limited data makes significant adversarial robustness improvement. Specifically, we achieve this by providing adversarially correlated text supervision that is end-to-end learned from adversarial examples. We also propose a novel training objective that enhances the consistency of multi-modal features while encourages differentiated uni-modal features between natural and adversarial examples. The proposed framework gives access to learn adversarial text supervision, which provides superior cross-modal adversarial alignment and matches state-of-the-art zero-shot adversarial robustness with only 1\% training data. Code is available at: https://github.com/lionel-w2/FAP.

</details>

---

## 53. SearchLVLMs: A Plug-and-Play Framework for Augmenting Large Vision-Language Models by Searching Up-to-Date Internet Knowledge

- [ ] SearchLVLMs: A Plug-and-Play Framework for Augmenting Large Vision-Language Models by Searching Up-to-Date Internet Knowledge | https://neurips.cc/virtual/2024/poster/93813

- **Link**: https://neurips.cc/virtual/2024/poster/93813

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) are ignorant of the up-to-date knowledge, such as LLaVA series, because they cannot be updated frequently due to the large amount of resources required, and therefore fail in many cases. For example, if a LVLM was released on January 2024, and it wouldn't know the singer of the theme song for the new Detective Conan movie, which wasn't released until April 2024. To solve the problem, a promising solution motivated by retrieval-augmented generation (RAG) is to provide LVLMs with up-to-date knowledge via internet search during inference, i.e., internet-augmented generation (IAG), which is already integrated in some closed-source commercial LVLMs such as GPT-4V. However, the specific mechanics underpinning them remain a mystery. In this paper, we propose a plug-and-play framework, for augmenting existing LVLMs in handling visual question answering (VQA) about up-to-date knowledge, dubbed SearchLVLMs. A hierarchical filtering model is trained to effectively and efficiently find the most helpful content from the websites returned by a search engine to prompt LVLMs with up-to-date knowledge. To train the model and evaluate our framework's performance, we propose a pipeline to automatically generate news-related VQA samples to construct a dataset, dubbed UDK-VQA. A multi-model voting mechanism is introduced to label the usefulness of website/content for VQA samples to construct the training set. Experimental results demonstrate the effectiveness of our framework, outperforming GPT-4o by $\sim$30\% in accuracy.

</details>

---

## 54. Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration

- [ ] Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration | https://neurips.cc/virtual/2024/poster/93873

- **Link**: https://neurips.cc/virtual/2024/poster/93873

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The swift advancement in Multimodal LLMs (MLLMs) also presents significant challenges for effective knowledge editing. Current methods, including intrinsic knowledge editing and external knowledge resorting, each possess strengths and weaknesses, struggling to balance the desired properties of reliability, generality, and locality when applied to MLLMs. In this paper, we propose \textbf{UniKE}, a novel multimodal editing method that establishes a unified perspective and paradigm for intrinsic knowledge editing and external knowledge resorting. Both types of knowledge are conceptualized as vectorized key-value memories, with the corresponding editing processes resembling the assimilation and accommodation phases of human cognition, conducted at the same semantic levels.  Within such a unified framework, we further promote knowledge collaboration by disentangling the knowledge representations into the semantic and truthfulness spaces. Extensive experiments validate the effectiveness of our method, which ensures that the post-edit MLLM simultaneously maintains excellent reliability, generality, and locality. The code for UniKE is available at https://github.com/beepkh/UniKE.

</details>

---

## 55. Vitron: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing

- [ ] Vitron: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing | https://neurips.cc/virtual/2024/poster/93896

- **Link**: https://neurips.cc/virtual/2024/poster/93896

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent developments of vision large language models (LLMs) have seen remarkable progress, yet still encounter challenges towards multimodal generalists, such as coarse-grained instance-level understanding, lack of unified support for both images and videos, and insufficient coverage across various vision tasks. In this paper we present Vitron, a universal pixel-level vision LLM designed for comprehensive understanding, generating, segmenting, and editing of both static images and dynamic videos. Building on top of an LLM backbone, Vitron incorporates encoders for images, videos, and pixel-level regional visuals within its frontend modules, while employing state-of-the-art visual specialists as its backend, via which Vitron supports a spectrum of vision end tasks, spanning visual comprehension to visual generation, from low level to high level. To ensure an effective and precise message passing from LLM to backend modules for function invocation, we propose a novel hybrid method by simultaneously integrating discrete textual instructions and continuous signal embeddings. Further, we design various pixel-level spatiotemporal vision-language alignment learning for Vitron to reach the best fine-grained visual capability. Finally, a cross-task synergy module is advised to learn to maximize the task-invariant fine-grained visual features, enhancing the synergy between different visual tasks. Demonstrated over 12 visual tasks and evaluated across 22 datasets, Vitron showcases its extensive capabilities in the four main vision task clusters. Overall, this work illuminates the great potential of developing a more unified multimodal generalist.

</details>

---

## 56. Towards Flexible Visual Relationship Segmentation

- [ ] Towards Flexible Visual Relationship Segmentation | https://neurips.cc/virtual/2024/poster/93908

- **Link**: https://neurips.cc/virtual/2024/poster/93908

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual relationship understanding has been studied separately in human-object interaction(HOI) detection, scene graph generation(SGG),  and referring relationships(RR) tasks. Given the complexity and interconnectedness of these tasks, it is crucial to have a flexible framework that can effectively address these tasks in a cohesive manner.In this work, we propose FleVRS, a single model that seamlessly integrates the above three aspects in standard and promptable visual relationship segmentation, and further possesses the capability for open-vocabulary segmentation to adapt to novel scenarios. FleVRS leverages the synergy between text and image modalities, to ground various types of relationships from images and use textual features from vision-language models to visual conceptual understanding.Empirical validation across various datasets demonstrates that our framework outperforms existing models in standard, promptable, and open-vocabulary tasks, e.g., +1.9 $mAP$ on HICO-DET, +11.4 $Acc$ on VRD,  +4.7 $mAP$ on unseen HICO-DET.Our FleVRS represents a significant step towards a more intuitive, comprehensive, and scalable understanding of visual relationships.

</details>

---

## 57. Dual Prototype Evolving for Test-Time Generalization of Vision-Language Models

- [ ] Dual Prototype Evolving for Test-Time Generalization of Vision-Language Models | https://neurips.cc/virtual/2024/poster/93929

- **Link**: https://neurips.cc/virtual/2024/poster/93929

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation, which enables models to generalize to diverse data with unlabeled test samples, holds significant value in real-world scenarios. Recently, researchers have applied this setting to advanced pre-trained vision-language models (VLMs), developing approaches such as test-time prompt tuning to further extend their practical applicability. However, these methods typically focus solely on adapting VLMs from a single modality and fail to accumulate task-specific knowledge as more samples are processed. To address this, we introduce Dual Prototype Evolving (DPE), a novel test-time adaptation approach for VLMs that effectively accumulates task-specific knowledge from multi-modalities. Specifically, we create and evolve two sets of prototypes—textual and visual—to progressively capture more accurate multi-modal representations for target classes during test time. Moreover, to promote consistent multi-modal representations, we introduce and optimize learnable residuals for each test sample to align the prototypes from both modalities. Extensive experimental results on 15 benchmark datasets demonstrate that our proposed DPE consistently outperforms previous state-of-the-art methods while also exhibiting competitive computational efficiency.

</details>

---

## 58. Towards Neuron Attributions in Multi-Modal Large Language Models

- [ ] Towards Neuron Attributions in Multi-Modal Large Language Models | https://neurips.cc/virtual/2024/poster/93962

- **Link**: https://neurips.cc/virtual/2024/poster/93962

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As Large Language Models (LLMs) demonstrate impressive capabilities, demystifying their internal mechanisms becomes increasingly vital. Neuron attribution, which attributes LLM outputs to specific neurons to reveal the semantic properties they learn, has emerged as a key interpretability approach. However, while neuron attribution has made significant progress in deciphering text-only LLMs, its application to Multimodal LLMs (MLLMs) remains less explored. To address this gap, we propose a novel Neuron Attribution method tailored for MLLMs, termed NAM. Specifically, NAM not only reveals the modality-specific semantic knowledge learned by neurons within MLLMs, but also highlights several intriguing properties of neurons, such as cross-modal invariance and semantic sensitivity. These properties collectively elucidate the inner workings mechanism of MLLMs, providing a deeper understanding of how MLLMs process and generate multi-modal content. Through theoretical analysis and empirical validation, we demonstrate the efficacy of NAM and the valuable insights it offers. Furthermore, leveraging NAM, we introduce a multi-modal knowledge editing paradigm, underscoring the practical significance of our approach for downstream applications of MLLMs.

</details>

---

## 59. Lumina-Next : Making Lumina-T2X Stronger and Faster with Next-DiT

- [ ] Lumina-Next : Making Lumina-T2X Stronger and Faster with Next-DiT | https://neurips.cc/virtual/2024/poster/93994

- **Link**: https://neurips.cc/virtual/2024/poster/93994

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Lumina-T2X is a nascent family of Flow-based Large Diffusion Transformers (Flag-DiT) that establishes a unified framework for transforming noise into various modalities, such as images and videos, conditioned on text instructions. Despite its promising capabilities, Lumina-T2X still encounters challenges including training instability, slow inference, and extrapolation artifacts. In this paper, we present Lumina-Next, an improved version of Lumina-T2X, showcasing stronger generation performance with increased training and inference efficiency. We begin with a comprehensive analysis of the Flag-DiT architecture and identify several suboptimal components, which we address by introducing the Next-DiT architecture with 3D RoPE and sandwich normalizations. To enable better resolution extrapolation, we thoroughly compare different context extrapolation methods applied to text-to-image generation with 3D RoPE, and propose Frequency- and Time-Aware Scaled RoPE tailored for diffusion transformers. Additionally, we introduce a sigmoid time discretization schedule for diffusion sampling, which achieves high-quality generation in 5-10 steps combined with higher-order ODE solvers. Thanks to these improvements, Lumina-Next not only improves the basic text-to-image generation but also demonstrates superior resolution extrapolation capabilities as well as multilingual generation using decoder-based LLMs as the text encoder, all in a zero-shot manner. To further validate Lumina-Next as a versatile generative framework, we instantiate it on diverse tasks including visual recognition, multi-views, audio, music, and point cloud generation, showcasing strong performance across these domains. By releasing all codes and model weights at https://github.com/Alpha-VLLM/Lumina-T2X, we aim to advance the development of next-generation generative AI capable of universal modeling.

</details>

---

## 60. CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

- [ ] CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts | https://neurips.cc/virtual/2024/poster/94037

- **Link**: https://neurips.cc/virtual/2024/poster/94037

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (LLMs) have focused primarily on scaling by increasing text-image pair data and enhancing LLMs to improve performance on multimodal tasks. However, these scaling approaches are computationally expensive and overlook the significance of efficiently improving model capabilities from the vision side. Inspired by the successful applications of Mixture-of-Experts (MoE) in LLMs, which improves model scalability during training while keeping inference costs similar to those of smaller models, we propose CuMo, which incorporates Co-upcycled Top-K sparsely-gated Mixture-of-experts blocks into both the vision encoder and the MLP connector, thereby enhancing the multimodal LLMs with neglectable additional activated parameters during inference.CuMo first pre-trains the MLP blocks and then initializes each expert in the MoE block from the pre-trained MLP block during the visual instruction tuning stage, with auxiliary losses to ensure a balanced loading of experts.CuMo outperforms state-of-the-art multimodal LLMs across various VQA and visual-instruction-following benchmarks within each model size group, all while training exclusively on open-sourced datasets.

</details>

---

## 61. DA-Ada: Learning Domain-Aware Adapter for Domain Adaptive Object Detection

- [ ] DA-Ada: Learning Domain-Aware Adapter for Domain Adaptive Object Detection | https://neurips.cc/virtual/2024/poster/94044

- **Link**: https://neurips.cc/virtual/2024/poster/94044

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Domain adaptive object detection (DAOD) aims to generalize detectors trained on an annotated source domain to an unlabelled target domain.As the visual-language models (VLMs) can provide essential general knowledge on unseen images, freezing the visual encoder and inserting a domain-agnostic adapter can learn domain-invariant knowledge for DAOD.However, the domain-agnostic adapter is inevitably biased to the source domain.It discards some beneficial knowledge discriminative on the unlabelled domain, \ie domain-specific knowledge of the target domain.To solve the issue, we propose a novel Domain-Aware Adapter (DA-Ada) tailored for the DAOD task.The key point is exploiting domain-specific knowledge between the essential general knowledge and domain-invariant knowledge.DA-Ada consists of the Domain-Invariant Adapter (DIA) for learning domain-invariant knowledge and the Domain-Specific Adapter (DSA) for injecting the domain-specific knowledge from the information discarded by the visual encoder.Comprehensive experiments over multiple DAOD tasks show that DA-Ada can efficiently infer a domain-aware visual encoder for boosting domain adaptive object detection.Our code is available at https://github.com/Therock90421/DA-Ada.

</details>

---

## 62. Vision-Language Models are Strong Noisy Label Detectors

- [ ] Vision-Language Models are Strong Noisy Label Detectors | https://neurips.cc/virtual/2024/poster/94056

- **Link**: https://neurips.cc/virtual/2024/poster/94056

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent research on fine-tuning vision-language models has demonstrated impressive performance in various downstream tasks. However, the challenge of obtaining accurately labeled data in real-world applications poses a significant obstacle during the fine-tuning process. To address this challenge, this paper presents a Denoising Fine-Tuning framework, called DeFT, for adapting vision-language models. DeFT utilizes the robust alignment of textual and visual features pre-trained on millions of auxiliary image-text pairs to sieve out noisy labels. The proposed framework establishes a noisy label detector by learning positive and negative textual prompts for each class. The positive prompt seeks to reveal distinctive features of the class, while the negative prompt serves as a learnable threshold for separating clean and noisy samples. We employ parameter-efficient fine-tuning for the adaptation of a pre-trained visual encoder to promote its alignment with the learned textual prompts. As a general framework, DeFT can seamlessly fine-tune many pre-trained models to downstream tasks by utilizing carefully selected clean samples. Experimental results on seven synthetic and real-world noisy datasets validate the effectiveness of DeFT in both noisy label detection and image classification. Our source code can be found in the supplementary material.

</details>

---

## 63. GraphVis: Boosting LLMs with Visual Knowledge Graph Integration

- [ ] GraphVis: Boosting LLMs with Visual Knowledge Graph Integration | https://neurips.cc/virtual/2024/poster/94055

- **Link**: https://neurips.cc/virtual/2024/poster/94055

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid evolution of large language models (LLMs) has expanded their capabilities across various data modalities, extending from well-established image data to increasingly popular graph data. Given the limitation of LLMs in hallucinations and inaccuracies in recalling factual knowledge, Knowledge Graph (KG) has emerged as a crucial data modality to support more accurate reasoning by LLMs. However, integrating structured knowledge from KGs into LLMs remains challenging, as most current KG-enhanced LLM methods directly convert the KG into linearized text triples, which is not as expressive as the original structured data. To address this, we introduce GraphVis, which conserves the intricate graph structure through the visual modality to enhance the comprehension of KGs with the aid of Large Vision Language Models (LVLMs). Our approach incorporates a unique curriculum fine-tuning scheme which first instructs LVLMs to recognize basic graphical features from the images, and subsequently incorporates reasoning on QA tasks with the visual graphs. This cross-modal methodology not only markedly enhances performance on standard textual QA  but also shows improved zero-shot VQA performance by utilizing synthetic graph images to augment the data for VQA tasks. We present comprehensive evaluations across commonsense reasoning QA benchmarks, where GraphVis provides an average improvement of 11.1% over its base model and outperforms existing KG-enhanced LLM approaches. Across VQA benchmarks such as ScienceQA that share similar scientific diagram images, GraphVis provides a notable gain of 4.32%.

</details>

---

## 64. KptLLM: Unveiling the Power of Large Language Model for Keypoint Comprehension

- [ ] KptLLM: Unveiling the Power of Large Language Model for Keypoint Comprehension | https://neurips.cc/virtual/2024/poster/94108

- **Link**: https://neurips.cc/virtual/2024/poster/94108

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have greatly improved their abilities in image understanding. However, these models often struggle with grasping pixel-level semantic details, e.g., the keypoints of an object. To bridge this gap, we introduce the novel challenge of Semantic Keypoint Comprehension, which aims to comprehend keypoints across different task scenarios, including keypoint semantic understanding, visual prompt-based keypoint detection, and textual prompt-based keypoint detection. Moreover, we introduce KptLLM, a unified multimodal model that utilizes an identify-then-detect strategy to effectively address these challenges. KptLLM underscores the initial discernment of semantics in keypoints, followed by the precise determination of their positions through a chain-of-thought process. With several carefully designed modules, KptLLM adeptly handles various modality inputs, facilitating the interpretation of both semantic contents and keypoint locations. Our extensive experiments demonstrate KptLLM's superiority in various keypoint detection benchmarks and its unique semantic capabilities in interpreting keypoints.

</details>

---

## 65. Q-VLM: Post-training Quantization for Large Vision-Language Models

- [ ] Q-VLM: Post-training Quantization for Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/94107

- **Link**: https://neurips.cc/virtual/2024/poster/94107

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose a post-training quantization framework of large vision-language models (LVLMs) for efficient multi-modal inference. Conventional quantization methods sequentially search the layer-wise rounding functions by minimizing activation discretization errors, which fails to acquire optimal quantization strategy without considering cross-layer dependency. On the contrary, we mine the cross-layer dependency that significantly influences discretization errors of the entire vision-language model, and embed this dependency into optimal quantization strategy searching with low search cost. Specifically, we observe the strong correlation between the activation entropy and the cross-layer dependency concerning output discretization errors. Therefore, we employ the entropy as the proxy to partition blocks optimally, which aims to achieve satisfying trade-offs between discretization errors and the search cost. Moreover, we optimize the visual encoder to disentangle the cross-layer dependency for fine-grained decomposition of search space, so that the search cost is further reduced without harming the quantization accuracy. Experimental results demonstrate that our method compresses the memory by 2.78x and increase generate speed by 1.44x about 13B LLaVA model without performance degradation on diverse multi-modal reasoning tasks.

</details>

---

## 66. Boosting Vision-Language Models with Transduction

- [ ] Boosting Vision-Language Models with Transduction | https://neurips.cc/virtual/2024/poster/94116

- **Link**: https://neurips.cc/virtual/2024/poster/94116

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Transduction is a powerful paradigm that leverages the structure of unlabeled data to boost predictive accuracy. We present TransCLIP, a novel and computationally efficient transductive approach designed for Vision-Language Models (VLMs). TransCLIP is applicable as a plug-and-play module on top of popular inductive zero- and few-shot models, consistently improving their performances. Our new objective function can be viewed as a regularized maximum-likelihood estimation, constrained by a KL divergence penalty that integrates the text-encoder knowledge and guides the transductive learning process. We further derive an iterative Block Majorize-Minimize (BMM) procedure for optimizing our objective, with guaranteed convergence and decoupled sample-assignment updates, yielding computationally efficient transduction for large-scale datasets. We report comprehensive evaluations, comparisons, and ablation studies that demonstrate: (i) Transduction can greatly enhance the generalization capabilities of inductive pretrained zero- and few-shot VLMs; (ii) TransCLIP substantially outperforms standard transductive few-shot learning methods relying solely on vision features, notably due to the KL-based language constraint.

</details>

---

## 67. Biologically Inspired Learning Model for Instructed Vision

- [ ] Biologically Inspired Learning Model for Instructed Vision | https://neurips.cc/virtual/2024/poster/94152

- **Link**: https://neurips.cc/virtual/2024/poster/94152

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As part of the effort to understand how the brain learns, ongoing research seeks to combine biological knowledge with current artificial intelligence (AI) modeling in an attempt to find an efficient biologically plausible learning scheme. Current models often use a cortical-like combination of bottom-up (BU) and top-down (TD) processing, where the TD part carries feedback signals for learning. However, in the visual cortex, the TD pathway plays a second major role in visual attention, by guiding the visual process toward locations and tasks of interest. A biological model should therefore integrate both learning and visual guidance. We introduce a model that uses a cortical-like combination of BU and TD processing that naturally integrates the two major functions of the TD stream. This integration is achieved through an appropriate connectivity pattern between the BU and TD streams, a novel processing cycle that uses the TD stream twice, and a 'Counter-Hebb' learning mechanism that operates across both streams. We show that the 'Counter-Hebb' mechanism can provide an exact backpropagation synaptic modification. Additionally, our model can effectively guide the visual stream to perform a task of interest, achieving competitive performance on standard multi-task learning benchmarks compared to AI models. The successful combination of learning and visual guidance could provide a new view on combining BU and TD processing in human vision and suggests possible directions for both biologically plausible models and artificial instructed models, such as vision-language models (VLMs).

</details>

---

## 68. Pandora's Box: Towards Building Universal Attackers against Real-World Large Vision-Language Models

- [ ] Pandora's Box: Towards Building Universal Attackers against Real-World Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/94158

- **Link**: https://neurips.cc/virtual/2024/poster/94158

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding tasks. Nevertheless, these models are susceptible to adversarial examples. In real-world applications, existing LVLM attackers generally rely on the detailed prior knowledge of the model to generate effective perturbations. Moreover, these attacks are task-specific, leading to significant costs for designing perturbation. Motivated by the research gap and practical demands, in this paper, we make the first attempt to build a universal attacker against real-world LVLMs, focusing on two critical aspects: (i) restricting access to only the LVLM inputs and outputs. (ii) devising a universal adversarial patch, which is task-agnostic and can deceive any LVLM-driven task when applied to various inputs. Specifically, we start by initializing the location and the pattern of the adversarial patch through random sampling, guided by the semantic distance between their output and the target label. Subsequently, we maintain a consistent patch location while refining the pattern to enhance semantic resemblance to the target. In particular, our approach incorporates a diverse set of LVLM task inputs as query samples to approximate the patch gradient, capitalizing on the importance of distinct inputs. In this way, the optimized patch is universally adversarial against different tasks and prompts, leveraging solely gradient estimates queried from the model. Extensive experiments are conducted to verify the strong universal adversarial capabilities of our proposed attack with prevalent LVLMs including LLaVA, MiniGPT-4, Flamingo, and BLIP-2, spanning a spectrum of tasks, all achieved without delving into the details of the model structures.

</details>

---

## 69. Harmonizing Visual Text Comprehension and Generation

- [ ] Harmonizing Visual Text Comprehension and Generation | https://neurips.cc/virtual/2024/poster/94183

- **Link**: https://neurips.cc/virtual/2024/poster/94183

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we present TextHarmony, a unified and versatile multimodal generative model proficient in comprehending and generating visual text. Simultaneously generating images and texts typically results in performance degradation due to the inherent inconsistency between vision and language modalities. To overcome this challenge, existing approaches resort to modality-specific data for supervised fine-tuning, necessitating distinct model instances. We propose Slide-LoRA, which dynamically aggregates modality-specific and modality-agnostic LoRA experts, partially decoupling the multimodal generation space. Slide-LoRA harmonizes the generation of vision and language within a singular model instance, thereby facilitating a more unified generative process. Additionally, we develop a high-quality image caption dataset, DetailedTextCaps-100K, synthesized with a sophisticated closed-source MLLM to enhance visual text generation capabilities further. Comprehensive experiments across various benchmarks demonstrate the effectiveness of the proposed approach. Empowered by Slide-LoRA, TextHarmony achieves comparable performance to modality-specific fine-tuning results with only a 2% increase in parameters and shows an average improvement of 2.5% in visual text comprehension tasks and 4.0% in visual text generation tasks. Our work delineates the viability of an integrated approach to multimodal generation within the visual text domain, setting a foundation for subsequent inquiries. Code is available at https://github.com/bytedance/TextHarmony.

</details>

---

## 70. Exploiting Descriptive Completeness Prior for Cross Modal Hashing with Incomplete Labels

- [ ] Exploiting Descriptive Completeness Prior for Cross Modal Hashing with Incomplete Labels | https://neurips.cc/virtual/2024/poster/94194

- **Link**: https://neurips.cc/virtual/2024/poster/94194

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we tackle the challenge of generating high-quality hash codes for cross-modal retrieval in the presence of incomplete labels, which creates uncertainty in distinguishing between positive and negative pairs. Vision-language models such as CLIP offer a potential solution by providing generic knowledge for missing label recovery, yet their zero-shot performance remains insufficient. To address this, we propose a novel Prompt Contrastive Recovery approach, \textbf{PCRIL}, which progressively identifies promising positive classes from unknown label sets and recursively searches for other relevant labels. Identifying unknowns is nontrivial due to the fixed and long-tailed patterns of positive label sets in training data, which hampers the discovery of new label combinations. Therefore, we consider each subset of positive labels and construct three types of negative prompts through deletion, addition, and replacement for prompt learning. The augmented supervision guides the model to measure the completeness of label sets, thus facilitating the subsequent greedy tree search for label completion. We also address extreme cases of significant unknown labels and lack of negative pairwise supervision by deriving two augmentation strategies: seeking unknown-complementary samples for mixup and random flipping for negative labels. Extensive experiments reveal the vulnerability of current methods and demonstrate the effectiveness of PCRIL, achieving an average 12\% mAP improvement to the current SOTA across all datasets. Our code is available at https://github.com/E-Galois/PCRIL.

</details>

---

## 71. MaVEn: An Effective Multi-granularity Hybrid Visual Encoding Framework for Multimodal Large Language Model

- [ ] MaVEn: An Effective Multi-granularity Hybrid Visual Encoding Framework for Multimodal Large Language Model | https://neurips.cc/virtual/2024/poster/94216

- **Link**: https://neurips.cc/virtual/2024/poster/94216

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents MaVEn, an innovative Multi-granularity Visual Encoding framework designed to enhance the capabilities of Multimodal Large Language Models (MLLMs) in multi-image reasoning. Current MLLMs primarily focus on single-image visual understanding, limiting their ability to interpret and integrate information across multiple images. MaVEn addresses this limitation by combining discrete visual symbol sequences, which abstract coarse-grained semantic concepts, with traditional continuous representation sequences that model fine-grained features. This dual approach bridges the semantic gap between visual and textual data, thereby improving the model's ability to process and interpret information from multiple images effectively. Additionally, we design a dynamic reduction mechanism by for long-sequence continuous features to enhance multi-image processing efficiency. Experimental results demonstrate that MaVEn significantly enhances MLLMs' understanding in complex multi-image scenarios, while also improving performance in single-image contexts.

</details>

---

## 72. Are We on the Right Way for Evaluating Large Vision-Language Models?

- [ ] Are We on the Right Way for Evaluating Large Vision-Language Models? | https://neurips.cc/virtual/2024/poster/94237

- **Link**: https://neurips.cc/virtual/2024/poster/94237

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have recently achieved rapid progress, sparking numerous studies to evaluate their multi-modal capabilities. However, we dig into current evaluation works and identify two primary issues: 1) Visual content is unnecessary for many samples. The answers can be directly inferred from the questions and options, or the world knowledge embedded in LLMs. This phenomenon is prevalent across current benchmarks. For instance, GeminiPro achieves 42.7% on the MMMU benchmark without any visual input, and outperforms the random choice baseline across six benchmarks near 24% on average. 2) Unintentional data leakage exists in LLM and LVLM training. LLM and LVLM could still answer some visual-necessary questions without visual content, indicating the memorizing of these samples within large-scale training data. For example, Sphinx-X-MoE gets 43.6% on MMMU without accessing images, surpassing its LLM backbone with 17.9%. Both problems lead to misjudgments of actual multi-modal gains and potentially misguide the study of LVLM. To this end, we present MMStar, an elite vision-indispensable multi-modal benchmark comprising 1,500 samples meticulously selected by humans. MMStar benchmarks 6 core capabilities and 18 detailed axes, aiming to evaluate LVLMs' multi-modal capacities with carefully balanced and purified samples. These samples are first roughly selected from current benchmarks with an automated pipeline, human review is then involved to ensure each curated sample exhibits visual dependency, minimal data leakage, and requires advanced multi-modal capabilities. Moreover, two metrics are developed to measure data leakage and actual performance gain in multi-modal training. We evaluate 16 leading LVLMs on MMStar to assess their multi-modal capabilities, and on 7 benchmarks with the proposed metrics to investigate their data leakage and actual multi-modal gain.

</details>

---

## 73. Frustratingly Easy Test-Time Adaptation of Vision-Language Models

- [ ] Frustratingly Easy Test-Time Adaptation of Vision-Language Models | https://neurips.cc/virtual/2024/poster/94270

- **Link**: https://neurips.cc/virtual/2024/poster/94270

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models seamlessly discriminate among arbitrary semantic categories, yet they still suffer from poor generalization when presented with challenging examples. For this reason, Episodic Test-Time Adaptation (TTA) strategies have recently emerged as powerful techniques to adapt VLMs in the presence of a single unlabeled image. The recent literature on TTA is dominated by the paradigm of prompt tuning by Marginal Entropy Minimization, which, relying on online backpropagation, inevitably slows down inference while increasing memory. In this work, we theoretically investigate the properties of this approach and unveil that a surprisingly strong TTA method lies dormant and hidden within it. We term this approach ZERO (TTA with “zero” temperature), whose design is both incredibly effective and frustratingly simple: augment N times, predict, retain the most confident predictions, and marginalize after setting the Softmax temperature to zero. Remarkably, ZERO requires a single batched forward pass through the vision encoder only and no backward passes. We thoroughly evaluate our approach following the experimental protocol established in the literature and show that ZERO largely surpasses or compares favorably w.r.t. the state-of-the-art while being almost 10× faster and 13× more memory friendly than standard Test-Time Prompt Tuning. Thanks to its simplicity and comparatively negligible computation, ZERO can serve as a strong baseline for future work in this field. Code will be available.

</details>

---

## 74. PLIP: Language-Image Pre-training for Person Representation Learning

- [ ] PLIP: Language-Image Pre-training for Person Representation Learning | https://neurips.cc/virtual/2024/poster/94298

- **Link**: https://neurips.cc/virtual/2024/poster/94298

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Language-image pre-training is an effective technique for learning powerful representations in general domains. However, when directly turning to person representation learning, these general pre-training methods suffer from unsatisfactory performance. The reason is that they neglect critical person-related characteristics, i.e., fine-grained attributes and identities. To address this issue, we propose a novel language-image pre-training framework for person representation learning, termed PLIP. Specifically, we elaborately design three pretext tasks: 1) Text-guided Image Colorization, aims to establish the correspondence between the person-related image regions and the fine-grained color-part textual phrases. 2) Image-guided Attributes Prediction, aims to mine fine-grained attribute information of the person body in the image; and 3) Identity-based Vision-Language Contrast, aims to correlate the cross-modal representations at the identity level rather than the instance level. Moreover, to implement our pre-train framework, we construct a large-scale person dataset with image-text pairs named SYNTH-PEDES by automatically generating textual annotations. We pre-train PLIP on SYNTH-PEDES and evaluate our models by spanning downstream person-centric tasks. PLIP not only significantly improves existing methods on all these tasks, but also shows great ability in the zero-shot and domain generalization settings. The code, dataset and weight will be made publicly available.

</details>

---

## 75. Free Lunch in Pathology Foundation Model: Task-specific Model Adaptation with Concept-Guided Feature Enhancement

- [ ] Free Lunch in Pathology Foundation Model: Task-specific Model Adaptation with Concept-Guided Feature Enhancement | https://neurips.cc/virtual/2024/poster/94308

- **Link**: https://neurips.cc/virtual/2024/poster/94308

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Whole slide image (WSI) analysis is gaining prominence within the medical imaging field. Recent advances in pathology foundation models have shown the potential to extract powerful feature representations from WSIs for downstream tasks. However, these foundation models are usually designed for general-purpose pathology image analysis and may not be optimal for specific downstream tasks or cancer types. In this work, we present Concept Anchor-guided Task-specific Feature Enhancement (CATE), an adaptable paradigm that can boost the expressivity and discriminativeness of pathology foundation models for specific downstream tasks. Based on a set of task-specific concepts derived from the pathology vision-language model with expert-designed prompts, we introduce two interconnected modules to dynamically calibrate the generic image features extracted by foundation models for certain tasks or cancer types. Specifically, we design a Concept-guided Information Bottleneck module to enhance task-relevant characteristics by maximizing the mutual information between image features and concept anchors while suppressing superfluous information. Moreover, a Concept-Feature Interference module is proposed to utilize the similarity between calibrated features and concept anchors to further generate discriminative task-specific features. The extensive experiments on public WSI datasets demonstrate that CATE significantly enhances the performance and generalizability of MIL models. Additionally, heatmap and umap visualization results also reveal the effectiveness and interpretability of CATE.

</details>

---

## 76. What matters when building vision-language models?

- [ ] What matters when building vision-language models? | https://neurips.cc/virtual/2024/poster/94309

- **Link**: https://neurips.cc/virtual/2024/poster/94309

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The growing interest in vision-language models (VLMs) has been driven by improvements in large language models and vision transformers. Despite the abundance of literature on this subject, we observe that critical decisions regarding the design of VLMs are often not justified. We argue that these unsupported decisions impede progress in the field by making it difficult to identify which choices improve model performance. To address this issue, we conduct extensive experiments around pre-trained models, architecture choice, data, and training methods. Our consolidation of findings includes the development of Idefics2, an efficient foundational VLM of 8 billion parameters. Idefics2 achieves state-of-the-art performance within its size category across various multimodal benchmarks, and is often on par with models four times its size. We release the model (base, instructed, and chat) along with the datasets created for its training.

</details>

---

## 77. LLMs Can Evolve Continually on Modality for $\mathbb{X}$-Modal Reasoning

- [ ] LLMs Can Evolve Continually on Modality for $\mathbb{X}$-Modal Reasoning | https://neurips.cc/virtual/2024/poster/94313

- **Link**: https://neurips.cc/virtual/2024/poster/94313

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have gained significant attention due to their impressive capabilities in multimodal understanding. However, existing methods rely heavily on extensive modal-specific pretraining and joint-modal tuning, leading to significant computational burdens when expanding to new modalities. In this paper, we propose \textbf{PathWeave}, a flexible and scalable framework with modal-\textbf{path} s\textbf{w}itching and \textbf{e}xp\textbf{a}nsion abilities that enables MLLMs to continually \textbf{ev}olve on modalities for $\mathbb{X}$-modal reasoning. We leverage the concept of Continual Learning and develop an incremental training strategy atop pre-trained MLLMs, enabling their expansion to new modalities using uni-modal data, without executing joint-modal pretraining. In detail, a novel Adapter-in-Adapter (AnA) framework is introduced, in which uni-modal and cross-modal adapters are seamlessly integrated to facilitate efficient modality alignment and collaboration. Additionally, an MoE-based gating module is applied between two types of adapters to further enhance the multimodal interaction. To investigate the proposed method, we establish a challenging benchmark called \textbf{C}ontinual \textbf{L}earning of \textbf{M}odality (MCL), which consists of high-quality QA data from five distinct modalities: image, video, \textcolor{black}{audio, depth} and point cloud. Extensive experiments demonstrate the effectiveness of the proposed AnA framework on learning plasticity and memory stability during continual learning. Furthermore, PathWeave performs comparably to state-of-the-art MLLMs while concurrently reducing parameter training burdens by 98.73\%. Our code locates at \url{https://github.com/JiazuoYu/PathWeave}.

</details>

---

## 78. Domain Adaptation for Large-Vocabulary Object Detectors

- [ ] Domain Adaptation for Large-Vocabulary Object Detectors | https://neurips.cc/virtual/2024/poster/94330

- **Link**: https://neurips.cc/virtual/2024/poster/94330

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-vocabulary object detectors (LVDs) aim to detect objects of many categories, which learn super objectness features and can locate objects accurately while applied to various downstream data. However, LVDs often struggle in recognizing the located objects due to domain discrepancy in data distribution and object vocabulary. At the other end, recent vision-language foundation models such as CLIP demonstrate superior open-vocabulary recognition capability. This paper presents KGD, a Knowledge Graph Distillation technique that exploits the implicit knowledge graphs (KG) in CLIP for effectively adapting LVDs to various downstream domains.KGD consists of two consecutive stages: 1) KG extraction that employs CLIP to encode downstream domain data as nodes and their feature distances as edges, constructing KG that inherits the rich semantic relations in CLIP explicitly; and 2) KG encapsulation that transfers the extracted KG into LVDs to enable accurate cross-domain object classification. In addition, KGD can extract both visual and textual KG independently, providing complementary vision and language knowledge for object localization and object classification in detection tasks over various downstream domains. Experiments over multiple widely adopted detection benchmarks show that KGD outperforms the state-of-the-art consistently by large margins. Codes will be released.

</details>

---

## 79. UMFC: Unsupervised Multi-Domain Feature Calibration for Vision-Language Models

- [ ] UMFC: Unsupervised Multi-Domain Feature Calibration for Vision-Language Models | https://neurips.cc/virtual/2024/poster/94349

- **Link**: https://neurips.cc/virtual/2024/poster/94349

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (e.g., CLIP) have shown powerful zero-shot transfer capabilities. But they still struggle with domain shifts and typically require labeled data to adapt to downstream tasks, which could be costly. In this work, we aim to  leverage unlabeled data that naturally spans multiple domains to enhance the transferability of vision-language models.  Under this unsupervised multi-domain setting, we have identified inherent model bias within CLIP, notably  in its visual and text encoders. Specifically, we observe that CLIP’s visual encoder tends to prioritize  encoding domain over discriminative category information, meanwhile its text encoder exhibits a preference for domain-relevant classes. To mitigate this model bias, we propose a training-free and label-free feature calibration method, Unsupervised Multi-domain Feature Calibration (UMFC). UMFC estimates image-level biases from domain-specific features and text-level biases from the direction of domain transition. These biases are subsequently   subtracted from original image and text features separately, to render them domain-invariant. We evaluate our method on multiple settings including transductive learning and test-time adaptation. Extensive experiments show that our method outperforms CLIP and performs on par with the state-of-the-arts that need additional annotations or optimization.Our code is available at https://github.com/GIT-LJc/UMFC.

</details>

---

## 80. Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models

- [ ] Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models | https://neurips.cc/virtual/2024/poster/94371

- **Link**: https://neurips.cc/virtual/2024/poster/94371

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) and vision-language models (VLMs) have demonstrated remarkable performance across a wide range of tasks and domains. Despite this promise, spatial understanding and reasoning—a fundamental component of human cognition—remains under-explored. We propose SpatialEval, a novel benchmark that covers diverse aspects of spatial reasoning such as relationship understanding, navigation, and counting. We conduct a comprehensive evaluation of competitive language and vision-language models. Our findings reveal several counter-intuitive insights that have been overlooked in the literature: (1) Spatial reasoning poses significant challenges where competitive models can fall behind random guessing; (2) Despite additional visual input, VLMs often under-perform compared to their LLM counterparts; (3) When both textual and visual information is available, multi-modal language models become less reliant on visual information if sufficient textual clues are provided. Additionally, we demonstrate that leveraging redundancy between vision and text can significantly enhance model performance. We hope our study will inform the development of multimodal models to improve spatial intelligence and further close the gap with human intelligence. Our code is available at https://github.com/jiayuww/SpatialEval.

</details>

---

## 81. Delta-CoMe: Training-Free Delta-Compression with Mixed-Precision for Large Language Models

- [ ] Delta-CoMe: Training-Free Delta-Compression with Mixed-Precision for Large Language Models | https://neurips.cc/virtual/2024/poster/94378

- **Link**: https://neurips.cc/virtual/2024/poster/94378

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning is a crucial process for adapting large language models (LLMs) to diverse applications. In certain scenarios, such as multi-tenant serving, deploying multiple LLMs becomes necessary to meet complex demands. Recent studies suggest decomposing a fine-tuned LLM into a base model and corresponding delta weights, which are then compressed using low-rank or low-bit approaches to reduce costs. In this work, we observe that existing low-rank and low-bit compression methods can significantly harm the model performance for task-specific fine-tuned LLMs (e.g., WizardMath for math problems). Motivated by the long-tail distribution of singular values in the delta weights, we propose a delta quantization approach using mixed-precision. This method employs higher-bit representation for singular vectors corresponding to larger singular values. We evaluate our approach on various fine-tuned LLMs, including math LLMs, code LLMs, chat LLMs, and even VLMs. Experimental results demonstrate that our approach performs comparably to full fine-tuned LLMs, surpassing both low-rank and low-bit baselines by a considerable margin. Additionally, we show that our method is compatible with various backbone LLMs, such as Llama-2, Llama-3, and Mistral, highlighting its generalizability.

</details>

---

## 82. OmniJARVIS: Unified Vision-Language-Action Tokenization Enables Open-World Instruction Following Agents

- [ ] OmniJARVIS: Unified Vision-Language-Action Tokenization Enables Open-World Instruction Following Agents | https://neurips.cc/virtual/2024/poster/94404

- **Link**: https://neurips.cc/virtual/2024/poster/94404

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents OmniJARVIS, a novel Vision-Language-Action (VLA) model for open-world instruction-following agents in Minecraft. Compared to prior works that either emit textual goals to separate controllers or produce the control command directly, OmniJARVIS seeks a different path to ensure both strong reasoning and efficient decision-making capabilities via unified tokenization of multimodal interaction data. First, we introduce a self-supervised approach to learn a behavior encoder that produces discretized tokens for behavior trajectories $\tau = \{o_0, a_0, \dots\}$ and an imitation learning policy decoder conditioned on these tokens. These additional behavior tokens will be augmented to the vocabulary of pretrained Multimodal Language Models. With this encoder, we then pack long-term multimodal interactions involving task instructions, memories, thoughts, observations, textual responses, behavior trajectories, etc into unified token sequences and model them with autoregressive transformers. Thanks to the semantically meaningful behavior tokens, the resulting VLA model, OmniJARVIS, can reason (by producing chain-of-thoughts), plan, answer questions, and act (by producing behavior tokens for the imitation learning policy decoder). OmniJARVIS demonstrates excellent performances on a comprehensive collection of atomic, programmatic, and open-ended tasks in open-world Minecraft. Our analysis further unveils the crucial design principles in interaction data formation, unified tokenization, and its scaling potentials. The dataset, models, and code will be released at https://craftjarvis.org/OmniJARVIS.

</details>

---

## 83. Text-Guided Attention is All You Need for Zero-Shot Robustness in Vision-Language Models

- [ ] Text-Guided Attention is All You Need for Zero-Shot Robustness in Vision-Language Models | https://neurips.cc/virtual/2024/poster/94422

- **Link**: https://neurips.cc/virtual/2024/poster/94422

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Due to the impressive zero-shot capabilities, pre-trained vision-language models (e.g. CLIP), have attracted widespread attention and adoption across various domains. Nonetheless, CLIP has been observed to be susceptible to adversarial examples. Through experimental analysis, we have observed a phenomenon wherein adversarial perturbations induce shifts in text-guided attention. Building upon this observation, we propose a simple yet effective strategy: Text-Guided Attention for Zero-Shot Robustness (TGA-ZSR). This framework incorporates two components: the Attention Refinement module and the Attention-based Model Constraint module. Our goal is to maintain the generalization of the CLIP model and enhance its adversarial robustness: The Attention Refinement module aligns the text-guided attention obtained from the target model via adversarial examples with the text-guided attention acquired from the original model via clean examples. This alignment enhances the model’s robustness. Additionally, the Attention-based Model Constraint module acquires text-guided attention from both the target and original models using clean examples. Its objective is to maintain model performance on clean samples while enhancing overall robustness. The experiments validate that our method yields a 9.58% enhancement in zero-shot robust accuracy over the current state-of-the-art techniques across 16 datasets. Our code is available at https://github.com/zhyblue424/TGA-ZSR.

</details>

---

## 84. Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models

- [ ] Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models | https://neurips.cc/virtual/2024/poster/94460

- **Link**: https://neurips.cc/virtual/2024/poster/94460

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Continual learning (CL) with Vision-Language Models (VLMs) has overcome the constraints of traditional CL, which only focuses on previously encountered classes. During the CL of VLMs, we need not only to prevent the catastrophic forgetting on incrementally learned knowledge but also to preserve the zero-shot ability of VLMs. However, existing methods require additional reference datasets to maintain such zero-shot ability and rely on domain-identity hints to classify images across different domains. In this study, we propose Regression-based Analytic Incremental Learning (RAIL), which utilizes a recursive ridge regression-based adapter to learn from a sequence of domains in a non-forgetting manner and decouple the cross-domain correlations by projecting features to a higher-dimensional space. Cooperating with a training-free fusion module, RAIL absolutely preserves the VLM's zero-shot ability on unseen domains without any reference data.Additionally, we introduce Cross-domain Task-Agnostic Incremental Learning (X-TAIL) setting. In this setting, a CL learner is required to incrementally learn from multiple domains and classify test images from both seen and unseen domains without any domain-identity hint.We theoretically prove RAIL's absolute memorization on incrementally learned domains. Experiment results affirm RAIL's state-of-the-art performance in both X-TAIL and existing Multi-domain Task-Incremental Learning settings. The code is released at https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning.

</details>

---

## 85. DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ

- [ ] DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ | https://neurips.cc/virtual/2024/poster/94474

- **Link**: https://neurips.cc/virtual/2024/poster/94474

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Creating high-quality scientific figures can be time-consuming and challenging, even though sketching ideas on paper is relatively easy. Furthermore, recreating existing figures that are not stored in formats preserving semantic information is equally complex. To tackle this problem, we introduce DeTikZify, a novel multimodal language model that automatically synthesizes scientific figures as semantics-preserving TikZ graphics programs based on sketches and existing figures. To achieve this, we create three new datasets: DaTikZv2, the largest TikZ dataset to date, containing over 360k human-created TikZ graphics; SketchFig, a dataset that pairs hand-drawn sketches with their corresponding scientific figures; and MetaFig, a collection of diverse scientific figures and associated metadata. We train DeTikZify on MetaFig and DaTikZv2, along with synthetically generated sketches learned from SketchFig. We also introduce an MCTS-based inference algorithm that enables DeTikZify to iteratively refine its outputs without the need for additional training. Through both automatic and human evaluation, we demonstrate that DeTikZify outperforms commercial Claude 3 and GPT-4V in synthesizing TikZ programs, with the MCTS algorithm effectively boosting its performance. We make our code, models, and datasets publicly available.

</details>

---

## 86. One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos

- [ ] One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos | https://neurips.cc/virtual/2024/poster/94482

- **Link**: https://neurips.cc/virtual/2024/poster/94482

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce VideoLISA, a video-based multimodal large language model designed to tackle the problem of language-instructed reasoning segmentation in videos. Leveraging the reasoning capabilities and world knowledge of large language models, and augmented by the Segment Anything Model, VideoLISA generates temporally consistent segmentation masks in videos based on language instructions. Existing image-based methods, such as LISA, struggle with video tasks due to the additional temporal dimension, which requires temporal dynamic understanding and consistent segmentation across frames. VideoLISA addresses these challenges by integrating a Sparse Dense Sampling strategy into the video-LLM, which balances temporal context and spatial detail within computational constraints. Additionally, we propose a One-Token-Seg-All approach using a specially designed token, enabling the model to segment and track objects across multiple frames. Extensive evaluations on diverse benchmarks, including our newly introduced ReasonVOS benchmark, demonstrate VideoLISA's superior performance in video object segmentation tasks involving complex reasoning, temporal understanding, and object tracking. While optimized for videos, VideoLISA also shows promising generalization to image segmentation, revealing its potential as a unified foundation model for language-instructed object segmentation. Code and model will be available at: https://github.com/showlab/VideoLISA.

</details>

---

## 87. VisMin: Visual Minimal-Change Understanding

- [ ] VisMin: Visual Minimal-Change Understanding | https://neurips.cc/virtual/2024/poster/94495

- **Link**: https://neurips.cc/virtual/2024/poster/94495

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-grained understanding of objects, attributes, and relationships between objects is crucial for visual-language models (VLMs). To evaluate VLMs' fine-grained understanding, existing benchmarks primarily focus on evaluating VLMs' capability to distinguish between two very similar captions given an image. In this paper, our focus is on evaluating VLMs' capability to distinguish between two very similar images given a caption. To this end, we introduce a new, challenging benchmark termed Visual Minimal-Change Understanding (VisMin), which requires models to predict the correct image-caption match given two images and two captions. Importantly, the image pair (as well as the caption pair) contains minimal changes, i.e., between the two images (as well as between the two captions), only one aspect changes at a time from among the following possible types of changes: object, attribute, count, and spatial relation. These four types of minimal changes are specifically designed to test the models' understanding of objects, attributes of objects (such as color, material, shape), counts of objects, and spatial relationships between objects. To curate our benchmark, we built an automatic pipeline using large language models and diffusion models, followed by a rigorous 4-step verification process by human annotators. Empirical experiments reveal that current VLMs exhibit notable deficiencies in understanding spatial relationships and counting abilities. Furthermore, leveraging the automated nature of our data creation process, we generate a large-scale training dataset, which we use to finetune CLIP (a foundational VLM) and Idefics2 (a multimodal large language model). Our findings show that both these models benefit significantly from fine-tuning on this data, as evident by marked improvements in fine-grained understanding across a wide range of benchmarks. Additionally, such fine-tuning improves CLIP's general image-text alignment capabilities too. All resources including the benchmark, the training data, and the finetuned model checkpoints will be released.

</details>

---

## 88. Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions

- [ ] Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions | https://neurips.cc/virtual/2024/poster/94508

- **Link**: https://neurips.cc/virtual/2024/poster/94508

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large vision language models have demonstrated remarkable proficiency across a wide range of tasks. Yet, these models still struggle with understanding the nuances of human humor through juxtaposition, particularly when it involves nonlinear narratives that underpin many jokes and humor cues.  This paper investigates this challenge by focusing on comics with contradictory narratives, where each comic consists of two panels that create a humorous contradiction. We introduce the YesBut benchmark, which comprises tasks of varying difficulty aimed at assessing AI's capabilities in recognizing and interpreting these comics, ranging from literal content comprehension to deep narrative reasoning. Through extensive experimentation and analysis of recent commercial or open-sourced large vision language models, we assess their capability to comprehend the complex interplay of the narrative humor inherent in these comics. Our results show that even the state-of-the-art models still struggle with this task. Our findings offer insights into the current limitations and potential improvements for AI in understanding human creative expressions.

</details>

---

## 89. Easy Regional Contrastive Learning of Expressive Fashion Representations

- [ ] Easy Regional Contrastive Learning of Expressive Fashion Representations | https://neurips.cc/virtual/2024/poster/94509

- **Link**: https://neurips.cc/virtual/2024/poster/94509

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

When learning vision-language models (VLM) for the fashion domain, most existing works design new architectures from vanilla BERT with additional objectives, or perform dense multi-task learning with fashion-specific tasks. Though progress has been made, their architecture or objectives are often intricate and the extendibility is limited.By contrast, with simple architecture (comprising only two unimodal encoders) and just the contrastive objective, popular pre-trained VL models (e.g., CLIP) achieve superior performance in general domains, which are further easily extended to downstream tasks.However, inheriting such benefits of CLIP in the fashion domain is non-trivial in the presence of the notable domain gap. Empirically, we find that directly finetuning on fashion data leads CLIP to frequently ignore minor yet important details such as logos and composition, which are critical in fashion tasks such as retrieval and captioning.In this work, to maintain CLIP's simple architecture and objective while explicitly attending to fashion details, we propose $E^2$: Easy Regional Contrastive Learning of Expressive Fashion Representations.$E^2$ introduces only a few selection tokens and fusion blocks (just 1.9\% additional parameters in total) with only contrastive losses. Despite lightweight, in our primary focus, cross-modal retrieval, $E^2$ notably outperforms existing fashion VLMs with various fashion-specific objectives.Moreover, thanks to CLIP's widespread use in downstream tasks in general domains (e.g., zero-shot composed image retrieval and image captioning), our model can easily extend these models  from general domain to the fashion domain with notable improvement.To conduct a comprehensive evaluation, we further collect data from Amazon Reviews to build a new dataset for cross-modal retrieval in the fashion domain.

</details>

---

## 90. Streaming Long Video Understanding with Large Language Models

- [ ] Streaming Long Video Understanding with Large Language Models | https://neurips.cc/virtual/2024/poster/94520

- **Link**: https://neurips.cc/virtual/2024/poster/94520

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents VideoStreaming, an advanced vision-language large model (VLLM) for video understanding, that capably understands arbitrary-length video with a constant number of video tokens streamingly encoded and adaptively selected.The challenge of video understanding in the vision language area mainly lies in the significant computational burden caused by the great number of tokens extracted from long videos. Previous works rely on sparse sampling or frame compression to reduce tokens. However, such approaches either disregard temporal information in a long time span or sacrifice spatial details, resulting in flawed compression. To address these limitations, our VideoStreaming has two core designs: Memory-Propagated Streaming Encoding and Adaptive Memory Selection. The Memory-Propagated Streaming Encoding architecture segments long videos into short clips and sequentially encodes each clip with a propagated memory. In each iteration, we utilize the encoded results of the preceding clip as historical memory, which is integrated with the current clip to distill a condensed representation that encapsulates the video content up to the current timestamp. This method not only incorporates long-term temporal dynamics into the streaming encoding process but also yields a fixed-length memory as a global representation for arbitrarily long videos. After the encoding process, the Adaptive Memory Selection strategy selects a constant number of question-related memories from all the historical memories, and feeds them into the LLM to generate informative responses. The question-related selection reduces redundancy within the memories, enabling efficient and precise video understanding. Meanwhile, the disentangled video extraction and reasoning design allows the LLM to answer different questions about a video by directly selecting corresponding memories, without the need to encode the whole video for each question. Through extensive experiments, our model achieves superior performance and higher efficiency on long video benchmarks, showcasing precise temporal comprehension for detailed question answering.

</details>

---

## 91. TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy

- [ ] TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy | https://neurips.cc/virtual/2024/poster/94530

- **Link**: https://neurips.cc/virtual/2024/poster/94530

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Tables contain factual and quantitative data accompanied by various structures and contents that pose challenges for machine comprehension. Previous methods generally design task-specific architectures and objectives for individual tasks, resulting in modal isolation and intricate workflows. In this paper, we present a novel large vision-language model, TabPedia, equipped with a concept synergy mechanism. In this mechanism, all the involved diverse visual table understanding (VTU) tasks and multi-source visual embeddings are abstracted as concepts. This unified framework allows TabPedia to seamlessly integrate VTU tasks, such as table detection, table structure recognition, table querying, and table question answering, by leveraging the capabilities of large language models (LLMs). Moreover, the concept synergy mechanism enables table perception-related and comprehension-related tasks to work in harmony, as they can effectively leverage the needed clues from the corresponding source perception embeddings. Furthermore, to better evaluate the VTU task in real-world scenarios, we establish a new and comprehensive table VQA benchmark, ComTQA, featuring approximately 9,000 QA pairs. Extensive quantitative and qualitative experiments on both table perception and comprehension tasks, conducted across various public benchmarks, validate the effectiveness of our TabPedia. The superior performance further confirms the feasibility of using LLMs for understanding visual tables when all concepts work in synergy. The benchmark ComTQA has been open-sourced at https://huggingface.co/datasets/ByteDance/ComTQA. The source code and model also have been released at https://github.com/zhaowc-ustc/TabPedia.

</details>

---

## 92. Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels

- [ ] Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels | https://neurips.cc/virtual/2024/poster/94555

- **Link**: https://neurips.cc/virtual/2024/poster/94555

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models like CLIP have demonstrated impressive open-vocabulary capabilities for image-level tasks, excelling in recognizing what objects are present. However, they struggle with pixel-level recognition tasks like semantic segmentation, which require understanding where the objects are located. In this work, we propose a novel method, PixelCLIP, to adapt the CLIP image encoder for pixel-level understanding by guiding the model on where, which is achieved using unlabeled images and masks generated from vision foundation models such as SAM and DINO. To address the challenges of leveraging masks without semantic labels, we devise an online clustering algorithm using learnable class names to acquire general semantic concepts. PixelCLIP shows significant performance improvements over CLIP and competitive results compared to caption-supervised methods in open-vocabulary semantic segmentation.

</details>

---

## 93. Graph-based Unsupervised Disentangled Representation Learning via Multimodal Large Language Models

- [ ] Graph-based Unsupervised Disentangled Representation Learning via Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/94595

- **Link**: https://neurips.cc/virtual/2024/poster/94595

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Disentangled representation learning (DRL) aims to identify and decompose underlying factors behind observations, thus facilitating data perception and generation. However, current DRL approaches often rely on the unrealistic assumption that semantic factors are statistically independent. In reality, these factors may exhibit correlations, which off-the-shelf solutions have yet to properly address. To tackle this challenge, we introduce a bidirectional weighted graph-based framework, to learn factorized attributes and their interrelations within complex data. Specifically, we propose a $\beta$-VAE based module to extract factors as the initial nodes of the graph, and leverage the multimodal large language model (MLLM) to discover and rank latent correlations, thereby updating the weighted edges. By integrating these complementary modules, our model successfully achieves fine-grained, practical and unsupervised disentanglement. Experiments demonstrate our method's superior performance in disentanglement and reconstruction. Furthermore, the model inherits enhanced interpretability and generalizability from MLLMs.

</details>

---

## 94. TripletCLIP:  Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives

- [ ] TripletCLIP:  Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives | https://neurips.cc/virtual/2024/poster/94621

- **Link**: https://neurips.cc/virtual/2024/poster/94621

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pretraining (CLIP) models maximize the mutual information between text and visual modalities to learn representations. This makes the nature of the training data a significant factor in the efficacy of CLIP for downstream tasks. However, the lack of compositional diversity in contemporary image-text datasets limits the compositional reasoning ability of CLIP. We show that generating ``hard'' negative captions via in-context learning and synthesizing corresponding negative images with text-to-image generators offers a solution. We introduce a novel contrastive pre-training strategy that leverages these hard negative captions and images in an alternating fashion to train CLIP. We demonstrate that our method, named TripletCLIP, when applied to existing datasets such as CC3M and CC12M, enhances the compositional capabilities of CLIP, resulting in an absolute improvement of over 9% on the SugarCrepe benchmark on an equal computational budget, as well as improvements in zero-shot image classification and image retrieval. Our code, models, and data are available at: tripletclip.github.io.

</details>

---

## 95. Voila-A: Aligning Vision-Language Models with User's Gaze Attention

- [ ] Voila-A: Aligning Vision-Language Models with User's Gaze Attention | https://neurips.cc/virtual/2024/poster/94630

- **Link**: https://neurips.cc/virtual/2024/poster/94630

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In recent years, the integration of vision and language understanding has led to significant advancements in artificial intelligence, particularly through Vision-Language Models (VLMs). However, existing VLMs face challenges in handling real-world applications with complex scenes and multiple objects, as well as aligning their focus with the diverse attention patterns of human users. In this paper, we introduce gaze information, feasibly collected by ubiquitous wearable devices such as MR glasses, as a proxy for human attention to guide VLMs. We propose a novel approach, Voila-A, for gaze alignment to enhance the effectiveness of these models in real-world applications. First, we collect hundreds of minutes of gaze data to demonstrate that we can mimic human gaze modalities using localized narratives. We then design an automatic data annotation pipeline utilizing GPT-4 to generate the VOILA-COCO dataset. Additionally, we introduce a new model VOILA-A that integrate gaze information into VLMs while maintain pretrained knowledge from webscale dataset. We evaluate Voila-A using a hold-out validation set and a newly collected VOILA-GAZE testset, which features real-life scenarios captured with a gaze-tracking device. Our experimental results demonstrate that Voila-A significantly outperforms several baseline models. By aligning model attention with human gaze patterns, Voila-A paves the way for more intuitive, user-centric VLMs and fosters engaging human-AI interaction across a wide range of applications.

</details>

---

## 96. Zero-shot Generalizable Incremental Learning for Vision-Language Object Detection

- [ ] Zero-shot Generalizable Incremental Learning for Vision-Language Object Detection | https://neurips.cc/virtual/2024/poster/94639

- **Link**: https://neurips.cc/virtual/2024/poster/94639

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents Incremental Vision-Language Object Detection (IVLOD), a novel learning task designed to incrementally adapt pre-trained Vision-Language Object Detection Models (VLODMs) to various specialized domains, while simultaneously preserving their zero-shot generalization capabilities for the generalized domain. To address this new challenge, we present the Zero-interference Reparameterizable Adaptation (ZiRa), a novel method that introduces Zero-interference Loss and reparameterization techniques to tackle IVLOD without incurring a significant increase in memory usage. Comprehensive experiments on COCO and ODinW-13 datasets demonstrate that ZiRa effectively safeguards the zero-shot generalization ability of VLODMs while continuously adapting to new tasks. Specifically, after training on ODinW-13 datasets, ZiRa exhibits superior performance compared to CL-DETR and iDETR, boosting zero-shot generalizability by substantial $\textbf{13.91}$ and $\textbf{8.74}$ AP, respectively. Our code is available at https://github.com/JarintotionDin/ZiRaGroundingDINO.

</details>

---

## 97. MemVLT: Vision-Language Tracking with Adaptive Memory-based Prompts

- [ ] MemVLT: Vision-Language Tracking with Adaptive Memory-based Prompts | https://neurips.cc/virtual/2024/poster/94643

- **Link**: https://neurips.cc/virtual/2024/poster/94643

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language tracking (VLT) enhances traditional visual object tracking by integrating language descriptions, requiring the tracker to flexibly understand complex and diverse text in addition to visual information. However, most existing vision-language trackers still overly rely on initial fixed multimodal prompts, which struggle to provide effective guidance for dynamically changing targets. Fortunately, the Complementary Learning Systems (CLS) theory suggests that the human memory system can dynamically store and utilize multimodal perceptual information, thereby adapting to new scenarios. Inspired by this, (i) we propose a Memory-based Vision-Language Tracker (MemVLT). By incorporating memory modeling to adjust static prompts, our approach can provide adaptive prompts for tracking guidance. (ii) Specifically, the memory storage and memory interaction modules are designed in accordance with CLS theory. These modules facilitate the storage and flexible interaction between short-term and long-term memories, generating prompts that adapt to target variations. (iii) Finally, we conduct extensive experiments on mainstream VLT datasets (e.g., MGIT, TNL2K, LaSOT and LaSOT$_{ext}$). Experimental results show that MemVLT achieves new state-of-the-art performance. Impressively, it achieves 69.4% AUC on the MGIT and 63.3% AUC on the TNL2K, improving the existing best result by 8.4% and 4.7%, respectively.

</details>

---

## 98. Conformal Alignment: Knowing When to Trust Foundation Models with Guarantees

- [ ] Conformal Alignment: Knowing When to Trust Foundation Models with Guarantees | https://neurips.cc/virtual/2024/poster/94658

- **Link**: https://neurips.cc/virtual/2024/poster/94658

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Before deploying outputs from foundation models in high-stakes tasks, it is imperative to ensure that they align with human values.For instance, in radiology report generation, reports generated by a vision-language model must align with human evaluations before their use in medical decision-making. This paper presents Conformal Alignment, a general framework for identifying units whose outputs meet a user-specified alignment criterion. It is guaranteed that on average, a prescribed fraction of selected units indeed meet the alignment criterion, regardless of the foundation model or the data distribution. Given any pre-trained model and new units with model-generated outputs, Conformal Alignment leverages a set of reference data with ground-truth alignment status to train an alignment predictor. It then selects new units whose predicted alignment scores surpass a data-dependent threshold, certifying their corresponding outputs as trustworthy. Through applications to question answering and radiology report generation, we demonstrate that our method is able to accurately identify units with trustworthy outputs via lightweight training over a moderate amount of reference data. En route, we investigate the informativeness of various features in alignment prediction and combine them with standard models to construct the alignment predictor.

</details>

---

## 99. Aggregate-and-Adapt Natural Language Prompts for Downstream Generalization of CLIP

- [ ] Aggregate-and-Adapt Natural Language Prompts for Downstream Generalization of CLIP | https://neurips.cc/virtual/2024/poster/94659

- **Link**: https://neurips.cc/virtual/2024/poster/94659

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large pretrained vision-language models like CLIP have shown promising generalization capability, but may struggle in specialized domains (e.g., satellite imagery) or fine-grained classification (e.g., car models) where the visual concepts are unseen or under-represented during pretraining. Prompt learning offers a parameter-efficient finetuning framework that can adapt CLIP to downstream tasks even when limited annotation data are available. In this paper, we improve prompt learning by distilling the textual knowledge from natural language prompts (either human- or LLM-generated) to provide rich priors for those under-represented concepts. We first obtain a prompt ``summary'' aligned to each input image via a learned prompt aggregator. Then we jointly train a prompt generator, optimized to produce a prompt embedding that stays close to the aggregated summary while minimizing task loss at the same time. We dub such prompt embedding as Aggregate-and-Adapted Prompt Embedding (AAPE). AAPE is shown to be able to generalize to different downstream data distributions and tasks, including vision-language understanding tasks (e.g., few-shot classification, VQA) and generation tasks (image captioning) where AAPE achieves competitive performance. We also show AAPE is particularly helpful to handle non-canonical and OOD examples. Furthermore, AAPE learning eliminates LLM-based inference cost as required by baselines, and scales better with data and LLM model size.

</details>

---

## 100. AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation

- [ ] AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation | https://neurips.cc/virtual/2024/poster/94677

- **Link**: https://neurips.cc/virtual/2024/poster/94677

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have shown impressive results in various visual classification tasks.However, we often fail to fully unleash their potential when adapting them for new concept understanding due to limited information on new classes.To address this limitation, we introduce a novel adaptation framework, AWT (Augment, Weight, then Transport). AWT comprises three key components: augmenting inputs with diverse visual perspectives and enriched class descriptions through image transformations and language models; dynamically weighting inputs based on the prediction entropy; and employing optimal transport to mine semantic correlations in the vision-language space.AWT can be seamlessly integrated into various VLMs, enhancing their zero-shot capabilities without additional training and facilitating few-shot learning through an integrated multimodal adapter module.We verify AWT in multiple challenging scenarios, including zero-shot and few-shot image classification, zero-shot video action recognition, and out-of-distribution generalization. AWT consistently outperforms the state-of-the-art methods in each setting. In addition, our extensive studies further demonstrate AWT's effectiveness and adaptability across different VLMs, architectures, and scales.

</details>

---

## 101. SpatialPIN: Enhancing Spatial Reasoning Capabilities of Vision-Language Models through Prompting and Interacting 3D Priors

- [ ] SpatialPIN: Enhancing Spatial Reasoning Capabilities of Vision-Language Models through Prompting and Interacting 3D Priors | https://neurips.cc/virtual/2024/poster/94696

- **Link**: https://neurips.cc/virtual/2024/poster/94696

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current state-of-the-art spatial reasoning-enhanced VLMs are trained to excel at spatial visual question answering (VQA). However, we believe that higher-level 3D-aware tasks, such as articulating dynamic scene changes and motion planning, require a fundamental and explicit 3D understanding beyond current spatial VQA datasets. In this work, we present SpatialPIN, a framework designed to enhance the spatial reasoning capabilities of VLMs through prompting and interacting with priors from multiple 3D foundation models in a zero-shot, training-free manner. Extensive experiments demonstrate that our spatial reasoning-imbued VLM performs well on various forms of spatial VQA and can extend to help in various downstream robotics tasks such as pick and stack and trajectory planning.

</details>

---

## 102. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models

- [ ] Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/94704

- **Link**: https://neurips.cc/virtual/2024/poster/94704

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Machine unlearning (MU) empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii)  Joint training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.

</details>

---

## 103. Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method

- [ ] Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method | https://neurips.cc/virtual/2024/poster/94723

- **Link**: https://neurips.cc/virtual/2024/poster/94723

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Integrating pretrained vision-language foundation models like CLIP into federated learning has attracted significant attention for enhancing generalization across diverse tasks. Typically, federated learning of vision-language models employs prompt learning to reduce communication and computational costs, i.e., prompt-based federated learning. However, there is limited theoretical analysis to understand the performance of prompt-based federated learning. In this work, we construct a theoretical analysis framework for prompt-based federated learning via feature learning theory. Specifically, we monitor the evolution of signal learning and noise memorization in prompt-based federated learning, demonstrating that performance can be assessed by the ratio of task-relevant to task-irrelevant coefficients. Furthermore, we draw an analogy between income and risk in portfolio optimization and the task-relevant and task-irrelevant terms in feature learning. Leveraging inspiration from portfolio optimization that combining two independent assets will maintain the income while reducing the risk, we introduce two prompts: global prompt and local prompt to construct a prompt portfolio to balance the generalization and personalization. Consequently, we showed the performance advantage of the prompt portfolio and derived the optimal mixing coefficient. These theoretical claims have been further supported by empirical experiments.

</details>

---

## 104. MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models

- [ ] MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/94738

- **Link**: https://neurips.cc/virtual/2024/poster/94738

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have demonstrated impressive capabilities across various vision-language tasks. However, a generalist MLLM typically underperforms compared with a specialist MLLM on most VL tasks, which can be attributed to task interference. In this paper, we propose a mixture of multimodal experts (MoME) to mitigate task interference and obtain a generalist MLLM. Our MoME is composed of two key components, a mixture of vision experts (MoVE) and a mixture of language experts (MoLE). MoVE can adaptively modulate the features transformed from various vision encoders, and has a strong compatibility in transformation architecture. MoLE incorporates sparsely gated experts into LLMs to achieve painless improvements with roughly unchanged inference costs. In response to task interference, our MoME specializes in both vision and language modality to adapt to task discrepancies. Extensive experiments show that MoME significantly improves the performance of generalist MLLMs across various VL tasks.

</details>

---

## 105. Training-Free Open-Ended Object Detection and Segmentation via Attention as Prompts

- [ ] Training-Free Open-Ended Object Detection and Segmentation via Attention as Prompts | https://neurips.cc/virtual/2024/poster/94761

- **Link**: https://neurips.cc/virtual/2024/poster/94761

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing perception models achieve great success by learning from large amounts of labeled data, but they still struggle with open-world scenarios. To alleviate this issue, researchers introduce open-set perception tasks to detect or segment unseen objects in the training set. However, these models require predefined object categories as inputs during inference, which are not available in real-world scenarios. Recently, researchers pose a new and more practical problem, i.e., open-ended object detection, which discovers unseen objects without any object categories as inputs. In this paper, we present VL-SAM, a training-free framework that combines the generalized object recognition model (i.e., Vision-Language Model) with the generalized object localization model (i.e., Segment-Anything Model), to address the open-ended object detection and segmentation task. Without additional training, we connect these two generalized models with attention maps as the prompts. Specifically, we design an attention map generation module by employing head aggregation and a regularized attention flow to aggregate and propagate attention maps across all heads and layers in VLM, yielding high-quality attention maps. Then, we iteratively sample positive and negative points from the attention maps with a prompt generation module and send the sampled points to SAM to segment corresponding objects. Experimental results on the long-tail instance segmentation dataset (LVIS) show that our method surpasses the previous open-ended method on the object detection task and can provide additional instance segmentation masks. Besides, VL-SAM achieves favorable performance on the corner case object detection dataset (CODA), demonstrating the effectiveness of VL-SAM in real-world applications. Moreover, VL-SAM exhibits good model generalization that can incorporate various VLMs and SAMs.

</details>

---

## 106. Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks

- [ ] Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks | https://neurips.cc/virtual/2024/poster/94762

- **Link**: https://neurips.cc/virtual/2024/poster/94762

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Building a general-purpose agent is a long-standing vision in the field of artificial intelligence. Existing agents have made remarkable progress in many domains, yet they still struggle to complete long-horizon tasks in an open world. We attribute this to the lack of necessary world knowledge and multimodal experience that can guide agents through a variety of long-horizon tasks. In this paper, we propose a Hybrid Multimodal Memory module to address the above challenges. It 1) transforms knowledge into Hierarchical Directed Knowledge Graph that allows agents to explicitly represent and learn world knowledge, and 2) summarises historical information into Abstracted Multimodal Experience Pool that provide agents with rich references for in-context learning. On top of the Hybrid Multimodal Memory module, a multimodal agent, Optimus-1, is constructed with dedicated Knowledge-guided Planner and Experience-Driven Reflector, contributing to a better planning and reflection in the face of long-horizon tasks in Minecraft. Extensive experimental results show that Optimus-1 significantly outperforms all existing agents on challenging long-horizon task benchmarks, and exhibits near human-level performance on many tasks. In addition, we introduce various Multimodal Large Language Models (MLLMs) as the backbone of Optimus-1. Experimental results show that Optimus-1 exhibits strong generalization with the help of the Hybrid Multimodal Memory module, outperforming the GPT-4V baseline on many tasks.

</details>

---

## 107. Textual Training for the Hassle-Free Removal of Unwanted Visual Data: Case Studies on OOD and Hateful Image Detection

- [ ] Textual Training for the Hassle-Free Removal of Unwanted Visual Data: Case Studies on OOD and Hateful Image Detection | https://neurips.cc/virtual/2024/poster/94785

- **Link**: https://neurips.cc/virtual/2024/poster/94785

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In our study, we explore methods for detecting unwanted content lurking in visual datasets. We provide a theoretical analysis demonstrating that a model capable of successfully partitioning visual data can be obtained using only textual data. Based on the analysis, we propose Hassle-Free Textual Training (HFTT), a streamlined method capable of acquiring detectors for unwanted visual content, using only textual data in conjunction with pre-trained vision-language models. HFTT features an innovative objective function that significantly reduces the necessity for human involvement in data annotation. Furthermore, HFTT employs a clever textual data synthesis method, effectively emulating the integration of unknown visual data distribution into the training process at no extra cost. The unique characteristics of HFTT extend its utility beyond traditional out-of-distribution detection, making it applicable to tasks that address more abstract concepts. We complement our analyses with experiments in hateful image detection and out-of-distribution detection. Our codes are available at https://github.com/HFTT-anonymous/HFTT.

</details>

---

## 108. OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding

- [ ] OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding | https://neurips.cc/virtual/2024/poster/94820

- **Link**: https://neurips.cc/virtual/2024/poster/94820

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current universal segmentation methods demonstrate strong capabilities in pixel-level image and video understanding. However, they lack reasoning abilities and cannot be controlled via text instructions. In contrast, large vision-language multimodal models exhibit powerful vision-based conversation and reasoning capabilities but lack pixel-level understanding and have difficulty accepting visual prompts for flexible user interaction. This paper proposes OMG-LLaVA, a new and elegant framework combining powerful pixel-level vision understanding with reasoning abilities. It can accept various visual and text prompts for flexible user interaction. Specifically, we use a universal segmentation method as the visual encoder, integrating image information, perception priors, and visual prompts into visual tokens provided to the LLM. The LLM is responsible for understanding the user's text instructions and providing text responses and pixel-level segmentation results based on the visual information. We propose perception prior embedding to better integrate perception priors with image features. OMG-LLaVA achieves image-level, object-level, and pixel-level reasoning and understanding in a single model, matching or surpassing the performance of specialized methods on multiple benchmarks. Rather than using LLM to connect each specialist, our work aims at end-to-end training on one encoder, one decoder, and one LLM. The code and model have been released for further research.

</details>

---

## 109. Leveraging Visual Tokens for Extended Text Contexts in Multi-Modal Learning

- [ ] Leveraging Visual Tokens for Extended Text Contexts in Multi-Modal Learning | https://neurips.cc/virtual/2024/poster/94826

- **Link**: https://neurips.cc/virtual/2024/poster/94826

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Training models with longer in-context lengths is a significant challenge for multimodal machine learning due to substantial GPU memory and computational costs. This exploratory study does not present state-of-the-art models; rather, it introduces an innovative method designed to increase in-context text length in multi-modality large language models (MLLMs) efficiently. We present \ModelFullName (\ModelName), which processes long in-context text using visual tokens. This technique significantly reduces GPU memory usage and floating point operations (FLOPs). For instance, our method expands the pre-training in-context length from 256 to 2048 tokens with fewer FLOPs for a 56 billion parameter MOE model. Experimental results demonstrate that \ModelName enhances OCR capabilities and delivers superior performance on common downstream benchmarks for in-context few-shot evaluation. Additionally, \ModelName proves effective for long context inference, achieving results comparable to full text input while maintaining computational efficiency.

</details>

---

## 110. IPO: Interpretable Prompt Optimization for Vision-Language Models

- [ ] IPO: Interpretable Prompt Optimization for Vision-Language Models | https://neurips.cc/virtual/2024/poster/94834

- **Link**: https://neurips.cc/virtual/2024/poster/94834

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models like CLIP have remarkably adapted to various downstream tasks. Nonetheless, their performance heavily depends on the specificity of the input text prompts, which requires skillful prompt template engineering. Instead, current approaches to prompt optimization learn the prompts through gradient descent, where the prompts are treated as adjustable parameters. However, these methods tend to lead to overfitting of the base classes seen during training and produce prompts that are no longer understandable by humans. This paper introduces a simple but interpretable prompt optimizer (IPO), that utilizes large language models (LLMs) to generate textual prompts dynamically. We introduce a Prompt Optimization Prompt that not only guides LLMs in creating effective prompts but also stores past prompts with their performance metrics, providing rich in-context information. Additionally, we incorporate a large multimodal model (LMM) to condition on visual content by generating image descriptions, which enhance the interaction between textual and visual modalities. This allows for the creation of dataset-specific prompts that improve generalization performance, while maintaining human comprehension. Extensive testing across 11 datasets reveals that IPO not only improves the accuracy of existing gradient-descent-based prompt learning methods but also considerably enhances the interpretability of the generated prompts. By leveraging the strengths of LLMs, our approach ensures that the prompts remain human-understandable, thereby facilitating better transparency and oversight for vision-language models.

</details>

---

## 111. LLaMo: Large Language Model-based Molecular Graph Assistant

- [ ] LLaMo: Large Language Model-based Molecular Graph Assistant | https://neurips.cc/virtual/2024/poster/94836

- **Link**: https://neurips.cc/virtual/2024/poster/94836

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have demonstrated remarkable generalization and instruction-following capabilities with instruction tuning. The advancements in LLMs and instruction tuning have led to the development of Large Vision-Language Models (LVLMs). However, the competency of the LLMs and instruction tuning have been less explored in the molecular domain. Thus, we propose LLaMo: Large Language Model-based Molecular graph assistant, which is an end-to- end trained large molecular graph-language model. To bridge the discrepancy between the language and graph modalities, we present the multi-level graph projector that transforms graph representations into graph tokens by abstracting the output representations of each GNN layer and motif representations with the cross-attention mechanism. We also introduce machine-generated molecular graph instruction data to instruction-tune the large molecular graph-language model for general-purpose molecule and language understanding. Our extensive experiments demonstrate that LLaMo shows the best performance on diverse tasks, such as molecular description generation, property prediction, and IUPAC name prediction. The code of LLaMo is available at https://github.com/mlvlab/LLaMo.

</details>

---

## 112. Web-Scale Visual Entity Recognition: An LLM-Driven Data Approach

- [ ] Web-Scale Visual Entity Recognition: An LLM-Driven Data Approach | https://neurips.cc/virtual/2024/poster/94878

- **Link**: https://neurips.cc/virtual/2024/poster/94878

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Web-scale visual entity recognition, the task of associating images with their corresponding entities within vast knowledge bases like Wikipedia, presents significant challenges due to the lack of clean, large-scale training data. In this paper, we propose a novel methodology to curate such a dataset, leveraging a multimodal large language model (LLM) for label verification, metadata generation, and rationale explanation. Instead of relying on the multimodal LLM to directly annotate data, which we found to be suboptimal, we prompt it to reason about potential candidate entity labels by accessing additional contextually relevant information (such as Wikipedia), resulting in more accurate annotations. We further use the multimodal LLM to enrich the dataset by generating question-answer pairs and a grounded fine-grained textual description (referred to as "rationale") that explains the connection between images and their assigned entities. Experiments demonstrate that models trained on this automatically curated data achieve state-of-the-art performance on web-scale visual entity recognition tasks (e.g. +6.9% improvement in OVEN entity task), underscoring the importance of high-quality training data in this domain.

</details>

---

## 113. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs

- [ ] Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs | https://neurips.cc/virtual/2024/poster/94880

- **Link**: https://neurips.cc/virtual/2024/poster/94880

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Cambrian-1, a family of multimodal LLMs (MLLMs) designed with a vision-centric approach. While stronger language models can enhance multimodal capabilities, the design choices for vision components are often insufficiently explored and disconnected from visual representation learning research. This gap hinders accurate sensory grounding in real-world scenarios. Our study uses LLMs and visual instruction tuning as an interface to evaluate various visual representations, offering new insights into different models and architectures—self-supervised, strongly supervised, or combinations thereof—based on experiments with over 15 vision models. We critically examine existing MLLM benchmarks, addressing the difficulties involved in consolidating and interpreting results from various tasks. To further improve visual grounding, we propose spatial vision aggregator (SVA), a dynamic and spatially-aware connector that integrates vision features with LLMs while reducing the number of tokens. Additionally, we discuss the curation of high-quality visual instruction-tuning data from publicly available sources, emphasizing the importance of distribution balancing. Collectively, Cambrian-1 not only achieves state-of-the-art performances but also serves as a comprehensive, open cookbook for instruction-tuned MLLMs. We provide model weights, code, supporting tools, datasets, and detailed instruction-tuning and evaluation recipes. We hope our release will inspire and accelerate advancements in multimodal systems and visual representation learning.

</details>

---

## 114. $\textit{Bifr\"ost}$: 3D-Aware Image Compositing with Language Instructions

- [ ] $\textit{Bifr\"ost}$: 3D-Aware Image Compositing with Language Instructions | https://neurips.cc/virtual/2024/poster/94882

- **Link**: https://neurips.cc/virtual/2024/poster/94882

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces $\textit{Bifröst}$, a novel 3D-aware framework that is built upon diffusion models to perform instruction-based image composition. Previous methods concentrate on image compositing at the 2D level, which fall short in handling complex spatial relationships ($\textit{e.g.}$, occlusion). $\textit{Bifröst}$ addresses these issues by training MLLM as a 2.5D location predictor and integrating depth maps as an extra condition during the generation process to bridge the gap between 2D and 3D, which enhances spatial comprehension and supports sophisticated spatial interactions. Our method begins by fine-tuning MLLM with a custom counterfactual dataset to predict 2.5D object locations in complex backgrounds from language instructions. Then, the image-compositing model is uniquely designed to process multiple types of input features, enabling it to perform high-fidelity image compositions that consider occlusion, depth blur, and image harmonization. Extensive qualitative and quantitative evaluations demonstrate that $\textit{Bifröst}$ significantly outperforms existing methods, providing a robust solution for generating realistically composited images in scenarios demanding intricate spatial understanding. This work not only pushes the boundaries of generative image compositing but also reduces reliance on expensive annotated datasets by effectively utilizing existing resources in innovative ways.

</details>

---

## 115. NoiseGPT: Label Noise Detection and Rectification through Probability Curvature

- [ ] NoiseGPT: Label Noise Detection and Rectification through Probability Curvature | https://neurips.cc/virtual/2024/poster/94898

- **Link**: https://neurips.cc/virtual/2024/poster/94898

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Machine learning craves high-quality data which is a major bottleneck during realistic deployment, as it takes abundant resources and massive human labor to collect and label data. Unfortunately, label noise where image data mismatches with incorrect label exists ubiquitously in all kinds of datasets, significantly degrading the learning performance of deep networks. Learning with Label Noise (LNL) has been a common strategy for mitigating the influence of noisy labels. However, existing LNL methods either require pertaining using the memorization effect to separate clean data from noisy ones or rely on dataset assumptions that cannot extend to various scenarios. Thanks to the development of Multimodal Large Language Models (MLLMs) which possess massive knowledge and hold In-Context Learning (ICL) ability, this paper proposes NoiseGPT to effectively leverage MLLMs as a knowledge expert for conducting label noise detection and rectification. Specifically, we observe a \textit{probability curvature} effect of MLLMs where clean and noisy examples reside on curvatures with different smoothness, further enabling the detection of label noise. By designing a token-wise Mix-of-Feature (MoF) technique to produce the curvature, we propose an In-Context Discrepancy (ICD) measure to determine the authenticity of an image-label pair. Subsequently, we repeat such a process to find the best matching pairs to complete our label rectification. Through extensive experiments, we carefully demonstrate the effectiveness of NoiseGPT on detecting and cleansing dataset noise, especially on ILSVRC12, the AUROC of NoiseGPT reached over 0.92. And by integrating with existing methods, the classification performance can be significantly improved on noisy datasets, typically by 22.8\% on 80\% symmetric CIFAR-10 with M-correction. Source code: \url{https://github.com/drunkerWang/NoiseGPT}

</details>

---

## 116. Renovating Names in Open-Vocabulary Segmentation Benchmarks

- [ ] Renovating Names in Open-Vocabulary Segmentation Benchmarks | https://neurips.cc/virtual/2024/poster/94935

- **Link**: https://neurips.cc/virtual/2024/poster/94935

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Names are essential to both human cognition and vision-language models. Open-vocabulary models utilize class names as text prompts to generalize to categories unseen during training. However, the precision of these names is often overlooked in existing datasets. In this paper, we address this underexplored problem by presenting a framework for "renovating" names in open-vocabulary segmentation benchmarks (RENOVATE). Our framework features a renaming model that enhances the quality of names for each visual segment. Through experiments, we demonstrate that our renovated names help train stronger open-vocabulary models with up to 15% relative improvement and significantly enhance training efficiency with improved data quality. We also show that our renovated names improve evaluation by better measuring misclassification and enabling fine-grained model analysis. We provide our code and relabelings for several popular segmentation datasets to the research community on our project page: https://andrehuang.github.io/renovate.

</details>

---

## 117. GenArtist: Multimodal LLM as an Agent for Unified Image Generation and Editing

- [ ] GenArtist: Multimodal LLM as an Agent for Unified Image Generation and Editing | https://neurips.cc/virtual/2024/poster/94941

- **Link**: https://neurips.cc/virtual/2024/poster/94941

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the success achieved by existing image generation and editing methods, current models still struggle with complex problems including intricate text prompts, and the absence of  verification and self-correction mechanisms makes the generated images unreliable. Meanwhile, a single model tends to specialize in particular tasks and possess the corresponding capabilities, making it inadequate for fulfilling all user requirements. We propose GenArtist, a unified image generation and editing system, coordinated by a multimodal large language model (MLLM) agent. We integrate a comprehensive range of existing models into the tool library and utilize the agent for tool selection and execution. For a complex problem, the MLLM agent decomposes it into simpler sub-problems and constructs a tree structure to systematically plan the procedure of generation, editing, and self-correction with step-by-step verification. By automatically generating missing position-related inputs and incorporating position information, the appropriate tool can be effectively employed to address each sub-problem. Experiments demonstrate that GenArtist can perform various generation and editing tasks, achieving state-of-the-art performance and surpassing existing models such as SDXL and DALL-E 3, as can be seen in Fig. 1. We will open-source the code for future research and applications.

</details>

---

## 118. No Filter: Cultural and Socioeconomic Diversity in Contrastive Vision-Language Models

- [ ] No Filter: Cultural and Socioeconomic Diversity in Contrastive Vision-Language Models | https://neurips.cc/virtual/2024/poster/94944

- **Link**: https://neurips.cc/virtual/2024/poster/94944

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We study cultural and socioeconomic diversity in contrastive vision-language models (VLMs). Using a broad range of benchmark datasets and evaluation metrics, we bring to attention several important findings. First, the common filtering of training data to English image-text pairs disadvantages communities of lower socioeconomic status and negatively impacts cultural understanding. Notably, this performance gap is not captured by - and even at odds with - the currently popular evaluation metrics derived from the Western-centric ImageNet and COCO datasets. Second, pretraining with global, unfiltered data before fine-tuning on English content can improve cultural understanding without sacrificing performance on said popular benchmarks. Third, we introduce the task of geo-localization as a novel evaluation metric to assess cultural diversity in VLMs. Our work underscores the value of using diverse data to create more inclusive multimodal systems and lays the groundwork for developing VLMs that better represent global perspectives.

</details>

---

## 119. RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models

- [ ] RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models | https://neurips.cc/virtual/2024/poster/94981

- **Link**: https://neurips.cc/virtual/2024/poster/94981

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuned vision-language models (VLMs) often capture spurious correlations between image features and textual attributes, resulting in degraded zero-shot performance at test time. Existing approaches for addressing spurious correlations (i) primarily operate at the global image-level rather than intervening directly on fine-grained image features and (ii) are predominantly designed for unimodal settings. In this work, we present RaVL, which takes a fine-grained perspective on VLM robustness by discovering and mitigating spurious correlations using local image features rather than operating at the global image level. Given a fine-tuned VLM, RaVL first discovers spurious correlations by leveraging a region-level clustering approach to identify precise image features contributing to zero-shot classification errors. Then, RaVL mitigates the identified spurious correlation with a novel region-aware loss function that enables the VLM to focus on relevant regions and ignore spurious relationships during fine-tuning. We evaluate RaVL on 654 VLMs with various model architectures, data domains, and learned spurious correlations. Our results show that RaVL accurately discovers (191% improvement over the closest baseline) and mitigates (8.2% improvement on worst-group image classification accuracy) spurious correlations. Qualitative evaluations on general-domain and medical-domain VLMs confirm our findings.

</details>

---

## 120. An eye for an ear: zero-shot audio description leveraging an image captioner with audio-visual token distribution matching

- [ ] An eye for an ear: zero-shot audio description leveraging an image captioner with audio-visual token distribution matching | https://neurips.cc/virtual/2024/poster/94989

- **Link**: https://neurips.cc/virtual/2024/poster/94989

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models have fueled progress in image captioning. These models, fine-tuned on vast image datasets, exhibit a deep understanding of semantic concepts.In this work, we show that this ability can be re-purposed for audio captioning, where the joint image-language decoder can be leveraged to describe auditory content associated with image sequences within videos featuring audiovisual content. This can be achieved via multimodal alignment.Yet, this multimodal alignment task is non-trivial due to the inherent disparity between audible and visible elements in real-world videos. Moreover, multimodal representation learning often relies on contrastive learning, facing the challenge of the so-called modality gap which hinders smooth integration between modalities. In this work, we introduce a novel methodology for bridging the audiovisual modality gap by matching the distributions of tokens produced by an audio backbone and those of an image captioner. Our approach aligns the audio token distribution with that of the image tokens, enabling the model to perform zero-shot audio captioning in an unsupervised fashion. This alignment allows for the use of either audio or audiovisual input by combining or substituting the image encoder with the aligned audio encoder. Our method achieves significantly improved performances in zero-shot audio captioning, compared to existing approaches.

</details>

---

## 121. Lever LM: Configuring In-Context Sequence to Lever Large Vision Language Models

- [ ] Lever LM: Configuring In-Context Sequence to Lever Large Vision Language Models | https://neurips.cc/virtual/2024/poster/95045

- **Link**: https://neurips.cc/virtual/2024/poster/95045

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As Archimedes famously said, ``Give me a lever long enough and a fulcrum on which to place it, and I shall move the world'', in this study, we propose to use a tiny Language Model (LM), \eg, a Transformer with 67M parameters, to lever much larger Vision-Language Models (LVLMs) with 9B parameters. Specifically, we use this tiny \textbf{Lever-LM} to configure effective in-context demonstration (ICD) sequences to improve the In-Context Learinng (ICL) performance of LVLMs. Previous studies show that diverse ICD configurations like the selection and ordering of the demonstrations heavily affect the ICL performance, highlighting the significance of configuring effective ICD sequences. Motivated by this and by re-considering the the process of configuring ICD sequence, we find this is a mirror process of human sentence composition and further assume that effective ICD configurations may contain internal statistical patterns that can be captured by Lever-LM. Then a dataset with effective ICD sequences is constructed to train Lever-LM. After training, given novel queries, new ICD sequences are configured by the trained Lever-LM to solve vision-language tasks through ICL. Experiments show that these ICD sequences can improve the ICL performance of two LVLMs compared with some strong baselines in Visual Question Answering and Image Captioning, validating that Lever-LM can really capture the statistical patterns for levering LVLMs. The code is available at \url{https://anonymous.4open.science/r/Lever-LM-604A/}.

</details>

---

## 122. Unveiling Encoder-Free Vision-Language Models

- [ ] Unveiling Encoder-Free Vision-Language Models | https://neurips.cc/virtual/2024/poster/95075

- **Link**: https://neurips.cc/virtual/2024/poster/95075

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing vision-language models (VLMs) mostly rely on vision encoders to extract visual features followed by large language models (LLMs) for visual-language tasks. However, the vision encoders set a strong inductive bias in abstracting visual representation, e.g., resolution, aspect ratio, and semantic priors, which could impede the flexibility and efficiency of the VLMs. Training pure VLMs that accept the seamless vision and language inputs, i.e., without vision encoders, remains challenging and rarely explored. Empirical observations reveal that direct training without encoders results in slow convergence and large performance gaps. In this work, we bridge the gap between encoder-based and encoder-free models, and present a simple yet effective training recipe towards pure VLMs. Specifically, we unveil the key aspects of training encoder-free VLMs efficiently via thorough experiments: (1) Bridging vision-language representation inside one unified decoder; (2) Enhancing visual recognition capability via extra supervision. With these strategies, we launch EVE, an encoder-free vision-language model that can be trained and forwarded efficiently. Notably, solely utilizing 35M publicly accessible data, EVE can impressively rival the encoder-based VLMs of similar capacities across multiple vision-language benchmarks. It significantly outperforms the counterpart Fuyu-8B with mysterious training procedures and undisclosed training data. We believe that EVE provides a transparent and efficient route for developing pure decoder-only architecture across modalities.

</details>

---

## 123. GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning

- [ ] GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning | https://neurips.cc/virtual/2024/poster/95093

- **Link**: https://neurips.cc/virtual/2024/poster/95093

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) are increasingly used for various tasks with graph structures. Though LLMs can process graph information in a textual format, they overlook the rich vision modality, which is an intuitive way for humans to comprehend structural information and conduct general graph reasoning. The potential benefits and capabilities of representing graph structures as visual images (i.e., $\textit{visual graph}$) are still unexplored. To fill the gap, we innovatively propose an end-to-end framework, called $\textbf{G}$raph to v$\textbf{I}$sual and $\textbf{T}$extual Integr$\textbf{A}$tion (GITA), which firstly incorporates visual graphs into general graph reasoning. Besides, we establish  $\textbf{G}$raph-based $\textbf{V}$ision-$\textbf{L}$anguage $\textbf{Q}$uestion $\textbf{A}$nswering (GVLQA) dataset from existing graph data, which is the first vision-language dataset for general graph reasoning purposes. Extensive experiments on the GVLQA dataset and five real-world datasets show that GITA outperforms mainstream LLMs in terms of general graph reasoning capabilities. Moreover, We highlight the effectiveness of the layout augmentation on visual graphs and pretraining on the GVLQA dataset.

</details>

---

## 124. Déjà Vu Memorization in Vision–Language Models

- [ ] Déjà Vu Memorization in Vision–Language Models | https://neurips.cc/virtual/2024/poster/95117

- **Link**: https://neurips.cc/virtual/2024/poster/95117

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have emerged as the state-of-the-art representation learning solution, with myriads of downstream applications such as image classification, retrieval and generation. A natural question is whether these models memorize their training data, which also has implications for generalization. We propose a new method for measuring memorization in VLMs, which we call dèjá vu memorization. For VLMs trained on image-caption pairs, we show that the model indeed retains information about individual objects in the training images beyond what can be inferred from correlations or the image caption. We evaluate dèjá vu memorization at both sample and population level, and show that it is significant for OpenCLIP trained on as many as 50M image-caption pairs. Finally, we show that text randomization considerably mitigates memorization risk while only moderately impacting the model’s downstream task performance.  The code is available here: https://github.com/facebookresearch/VLMDejaVu.

</details>

---

## 125. Alleviating Hallucinations in Large Vision-Language Models through Hallucination-Induced Optimization

- [ ] Alleviating Hallucinations in Large Vision-Language Models through Hallucination-Induced Optimization | https://neurips.cc/virtual/2024/poster/95118

- **Link**: https://neurips.cc/virtual/2024/poster/95118

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although Large Visual Language Models (LVLMs) have demonstrated exceptional abilities in understanding multimodal data, they invariably suffer from hallucinations, leading to a disconnection between the generated text and the corresponding images. Almost all  current visual contrastive decoding methods attempt to mitigate these hallucinations by introducing visual uncertainty information that appropriately widens the contrastive logits gap between hallucinatory and targeted ones.    However, due to uncontrollable nature of the global visual uncertainty, they struggle to precisely induce the hallucinatory tokens, which severely limits their effectiveness in mitigating hallucinations and may even lead to the generation of undesired hallucinations.    To tackle this issue, we conducted the theoretical analysis to promote the effectiveness of contrast decoding. Building on this insight, we introduce a novel optimization strategy named Hallucination-Induced Optimization (HIO). This strategy seeks to amplify the contrast between hallucinatory and targeted tokens relying on a fine-tuned theoretical preference model (i.e., Contrary Bradley-Terry Model), thereby facilitating efficient contrast decoding to alleviate hallucinations in LVLMs.    Extensive experimental research demonstrates that our HIO strategy can effectively reduce hallucinations in LVLMs, outperforming state-of-the-art methods across various benchmarks.

</details>

---

## 126. Cloud Object Detector Adaptation by Integrating Different Source Knowledge

- [ ] Cloud Object Detector Adaptation by Integrating Different Source Knowledge | https://neurips.cc/virtual/2024/poster/95127

- **Link**: https://neurips.cc/virtual/2024/poster/95127

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose to explore an interesting and promising problem, Cloud Object Detector Adaptation (CODA), where the target domain leverages detections provided by a large cloud model to build a target detector. Despite with powerful generalization capability, the cloud model still cannot achieve error-free detection in a specific target domain. In this work, we present a novel Cloud Object detector adaptation method by Integrating different source kNowledge (COIN). The key idea is to incorporate a public vision-language model (CLIP) to distill positive knowledge while refining negative knowledge for adaptation by self-promotion gradient direction alignment. To that end, knowledge dissemination, separation, and distillation are carried out successively. Knowledge dissemination combines knowledge from cloud detector and CLIP model to initialize a target detector and a CLIP detector in target domain. By matching CLIP detector with the cloud detector, knowledge separation categorizes detections into three parts: consistent, inconsistent and private detections such that divide-and-conquer strategy can be used for knowledge distillation. Consistent and private detections are directly used to train target detector; while inconsistent detections are fused based on a consistent knowledge generation network, which is trained by aligning the gradient direction of inconsistent detections to that of consistent detections, because it provides a direction toward an optimal target detector. Experiment results demonstrate that the proposed COIN method achieves the state-of-the-art performance.

</details>

---

## 127. Who Evaluates the Evaluations? Objectively Scoring Text-to-Image Prompt Coherence Metrics with T2IScoreScore (TS2)

- [ ] Who Evaluates the Evaluations? Objectively Scoring Text-to-Image Prompt Coherence Metrics with T2IScoreScore (TS2) | https://neurips.cc/virtual/2024/poster/95132

- **Link**: https://neurips.cc/virtual/2024/poster/95132

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With advances in the quality of text-to-image (T2I) models has come interest in benchmarking their prompt faithfulness---the semantic coherence of generated images to the prompts they were conditioned on. A variety of T2I faithfulness metrics have been proposed, leveraging advances in cross-modal embeddings and vision-language models (VLMs). However, these metrics are not rigorously compared and benchmarked, instead presented with correlation to human Likert scores over a set of easy-to-discriminate images against seemingly weak baselines. We introduce T2IScoreScore, a curated set of semantic error graphs containing a prompt and a set of increasingly erroneous images. These allow us to rigorously judge whether a given prompt faithfulness metric can correctly order images with respect to their objective error count and significantly discriminate  between different error nodes, using meta-metric scores derived from established statistical tests. Surprisingly, we find that the state-of-the-art VLM-based metrics (e.g., TIFA, DSG, LLMScore, VIEScore) we tested fail to significantly outperform simple (and supposedly worse) feature-based metrics like CLIPScore, particularly on a hard subset of naturally-occurring T2I model errors. TS2 will enable the development of better T2I prompt faithfulness metrics through more rigorous comparison of their conformity to expected orderings and separations under objective criteria.

</details>

---

## 128. Coarse-to-Fine Concept Bottleneck Models

- [ ] Coarse-to-Fine Concept Bottleneck Models | https://neurips.cc/virtual/2024/poster/95178

- **Link**: https://neurips.cc/virtual/2024/poster/95178

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Deep learning algorithms have recently gained significant attention due to their impressive performance. However, their high complexity and un-interpretable mode of operation hinders their confident deployment in real-world safety-critical tasks. This work targets ante hoc interpretability, and specifically Concept Bottleneck Models (CBMs). Our goal is to design a framework that admits a highly interpretable decision making process with respect to human understandable concepts, on two levels of granularity. To this end, we propose a novel two-level concept discovery formulation leveraging: (i) recent advances in vision-language models, and (ii) an innovative formulation for coarse-to-fine concept selection via data-driven and sparsity inducing Bayesian arguments. Within this framework, concept information does not solely rely on the similarity between the whole image and general unstructured concepts; instead, we introduce the notion of concept hierarchy to uncover and exploit more granular concept information residing in patch-specific regions of the image scene. As we experimentally show, the proposed construction not only outperforms recent CBM approaches, but also yields a principled framework towards interpetability.

</details>

---

## 129. EZ-HOI: VLM Adaptation via Guided Prompt Learning for Zero-Shot HOI Detection

- [ ] EZ-HOI: VLM Adaptation via Guided Prompt Learning for Zero-Shot HOI Detection | https://neurips.cc/virtual/2024/poster/95203

- **Link**: https://neurips.cc/virtual/2024/poster/95203

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Detecting Human-Object Interactions (HOI) in zero-shot settings, where models must handle unseen classes, poses significant challenges. Existing methods that rely on aligning visual encoders with large Vision-Language Models (VLMs) to tap into the extensive knowledge of VLMs, require large, computationally expensive models and encounter training difficulties. Adapting VLMs with prompt learning offers an alternative to direct alignment. However, fine-tuning on task-specific datasets often leads to overfitting to seen classes and suboptimal performance on unseen classes, due to the absence of unseen class labels. To address these challenges, we introduce a novel prompt learning-based framework for Efficient Zero-Shot HOI detection (EZ-HOI). First, we introduce Large Language Model (LLM) and VLM guidance for learnable prompts, integrating detailed HOI descriptions and visual semantics to adapt VLMs to HOI tasks. However, because training datasets contain seen-class labels alone, fine-tuning VLMs on such datasets tends to optimize learnable prompts for seen classes instead of unseen ones. Therefore, we design prompt learning for unseen classes using information from related seen classes, with LLMs utilized to highlight the differences between unseen and related seen classes. Quantitative evaluations on benchmark datasets demonstrate that our EZ-HOI achieves state-of-the-art performance across various zero-shot settings with only 10.35\% to 33.95\% of the trainable parameters compared to existing methods. Code is available at https://github.com/ChelsieLei/EZ-HOI.

</details>

---

## 130. Aligning Audio-Visual Joint Representations with an Agentic Workflow

- [ ] Aligning Audio-Visual Joint Representations with an Agentic Workflow | https://neurips.cc/virtual/2024/poster/95239

- **Link**: https://neurips.cc/virtual/2024/poster/95239

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual content and accompanied audio signals naturally formulate a joint representation to improve audio-visual (AV) related applications. While studies develop various AV representation learning frameworks, the importance of AV data alignment is usually undermined for achieving high-quality representation. We observe that an audio signal may contain background noise interference. Also, non-synchronization may appear between audio and video streams. These non-strict data alignment limits representation quality and downgrade application performance. In this paper, we propose to improve AV joint representations from a data-centric perspective by aligning audio signals to visual data. Our alignment is conducted in an agentic workflow controlled by an LLM-based assistant named AVAgent. For each input AV data pair, our AVAgent uses a multi-modal LLM to convert audio and visual data into language descriptions separately (i.e., tool use). Then, AVAgent reasons whether this paired data is aligned well and plans to edit the audio signal if needed (i.e., planning). The audio editing is executed by predefined actions that filter noise or augment data. Moreover, we use a VLM to evaluate how modified audio signals match the visual content and provide feedback to AVAgent (i.e., reflection). The tool use, planning, and reflection steps operate cyclically to become an agentic workflow where audio signals are gradually aligned to visual content. To this end, existing methods can directly leverage the aligned AV data via our agentic workflow to improve AV joint representations. The experimental results comprehensively demonstrate the state-of-the-art performance of the proposed approach against previous baselines in diverse downstream tasks.

</details>

---

## 131. DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution

- [ ] DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution | https://neurips.cc/virtual/2024/poster/95242

- **Link**: https://neurips.cc/virtual/2024/poster/95242

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable comprehension and reasoning capabilities with complex language and visual data.These advances have spurred the vision of establishing a generalist robotic MLLM proficient in understanding complex human instructions and accomplishing various embodied tasks, whose feasibility has been recently verified~\cite{rt-2,rt-x}.However, developing MLLMs for real-world robots is challenging due to the typically limited computation and memory capacities available on robotic platforms. In contrast, the inference of MLLMs usually incorporates storing billions of parameters and performing tremendous computation, imposing significant hardware demands.In our paper, we seek to address this challenge by leveraging an intriguing observation: relatively easier situations make up the bulk of the procedure of controlling robots to fulfill diverse tasks, and they generally require far smaller models to obtain the correct robotic actions.Motivated by this observation, we propose a \emph{DynamicEarly-Exit for Robotic MLLM} (DeeR) framework that automatically adjusts the size of the activated MLLM based on each situation at hand. The approach leverages a multi-exit architecture in MLLMs, which allows the model to cease processing once a proper size of the model has been activated for a specific situation, thus avoiding further redundant computation. Additionally, we develop novel algorithms that establish early-termination criteria for DeeR, conditioned on predefined demands such as average computational cost (\emph{i.e.}, power consumption), as well as peak computational consumption (\emph{i.e.}, latency) and GPU memory usage. These enhancements ensure that DeeR operates efficiently under varying resource constraints while maintaining competitive performance.Moreover, we design a tailored training method for integrating temporal information on top of such multi-exit architectures to predict actions reasonably. On the CALVIN robot manipulation benchmark, DeeR demonstrates significant reductions in computational costs by 5.2-6.5x and GPU memory by 2x without compromising performance.Code and checkpoints are available at https://github.com/yueyang130/DeeR-VLA.

</details>

---

## 132. Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem

- [ ] Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem | https://neurips.cc/virtual/2024/poster/95266

- **Link**: https://neurips.cc/virtual/2024/poster/95266

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent work has documented striking heterogeneity in the performance of state-of-the-art vision language models (VLMs), including both multimodal language models and text-to-image models. These models are able to describe and generate a diverse array of complex, naturalistic images, yet they exhibit surprising failures on basic multi-object reasoning tasks -- such as counting, localization, and simple forms of visual analogy -- that humans perform with near perfect accuracy. To better understand this puzzling pattern of successes and failures, we turn to theoretical accounts of the binding problem in cognitive science and neuroscience, a fundamental problem that arises when a shared set of representational resources must be used to represent distinct entities (e.g., to represent multiple objects in an image), necessitating the use of serial processing to avoid interference. We find that many of the puzzling failures of state-of-the-art VLMs can be explained as arising due to the binding problem, and that these failure modes are strikingly similar to the limitations exhibited by rapid, feedforward processing in the human brain.

</details>

---

## 133. What Makes CLIP More Robust to Long-Tailed Pre-Training Data? A Controlled Study for Transferable Insights

- [ ] What Makes CLIP More Robust to Long-Tailed Pre-Training Data? A Controlled Study for Transferable Insights | https://neurips.cc/virtual/2024/poster/95296

- **Link**: https://neurips.cc/virtual/2024/poster/95296

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Severe data imbalance naturally exists among web-scale vision-language datasets. Despite this, we find CLIP pre-trained thereupon exhibits notable robustness to the data imbalance compared to supervised learning, and demonstrates significant effectiveness in learning generalizable representations. With an aim to investigate the reasons behind this finding, we conduct controlled experiments to study various underlying factors, and reveal that CLIP's pretext task forms a dynamic classification problem wherein only a subset of classes is present in training. This isolates the bias from dominant classes and implicitly balances the learning signal. Furthermore, the robustness and discriminability of CLIP improve with more descriptive language supervision, larger data scale, and broader open-world concepts, which are inaccessible to supervised learning. Our study not only uncovers the mechanisms behind CLIP's generalizability beyond data imbalance but also provides transferable insights for the research community. The findings are validated in both supervised and self-supervised learning, enabling models trained on imbalanced data to achieve CLIP-level performance on diverse recognition tasks. Code and data are available at: https://github.com/CVMI-Lab/clip-beyond-tail.

</details>

---

## 134. Hierarchical Visual Feature Aggregation for OCR-Free Document Understanding

- [ ] Hierarchical Visual Feature Aggregation for OCR-Free Document Understanding | https://neurips.cc/virtual/2024/poster/95304

- **Link**: https://neurips.cc/virtual/2024/poster/95304

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present a novel OCR-free document understanding framework based on pretrained Multimodal Large Language Models (MLLMs). Our approach employs multi-scale visual features to effectively handle various font sizes within document images.To address the increasing costs of considering the multi-scale visual inputs for MLLMs, we propose the Hierarchical Visual Feature Aggregation (HVFA) module, designed to reduce the number of input tokens to LLMs. Leveraging a feature pyramid with cross-attentive pooling, our approach effectively manages the trade-off between information loss and efficiency without being affected by varying document image sizes.Furthermore, we introduce a novel instruction tuning task, which facilitates the model's text-reading capability by learning to predict the relative positions of input text, eventually minimizing the risk of truncated text caused by the limited capacity of LLMs.Comprehensive experiments validate the effectiveness of our approach, demonstrating superior performance in various document understanding tasks.

</details>

---

## 135. Does Video-Text Pretraining Help Open-Vocabulary Online Action Detection?

- [ ] Does Video-Text Pretraining Help Open-Vocabulary Online Action Detection? | https://neurips.cc/virtual/2024/poster/95303

- **Link**: https://neurips.cc/virtual/2024/poster/95303

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video understanding relies on accurate action detection for temporal analysis. However, existing mainstream methods have limitations in real-world applications due to their offline and closed-set evaluation approaches, as well as their dependence on manual annotations. To address these challenges and enable real-time action understanding in open-world scenarios, we propose OV-OAD, a zero-shot online action detector that leverages vision-language models and learns solely from text supervision. By introducing an object-centered decoder unit into a Transformer-based model, we aggregate frames with similar semantics using video-text correspondence. Extensive experiments on four action detection benchmarks demonstrate that OV-OAD outperforms other advanced zero-shot methods. Specifically, it achieves 37.5\% mean average precision on THUMOS’14 and 73.8\% calibrated average precision on TVSeries. This research establishes a robust baseline for zero-shot transfer in online action detection, enabling scalable solutions for open-world temporal understanding. The code will be available for download at \url{https://github.com/OpenGVLab/OV-OAD}.

</details>

---

## 136. Relationship Prompt Learning is Enough for Open-Vocabulary Semantic Segmentation

- [ ] Relationship Prompt Learning is Enough for Open-Vocabulary Semantic Segmentation | https://neurips.cc/virtual/2024/poster/95318

- **Link**: https://neurips.cc/virtual/2024/poster/95318

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation (OVSS) aims to segment unseen classes without corresponding labels. Existing Vision-Language Model (VLM)-based methods leverage VLM's rich knowledge to enhance additional explicit segmentation-specific networks, yielding competitive results, but at the cost of extensive training cost. To reduce the cost, we attempt to enable VLM to directly produce the segmentation results without any segmentation-specific networks. Prompt learning offers a direct and parameter-efficient approach, yet it falls short in guiding VLM for pixel-level visual classification. Therefore, we propose the ${\bf R}$elationship ${\bf P}$rompt ${\bf M}$odule (${\bf RPM}$), which generates the relationship prompt that directs VLM to extract pixel-level semantic embeddings suitable for OVSS. Moreover, RPM integrates with VLM to construct the ${\bf R}$elationship ${\bf P}$rompt ${\bf N}$etwork (${\bf RPN}$), achieving OVSS without any segmentation-specific networks. RPN attains state-of-the-art performance with merely about ${\bf 3M}$ trainable parameters (2\% of total parameters).

</details>

---

## 137. FlexCap: Describe Anything in Images in Controllable Detail

- [ ] FlexCap: Describe Anything in Images in Controllable Detail | https://neurips.cc/virtual/2024/poster/95332

- **Link**: https://neurips.cc/virtual/2024/poster/95332

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce FlexCap, a vision-language model that generates region-specific descriptions of varying lengths. FlexCap is trained to produce length-conditioned captions for input boxes, enabling control over information density, with descriptions ranging from concise object labels to detailed captions.  To achieve this, we create large-scale training datasets of image region descriptions with varying lengths from captioned web images. We demonstrate FlexCap’s effectiveness in several applications: first, it achieves strong performance in dense captioning tasks on the Visual Genome dataset. Second, we show how FlexCap’s localized descriptions can serve as input to a large language model to create a visual question answering (VQA) system, achieving state-of-the-art zero-shot performance on multiple VQA benchmarks. Our experiments illustrate FlexCap’s utility for tasks including image labeling, object attribute recognition, and visual dialog. Project webpage: https://flex-cap.github.io.

</details>

---

## 138. Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting

- [ ] Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting | https://neurips.cc/virtual/2024/poster/95379

- **Link**: https://neurips.cc/virtual/2024/poster/95379

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models, such as CLIP, have shown impressive generalization capacities when using appropriate text descriptions. While optimizing prompts on downstream labeled data has proven effective in improving performance, these methods entail labor costs for annotations and are limited by their quality. Additionally, since CLIP is pre-trained on highly imbalanced Web-scale data, it suffers from inherent label bias that leads to suboptimal performance.  To tackle the above challenges, we propose a label-**F**ree p**ro**mpt distribution **l**earning and b**i**as **c**orrection framework, dubbed as **Frolic**, which boosts zero-shot performance without the need for labeled data. Specifically, our Frolic learns distributions over prompt prototypes to capture diverse visual representations and adaptively fuses these with the original CLIP through confidence matching.This fused model is further enhanced by correcting label bias via a label-free logit adjustment. Notably, our method is not only training-free but also circumvents the necessity for hyper-parameter tuning. Extensive experimental results across 16 datasets demonstrate the efficacy of our approach, particularly outperforming the state-of-the-art by an average of $2.6\%$ on 10 datasets with CLIP ViT-B/16 and achieving an average margin of $1.5\%$ on ImageNet and its five distribution shifts with CLIP ViT-B/16. Codes are available in [https://github.com/zhuhsingyuu/Frolic](https://github.com/zhuhsingyuu/Frolic).

</details>

---

## 139. Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration

- [ ] Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration | https://neurips.cc/virtual/2024/poster/95398

- **Link**: https://neurips.cc/virtual/2024/poster/95398

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Mobile device operation tasks are increasingly becoming a popular multi-modal AI application scenario. Current Multi-modal Large Language Models (MLLMs), constrained by their training data, lack the capability to function effectively as operation assistants. Instead, MLLM-based agents, which enhance capabilities through tool invocation, are gradually being applied to this scenario. However, the two major navigation challenges in mobile device operation tasks — task progress navigation and focus content navigation — are difficult to effectively solve under the single-agent architecture of existing work. This is due to the overly long token sequences and the interleaved text-image data format, which limit performance. To address these navigation challenges effectively, we propose Mobile-Agent-v2, a multi-agent architecture for mobile device operation assistance. The architecture comprises three agents: planning agent, decision agent, and reflection agent. The planning agent condenses lengthy, interleaved image-text history operations and screens summaries into a pure-text task progress, which is then passed on to the decision agent. This reduction in context length makes it easier for decision agent to navigate the task progress. To retain focus content, we design a memory unit that updates with task progress by decision agent. Additionally, to correct erroneous operations, the reflection agent observes the outcomes of each operation and handles any mistake accordingly. Experimental results indicate that Mobile-Agent-v2 achieves over a 30% improvement in task completion compared to the single-agent architecture of Mobile-Agent. The code is open-sourced at https://github.com/X-PLUG/MobileAgent.

</details>

---

## 140. Seeing the Image: Prioritizing Visual Correlation by Contrastive Alignment

- [ ] Seeing the Image: Prioritizing Visual Correlation by Contrastive Alignment | https://neurips.cc/virtual/2024/poster/95404

- **Link**: https://neurips.cc/virtual/2024/poster/95404

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing image-text modality alignment in Vision Language Models (VLMs) treats each text token equally in an autoregressive manner. Despite being simple and effective, this method results in sub-optimal cross-modal alignment by over-emphasizing the text tokens that are less correlated with or even contradictory with the input images. In this paper, we advocate for distinct contributions for each text token based on its visual correlation. Specifically, we present by contrasting image inputs, the difference in prediction logits on each text token provides strong guidance of visual correlation. We therefore introduce Contrastive Alignment (CAL), a simple yet effective re-weighting strategy that prioritizes training visually correlated tokens. Our experimental results demonstrate that CAL consistently improves different types of VLMs across different resolutions and model sizes on various benchmark datasets. Importantly, our method incurs minimal additional computational overhead, rendering it highly efficient compared to alternative data scaling strategies.

</details>

---

## 141. VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation

- [ ] VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation | https://neurips.cc/virtual/2024/poster/95449

- **Link**: https://neurips.cc/virtual/2024/poster/95449

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A well-known dilemma in large vision-language models (e.g., GPT-4, LLaVA) is that while increasing the number of vision tokens generally enhances visual understanding, it also significantly raises memory and computational costs, especially in long-term, dense video frame streaming scenarios. Although learnable approaches like Q-Former and Perceiver Resampler have been developed to reduce the vision token burden, they overlook the context causally modeled by LLMs (i.e., key-value cache), potentially leading to missed visual cues when addressing user queries. In this paper, we introduce a novel approach to reduce vision compute by leveraging redundant vision tokens ``skipping layers'' rather than decreasing the number of vision tokens. Our method, VideoLLM-MoD, is inspired by mixture-of-depths LLMs and addresses the challenge of numerous vision tokens in long-term or streaming video. Specifically, for certain transformer layer, we learn to skip the computation for a high proportion (e.g., 80\%) of vision tokens, passing them directly to the next layer. This approach significantly enhances model efficiency, achieving approximately 42% time and 30% memory savings for the entire training. Moreover, our method reduces the computation in the context and avoid decreasing the vision tokens, thus preserving or even improving performance compared to the vanilla model. We conduct extensive experiments to demonstrate the effectiveness of VideoLLM-MoD, showing its state-of-the-art results on multiple benchmarks, including narration, forecasting, and summarization tasks in COIN, Ego4D, and Ego-Exo4D datasets. The code and checkpoints will be made available at github.com/showlab/VideoLLM-online.

</details>

---

## 142. UniAudio 1.5: Large Language Model-Driven Audio Codec is A Few-Shot Audio Task Learner

- [ ] UniAudio 1.5: Large Language Model-Driven Audio Codec is A Few-Shot Audio Task Learner | https://neurips.cc/virtual/2024/poster/95454

- **Link**: https://neurips.cc/virtual/2024/poster/95454

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Language models (LLMs) have demonstrated supreme capabilities in textual understanding and generation, but cannot be directly applied to cross-modal tasks without fine-tuning. This paper proposes a cross-modal in-context learning approach, empowering the frozen LLMs to achieve multiple audio tasks in a few-shot style without any parameter update. Specifically, we propose a novel LLM-driven audio codec model, LLM-Codec, which transfers the audio modality into textual space by representing audio tokens with words or sub-words from the LLM vocabulary, while maintaining high audio reconstruction quality.The key idea is to reduce the modality heterogeneity between text and audio by compressing the audio modality into the well-trained textual space of LLMs. Thus, the audio representation can be viewed as a new \textit{foreign language}, and LLMs can learn the new \textit{foreign language} with several demonstrations. In experiments, we investigate the performance of the proposed approach across multiple audio understanding and generation tasks, \textit{e.g.} speech emotion classification, audio classification, text-to-speech generation, speech enhancement, etc. Experimental results show that LLMs equipped with the LLM-Codec, named as UniAudio 1.5, prompted by only a few examples, can perform effectively in simple scenarios, validating our cross-modal in-context learning approach.To facilitate research on few-shot audio task learning and multi-modal LLMs, we have open-sourced the LLM-Codec model.

</details>

---

## 143. DiPEx: Dispersing Prompt Expansion for Class-Agnostic Object Detection

- [ ] DiPEx: Dispersing Prompt Expansion for Class-Agnostic Object Detection | https://neurips.cc/virtual/2024/poster/95458

- **Link**: https://neurips.cc/virtual/2024/poster/95458

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Class-agnostic object detection (OD) can be a cornerstone or a bottleneck for many downstream vision tasks. Despite considerable advancements in bottom-up and multi-object discovery methods that leverage basic visual cues to identify salient objects, consistently achieving a high recall rate remains difficult due to the diversity of object types and their contextual complexity. In this work, we investigate using vision-language models (VLMs) to enhance object detection via a self-supervised prompt learning strategy. Our initial findings indicate that manually crafted text queries often result in undetected objects, primarily because detection confidence diminishes when the query words exhibit semantic overlap. To address this, we propose a Dispersing Prompt Expansion (DiPEx) approach. DiPEx progressively learns to expand a set of distinct, non-overlapping hyperspherical prompts to enhance recall rates, thereby improving performance in downstream tasks such as out-of-distribution OD. Specifically, DiPEx initiates the process by self-training generic parent prompts and selecting the one with the highest semantic uncertainty for further expansion. The resulting child prompts are expected to inherit semantics from their parent prompts while capturing more fine-grained semantics. We apply dispersion losses to ensure high inter-class discrepancy among child prompts while preserving semantic consistency between parent-child prompt pairs. To prevent excessive growth of the prompt sets, we utilize the maximum angular coverage (MAC) of the semantic space as a criterion for early termination. We demonstrate the effectiveness of DiPEx through extensive class-agnostic OD and OOD-OD experiments on MS-COCO and LVIS, surpassing other prompting methods by up to 20.1% in AR and achieving a 21.3% AP improvement over SAM.

</details>

---

## 144. Why are Visually-Grounded Language Models Bad at Image Classification?

- [ ] Why are Visually-Grounded Language Models Bad at Image Classification? | https://neurips.cc/virtual/2024/poster/95478

- **Link**: https://neurips.cc/virtual/2024/poster/95478

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image classification is one of the most fundamental capabilities of machine vision intelligence. In this work, we revisit the image classification task using visually-grounded language models (VLMs) such as GPT-4V and LLaVA. We find that existing proprietary and public VLMs, despite often using CLIP as a vision encoder and having many more parameters, significantly underperform CLIP on standard image classification benchmarks like ImageNet. To understand the reason, we explore several hypotheses concerning the inference algorithms, training objectives, and data processing in VLMs. Our analysis reveals that the primary cause is data-related: critical information for image classification is encoded in the VLM's latent space but can only be effectively decoded with enough training data. Specifically, there is a strong correlation between the frequency of class exposure during VLM training and instruction-tuning and the VLM's performance in those classes; when trained with sufficient data, VLMs can match the accuracy of state-of-the-art classification models. Based on these findings, we enhance a VLM by integrating classification-focused datasets into its training, and demonstrate that the enhanced classification performance of the VLM transfers to its general capabilities, resulting in an improvement of 11.8% on the newly collected ImageWikiQA dataset.

</details>

---

## 145. A Concept-Based Explainability Framework for Large Multimodal Models

- [ ] A Concept-Based Explainability Framework for Large Multimodal Models | https://neurips.cc/virtual/2024/poster/95482

- **Link**: https://neurips.cc/virtual/2024/poster/95482

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal models (LMMs) combine unimodal encoders and large language models (LLMs) to perform multimodal tasks. Despite recent advancements towards the interpretability of these models, understanding internal representations of LMMs remains largely a mystery. In this paper, we present a novel framework for the interpretation of LMMs. We propose a dictionary learning based approach, applied to the representation of tokens. The elements of the learned dictionary correspond to our proposed concepts. We show that these concepts are well semantically grounded in both vision and text. Thus we refer to these as ``multi-modal concepts''. We qualitatively and quantitatively evaluate the results of the learnt concepts. We show that the extracted multimodal concepts are useful to interpret representations of test samples. Finally, we evaluate the disentanglement between different concepts and the quality of grounding concepts visually and textually. Our implementation is publicly available: https://github.com/mshukor/xl-vlms.

</details>

---

## 146. CLIPCEIL: Domain Generalization through CLIP via Channel rEfinement and Image-text aLignment

- [ ] CLIPCEIL: Domain Generalization through CLIP via Channel rEfinement and Image-text aLignment | https://neurips.cc/virtual/2024/poster/95489

- **Link**: https://neurips.cc/virtual/2024/poster/95489

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Domain generalization (DG) is a fundamental yet challenging topic in machine learning. Recently, the remarkable zero-shot capabilities of the large pre-trained vision-language model (e.g., CLIP) have made it popular for various downstream tasks. However, the effectiveness of this capacity often degrades when there are shifts in data distribution during testing compared to the training data. In this paper, we propose a novel method, known as CLIPCEIL, a model that utilizes Channel rEfinement and Image-text aLignment to facilitate the CLIP to the inaccessible $\textit{out-of-distribution}$ test datasets that exhibit domain shifts. Specifically, we refine the feature channels in the visual domain to ensure they contain domain-invariant and class-relevant features by using a lightweight adapter. This is achieved by minimizing the inter-domain variance while maximizing the inter-class variance. In the meantime, we ensure the image-text alignment by aligning text embeddings of the class descriptions and their corresponding image embedding while further removing the domain-specific features. Moreover, our model integrates multi-scale CLIP features by utilizing a self-attention fusion module, technically implemented through one Transformer layer. Extensive experiments on five widely used benchmark datasets demonstrate that CLIPCEIL outperforms the existing state-of-the-art methods. The source code is available at \url{https://github.com/yuxi120407/CLIPCEIL}.

</details>

---

## 147. LG-CAV: Train Any Concept Activation Vector with Language Guidance

- [ ] LG-CAV: Train Any Concept Activation Vector with Language Guidance | https://neurips.cc/virtual/2024/poster/95495

- **Link**: https://neurips.cc/virtual/2024/poster/95495

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Concept activation vector (CAV) has attracted broad research interest in explainable AI, by elegantly attributing model predictions to specific concepts. However, the training of CAV often necessitates a large number of high-quality images, which are expensive to curate and thus limited to a predefined set of concepts. To address this issue, we propose Language-Guided CAV (LG-CAV) to harness the abundant concept knowledge within the certain pre-trained vision-language models (e.g., CLIP). This method allows training any CAV without labeled data, by utilizing the corresponding concept descriptions as guidance. To bridge the gap between vision-language model and the target model, we calculate the activation values of concept descriptions on a common pool of images (probe images) with vision-language model and utilize them as language guidance to train the LG-CAV. Furthermore, after training high-quality LG-CAVs related to all the predicted classes in the target model, we propose the activation sample reweighting (ASR), serving as a model correction technique, to improve the performance of the target model in return. Experiments on four datasets across nine architectures demonstrate that LG-CAV achieves significantly superior quality to previous CAV methods given any concept, and our model correction method achieves state-of-the-art performance compared to existing concept-based methods. Our code is available at https://github.com/hqhQAQ/LG-CAV.

</details>

---

## 148. ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models

- [ ] ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/95574

- **Link**: https://neurips.cc/virtual/2024/poster/95574

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we propose a training-free method to inject visual prompts into Multimodal Large Language Models (MLLMs) through learnable latent variable optimization. We observe that attention, as the core module of MLLMs, connects text prompt tokens and visual tokens, ultimately determining the final results. Our approach involves adjusting visual tokens from the MLP output during inference, controlling the attention response to ensure text prompt tokens attend to visual tokens in referring regions. We optimize a learnable latent variable based on an energy function, enhancing the strength of referring regions in the attention map. This enables detailed region description and reasoning without the need for substantial training costs or model retraining. Our method offers a promising direction for integrating referring abilities into MLLMs, and supports referring with box, mask, scribble and point. The results demonstrate that our method exhibits out-of-domain generalization and interpretability.

</details>

---

## 149. Privacy Backdoors: Enhancing Membership Inference through Poisoning Pre-trained Models

- [ ] Privacy Backdoors: Enhancing Membership Inference through Poisoning Pre-trained Models | https://neurips.cc/virtual/2024/poster/95643

- **Link**: https://neurips.cc/virtual/2024/poster/95643

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

It is commonplace to produce application-specific models by fine-tuning large pre-trained models using a small bespoke dataset. The widespread availability of foundation model checkpoints on the web poses considerable risks, including the vulnerability to backdoor attacks. In this paper, we unveil a new vulnerability: the privacy backdoor attack. This black-box privacy attack aims to amplify the privacy leakage that arises when fine-tuning a model: when a victim fine-tunes a backdoored model, their training data will be leaked at a significantly higher rate than if they had fine-tuned a typical model. We conduct extensive experiments on various datasets and models, including both vision-language models (CLIP) and large language models, demonstrating the broad applicability and effectiveness of such an attack. Additionally, we carry out multiple ablation studies with different fine-tuning methods and inference strategies to thoroughly analyze this new threat. Our findings highlight a critical privacy concern within the machine learning community and call for a re-evaluation of safety protocols in the use of open-source pre-trained models.

</details>

---

## 150. Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?

- [ ] Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable? | https://neurips.cc/virtual/2024/poster/95656

- **Link**: https://neurips.cc/virtual/2024/poster/95656

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, interpretable machine learning has re-explored concept bottleneck models (CBM). An advantage of this model class is the user's ability to intervene on predicted concept values, affecting the downstream output. In this work, we introduce a method to perform such concept-based interventions on pretrained neural networks, which are not interpretable by design, only given a small validation set with concept labels. Furthermore, we formalise the notion of intervenability as a measure of the effectiveness of concept-based interventions and leverage this definition to fine-tune black boxes. Empirically, we explore the intervenability of black-box classifiers on synthetic tabular and natural image benchmarks. We focus on backbone architectures of varying complexity, from simple, fully connected neural nets to Stable Diffusion. We demonstrate that the proposed fine-tuning improves intervention effectiveness and often yields better-calibrated predictions. To showcase the practical utility of our techniques, we apply them to deep chest X-ray classifiers and show that fine-tuned black boxes are more intervenable than CBMs. Lastly, we establish that our methods are still effective under vision-language-model-based concept annotations, alleviating the need for a human-annotated validation set.

</details>

---

## 151. Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control

- [ ] Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control | https://neurips.cc/virtual/2024/poster/95658

- **Link**: https://neurips.cc/virtual/2024/poster/95658

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Embodied AI agents require a fine-grained understanding of the physical world mediated through visual and language inputs. Such capabilities are difficult to learn solely from task-specific data. This has led to the emergence of pre-trained vision-language models as a tool for transferring representations learned from internet-scale data to downstream tasks and new domains. However, commonly used contrastively trained representations such as in CLIP have been shown to fail at enabling embodied agents to gain a sufficiently fine-grained scene understanding—a capability vital for control. To address this shortcoming, we consider representations from pre-trained text-to-image diffusion models, which are explicitly optimized to generate images from text prompts and as such, contain text-conditioned representations that reflect highly fine-grained visuo-spatial information. Using pre-trained text-to-image diffusion models, we construct Stable Control Representations which allow learning downstream control policies that generalize to complex, open-ended environments. We show that policies learned using Stable Control Representations are competitive with state-of-the-art representation learning approaches across a broad range of simulated control settings, encompassing challenging manipulation and navigation tasks. Most notably, we show that Stable Control Representations enable learning policies that exhibit state-of-the-art performance on OVMM, a difficult open-vocabulary navigation benchmark.

</details>

---

## 152. Multi-Object Hallucination in Vision Language Models

- [ ] Multi-Object Hallucination in Vision Language Models | https://neurips.cc/virtual/2024/poster/95666

- **Link**: https://neurips.cc/virtual/2024/poster/95666

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision language models (LVLMs) often suffer from object hallucination, producing objects not present in the given images. While current benchmarks for object hallucination primarily concentrate on the presence of a single object class rather than individual entities, this work systematically investigates multi-object hallucination, examining how models misperceive (e.g., invent nonexistent objects or become distracted) when tasked with focusing on multiple objects simultaneously.We introduce Recognition-based Object Probing Evaluation (ROPE), an automated evaluation protocol that considers the distribution of object classes within a single image during testing and uses visual referring prompts to eliminate ambiguity. With comprehensive empirical studies and analysis of potential factors leading to multi-object hallucination, we found that (1) LVLMs suffer more hallucinations when focusing on multiple objects compared to a single object. (2) The tested object class distribution affects hallucination behaviors, indicating that LVLMs may follow shortcuts and spurious correlations.(3) Hallucinatory behaviors are influenced by data-specific factors, salience and frequency, and model intrinsic behaviors.We hope to enable LVLMs to recognize and reason about multiple objects that often occur in realistic visual scenes, provide insights, and quantify our progress towards mitigating the issues.

</details>

---

## 153. Accelerating Pre-training of Multimodal LLMs via Chain-of-Sight

- [ ] Accelerating Pre-training of Multimodal LLMs via Chain-of-Sight | https://neurips.cc/virtual/2024/poster/95674

- **Link**: https://neurips.cc/virtual/2024/poster/95674

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces Chain-of-Sight, a vision-language bridge module that accelerates the pre-training of Multimodal Large Language Models (MLLMs). Our approach employs a sequence of visual resamplers that capture visual details at various spacial scales.This architecture not only leverages global and local visual contexts effectively, but also facilitates the flexible extension of visual tokens through a compound token scaling strategy, allowing up to a 16x increase in the token count post pre-training.Consequently, Chain-of-Sight requires significantly fewer visual tokens in the pre-training phase compared to the fine-tuning phase. This intentional reduction of visual tokens during pre-training notably accelerates the pre-training process, cutting down the wall-clock training time by $\sim$73\%.Empirical results on a series of vision-language benchmarks reveal that the pre-train acceleration through Chain-of-Sight is achieved without sacrificing performance, matching or surpassing the standard pipeline of utilizing all visual tokens throughout the entire training process. Further scaling up the number of visual tokens for pre-training leads to stronger performances, competitive to existing approaches in a series of benchmarks.

</details>

---

## 154. RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation

- [ ] RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation | https://neurips.cc/virtual/2024/poster/95690

- **Link**: https://neurips.cc/virtual/2024/poster/95690

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A fundamental objective in robot manipulation is to enable models to comprehend visual scenes and execute actions. Although existing Vision-Language-Action (VLA) models for robots can handle a range of basic tasks, they still face challenges in two areas: (1) insufficient reasoning ability to tackle complex tasks, and (2) high computational costs for VLA model fine-tuning and inference. The recently proposed state space model (SSM) known as Mamba demonstrates promising capabilities in non-trivial sequence modeling with linear inference complexity. Inspired by this, we introduce RoboMamba, an end-to-end robotic VLA model that leverages Mamba to deliver both robotic reasoning and action capabilities, while maintaining efficient fine-tuning and inference. Specifically, we first integrate the vision encoder with Mamba, aligning visual tokens with language embedding through co-training, empowering our model with visual common sense and robotic-related reasoning. To further equip RoboMamba with SE(3) pose prediction abilities, we explore an efficient fine-tuning strategy with a simple policy head. We find that once RoboMamba possesses sufficient reasoning capability, it can acquire manipulation skills with minimal fine-tuning parameters (0.1\% of the model) and time. In experiments, RoboMamba demonstrates outstanding reasoning capabilities on general and robotic evaluation benchmarks. Meanwhile, our model showcases impressive pose prediction results in both simulation and real-world experiments, achieving inference speeds 3 times faster than existing VLA models.

</details>

---

## 155. VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance

- [ ] VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance | https://neurips.cc/virtual/2024/poster/95698

- **Link**: https://neurips.cc/virtual/2024/poster/95698

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Concept Bottleneck Models (CBMs) provide interpretable prediction by introducing an intermediate Concept Bottleneck Layer (CBL), which encodes human-understandable concepts to explain models' decision. Recent works proposed to utilize Large Language Models and pre-trained Vision-Language Models to automate the training of CBMs, making it more scalable and automated. However, existing approaches still fall short in two aspects: First, the concepts predicted by CBL often mismatch the input image, raising doubts about the faithfulness of interpretation. Second, it has been shown that concept values encode unintended information: even a set of random concepts could achieve comparable test accuracy to state-of-the-art CBMs. To address these critical limitations, in this work, we propose a novel framework called Vision-Language-Guided Concept Bottleneck Model (VLG-CBM) to enable faithful interpretability with the benefits of boosted performance. Our method leverages off-the-shelf open-domain grounded object detectors to provide visually grounded concept annotation, which largely enhances the faithfulness of concept prediction while further improving the model performance. In addition, we propose a new metric called Number of Effective Concepts (NEC) to control the information leakage and provide better interpretability. Extensive evaluations across five standard benchmarks show that our method, VLG-CBM, outperforms existing methods by at least 4.27\% and up to 51.09\% on Accuracy at NEC=5 (denoted as ANEC-5), and by at least 0.45\% and up to 29.78\% on average accuracy (denoted as ANEC-avg), while preserving both faithfulness and interpretability of the learned concepts as demonstrated in extensive experiments.

</details>

---

## 156. Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models

- [ ] Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models | https://neurips.cc/virtual/2024/poster/95705

- **Link**: https://neurips.cc/virtual/2024/poster/95705

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) excel in generating textual responses from visual inputs, but their versatility raises security concerns. This study takes the first step in exposing VLMs’ susceptibility to data poisoning attacks that can manipulate responses to innocuous, everyday prompts. We introduce Shadowcast, a stealthy data poisoning attack where poison samples are visually indistinguishable from benign images with matching texts. Shadowcast demonstrates effectiveness in two attack types. The first is a traditional Label Attack, tricking VLMs into misidentifying class labels, such as confusing Donald Trump for Joe Biden. The second is a novel Persuasion Attack, leveraging VLMs’ text generation capabilities to craft persuasive and seemingly rational narratives for misinformation, such as portraying junk food as healthy. We show that Shadowcast effectively achieves the attacker’s intentions using as few as 50 poison samples. Crucially, the poisoned samples demonstrate transferability across different VLM architectures, posing a significant concern in black-box settings. Moreover, Shadowcast remains potent under realistic conditions involving various text prompts, training data augmentation, and image compression techniques. This work reveals how poisoned VLMs can disseminate convincing yet deceptive misinformation to everyday, benign users, emphasizing the importance of data integrity for responsible VLM deployments. Our code is available at: https://github.com/umd-huang-lab/VLM-Poisoning.

</details>

---

## 157. Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models

- [ ] Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models | https://neurips.cc/virtual/2024/poster/95711

- **Link**: https://neurips.cc/virtual/2024/poster/95711

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of large language and vision models (LLVMs) has been driven by advances in visual instruction tuning. Recently, open-source LLVMs have curated high-quality visual instruction tuning datasets and utilized additional vision encoders or multiple computer vision models in order to narrow the performance gap with powerful closed-source LLVMs. These advancements are attributed to multifaceted information required for diverse capabilities, including fundamental image understanding, real-world knowledge about common-sense and non-object concepts (e.g., charts, diagrams, symbols, signs, and math problems), and step-by-step procedures for solving complex questions. Drawing from the multifaceted information, we present a new efficient LLVM, Mamba-based traversal of rationales (Meteor), which leverages multifaceted rationale to enhance understanding and answering capabilities. To embed lengthy rationales containing abundant information, we employ the Mamba architecture, capable of processing sequential data with linear time complexity. We introduce a new concept of traversal of rationale that facilitates efficient embedding of rationale. Subsequently, the backbone multimodal language model (MLM) is trained to generate answers with the aid of rationale. Through these steps, Meteor achieves significant improvements in vision language performances across multiple evaluation benchmarks requiring diverse capabilities, without scaling up the model size or employing additional vision encoders and computer vision models.

</details>

---

## 158. Visual Perception by Large Language Model’s Weights

- [ ] Visual Perception by Large Language Model’s Weights | https://neurips.cc/virtual/2024/poster/95713

- **Link**: https://neurips.cc/virtual/2024/poster/95713

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing Multimodal Large Language Models (MLLMs) follow the paradigm that perceives visual information by aligning visual features with the input space of Large Language Models (LLMs) and concatenating visual tokens with text tokens to form a unified sequence input for LLMs. These methods demonstrate promising results on various vision-language tasks but are limited by the high computational effort due to the extended input sequence resulting from the involvement of visual tokens. In this paper, instead of input space alignment, we propose a novel parameter space alignment paradigm that represents visual information as model weights. For each input image, we use a vision encoder to extract visual features, convert features into perceptual weights, and merge the perceptual weights with LLM's weights. In this way, the input of LLM does not require visual tokens, which reduces the length of the input sequence and greatly improves efficiency. Following this paradigm, we propose VLoRA with the perceptual weights generator. The perceptual weights generator is designed to convert visual features to perceptual weights with low-rank property, exhibiting a form similar to LoRA. The experimental results show that our VLoRA achieves comparable performance on various benchmarks for MLLMs, while significantly reducing the computational costs for both training and inference. Code and models are released at \url{https://github.com/FeipengMa6/VLoRA}.

</details>

---

## 159. SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models

- [ ] SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models | https://neurips.cc/virtual/2024/poster/95720

- **Link**: https://neurips.cc/virtual/2024/poster/95720

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have demonstrated remarkable performance in 2D vision and language tasks. However, their ability to reason about spatial arrangements remains limited. In this work, we introduce Spatial Region GPT (SpatialRGPT) to enhance VLMs’ spatial perception and reasoning capabilities. SpatialRGPT advances VLMs’ spatial understanding through two key innovations: (i) a data curation pipeline that enables effective learning of regional representation from 3D scene graphs, and (ii) a flexible ``plugin'' module for integrating depth information into the visual encoder of existing VLMs. During inference, when provided with user-specified region proposals, SpatialRGPT can accurately perceive their relative directions and distances. Additionally, we propose SpatialRGBT-Bench, a benchmark with ground-truth 3D annotations encompassing indoor, outdoor, and simulated environments, for evaluating 3D spatial cognition in Vision-Language Models (VLMs). Our results demonstrate that SpatialRGPT significantly enhances performance in spatial reasoning tasks, both with and without local region prompts. The model also exhibits strong generalization capabilities, effectively reasoning about complex spatial relations and functioning as a region-aware dense reward annotator for robotic tasks. Code, dataset, and benchmark are released at https://www.anjiecheng.me/SpatialRGPT.

</details>

---

## 160. Dense Connector for MLLMs

- [ ] Dense Connector for MLLMs | https://neurips.cc/virtual/2024/poster/95751

- **Link**: https://neurips.cc/virtual/2024/poster/95751

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Do we fully leverage the potential of visual encoder in Multimodal Large Language Models (MLLMs)? The recent outstanding performance of MLLMs in multimodal understanding has garnered broad attention from both academia and industry. In the current MLLM rat race, the focus seems to be predominantly on the linguistic side. We witness the rise of larger and higher-quality instruction datasets, as well as the involvement of larger-sized LLMs. Yet, scant attention has been directed towards the visual signals utilized by MLLMs, often assumed to be the final high-level features extracted by a frozen visual encoder. In this paper, we introduce the Dense Connector - a simple, effective, and plug-and-play vision-language connector that significantly enhances existing MLLMs by leveraging multi-layer visual features, with minimal additional computational overhead. Building on this, we also propose the Efficient Dense Connector, which achieves performance comparable to LLaVA-v1.5 with only 25% of the visual tokens. Furthermore, our model, trained solely on images, showcases remarkable zero-shot capabilities in video understanding as well. Experimental results across various vision encoders, image resolutions, training dataset scales, varying sizes of LLMs (2.7B→70B), and diverse architectures of MLLMs (e.g., LLaVA-v1.5, LLaVA-NeXT and Mini-Gemini) validate the versatility and scalability of our approach, achieving state-of-the-art performance across 19 image and video benchmarks. We hope that this work will provide valuable experience and serve as a basic module for future MLLM development. Code is available at https://github.com/HJYao00/DenseConnector.

</details>

---

## 161. Improving Alignment and Robustness with Circuit Breakers

- [ ] Improving Alignment and Robustness with Circuit Breakers | https://neurips.cc/virtual/2024/poster/95761

- **Link**: https://neurips.cc/virtual/2024/poster/95761

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

</details>

---

## 162. Multimodal Large Language Models Make Text-to-Image Generative Models Align Better

- [ ] Multimodal Large Language Models Make Text-to-Image Generative Models Align Better | https://neurips.cc/virtual/2024/poster/95768

- **Link**: https://neurips.cc/virtual/2024/poster/95768

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have demonstrated the exceptional potentials of leveraging human preference datasets to refine text-to-image generative models, enhancing the alignment between generated images and textual prompts. Despite these advances, current human preference datasets are either prohibitively expensive to construct or suffer from a lack of diversity in preference dimensions, resulting in limited applicability for instruction tuning in open-source text-to-image generative models and hinder further exploration. To address these challenges and promote the alignment of generative models through instruction tuning, we leverage multimodal large language models to create VisionPrefer, a high-quality and fine-grained preference dataset that captures multiple preference aspects. We aggregate feedback from AI annotators across four aspects: prompt-following, aesthetic, fidelity, and harmlessness to construct VisionPrefer. To validate the effectiveness of VisionPrefer, we train a reward model VP-Score over VisionPrefer to guide the training of text-to-image generative models and the preference prediction accuracy of VP-Score is comparable to human annotators. Furthermore, we use two reinforcement learning methods to supervised fine-tune generative models to evaluate the performance of VisionPrefer, and extensive experimental results demonstrate that VisionPrefer significantly improves text-image alignment in compositional image generation across diverse aspects, e.g., aesthetic, and generalizes better than previous human-preference metrics across various image distributions. Moreover, VisionPrefer indicates that the integration of AI-generated synthetic data as a supervisory signal is a promising avenue for achieving improved alignment with human preferences in vision generative models.

</details>

---

## 163. OpenDlign: Open-World Point Cloud Understanding with Depth-Aligned Images

- [ ] OpenDlign: Open-World Point Cloud Understanding with Depth-Aligned Images | https://neurips.cc/virtual/2024/poster/95778

- **Link**: https://neurips.cc/virtual/2024/poster/95778

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent open-world 3D representation learning methods using Vision-Language Models (VLMs) to align 3D point clouds with image-text information have shown superior 3D zero-shot performance. However, CAD-rendered images for this alignment often lack realism and texture variation, compromising alignment robustness. Moreover, the volume discrepancy between 3D and 2D pretraining datasets highlights the need for effective strategies to transfer the representational abilities of VLMs to 3D learning. In this paper, we present OpenDlign, a novel open-world 3D model using depth-aligned images generated from a diffusion model for robust multimodal alignment. These images exhibit greater texture diversity than CAD renderings due to the stochastic nature of the diffusion model. By refining the depth map projection pipeline and designing depth-specific prompts, OpenDlign leverages rich knowledge in pre-trained VLM for 3D representation learning with streamlined fine-tuning. Our experiments show that OpenDlign achieves high zero-shot and few-shot performance on diverse 3D tasks, despite only fine-tuning 6 million parameters on a limited ShapeNet dataset. In zero-shot classification, OpenDlign surpasses previous models by 8.0\% on ModelNet40 and 16.4\% on OmniObject3D. Additionally, using depth-aligned images for multimodal alignment consistently enhances the performance of other state-of-the-art models.

</details>

---

## 164. ChatTracker: Enhancing Visual Tracking Performance via Chatting with Multimodal Large Language Model

- [ ] ChatTracker: Enhancing Visual Tracking Performance via Chatting with Multimodal Large Language Model | https://neurips.cc/virtual/2024/poster/95794

- **Link**: https://neurips.cc/virtual/2024/poster/95794

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual object tracking aims to locate a targeted object in a video sequence based on an initial bounding box. Recently, Vision-Language~(VL) trackers have proposed to utilize additional natural language descriptions to enhance versatility in various applications. However, VL trackers are still inferior to State-of-The-Art (SoTA) visual trackers in terms of tracking performance. We found that this inferiority primarily results from their heavy reliance on manual textual annotations, which include the frequent provision of ambiguous language descriptions. In this paper, we propose ChatTracker to leverage the wealth of world knowledge in the Multimodal Large Language Model (MLLM) to generate high-quality language descriptions and enhance tracking performance. To this end, we propose a novel reflection-based prompt optimization module to iteratively refine the ambiguous and inaccurate descriptions of the target with tracking feedback. To further utilize semantic information produced by MLLM, a simple yet effective VL tracking framework is proposed and can be easily integrated as a plug-and-play module to boost the performance of both VL and visual trackers. Experimental results show that our proposed ChatTracker achieves a performance comparable to existing methods.

</details>

---

## 165. Classification Done Right for Vision-Language Pre-Training

- [ ] Classification Done Right for Vision-Language Pre-Training | https://neurips.cc/virtual/2024/poster/95818

- **Link**: https://neurips.cc/virtual/2024/poster/95818

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce SuperClass, a super simple classification method for vision-language pre-training on image-text data. Unlike its contrastive counterpart CLIP who contrast with a text encoder, SuperClass directly utilizes tokenized raw text as supervised classification labels, without the need for additional text filtering or selection. Due to the absence of the text encoding as contrastive target, SuperClass does not require a text encoder and does not need to maintain a large batch size as CLIP does. SuperClass demonstrated superior performance on various downstream tasks, including classic computer vision benchmarks and vision language downstream tasks. We further explored the scaling behavior of SuperClass on model size, training length, or data size, and reported encouraging results and comparisons to CLIP. https://github.com/x-cls/superclass

</details>

---

## 166. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users

- [ ] ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users | https://neurips.cc/virtual/2024/poster/95859

- **Link**: https://neurips.cc/virtual/2024/poster/95859

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.

</details>

---

## 167. Towards Calibrated Robust Fine-Tuning of Vision-Language Models

- [ ] Towards Calibrated Robust Fine-Tuning of Vision-Language Models | https://neurips.cc/virtual/2024/poster/95878

- **Link**: https://neurips.cc/virtual/2024/poster/95878

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Improving out-of-distribution (OOD) generalization during in-distribution (ID) adaptation is a primary goal of robust fine-tuning of zero-shot models beyond naive fine-tuning. However, despite decent OOD generalization performance from recent robust fine-tuning methods, confidence calibration for reliable model output has not been fully addressed. This work proposes a robust fine-tuning method that improves both OOD accuracy and confidence calibration simultaneously in vision language models. Firstly, we show that both OOD classification and OOD calibration errors have a shared upper bound consisting of two terms of ID data: 1) ID calibration error and 2) the smallest singular value of the ID input covariance matrix. Based on this insight, we design a novel framework that conducts fine-tuning with a constrained multimodal contrastive loss enforcing a larger smallest singular value, which is further guided by the self-distillation of a moving-averaged model to achieve calibrated prediction as well. Starting from empirical evidence supporting our theoretical statements, we provide extensive experimental results on ImageNet distribution shift benchmarks that demonstrate the effectiveness of our theorem and its practical implementation.

</details>

---

## 168. Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models

- [ ] Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models | https://neurips.cc/virtual/2024/poster/95908

- **Link**: https://neurips.cc/virtual/2024/poster/95908

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Humans draw to facilitate reasoning: we draw auxiliary lines when solving geometry problems; we mark and circle when reasoning on maps; we use sketches to amplify our ideas and relieve our limited-capacity working memory. However, such actions are missing in current multimodal language models (LMs). Current chain-of-thought and tool-use paradigms only use text as intermediate reasoning steps. In this work, we introduce Sketchpad, a framework that gives multimodal LMs a visual sketchpad and tools to draw on the sketchpad. The LM conducts planning and reasoning according to the visual artifacts it has drawn. Different from prior work, which uses text-to-image models to enable LMs to draw, Sketchpad enables LMs to draw with lines, boxes, marks, etc., which is closer to human sketching and better facilitates reasoning. \name can also use specialist vision models during the sketching process (e.g., draw bounding boxes with object detection models, draw masks with segmentation models), to further enhance visual perception and reasoning. We experiment on a wide range of math tasks (including geometry, functions, graph, chess) and complex visual reasoning tasks. Sketchpad substantially improves performance on all tasks over strong base models with no sketching, yielding an average gain of 12.7% on math tasks, and 8.6% on vision tasks. GPT-4o with Sketchpad sets a new state of the art on all tasks, including V*Bench (80.3%), BLINK spatial reasoning (83.9%), and visual correspondence (80.8%). We will release all code and data.

</details>

---

## 169. Artemis:  Towards Referential Understanding in Complex Videos

- [ ] Artemis:  Towards Referential Understanding in Complex Videos | https://neurips.cc/virtual/2024/poster/95960

- **Link**: https://neurips.cc/virtual/2024/poster/95960

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Videos carry rich visual information including object description, action, interaction, etc., but the existing multimodal large language models (MLLMs) fell short in referential understanding scenarios such as video-based referring. In this paper, we present Artemis, an MLLM that pushes video-based referential understanding to a finer level. Given a video, Artemis receives a natural-language question with a bounding box in any video frame and describes the referred target in the entire video. The key to achieving this goal lies in extracting compact, target-specific video features, where we set a solid baseline by tracking and selecting spatiotemporal features from the video. We train Artemis on the newly established ViderRef45K dataset with 45K video-QA pairs and design a computationally efficient, three-stage training procedure. Results are promising both quantitatively and qualitatively. Additionally, we show that Artemis can be integrated with video grounding and text summarization tools to understand more complex scenarios. Code and data are available at https://github.com/NeurIPS24Artemis/Artemis.

</details>

---

## 170. Enhancing Large Vision Language Models with Self-Training on Image Comprehension

- [ ] Enhancing Large Vision Language Models with Self-Training on Image Comprehension | https://neurips.cc/virtual/2024/poster/95961

- **Link**: https://neurips.cc/virtual/2024/poster/95961

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision language models (LVLMs) integrate large language models (LLMs) with pre-trained vision encoders, thereby activating the perception capability of the model to understand image inputs for different queries and conduct subsequent reasoning. Improving this capability requires high-quality vision-language data, which is costly and labor-intensive to acquire. Self-training approaches have been effective in single-modal settings to alleviate the need for labeled data by leveraging model's own generation. However, effective self-training remains a challenge regarding the unique visual perception and reasoning capability of LVLMs. To address this, we introduce S elf- T raining on I mage C omprehension ( STIC ), which emphasizes a self-training approach specifically for image comprehension. First, the model self-constructs a preference dataset for image descriptions using unlabeled images. Preferred responses are generated through a step-by-step prompt, while dis-preferred responses are generated from either corrupted images or misleading prompts. To further self-improve reasoning on the extracted visual information, we let the model reuse a small portion of existing instruction-tuning data and append its self-generated image descriptions to the prompts. We validate the effectiveness of STIC across seven different benchmarks, demonstrating substantial performance gains of 4.0% on average while using 70% less supervised fine-tuning data than the current method. Further studies dive into various components of STIC and highlight its potential to leverage vast quantities of unlabeled images for self-training.

</details>

---

## 171. LLaNA: Large Language and NeRF Assistant

- [ ] LLaNA: Large Language and NeRF Assistant | https://neurips.cc/virtual/2024/poster/96007

- **Link**: https://neurips.cc/virtual/2024/poster/96007

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated an excellent understanding of images and 3D data. However, both modalities have shortcomings in holistically capturing the appearance and geometry of objects. Meanwhile, Neural Radiance Fields (NeRFs), which encode information within the weights of a simple Multi-Layer Perceptron (MLP), have emerged as an increasingly widespread modality that simultaneously encodes the geometry and photorealistic appearance of objects. This paper investigates the feasibility and effectiveness of ingesting NeRF into MLLM. We create LLaNA, the first general-purpose NeRF-languageassistant capable of performing new tasks such as NeRF captioning and Q&A. Notably, our method directly processes the weights of the NeRF’s MLP to extract information about the represented objects without the need to render images or materialize 3D data structures. Moreover, we build a dataset of NeRFs with text annotations for various NeRF-language tasks with no human intervention.Based on this dataset, we develop a benchmark to evaluate the NeRF understanding capability of our method. Results show that processing NeRF weights performs favourably against extracting 2D or 3D representations from NeRFs.

</details>

---

## 172. Mitigating Object Hallucination via Concentric Causal Attention

- [ ] Mitigating Object Hallucination via Concentric Causal Attention | https://neurips.cc/virtual/2024/poster/96152

- **Link**: https://neurips.cc/virtual/2024/poster/96152

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent Large Vision Language Models (LVLMs) present remarkable zero-shot conversational and reasoning capabilities given multimodal queries. Nevertheless, they suffer from object hallucination, a phenomenon where LVLMs are prone to generate textual responses not factually aligned with image inputs. Our pilot study reveals that object hallucination is closely tied with Rotary Position Encoding (RoPE), a widely adopted positional dependency modeling design in existing LVLMs. Due to the long-term decay in RoPE, LVLMs tend to hallucinate more when relevant visual cues are distant from instruction tokens in the multimodal input sequence, Additionally, we observe a similar effect when reversing the sequential order of visual tokens during multimodal alignment. Our tests indicate that long-term decay in RoPE poses challenges to LVLMs while capturing visual-instruction interactions across long distances. We propose Concentric Causal Attention (CCA), a simple yet effective positional alignment strategy that mitigates the impact of RoPE long-term decay in LVLMs by naturally reducing relative distance between visual and instruction tokens. With CCA, visual tokens can better interact with instruction tokens, thereby enhancing model's perception capability and alleviating object hallucination. Without bells and whistles, our positional alignment method surpasses existing hallucination mitigation strategies by large margins on multiple object hallucination benchmarks.

</details>

---

## 173. Mind's Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models

- [ ] Mind's Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models | https://neurips.cc/virtual/2024/poster/96156

- **Link**: https://neurips.cc/virtual/2024/poster/96156

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have exhibited impressive performance in language comprehension and various reasoning tasks. However, their abilities in spatial reasoning, a crucial aspect of human cognition, remain relatively unexplored. Human possess a remarkable ability to create mental images of unseen objects and actions through a process known as the Mind's Eye, enabling the imagination of the unseen world. Inspired by this cognitive capacity, we propose Visualization-of-Thought (VoT) prompting. VoT aims to elicit spatial reasoning of LLMs by visualizing their reasoning traces, thereby guiding subsequent reasoning steps. We employed VoT for multi-hop spatial reasoning tasks, including natural language navigation, visual navigation, and visual tiling in 2D grid worlds. Experimental results demonstrated that VoT significantly enhances the spatial reasoning abilities of LLMs. Notably, VoT outperformed existing multimodal large language models (MLLMs) in these tasks. While VoT works surprisingly well on LLMs, the ability to generate mental images to facilitate spatial reasoning resembles the mind's eye process, suggesting its potential viability in MLLMs. Please find the dataset and codes in our project page .

</details>

---

## 174. Stabilizing Zero-Shot Prediction: A Novel Antidote to Forgetting in Continual Vision-Language Tasks

- [ ] Stabilizing Zero-Shot Prediction: A Novel Antidote to Forgetting in Continual Vision-Language Tasks | https://neurips.cc/virtual/2024/poster/96161

- **Link**: https://neurips.cc/virtual/2024/poster/96161

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Continual learning (CL) empowers pre-trained vision-language (VL) models to efficiently adapt to a sequence of downstream tasks. However, these models often encounter challenges in retaining previously acquired skills due to parameter shifts and limited access to historical data. In response, recent efforts focus on devising specific frameworks and various replay strategies, striving for a typical learning-forgetting trade-off. Surprisingly, both our empirical research and theoretical analysis demonstrate that the stability of the model in consecutive zero-shot predictions serves as a reliable indicator of its anti-forgetting capabilities for previously learned tasks. Motivated by these insights, we develop a novel replay-free CL method named ZAF (Zero-shot Antidote to Forgetting), which preserves acquired knowledge through a zero-shot stability regularization applied to wild data in a plug-and-play manner. To enhance efficiency in adapting to new tasks and seamlessly access historical models, we introduce a parameter-efficient EMA-LoRA neural architecture based on the Exponential Moving Average (EMA). ZAF utilizes new data for low-rank adaptation (LoRA), complemented by a zero-shot antidote on wild data, effectively decoupling learning from forgetting. Our extensive experiments demonstrate ZAF's superior performance and robustness in pre-trained models across various continual VL concept learning tasks, achieving leads of up to 3.70\%, 4.82\%, and 4.38\%, along with at least a 10x acceleration in training speed on three benchmarks, respectively. Additionally, our zero-shot antidote significantly reduces forgetting in existing models by at least 6.37\%. Our code is available at https://github.com/Zi-Jian-Gao/Stabilizing-Zero-Shot-Prediction-ZAF.

</details>

---

## 175. VLMimic: Vision Language Models are Visual Imitation Learner for Fine-grained Actions

- [ ] VLMimic: Vision Language Models are Visual Imitation Learner for Fine-grained Actions | https://neurips.cc/virtual/2024/poster/96165

- **Link**: https://neurips.cc/virtual/2024/poster/96165

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual imitation learning (VIL) provides an efficient and intuitive strategy for robotic systems to acquire novel skills. Recent advancements in Vision Language Models (VLMs) have demonstrated remarkable performance in vision and language reasoning capabilities for VIL tasks. Despite the progress, current VIL methods naively employ VLMs to learn high-level plans from human videos, relying on pre-defined motion primitives for executing physical interactions, which remains a major bottleneck. In this work, we present VLMimic, a novel paradigm that harnesses VLMs to directly learn even fine-grained action levels, only given a limited number of human videos. Specifically, VLMimic first grounds object-centric movements from human videos, and learns skills using hierarchical constraint representations, facilitating the derivation of skills with fine-grained action levels from limited human videos. These skills are refined and updated through an iterative comparison strategy, enabling efficient adaptation to unseen environments. Our extensive experiments exhibit that our VLMimic, using only 5 human videos, yields significant improvements of over 27% and 21% in RLBench and real-world manipulation tasks, and surpasses baselines by more than 37% in long-horizon tasks. Code and videos are available on our anonymous homepage.

</details>

---

## 176. Matryoshka Query Transformer for Large Vision-Language Models

- [ ] Matryoshka Query Transformer for Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/96220

- **Link**: https://neurips.cc/virtual/2024/poster/96220

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) typically encode an image into a fixed number of visual tokens (e.g., 576) and process these tokens with a language model. Despite their strong performance, LVLMs face challenges in adapting to varying computational constraints. This raises the question: can we achieve flexibility in the number of visual tokens to suit different tasks and computational resources? We answer this with an emphatic yes. Inspired by Matryoshka Representation Learning, we introduce the Matryoshka Query Transformer (MQT), capable of encoding an image into $m$ visual tokens during inference, where $m$ can be any number up to a predefined maximum. This is achieved by employing a query transformer with $M$ latent query tokens to compress the visual embeddings. During each training step, we randomly select $m \leq M$ latent query tokens and train the model using only these first $m$ tokens, discarding the rest.Combining MQT with LLaVA, we train a single model once, and flexibly and drastically reduce the number of inference-time visual tokens while maintaining similar or better performance compared to training independent models for each number of tokens. Our model, MQT-LLaVA, matches LLaVA-1.5 performance across 11 benchmarks using a maximum of 256 tokens instead of LLaVA’s fixed 576. Reducing to 16 tokens (8x less TFLOPs) only sacrifices the performance by 2.4 points on MMBench. On certain tasks such as ScienceQA and MMMU, we can even go down to only 2 visual tokens with performance drops of just 3\% and 6\% each.Our exploration of the trade-off between the accuracy and computational cost brought about by the number of visual tokens facilitates future research to achieve the best of both worlds.

</details>

---

## 177. EAGLE: Efficient Adaptive Geometry-based Learning in Cross-view Understanding

- [ ] EAGLE: Efficient Adaptive Geometry-based Learning in Cross-view Understanding | https://neurips.cc/virtual/2024/poster/96249

- **Link**: https://neurips.cc/virtual/2024/poster/96249

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Unsupervised Domain Adaptation has been an efficient approach to transferring the semantic segmentation model across data distributions. Meanwhile, the recent Open-vocabulary Semantic Scene understanding based on large-scale vision language models is effective in open-set settings because it can learn diverse concepts and categories. However, these prior methods fail to generalize across different camera views due to the lack of cross-view geometric modeling. At present, there are limited studies analyzing cross-view learning. To address this problem, we introduce a novel Unsupervised Cross-view Adaptation Learning approach to modeling the geometric structural change across views in Semantic Scene Understanding. First, we introduce a novel Cross-view Geometric Constraint on Unpaired Data to model structural changes in images and segmentation masks across cameras. Second, we present a new Geodesic Flow-based Correlation Metric to efficiently measure the geometric structural changes across camera views. Third, we introduce a novel view-condition prompting mechanism to enhance the view-information modeling of the open-vocabulary segmentation network in cross-view adaptation learning. The experiments on different cross-view adaptation benchmarks have shown the effectiveness of our approach in cross-view modeling, demonstrating that we achieve State-of-the-Art (SOTA) performance compared to prior unsupervised domain adaptation and open-vocabulary semantic segmentation methods.

</details>

---

## 178. Testing Semantic Importance via Betting

- [ ] Testing Semantic Importance via Betting | https://neurips.cc/virtual/2024/poster/96285

- **Link**: https://neurips.cc/virtual/2024/poster/96285

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent works have extended notions of feature importance to semantic concepts that are inherently interpretable to the users interacting with a black-box predictive model. Yet, precise statistical guarantees such as false positive rate and false discovery rate control are needed to communicate findings transparently, and to avoid unintended consequences in real-world scenarios. In this paper, we formalize the global (i.e., over a population) and local (i.e., for a sample) statistical importance of semantic concepts for the predictions of opaque models by means of conditional independence, which allows for rigorous testing. We use recent ideas of sequential kernelized independence testing to induce a rank of importance across concepts, and we showcase the effectiveness and flexibility of our framework on synthetic datasets as well as on image classification using several vision-language models.

</details>

---

## 179. Ask, Attend, Attack: An Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models

- [ ] Ask, Attend, Attack: An Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models | https://neurips.cc/virtual/2024/poster/96292

- **Link**: https://neurips.cc/virtual/2024/poster/96292

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.

</details>

---

## 180. Leveraging Hallucinations to Reduce Manual Prompt Dependency in Promptable Segmentation

- [ ] Leveraging Hallucinations to Reduce Manual Prompt Dependency in Promptable Segmentation | https://neurips.cc/virtual/2024/poster/96318

- **Link**: https://neurips.cc/virtual/2024/poster/96318

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Promptable segmentation typically requires instance-specific manual prompts to guide the segmentation of each desired object. To minimize such a need, task-generic promptable segmentation has been introduced, which employs a single task-generic prompt to segment various images of different objects in the same task. Current methods use Multimodal Large Language Models (MLLMs) to reason detailed instance-specific prompts from a task-generic prompt for improving segmentation accuracy. The effectiveness of this segmentation heavily depends on the precision of these derived prompts. However, MLLMs often suffer hallucinations during reasoning, resulting in inaccurate prompting.  While existing methods focus on eliminating hallucinations to improve a model, we argue that MLLM hallucinations can reveal valuable contextual insights when leveraged correctly, as they represent pre-trained large-scale knowledge beyond individual images. In this paper, we first utilize hallucinations to mine task-related information from images and verify its accuracy to enhance precision of the generated prompts.  Specifically, we introduce an iterative \textbf{Pro}mpt-\textbf{Ma}sk \textbf{C}ycle generation framework (ProMaC) with a prompt generator and a mask generator.  The prompt generator uses a multi-scale chain of thought prompting, initially leveraging hallucinations to extract extended contextual prompts on a test image. These hallucinations are then minimized to formulate precise instance-specific prompts, directing the mask generator to produce masks that are consistent with task semantics by mask semantic alignment. Iteratively the generated masks induce the prompt generator to focus more on task-relevant image areas and reduce irrelevant hallucinations, resulting jointly in better prompts and masks. Experiments on 5 benchmarks demonstrate the effectiveness of ProMaC. Code is in https://lwpyh.github.io/ProMaC/.

</details>

---

## 181. BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping

- [ ] BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping | https://neurips.cc/virtual/2024/poster/96342

- **Link**: https://neurips.cc/virtual/2024/poster/96342

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Adaptation of pretrained vision-language models such as CLIP to various downstream tasks have raised great interest in recent researches. Previous works have proposed a variety of test-time adaptation (TTA) methods to achieve strong generalization without any knowledge of the target domain. However, existing training-required TTA approaches like TPT necessitate entropy minimization that involves large computational overhead, while training-free methods like TDA overlook the potential for information mining from the test samples themselves.In this paper, we break down the design of existing popular training-required and training-free TTA methods and bridge the gap between them within our framework.Specifically, we maintain a light-weight key-value memory for feature retrieval from  instance-agnostic historical samples and instance-aware boosting samples. The historical samples are filtered from the testing data stream and serve to extract useful information from the target distribution, while the boosting samples are drawn from regional bootstrapping and capture the knowledge of the test sample itself.We theoretically justify the rationality behind our method and empirically verify its effectiveness on both the out-of-distribution and the cross-domain datasets, showcasing its applicability in  real-world situations.

</details>

---

## 182. Make-it-Real: Unleashing Large Multimodal Model for Painting 3D Objects with Realistic Materials

- [ ] Make-it-Real: Unleashing Large Multimodal Model for Painting 3D Objects with Realistic Materials | https://neurips.cc/virtual/2024/poster/96391

- **Link**: https://neurips.cc/virtual/2024/poster/96391

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Physically realistic materials are pivotal in augmenting the realism of 3D assets across various applications and lighting conditions. However, existing 3D assets and generative models often lack authentic material properties. Manual assignment of materials using graphic software is a tedious and time-consuming task. In this paper, we exploit advancements in Multimodal Large Language Models (MLLMs), particularly GPT-4V, to present a novel approach, Make-it-Real: 1) We demonstrate that GPT-4V can effectively recognize and describe materials, allowing the construction of a detailed material library. 2) Utilizing a combination of visual cues and hierarchical text prompts, GPT-4V precisely identifies and aligns materials with the corresponding components of 3D objects. 3) The correctly matched materials are then meticulously applied as reference for the new SVBRDF material generation according to the original albedo map, significantly enhancing their visual authenticity. Make-it-Real offers a streamlined integration into the 3D content creation workflow, showcasing its utility as an essential tool for developers of 3D assets.

</details>

---

## 183. HENASY: Learning to Assemble Scene-Entities for Interpretable Egocentric Video-Language Model

- [ ] HENASY: Learning to Assemble Scene-Entities for Interpretable Egocentric Video-Language Model | https://neurips.cc/virtual/2024/poster/96412

- **Link**: https://neurips.cc/virtual/2024/poster/96412

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current video-language models (VLMs) rely extensively on instance-level alignment between video and language modalities, which presents two major limitations: (1) visual reasoning disobeys the natural perception that humans do in first-person perspective, leading to a lack of reasoning interpretation; and (2) learning is limited in capturing inherent fine-grained relationships between two modalities.In this paper, we take an inspiration from human perception and explore a compositional approach for egocentric video representation. We introduce HENASY (Hierarchical ENtities ASsemblY), which includes a spatiotemporal token grouping mechanism to explicitly assemble dynamically evolving scene entities through time and model their relationship for video representation. By leveraging compositional structure understanding, HENASY possesses strong interpretability via visual grounding with free-form text queries. We further explore a suite of multi-grained contrastive losses to facilitate entity-centric understandings. This comprises three alignment types: video-narration, noun-entity, verb-entities alignments.Our method demonstrates strong interpretability in both quantitative and qualitative experiments; while maintaining competitive performances on five downstream tasks via zero-shot transfer or as video/text representation, including video/text retrieval, action recognition, multi-choice query, natural language query, and moments query.Project page: https://uark-aicv.github.io/HENASY

</details>

---

## 184. CALVIN: Improved Contextual Video Captioning via Instruction Tuning

- [ ] CALVIN: Improved Contextual Video Captioning via Instruction Tuning | https://neurips.cc/virtual/2024/poster/96463

- **Link**: https://neurips.cc/virtual/2024/poster/96463

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent emergence of powerful Vision-Language models (VLMs) has significantly improved image captioning. Some of these models are extended to caption videos as well. However, their capabilities to understand complex scenes are limited, and the descriptions they provide for scenes tend to be overly verbose and focused on the superficial appearance of objects. Scene descriptions, especially in movies, require a deeper contextual understanding, unlike general-purpose video captioning. To address this challenge, we propose a model, CALVIN, a specialized video LLM that leverages previous movie context to generate fully "contextual" scene descriptions. To achieve this, we train our model on a suite of tasks that integrate both image-based question-answering and video captioning within a unified framework, before applying instruction tuning to refine the model's ability to provide scene captions. Lastly, we observe that our model responds well to prompt engineering and few-shot in-context learning techniques, enabling the user to adapt it to any new movie with very little additional annotation.

</details>

---

## 185. Slot-VLM: Object-Event Slots for Video-Language Modeling

- [ ] Slot-VLM: Object-Event Slots for Video-Language Modeling | https://neurips.cc/virtual/2024/poster/96465

- **Link**: https://neurips.cc/virtual/2024/poster/96465

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video-Language Models (VLMs), powered by the advancements in Large Language Models (LLMs), are charting new frontiers in video understanding. A pivotal challenge is the development of an effective method to encapsulate video content into a set of representative tokens to align with LLMs. In this work, we introduce Slot-VLM, a new framework designed to generate semantically decomposed video tokens, in terms of object-wise and event-wise visual representations, to facilitate LLM inference. Particularly, we design an Object-Event Slots module, i.e., OE-Slots, that adaptively aggregates the dense video tokens from the vision encoder to a set of representative slots. In order to take into account both the spatial object details and the varied temporal dynamics, we build OE-Slots with two branches: the Object-Slots branch and the Event-Slots branch. The Object-Slots branch focuses on extracting object-centric slots from features of high spatial resolution but low frame sample rate, emphasizing detailed object information. The Event-Slots branch is engineered to learn event-centric slots from high temporal sample rate but low spatial resolution features. These complementary slots are combined to form the vision context, serving as the input to the LLM for effective video reasoning. Our experimental results demonstrate the effectiveness of our Slot-VLM, which achieves the state-of-the-art performance on video question-answering.

</details>

---

## 186. Right this way: Can VLMs Guide Us to See More to Answer Questions?

- [ ] Right this way: Can VLMs Guide Us to See More to Answer Questions? | https://neurips.cc/virtual/2024/poster/96477

- **Link**: https://neurips.cc/virtual/2024/poster/96477

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In question-answering scenarios, humans can assess whether the available information is sufficient and seek additional information if necessary, rather than providing a forced answer. In contrast, Vision Language Models (VLMs) typically generate direct, one-shot responses without evaluating the sufficiency of the information. To investigate this gap, we identify a critical and challenging task in the Visual Question Answering (VQA) scenario: can VLMs indicate how to adjust an image when the visual information is insufficient to answer a question? This capability is especially valuable for assisting visually impaired individuals who often need guidance to capture images correctly. To evaluate this capability of current VLMs, we introduce a human-labeled dataset as a benchmark for this task. Additionally, we present an automated framework that generates synthetic training data by simulating ``where to know'' scenarios. Our empirical results show significant performance improvements in mainstream VLMs when fine-tuned with this synthetic data. This study demonstrates the potential to narrow the gap between information assessment and acquisition in VLMs, bringing their performance closer to humans.

</details>

---

## 187. DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation

- [ ] DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation | https://neurips.cc/virtual/2024/poster/96507

- **Link**: https://neurips.cc/virtual/2024/poster/96507

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image restoration (IR) in real-world scenarios presents significant challenges due to the lack of high-capacity models and comprehensive datasets.To tackle these issues, we present a dual strategy: GenIR, an innovative data curation pipeline, and DreamClear, a cutting-edge Diffusion Transformer (DiT)-based image restoration model. GenIR , our pioneering contribution, is a dual-prompt learning pipeline that overcomes the limitations of existing datasets, which typically comprise only a few thousand images and thus offer limited generalizability for larger models. GenIR streamlines the process into three stages: image-text pair construction, dual-prompt based fine-tuning, and data generation \& filtering. This approach circumvents the laborious data crawling process, ensuring copyright compliance and providing a cost-effective, privacy-safe solution for IR dataset construction. The result is a large-scale dataset of one million high-quality images.Our second contribution, DreamClear , is a DiT-based image restoration model. It utilizes the generative priors of text-to-image (T2I) diffusion models and the robust perceptual capabilities of multi-modal large language models (MLLMs) to achieve photorealistic restoration. To boost the model's adaptability to diverse real-world degradations, we introduce the Mixture of Adaptive Modulator (MoAM). It employs token-wise degradation priors to dynamically integrate various restoration experts, thereby expanding the range of degradations the model can address.Our exhaustive experiments confirm DreamClear's superior performance, underlining the efficacy of our dual strategy for real-world image restoration. Code and pre-trained models are available at: https://github.com/shallowdream204/DreamClear.

</details>

---

## 188. CogVLM: Visual Expert for Pretrained Language Models

- [ ] CogVLM: Visual Expert for Pretrained Language Models | https://neurips.cc/virtual/2024/poster/96510

- **Link**: https://neurips.cc/virtual/2024/poster/96510

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce CogVLM, a powerful open-source visual language foundation model. Different from the popular \emph{shallow alignment} method which maps image features into the input space of language model, CogVLM bridges the gap between the frozen pretrained language model and image encoder by a trainable visual expert module in the attention and FFN layers. As a result, CogVLM enables a deep fusion of vision language features without sacrificing any performance on NLP tasks. CogVLM-17B achieves state-of-the-art performance on 17 classic cross-modal benchmarks, including 1) image captioning datasets: NoCaps, Flicker30k, 2) VQA datasets: OKVQA, TextVQA, OCRVQA, ScienceQA, 3) LVLM benchmarks: MM-Vet, MMBench, SEED-Bench, LLaVABench, POPE, MMMU, MathVista, 4) visual grounding datasets: RefCOCO, RefCOCO+, RefCOCOg, Visual7W. Codes and checkpoints are available at Github.

</details>

---

## 189. SeTAR: Out-of-Distribution Detection with Selective Low-Rank Approximation

- [ ] SeTAR: Out-of-Distribution Detection with Selective Low-Rank Approximation | https://neurips.cc/virtual/2024/poster/96549

- **Link**: https://neurips.cc/virtual/2024/poster/96549

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection is crucial for the safe deployment of neural networks. Existing CLIP-based approaches perform OOD detection by devising novel scoring functions or sophisticated fine-tuning methods. In this work, we propose SeTAR, a novel, training-free OOD detection method that leverages selective low-rank approximation of weight matrices in vision-language and vision-only models. SeTAR enhances OOD detection via post-hoc modification of the model's weight matrices using a simple greedy search algorithm. Based on SeTAR, we further propose SeTAR+FT, a fine-tuning extension optimizing model performance for OOD detection tasks. Extensive evaluations on ImageNet1K and Pascal-VOC benchmarks show SeTAR's superior performance, reducing the relatively false positive rate by up to 18.95\% and 36.80\% compared to zero-shot and fine-tuning baselines. Ablation studies further validate our approach's effectiveness, robustness, and generalizability across different model backbones. Our work offers a scalable, efficient solution for OOD detection, setting a new state-of-the-art in this area.

</details>

---

## 190. Efficient Large Multi-modal Models via Visual Context Compression

- [ ] Efficient Large Multi-modal Models via Visual Context Compression | https://neurips.cc/virtual/2024/poster/96558

- **Link**: https://neurips.cc/virtual/2024/poster/96558

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While significant advancements have been made in compressed representations for text embeddings in large language models (LLMs), the compression of visual tokens in multi-modal LLMs (MLLMs) has remained a largely overlooked area. In this work, we present the study on the analysis of redundancy concerning visual tokens and efficient training within these models. Our initial experimentsshow that eliminating up to 70% of visual tokens at the testing stage by simply average pooling only leads to a minimal 3% reduction in visual question answering accuracy on the GQA benchmark, indicating significant redundancy in visual context. Addressing this, we introduce Visual Context Compressor, which reduces the number of visual tokens to enhance training and inference efficiency without sacrificing performance. To minimize information loss caused by the compression on visual tokens while maintaining training efficiency, we develop LLaVolta as a light and staged training scheme that incorporates stage-wise visual context compression to progressively compress the visual tokens from heavily to lightly compression during training, yielding no loss of information when testing. Extensive experiments demonstrate that our approach enhances the performance of MLLMs in both image-language and video-language understanding, while also significantly cutting training costs and improving inference efficiency.

</details>

---

## 191. Private Attribute Inference from Images with Vision-Language Models

- [ ] Private Attribute Inference from Images with Vision-Language Models | https://neurips.cc/virtual/2024/poster/96590

- **Link**: https://neurips.cc/virtual/2024/poster/96590

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As large language models (LLMs) become ubiquitous in our daily tasks and digital interactions, associated privacy risks are increasingly in focus. While LLM privacy research has primarily focused on the leakage of model training data, it has recently been shown that LLMs can make accurate privacy-infringing inferences from previously unseen texts. With the rise of vision-language models (VLMs), capable of understanding both images and text, a key question is whether this concern transfers to the previously unexplored domain of benign images posted online. To answer this question, we compile an image dataset with human-annotated labels of the image owner's personal attributes. In order to understand the privacy risks posed by VLMs beyond traditional human attribute recognition, our dataset consists of images where the inferable private attributes do not stem from direct depictions of humans. On this dataset, we evaluate 7 state-of-the-art VLMs, finding that they can infer various personal attributes at up to 77.6% accuracy. Concerningly, we observe that accuracy scales with the general capabilities of the models, implying that future models can be misused as stronger inferential adversaries, establishing an imperative for the development of adequate defenses.

</details>

---

## 192. VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs of Thought

- [ ] VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs of Thought | https://neurips.cc/virtual/2024/poster/96600

- **Link**: https://neurips.cc/virtual/2024/poster/96600

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale generative language and vision-language models (LLMs and VLMs) excel in few-shot in-context learning for decision making and instruction following. However, they require high-quality exemplar demonstrations to be included in their context window. In this work, we ask: Can LLMs and VLMs generate their own examples from generic, sub-optimal demonstrations? We propose In-Context Abstraction Learning (ICAL), a method that builds a memory of multimodal experience from sub-optimal demonstrations and human feedback. Given a task demonstration that may contain inefficiencies or mistakes, a VLM abstracts the trajectory into a generalized program by correcting inefficient actions and annotating cognitive abstractions: causal relationships, object state changes, temporal subgoals, and task-relevant visual elements. These abstractions are iteratively improved and adapted through human feedback while the agent attempts to execute the trajectory in a similar environment. The resulting examples, when used as exemplars in the prompt, significantly improve decision-making in retrieval-augmented LLM and VLM agents. Moreover, as the agent's library of examples grows, it becomes more efficient, relying less on human feedback and requiring fewer environment interactions per demonstration. Our ICAL agent surpasses the state-of-the-art in dialogue-based instruction following in TEACh, multimodal web agents in VisualWebArena, and action anticipation in Ego4D. In TEACh, we achieve a 12.6% improvement in goal-condition success. In VisualWebArena, our task success rate improves over the SOTA from 14.3% to 22.7% using GPT4V. In Ego4D action forecasting, we improve over few-shot GPT-4V and remain competitive with supervised models. We show finetuning our retrieval-augmented in-context agent yields additional improvements. Our approach significantly reduces reliance on manual prompt engineering and consistently outperforms in-context learning from action plans that lack such abstractions.

</details>

---

## 193. Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function

- [ ] Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function | https://neurips.cc/virtual/2024/poster/96637

- **Link**: https://neurips.cc/virtual/2024/poster/96637

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Text-to-image diffusion models particularly Stable Diffusion, have revolutionized the field of computer vision. However, the synthesis quality often deteriorates when asked to generate images that faithfully represent complex prompts involving multiple attributes and objects. While previous studies suggest that blended text embeddings lead to improper attribute binding, few have explored this in depth. In this work, we critically examine the limitations of the CLIP text encoder in understanding attributes and investigate how this affects diffusion models. We discern a phenomenon of attribute bias in the text space and highlight a contextual issue in padding embeddings that entangle different concepts. We propose Magnet, a novel training-free approach to tackle the attribute binding problem. We introduce positive and negative binding vectors to enhance disentanglement, further with a neighbor strategy to increase accuracy. Extensive experiments show that Magnet significantly improves synthesis quality and binding accuracy with negligible computational cost, enabling the generation of unconventional and unnatural concepts.

</details>

---

## 194. MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning

- [ ] MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning | https://neurips.cc/virtual/2024/poster/96642

- **Link**: https://neurips.cc/virtual/2024/poster/96642

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero- and few-shot visual anomaly segmentation relies on powerful vision-language models that detect unseen anomalies using manually designed textual prompts. However, visual representations are inherently independent of language. In this paper, we explore the potential of a pure visual foundation model as an alternative to widely used vision-language models for universal visual anomaly segmentation.We present a novel paradigm that unifies anomaly segmentation into change segmentation. This paradigm enables us to leverage large-scale synthetic image pairs, featuring object-level and local region changes, derived from existing image datasets, which are independent of target anomaly datasets. We propose a one-prompt Meta-learning framework for Universal Anomaly Segmentation (MetaUAS) that is trained on this synthetic dataset and then generalizes well to segment any novel or unseen visual anomalies in the real world. To handle geometrical variations between prompt and query images, we propose a soft feature alignment module that bridges paired-image change perception and single-image semantic segmentation. This is the first work to achieve universal anomaly segmentation using a pure vision model without relying on special anomaly detection datasets and pre-trained visual-language models. Our method effectively and efficiently segments any anomalies with only one normal image prompt and enjoys training-free without guidance from language. Our MetaUAS significantly outperforms previous zero-shot, few-shot, and even full-shot anomaly segmentation methods. Code and Models: https://github.com/gaobb/MetaUAS.

</details>

---

## 195. DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning

- [ ] DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning | https://neurips.cc/virtual/2024/poster/96658

- **Link**: https://neurips.cc/virtual/2024/poster/96658

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision language models (VLMs), though powerful, typically lack training on decision-centric data, rendering them sub-optimal for decision-making tasks such as in-the-wild device control through Graphical User Interfaces (GUIs) when used off-the-shelf. While training with static demonstrations has shown some promise, we show that such methods fall short when controlling real GUIs due to their failure to deal with real world stochasticity and dynamism not captured in static observational data. This paper introduces a novel autonomous RL approach, called DigiRL, for training in-the-wild device control agents through fine-tuning a pre-trained VLM in two stages: offline and offline-to-online RL. We first build a scalable and parallelizable Android learning environment equipped with a VLM-based general-purpose evaluator and then identify the key design choices for simple and effective RL in this domain. We demonstrate the effectiveness of DigiRL using the Android-in-the-Wild (AitW) dataset, where our 1.5B VLM trained with RL achieves a 49.5\% absolute improvement -- from 17.7 to 67.2\% success rate -- over supervised fine-tuning with static human demonstration data. It is worth noting that such improvement is achieved without any additional supervision or demonstration data. These results significantly surpass not only the prior best agents, including AppAgent with GPT-4V (8.3\% success rate) and the 17B CogAgent trained with AitW data (14.4\%), but also our implementation of prior best autonomous RL approach based on filtered behavior cloning (57.8\%), thereby establishing a new state-of-the-art for digital agents for in-the-wild device control.

</details>

---

## 196. WATT: Weight Average Test Time Adaptation of CLIP

- [ ] WATT: Weight Average Test Time Adaptation of CLIP | https://neurips.cc/virtual/2024/poster/96685

- **Link**: https://neurips.cc/virtual/2024/poster/96685

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) such as CLIP have yielded unprecedented performances for zero-shot image classification, yet their generalization capability may still be seriously challenged when confronted to domain shifts. In response, we present Weight Average Test-Time Adaptation (WATT) of CLIP, a new approach facilitating full test-time adaptation (TTA) of this VLM. Our method employs a diverse set of templates for text prompts, augmenting the existing framework of CLIP. Predictions are utilized as pseudo labels for model updates, followed by weight averaging to consolidate the learned information globally. Furthermore, we introduce a text ensemble strategy, enhancing the overall test performance by aggregating diverse textual cues.Our findings underscore the effectiveness of WATT across diverse datasets, including CIFAR-10-C, CIFAR-10.1, CIFAR-100-C, VisDA-C, and several other challenging datasets, effectively covering a wide range of domain shifts. Notably, these enhancements are achieved without the need for additional model transformations or trainable modules. Moreover, compared to other TTA methods, our approach can operate effectively with just a single image. The code is available at: https://github.com/Mehrdad-Noori/WATT.

</details>

---

## 197. Boosting Text-to-Video Generative Model with MLLMs Feedback

- [ ] Boosting Text-to-Video Generative Model with MLLMs Feedback | https://neurips.cc/virtual/2024/poster/96722

- **Link**: https://neurips.cc/virtual/2024/poster/96722

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in text-to-video generative models, such as Sora, have showcased impressive capabilities. These models have attracted significant interest for their potential applications. However, they often rely on extensive datasets of variable quality, which can result in generated videos that lack aesthetic appeal and do not accurately reflect the input text prompts. A promising approach to mitigate these issues is to leverage Reinforcement Learning from Human Feedback (RLHF), which aims to align the outputs of text-to-video generative with human preferences. However, the considerable costs associated with manual annotation have led to a scarcity of comprehensive preference datasets. In response to this challenge, our study begins by investigating the efficacy of Multimodal Large Language Models (MLLMs) generated annotations in capturing video preferences, discovering a high degree of concordance with human judgments. Building upon this finding, we utilize MLLMs to perform fine-grained video preference annotations across two dimensions, resulting in the creation of VideoPrefer, which includes 135,000 preference annotations. Utilizing this dataset, we introduce VideoRM, the first general-purpose reward model tailored for video preference in the text-to-video domain. Our comprehensive experiments confirm the effectiveness of both VideoPrefer and VideoRM, representing a significant step forward in the field.

</details>

---

## 198. Lexicon3D: Probing Visual Foundation Models for Complex 3D Scene Understanding

- [ ] Lexicon3D: Probing Visual Foundation Models for Complex 3D Scene Understanding | https://neurips.cc/virtual/2024/poster/96742

- **Link**: https://neurips.cc/virtual/2024/poster/96742

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Complex 3D scene understanding has gained increasing attention, with scene encoding strategies built on top of visual foundation models playing a crucial role in this success. However, the optimal scene encoding strategies for various scenarios remain unclear, particularly compared to their image-based counterparts. To address this issue, we present the first comprehensive study that probes various visual encoding models for 3D scene understanding, identifying the strengths and limitations of each model across different scenarios. Our evaluation spans seven vision foundation encoders, including image, video, and 3D foundation models. We evaluate these models in four tasks: Vision-Language Scene Reasoning, Visual Grounding, Segmentation, and Registration, each focusing on different aspects of scene understanding. Our evaluation yields key intriguing findings: Unsupervised image foundation models demonstrate superior overall performance, video models excel in object-level tasks, diffusion models benefit geometric tasks, language-pretrained models show unexpected limitations in language-related tasks, and the mixture-of-vision-expert (MoVE) strategy leads to consistent performance improvement. These insights challenge some conventional understandings, provide novel perspectives on leveraging visual foundation models, and highlight the need for more flexible encoder selection in future vision-language and scene understanding tasks.

</details>

---

## 199. Visual Anchors Are Strong Information Aggregators For Multimodal Large Language Model

- [ ] Visual Anchors Are Strong Information Aggregators For Multimodal Large Language Model | https://neurips.cc/virtual/2024/poster/96811

- **Link**: https://neurips.cc/virtual/2024/poster/96811

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the realm of Multimodal Large Language Models (MLLMs), vision-language connector plays a crucial role to link the pre-trained vision encoders with Large Language Models (LLMs). Despite its importance, the vision-language connector has been relatively less explored. In this study, we aim to propose a strong vision-language connector that enables MLLM to simultaneously achieve high accuracy and low computation cost. We first reveal the existence of the visual anchors in Vision Transformer and propose a cost-effective search algorithm to progressively extract them. Building on these findings, we introduce the Anchor Former (AcFormer), a novel vision-language connector designed to leverage the rich prior knowledge obtained from these visual anchors during pretraining, guiding the aggregation of information. Through extensive experimentation, we demonstrate that the proposed method significantly reduces computational costs by nearly two-thirds, while simultaneously outperforming baseline methods. This highlights the effectiveness and efficiency of AcFormer.

</details>

---

## 200. EvolveDirector: Approaching Advanced Text-to-Image Generation with Large Vision-Language Models

- [ ] EvolveDirector: Approaching Advanced Text-to-Image Generation with Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/96830

- **Link**: https://neurips.cc/virtual/2024/poster/96830

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in generation models have showcased remarkable capabilities in generating fantastic content. However, most of them are trained on proprietary high-quality data, and some models withhold their parameters and only provide accessible application programming interfaces (APIs), limiting their benefits for downstream tasks. To explore the feasibility of training a text-to-image generation model comparable to advanced models using publicly available resources, we introduce EvolveDirector. This framework interacts with advanced models through their public APIs to obtain text-image data pairs to train a base model. Our experiments with extensive data indicate that the model trained on generated data of the advanced model can approximate its generation capability. However, it requires large-scale samples of 10 million or more. This incurs significant expenses in time, computational resources, and especially the costs associated with calling fee-based APIs. To address this problem, we leverage pre-trained large vision-language models (VLMs) to guide the evolution of the base model. VLM continuously evaluates the base model during training and dynamically updates and refines the training dataset by the discrimination, expansion, deletion, and mutation operations. Experimental results show that this paradigm significantly reduces the required data volume. Furthermore, when approaching multiple advanced models, EvolveDirector can select the best samples generated by them to learn powerful and balanced abilities. The final trained model Edgen is demonstrated to outperform these advanced models. The code and model weights are available at https://github.com/showlab/EvolveDirector.

</details>

---

## 201. Multilingual Diversity Improves Vision-Language Representations

- [ ] Multilingual Diversity Improves Vision-Language Representations | https://neurips.cc/virtual/2024/poster/96862

- **Link**: https://neurips.cc/virtual/2024/poster/96862

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Massive web-crawled image-text datasets lay the foundation for recent progress in multimodal learning. These datasets are designed with the goal of training a model to do well on standard computer vision benchmarks, many of which, however, have been shown to be English-centric (e.g., ImageNet). Consequently, existing data curation techniques gravitate towards using predominantly English image-text pairs and discard many potentially useful non-English samples. Our work questions this practice. Multilingual data is inherently enriching not only because it provides a gateway to learn about culturally salient concepts, but also because it depicts common concepts differently from monolingual data. We thus conduct a systematic study to explore the performance benefits of using more samples of non-English origins with respect to English vision tasks. By translating all multilingual image-text pairs from a raw web crawl to English and re-filtering them, we increase the prevalence of (translated) multilingual data in the resulting training set. Pre-training on this dataset outperforms using English-only or English-dominated datasets on ImageNet, ImageNet distribution shifts, image-English-text retrieval and on average across 38 tasks from the DataComp benchmark. On a geographically diverse task like GeoDE, we also observe improvements across all regions, with the biggest gain coming from Africa. In addition, we quantitatively show that English and non-English data are significantly different in both image and (translated) text space. We hope that our findings motivate future work to be more intentional about including multicultural and multilingual data, not just when non-English or geographically diverse tasks are involved, but to enhance model capabilities at large.

</details>

---

## 202. FINALLY: fast and universal speech enhancement with studio-like quality

- [ ] FINALLY: fast and universal speech enhancement with studio-like quality | https://neurips.cc/virtual/2024/poster/96882

- **Link**: https://neurips.cc/virtual/2024/poster/96882

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we address the challenge of speech enhancement in real-world recordings, which often contain various forms of distortion, such as background noise, reverberation, and microphone artifacts.We revisit the use of Generative Adversarial Networks (GANs) for speech enhancement and theoretically show that GANs are naturally inclined to seek the point of maximum density within the conditional clean speech distribution, which, as we argue, is essential for speech enhancement task.We study various feature extractors for perceptual loss to facilitate the stability of adversarial training, developing a methodology for probing the structure of the feature space.This leads us to integrate WavLM-based perceptual loss into MS-STFT adversarial training pipeline, creating an effective and stable training procedure for the speech enhancement model.The resulting speech enhancement model, which we refer to as FINALLY, builds upon the HiFi++ architecture, augmented with a WavLM encoder and a novel training pipeline.Empirical results on various datasets confirm our model's ability to produce clear, high-quality speech at 48 kHz, achieving state-of-the-art performance in the field of speech enhancement. Demo page: https://samsunglabs.github.io/FINALLY-page/

</details>

---

## 203. Unleashing Region Understanding in Intermediate Layers for MLLM-based Referring Expression Generation

- [ ] Unleashing Region Understanding in Intermediate Layers for MLLM-based Referring Expression Generation | https://neurips.cc/virtual/2024/poster/96885

- **Link**: https://neurips.cc/virtual/2024/poster/96885

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The Multi-modal Large Language Model (MLLM) based Referring Expression Generation (REG) task has gained increasing popularity, which aims to generate an unambiguous text description that applies to exactly one object or region in the image by leveraging foundation models. We empirically found that there exists a potential trade-off between the detailedness and the correctness of the descriptions for the referring objects. On the one hand, generating sentences with more details is usually required in order to provide more precise object descriptions. On the other hand, complicated sentences could easily increase the probability of hallucinations. To address this issue, we propose a training-free framework, named ``unleash-then-eliminate'', which first elicits the latent information in the intermediate layers, and then adopts a cycle-consistency-based decoding method to alleviate the production of hallucinations. Furthermore, to reduce the computational load of cycle-consistency-based decoding, we devise a Probing-based Importance Estimation method to statistically estimate the importance weights of intermediate layers within a subset. These importance weights are then incorporated into the decoding process over the entire dataset, intervening in the next token prediction from intermediate layers.Extensive experiments conducted on the RefCOCOg and PHD benchmarks show that our proposed framework could outperform existing methods on both semantic and hallucination-related metrics. Code will be made available in https://github.com/Glupayy/unleash-eliminate.

</details>

---

## 204. A Unified Debiasing Approach for Vision-Language Models across Modalities and Tasks

- [ ] A Unified Debiasing Approach for Vision-Language Models across Modalities and Tasks | https://neurips.cc/virtual/2024/poster/96884

- **Link**: https://neurips.cc/virtual/2024/poster/96884

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language Models (VLMs) have enabled complex multimodal tasks by processing text and image data simultaneously, significantly enhancing the field of artificial intelligence. However, these models often exhibit biases that can skew outputs towards societal stereotypes, thus necessitating debiasing strategies. Existing debiasing methods focus narrowly on specific modalities or tasks, and require extensive retraining. To address these limitations, this paper introduces Selective Feature Imputation for Debiasing (SFID), a novel methodology that integrates feature pruning and low confidence imputation (LCI) to effectively reduce biases in VLMs. SFID is versatile, maintaining the semantic integrity of outputs and costly effective by eliminating the need for retraining. Our experimental results demonstrate SFID's effectiveness across various VLMs tasks including zero-shot classification, text-to-image retrieval, image captioning, and text-to-image generation, by significantly reducing gender biases without compromising performance. This approach not only enhances the fairness of VLMs applications but also preserves their efficiency and utility across diverse scenarios.

</details>

---

## 205. Bayesian-guided Label Mapping for Visual Reprogramming

- [ ] Bayesian-guided Label Mapping for Visual Reprogramming | https://neurips.cc/virtual/2024/poster/96890

- **Link**: https://neurips.cc/virtual/2024/poster/96890

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual reprogramming (VR) leverages the intrinsic capabilities of pretrained vision models by adapting their input or output interfaces to solve downstream tasks whose labels (i.e., downstream labels) might be totally different from the labels associated with the pretrained models (i.e., pretrained labels). When adapting the output interface, label mapping methods transform the pretrained labels to downstream labels by establishing a gradient-free one-to-one correspondence between the two sets of labels.However, in this paper, we reveal that one-to-one mappings may overlook the complex relationship between pretrained and downstream labels. Motivated by this observation, we propose a B ayesian-guided L abel M apping (BLM) method. BLM constructs an iteratively-updated probabilistic label mapping matrix, with each element quantifying a pairwise relationship between pretrained and downstream labels.The assignment of values to the constructed matrix is guided by Bayesian conditional probability, considering the joint distribution of the downstream labels and the labels predicted by the pretrained model on downstream samples. Experiments conducted on both pretrained vision models (e.g., ResNeXt) and vision-language models (e.g., CLIP) demonstrate the superior performance of BLM over existing label mapping methods. The success of BLM also offers a probabilistic lens through which to understand and analyze the effectiveness of VR.Our code is available at https://github.com/tmlr-group/BayesianLM.

</details>

---

## 206. Enhancing Domain Adaptation through Prompt Gradient Alignment

- [ ] Enhancing Domain Adaptation through Prompt Gradient Alignment | https://neurips.cc/virtual/2024/poster/96889

- **Link**: https://neurips.cc/virtual/2024/poster/96889

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prior Unsupervised Domain Adaptation (UDA) methods often aim to train a domain-invariant feature extractor, which may hinder the model from learning sufficiently discriminative features. To tackle this, a line of works based on prompt learning leverages the power of large-scale pre-trained vision-language models to learn both domain-invariant and specific features through a set of domain-agnostic and domain-specific learnable prompts. Those studies typically enforce invariant constraints on representation, output, or prompt space to learn such prompts. Differently, we cast UDA as a multiple-objective optimization problem in which each objective is represented by a domain loss. Under this new framework, we propose aligning per-objective gradients to foster consensus between them. Additionally, to prevent potential overfitting when fine-tuning this deep learning architecture, we penalize the norm of these gradients. To achieve these goals, we devise a practical gradient update procedure that can work under both single-source and multi-source UDA. Empirically, our method consistently surpasses other vision language model adaptation methods by a large margin on a wide range of benchmarks. The implementation is available at https://github.com/VietHoang1512/PGA.

</details>

---

## 207. Grounding Multimodal Large Language Models in Actions

- [ ] Grounding Multimodal Large Language Models in Actions | https://neurips.cc/virtual/2024/poster/96941

- **Link**: https://neurips.cc/virtual/2024/poster/96941

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated a wide range of capabilities across many domains including Embodied AI. In this work, we study how to best ground a MLLM into different embodiments and their associated action spaces, including both continuous and discrete actions. For continuous actions, a set of learned tokenizations that capture an action at various resolutions allows for sufficient modeling precision, yielding the best performance on downstream tasks. For discrete actions, semantically aligning these actions with the native output token space of the MLLM leads to the strongest performance. We arrive at these lessons via a thorough study of seven action grounding approaches on five different environments, encompassing over 114 embodied tasks.

</details>

---

## 208. Bridge the Modality and Capability Gaps in Vision-Language Model Selection

- [ ] Bridge the Modality and Capability Gaps in Vision-Language Model Selection | https://neurips.cc/virtual/2024/poster/96960

- **Link**: https://neurips.cc/virtual/2024/poster/96960

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) excel in zero-shot image classification by pairing images with textual category names. The expanding variety of Pre-Trained VLMs enhances the likelihood of identifying a suitable VLM for specific tasks. To better reuse the VLM resource and fully leverage its potential on different zero-shot image classification tasks, a promising strategy is selecting appropriate Pre-Trained VLMs from the VLM Zoo, relying solely on the text data of the target dataset without access to the dataset’s images. In this paper, we analyze two inherent challenges in assessing the ability of a VLM in this Language-Only VLM selection: the “Modality Gap”—the disparity in VLM’s embeddings across two different modalities, making text a less reliable substitute for images; and the “Capability Gap”— the discrepancy between the VLM’s overall ranking and its ranking for target dataset, hindering direct prediction of a model’s dataset-specific performance from its general performance. We propose VLM Selection With gAp Bridging (SWAB) to mitigate the negative impact of two gaps. SWAB first adopts optimal transport to capture the relevance between open-source and target datasets with a transportation matrix. It then uses this matrix to transfer useful statistics of VLMs from open-source datasets to the target dataset for bridging two gaps. By bridging two gaps to obtain better substitutes for test images, SWAB can accurately predict the performance ranking of different VLMs on the target task without the need for the dataset’s images. Experiments across various VLMs and image classification datasets validate SWAB’s effectiveness. Code is available at: https://github.com/YCaigogogo/SWAB.

</details>

---

## 209. MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations

- [ ] MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations | https://neurips.cc/virtual/2024/poster/97429

- **Link**: https://neurips.cc/virtual/2024/poster/97429

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the emergence of LLMs and their integration with other data modalities, multi-modal 3D perception attracts more attention due to its connectivity to the physical world and makes rapid progress. However, limited by existing datasets, previous works mainly focus on understanding object properties or inter-object spatial relationships in a 3D scene. To tackle this problem, this paper builds the first largest ever multi-modal 3D scene dataset and benchmark with hierarchical grounded language annotations, MMScan. It is constructed based on a top-down logic, from region to object level, from a single target to inter-target relationships, covering holistic aspects of spatial and attribute understanding. The overall pipeline incorporates powerful VLMs via carefully designed prompts to initialize the annotations efficiently and further involve humans' correction in the loop to ensure the annotations are natural, correct, and comprehensive. Built upon existing 3D scanning data, the resulting multi-modal 3D dataset encompasses 1.4M meta-annotated captions on 109k objects and 7.7k regions as well as over 3.04M diverse samples for 3D visual grounding and question-answering benchmarks. We evaluate representative baselines on our benchmarks, analyze their capabilities in different aspects, and showcase the key problems to be addressed in the future. Furthermore, we use this high-quality dataset to train state-of-the-art 3D visual grounding and LLMs and obtain remarkable performance improvement both on existing benchmarks and in-the-wild evaluation.

</details>

---

## 210. SeafloorAI: A Large-scale Vision-Language Dataset for Seafloor Geological Survey

- [ ] SeafloorAI: A Large-scale Vision-Language Dataset for Seafloor Geological Survey | https://neurips.cc/virtual/2024/poster/97432

- **Link**: https://neurips.cc/virtual/2024/poster/97432

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A major obstacle to the advancements of machine learning models in marine science, particularly in sonar imagery analysis, is the scarcity of AI-ready datasets.  While there have been efforts to make AI-ready sonar image dataset publicly available, they suffer from limitations in terms of environment setting and scale.  To bridge this gap, we introduce $\texttt{SeafloorAI}$, the first extensive AI-ready datasets for seafloor mapping across 5 geological layers that is curated in collaboration with marine scientists. We further extend the dataset to $\texttt{SeafloorGenAI}$ by incorporating the language component in order to facilitate the development of both $\textit{vision}$- and $\textit{language}$-capable machine learning models for sonar imagery.  The dataset consists of 62 geo-distributed data surveys spanning 17,300 square kilometers, with 696K sonar images, 827K annotated segmentation masks, 696K detailed language descriptions and approximately 7M question-answer pairs.   By making our data processing source code publicly available, we aim to engage the marine science community to enrich the data pool and inspire the machine learning community to develop more robust models.   This collaborative approach will enhance the capabilities and applications of our datasets within both fields.

</details>

---

## 211. MLLM-CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs

- [ ] MLLM-CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs | https://neurips.cc/virtual/2024/poster/97438

- **Link**: https://neurips.cc/virtual/2024/poster/97438

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability to compare objects, scenes, or situations is crucial for effective decision-making and problem-solving in everyday life. For instance, comparing the freshness of apples enables better choices during grocery shopping, while comparing sofa designs helps optimize the aesthetics of our living space. Despite its significance, the comparative capability is largely unexplored in artificial general intelligence (AGI). In this paper, we introduce MLLM-CompBench, a benchmark designed to evaluate the comparative reasoning capability of multimodal large language models (MLLMs). MLLM-CompBench mines and pairs images through visually oriented questions covering eight dimensions of relative comparison: visual attribute, existence, state, emotion, temporality, spatiality, quantity, and quality. We curate a collection of around 40K image pairs using metadata from diverse vision datasets and CLIP similarity scores. These image pairs span a broad array of visual domains, including animals, fashion, sports, and both outdoor and indoor scenes. The questions are carefully crafted to discern relative characteristics between two images and are labeled by human annotators for accuracy and relevance. We use MLLM-CompBench to evaluate recent MLLMs, including GPT-4V(ision), Gemini-Pro, and LLaVA-1.6. Our results reveal notable shortcomings in their comparative abilities. We believe MLLM-CompBench not only sheds light on these limitations but also establishes a solid foundation for future enhancements in the comparative capability of MLLMs.

</details>

---

## 212. Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning

- [ ] Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning | https://neurips.cc/virtual/2024/poster/97450

- **Link**: https://neurips.cc/virtual/2024/poster/97450

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present a novel multimodal preference dataset for creative tasks, consisting of over 250 million human votes on more than 2.2 million captions, collected through crowdsourcing rating data for The New Yorker's weekly cartoon caption contest over the past eight years. This unique dataset supports the development and evaluation of multimodal large language models and preference-based fine-tuning algorithms for humorous caption generation. We propose novel benchmarks for judging the quality of model-generated captions, utilizing both GPT4 and human judgments to establish ranking-based evaluation strategies. Our experimental results highlight the limitations of current fine-tuning methods, such as RLHF and DPO, when applied to creative tasks. Furthermore, we demonstrate that even state-of-the-art models like GPT4 and Claude currently underperform top human contestants in generating humorous captions. As we conclude this extensive data collection effort, we release the entire preference dataset to the research community, fostering further advancements in AI humor generation and evaluation.

</details>

---

## 213. MARVEL: Multidimensional Abstraction and Reasoning through Visual Evaluation and Learning

- [ ] MARVEL: Multidimensional Abstraction and Reasoning through Visual Evaluation and Learning | https://neurips.cc/virtual/2024/poster/97456

- **Link**: https://neurips.cc/virtual/2024/poster/97456

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While multi-modal large language models (MLLMs) have shown significant progress across popular visual reasoning benchmarks, whether they possess abstract visual reasoning abilities remains an open question. Similar to the Sudoku puzzles, abstract visual reasoning (AVR) problems require finding high-level patterns (e.g., repetition constraints on numbers) that control the input shapes (e.g., digits) in a specific task configuration (e.g., matrix). However, existing AVR benchmarks only consider a limited set of patterns (addition, conjunction), input shapes (rectangle, square), and task configurations (3 × 3 matrices). And they fail to capture all abstract reasoning patterns in human cognition necessary for addressing real-world tasks, such as geometric properties and object boundary understanding in real-world navigation. To evaluate MLLMs’ AVR abilities systematically, we introduce MARVEL founded on the core knowledge system in human cognition, a multi-dimensional AVR benchmark with 770 puzzles composed of six core knowledge patterns, geometric and abstract shapes, and five different task configurations. To inspect whether the model performance is grounded in perception or reasoning, MARVEL complements the standard AVR question with perception questions in a hierarchical evaluation framework. We conduct comprehensive experiments on MARVEL with ten representative MLLMs in zero-shot and few-shot settings. Our experiments reveal that all MLLMs show near-random performance on MARVEL, with significant performance gaps (40%) compared to humans across all patterns and task configurations. Further analysis of perception questions reveals that MLLMs struggle to comprehend the visual features (near-random performance). Although closed-source MLLMs, such as GPT-4V, show a promising understanding of reasoning patterns (on par with humans) after adding textual descriptions, this advantage is hindered by their weak perception abilities. We release our entirecode and dataset at https://github.com/1171-jpg/MARVEL_AVR.

</details>

---

## 214. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments

- [ ] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments | https://neurips.cc/virtual/2024/poster/97468

- **Link**: https://neurips.cc/virtual/2024/poster/97468

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Autonomous agents that accomplish complex computer tasks with minimal human interventions have the potential to transform human-computer interaction, significantly enhancing accessibility and productivity. However, existing benchmarks either lack an interactive environment or are limited to environments specific to certain applications or domains, failing to reflect the diverse and complex nature of real-world computer use, thereby limiting the scope of tasks and agent scalability. To address this issue, we introduce OSWorld, the first-of-its-kind scalable, real computer environment for multimodal agents, supporting task setup, execution-based evaluation, and interactive learning across various operating systems such as Ubuntu, Windows, and macOS. OSWorld can serve as a unified, integrated computer environment for assessing open-ended computer tasks that involve arbitrary applications. Building upon OSWorld, we create a benchmark of 369 computer tasks involving real web and desktop apps in open domains, OS file I/O, and workflows spanning multiple applications. Each task example is derived from real-world computer use cases and includes a detailed initial state setup configuration and a custom execution-based evaluation script for reliable, reproducible evaluation. Extensive evaluation of state-of-the-art LLM/VLM-based agents on OSWorld reveals significant deficiencies in their ability to serve as computer assistants. While humans can accomplish over 72.36% of the tasks, the best model achieves only 12.24% success, primarily struggling with GUI grounding and operational knowledge. Comprehensive analysis using OSWorld provides valuable insights for developing multimodal generalist agents that were not possible with previous benchmarks. Our code, environment, baseline models, and data are publicly available at this https URL .

</details>

---

## 215. MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs

- [ ] MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs | https://neurips.cc/virtual/2024/poster/97480

- **Link**: https://neurips.cc/virtual/2024/poster/97480

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generating natural and meaningful responses to communicate with multi-modal human inputs is a fundamental capability of Large Vision-Language Models (LVLMs). While current open-source LVLMs demonstrate promising performance in simplified scenarios such as single-turn single-image input, they fall short in real-world conversation scenarios such as following instructions in a long context history with multi-turn and multi-images. Existing LVLM benchmarks primarily focus on single-choice questions or short-form responses, which do not adequately assess the capabilities of LVLMs in real-world human-AI interaction applications. Therefore, we introduce MMDU, a comprehensive benchmark, and MMDU-45k, a large-scale instruction tuning dataset, designed to evaluate and improve LVLMs' abilities in multi-turn and multi-image conversations. We employ the clustering algorithm to find the relevant images and textual descriptions from the open-source Wikipedia and construct the question-answer pairs by human annotators with the assistance of the GPT-4o model.MMDU has a maximum of 18k image+text tokens, 20 images, and 27 turns, which is at least 5x longer than previous benchmarks and poses challenges to current LVLMs. Our in-depth analysis of 15 representative LVLMs using MMDU reveals that open-source LVLMs lag behind closed-source counterparts due to limited conversational instruction tuning data.We demonstrate that fine-tuning open-source LVLMs on MMDU-45k significantly address this gap, generating longer and more accurate conversations, and improving scores on MMDU and existing benchmarks (MMStar: +1.1%, MathVista: +1.5%, ChartQA: +1.2%). Our contributions pave the way for bridging the gap between current LVLM models and real-world application demands. The links to MMDU, and MMDU-45k are available in the supplementary material.

</details>

---

## 216. What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction

- [ ] What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction | https://neurips.cc/virtual/2024/poster/97489

- **Link**: https://neurips.cc/virtual/2024/poster/97489

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models have shown impressive progress in recent years. However, existing models are largely limited to turn-based interactions, where each turn must be stepped (i.e., prompted) by the user. Open-ended, asynchronous interactions, where an AI model may proactively deliver timely responses or feedback based on the unfolding situation in real-time, are an open challenge. In this work, we present the QEVD benchmark and dataset, which explores human-AI interaction in the challenging, yet controlled, real-world domain of fitness coaching – a task which intrinsically requires monitoring live user activity and providing immediate feedback. The benchmark requires vision-language models to recognize complex human actions, identify possible mistakes, and provide appropriate feedback in real-time. Our experiments reveal the limitations of existing state-of-the-art vision-language models for such asynchronous situated interactions. Motivated by this, we propose a simple end-to-end streaming baseline that can respond asynchronously to human actions with appropriate feedback at the appropriate time.

</details>

---

## 217. MMM-RS: A Multi-modal, Multi-GSD, Multi-scene Remote Sensing  Dataset and Benchmark for Text-to-Image Generation

- [ ] MMM-RS: A Multi-modal, Multi-GSD, Multi-scene Remote Sensing  Dataset and Benchmark for Text-to-Image Generation | https://neurips.cc/virtual/2024/poster/97495

- **Link**: https://neurips.cc/virtual/2024/poster/97495

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, the diffusion-based generative paradigm has achieved impressive general image generation capabilities with text prompts due to its accurate distribution modeling and stable training process. However, generating diverse remote sensing (RS) images that are tremendously different from general images in terms of scale and perspective remains a formidable challenge due to the lack of a comprehensive remote sensing image generation dataset with various modalities, ground sample distances (GSD), and scenes. In this paper, we propose a Multi-modal, Multi-GSD, Multi-scene Remote Sensing (MMM-RS) dataset and benchmark for text-to-image generation in diverse remote sensing scenarios. Specifically, we first collect nine publicly available RS datasets and conduct standardization for all samples. To bridge RS images to textual semantic information, we utilize a large-scale pretrained vision-language model to automatically output text prompts and perform hand-crafted rectification, resulting in information-rich text-image pairs (including multi-modal images). In particular, we design some methods to obtain the images with different GSD and various environments (e.g., low-light, foggy) in a single sample. With extensive manual screening and refining annotations, we ultimately obtain a MMM-RS dataset that comprises approximately 2.1 million text-image pairs. Extensive experimental results verify that our proposed MMM-RS dataset allows off-the-shelf diffusion models to generate diverse RS images across various modalities, scenes, weather conditions, and GSD. The dataset is available at https://github.com/ljl5261/MMM-RS.

</details>

---

## 218. Task Me Anything

- [ ] Task Me Anything | https://neurips.cc/virtual/2024/poster/97494

- **Link**: https://neurips.cc/virtual/2024/poster/97494

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Benchmarks for large multimodal language models (MLMs) now serve to simultaneously assess the general capabilities of models instead of evaluating for a specific capability. As a result, when a developer wants to identify which models to use for their application, they are overwhelmed by the number of benchmarks and remain uncertain about which benchmark's results are most reflective of their specific use case. This paper introduces Task-Me-Anything, a benchmark generation engine which produces a benchmark tailored to a user's needs. Task-Me-Anything maintains an extendable taxonomy of visual assets and can programmatically generate a vast number of task instances. Additionally, it algorithmically addresses user queries regarding MLM performance efficiently within a computational budget. It contains 113K images, 10K videos, 2K 3D object assets, over 365 object categories, 655 attributes, and 335 relationships. It can generate 500M image/video question-answering pairs, which focus on evaluating MLM perceptual capabilities. Task-Me-Anything reveals critical insights: open-source MLMs excel in object and attribute recognition but lack spatial and temporal understanding; each model exhibits unique strengths and weaknesses; larger models generally perform better, though exceptions exist; and GPT4O demonstrates challenges in recognizing rotating/moving objects and distinguishing colors.

</details>

---

## 219. JourneyBench: A Challenging One-Stop Vision-Language Understanding Benchmark of Generated Images

- [ ] JourneyBench: A Challenging One-Stop Vision-Language Understanding Benchmark of Generated Images | https://neurips.cc/virtual/2024/poster/97518

- **Link**: https://neurips.cc/virtual/2024/poster/97518

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing vision-language understanding benchmarks largely consist of images of objects in their usual contexts.As a consequence, recent multimodal large language models can perform well with only a shallow visual understanding by relying on background language biases. Thus, strong performance on these benchmarks does not necessarily correlate with strong visual understanding. In this paper, we release JourneyBench, a comprehensive human-annotated benchmark of generated images designed to assess the model's fine-grained multimodal reasoning abilities across five tasks: complementary multimodal chain of thought, multi-image VQA, imaginary image captioning, VQA with hallucination triggers, and fine-grained retrieval with sample-specific distractors.Unlike existing benchmarks, JourneyBench explicitly requires fine-grained multimodal reasoning in unusual imaginary scenarios where language bias and holistic image gist are insufficient. We benchmark state-of-the-art models on JourneyBench and analyze performance along a number of fine-grained dimensions. Results across all five tasks show that JourneyBench is exceptionally challenging for even the best models, indicating that models' visual reasoning abilities are not as strong as they first appear. We discuss the implications of our findings and propose avenues for further research.

</details>

---

## 220. MMLONGBENCH-DOC: Benchmarking Long-context Document Understanding with Visualizations

- [ ] MMLONGBENCH-DOC: Benchmarking Long-context Document Understanding with Visualizations | https://neurips.cc/virtual/2024/poster/97524

- **Link**: https://neurips.cc/virtual/2024/poster/97524

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding documents with rich layouts and multi-modal components is a long-standing and practical task. Recent Large Vision-Language Models (LVLMs) have made remarkable strides in various tasks, particularly in single-page document understanding (DU). However, their abilities on long-context DU remain an open problem. This work presents MMLONGBENCH-DOC, a long-context, multi- modal benchmark comprising 1,082 expert-annotated questions. Distinct from previous datasets, it is constructed upon 135 lengthy PDF-formatted documents with an average of 47.5 pages and 21,214 textual tokens. Towards comprehensive evaluation, answers to these questions rely on pieces of evidence from (1) different sources (text, image, chart, table, and layout structure) and (2) various locations (i.e., page number). Moreover, 33.7\% of the questions are cross-page questions requiring evidence across multiple pages. 20.6\% of the questions are designed to be unanswerable for detecting potential hallucinations. Experiments on 14 LVLMs demonstrate that long-context DU greatly challenges current models. Notably, the best-performing model, GPT-4o, achieves an F1 score of only 44.9\%, while the second-best, GPT-4V, scores 30.5\%. Furthermore, 12 LVLMs (all except GPT-4o and GPT-4V) even present worse performance than their LLM counterparts which are fed with lossy-parsed OCR documents. These results validate the necessity of future research toward more capable long-context LVLMs.

</details>

---

## 221. VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding

- [ ] VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding | https://neurips.cc/virtual/2024/poster/97530

- **Link**: https://neurips.cc/virtual/2024/poster/97530

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a new benchmark designed to advance the development of general-purpose, large-scale vision-language models for remote sensing images. Although several vision-language datasets in remote sensing have been proposed to pursue this goal, existing datasets are typically tailored to single tasks, lack detailed object information, or suffer from inadequate quality control. Exploring these improvement opportunities, we present a Versatile vision-language Benchmark for Remote Sensing image understanding, termed VRSBench. This benchmark comprises 29,614 images, with 29,614 human-verified detailed captions, 52,472 object references, and 123,221 question-answer pairs. It facilitates the training and evaluation of vision-language models across a broad spectrum of remote sensing image understanding tasks. We further evaluated state-of-the-art models on this benchmark for three vision-language tasks: image captioning, visual grounding, and visual question answering. Our work aims to significantly contribute to the development of advanced vision-language models in the field of remote sensing. The data and code can be accessed at https://vrsbench.github.io.

</details>

---

## 222. Image Textualization: An Automatic Framework for Generating Rich and Detailed Image Descriptions

- [ ] Image Textualization: An Automatic Framework for Generating Rich and Detailed Image Descriptions | https://neurips.cc/virtual/2024/poster/97538

- **Link**: https://neurips.cc/virtual/2024/poster/97538

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image description datasets play a crucial role in the advancement of various applications such as image understanding, text-to-image generation, and text-image retrieval. Currently, image description datasets primarily originate from two sources. One source is the scraping of image-text pairs from the web. Despite their abundance, these descriptions are often of low quality and noisy. Another way is through human labeling. Datasets such as COCO are generally very short and lack details. Although detailed image descriptions can be annotated by humans, the high cost limits their quantity and feasibility. These limitations underscore the need for more efficient and scalable methods to generate accurate and detailed image descriptions. In this paper, we propose an innovative framework termed Image Textualization, which automatically produces high-quality image descriptions by leveraging existing mult-modal large language models (MLLMs) and multiple vision expert models in a collaborative manner. We conduct various experiments to validate the high quality of the descriptions constructed by our framework. Furthermore, we show that MLLMs fine-tuned on our dataset acquire an unprecedented capability of generating richer image descriptions, substantially increasing the length and detail of their output with even less hallucinations.

</details>

---

## 223. MLLMGuard: A Multi-dimensional Safety Evaluation Suite for Multimodal Large Language Models

- [ ] MLLMGuard: A Multi-dimensional Safety Evaluation Suite for Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/97540

- **Link**: https://neurips.cc/virtual/2024/poster/97540

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Powered by remarkable advancements in Large Language Models (LLMs), Multimodal Large Language Models (MLLMs) demonstrate impressive capabilities in manifold tasks.However, the practical application scenarios of MLLMs are intricate, exposing them to potential malicious instructions and thereby posing safety risks.While current benchmarks do incorporate certain safety considerations, they often lack comprehensive coverage and fail to exhibit the necessary rigor and robustness.For instance, the common practice of employing GPT-4V as both the evaluator and a model to be evaluated lacks credibility, as it tends to exhibit a bias toward its own responses.In this paper, we present MLLMGuard, a multi-dimensional safety evaluation suite for MLLMs, including a bilingual image-text evaluation dataset, inference utilities, and a lightweight evaluator.MLLMGuard's assessment comprehensively covers two languages (English and Chinese) and five important safety dimensions (Privacy, Bias, Toxicity, Truthfulness, and Legality), each with corresponding rich subtasks.Focusing on these dimensions, our evaluation dataset is primarily sourced from platforms such as social media, and it integrates text-based and image-based red teaming techniques with meticulous annotation by human experts.This can prevent inaccurate evaluation caused by data leakage when using open-source datasets and ensures the quality and challenging nature of our benchmark.Additionally, a fully automated lightweight evaluator termed GuardRank is developed, which achieves significantly higher evaluation accuracy than GPT-4.Our evaluation results across 13 advanced models indicate that MLLMs still have a substantial journey ahead before they can be considered safe and responsible.

</details>

---

## 224. INQUIRE: A Natural World Text-to-Image Retrieval Benchmark

- [ ] INQUIRE: A Natural World Text-to-Image Retrieval Benchmark | https://neurips.cc/virtual/2024/poster/97543

- **Link**: https://neurips.cc/virtual/2024/poster/97543

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce INQUIRE, a text-to-image retrieval benchmark designed to challenge multimodal vision-language models on expert-level queries. INQUIRE includes iNaturalist 2024 (iNat24), a new dataset of five million natural world images, along with 250 expert-level retrieval queries. These queries are paired with all relevant images comprehensively labeled within iNat24, comprising 33,000 total matches. Queries span categories such as species identification, context, behavior, and appearance, emphasizing tasks that require nuanced image understanding and domain expertise. Our benchmark evaluates two core retrieval tasks: (1) INQUIRE-Fullrank, a full dataset ranking task, and (2) INQUIRE-Rerank, a reranking task for refining top-100 retrievals. Detailed evaluation of a range of recent multimodal models demonstrates that INQUIRE poses a significant challenge, with the best models failing to achieve an mAP@50 above 50%. In addition, we show that reranking with more powerful multimodal models can enhance retrieval performance, yet there remains a significant margin for improvement. By focusing on scientifically-motivated ecological challenges, INQUIRE aims to bridge the gap between AI capabilities and the needs of real-world scientific inquiry, encouraging the development of retrieval systems that can assist with accelerating ecological and biodiversity research.

</details>

---

## 225. BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays

- [ ] BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays | https://neurips.cc/virtual/2024/poster/97555

- **Link**: https://neurips.cc/virtual/2024/poster/97555

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Medical Vision-Language Pretraining (MedVLP) shows promise in learning generalizable and transferable visual representations from paired and unpaired medical images and reports. MedVLP can provide useful features to downstream tasks and facilitate adapting task-specific models to new setups using fewer examples. However, existing MedVLP methods often differ in terms of datasets, preprocessing, and finetuning implementations. This pose great challenges in evaluating how well a MedVLP method generalizes to various clinically-relevant tasks due to the lack of unified, standardized, and comprehensive benchmark. To fill this gap, we propose BenchX, a unified benchmark framework that enables head-to-head comparison and systematical analysis between MedVLP methods using public chest X-ray datasets. Specifically, BenchX is composed of three components: 1) Comprehensive datasets covering nine datasets and four medical tasks; 2) Benchmark suites to standardize data preprocessing, train-test splits, and parameter selection; 3) Unified finetuning protocols that accommodate heterogeneous MedVLP methods for consistent task adaptation in classification, segmentation, and report generation, respectively. Utilizing BenchX, we establish baselines for nine state-of-the-art MedVLP methods and found that the performance of some early MedVLP methods can be enhanced to surpass more recent ones, prompting a revisiting of the developments and conclusions from prior works in MedVLP. Our code are available at https://github.com/yangzhou12/BenchX.

</details>

---

## 226. WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences

- [ ] WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences | https://neurips.cc/virtual/2024/poster/97560

- **Link**: https://neurips.cc/virtual/2024/poster/97560

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent breakthroughs in vision-language models (VLMs) emphasize the necessity of benchmarking human preferences in real-world multimodal interactions. To address this gap, we launched WildVision-Arena (WV-Arena), an online platform that collects human preferences to evaluate VLMs. We curated WV-Bench by selecting 500 high-quality samples from 8,000 user submissions in WV-Arena. WV-Bench uses GPT-4 as the judge to compare each VLM with Claude-3-Sonnet, achieving a Spearman correlation of 0.94 with the WV-Arena Elo. This significantly outperforms other benchmarks like MMVet, MMMU, and MMStar.Our comprehensive analysis of 20K real-world interactions reveals important insights into the failure cases of top-performing VLMs. For example, we find that although GPT-4V surpasses many other models like Reka-Flash, Opus, and Yi-VL-Plus in simple visual recognition and reasoning tasks, it still faces challenges with subtle contextual cues, spatial reasoning, visual imagination, and expert domain knowledge. Additionally, current VLMs exhibit issues with hallucinations and safety when intentionally provoked. We are releasing our chat and feedback data to further advance research in the field of VLMs.

</details>

---

## 227. II-Bench: An Image Implication Understanding Benchmark for Multimodal Large Language Models

- [ ] II-Bench: An Image Implication Understanding Benchmark for Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/97557

- **Link**: https://neurips.cc/virtual/2024/poster/97557

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancements in the development of multimodal large language models (MLLMs) have consistently led to new breakthroughs on various benchmarks. In response, numerous challenging and comprehensive benchmarks have been proposed to more accurately assess the capabilities of MLLMs. However, there is a dearth of exploration of the higher-order perceptual capabilities of MLLMs. To fill this gap, we propose the Image Implication understanding Benchmark, II-Bench, which aims to evaluate the model's higher-order perception of images. Through extensive experiments on II-Bench across multiple MLLMs, we have made significant findings. Initially, a substantial gap is observed between the performance of MLLMs and humans on II-Bench. The pinnacle accuracy of MLLMs attains 74.8%, whereas human accuracy averages 90%, peaking at an impressive 98%. Subsequently, MLLMs perform worse on abstract and complex images, suggesting limitations in their ability to understand high-level semantics and capture image details. Finally, it is observed that most models exhibit enhanced accuracy when image sentiment polarity hints are incorporated into the prompts. This observation underscores a notable deficiency in their inherent understanding of image sentiment. We believe that II-Bench will inspire the community to develop the next generation of MLLMs, advancing the journey towards expert  artificial general intelligence (AGI). II-Bench is publicly available at https://huggingface.co/datasets/m-a-p/II-Bench.

</details>

---

## 228. DenseFusion-1M: Merging Vision Experts for Comprehensive Multimodal Perception

- [ ] DenseFusion-1M: Merging Vision Experts for Comprehensive Multimodal Perception | https://neurips.cc/virtual/2024/poster/97564

- **Link**: https://neurips.cc/virtual/2024/poster/97564

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing Multimodal Large Language Models (MLLMs) increasingly emphasize complex understanding of various visual elements, including multiple objects, text information, spatial relations. Their development for comprehensive visual perception hinges on the availability of high-quality image-text datasets that offer diverse visual elements and throughout image descriptions. However, the scarcity of such hyper-detailed datasets currently hinders progress within the MLLM community. The bottleneck stems from the limited perceptual capabilities of current caption engines, which fall short in providing complete and accurate annotations. To facilitate the cutting-edge research of MLLMs on comprehensive vision perception, we thereby propose Perceptual Fusion, using a low-budget but highly effective caption engine for complete and accurate image descriptions.  Specifically, Perceptual Fusion integrates diverse perception experts as image priors to provide explicit information on visual elements and adopts an efficient MLLM as a centric pivot to mimic advanced MLLMs' perception abilities. We carefully select 1M highly representative images from uncurated LAION dataset and generate dense descriptions using our engine, dubbed DenseFusion-1M. Extensive experiments validate that our engine outperforms its counterparts, where the resulting dataset significantly improves the perception and cognition abilities of existing MLLMs across diverse vision-language benchmarks, especially with high-resolution images as inputs. The code and dataset are available at https://huggingface.co/datasets/BAAI/DenseFusion-1M.

</details>

---

## 229. ViLCo-Bench: VIdeo Language COntinual learning Benchmark

- [ ] ViLCo-Bench: VIdeo Language COntinual learning Benchmark | https://neurips.cc/virtual/2024/poster/97567

- **Link**: https://neurips.cc/virtual/2024/poster/97567

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video language continual learning involves continuously adapting to information from video and text inputs, enhancing a model’s ability to handle new tasks while retaining prior knowledge. This field is a relatively under-explored area, and establishing appropriate datasets is crucial for facilitating communication and research in this field. In this study, we present the first dedicated benchmark, ViLCo-Bench, designed to evaluate continual learning models across a range of video-text tasks. The dataset comprises ten-minute-long videos and corresponding language queries collected from publicly available datasets. Additionally, we introduce a novel memory-efficient framework that incorporates self-supervised learning and mimics long-term and short-term memory effects. This framework addresses challenges including memory complexity from long video clips, natural language complexity from open queries, and text-video misalignment. We posit that ViLCo-Bench, with greater complexity compared to existing continual learning benchmarks, would serve as a critical tool for exploring the video-language domain, extending beyond conventional class-incremental tasks, and addressing complex and limited annotation issues. The curated data, evaluations, and our novel method are available at https://github.com/cruiseresearchgroup/ViLCo.

</details>

---

## 230. Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs

- [ ] Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs | https://neurips.cc/virtual/2024/poster/97572

- **Link**: https://neurips.cc/virtual/2024/poster/97572

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have shown impressive success across modalities such as image, video, and audio in a variety of understanding and generation tasks.  However, current MLLMs are surprisingly poor at understanding webpage screenshots and generating their corresponding HTML code.  To address this problem,   we propose Web2Code, a benchmark consisting of a new large-scale webpage-to-code dataset for instruction tuning and an evaluation framework for the webpage understanding and HTML code translation abilities of MLLMs.   For dataset construction, we leverage pretrained LLMs to enhance existing webpage-to-code datasets as well as generate a diverse pool of new webpages rendered into images.  Specifically, the inputs are webpage images and instructions, while the responses are the webpage's HTML code.  We further include diverse natural language QA pairs about the webpage content in the responses to enable a more comprehensive understanding of the web content.  To evaluate model performance in these tasks, we develop an evaluation framework for testing MLLMs' abilities in webpage understanding and web-to-code generation.  Extensive experiments show that our proposed dataset is beneficial not only to our proposed tasks but also in the general visual domain.  We hope our work will contribute to the development of general MLLMs suitable for web-based content generation and task automation.  Our data and code are available at https://github.com/MBZUAI-LLM/web2code.

</details>

---

## 231. SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers

- [ ] SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers | https://neurips.cc/virtual/2024/poster/97575

- **Link**: https://neurips.cc/virtual/2024/poster/97575

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Seeking answers to questions within long scientific research articles is a crucial area of study that aids readers in quickly addressing their inquiries. However, existing question-answering (QA) datasets based on scientific papers are limited in scale and focus solely on textual content. We introduce SPIQA (Scientific Paper Image Question Answering), the first large-scale QA dataset specifically designed to interpret complex figures and tables within the context of scientific research articles across various domains of computer science. Leveraging the breadth of expertise and ability of multimodal large language models (MLLMs) to understand figures, we employ automatic and manual curation to create the dataset. We craft an information-seeking task on interleaved images and text that involves multiple images covering a wide variety of plots, charts, tables, schematic diagrams, and result visualizations. SPIQA comprises 270K questions divided into training, validation, and three different evaluation splits. Through extensive experiments with 12 prominent foundational models, we evaluate the ability of current multimodal systems to comprehend the nuanced aspects of research articles. Additionally, we propose a Chain-of-Thought (CoT) evaluation strategy with in-context retrieval that allows fine-grained, step-by-step assessment and improves model performance. We further explore the upper bounds of performance enhancement with additional textual information, highlighting its promising potential for future research and the dataset’s impact on revolutionizing how we interact with scientific literature.

</details>

---

## 232. UniBench: Visual Reasoning Requires Rethinking Vision-Language Beyond Scaling

- [ ] UniBench: Visual Reasoning Requires Rethinking Vision-Language Beyond Scaling | https://neurips.cc/virtual/2024/poster/97581

- **Link**: https://neurips.cc/virtual/2024/poster/97581

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Significant research efforts have been made to scale and improve vision-language model (VLM) training approaches. Yet, with an ever-growing number of benchmarks,researchers are tasked with the heavy burden of implementing each protocol, bearing a non-trivial computational cost, and making sense of how all these benchmarks translate into meaningful axes of progress.To facilitate a systematic evaluation of VLM progress, we introduce UniBench: a unified implementation of 50+ VLM benchmarks spanning a range of carefully categorized vision-centric capabilities from object recognition to spatial awareness, counting, and much more. We showcase the utility of UniBench for measuring progress by evaluating nearly 60 publicly available vision-language models, trained on scales of up to 12.8B samples. We find that while scaling training data or model size can boost many vision-language model capabilities, scaling offers little benefit for reasoning or relations.  Surprisingly, we also discover today's best VLMs struggle on simple digit recognition and counting tasks, e.g. MNIST, which much simpler networks can solve. Where scale falls short, we find that more precise interventions, such as data quality or tailored-learning objectives offer more promise. For practitioners, we also offer guidance on selecting a suitable VLM for a given application. Finally, we release an easy-to-run UniBench code-base with the full set of 50+ benchmarks and comparisons across 59 models as well as a distilled, representative set of benchmarks that runs in 5 minutes on a single GPU. UniBench with model evaluations on all benchmarks are provided as a toolbox at: https://github.com/facebookresearch/unibench

</details>

---

## 233. Micro-Bench: A Microscopy Benchmark for Vision-Language Understanding

- [ ] Micro-Bench: A Microscopy Benchmark for Vision-Language Understanding | https://neurips.cc/virtual/2024/poster/97589

- **Link**: https://neurips.cc/virtual/2024/poster/97589

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in microscopy have enabled the rapid generation of terabytes of image data in cell biology and biomedical research. Vision-language models (VLMs) offer a promising solution for large-scale biological image analysis, enhancing researchers’ efficiency, identifying new image biomarkers, and accelerating hypothesis generation and scientific discovery. However, there is a lack of standardized, diverse, and large-scale vision-language benchmarks to evaluate VLMs’ perception and cognition capabilities in biological image understanding. To address this gap, we introduce Micro-Bench, an expert-curated benchmark encompassing 24 biomedical tasks across various scientific disciplines (biology, pathology), microscopy modalities (electron, fluorescence, light), scales (subcellular, cellular, tissue), and organisms in both normal and abnormal states. We evaluate state-of-the-art biomedical, pathology, and general VLMs on Micro-Bench and find that: i) current models struggle on all categories, even for basic tasks such as distinguishing microscopy modalities; ii) current specialist models fine-tuned on biomedical data often perform worse than generalist models; iii) fine-tuning in specific microscopy domains can cause catastrophic forgetting, eroding prior biomedical knowledge encoded in their base model. iv) weight interpolation between fine-tuned and pre-trained models offers one solution to forgetting and improves general performance across biomedical tasks. We release Micro-Bench under a permissive license to accelerate the research and development of microscopy foundation models.

</details>

---

## 234. CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs

- [ ] CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs | https://neurips.cc/virtual/2024/poster/97598

- **Link**: https://neurips.cc/virtual/2024/poster/97598

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Chart understanding plays a pivotal role when applying Multimodal Large Language Models (MLLMs) to real-world tasks such as analyzing scientific papers or financial reports. However, existing datasets often focus on oversimplified and homogeneous charts with template-based questions, leading to an overly optimistic measure of progress. We demonstrate that although open-source models can appear to outperform strong proprietary models on these benchmarks, a simple stress test with slightly different charts or questions deteriorates performance by up to 34.5%. In this work, we propose CharXiv, a comprehensive evaluation suite involving 2,323 natural, challenging, and diverse charts from scientific papers. CharXiv includes two types of questions: 1) descriptive questions about examining basic chart elements and 2) reasoning questions that require synthesizing information across complex visual elements in the chart. To ensure quality, all charts and questions are handpicked, curated, and verified by human experts. Our results reveal a substantial, previously underestimated gap between the reasoning skills of the strongest proprietary model (i.e., GPT-4o), which achieves 47.1% accuracy, and the strongest open-source model (i.e., InternVL Chat V1.5), which achieves 29.2%. All models lag far behind human performance of 80.5%, underscoring weaknesses in the chart understanding capabilities of existing MLLMs. We hope that CharXiv facilitates future research on MLLM chart understanding by providing a more realistic and faithful measure of progress. Project website: https://charxiv.github.io/

</details>

---

## 235. CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset

- [ ] CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset | https://neurips.cc/virtual/2024/poster/97600

- **Link**: https://neurips.cc/virtual/2024/poster/97600

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Machine learning models are increasingly being deployed in real-world contexts. However, systematic studies on their transferability to specific and critical applications are underrepresented in the research literature. An important example is visual anomaly detection (VAD) for robotic power line inspection. While existing VAD methods perform well in controlled environments, real-world scenarios present diverse and unexpected anomalies that current datasets fail to capture. To address this gap, we introduce CableInspect-AD, a high-quality, publicly available dataset created and annotated by domain experts from Hydro-Québec, a Canadian public utility. This dataset includes high-resolution images with challenging real-world anomalies, covering defects with varying severity levels. To address the challenges of collecting diverse anomalous and nominal examples for setting a detection threshold, we propose an enhancement to the celebrated PatchCore algorithm. This enhancement enables its use in scenarios with limited labeled data. We also present a comprehensive evaluation protocol based on cross-validation to assess models' performances. We evaluate our Enhanced-PatchCore for few-shot and many-shot detection, and Vision-Language Models for zero-shot detection. While promising, these models struggle to detect all anomalies, highlighting the dataset's value as a challenging benchmark for the broader research community. Project page: https://mila-iqia.github.io/cableinspect-ad/.

</details>

---

## 236. A Hitchhiker's Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning

- [ ] A Hitchhiker's Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning | https://neurips.cc/virtual/2024/poster/97603

- **Link**: https://neurips.cc/virtual/2024/poster/97603

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Explainability in artificial intelligence is crucial for restoring trust, particularly in areas like face forgery detection, where viewers often struggle to distinguish between real and fabricated content. Vision and Large Language Models (VLLM) bridge computer vision and natural language, offering numerous applications driven by strong common-sense reasoning. Despite their success in various tasks, the potential of vision and language remains underexplored in face forgery detection, where they hold promise for enhancing explainability by leveraging the intrinsic reasoning capabilities of language to analyse fine-grained manipulation areas.    For that reason, few works have recently started to frame the problem of deepfake detection as a Visual Question Answering (VQA) task, nevertheless omitting the realistic and informative open-ended multi-label setting. With the rapid advances in the field of VLLM, an exponential rise of investigations in that direction is expected.    As such, there is a need for a clear experimental methodology that converts face forgery detection to a Visual Question Answering (VQA) task to systematically and fairly evaluate different VLLM architectures. Previous evaluation studies in deepfake detection have mostly focused on the simpler binary task, overlooking evaluation protocols for multi-label fine-grained detection and text-generative models. We propose a multi-staged approach that diverges from the traditional binary evaluation protocol and conducts a comprehensive evaluation study to compare the capabilities of several VLLMs in this context.    In the first stage, we assess the models' performance on the binary task and their sensitivity to given instructions using several prompts. In the second stage, we delve deeper into fine-grained detection by identifying areas of manipulation in a multiple-choice VQA setting. In the third stage, we convert the fine-grained detection to an open-ended question and compare several matching strategies for the multi-label classification task. Finally, we qualitatively evaluate the fine-grained responses of the VLLMs included in the benchmark.    We apply our benchmark to several popular models, providing a detailed comparison of binary, multiple-choice, and open-ended VQA evaluation across seven datasets. \url{https://nickyfot.github.io/hitchhickersguide.github.io/}

</details>

---

## 237. WebUOT-1M: Advancing Deep Underwater Object Tracking with A Million-Scale Benchmark

- [ ] WebUOT-1M: Advancing Deep Underwater Object Tracking with A Million-Scale Benchmark | https://neurips.cc/virtual/2024/poster/97605

- **Link**: https://neurips.cc/virtual/2024/poster/97605

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Underwater Object Tracking (UOT) is essential for identifying and tracking submerged objects in underwater videos, but existing datasets are limited in scale, diversity of target categories and scenarios covered, impeding the development of advanced tracking algorithms. To bridge this gap, we take the first step and introduce WebUOT-1M, \ie, the largest public UOT benchmark to date, sourced from complex and realistic underwater environments. It comprises 1.1 million frames across 1,500 video clips filtered from 408 target categories, largely surpassing previous UOT datasets, \eg, UVOT400. Through meticulous manual annotation and verification, we provide high-quality bounding boxes for underwater targets. Additionally, WebUOT-1M includes language prompts for video sequences, expanding its application areas, \eg, underwater vision-language tracking. Given that most existing trackers are designed for open-air conditions and perform poorly in underwater environments due to domain gaps, we propose a novel framework that uses omni-knowledge distillation to train a student Transformer model effectively. To the best of our knowledge, this framework is the first to effectively transfer open-air domain knowledge to the UOT model through knowledge distillation, as demonstrated by results on both existing UOT datasets and the newly proposed WebUOT-1M. We have thoroughly tested WebUOT-1M with 30 deep trackers, showcasing its potential as a benchmark for future UOT research. The complete dataset, along with codes and tracking results, are publicly accessible at \href{https://github.com/983632847/Awesome-Multimodal-Object-Tracking}{\color{magenta}{here}}.

</details>

---

## 238. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models

- [ ] CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models | https://neurips.cc/virtual/2024/poster/97614

- **Link**: https://neurips.cc/virtual/2024/poster/97614

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://github.com/richard-peng-xia/CARES.

</details>

---

## 239. Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning

- [ ] Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning | https://neurips.cc/virtual/2024/poster/97623

- **Link**: https://neurips.cc/virtual/2024/poster/97623

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-Modal Large Language Models (MLLMs) have demonstrated impressive performance in various VQA tasks. However, they often lack interpretability and struggle with complex visual inputs, especially when the resolution of the input image is high or when the interested region that could provide key information for answering the question is small. To address these challenges, we collect and introduce the large-scale Visual CoT dataset comprising 438k question-answer pairs, annotated with intermediate bounding boxes highlighting key regions essential for answering the questions. Additionally, about 98k pairs of them are annotated with detailed reasoning steps. Importantly, we propose a multi-turn processing pipeline that dynamically focuses on visual inputs and provides interpretable thoughts. We also introduce the related benchmark to evaluate the MLLMs in scenarios requiring specific local region identification.Extensive experiments demonstrate the effectiveness of our framework and shed light on better inference strategies. The Visual CoT dataset, benchmark, and pre-trained models are available on this website to support further research in this area.

</details>

---

## 240. Evaluating Large Vision-and-Language Models on Children's Mathematical Olympiads

- [ ] Evaluating Large Vision-and-Language Models on Children's Mathematical Olympiads | https://neurips.cc/virtual/2024/poster/97639

- **Link**: https://neurips.cc/virtual/2024/poster/97639

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent years have seen a significant progress in the general-purpose problem solving abilities of large vision and language models (LVLMs), such as ChatGPT, Gemini, etc.; some of these breakthroughs even seem to enable AI models to outperform human abilities in varied tasks that demand higher-order cognitive skills. Are the current large AI models indeed capable of generalized problem solving as humans do?  A systematic analysis of AI capabilities for joint vision and text reasoning, however, is missing in the current scientific literature. In this paper, we make an effort towards filling this gap, by evaluating state-of-the-art LVLMs on their mathematical and algorithmic reasoning abilities using visuo-linguistic problems from children's Olympiads. Specifically, we consider problems from the Mathematical Kangaroo (MK) Olympiad, which is a popular international competition targeted at children from grades 1-12, that tests children's deeper mathematical abilities using puzzles that are appropriately gauged to their age and skills. Using the puzzles from MK, we created a dataset, dubbed SMART-840, consisting of 840 problems from years 2020-2024. With our dataset, we analyze LVLMs power on mathematical reasoning; their responses on our puzzles offer a direct way to compare against that of children. Our results show that modern LVLMs do demonstrate increasingly powerful reasoning skills in solving problems for higher grades, but lack the foundations to correctly answer problems designed for younger children. Further analysis shows that there is no significant correlation between the reasoning capabilities of AI models and that of young children, and their capabilities appear to be based on a different type of reasoning than the cumulative knowledge that underlies children's mathematical skills.

</details>

---

## 241. BiVLC: Extending Vision-Language Compositionality Evaluation with Text-to-Image Retrieval

- [ ] BiVLC: Extending Vision-Language Compositionality Evaluation with Text-to-Image Retrieval | https://neurips.cc/virtual/2024/poster/97657

- **Link**: https://neurips.cc/virtual/2024/poster/97657

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing Vision-Language Compositionality (VLC) benchmarks like SugarCrepe are formulated as image-to-text retrieval problems, where, given an image, the models need to select between the correct textual description and a synthetic hard negative text. In this work, we present the Bidirectional Vision-Language Compositionality (BiVLC) dataset. The novelty of BiVLC is to add a synthetic hard negative image generated from the synthetic text, resulting in two image-to-text retrieval examples (one for each image) and, more importantly, two text-to-image retrieval examples (one for each text). Human annotators filter out ill-formed examples ensuring the validity of the benchmark. The experiments on BiVLC uncover a weakness of current multimodal models, as they perform poorly in the text-to-image direction. In fact, when considering both retrieval directions, the conclusions obtained in previous works change significantly. In addition to the benchmark, weshow that a contrastive model trained using synthetic images and texts significantly improves over the base model in SugarCrepe and in BiVLC for both retrieval directions. The gap to human performance in BiVLC confirms that Vision-Language Compositionality is still a challenging problem.

</details>

---

## 242. Hidden in Plain Sight: Evaluating Abstract Shape Recognition in Vision-Language Models

- [ ] Hidden in Plain Sight: Evaluating Abstract Shape Recognition in Vision-Language Models | https://neurips.cc/virtual/2024/poster/97667

- **Link**: https://neurips.cc/virtual/2024/poster/97667

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the importance of shape perception in human vision, early neural image classifiers relied less on shape information for object recognition than other (often spurious) features. While recent research suggests that current large Vision-Language Models (VLMs) exhibit more reliance on shape, we find them to still be seriously limited in this regard. To quantify such limitations, we introduce IllusionBench, a dataset that challenges current cutting-edge VLMs to decipher shape information when the shape is represented by an arrangement of visual elements in a scene. Our extensive evaluations reveal that, while these shapes are easily detectable by human annotators, current VLMs struggle to recognize them, indicating important avenues for future work in developing more robust visual perception systems. The full dataset and codebase are available at: https://arshiahemmat.github.io/illusionbench/

</details>

---

## 243. VLM4Bio: A Benchmark Dataset to Evaluate Pretrained Vision-Language Models for Trait Discovery from Biological Images

- [ ] VLM4Bio: A Benchmark Dataset to Evaluate Pretrained Vision-Language Models for Trait Discovery from Biological Images | https://neurips.cc/virtual/2024/poster/97668

- **Link**: https://neurips.cc/virtual/2024/poster/97668

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Images are increasingly becoming the currency for documenting biodiversity on the planet, providing novel opportunities for accelerating scientific discoveries in the field of organismal biology, especially with the advent of large vision-language models (VLMs). We ask if pre-trained VLMs can aid scientists in answering a range of biologically relevant questions without any additional fine-tuning. In this paper, we evaluate the effectiveness of $12$ state-of-the-art (SOTA) VLMs in the field of organismal biology using a novel dataset, VLM4Bio, consisting of $469K$ question-answer pairs involving $30K$ images from three groups of organisms: fishes, birds, and butterflies, covering five biologically relevant tasks. We also explore the effects of applying prompting techniques and tests for reasoning hallucination on the performance of VLMs, shedding new light on the capabilities of current SOTA VLMs in answering biologically relevant questions using images.

</details>

---

## 244. Needle In A Multimodal Haystack

- [ ] Needle In A Multimodal Haystack | https://neurips.cc/virtual/2024/poster/97674

- **Link**: https://neurips.cc/virtual/2024/poster/97674

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of multimodal large language models (MLLMs), their evaluation has become increasingly comprehensive. However, understanding long multimodal content, as a foundational ability for real-world applications, remains underexplored. In this work, we present Needle In A Multimodal Haystack (MM-NIAH), the first benchmark specifically designed to systematically evaluate the capability of existing MLLMs to comprehend long multimodal documents. Our benchmark includes three types of evaluation tasks: multimodal retrieval, counting, and reasoning. In each task, the model is required to answer the questions according to different key information scattered throughout the given multimodal document. Evaluating the leading MLLMs on MM-NIAH, we observe that existing models still have significant room for improvement on these tasks, especially on vision-centric evaluation. We hope this work can provide a platform for further research on long multimodal document comprehension and contribute to the advancement of MLLMs. Code and benchmark are released at https://github.com/OpenGVLab/MM-NIAH.

</details>

---

## 245. VHELM: A Holistic Evaluation of Vision Language Models

- [ ] VHELM: A Holistic Evaluation of Vision Language Models | https://neurips.cc/virtual/2024/poster/97677

- **Link**: https://neurips.cc/virtual/2024/poster/97677

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current benchmarks for assessing vision-language models (VLMs) often focus on their perception or problem-solving capabilities and neglect other critical aspects such as fairness, multilinguality, or toxicity. Furthermore, they differ in their evaluation procedures and the scope of the evaluation, making it difficult to compare models. To address these issues, we extend the HELM framework to VLMs to present the Holistic Evaluation of Vision Language Models (VHELM). VHELM aggregates various datasets to cover one or more of the 9 aspects: visual perception , knowledge , reasoning , bias , fairness , multilinguality , robustness , toxicity , and safety . In doing so, we produce a comprehensive, multi-dimensional view of the capabilities of the VLMs across these important factors. In addition, we standardize the standard inference parameters, methods of prompting, and evaluation metrics to enable fair comparisons across models. Our framework is designed to be lightweight and automatic so that evaluation runs are cheap and fast. Our initial run evaluates 22 VLMs on 21 existing datasets to provide a holistic snapshot of the models. We uncover new key findings, such as the fact that efficiency-focused models (e.g., Claude 3 Haiku or Gemini 1.5 Flash) perform significantly worse than their full models (e.g., Claude 3 Opus or Gemini 1.5 Pro) on the bias benchmark but not when evaluated on the other aspects. For transparency, we release the raw model generations and complete results on our website at https://crfm.stanford.edu/helm/vhelm/v2.0.1. VHELM is intended to be a living benchmark, and we hope to continue adding new datasets and models over time.

</details>

---

## 246. VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark

- [ ] VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark | https://neurips.cc/virtual/2024/poster/97679

- **Link**: https://neurips.cc/virtual/2024/poster/97679

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, knowledge editing on large language models (LLMs) has received considerable attention. Compared to this, editing Large Vision-Language Models (LVLMs) faces extra challenges from diverse data modalities and complicated model components, and data for LVLMs editing are limited. The existing LVLM editing benchmark, which comprises three metrics (Reliability, Locality, and Generality), falls short in the quality of synthesized evaluation images and cannot assess whether models apply edited knowledge in relevant content. Therefore, we employ more reliable data collection methods to construct a new Large $\textbf{V}$ision-$\textbf{L}$anguage Model $\textbf{K}$nowledge $\textbf{E}$diting $\textbf{B}$enchmark, $\textbf{VLKEB}$, and extend the Portability metric for more comprehensive evaluation. Leveraging a multi-modal knowledge graph, our image data are bound with knowledge entities. This can be further used to extract entity-related knowledge, which constitutes the base of editing data. We conduct experiments of different editing methods on five LVLMs, and thoroughly analyze how do they impact the models. The results reveal strengths and deficiencies of these methods and hopefully provide insights for future research. The codes and dataset are available at: https://github.com/VLKEB/VLKEB.

</details>

---

## 247. Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows?

- [ ] Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows? | https://neurips.cc/virtual/2024/poster/97692

- **Link**: https://neurips.cc/virtual/2024/poster/97692

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Data science and engineering workflows often span multiple stages, from warehousing to orchestration, using tools like BigQuery, dbt, and Airbyte. As vision language models (VLMs) advance in multimodal understanding and code generation, VLM-based agents could potentially automate these workflows by generating SQL queries, Python code, and GUI operations. This automation can improve the productivity of experts while democratizing access to large-scale data analysis. In this paper, we introduce Spider2-V, the first multimodal agent benchmark focusing on professional data science and engineering workflows, featuring 494 real-world tasks in authentic computer environments and incorporating 20 enterprise-level professional applications. These tasks, derived from real-world use cases, evaluate the ability of a multimodal agent to perform data-related tasks by writing code and managing the GUI in enterprise data software systems. To balance realistic simulation with evaluation simplicity, we devote significant effort to developing automatic configurations for task setup and carefully crafting evaluation metrics for each task. Furthermore, we supplement multimodal agents with comprehensive documents of these enterprise data software systems. Our empirical evaluation reveals that existing state-of-the-art LLM/VLM-based agents do not reliably automate full data workflows (14.0% success). Even with step-by-step guidance, these agents still underperform in tasks that require fine-grained, knowledge-intensive GUI actions (16.2%) and involve remote cloud-hosted workspaces (10.6%). We hope that Spider2-V paves the way for autonomous multimodal agents to transform the automation of data science and engineering workflow. Our code and data are available at https://spider2-v.github.io.

</details>

---

## 248. MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding

- [ ] MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding | https://neurips.cc/virtual/2024/poster/97696

- **Link**: https://neurips.cc/virtual/2024/poster/97696

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The advent of large vision-language models (LVLMs) has spurred research into their applications in multi-modal contexts, particularly in video understanding. Traditional VideoQA benchmarks, despite providing quantitative metrics, often fail to encompass the full spectrum of video content and inadequately assess models' temporal comprehension. To address these limitations, we introduce MMBench-Video, a quantitative benchmark designed to rigorously evaluate LVLMs' proficiency in video understanding. MMBench-Video incorporates lengthy videos from YouTube and employs free-form questions, mirroring practical use cases. The benchmark is meticulously crafted to probe the models' temporal reasoning skills, with all questions human-annotated according to a carefully constructed ability taxonomy.We employ GPT-4 for automated assessment, demonstrating superior accuracy and robustness over earlier LLM-based evaluations. Utilizing MMBench-Video, we have conducted comprehensive evaluations that include both proprietary and open-source LVLMs for images and videos. MMBench-Video stands as a valuable resource for the research community, facilitating improved evaluation of LVLMs and catalyzing progress in the field of video understanding.

</details>

---

## 249. ConvBench: A Multi-Turn Conversation Evaluation Benchmark with Hierarchical Ablation Capability for Large Vision-Language Models

- [ ] ConvBench: A Multi-Turn Conversation Evaluation Benchmark with Hierarchical Ablation Capability for Large Vision-Language Models | https://neurips.cc/virtual/2024/poster/97705

- **Link**: https://neurips.cc/virtual/2024/poster/97705

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-turn visual conversation is an important ability of real-world AI assistants. However,  the related evaluation benchmark is missed. This paper presents ConvBench, a multi-turn conversation benchmark with hierarchical capabilities ablation evaluation for Large Vision-Language Models (LVLMs). ConvBench comprises 577 curated multi-turn conversations, encompassing 215 tasks. These tasks are broad and open-ended, which resemble real-world user behaviors. ConvBench progressively examines the LVLMs' perception, reasoning, and creativity capabilities in each conversation and can decouple these capabilities in evaluations and thus perform reliable error attribution. Besides, considering the diversity of open-ended questions, we introduce an efficient and reliable automatic evaluation framework. Experimental results reveal that ConvBench is a significant challenge for current LVLMs, even for GPT4V, which achieves only a 39.51% score. Besides, we have some insightful findings, such as the weak perception of LVLMs inhibits authentic strengths in reasoning and creation. We believe our design of hierarchical capabilities, decoupling capabilities evaluation, and multi-turn conversation can blaze a new trail in LVLMs evaluation. Code and benchmark are released at https://github.com/shirlyliu64/ConvBench.

</details>

---

## 250. WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks

- [ ] WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks | https://neurips.cc/virtual/2024/poster/97713

- **Link**: https://neurips.cc/virtual/2024/poster/97713

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability of large language models (LLMs) to mimic human-like intelligence has led to a surge in LLM-based autonomous agents. Though recent LLMs seem capable of planning and reasoning given user instructions, their effectiveness in applying these capabilities for autonomous task solving remains underexplored. This is especially true in enterprise settings, where automated agents hold the promise of a high impact. To fill this gap, we propose WorkArena++, a novel benchmark consisting of 682 tasks corresponding to realistic workflows routinely performed by knowledge workers. WorkArena++ is designed to evaluate the planning, problem-solving, logical/arithmetic reasoning, retrieval, and contextual understanding abilities of web agents. Our empirical studies across state-of-the-art LLMs and vision-language models (VLMs), as well as human workers, reveal several challenges for such models to serve as useful assistants in the workplace. In addition to the benchmark, we provide a mechanism to effortlessly generate thousands of ground-truth observation/action traces, which can be used for fine-tuning existing models. Overall, we expect this work to serve as a useful resource to help the community progress towards capable autonomous agents. The benchmark can be found at https://github.com/ServiceNow/WorkArena.

</details>

---

## 251. ConMe: Rethinking Evaluation of Compositional Reasoning for Modern VLMs

- [ ] ConMe: Rethinking Evaluation of Compositional Reasoning for Modern VLMs | https://neurips.cc/virtual/2024/poster/97716

- **Link**: https://neurips.cc/virtual/2024/poster/97716

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Compositional Reasoning (CR) entails grasping the significance of attributes, relations, and word order. Recent Vision-Language Models (VLMs), comprising a visual encoder and a Large Language Model (LLM) decoder, have demonstrated remarkable proficiency in such reasoning tasks. This prompts a crucial question: have VLMs effectively tackled the CR challenge? We conjecture that existing CR benchmarks may not adequately push the boundaries of modern VLMs due to the reliance on an LLM only negative text generation pipeline. Consequently, the negatives produced either appear as outliers from the natural language distribution learned by VLMs' LLM decoders or as improbable within the corresponding image context. To address these limitations, we introduce ConMe\footnote{ConMe is an abbreviation for Confuse Me.} -- a compositional reasoning benchmark and a novel data generation pipeline leveraging VLMs to produce `hard CR Q&A'. Through a new concept of VLMs conversing with each other to collaboratively expose their weaknesses, our pipeline autonomously generates, evaluates, and selects challenging compositional reasoning questions, establishing a robust CR benchmark, also subsequently validated manually. Our benchmark provokes a noteworthy, up to 33%, decrease in CR performance compared to preceding benchmarks, reinstating the CR challenge even for state-of-the-art VLMs.

</details>

---

## 252. Multi-modal Situated Reasoning in 3D Scenes

- [ ] Multi-modal Situated Reasoning in 3D Scenes | https://neurips.cc/virtual/2024/poster/97727

- **Link**: https://neurips.cc/virtual/2024/poster/97727

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Situation awareness is essential for understanding and reasoning about 3D scenes in embodied AI agents. However, existing datasets and benchmarks for situated understanding suffer from severe limitations in data modality, scope, diversity, and scale. To address these limitations, we propose Multi-modal Situated Question Answering (MSQA), a large-scale multi-modal situated reasoning dataset, scalably collected leveraging 3D scene graphs and vision-language models (VLMs) across a diverse range of real-world 3D scenes. MSQA includes 251K situated questionanswering pairs across 9 distinct question categories, covering complex scenarios and object modalities within 3D scenes. We introduce a novel interleaved multimodal input setting in our benchmark to provide both texts, images, and point clouds for situation and question description, aiming to resolve ambiguity in describing situations with single-modality inputs (e.g., texts). Additionally, we devise the Multi-modal Next-step Navigation (MSNN) benchmark to evaluate models’ grounding of actions and transitions between situations. Comprehensive evaluations on reasoning and navigation tasks highlight the limitations of existing vision-language models and underscore the importance of handling multi-modal interleaved inputs and situation modeling. Experiments on data scaling and crossdomain transfer further demonstrate the effectiveness of leveraging MSQA as a pre-training dataset for developing more powerful situated reasoning models, contributing to advancements in 3D scene understanding for embodied AI.

</details>

---

## 253. ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty

- [ ] ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty | https://neurips.cc/virtual/2024/poster/97734

- **Link**: https://neurips.cc/virtual/2024/poster/97734

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Compositionality is a critical capability in Text-to-Image (T2I) models, as it reflects their ability to understand and combine multiple concepts from text descriptions. Existing evaluations of compositional capability rely heavily on human-designed text prompts or fixed templates, limiting their diversity and complexity, and yielding low discriminative power. We propose ConceptMix, a scalable, controllable, and customizable benchmark which automatically evaluates compositional generation ability of T2I models. This is done in two stages. First, ConceptMix generates the text prompts: concretely, using categories of visual concepts (e.g., objects, colors, shapes, spatial relationships), it randomly samples an object and k-tuples of visual concepts, then uses GPT-4o to generate text prompts for image generation based on these sampled concepts. Second, ConceptMix evaluates the images generated in response to these prompts: concretely, it checks how many of the k concepts actually appeared in the image by generating one question per visual concept and using a strong VLM to answer them. Through administering ConceptMix to a diverse set of T2I models (proprietary as well as open ones) using increasing values of k, we show that our ConceptMix has higher discrimination power than earlier benchmarks. Specifically, ConceptMix reveals that the performance of several models, especially open models, drops dramatically with increased k. Importantly, it also provides insight into the lack of prompt diversity in widely-used training datasets. Additionally, we conduct extensive human studies to validate the design of ConceptMix and compare our automatic grading with human judgement. We hope it will guide future T2I model development.

</details>

---

## 254. UKnow: A Unified Knowledge Protocol with Multimodal Knowledge Graph Datasets for Reasoning and Vision-Language Pre-Training

- [ ] UKnow: A Unified Knowledge Protocol with Multimodal Knowledge Graph Datasets for Reasoning and Vision-Language Pre-Training | https://neurips.cc/virtual/2024/poster/97740

- **Link**: https://neurips.cc/virtual/2024/poster/97740

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This work presents a unified knowledge protocol, called UKnow, which facilitates knowledge-based studies from the perspective of data. Particularly focusing on visual and linguistic modalities, we categorize data knowledge into five unit types, namely, in-image, in-text, cross-image, cross-text, and image-text, and set up an efficient pipeline to help construct the multimodal knowledge graph from any data collection. Thanks to the logical information naturally contained in knowledge graph, organizing datasets under UKnow format opens up more possibilities of data usage compared to the commonly used image-text pairs. Following UKnow protocol, we collect, from public international news, a large-scale multimodal knowledge graph dataset that consists of 1,388,568 nodes (with 571,791 vision-related ones) and 3,673,817 triplets. The dataset is also annotated with rich event tags, including 11 coarse labels and 9,185 fine labels. Experiments on four benchmarks demonstrate the potential of UKnow in supporting common-sense reasoning and boosting vision-language pre-training with a single dataset, benefiting from its unified form of knowledge organization. Code, dataset, and models will be made publicly available. See Appendix to download the dataset.

</details>

---

## 255. E.T. Bench: Towards Open-Ended Event-Level Video-Language Understanding

- [ ] E.T. Bench: Towards Open-Ended Event-Level Video-Language Understanding | https://neurips.cc/virtual/2024/poster/97748

- **Link**: https://neurips.cc/virtual/2024/poster/97748

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Video Large Language Models (Video-LLMs) have demonstrated their great potential in general-purpose video understanding. To verify the significance of these models, a number of benchmarks have been proposed to diagnose their capabilities in different scenarios. However, existing benchmarks merely evaluate models through video-level question-answering, lacking fine-grained event-level assessment and task diversity. To fill this gap, we introduce E.T. Bench (Event-Level & Time-Sensitive Video Understanding Benchmark), a large-scale and high-quality benchmark for open-ended event-level video understanding. Categorized within a 3-level task taxonomy, E.T. Bench encompasses 7.3K samples under 12 tasks with 7K videos (251.4h total length) under 8 domains, providing comprehensive evaluations. We extensively evaluated 8 Image-LLMs and 12 Video-LLMs on our benchmark, and the results reveal that state-of-the-art models for coarse-level (video-level) understanding struggle to solve our fine-grained tasks, e.g., grounding event-of-interests within videos, largely due to the short video context length, improper time representations, and lack of multi-event training data. Focusing on these issues, we further propose a strong baseline model, E.T. Chat, together with an instruction-tuning dataset E.T. Instruct 164K tailored for fine-grained event-level understanding. Our simple but effective solution demonstrates superior performance in multiple scenarios.

</details>

---

## 256. Enhancing vision-language models for medical imaging: bridging the 3D gap with innovative slice selection

- [ ] Enhancing vision-language models for medical imaging: bridging the 3D gap with innovative slice selection | https://neurips.cc/virtual/2024/poster/97755

- **Link**: https://neurips.cc/virtual/2024/poster/97755

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent approaches to vision-language tasks are built on the remarkable capabilities of large vision-language models (VLMs). These models excel in zero-shot and few-shot learning, enabling them to learn new tasks without parameter updates. However, their primary challenge lies in their design, which primarily accommodates 2D input, thus limiting their effectiveness for medical images, particularly radiological images like MRI and CT, which are typically 3D. To bridge the gap between state-of-the-art 2D VLMs and 3D medical image data, we developed an innovative, one-pass, unsupervised representative slice selection method called Vote-MI, which selects representative 2D slices from 3D medical imaging. To evaluate the effectiveness of vote-MI when implemented with VLMs, we introduce BrainMD, a robust, multimodal dataset comprising 2,453 annotated 3D MRI brain scans with corresponding textual radiology reports and electronic health records. Based on BrainMD, we further develop two benchmarks, BrainMD-select (including the most representative 2D slice of 3D image) and BrainBench (including various vision-language downstream tasks). Extensive experiments on the BrainMD dataset and its two corresponding benchmarks demonstrate that our representative selection method significantly improves performance in zero-shot and few-shot learning tasks. On average, Vote-MI achieves a 14.6\% and 16.6\% absolute gain for zero-shot and few-shot learning, respectively, compared to randomly selecting examples. Our studies represent a significant step toward integrating AI in medical imaging to enhance patient care and facilitate medical research. We hope this work will serve as a foundation for data selection as vision-language models are increasingly applied to new tasks.

</details>

---

## 257. GMAI-MMBench: A Comprehensive Multimodal Evaluation Benchmark Towards General Medical AI

- [ ] GMAI-MMBench: A Comprehensive Multimodal Evaluation Benchmark Towards General Medical AI | https://neurips.cc/virtual/2024/poster/97754

- **Link**: https://neurips.cc/virtual/2024/poster/97754

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) are capable of handling diverse data types such as imaging, text, and physiological signals, and can be applied in various fields. In the medical field, LVLMs have a high potential to offer substantial assistance for diagnosis and treatment. Before that, it is crucial to develop benchmarks to evaluate LVLMs' effectiveness in various medical applications. Current benchmarks are often built upon specific academic literature, mainly focusing on a single domain, and lacking varying perceptual granularities. Thus, they face specific challenges, including limited clinical relevance, incomplete evaluations, and insufficient guidance for interactive LVLMs. To address these limitations, we developed the GMAI-MMBench, the most comprehensive general medical AI benchmark with well-categorized data structure and multi-perceptual granularity to date. It is constructed from 284 datasets across 38 medical image modalities, 18 clinical-related tasks, 18 departments, and 4 perceptual granularities in a Visual Question Answering (VQA) format. Additionally, we implemented a lexical tree structure that allows users to customize evaluation tasks, accommodating various assessment needs and substantially supporting medical AI research and applications. We evaluated 50 LVLMs, and the results show that even the advanced GPT-4o only achieves an accuracy of 53.96\%, indicating significant room for improvement. Moreover, we identified five key insufficiencies in current cutting-edge LVLMs that need to be addressed to advance the development of better medical applications. We believe that GMAI-MMBench will stimulate the community to build the next generation of LVLMs toward GMAI.

</details>

---

## 258. Streaming Detection of Queried Event Start

- [ ] Streaming Detection of Queried Event Start | https://neurips.cc/virtual/2024/poster/97778

- **Link**: https://neurips.cc/virtual/2024/poster/97778

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Robotics, autonomous driving, augmented reality, and many embodied computer vision applications must quickly react to user-defined events unfolding in real time. We address this setting by proposing a novel task for multimodal video understanding---Streaming Detection of Queried Event Start (SDQES).The goal of SDQES is to identify the beginning of a complex event as described by a natural language query, with high accuracy and low latency. We introduce a new benchmark based on  the Ego4D dataset, as well as new task-specific metrics to study streaming multimodal detection of diverse events in an egocentric video setting.Inspired by parameter-efficient fine-tuning methods in NLP and for video tasks, we propose adapter-based baselines that enable image-to-video transfer learning, allowing for efficient online video modeling.We evaluate three vision-language backbones and three adapter architectures on both short-clip and untrimmed video settings.

</details>

---

## 259. WikiDO: A New Benchmark Evaluating Cross-Modal Retrieval for Vision-Language Models

- [ ] WikiDO: A New Benchmark Evaluating Cross-Modal Retrieval for Vision-Language Models | https://neurips.cc/virtual/2024/poster/97785

- **Link**: https://neurips.cc/virtual/2024/poster/97785

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Cross-modal (image-to-text and text-to-image) retrieval is an established task used in evaluation benchmarks to test the performance of vision-language models (VLMs). Several state-of-the-art VLMs (e.g. CLIP, BLIP-2) have achieved near-perfect performance on widely-used image-text retrieval benchmarks such as MSCOCO-Test-5K and Flickr30K-Test-1K. As a measure of out-of-distribution (OOD) generalization, prior works rely on zero-shot performance evaluated on one dataset (Flickr) using a VLM finetuned on another one (MSCOCO). We argue that such comparisons are insufficient to assess the OOD generalization capability of models due to high visual and linguistic similarity between the evaluation and finetuning datasets. To address this gap, we introduce WikiDO (drawn from Wikipedia Diversity Observatory), a novel cross-modal retrieval benchmark to assess the OOD generalization capabilities of pretrained VLMs. This consists of newly scraped 380K image-text pairs from Wikipedia with domain labels, a carefully curated, human-verified a)in-distribution (ID) test set (3K) and b) OOD test set (3K). The image-text pairs are very diverse in topics and geographical locations. We evaluate different VLMs of varying capacity on the \wikido benchmark; BLIP-2 achieves zero-shot performance of $R@1\approx66\%$ on the OOD test set, compared to $\approx$ $81\%$ on COCO and $\approx95\%$ on Flickr. When fine-tuned on WikiDO, the $R@1$ improvement is at most $\approx5\%$ on OOD instances compared to $\approx12\%$ on ID instances. We probe the VLMs with varying finetuning objectives and datasets of varying sizes to identify what aids OOD generalization the most. Our results confirm that WikiDO offers a strong cross-modal benchmark for current VLMs in specifically evaluating for OOD generalization. Our benchmark is hosted as a competition at https://kaggle.com/competitions/wikido24 with public access to dataset and code.

</details>

---

## 260. CoIN: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models

- [ ] CoIN: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models | https://neurips.cc/virtual/2024/poster/97786

- **Link**: https://neurips.cc/virtual/2024/poster/97786

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuning demonstrates impressive performance in adapting Multimodal Large Language Models (MLLMs) to follow task instructions and improve generalization ability.  By extending tuning across diverse tasks, MLLMs can further enhance their understanding of world knowledge and instruction intent. However, continual instruction tuning has been largely overlooked and there are no public benchmarks available. In this paper, we present CoIN, a comprehensive benchmark tailored for assessing the behavior of existing MLLMs under continual instruction tuning. CoIN comprises 10 meticulously crafted datasets spanning 8 tasks, ensuring diversity and serving as a robust evaluation framework to assess crucial aspects of continual instruction tuning, such as task order, instruction diversity and volume. Additionally, apart from traditional evaluation, we design another LLM-based metric to assess the knowledge preserved within MLLMs for reasoning. Following an in-depth evaluation of several MLLMs, we demonstrate that they still suffer catastrophic forgetting, and the failure in instruction alignment assumes the main responsibility, instead of reasoning knowledge forgetting.  To this end, we introduce MoELoRA which is effective in retaining the previous instruction alignment.

</details>

---

## 261. ShareGPT4Video: Improving Video Understanding and Generation with Better Captions

- [ ] ShareGPT4Video: Improving Video Understanding and Generation with Better Captions | https://neurips.cc/virtual/2024/poster/97789

- **Link**: https://neurips.cc/virtual/2024/poster/97789

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present the ShareGPT4Video series, aiming to facilitate the video understanding of large video-language models (LVLMs) and the video generation of text-to-video models (T2VMs) via dense and precise captions. The series comprises: 1) ShareGPT4Video, 40K GPT4V annotated dense captions of videos with various lengths and sources, developed through carefully designed data filtering and annotating strategy. 2) ShareCaptioner-Video, an efficient and capable captioning model for arbitrary videos, with 4.8M high-quality aesthetic videos annotated by it. 3) ShareGPT4Video-8B, a simple yet superb LVLM that reached SOTA performance on three advancing video benchmarks. To achieve this, taking aside the non-scalable costly human annotators, we find using GPT4V to caption video with a naive multi-frame or frame-concatenation input strategy leads to less detailed and sometimes temporal-confused results. We argue the challenge of designing a high-quality video captioning strategy lies in three aspects: 1) Inter-frame precise temporal change understanding. 2) Intra-frame detailed content description. 3) Frame-number scalability for arbitrary-length videos. To this end, we meticulously designed a differential video captioning strategy, which is stable, scalable, and efficient for generating captions for videos with arbitrary resolution, aspect ratios, and length. Based on it, we construct ShareGPT4Video, which contains 40K high-quality videos spanning a wide range of categories, and the resulting captions encompass rich world knowledge, object attributes, camera movements, and crucially, detailed and precise temporal descriptions of events. Based on ShareGPT4Video, we further develop ShareCaptioner-Video, a superior captioner capable of efficiently generating high-quality captions for arbitrary videos. We annotated 4.8M aesthetically appealing videos by it and verified their effectiveness on a 10-second text2video generation task. For video understanding, we verified the effectiveness of ShareGPT4Video on several current LVLM architectures and presented our superb new LVLM ShareGPT4Video-8B. All the models, strategies, and annotations will be open-sourced and we hope this project can serve as a pivotal resource for advancing both the LVLMs and T2VMs community.

</details>

---

## 262. HourVideo: 1-Hour Video-Language Understanding

- [ ] HourVideo: 1-Hour Video-Language Understanding | https://neurips.cc/virtual/2024/poster/97793

- **Link**: https://neurips.cc/virtual/2024/poster/97793

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present HourVideo , a benchmark dataset for hour-long video-language understanding. Our dataset consists of a novel task suite comprising summarization, perception ( recall , tracking ), visual reasoning ( spatial , temporal , predictive , causal , counterfactual ), and navigation ( room-to-room , object retrieval ) tasks. HourVideo includes 500 manually curated egocentric videos from the Ego4D dataset, spanning durations of 20 to 120 minutes, and features 12,976 high-quality, five-way multiple-choice questions . Benchmarking results reveal that multimodal models, including GPT-4 and LLaVA-NeXT, achieve marginal improvements over random chance. In stark contrast, human experts significantly outperform the state-of-the-art long-context multimodal model, Gemini Pro 1.5 (85.0\% vs. 37.3\%), highlighting a substantial gap in multimodal capabilities. Our benchmark, evaluation toolkit, prompts, and documentation are available at https://hourvideo.stanford.edu.

</details>

---

## 263. CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark

- [ ] CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark | https://neurips.cc/virtual/2024/poster/97798

- **Link**: https://neurips.cc/virtual/2024/poster/97798

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual Question Answering~(VQA) is an important task in multimodal AI, which requires models to understand and reason on knowledge present in visual and textual data. However, most of the current VQA datasets and models are primarily focused on English and a few major world languages, with images that are Western-centric. While recent efforts have tried to increase the number of languages covered on VQA datasets, they still lack diversity in low-resource languages. More importantly, some datasets extend the text to other languages, either via translation or some other approaches, but usually keep the same images, resulting in narrow cultural representation. To address these limitations, we create CVQA, a new Culturally-diverse Multilingual Visual Question Answering benchmark dataset, designed to cover a rich set of languages and regions, where we engage native speakers and cultural experts in the data collection process. CVQA includes culturally-driven images and questions from across 28 countries in four continents, covering 26 languages with 11 scripts, providing a total of 9k questions. We benchmark several Multimodal Large Language Models (MLLMs) on CVQA, and we show that the dataset is challenging for the current state-of-the-art models. This benchmark will serve as a probing evaluation suite for assessing the cultural bias of multimodal models and hopefully encourage more research efforts towards increasing cultural awareness and linguistic diversity in this field.

</details>

---

## 264. NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples

- [ ] NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples | https://neurips.cc/virtual/2024/poster/97799

- **Link**: https://neurips.cc/virtual/2024/poster/97799

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have made significant progress in recent visual-question-answering (VQA) benchmarks that evaluate complex visio-linguistic reasoning. However, are these models truly effective? In this work, we show that VLMs still struggle with natural images and questions that humans can easily answer, which we term $\textbf{natural adversarial samples}$. We also find it surprisingly easy to generate these VQA samples from natural image-text corpora using off-the-shelf models like CLIP and ChatGPT. We propose a semi-automated approach to collect a new benchmark, ${\bf NaturalBench}$, for reliably evaluating VLMs with 10,000 human-verified VQA samples. Crucially, we adopt a $\textbf{vision-centric}$ design by pairing each question with two images that yield different answers, preventing ``blind'' solutions from answering without using the images. This makes NaturalBench more challenging than previous benchmarks that can largely be solved with language priors like commonsense knowledge. We evaluate ${\bf 53}$ state-of-the-art VLMs on NaturalBench, showing that models like BLIP-3, LLaVA-OneVision, Cambrian-1, InternLM-XC2, Llama3.2-Vision, Molmo, Qwen2-VL, and even the (closed-source) GPT-4o lag 50%-70% behind human performance (which is above 90%). We analyze why NaturalBench is hard from two angles: (1) ${\bf Compositionality:}$ Solving NaturalBench requires diverse visio-linguistic skills, including understanding attribute bindings, object relationships, and advanced reasoning like logic and counting. To this end, unlike prior work that uses a single tag per sample, we tag each NaturalBench sample with 1 to 8 skill tags for fine-grained evaluation. (2) ${\bf Biases: }$ NaturalBench exposes severe biases in VLMs, as models often choose the same answer regardless of the image. We show that debiasing can be crucial for VLM performance. Lastly, we apply our benchmark curation method to diverse data sources, including long captions (over 100 words) and non-English languages like Chinese and Hindi, highlighting its potential for dynamic evaluations of VLMs.

</details>

---

## 265. FIRE: A Dataset for Feedback Integration and Refinement Evaluation of Multimodal Models

- [ ] FIRE: A Dataset for Feedback Integration and Refinement Evaluation of Multimodal Models | https://neurips.cc/virtual/2024/poster/97805

- **Link**: https://neurips.cc/virtual/2024/poster/97805

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) have achieved impressive progress in diverse applications, becoming a prevalent research direction. In this paper, we build FIRE, a feedback-refinement dataset, consisting of 1.1M multi-turn conversations that are derived from 27 source datasets, empowering VLMs to spontaneously refine their responses based on user feedback across diverse tasks. To scale up the data collection, FIRE is collected in two components: FIRE-100K and FIRE-1M, where FIRE-100K is generated by GPT-4V, and FIRE-1M is freely generated via models trained on FIRE-100K. Then, we build FIRE-Bench, a benchmark to comprehensively evaluate the feedback-refining capability of VLMs, which contains 11K feedback-refinement conversations as the test data, two evaluation settings, and a model to provide feedback for VLMs. We develop the FIRE-LLaVA model by fine-tuning LLaVA on FIRE-100K and FIRE-1M, which shows remarkable feedback-refining capability on FIRE-Bench and outperforms untrained VLMs by 50%, making more efficient user-agent interactions and underscoring the significance of the FIRE dataset.

</details>

---

## 266. SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations

- [ ] SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations | https://neurips.cc/virtual/2024/poster/97833

- **Link**: https://neurips.cc/virtual/2024/poster/97833

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite their remarkable successes, state-of-the-art large language models (LLMs), including vision-and-language models (VLMs) and unimodal language models (ULMs), fail to understand precise semantics. For example, semantically equivalent sentences expressed using different lexical compositions elicit diverging representations. The degree of this divergence and its impact on encoded semantics is not very well understood. In this paper, we introduce the SUGARCREPE++ dataset to analyze the sensitivity of VLMs and ULMs to lexical and semantic alterations. Each sample in SUGARCREPE++ dataset consists of an image and a corresponding triplet of captions: a pair of semantically equivalent but lexically different positive captions and one hard negative caption. This poses a 3-way semantic (in)equivalence problem to the language models. We comprehensively evaluate VLMs and ULMs that differ in architecture, pre-training objectives and datasets to benchmark the performance of SUGARCREPE++ dataset. Experimental results highlight the difficulties of VLMs in distinguishing between lexical and semantic variations, particularly to object attributes and spatial relations. Although VLMs with larger pre-training datasets, model sizes, and multiple pre-training objectives achieve better performance on SUGARCREPE++, there is a significant opportunity for improvement. We demonstrate that models excelling on compositionality datasets may not perform equally well on SUGARCREPE++. This indicates that compositionality alone might not be sufficient to fully understand semantic and lexical alterations. Given the importance of the property that the SUGARCREPE++ dataset targets, it serves as a new challenge to the vision-and-language community. Data and code is available at https://github.com/Sri-Harsha/scpp.

</details>

---

## 267. Image2Struct: Benchmarking Structure Extraction for Vision-Language Models

- [ ] Image2Struct: Benchmarking Structure Extraction for Vision-Language Models | https://neurips.cc/virtual/2024/poster/97829

- **Link**: https://neurips.cc/virtual/2024/poster/97829

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Image2Struct, a benchmark to evaluate vision-language models (VLMs) on extracting structure from images.Our benchmark 1) captures real-world use cases, 2) is fully automatic and does not require human judgment, and 3) is based on a renewable stream of fresh data.In Image2Struct, VLMs are prompted to generate the underlying structure (e.g., LaTeX code or HTML) from an input image (e.g., webpage screenshot).The structure is then rendered to produce an output image (e.g., rendered webpage), which is compared against the input image to produce a similarity score.This round-trip evaluation allows us to quantitatively evaluate VLMs on tasks with multiple valid structures.We create a pipeline that downloads fresh data from active online communities upon execution and evaluates the VLMs without human intervention.We introduce three domains (Webpages, LaTeX, and Musical Scores) and use five image metrics (pixel similarity, cosine similarity between the Inception vectors, learned perceptual image patch similarity, structural similarity index measure, and earth mover similarity) that allow efficient and automatic comparison between pairs of images. We evaluate Image2Struct on 14 prominent VLMs and find that scores vary widely, indicating that Image2Struct can differentiate between the performances of different VLMs.Additionally, the best score varies considerably across domains (e.g., 0.402 on sheet music vs. 0.830 on LaTeX equations), indicating that Image2Struct contains tasks of varying difficulty.For transparency, we release the full results at  https://crfm.stanford.edu/helm/image2struct/v1.0.1/.

</details>

---

## 268. ClevrSkills: Compositional Language And Visual Reasoning in Robotics

- [ ] ClevrSkills: Compositional Language And Visual Reasoning in Robotics | https://neurips.cc/virtual/2024/poster/97843

- **Link**: https://neurips.cc/virtual/2024/poster/97843

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Robotics tasks are highly compositional by nature. For example, to perform a high-level task like cleaning the table a robot must employ low-level capabilities of moving the effectors to the objects on the table, pick them up and then move them off the table one-by-one, while re-evaluating the consequently dynamic scenario in the process. Given that large vision language models (VLMs) have shown progress on many tasks that require high level, human-like reasoning, we ask the question: if the models are taught the requisite low-level capabilities, can they compose them in novel ways to achieve interesting high-level tasks like cleaning the table without having to be explicitly taught so? To this end, we present ClevrSkills - a benchmark suite for compositional reasoning in robotics. ClevrSkills is an environment suite developed on top of the ManiSkill2 simulator and an accompanying dataset. The dataset contains trajectories generated on a range of robotics tasks with language and visual annotations as well as multi-modal prompts as task specification. The suite includes a curriculum of tasks with three levels of compositional understanding, starting with simple tasks requiring basic motor skills. We benchmark multiple different VLM baselines on ClevrSkills and show that even after being pre-trained on large numbers of tasks, these models fail on compositional reasoning in robotics tasks.

</details>

---

## 269. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models

- [ ] MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models | https://neurips.cc/virtual/2024/poster/97845

- **Link**: https://neurips.cc/virtual/2024/poster/97845

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust , the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness , safety , robustness , fairness , and privacy . Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/ .

</details>

---

## 270. ReXTime: A Benchmark Suite for Reasoning-Across-Time in Videos

- [ ] ReXTime: A Benchmark Suite for Reasoning-Across-Time in Videos | https://neurips.cc/virtual/2024/poster/97852

- **Link**: https://neurips.cc/virtual/2024/poster/97852

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce ReXTime, a benchmark designed to rigorously test AI models' ability to perform temporal reasoning within video events.Specifically, ReXTime focuses on reasoning across time, i.e. human-like understanding when the question and its corresponding answer occur in different video segments. This form of reasoning, requiring advanced understanding of cause-and-effect relationships across video segments, poses significant challenges to even the frontier multimodal large language models. To facilitate this evaluation, we develop an automated pipeline for generating temporal reasoning question-answer pairs, significantly reducing the need for labor-intensive manual annotations. Our benchmark includes 921 carefully vetted validation samples and 2,143 test samples, each manually curated for accuracy and relevance. Evaluation results show that while frontier large language models outperform academic models, they still lag behind human performance by a significant 14.3\% accuracy gap. Additionally, our pipeline creates a training dataset of 9,695 machine generated samples without manual effort, which empirical studies suggest can enhance the across-time reasoning via fine-tuning.

</details>

---

## 271. VastTrack: Vast Category Visual Object Tracking

- [ ] VastTrack: Vast Category Visual Object Tracking | https://neurips.cc/virtual/2024/poster/97849

- **Link**: https://neurips.cc/virtual/2024/poster/97849

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose a novel benchmark, named VastTrack, aiming to facilitate the development of general visual tracking via encompassing abundant classes and videos. VastTrack consists of a few attractive properties: (1) Vast Object Category. In particular, it covers targets from 2,115 categories, significantly surpassing object classes of existing popular benchmarks (e.g., GOT-10k with 563 classes and LaSOT with 70 categories). Through providing such vast object classes, we expect to learn more general object tracking. (2) Larger scale. Compared with current benchmarks, VastTrack provides 50,610 videos with 4.2 million frames, which makes it to date the largest dataset in term of the number of videos, and hence could benefit training even more powerful visual trackers in the deep learning era. (3) Rich Annotation. Besides conventional bounding box annotations, VastTrack also provides linguistic descriptions with more than 50K sentences for the videos. Such rich annotations of VastTrack enable the development of both vision-only and vision-language tracking. In order to ensure precise annotation, each frame in the videos is manually labeled with multi-stage of careful inspections and refinements. To understand performance of existing trackers and to provide baselines for future comparison, we extensively evaluate 25 representative trackers. The results, not surprisingly, display significant drops compared to those on current datasets due to lack of abundant categories and videos from diverse scenarios for training, and more efforts are urgently required to improve general visual tracking. Our VastTrack, the toolkit, and evaluation results are publicly available at https://github.com/HengLan/VastTrack.

</details>

---

## 272. LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding

- [ ] LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding | https://neurips.cc/virtual/2024/poster/97862

- **Link**: https://neurips.cc/virtual/2024/poster/97862

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal models (LMMs) are processing increasingly longer and richer inputs. Albeit the progress, few public benchmark is available to measure such development. To mitigate this gap, we introduce LongVideoBench, a question-answering benchmark that features video-language interleaved inputs up to an hour long. Our benchmark includes 3,763 varying-length web-collected videos with their subtitles across diverse themes, designed to comprehensively evaluate LMMs on long-term multimodal understanding. To achieve this, we interpret the primary challenge as to accurately retrieve and reason over detailed multimodal information from long inputs. As such, we formulate a novel video question-answering task termed referring reasoning. Specifically, as part of the question, it contains a referring query that references related video contexts, called referred context. The model is then required to reason over relevant video details from the referred context. Following the paradigm of referring reasoning, we curate 6,678 human-annotated multiple-choice questions in 17 fine-grained categories, establishing one of the most comprehensive benchmarks for long-form video understanding. Evaluations suggest that the LongVideoBench presents significant challenges even for the most advanced proprietary models (e.g. GPT-4o, Gemini-1.5-Pro), while their open-source counterparts show an even larger performance gap. In addition, our results indicate that model performance on the benchmark improves only when they are capable of processing more frames, positioning LongVideoBench as a valuable benchmark for evaluating future-generation long-context LMMs.

</details>

---

## 273. Revisiting Few-Shot Object Detection with Vision-Language Models

- [ ] Revisiting Few-Shot Object Detection with Vision-Language Models | https://neurips.cc/virtual/2024/poster/97860

- **Link**: https://neurips.cc/virtual/2024/poster/97860

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The era of vision-language models (VLMs) trained on web-scale datasets challenges conventional formulations of “open-world" perception. In this work, we revisit the task of few-shot object detection (FSOD) in the context of recent foundational VLMs. First, we point out that zero-shot predictions from VLMs such as GroundingDINO significantly outperform state-of-the-art few-shot detectors (48 vs. 33 AP) on COCO. Despite their strong zero-shot performance, such foundation models may still be sub-optimal. For example, trucks on the web may be defined differently from trucks for a target applications such as autonomous vehicle perception. We argue that the task of few-shot recognition can be reformulated as aligning foundation models to target concepts using a few examples. Interestingly, such examples can be multi-modal, using both text and visual cues, mimicking instructions that are often given to human annotators when defining a target concept of interest. Concretely, we propose Foundational FSOD, a new benchmark protocol that evaluates detectors pre-trained on any external data and fine-tuned on multi-modal (text and visual) K-shot examples per target class. We repurpose nuImages for Foundational FSOD, benchmark several popular open-source VLMs, and provide an empirical analysis of state-of-the-art methods. Lastly, we discuss our recent CVPR 2024 Foundational FSOD competition and share insights from the community. Notably, the winning team significantly outperforms our baseline by 23.3 mAP!

</details>

---

## 274. T2Vs Meet VLMs: A Scalable Multimodal Dataset for Visual Harmfulness Recognition

- [ ] T2Vs Meet VLMs: A Scalable Multimodal Dataset for Visual Harmfulness Recognition | https://neurips.cc/virtual/2024/poster/97879

- **Link**: https://neurips.cc/virtual/2024/poster/97879

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While widespread access to the Internet and the rapid advancement of generative models boost people's creativity and productivity, the risk of encountering inappropriate or harmful content also increases. To address the aforementioned issue, researchers managed to incorporate several harmful contents datasets with machine learning methods to detect harmful concepts. However, existing harmful datasets are curated by the presence of a narrow range of harmful objects, and only cover real harmful content sources. This restricts the generalizability of methods based on such datasets and leads to the potential misjudgment in certain cases. Therefore, we propose a comprehensive and extensive harmful dataset, VHD11K , consisting of 10,000 images and 1,000 videos, crawled from the Internet and generated by 4 generative models, across a total of 10 harmful categories covering a full spectrum of harmful concepts with non-trival definition. We also propose a novel annotation framework by formulating the annotation process as a multi-agent Visual Question Answering (VQA) task, having 3 different VLMs "debate" about whether the given image/video is harmful, and incorporating the in-context learning strategy in the debating process. Therefore, we can ensure that the VLMs consider the context of the given image/video and both sides of the arguments thoroughly before making decisions, further reducing the likelihood of misjudgments in edge cases. Evaluation and experimental results demonstrate that (1) the great alignment between the annotation from our novel annotation framework and those from human, ensuring the reliability of VHD11K;(2) our full-spectrum harmful dataset successfully identifies the inability of existing harmful content detection methods to detect extensive harmful contents and improves the performance of existing harmfulness recognition methods;(3) our dataset outperforms the baseline dataset, SMID, as evidenced by the superior improvement in harmfulness recognition methods.The entire dataset is publicly available: https://huggingface.co/datasets/denny3388/VHD11K

</details>

---

## 275. GenAI Arena: An Open Evaluation Platform for Generative Models

- [ ] GenAI Arena: An Open Evaluation Platform for Generative Models | https://neurips.cc/virtual/2024/poster/97878

- **Link**: https://neurips.cc/virtual/2024/poster/97878

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generative AI has made remarkable strides to revolutionize fields such as image and video generation. These advancements are driven by innovative algorithms, architecture, and data. However, the rapid proliferation of generative models has highlighted a critical gap: the absence of trustworthy evaluation metrics. Current automatic assessments such as FID, CLIP, FVD, etc often fail to capture the nuanced quality and user satisfaction associated with generative outputs. This paper proposes an open platform GenAI-Arena to evaluate different image and video generative models, where users can actively participate in evaluating these models. By leveraging collective user feedback and votes, GenAI-Arena aims to provide a more democratic and accurate measure of model performance. It covers three tasks of text-to-image generation, text-to-video generation, and image editing respectively. Currently, we cover a total of 35 open-source generative models. GenAI-Arena has been operating for seven months, amassing over 9000 votes from the community. We describe our platform, analyze the data, and explain the statistical methods for ranking the models. To further promote the research in building model-based evaluation metrics, we release a cleaned version of our preference data for the three tasks, namely GenAI-Bench. We prompt the existing multi-modal models like Gemini, and GPT-4o to mimic human voting. We compute the accuracy by comparing the model voting with the human voting to understand their judging abilities. Our results show existing multimodal models are still lagging in assessing the generated visual content, even the best model GPT-4o only achieves an average accuracy of $49.19\%$ across the three generative tasks. Open-source MLLMs perform even worse due to the lack of instruction-following and reasoning ability in complex vision scenarios.

</details>

---

## 276. Reproducibility study of “LICO: Explainable Models with Language-Image Consistency"

- [ ] Reproducibility study of “LICO: Explainable Models with Language-Image Consistency" | https://neurips.cc/virtual/2024/poster/99344

- **Link**: https://neurips.cc/virtual/2024/poster/99344

- **Conference**: NeurIPS

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The growing reproducibility crisis in machine learning has brought forward a need for careful examination of research findings. This paper investigates the claims made by Lei et al. (2023) regarding their proposed method, LICO, for enhancing post-hoc interpretability techniques and improving image classification performance. LICO leverages natural language supervision from a vision-language model to enrich feature representations and guide the learning process. We conduct a comprehensive reproducibility study, employing (Wide) ResNets and established interpretability methods like Grad-CAM and RISE. We were mostly unable to reproduce the authors' results. In particular, we did not find that LICO consistently led to improved classification performance or improvements in quantitative and qualitative measures of interpretability. Thus, our findings highlight the importance of rigorous evaluation and transparent reporting in interpretability research.

</details>

---

