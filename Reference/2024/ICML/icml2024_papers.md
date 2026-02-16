# ICML 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_icml2024_papers.csv

## 1. Exploring Intrinsic Dimension for Vision-Language Model Pruning

- [ ] Exploring Intrinsic Dimension for Vision-Language Model Pruning | https://icml.cc/virtual/2024/poster/32685

- **Link**: https://icml.cc/virtual/2024/poster/32685

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The intrinsic dimension (ID) represents the minimum dimension needed to describe data on a lower-dimensional manifold within high-dimensional spaces. Network pruning aims to reduce the complexity of high-dimensional networks while minimizing performance trade-offs. This symmetry motivates the exploration of ID as a metric for effective pruning. For vision-language models, we investigate whether different modalities exist on separate manifolds, indicating varying complexity and prunability. We empirically study ID variations in large-scale vision-language pre-trained models and examine the contributions of different modalities to model prunability. We propose a layer importance metric based on ID, which can conveniently integrate with current metrics and enhance performance in vision-language model pruning. The experimental results show a high correlation between ID and modality prunability. Visual representations are more sensitive and crucial to model performance, while language representations are more robust and offer greater prunability. Our findings suggest an asymmetric pruning strategy for vision and language modalities, guided by the ID metric. The code is available at https://github.com/Nofear18/ID VL Pruning

</details>

---

## 2. Evaluating and Analyzing Relationship Hallucinations in Large Vision-Language Models

- [ ] Evaluating and Analyzing Relationship Hallucinations in Large Vision-Language Models | https://icml.cc/virtual/2024/poster/32692

- **Link**: https://icml.cc/virtual/2024/poster/32692

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The issue of hallucinations is a prevalent concern in existing Large Vision-Language Models (LVLMs). Previous efforts have primarily focused on investigating object hallucinations, which can be easily alleviated by introducing object detectors. However, these efforts neglect hallucinations in inter-object relationships, which is essential for visual comprehension. In this work, we introduce R-Bench, a novel benchmark for evaluating Vision Relationship Hallucination. R-Bench features image-level questions that focus on the existence of relationships and instance-level questions that assess local visual comprehension. We identify three types of relationship co-occurrences that lead to hallucinations: relationship-relationship, subject-relationship, and relationship-object. The visual instruction tuning dataset's long-tail distribution significantly impacts LVLMs' understanding of visual relationships. Additionally, our analysis reveals that current LVLMs tend to overlook visual content, overly rely on the common sense knowledge of Large Language Models (LLMs), and struggle with spatial relationship reasoning based on contextual information.

</details>

---

## 3. RoboCodeX: Multimodal Code Generation for Robotic Behavior Synthesis

- [ ] RoboCodeX: Multimodal Code Generation for Robotic Behavior Synthesis | https://icml.cc/virtual/2024/poster/32693

- **Link**: https://icml.cc/virtual/2024/poster/32693

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Robotic behavior synthesis, the problem of understanding multimodal inputs and generating precise physical control for robots, is an important part of Embodied AI. Despite successes in applying multimodal large language models for high-level understanding, it remains challenging to translate these conceptual understandings into detailed robotic actions while achieving generalization across various scenarios. In this paper, we propose a tree-structured multimodal code generation framework for generalized robotic behavior synthesis, termed RoboCodeX. RoboCodeX decomposes high-level human instructions into multiple object-centric manipulation units consisting of physical preferences such as affordance and safety constraints, and applies code generation to introduce generalization ability across various robotics platforms. To further enhance the capability to map conceptual and perceptual understanding into control commands, a specialized multimodal reasoning dataset is collected for pre-training and an iterative self-updating methodology is introduced for supervised fine-tuning. Extensive experiments demonstrate that RoboCodeX achieves state-of-the-art performance in both simulators and real robots on four different kinds of manipulation tasks and one embodied navigation task.

</details>

---

## 4. Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection

- [ ] Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection | https://icml.cc/virtual/2024/poster/32704

- **Link**: https://icml.cc/virtual/2024/poster/32704

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Detecting out-of-distribution (OOD) samples is essential when deploying machine learning models in open-world scenarios. Zero-shot OOD detection, requiring no training on in-distribution (ID) data, has been possible with the advent of vision-language models like CLIP. Existing methods build a text-based classifier with only closed-set labels. However, this largely restricts the inherent capability of CLIP to recognize samples from large and open label space. In this paper, we propose to tackle this constraint by leveraging the expert knowledge and reasoning capability of large language models (LLM) to Envision potential Outlier Exposure, termed EOE, without access to any actual OOD data. Owing to better adaptation to open-world scenarios, EOE can be generalized to different tasks, including far, near, and fine-grained OOD detection. Technically, we design (1) LLM prompts based on visual similarity to generate potential outlier class labels specialized for OOD detection, as well as (2) a new score function based on potential outlier penalty to distinguish hard OOD samples effectively. Empirically, EOE achieves state-of-the-art performance across different OOD tasks and can be effectively scaled to the ImageNet-1K dataset. The code is publicly available at: https://github.com/tmlr-group/EOE.

</details>

---

## 5. CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection

- [ ] CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection | https://icml.cc/virtual/2024/poster/32722

- **Link**: https://icml.cc/virtual/2024/poster/32722

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language pre-trained models (VL-PTMs) have shown remarkable success in open-vocabulary tasks. However, downstream use cases often involve further fine-tuning of VL-PTMs, which may distort their general knowledge and impair their ability to handle distribution shifts. In real-world scenarios, machine learning systems inevitably encounter both covariate shifts (e.g., changes in image styles) and semantic shifts (e.g., test-time unseen classes). This highlights the importance of enhancing out-of-distribution (OOD) generalization on covariate shifts and simultaneously detecting semantic-shifted unseen classes. Thus a critical but underexplored question arises: How to improve VL-PTMs' generalization ability to closed-set OOD data, while effectively detecting open-set unseen classes during fine-tuning? In this paper, we propose a novel objective function of OOD detection that also serves to improve OOD generalization. We show that minimizing the gradient magnitude of energy scores on training data leads to domain-consistent Hessians of classification loss, a strong indicator for OOD generalization revealed by theoretical analysis. Based on this finding, we have developed a unified fine-tuning framework that allows for concurrent optimization of both tasks. Extensive experiments have demonstrated the superiority of our method. The code is available at https://github.com/LinLLLL/CRoFT.

</details>

---

## 6. A Touch, Vision, and Language Dataset for Multimodal Alignment

- [ ] A Touch, Vision, and Language Dataset for Multimodal Alignment | https://icml.cc/virtual/2024/poster/32873

- **Link**: https://icml.cc/virtual/2024/poster/32873

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Touch is an important sensing modality for humans, but it has not yet been incorporated into a multimodal generative language model. This is partially due to the difficulty of obtaining natural language labels for tactile data and the complexity of aligning tactile readings with both visual observations and language descriptions. As a step towards bridging that gap, this work introduces a new dataset of 44K in-the-wild visiontouch pairs, with English language labels annotated by humans (10%) and textual pseudo-labels from GPT-4V (90%). We use this dataset to train a vision-language-aligned tactile encoder for open-vocabulary classification and a touch-visionlanguage (TVL) model for text generation using the trained encoder. Results suggest that by incorporating touch, the TVL model improves (+29% classification accuracy) tactile-vision-language alignment over existing models trained on any pair of those modalities. Although only a small fraction of the dataset is human labeled, the TVL model demonstrates improved visual-tactile understanding over GPT-4V (+12%) and open-source vision-language models (+32%) on a new touch-vision understanding benchmark. Code, checkpoints and data are available on https: //tactile-vlm.github.io.

</details>

---

## 7. SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models

- [ ] SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models | https://icml.cc/virtual/2024/poster/32875

- **Link**: https://icml.cc/virtual/2024/poster/32875

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose SPHINX-X, an extensive Multi-modality Large Language Model (MLLM) series developed upon SPHINX. To improve the architecture and training efficiency, we modify the SPHINX framework by removing redundant visual encoders, bypassing fully-padded sub-images with skip tokens, and simplifying multi-stage training into a one-stage all-in-one paradigm. To fully unleash the potential of MLLMs, we assemble a comprehensive multi-domain and multi-modal dataset covering publicly available resources in language, vision, and vision-language tasks. We further enrich this collection with our curated OCR intensive and Set-of-Mark datasets, extending the diversity and generality. By training over different base LLMs including TinyLlama-1.1B, InternLM2-7B, LLaMA2-13B, and Mixtral-8$\times$7B, we obtain a spectrum of MLLMs that vary in parameter size and multilingual capabilities. Comprehensive benchmarking reveals a strong correlation between the multi-modal performance with the data and parameter scales. Code and models are released at https://github.com/Alpha-VLLM/LLaMA2-Accessory.

</details>

---

## 8. Language-Driven Cross-Modal Classifier for Zero-Shot Multi-Label Image Recognition

- [ ] Language-Driven Cross-Modal Classifier for Zero-Shot Multi-Label Image Recognition | https://icml.cc/virtual/2024/poster/32917

- **Link**: https://icml.cc/virtual/2024/poster/32917

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained vision-language models (e.g., CLIP) have shown powerful zero-shot transfer capabilities in image recognition tasks. Recent approaches typically employ supervised fine-tuning methods to adapt CLIP for zero-shot multi-label image recognition tasks. However, obtaining sufficient multi-label annotated image data for training is challenging and not scalable. In this paper, we propose a new language-driven framework for zero-shot multi-label recognition that eliminates the need for annotated images during training. Leveraging the aligned CLIP multi-modal embedding space, our method utilizes language data generated by LLMs to train a cross-modal classifier, which is subsequently transferred to the visual modality. During inference, directly applying the classifier to visual inputs may limit performance due to the modality gap. To address this issue, we introduce a cross-modal mapping method that maps image embeddings to the language modality while retaining crucial visual information. Comprehensive experiments demonstrate that our method outperforms other zero-shot multi-label recognition methods and achieves competitive results compared to few-shot methods.

</details>

---

## 9. Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data

- [ ] Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data | https://icml.cc/virtual/2024/poster/32922

- **Link**: https://icml.cc/virtual/2024/poster/32922

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning vision-language models (VLMs) with abundant unlabeled data recently has attracted increasing attention. Existing methods that resort to the pseudolabeling strategy would suffer from heavily incorrect hard pseudolabels when VLMs exhibit low zero-shot performance in downstream tasks. To alleviate this issue, we propose a C andidate P seudolabel L earning method, termed CPL , to fine-tune VLMs with suitable candidate pseudolabels of unlabeled data in downstream tasks. The core of our method lies in the generation strategy of candidate pseudolabels, which progressively generates refined candidate pseudolabels by both intra- and inter-instance label selection, based on a confidence score matrix for all unlabeled data. This strategy can result in better performance in true label inclusion and class-balanced instance selection. In this way, we can directly apply existing loss functions to learn with generated candidate psueudolabels. Extensive experiments on nine benchmark datasets with three learning paradigms demonstrate the effectiveness of our method. Our code can be found here.

</details>

---

## 10. An Empirical Study Into What Matters for Calibrating Vision-Language Models

- [ ] An Empirical Study Into What Matters for Calibrating Vision-Language Models | https://icml.cc/virtual/2024/poster/32976

- **Link**: https://icml.cc/virtual/2024/poster/32976

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have emerged as the dominant approach for zero-shot recognition, adept at handling diverse scenarios and significant distribution changes. However, their deployment in risk-sensitive areas requires a deeper understanding of their uncertainty estimation capabilities, a relatively uncharted area. In this study, we explore the calibration properties of VLMs across different architectures, datasets, and training strategies. In particular, we analyze the uncertainty estimation performance of VLMs when calibrated in one domain, label set or hierarchy level, and tested in a different one. Our findings reveal that while VLMs are not inherently calibrated for uncertainty, temperature scaling significantly and consistently improves calibration, even across shifts in distribution and changes in label set. Moreover, VLMs can be calibrated with a very small set of examples. Through detailed experimentation, we highlight the potential applications and importance of our insights, aiming for more reliable and effective use of VLMs in critical, real-world scenarios.

</details>

---

## 11. Extracting Training Data From Document-Based VQA Models

- [ ] Extracting Training Data From Document-Based VQA Models | https://icml.cc/virtual/2024/poster/32989

- **Link**: https://icml.cc/virtual/2024/poster/32989

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have made remarkable progress in document-based Visual Question Answering (i.e., responding to queries about the contents of an input document provided as an image). In this work, we show these models can memorize responses for training samples and regurgitate them even when the relevant visual information has been removed. This includes Personal Identifiable Information (PII) repeated once in the training set, indicating these models could divulge memorised sensitive information and therefore pose a privacy risk. We quantitatively measure the extractability of information in controlled experiments and differentiate between cases where it arises from generalization capabilities or from memorization. We further investigate the factors that influence memorization across multiple state-of-the-art models and propose an effective heuristic countermeasure that empirically prevents the extractability of PII.

</details>

---

## 12. Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models

- [ ] Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models | https://icml.cc/virtual/2024/poster/33030

- **Link**: https://icml.cc/virtual/2024/poster/33030

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Catastrophic forgetting emerges as a critical challenge when fine-tuning multi-modal large language models (MLLMs), where improving performance on unseen tasks often leads to a significant performance drop on the original tasks. This paper presents a comprehensive analysis of catastrophic forgetting in MLLMs and introduces a post-training adjustment method called Model Tailor. Our method primarily preserves the pre-trained parameters while replacing a small number ($\leq$ 10%) of fine-tuned parameters, maintaining $\sim$ 99% effectiveness on original tasks versus pre-training, and achieving $\sim$ 97% on new tasks compared to standard fine-tuning. Specifically, we derive a sparse mask to identify the model patch, based on a fusion strategy that integrates salience and sensitivity analysis. Subsequently, a compensation mechanism is introduced to decorate the patch, enhancing the model's performance on both target and original tasks. Additionally, our method is adaptable to multi-task scenarios. Through extensive experiments on InstructBLIP and LLaVA-1.5 in both image captioning and visual question answering tasks, our approach demonstrates significant task adaptability while preserving inherent pre-trained capabilities.

</details>

---

## 13. Open-Vocabulary Calibration for Fine-tuned CLIP

- [ ] Open-Vocabulary Calibration for Fine-tuned CLIP | https://icml.cc/virtual/2024/poster/33036

- **Link**: https://icml.cc/virtual/2024/poster/33036

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have emerged as formidable tools, showing their strong capability in handling various open-vocabulary tasks in image recognition, text-driven visual content generation, and visual chatbots, to name a few. In recent years, considerable efforts and resources have been devoted to adaptation methods for improving downstream performance of VLMs, particularly on parameter-efficient fine-tuning methods like prompt learning. However, a crucial aspect that has been largely overlooked is the confidence calibration problem in fine-tuned VLMs, which could greatly reduce reliability when deploying such models in the real world. This paper bridges the gap by systematically investigating the confidence calibration problem in the context of prompt learning and reveals that existing calibration methods are insufficient to address the problem, especially in the open-vocabulary setting. To solve the problem, we present a simple and effective approach called Distance-Aware Calibration (DAC), which is based on scaling the temperature using as guidance the distance between predicted text labels and base classes. The experiments with 7 distinct prompt learning methods applied across 11 diverse downstream datasets demonstrate the effectiveness of DAC, which achieves high efficacy without sacrificing the inference speed.

</details>

---

## 14. Revealing Vision-Language Integration in the Brain with Multimodal Networks

- [ ] Revealing Vision-Language Integration in the Brain with Multimodal Networks | https://icml.cc/virtual/2024/poster/33050

- **Link**: https://icml.cc/virtual/2024/poster/33050

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We use (multi)modal deep neural networks (DNNs) to probe for sites of multimodal integration in the human brain by predicting stereoencephalography (SEEG) recordings taken while human subjects watched movies. We operationalize sites of multimodal integration as regions where a multimodal vision-language model predicts recordings better than unimodal language, unimodal vision, or linearly-integrated language-vision models. Our target DNN models span different architectures (e.g., convolutional networks and transformers) and multimodal training techniques (e.g., cross-attention and contrastive learning). As a key enabling step, we first demonstrate that trained vision and language models systematically outperform their randomly initialized counterparts in their ability to predict SEEG signals. We then compare unimodal and multimodal models against one another. Because our target DNN models often have different architectures, number of parameters, and training sets (possibly obscuring those differences attributable to integration), we carry out a controlled comparison of two models (SLIP and SimCLR), which keep all of these attributes the same aside from input modality. Using this approach, we identify a sizable number of neural sites (on average 141 out of 1090 total sites or 12.94%) and brain regions where multimodal integration seems to occur. Additionally, we find that among the variants of multimodal training techniques we assess, CLIP-style training is the best suited for downstream prediction of the neural activity in these sites.

</details>

---

## 15. Using Left and Right Brains Together: Towards Vision and Language Planning

- [ ] Using Left and Right Brains Together: Towards Vision and Language Planning | https://icml.cc/virtual/2024/poster/33100

- **Link**: https://icml.cc/virtual/2024/poster/33100

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) and Large Multi-modality Models (LMMs) have demonstrated remarkable decision masking capabilities on a variety of tasks. However, they inherently operate planning within the language space, lacking the vision and spatial imagination ability. In contrast, humans utilize both left and right hemispheres of the brain for language and visual planning during the thinking process. Therefore, we introduce a novel vision-language planning framework in this work to perform concurrent visual and language planning for tasks with inputs of any form. Our framework incorporates visual planning to capture intricate environmental details, while language planning enhances the logical coherence of the overall system. We evaluate the effectiveness of our framework across vision-language tasks, vision-only tasks, and language-only tasks. The results demonstrate the superior performance of our approach, indicating that the integration of visual and language planning yields better contextually aware task execution.

</details>

---

## 16. WebLINX: Real-World Website Navigation with Multi-Turn Dialogue

- [ ] WebLINX: Real-World Website Navigation with Multi-Turn Dialogue | https://icml.cc/virtual/2024/poster/33174

- **Link**: https://icml.cc/virtual/2024/poster/33174

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose the problem of conversational web navigation, where a digital agent controls a web browser and follows user instructions to solve real-world tasks in a multi-turn dialogue fashion. To support this problem, we introduce WEBLINX - a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. Our benchmark covers a broad range of patterns on over 150 real-world websites and can be used to train and evaluate agents in diverse scenarios. Due to the magnitude of information present, Large Language Models (LLMs) cannot process entire web pages in real-time. To solve this bottleneck, we design a retrieval-inspired model that efficiently prunes HTML pages by ranking relevant elements. We use the selected elements, along with screenshots and action history, to assess a variety of models for their ability to replicate human behavior when navigating the web. Our experiments span from small text-only to proprietary multimodal LLMs. We find that smaller finetuned decoders surpass the best zero-shot LLMs (including GPT-4V), but also larger finetuned multimodal models which were explicitly pretrained on screenshots. However, all finetuned models struggle to generalize to unseen websites. Our findings highlight the need for large multimodal models that can generalize to novel settings.

</details>

---

## 17. A Multimodal Automated Interpretability Agent

- [ ] A Multimodal Automated Interpretability Agent | https://icml.cc/virtual/2024/poster/33183

- **Link**: https://icml.cc/virtual/2024/poster/33183

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper describes MAIA, a Multimodal Automated Interpretability Agent. MAIA is a system that uses neural models to automate neural model understanding tasks like feature interpretation and failure mode discovery. It equips a pre-trained vision-language model with a set of tools that support iterative experimentation on subcomponents of other models to explain their behavior. These include tools commonly used by human interpretability researchers: for synthesizing and editing inputs, computing maximally activating exemplars from real-world datasets, and summarizing and describing experimental results. Interpretability experiments proposed by MAIA compose these tools to describe and explain system behavior. We evaluate applications of MAIA to computer vision models. We first characterize MAIA’s ability to describe (neuron-level) features in learned representations of images. Across several trained models and a novel dataset of synthetic vision neurons with paired ground-truth descriptions, MAIA produces descriptions comparable to those generated by expert human experimenters. We then show that MAIA can aid in two additional interpretability tasks: reducing sensitivity to spurious features, and automatically identifying inputs likely to be mis-classified.

</details>

---

## 18. Connecting the Dots: Collaborative Fine-tuning for Black-Box Vision-Language Models

- [ ] Connecting the Dots: Collaborative Fine-tuning for Black-Box Vision-Language Models | https://icml.cc/virtual/2024/poster/33298

- **Link**: https://icml.cc/virtual/2024/poster/33298

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the emergence of pretrained vision-language models (VLMs), considerable efforts have been devoted to fine-tuning them for downstream tasks. Despite the progress made in designing efficient fine-tuning methods, such methods require access to the model's parameters, which can be challenging as model owners often opt to provide their models as a black box to safeguard model ownership. This paper proposes a C ollabo ra tive F ine- T uning ( CraFT ) approach for fine-tuning black-box VLMs to downstream tasks, where one only has access to the input prompts and the output predictions of the model. CraFT comprises two modules, a prompt generation module for learning text prompts and a prediction refinement module for enhancing output predictions in residual style. Additionally, we introduce an auxiliary prediction-consistent loss to promote consistent optimization across these modules. These modules are optimized by a novel collaborative training algorithm. Extensive experiments on few-shot classification over 15 datasets demonstrate the superiority of CraFT. The results show that CraFT achieves a decent gain of about 12% with 16-shot datasets and only 8,000 queries. Moreover, CraFT trains faster and uses only about 1/80 of the memory footprint for deployment, while sacrificing only 1.62% compared to the white-box method. Our code is publicly available at https://github.com/mrflogs/CraFT.

</details>

---

## 19. SyCoCa: Symmetrizing Contrastive Captioners with Attentive Masking for Multimodal Alignment

- [ ] SyCoCa: Symmetrizing Contrastive Captioners with Attentive Masking for Multimodal Alignment | https://icml.cc/virtual/2024/poster/33300

- **Link**: https://icml.cc/virtual/2024/poster/33300

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal alignment between language and vision is the fundamental topic in current vision-language model research. Contrastive Captioners (CoCa), as a representative method, integrates Contrastive Language-Image Pretraining (CLIP) and Image Caption (IC) into a unified framework, resulting in impressive results. CLIP imposes a bidirectional constraints on global representations of entire images and sentences. Although IC conducts an unidirectional image-to-text generation on local representation, it lacks any constraint on local text-to-image reconstruction, which limits the ability to understand images at a fine-grained level when aligned with texts. To achieve multimodal alignment from both global and local perspectives, this paper proposes Symmetrizing Contrastive Captioners (SyCoCa), which introduces bidirectional interactions on images and texts across the global and local representation levels. Specifically, we expand a Text-Guided Masked Image Modeling (TG-MIM) head based on ITC and IC heads. The improved SyCoCa further leverages textual cues to reconstruct contextual images and visual cues to predict textual contents. When implementing bidirectional local interactions, the local contents of images tend to be cluttered or unrelated to their textual descriptions. Thus, we employ an attentive masking strategy to select effective image patches for interaction. Extensive experiments on five vision-language tasks, including image-text retrieval, image-captioning, visual question answering, and zero-shot/finetuned image classification, validate the effectiveness of our proposed method.

</details>

---

## 20. Modeling Caption Diversity in Contrastive Vision-Language Pretraining

- [ ] Modeling Caption Diversity in Contrastive Vision-Language Pretraining | https://icml.cc/virtual/2024/poster/33348

- **Link**: https://icml.cc/virtual/2024/poster/33348

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

There are a thousand ways to caption an image. Contrastive Language Pretraining (CLIP) on the other hand, works by mapping an image and its caption to a single vector -- limiting how well CLIP-like models can represent the diverse ways to describe an image. In this work, we introduce Llip, Latent Language Image Pretraining, which models the diversity of captions that could match an image. Llip's vision encoder outputs a set of visual features that are mixed into a final representation by conditioning on information derived from the text. We show that Llip outperforms non-contextualized baselines like CLIP and SigLIP on a variety of tasks even with large-scale encoders. Llip improves zero-shot classification by an average of 2.9% zero-shot classification benchmarks with a ViT-G/14 encoder. Specifically, Llip attains a zero-shot top-1 accuracy of 83.5% on ImageNet outperforming a similarly sized CLIP by 1.4%. We also demonstrate improvement on zero-shot retrieval on MS-COCO by 6.0%. We provide a comprehensive analysis of the components introduced by the method and demonstrate that Llip leads to richer visual representations.

</details>

---

## 21. Referee Can Play: An Alternative Approach to Conditional Generation via Model Inversion

- [ ] Referee Can Play: An Alternative Approach to Conditional Generation via Model Inversion | https://icml.cc/virtual/2024/poster/33389

- **Link**: https://icml.cc/virtual/2024/poster/33389

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As a dominant force in text-to-image generation tasks, Diffusion Probabilistic Models (DPMs) face a critical challenge in controllability, struggling to adhere strictly to complex, multi-faceted instructions. In this work, we aim to address this alignment challenge for conditional generation tasks. First, we provide an alternative view of state-of-the-art DPMs as a way of inverting advanced Vision-Language Models (VLMs). With this formulation, we naturally propose a training-free approach that bypasses the conventional sampling process associated with DPMs. By directly optimizing images with the supervision of discriminative VLMs, the proposed method can potentially achieve a better text-image alignment. As proof of concept, we demonstrate the pipeline with the pre-trained BLIP-2 model and identify several key designs for improved image generation. To further enhance the image fidelity, a Score Distillation Sampling module of Stable Diffusion is incorporated. By carefully balancing the two components during optimization, our method can produce high-quality images with near state-of-the-art performance on T2I-Compbench. The code is available at https://github.com/Pepper-lll/VLMinv.

</details>

---

## 22. SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code

- [ ] SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code | https://icml.cc/virtual/2024/poster/33438

- **Link**: https://icml.cc/virtual/2024/poster/33438

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces SceneCraft, a Large Language Model (LLM) Agent converting text descriptions into Blender-executable Python scripts which render complex scenes with up to a hundred 3D assets. This process requires complex spatial planning and arrangement. We tackle these challenges through a combination of advanced abstraction, strategic planning, and library learning. SceneCraft first models a scene graph as a blueprint, detailing the spatial relationships among assets in the scene. SceneCraft then writes Python scripts based on this graph, translating relationships into numerical constraints for asset layout. Next, SceneCraft leverages the perceptual strengths of vision-language foundation models like GPT-V to analyze rendered images and iteratively refine the scene. On top of this process, SceneCraft features a library learning mechanism that compiles common script functions into a reusable library, facilitating continuous self-improvement without expensive LLM parameter tuning. Our evaluation demonstrates that SceneCraft surpasses existing LLM-based agents in rendering complex scenes, as shown by its adherence to constraints and favorable human assessments. We also showcase the broader application potential of SceneCraft by reconstructing detailed 3D scenes from the Sintel movie and guiding a video generative model with generated scenes as intermediary control signal.

</details>

---

## 23. Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition

- [ ] Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition | https://icml.cc/virtual/2024/poster/33467

- **Link**: https://icml.cc/virtual/2024/poster/33467

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing research of video understanding still struggles to achieve in-depth comprehension and reasoning in complex videos, primarily due to the under-exploration of two key bottlenecks: fine-grained spatial-temporal perceptive understanding and cognitive-level video scene comprehension. This paper bridges the gap by presenting a novel solution. We first introduce a novel video Multimodal Large Language Model (MLLM), MotionEpic, which achieves fine-grained pixel-level spatial-temporal video grounding by integrating video spatial-temporal scene graph (STSG) representation. Building upon MotionEpic, we then develop a Video-of-Thought (VoT) reasoning framework. VoT inherits the Chain-of-Thought (CoT) core, breaking down a complex task into simpler and manageable sub-problems, and addressing them step-by-step from a low-level pixel perception to high-level cognitive interpretation. Extensive experiments across various complex video QA benchmarks demonstrate that our overall framework strikingly boosts existing state-of-the-art. To our knowledge, this is the first attempt at successfully implementing the CoT technique for achieving human-level video reasoning, where we show great potential in extending it to a wider range of video understanding scenarios. Systems and codes will be open later.

</details>

---

## 24. Amend to Alignment: Decoupled Prompt Tuning for Mitigating Spurious Correlation in Vision-Language Models

- [ ] Amend to Alignment: Decoupled Prompt Tuning for Mitigating Spurious Correlation in Vision-Language Models | https://icml.cc/virtual/2024/poster/33470

- **Link**: https://icml.cc/virtual/2024/poster/33470

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning the learnable prompt for a pre-trained vision-language model (VLM), such as CLIP, has demonstrated exceptional efficiency in adapting to a broad range of downstream tasks. Existing prompt tuning methods for VLMs do not distinguish spurious features introduced by biased training data from invariant features, and employ a uniform alignment process when adapting to unseen target domains. This can impair the cross-modal feature alignment when the testing data significantly deviate from the distribution of the training data, resulting in a poor out-of-distribution (OOD) generalization performance. In this paper, we reveal that the prompt tuning failure in such OOD scenarios can be attribute to the undesired alignment between the textual and the spurious feature. As a solution, we propose CoOPood , a fine-grained prompt tuning method that can discern the causal features and deliberately align the text modality with the invariant feature. Specifically, we design two independent contrastive phases using two lightweight projection layers during the alignment, each with different objectives: 1) pulling the text embedding closer to invariant image embedding and 2) pushing text embedding away from spurious image embedding. We have illustrated that CoOPood can serve as a general framework for VLMs and can be seamlessly integrated with existing prompt tuning methods. Extensive experiments on various OOD datasets demonstrate the performance superiority over state-of-the-art methods.

</details>

---

## 25. Image Fusion via Vision-Language Model

- [ ] Image Fusion via Vision-Language Model | https://icml.cc/virtual/2024/poster/33477

- **Link**: https://icml.cc/virtual/2024/poster/33477

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image fusion integrates essential information from multiple images into a single composite, enhancing structures, textures, and refining imperfections. Existing methods predominantly focus on pixel-level and semantic visual features for recognition, but often overlook the deeper text-level semantic information beyond vision. Therefore, we introduce a novel fusion paradigm named image Fusion via vIsion-Language Model (FILM), for the first time, utilizing explicit textual information from source images to guide the fusion process. Specifically, FILM generates semantic prompts from images and inputs them into ChatGPT for comprehensive textual descriptions. These descriptions are fused within the textual domain and guide the visual information fusion, enhancing feature extraction and contextual understanding, directed by textual semantic information via cross-attention. FILM has shown promising results in four image fusion tasks: infrared-visible, medical, multi-exposure, and multi-focus image fusion. We also propose a vision-language dataset containing ChatGPT-generated paragraph descriptions for the eight image fusion datasets across four fusion tasks, facilitating future research in vision-language model-based image fusion. Code and dataset are available at https://github.com/Zhaozixiang1228/IF-FILM.

</details>

---

## 26. RoboMP$^2$: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models

- [ ] RoboMP$^2$: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models | https://icml.cc/virtual/2024/poster/33506

- **Link**: https://icml.cc/virtual/2024/poster/33506

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown impressive reasoning abilities and general intelligence in various domains. It inspires researchers to train end-to-end MLLMs or utilize large models to generate policies with human-selected prompts for embodied agents. However, these methods exhibit limited generalization capabilities on unseen tasks or scenarios, and overlook the multimodal environment information which is critical for robots to make decisions. In this paper, we introduce a novel **Robo**tic **M**ultimodal **P**erception-**P**lanning (**RoboMP$^2$**) framework for robotic manipulation which consists of a Goal-Conditioned Multimodal Preceptor (GCMP) and a Retrieval-Augmented Multimodal Planner (RAMP). Specially, GCMP captures environment states by employing a tailored MLLMs for embodied agents with the abilities of semantic reasoning and localization. RAMP utilizes coarse-to-fine retrieval method to find the $k$ most-relevant policies as in-context demonstrations to enhance the planner. Extensive experiments demonstrate the superiority of RoboMP$^2$ on both VIMA benchmark and real-world tasks, with around 10% improvement over the baselines.

</details>

---

## 27. MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark

- [ ] MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark | https://icml.cc/virtual/2024/poster/33545

- **Link**: https://icml.cc/virtual/2024/poster/33545

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have gained significant attention recently, showing remarkable potential in artificial general intelligence. However, assessing the utility of MLLMs presents considerable challenges, primarily due to the absence multimodal benchmarks that align with human preferences. Drawing inspiration from the concept of LLM-as-a-Judge within LLMs, this paper introduces a novel benchmark, termed MLLM-as-a-Judge, to assess the ability of MLLMs in assisting judges across diverse modalities, encompassing three distinct tasks: Scoring Evaluation, Pair Comparison, and Batch Ranking. Our study reveals that, while MLLMs demonstrate remarkable human-like discernment in Pair Comparisons, there is a significant divergence from human preferences in Scoring Evaluation and Batch Ranking tasks. Furthermore, a closer examination reveals persistent challenges in the evaluative capacities of LLMs, including diverse biases, hallucinatory responses, and inconsistencies in judgment, even in advanced models such as GPT-4V. These findings emphasize the pressing need for enhancements and further research efforts to be undertaken before regarding MLLMs as fully reliable evaluators. In light of this, we advocate for additional efforts dedicated to supporting the continuous development within the domain of MLLM functioning as judges. The code and dataset are publicly available at our project homepage: https://mllm-judge.github.io/.

</details>

---

## 28. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models

- [ ] Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models | https://icml.cc/virtual/2024/poster/33636

- **Link**: https://icml.cc/virtual/2024/poster/33636

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset will be open-sourced.

</details>

---

## 29. Watermarks in the Sand: Impossibility of Strong Watermarking for Language Models

- [ ] Watermarks in the Sand: Impossibility of Strong Watermarking for Language Models | https://icml.cc/virtual/2024/poster/33645

- **Link**: https://icml.cc/virtual/2024/poster/33645

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023), and include preliminary results on vision-language models. The same attack successfully removes the watermarks planted by all schemes, with only minor quality degradation.

</details>

---

## 30. Bridging Environments and Language with Rendering Functions and Vision-Language Models

- [ ] Bridging Environments and Language with Rendering Functions and Vision-Language Models | https://icml.cc/virtual/2024/poster/33722

- **Link**: https://icml.cc/virtual/2024/poster/33722

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have tremendous potential for grounding language, and thus enabling language-conditioned agents (LCAs) to perform diverse tasks specified with text. This has motivated the study of LCAs based on reinforcement learning (RL) with rewards given by rendering images of an environment and evaluating those images with VLMs. If single-task RL is employed, such approaches are limited by the cost and time required to train a policy for each new task. Multi-task RL (MTRL) is a natural alternative, but requires a carefully designed corpus of training tasks and does not always generalize reliably to new tasks. Therefore, this paper introduces a novel decomposition of the problem of building an LCA: first find an environment configuration that has a high VLM score for text describing a task; then use a (pretrained) goal-conditioned policy to reach that configuration. We also explore several enhancements to the speed and quality of VLM-based LCAs, notably, the use of distilled models, and the evaluation of configurations from multiple viewpoints to resolve the ambiguities inherent in a single 2D view. We demonstrate our approach on the Humanoid environment, showing that it results in LCAs that outperform MTRL baselines in zero-shot generalization, without requiring any textual task descriptions or other forms of environment-specific annotation during training.

</details>

---

## 31. MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions

- [ ] MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions | https://icml.cc/virtual/2024/poster/33731

- **Link**: https://icml.cc/virtual/2024/poster/33731

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image retrieval, i.e., finding desired images given a reference image, inherently encompasses rich, multi-faceted search intents that are difficult to capture solely using image-based measures. Recent works leverage text instructions to allow users to more freely express their search intents. However, they primarily focus on image pairs that are visually similar and/or can be characterized by a small set of pre-defined relations. The core thesis of this paper is that text instructions can enable retrieving images with richer relations beyond visual similarity. To show this, we introduce MagicLens, a series of self-supervised image retrieval models that support open-ended instructions. MagicLens is built on a key novel insight: image pairs that naturally occur on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we can bring those implicit relations explicit by synthesizing instructions via foundation models. Trained on 36.7M (query image, instruction, target image) triplets with rich semantic relations mined from the web, MagicLens achieves results comparable with or better than prior best on eight benchmarks of various image retrieval tasks, while maintaining high parameter efficiency with a significantly smaller model size. Additional human analyses on a 1.4M-image unseen corpus further demonstrate the diversity of search intents supported by MagicLens. Code and models are publicly available at the https://open-vision-language.github.io/MagicLens/.

</details>

---

## 32. Harmonizing Generalization and Personalization in Federated Prompt Learning

- [ ] Harmonizing Generalization and Personalization in Federated Prompt Learning | https://icml.cc/virtual/2024/poster/33769

- **Link**: https://icml.cc/virtual/2024/poster/33769

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Federated Prompt Learning (FPL) incorporates large pre-trained Vision-Language models (VLM) into federated learning through prompt tuning. The transferable representations and remarkable generalization capacity of VLM make them highly compatible with the integration of federated learning. Addressing data heterogeneity in federated learning requires personalization, but excessive focus on it across clients could compromise the model's ability to generalize effectively. To preserve the impressive generalization capability of VLM, it is crucial to strike a balance between personalization and generalization in FPL. To tackle this challenge, we proposed Federated Prompt Learning with CLIP Generalization and low-rank Personalization (FedPGP), which employs pre-trained CLIP to provide knowledge-guidance on the global prompt for improved generalization and incorporates a low-rank adaptation term to personalize the global prompt. Further, FedPGP integrates a prompt-wise contrastive loss to achieve knowledge guidance and personalized adaptation simultaneously, enabling a harmonious balance between personalization and generalization in FPL. We conduct extensive experiments on various datasets to explore base-to-novel generalization in both category-level and domain-level scenarios with heterogeneous data, showing the superiority of FedPGP in balancing generalization and personalization.

</details>

---

## 33. RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback

- [ ] RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback | https://icml.cc/virtual/2024/poster/33772

- **Link**: https://icml.cc/virtual/2024/poster/33772

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Reward engineering has long been a challenge in Reinforcement Learning (RL) research, as it often requires extensive human effort and iterative processes of trial-and-error to design effective reward functions. In this paper, we propose RL-VLM-F, a method that automatically generates reward functions for agents to learn new tasks, using only a text description of the task goal and the agent's visual observations, by leveraging feedbacks from vision language foundation models (VLMs). The key to our approach is to query these models to give preferences over pairs of the agent's image observations based on the text description of the task goal, and then learn a reward function from the preference labels, rather than directly prompting these models to output a raw reward score, which can be noisy and inconsistent. We demonstrate that RL-VLM-F successfully produces effective rewards and policies across various domains — including classic control, as well as manipulation of rigid, articulated, and deformable objects — without the need for human supervision, outperforming prior methods that use large pretrained models for reward generation under the same assumptions. Videos can be found on our project website: https://rlvlmf2024.github.io/

</details>

---

## 34. Realistic Unsupervised CLIP Fine-tuning with Universal Entropy Optimization

- [ ] Realistic Unsupervised CLIP Fine-tuning with Universal Entropy Optimization | https://icml.cc/virtual/2024/poster/33795

- **Link**: https://icml.cc/virtual/2024/poster/33795

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The emergence of vision-language models, such as CLIP, has spurred a significant research effort towards their application for downstream supervised learning tasks. Although some previous studies have explored the unsupervised fine-tuning of CLIP, they often rely on prior knowledge in the form of class names associated with ground truth labels. This paper explores a realistic unsupervised fine-tuning scenario, considering the presence of out-of-distribution samples from unknown classes within the unlabeled data. In particular, we focus on simultaneously enhancing out-of-distribution detection and the recognition of instances associated with known classes. To tackle this problem, we present a simple, efficient, and effective approach called Universal Entropy Optimization (UEO). UEO leverages sample-level confidence to approximately minimize the conditional entropy of confident instances and maximize the marginal entropy of less confident instances. Apart from optimizing the textual prompt, UEO incorporates optimization of channel-wise affine transformations within the visual branch of CLIP. Extensive experiments across 15 domains and 4 different types of prior knowledge validate the effectiveness of UEO compared to baseline methods. The code is at https://github.com/tim-learn/UEO.

</details>

---

## 35. GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model

- [ ] GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model | https://icml.cc/virtual/2024/poster/33861

- **Link**: https://icml.cc/virtual/2024/poster/33861

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This work tackles the problem of geo-localization with a new paradigm using a large vision-language model (LVLM) augmented with human inference knowledge. A primary challenge here is the scarcity of data for training the LVLM - existing street-view datasets often contain numerous low-quality images lacking visual clues, and lack any reasoning inference. To address the data-quality issue, we devise a CLIP-based network to quantify the degree of street-view images being locatable, leading to the creation of a new dataset comprising highly locatable street views. To enhance reasoning inference, we integrate external knowledge obtained from real geo-localization games, tapping into valuable human inference capabilities. The data are utilized to train GeoReasoner, which undergoes fine-tuning through dedicated reasoning and location-tuning stages. Qualitative and quantitative evaluations illustrate that GeoReasoner outperforms counterpart LVLMs by more than 25% at country-level and 38% at city-level geo-localization tasks, and surpasses StreetCLIP performance while requiring fewer training resources. The data and code are available at https://github.com/lingli1996/GeoReasoner.

</details>

---

## 36. Cascade-CLIP: Cascaded Vision-Language Embeddings Alignment for Zero-Shot Semantic Segmentation

- [ ] Cascade-CLIP: Cascaded Vision-Language Embeddings Alignment for Zero-Shot Semantic Segmentation | https://icml.cc/virtual/2024/poster/33865

- **Link**: https://icml.cc/virtual/2024/poster/33865

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models, e.g., CLIP, have been successfully applied to zero-shot semantic segmentation. Existing CLIP-based approaches primarily utilize visual features from the last layer to align with text embeddings, while they neglect the crucial information in intermediate layers that contain rich object details. However, we find that directly aggregating the multi-level visual features weakens the zero-shot ability for novel classes. The large differences between the visual features from different layers make these features hard to align well with the text embeddings. We resolve this problem by introducing a series of independent decoders to align the multi-level visual features with the text embeddings in a cascaded way, forming a novel but simple framework named Cascade-CLIP. Our Cascade-CLIP is flexible and can be easily applied to existing zero-shot semantic segmentation methods. Experimental results show that our simple Cascade-CLIP achieves superior zero-shot performance on segmentation benchmarks, like COCO-Stuff, Pascal-VOC, and Pascal-Context. Our code is available at https://github.com/HVision-NKU/Cascade-CLIP.

</details>

---

## 37. Gradient-based Visual Explanation for Transformer-based CLIP

- [ ] Gradient-based Visual Explanation for Transformer-based CLIP | https://icml.cc/virtual/2024/poster/33867

- **Link**: https://icml.cc/virtual/2024/poster/33867

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Significant progress has been achieved on the improvement and downstream usages of the Contrastive Language-Image Pre-training (CLIP) vision-language model, while less attention is paid to the interpretation of CLIP. We propose a Gradient-based visual Explanation method for CLIP (Grad-ECLIP), which interprets the matching result of CLIP for specific input image-text pair. By decomposing the architecture of the encoder and discovering the relationship between the matching similarity and intermediate spatial features, Grad-ECLIP produces effective heat maps that show the influence of image regions or words on the CLIP results. Different from the previous Transformer interpretation methods that focus on the utilization of self-attention maps, which are typically extremely sparse in CLIP, we produce high-quality visual explanations by applying channel and spatial weights on token features. Qualitative and quantitative evaluations verify the superiority of Grad-ECLIP compared with the state-of-the-art methods. A series of analysis are conducted based on our visual explanation results, from which we explore the working mechanism of image-text matching, and the strengths and limitations in attribution identification of CLIP. Codes are available here: https://github.com/Cyang-Zhao/Grad-Eclip.

</details>

---

## 38. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models

- [ ] Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models | https://icml.cc/virtual/2024/poster/33875

- **Link**: https://icml.cc/virtual/2024/poster/33875

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available on GitHub.

</details>

---

## 39. An Embodied Generalist Agent in 3D World

- [ ] An Embodied Generalist Agent in 3D World | https://icml.cc/virtual/2024/poster/33925

- **Link**: https://icml.cc/virtual/2024/poster/33925

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Leveraging massive knowledge from large language models (LLMs), recent machine learning models show notable successes in general-purpose task solving in diverse domains such as computer vision and robotics. However, several significant challenges remain: (i) most of these models rely on 2D images yet exhibit a limited capacity for 3D input; (ii) these models rarely explore the tasks inherently defined in 3D world, e.g., 3D grounding, embodied reasoning and acting. We argue these limitations significantly hinder current models from performing real-world tasks and approaching general intelligence. To this end, we introduce LEO, an embodied multi-modal generalist agent that excels in perceiving, grounding, reasoning, planning, and acting in the 3D world. LEO is trained with a unified task interface, model architecture, and objective in two stages: (i) 3D vision-language (VL) alignment and (ii) 3D vision-language-action (VLA) instruction tuning. We collect large-scale datasets comprising diverse object-level and scene-level tasks, which require considerable understanding of and interaction with the 3D world. Moreover, we meticulously design an LLM-assisted pipeline to produce high-quality 3D VL data. Through extensive experiments, we demonstrate LEO's remarkable proficiency across a wide spectrum of tasks, including 3D captioning, question answering, embodied reasoning, navigation and manipulation. Our ablative studies and scaling analyses further provide valuable insights for developing future embodied generalist agents. Code and data are available on project page .

</details>

---

## 40. Auto-Encoding Morph-Tokens for Multimodal LLM

- [ ] Auto-Encoding Morph-Tokens for Multimodal LLM | https://icml.cc/virtual/2024/poster/33953

- **Link**: https://icml.cc/virtual/2024/poster/33953

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

For multimodal LLMs, the synergy of visual comprehension (textual output) and generation (visual output) presents an ongoing challenge. This is due to a conflicting objective: for comprehension, an MLLM needs to abstract the visuals; for generation, it needs to preserve the visuals as much as possible. Thus, the objective is a dilemma for visual-tokens. To resolve the conflict, we propose encoding images into morph-tokens to serve a dual purpose: for comprehension, they act as visual prompts instructing MLLM to generate texts; for generation, they take on a different, non-conflicting role as complete visual-tokens for image reconstruction, where the missing visual cues are recovered by the MLLM. Extensive experiments show that morph-tokens can achieve a new SOTA for multimodal comprehension and generation simultaneously. Our project is available at https://github.com/DCDmllm/MorphTokens.

</details>

---

## 41. Video-LaVIT: Unified Video-Language Pre-training with Decoupled Visual-Motional Tokenization

- [ ] Video-LaVIT: Unified Video-Language Pre-training with Decoupled Visual-Motional Tokenization | https://icml.cc/virtual/2024/poster/34023

- **Link**: https://icml.cc/virtual/2024/poster/34023

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In light of recent advances in multimodal Large Language Models (LLMs), there is increasing attention to scaling them from image-text data to more informative real-world videos. Compared to static images, video poses unique challenges for effective large-scale pre-training due to the modeling of its spatiotemporal dynamics. In this paper, we address such limitations in video-language pre-training with an efficient video decomposition that represents each video as keyframes and temporal motions. These are then adapted to an LLM using well-designed tokenizers that discretize visual and temporal information as a few tokens, thus enabling unified generative pre-training of videos, images, and text. At inference, the generated tokens from the LLM are carefully recovered to the original continuous pixel space to create various video content. Our proposed framework is both capable of comprehending and generating image and video content, as demonstrated by its competitive performance across 13 multimodal benchmarks in image and video understanding and generation. Our code and models are available at https://video-lavit.github.io.

</details>

---

## 42. Let Go of Your Labels with Unsupervised Transfer

- [ ] Let Go of Your Labels with Unsupervised Transfer | https://icml.cc/virtual/2024/poster/34048

- **Link**: https://icml.cc/virtual/2024/poster/34048

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Foundation vision-language models have enabled remarkable zero-shot transferability of the pre-trained representations to a wide range of downstream tasks. However, to solve a new task, zero-shot transfer still necessitates human guidance to define visual categories that appear in the data. Here, we show that fully unsupervised transfer emerges when searching for the labeling of a dataset that induces maximal margin classifiers in representation spaces of different foundation models. We present TURTLE, a fully unsupervised method that effectively employs this guiding principle to uncover the underlying labeling of a downstream dataset without any supervision and task-specific representation learning. We evaluate TURTLE on a diverse benchmark suite of 26 datasets and show that it achieves new state-of-the-art unsupervised performance. Furthermore, TURTLE, although being fully unsupervised, outperforms zero-shot transfer baselines on a wide range of datasets. In particular, TURTLE matches the average performance of CLIP zero-shot on 26 datasets by employing the same representation space, spanning a wide range of architectures and model sizes. By guiding the search for the underlying labeling using the representation spaces of two foundation models, TURTLE surpasses zero-shot transfer and unsupervised prompt tuning baselines, demonstrating the surprising power and effectiveness of unsupervised transfer.

</details>

---

## 43. Understanding Retrieval-Augmented Task Adaptation for Vision-Language Models

- [ ] Understanding Retrieval-Augmented Task Adaptation for Vision-Language Models | https://icml.cc/virtual/2024/poster/34055

- **Link**: https://icml.cc/virtual/2024/poster/34055

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained contrastive vision-language models have demonstrated remarkable performance across a wide range of tasks. However, they often struggle on fine-trained datasets with categories not adequately represented during pre-training, which makes adaptation necessary. Recent works have shown promising results by utilizing samples from web-scale databases for retrieval-augmented adaptation, especially in low-data regimes. Despite the empirical success, understanding how retrieval impacts the adaptation of vision-language models remains an open research question. In this work, we adopt a reflective perspective by presenting a systematic study to understand the roles of key components in retrieval-augmented adaptation. We unveil new insights on uni-modal and cross-modal retrieval and highlight the critical role of logit ensemble for effective adaptation. We further present theoretical underpinnings that directly support our empirical observations.

</details>

---

## 44. MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI

- [ ] MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI | https://icml.cc/virtual/2024/poster/34062

- **Link**: https://icml.cc/virtual/2024/poster/34062

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) show significant strides in general-propose multimodal applications such as visual dialogue and embodied navigation. However, existing multimodal evaluation benchmarks cover a limited number of multimodal tasks testing rudimentary capabilities, falling short in tracking LVLM development. In this study, we present MMT-Bench, a comprehensive benchmark designed to assess LVLMs across massive multimodal tasks requiring expert knowledge and deliberate visual recognition, localization, and reasoning. MMT-Bench comprises $31,325$ meticulously curated multi-choice visual questions from various multimodal scenarios such as vehicle driving and embodied navigation, covering $32$ core meta-tasks and $162$ subtasks in multimodal understanding. Due to its extensive task coverage, MMT-Bench enables the evaluation of LVLMs using a task map, facilitating the discovery of in- and out-of-domain tasks. Evaluation results involving $20$ publicly available LVLMs such as the proprietary GeminiProVision model, underscore the significant challenges posed by MMT-Bench. We anticipate that MMT-Bench will inspire the community to develop next-generation multimodal foundation models aimed at achieving general-purpose multimodal intelligence.

</details>

---

## 45. Diagnosing the Compositional Knowledge of Vision Language Models from a Game-Theoretic View

- [ ] Diagnosing the Compositional Knowledge of Vision Language Models from a Game-Theoretic View | https://icml.cc/virtual/2024/poster/34101

- **Link**: https://icml.cc/virtual/2024/poster/34101

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Compositional reasoning capabilities are usually considered as fundamental skills to characterize human perception. Recent studies show that current Vision Language Models (VLMs) surprisingly lack sufficient knowledge with respect to such capabilities. To this end, we propose to thoroughly diagnose the composition representations encoded by VLMs, systematically revealing the potential cause for this weakness. Specifically, we propose evaluation methods from a novel game-theoretic view to assess the vulnerability of VLMs on different aspects of compositional understanding, e.g., relations and attributes. Extensive experimental results demonstrate and validate several insights to understand the incapabilities of VLMs on compositional reasoning, which provide useful and reliable guidance for future studies. The deliverables will be updated here .

</details>

---

## 46. Prompt-based Visual Alignment for Zero-shot Policy Transfer

- [ ] Prompt-based Visual Alignment for Zero-shot Policy Transfer | https://icml.cc/virtual/2024/poster/34124

- **Link**: https://icml.cc/virtual/2024/poster/34124

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Overfitting in RL has become one of the main obstacles to applications in reinforcement learning(RL). Existing methods do not provide explicit semantic constrain for the feature extractor, hindering the agent from learning a unified cross-domain representation and resulting in performance degradation on unseen domains. Besides, abundant data from multiple domains are needed. To address these issues, in this work, we propose prompt-based visual alignment (PVA), a robust framework to mitigate the detrimental domain bias in the image for zero-shot policy transfer. Inspired that Visual-Language Model (VLM) can serve as a bridge to connect both text space and image space, we leverage the semantic information contained in a text sequence as an explicit constraint to train a visual aligner. Thus, the visual aligner can map images from multiple domains to a unified domain and achieve good generalization performance. To better depict semantic information, prompt tuning is applied to learn a sequence of learnable tokens. With explicit constraints of semantic information, PVA can learn unified cross-domain representation under limited access to cross-domain data and achieves great zero-shot generalization ability in unseen domains. We verify PVA on a vision-based autonomous driving task with CARLA simulator. Experiments show that the agent generalizes well on unseen domains under limited access to multi-domain data.

</details>

---

## 47. Differentially Private Representation Learning via Image Captioning

- [ ] Differentially Private Representation Learning via Image Captioning | https://icml.cc/virtual/2024/poster/34185

- **Link**: https://icml.cc/virtual/2024/poster/34185

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Differentially private (DP) machine learning is considered the gold-standard solution for training a model from sensitive data while still preserving privacy. However, a major barrier to achieving this ideal is its sub-optimal privacy-accuracy trade-off, which is particularly visible in DP representation learning. Specifically, it has been shown that under modest privacy budgets, most models learn representations that are not significantly better than hand-crafted features. In this work, we show that effective DP representation learning can be done via image captioning and scaling up to internet-scale multimodal datasets. Through a series of engineering tricks, we successfully train a DP image captioner (DP-Cap) on a 233M subset of LAION-2B from scratch using a reasonable amount of computation, and obtaining unprecedented high-quality image features that can be used in a variety of downstream vision and vision-language tasks. For example, under a privacy budget of $\varepsilon=8$ for the LAION dataset, a linear classifier trained on top of learned DP-Cap features attains $65.8\%$ accuracy on ImageNet-1K, considerably improving the previous SOTA of $56.5\%$. Our work challenges the prevailing sentiment that high-utility DP representation learning cannot be achieved by training from scratch.

</details>

---

## 48. Improving Context Understanding in Multimodal Large Language Models via Multimodal Composition Learning

- [ ] Improving Context Understanding in Multimodal Large Language Models via Multimodal Composition Learning | https://icml.cc/virtual/2024/poster/34189

- **Link**: https://icml.cc/virtual/2024/poster/34189

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Previous efforts using frozen Large Language Models (LLMs) for visual understanding, via image captioning or image-text retrieval tasks, face challenges when dealing with complex multimodal scenarios. In order to enhance the capabilities of Multimodal Large Language Models (MLLM) in comprehending the context of vision and language, we introduce Multimodal Composition Learning (MCL) for the purpose of mapping or aligning the vision and language input. In particular, we introduce two tasks: Multimodal-Context Captioning (MC-Cap) and Multimodal-Context Retrieval (MC-Ret) to guide a frozen LLM in comprehending the vision and language context. These specialized tasks are crafted to improve the LLM’s capacity for efficient processing and utilization of multimodal inputs, thereby enhancing its proficiency in generating more accurate text or visual representations. Extensive experiments on both retrieval tasks (i.e., zero-shot composed image retrieval, visual storytelling image retrieval and visual dialog image retrieval) and text generation tasks (i.e., visual question answering) demonstrate the effectiveness of the proposed method. The code is available at: https://github.com/dhg-wei/MCL.

</details>

---

## 49. NExT-GPT: Any-to-Any Multimodal LLM

- [ ] NExT-GPT: Any-to-Any Multimodal LLM | https://icml.cc/virtual/2024/poster/34200

- **Link**: https://icml.cc/virtual/2024/poster/34200

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While recently Multimodal Large Language Models (MM-LLMs) have made exciting strides, they mostly fall prey to the limitation of only input-side multimodal understanding, without the ability to produce content in multiple modalities. As we humans always perceive the world and communicate with people through various modalities, developing any-to-any MM-LLMs capable of accepting and delivering content in any modality becomes essential to human-level AI. To fill the gap, we present an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT. We connect an LLM with multimodal adaptors and different diffusion decoders, enabling NExT-GPT to perceive inputs and generate outputs in arbitrary combinations of text, image, video, and audio. By leveraging the existing well-trained high-performing encoders and decoders, NExT-GPT is tuned with only a small amount of parameter (1%) of certain projection layers, which not only benefits low-cost training but also facilitates convenient expansion to more potential modalities. Moreover, we introduce a modality-switching instruction tuning (MosIT) and manually curate a high-quality dataset for MosIT, based on which NExT-GPT is empowered with complex cross-modal semantic understanding and content generation. Overall, our research showcases the promising possibility of building a unified AI agent capable of modeling universal modalities, paving the way for more human-like AI research in the community.

</details>

---

## 50. DeCoOp: Robust Prompt Tuning with Out-of-Distribution Detection

- [ ] DeCoOp: Robust Prompt Tuning with Out-of-Distribution Detection | https://icml.cc/virtual/2024/poster/34225

- **Link**: https://icml.cc/virtual/2024/poster/34225

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs), such as CLIP, have demonstrated impressive zero-shot capabilities for various downstream tasks. Their performance can be further enhanced through few-shot prompt tuning methods. However, current studies evaluate the performance of learned prompts separately on base and new classes. This evaluation lacks practicality for real-world applications since downstream tasks cannot determine whether the data belongs to base or new classes in advance. In this paper, we explore a problem setting called O pen-world P rompt T uning (OPT), which involves tuning prompts on base classes and evaluating on a combination of base and new classes. By introducing De composed P rompt T uning framework (DePT), we theoretically demonstrate that OPT can be solved by incorporating out-of-distribution detection into prompt tuning, thereby enhancing the base-to-new discriminability. Based on DePT, we present a novel prompt tuning approach, namely, De composed Co ntext Op timization (DeCoOp), which introduces new-class detectors and sub-classifiers to further enhance the base-class and new-class discriminability. Experimental results on 11 benchmark datasets validate the effectiveness of DePT and demonstrate that DeCoOp outperforms current state-of-the-art methods, providing a significant 2% average accuracy improvement.

</details>

---

## 51. ArtWhisperer: A Dataset for Characterizing Human-AI Interactions in Artistic Creations

- [ ] ArtWhisperer: A Dataset for Characterizing Human-AI Interactions in Artistic Creations | https://icml.cc/virtual/2024/poster/34244

- **Link**: https://icml.cc/virtual/2024/poster/34244

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we investigate how people use text-to-image models to generate desired target images. To study this interaction, we created ArtWhisperer, an online game where users are given a target image and are tasked with iteratively finding a prompt that creates a similar-looking image as the target. Through this game, we recorded over 50,000 human-AI interactions; each interaction corresponds to one text prompt created by a user and the corresponding generated image. The majority of these are repeated interactions where a user iterates to find the best prompt for their target image, making this a unique sequential dataset for studying human-AI collaborations. In an initial analysis of this dataset, we identify several characteristics of prompt interactions and user strategies. People submit diverse prompts and are able to discover a variety of text descriptions that generate similar images. Interestingly, prompt diversity does not decrease as users find better prompts. We further propose a new metric to quantify AI model steerability using our dataset. We define steerability as the expected number of interactions required to adequately complete a task. We estimate this value by fitting a Markov chain for each target task and calculating the expected time to reach an adequate score. We quantify and compare AI steerability across different types of target images and two different models, finding that images of cities and nature are more steerable than artistic and fantasy images. We also evaluate popular vision-language models to assess their image understanding and ability to incorporate feedback. These findings provide insights into human-AI interaction behavior, present a concrete method of assessing AI steerability, and demonstrate the general utility of the ArtWhisperer dataset.

</details>

---

## 52. Machine Vision Therapy: Multimodal Large Language Models Can Enhance Visual Robustness via Denoising In-Context Learning

- [ ] Machine Vision Therapy: Multimodal Large Language Models Can Enhance Visual Robustness via Denoising In-Context Learning | https://icml.cc/virtual/2024/poster/34263

- **Link**: https://icml.cc/virtual/2024/poster/34263

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although pre-trained models such as Contrastive Language-Image Pre-Training (CLIP) show impressive generalization results, their robustness is still limited under Out-of-Distribution (OOD) scenarios. Instead of undesirably leveraging human annotation as commonly done, it is possible to leverage the visual understanding power of Multi-modal Large Language Models (MLLMs). However, MLLMs struggle with vision problems due to task incompatibility, thus hindering their effectiveness. In this paper, we propose to effectively leverage MLLMs via Machine Vision Therapy which aims to rectify erroneous predictions of specific vision models. By supervising vision models using MLLM predictions, visual robustness can be boosted in a nearly unsupervised manner. Moreover, we propose a Denoising In-Context Learning (DICL) strategy to solve the incompatibility issue. Concretely, by examining the noise probability of each example through a transition matrix, we construct an instruction containing a correct exemplar and a probable erroneous one, which enables MLLMs to detect and rectify the incorrect predictions of vision models. Under mild assumptions, we theoretically show that our DICL method is guaranteed to find the ground truth. Through extensive experiments on various OOD datasets, our method demonstrates powerful capabilities for enhancing visual robustness under many OOD scenarios.

</details>

---

## 53. Beyond Sole Strength: Customized Ensembles for Generalized Vision-Language Models

- [ ] Beyond Sole Strength: Customized Ensembles for Generalized Vision-Language Models | https://icml.cc/virtual/2024/poster/34282

- **Link**: https://icml.cc/virtual/2024/poster/34282

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning pre-trained vision-language models (VLMs), e.g., CLIP, for the open-world generalization has gained increasing popularity due to its practical value. However, performance advancements are limited when relying solely on intricate algorithmic designs for a single model, even one exhibiting strong performance, e.g., CLIP-ViT-B/16. This paper, for the first time, explores the collaborative potential of leveraging much weaker VLMs to enhance the generalization of a robust single model. The affirmative findings motivate us to address the generalization problem from a novel perspective, i.e., ensemble of pre-trained VLMs. We introduce three customized ensemble strategies, each tailored to one specific scenario. Firstly, we introduce the zero-shot ensemble, automatically adjusting the logits of different models based on their confidence when only pre-trained VLMs are available. Furthermore, for scenarios with extra few-shot samples, we propose the training-free and tuning ensemble, offering flexibility based on the availability of computing resources. The code is available at https://github.com/zhiheLu/Ensemble_VLM.git.

</details>

---

## 54. MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities

- [ ] MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities | https://icml.cc/virtual/2024/poster/34344

- **Link**: https://icml.cc/virtual/2024/poster/34344

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose MM-Vet, an evaluation benchmark that examines large multimodal models (LMMs) on complicated multimodal tasks. Recent LMMs have shown various intriguing abilities, such as solving math problems written on the blackboard, reasoning about events and celebrities in news images, and explaining visual jokes. Rapid model advancements pose challenges to evaluation benchmark development. Problems include: (1) How to systematically structure and evaluate the complicated multimodal tasks; (2) How to design evaluation metrics that work well across question and answer types; and (3) How to give model insights beyond a simple performance ranking. To this end, we present MM-Vet, designed based on the insight that the intriguing ability to solve complicated tasks is often achieved by a generalist model being able to integrate different core vision-language (VL) capabilities. MM-Vet defines 6 core VL capabilities and examines the 16 integrations of interest derived from the capability combination. For evaluation metrics, we propose an LLM-based evaluator for open-ended outputs. The evaluator enables the evaluation across different question types and answer styles, resulting in a unified scoring metric. We evaluate representative LMMs on MM-Vet, providing insights into the capabilities of different LMM system paradigms and models.

</details>

---

## 55. Visual-Text Cross Alignment: Refining the Similarity Score in Vision-Language Models

- [ ] Visual-Text Cross Alignment: Refining the Similarity Score in Vision-Language Models | https://icml.cc/virtual/2024/poster/34359

- **Link**: https://icml.cc/virtual/2024/poster/34359

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

It has recently been discovered that using a pre-trained vision-language model (VLM), e.g., CLIP, to align a whole query image with several finer text descriptions generated by a large language model can significantly enhance zero-shot performance. However, in this paper, we empirically find that the finer descriptions tend to align more effectively with local areas of the query image rather than the whole image, and then we theoretically validate this finding. Thus, we present a method called weighted visual-text cross alignment (WCA). This method begins with a localized visual prompting technique, designed to identify local visual areas within the query image. The local visual areas are then cross-aligned with the finer descriptions by creating a similarity matrix using the pre-trained VLM. To determine how well a query image aligns with each category, we develop a score function based on the weighted similarities in this matrix. Extensive experiments demonstrate that our method significantly improves zero-shot performance across various datasets, achieving results that are even comparable to few-shot learning methods.

</details>

---

## 56. Revisiting the Role of Language Priors in Vision-Language Models

- [ ] Revisiting the Role of Language Priors in Vision-Language Models | https://icml.cc/virtual/2024/poster/34400

- **Link**: https://icml.cc/virtual/2024/poster/34400

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) are impactful in part because they can be applied to a variety of visual understanding tasks in a zero-shot fashion, without any fine-tuning. We study $\textit{generative VLMs}$ that are trained for next-word generation given an image. We explore their zero-shot performance on the illustrative task of image-text retrieval across nine popular vision-language benchmarks. Our first observation is that they can be repurposed for discriminative tasks (such as image-text retrieval) by simply computing the match score of generating a particular text string given an image. We call this probabilistic score the Visual Generative Pre-Training Score (VisualGPTScore). While the VisualGPTScore produces near-perfect accuracy on some retrieval benchmarks, it yields poor accuracy on others. We analyze this behavior through a probabilistic lens, pointing out that some benchmarks inadvertently capture unnatural language distributions by creating adversarial but unlikely text captions. In fact, we demonstrate that even a "blind" language model that ignores any image evidence can sometimes outperform all prior art, reminiscent of similar challenges faced by the visual-question answering (VQA) community many years ago. We derive a probabilistic post-processing scheme that controls for the amount of linguistic bias in generative VLMs at test time without having to retrain or fine-tune the model. We show that the VisualGPTScore, when appropriately debiased, is a strong zero-shot baseline for vision-language understanding, oftentimes producing state-of-the-art accuracy.

</details>

---

## 57. Fool Your (Vision and) Language Model with Embarrassingly Simple Permutations

- [ ] Fool Your (Vision and) Language Model with Embarrassingly Simple Permutations | https://icml.cc/virtual/2024/poster/34427

- **Link**: https://icml.cc/virtual/2024/poster/34427

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language and vision-language models are rapidly being deployed in practice thanks to their impressive capabilities in instruction following, in-context learning, and so on. This raises an urgent need to carefully analyse their robustness so that stakeholders can understand if and when such models are trustworthy enough to be relied upon in any given application. In this paper, we highlight a specific vulnerability in popular models, namely permutation sensitivity in multiple-choice question answering (MCQA). Specifically, we show empirically that popular models are vulnerable to adversarial permutation in answer sets for multiple-choice prompting, which is surprising as models should ideally be as invariant to prompt permutation as humans are. These vulnerabilities persist across various model sizes, and exist in very recent language and vision-language models. Code to reproduce all experiments is provided in supplementary materials.

</details>

---

## 58. LCA-on-the-Line: Benchmarking Out of Distribution Generalization with Class Taxonomies

- [ ] LCA-on-the-Line: Benchmarking Out of Distribution Generalization with Class Taxonomies | https://icml.cc/virtual/2024/poster/34465

- **Link**: https://icml.cc/virtual/2024/poster/34465

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We tackle the challenge of predicting models' Out-of-Distribution (OOD) performance using in-distribution (ID) measurements without requiring OOD data. Existing evaluations with ``Effective robustness'', which use ID accuracy as an indicator of OOD accuracy, encounter limitations when models are trained with diverse supervision and distributions, such as class labels (*Vision Models, VMs, on ImageNet*) and textual descriptions (*Visual-Language Models, VLMs, on LAION*). VLMs often generalize better to OOD data than VMs despite having similar or lower ID performance. To improve the prediction of models' OOD performance from ID measurements, we introduce the *Lowest Common Ancestor (LCA)-on-the-Line* framework. This approach revisits the established concept of LCA distance, which measures the hierarchical distance between labels and predictions within a predefined class hierarchy, such as WordNet. We assess 75 models using ImageNet as the ID dataset and five significantly shifted OOD variants, uncovering a strong linear correlation between ID LCA distance and OOD top-1 accuracy. Our method provides a compelling alternative for understanding why VLMs tend to generalize better. Additionally, we propose a technique to construct a taxonomic hierarchy on any dataset using $K$-means clustering, demonstrating that LCA distance is robust to the constructed taxonomic hierarchy. Moreover, we demonstrate that aligning model predictions with class taxonomies, through soft labels or prompt engineering, can enhance model generalization. Open source code in our [Project Page](https://elvishelvis.github.io/papers/lca/).

</details>

---

## 59. Retrieval Across Any Domains via Large-scale Pre-trained Model

- [ ] Retrieval Across Any Domains via Large-scale Pre-trained Model | https://icml.cc/virtual/2024/poster/34499

- **Link**: https://icml.cc/virtual/2024/poster/34499

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In order to enhance the generalization ability towards unseen domains, universal cross-domain image retrieval methods require a training dataset encompassing diverse domains, which is costly to assemble. Given this constraint, we introduce a novel problem of data-free adaptive cross-domain retrieval, eliminating the need for real images during training. Towards this goal, we propose a novel Text-driven Knowledge Integration (TKI) method, which exclusively utilizes a pre-trained vision-language model to implement an ``aggregation after expansion" training strategy. Specifically, we extract diverse implicit domain-specific information through a set of learnable domain word vectors. Subsequently, a domain-agnostic universal projection, equipped with a non-Euclidean multi-layer perceptron, can be optimized using these assorted text descriptions through the text-proxied domain aggregation. Leveraging the cross-modal transferability phenomenon of the shared latent space, we can integrate the trained domain-agnostic universal projection with the pre-trained visual encoder to extract the features of the input image for the following retrieval during testing. Extensive experimental results on several benchmark datasets demonstrate the superiority of our method.

</details>

---

## 60. Memory-Space Visual Prompting for Efficient Vision-Language Fine-Tuning

- [ ] Memory-Space Visual Prompting for Efficient Vision-Language Fine-Tuning | https://icml.cc/virtual/2024/poster/34543

- **Link**: https://icml.cc/virtual/2024/poster/34543

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current solutions for efficiently constructing large vision-language (VL) models follow a two-step paradigm: projecting the output of pre-trained vision encoders to the input space of pre-trained language models as visual prompts; and then transferring the models to downstream VL tasks via end-to-end parameter-efficient fine-tuning (PEFT). However, this paradigm still exhibits inefficiency since it significantly increases the input length of the language models. In this paper, in contrast to integrating visual prompts into inputs, we regard visual prompts as additional knowledge that facilitates language models in addressing tasks associated with visual information. Motivated by the finding that Feed-Forward Network (FFN) of language models acts as "key-value memory", we introduce a novel approach termed memory-space visual prompting (MemVP), wherein visual prompts are concatenated with the weights of FFN for visual knowledge injection. Experimental results across various VL tasks and language models reveal that MemVP significantly reduces the training time and inference latency of the finetuned VL models and surpasses the performance of previous PEFT methods.

</details>

---

## 61. Libra: Building Decoupled Vision System on Large Language Models

- [ ] Libra: Building Decoupled Vision System on Large Language Models | https://icml.cc/virtual/2024/poster/34554

- **Link**: https://icml.cc/virtual/2024/poster/34554

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we introduce Libra , a prototype model with a decoupled vision system on a large language model (LLM). The decoupled vision system decouples inner-modal modeling and cross-modal interaction, yielding unique visual information modeling and effective cross-modal comprehension. Libra is trained through discrete auto-regressive modeling on both vision and language inputs. Specifically, we incorporate a routed visual expert with a cross-modal bridge module into a pretrained LLM to route the vision and language flows during attention computing to enable different attention patterns in inner-modal modeling and cross-modal interaction scenarios. Experimental results demonstrate that the dedicated design of Libra achieves a strong MLLM baseline that rivals existing works in the image-to-text scenario with merely 50 million training data, providing a new perspective for future multimodal foundation models. Code is available at https://github.com/YifanXu74/Libra.

</details>

---

## 62. 3D-VLA: A 3D Vision-Language-Action Generative World Model

- [ ] 3D-VLA: A 3D Vision-Language-Action Generative World Model | https://icml.cc/virtual/2024/poster/34575

- **Link**: https://icml.cc/virtual/2024/poster/34575

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language-action (VLA) models rely on 2D inputs, lacking integration with the broader realm of the 3D physical world. Furthermore, they perform action prediction by learning a direct mapping from perception to action, neglecting the vast dynamics of the world and the relations between actions and dynamics. In contrast, human beings are endowed with world models that depict imagination about future scenarios to plan action accordingly. To this end, we propose 3D-VLA by introducing a new family of embodied foundation models that seamlessly link 3D perception, reasoning, and action through a generative world model. Specifically, 3D-VLA is built on top of a 3D-based large language model (LLM) and a set of action tokens is introduced to engage with the embodied environment. Furthermore, to inject generation abilities into the model, we train the embodied diffusion models and align them into the LLM for predicting the goal image and point cloud. To train our 3D-VLA, we curate a large-scale 3D embodied instruction dataset by extracting vast 3D-related information from existing robotics datasets. Our experiments on held-in datasets demonstrate that 3D-VLA significantly improves the reasoning, multimodality generation and planning capabilities in embodied environments, showcasing its potential in real-world applications.

</details>

---

## 63. HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding

- [ ] HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding | https://icml.cc/virtual/2024/poster/34578

- **Link**: https://icml.cc/virtual/2024/poster/34578

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While large vision-language models (LVLMs) have demonstrated impressive capabilities in interpreting multi-modal contexts, they invariably suffer from object hallucinations (OH). We introduce HALC, a novel decoding algorithm designed to mitigate OH in LVLMs. HALC leverages distinct fine-grained optimal visual information in vision-language tasks and operates on both local and global contexts simultaneously. Specifically, HALC integrates a robust auto-focal grounding mechanism (locally) to correct hallucinated tokens on the fly, and a specialized beam search algorithm (globally) to significantly reduce OH while preserving text generation quality. Additionally, HALC can be integrated into any LVLMs as a plug-and-play module without extra training. Extensive experimental studies demonstrate HALC’s effectiveness in reducing OH, outperforming state-of-the-arts across four benchmarks. Code is released at https://github.com/BillChan226/HALC.

</details>

---

## 64. Failures Are Fated, But Can Be Faded: Characterizing and Mitigating Unwanted Behaviors in Large-Scale Vision and Language Models

- [ ] Failures Are Fated, But Can Be Faded: Characterizing and Mitigating Unwanted Behaviors in Large-Scale Vision and Language Models | https://icml.cc/virtual/2024/poster/34614

- **Link**: https://icml.cc/virtual/2024/poster/34614

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In large deep neural networks that seem to perform surprisingly well on many tasks, we also observe a few failures related to accuracy, social biases, and alignment with human values, among others. Therefore, before deploying these models, it is crucial to characterize this failure landscape for engineers to debug and legislative bodies to audit models. Nevertheless, it is infeasible to exhaustively test for all possible combinations of factors that could lead to a model's failure. In this paper, we introduce a post-hoc method that utilizes deep reinforcement learning to explore and construct the landscape of failure modes in pre-trained discriminative and generative models. With the aid of limited human feedback, we then demonstrate how to restructure the failure landscape to be more desirable by moving away from the discovered failure modes. We empirically show the effectiveness of the proposed method across common Computer Vision, Natural Language Processing, and Vision-Language tasks.

</details>

---

## 65. Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs

- [ ] Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs | https://icml.cc/virtual/2024/poster/34618

- **Link**: https://icml.cc/virtual/2024/poster/34618

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models have exhibit exceptional performance in text-to-image generation and editing. However, existing methods often face challenges when handling complex text prompts that involve multiple objects with multiple attributes and relationships. In this paper, we propose a brand new training-free text-to-image generation/editing framework, namely Recaption, Plan and Generate (RPG), harnessing the powerful chain-of-thought reasoning ability of multimodal LLMs to enhance the compositionality of text-to-image diffusion models. Our approach employs the MLLM as a global planner to decompose the process of generating complex images into multiple simpler generation tasks within subregions. We propose complementary regional diffusion to enable region-wise compositional generation. Furthermore, we integrate text-guided image generation and editing within the proposed RPG in a closed-loop fashion, thereby enhancing generalization ability. Extensive experiments demonstrate our RPG outperforms state-of-the-art text-to-image models, including DALL-E 3 and SDXL, particularly in multi-category object composition and text-image semantic alignment. Notably, our RPG framework exhibits wide compatibility with various MLLM architectures and diffusion backbones. Our code is available at https://github.com/YangLing0818/RPG-DiffusionMaster

</details>

---

## 66. Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast

- [ ] Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast | https://icml.cc/virtual/2024/poster/34623

- **Link**: https://icml.cc/virtual/2024/poster/34623

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A multimodal large language model (MLLM) agent can receive instructions, capture images, retrieve histories from memory, and decide which tools to use. Nonetheless, red-teaming efforts have revealed that adversarial images/prompts can jailbreak an MLLM and cause unaligned behaviors. In this work, we report an even more severe safety issue in multi-agent environments, referred to as infectious jailbreak. It entails the adversary simply jailbreaking a single agent, and without any further intervention from the adversary, (almost) all agents will become infected exponentially fast and exhibit harmful behaviors. To validate the feasibility of infectious jailbreak, we simulate multi-agent environments containing up to one million LLaVA-1.5 agents, and employ randomized pair-wise chat as a proof-of-concept instantiation for multi-agent interaction. Our results show that feeding an (infectious) adversarial image into the memory of any randomly chosen agent is sufficient to achieve infectious jailbreak. Finally, we derive a simple principle for determining whether a defense mechanism can provably restrain the spread of infectious jailbreak, but how to design a practical defense that meets this principle remains an open question to investigate.

</details>

---

## 67. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior

- [ ] Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior | https://icml.cc/virtual/2024/poster/34683

- **Link**: https://icml.cc/virtual/2024/poster/34683

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.

</details>

---

## 68. CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers

- [ ] CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers | https://icml.cc/virtual/2024/poster/34682

- **Link**: https://icml.cc/virtual/2024/poster/34682

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language models have achieved tremendous advances. However, their computational costs are also escalating dramatically, making model acceleration exceedingly critical. To pursue more efficient vision-language Transformers, this paper introduces Cross-Guided Ensemble of Tokens (CrossGET), a general acceleration framework for vision-language Transformers. This framework adaptively combines tokens in real-time during inference, significantly reducing computational costs while maintaining high performance. CrossGET features two primary innovations: 1) Cross-Guided Matching and Ensemble. CrossGET leverages cross-modal guided token matching and ensemble to effectively utilize cross-modal information, achieving wider applicability across both modality-independent models, e.g., CLIP, and modality-dependent ones, e.g., BLIP2. 2) Complete-Graph Soft Matching. CrossGET introduces an algorithm for the token-matching mechanism, ensuring reliable matching results while facilitating parallelizability and high efficiency. Extensive experiments have been conducted on various vision-language tasks, such as image-text retrieval, visual reasoning, image captioning, and visual question answering. The performance on both classic multimodal architectures and emerging multimodal LLMs demonstrates the framework's effectiveness and versatility. The code is available at https://github.com/sdc17/CrossGET.

</details>

---

## 69. FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning

- [ ] FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning | https://icml.cc/virtual/2024/poster/34712

- **Link**: https://icml.cc/virtual/2024/poster/34712

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we investigate how to leverage pre-trained visual-language models (VLM) for online Reinforcement Learning (RL). In particular, we focus on sparse reward tasks with pre-defined textual task descriptions. We first identify the problem of reward misalignment when applying VLM as a reward in RL tasks. To address this issue, we introduce a lightweight fine-tuning method, named Fuzzy VLM reward-aided RL (FuRL), based on reward alignment and relay RL. Specifically, we enhance the performance of SAC/DrQ baseline agents on sparse reward tasks by fine-tuning VLM representations and using relay RL to avoid local minima. Extensive experiments on the Meta-world benchmark tasks demonstrate the efficacy of the proposed method. Code is available at: https://github.com/fuyw/FuRL.

</details>

---

## 70. Image Hijacks: Adversarial Images can Control Generative Models at Runtime

- [ ] Image Hijacks: Adversarial Images can Control Generative Models at Runtime | https://icml.cc/virtual/2024/poster/34839

- **Link**: https://icml.cc/virtual/2024/poster/34839

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Are foundation models secure against malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control the behaviour of VLMs at inference time, and introduce the general Behaviour Matching algorithm for training image hijacks. From this, we derive the Prompt Matching method, allowing us to train hijacks matching the behaviour of an arbitrary user-defined text prompt (e.g. 'the Eiffel Tower is now located in Rome') using a generic, off-the-shelf dataset unrelated to our choice of prompt. We use Behaviour matching to craft hijacks for four types of attack: forcing VLMs to generate outputs of the adversary’s choice, leak information from their context window, override their safety training, and believe false statements. We study these attacks against LLaVA, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all attack types achieve a success rate of over 80%. Moreover, our attacks are automated and require only small image perturbations.

</details>

---

## 71. Unlocking the Power of Spatial and Temporal Information in Medical Multimodal Pre-training

- [ ] Unlocking the Power of Spatial and Temporal Information in Medical Multimodal Pre-training | https://icml.cc/virtual/2024/poster/34857

- **Link**: https://icml.cc/virtual/2024/poster/34857

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Medical vision-language pre-training methods mainly leverage the correspondence between paired medical images and radiological reports. Although multi-view spatial images and temporal sequences of image-report pairs are available in off-the-shelf multi-modal medical datasets, most existing methods have not thoroughly tapped into such extensive supervision signals. In this paper, we introduce the Med-ST framework for fine-grained spatial and temporal modeling to exploit information from multiple spatial views of chest radiographs and temporal historical records. For spatial modeling, Med-ST employs the Mixture of View Expert (MoVE) architecture to integrate different visual features from both frontal and lateral views. To achieve a more comprehensive alignment, Med-ST not only establishes the global alignment between whole images and texts but also introduces modality-weighted local alignment between text tokens and spatial regions of images. For temporal modeling, we propose a novel cross-modal bidirectional cycle consistency objective by forward mapping classification (FMC) and reverse mapping regression (RMR). By perceiving temporal information from simple to complex, Med-ST can learn temporal semantics. Experimental results across four distinct tasks demonstrate the effectiveness of Med-ST, especially in temporal classification tasks. Our code and model are available at https://github.com/SVT-Yang/MedST.

</details>

---

## 72. Code as Reward: Empowering Reinforcement Learning with VLMs

- [ ] Code as Reward: Empowering Reinforcement Learning with VLMs | https://icml.cc/virtual/2024/poster/34923

- **Link**: https://icml.cc/virtual/2024/poster/34923

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained Vision-Language Models (VLMs) are able to understand visual concepts, describe and decompose complex tasks into sub-tasks, and provide feedback on task completion. In this paper, we aim to leverage these capabilities to support the training of reinforcement learning (RL) agents. In principle, VLMs are well suited for this purpose, as they can naturally analyze image-based observations and provide feedback (reward) on learning progress. However, inference in VLMs is computationally expensive, so querying them frequently to compute rewards would significantly slowdown the training of an RL agent. To address this challenge, we propose a framework named Code as Reward (VLM-CaR). VLM-CaR produces dense reward functions from VLMs through code generation, thereby significantly reducing the computational burden of querying the VLM directly. We show that the dense rewards generated through our approach are very accurate across a diverse set of discrete and continuous environments, and can be more effective in training RL policies than the original sparse environment rewards.

</details>

---

## 73. Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models

- [ ] Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models | https://icml.cc/virtual/2024/poster/34932

- **Link**: https://icml.cc/virtual/2024/poster/34932

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visually-conditioned language models (VLMs) have seen growing adoption in applications such as visual dialogue, scene understanding, and robotic task planning; adoption that has fueled a wealth of new models such as LLaVa, InstructBLIP, and PaLI-3. Despite the volume of new releases, key design decisions around image preprocessing, architecture, and optimization are under-explored, making it challenging to understand what factors account for model performance – a challenge further complicated by the lack of objective, consistent evaluations. To address these gaps, we first compile a suite of standardized evaluations spanning visual question answering, object localization, and challenge sets that probe properties such as hallucination; evaluations that provide fine-grained insight VLM capabilities. Second, we rigorously investigate VLMs along key design axes, including pretrained visual representations and training from base vs. instruct-tuned language models, amongst others. We couple our analysis with three resource contributions: (1) a unified framework for evaluating VLMs, (2) optimized, flexible training code, and (3) checkpoints for all models, including a family of VLMs at the 7-13B scale that strictly outperform InstructBLIP and LLaVa v1.5, the state-of-the-art in open VLMs.

</details>

---

## 74. Improving fine-grained understanding in image-text pre-training

- [ ] Improving fine-grained understanding in image-text pre-training | https://icml.cc/virtual/2024/poster/34962

- **Link**: https://icml.cc/virtual/2024/poster/34962

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce SPARse fine-grained Contrastive alignment (SPARC), a simple method for pretraining more fine-grained multimodal representations from image-text pairs. Given that multiple image patches often correspond to single words, we propose to learn a grouping of image patches for every token in the caption. To achieve this, we use a sparse similarity metric between image patches and language tokens and compute for each token a language-grouped vision embedding as the weighted average of patches. The token and language-grouped vision embeddings are then contrasted through a fine-grained sequence-wise loss that only depends on individual samples and does not require other batch samples as negatives, i.e., more detailed information is encoded in a computationally inexpensive way. SPARC combines this fine-grained loss with a contrastive loss between global image and text embeddings to learn representations that simultaneously encode global and local information. We thoroughly evaluate SPARC and show improved performance over competing approaches both on image-level tasks relying on coarse-grained information, e.g. classification, as well as region-level tasks relying on fine-grained information, e.g., retrieval, object detection, segmentation while also improving model faithfulness and captioning in foundational vision-language models.

</details>

---

## 75. Leveraging VLM-Based Pipelines to Annotate 3D Objects

- [ ] Leveraging VLM-Based Pipelines to Annotate 3D Objects | https://icml.cc/virtual/2024/poster/34981

- **Link**: https://icml.cc/virtual/2024/poster/34981

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pretrained vision language models (VLMs) present an opportunity to caption unlabeled 3D objects at scale. The leading approach to summarize VLM descriptions from different views of an object (Luo et al., 2023) relies on a language model (GPT4) to produce the final output. This text-based aggregation is susceptible to hallucinations as it merges potentially contradictory descriptions. We propose an alternative algorithm to marginalize over factors such as the viewpoint that affect the VLM's response. Instead of merging text-only responses, we utilize the VLM's joint image-text likelihoods. We show our probabilistic aggregation is not only more reliable and efficient, but sets the SoTA on inferring object types with respect to human-verified labels. The aggregated annotations are also useful for conditional inference; they improve downstream predictions (e.g., of object material) when the object’s type is specified as an auxiliary text-based input. Such auxiliary inputs allow ablating the contribution of visual reasoning over visionless reasoning in an unsupervised setting. With these supervised and unsupervised evaluations, we show how a VLM-based pipeline can be leveraged to produce reliable annotations for 764K objects from the Objaverse dataset.

</details>

---

## 76. PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs

- [ ] PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs | https://icml.cc/virtual/2024/poster/35217

- **Link**: https://icml.cc/virtual/2024/poster/35217

- **Conference**: ICML

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) have shown impressive capabilities across a variety of tasks, from logical reasoning to visual understanding. This opens the door to richer interaction with the world, for example robotic control. However, VLMs produce only textual outputs, while robotic control and other spatial tasks require outputting continuous coordinates, actions, or trajectories. How can we enable VLMs to handle such settings without fine-tuning on task-specific data? In this paper, we propose a novel visual prompting approach for VLMs that we call Prompting with Iterative Visual Optimization (PIVOT), which casts tasks as iterative visual question answering. In each iteration, the image is annotated with a visual representation of proposals that the VLM can refer to (e.g., candidate robot actions, localizations, or trajectories). The VLM then selects the best ones for the task. These proposals are iteratively refined, allowing the VLM to eventually zero in on the best available answer. We investigate PIVOT on real-world robotic navigation, real-world manipulation from images, instruction following in simulation, and additional spatial inference tasks such as localization. We find, perhaps surprisingly, that our approach enables zero-shot control of robotic systems without any robot training data, navigation in a variety of environments, and other capabilities. Although current performance is far from perfect, our work highlights potentials and limitations of this new regime and shows a promising approach for Internet-Scale VLMs in robotic and spatial reasoning domains.

</details>

---

