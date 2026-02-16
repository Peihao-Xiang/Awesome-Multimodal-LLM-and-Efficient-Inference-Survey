# IJCAI 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_ijcai2025_papers.csv

## 1. Hallucination Reduction in Video-Language Models via Hierarchical Multimodal Consistency

- [ ] Hallucination Reduction in Video-Language Models via Hierarchical Multimodal Consistency | https://www.ijcai.org/proceedings/2025/1019

- **Link**: https://www.ijcai.org/proceedings/2025/1019

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of large language models (LLMs) has led to the widespread adoption of video-language models (VLMs) across various domains. However, VLMs are often hindered by their limited semantic discrimination capability, exacerbated by the limited diversity and biased sample distribution of most video-language datasets. This limitation results in a biased understanding of the semantics between visual concepts, leading to hallucinations. To address this challenge, we propose a Multi-level Multimodal Alignment (MMA) framework that leverages a text encoder and semantic discriminative loss to achieve multi-level alignment. This enables the model to capture both low-level and high-level semantic relationships, thereby reducing hallucinations. By incorporating language-level alignment into the training process, our approach ensures stronger semantic consistency between video and textual modalities. Furthermore, we introduce a two-stage progressive training strategy that exploits larger and more diverse datasets to enhance semantic alignment and better capture general semantic relationships between visual and textual modalities. Our comprehensive experiments demonstrate that the proposed MMA method significantly mitigates hallucinations and achieves state-of-the-art performance across multiple video-language tasks, establishing a new benchmark in the field.

</details>

---

## 2. REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLMs

- [ ] REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLMs | https://www.ijcai.org/proceedings/2025/1081

- **Link**: https://www.ijcai.org/proceedings/2025/1081

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.

We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate (16.55%) while Qwen2-VL showed the highest MT refusal rate (19.1%).

</details>

---

## 3. AI-Assisted Triage and Decision Support of Head and Neck Cancer Screening and Diagnosis in Low-Resourced Settings

- [ ] AI-Assisted Triage and Decision Support of Head and Neck Cancer Screening and Diagnosis in Low-Resourced Settings | https://www.ijcai.org/proceedings/2025/1087

- **Link**: https://www.ijcai.org/proceedings/2025/1087

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The mortality burden of head and neck cancer (HNC) is increasing globally and disproportionately affects people in low-and middle-income countries with limited medical workforce. To address this issue, artificial intelligence (AI) algorithms are increasingly being explored to process medical imaging data, demonstrating competitive performance. However, the clinical adoption of AI remains challenging as clinicians struggle to understand how complex AI works and trust it to use in practice. In addition, AI may not perform well on varying data qualities of endoscopy videos for HNC screening and diagnosis from multiple sites. 

In this project, our international and interdisciplinary team will collaborate with clinicians from multiple sites (e.g. Singapore, the U.S., and Bangladesh) to collect a diverse, multi-site dataset. In addition, we aim to design and develop computational techniques and practices to improve collaborations between clinicians and AI for the triage and diagnosis of HNC. Specifically, these techniques include a YOLOv5-based glottis detector, a classifier of patient's status using clinical endoscopy videos, uncertainty quantification techniques, and interactive Vision Language Model-based AI explanations, which will enable clinicians to understand AI outputs and provide their inputs to improve AI. After developing our system, we will evaluate the effectiveness of these computational techniques in enabling AI-assisted point-of-care triage and decision-support for HNC, particularly in resource-limited settings.

</details>

---

## 4. ContextAware: A Multi-Agent Framework for Detecting Harmful Image-Based Comments on Social Media

- [ ] ContextAware: A Multi-Agent Framework for Detecting Harmful Image-Based Comments on Social Media | https://www.ijcai.org/proceedings/2025/1103

- **Link**: https://www.ijcai.org/proceedings/2025/1103

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Detecting hidden stigmatization in social media poses significant challenges due to semantic misalignments between textual and visual modalities, as well as the subtlety of implicit stigmatization. Traditional approaches often fail to capture these complexities in real-world, multimodal content. To address this gap, we introduce ContextAware, an agent-based framework that leverages specialized modules to collaboratively process and analyze images, textual context, and social interactions. Our approach begins by clustering image embeddings to identify recurring content, activating high-likes agents for deeper analysis of images receiving substantial user engagement, while comprehensive agents handle lower-engagement images. By integrating case-based learning, textual sentiment, and vision-language models (VLMs), ContextAware refines its detection of harmful content. We evaluate ContextAware on a self-collected Douyin dataset focused on interracial relationships, comprising 871 short videos and 885,502 comments—of which a notable portion are image-based. Experimental results show that ContextAware not only outperforms state-of-the-art methods in accuracy and F1 score but also effectively detects implicit stigmatization within the highly contextual environment of social media. Our findings underscore the importance of agent-based architectures and multimodal alignment in capturing nuanced, culturally specific forms of harmful content.

</details>

---

## 5. Reliable and Diverse Hierarchical Adapter for Zero-shot Video Classification

- [ ] Reliable and Diverse Hierarchical Adapter for Zero-shot Video Classification | https://www.ijcai.org/proceedings/2025/115

- **Link**: https://www.ijcai.org/proceedings/2025/115

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adapting pre-trained vision-language models to downstream tasks has emerged as a novel paradigm for zero-shot learning. Existing test-time adaptation (TTA) methods such as TPT attempt to fine-tune visual or textual representations to accommodate downstream tasks but still require expensive optimization costs. To this end, Training-free Dynamic Adapter (TDA) maintains a cache containing visual features for each category in a parameter-free manner and measures sample confidence based on prediction entropy of test samples. Inspired by TDA, this work aims to develop the first training-free adapter for zero-shot video classification. Capturing the intrinsic temporal relationships within video data to construct and maintain the video cache is key to extending TDA to the video domain. In this work, we propose a reliable and diverse Hierarchical Adapter for zero-shot video classification, which consists of Frame-level Cache Refiner and Video-level Cache Updater. Before each video sample enters the corresponding cache, it needs to be refined at frame level based on prediction entropy and temporal probability difference. Due to the limited capacity of the cache, we update the cache during inference based on the principle of diversity. Experiments on four popular video classification benchmarks demonstrate the effectiveness of Hierarchical Adapter. The code is available at https://github.com/Gwxer/Hierarchical-Adapter.

</details>

---

## 6. Words Over Pixels? Rethinking Vision in Multimodal Large Language Models

- [ ] Words Over Pixels? Rethinking Vision in Multimodal Large Language Models | https://www.ijcai.org/proceedings/2025/1164

- **Link**: https://www.ijcai.org/proceedings/2025/1164

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) promise seamless integration of vision and language understanding. However, despite their strong performance, recent studies reveal that MLLMs often fail to effectively utilize visual information, frequently relying on textual cues instead. This survey provides a comprehensive analysis of the vision component in MLLMs, covering both application-level and architectural aspects. We investigate critical challenges such as weak spatial reasoning, poor fine-grained visual perception, and suboptimal fusion of visual and textual modalities. Additionally, we explore limitations in current vision encoders, benchmark inconsistencies, and their implications for downstream tasks. By synthesizing recent advancements, we highlight key research opportunities to enhance visual understanding, improve cross-modal alignment, and develop more robust and efficient MLLMs. Our observations emphasize the urgent need to elevate vision to an equal footing with language, paving the path for more reliable and perceptually aware multimodal models.

</details>

---

## 7. DDPA-3DVG: Vision-Language Dual-Decoupling and Progressive Alignment for 3D Visual Grounding

- [ ] DDPA-3DVG: Vision-Language Dual-Decoupling and Progressive Alignment for 3D Visual Grounding | https://www.ijcai.org/proceedings/2025/117

- **Link**: https://www.ijcai.org/proceedings/2025/117

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D visual grounding aims to localize target objects in point clouds based on free-form natural language, which often describes both target and reference objects. Effective alignment between visual and text features is crucial for this task. However, existing two-stage methods that rely solely on object-level features can yield suboptimal accuracy, while one-stage methods that align only point-level features can be prone to noise. In this paper, we propose DDPA-3DVG, a novel framework that progressively aligns visual locations and language descriptions at multiple granularities. Specifically, we decouple natural language descriptions into distinct representations of target objects, reference objects, and their mutual relationships, while disentangling 3D scenes into object-level, voxel-level, and point-level features. By progressively fusing these dual-decoupled features from coarse to fine, our method enhances cross-modal alignment and achieves state-of-the-art performance on three challenging benchmarks—ScanRefer, Nr3D, and Sr3D. The code will be released at https://github.com/HDU-VRLab/DDPA-3DVG.

</details>

---

## 8. Harnessing Vision Models for Time Series Analysis: A Survey

- [ ] Harnessing Vision Models for Time Series Analysis: A Survey | https://www.ijcai.org/proceedings/2025/1178

- **Link**: https://www.ijcai.org/proceedings/2025/1178

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Time series analysis has evolved from traditional autoregressive models to deep learning, Transformers, and Large Language Models (LLMs). While vision models have also been explored along the way, their contributions are less recognized due to the predominance of sequence modeling. However, challenges such as the mismatch between continuous time series and LLMs’ discrete token space, and the difficulty in capturing multivariate correlations, have led to growing interest in Large Vision Models (LVMs) and Vision-Language Models (VLMs). This survey highlights the advantages of vision models over LLMs in time series analysis, offering a comprehensive dual-view taxonomy that answers key research questions like how to encode time series as images and how to model imaged time series. Additionally, we address pre- and post-processing challenges in this framework and outline future directions for advancing the field.

</details>

---

## 9. Image Captioning Evaluation in the Age of Multimodal LLMs: Challenges and Future Perspectives

- [ ] Image Captioning Evaluation in the Age of Multimodal LLMs: Challenges and Future Perspectives | https://www.ijcai.org/proceedings/2025/1180

- **Link**: https://www.ijcai.org/proceedings/2025/1180

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The evaluation of machine-generated captions is a complex and evolving challenge. With the advent of Multimodal Large Language Models (MLLMs), image captioning has become a core task, increasing the need for robust and reliable evaluation metrics. This survey provides a comprehensive overview of advancements in image captioning evaluation, analyzing the evolution, strengths, and limitations of existing metrics. We assess these metrics across multiple dimensions, including correlation with human judgment, ranking accuracy, and sensitivity to hallucinations. Additionally, we explore the challenges posed by the longer and more detailed captions generated by MLLMs and examine the adaptability of current metrics to these stylistic variations. Our analysis highlights some limitations of standard evaluation approaches and suggests promising directions for future research in image captioning assessment. For a comprehensive overview of captioning evaluation refer to our project page available at https://github.com/aimagelab/awesome-captioning-evaluation.

</details>

---

## 10. The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning

- [ ] The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning | https://www.ijcai.org/proceedings/2025/1181

- **Link**: https://www.ijcai.org/proceedings/2025/1181

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reinforcement learning (RL) has shown impressive results in sequential decision-making tasks. Large Language Models (LLMs) and Vision-Language Models (VLMs) have recently emerged, exhibiting impressive capabilities in multimodal understanding and reasoning. These advances have led to a surge of research integrating LLMs and VLMs into RL. This survey reviews representative works in which LLMs and VLMs are used to overcome key challenges in RL, such as lack of prior knowledge, long-horizon planning, and reward design. We present a taxonomy that categorizes these LLM/VLM-assisted RL approaches into three roles: agent, planner, and reward. We conclude by exploring open problems, including grounding, bias mitigation, improved representations, and action advice. By consolidating existing research and identifying future directions, this survey establishes a framework for integrating LLMs and VLMs into RL, advancing approaches that unify natural language and visual understanding with sequential decision-making.

</details>

---

## 11. An Empirical Study of Federated Prompt Learning for Vision Language Model

- [ ] An Empirical Study of Federated Prompt Learning for Vision Language Model | https://www.ijcai.org/proceedings/2025/1188

- **Link**: https://www.ijcai.org/proceedings/2025/1188

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Vision Language Model (VLM) excels in aligning vision and language representations, and prompt learning has emerged as a key technique for adapting such models to downstream tasks. However, the application of prompt learning with VLM in federated learning (FL) scenarios remains underexplored. This paper systematically investigates the behavioral differences between language prompt learning (LPT) and vision prompt learning (VPT) under data heterogeneity challenges, including label skew and domain shift. We conduct extensive experiments to evaluate the impact of various FL and prompt configurations, such as client scale, aggregation strategies, and prompt length, to assess the robustness of Federated Prompt Learning (FPL). Furthermore, we explore strategies for enhancing prompt learning in complex scenarios where label skew and domain shift coexist, including leveraging both prompt types when computational resources allow. Our findings offer practical insights into optimizing prompt learning in federated settings, contributing to the broader deployment of VLMs in privacy-preserving environments.

</details>

---

## 12. Connector-S: A Survey of Connectors in Multi-modal Large Language Models

- [ ] Connector-S: A Survey of Connectors in Multi-modal Large Language Models | https://www.ijcai.org/proceedings/2025/1202

- **Link**: https://www.ijcai.org/proceedings/2025/1202

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancements in multi-modal large language models (MLLMs), connectors play a pivotal role in bridging diverse modalities and enhancing model performance. However, the design and evolution of connectors have not been comprehensively analyzed, leaving gaps in understanding how these components function and hindering the development of more powerful connectors. In this survey, we systematically review the current progress of connectors in MLLMs and present a structured taxonomy that categorizes connectors into atomic operations (mapping, compression, mixture of experts) and holistic designs (multi-layer, multi-encoder, multi-modal scenarios), highlighting their technical contributions and advancements. Furthermore, we discuss several promising research frontiers and challenges, including high-resolution input, dynamic compression, guide information selection, combination strategy, and interpretability. This survey is intended to serve as a foundational reference and a clear roadmap for researchers, providing valuable insights into the design and optimization of next-generation connectors to enhance the performance and adaptability of MLLMs.

</details>

---

## 13. Domain Prompt Learning with Quaternion Networks (Extended Abstract)

- [ ] Domain Prompt Learning with Quaternion Networks (Extended Abstract) | https://www.ijcai.org/proceedings/2025/1209

- **Link**: https://www.ijcai.org/proceedings/2025/1209

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Foundational vision-language models (VLMs) like CLIP have revolutionized image recognition, but adapting them to specialized domains with limited data remains challenging. We propose Domain Prompt Learning with Quaternion Networks (DPLQ), which leverages domain-specific foundation models and quaternion-based prompt tuning to effectively transfer recognition capabilities. Our method achieves state-of-the-art results in remote sensing and medical imaging tasks. This extended abstract highlights the key contributions and performance of DPLQ.

</details>

---

## 14. Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models

- [ ] Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models | https://www.ijcai.org/proceedings/2025/123

- **Link**: https://www.ijcai.org/proceedings/2025/123

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are experiencing rapid growth, yielding a plethora of novel works recently. The prevailing trend involves adopting data-driven methodologies, wherein diverse instruction-following datasets were collected. However, these approaches always face the challenge of limited visual perception capabilities, as they solely utilizing CLIP-like encoders to extract visual information from inputs. Though these encoders are pre-trained on billions of image-text pairs, they still grapple with the information loss dilemma, given that textual captions only partially capture the contents depicted in images. To address this limitation, this paper proposes to improve the visual perception ability of MLLMs through a mixture-of-experts knowledge enhancement mechanism. Specifically, this work introduces a novel method that incorporates multi-task encoders and existing visual tools into the MLLMs training and inference pipeline, aiming to provide a more comprehensive summarization of visual inputs. Extensive experiments have evaluated its effectiveness of advancing MLLMs, showcasing improved visual perception capability achieved through the integration of visual experts.

</details>

---

## 15. INT: Instance-Specific Negative Mining for Task-Generic Promptable Segmentation

- [ ] INT: Instance-Specific Negative Mining for Task-Generic Promptable Segmentation | https://www.ijcai.org/proceedings/2025/124

- **Link**: https://www.ijcai.org/proceedings/2025/124

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Task-generic promptable image segmentation aims to achieve segmentation of diverse samples under a single task description by utilizing only one task-generic prompt. Current methods leverage the generalization capabilities of Vision-Language Models (VLMs) to infer instance-specific prompts from these task-generic prompts in order to guide the segmentation process. However, when VLMs struggle to generalise to some image instances, predicting instance-specific prompts becomes poor.  To solve this problem, we introduce Instance-specific Negative Mining for Task-Generic Promptable Segmentation (INT).  The key idea of INT is to adaptively reduce the influence of irrelevant (negative) prior knowledge whilst to increase the use the most plausible prior knowledge, selected by negative mining with higher contrast, in order to optimise instance-specific prompts generation. Specifically, INT consists of two components: (1) instance-specific prompt generation, which progressively fliters out incorrect information in prompt generation; (2) semantic mask generation, which ensures each image instance segmentation matches correctly the semantics of the instance-specific prompts. INT is validated on six datasets, including camouflaged objects and medical images, demonstrating its effectiveness, robustness and scalability.

</details>

---

## 16. Boosting Visual Knowledge-Intensive Training for LVLMs Through Causality-Driven Visual Object Completion

- [ ] Boosting Visual Knowledge-Intensive Training for LVLMs Through Causality-Driven Visual Object Completion | https://www.ijcai.org/proceedings/2025/126

- **Link**: https://www.ijcai.org/proceedings/2025/126

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have experienced significant advancements in recent years. However, their performance still falls short in tasks requiring deep visual perception, such as identifying subtle differences between images. A potential cause is the scarcity of visual knowledge in popular instruction-tuning corpora, resulting in inadequate visual perception and reasoning capabilities. To address this challenge, we introduce a self-improvement framework grounded in a novel visual knowledge-intensive task, Causality-driven Visual object Completion (CVC). This task requires LVLMs to infer the masked object in an image based on its causal relationships with the other visible information. We first obtain rich examples cheaply through our automated instance construction pipeline, without relying on sophisticated LVLMs (e.g., GPT-4V) or human assistance. Then, LVLMs effectively self-improve through trial and error learning using these created instances. Our experiments demonstrate substantial gains across four challenging specialized tasks and four widely-used comprehensive benchmarks. Especially on specialized tasks, our method achieves an average improvement of 5.4% and 4.0% compared to the corresponding baselines when utilizing LLaVA-1.5-7B and LLaVA-1.5-13B, respectively. Code and the supplementary file are available at https://github.com/XMUDeepLIT/CVC.

</details>

---

## 17. What If LLMs Can Smell: A Prototype

- [ ] What If LLMs Can Smell: A Prototype | https://www.ijcai.org/proceedings/2025/1280

- **Link**: https://www.ijcai.org/proceedings/2025/1280

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The olfaction is hardly mentioned in the studies of multi-modal Large Language Models (LLMs). This demo presents a prototypical framework to embody prevalent LLMs with smelling ability using a plug-and-play olfactory signal processing service. To this end, we collect a dataset on Korean beers by self-developed electronic noses (e-noses) and an open-source dataset. An olfaction-related question-answering corpus is also generated to fine-tune LLMs. A gas classification model is applied to identify the smelling liquor upon the e-nose data. We then adopt and fine-tune LLMs on the generated datasets. The results show that LLMs under this framework can interact with the environment by its `nose' and provide olfaction-related answers augmented by our dataset. To the best of our knowledge, this is the first work on embodying LLMs with artificial olfaction. We additionally deployed the gas classification model and the trained LLM in a simple web-based system to show the feasibility of our prototype. Our demo video can be found at: https://bit.ly/4j8x6ZY.

</details>

---

## 18. Counterfactual Knowledge Maintenance for Unsupervised Domain Adaptation

- [ ] Counterfactual Knowledge Maintenance for Unsupervised Domain Adaptation | https://www.ijcai.org/proceedings/2025/165

- **Link**: https://www.ijcai.org/proceedings/2025/165

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional unsupervised domain adaptation (UDA) struggles to extract rich semantics due to backbone limitations. Recent large-scale pre-trained visual-language models (VLMs) have shown strong zero-shot learning capabilities in UDA tasks. However, directly using VLMs results in a mixture of semantic and domain-specific information, complicating knowledge transfer. Complex scenes with subtle semantic differences are prone to misclassification, which in turn can result in the loss of features that are crucial for distinguishing between classes. To address these challenges, we propose a novel counterfactual knowledge maintenance UDA framework. Specifically, we employ counterfactual disentanglement to separate the representation of semantic information from domain features, thereby reducing domain bias. Furthermore, to clarify ambiguous visual information specific to classes, we maintain the discriminative knowledge of both visual and textual information. This approach synergistically leverages multimodal information to preserve modality-specific distinguishable features. We conducted extensive experimental evaluations on several public datasets to demonstrate the effectiveness of our method. The source code is available at https://github.com/LiYaolab/CMKUDA

</details>

---

## 19. PatternCIR Benchmark and TisCIR: Advancing Zero-Shot Composed Image Retrieval in Remote Sensing

- [ ] PatternCIR Benchmark and TisCIR: Advancing Zero-Shot Composed Image Retrieval in Remote Sensing | https://www.ijcai.org/proceedings/2025/171

- **Link**: https://www.ijcai.org/proceedings/2025/171

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Remote sensing composed image retrieval
(RSCIR) is a new vision-language task that takes
a composed query of an image and text, aiming to
search for a target remote sensing image satisfying
two conditions from intricate remote sensing
imagery. However, the existing attribute-based
benchmark Patterncom in RSCIR has significant
flaws, including the lack of query text sentences
and paired triplets, thus making it unable to evaluate the latest methods. To address this, we propose
the Zero-Shot Query Text Generator (ZS-QTG)
that can generate full query text sentences based on
attributes, and then, by capitalizing on ZS-QTG,
we develop the PatternCIR benchmark. PatternCIR rectifies Patterncom’s deficiencies and enables
the evaluation of existing methods. Additionally,
we explore zero-shot composed image retrieval
methods that do not rely on massive pre-collected
triplets for training. Existing methods use only
the text during retrieval, performing poorly in
RSCIR. To improve this, we propose Text-image
Sequential Training of Composed Image Retrieval
(TisCIR). TisCIR undergoes sequential training of
multiple self-masking projection and fine-grained
image attention modules, which endows it with
the capacity to filter out conflicting information
between the image and text, enhancing the retrieval
by utilizing both modalities in harmony. TisCIR
outperforms existing methods by 12.40% to
62.03% on PatternCIR, achieving state-of-the-art
performance in RSCIR. The data and code are
available here.

</details>

---

## 20. OT-DETECTOR: Delving into Optimal Transport for Zero-shot Out-of-Distribution Detection

- [ ] OT-DETECTOR: Delving into Optimal Transport for Zero-shot Out-of-Distribution Detection | https://www.ijcai.org/proceedings/2025/184

- **Link**: https://www.ijcai.org/proceedings/2025/184

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection is crucial for ensuring the reliability and safety of machine learning models in real-world applications. While zero-shot OOD detection, which requires no training on in-distribution (ID) data, has become feasible with the emergence of vision-language models like CLIP, existing methods primarily focus on semantic matching and fail to fully capture distributional discrepancies. To address these limitations, we propose OT-DETECTOR, a novel framework that employs Optimal Transport (OT) to quantify both semantic and distributional discrepancies between test samples and ID labels. Specifically, we introduce cross-modal transport mass and transport cost as semantic-wise and distribution-wise OOD scores, respectively, enabling more robust detection of OOD samples. Additionally, we present a semantic-aware content refinement (SaCR) module, which utilizes semantic cues from ID labels to amplify the distributional discrepancy between ID and hard OOD samples. Extensive experiments on several benchmarks demonstrate that OT-DETECTOR achieves state-of-the-art performance across various OOD detection tasks, particularly in challenging hard-OOD scenarios.

</details>

---

## 21. Understanding Visual Detail Hallucinations of Large Vision-Language Models

- [ ] Understanding Visual Detail Hallucinations of Large Vision-Language Models | https://www.ijcai.org/proceedings/2025/212

- **Link**: https://www.ijcai.org/proceedings/2025/212

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding small visual objects is crucial in fields such as video surveillance, remote sensing, and autonomous driving. In this paper, we investigate the capability of advanced large vision-language models (LVLMs) to recognize and interpret small objects in visual data. To this end, we curate a specialized dataset for evaluating fine-grained visual hallucinations, incorporating two object categories and three types of hallucinations. 
First, we assess 11 state-of-the-art LVLMs, yielding several key insights, as anticipated, LVLMs perform significantly worse on queries related to small objects compared to regular-sized ones, with performance on regular objects proving to be an unreliable predictor of that on small objects. This finding underscores the need for dedicated research on fine-grained visual hallucinations. Second, we evaluate three training-free methods: Scaffold, Chain of Thought (CoT), and Image Resizing,  all of which result in varying degrees of improvement. Furthermore, we conduct a series of detailed ablation studies on the visual encoders of Eagle-X5, examining their performance across fine-grained visual hallucination tasks. Our findings reveal that ConvNeXt architecture is critical for object existence recognition tasks. In contrast, for mitigating other types of hallucinations, integrating information from multiple visual encoders is significantly more effective than relying on a single encoder.
These results highlight several promising directions for advancing small object recognition with LVLMs.

</details>

---

## 22. METOR: A Unified Framework for Mutual Enhancement of Objects and Relationships in Open-vocabulary Video Visual Relationship Detection

- [ ] METOR: A Unified Framework for Mutual Enhancement of Objects and Relationships in Open-vocabulary Video Visual Relationship Detection | https://www.ijcai.org/proceedings/2025/223

- **Link**: https://www.ijcai.org/proceedings/2025/223

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary video visual relationship detection aims to detect objects and their relationships in videos without being restricted by predefined object or relationship categories. Existing methods leverage the rich semantic knowledge of pre-trained vision-language models such as CLIP to identify novel categories. They typically adopt a cascaded pipeline to first detect objects and then classify relationships based on the detected objects, which may lead to error propagation and thus suboptimal performance. In this paper, we propose Mutual EnhancemenT of Objects and Relationships (METOR), a query-based unified framework to jointly model and mutually enhance object detection and relationship classification in open-vocabulary scenarios. Under this framework, we first design a CLIP-based contextual refinement encoding module that extracts visual contexts of objects and relationships to refine the encoding of text features and object queries, thus improving the  generalization of encoding to novel categories. Then we propose an iterative enhancement module to alternatively enhance the representations of objects and relationships by fully exploiting their interdependence to improve recognition performance. Extensive experiments on two public datasets, VidVRD and VidOR, demonstrate that our framework achieves state-of-the-art performance. Codes are at https://github.com/wangyongqi558/METOR.

</details>

---

## 23. TP-Eval: Tap Multimodal LLMs' Potential in Evaluation by Customizing Prompts

- [ ] TP-Eval: Tap Multimodal LLMs' Potential in Evaluation by Customizing Prompts | https://www.ijcai.org/proceedings/2025/232

- **Link**: https://www.ijcai.org/proceedings/2025/232

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, multimodal large language models (MLLMs) have received much attention for their impressive capabilities. The evaluation of MLLMs is becoming critical to analyzing attributes of MLLMs and providing valuable insights. However, current benchmarks overlook the problem of prompt sensitivity - minor prompt variations may lead to significant performance fluctuations. Thus, inappropriate prompts may obscure the models' capabilities, underestimating the models' performance. Moreover, different models have different preferences for different prompts, and thus, using the same prompt for all models will cause evaluation bias. This paper analyzes this deficiency in existing benchmarks and further introduces a new evaluation framework named TP-Eval, which introduces a prompt customization method to reduce evaluation biases and tap models' potential. TP-Eval will rewrite the original prompts to different customized prompts for different models. In particular, we propose some well-designed modules for prompt customization tailored to the scenario of MLLM evaluation. Extensive experiments demonstrate the effectiveness of our approach to uncovering models' capabilities, and TP-Eval should benefit the community in developing more comprehensive and convincing MLLM evaluation benchmarks.

</details>

---

## 24. TEST-V: TEst-time Support-set Tuning for Zero-shot Video Classification

- [ ] TEST-V: TEst-time Support-set Tuning for Zero-shot Video Classification | https://www.ijcai.org/proceedings/2025/239

- **Link**: https://www.ijcai.org/proceedings/2025/239

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, adapting Vision Language Models (VLMs) to zero-shot visual classification by tuning class embedding with a few prompts (Test-time Prompt Tuning, TPT) or replacing class names with generated visual samples (support-set) has shown promising results. However, TPT cannot avoid the semantic gap between modalities while the support-set cannot be tuned. To this end, we draw on each other's strengths and propose a novel framework, namely TEst-time Support-set Tuning for zero-shot Video Classification (TEST-V). It first dilates the support-set with multiple prompts (Multi-prompting Support-set Dilation, MSD) and then erodes the support-set via learnable weights to mine key cues dynamically (Temporal-aware Support-set Erosion, TSE). Specifically, i) MSD expands the support samples for each class based on multiple prompts inquired from LLMs to enrich the diversity of the support-set. ii) TSE tunes the support-set with factorized learnable weights according to the temporal prediction consistency in a self-supervised manner to dig pivotal supporting cues for each class. TEST-V achieves state-of-the-art results across four benchmarks and shows good interpretability.

</details>

---

## 25. Leveraging MLLM Embeddings and Attribute Smoothing for Compositional Zero-Shot Learning

- [ ] Leveraging MLLM Embeddings and Attribute Smoothing for Compositional Zero-Shot Learning | https://www.ijcai.org/proceedings/2025/243

- **Link**: https://www.ijcai.org/proceedings/2025/243

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Compositional zero-shot learning (CZSL) aims to recognize novel compositions of attributes and objects learned from seen compositions. Previous works disentangle attributes and objects by extracting shared and exclusive parts between the image pair sharing the same attribute (object), as well as aligning them with pretrained word embeddings to improve unseen attribute-object recognition. Despite the significant achievements of existing efforts, they are hampered by three limitations: (1) The efficacy of disentanglement is compromised due to the influence of the background and the intricate entanglement of attributes with objects in the same parts. (2) Existing word embeddings fail to capture complex multimodal semantic information. (3) Overconfidence exhibited by existing models in seen compositions hinders their generalization to novel compositions. Being aware of these, we propose a novel framework named multimodal large language model (MLLM) embeddings and attribute smoothing guided disentanglement for CZSL. First, we leverage feature adaptive aggregation modules to mitigate the impact of background, and utilize learnable condition masks to capture multi-granularity features for disentanglement. Moreover, the last hidden states of MLLM are employed as word embeddings for their superior representation capabilities. Furthermore, we propose attribute smoothing with auxiliary attributes generated by the large language model (LLM) for seen compositions to address the overconfidence challenge. Extensive experiments demonstrate that our method achieves state-of-the-art performance on three challenging datasets. The supplementary material and source code will be available at https://github.com/xud-yan/Trident.

</details>

---

## 26. SCVBench: A Benchmark with Multi-turn Dialogues for Story-Centric Video Understanding

- [ ] SCVBench: A Benchmark with Multi-turn Dialogues for Story-Centric Video Understanding | https://www.ijcai.org/proceedings/2025/255

- **Link**: https://www.ijcai.org/proceedings/2025/255

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video understanding seeks to enable machines to interpret visual content across three levels: action, event, and story.  Existing models are limited in their ability to perform high-level long-term story understanding, due to (1) the oversimplified treatment of temporal information and (2) the training bias introduced by action/event-centric datasets. To address this, we introduce SCVBench, a novel benchmark for story-centric video understanding. SCVBench evaluates LVLMs through an event ordering task decomposed into sub-questions leading to a final question, quantitatively measuring historical dialogue exploration. We collected 1,253 final questions and 6,027 sub-question pairs from 925 videos, constructing continuous multi-turn dialogues. Experimental results show that while closed-source GPT-4o outperforms other models, most open-source LVLMs struggle with story-centric video understanding. Additionally, our StoryCoT model significantly surpasses open-source LVLMs on SCVBench. SCVBench aims to advance research by comprehensively analyzing LVLMs' temporal reasoning and comprehension capabilities. Code can be accessed at https://github.com/yuanrr/SCVBench.

</details>

---

## 27. Multimodal Prior Learning with Double Constraint Alignment for Snapshot Spectral Compressive Imaging

- [ ] Multimodal Prior Learning with Double Constraint Alignment for Snapshot Spectral Compressive Imaging | https://www.ijcai.org/proceedings/2025/263

- **Link**: https://www.ijcai.org/proceedings/2025/263

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The objective of snapshot spectral compressive imaging reconstruction is to recover the 3D hyperspectral image (HSI) from a 2D measurement. Existing methods either focus on network architecture design or simply introduce image-level prior to the model. However, these methods lack guiding information for accurate reconstruction. Recognizing that textual description contain rich semantic information that can significantly enhance details, this paper introduces a novel framework, CAMM, which integrates text information into the model to improve the performance. The framework comprises two key components: Fine-grained Alignment Module (FAM) and Multimodal Fusion Mamba (MFM). Specifically, FAM is used to reduce the knowledge gap between the RGB domain obtained by the pre-trained vision-language model and the HSI domain. Through the double constraints of distribution similarity and entropy, the adaptive alignment of different complexity features is realized, which makes the encoded features more accurate. MFM aims to identify the guiding effect of RGB features and text features on HSI in space and channel dimensions. Instead of fusing features directly, it integrates prior at image-level and text-level prior into Mamba's state-space equation, so that each scanning step can be accurately guided. This kind of positive feedback adjustment ensures the authenticity of the guiding information. To our knowledge, this is the first text-guided model for compressive spectral imaging. Extensive experimental results the public datasets demonstrate the superior performance of CAMM, validating the effectiveness of our proposed method.

</details>

---

## 28. Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner

- [ ] Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner | https://www.ijcai.org/proceedings/2025/279

- **Link**: https://www.ijcai.org/proceedings/2025/279

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained foundation models have recently made significant progress in table-related tasks such as table understanding and reasoning. However, recognizing the structure and content of unstructured tables using Vision Large Language Models (VLLMs) remains under-explored. To bridge this gap, we propose a benchmark based on a hierarchical design philosophy to evaluate the recognition capabilities of VLLMs in training-free scenarios. Through in-depth evaluations, we find that low-quality image input is a significant bottleneck in the recognition process. Drawing inspiration from this, we propose the Neighbor-Guided Toolchain Reasoner (NGTR) framework, which is characterized by integrating diverse lightweight tools for visual operations aimed at mitigating issues with low-quality images. Specifically, we transfer a tool selection experience from a similar neighbor to the input and design a reflection module to supervise the tool invocation process. Extensive experiments on public datasets demonstrate that our approach significantly enhances the recognition capabilities of the vanilla VLLMs. We believe that the benchmark and framework could provide an alternative solution to table recognition.

</details>

---

## 29. Empowering Multimodal Road Traffic Profiling with Vision Language Models and Frequency Spectrum Fusion

- [ ] Empowering Multimodal Road Traffic Profiling with Vision Language Models and Frequency Spectrum Fusion | https://www.ijcai.org/proceedings/2025/300

- **Link**: https://www.ijcai.org/proceedings/2025/300

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid urbanization in the modern era, smart traffic profiling based on multimodal sources of data has been playing a significant role in ensuring safe travel, reducing traffic congestion and optimizing urban mobility. Most existing methods for traffic profiling on the road level usually utilize single-modality data, i.e., they mainly focus on image processing with deep vision models or auxiliary analysis on the textual data. However, the joint modeling and multimodal fusion of the textual and visual modalities have been rarely studied in road traffic profiling, which largely hinders the accurate prediction or classification of traffic conditions. To address this issue, we propose a novel multimodal learning and fusion framework for road traffic profiling, named TraffiCFUS. Specifically, given the traffic images, our TraffiCFUS framework first introduces Vision Language Models (VLMs) to generate text and then creates tailored prompt instructions for refining this text according to the specific scene requirements of road traffic profiling. Next, we apply the discrete Fourier transform to convert multimodal data from the spatial domain to the frequency domain and perform a cross-modal spectrum transform to filter out irrelevant information for traffic profiling. Furthermore, the processed spatial multimodal data is combined to generate fusion loss and interaction loss with contrastive learning. Finally, extensive experiments on four real-world datasets illustrate superior performance compared with the state-of-the-art approaches.

</details>

---

## 30. Cyclic Vision-Language Manipulator: Towards Reliable and Fine-Grained Image Interpretation for Automated  Report Generation

- [ ] Cyclic Vision-Language Manipulator: Towards Reliable and Fine-Grained Image Interpretation for Automated  Report Generation | https://www.ijcai.org/proceedings/2025/41

- **Link**: https://www.ijcai.org/proceedings/2025/41

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant advancements in automated report generation, the opaqueness of text interpretability continues to cast doubt on the reliability of the content produced. This paper introduces a novel approach to identify specific image features in X-ray images that influence the outputs of report generation models. Specifically, we propose Cyclic Vision-Language Manipulator (CVLM), a module to generate a manipulated X-ray from an original X-ray and its report from a designated report generator. The essence of CVLM is that cycling manipulated X-rays to the report generator produces altered reports aligned with the alterations pre-injected into the reports for X-ray generation, achieving the term ``cyclic manipulation''. This process allows direct comparison between original and manipulated X-rays, clarifying the critical image features driving changes in reports and enabling model users to assess the reliability of the generated texts. Empirical evaluations demonstrate that CVLM can identify more precise and reliable features compared to existing explanation methods, significantly enhancing the transparency and applicability of AI-generated reports.

</details>

---

## 31. Modality-Fair Preference Optimization for Trustworthy MLLM Alignment

- [ ] Modality-Fair Preference Optimization for Trustworthy MLLM Alignment | https://www.ijcai.org/proceedings/2025/46

- **Link**: https://www.ijcai.org/proceedings/2025/46

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have achieved remarkable success across various tasks. However, separate training of visual and textual encoders often results in a misalignment of the modality. Such misalignment may lead models to generate content that is absent from the input image, a phenomenon referred to as hallucination. These inaccuracies severely undermine the trustworthiness of MLLMs in real-world applications. Despite attempts to optimize text preferences to mitigate this issue, our initial investigation indicates that the trustworthiness of MLLMs remains inadequate. Specifically, these models tend to provide preferred answers even when the input image is heavily distorted. Analysis of visual token attention also indicates that the model focuses primarily on the surrounding context rather than the key object referenced in the question. These findings highlight a misalignment between the modalities, where answers inadequately leverage input images. Motivated by our findings, we propose Modality-Fair Preference Optimization (MFPO), which comprises three components: the construction of a multimodal preference dataset in which dispreferred images differ from originals solely in key regions; an image reward loss function encouraging the model to generate answers better aligned with the input images; and an easy-to-hard iterative alignment strategy to stabilize joint modality training. Extensive experiments on three trustworthiness benchmarks demonstrate that MFPO significantly enhances the trustworthiness of MLLMs. In particular, it enables the 7B models to attain trustworthiness levels on par with, or even surpass, those of the 13B, 34B, and larger models.

</details>

---

## 32. CABIN: Debiasing Vision-Language Models Using Backdoor Adjustments

- [ ] CABIN: Debiasing Vision-Language Models Using Backdoor Adjustments | https://www.ijcai.org/proceedings/2025/55

- **Link**: https://www.ijcai.org/proceedings/2025/55

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have demonstrated strong zero-shot inference capabilities but may exhibit stereotypical biases toward certain demographic groups. Consequently, downstream tasks leveraging these models may yield unbalanced performance across different target social groups, potentially reinforcing harmful stereotypes. Mitigating such biases is critical for ensuring fairness in practical applications. Existing debiasing approaches typically rely on curated face-centric datasets for fine-tuning or retraining, risking overfitting and limiting generalisability. To address this issue, we propose a novel framework, CABIN (Causal Adjustment Based INtervention). It leverages a causal framework to identify sensitive attributes in images as confounding factors. Employing a learned mapper, which is trained on general large-scale image-text pairs rather than face-centric datasets, CABIN may use text to adjust sensitive attributes in the image embedding, ensuring independence between these sensitive attributes and image embeddings. This independence enables a backdoor adjustment for unbiased inference without the drawbacks of additional fine-tuning or retraining on narrowly tailored datasets. Through comprehensive experiments and analyses, we demonstrate that CABIN effectively mitigates biases and improves fairness metrics while preserving the zero-shot strengths of VLMs. The code is available at:  https://github.com/ipangbo/causal-debias

</details>

---

## 33. Screening, Rectifying, and Re-Screening: A Unified Framework for Tuning Vision-Language Models with Noisy Labels

- [ ] Screening, Rectifying, and Re-Screening: A Unified Framework for Tuning Vision-Language Models with Noisy Labels | https://www.ijcai.org/proceedings/2025/568

- **Link**: https://www.ijcai.org/proceedings/2025/568

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models have shown remarkable potential for downstream tasks. However, their fine-tuning under noisy labels remains an open problem due to challenges like self-confirmation bias and the limitations of conventional small-loss criteria. In this paper, we propose a unified framework to address these issues, consisting of three key steps: Screening, Rectifying, and Re-Screening. First, a dual-level semantic matching mechanism is introduced to categorize samples into clean, ambiguous, and noisy samples by leveraging both macro-level and micro-level textual prompts. Second, we design tailored pseudo-labeling strategies to rectify noisy and ambiguous labels, enabling their effective incorporation into the training process. Finally, a re-screening step, utilizing cross-validation with an auxiliary vision-language model, mitigates self-confirmation bias and enhances the robustness of the framework. Extensive experiments across ten datasets demonstrate that the proposed method significantly outperforms existing approaches for tuning vision-language pre-trained models with noisy labels.

</details>

---

## 34. Visual Perturbation and Adaptive Hard Negative Contrastive Learning for  Compositional Reasoning in Vision-Language Models

- [ ] Visual Perturbation and Adaptive Hard Negative Contrastive Learning for  Compositional Reasoning in Vision-Language Models | https://www.ijcai.org/proceedings/2025/605

- **Link**: https://www.ijcai.org/proceedings/2025/605

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) are essential for multimodal tasks, especially compositional reasoning (CR) tasks, which require distinguishing fine-grained semantic differences between visual and textual embeddings. However, existing methods primarily fine-tune the model by generating text-based hard negative samples, neglecting the importance of image-based negative samples, which results in insufficient training of the visual encoder and ultimately impacts the overall performance of the model. Moreover, negative samples are typically treated uniformly, without considering their difficulty levels, and the alignment of positive samples is insufficient, which leads to challenges in aligning difficult sample pairs. To address these issues, we propose Adaptive Hard Negative Perturbation Learning (AHNPL). AHNPL translates text-based hard negatives into the visual domain to generate semantically disturbed image-based negatives for training the model, thereby enhancing its overall performance. AHNPL also introduces a contrastive learning approach using a multimodal hard negative loss to improve the model's discrimination of hard negatives within each modality and a dynamic margin loss that adjusts the contrastive margin according to sample difficulty to enhance the distinction of challenging sample pairs. Experiments on three public datasets demonstrate that our method effectively boosts VLMs' performance on complex CR tasks. The source code is available at https://github.com/nynu-BDAI/AHNPL.

</details>

---

## 35. Differentiable Prompt Learning for Vision Language Models

- [ ] Differentiable Prompt Learning for Vision Language Models | https://www.ijcai.org/proceedings/2025/606

- **Link**: https://www.ijcai.org/proceedings/2025/606

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning is an effective way to exploit the potential of large-scale pre-trained foundational models. Continuous prompts parameterize context tokens in prompts by turning them into differentiable vectors. Deep continuous prompts insert prompts not only in the input but also in the intermediate hidden representations. Manually designed deep continuous prompts exhibit a remarkable improvement compared to the zero-shot pre-trained model on downstream tasks. How to automate the continuous prompt design is an underexplored area, and a fundamental question arises, is manually designed deep prompt strategy optimal? To answer this question, we propose a method dubbed differentiable prompt learning (DPL). The DPL method is formulated as an optimization problem to automatically determine the optimal context length of the prompt to be added to each layer, where the objective is to maximize the performance. We test the DPL method on the pre-trained CLIP. We empirically find that by using only limited data, our DPL method can find deep continuous prompt configuration with high confidence. The performance on the downstream tasks exhibits the superiority of the automatic design: our method boosts the average test accuracy by 2.60% on 11 datasets compared to baseline methods. Besides, our method focuses only on the prompt configuration (i.e. context length for each layer), which means that our method is compatible with the baseline methods that have sophisticated designs to boost the performance. We release our code in https://github.com/Zhenhan-Huang/Differentiable-Prompt-Learn.

</details>

---

## 36. BMIP: Bi-directional Modality Interaction Prompt Learning for VLM

- [ ] BMIP: Bi-directional Modality Interaction Prompt Learning for VLM | https://www.ijcai.org/proceedings/2025/655

- **Link**: https://www.ijcai.org/proceedings/2025/655

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have exhibited remarkable generalization capabilities, and prompt learning for VLMs has attracted great attention for the ability to adapt pre-trained VLMs to specific downstream tasks. However, existing studies mainly focus on single-modal prompts or uni-directional modality interaction, overlooking the powerful alignment effects resulting from the interaction between the vision and language modalities. To this end, we propose a novel prompt learning method called Bi-directional Modality Interaction Prompt (BMIP), which dynamically weights bi-modal information through learning the information of the attention layer, enhancing trainability and inter-modal consistency compared to simple information aggregation methods. To evaluate the effectiveness of prompt learning methods, we propose a more realistic evaluation paradigm called open-world generalization complementing the widely adopted cross-dataset transfer and domain generalization tasks. Comprehensive experiments on various datasets reveal that BMIP not only outperforms current state-of-the-art methods across all three evaluation paradigms but is also flexible enough to be combined with other prompt-based methods for consistent performance enhancement.

</details>

---

## 37. MsRAG: Knowledge Augumented Image Captioning with Object-level Multi-source RAG

- [ ] MsRAG: Knowledge Augumented Image Captioning with Object-level Multi-source RAG | https://www.ijcai.org/proceedings/2025/678

- **Link**: https://www.ijcai.org/proceedings/2025/678

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language-Visual Large Models (LVLMs) have made significant strides in enhancing visual understanding capabilities. However, these models often struggle with knowledge-based visual tasks due to constrains in their pre-training data scope and timeliness. Existing Retrieval-Augmented Generation (RAG) methods can effectively solve the problem but primarily rely on user queries, limiting their applicability in scenarios without explicit language input. To overcome these challenges, we introduce MsRAG, a knowledge-augmented captioning framework designed to effectively retrieve and utilize external real-world knowledge, particularly in the absence of user queries, and perform dense captioning for subjects. MsRAG comprises three key components: (1) Parallel Visual Search Module. It retrieves fine-grained object-level knowledge using both online visual search engines and offline domain-knowledge databases, enhancing the robustness and richness of retrieved information. (2) Prompt Templates Pool. The prompt pool dynamically assigns appropriate prompts based on retrieved information, optimizing LVLMs' ability to leverage relevant data under complex RAG conditions. (3) Visual-RAG Alignment Module, which employs a novel visual prompting method to bridge the modality gap between textual RAG content and corresponding visual objects, enabling precise alignment of visual elements with their text-format RAG content. To validate the effectiveness of MsRAG, we conducted a series of qualitative and quantitative experiments. The evaluation results demonstrate the superiority of MsRAG over other methods.

</details>

---

## 38. M4Bench: A Benchmark of Multi-domain Multi-granularity Multi-image Understanding for Multi-modal Large Language Models

- [ ] M4Bench: A Benchmark of Multi-domain Multi-granularity Multi-image Understanding for Multi-modal Large Language Models | https://www.ijcai.org/proceedings/2025/762

- **Link**: https://www.ijcai.org/proceedings/2025/762

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The increasing demands in analyzing complex associated scenes pose necessities to researching multi-image understanding abilities. Compared with understanding individual images, both the alignments and differences between images are essential aspects of understanding the intricate relationships for multi-image inference tasks. However, existing benchmarks face difficulties in addressing both of these aspects simultaneously, resulting in obstacles to modeling relationships under various granularities and domains of images. In this paper, we introduce M4Bench to enhance the capability of aligning and distinguishing multi-images with multi-domain multi-granularity comparison. We carefully design five comparison tasks related to coarse and fine-grained granularities in single and multiple domains of images and evaluate them on 13 state-of-the-art multi-modal large language models with various sizes. Besides, we analyze the evaluation results and provide several observations and viewpoints for the multi-image understanding research. The data and evaluation code are available at https://github.com/eaglelab-zju/M4Bench.

</details>

---

## 39. ExpertDiff: Head-less Model Reprogramming with Diffusion Classifiers for Out-of-Distribution Generalization

- [ ] ExpertDiff: Head-less Model Reprogramming with Diffusion Classifiers for Out-of-Distribution Generalization | https://www.ijcai.org/proceedings/2025/764

- **Link**: https://www.ijcai.org/proceedings/2025/764

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models have achieved remarkable performance across various tasks by leveraging large-scale multimodal training data. However, their ability to generalize to out-of-distribution (OOD) domains requiring expert-level knowledge remains an open challenge. To address this, we investigate cross-domain transfer learning approaches for efficiently adapting diffusion classifiers to new target domains demanding expert-level domain knowledge. Specifically, we propose ExpertDiff, a head-less model reprogramming technique that optimizes the instruction-following abilities of text-to-image diffusion models via learnable prompts, while leveraging the diffusion classifier objective as a modular plug-and-play adaptor. Our approach eliminates the need for conventional output mapping layers (e.g., linear probes), enabling seamless integration with off-the-shelf diffusion frameworks like Stable Diffusion. We demonstrate the effectiveness of ExpertDiff on the various OOD datasets (i.e., medical and satellite imagery). Furthermore, we qualitatively showcase ExpertDiff’s ability to faithfully reconstruct input images, highlighting its potential for both downstream discriminative and upstream generative tasks. Our work paves the way for effectively repurposing powerful foundation models for novel OOD applications requiring domain expertise.

</details>

---

## 40. Towards VLM-based Hybrid Explainable Prompt Enhancement for Zero-Shot Industrial Anomaly Detection

- [ ] Towards VLM-based Hybrid Explainable Prompt Enhancement for Zero-Shot Industrial Anomaly Detection | https://www.ijcai.org/proceedings/2025/80

- **Link**: https://www.ijcai.org/proceedings/2025/80

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-Shot Industrial Anomaly Detection (ZSIAD) aims to identify and localize anomalies in industrial images from unseen categories. Owing to the powerful generalization capabilities, Vision-Language Models (VLMs) have achieved growing interest in ZSIAD. To guide the model toward understanding and localizing the semantically complex industrial anomalies, existing VLM-based methods have attempted to provide additional prompts to the model through learnable text prompt templates. However, these zero-shot methods lack detailed descriptions of specific anomalies, making it difficult to classify and segment the diverse range of industrial anomalies accurately. To address the aforementioned issue, we firstly propose the multi-stage prompt generation agent for ZSIAD. Specifically, we leverage the Multi-modal Language Large Model (MLLM) to articulate the detailed differential information between normal and test samples, which can provide detailed text prompts to the model through further refinement and anti-false alarm constraint. Moreover, we introduce the Visual Fundamental Model (VFM) to generate anomaly-related attention prompts for more accurate localization of anomalies with varying sizes and shapes. Extensive experiments on seven real-world industrial anomaly detection datasets have shown that the proposed method not only outperforms recent SOTA methods, but also its explainable prompts provide the model with a more intuitive basis for anomaly identification.

</details>

---

## 41. IterMeme: Expert-Guided Multimodal LLM for Interactive Meme Creation with Layout-Aware Generation

- [ ] IterMeme: Expert-Guided Multimodal LLM for Interactive Meme Creation with Layout-Aware Generation | https://www.ijcai.org/proceedings/2025/81

- **Link**: https://www.ijcai.org/proceedings/2025/81

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Meme creation is a creative process that blends images and text. However, existing methods lack critical components, failing to support intent-driven caption-layout generation and personalized generation, making it difficult to generate high-quality memes. To address this limitation, we propose IterMeme, an end-to-end interactive meme creation framework that utilizes a unified Multimodal Large Language Model (MLLM) to facilitate seamless collaboration among multiple components. To overcome the absence of a caption-layout generation component, we develop a robust layout representation method and construct a large-scale image-caption-layout dataset, MemeCap, which enhances the model’s ability to comprehend emotions and coordinate caption-layout generation effectively.
To address the lack of a personalization component, we introduce a parameter-shared dual-LLM architecture that decouples the intricate representations of reference images and text. Furthermore, we incorporate the expert-guided M³OE for fine-grained identity properties (IP) feature extraction and cross-modal fusion. By dynamically injecting features into every layer of the model, we enable adaptive refinement of both visual and semantic information.
Experimental results demonstrate that IterMeme significantly advances the field of meme creation by delivering consistently high-quality outcomes. The code, model, and dataset will be open-sourced to the community.

</details>

---

## 42. Contrastive Unlearning: A Contrastive Approach to Machine Unlearning

- [ ] Contrastive Unlearning: A Contrastive Approach to Machine Unlearning | https://www.ijcai.org/proceedings/2025/830

- **Link**: https://www.ijcai.org/proceedings/2025/830

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Machine unlearning aims to eliminate the influence of a subset of training samples (i.e., unlearning samples) from a trained model. Effectively and efficiently removing the unlearning samples without negatively impacting the overall model performance is challenging. Existing works mainly exploit input and output space and classification loss, which can result in ineffective unlearning or performance loss.   In addition, they utilize  unlearning or remaining samples ineffectively, sacrificing either unlearning efficacy or efficiency. 
    Our main insight is that the direct optimization on the representation space utilizing both unlearning and remaining samples can effectively remove influence of unlearning samples while maintaining representations learned from remaining samples. We propose a contrastive unlearning framework, leveraging the concept of representation learning for more effective unlearning. It removes the influence of unlearning samples by contrasting their embeddings against the remaining samples' embeddings 
    so that their embeddings are closer to the embeddings of unseen samples.
    Experiments on a variety of datasets and models on both class unlearning and sample unlearning showed that contrastive unlearning achieves the best unlearning effects and efficiency with the lowest performance loss compared with the state-of-the-art algorithms. In addition, it is generalizable to different contrastive frameworks and other models such as vision-language models. Our main code is available on github.com/Emory-AIMS/Contrastive-Unlearning

</details>

---

## 43. Can Retelling Have Adequate Information for Reasoning? An Enhancement Method for Imperfect Video Understanding with Large Language Model

- [ ] Can Retelling Have Adequate Information for Reasoning? An Enhancement Method for Imperfect Video Understanding with Large Language Model | https://www.ijcai.org/proceedings/2025/906

- **Link**: https://www.ijcai.org/proceedings/2025/906

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) demonstrate strong capabilities in video understanding. However, it exhibits hallucinations and factual errors in video description. On the one hand, existing Multimodal Large Language Models (MLLMs) are primarily trained by combining language models and vision models, with their visual understanding capabilities depending on the performance of the backbone. Moreover, video descriptions often suffer from incomplete content and the possibility of errors. Given the proven assessment of the strong reasoning capabilities of LLMs, this paper proposes ERSR, a novel Entity and Relationship based Self-Enhanced Reasoning method for imperfect video understanding. Specifically, an entities and relationships strategy is designed to perform scene graphs based on the limited observed entity relationships, thereby enhancing video descriptions. Furthermore, by providing question feedbacks, a self-enhanced forward and feedback reasoning strategy is provided to enhance reasoning logic. Finally, the prediction question answering results are re-validated through rethinking and verifying using the LLMs. Extensive experiments show that the proposed method achieves competitive results on real-world video understanding datasets, with an overall improvement of no less than 1.4%.

</details>

---

## 44. Language-Conditioned Open-Vocabulary Mobile Manipulation with Pretrained Models

- [ ] Language-Conditioned Open-Vocabulary Mobile Manipulation with Pretrained Models | https://www.ijcai.org/proceedings/2025/976

- **Link**: https://www.ijcai.org/proceedings/2025/976

- **Conference**: IJCAI

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary mobile manipulation (OVMM) that involves the handling of novel and unseen objects across different workspaces remains a significant challenge for real-world robotic applications. In this paper, we propose a novel Language-conditioned Open-Vocabulary Mobile Manipulation framework, named LOVMM, incorporating the large language model (LLM) and vision-language model (VLM) to tackle various mobile manipulation tasks in household environments. Our approach is capable of solving various OVMM tasks with free-form natural language instructions (e.g. "toss the food boxes on the office room desk to the trash bin in the corner", and "pack the bottles from the bed to the box in the guestroom"). Extensive experiments simulated in complex household environments show strong zero-shot generalization and multi-task learning abilities of LOVMM. Moreover, our approach can also generalize to multiple tabletop manipulation tasks and achieve better success rates compared to other state-of-the-art methods.

</details>

---

