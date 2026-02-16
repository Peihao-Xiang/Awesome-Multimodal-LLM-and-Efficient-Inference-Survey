# ECCV 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_eccv2024_papers.csv

## 1. MotionChain: Conversational Motion Controllers via Multimodal Prompts

- [ ] MotionChain: Conversational Motion Controllers via Multimodal Prompts | https://eccv.ecva.net/virtual/2024/poster/1008

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1008

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in language models have demonstrated their adeptness in conducting multi-turn dialogues and retaining conversational context. However, this proficiency remains largely unexplored in other multimodal generative models, particularly in human motion models. By integrating multi-turn conversations in controlling continuous virtual human movements, generative human motion models can achieve an intuitive and step-by-step process of human task execution for humanoid robotics, game agents, or other embodied systems. In this work, we present MotionChain, a conversational human motion controller to generate continuous and long-term human motion through multimodal prompts. Specifically, MotionChain consists of multi-modal tokenizers that transform various data types such as text, image, and motion, into discrete tokens, coupled with a Vision-Motion-aware Language model. By leveraging large-scale language, vision-language, and vision-motion data to assist motion-related generation tasks, MotionChain thus comprehends each instruction in multi-turn conversation and generates human motions followed by these prompts. Extensive experiments validate the efficacy of MotionChain, demonstrating state-of-the-art performance in conversational motion generation, as well as more intuitive manners of controlling and interacting with virtual humans.

</details>

---

## 2. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks

- [ ] DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks | https://eccv.ecva.net/virtual/2024/poster/1065

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1065

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications. This paper introduces DIFFender, a novel defense framework that harnesses the capabilities of a text-guided diffusion model to combat patch attacks. Central to our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which empowers the diffusion model to detect and localize adversarial patches through the analysis of distributional discrepancies. DIFFender integrates dual tasks of patch localization and restoration within a single diffusion model framework, utilizing their close interaction to enhance defense efficacy. Moreover, DIFFender utilizes vision-language pre-training coupled with an efficient few-shot prompt-tuning algorithm, which streamlines the adaptation of the pre-trained diffusion model to defense tasks, thus eliminating the need for extensive retraining. Our comprehensive evaluation spans image classification and face recognition tasks, extending to real-world scenarios, where DIFFender shows good robustness against adversarial attacks. The versatility and generalizability of DIFFender are evident across a variety of settings, classifiers, and attack methodologies, marking an advancement in adversarial patch defense strategies.

</details>

---

## 3. Unified Embedding Alignment for Open-Vocabulary Video Instance Segmentation

- [ ] Unified Embedding Alignment for Open-Vocabulary Video Instance Segmentation | https://eccv.ecva.net/virtual/2024/poster/1085

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1085

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-Vocabulary Video Instance Segmentation(VIS) is attracting increasing attention due to its ability to segment and track arbitrary objects. However, the recent Open-Vocabulary VIS attempts obtained unsatisfactory results, especially in terms of generalization ability of novel categories. We discover that the domain gap between the VLM features and the instance queries and the underutilization of temporal consistency are two central causes. To mitigate these issues, we design and train a novel Open-Vocabulary VIS baseline called OVFormer. OVFormer utilizes a lightweight module for unified embedding alignment between query embeddings and CLIP image embeddings to remedy the domain gap. Unlike previous image-based training methods, we conduct video-based model training and deploy a semi-online inference scheme to fully mine the temporal consistency in the video. Without bells and whistles, OVFormer achieves 21.9 mAP with a ResNet-50 backbone on LV-VIS, exceeding the previous state-of-the-art performance by +7.7(an improvement of 54% over OV2Seg). Extensive experiments on some Close-Vocabulary VIS datasets also demonstrate the strong zero-shot generalization ability of OVFormer (+7.6 mAP on YouTube-VIS 2019, +3.9 mAP on OVIS). Code is available at https://github.com/Anonymous668/OVFormer.

</details>

---

## 4. Elysium: Exploring Object-level Perception in Videos through Semantic Integration Using MLLMs

- [ ] Elysium: Exploring Object-level Perception in Videos through Semantic Integration Using MLLMs | https://eccv.ecva.net/virtual/2024/poster/1089

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1089

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated their ability to perceive objects in still images, but their application in video-related tasks, such as object tracking, remains relatively unexplored. This lack of exploration is primarily due to two key challenges. Firstly, extensive pretraining on large-scale video datasets is required to equip MLLMs with the capability to perceive objects across multiple frames and understand inter-frame relationships. Secondly, processing a large number of frames within the context window of Large Language Models (LLMs) can impose a significant computational burden. To address the first challenge, we introduce ElysiumTrack-1M, a large-scale video dataset paired with novel tasks: Referring Single Object Tracking (RSOT) and Video Referring Expression Generation (Video-REG). ElysiumTrack-1M contains 1.27 million annotated video frames with corresponding object boxes and descriptions. Leveraging this dataset, we conduct training of MLLMs and propose a token-compression model T-Selector to tackle the second challenge. Our proposed approach, Elysium: Exploring Object-level Perception in Videos via MLLM, is an end-to-end trainable MLLM that makes the first attempt to conduct object-level tasks in videos without requiring any additional plug-in or expert models. All codes and datasets will be released soon.

</details>

---

## 5. Multi-Modal Video Dialog State Tracking in the Wild

- [ ] Multi-Modal Video Dialog State Tracking in the Wild | https://eccv.ecva.net/virtual/2024/poster/1094

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1094

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present MSTMIXER – a novel video dialog model operating over a generic multi-modal state tracking scheme. Current models that claim to perform multi-modal state tracking fall short of two major aspects: (1) They either track only one modality (mostly the visual input) or (2) they target synthetic datasets that do not reflect the complexity of real-world in the wild scenarios. Our model addresses these two limitations in an attempt to close this crucial research gap. Specifically, MST-MIXER first tracks the most important constituents of each input modality. Then, it predicts the missing underlying structure of the selected constituents of each modality by learning local latent graphs using a novel multi-modal graph structure learning method. Subsequently, the learned local graphs and features are parsed together to form a global graph operating on the mix of all modalities which further refines its structure and node embeddings. Finally, the fine-grained graph node features are used to enhance the hidden states of the backbone Vision-Language Model (VLM). MST-MIXER achieves new SOTA results on five challenging benchmarks.

</details>

---

## 6. VideoAgent: Long-form Video Understanding with Large Language Model as Agent

- [ ] VideoAgent: Long-form Video Understanding with Large Language Model as Agent | https://eccv.ecva.net/virtual/2024/poster/1090

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1090

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Long-form video understanding represents a significant challenge within computer vision, demanding a model capable of reasoning over long multi-modal sequences. Motivated by the human cognitive process for long-form video understanding, we emphasize interactive reasoning and planning over the ability to process lengthy visual inputs. We introduce a novel agent-based system, VideoAgent, that employs a large language model as a central agent to iteratively identify and compile crucial information to answer a question, with vision-language foundation models serving as tools to translate and retrieve visual information. Evaluated on the challenging EgoSchema and NExT-QA benchmarks, VideoAgent achieves 54.1% and 71.3% zero-shot accuracy with only 8.4 and 8.2 frames used on average. These results demonstrate superior effectiveness and efficiency of our method over the current state-of-the-art methods, highlighting the potential of agent-based approaches in advancing long-form video understanding.

</details>

---

## 7. Learning Video Context as Interleaved Multimodal Sequences

- [ ] Learning Video Context as Interleaved Multimodal Sequences | https://eccv.ecva.net/virtual/2024/poster/1093

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1093

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Narrative videos, such as movies, pose significant challenges in video understanding due to their rich contexts (characters, dialogues, storylines) and diverse demands (identify who~\cite{autoad2}, relationship~\cite{lfvu}, and reason~\cite{movieqa}). In this paper, we introduce~{\our}, a multimodal language model developed to address the wide range of challenges in understanding video contexts. Our core idea is to represent videos as interleaved multimodal sequences (including images, plots, videos, and subtitles), either by linking external knowledge databases or using offline models (such as whisper for subtitles). Through instruction-tuning, this approach empowers the language model to interact with videos using interleaved multimodal instructions. For example, instead of solely relying on video as input, we jointly provide character photos alongside their names and dialogues, allowing the model to associate these elements and generate more comprehensive responses. To demonstrate its effectiveness, we validate \our's performance on six datasets (LVU, MAD, Movienet, CMD, TVC, MovieQA) across five settings (video classifcation, audio description, video-text retrieval, video captioning, and video question-answering).

</details>

---

## 8. AutoEval-Video: An Automatic Benchmark for Assessing Large Vision Language Models in Open-Ended Video Question Answering

- [ ] AutoEval-Video: An Automatic Benchmark for Assessing Large Vision Language Models in Open-Ended Video Question Answering | https://eccv.ecva.net/virtual/2024/poster/1092

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1092

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose a novel and challenging benchmark, AutoEval-Video, to comprehensively evaluate large vision-language models in open-ended video question answering. The comprehensiveness of AutoEval-Video is demonstrated in two aspects: 1) AutoEval-Video constructs open-ended video-questions across 9 skill dimensions, addressing capabilities of perception, comprehension, and generation. 2) AutoEval-Video contains newly collected videos that cover over 40 distinct themes. To efficiently evaluate responses to the open-ended questions, we employ an LLM-based evaluation approach, but instead of merely providing a reference answer, we annotate unique evaluation rules for every single instance (video-question pair). To maximize the robustness of these rules, we develop a novel adversarial annotation mechanism. By using instance-specific rules as prompt, GPT-4, as an automatic evaluator, can achieve a stable evaluation accuracy of around 97.0%, comparable to the 94.9% - 97.5% accuracy of a human evaluator. Furthermore, we assess the performance of eight large vision-language models on AutoEval-Video. Among them, GPT-4V(ision) significantly outperforms other models, achieving an accuracy of 32.2%. However, there is still substantial room for improvement compared to human accuracy of 72.8%. By conducting an extensive case study, we uncover several drawbacks of GPT-4V, such as limited temporal and dynamic comprehension, and overly general responses.

</details>

---

## 9. VITATECS: A Diagnostic Dataset for Temporal Concept Understanding of Video-Language Models

- [ ] VITATECS: A Diagnostic Dataset for Temporal Concept Understanding of Video-Language Models | https://eccv.ecva.net/virtual/2024/poster/1091

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1091

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability to perceive how objects change over time is a crucial ingredient in human intelligence. However, current benchmarks cannot faithfully reflect the temporal understanding abilities of video-language models (VidLMs) due to the existence of static visual shortcuts. To remedy this issue, we present VITATECS, a diagnostic VIdeo-Text dAtaset for the evaluation of TEmporal Concept underStanding. Specifically, we first introduce a fine-grained taxonomy of temporal concepts in natural language in order to diagnose the capability of VidLMs to comprehend different temporal aspects. Furthermore, to disentangle the correlation between static and temporal information, we generate counterfactual video descriptions that differ from the original one only in the specified temporal aspect. We employ a semi-automatic data collection framework using large language models and human-in-the-loop annotation to obtain high-quality counterfactual descriptions efficiently. Evaluation of representative video-language understanding models confirms their deficiency in temporal understanding, revealing the need for greater emphasis on the temporal elements in video-language research. Our dataset is publicly available at https://github.com/lscpku/VITATECS.

</details>

---

## 10. QUAR-VLA: Vision-Language-Action Model for Quadruped Robots

- [ ] QUAR-VLA: Vision-Language-Action Model for Quadruped Robots | https://eccv.ecva.net/virtual/2024/poster/1103

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1103

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The important manifestation of robot intelligence is the ability to naturally interact and autonomously make decisions. Traditional approaches to robot control often compartmentalize perception, planning, and decision-making, simplifying system design but limiting the synergy between different information streams. This compartmentalization poses challenges in achieving seamless autonomous reasoning, decision-making, and action execution. To address these limitations, a novel paradigm, named Vision-Language-Action tasks for QUAdruped Robots (QUAR-VLA), has been introduced in this paper. This approach tightly integrates visual information and instructions to generate executable actions, effectively merging perception, planning, and decision-making. The central idea is to elevate the overall intelligence of the robot. Within this framework, a notable challenge lies in aligning fine-grained instructions with visual perception information. This emphasizes the complexity involved in ensuring that the robot accurately interprets and acts upon detailed instructions in harmony with its visual observations. Consequently, we propose QUAdruped Robotic Transformer (QUART), a VLA model to integrate visual information and instructions from diverse modalities as input and generates executable actions for real-world robots and present QUAdruped Robot Dataset (QUARD), a large-scale multi-task dataset including navigation, complex terrain locomotion, and whole-body manipulation tasks for training QUART models. Our extensive evaluation shows that our approach leads to performant robotic policies and enables QUART to obtain a range of generalization capabilities.

</details>

---

## 11. M3DBench: Towards Omni 3D Assistant with Interleaved Multi-modal Instructions

- [ ] M3DBench: Towards Omni 3D Assistant with Interleaved Multi-modal Instructions | https://eccv.ecva.net/virtual/2024/poster/1106

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1106

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, the understanding of the 3D world has garnered increased attention, facilitating autonomous agents to perform further decision-making. However, the majority of existing 3D vision-language datasets and methods are often limited to specific tasks, limiting their applicability in diverse scenarios. The recent advance of Large Language Models (LLMs) and Multi-modal Language Models (MLMs) has shown mighty capability in solving various language and image tasks. Therefore, it is interesting to unlock MLM’s potential to be an omni 3D assistant for wider tasks. However, current MLMs’ research has been less focused on 3D due to the scarcity of large-scale visual-language datasets. In this work, we introduce M3DBench, a comprehensive multi-modal instruction dataset for complex 3D environments with over 320k instruction-response pairs that: 1) supports general interleaved multi-modal instructions with text, user clicks, images, and other visual prompts, 2) unifies diverse region- and scene-level 3D tasks, composing various fundamental abilities in real-world 3D environments. Furthermore, we establish a new benchmark for assessing the performance of large models in understanding interleaved multi-modal instructions. With extensive quantitative and qualitative experiments, we show the effectiveness of our dataset and baseline model in understanding complex human-environment interactions and accomplishing general 3D-centric tasks. We will release the data and code to accelerate future research on developing 3D MLMs.

</details>

---

## 12. Navigation Instruction Generation with BEV Perception and Large Language Models

- [ ] Navigation Instruction Generation with BEV Perception and Large Language Models | https://eccv.ecva.net/virtual/2024/poster/1104

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1104

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Navigation instruction generation, which requires embodied agents to describe the navigation routes, has been of great interest in robotics and human-computer interaction. Existing studies directly map the sequence of 2D  perspective observations to route descriptions. Though straightforward, they overlook the geometric information and object semantics of the 3D environment. To address these challenges, we propose BEVInstructor that incorporates Bird’s Eye View (BEV) features into Multi-Modal Large Language Models (MLLMs) for embodied instruction generation. Specifically, BEVInstructor constructs a Perspective-BEV Visual Encoder to boost the comprehension of 3D environments by fusing the BEV and perspective features. The fused embeddings are served as visual prompts for MLLMs. To leverage the powerful language capabilities of MLLMs,  perspective-BEV prompt tuning is proposed for parameter-efficient updating. Based on the perspective-BEV prompts, we further devise an instance-guided iterative refinement pipeline, which improves the instructions in a progressive manner. BEVInstructor achieves impressive performance across diverse datasets (i.e., R2R, REVERIE, and UrbanWalk). Our code will be released.

</details>

---

## 13. UMBRAE: Unified Multimodal Brain Decoding

- [ ] UMBRAE: Unified Multimodal Brain Decoding | https://eccv.ecva.net/virtual/2024/poster/1109

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1109

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we aim to tackle the two prevailing challenges in brain-powered research.  To extract instance-level conceptual and spatial details from neural signals, we introduce an efficient universal brain encoder for multimodal-brain alignment and recover object descriptions at multiple levels of granularity from subsequent multimodal large language models. To overcome unique brain patterns of different individuals, we introduce a cross-subject training strategy.  This allows neural signals from multiple subjects to be trained within the same model without additional training resources or time, and benefits from user diversity, yielding better results than focusing on a single subject. To better assess our method, we have introduced a comprehensive brain understanding benchmark BrainHub.  Experiments demonstrate that our proposed method UMBRAE not only achieves superior results in the newly introduced tasks but also outperforms models in established tasks with recognized metrics. Code and data will be made publicly available to facilitate further research.

</details>

---

## 14. Unifying 3D Vision-Language Understanding via Promptable Queries

- [ ] Unifying 3D Vision-Language Understanding via Promptable Queries | https://eccv.ecva.net/virtual/2024/poster/1108

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1108

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A unified model for 3D vision-language (3D-VL) understanding is expected to take various scene representations and perform a wide range of tasks in a 3D scene. However, a considerable gap exists between existing methods and such a unified model. In this paper, we introduce PQ3D, a unified model capable of using Promptable Queries to tackle a wide range of 3D-VL tasks, from low-level instance segmentation to high-level reasoning and planning. This is achieved through three key innovations: (1) unifying different 3D scene representations (\ie, voxels, point clouds, multi-view images) into a common 3D coordinate space by segment-level grouping, (2) an attention-based query decoder for task-specific information retrieval guided by prompts, and (3) universal output heads for different tasks to support multi-task training. Tested across ten diverse 3D-VL datasets, PQ3D demonstrates superior performance on most tasks, setting new records on most benchmarks. Particularly, PQ3D boosts the state-of-the-art on ScanNet200 by 1.8% (AP), ScanRefer by 5.4% (acc@0.5), Multi3DRef by 11.7% (F1@0.5), and Scan2Cap by 13.4% (CIDEr@0.5). Moreover, PQ3D supports flexible inference with whatever 3D representations are available, e.g., solely relying on voxels.

</details>

---

## 15. A Comprehensive Study of Multimodal Large Language Models for Image Quality Assessment

- [ ] A Comprehensive Study of Multimodal Large Language Models for Image Quality Assessment | https://eccv.ecva.net/virtual/2024/poster/1112

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1112

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While Multimodal Large Language Models (MLLMs) have experienced significant advancement on visual understanding and reasoning, the potential they hold as powerful, flexible and text-driven models for Image Quality Assessment (IQA) remains largely unexplored. In this paper, we conduct a comprehensive study of prompting MLLMs for IQA at the system level. Specifically, we first investigate nine system-level prompting methods for MLLMs as the combinations of three standardized testing procedures in psychophysics (i.e., the single-stimulus, double-stimulus, and multiple-stimulus methods) and three popular prompting tricks in natural language processing (i.e., standard, in-context, and chain-of-thought prompting). We then propose a difficult sample selection procedure, taking into account sample diversity and human uncertainty, to further challenge MLLMs coupled with the respective optimal prompting methods identified in the previous step. In our experiments, we assess three open-source and one close-source MLLMs on several visual attributes of image quality (e.g., structural and textural distortions, color differences, and geometric transformations) under both full-reference and no-reference settings, and gain valuable insights into the development of better MLLMs for IQA.

</details>

---

## 16. CoReS: Orchestrating the Dance of Reasoning and Segmentation

- [ ] CoReS: Orchestrating the Dance of Reasoning and Segmentation | https://eccv.ecva.net/virtual/2024/poster/1111

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1111

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The reasoning segmentation task, which demands a nuanced comprehension of intricate queries to accurately pinpoint object regions, is attracting increasing attention. However, Multi-modal Large Language Models (MLLM) often find it difficult to accurately localize the objects described in complex reasoning contexts. We believe that the act of reasoning segmentation should mirror the cognitive stages of human visual search, where each step is a progressive refinement of thought toward the final object. Thus we introduce the Chains of Reasoning and Segmenting (CoReS) and find this top-down visual hierarchy indeed enhances the visual search process. Specifically, we propose a dual-chain structure that generates multi-modal, chain-like outputs to aid the segmentation process. Furthermore, to steer the MLLM's outputs into this intended hierarchy, we incorporate in-context inputs as guidance. Extensive experiments demonstrate the superior performance of our CoReS, which surpasses the state-of-the-art method by 7.1% on the ReasonSeg dataset. The code will be released soon.

</details>

---

## 17. Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models

- [ ] Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models | https://eccv.ecva.net/virtual/2024/poster/1114

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1114

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Groma, a Multimodal Large Language Model (MLLM) with grounded and fine-grained visual perception ability. Beyond holistic image understanding, Groma is adept at region-level tasks such as region captioning and visual grounding. Such capabilities are built upon a localized visual tokenization mechanism, where an image input is decomposed into regions of interest and subsequently encoded into region tokens. By integrating region tokens into user instructions and model responses, we seamlessly enable Groma to understand user-specified region inputs and ground its textual output to images. Besides, to enhance the grounded chat ability of Groma, we curate a visually grounded instruction dataset by leveraging the powerful GPT-4V and visual prompting techniques. Compared with MLLMs that rely on the language model or external module for localization, Groma consistently demonstrates superior performances in standard referring and grounding benchmarks, highlighting the advantages of embedding localization into image tokenization. Project page: https://groma-mllm.github.io/.

</details>

---

## 18. Grounding Language Models for Visual Entity Recognition

- [ ] Grounding Language Models for Visual Entity Recognition | https://eccv.ecva.net/virtual/2024/poster/1113

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1113

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce AutoVER, an Autoregressive model for Visual Entity Recognition. Our model extends an autoregressive Multi-modal Large Language Model by employing retrieval augmented constrained generation. It mitigates low performance on out-of-domain entities while excelling in queries that require visual reasoning. Our method learns to distinguish similar entities within a vast label space by contrastively training on hard negative pairs in parallel with a sequence-to-sequence objective without an external retriever. During inference, a list of retrieved candidate answers explicitly guides language generation by removing invalid decoding paths. The proposed method achieves significant improvements across different dataset splits in the recently proposed Oven-Wiki benchmark with accuracy on the Entity seen split rising from 32.7% to 61.5%. It demonstrates superior performance on the unseen and query splits by a substantial double-digit margin, while also preserving the ability to effectively transfer to other generic visual question answering benchmarks without further training.

</details>

---

## 19. The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?

- [ ] The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models? | https://eccv.ecva.net/virtual/2024/poster/1115

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1115

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layer of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against multi-modal jailbreaking attack, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, indicating potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicting uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improve models' performance but is still inferior to linear probing on these tasks.

</details>

---

## 20. AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting

- [ ] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting | https://eccv.ecva.net/virtual/2024/poster/1116

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1116

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), the imperative to ensure their safety has become increasingly pronounced. However, with the integration of additional modalities, MLLMs are exposed to new vulnerabilities, rendering them prone to structured-based jailbreak attacks, where semantic content (e.g., ``harmful text'') has been injected into the images to mislead MLLMs.  In this work, we aim to defend against such threats.  Specifically, we propose Adaptive Shield Prompting (AdaShield), which prepends inputs with defense prompts to defend MLLMs against structure-based jailbreak attacks without fine-tuning MLLMs or training additional modules (e.g., post-stage content detector). Initially, we present a manually designed static defense prompt, which thoroughly examines the image and instruction content step by step and specifies response methods to malicious queries.  Furthermore, we introduce an adaptive auto-refinement framework, consisting of a target MLLM and a LLM-based defense prompt generator (Defender). These components collaboratively and iteratively communicate to generate a defense prompt. Extensive experiments on the popular structure-based jailbreak attacks and benign datasets show that our methods can consistently improve MLLMs' robustness against structure-based jailbreak attacks without compromising the model's general capabilities evaluated on standard benign tasks.

</details>

---

## 21. X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs

- [ ] X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs | https://eccv.ecva.net/virtual/2024/poster/1118

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1118

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have revolutionized the field of vision-language understanding by incorporating visual perceptioning capabilities into Large Language Models (LLMs). The prevailing trend in this field involves the utilization of a vision encoder derived from vision-language contrastive learning (CL),  showing expertise  in capturing overall representations while facing difficulties in capturing detailed local patterns. In this work, we focus on enhancing the visual representations for MLLMs by combining high-frequency and fine-grained representations, obtained through masked image modeling (MIM), with semantically-enriched low-frequency representations captured by CL. To achieve this goal, we introduce X-Former which is a lightweight transformer module designed to exploit the complementary strengths of CL and MIM through an innovative interaction mechanism. Specifically, X-Former first bootstraps vision-language representation learning and multimodal-to-multimodal generative learning from two frozen vision encoders, i.e., CLIP-ViT (CL-based) \cite{radford2021clip} and MAE-ViT (MIM-based) \cite{he2022mae}. It further bootstraps vision-to-language generative learning from a frozen LLM to ensure visual features from X-Former can be interpreted by the LLM. To demonstrate the effectiveness of our approach, we assess its performance on tasks demanding fine-grained visual understanding. Our extensive empirical evaluations indicate that X-Former excels in visual reasoning tasks encompassing both structural and semantic categories within the GQA dataset. Assessment on a fine-grained visual perception benchmark further confirms its superior capabilities in visual understanding.

</details>

---

## 22. UniCode : Learning a Unified Codebook for Multimodal Large Language Models

- [ ] UniCode : Learning a Unified Codebook for Multimodal Large Language Models | https://eccv.ecva.net/virtual/2024/poster/1117

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1117

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose \textbf{UniCode}, a novel approach within the domain of multimodal large language models (MLLMs) that learns a unified codebook to efficiently tokenize visual, text, and potentially other types of signals. This innovation addresses a critical limitation in existing MLLMs: their reliance on a text-only codebook, which restricts MLLM's ability to generate images and texts in a multimodal context. Towards this end, we propose a language-driven iterative training paradigm, coupled with an in-context pre-training task we term ``image decompression'', enabling our model to interpret compressed visual data and generate high-quality images. The unified codebook empowers our model to extend visual instruction tuning to non-linguistic generation tasks. Moreover, UniCode is adaptable to diverse stacked quantization approaches in order to compress visual signals into a more compact token representation. Despite using significantly fewer parameters and less data during training, Unicode demonstrates exceptional capabilities in visual reconstruction and generation. It also achieves performances comparable to leading MLLMs across a spectrum of VQA benchmarks. Our codes are available at \url{https://anonymous.4open.science/r/UniCode}.

</details>

---

## 23. The Hard Positive Truth about Vision-Language Compositionality

- [ ] The Hard Positive Truth about Vision-Language Compositionality | https://eccv.ecva.net/virtual/2024/poster/1122

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1122

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Several benchmarks have concluded that our best vision-language models (e.g., CLIP) are lacking in compositionality. Given an image, these benchmarks probe a model's ability to identify its associated caption amongst a set of compositional distractors. In response, a surge of recent proposals show improvements by finetuning CLIP with distractors as hard negatives. Our investigations reveal that these improvements have been overstated --- because existing benchmarks do not probe whether finetuned models remain invariant to hard positives. By curating an evaluation dataset with 112,382 both hard negatives and hard positives, we uncover that including hard positives decreases CLIP's performance by 12.9%, while humans perform effortlessly at 99%. CLIP finetuned with hard negatives results in an even larger decrease, up to 38.7%. With this finding, we then produce a 1,775,259 training set with both hard negatives and hard positives captions. By training with both, we see improvements on existing benchmarks while simultaneously improving performance on hard positives, indicating an improvement in compositionality. Our work suggests the need for future research to rigorously test and improve CLIP's understanding of semantic relationships between related ``positive'' concepts.

</details>

---

## 24. Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities

- [ ] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities | https://eccv.ecva.net/virtual/2024/poster/1121

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1121

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language (VL) models have sparked interest in their deployment on edge devices, yet challenges in handling diverse visual modalities, manual annotation, and computational constraints remain. We introduce EdgeVL, a novel framework that bridges this gap by seamlessly integrating dual-modality knowledge distillation and quantization-aware contrastive learning. This approach enables the adaptation of large VL models, like CLIP, for efficient use with both RGB and non-RGB images on resource-limited devices without the need for manual annotations. EdgeVL not only transfers visual language alignment capabilities to compact models but also maintains feature quality post-quantization, significantly enhancing open-vocabulary classification performance across various visual modalities. Our work represents the first systematic effort to adapt large VL models for edge deployment, showcasing up to 15.4% accuracy improvements on multiple datasets and up to 93-fold reduction in model size.

</details>

---

## 25. EventBind: Learning a Unified Representation to Bind Them All for Event-based Open-world Understanding

- [ ] EventBind: Learning a Unified Representation to Bind Them All for Event-based Open-world Understanding | https://eccv.ecva.net/virtual/2024/poster/1119

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1119

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose EventBind, a novel and effective framework that unleashes the potential of vision-language models (VLMs) for event-based recognition to compensate for the lack of large-scale event-based datasets. In particular, due to the distinct modality gap with the image-text data and the lack of large-scale datasets, learning a common representation space for images, texts, and events is non-trivial. Intuitively, we need to address two key challenges: 1) how to generalize CLIP's visual encoder to event data while fully leveraging events' unique properties, e.g., sparsity and high temporal resolution; 2) how to effectively align the multi-modal embeddings, i.e., image, text, and events. Accordingly, we first introduce a novel event encoder that subtly models the temporal information from events and meanwhile generates event prompts for modality bridging. We then design a text encoder that generates content prompts and utilizes hybrid text prompts to enhance EventBind's generalization ability across diverse datasets. With the proposed event encoder, text encoder, and image encoder, a novel Hierarchical Triple Contrastive Alignment (HTCA}) module is introduced to jointly optimize the correlation and enable efficient knowledge transfer among the three modalities. We evaluate various settings, including fine-tuning and few-shot on three benchmarks and our EventBind achieves new state-of-art accuracy compared with the previous methods, such as N-Caltech101 (+5.34% and +1.70%) and N-Imagenet (+5.65% and +1.99%) with fine-tuning and 20-shot settings respectively. Moreover, our EventBind can be flexibly extended to the event retrieval task using text or image queries, showing plausible performance. Our project code will be made publicly available.

</details>

---

## 26. HiFi-Score: Fine-grained Image Description Evaluation with Hierarchical Parsing Graphs

- [ ] HiFi-Score: Fine-grained Image Description Evaluation with Hierarchical Parsing Graphs | https://eccv.ecva.net/virtual/2024/poster/1123

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1123

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the advancements of vision-language models, the growing demand for generating customized image descriptions under length, target regions, and other various control conditions brings new challenges for evaluation. Most existing metrics, designed primarily for single-sentence image captioning with an overall matching score, struggle to accommodate complex description requirements, resulting in insufficient accuracy and interpretability of evaluation. Therefore, we propose HiFi-Score, a hierarchical parsing graph-based fine-grained evaluation metric. Specifically, we model both text and images as parsing graphs, which organize multi-granular instances into a hierarchical structure according to their inclusion relationships, which provides a comprehensive scene analysis for both modalities from global to local. Based on the fine-grained matching between the graphs, we can evaluate the fidelity to ensure text contents are related to image and the adequacy to ensure the image is covered by text at multiple levels. Furthermore, we employ the large language model to evaluate fluency of the language expression. Human correlation experiments on four caption-level benchmarks show that the proposed metric outperforms existing metrics. At the paragraph-level, we construct a novel dataset ParaEval and demonstrate the accuracy of the HiFi-Score in evaluating long texts. We further show its superiority in assessing vision-language models and its flexibility when applied to various image description tasks.

</details>

---

## 27. LLMCO4MR: LLMs-aided Neural Combinatorial Optimization for Ancient Manuscript Restoration from Fragments with Case Studies on Dunhuang

- [ ] LLMCO4MR: LLMs-aided Neural Combinatorial Optimization for Ancient Manuscript Restoration from Fragments with Case Studies on Dunhuang | https://eccv.ecva.net/virtual/2024/poster/1124

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1124

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Restoring ancient manuscripts fragments, such as those from Dunhuang, is crucial for preserving human historical culture. However, their worldwide dispersal and the shifts in cultural and historical contexts pose significant restoration challenges. Traditional archaeological efforts primarily focus on manually piecing major fragments together, yet the vast majority of small, more intricate pieces remain largely unexplored, which is technically due to their irregular shapes, sparse textual content, and extensive combinatorial space for reassembly. In this paper, we formalize the task of restoring the ancient manuscript from fragments as a cardinality-constrained combinatorial optimization problem, and propose a general framework named LLMCO4MS: (Multimodal) Large Language Model-aided Combinatorial Optimization Neural Networks for Ancient Manuscript Restoration. Specifically, LLMCO4MS encapsulates a neural combinatorial solver equipped with a differentiable optimal transport (OT) layer, to efficiently predict the Top-K likely reassembly candidates. Innovatively, the Multimodal Large Language Model (MLLM) is then adopted and prompted to yield pairwise matching confidence and relative directions for final restoration. Extensive experiments on both synthetic data and real-world famous Dunhuang fragments demonstrate superior performance of our approach. In particular, LLMCO4MS has facilitated the discovery of previously unknown civilian economic documents from the 10-th century in real-world applications.

</details>

---

## 28. Language-Image Pre-training with Long Captions

- [ ] Language-Image Pre-training with Long Captions | https://eccv.ecva.net/virtual/2024/poster/1125

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1125

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Language-image pre-training largely relies on how precisely and thoroughly a text describes its paired image. In practice, however, the contents of an image can be so rich that well describing them requires lengthy captions (\textit{e.g.}, with 10 sentences), which are usually missing in existing datasets. Consequently, there are currently no clear evidences on whether and how language-image pre-training could benefit from long captions. To figure this out, we first re-caption 30M images with detailed descriptions using a pre-trained Multi-modality Large Language Model (MLLM), and then study the usage of the resulting captions under a contrastive learning framework. We observe that, each sentence within a long caption is very likely to describe the image partially (\textit{e.g.}, an object). Motivated by this, we propose to dynamically sample sub-captions from the text label to construct multiple positive pairs, and introduce a grouping loss to match the embeddings of each sub-caption with its corresponding local image patches in a self-supervised manner. Experimental results on a wide rage of downstream tasks demonstrate the consistent superiority of our method, termed DreamLIP, over previous alternatives, highlighting its fine-grained representational capacity. It is noteworthy that, on the tasks of image-text retrieval and semantic segmentation, our model trained with 30M image-text pairs achieves on par or even better performance than CLIP trained with 400M pairs. The code and generated captions will be made publicly available.

</details>

---

## 29. Cascade Prompt Learning for Visual-Language Model Adaptation

- [ ] Cascade Prompt Learning for Visual-Language Model Adaptation | https://eccv.ecva.net/virtual/2024/poster/1130

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1130

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has surfaced as an effective approach to enhance the performance of Vision-Language Models (VLMs) like CLIP when applied to downstream tasks. However, current learnable prompt tokens are primarily used for the single phase of adapting to tasks (i.e., adapting prompt), easily leading to overfitting risks. In this work, we propose a novel Cascade Prompt Learning (CasPL) framework to enable prompt learning to serve both generic and specific expertise (i.e., boosting and adapting prompt) simultaneously. Specifically, CasPL is a new learning paradigm comprising two distinct phases of learnable prompts: the first boosting prompt is crafted to extract domain-general knowledge from a senior larger CLIP teacher model by aligning their predicted logits using extensive unlabeled domain images. The second adapting prompt is then cascaded with the frozen first set to fine-tune the downstream tasks, following the approaches employed in prior research. In this manner, CasPL can effectively capture both domain-general and task-specific representations into explicitly different gradual groups of prompts, thus potentially alleviating overfitting issues in the target domain. It's worth noting that CasPL serves as a plug-and-play module that can seamlessly integrate into any existing prompt learning approach. CasPL achieves a significantly better balance between performance and inference speed, which is especially beneficial for deploying smaller VLM models in resource-constrained environments. Compared to the previous state-of-the-art method PromptSRC, CasPL shows an average improvement of 1.85% for base classes, 3.44% for novel classes, and 2.72% for the harmonic mean over 11 image classification datasets.

</details>

---

## 30. ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling

- [ ] ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling | https://eccv.ecva.net/virtual/2024/poster/1133

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1133

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recognizing and disentangling visual attributes from objects is a foundation to many computer vision applications. While large vision language representations like CLIP had largely resolved the task of zeroshot object recognition, zero-shot visual attribute recognition remains a challenge because CLIP’s contrastively-learned vision-language representation cannot effectively capture object-attribute dependencies. In this paper, we target this weakness and propose a sentence generation-based retrieval formulation for attribute recognition that is novel in 1) explicitly modeling a to-be-measured and retrieved object-attribute relation as a conditional probability graph, which converts the recognition problem into a dependency-sensitive language-modeling problem, and 2) applying a large pretrained Vision-Language Model (VLM) on this reformulation and naturally distilling its knowledge of image-object-attribute relations to use towards attribute recognition. Specifically, for each attribute to be recognized on an image, we measure the visual-conditioned probability of generating a short sentence encoding the attribute’s relation to objects on the image. Unlike contrastive retrieval, which measures likelihood by globally aligning elements of the sentence to the image, generative retrieval is sensitive to the order and dependency of objects and attributes in the sentence. We demonstrate through experiments that generative retrieval consistently outperforms contrastive retrieval on two visual reasoning datasets, Visual Attribute in the Wild (VAW), and our newly-proposed Visual Genome Attribute Ranking (VGARank).

</details>

---

## 31. SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression Segmentation

- [ ] SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression Segmentation | https://eccv.ecva.net/virtual/2024/poster/1139

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1139

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce SAM4MLLM, an innovative approach which integrates the Segment Anything Model (SAM) with Multi-Modal Large Language Models (MLLMs) for pixel-aware tasks. Our method enables MLLMs to learn pixel-level   location information without requiring excessive modifications to the existing model architecture or adding specialized tokens. We introduce an inquiry-based approach that can effectively find prompt points for SAM to perform segmentation based on MLLM. It combines detailed visual information with the powerful expressive capabilities of large language models in a unified language-based manner without additional computational overhead in learning. Experimental results on pubic benchmarks demonstrate the effectiveness of our approach.

</details>

---

## 32. MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection

- [ ] MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection | https://eccv.ecva.net/virtual/2024/poster/1137

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1137

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Learning from pseudo-labels that generated with VLMs (Vision Language Models) has been shown as a promising solution to assist open vocabulary detection (OVD) in recent studies. However, due to the domain gap between VLM and vision-detection tasks, pseudo-labels produced by the VLMs are prone to be noisy, while the training design of the detector further amplifies the bias. In this work, we investigate the root cause of VLMs' biased prediction under the OVD context. Our observations lead to a simple yet effective paradigm, coded MarvelOVD, that generates significantly better training targets and optimizes the learning procedure in an online manner by marrying the capability of the detector with the vision-language model. Our key insight is that the detector itself can act as a strong auxiliary guidance to accommodate VLM's inability of understanding both the ` background'' and the context of a proposal within the image. Based on it, we greatly purify the noisy pseudo-labels via Online Mining and propose Adaptive Reweighting to effectively suppress the biased training boxes that are not well aligned with the target object. In addition, we also identify a neglected base-novel-conflict' problem and introduce stratified label assignments to prevent it. Extensive experiments on COCO and LVIS datasets demonstrate that our method outperforms the other state-of-the-arts by significant margins. Code will be released.

</details>

---

## 33. Dense Multimodal Alignment for Open-Vocabulary 3D Scene Understanding

- [ ] Dense Multimodal Alignment for Open-Vocabulary 3D Scene Understanding | https://eccv.ecva.net/virtual/2024/poster/1138

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1138

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language pre-training models have exhibited remarkable generalization ability in zero-shot recognition tasks. However, their applications to 3D dense prediction tasks often encounter the difficulties of limited high-quality and densely-annotated 3D data. Previous open-vocabulary 3D scene understanding methods mostly focus on training 3D models using either image or text supervision while neglecting the collective strength of all modalities. In this work, we propose a Dense Multimodal Alignment (DMA) framework to densely co-embed different modalities into a common space for maximizing their synergistic benefits. Instead of extracting coarse view- or region-level text prompts, we leverage large vision-language models to extract complete category information and scalable scene descriptions to build the text modality, and take image modality as the bridge to build dense point-pixel-text associations. Besides, in order to enhance the generalization ability of the 2D model for downstream 3D tasks without compromising the open-vocabulary capability, we employ a dual-path integration approach to combine frozen CLIP visual features and learnable mask features. Extensive experiments show that our DMA method produces highly competitive open-vocabulary segmentation performance on various indoor and outdoor tasks.

</details>

---

## 34. ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference

- [ ] ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference | https://eccv.ecva.net/virtual/2024/poster/1141

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1141

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the success of large-scale pretrained Vision-Language Models (VLMs) especially CLIP in various open-vocabulary tasks, their application to semantic segmentation remains challenging, producing noisy segmentation maps with mis-segmented regions. In this paper, we carefully re-investigate the architecture of CLIP, and identify residual connections as the primary source of noise that degrades segmentation quality. With a comparative analysis of statistical properties in the residual connection and the attention output across different pretrained models, we discover that CLIP's image-text contrastive training paradigm emphasizes global features at the expense of local discriminability, leading to noisy segmentation results. In response, we propose ClearCLIP, a novel approach that decomposes CLIP's representations to enhance open-vocabulary semantic segmentation. We introduce three simple modifications to the final layer: removing the residual connection, implementing the self-self attention, and discarding the feed-forward network. ClearCLIP consistently generates clearer and more accurate segmentation maps and outperforms existing approaches across multiple benchmarks, affirming the significance of our discoveries.

</details>

---

## 35. Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation

- [ ] Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation | https://eccv.ecva.net/virtual/2024/poster/1142

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1142

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

CLIP, as a vision-language model, has significantly advanced Open-Vocabulary Semantic Segmentation (OVSS) with its zero-shot capabilities. Despite its success, its application to OVSS faces challenges due to its initial image-level alignment training, which affects its performance in tasks requiring detailed local context. Our study delves into the impact of CLIP's [CLS] token on patch feature correlations, revealing a dominance of "global" patches that hinders local feature discrimination. To overcome this, we propose CLIPtrase, a novel training-free semantic segmentation strategy that enhances local feature awareness through recalibrated self-correlation among patches. This approach demonstrates notable improvements in segmentation accuracy and the ability to maintain semantic coherence across objects. Experiments show that we are 22.3% ahead of CLIP on average on 9 segmentation benchmarks, outperforming existing state-of-the-art training-free methods.

</details>

---

## 36. Prioritized Semantic Learning for Zero-shot Instance Navigation

- [ ] Prioritized Semantic Learning for Zero-shot Instance Navigation | https://eccv.ecva.net/virtual/2024/poster/1145

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1145

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We study zero-shot instance navigation, in which the agent navigates to a specific object without using object annotations for training. Previous object navigation approaches apply the image-goal navigation (ImageNav) task (go to the location of an image) for pretraining, and transfer the agent to achieve object goals using a vision-language model. However, these approaches lead to issues of semantic neglect, where the model fails to learn meaningful semantic alignments. In this paper, we propose a Prioritized Semantic Learning (PSL) method to improve the semantic understanding ability of navigation agents. Specifically, a semantic-enhanced PSL agent is proposed and a prioritized semantic training strategy is introduced to select goal images that exhibit clear semantic supervision and relax the reward function from strict exact view matching. At inference time, a semantic expansion inference scheme is designed to preserve the same granularity level of the goal-semantic as training. Furthermore, for the popular HM3D environment, we present an Instance Navigation (InstanceNav) task that requires going to a specific object instance with detailed descriptions, as opposed to the Object Navigation (ObjectNav) task where the goal is defined merely by the object category. Our PSL agent outperforms the previous state-of-the-art by 66% on zero-shot ObjectNav in terms of success rate and is also superior on the new InstanceNav task.

</details>

---

## 37. SemiVL: Semi-Supervised Semantic Segmentation with Vision-Language Guidance

- [ ] SemiVL: Semi-Supervised Semantic Segmentation with Vision-Language Guidance | https://eccv.ecva.net/virtual/2024/poster/1147

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1147

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In semi-supervised semantic segmentation, a model is trained with a limited number of labeled images along with a large corpus of unlabeled images to reduce the high annotation effort. While previous methods are able to learn good segmentation boundaries, they are prone to confuse classes with similar visual appearance due to the limited supervision. On the other hand, vision-language models (VLMs) are able to learn diverse semantic knowledge from image-caption datasets but produce noisy segmentation due to the image-level training. In SemiVL, we newly propose to integrate rich priors from VLM pre-training into semi-supervised semantic segmentation to learn better semantic decision boundaries. To adapt the VLM from global to local reasoning, we introduce a spatial fine-tuning strategy for label-efficient learning. Further, we design a language-guided decoder to jointly reason over vision and language. Finally, we propose to handle inherent ambiguities in class labels by instructing the model with language guidance in the form of class definitions. We evaluate SemiVL on 4 semantic segmentation datasets, where it significantly outperforms previous semi-supervised methods. For instance, SemiVL improves the state of the art by +13.5 mIoU on COCO with 232 annotated images and by +6.1 mIoU on Pascal VOC with 92 annotated images. The source code will be released with the paper.

</details>

---

## 38. Adaptive Multi-task Learning for Few-shot Object Detection

- [ ] Adaptive Multi-task Learning for Few-shot Object Detection | https://eccv.ecva.net/virtual/2024/poster/1155

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1155

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The majority of few-shot object detection methods use a shared feature map for both classification and localization, despite the conflicting requirements of these two tasks. Localization needs scale and positional sensitive features, whereas classification requires features that are robust to scale and positional variations. Although few methods have recognized this challenge and attempted to address it, they may not provide a comprehensive resolution to the issue. To overcome the contradictory preferences between classification and localization in few-shot object detection, an adaptive multi-task learning method, featuring a novel precision-driven gradient balancer, is proposed. This balancer effectively mitigates the conflicts by dynamically adjusting the backward gradient ratios for both tasks. Furthermore, a knowledge distillation and classification refinement scheme based on CLIP is introduced, aiming to enhance individual tasks by leveraging the capabilities of large vision-language models. Experimental results of the proposed method consistently show improvements over strong few-shot detection baselines on benchmark datasets.

</details>

---

## 39. Unified Medical Image Pre-training in Language-Guided Common Semantic Space

- [ ] Unified Medical Image Pre-training in Language-Guided Common Semantic Space | https://eccv.ecva.net/virtual/2024/poster/1165

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1165

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pre-training (VLP) has shown the merits of analysing medical images. It efficiently learns visual representations by leveraging supervisions in their corresponding reports, and in turn facilitates analysis and interpretation of intricate imaging data. However, such observation is predominantly justified on single-modality data (mostly 2D images like X-rays), adapting VLP to learning unified representations for medical images in real scenario remains an open challenge. This arises from medical images often encompass a variety of modalities, especially modalities with different dimensions (e.g., 3D images like Computed Tomography), and there are almost no paired multi-dimension data here. To overcome the aforementioned challenges, we propose an \textbf{U}nified \textbf{Med}ical \textbf{I}mage Pre-training framework, namely UniMedI, which utilizes diagnostic reports as common semantic space to create unified representations for diverse modalities of medical images (especially for 2D and 3D images). Under the text's guidance, UniMedI effectively select text-related 2D slices from sophisticated 3D volume, which acts as pseudo-pairs to bridge 2D and 3D data, ultimately enhancing the consistency across various medical imaging modalities. To demonstrate the effectiveness and versatility of UniMedI, we evaluate its performance on both 2D and 3D images across several different datasets, covering a wide range of medical image tasks such as classification, segmentation, and retrieval. UniMedI has demonstrated superior performance in downstream tasks, showcasing its effectiveness in establishing a universal medical visual representation.

</details>

---

## 40. VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation

- [ ] VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation | https://eccv.ecva.net/virtual/2024/poster/1169

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1169

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, large-scale vision-language models such as CLIP have demonstrated immense potential in zero-shot anomaly segmentation (ZSAS) task, utilizing a unified model to directly detect anomalies on any unseen product given painstakingly crafted text prompts. However, existing methods often assume that the product category to be inspected is known, thus setting product-specific text prompts, which is difficult to achieve in the data privacy scenario. Moreover, even the same type of product exhibit significant differences due to specific components and variations in the production process, posing significant challenges to the design of text prompts. In this end, we propose a visual context prompting model (VCP-CLIP) for ZSAS task based on CLIP. The insight behind VCP-CLIP is to employ visual context prompting to activate CLIP’s anomalous semantic perception ability. In specific, we first design a Pre-VCP module to embed global visual information into the text prompt, thus eliminating the necessity for product-specific prompts. Then, we propose a novel Post-VCP module, that adjust the text embeddings utilizing the fine-grained features of the images. In extensive experiments conducted on 10 real-world industrial anomaly segmentation datasets, VCP-CLIP achieved state-of-the-art performance in ZSAS task. The implementation of this method will be published upon acceptance.

</details>

---

## 41. Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models

- [ ] Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1186

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1186

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study addresses the Domain-Class Incremental Learning problem, a realistic but challenging continual learning scenario where both the domain distribution and target classes vary across tasks. To handle these diverse tasks, pre-trained Vision-Language Models (VLMs) are introduced for their strong generalizability. However, this incurs a new problem: the knowledge encoded in the pre-trained VLMs may be disturbed when adapting to new tasks, compromising their inherent zero-shot ability. Existing methods tackle it by tuning VLMs with knowledge distillation on extra datasets, which demands heavy computation overhead. To address this problem efficiently, we propose the Distribution-aware Interference-free Knowledge Integration (DIKI) framework, retaining pre-trained knowledge of VLMs from a perspective of avoiding information interference. Specifically, we design a fully residual mechanism to infuse newly learned knowledge into a frozen backbone, while introducing minimal adverse impacts on pre-trained knowledge. Besides, this residual property enables our distribution-aware integration calibration scheme, explicitly controlling the information implantation process for test data from unseen distributions. Experiments demonstrate that our DIKI surpasses the current state-of-the-art approach using only 0.86% of the trained parameters and requiring substantially less training time.

</details>

---

## 42. Robust Calibration of Large Vision-Language Adapters

- [ ] Robust Calibration of Large Vision-Language Adapters | https://eccv.ecva.net/virtual/2024/poster/1187

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1187

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper addresses the critical issue of miscalibration in CLIP-based adaptation models in the challenging scenario of out-of-distribution samples, which has been overlooked in the existing literature on CLIP adaptation. We empirically demonstrate that popular CLIP adaptation approaches, such as adapters, prompt learning, and test-time prompt tuning, substantially degrade the calibration capabilities of the zero-shot baseline in the presence of distributional drift. We identify the increase in logit ranges as the underlying cause of miscalibration of CLIP adaptation methods, contrasting with previous work on calibrating fully supervised models. Motivated by these observations, we present a simple and model-agnostic solution to mitigate miscalibration, by scaling the logit range of each sample based on its zero-shot prediction logits. We explore three different alternatives to achieve this, which can be either integrated during adaptation, or directly used at inference time. Comprehensive experiments on popular OOD classification benchmarks demonstrate the effectiveness of the proposed approaches in mitigating miscalibration while maintaining discriminative performance, whose improvements are consistent across the three families of these increasingly popular approaches.

</details>

---

## 43. Benchmarking Spurious Bias in Few-Shot Image Classifiers

- [ ] Benchmarking Spurious Bias in Few-Shot Image Classifiers | https://eccv.ecva.net/virtual/2024/poster/1190

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1190

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Few-shot image classifiers are designed to recognize and classify new data with minimal supervision and limited data but often show reliance on spurious correlations between classes and spurious attributes, known as spurious bias. Spurious correlations commonly hold in a certain set of samples and few-shot  classifiers can suffer from spurious bias induced from it. There is an absence of an automatic benchmarking system to assess the robustness of few-shot classifiers against spurious bias.  In this paper, we propose a systematic and rigorous benchmark framework, termed FewSTAB, to fairly demonstrate and quantify varied degrees of robustness of few-shot classifiers to spurious bias. FewSTAB creates few-shot evaluation tasks with biased attributes so that using them for predictions can demonstrate poor performance. We propose attribute-based sample selection strategies based on a pre-trained vision-language model to construct these few-shot tasks, eliminating the need for manual dataset curation. This allows FewSTAB to automatically benchmark spurious bias using any existing test data.  FewSTAB offers evaluation results in a new dimension along with a new design guideline for building robust classifiers. Moreover, it can benchmark spurious bias in varied degrees and enable designs for varied degrees of robustness. Its effectiveness is demonstrated through experiments on ten few-shot learning methods across three datasets. We hope our framework can inspire new designs of robust few-shot classifiers. The code will be public upon acceptance.

</details>

---

## 44. LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning

- [ ] LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning | https://eccv.ecva.net/virtual/2024/poster/1257

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1257

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generating instructional images of human daily actions from an egocentric viewpoint serves a key step towards efficient skill transfer. In this paper, we introduce a novel problem -- egocentric action frame generation. The goal is to synthesize the action frame conditioning on the user prompt question and an input egocentric image that captures the user's environment. Notably, existing egocentric action datasets lack the detailed annotations that describe the execution of actions. Additionally, the existing diffusion-based image manipulation models are sub-optimal in controlling the state transition of an action in egocentric image pixel space because of the domain gap. To this end, we propose to Learn EGOcentric (LEGO) action frame  generation via visual instruction tuning. First, we introduce a prompt enhancement scheme to generate enriched action descriptions from a visual large language model (VLLM) by visual instruction tuning. Then we propose a novel method to leverage image and text embeddings from VLLM as additional conditioning to improve the performance of a diffusion model. We validate our model on two egocentric datasets -- Ego4D and Epic-Kitchens. Our experiments show prominent improvement over prior image manipulation models in both quantitative and qualitative evaluation. We also conduct detailed ablation studies and analysis to provide insights in our method.

</details>

---

## 45. DriveLM: Driving with Graph Visual Question Answering

- [ ] DriveLM: Driving with Graph Visual Question Answering | https://eccv.ecva.net/virtual/2024/poster/130

- **Link**: https://eccv.ecva.net/virtual/2024/poster/130

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We study how vision-language models (VLMs) trained on web-scale data can be integrated into end-to-end driving systems to boost generalization and enable interactivity with human users. While recent approaches adapt VLMs to driving via single-round visual question answering (VQA), human drivers reason about decisions in multiple steps. Starting from the localization of key objects, humans estimate object interactions before taking actions. The key insight is that with our proposed task, Graph VQA, where we model graph-structured reasoning through perception, prediction and planning question-answer pairs, we obtain a suitable proxy task to mimic the human reasoning process. We instantiate datasets (DriveLM-Data) built upon nuScenes and CARLA, and propose a VLM-based baseline approach (DriveLM-Agent) for jointly performing Graph VQA and end-to-end driving. The experiments demonstrate that Graph VQA provides a simple, principled framework for reasoning about a driving scene, and DriveLM-Data provides a challenging benchmark for this task. Our DriveLM-Agent baseline performs end-to-end autonomous driving competitively in comparison to state-of-the-art driving-specific architectures. Notably, its benefits are pronounced when it is evaluated zero-shot on unseen sensor configurations. Our question-wise ablation study shows that the performance gain comes from the rich annotation of prediction and planning QA pairs in the graph structure. To facilitate future research, all code, data, models and an official evaluation server are available to the public.

</details>

---

## 46. AWOL: Analysis WithOut synthesis using Language

- [ ] AWOL: Analysis WithOut synthesis using Language | https://eccv.ecva.net/virtual/2024/poster/1324

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1324

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

There exist many classical parametric 3D shape models but creating novel shapes with such models requires expert knowledge of their parameters. For example, imagine creating a specific type of tree using procedural graphics or a new kind of animal from a statistical shape model. Our key idea is to leverage language to control such existing models to produce novel shapes. This involves learning a mapping between the latent space of a vision-language model and the parameter space of the 3D model, which we do using a small set of shape and text pairs.  Our hypothesis is this mapping from language to parameters allows us to generate parameters for objects that were never seen during training. If the mapping between language and parameters is sufficiently smooth, then interpolation or generalization in language should translate appropriately into novel 3D shapes. We test our approach with two very different types of parametric shape models (quadrupeds and arboreal trees). We use a learned statistical shape model of quadrupeds and show that we can use text to generate new animals not present during training. In particular, we demonstrate state-of-the-art shape estimation of 3D dogs. This work also constitutes the first language-driven method for generating 3D trees. Finally, embedding images in the CLIP latent space enables us to generate animals and trees directly from images.

</details>

---

## 47. SemGrasp: Semantic Grasp Generation via Language Aligned Discretization

- [ ] SemGrasp: Semantic Grasp Generation via Language Aligned Discretization | https://eccv.ecva.net/virtual/2024/poster/1361

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1361

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generating natural human grasps necessitates consideration of not just object geometry but also semantic information. Solely depending on object shape for grasp generation confines the applications of prior methods in downstream tasks. This paper presents a novel semantic-based grasp generation method, termed SemGrasp, which generates a static human grasp pose by incorporating semantic information into the grasp representation. We introduce a discrete representation that aligns the grasp space with semantic space, enabling the generation of grasp postures in accordance with language instructions. A Multimodal Large Language Model (MLLM) is subsequently fine-tuned, integrating object, grasp, and language within a unified semantic space. To facilitate the training of SemGrasp, we compile a large-scale, grasp-text-aligned dataset named CapGrasp, featuring over 300k detailed captions and 50k diverse grasps. Experimental findings demonstrate that SemGrasp efficiently generates natural human grasps in alignment with linguistic intentions. Our code, models, and dataset are available publicly at: https://kailinli.github.io/SemGrasp.

</details>

---

## 48. FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models

- [ ] FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models | https://eccv.ecva.net/virtual/2024/poster/1395

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1395

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Human motion synthesis is a fundamental task in computer animation. Despite recent progress in this field utilizing deep learning and motion capture data, existing methods are always limited to specific motion categories, environments, and styles. This poor generalizability can be partially attributed to the difficulty and expense of collecting large-scale and high-quality motion data. At the same time, foundation models trained with internet-scale image and text data have demonstrated surprising world knowledge and reasoning ability for various downstream tasks. Utilizing these foundation models may help with human motion synthesis, which some recent works have superficially explored. However, these methods didn't fully unveil the foundation models' potential for this task and only support several simple actions and environments. In this paper, we for the first time, without any motion data, explore open-set human motion synthesis using natural language instructions as user control signals based on MLLMs across any motion task and environment. Our framework can be split into two stages: 1) sequential keyframe generation by utilizing MLLMs as a keyframe designer and animator; 2) motion filling between keyframes through interpolation and motion tracking. Our method can achieve general human motion synthesis for many downstream tasks. The promising results demonstrate the worth of mocap-free human motion synthesis aided by MLLMs and pave the way for future research.

</details>

---

## 49. Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models

- [ ] Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1435

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1435

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper addresses the limitations of existing adverse weather image restoration methods trained on synthetic data when applied to real-world scenarios. We formulate a semi-supervised learning framework utilizing vision-language models to enhance restoration performance across diverse adverse weather conditions in real-world settings. Our approach involves assessing image clarity and providing semantics using vision-language models on real data, serving as supervision signals for training restoration models. For clearness enhancement, we use real-world data, employing a dual-step strategy with pseudo-labels generated by vision-language models and weather prompt learning. For semantic enhancement, we integrate real-world data by adjusting weather conditions in vision-language model descriptions while preserving semantic meaning. Additionally, we introduce an efficient training strategy to alleviate computational burdens. Our approach achieves superior results in real-world adverse weather image restoration, demonstrated through qualitative and quantitative comparisons with state-of-the-art approaches.

</details>

---

## 50. Bottom-Up Domain Prompt Tuning for Generalized Face Anti-Spoofing

- [ ] Bottom-Up Domain Prompt Tuning for Generalized Face Anti-Spoofing | https://eccv.ecva.net/virtual/2024/poster/1451

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1451

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Face anti-spoofing (FAS) which plays an important role in securing face recognition systems has been attracting increasing attention. Recently, the popular vision-language model CLIP has been proven to be effective for FAS, where outstanding performance can be achieved by simply transferring the class label into textual prompt. In this work,  we aim to improve the generalization ability of CLIP-based FAS from a prompt learning perspective. Specifically, a Bottom-Up Domain Prompt Tuning method (BUDoPT) that covers the different levels of domain variance, including the domain of recording settings and domain of attack types is proposed. To handle domain discrepancies of recording settings, we design a context-aware adversarial domain-generalized prompt learning strategy that can learn domain-invariant prompt. For the spoofing domain with different attack types, we construct a fine-grained textual  prompt that guides CLIP to look through the subtle details of different attack instruments. Extensive experiments are conducted on five FAS datasets with a large number of variations (camera types, resolutions, image qualities, lighting conditions, and recording environments). The effectiveness of our proposed method is evaluated with different amounts of source domains from multiple angles, where we boost the generalizability compared with the state of the arts with multiple training datasets or with only one dataset.

</details>

---

## 51. Open Vocabulary Multi-Label Video Classification

- [ ] Open Vocabulary Multi-Label Video Classification | https://eccv.ecva.net/virtual/2024/poster/1471

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1471

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have enabled significant progress in open vocabulary computer vision tasks such as image classification, object detection and image segmentation. Some recent works have focused on extending VLMs to open vocabulary single label action classification in videos. However, previous methods fall short in holistic video understanding which requires the ability to simultaneously recognize multiple actions and entities e.g., objects in the video in an open vocabulary setting. We formulate this problem as open vocabulary multi-label video classification and propose a method to adapt a pre-trained VLM such as CLIP to solve this task. We leverage large language models (LLMs) to provide semantic guidance to the VLM about class labels to improve its open vocabulary performance with two key contributions. First, we propose an end-to-end trainable architecture that learns to prompt an LLM to generate soft attributes for the CLIP text-encoder to enable it to recognize novel classes. Second, we integrate a temporal modeling module into CLIP's vision encoder to effectively model the spatio-temporal dynamics of video concepts as well as propose a novel regularized finetuning technique to ensure strong open vocabulary classification performance in the video domain. Our extensive experimentation showcases the efficacy of our approach on multiple benchmark datasets.

</details>

---

## 52. Leveraging temporal contextualization for video action recognition

- [ ] Leveraging temporal contextualization for video action recognition | https://eccv.ecva.net/virtual/2024/poster/1473

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1473

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pretrained vision-language models (VLM) have shown effectiveness in video understanding. However, recent studies have not sufficiently leveraged essential temporal information from videos, simply averaging frame-wise representations or referencing consecutive frames. We introduce Temporally Contextualized CLIP (TC-CLIP), a pioneering framework for video understanding that effectively and efficiently leverages comprehensive video information. We propose Temporal Contextualization (TC), a novel layer-wise temporal information infusion mechanism for video that extracts core information from each frame, interconnects relevant information across the video to summarize into context tokens, and ultimately leverages the context tokens during the feature encoding process. Furthermore, Our Video-conditional Prompting (VP) module manufactures context tokens to generate informative prompts in text modality. We conduct extensive experiments in zero-shot, few-shot, base-to-novel, and fully-supervised settings to validate the superiority of our TC-CLIP. Ablation studies for TC and VP guarantee our design choices. Our code will be publicly available.

</details>

---

## 53. VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding

- [ ] VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding | https://eccv.ecva.net/virtual/2024/poster/1474

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1474

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We explore how reconciling several foundation models (large language models and vision-language models) with a novel unified memory mechanism could tackle the challenging video understanding problem, especially capturing the long-term temporal relations in lengthy videos. In particular, the proposed multimodal agent VideoAgent: 1) constructs a structured memory to store both the generic temporal event descriptions and object-centric tracking states of the video; 2) given an input task query, it employs tools including video segment localization and object memory querying along with other visual foundation models to interactively solve the task, utilizing the zero-shot tool-use ability of LLMs. \method demonstrates impressive performances on several long-horizon video understanding benchmarks, on average increasing 6.6% on NExT-QA and 26.0% on EgoSchema over baselines. The code will be released to the public.

</details>

---

## 54. Octopus: Embodied Vision-Language Programmer from Environmental Feedback

- [ ] Octopus: Embodied Vision-Language Programmer from Environmental Feedback | https://eccv.ecva.net/virtual/2024/poster/1486

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1486

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) have achieved substantial progress in multimodal perception and reasoning. When integrated into an embodied agent, existing embodied VLM works either output detailed action sequences at the manipulation level or only provide plans at an abstract level, leaving a gap between high-level planning and real-world manipulation. To bridge this gap, we introduce Octopus, an embodied vision-language programmer that uses executable code generation as a medium to connect planning and manipulation. Octopus is designed to 1) proficiently comprehend an agent's visual and textual task objectives, 2) formulate intricate action sequences, and 3) generate executable code.  To facilitate Octopus model development, we introduce OctoVerse: a suite of environments tailored for benchmarking vision-based code generators on a wide spectrum of tasks, ranging from mundane daily chores in simulators to sophisticated interactions in complex video games such as Grand Theft Auto (GTA) and Minecraft. To train Octopus, we leverage GPT-4 to control an explorative agent that generates training data, i.e., action blueprints and corresponding executable code. We also collect feedback that enables an enhanced training scheme called Reinforcement Learning with Environmental Feedback (RLEF). Through a series of experiments, we demonstrate Octopus's functionality and present compelling results, showing that the proposed RLEF refines the agent's decision-making. By open-sourcing our simulation environments, dataset, and model architecture, we aspire to ignite further innovation and foster collaborative applications within the broader embodied AI community.

</details>

---

## 55. ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities

- [ ] ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities | https://eccv.ecva.net/virtual/2024/poster/1489

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1489

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although great progress has been made in 3D visual grounding, current models still rely on explicit textual descriptions for grounding and lack the ability to reason human intentions from implicit instructions. We propose a new task called  3D reasoning grounding and introduce a new benchmark ScanReason which provides over 10K question-answer-location pairs from five reasoning types that require the synerization of reasoning and grounding. This benchmark challenges models to conduct joint reasoning on questions and the 3D environment before predicting the 3D locations of target objects. We further design our approach, ReGround3D, composed of the visual-centric reasoning module empowered by Multi-modal Large Language Model (MLLM) and the 3D grounding module to obtain accurate object locations by looking back to the enhanced geometry and fine-grained details from the 3D scenes. A chain-of-grounding mechanism is proposed to further boost the performance with interleaved reasoning and grounding steps during inference. Extensive experiments on the proposed benchmark validate the effectiveness of our proposed approach. Our code will be released to the community.

</details>

---

## 56. Embodied Understanding of Driving Scenarios

- [ ] Embodied Understanding of Driving Scenarios | https://eccv.ecva.net/virtual/2024/poster/1485

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1485

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Embodied scene understanding serves as the cornerstone for autonomous agents to perceive, interpret, and respond to open driving scenarios. Such understanding is typically founded upon Vision-Language Models (VLMs). Nevertheless, existing VLMs are restricted to the 2D domain, devoid of spatial awareness and long-horizon extrapolation proficiencies. We revisit the key aspects of autonomous driving and formulate appropriate rubrics. Hereby, we introduce the Embodied Language Model (ELM), a comprehensive framework tailored for agents' understanding of driving scenes with large spatial and temporal spans. ELM incorporates space-aware pre-training to endow the agent with robust spatial localization capabilities. Besides, the model employs time-aware token selection to accurately inquire about temporal cues. We instantiate ELM on the reformulated multi-faced benchmark, and it surpasses previous state-of-the-art approaches in all aspects. All code, data, and models are accessible.

</details>

---

## 57. Uni3DL: A Unified Model for 3D Vision-Language Understanding

- [ ] Uni3DL: A Unified Model for 3D Vision-Language Understanding | https://eccv.ecva.net/virtual/2024/poster/1490

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1490

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we present Uni3DL, a unified model for 3D Vision-Language understanding. Distinct from existing unified vision-language models in 3D which are limited in task variety and predominantly dependent on projected multi-view images, Uni3DL operates directly on point clouds. This approach significantly expands the range of supported tasks in 3D, encompassing both vision and vision-language tasks in 3D. At the core of Uni3DL, a query transformer is designed to learn task-agnostic semantic and mask outputs by attending to 3D visual features, and a task router is employed to selectively generate task-specific outputs required for diverse tasks. With a unified architecture, our Uni3DL model enjoys seamless task decomposition and substantial parameter sharing across tasks. Uni3DL has been rigorously evaluated across diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, visual grounding, 3D captioning, and text-3D cross-modal retrieval. It demonstrates performance on par with or surpassing state-of-the-art (SOTA) task-specific models. We hope our benchmark and Uni3DL model will serve as a solid step to ease future research in unified models in the realm of 3D and language understanding.

</details>

---

## 58. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation

- [ ] Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation | https://eccv.ecva.net/virtual/2024/poster/1496

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1496

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed due to the introduction of image features. To construct safe MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of the pre-aligned LLMs in MLLMs. Experiments with five state-of-the-art (SOTA) MLLMs demonstrate that ECSO significantly enhances model safety (e.g., a 37.6% improvement on MM-SafetyBench (SD+OCR), and 71.3% on VLSafe for LLaVA-1.5-7B),  while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we demonstrate that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for the alignment of  MLLMs without extra human intervention.

</details>

---

## 59. Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Models

- [ ] Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1497

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1497

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) rely on vision encoders and Large Language Models (LLMs) to exhibit remarkable capabilities on various multi-modal tasks in the joint space of vision and language. However, typographic attacks, which disrupt Vision-Language Models (VLMs) such as Contrastive Language-Image Pretraining (CLIP), have also been expected to be a security threat to LVLMs. Firstly, we verify typographic attacks on current well-known commercial and open-source LVLMs and uncover the widespread existence of this threat. Secondly, to better assess this vulnerability, we propose the most comprehensive and largest-scale Typographic Dataset to date. The Typographic Dataset not only considers the evaluation of typographic attacks under various multi-modal tasks but also evaluates the effects of typographic attacks, influenced by texts generated with diverse factors. Based on the evaluation results, we investigate the causes why typographic attacks impacting VLMs and LVLMs, leading to three highly insightful discoveries. During the process of further validating the rationality of our discoveries, we can reduce the performance degradation caused by typographic attacks from 42.07\% to 13.90\%.

</details>

---

## 60. The All-Seeing Project V2: Towards General Relation Comprehension of the Open World

- [ ] The All-Seeing Project V2: Towards General Relation Comprehension of the Open World | https://eccv.ecva.net/virtual/2024/poster/1493

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1493

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present the All-Seeing Project V2: a new model and dataset designed for understanding object relations in images. Specifically, we propose the All-Seeing Model V2 (ASMv2) that integrates the formulation of text generation, object localization, and relation comprehension into a relation conversation (ReC) task. Leveraging this unified task, our model excels not only in perceiving and recognizing all objects within the image but also in grasping the intricate relation graph between them, diminishing the relation hallucination often encountered by Multi-modal Large Language Models (MLLMs). To facilitate training and evaluation of MLLMs in relation understanding, we created the first high-quality ReC dataset ({AS-V2) which is aligned with the format of standard instruction tuning data. In addition, we design a new benchmark, termed Circular-based Relation Probing Evaluation (CRPE) for comprehensively evaluating the relation comprehension capabilities of MLLMs. Notably, our ASMv2 achieves an overall accuracy of 52.04 on this relation-aware benchmark, surpassing the 43.14 of LLaVA-1.5 by a large margin. We hope that our work can inspire more future research and contribute to the evolution towards artificial general intelligence. Data, model, and code shall be released.

</details>

---

## 61. MoAI: Mixture of All Intelligence for Large Language and Vision Models

- [ ] MoAI: Mixture of All Intelligence for Large Language and Vision Models | https://eccv.ecva.net/virtual/2024/poster/1498

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1498

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rise of large language models (LLMs) and instruction tuning has led to the current trend of instruction-tuned large language and vision models (LLVMs). This trend involves either meticulously curating numerous instruction tuning datasets tailored to specific objectives or enlarging LLVMs to manage vast amounts of vision language (VL) data. However, current LLVMs have disregarded the detailed and comprehensive real-world scene understanding available from specialized computer vision (CV) models in visual perception tasks such as segmentation, detection, scene graph generation (SGG), and optical character recognition (OCR). Instead, the existing LLVMs rely mainly on the large capacity and emergent capabilities of their LLM backbones. Therefore, we present a new LLVM, Mixture of All Intelligence (MoAI), which leverages auxiliary visual information obtained from the outputs of external segmentation, detection, SGG, and OCR models. MoAI operates through two newly introduced modules: MoAI-Compressor and MoAI-Mixer. After verbalizing the outputs of the external CV models, the MoAI-Compressor aligns and condenses them to efficiently use relevant auxiliary visual information for VL tasks. MoAI-Mixer then blends three types of intelligence—(1) visual features, (2) auxiliary features from the external CV models, and (3) language features—utilizing the concept of Mixture of Experts. Through this integration, MoAI significantly outperforms both open-source and closed-source LLVMs in numerous zero-shot VL tasks, particularly those related to real-world scene understanding such as object existence, positions, relations, and OCR without enlarging the model size or curating extra visual instruction tuning datasets.

</details>

---

## 62. ViGoR: Improving Visual Grounding of Large Vision Language Models with Fine-Grained Reward Modeling

- [ ] ViGoR: Improving Visual Grounding of Large Vision Language Models with Fine-Grained Reward Modeling | https://eccv.ecva.net/virtual/2024/poster/1495

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1495

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

By combining natural language understanding, generation capabilities, and breadth of knowledge of large language models with image perception, recent large vision language models (LVLMs) have shown unprecedented visual reasoning capabilities. However, the generated text often suffers from inaccurate grounding in the visual input, resulting in errors such as hallucination of nonexistent scene elements, missing significant parts of the scene, and inferring incorrect attributes of and relationships between objects. To address these issues, we introduce a novel framework, ViGoR ([Vi]sual [G]r[o]unding Through Fine-Grained [R]eward Modeling) that utilizes fine-grained reward modeling to significantly enhance the visual grounding of LVLMs over pre-trained baselines. This improvement is efficiently achieved using much cheaper human evaluations instead of full supervisions, as well as automated methods. We show the effectiveness of our approach through a variety of evaluation methods and benchmarks. Additionally, we released our human annotation (https://github.com/amazon-science/vigor) comprising 15,440 images and generated text pairs with fine-grained evaluations to contribute to related research in the community.

</details>

---

## 63. Training A Small Emotional Vision Language Model for Visual Art Comprehension

- [ ] Training A Small Emotional Vision Language Model for Visual Art Comprehension | https://eccv.ecva.net/virtual/2024/poster/1499

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1499

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper develops small vision language models to understand visual art, which, given an art work, aims to identify its emotion category and explain this prediction with natural language. While small models are computationally efficient, their capacity is much limited compared with large models. To break this trade-off, this paper builds a small emotional vision language model (SEVLM) by emotion modeling and input-output feature alignment. On the one hand, based on valence-arousal-dominance (VAD) knowledge annotated by psychology experts, we introduce and fuse emotional features derived through VAD dictionary and a VAD head to align VAD vectors of predicted emotion explanation and the ground truth. This allows the vision language model to better understand and generate emotional texts, compared with using traditional text embeddings alone. On the other hand, we design a contrastive head to pull close embeddings of the image, its emotion class, and explanation, which aligns model outputs and inputs. On two public affective explanation datasets, we show that the proposed techniques consistently improve the visual art understanding performance of baseline SEVLMs. Importantly, the proposed model can be trained and evaluated on a single RTX 2080 Ti while exhibiting very strong performance: it not only outperforms the state-of-the-art small models but is also competitive compared with LLaVA 7B after fine-tuning and GPT4(V). Code will be made publicly available.

</details>

---

## 64. Quantized Prompt for Efficient Generalization of Vision-Language Models

- [ ] Quantized Prompt for Efficient Generalization of Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1500

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1500

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the past few years, large-scale pre-trained vision-language models like CLIP have achieved tremendous success in various fields. Naturally, how to transfer the rich knowledge in such huge pre-trained models to downstream tasks and datasets becomes a hot topic. During downstream adaptation, the most challenging problems are overfitting and catastrophic forgetting, which can cause the model to overly focus on the current data and lose more crucial domain-general knowledge. Existing works use classic regularization techniques to solve the problems. As solutions become increasingly complex, the ever-growing storage and inference costs are also a significant problem that urgently needs to be addressed. While in this paper, we start from an observation that proper random noise can suppress overfitting and catastrophic forgetting. Then we regard quantization error as a kind of noise, and explore quantization for regularizing vision-language model, which is quite efficiency and effective. Furthermore, to improve the model's generalization capability while maintaining its specialization capacity at minimal cost, we deeply analyze the characteristics of the weight distribution in prompts, conclude several principles for quantization module design and follow such principles to create several competitive baselines. The proposed method is significantly efficient due to its inherent lightweight nature, making it possible to adapt on extremely resource-limited devices. Our method can be fruitfully integrated into many existing approaches like MaPLe, enhancing accuracy while reducing storage overhead, making it more powerful yet versatile. Extensive experiments on 11 datasets shows great superiority of our method sufficiently.

</details>

---

## 65. Getting it Right: Improving Spatial Consistency in Text-to-Image Models

- [ ] Getting it Right: Improving Spatial Consistency in Text-to-Image Models | https://eccv.ecva.net/virtual/2024/poster/1502

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1502

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

One of the key shortcomings in current text-to-image (T2I) models is their inability to consistently generate images which faithfully follow the spatial relationships specified in the text prompt. In this paper, we offer a comprehensive investigation of this limitation, while also developing datasets and methods that achieve state-of-the-art performance. First, we find that current vision-language datasets do not represent spatial relationships well enough; to alleviate this bottleneck, we create SPRIGHT, the first spatially-focused, large scale dataset, by re-captioning 6 million images from 4 widely used vision datasets. Through a 3-fold evaluation and analysis pipeline, we find that SPRIGHT largely improves upon existing datasets in capturing spatial relationships. To demonstrate its efficacy, we leverage only 0.25% of SPRIGHT and achieve a 22% improvement in generating spatially accurate images while improving the FID and CMMD scores. Secondly, we find that training on images containing a large number of objects results in substantial improvements in spatial consistency. Notably, we attain state-of-the-art on T2I-CompBench with a spatial score of 0.2133, by fine-tuning on <500 images. Finally, through a set of controlled experiments and ablations, we document multiple findings that we believe will enhance the understanding of factors that affect spatial consistency in text-to-image models. We will publicly release all our code, data, and models.

</details>

---

## 66. VeCLIP: Improving CLIP Training via Visual-enriched Captions

- [ ] VeCLIP: Improving CLIP Training via Visual-enriched Captions | https://eccv.ecva.net/virtual/2024/poster/1505

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1505

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale web-crawled datasets are fundamental for the success of pre-training vision-language models, such as CLIP. However, the inherent noise and potential irrelevance of web-crawled AltTexts pose challenges in achieving precise image-text alignment. Existing methods utilizing large language models (LLMs) for caption rewriting have shown promise on small, curated datasets like CC3M and CC12M. This study introduces a scalable pipeline for noisy caption rewriting. Unlike recent LLM rewriting techniques, we emphasize the incorporation of visual concepts into captions, termed as Visual-enriched Captions (VeCap). To ensure data diversity, we propose a novel mixed training scheme that optimizes the utilization of AltTexts alongside newly generated VeCap. We showcase the adaptation of this method for training CLIP on large-scale web-crawled datasets, termed VeCLIP. Employing this cost-effective pipeline, we effortlessly scale our dataset up to 300 million samples named VeCap dataset. Our results show significant advantages in image-text alignment and overall model performance. For example, VeCLIP achieves up to +25.2% gain in COCO and Flickr30k retrieval tasks under the 12M setting. For data efficiency, VeCLIP achieves +3% gain while only using 14% of the data employed in the vanilla CLIP and 11% in ALIGN. We also note the VeCap data is complementary with other well curated datasets good for zero-shot classification tasks. When combining VeCap and DFN, our model can achieve strong performance on both of image-text retrieval and zero-shot classification tasks, e.g. 83.1% accuracy@1 on ImageNet zero-shot for a H/14 model.

</details>

---

## 67. LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models

- [ ] LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1510

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1510

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection is crucial for model reliability, as it identifies samples from unknown classes and reduces errors due to unexpected inputs. Vision-Language Models (VLMs) such as CLIP are emerging as powerful tools for OOD detection by integrating multi-modal information. However, the practical application of such systems is challenged by manual prompt engineering, which demands domain expertise and is sensitive to linguistic nuances. In this paper, we introduce \textbf{L}abel-dirven \textbf{A}utomated \textbf{P}rompt \textbf{T}uning (LAPT), a novel approach to OOD detection that reduces the need for manual prompt engineering. We develop distribution-aware prompts with in-distribution (ID) class names and negative labels mined automatically. Training samples linked to these class labels are collected autonomously via image synthesis and retrieval methods, allowing for prompt learning without manual effort. We utilize a simple cross-entropy loss for prompt optimization, with cross-modal and cross-distribution mixing strategies to reduce image noise and explore the intermediate space between distributions, respectively. The LAPT framework operates autonomously, requiring only ID class names as input and eliminating the need for manual intervention. With extensive experiments, LAPT consistently outperforms manually crafted prompts, setting a new standard for OOD detection. Moreover, LAPT not only enhances the distinction between ID and OOD samples, but also improves ID classification accuracy and strengthens generalization robustness to covariate shifts, resulting in outstanding performance in challenging full-spectrum OOD detection tasks. Codes will be released.

</details>

---

## 68. Adapt without Forgetting: Distill Proximity from Dual Teachers in Vision-Language Models

- [ ] Adapt without Forgetting: Distill Proximity from Dual Teachers in Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1507

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1507

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal models such as CLIP possess remarkable zero-shot transfer capabilities, making them highly effective in continual learning tasks. However, this advantage is severely compromised by catastrophic forgetting, which undermines the valuable zero-shot learning abilities of these models. Existing methods predominantly focus on preserving zero-shot capabilities but often fall short in fully exploiting the rich modal information inherent in multi-modal models. In this paper, we propose a strategy to enhance both the zero-shot transfer ability and adaptability to new data distribution. We introduce a novel graph-based multi-modal proximity distillation approach that preserves the intra- and inter-modal information for visual and textual modalities. This approach is further enhanced with a sample re-weighting mechanism, dynamically adjusting the influence of teachers for each individual sample. Experimental results demonstrate a considerable improvement over existing methodologies, which illustrate the effectiveness of the proposed method in the field of continual learning.

</details>

---

## 69. Textual Query-Driven Mask Transformer for Domain Generalized Segmentation

- [ ] Textual Query-Driven Mask Transformer for Domain Generalized Segmentation | https://eccv.ecva.net/virtual/2024/poster/1521

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1521

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce a method to tackle Domain Generalized Semantic Segmentation (DGSS) by utilizing domain-invariant semantic knowledge from text embeddings of vision-language models. We employ the text embeddings as object queries within a transformer-based segmentation framework (textual object queries). These queries are regarded as a domain-invariant basis for pixel grouping in DGSS. To leverage the power of textual object queries, we introduce a novel framework named the textual query-driven mask transformer (tqdm). Our tqdm aims to (1) generate textual object queries that maximally encode domain-invariant semantics and (2) enhance the semantic clarity of dense visual features. Additionally, we suggest three regularization losses to improve the efficacy of tqdm by aligning between visual and textual features. By utilizing our method, the model can comprehend inherent semantic information for classes of interest, enabling it to generalize to extreme domains (e.g., sketch style). Our tqdm achieves 68.9 mIoU on GTA5→Cityscapes, outperforming the prior state-of-the-art method by 2.5 mIoU. Source code will be released.

</details>

---

## 70. SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference

- [ ] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference | https://eccv.ecva.net/virtual/2024/poster/1518

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1518

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in contrastive language-image pretraining (CLIP) have demonstrated strong capabilities in zero-shot classification by aligning visual and textual features at an image level. However, in dense prediction tasks, CLIP often struggles to localize visual features within an image and fails to attain favorable pixel-level segmentation results. In this work, we investigate in CLIP's spatial reasoning mechanism and identify that its failure of dense prediction is caused by a location misalignment issue in the self-attention process. Based on this observation, we propose a training-free adaptation approach for CLIP's semantic segmentation, which only introduces a very simple modification to CLIP but can effectively address the issue of location misalignment. Specifically, we reform the self-attention mechanism with leveraging query-to-query and key-to-key similarity to determine attention scores. Remarkably, this minimal modification to CLIP significantly enhances its capability in dense prediction, improving the original CLIP's 14.1% average zero-shot mIoU over eight semantic segmentation benchmarks to 38.2%, and outperforming the existing SoTA's 33.9% by a large margin.

</details>

---

## 71. DECIDER: Leveraging Foundation Model Priors for Improved Model Failure Detection and Explanation

- [ ] DECIDER: Leveraging Foundation Model Priors for Improved Model Failure Detection and Explanation | https://eccv.ecva.net/virtual/2024/poster/1549

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1549

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we focus on the problem of detecting samples that can lead to model failure under the classification setting. Failures can stem from various sources, such as spurious correlations between image features and labels, class imbalances in the training data, and covariate shifts between training and test distributions. Existing approaches often rely on classifier prediction scores and do not comprehensively identify all failure scenarios. Instead, we pose failure detection as the problem of identifying the discrepancies between the classifier and its enhanced version. We build such an enhanced model by infusing task-agnostic prior knowledge from a vision-language model (e.g., CLIP) that encodes general-purpose visual and semantic relationships. Unlike conventional training, our enhanced model, named the Prior Induced Model (PIM) learns to map the pre-trained model features to the VLM latent space and aligns the same with a set of pre-specified, fine-grained class-level attributes which are later aggregated to estimate the class prediction. We propose that such a training strategy allows the model to concentrate only on the task specific attributes while making predictions in lieu of the pre-trained model and also enables human-interpretable explanations for failure. We conduct extensive empirical studies on various benchmark datasets and baselines, observing substantial improvements in failure detection.

</details>

---

## 72. Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models

- [ ] Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1560

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1560

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models (VLMs) have shown a strong zero-shot generalization capability on unseen-domain data. However, when adapting pre-trained VLMs to a sequence of downstream tasks, they are prone to forgetting previously learned knowledge and degrade their zero-shot classification capability. To tackle this problem, we propose a unique Selective Dual-Teacher Knowledge Transfer framework that leverages the most recent fine-tuned and the original pre-trained VLMs as dual teachers to preserve the previously learned knowledge and zero-shot capabilities, respectively. With only access to an unlabeled reference dataset, our proposed framework performs a selective knowledge distillation mechanism by measuring the feature discrepancy from the dual teacher VLMs. Consequently, our selective dual-teacher knowledge distillation would mitigate catastrophic forgetting of previously learned knowledge while preserving the zero-shot capabilities from pre-trained VLMs. Through extensive experiments on benchmark datasets, we show that our proposed framework is favorable against state-of-the-art continual learning approaches for preventing catastrophic forgetting and zero-shot degradation.

</details>

---

## 73. SAFT: Towards Out-of-Distribution Generalization in Fine-Tuning

- [ ] SAFT: Towards Out-of-Distribution Generalization in Fine-Tuning | https://eccv.ecva.net/virtual/2024/poster/1561

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1561

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Handling distribution shifts from training data, known as out-of-distribution (OOD) generalization, poses a significant challenge in the field of machine learning. While a pre-trained vision-language model like CLIP has demonstrated remarkable zero-shot performance, further adaptation of the model to downstream tasks leads to undesirable degradation for OOD data. In this work, we introduce Sparse Adaptation for Fine-Tuning (SAFT), a method that prevents fine-tuning from forgetting the general knowledge in the pre-trained model. SAFT only updates a small subset of important parameters whose gradient magnitude is large, while keeping the other parameters frozen. SAFT is straightforward to implement and conceptually simple. Extensive experiments show that with only 0.1% of the model parameters, SAFT can significantly improve the performance of CLIP. It consistently outperforms baseline methods across several benchmarks. On the few-shot learning benchmark of ImageNet and its variants, SAFT gives a gain of 5.15% on average over the conventional fine-tuning method in OOD settings.

</details>

---

## 74. HVCLIP: High-dimensional Vector in CLIP for Unsupervised Domain Adaptation

- [ ] HVCLIP: High-dimensional Vector in CLIP for Unsupervised Domain Adaptation | https://eccv.ecva.net/virtual/2024/poster/1571

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1571

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancement in the large-scale pre-training model (such as CLIP) has significantly improved unsupervised domain adaptation (UDA) by leveraging the pre-trained knowledge to bridge the source and target domain gap. Catastrophic forgetting is the main challenge of CLIP in UDA where the traditional fine-tuning to adjust CLIP on a target domain can quickly override CLIP's pre-trained knowledge. To address the above issue, we propose to convert CLIP's features into high-dimensional vector (hypervector) space to utilize the robustness property of hypervector to mitigate catastrophic forgetting. We first study the feature dimension size in the hypervector space to empirically find the dimension threshold that allows enough feature patterns to be redundant to avoid excessive training (thus mitigating catastrophic forgetting). To further utilize the robustness of hypervector, we propose Discrepancy Reduction to reduce the domain shift between source and target domains, and Feature Augmentation to synthesize labeled target domain features from source domain features. We achieved the best results on four public UDA datasets, and we show the generalization of our method to other applications (few-shot learning, continual learning) and the model-agnostic property of our method across vision-language and vision backbones.

</details>

---

## 75. Learning Modality-agnostic Representation for Semantic Segmentation from Any Modalities

- [ ] Learning Modality-agnostic Representation for Semantic Segmentation from Any Modalities | https://eccv.ecva.net/virtual/2024/poster/1593

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1593

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image modality is not perfect as it often fails in certain conditions, e.g., night and fast motion. This significantly limits the robustness and versatility of existing multi-modal (i.e., Image+X) semantic segmentation methods when confronting modality absence or failure, as often occurred in real-world applications. Inspired by the open-world learning capability of multi-modal vision-language models (MVLMs), we explore a new direction in learning the modality-agnostic representation via knowledge distillation (KD) from MVLMs. Intuitively, we propose Any2Seg, a novel framework that can achieve robust segmentation from any combination of modalities in any visual conditions. Specifically, we first introduce a novel language-guided semantic correlation distillation (LSCD) module to transfer both inter-modal and intra-modal semantic knowledge in the embedding space from MVLMs, e.g., LanguageBind. This enables us to minimize the modality gap and alleviate semantic ambiguity to combine any modalities in any visual conditions. Then, we introduce a modality-agnostic feature fusion (MFF) module that reweights the multi-modal features based on the inter-modal correlation and selects the fine-grained feature. This way, our Any2Seg finally yields an optimal modality-agnostic representation. Extensive experiments on two benchmarks with four modalities demonstrate that Any2Seg achieves the state-of-the-art under the multi-modal setting (+3.54 mIoU) and excels in the challenging modality-incomplete setting (+19.79 mIoU).

</details>

---

## 76. Diffusion Models for Open-Vocabulary Segmentation

- [ ] Diffusion Models for Open-Vocabulary Segmentation | https://eccv.ecva.net/virtual/2024/poster/1595

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1595

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary segmentation is the task of segmenting anything that can be named in an image. Recently, large-scale vision-language modelling has led to significant advances in open-vocabulary segmentation, but at the cost of gargantuan and increasing training and annotation efforts. Hence, we ask if it is possible to use existing foundation models to synthesise on-demand efficient segmentation algorithms for specific class sets, making them applicable in an open-vocabulary setting without the need to collect further data, annotations or perform training. To that end, we present OVDiff, a novel method that leverages generative text-to-image diffusion models for unsupervised open-vocabulary segmentation. OVDiff synthesises support image sets for arbitrary textual categories, creating for each a set of prototypes representative of both the category and its surrounding context (background). It relies solely on pre-trained components and outputs the synthesised segmenter directly, without training. Our approach shows strong performance on a range of benchmarks, obtaining a lead of more than 5% over prior work on PASCAL VOC.

</details>

---

## 77. Collaborative Vision-Text Representation Optimizing for Open-Vocabulary Segmentation

- [ ] Collaborative Vision-Text Representation Optimizing for Open-Vocabulary Segmentation | https://eccv.ecva.net/virtual/2024/poster/1597

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1597

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models, e.g., CLIP, have been increasingly used to address the challenging Open-Vocabulary Segmentation (OVS) task, benefiting from their well-aligned vision-text embedding space. Typical solutions involve either freezing CLIP during training to unilaterally maintain its zero-shot capability, or fine-tuning CLIP vision encoder to achieve perceptual sensitivity to local regions.  However, few of them incorporate vision-text collaborative optimization. Based on this, we propose the Content-Dependent Transfer to adaptively enhance each text embedding by interacting with the input image, which presents a parameter-efficient way to optimize the text representation. Besides, we additionally introduce a Representation Compensation strategy, reviewing the original CLIP-V representation as compensation to maintain the zero-shot capability of CLIP. In this way, the vision and text representation of CLIP are optimized collaboratively, enhancing the alignment of the vision-text feature space. To the best of our knowledge, we are the first to establish the collaborative vision-text optimizing mechanism within the OVS field. Extensive experiments demonstrate our method achieves superior performance on popular OVS benchmarks. In open-vocabulary semantic segmentation, our method outperforms the previous state-of-the-art approaches by +0.5, +2.3, +3.4, +0.4 and +1.1 mIoU, respectively on A-847, A-150, PC-459, PC-59 and PAS-20. Furthermore, in a panoptic setting on the ADE20K dataset, we achieve the performance of 27.1 PQ, 73.5 SQ, and 32.9 RQ.

</details>

---

## 78. Emergent Visual-Semantic Hierarchies in Image-Text Representations

- [ ] Emergent Visual-Semantic Hierarchies in Image-Text Representations | https://eccv.ecva.net/virtual/2024/poster/1627

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1627

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While recent vision-and-language models (VLMs) like CLIP are a powerful tool for analyzing text and images in a shared semantic space, they do not explicitly model the hierarchical nature of the set of texts which may describe an image. Conversely, existing multimodal hierarchical representation learning methods require costly training from scratch, failing to leverage the knowledge encoded by state-of-the-art multimodal foundation models. In this work, we study the knowledge of existing foundation models, finding that they exhibit emergent understanding of visual-semantic hierarchies despite not being directly trained for this purpose. We propose the Radial Embedding (RE) framework for probing and optimizing hierarchical understanding, and contribute the HierarCaps dataset, a benchmark facilitating the study of hierarchical knowledge in image--text representations, constructed automatically via large language models. Our results show that foundation VLMs exhibit zero-shot hierarchical understanding, surpassing the performance of prior models explicitly designed for this purpose. Furthermore, we show that foundation models may be better aligned to hierarchical reasoning via a text-only fine-tuning phase, while retaining pretraining knowledge. We will release our data, code, and trained models.

</details>

---

## 79. PiTe: Pixel-Temporal Alignment for Large Video-Language Model

- [ ] PiTe: Pixel-Temporal Alignment for Large Video-Language Model | https://eccv.ecva.net/virtual/2024/poster/1629

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1629

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fueled by the Large Language Models (LLMs) wave, Large Visual-Language Models (LVLMs) have emerged as a pivotal advancement, bridging the gap between image and text. However, video making it challenging for LVLMs to perform adequately due to the complexity of the relationship between language and spatial-temporal data structure. Recent Large Video-Language Models (LVidLMs) align feature of static visual data like image into latent space of language feature, by general multi-modal tasks to leverage abilities of LLMs sufficiently. In this paper, we explore fine-grained alignment approach via object trajectory for different modalities across both spatial and temporal dimensions simultaneously. Thus, we propose a novel LVidLM by trajectory-guided Pixel-Temporal Alignment, dubbed PiTe, that exhibits promising applicable model property. To achieve fine-grained video-language alignment, we curate a multi-modal pre-training dataset PiTe-143k, the dataset provision of moving trajectories in pixel level for all individual objects, that appear and mention in the video and caption both, by our automatic annotation pipeline. Meanwhile, PiTe demonstrates astounding capabilities on myriad video-related multi-modal tasks through beat the state-of-the-art methods by a large margin.

</details>

---

## 80. CadVLM: Bridging Language and Vision in the Generation of Parametric CAD Sketches

- [ ] CadVLM: Bridging Language and Vision in the Generation of Parametric CAD Sketches | https://eccv.ecva.net/virtual/2024/poster/1677

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1677

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Parametric Computer-Aided Design (CAD) is central to contemporary mechanical design. We harness the capabilities of pre-trained foundation models, renowned for their successes in natural language processing and computer vision, to develop generative models specifically for CAD. These models are adept at understanding complex geometries and design reasoning, a crucial advancement in CAD technology. In this paper, we propose CadVLM, an end-to-end vision language model for CAD generation. Our approach involves adapting pre-trained foundation models to manipulate engineering sketches effectively, integrating both sketch primitive sequences and sketch images. Extensive experiments demonstrate superior performance on multiple CAD sketch generation tasks such as CAD autocompletion, CAD autoconstraint, and image conditional generation. To our knowledge, this is the first instance of a multimodal Large Language Model (LLM) being successfully applied to parametric CAD generation, representing a pioneering step in the field of computer-aided mechanical design. The code is available at https://anonymous.4open.science/r/CadVLM.

</details>

---

## 81. F-HOI: Toward Fine-grained Semantic-Aligned 3D Human-Object Interactions

- [ ] F-HOI: Toward Fine-grained Semantic-Aligned 3D Human-Object Interactions | https://eccv.ecva.net/virtual/2024/poster/1747

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1747

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing 3D human object interaction (HOI) datasets and models simply align global descriptions with the long HOI sequence, while lacking a detailed understanding of intermediate states and the transitions between states. In this paper, we argue that fine-grained semantic alignment, which utilizes state-level descriptions, offers a promising paradigm for learning semantically rich HOI representations. To achieve this, we introduce Semantic-HOI, a new dataset comprising over 20K paired HOI states with fine-grained descriptions for each HOI state and the body movements that happen between two consecutive states. Leveraging the proposed dataset, we design three state-level HOI tasks to accomplish fine-grained semantic alignment within the HOI sequence. Additionally, we propose a unified model called \ModelName, designed to leverage multimodal instructions and empower the Multi-modal Large Language Model to efficiently handle diverse HOI tasks. F-HOI offers multiple advantages: (1) It employs a unified task formulation that supports the use of versatile multimodal inputs. (2) It maintains consistency in HOI across 2D, 3D, and linguistic spaces. (3) It utilizes fine-grained textual supervision for direct optimization, avoiding intricate modeling of HOI states. Extensive experiments reveal that \ModelName effectively aligns HOI states with fine-grained semantic descriptions, adeptly tackling understanding, reasoning, generation, and reconstruction tasks.

</details>

---

## 82. Efficient 3D-Aware Facial Image Editing via Attribute-Specific Prompt Learning

- [ ] Efficient 3D-Aware Facial Image Editing via Attribute-Specific Prompt Learning | https://eccv.ecva.net/virtual/2024/poster/1774

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1774

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Drawing upon StyleGAN's expressivity and disentangled latent space, existing 2D approaches employ textual prompting to edit facial images with different attributes. In contrast, 3D-aware approaches that generate faces at different target poses require attribute-specific classifiers, learning separate model weights for each attribute, and are not scalable for novel attributes. In this work, we propose an efficient, plug-and-play, 3D-aware face editing framework, based on attribute-specific prompt learning, enabling the generation of facial images with controllable attributes across various target poses. To this end, we introduce a text-driven learnable style token-based latent attribute editor (LAE). The LAE harnesses a pre-trained vision-language model to find text-guided attribute-specific editing direction in the latent space of any pre-trained 3D-aware GAN. It utilizes learnable style tokens and style mappers to learn and transform this editing direction to 3D latent space. To train LAE with multiple attributes, we use directional contrastive loss and style token loss. Furthermore, to ensure view consistency and identity preservation across different poses and attributes, we employ several 3D-aware identity and pose preservation losses. Our experiments show that our proposed framework generates high-quality images with 3D awareness and view consistency while maintaining attribute-specific features. We demonstrate the effectiveness of our method on different facial attributes, including hair color and style, expression, and others. Our Source code and models will be publicly released.

</details>

---

## 83. When Do We Not Need Larger Vision Models?

- [ ] When Do We Not Need Larger Vision Models? | https://eccv.ecva.net/virtual/2024/poster/1814

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1814

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Scaling up the size of vision models has been the de facto standard to obtain more powerful visual representations. In this work, we discuss the point beyond which larger vision models are not necessary. First, we demonstrate the power of Scaling on Scales (S^2), whereby a pre-trained and frozen smaller vision model (e.g., ViT-B or ViT-L), run over multiple image scales, can outperform larger models (e.g., ViT-H or ViT-G) on classification, segmentation, depth estimation, Multimodal LLM (MLLM) benchmarks, and robotic manipulation. Notably, S^2 achieves state-of-the-art performance in detailed understanding of MLLM on V* benchmark, surpassing models such as GPT-4V. We examine the conditions under which S^2 is a preferred scaling approach compared to scaling on model size. While larger models have the advantage of better generalization on hard examples, we show that features of larger vision models can be well approximated by those of multi-scale smaller models. This suggests most, if not all, of the representations learned by current large pre-trained models can also be obtained from multi-scale smaller models. Our results confirm that a multi-scale smaller model has comparable learning capacity to a larger model, and show that pre-training smaller models with S^2 can match or even exceed the advantage of larger models.

</details>

---

## 84. Facial Affective Behavior Analysis with Instruction Tuning

- [ ] Facial Affective Behavior Analysis with Instruction Tuning | https://eccv.ecva.net/virtual/2024/poster/1812

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1812

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Facial affective behavior analysis (FABA) is crucial for understanding human mental states from images. However, traditional approaches primarily deploy models to discriminate among discrete emotion categories, and lack the fine granularity and reasoning capability for complex facial behaviors. The advent of Multi-modal Large Language Models (MLLMs) has been proven successful in general visual understanding tasks. However, directly harnessing MLLMs for FABA is challenging due to the scarcity of datasets and benchmarks, neglecting facial prior knowledge, and low training efficiency. To address these challenges, we introduce (i) an instruction-following dataset for two FABA tasks, e.g., emotion and action unit recognition, (ii) a benchmark FABA-Bench with a new metric considering both recognition and generation ability, and (iii) a new MLLM ''EmoLA'' as a strong baseline to the community. Our initiative on the dataset and benchmarks reveal the nature and rationale of facial affective behaviors, i.e., fine-grained facial movement, interpretability, and reasoning.  Moreover, to build an effective and efficient FABA MLLM, we introduce a facial prior expert module with face structure knowledge and a low-rank adaptation module into pre-trained MLLM. We conduct extensive experiments on FABA-Bench and four commonly-used FABA datasets. The results demonstrate that the proposed facial prior expert can boost the performance and EmoLA achieves the best results on our FABA-Bench. On commonly-used FABA datasets, EmoLA is competitive rivaling task-specific state-of-the-art models.

</details>

---

## 85. Discovering Novel Actions from Open World Egocentric Videos with Object-Grounded Visual Commonsense Reasoning

- [ ] Discovering Novel Actions from Open World Egocentric Videos with Object-Grounded Visual Commonsense Reasoning | https://eccv.ecva.net/virtual/2024/poster/1823

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1823

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Learning to infer labels in an open world, i.e., in an environment where the target "labels" are unknown, is an important characteristic for achieving autonomy. Foundation models, pre-trained on enormous amounts of data, have shown remarkable generalization skills through prompting, particularly in zero-shot inference. However, their performance is restricted to the correctness of the target label's search space, i.e., candidate labels provided in the prompt. This target search space can be unknown or exceptionally large in an open world, severely restricting their performance. To tackle this challenging problem, we propose a two-step, neuro-symbolic framework called ALGO - Action Learning with Grounded Object recognition that uses symbolic knowledge stored in large-scale knowledge bases to infer activities in egocentric videos with limited supervision. First, we propose a neuro-symbolic prompting approach that uses object-centric vision-language models as a noisy oracle to ground objects in the video through evidence-based reasoning. Second, driven by prior commonsense knowledge, we discover plausible activities through an energy-based symbolic pattern theory framework and learn to ground knowledge-based action (verb) concepts in the video. Extensive experiments on four publicly available datasets (EPIC-Kitchens, GTEA Gaze, GTEA Gaze Plus, and Charades-Ego) demonstrate its performance on open-world activity inference. We also show that ALGO can be extended to zero-shot inference and demonstrate its competitive performance. Code and additional qualitative analysis are provided as part of the supplementary and will be publicly available after review.

</details>

---

## 86. Merlin: Empowering Multimodal LLMs with Foresight Minds

- [ ] Merlin: Empowering Multimodal LLMs with Foresight Minds | https://eccv.ecva.net/virtual/2024/poster/1827

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1827

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Humans can foresee the future based on present observations, a skill we term as foresight minds. However, this capability remains under-explored within existing MLLMs, hindering their capacity to understand intentions behind subjects. To address this, we integrate the future modeling into MLLMs. By utilizing the trajectory, a highly structured representation, as a learning objective, we aim to equip the model to understand spatiotemporal dynamics. Inspired by the learning paradigm of LLMs, we first propose Foresight Pre-Training (FPT) that jointly learns various tasks centered on trajectories, enabling MLLMs to predict entire trajectories from a given initial observation. Then, we propose Foresight Instruction-Tuning (FIT) that requires MLLMs to reason about potential future events based on predicted trajectories. Aided by FPT and FIT, we build an unified MLLM named Merlin that supports complex future reasoning. Experiments show Merlin’s foresight minds with impressive performance on both future reasoning and visual comprehension tasks.

</details>

---

## 87. LITA: Language Instructed Temporal-Localization Assistant

- [ ] LITA: Language Instructed Temporal-Localization Assistant | https://eccv.ecva.net/virtual/2024/poster/1836

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1836

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

There has been tremendous progress in multimodal Large Language Models (LLMs). Recent works have extended these models to video input with promising instruction following capabilities. However, an important missing piece is temporal localization. These models cannot accurately answer the "When?" questions. We identify three key aspects that limit their temporal localization capabilities: (i) time representation, (ii) architecture, and (iii) data. We address these shortcomings by proposing Language Instructed Temporal-Localization Assistant (LITA) with the following features: (1) We introduce time tokens that encode timestamps relative to the video length to better represent time in videos. (2) We introduce SlowFast tokens in the architecture to capture temporal information at fine temporal resolution. (3) We emphasize temporal localization data for LITA. In addition to leveraging existing video datasets with timestamps, we propose a new task, Reasoning Temporal Localization (RTL), along with the dataset, ActivityNet-RTL, for learning and evaluating this task. Reasoning temporal localization requires both the reasoning and temporal localization of Video LLMs. LITA demonstrates strong performance on this challenging task, nearly doubling the temporal mean intersection-over-union (mIoU) of baselines. In addition, we show that our emphasis on temporal localization also substantially improves video-based text generation compared to existing Video LLMs, including a 36% relative improvement of Temporal Understanding.

</details>

---

## 88. WTS: A Pedestrian-Centric Traffic Video Dataset for Fine-grained Spatial-Temporal Understanding

- [ ] WTS: A Pedestrian-Centric Traffic Video Dataset for Fine-grained Spatial-Temporal Understanding | https://eccv.ecva.net/virtual/2024/poster/1840

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1840

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we address the challenge of fine-grained video event understanding in traffic scenarios, vital for autonomous driving and safety. Traditional datasets focus on driver or vehicle behavior, often neglecting pedestrian perspectives. To fill this gap, we introduce the WTS dataset, highlighting detailed behaviors of both vehicles and pedestrians across over 1.2k video events in over hundreds traffic scenarios. WTS integrates diverse perspectives from vehicle ego and fixed overhead cameras in a vehicle-infrastructure cooperative environment, enriched with comprehensive textual descriptions and unique 3D Gaze data for a synchronized 2D/3D view, focusing on pedestrian analysis. We also provide annotations for 5k publicly sourced pedestrian-related traffic videos. Additionally, we introduce LLMScorer, an LLM-based evaluation metric to align inference captions with ground truth. Using WTS, we establish a benchmark for dense video-to-text tasks, exploring state-of-the-art Vision-Language Models with an instance-aware VideoLLM method as a baseline. WTS aims to advance fine-grained video event understanding, enhancing traffic safety and autonomous driving development. Dataset page: https://woven-visionai.github.io/wts-dataset-homepage/.

</details>

---

## 89. Reinforcement Learning Friendly Vision-Language Model for Minecraft

- [ ] Reinforcement Learning Friendly Vision-Language Model for Minecraft | https://eccv.ecva.net/virtual/2024/poster/1845

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1845

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

One of the essential missions in the AI research community is to build an autonomous embodied agent that can achieve high-level performance across a wide spectrum of tasks. However, acquiring or manually designing rewards for all open-ended tasks is unrealistic. In this paper, we propose a novel cross-modal contrastive learning framework architecture, CLIP4MC, aiming to learn a reinforcement learning (RL) friendly vision-language model (VLM) that serves as an intrinsic reward function for open-ended tasks. Simply utilizing the similarity between the video snippet and the language prompt is not RL-friendly since standard VLMs may only capture the similarity at a coarse level. To achieve RL-friendliness, we incorporate the task completion degree into the VLM training objective, as this information can assist agents in distinguishing the importance between different states. Moreover, we provide neat YouTube datasets based on the large-scale YouTube database provided by MineDojo. Specifically, two rounds of filtering operations guarantee that the dataset covers enough essential information and that the video-text pair is highly correlated. Empirically, we demonstrate that the proposed method achieves better performance on RL tasks compared with baselines.

</details>

---

## 90. DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-level Control

- [ ] DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-level Control | https://eccv.ecva.net/virtual/2024/poster/1846

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1846

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Building a general-purpose intelligent home-assistant agent skilled in diverse tasks by human commands is a long-term blueprint of embodied AI research, which poses requirements on task planning, environment modeling, and object interaction. In this work, we study primitive mobile manipulations for embodied agents, i.e. how to navigate and interact based on an instructed verb-noun pair. We propose DISCO, which features non-trivial advancements in contextualized scene modeling and efficient controls. In particular, DISCO incorporates differentiable scene representations of rich semantics in object and affordance, which is dynamically explored on the fly and facilitates navigation planning.  Besides, we propose dual-level coarse-to-fine action controls leveraging both global and local cues to accomplish mobile manipulation tasks efficiently.  DISCO easily integrates into embodied tasks such as embodied instruction following. To validate our approach, we take the ALFRED benchmark, of large-scale long-horizon vision-language navigation and interaction tasks, as a test bed. In extensive experiments, we make comprehensive evaluations and demonstrate that DISCO outperforms the art by a sizable +8.6\% success rate margin in unseen scenes, even without step-by-step instructions. Our code and model will be made publicly available.

</details>

---

## 91. HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning

- [ ] HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning | https://eccv.ecva.net/virtual/2024/poster/1849

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1849

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in visual reasoning (VR), particularly with the aid of Large Vision-Language Models (VLMs), show promise but require access to large-scale datasets and face challenges such as high computational costs and limited generalization capabilities. Compositional visual reasoning approaches have emerged as effective strategies; however, they heavily rely on the commonsense knowledge encoded in Large Language Models (LLMs) to perform planning, reasoning, or both, without considering the effect of their decisions on the visual reasoning process, which can lead to errors or failed procedures. To address these challenges, we introduce HYDRA, a multi-stage dynamic compositional visual reasoning framework designed for reliable and incrementally progressive general reasoning. HYDRA integrates three essential modules: a planner, a Reinforcement Learning (RL) agent serving as a cognitive controller, and a reasoner. The planner and reasoner modules utilize an LLM to generate instruction samples and executable code from the selected instruction, respectively, while the RL agent dynamically interacts with these modules, making high-level decisions on selection of the best instruction sample given information from the historical state stored through a feedback loop. This adaptable design enables HYDRA to adjust its actions based on previous feedback received during the reasoning process, leading to more reliable reasoning outputs and ultimately enhancing its overall effectiveness. Our framework demonstrates state-of-the-art performance in various VR tasks on four different widely-used datasets.

</details>

---

## 92. Multi-Task Domain Adaptation for Language Grounding with 3D Objects

- [ ] Multi-Task Domain Adaptation for Language Grounding with 3D Objects | https://eccv.ecva.net/virtual/2024/poster/1851

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1851

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The existing works on object-level language grounding with 3D objects mostly focus on improving performance by utilizing the off-the-shelf pre-trained models to capture features, such as viewpoint selection or geometric priors.  However, they have failed to consider exploring the cross-modal representation of language-vision alignment in the cross-domain field.  To answer this problem, we propose a novel method called Domain Adaptation for Language Grounding (DA4LG) with 3D objects. Specifically, the proposed DA4LG consists of a visual adapter module with multi-task learning to realize vision-language alignment by comprehensive multimodal feature representation. Experimental results demonstrate that DA4LG competitively performs across visual and non-visual language descriptions, independent of the completeness of observation.  DA4LG achieves state-of-the-art performance in the single-view setting and multi-view setting with the accuracy of 83.8 % and 86.8 % respectively in the language grounding benchmark SNARE. The simulation experiments show the well-practical and generalized performance of DA4LG compared to the existing methods. Our project is available anonymously at https://sites.google.com/view/da4lg.

</details>

---

## 93. MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?

- [ ] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems? | https://eccv.ecva.net/virtual/2024/poster/1852

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1852

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The remarkable progress of Multi-modal Large Language Models (MLLMs) has garnered unparalleled attention, due to their superior performance in visual contexts. However, their capabilities in visual math problem-solving remain insufficiently evaluated and understood. We investigate current benchmarks to incorporate excessive visual content within textual questions, which potentially assist MLLMs in deducing answers without truly interpreting the input diagrams. To this end, we introduce MathVerse, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. We meticulously collect 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. Each problem is then transformed by human annotators into six distinct versions, each offering varying degrees of information content in multi-modality, contributing to 15K test samples in total. This approach allows MathVerse to comprehensively assess whether and how much MLLMs can truly understand the visual diagrams for mathematical reasoning. In addition, we propose a Chain-of-Thought (CoT) evaluation strategy for a fine-grained assessment of the output answers. Rather than naively judging true or false, we employ GPT-4(V) to adaptively extract crucial reasoning steps, and then assess each step with error analysis to derive a total score, which can reveal the inner CoT reasoning quality by MLLMs. With MathVerse, we unveil that, most existing MLLMs struggle to understand math diagrams, relying heavily on textual questions. Surprisingly, some of them even achieve 5%+ higher accuracy without the visual input, e.g., Gemini-Pro and SPHINX-MoE. In contrast, GPT-4V and InternLM-XComposer2 demonstrate relatively better comprehension of the visual content for mathematical reasoning. We hope the MathVerse benchmark may provide unique insights to guide the future development of MLLMs.

</details>

---

## 94. Q&A Prompts: Discovering Rich Visual Clues through Mining Question-Answer Prompts for VQA requiring Diverse World Knowledge

- [ ] Q&A Prompts: Discovering Rich Visual Clues through Mining Question-Answer Prompts for VQA requiring Diverse World Knowledge | https://eccv.ecva.net/virtual/2024/poster/1853

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1853

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the breakthrough of multi-modal large language models, answering complex visual questions that demand advanced reasoning abilities and world knowledge has become a much more important testbed for developing AI models than ever. However, equipping AI models with robust cross-modality reasoning ability remains challenging since the cognition scheme of humans has not been understood systematically. In this paper, we believe that if we can collect visual clues of each instance in the given image, we will recognize the image more accurately, understand the question better, recall relevant knowledge more easily, and finally reason out the answer. We discover these important and rich visual clues by mining question-answer pairs in images and sending them into multi-modal large language models as prompts. We call the proposed method Q&A Prompts. Specifically, we first use the image-answer pairs and the corresponding questions in the training set as inputs and outputs to train a visual question generation model. Then, we use an image tagging model to identify various instances and send packaged image-tag pairs into the visual question generation model to generate relevant questions with the extracted image tags as answers. Finally, we encode these generated question-answer pairs as prompts with a visual-aware prompting module and send them into pre-trained multi-modal large language models to reason out the final answers. Experimental results show that, compared with state-of-the-art methods, our Q&A Prompts achieves substantial improvements on the challenging visual question answering datasets requiring reasoning over diverse world knowledge, such as OK-VQA and A-OKVQA.

</details>

---

## 95. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory

- [ ] Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory | https://eccv.ecva.net/virtual/2024/poster/1857

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1857

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening adversarial attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can stimulate further research on constructing reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks (e.g., Image-Text Retrieval(ITR), Visual Grounding(VG), Image Captioning(IC)).

</details>

---

## 96. How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs

- [ ] How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs | https://eccv.ecva.net/virtual/2024/poster/1855

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1855

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This work focuses on the potential of vision large language models (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite Unicorn, covering out-of-distribution (OOD) generalization and adversarial robustness.  For the OOD evaluation, we present two novel visual question-answering (VQA) datasets, each with one variant, designed to test model performance under challenging conditions.  In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language input of VLLMs. Our evaluation of 22 diverse models, ranging from open-source VLLMs to GPT-4V and Gemini Pro, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at https://github.com/UCSC-VLAA/vllm-safety-benchmark.

</details>

---

## 97. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models

- [ ] MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models | https://eccv.ecva.net/virtual/2024/poster/1856

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1856

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits.

</details>

---

## 98. Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models

- [ ] Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/1860

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1860

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Hallucinations in vision-language models pose a significant challenge to their reliability, particularly in the generation of long captions. Current methods fall short of accurately identifying and mitigating these hallucinations. To address this issue, we introduce ESREAL, a novel unsupervised learning framework designed to suppress the generation of hallucinations through accurate localization and penalization of hallucinated tokens. Initially, ESREAL creates a reconstructed image based on the generated caption and aligns its corresponding regions with those of the original image. This semantic reconstruction aids in identifying both the presence and type of token-level hallucinations within the generated caption. Subsequently, ESREAL computes token-level hallucination scores by assessing the semantic similarity of aligned regions based on the type of hallucination. Finally, ESREAL employs a proximal policy optimization algorithm, where it selectively penalizes hallucinated tokens according to their token-level hallucination scores. Our framework notably reduces hallucinations in LLaVA, InstructBLIP, and mPLUG-Owl2 by 32.81%, 27.08%, and 7.46% on the CHAIR metric. This improvement is achieved solely through signals derived from the image itself, without the need for any image-text pairs.

</details>

---

## 99. Introducing Routing Functions to Vision-Language Parameter-Efficient Fine-Tuning with Low-Rank Bottlenecks

- [ ] Introducing Routing Functions to Vision-Language Parameter-Efficient Fine-Tuning with Low-Rank Bottlenecks | https://eccv.ecva.net/virtual/2024/poster/1861

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1861

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Mainstream parameter-efficient fine-tuning (PEFT) methods, such as LoRA or Adapter, project a model's hidden states to a lower dimension, allowing pre-trained models to adapt to new data through this low-rank bottleneck. However, PEFT tasks involving multiple modalities, like vision-language (VL) tasks, require not only adaptation to new data but also learning the relationship between different modalities. Targeting at VL PEFT tasks, we propose a family of operations, called routing functions, to enhance VL alignment in the low-rank bottlenecks. The routing functions adopt linear operations and do not introduce new trainable parameters. In-depth analyses are conducted to study their behavior. In various VL PEFT settings, the routing functions significantly improve performance of the original PEFT methods, achieving over 20\% improvement on VQAv2 ($\text{RoBERTa}_{\text{large}}$+ViT-L/16) and 30\% on COCO Captioning (GPT2-medium+ViT-L/16). Also when fine-tuning a pre-trained multimodal model such as CLIP-BART, we observe smaller but consistent improvements across a range of VL PEFT tasks.

</details>

---

## 100. UMG-CLIP: A Unified Multi-Granularity Vision Generalist for Open-World Understanding

- [ ] UMG-CLIP: A Unified Multi-Granularity Vision Generalist for Open-World Understanding | https://eccv.ecva.net/virtual/2024/poster/1862

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1862

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language foundation models, represented by Contrastive Language-Image Pre-training (CLIP), have gained increasing attention for jointly understanding both vision and textual tasks. However, existing approaches primarily focus on training models to match global image representations with textual descriptions, thereby overlooking the critical alignment between local regions and corresponding text tokens. This paper extends CLIP with multi-granularity alignment. Notably, we deliberately construct a new dataset comprising pseudo annotations at various levels of granularities, encompassing image-level, region-level as well as pixel-level captions and tags. Accordingly, we develop a Unified Multi-Granularity learning framework, termed UMG-CLIP, which simultaneously empowers the model with versatile perception abilities across different levels of detail. With parameter efficient tuning, UMG-CLIP surpasses current widely used CLIP variants and achieves state-of-the-art performance on diverse image understanding benchmarks, including open-world recognition, retrieval, semantic segmentation, and panoptic segmentation tasks. We believe that UMG-CLIP represents a valuable advancement in vision-language foundation models.

</details>

---

## 101. Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning

- [ ] Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning | https://eccv.ecva.net/virtual/2024/poster/1866

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1866

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, zero-shot image captioning has gained increasing attention, where only text data is available for training. The remarkable progress in text-to-image diffusion model presents the potential to resolve this task by employing synthetic image-caption pairs generated by this pre-trained prior. Nonetheless, the defective details in the salient regions of the synthetic images introduce semantic misalignment between the synthetic image and text, leading to compromised results. To address this challenge, we propose a novel Patch-wise Cross-modal feature Mix-up (PCM) mechanism to adaptively mitigate the unfaithful contents in a fine-grained manner during training, which can be integrated into most of encoder-decoder frameworks, introducing our PCM-Net. Specifically, for each input image, salient visual concepts in the image are first detected considering the image-text similarity in CLIP space. Next, the patch-wise visual features of the input image are selectively fused with the textual features of the salient visual concepts, leading to a mixed-up feature map with less defective content. Finally, a visual-semantic encoder is exploited to refine the derived feature map, which is further incorporated into the sentence decoder for caption generation. Additionally, to facilitate the model training with synthetic data, a novel CLIP-weighted cross-entropy loss is devised to prioritize the high-quality image-text pairs over the low-quality counterparts. Extensive experiments on MSCOCO and Flickr30k datasets demonstrate the superiority of our PCM-Net compared with state-of-the-art VLMs-based approaches. It is noteworthy that our PCM-Net ranks first in both in-domain and cross-domain zero-shot image captioning. The synthetic dataset SynthImgCap and code are available at https://jianjieluo.github.io/SynthImgCap.

</details>

---

## 102. GalLop: Learning global and local prompts for vision-language models

- [ ] GalLop: Learning global and local prompts for vision-language models | https://eccv.ecva.net/virtual/2024/poster/1871

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1871

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has been widely adopted to efficiently adapt vision-language models (VLMs), e.g. CLIP, for few-shot image classification. Despite their success, most prompt learning methods trade-off between classification accuracy and robustness, e.g. in domain generalization or out-of-distribution (OOD) detection. In this work, we introduce Global-Local Prompts (GalLoP), a new prompt learning method that learns multiple diverse prompts leveraging both global and local visual features. The training of the local prompts relies on local features with an enhanced vision-text alignment. To focus only on pertinent features, this local alignment is coupled with a sparsity strategy in the selection of the local features. We enforce diversity on the set of prompts using a new ``prompt dropout'' technique and a multiscale strategy on the local prompts. GalLoP outperforms previous prompt learning methods on accuracy on eleven datasets in different few shots settings and with various backbones. Furthermore, GalLoP shows strong robustness performances in both domain generalization and OOD detection, even outperforming dedicated OOD detection methods. Code and instructions to reproduce our results will be open-sourced.

</details>

---

## 103. CoLA: Conditional Dropout and Language-driven Robust Dual-modal Salient Object Detection

- [ ] CoLA: Conditional Dropout and Language-driven Robust Dual-modal Salient Object Detection | https://eccv.ecva.net/virtual/2024/poster/1873

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1873

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The depth/thermal information is beneficial for detecting salient object with conventional RGB images. However, in dual-modal salient object detection (SOD) model, the robustness against noisy inputs and modality missing is crucial but rarely studied. To tackle this problem, we introduce \textbf{Co}nditional Dropout and \textbf{LA}nguage-driven(\textbf{CoLA}) framework comprising two core components. 1) Language-driven Quality Assessment (LQA): Leveraging a pretrained vision-language model with a prompt learner, the LQA recalibrates image contributions without requiring additional quality annotations. This approach effectively mitigates the impact of noisy inputs. 2) Conditional Dropout (CD):  A learning method to strengthen the model's adaptability in scenarios with missing modalities, while preserving its performance under complete modalities. The CD serves as a plug-in training scheme that treats modality-missing as conditions, strengthening the overall robustness of various dual-modal SOD models. Extensive experiments demonstrate that the proposed method outperforms state-of-the-art dual-modal SOD models, under both modality-complete and modality-missing conditions.

</details>

---

## 104. Griffon: Spelling out All Object Locations at Any Granularity with Large Language Models

- [ ] Griffon: Spelling out All Object Locations at Any Granularity with Large Language Models | https://eccv.ecva.net/virtual/2024/poster/1874

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1874

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Replicating the innate human ability to detect all objects based on free-form texts at any granularity remains a formidable challenge for Large Vision Language Models (LVLMs). Current LVLMs are predominantly constrained to locate a single, pre-existing object. This limitation leads to a compromise in model design, necessitating the introduction of visual expert models or the customized head structures. Beyond these constraints, our research uncovers LVLMs' capability for basic object perception, allowing them to accurately identify and locate objects of interest. Building on this insight, we introduce a novel Language-prompted Localization Dataset to fully unleash the capabilities of LVLMs in fine-grained object perception and precise location awareness. More importantly, we present Griffon, a purely LVLM-based baseline, which does not introduce any special tokens, expert models, or additional detection modules. It simply maintains a consistent structure with popular LVLMs by unifying data formats across various localization-related scenarios and is trained end-to-end through a well-designed pipeline. Comprehensive experiments demonstrate that Griffon not only achieves state-of-the-art performance on the fine-grained RefCOCO series and Flickrs30K Entities but also approaches the capabilities of the expert model Faster RCNN on the detection benchmark MSCOCO. Dataset, codes and models will be released.

</details>

---

## 105. PartGLEE: A Foundation Model for Recognizing and Parsing Any Objects

- [ ] PartGLEE: A Foundation Model for Recognizing and Parsing Any Objects | https://eccv.ecva.net/virtual/2024/poster/1887

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1887

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present PartGLEE, a part-level foundation model for locating and identifying both objects and parts in images. Through a unified framework, PartGLEE accomplishes detection, segmentation, and grounding of instances at any granularity in the open world scenario. Specifically, we propose a Q-Former to construct the hierarchical relationship between objects and parts, parsing every object into corresponding semantic parts. By incorporating a large amount of object-level data, the hierarchical relationships can be extended, enabling PartGLEE to recognize a rich variety of parts. We conduct comprehensive empirical studies to validate the effectiveness of our method, PartGLEE achieves the state-of-the-art performance across various part-level tasks and maintain comparable results on object-level tasks. Our further analysis indicates that the hierarchical cognitive ability of PartGLEE is able to facilitate a detailed comprehension in images for mLLMs. Code will be released.

</details>

---

## 106. BlenderAlchemy: Editing 3D Graphics with Vision-Language Models

- [ ] BlenderAlchemy: Editing 3D Graphics with Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/190

- **Link**: https://eccv.ecva.net/virtual/2024/poster/190

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Graphics design is important for various applications, including movie production and game design. To create a high-quality scene, designers usually need to spend hours in software like Blender, in which they might need to interleave and repeat operations, such as connecting material nodes, hundreds of times. Moreover, slightly different design goals may require completely different sequences, making automation difficult. In this paper, we propose a system that leverages Vision-Language Models (VLMs), like GPT-4V, to intelligently search the design action space to arrive at an answer that can satisfy a user's intent. Specifically, we design a vision-based edit generator and state evaluator to work together to find the correct sequence of actions to achieve the goal. Inspired by the role of visual imagination in the human design process, we supplement the visual reasoning capabilities of VLMs with ``imagined'' reference images from image-generation models, providing visual grounding of abstract language descriptions. In this paper, we provide empirical evidence suggesting our system can produce simple but tedious Blender editing sequences for tasks such as editing procedural materials from text and/or reference images, as well as adjusting lighting configurations for product renderings in complex scenes.

</details>

---

## 107. Efficient and Versatile Robust Fine-Tuning of Zero-shot Models

- [ ] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models | https://eccv.ecva.net/virtual/2024/poster/1927

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1927

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale image-text pre-trained models enable zero-shot classification and provide consistent accuracy across various data distributions. Nonetheless, optimizing these models in downstream tasks typically requires fine-tuning, which reduces generalization to out-of-distribution (OOD) data and demands extensive computational resources. We introduce Robust Adapter (R-Adapter), a novel method for fine-tuning zero-shot models to downstream tasks while simultaneously addressing both these issues. Our method integrates lightweight modules into the pre-trained model and employs novel self-ensemble techniques to boost OOD robustness and reduce storage expenses substantially. Furthermore, we propose MPM-NCE loss designed for fine-tuning on vision-language downstream tasks. It ensures precise alignment of multiple image-text pairs and discriminative feature learning. By extending the benchmark for robust fine-tuning beyond classification to include diverse tasks such as cross-modal retrieval and open vocabulary segmentation, we demonstrate the broad applicability of R-Adapter. Our extensive experiments demonstrate that R-Adapter achieves state-of-the-art performance across a diverse set of tasks, tuning only 13% of the parameters of the CLIP encoders.

</details>

---

## 108. Towards Neuro-Symbolic Video Understanding

- [ ] Towards Neuro-Symbolic Video Understanding | https://eccv.ecva.net/virtual/2024/poster/1979

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1979

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The unprecedented surge in video data production in recent years necessitates efficient tools for extracting meaningful frames from videos for downstream tasks. Long-term temporal reasoning is a key desideratum for frame retrieval systems. While state-of-the-art foundation models, like VideoLLaMA and ViCLIP, are proficient in short-term semantic understanding, they surprisingly fail at long-term reasoning across frames. A key reason for their failure is that they intertwine per-frame perception and temporal reasoning into a single deep network. Hence, decoupling but co-designing semantic understanding and temporal reasoning is essential for efficient scene identification. We propose a system that leverages vision-language models for semantic understanding of individual frames but effectively reasons about the long-term evolution of events using state machines and temporal logic (TL) formulae that inherently capture memory. Our TL-based reasoning improves the F1 score of complex event identification by 9-15% compared to benchmarks that use GPT4 for reasoning on state-of-the-art self-driving datasets such as Waymo and NuScenes.

</details>

---

## 109. LongVLM: Efficient Long Video Understanding via Large Language Models

- [ ] LongVLM: Efficient Long Video Understanding via Large Language Models | https://eccv.ecva.net/virtual/2024/poster/1989

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1989

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Empowered by Large Language Models (LLMs), recent advancements in VideoLLMs have driven progress in various video understanding tasks. These models encode video representations through pooling or query aggregation over a vast amount of visual tokens, making computational and memory costs affordable. Despite successfully providing an overall comprehension of video content, existing VideoLLMs still face challenges in achieving detailed understanding in videos due to overlooking local information in long-term videos. To tackle this challenge, we introduce LongVLM, a straightforward yet powerful VideoLLM for long video understanding, building upon the observation that long videos often consist of sequential key events, complex actions, and camera movements. Our approach proposes to decompose long videos into multiple short-term segments and encode local features for each local segment via a hierarchical token merging module. These features are concatenated in temporal order to maintain the storyline across sequential short-term segments. Additionally, we propose to integrate global semantics into each local feature to enhance context understanding. In this way, we encode video representations that incorporate both local and global information, enabling the LLM to generate comprehensive responses for long-term videos. Experimental results on the VideoChatGPT benchmark and zero-shot video question-answering datasets demonstrate the superior capabilities of our model over the previous state-of-the-art methods. Qualitative examples demonstrate that our model produces more precise responses for long videos understanding.

</details>

---

## 110. Turbo: Informativity-Driven Acceleration Plug-In for Vision-Language Large Models

- [ ] Turbo: Informativity-Driven Acceleration Plug-In for Vision-Language Large Models | https://eccv.ecva.net/virtual/2024/poster/1997

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1997

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Large Models (VLMs) recently become primary backbone of AI, due to the impressive performance. However, their expensive computation costs,  i.e., throughput and delay, impede potentials in the real-world scenarios. To achieve acceleration for VLMs, most existing methods focus on the model perspective: pruning, distillation, quantization, but completely overlook the data-perspective redundancy. To fill the overlook, this paper pioneers the severity of data redundancy, and designs one plug-and-play Turbo module guided by information degree to prune inefficient tokens from visual or textual data. In pursuit of efficiency-performance trade-offs, information degree takes two crucial factors into consideration: mutual redundancy and semantic value. Concretely, the former evaluates data duplication between sequential tokens; while the latter evaluates each token by its contribution to the overall semantics. As a result, tokens with high information degree carry less redundancy and stronger semantics. For VLMs' calculation, Turbo works as a user-friendly plug-in that sorts data referring to information degree, utilizing only top-level ones to save costs. Its advantages are multifaceted, e.g., being generally compatible to various VLMs across understanding and generation, simple use without retraining and trivial engineering efforts. On multiple VLMs benchmarks, we fully experiment to reveal good acceleration of Turbo, under negligible performance drop.

</details>

---

## 111. Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization

- [ ] Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization | https://eccv.ecva.net/virtual/2024/poster/1993

- **Link**: https://eccv.ecva.net/virtual/2024/poster/1993

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) excel in generating responses based on visual inputs. However, they often suffer from a bias towards generating responses similar to their pretraining corpus, overshadowing the importance of visual information. We treat this bias as a "preference" for pretraining statistics, which hinders the model's grounding in visual input. To mitigate this issue, we propose Bootstrapped Preference Optimization (BPO), which conducts preference learning with datasets containing negative responses bootstrapped from the model itself. Specifically, we propose the following two strategies: 1) using distorted image inputs to the MLLM for eliciting responses that contain signified pretraining bias; 2) leveraging text-based LLM to explicitly inject erroneous but common elements into the original response. Those undesirable responses are paired with original annotated responses from the datasets to construct the preference dataset, which is subsequently utilized to perform preference learning. Our approach effectively suppresses pretrained LLM bias, enabling enhanced grounding in visual inputs. Extensive experimentation demonstrates significant performance improvements across multiple benchmarks, advancing the state-of-the-art in multimodal conversational systems.

</details>

---

## 112. BRAVE: Broadening the visual encoding of vision-language models

- [ ] BRAVE: Broadening the visual encoding of vision-language models | https://eccv.ecva.net/virtual/2024/poster/2001

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2001

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) are typically composed of a vision encoder, e.g. CLIP, and a language model (LM) that interprets the encoded features to solve downstream tasks. Despite remarkable progress, VLMs are subject to several shortcomings due to the limited capabilities of vision encoders, e.g. ``blindness'' to certain image features, visual hallucination, etc. To address these issues, we study broadening of the visual encoding capabilities of VLMs. We first comprehensively benchmark several vision encoders with different inductive biases for solving VLM tasks. We observe that there is no single encoding configuration that consistently achieves top performance across different tasks, and encoders with different biases can perform surprisingly similarly. Motivated by this, we introduce a method, named BRAVE, that consolidates features from multiple frozen encoders into a more versatile representation that can be directly fed as the input to a frozen LM. BRAVE achieves state-of-the-art performance on a broad range of captioning and VQA benchmarks and significantly reduces the aforementioned issues of VLMs, while requiring a smaller number of trainable parameters than existing methods and having a more compressed representation. Our results highlight the potential of incorporating different visual biases for a more broad and contextualized visual understanding of VLMs.

</details>

---

## 113. MMBENCH: Is Your Multi-Modal Model an All-around Player?

- [ ] MMBENCH: Is Your Multi-Modal Model an All-around Player? | https://eccv.ecva.net/virtual/2024/poster/2003

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2003

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) have recently achieved remarkable progress, exhibiting impressive multimodal perception and reasoning abilities. However, effectively evaluating these large VLMs remains a major challenge, hindering future development in this domain. Traditional benchmarks like VQAv2 or COCO Caption provide quantitative performance measurements but lack fine-grained ability assessment and robust evaluation metrics. Meanwhile, subjective benchmarks, such as OwlEval, offer comprehensive evaluations of a model's abilities by incorporating human labor, which is not scalable and may display significant bias. In response to these challenges, we propose MMBench, a bilingual benchmark for assessing the multi-modal capabilities of VLMs. MMBench methodically develops a comprehensive evaluation pipeline, primarily comprised of the following key features: 1. MMBench is meticulously curated with well-designed quality control schemes, surpassing existing similar benchmarks in terms of the number and variety of evaluation questions and abilities; 2. MMBench introduces a rigorous CircularEval strategy and incorporates large language models to convert free-form predictions into pre-defined choices, which helps to yield accurate evaluation results for models with limited instruction-following capabilities. 3. MMBench incorporates multiple-choice questions in both English and Chinese versions, enabling an apples-to-apples comparison of VLMs' performance under a bilingual context. To summarize, MMBench is a systematically designed objective benchmark for a robust and holistic evaluation of vision-language models. We hope MMBench will assist the research community in better evaluating their models and facilitate future progress in this area.

</details>

---

## 114. uCAP: An Unsupervised Prompting Method for Vision-Language Models

- [ ] uCAP: An Unsupervised Prompting Method for Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/2005

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2005

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper addresses a significant limitation that prevents Contrastive Language-Image Pretrained Models (CLIP) from achieving optimal performance on downstream image classification tasks. The key problem with CLIP-style zero-shot classification is that it requires domain-specific context in the form of prompts to better align the class descriptions to the downstream data distribution. In particular, prompts for vision-language models are domain-level texts (e.g., ``a centered satellite image of ...'') which, together with the class names, are fed into the text encoder to provide more context for the downstream dataset. These prompts are typically manually tuned, which is time consuming and often sub-optimal. To overcome this bottleneck, this paper proposes uCAP, a method to automatically learn domain-specific prompts/contexts using only unlabeled in-domain images. We achieve this by modeling the generation of images given the class names and a domain-specific prompt with an unsupervised likelihood distribution, and then performing inference of the prompts. We validate the proposed method across various models and datasets, showing that uCAP consistently outperforms manually tuned prompts and related baselines on the evaluated datasets: ImageNet, CIFAR-10, CIFAR-100, OxfordPets (up to 2\%), SUN397 (up to 5\%), and Caltech101 (up to 3\%).

</details>

---

## 115. An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models

- [ ] An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/2009

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2009

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this study, we identify the inefficient attention phenomena in Large Vision-Language Models (LVLMs), notably within prominent models like LLaVA-1.5, QwenVL-Chat and Video-LLaVA. We find out that the attention computation over visual tokens is of extreme inefficiency in the deep layers of popular LVLMs, suggesting a need for a sparser approach compared to textual data handling. To this end, we introduce FastV, a versatile plug-and-play method designed to optimize computational efficiency by learning adaptive attention patterns in early layers and pruning visual tokens in subsequent ones.  Our evaluations demonstrate FastV's ability to dramatically reduce computational costs (e.g., a 45\% reduction in FLOPs for LLaVA-1.5-13B) without sacrificing performance in a wide range of image and video understanding tasks. The computational efficiency and performance trade-off of FastV are highly customizable and pareto-efficient. It can compress the FLOPs of a 13B-parameter model to achieve a lower budget than that of a 7B-parameter model, while still maintaining superior performance. We believe FastV has practical values for deployment of LVLMs in edge devices and commercial models.  Code will be released upon acceptance.

</details>

---

## 116. Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models

- [ ] Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models | https://eccv.ecva.net/virtual/2024/poster/2013

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2013

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pre-training (VLP) models like CLIP have achieved remarkable success in computer vision and particularly demonstrated superior robustness to distribution shifts of 2D images. However, their robustness under 3D viewpoint variations is still limited, which can hinder the development for real-world applications. This paper successfully addresses this concern while keeping VLPs' original performance by breaking through two primary obstacles: 1) the scarcity of training data and 2) the suboptimal fine-tuning paradigms. To combat data scarcity, we build the Multi-View Caption (MVCap) dataset --- a comprehensive collection of over four million multi-view image-text pairs across more than 100K objects, providing more potential for VLP models to develop generalizable viewpoint-invariant representations. To address the limitations of existing paradigms in performance trade-offs and training efficiency, we design a novel fine-tuning framework named Omniview-Tuning (OVT). Specifically, OVT introduces a Cross-Viewpoint Alignment objective through a minimax-like optimization strategy, which effectively aligns representations of identical objects from diverse viewpoints without causing overfitting. Additionally, OVT fine-tunes VLP models in a parameter-efficient manner, leading to minimal computational cost. Extensive experiments on various VLP models with different architectures validate that OVT significantly improves the models' resilience to viewpoint shifts and keeps the original performance, establishing a pioneering standard for boosting viewpoint invariance of VLP models.

</details>

---

## 117. Chat-Edit-3D: Interactive 3D Scene Editing via Text Prompts

- [ ] Chat-Edit-3D: Interactive 3D Scene Editing via Text Prompts | https://eccv.ecva.net/virtual/2024/poster/2051

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2051

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent work on image content manipulation based on vision-language pre-training models has been effectively extended to text-driven 3D scene editing. However, existing schemes for 3D scene editing still exhibit certain shortcomings, hindering their further interactive design. Such schemes typically adhere to fixed input patterns, limiting users' flexibility in text input. Moreover, their editing capabilities are constrained by a single or a few 2D visual models and require intricate pipeline design to integrate these models into 3D reconstruction processes. To address the aforementioned issues, we propose a dialogue-based 3D scene editing approach, termed CE3D, which is centered around a large language model that allows for arbitrary textual input from users and interprets their intentions, subsequently facilitating the autonomous invocation of the corresponding visual expert models. Furthermore, we design a scheme utilizing Hash-Atlas to represent 3D scene views, which transfers the editing of 3D scenes onto 2D atlas images. This design achieves complete decoupling between the 2D editing and 3D reconstruction processes, enabling CE3D to flexibly integrate a wide range of existing 2D or 3D visual models without necessitating intricate fusion designs. Experimental results demonstrate that CE3D effectively integrates multiple visual models to achieve diverse editing visual effects, possessing strong scene comprehension and multi-round dialog capabilities.

</details>

---

## 118. XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution

- [ ] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution | https://eccv.ecva.net/virtual/2024/poster/2167

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2167

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Diffusion-based methods, endowed with a formidable generative prior, have received increasing attention in Image Super-Resolution (ISR) recently. However, as low-resolution (LR) images often undergo severe degradation, it is challenging for ISR models to perceive the semantic and degradation information, resulting in restoration images with incorrect content or unrealistic artifacts. To address these issues, we propose a \textit{Cross-modal Priors for Super-Resolution (XPSR)} framework. Within XPSR, to acquire precise and comprehensive semantic conditions for the diffusion model, cutting-edge Multimodal Large Language Models (MLLMs) are utilized. To facilitate better fusion of cross-modal priors, a \textit{Semantic-Fusion Attention} is raised. To distill semantic-preserved information instead of undesired degradations, a \textit{Degradation-Free Constraint} is attached between LR and its high-resolution (HR) counterpart. Quantitative and qualitative results show that XPSR is capable of generating high-fidelity and high-realism images across synthetic and real-world datasets. The model and codes will be made publicly available.

</details>

---

## 119. Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning

- [ ] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning | https://eccv.ecva.net/virtual/2024/poster/2207

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2207

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The integration with CLIP (Contrastive Vision-Language Pre-training) has significantly refreshed the accuracy leaderboard of FSAR (Few-shot Action Recognition). However, the trainable overhead of ensuring that the domain alignment of CLIP and FSAR is often unbearable. To mitigate this issue, we present an Efficient Multi-Level Post-Reasoning Network, namely EMP-Net. By design,  a post-reasoning mechanism is been proposed for domain adaptation, which avoids most gradient backpropagation, improving the efficiency; meanwhile, a multi-level representation is utilised during the reasoning and matching processes to improve the discriminability, ensuring effectiveness. Specifically, the proposed EMP-Net starts with a skip-fusion involving cached multi-stage features extracted by CLIP.  After that, current feature are decoupled into multi-level representations, including global-level, patch-level, and frame-level.  The ensuing spatiotemporal reasoning module operates on multi-level representations to generate discriminative features. As for matching, the multi-level contrasts between text-visual and support-query are integrated to provide a comprehensive guidance.  The experimental results demonstrate that EMP-Net can unlock the potential performance of CLIP in a more efficient manner. Please find our code in supplementary materials.

</details>

---

## 120. ViLA: Efficient Video-Language Alignment for Video Question Answering

- [ ] ViLA: Efficient Video-Language Alignment for Video Question Answering | https://eccv.ecva.net/virtual/2024/poster/2212

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2212

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we propose an efficient Video-Language Alignment (ViLA) network. Our ViLA model addresses both efficient frame sampling and effective cross-modal alignment in a unified way. In our ViLA network, we design a new learnable text-guided Frame-Prompter together with a cross-modal distillation (QFormer-Distiller) module. Pre-trained large image-language models have shown promising results on problems such as visual question answering (VQA). However, how to efficiently and effectively sample video frames when adapting pre-trained large image-language model to video-language alignment is still the major challenge. Compared with prior work, our ViLA model demonstrates the capability of selecting key frames with critical contents, thus improving the video-language alignment accuracy while reducing the inference latency (+3.3% on NExT-QA Temporal with 3.0X speed up).  Overall, our ViLA network outperforms the state-of-the-art methods on the video question-answering benchmarks: +4.6% on STAR Interaction, +2.2% on STAR average with 3.0X speed up, ours 2-frames out-perform SeViLA 4-frames on the VLEP dataset with 4.2X speed-up.

</details>

---

## 121. ShapeLLM: Universal 3D Object Understanding for Embodied Interaction

- [ ] ShapeLLM: Universal 3D Object Understanding for Embodied Interaction | https://eccv.ecva.net/virtual/2024/poster/2224

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2224

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents ShapeLLM, the first 3D Multimodal Large Language Model (LLM) designed for embodied interaction, exploring a universal 3D object understanding with 3D point clouds and languages. ShapeLLM is built upon an improved 3D encoder by extending ReCon to ReCon++ that benefits from multi-view image distillation for enhanced geometry understanding. By utilizing ReCon++ as the 3D point cloud input encoder for LLMs, ShapeLLM is trained on constructed instruction-following data and tested on our newly human-curated benchmark, 3D MM-Vet. ReCon++ and ShapeLLM achieve state-of-the-art performance in 3D geometry understanding and language–unified 3D interaction tasks, such as embodied visual grounding.

</details>

---

## 122. SegPoint: Segment Any Point Cloud via Large Language Model

- [ ] SegPoint: Segment Any Point Cloud via Large Language Model | https://eccv.ecva.net/virtual/2024/poster/2226

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2226

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite significant progress in 3D point cloud segmentation, existing methods primarily address specific tasks and depend on explicit instructions to identify targets, lacking the capability to infer and understand implicit user intentions in a unified framework. In this work, we propose a model, called SegPoint, that leverages the reasoning capabilities of a multi-modal Large Language Model (LLM) to produce point-wise segmentation masks across a diverse range of tasks: 1) 3D instruction segmentation, 2) 3D referring segmentation, 3) 3D semantic segmentation, and 4) 3D open-vocabulary semantic segmentation. To advance 3D instruction research, we introduce a new benchmark, Instruct3D, designed to evaluate segmentation performance from complex and implicit instructional texts, featuring 2,565 point cloud-instruction pairs. Our experimental results demonstrate that SegPoint achieves competitive performance on established benchmarks such as ScanRefer for referring segmentation and ScanNet for semantic segmentation, while delivering outstanding outcomes on the Instruct3D dataset. To our knowledge, SegPoint is the first model to address these varied segmentation tasks within a single framework, achieving satisfactory performance.

</details>

---

## 123. Reflective Instruction Tuning: Mitigating Hallucinations in Large Vision-Language Models

- [ ] Reflective Instruction Tuning: Mitigating Hallucinations in Large Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/2231

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2231

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have shown promising performance on a variety of vision-language tasks. However, they remain susceptible to hallucinations, generating outputs misaligned with visual content or instructions. While various mitigation strategies have been proposed, they often neglect a key contributor to hallucinations: lack of fine-grained reasoning supervision during training. Without intermediate reasoning steps, models may establish superficial shortcuts between instructions and responses, failing to internalize the inherent reasoning logic. To address this challenge, we propose reflective instruction tuning, which integrates rationale learning into visual instruction tuning. Unlike previous methods that learning from responses only, our approach entails the model predicting rationales justifying why responses are correct or incorrect. This fosters a deeper engagement with the fine-grained reasoning underlying each response, thus enhancing the model’s reasoning proficiency. To facilitate this approach, we propose REVERIE, the first large-scale instruction-tuning dataset with ReflEctiVE RatIonalE annotations. REVERIE comprises 115k machine-generated reasoning instructions, each meticulously annotated with a corresponding pair of correct and confusing responses, alongside comprehensive rationales elucidating the justification behind the correctness or erroneousness of each response. Experimental results on multiple LVLM benchmarks reveal that reflective instruction tuning with the REVERIE dataset yields noticeable performance gain over the baseline model, demonstrating the effectiveness of reflecting from the rationales. Project page is at https://zjr2000.github.io/projects/reverie

</details>

---

## 124. BLINK: Multimodal Large Language Models Can See but Not Perceive

- [ ] BLINK: Multimodal Large Language Models Can See but Not Perceive | https://eccv.ecva.net/virtual/2024/poster/2230

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2230

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce BLINK, a new benchmark for multimodal language models (LLMs) that focuses on core visual perception abilities not found in other evaluations. Most of the BLINK tasks can be solved by humans ``within a blink'' (e.g., depth estimation, correspondence, forensics detection, and multi-view reasoning). However, we find these perception-demanding tasks cast significant challenges for current multimodal LLMs because they resist mediation through natural language. BLINK reformats 14 classic computer vision tasks into 3,978 multiple-choice questions, paired with single or multiple images and visual prompting. While humans get 95.70% accuracy on average, BLINK is surprisingly challenging for existing multimodal LLMs: even the best-performing GPT-4V and Gemini achieve accuracies of 51.32% and 45.46%, only 13.23% and 7.47% higher than random guessing, indicating that such perception abilities have not "emerged" yet in recent multimodal LLMs. Our analysis also highlights that specialist CV models could solve these problems much better, suggesting potential pathways for future improvements. We believe BLINK will stimulate the community to help multimodal LLMs catch up with human-level perception.

</details>

---

## 125. Teach CLIP to Develop a Number Sense for Ordinal Regression

- [ ] Teach CLIP to Develop a Number Sense for Ordinal Regression | https://eccv.ecva.net/virtual/2024/poster/2232

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2232

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Ordinal regression is a fundamental problem within the field of computer vision, with customised well-trained models on specific tasks. While pre-trained vision-language models (VLMs) have exhibited impressive performance on various vision tasks, their potential for ordinal regression has received less exploration. In this study, we first investigate CLIP's potential for ordinal regression, from which we expect the model could generalise to different ordinal regression tasks and scenarios. Unfortunately, vanilla CLIP fails on this task, since current VLMs have a well-documented limitation of encapsulating compositional concepts such as number sense. We propose a simple yet effective method called NumCLIP to improve the quantitative understanding of VLMs. We disassemble the exact image to number-specific text matching problem into coarse classification and fine prediction stages. We discretize and phrase each numerical bin with common language concept to better leverage the available pre-trained alignment in CLIP. To consider the inherent continuous property of ordinal regression, we propose a novel fine-grained cross-modal ranking-based regularisation loss specifically designed to keep both semantic and ordinal alignment in CLIP's feature space. Experimental results on three general ordinal regression tasks demonstrate the effectiveness of NumCLIP, with 10% and 3.83% accuracy improvement on historical image dating and image aesthetics assessment task, respectively.

</details>

---

## 126. Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models

- [ ] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/2237

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2237

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Most Large Vision-Language Models (LVLMs) enjoy the same vision vocabulary, i.e., CLIP, for common vision tasks. However, for some special task that needs dense and fine-grained perception, the CLIP-style vocabulary may encounter low efficiency in tokenizing corresponding vision knowledge and even suffer out-of-vocabulary problems. Accordingly, we propose Vary, an efficient and productive method to scale up the  Vision vocabulary of LVLMs. The procedures of Vary are naturally divided into two folds: the generation and integration of a new vision vocabulary. In the first phase, we devise a vocabulary network along with a tiny decoder-only transformer to compress rich vision signals. In the next, we scale up the vanilla vision vocabulary by merging the new with the original one (CLIP), enabling the LVLMs can effectively garner new features. We present frameworks with two sizes: Vary-base (7B) and Vary-toy (1.8B), both of which enjoy excellent fine-grained perception performance while maintaining great general ability.

</details>

---

## 127. DOCCI: Descriptions of Connected and Contrasting Images

- [ ] DOCCI: Descriptions of Connected and Contrasting Images | https://eccv.ecva.net/virtual/2024/poster/2240

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2240

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language datasets are vital for both text-to-image (T2I) and image-to-text (I2T) research. However, current datasets lack descriptions with fine-grained detail that would allow for richer associations to be learned by models. To fill the gap, we introduce Descriptions of Connected and Contrasting Images (DOCCI), a dataset with long, human-annotated English descriptions for 15k images that were taken, curated and donated by a single researcher intent on capturing key challenges such as spatial relations, counting, text rendering, world knowledge, and more. We instruct human annotators to create comprehensive descriptions for each image; these average 136 words in length and are crafted to clearly distinguish each image from those that are related or similar. Each description is highly compositional and typically encompasses multiple challenges. Through both quantitative and qualitative analyses, we demonstrate that DOCCI serves as an effective training resource for image-to-text generation -- a PaLI 5B model finetuned on DOCCI shows equal or superior results compared to highly-performant larger models like LLaVA-1.5 7B and InstructBLIP 7B. Furthermore, we show that DOCCI is a useful testbed for text-to-image generation, highlighting the limitations of current text-to-image models in capturing long descriptions and fine details.

</details>

---

## 128. CLIP-DPO: Vision-Language Models as a Source of Preference for Fixing Hallucinations in LVLMs

- [ ] CLIP-DPO: Vision-Language Models as a Source of Preference for Fixing Hallucinations in LVLMs | https://eccv.ecva.net/virtual/2024/poster/2238

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2238

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present CLIP-DPO, a preference optimization method that leverages pretrained V-L (Vision-Language) embeddings models, such as CLIP, for DPO-based optimization of Vision LLMs. Starting from the initial pool of supervised fine-tuning data, we generate a diverse set of predictions, which are then ranked based on their CLIP image-text similarities to obtain a set of positive and negative pairs for DPO-based training. We show that this simple approach offers notable performance gains over a diverse set of benchmarks and vision-language tasks.

</details>

---

## 129. Conceptual Codebook Learning for Vision-Language Models

- [ ] Conceptual Codebook Learning for Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/2245

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2245

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose Conceptual Codebook Learning (CoCoLe), a novel fine-tuning method for vision-language models (VLMs) to address the challenge of improving the generalization capability of VLMs while fine-tuning them on downstream tasks in a few-shot setting. We recognize that visual concepts, such as textures, shapes, and colors are naturally transferable across domains and play a crucial role in generalization tasks. Motivated by this interesting finding, we learn a conceptual codebook consisting of visual concepts as keys and conceptual prompts as values, which serves as a link between the image encoder's outputs and the text encoder's inputs. Specifically, for a given image, we leverage the codebook to identify the most relevant conceptual prompts associated with the class embeddings to perform the classification. Additionally, we incorporate a handcrafted concept cache as a regularization to alleviate the overfitting issues in low-shot scenarios. We observe that this conceptual codebook learning method is able to achieve enhanced alignment between visual and linguistic modalities. Extensive experimental results demonstrate that our CoCoLe method remarkably outperforms the existing state-of-the-art methods across various evaluation settings, including base-to-new generalization, cross-dataset evaluation, and domain generalization tasks. Detailed ablation studies further confirm the efficacy of each component in CoCoLe.

</details>

---

## 130. Discovering Unwritten Visual Classifiers with Large Language Models

- [ ] Discovering Unwritten Visual Classifiers with Large Language Models | https://eccv.ecva.net/virtual/2024/poster/2250

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2250

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal pre-trained models, such as CLIP, are popular for zero-shot classification due to their open-vocabulary flexibility and high performance. However, vision-language models, which compute similarity scores between images and class labels, are largely black-box, with limited interpretability, risk for bias, and inability to discover new visual concepts not written down. Moreover, in practical settings, the vocabulary for class names and attributes of specialized concepts will not be known, preventing these methods from performing well on images uncommon in large-scale vision-language datasets. To address these limitations, we present a novel method that discovers interpretable yet discriminative sets of attributes for visual recognition. We introduce an evolutionary search algorithm that utilizes a large language model and its in-context learning abilities to iteratively mutate a concept bottleneck of attributes for classification. Our method produces state-of-the-art, interpretable fine-grained classifiers. We outperform the latest baselines by 18.4% on five fine-grained iNaturalist datasets and by 22.2% on two KikiBouba datasets, despite the baselines having access to privileged information about class names.

</details>

---

## 131. DetToolChain: A New Prompting Paradigm to Unleash Detection Ability of MLLM

- [ ] DetToolChain: A New Prompting Paradigm to Unleash Detection Ability of MLLM | https://eccv.ecva.net/virtual/2024/poster/2251

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2251

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present DetToolChain, a novel prompting paradigm, to unleash the zero-shot object detection ability of multimodal large language models (MLLMs), such as GPT-4V and Gemini. Our approach consists of a detection prompting toolkit inspired by high-precision detection priors and a new Chain-of-Thought to implement these prompts. Specifically, the prompts in the toolkit are designed to guide the MLLM to focus on regional information (e.g, zooming in), read coordinates according to measure standards (e.g., overlaying rulers and compasses), and infer from the contextual information (e.g., overlaying scene graphs). Building upon these tools, the new detection chain-of-thought can automatically decompose the task into simple subtasks, diagnose the predictions, and plan for progressive box refinements. The effectiveness of our framework is demonstrated across a spectrum of detection tasks, especially hard cases. Compared to existing state-of-the-art methods, GPT-4V with our DetToolChain improves state-of-the-art object detectors by +21.5% AP50 on MS COCO Novel class set for open-vocabulary detection, +24.23% Acc on RefCOCO val set for referring expression comprehension, +14.5% AP on D-cube describe object detection FULL setting. The codes shall be released upon acceptance.

</details>

---

## 132. Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs

- [ ] Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs | https://eccv.ecva.net/virtual/2024/poster/2248

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2248

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt ensembling of Large Language Model (LLM) generated category-specific prompts has emerged as an effective method to enhance zero-shot recognition ability of Vision-Language Models (VLMs). To obtain these category-specific prompts, the present methods rely on hand-crafting the prompts to the LLMs for generating VLM prompts for the downstream tasks. However, this requires manually composing these task-specific prompts and still, they might not cover the diverse set of visual concepts and task-specific styles associated with the categories of interest. To effectively take humans out of the loop and completely automate the prompt generation process for zero-shot recognition, we propose Meta-Prompting for Visual Recognition (MPVR). Taking as input only minimal information about the target task, in the form of its short natural language description, and a list of associated class labels, MPVR automatically produces a diverse set of category-specific prompts resulting in a strong zero-shot classifier. MPVR generalizes effectively across various popular zero-shot image recognition benchmarks belonging to widely different domains when tested with multiple LLMs and VLMs. For example, MPVR obtains a zero-shot recognition improvement over CLIP by up to 19.8% and 18.2% (5.0% and 4.5% on average over 20 datasets) leveraging GPT and Mixtral LLMs, respectively.

</details>

---

## 133. LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction

- [ ] LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction | https://eccv.ecva.net/virtual/2024/poster/2252

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2252

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing methods enhance open-vocabulary object detection by leveraging the robust open-vocabulary recognition capabilities of Vision-Language Models (VLMs), such as CLIP.  However, two main challenges emerge:  (1) A deficiency in concept representation, where the category names in CLIP's text space lack textual and visual knowledge. (2) An overfitting tendency towards base categories, with the open vocabulary knowledge biased towards base categories during the transfer from VLMs to detectors. To address these challenges, we propose the Language Model Instruction (LaMI) strategy, which leverages the relationships between visual concepts and applies them within a simple yet effective DETR-like detector, termed LaMI-DETR.  LaMI utilizes GPT to construct visual concepts and employs T5 to investigate visual similarities across categories.  These inter-category relationships refine concept representation and avoid overfitting to base categories. Comprehensive experiments validate our approach's superior performance over existing methods in the same rigorous setting without reliance on external training resources. LaMI-DETR achieves a rare box AP of 43.4 on OV-LVIS, surpassing the previous best by 7.8 rare box AP.

</details>

---

## 134. SILC: Improving Vision Language Pretraining with Self-Distillation

- [ ] SILC: Improving Vision Language Pretraining with Self-Distillation | https://eccv.ecva.net/virtual/2024/poster/2257

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2257

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image-Text pretraining on web-scale image caption datasets has become the default recipe for open vocabulary classification and retrieval models thanks to the success of CLIP and its variants. Several works have also used CLIP features for dense prediction tasks and have shown the emergence of open-set abilities. However, the contrastive objective used by these models only focuses on image-text alignment and does not incentivise image feature learning for dense prediction tasks. In this work, we introduce SILC, a novel framework for vision language pretraining. SILC improves image-text contrastive learning with the simple addition of local-to-global correspondence learning by self-distillation. We show that distilling local image features from an exponential moving average (EMA) teacher model significantly improves model performance on dense predictions tasks like detection and segmentation, while also providing improvements on image-level tasks such as classification and retrieval. SILC models sets a new state of the art for zero-shot classification, few shot classification, image and text retrieval, zero-shot segmentation, and open vocabulary segmentation. We further show that SILC features greatly benefit open vocabulary detection, captioning and visual question answering.

</details>

---

## 135. CoPT: Unsupervised Domain Adaptive Segmentation using Domain-Agnostic Text Embeddings

- [ ] CoPT: Unsupervised Domain Adaptive Segmentation using Domain-Agnostic Text Embeddings | https://eccv.ecva.net/virtual/2024/poster/2261

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2261

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Unsupervised domain adaptation (UDA) involves learning class semantics from labeled data within a source domain that generalize to an unseen target domain. UDA methods are particularly impactful for semantic segmentation, where annotations are more difficult to collect than in image classification. Despite recent advances in large-scale vision-language representation learning, UDA methods for segmentation have not taken advantage of the domain-agnostic properties of text. To address this, we present a novel covariance-based pixel-text loss, CoPT, that uses domain-agnostic text embeddings to learn domain-invariant features in an image segmentation encoder. The text embeddings are generated through our LLM Domain Template process, where an LLM is used to generate source and target domain descriptions that are combined and fed to a frozen CLIP model. In experiments on GTA$\rightarrow$Cityscapes and Synthia$\rightarrow$Cityscapes, we show that a model trained using CoPT achieves the new state of the art performance on UDA for segmentation.

</details>

---

## 136. 3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation

- [ ] 3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation | https://eccv.ecva.net/virtual/2024/poster/2264

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2264

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

3D panoptic segmentation is a challenging perception task, especially in autonomous driving. It aims to predict both semantic and instance annotations for 3D points in a scene. Although prior 3D panoptic segmentation approaches have achieved great performance on closed-set benchmarks, generalizing these approaches to unseen things and unseen stuff categories remains an open problem. For unseen object categories, 2D open-vocabulary segmentation has achieved promising results that solely rely on frozen CLIP backbones and ensembling multiple classification outputs. However, we find that simply extending these 2D models to 3D does not guarantee good performance due to poor per-mask classification quality, especially for novel stuff categories. In this paper, we propose the first method to tackle 3D open-vocabulary panoptic segmentation. Our model takes advantage of the fusion between learnable LiDAR features and dense frozen vision CLIP features, using a single classification head to make predictions for both base and novel classes. To further improve the classification performance on novel classes and leverage the CLIP model, we propose two novel loss functions: object-level distillation loss and voxel-level distillation loss. Our experiments on the nuScenes and SemanticKITTI datasets show that our method outperforms the strong baseline by a large margin.

</details>

---

## 137. Improving Zero-Shot Generalization for CLIP with Variational Adapter

- [ ] Improving Zero-Shot Generalization for CLIP with Variational Adapter | https://eccv.ecva.net/virtual/2024/poster/2303

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2303

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Thanks to the excellent generalization capability of pre-trained Vision-Language Models (VLMs) such as CLIP, fine-tuning VLMs for downstream tasks (e.g., zero-shot generalization) has become a popular choice. Despite achieving promising performance in the professionality of base classes, most existing fine-tuned methods suffer from feature confusion of novel classes, resulting in unsatisfactory transferability. To address this problem, we propose a divide-and-conquer approach called Prompt-based Variational Adapter (PVA) that explicitly reduce the prediction bias by separating base and novel samples. Specifically, we design two variational adapters with learnable textual tokens to align latent representations for each modalities in a shared latent space. Once trained, we can separate novel samples from entangled space using the similarity metric of latent features i.e., converting confusion task into two independent ones (One for base classes and the other for novel classes). Moreover, to improve the transferability for novel classes, we further refine the output features of the learned adapters with the global features via a residual connection. To the best of our knowledge, this is the first framework which combines prompt learning and adapter tuning to tackle the feature confusion issue. We conduct extensive experiments on GZSL and Cross-Dataset Transfer Learning to demonstrate the superiority of our approach and establish a new state-of-the-art on four popular benchmarks.

</details>

---

## 138. Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models

- [ ] Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models | https://eccv.ecva.net/virtual/2024/poster/2343

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2343

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we study the harmlessness alignment problem of multimodal large language models (MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate (ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.

</details>

---

## 139. Language-Driven Physics-Based Scene Synthesis and Editing via Feature Splatting

- [ ] Language-Driven Physics-Based Scene Synthesis and Editing via Feature Splatting | https://eccv.ecva.net/virtual/2024/poster/2413

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2413

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Scene representations using 3D Gaussian primitives have produced excellent results in modeling the appearance of static and dynamic 3D scenes. Many graphics applications, however, demand the ability to manipulate both the appearance and the physical properties of objects. We introduce Feature Splatting, an approach that unifies physics-based dynamic scene synthesis with rich semantics from vision language foundation models that are grounded by natural language. Our first contribution is a way to distill high-quality, object-centric vision-language features into 3D Gaussians, that enables semi-automatic scene decomposition using text queries. Our second contribution is a way to synthesize physics-based dynamics from an otherwise static scene using a particle-based simulator, in which material properties are assigned automatically via text queries. We ablate key techniques used in this pipeline, to illustrate the challenge and opportunities in using feature-carrying 3D Gaussians as a unified format for appearance, geometry, material properties and semantics grounded on natural language.

</details>

---

## 140. Visual Text Generation in the Wild

- [ ] Visual Text Generation in the Wild | https://eccv.ecva.net/virtual/2024/poster/2508

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2508

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, with the rapid advancements of generative models, the field of visual text generation has witnessed significant progress. However, it is still challenging to render high-quality text images in real-world scenarios, as three critical criteria should be satisfied: (1) Fidelity:  the generated text images should be photo-realistic and the contents are expected to be the same as specified in the given conditions; (2) Reasonability: the regions and contents of the generated text should cohere with the scene; (3) Utility: the generated text images can facilitate related tasks (e.g., text detection and recognition). Upon investigation, we find that existing methods, either rendering-based or diffusion-based, can hardly meet all these aspects simultaneously, limiting their application range. Therefore, we propose in this paper a visual text generator (termed SceneVTG), which can produce high-quality text images in the wild. Following a two-stage paradigm, SceneVTG leverages a Multimodal Large Language Model to recommend reasonable text regions and contents across multiple scales and levels, which are used by a conditional diffusion model as conditions to generate text images. Extensive experiments demonstrate that the proposed SceneVTG significantly outperforms traditional rendering-based methods and recent diffusion-based methods in terms of fidelity and reasonability. Besides, the images generated by SceneVTG provide superior utility for tasks involving text detection and text recognition. Code and datasets are available at AdvancedLiterateMachinery.

</details>

---

## 141. Free-ATM: Harnessing Free Attention Masks for Representation Learning on Diffusion-Generated Images

- [ ] Free-ATM: Harnessing Free Attention Masks for Representation Learning on Diffusion-Generated Images | https://eccv.ecva.net/virtual/2024/poster/2519

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2519

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper studies visual representation learning with diffusion-generated synthetic images. We start by uncovering that diffusion models' cross-attention layers inherently provide annotation-free attention masks aligned with corresponding text inputs on generated images. We then investigate the problems of three prevalent representation learning methods i.e., contrastive learning, masked modeling, and vision-language pretraining) on diffusion-generated synthetic data and introduce customized solutions by fully exploiting the aforementioned free attention masks, namely Free-ATM. Comprehensive experiments demonstrate Free-ATM's ability to enhance the performance of various representation learning frameworks when utilizing synthetic data. This improvement is consistent across diverse downstream tasks including image classification, detection, segmentation and image-text retrieval. Meanwhile, by utilizing Free-ATM, we can accelerate the pretraining on synthetic images significantly  and close the performance gap between representation learning on synthetic data and real-world scenarios.

</details>

---

## 142. AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion

- [ ] AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion | https://eccv.ecva.net/virtual/2024/poster/2528

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2528

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present AutoDIR, an innovative all-in-one image restoration system incorporating latent diffusion. AutoDIR excels in its ability to automatically identify and restore images suffering from a range of unknown degradations. AutoDIR offers intuitive open-vocabulary image editing, empowering users to customize and enhance images according to their preferences. AutoDIR consists of two key stages: a Blind Image Quality Assessment (BIQA) stage based on a semantic-agnostic vision-language model which automatically detects unknown image degradations for input images, an All-in-One Image Restoration (AIR) stage utilizes structural-corrected latent diffusion which handles multiple types of image degradations. Extensive experimental evaluation demonstrates that AutoDIR outperforms state-of-the-art approaches for a wider range of image restoration tasks. The design of AutoDIR also enables flexible user control (via text prompt) and generalization to new tasks as a foundation model of image restoration.

</details>

---

## 143. TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing

- [ ] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing | https://eccv.ecva.net/virtual/2024/poster/2550

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2550

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generalizable Face anti-spoofing (FAS) approaches have recently garnered considerable attention due to their robustness in unseen scenarios. Some recent methods incorporate vision-language models into FAS, leveraging their impressive pre-trained performance to improve the generalization. However, these methods only utilize coarse-grained or single-element prompts for fine-tuning FAS tasks, without fully exploring the potential of language supervision, leading to unsatisfactory generalization ability. To address these concerns, we propose a novel framework called TF-FAS, which aims to thoroughly explore and harness twofold-element fine-grained semantic guidance to enhance generalization. Specifically, the Content Element Decoupling Module (CEDM) is proposed to comprehensively explore the semantic elements related to content. It is subsequently employed to supervise the decoupling of categorical features from content-related features, thereby enhancing the generalization abilities. Moreover, recognizing the subtle differences within the data of each class in FAS, we present a Fine-Grained Categorical Element Module (FCEM) to explore fine-grained categorical element guidance, then adaptively integrate them to facilitate the distribution modeling for each class. Comprehensive experiments and analysis demonstrate the superiority of our method over state-of-the-art competitors.

</details>

---

## 144. PALM: Predicting Actions through Language Models

- [ ] PALM: Predicting Actions through Language Models | https://eccv.ecva.net/virtual/2024/poster/2568

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2568

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding human activity is a crucial yet intricate task in egocentric vision, a field that focuses on capturing visual perspectives from the camera wearer's viewpoint. Traditional methods heavily rely on representation learning that is trained on a large amount of video data. However, a major challenge arises from the difficulty of obtaining effective video representation. This difficulty stems from the complex and variable nature of human activities, which contrasts with the limited availability of data. In this study, we introduce PALM, an approach that excels in tackling the complex challenges associated with long-term video understanding without the need for extensive training. Specifically, we focus on the task of long-term action anticipation, which aims to forecast forthcoming sequences of actions over an extended period. Our method PALM incorporates an action recognition model to track previous action sequences and a vision-language model to articulate relevant environmental details. By leveraging the context provided by these past events, we devise a prompting strategy for action anticipation using large language models (LLMs). Moreover, we implement maximal marginal relevance for example selection to facilitate in-context learning of the LLMs. Our experimental results demonstrate that PALM surpasses the state-of-the-art methods in the task of long-term action anticipation on the Ego4D benchmark. We further validate PALM on two additional benchmarks, affirming its capacity for generalization across intricate activities with different sets of taxonomies.

</details>

---

## 145. LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models

- [ ] LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models | https://eccv.ecva.net/virtual/2024/poster/2576

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2576

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we present a novel method to tackle the token generation challenge in Vision Language Models (VLMs) for video and image understanding, called LLaMA-VID.  Current VLMs, while proficient in tasks like image captioning and visual question answering, face computational burdens when processing long videos due to the excessive visual tokens.   LLaMA-VID addresses this issue by representing each frame with two distinct tokens, namely context token and content token.  The context token encodes the overall image context based on user input, whereas the content token encapsulates visual cues in each frame. This dual-token strategy significantly reduces the overload of long videos while preserving critical information. Generally, LLaMA-VID empowers existing frameworks to support hour-long videos and pushes their upper limit with an extra context token. It is demonstrated to surpass previous methods on most of video- or image-based benchmarks. Code and models will be released to the public.

</details>

---

## 146. VISA: Reasoning Video Object Segmentation via Large Language Model

- [ ] VISA: Reasoning Video Object Segmentation via Large Language Model | https://eccv.ecva.net/virtual/2024/poster/2575

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2575

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing Video Object Segmentation(VOS) relies on explicit user instructions, such as categories, masks, or short phrases, restricting their ability to perform complex video segmentation requiring reasoning with world knowledge. In this paper, we introduce a new task, Reasoning Video Object Segmentation(ReasonVOS). This task aims to generate a sequence of segmentation masks in response to implicit text queries that require complex reasoning abilities based on world knowledge and video contexts, which is crucial for structured environment understanding and object-centric interactions, pivotal in the development of embodied AI. To tackle ReasonVOS, we introduce VISA(Video-based large language Instructed Segmentation Assistant), to leverage the world knowledge reasoning capabilities of multi-modal LLMs while possessing the ability to segment and track objects in videos with a mask decoder. Moreover, we establish a comprehensive benchmark consisting of 12,709 instruction-mask sequence pairs from 1,038 diverse videos, which incorporates complex world knowledge reasoning into segmentation tasks for instruction-tuning and evaluation purposes of ReasonVOS models. Experiments conducted on 8 datasets demonstrate the effectiveness of VISA in tackling complex reasoning segmentation and vanilla referring segmentation in both video and image domains. The code and dataset are available at https://anonymous.4open.science/r/VISA-36D6.

</details>

---

## 147. COM Kitchens: An Unedited Overhead-view Procedural Videos Dataset a Vision-Language Benchmark

- [ ] COM Kitchens: An Unedited Overhead-view Procedural Videos Dataset a Vision-Language Benchmark | https://eccv.ecva.net/virtual/2024/poster/2578

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2578

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Procedural video understanding is gaining attention in the vision and language community. Deep learning-based video analysis requires extensive data. Consequently, existing works often use web videos as training resources, making it challenging to query contents from raw video observations. To address this issue, we propose a new dataset, COM Kitchens. The dataset consists of unedited overhead-view videos captured by smartphones, in which participants performed food preparation based on given recipes. Fixed-viewpoint video datasets often lack environmental diversity due to high camera setup costs. We used modern wide-angle smartphone lenses to cover cooking counters from sink to cooktop in an overhead view, capturing activity without in-person assistance. With this setup, we collected a diverse dataset by distributing smartphones to participants. With this dataset, we propose the novel video-to-text retrieval task, Online Recipe Retrieval (OnRR), and new video captioning domain, Dense Video Captioning on unedited Overhead-View videos (DVC-OV). Our experiments verified the capabilities and limitations of current web-video-based SOTA methods in handling these tasks.

</details>

---

## 148. TrajPrompt: Aligning Color Trajectory with Vision-Language Representations

- [ ] TrajPrompt: Aligning Color Trajectory with Vision-Language Representations | https://eccv.ecva.net/virtual/2024/poster/2582

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2582

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Cross-modal learning has shown promising potential to overcome the limitations of single-modality tasks. However, without a proper design of representation alignment between different data sources, the external modality has no way to exhibit its value. We find that recent trajectory prediction approaches use Bird's-Eye-View (BEV) scene as additional source, but do not significantly improve the performance compared to the single-source strategies. This indicates that the representation of BEV scene and trajectory is not effectively combined. To overcome this problem, we propose TrajPrompt, a prompt-based approach that seamlessly incorporates trajectory representation into the vision-language framework, i.e. CLIP, for BEV scene understanding and future forecasting. We discover that CLIP can attend to the local area of BEV scene by utilizing our innovative design of text prompt and colored lines. Comprehensive results demonstrate TrajPrompt's effectiveness via outperforming the state-of-the-art trajectory predictors by a significant margin (over 35% improvement for ADE and FDE metrics on SDD and DroneCrowd dataset), using fewer learnable parameters than the previous trajectory modeling approaches with scene information included.

</details>

---

## 149. Meerkat: Audio-Visual Large Language Model for Grounding in Space and Time

- [ ] Meerkat: Audio-Visual Large Language Model for Grounding in Space and Time | https://eccv.ecva.net/virtual/2024/poster/2580

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2580

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Leveraging Large Language Models’ remarkable proficiency in text-based tasks, recent works on Multimodal-LLMs (MLLMs) extend them to other modalities like vision and audio. However, the progress in these directions has been mostly focused on tasks that only require a coarse-grained understanding of the audio-visual semantics. We present Meerkat, an audio-visual LLM equipped with a fine-grained understanding of image and audio both spatially and temporally. With a new modality alignment module based on optimal transport and a cross-attention module that enforces audio-visual consistency, Meerkat can tackle challenging tasks such as audio referred visual grounding, image guided audio temporal localization, and audio-visual fact-checking. Moreover, we carefully curate a large dataset AVFIT that comprises 3M instruction tuning samples collected from open-source datasets, and introduce MeerkatBench that unifies five challenging audio-visual tasks. We achieve state-of-the-art performance on all these downstream tasks with a relative improvement of up to 37.12%.

</details>

---

## 150. Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving

- [ ] Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving | https://eccv.ecva.net/virtual/2024/poster/2587

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2587

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) have garnered increasing interest in autonomous driving areas, due to their advanced capabilities in complex reasoning tasks essential for highly autonomous vehicle behavior. Despite their potential, research in autonomous systems is hindered by the lack of datasets with annotated reasoning chains that explain the decision-making processes in driving. To bridge this gap, we present Reason2Drive, a benchmark dataset with over 600K video-text pairs, aimed at facilitating the study of interpretable reasoning in complex driving environments. We distinctly characterize the autonomous driving process as a sequential combination of perception, prediction, and reasoning steps, and the question-answer pairs are automatically collected from a diverse range of open-source outdoor driving datasets, including nuScenes, Waymo and ONCE. Moreover, we introduce a novel aggregated evaluation metric to assess chain-based reasoning performance in autonomous systems, addressing the reasoning ambiguities of existing metrics such as BLEU and CIDEr. Based on the proposed benchmark, we conduct experiments to assess various existing VLMs, revealing insights into their reasoning capabilities. Additionally, we develop an efficient approach to empower VLMs to leverage object-level perceptual elements in both feature extraction and prediction, further enhancing their reasoning accuracy. Extendable experiments demonstrate the supportive effect of Reason2Drive towards visual reasoning and downstream planning tasks. The code and dataset will be released.

</details>

---

## 151. Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts

- [ ] Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts | https://eccv.ecva.net/virtual/2024/poster/2588

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2588

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

For a general-purpose robot to operate in reality, executing a broad range of instructions across various environments is imperative. Central to the reinforcement learning and planning for such robotic agents is a generalizable reward function. Recent advances in vision-language models, such as CLIP, have shown remarkable performance in the domain of deep learning, paving the way for open-domain visual recognition. However, collecting data on robots executing various language instructions across multiple environments remains a challenge. This paper aims to transfer video-language models with robust generalization into a generalizable language-conditioned reward function, only utilizing robot video data from a minimal amount of tasks in a singular environment. Unlike common robotic datasets to train reward functions, human video-language datasets seldom include trivial failure videos. To enhance the model's ability to discriminate between successful and failed robot executions, we cluster failure data with the aspiration that the model identifies patterns in failure videos. For each cluster, we incorporate a newly trained failure prompt into the text encoder to bolster its performance in distinguishing failure from success in robot task executions. Our language-conditioned reward function shows outstanding generalization to new environments and new instructions for robot planning and reinforcement learning.

</details>

---

## 152. LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents

- [ ] LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents | https://eccv.ecva.net/virtual/2024/poster/2590

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2590

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce LLaVA-Plus, an end-to-end training approach to systematically expanding the capabilities of large multimodal models (LMM), towards building general-purpose multimodal agents. It maintains a skill repository that contains a wide range of vision and vision-language pre-trained models as multimodal tools. Based on the user instruction and input image, LMM is trained to activate the appropriated tools when needed, grasping skills on the fly and aggregating the tool execution results to complete the real-world tasks in the wild. To facilitate the model capability on learning to use skills, we make the first attempt to build multimodal instruction-following data for tool use, covering skills in visual understanding, generation, external knowledge and their compositions. Empirical results show that LLaVA-Plus outperforms LLaVA in existing capabilities, and extends many new capabilities. Compared with large language model (LLM) based tool use methods, LLaVA-Plus is distinct in that the query image is considered throughout the entire interaction process, yielding higher multimodal tool use performance and enabling new scenarios.

</details>

---

## 153. Agent3D-Zero:  An Agent for Zero-shot 3D Understanding

- [ ] Agent3D-Zero:  An Agent for Zero-shot 3D Understanding | https://eccv.ecva.net/virtual/2024/poster/2592

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2592

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability to understand and reason the 3D real world is a crucial milestone towards artificial general intelligence. The current common practice is to finetune Large Language Models (LLMs) with 3D data and texts to enable 3D understanding. Despite their effectiveness, these approaches are inherently limited by the scale and diversity of the available 3D data. Alternatively, in this work, we introduce Agent3D-Zero, an innovative 3D-aware agent framework addressing the 3D scene understanding in a zero-shot manner. The essence of our approach centers on reconceptualizing the challenge of 3D scene perception as a process of understanding and synthesizing insights from multiple images, inspired by how our human beings attempt to understand 3D scenes. By consolidating this idea, we propose a novel way to make use of a Large Visual Language Model (VLM) via actively selecting and analyzing a series of viewpoints for 3D understanding.  Specifically, given an input 3D scene, Agent3D-Zero first processes a bird's-eye view image with custom-designed visual prompts, then iteratively chooses the next viewpoints to observe and summarize the underlying knowledge. A distinctive advantage of Agent3D-Zero is the introduction of novel visual prompts, which significantly unleash the VLMs' ability to identify the most informative viewpoints and thus facilitate observing 3D scenes.  Extensive experiments demonstrate the effectiveness of the proposed framework in understanding diverse and previously unseen 3D environments.

</details>

---

## 154. Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning

- [ ] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning | https://eccv.ecva.net/virtual/2024/poster/2597

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2597

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the remarkable success of large vision-language models (LVLMs) on various tasks, their susceptibility to knowledge bias inherited from training data hinders their ability to generalize to new scenarios and limits their real-world applicability. To address this challenge, we propose the Counterfactual Bias-Robust Reasoning (CoBRa) dataset that tackles knowledge bias by offering a novel collection of VQA examples designed to evaluate and mitigate bias in LVLMs. These examples encourage counterfactual thinking by providing edited knowledge graphs and image contents, with detailed annotations of reasoning processes to facilitate a comprehensive understanding of the examples. Based on the dataset, we introduce a Chain of Counterfactual Thought (CoCT) method that learns the bias-robust reasoning processes and provides in-context examples demonstrating how existing reasoning generalizes to counterfactual scenarios. This enables LVLMs to explicitly reason step-by-step rather than relying on biased knowledge, leading to more generalizable solutions. Our extensive evaluation demonstrates that CoCT outperforms existing approaches on tasks requiring reasoning under knowledge bias. Our work is available at https://shorturl.at/GOR45.

</details>

---

## 155. BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-language Models

- [ ] BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-language Models | https://eccv.ecva.net/virtual/2024/poster/2598

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2598

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision language models (LVLMs) perceive the world through a combination of a visual encoder and large language models (LLMs). The visual encoder, pre-trained on large-scale vision-text datasets, provides zero-shot generalization to visual data, and LLMs endow the high reasoning ability to LVLMs. It leads LVLMs to achieve high performance on wide benchmarks without fine-tuning, known as zero or few-shot capability of LLMs. However, recent studies show that LVLMs are vulnerable to hallucination. This undesirable behavior degrades reliability and credibility, thereby making users unable to fully trust the output from LVLMs. To enhance trustworthiness and better tackle the hallucination of LVLMs, we curate a new evaluation dataset, called the BEfore-AFter hallucination dataset (BEAF), and introduce new metrics: True Understanding (TU), IGnorance (IG), StuBbornness (SB), and InDecision (ID). Unlike prior works that focus only on constructing questions and answers, the key idea of our benchmark is that we manipulate visual scene information by image editing models and design the metrics based on scene changes. This allows us to clearly assess whether LVLMs correctly understand a given scene by observing the ability to perceive changes. We also visualize the correctness heatmap by virtue of our two-axis view: vision and text. Upon evaluating LVLMs with our dataset, we observed that our metrics can reveal different aspects of LVLM hallucination.

</details>

---

## 156. SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant

- [ ] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant | https://eccv.ecva.net/virtual/2024/poster/2596

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2596

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in the vision-language model have shown notable generalization in vision-language tasks after visual instruction tuning. However, bridging the gap between the pre-trained vision encoder and the large language models becomes the whole network's bottleneck. To improve cross-modality alignment, existing works usually consider more visual instruction data covering a broader range of vision tasks to fine-tune the model for question-answering, which are costly to obtain. However, the image contains rich contextual information that has been largely under-explored. This paper first attempts to harness this overlooked context within visual instruction data, training the model to self-supervised `learning' how to ask high-quality questions. In this way, we introduce a novel framework named SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant. SQ-LLaVA exhibits proficiency in generating flexible and meaningful image-related questions while analyzing the visual clue and prior language knowledge, signifying an advanced level of generalized visual understanding. Moreover, fine-tuning SQ-LLaVA on higher-quality instruction data shows a consistent performance improvement compared with traditional visual-instruction tuning methods. This improvement highlights the efficacy of self-questioning techniques in achieving a deeper and more nuanced comprehension of visual content across various contexts.

</details>

---

## 157. Paying More Attention to Images: A Training-Free Method for Alleviating Hallucination in LVLMs

- [ ] Paying More Attention to Images: A Training-Free Method for Alleviating Hallucination in LVLMs | https://eccv.ecva.net/virtual/2024/poster/2599

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2599

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) align image features to the input of Large Language Models (LLMs), enhancing multi-modal reasoning and knowledge utilization capabilities.  However, the disparity in scale between models of different modalities has resulted in LLMs assuming a predominant role in multimodal comprehension. This imbalance in model integration can lead to instances of hallucinatory outputs.  In particular, LVLMs may generate descriptions that persist in the absence of visual input, suggesting that these narratives are disproportionately influenced by the textual context.  We refer to this phenomenon as ``text inertia.'' To counteract this issue, we introduce a training-free algorithm designed to find an equilibrium between image comprehension and language inference.  Specifically, we firstly involve adjusting and amplifying the attention weights assigned to image tokens, thereby granting greater prominence to visual elements. Meanwhile, we subtract the logits of multimodal inputs from the model logits of pure text input, which can let model not biased towards only LLM. By enhancing images tokens and reducing the stubborn output of LLM, we can let LVLM pay more attention to images, towards alleviating text inertia and reducing the hallucination in LVLMs. Our extensive experiments shows that this method substantially reduces the frequency of hallucinatory outputs in various LVLMs in terms of different metrics.

</details>

---

## 158. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks

- [ ] Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks | https://eccv.ecva.net/virtual/2024/poster/2601

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2601

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages CLIP to enhance the transferability of adversarial perturbations generated within a generative model-based attack framework. Specifically, we exploit the joint vision-language space to formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the input ground truth. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.

</details>

---

## 159. TrojVLM: Backdoor Attack Against Vision Language Models

- [ ] TrojVLM: Backdoor Attack Against Vision Language Models | https://eccv.ecva.net/virtual/2024/poster/2600

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2600

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The emergence of Vision Language Models (VLMs) is a significant advancement in integrating computer vision with Large Language Models (LLMs) to produce detailed text descriptions based on visual inputs, yet it introduces new security vulnerabilities. Unlike prior work that centered on single modalities or classification tasks, this study introduces TrojVLM, the first exploration of backdoor attacks aimed at VLMs engaged in complex image-to-text generation.Specifically, TrojVLM inserts predetermined target text into output text when encountering poisoned images.  Moreover, a novel semantic preserving loss is proposed to ensure the semantic integrity of the original image content. Our evaluation on image captioning and visual question answering (VQA) tasks confirms the effectiveness of TrojVLM in maintaining original semantic content while triggering specific target text outputs. This study not only uncovers a critical security risk in VLMs and image-to-text generation but also sets a foundation for future research on securing multimodal models against such sophisticated threats.

</details>

---

## 160. Attention Prompting on Image for Large Vision-Language Models

- [ ] Attention Prompting on Image for Large Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/2603

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2603

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Compared with Large Language Models (LLMs), Large Vision-Language Models (LVLMs) can also accept images as input, thus showcasing more interesting emergent capabilities and demonstrating impressive performance on various vision-language tasks. Motivated by text prompting in LLMs, visual prompting has been explored to enhance LVLMs' capabilities of perceiving visual information. However, previous visual prompting techniques solely process visual inputs without considering text queries, limiting the models' ability to follow text instructions to complete tasks. To fill this gap, in this work, we propose a new prompting technique named Attention Prompting on Image (API), which just simply overlays a text-query-guided attention heatmap on the original input image and effectively enhances LVLM on various tasks. Specifically, we generate an attention heatmap for the input image dependent on the text query with an auxiliary model like CLIP. Then the heatmap simply multiplies the pixel values of the original image to obtain the actual input image for the LVLM. Extensive experiments on various vison-language benchmarks verify the effectiveness of our technique. For example, API improves LLaVA-1.5 by 3.8% and 2.9% on MM-Vet and LLaVA-Wild benchmarks, respectively.

</details>

---

## 161. LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model

- [ ] LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model | https://eccv.ecva.net/virtual/2024/poster/2604

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2604

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The revolutionary capabilities of large language models (LLMs) have paved the way for multimodal large language models (MLLMs) and fostered diverse applications across various specialized domains. In the remote sensing (RS) field, however, the diverse geographical landscapes and varied objects in RS imagery are not adequately considered in recent MLLM endeavors. To bridge this gap, we construct a large-scale RS image-text dataset, LHRS-Align, and an informative RS-specific instruction dataset, LHRS-Instruct, leveraging the extensive volunteered geographic information (VGI) and globally available RS images. Building on this foundation, we introduce LHRS-Bot, an MLLM tailored for RS image understanding through a novel multi-level vision-language alignment strategy and a curriculum learning method. Additionally, we introduce LHRS-Bench, a benchmark for thoroughly evaluating MLLMs’ abilities in RS image understanding. Comprehensive experiments demonstrate that LHRS-Bot exhibits a profound understanding of RS images and the ability to perform nuanced reasoning within the RS domain.

</details>

---

## 162. Generalizing to Unseen Domains via Text-guided Augmentation

- [ ] Generalizing to Unseen Domains via Text-guided Augmentation | https://eccv.ecva.net/virtual/2024/poster/2605

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2605

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

To avoid the high cost of collecting visual data from all test domains in the domain adaptation task, recent work takes advantage of the pre-trained large-scale vision language models,  such as CLIP, and augments training data with only text descriptions (e.g.,``a photo/painting/sketch...'') of each test domain. However, in many real-world applications, such text information of test domains is not always available in advance. Moreover, even if we can verbalize all test domains, it is laborious for existing work (Dunlap et al., 2023) to train a different augmentation network for each possible unseen domain, which suffers from time-inefficiency. To overcome these challenges, we benefit from the multimodal embedding space of a pre-trained vision-language model and propose to acquire training-free and domain-invariant augmentations with text descriptions of arbitrary crafted unseen domains, which not necessarily match test domains. Beyond achieving state-of-the-art results, compared with existing works that require trainable augmentation networks, our approach is also notably more time-efficient, and exhibits a more solid theoretical support. Code will be publicly available.

</details>

---

## 163. MoMA: Multimodal LLM Adapter for Fast Personalized Image Generation

- [ ] MoMA: Multimodal LLM Adapter for Fast Personalized Image Generation | https://eccv.ecva.net/virtual/2024/poster/2606

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2606

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present MoMA: an open-vocabulary, training-free personalized image model that boasts flexible zero-shot capabilities. As foundational text-to-image models rapidly evolve, the demand for robust image-to-image translation grows. Addressing this need, MoMA specializes in subject-driven personalized image generation. Utilizing an open-source, Multimodal Large Language Model (MLLM), we train MoMA to serve a dual role as both a feature extractor and a generator. This approach effectively synergizes reference image and text prompt information to produce valuable image features, facilitating an image diffusion model. To better leverage the generated features, we further introduce a novel self-attention shortcut method that efficiently transfers image features to an image diffusion model, improving the resemblance of the target object in generated images. Remarkably, as a tuning-free plug-and-play module, our model requires only a single reference image and outperforms existing methods in generating images with high detail fidelity, enhanced identity-preservation and prompt faithfulness. We commit to making our work open-source, thereby providing universal access to these advancements.

</details>

---

## 164. Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection

- [ ] Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection | https://eccv.ecva.net/virtual/2024/poster/2615

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2615

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language models (VLMs) such as CLIP have shown promise in Out-of-distribution (OOD) detection through their generalizable multimodal representations. Existing CLIP-based OOD detection methods only utilize a single modality of in-distribution (ID) information (e.g., textual cues). However, we find that the ID visual information helps to leverage CLIP's full potential for OOD detection. In this paper, we pursue a different approach and explore the regime to leverage both the visual and textual ID information. Specifically, we propose Dual-Pattern Matching (DPM), efficiently adapting CLIP for OOD detection by leveraging both textual and visual ID patterns. DPM stores ID class-wise text features as the textual pattern and the aggregated ID visual information as the visual pattern. At test time, the similarity to both patterns is computed to detect OOD inputs. We further extend DPM with lightweight adaptation for enhanced OOD detection. Experiments demonstrate DPM's advantages, outperforming existing methods on common benchmarks. The dual-pattern approach provides a simple yet effective way to exploit multi-modality for OOD detection with vision-language representations.

</details>

---

## 165. APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension

- [ ] APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension | https://eccv.ecva.net/virtual/2024/poster/2620

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2620

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Referring Expression Comprehension (REC) aims to ground the target object based on a given referring expression, which requires expensive instance-level annotations for training. To address this issue, recent advances explore an efficient one-stage weakly supervised REC model called RefCLIP. Particularly, RefCLIP utilizes anchor features of pre-trained one-stage detection networks to represent candidate objects and conducts anchor-text ranking to locate the referent. Despite the effectiveness, we identify that  visual semantics of RefCLIP are  ambiguous and insufficient for weakly supervised REC modeling. To address this issue, we propose a novel method that enriches visual semantics with various prompt information, called anchor-based prompt learning (APL). Specifically, APL contains an innovative anchor-based prompt encoder (APE) to produce discriminative prompts covering three aspects of REC modeling, e.g., position, color and category.   These prompts are dynamically fused into anchor features to improve the visual description power. In addition, we propose two novel auxiliary objectives to achieve  accurate vision-language alignment in APL, namely   text reconstruction loss and visual alignment loss.   To validate APL, we  conduct extensive experiments on four REC benchmarks, namely RefCOCO, RefCOCO+, RefCOCOg and ReferIt. Experimental results not only show the state-of-the-art performance of APL against existing methods on four benchmarks, e.g., +6.44% over RefCLIP on RefCOCO, but also confirm its strong generalization  ability on weakly supervised referring expression segmentation. Source codes are anonymously released at: https://anonymous.4open.science/r/APL-B297.

</details>

---

## 166. MTA-CLIP: Language-Guided Semantic Segmentation with Mask-Text Alignment

- [ ] MTA-CLIP: Language-Guided Semantic Segmentation with Mask-Text Alignment | https://eccv.ecva.net/virtual/2024/poster/2624

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2624

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent approaches have shown that large-scale vision-language models such as CLIP can improve semantic segmentation performance. These methods typically aim for pixel-level vision language alignment, but often rely on low-resolution image features from CLIP, resulting in class ambiguities along boundaries. Moreover, the global scene representations in CLIP text embeddings do not directly correlate with the local and detailed pixel-level features, making meaningful alignment more difficult. To address these limitations, we introduce MTA-CLIP, a novel framework employing mask-level vision-language alignment. Specifically, we first propose Mask-Text Decoder that enhances the mask representations using rich textual data with the CLIP language model. Subsequently, it aligns mask representations with text embeddings using Mask-to-Text Contrastive Learning. Furthermore, we introduce Mask-Text Prompt Learning, utilizing multiple context-specific prompts for text embeddings to capture diverse class representations across masks. Overall, MTA-CLIP achieves state-of-the-art, surpassing prior works by an average of 2.8% and 1.3% on standard benchmark datasets, ADE20k and Cityscapes, respectively.

</details>

---

## 167. Cross-Domain Semantic Segmentation on Inconsistent Taxonomy using VLMs

- [ ] Cross-Domain Semantic Segmentation on Inconsistent Taxonomy using VLMs | https://eccv.ecva.net/virtual/2024/poster/2628

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2628

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The challenge of semantic segmentation in Unsupervised Domain Adaptation (UDA) emerges not only from domain shifts between source and target images but also from discrepancies in class taxonomies across domains. Traditional UDA research assumes consistent taxonomy between the source and target domains, thereby limiting their ability to recognize and adapt to the taxonomy of the target domain. This paper introduces a novel approach, Cross-Doamin Semantic Segmentation on Inconsistent Taxonomy using Vision Language Models (CSI), which effectively performs domain-adaptive semantic segmentation even in situations of source-target class mismatches. CSI leverages the semantic generalization potential of Visual Language Models (VLMs) to create synergy with previous UDA methods. It utilizes segment reasoning obtained through traditional UDA methods, alongside the rich semantic knowledge embedded in VLMs, to perform relabeling to classes of the target domain. This approach allows for effective adaptation to changed taxonomies without requiring any ground truth label for the target domain. Our method has shown to be effective across various benchmarks in situations of inconsistent taxonomy settings, such as coarse-to-fine taxonomy and open taxonomy, and demonstrates consistent synergy effects when integrated with previous state-of-the-art UDA methods.

</details>

---

## 168. Towards Multi-modal Transformers in Federated Learning

- [ ] Towards Multi-modal Transformers in Federated Learning | https://eccv.ecva.net/virtual/2024/poster/2665

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2665

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal transformers mark significant progress in different domains, but siloed high-quality data hinders their further improvement. To remedy this, federated learning (FL) has emerged as a promising privacy-preserving paradigm for training models without direct access to the raw data held by different clients. Despite its potential, a considerable research direction regarding the unpaired uni-modal clients and the transformer architecture in FL remains unexplored. To fill this gap, this paper explores a transfer multi-modal federated learning (MFL) scenario within the vision-language domain, where clients possess data of various modalities distributed across different datasets. We systematically evaluate the performance of existing methods when a transformer architecture is utilized and introduce a novel framework called Federated modality complementary and collaboration (FedCola) by addressing the in-modality and cross-modality gaps among clients. Through extensive experiments across various FL settings, FedCola demonstrates superior performance over previous approaches, offering new perspectives on future federated training of multi-modal transformers.

</details>

---

## 169. Soft Prompt Generation for Domain Generalization

- [ ] Soft Prompt Generation for Domain Generalization | https://eccv.ecva.net/virtual/2024/poster/2668

- **Link**: https://eccv.ecva.net/virtual/2024/poster/2668

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large pre-trained vision language models (VLMs) have shown impressive zero-shot ability on downstream tasks with manually designed prompt, which are not optimal for specific domains. To further adapt VLMs to downstream tasks, soft prompt is proposed to replace manually designed prompt, which acts as a learning vector that undergoes fine-tuning based on specific domain data. Prior prompt learning methods primarily learn a fixed prompt and residuled prompt from training samples. However, the learned prompts lack diversity and ignore information about unseen domains, potentially compromising the transferability of the prompts. In this paper, we reframe the prompt learning framework from a generative perspective and propose a simple yet efficient method for the Domain Generalization (DG) task, namely Soft Prompt Generation (SPG). To the best of our knowledge, we are the first to introduce the generative model into prompt learning in VLMs and explore its potential for producing soft prompts by relying solely on the generative model, ensuring the diversity of prompts. Specifically, SPG consists of a two-stage training phase and an inference phase. During the training phase, we introduce soft prompt labels for each domain, aiming to incorporate the generative model domain knowledge. During the inference phase, the generator of the generative model is employed to obtain instance-specific soft prompts for the unseen target domain. Extensive experiments on five domain generalization benchmarks of three DG tasks demonstrate that our proposed SPG achieves state-of-the-art performance. The code is available in supplementary materials.

</details>

---

## 170. Vision-Language Action Knowledge Learning for Semantic-Aware Action Quality Assessment

- [ ] Vision-Language Action Knowledge Learning for Semantic-Aware Action Quality Assessment | https://eccv.ecva.net/virtual/2024/poster/340

- **Link**: https://eccv.ecva.net/virtual/2024/poster/340

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Action quality assessment (AQA) is a challenging vision task that requires discerning and quantifying subtle differences in actions from the same class. While recent research has made strides in creating fine-grained annotations for more precise analysis, existing methods primarily focus on coarse action segmentation, leading to limited identification of discriminative action frames. To address this issue, we propose a Vision-Language Action Knowledge Learning approach for action quality assessment, along with a multi-grained alignment framework to understand different levels of action knowledge. In our framework, prior knowledge, such as specialized terminology, is embedded into video-level, stage-level, and frame-level representations via CLIP. We further propose a new semantic-aware collaborative attention module to prevent confusing interactions and preserve textual knowledge in cross-modal and cross-semantic spaces. Specifically, we leverage the powerful cross-modal knowledge of CLIP to embed textual semantics into image features, which then guide action spatial-temporal representations. Our approach can be plug-and-played with existing AQA methods, frame-wise annotations or not. Extensive experiments and ablation studies show that our approach achieves state-of-the-art on four public short and long-term AQA benchmarks: FineDiving, MTL-AQA, JIGSAWS, and Fis-V.

</details>

---

## 171. Goldfish: Vision-Language Understanding of Arbitrarily Long Videos

- [ ] Goldfish: Vision-Language Understanding of Arbitrarily Long Videos | https://eccv.ecva.net/virtual/2024/poster/345

- **Link**: https://eccv.ecva.net/virtual/2024/poster/345

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Most current LLM-based models for video understanding can process videos within minutes. However, they struggle with lengthy videos due to challenges such as “noise and redundancy", as well as “memory and computation" constraints. In this paper, we present Goldfish, a methodology tailored for comprehending videos of arbitrary lengths. We also introduce the TVQA-long benchmark, specifically designed to evaluate models’ capabilities in understanding long videos with questions in both vision and text content. Goldfish approaches these challenges with an efficient retrieval mechanism that initially gathers the top-k video clips relevant to the instruction before proceeding to provide the desired response. This design of the retrieval mechanism enables the Goldfish to efficiently process arbitrarily long video sequences, facilitating its application in contexts such as movies or television series. To facilitate the retrieval process, we developed MiniGPT4-Video that generates detailed descriptions for the video clips. In addressing the scarcity of benchmarks for long video evaluation, we adapted the TVQA short video benchmark for extended content analysis by aggregating questions from entire episodes, thereby shifting the evaluation from partial to full episode comprehension. We attained a 41.78% accuracy rate on the TVQA-long benchmark, surpassing previous methods by 14.94%. Our MiniGPT4-Video also shows exceptional performance in short video comprehension, exceeding existing state-of-the-art methods by 3.23%, 2.03%, 16.5% and 23.59% on the MSVD, MSRVTT, TGIF, and TVQA short video benchmarks, respectively. These results indicate that our models have significant improvements in both long and short-video understanding.

</details>

---

## 172. Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning

- [ ] Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning | https://eccv.ecva.net/virtual/2024/poster/346

- **Link**: https://eccv.ecva.net/virtual/2024/poster/346

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Data quality stands at the forefront of deciding the effectiveness of video-language representation learning. However, video-text pairs in previous data typically do not align perfectly with each other, which might lead to video-language representations that do not accurately reflect cross-modal semantics. Moreover, previous data also possess an uneven distribution of concepts, thereby hampering the downstream performance across unpopular subjects. To address these problems, we propose a contrastive objective with a subtractive angular margin to regularize cross-modal representations in their effort to reach perfect similarity. Furthermore, to adapt to the non-uniform concept distribution, we propose a multi-layer perceptron (MLP)-parameterized weighting function that maps loss values to sample weights which enable dynamic adjustment of the model’s focus throughout the training. With the training guided by a small amount of unbiased meta-data and augmented by video-text data generated by large vision-language model, we improve video-language representations and achieve superior performances on commonly used video question answering and text-video retrieval datasets.

</details>

---

## 173. CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios

- [ ] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios | https://eccv.ecva.net/virtual/2024/poster/348

- **Link**: https://eccv.ecva.net/virtual/2024/poster/348

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper focuses on the challenge of answering questions in scenarios that are composed of rich and complex dynamic audio-visual components. Although existing Multimodal Large Language Models (MLLMs) can respond to audio-visual content, these responses are sometimes ambiguous and fail to describe specific audio-visual events. To overcome this limitation, we introduce the CAT, which enhances MLLM in three ways: 1) besides straightforwardly bridging audio and video, we design a clue aggregator that aggregates question-related clues in dynamic audio-visual scenarios to enrich the detailed knowledge required for large language models. 2) CAT is trained on a mixed multimodal dataset, allowing direct application in audio-visual scenarios. Notably, we collect an audio-visual joint instruction dataset named AVinstruct, to further enhance the capacity of CAT to model cross-semantic correlations. 3) we propose AI-assisted ambiguity-aware direct preference optimization, a strategy specialized in retraining the model to favor the non-ambiguity response and improve the ability to localize specific audio-visual objects. Extensive experimental results demonstrate that CAT outperforms existing methods on multimodal tasks, especially in Audio-Visual Question Answering (AVQA) tasks. The codes and the collected instructions will be released soon.

</details>

---

## 174. LingoQA: Video Question Answering for Autonomous Driving

- [ ] LingoQA: Video Question Answering for Autonomous Driving | https://eccv.ecva.net/virtual/2024/poster/355

- **Link**: https://eccv.ecva.net/virtual/2024/poster/355

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce LingoQA, a novel dataset and benchmark for video question answering in autonomous driving. The dataset contains 28K unique short video scenarios, and 419K annotations. Evaluating state-of-the-art vision-language models on our benchmark shows that their performance is below human capabilities, with GPT-4V responding truthfully to 56.67% of the questions compared to 93.4% for humans. For evaluation, in addition to conducting a human study, we propose a truthfulness classifier, called Lingo-Judge, that achieves a 0.95 Spearman correlation coefficient to human evaluations, surpassing existing techniques like METEOR, BLEU, CIDEr, and GPT-4. We establish a baseline vision-language model and run extensive ablation studies to understand its performance. We release our dataset and benchmark as an evaluation platform for vision-language models in autonomous driving.

</details>

---

## 175. Dolphins: Multimodal Language Model for Driving

- [ ] Dolphins: Multimodal Language Model for Driving | https://eccv.ecva.net/virtual/2024/poster/356

- **Link**: https://eccv.ecva.net/virtual/2024/poster/356

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The quest for fully autonomous vehicles (AVs) capable of navigating complex real-world scenarios with human-like understanding and responsiveness. In this paper, we introduce Dolphins, a novel vision-language model architected to imbibe human-like abilities as a conversational driving assistant. Dolphins is adept at processing multimodal inputs comprising video (or image) data, text instructions, and historical control signals to generate informed outputs corresponding to the provided instructions. Building upon the open-sourced pretrained Vision-Language Model, OpenFlamingo, we first enhance Dolphins's reasoning capabilities through an innovative Grounded Chain of Thought (GCoT) process in the general domain. Then we tailored Dolphins to the driving domain by constructing driving-specific instruction data and conducting instruction tuning. Through the utilization of the BDD-X dataset, we designed and consolidated four distinct AV tasks into Dolphins to foster a holistic understanding of intricate driving scenarios. As a result, the distinctive features of Dolphins are characterized into two dimensions: (1) the ability to provide a comprehensive understanding of complex and long-tailed open-world driving scenarios and solve a spectrum of AV tasks, and (2) the emergence of human-like capabilities including gradient-free instant adaptation via in-context learning and error recovery via reflection.

</details>

---

## 176. AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization

- [ ] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization | https://eccv.ecva.net/virtual/2024/poster/354

- **Link**: https://eccv.ecva.net/virtual/2024/poster/354

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this study, we introduce a new problem raised by social media and photojournalism, named Image Address Localization (IAL), which aims to predict the readable textual address where an image was taken. Existing two-stage approaches involve predicting geographical coordinates and converting them into human-readable addresses, which can lead to ambiguity and be resource-intensive. In contrast, we propose an end-to-end framework named AddressCLIP to solve the problem with more semantics, consisting of two key ingredients: i) image-text alignment to align images with addresses and scene captions by contrastive learning, and ii) image-geography matching to constrain image features with the spatial distance in terms of manifold learning. Additionally, we have built three datasets from Pittsburgh and San Francisco on different scales specifically for the IAL problem. Experiments demonstrate that our approach achieves compelling performance on the proposed datasets and outperforms representative transfer learning methods for vision-language models. Furthermore, extensive ablations and visualizations exhibit the effectiveness of the proposed method. The datasets and source code are available at https://github.com/xsx1001/AddressCLIP.

</details>

---

## 177. Depicting Beyond Scores: Advancing Image Quality Assessment through Multi-modal Language Models

- [ ] Depicting Beyond Scores: Advancing Image Quality Assessment through Multi-modal Language Models | https://eccv.ecva.net/virtual/2024/poster/363

- **Link**: https://eccv.ecva.net/virtual/2024/poster/363

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a Depicted image Quality Assessment method (DepictQA), overcoming the constraints of traditional score-based methods. DepictQA allows for detailed, language-based, human-like evaluation of image quality by leveraging Multi-modal Large Language Models (MLLMs). Unlike conventional Image Quality Assessment (IQA) methods relying on scores, DepictQA interprets image content and distortions descriptively and comparatively, aligning closely with humans' reasoning process. To build the DepictQA model, we establish a hierarchical task framework, and collect a multi-modal IQA training dataset. To tackle the challenges of limited training data and multi-image processing, we propose to use multi-source training data and specialized image tags. These designs result in a better performance of DepictQA than score-based approaches on multiple benchmarks. Moreover, compared with general MLLMs, DepictQA can generate more accurate reasoning descriptive languages. Our work shows the research potential of multi-modal IQA tasks. Codes and datasets are available in https://depictqa.github.io.

</details>

---

## 178. Visual Grounding for Object-Level Generalization in Reinforcement Learning

- [ ] Visual Grounding for Object-Level Generalization in Reinforcement Learning | https://eccv.ecva.net/virtual/2024/poster/359

- **Link**: https://eccv.ecva.net/virtual/2024/poster/359

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generalization is a pivotal challenge for agents following natural language instructions. To approach this goal, we leverage a vision-language model (VLM) for visual grounding and transfer its vision-language knowledge into reinforcement learning (RL) for object-centric tasks, which makes the agent capable of zero-shot generalization to unseen objects and instructions. By visual grounding, we obtain an object-grounded confidence map for the target object indicated in the instruction. Based on this map, we introduce two routes to transfer VLM knowledge into RL. Firstly, we propose an object-grounded intrinsic reward function derived from the confidence map to more effectively guide the agent towards the target object. Secondly, the confidence map offers a more unified, accessible task representation for the agent's policy, compared to language embeddings. This enables the agent to process unseen objects and instructions through comprehensible visual confidence maps, facilitating zero-shot object-level generalization. Single-task experiments prove that our intrinsic reward significantly improves performance on challenging skill learning. In multi-task experiments, through testing on tasks beyond the training set, we show that the agent, when provided with the confidence map as the task representation, possesses better generalization capabilities than language-based conditioning.

</details>

---

## 179. HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning

- [ ] HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning | https://eccv.ecva.net/virtual/2024/poster/364

- **Link**: https://eccv.ecva.net/virtual/2024/poster/364

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Hallucination has been a major problem for large language models and remains a critical challenge when it comes to multimodality in which vision-language models (VLMs) have to deal with not just textual but also visual inputs. Despite rapid progress in VLMs, resources for evaluating and addressing multimodal hallucination are limited and mostly focused on evaluation. This work introduces HaloQuest, a novel visual question answering dataset that captures various aspects of multimodal hallucination such as false premises, insufficient contexts, and visual challenges. A novel idea from HaloQuest is to leverage synthetic images, apart from real ones, to enable dataset creation at scale. With over 7.7K examples spanning across a wide variety of categories, HaloQuest was designed to be both a challenging  benchmark for VLMs and a fine-tuning dataset for advancing multimodal reasoning.  Our experiments reveal that current models struggle with HaloQuest, with all open-source VLMs achieving below 36\% accuracy.  On the other hand, fine-tuning on HaloQuest significantly reduces hallucination rates while preserving performance on standard reasoning tasks. Our results discover that benchmarking with generated images is highly correlated (r=0.97) with real images. Last but not least, we propose a novel Auto-Eval mechanism that is highly correlated with human raters (r=0.99) for evaluating VLMs.  In sum, this work makes concrete strides towards understanding, evaluating, and mitigating hallucination in VLMs, serving as an important step towards more reliable multimodal AI systems in the future.

</details>

---

## 180. REVISION: Rendering Tools Enable Spatial Fidelity in Vision-Language Models

- [ ] REVISION: Rendering Tools Enable Spatial Fidelity in Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/365

- **Link**: https://eccv.ecva.net/virtual/2024/poster/365

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid progression of Text-to-Image (T2I) and Multimodal Large Language Models (MLLMs) has resulted in their widespread adoption across multiple computer vision and natural language processing tasks. However, a common mode of failure that persists across both classes of models is their inability to correctly reason over spatial relationships. To tackle this shortcoming, we develop the REVISION framework which improves and evaluates spatial fidelity in vision-language models. REVISION is a 3D rendering based pipeline that generates spatially accurate synthetic images, given a textual prompt. REVISION is an extendable framework, which currently supports 101 3D assets, 11 spatial relationships, all with diverse camera perspectives and backgrounds. Leveraging images from REVISION as additional guidance in a training-free manner consistently improves the spatial consistency of T2I models across all spatial relationships, achieving competitive performance on the VISOR and T2I-CompBench benchmarks. We also introduce the REVISION benchmark to evaluate the spatial reasoning abilities of MLLMs, and find that state-of-the-art models are not robust to complex spatial reasoning under adversarial settings. Our results and findings indicate that utilizing rendering-based frameworks is an efficient approach for developing reasoning-aware generative models.

</details>

---

## 181. ViG-Bias: Visually Grounded Bias Discovery and Mitigation

- [ ] ViG-Bias: Visually Grounded Bias Discovery and Mitigation | https://eccv.ecva.net/virtual/2024/poster/366

- **Link**: https://eccv.ecva.net/virtual/2024/poster/366

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The proliferation of machine learning models in critical decision making processes has underscored the need for bias discovery and mitigation strategies. Identifying the reasons behind a biased system is not straightforward, since in many occasions they are associated with hidden spurious correlations which are not easy to spot. Standard approaches rely on bias audits performed by analyzing model performance in pre-defined subgroups of data samples, usually characterized by common attributes like gender or ethnicity when it comes to people, or other specific attributes defining semantically coherent groups of images. However, it is not always possible to know a-priori the specific attributes defining the failure modes of visual recognition systems. Recent approaches propose to discover these groups by leveraging large vision language models, which enable the extraction of cross-modal embeddings and the generation of textual descriptions to characterize the subgroups where a certain model is underperforming. In this work, we argue that incorporating visual explanations (e.g. heatmaps generated via GradCAM or other approaches) can boost the performance of such bias discovery and mitigation frameworks. To this end, we introduce Visually Grounded Bias Discovery and Mitigation (ViG-Bias), a simple yet effective technique which can be integrated to a variety of existing frameworks to improve both, discovery and mitigation performance. Our comprehensive evaluation shows that incorporating visual explanations enhances existing techniques like DOMINO, FACTS and Bias-to-Text, across several challenging datasets, including CelebA, Waterbirds, and NICO++.

</details>

---

## 182. GENIXER: Empowering Multimodal Large Language Models as a Powerful Data Generator

- [ ] GENIXER: Empowering Multimodal Large Language Models as a Powerful Data Generator | https://eccv.ecva.net/virtual/2024/poster/367

- **Link**: https://eccv.ecva.net/virtual/2024/poster/367

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) demonstrate exceptional problem-solving capabilities, but few research studies aim to gauge the ability to generate visual instruction tuning data. This paper proposes to explore the potential of empowering MLLMs to generate data independently without relying on GPT-4. We introduce Genixer, a comprehensive data generation pipeline consisting of four key steps: (i) instruction data collection, (ii) instruction template design, (iii) empowering MLLMs, and (iv) data generation and filtering.  Additionally, we outline two modes of data generation: task-agnostic and task-specific, enabling controllable output. We demonstrate that a synthetic VQA-like dataset trained with LLaVA1.5 enhances performance on 10 out of 12 multimodal benchmarks. Additionally, the grounding MLLM Shikra, when trained with a REC-like synthetic dataset, shows improvements on 7 out of 8 REC datasets. Through experiments and synthetic data analysis, our findings are: (1) current MLLMs can serve as robust data generators without assistance from GPT-4V; (2) MLLMs trained with task-specific datasets can surpass GPT-4V in generating complex instruction tuning data; (3) synthetic datasets enhance performance across various multimodal benchmarks and help mitigate model hallucinations.

</details>

---

## 183. Adversarial Prompt Tuning for Vision-Language Models

- [ ] Adversarial Prompt Tuning for Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/368

- **Link**: https://eccv.ecva.net/virtual/2024/poster/368

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code will be available upon publication of the paper.

</details>

---

## 184. MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training

- [ ] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training | https://eccv.ecva.net/virtual/2024/poster/369

- **Link**: https://eccv.ecva.net/virtual/2024/poster/369

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and comprehensive ablations of the image encoder, the vision-language connector, and various pre-training data choices, we identify performant components as well as design lessons. For example, we demonstrate that for large-scale multimodal pre-training using a careful mix of image-caption, interleaved image-text, and text-only data is crucial for achieving state-of-the-art (SOTA) few-shot results across multiple benchmarks, compared to other published pre-training results. Further, we show that the image encoder together with image resolution and the image token count has substantial impact, while the vision-language connector design is of comparatively negligible importance. By scaling up the presented recipe, we build MM1, a family of multimodal models up to 30B parameters, that are SOTA in pre-training metrics and achieve competitive performance after supervised fine-tuning on a range of established multimodal benchmarks. Thanks to large-scale pre-training, MM1 enjoys appealing properties such as enhanced in-context learning, and multi-image reasoning, enabling few-shot chain-of-thought prompting.

</details>

---

## 185. Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection

- [ ] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection | https://eccv.ecva.net/virtual/2024/poster/373

- **Link**: https://eccv.ecva.net/virtual/2024/poster/373

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language (VL) models have shown to be very effective in a variety of object detection tasks by utilizing weakly supervised image-text pairs from the web. However, these models exhibit a limited understanding of complex compositions of visual objects (e.g., attributes, shapes, and their relations), resulting in a significant performance drop given complex and diverse language queries. While conventional methods try to enhance VL models through the use of hard negative synthetic augmentation on the text domain, their effectiveness remains restricted without densely paired image-text augmentation. In this paper, we introduce a structured synthetic data generation approach to improve the compositional understanding of VL models for language-based object detection, which generates densely paired positive and negative triplets (object, text descriptions, bounding boxes) in both image and text domains. In addition, in order to train VL models effectively, we propose a new compositional contrastive learning formulation that discovers semantics and structures in complex descriptions from synthetic triplets. As a result, VL models trained with our synthetic data generation exhibit a significant performance boost in the Omnilabel benchmark by up to +5AP and the D3 benchmark by +6.9AP upon existing baselines.

</details>

---

## 186. FlexAttention for Efficient High-Resolution Vision-Language Models

- [ ] FlexAttention for Efficient High-Resolution Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/371

- **Link**: https://eccv.ecva.net/virtual/2024/poster/371

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current high-resolution vision-language models encode images as high-resolution image tokens and exhaustively take all these tokens to compute attention, which significantly increases the computational cost. To address this problem, we propose FlexAttention, a flexible attention mechanism for efficient high-resolution vision-language models. Specifically, a high-resolution image is encoded both as high-resolution tokens and low-resolution tokens, where only the low-resolution tokens and a few selected high-resolution tokens are utilized to calculate the attention map, which greatly shrinks the computational cost. The high-resolution tokens are selected via a high-resolution selection module which could retrieve tokens of relevant regions based on an input attention map. The selected high-resolution tokens are then concatenated to the low-resolution tokens and text tokens, and input to a hierarchical self-attention layer which produces an attention map that could be used for the next-step high-resolution token selection. The hierarchical self-attention process and high-resolution token selection process are performed iteratively for each attention layer. Experiments on multimodal benchmarks prove that our FlexAttention outperforms existing high-resolution VLMs (e.g., relatively ~9% in V* Bench, ~7% in TextVQA), while also significantly reducing the computational cost by nearly 40%.

</details>

---

## 187. Mismatch Quest: Visual and Textual Feedback for Image-Text Misalignment

- [ ] Mismatch Quest: Visual and Textual Feedback for Image-Text Misalignment | https://eccv.ecva.net/virtual/2024/poster/374

- **Link**: https://eccv.ecva.net/virtual/2024/poster/374

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While existing image-text alignment models reach high quality binary assessments, they fall short of pinpointing the exact source of misalignment. In this paper, we present a method to provide detailed textual and visual explanation of detected misalignments between text-image pairs.  We leverage large language models and visual grounding models to automatically construct a training set that holds plausible misaligned captions for a given image and corresponding textual explanations and visual indicators.  We also publish a new human curated test set comprising ground-truth textual and visual misalignment annotations. Empirical results show that fine-tuning vision language models on our training set enables them to articulate misalignments and visually indicate them within images, outperforming strong baselines both on the binary alignment classification and the explanation generation tasks.

</details>

---

## 188. CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts

- [ ] CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts | https://eccv.ecva.net/virtual/2024/poster/377

- **Link**: https://eccv.ecva.net/virtual/2024/poster/377

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastive vision-language models, such as CLIP, have garnered considerable attention for various dowmsteam tasks, mainly due to the remarkable ability of the learned features for generalization. However, the features they learned often blend content and style information, which somewhat limits their generalization capabilities under distribution shifts. To address this limitation, we adopt a causal generative perspective for multimodal data and propose contrastive learning with data augmentation to disentangle content features from the original representations. To achieve this, we begins with exploring image augmentation techniques and develop a method to seamlessly integrate them into pre-trained CLIP-like models to extract pure content features. Taking a step further, recognizing the inherent semantic richness and logical structure of text data, we explore the use of text augmentation to isolate latent content from style features. This enables CLIP-like model's encoders to concentrate on latent content information, refining the learned representations by pre-trained CLIP-like models. Our extensive experiments across diverse datasets demonstrate significant improvements in zero-shot and few-shot classification tasks, alongside enhanced robustness to various perturbations. These results underscore the effectiveness of our proposed methods in refining vision-language representations and advancing the state-of-the-art in multimodal learning.

</details>

---

## 189. Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning

- [ ] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning | https://eccv.ecva.net/virtual/2024/poster/378

- **Link**: https://eccv.ecva.net/virtual/2024/poster/378

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We address the challenges inherent in sketch-based image retrieval (SBIR) across various settings, including zero-shot SBIR, generalized zero-shot SBIR, and fine-grained zero-shot SBIR, by leveraging the vision-language foundation model, CLIP. While recent endeavors have employed CLIP to enhance SBIR, these approaches predominantly follow uni-modal prompt processing and overlook to fully exploit CLIP's integrated visual and textual capabilities. To bridge this gap, we introduce \textsc{SpLIP}, a novel multi-modal prompt learning scheme designed to operate effectively with frozen CLIP backbones. We diverge from existing multi-modal prompting methods that either treat visual and textual prompts independently or integrate them in a limited fashion, leading to suboptimal generalization. \textsc{SpLIP} implements a bi-directional prompt-sharing strategy that enables mutual knowledge exchange between CLIP's visual and textual encoders, fostering a more cohesive and synergistic prompt processing mechanism that significantly reduces the semantic gap between the sketch and photo embeddings. In addition to pioneering multi-modal prompt learning, we propose two innovative strategies for further refining the embedding space. The first is an adaptive margin generation for the sketch-photo triplet loss, regulated by CLIP's class textual embeddings. The second introduces a novel task, termed conditional cross-modal jigsaw, aimed at enhancing fine-grained sketch-photo alignment, by focusing on implicitly modelling the viable patch arrangement of sketches using knowledge of unshuffled photos. Our comprehensive experimental evaluations across multiple benchmarks demonstrate the superior performance of \textsc{SpLIP} in all three SBIR scenarios.

</details>

---

## 190. Open-Set Recognition in the Age of Vision-Language Models

- [ ] Open-Set Recognition in the Age of Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/394

- **Link**: https://eccv.ecva.net/virtual/2024/poster/394

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Are vision-language models (VLMs) open-set models because they are trained on internet-scale datasets? We answer this question with a clear no -- VLMs introduce closed-set assumptions via their finite query set, making them vulnerable to open-set conditions. We systematically evaluate VLMs for open-set recognition and find they frequently misclassify objects not contained in their query set, leading to alarmingly low precision when tuned for high recall and vice versa. We show that naively increasing the size of the query set to contain more and more classes does not mitigate this problem, but instead causes diminishing task performance and open-set performance. We establish a revised definition of the open-set problem for the age of VLMs, define a new benchmark and evaluation protocol to facilitate standardised evaluation and research in this important area, and evaluate promising baseline approaches based on predictive uncertainty and dedicated negative embeddings on a range of VLM classifiers and object detectors.

</details>

---

## 191. Unlocking Textual and Visual Wisdom: Open-Vocabulary 3D Object Detection Enhanced by Comprehensive Guidance from Text and Image

- [ ] Unlocking Textual and Visual Wisdom: Open-Vocabulary 3D Object Detection Enhanced by Comprehensive Guidance from Text and Image | https://eccv.ecva.net/virtual/2024/poster/398

- **Link**: https://eccv.ecva.net/virtual/2024/poster/398

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary 3D object detection (OV-3DDET) is a challenging task aimed at locating and recognizing objects in a 3D scene, encompassing both seen and previously unseen categories. Unlike in the vision and language domain where abundant training data is available to train generalized models, 3D detection models suffer from the scarcity of training data. Despite this challenge, the flourishing field of vision-language models (VLMs) offers valuable insights that can guide the learning process for OV-3DDET. While some efforts have been made to incorporate VLMs into OV-3DDET learning, existing methods often fall short in establishing a comprehensive association between 3D detectors and VLMs. In this paper, we investigate the utilization of VLMs for the task of open-vocabulary 3D detection. We use a vision model to guide novel class discovery in 3D scenes, and hierarchically align the 3D feature and vision-language feature space. Specifically, we employ an off-the-shelf 2D detector to seed and select novel 3D objects. The discovered novel objects are then stored for retraining the 3D detector. Finally, we align the 3D feature space with the vision-language feature space using a pre-trained vision-language model at the instance, category, and scene levels. Through extensive experimentation, we demonstrate substantial improvements in accuracy and generalization, underscoring the potential of VLMs in advancing 3D object detection for real-world applications.

</details>

---

## 192. OpenIns3D: Snap and Lookup for 3D Open-vocabulary Instance Segmentation

- [ ] OpenIns3D: Snap and Lookup for 3D Open-vocabulary Instance Segmentation | https://eccv.ecva.net/virtual/2024/poster/401

- **Link**: https://eccv.ecva.net/virtual/2024/poster/401

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we introduce OpenIns3D, a new framework for 3D open-vocabulary scene understanding at instance level. Unlike all existing methods, the proposed pipeline requires no well-aligned images as input and works effectively on a wide range of scenarios. The OpenIns3D framework employs a "Mask-Snap-Lookup" scheme, where the "Mask" module learns class-agnostic mask proposals in 3D point clouds, the "Snap" module generates synthetic scene-level images at multiple scales and leverages 2D vision language models to extract interesting objects, and the "Lookup" module searches through the outcomes of "Snap" to assign category names to the proposed masks. This approach, free from 2D input requirements yet simple and flexible, achieves state-of-the-art performance across a wide range of 3D open-vocabulary tasks, including recognition, object detection, and instance segmentation, on both indoor and outdoor datasets. Moreover, OpenIns3D facilitates effortless switching between different 2D detectors without requiring retraining. When integrated with powerful 2D open-world models, it achieves excellent results in scene understanding tasks. Furthermore, when combined with LLM-powered 2D models, OpenIns3D exhibits a remarkable capability to comprehend and process highly complex text queries that demand intricate reasoning and real-world knowledge.

</details>

---

## 193. 3D Weakly Supervised Semantic Segmentation with 2D Vision-Language Guidance

- [ ] 3D Weakly Supervised Semantic Segmentation with 2D Vision-Language Guidance | https://eccv.ecva.net/virtual/2024/poster/407

- **Link**: https://eccv.ecva.net/virtual/2024/poster/407

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

3D weakly supervised semantic segmentation aims to learn semantic segmentation without using dense annotations. Previous methods mostly use Class Activation Map to solve this challenge. In such a paradigm, the model is supervised given the scene-level or subcloud-level labels, however, remaining less-explored in the potential textually semantic information from the category labels. In this paper, we propose 3DSS-VLG, a weakly supervised approach for 3D Semantic Segmentation with 2D Vision-Language Guidance, an alternative approach that a 3D model predicts dense-embedding for each point which is co-embedded with both the aligned image and text spaces from the 2D vision-language model. Specifically, our method exploits the superior generalization ability of the 2D vision-language models and proposes Embeddings Soft-Guidance Stage to utilize it to implicitly align 3D embeddings and text embeddings. Moreover, we introduce Embeddings Specialization Stage to purify the feature representation with the help of given scene-level label, specifying better feature supervised by the corresponding text embedding. Thus, the 3D model is able to gain the informative supervisions both from the image embedding and text embedding, leading to competitive segmentation performances. To the best of our knowledge, this is the first work to investigate 3D weakly supervised semantic segmentation by using the textual semantic information of text category labels. Moreover, with extensive quantitative and qualitative experiments, we present that our 3DSS-VLG is able to not only achieve the state-of-the-art performance on both S3DIS and ScanNet dataset, but also maintain strong generalization capability.

</details>

---

## 194. Open-Vocabulary RGB-Thermal Semantic Segmentation

- [ ] Open-Vocabulary RGB-Thermal Semantic Segmentation | https://eccv.ecva.net/virtual/2024/poster/409

- **Link**: https://eccv.ecva.net/virtual/2024/poster/409

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

RGB-Thermal (RGB-T) semantic segmentation is an important research branch of multi-modal image segmentation. The current RGB-T semantic segmentation methods generally have two unsolved and typical shortcomings. First, they do not have the open-vocabulary recognition ability, which significantly limits their application scenarios. Second, when fusing RGB and thermal images, they often need to design complex fusion network structures, which usually results in low network training efficiency. We present OpenRSS, the Open-vocabulary RGB-T Semantic Segmentation method, to solve these two disadvantages. To our knowledge, OpenRSS is the first RGB-T semantic segmentation method with open-vocabulary segmentation capability. OpenRSS modifies the basic segmentation model SAM for RGB-T semantic segmentation by adding the proposed thermal information prompt module and dynamic low-rank adaptation strategy to SAM. These designs effectively fuse the RGB and thermal information, but with much fewer trainable parameters than other methods. OpenRSS achieves the open-vocabulary capability by jointly utilizing the vision-language model CLIP and the modified SAM. Through extensive experiments, OpenRSS demonstrates its effective open-vocabulary semantic segmentation ability on RGB-T images. It outperforms other state-of-the-art RGB open-vocabulary semantic segmentation methods on multiple RGB-T semantic segmentation benchmarks: +12.1% mIoU on the MFNet dataset, +18.4% mIoU on the MCubeS dataset, and +21.4% mIoU on the Freiburg Thermal dataset.

</details>

---

## 195. AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection

- [ ] AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection | https://eccv.ecva.net/virtual/2024/poster/434

- **Link**: https://eccv.ecva.net/virtual/2024/poster/434

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot anomaly detection (ZSAD) targets the identification of anomalies within images from arbitrary novel categories. This study introduces AdaCLIP for the ZSAD task, leveraging a pre-trained vision-language model (VLM), CLIP. AdaCLIP incorporates learnable prompts into CLIP and optimizes them through training on auxiliary annotated anomaly detection data. Two types of learnable prompts are proposed: \textit{static} and \textit{dynamic}. Static prompts are shared across all images, serving to preliminarily adapt CLIP for ZSAD. In contrast, dynamic prompts are generated for each test image, providing CLIP with dynamic adaptation capabilities. The combination of static and dynamic prompts is referred to as hybrid prompts, and yields enhanced ZSAD performance. Extensive experiments conducted across 14 real-world anomaly detection datasets from industrial and medical domains indicate that AdaCLIP outperforms other ZSAD methods and can generalize better to different categories and even domains. Finally, our analysis highlights the importance of diverse auxiliary data and optimized prompts for enhanced generalization capacity. Code is available at \texttt{removed for blind review}.

</details>

---

## 196. Robustness Preserving Fine-tuning using Neuron Importance

- [ ] Robustness Preserving Fine-tuning using Neuron Importance | https://eccv.ecva.net/virtual/2024/poster/449

- **Link**: https://eccv.ecva.net/virtual/2024/poster/449

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Robust fine-tuning aims to adapt a vision-language model to downstream tasks while preserving its zero-shot capabilities on unseen data.  Recent studies have introduced fine-tuning strategies to improve in-distribution (ID) performance on the downstream tasks while minimizing deterioration in out-of-distribution (OOD) performance on unseen data. This balance is achieved either by aligning the fine-tuned representations with the pre-trained ones or by constraining significant deviations in fine-tuned weights compared to the pre-trained model. In the latter approach, the regularization term is uniformly applied to all parameters. Our work proposes to selectively apply the regularization term based on the ``importance'' of each neuron to the fine-tuning dataset. To this end, we develop an importance-score metric to quantify each neurons’ importance to the downstream task and then leverage this to develop two fine-tuning strategies: importance-guided selective fine-tuning and importance-guided regularization. Our approach can be used concurrently with representation space-based methods, outperforming other approaches based on parameter space. We improve the state-of-the-art on standard robust fine-tuning benchmarks across datasets in both the full-shot and low-shot settings.

</details>

---

## 197. Online Zero-Shot Classification with CLIP

- [ ] Online Zero-Shot Classification with CLIP | https://eccv.ecva.net/virtual/2024/poster/450

- **Link**: https://eccv.ecva.net/virtual/2024/poster/450

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training such as CLIP enables zero-shot transfer that can classify images according to the candidate class names. While CLIP demonstrates an impressive zero-shot performance on diverse downstream tasks, the distribution from the target data has not been leveraged sufficiently. In this work, we study a novel online zero-shot transfer scenario, where each image arrives in a random order for classification and is visited only once to obtain prediction immediately without storing its representation. Compared with the vanilla zero-shot classification, the proposed problem preserves its flexibility for online service but considers the statistics of the arrived images as the side information to capture the distribution of target data better. To tackle the challenge of effective online optimization, we first develop online label learning to model the target data distribution. Then, the proxy of each class in the vision space can be further optimized with the proposed online proxy learning method to mitigate the modality gap between images and text. The convergence of both online strategies can be theoretically guaranteed. By combining the predicted label from the online label learning and proxy learning, our online zero-shot transfer method (OnZeta) can trade between bias and variance in predictions, which helps achieve $78.94\%$ accuracy on ImageNet without accessing the entire data set. Moreover, extensive experiments on other 13 downstream tasks with different vision encoders show a more than $3\%$ improvement on average, which demonstrates the effectiveness of our proposal.

</details>

---

## 198. LLMGA: Multimodal Large Language Model based Generation Assistant

- [ ] LLMGA: Multimodal Large Language Model based Generation Assistant | https://eccv.ecva.net/virtual/2024/poster/535

- **Link**: https://eccv.ecva.net/virtual/2024/poster/535

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce a Multimodal Large Language Model-based Generation Assistant (LLMGA), leveraging the vast reservoir of knowledge and proficiency in reasoning, comprehension, and response inherent in Large Language Models (LLMs) to assist users in image generation and editing. Diverging from existing approaches where Multimodal Large Language Models (MLLMs) generate fixed-size embeddings to control Stable Diffusion (SD), our LLMGA provides a detailed language generation prompt for precise control over SD. This not only augments LLM context understanding but also reduces noise in generation prompts, yields images with more intricate and precise content, and elevates the interpretability of the network. To this end, we curate a comprehensive dataset comprising prompt refinement, similar image generation, inpainting \& outpainting, and instruction-based editing. Moreover, we propose a two-stage training scheme. In the first stage, we train the MLLM to grasp the properties of image generation and editing, enabling it to generate detailed prompts. In the second stage, we optimize SD to align with the MLLM's generation prompts. Additionally, we propose a reference-based restoration network to alleviate texture, brightness, and contrast disparities between generated and preserved regions during inpainting and outpainting. Extensive results show that LLMGA has promising generation and editing capabilities and can enable more flexible and expansive applications. in an interactive manner.

</details>

---

## 199. Structured-NeRF: Hierarchical Scene Graph with Neural Representation

- [ ] Structured-NeRF: Hierarchical Scene Graph with Neural Representation | https://eccv.ecva.net/virtual/2024/poster/578

- **Link**: https://eccv.ecva.net/virtual/2024/poster/578

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present Structured Neural Radiance Field (Structured-NeRF) for indoor scene representaion based on a novel hierarchical scene graph structure to organize the neural radiance field. Existing object-centric methods focus only on the inherent characteristics of objects, while overlooking the semantic and physical relationships between them. Our scene graph is adept at managing the complex real-world correlation between objects within a scene, enabling functionality beyond novel view synthesis, such as scene re-arrangement. Based on the hierarchical structure, we introduce the optimization strategy based on semantic and physical relationships, thus simplifying the operations involved in scene editing and ensuring both efficiency and accuracy. Moreover, we conduct shadow rendering on objects to further intensify the realism of the rendered images. Experimental results demonstrate our structured representation not only achieves state-of-the-art (SOTA) performance in object-level and scene-level rendering, but also advances downstream applications in union with LLM/VLM, such as automatic and instruction/image conditioned scene re-arrangement, thereby extending the NeRF to interactive editing conveniently and controllably.

</details>

---

## 200. UniProcessor: A Text-induced Unified Low-level Image Processor

- [ ] UniProcessor: A Text-induced Unified Low-level Image Processor | https://eccv.ecva.net/virtual/2024/poster/685

- **Link**: https://eccv.ecva.net/virtual/2024/poster/685

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image processing, including image restoration, image enhancement, etc., involves generating a high-quality clean image from a degraded input. Deep learning-based methods have shown superior performance for various image processing tasks in terms of single-task conditions. However, they require to train separate models for different degradations and levels, which limits the generalization abilities of these models and restricts their applications in real-world. In this paper, we propose a text-induced Unified image Processor for low-level vision tasks, termed UniProcessor, which can effectively process various degradation types and levels, and support multimodal control. Specifically, our UniProcessor encodes degradation-specific information with the subject prompt and process degradations with the manipulation prompt. These context control features are injected into the UniProcessor backbone via cross-attention to control the processing procedure. For automatic subject-prompt generation, we further build a vision-language model for general-purpose low-level degradation perception via instruction tuning techniques. Our UniProcessor covers 30 degradation types, and extensive experiments demonstrate that our UniProcessor can well process these degradations without additional training or tuning and outperforms other competing methods. Moreover, with the help of degradation-aware context control, our UniProcessor first shows the ability to individually handle a single distortion in an image with multiple degradations.

</details>

---

## 201. Training-free Video Temporal Grounding using Large-scale Pre-trained Models

- [ ] Training-free Video Temporal Grounding using Large-scale Pre-trained Models | https://eccv.ecva.net/virtual/2024/poster/729

- **Link**: https://eccv.ecva.net/virtual/2024/poster/729

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video temporal grounding aims to identify video segments within untrimmed videos that are most relevant to a given natural language query. Existing video temporal localization models rely on specific datasets for training, with high data collection costs, but exhibit poor generalization capability under the across-dataset and out-of-distribution (OOD) settings. In this paper, we propose a Training-Free zero-shot Video Temporal Grounding (TFVTG) approach that leverages the ability of pre-trained large models. A naive baseline is to enumerate proposals in the video and use the pre-trained visual language models (VLMs) to select the best proposal according to the vision-language alignment. However, most existing VLMs are trained on image-text pairs or trimmed video clip-text pairs, making it struggle to (1) grasp the relationship and distinguish the temporal boundaries of multiple events within the same video; (2) comprehend and be sensitive to the dynamic transition of events (the transition from one event to another) in the video.  To address these issues, firstly, we propose leveraging large language models (LLMs) to analyze multiple sub-events contained in the query text and analyze the temporal order and relationships between these events.  Secondly, we split a sub-event into dynamic transition and static status parts and propose the dynamic and static scoring functions using VLMs to better evaluate the relevance between the event and the description. Finally, for each sub-event description provided by LLMs, we use VLMs to locate the top-k proposals that are most relevant to the description and leverage the order and relationships between sub-events provided by LLMs to filter and integrate these proposals.  Our method achieves the best performance on zero-shot video temporal grounding on Charades-STA and ActivityNet Captions datasets without any training and demonstrates better generalization capabilities in cross-dataset and OOD settings.

</details>

---

## 202. FunQA: Towards Surprising Video Comprehension

- [ ] FunQA: Towards Surprising Video Comprehension | https://eccv.ecva.net/virtual/2024/poster/732

- **Link**: https://eccv.ecva.net/virtual/2024/poster/732

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Surprising videos, e.g., funny clips, creative performances, or visual illusions, attract significant attention. Enjoyment of these videos is not simply a response to visual stimuli; rather, it hinges on the human capacity to understand (and appreciate) commonsense violations depicted in these videos. We introduce FunQA, a challenging video question-answering (QA) dataset specifically designed to evaluate and enhance the depth of video reasoning based on counter-intuitive and fun videos. Unlike most video QA clips, spanning a total of 24 video hours. Moreover, we propose FunMentor, an agent designed for Vision-Language Models (VLMs) that uses multi-turn dialogues to enhance models’ understanding of counter-intuitiveness. Extensive experiments with existing VLMs demonstrate the effectiveness of FunMentor and reveal significant performance gaps for the FunQA videos across spatial-temporal reasoning, visual-centered reasoning, and free-text generation.

</details>

---

## 203. Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach

- [ ] Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach | https://eccv.ecva.net/virtual/2024/poster/738

- **Link**: https://eccv.ecva.net/virtual/2024/poster/738

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we construct a large-scale benchmark dataset for Ground-to-Aerial Video-based person Re-Identification, named G2A-VReID, which comprises 185,907 images and 5,576 tracklets, featuring 2,788 distinct identities. To our knowledge, this is the first dataset for video ReID under Ground-to-Aerial scenarios. G2A-VReID dataset has the following characteristics: 1) Drastic view changes; 2) Large number of annotated identities; 3) Rich outdoor scenarios; 4) Huge difference in resolution. Additionally, we propose a new benchmark approach for cross-platform ReID by transforming the cross-platform visual alignment problem into visual-semantic alignment through vision-language model (i.e., CLIP) and applying a parameter-efficient Video Set-Level-Adapter module to adapt image-based foundation model to video ReID tasks, termed VSLA-CLIP. Besides, to further reduce the great discrepancy across the platforms, we also devise the platform-bridge prompts for efficient visual feature alignment. Extensive experiments demonstrate the superiority of the proposed method on all existing video ReID datasets and our proposed G2A-VReID dataset.

</details>

---

## 204. NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models

- [ ] NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/745

- **Link**: https://eccv.ecva.net/virtual/2024/poster/745

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Capitalizing on the remarkable advancements in Large Language Models (LLMs), there is a burgeoning initiative to harness LLMs for instruction following robotic navigation. Such a trend underscores the potential of LLMs to generalize navigational reasoning and diverse language understanding. However, a significant discrepancy in agent performance is observed when integrating LLMs in the Vision-and-Language navigation (VLN) tasks compared to previous downstream specialist models. Furthermore, the inherent capacity of language to interpret and facilitate communication in agent interactions is often underutilized in these integrations. In this work, we strive to bridge the divide between VLN-specialized models and LLM-based navigation paradigms, while maintaining the interpretative prowess of LLMs in generating linguistic navigational reasoning. By aligning visual content in a frozen LLM, we encompass visual observation comprehension for LLMs and exploit a way to incorporate LLMs and navigation policy networks for effective action predictions and navigational reasoning. We demonstrate the data efficiency of the proposed methods and eliminate the gap between LM-based agents and state-of-the-art VLN specialists.

</details>

---

## 205. INTRA: Interaction Relationship-aware Weakly Supervised Affordance Grounding

- [ ] INTRA: Interaction Relationship-aware Weakly Supervised Affordance Grounding | https://eccv.ecva.net/virtual/2024/poster/747

- **Link**: https://eccv.ecva.net/virtual/2024/poster/747

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Affordance denotes the potential interactions inherent in objects. The perception of affordance can enable intelligent agents to navigate and interact with new environments efficiently. Weakly supervised affordance grounding teaches agents the concept of affordance without costly pixel-level annotations, but with exocentric images. Although recent advances in weakly supervised affordance grounding yielded promising results, there remain challenges including the requirement for paired exocentric and egocentric image dataset, and the complexity in grounding diverse affordances for a single object. To address them, we propose  INTeraction Relationship-aware weakly supervised Affordance grounding (INTRA). Unlike prior arts, INTRA recasts this problem as representation learning to identify unique features of interactions through contrastive learning with exocentric images only, eliminating the need for paired datasets. Moreover, we leverage vision-language model embeddings for performing affordance grounding flexibly with any text, designing text-conditioned affordance map generation to reflect interaction relationship for contrastive learning and enhancing robustness with our text synonym augmentation. Our method outperformed prior arts on diverse datasets such as AGD20K, IIT-AFF, CAD and UMD. Additionally, experimental results demonstrate that our method has remarkable domain scalability for synthesized images / illustrations and is capable of performing affordance grounding for novel interactions and objects.

</details>

---

## 206. Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs

- [ ] Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs | https://eccv.ecva.net/virtual/2024/poster/749

- **Link**: https://eccv.ecva.net/virtual/2024/poster/749

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent advancements in multimodal large language models (MLLMs) have been noteworthy, yet, these general-domain MLLMs often fall short in their ability to comprehend and interact effectively with user interface (UI) screens. In this paper, we construct Ferret-UI, a new MLLM tailored for enhanced understanding of mobile UI screens, equipped with referring, grounding, and reasoning capabilities. we meticulously gathered training samples from an extensive range of fundamental UI tasks, such as icon recognition, find text, and widget listing. These samples are formatted for instruction-following with region annotations to facilitate precise referring and grounding. Moreover, to augment the model's reasoning ability, we compile a dataset for advanced tasks inspired by Ferret, but with a focus on mobile screens. This methodology enables the training of Ferret-UI, a model that exhibits outstanding comprehension of UI screens and the ability to execute open-ended instructions, thereby facilitating UI operations. To rigorously evaluate its capabilities, we establish a comprehensive benchmark encompassing the aforementioned tasks. Ferret-UI not only outstrips most open-source UI MLLMs in performance but also achieves parity with GPT-4V, marking a significant advancement in the field.

</details>

---

## 207. SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding

- [ ] SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding | https://eccv.ecva.net/virtual/2024/poster/748

- **Link**: https://eccv.ecva.net/virtual/2024/poster/748

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

3D vision-language grounding, which aims to align language with 3D physical environments, stands as a cornerstone in developing embodied agents. In comparison to recent advancements in the 2D domain, grounding language in 3D scenes faces two significant challenges: (i) the scarcity of paired 3D vision-language data to support grounded learning of 3D scenes, especially considering complexities within diverse object configurations, rich attributes, and intricate relationships; and (ii) the absence of a unified learning framework to distill knowledge from grounded 3D data. In this work, we aim to address these major challenges in 3D vision-language by examining the potential of systematically upscaling 3D vision-language learning in indoor environments. We introduce the first million-scale 3D vision-language dataset, SceneVerse, encompassing about 68K 3D indoor scenes and comprising 2.5M vision-language pairs derived from both human annotations and our scalable scene-graph-based generation approach. We demonstrate that this scaling allows for a unified pre-training framework, Grounded Pre-training for Scenes (GPS), for 3D vision-language learning. Through extensive experiments, we showcase the effectiveness of GPS by achieving state-of-the-art performance on existing 3D visual grounding and question-answering benchmarks. We also show that the data scale-up effect is not limited to GPS, but is generally beneficial for 3D models on 3D vision-language (3D-VL) tasks like semantic segmentation. The vast potential of SceneVerse and GPS is unveiled through zero-shot transfer experiments in the challenging 3D vision-language tasks.

</details>

---

## 208. DEAL: Disentangle and Localize Concept-level Explanations for VLMs

- [ ] DEAL: Disentangle and Localize Concept-level Explanations for VLMs | https://eccv.ecva.net/virtual/2024/poster/755

- **Link**: https://eccv.ecva.net/virtual/2024/poster/755

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large pre-trained Vision-Language Models (VLMs) have become ubiquitous foundational components of other models and downstream tasks. Although powerful, our empirical results reveal that such models might not be able to identify fine-grained concepts. Specifically, the explanations of VLMs with respect to fine-grained concepts are entangled and mislocalized. To address this issue, we propose to DisEntAngle and Localize (DEAL) the concept-level explanations for VLMs without human annotations. The key idea is encouraging the concept-level explanations to be distinct while maintaining consistency with category-level explanations. We conduct extensive experiments and ablation studies on a wide range of benchmark datasets and vision-language models. Our empirical results demonstrate that the proposed method significantly improves the concept-level explanations of the model in terms of disentanglability and localizability. Surprisingly, the improved explainability alleviates the model's reliance on spurious correlations, which further benefits the prediction accuracy.

</details>

---

## 209. A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis

- [ ] A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis | https://eccv.ecva.net/virtual/2024/poster/753

- **Link**: https://eccv.ecva.net/virtual/2024/poster/753

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While conversational generative AI has shown considerable potential in enhancing decision-making for agricultural professionals, its exploration has predominantly been anchored in text-based interactions. The evolution of multimodal conversational AI, leveraging vast amounts of image-text data from diverse sources, marks a significant stride forward. However, the application of such advanced vision-language models in the agricultural domain, particularly for crop disease diagnosis, remains underexplored. In this work, we present the crop disease domain multimodal (CDDM) dataset, a pioneering resource designed to advance the field of agricultural research through the application of multimodal learning techniques. The dataset comprises 137,000 images of various crop diseases, accompanied by 1 million question-answer pairs that span a broad spectrum of agricultural knowledge, from disease identification to management practices. By integrating visual and textual data, CDDM facilitates the development of sophisticated question-answering systems capable of providing precise, useful advice to farmers and agricultural professionals. We demonstrate the utility of the dataset by finetuning state-of-the-art multimodal models, showcasing significant improvements in crop disease diagnosis. Specifically, we employed a novel finetuning strategy that utilizes low-rank adaptation (LoRA) to finetune the visual encoder, adapter and language model simultaneously. Our contributions include not only the dataset but also a finetuning strategy and a benchmark to stimulate further research in agricultural technology, aiming to bridge the gap between advanced AI techniques and practical agricultural applications.

</details>

---

## 210. Contrastive Region Guidance: Improving Grounding in Vision-Language Models without Training

- [ ] Contrastive Region Guidance: Improving Grounding in Vision-Language Models without Training | https://eccv.ecva.net/virtual/2024/poster/754

- **Link**: https://eccv.ecva.net/virtual/2024/poster/754

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

​​Highlighting particularly relevant regions of an image can improve the performance of vision-language models (VLMs) on various vision-language (VL) tasks by guiding the model to attend more closely to these regions of interest. For example, VLMs can be given a “visual prompt”, where visual markers such as bounding boxes delineate key image regions. However, current VLMs that can incorporate visual guidance are either proprietary and expensive or require costly training on curated data with visual prompts. We introduce Contrastive Region Guidance (CRG), a training-free guidance method that enables open-source VLMs to respond to visual prompts. CRG contrasts model outputs produced with and without visual prompts, factoring out biases revealed by the model when answering without the information required to produce a correct answer. CRG achieves substantial improvements in a wide variety of VL tasks: When region annotations are provided, CRG increases absolute accuracy by up to 11.1% on ViP-Bench, a collection of six diverse region-based tasks such as recognition, math, and object relationship reasoning. We also show CRG’s applicability to spatial reasoning, with 10% improvement on What’sUp, as well as to compositional generalization – improving accuracy by 11.5% and 7.5% on two challenging splits from SugarCrepe – and to image-text alignment for generated images, where we improve by 8.4 AUROC and 6.8 F1 points on SeeTRUE. CRG also allows us to re-rank proposed regions in referring expression comprehension and phrase grounding benchmarks like RefCOCO/+/g and Flickr30K Entities, with an average gain of 3.2% in accuracy. Our analysis explores alternative masking strategies for CRG, empirically validating CRG’s design choices.

</details>

---

## 211. IVTP: Instruction-guided Visual Token Pruning for Large Vision-Language Models

- [ ] IVTP: Instruction-guided Visual Token Pruning for Large Vision-Language Models | https://eccv.ecva.net/virtual/2024/poster/759

- **Link**: https://eccv.ecva.net/virtual/2024/poster/759

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Inspired by the remarkable achievements of Large Language Models (LLMs), Large Vision-Language Models (LVLMs) have likewise experienced significant advancements. However, the increased computational cost and token budget occupancy associated with lengthy visual tokens pose significant challenge to the practical applications. Considering that not all visual tokens are essential to the final response, selectively pruning redundant visual tokens can effectively alleviate this challenge. In this paper, we present a novel Instruction-guided Visual Token Pruning (IVTP) approach for LVLMs, which is designed to strike a better balance between computational efficiency and the performance. Specifically, a Group-wise Token Pruning (GTP) module based on attention rollout is integrated into the grouped transformer layer to achieve intra-group attention aggregation via residual connection, thereby improving the assessment of visual token importance, especially for LVLMs with a frozen visual encoder. We then extend the module to LLM in order to further filter out visual tokens that are pertinent to the current textual instructions, by introducing a semantically related pseudo CLS token to serve as a reference for token pruning. This two-stage token pruning mechanism permits a systematic and efficient reduction in the quantity of visual tokens while preserving essential visual information. We apply the proposed method to the most representative LVLM, i.e. LLaVA-1.5. Experimental results demonstrate that when the number of visual tokens is reduced by 88.9%, the computational complexity is decreased by over 46%, with only an average 1.0% accuracy drop across 12 benchmarks, and remarkably surpasses the state-of-the-art token pruning methods. It is worth noting that the proposed method can also work without requiring retraining, thus enabling it to serve as a plug-in across a broader range of LVLMs. Code and trained weights will be available.

</details>

---

## 212. FineMatch: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction

- [ ] FineMatch: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction | https://eccv.ecva.net/virtual/2024/poster/757

- **Link**: https://eccv.ecva.net/virtual/2024/poster/757

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in large-scale pre-training has led to the development of advanced vision-language models (VLMs) with remarkable proficiency in comprehending and generating multimodal content. Despite the impressive ability to perform complex reasoning for VLMs, current models often struggle to effectively and precisely capture the compositional information on both the image and text sides. To address this, we propose FineMatch, a new aspect-based fine-grained text and image matching benchmark, focusing on text and image mismatch detection and correction. This benchmark introduces a novel task for boosting and evaluating the VLMs’ compositionality for aspect-based fine-grained text and image matching. In this task, the models need to predict the mismatched aspect phrases, identify the class of the aspect, and suggest their corrections for a given image and a text caption with 0 to 3 mismatched aspects. To evaluate the models’ performance on this new task, we propose a new evaluation metric named ITM-IoU for which our experiments show a high correlation to human evaluation. In addition, we also provide a comprehensive experimental analysis of existing mainstream VLMs, including fully supervised learning and in-context learning settings. We have found that models trained on FineMatch demonstrate enhanced proficiency in detecting fine-grained text and image mismatches. Moreover, models (e.g., GPT-4V, Gemini Pro Vision) with strong abilities to perform multimodal in-context learning are not as skilled at fine-grained compositional image and text matching analysis as we might have expected. With FineMatch, we are able to build a system for text-to-image generation hallucination detection and correction.

</details>

---

## 213. Instruction Tuning-free Visual Token Complement for Multimodal LLMs

- [ ] Instruction Tuning-free Visual Token Complement for Multimodal LLMs | https://eccv.ecva.net/virtual/2024/poster/758

- **Link**: https://eccv.ecva.net/virtual/2024/poster/758

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As the open community of large language models (LLMs) matures, multimodal LLMs (MLLMs) have promised an elegant bridge between vision and language. However, current research is inherently constrained by challenges such as the need for high-quality instruction pairs and the loss of visual information in image-to-text training objectives. To this end, we propose a Visual Token Complement framework (VTC), that helps MLLMs regain the missing visual features and thus improve response accuracy. Specifically, our VTC integrates text-to-image generation as a guide to identifying the text-irrelevant features, and a visual selector is then developed to generate complementary visual tokens to enrich the original visual input. Moreover, an iterative strategy is further designed to extract more visual information by iteratively using the visual selector without any additional training. Notably, the training pipeline requires no additional image-text pairs, resulting in a desired instruction tuning-free property. Both qualitative and quantitative experiments demonstrate the superiority and efficiency of our VTC. Codes are in the Appendix.

</details>

---

## 214. SPHINX: A Mixer of Weights, Visual Embeddings and Image Scales for Multi-modal Large Language Models

- [ ] SPHINX: A Mixer of Weights, Visual Embeddings and Image Scales for Multi-modal Large Language Models | https://eccv.ecva.net/virtual/2024/poster/761

- **Link**: https://eccv.ecva.net/virtual/2024/poster/761

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present SPHINX, a versatile multi-modal large language model (MLLM) with a joint mixing of model weights, visual embeddings and image scales. First, for stronger vision-language alignment, we unfreeze the large language model (LLM) during pre-training, and introduce a weight mix strategy between LLMs trained by real-world and synthetic data. By directly integrating the weights from two domains, the mixed LLM can efficiently incorporate diverse semantics with favorable robustness. Then, we propose to extract comprehensive visual embeddings from various network architectures, pre-training paradigms, and information granularity, providing language models with more robust image representations. We further propose an efficient strategy aiming to better capture fine-grained appearances of high-resolution images. With a mixing of different scales and high-resolution sub-images, SPHINX attains exceptional visual parsing and reasoning performance on existing evaluation benchmarks. Based on our proposed joint mixing, SPHINX exhibits superior multi-modal understanding capabilities on a wide range of applications, with highlighted fine-grained visual recognition abilities such as region-level understanding, caption grounding, document layout detection, and human pose estimation. We hope our work may cast a light on the exploration of joint mixing in future MLLM research.

</details>

---

## 215. Integration of Global and Local Representations for Fine-grained Cross-modal Alignment

- [ ] Integration of Global and Local Representations for Fine-grained Cross-modal Alignment | https://eccv.ecva.net/virtual/2024/poster/762

- **Link**: https://eccv.ecva.net/virtual/2024/poster/762

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fashion is one of the representative domains of fine-grained Vision-Language Pre-training (VLP) involving a large number of images and text. Previous fashion VLP research has proposed various pre-training tasks to account for fine-grained details in multimodal fusion. However, fashion VLP research has not yet addressed the need to focus on (1) uni-modal embeddings that reflect fine-grained features and (2) hard negative samples to improve the performance of fine-grained V+L retrieval tasks. In this paper, we propose Fashion-FINE (Fashion VLP with Fine-grained Cross-modal Alignment using the INtegrated representations of global and local patch Embeddings), which consists of three key modules. First, a modality-agnostic adapter (MAA) learns uni-modal integrated representations and reflects fine-grained details contained in local patches. Second, hard negative mining with focal loss (HNM-F) performs cross-modal alignment using the integrated representations, focusing on hard negatives to boost the learning of fine-grained cross-modal alignment. Third, comprehensive cross-modal alignment (C-CmA) extracts low- and high-level fashion information from the text and learns the semantic alignment to encourage disentangled embedding of the integrated image representations. Fashion-FINE achieved state-of-the-art performance on two representative public benchmarks (i.e., FashionGen and FashionIQ) in three representative V+L retrieval tasks, demonstrating its effectiveness in learning fine-grained features.

</details>

---

## 216. ShareGPT4V: Improving Large Multi-Modal Models with Better Captions

- [ ] ShareGPT4V: Improving Large Multi-Modal Models with Better Captions | https://eccv.ecva.net/virtual/2024/poster/765

- **Link**: https://eccv.ecva.net/virtual/2024/poster/765

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Modality alignment serves as the cornerstone for large multi-modal models (LMMs). However, the impact of different attributes (e.g., data type, quality, and scale) of training data on facilitating effective alignment is still under-explored. In this paper, we delve into the influence of training data on LMMs, uncovering three pivotal findings: 1) Highly detailed captions enable more nuanced vision-language alignment, significantly boosting the performance of LMMs in diverse benchmarks, surpassing outcomes from brief captions or VQA data; 2) Cutting-edge LMMs can be close to the captioning capability of costly human annotators, and open-source LMMs could reach similar quality after lightweight fine-tuning; 3) The performance of LMMs scales with the number of detailed captions, exhibiting remarkable improvements across a range from thousands to millions of captions. Drawing from these findings, we introduce the ShareGPT4V series for advanced modality alignment. It includes ShareGPT4V, consisting of 100K high-quality captions curated from GPT4-Vision; ShareGPT4V-PT, containing 1.2M captions produced by our Share-Captioner that can be close to the captioning capabilities of GPT4-Vision; and ShareGPT4V-7B, a simple yet superior LMM excelling in most multi-modal benchmarks, which realized better alignment based on our large-scale high-quality captions.

</details>

---

## 217. MyVLM: Personalizing VLMs for User-Specific Queries

- [ ] MyVLM: Personalizing VLMs for User-Specific Queries | https://eccv.ecva.net/virtual/2024/poster/764

- **Link**: https://eccv.ecva.net/virtual/2024/poster/764

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent large-scale vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and generating textual descriptions for visual content. However, these models lack an understanding of user-specific concepts. In this work, we take a first step toward the personalization of VLMs, enabling them to learn and reason over user-provided concepts.  For example, we explore whether these models can learn to recognize you in an image and communicate what you are doing, tailoring the model to reflect your personal experiences and relationships. To effectively recognize a variety of user-specific concepts, we augment the VLM with external concept heads that function as toggles for the model, enabling the VLM the identify the presence of specific target concepts in a given image. Having recognized the concept, we learn a new concept embedding in the intermediate feature space of the VLM. This embedding is tasked with guiding the language model to naturally integrate the target concept in its generated response. We apply our technique to BLIP-2 and LLaVA for personalized image captioning and further show its applicability for personalized visual question-answering. Our experiments demonstrate our ability to generalize to unseen images of learned concepts while preserving the model behavior on unrelated inputs. Code and data will be made available upon acceptance.

</details>

---

## 218. LG-Gaze: Learning Geometry-aware Continuous Prompts for Language-Guided Gaze Estimation

- [ ] LG-Gaze: Learning Geometry-aware Continuous Prompts for Language-Guided Gaze Estimation | https://eccv.ecva.net/virtual/2024/poster/769

- **Link**: https://eccv.ecva.net/virtual/2024/poster/769

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability of gaze estimation models to generalize is often significantly hindered by various factors unrelated to gaze, especially when the training dataset is limited. Current strategies aim to address this challenge through different domain generalization techniques, yet they have had limited success due to the risk of overfitting when solely relying on value labels for regression. Recent progress in pre-trained vision-language models has motivated us to capitalize on the abundant semantic information available. We propose a novel approach in this paper, reframing the gaze estimation task as a vision-language alignment issue. Our proposed framework, named Language-Guided Gaze Estimation (LG-Gaze), learns continuous and geometry-sensitive features for gaze estimation benefit from the rich prior knowledges of vision-language models. Specifically, LG-Gaze aligns gaze features with continuous linguistic features through our proposed multimodal contrastive regression loss, which customizes adaptive weights for different negative samples. Furthermore, to better adapt to the labels for gaze estimation task, we propose a geometry-aware interpolation method to obtain more precise gaze embeddings. Through extensive experiments, we validate the efficacy of our framework in four different cross-domain evaluation tasks.

</details>

---

## 219. TAG: Text Prompt Augmentation for Zero-Shot Out-of-Distribution Detection

- [ ] TAG: Text Prompt Augmentation for Zero-Shot Out-of-Distribution Detection | https://eccv.ecva.net/virtual/2024/poster/772

- **Link**: https://eccv.ecva.net/virtual/2024/poster/772

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection has been extensively studied for the reliable deployment of deep-learning models. Despite great progress in this research direction, most works focus on discriminative classifiers and perform OOD detection based on single-modal representations that consist of either visual or textual features. Moreover, they rely on training with in-distribution (ID) data. The emergence of vision-language models (e.g. \CLIPc) allows to perform zero-shot OOD detection by leveraging multi-modal feature embeddings and therefore only rely on labels defining ID data. Several approaches have been devised but these either need a given OOD label set, which might deviate from real OOD data, or fine-tune  CLIP, which potentially has to be done for different ID datasets. In this paper, we first adapt various OOD scores developed for discriminative classifiers to \CLIP. Further, we propose an enhanced method named \emph{TAG} based on Text prompt AuGmentation to amplify the separation between ID and OOD data, which is simple but effective, and can be applied on various score functions. Its performance is demonstrated on CIFAR-100 and large-scale ImageNet-1k OOD detection benchmarks. It consistently improves AUROC and FPR95 on CIFAR-100 across five commonly used architectures over four baseline OOD scores. The average AUROC and FPR95 improvements are 6.35 % and 10.67 %, respectively. The results for ImageNet-1k follow a similar, but less pronounced pattern.

</details>

---

## 220. Find n' Propagate: Open-Vocabulary 3D Object Detection in Urban Environments

- [ ] Find n' Propagate: Open-Vocabulary 3D Object Detection in Urban Environments | https://eccv.ecva.net/virtual/2024/poster/776

- **Link**: https://eccv.ecva.net/virtual/2024/poster/776

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we tackle the limitations of current LiDAR-based 3D object detection systems, which are hindered by a restricted class vocabulary and the high costs associated with annotating new object classes. Our exploration of open-vocabulary (OV) learning in urban environments aims to capture novel instances using pre-trained vision-language models (VLMs) with multi-sensor data. We design and benchmark a set of four potential solutions as baselines, categorizing them into either top-down or bottom-up approaches based on their input data strategies. While effective, these methods exhibit certain limitations, such as missing novel objects in 3D box estimation or applying rigorous priors, leading to biases towards objects near the camera or of rectangular geometries. To overcome these limitations, we introduce a universal \textsc{Find n' Propagate} approach for 3D OV tasks, aimed at maximizing the recall of novel objects and propagating this detection capability to more distant areas thereby progressively capturing more. In particular, we utilize a greedy box seeker to search against 3D novel boxes of varying orientations and depth in each generated frustum and ensure the reliability of newly identified boxes by cross alignment and density ranker. Additionally, the inherent bias towards camera-proximal objects is alleviated by the proposed remote simulator, which randomly diversifies pseudo-labeled novel instances in the self-training process, combined with the fusion of base samples in the memory bank. Extensive experiments demonstrate a 53\% improvement in novel recall across diverse OV settings, VLMs, and 3D detectors. Notably, we achieve up to a 3.97-fold increase in Average Precision (AP) for novel object classes. The \texttt{source code} is made available in the supplementary material.

</details>

---

## 221. Open-Vocabulary Camouflaged Object Segmentation

- [ ] Open-Vocabulary Camouflaged Object Segmentation | https://eccv.ecva.net/virtual/2024/poster/786

- **Link**: https://eccv.ecva.net/virtual/2024/poster/786

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, the emergence of the large-scale vision-language model (VLM), such as CLIP, has opened the way towards open-world object perception. Many works have explored the utilization of pre-trained VLM for the challenging open-vocabulary dense prediction task that requires perceiving diverse objects with novel classes at inference time. Existing methods construct experiments based on the public datasets of related tasks, which are not tailored for open vocabulary and rarely involve imperceptible objects camouflaged in complex scenes due to data collection bias and annotation costs. To fill in the gaps, we introduce a new task, open-vocabulary camouflaged object segmentation (OVCOS), and construct a large-scale complex scene dataset (\textbf{OVCamo}) containing 11,483 hand-selected images with fine annotations and corresponding object classes. Further, we build a strong single-stage open-vocabulary \underline{c}amouflaged \underline{o}bject \underline{s}egmentation transform\underline{er} baseline \textbf{OVCoser} attached to the parameter-fixed CLIP with iterative semantic guidance and structure enhancement. By integrating the guidance of class semantic knowledge and the supplement of visual structure cues from the edge and depth information, the proposed method can efficiently capture camouflaged objects. Moreover, this effective framework also surpasses previous state-of-the-arts of open-vocabulary semantic image segmentation by a large margin on our OVCamo dataset. With the proposed dataset and baseline, we hope that this new task with more practical value can further expand the research on open-vocabulary dense prediction tasks. Our code and data can be found in the \href{https://github.com/lartpang/OVCamo}{link}.

</details>

---

## 222. Zero-shot Object Counting with Good Exemplars

- [ ] Zero-shot Object Counting with Good Exemplars | https://eccv.ecva.net/virtual/2024/poster/795

- **Link**: https://eccv.ecva.net/virtual/2024/poster/795

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot object counting (ZOC) aims to enumerate objects in images using only the names of object classes during testing, without the need for manual annotations. However, a critical challenge in current ZOC methods lies in their inability to effectively identify high-quality exemplars. This deficiency hampers scalability across diverse classes and undermines the development of strong visual associations between the identified classes and image content. To this end, we propose the Visual Association-based Zero-shot Object Counting (VA-Count) framework. VA-Count consists of an Exemplar Enhancement Module (EEM) and a Noise Suppression Module (NSM) that synergistically refine the process of class exemplar identification while minimizing the consequences of incorrect object identification. The EEM utilizes advanced Vision-Language Pretaining models to discover potential exemplars, ensuring the framework's adaptability to various classes. Meanwhile, the NSM employs contrastive learning to differentiate between optimal and suboptimal exemplar pairs, reducing the negative effects of erroneous exemplars. The effectiveness and scalability of VA-Count in zero-shot contexts are demonstrated through its superior performance on three object counting datasets.

</details>

---

## 223. Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation

- [ ] Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation | https://eccv.ecva.net/virtual/2024/poster/823

- **Link**: https://eccv.ecva.net/virtual/2024/poster/823

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) demonstrate remarkable zero-shot generalization to unseen tasks, but fall short of the performance of supervised methods in generalizing to downstream tasks with limited data. Prompt learning is emerging as a parameter-efficient method for adapting VLMs, but state-of-the-art approaches require annotated samples. In this paper we propose a novel approach to prompt learning based on unsupervised knowledge distillation from more powerful models. Our approach, which we call Knowledge Distillation Prompt Learning (KDPL), can be integrated into existing prompt learning techniques and eliminates the need for labeled examples during adaptation. Our experiments on more than ten standard benchmark datasets demonstrate that KDPL is very effective at improving generalization of learned prompts for zero-shot domain generalization, zero-shot cross-dataset generalization, and zero-shot base-to-novel class generalization problems. KDPL requires no ground-truth labels for adaptation, and moreover we show that even in the absence of any knowledge of training class names it can be used to effectively transfer knowledge.

</details>

---

## 224. Explain via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts

- [ ] Explain via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts | https://eccv.ecva.net/virtual/2024/poster/825

- **Link**: https://eccv.ecva.net/virtual/2024/poster/825

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The concept bottleneck model (CBM) is an interpretable-by-design framework that makes decisions by first predicting a set of interpretable concepts, and then predicting the class label based on the given concepts. Existing CBMs are trained with a fixed set of concepts (concepts are either annotated by the dataset or queried from language models). However, this closed-world assumption is unrealistic in practice, as users may wonder about the role of any desired concept in decision-making after the model is deployed. Inspired by the large success of recent vision-language pre-trained models such as CLIP in zero-shot classification, we propose OpenCBM'' to equip the CBM with open vocabulary concepts via: (1) Aligning the feature space of a trainable image feature extractor with that of a CLIP's image encoder via a prototype based feature alignment; (2) Simultaneously training an image classifier on the downstream dataset; (3) Reconstructing the trained classification head via any set of user-desired textual concepts encoded by CLIP's text encoder. To reveal potentially missing concepts from users, we further propose to iteratively find the closest concept embedding to the residual parameters during the reconstruction until the residual is small enough. To the best of our knowledge, our OpenCBM'' is the first CBM with concepts of open vocabularies, providing users the unique benefit such as removing, adding, or replacing any desired concept to explain the model's prediction even after a model is trained. Moreover, our model significantly outperforms the previous state-of-the-art CBM by 9% in the classification accuracy on the benchmark dataset CUB-200-2011. The code will be released upon acceptance.

</details>

---

## 225. Class-Incremental Learning with CLIP: Adaptive Representation Adjustment and Parameter Fusion

- [ ] Class-Incremental Learning with CLIP: Adaptive Representation Adjustment and Parameter Fusion | https://eccv.ecva.net/virtual/2024/poster/837

- **Link**: https://eccv.ecva.net/virtual/2024/poster/837

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Class-incremental learning is a challenging problem, where the goal is to train a model that can classify data from an increasing number of classes over time. With the advancement of vision-language pre-trained models such as CLIP, they demonstrate good generalization ability that allows them to excel in class-incremental learning with completely frozen parameters. However, further adaptation to downstream tasks by simply fine-tuning the model leads to severe forgetting.  Most existing works with pre-trained models assume that the forgetting of old classes is uniform when the model acquires new knowledge. In this paper, we propose a method that leverages the textual features of class names to measure the degree of influence on old classes by new classes and adjusts their representations accordingly to reduce forgetting. In addition, we also propose a decomposed parameter fusion method for the adapter module. It can greatly reduce the forgetting caused by fine-tuning the adapter modules with new data. Experiments on several conventional benchmarks show that our method achieves state-of-the-art results.

</details>

---

## 226. Parrot Captions Teach CLIP to Spot Text

- [ ] Parrot Captions Teach CLIP to Spot Text | https://eccv.ecva.net/virtual/2024/poster/853

- **Link**: https://eccv.ecva.net/virtual/2024/poster/853

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite CLIP being the foundation model in numerous vision-language applications, the CLIP suffers from a severe text spotting bias. Such bias causes CLIP models to `Parrot' the visual text embedded within images while disregarding the authentic visual semantics. We uncover that in the most popular image-text dataset LAION-2B, the captions also densely parrot (spell) the text embedded in images. Our analysis shows that around 50% of images are embedded with visual text content and around 30% of captions words are concurrently embedded in the visual content. Based on such observation, we thoroughly inspect the different released versions of CLIP models and verify that the visual text is a dominant factor in measuring the LAION-style image-text similarity for these models. To examine whether these parrot captions shape the text spotting bias, we train a series of CLIP models with LAION subsets curated by different parrot-caption-oriented criteria. We show that training with parrot captions easily shapes such bias but harms the expected visual-language representation learning in CLIP models across various vision-language downstream tasks. This suggests that it is urgent to revisit either the design of CLIP-like models or the existing image-text dataset curation pipeline built on CLIP score filtering.

</details>

---

## 227. MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description

- [ ] MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description | https://eccv.ecva.net/virtual/2024/poster/865

- **Link**: https://eccv.ecva.net/virtual/2024/poster/865

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent foundation models trained on a tremendous scale of data have shown great promise in a wide range of computer vision tasks and application domains. However, less attention has been paid to the marine realms, which in contrast cover the majority of our blue planet. The scarcity of labeled data is the most hindering issue, and marine photographs illustrate significantly different appearances and contents from general in-air images. Using existing foundation models for marine visual analysis does not yield satisfactory performance, due to not only the data distribution shift, but also the intrinsic limitations of the existing foundation models (e.g., lacking semantics, redundant mask generation, or restricted to image-level scene understanding). In this work, we emphasize both model and data approaches for understanding marine ecosystems. We introduce MarineInst, a foundation model for the analysis of the marine realms with instance visual description, which outputs instance masks and captions for marine object instances. To train MarineInst, we acquire MarineInst20M, the largest marine image dataset to date, which contains a wide spectrum of marine images with high-quality semantic instance masks constructed by a mixture of human-annotated instance masks and model-generated instance masks from our automatic procedure of binary instance filtering. To generate informative and detailed semantic instance captions, we use vision-language models to produce semantic richness with various granularities. Our model and dataset support a wide range of marine visual analysis tasks, from image-level scene understanding to regional mask-level instance understanding. More significantly, MarineInst exhibits strong generalization ability and flexibility to support a wide range of downstream tasks with state-of-the-art performance.

</details>

---

## 228. PointLLM: Empowering Large Language Models to Understand Point Clouds

- [ ] PointLLM: Empowering Large Language Models to Understand Point Clouds | https://eccv.ecva.net/virtual/2024/poster/879

- **Link**: https://eccv.ecva.net/virtual/2024/poster/879

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The unprecedented advancements in Large Language Models (LLMs) have shown a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This paper introduces PointLLM, a preliminary effort to fill this gap, empowering LLMs to understand point clouds and offering a new avenue beyond 2D data. PointLLM understands colored object point clouds with human instructions and generates contextually appropriate responses, illustrating its grasp of point clouds and common sense. Specifically, it leverages a point cloud encoder with a powerful LLM to effectively fuse geometric, appearance, and linguistic information. To overcome the scarcity of point-text instruction following data, we developed an automated data generation pipeline, collecting a large-scale dataset of more than 730K samples with 660K different objects, which facilitates the adoption of the two-stage training strategy prevalent in MLLM development. Additionally, we address the absence of appropriate benchmarks and the limitations of current evaluation metrics by proposing two novel benchmarks: Generative 3D Object Classification and 3D Object Captioning, which are supported by new, comprehensive evaluation metrics derived from human and GPT analyses. Through exploring various training strategies, we develop PointLLM, significantly surpassing 2D and 3D baselines, with a notable achievement in human-evaluated object captioning tasks where it surpasses human annotators in over 50% of the samples. Codes, datasets, and benchmarks are available at https://github.com/OpenRobotLab/PointLLM.

</details>

---

## 229. VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions

- [ ] VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions | https://eccv.ecva.net/virtual/2024/poster/993

- **Link**: https://eccv.ecva.net/virtual/2024/poster/993

- **Conference**: ECCV

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Predicting future trajectories for other road agents is an essential task for autonomous vehicles.  Established trajectory prediction methods primarily use agent tracks generated by a detection and tracking system and HD map as inputs to a model which predicts agent trajectories. In this work, we propose a novel method that also incorporates visual input from surround-view cameras, allowing the model to utilize visual cues such as human gazes and gestures, road conditions, vehicle turn signals, etc, which are typically hidden from the model in prior trajectory prediction methods. Furthermore, we use textual descriptions generated by a Vision-Language Model (VLM) and refined by a Large Language Model (LLM) as supervision to guide the model on what to learn from the input data. Our experiments show that both the visual inputs and the textual descriptions contribute to improvements in trajectory prediction performance, and our qualitative analysis highlights how the model is able to exploit these additional inputs. Despite using these extra inputs, our method achieves a latency of 53 ms, significantly lower than that of previous single-agent prediction methods with similar performance. Lastly, in this work we create and release the nuScenes-Text dataset, which augments the established nuScenes dataset with rich textual annotations for every scene, demonstrating the positive impact of utilizing VLM on trajectory prediction.

</details>

---

