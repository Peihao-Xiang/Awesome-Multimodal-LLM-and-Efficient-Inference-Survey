# CVPR 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_cvpr2024_papers.csv

## 1. Learning Object State Changes in Videos: An Open-World Perspective

- [ ] Learning Object State Changes in Videos: An Open-World Perspective | https://cvpr.thecvf.com/virtual/2024/poster/29181

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29181

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Object State Changes (OSCs) are pivotal for video understanding. While humans can effortlessly generalize OSC understanding from familiar to unknown objects, current approaches are confined to a closed vocabulary. Addressing this gap, we introduce a novel open-world formulation for the video OSC problem. The goal is to temporally localize the three stages of an OSC---the object's initial state, its transitioning state, and its end state---whether or not the object has been observed during training. Towards this end, we develop VidOSC, a holistic learning approach that: (1) leverages text and vision-language models for supervisory signals to obviate manually labeling OSC training data, and (2) abstracts fine-grained shared state representations from objects to enhance generalization. Furthermore, we present HowToChange, the first open-world benchmark for video OSC localization, which offers an order of magnitude increase in the label space and annotation volume compared to the best existing benchmark. Experimental results demonstrate the efficacy of our approach, in both traditional closed-world and open-world scenarios.

</details>

---

## 2. MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer

- [ ] MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer | https://cvpr.thecvf.com/virtual/2024/poster/29185

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29185

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Transformers (VLTs) have shown great success recently, but are meanwhile accompanied by heavy computation costs, where a major reason can be attributed to the large number of visual and language tokens. Existing token pruning research for compressing VLTs mainly follows a single-modality-based scheme yet ignores the critical role of aligning different modalities for guiding the token pruning process, causing the important tokens for one modality to be falsely pruned in another modality branch. Meanwhile, existing VLT pruning works also lack the flexibility to dynamically compress each layer based on different input samples. To this end, we propose a novel framework named Multimodal Alignment-Guided Dynamic Token Prunning (MADTP) for accelerating various VLTs. Specifically, we first introduce a well-designed Multi-modality Alignment Guidance (MAG) module that can align features of the same semantic concept from different modalities, to ensure the pruned tokens are less important for all modalities. We further design a novel Dynamic Token Pruning (DTP) module, which can adaptively adjust the token compression ratio in each layer based on different input instances. Extensive experiments on various benchmarks demonstrate that MADTP significantly reduces the computational complexity of kinds of multimodal models while preserving competitive performance. Notably, when applied to the BLIP model in the NLVR2 dataset, MADTP can reduce the GFLOPs by 80% with less than 4% performance degradation.

</details>

---

## 3. A Simple Recipe for Language-guided Domain Generalized Segmentation

- [ ] A Simple Recipe for Language-guided Domain Generalized Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/29196

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29196

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generalization to new domains not seen during training is one of the long-standing challenges in deploying neural networks in real-world applications. Existing generalization techniques either necessitate external images for augmentation, and/or aim at learning invariant representations by imposing various alignment constraints. Large-scale pretraining has recently shown promising generalization capabilities, along with the potential of binding different modalities. For instance, the advent of vision-language models like CLIP has opened the doorway for vision models to exploit the textual modality. In this paper, we introduce a simple framework for generalizing semantic segmentation networks by employing language as the source of randomization. Our recipe comprises three key ingredients: (i) the preservation of the intrinsic CLIP robustness through minimal fine-tuning, (ii) language-driven local style augmentation, and (iii) randomization by locally mixing the source and augmented styles during training. Extensive experiments report state-of-the-art results on various generalization benchmarks.

</details>

---

## 4. OVMR: Open-Vocabulary Recognition with Multi-Modal References

- [ ] OVMR: Open-Vocabulary Recognition with Multi-Modal References | https://cvpr.thecvf.com/virtual/2024/poster/29198

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29198

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The challenge of open-vocabulary recognition lies in the model has no clue of new categories it is applied to. Existing works have proposed different methods to embed category cues into the model, e.g., through few-shot fine-tuning, providing category names or textual descriptions to Vision-Language Models. Fine-tuning is time-consuming and degrades the generalization capability. Textual descriptions could be ambiguous and fail to depict visual details. This paper tackles open-vocabulary recognition from a different perspective by referring to multi-modal clues composed of textual descriptions and exemplar images. Our method, named OVMR, adopts two innovative components to pursue a more robust category cues embedding. A multi-modal classifier is first generated by dynamically complementing textual descriptions with image exemplars. A preference-based refinement module is hence applied to fuse uni-modal and multi-modal classifiers, with the aim to alleviate issues of low-quality exemplar images or textual descriptions. The proposed OVMR is a plug-and-play module, and works well with exemplar images randomly crawled from the Internet.Extensive experiments have demonstrated the promising performance of OVMR, e.g., it outperforms existing methods across various scenarios and setups.

</details>

---

## 5. PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection

- [ ] PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection | https://cvpr.thecvf.com/virtual/2024/poster/29202

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29202

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The vision-language model has brought great improvement to few-shot industrial anomaly detection, which usually needs to design of hundreds of prompts through prompt engineering. For automated scenarios, we first use conventional prompt learning with many-class paradigm as the baseline to automatically learn prompts but found that it can not work well in one-class anomaly detection. To address the above problem, this paper proposes a one-class prompt learning method for few-shot anomaly detection, termed PromptAD. First, we propose semantic concatenation which can transpose normal prompts into anomaly prompts by concatenating normal prompts with anomaly suffixes, thus constructing a large number of negative samples used to guide prompt learning in one-class setting. Furthermore, to mitigate the training challenge caused by the absence of anomaly images, we introduce the concept of explicit anomaly margin, which is used to explicitly control the margin between normal prompt features and anomaly prompt features through a hyper-parameter. For image-level/pixel-level anomaly detection, PromptAD achieves first place in 11/12 few-shot settings on MVTec and VisA.

</details>

---

## 6. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding

- [ ] Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding | https://cvpr.thecvf.com/virtual/2024/poster/29205

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29205

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have advanced considerably, intertwining visual recognition and language understanding to generate content that is not only coherent but also contextually attuned. Despite their success, LVLMs still suffer from the issue of object hallucinations, where models generate plausible yet incorrect outputs that include objects that do not exist in the images. To mitigate this issue, we introduce Visual Contrastive Decoding (VCD), a simple and training-free method that contrasts output distributions derived from original and distorted visual inputs. The proposed VCD effectively reduces the over-reliance on statistical bias and unimodal priors, two essential causes of object hallucinations. This adjustment ensures the generated content is closely grounded to visual inputs, resulting in contextually accurate outputs. Our experiments show that VCD, without either additional training or the usage of external tools, significantly mitigates the object hallucination issue across different LVLM families. Beyond mitigating object hallucinations, VCD also excels in general LVLM benchmarks, highlighting its wide-ranging applicability. Codes will be released.

</details>

---

## 7. Hallucination Augmented Contrastive Learning for Multimodal Large Language Model

- [ ] Hallucination Augmented Contrastive Learning for Multimodal Large Language Model | https://cvpr.thecvf.com/virtual/2024/poster/29216

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29216

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have been shown to efficiently integrate natural language with visual information to handle multi-modal tasks. However, MLLMs still face a fundamental limitation of hallucinations, where they tend to generate erroneous or fabricated information. In this paper, we address hallucinations in MLLMs from a novel perspective of representation learning. We first analyzed the representation distribution of textual and visual tokens in MLLM, revealing two important findings: 1) there is a significant gap between textual and visual representations, indicating unsatisfactory cross-modal representation alignment; 2) representations of texts that contain and do not contain hallucinations are entangled, making it challenging to distinguish them. These two observations inspire us with a simple yet effective method to mitigate hallucinations. Specifically, we introduce contrastive learning into MLLMs and use text with hallucination as hard negative examples, naturally bringing representations of non-hallucinatory text and visual samples closer while pushing way representations of non-hallucinatory and hallucinatory text. We evaluate our method quantitatively and qualitatively, showing its effectiveness in reducing hallucination occurrences and improving performance across multiple benchmarks. On the MMhal-Bench benchmark, our method obtains a 34.66\% /29.5\% improvement over the baseline MiniGPT-4/LLaVA.

</details>

---

## 8. DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback

- [ ] DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback | https://cvpr.thecvf.com/virtual/2024/poster/29224

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29224

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present DRESS, a large vision language model (LVLM) that innovatively exploits Natural Language feedback (NLF) from Large Language Models to enhance its alignment and interactions by addressing two key limitations in the state-of-the-art LVLMs. First, prior LVLMs generally rely only on the instruction finetuning stage to enhance alignment with human preferences. Without incorporating extra feedback, they are still prone to generate unhelpful, hallucinated, or harmful responses. Second, while the visual instruction tuning data is generally structured in a multi-turn dialogue format, the connections and dependencies among consecutive conversational turns are weak. This reduces the capacity for effective multi-turn interactions. To tackle these, we propose a novel categorization of the NLF into two key types: critique and refinement. The critique NLF identifies the strengths and weaknesses of the responses and is used to align the LVLMs with human preferences. The refinement NLF offers concrete suggestions for improvement and is adopted to improve the interaction ability of the LVLMs-- which focuses on LVLMs' ability to refine responses by incorporating feedback in multi-turn interactions. To address the non-differentiable nature of NLF, we generalize conditional reinforcement learning for training. Our experimental results demonstrate that DRESS can generate more helpful (9.76%), honest (11.52%), and harmless (21.03%) responses, and more effectively learn from feedback during multi-turn interactions compared to SOTA LVLMs.

</details>

---

## 9. LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding

- [ ] LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding | https://cvpr.thecvf.com/virtual/2024/poster/29235

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29235

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, leveraging large language models (LLMs) or multimodal large language models (MLLMs) for document understanding has been proven very promising. However, previous works that employ LLMs/MLLMs for document understanding have not fully explored and utilized the document layout information, which is vital for precise document understanding. In this paper, we propose LayoutLLM, an LLM/MLLM based method for document understanding. The core of LayoutLLM is a layout instruction tuning strategy, which is specially designed to enhance the comprehension and utilization of document layouts. The proposed layout instruction tuning strategy consists of two components: Layout-aware Pre-training and Layout-aware Supervised Fine-tuning. To capture the characteristics of document layout in Layout-aware Pre-training, three groups of pre-training tasks, corresponding to document-level, region-level and segment-level information, are introduced. Furthermore, a novel module called layout chain-of-thought (LayoutCoT) is devised to enable LayoutLLM to focus on regions relevant to the question and generate accurate answers. LayoutCoT is effective for boosting the performance of document understanding. Meanwhile, it brings a certain degree of interpretability, which could facilitate manual inspection and correction. Experiments on standard benchmarks show that the proposed LayoutLLM significantly outperforms existing methods that adopt open-source 7B LLMs/MLLMs for document understanding.

</details>

---

## 10. Harnessing Large Language Models for Training-free Video Anomaly Detection

- [ ] Harnessing Large Language Models for Training-free Video Anomaly Detection | https://cvpr.thecvf.com/virtual/2024/poster/29246

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29246

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video anomaly detection (VAD) aims to temporally locate abnormal events in a video. Existing works mostly rely on training deep models to learn the distribution of normality with either video-level supervision, one-class supervision, or in an unsupervised setting. Training-based methods are prone to be domain-specific, thus being costly for practical deployment as any domain change will involve data collection and model training. In this paper, we radically depart from previous efforts and propose LAnguage-based VAD (LAVAD), a method tackling VAD in a novel, training-free paradigm, exploiting the capabilities of pre-trained large language models (LLMs) and existing vision-language models (VLMs). We leverage VLM-based captioning models to generate textual descriptions for each frame of any test video. With the textual scene description, we then devise a prompting mechanism to unlock the capability of LLMs in terms of temporal aggregation and anomaly score estimation, turning LLMs into an effective video anomaly detector. We further leverage modality-aligned VLMs and propose effective techniques based on cross-modal similarity for cleaning noisy captions and refining the LLM-based anomaly scores. We evaluate LAVAD on two large datasets featuring real-world surveillance scenarios (UCF-Crime and XD-Violence), showing that it outperforms both unsupervised and one-class methods without requiring any training or data collection.

</details>

---

## 11. SyncMask: Synchronized Attentional Masking for Fashion-centric Vision-Language Pretraining

- [ ] SyncMask: Synchronized Attentional Masking for Fashion-centric Vision-Language Pretraining | https://cvpr.thecvf.com/virtual/2024/poster/29248

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29248

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have made significant strides in cross-modal understanding through large-scale paired datasets. However, in fashion domain, datasets often exhibit a disparity between the information conveyed in image and text. This issue stems from datasets containing multiple images of a single fashion item all paired with one text, leading to cases where some textual details are not visible in individual images. This mismatch, particularly when non-co-occurring elements are masked, undermines the training of conventional VLM objectives like Masked Language Modeling and Masked Image Modeling, thereby hindering the model’s ability to accurately align fine-grained visual and textual features. Addressing this problem, we propose Synchronized attentional Masking (SyncMask), which generate masks that pinpoint the image patches and word tokens where the information co-occur in both image and text. This synchronization is accomplished by harnessing cross-attentional features obtained from a momentum model, ensuring a precise alignment between the two modalities. Additionally, we enhance grouped batch sampling with semi-hard negatives, effectively mitigating false negative issues in Image-Text Matching and Image-Text Contrastive learning objectives within fashion datasets. Our experiments demonstrate the effectiveness of the proposed approach, outperforming existing methods in three downstream tasks.

</details>

---

## 12. G^3-LQ: Marrying Hyperbolic Alignment with Explicit Semantic-Geometric Modeling for 3D Visual Grounding

- [ ] G^3-LQ: Marrying Hyperbolic Alignment with Explicit Semantic-Geometric Modeling for 3D Visual Grounding | https://cvpr.thecvf.com/virtual/2024/poster/29251

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29251

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Grounding referred objects in 3D scenes is a burgeoning vision-language task pivotal for propelling Embodied AI, as it endeavors to connect the 3D physical world with free-form descriptions. Compared to the 2D counterparts, challenges posed by the variability of 3D visual grounding remain relatively unsolved in existing studies: 1) the underlying geometric and complex spatial relationships in 3D scene. 2) the inherent complexity of 3D grounded language. 3) the inconsistencies between text and geometric features. To tackle these issues, we propose G$^3$-LQ, a DEtection TRansformer-based model tailored for 3D visual grounding task. G$^3$-LQ explicitly models $\textbf{G}$eometric-aware visual representations and $\textbf{G}$enerates fine-$\textbf{G}$rained $\textbf{L}$anguage-guided object $\textbf{Q}$ueries in an overarching framework, which comprises two dedicated modules. Specifically, the Position Adaptive Geometric Exploring (PAGE) unearths underlying information of 3D objects in the geometric details and spatial relationships perspectives. The Fine-grained Language-guided Query Selection (Flan-QS) delves into syntactic structure of texts and generates object queries that exhibit higher relevance towards fine-grained text features. Finally, a pioneering Poincaré Semantic Alignment (PSA) loss establishes semantic-geometry consistencies by modeling non-linear vision-text feature mappings and aligning them on a hyperbolic prototype—Poincaré ball. Extensive experiments verify the superiority of our G$^3$-LQ method, trumping the state-of-the-arts by a considerable margin.

</details>

---

## 13. OST: Refining Text Knowledge with Optimal Spatio-Temporal Descriptor for General Video Recognition

- [ ] OST: Refining Text Knowledge with Optimal Spatio-Temporal Descriptor for General Video Recognition | https://cvpr.thecvf.com/virtual/2024/poster/29252

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29252

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Due to the resource-intensive nature of training vision-language models on expansive video data, a majority of studies have centered on adapting pre-trained image-language models to the video domain. Dominant pipelines propose to tackle the visual discrepancies with additional temporal learners while overlooking the substantial discrepancy for web-scaled descriptive narratives and concise action category names, leading to less distinct semantic space and potential performance limitations. In this work, we prioritize the refinement of text knowledge to facilitate generalizable video recognition. To address the limitations of the less distinct semantic space of category names, we prompt a large language model (LLM) to augment action class names into Spatio-Temporal Descriptors thus bridging the textual discrepancy and serving as a knowledge base for general recognition. Moreover, to assign the best descriptors with different video instances, we propose Optimal Descriptor Solver, forming the video recognition problem as solving the optimal matching flow across frame-level representations and descriptors. Comprehensive evaluations in zero-shot, few-shot, and fully supervised video recognition highlight the effectiveness of our approach. Our best model achieves a state-of-the-art zero-shot accuracy of 75.1% on Kinetics-600.

</details>

---

## 14. Towards Better Vision-Inspired Vision-Language Models

- [ ] Towards Better Vision-Inspired Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29276

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29276

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language (VL) models have achieved unprecedented success recently, in which the connection module is the key to bridge the modality gap. Nevertheless, the abundant visual clues are not sufficiently exploited in most existing methods. On the vision side, most existing approaches only use the last feature of the vision tower, without using the low-level features. On the language side, most existing methods only introduce shallow vision-language interactions. In this paper, we present a vision-inspired vision-language connection module, dubbed as VIVL, which efficiently exploits the vision cue for VL models. To take advantage of the lower-level information from the vision tower, a feature pyramid extractor (FPE) is introduced to combine features from different intermediate layers, which enriches the visual cue with negligible parameters and computation overhead. To enhance VL interactions, we propose deep vision-language prompts (DVLP) that allows deep interactions of vision and language features efficiently. Our VIVL exceeds the previous state-of-the-art method by 18.1 CIDEr when training from scratch on the COCO caption task, which greatly improves the data efficiency. When used as a plug-in module, VIVL consistently improves the performance for various backbones and VL frameworks, delivering new state-of-the-art results on multiple benchmarks, e.g., NoCaps and VQAv2.

</details>

---

## 15. How to Configure Good In-Context Sequence for Visual Question Answering

- [ ] How to Configure Good In-Context Sequence for Visual Question Answering | https://cvpr.thecvf.com/virtual/2024/poster/29319

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29319

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Inspired by the success of Large Language Models in dealing with new tasks via In-Context Learning (ICL) in NLP, researchers have also developed Large Vision-Language Models (LVLMs) with ICL capabilities. However, when implementing ICL using these LVLMs, researchers usually resort to the simplest way like random sampling to configure the in-context sequence, thus leading to sub-optimal results. To enhance the ICL performance, in this study, we use  Visual Question Answering (VQA) as case study to explore diverse in-context configurations to find the powerful ones. Additionally, through observing the changes of the LVLM outputs by altering the in-context sequence, we gain insights into the inner properties of LVLMs, improving our understanding of them. Specifically, to explore in-context configurations, we design diverse retrieval methods and employ different strategies to manipulate the retrieved in-context samples. Through exhaustive experiments on three VQA datasets: VQAv2, VizWiz, and OK-VQA, we uncover three important inner properties of the applied LVLM and demonstrate which strategies can consistently improve the ICL VQA performance. Our code is provided in: https://anonymous.4open.science/r/CVPR2024 ICL VQA.

</details>

---

## 16. Investigating Compositional Challenges in Vision-Language Models for Visual Grounding

- [ ] Investigating Compositional Challenges in Vision-Language Models for Visual Grounding | https://cvpr.thecvf.com/virtual/2024/poster/29329

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29329

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have achieved high performance on various downstream tasks, which have been widely used for visual grounding tasks in a weakly supervised manner. However, despite the performance gains contributed by large vision and language pre-training, we find that state-of-the-art VLMs struggle with compositional reasoning on grounding tasks. To demonstrate this, we propose Attribute, Relation, and Priority Grounding (ARPGrounding) benchmark to test VLMs' compositional reasoning ability on visual grounding tasks. ARPGrounding contains 11,425 samples and evaluates the compositional understanding of VLMs in three dimensions: 1) attribute, denoting comprehension of objects' properties, 2) relation, indicating an understanding of relation between objects, 3) priority, reflecting an awareness of the part of speech associated with nouns. Using the ARPGrounding benchmark, we evaluate several mainstream VLMs. We empirically find that these models perform quite well on conventional visual grounding datasets, achieving performance comparable to or surpassing state-of-the-art methods. However, they show strong deficiencies in compositional reasoning, as evidenced by their inability to establish links between objects and their associated attributes, a limited grasp of relational understanding, and insensitivity towards the prioritization of objects. Furthermore, we propose a composition-aware fine-tuning pipeline, demonstrating the potential to leverage cost-effective image-text annotations for enhancing the compositional understanding of VLMs in grounding tasks.

</details>

---

## 17. VLP: Vision Language Planning for Autonomous Driving

- [ ] VLP: Vision Language Planning for Autonomous Driving | https://cvpr.thecvf.com/virtual/2024/poster/29341

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29341

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Autonomous driving is a complex and challenging task that aims at safe motion planning through scene understanding and reasoning. While vision-only autonomous driving methods have recently achieved notable performance, through enhanced scene understanding, several key issues, including lack of reasoning, low generalization performance and long-tail scenarios, still need to be addressed. In this paper, we present VLP, a novel Vision-Language-Planning framework that exploits language models to bridge the gap between linguistic understanding and autonomous driving. VLP enhances autonomous driving systems by strengthening both the source memory foundation and the self-driving car's contextual understanding. VLP achieves state-of-the-art end-to-end planning performance on the challenging NuScenes dataset by achieving  35.9\% and 60.5\% reduction in terms of average L2 error and collision rates, respectively, compared to the previous best method. Moreover, VLP shows improved performance in challenging long-tail scenarios and strong generalization capabilities when faced with new urban environments.

</details>

---

## 18. A Pedestrian is Worth One Prompt: Towards Language Guidance Person Re-Identification

- [ ] A Pedestrian is Worth One Prompt: Towards Language Guidance Person Re-Identification | https://cvpr.thecvf.com/virtual/2024/poster/29372

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29372

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Extensive advancements have been made in person ReID through the mining of semantic information. Nevertheless, existing methods that utilize semantic-parts from a single image modality do not explicitly achieve this goal. Whiteness the impressive capabilities in multimodal understanding  of Vision Language Foundation Model CLIP, a recent two-stage CLIP-based method employs automated prompt engineering to obtain specific textual labels for classifying pedestrians. However, we note that the predefined soft prompts may be inadequate in expressing the entire visual context and struggle to generalize to unseen classes. This paper presents an end-to-end Prompt -driven S emantic G uidance ( PromptSG ) framework that harnesses the rich semantics inherent in CLIP. Specifically, we guide the model to attend to regions that are semantically faithful to the prompt. To provide the personalized language descriptions for specific individuals, we propose learning pseudo tokens that represent specific visual context. This design not only facilitates learning fine-grained attribute information but also can inherently leverage language prompts during inference. Without requiring additional labeling efforts, our PromptSG achieves state-of-the-art by over 10\% on MSMT17 and nearly 5\% on the Market-1501 benchmark.

</details>

---

## 19. SOK-Bench: A Situated Video Reasoning Benchmark with Aligned Open-World Knowledge

- [ ] SOK-Bench: A Situated Video Reasoning Benchmark with Aligned Open-World Knowledge | https://cvpr.thecvf.com/virtual/2024/poster/29403

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29403

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Reasoning from visual dynamics scenes has many real-world applications. However, existing video reasoning benchmarks are still inadequate since they were mainly designed for factual or situated reasoning and rarely involve broader knowledge in the real world.Our work aims to delve deeper into reasoning evaluations, specifically within dynamic, open-world, and structured context knowledge. We propose a new benchmark (SOK-Bench), consisting of 44K questions and 10K situations with instance-level annotations depicted in the videos. The reasoning process is required to understand and apply situated knowledge and general knowledge for problem-solving.To create such a dataset, we propose an automatic and scalable generation method to generate question-answer pairs, knowledge graphs, and rationales by instructing the combinations of LLMs and MLLMs. Concretely, we first extract observable situated entities, relations, and processes from videos for situated knowledge and then extend to open-world knowledge beyond the visible content. The task generation is facilitated through multiple dialogues as iterations and subsequently corrected and refined by our designed self-promptings and demonstrations. With a corpus of both explicit situated facts and implicit commonsense, we generate associated question-answer pairs and reasoning processes, finally followed by manual reviews for quality assurance. We evaluated recent mainstream large vision-language models on the benchmark and found several insightful conclusions. For more information, please refer to our benchmark at www.bobbywu.com/SOKBench.

</details>

---

## 20. ConCon-Chi: Concept-Context Chimera Benchmark for Personalized Vision-Language Tasks

- [ ] ConCon-Chi: Concept-Context Chimera Benchmark for Personalized Vision-Language Tasks | https://cvpr.thecvf.com/virtual/2024/poster/29409

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29409

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While recent Vision-Language (VL) models excel at open-vocabulary tasks, it is unclear how to use them with specific or uncommon concepts. Personalized Text-to-Image Retrieval (TIR) or Generation (TIG) are recently introduced tasks that represent this challenge, where the VL model has to learn a concept from few images and respectively discriminate or generate images of the target concept in arbitrary contexts. We identify the ability to learn new meanings and their compositionality with known ones as two key properties of a personalized system. We show that the available benchmarks offer a limited validation of personalized textual concept learning from images with respect to the above properties and introduce ConCon-Chi as a benchmark for both personalized TIR and TIG, designed to fill this gap.We modelled the new-meaning concepts by crafting chimeric objects and formulating a large, varied set of contexts where we photographed each object. To promote the compositionality assessment of the learned concepts with known contexts, we combined different contexts with the same concept, and vice-versa. We carry out a thorough evaluation of state-of-the-art methods on the resulting dataset. Our study suggests that future work on personalized TIR and TIG methods should focus on the above key properties, and we propose principles and a dataset for their performance assessment. Dataset: https://doi.org/10.48557/QJ1166 and code: https://github.com/hsp-iit/concon-chi_benchmark.

</details>

---

## 21. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models

- [ ] HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29422

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29422

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce “HallusionBench,” a comprehensive benchmark designed for the evaluation of image-context reasoning. This benchmark presents significant challenges to advanced large visual-language models (LVLMs), such as GPT-4V(ision), Gemini Pro Vision, Claude 3, and LLaVA-1.5, by emphasizing nuanced understanding and interpretation of visual data. The benchmark comprises 346 images paired with 1129 questions, all meticulously crafted by human experts. We introduce a novel structure for these visual questions designed to establish control groups. This structure enables us to conduct a quantitative analysis of the models' response tendencies, logical consistency, and various failure modes. In our evaluation on HallusionBench, we benchmarked 15 different models, highlighting a 31.42% question-pair accuracy achieved by the state-of-the-art GPT-4V. Notably, all other evaluated models achieve accuracy below 16%. Moreover, our analysis not only highlights the observed failure modes, including language hallucination and visual illusion but also deepens an under standing of these pitfalls. Our comprehensive case studies within HallusionBench shed light on the challenges of hallucination and illusion in LVLMs. Based on these insights, we suggest potential pathways for their future improvement. The benchmark and codebase can be accessed at https://github.com/tianyilab/HallusionBench.

</details>

---

## 22. Video ReCap: Recursive Captioning of Hour-Long Videos

- [ ] Video ReCap: Recursive Captioning of Hour-Long Videos | https://cvpr.thecvf.com/virtual/2024/poster/29430

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29430

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Most video captioning models are designed to process short video clips of few seconds and output text describing low-level visual concepts (e.g., objects, scenes, atomic actions). However, most real-world videos last for minutes or hours and have a complex hierarchical structure spanning different temporal granularities. We propose Video ReCap, a recursive video captioning model that can process video inputs of dramatically different lengths (from 1 second to 2 hours) and output video captions at multiple hierarchy levels. The recursive video-language architecture exploits the synergy between different video hierarchies and can process hour-long videos efficiently. We utilize a curriculum learning training scheme to learn the hierarchical structure of videos, starting from clip-level captions describing atomic actions, then focusing on segment-level descriptions, and concluding with generating summaries for hour-long videos. Furthermore, we introduce Ego4D-HCap dataset by augmenting Ego4D with 8,267 manually collected long-range video summaries. Our recursive model can flexibly generate captions at different hierarchy levels while also being useful for other complex video understanding tasks, such as VideoQA on EgoSchema. Data, code, and models are publicly available at https://sites.google.com/view/vidrecap.

</details>

---

## 23. Label Propagation for Zero-shot Classification with Vision-Language Models

- [ ] Label Propagation for Zero-shot Classification with Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29452

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29452

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have demonstrated impressive performance on zero-shot classification, i.e. classification when provided merely with a list of class names. In this paper, we tackle the case of zero-shot classification in the presence of unlabeled data. We leverage the graph structure of the unlabeled data and introduce ZLaP, a method based on label propagation (LP) that utilizes geodesic distances for classification. We tailor LP to graphs containing both text and image features and further propose an efficient method for performing inductive inference based on a dual solution and a sparsification step. We perform extensive experiments to evaluate the effectiveness of our method on 14 common datasets and show that ZLaP outperforms the latest related works. Code: https://github.com/vladan-stojnic/ZLaP

</details>

---

## 24. AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving

- [ ] AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving | https://cvpr.thecvf.com/virtual/2024/poster/29457

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29457

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Autonomous driving (AV) systems rely on robust perception models as a cornerstone of safety assurance. However, objects encountered on the road exhibit a long-tailed distribution, with rare or unseen categories posing challenges to a deployed perception model. This necessitates an expensive process of continuously curating and annotating data with significant human effort. We propose to leverage recent advances in vision-language and large language models to design an Automatic Data Engine (AIDE) that automatically identifies issues, efficiently curates data, improves the model through auto-labeling, and verifies the model through generation of diverse scenarios. This process operates iteratively, allowing for continuous self-improvement of the model. We further establish a benchmark for open-world detection on AV datasets to comprehensively evaluate various learning paradigms, demonstrating our method's superior performance at a reduced cost.

</details>

---

## 25. Learning to Segment Referred Objects from Narrated Egocentric Videos

- [ ] Learning to Segment Referred Objects from Narrated Egocentric Videos | https://cvpr.thecvf.com/virtual/2024/poster/29467

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29467

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Egocentric videos provide a first-person perspective of the wearer's activities, involving simultaneous interactions with multiple objects. In this work, we propose the task of weakly-supervised Narration-based Video Object Segmentation (NVOS). Given an egocentric video clip and a narration of the wearer's activities, our aim is to segment object instances mentioned in the narration, without using any spatial annotations during training. Existing weakly-supervised video object grounding methods typically yield bounding boxes for referred objects. In contrast, we propose ROSA, a weakly-supervised pixel-level grounding framework learning alignments between referred objects and segmentation mask proposals. Our model harnesses vision-language models pre-trained on image-text pairs to embed region masks and object phrases. During training, we combine (a) a video-narration contrastive loss that implicitly supervises the alignment between regions and phrases, and (b) a region-phrase contrastive loss based on inferred latent alignments. To address the lack of annotated NVOS datasets in egocentric videos, we create a new evaluation benchmark, VISOR-NVOS, leveraging existing annotations of segmentation masks from VISOR alongside 12k newly-collected, object-based video clip narrations. Our approach achieves state-of-the-art zero-shot pixel-level grounding performance compared to strong baselines under similar supervision. Additionally, we demonstrate generalization capabilities for zero-shot video object grounding on YouCook2, a third-person instructional video dataset.

</details>

---

## 26. ZONE: Zero-Shot Instruction-Guided Local Editing

- [ ] ZONE: Zero-Shot Instruction-Guided Local Editing | https://cvpr.thecvf.com/virtual/2024/poster/29478

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29478

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language models like Stable Diffusion have shown remarkable power in creative image synthesis and editing.However, most existing text-to-image editing methods encounter two obstacles: First, the text prompt needs to be carefully crafted to achieve good results, which is not intuitive or user-friendly. Second, they are insensitive to local edits and can irreversibly affect non-edited regions, leaving obvious editing traces. To tackle these problems, we propose a Zero-shot instructiON-guided local image Editing approach, termed $\texttt{ZONE}$. We first convert the editing intent from the user-provided instruction (e.g., ``make his tie blue") into specific image editing regions through InstructPix2Pix. We then propose a Region-IoU scheme for precise image layer extraction from an off-the-shelf segment model. We further develop an edge smoother based on FFT for seamless blending between the layer and the image.Our method allows for arbitrary manipulation of a specific region with a single instruction while preserving the rest. Extensive experiments demonstrate that our $\texttt{ZONE}$ achieves remarkable local editing results and user-friendliness, outperforming state-of-the-art methods. Code is available at https://github.com/lsl001006/ZONE.

</details>

---

## 27. Siamese Learning with Joint Alignment and Regression for Weakly-Supervised Video Paragraph Grounding

- [ ] Siamese Learning with Joint Alignment and Regression for Weakly-Supervised Video Paragraph Grounding | https://cvpr.thecvf.com/virtual/2024/poster/29499

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29499

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video Paragraph Grounding (VPG) is an emerging task in video-language understanding, which aims at localizing multiple sentences with semantic relations and temporal order from an untrimmed video. However, existing VPG approaches are heavily reliant on a considerable number of temporal labels that are laborious and  time-consuming to acquire. In this work, we introduce and explore Weakly-Supervised Video Paragraph Grounding (WSVPG) to eliminate the need of temporal annotations. Different from previous weakly-supervised grounding frameworks based on multiple instance learning or reconstruction learning for two-stage candidate ranking, we propose a novel siamese learning framework that jointly learns the cross-modal feature alignment and temporal coordinate regression without timestamp labels to achieve concise one-stage localization for WSVPG. Specifically, we devise a Siamese Grounding TRansformer (SiamGTR) consisting of two weight-sharing branches for learning complementary supervision. An Augmentation Branch is utilized for directly regressing the temporal boundaries of a complete paragraph within a pseudo video, and an Inference Branch is designed to capture the order-guided feature correspondence for localizing multiple sentences in a normal video. We demonstrate by extensive experiments that our paradigm has superior practicability and flexibility to achieve efficient weakly-supervised or semi-supervised learning, outperforming state-of-the-art methods trained with the same or stronger supervision.

</details>

---

## 28. Improved Baselines with Visual Instruction Tuning

- [ ] Improved Baselines with Visual Instruction Tuning | https://cvpr.thecvf.com/virtual/2024/poster/29558

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29558

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal models (LMM) have recently shown encouraging progress with visual instruction tuning. In this paper, we present the first systematic study to investigate the design choices of LMMs in a controlled setting under the LLaVA framework. We show that the fully-connected vision-language connector in LLaVA is surprisingly powerful and data-efficient. With simple modifications to LLaVA, namely, using CLIP-ViT-L-336px with an MLP projection and adding academic-task-oriented VQA data with response formatting prompts, we establish stronger baselines that achieve state-of-the-art across 11 benchmarks. Our final 13B checkpoint uses merely 1.2M publicly available data, and finishes full training in ~1 day on a single 8-A100 node. Furthermore, we present some early exploration of open problems in LMMs, including scaling to higher resolution inputs, compositional capabilities, and model hallucination, etc. We hope this makes state-of-the-art LMM research more accessible. Code and model will be publicly available.

</details>

---

## 29. ChatPose: Chatting about 3D Human Pose

- [ ] ChatPose: Chatting about 3D Human Pose | https://cvpr.thecvf.com/virtual/2024/poster/29560

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29560

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce PoseGPT, a framework employing Large Language Models (LLMs) to understand and reason about 3D human poses from images or textual descriptions. Our work is motivated by the human ability to intuitively understand postures from a single image or a brief description, a process that intertwines image interpretation, world knowledge, and an understanding of body language. Traditional human pose estimation methods, whether image-based or text-based, often lack holistic scene comprehension and nuanced reasoning, leading to a disconnect between visual data and its real-world implications. PoseGPT addresses these limitations by embedding SMPL poses as a distinct signal token within a multi-modal LLM, enabling direct generation of 3D body poses from both textual and visual inputs. This approach not only simplifies pose prediction but also empowers LLMs to apply their world knowledge in reasoning about human poses, fostering two advanced tasks: speculative pose generation and reasoning about pose estimation. These tasks involve generating human poses from subtle text queries, possibly accompanied by images, after comprehensive reasoning. We establish benchmarks for these tasks, moving beyond the confines of traditional pose generation and estimation methodologies. Our results show that PoseGPT outperforms existing multimodal LLMs and task-sepcific methods on these newly proposed tasks. Furthermore, PoseGPT's ability to understand and generate 3D human poses based on complex reasoning opens new directions in human pose analysis. We will release the models and training code for research purposes.

</details>

---

## 30. Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships

- [ ] Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships | https://cvpr.thecvf.com/virtual/2024/poster/29563

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29563

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current approaches for 3D scene graph prediction rely on labeled datasets to train models for a fixed set of known object classes and relationship categories. We present Open3DSG, an alternative approach to learn 3D scene graph prediction in an open world without requiring labeled scene graph data. We co-embed the features from a 3D scene graph prediction backbone with the feature space of powerful open world 2D vision language foundation models. This enables us to predict 3D scene graphs from 3D point clouds in a zero-shot manner by querying object classes from an open vocabulary and predicting the inter-object relationships from a grounded LLM with scene graph features and queried object classes as context. Open3DSG is the first 3D point cloud method to predict not only explicit open-vocabulary object classes, but also open-set relationships that are not limited to a predefined label set, making it possible to express rare as well as specific objects and relationships in the predicted 3D scene graph. Our experiments show that Open3DSG is effective at predicting arbitrary object classes as well as their complex inter-object relationships describing spatial, supportive, semantic and comparative relationships.

</details>

---

## 31. Semantic Shield: Defending Vision-Language Models Against Backdooring and Poisoning via Fine-grained Knowledge Alignment

- [ ] Semantic Shield: Defending Vision-Language Models Against Backdooring and Poisoning via Fine-grained Knowledge Alignment | https://cvpr.thecvf.com/virtual/2024/poster/29569

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29569

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In recent years there has been enormous interest in vision-language models trained using self-supervised objectives. However, the use of large-scale  datasets scraped from the web for training also makes these models vulnerable to potential security threats, such as backdooring and poisoning attacks. In this paper, we propose a method for mitigating such attacks on contrastively trained vision-language models. Our approach, Semantic Shield, leverages external knowledge extracted from a language model to prevent models from learning correlations between image regions which lack strong alignment with external knowledge. We do this by imposing constraints to enforce that attention paid by the model to visual regions is proportional to the alignment of those regions with external knowledge.We conduct extensive experiments using a variety of recent backdooring and poisoning attacks on multiple datasets and architectures. Our results clearly demonstrate that our proposed approach is highly effective at defending against such attacks across multiple settings, while maintaining model utility and without requiring any changes at inference time.

</details>

---

## 32. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action

- [ ] Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action | https://cvpr.thecvf.com/virtual/2024/poster/29572

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29572

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present Unified-IO 2, a multimodal and multi-skill unified model capable of following novel instructions. Unified-IO 2 can use text, images, audio, and/or videos as input and can generate text, image, or audio outputs, which is accomplished in a unified way by tokenizing these different inputs and outputs into a shared semantic space that can then be processed by a single encoder-decoder transformer model. Unified-IO 2 is trained from scratch on a custom-built multimodal pre-training corpus and then learns an expansive set of skills through fine-tuning on over 120 datasets, including datasets for segmentation, object detection, image editing, audio localization, video tracking, embodied AI, and 3D detection. To facilitate instruction-following, we add prompts and other data augmentations to these tasks to allow Unified-IO 2 to generalize these skills to new tasks zero-shot.Unified-IO 2 is the first model to be trained on such a diverse and wide-reaching set of skills and unify three separate generation capabilities. Unified-IO 2 achieves state-of-the-art performance on the multi-task GRIT benchmark and achieves strong results on 30 diverse datasets, including SEED-Bench image and video understanding, TIFA image generation, VQA 2.0, ScienceQA, VIMA robotic manipulation, VGG-Sound, and Kinetics-Sounds and can perform unseen tasks and generate free-form responses. We release our model and code to facilitate future work.

</details>

---

## 33. ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts

- [ ] ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts | https://cvpr.thecvf.com/virtual/2024/poster/29580

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29580

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While existing large vision-language multimodal models focus on whole image understanding, there is a prominent gap in achieving region-specific comprehension. Current approaches that use textual coordinates or spatial encodings often fail to provide a user-friendly interface for visual prompting. To address this challenge, we introduce a novel multimodal model capable of decoding arbitrary (free-form) visual prompts. This allows users to intuitively mark images and interact with the model using natural cues like a red bounding box or pointed arrow . Our simple design directly overlays visual markers onto the RGB image, eliminating the need for complex region encodings, yet achieves state-of-the-art performance on region-understanding tasks like Visual7W, PointQA, and Visual Commonsense Reasoning benchmark. Furthermore, we present RegionBench, a comprehensive benchmark to assess the capability of models in understanding visual prompts across multiple dimensions, enabling future research in this domain. Code and demo will be released.

</details>

---

## 34. Jack of All Tasks Master of Many: Designing General-Purpose Coarse-to-Fine Vision-Language Model

- [ ] Jack of All Tasks Master of Many: Designing General-Purpose Coarse-to-Fine Vision-Language Model | https://cvpr.thecvf.com/virtual/2024/poster/29592

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29592

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability of large language models (LLMs) to process visual inputs has given rise to general-purpose vision systems, unifying various vision-language (VL) tasks by instruction tuning. However, due to the enormous diversity in input-output formats in the vision domain, existing general-purpose models fail to successfully integrate segmentation and multi-image inputs with coarse-level tasks into a single framework. In this work, we introduce VistaLLM, a powerful visual system that addresses coarse- and fine-grained VL tasks over single and multiple input images using a unified framework. VistaLLM utilizes an instruction-guided image tokenizer that filters global embeddings using task descriptions to extract compressed and refined features from numerous images. Moreover, VistaLLM employs a gradient-aware adaptive sampling technique to represent binary segmentation masks as sequences, significantly improving over previously used uniform sampling. To bolster the desired capability of VistaLLM, we curate CoinIt, a comprehensive coarse-to-fine instruction tuning dataset with $6.8$M samples. We also address the lack of multi-image grounding datasets by introducing a novel task, AttCoSeg (Attribute-level Co-Segmentation), which boosts the model's reasoning and grounding capability over multiple input images. Extensive experiments on a wide range of V- and VL tasks demonstrate the effectiveness of VistaLLM by achieving consistent state-of-the-art performance over strong baselines across all downstream tasks. Code and data for training VistaLLM will be publicly released.

</details>

---

## 35. From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models

- [ ] From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29596

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29596

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Scene graph generation (SGG) aims to parse a visual scene into an intermediate graph representation for downstream reasoning tasks.Despite recent advancements, existing methods struggle to generate scene graphs with novel visual relation concepts.To address this challenge, we introduce a new open-vocabulary SGG framework based on sequence generation.Our framework leverages vision-language pre-trained models (VLM) by incorporating an image-to-graph generation paradigm.Specifically, we generate scene graph sequences via image-to-text generation with VLM and then construct scene graphs from these sequences.By doing so, we harness the strong capabilities of VLM for open-vocabulary SGG and seamlessly integrate explicit relational modeling for enhancing the VL tasks.Experimental results demonstrate that our design not only achieves superior performance with an open vocabulary but also enhances downstream vision-language task performance through explicit relation modeling knowledge.

</details>

---

## 36. Language-Driven Anchors for Zero-Shot Adversarial Robustness

- [ ] Language-Driven Anchors for Zero-Shot Adversarial Robustness | https://cvpr.thecvf.com/virtual/2024/poster/29619

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29619

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.

</details>

---

## 37. Enhancing Visual Document Understanding with Contrastive Learning in Large Visual-Language Models

- [ ] Enhancing Visual Document Understanding with Contrastive Learning in Large Visual-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29620

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29620

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, the advent of Large Visual-Language Models (LVLMs) has received increasing attention across various domains, particularly in the field of visual document understanding (VDU). Different from conventional vision-language tasks, VDU is specifically concerned with text-rich scenarios containing abundant document elements. Nevertheless, the importance of fine-grained features remains largely unexplored within the community of LVLMs, leading to suboptimal performance in text-rich scenarios. In this paper, we abbreviate it as the fine-grained feature collapse issue. With the aim of filling this gap, we propose a contrastive learning framework, termed Document Object COntrastive learning (DoCo), specifically tailored for the downstream tasks of VDU. DoCo leverages an auxiliary multimodal encoder to obtain the features of document objects and align them to the visual features generated by the vision encoder of LVLM, which enhances visual representation in text-rich scenarios. It can represent that the contrastive learning between the visual holistic representations and the multimodal fine-grained features of document objects can assist the vision encoder in acquiring more effective visual cues, thereby enhancing the comprehension of text-rich documents in LVLMs. We also demonstrate that the proposed DoCo serves as a plug-and-play pre-training method, which can be employed in the pre-training of various LVLMs without inducing any increase in computational complexity during the inference process. Extensive experimental results on multiple benchmarks of VDU reveal that LVLMs equipped with our proposed DoCo can achieve superior performance and mitigate the gap between VDU and generic vision-language tasks.

</details>

---

## 38. MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception

- [ ] MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception | https://cvpr.thecvf.com/virtual/2024/poster/29631

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29631

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

It is a long-lasting goal to design an embodied system that can solve long-horizon open-world tasks in human-like ways. However, existing approaches usually struggle with compound difficulties caused by the logic-aware decomposition and context-aware execution of these tasks. To this end, we introduce MP5, an open-ended multimodal embodied system built upon the challenging Minecraft simulator, which can decompose feasible sub-objectives, design sophisticated situation-aware plans, and perform embodied action control, with frequent communication with a goal-conditioned active perception scheme. Specifically, MP5 is developed on top of recent advances in Multimodal Large Language Models (MLLMs), and the system is modulated into functional modules that can be scheduled and collaborated to ultimately solve pre-defined context- and process-dependent tasks. Extensive experiments prove that MP5 can achieve a 22% success rate on difficult process-dependent tasks and a 91% success rate on tasks that heavily depend on the context.  Moreover, MP5 exhibits a remarkable ability to address many open-ended tasks that are entirely novel.

</details>

---

## 39. Self-Training Large Language Models for Improved Visual Program Synthesis With Visual Reinforcement

- [ ] Self-Training Large Language Models for Improved Visual Program Synthesis With Visual Reinforcement | https://cvpr.thecvf.com/virtual/2024/poster/29639

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29639

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual program synthesis is a promising approach to exploit the reasoning abilities of large language models for compositional computer vision tasks. Previous work has used few-shot prompting with frozen LLMs to synthesize visual programs.Training an LLM to write better visual programs is an attractive prospect, but it is unclear how to accomplish this.No dataset of visual programs for training exists, and acquisition of a visual program dataset cannot be easily crowdsourced due to the need for expert annotators.To get around the lack of direct supervision, we explore improving the program synthesis abilities of a LLM using feedback from interactive experience.We propose a method in which we exploit existing annotations for a vision-language task to improvise a coarse reward signal for that task, treat the LLM as a policy, and apply reinforced self-training to improve the visual program synthesis ability of the LLM for that task. We describe a series of experiments on object detection, compositional visual question answering, and image-text retrieval, and show that in each case, the self-trained LLM outperforms or performs on par with few-shot frozen LLMs that are an order of magnitude larger.

</details>

---

## 40. Honeybee: Locality-enhanced Projector for Multimodal LLM

- [ ] Honeybee: Locality-enhanced Projector for Multimodal LLM | https://cvpr.thecvf.com/virtual/2024/poster/29641

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29641

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In Multimodal Large Language Models (MLLMs), a visual projector plays a crucial role in bridging pre-trained vision encoders with LLMs, enabling profound visual understanding while harnessing the LLMs' robust capabilities. Despite the importance of the visual projector, it has been relatively less explored. In this study, we first identify two essential projector properties: (i) flexibility in managing the number of visual tokens, crucial for MLLMs' overall efficiency, and (ii) preservation of local context from visual features, vital for spatial understanding. Based on these findings, we propose a novel projector design that is both flexible and locality-enhanced, effectively satisfying the two desirable properties. Additionally, we present comprehensive strategies to effectively utilize multiple and multifaceted instruction datasets. Through extensive experiments, we examine the impact of individual design choices. Finally, our proposed MLLM, Honeybee, remarkably outperforms previous state-of-the-art methods across various benchmarks, including MME, MMBench, SEED-Bench, and LLaVA-Bench, achieving significantly higher efficiency. We will release the code and model publicly available.

</details>

---

## 41. VicTR: Video-conditioned Text Representations for Activity Recognition

- [ ] VicTR: Video-conditioned Text Representations for Activity Recognition | https://cvpr.thecvf.com/virtual/2024/poster/29654

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29654

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language models (VLMs) have excelled in the image-domain--- especially in zero-shot settings--- thanks to the availability of vast pretraining data (i.e., paired image-text samples). However for videos, such paired data is not as abundant. Therefore, video-VLMs are usually designed by adapting pretrained image-VLMs to the video-domain, instead of training from scratch. All such recipes rely on augmenting visual embeddings with temporal information (i.e., image $\rightarrow$ video), often keeping text embeddings unchanged or even being discarded. In this paper, we argue the contrary, that better video-VLMs can be designed by focusing more on augmenting text, rather than visual information. More specifically, we introduce Video-conditioned Text Representations (VicTR): a form of text embeddings optimized w.r.t. visual embeddings, creating a more-flexible contrastive latent space. Our model can further make use of freely-available semantic information, in the form of visually-grounded auxiliary text (e.g. object or scene information). We evaluate our model on few-shot, zero-shot (HMDB-51, UCF-101), short-form (Kinetics-400) and long-form (Charades) activity recognition benchmarks, showing strong performance among video-VLMs.

</details>

---

## 42. Image-Text Co-Decomposition for Text-Supervised Semantic Segmentation

- [ ] Image-Text Co-Decomposition for Text-Supervised Semantic Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/29659

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29659

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper addresses text-supervised semantic segmentation, aiming to learn a model capable of segmenting arbitrary visual concepts within images by using only image-text pairs without dense annotations. Existing methods have demonstrated that contrastive learning on image-text pairs effectively aligns visual segments with the meanings of texts. We notice that there is a discrepancy between text alignment and semantic segmentation: A text often consists of multiple semantic concepts, whereas semantic segmentation strives to create semantically homogeneous segments. To address this issue, we propose a novel framework, Image-Text Co-Decomposition (CoDe), where the paired image and text are jointly decomposed into a set of image regions and a set of word segments, respectively, and contrastive learning is developed to enforce region-word alignment. To work with a vision-language model, we present a prompt learning mechanism that derives an extra representation to highlight an image segment or a word segment of interest, with which more effective features can be extracted from that segment. Comprehensive experimental results demonstrate that our method performs favorably against existing text-supervised semantic segmentation methods on six benchmark datasets.

</details>

---

## 43. Hierarchical Intra-modal Correlation Learning for Label-free 3D Semantic Segmentation

- [ ] Hierarchical Intra-modal Correlation Learning for Label-free 3D Semantic Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/29673

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29673

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent methods for label-free 3D semantic segmentation aim to assist 3D model training by leveraging the open-world recognition ability of pre-trained vision language models. However, these methods usually suffer from inconsistent and noisy pseudo-labels provided by the vision language models. To address this issue, we present a hierarchical intra-modal correlation learning framework that captures visual and geometric correlations in 3D scenes at three levels: intra-set, intra-scene, and inter-scene, to help learn more compact 3D representations. We refine pseudo-labels using intra-set correlations within each geometric consistency set and align features of visually and geometrically similar points using intra-scene and inter-scene correlation learning. We also introduce a feedback mechanism to distill the correlation learning capability into the 3D model. Experiments on both indoor and outdoor datasets show the superiority of our method. We achieve a state-of-the-art 36.6% mIoU on the ScanNet dataset, and a 23.0% mIoU on the nuScenes dataset, with improvements of 7.8% mIoU and 2.2% mIoU compared with previous SOTA. We also provide theoretical analysis and qualitative visualization results to discuss the mechanism and conduct thorough ablation studies to support the effectiveness of our framework.

</details>

---

## 44. VISTA-LLAMA: Reducing Hallucination in Video Language Models via Equal Distance to Visual Tokens

- [ ] VISTA-LLAMA: Reducing Hallucination in Video Language Models via Equal Distance to Visual Tokens | https://cvpr.thecvf.com/virtual/2024/poster/29676

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29676

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large video-language models have shown promising results in the field of video understanding.Recent advances in large video-language models have displayed promising outcomes in video comprehension. Current approaches straightforwardly convert video into language tokens and employ large language models for multi-modal tasks.However, this method often leads to the generation of irrelevant content, commonly known as ''hallucination'', as the length of the text increases and the impact of the video diminishes.To address this problem, we propose Vista-LLaMA , a novel framework that maintains the consistent distance between all visual tokens and any language tokens, irrespective of the generated text length.Vista-LLaMA omits relative position encoding when determining attention weights between visual and text tokens, retaining the position encoding for text and text tokens. This amplifies the effect of visual tokens on text generation, especially when the relative distance is longer between visual and text tokens. The proposed attention mechanism significantly reduces the chance of producing irrelevant text related to the video content.Furthermore, we present a sequential visual projector that projects the current video frame into tokens of language space with the assistance of the previous frame. This approach not only captures the temporal relationship within the video, but also allows less visual tokens to encompass the entire video.Our approach significantly outperforms various previous methods (e.g., Video-ChatGPT, MovieChat) on four challenging open-ended video question answering benchmarks. We reach an accuracy of 60.7 on the zero-shot NExT-QA and 60.5 on the zero-shot MSRVTT-QA, setting a new state-of-the-art performance.

</details>

---

## 45. MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World

- [ ] MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World | https://cvpr.thecvf.com/virtual/2024/poster/29685

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29685

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Human beings possess the capability to multiply a mélange of multisensory cues while actively exploring and interacting with the 3D world.Current multi-modal large language models, however, passively absorb sensory data as inputs, lacking the capacity to actively interact with the objects in the 3D environment and dynamically collect their multisensory information.To usher in the study of this area,  we propose MultiPLY, a multisensory embodied LLM that could incorporate multisensory interactive data, including visual, audio, tactile, and thermal information into large language models, thereby establishing the correlation among words, actions, and percepts. To this end, we first collect Multisensory Universe, a large-scale multisensory interaction dataset comprising 500k data by deploying an LLM-powered embodied agent to engage with the 3D environment. To perform instruction tuning with pre-trained LLM on such generated data, we first encode the 3D scene as abstracted object-centric representations and then introduce action tokens denoting that the embodied agent takes the actions within the environment, and state tokens that represent the multisensory state observations of the agent at each time step.    In the inference time, MultiPLY could generate action tokens, instructing the agent to take the action in the environment and obtain the next multisensory state observation. The observation is then appended back to the LLM via state tokens to generate subsequent text or action tokens. We demonstrate MultiPLY outperforms baselines by a large margin through a diverse set of embodied tasks involving object retrieval, tool use, multisensory captioning, and task decomposition.

</details>

---

## 46. The Devil is in the Fine-Grained Details: Evaluating Open-Vocabulary Object Detectors for Fine-Grained Understanding

- [ ] The Devil is in the Fine-Grained Details: Evaluating Open-Vocabulary Object Detectors for Fine-Grained Understanding | https://cvpr.thecvf.com/virtual/2024/poster/29694

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29694

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large vision-language models enabled visual object detection in open-vocabulary scenarios, where object classes are defined in free-text formats during inference.In this paper, we aim to probe the state-of-the-art methods for open-vocabulary object detection to determine to what extent they understand fine-grained properties of objects and their parts.To this end, we introduce an evaluation protocol based on dynamic vocabulary generation to test whether models detect, discern, and assign the correct fine-grained description to objects in the presence of hard-negative classes.We contribute with a benchmark suite of increasing difficulty and probing different properties like color, pattern, and material.We further enhance our investigation by evaluating several state-of-the-art open-vocabulary object detectors using the proposed protocol and find that most existing solutions, which shine in standard open-vocabulary benchmarks, struggle to accurately capture and distinguish finer object details.We conclude the paper by highlighting the limitations of current methodologies and exploring promising research directions to overcome the discovered drawbacks. Data and code are available at https://lorebianchi98.github.io/FG-OVD .

</details>

---

## 47. De-Diffusion Makes Text a Strong Cross-Modal Interface

- [ ] De-Diffusion Makes Text a Strong Cross-Modal Interface | https://cvpr.thecvf.com/virtual/2024/poster/29703

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29703

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We demonstrate text as a strong cross-modal interface. Rather than relying on deep embeddings to connect image and language as the interface representation, our approach represents an image as text, from which we enjoy the interpretability and flexibility inherent to natural language. We employ an autoencoder that uses a pre-trained text-to-image diffusion model for decoding. The encoder is trained to transform an input image into text, which is then fed into the fixed text-to-image diffusion decoder to reconstruct the original input -- a process we term De-Diffusion. Experiments validate both the precision and comprehensiveness of De-Diffusion text representing images, such that it can be readily ingested by off-the-shelf text-to-image tools and LLMs for diverse multi-modal tasks. For example, a single De-Diffusion model can generalize to provide transferable prompts for different text-to-image tools, and also achieves a new state of the art on open-ended vision-language tasks by simply prompting LLMs with few-shot examples. Project page: https://dediffusion.github.io/

</details>

---

## 48. Pre-trained Model Guided Fine-Tuning for Zero-Shot Adversarial Robustness

- [ ] Pre-trained Model Guided Fine-Tuning for Zero-Shot Adversarial Robustness | https://cvpr.thecvf.com/virtual/2024/poster/29705

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29705

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained vision-language models like CLIP have demonstrated impressive performance across various tasks, and exhibit remarkable zero-shot generalization capability, while they are also vulnerable to imperceptible adversarial examples. Existing works typically employ adversarial training (fine-tuning) as a defense method against adversarial examples. However, direct application to the CLIP model may result in overfitting, compromising the model's capacity for generalization.In this paper, we propose Pre-trained Model Guided Adversarial Fine-Tuning (PMG-AFT) method, which leverages supervision from the original pre-trained model by carefully designing an auxiliary branch, to enhance the model's zero-shot adversarial robustness.Specifically, PMG-AFT minimizes the distance between the features of adversarial examples in the target model and those in the pre-trained model, aiming to preserve the generalization features already captured by the pre-trained model.Extensive Experiments on 15 zero-shot datasets demonstrate that PMG-AFT significantly outperforms the state-of-the-art method, improving the top-1 robust accuracy by an average of 4.99\%.Furthermore, our approach consistently improves clean accuracy by an average of 8.72\%.Our code is available at \href{https://github.com/serendipity1122/Pre-trained-Model-Guided-Fine-Tuning-for-Zero-Shot-Adversarial-Robustness}{here}.\footnote{https://github.com/serendipity1122/Pre-trained-Model-Guided-Fine-Tuning-for-Zero-Shot-Adversarial-Robustness}

</details>

---

## 49. SAI3D: Segment Any Instance in 3D Scenes

- [ ] SAI3D: Segment Any Instance in 3D Scenes | https://cvpr.thecvf.com/virtual/2024/poster/29714

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29714

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Advancements in 3D instance segmentation have traditionally been tethered to the availability of annotated datasets, limiting their application to a narrow spectrum of object categories. Recent efforts have sought to harness vision-language models like CLIP for open-set semantic reasoning, yet these methods struggle to distinguish between objects of the same categories and rely on specific prompts that are not universally applicable. In this paper, we introduce SAI3D, a novel zero-shot 3D instance segmentation approach that synergistically leverages geometric priors and semantic cues derived from Segment Anything Model (SAM). Our method partitions a 3D scene into geometric primitives, which are then progressively merged into 3D instance segmentations that are consistent with the multi-view SAM masks. Moreover, we design a hierarchical region-growing algorithm with a dynamic thresholding mechanism, which largely improves the robustness of fine-grained 3D scene parsing. Empirical evaluations on ScanNet, Matterport3D and the more challenging ScanNet++ datasets demonstrate the superiority of our approach. Notably, SAI3D outperforms existing open-vocabulary baselines and even surpasses fully-supervised methods in class-agnostic segmentation on ScanNet++. Our project page is at https://yd-yin.github.io/SAI3D/.

</details>

---

## 50. Exploring Region-Word Alignment in Built-in Detector for Open-Vocabulary Object Detection

- [ ] Exploring Region-Word Alignment in Built-in Detector for Open-Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2024/poster/29717

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29717

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary object detection aims to detect novel categories that are independent from the base categories used during training. Most modern methods adhere to the paradigm of learning vision-language space from a large-scale multi-modal corpus and subsequently transferring the acquired knowledge to off-the-shelf detectors like Faster-RCNN. However, information attenuation or destruction may occur during the process of knowledge transfer due to the domain gap, hampering the generalization ability on novel categories. To mitigate this predicament, in this paper, we present a novel framework named BIND, standing for Bulit-IN Detector, to eliminate the need for module replacement or knowledge transfer to off-the-shelf detectors. Specifically, we design a two-stage training framework with an Encoder-Decoder structure. In the first stage, an image-text dual encoder is trained to learn region-word alignment from a corpus of image-text pairs. In the second stage, a DETR-style decoder is trained to perform detection on annotated object detection datasets. In contrast to conventional manually designed non-adaptive anchors, which generate numerous redundant proposals, we develop an anchor proposal network that generates anchor proposals with high likelihood based on candidates adaptively, thereby substantially improving detection efficiency. Experimental results on two public benchmarks, COCO and LVIS, demonstrate that our method stands as a state-of-the-art approach for open-vocabulary object detection. The code and models will be publicly available.

</details>

---

## 51. Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Compositional Understanding

- [ ] Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Compositional Understanding | https://cvpr.thecvf.com/virtual/2024/poster/29738

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29738

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs), such as CLIP, exhibit strong image-text comprehension abilities, facilitating advances in several downstream tasks such as zero-shot image classification, image-text retrieval, and text-to-image generation. However, the compositional reasoning abilities of existing VLMs remains subpar. The root of this limitation lies in the inadequate alignment between the images and captions in the pretraining datasets. Additionally, the current contrastive learning objective fails to focus on fine-grained grounding components like relations, actions, and attributes, resulting in "bag-of-words" representations. We introduce a simple and effective method to improve compositional reasoning in VLMs. Our method better leverages available datasets by refining and expanding the standard image-text contrastive learning framework. Our approach does not require specific annotations and does not incur extra parameters. When integrated with CLIP, our technique yields notable improvement over state-of-the-art baselines across five vision-language compositional benchmarks.

</details>

---

## 52. GOV-NeSF: Generalizable Open-Vocabulary Neural Semantic Fields

- [ ] GOV-NeSF: Generalizable Open-Vocabulary Neural Semantic Fields | https://cvpr.thecvf.com/virtual/2024/poster/29743

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29743

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in vision-language foundation models have significantly enhanced open-vocabulary 3D scene understanding. However, the generalizability of existing methods is constrained due to their framework designs and their reliance on 3D data. We address this limitation by introducing Generalizable Open-Vocabulary Neural Semantic Fields (GOV-NeSF), a novel approach offering a generalizable implicit representation of 3D scenes with open-vocabulary semantics. We aggregate the geometry-aware features using a cost volume, and propose a Multi-view Joint Fusion module to aggregate multi-view features through a cross-view attention mechanism, which effectively predicts view-specific blending weights for both colors and open-vocabulary features. Remarkably, our GOV-NeSF exhibits state-of-the-art performance in both 2D and 3D open-vocabulary semantic segmentation, eliminating the need for ground truth semantic labels or depth priors, and effectively generalize across scenes and datasets without fine-tuning.

</details>

---

## 53. Can’t Make an Omelette Without Breaking Some Eggs: Plausible Action Anticipation Using Large Video-Language Models

- [ ] Can’t Make an Omelette Without Breaking Some Eggs: Plausible Action Anticipation Using Large Video-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29758

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29758

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce PlausiVL, a large video-language model for anticipating action sequences that are plausible in the real-world. While significant efforts have been made towards anticipating future actions, prior approaches do not take into account the aspect of plausibility in an action sequence. To address this limitation, we explore the generative capability of a large video-language model in our work and further, develop the understanding of plausibility in an action sequence by introducing two objective functions, a counterfactual-based plausible action sequence learning loss and a long-horizon action repetition loss. We utilize temporal logical constraints as well as verb-noun action pair logical constraints to create implausible/counterfactual action sequences and use them to train the model with plausible action sequence learning loss. This loss helps the model to differentiate between plausible and not plausible action sequences and also helps the model to learn implicit temporal cues crucial for the task of action anticipation. The long-horizon action repetition loss puts a higher penalty on the actions that are more prone to repetition over a longer temporal window. With this penalization, the model is able to generate diverse, plausible action sequences. We evaluate our approach on two large-scale datasets, Ego4D and EPIC-Kitchens-100 and show improvements on the task of action anticipation.

</details>

---

## 54. Enhancing Vision-Language Pre-training with Rich Supervisions

- [ ] Enhancing Vision-Language Pre-training with Rich Supervisions | https://cvpr.thecvf.com/virtual/2024/poster/29766

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29766

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose Strongly Supervised pre-training with ScreenShots (S4) - a novel pre-training paradigm for Vision-Language Models using data from large-scale web screenshot rendering. Using web screenshot unlock a treasure trove of visual and textual cues that are simply not present in using image-text pairs. In S4, we leverage the inherent tree-structure hierarchy of HTML elements and the spatial localization to carefully design 10 pre-training tasks with large scale annotated data. These tasks resembles downstream tasks across different domains and the annotations are cheap to obtain. We demonstrate that, comparing to current screenshot pre-training objectives, our innovative pre-training method significantly enhances performance of image-to-text model in nine varied and popular downstream tasks - up to 76.1% improvements on Table Detection, and at least 1% on Widget Captioning.

</details>

---

## 55. FFF: Fixing Flawed Foundations in Contrastive Pre-Training Results in Very Strong Vision-Language Models

- [ ] FFF: Fixing Flawed Foundations in Contrastive Pre-Training Results in Very Strong Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29774

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29774

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite noise and caption quality having been acknowledged as important factors impacting vision-language contrastive pre-training, in this paper, we show that the full potential of improving the training process by addressing such issues is yet to be realized. Specifically, we firstly study and analyze two issues affecting training: incorrect assignment of negative pairs, and low caption quality and diversity. Then, we devise effective solutions for addressing both problems, which essentially require training with multiple true positive pairs. Finally, we propose training with sigmoid loss to address such a requirement. We show very large gains over the current state-of-the-art for both image recognition ($\sim +6\%$ on average over 11 datasets) and image retrieval ($\sim +19\%$ on Flickr30k and $\sim +15\%$ on MSCOCO).

</details>

---

## 56. Geometrically-driven Aggregation for Zero-shot 3D Point Cloud Understanding

- [ ] Geometrically-driven Aggregation for Zero-shot 3D Point Cloud Understanding | https://cvpr.thecvf.com/virtual/2024/poster/29775

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29775

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot 3D point cloud understanding can be achieved via 2D Vision-Language Models (VLMs). Existing strategies directly map VLM representations from 2D pixels of rendered or captured views to 3D points, overlooking the inherent and expressible point cloud geometric structure. Geometrically similar or close regions can be exploited for bolstering point cloud understanding as they are likely to share semantic information. To this end, we introduce the first training-free aggregation technique that leverages the point cloud's 3D geometric structure to improve the quality of the transferred VLM representations. Our approach operates iteratively, performing local-to-global aggregation based on geometric and semantic point-level reasoning. We benchmark our approach on three downstream tasks, including classification, part segmentation, and semantic segmentation, with a variety of datasets representing both synthetic/real-world, and indoor/outdoor scenarios. Our approach achieves new state-of-the-art results in all benchmarks.We will release the source code publicly.

</details>

---

## 57. CLIB-FIQA: Face Image Quality Assessment with Confidence Calibration

- [ ] CLIB-FIQA: Face Image Quality Assessment with Confidence Calibration | https://cvpr.thecvf.com/virtual/2024/poster/29781

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29781

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Face Image Quality Assessment (FIQA) is pivotal for guaranteeing the accuracy of face recognition in unconstrained environments. Recent progress in deep quality-fitting-based methods which train models to align with quality anchors, has shown promise in FIQA. However, these methods heavily depend on a recognition model to yield quality anchors and indiscriminately treat the confidence of inaccurate anchors as equivalent to that of accurate ones during the FIQA model training, leading to a fitting bottleneck issue. This paper seeks a solution by putting forward the Confidence-Calibrated Face Image Quality Assessment (CLIB-FIQA) approach, underpinned by the synergistic interplay between the quality anchors and objective quality factors such as blur, pose, expression, occlusion, and illumination. Specifically, we devise a joint learning framework built upon the vision-language alignment model, which leverages the joint distribution with multiple quality factors to facilitate the quality fitting of the FIQA model. Furthermore, to alleviate the issue of the model placing excessive trust in inaccurate quality anchors, we propose a confidence calibration method to correct the quality distribution by exploiting to the fullest extent of these objective quality factors characterized as the merged-factor distribution during training. Experimental results on eight datasets reveal the superior performance of the proposed method. The source code will be made publicly available.

</details>

---

## 58. Towards Language-Driven Video Inpainting via Multimodal Large Language Models

- [ ] Towards Language-Driven Video Inpainting via Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29810

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29810

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a new task -- language-driven video inpainting, which uses natural language instructions to guide the inpainting process. This approach overcomes the limitations of traditional video inpainting methods that depend on manually labeled binary masks, a process often tedious and labor-intensive. We present the Remove Objects from Videos by Instructions (ROVI) dataset, containing 5,650 videos and 9,091 inpainting results, to support training and evaluation for this task. We also propose a novel diffusion-based language-driven video inpainting framework, the first end-to-end baseline for this task, integrating Multimodal Large Language Models to understand and execute complex language-based inpainting requests effectively. Our comprehensive results showcase the dataset's versatility and the model's effectiveness in various language-instructed inpainting scenarios. We have made datasets, code, and models publicly available at \url{https://github.com/jianzongwu/Language-Driven-Video-Inpainting}.

</details>

---

## 59. Hyperbolic Learning with Synthetic Captions for Open-World Detection

- [ ] Hyperbolic Learning with Synthetic Captions for Open-World Detection | https://cvpr.thecvf.com/virtual/2024/poster/29814

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29814

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-world detection poses significant challenges, as it requires the detection of any object using either object class labels or free-form texts. Existing related works often use large-scale manual annotated caption datasets for training, which are extremely expensive to collect. Instead, we propose to transfer knowledge from vision-language models (VLMs) to enrich the open-vocabulary descriptions automatically. Specifically, we bootstrap dense synthetic captions using pre-trained VLMs to provide rich descriptions on different regions in images, and incorporate these captions to train a novel detector that generalizes to novel concepts. To mitigate the noise caused by hallucination in synthetic captions, we also propose a novel hyperbolic vision-language learning approach to impose a hierarchy between visual and caption embeddings. We call our detector ``HyperLearner''. We conduct extensive experiments on a wide variety of open-world detection benchmarks (COCO, LVIS, Object Detection in the Wild, RefCOCO) and our results show that our model consistently outperforms existing state-of-the-art methods, such as GLIP, GLIPv2 and Grounding DINO, when using the same backbone.

</details>

---

## 60. SocialCounterfactuals: Probing and Mitigating Intersectional Social Biases in Vision-Language Models with Counterfactual Examples

- [ ] SocialCounterfactuals: Probing and Mitigating Intersectional Social Biases in Vision-Language Models with Counterfactual Examples | https://cvpr.thecvf.com/virtual/2024/poster/29820

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29820

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While vision-language models (VLMs) have achieved remarkable performance improvements recently, there is growing evidence that these models also posses harmful biases with respect to social attributes such as gender and race. Prior studies have primarily focused on probing such bias attributes individually while ignoring biases associated with intersections between social attributes. This could be due to the difficulty of collecting an exhaustive set of image-text pairs for various combinations of social attributes. To address this challenge, we employ text-to-image diffusion models to produce counterfactual examples for probing intersectional social biases at scale. Our approach utilizes Stable Diffusion with cross attention control to produce sets of counterfactual image-text pairs that are highly similar in their depiction of a subject (e.g., a given occupation) while differing only in their depiction of intersectional social attributes (e.g., race & gender). Through our over-generate-then-filter methodology, we produce SocialCounterfactuals, a high-quality dataset containing 171k image-text pairs for probing intersectional biases related to gender, race, and physical characteristics. We conduct extensive experiments to demonstrate the usefulness of our generated dataset for probing and mitigating intersectional social biases in state-of-the-art VLMs.

</details>

---

## 61. Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models

- [ ] Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/29843

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29843

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

From an enormous amount of image-text pairs, large-scale vision-language models (VLMs) learn to implicitly associate image regions with words, which is vital for tasks such as image captioning and visual question answering. However, leveraging such pre-trained models for open-vocabulary semantic segmentation remains a challenge.In this paper, we propose a simple, yet extremely effective, training-free technique, Plug-and-Play Open-Vocabulary Semantic Segmentation (PnP-OVSS) for this task. PnP-OVSS leverages a VLM with direct text-to-image cross-attention and an image-text matching loss to produce semantic segmentation. However, cross-attention alone tends to over-segment, whereas cross-attention plus GradCAM tend to under-segment. To alleviate this issue, we introduce Salience Dropout; by iteratively dropping patches that the model is most attentive to, we are able to better resolve the entire extent of the segmentation mask. Compared to existing techniques, the proposed method does not require any neural network training and performs hyperparameter tuning without the need for any segmentation annotations, even for a validation set. PnP-OVSS demonstrates substantial improvements over a comparable baseline (+29.4\% on Pascal VOC, +13.2\% on Pascal Context, +14.0\% mIoU on MS COCO, +2.4\% on COCO Stuff) and even outperforms most baselines that conduct additional network training on top of pretrained VLMs.

</details>

---

## 62. Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts

- [ ] Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts | https://cvpr.thecvf.com/virtual/2024/poster/29846

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29846

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper explores the problem of Generalist Anomaly Detection (GAD), aiming to train one single detection model that can generalize to detect anomalies in diverse datasets from different application domains without any further training on the target data. Some recent studies have shown that large pre-trained Visual-Language Models (VLMs) like CLIP have strong generalization capabilities on detecting industrial defects from various datasets, but their methods rely heavily on handcrafted text prompts about defects, making them difficult to generalize to anomalies in other applications, e.g., medical image anomalies or semantic anomalies in natural images. In this work, we propose to train a GAD model with few-shot normal images as sample prompts for AD on diverse datasets on the fly. To this end, we introduce a novel approach that learns an in-context residual learning model for GAD, termed InCTRL. It is trained on an auxiliary dataset to discriminate anomalies from normal samples based on a holistic evaluation of the residuals between query images and few-shot normal sample prompts. Regardless of the datasets, per definition of anomaly, larger residuals are expected for anomalies than normal samples, thereby enabling InCTRL to generalize across different domains without further training. Comprehensive experiments on nine AD datasets are performed to establish a GAD benchmark that encapsulate the detection of industrial defect anomalies, medical anomalies, and semantic anomalies in both one-vs-all and multi-class setting, on which InCTRL is the best performer and significantly outperforms state-of-the-art competing methods. Code is available at https://github.com/mala-lab/InCTRL.

</details>

---

## 63. LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding Reasoning and Planning

- [ ] LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding Reasoning and Planning | https://cvpr.thecvf.com/virtual/2024/poster/29865

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29865

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in Large Multimodal Models (LMM) has opened up great possibilities for various applications in the field of human-machine interactions. However, developing LMMs that can comprehend, reason, and plan in complex and diverse 3D environments remains a challenging topic, especially considering the demand for understanding permutation-invariant point cloud representations of the 3D scene. Existing works seek help from multi-view images by projecting 2D features to 3D space, which inevitably leads to huge computational overhead and performance degradation. In this paper, we present LL3DA, a Large Language 3D Assistant that takes point cloud as the direct input and responds to both text instructions and visual interactions. The additional visual interaction enables LMMs to better comprehend human interactions with the 3D environment and further remove the ambiguities within plain texts. Experiments show that LL3DA achieves remarkable results and surpasses various 3D vision-language models on both 3D Dense Captioning and 3D Question Answering.

</details>

---

## 64. Pink: Unveiling the Power of Referential Comprehension for Multi-modal LLMs

- [ ] Pink: Unveiling the Power of Referential Comprehension for Multi-modal LLMs | https://cvpr.thecvf.com/virtual/2024/poster/29891

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29891

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) have shown remarkable capabilities in various multi-modal tasks. Nevertheless, their performance in fine-grained image understanding tasks is still limited. To address this issue, this paper proposes a new framework that aims to enhance the fine-grained image understanding abilities of MLLMs. Specifically, we present a new method for constructing the instruction tuning dataset at a low cost by leveraging annotations in existing datasets. A self-consistent bootstrapping method is also introduced to extend existing dense object annotations into high-quality referring-expression-bounding-box pairs. These methods enable the generation of high-quality instruction data which includes a wide range of fundamental abilities essential for fine-grained image perception. Moreover, we argue that the visual encoder should be tuned during instruction tuning to mitigate the gap between full image perception and fine-grained image perception. Experimental results demonstrate the superior performance of our method. For instance, our model exhibits a 5.2% accuracy improvement over Qwen-VL on GQA and surpasses the accuracy of Kosmos-2 by 24.7% on RefCOCO_val. We also attain the top rank on the leaderboard of MMBench. This promising performance is achieved by training on only publicly available data, making it easily reproducible. We will release the models, datasets, and codes for further research.

</details>

---

## 65. Exploring the Potential of Large Foundation Models for Open-Vocabulary HOI Detection

- [ ] Exploring the Potential of Large Foundation Models for Open-Vocabulary HOI Detection | https://cvpr.thecvf.com/virtual/2024/poster/29911

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29911

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary human-object interaction (HOI) detection, which is concerned with the problem of detecting novel HOIs guided by natural language, is crucial for understanding human-centric scenes. However, prior zero-shot HOI detectors often employ the same levels of feature maps to model HOIs with varying distances, leading to suboptimal performance in scenes containing human-object pairs with a wide range of distances.In addition, these detectors primarily rely on category names and overlook the rich contextual information that language can provide, which is essential for capturing open vocabulary concepts that are typically rare and not well-represented by category names alone.In this paper, we introduce a novel end-to-end open vocabulary HOI detection framework with conditional multi-level decoding and fine-grained semantic enhancement~(CMD-SE), harnessing the potential of Visual-Language Models (VLMs). Specifically, we propose to model human-object pairs with different distances with different levels of feature maps by incorporating a soft constraint during the bipartite matching process. Furthermore, by leveraging large language models (LLMs) such as GPT models, we exploit their extensive world knowledge to generate descriptions of human body part states for various interactions. Then we integrate the generalizable and fine-grained semantics of human body parts to improve interaction recognition.Experimental results on two datasets, SWIG-HOI and HICO-DET, demonstrate that our proposed method achieves state-of-the-art results in open vocabulary HOI detection.

</details>

---

## 66. SRTube: Video-Language Pre-Training with Action-Centric Video Tube Features and Semantic Role Labeling

- [ ] SRTube: Video-Language Pre-Training with Action-Centric Video Tube Features and Semantic Role Labeling | https://cvpr.thecvf.com/virtual/2024/poster/29914

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29914

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In recent years, large-scale video-language pre-training (VidLP) has received considerable attention for its effectiveness in relevant tasks. In this paper, we propose a novel action-centric VidLP framework that employs video tube features for temporal modeling and language features based on semantic role labeling (SRL). Our video encoder generates multiple tube features along object trajectories, identifying action-related regions within videos, to overcome the limitations of existing query-driven attention mechanisms. Additionally, our text encoder incorporates high-level, action-related language knowledge, previously underutilized in current VidLP models. The SRL captures action-verbs and related semantics among objects in sentences and enhances the ability to perform instance-level text matching, thus enriching the cross-modal (CM) alignment process. We also introduce two novel pre-training objectives and a self-supervision strategy to produce a more faithful CM representation. Experimental results demonstrate that our method outperforms existing VidLP frameworks in various downstream tasks and datasets, establishing our model a baseline in the modern VidLP framework.

</details>

---

## 67. OVFoodSeg: Elevating Open-Vocabulary Food Image Segmentation via Image-Informed Textual Representation

- [ ] OVFoodSeg: Elevating Open-Vocabulary Food Image Segmentation via Image-Informed Textual Representation | https://cvpr.thecvf.com/virtual/2024/poster/29936

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29936

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the realm of food computing, segmenting ingredients from images poses substantial challenges due to the large intra-class variance among the same ingredients, the emergence of new ingredients, and the high annotation costs associated with large food segmentation datasets.Existing approaches primarily utilize a closed-vocabulary and static text embeddings setting. These methods often fall short in effectively handling the ingredients, particularly new and diverse ones. In response to these limitations, we introduce OVFoodSeg, a framework that adopts an open-vocabulary setting and enhances text embeddings with visual context.By integrating vision-language models (VLMs), our approach enriches text embedding with image-specific information through two innovative modules, \eg, an image-to-text learner FoodLearner and an Image-Informed Text Encoder.The training process of OVFoodSeg is divided into two stages: the pre-training of FoodLearner and the subsequent learning phase for segmentation. The pre-training phase equips FoodLearner with the capability to align visual information with corresponding textual representations that are specifically related to food, while the second phase adapts both the FoodLearner and the Image-Informed Text Encoder for the segmentation task.By addressing the deficiencies of previous models, OVFoodSeg demonstrates a significant improvement, achieving an 4.9\% increase in mean Intersection over Union (mIoU) on the FoodSeg103 dataset, setting a new milestone for food image segmentation.

</details>

---

## 68. Generalizable Whole Slide Image Classification with Fine-Grained Visual-Semantic Interaction

- [ ] Generalizable Whole Slide Image Classification with Fine-Grained Visual-Semantic Interaction | https://cvpr.thecvf.com/virtual/2024/poster/29944

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29944

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Whole Slide Image (WSI) classification is often formulated as a Multiple Instance Learning (MIL) problem. Recently, Vision-Language Models (VLMs) have demonstrated remarkable performance in WSI classification. However, existing methods leverage coarse-grained pathogenetic descriptions for visual representation supervision, which are insufficient to capture the complex visual appearance of pathogenetic images, hindering the generalizability of models on diverse downstream tasks. Additionally, processing high-resolution WSIs can be computationally expensive. In this paper, we propose a novel ``Fine-grained Visual-Semantic Interaction" (FiVE) framework for WSI classification. It is designed to enhance the model's generalizability by leveraging the interaction between localized visual patterns and fine-grained pathological semantics. Specifically, with meticulously designed queries, we start by utilizing a large language model to extract fine-grained pathological descriptions from various non-standardized raw reports. The output descriptions are then reconstructed into fine-grained labels used for training. By introducing a Task-specific Fine-grained Semantics (TFS) module, we enable prompts to capture crucial visual information in WSIs, which enhances representation learning and augments generalization capabilities significantly. Furthermore, given that pathological visual patterns are redundantly distributed across tissue slices, we sample a subset of visual instances during training. Our method demonstrates robust generalizability and strong transferability, dominantly outperforming the counterparts on the TCGA Lung Cancer dataset with at least 9.19\% higher accuracy in few-shot experiments. The code is available at: https://github.com/ls1rius/WSI_FiVE.

</details>

---

## 69. Efficient Vision-Language Pre-training by Cluster Masking

- [ ] Efficient Vision-Language Pre-training by Cluster Masking | https://cvpr.thecvf.com/virtual/2024/poster/29946

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29946

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The quest for optimal vision-language pretraining strategies has led to the exploration of masking techniques as a way to enhance data efficiency. Previous approaches include random masking and semantic masking, the latter requiring the retention or exclusion of patches in areas with similar semantics. Despite its effectiveness, semantic masking often needs an additional, complex model for identifying semantically related patches, increasing computational demands. Our method utilizes naturally emerging clusters within images unlike other approaches using text supervision. We employ random clusters of image patches for masking, utilizing the raw RGB values of patches as the feature representation. This method capitalizes on the observation that basic visual similarity measures can effectively identify coherent visual structures, such as parts of objects. Our approach, therefore, combines the computational efficiency of random patch dropping with the enhanced performance achieved through masking coherent visual structures.

</details>

---

## 70. Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning

- [ ] Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning | https://cvpr.thecvf.com/virtual/2024/poster/29964

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29964

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recovering the 3D scene geometry from a single view is a fundamental yet ill-posed problem in computer vision. While classical depth estimation methods infer only a 2.5D scene representation limited to the image plane, recent approaches based on radiance fields reconstruct a full 3D representation. However, these methods still struggle with occluded regions since inferring geometry without visual observation requires (i) semantic knowledge of the surroundings, and (ii) reasoning about spatial context. We propose KYN, a novel method for single-view scene reconstruction that reasons about semantic and spatial context to predict each point's density. We introduce a vision-language modulation module to enrich point features with fine-grained semantic information. We aggregate point representations across the scene through a language-guided spatial attention mechanism to yield per-point density predictions aware of the 3D semantic context. We show that KYN improves 3D shape recovery compared to predicting density for each 3D point in isolation. We achieve state-of-the-art results in scene and object reconstruction on KITTI-360, and show improved zero-shot generalization compared to prior work. Project page: https://ruili3.github.io/kyn.

</details>

---

## 71. Grounding Everything: Emerging Localization Properties in Vision-Language Transformers

- [ ] Grounding Everything: Emerging Localization Properties in Vision-Language Transformers | https://cvpr.thecvf.com/virtual/2024/poster/29967

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29967

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language foundation models have shown remarkable performance in various zero-shot settings such as image retrieval,  classification, or captioning. But so far, those models seem to fall behind when it comes to zero-shot localization of referential expressions and objects in images. As a result, they need to be fine-tuned for this task.In this paper we show that pretrained vision-language (VL) models allow for zero-shot open-vocabulary object localization without any fine-tuning. To leverage those capabilities, we propose a Grounding Everything Module (GEM) that generalizes the idea of value-value attention introduced by CLIPSurgery to a self-self attention path. We show that the concept of self-self attention corresponds to clustering, thus enforcing groups of tokens arising from the same object to be similar while preserving the alignment with the language space. To further guide the group formation, we propose a set of regularizations that allows the model to finally generalize across datasets and backbones. We evaluate the proposed GEM framework on various benchmark tasks and datasets for semantic segmentation. It shows that GEM not only outperforms other training-free open-vocabulary localization methods, but also achieves state-of-the-art results on the recently proposed OpenImagesV7 large-scale segmentation benchmark.

</details>

---

## 72. Prompt Learning via Meta-Regularization

- [ ] Prompt Learning via Meta-Regularization | https://cvpr.thecvf.com/virtual/2024/poster/29971

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29971

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models have shown impressive success on various computer vision tasks with their zero-shot generalizability. Recently, prompt learning approaches have been explored to efficiently and effectively adapt the vision-language models to a variety of downstream tasks. However, most existing prompt learning methods suffer from \textit{task overfitting} since the general knowledge of the pre-trained vision language models is forgotten while the prompts are finetuned on a small data set from a specific target task.  To address this issue, we propose a Prompt Meta-Regularization~(ProMetaR) to improve the generalizability of prompt learning for vision-language models. Specifically, ProMetaR meta-learns both the regularizer and the soft prompts to harness the task-specific knowledge from the downstream tasks and task-agnostic general knowledge from the vision-language models. Further, ProMetaR augments the task to generate multiple virtual tasks to alleviate the meta-overfitting. In addition, we provide the analysis to comprehend how ProMetaR improves the generalizability of prompt tuning in the perspective of the gradient alignment. Our extensive experiments demonstrate that our ProMetaR improves the generalizability of conventional prompt learning methods under base-to-base/base-to-new and domain generalization settings.

</details>

---

## 73. GeoChat: Grounded Large Vision-Language Model for Remote Sensing

- [ ] GeoChat: Grounded Large Vision-Language Model for Remote Sensing | https://cvpr.thecvf.com/virtual/2024/poster/29981

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29981

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision-Language Models (VLMs) have shown great promise in natural image domains, allowing users to hold a dialogue about given visual content. However, such general-domain VLMs perform poorly for Remote Sensing (RS) scenarios, leading to inaccurate or fabricated information when presented with RS domain-specific queries.  Such a behavior emerges due to the unique challenges introduced by RS imagery. For example, to handle high-resolution RS imagery with diverse scale changes across categories and many small objects, region-level reasoning is necessary alongside holistic scene interpretation. Furthermore, the lack of domain-specific multimodal instruction following data as well as strong backbone models for RS make it hard for the models to align their behavior with user queries. To address these limitations, we propose GeoChat - the first versatile remote sensing VLM that offers multitask conversational capabilities with high-resolution RS images. Specifically, GeoChat can not only answer image-level queries, but also accepts region inputs to hold region-specific dialogue. Furthermore, it can visually ground objects in its responses by referring to their spatial coordinates. To address the lack of domain-specific datasets, we generate a novel RS multimodal instruction-following dataset by extending image-text pairs from existing diverse RS datasets. Leveraging this rich dataset, we fine-tune our remote sensing VLM based on the LLaVA-1.5 architecture.  We establish a comprehensive benchmark for RS multitask conversations and compare with a number of baseline methods. GeoChat demonstrates robust zero-shot performance on various remote sensing tasks, e.g., image and region captioning, visual question answering, scene classification, visually grounded conversations and referring object detection. Our codes will be open-sourced.

</details>

---

## 74. HRVDA: High-Resolution Visual Document Assistant

- [ ] HRVDA: High-Resolution Visual Document Assistant | https://cvpr.thecvf.com/virtual/2024/poster/29985

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29985

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Leveraging vast training data, multimodal large language models (MLLMs) have demonstrated formidable general visual comprehension capabilities and achieved remarkable performance across various tasks.  However, their performance in visual document understanding still leaves much room for improvement. This discrepancy is primarily attributed to the fact that visual document understanding is a fine-grained prediction task. In natural scenes, MLLMs typically use low-resolution images, leading to a substantial loss of visual information. Furthermore, general-purpose MLLMs do not excel in handling document-oriented instructions. In this paper, we propose a High-Resolution Visual Document Assistant (HRVDA), which bridges the gap between MLLMs and visual document understanding. This model employs a content filtering mechanism and an instruction filtering module to separately filter out the content-agnostic visual tokens and instruction-agnostic visual tokens, thereby achieving efficient model training and inference for high-resolution images. In addition, we construct a document-oriented visual instruction tuning dataset and apply a multi-stage training strategy to enhance the model's document modeling capabilities. Extensive experiments demonstrate that our model achieves state-of-the-art performance across multiple document understanding datasets, while maintaining training efficiency and inference speed comparable to low-resolution models.

</details>

---

## 75. Learning Background Prompts to Discover Implicit Knowledge for Open Vocabulary Object Detection

- [ ] Learning Background Prompts to Discover Implicit Knowledge for Open Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2024/poster/29989

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29989

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open vocabulary object detection (OVD) aims at seeking an optimal object detector capable of recognizing objects from both base and novel categories. Recent advances leverage knowledge distillation to transfer insightful knowledge from pre-trained large-scale vision-language models to the task of object detection, significantly generalizing the powerful capabilities of the detector to identify more unknown object categories. However, these methods face significant challenges in background interpretation and model overfitting and thus often result in the loss of crucial background knowledge, giving rise to sub-optimal inference performance of the detector. To mitigate these issues, we present a novel OVD framework termed LBP to propose learning background prompts to harness explored implicit background knowledge, thus enhancing the detection performance w.r.t. base and novel categories. Specifically, we devise three modules: Background Category-specific Prompt, Background Object Discovery, and Inference Probability Rectification, to empower the detector to discover, represent, and leverage implicit object knowledge explored from background proposals. Evaluation on two benchmark datasets, OV-COCO and  OV-LVIS, demonstrates the superiority of our proposed method over existing state-of-the-art approaches in handling the OVD tasks.

</details>

---

## 76. Taming Self-Training for Open-Vocabulary Object Detection

- [ ] Taming Self-Training for Open-Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2024/poster/29999

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/29999

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have shown promising performance in open-vocabulary object detection (OVD) by utilizing pseudo labels (PLs) from pretrained vision and language models (VLMs). However, teacher-student self-training, a powerful and widely used paradigm to leverage PLs, is rarely explored for OVD. This work identifies two challenges of using self-training in OVD: noisy PLs from VLMs and frequent distribution changes of PLs. To address these challenges, we propose SAS-Det that tames self-training for OVD from two key perspectives. First, we present a split-and-fusion (SAF) head that splits a standard detection into an open-branch and a closed-branch. This design can reduce noisy supervision from pseudo boxes. Moreover, the two branches learn complementary knowledge from different training data, significantly enhancing performance when fused together. Second, in our view, unlike in closed-set tasks, the PL distributions in OVD are solely determined by the teacher model. We introduce a periodic update strategy to decrease the number of updates to the teacher, thereby decreasing the frequency of changes in PL distributions, which stabilizes the training process. Extensive experiments demonstrate SAS-Det is both efficient and effective. SAS-Det outperforms recent models of the same scale by a clear margin and achieves 37.4 AP50 and 29.1 APr on novel categories of the COCO and LVIS benchmarks, respectively. Code is available at https://github.com/xiaofeng94/SAS-Det.

</details>

---

## 77. YOLO-World: Real-Time Open-Vocabulary Object Detection

- [ ] YOLO-World: Real-Time Open-Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2024/poster/30009

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30009

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The You Only Look Once (YOLO) series of detectors have established themselves as efficient and practical tools. However, their reliance on predefined and trained object categories limits their applicability in open scenarios. Addressing this limitation, we introduce YOLO-World, an innovative approach that enhances YOLO with open-vocabulary detection capabilities through vision-language modeling and pre-training on large-scale datasets. Specifically, we propose a new Re-parameterizable Vision-Language Path Aggregation Network (RepVL-PAN) and region-text contrastive loss to facilitate the interaction between visual and linguistic information. Our method excels in detecting a wide range of objects in a zero-shot manner with high efficiency. On the challenging LVIS dataset, YOLO-World achieves 35.4 AP with 52.0 FPS on V100, which outperforms many state-of-the-art methods in terms of both accuracy and speed. Furthermore, the fine-tuned YOLO-World  achieves remarkable performance on several downstream tasks, including object detection and open-vocabulary instance segmentation. The code will be made available.

</details>

---

## 78. Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs

- [ ] Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs | https://cvpr.thecvf.com/virtual/2024/poster/30013

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30013

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Is vision good enough for language? Recent advancements in multimodal models primarily stem from the powerful reasoning abilities of large language models (LLMs). However, the visual component typically depends only on the instance-level contrastive language-image pre-training (CLIP). Our research reveals that the visual capabilities in recent Multimodal LLMs (MLLMs) still exhibit systematic shortcomings. To understand the roots of these errors, we explore the gap between the visual embedding space of CLIP and vision-only self-supervised learning. We identify ``CLIP-blind pairs'' – images that CLIP perceives as similar despite their clear visual differences. With these pairs, we construct the Multimodal Visual Patterns (MMVP) benchmark. MMVP exposes areas where state-of-the-art systems, including GPT-4V, struggle with straightforward questions across nine basic visual patterns, often providing incorrect answers and hallucinated explanations. We further evaluate various CLIP-based vision-and-language models and found a notable correlation between visual patterns that challenge CLIP models and those problematic for multimodal LLMs. As an initial effort to address these issues, we propose a Mixture of Features (MoF) approach, demonstrating that integrating vision self-supervised learning features with MLLMs can significantly enhance their visual grounding capabilities. Together, our research suggests visual representation learning remains an open challenge, and accurate visual grounding is crucial for future successful multimodal systems.

</details>

---

## 79. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks

- [ ] InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks | https://cvpr.thecvf.com/virtual/2024/poster/30014

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30014

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The exponential growth of large language models (LLMs) has opened up numerous possibilities for multi-modal AGI systems. However, the progress in vision and vision-language foundation models, which are also critical elements of multi-modal AGI, has not kept pace with LLMs. In this work, we design a large-scale vision-language foundation model (InternVL), which scales up the vision foundation model to 6 billion parameters and progressively aligns it with the LLM, using web-scale image-text data from various sources. This model can be broadly applied to and achieve state-of-the-art performance on 32 generic visual-linguistic benchmarks including visual perception tasks such as image-level or pixel-level recognition, vision-language tasks such as zero-shot image/video classification, zero-shot image/video-text retrieval, and link with LLMs to create multi-modal dialogue systems. It has powerful visual capabilities and can be a good alternative to the ViT-22B. We hope that our research could contribute to the development of multi-modal large models.

</details>

---

## 80. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding

- [ ] TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding | https://cvpr.thecvf.com/virtual/2024/poster/30015

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30015

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This work proposes TimeChat, a time-sensitive multimodal large language model specifically designed for long video understanding. Our model incorporates two key architectural contributions: (1) a timestamp-aware frame encoder that binds visual content with the timestamp of each frame, and (2) a sliding video Q-Former that produces a video token sequence of varying lengths to accommodate videos of various durations. Additionally, we construct an instruction-tuning dataset, encompassing 6 tasks and a total of 125K instances, to further enhance TimeChat's instruction-following performance. Experiment results across various video understanding tasks, such as dense captioning, temporal grounding, and highlight detection, demonstrate TimeChat's strong zero-shot temporal localization and reasoning capabilities. For example, it achieves +9.2 F1 score and +2.8 CIDEr on YouCook2, +5.8 HIT@1 on QVHighlights, and +27.5 R@1 (IoU=0.5) on Charades, compared to state-of-the-art video large language models, holding the potential to serve as a versatile video assistant for long-form video comprehension tasks and satisfy realistic user requirements.

</details>

---

## 81. SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models

- [ ] SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30021

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30021

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current instruction-based image editing methods, such as InstructPix2Pix, often fail to produce satisfactory results in complex scenarios due to their dependence on the simple CLIP text encoder in diffusion models. To rectify this, this paper introduces SmartEdit, a novel approach of instruction-based image editing that leverages Multimodal Large Language Models (MLLMs) to enhance its understanding and reasoning capabilities. However, direct integration of these elements still faces challenges in situations requiring complex reasoning. To mitigate this, we propose a Bidirectional Interaction Module (BIM) that enables comprehensive bidirectional information interactions between the input image and the MLLM output. During training, we initially incorporate perception data to boost the perception and understanding capabilities of diffusion models. Subsequently, we demonstrate that a small amount of complex instruction editing data can effectively stimulate SmartEdit's editing capabilities for more complex instructions. We further construct a new evaluation dataset, Reason-Edit, specifically tailored for complex instruction-based image editing. Both quantitative and qualitative results on this evaluation dataset indicate that our SmartEdit surpasses previous methods, paving the way for the practical application of complex instruction-based image editing.

</details>

---

## 82. Distilling Vision-Language Models on Millions of Videos

- [ ] Distilling Vision-Language Models on Millions of Videos | https://cvpr.thecvf.com/virtual/2024/poster/30028

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30028

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent advance in vision-language models is largely attributed to the abundance of image-text data. We aim to replicate this success for video-language models, but there simply is not enough human-curated video-text data available. We thus resort to fine-tuning a video-language model from a strong image-language baseline with synthesized instructional data. The resulting video-language model is then used to auto-label millions of videos to generate high-quality captions. We show the adapted video-language model performs well on a wide range of video-language benchmarks. For instance, it surpasses the best prior result on open-ended NExT-QA by 2.8%. Besides, our model generates detailed descriptions for previously unseen videos, which provide better textual supervision than existing methods. Experiments show that a video-language dual-encoder model contrastively trained on these auto-generated captions is 3.8% better than the strongest baseline that also leverages vision-language models. Our best model outperforms state-of-the-art methods on MSR-VTT zero-shot text-to-video retrieval by 6%.

</details>

---

## 83. SaCo Loss: Sample-wise Affinity Consistency for Vision-Language Pre-training

- [ ] SaCo Loss: Sample-wise Affinity Consistency for Vision-Language Pre-training | https://cvpr.thecvf.com/virtual/2024/poster/30034

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30034

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training (VLP) aims to learn joint representations of vision and language modalities. The contrastive paradigm is currently dominant in this field. However, we observe a notable misalignment phenomenon, that is, the affinity between samples has an obvious disparity across different modalities, namely ''Affinity Inconsistency Problem''. Our intuition is that, for a well-aligned model, two images that look similar to each other should have the same level of similarity as their corresponding texts that describe them. In this paper, we first investigate the reason of this inconsistency problem. We discover that the lack of consideration for sample-wise affinity consistency across modalities in existing training objectives is the central cause. To address this problem, we propose a novel loss function, named Sample-wise affinity Consistency (SaCo) loss, which is designed to enhance such consistency by minimizing the distance between image embedding similarity and text embedding similarity for any two samples. Our SaCo loss can be easily incorporated into existing vision-language models as an additional loss due to its complementarity for most training objectives. In addition, considering that pre-training from scratch is computationally expensive, we also provide a more efficient way to continuously pre-train on a converged model by integrating our loss. Experimentally, the model trained with our SaCo loss significantly outperforms the baseline on a variety of vision and language tasks.

</details>

---

## 84. MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding

- [ ] MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding | https://cvpr.thecvf.com/virtual/2024/poster/30043

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30043

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the success of large language models (LLMs), integrating the vision model into LLMs to build vision-language foundation models has gained much more interest recently. However, existing LLM-based large multimodal models (e.g., Video-LLaMA, VideoChat) can only take in a limited number of frames for short video understanding. In this study, we mainly focus on designing an efficient and effective model for long-term video understanding. Instead of trying to process more frames simultaneously like most existing work, we propose to process videos in an online manner and store past video information in a memory bank. This allows our model to reference historical video content for long-term analysis without exceeding LLMs' context length constraints or GPU memory limits. Our memory bank can be seamlessly integrated into current multimodal LLMs in an off-the-shelf manner. We conduct extensive experiments on various video understanding tasks, such as long-video understanding, video question answering, and video captioning, and our model can achieve state-of-the-art performances across multiple datasets.

</details>

---

## 85. Querying as Prompt: Parameter-Efficient Learning for Multimodal Language Model

- [ ] Querying as Prompt: Parameter-Efficient Learning for Multimodal Language Model | https://cvpr.thecvf.com/virtual/2024/poster/30051

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30051

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in language models pre-trained on large-scale corpora have significantly propelled developments in the NLP domain and advanced progress in multimodal tasks. In this paper, we propose a Parameter-Efficient multimodal language model learning strategy, named QaP (Querying as Prompt). Its core innovation is a novel modality-bridging method that allows a set of modality-specific queries to be input as soft prompts into a frozen pre-trained language model. Specifically, we introduce an efficient Text-Conditioned Resampler that is easy to incorporate into the language models, which enables adaptive injection of text-related multimodal information at different levels of the model through query learning. This approach effectively bridges multimodal information to the language models while fully leveraging its token fusion and representation potential. We validated our method across four datasets in three distinct multimodal tasks. The results demonstrate that our QaP multimodal language model achieves state-of-the-art performance in various tasks with training only 4.6% parameters.

</details>

---

## 86. Holistic Autonomous Driving Understanding by Bird’s-Eye-View Injected Multi-Modal Large Models

- [ ] Holistic Autonomous Driving Understanding by Bird’s-Eye-View Injected Multi-Modal Large Models | https://cvpr.thecvf.com/virtual/2024/poster/30053

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30053

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rise of multimodal large language models (MLLMs) has spurred interest in language-based driving tasks. However, existing research typically focuses on limited tasks and often omits key multi-view and temporal information which is crucial for robust autonomous driving. To bridge these gaps, we introduce NuInstruct, a novel dataset with 91K multi-view video-QA pairs across 17 subtasks, where each task demands holistic information (e.g., temporal, multi-view, and spatial), significantly elevating the challenge level. To obtain NuInstruct, we propose a novel SQL-based method to generate instruction-response pairs automatically, which is inspired by the driving logical progression of humans. We further present BEV-InMLLM, an end-to-end method for efficiently deriving instruction-aware Bird’s-Eye-View (BEV) features, language-aligned for large language models. BEV-InMLLM integrates multi-view, spatial awareness, and temporal semantics to enhance MLLMs' capabilities on NuInstruct tasks. Moreover, our proposed BEV injection module is a plug-and-play method for existing MLLMs. Our experiments on NuInstruct demonstrate that BEV-InMLLM significantly outperforms existing MLLMs, e.g. ~9% improvement on various tasks. We plan to release our \dataset~for future research development.

</details>

---

## 87. Split to Merge: Unifying Separated Modalities for Unsupervised Domain Adaptation

- [ ] Split to Merge: Unifying Separated Modalities for Unsupervised Domain Adaptation | https://cvpr.thecvf.com/virtual/2024/poster/30062

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30062

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) like CLIP have demonstrated good zero-shot learning performance in the unsupervised domain adaptation task. Yet, most transfer approaches for VLMs focus on either the language or visual branches, overlooking the nuanced interplay between both modalities. In this work, we introduce a Unified Modality Separation (UniMoS) framework for unsupervised domain adaptation. Leveraging insights from modality gap studies, we craft a nimble modality separation network that distinctly disentangles CLIP's features into language-associated and vision-associated components. Our proposed Modality-Ensemble Training (MET) method fosters the exchange of modality-agnostic information while maintaining modality-specific nuances. We align features across domains using a modality discriminator. Comprehensive evaluations on three benchmarks reveal our approach sets a new state-of-the-art with minimal computational costs.

</details>

---

## 88. Language Models as Black-Box Optimizers for Vision-Language Models

- [ ] Language Models as Black-Box Optimizers for Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30073

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30073

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) pre-trained on web-scale datasets have demonstrated remarkable capabilities on downstream tasks when fine-tuned with minimal data. However, many VLMs rely on proprietary data and are not open-source, which restricts the use of white-box approaches for fine-tuning. As such, we aim to develop a black-box approach to optimize VLMs through natural language prompts , thereby avoiding the need to access model parameters, feature embeddings, or even output logits. We propose employing chat-based LLMs to search for the best text prompt for VLMs. Specifically, we adopt an automatic "hill-climbing" procedure that converges to an effective prompt by evaluating the performance of current prompts and asking LLMs to refine them based on textual feedback, all within a conversational process without human-in-the-loop. In a challenging 1-shot image classification setup, our simple approach surpasses the white-box continuous prompting method (CoOp) by an average of 1.5% across 11 datasets including ImageNet. Our approach also outperforms both human-engineered and LLM-generated prompts. We highlight the advantage of conversational feedback that incorporates both positive and negative prompts, suggesting that LLMs can utilize the implicit "gradient" direction in textual feedback for a more efficient search. In addition, we find that the text prompts generated through our strategy are not only more interpretable but also transfer well across different VLM architectures in a black-box manner. Lastly, we demonstrate our framework on a state-of-the-art black-box VLM (DALLE-3) for text-to-image optimization.

</details>

---

## 89. TCP:Textual-based Class-aware Prompt tuning for Visual-Language Model

- [ ] TCP:Textual-based Class-aware Prompt tuning for Visual-Language Model | https://cvpr.thecvf.com/virtual/2024/poster/30087

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30087

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning represents a valuable technique for adapting pre-trained visual-language models (VLM) to various downstream tasks.Recent advancements in CoOp-based methods propose a set of learnable domain-shared or image-conditional textual tokens to facilitate the generation of task-specific textual classifiers. However, those textual tokens have a limited generalization ability regarding unseen domains, as they cannot dynamically adjust to the distribution of testing classes.To tackle this issue, we present a novel Textual-based Class-aware Prompt tuning(TCP) that explicitly incorporates prior knowledge about classes to enhance their discriminability.The critical concept of TCP involves leveraging Textual Knowledge Embedding (TKE) to map the high generalizability of class-level textual knowledge into class-aware textual tokens.By seamlessly integrating these class-aware prompts into the Text Encoder, a dynamic class-aware classifier is generated to enhance discriminability for unseen domains.During inference, TKE dynamically generates class-aware prompts related to the unseen classes.Comprehensive evaluations demonstrate that TKE serves as a plug-and-play module effortlessly combinable with existing methods. Furthermore, TCP consistently achieves superior performance while demanding less training time.

</details>

---

## 90. Low-Rank Approximation for Sparse Attention in Multi-Modal LLMs

- [ ] Low-Rank Approximation for Sparse Attention in Multi-Modal LLMs | https://cvpr.thecvf.com/virtual/2024/poster/30084

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30084

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper focuses on the high computational complexity in Large Language Models (LLMs), a significant challenge in both natural language processing (NLP) and multi-modal tasks. We propose Low-Rank Approximation for Sparse Attention (LoRA-Sparse), an innovative approach that strategically reduces this complexity. LoRA-Sparse introduces low-rank linear projection layers for sparse attention approximation. It utilizes an order-mimic training methodology, which is crucial for efficiently approximating the self-attention mechanism in LLMs. We empirically show that sparse attention not only reduces computational demands, but also enhances model performance in both NLP and multi-modal tasks. This surprisingly shows that redundant attention in LLMs might be non-beneficial. We extensively validate LoRA-Sparse through rigorous empirical studies in both (NLP) and multi-modal tasks, demonstrating its effectiveness and general applicability. Based on LLaMA and LLaVA models, our methods can reduce more than half of the self-attention computation with even better performance than full-attention baselines. Code will be made available.

</details>

---

## 91. PIN: Positional Insert Unlocks Object Localisation Abilities in VLMs

- [ ] PIN: Positional Insert Unlocks Object Localisation Abilities in VLMs | https://cvpr.thecvf.com/virtual/2024/poster/30091

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30091

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs), such as Flamingo and GPT-4V, have shown immense potential by integrating large language models with vision systems. Nevertheless, these models face challenges in the fundamental computer vision task of object localisation, due to their training on multimodal data containing mostly captions without explicit spatial grounding. While it is possible to construct custom, supervised training pipelines with bounding box annotations that integrate with VLMs, these result in specialized and hard-to-scale models. In this paper, we aim to explore the limits of caption-based VLMs and instead propose to tackle the challenge in a simpler manner by i) keeping the weights of a caption-based VLM frozen and ii) not using any supervised detection data. To this end, we introduce an input-agnostic Positional Insert (PIN), a learnable spatial prompt, containing a minimal set of parameters that are slid inside the frozen VLM, unlocking object localisation capabilities. Our PIN module is trained with a simple next-token prediction task on synthetic data without requiring the introduction of new output heads. Our experiments demonstrate strong zero-shot localisation performances on a variety of images, including Pascal VOC, COCO, LVIS, and diverse images like paintings or cartoons.

</details>

---

## 92. AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One

- [ ] AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One | https://cvpr.thecvf.com/virtual/2024/poster/30113

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30113

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A handful of visual foundation models (VFMs) have recently emerged as the backbones for numerous downstream tasks. VFMs like CLIP, DINOv2, SAM are trained with distinct objectives, exhibiting unique characteristics for various downstream tasks. We find that despite their conceptual differences, these models can be effectively merged into a unified model through multi-teacher distillation. We name this approach AM-RADIO (Agglomerative Model -- Reduce All Domains Into One). This integrative approach not only surpasses the performance of individual teacher models but also amalgamates their distinctive features, such as zero-shot vision-language comprehension, detailed pixel-level understanding, and open vocabulary segmentation capabilities. In pursuit of the most hardware-efficient backbone, we evaluated numerous architectures in our multi-teacher distillation pipeline using the same training recipe. This led to the development of a novel architecture (E-RADIO) that exceeds the performance of its predecessors and is at least 7x faster than the teacher models. Our comprehensive benchmarking process covers downstream tasks including ImageNet classification, ADE20k semantic segmentation, COCO object detection and LLaVa-1.5 framework.

</details>

---

## 93. LISA: Reasoning Segmentation via Large Language Model

- [ ] LISA: Reasoning Segmentation via Large Language Model | https://cvpr.thecvf.com/virtual/2024/poster/30109

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30109

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although perception systems have made remarkable advancements in recent years, they still rely on explicit human instruction or pre-defined categories to identify the target objects before executing visual recognition tasks. Such systems cannot actively reason and comprehend implicit user intention. In this work, we propose a new segmentation task --- reasoning segmentation. The task is designed to output a segmentation mask given a complex and implicit query text. Furthermore, we establish a benchmark comprising over one thousand image-instruction-mask data samples, incorporating intricate reasoning and world knowledge for evaluation purposes. Finally, we present LISA: large Language Instructed Segmentation Assistant, which inherits the language generation capabilities of multimodal Large Language Models (LLMs) while also possessing the ability to produce segmentation masks. We expand the original vocabulary with a token and propose the embedding-as-mask paradigm to unlock the segmentation capability. Remarkably, LISA can handle cases involving complex reasoning and world knowledge. Also, it demonstrates robust zero-shot capability when trained exclusively on reasoning-free datasets. In addition, fine-tuning the model with merely 239 reasoning segmentation data samples results in further performance enhancement. Both quantitative and qualitative experiments show our method effectively unlocks new reasoning segmentation capabilities for multimodal LLMs. Code, models, and data are available at github.com/dvlab-research/LISA.

</details>

---

## 94. Tune-An-Ellipse: CLIP Has Potential to Find What You Want

- [ ] Tune-An-Ellipse: CLIP Has Potential to Find What You Want | https://cvpr.thecvf.com/virtual/2024/poster/30111

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30111

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual prompting of large vision language models such as CLIP exhibit intriguing zero-shot capabilities. A manually drawn red circle, commonly used for highlighting, can guide CLIP's attention to the surrounding region, to identify specific objects within an image. Without precise object proposals, however, it is insufficient for localization. Our novel, simple yet effective approach enables CLIP to zero-shot localize: given an image and a text prompt describing an object, we first pick an initial ellipse from uniformly distributed anchor ellipses on the image grid via visual prompting, then use three loss functions to tune the ellipse coefficients to encapsulate the target region gradually. This yields promising experimental results for referring expression comprehension without precisely specified object proposals. In addition, we systematically present the limitations of visual prompting inherent in CLIP and discuss potential avenues for improvement. Code will be released.

</details>

---

## 95. EgoThink: Evaluating First-Person Perspective Thinking Capability of Vision-Language Models

- [ ] EgoThink: Evaluating First-Person Perspective Thinking Capability of Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30136

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30136

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have recently shown promising results in traditional downstream tasks.Evaluation studies have emerged to assess their abilities, with the majority focusing on the third-person perspective, and only a few addressing specific tasks from the first-person perspective.However, the capability of VLMs to "think" from a first-person perspective, a crucial attribute for advancing autonomous agents and robotics, remains largely unexplored. To bridge this research gap, we introduce EgoThink, a novel visual question-answering benchmark that encompasses six core capabilities with twelve detailed dimensions.The benchmark is constructed using selected clips from egocentric videos, with manually annotated question-answer pairs containing first-person information. To comprehensively assess VLMs, we evaluate twenty-one popular VLMs on EgoThink. Moreover, given the open-ended format of the answers, we use GPT-4 as the automatic judge to compute single-answer grading.Experimental results indicate that although GPT-4V leads in numerous dimensions, all evaluated VLMs still possess considerable potential for improvement in first-person perspective tasks.Meanwhile, enlarging the number of trainable parameters has the most significant impact on model performance on EgoThink.In conclusion, EgoThink serves as a valuable addition to existing evaluation benchmarks for VLMs, providing an indispensable resource for future research in the realm of embodied artificial intelligence and robotics.

</details>

---

## 96. LLaMA-Excitor: General Instruction Tuning via Indirect Feature Interaction

- [ ] LLaMA-Excitor: General Instruction Tuning via Indirect Feature Interaction | https://cvpr.thecvf.com/virtual/2024/poster/30137

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30137

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing methods to fine-tune LLMs, like Adapter, Prefix-tuning, and LoRA, which introduce extra modules or additional input sequences to inject new skills or knowledge, may compromise the innate abilities of LLMs.In this paper, we propose LLaMA-Excitor, a lightweight method that stimulates the LLMs' potential to better follow instructions by gradually paying more attention to worthwhile information.Specifically, the LLaMA-Excitor does not directly change the intermediate hidden state during the self-attention calculation of the transformer structure.We designed the Excitor block as a bypass module for the similarity score computation in LLMs' self-attention to reconstruct keys and change the importance of values by learnable prompts. LLaMA-Excitor ensures a self-adaptive allocation of additional attention to input instructions, thus effectively preserving LLMs' pre-trained knowledge when fine-tuning LLMs on low-quality instruction-following datasets. Furthermore, we unify the modeling of multi-modal tuning and language-only tuning, extending LLaMA-Excitor to a powerful visual instruction follower without the need for complex multi-modal alignment. Our proposed approach is evaluated in language-only and multi-modal tuning experimental scenarios. Notably, LLaMA-Excitor is the only method that maintains basic capabilities while achieving a significant improvement (+6\%) on the MMLU benchmark.In the visual instruction tuning, we achieve a new state-of-the-art image captioning performance of 157.5 CIDEr on MSCOCO, and a comparable performance (88.39\%) on ScienceQA to cutting-edge models with more parameters and extensive vision-language pertaining.

</details>

---

## 97. PeVL: Pose-Enhanced Vision-Language Model for Fine-Grained Human Action Recognition

- [ ] PeVL: Pose-Enhanced Vision-Language Model for Fine-Grained Human Action Recognition | https://cvpr.thecvf.com/virtual/2024/poster/30142

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30142

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in Vision-Language Foundation (VLF) models has revealed the great advantages of cross-modality learning. However, due to a large gap between vision and text, they might not be able to sufficiently utilize the benefits of cross-modality information. In the field of human action recognition, the additional pose modality may bridge the gap between vision and text to improve the effectiveness of cross-modality learning. In this paper, we propose a novel framework, called Pose-enhanced Vision-Language (PeVL) model, to adapt the VL model with pose modality to learn effective knowledge of fine-grained human actions. Our PeVL model includes two novel components: an Unsymmetrical Cross-Modality Refinement (UCMR) block and a Semantic-Guided Multi-level Contrastive (SGMC) module. The UCMR block includes Pose-guided Visual Refinement (P2V-R) and Visual-enriched Pose Refinement (V2P-R) for effective cross-modality learning. The SGMC module includes Multi-level Contrastive Associations of vision-text and pose-text at both action and sub-action levels, and a Semantic-Guided Loss, enabling effective contrastive learning among three modalities (i.e., video, pose, and text). Built upon a pre-trained VL foundation model, our model integrates trainable adapters and can be trained end-to-end. Our novel PeVL design over VL foundation model yields remarkable performance gains on four fine-grained human action recognition datasets, achieving new SOTA with a significantly small number of tunable parameters for low-cost re-training.

</details>

---

## 98. Consistency and Uncertainty: Identifying Unreliable Responses From Black-Box Vision-Language Models for Selective Visual Question Answering

- [ ] Consistency and Uncertainty: Identifying Unreliable Responses From Black-Box Vision-Language Models for Selective Visual Question Answering | https://cvpr.thecvf.com/virtual/2024/poster/30163

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30163

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The goal of selective prediction is to allow an a model to abstain when it may not be able to deliver a reliable prediction, which is important in safety-critical contexts. Existing approaches to selective prediction typically require access to the internals of a model, require retraining a model or study only unimodal models. However, the most powerful models (e.g. GPT-4) are typically only available as black boxes with inaccessible internals, are not retrainable by end-users, and are frequently used for multimodal tasks. We study the possibility of selective prediction for vision-language models in a realistic, black-box setting. We propose using the principle of neighborhood consistency to identify unreliable responses from a black-box vision-language model in question answering tasks. We hypothesize that given only a visual question and model response, the consistency of the model's responses over the neighborhood of a visual question will indicate reliability. It is impossible to directly sample neighbors in feature space in a black-box setting. Instead, we show that it is possible to use a smaller proxy model to approximately sample from the neighborhood. We find that neighborhood consistency can be used to identify model responses to visual questions that are likely unreliable, even in adversarial settings or settings that are out-of-distribution to the proxy model.

</details>

---

## 99. CogAgent: A Visual Language Model for GUI Agents

- [ ] CogAgent: A Visual Language Model for GUI Agents | https://cvpr.thecvf.com/virtual/2024/poster/30177

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30177

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

People are spending an enormous amount of time on digital devices through graphical user interfaces (GUIs), e.g., computer or smartphone screens.  Large language models (LLMs) such as ChatGPT can assist people in tasks like writing emails, but struggle to understand and interact with GUIs, thus limiting their potential to increase automation levels. In this paper, we introduce CogAgent, an 18-billion-parameter visual language model (VLM) specializing in GUI understanding and navigation. By utilizing both low-resolution and high-resolution image encoders, CogAgent supports input at a resolution of $1120\times 1120$, enabling it to recognize tiny page elements and text. As a generalist visual language model, CogAgent achieves the state of the art on five text-rich and four general VQA benchmarks, including VQAv2, OK-VQA, Text-VQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE. CogAgent, using only screenshots as input, outperforms LLM-based methods that consume extracted HTML text on both PC and Android GUI navigation tasks---Mind2Web and AITW, advancing the state of the art. The model and codes are available at https://github.com/THUDM/CogVLM.

</details>

---

## 100. MAPLM: A Real-World Large-Scale Vision-Language Benchmark for Map and Traffic Scene Understanding

- [ ] MAPLM: A Real-World Large-Scale Vision-Language Benchmark for Map and Traffic Scene Understanding | https://cvpr.thecvf.com/virtual/2024/poster/30180

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30180

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language generative AI has demonstrated remarkable promise for empowering cross-modal scene understanding of autonomous driving and high-definition (HD) map systems. However, current benchmark datasets lack multi-modal point cloud, image, and language data pairs. Recent approaches utilize visual instruction learning and cross-modal prompt engineering to expand vision-language models into this domain. In this paper, we propose a new vision-language benchmark that can be used to finetune traffic and HD map domain-specific foundation models. Specifically, we annotate and leverage large-scale, broad-coverage traffic and map data extracted from huge HD map annotations, and use CLIP and LLaMA-2 / Vicuna to finetune a baseline model with instruction-following data. Our experimental results across various algorithms reveal that while visual instruction-tuning large language models (LLMs) can effectively learn meaningful representations from MAPLM-QA, there remains significant room for further advancements. To facilitate applying LLMs and multi-modal data into self-driving research, we will release our visual-language QA data, and the baseline models at GitHub.com/LLVM-AD/MAPLM.

</details>

---

## 101. Efficient Test-Time Adaptation of Vision-Language Models

- [ ] Efficient Test-Time Adaptation of Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30197

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30197

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation with pre-trained vision-language models has attracted increasing attention for tackling distribution shifts during the test time. Though prior studies have achieved very promising performance, they involve intensive computation which is severely unaligned with test-time adaptation. We design TDA, a training-free dynamic adapter that enables effective and efficient test-time adaptation with vision-language models. TDA works with a lightweight key-value cache that maintains a dynamic queue with few-shot pseudo labels as values and the corresponding test-sample features as keys. Leveraging the key-value cache, TDA allows adapting to test data gradually via progressive pseudo label refinement which is super-efficient without incurring any backpropagation. In addition, we introduce negative pseudo labeling that alleviates the adverse impact of pseudo label noises by assigning pseudo labels to certain negative classes when the model is uncertain about its pseudo label predictions. Extensive experiments over two benchmarks demonstrate TDA’s superior effectiveness and efficiency as compared with the state-of-the-art. The code has been released in https://kdiaaa.github.io/tda/.

</details>

---

## 102. MMA: Multi-Modal Adapter for Vision-Language Models

- [ ] MMA: Multi-Modal Adapter for Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30210

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30210

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained Vision-Language Models (VLMs) have served as excellent foundation models for transfer learning in diverse downstream tasks. However, tuning VLMs for few-shot generalization tasks faces a discrimination — generalization dilemma, i.e., general knowledge should be preserved and task-specific knowledge should be fine-tuned. How to precisely identify these two types of representations remains a challenge. In this paper, we propose a Multi-Modal Adapter (MMA) for VLMs to improve the alignment between representations from text and vision branches. MMA aggregates features from different branches into a shared feature space so that gradients can be communicated across branches. To determine how to incorporate MMA, we systematically analyze the discriminability and generalizability of features across diverse datasets in both the vision and language branches, and find that (1) higher layers contain discriminable dataset-specific knowledge, while lower layers contain more generalizable knowledge, and (2) language features are more discriminable than visual features, and there are large semantic gaps between the features of the two modalities, especially in the lower layers. Therefore, we only incorporate MMA to a few higher layers of transformers to achieve an optimal balance between discrimination and generalization. We evaluate the effectiveness of our approach on three tasks: generalization to novel classes, novel target datasets, and domain generalization. Compared to many state-of-the-art methods, our MMA achieves leading performance in all evaluations. Code is at https://github.com/ZjjConan/Multi-Modal-Adapter

</details>

---

## 103. Calibrating Multi-modal Representations: A Pursuit of Group Robustness without Annotations

- [ ] Calibrating Multi-modal Representations: A Pursuit of Group Robustness without Annotations | https://cvpr.thecvf.com/virtual/2024/poster/30214

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30214

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning pre-trained vision-language models, like CLIP, has yielded success on diverse downstream tasks. However, several pain points persist for this paradigm: (i) directly tuning entire pre-trained models becomes both time-intensive and computationally costly. Additionally, these tuned models tend to become highly specialized, limiting their practicality for real-world deployment; (ii) recent studies indicate that pre-trained vision-language classifiers may overly depend on spurious features -- patterns that correlate with the target in training data, but are not related to the true labeling function; and (iii) existing studies on mitigating the reliance on spurious features, largely based on the assumption that we can identify such features, does not provide definitive assurance for real-world applications. As a piloting study, this work focuses on exploring mitigating the reliance on spurious features for CLIP without using any group annotation. To this end, we systematically study the existence of spurious correlation on CLIP and CILP+ERM. We first, following recent work on Deep Feature Reweighting (DFR), verify that last-layer retraining can greatly improve group robustness on pretrained CLIP. In view of them, we advocate a lightweight representation calibration method for fine-tuning CLIP, by first generating a calibration set using the pretrained CLIP, and then calibrating representations of samples within this set through contrastive learning, all without the need for group labels. Extensive experiments and in-depth visualizations on several benchmarks validate the effectiveness of our proposals, largely reducing reliance and significantly boosting the model generalization.

</details>

---

## 104. Source-Free Domain Adaptation with Frozen Multimodal Foundation Model

- [ ] Source-Free Domain Adaptation with Frozen Multimodal Foundation Model | https://cvpr.thecvf.com/virtual/2024/poster/30212

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30212

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Source-Free Domain Adaptation (SFDA) aims to adapt a source model for a target domain, with only access to unlabeled target training data and the source model pre-trained on a supervised source domain. Relying on pseudo labeling and/or auxiliary supervision, conventional methods are inevitably error-prone. To mitigate this limitation, in this work we for the first time explore the potentials of off-the-shelf vision-language (ViL) multimodal models (e.g., CLIP) with rich whilst heterogeneous knowledge. We find that directly applying the ViL model to the target domain in a zero-shot fashion is unsatisfactory, as it is not specialized for this particular task but largely generic. To make it task specific, we propose a novel Distilling multImodal Foundation mOdel (DIFO) approach. Specifically, DIFO alternates between two steps during adaptation: (i) Customizing the ViL model by maximizing the mutual information with the target model in a prompt learning manner, (ii) Distilling the knowledge of this customized ViL model to the target model. For more fine-grained and reliable distillation, we further introduce two effective regularization terms, namely most likely category encouragement and predictive consistency. Extensive experiments show that DIFO significantly outperforms the state-of-the-art alternatives. Our source code will be released.

</details>

---

## 105. AHIVE: Anatomy-aware Hierarchical Vision Encoding for Interactive Radiology Report Retrieval

- [ ] AHIVE: Anatomy-aware Hierarchical Vision Encoding for Interactive Radiology Report Retrieval | https://cvpr.thecvf.com/virtual/2024/poster/30220

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30220

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Automatic radiology report generation using deep learning models has been recently explored and found promising. Neural decoders are commonly used for the report generation, where irrelevant and unfaithful contents are unavoidable. The retrieval-based approach alleviates the limitation by identifying reports which are relevant to the input to assist the generation. To achieve clinically accurate report retrieval, we make reference to clinicians' diagnostic steps of examining a radiology image where anatomical and diagnostic details are typically focused, and propose a novel hierarchical visual concept representation called anatomy-aware hierarchical vision encoding (AHIVE). To learn AHIVE, we first derive a methodology to extract hierarchical diagnostic descriptions from radiology reports and develop a CLIP-based framework for the model training. Also, the hierarchical architecture of AHIVE is designed to support interactive report retrieval so that report revision made at one layer can be propagated to the subsequent ones to trigger other necessary revisions. We conduct extensive experiments and show that AHIVE can outperform the SOTA vision-language retrieval methods in terms of clinical accuracy by a large margin. We provide also a case study to illustrate how it enables interactive report retrieval.

</details>

---

## 106. The Neglected Tails in Vision-Language Models

- [ ] The Neglected Tails in Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30229

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30229

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) such as CLIP excel in zero-shot recognition but exhibit drastically imbalanced performance across visual concepts in downstream tasks. For example, despite an impressive mean zero-shot accuracy on ImageNet (72.7\%), CLIP yields $<$10\% on ten concepts (e.g., ${\tt night}$ ${\tt snake}$ and ${\tt snoek}$). This is presumably because these concepts are under-represented in VLMs' pretraining datasets, which are believed to exhibit imbalanced distributions of concepts. Yet, assessing this imbalance is challenging, as calculating the frequency of specific concepts within large-scale pretraining data is not straightforward. In this work, we make the first attempt to measure the concept frequency in VLMs' pretraining data by counting relevant pretraining texts. We also use off-the-shelf language models to help count relevant texts that contain synonyms of the given concepts and resolve ambiguous cases. Our analysis confirms that visual concepts follow a long-tailed distribution in the pretraining data, which strongly correlates with per-class accuracies. Further, to mitigate VLMs' imbalanced performance in zero-shot recognition, we propose ${\bf RE}$trieval-${\bf A}$ugmented ${\bf L}$earning (REAL). First, instead of prompting VLMs using the original class names defined in a downstream task, REAL uses their most frequent synonyms found in the pretraining texts. This already outperforms human-engineered and LLM-generated prompts over nine benchmark datasets, likely because VLMs have seen more images associated with the more frequent synonyms. Second, REAL uses all the concept synonyms to retrieve a small class-balanced subset of images from the pretraining data to learn a robust classifier. REAL rivals the recent retrieval-augmented solution REACT, using $400\times$ less storage and 10,000$\times$ less training time!

</details>

---

## 107. Situational Awareness Matters in 3D Vision Language Reasoning

- [ ] Situational Awareness Matters in 3D Vision Language Reasoning | https://cvpr.thecvf.com/virtual/2024/poster/30234

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30234

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Being able to carry out complicated vision-language reasoning tasks in 3D space represents a significant milestone in developing household robots and human-centered embodied AI. In this work, we demonstrate that a critical and distinct challenge in 3D vision language reasoning is the situational awareness, which incorporates two key components: (1) The autonomous agent grounds its self-location based on a language prompt. (2) The agent answers open-ended questions from the perspective of its calculated position. To address this challenge, we propose SIG3D, an end-to-end Situation-Grounded model for 3D vision language reasoning. We tokenize the 3D scene into sparse voxel representation, and propose a language-grounded situation estimator followed by a situated question answering module. Experiments on the SQA3D and ScanQA datasets show that SIG3D outperforms state-of-the-art models in situational estimation and question answering by a larger margin (e.g., an enhancement of over 30% on situation accuracy). Subsequent analysis corroborates our architectural design choices, explores the distinct functions of visual and textual tokens, and highlights the importance of situational awareness in the domain of 3D question-answering.

</details>

---

## 108. Learning by Correction: Efficient Tuning Task for Zero-Shot Generative Vision-Language Reasoning

- [ ] Learning by Correction: Efficient Tuning Task for Zero-Shot Generative Vision-Language Reasoning | https://cvpr.thecvf.com/virtual/2024/poster/30235

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30235

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generative vision-language models (VLMs) have shown impressive performance on zero-shot vision-language tasks through generalization-based text generation. However, recent studies have improved upon this generalization capability for zero-shot VL tasks by utilizing second-stage instruction tuning with a large amount of human-labeled and externally generated data from large language models.In this work, we propose the image-conditioned text correction task for enhancing zero-shot text generation with unlabeled data. By leveraging the inherent structure of language, we produce the image-text pair containing mismatched concepts, and VLMs are required to identify and correct the error according to the vision modality by text generation.Our experiments demonstrate that our second-stage tuning framework significantly enhances the generalization capabilities of VLMs on various image-to-text generation-based VL tasks.This work represents a promising direction for advancing the field of zero-shot inference in VLMs by providing a cost-effective and scalable solution for enhancing generalization performance.

</details>

---

## 109. ModaVerse: Efficiently Transforming Modalities with LLMs

- [ ] ModaVerse: Efficiently Transforming Modalities with LLMs | https://cvpr.thecvf.com/virtual/2024/poster/30292

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30292

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Humans possess the capability to comprehend diverse modalities and seamlessly transfer information between them. In this work, we introduce ModaVerse, a Multi-modal Large Language Model (MLLM) capable of comprehending and transforming content across various modalities including images, videos, and audio. Predominant MLLM frameworks have largely relied on the alignment of latent spaces of textual and non-textual features. This alignment process, which synchronizes a language model trained on textual data with encoders and decoders trained on multi-modal data, often necessitates extensive training of several projection layers in multiple stages. Inspired by LLM-as-agent methodologies, we propose a novel Input/Output (I/O) alignment mechanism that operates directly at the level of natural language. It aligns the LLM's output with the input of generative models, avoiding the complexities associated with latent feature alignments, and simplifying the multiple training stages of existing MLLMs into a single, efficient process. This conceptual advancement leads to significant reductions in both data and computational costs. By conducting experiments on several benchmarks, we demonstrate that our approach attains comparable performance with the state of the art while achieving considerable efficiencies in data usage and training duration.

</details>

---

## 110. Multi-Modal Hallucination Control by Visual Information Grounding

- [ ] Multi-Modal Hallucination Control by Visual Information Grounding | https://cvpr.thecvf.com/virtual/2024/poster/30302

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30302

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generative Vision-Language Models (VLMs) are prone to generate plausible-sounding textual answers which, however, are not always grounded in the input image. We investigate this phenomenon, usually referred to as "hallucination", and show that it stems from an excessive reliance on the language prior. In particular, we show that as more tokens are generated the reliance on the visual prompt decreases and this behavior strongly correlates with the emergence of hallucinations. To reduce hallucinations, we introduce Multi-Modal Mutual-Information Decoding (M3ID), a new sampling method for prompt amplification. M3ID amplifies the influence of the reference image over the language prior, hence favoring the generation of tokens with higher mutual information with the visual prompt. M3ID can be applied to any pre-trained autoregressive VLM at inference time without necessitating further training and with minimal computational overhead. If training is an option, we show that M3ID can be paired with Direct Preference Optimization (DPO) to improve the model's reliance on the prompt image without requiring any labels. Our empirical findings show that our algorithms maintain the fluency and linguistic capabilities of pre-trained VLMs while reducing hallucinations by mitigating visually ungrounded answers. Specifically, for the LLaVA 13B model, M3ID and M3ID+DPO reduce the percentage of hallucinated objects in captioning tasks by 25% and 28%, respectively, and improve the accuracy on VQA benchmarks such as POPE by 21% and 24%.

</details>

---

## 111. Scaling Laws for Data Filtering— Data Curation cannot be Compute Agnostic

- [ ] Scaling Laws for Data Filtering— Data Curation cannot be Compute Agnostic | https://cvpr.thecvf.com/virtual/2024/poster/30322

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30322

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) are trained for thousands of GPU hours on massive web scrapes, but they are not trained on all data indiscriminately. For instance, the LAION public dataset retained only about 10\% of the total crawled data. In recent times, data curation has gained prominence with several works developing strategies to retain "high-quality" subsets of "raw" scraped data. However, these strategies are typically developed agnostic to the available compute for training. In this paper, we demonstrate that making filtering decisions independent of training compute is often suboptimal---well-curated data rapidly loses its utilitywhen repeated, eventually decreasing below the utility of "unseen" but "lower-quality" data. In fact, we show that even a model trained on $\textit{unfiltered common crawl}$ obtains higher accuracy than that trained on the LAION dataset post 40 or more repetitions.While past research in neural scaling laws has considered web data to be homogenous, real data is not.Our work bridges this important gap in the literature by developing scaling trends that characterize the "utility" of various data subsets, accounting for the diminishing utility of a data point at its "nth" repetition.Our key message is that data curation $\textit{can not}$ be agnostic of the total compute a model will be trained for. Based on our analysis, we propose FADU (Filter by Assessing Diminishing Utility) that curates the best possible pool for achieving top performance on Datacomp at various compute budgets, carving out a pareto-frontier for data curation.

</details>

---

## 112. PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation

- [ ] PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation | https://cvpr.thecvf.com/virtual/2024/poster/30350

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30350

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper proposes a cross-modal distillation framework, PartDistill, which transfers 2D knowledge from vision-language models (VLMs) to facilitate 3D shape part segmentation. PartDistill addresses three major challenges in this task: the lack of 3D segmentation in invisible or undetected regions in the 2D projections, inconsistent 2D predictions by VLMs, and the lack of knowledge accumulation across different 3D shapes. PartDistill consists of a teacher network that uses a VLM to make 2D predictions and a student network that learns from the 2D predictions while extracting geometrical features from multiple 3D shapes to carry out 3D part segmentation. A bi-directional distillation, including forward and backward distillations, is carried out within the framework, where the former forward distills the 2D predictions to the student network, and the latter improves the quality of the 2D predictions, which subsequently enhances the final 3D segmentation. Moreover, PartDistill can exploit generative models that facilitate effortless 3D shape creation for generating knowledge sources to be distilled. Through extensive experiments, PartDistill boosts the existing methods with substantial margins on widely used ShapeNetPart and PartNetE datasets, by more than 15\% and 12\% higher mIoU scores, respectively.  The code for this work is available at https://github.com/ardianumam/PartDistill.

</details>

---

## 113. Decomposing Disease Descriptions for Enhanced Pathology Detection: A Multi-Aspect Vision-Language Pre-training Framework

- [ ] Decomposing Disease Descriptions for Enhanced Pathology Detection: A Multi-Aspect Vision-Language Pre-training Framework | https://cvpr.thecvf.com/virtual/2024/poster/30353

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30353

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Medical vision language pre-training (VLP) has emerged as a frontier of research, enabling zero-shot pathological recognition by comparing the query image with the textual descriptions for each disease. Due to the complex semantics of biomedical texts, current methods struggle to align medical images with key pathological findings in unstructured reports. This leads to the misalignment with the target disease's textual representation. In this paper, we introduce a novel VLP framework designed to dissect disease descriptions into their fundamental aspects, leveraging prior knowledge about the visual manifestations of pathologies. This is achieved by consulting a large language model and medical experts. Integrating a Transformer module, our approach aligns an input image with the diverse elements of a disease, generating aspect-centric image representations. By consolidating the matches from each aspect, we improve the compatibility between an image and its associated disease. Additionally, capitalizing on the aspect-oriented representations, we present a dual-head Transformer tailored to process known and unknown diseases, optimizing the comprehensive detection efficacy. Conducting experiments on seven downstream datasets, ours improves the accuracy of recent methods by up to 8.56% and 17.0% for seen and unseen categories, respectively. Our code is released at https://github.com/HieuPhan33/MAVL.

</details>

---

## 114. KVQ: Kwai Video Quality Assessment for Short-form Videos

- [ ] KVQ: Kwai Video Quality Assessment for Short-form Videos | https://cvpr.thecvf.com/virtual/2024/poster/30360

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30360

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Short-form UGC video platforms, like Kwai and TikTok, have been an emerging and irreplaceable mainstream media form, thriving on user-friendly engagement, and kaleidoscope creation, etc. However, the advancing contentgeneration modes, e.g., special effects, and sophisticated processing workflows, e.g., de-artifacts, have introduced significant challenges to recent UGC video quality assessment: (i) the ambiguous contents hinder the identification of quality-determined regions. (ii) the diverse and complicated hybrid distortions are hard to distinguish. To tackle the above challenges and assist in the development of short-form videos, we establish the first large-scale Kwai short Video database for Quality assessment, termed KVQ, which comprises 600 user-uploaded short videos and 3600processed videos through the diverse practical processing workflows, including pre-processing, transcoding, and enhancement. Among them, the absolute quality score of each video and partial ranking score among indistinguish samples are provided by a team of professional researchers. specializing in image processing. Based on this database, we propose the first short-form video quality evaluator,i.e., KSVQE, which enables the quality evaluator to identify the quality-determined semantics with the content understanding of large vision language models (i.e., CLIP) and distinguish the distortions with the distortion understanding module. Experimental results have shown the effectiveness of KSVQE on our KVQ database and popular VQA databases. The project can be found at https://lixinustc.github.io/projects/KVQ/.

</details>

---

## 115. FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication

- [ ] FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication | https://cvpr.thecvf.com/virtual/2024/poster/30381

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30381

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent dataset deduplication techniques have demonstrated that content-aware dataset pruning can dramatically reduce the cost of training Vision-Language Pretrained (VLP) models without significant performance losses compared to training on the original dataset. These results have been based on pruning commonly used image-caption datasets collected from the web -- datasets that are known to harbor harmful social biases that may then be codified in trained models. In this work, we evaluate how deduplication affects the prevalence of these biases in the resulting trained models and introduce an easy-to-implement modification to the recent SemDeDup algorithm that can reduce the negative effects that we observe. When examining CLIP-style models trained on deduplicated variants of LAION-400M, we find our proposed FairDeDup algorithm consistently leads to improved fairness metrics over SemDeDup on the FairFace and FACET datasets while maintaining zero-shot performance on CLIP benchmarks.

</details>

---

## 116. On the Test-Time Zero-Shot Generalization of Vision-Language Models: Do We Really Need Prompt Learning?

- [ ] On the Test-Time Zero-Shot Generalization of Vision-Language Models: Do We Really Need Prompt Learning? | https://cvpr.thecvf.com/virtual/2024/poster/30379

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30379

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The development of large vision-language models, notably CLIP, has catalyzed research into effective adaptation techniques, with a particular focus on soft prompt tuning. Conjointly, test-time augmentation, which utilizes multiple augmented views of a single image to enhance zero-shot generalization, is emerging as a significant area of interest. This has predominantly directed research efforts towards test-time prompt tuning. In contrast, we introduce a robust $\textbf{M}$eanShift for $\textbf{T}$est-time $\textbf{A}$ugmentation (MTA), which surpasses prompt-based methods without requiring this intensive training procedure. This positions MTA as an ideal solution for both standalone and API-based applications. Additionally, our method does not rely on ad hoc rules (e.g., confidence threshold) used in some previous test-time augmentation techniques to filter the augmented views. Instead, MTA incorporates a quality assessment variable for each view directly into its optimization process, termed as the inlierness score. This score is jointly optimized with a density mode seeking process, leading to an efficient training- and hyperparameter-free approach. We extensively benchmark our method on 15 datasets and demonstrate MTA's superiority and computational efficiency. Deployed easily as plug-and-play module on top of zero-shot models and state-of-the-art few-shot methods, MTA shows systematic and consistent improvements.

</details>

---

## 117. One-Shot Open Affordance Learning with Foundation Models

- [ ] One-Shot Open Affordance Learning with Foundation Models | https://cvpr.thecvf.com/virtual/2024/poster/30384

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30384

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce One-shot Open Affordance Learning (OOAL), where a model is trained with just one example per base object category, but is expected to identify novel objects and affordances. While vision-language models excel at recognizing novel objects and scenes, they often struggle to understand finer levels of granularity such as affordances. To handle this issue, we conduct a comprehensive analysis of existing foundation models, to explore their inherent understanding of affordances and assess the potential for data-limited affordance learning. We then propose a vision-language framework with simple and effective designs that boost the alignment between visual features and affordance text embeddings. Experiments on two affordance segmentation benchmarks show that the proposed method outperforms state-of-the-art models with less than 1\% of the full training data, and exhibits reasonable generalization capability on unseen objects and affordances.

</details>

---

## 118. Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld

- [ ] Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld | https://cvpr.thecvf.com/virtual/2024/poster/30385

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30385

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While large language models (LLMs) excel in a simulated world of texts, they struggle to interact with the more realistic world without perceptions of other modalities such as visual or audio signals. Although vision-language models (VLMs) integrate LLM modules (1) aligned with static image features, and (2) may possess prior knowledge of world dynamics (as demonstrated in the text world), they have not been trained in an embodied visual world and thus cannot align with its dynamics. On the other hand, training an embodied agent in a noisy visual world without expert guidance is often challenging and inefficient. In this paper, we train a VLM agent living in a visual world using an LLM agent excelling in a parallel text world. Specifically, we distill LLM's reflection outcomes (improved actions by analyzing mistakes) in a text world's tasks to finetune the VLM on the same tasks of the visual world, resulting in an Embodied Multi-Modal Agent (EMMA) quickly adapting to the visual world dynamics. Such cross-modality imitation learning between the two parallel worlds is achieved by a novel DAgger-DPO algorithm, enabling EMMA to generalize to a broad scope of new tasks without any further guidance from the LLM expert. Extensive evaluations on the ALFWorld benchmark's diverse tasks highlight EMMA's superior performance to SOTA VLM-based agents, e.g., 20%-70% improvement in the success rate.

</details>

---

## 119. Building Vision-Language Models on Solid Foundations with Masked Distillation

- [ ] Building Vision-Language Models on Solid Foundations with Masked Distillation | https://cvpr.thecvf.com/virtual/2024/poster/30391

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30391

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language Models (VLMs) have marked a significant leap in bridging the gap between computer vision and natural language processing. However, traditional VLMs, trained through contrastive learning on limited and noisy image-text pairs, often lack the spatial and linguistic understanding to generalize well to dense vision tasks or less common languages. Our approach, SF-CLIP, circumvents this issue by implicitly building on the solid visual and language understanding of foundational models trained on vast amounts of unimodal data. SF-CLIP integrates contrastive image-text pretraining with a masked knowledge distillation from large foundational text and vision models. This methodology guides our VLM in developing robust text and image representations.As a result, SF-CLIP shows exceptional zero-shot classification accuracy and enhanced image and text retrieval capabilities, setting a new state of the art for ViT-B/16 trained on YFCC15M and CC12M. Moreover, the dense per-patch supervision enhances our zero-shot and linear probe performance in semantic segmentation tasks. A remarkable aspect of our model is its multilingual proficiency, evidenced by strong retrieval results in multiple languages despite being trained predominantly on English data. We achieve all of these improvements without sacrificing the training efficiency through our selective application of masked distillation and the inheritance of teacher word embeddings.

</details>

---

## 120. Unknown Prompt the only Lacuna: Unveiling CLIP's Potential for Open Domain Generalization

- [ ] Unknown Prompt the only Lacuna: Unveiling CLIP's Potential for Open Domain Generalization | https://cvpr.thecvf.com/virtual/2024/poster/30422

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30422

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We delve into Open Domain Generalization (ODG), marked by domain and category shifts between training's labeled source and testing's unlabeled target domains. Existing solutions to ODG face limitations due to constrained generalizations of traditional CNN backbones and errors in detecting target open samples in the absence of prior knowledge. Addressing these pitfalls, we introduce ODG-CLIP, harnessing the semantic prowess of the vision-language model, CLIP. Our framework brings forth three primary innovations:Firstly, distinct from prevailing paradigms, we conceptualize ODG as a multi-class classification challenge encompassing both known and novel categories. Central to our approach is modeling a unique prompt tailored for detecting unknown class samples, and to train this, we employ a readily accessible stable diffusion model, elegantly generating proxy images for the open class.Secondly, aiming for domain-tailored classification (prompt) weights while ensuring a balance of precision and simplicity, we devise a novel visual style-centric prompt learning mechanism.Finally, we infuse images with class-discriminative knowledge derived from the prompt space to augment the fidelity of CLIP's visual embeddings. We introduce a novel objective to safeguard the continuity of this infused semantic intel across domains, especially for the shared classes.Through rigorous testing on diverse datasets, covering closed and open-set DG contexts, ODG-CLIP demonstrates clear supremacy, consistently outpacing peers with performance boosts between 8\%-16\%.

</details>

---

## 121. MoPE-CLIP: Structured Pruning for Efficient Vision-Language Models with Module-wise Pruning Error Metric

- [ ] MoPE-CLIP: Structured Pruning for Efficient Vision-Language Models with Module-wise Pruning Error Metric | https://cvpr.thecvf.com/virtual/2024/poster/30442

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30442

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-trained models have achieved impressive performance on various downstream tasks.However, their large model sizes hinder their utilization on platforms with limited computational resources.We find that directly using smaller pre-trained models and applying magnitude-based pruning on CLIP models leads to inflexibility and inferior performance.Recent efforts for VLP compression either adopt uni-modal compression metrics resulting in limited performance or involve costly mask-search processes with learnable masks.In this paper, we first propose the Module-wise Pruning Error (MoPE) metric, accurately assessing CLIP module importance by performance decline on cross-modal tasks.Using the MoPE metric, we introduce a unified pruning framework applicable to both pre-training and task-specific fine-tuning compression stages. For pre-training, MoPE-CLIP effectively leverages knowledge from the teacher model, significantly reducing pre-training costs while maintaining strong zero-shot capabilities.For fine-tuning, consecutive pruning from width to depth yields highly competitive task-specific models.Extensive experiments in two stages demonstrate the effectiveness of the MoPE metric, and MoPE-CLIP outperforms previous state-of-the-art VLP compression methods.

</details>

---

## 122. Instruct-Imagen: Image Generation with Multi-modal Instruction

- [ ] Instruct-Imagen: Image Generation with Multi-modal Instruction | https://cvpr.thecvf.com/virtual/2024/poster/30449

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30449

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents Instruct-Imagen, a model that tackles heterogeneous image generation tasks and generalizes across unseen tasks.We introduce multi-modal instruction for image generation, a task representation articulating a range of generation intents with precision.It uses natural language to amalgamate disparate modalities (e.g., text, edge, style, subject, \etc), such that abundant generation intents can be standardized in a uniform format.We then build Instruct-Imagen by fine-tuning a pre-trained text-to-image diffusion model with two stages. First, we adapt the model using the retrieval-augmented training, to enhance model's capabilities to ground its generation on external multi-modal context.Subsequently, we fine-tune the adapted model on diverse image generation tasks that requires vision-language understanding (e.g., subject-driven generation, etc.), each paired with a multi-modal instruction encapsulating the task's essence. Human evaluation on various image generation datasets reveals that Instruct-Imagen matches or surpasses prior task-specific models in-domain and demonstrates promising generalization to unseen and more complex tasks. Our evaluation suite will be made publicly available.

</details>

---

## 123. HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data

- [ ] HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data | https://cvpr.thecvf.com/virtual/2024/poster/30455

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30455

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) tuned on machine-generated instruction-following data have demonstrated remarkable performance in various multi-modal understanding and generation tasks. However, the hallucinations inherent in machine-generated data, which could lead to hallucinatory outputs in MLLMs, remain under-explored. This work aims to investigate various hallucinations (i.e., object, relation, attribute hallucinations) and mitigate those hallucinatory toxicities in large-scale machine-generated visual instruction datasets. Drawing on the human ability to identify factual errors, we present a novel hallucination detection and elimination framework, HalluciDoctor, based on the cross-checking paradigm. We use our framework to identify and eliminate hallucinations in the training data automatically. Interestingly, HalluciDoctor also indicates that spurious correlations arising from long-tail object co-occurrences contribute to hallucinations. Based on that, we execute counterfactual visual instruction expansion to balance data distribution, thereby enhancing MLLMs' resistance to hallucinations. Comprehensive experiments on hallucination evaluation benchmarks show that our method successfully mitigates 44.6% hallucinations relatively and maintains competitive performance compared to LLaVA. The data and code for this paper are publicly available.

</details>

---

## 124. Causal-CoG: A Causal-Effect Look at Context Generation for Boosting Multi-modal Language Models

- [ ] Causal-CoG: A Causal-Effect Look at Context Generation for Boosting Multi-modal Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30460

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30460

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While Multi-modal Language Models (\textit{MLMs}) demonstrate impressive multimodal ability, they still struggle on providing factual and precise responses for tasks like visual question answering (\textit{VQA}).In this paper, we address this challenge from the perspective of contextual information. We propose Causal Context Generation, \textbf{Causal-CoG}, which is a prompting strategy that engages contextual information to enhance precise VQA during inference. Specifically, we prompt MLMs to generate contexts, i.e, text description of an image, and engage the generated contexts for question answering. Moreover, we investigate the advantage of contexts on VQA from a causality perspective, introducing causality filtering to select samples for which contextual information is helpful. To show the effectiveness of Causal-CoG, we run extensive experiments on 10 multimodal benchmarks and show consistent improvements, \emph{e.g.}, +6.30\% on POPE, +13.69\% on Vizwiz and +6.43\% on VQAv2 compared to direct decoding, surpassing existing methods. We hope Casual-CoG inspires explorations of context knowledge in multimodal models, and serves as a plug-and-play strategy for MLM decoding.

</details>

---

## 125. ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation

- [ ] ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation | https://cvpr.thecvf.com/virtual/2024/poster/30495

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30495

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Robot manipulation relies on accurately predicting contact points and end-effector directions to ensure successful operation. However, learning-based robot manipulation, trained on a limited category within a simulator, often struggles to achieve generalizability, especially when confronted with extensive categories.Therefore, we introduce an innovative approach for robot manipulation that leverages the robust reasoning capabilities of Multimodal Large Language Models (MLLMs) to enhance the stability and generalization of manipulation. By fine-tuning the injected adapters, we preserve the inherent common sense and reasoning ability of the MLLMs while equipping them with the ability for manipulation. The fundamental insight lies in the introduced fine-tuning paradigm, encompassing object category understanding, affordance prior reasoning, and object-centric pose prediction to stimulate the reasoning ability of MLLM in manipulation. During inference, our approach utilizes an RGB image and text prompt to predict the end effector's pose in chain of thoughts. After the initial contact is established, an active impedance adaptation policy is introduced to plan the upcoming waypoints in a closed-loop manner. Moreover, in real world, we design a test-time adaptation (TTA) strategy for manipulation to enable the model better adapt to the current real-world scene configuration. Experiments in simulator and real-world show the promising performance of ManipLLM.

</details>

---

## 126. Semantics-aware Motion Retargeting with Vision-Language Models

- [ ] Semantics-aware Motion Retargeting with Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30503

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30503

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Capturing and preserving motion semantics is essential to motion retargeting between animation characters. However, most of the previous works neglect the semantic information or rely on human-designed joint-level representations. Here, we present a novel Semantics-aware Motion reTargeting (SMT) method with the advantage of vision-language models to extract and maintain meaningful motion semantics. We utilize a differentiable module to render 3D motions. Then the high-level motion semantics are incorporated into the motion retargeting process by feeding the vision-language model with the rendered images and aligning the extracted semantic embeddings. To ensure the preservation of fine-grained motion details and high-level semantics, we adopt a two-stage pipeline consisting of skeleton-aware pre-training and fine-tuning with semantics and geometry constraints. Experimental results show the effectiveness of the proposed method in producing high-quality motion retargeting results while accurately preserving motion semantics. Project page can be found at  https://sites.google.com/view/smtnet.

</details>

---

## 127. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks

- [ ] Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks | https://cvpr.thecvf.com/virtual/2024/poster/30529

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30529

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Florence-2, a novel vision foundation model with a unified, prompt-based representation for a variety of computer vision and vision-language tasks. While existing large vision models excel in transfer learning, they struggle to perform a diversity of tasks with simple instructions, a capability that implies handling the complexity of various spatial hierarchy and semantic granularity. Florence-2 was designed to take text-prompt as task instructions and generate desirable results in text forms, whether it be captioning, object detection, grounding or segmentation. This multi-task learning setup demands large-scale, high-quality annotated data. To this end, we co-developed FLD-5B that consists of 5.4 billion comprehensive visual annotations on 126 million images, using an iterative strategy of automated image annotation and model refinement. We adopted a sequence-to-sequence structure to train Florence-2 to perform versatile and comprehensive vision tasks. Extensive evaluations on numerous tasks demonstrated Florence-2 to be a strong vision foundation model contender with unprecedented zero-shot and fine-tuning capabilities.

</details>

---

## 128. Interactive Continual Learning: Fast and Slow Thinking

- [ ] Interactive Continual Learning: Fast and Slow Thinking | https://cvpr.thecvf.com/virtual/2024/poster/30562

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30562

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Advanced life forms, sustained by the synergistic interaction of neural cognitive mechanisms, continually acquire and transfer knowledge throughout their lifespan. In contrast, contemporary machine learning paradigms exhibit limitations in emulating the facets of continual learning (CL). Nonetheless, the emergence of large language models (LLMs) presents promising avenues for realizing CL via interactions with these models.Drawing on Complementary Learning System theory, this paper presents a novel Interactive Continual Learning (ICL) framework, enabled by collaborative interactions among models of various sizes. Specifically, we assign the ViT model as System1 and multimodal LLM as System2.To enable the memory module to deduce tasks from class information and enhance Set2Set retrieval, we propose the Class-Knowledge-Task Multi-Head Attention (CKT-MHA).Additionally, to improve memory retrieval in System1 through enhanced geometric representation, we introduce the CL-vMF mechanism, based on the von Mises-Fisher (vMF) distribution. Meanwhile, we introduce the von Mises-Fisher Outlier Detection and Interaction (vMF-ODI) strategy to identify hard examples, thus enhancing collaboration between System1 and System2 for complex reasoning realization.Comprehensive evaluation of our proposed ICL demonstrates significant resistance to forgetting and superior performance relative to existing methods.

</details>

---

## 129. CPLIP: Zero-Shot Learning for Histopathology with Comprehensive Vision-Language Alignment

- [ ] CPLIP: Zero-Shot Learning for Histopathology with Comprehensive Vision-Language Alignment | https://cvpr.thecvf.com/virtual/2024/poster/30577

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30577

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper proposes Comprehensive Pathology Language Image Pre-training (CPLIP), a new unsupervised technique designed to enhance the alignment of images and text in histopathology for tasks such as classification and segmentation. This methodology enriches vision-language models by leveraging extensive data without needing ground truth annotations. CPLIP involves constructing a pathology-specific vocabulary, generating textual descriptions for images using language models, and retrieving relevant images for each text snippet via a pre-trained model.  The model is then fine-tuned using a many-to-many contrastive learning method to align complex interrelated concepts across both modalities. Evaluated across multiple histopathology tasks, CPLIP shows notable improvements in zero-shot learning scenarios, outperforming existing methods in both interpretability and robustness and setting a higher benchmark for the application of vision-language models in the field. To encourage further research and replication, the code for CPLIP is available on GitHub at xxx.

</details>

---

## 130. ECLIPSE: A Resource-Efficient Text-to-Image Prior for Image Generations

- [ ] ECLIPSE: A Resource-Efficient Text-to-Image Prior for Image Generations | https://cvpr.thecvf.com/virtual/2024/poster/30613

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30613

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Text-to-image (T2I) diffusion models, notably the unCLIP models (e.g., DALL-E-2), achieve state-of-the-art (SOTA) performance on various compositional T2I benchmarks, at the cost of significant computational resources. The unCLIP stack comprises T2I prior and diffusion image decoder. The T2I prior model alone adds a billion parameters compared to the Latent Diffusion Models, which increases the computational and high-quality data requirements. We introduce ECLIPSE, a novel contrastive learning method that is both parameter and data-efficient. ECLIPSE leverages pre-trained vision-language models (e.g., CLIP) to distill the knowledge into the prior model. We demonstrate that the ECLIPSE trained prior, with only 3.3% of the parameters and trained on a mere 2.8% of the data, surpasses the baseline T2I priors with an average of 71.6% preference score under resource-limited setting.  It also attains performance on par with SOTA larger models, achieving an average of 63.36% preference score in terms of the ability to follow the text compositions. Extensive experiments on two unCLIP diffusion image decoders, Karlo and Kandinsky, affirm that ECLIPSE consistently delivers high performance while significantly reducing resource dependency.

</details>

---

## 131. Summarize the Past to Predict the Future: Natural Language Descriptions of Context Boost Multimodal Object Interaction Anticipation

- [ ] Summarize the Past to Predict the Future: Natural Language Descriptions of Context Boost Multimodal Object Interaction Anticipation | https://cvpr.thecvf.com/virtual/2024/poster/30633

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30633

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We study object interaction anticipation in egocentric videos. This task requires an understanding of the spatio-temporal context formed by past actions on objects, coined "action context". We propose TransFusion, a multimodal transformer-based architecture for short-term object interaction anticipation. Our method exploits the representational power of language by summarizing the action context textually, after leveraging pre-trained vision-language foundation models to extract the action context from past video frames. The summarized action context and the last observed video frame are processed by the multimodal fusion module to forecast the next object interaction. Experiments on the Ego4D next active object interaction dataset show the effectiveness of our multimodal fusion model and highlight the benefits of using the power of foundation models and language-based context summaries in a task where vision may appear to suffice. Our novel approach outperforms all state-of-the-art methods on both versions of the Ego4D dataset.

</details>

---

## 132. MAFA: Managing False Negatives for Vision-Language Pre-training

- [ ] MAFA: Managing False Negatives for Vision-Language Pre-training | https://cvpr.thecvf.com/virtual/2024/poster/30638

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30638

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We consider a critical issue of false negatives in Vision- Language Pre-training (VLP), a challenge that arises from the inherent many-to-many correspondence of image-text pairs in large-scale web-crawled datasets. The presence of false negatives can impede achieving optimal performance and even lead to a significant performance drop. To address this challenge, we propose MAFA (MAnaging FAlse nega- tives), which consists of two pivotal components building upon the recently developed GRouped mIni-baTch sampling (GRIT) strategy: 1) an efficient connection mining process that identifies and converts false negatives into positives, and 2) label smoothing for the image-text contrastive (ITC) loss. Our comprehensive experiments verify the effectiveness of MAFA across multiple downstream tasks, emphasizing the crucial role of addressing false negatives in VLP, potentially even surpassing the importance of addressing false posi- tives. In addition, the compatibility of MAFA with the recent BLIP-family model is also demonstrated. Code is available at https://github.com/jaeseokbyun/MAFA.

</details>

---

## 133. OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM

- [ ] OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM | https://cvpr.thecvf.com/virtual/2024/poster/30658

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30658

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in various multimodal tasks. However, their potential in the medical domain remains largely unexplored. A significant challenge arises from the scarcity of diverse medical images spanning various modalities and anatomical regions, which is essential in real-world medical applications. To solve this problem, in this paper, we introduce OmniMedVQA, a novel comprehensive medical Visual Question Answering (VQA) benchmark. This benchmark is collected from 73 different medical datasets, including 12 different modalities and covering more than 20 distinct anatomical regions. Importantly, all images in this benchmark are sourced from authentic medical scenarios, ensuring alignment with the requirements of the medical field and suitability for evaluating LVLMs. Through our extensive experiments, we have found that existing LVLMs struggle to address these medical VQA problems effectively. Moreover, what surprises us is that medical-specialized LVLMs even exhibit inferior performance to those general-domain models, calling for a more versatile and robust LVLM in the biomedical field. The evaluation results not only reveal the current limitations of LVLM in understanding real medical images but also highlight our dataset's significance.  Our code with dataset are available at https://github.com/OpenGVLab/Multi-Modality-Arena.

</details>

---

## 134. PairAug: What Can Augmented Image-Text Pairs Do for Radiology?

- [ ] PairAug: What Can Augmented Image-Text Pairs Do for Radiology? | https://cvpr.thecvf.com/virtual/2024/poster/30664

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30664

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current vision-language pre-training (VLP) methodologies predominantly depend on paired image-text datasets, a resource that is challenging to acquire in radiology due to privacy considerations and labelling complexities. Data augmentation provides a practical solution to overcome the issue of data scarcity, however, most augmentation methods exhibit a limited focus, prioritising either image or text augmentation exclusively. Acknowledging this limitation, our objective is to devise a framework capable of concurrently augmenting medical image and text data. We design a Pairwise Augmentation (PairAug) approach that contains an Inter-patient Augmentation (InterAug) branch and an Intra-patient Augmentation (IntraAug) branch. Specifically, the InterAug branch of our approach generates radiology images using synthesised yet plausible reports derived from a Large Language Model (LLM). The generated pairs can be considered a collection of new patient cases since they are artificially created and may not exist in the original dataset.In contrast, the IntraAug branch uses newly generated reports to manipulate images. This process allows us to create new paired data for each individual with diverse medical conditions. Our extensive experiments on various downstream tasks covering medical image classification zero-shot and fine-tuning analysis demonstrate that our PairAug, concurrently expanding both image and text data, substantially outperforms image-/text-only expansion baselines and advanced medical VLP baselines.

</details>

---

## 135. Prompt-Driven Referring Image Segmentation with Instance Contrasting

- [ ] Prompt-Driven Referring Image Segmentation with Instance Contrasting | https://cvpr.thecvf.com/virtual/2024/poster/30670

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30670

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Referring image segmentation (RIS) aims to segment the target referent described by natural language. Recently, large-scale pre-trained models, e.g., CLIP and SAM, have been successfully applied in many downstream tasks, but they are not well adapted to RIS task due to inter-task differences. In this paper, we propose a new prompt-driven framework named Prompt-RIS, which bridges CLIP and SAM end-to-end and transfers their rich knowledge and powerful capabilities to RIS task through prompt learning. To adapt CLIP to pixel-level task, we first propose a Cross-Modal Prompting method, which acquires more comprehensive vision-language interaction and fine-grained text-to-pixel alignment by performing bidirectional prompting.  Then, the prompt-tuned CLIP generates masks, points, and text prompts for SAM to generate more accurate mask predictions. Moreover, we further propose Instance Contrastive Learning to improve the model's discriminability to different instances and robustness to diverse languages describing the same instance. Extensive experiments demonstrate that the performance of our method outperforms the state-of-the-art methods consistently in both general and open-vocabulary settings.

</details>

---

## 136. X-MIC: Cross-Modal Instance Conditioning for Egocentric Action Generalization

- [ ] X-MIC: Cross-Modal Instance Conditioning for Egocentric Action Generalization | https://cvpr.thecvf.com/virtual/2024/poster/30674

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30674

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Lately, there has been growing interest in adapting vision-language models (VLMs) to image and third-person video classification due to their success in zero-shot recognition. However, the adaptation of these models to egocentric videos has been largely unexplored. To address this gap, we propose a simple yet effective cross-modal adaptation framework, which we call X-MIC. Using a video adapter, our pipeline learns to align frozen text embeddings to each egocentric video directly in the shared embedding space. Our novel adapter architecture retains and improves generalization of the pre-trained VLMs by disentangling learnable temporal modeling and frozen visual encoder. This results in an enhanced alignment of text embeddings to each egocentric video, leading to a significant improvement in cross-dataset generalization. We evaluate our approach on the Epic-Kitchens, Ego4D, and EGTEA datasets for fine-grained cross-dataset action generalization, demonstrating the effectiveness of our method.

</details>

---

## 137. RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding

- [ ] RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding | https://cvpr.thecvf.com/virtual/2024/poster/30681

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30681

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose a lightweight and scalable Regional Point-Language Contrastive learning framework, namely RegionPLC, for open-world 3D scene understanding, aiming to identify and recognize open-set objects and categories. Specifically, based on our empirical studies, we introduce a 3D-aware SFusion strategy that fuses 3D vision-language pairs derived from multiple 2D foundation models, yielding high-quality, dense region-level language descriptions without human 3D annotations. Subsequently, we devise a region-aware point-discriminative contrastive learning objective to enable robust and effective 3D learning from dense regional language supervision. We carry out extensive experiments on ScanNet, ScanNet200, and nuScenes datasets, and our model outperforms prior 3D open-world scene understanding approaches by an average of 17.2\% and 9.1\% for semantic and instance segmentation, respectively, while maintaining greater scalability and lower resource demands. Furthermore, our method has the flexibility to be effortlessly integrated with language models to enable open-ended grounded 3D reasoning without extra task-specific training. Code will be released.

</details>

---

## 138. SC-Tune: Unleashing Self-Consistent Referential Comprehension in Large Vision Language Models

- [ ] SC-Tune: Unleashing Self-Consistent Referential Comprehension in Large Vision Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30700

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30700

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent trends in Large Vision Language Models (LVLMs) research have been increasingly focusing on advancing beyond general image understanding towards more nuanced, object-level referential comprehension. In this paper, we present and delve into the self-consistency capability of LVLMs, a crucial aspect that reflects the models' ability to both generate informative captions for specific objects and subsequently utilize these captions to accurately re-identify the objects in a closed-loop process. This capability significantly mirrors the precision and reliability of fine-grained visual-language understanding.Our findings reveal that the self-consistency level of existing LVLMs falls short of expectations, posing limitations on their practical applicability and potential. To address this gap, we introduce a novel fine-tuning paradigm named \textbf{Self-Consistency Tuning (SC-Tune)}. It features the synergistic learning of a cyclic describer-locator system. This paradigm is not only data-efficient but also exhibits generalizability across multiple LVLMs. Through extensive experiments, we demonstrate that SC-Tune significantly elevates performance across a spectrum of object-level vision-language benchmarks and maintains competitive or improved performance on image-level vision-language benchmarks. Both our model and code will be publicly available.

</details>

---

## 139. SEED-Bench: Benchmarking Multimodal Large Language Models

- [ ] SEED-Bench: Benchmarking Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30703

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30703

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs), building upon the foundation of powerful large language models (LLMs), have recently demonstrated exceptional capabilities in generating not only texts but also images given interleaved multimodal inputs (acting like a combination of GPT-4V and DALL-E 3). However, existing MLLM benchmarks remain limited to assessing only models' comprehension ability of single image-text inputs, failing to keep up with the strides made in MLLMs. A comprehensive benchmark is imperative for investigating the progress and uncovering the limitations of current MLLMs. In this work, we categorize the capabilities of MLLMs into hierarchical levels from $L_0$ to $L_4$ based on the modalities they can accept and generate, and propose SEED-Bench, a comprehensive benchmark that evaluates the hierarchical capabilities of MLLMs. Specifically, SEED-Bench comprises 24K multiple-choice questions with accurate human annotations, which spans 27 dimensions, including the evaluation of both text and image generation. Multiple-choice questions with groundtruth options derived from human annotation enables an objective and efficient assessment of model performance, eliminating the need for human or GPT intervention during evaluation. We further evaluate the performance of 22 prominent open-source MLLMs and summarize valuable observations. By revealing the limitations of existing MLLMs through extensive evaluations, we aim for SEED-Bench to provide insights that will motivate future research towards the goal of General Artificial Intelligence.

</details>

---

## 140. Generating Illustrated Instructions

- [ ] Generating Illustrated Instructions | https://cvpr.thecvf.com/virtual/2024/poster/30717

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30717

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a new task of generating “Illustrated Instructions”, i.e. visual instructions customized to a user’s needs. We identify desiderata unique to this task, and formalize it through a suite of automatic and human evaluation metrics, designed to measure the validity, consistency, and efficacy of the generations. We combine the power of large language models (LLMs) together with strong text-to-image generation diffusion models to propose a simple approach called StackedDiffusion, that generates such illustrated instructions given text as input. The resulting model strongly outperforms baseline approaches and state-of-the-art multimodal LLMs; and in 30% of cases, users even prefer it to human-generated articles. Most notably, it enables various new and exciting applications far beyond what static articles on the web can provide, such as personalized instructions complete with intermediate steps and pictures in response to a user’s individual situation.

</details>

---

## 141. VidLA: Video-Language Alignment at Scale

- [ ] VidLA: Video-Language Alignment at Scale | https://cvpr.thecvf.com/virtual/2024/poster/30730

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30730

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose VidLA, an approach for video-language alignment at scale. There are two major limitations of previous video-language alignment approaches. First, they do not capture both short-range and long-range temporal dependencies, and typically employ complex hierarchical deep network architectures that are hard to integrate with existing pretrained image-text foundation models. To effectively address this limitation, we instead keep the network architecture simple and use a set of data tokens that operate at different temporal resolutions in a hierarchical manner, accounting for the temporally hierarchical nature of videos. By employing a simple two-tower architecture, we are able to initialize our video-language model with pretrained image-text foundation models, thereby boosting the final performance. Second, existing video-language alignment works struggle due to the lack of semantically aligned large-scale training data. To overcome it, we leverage recent LLMs to curate the largest video-language dataset to-date with better visual grounding. Furthermore, unlike existing video-text datasets which only contains short clips, our dataset is enriched with video clips of varying durations to aid our temporally hierarchical data tokens in extracting better representations at varying temporal scales. Overall, empirical results show that our proposed approach surpasses state-of-the-art methods on multiple retrieval benchmarks, especially on longer videos, and performs competitively on classification benchmarks.

</details>

---

## 142. One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models

- [ ] One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30782

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30782

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large pre-trained Vision-Language Models (VLMs) like CLIP, despite having remarkable generalization ability, are highly vulnerable to adversarial examples. This work studies the adversarial robustness of VLMs from the novel perspective of the text prompt instead of the extensively studied model weights (frozen in this work). We first show that the effectiveness of both adversarial attack and defense are sensitive to the used text prompt.Inspired by this, we propose a method to improve resilience to adversarial attacks by learning a robust text prompt for VLMs. The proposed method, named Adversarial Prompt Tuning (APT), is effective while being both computationally and data efficient.Extensive experiments are conducted across 15 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show APT's superiority over hand-engineered prompts and other state-of-the-art adaption methods. APT demonstrated excellent abilities in terms of the in-distribution performance and the generalization under input distribution shift and across datasets.Surprisingly, by simply adding one learned word to the prompts, APT can significantly boost the accuracy and robustness ($\epsilon=4/255$) over the hand-engineered prompts by +13% and +8.5% on average respectively. The improvement further increases, in our most effective setting, to +26.4% for accuracy and +16.7% for robustness.

</details>

---

## 143. Prompt Highlighter: Interactive Control for Multi-Modal LLMs

- [ ] Prompt Highlighter: Interactive Control for Multi-Modal LLMs | https://cvpr.thecvf.com/virtual/2024/poster/30785

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30785

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study targets a critical aspect of multi-modal LLMs' (LLMs\&VLMs) inference: explicit controllable text generation. Multi-modal LLMs empower multi-modality understanding with the capability of semantic generation yet bring less explainability and heavier reliance on prompt contents due to their autoregressive generative nature. While manipulating prompt formats could improve outputs, designing specific and precise prompts per task can be challenging and ineffective. To tackle this issue, we introduce a novel inference method, Prompt Highlighter, which enables users to highlight specific prompt spans to interactively control the focus during generation. Motivated by the classifier-free diffusion guidance, we form regular and unconditional context pairs based on highlighted tokens, demonstrating that the autoregressive generation in models can be guided in a classifier-free way. Notably, we find that, during inference, guiding the models with highlighted tokens through the attention weights leads to more desired outputs. Our approach is compatible with current LLMs and VLMs, achieving impressive customized generation results without training. Experiments confirm its effectiveness in focusing on input contexts and generating reliable content. Without tuning on LLaVA-v1.5, our method secured 70.7 in the MMBench test and 1552.5 in MME-perception. Code is available at https://github.com/dvlab-research/Prompt-Highlighter

</details>

---

## 144. GROUNDHOG: Grounding Large Language Models to Holistic Segmentation

- [ ] GROUNDHOG: Grounding Large Language Models to Holistic Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/30796

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30796

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Most multimodal large language models (MLLMs) learn language-to-object grounding through causal language modeling where grounded objects are captured by bounding boxes as sequences of location tokens. This paradigm lacks pixel-level representations that are important for fine-grained visual understanding and diagnosis. In this work, we introduce GROUNDHOG, an MLLM developed by Grounding Large Language Models to holistic Segmentation. GROUNDHOG incorporates a masked feature extractor and converts extracted features into visual entity tokens for the MLLM backbone, which then connects groundable phrases to unified grounding masks by retrieving and merging the entity masks. To train GROUNDHOG, we carefully curated a grounded visual instruction tuning dataset - Multi-Modal Multi-Grained Grounding (M3G2) - by harvesting a collection of segmentation-grounded datasets with rich annotations. Our experimental results show that GROUNDHOG achieves superior performance on various language grounding tasks without task-specific fine-tuning. GROUNDHOG demonstrates better grounding towards complex forms of visual input and provides easy-to-understand diagnosis in failure cases.

</details>

---

## 145. PELA: Learning Parameter-Efficient Models with Low-Rank Approximation

- [ ] PELA: Learning Parameter-Efficient Models with Low-Rank Approximation | https://cvpr.thecvf.com/virtual/2024/poster/30829

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30829

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Applying a pre-trained large model to downstream tasks is prohibitive under resource-constrained conditions.Recent dominant approaches for addressing efficiency issues involve adding a few learnable parameters to the fixed backbone model.This strategy, however, leads to more challenges in loading large models for downstream fine-tuning with limited resources.In this paper, we propose a novel method for increasing the parameter efficiency of pre-trained models by introducing an intermediate pre-training stage.To this end, we first employ low-rank approximation to compress the original large model and then devise a feature distillation module and a weight perturbation regularization module.These modules are specifically designed to enhance the low-rank model.In particular, we update only the low-rank model while freezing the backbone parameters during pre-training. This allows for direct and efficient utilization of the low-rank model for downstream fine-tuning tasks.The proposed method achieves both efficiencies in terms of required parameters and computation time while maintaining comparable results with minimal modifications to the backbone architecture.Specifically, when applied to three vision-only and one vision-language Transformer models, our approach often demonstrates a merely $\sim$0.6 point decrease in performance while reducing the original parameter size by 1/3 to 2/3.The code has been released in the supplementary material for reproduction.

</details>

---

## 146. Scene-adaptive and Region-aware Multi-modal Prompt for Open Vocabulary Object Detection

- [ ] Scene-adaptive and Region-aware Multi-modal Prompt for Open Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2024/poster/30840

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30840

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open Vocabulary Object Detection (OVD) aims to detect objects from novel classes described by text inputs based on the generalization ability of trained classes. Existing methods mainly focus on transferring knowledge from large Vision and Language models (VLM) to detectors based on knowledge distillation. However, these approaches show weak ability in adaptation to diverse classes and alignment between the image-level pre-training and region-level detection, hindering successful knowledge transfer. Motivated by the prompt tuning, we propose scene-adaptive and region-aware multi-modal prompts to address these issues by effectively adapting class-aware knowledge from VLM to the detector at the region level. Specifically, to enhance the adaptability to diverse classes, we design a scene-adaptive prompt generator from a scene perspective to consider both the commonality and diversity of the class distributions, and formulate a novel selection mechanism to facilitate the acquisition of common knowledge across all classes and specific insights relevant to each scene. Meanwhile, to bridge the gap between the pre-trained model and the detector, we present a region-aware multi-modal alignment module, which employs the region prompt to incorporate the positional information for feature distillation and integrates textual prompts to align visual and linguistic representations. Extensive experimental results demonstrate that the proposed method significantly outperforms the state-of-the-art models on the OV-COCO and OV-LVIS datasets, surpassing the current method by 3.0% mAP and 4.6% $\text{AP}_r$.

</details>

---

## 147. Bayesian Exploration of Pre-trained Models for Low-shot Image Classification

- [ ] Bayesian Exploration of Pre-trained Models for Low-shot Image Classification | https://cvpr.thecvf.com/virtual/2024/poster/30855

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30855

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Low-shot image classification is a fundamental task in computer vision, and the emergence of large-scale vision-language models such as CLIP has greatly advanced the forefront of research in this field. However, most existing CLIP-based methods lack the flexibility to effectively incorporate other pre-trained models that encompass knowledge distinct from CLIP. To bridge the gap, this work proposes a simple and effective probabilistic model ensemble framework based on Gaussian processes, which have previously demonstrated remarkable efficacy in processing small data. We achieve the integration of prior knowledge by specifying the mean function with CLIP and the kernel function with an ensemble of deep kernels built upon various pre-trained models. By regressing the classification label directly, our framework enables analytical inference, straightforward uncertainty quantification, and principled hyper-parameter tuning. Through extensive experiments on standard benchmarks, we demonstrate that our method consistently outperforms competitive ensemble baselines regarding predictive performance. Additionally, we assess the robustness of our method and the quality of the yielded uncertainty estimates on out-of-distribution datasets. We also illustrate that our method, despite relying on label regression, still enjoys superior model calibration compared to most deterministic baselines.

</details>

---

## 148. What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models

- [ ] What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30858

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30858

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Counterfactual reasoning, a fundamental aspect of human cognition, involves contemplating alternatives to established facts or past events, significantly enhancing our abilities in planning and decision-making. In light of the advancements in current multi-modal large language models, we explore their effectiveness in counterfactual reasoning. To facilitate this investigation, we introduce a novel dataset, C-VQA, specifically designed to examine the counterfactual reasoning capabilities of modern multi-modal large language models. This dataset is constructed by infusing original questions with counterfactual presuppositions, spanning various types such as numerical and boolean queries. It encompasses a mix of real and synthetic data, representing a wide range of difficulty levels. Our thorough evaluations of contemporary vision-language models using this dataset have revealed substantial performance drops, with some models showing up to a 40\% decrease, highlighting a significant gap between current models and human-like vision reasoning capabilities. We hope our dataset will serve as a vital benchmark for evaluating the counterfactual reasoning capabilities of models.Code and dataset are publicly available at https://bzhao.me/C-VQA/.

</details>

---

## 149. A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models

- [ ] A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30861

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30861

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Efficient transfer learning (ETL) is receiving increasing attention to adapt large pre-trained language-vision models on downstream tasks with a few labeled samples. While significant progress has been made, we reveal that state-of-the-art ETL approaches exhibit strong performance only in narrowly-defined experimental setups, and with a careful adjustment of hyperparameters based on a large corpus of labeled samples. In particular, we make two interesting, and surprising empirical observations. First, to outperform a simple Linear Probing baseline, these methods require to optimize their hyper-parameters on each target task. And second, they typically underperform --sometimes dramatically-- standard zero-shot predictions in the presence of distributional drifts. Motivated by the unrealistic assumptions made in the existing literature, i.e., access to a large validation set and case-specific grid-search for optimal hyperparameters, we propose a novel approach that meets the requirements of real-world scenarios. More concretely, we introduce a CLass-Adaptive linear Probe (CLAP) objective, whose balancing term is optimized via an adaptation of the general Augmented Lagrangian method tailored to this context. We comprehensively evaluate CLAP on a broad span of datasets and scenarios, demonstrating that it consistently outperforms SoTA approaches, while yet being a much more efficient alternative.

</details>

---

## 150. VILA: On Pre-training for Visual Language Models

- [ ] VILA: On Pre-training for Visual Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30868

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30868

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual language models (VLMs) rapidly progressed with the recent success of large language models. There have been growing efforts on visual instruction tuning to extend the LLM with visual inputs, but lacks an in-depth study of the visual language pre-training process, where the model learns to perform joint modeling on both modalities. In this work, we examine the design options for VLM pre-training by augmenting LLM towards VLM through step-by-step controllable comparisons. We introduce three main findings: (1) freezing LLMs during pre-training can achieve decent zero-shot performance, but lack in-context learning capability, which requires unfreezing the LLM; (2) interleaved pre-training data is beneficial whereas image-text pairs alone are not optimal; (3) re-blending text-only instruction data to image-text data during instruction fine-tuning not only remedies the degradation of text-only tasks, but also boosts VLM task accuracy. With an enhanced pre-training recipe we build VILA ,  a Vi sual La nguage model family that consistently outperforms the state-of-the-art models, e.g., LLaVA-1.5, across main benchmarks without bells and whistles. Multi-modal pre-training also helps unveil appealing properties of VILA, including multi-image reasoning, enhanced in-context learning, and better world knowledge.

</details>

---

## 151. Discovering and Mitigating Visual Biases through Keyword Explanation

- [ ] Discovering and Mitigating Visual Biases through Keyword Explanation | https://cvpr.thecvf.com/virtual/2024/poster/30884

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30884

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Addressing biases in computer vision models is crucial for real-world AI system deployments. However, mitigating visual biases is challenging due to their unexplainable nature, often identified indirectly through visualization or sample statistics, which necessitates additional human supervision for interpretation. To tackle this issue, we propose the Bias-to-Text (B2T) framework, which interprets visual biases as keywords. Specifically, we extract common keywords from the captions of mispredicted images to identify potential biases in the model. We then validate these keywords by measuring their similarity to the mispredicted images using a vision-language scoring model. The keyword explanation form of visual bias offers several advantages, such as a clear group naming for bias discovery and a natural extension for debiasing using these group names. Our experiments demonstrate that B2T can identify known biases, such as gender bias in CelebA, background bias in Waterbirds, and distribution shifts in ImageNet-R and ImageNet-C. Additionally, B2T uncovers novel biases in larger datasets, such as Dollar Street and ImageNet. For example, we discovered a contextual bias between "bee" and "flower" in ImageNet. We also highlight various applications of B2T keywords, including debiased training, CLIP prompting, model comparison, and label diagnosis.

</details>

---

## 152. MULTIFLOW: Shifting Towards Task-Agnostic Vision-Language Pruning

- [ ] MULTIFLOW: Shifting Towards Task-Agnostic Vision-Language Pruning | https://cvpr.thecvf.com/virtual/2024/poster/30898

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30898

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While excellent in transfer learning, Vision-Language models (VLMs) come with high computational costs due to their large number of parameters. To address this issue, removing parameters via model pruning is a viable solution. However, existing techniques for VLMs are task-specific, and thus require pruning the network from scratch for each new task of interest. In this work, we explore a new direction: Task-Agnostic Vision-Language Pruning (TA-VLP). Given a pretrained VLM, the goal is to find a unique pruned counterpart transferable to multiple unknown downstream tasks. In this challenging setting, the transferable representations already encoded in the pretrained model are a key aspect to preserve. Thus, we propose Multimodal Flow Pruning (MULTIFLOW), a first, gradient-free, pruning framework for TA-VLP where: (i) the importance of a parameter is expressed in terms of its magnitude and its information flow, by incorporating the saliency of the neurons it connects; and (ii) pruning is driven by the emergent (multimodal) distribution of the VLM parameters after pretraining. We benchmark eight state-of-the-art pruning algorithms in the context of TA-VLP, experimenting with two VLMs, three vision-language tasks, and three pruning ratios. Our experimental results show that MULTIFLOW outperforms recent sophisticated, combinatorial competitors in the vast majority of the cases, paving the way towards addressing TA-VLP. The code is publicly available at https://github.com/FarinaMatteo/multiflow.

</details>

---

## 153. Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models

- [ ] Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models | https://cvpr.thecvf.com/virtual/2024/poster/30904

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30904

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs) have shown promise in vision-language tasks but struggle with high-resolution input and detailed scene understanding. Addressing these challenges, we introduce Monkey to enhance LMM capabilities. Firstly, Monkey processes input images by dividing them into uniform patches, each matching the size (e.g., 448$\times$448) used in the original training of the well-trained vision encoder. Equipped with individual adapter for each patch, Monkey can handle higher resolutions up to 1344$\times$896 pixels, enabling the detailed capture of complex visual information. Secondly, it employs a multi-level description generation method, enriching the context for scene-object associations. This two-part strategy ensures more effective learning from generated data: the higher resolution allows for a more detailed capture of visuals, which in turn enhances the effectiveness of comprehensive descriptions. Extensive ablative results validate the effectiveness of our designs. Additionally, experiments on 18 datasets further demonstrate that Monkey surpasses existing LMMs in many tasks like Image Captioning and various Visual Question Answering formats. Specially, in qualitative tests focused on dense text question answering, Monkey has exhibited encouraging results compared with GPT4V. Code is available at https://github.com/Yuliang-Liu/Monkey.

</details>

---

## 154. Modeling Collaborator: Enabling Subjective Vision Classification With Minimal Human Effort via LLM Tool-Use

- [ ] Modeling Collaborator: Enabling Subjective Vision Classification With Minimal Human Effort via LLM Tool-Use | https://cvpr.thecvf.com/virtual/2024/poster/30909

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30909

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

From content moderation to wildlife conservation, the number of applications that require models to recognize nuanced or subjective visual concepts is growing. Traditionally, developing classifiers for such concepts requires substantial manual effort measured in hours, days, or even months to identify and annotate data needed for training. Even with recently proposed Agile Modeling techniques, which enable rapid bootstrapping of image classifiers, users are still required to spend 30 minutes or more of monotonous, repetitive data labeling just to train a single classifier. Drawing on Fiske’s Cognitive Miser theory, we propose a new framework that alleviates manual effort by replacing human labeling with natural language interactions – reducing the total effort required to define a concept by an order of magnitude: from labeling 2,000 images to only 100 plus some natural language interactions. Our framework leverages the recent advances in foundation models, both large language models and vision-language models, to carve out the concept space through conversation and by automatically labeling training data points. Most importantly, our framework eliminates the need for crowd-sourced annotations. Moreover, our framework ultimately produces light-weight classification models that are deployable in cost-sensitive scenarios. Across 15 subjective concepts and across 2 public image classification datasets, our trained models outperform traditional Agile Modeling as well as state-of-the-art zero-shot classification models like ALIGN, CLIP, CuPL, and large visual question answering models like PaLI-X.

</details>

---

## 155. Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models

- [ ] Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/30913

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30913

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Solving complex visual tasks such as ``Who invented the musical instrument on the right?'' involves a composition of skills: understanding space, recognizing instruments, and also retrieving prior knowledge. Recent work shows promise by decomposing such tasks using a large language model (LLM) into an executable program that invokes specialized vision models. However, generated programs are error-prone: they omit necessary steps, include spurious ones, and are unable to recover when the specialized models give incorrect outputs.Moreover, they require loading multiple models, incurring high latency and computation costs. We propose Visual Program Distillation (VPD), an instruction tuning framework that produces a vision-language model (VLM) capable of solving complex visual tasks with a single forward pass. VPD distills the reasoning ability of LLMs by using them to sample multiple candidate programs, which are then executed and verified to identify a correct one. It translates each correct program into a language description of the reasoning steps, which are then distilled into a VLM. Extensive experiments show that VPD improves the VLM's ability to count, understand spatial relations, and reason compositionally. Our VPD-trained PaLI-X outperforms all prior VLMs, achieving state-of-the-art performance across complex vision tasks, including MMBench, OK-VQA, A-OKVQA, TallyQA, POPE, and Hateful Memes. An evaluation with human annotators also confirms that VPD improves model response factuality and consistency. Finally, experiments on content moderation demonstrate that VPD is also helpful for adaptation to real-world applications with limited data.

</details>

---

## 156. Text-conditional Attribute Alignment across Latent Spaces for 3D Controllable Face Image Synthesis

- [ ] Text-conditional Attribute Alignment across Latent Spaces for 3D Controllable Face Image Synthesis | https://cvpr.thecvf.com/virtual/2024/poster/30922

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30922

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the advent of generative models and vision-language pretraining, significant improvement has been made in text-driven face manipulation. The text embedding can be used as target supervision for expression control. However, it is non-trivial to associate with its 3D attributes, \ie, pose and illumination. To address these issues, we propose a Text-conditional Attribute aLignment approach for 3D controllable face image synthesis, and our model is referred to as TcALign. Specifically, since the 3D rendered image can be precisely controlled with the 3D face representation, we first propose a Text-conditional 3D Editor to produce the target face representation to realize text-driven manipulation in the 3D space. An attribute embedding space spanned by the target-related attributes embeddings is also introduced to infer the disentangled task-specific direction.Next, we train a cross-modal latent mapping network conditioned on the derived difference of 3D representation to infer a correct vector in the latent space of StyleGAN. This correction vector learning design can accurately transfer the attribute manipulation on 3D images to 2D images. We show that the proposed method delivers more precise text-driven multi-attribute manipulation for 3D controllable face image synthesis. Extensive qualitative and quantitative experiments verify the effectiveness and superiority of our method over the other competing methods.

</details>

---

## 157. Large Language Models are Good Prompt Learners for Low-Shot Image Classification

- [ ] Large Language Models are Good Prompt Learners for Low-Shot Image Classification | https://cvpr.thecvf.com/virtual/2024/poster/30926

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30926

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Low-shot image classification, where training images are limited or inaccessible, has benefited from recent progress on pre-trained vision-language (VL) models with strong generalizability, e.g. CLIP. Prompt learning methods built with VL models generate text features from the class names that only have confined class-specific information. Large Language Models (LLMs), with their vast encyclopedic knowledge, emerge as the complement. Thus, in this paper, we discuss the integration of LLMs to enhance pre-trained VL models, specifically on low-shot classification. However, the domain gap between language and vision blocks the direct application of LLMs. Thus, we propose LLaMP, Large Language Models as Prompt learners, that produces adaptive prompts for the CLIP text encoder, establishing it as the connecting bridge. Experiments show that, compared with other state-of-the-art prompt learning methods, LLaMP yields better performance on both zero-shot generalization and few-shot image classification, over a spectrum of 11 datasets.

</details>

---

## 158. Retrieval-Augmented Open-Vocabulary Object Detection

- [ ] Retrieval-Augmented Open-Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2024/poster/30943

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30943

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary object detection (OVD) has been studied with Vision-Language Models (VLMs) to detect novel objects beyond the pre-trained categories. Previous approaches improve the generalization ability to expand the knowledge of the detector, using 'positive' pseudo-labels with additional 'class' names, e.g., sock, iPod, and alligator. To extend the previous methods in two aspects, we propose Retrieval-Augmented Losses and visual Features (RALF). Our method retrieves related 'negative' classes and augments loss functions. Also, visual features are augmented with 'verbalized concepts' of classes, e.g., worn on the feet, handheld music player, and sharp teeth. Specifically, RALF consists of two modules: Retrieval Augmented Losses (RAL) and Retrieval-Augmented visual Features (RAF). RAL constitutes two losses reflecting the semantic similarity with negative vocabularies. In addition, RAF augments visual features with the verbalized concepts from a large language model (LLM). Our experiments demonstrate the effectiveness of RALF on COCO and LVIS benchmark datasets. We achieve improvement up to 3.4 box $AP_{50}^{\text{N}}$ on novel categories of the COCO dataset and 3.6 mask $AP_{\text{r}}$ gains on the LVIS dataset. Code is available at https://github.com/mlvlab/RALF.

</details>

---

## 159. CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing

- [ ] CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing | https://cvpr.thecvf.com/virtual/2024/poster/30953

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30953

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Domain generalization (DG) based Face Anti-Spoofing (FAS) aims to improve the model's performance on unseen domains. Existing methods either rely on domain labels to align domain-invariant feature spaces, or disentangle generalizable features from the whole sample, which inevitably lead to the distortion of semantic feature structures and achieve limited generalization. Instead of directly manipulating visual features, we make use of large-scale vision-language models (VLMs) like CLIP and leverage the textual feature to dynamically adjust the classifier's weights for exploring generalizable visual features. Specifically, we propose a novel Class Free Prompt Learning (CFPL) paradigm for DG FAS, which utilizes two lightweight transformers, namely Content Q-Former (CQF) and Style Q-Former (SQF), to learn the different semantic prompts conditioned on content and style features by using a set of learnable query vectors, respectively. Thus, the generalizable prompt can be learned by two improvements: (1) A Prompt-Text Matched (PTM) supervision is introduced to ensure CQF learns visual representation that is most informative of the content description. (2) A Diversified Style Prompt (DSP) technology is proposed to diversify the learning of style prompts by mixing feature statistics between instance-specific styles. Finally, the learned text features modulate visual features to generalization through the designed Prompt Modulation (PM). Extensive experiments show that the CFPL is effective and outperforms the state-of-the-art methods on several cross-domain datasets.

</details>

---

## 160. OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation

- [ ] OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation | https://cvpr.thecvf.com/virtual/2024/poster/30961

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/30961

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Hallucination, posed as a pervasive challenge of multi-modal large language models (MLLMs), has significantly impeded their real-world usage that demands precise judgment. Existing methods mitigate this issue with either training with specific designed data or inferencing with external knowledge from other sources, incurring inevitable additional costs. In this paper, we present $\textbf{OPERA}$, a novel MLLM decoding method grounded in an $\textbf{O}$ver-trust $\textbf{Pe}$nalty and a $\textbf{R}$etrospection-$\textbf{A}$llocation strategy, serving as a nearly $\textbf{free lunch}$ to alleviate the hallucination issue without additional data, knowledge, or training. Our approach begins with an interesting observation that, most hallucinations are closely tied to the knowledge aggregation patterns manifested in the self-attention matrix, i.e., MLLMs tend to generate new tokens by focusing on a few summary tokens, but not all the previous tokens. Such partial over-trust inclination results in the neglecting of image tokens and describes the image content with hallucination. Based on the observation, OPERA introduces a penalty term on the model logits during the beam-search decoding to mitigate the over-trust issue, along with a rollback strategy that retrospects the presence of summary tokens in the previously generated tokens, and re-allocate the token selection if necessary. With extensive experiments, OPERA shows significant hallucination-mitigating performance on different MLLMs and metrics, proving its effectiveness and generality. Our code is at: https://github.com/shikiw/OPERA.

</details>

---

## 161. Can I Trust Your Answer? Visually Grounded Video Question Answering

- [ ] Can I Trust Your Answer? Visually Grounded Video Question Answering | https://cvpr.thecvf.com/virtual/2024/poster/31031

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31031

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We study visually grounded VideoQA in response to the emerging trends of utilizing pretraining techniques for video-language understanding. Specifically, by forcing vision-language models (VLMs) to answer questions and simultaneously provide visual evidence, we seek to ascertain the extent to which the predictions of such techniques are genuinely anchored in relevant video content, versus spurious correlations from language or irrelevant visual context. Towards this, we construct NExT-GQA -- an extension of NExT-QA with 10.5$K$ temporal grounding (or location) labels tied to the original QA pairs. With NExT-GQA, we scrutinize a series of state-of-the-art VLMs. Through post-hoc attention analysis, we find that these models are extremely weak in substantiating the answers despite their strong QA performance. This exposes the limitation of current VLMs in making reliable predictions. As a remedy, we further explore and propose a grounded-QA method via Gaussian mask optimization and cross-modal learning. Experiments with different backbones demonstrate that this grounding mechanism improves both grounding and QA. Our dataset and code will be released. With these efforts, we aim to push towards trustworthy VLMs in VQA systems.

</details>

---

## 162. Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models

- [ ] Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31048

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31048

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the emergence of pre-trained vision-language models like CLIP, how to adapt them to various downstream classification tasks has garnered significant attention in recent research. The adaptation strategies can be typically categorized into three paradigms: zero-shot adaptation, few-shot adaptation, and the recently-proposed training-free few-shot adaptation. Most existing approaches are tailored for a specific setting and can only cater to one or two of these paradigms. In this paper, we introduce a versatile adaptation approach that can effectively work under all three settings.  Specifically, we propose the dual memory networks that comprise dynamic and static memory components. The static memory caches training data knowledge, enabling training-free few-shot adaptation, while the dynamic memory preserves historical test features online during the testing process, allowing for the exploration of additional data insights beyond the training set. This novel capability enhances model performance in the few-shot setting and enables model usability in the absence of training data.The two memory networks employ the same flexible memory interactive strategy, which can operate in a training-free mode and can be further enhanced by incorporating learnable projection layers.  Our approach is tested across 11 datasets under the three task settings. Remarkably, in the zero-shot scenario, it outperforms existing methods by over 3\% and even shows superior results against methods utilizing external training data. Additionally, our method exhibits robust performance against natural distribution shifts. Codes are available at \url{https://github.com/YBZh/DMN}.

</details>

---

## 163. PerceptionGPT: Effectively Fusing Visual Perception into LLM

- [ ] PerceptionGPT: Effectively Fusing Visual Perception into LLM | https://cvpr.thecvf.com/virtual/2024/poster/31054

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31054

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The integration of visual inputs with large language models (LLMs) has led to remarkable advancements in multi-modal capabilities, giving rise to vision large language models (VLLMs). However, effectively harnessing LLMs for intricate visual perception tasks, such as detection and segmentation, remains a challenge. Conventional approaches achieve this by transforming perception signals (e.g., bounding boxes, segmentation masks) into sequences of discrete tokens, which struggle with the precision errors and introduces further complexities for training. In this paper, we present a novel end-to-end framework named PerceptionGPT, which represent the perception signals using LLM's dynamic token embedding. Specifically, we leverage lightweight encoders and decoders to handle the perception signals in LLM's embedding space, which takes advantage of the representation power of the high-dimensional token embeddings. Our approach significantly eases the training difficulties associated with the discrete representations in prior methods. Furthermore, owing to our compact representation, the inference speed is also greatly boosted. Consequently, PerceptionGPT enables accurate, flexible and efficient handling of complex perception signals. We validate the effectiveness of our approach through extensive experiments. The results demonstrate significant improvements over previous methods with only 4% trainable parameters and less than 25% training time.

</details>

---

## 164. Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models

- [ ] Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models | https://cvpr.thecvf.com/virtual/2024/poster/31067

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31067

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generative models have recently exhibited exceptional capabilities in text-to-image generation, but still struggle to generate image sequences coherently. In this work, we focus on a novel, yet challenging task of generating a coherent image sequence based on a given storyline, denoted as open-ended visual storytelling.We make the following three contributions:(i) to fulfill the task of visual storytelling,we propose a learning-based auto-regressive image generation model, termed as StoryGen, with a novel vision-language context module, that enables to generate the current frame by conditioning on the corresponding text prompt and preceding image-caption pairs;(ii) to address the data shortage of visual storytelling, we collect paired image-text sequences by sourcing from online videos and open-source E-books, establishing processing pipeline for constructing a large-scale dataset with diverse characters, storylines, and artistic styles, named StorySalon;(iii) Quantitative experiments and human evaluations have validated the superiority of our StoryGen, where we show StoryGen can generalize to unseen characters without any optimization, and generate image sequences with coherent content and consistent character.The code, model, and dataset will be made publicly available to the research community.

</details>

---

## 165. SonicVisionLM: Playing Sound with Vision Language Models

- [ ] SonicVisionLM: Playing Sound with Vision Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31076

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31076

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

There has been a growing interest in the task of generating sound for silent videos, primarily because of its practicality in streamlining video post-production. However, existing methods for video-sound generation attempt to directly create sound from visual representations, which can be challenging due to the difficulty of aligning visual representations with audio representations. In this paper, we present SonicVisionLM, a novel framework aimed at generating a wide range of sound effects by leveraging vision-language models(VLMs). Instead of generating audio directly from video, we use the capabilities of powerful VLMs. When provided with a silent video, our approach first identifies events within the video using a VLM to suggest possible sounds that match the video content. This shift in approach transforms the challenging task of aligning image and audio into more well-studied sub-problems of aligning image-to-text and text-to-audio through the popular diffusion models. To improve the quality of audio recommendations with LLMs, we have collected an extensive dataset that maps text descriptions to specific sound effects and developed a time-controlled audio adapter. Our approach surpasses current state-of-the-art methods for converting video to audio, enhancing synchronization with the visuals, and improving alignment between audio and video components. Project page: https://yusiissy.github.io/SonicVisionLM.github.io/

</details>

---

## 166. GLaMM: Pixel Grounding Large Multimodal Model

- [ ] GLaMM: Pixel Grounding Large Multimodal Model | https://cvpr.thecvf.com/virtual/2024/poster/31094

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31094

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs) extend Large Language Models to the vision domain. Initial LMMs used holistic images and text prompts to generate ungrounded textual responses. Very recently, region-level LMMs have been used to generate visually grounded responses. However, they are limited to only referring to a single object category at a time, require users to specify the regions in inputs, or cannot offer dense pixel-wise object grounding. In this work, we present Grounding LMM (GLaMM), the first model that can generate natural language responses seamlessly intertwined with corresponding object segmentation masks. GLaMM not only grounds objects appearing in the conversations but is flexible enough to accept both textual and optional visual prompts (region of interest) as input. This empowers users to interact with the model at various levels of granularity, both in textual and visual domains. Due to the lack of standard benchmarks for the novel setting of visually Grounded Conversation Generation (GCG), we introduce a comprehensive evaluation protocol with our curated grounded conversations. Our proposed GCG task requires densely grounded concepts in natural scenes at a large-scale. To this end, we propose a densely annotated Grounding-anything Dataset (GranD) using our proposed automated annotation pipeline that encompasses 7.5M unique concepts grounded in a total of 810M regions available with segmentation masks. Besides GCG, GLaMM also performs effectively on several downstream tasks, e.g., referring expression segmentation, image and region-level captioning and vision-language conversations. Our codes, data and models will be publicly released.

</details>

---

## 167. ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models

- [ ] ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31098

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31098

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although soft prompt tuning is effective in efficiently adapting Vision-Language (V\&L) models for downstream tasks, it shows limitations in dealing with distribution shifts. We address this issue with Attribute-Guided Prompt Tuning (ArGue), making three key contributions. 1) In contrast to the conventional approach of directly appending soft prompts preceding class names, we align the model with primitive visual attributes generated by Large Language Models (LLMs). We posit that a model's ability to express high confidence in these attributes signifies its capacity to discern the correct class rationales. 2) We introduce attribute sampling to eliminate disadvantageous attributes, thus only semantically meaningful attributes are preserved. 3) We propose negative prompting, explicitly enumerating class-agnostic attributes to activate spurious correlations and encourage the model to generate highly orthogonal probability distributions in relation to these negative features. In experiments, our method significantly outperforms current state-of-the-art prompt tuning methods on both novel class prediction and out-of-distribution generalization tasks.

</details>

---

## 168. SnAG: Scalable and Accurate Video Grounding

- [ ] SnAG: Scalable and Accurate Video Grounding | https://cvpr.thecvf.com/virtual/2024/poster/31102

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31102

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Temporal grounding of text descriptions in video is an important task amid vision-language learning, and remains a challenging problem in video understanding. Existing methods focus on grounding a few text queries within minute-long videos, yet fail to scale up to hour-long videos with hundreds of queries. In this paper, we present a systematic study for the design of scalable video grounding models. We compare design choices for cross-modal fusion, analyze their computational cost, and point out key insight and a new training scheme that enables scalable video grounding. We further present a simple model following our key findings. Our model attains superior accuracy and efficiency on recent benchmarks for long-form video grounding, while remaining highly competitive on previous benchmarks comprising short videos.

</details>

---

## 169. TRINS: Towards Multimodal Language Models that Can Read

- [ ] TRINS: Towards Multimodal Language Models that Can Read | https://cvpr.thecvf.com/virtual/2024/poster/31114

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31114

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal language models have shown remarkable proficiency in understanding and editing images. However, a majority of these visually-tuned models struggle to comprehend the textual content embedded in images, primarily due to the limitation of training data. In this work, we introduce TRINS: a Text-Rich image\footnote{In this work, we use the phrase ``text-rich images'' to describe images with rich textual information, such as posters and book covers.} INStruction dataset, with the objective of enhancing the reading ability of the multimodal large language model. TRINS is built using hybrid data annotation strategies including machine-assisted and human-assisted annotation process. It contains 39,153 text-rich images, captions and 102,437 questions.  Specifically, we show that the number of words per annotation in TRINS is significantly longer than that of related datasets, providing new challenges. Furthermore, we introduce a simple and effective architecture, called Language-vision Reading Assistant (LaRA), that is good at understanding textual contents within images. LaRA outperforms existing state-of-the-art multimodal large language models on the TRINS dataset as well as other classical benchmarks. Lastly, we conducted a comprehensive evaluation with TRINS on various text-rich image understanding and generation tasks, demonstrating its effectiveness.

</details>

---

## 170. RegionGPT: Towards Region Understanding Vision Language Model

- [ ] RegionGPT: Towards Region Understanding Vision Language Model | https://cvpr.thecvf.com/virtual/2024/poster/31126

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31126

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) have experienced rapid advancements through the integration of large language models (LLMs) with image-text pairs, yet they struggle with detailed regional visual understanding due to limited spatial awareness of the vision encoder, and the use of coarse-grained training data that lacks detailed, region-specific captions.To address this, we introduce RegionGPT (short as RGPT), a novel framework designed for complex region-level captioning and understanding. RGPT enhances the spatial awareness of regional representation with simple yet effective modifications to existing visual encoders in VLMs. We further improve performance on tasks requiring a specific output scope by integrating task-guided instruction prompts during both training and inference phases, while maintaining the model's versatility for general-purpose tasks. Additionally, we develop an automated region caption data generation pipeline, enriching the training set with detailed region-level captions. We demonstrate that a universal RGPT model can be effectively applied and significantly enhancing performance across a range of region-level tasks, including but not limited to complex region descriptions, reasoning, object classification, and referring expressions comprehension.

</details>

---

## 171. Test-Time Zero-Shot Temporal Action Localization

- [ ] Test-Time Zero-Shot Temporal Action Localization | https://cvpr.thecvf.com/virtual/2024/poster/31142

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31142

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-Shot Temporal Action Localization (ZS-TAL) seeks to identify and locate actions in untrimmed videos unseen during training. Existing ZS-TAL methods involve fine-tuning a model on a large amount of annotated training data. While effective, training-based ZS-TAL approaches assume the availability of labeled data for supervised learning, which can be impractical in some applications. Furthermore, the training process naturally induces a domain bias into the learned model, which may adversely affect the model's generalization ability to arbitrary videos. These considerations prompt us to approach the ZS-TAL problem from a radically novel perspective, relaxing the requirement for training data. To this aim, we introduce a novel method that performs $\textbf{T}$est-$\textbf{T}$ime adaptation for $\textbf{T}$emporal $\textbf{A}$ction $\textbf{L}$ocalization ($\textbf{T3AL}$). In a nutshell, $T3AL$ adapts a pre-trained Vision and Language Model (VLM) at inference time on a sample basis. $T3AL$ operates in three steps. First, a video-level pseudo-label of the action category is computed by aggregating information from the entire video. Then, action localization is performed adopting a novel procedure inspired by self-supervised learning. Finally, frame-level textual descriptions extracted with a state-of-the-art captioning model are employed for refining the action region proposals.  We validate the effectiveness of $T3AL$ by conducting experiments on the THUMOS14 and the ActivityNet-v1.3 datasets. Our results demonstrate that $T3AL$ significantly outperforms zero-shot baselines based on state-of-the-art VLMs, confirming the benefit of a test-time adaptation approach.

</details>

---

## 172. GenZI: Zero-Shot 3D Human-Scene Interaction Generation

- [ ] GenZI: Zero-Shot 3D Human-Scene Interaction Generation | https://cvpr.thecvf.com/virtual/2024/poster/31153

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31153

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Can we synthesize 3D humans interacting with scenes without learning from any 3D human-scene interaction data? We propose GenZI, the first zero-shot approach to generating 3D human-scene interactions. Key to GenZI is our distillation of interaction priors from large vision-language models (VLMs), which have learned a rich semantic space of 2D human-scene compositions. Given a natural language description and a coarse point location of the desired interaction in a 3D scene, we first leverage VLMs to imagine plausible 2D human interactions inpainted into multiple rendered views of the scene. We then formulate a robust iterative optimization to synthesize the pose and shape of a 3D human model in the scene, guided by consistency with the 2D interaction hypotheses. In contrast to existing learning-based approaches, GenZI circumvents the conventional need for captured 3D interaction data, and allows for flexible control of the 3D interaction synthesis with easy-to-use text prompts. Extensive experiments show that our zero-shot approach has high flexibility and generality, making it applicable to diverse scene types, including both indoor and outdoor environments.

</details>

---

## 173. Exploring the Transferability of Visual Prompting for Multimodal Large Language Models

- [ ] Exploring the Transferability of Visual Prompting for Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31172

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31172

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although Multimodal Large Language Models (MLLMs) have demonstrated promising versatile capabilities, their performance is still inferior to specialized models on downstream tasks, which makes adaptation necessary to enhance their utility. However, fine-tuning methods require independent training for every model, leading to huge computation and memory overheads. In this paper, we propose a novel setting where we aim to improve the performance of diverse MLLMs with a group of shared parameters optimized for a downstream task. To achieve this, we propose Transferable Visual Prompting (TVP), a simple and effective approach to generate visual prompts that can transfer to different models and improve their performance on downstream tasks after trained on only one model. We introduce two strategies to address the issue of cross-model feature corruption of existing visual prompting methods and enhance the transferability of the learned prompts, including 1) Feature Consistency Alignment: which imposes constraints to the prompted feature changes to maintain task-agnostic knowledge; 2) Task Semantics Enrichment: which encourages the prompted images to contain richer task-specific semantics with language guidance. We validate the effectiveness of TVP through extensive experiments with 6 modern MLLMs on a wide variety of tasks ranging from object recognition and counting to multimodal reasoning and hallucination correction.

</details>

---

## 174. Anchor-based Robust Finetuning of Vision-Language Models

- [ ] Anchor-based Robust Finetuning of Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31194

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31194

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We aim at finetuning a vision-language model without hurting its out-of-distribution (OOD) generalization. We address two types of OOD generalization, i.e., i) domain shift such as natural to sketch images, and ii) zero-shot capability to recognize the category that was not contained in the finetune data. Arguably, the diminished OOD generalization after finetuning stems from the excessively simplified finetuning target, which only provides the class information, such as ``a photo of a [CLASS]''. This is distinct from the process in that CLIP was pretrained, where there is abundant text supervision with rich semantic information. Therefore, we propose to compensate for the finetune process using auxiliary supervision with rich semantic information, which acts as anchors to preserve the OOD generalization. Specifically, two types of anchors are elaborated in our methods, including i) text-compensated anchor which uses the images from the finetune set but enriches the text supervision from a pretrained captioner, ii) image-text-pair anchor which is retrieved from the dataset similar to pretraining data of CLIP according to the downstream task, associating with the original CLIP text with rich semantics. Those anchors are utilized as auxiliary semantic information to maintain the original feature space of CLIP, thereby preserving the OOD generalization capabilities. Comprehensive experiments demonstrate that our method achieves in-distribution performance akin to conventional finetuning while attaining new state-of-the-art results on domain shift and zero-shot learning benchmarks.

</details>

---

## 175. DiaLoc: An Iterative Approach to Embodied Dialog Localization

- [ ] DiaLoc: An Iterative Approach to Embodied Dialog Localization | https://cvpr.thecvf.com/virtual/2024/poster/31198

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31198

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal learning has advanced the performance for many vision-language tasks. However, most existing works in embodied dialog research focus on navigation and leave the localization task understudied. The few existing dialog-based localization approaches assume the availability of entire dialog prior to localizaiton, which is impractical for deployed dialog-based localization. In this paper, we propose DiaLoc, a new dialog-based localization framework which aligns with a real human operator behavior. Specifically, we produce an iterative refinement of location predictions which can visualize current pose believes after each dialog turn. DiaLoc effectively utilizes the multimodal datafor multi-shot localization, where a fusion encoder fuses vision and dialog information iteratively. We achieve state-of-the-art results on embodied dialog-based localization task, in single-shot (+7.08% in Acc5@valUnseen) and multi-shot settings (+10.85% in Acc5@valUnseen). DiaLoc narrows the gap between simulation and real-world applications, opening doors for future research on collaborative localization and navigation.

</details>

---

## 176. Question Aware Vision Transformer for Multimodal Reasoning

- [ ] Question Aware Vision Transformer for Multimodal Reasoning | https://cvpr.thecvf.com/virtual/2024/poster/31218

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31218

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language (VL) models have gained significant research focus, enabling remarkable advances in multimodal reasoning. These architectures typically comprise a vision encoder, a Large Language Model (LLM), and a projection module that aligns visual features with the LLM's representation space. Despite their success, a critical limitation persists: the vision encoding process remains decoupled from user queries, often in the form of image-related questions. Consequently, the resulting visual features may not be optimally attuned to the query-specific elements of the image.To address this, we introduce QA-ViT, a Question Aware Vision Transformer approach for multimodal reasoning, which embeds question awareness directly within the vision encoder.This integration results in dynamic visual features focusing on relevant image aspects to the posed question.QA-ViT is model-agnostic and can be incorporated efficiently into any VL architecture.Extensive experiments demonstrate the effectiveness of applying our method to various multimodal architectures, leading to consistent improvement across diverse tasks and showcasing its potential for enhancing visual and scene-text understanding.

</details>

---

## 177. OneLLM: One Framework to Align All Modalities with Language

- [ ] OneLLM: One Framework to Align All Modalities with Language | https://cvpr.thecvf.com/virtual/2024/poster/31234

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31234

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have gained significant attention due to their strong multimodal understanding capability. However, existing works rely heavily on modality-specific encoders, which usually differ in architecture and are limited to common modalities. In this paper, we present OneLLM , an MLLM that aligns eight modalities to language using a unified framework. We achieve this through a unified multimodal encoder and a progressive multimodal alignment pipeline. In detail, we first train an image projection module to connect a vision encoder with LLM. Then, we build a universal projection module (UPM) by mixing multiple image projection modules and dynamic routing. Finally, we progressively align more modalities to LLM with the UPM. To fully leverage the potential of OneLLM in following instructions, we also curated a comprehensive multimodal instruction dataset, including 2M items from image, audio, video, point cloud, depth/normal map, IMU and fMRI brain activity. OneLLM is evaluated on 25 diverse benchmarks, encompassing tasks such as multimodal captioning, question answering and reasoning, where it delivers excellent performance. Code, data, model and online demo are available at https://github.com/csuhan/OneLLM

</details>

---

## 178. Seeing the Unseen: Visual Common Sense for Semantic Placement

- [ ] Seeing the Unseen: Visual Common Sense for Semantic Placement | https://cvpr.thecvf.com/virtual/2024/poster/31244

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31244

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Computer vision tasks typically involve describing what is visible in an image (e.g. classification,  detection, segmentation, and captioning). We study a visual common sense task that requires understanding 'what is not visible'. Specifically, given an image (e.g. of a living room) and a name of an object ("cushion"), a vision system is asked to predict semantically-meaningful regions (masks or bounding boxes) in the image where that object could be placed or is likely be placed by humans (e.g. on the sofa). We call this task: Semantic Placement (SP) and believe that such common-sense visual understanding is critical for assitive robots (tidying a house), AR devices (automatically rendering an object in the user's space), and visually-grounded chatbots with common sense. Studying the invisible is hard. Datasets for image description are typically constructed by curating relevant images (e.g. via image search with object names) and asking humans to annotate the contents of the image; neither of those two steps are straightforward for objects not present in the image. We overcome this challenge by operating in the opposite direction: we start with an image of an object in context (which is easy to find online) and remove that object from the image via inpainting. This automated pipeline converts unstructured web data into a paired with/without object dataset. With this proposed data generation pipeline, we collect a novel dataset, containing ~1.3M images across 9 object categories. We then train a SP prediction model, called CLIP-UNet, on our dataset. The CLIP-UNet outperforms existing VLMs and baselines that combine semantic priors with object detectors, generalizes well to real-world and simulated images, exhibits semantics-aware reasoning for object placement, and enables downstream applications like tidying robots in indoor environments.

</details>

---

## 179. Non-autoregressive Sequence-to-Sequence Vision-Language Models

- [ ] Non-autoregressive Sequence-to-Sequence Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31246

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31246

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Sequence-to-sequence vision-language models are showing promise, but their applicability is limited by their inference latency due to their autoregressive way of generating predictions.We propose a sequence-to-sequence vision-language model with a flexible hypothesis space, manifest in the training set and encoded in a layer of learnable query tokens. The architecture is trained with a novel loss, inspired by the language domain, that marginalizes over multiple inference paths in the decoder. This enables us the flexibility to adapt the hypothesis space to the task, rather than restricting to the embedding of a single token as in an autoregressive model. The resulting model, NARVL, achieves performance on-par with its autoregressive counterpart, but is faster at inference time since the decoder has to be executed once to jointly produce all output tokens, rather than sequentially to produce them one at a time. We test our model on four vision-language tasks, and perform ablation studies to single out the contribution of each component.

</details>

---

## 180. CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor

- [ ] CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor | https://cvpr.thecvf.com/virtual/2024/poster/31270

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31270

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing open-vocabulary image segmentation methods require a fine-tuning step on mask labels and/or image-text datasets. Mask labels are labor-intensive, which limits the number of categories in segmentation datasets. Consequently, the vocabulary capacity of pre-trained VLMs is severely reduced after fine-tuning. However, without fine-tuning, VLMs trained under weak image-text supervision tend to make suboptimal mask predictions.To alleviate these issues, we introduce a novel recurrent framework that progressively filters out irrelevant texts and enhances mask quality without training efforts. The recurrent unit is a two-stage segmenter built upon a frozen VLM. Thus, our model retains the VLM's broad vocabulary space and equips it with segmentation ability.Experiments show that our method outperforms not only the training-free counterparts, but also those fine-tuned with millions of data samples, and sets the new state-of-the-art records for both zero-shot semantic and referring segmentation. Concretely, we improve the current record by 28.8, 16.0, and 6.9 mIoU on Pascal VOC, COCO Object, and Pascal Context.

</details>

---

## 181. SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection

- [ ] SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection | https://cvpr.thecvf.com/virtual/2024/poster/31274

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31274

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Misinformation is a prevalent societal issue due to its potential high risks. Out-Of-Context (OOC) misinformation, where authentic images are repurposed with false text, is one of the easiest and most effective ways to mislead audiences. Current methods focus on assessing image-text consistency but lack convincing explanations for their judgments, which are essential for debunking misinformation. While Multimodal Large Language Models (MLLMs)  have rich knowledge and innate capability for visual reasoning and explanation generation, they still lack sophistication in understanding and discovering the subtle cross-modal differences. In this paper, we introduce SNIFFER, a novel multimodal large language model specifically engineered for OOC misinformation detection and explanation. SNIFFER employs two-stage instruction tuning on InstructBLIP. The first stage refines the model's concept alignment of generic objects with news-domain entities and the second stage leverages OOC-specific instruction data generated by language-only GPT-4 to fine-tune the model's discriminatory powers. Enhanced by external tools and retrieval, SNIFFER not only detects inconsistencies between text and image but also utilizes external knowledge for contextual verification. Our experiments show that SNIFFER surpasses the original MLLM by over 40\% and outperforms state-of-the-art methods in detection accuracy. SNIFFER also provides accurate and persuasive explanations as validated by quantitative and human evaluations.

</details>

---

## 182. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities

- [ ] SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities | https://cvpr.thecvf.com/virtual/2024/poster/31284

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31284

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding and reasoning about spatial relationships is crucial for Visual Question Answering (VQA) and robotics. Vision Language Models (VLMs) have shown impressive performance in some VQA benchmarks but struggle with 3D spatial reasoning, such as recognizing distances or size differences between physical objects. This limitation may stem from a lack of 3D spatial knowledge in their training data. To address this, we propose training VLMs with extensive spatial reasoning data from the internet. Our approach includes developing an automatic 3D spatial VQA data generation framework, capable of creating 2 billion VQA examples from 10 million real-world images. We explore various factors in the training process, such as data quality, training pipeline, and VLM architecture. Our work introduces the first Internet-scale 3D spatial reasoning dataset in metric space. By co-training a VLM with this dataset, we significantly improve its performance in both qualitative and quantitative spatial VQA. Additionally, this enhanced VLM enables new applications in chain-of-thought spatial reasoning and robotics, particularly in quantitative estimation.

</details>

---

## 183. CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation

- [ ] CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/31291

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31291

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation presents the challenge of labeling each pixels within an image based on wide range of text descriptions. In this work, we introduce a novel cost-based approach to adapt vision-language foundation models, notably CLIP, for the intricate task of semantic segmentation.  Through aggregating the cosine similarity score, i.e. the cost volume between image and text embeddings, our method potently adapts CLIP for segmenting seen and unseen classes by fine-tuning its encoders, addressing the challenges faced by existing methods in handling unseen classes. Building upon this, we explore methods to effectively aggregate the cost volume considering its multi-modal nature of being established between image and text embeddings. Furthermore, we examine various methods for efficiently fine-tuning CLIP. Our framework, dubbed CAT-Seg, shows state-of-the-art performance on standard benchmarks with significant margins, and further exerts strengths in more challenging scenarios from various domains.

</details>

---

## 184. GPT4Point: A Unified Framework for Point-Language Understanding and Generation

- [ ] GPT4Point: A Unified Framework for Point-Language Understanding and Generation | https://cvpr.thecvf.com/virtual/2024/poster/31298

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31298

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have excelled in 2D image-text comprehension and image generation, but their understanding of the 3D world is notably deficient, limiting progress in 3D language understanding and generation. To solve this problem, we introduce GPT4Point, an innovative groundbreaking point-language multimodal model designed specifically for unified 3D object understanding and generation within the MLLM framework. GPT4Point as a powerful 3D MLLM seamlessly can execute a variety of point-text reference tasks such as point-cloud captioning and Q&A. Additionally, GPT4Point is equipped with advanced capabilities for controllable 3D generation, it can get high-quality results through a low-quality point-text feature maintaining the geometric shapes and colors. To support the expansive needs of 3D object-text pairs, we develop Pyramid-XL, a point-language dataset annotation engine. It constructs a large-scale database over 1M objects of varied text granularity levels from the Objaverse-XL dataset, essential for training GPT4Point. A comprehensive benchmark has been proposed to evaluate 3D point-language understanding capabilities. In extensive evaluations, GPT4Point has demonstrated superior performance in understanding and generation.

</details>

---

## 185. Synthesize Diagnose and Optimize: Towards Fine-Grained Vision-Language Understanding

- [ ] Synthesize Diagnose and Optimize: Towards Fine-Grained Vision-Language Understanding | https://cvpr.thecvf.com/virtual/2024/poster/31306

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31306

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLM) have demonstrated remarkable performance across various downstream tasks. However, understanding fine-grained visual-linguistic concepts, such as attributes and inter-object relationships, remains a significant challenge. While several benchmarks aim to evaluate VLMs in finer granularity, their primary focus remains on the linguistic aspect, neglecting the visual dimension. Here, we highlight the importance of evaluating VLMs from both a textual and visual perspective. We introduce a progressive pipeline to synthesize images that vary in a specific attribute while ensuring consistency in all other aspects. Utilizing this data engine, we carefully design a benchmark, SPEC, to diagnose the comprehension of object size, position, existence, and count. Subsequently, we conduct a thorough evaluation of four leading VLMs on SPEC. Surprisingly, their performance is close to random guesses, revealing significant limitations. With this in mind, we propose a simple yet effective approach to optimize VLMs in fine-grained understanding, achieving significant improvements on SPEC without compromising the zero-shot performance. Results on two additional fine-grained benchmarks also show consistent improvements,  further validating the transferability of our approach. Code and data are available at https://github.com/wjpoom/SPEC.

</details>

---

## 186. JoAPR: Cleaning the Lens of Prompt Learning for Vision-Language Models

- [ ] JoAPR: Cleaning the Lens of Prompt Learning for Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31320

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31320

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Leveraging few-shot datasets in prompt learning for Vision-Language Models eliminates the need for manual prompt engineering while highlighting the necessity of accurate annotations for the labels. However, high-level or complex label noise challenges prompt learning for Vision-Language Models. Aiming at this issue, we propose a new framework for improving its robustness. Specifically, we introduce the Joint Adaptive Partitioning for Label Refurbishment (JoAPR), a structured framework encompassing two key steps. 1) Data Partitioning, where we differentiate between clean and noisy data using joint adaptive thresholds. 2) Label Refurbishment, where we correct the labels based on the partition outcomes before retraining the network. Our comprehensive experiments confirm that JoAPR substantially enhances the robustness of prompt learning for Vision-Language Models against label noise, offering a promising direction for future research.

</details>

---

## 187. AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning

- [ ] AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning | https://cvpr.thecvf.com/virtual/2024/poster/31326

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31326

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, pre-trained vision-language models (e.g., CLIP) have shown great potential in few-shot learning and attracted a lot of research interest. Although many efforts have been made to improve few-shot generalization ability of CLIP,  key factors on the effectiveness of existing methods have not been well studied, limiting further exploration of CLIP's potential in few-shot learning. In this paper, we first introduce a unified formulation to analyze CLIP-based few-shot learning methods from a perspective of logit bias, which encourages us to learn an effective logit bias for further improving performance of CLIP-based few-shot learning methods. To this end, we disassemble three key components involved in computation of logit bias (i.e., logit features, logit predictor, and logit fusion) and empirically analyze the effect on performance of few-shot classification. According to the analysis on the key components, this paper proposes a novel AMU-Tuning method to learn effective logit bias for CLIP-based few-shot classification. Specifically, our AMU-Tuning predicts logit bias by exploiting the appropriate $\underline{\textbf{A}}$uxiliary features, which are fed into an efficient feature-initialized linear classifier with $\underline{\textbf{M}}$ulti-branch training. Finally, an $\underline{\textbf{U}}$ncertainty-based fusion is developed to incorporate logit bias into CLIP for few-shot classification. The experiments are conducted on several widely used benchmarks, and the results show our proposed AMU-Tuning clearly outperforms its counterparts while achieving state-of-the-art performance without bells and whistles.

</details>

---

## 188. Sieve: Multimodal Dataset Pruning using Image Captioning Models

- [ ] Sieve: Multimodal Dataset Pruning using Image Captioning Models | https://cvpr.thecvf.com/virtual/2024/poster/31327

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31327

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) are pretrained on large, diverse, and noisy web-crawled datasets. This underscores the critical need for dataset pruning, as the quality of these datasets is strongly correlated with the performance of VLMs on downstream tasks. Using CLIPScore from a pretrained model to only train models using highly-aligned samples is one of the most successful methods for pruning. We argue that this approach suffers from multiple limitations including: false positives and negatives due to CLIP's pretraining on noisy labels. We propose a pruning signal, Sieve, that employs synthetic captions generated by image-captioning models pretrained on small, diverse, and well-aligned image-text pairs to evaluate the alignment of noisy image-text pairs. To bridge the gap between the limited diversity of generated captions and the high diversity of alternative text (alt-text), we estimate the semantic textual similarity in the embedding space of a language model pretrained on unlabeled text corpus. Using DataComp, a multimodal dataset filtering benchmark, when evaluating on 38 downstream tasks, our pruning approach, surpasses CLIPScore by 2.6\% and 1.7\% on medium and large scale respectively. In addition, on retrieval tasks, Sieve leads to a significant improvement of 2.7\% and 4.5\% on medium and large scale respectively.

</details>

---

## 189. Discover and Mitigate Multiple Biased Subgroups in Image Classifiers

- [ ] Discover and Mitigate Multiple Biased Subgroups in Image Classifiers | https://cvpr.thecvf.com/virtual/2024/poster/31352

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31352

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Machine learning models can perform well on in-distribution data but often fail on biased subgroups that are underrepresented in the training data, hindering the robustness of models for reliable applications. Such subgroups are typically unknown due to the absence of subgroup labels. Discovering biased subgroups is the key to understanding models' failure modes and further improving models' robustness. Most previous works of subgroup discovery make an implicit assumption that models only underperform on a single biased subgroup, which does not hold on in-the-wild data where multiple biased subgroups exist.    In this work, we propose Decomposition, Interpretation, and Mitigation (DIM), a novel method to address a more challenging but also more practical problem of discovering multiple biased subgroups in image classifiers. Our approach decomposes the image features into multiple components that represent multiple subgroups. This decomposition is achieved via a bilinear dimension reduction method, Partial Least Square (PLS), guided by useful supervision from the image classifier. We further interpret the semantic meaning of each subgroup component by generating natural language descriptions using vision-language foundation models. Finally, DIM mitigates multiple biased subgroups simultaneously via two strategies, including the data- and model-centric strategies. Extensive experiments on CIFAR-100 and Breeds datasets demonstrate the effectiveness of DIM in discovering and mitigating multiple biased subgroups. Furthermore, DIM uncovers the failure modes of the classifier on Hard ImageNet, showcasing its broader applicability to understanding model bias in image classifiers.

</details>

---

## 190. Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification

- [ ] Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification | https://cvpr.thecvf.com/virtual/2024/poster/31364

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31364

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) such as CLIP are trained on large amounts of image-text pairs, resulting in remarkable generalization across several data distributions. However, in several cases, their expensive training and data collection/curation costs do not justify the end application. This motivates a vendor-client paradigm, where a vendor trains a large-scale VLM and grants only input-output access to clients on a pay-per-query basis in a black-box setting. The client aims to minimize inference cost by distilling the VLM to a student model using the limited available task-specific data, and further deploying this student model in the downstream application. While naive distillation largely improves the In-Domain (ID) accuracy of the student, it fails to transfer the superior out-of-distribution (OOD) generalization of the VLM teacher using the limited available labeled images. To mitigate this, we propose Vision-Language to Vision - Align, Distill, Predict (VL2V-ADiP), which first aligns the vision and language modalities of the teacher model with the vision modality of a pre-trained student model, and further distills the aligned VLM representations to the student. This maximally retains the pre-trained features of the student, while also incorporating the rich representations of the VLM image encoder and the superior generalization of the text embeddings. The proposed approach achieves state-of-the-art results on the standard Domain Generalization benchmarks in a black-box teacher setting as well as a white-box setting where the weights of the VLM are accessible.

</details>

---

## 191. Detours for Navigating Instructional Videos

- [ ] Detours for Navigating Instructional Videos | https://cvpr.thecvf.com/virtual/2024/poster/31370

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31370

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce the video detours problem for navigating instructional videos.  Given a source video and a natural language query asking to alter the how-to video's current path of execution in a certain way, the goal is to find a related "detour video" that satisfies the requested alteration.  To address this challenge, we propose VidDetours, a novel video-language approach that learns to retrieve the targeted temporal segments from a large repository of how-to's using video-and-text conditioned queries.  Furthermore, we devise a language-based pipeline that exploits how-to video narration text to create weakly supervised training data. We demonstrate our idea applied to the domain of how-to cooking videos, where a user can detour from their current recipe to find steps with alternate ingredients, tools, and techniques.  Validating on a ground truth annotated dataset of 16K samples, we show our model's significant improvements over best available methods for video retrieval and question answering, with recall rates exceeding the state of the art by 35%.

</details>

---

## 192. Domain-Agnostic Mutual Prompting for Unsupervised Domain Adaptation

- [ ] Domain-Agnostic Mutual Prompting for Unsupervised Domain Adaptation | https://cvpr.thecvf.com/virtual/2024/poster/31368

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31368

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Conventional Unsupervised Domain Adaptation (UDA) strives to minimize distribution discrepancy between domains, which neglects to harness rich semantics from data and struggles to handle complex domain shifts. A promising technique is to leverage the knowledge of large-scale pre-trained vision-language models for more guided adaptation. Despite some endeavors, current methods often learn textual prompts to embed domain semantics for source and target domains separately and perform classification within each domain, limiting cross-domain knowledge transfer. Moreover, prompting only the language branch lacks flexibility to adapt both modalities dynamically. To bridge this gap, we propose Domain-Agnostic Mutual Prompting (DAMP) to exploit domain-invariant semantics by mutually aligning visual and textual embeddings. Specifically, the image contextual information is utilized to prompt the language model in a domain-agnostic and instance-conditioned way. Meanwhile, visual prompts are imposed based on the domain-agnostic textual prompt to elicit domain-invariant visual embeddings. These two branches of prompts are learned mutually with a cross-attention module and regularized with a semantic-consistency loss and an instance-discrimination contrastive loss. Experiments on three UDA benchmarks demonstrate the superiority of DAMP over state-of-the-art approaches.

</details>

---

## 193. Iterated Learning Improves Compositionality in Large Vision-Language Models

- [ ] Iterated Learning Improves Compositionality in Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31371

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31371

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A fundamental characteristic common to both human vision and natural language is their compositional nature. Yet, despite the performance gains contributed by large vision and language pretraining, recent investigations find that most - if not all - our state-of-the-art vision-language models struggle at compositionality. They are unable to distinguish between images of "a girl in white facing a man in black" and "a girl in black facing a man in white". Moreover, prior work suggests that compositionality doesn't arise with scale: larger model sizes or training data don't help. This paper develops a new iterated training algorithm that incentivizes compositionality. We draw on decades of cognitive science research that identifies cultural transmission - the need to teach a new generation - as a necessary inductive prior that incentivizes humans to develop compositional languages. Specifically, we reframe vision-language contrastive learning as the Lewis Signaling Game between a vision agent and language agent, and operationalize cultural transmission by iteratively resetting one of the agent's weights during training. After every iteration, this training paradigm induces representations that become "easier to learn", a property of compositional languages: e.g. our model trained in CC3M and CC12M improves standard CLIP by 4.7%, 4.0% respectfully in the SugarCrepe benchmark.

</details>

---

## 194. PromptKD: Unsupervised Prompt Distillation for Vision-Language Models

- [ ] PromptKD: Unsupervised Prompt Distillation for Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31372

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31372

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has emerged as a valuable technique in enhancing vision-language models (VLMs) such as CLIP for downstream tasks in specific domains. Existing work mainly focuses on designing various learning forms of prompts, neglecting the potential of prompts as effective distillers for learning from larger teacher models. In this paper, we introduce an unsupervised domain prompt distillation framework, which aims to transfer the knowledge of a larger teacher model to a lightweight target model through prompt-based imitation using unlabeled domain images. Specifically, our framework consists of two distinct stages. In the initial stage, we pre-train a large CLIP teacher model using domain few-shot labels. After pre-training, we leverage the unique decoupled-modality characteristics of CLIP by pre-computing and storing the text features as class vectors only once through the teacher text encoder. In the subsequent stage, the stored class vectors are shared across teacher and student image encoders for calculating the predicted logits. We align the logits of both the teacher and student models via KL divergence, encouraging the student image encoder to generate similar probability distributions to the teacher through the learnable prompts. The proposed prompt distillation process eliminates the reliance on labeled data, enabling the algorithm to leverage a vast amount of unlabeled images within the domain.Finally, the well-trained student image encoders and pre-stored text features (class vectors) are utilized for inference. To our best knowledge, we are the first to perform domain-specific prompt-based knowledge distillation for CLIP using unlabeled data. Extensive experiments on 11 datasets demonstrate the effectiveness of our method.

</details>

---

## 195. Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters

- [ ] Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters | https://cvpr.thecvf.com/virtual/2024/poster/31379

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31379

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Continual learning can empower vision-language models to continuously acquire new knowledge, without the need for access to the entire historical dataset. However, mitigating the performance degradation in large-scale models is non-trivial due to (i) parameter shifts throughout lifelong learning and (ii) significant computational burdens associated with full-model tuning. In this work, we present a parameter-efficient continual learning framework to alleviate long-term forgetting in incremental learning with vision-language models. Our approach involves the dynamic expansion of a pre-trained CLIP model, through the integration of Mixture-of-Experts (MoE) adapters in response to new tasks. To preserve the zero-shot recognition capability of vision-language models, we further introduce a Distribution Discriminative Auto-Selector (DDAS) that automatically routes in-distribution and out-of-distribution inputs to the MoE Adapter and the original CLIP, respectively. Through extensive experiments across various settings, our proposed method consistently outperforms previous state-of-the-art approaches while concurrently reducing parameter training burdens by 60%. Our code locates at https://github.com/JiazuoYu/MoE-Adapters4CL

</details>

---

## 196. Volumetric Environment Representation for Vision-Language Navigation

- [ ] Volumetric Environment Representation for Vision-Language Navigation | https://cvpr.thecvf.com/virtual/2024/poster/31391

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31391

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language navigation (VLN) requires an agent to navigate through an 3D environment based on visual observations and natural language instructions. It is clear that the pivotal factor for successful navigation lies in the comprehensive scene understanding. Previous VLN agents employ monocular frameworks to extract 2D features of perspective views directly. Though straightforward, they struggle for capturing 3D geometry and semantics, leading to a partial and incomplete environment representation. To achieve a comprehensive 3D representation with fine-grained details, we introduce a Volumetric Environment Representation (VER), which voxelizes the physical world into structured 3D cells. For each cell, VER aggregates multi-view 2D features into such a unified 3D space via 2D-3D sampling. Through coarse-to-fine feature extraction and multi-task learning for VER, our agent predicts 3D occupancy, 3D room layout, and 3D bounding boxes jointly. Based on online collected VERs, our agent performs volume state estimation and builds episodic memory for predicting the next step. Experimental results show our environment representations from multi-task learning lead to evident performance gains on VLN. Our model achieves state-of-the-art performance across VLN benchmarks (R2R, REVERIE, and R4R).

</details>

---

## 197. Language-driven All-in-one Adverse Weather Removal

- [ ] Language-driven All-in-one Adverse Weather Removal | https://cvpr.thecvf.com/virtual/2024/poster/31397

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31397

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

All-in-one (AiO) frameworks restore various adverse weather degradations with a single set of networks jointly. To handle various weather conditions, an AiO framework is expected to adaptively learn weather-specific knowledge for different degradations and shared knowledge for common patterns. However, existing method: 1) rely on extra supervision signals, which are usually unknown in real-world applications;  2) employ fixed network structures, which restrict the diversity of weather-specific knowledge. In this paper, we propose a Language-driven Restoration framework (LDR) to alleviate the aforementioned issues. First, we leverage the power of pre-trained vision-language (PVL) models to enrich the diversity of weather-specific knowledge by reasoning about the occurrence, type, and severity of degradation, generating description-based degradation priors. Then, with the guidance of degradation prior, we sparsely select restoration experts from a candidate list dynamically based on a Mixture-of-Experts (MoE) structure. This enables us to adaptively learn the weather-specific and shared knowledge to handle various weather conditions (e.g., unknown or mixed weather). Experiments on extensive restoration scenarios show our superior performance (see Fig. 1). The source code will be made available.

</details>

---

## 198. VCoder: Versatile Vision Encoders for Multimodal Large Language Models

- [ ] VCoder: Versatile Vision Encoders for Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31412

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31412

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Humans possess the remarkable skill of Visual Perception, the ability to see and understand the seen, helping them make sense of the visual world and, in turn, reason. Multimodal Large Language Models (MLLM) have recently achieved impressive performance on vision-language tasks ranging from visual question-answering and image captioning to visual reasoning and image generation. However, when prompted to identify or count (perceive) the entities in a given image, existing MLLM systems fail. Working towards developing an accurate MLLM system for perception and reasoning, we propose using Versatile vision enCoders (VCoder) as perception eyes for Multimodal LLMs. We feed the VCoder with perception modalities such as segmentation or depth maps, improving the MLLM's perception abilities. Secondly, we leverage the images from COCO and outputs from off-the-shelf vision perception models to create our COCO Segmentation Text (COST) dataset for training and evaluating MLLMs on the object perception task. Thirdly, we introduce metrics to assess the object perception abilities in MLLMs on our COST dataset. Lastly, we provide extensive experimental evidence proving the VCoder's improved object-level perception skills over existing Multimodal LLMs, including GPT-4V. We open-source our dataset, code, and models to promote research.

</details>

---

## 199. Unveiling Parts Beyond Objects: Towards Finer-Granularity Referring Expression Segmentation

- [ ] Unveiling Parts Beyond Objects: Towards Finer-Granularity Referring Expression Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/31434

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31434

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Referring expression segmentation (RES) aims at segmenting the foreground masks of the entities that match the descriptive natural language expression. Previous datasets and methods for classic RES task heavily rely on the prior assumption that one expression must refer to object-level targets. In this paper, we take a step further to finer-grained part-level RES task. To promote the object-level RES task towards finer-grained vision-language understanding, we put forward a new multi-granularity referring expression segmentation (MRES) task and construct an evaluation benchmark called RefCOCOm by manual annotations. By employing our automatic model-assisted data engine, we build the largest visual grounding dataset namely MRES-32M, which comprises over 32.2M high-quality masks and captions on the provided 1M images. Besides, a simple yet strong model named UniRES is designed to accomplish the unified object-level and part-level grounding task. Extensive experiments on our RefCOCOm for MRES and three datasets (i.e., RefCOCO(+/g)) for classic RES task demonstrate the superiority of our method over previous state-of-the-art methods. To foster future research into fine-grained visual grounding, our benchmark RefCOCOm, the MRES-32M dataset and model UniRES will be publicly available.

</details>

---

## 200. UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All

- [ ] UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All | https://cvpr.thecvf.com/virtual/2024/poster/31443

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31443

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present UniBind, a flexible and efficient approach that learns a unified representation space for seven diverse modalities-- images, text, audio, point cloud, thermal, video, and event data. Existing works, eg., ImageBind, treat the image as the central modality and build an image-centered representation space; however, the space may be sub-optimal as it leads to an unbalanced representation space among all modalities. Moreover, the category names are directly used to extract text embeddings for the downstream tasks, making it hardly possible to represent the semantics of multi-modal data. The 'out-of-the-box' insight of our UniBind is to make the alignment center modality-agnostic and further learn a unified and balanced representation space, empowered by the large language models (LLMs). UniBind is superior in its flexible application to all CLIP-style models and delivers remarkable performance boosts. To make this possible, we 1) construct a knowledge base of text embeddings with the help of LLMs and multi-modal LLMs; 2) adaptively build LLM-augmented class-wise embedding center on top of the knowledge base and encoded visual embeddings; 3) align all the embeddings to the LLM-augmented embedding center via contrastive learning to achieve a unified and balanced representation space. UniBind shows strong zero-shot recognition performance gains over prior arts by an average of 6.36%. Finally, we achieve new state-of-the-art performance, eg., a 6.75% gain on ImageNet, on the multi-modal fine-tuning setting while reducing 90% of the learnable parameters.

</details>

---

## 201. VideoCon: Robust Video-Language Alignment via Contrast Captions

- [ ] VideoCon: Robust Video-Language Alignment via Contrast Captions | https://cvpr.thecvf.com/virtual/2024/poster/31448

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31448

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite being (pre)trained on a massive amount of data, state-of-the-art video-language alignment models are not robust to semantically-plausible contrastive changes in the video captions. Our work addresses this by identifying a broad spectrum of contrast misalignments, such as replacing entities, actions, and flipping event order, which alignment models should be robust against. To this end, we introduce the VideoCon, a video-language alignment dataset constructed by a large language model that generates plausible contrast video captions and explanations for differences between original and contrast video captions. Then, a generative video-language model is finetuned with VideoCon to assess video-language entailment and generate explanations. Our VideoCon-based alignment model significantly outperforms current models. It exhibits a $12$-point increase in AUC for the video-language alignment task on human-generated contrast captions. Finally, our model sets new state of the art zero-shot performance in temporally-extensive video-language tasks such as text-to-video retrieval (SSv2-Temporal) and video question answering (ATP-Hard). Moreover, our model shows superior performance on novel videos and human-crafted captions and explanations.

</details>

---

## 202. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP

- [ ] BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP | https://cvpr.thecvf.com/virtual/2024/poster/31458

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31458

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings. The code is available at https://github.com/jiawangbai/BadCLIP.

</details>

---

## 203. Cloud-Device Collaborative Learning for Multimodal Large Language Models

- [ ] Cloud-Device Collaborative Learning for Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31459

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31459

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The burgeoning field of Multimodal Large Language Models (MLLMs) has exhibited remarkable performance in diverse tasks such as captioning, commonsense reasoning, and visual scene understanding. However, the deployment of these large-scale MLLMs on client devices is hindered by their extensive model parameters, leading to a notable decline in generalization capabilities when these models are compressed for device deployment. Addressing this challenge, we introduce a Cloud-Device Collaborative Continual Adaptation framework, designed to enhance the performance of compressed, device-deployed MLLMs by leveraging the robust capabilities of cloud-based, larger-scale MLLMs.Our framework is structured into three key components: a device-to-cloud uplink for efficient data transmission, cloud-based knowledge adaptation, and an optimized cloud-to-device downlink for model deployment. In the uplink phase, we employ an Uncertainty-guided Token Sampling (UTS) strategy to effectively filter out-of-distribution tokens, thereby reducing transmission costs and improving training efficiency. On the cloud side, we propose Adapter-based Knowledge Distillation (AKD) method to transfer refined knowledge from large-scale to compressed, pocket-size MLLMs. Furthermore, we propose a Dynamic Weight update Compression (DWC) strategy for the downlink, which adaptively selects and quantizes updated weight parameters, enhancing transmission efficiency and reducing the representational disparity between cloud and device models. Extensive experiments on several multimodal benchmarks demonstrate the superiority of our proposed framework over prior Knowledge Distillation and device-cloud collaboration methods. Notably, we also validate the feasibility of our approach to real-world experiments.

</details>

---

## 204. Link-Context Learning for Multimodal LLMs

- [ ] Link-Context Learning for Multimodal LLMs | https://cvpr.thecvf.com/virtual/2024/poster/31464

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31464

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability to learn from context with novel concepts, and deliver appropriate responses are essential in human conversations. Despite current Multimodal Large Language Models (MLLMs) and Large Language Models (LLMs) being trained on mega-scale datasets, recognizing unseen images or understanding novel concepts in a training-free manner remains a challenge. In-Context Learning (ICL) explores training-free few-shot learning, where models are encouraged to "learn to learn" from limited tasks and generalize to unseen tasks. In this work, we propose link-context learning (LCL), which emphasizes "reasoning from cause and effect" to augment the learning capabilities of MLLMs. LCL goes beyond traditional ICL by explicitly strengthening the causal relationship between the support set and the query set. By providing demonstrations with causal links, LCL guides the model to discern not only the analogy but also the underlying causal associations between data points, which empowers MLLMs to recognize unseen images and understand novel concepts more effectively. To facilitate the evaluation of this novel approach, we introduce the ISEKAI dataset, comprising exclusively of unseen generated image-label pairs designed for link-context learning. Extensive experiments show that our LCL-MLLM exhibits strong link-context learning capabilities to novel concepts over vanilla MLLMs.

</details>

---

## 205. Alpha-CLIP: A CLIP Model Focusing on Wherever You Want

- [ ] Alpha-CLIP: A CLIP Model Focusing on Wherever You Want | https://cvpr.thecvf.com/virtual/2024/poster/31492

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31492

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) plays an essential role in extracting valuable content information from images across diverse tasks. It aligns textual and visual modalities to comprehend the entire image, including all the details, even those irrelevant to specific tasks. However, for a finer understanding and controlled editing of images, it becomes crucial to focus on specific regions of interest, which can be indicated as points, masks, or boxes by humans or perception models. To fulfill the requirements, we introduce Alpha-CLIP, an enhanced version of CLIP with an auxiliary alpha channel to suggest attentive regions and fine-tuned with constructed millions of RGBA region-text pairs. Alpha-CLIP not only preserves the visual recognition ability of CLIP but also enables precise control over the emphasis of image contents. It demonstrates effectiveness in various tasks, including but not limited to open-world recognition, multimodal large language models, and conditional 2D / 3D generation. It has a strong potential to serve as a versatile tool for image-related tasks. All the code, models, and training data will be publicly available.

</details>

---

## 206. CoDi-2: In-Context Interleaved and Interactive Any-to-Any Generation

- [ ] CoDi-2: In-Context Interleaved and Interactive Any-to-Any Generation | https://cvpr.thecvf.com/virtual/2024/poster/31512

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31512

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present CoDi-2, a Multimodal Large Language Model (MLLM) for learning in-context interleaved multi-modal representations. By aligning modalities with language for both encoding and generation, CoDi-2 empowers Large Language Models (LLMs) to understand modality- interleaved instructions and in-context examples and autoregressively generate grounded and coherent multimodal outputs in an any-to-any input-output modality paradigm. To train CoDi-2, we build a large-scale generation dataset encompassing in-context multimodal instructions across text, vision, and audio. CoDi-2 demonstrates a wide range of zero-shot and few-shot capabilities for tasks like editing, exemplar learning, composition, reasoning, etc. CoDi-2 surpasses previous domain-specific models on tasks such as subject-driven image generation, vision transformation, and audio editing and showcases a significant advancement for integrating diverse multimodal tasks with sequential generation.

</details>

---

## 207. Transferable and Principled Efficiency for Open-Vocabulary Segmentation

- [ ] Transferable and Principled Efficiency for Open-Vocabulary Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/31509

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31509

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent success of pre-trained foundation vision-language models makes Open-Vocabulary Segmentation (OVS) possible. Despite the promising performance, this approach introduces heavy computational overheads for two challenges: 1) large model sizes of the backbone; 2) expensive costs during the fine-tuning. These challenges hinder this OVS strategy from being widely applicable and affordable in real-world scenarios. Although traditional methods such as model compression and efficient fine-tuning can address these challenges, they often rely on heuristics. This means that their solutions cannot be easily transferred and necessitate re-training on different models, which comes at a cost. In the context of efficient OVS, we target achieving performance that is comparable to or even better than prior OVS works based on large vision-language foundation models, by utilizing smaller models that incur lower training costs. The core strategy is to make our efficiency principled and thus seamlessly transferable from one OVS framework to others without further customization. Comprehensive experiments on diverse OVS benchmarks demonstrate our superior trade-off between segmentation accuracy and computation costs over previous works.

</details>

---

## 208. Multi-modal Instruction Tuned LLMs with Fine-grained Visual Perception

- [ ] Multi-modal Instruction Tuned LLMs with Fine-grained Visual Perception | https://cvpr.thecvf.com/virtual/2024/poster/31513

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31513

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Model (MLLMs) leverages Large Language Models as a cognitive framework for diverse visual-language tasks. Recent efforts have been made to equip MLLMs with visual perceiving and grounding capabilities.However, there still remains a gap in providing fine-grained pixel-level perceptions and extending interactions beyond text-specific inputs. In this work, we propose AnyRef, a general MLLM model that can generate pixel-wise object perceptions and natural language descriptions from multi-modality references, such as texts, boxes, images, or audios. This innovation empowers users with greater flexibility to engage with the model beyond textual and regional prompts, without modality-specific designs. Through our proposed refocusing mechanism, the generated grounding output is guided to focus more on the referenced object, implicitly incorporating additional pixel-level supervision. This simple modification utilizes attention scores generated during the inference of LLM, eliminating the need for extra computations while exhibiting performance enhancements in both grounding masks and referring expressions. With only publicly available training data, our model achieves state-of-the-art results across multiple benchmarks, including diverse modality referring segmentation and region-level referring expression generation.

</details>

---

## 209. Active Prompt Learning in Vision Language Models

- [ ] Active Prompt Learning in Vision Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31523

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31523

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained Vision Language Models (VLMs) have demonstrated notable progress in various zero-shot tasks, such as classification and retrieval. Despite their performance, because improving performance on new tasks requires task-specific knowledge, their adaptation is essential. While labels are needed for the adaptation, acquiring them is typically expensive. To overcome this challenge, active learning, a method of achieving a high performance by obtaining labels for a small number of samples from experts, has been studied. Active learning primarily focuses on selecting unlabeled samples for labeling and leveraging them to train models. In this study, we pose the question, "how can the pre-trained VLMs be adapted under the active learning framework?" In response to this inquiry, we observe that (1) simply applying a conventional active learning framework to pre-trained VLMs even may degrade performance compared to random selection because of the class imbalance in labeling candidates, and (2) the knowledge of VLMs can provide hints for achieving the balance before labeling. Based on these observations, we devise a novel active learning framework for VLMs, denoted as PCB. To assess the effectiveness of our approach, we conduct experiments on seven different real-world datasets, and the results demonstrate that PCB surpasses conventional active learning and random sampling methods.

</details>

---

## 210. Discovering Syntactic Interaction Clues for Human-Object Interaction Detection

- [ ] Discovering Syntactic Interaction Clues for Human-Object Interaction Detection | https://cvpr.thecvf.com/virtual/2024/poster/31525

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31525

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, Vision-Language Model (VLM) has greatly advanced the Human-Object Interaction (HOI) detection. The existing VLM-based HOI detectors typically adopt a hand-crafted template (e.g., a photo of a person [action] a$/$an [object]) to acquire text knowledge through the VLM text encoder. However, such approaches, only encoding the action-specific text prompts in vocabulary level, may suffer from learning ambiguity without exploring the fine-grained clues from the perspective of interaction context. In this paper, we propose a novel method to discover Syntactic Interaction Clues for HOI detection (SICHOI) by using VLM. Specifically, we first investigate what are the essential elements for an interaction context, and then establish a syntactic interaction bank from three levels: spatial relationship, action-oriented posture and situational condition. Further, to align visual features with the syntactic interaction bank, we adopt a multi-view extractor to jointly aggregate visual features from instance, interaction, and image levels accordingly. In addition, we also introduce a dual cross-attention decoder to perform context propagation between text knowledge and visual features, thereby enhancing the HOI detection. Experimental results demonstrate that our proposed method achieves state-of-the-art performance on HICO-DET and V-COCO.

</details>

---

## 211. Open-Vocabulary Object 6D Pose Estimation

- [ ] Open-Vocabulary Object 6D Pose Estimation | https://cvpr.thecvf.com/virtual/2024/poster/31549

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31549

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce the new setting of open-vocabulary object 6D pose estimation, in which a textual prompt is used to specify the object of interest.In contrast to existing approaches, in our setting(i) the object of interest is specified solely through the textual prompt,(ii) no object model (e.g. CAD or video sequence) is required at inference,(iii) the object is imaged from two different viewpoints of two different scenes, and(iv) the object was not observed during the training phase.To operate in this setting, we introduce a novel approach that leverages a Vision-Language Model to segment the object of interest from two distinct scenes and to estimate its relative 6D pose.The key of our approach is a carefully devised strategy to fuse object-level information provided by the prompt with local image features, resulting in a feature space that can generalize to novel concepts.We validate our approach on a new benchmark based on two popular datasets, REAL275 and Toyota-Light, which collectively encompass 39 object instances appearing in four thousand image pairs. The results demonstrate that our approach outperforms both a well-established hand-crafted method and a recent deep learning-based baseline in estimating the relative 6D pose of objects in different scenes.

</details>

---

## 212. ViTamin: Designing Scalable Vision Models in the Vision-Language Era

- [ ] ViTamin: Designing Scalable Vision Models in the Vision-Language Era | https://cvpr.thecvf.com/virtual/2024/poster/31575

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31575

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent breakthroughs in vision-language models (VLMs) start a new page in the vision community. The VLMs provide stronger and more generalizable feature embeddings compared to those from ImageNet-pretrained models, thanks to the training on the large-scale Internet image-text pairs. However, despite the amazing achievement from the VLMs, vanilla Vision Transformers (ViTs) remain the default choice for the image encoder. Although pure transformer proves its effectiveness in the text encoding area, it remains questionable whether it is also the case for image encoding, especially considering that various types of networks are proposed on the ImageNet benchmark, which, unfortunately, are rarely studied in VLMs. Due to small data/model scale, the original conclusions of model design on ImageNet can be limited and biased. In this paper, we aim at building an evaluation protocol of vision models in the vision-language era under the contrastive language-image pretraining (CLIP) framework. We provide a comprehensive way to benchmark different vision models, covering their zero-shot performance and scalability in both model and training data sizes. To this end, we introduce ViTamin, a new vision models tailored for VLMs. ViTamin-L significantly outperforms ViT-L by 2.0% ImageNet zero-shot accuracy, when using the same publicly available DataComp-1B dataset and the same OpenCLIP training scheme. ViTamin-L presents promising results on 60 diverse benchmarks, including classification, retrieval, open-vocabulary detection and segmentation, and large multi-modal models. When further scaling up the model size, our ViTamin-XL with only 436M parameters attains 82.9% ImageNet zero-shot accuracy, surpassing 82.0% achieved by EVA-E that has ten times more parameters (4.4B).

</details>

---

## 213. V?: Guided Visual Search as a Core Mechanism in Multimodal LLMs

- [ ] V?: Guided Visual Search as a Core Mechanism in Multimodal LLMs | https://cvpr.thecvf.com/virtual/2024/poster/31596

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31596

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

When we look around and perform complex tasks, how we see and selectively process what we see is crucial. However, the lack of this visual search mechanism in current multimodal LLMs (MLLMs) hinders their ability to focus on important visual details, especially when handling high-resolution and visually crowded images. To address this, we introduce V$^*$, an LLM-guided visual search mechanism that employs the world knowledge in LLMs for efficient visual querying. When combined with an MLLM, this mechanism enhances collaborative reasoning, contextual understanding, and precise visual grounding. This integration results in a new MLLM meta-architecture, named **S**how, S**EA**rch, and Tel**L** (SEAL). We further create V$^*$Bench, a benchmark specifically designed to evaluate MLLMs in their ability to process high-resolution images and focus on visual details. Our study highlights the necessity of incorporating visual search capabilities into multimodal systems. The code is available at https://github.com/penghao-wu/vstar

</details>

---

## 214. A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions

- [ ] A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions | https://cvpr.thecvf.com/virtual/2024/poster/31604

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31604

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Curation methods for massive vision-language datasets trade off between dataset size and quality. However, even the highest quality of available curated captions are far too short to capture the rich visual detail in an image. To show the value of dense and highly-aligned image-text pairs, we collect the Densely Captioned Images (DCI) dataset, containing 8012 natural images human-annotated with mask-aligned descriptions averaging above 1000 words each. With precise and reliable captions associated with specific parts of an image, we can evaluate vision-language models' (VLMs) understanding of image content with a novel task that matches each caption with its corresponding subcrop. As current models are often limited to 77 text tokens, we also introduce a summarized version (sDCI) in which each caption length is limited. We show that modern techniques that make progress on standard benchmarks do not correspond with significant improvement on our sDCI based benchmark. Lastly, we finetune CLIP using sDCI and show significant improvements over the baseline despite a small training set. By releasing the first human annotated dense image captioning dataset, we hope to enable the development of new benchmarks or fine-tuning recipes for the next generation of VLMs to come.

</details>

---

## 215. GSVA: Generalized Segmentation via Multimodal Large Language Models

- [ ] GSVA: Generalized Segmentation via Multimodal Large Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31608

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31608

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generalized Referring Expression Segmentation (GRES) extends the scope of classic RES to refer to multiple ob-jects in one expression or identify the empty targets absent in the image. GRES poses challenges in modeling the com-plex spatial relationships of the instances in the image and identifying non-existing referents. Multimodal Large Lan-guage Models (MLLMs) have recently shown tremendous progress in these complicated vision-language tasks. Con-necting Large Language Models (LLMs) and vision models, MLLMs are proficient in understanding contexts with visual inputs. Among them, LISA, as a representative, adopts a special [SEG] token to prompt a segmentation mask de-coder, e.g., SAM, to enable MLLMs in the RES task. How-ever, existing solutions to GRES remain unsatisfactory since current segmentation MLLMs cannot correctly handle the cases where users might reference multiple subjects in a singular prompt or provide descriptions incongruent with any image target. In this paper, we propose Generalized Segmentation Vision Assistant (GSVA) to address this gap. Specifically, GSVA reuses the [SEG] token to prompt the segmentation model towards supporting multiple mask ref-erences simultaneously and innovatively learns to generate a [REJ] token to reject the null targets explicitly. Experi-ments validate GSVA’s efficacy in resolving the GRES issue, marking a notable enhancement and setting a new record on the GRES benchmark gRefCOCO dataset. GSVA also proves effective across various classic referring segmenta-tion and comprehension tasks. Code will be available at https://github.com/LeapLabTHU/GSVA.

</details>

---

## 216. ScanFormer: Referring Expression Comprehension by Iteratively Scanning

- [ ] ScanFormer: Referring Expression Comprehension by Iteratively Scanning | https://cvpr.thecvf.com/virtual/2024/poster/31612

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31612

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Referring Expression Comprehension (REC) aims to localize the target objects specified by free-form natural language descriptions in images. While state-of-the-art methods achieve impressive performance, they perform dense perception of images, which incorporates redundant visual regions unrelated to linguistic queries, leading to additional computational overhead. This inspires us to explore a question: can we eliminate linguistic-irrelevant redundant visual regions to improve the efficiency of the model ? Existing relevant methods primarily focus on fundamental visual tasks, with limited exploration in vision-language fields. To address this, we propose a coarse-to-fine iterative perception framework, called ScanFormer. It can iteratively exploit the image scale pyramid to extract linguistic-relevant visual patches from top to bottom. In each iteration, irrelevant patches are discarded by our designed informativeness prediction. Furthermore, we propose a patch selection strategy for discarded patches to accelerate inference. Experiments on widely used datasets, namely RefCOCO, RefCOCO+, RefCOCOg, and ReferItGame, verify the effectiveness of our method, which can strike a balance between accuracy and efficiency.

</details>

---

## 217. RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback

- [ ] RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback | https://cvpr.thecvf.com/virtual/2024/poster/31610

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31610

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have recently demonstrated impressive capabilities in multimodal understanding, reasoning, and interaction. However, existing MLLMs prevalently suffer from serious hallucination problems, generating text that is not factually grounded in associated images. The problem makes existing MLLMs untrustworthy and thus impractical in real-world (especially high-stakes) applications. To address the challenge, we present RLHF-V, which enhances MLLM trustworthiness via behavior alignment from fine-grained correctional human feedback. Specifically, RLHF-V collects human preference in the form of segment-level corrections on hallucinations, and performs dense direct preference optimization over the human feedback. Comprehensive experiments on five benchmarks in both automatic and human evaluation show that, RLHF-V can enable substantially more trustworthy MLLM behaviors with promising data and computation efficiency. Remarkably, using 1.4k annotated data samples, RLHF-V significantly reduces the hallucination rate of the base MLLM by 34.8%, outperforming the concurrent LLaVA-RLHF trained on 10k annotated data. The final model achieves state-of-the-art performance in trustworthiness among open-source MLLMs, and shows better robustness than GPT-4V in preventing hallucinations aroused from over-generalization. All the data, code and model weights will be released to facilitate future research.

</details>

---

## 218. Pixel-Aligned Language Model

- [ ] Pixel-Aligned Language Model | https://cvpr.thecvf.com/virtual/2024/poster/31639

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31639

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models have achieved great success in recent years, so as their variants in vision. Existing vision-language models can describe images in natural languages, answer visual-related questions, or perform complex reasoning about the image. However, it is yet unclear how localization tasks, such as word grounding or referring localization, can be performed using large language models. In this work, we aim to develop a vision-language model that can take locations, for example, a set of points or boxes, as either inputs or outputs. When taking locations as inputs, the model performs location-conditioned captioning, which generates captions for the indicated object or region. When generating locations as outputs, our model regresses pixel coordinates for each output word generated by the language model, and thus performs dense word grounding. Our model is pre-trained on the Localized Narrative dataset, which contains pixel-word-aligned captioning from human attention. We show our model can be applied to various location-aware vision-language tasks, including referring localization, location-conditioned captioning, and dense object captioning, archiving state-of-the-art performance on RefCOCO and Visual Genome.

</details>

---

## 219. LION: Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge

- [ ] LION: Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge | https://cvpr.thecvf.com/virtual/2024/poster/31645

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31645

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have endowed LLMs with the ability to perceive and understand multi-modal signals. However, most of the existing MLLMs mainly adopt vision encoders pretrained on coarsely aligned image-text pairs, leading to insufficient extraction and reasoning of visual knowledge. To address this issue, we devise a dual-$\textbf{L}$evel v$\textbf{I}$sual kn$\textbf{O}$wledge e$\textbf{N}$hanced Multimodal Large Language Model ($\textbf{LION}$), which empowers the MLLM by injecting visual knowledge in two levels. $\textbf{1)}$ $\textbf{Progressive}$ $\textbf{incorporation}$ $\textbf{of}$ $\textbf{fine-grained}$ $\textbf{spatial-aware}$ $\textbf{visual}$ $\textbf{knowledge}$. We design a vision aggregator cooperated with region-level vision-language (VL) tasks to incorporate fine-grained spatial-aware visual knowledge into the MLLM. To alleviate the conflict between image-level and region-level VL tasks during incorporation, we devise a dedicated stage-wise instruction-tuning strategy with mixture-of-adapters. This progressive incorporation scheme contributes to the mutual promotion between these two kinds of VL tasks. $\textbf{2)}$ $\textbf{Soft}$ $\textbf{prompting}$ $\textbf{of}$ $\textbf{high-level}$ $\textbf{semantic}$ $\textbf{visual}$ $\textbf{evidence}$. We facilitate the MLLM with high-level semantic visual evidence by leveraging diverse image tags. To mitigate the potential influence caused by imperfect predicted tags, we propose a soft prompting method by embedding a learnable token into the tailored text instruction. Comprehensive experiments on several multi-modal benchmarks demonstrate the superiority of our model ($\textit{e.g.}$, improvement of 5% accuracy on VSR  and 3% CIDEr on TextCaps over InstructBLIP, 5% accuracy on RefCOCOg over Kosmos-2).

</details>

---

## 220. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark

- [ ] MVBench: A Comprehensive Multi-modal Video Understanding Benchmark | https://cvpr.thecvf.com/virtual/2024/poster/31654

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31654

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the rapid development of Multi-modal Large Language Models (MLLMs), a number of diagnostic benchmarks have recently emerged to evaluate the comprehension capabilities of these models. However, most benchmarks predominantly assess spatial understanding in the static image tasks, while overlooking temporal understanding in the dynamic video tasks. To alleviate this issue, we introduce a comprehensive Multi-modal Video understanding Benchmark, namely MVBench, which covers 20 challenging video tasks that can not be effectively solved with a single frame. Specifically, we first introduce a novel static-to-dynamic method to define these temporal-related tasks. By transforming various static tasks into dynamic ones, we enable the systematic generation of video tasks that require a broad spectrum of temporal skills, ranging from perception to cognition. Then, guided by the task definition, we automatically convert public video annotations into multiple-choice QA to evaluate each task. On one hand, such a distinct paradigm allows us to build MVBench efficiently, without much manual intervention. On the other hand, it guarantees evaluation fairness with ground-truth video annotations, avoiding the biased scoring of LLMs. Moreover, we further develop a robust video MLLM baseline, i.e., MVChat, by progressive multi-modal training with diverse instruction-tuning data. The extensive results on our MVBench reveal that, the existing MLLMs are far from satisfactory in temporal understanding, while our MVChat largely surpasses these leading models by over 15% on MVBench. All models and data will be publicly available.

</details>

---

## 221. Zero-shot Referring Expression Comprehension via Structural Similarity Between Images and Captions

- [ ] Zero-shot Referring Expression Comprehension via Structural Similarity Between Images and Captions | https://cvpr.thecvf.com/virtual/2024/poster/31655

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31655

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot referring expression comprehension aims at localizing bounding boxes in an image corresponding to provided textual prompts, which requires: (i) a fine-grained disentanglement of complex visual scene and textual context, and (ii) a capacity to understand relationships among disentangled entities. Unfortunately, existing large vision-language alignment (VLA) models, e.g., CLIP, struggle with both aspects so cannot be directly used for this task. To mitigate this gap, we leverage large foundation models to disentangle both images and texts into triplets in the format of (subject, predicate, object). After that, grounding is accomplished by calculating the structural similarity matrix between visual and textual triplets with a VLA model, and subsequently propagate it to an instance-level similarity matrix. Furthermore, to equip VLA models with the ability of relationship understanding, we design a triplet-matching objective to fine-tune the VLA models on a collection of curated dataset containing abundant entity relationships. Experiments demonstrate that our visual grounding performance increase of up to 19.5% over the SOTA zero-shot model on RefCOCO/+/g. On the more challenging Who’s Waldo dataset, our zero-shot approach achieves comparable accuracy to the fully supervised model

</details>

---

## 222. SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation

- [ ] SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2024/poster/31675

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31675

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation strives to distinguish pixels into different semantic groups from an open set of categories. Most existing  methods explore utilizing pre-trained vision-language models, in which the key is to adapt the image-level model for pixel-level segmentation task. In this paper, we propose a simple encoder-decoder, named SED, for open-vocabulary semantic segmentation, which comprises a hierarchical encoder-based cost map generation and a gradual fusion decoder with category early rejection. The hierarchical encoder-based cost map generation employs hierarchical backbone, instead of plain transformer, to predict pixel-level image-text cost map. Compared to plain transformer, hierarchical backbone better captures local spatial information and has linear computational complexity with respect to input size. Our gradual fusion decoder employs a top-down structure to combine cost map and the feature maps of different backbone levels  for segmentation. To accelerate  inference speed, we introduce a category early rejection scheme in the decoder that rejects many no-existing categories at the early layer of decoder, resulting in at most  4.7 times acceleration without accuracy degradation. Experiments are performed on multiple open-vocabulary semantic segmentation datasets, which demonstrates the efficacy of our SED method. When using ConvNeXt-B, our  SED method achieves mIoU score of 31.6\% on ADE20K with 150 categories at 82 millisecond ($ms$) per image on a single A6000. Our source code is available at https://github.com/xb534/SED.

</details>

---

## 223. Do Vision and Language Encoders Represent the World Similarly?

- [ ] Do Vision and Language Encoders Represent the World Similarly? | https://cvpr.thecvf.com/virtual/2024/poster/31676

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31676

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Aligned text-image encoders such as CLIP have become the de-facto model for vision-language tasks. Furthermore, modality-specific encoders achieve impressive performances in their respective domains. This raises a central question: does an alignment exist between uni-modal vision and language encoders since they fundamentally represent the same physical world? Analyzing the latent spaces structure of vision and language models on image-caption benchmarks using the Centered Kernel Alignment (CKA), we find that the representation spaces of unaligned and aligned encoders are semantically similar. In the absence of statistical similarity in aligned encoders like CLIP, we show that a possible matching of unaligned encoders exists without any training. We frame this as a seeded graph-matching problem exploiting the semantic similarity between graphs and propose two methods - a Fast Quadratic Assignment Problem optimization, and a novel localized CKA metric-based matching/retrieval. We demonstrate the effectiveness of this on several downstream tasks including cross-lingual, cross-domain caption matching and image classification.

</details>

---

## 224. The STVchrono Dataset: Towards Continuous Change Recognition in Time

- [ ] The STVchrono Dataset: Towards Continuous Change Recognition in Time | https://cvpr.thecvf.com/virtual/2024/poster/31679

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31679

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recognizing continuous changes offers valuable insights into past historical events, supports current trend analysis, and facilitates future planning. This knowledge is crucial for a variety of fields, such as meteorology and agriculture, environmental science, urban planning and construction, tourism, and cultural preservation. Currently, available datasets in the field of scene change understanding primarily concentrate on two main tasks: the detection of changed regions within a scene and the linguistic description of the change content. Existing datasets focus on recognizing discrete changes, such as adding or deleting an object from two images, and largely rely on artificially generated images. Consequently, the existing change understanding methods primarily focus on identifying distinct object differences, overlooking the importance of continuous, gradual changes occurring over extended time intervals. To address the above issues, we propose a novel benchmark dataset, STVchrono, targeting the localization and description of long-term continuous changes in real-world scenes. The dataset consists of 71,900 photographs from Google Street View API taken over an 18-year span across 50 cities all over the world. Our STVchrono dataset is designed to support change recognition and description in both image pairs and extended image sequences, while also enabling the segmentation of changed regions. We conduct experiments to evaluate state-of-the-art methods on change description and segmentation, as well as multimodal Large Language Models for describing changes. Our findings reveal that even the most advanced methods lag human performance, emphasizing the need to adapt them to continuously changing real-world scenarios. We hope that our benchmark dataset will further facilitate the research of temporal change recognition in a dynamic world.

</details>

---

## 225. Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation

- [ ] Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation | https://cvpr.thecvf.com/virtual/2024/poster/31681

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31681

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-and-language navigation (VLN) enables the agent to navigate to a remote location following the natural language instruction in 3D environments. At each navigation step, the agent selects from possible candidate locations and then makes the move. For better navigation planning, the lookahead exploration strategy aims to effectively evaluate the agent's next action by accurately anticipating the future environment of candidate locations. To this end, some existing works predict RGB images for future environments, while this strategy suffers from image distortion and high computational cost. To address these issues, we propose the pre-trained hierarchical neural radiance representation model (HNR) to produce multi-level semantic features for future environments, which are more descriptive and efficientthan pixel-wise RGB reconstruction. Furthermore, with the predicted future environmental representations, our lookahead VLN model is able to construct the navigable future path tree and select the optimal path branch via efficient parallel evaluation. Extensive experiments on the VLN-CE datasets confirm the effectiveness of our proposed method.

</details>

---

## 226. WonderJourney: Going from Anywhere to Everywhere

- [ ] WonderJourney: Going from Anywhere to Everywhere | https://cvpr.thecvf.com/virtual/2024/poster/31689

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31689

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce WonderJourney, a modular framework for perpetual 3D scene generation. Unlike prior work on view generation that focuses on a single type of scenes, we start at any user-provided location (by a text description or an image), and generate a journey through a long sequence of diverse yet coherently connected 3D scenes. We leverage an LLM to generate textual descriptions of the scenes in this journey, a text-driven point cloud generation pipeline to make a compelling and coherent sequence of 3D scenes, and a large VLM to verify the generated scenes. We show compelling, diverse visual results across various scene types and styles, forming imaginary “wonderjourneys”. Project website: https://kovenyu.com/WonderJourney/.

</details>

---

## 227. Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding

- [ ] Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding | https://cvpr.thecvf.com/virtual/2024/poster/31713

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31713

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models have demonstrated impressive universal capabilities across a wide range of open-ended tasks and have extended their utility to encompass multimodal conversations. However, existing methods encounter challenges in effectively handling both image and video understanding, particularly with limited visual tokens. In this work, we introduce Chat-UniVi, a Unified Vision-language model capable of comprehending and engaging in conversations involving images and videos through a unified visual representation. Specifically, we employ a set of dynamic visual tokens to uniformly represent images and videos. This representation framework empowers the model to efficiently utilize a limited number of visual tokens to simultaneously capture the spatial details necessary for images and the comprehensive temporal relationship required for videos. Moreover, we leverage a multi-scale representation, enabling the model to perceive both high-level semantic concepts and low-level visual details. Notably, Chat-UniVi is trained on a mixed dataset containing both images and videos, allowing direct application to tasks involving both mediums without requiring any modifications. Extensive experimental results demonstrate that Chat-UniVi consistently outperforms even existing methods exclusively designed for either images or videos. Code is available at https://github.com/PKU-YuanGroup/Chat-UniVi.

</details>

---

## 228. JRDB-Social: A Multifaceted Robotic Dataset for Understanding of Context and Dynamics of Human Interactions Within Social Groups

- [ ] JRDB-Social: A Multifaceted Robotic Dataset for Understanding of Context and Dynamics of Human Interactions Within Social Groups | https://cvpr.thecvf.com/virtual/2024/poster/31714

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31714

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding human social behaviour is crucial in computer vision and robotics. Micro-level observations like individual actions fall short, necessitating a comprehensive approach that considers individual behaviour, intra-group dynamics, and social group levels for a thorough understanding. To address dataset limitations, this paper introduces JRDB-Social, an extension of JRDB. Designed to fill gaps in human understanding across diverse indoor and outdoor social contexts, JRDB-Social provides annotations at three levels: individual attributes, intra-group interactions, and social group context. This dataset aims to enhance our grasp of human social dynamics for robotic applications. Utilizing the recent cutting-edge multi-modal large language models, we evaluated our benchmark to explore their capacity to decipher social human behaviour.

</details>

---

## 229. MeaCap: Memory-Augmented Zero-shot Image Captioning

- [ ] MeaCap: Memory-Augmented Zero-shot Image Captioning | https://cvpr.thecvf.com/virtual/2024/poster/31735

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31735

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot image captioning (IC) without well-paired image-text data can be divided into two categories: training-free and text-only-training. Generally, these two types of methods realize zero-shot IC by integrating pre-trained vision-language models like CLIP for image-text similarity evaluation and a pre-trained language model (LM) for caption generation. The main difference between them is whether to use a textual corpus to train the LM. Despite achieving attractive performance with respect to some metrics, existing methods often exhibit common drawbacks. Training-free methods tend to produce hallucinations, while text-only-training methods often lose generalization capability. To advance the field, this paper proposes a novel Memory-Augmented zero-shot image Captioning framework (MeaCap). Specifically, equipped with textual memory, we introduce a retrieve-then-filter module to extract key concepts highly related to the image. By deploying our proposed memory-augmented visual-related fusion score in a keywords-to-sentence LM, MeaCap can generate concept-centered captions that maintain high consistency with the image, reducing hallucinations and incorporating more world knowledge. The MeaCap framework achieves state-of-the-art performance across a series of zero-shot IC settings. The code is provided in the Supplement for further exploration.

</details>

---

## 230. Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection

- [ ] Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection | https://cvpr.thecvf.com/virtual/2024/poster/31740

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31740

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we study the problem of generalizable synthetic image detection, aiming to detect forgery images from diverse generative methods, e.g., GANs and diffusion models. Cutting-edge solutions start to explore the benefits of pre-trained models, and mainly follow the fixed paradigm of solely training an attached classifier, e.g., combining frozen CLIP-ViT with a learnable linear layer in UniFD [35]. However, our analysis shows that such a fixed paradigm is prone to yield detectors with insufficient learning regarding forgery representations. We attribute the key challenge to the lack of forgery adaptation, and present a novel forgery-aware adaptive transformer approach, namely FatFormer. Based on the pre-trained vision-language spaces of CLIP, FatFormer introduces two core designs for the adaption to build generalized forgery representations. First, motivated by the fact that both image and frequency analysis are essential for synthetic image detection, we develop a forgery-aware adapter to adapt image features to discern and integrate local forgery traces within image and frequency domains. Second, we find that considering the contrastive objectives between adapted image features and text prompt embeddings, a previously overlooked aspect, results in a nontrivial generalization improvement. Accordingly, we introduce language-guided alignment to supervise the forgery adaptation with image and text prompts in FatFormer. Experiments show that, by coupling these two designs, our approach tuned on 4-class ProGAN data attains a remarkable detection performance, achieving an average of 98% accuracy to unseen GANs, and surprisingly generalizes to unseen diffusion models with 95% accuracy.

</details>

---

## 231. Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning

- [ ] Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning | https://cvpr.thecvf.com/virtual/2024/poster/31750

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31750

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent compositional zero-shot learning (CZSL) methods adapt pre-trained vision-language models (VLMs) by constructing trainable prompts only for composed state-object pairs. Relying on learning the joint representation of seen compositions, these methods ignore the explicit modeling of the state and object, thus limiting the exploitation of pre-trained knowledge and generalization to unseen compositions. With a particular focus on the universality of the solution, in this work, we propose a novel paradigm for CZSL models that establishes three identification branches (i.e., Multi-Path) to jointly model the state, object, and composition. The presented Troika is an outstanding implementation that aligns the branch-specific prompt representations with decomposed visual features. To calibrate the bias between semantically similar multi-modal representations, we further devise a Cross-Modal Traction module into Troika that shifts the prompt representation towards the current visual content. We conduct extensive experiments on three popular benchmarks, where our method significantly outperforms existing methods in both closed-world and open-world settings.

</details>

---

## 232. Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models

- [ ] Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models | https://cvpr.thecvf.com/virtual/2024/poster/31760

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31760

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modality large language models (MLLMs), as represented by GPT-4V, have introduced a paradigm shift for visual perception and understanding tasks, that a variety of abilities can be achieved within one foundation model. While current MLLMs demonstrate primary low-level visual abilities from the identification of low-level visual attributes (e.g., clarity, brightness) to the evaluation on image quality, there's still an imperative to further improve the accuracy of MLLMs to substantially alleviate human burdens. To address this, we collect the first dataset consisting of human natural language feedback on low-level vision. Each feedback offers a comprehensive description of an image's low-level visual attributes, culminating in an overall quality assessment. The constructed Q-Pathway dataset includes 58K detailed human feedbacks on 18,973 multi-sourced images with diverse low-level appearance. To ensure MLLMs can adeptly handle diverse queries, we further propose a GPT-participated transformation to convert these feedbacks into a rich set of 200K instruction-response pairs, termed Q-Instruct. Experimental results indicate that the Q-Instruct consistently elevates various low-level visual capabilities across multiple base models. We anticipate that our datasets can pave the way for a future that foundation models can assist humans on low-level visual tasks.

</details>

---

## 233. mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration

- [ ] mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration | https://cvpr.thecvf.com/virtual/2024/poster/31761

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31761

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) have demonstrated impressive instruction abilities across various open-ended tasks. However, previous methods have primarily focused on enhancing multi-modal capabilities. In this work, we introduce a versatile multi-modal large language model, mPLUG-Owl2, which effectively leverages modality collaboration to improve performance in both text and multi-modal tasks. mPLUG-Owl2 utilizes a modularized network design, with the language decoder acting as a universal interface for managing different modalities. Specifically, mPLUG-Owl2 incorporates shared functional modules to facilitate modality collaboration and introduces a modality-adaptive module that preserves modality-specific features. Extensive experiments reveal that mPLUG-Owl2 is capable of generalizing both text tasks and multi-modal tasks while achieving state-of-the-art performances with a single generalized model. Notably, mPLUG-Owl2 is the first MLLM model that demonstrates the modality collaboration phenomenon in both pure-text and multi-modal scenarios, setting a pioneering path in the development of future multi-modal foundation models.

</details>

---

## 234. Koala: Key Frame-Conditioned Long Video-LLM

- [ ] Koala: Key Frame-Conditioned Long Video-LLM | https://cvpr.thecvf.com/virtual/2024/poster/31782

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31782

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Long video question answering is a challenging task that involves recognizing short-term activities and reasoning about their fine-grained relationships. State-of-the-art video Large Language Models (vLLMs) hold promise as a viable solution due to their demonstrated emergent capabilities on new tasks. However, despite being trained on millions of short seconds-long videos, vLLMs are unable to understand minutes-long videos and accurately answer questions about them. To address this limitation, we propose a lightweight and self-supervised approach, Key frame-conditioned long video-LLM (Koala), that introduces learnable spatiotemporal queries to adapt pretrained vLLMs for generalizing to longer videos. Our approach introduces two new tokenizers that condition on visual tokens computed from sparse video key frames for understanding short and long video moments. We train our proposed approach on HowTo100M and  demonstrate its effectiveness on zero-shot long video understanding benchmarks, where it outperforms state-of-the-art large models by 3 - 6% in absolute accuracy across all tasks. Surprisingly, we also empirically show that our approach not only helps a pretrained vLLM to understand long videos but also improves its accuracy on short-term action recognition.

</details>

---

## 235. PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization

- [ ] PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization | https://cvpr.thecvf.com/virtual/2024/poster/31781

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31781

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Domain Generalization (DG) aims to resolve distribution shifts between source and target domains, and current DG methods are default to the setting that data from source and target domains share identical categories. Nevertheless, there exists unseen classes from target domains in practical scenarios. To address this issue, Open Set Domain Generalization (OSDG) has emerged and several methods have been exclusively proposed. However, most existing methods adopt complex architectures with slight improvement compared with DG methods. Recently, vision-language models (VLMs) have been introduced in DG following the fine-tuning paradigm, but consume huge training overhead with large vision models. Therefore, in this paper, we innovate to transfer knowledge from VLMs to lightweight vision models and improve the robustness by introducing Perturbation Distillation (PD) from three perspectives, including Score, Class and Instance (SCI), named SCI-PD. Moreover, previous methods are oriented by the benchmarks with identical and fixed splits, ignoring the divergence between source domains. These methods are revealed to suffer from sharp performance decay with our proposed new benchmark Hybrid Domain Generalization (HDG) and a novel metric $H^{2}$-CV, which construct various splits to comprehensively assess the robustness of algorithms. Extensive experiments demonstrate that our method outperforms state-of-the-art algorithms on multiple datasets, especially improving the robustness when confronting data scarcity.

</details>

---

## 236. Domain Prompt Learning with Quaternion Networks

- [ ] Domain Prompt Learning with Quaternion Networks | https://cvpr.thecvf.com/virtual/2024/poster/31794

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31794

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has emerged as an effective and data-efficient technique in large Vision-Language Models (VLMs). However, when adapting VLMs to specialized domains such as remote sensing and medical imaging, domain prompt learning remains underexplored. While large-scale domain-specific foundation models can help tackle this challenge, their concentration on a single vision level makes it challenging to prompt both vision and language modalities. To overcome this, we propose to leverage domain-specific knowledge from domain-specific foundation models to transfer the robust recognition ability of VLMs from generalized to specialized domains, using quaternion networks. Specifically, the proposed method involves using domain-specific vision features from domain-specific foundation models to guide the transformation of generalized contextual embeddings from the language branch into a specialized space within the quaternion networks. Moreover, we present a hierarchical approach that generates vision prompt features by analyzing intermodal relationships between hierarchical language prompt features and domain-specific vision features. In this way, quaternion networks can effectively mine the intermodal relationships in the specific domain, facilitating domain-specific vision-language contrastive learning. Extensive experiments on domain-specific datasets show that our proposed method achieves new state-of-the-art results in prompt learning.

</details>

---

## 237. Osprey: Pixel Understanding with Visual Instruction Tuning

- [ ] Osprey: Pixel Understanding with Visual Instruction Tuning | https://cvpr.thecvf.com/virtual/2024/poster/31799

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31799

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have recently achieved impressive general-purpose vision-language capabilities through visual instruction tuning. However, current MLLMs primarily focus on image-level or box-level understanding, falling short in achieving fine-grained vision-language alignment at pixel level. Besides, the lack of mask-based instruction data limits their advancements. In this paper, we propose Osprey, a mask-text instruction tuning approach, to extend MLLMs by incorporating fine-grained mask regions into language instruction, aiming at achieving pixel-wise visual understanding. To achieve this goal, we first meticulously curate a mask-based region-text dataset with 724K samples, and then design a vision-language model by injecting pixel-level representation into LLM. Specifically, Osprey adopts a convolutional CLIP backbone as the vision encoder and employs a mask-aware visual extractor to extract precise visual mask features from high resolution input.  Experimental results demonstrate  Osprey's superiority in various region understanding tasks, showcasing its new capability for pixel-level instruction tuning.  In particular, Osprey can be integrated with Segment Anything Model (SAM) seamlessly to obtain multi-granularity semantics. The source code, dataset and demo can be found at https://github.com/CircleRadon/Osprey.

</details>

---

## 238. Overcoming Generic Knowledge Loss with Selective Parameter Update

- [ ] Overcoming Generic Knowledge Loss with Selective Parameter Update | https://cvpr.thecvf.com/virtual/2024/poster/31810

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31810

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Foundation models encompass an extensive knowledge base and offer remarkable transferability. However, this knowledge becomes outdated or insufficient over time. The challenge lies in continuously updating foundation models to accommodate novel information while retaining their original capabilities.  Leveraging the fact that foundation models have initial knowledge on various tasks and domains, we propose a novel approach that,  instead of updating all parameters equally,  localizes the updates to a sparse set of parameters relevant to the task being learned. We strike a  balance between efficiency and new task performance, while maintaining the transferability and generalizability of foundation models. We extensively evaluate our method on foundational vision-language models with a diverse spectrum of continual learning tasks. Our method achieves improvements on the accuracy of the newly learned tasks up to 7% while preserving the pretraining knowledge with a negligible decrease of 0.9% on a representative control set accuracy.

</details>

---

## 239. Harnessing the Power of MLLMs for Transferable Text-to-Image Person ReID

- [ ] Harnessing the Power of MLLMs for Transferable Text-to-Image Person ReID | https://cvpr.thecvf.com/virtual/2024/poster/31838

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31838

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Text-to-image person re-identification (ReID) retrieves pedestrian images according to textual descriptions. Manually annotating textual descriptions is time-consuming, restricting the scale of existing datasets and therefore the generalization ability of ReID models. As a result, we study the transferable text-to-image ReID problem, where we train a model on our proposed large-scale database and directly deploy it to various datasets for evaluation. We obtain substantial training data via Multi-modal Large Language Models (MLLMs). Moreover, we identify and address two key challenges in utilizing the obtained textual descriptions. First, an MLLM tends to generate descriptions with similar structures, causing the model to overfit specific sentence patterns. Thus, we propose a novel method that uses MLLMs to caption images according to various templates. These templates are obtained using a multi-turn dialogue with a Large Language Model (LLM). Therefore, we can build a large-scale dataset with diverse textual descriptions. Second, an MLLM may produce incorrect descriptions. Hence, we introduce a novel method that automatically identifies words in a description that do not correspond with the image. This method is based on the similarity between one text and all patch token embeddings in the image. Then, we mask these words with a larger probability in the subsequent training epoch, alleviating the impact of noisy textual descriptions. The experimental results demonstrate that our methods significantly boost the direct transfer text-to-image ReID performance. Benefiting from the pre-trained model weights, we also achieve state-of-the-art performance in the traditional evaluation settings. The code and dataset for this study will be released.

</details>

---

## 240. Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions

- [ ] Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions | https://cvpr.thecvf.com/virtual/2024/poster/31839

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31839

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The zero-shot performance of existing vision-language models (VLMs) such as CLIP is limited by the availability of large-scale, aligned image and text datasets in specific domains. In this work, we leverage two complementary sources of information---descriptions of categories generated by large language models (LLMs) and abundant, fine-grained image classification datasets---to improve the zero-shot classification performance of VLMs across fine-grained domains. On the technical side, we develop methods to train VLMs with this ``bag-level" image-text supervision. We find that simply using these attributes at test-time does not improve performance, but our training strategy, for example, on the iNaturalist dataset, leads to an average improvement of 4-5\% in zero-shot classification accuracy for novel categories of birds and flowers. Similar improvements are observed in domains where a subset of the categories was used to fine-tune the model. By prompting LLMs in various ways, we generate descriptions that capture visual appearance, habitat, and geographical regions and pair them with existing attributes such as the taxonomic structure of the categories. We systematically evaluate their ability to improve zero-shot categorization in natural domains. Our findings suggest that geographic priors can be just as effective and are complementary to visual appearance. Our method also outperforms prior work on prompt-based tuning of VLMs. We plan to release the benchmark, consisting of 7 datasets, which will contribute to future research in zero-shot recognition.

</details>

---

## 241. THRONE: An Object-based Hallucination Benchmark for the Free-form Generations of Large Vision-Language Models

- [ ] THRONE: An Object-based Hallucination Benchmark for the Free-form Generations of Large Vision-Language Models | https://cvpr.thecvf.com/virtual/2024/poster/31860

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31860

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Mitigating hallucinations in large vision-language models (LVLMs) remains an open problem. Recent benchmarks do not address hallucinations in open-ended free-form responses, which we term “Type I hallucinations”. They focus on, if at all, hallucinations responding to very specific questions—yes-no or multiple-choice questions regarding a particular object or attribute—which we term “Type II hallucinations”, and they often require closed-source models which are subject to arbitrary change. Additionally, we observe a reduction in Type II hallucinations does not lead to a congruent reduction in Type I hallucations; rather, it often increases. We propose THRONE, a novel automatic framework for quantitatively evaluating Type I hallucinations in LVLM free-form outputs. We use public language models (LMs) to identify hallucinations in LVLM responses and compute informative metrics. We evaluate a large selection of recent LVLMs using public datasets. Our results show advances on existing metrics are disconnected from the reduction of Type I hallucinations, and established benchmarks for measuring Type I hallucination prevalence are incomplete. Finally, we provide a simple and effective data augmentation method to reduce Type I and Type II hallucinations as a strong baseline.

</details>

---

## 242. Bootstrapping SparseFormers from Vision Foundation Models

- [ ] Bootstrapping SparseFormers from Vision Foundation Models | https://cvpr.thecvf.com/virtual/2024/poster/31862

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31862

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recently proposed SparseFormer architecture provides an alternative approach to visual understanding by utilizing a significantly lower number of visual tokens via adjusting RoIs, greatly reducing computational costs while still achieving promising performance. However, training SparseFormers from scratch is still expensive, and scaling up the number of parameters can be challenging. In this paper, we propose to bootstrap SparseFormers from ViT-based vision foundation models in a simple and efficient way. Since the majority of SparseFormer blocks are the standard transformer ones, we can inherit weights from large-scale pre-trained vision transformers and freeze them as much as possible. Therefore, we only need to train the SparseFormer-specific lightweight focusing transformer to adjust token RoIs and fine-tune a few early pre-trained blocks to align the final token representation. In such a way, we can bootstrap SparseFormer architectures from various large-scale pre-trained models (e.g., IN-21K pre-trained AugRegs or CLIPs) using a rather smaller amount of training samples (e.g., IN-1K) and without labels or captions within just a few hours. As a result, the bootstrapped unimodal SparseFormer (from AugReg-ViT-L/16-384) can reach $84.9\%$ accuracy on IN-1K with only $49$ tokens, and the multimodal SparseFormer from CLIPs also demonstrates notable zero-shot performance with highly reduced computational cost without seeing any caption during the bootstrapping procedure. In addition, CLIP-bootstrapped SparseFormers, which align the output space with language without seeing a word, can serve as efficient vision encoders in multimodal large language models. Code and models are available at https://github.com/showlab/sparseformer}{https://github.com/showlab/sparseformer

</details>

---

## 243. Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs

- [ ] Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs | https://cvpr.thecvf.com/virtual/2024/poster/31877

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31877

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Integration of Large Language Models (LLMs) into visual domain tasks, resulting in visual-LLMs (V-LLMs), has enabled exceptional performance in vision-language tasks, particularly for visual question answering (VQA). However, existing V-LLMs (e.g. BLIP-2, LLaVA) demonstrate weak spatial reasoning and localization awareness. Despite generating highly descriptive and elaborate textual answers, these models fail at simple tasks like distinguishing a left vs right location. In this work, we explore how image-space coordinate based instruction fine-tuning objectives could inject spatial awareness into V-LLMs. We discover optimal coordinate representations, data-efficient instruction fine-tuning objectives, and pseudo-data generation strategies that lead to improved spatial awareness in V-LLMs. Additionally, our resulting model improves VQA across image and video domains, reduces undesired hallucination, and generates better contextual object descriptions. Experiments across 5 vision-language tasks involving 14 different datasets establish the clear performance improvements achieved by our proposed framework.

</details>

---

## 244. FairCLIP: Harnessing Fairness in Vision-Language Learning

- [ ] FairCLIP: Harnessing Fairness in Vision-Language Learning | https://cvpr.thecvf.com/virtual/2024/poster/31879

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31879

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fairness is a critical concern in deep learning, especially in healthcare, where these models influence diagnoses and treatment decisions. Although fairness has been investigated in the vision-only domain, the fairness of medical vision-language (VL) models remains unexplored due to the scarcity of medical VL datasets for studying fairness. To bridge this research gap, we introduce the first fair vision-language medical dataset FairVLMed that provides detailed demographic attributes, ground-truth labels, and clinical notes to facilitate an in-depth examination of fairness within VL foundation models. Using FairVLMed, we conduct a comprehensive fairness analysis of two widely-used VL models (CLIP and BLIP2), pre-trained on both natural and medical domains, across four different protected attributes. Our results highlight significant biases in all VL models, with Asian, Male, Non-Hispanic, and Spanish being the preferred subgroups across the protected attributes of race, gender, ethnicity, and language, respectively. In order to alleviate these biases, we propose FairCLIP, an optimal-transport-based approach that achieves a favorable trade-off between performance and fairness by reducing the Sinkhorn distance between the overall sample distribution and the distributions corresponding to each demographic group. As the first VL dataset of its kind, FairVLMed holds the potential to catalyze advancements in the development of machine learning models that are both ethically aware and clinically effective. Our dataset and code are available at https://ophai.hms.harvard.edu/datasets/fairvlmed10k.

</details>

---

## 245. ViLa-MIL: Dual-scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification

- [ ] ViLa-MIL: Dual-scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification | https://cvpr.thecvf.com/virtual/2024/poster/31886

- **Link**: https://cvpr.thecvf.com/virtual/2024/poster/31886

- **Conference**: CVPR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multiple instance learning (MIL)-based framework has become the mainstream for processing the whole slide image (WSI) with giga-pixel size and hierarchical image context in digital pathology. However, these methods heavily depend on a substantial number of bag-level labels and solely learn from the original slides, which are easily affected by variations in data distribution. Recently, vision language model (VLM)-based methods introduced the language prior by pre-training on large-scale pathological image-text pairs. However, the previous text prompt lacks the consideration of pathological prior knowledge, therefore does not substantially boost the model's performance. Moreover, the collection of such pairs and the pre-training process are very time-consuming and source-intensive. To solve the above problems, we propose a dual-scale vision-language multiple instance learning (ViLa-MIL) framework for whole slide image classification. Specifically, we propose a dual-scale visual descriptive text prompt based on the frozen large language model (LLM) to boost the performance of VLM effectively. To transfer the VLM to process WSI efficiently, for the image branch, we propose a prototype-guided patch decoder to aggregate the patch features progressively by grouping similar patches into the same prototype; for the text branch, we introduce a context-guided text decoder to enhance the text features by incorporating the multi-granular image contexts. Extensive studies on three multi-cancer and multi-center subtyping datasets demonstrate the superiority of ViLa-MIL.

</details>

---

