# ICLR 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_iclr2024_papers.csv

## 1. Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models

- [ ] Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models | https://iclr.cc/virtual/2024/poster/17371

- **Link**: https://iclr.cc/virtual/2024/poster/17371

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the prevalence of large-scale pretrained vision-language models (VLMs), such as CLIP, soft-prompt tuning has become a popular method for adapting these models to various downstream tasks. However, few works delve into the inherent properties of learnable soft-prompt vectors, specifically the impact of their norms to the performance of VLMs. This motivates us to pose an unexplored research question: ``Do we need to normalize the soft prompts in VLMs?'' To fill this research gap, we first uncover a phenomenon, called the $\textbf{Low-Norm Effect}$ by performing extensive corruption experiments, suggesting that reducing the norms of certain learned prompts occasionally enhances the performance of VLMs, while increasing them often degrades it. To harness this effect, we propose a novel method named $\textbf{N}$ormalizing th$\textbf{e}$ soft-pro$\textbf{m}$pt v$\textbf{e}$ctors of vi$\textbf{si}$on-language model$\textbf{s}$ ($\textbf{Nemesis}$) to normalize soft-prompt vectors in VLMs. To the best of our knowledge, our work is the first to systematically investigate the role of norms of soft-prompt vector in VLMs, offering valuable insights for future research in soft-prompt tuning.

</details>

---

## 2. VDC: Versatile Data Cleanser based on Visual-Linguistic Inconsistency by Multimodal Large Language Models

- [ ] VDC: Versatile Data Cleanser based on Visual-Linguistic Inconsistency by Multimodal Large Language Models | https://iclr.cc/virtual/2024/poster/17409

- **Link**: https://iclr.cc/virtual/2024/poster/17409

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The role of data in building AI systems has recently been emphasized by the emerging concept of data-centric AI. Unfortunately, in the real-world, datasets may contain dirty samples, such as poisoned samples from backdoor attack, noisy labels in crowdsourcing, and even hybrids of them. The presence of such dirty samples makes the DNNs vunerable and unreliable.Hence, it is critical to detect dirty samples to improve the quality and realiability of dataset. Existing detectors only focus on detecting poisoned samples or noisy labels, that are often prone to weak generalization when dealing with dirty samples from other fields.In this paper, we find a commonality of various dirty samples is visual-linguistic inconsistency between images and associated labels. To capture the semantic inconsistency between modalities, we propose versatile data cleanser (VDC) leveraging the surpassing capabilities of multimodal large language models (MLLM) in cross-modal alignment and reasoning.It consists of three consecutive modules: the visual question generation module to generate insightful questions about the image; the visual question answering module to acquire the semantics of the visual content by answering the questions with MLLM; followed by the visual answer evaluation module to evaluate the inconsistency.Extensive experiments demonstrate its superior performance and generalization to various categories and types of dirty samples.The code is available at https://github.com/zihao-ai/vdc .

</details>

---

## 3. DreamLLM: Synergistic Multimodal Comprehension and Creation

- [ ] DreamLLM: Synergistic Multimodal Comprehension and Creation | https://iclr.cc/virtual/2024/poster/17430

- **Link**: https://iclr.cc/virtual/2024/poster/17430

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents DreamLLM, a learning framework that first achieves versatile Multimodal Large Language Models (MLLMs) empowered with frequently overlooked synergy between multimodal comprehension and creation. DreamLLM operates on two fundamental principles. The first focuses on the generative modeling of both language and image posteriors by direct sampling in the raw multimodal space. This approach circumvents the limitations and information loss inherent to external feature extractors like CLIP, and a more thorough multimodal understanding is obtained. Second, DreamLLM fosters the generation of raw, interleaved documents, modeling both text and image contents, along with unstructured layouts. This allows DreamLLM to learn all conditional, marginal, and joint multimodal distributions effectively. As a result, DreamLLM is the first MLLM capable of generating free-form interleaved content. Comprehensive experiments highlight DreamLLM's superior performance as a zero-shot multimodal generalist, reaping from the enhanced learning synergy. Project page: https://dreamllm.github.io.

</details>

---

## 4. Negative Label Guided OOD Detection with Pretrained Vision-Language Models

- [ ] Negative Label Guided OOD Detection with Pretrained Vision-Language Models | https://iclr.cc/virtual/2024/poster/17453

- **Link**: https://iclr.cc/virtual/2024/poster/17453

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection aims at identifying samples from unknown classes, playing a crucial role in trustworthy models against errors on unexpected inputs.  Extensive research has been dedicated to exploring OOD detection in the vision modality. {Vision-language models (VLMs) can leverage both textual and visual information for various multi-modal applications, whereas few OOD detection methods take into account information from the text modality. In this paper, we propose a novel post hoc OOD detection method, called NegLabel, which takes a vast number of negative labels from extensive corpus databases. We design a novel scheme for the OOD score collaborated with negative labels.Theoretical analysis helps to understand the mechanism of negative labels. Extensive experiments demonstrate that our method NegLabel achieves state-of-the-art performance on various OOD detection benchmarks and generalizes well on multiple VLM architectures. Furthermore, our method NegLabel exhibits remarkable robustness against diverse domain shifts. The codes are available at https://github.com/tmlr-group/NegLabel.

</details>

---

## 5. Tag2Text: Guiding Vision-Language Model via Image Tagging

- [ ] Tag2Text: Guiding Vision-Language Model via Image Tagging | https://iclr.cc/virtual/2024/poster/17468

- **Link**: https://iclr.cc/virtual/2024/poster/17468

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents Tag2Text, a vision language pre-training (VLP) framework, which introduces image tagging into vision-language models to guide the learning of visual-linguistic features. In contrast to prior works which utilize object tags either manually labeled or automatically detected with a limited detector, our approach utilizes tags parsed from its paired text to learn an image tagger and meanwhile provides guidance to vision-language models. Given that, Tag2Text can utilize large-scale annotation-free image tags in accordance with image-text pairs, and provides more diverse tag categories beyond objects. Strikingly, Tag2Text showcases the ability of a foundational image tagging model, with superior zero-shot performance even comparable to full supervision manner. Moreover, by leveraging tagging guidance, Tag2Text effectively enhances the performance of vision-language models on both generation-based and alignment-based tasks. Across a wide range of downstream benchmarks, Tag2Text achieves state-of-the-art results with similar model sizes and data scales, demonstrating the efficacy of the proposed tagging guidance.

</details>

---

## 6. Consistency-guided Prompt Learning for Vision-Language Models

- [ ] Consistency-guided Prompt Learning for Vision-Language Models | https://iclr.cc/virtual/2024/poster/17475

- **Link**: https://iclr.cc/virtual/2024/poster/17475

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose Consistency-guided Prompt learning (CoPrompt), a new fine-tuning method for vision-language models. Our approach improves the generalization of large foundation models when fine-tuned on downstream tasks in a few-shot setting. The basic idea of CoPrompt is to enforce a consistency constraint in the prediction of the trainable and pre-trained models to prevent overfitting on the downstream task. Additionally, we introduce the following two components into our consistency constraint to further boost the performance: enforcing consistency on two perturbed inputs and combining two dominant paradigms of tuning, prompting and adapter. Enforcing consistency on perturbed input serves to further regularize the consistency constraint, thereby improving generalization. Moreover, the integration of adapters and prompts not only enhances performance on downstream tasks but also offers increased tuning flexibility in both input and output spaces. This facilitates more effective adaptation to downstream tasks in a few-shot learning setting. Experiments show that CoPrompt outperforms existing methods on a range of evaluation suites, including base-to-novel generalization, domain generalization, and cross-dataset evaluation. On generalization, CoPrompt improves the state-of-the-art on zero-shot tasks and the overall harmonic mean over 11 datasets. Detailed ablation studies show the effectiveness of each of the components in CoPrompt. We make our code available at https://github.com/ShuvenduRoy/CoPrompt.

</details>

---

## 7. BarLeRIa: An Efficient Tuning Framework for Referring Image Segmentation

- [ ] BarLeRIa: An Efficient Tuning Framework for Referring Image Segmentation | https://iclr.cc/virtual/2024/poster/17498

- **Link**: https://iclr.cc/virtual/2024/poster/17498

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-training followed by full fine-tuning has gradually been substituted by Parameter-Efficient Tuning (PET) in the field of computer vision. PET has gained popularity, especially in the context of large-scale models, due to its ability to reduce transfer learning costs and conserve hardware resources. However, existing PET approaches primarily focus on recognition tasks and typically support uni-modal optimization, while neglecting dense prediction tasks and vision language interactions. To address this limitation, we propose a novel PET framework called B i-direction a l Inte r twined Vision L anguage Effici e nt Tuning for R eferring I mage Segment a tion ( BarLeRIa ), which leverages bi-directional intertwined vision language adapters to fully exploit the frozen pre-trained models' potential in cross-modal dense prediction tasks. In BarLeRIa, two different tuning modules are employed for efficient attention, one for global, and the other for local, along with an intertwined vision language tuning module for efficient modal fusion.Extensive experiments conducted on RIS benchmarks demonstrate the superiority of BarLeRIa over prior PET methods with a significant margin, i.e., achieving an average improvement of 5.6\%. Remarkably, without requiring additional training datasets, BarLeRIa even surpasses SOTA full fine-tuning approaches. The code is available at https://github.com/NastrondAd/BarLeRIa.

</details>

---

## 8. Remote Sensing Vision-Language Foundation Models without Annotations via Ground Remote Alignment

- [ ] Remote Sensing Vision-Language Foundation Models without Annotations via Ground Remote Alignment | https://iclr.cc/virtual/2024/poster/17503

- **Link**: https://iclr.cc/virtual/2024/poster/17503

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a method to train vision-language models for remote-sensing images without using any textual annotations. Our key insight is to use co-located internet imagery taken on the ground as an intermediary for connecting remote-sensing images and language.  Specifically, we train an image encoder for remote sensing images to align with the image encoder of CLIP using a large amount of paired internet and satellite images.  Our unsupervised approach enables the training of a first-of-its-kind large scale VLM for remote sensing images at two different resolutions. We show that these VLMs enable zero-shot, open-vocabulary image classification, retrieval, segmentation and visual question answering for satellite images. On each of these tasks, our VLM trained without textual annotations outperforms existing VLMs trained with supervision, with gains of up to 20\% for classification and 80\% for segmentation.

</details>

---

## 9. LLMs Meet VLMs: Boost Open Vocabulary Object Detection with Fine-grained Descriptors

- [ ] LLMs Meet VLMs: Boost Open Vocabulary Object Detection with Fine-grained Descriptors | https://iclr.cc/virtual/2024/poster/17561

- **Link**: https://iclr.cc/virtual/2024/poster/17561

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Inspired by the outstanding zero-shot capability of vision language models (VLMs) in image classification tasks, open-vocabulary object detection has attracted increasing interest by distilling the broad VLM knowledge into detector training. However, most existing open-vocabulary detectors learn by aligning region embeddings with categorical labels (e.g., bicycle) only, disregarding the capability of VLMs on aligning visual embeddings with fine-grained text descriptions of object parts (e.g., pedals and bells). This paper presents DVDet, a Descriptor-Enhanced Open Vocabulary Detector that introduces conditional context prompts and hierarchical textual descriptors that enable precise region-text alignment as well as open-vocabulary detection training in general. Specifically, the conditional context prompt transforms regional embeddings into image-like representations that can be directly integrated into general open vocabulary detection training. In addition, we introduce large language models as an interactive and implicit knowledge repository which enables iterative mining and refining visually oriented textual descriptors for precise region-text alignment. Extensive experiments over multiple large-scale benchmarks show that DVDet outperforms the state-of-the-art consistently by large margins.

</details>

---

## 10. Controlling Vision-Language Models for Multi-Task Image Restoration

- [ ] Controlling Vision-Language Models for Multi-Task Image Restoration | https://iclr.cc/virtual/2024/poster/17626

- **Link**: https://iclr.cc/virtual/2024/poster/17626

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models such as CLIP have shown great impact on diverse downstream tasks for zero-shot or label-free predictions. However, when it comes to low-level vision such as image restoration their performance deteriorates dramatically due to corrupted inputs. In this paper, we present a degradation-aware vision-language model (DA-CLIP) to better transfer pretrained vision-language models to low-level vision tasks as a multi-task framework for image restoration. More specifically, DA-CLIP trains an additional controller that adapts the fixed CLIP image encoder to predict high-quality feature embeddings. By integrating the embedding into an image restoration network via cross-attention, we are able to pilot the model to learn a high-fidelity image reconstruction. The controller itself will also output a degradation feature that matches the real corruptions of the input, yielding a natural classifier for different degradation types. In addition, we construct a mixed degradation dataset with synthetic captions for DA-CLIP training. Our approach advances state-of-the-art performance on both degradation-specific and unified image restoration tasks, showing a promising direction of prompting image restoration with large-scale pretrained vision-language models. Our code is available at https://github.com/Algolzw/daclip-uir.

</details>

---

## 11. Frozen Transformers in Language Models Are Effective Visual Encoder Layers

- [ ] Frozen Transformers in Language Models Are Effective Visual Encoder Layers | https://iclr.cc/virtual/2024/poster/17627

- **Link**: https://iclr.cc/virtual/2024/poster/17627

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper reveals that large language models (LLMs), despite being trained solely on text data, are surprisingly}strong encoders for purely visual tasks in the absence of language. Even more intriguingly, this can be achieved by a simple yet previously overlooked strategy -- employing a frozen transformer block from pre-trained LLMs as a constituent encoder layer to directly process visual tokens. Our work pushes the boundaries of leveraging LLMs for computer vision tasks, significantly departing from conventional practices that typically necessitate a multi-modal vision-language setup with associated language prompts, inputs, or outputs. We demonstrate that our approach consistently enhances performance across a diverse range of tasks} encompassing pure 2D or 3D visual recognition tasks (e.g., image and point cloud classification), temporal modeling tasks (e.g., action recognition), non-semantic tasks (e.g., motion forecasting), and multi-modal tasks (e.g., 2D/3D visual question answering and image-text retrieval). Such improvements are a general phenomenon, applicable to various types of LLMs (e.g., LLaMA and OPT) and different LLM transformer blocks. We additionally propose the information filtering hypothesis to explain the effectiveness of pre-trained LLMs in visual encoding -- the pre-trained LLM transformer blocks discern informative visual tokens and further amplify their effect. This hypothesis is empirically supported by the observation that the feature activation, after training with LLM transformer blocks, exhibits a stronger focus on relevant regions. We hope that our work inspires new perspectives on utilizing LLMs and deepening our understanding of their underlying mechanisms.

</details>

---

## 12. Measuring Vision-Language STEM Skills of Neural Models

- [ ] Measuring Vision-Language STEM Skills of Neural Models | https://iclr.cc/virtual/2024/poster/17631

- **Link**: https://iclr.cc/virtual/2024/poster/17631

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a new challenge to test the STEM skills of neural models. The problems in the real world often require solutions, combining knowledge from STEM (science, technology, engineering, and math). Unlike existing datasets, our dataset requires the understanding of multimodal vision-language information of STEM. Our dataset features one of the largest and most comprehensive datasets for the challenge. It includes $448$ skills and $1,073,146$ questions spanning all STEM subjects. Compared to existing datasets that often focus on examining expert-level ability, our dataset includes fundamental skills and questions designed based on the K-12 curriculum. We also add state-of-the-art foundation models such as CLIP and GPT-3.5-Turbo to our benchmark. Results show that the recent model advances only help master a very limited number of lower grade-level skills ($2.5$% in the third grade) in our dataset. In fact, these models are still well below (averaging $54.7$%) the performance of elementary students, not to mention near expert-level performance. To understand and increase the performance on our dataset, we teach the models on a training split of our dataset.Even though we observe improved performance, the model performance remains relatively low compared to average elementary students. To solve STEM problems, we will need novel algorithmic innovations from the community.

</details>

---

## 13. Learning Interactive Real-World Simulators

- [ ] Learning Interactive Real-World Simulators | https://iclr.cc/virtual/2024/poster/17660

- **Link**: https://iclr.cc/virtual/2024/poster/17660

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generative models trained on internet data have revolutionized how text, image, and video content can be created. Perhaps the next milestone for generative models is to simulate realistic experience in response to actions taken by humans, robots, and other interactive agents. Applications of a real-world simulator range from controllable content creation in games and movies, to training embodied agents purely in simulation that can be directly deployed in the real world. We explore the possibility of learning a universal simulator (UniSim) of real-world interaction through generative modeling. We first make the important observation that natural datasets available for learning a real-world simulator are often rich along different axes (e.g., abundant objects in image data, densely sampled actions in robotics data, and diverse movements in navigation data). With careful orchestration of diverse datasets, each providing a different aspect of the overall experience, UniSim can emulate how humans and agents interact with the world by simulating the visual outcome of both high-level instructions such as “open the drawer” and low-level controls such as “move by x,y” from otherwise static scenes and objects. There are numerous use cases for such a real-world simulator. As an example, we use UniSim to train both high-level vision-language planners and low-level reinforcement learning policies, each of which exhibit zero-shot real-world transfer after training purely in a learned real-world simulator. We also show that other types of intelligence such as video captioning models can benefit from training with simulated experience in UniSim, opening up even wider applications.

</details>

---

## 14. Faithful Vision-Language Interpretation via Concept Bottleneck Models

- [ ] Faithful Vision-Language Interpretation via Concept Bottleneck Models | https://iclr.cc/virtual/2024/poster/17682

- **Link**: https://iclr.cc/virtual/2024/poster/17682

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The demand for transparency in healthcare and finance has led to interpretable machine learning (IML) models, notably the concept bottleneck models (CBMs), valued for their potential in performance and insights into deep neural networks. However, CBM's reliance on manually annotated data poses challenges. Label-free CBMs have emerged to address this, but they remain unstable, affecting their faithfulness as explanatory tools. To address this issue of inherent instability, we introduce a formal definition for an alternative concept called the Faithful Vision-Language Concept (FVLC) model. We present a methodology for constructing an FVLC that satisfies four critical properties. Our extensive experiments on four benchmark datasets using Label-free CBM model architectures demonstrate that our FVLC outperforms other baselines regarding stability against input and concept set perturbations. Our approach incurs minimal accuracy degradation compared to the vanilla CBM, making it a promising solution for reliable and faithful model interpretation.

</details>

---

## 15. Octavius: Mitigating Task Interference in MLLMs via LoRA-MoE

- [ ] Octavius: Mitigating Task Interference in MLLMs via LoRA-MoE | https://iclr.cc/virtual/2024/poster/17691

- **Link**: https://iclr.cc/virtual/2024/poster/17691

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have demonstrated Large Language Models (LLMs) can extend their zero-shot generalization capabilities to multimodal learning through instruction tuning. As more modalities and downstream tasks are introduced, negative conflicts and interference may have a worse impact on performance. While this phenomenon has been overlooked in previous work, we propose a novel and extensible framework, called Octavius, for comprehensive studies and experimentation on multimodal learning with Multimodal Large Language Models (MLLMs). Specifically, to mitigate the interference, we combine the concept of Mixture-of-Experts (MoE) with LoRA and design a multimodal LoRA-MoE decoder for task- and modality-specific learning. To the best of our knowledge, we are one of the pioneering efforts to introduce MoE into MLLMs to address this problem. The experimental results (about 20% improvement) have shown the effectiveness and versatility of our design in various 2D and 3D downstream tasks. Code and corresponding dataset will be availablesoon.

</details>

---

## 16. Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models

- [ ] Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models | https://iclr.cc/virtual/2024/poster/17767

- **Link**: https://iclr.cc/virtual/2024/poster/17767

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce new jailbreak attacks on vision language models (VLMs), which use aligned LLMs and are resilient to text-only jailbreak attacks. Specifically, we develop cross-modality attacks on alignment where we pair adversarial images going through the vision encoder with textual prompts to break the alignment of the language model. Our attacks employ a novel compositional strategy that combines an image, adversarially targeted towards toxic embeddings, with generic prompts to accomplish the jailbreak. Thus, the LLM draws the context to answer the generic prompt from the adversarial image. The generation of benign-appearing adversarial images leverages a novel embedding-space-based methodology, operating with no access to the LLM model. Instead, the attacks require access only to the vision encoder and utilize one of our four embedding space targeting strategies. By not requiring access to the LLM, the attacks lower the entry barrier for attackers, particularly when vision encoders such as CLIP are embedded in closed-source LLMs. The attacks achieve a high success rate across different VLMs, highlighting the risk of cross-modality alignment vulnerabilities, and the need for new alignment approaches for multi-modal models.

</details>

---

## 17. Analyzing and Mitigating Object Hallucination in Large Vision-Language Models

- [ ] Analyzing and Mitigating Object Hallucination in Large Vision-Language Models | https://iclr.cc/virtual/2024/poster/17813

- **Link**: https://iclr.cc/virtual/2024/poster/17813

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have shown remarkable abilities in understanding visual information with human languages. However, LVLMs still suffer from object hallucination, which is the problem of generating descriptions that include objects that do not actually exist in the images. This can negatively impact many vision-language tasks, such as visual summarization and reasoning. To address this issue, we propose a simple yet powerful algorithm, LVLM Hallucination Revisor (LURE), to post-hoc rectify object hallucination in LVLMs by reconstructing less hallucinatory descriptions. LURE is grounded in a rigorous statistical analysis of the key factors underlying object hallucination, including co-occurrence (the frequent appearance of certain objects alongside others in images), uncertainty (objects with higher uncertainty during LVLM decoding), and object position (hallucination often appears in the later part of the generated text). LURE can also be seamlessly integrated with any LVLMs. We evaluate LURE on six open-source LVLMs and found it outperforms the previous best approach in both general object hallucination evaluation metrics, GPT, and human evaluations.

</details>

---

## 18. An Image Is Worth 1000 Lies: Transferability of Adversarial Images across Prompts on Vision-Language Models

- [ ] An Image Is Worth 1000 Lies: Transferability of Adversarial Images across Prompts on Vision-Language Models | https://iclr.cc/virtual/2024/poster/17850

- **Link**: https://iclr.cc/virtual/2024/poster/17850

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Different from traditional task-specific vision models, recent large VLMs can readily adapt to different vision tasks by simply using different textual instructions, i.e., prompts. However, a well-known concern about traditional task-specific vision models is that they can be misled by imperceptible adversarial perturbations. Furthermore, the concern is exacerbated by the phenomenon that the same adversarial perturbations can fool different task-specific models. Given that VLMs rely on prompts to adapt to different tasks, an intriguing question emerges: Can a single adversarial image mislead all predictions of VLMs when a thousand different prompts are given? This question essentially introduces a novel perspective on adversarial transferability: cross-prompt adversarial transferability. In this work, we propose the Cross-Prompt Attack (CroPA). This proposed method updates the visual adversarial perturbation with learnable textual prompts, which are designed to counteract the misleading effects of the adversarial image. By doing this, CroPA significantly improves the transferability of adversarial examples across prompts. Extensive experiments are conducted to verify the strong cross-prompt adversarial transferability of CroPA with prevalent VLMs including Flamingo, BLIP-2, and InstructBLIP in various different tasks.

</details>

---

## 19. Listen, Think, and Understand

- [ ] Listen, Think, and Understand | https://iclr.cc/virtual/2024/poster/17867

- **Link**: https://iclr.cc/virtual/2024/poster/17867

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The ability of artificial intelligence (AI) systems to perceive and comprehend audio signals is crucial for many applications. Although significant progress has been made in this area since the development of AudioSet, most existing models are designed to map audio inputs to pre-defined, discrete sound label sets. In contrast, humans possess the ability to not only classify sounds into general categories, but also to listen to the finer details of the sounds, explain the reason for the predictions, think about what the sound infers, and understand the scene and what action needs to be taken, if any. Such capabilities beyond perception are not yet present in existing audio models. On the other hand, modern large language models (LLMs) exhibit emerging reasoning ability but they lack audio perception capabilities. Therefore, we ask the question: can we build a model that has both audio perception and reasoning ability? In this paper, we propose a new audio foundation model, called LTU (Listen, Think, and Understand). To train LTU, we created a new OpenAQA-5M dataset consisting of 1.9 million closed-ended and 3.7 million open-ended, diverse (audio, question, answer) tuples, and have used an autoregressive training framework with a perception-to-understanding curriculum. LTU demonstrates strong performance and generalization ability on conventional audio tasks such as classification and captioning. More importantly, it exhibits emerging audio reasoning and comprehension abilities that are absent in existing audio models. To the best of our knowledge, LTU is the first multimodal large language model that focuses on general audio (rather than just speech) understanding.

</details>

---

## 20. BrainSCUBA: Fine-Grained Natural Language Captions of Visual Cortex Selectivity

- [ ] BrainSCUBA: Fine-Grained Natural Language Captions of Visual Cortex Selectivity | https://iclr.cc/virtual/2024/poster/17893

- **Link**: https://iclr.cc/virtual/2024/poster/17893

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding the functional organization of higher visual cortex is a central focus in neuroscience. Past studies have primarily mapped the visual and semantic selectivity of neural populations using hand-selected stimuli, which may potentially bias results towards pre-existing hypotheses of visual cortex functionality. Moving beyond conventional approaches, we introduce a data-driven method that generates natural language descriptions for images predicted to maximally activate individual voxels of interest. Our method -- Semantic Captioning Using Brain Alignments ("BrainSCUBA") -- builds upon the rich embedding space learned by a contrastive vision-language model and utilizes a pre-trained large language model to generate interpretable captions. We validate our method through fine-grained voxel-level captioning across higher-order visual regions. We further perform text-conditioned image synthesis with the captions, and show that our images are semantically coherent and yield high predicted activations. Finally, to demonstrate how our method enables scientific discovery, we perform exploratory investigations on the distribution of "person" representations in the brain, and discover fine-grained semantic selectivity in body-selective areas. Unlike earlier studies that decode text, our method derives voxel-wise captions of semantic selectivity . Our results show that BrainSCUBA is a promising means for understanding functional preferences in the brain, and provides motivation for further hypothesis-driven investigation of visual cortex.

</details>

---

## 21. ViLMA: A Zero-Shot Benchmark for Linguistic and Temporal Grounding in Video-Language Models

- [ ] ViLMA: A Zero-Shot Benchmark for Linguistic and Temporal Grounding in Video-Language Models | https://iclr.cc/virtual/2024/poster/17922

- **Link**: https://iclr.cc/virtual/2024/poster/17922

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the ever-increasing popularity of pretrained Video-Language Models (VidLMs), there is a pressing need to develop robust evaluation methodologies that delve deeper into their visio-linguistic capabilities. To address this challenge, we present ViLMA (Video Language Model Assessment), a task-agnostic benchmark that places the assessment of fine-grained capabilities of these models on a firm footing. Task-based evaluations, while valuable, fail to capture the complexities and specific temporal aspects of moving images that VidLMs need to process. Through carefully curated counterfactuals, ViLMA offers a controlled evaluation suite that sheds light on the true potential of these models, as well as their performance gaps compared to human-level understanding. ViLMA also includes proficiency tests, which assess basic capabilities deemed essential to solving the main counterfactual tests. We show that current VidLMs’ grounding abilities are no better than those of vision-language models which use static images. This is especially striking once the performance on proficiency tests is factored in. Our benchmark serves as a catalyst for future research on VidLMs, helping to highlight areas that still need to be explored.

</details>

---

## 22. Grounding Multimodal Large Language Models to the World

- [ ] Grounding Multimodal Large Language Models to the World | https://iclr.cc/virtual/2024/poster/17934

- **Link**: https://iclr.cc/virtual/2024/poster/17934

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent text spans (i.e., referring expressions and noun phrases) as links in Markdown, i.e., text span , where object descriptions are sequences of location tokens. To train the model, we construct a large-scale dataset about grounded image-text pairs (GrIT) together with multimodal corpora. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability to downstream applications, while maintaining the conventional capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning). Kosmos-2 is evaluated on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This study sheds a light on the big convergence of language, multimodal perception, and world modeling, which is a key step toward artificial general intelligence. Code can be found in https://aka.ms/kosmos-2 .

</details>

---

## 23. Bridging Vision and Language Spaces with Assignment Prediction

- [ ] Bridging Vision and Language Spaces with Assignment Prediction | https://iclr.cc/virtual/2024/poster/17937

- **Link**: https://iclr.cc/virtual/2024/poster/17937

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces VLAP, a novel approach that bridges pretrained vision models and large language models (LLMs) to make frozen LLMs understand the visual world. VLAP transforms the embedding space of pretrained vision models into the LLMs' word embedding space using a single linear layer for efficient and general-purpose visual and language understanding. Specifically, we harness well-established word embeddings to bridge two modality embedding spaces. The visual and text representations are simultaneously assigned to a set of word embeddings within pretrained LLMs by formulating the assigning procedure as an optimal transport problem. We predict the assignment of one modality from the representation of another modality data, enforcing consistent assignments for paired multimodal data. This allows vision and language representations to contain the same information, grounding the frozen LLMs' word embedding space in visual data. Moreover, a robust semantic taxonomy of LLMs can be preserved with visual data since the LLMs interpret and reason linguistic information from correlations between word embeddings. Experimental results show that VLAP achieves substantial improvements over the previous linear transformation-based approaches across a range of vision-language tasks, including image captioning, visual question answering, and cross-modal retrieval. We also demonstrate the learned visual representations hold a semantic taxonomy of LLMs, making visual semantic arithmetic possible.

</details>

---

## 24. Vision-Language Foundation Models as Effective Robot Imitators

- [ ] Vision-Language Foundation Models as Effective Robot Imitators | https://iclr.cc/virtual/2024/poster/17943

- **Link**: https://iclr.cc/virtual/2024/poster/17943

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in vision language foundation models has shown their ability to understand multimodal data and resolve complicated vision language tasks, including robotics manipulation. We seek a straightforward way of making use of existing vision-language models (VLMs) with simple fine-tuning on robotics data.To this end, we derive a simple and novel vision-language manipulation framework, dubbed RoboFlamingo, built upon the open-source VLMs, OpenFlamingo. Unlike prior works, RoboFlamingo utilizes pre-trained VLMs for single-step vision-language comprehension, models sequential history information with an explicit policy head, and is slightly fine-tuned by imitation learning only on language-conditioned manipulation datasets. Such a decomposition provides RoboFlamingo the flexibility for open-loop control and deployment on low-performance platforms. By exceeding the state-of-the-art performance with a large margin on the tested benchmark, we show RoboFlamingo can be an effective and competitive alternative to adapt VLMs to robot control.Our extensive experimental results also reveal several interesting conclusions regarding the behavior of different pre-trained VLMs on manipulation tasks. We believe RoboFlamingo has the potential to be a cost-effective and easy-to-use solution for robotics manipulation, empowering everyone with the ability to fine-tune their own robotics policy. Our code will be made public upon acceptance.

</details>

---

## 25. Towards Unified Multi-Modal Personalization: Large Vision-Language Models for Generative Recommendation and Beyond

- [ ] Towards Unified Multi-Modal Personalization: Large Vision-Language Models for Generative Recommendation and Beyond | https://iclr.cc/virtual/2024/poster/17967

- **Link**: https://iclr.cc/virtual/2024/poster/17967

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Developing a universal model that can effectively harness heterogeneous resources and respond to a wide range of personalized needs has been a longstanding community aspiration. Our daily choices, especially in domains like fashion and retail, are substantially shaped by multi-modal data, such as pictures and textual descriptions. These modalities not only offer intuitive guidance but also cater to personalized user preferences. However, the predominant personalization approaches mainly focus on ID or text-based recommendation problems, failing to comprehend the information spanning various tasks or modalities. In this paper, our goal is to establish a Unified paradigm for Multi-modal Personalization systems (UniMP), which effectively leverages multi-modal data while eliminating the complexities associated with task- and modality-specific customization. We argue that the advancements in foundational generative modeling have provided the flexibility and effectiveness necessary to achieve the objective. In light of this, we develop a generic and extensible personalization generative framework, that can handle a wide range of personalized needs including item recommendation, product search, preference prediction, explanation generation, and further user-guided image generation. Our methodology enhances the capabilities of foundational language models for personalized tasks by seamlessly ingesting interleaved cross-modal user history information, ensuring a more precise and customized experience for users. To train and evaluate the proposed multi-modal personalized tasks, we also introduce a novel and comprehensive benchmark covering a variety of user requirements. Our experiments on the real-world benchmark showcase the model's potential, outperforming competitive methods specialized for each task.

</details>

---

## 26. Large Language Models as Automated Aligners for  benchmarking  Vision-Language Models

- [ ] Large Language Models as Automated Aligners for  benchmarking  Vision-Language Models | https://iclr.cc/virtual/2024/poster/17968

- **Link**: https://iclr.cc/virtual/2024/poster/17968

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the advancements in Large Language Models (LLMs), Vision-Language Models (VLMs) have reached a new level of sophistication, showing notable competence in executing intricate cognition and reasoning tasks. However, existing evaluation benchmarks, primarily relying on rigid, hand-crafted datasets to measure task-specific performance, face significant limitations in assessing the alignment of these increasingly anthropomorphic models with human intelligence. In this work, we address the limitations via Auto-Bench, which delves into exploring LLMs as proficient aligners, measuring the alignment between VLMs and human intelligence and value through automatic data curation and assessment. Specifically, for data curation, Auto-Bench utilizes LLMs (e.g., GPT-4) to automatically generate a vast set of question-answer-reasoning triplets via prompting on visual symbolic representations (e.g., captions, object locations, instance relationships, and etc. The curated data closely matches human intent, owing to the extensive world knowledge embedded in LLMs. Through this pipeline, a total of 28.5K human-verified and 3,504K unfiltered question-answer-reasoning triplets have been curated, covering 4 primary abilities and 16 sub-abilities. We subsequently engage LLMs like GPT-3.5 to serve as judges, implementing the quantitative and qualitative automated assessments to facilitate a comprehensive evaluation of VLMs. Our validation results reveal that LLMs are proficient in both evaluation data curation and model assessment, achieving an average agreement rate of 85%. We envision Auto-Bench as a flexible, scalable, and comprehensive benchmark for evaluating the evolving sophisticated VLMs.

</details>

---

## 27. Leveraging Unpaired Data for Vision-Language Generative Models via Cycle Consistency

- [ ] Leveraging Unpaired Data for Vision-Language Generative Models via Cycle Consistency | https://iclr.cc/virtual/2024/poster/17977

- **Link**: https://iclr.cc/virtual/2024/poster/17977

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current vision-language generative models rely on expansive corpora of $\textit{paired}$ image-text data to attain optimal performance and generalization capabilities. However, automatically collecting such data (e.g. via large-scale web scraping) leads to low quality and poor image-text correlation, while human annotation is more accurate but requires significant manual effort and expense. We introduce $\textbf{ITIT}$ ($\textbf{I}$n$\textbf{T}$egrating $\textbf{I}$mage $\textbf{T}$ext): an innovative training paradigm grounded in the concept of cycle consistency which allows vision-language training on $\textit{unpaired}$ image and text data. ITIT is comprised of a joint image-text encoder with disjoint image and text decoders that enable bidirectional image-to-text and text-to-image generation in a single framework. During training, ITIT leverages a small set of paired image-text data to ensure its output matches the input reasonably well in both directions. Simultaneously, the model is also trained on much larger datasets containing only images or texts. This is achieved by enforcing cycle consistency between the original unpaired samples and the cycle-generated counterparts. For instance, it generates a caption for a given input image and then uses the caption to create an output image, and enforces similarity between the input and output images. Our experiments show that ITIT with unpaired datasets exhibits similar scaling behavior as using high-quality paired data. We demonstrate image generation and captioning performance on par with state-of-the-art text-to-image and image-to-text models with orders of magnitude fewer (only 3M) paired image-text data. Code will be released at https://github.com/LTH14/itit.

</details>

---

## 28. Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models

- [ ] Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models | https://iclr.cc/virtual/2024/poster/17984

- **Link**: https://iclr.cc/virtual/2024/poster/17984

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

One fascinating aspect of pre-trained vision-language models (VLMs) learning under language supervision is their impressive zero-shot generalization capability.However, this ability is hindered by distribution shifts between the training and testing data.Previous test time adaptation (TTA) methods for VLMs in zero-shot classification rely on minimizing the entropy of model outputs, tending to be stuck in incorrect model predictions.In this work, we propose TTA with feedback to rectify the model output and prevent the model from becoming blindly confident.Specifically, a CLIP model is adopted as the reward model during TTA and provides feedback for the VLM.Given a single test sample,the VLM is forced to maximize the CLIP reward between the input and sampled results from the VLM output distribution.The proposed \textit{reinforcement learning with CLIP feedback~(RLCF)} framework is highly flexible and universal.Beyond the classification task, with task-specific sampling strategies and a proper reward baseline choice, RLCF can be easily extended to not only discrimination tasks like retrieval but also generalization tasks like image captioning,improving the zero-shot generalization capacity of VLMs.According to the characteristics of these VL tasks, we build different fully TTA pipelines with RLCF to improve the zero-shot generalization ability of various VLMs.Extensive experiments along with promisingempirical results demonstrate the effectiveness of RLCF.The code is available at https://github.com/mzhaoshuai/RLCF.

</details>

---

## 29. C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion

- [ ] C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion | https://iclr.cc/virtual/2024/poster/17996

- **Link**: https://iclr.cc/virtual/2024/poster/17996

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In deep learning, test-time adaptation has gained attention as a method for model fine-tuning without the need for labeled data. A prime exemplification is the recently proposed test-time prompt tuning for large-scale vision-language models such as CLIP. Unfortunately, these prompts have been mainly developed to improve accuracy, overlooking the importance of calibration, which is a crucial aspect for quantifying prediction uncertainty. However, traditional calibration methods rely on substantial amounts of labeled data, making them impractical for test-time scenarios. To this end, this paper explores calibration during test-time prompt tuning by leveraging the inherent properties of CLIP. Through a series of observations, we find that the prompt choice significantly affects the calibration in CLIP, where the prompts leading to higher text feature dispersion result in better-calibrated predictions. Introducing the Average Text Feature Dispersion (ATFD), we establish its relationship with calibration error and present a novel method, Calibrated Test-time Prompt Tuning (C-TPT), for optimizing prompts during test-time with enhanced calibration. Through extensive experiments on different CLIP architectures and datasets, we show that C-TPT can effectively improve the calibration of test-time prompt tuning without needing labeled data. The code is publicly accessible at https://github.com/hee-suk-yoon/C-TPT.

</details>

---

## 30. Language-Informed Visual Concept Learning

- [ ] Language-Informed Visual Concept Learning | https://iclr.cc/virtual/2024/poster/18001

- **Link**: https://iclr.cc/virtual/2024/poster/18001

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Our understanding of the visual world is centered around various concept axes, characterizing different aspects of visual entities. While different concept axes can be easily specified by language, e.g., color, the exact visual nuances along each axis often exceed the limitations of linguistic articulations, e.g., a particular style of painting. In this work, our goal is to learn a language-informed visual concept representation, by simply distilling large pre-trained vision-language models. Specifically, we train a set of concept encoders to encode the information pertinent to a set of language-informed concept axes, with an objective of reproducing the input image through a pre-trained Text-to-Image (T2I) model. To encourage better disentanglement of different concept encoders, we anchor the concept embeddings to a set of text embeddings obtained from a pre-trained Visual Question Answering (VQA) model. At inference time, the model extracts concept embeddings along various axes from new test images, which can be remixed to generate images with novel compositions of visual concepts. With a lightweight test-time finetuning procedure, it can also generalize to novel concepts unseen at training.Project page at https://cs.stanford.edu/~yzzhang/projects/concept-axes.

</details>

---

## 31. Look, Remember and Reason: Grounded Reasoning in Videos with Language Models

- [ ] Look, Remember and Reason: Grounded Reasoning in Videos with Language Models | https://iclr.cc/virtual/2024/poster/18014

- **Link**: https://iclr.cc/virtual/2024/poster/18014

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal language models (LM) have recently shown promising performance in high-level reasoning tasks on videos. However, existing methods still fall short in tasks like causal or  compositional spatiotemporal reasoning over actions, in which model predictions need to be grounded in fine-grained low-level details, such as object motions and object interactions.In this work, we propose training an LM end-to-end on low-level surrogate tasks, including object detection, re-identification, and tracking, to endow the  model with the required low-level visual capabilities. We show that a two-stream video encoder with spatiotemporal attention is effective at capturing the required static and motion-based cues in the video. By leveraging the LM's ability to perform the low-level surrogate tasks,  we can cast reasoning in videos as the three-step process of Look, Remember, Reason , wherein visual information is extracted using low-level visual skills step-by-step and then integrated to arrive at a final answer. We demonstrate the effectiveness of our framework on diverse visual reasoning tasks from the ACRE, CATER, Something-Else and STAR datasets. Our approach is trainable end-to-end and surpasses state-of-the-art task-specific methods across these tasks by a large margin.

</details>

---

## 32. ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models

- [ ] ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models | https://iclr.cc/virtual/2024/poster/18067

- **Link**: https://iclr.cc/virtual/2024/poster/18067

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) can understand the world comprehensively by integrating rich information from different modalities, achieving remarkable performance improvements on various multimodal downstream tasks. However, deploying LVLMs is often problematic due to their massive computational/energy costs and carbon consumption, making it infeasible to adopt conventional iterative global pruning, which is costly due to computing the Hessian matrix of the entire large model for sparsification. Alternatively, several studies have recently proposed layer-wise pruning approaches to avoid the expensive computation of global pruning and efficiently compress model weights according to their importance within a layer. However, these methods often suffer from suboptimal model compression due to their lack of a global perspective. To address this limitation in recent efficient pruning methods for large models, we propose Efficient Coarse-to-Fine Layer-Wise Pruning (ECoFLaP), a two-stage coarse-to-fine weight pruning approach for LVLMs. We first determine the sparsity ratios of different layers or blocks by leveraging the global importance score, which is efficiently computed based on the zeroth-order approximation of the global model gradients. Then, the multimodal model performs layer-wise unstructured weight pruning. We validate our proposed method across various multi-modal and single-modal models and datasets, demonstrating significant performance improvements over prevalent pruning techniques in the high-sparsity regime.

</details>

---

## 33. InstructDET: Diversifying Referring Object Detection with Generalized Instructions

- [ ] InstructDET: Diversifying Referring Object Detection with Generalized Instructions | https://iclr.cc/virtual/2024/poster/18082

- **Link**: https://iclr.cc/virtual/2024/poster/18082

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose InstructDET, a data-centric method for referring object detection (ROD) that localizes target objects based on user instructions. While deriving from referring expressions (REC), the instructions we leverage are greatly diversified to encompass common user intentions related to object detection. For one image, we produce tremendous instructions that refer to every single object and different combinations of multiple objects. Each instruction and its corresponding object bounding boxes (bbxs) constitute one training data pair. In order to encompass common detection expressions, we involve emerging vision-language model (VLM) and large language model (LLM) to generate instructions guided by text prompts and object bbxs, as the generalizations of foundation models are effective to produce human-like expressions (e.g., describing object property, category, and relationship). We name our constructed dataset as InDET. It contains images, bbxs and generalized instructions that are from foundation models. Our InDET is developed from existing REC datasets and object detection datasets, with the expanding potential that any image with object bbxs can be incorporated through using our InstructDET method. By using our InDET dataset, we show that a conventional ROD model surpasses existing methods on standard REC datasets and our InDET test set. Our data-centric method InstructDET, with automatic data expansion by leveraging foundation models, directs a promising field that ROD can be greatly diversified to execute common object detection instructions.

</details>

---

## 34. Kosmos-G: Generating Images in Context with Multimodal Large Language Models

- [ ] Kosmos-G: Generating Images in Context with Multimodal Large Language Models | https://iclr.cc/virtual/2024/poster/18091

- **Link**: https://iclr.cc/virtual/2024/poster/18091

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in subject-driven image generation have made significant strides. However, current methods still fall short in diverse application scenarios, as they require test-time tuning and cannot accept interleaved multi-image and text input. These limitations keep them far from the ultimate goal of "image as a foreign language in image generation." This paper presents Kosmos-G, a model that leverages the advanced multimodal perception capabilities of Multimodal Large Language Models (MLLMs) to tackle the aforementioned challenge. Our approach aligns the output space of MLLM with CLIP using the textual modality as an anchor and performs compositional instruction tuning on curated data. Kosmos-G demonstrates an impressive capability of zero-shot subject-driven generation with interleaved multi-image and text input. Notably, the score distillation instruction tuning requires no modifications to the image decoder. This allows for a seamless substitution of CLIP and effortless integration with a myriad of U-Net techniques ranging from fine-grained controls to personalized image decoder variants. We posit Kosmos-G as an initial attempt towards the goal of "image as a foreign language in image generation."

</details>

---

## 35. Bongard-OpenWorld: Few-Shot Reasoning for Free-form Visual Concepts in the Real World

- [ ] Bongard-OpenWorld: Few-Shot Reasoning for Free-form Visual Concepts in the Real World | https://iclr.cc/virtual/2024/poster/18093

- **Link**: https://iclr.cc/virtual/2024/poster/18093

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Bongard-OpenWorld, a new benchmark for evaluating real-world few-shot reasoning for machine vision. It originates from the classical Bongard Problems (BPs): Given two sets of images (positive and negative), the model needs to identify the set that query images belong to by inducing the visual concepts, which is exclusively depicted by images from the positive set. Our benchmark inherits the few-shot concept induction of the original BPs while adding the two novel layers of challenge: 1) open-world free-form concepts, as the visual concepts in Bongard-OpenWorld are unique compositions of terms from an open vocabulary, ranging from object categories to abstract visual attributes and commonsense factual knowledge; 2)  real-world images, as opposed to the synthetic diagrams used by many counterparts. In our exploration, Bongard-OpenWorld already imposes a significant challenge to current few-shot reasoning algorithms. We further investigate to which extent the recently introduced Large Language Models (LLMs) and Vision-Language Models (VLMs) can solve our task, by directly probing VLMs, and combining VLMs and LLMs in an interactive reasoning scheme. We even conceived a neuro-symbolic reasoning approach that reconciles LLMs & VLMs with logical reasoning to emulate the human problem-solving process for Bongard Problems. However, none of these approaches manage to close the human-machine gap, as the best learner achieves 64% accuracy while human participants easily reach 91%. We hope Bongard-OpenWorld can help us better understand the limitations of current visual intelligence and facilitate future research on visual agents with stronger few-shot visual reasoning capabilities.

</details>

---

## 36. Follow-Up Differential Descriptions: Language Models Resolve Ambiguities for Image Classification

- [ ] Follow-Up Differential Descriptions: Language Models Resolve Ambiguities for Image Classification | https://iclr.cc/virtual/2024/poster/18158

- **Link**: https://iclr.cc/virtual/2024/poster/18158

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A promising approach for improving the performance of vision-language models like CLIP for image classification is to extend the class descriptions (i.e., prompts) with related attributes, e.g., using brown sparrow instead of sparrow. However, current zero-shot methods select a subset of attributes regardless of commonalities between the target classes, potentially providing no useful information that would have helped to distinguish between them. For instance, they may use color instead of bill shape to distinguish between sparrows and wrens, which are both brown. We propose Follow-up Differential Descriptions (FuDD), a zero-shot approach that tailors the class descriptions to each dataset and leads to additional attributes that better differentiate the target classes. FuDD first identifies the ambiguous classes for each image, and then uses a Large Language Model (LLM) to generate new class descriptions that differentiate between them. The new class descriptions resolve the initial ambiguity and help predict the correct label. In our experiments, FuDD consistently outperforms generic description ensembles and naive LLM-generated descriptions on 12 datasets. We show that differential descriptions are an effective tool to resolve class ambiguities, which otherwise significantly degrade the performance. We also show that high quality natural language class descriptions produced by FuDD result in comparable performance to few-shot adaptation methods.

</details>

---

## 37. Improved Probabilistic Image-Text Representations

- [ ] Improved Probabilistic Image-Text Representations | https://iclr.cc/virtual/2024/poster/18166

- **Link**: https://iclr.cc/virtual/2024/poster/18166

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image-Text Matching (ITM) task, a fundamental vision-language (VL) task, suffers from the inherent ambiguity arising from multiplicity and imperfect annotations. Deterministic functions are not sufficiently powerful to capture ambiguity, prompting the exploration of probabilistic embeddings to tackle the challenge. However, the existing probabilistic ITM approach encounters two key shortcomings; the burden of heavy computations due to the Monte Carlo approximation, and the loss saturation issue in the face of abundant false negatives. To overcome the issues, this paper presents an improved Probabilistic Cross-Modal Embeddings (named PCME++) by introducing a new probabilistic distance with a closed-form solution. In addition, two optimization techniques are proposed to enhance PCME++ further: first, the incorporation of pseudo-positives to prevent the negative effect under massive false negatives; second, mixed sample data augmentation for probabilistic matching. Experimental results on MS-COCO Caption and two extended benchmarks, CxC and ECCV Caption, demonstrate the effectiveness of PCME++ compared to state-of-the-art ITM methods. The robustness of PCME++ is also evaluated under noisy image-text correspondences. In addition, the potential applicability of PCME++ in automatic prompt-filtering for zero-shot classification is shown. The code is available at https://github.com/naver-ai/pcmepp.

</details>

---

## 38. UniAdapter: Unified Parameter-Efficient Transfer Learning for Cross-modal Modeling

- [ ] UniAdapter: Unified Parameter-Efficient Transfer Learning for Cross-modal Modeling | https://iclr.cc/virtual/2024/poster/18198

- **Link**: https://iclr.cc/virtual/2024/poster/18198

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language pre-trained models have shown promising transferability to various downstream tasks. As the size of these foundation models and the number of downstream tasks grow, the standard full fine-tuning paradigm becomes unsustainable due to heavy computational and storage costs. This paper proposes UniAdapter, which unifies unimodal and multimodal adapters for parameter-efficient cross-modal adaptation on pre-trained vision-language models. Specifically, adapters are distributed to different modalities and their interactions, with the total number of tunable parameters reduced by partial weight sharing. The unified and knowledge-sharing design enables powerful cross-modal representations that can benefit various downstream tasks, requiring only 1.0%-2.0% tunable parameters of the pre-trained model. Extensive experiments on 7 cross-modal downstream benchmarks (including video-text retrieval, image-text retrieval, VideoQA, VQA and Caption) show that in most cases, UniAdapter not only outperforms the state-of-the-arts, but even beats the full fine-tuning strategy. Particularly, on the MSRVTT retrieval task, UniAdapter achieves 49.7% recall@1 with 2.2% model parameters, outperforming the latest competitors by 2.0%. The code and models are available at https://github.com/RERV/UniAdapter.

</details>

---

## 39. Multimodal Web Navigation with Instruction-Finetuned Foundation Models

- [ ] Multimodal Web Navigation with Instruction-Finetuned Foundation Models | https://iclr.cc/virtual/2024/poster/18215

- **Link**: https://iclr.cc/virtual/2024/poster/18215

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The progress of autonomous web navigation has been hindered by the dependence on billions of exploratory interactions via online reinforcement learning, and domain-specific model designs that make it difficult to leverage generalization from rich out-of-domain data.In this work, we study data-driven offline training for web agents with vision-language foundation models.We propose an instruction-following multimodal agent, WebGUM, that observes both webpage screenshots and HTML pages and outputs web navigation actions, such as click and type.WebGUM is trained by jointly finetuning an instruction-finetuned language model and a vision encoder with temporal and local perception on a large corpus of demonstrations.We empirically demonstrate this recipe improves the agent's ability of grounded multimodal perception, HTML comprehension, and multi-step reasoning, outperforming prior works by a significant margin. On the MiniWoB, we improve over the previous best offline methods by more than 45.8%, even outperforming online-finetuned SoTA, humans, and GPT-4-based agent. On the WebShop benchmark, our 3-billion-parameter model achieves superior performance to the existing SoTA, PaLM-540B.Furthermore, WebGUM exhibits strong positive transfer to the real-world planning tasks on the Mind2Web.We also collect 347K high-quality demonstrations using our trained models, 38 times larger than prior work, and make them available to promote future research in this direction.

</details>

---

## 40. PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis

- [ ] PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis | https://iclr.cc/virtual/2024/poster/18231

- **Link**: https://iclr.cc/virtual/2024/poster/18231

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The most advanced text-to-image (T2I) models require significant training costs (e.g., millions of GPU hours), seriously hindering the fundamental innovation for the AIGC community while increasing CO2 emissions. This paper introduces PixArt-$\alpha$, a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), reaching near-commercial application standards. Additionally, it supports high-resolution image synthesis up to 1024px resolution with low training cost, as shown in Figure 1 and 2. To achieve this goal, three core designs are proposed: (1) Training strategy decomposition: We devise three distinct training steps that separately optimize pixel dependency, text-image alignment, and image aesthetic quality; (2) Efficient T2I Transformer: We incorporate cross-attention modules into Diffusion Transformer (DiT) to inject text conditions and streamline the computation-intensive class-condition branch; (3) High-informative data: We emphasize the significance of concept density in text-image pairs and leverage a large Vision-Language model to auto-label dense pseudo-captions to assist text-image alignment learning. As a result, PixArt-$\alpha$'s training speed markedly surpasses existing large-scale T2I models, e.g., PixArt-$\alpha$ only takes 10.8% of Stable Diffusion v1.5's training time (~675 vs. ~6,250 A100 GPU days), saving nearly \\$300,000 (\\$26,000 vs. \\$320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, our training cost is merely 1%. Extensive experiments demonstrate that PixArt-$\alpha$ excels in image quality, artistry, and semantic control. We hope PixArt-$\alpha$ will provide new insights to the AIGC community and startups to accelerate building their own high-quality yet low-cost generative models from scratch.

</details>

---

## 41. LLaMA-Adapter: Efficient Fine-tuning of Large Language Models with Zero-initialized Attention

- [ ] LLaMA-Adapter: Efficient Fine-tuning of Large Language Models with Zero-initialized Attention | https://iclr.cc/virtual/2024/poster/18277

- **Link**: https://iclr.cc/virtual/2024/poster/18277

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the rising tide of large language models (LLMs), there has been a growing interest in developing general-purpose instruction-following models, e.g., ChatGPT. To this end, we present LLaMA-Adapter, a lightweight adaption method for efficient instruction tuning of LLaMA. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning. Specifically, a zero-initialized attention mechanism is proposed. It adopts a learnable zero gating to adaptively inject the instructional cues into LLaMA within self-attention layers, contributing to a stable training process and superior final performance. In this way, LLaMA-Adapter can generate high-quality responses to diverse language instructions, comparable to Alpaca with fully fine-tuned 7B parameters. Besides language commands, by incorporating an image encoder, our approach can be simply extended to a multi-modal LLM for image-conditioned instruction following, which achieves superior multi-modal reasoning capacity on several popular benchmarks (MME, MMBench, LVLM-eHub). Furthermore, we also verify the proposed zero-initialized attention mechanism for fine-tuning other pre-trained models (ViT, RoBERTa, CLIP) on traditional vision and language tasks, demonstrating the effectiveness and generalizability of our approach. Code and models are released at https://github.com/OpenGVLab/LLaMA-Adapter.

</details>

---

## 42. The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World

- [ ] The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World | https://iclr.cc/virtual/2024/poster/18312

- **Link**: https://iclr.cc/virtual/2024/poster/18312

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present the All-Seeing (AS) project: a large-scale dataset and model for recognizing and understanding everything in the open world.Using a scalable data engine that incorporates human feedback and efficient models in the loop, we create a new dataset (AS-1B) with over 1.2 billion regions annotated with semantic tags, question-answering pairs, and detailed captions. It covers a wide range of 3.5 million common and rare concepts in the real world and has 132.2 billion tokens that describe the concepts and their attributes. Leveraging this new dataset, we develop the All-Seeing model (ASM), a unified framework for panoptic visual recognition and understanding. The model is trained with open-ended language prompts and locations, which allows it to generalize to various vision and language tasks with remarkable zero-shot performance, including both region- and image-level retrieval, region recognition, captioning, and question-answering. We hope that this project can serve as a foundation for vision-language artificial general intelligence research. Code is available at https://github.com/OpenGVLab/all-seeing.

</details>

---

## 43. AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection

- [ ] AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection | https://iclr.cc/virtual/2024/poster/18318

- **Link**: https://iclr.cc/virtual/2024/poster/18318

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot anomaly detection (ZSAD) requires detection models trained using auxiliarydata to detect anomalies without any training sample in a target dataset. Itis a crucial task when training data is not accessible due to various concerns, e.g.,data privacy, yet it is challenging since the models need to generalize to anomaliesacross different domains where the appearance of foreground objects, abnormalregions, and background features, such as defects/tumors on different products/organs, can vary significantly. Recently large pre-trained vision-languagemodels (VLMs), such as CLIP, have demonstrated strong zero-shot recognitionability in various vision tasks, including anomaly detection. However, their ZSADperformance is weak since the VLMs focus more on modeling the class semanticsof the foreground objects rather than the abnormality/normality in the images. Inthis paper we introduce a novel approach, namely AnomalyCLIP, to adapt CLIPfor accurate ZSAD across different domains. The key insight of AnomalyCLIPis to learn object-agnostic text prompts that capture generic normality and abnormalityin an image regardless of its foreground objects. This allows our model tofocus on the abnormal image regions rather than the object semantics, enablinggeneralized normality and abnormality recognition on diverse types of objects.Large-scale experiments on 17 real-world anomaly detection datasets show thatAnomalyCLIP achieves superior zero-shot performance of detecting and segmentinganomalies in datasets of highly diverse class semantics from various defectinspection and medical imaging domains. Code will be made available at https://github.com/zqhang/AnomalyCLIP.

</details>

---

## 44. COSA: Concatenated Sample Pretrained Vision-Language Foundation Model

- [ ] COSA: Concatenated Sample Pretrained Vision-Language Foundation Model | https://iclr.cc/virtual/2024/poster/18341

- **Link**: https://iclr.cc/virtual/2024/poster/18341

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Due to the limited scale and quality of video-text training corpus, most  vision-language  foundation  models employ  image-text datasets for pretraining and primarily focus on modeling visually semantic representations while disregarding temporal semantic representations and correlations. To address this issue, we propose COSA, a COncatenated SAmple pretrained vision-language foundation model. COSA can jointly model visual contents and event-level temporal cues using only image-text corpora.  We achieve this by sequentially concatenating multiple image-text pairs as inputs for pretraining. This transformation effectively converts existing image-text corpora into a pseudo  video-paragraph corpus, enabling richer scene transformations and explicit event-description correspondence. Extensive experiments demonstrate that COSA consistently improves performance across a broad range of semantic vision-language downstream tasks, including paragraph-to-video retrieval, text-to-video/image retrieval, video/image captioning and video QA. Notably, COSA achieves state-of-the-art results on various competitive benchmarks. Code and model are released at https://github.com/TXH-mercury/COSA.

</details>

---

## 45. Understanding prompt engineering may not require rethinking generalization

- [ ] Understanding prompt engineering may not require rethinking generalization | https://iclr.cc/virtual/2024/poster/18377

- **Link**: https://iclr.cc/virtual/2024/poster/18377

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot learning in prompted vision-language models, the practice of crafting prompts to build classifiers without an explicit training process, has achieved impressive performance in many settings. This success presents a seemingly surprising observation: these methods suffer relatively little from overfitting, i.e., when a prompt is manually engineered to achieve low error on a given training set (thus rendering the method no longer actually zero-shot), the approach still performs well on held-out test data. In this paper, we show that we can explain such performance well via recourse to classical PAC-Bayes bounds.  Specifically, we show that the discrete nature of prompts, combined with a PAC-Bayes prior given by a language model, results in generalization bounds that are remarkably tight by the standards of the literature: for instance, the generalization bound of an ImageNet classifier is often within a few percentage points of the true test error. We demonstrate empirically that this holds for existing handcrafted prompts and prompts generated through simple greedy search. Furthermore, the resulting bound is well-suited for model selection: the models with the best bound typically also have the best test performance. This work thus provides a possible justification for the widespread practice of "prompt engineering," even if it seems that such methods could potentially overfit the training data.

</details>

---

## 46. Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning

- [ ] Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning | https://iclr.cc/virtual/2024/poster/18422

- **Link**: https://iclr.cc/virtual/2024/poster/18422

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces an efficient strategy to transform Large Language Models (LLMs) into Multi-Modal Large Language Models. By conceptualizing this transformation as a domain adaptation process, \ie, transitioning from text understanding to embracing multiple modalities, we intriguingly note that, within each attention block, tuning LayerNorm suffices to yield strong performance. Moreover, when benchmarked against other tuning approaches like full parameter finetuning or LoRA, its benefits on efficiency are substantial.For example, when compared to LoRA on a 13B model scale, performance can be enhanced by an average of over 20\% across five multi-modal tasks, and meanwhile, results in a significant reduction of trainable parameters by 41.9\% and a decrease in GPU memory usage by 17.6\%. On top of this LayerNorm strategy, we showcase that selectively tuning only with conversational data can improve efficiency further. Beyond these empirical outcomes, we provide a comprehensive analysis to explore the role of LayerNorm in adapting LLMs to the multi-modal domain and improving the expressive power of the model.

</details>

---

## 47. Training Diffusion Models with Reinforcement Learning

- [ ] Training Diffusion Models with Reinforcement Learning | https://iclr.cc/virtual/2024/poster/18432

- **Link**: https://iclr.cc/virtual/2024/poster/18432

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models are a class of flexible generative models trained with an approximation to the log-likelihood objective. However, most use cases of diffusion models are not concerned with likelihoods, but instead with downstream objectives such as human-perceived image quality or drug effectiveness. In this paper, we investigate reinforcement learning methods for directly optimizing diffusion models for such objectives. We describe how posing denoising as a multi-step decision-making problem enables a class of policy gradient algorithms, which we refer to as denoising diffusion policy optimization ( DDPO), that are more effective than alternative reward-weighted likelihood approaches. Empirically, DDPO can adapt text-to-image diffusion models to objectives that are difficult to express via prompting, such as image compressibility, and those derived from human feedback, such as aesthetic quality. Finally, we show that DDPO can improve prompt-image alignment using feedback from a vision-language model without the need for additional data collection or human annotation. The project’s website can be found at http://rl-diffusion.github.io.

</details>

---

## 48. Visual Data-Type Understanding does not emerge from scaling Vision-Language Models

- [ ] Visual Data-Type Understanding does not emerge from scaling Vision-Language Models | https://iclr.cc/virtual/2024/poster/18460

- **Link**: https://iclr.cc/virtual/2024/poster/18460

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in the development of vision-language models (VLMs) are yielding remarkable success in recognizing visual semantic content, including impressive instances of compositional image understanding. Here, we introduce the novel task of Visual Data-Type Identification, a basic perceptual skill with implications for data curation (e.g., noisy data-removal from large datasets, domains pecific retrieval) and autonomous vision (e.g., distinguishing changing weather conditions from camera lens staining). We develop two datasets consisting of animal images altered across a diverse set of 27 visual data-types, spanning four broad categories. An extensive zero-shot evaluation of 39 VLMs, ranging from 100M to 80B parameters, shows a nuanced performance landscape. While VLMs are reasonably good at identifying certain stylistic data-types, such as cartoons and sketches, they struggle with simpler data-types arising from basic manipulations like image rotations or additive noise. Our findings reveal that (i) model scaling alone yields marginal gains for contrastively-trained models like CLIP, and (ii) there is a pronounced drop in performance for the largest auto-regressively trained VLMs like OpenFlamingo. This finding points to a blind spot in current frontier VLMs: they excel in recognizing semantic content but fail to acquire anunderstanding of visual data-types through scaling. By analyzing the pre-training distributions of these models and incorporating data-type information into the captions during fine-tuning, we achieve a significant enhancement in performance. By exploring this previously uncharted task, we aim to set the stage for further advancing VLMs to equip them with visual data-type understanding.

</details>

---

## 49. TiC-CLIP: Continual Training of CLIP Models

- [ ] TiC-CLIP: Continual Training of CLIP Models | https://iclr.cc/virtual/2024/poster/18576

- **Link**: https://iclr.cc/virtual/2024/poster/18576

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Keeping large foundation models up to date on latest data is inherently expensive. To avoid the prohibitive costs of constantly retraining, it is imperative to continually train these models. This problem is exacerbated by the lack of any large scale continual learning benchmarks or baselines. We introduce the first set of web-scale Time-Continual (TiC) benchmarks for training vision-language models: TiC-DataComp, TiC-YFCC, and TiC-Redcaps. TiC-DataComp, our largest dataset, contains over 12.7B timestamped image-text pairs spanning 9 years (2014-2022). We first use our benchmarks to curate various dynamic evaluations to measure temporal robustness of existing models. We show OpenAI's CLIP (trained on data up to 2020) loses $\approx 8\%$ zero-shot accuracy on our curated retrieval task from 2021-2022 compared with more recently trained models in OpenCLIP repository. We then study how to efficiently train models on time-continuous data. We demonstrate that a simple rehearsal-based approach that continues training from the last checkpoint  and replays old data reduces compute by $2.5\times$ when compared to the standard practice of retraining from scratch. Code is available at https://github.com/apple/ml-tic-clip.

</details>

---

## 50. OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views

- [ ] OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views | https://iclr.cc/virtual/2024/poster/18594

- **Link**: https://iclr.cc/virtual/2024/poster/18594

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large visual-language models (VLMs), like CLIP, enable open-set image segmentation to segment arbitrary concepts from an image in a zero-shot manner. This goes beyond the traditional closed-set assumption, i.e., where models can only segment classes from a pre-defined training set. More recently, first works on open-set segmentation in 3D scenes have appeared in the literature. These methods are heavily influenced by closed-set 3D convolutional approaches that process point clouds or polygon meshes. However, these 3D scene representations do not align well with the image-based nature of the visual-language models. Indeed, point cloud and 3D meshes typically have a lower resolution than images and the reconstructed 3D scene geometry might not project well to the underlying 2D image sequences used to compute pixel-aligned CLIP features. To address these challenges, we propose OpenNeRF which naturally operates on posed images and directly encodes the VLM features within the NeRF. This is similar in spirit to LERF, however our work shows that using pixel-wise VLM features (instead of global CLIP features) results in an overall less complex architecture without the need for additional DINO regularization. Our OpenNeRF further leverages NeRF’s ability to render novel views and extract open-set VLM features from areas that are not well observed in the initial posed images. For 3D point cloud segmentation on the Replica dataset, OpenNeRF outperforms recent open-vocabulary methods such as LERF and OpenScene by at least +4.9 mIoU.

</details>

---

## 51. Understanding Transferable Representation Learning and Zero-shot Transfer in CLIP

- [ ] Understanding Transferable Representation Learning and Zero-shot Transfer in CLIP | https://iclr.cc/virtual/2024/poster/18616

- **Link**: https://iclr.cc/virtual/2024/poster/18616

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal learning has become increasingly popular due to its ability to leverage information from different data sources (e.g., text and images) to improve the model performance. Recently, CLIP has emerged as an effective approach that employs vision-language contrastive pretraining to learn joint image and text representations and exhibits remarkable performance in zero-shot learning and text-guided natural image generation. Despite the substantial practical success of CLIP, its theoretical understanding remains elusive. In this paper, we formally study transferrable representation learning underlying CLIP and demonstrate how features from different modalities get aligned. We also analyze its zero-shot transfer performance on the downstream tasks. In addition, we conduct empirical evaluations on real data to backup our theory. Inspired by our analysis, we propose a new CLIP-type approach, which achieves better performance than CLIP and other state-of-the-art methods on benchmark datasets.

</details>

---

## 52. Guiding Instruction-based Image Editing via Multimodal Large Language Models

- [ ] Guiding Instruction-based Image Editing via Multimodal Large Language Models | https://iclr.cc/virtual/2024/poster/18621

- **Link**: https://iclr.cc/virtual/2024/poster/18621

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Instruction-based image editing improves the controllability and flexibility of image manipulation via natural commands without elaborate descriptions or regional masks. However, human instructions are sometimes too brief for current methods to capture and follow. Multimodal large language models (MLLMs) show promising capabilities in cross-modal understanding and visual-aware response generation via LMs. We investigate how MLLMs facilitate edit instructions and present MLLM-Guided Image Editing (MGIE). MGIE learns to derive expressive instructions and provides explicit guidance. The editing model jointly captures this visual imagination and performs manipulation through end-to-end training. We evaluate various aspects of Photoshop-style modification, global photo optimization, and local editing. Extensive experimental results demonstrate that expressive instructions are crucial to instruction-based image editing, and our MGIE can lead to a notable improvement in automatic metrics and human evaluation while maintaining competitive inference efficiency.

</details>

---

## 53. LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment

- [ ] LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment | https://iclr.cc/virtual/2024/poster/18668

- **Link**: https://iclr.cc/virtual/2024/poster/18668

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The video-language (VL) pretraining has achieved remarkable improvement in multiple downstream tasks. However, the current VL pretraining framework is hard to extend to multiple modalities (N modalities, N ≥ 3) beyond vision and language. We thus propose LanguageBind, taking the language as the bind across different modalities because the language modality is well-explored and contains rich semantics. Specifically, we freeze the language encoder acquired by VL pretraining and then train encoders for other modalities with contrastive learning. As a result, all modalities are mapped to a shared feature space, implementing multi-modal semantic alignment. While LanguageBind ensures that we can extend VL modalities to N modalities, we also need a high-quality dataset with alignment data pairs centered on language. We thus propose VIDAL-10M with 10 Million data with Video, Infrared, Depth, Audio and their corresponding Language. In our VIDAL-10M, all videos are from short video platforms with complete semantics rather than truncated segments from long videos, and all the video, depth, infrared, and audio modalities are aligned to their textual descriptions. LanguageBind has achieved superior performance on a wide range of 15 benchmarks covering video, audio, depth, and infrared. Moreover, multiple experiments have provided evidence for the effectiveness of LanguageBind in achieving indirect alignment and complementarity among diverse modalities.

</details>

---

## 54. Overcoming the Pitfalls of Vision-Language Model Finetuning for OOD Generalization

- [ ] Overcoming the Pitfalls of Vision-Language Model Finetuning for OOD Generalization | https://iclr.cc/virtual/2024/poster/18711

- **Link**: https://iclr.cc/virtual/2024/poster/18711

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing vision-language models exhibit strong generalization on a variety of visual domains and tasks. However, such models mainly perform zero-shot recognition in a closed-set manner, and thus struggle to handle open-domain visual concepts by design. There are recent finetuning methods, such as prompt learning, that not only study the discrimination between in-distribution (ID) and out-of-distribution (OOD) samples, but also show some improvements in both ID and OOD accuracies. In this paper, we first demonstrate that vision-language models, after long enough finetuning but without proper regularization, tend to overfit the known classes in the given dataset, with degraded performance on unknown classes. Then we propose a novel approach OGEN to address this pitfall, with the main focus on improving the OOD GENeralization of finetuned models. Specifically, a class-conditional feature generator is introduced to synthesize OOD features using just the class name of any unknown class. Such synthesized features will provide useful knowledge about unknowns and help regularize the decision boundary between ID and OOD data when optimized jointly. Equally important is our adaptive self-distillation mechanism to regularize our feature generation model during joint optimization, i.e., adaptively transferring knowledge between model states to further prevent overfitting. Experiments validate that our method yields convincing gains in OOD generalization performance in different settings. Code: https://github.com/apple/ml-ogen.

</details>

---

## 55. CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding

- [ ] CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding | https://iclr.cc/virtual/2024/poster/18715

- **Link**: https://iclr.cc/virtual/2024/poster/18715

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A remarkable ability of human beings resides in compositional reasoning, i.e., the capacity to make "infinite use of finite means". However, current large vision-language foundation models (VLMs)  fall short of such compositional abilities due to their ``bag-of-words" behaviors and inability to construct words that correctly represent visual entities and the relations among the entities. To this end, we propose CoVLM, which can guide the LLM to explicitly compose visual entities and relationships among the text and dynamically communicate with the vision encoder and detection network to achieve vision-language communicative decoding. Specifically, we first devise a set of novel communication tokens for the LLM, for dynamic communication between the visual detection system and the language system. A communication token is generated by the LLM following a visual entity or a relation, to inform the detection network to propose regions that are relevant to the sentence generated so far. The proposed regions-of-interests (ROIs) are then fed back into the LLM for better language generation contingent on the relevant regions. The LLM is thus able to compose the visual entities and relationships through the communication tokens. The vision-to-language and language-to-vision communication are iteratively performed until the entire sentence is generated. Our framework seamlessly bridges the gap between visual perception and LLMs and outperforms previous VLMs by a large margin on compositional reasoning benchmarks (e.g., ~20% in HICO-DET mAP, ~14% in Cola top-1 accuracy, and ~3% on ARO top-1 accuracy). We also achieve state-of-the-art performances on traditional vision-language tasks such as referring expression comprehension and visual question answering.

</details>

---

## 56. Federated Text-driven Prompt Generation for Vision-Language Models

- [ ] Federated Text-driven Prompt Generation for Vision-Language Models | https://iclr.cc/virtual/2024/poster/18784

- **Link**: https://iclr.cc/virtual/2024/poster/18784

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning for vision-language models, e.g., CoOp, has shown great success in adapting CLIP to different downstream tasks, making it a promising solution for federated learning due to computational reasons. Existing prompt learning techniques replace hand-crafted text prompts with learned vectors that offer improvements on seen classes, but struggle to generalize to unseen classes. Our work addresses this challenge by proposing Federated Text-driven Prompt Generation (FedTPG), which learns a unified prompt generation network across multiple remote clients in a scalable manner. The prompt generation network is conditioned on task-related text input, thus is context-aware, making it suitable to generalize for both seen and unseen classes. Our comprehensive empirical evaluations on nine diverse image classification datasets show that our method is superior to existing federated prompt learning methods, achieving better overall generalization on both seen and unseen classes, as well as datasets.

</details>

---

## 57. Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning

- [ ] Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning | https://iclr.cc/virtual/2024/poster/18802

- **Link**: https://iclr.cc/virtual/2024/poster/18802

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Reinforcement learning (RL) requires either manually specifying a reward function, which is often infeasible, or learning a reward model from a large amount of human feedback, which is often very expensive. We study a more sample-efficient alternative: using pretrained vision-language models (VLMs) as zero-shot reward models (RMs) to specify tasks via natural language. We propose a natural and general approach to using VLMs as reward models, which we call VLM-RMs. We use VLM-RMs based on CLIP to train a MuJoCo humanoid to learn complex tasks without a manually specified reward function, such as kneeling, doing the splits, and sitting in a lotus position. For each of these tasks, we only provide a single sentence text prompt describing the desired task with minimal prompt engineering. We provide videos of the trained agents at: https://sites.google.com/view/vlm-rm. We can improve performance by providing a second "baseline" prompt and projecting out parts of the CLIP embedding space irrelevant to distinguish between goal and baseline. Further, we find a strong scaling effect for VLM-RMs: larger VLMs trained with more compute and data are better reward models. The failure modes of VLM-RMs we encountered are all related to known capability limitations of current VLMs, such as limited spatial reasoning ability or visually unrealistic environments that are far off-distribution for the VLM. We find that VLM-RMs are remarkably robust as long as the VLM is large enough. This suggests that future VLMs will become more and more useful reward models for a wide range of RL applications.

</details>

---

## 58. GENOME: Generative Neuro-Symbolic Visual Reasoning by Growing and Reusing Modules

- [ ] GENOME: Generative Neuro-Symbolic Visual Reasoning by Growing and Reusing Modules | https://iclr.cc/virtual/2024/poster/18827

- **Link**: https://iclr.cc/virtual/2024/poster/18827

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent works have shown that Large Language Models (LLMs) could empower traditional neuro-symbolic models via programming capabilities to translate languages into module descriptions, thus achieving strong visual reasoning results while maintaining the model’s transparency and efficiency. However, these models usually exhaustively generate the entire code snippet given each new instance of a task, which is extremely ineffective. On the contrary, human beings gradually acquire knowledge that can be reused and grow into more profound skills for fast generalization to new tasks since we are an infant. Inspired by this, we propose generative neuro-symbolic visual reasoning by growing and reusing modules. Specifically, our model consists of three unique stages, module initialization, module generation, and module execution. First, given a vision-language task, we adopt LLMs to examine whether we could reuse and grow over established modules to handle this new task. If not, we initialize a new module needed by the task and specify the inputs and outputs of this new module. After that, the new module is created by querying LLMs to generate corresponding code snippets that match the requirements. In order to get a better sense of the new module’s ability, we treat few-shot training examples as test cases to see if our new module could pass these cases. If yes, the new module is added to the module library for future reuse. Finally, we evaluate the performance of our model on the testing set by executing the parsed programs with the newly made visual modules to get the results. We find the proposed GENOME model possesses several advantages. First, it performs competitively on standard tasks like visual question answering and referring expression comprehension; Second, the visual modules learned from one task can be seamlessly transferred to new tasks; Last but not least, it is able to adapt to new visual reasoning tasks by observing a few training examples and reusing modules.

</details>

---

## 59. InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation

- [ ] InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation | https://iclr.cc/virtual/2024/poster/18829

- **Link**: https://iclr.cc/virtual/2024/poster/18829

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces InternVid, a large-scale video-centric multimodal dataset that enables learning powerful and transferable video-text representations for multimodal understanding and generation. InternVid contains over 7 million videos lasting nearly 760K hours, yielding 234M video clips accompanied by detailed descriptions of total 4.1B words. Our core contribution is to develop a scalable approach to autonomously build a high-quality video-text dataset with large language models (LLM), thereby showcasing its efficacy in learning video-language representation at scale. Specifically, we utilize a multi-scale approach to generate video-related descriptions. Furthermore, we introduce ViCLIP, a video-text representation learning model based on ViT-L. Learned on InternVid via contrastive learning, this model demonstrates leading zero-shot action recognition and competitive video retrieval performance. Beyond basic video understanding tasks like recognition and retrieval, our dataset and model have broad applications. They are particularly beneficial for generating interleaved video-text data for learning a video-centric dialogue system, advancing video-to-text and text-to-video generation research. These proposed resources provide a tool for researchers and practitioners interested in multimodal video understanding and generation.

</details>

---

## 60. Rephrase, Augment, Reason: Visual Grounding of Questions for Vision-Language Models

- [ ] Rephrase, Augment, Reason: Visual Grounding of Questions for Vision-Language Models | https://iclr.cc/virtual/2024/poster/18874

- **Link**: https://iclr.cc/virtual/2024/poster/18874

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

An increasing number of vision-language tasks can be handled with little to no training, i.e., in a zero and few-shot manner, by marrying large language models (LLMs) to vision encoders, resulting in large vision-language models (LVLMs). While this has huge upsides, such as not requiring training data or custom architectures, how an input is presented to an LVLM can have a major impact on zero-shot model performance. In particular, inputs phrased in an underspecified way can result in incorrect answers due to factors like missing visual information, complex implicit reasoning, or linguistic ambiguity. Therefore, adding visually-grounded information to the input as a preemptive clarification should improve model performance by reducing underspecification, e.g., by localizing objects and disambiguating references. Similarly, in the VQA setting, changing the way questions are framed can make them easier for models to answer. To this end, we present Rep hrase, A ugment and Re ason (RepARe), a gradient-free framework that extracts salient details about the image using the underlying LVLM as a captioner and reasoner, in order to propose modifications to the original question. We then use the LVLM’s confidence over a generated answer as an unsupervised scoring function to select the rephrased question most likely to improve zero-shot performance. Focusing on three visual question answering tasks, we show that RepARe can result in a 3.85% (absolute) increase in zero-shot accuracy on VQAv2, 6.41%, and 7.94% points increase on A-OKVQA, and VizWiz respectively. Additionally, we find that using gold answers for oracle question candidate selection achieves a substantial gain in VQA accuracy by up to 14.41%. Through extensive analysis, we demonstrate that outputs from RepARe increase syntactic complexity, and effectively utilize vision-language interaction and the frozen LLM.

</details>

---

## 61. DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning

- [ ] DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning | https://iclr.cc/virtual/2024/poster/18890

- **Link**: https://iclr.cc/virtual/2024/poster/18890

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning (PT), where a small amount of trainable soft (continuous) prompt vectors is affixed to the input of language models (LM), has shown promising results across various tasks and models for parameter-efficient fine-tuning (PEFT). PT stands out from other PEFT approaches because it maintains competitive performance with fewer trainable parameters and does not drastically scale up its parameters as the model size expands. However, PT introduces additional soft prompt tokens, leading to longer input sequences, which significantly impacts training and inference time and memory usage due to the Transformer's quadratic complexity. Particularly concerning for Large Language Models (LLMs) that face heavy daily querying. To address this issue, we propose Decomposed Prompt Tuning (DePT), which decomposes the soft prompt into a shorter soft prompt and a pair of low-rank matrices that are then optimised with two different learning rates. This allows DePT to achieve better performance while saving substantial memory and time costs compared to vanilla PT and its variants, without changing trainable parameter sizes. Through extensive experiments on 23 natural language processing (NLP) and vision-language (VL) tasks, we demonstrate that DePT outperforms state-of-the-art PEFT approaches, including the full fine-tuning baseline, in some scenarios. Additionally, we empirically show that DEPT grows more efficient as the model size increases. Our further study reveals that DePT integrates seamlessly with parameter-efficient transfer learning in the few-shot learning setting and highlights its adaptability to various model architectures and sizes.

</details>

---

## 62. FairerCLIP: Debiasing CLIP's Zero-Shot Predictions using Functions in RKHSs

- [ ] FairerCLIP: Debiasing CLIP's Zero-Shot Predictions using Functions in RKHSs | https://iclr.cc/virtual/2024/poster/18989

- **Link**: https://iclr.cc/virtual/2024/poster/18989

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large pre-trained vision-language models such as CLIP provide compact and general-purpose representations of text and images that are demonstrably effective across multiple downstream zero-shot prediction tasks. However, owing to the nature of their training process, these models have the potential to 1) propagate or amplify societal biases in the training data and 2) learn to rely on spurious features. This paper proposes FairerCLIP, a general approach for making zero-shot predictions of CLIP more fair and robust to spurious correlations. We formulate the problem of jointly debiasing CLIP’s image and text representations in reproducing kernel Hilbert spaces (RKHSs), which affords multiple benefits: 1) Flexibility: Unlike existing approaches, which are specialized to either learn with or without ground-truth labels, FairerCLIP is adaptable to learning in both scenarios. 2) Ease of Optimization: FairerCLIP lends itself to an iterative optimization involving closed-form solvers, which leads to 4×-10× faster training than the existing methods. 3) Sample Efficiency: Under sample-limited conditions, FairerCLIP significantly outperforms baselines when they fail entirely. And, 4) Performance: Empirically, FairerCLIP achieves appreciable accuracy gains on benchmark fairness and spurious correlation datasets over their respective baselines.

</details>

---

## 63. Image Clustering Conditioned on Text Criteria

- [ ] Image Clustering Conditioned on Text Criteria | https://iclr.cc/virtual/2024/poster/19041

- **Link**: https://iclr.cc/virtual/2024/poster/19041

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Classical clustering methods do not provide users with direct control of the clustering results, and the clustering results may not be consistent with the relevant criterion that a user has in mind. In this work, we present a new methodology for performing image clustering based on user-specified criteria in the form of text by leveraging modern Vision-Language Models and Large Language Models. We call our method Image Clustering Conditioned on Text Criteria (IC$|$TC), and it represents a different paradigm of image clustering. IC$|$TC requires a minimal and practical degree of human intervention and grants the user significant control over the clustering results in return. Our experiments show that IC$|$TC can effectively cluster images with various criteria, such as human action, physical location, or the person's mood, significantly outperforming baselines.

</details>

---

## 64. CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets

- [ ] CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets | https://iclr.cc/virtual/2024/poster/19043

- **Link**: https://iclr.cc/virtual/2024/poster/19043

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) are often augmented with tools to solve complex tasks. By generating code snippets and executing them through task-specific Application Programming Interfaces (APIs), they can offload certain functions to dedicated external modules, such as image encoding and performing calculations. However, most existing approaches to augment LLMs with tools are constrainedby general-purpose APIs and lack the flexibility for tailoring them to specific tasks. In this work, we present CRAFT, a general tool creation and retrieval framework for LLMs. It creates toolsets specifically curated for the tasks and equips LLMs with a component that retrieves tools from these sets to enhance their capability to solve complex tasks. For each task, we collect specific code solutions by promptingGPT-4 to solve the training examples. Following a validation step ensuring the correctness, these solutions are abstracted into code snippets to enhance reusability, and deduplicated for higher quality. At inference time, the language model retrieves snippets from the toolsets and then executes them or generates the output conditioning on the retrieved snippets. Our method is designed to be flexible andoffers a plug-and-play approach to adapt off-the-shelf LLMs to unseen domains and modalities, without any finetuning. Experiments on vision-language, tabular processing, and mathematical reasoning tasks show that our approach achieves substantial improvements compared to strong baselines. In addition, our in-depth analysis reveals that: (1) consistent performance improvement can be achieved byscaling up the number of tools and the capability of the backbone models; (2) each component of our approach contributes to the performance gains; (3) the created tools are well-structured and reliable with low complexity  and atomicity.

</details>

---

## 65. Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization

- [ ] Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization | https://iclr.cc/virtual/2024/poster/19049

- **Link**: https://iclr.cc/virtual/2024/poster/19049

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, the remarkable advance of the Large Language Model (LLM) has inspired researchers to transfer its extraordinary reasoning capability to both vision and language data. However, the prevailing approaches primarily regard the visual input as a prompt and focus exclusively on optimizing the text generation process conditioned upon vision content by a frozen LLM. Such an inequitable treatment of vision and language heavily constrains the model's potential. In this paper, we break through this limitation by representing both vision and language in a unified form. Specifically, we introduce a well-designed visual tokenizer to translate the non-linguistic image into a sequence of discrete tokens like a foreign language that LLM can read. The resulting visual tokens encompass high-level semantics worthy of a word and also support dynamic sequence length varying from the image. Coped with this tokenizer, the presented foundation model called LaVIT can handle both image and text indiscriminately under the same generative learning paradigm. This unification empowers LaVIT to serve as an impressive generalist interface to understand and generate multi-modal content simultaneously. Extensive experiments further showcase that it outperforms the existing models by a large margin on massive vision-language tasks. Our code and models are available at https://github.com/jy0205/LaVIT.

</details>

---

## 66. Open-ended VQA benchmarking of Vision-Language models by exploiting Classification datasets and their semantic hierarchy

- [ ] Open-ended VQA benchmarking of Vision-Language models by exploiting Classification datasets and their semantic hierarchy | https://iclr.cc/virtual/2024/poster/19102

- **Link**: https://iclr.cc/virtual/2024/poster/19102

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The evaluation of text-generative vision-language models is a challenging yet crucial endeavor. By addressing the limitations of existing Visual Question Answering (VQA) benchmarks and proposing innovative evaluation methodologies, our research seeks to advance our understanding of these models’ capabilities. We propose a novel VQA benchmark based on well-known visual classification datasets which allows a granular evaluation of text-generative vision-language models and their comparison with discriminative vision-language models. To improve the assessment of coarse answers on fine-grained classification tasks, we suggest using the semantic hierarchy of the label space to ask automatically generated follow-up questions about the ground-truth category. Finally, we compare traditional NLP and LLM-based metrics for the problem of evaluating model predictions given ground-truth answers. We perform a human evaluation study upon which we base our decision on the final metric. We apply our benchmark to a suite of vision-language models and show a detailed comparison of their abilities on object, action, and attribute classification. Our contributions aim to lay the foundation for more precise and meaningful assessments, facilitating targeted progress in the exciting field of vision-language modeling.

</details>

---

## 67. Vision-by-Language for Training-Free Compositional Image Retrieval

- [ ] Vision-by-Language for Training-Free Compositional Image Retrieval | https://iclr.cc/virtual/2024/poster/19114

- **Link**: https://iclr.cc/virtual/2024/poster/19114

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Given an image and a target modification (e.g an image of the Eiffel tower and the text “without people and at night-time”), Compositional Image Retrieval (CIR) aims to retrieve the relevant target image in a database. While supervised approaches rely on annotating triplets that is costly (i.e. query image, textual modification, and target image), recent research sidesteps this need by using large-scale vision-language models (VLMs), performing Zero-Shot CIR (ZS-CIR). However, state-of-the-art approaches in ZS-CIR still require training task-specific, customized models over large amounts of image-text pairs. In this work, we proposeto tackle CIR in a training-free manner via our Compositional Image Retrieval through Vision-by-Language (CIReVL), a simple, yet human-understandable and scalable pipeline that effectively recombines large-scale VLMs with large language models (LLMs). By captioning the reference image using a pre-trained generative VLM and asking a LLM to recompose the caption based on the textual target modification for subsequent retrieval via e.g. CLIP, we achieve modular language reasoning. In four ZS-CIR benchmarks, we find competitive, in-part state-of-the-art performance - improving over supervised methods Moreover, the modularity of CIReVL offers simple scalability without re-training, allowing us to both investigate scaling laws and bottlenecks for ZS-CIR while easily scaling up to in parts more than double of previously reported results. Finally, we show that CIReVL makes CIR human-understandable by composing image and text in a modular fashion in the language domain, thereby making it intervenable, allowing to post-hoc re-align failure cases. Code will be released upon acceptance.

</details>

---

## 68. CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction

- [ ] CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction | https://iclr.cc/virtual/2024/poster/19128

- **Link**: https://iclr.cc/virtual/2024/poster/19128

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary dense prediction tasks including object detection and image segmentation have been advanced by the success of Contrastive Language-Image Pre-training (CLIP). CLIP models, particularly those incorporating vision transformers (ViTs), have exhibited remarkable generalization ability in zero-shot image classification. However, when transferring the vision-language alignment of CLIP from global image representation to local region representation for the open-vocabulary dense prediction tasks, CLIP ViTs suffer from the domain shift from full images to local image regions. In this paper, we embark on an in-depth analysis of the region-language alignment in CLIP models, which is essential for downstream open-vocabulary dense prediction tasks. Subsequently, we propose an approach named CLIPSelf, which adapts the image-level recognition ability of CLIP ViT to local image regions without needing any region-text pairs. CLIPSelf empowers ViTs to distill itself by aligning a region representation extracted from its dense feature map with the image-level representation of the corresponding image crop. With the enhanced CLIP ViTs, we achieve new state-of-the-art performance on open-vocabulary object detection, semantic segmentation, and panoptic segmentation across various benchmarks. Models and code are released at https://github.com/wusize/CLIPSelf.

</details>

---

## 69. MetaCoCo: A New Few-Shot Classification Benchmark with Spurious Correlation

- [ ] MetaCoCo: A New Few-Shot Classification Benchmark with Spurious Correlation | https://iclr.cc/virtual/2024/poster/19133

- **Link**: https://iclr.cc/virtual/2024/poster/19133

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) problems in few-shot classification (FSC) occur when novel classes sampled from testing distributions differ from base classes drawn from training distributions, which considerably degrades the performance of deep learning models deployed in real-world applications. Recent studies suggest that the OOD problems in FSC mainly including: (a) cross-domain few-shot classification (CD-FSC) and (b) spurious-correlation few-shot classification (SC-FSC). Specifically, CD-FSC occurs when a classifier learns transferring knowledge from base classes drawn from \underline{seen} training distributions but recognizes novel classes sampled from unseen testing distributions. In contrast, SC-FSC arises when a classifier relies on non-causal features (or contexts) that happen to be correlated with the labels (or concepts) in base classes but such relationships no longer hold during the model deployment. Despite CD-FSC has been extensively studied, SC-FSC remains understudied due to lack of the corresponding evaluation benchmarks. To this end, we present Meta Concept Context (MetaCoCo), a benchmark with spurious-correlation shifts collected from real-world scenarios. Moreover, to quantify the extent of spurious-correlation shifts of the presented MetaCoCo, we further propose a metric by using CLIP as a pre-trained vision-language model. Extensive experiments on the proposed benchmark are performed to evaluate the state-of-the-art methods in FSC, cross-domain shifts, and self-supervised learning. The experimental results show that the performance of the existing methods degrades significantly in the presence of spurious-correlation shifts. We open-source all codes of our benchmark and hope that the proposed MetaCoCo can facilitate future research on spurious-correlation shifts problems in FSC.

</details>

---

## 70. LLCP: Learning Latent Causal Processes for Reasoning-based Video Question Answer

- [ ] LLCP: Learning Latent Causal Processes for Reasoning-based Video Question Answer | https://iclr.cc/virtual/2024/poster/19158

- **Link**: https://iclr.cc/virtual/2024/poster/19158

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current approaches to Video Question Answering (VideoQA) primarily focus on cross-modality matching, which is limited by the requirement for extensive data annotations and the insufficient capacity for causal reasoning (e.g. attributing accidents). To address these challenges, we introduce a causal framework for video reasoning, termed Learning Latent Causal Processes (LLCP). At the heart of LLCP lies a multivariate generative model designed to analyze the spatial-temporal dynamics of objects within events. Leveraging the inherent modularity of causal mechanisms, we train the model through self-supervised local auto-regression eliminating the need for annotated question-answer pairs. During inference, the model is applied to answer two types of reasoning questions: accident attribution, which infers the cause from observed effects, and counterfactual prediction, which predicts the effects of counterfactual conditions given the factual evidence. In the first scenario, we identify variables that deviate from the established distribution by the learned model, signifying the root cause of accidents. In the second scenario, we replace embeddings of previous variables with counterfactual ones, enabling us to forecast potential developments. Once we have identified these cause/effect variables, natural language answers are derived through a combination of grammatical parsing and a pre-trained vision-language model. We assess the efficacy of LLCP on both synthetic and real-world data, demonstrating comparable performance to supervised methods despite our framework using no paired textual annotations.

</details>

---

## 71. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images

- [ ] Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images | https://iclr.cc/virtual/2024/poster/19194

- **Link**: https://iclr.cc/virtual/2024/poster/19194

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87× and 8.56× compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications.

</details>

---

## 72. LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation

- [ ] LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation | https://iclr.cc/virtual/2024/poster/19198

- **Link**: https://iclr.cc/virtual/2024/poster/19198

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Following the impressive development of LLMs, vision-language alignment in LLMs is actively being researched to enable multimodal reasoning and visual input/output. This direction of research is particularly relevant to medical imaging because accurate medical image analysis and generation consist of a combination of reasoning based on visual features and prior knowledge. Many recent works have focused on training adapter networks that serve as an information bridge between image processing (encoding or generating) networks and LLMs; but presumably, in order to achieve maximum reasoning potential of LLMs on visual information as well, visual and language features should be allowed to interact more freely. This is especially important in the medical domain because understanding and generating medical images such as chest X-rays (CXR) require not only accurate visual and language-based reasoning but also a more intimate mapping between the two modalities. Thus, taking inspiration from previous work on the transformer and VQ-GAN combination for bidirectional image and text generation, we build upon this approach and develop a method for instruction-tuning an LLM pre-trained only on text to gain vision-language capabilities for medical images. Specifically, we leverage a pretrained LLM’s existing question-answering and instruction-following abilities to teach it to understand visual inputs by instructing it to answer questions about image inputs and, symmetrically, output both text and image responses appropriate to a given query by tuning the LLM with diverse tasks that encompass image-based text-generation and text-based image-generation. We show that our LLM-CXR trained in this approach shows better image-text alignment in both CXR understanding and generation tasks while being smaller in size compared to previously developed models that perform a narrower range of tasks.

</details>

---

## 73. Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions

- [ ] Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions | https://iclr.cc/virtual/2024/poster/19212

- **Link**: https://iclr.cc/virtual/2024/poster/19212

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have been utilizing Visual Prompt Generators (VPGs) to convert visual features into tokens that LLMs can recognize. This is achieved by training the VPGs on millions of image-caption pairs, where the VPG-generated tokens of images are fed into a frozen LLM to generate the corresponding captions. However, this image-captioning based training objective inherently biases the VPG to concentrate solely on the primary visual contents sufficient for caption generation, often neglecting other visual details. This shortcoming results in MLLMs’ underperformance in comprehending demonstrative instructions consisting of multiple, interleaved, and multimodal instructions that demonstrate the required context to complete a task. To address this issue, we introduce a generic and lightweight Visual Prompt Generator Complete module (VPG-C), which can infer and complete the missing details essential for comprehending demonstrative instructions. Further, we propose a synthetic discriminative training strategy to fine-tune VPG-C, eliminating the need for supervised demonstrative instructions. As for evaluation, we build DEMON, a comprehensive benchmark for demonstrative instruction understanding. Synthetically trained with the proposed strategy, VPG-C achieves significantly stronger zero-shot performance across all tasks of DEMON. Further evaluation on the MME and OwlEval benchmarks also demonstrate the superiority of VPG-C. The code and models are available at https://github.com/DCDmllm/Cheetah.

</details>

---

## 74. Rethinking Model Ensemble in Transfer-based Adversarial Attacks

- [ ] Rethinking Model Ensemble in Transfer-based Adversarial Attacks | https://iclr.cc/virtual/2024/poster/19251

- **Link**: https://iclr.cc/virtual/2024/poster/19251

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

It is widely recognized that deep learning models lack robustness to adversarial examples. An intriguing property of adversarial examples is that they can transfer across different models, which enables black-box attacks without any knowledge of the victim model. An effective strategy to improve the transferability is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble methods can strongly improve the transferability. In this paper, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with two properties: 1) the flatness of loss landscape; and 2) the closeness to the local optimum of each model. We empirically and theoretically show that both properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improving the adversarial transferability, especially when attacking adversarially trained models. We also successfully apply our method to attack a black-box large vision-language model -- Google's Bard, showing the practical effectiveness. Code is available at \url{https://github.com/huanranchen/AdversarialAttacks}.

</details>

---

## 75. Video Language Planning

- [ ] Video Language Planning | https://iclr.cc/virtual/2024/poster/19282

- **Link**: https://iclr.cc/virtual/2024/poster/19282

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We are interested in enabling visual planning for complex long-horizon tasks in the space of generated videos and language, leveraging recent advances in large generative models pretrained on Internet-scale data.  To this end, we present video language planning (VLP), an algorithm that consists of a tree search procedure, where we train (i) vision-language models to serve as both policies and value functions, and (ii) text-to-video models as dynamics models. VLP takes as input a long-horizon task instruction and current image observation, and outputs a long video plan that provides detailed multimodal (video and language) specifications that describe how to complete the final task. VLP scales with increasing computation budget where more computation time results in improved video plans, and is able to synthesize long-horizon video plans across different robotics domains -- from multi-object rearrangement, to multi-camera bi-arm dexterous manipulation. Generated video plans can be translated into real robot actions via goal-conditioned policies, conditioned on each intermediate frame of the generated video. Experiments show that VLP substantially improves long-horizon task success rates compared to prior methods on both simulated and real robots (across 3 hardware platforms).

</details>

---

## 76. Bootstrapping Variational Information Pursuit with Large Language and Vision Models for Interpretable Image Classification

- [ ] Bootstrapping Variational Information Pursuit with Large Language and Vision Models for Interpretable Image Classification | https://iclr.cc/virtual/2024/poster/19290

- **Link**: https://iclr.cc/virtual/2024/poster/19290

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Variational Information Pursuit (V-IP) is an interpretable-by-design framework that makes predictions by sequentially selecting a short chain of user-defined, interpretable queries about the data that are most informative for the task. The prediction is based solely on the obtained query answers, which also serve as a faithful explanation for the prediction. Applying the framework to any task requires (i) specification of a query set, and (ii) densely annotated data with query answers to train classifiers to answer queries at test time. This limits V-IP's application to small-scale tasks where manual data annotation is feasible. In this work, we focus on image classification tasks and propose to relieve this bottleneck by leveraging pretrained language and vision models. Specifically, following recent work, we propose to use GPT, a Large Language Model, to propose semantic concepts as queries for a given classification task. To answer these queries, we propose a light-weight Concept Question-Answering network (Concept-QA) which learns to answer binary queries about semantic concepts in images. We design pseudo-labels to train our Concept-QA model using GPT and CLIP (a Vision-Language Model). Empirically, we find our Concept-QA model to be competitive with state-of-the-art VQA models in terms of answering accuracy but with an order of magnitude fewer parameters. This allows for seamless integration of Concept-QA into the V-IP framework as a fast-answering mechanism. We name this method Concept-QA+V-IP. Finally, we show on several datasets that Concept-QA+V-IP produces shorter, interpretable query chains which are more accurate than V-IP trained with CLIP-based answering systems. Code available at https://github.com/adityac94/conceptqa_vip.

</details>

---

## 77. Multi-granularity Correspondence Learning from Long-term Noisy Videos

- [ ] Multi-granularity Correspondence Learning from Long-term Noisy Videos | https://iclr.cc/virtual/2024/poster/19303

- **Link**: https://iclr.cc/virtual/2024/poster/19303

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing video-language studies mainly focus on learning short video clips, leaving long-term temporal dependencies rarely explored due to over-high computational cost of modeling long videos. To address this issue, one feasible solution is learning the correspondence between video clips and captions, which however inevitably encounters the multi-granularity noisy correspondence (MNC) problem. To be specific, MNC refers to the clip-caption misalignment (coarse-grained) and frame-word misalignment (fine-grained), hindering temporal learning and video understanding. In this paper, we propose NOise Robust Temporal Optimal traNsport (Norton) that addresses MNC in a unified optimal transport (OT) framework. In brief, Norton employs video-paragraph and clip-caption contrastive losses to capture long-term dependencies based on OT. To address coarse-grained misalignment in video-paragraph contrast, Norton filters out the irrelevant clips and captions through an alignable prompt bucket and realigns asynchronous clip-caption pairs based on transport distance. To address the fine-grained misalignment, Norton incorporates a soft-maximum operator to identify crucial words and key frames. Additionally, Norton exploits the potential faulty negative samples in clip-caption contrast by rectifying the alignment target with OT assignment to ensure precise temporal modeling. Extensive experiments on video retrieval, videoQA, and action segmentation verify the effectiveness of our method. Code is available at https://lin-yijie.github.io/projects/Norton.

</details>

---

## 78. SLiMe: Segment Like Me

- [ ] SLiMe: Segment Like Me | https://iclr.cc/virtual/2024/poster/19368

- **Link**: https://iclr.cc/virtual/2024/poster/19368

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Significant strides have been made using large vision-language models, like Stable Diffusion (SD), for a variety of downstream tasks, including image generation, image editing, and 3D shape generation. Inspired by these advancements, we explore leveraging these vision-language models for segmenting images at any desired granularity using as few as one annotated sample. We propose SLiMe, which frames this problem as an optimization task. Specifically, given a single image and its segmentation mask, we first extract our novel “weighted accumulated self-attention map” along with cross-attention map from the SD prior. Then, using these extracted maps, the text embeddings of SD are optimized to highlight the segmented region in these attention maps, which in turn can be used to derive new segmentation results. Moreover, leveraging additional training data when available, i.e. few-shot, improves the performance of SLiMe. We performed comprehensive experiments examining various design factors and showed that SLiMe outperforms other existing one-shot and few-shot segmentation methods.

</details>

---

## 79. INViTE: INterpret and Control Vision-Language Models with Text Explanations

- [ ] INViTE: INterpret and Control Vision-Language Models with Text Explanations | https://iclr.cc/virtual/2024/poster/19418

- **Link**: https://iclr.cc/virtual/2024/poster/19418

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained vision foundation models, such as CLIP, have become de facto backbones for various vision tasks. However, due to their black-box nature, understanding the underlying rules behind these models’ predictions and controlling model behaviors have remained open challenges. We present INViTE: a framework for INterpreting Vision Transformer’s latent tokens with Text Explanations. Given a latent token, INViTE retains its semantic information to the final layer using transformer’s local operations and retrieves the closest text for explanation. INViTE enables understanding of model visual reasoning procedure without needing additional model training or data collection. Based on the obtained interpretations, INViTE allows for model editing that controls model reasoning behaviors and improves model robustness against biases and spurious correlations. Our code is available at https://github.com/tonychenxyz/vit-interpret.

</details>

---

## 80. Structured Video-Language Modeling with Temporal Grouping and Spatial Grounding

- [ ] Structured Video-Language Modeling with Temporal Grouping and Spatial Grounding | https://iclr.cc/virtual/2024/poster/19422

- **Link**: https://iclr.cc/virtual/2024/poster/19422

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing video-language pre-training methods primarily focus on instance-level alignment between video clips and captions via global contrastive learning but neglect rich fine-grained local information in both videos and text, which is of importance to downstream tasks requiring temporal localization and semantic reasoning. A powerful model is expected to be capable of capturing region-object correspondences and recognizing scene changes in a video clip, reflecting spatial and temporal granularity, respectively. To strengthen model's understanding into such fine-grained details, we propose a simple yet effective video-language modeling framework, S-ViLM, by exploiting the intrinsic structures of these two modalities. It includes two novel designs, inter-clip spatial grounding and intra-clip temporal grouping, to promote learning region-object alignment and temporal-aware features, simultaneously. Comprehensive evaluations demonstrate that S-ViLM performs favorably against existing approaches in learning more expressive representations. Specifically, S-ViLM surpasses the state-of-the-art methods substantially on four representative downstream tasks, covering text-video retrieval, video question answering, video action recognition, and temporal action localization.

</details>

---

## 81. MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning

- [ ] MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning | https://iclr.cc/virtual/2024/poster/19429

- **Link**: https://iclr.cc/virtual/2024/poster/19429

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Since the resurgence of deep learning, vision-language models (VLMs) enhanced by large language models (LLMs) have grown exponentially in popularity. However, while LLMs can utilize extensive background knowledge and task information with in-context learning, most VLMs still struggle with understanding complex multi-modal prompts with multiple images, making VLMs less effective in downstream vision-language tasks.In this paper, we address the limitation above by 1) introducing vision-language Model with M ulti- M odal I n- C ontext L earning(MMICL), a new approach to allow the VLM to deal with multi-modal inputs efficiently; 2) proposing a novel context scheme to augment the in-context learning ability of the VLM; 3) constructing the Multi-modal In-Context Learning (MIC) dataset, designed to enhance the VLM's ability to understand complex multi-modal prompts.Our experiments confirm that MMICL achieves new state-of-the-art zero-shot performance on a wide range of general vision-language tasks, especially for complex benchmarks, including MME and MMBench. Our analysis demonstrates that MMICL effectively tackles the challenge of complex multi-modal prompt understanding and emerges the impressive ICL ability. Furthermore, we observe that MMICL successfully alleviates language bias in VLMs, a common issue for VLMs that often leads to hallucination when faced with extensive textual context.Our code, dataset, dataset tool, and model are available at https://github.com/PKUnlp-icler/MIC.

</details>

---

## 82. Image2Sentence based Asymmetrical Zero-shot Composed Image Retrieval

- [ ] Image2Sentence based Asymmetrical Zero-shot Composed Image Retrieval | https://iclr.cc/virtual/2024/poster/19437

- **Link**: https://iclr.cc/virtual/2024/poster/19437

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The task of composed image retrieval (CIR) aims to retrieve images based on the query image and the text describing the users' intent. Existing methods have made great progress with the advanced large vision-language (VL) model in CIR task, however, they generally suffer from two main issues: lack of labeled triplets for model training and difficulty of deployment on resource-restricted environments when deploying the large vision-language model. To tackle the above problems, we propose Image2Sentence based Asymmetric zero-shot composed image retrieval (ISA), which takes advantage of the VL model and only relies on unlabeled images for composition learning. In the framework, we propose a new adaptive token learner that maps an image to a sentence in the word embedding space of VL model.  The sentence adaptively captures discriminative visual information and is further integrated with the text modifier. An asymmetric structure is devised for flexible deployment, in which the lightweight model is adopted for the query side while the large VL model is deployed on the gallery side. The global contrastive distillation and the local alignment regularization are adopted for the alignment between the light model and the VL model for CIR task.  Our experiments demonstrate that the proposed ISA could better cope with the real retrieval scenarios and further improve retrieval accuracy and efficiency.

</details>

---

## 83. Neurosymbolic Grounding for Compositional World Models

- [ ] Neurosymbolic Grounding for Compositional World Models | https://iclr.cc/virtual/2024/poster/19472

- **Link**: https://iclr.cc/virtual/2024/poster/19472

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Cosmos, a framework for object-centric world modeling that is designed for compositional generalization (CompGen), i.e., high performance on unseen input scenes obtained through the composition of known visual "atoms." The central insight behind Cosmos is the use of a novel form of neurosymbolic grounding. Specifically, the framework introduces two new tools: (i) neurosymbolic scene encodings, which represent each entity in a scene using a real vector computed using a neural encoder, as well as a vector of composable symbols describing attributes of the entity, and (ii) a neurosymbolic attention mechanism that binds these entities to learned rules of interaction. Cosmos is end-to-end differentiable; also, unlike traditional neurosymbolic methods that require representations to be manually mapped to symbols, it computes an entity's symbolic attributes using vision-language foundation models. Through an evaluation that considers two different forms of CompGen on an established blocks-pushing domain, we show that the framework establishes a new state-of-the-art for CompGen in world modeling. Artifacts are available at: https://trishullab.github.io/cosmos-web/

</details>

---

## 84. Ferret: Refer and Ground Anything Anywhere at Any Granularity

- [ ] Ferret: Refer and Ground Anything Anywhere at Any Granularity | https://iclr.cc/virtual/2024/poster/19537

- **Link**: https://iclr.cc/virtual/2024/poster/19537

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce Ferret, a new Multimodal Large Language Model (MLLM) capable of understanding spatial referring of any shape or granularity within an image and accurately grounding open-vocabulary descriptions. To unify referring and grounding in the LLM paradigm, Ferret employs a novel and powerful hybrid region representation that integrates discrete coordinates and continuous features jointly to represent a region in the image. To extract the continuous features of versatile regions,  we propose a spatial-aware visual sampler, adept at handling varying sparsity across different shapes. Consequently, Ferret can accept diverse region inputs, such as points, bounding boxes, and free-form shapes. To bolster the desired capability of Ferret, we curate GRIT, a comprehensive refer-and-ground instruction tuning dataset including 1.1M samples that contain rich hierarchical spatial knowledge, with an additional 130K hard negative data to promote model robustness. The resulting model not only achieves superior performance in classical referring and grounding tasks, but also greatly outperforms existing MLLMs in region-based and localization-demanded multimodal chatting. Our evaluations also reveal a significantly improved capability of describing image details and a remarkable alleviation in object hallucination.

</details>

---

## 85. PerceptionCLIP: Visual Classification by Inferring and Conditioning on Contexts

- [ ] PerceptionCLIP: Visual Classification by Inferring and Conditioning on Contexts | https://iclr.cc/virtual/2024/poster/19552

- **Link**: https://iclr.cc/virtual/2024/poster/19552

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models like CLIP are widely used in zero-shot image classification due to their ability to understand various visual concepts and natural language descriptions. However, how to fully leverage CLIP's unprecedented human-like understanding capabilities to achieve better performance is still an open question. This paper draws inspiration from the human visual perception process: when classifying an object, humans first infer contextual attributes (e.g., background and orientation) which help separate the foreground object from the background, and then classify the object based on this information. Inspired by it, we observe that providing CLIP with contextual attributes improves zero-shot image classification and mitigates reliance on spurious features. We also observe that CLIP itself can reasonably infer the attributes from an image. With these observations, we propose a training-free, two-step zero-shot classification method PerceptionCLIP. Given an image, it first infers contextual attributes (e.g., background) and then performs object classification conditioning on them. Our experiments show that PerceptionCLIP achieves better generalization, group robustness, and interpretability.

</details>

---

## 86. Lipsum-FT: Robust Fine-Tuning of Zero-Shot Models Using Random Text Guidance

- [ ] Lipsum-FT: Robust Fine-Tuning of Zero-Shot Models Using Random Text Guidance | https://iclr.cc/virtual/2024/poster/19555

- **Link**: https://iclr.cc/virtual/2024/poster/19555

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale contrastive vision-language pre-trained models provide the zero-shot model achieving competitive performance across a range of image classification tasks without requiring training on downstream data. Recent works have confirmed that while additional fine-tuning of the zero-shot model on the reference data results in enhanced downstream performance, it compromises the model's robustness against distribution shifts. Our investigation begins by examining the conditions required to achieve the goals of robust fine-tuning, employing descriptions based on feature distortion theory and joint energy-based models. Subsequently, we propose a novel robust fine-tuning algorithm, Lipsum-FT, that effectively utilizes the language modeling aspect of the vision-language pre-trained models. Extensive experiments conducted on distribution shift scenarios in DomainNet and ImageNet confirm the superiority of our proposed Lipsum-FT approach over existing robust fine-tuning methods.

</details>

---

## 87. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models

- [ ] MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models | https://iclr.cc/virtual/2024/poster/19567

- **Link**: https://iclr.cc/virtual/2024/poster/19567

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent GPT-4 has demonstrated extraordinary multi-modal abilities, such as directly generating websites from handwritten text and identifying humorous elements within images. These features are rarely observed in previous vision-language models. However, the technical details behind GPT-4 continue to remain undisclosed.We believe that the enhanced multi-modal generation capabilities of GPT-4 stem from the utilization of sophisticated large language models (LLM). To examine this phenomenon, we present MiniGPT-4, which aligns a frozen visual encoder with a frozen advanced LLM, Vicuna, using one projection layer. Our work, for the first time, uncovers that properly aligning the visual features with an advanced large language model can possess numerous advanced multi-modal abilities demonstrated by GPT-4, such as detailed image description generation and website creation from hand-drawn drafts.Furthermore, we also observe other emerging capabilities in MiniGPT-4, including writing stories and poems inspired by given images, teaching users how to cook based on food photos, and so on. In our experiment, we found that the model trained on short image caption pairs could produce unnatural language outputs (e.g., repetition and fragmentation). To address this problem, we curate a detailed image description dataset in the second stage to finetune the model, which consequently improves the model's generation reliability and overall usability.

</details>

---

## 88. Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision

- [ ] Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision | https://iclr.cc/virtual/2024/poster/19616

- **Link**: https://iclr.cc/virtual/2024/poster/19616

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid evolution of Multi-modality Large Language Models (MLLMs) has catalyzed a shift in computer vision from specialized models to general-purpose foundation models. Nevertheless, there is still an inadequacy in assessing the abilities of MLLMs on low-level visual perception and understanding . To address this gap, we present Q-Bench , a holistic benchmark crafted to systematically evaluate potential abilities of MLLMs on three realms: low-level visual perception, low-level visual description, and overall visual quality assessment. a) To evaluate the low-level perception ability, we construct the LLVisionQA dataset, consisting of 2,990 diverse-sourced images, each equipped with a human-asked question focusing on its low-level attributes. We then measure the correctness of MLLMs on answering these questions. b) To examine the description ability of MLLMs on low-level information, we propose the LLDescribe dataset consisting of long expert-labelled golden low-level text descriptions on 499 images, and a GPT-involved comparison pipeline between outputs of MLLMs and the golden descriptions. c) Besides these two tasks, we further measure their visual quality assessment ability to align with human opinion scores. Specifically, we design a softmax-based strategy that enables MLLMs to predict quantifiable quality scores, and evaluate them on various existing image quality assessment (IQA) datasets. Our evaluation across the three abilities confirms that MLLMs possess preliminary low-level visual skills. However, these skills are still unstable and relatively imprecise, indicating the need for specific enhancements on MLLMs towards these abilities. We hope that our benchmark can encourage the research community to delve deeper to discover and enhance these untapped potentials of MLLMs.

</details>

---

## 89. Making LLaMA SEE and Draw with SEED Tokenizer

- [ ] Making LLaMA SEE and Draw with SEED Tokenizer | https://iclr.cc/virtual/2024/poster/19618

- **Link**: https://iclr.cc/virtual/2024/poster/19618

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The great success of Large Language Models (LLMs) has expanded the potential of multimodality, contributing to the gradual evolution of General Artificial Intelligence (AGI). A true AGI agent should not only possess the capability to perform predefined multi-tasks but also exhibit emergent abilities in an open-world context. However, despite the considerable advancements made by recent multimodal LLMs, they still fall short in effectively unifying comprehension and generation tasks, let alone open-world emergent abilities. We contend that the key to overcoming the present impasse lies in enabling text and images to be represented and processed interchangeably within a unified autoregressive Transformer. To this end, we introduce $\textbf{SEED}$, an elaborate image tokenizer that empowers LLMs with the ability to $\textbf{SEE}$ and $\textbf{D}$raw at the same time. We identify two crucial design principles: (1) Image tokens should be independent of 2D physical patch positions and instead be produced with a $\textit{1D causal dependency}$, exhibiting intrinsic interdependence that aligns with the left-to-right autoregressive prediction mechanism in LLMs. (2) Image tokens should capture $\textit{high-level semantics}$ consistent with the degree of semantic abstraction in words, and be optimized for both discriminativeness and reconstruction during the tokenizer training phase. With SEED tokens, LLM is able to perform scalable multimodal autoregression under its original training recipe, i.e., next-word prediction. SEED-LLaMA is therefore produced by large-scale pretraining and instruction tuning on the interleaved textual and visual data, demonstrating impressive performance on a broad range of multimodal comprehension and generation tasks. More importantly, SEED-LLaMA has exhibited compositional emergent abilities such as multi-turn in-context multimodal generation, acting like your AI assistant. The code (training and inference) and models are released in https://github.com/AILab-CVC/SEED.

</details>

---

## 90. Improved baselines for vision-language pre-training

- [ ] Improved baselines for vision-language pre-training | https://iclr.cc/virtual/2024/poster/21755

- **Link**: https://iclr.cc/virtual/2024/poster/21755

- **Conference**: ICLR

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Abstract not found

</details>

---

