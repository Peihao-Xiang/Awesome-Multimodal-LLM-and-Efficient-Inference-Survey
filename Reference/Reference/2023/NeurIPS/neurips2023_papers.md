# NeurIPS 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_neurips2023_papers.csv

## 1. On Evaluating Adversarial Robustness of Large Vision-Language Models

- [ ] On Evaluating Adversarial Robustness of Large Vision-Language Models | https://neurips.cc/virtual/2023/poster/69997

- **Link**: https://neurips.cc/virtual/2023/poster/69997

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) such as GPT-4 have achieved unprecedented performance in response generation, especially with visual inputs, enabling more creative and adaptable interaction than large language models such as ChatGPT. Nonetheless, multimodal generation exacerbates safety concerns, since adversaries may successfully evade the entire system by subtly manipulating the most vulnerable modality (e.g., vision). To this end, we propose evaluating the robustness of open-source large VLMs in the most realistic and high-risk setting, where adversaries have only black-box system access and seek to deceive the model into returning the targeted responses. In particular, we first craft targeted adversarial examples against pretrained models such as CLIP and BLIP, and then transfer these adversarial examples to other VLMs such as MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, and Img2Prompt. In addition, we observe that black-box queries on these VLMs can further improve the effectiveness of targeted evasion, resulting in a surprisingly high success rate for generating targeted responses. Our findings provide a quantitative understanding regarding the adversarial vulnerability of large VLMs and call for a more thorough examination of their potential security flaws before deployment in practice. Our project page: https://yunqing-me.github.io/AttackVLM/.

</details>

---

## 2. LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching

- [ ] LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching | https://neurips.cc/virtual/2023/poster/70022

- **Link**: https://neurips.cc/virtual/2023/poster/70022

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Obtaining large pre-trained models that can be fine-tuned to new tasks with limited annotated samples has remained an open challenge for medical imaging data. While pre-trained networks on ImageNet and vision-language foundation models trained on web-scale data are the prevailing approaches, their effectiveness on medical tasks is limited due to the significant domain shift between natural and medical images. To bridge this gap, we introduce LVM-Med, the first family of deep networks trained on large-scale medical datasets. We have collected approximately 1.3 million medical images from 55 publicly available datasets, covering a large number of organs and modalities such as CT, MRI, X-ray, and Ultrasound. We benchmark several state-of-the-art self-supervised algorithms on this dataset and propose a novel self-supervised contrastive learning algorithm using a graph-matching formulation. The proposed approach makes three contributions: (i) it integrates prior pair-wise image similarity metrics based on local and global information; (ii) it captures the structural constraints of feature embeddings through a loss function constructed through a combinatorial graph-matching objective, and (iii) it can be trained efficiently end-to-end using modern gradient-estimation techniques for black-box solvers. We thoroughly evaluate the proposed LVM-Med on 15 downstream medical tasks ranging from segmentation and classification to object detection, and both for the in and out-of-distribution settings. LVM-Med empirically outperforms a number of state-of-the-art supervised, self-supervised, and foundation models. For challenging tasks such as Brain Tumor Classification or Diabetic Retinopathy Grading, LVM-Med improves previous vision-language models trained on 1 billion masks by 6-7%  while using only a ResNet-50.

</details>

---

## 3. Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models

- [ ] Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models | https://neurips.cc/virtual/2023/poster/70046

- **Link**: https://neurips.cc/virtual/2023/poster/70046

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pretrained vision-language models, such as CLIP, have demonstrated strong generalization capabilities, making them promising tools in the realm of zero-shot visual recognition. Visual relation detection (VRD) is a typical task that identifies relationship (or interaction) types between object pairs within an image. However, naively utilizing CLIP with prevalent class-based prompts for zero-shot VRD has several weaknesses, e.g., it struggles to distinguish between different fine-grained relation types and it neglects essential spatial information of two objects. To this end, we propose a novel method for zero-shot VRD: RECODE, which solves RElation detection via COmposite DEscription prompts. Specifically, RECODE first decomposes each predicate category into subject, object, and spatial components. Then, it leverages large language models (LLMs) to generate description-based prompts (or visual cues) for each component. Different visual cues enhance the discriminability of similar relation categories from different perspectives, which significantly boosts performance in VRD. To dynamically fuse different cues, we further introduce a chain-of-thought method that prompts LLMs to generate reasonable weights for different visual cues. Extensive experiments on four VRD benchmarks have demonstrated the effectiveness and interpretability of RECODE.

</details>

---

## 4. Visual Instruction Tuning

- [ ] Visual Instruction Tuning | https://neurips.cc/virtual/2023/poster/70080

- **Link**: https://neurips.cc/virtual/2023/poster/70080

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuning large language models (LLMs) using machine-generated instruction-following data has been shown to improve zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. We present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and an LLM for general-purpose visual and language understanding. To facilitate future research on visual instruction following, we construct two evaluation benchmarks with diverse and challenging application-oriented tasks. Our experiments show that LLaVA demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model, and code publicly available.

</details>

---

## 5. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

- [ ] InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning | https://neurips.cc/virtual/2023/poster/70085

- **Link**: https://neurips.cc/virtual/2023/poster/70085

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-training and instruction tuning have been successful at creating general-purpose language models with broad competence. However, building general-purpose vision-language models is challenging due to the rich input distributions and task diversity resulting from the additional visual input. Although vision-language pretraining has been widely studied, vision-language instruction tuning remains under-explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. We gather 26 publicly available datasets, covering a wide variety of tasks and capabilities, and transform them into instruction tuning format. Additionally, we introduce an instruction-aware Query Transformer, which extracts informative features tailored to the given instruction. Trained on 13 held-in datasets, InstructBLIP attains state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and larger Flamingo models. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA questions with image contexts). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models. All InstructBLIP models are open-source.

</details>

---

## 6. Focus Your Attention when Few-Shot Classification

- [ ] Focus Your Attention when Few-Shot Classification | https://neurips.cc/virtual/2023/poster/70162

- **Link**: https://neurips.cc/virtual/2023/poster/70162

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Since many pre-trained vision transformers emerge and provide strong representation for various downstream tasks, we aim to adapt them to few-shot image classification tasks in this work. The input images typically contain multiple entities. The model may not focus on the class-related entities for the current few-shot task, even with fine-tuning on support samples, and the noise information from the class-independent ones harms performance. To this end, we first propose a method that uses the attention and gradient information to automatically locate the positions of key entities, denoted as position prompts, in the support images. Then we employ the cross-entropy loss between their many-hot presentation and the attention logits to optimize the model to focus its attention on the key entities during fine-tuning. This ability then can generalize to the query samples. Our method is applicable to different vision transformers (e.g., columnar or pyramidal ones), and also to different pre-training ways (e.g., single-modal or vision-language pre-training). Extensive experiments show that our method can improve the performance of full or parameter-efficient fine-tuning methods on few-shot tasks. Code is available at https://github.com/Haoqing-Wang/FORT.

</details>

---

## 7. Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models

- [ ] Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models | https://neurips.cc/virtual/2023/poster/70226

- **Link**: https://neurips.cc/virtual/2023/poster/70226

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recently,  growing interest has been aroused in extending the multimodal capability of large language models (LLMs), e.g., vision-language (VL) learning, which is regarded as the next milestone of artificial general intelligence. However, existing solutions are prohibitively expensive, which not only need to optimize excessive parameters, but also require another large-scale pre-training before VL instruction tuning. In this paper, we propose a novel and  affordable  solution for the effective VL adaption of LLMs, called  Mixture-of-Modality Adaptation (MMA).  Instead of using large neural networks to connect the image encoder and LLM, MMA adopts lightweight modules, i.e., adapters, to  bridge the gap between LLMs and VL tasks, which also enables the joint optimization of the image and language models. Meanwhile, MMA is also equipped with a routing algorithm to help LLMs achieve an  automatic  shift between single- and multi-modal instructions without compromising their ability of natural language understanding.  To validate MMA, we apply it to a recent LLM called LLaMA and term this formed large vision-language instructed model as LaVIN.  To validate MMA and LaVIN, we conduct extensive experiments  under two setups, namely  multimodal science question answering and multimodal dialogue. The experimental results not only demonstrate the competitive performance and the superior training efficiency  of LaVIN  than existing multimodal LLMs, but also confirm  its  great potential   as a general-purpose chatbot. More importantly, the actual expenditure of LaVIN is extremely cheap, e.g., only 1.4 training hours with 3.8M trainable parameters, greatly confirming the effectiveness of MMA.   Our  code is anonymously released at:  https://anonymous.4open.science/r/LaVIN--1067.

</details>

---

## 8. Stable and low-precision training for large-scale vision-language models

- [ ] Stable and low-precision training for large-scale vision-language models | https://neurips.cc/virtual/2023/poster/70245

- **Link**: https://neurips.cc/virtual/2023/poster/70245

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce new methods for 1) accelerating and 2) stabilizing training for large language-vision models. 1) For acceleration, we introduce SwitchBack, a linear layer for int8 quantized training which provides a speed-up of 13-25% while matching the performance of bfloat16 training within 0.1 percentage points for the 1B parameter CLIP ViT-Huge---the largest int8 training to date. Our main focus is int8 as GPU support for float8 is rare, though we also analyze float8 training through simulation. While SwitchBack proves effective for float8, we show that standard techniques are also successful if the network is trained and initialized so that large feature magnitudes are discouraged, which we accomplish via layer-scale initialized with zeros. 2) For stability, we analyze loss spikes and find they consistently occur 1-8 iterations after the squared gradients become under-estimated by their AdamW second moment estimator. As a result, we recommend an AdamW-Adafactor hybrid which avoids loss spikes when training a CLIP ViT-Huge model and outperforms gradient clipping at the scales we test.

</details>

---

## 9. What’s Left? Concept Grounding with Logic-Enhanced Foundation Models

- [ ] What’s Left? Concept Grounding with Logic-Enhanced Foundation Models | https://neurips.cc/virtual/2023/poster/70248

- **Link**: https://neurips.cc/virtual/2023/poster/70248

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent works such as VisProg and ViperGPT have smartly composed foundation models for visual reasoning—using large language models (LLMs) to produce programs that can be executed by pre-trained vision-language models. However, they operate in limited domains, such as 2D images, not fully exploiting the generalization of language: abstract concepts like “ left ” can also be grounded in 3D, temporal, and action data, as in moving to your left . This limited generalization stems from these inference-only methods’ inability to learn or adapt pre-trained models to a new domain. We propose the L ogic- E nhanced F ounda T ion Model ( LEFT ), a unified framework that learns to ground and reason with concepts across domains with a differentiable, domain-independent, first-order logic-based program executor. LEFT has an LLM interpreter that outputs a program represented in a general, logic-based reasoning language, which is shared across all domains and tasks. LEFT’s executor then executes the program with trainable domain-specific grounding modules. We show that LEFT flexibly learns concepts in four domains: 2D images, 3D scenes, human motions, and robotic manipulation. It exhibits strong reasoning ability in a wide variety of tasks, including those that are complex and not seen during training, and can be easily applied to new domains.

</details>

---

## 10. VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models

- [ ] VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models | https://neurips.cc/virtual/2023/poster/70371

- **Link**: https://neurips.cc/virtual/2023/poster/70371

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLATTACK to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multi-modal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multi-modal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level.  We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLATTACK framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models.

</details>

---

## 11. Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models

- [ ] Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models | https://neurips.cc/virtual/2023/poster/70412

- **Link**: https://neurips.cc/virtual/2023/poster/70412

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Human-object interaction (HOI) detection aims to comprehend the intricate relationships between humans and objects, predicting triplets, and serving as the foundation for numerous computer vision tasks. The complexity and diversity of human-object interactions in the real world, however, pose significant challenges for both annotation and recognition, particularly in recognizing interactions within an open world context. This study explores the universal interaction recognition in an open-world setting through the use of Vision-Language (VL) foundation models and large language models (LLMs). The proposed method is dubbed as UniHOI. We conduct a deep analysis of the three hierarchical features inherent in visual HOI detectors and propose a method for high-level relation extraction aimed at VL foundation models, which we call HO prompt-based learning. Our design includes an HO Prompt-guided Decoder (HOPD), facilitates the association of high-level relation representations in the foundation model with various HO pairs within the image. Furthermore, we utilize a LLM (i.e. GPT) for interaction interpretation, generating a richer linguistic understanding for complex HOIs. For open-category interaction recognition, our method supports either of two input types: interaction phrase or interpretive sentence.  Our efficient architecture design and learning methods effectively unleash the potential of the VL foundation models and LLMs, allowing UniHOI to surpass all existing methods with a substantial margin, under both supervised and zero-shot settings. The code and pre-trained weights will be made publicly available.

</details>

---

## 12. CLIP4HOI: Towards Adapting CLIP for Practical Zero-Shot HOI Detection

- [ ] CLIP4HOI: Towards Adapting CLIP for Practical Zero-Shot HOI Detection | https://neurips.cc/virtual/2023/poster/70485

- **Link**: https://neurips.cc/virtual/2023/poster/70485

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot Human-Object Interaction (HOI) detection aims to identify both seen and unseen HOI categories. A strong zero-shot HOI detector is supposed to be not only capable of discriminating novel interactions but also robust to positional distribution discrepancy between seen and unseen categories when locating human-object pairs. However, top-performing zero-shot HOI detectors rely on seen and predefined unseen categories to distill knowledge from CLIP and jointly locate human-object pairs without considering the potential positional distribution discrepancy, leading to impaired transferability. In this paper, we introduce CLIP4HOI, a novel framework for zero-shot HOI detection. CLIP4HOI is developed on the vision-language model CLIP and ameliorates the above issues in the following two aspects. First, to avoid the model from overfitting to the joint positional distribution of seen human-object pairs, we seek to tackle the problem of zero-shot HOI detection in a disentangled two-stage paradigm. To be specific, humans and objects are independently identified and all feasible human-object pairs are processed by Human-Object interactor for pairwise proposal generation. Second, to facilitate better transferability, the CLIP model is elaborately adapted into a fine-grained HOI classifier for proposal discrimination, avoiding data-sensitive knowledge distillation. Finally, experiments on prevalent benchmarks show that our CLIP4HOI outperforms previous approaches on both rare and unseen categories, and sets a series of state-of-the-art records under a variety of zero-shot settings.

</details>

---

## 13. Symbolic Discovery of Optimization Algorithms

- [ ] Symbolic Discovery of Optimization Algorithms | https://neurips.cc/virtual/2023/poster/70492

- **Link**: https://neurips.cc/virtual/2023/poster/70492

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training. We leverage efficient search techniques to explore an infinite and sparse program space. To bridge the large generalization gap between proxy and target tasks, we also introduce program selection and simplification strategies.Our method discovers a simple and effective optimization algorithm, $\textbf{Lion}$ ($\textit{Evo$\textbf{L}$ved S$\textbf{i}$gn M$\textbf{o}$me$\textbf{n}$tum}$). It is more memory-efficient than Adam as it only keeps track of the momentum. Different from adaptive optimizers, its update has the same magnitude for each parameter calculated through the sign operation.We compare Lion with widely used optimizers, such as Adam and Adafactor, for training a variety of models on different tasks. On image classification, Lion boosts the accuracy of ViT by up to 2\% on ImageNet and saves up to 5x the pre-training compute on JFT. On vision-language contrastive learning, we achieve 88.3\% $\textit{zero-shot}$ and 91.1\% $\textit{fine-tuning}$ accuracy on ImageNet, surpassing the previous best results by 2\% and 0.1\%, respectively. On diffusion models, Lion outperforms Adam by achieving a better FID score and reducing the training compute by up to 2.3x. For autoregressive, masked language modeling, and fine-tuning, Lion exhibits a similar or better performance compared to Adam. Our analysis of Lion reveals that its performance gain grows with the training batch size. It also requires a smaller learning rate than Adam due to the larger norm of the update produced by the sign function. Additionally, we examine the limitations of Lion and identify scenarios where its improvements are small or not statistically significant.

</details>

---

## 14. The Rise of AI Language Pathologists: Exploring Two-level Prompt Learning for Few-shot Weakly-supervised Whole Slide Image Classification

- [ ] The Rise of AI Language Pathologists: Exploring Two-level Prompt Learning for Few-shot Weakly-supervised Whole Slide Image Classification | https://neurips.cc/virtual/2023/poster/70552

- **Link**: https://neurips.cc/virtual/2023/poster/70552

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces the novel concept of few-shot weakly supervised learning for pathology Whole Slide Image (WSI) classification, denoted as FSWC. A solution is proposed based on prompt learning and the utilization of a large language model, GPT-4. Since a WSI is too large and needs to be divided into patches for processing, WSI classification is commonly approached as a Multiple Instance Learning (MIL) problem. In this context, each WSI is considered a bag, and the obtained patches are treated as instances. The objective of FSWC is to classify both bags and instances with only a limited number of labeled bags. Unlike conventional few-shot learning problems, FSWC poses additional challenges due to its weak bag labels within the MIL framework. Drawing inspiration from the recent achievements of vision-language models (V-L models) in downstream few-shot classification tasks, we propose a two-level prompt learning MIL framework tailored for pathology, incorporating language prior knowledge. Specifically, we leverage CLIP to extract instance features for each patch, and introduce a prompt-guided pooling strategy to aggregate these instance features into a bag feature. Subsequently, we employ a small number of labeled bags to facilitate few-shot prompt learning based on the bag features. Our approach incorporates the utilization of GPT-4 in a question-and-answer mode to obtain language prior knowledge at both the instance and bag levels, which are then integrated into the instance and bag level language prompts. Additionally, a learnable component of the language prompts is trained using the available few-shot labeled data. We conduct extensive experiments on three real WSI datasets encompassing breast cancer, lung cancer, and cervical cancer, demonstrating the notable performance of the proposed method in bag and instance classification. All codes will be made publicly accessible.

</details>

---

## 15. Scaling Open-Vocabulary Object Detection

- [ ] Scaling Open-Vocabulary Object Detection | https://neurips.cc/virtual/2023/poster/70553

- **Link**: https://neurips.cc/virtual/2023/poster/70553

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary object detection has benefited greatly from pretrained vision-language models, but is still limited by the amount of available detection training data. While detection training data can be expanded by using Web image-text pairs as weak supervision, this has not been done at scales comparable to image-level pretraining. Here, we scale up detection data with self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. Major challenges in scaling self-training are the choice of label space, pseudo-annotation filtering, and training efficiency. We present the OWLv2 model and OWL-ST self-training recipe, which address these challenges. OWLv2 surpasses the performance of previous state-of-the-art open-vocabulary detectors already at comparable training scales (~10M examples). However, with OWL-ST, we can scale to over 1B examples, yielding further large improvement: With an L/14 architecture, OWL-ST improves AP on LVIS rare classes, for which the model has seen no human box annotations, from 31.2% to 44.6% (43% relative improvement). OWL-ST unlocks Web-scale training for open-world localization, similar to what has been seen for image classification and language modelling. Code and checkpoints are available on GitHub.

</details>

---

## 16. StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

- [ ] StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models | https://neurips.cc/virtual/2023/poster/70566

- **Link**: https://neurips.cc/virtual/2023/poster/70566

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs. The audio demos and source code are available at https://styletts2.github.io/.

</details>

---

## 17. Fine-Grained Visual Prompting

- [ ] Fine-Grained Visual Prompting | https://neurips.cc/virtual/2023/poster/70615

- **Link**: https://neurips.cc/virtual/2023/poster/70615

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs), such as CLIP, have demonstrated impressive zero-shot transfer capabilities in image-level visual perception. However, these models have shown limited performance in instance-level tasks that demand precise localization and recognition. Previous works have suggested that incorporating visual prompts, such as colorful boxes or circles, can improve the ability of models to recognize objects of interest. Nonetheless, compared to language prompting, visual prompting designs are rarely explored. Existing approaches, which employ coarse visual cues such as colorful boxes or circles, often result in sub-optimal performance due to the inclusion of irrelevant and noisy pixels. In this paper, we carefully study the visual prompting designs by exploring more fine-grained markings, such as segmentation masks and their variations. In addition, we introduce a new zero-shot framework that leverages pixel-level annotations acquired from a generalist segmentation model for fine-grained visual prompting. Consequently, our investigation reveals that a straightforward application of blur outside the target mask, referred to as the Blur Reverse Mask, exhibits exceptional effectiveness. This proposed prompting strategy leverages the precise mask annotations to reduce focus on weakly related regions while retaining spatial coherence between the target and the surrounding background. Our F ine- G rained V isual P rompting ( FGVP ) demonstrates superior performance in zero-shot comprehension of referring expressions on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks. It outperforms prior methods by an average margin of 3.0\% to 4.6\%, with a maximum improvement of 12.5\% on the RefCOCO+ testA subset. The part detection experiments conducted on the PACO dataset further validate the preponderance of FGVP over existing visual prompting techniques. Code is available at https://github.com/ylingfeng/FGVP.

</details>

---

## 18. Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition

- [ ] Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition | https://neurips.cc/virtual/2023/poster/70637

- **Link**: https://neurips.cc/virtual/2023/poster/70637

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This work proposes POMP, a prompt pre-training method for vision-language models. Being memory and computation efficient, POMP enables the learned prompt to condense semantic information for a rich set of visual concepts with over twenty-thousand classes. Once pre-trained, the prompt with a strong transferable ability can be directly plugged into a variety of visual recognition tasks including image classification, semantic segmentation, and object detection, to boost recognition performances in a zero-shot manner. Empirical evaluation shows that POMP achieves state-of-the-art performances on 21 datasets, e.g., 67.0% average accuracy on 10 classification datasets (+3.1% compared to CoOp) and 84.4 hIoU on open-vocabulary Pascal VOC segmentation (+6.9 compared to ZSSeg).

</details>

---

## 19. What You See is What You Read? Improving Text-Image Alignment Evaluation

- [ ] What You See is What You Read? Improving Text-Image Alignment Evaluation | https://neurips.cc/virtual/2023/poster/70707

- **Link**: https://neurips.cc/virtual/2023/poster/70707

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Automatically determining whether a text and a corresponding image are semantically aligned is a significant challenge for vision-language models, with applications in generative text-to-image and image-to-text tasks. In this work, we study methods for automatic text-image alignment evaluation. We first introduce SeeTRUE: a comprehensive evaluation set, spanning multiple datasets from both text-to-image and image-to-text generation tasks, with human judgements for whether a given text-image pair is semantically aligned. We then describe two automatic methods to determine alignment: the first involving a pipeline based on question generation and visual question answering models, and the second employing an end-to-end classification approach by finetuning multimodal pretrained models. Both methods surpass prior approaches in various text-image alignment tasks, with significant improvements in challenging cases that involve complex composition or unnatural images. Finally, we demonstrate how our approaches can localize specific misalignments between an image and a given text, and how they can be used to automatically re-rank candidates in text-to-image generation.

</details>

---

## 20. Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models

- [ ] Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models | https://neurips.cc/virtual/2023/poster/70716

- **Link**: https://neurips.cc/virtual/2023/poster/70716

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We propose a conceptually simple and lightweight framework for improving the robustness of vision models through the combination of knowledge distillation and data augmentation. We address the conjecture that larger models do not make for better teachers by showing strong gains in out-of-distribution robustness when distilling from pretrained foundation models. Following this finding, we propose Discrete Adversarial Distillation (DAD), which leverages a robust teacher to generate adversarial examples and a VQGAN to discretize them, creating more informative samples than standard data augmentation techniques. We provide a theoretical framework for the use of a robust teacher in the knowledge distillation with data augmentation setting and demonstrate strong gains in out-of-distribution robustness and clean accuracy across different student architectures. Notably, our method adds minor computational overhead compared to similar techniques and can be easily combined with other data augmentations for further improvements.

</details>

---

## 21. Parts of Speech–Grounded Subspaces in Vision-Language Models

- [ ] Parts of Speech–Grounded Subspaces in Vision-Language Models | https://neurips.cc/virtual/2023/poster/70789

- **Link**: https://neurips.cc/virtual/2023/poster/70789

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Latent image representations arising from vision-language models have proved immensely useful for a variety of downstream tasks. However, their utility is limited by their entanglement with respect to different visual attributes. For instance, recent work has shown that CLIP image representations are often biased toward specific visual properties (such as objects or actions) in an unpredictable manner. In this paper, we propose to separate representations of the different visual modalities in CLIP’s joint vision-language space by leveraging the association between parts of speech and specific visual modes of variation (e.g. nouns relate to objects, adjectives describe appearance). This is achieved by formulating an appropriate component analysis model that learns subspaces capturing variability corresponding to a specific part of speech, while jointly minimising variability to the rest. Such a subspace yields disentangled representations of the different visual properties of an image or text in closed form while respecting the underlying geometry of the manifold on which the representations lie. What’s more, we show the proposed model additionally facilitates learning subspaces corresponding to specific visual appearances (e.g. artists’ painting styles), which enables the selective removal of entire visual themes from CLIP-based text-to-image synthesis. We validate the model both qualitatively, by visualising the subspace projections with a text-to-image model and by preventing the imitation of artists’ styles, and quantitatively, through class invariance metrics and improvements to baseline zero-shot classification.

</details>

---

## 22. Text-to-Image Diffusion Models are Zero Shot Classifiers

- [ ] Text-to-Image Diffusion Models are Zero Shot Classifiers | https://neurips.cc/virtual/2023/poster/70878

- **Link**: https://neurips.cc/virtual/2023/poster/70878

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The excellent generative capabilities of text-to-image diffusion models suggest they learn informative representations of image-text data.However, what knowledge their representations capture is not fully understood, and they have not been thoroughly explored on downstream tasks.We investigate diffusion models by proposing a method for evaluating them as zero-shot classifiers.The key idea is using a diffusion model's ability to denoise a noised image given a text description of a label as a proxy for that label's likelihood.We apply our method to Stable Diffusion and Imagen, using it to probe fine-grained aspects of the models' knowledge and comparing them with CLIP's zero-shot abilities. They perform competitively with CLIP on a wide range of zero-shot image classification datasets. Additionally, they achieve state-of-the-art results on shape/texture bias tests and can successfully perform attribute binding while CLIP cannot.Although generative pre-training is prevalent in NLP, visual foundation models often use other methods such as contrastive learning. Based on our findings, we argue that generative pre-training should be explored as a compelling alternative for vision and vision-language problems.

</details>

---

## 23. Learning Domain-Aware Detection Head with Prompt Tuning

- [ ] Learning Domain-Aware Detection Head with Prompt Tuning | https://neurips.cc/virtual/2023/poster/70904

- **Link**: https://neurips.cc/virtual/2023/poster/70904

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Domain adaptive object detection (DAOD) aims to generalize detectors trained on an annotated source domain to an unlabelled target domain.  However, existing methods focus on reducing the domain bias of the detection backbone by inferring a discriminative visual encoder, while ignoring the domain bias in the detection head.  Inspired by the high generalization of vision-language models (VLMs), applying a VLM as the robust detection backbone following a domain-aware detection head is a reasonable way to learn the discriminative detector for each domain, rather than reducing the domain bias in traditional methods.  To achieve the above issue, we thus propose a novel DAOD framework named Domain-Aware detection head with Prompt tuning (DA-Pro), which applies the learnable domain-adaptive prompt to generate the dynamic detection head for each domain.   Formally, the domain-adaptive prompt consists of the domain-invariant tokens, domain-specific tokens, and the domain-related textual description along with the class label.   Furthermore, two constraints between the source and target domains are applied to ensure that the domain-adaptive prompt can capture the domains-shared and domain-specific knowledge.  A prompt ensemble strategy is also proposed to reduce the effect of prompt disturbance.   Comprehensive experiments over multiple cross-domain adaptation tasks demonstrate that using the domain-adaptive prompt can produce an effectively domain-related detection head for boosting domain-adaptive object detection.  Our code is available at https://github.com/Therock90421/DA-Pro.

</details>

---

## 24. DiffVL: Scaling Up Soft Body Manipulation using Vision-Language Driven Differentiable Physics

- [ ] DiffVL: Scaling Up Soft Body Manipulation using Vision-Language Driven Differentiable Physics | https://neurips.cc/virtual/2023/poster/70947

- **Link**: https://neurips.cc/virtual/2023/poster/70947

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Combining gradient-based trajectory optimization with differentiable physics simulation is an efficient technique for solving soft-body manipulation problems.Using a well-crafted optimization objective, the solver can quickly converge onto a valid trajectory.However, writing the appropriate objective functions requires expert knowledge, making it difficult to collect a large set of naturalistic problems from non-expert users.We introduce DiffVL, a method that enables non-expert users to communicate soft-body manipulation tasks -- a combination of vision and natural language, given in multiple stages -- that can be readily leveraged by a differential physics solver. We have developed GUI tools that enable non-expert users to specify 100 tasks inspired by real-life soft-body manipulations from online videos, which we'll make public.We leverage large language models to translate task descriptions into machine-interpretable optimization objectives. The optimization objectives can help differentiable physics solvers to solve these long-horizon multistage tasks that are challenging for previous baselines.

</details>

---

## 25. POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images

- [ ] POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images | https://neurips.cc/virtual/2023/poster/70977

- **Link**: https://neurips.cc/virtual/2023/poster/70977

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We describe an approach to predict open-vocabulary 3D semantic voxel occupancy map from input 2D images with the objective of enabling 3D grounding, segmentation and retrieval of free-form language queries. This is a challenging problem because of the 2D-3D ambiguity and the open-vocabulary nature of the target tasks, where obtaining annotated training data in 3D is difficult. The contributions of this work are three-fold. First, we design a new model architecture for open-vocabulary 3D semantic occupancy prediction.  The architecture consists of a 2D-3D encoder together with occupancy prediction and 3D-language heads. The output is a dense voxel map of 3D grounded language embeddings enabling a range of open-vocabulary tasks. Second, we develop a tri-modal self-supervised learning algorithm that leverages three modalities: (i) images, (ii) language and (iii) LiDAR point clouds, and enables training the proposed architecture using a strong pre-trained vision-language model without the need for any 3D manual language annotations. Finally, we demonstrate quantitatively the strengths of the proposed model on several open-vocabulary tasks:Zero-shot 3D semantic segmentation using existing datasets; 3D grounding and retrieval of free-form language queries, using a small dataset that we propose as an extension of nuScenes. You can find the project page here https://vobecant.github.io/POP3D.

</details>

---

## 26. Exploring Diverse In-Context Configurations for Image Captioning

- [ ] Exploring Diverse In-Context Configurations for Image Captioning | https://neurips.cc/virtual/2023/poster/71057

- **Link**: https://neurips.cc/virtual/2023/poster/71057

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

After discovering that Language Models (LMs) can be good in-context few-shot learners, numerous strategies have been proposed to optimize in-context sequence configurations. Recently, researchers in Vision-Language (VL) domains also develop their few-shot learners, while they only use the simplest way, \ie, randomly sampling, to configure in-context image-text pairs. In order to explore the effects of varying configurations on VL in-context learning, we devised four strategies for image selection and four for caption assignment to configure in-context image-text pairs for image captioning. Here Image Captioning is used as the case study since it can be seen as the visually-conditioned LM. Our comprehensive experiments yield two counter-intuitive but valuable insights, highlighting the distinct characteristics of VL in-context learning due to multi-modal synergy, as compared to the NLP case. Furthermore, in our exploration of optimal combination strategies, we observed an average performance enhancement of 20.9 in CIDEr scores compared to the baseline. The code is given in https://github.com/yongliang-wu/ExploreCfg.

</details>

---

## 27. Self-Chained Image-Language Model for Video Localization and Question Answering

- [ ] Self-Chained Image-Language Model for Video Localization and Question Answering | https://neurips.cc/virtual/2023/poster/71124

- **Link**: https://neurips.cc/virtual/2023/poster/71124

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have shown promising results on utilizing large pre-trained image-language models for video question answering. While these image-language models can efficiently bootstrap the representation learning of video-language models, they typically concatenate uniformly sampled video frames as visual inputs without explicit language-aware, temporal modeling. When only a portion of a video input is relevant to the language query, such uniform frame sampling can often lead to missing important visual cues. Although humans often find a video moment to focus on and rewind the moment to answer questions, training a query-aware video moment localizer often requires expensive annotations and high computational costs. To address this issue, we propose Self-Chained Video Localization-Answering (SeViLA), a novel framework that leverages a single image-language model (BLIP- 2) to tackle both temporal keyframe localization and question answering on videos. SeViLA framework consists of two modules: Localizer and Answerer, where both are parameter-efficiently fine-tuned from BLIP-2. We propose two ways of chaining these modules for cascaded inference and self-refinement. First, in the forward chain, the Localizer finds multiple language-aware keyframes in a video, which the Answerer uses to predict the answer. Second, in the reverse chain, the Answerer generates keyframe pseudo-labels to refine the Localizer, alleviating the need for expensive video moment localization annotations. Our SeViLA framework outperforms several strong baselines/previous works on five challenging video question answering and event prediction benchmarks, and achieves the state-of-the-art in both fine-tuning (NExT-QA and STAR) and zero-shot (NExT-QA, STAR, How2QA, and VLEP) settings. We show a comprehensive analysis of our framework, including the impact of Localizer, comparisons of Localizer with other temporal localization models, pre-training/self-refinement of Localizer, and varying the number of keyframes.

</details>

---

## 28. Active Reasoning in an Open-World Environment

- [ ] Active Reasoning in an Open-World Environment | https://neurips.cc/virtual/2023/poster/71129

- **Link**: https://neurips.cc/virtual/2023/poster/71129

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language learning have achieved notable success on complete-information question-answering datasets through the integration of extensive world knowledge. Yet, most models operate passively , responding to questions based on pre-stored knowledge. In stark contrast, humans possess the ability to actively explore, accumulate, and reason using both newfound and existing information to tackle incomplete-information questions. In response to this gap, we introduce Conan , an interactive open-world environment devised for the assessment of active reasoning . Conan facilitates active exploration and promotes multi-round abductive inference, reminiscent of rich, open-world settings like Minecraft. Diverging from previous works that lean primarily on single-round deduction via instruction following, Conan compels agents to actively interact with their surroundings, amalgamating new evidence with prior knowledge to elucidate events from incomplete observations. Our analysis on \bench underscores the shortcomings of contemporary state-of-the-art models in active exploration and understanding complex scenarios. Additionally, we explore Abduction from Deduction , where agents harness Bayesian rules to recast the challenge of abduction as a deductive process. Through Conan , we aim to galvanize advancements in active reasoning and set the stage for the next generation of artificial intelligence agents adept at dynamically engaging in environments.

</details>

---

## 29. Paxion: Patching Action Knowledge in Video-Language Foundation Models

- [ ] Paxion: Patching Action Knowledge in Video-Language Foundation Models | https://neurips.cc/virtual/2023/poster/71132

- **Link**: https://neurips.cc/virtual/2023/poster/71132

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Action knowledge involves the understanding of textual, visual, and temporal aspects of actions. We introduce the Action Dynamics Benchmark (ActionBench) containing two carefully designed probing tasks: Action Antonym and Video Reversal, which targets multimodal alignment capabilities and temporal understanding skills of the model, respectively. Despite recent video-language models’ (VidLM) impressive performance on various benchmark tasks, our diagnostic tasks reveal their surprising deficiency (near-random performance) in action knowledge, suggesting that current models rely on object recognition abilities as a shortcut for action understanding. To remedy this, we propose a novel framework, Paxion , along with a new Discriminative Video Dynamics Modeling (DVDM) objective. The Paxion framework utilizes a Knowledge Patcher network to encode new action knowledge and a Knowledge Fuser component to integrate the Patcher into frozen VidLMs without compromising their existing capabilities. Due to limitations of the widely-used Video-Text Contrastive (VTC) loss for learning action knowledge, we introduce the DVDM objective to train the Knowledge Patcher. DVDM forces the model to encode the correlation between the action text and the correct ordering of video frames. Our extensive analyses show that Paxion and DVDM together effectively fill the gap in action knowledge understanding (~50% → 80%), while maintaining or improving performance on a wide spectrum of both object- and action-centric downstream tasks.

</details>

---

## 30. Text Promptable Surgical Instrument Segmentation with Vision-Language Models

- [ ] Text Promptable Surgical Instrument Segmentation with Vision-Language Models | https://neurips.cc/virtual/2023/poster/71267

- **Link**: https://neurips.cc/virtual/2023/poster/71267

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose a novel text promptable surgical instrument segmentation approach to overcome challenges associated with diversity and differentiation of surgical instruments in minimally invasive surgeries. We redefine the task as text promptable, thereby enabling a more nuanced comprehension of surgical instruments and adaptability to new instrument types. Inspired by recent advancements in vision-language models, we leverage pretrained image and text encoders as our model backbone and design a text promptable mask decoder consisting of attention- and convolution-based prompting schemes for surgical instrument segmentation prediction. Our model leverages multiple text prompts for each surgical instrument through a new mixture of prompts mechanism, resulting in enhanced segmentation performance. Additionally, we introduce a hard instrument area reinforcement module to improve image feature comprehension and segmentation precision. Extensive experiments on several surgical instrument segmentation datasets demonstrate our model's superior performance and promising generalization capability. To our knowledge, this is the first implementation of a promptable approach to surgical instrument segmentation, offering significant potential for practical application in the field of robotic-assisted surgery. Code is available at https://github.com/franciszzj/TP-SIS.

</details>

---

## 31. GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph

- [ ] GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph | https://neurips.cc/virtual/2023/poster/71275

- **Link**: https://neurips.cc/virtual/2023/poster/71275

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Adapter-style efficient transfer learning (ETL) has shown excellent performance in the tuning of vision-language models (VLMs) under the low-data regime, where only a few additional parameters are introduced to excavate the task-specific knowledge based on the general and powerful representation of VLMs. However, most adapter-style works face two limitations: (i) modeling task-specific knowledge with a single modality only; and (ii) overlooking the exploitation of the inter-class relationships in downstream tasks, thereby leading to sub-optimal solutions. To mitigate that, we propose an effective adapter-style tuning strategy, dubbed GraphAdapter, which performs the textual adapter by explicitly modeling the dual-modality structure knowledge (i.e., the correlation of different semantics/classes in textual and visual modalities) with a dual knowledge graph. In particular, the dual knowledge graph is established with two sub-graphs, i.e., a textual knowledge sub-graph, and a visual knowledge sub-graph, where the nodes and edges represent the semantics/classes and their correlations in two modalities, respectively. This enables the textual feature of each prompt to leverage the task-specific structure knowledge from both textual and visual modalities, yielding a more effective classifier for downstream tasks. Extensive experimental results on 11 benchmark datasets reveal that our GraphAdapter significantly outperforms the previous adapter-based methods.

</details>

---

## 32. 3D-LLM: Injecting the 3D World into Large Language Models

- [ ] 3D-LLM: Injecting the 3D World into Large Language Models | https://neurips.cc/virtual/2023/poster/71298

- **Link**: https://neurips.cc/virtual/2023/poster/71298

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) and Vision-Language Models (VLMs) have been proved to excel at multiple tasks, such as commonsense reasoning. Powerful as these models can be, they are not grounded in the 3D physical world, which involves richer concepts such as spatial relationships, affordances, physics, layout, and so on. In this work, we propose to inject the 3D world into large language models, and introduce a whole new family of 3D-LLMs. Specifically, 3D-LLMs can take 3D point clouds and their features as input and perform a diverse set of 3D-related tasks, including captioning, dense captioning, 3D question answering, task decomposition, 3Dgrounding, 3D-assisted dialog, navigation, and so on. Using three types of prompting mechanisms that we design, we are able to collect over 300k 3D-language data covering these tasks. To efficiently train 3D-LLMs, we first utilize a 3D feature extractor that obtains 3D features from rendered multi-view images. Then, we use 2D VLMs as our backbones to train our 3D-LLMs. By introducing a 3D localization mechanism, 3D-LLMs could better capture 3D spatial information.  Experiments on ScanQA  show that our model outperforms state-of-the-art baselines by a large margin (\textit{e.g.}, the BLEU-1 score surpasses state-of-the-art score by 9\%). Furthermore, experiments on our held-in datasets for 3D captioning, task composition, and 3D-assisted dialogue show that our model outperforms 2D VLMs. Qualitative examples also show that our model could perform more tasks beyond the scope of existing LLMs and VLMs. Our model and data will be publicly available.

</details>

---

## 33. Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification

- [ ] Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification | https://neurips.cc/virtual/2023/poster/71353

- **Link**: https://neurips.cc/virtual/2023/poster/71353

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present a novel language-driven ordering alignment method for ordinal classification. The labels in ordinal classification contain additional ordering relations, making them prone to overfitting when relying solely on training data. Recent developments in pre-trained vision-language models inspire us to leverage the rich ordinal priors in human language by converting the original task into a vision-language alignment task. Consequently, we propose L2RCLIP, which fully utilizes the language priors from two perspectives. First, we introduce a complementary prompt tuning technique called RankFormer, designed to enhance the ordering relation of original rank prompts. It employs token-level attention with residual-style prompt blending in the word embedding space. Second, to further incorporate language priors, we revisit the approximate bound optimization of vanilla cross-entropy loss and restructure it within the cross-modal embedding space. Consequently, we propose a cross-modal ordinal pairwise loss to refine the CLIP feature space, where texts and images maintain both semantic alignment and ordering alignment.  Extensive experiments on three ordinal classification tasks, including facial age estimation, historical color image (HCI) classification, and aesthetic assessment demonstrate its promising performance.

</details>

---

## 34. UP-DP: Unsupervised Prompt Learning for Data Pre-Selection with Vision-Language Models

- [ ] UP-DP: Unsupervised Prompt Learning for Data Pre-Selection with Vision-Language Models | https://neurips.cc/virtual/2023/poster/71462

- **Link**: https://neurips.cc/virtual/2023/poster/71462

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this study, we investigate the task of data pre-selection, which aims to select instances for labeling from an unlabeled dataset through a single pass, thereby optimizing performance for undefined downstream tasks with a limited annotation budget. Previous approaches to data pre-selection relied solely on visual features extracted from foundation models, such as CLIP and BLIP-2, but largely ignored the powerfulness of text features. In this work, we argue that, with proper design, the joint feature space of both vision and text can yield a better representation for data pre-selection. To this end, we introduce UP-DP, a simple yet effective unsupervised prompt learning approach that adapts vision-language models, like BLIP-2, for data pre-selection. Specifically, with the BLIP-2 parameters frozen, we train text prompts to extract the joint features with improved representation, ensuring a diverse cluster structure that covers the entire dataset. We extensively compare our method with the state-of-the-art using seven benchmark datasets in different settings, achieving up to a performance gain of 20\%. Interestingly, the prompts learned from one dataset demonstrate significant generalizability and can be applied directly to enhance the feature extraction of BLIP-2 from other datasets. To the best of our knowledge, UP-DP is the first work to incorporate unsupervised prompt learning in a vision-language model for data pre-selection.

</details>

---

## 35. Localized Symbolic Knowledge Distillation for Visual Commonsense Models

- [ ] Localized Symbolic Knowledge Distillation for Visual Commonsense Models | https://neurips.cc/virtual/2023/poster/71475

- **Link**: https://neurips.cc/virtual/2023/poster/71475

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Instruction following vision-language (VL) models offer a flexibleinterface that supports a broad range of multimodal tasks in a zero-shot fashion.However, interfaces that operate on full images do not directly enable the user to“point to" and access specific regions within images. This capability is importantnot only to support reference-grounded VL benchmarks, but also, for practicalapplications that require precise within-image reasoning. We build LocalizedVisual Commonsense model which allows users to specify (multiple) regions-as-input. We train our model by sampling localized commonsense knowledgefrom a large language model (LLM): specifically, we prompt a LLM to collectcommonsense knowledge given a global literal image description and a localliteral region description automatically generated by a set of VL models. Thispipeline is scalable and fully automatic, as no aligned or human-authored imageand text pairs are required. With a separately trained critic model that selectshigh quality examples, we find that training on the localized commonsense corpusexpanded solely from images can successfully distill existing VL models to supporta reference-as-input interface. Empirical results and human evaluations in zero-shotsettings demonstrate that our distillation method results in more precise VL modelsof reasoning compared to a baseline of passing a generated referring expression.

</details>

---

## 36. Language Is Not All You Need: Aligning Perception with Language Models

- [ ] Language Is Not All You Need: Aligning Perception with Language Models | https://neurips.cc/virtual/2023/poster/71488

- **Link**: https://neurips.cc/virtual/2023/poster/71488

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

A big convergence of language, multimodal perception, action, and world modeling is a key step toward artificial general intelligence. In this work, we introduce KOSMOS-1, a Multimodal Large Language Model (MLLM) that can perceive general modalities, learn in context (i.e., few-shot), and follow instructions (i.e., zero-shot). Specifically, we train KOSMOS-1 from scratch on web-scale multi-modal corpora, including arbitrarily interleaved text and images, image-caption pairs, and text data. We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning. Experimental results show that KOSMOS-1 achieves impressive performance on (i) language understanding, generation, and even OCR-free NLP (directly fed with document images), (ii) perception-language tasks, including multimodal dialogue, image captioning, visual question answering, and (iii) vision tasks, such as image recognition with descriptions (specifying classification via text instructions). We also show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge from language to multimodal, and from multimodal to language. In addition, we introduce a dataset of Raven IQ test, which diagnoses the nonverbal reasoning capability of MLLMs.

</details>

---

## 37. LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning

- [ ] LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning | https://neurips.cc/virtual/2023/poster/71493

- **Link**: https://neurips.cc/virtual/2023/poster/71493

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present a novel vision-language prompt learning approach for few-shot out-of-distribution (OOD) detection. Few-shot OOD detection aims to detect OOD images from classes that are unseen during training using only a few labeled in-distribution (ID) images. While prompt learning methods such as CoOp have shown effectiveness and efficiency in few-shot ID classification, they still face limitations in OOD detection due to the potential presence of ID-irrelevant information in text embeddings. To address this issue, we introduce a new approach called $\textbf{Lo}$cal regularized $\textbf{Co}$ntext $\textbf{Op}$timization (LoCoOp), which performs OOD regularization that utilizes the portions of CLIP local features as OOD features during training. CLIP's local features have a lot of ID-irrelevant nuisances ($\textit{e.g.}$, backgrounds), and by learning to push them away from the ID class text embeddings, we can remove the nuisances in the ID class text embeddings and enhance the separation between ID and OOD. Experiments on the large-scale ImageNet OOD detection benchmarks demonstrate the superiority of our LoCoOp over zero-shot, fully supervised detection methods and prompt learning methods. Notably, even in a one-shot setting -- just one label per class, LoCoOp outperforms existing zero-shot and fully supervised detection methods. The code is available via https://github.com/AtsuMiyai/LoCoOp.

</details>

---

## 38. Generating Images with Multimodal Language Models

- [ ] Generating Images with Multimodal Language Models | https://neurips.cc/virtual/2023/poster/71499

- **Link**: https://neurips.cc/virtual/2023/poster/71499

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We propose a method to fuse frozen text-only large language models (LLMs) with pre-trained image encoder and decoder models, by mapping between their embedding spaces. Our model demonstrates a wide suite of multimodal capabilities: image retrieval, novel image generation, and multimodal dialogue. Ours is the first approach capable of conditioning on arbitrarily interleaved image and text inputs to generate coherent image (and text) outputs. To achieve strong performance on image generation, we propose an efficient mapping network to ground the LLM to an off-the-shelf text-to-image generation model. This mapping network translates hidden representations of text into the embedding space of the visual models, enabling us to leverage the strong text representations of the LLM for visual outputs. Our approach outperforms baseline generation models on tasks with longer and more complex language. In addition to novel image generation, our model is also capable of image retrieval from a prespecified dataset, and decides whether to retrieve or generate at inference time. This is done with a learnt decision module which conditions on the hidden representations of the LLM. Our model exhibits a wider range of capabilities compared to prior multimodal language models. It can process image-and-text inputs, and produce retrieved images, generated images, and generated text — outperforming non-LLM based generation models across several text-to-image tasks that measure context dependence.

</details>

---

## 39. Meta-Adapter: An Online Few-shot Learner for Vision-Language Model

- [ ] Meta-Adapter: An Online Few-shot Learner for Vision-Language Model | https://neurips.cc/virtual/2023/poster/71536

- **Link**: https://neurips.cc/virtual/2023/poster/71536

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The contrastive vision-language pre-training, known as CLIP, demonstrates remarkable potential in perceiving open-world visual concepts, enabling effective zero-shot image recognition.  Nevertheless, few-shot learning methods based on CLIP typically require offline fine-tuning of the parameters on few-shot samples, resulting in longer inference time and the risk of overfitting in certain domains.  To tackle these challenges, we propose the Meta-Adapter, a lightweight residual-style adapter, to refine the CLIP features guided by the few-shot samples in an online manner.  With a few training samples, our method can enable effective few-shot learning capabilities and generalize to unseen data or tasks without additional fine-tuning, achieving competitive performance and high efficiency.  Without bells and whistles, our approach outperforms the state-of-the-art online few-shot learning method by an average of 3.6\% on eight image classification datasets with higher inference speed.  Furthermore, our model is simple and flexible, serving as a plug-and-play module directly applicable to downstream tasks.  Without further fine-tuning, Meta-Adapter obtains notable performance improvements in open-vocabulary object detection and segmentation tasks.

</details>

---

## 40. Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models

- [ ] Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models | https://neurips.cc/virtual/2023/poster/71562

- **Link**: https://neurips.cc/virtual/2023/poster/71562

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

With ever increasing parameters and computation, vision-language pre-trained (VLP) models exhibit prohibitive expenditure in downstream task adaption. Recent endeavors mainly focus on parameter efficient transfer learning (PETL) for VLP models by only updating a small number of parameters. However, excessive computational overhead still plagues the application of VLPs. In this paper, we aim at parameter and computation efficient transfer learning (PCETL) for VLP models. In particular, PCETL not only needs to limit the number of trainable parameters in VLP models, but also to reduce the computational redundancy during inference, thus enabling a more efficient transfer. To approach this target, we propose a novel dynamic architecture skipping (DAS) approach towards effective PCETL. Instead of directly optimizing the intrinsic architectures of VLP models, DAS first observes the significances of their modules to downstream tasks via a reinforcement learning (RL) based process, and then skips the redundant ones with lightweight networks, i.e. adapters, according to the obtained rewards. In this case, the VLP model can well maintain the scale of trainable parameters while speeding up its inference on downstream tasks. To validate DAS, we apply it to two representative VLP models, namely ViLT and METER, and conduct extensive experiments on a bunch of VL tasks. The experimental results not only show the great advantages of DAS in reducing computational complexity, e.g. -11.97%  FLOPs of METER on VQA2.0, but also confirm its competitiveness against existing PETL methods in terms of parameter scale and performance. Our source code is given in our appendix.

</details>

---

## 41. CoDet: Co-occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection

- [ ] CoDet: Co-occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection | https://neurips.cc/virtual/2023/poster/71566

- **Link**: https://neurips.cc/virtual/2023/poster/71566

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Deriving reliable region-word alignment from image-text pairs is critical to learnobject-level vision-language representations for open-vocabulary object detection.Existing methods typically rely on pre-trained or self-trained vision-languagemodels for alignment, which are prone to limitations in localization accuracy orgeneralization capabilities. In this paper, we propose CoDet, a novel approachthat overcomes the reliance on pre-aligned vision-language space by reformulatingregion-word alignment as a co-occurring object discovery problem. Intuitively, bygrouping images that mention a shared concept in their captions, objects corresponding to the shared concept shall exhibit high co-occurrence among the group.CoDet then leverages visual similarities to discover the co-occurring objects andalign them with the shared concept. Extensive experiments demonstrate that CoDethas superior performances and compelling scalability in open-vocabulary detection,e.g., by scaling up the visual backbone, CoDet achieves 37.0 $AP^m_{novel}$ and 44.7 $AP^m_{all}$ on OV-LVIS, surpassing the previous SoTA by 4.2 $AP^m_{novel}$ and 9.8 $AP^m_{all}$. Code is available at https://github.com/CVMI-Lab/CoDet.

</details>

---

## 42. Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions

- [ ] Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions | https://neurips.cc/virtual/2023/poster/71624

- **Link**: https://neurips.cc/virtual/2023/poster/71624

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The key to OOD detection has two aspects: generalized feature representation and precise category description. Recently, vision-language models such as CLIP provide significant advances in both two issues, but constructing precise category descriptions is still in its infancy due to the absence of unseen categories. This work introduces two hierarchical contexts, namely perceptual context and spurious context, to carefully describe the precise category boundary through automatic prompt tuning. Specifically, perceptual contexts perceive the inter-category difference (e.g., cats vs apples) for current classification tasks, while spurious contexts further identify spurious (similar but exactly not) OOD samples for every single category (e.g., cats vs panthers, apples vs peaches). The two contexts hierarchically construct the precise description for a certain category, which is, first roughly classifying a sample to the predicted category and then delicately identifying whether it is truly an ID sample or actually OOD. Moreover, the precise descriptions for those categories within the vision-language framework present a novel application: CATegory-EXtensible OOD detection (CATEX). One can efficiently extend the set of recognizable categories by simply merging the hierarchical contexts learned under different sub-task settings. And extensive experiments are conducted to demonstrate CATEX’s effectiveness, robustness, and category-extensibility. For instance, CATEX consistently surpasses the rivals by a large margin with several protocols on the challenging ImageNet-1K dataset. In addition, we offer new insights on how to efficiently scale up the prompt engineering in vision-language models to recognize thousands of object categories, as well as how to incorporate large language models (like GPT-3) to boost zero-shot applications.

</details>

---

## 43. Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks

- [ ] Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks | https://neurips.cc/virtual/2023/poster/71818

- **Link**: https://neurips.cc/virtual/2023/poster/71818

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of targeted data poisoning and backdoor attacks. Despite this vulnerability, robust contrastive vision-language pre-training against such attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a relatively large and varying pool of random captions, and matching every image with the text that is most similar to it in the pool instead of its own caption, every few epochs.It also leverages image and text augmentations to further strengthen the defense and improve the performance of the model. Our extensive experiments show that RoCLIP renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training CLIP models. In particular, RoCLIP decreases the success rate for targeted data poisoning attacks from 93.75% to 12.5% and that of backdoor attacks down to 0%, while improving the model's linear probe performance by 10% and maintains a similar zero shot performance compared to CLIP. By increasing the frequency of matching, RoCLIP is able to defend strong attacks, which add up to 1% poisoned examples to the data, and successfully maintain a low attack success rate of 12.5%, while trading off the performance on some tasks.

</details>

---

## 44. S-CLIP: Semi-supervised Vision-Language Learning using Few Specialist Captions

- [ ] S-CLIP: Semi-supervised Vision-Language Learning using Few Specialist Captions | https://neurips.cc/virtual/2023/poster/71829

- **Link**: https://neurips.cc/virtual/2023/poster/71829

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models, such as contrastive language-image pre-training (CLIP), have demonstrated impressive results in natural image domains. However, these models often struggle when applied to specialized domains like remote sensing, and adapting to such domains is challenging due to the limited number of image-text pairs available for training. To address this, we propose S-CLIP, a semi-supervised learning method for training CLIP that utilizes additional unpaired images. S-CLIP employs two pseudo-labeling strategies specifically designed for contrastive learning and the language modality. The caption-level pseudo-label is given by a combination of captions of paired images, obtained by solving an optimal transport problem between unpaired and paired images. The keyword-level pseudo-label is given by a keyword in the caption of the nearest paired image, trained through partial label learning that assumes a candidate set of labels for supervision instead of the exact one. By combining these objectives, S-CLIP significantly enhances the training of CLIP using only a few image-text pairs, as demonstrated in various specialist domains, including remote sensing, fashion, scientific figures, and comics. For instance, S-CLIP improves CLIP by 10% for zero-shot classification and 4% for image-text retrieval on the remote sensing benchmark, matching the performance of supervised CLIP while using three times fewer image-text pairs.

</details>

---

## 45. Efficient Policy Adaptation with Contrastive Prompt Ensemble for Embodied Agents

- [ ] Efficient Policy Adaptation with Contrastive Prompt Ensemble for Embodied Agents | https://neurips.cc/virtual/2023/poster/71833

- **Link**: https://neurips.cc/virtual/2023/poster/71833

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

For embodied reinforcement learning (RL) agents interacting with the environment, it is desirable to have rapid policy adaptation to unseen visual observations, but achieving zero-shot adaptation capability is considered as a challenging problem in the RL context. To address the problem, we present a novel contrastive prompt ensemble (ConPE) framework which utilizes a pretrained vision-language model and a set of visual prompts, thus enables efficient policy learning and adaptation upon a wide range of environmental and physical changes encountered by embodied agents. Specifically, we devise a guided-attention-based ensemble approach with multiple visual prompts on the vision-language model to construct robust state representations. Each prompt is contrastively learned in terms of an individual domain factors that significantly affects the agent's egocentric perception and observation. For a given task, the attention-based ensemble and policy are jointly learned so that the resulting state representations not only generalize to various domains but are also optimized for learning the task. Through experiments, we show that ConPE outperforms other state-of-the-art algorithms for several embodied agent tasks including navigation in AI2THOR, manipulation in Metaworld, and autonomous driving in CARLA, while also improving the sample efficiency of policy learning and adaptation.

</details>

---

## 46. Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP

- [ ] Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP | https://neurips.cc/virtual/2023/poster/71856

- **Link**: https://neurips.cc/virtual/2023/poster/71856

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training methods, e.g., CLIP, demonstrate an impressive zero-shot performance on visual categorizations with the class proxy from the text embedding of the class name. However, the modality gap between the text and vision space can result in a sub-optimal performance. We theoretically show that the gap cannot be reduced sufficiently by minimizing the contrastive loss in CLIP and the optimal proxy for vision tasks may reside only in the vision space. Therefore, given unlabeled target vision data, we propose to learn the vision proxy directly with the help from the text proxy for zero-shot transfer. Moreover, according to our theoretical analysis, strategies are developed to further refine the pseudo label obtained by the text proxy to facilitate the intra-modal proxy learning (InMaP) for vision. Experiments on extensive downstream tasks confirm the effectiveness and efficiency of our proposal. Concretely, InMaP can obtain the vision proxy within one minute on a single GPU while improving the zero-shot accuracy from $77.02\%$ to $80.21\%$ on ImageNet with ViT-L/14@336 pre-trained by CLIP.

</details>

---

## 47. Mitigating Test-Time Bias for Fair Image Retrieval

- [ ] Mitigating Test-Time Bias for Fair Image Retrieval | https://neurips.cc/virtual/2023/poster/71886

- **Link**: https://neurips.cc/virtual/2023/poster/71886

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We address the challenge of generating fair and unbiased image retrieval results given neutral textual queries (with no explicit gender or race connotations), while maintaining the utility (performance) of the underlying vision-language (VL) model. Previous methods aim to disentangle learned representations of images and text queries from gender and racial characteristics. However, we show these are inadequate at alleviating bias for the desired equal representation result, as there usually exists test-time bias in the target retrieval set. So motivated, we introduce a straightforward technique, Post-hoc Bias Mitigation (PBM), that post-processes the outputs from the pre-trained vision-language model. We evaluate our algorithm on real-world image search datasets, Occupation 1 and 2, as well as two large-scale image-text datasets, MS-COCO and Flickr30k. Our approach achieves the lowest bias, compared with various existing bias-mitigation methods, in text-based image retrieval result while maintaining satisfactory retrieval performance. The source code is publicly available at \url{https://github.com/timqqt/Fair Text based Image Retrieval}.

</details>

---

## 48. Exploring Question Decomposition for Zero-Shot VQA

- [ ] Exploring Question Decomposition for Zero-Shot VQA | https://neurips.cc/virtual/2023/poster/71957

- **Link**: https://neurips.cc/virtual/2023/poster/71957

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual question answering (VQA) has traditionally been treated as a single-step task where each question receives the same amount of effort, unlike natural human question-answering strategies. We explore a question decomposition strategy for VQA to overcome this limitation. We probe the ability of recently developed large vision-language models to use human-written decompositions and produce their own decompositions of visual questions, finding they are capable of learning both tasks from demonstrations alone.However, we show that naive application of model-written decompositions can hurt performance.We introduce a model-driven selective decomposition approach for second-guessing predictions and correcting errors, and validate its effectiveness on eight VQA tasks across three domains, showing consistent improvements in accuracy, including improvements of >20% on medical VQA datasets and boosting the zero-shot performance of BLIP-2 above chance on a VQA reformulation of the challenging Winoground task. Project Site: https://zaidkhan.me/decomposition-0shot-vqa/

</details>

---

## 49. Three Towers: Flexible Contrastive Learning with Pretrained Image Models

- [ ] Three Towers: Flexible Contrastive Learning with Pretrained Image Models | https://neurips.cc/virtual/2023/poster/71962

- **Link**: https://neurips.cc/virtual/2023/poster/71962

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce Three Towers (3T), a flexible method to improve the contrastive learning of vision-language models by incorporating pretrained image classifiers. While contrastive models are usually trained from scratch, LiT (Zhai et al., 2022) has recently shown performance gains from using pretrained classifier embeddings. However, LiT directly replaces the image tower with the frozen embeddings, excluding any potential benefits from training the image tower contrastively. With 3T, we propose a more flexible strategy that allows the image tower to benefit from both pretrained embeddings and contrastive training. To achieve this, we introduce a third tower that contains the frozen pretrained embeddings, and we encourage alignment between this third tower and the main image-text towers. Empirically, 3T consistently improves over LiT and the CLIP-style from-scratch baseline for retrieval tasks. For classification, 3T reliably improves over the from-scratch baseline, and while it underperforms relative to LiT for JFT-pretrained models, it outperforms LiT for ImageNet-21k and Places365 pretraining.

</details>

---

## 50. Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

- [ ] Learning Mask-aware CLIP Representations for Zero-Shot Segmentation | https://neurips.cc/virtual/2023/poster/72034

- **Link**: https://neurips.cc/virtual/2023/poster/72034

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recently, pre-trained vision-language models have been increasingly used to tackle the challenging zero-shot segmentation task. Typical solutions follow the paradigm of first generating mask proposals and then adopting CLIP to classify them. To maintain the CLIP's zero-shot transferability, previous practices favour to freeze CLIP during training. However, in the paper, we reveal that CLIP is insensitive to different mask proposals and tends to produce similar predictions for various mask proposals of the same image. This insensitivity results in numerous false positives when classifying mask proposals. This issue mainly relates to the fact that CLIP is trained with image-level supervision. To alleviate this issue, we propose a simple yet effective method, named Mask-aware Fine-tuning (MAFT). Specifically,  Image-Proposals CLIP Encoder (IP-CLIP Encoder) is proposed to handle arbitrary numbers of image and mask proposals simultaneously. Then, mask-aware loss and self-distillation loss are designed to fine-tune IP-CLIP Encoder, ensuring CLIP is responsive to different mask proposals while not sacrificing transferability. In this way, mask-aware representations can be easily learned to make the true positives stand out. Notably, our solution can seamlessly plug into most existing methods without introducing any new parameters during the fine-tuning process. We conduct extensive experiments on the popular zero-shot benchmarks. With MAFT, the performance of the state-of-the-art methods is promoted by a large margin: 50.4\% (+ 8.2\%) on COCO, 81.8\% (+ 3.2\%) on Pascal-VOC, and 8.7\% (+4.3\%) on ADE20K in terms of mIoU for unseen classes. Codes will be provided for reproducibility. Code is available at https://github.com/jiaosiyu1999/MAFT.git .

</details>

---

## 51. Vocabulary-free Image Classification

- [ ] Vocabulary-free Image Classification | https://neurips.cc/virtual/2023/poster/72064

- **Link**: https://neurips.cc/virtual/2023/poster/72064

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in large vision-language models have revolutionized the image classification paradigm. Despite showing impressive zero-shot capabilities, a pre-defined set of categories, a.k.a. the vocabulary, is assumed at test time for composing the textual prompts. However, such assumption can be impractical when the semantic context is unknown and evolving. We thus formalize a novel task, termed as Vocabulary-free Image Classification (VIC), where we aim to assign to an input image a class that resides in an unconstrained language-induced semantic space, without the prerequisite of a known vocabulary. VIC is a challenging task as the semantic space is extremely large, containing millions of concepts, with hard-to-discriminate fine-grained categories. In this work, we first empirically verify that representing this semantic space by means of an external vision-language database is the most effective way to obtain semantically relevant content for classifying the image. We then propose Category Search from External Databases (CaSED), a method that exploits a pre-trained vision-language model and an external vision-language database to address VIC in a training-free manner. CaSED first extracts a set of candidate categories from captions retrieved from the database based on their semantic similarity to the image, and then assigns to the image the best matching candidate category according to the same vision-language model. Experiments on benchmark datasets validate that CaSED outperforms other complex vision-language frameworks, while being efficient with much fewer parameters, paving the way for future research in this direction.

</details>

---

## 52. Glance and Focus: Memory Prompting for Multi-Event Video Question Answering

- [ ] Glance and Focus: Memory Prompting for Multi-Event Video Question Answering | https://neurips.cc/virtual/2023/poster/72076

- **Link**: https://neurips.cc/virtual/2023/poster/72076

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video Question Answering (VideoQA) has emerged as a vital tool to evaluate agents’ ability to understand human daily behaviors. Despite the recent success of large vision language models in many multi-modal tasks, complex situation reasoning over videos involving multiple human-object interaction events still remains challenging. In contrast, humans can easily tackle it by using a series of episode memories as anchors to quickly locate question-related key moments for reasoning. To mimic this effective reasoning strategy, we propose the Glance- Focus model. One simple way is to apply an action detection model to predict a set of actions as key memories. However, these actions within a closed set vocabulary are hard to generalize to various video domains. Instead of that, we train an Encoder-Decoder to generate a set of dynamic event memories at the glancing stage. Apart from using supervised bipartite matching to obtain the event memories, we further design an unsupervised memory generation method to get rid of dependence on event annotations. Next, at the focusing stage, these event memories act as a bridge to establish the correlation between the questions with high-level event concepts and low-level lengthy video content. Given the question, the model first focuses on the generated key event memory, then focuses on the most relevant moment for reasoning through our designed multi-level cross- attention mechanism. We conduct extensive experiments on four Multi-Event VideoQA benchmarks including STAR, EgoTaskQA, AGQA, and NExT-QA. Our proposed model achieves state-of-the-art results, surpassing current large models in various challenging reasoning tasks. The code and models are available at https://github.com/ByZ0e/Glance-Focus.

</details>

---

## 53. DesCo: Learning Object Recognition with Rich Language Descriptions

- [ ] DesCo: Learning Object Recognition with Rich Language Descriptions | https://neurips.cc/virtual/2023/poster/72079

- **Link**: https://neurips.cc/virtual/2023/poster/72079

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent development in vision-language approaches has instigated a paradigm shift in learning visual recognition models from language supervision. These approaches align objects with language queries (e.g. "a photo of a cat") and thus improve the models' adaptability to novel objects and domains. Recent studies have attempted to query these models with complex language expressions that include specifications of fine-grained details, such as colors, shapes, and relations. However, simply incorporating language descriptions into queries does not guarantee accurate interpretation by the models. In fact, our experiments show that GLIP, a state-of-the-art vision-language model for object detection, often disregards contextual information in the language descriptions and instead relies heavily on detecting objects solely by their names. To tackle the challenge, we propose a new description-conditioned (DesCo) paradigm of learning object recognition models with rich language descriptions consisting of two innovations: 1) we employ a large language model as a commonsense knowledge engine to generate rich language descriptions of objects; 2) we design context-sensitive queries to improve the model's ability in deciphering intricate nuances embedded within descriptions and enforce the model to focus on context rather than object names alone.  On two novel object detection benchmarks, LVIS and OminiLabel, under the zero-shot detection setting, our approach achieves 34.8 APr minival (+9.1) and 29.3 AP (+3.6), respectively, surpassing the prior state-of-the-art models, GLIP and FIBER, by a large margin.

</details>

---

## 54. EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought

- [ ] EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought | https://neurips.cc/virtual/2023/poster/72127

- **Link**: https://neurips.cc/virtual/2023/poster/72127

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Embodied AI is a crucial frontier in robotics, capable of planning and executing action sequences for robots to accomplish long-horizon tasks in physical environments.In this work, we introduce EmbodiedGPT, an end-to-end multi-modal foundation model for embodied AI, empowering embodied agents with multi-modal understanding and execution capabilities. To achieve this, we have made the following efforts: (i) We craft a large-scale embodied planning dataset, termed EgoCOT. The dataset consists of carefully selected videos from the Ego4D dataset, along with corresponding high-quality language instructions. Specifically, we generate a sequence of sub-goals with the "Chain of Thoughts" mode for effective embodied planning.(ii) We introduce an efficient training approach to EmbodiedGPT for high-quality plan generation, by adapting a 7B large language model (LLM) to the EgoCOT dataset via prefix tuning. (iii) We introduce a paradigm for extracting task-related features from LLM-generated planning queries to form a closed loop between high-level planning and low-level control.Extensive experiments show the effectiveness of EmbodiedGPT on embodied tasks, including embodied planning, embodied control, visual captioning, and visual question answering.Notably, EmbodiedGPT significantly enhances the success rate of the embodied control task by extracting more effective features. It has achieved a remarkable 1.6 times increase in success rate on the Franka Kitchen benchmark and a 1.3 times increase on the Meta-World benchmark, compared to the BLIP-2 baseline fine-tuned with the Ego4D dataset.

</details>

---

## 55. ChatGPT-Powered Hierarchical Comparisons for Image Classification

- [ ] ChatGPT-Powered Hierarchical Comparisons for Image Classification | https://neurips.cc/virtual/2023/poster/72209

- **Link**: https://neurips.cc/virtual/2023/poster/72209

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The zero-shot open-vocabulary setting poses challenges for image classification.Fortunately, utilizing a vision-language model like CLIP, pre-trained on image-textpairs, allows for classifying images by comparing embeddings. Leveraging largelanguage models (LLMs) such as ChatGPT can further enhance CLIP’s accuracyby incorporating class-specific knowledge in descriptions. However, CLIP stillexhibits a bias towards certain classes and generates similar descriptions for similarclasses, disregarding their differences. To address this problem, we present anovel image classification framework via hierarchical comparisons. By recursivelycomparing and grouping classes with LLMs, we construct a class hierarchy. Withsuch a hierarchy, we can classify an image by descending from the top to the bottomof the hierarchy, comparing image and text embeddings at each level. Throughextensive experiments and analyses, we demonstrate that our proposed approach isintuitive, effective, and explainable. Code will be released upon publication.

</details>

---

## 56. SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models

- [ ] SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models | https://neurips.cc/virtual/2023/poster/72303

- **Link**: https://neurips.cc/virtual/2023/poster/72303

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation (TTA) is a special and practical setting in unsupervised domain adaptation, which allows a pre-trained model in a source domain to adapt to unlabeled test data in another target domain. To avoid the computation-intensive backbone fine-tuning process, the zero-shot generalization potentials of the emerging pre-trained vision-language models (e.g., CLIP, CoOp) are leveraged to only tune the run-time prompt for unseen test domains. However, existing solutions have yet to fully exploit the representation capabilities of pre-trained models as they only focus on the entropy-based optimization and the performance is far below the supervised prompt adaptation methods, e.g., CoOp. In this paper, we propose SwapPrompt, a novel framework that can effectively leverage the self-supervised contrastive learning to facilitate the test-time prompt adaptation. SwapPrompt employs a dual prompts paradigm, i.e., an online prompt and a target prompt that averaged from the online prompt to retain historical information. In addition, SwapPrompt applies a swapped prediction mechanism, which takes advantage of the representation capabilities of pre-trained models to enhance the online prompt via contrastive learning. Specifically, we use the online prompt together with an augmented view of the input image to predict the class assignment generated by the target prompt together with an alternative augmented view of the same image. The proposed SwapPrompt can be easily deployed on vision-language models without additional requirement, and experimental results show that it achieves state-of-the-art test-time adaptation performance on ImageNet and nine other datasets. It is also shown that SwapPrompt can even achieve comparable performance with supervised prompt adaptation methods.

</details>

---

## 57. AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation

- [ ] AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation | https://neurips.cc/virtual/2023/poster/72319

- **Link**: https://neurips.cc/virtual/2023/poster/72319

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation is a challenging task that requires segmenting novel object categories at inference time. Recent works explore vision-language pre-training to handle this task, but suffer from unrealistic assumptions in practical scenarios, i.e., low-quality textual category names.For example, this paradigm assumes that new textual categories will be accurately and completely provided, and exist in lexicons during pre-training.However, exceptions often happen when meet with ambiguity for brief or incomplete names, new words that are not present in the pre-trained lexicons, and difficult-to-describe categories for users.To address these issues, this work proposes a novel attribute decomposition-aggregation framework, AttrSeg , inspired by human cognition in understanding new concepts. Specifically, in the decomposition stage, we decouple class names into diverse attribute descriptions to complement semantic contexts from multiple perspectives.Two attribute construction strategies are designed: using large language models for common categories, and involving manually labelling for human-invented categories. In the aggregation stage, we group diverse attributes into an integrated global description, to form a discriminative classifier that distinguishes the target object from others. One hierarchical aggregation architecture is further proposed to achieve multi-level aggregation, leveraging the meticulously designed clustering module.The final result is obtained by computing the similarity between aggregated attributes and images embedding.To evaluate the effectiveness, we annotate three datasets with attribute descriptions, and conduct extensive experiments and ablation studies. The results show the superior performance of attribute decomposition-aggregation.We refer readers to the latest arXiv version at https://arxiv.org/abs/2309.00096.

</details>

---

## 58. Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution

- [ ] Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution | https://neurips.cc/virtual/2023/poster/72331

- **Link**: https://neurips.cc/virtual/2023/poster/72331

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pretrained models have seen remarkable success, but their application to safety-critical settings is limited by their lack of interpretability. To improve the interpretability of vision-language models such as CLIP, we propose a multi-modal information bottleneck (M2IB) approach that learns latent representations that compress irrelevant information while preserving relevant visual and textual features. We demonstrate how M2IB can be applied to attribution analysis of vision-language pretrained models, increasing attribution accuracy and improving the interpretability of such models when applied to safety-critical domains such as healthcare. Crucially, unlike commonly used unimodal attribution methods, M2IB does not require ground truth labels, making it possible to audit representations of vision-language pretrained models when multiple modalities but no ground-truth data is available. Using CLIP as an example, we demonstrate the effectiveness of M2IB attribution and show that it outperforms gradient-based, perturbation-based, and attention-based attribution methods both qualitatively and quantitatively.

</details>

---

## 59. RoboCLIP: One Demonstration is Enough to Learn Robot Policies

- [ ] RoboCLIP: One Demonstration is Enough to Learn Robot Policies | https://neurips.cc/virtual/2023/poster/72364

- **Link**: https://neurips.cc/virtual/2023/poster/72364

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Reward specification is a notoriously difficult problem in reinforcement learning, requiring extensive expert supervision to design robust reward functions. Imitation learning (IL) methods attempt to circumvent these problems by utilizing expert demonstrations instead of using an extrinsic reward function but typically require a large number of in-domain expert demonstrations. Inspired by advances in the field of Video-and-Language Models (VLMs), we present RoboCLIP, an online imitation learning method that uses a single demonstration (overcoming the large data requirement) in the form of a video demonstration or a textual description of the task to generate rewards without manual reward function design. Additionally, RoboCLIP can also utilize out-of-domain demonstrations, like videos of humans solving the task for reward generation, circumventing the need to have the same demonstration and deployment domains. RoboCLIP utilizes pretrained VLMs without any finetuning for reward generation. Reinforcement learning agents trained with RoboCLIP rewards demonstrate 2-3 times higher zero-shot performance than competing imitation learning methods on downstream robot manipulation tasks, doing so using only one video/text demonstration. Visit our website at https://sites.google.com/view/roboclip/home for experiment videos.

</details>

---

## 60. Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization

- [ ] Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization | https://neurips.cc/virtual/2023/poster/72406

- **Link**: https://neurips.cc/virtual/2023/poster/72406

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The promising zero-shot generalization of vision-language models such as CLIP has led to their adoption using prompt learning for numerous downstream tasks. Previous works have shown test-time prompt tuning using entropy minimization to adapt text prompts for unseen domains. While effective, this overlooks the key cause for performance degradation to unseen domains -- distribution shift. In this work, we explicitly handle this problem by aligning the out-of-distribution (OOD) test sample statistics to those of the source data using prompt tuning. We use a single test sample to adapt multi-modal prompts at test time by minimizing the feature distribution shift to bridge the gap in the test domain. Evaluating against the domain generalization benchmark, our method improves zero-shot top-1 accuracy beyond existing prompt-learning techniques, with a 3.08% improvement over the baseline MaPLe. In cross-dataset generalization with unseen categories across 10 datasets, our method improves consistently across all datasets compared to the existing state-of-the-art. Our source code and models are available at https://jameelhassan.github.io/promptalign

</details>

---

## 61. Towards Label-free Scene Understanding by Vision Foundation Models

- [ ] Towards Label-free Scene Understanding by Vision Foundation Models | https://neurips.cc/virtual/2023/poster/72450

- **Link**: https://neurips.cc/virtual/2023/poster/72450

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision foundation models such as Contrastive Vision-Language Pre-training (CLIP) and Segment Anything (SAM) have demonstrated impressive zero-shot performance on image classification and segmentation tasks. However, the incorporation of CLIP and SAM for label-free scene understanding has yet to be explored. In this paper, we investigate the potential of vision foundation models in enabling networks to comprehend 2D and 3D worlds without labelled data. The primary challenge lies in effectively supervising networks under extremely noisy pseudo labels, which are generated by CLIP and further exacerbated during the propagation from the 2D to the 3D domain. To tackle these challenges, we propose a novel Cross-modality Noisy Supervision (CNS) method that leverages the strengths of CLIP and SAM to supervise 2D and 3D networks simultaneously. In particular, we introduce a prediction consistency regularization to co-train 2D and 3D networks, then further impose the networks' latent space consistency using the SAM's robust feature representation. Experiments conducted on diverse indoor and outdoor datasets demonstrate the superior performance of our method in understanding 2D and 3D open environments. Our 2D and 3D network achieves label-free semantic segmentation with 28.4\% and 33.5\% mIoU on ScanNet, improving 4.7\% and 7.9\%, respectively. For nuImages and nuScenes datasets, the performance is 22.1\% and 26.8\% with improvements of 3.5\% and 6.0\%, respectively. Code is available. (https://github.com/runnanchen/Label-Free-Scene-Understanding)

</details>

---

## 62. Tuning Multi-mode Token-level Prompt Alignment across Modalities

- [ ] Tuning Multi-mode Token-level Prompt Alignment across Modalities | https://neurips.cc/virtual/2023/poster/72564

- **Link**: https://neurips.cc/virtual/2023/poster/72564

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Advancements in prompt tuning of vision-language models have underscored their potential in enhancing open-world visual concept comprehension. However, prior works only primarily focus on single-mode (only one prompt for each modality) and holistic level (image or sentence) semantic alignment, which fails to capture the sample diversity, leading to sub-optimal prompt discovery. To address the limitation, we propose a multi-mode token-level tuning framework that leverages the optimal transportation to learn and align a set of prompt tokens across modalities. Specifically, we rely on two essential factors: 1) multi-mode prompts discovery, which guarantees diverse semantic representations, and 2) token-level alignment, which helps explore fine-grained similarity. Consequently, the similarity can be calculated as a hierarchical transportation problem between the modality-specific sets. Extensive experiments on popular image recognition benchmarks show the superior generalization and few-shot abilities of our approach. The qualitative analysis demonstrates that the learned prompt tokens have the ability to capture diverse visual concepts.

</details>

---

## 63. Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation

- [ ] Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation | https://neurips.cc/virtual/2023/poster/72582

- **Link**: https://neurips.cc/virtual/2023/poster/72582

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from languagesupervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from a clear semantic gap between visual and textual modalities: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of closing semantic gap in pre-training data.

</details>

---

## 64. Bootstrapping Vision-Language Learning with Decoupled Language Pre-training

- [ ] Bootstrapping Vision-Language Learning with Decoupled Language Pre-training | https://neurips.cc/virtual/2023/poster/72654

- **Link**: https://neurips.cc/virtual/2023/poster/72654

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present a novel methodology aimed at optimizing the application of frozen large language models (LLMs) for resource-intensive vision-language (VL) pre-training. The current paradigm uses visual features as prompts to guide language models, with a focus on determining the most relevant visual features for corresponding text. Our approach diverges by concentrating on the language component, specifically identifying the optimal prompts to align with visual features. We introduce the Prompt-Transformer (P-Former), a model that predicts these ideal prompts, which is trained exclusively on linguistic data, bypassing the need for image-text pairings. This strategy subtly bifurcates the end-to-end VL training process into an additional, separate stage. Our experiments reveal that our framework significantly enhances the performance of a robust image-to-text baseline (BLIP-2), and effectively narrows the performance gap between models trained with either 4M or 129M image-text pairs. Importantly, our framework is modality-agnostic and flexible in terms of architectural design, as validated by its successful application in a video learning task using varied base modules. The code will be made available at https://github.com/yiren-jian/BLIText.

</details>

---

## 65. VPGTrans: Transfer Visual Prompt Generator across LLMs

- [ ] VPGTrans: Transfer Visual Prompt Generator across LLMs | https://neurips.cc/virtual/2023/poster/72728

- **Link**: https://neurips.cc/virtual/2023/poster/72728

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Since developing a new multimodal LLM (MLLM) by pre-training on tremendous image-text pairs from scratch can be exceedingly resource-consuming, connecting an existing LLM with a comparatively lightweight visual prompt generator (VPG) becomes a feasible paradigm. However, further tuning the VPG component of the MLLM still incurs significant computational costs, such as thousands of GPU hours and millions of training data points. An alternative solution is transferring an existing VPG from one MLLM to the target MLLM. In this work, we investigate VPG transferability across LLMs for the first time, aiming to reduce the cost of VPG training.  Specifically, we explore VPG transfer across different LLM sizes (e.g., small-to-large) and types.  We identify key factors to maximize transfer efficiency, based on which we develop a simple yet highly effective two-stage transfer framework, called VPGTrans. Notably, it enables VPG transfer from BLIP-2 OPT 2.7B to BLIP-2 OPT 6.7B with less than 10% of the GPU hours using only 10.7% of the training data compared to training a VPG for OPT 6.7B from scratch. Furthermore, we provide a series of intriguing findings and discuss potential explanations behind them. Finally, we showcase the practical value of our VPGTrans approach, by customizing two novel MLLMs, including VL-LLaMA and VL-Vicuna, with recently released LLaMA and Vicuna LLMs.

</details>

---

## 66. In-Context Learning Unlocked for Diffusion Models

- [ ] In-Context Learning Unlocked for Diffusion Models | https://neurips.cc/virtual/2023/poster/72769

- **Link**: https://neurips.cc/virtual/2023/poster/72769

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present Prompt Diffusion, a framework for enabling in-context learning in diffusion-based generative models. Given a pair of task-specific example images, such as depth from/to image and scribble from/to image, and a text guidance, our model automatically understands the underlying task and performs the same task on a new query image following the text guidance. To achieve this, we propose a vision-language prompt that can model a wide range of vision-language tasks and a diffusion model that takes it as input. The diffusion model is trained jointly on six different tasks using these prompts. The resulting Prompt Diffusion model becomes the first diffusion-based vision-language foundation model capable of in-context learning. It demonstrates high-quality in-context generation for the trained tasks and effectively generalizes to new, unseen vision tasks using their respective prompts. Our model also shows compelling text-guided image editing results. Our framework aims to facilitate research into in-context learning for computer vision. We share our code and pre-trained models at https://github.com/Zhendong-Wang/Prompt-Diffusion.

</details>

---

## 67. Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias

- [ ] Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias | https://neurips.cc/virtual/2023/poster/72830

- **Link**: https://neurips.cc/virtual/2023/poster/72830

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The scarcity of data presents a critical obstacle to the efficacy of medical vision-language pre-training (VLP). A potential solution lies in the combination of datasets from various language communities.Nevertheless, the main challenge stems from the complexity of integrating diverse syntax and semantics, language-specific medical terminology, and culture-specific implicit knowledge. Therefore, one crucial aspect to consider is the presence of community bias caused by different languages.This paper presents a novel framework named Unifying Cross-Lingual Medical Vision-Language Pre-Training (\textbf{Med-UniC}), designed to integrate multi-modal medical data from the two most prevalent languages, English and Spanish. Specifically, we propose \textbf{C}ross-lingual \textbf{T}ext Alignment \textbf{R}egularization (\textbf{CTR}) to explicitly unify cross-lingual semantic representations of medical reports originating from diverse language communities. \textbf{CTR} is optimized through latent language disentanglement, rendering our optimization objective to not depend on negative samples, thereby significantly mitigating the bias from determining positive-negative sample pairs within analogous medical reports. Furthermore, it ensures that the cross-lingual representation is not biased toward any specific language community.\textbf{Med-UniC} reaches superior performance across 5 medical image tasks and 10 datasets encompassing over 30 diseases, offering a versatile framework for unifying multi-modal medical data within diverse linguistic communities.The experimental outcomes highlight the presence of community bias in cross-lingual VLP. Reducing this bias enhances the performance not only in vision-language tasks but also in uni-modal visual tasks.

</details>

---

## 68. HiBug: On Human-Interpretable Model Debug

- [ ] HiBug: On Human-Interpretable Model Debug | https://neurips.cc/virtual/2023/poster/72832

- **Link**: https://neurips.cc/virtual/2023/poster/72832

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Machine learning models can frequently produce systematic errors on critical subsets (or slices) of data that share common attributes. Discovering and explaining such model bugs is crucial for reliable model deployment. However, existing bug discovery and interpretation methods usually involve heavy human intervention and annotation, which can be cumbersome and have low bug coverage.In this paper, we propose HiBug, an automated framework for interpretable model debugging. Our approach utilizes large pre-trained models, such as chatGPT, to suggest human-understandable attributes that are related to the targeted computer vision tasks. By leveraging pre-trained vision-language models, we can efficiently identify common visual attributes of underperforming data slices using human-understandable terms. This enables us to uncover rare cases in the training data, identify spurious correlations in the model, and use the interpretable debug results to select or generate new training data for model improvement. Experimental results demonstrate the efficacy of the HiBug framework.

</details>

---

## 69. Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning

- [ ] Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning | https://neurips.cc/virtual/2023/poster/72947

- **Link**: https://neurips.cc/virtual/2023/poster/72947

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning vision-language models (VLMs) like CLIP to downstream tasks is often necessary to optimize their performance. However, a major obstacle is the limited availability of labeled data. We study the use of pseudolabels, i.e., heuristic labels for unlabeled data, to enhance CLIP via prompt tuning. Conventional pseudolabeling trains a model on labeled data and then generates labels for unlabeled data. VLMs' zero-shot capabilities enable a ``second generation'' of pseudolabeling approaches that do not require task-specific training on labeled data. By using zero-shot pseudolabels as a source of supervision, we observe that learning paradigms such as semi-supervised, transductive zero-shot, and unsupervised learning can all be seen as optimizing the same loss function. This unified view enables the development of versatile training strategies that are applicable across learning paradigms. We investigate them on image classification tasks where CLIP exhibits limitations, by varying prompt modalities, e.g., textual or visual prompts, and learning paradigms. We find that(1) unexplored prompt tuning strategies that iteratively refine pseudolabels consistently improve CLIP accuracy, by 19.5 points in semi-supervised learning, by 28.4 points in transductive zero-shot learning, and by 15.2 points in unsupervised learning, and (2) unlike conventional semi-supervised pseudolabeling, which exacerbates model biases toward classes with higher-quality pseudolabels, prompt tuning leads to a more equitable distribution of per-class accuracy. The code to reproduce the experiments is at https://github.com/BatsResearch/menghini-neurips23-code.

</details>

---

## 70. Scaling laws for language encoding models in fMRI

- [ ] Scaling laws for language encoding models in fMRI | https://neurips.cc/virtual/2023/poster/72951

- **Link**: https://neurips.cc/virtual/2023/poster/72951

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Representations from transformer-based unidirectional language models are known to be effective at predicting brain responses to natural language. However, most studies comparing language models to brains have used GPT-2 or similarly sized language models. Here we tested whether larger open-source models such as those from the OPT and LLaMA families are better at predicting brain responses recorded using fMRI. Mirroring scaling results from other contexts, we found that brain prediction performance scales logarithmically with model size from 125M to 30B parameter models, with ~15% increased encoding performance as measured by correlation with a held-out test set across 3 subjects. Similar log-linear behavior was observed when scaling the size of the fMRI training set. We also characterized scaling for acoustic encoding models that use HuBERT, WavLM, and Whisper, and we found comparable improvements with model size. A noise ceiling analysis of these large, high-performance encoding models showed that performance is nearing the theoretical maximum for brain areas such as the precuneus and higher auditory cortex. These results suggest that increasing scale in both models and data will yield incredibly effective models of language processing in the brain, enabling better scientific understanding as well as applications such as decoding.

</details>

---

## 71. Large Language Models are Visual Reasoning Coordinators

- [ ] Large Language Models are Visual Reasoning Coordinators | https://neurips.cc/virtual/2023/poster/72992

- **Link**: https://neurips.cc/virtual/2023/poster/72992

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual reasoning requires multimodal perception and commonsense cognition of the world. Recently, multiple vision-language models (VLMs) have been proposed with excellent commonsense reasoning ability in various domains. However, how to harness the collective power of these complementary VLMs is rarely explored. Existing methods like ensemble still struggle to aggregate these models with the desired higher-order communications. In this work, we propose Cola, a novel paradigm that coordinates multiple VLMs for visual reasoning. Our key insight is that a large language model (LLM) can efficiently coordinate multiple VLMs by facilitating natural language communication that leverages their distinct and complementary capabilities. Extensive experiments demonstrate that our instruction tuning variant, Cola-FT, achieves state-of-the-art performance on visual question answering (VQA), outside knowledge VQA, visual entailment, and visual spatial reasoning tasks. Moreover, we show that our in-context learning variant, Cola-Zero, exhibits competitive performance in zero and few-shot settings, without finetuning. Through systematic ablation studies and visualizations, we validate that a coordinator LLM indeed comprehends the instruction prompts as well as the separate functionalities of VLMs; it then coordinates them to enable impressive visual reasoning capabilities.

</details>

---

## 72. HeadSculpt: Crafting 3D Head Avatars with Text

- [ ] HeadSculpt: Crafting 3D Head Avatars with Text | https://neurips.cc/virtual/2023/poster/73029

- **Link**: https://neurips.cc/virtual/2023/poster/73029

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recently, text-guided 3D generative methods have made remarkable advancements in producing high-quality textures and geometry, capitalizing on the proliferation of large vision-language and image diffusion models. However, existing methods still struggle to create high-fidelity 3D head avatars in two aspects: (1) They rely mostly on a pre-trained text-to-image diffusion model whilst missing the necessary 3D awareness and head priors. This makes them prone to inconsistency and geometric distortions in the generated avatars. (2) They fall short in fine-grained editing. This is primarily due to the inherited limitations from the pre-trained 2D image diffusion models, which become more pronounced when it comes to 3D head avatars. In this work, we address these challenges by introducing a versatile coarse-to-fine pipeline dubbed HeadSculpt for crafting (i.e., generating and editing) 3D head avatars from textual prompts. Specifically, we first equip the diffusion model with 3D awareness by leveraging landmark-based control and a learned textual embedding representing the back view appearance of heads, enabling 3D-consistent head avatar generations. We further propose a novel identity-aware editing score distillation strategy to optimize a textured mesh with a high-resolution differentiable rendering technique. This enables identity preservation while following the editing instruction.We showcase HeadSculpt's superior fidelity and editing capabilities through comprehensive experiments and comparisons with existing methods.

</details>

---

## 73. Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models

- [ ] Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models | https://neurips.cc/virtual/2023/poster/73073

- **Link**: https://neurips.cc/virtual/2023/poster/73073

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Task arithmetic has recently emerged as a cost-effective and scalable approach to edit pre-trained models directly in weight space: By adding the fine-tuned weights of different tasks, the model's performance can be improved on these tasks, while negating them leads to task forgetting. Yet, our understanding of the effectiveness of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show that fine-tuning models in their tangent space by linearizing them amplifies weight disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we provide theoretical and empirical analyses of the neural tangent kernel (NTK) of these models and establish a compelling link between task arithmetic and the spatial localization of the NTK eigenfunctions. Overall, our work uncovers novel insights into the fundamental mechanisms of task arithmetic and offers a more reliable and effective approach to edit pre-trained models through the NTK linearization.

</details>

---

## 74. On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

- [ ] On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection | https://neurips.cc/virtual/2023/poster/73075

- **Link**: https://neurips.cc/virtual/2023/poster/73075

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Successful detection of Out-of-Distribution (OoD) data is becoming increasingly important to ensure safe deployment of neural networks. One of the main challenges in OoD detection is that neural networks output overconfident predictions on OoD data, make it difficult to determine OoD-ness of data solely based on their predictions. Outlier exposure addresses this issue by introducing an additional loss that encourages low-confidence predictions on OoD data during training. While outlier exposure has shown promising potential in improving OoD detection performance, all previous studies on outlier exposure have been limited to utilizing visual outliers. Drawing inspiration from the recent advancements in vision-language pre-training, this paper venture out to the uncharted territory of textual outlier exposure. First, we uncover the benefits of using textual outliers by replacing real or virtual outliers in the image-domain with textual equivalents. Then, we propose various ways of generating preferable textual outliers. Our extensive experiments demonstrate that generated textual outliers achieve competitive performance on large-scale OoD and hard OoD benchmarks. Furthermore, we conduct empirical analyses of textual outliers to provide primary criteria for designing advantageous textual outliers: near-distribution, descriptiveness, and inclusion of visual semantics.

</details>

---

## 75. T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation

- [ ] T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation | https://neurips.cc/virtual/2023/poster/73424

- **Link**: https://neurips.cc/virtual/2023/poster/73424

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite the stunning ability to generate high-quality images by recent text-to-image models, current approaches often struggle to effectively compose objects with different attributes and relationships into a complex and coherent scene. We propose T2I-CompBench, a comprehensive benchmark for open-world compositional text-to-image generation, consisting of 6,000 compositional text prompts from 3 categories (attribute binding, object relationships, and complex compositions) and 6 sub-categories (color binding, shape binding, texture binding, spatial relationships, non-spatial relationships, and complex compositions). We further propose several evaluation metrics specifically designed to evaluate compositional text-to-image generation and explore the potential and limitations of multimodal LLMs for evaluation. We introduce a new approach, Generative mOdel finetuning with Reward-driven Sample selection (GORS), to boost the compositional text-to-image generation abilities of pretrained text-to-image models. Extensive experiments and evaluations are conducted to benchmark previous methods on T2I-CompBench, and to validate the effectiveness of our proposed evaluation metrics and GORS approach. Project page is available at https://karine-h.github.io/T2I-CompBench/.

</details>

---

## 76. JourneyDB: A Benchmark for Generative Image Understanding

- [ ] JourneyDB: A Benchmark for Generative Image Understanding | https://neurips.cc/virtual/2023/poster/73428

- **Link**: https://neurips.cc/virtual/2023/poster/73428

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

While recent advancements in vision-language models have had a transformative impact on multi-modal comprehension, the extent to which these models possess the ability to comprehend generated images remains uncertain. Synthetic images, in comparison to real data, encompass a higher level of diversity in terms of both content and style, thereby presenting significant challenges for the models to fully grasp. In light of this challenge, we introduce a comprehensive dataset, referred to as JourneyDB, that caters to the domain of generative images within the context of multi-modal visual understanding. Our meticulously curated dataset comprises 4 million distinct and high-quality generated images, each paired with the corresponding text prompts that were employed in their creation. Furthermore, we additionally introduce an external subset with results of another 22 text-to-image generative models, which makes JourneyDB a comprehensive benchmark for evaluating the comprehension of generated images. On our dataset, we have devised four benchmarks to assess the performance of generated image comprehension in relation to both content and style interpretation. These benchmarks encompass prompt inversion, style retrieval, image captioning, and visual question answering. Lastly, we evaluate the performance of state-of-the-art multi-modal models when applied to the JourneyDB dataset, providing a comprehensive analysis of their strengths and limitations in comprehending generated content. We anticipate that the proposed dataset and benchmarks will facilitate further research in the field of generative content understanding. The dataset is publicly available at https://journeydb.github.io.

</details>

---

## 77. Cola: A Benchmark for Compositional Text-to-image Retrieval

- [ ] Cola: A Benchmark for Compositional Text-to-image Retrieval | https://neurips.cc/virtual/2023/poster/73493

- **Link**: https://neurips.cc/virtual/2023/poster/73493

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Compositional reasoning is a hallmark of human visual intelligence. Yet, despite the size of large vision-language models, they struggle to represent simple compositions by combining objects with their attributes. To measure this lack of compositional capability, we design Cola, a text-to-image retrieval benchmark to Compose Objects Localized with Attributes. To solve Cola, a model must retrieve images with the correct configuration of attributes and objects and avoid choosing a distractor image with the same objects and attributes but in the wrong configuration. Cola contains about 1.2k composed queries of 168 objects and 197 attributes on around 30K images. Our human evaluation finds that Cola is 83.33% accurate, similar to contemporary compositionality benchmarks. Using Cola as a testbed, we explore empirical modeling designs to adapt pre-trained vision-language models to reason compositionally. We explore 6 adaptation strategies on 2 seminal vision-language models, using compositionality-centric test benchmarks - Cola and CREPE. We find the optimal adaptation strategy is to train a multi-modal attention layer that jointly attends over the frozen pre-trained image and language features. Surprisingly, training multimodal layers on CLIP performs better than tuning a larger FLAVA model with already pre-trained multimodal layers. Furthermore, our adaptation strategy improves CLIP and FLAVA to comparable levels, suggesting that training multimodal layers using contrastive attribute-object data is key, as opposed to using them pre-trained. Lastly, we show that Cola is harder than a closely related contemporary benchmark, CREPE, since simpler fine-tuning strategies without multimodal layers suffice on CREPE, but not on Cola. However, we still see a significant gap between our best adaptation and human accuracy, suggesting considerable room for further research. Project page: https://cs-people.bu.edu/array/research/cola/

</details>

---

## 78. M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models

- [ ] M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models | https://neurips.cc/virtual/2023/poster/73506

- **Link**: https://neurips.cc/virtual/2023/poster/73506

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite the existence of various benchmarks for evaluating natural language processing models, we argue that human exams are a more suitable means of evaluating general intelligence for large language models (LLMs), as they inherently demand a much wider range of abilities such as language understanding, domain knowledge, and problem-solving skills. To this end, we introduce M3Exam, a novel benchmark sourced from real and official human exam questions for evaluating LLMs in a multilingual, multimodal, and multilevel context. M3Exam exhibits three unique characteristics: (1) multilingualism, encompassing questions from multiple countries that require strong multilingual proficiency and cultural knowledge; (2) multimodality, accounting for the multimodal nature of many exam questions to test the model's multimodal understanding capability; and (3) multilevel structure, featuring exams from three critical educational periods to comprehensively assess a model's proficiency at different levels. In total, M3Exam contains 12,317 questions in 9 diverse languages with three educational levels, where about 23\% of the questions require processing images for successful solving. We assess the performance of top-performing LLMs on M3Exam and find that current models, including GPT-4, still struggle with multilingual text, particularly in low-resource and non-Latin script languages. Multimodal LLMs also perform poorly with complex multimodal questions. We believe that M3Exam can be a valuable resource for comprehensively evaluating LLMs by examining their multilingual and multimodal abilities and tracking their development. Data and evaluation code is available at \url{https://github.com/DAMO-NLP-SG/M3Exam}.

</details>

---

## 79. LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark

- [ ] LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark | https://neurips.cc/virtual/2023/poster/73519

- **Link**: https://neurips.cc/virtual/2023/poster/73519

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large language models have emerged as a promising approach towards achieving general-purpose AI agents. The thriving open-source LLM community has greatly accelerated the development of agents that support human-machine dialogue interaction through natural language processing. However, human interaction with the world extends beyond only text as a modality, and other modalities such as vision are also crucial. Recent works on multi-modal large language models, such as GPT-4V and Bard, have demonstrated their effectiveness in handling visual modalities. However, the transparency of these works is limited and insufficient to support academic research. To the best of our knowledge, we present one of the very first open-source endeavors in the field, LAMM, encompassing a Language-Assisted Multi-Modal instruction tuning dataset, framework, and benchmark. Our aim is to establish LAMM as a growing ecosystem for training and evaluating MLLMs, with a specific focus on facilitating AI agents capable of bridging the gap between ideas and execution, thereby enabling seamless human-AI interaction. Our main contribution is three-fold: 1) We present a comprehensive dataset and benchmark, which cover a wide range of vision tasks for 2D and 3D vision. Extensive experiments validate the effectiveness of our dataset and benchmark. 2) We outline the detailed methodology of constructing multi-modal instruction tuning datasets and benchmarks for MLLMs, enabling rapid scaling and extension of MLLM research to diverse domains, tasks, and modalities. 3) We provide a primary but potential MLLM training framework optimized for modality extension. We also provide baseline models, comprehensive experimental observations, and analysis to accelerate future research. Our baseline model is trained within 24 A100 GPU hours, framework supports training with V100 and RTX3090 is available thanks to the open-source society. Codes and data are now available at https://openlamm.github.io.

</details>

---

## 80. VidChapters-7M: Video Chapters at Scale

- [ ] VidChapters-7M: Video Chapters at Scale | https://neurips.cc/virtual/2023/poster/73545

- **Link**: https://neurips.cc/virtual/2023/poster/73545

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Segmenting untrimmed videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. We benchmark both simple baselines as well as state-of-the-art video-language models on these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset.

</details>

---

## 81. VisIT-Bench: A Dynamic Benchmark for Evaluating Instruction-Following Vision-and-Language Models

- [ ] VisIT-Bench: A Dynamic Benchmark for Evaluating Instruction-Following Vision-and-Language Models | https://neurips.cc/virtual/2023/poster/73556

- **Link**: https://neurips.cc/virtual/2023/poster/73556

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for evaluating instruction-following vision-language models for real-world use. Our starting point is curating 70 "instruction families" that we envision instruction tuned vision-language models should be able to address. Extending beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to game playing and creative generation. Following curation, our dataset comprises 592 test queries, each with a human-authored instruction-conditioned caption. These descriptions surface instruction-specific factors, e.g., for an instruction asking about the accessibility of a storefront for wheelchair users, the instruction-conditioned caption describes ramps/potential obstacles. These descriptions enable 1) collecting human-verified reference outputs for each instance; and 2) automatic evaluation of candidate multimodal generations using a text-only LLM, aligning with human judgment. We quantify quality gaps between models and references using both human and automatic evaluations; e.g., the top-performing instruction-following model wins against the GPT-4 reference in just 27% of the comparison. VisIT-Bench is dynamic to participate, practitioners simply submit their model's response on the project website; Data, code and leaderboard is available at https://visit-bench.github.io/.

</details>

---

## 82. Improving multimodal datasets with image captioning

- [ ] Improving multimodal datasets with image captioning | https://neurips.cc/virtual/2023/poster/73574

- **Link**: https://neurips.cc/virtual/2023/poster/73574

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Massive web datasets play a key role in the success of large vision-language models like CLIP and Flamingo. However, the raw web data is noisy, and existing filtering methods to reduce noise often come at the expense of data diversity. Our work focuses on caption quality as one major source of noise, and studies how generated captions can increase the utility of web-scraped datapoints with nondescript text. Through exploring different mixing strategies for raw and generated captions, we outperform the best filtering method proposed by the DataComp benchmark by 2% on ImageNet and 4% on average across 38 tasks, given a candidate pool of 128M image-text pairs. Our best approach is also 2x better at Flickr and MS-COCO retrieval. We then analyze what makes synthetic captions an effective source of text supervision. In experimenting with different image captioning models, we also demonstrate that the performance of a model on standard image captioning benchmarks (e.g., NoCaps CIDEr) is not a reliable indicator of the utility of the captions it generates for multimodal training. Finally, our experiments with using generated captions at DataComp's large scale (1.28B image-text pairs) offer insights into the limitations of synthetic text, as well as the importance of image curation with increasing training data quantity. The synthetic captions used in our experiments are now available on HuggingFace.

</details>

---

## 83. How hard are computer vision datasets? Calibrating dataset difficulty to viewing time

- [ ] How hard are computer vision datasets? Calibrating dataset difficulty to viewing time | https://neurips.cc/virtual/2023/poster/73596

- **Link**: https://neurips.cc/virtual/2023/poster/73596

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Humans outperform object recognizers despite the fact that models perform well on current datasets, including those explicitly designed to challenge machines with debiased images or distribution shift. This problem persists, in part, because we have no guidance on the absolute difficulty of an image or dataset making it hard to objectively assess progress toward human-level performance, to cover the range of human abilities, and to increase the challenge posed by a dataset. We develop a dataset difficulty metric MVT, Minimum Viewing Time, that addresses these three problems. Subjects view an image that flashes on screen and then classify the object in the image. Images that require brief flashes to recognize are easy, those which require seconds of viewing are hard. We compute the ImageNet and ObjectNet image difficulty distribution, which we find significantly undersamples hard images. Nearly 90% of current benchmark performance is derived from images that are easy for humans. Rather than hoping that we will make harder datasets, we can for the first time objectively guide dataset difficulty during development. We can also subset recognition performance as a function of difficulty: model performance drops precipitously while human performance remains stable. Difficulty provides a new lens through which to view model performance, one which uncovers new scaling laws: vision-language models stand out as being the most robust and human-like while all other techniques scale poorly. We release tools to automatically compute MVT, along with image sets which are tagged by difficulty. Objective image difficulty has practical applications – one can measure how hard a test set is before deploying a real-world system – and scientific applications such as discovering the neural correlates of image difficulty and enabling new object recognition techniques that eliminate the benchmark-vs- real-world performance gap.

</details>

---

## 84. Quilt-1M: One Million Image-Text Pairs for Histopathology

- [ ] Quilt-1M: One Million Image-Text Pairs for Histopathology | https://neurips.cc/virtual/2023/poster/73608

- **Link**: https://neurips.cc/virtual/2023/poster/73608

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent accelerations in multi-modal applications have been made possible with the plethora of image and text data available online. However, the scarcity of analogous data in the medical field, specifically in histopathology, has slowed comparable progress. To enable similar representation learning for histopathology, we turn to YouTube, an untapped resource of videos, offering $1,087$ hours of valuable educational histopathology videos from expert clinicians.From YouTube, we curate QUILT: a large-scale vision-language dataset consisting of $802, 144$ image and text pairs.QUILT was automatically curated using a mixture of models, including large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition.In comparison, the most comprehensive datasets curated for histopathology amass only around $200$K samples.We combine QUILT with datasets from other sources, including Twitter, research papers, and the internet in general, to create an even larger dataset: QUILT-1M, with $1$M paired image-text samples, marking it as the largest vision-language histopathology dataset to date. We demonstrate the value of QUILT-1M by fine-tuning a pre-trained CLIP model. Our model outperforms state-of-the-art models on both zero-shot and linear probing tasks for classifying new histopathology images across $13$ diverse patch-level datasets of $8$ different sub-pathologies and cross-modal retrieval tasks.

</details>

---

## 85. LOVM: Language-Only Vision Model Selection

- [ ] LOVM: Language-Only Vision Model Selection | https://neurips.cc/virtual/2023/poster/73618

- **Link**: https://neurips.cc/virtual/2023/poster/73618

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained multi-modal vision-language models (VLMs) are becoming increasingly popular due to their exceptional performance on downstream vision applications, particularly in the few- and zero-shot settings. However, selecting the best-performing VLM for some downstream applications is non-trivial, as it is dataset and task-dependent. Meanwhile, the exhaustive evaluation of all available VLMs on a novel application is not only time and  computationally demanding but also necessitates the collection of a labeled dataset for evaluation. As the number of open-source VLM variants increases, there is a need for an efficient model selection strategy that does not require access to a curated evaluation dataset. This paper proposes a novel task and benchmark for efficiently evaluating VLMs' zero-shot performance on downstream applications without access to the downstream task dataset. Specifically, we introduce a new task LOVM: L anguage- O nly V ision M odel Selection , where methods are expected to perform both model selection and performance prediction based solely on a text description of the desired downstream application. We then introduced an extensive LOVM benchmark consisting of ground-truth evaluations of 35 pre-trained VLMs and 23 datasets, where methods are expected to rank the pre-trained VLMs and predict their zero-shot performance.

</details>

---

## 86. SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality

- [ ] SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality | https://neurips.cc/virtual/2023/poster/73628

- **Link**: https://neurips.cc/virtual/2023/poster/73628

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In the last year alone, a surge of new benchmarks to measure $\textit{compositional}$ understanding of vision-language models have permeated the machine learning ecosystem.Given an image, these benchmarks probe a model's ability to identify its associated caption amongst a set of compositional distractors.Surprisingly, we find significant biases in $\textit{all}$ these benchmarks rendering them hackable. This hackability is so dire that blind models with no access to the image outperform state-of-the-art vision-language models.To remedy this rampant vulnerability, we introduce $\textit{SugarCrepe}$, a new benchmark for vision-language compositionality evaluation.We employ large language models, instead of rule-based templates used in previous benchmarks, to generate fluent and sensical hard negatives, and utilize an adversarial refinement mechanism to maximally reduce biases. We re-evaluate state-of-the-art models and recently proposed compositionality inducing strategies, and find that their improvements were hugely overestimated, suggesting that more innovation is needed in this important direction.We release $\textit{SugarCrepe}$ and the code for evaluation at: https://github.com/RAIVNLab/sugar-crepe.

</details>

---

## 87. EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding

- [ ] EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding | https://neurips.cc/virtual/2023/poster/73630

- **Link**: https://neurips.cc/virtual/2023/poster/73630

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce EgoSchema, a very long-form video question-answering dataset, and benchmark to evaluate long video understanding capabilities of modern vision and language systems. Derived from Ego4D, EgoSchema consists of over 5000 human curated multiple choice question answer pairs, spanning over 250 hours of real video data, covering a very broad range of natural human activity and behavior. For each question, EgoSchema requires the correct answer to be selected between five given options based on a three-minute-long video clip. While some prior works have proposed video datasets with long clip lengths, we posit that merely the length of the video clip does not truly capture the temporal difficulty of the video task that is being considered. To remedy this, we introduce temporal certificate sets, a general notion for capturing the intrinsic temporal understanding length associated with a broad range of video understanding tasks & datasets. Based on this metric, we find EgoSchema to have intrinsic temporal lengths over 5.7x longer than the second closest dataset and 10x to 100x longer than any other video understanding dataset. Further, our evaluation of several current state-of-the-art video and language models shows them to be severely lacking in long-term video understanding capabilities. Even models with several billions of parameters achieve QA accuracy less than 33% (random is 20%) on the EgoSchema multi-choice question answering task, while humans achieve about 76% accuracy. We posit that EgoSchema, with its long intrinsic temporal structures and diverse complexity, would serve as a valuable evaluation probe for developing effective long-term video understanding systems in the future. Data and Zero-shot model evaluation code will all be open-sourced under the Ego4D license at http://egoschema.github.io.

</details>

---

## 88. LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day

- [ ] LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day | https://neurips.cc/virtual/2023/poster/73643

- **Link**: https://neurips.cc/virtual/2023/poster/73643

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Conversational generative AI has demonstrated remarkable promise for empowering biomedical practitioners, but current investigations focus on unimodal text. Multimodal conversational AI has seen rapid progress by leveraging billions of image-text pairs from the public web, but such general-domain vision-language models still lack sophistication in understanding and conversing about biomedical images. In this paper, we propose a cost-efficient approach for training a vision-language conversational assistant that can answer open-ended research questions of biomedical images. The key idea is to leverage a large-scale, broad-coverage biomedical figure-caption dataset extracted from PubMed Central, use GPT-4 to self-instruct open-ended instruction-following data from the captions, and then fine-tune a large general-domain vision-language model using a novel curriculum learning method. Specifically, the model first learns to align biomedical vocabulary using the figure-caption pairs as is, then learns to master open-ended conversational semantics using GPT-4 generated instruction-following data, broadly mimicking how a layperson gradually acquires biomedical knowledge. This enables us to train a Large Language and Vision Assistant for BioMedicine (LLaVA-Med) in less than 15 hours (with eight A100s). LLaVA-Med exhibits excellent multimodal conversational capability and can follow open-ended instruction to assist with inquiries about a biomedical image. On three standard biomedical visual question answering datasets, LLaVA-Med outperforms previous supervised state-of-the-art on certain metrics. To facilitate biomedical multimodal research, we will release our instruction-following data and the LLaVA-Med model.

</details>

---

## 89. ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification

- [ ] ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification | https://neurips.cc/virtual/2023/poster/73660

- **Link**: https://neurips.cc/virtual/2023/poster/73660

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Image classifiers are information-discarding machines, by design. Yet, how these models discard information remains mysterious. We hypothesize that one way for image classifiers to reach high accuracy is to first zoom to the most discriminative region in the image and then extract features from there to predict image labels, discarding the rest of the image. Studying six popular networks ranging from AlexNet to CLIP, we find that proper framing of the input image can lead to the correct classification of 98.91% of ImageNet images. Furthermore, we  uncover positional biases in various datasets, especially a strong center bias in two popular datasets: ImageNet-A and ObjectNet. Finally, leveraging our insights into the potential of zooming, we propose a test-time augmentation (TTA) technique that improves classification accuracy by forcing models to explicitly perform zoom-in operations before making predictions.Our method is more interpretable, accurate, and faster than MEMO, a state-of-the-art (SOTA) TTA method. We introduce ImageNet-Hard, a new benchmark that challenges SOTA classifiers including large vision-language models even when optimal zooming is allowed.

</details>

---

## 90. VisoGender:  A dataset for benchmarking gender bias in image-text pronoun resolution

- [ ] VisoGender:  A dataset for benchmarking gender bias in image-text pronoun resolution | https://neurips.cc/virtual/2023/poster/73666

- **Link**: https://neurips.cc/virtual/2023/poster/73666

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce VisoGender, a novel dataset for benchmarking gender bias in vision-language models. We focus on occupation-related biases within a hegemonic system of binary gender, inspired by Winograd and Winogender schemas, where each image is associated with a caption containing a pronoun relationship of subjects and objects in the scene. VisoGender is balanced by gender representation in professional roles, supporting bias evaluation in two ways: i) resolution bias, where we evaluate the difference between pronoun resolution accuracies for image subjects with gender presentations perceived as masculine versus feminine by human annotators and ii) retrieval bias, where we compare ratios of professionals perceived to have masculine and feminine gender presentations retrieved for a gender-neutral search query. We benchmark several state-of-the-art vision-language models and find that they demonstrate bias in resolving binary gender in complex scenes. While the direction and magnitude of gender bias depends on the task and the model being evaluated, captioning models are generally less biased than Vision-Language Encoders.

</details>

---

## 91. What a MESS: Multi-Domain Evaluation of Zero-Shot Semantic Segmentation

- [ ] What a MESS: Multi-Domain Evaluation of Zero-Shot Semantic Segmentation | https://neurips.cc/virtual/2023/poster/73669

- **Link**: https://neurips.cc/virtual/2023/poster/73669

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

While semantic segmentation has seen tremendous improvements in the past, there are still significant labeling efforts necessary and the problem of limited generalization to classes that have not been present during training. To address this problem, zero-shot semantic segmentation makes use of large self-supervised vision-language models, allowing zero-shot transfer to unseen classes. In this work, we build a benchmark for Multi-domain Evaluation of Zero-Shot Semantic Segmentation (MESS), which allows a holistic analysis of performance across a wide range of domain-specific datasets such as medicine, engineering, earth monitoring, biology, and agriculture. To do this, we reviewed 120 datasets, developed a taxonomy, and classified the datasets according to the developed taxonomy. We select a representative subset consisting of 22 datasets and propose it as the MESS benchmark. We evaluate eight recently published models on the proposed MESS benchmark and analyze characteristics for the performance of zero-shot transfer models. The toolkit is available at https://github.com/blumenstiel/MESS.

</details>

---

## 92. COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs

- [ ] COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs | https://neurips.cc/virtual/2023/poster/73688

- **Link**: https://neurips.cc/virtual/2023/poster/73688

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Counterfactual examples have proven to be valuable in the field of natural language processing (NLP) for both evaluating and improving the robustness of language models to spurious correlations in datasets. Despite their demonstrated utility for NLP, multimodal counterfactual examples have been relatively unexplored due to the difficulty of creating paired image-text data with minimal counterfactual changes. To address this challenge, we introduce a scalable framework for automatic generation of counterfactual examples using text-to-image diffusion models. We use our framework to create COCO-Counterfactuals, a multimodal counterfactual dataset of paired image and text captions based on the MS-COCO dataset. We validate the quality of COCO-Counterfactuals through human evaluations and show that existing multimodal models are challenged by our counterfactual image-text pairs. Additionally, we demonstrate the usefulness of COCO-Counterfactuals for improving out-of-domain generalization of multimodal vision-language models via training data augmentation. We make our code and the COCO-Counterfactuals dataset publicly available.

</details>

---

## 93. Into the LAION’s Den: Investigating Hate in Multimodal Datasets

- [ ] Into the LAION’s Den: Investigating Hate in Multimodal Datasets | https://neurips.cc/virtual/2023/poster/73694

- **Link**: https://neurips.cc/virtual/2023/poster/73694

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

`Scale the model, scale the data, scale the compute' is the reigning sentiment in the world of generative AI today. While the impact of model scaling has been extensively studied, we are only beginning to scratch the surface of data scaling and its consequences. This is especially of critical importance in the context of vision-language datasets such as LAION. These datasets are continually growing in size and are built based on large-scale internet dumps such as the Common Crawl, which is known to have numerous drawbacks ranging from quality, legality, and content. The datasets then serve as the backbone for large generative models, contributing to the operationalization and perpetuation of harmful societal and historical biases and stereotypes. In this paper, we investigate the effect of scaling datasets on hateful content through a comparative audit of two datasets: LAION-400M and LAION-2B. Our results show that hate content increased by nearly 12% with dataset scale, measured both qualitatively and quantitatively using a metric that we term as Hate Content Rate (HCR). We also found that filtering dataset contents based on Not Safe For Work (NSFW) values calculated based on images alone does not exclude all the harmful content in alt-text. Instead, we found that trace amounts of hateful, targeted, and aggressive text remain even when carrying out conservative filtering. We end with a reflection and a discussion of the significance of our results for dataset curation and usage in the AI community.Code and the meta-data assets curated in this paper are publicly available at https://github.com/vinayprabhu/hate_scaling. Content warning: This paper contains examples of hateful text that might be disturbing, distressing, and/or offensive.

</details>

---

## 94. Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models

- [ ] Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models | https://neurips.cc/virtual/2023/poster/73702

- **Link**: https://neurips.cc/virtual/2023/poster/73702

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Various adaptation methods, such as LoRA, prompts, and adapters, have been proposed to enhance the performance of pre-trained vision-language models in specific domains. As test samples in real-world applications usually differ from adaptation data, the robustness of these adaptation methods against distribution shifts are essential. In this study, we assess the robustness of 11 widely-used adaptation methods across 4 vision-language datasets under multimodal corruptions. Concretely, we introduce 7 benchmark datasets, including 96 visual and 87 textual corruptions, to investigate the robustness of different adaptation methods, the impact of available adaptation examples, and the influence of trainable parameter size during adaptation. Our analysis reveals that: 1) Adaptation methods are more sensitive to text corruptions than visual corruptions. 2) Full fine-tuning does not consistently provide the highest robustness; instead, adapters can achieve better robustness with comparable clean performance. 3) Contrary to expectations, our findings indicate that increasing the number of adaptation data and parameters does not guarantee enhanced robustness; instead, it results in even lower robustness. We hope this study could benefit future research in the development of robust multimodal adaptation methods. The benchmark, code, and dataset used in this study can be accessed at https://adarobustness.github.io.

</details>

---

## 95. What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks

- [ ] What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks | https://neurips.cc/virtual/2023/poster/73716

- **Link**: https://neurips.cc/virtual/2023/poster/73716

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) with strong abilities in natural language processing tasks have emerged and have been applied in various kinds of areas such as science, finance and software engineering. However, the capability of LLMs to advance the field of chemistry remains unclear. In this paper, rather than pursuing state-of-the-art performance, we aim to evaluate capabilities of LLMs in a wide range of tasks across the chemistry domain. We identify three key chemistry-related capabilities including understanding, reasoning and explaining to explore in LLMs and establish a benchmark containing eight chemistry tasks. Our analysis draws on widely recognized datasets facilitating a broad exploration of the capacities of LLMs within the context of practical chemistry. Five LLMs (GPT-4,GPT-3.5, Davinci-003, Llama and Galactica) are evaluated for each chemistry task in zero-shot and few-shot in-context learning settings with carefully selected demonstration examples and specially crafted prompts. Our investigation found that GPT-4 outperformed other models and LLMs exhibit different competitive levels in eight chemistry tasks. In addition to the key findings from the comprehensive benchmark analysis, our work provides insights into the limitation of current LLMs and the impact of in-context learning settings on LLMs’ performance across various chemistry tasks. The code and datasets used in this study are available at https://github.com/ChemFoundationModels/ChemLLMBench.

</details>

---

## 96. Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks

- [ ] Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks | https://neurips.cc/virtual/2023/poster/73713

- **Link**: https://neurips.cc/virtual/2023/poster/73713

- **Conference**: NeurIPS

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Neural network based computer vision systems are typically built on a backbone, a pretrained or randomly initialized feature extractor.  Several years ago, the default option was an ImageNet-trained convolutional neural network.  However, the recent past has seen the emergence of countless backbones pretrained using various algorithms and datasets. While this abundance of choice has led to performance increases for a range of systems, it is difficult for practitioners to make informed decisions about which backbone to choose.  Battle of the Backbones (BoB) makes this choice easier by benchmarking a diverse suite of pretrained models, including vision-language models, those trained via self-supervised learning, and the Stable Diffusion backbone, across a diverse set of computer vision tasks ranging from classification to object detection to OOD generalization and more.  Furthermore, BoB sheds light on promising directions for the research community to advance computer vision by illuminating strengths and weakness of existing approaches through a comprehensive analysis conducted on more than 1500 training runs.  While vision transformers (ViTs) and self-supervised learning (SSL) are increasingly popular, we find that convolutional neural networks pretrained in a supervised fashion on large training sets still perform best on most tasks among the models we consider. Moreover, in apples-to-apples comparisons on the same architectures and similarly sized pretraining datasets, we find that SSL backbones are highly competitive, indicating that future works should perform SSL pretraining with advanced architectures and larger pretraining datasets.  We release the raw results of our experiments along with code that allows researchers to put their own backbones through the gauntlet here: https://github.com/hsouri/Battle-of-the-Backbones.

</details>

---

