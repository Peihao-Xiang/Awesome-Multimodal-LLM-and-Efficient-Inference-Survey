# ICLR 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_iclr2023_papers.csv

## 1. HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention

- [ ] HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention | https://iclr.cc/virtual/2023/poster/10805

- **Link**: https://iclr.cc/virtual/2023/poster/10805

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The success of large-scale contrastive vision-language pretraining (CLIP) has benefited both visual recognition and multimodal content understanding. The concise design brings CLIP the advantage in inference efficiency against other vision-language models with heavier cross-attention fusion layers, making it a popular choice for a wide spectrum of downstream tasks. However, CLIP does not explicitly capture the hierarchical nature of high-level and fine-grained semantics conveyed in images and texts, which is arguably critical to vision-language understanding and reasoning. To this end, we equip both the visual and language branches in CLIP with hierarchy-aware attentions, namely Hierarchy-aware CLIP (HiCLIP), to progressively discover semantic hierarchies layer-by-layer from both images and texts in an unsupervised manner. As a result, such hierarchical aggregation significantly improves the cross-modal alignment. To demonstrate the advantages of HiCLIP, we conduct qualitative analysis on its unsupervised hierarchy induction during inference, as well as extensive quantitative experiments on both visual recognition and vision-language downstream tasks.

</details>

---

## 2. Spotlight: Mobile UI Understanding using Vision-Language Models with a Focus

- [ ] Spotlight: Mobile UI Understanding using Vision-Language Models with a Focus | https://iclr.cc/virtual/2023/poster/10862

- **Link**: https://iclr.cc/virtual/2023/poster/10862

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Mobile UI understanding is important for enabling various interaction tasks such as UI automation and accessibility. Previous mobile UI modeling often depends on the view hierarchy information of a screen, which directly provides the structural data of the UI, with the hope to bypass challenging tasks of visual modeling from screen pixels. However, view hierarchies are not always available, and are often corrupted with missing object descriptions or misaligned structure information. As a result, despite the use of view hierarchies could offer short-term gains, it may ultimately hinder the applicability and performance of the model. In this paper, we propose Spotlight, a vision-only approach for mobile UI understanding. Specifically, we enhance a vision-language model that only takes the screenshot of the UI and a region of interest on the screen---the focus---as the input. This general architecture of Spotlight is easily scalable and capable of performing a range of UI modeling tasks. Our experiments show that our model establishes SoTA results on several representative UI tasks and outperforms previous methods that use both screenshots and view hierarchies as inputs. Furthermore, we explore multi-task learning and few-shot prompting capacities of the proposed models, demonstrating promising results in the multi-task learning direction.

</details>

---

## 3. When and Why Vision-Language Models Behave like Bags-Of-Words, and What to Do About It?

- [ ] When and Why Vision-Language Models Behave like Bags-Of-Words, and What to Do About It? | https://iclr.cc/virtual/2023/poster/10875

- **Link**: https://iclr.cc/virtual/2023/poster/10875

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite the success of large vision and language models (VLMs) in many downstream applications, it is unclear how well they encode the compositional relationships between objects and attributes. Here, we create the Attribution, Relation, and Order (ARO) benchmark to systematically evaluate the ability of VLMs to understand different types of relationships, attributes, and order information. ARO consists of \emph{Visual Genome Attribution}, to test the understanding of objects' properties; \emph{Visual Genome Relation}, to test for relational understanding; and \emph{COCO-Order \& Flickr30k-Order}, to test for order sensitivity in VLMs. ARO is orders of magnitude larger than previous benchmarks of compositionality, with more than 50,000 test cases. We present the settings where  state-of-the-art VLMs behave like bags-of-words---i.e. when they have poor relational understanding, can blunder when linking objects to their attributes, and demonstrate a severe lack of order sensitivity. VLMs are predominantly trained and evaluated on large scale datasets with rich compositional structure in the images and captions. Yet, training on these datasets has not been enough to address the lack of compositional understanding, and evaluating on these datasets has failed to surface this deficiency. To understand why these limitations emerge and are not represented in the standard tests, we zoom into the evaluation and training procedures. We demonstrate that it is possible to perform well on image-text retrieval over existing datasets without using the composition and order information. This further motivates the value of using ARO to benchmark VLMs. Given that contrastive pretraining optimizes for retrieval on large datasets with similar shortcuts, we hypothesize that this can explain why the models do not need to learn to represent compositional information. This finding suggests a natural solution: composition-aware hard negative mining. We show that a simple-to-implement modification of contrastive learning significantly improves the performance on tasks requiring understanding of order and compositionality.

</details>

---

## 4. Compositional Prompt Tuning with Motion Cues for Open-vocabulary Video Relation Detection

- [ ] Compositional Prompt Tuning with Motion Cues for Open-vocabulary Video Relation Detection | https://iclr.cc/virtual/2023/poster/11182

- **Link**: https://iclr.cc/virtual/2023/poster/11182

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning with large-scale pretrained vision-language models empowers open-vocabulary prediction trained on limited base categories, e.g., object classification and detection. In this paper, we propose compositional prompt tuning with motion cues: an extended prompt tuning paradigm for compositional predictions of video data. In particular, we present Relation Prompt (RePro) for Open-vocabulary Video Visual Relation Detection (Open-VidVRD), where conventional prompt tuning is easily biased to certain subject-object combinations and motion patterns. To this end, RePro addresses the two technical challenges of Open-VidVRD: 1) the prompt tokens should respect the two different semantic roles of subject and object, and 2) the tuning should account for the diverse spatiotemporal motion patterns of the subject-object compositions. Our RePro achieves a new state-of-the-art performance on two VidVRD benchmarks of not only the base training object and predicate categories, but also the unseen ones. Extensive ablations also demonstrate the effectiveness of the proposed compositional and multi-mode design of prompt. Code is available at https://github.com/Dawn-LX/OpenVoc-VidVRD.

</details>

---

## 5. Visually-Augmented Language Modeling

- [ ] Visually-Augmented Language Modeling | https://iclr.cc/virtual/2023/poster/11283

- **Link**: https://iclr.cc/virtual/2023/poster/11283

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Human language is grounded on multimodal knowledge including visual knowledge like colors, sizes, and shapes. However, current large-scale pre-trained language models rely on the text-only self-supervised training with massive text data, which precludes them from utilizing relevant visual information when necessary. To address this, we propose a novel pre-training framework, named VaLM, to Visually-augment text tokens with retrieved relevant images for Language Modeling. Specifically, VaLM builds on a novel latent text-image alignment method via an image retrieval module to fetch corresponding images given a textual context. With the visually-augmented context, VaLM uses a visual knowledge fusion layer to enable multimodal grounded language modeling by attending on both text context and visual knowledge in images. We evaluate VaLM on various visual knowledge intensive commonsense reasoning tasks, which require visual information to excel. The experimental results illustrate that VaLM outperforms all strong language-only and vision-language baselines with substantial gains on reasoning object commonsense including color, size, and shape.

</details>

---

## 6. Open-Vocabulary Object Detection upon Frozen Vision and Language Models

- [ ] Open-Vocabulary Object Detection upon Frozen Vision and Language Models | https://iclr.cc/virtual/2023/poster/11429

- **Link**: https://iclr.cc/virtual/2023/poster/11429

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present F-VLM, a simple open-vocabulary object detection method built uponFrozenVision andLanguageModels.  F-VLM simplifies the current multi-stage training pipeline by eliminating the need for knowledge distillation or detection-tailored pretraining.  Surprisingly, we observe that a frozen VLM: 1) retains the locality-sensitive features necessary for detection, and 2) is a strong region classifier.  We finetune only the detector head and combine the detector and VLM outputs for each region at inference time.  F-VLM shows compelling scaling behavior and achieves +6.5 mask AP improvement over the previous state of theart  on  novel  categories  of  LVIS  open-vocabulary  detection  benchmark. In addition, we demonstrate very competitive results on COCO open-vocabulary detection benchmark and cross-dataset transfer detection, in addition to significant training speed-up and compute savings. Code will be released.

</details>

---

## 7. Boosting Adversarial Transferability using Dynamic Cues

- [ ] Boosting Adversarial Transferability using Dynamic Cues | https://iclr.cc/virtual/2023/poster/11498

- **Link**: https://iclr.cc/virtual/2023/poster/11498

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The transferability of adversarial perturbations between image models has been extensively studied. In this case, an attack is generated from a known surrogate \eg, the ImageNet trained model, and transferred to change the decision of an unknown (black-box) model trained on an image dataset. However, attacks generated from image models do not capture the dynamic nature of a moving object or a changing scene due to a lack of temporal cues within image models. This leads to reduced transferability of adversarial attacks from representation-enriched \emph{image} models such as Supervised Vision Transformers (ViTs), Self-supervised ViTs (\eg, DINO), and Vision-language models (\eg, CLIP) to black-box \emph{video} models. In this work, we induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize \emph{temporal prompts} through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics. Specifically, we introduce spatial (image) and temporal (video) cues within the same source model through task-specific prompts. Attacking such prompts maximizes the adversarial transferability from image-to-video and image-to-image models using the attacks designed for image models. As an example, an iterative attack launched from image model Deit-B with temporal prompts reduces generalization (top1 \% accuracy) of a video model by 35\% on Kinetics-400. Our approach also improves adversarial transferability to image models by 9\% on ImageNet w.r.t the current state-of-the-art approach. Our attack results indicate that the attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time. Code is available at \url{https://bit.ly/3Xd9gRQ}

</details>

---

## 8. MEDICAL IMAGE UNDERSTANDING WITH PRETRAINED VISION LANGUAGE MODELS: A COMPREHENSIVE STUDY

- [ ] MEDICAL IMAGE UNDERSTANDING WITH PRETRAINED VISION LANGUAGE MODELS: A COMPREHENSIVE STUDY | https://iclr.cc/virtual/2023/poster/11569

- **Link**: https://iclr.cc/virtual/2023/poster/11569

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The large-scale pre-trained vision language models (VLM) have shown remarkable domain transfer capability on natural images. However, it remains unknown whether this capability can also apply to the medical image domain. This paper thoroughly studies the knowledge transferability of pre-trained VLMs to the medical domain, where we show that well-designed medical prompts are the key to elicit knowledge from pre-trained VLMs. We demonstrate that by prompting with expressive attributes that are shared between domains, the VLM can carry the knowledge across domains and improve its generalization. This mechanism empowers VLMs to recognize novel objects with fewer or without image samples. Furthermore, to avoid the laborious manual designing process, we develop three approaches for automatic generation of medical prompts, which can inject expert-level medical knowledge and image-specific information into the prompts for fine-grained grounding. We conduct extensive experiments on thirteen different medical datasets across various modalities, showing that our well-designed prompts greatly improve the zero-shot performance compared to the default prompts, and our fine-tuned models surpass the supervised models by a significant margin.

</details>

---

## 9. CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Alignment

- [ ] CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Alignment | https://iclr.cc/virtual/2023/poster/11587

- **Link**: https://iclr.cc/virtual/2023/poster/11587

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained image-text models, like CLIP, have demonstrated the strong power of vision-language representation learned from a large scale of web-collected image-text data. In light of the well-learned visual features, there are works that transfer image representation to the video domain and achieve good results. However, adapting image-text pre-trained models to video-text pre-training (i.e., post-pretraining) has not demonstrated a significant advantage yet. In this paper, we tackle this challenge by raising and addressing two questions: 1) what are the factors hindering post-pretraining CLIP from improving performance on video-text tasks, and 2) how to mitigate the impact of these factors. Through a series of comparative experiments and analyses, we find that the data scale and domain gap between language sources have large impacts. By these observations, we propose an Omnisource Cross-modal Learning method equipped with a Video Proxy mechanism on the basis of CLIP, namely CLIP-ViP. Extensive results show that our approach improves the performance of CLIP on video-text retrieval by a large margin. Our model achieves state-of-the-art results on a variety of datasets, including MSR-VTT, DiDeMo, LSMDC, and ActivityNet. We release our code and pre-trained CLIP-ViP models at \url{https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP}.

</details>

---

## 10. Learning to Decompose Visual Features with Latent Textual Prompts

- [ ] Learning to Decompose Visual Features with Latent Textual Prompts | https://iclr.cc/virtual/2023/poster/11608

- **Link**: https://iclr.cc/virtual/2023/poster/11608

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in pre-training vision-language models like CLIP have shown great potential in learning transferable visual representations. Nonetheless, for downstream inference, CLIP-like models suffer from either 1) degraded accuracy and robustness in the case of inaccurate text descriptions during retrieval-based inference (the challenge for zero-shot protocol); or 2) breaking the well-established vision-language alignment (the challenge for linear probing). To address them, we propose Decomposed Feature Prompting (DeFo). DeFo leverages a flexible number of learnable embeddings as textual input while maintaining the vision-language dual-model architecture, which enables the model to learn decomposed visual features with the help of feature-level textual prompts. We further use an additional linear layer to perform classification, allowing a scalable size of language inputs. Our empirical study shows DeFo's significance in improving the vision-language models. For example, DeFo obtains 73.2% test accuracy on ImageNet with a ResNet-50 backbone without tuning any pretrained weights of both the vision and language encoder, outperforming zero-shot CLIP by a large margin of 15.0%, and outperforming state-of-the-art vision-language prompt tuning method by 7.6%.

</details>

---

## 11. Contrastive Alignment of Vision to Language Through Parameter-Efficient Transfer Learning

- [ ] Contrastive Alignment of Vision to Language Through Parameter-Efficient Transfer Learning | https://iclr.cc/virtual/2023/poster/11712

- **Link**: https://iclr.cc/virtual/2023/poster/11712

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive vision-language models (e.g. CLIP) are typically created by updating all the parameters of a vision model and language model through contrastive training. Can such models be created by a small number of parameter updates to an already-trained language model and vision model? The literature describes techniques that can create vision-language models by updating a small number of parameters in a language model, but these require already aligned visual representations and are non-contrastive, hence unusable for latency-sensitive applications such as neural search. We explore the feasibility and benefits of parameter-efficient contrastive vision-language alignment through transfer learning: creating a model such as CLIP by minimally updating an already-trained vision and language model. We find that a minimal set of parameter updates ($<$7\%) can achieve the same performance as full-model training, and updating specific components ($<$1\% of parameters) can match 75\% of full-model training. We describe a series of experiments: we show that existing knowledge is conserved more strongly in parameter-efficient training and that parameter-efficient scaling scales with model and dataset size. Where paired-image text data is scarce but strong multilingual language models exist (e.g. low resource languages), parameter-efficient training is even preferable to full-model training. Given a fixed compute budget, parameter-efficient training allows training larger models on the same hardware, achieving equivalent performance in less time. Parameter-efficient training hence constitutes an energy-efficient and effective training strategy for contrastive vision-language models that may be preferable to the full-model training paradigm for common use cases.Code and weights at https://github.com/codezakh/LilT.

</details>

---

## 12. Write and Paint: Generative Vision-Language Models are Unified Modal Learners

- [ ] Write and Paint: Generative Vision-Language Models are Unified Modal Learners | https://iclr.cc/virtual/2023/poster/11724

- **Link**: https://iclr.cc/virtual/2023/poster/11724

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language pre-training have pushed the state-of-the-art on various vision-language tasks, making machines more capable of multi-modal writing (image-to-text generation) and painting (text-to-image generation). However, few studies investigate if these two essential capabilities can be learned together and boost each other, making a versatile and powerful multi-modal foundation model. In this work, we disclose the potential of symmetric generative vision-language pre-training in learning to write and paint concurrently, and propose a new unified modal model, named DaVinci, trained with prefix language modeling and prefix image modeling, a simple generative self-supervised objective on image-text pairs. Thanks to the proposed prefix multi-modal modeling framework, DaVinci is simple to train, scalable to huge data, adaptable to both writing and painting tasks, and also strong on other vision, text, and multi-modal understanding tasks. DaVinci achieves competitive performance on a wide range of 27 generation/understanding tasks and demonstrates the superiority of combining vision/language generative pre-training. Furthermore, we carefully benchmark the performance of different vision-language pre-training objectives on different scales of pre-training datasets on a heterogeneous and broad distribution coverage. Our results demonstrate the potential of exploiting self-supervision in both language and vision inputs, and establish new, stronger baselines for future comparisons at different data scales. The code and pre-trained models are available at https://github.com/shizhediao/DaVinci.

</details>

---

## 13. Visual Classification via Description from Large Language Models

- [ ] Visual Classification via Description from Large Language Models | https://iclr.cc/virtual/2023/poster/11733

- **Link**: https://iclr.cc/virtual/2023/poster/11733

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models such as CLIP have shown promising performance on a variety of recognition tasks using the standard zero-shot classification procedure -- computing similarity between the query image and the embedded words for each category. By only using the category name, they neglect to make use of the rich context of additional information that language affords. The procedure gives no intermediate understanding of why a category is chosen, and furthermore provides no mechanism for adjusting the criteria used towards this decision. We present an alternative framework for classification with VLMs, which we call classification by description. We ask VLMs to check for descriptive features rather than broad categories: to find a tiger, look for its stripes; its claws; and more. By basing decisions on these descriptors, we can provide additional cues that encourage using the features we want to be used. In the process, we can get a clear idea of what the model ``thinks" it is seeing to make its decision; it gains some level of inherent explainability. We query large language models (e.g., GPT-3) for these descriptors to obtain them in a scalable way. Extensive experiments show our framework has numerous advantages past interpretability. We show improvements in accuracy on ImageNet across distribution shifts; demonstrate the ability to adapt VLMs to recognize concepts unseen during training; and illustrate how descriptors can be edited to effectively mitigate bias compared to the baseline.

</details>

---

## 14. Unified Discrete Diffusion for Simultaneous Vision-Language Generation

- [ ] Unified Discrete Diffusion for Simultaneous Vision-Language Generation | https://iclr.cc/virtual/2023/poster/11819

- **Link**: https://iclr.cc/virtual/2023/poster/11819

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The recently developed discrete diffusion model performs extraordinarily well in generation tasks, especially in the text-to-image task, showing great potential for modeling multimodal signals. In this paper, we leverage these properties and present a unified multimodal generation model, which can perform text-based, image-based, and even vision-language simultaneous generation using a single model. Specifically, we unify the discrete diffusion process for multimodal signals by proposing a unified Markov transition matrix and a unified objective. Moreover, we design a multimodal mutual attention module to highlight the inter-modal linkages, which is vital for multimodal generation. Extensive experiments indicate that our proposed method can perform comparably to the state-of-the-art solutions in various generation tasks.

</details>

---

## 15. Programmatically Grounded, Compositionally Generalizable Robotic Manipulation

- [ ] Programmatically Grounded, Compositionally Generalizable Robotic Manipulation | https://iclr.cc/virtual/2023/poster/11828

- **Link**: https://iclr.cc/virtual/2023/poster/11828

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Robots operating in the real world require both rich manipulation skills as well as the ability to semantically reason about when to apply those skills. Towards this goal, recent works have integrated semantic representations from large-scale pretrained vision-language (VL) models into manipulation models, imparting them with more general reasoning capabilities. However, we show that the conventional {\it pretraining-finetuning} pipeline for integrating such representations entangles the learning of domain-specific action information and domain-general visual information, leading to less data-efficient training and poor generalization to unseen objects and tasks. To this end, we propose \ours, a {\it modular} approach to better leverage pretrained VL models by exploiting the syntactic and semantic structures of language instructions. Our framework uses a semantic parser to recover an executable program, composed of functional modules grounded on vision and action across different modalities. Each functional module is realized as a combination of deterministic computation and learnable neural networks. Program execution produces parameters to general manipulation primitives for a robotic end-effector. The entire modular network can be trained with end-to-end imitation learning objectives. Experiments show that our model successfully disentangles action and perception, translating to improved zero-shot and compositional generalization in a variety of manipulation behaviors. Project webpage at: \url{https://progport.github.io}.

</details>

---

## 16. Understanding Zero-shot Adversarial Robustness for Large-Scale Models

- [ ] Understanding Zero-shot Adversarial Robustness for Large-Scale Models | https://iclr.cc/virtual/2023/poster/11946

- **Link**: https://iclr.cc/virtual/2023/poster/11946

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pretrained large-scale vision-language models like CLIP have exhibited strong generalization over unseen tasks. Yet imperceptible adversarial perturbations can significantly reduce CLIP's performance on new tasks. In this work, we identify and explore the problem of adapting large-scale models for zero-shot adversarial robustness. We first identify two key factors during model adaption--training losses and adaptation methods--that affect the model's zero-shot adversarial robustness. We then propose a text-guided contrastive adversarial training loss, which aligns the text embeddings and the adversarial visual features with contrastive learning on a small set of training data. We apply this training loss to two adaption methods, model finetuning and visual prompt tuning. We find that visual prompt tuning is more effective in the absence of texts, while finetuning wins in the existence of text guidance. Overall, our approach significantly improves the zero-shot adversarial robustness over CLIP, seeing an average improvement of 31 points over ImageNet and 15 zero-shot datasets. We hope this work can shed light on understanding the zero-shot adversarial robustness of large-scale models.

</details>

---

## 17. Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval

- [ ] Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval | https://iclr.cc/virtual/2023/poster/11952

- **Link**: https://iclr.cc/virtual/2023/poster/11952

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper presents Universal Vision-Language Dense Retrieval (UniVL-DR), which builds a unified model for multi-modal retrieval. UniVL-DR encodes queries and multi-modality resources in an embedding space for searching candidates from different modalities. To learn a unified embedding space for multi-modal retrieval, UniVL-DR proposes two techniques: 1) Universal embedding optimization strategy, which contrastively optimizes the embedding space using the modality-balanced hard negatives; 2) Image verbalization method, which bridges the modality gap between images and texts in the raw data space. UniVL-DR achieves the state-of-the-art on the multi-modal open-domain question answering benchmark, WebQA, and outperforms all retrieval models on the two subtasks, text-text retrieval and text-image retrieval. It demonstrates that universal multi-modal search is feasible to replace the divide-and-conquer pipeline with a united model and also benefits single/cross modality tasks. All source codes of this work are available at https://github.com/OpenMatch/UniVL-DR.

</details>

---

## 18. PLOT: Prompt Learning with Optimal Transport for Vision-Language Models

- [ ] PLOT: Prompt Learning with Optimal Transport for Vision-Language Models | https://iclr.cc/virtual/2023/poster/12058

- **Link**: https://iclr.cc/virtual/2023/poster/12058

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

With the increasing attention to large vision-language models such as CLIP, there has been a significant amount of effort dedicated to building efficient prompts. Unlike conventional methods of only learning one single prompt, we propose to learn multiple comprehensive prompts to describe diverse characteristics of categories such as intrinsic attributes or extrinsic contexts. However, directly matching each prompt to the same visual feature is problematic, as it pushes the prompts to converge to one point. To solve this problem, we propose to apply optimal transport to match the vision and text modalities. Specifically, we first model images and the categories with visual and textual feature sets. Then, we apply a two-stage optimization strategy to learn the prompts. In the inner loop, we optimize the optimal transport distance to align visual features and prompts by the Sinkhorn algorithm, while in the outer loop, we learn the prompts by this distance from the supervised data. Extensive experiments are conducted on the few-shot recognition task and the improvement demonstrates the superiority of our method. The code is available at https://github.com/CHENGY12/PLOT.

</details>

---

## 19. Learning to Compose Soft Prompts for Compositional Zero-Shot Learning

- [ ] Learning to Compose Soft Prompts for Compositional Zero-Shot Learning | https://iclr.cc/virtual/2023/poster/12162

- **Link**: https://iclr.cc/virtual/2023/poster/12162

- **Conference**: ICLR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce compositional soft prompting (CSP), a parameter-efficient learning technique to improve the zero-shot compositionality of large-scale pretrained vision-language models (VLMs) like CLIP. We develop CSP for compositional zero-shot learning, the task of predicting unseen attribute-object compositions (e.g., old cat and young tiger). VLMs have a flexible text encoder that can represent arbitrary classes as natural language prompts but they often underperform task-specific architectures on the compositional zero-shot benchmark datasets. CSP treats the attributes and objects that define classes as learnable tokens of vocabulary. During training, the vocabulary is tuned to recognize classes that compose tokens in multiple ways (e.g., old cat and white cat). At test time, we recompose the learned attribute-object vocabulary in new combinations to recognize novel classes. We show that CSP outperforms the CLIP on benchmark datasets by an average of 10.9 percentage points on AUC. CSP also outperforms CoOp, a soft prompting method that fine-tunes the prefix context tokens, by an average of 5.8 percentage points on AUC. We perform additional experiments to show that CSP improves generalization to higher-order attribute-attribute-object compositions (e.g., old white cat) and combinations of pretrained attributes and fine-tuned objects. The code is available at https://github.com/BatsResearch/csp.

</details>

---

