# ICML 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_icml2023_papers.csv

## 1. One-shot Imitation in a Non-Stationary Environment via Multi-Modal Skill

- [ ] One-shot Imitation in a Non-Stationary Environment via Multi-Modal Skill | https://icml.cc/virtual/2023/poster/23606

- **Link**: https://icml.cc/virtual/2023/poster/23606

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

One-shot imitation is to learn a new task from a single demonstration, yet it is a challenging problem to adopt it for complex tasks with the high domain diversity inherent in a non-stationary environment. To tackle the problem, we explore the compositionality of complex tasks, and present a novel skill-based imitation learning framework enabling one-shot imitation and zero-shot adaptation; from a single demonstration for a complex unseen task, a semantic skill sequence is inferred and then each skill in the sequence is converted into an action sequence optimized for environmental hidden dynamics that can vary over time. Specifically, we leverage a vision-language model to learn a semantic skill set from offline video datasets, where each skill is represented on the vision-language embedding space, and adapt meta-learning with dynamics inference to enable zero-shot skill adaptation. We evaluate our framework with various one-shot imitation scenarios for extended multi-stage Meta-world tasks, showing its superiority in learning complex tasks, generalizing to dynamics changes, and extending to different demonstration conditions and modalities, compared to other baselines.

</details>

---

## 2. mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video

- [ ] mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video | https://icml.cc/virtual/2023/poster/23634

- **Link**: https://icml.cc/virtual/2023/poster/23634

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent years have witnessed a big convergence of language, vision, and multi-modal pretraining. In this work, we present mPLUG-2, a new unified paradigm with modularized design for multi-modal pretraining, which can benefit from modality collaboration while addressing the problem of modality entanglement. In contrast to predominant paradigms of solely relying on sequence-to-sequence generation or encoder-based instance discrimination, mPLUG-2 introduces a multi-module composition network by sharing common universal modules for modality collaboration and disentangling different modality modules to deal with modality entanglement. It is flexible to select different modules for different understanding and generation tasks across all modalities including text, image, and video. Empirical study shows that mPLUG-2 achieves state-of-the-art or competitive results on a broad range of over 30 downstream tasks, spanning multi-modal tasks of image-text and video-text understanding and generation, and uni-modal tasks of text-only, image-only, and video-only understanding. Notably, mPLUG-2 shows new state-of-the-art results of 48.0 top-1 accuracy and 80.3 CIDEr on the challenging MSRVTT video QA and video caption tasks with a far smaller model size and data scale. It also demonstrates strong zero-shot transferability on vision-language and video-language tasks. Code and models will be released in https://github.com/X-PLUG/mPLUG-2.

</details>

---

## 3. Identifying Interpretable Subspaces in Image Representations

- [ ] Identifying Interpretable Subspaces in Image Representations | https://icml.cc/virtual/2023/poster/23650

- **Link**: https://icml.cc/virtual/2023/poster/23650

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We propose Automatic Feature Explanation using Contrasting Concepts (FALCON), an interpretability framework to explain features of image representations. For a target feature, FALCON captions its highly activating cropped images using a large captioning dataset (like LAION-400m) and a pre-trained vision-language model like CLIP. Each word among the captions is scored and ranked leading to a small number of shared, human-understandable concepts that closely describe the target feature. FALCON also applies contrastive interpretation using lowly activating (counterfactual) images, to eliminate spurious concepts. Although many existing approaches interpret features independently, we observe in state-of-the-art self-supervised and supervised models, that less than 20% of the representation space can be explained by individual features. We show that features in larger spaces become more interpretable when studied in groups and can be explained with high-order scoring concepts through FALCON. We discuss how extracted concepts can be used to explain and debug failures in downstream tasks. Finally, we present a technique to transfer concepts from one (explainable) representation space to another unseen representation space by learning a simple linear transformation.

</details>

---

## 4. LIV: Language-Image Representations and Rewards for Robotic Control

- [ ] LIV: Language-Image Representations and Rewards for Robotic Control | https://icml.cc/virtual/2023/poster/23808

- **Link**: https://icml.cc/virtual/2023/poster/23808

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present Language-Image Value learning (LIV), a unified objective for vision-language representation and reward learning from action-free videos with text annotations. Exploiting a novel connection between dual reinforcement learning and mutual information contrastive learning, the LIV objective trains a multi-modal representation that implicitly encodes a universal value function for tasks specified as language or image goals. We use LIV to pre-train the first control-centric vision-language representation from large human video datasets such as EpicKitchen. Given only a language or image goal, the pre-trained LIV model can assign dense rewards to each frame in videos of unseen robots or humans attempting that task in unseen environments. Further, when some target domain-specific data is available, the same objective can be used to fine-tune and improve LIV and even other pre-trained representations for robotic control and reward specification in that domain. In our experiments on several simulated and real-world robot environments, LIV models consistently outperform the best prior input state representations for imitation learning, as well as reward specification methods for policy synthesis. Our results validate the advantages of joint vision-language representation and reward learning within the unified, compact LIV framework.

</details>

---

## 5. PaLM-E: An Embodied Multimodal Language Model

- [ ] PaLM-E: An Embodied Multimodal Language Model | https://icml.cc/virtual/2023/poster/23969

- **Link**: https://icml.cc/virtual/2023/poster/23969

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large language models excel at a wide range of complex tasks. However, enabling general inference in the real world, e.g. for robotics problems, raises the challenge of grounding. We propose embodied language models to directly incorporate real-world continuous sensor modalities into language models and thereby establish the link between words and percepts. Input to our embodied language model are multimodal sentences that interleave visual, continuous state estimation, and textual input encodings. We train these encodings end-to-end, in conjunction with a pre-trained large language model, for multiple embodied tasks including sequential robotic manipulation planning, visual question answering, and captioning. Our evaluations show that PaLM-E, a single large embodied multimodal model, can address a variety of embodied reasoning tasks, from a variety of observation modalities, on multiple embodiments, and further, exhibits positive transfer: the model benefits from diverse joint training across internet-scale language, vision, and visual-language domains. Our largest model with 562B parameters, in addition to being trained on robotics tasks, is a visual-language generalist with state-of-the-art performance on OK-VQA, and retains generalist language capabilities with increasing scale.

</details>

---

## 6. UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers

- [ ] UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers | https://icml.cc/virtual/2023/poster/23979

- **Link**: https://icml.cc/virtual/2023/poster/23979

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Real-world data contains a vast amount of multimodal information, among which vision and language are the two most representative modalities. Moreover, increasingly heavier models, e.g., Transformers, have attracted the attention of researchers to model compression. However, how to compress multimodal models, especially vison-language Transformers, is still under-explored. This paper proposes the Unified and Progressive Pruning (UPop) as a universal vison-language Transformer compression framework, which incorporates 1) unifiedly searching multimodal subnets in a continuous optimization space from the original model, which enables automatic assignment of pruning ratios among compressible modalities and structures; 2) progressively searching and retraining the subnet, which maintains convergence between the search and retrain to attain higher compression ratios. Experiments on various tasks, datasets, and model architectures demonstrate the effectiveness and versatility of the proposed UPop framework. The code is available at https://github.com/sdc17/UPop.

</details>

---

## 7. TRAK: Attributing Model Behavior at Scale

- [ ] TRAK: Attributing Model Behavior at Scale | https://icml.cc/virtual/2023/poster/24558

- **Link**: https://icml.cc/virtual/2023/poster/24558

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The goal of data attribution is to trace model predictions back to training data. Despite a long line of work towards this goal, existing approaches to data attribution tend to force users to choose between computational tractability and efficacy. That is, computationally tractable methods can struggle with accurately attributing model predictions in non-convex settings (e.g., in the context of deep neural networks), while methods that are effective in such regimes require training thousands of models, which makes them impractical for large models or datasets. In this work, we introduce TRAK (Tracing with the Randomly-projected After Kernel), a data attribution method that is both effective and computationally tractable for large-scale, differentiable models. In particular, by leveraging only a handful of trained models, TRAK can match the performance of attribution methods that require training thousands of models. We demonstrate the utility of TRAK across various modalities and scales: image classifiers trained on ImageNet, vision-language models (CLIP), and language models (BERT and mT5). We provide code for using TRAK (and reproducing our work) at https://github.com/MadryLab/trak .

</details>

---

## 8. $\pi$-Tuning: Transferring Multimodal Foundation Models with Optimal Multi-task Interpolation

- [ ] $\pi$-Tuning: Transferring Multimodal Foundation Models with Optimal Multi-task Interpolation | https://icml.cc/virtual/2023/poster/24642

- **Link**: https://icml.cc/virtual/2023/poster/24642

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Foundation models have achieved great advances in multi-task learning with a unified interface of unimodal and multimodal tasks. However, the potential of such multi-task learners has not been exploited during transfer learning. In this work, we present a universal parameter-efficient transfer learning method, termed Predict-Interpolate Tuning ($\pi$-Tuning), for vision, language, and vision-language tasks. It aggregates the parameters of lightweight task-specific experts learned from similar tasks to aid the target downstream task. The task similarities are predicted in a unified modality-independent space, yielding a scalable graph to demonstrate task relationships. $\pi$-Tuning has several appealing benefits. First, it flexibly explores both intra- and inter-modal transferability between similar tasks to improve the accuracy and robustness of transfer learning, especially in data-scarce scenarios. Second, it offers a systematical solution for transfer learning with multi-task prediction-and-then-interpolation, compatible with diverse types of parameter-efficient experts, such as prompt and adapter. Third, an extensive study of task-level mutual benefits on 14 unimodal and 6 multimodal datasets shows that $\pi$-Tuning surpasses fine-tuning and other parameter-efficient transfer learning methods both in full-shot and low-shot regimes. The task graph also enables an in-depth interpretable analysis of task transferability across modalities. The code will be available at https://github.com/TencentARC/pi-Tuning.

</details>

---

## 9. ILLUME: Rationalizing Vision-Language Models through Human Interactions

- [ ] ILLUME: Rationalizing Vision-Language Models through Human Interactions | https://icml.cc/virtual/2023/poster/24655

- **Link**: https://icml.cc/virtual/2023/poster/24655

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Bootstrapping from pre-trained language models has been proven to be an efficient approach for building vision-language models (VLM) for tasks such as image captioning or visual question answering. However, outputs of these models rarely align with user's rationales for specific answers. In order to improve this alignment and reinforce commonsense reasons, we propose a tuning paradigm based on human interactions with machine-generated data. Our ILLUME executes the following loop: Given an image-question-answer prompt, the VLM samples multiple candidate rationales, and a human critic provides feedback via preference selection, used for fine-tuning. This loop increases the training data and gradually carves out the VLM's rationalization capabilities that are aligned with human intent. Our exhaustive experiments demonstrate that ILLUME is competitive with standard supervised finetuning while using significantly fewer training data and only requiring minimal feedback.

</details>

---

## 10. Distilling Internet-Scale Vision-Language Models into Embodied Agents

- [ ] Distilling Internet-Scale Vision-Language Models into Embodied Agents | https://icml.cc/virtual/2023/poster/24664

- **Link**: https://icml.cc/virtual/2023/poster/24664

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Instruction-following agents must ground language into their observation and action spaces. Learning to ground language is challenging, typically requiring domain-specific engineering or large quantities of human interaction data. To address this challenge, we propose using pretrained vision-language models (VLMs) to supervise embodied agents. We combine ideas from model distillation and hindsight experience replay (HER), using a VLM to retroactively generate language describing the agent's behavior. Simple prompting allows us to control the supervision signal, teaching an agent to interact with novel objects based on their names (e.g., planes) or their features (e.g., colors) in a 3D rendered environment. Fewshot prompting lets us teach abstract category membership, including pre-existing categories (food vs toys) and ad-hoc ones (arbitrary preferences over objects). Our work outlines a new and effective way to use internet-scale VLMs, repurposing the generic language grounding acquired by such models to teach task-relevant groundings to embodied agents.

</details>

---

## 11. Self-supervised Neural Factor Analysis for Disentangling Utterance-level Speech Representations

- [ ] Self-supervised Neural Factor Analysis for Disentangling Utterance-level Speech Representations | https://icml.cc/virtual/2023/poster/24759

- **Link**: https://icml.cc/virtual/2023/poster/24759

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Self-supervised learning (SSL) speech models such as wav2vec and HuBERT have demonstrated state-of-the-art performance on automatic speech recognition (ASR) and proved to be extremely useful in low label-resource settings. However, the success of SSL models has yet to transfer to utterance-level tasks such as speaker, emotion, and language recognition, which still require supervised fine-tuning of the SSL models to obtain good performance. We argue that the problem is caused by the lack of disentangled representations and an utterance-level learning objective for these tasks. Inspired by how HuBERT uses clustering to discover hidden acoustic units, we formulate a factor analysis (FA) model that uses the discovered hidden acoustic units to align the SSL features. The underlying utterance-level representations are disentangled using probabilistic inference on the aligned features. Furthermore, the variational lower bound derived from the FA model provides an utterance-level objective, allowing error gradients to be backpropagated to the Transformer layers to learn highly discriminative acoustic units. When used in conjunction with HuBERT's masked prediction training, our models outperform the current best model, WavLM, on all utterance-level non-semantic tasks on the SUPERB benchmark with only 20% of labeled data.

</details>

---

## 12. Continual Vision-Language Representation Learning with Off-Diagonal Information

- [ ] Continual Vision-Language Representation Learning with Off-Diagonal Information | https://icml.cc/virtual/2023/poster/25050

- **Link**: https://icml.cc/virtual/2023/poster/25050

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale multi-modal contrastive learning frameworks like CLIP typically require a large amount of image-text samples for training. However, these samples are always collected continuously in real scenarios. This paper discusses the feasibility of continual CLIP training using streaming data. Unlike continual learning based on self-supervised learning methods for pure images, which is empirically robust against catastrophic forgetting, CLIP's performance degeneration in the continual setting is significant and non-neglectable. By analyzing the changes in the model's representation space during continual CLIP training from a spatial geometry perspective, we explore and summarize these spatial variations as Spatial Disorder (SD) , which can be divided into Intra-modal Rotation and Inter-modal Deviation . Moreover, we empirically and theoretically demonstrate how SD leads to a performance decline for CLIP on cross-modal retrieval tasks. To alleviate SD, we propose a new continual vision-language representation learning framework Mod-X : M aintain o ff- d iagonal information-matri X . By selectively aligning the off-diagonal information distribution of contrastive matrices, the Mod-X improves the capability of the multi-modal model by maintaining the multi-modal representation space alignment on the old data domain during continuously fitting the new training data domain. Experiments on commonly used datasets with different scales and scopes have demonstrated the effectiveness of our method.

</details>

---

## 13. RLEG: Vision-Language Representation Learning with Diffusion-based Embedding Generation

- [ ] RLEG: Vision-Language Representation Learning with Diffusion-based Embedding Generation | https://icml.cc/virtual/2023/poster/25118

- **Link**: https://icml.cc/virtual/2023/poster/25118

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language representation learning models (e.g., CLIP) have achieved state-of-the-art performance on various downstream tasks, which usually need large-scale training data to learn discriminative representation. Recent progress on generative diffusion models (e.g., DALL-E 2) has demonstrated that diverse high-quality samples can be synthesized by randomly sampling from generative distribution. By virtue of generative capability in this paper, we propose a novel vision-language Representation Learning method with diffusion-based Embedding Generation (RLEG), which exploits diffusion models to generate feature embedding online for learning effective vision-language representation. Specifically, we first adopt image and text encoders to extract the corresponding embeddings. Secondly, pretrained diffusion-based embedding generators are harnessed to transfer the embedding modality online between vision and language domains. The embeddings generated from the generators are then served as augmented embedding-level samples, which are applied to contrastive learning with the variant of the CLIP framework. Experimental results show that the proposed method could learn effective representation and achieve state-of-the-art performance on various tasks including image classification, image-text retrieval, object detection, semantic segmentation, and text-conditional image generation.

</details>

---

## 14. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

- [ ] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | https://icml.cc/virtual/2023/poster/25182

- **Link**: https://icml.cc/virtual/2023/poster/25182

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes BLIP-2, a generic and efficient pre-training strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pre-trained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various vision-language tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model's emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions.

</details>

---

## 15. Retrieval-Augmented Multimodal Language Modeling

- [ ] Retrieval-Augmented Multimodal Language Modeling | https://icml.cc/virtual/2023/poster/25248

- **Link**: https://icml.cc/virtual/2023/poster/25248

- **Conference**: ICML

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent multimodal models such as DALL-E and CM3 have achieved remarkable progress in text-to-image and image-to-text generation. However, these models store all their knowledge (e.g., the appearance of the Eiffel Tower) in the model parameters, requiring increasingly larger models and training data to capture more knowledge. To integrate knowledge in a more scalable and modular way, we propose a retrieval-augmented multimodal model, which enables a base multimodal model (generator) to refer to relevant text and images fetched by a retriever from external memory (e.g., documents on the web). Specifically, for the retriever, we use a pretrained CLIP, and for the generator, we train a CM3 Transformer on the LAION dataset. Our resulting model, named Retrieval-Augmented CM3 (RA-CM3), is the first multimodal model that can retrieve and generate both text and images. We show that RA-CM3 significantly outperforms baseline multimodal models such as DALL-E and CM3 on both image and caption generation tasks (12 FID and 17 CIDEr improvements on MS-COCO), while requiring much less compute for training (<30% of DALL-E). Moreover, we show that RA-CM3 exhibits novel capabilities such as faithful image generation and multimodal in-context learning (e.g., image generation from demonstrations).

</details>

---

