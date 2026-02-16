# EMNLP 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_emnlp2023_papers.csv

## 1. Violet: A Vision-Language Model forArabic Image Captioning with Gemini Decoder

- [ ] Violet: A Vision-Language Model forArabic Image Captioning with Gemini Decoder | https://aclanthology.org/2023.arabicnlp-1.1/

- **Link**: https://aclanthology.org/2023.arabicnlp-1.1/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Although image captioning has a vast array of applications, it has not reached its full potential in languages other than English. Arabic, for instance, although the native language of more than 400 million people, remains largely underrepresented in this area. This is due to the lack of labeled data and powerful Arabic generative models. We alleviate this issue by presenting a novel vision-language model dedicated to Arabic, dubbed Violet. Our model is based on a vision encoder and a Gemini text decoder that maintains generation fluency while allowing fusion between the vision and language components. To train our model, we introduce a new method for automatically acquiring data from available English datasets. We also manually prepare a new dataset for evaluation. Violet performs sizeably better than our baselines on all of our evaluation datasets. For example, it reaches a CIDEr score of 61.2 on our manually annotated dataset and achieves an improvement of 13 points on Flickr8k.

</details>

---

## 2. ArchBERT: Bi-Modal Understanding of Neural Architectures and Natural Languages

- [ ] ArchBERT: Bi-Modal Understanding of Neural Architectures and Natural Languages | https://aclanthology.org/2023.conll-1.7/

- **Link**: https://aclanthology.org/2023.conll-1.7/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Building multi-modal language models has been a trend in the recent years, where additional modalities such as image, video, speech, etc. are jointly learned along with natural languages (i.e., textual information). Despite the success of these multi-modal language models with different modalities, there is no existing solution for neural network architectures and natural languages. Providing neural architectural information as a new modality allows us to provide fast architecture-2-text and text-2-architecture retrieval/generation services on the cloud with a single inference. Such solution is valuable in terms of helping beginner and intermediate ML users to come up with better neural architectures or AutoML approaches with a simple text query. In this paper, we propose ArchBERT, a bi-modal model for joint learning and understanding of neural architectures and natural languages, which opens up new avenues for research in this area. We also introduce a pre-training strategy named Masked Architecture Modeling (MAM) for a more generalized joint learning. Moreover, we introduce and publicly release two new bi-modal datasets for training and validating our methods. The ArchBERT’s performance is verified through a set of numerical experiments on different downstream tasks such as architecture-oriented reasoning, question answering, and captioning (summarization). Datasets, codes, and demos are available as supplementary materials.

</details>

---

## 3. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding

- [ ] Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding | https://aclanthology.org/2023.emnlp-demo.49/

- **Link**: https://aclanthology.org/2023.emnlp-demo.49/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present Video-LLaMA, a multi-modal framework that empowers Large Language Models (LLMs) with the capability of understanding both visual and auditory content in the video. Video-LLaMA bootstraps cross-modal training from the frozen pre-trained visual & audio encoders and the frozen LLMs. Unlike previous works that complement LLMs to process the visual or audio signals only, Video-LLaMA enables video comprehension by tackling two challenges: (1) capturing the temporal changes in visual scenes, (2) integrating audio-visual signals. To counter the first challenge, we propose a Video Q-former to assemble a pre-trained image encoder into our video encoder and introduce a video-to-text generation task to learn video-language correspondence. For the second challenge, we leverage ImageBind, a universal embedding model aligning multiple modalities, as the pre-trained audio encoder and introduce an Audio Q-former on top of ImageBind to learn reasonable auditory query embeddings for the LLM module. To align the output of both visual & audio encoders with LLM’s embedding space, we first train Video-LLaMA on massive video/image-caption pairs and then tune our model with visual-instruction datasets of moderate amount but higher quality. We found Video-LLaMA shows the ability to perceive and comprehend video content and generate meaningful responses grounded in the visual and auditory information presented in the videos.

</details>

---

## 4. Query-aware Multi-modal based Ranking Relevance in Video Search

- [ ] Query-aware Multi-modal based Ranking Relevance in Video Search | https://aclanthology.org/2023.emnlp-industry.31/

- **Link**: https://aclanthology.org/2023.emnlp-industry.31/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Relevance ranking system plays a crucial role in video search on streaming platforms. Most relevance ranking methods focus on text modality, incapable of fully exploiting cross-modal cues present in video. Recent multi-modal models have demonstrated promise in various vision-language tasks but provide limited help for downstream query-video relevance tasks due to the discrepency between relevance ranking-agnostic pre-training objectives and the real video search scenarios that demand comprehensive relevance modeling. To address these challenges, we propose a QUery-Aware pre-training model with multi-modaLITY (QUALITY) that incorporates hard-mined query information as alignment targets and utilizes video tag information for guidance. QUALITY is integrated into our relevance ranking model, which leverages multi-modal knowledge and improves ranking optimization method based on ordinal regression. Extensive experiments show our proposed model significantly enhances video search performance.

</details>

---

## 5. A Suite of Generative Tasks for Multi-Level Multimodal Webpage Understanding

- [ ] A Suite of Generative Tasks for Multi-Level Multimodal Webpage Understanding | https://aclanthology.org/2023.emnlp-main.119/

- **Link**: https://aclanthology.org/2023.emnlp-main.119/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Webpages have been a rich, scalable resource for vision-language and language only tasks. Yet only pieces of webpages are kept in existing datasets: image-caption pairs, long text articles, or raw HTML, never all in one place. Webpage tasks have resultingly received little attention and structured image-text data left underused. To study multimodal webpage understanding, we introduce the Wikipedia Webpage suite (WikiWeb2M) containing 2M pages with all of the associated image, text, and structure data. We verify its utility on three generative tasks: page description generation, section summarization, and contextual image captioning. We design a novel attention mechanism Prefix Global, which selects the most relevant image and text content as global tokens to attend to the rest of the webpage for context. By using page structure to separate such tokens, it performs better than full attention with lower computational complexity. Extensive experiments show that the new data in WikiWeb2M improves task performance compared to prior work.

</details>

---

## 6. Learning the Visualness of Text Using Large Vision-Language Models

- [ ] Learning the Visualness of Text Using Large Vision-Language Models | https://aclanthology.org/2023.emnlp-main.147/

- **Link**: https://aclanthology.org/2023.emnlp-main.147/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual text evokes an image in a person’s mind, while non-visual text fails to do so. A method to automatically detect visualness in text will enable text-to-image retrieval and generation models to augment text with relevant images. This is particularly challenging with long-form text as text-to-image generation and retrieval models are often triggered for text that is designed to be explicitly visual in nature, whereas long-form text could contain many non-visual sentences. To this end, we curate a dataset of 3,620 English sentences and their visualness scores provided by multiple human annotators. We also propose a fine-tuning strategy that adapts large vision-language models like CLIP by modifying the model’s contrastive learning objective to map text identified as non-visual to a common NULL image while matching visual text to their corresponding images in the document. We evaluate the proposed approach on its ability to (i) classify visual and non-visual text accurately, and (ii) attend over words that are identified as visual in psycholinguistic studies. Empirical evaluation indicates that our approach performs better than several heuristics and baseline models for the proposed task. Furthermore, to highlight the importance of modeling the visualness of text, we conduct qualitative analyses of text-to-image generation systems like DALL-E.

</details>

---

## 7. Let’s Think Frame by Frame withVIP: A Video Infilling and Prediction Dataset for Evaluating Video Chain-of-Thought

- [ ] Let’s Think Frame by Frame withVIP: A Video Infilling and Prediction Dataset for Evaluating Video Chain-of-Thought | https://aclanthology.org/2023.emnlp-main.15/

- **Link**: https://aclanthology.org/2023.emnlp-main.15/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite exciting recent results showing vision-language systems’ capacity to reason about images using natural language, their capacity for video reasoning remains underexplored. We motivate framing video reasoning as the sequential understanding of a small number of keyframes, thereby leveraging the power and robustness of vision-language while alleviating the computational complexities of processing videos. To evaluate this novel application, we introduce VIP, an inference-time challenge dataset designed to explore models’ reasoning capabilities through video chain-of-thought. Inspired by visually descriptive scene plays, we propose two formats for keyframe description: unstructured dense captions and structured scene descriptions that identify the focus, action, mood, objects, and setting (FAMOuS) of the keyframe. To evaluate video reasoning, we propose two tasks: Video Infilling and Video Prediction, which test abilities to generate multiple intermediate keyframes and predict future keyframes, respectively. We benchmark GPT-4, GPT-3, and VICUNA on VIP, demonstrate the performance gap in these complex video reasoning tasks, and encourage future work to prioritize language models for efficient and generalized video reasoning.

</details>

---

## 8. A Framework for Vision-Language Warm-up Tasks in Multimodal Dialogue Models

- [ ] A Framework for Vision-Language Warm-up Tasks in Multimodal Dialogue Models | https://aclanthology.org/2023.emnlp-main.167/

- **Link**: https://aclanthology.org/2023.emnlp-main.167/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Most research on multimodal open-domain dialogue agents has focused on pretraining and multi-task learning using additional rich datasets beyond a given target dataset. However, methods for exploiting these additional datasets can be quite limited in real-world settings, creating a need for more efficient methods for constructing agents based solely on the target dataset. To address these issues, we present a new learning strategy called vision-language warm-up tasks for multimodal dialogue models (VLAW-MDM). This strategy does not require the use of large pretraining or multi-task datasets but rather relies solely on learning from target data. Moreover, our proposed approach automatically generate captions for images and incorporate them into the model’s input to improve the contextualization of visual information. Using this novel approach, we empirically demonstrate that our learning strategy is effective for limited data and relatively small models. The result show that our method achieved comparable and in some cases superior performance compared to existing state-of-the-art models on various evaluation metrics.

</details>

---

## 9. WhyLLMs Hallucinate, and How to Get (Evidential) Closure: Perceptual, Intensional, and Extensional Learning for Faithful Natural Language Generation

- [ ] WhyLLMs Hallucinate, and How to Get (Evidential) Closure: Perceptual, Intensional, and Extensional Learning for Faithful Natural Language Generation | https://aclanthology.org/2023.emnlp-main.192/

- **Link**: https://aclanthology.org/2023.emnlp-main.192/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We show that LLMs hallucinate because their output is not constrained to be synonymous with claims for which they have evidence: a condition that we call evidential closure. Information about the truth or falsity of sentences is not statistically identified in the standard neural language generation setup, and so cannot be conditioned on to generate new strings. We then show how to constrain LLMs to produce output that satisfies evidential closure. A multimodal LLM must learn about the external world (perceptual learning); it must learn a mapping from strings to states of the world (extensional learning); and, to achieve fluency when generalizing beyond a body of evidence, it must learn mappings from strings to their synonyms (intensional learning). The output of a unimodal LLM must be synonymous with strings in a validated evidence set. Finally, we present a heuristic procedure, Learn-Babble-Prune, that yields faithful output from an LLM by rejecting output that is not synonymous with claims for which the LLM has evidence.

</details>

---

## 10. Evaluating Object Hallucination in Large Vision-Language Models

- [ ] Evaluating Object Hallucination in Large Vision-Language Models | https://aclanthology.org/2023.emnlp-main.20/

- **Link**: https://aclanthology.org/2023.emnlp-main.20/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Inspired by the superior language abilities of large language models (LLM), large vision-language models (LVLM) have been recently proposed by integrating powerful LLMs for improving the performance on complex multimodal tasks. Despite the promising progress on LVLMs, we find that they suffer from object hallucinations, i.e., they tend to generate objects inconsistent with the target images in the descriptions. To investigate it, this work presents the first systematic study on object hallucination of LVLMs. We conduct the evaluation experiments on several representative LVLMs, and show that they mostly suffer from severe object hallucination issues. We further discuss that the visual instructions may influence the hallucination, and find that: objects that frequently appear in the visual instructions or co-occur with the image objects are obviously prone to be hallucinated by LVLMs. Besides, we further design a polling-based query method called POPE for better evaluation of object hallucination. Experiment results show that our POPE can evaluate object hallucination in a more stable and flexible way.

</details>

---

## 11. Text encoders bottleneck compositionality in contrastive vision-language models

- [ ] Text encoders bottleneck compositionality in contrastive vision-language models | https://aclanthology.org/2023.emnlp-main.301/

- **Link**: https://aclanthology.org/2023.emnlp-main.301/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Performant vision-language (VL) models like CLIP represent captions using a single vector. How much information about language is lost in this bottleneck? We first curate CompPrompts, a set of increasingly compositional image captions that VL models should be able to capture (e.g., single object, to object+property, to multiple interacting objects). Then, we train text-only recovery probes that aim to reconstruct captions from single-vector text representations produced by several VL models. This approach does not require images, allowing us to test on a broader range of scenes compared to prior work. We find that: 1) CLIP’s text encoder falls short on more compositional inputs, including object relationships, attribute-object association, counting, and negations; 2) some text encoders work significantly better than others; and 3) text-only recovery performance predicts multimodal matching performance on ControlledImCaps: a new evaluation benchmark we collect and release consisting of fine-grained compositional images and captions. Specifically, our results suggest text-only recoverability is a necessary (but not sufficient) condition for modeling compositional factors in contrastive VL models. We release our datasets and code.

</details>

---

## 12. Compressing and Debiasing Vision-Language Pre-Trained Models for Visual Question Answering

- [ ] Compressing and Debiasing Vision-Language Pre-Trained Models for Visual Question Answering | https://aclanthology.org/2023.emnlp-main.34/

- **Link**: https://aclanthology.org/2023.emnlp-main.34/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite the excellent performance of vision-language pre-trained models (VLPs) on conventional VQA task, they still suffer from two problems: First, VLPs tend to rely on language biases in datasets and fail to generalize to out-of-distribution (OOD) data. Second, they are inefficient in terms of memory footprint and computation. Although promising progress has been made in both problems, most existing works tackle them independently. To facilitate the application of VLP to VQA tasks, it is imperative to jointly study VLP compression and OOD robustness, which, however, has not yet been explored. This paper investigates whether a VLP can be compressed and debiased simultaneously by searching sparse and robust subnetworks. To this end, we systematically study the design of a training and compression pipeline to search the subnetworks, as well as the assignment of sparsity to different modality-specific modules. Our experiments involve 2 VLPs, 2 compression methods, 4 training methods, 2 datasets and a range of sparsity levels. Our results show that there indeed exist sparse and robust subnetworks, which are competitive with the debiased full VLP and clearly outperform the debiasing SoTAs with fewer parameters on OOD datasets VQA-CP v2 and VQA-VS. The codes can be found at https://github.com/PhoebusSi/Compress-Robust-VQA.

</details>

---

## 13. Grounding Visual Illusions in Language: Do Vision-Language Models Perceive Illusions Like Humans?

- [ ] Grounding Visual Illusions in Language: Do Vision-Language Models Perceive Illusions Like Humans? | https://aclanthology.org/2023.emnlp-main.348/

- **Link**: https://aclanthology.org/2023.emnlp-main.348/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) are trained on vast amounts of data captured by humans emulating our understanding of the world. However, known as visual illusions, human’s perception of reality isn’t always faithful to the physical world. This raises a key question: do VLMs have the similar kind of illusions as humans do, or do they faithfully learn to represent reality? To investigate this question, we build a dataset containing five types of visual illusions and formulate four tasks to examine visual illusions in state-of-the-art VLMs. Our findings have shown that although the overall alignment is low, larger models are closer to human perception and more susceptible to visual illusions. Our dataset and initial findings will promote a better understanding of visual illusions in humans and machines and provide a stepping stone for future computational models that can better align humans and machines in perceiving and communicating about the shared visual world. The code and data are available at [github.com/vl-illusion/dataset](https://github.com/vl-illusion/dataset).

</details>

---

## 14. VLIS: Unimodal Language Models Guide Multimodal Language Generation

- [ ] VLIS: Unimodal Language Models Guide Multimodal Language Generation | https://aclanthology.org/2023.emnlp-main.46/

- **Link**: https://aclanthology.org/2023.emnlp-main.46/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multimodal language generation, which leverages the synergy of language and vision, is a rapidly expanding field. However, existing vision-language models face challenges in tasks that require complex linguistic understanding. To address this issue, we introduce Visual-Language models as Importance Sampling weights (VLIS), a novel framework that combines the visual conditioning capability of vision-language models with the language understanding of unimodal text-only language models without further training. It extracts pointwise mutual information of each image and text from a visual-language model and uses the value as an importance sampling weight to adjust the token likelihood from a text-only model. VLIS improves vision-language models on diverse tasks, including commonsense understanding (WHOOPS, OK-VQA, and ScienceQA) and complex text generation (Concadia, Image Paragraph Captioning, and ROCStories). Our results suggest that VLIS represents a promising new direction for multimodal language generation.

</details>

---

## 15. Coarse-to-Fine Contrastive Learning in Image-Text-Graph Space for Improved Vision-Language Compositionality

- [ ] Coarse-to-Fine Contrastive Learning in Image-Text-Graph Space for Improved Vision-Language Compositionality | https://aclanthology.org/2023.emnlp-main.56/

- **Link**: https://aclanthology.org/2023.emnlp-main.56/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastively trained vision-language models have achieved remarkable progress in vision and language representation learning. However, recent research has highlighted severe limitations of these models in their ability to perform compositional reasoning over objects, attributes, and relations. Scene graphs have emerged as an effective way to understand images compositionally. These are graph-structured semantic representations of images that contain objects, their attributes, and relations with other objects in a scene. In this work, we consider the scene graph parsed from text as a proxy for the image scene graph and propose a graph decomposition and augmentation framework along with a coarse-to-fine contrastive learning objective between images and text that aligns sentences of various complexities to the same image. We also introduce novel negative mining techniques in the scene graph space for improving attribute binding and relation understanding. Through extensive experiments, we demonstrate the effectiveness of our approach that significantly improves attribute binding, relation understanding, systematic generalization, and productivity on multiple recently proposed benchmarks (For example, improvements up to18% for systematic generalization,16.5% for relation understanding over a strong baseline), while achieving similar or better performance than CLIP on various general multimodal tasks.

</details>

---

## 16. What’s “up” with vision-language models? Investigating their struggle with spatial reasoning

- [ ] What’s “up” with vision-language models? Investigating their struggle with spatial reasoning | https://aclanthology.org/2023.emnlp-main.568/

- **Link**: https://aclanthology.org/2023.emnlp-main.568/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language (VL) models are powerful, but can they reliably distinguish “right” from “left”? We curate three new corpora to quantify model comprehension of such basic spatial relations. These tests isolate spatial reasoning more precisely than existing datasets like VQAv2, e.g., our What’sUp benchmark contains sets of photographs varying only the spatial relations of objects, keeping their identity fixed (see Figure 1: models must comprehend not only the usual case of a dog under a table, but also, the same dog on top of the same table). We evaluate 18 VL models, finding that all perform poorly, e.g., BLIP finetuned on VQAv2, which nears human parity on VQAv2, achieves 56% accuracy on our benchmarks vs. humans at 99%. We conclude by studying causes of this surprising behavior, finding: 1) that popular vision-language pretraining corpora like LAION-2B contain little reliable data for learning spatial relationships; and 2) that basic modeling interventions like up-weighting preposition-containing instances or fine-tuning on our corpora are not sufficient to address the challenges our benchmarks pose. We are hopeful that these corpora will facilitate further research, and we release our data and code at https://github.com/amitakamath/whatsup_vlms.

</details>

---

## 17. Describe Me an Auklet: Generating Grounded Perceptual Category Descriptions

- [ ] Describe Me an Auklet: Generating Grounded Perceptual Category Descriptions | https://aclanthology.org/2023.emnlp-main.580/

- **Link**: https://aclanthology.org/2023.emnlp-main.580/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Human speakers can generate descriptions of perceptual concepts, abstracted from the instance-level. Moreover, such descriptions can be used by other speakers to learn provisional representations of those concepts. Learning and using abstract perceptual concepts is under-investigated in the language-and-vision field. The problem is also highly relevant to the field of representation learning in multi-modal NLP. In this paper, we introduce a framework for testing category-level perceptual grounding in multi-modal language models. In particular, we train separate neural networks to **generate** and **interpret** descriptions of visual categories. We measure the *communicative success* of the two models with the zero-shot classification performance of the interpretation model, which we argue is an indicator of perceptual grounding. Using this framework, we compare the performance of *prototype*- and *exemplar*-based representations. Finally, we show that communicative success exposes performance issues in the generation model, not captured by traditional intrinsic NLG evaluation metrics, and argue that these issues stem from a failure to properly ground language in vision at the category level.

</details>

---

## 18. On Bilingual Lexicon Induction with Large Language Models

- [ ] On Bilingual Lexicon Induction with Large Language Models | https://aclanthology.org/2023.emnlp-main.595/

- **Link**: https://aclanthology.org/2023.emnlp-main.595/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Bilingual Lexicon Induction (BLI) is a core task in multilingual NLP that still, to a large extent, relies on calculating cross-lingual word representations. Inspired by the global paradigm shift in NLP towards Large Language Models (LLMs), we examine the potential of the latest generation of LLMs for the development of bilingual lexicons. We ask the following research question: Is it possible to prompt and fine-tune multilingual LLMs (mLLMs) for BLI, and how does this approach compare against and complement current BLI approaches? To this end, we systematically study 1) zero-shot prompting for unsupervised BLI and 2) few-shot in-context prompting with a set of seed translation pairs, both without any LLM fine-tuning, as well as 3) standard BLI-oriented fine-tuning of smaller LLMs. We experiment with 18 open-source text-to-text mLLMs of different sizes (from 0.3B to 13B parameters) on two standard BLI benchmarks covering a range of typologically diverse languages. Our work is the first to demonstrate strong BLI capabilities of text-to-text mLLMs. The results reveal that few-shot prompting with in-context examples from nearest neighbours achieves the best performance, establishing new state-of-the-art BLI scores for many language pairs. We also conduct a series of in-depth analyses and ablation studies, providing more insights on BLI with (m)LLMs, also along with their limitations.

</details>

---

## 19. Prompting Scientific Names for Zero-Shot Species Recognition

- [ ] Prompting Scientific Names for Zero-Shot Species Recognition | https://aclanthology.org/2023.emnlp-main.610/

- **Link**: https://aclanthology.org/2023.emnlp-main.610/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Trained on web-scale image-text pairs, Vision-Language Models (VLMs) such as CLIP can recognize images of common objects in a zero-shot fashion. However, it is underexplored how to use CLIP for zero-shot recognition of highly specialized concepts, e.g., species of birds, plants, and animals, for which their scientific names are written in Latin or Greek. Indeed, CLIP performs poorly for zero-shot species recognition with prompts that use scientific names, e.g., “a photo of Lepus Timidus” (which is a scientific name in Latin). This is because these names are usually not included in CLIP’s training set. To improve performance, we explore using large-language models (LLMs) to generate descriptions (e.g., of species color and shape) and additionally use them in prompts. However, this method improves only marginally. Instead, we are motivated to translate scientific names (e.g., Lepus Timidus) to common English names (e.g., mountain hare) and use such in the prompts. We find that common names are more likely to be included in CLIP’s training set, and prompting them achieves 2~5 times higher accuracy on benchmarking datasets of fine-grained species recognition.

</details>

---

## 20. APoLLo : Unified Adapter and Prompt Learning for Vision Language Models

- [ ] APoLLo : Unified Adapter and Prompt Learning for Vision Language Models | https://aclanthology.org/2023.emnlp-main.629/

- **Link**: https://aclanthology.org/2023.emnlp-main.629/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The choice of input text prompt plays a critical role in the performance of Vision-Language Pretrained (VLP) models such as CLIP. We present APoLLo, a unified multi-modal approach that combines Adapter and Prompt learning for Vision-Language models. Our method is designed to substantially improve the generalization capabilities of VLP models when they are fine-tuned in a few-shot setting. We introduce trainable cross-attention-based adapter layers in conjunction with vision and language encoders to strengthen the alignment between the two modalities. We enforce consistency between the respective encoder branches (receiving augmented inputs) to prevent overfitting in downstream tasks. Our method is evaluated on three representative tasks: generalization to novel classes, cross-dataset evaluation, and unseen domain shifts. In practice, APoLLo achieves a relative gain up to 6.03% over MaPLe (SOTA) on novel classes for 10 diverse image recognition datasets.

</details>

---

## 21. Bridging the Digital Divide: Performance Variation across Socio-Economic Factors in Vision-Language Models

- [ ] Bridging the Digital Divide: Performance Variation across Socio-Economic Factors in Vision-Language Models | https://aclanthology.org/2023.emnlp-main.660/

- **Link**: https://aclanthology.org/2023.emnlp-main.660/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite the impressive performance of current AI models reported across various tasks, performance reports often do not include evaluations of how these models perform on the specific groups that will be impacted by these technologies. Among the minority groups under-represented in AI, data from low-income households are often overlooked in data collection and model evaluation. We evaluate the performance of a state-of-the-art vision-language model (CLIP) on a geo-diverse dataset containing household images associated with different income values (DollarStreet) and show that performance inequality exists among households of different income levels. Our results indicate that performance for the poorer groups is consistently lower than the wealthier groups across various topics and countries. We highlight insights that can help mitigate these issues and propose actionable steps for economic-level inclusive AI development.

</details>

---

## 22. Can Language Models Understand Physical Concepts?

- [ ] Can Language Models Understand Physical Concepts? | https://aclanthology.org/2023.emnlp-main.726/

- **Link**: https://aclanthology.org/2023.emnlp-main.726/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Language models (LMs) gradually become general-purpose interfaces in the interactive and embodied world, where the understanding of physical concepts is an essential prerequisite. However, it is unclear whether LMs can understand physical concepts in the human world. To investigate this, we design a benchmark VEC that covers the tasks of (i) Visual concepts, such as the shape and material of objects, and (ii) Embodied Concepts, learned from the interaction with the world such as the temperature of objects. Our zero (few)-shot prompting results show that the understanding of certain visual concepts emerges as scaling up LMs, but there are still basic concepts to which the scaling law does not apply. For example, OPT-175B performs close to humans with a zero-shot accuracy of 85% on the material concept, yet behaves like random guessing on the mass concept. Instead, vision-augmented LMs such as CLIP and BLIP achieve a human-level understanding of embodied concepts. Analysis indicates that the rich semantics in visual representation can serve as a valuable source of embodied knowledge. Inspired by this, we propose a distillation method to transfer embodied knowledge from VLMs to LMs, achieving performance gain comparable with that by scaling up parameters of LMs134×. Our dataset is available at https://github.com/TobiasLee/VEC.

</details>

---

## 23. Enhancing Textbooks with Visuals from the Web for Improved Learning

- [ ] Enhancing Textbooks with Visuals from the Web for Improved Learning | https://aclanthology.org/2023.emnlp-main.731/

- **Link**: https://aclanthology.org/2023.emnlp-main.731/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Textbooks are one of the main mediums for delivering high-quality education to students. In particular, explanatory and illustrative visuals play a key role in retention, comprehension and general transfer of knowledge. However, many textbooks lack these interesting visuals to support student learning. In this paper, we investigate the effectiveness of vision-language models to automatically enhance textbooks with images from the web. We collect a dataset of e-textbooks in the math, science, social science and business domains. We then set up a text-image matching task that involves retrieving and appropriately assigning web images to textbooks, which we frame as a matching optimization problem. Through a crowd-sourced evaluation, we verify that (1) while the original textbook images are rated higher, automatically assigned ones are not far behind, and (2) the precise formulation of the optimization problem matters. We release the dataset of textbooks with an associated image bank to inspire further research in this intersectional area of computer vision and NLP for education.

</details>

---

## 24. From Wrong To Right: A Recursive Approach Towards Vision-Language Explanation

- [ ] From Wrong To Right: A Recursive Approach Towards Vision-Language Explanation | https://aclanthology.org/2023.emnlp-main.75/

- **Link**: https://aclanthology.org/2023.emnlp-main.75/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Addressing the challenge of adapting pre-trained vision-language models for generating insightful explanations for visual reasoning tasks with limited annotations, we present ReVisE: a Recursive Visual Explanation algorithm. Our method iteratively computes visual features (conditioned on the text input), an answer, and an explanation, to improve the explanation quality step by step until the answer converges. We find that this multi-step approach guides the model to correct its own answers and outperforms single-step explanation generation. Furthermore, explanations generated by ReVisE also serve as valuable annotations for few-shot self-training. Our approach outperforms previous methods while utilizing merely 5% of the human-annotated explanations across 10 metrics, demonstrating up to a 4.2 and 1.3 increase in BLEU-1 score on the VCR and VQA-X datasets, underscoring the efficacy and data-efficiency of our method.

</details>

---

## 25. CLEVR-Implicit: A Diagnostic Dataset for Implicit Reasoning in Referring Expression Comprehension

- [ ] CLEVR-Implicit: A Diagnostic Dataset for Implicit Reasoning in Referring Expression Comprehension | https://aclanthology.org/2023.emnlp-main.791/

- **Link**: https://aclanthology.org/2023.emnlp-main.791/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recently, pre-trained vision-language (VL) models have achieved remarkable success in various cross-modal tasks, including referring expression comprehension (REC). These models are pre-trained on the large-scale image-text pairs to learn the alignment between words in textual descriptions and objects in the corresponding images and then fine-tuned on downstream tasks. However, the performance of VL models is hindered when dealing with implicit text, which describes objects through comparisons between two or more objects rather than explicitly mentioning them. This is because the models struggle to align the implicit text with the objects in the images. To address the challenge, we introduce CLEVR-Implicit, a dataset consisting of synthetic images and corresponding two types of implicit text for the REC task. Additionally, to enhance the performance of VL models on implicit text, we propose a method called Transforming Implicit text into Explicit text (TIE), which enables VL models to reason with the implicit text. TIE consists of two modules: (1) the prompt design module builds prompts for implicit text by adding masked tokens, and (2) the cloze procedure module fine-tunes the prompts by utilizing masked language modeling (MLM) to predict the explicit words with the implicit prompts. Experimental results on our dataset demonstrate a significant improvement of 37.94% in the performance of VL models on implicit text after employing our TIE method.

</details>

---

## 26. ViStruct: Visual Structural Knowledge Extraction via Curriculum Guided Code-Vision Representation

- [ ] ViStruct: Visual Structural Knowledge Extraction via Curriculum Guided Code-Vision Representation | https://aclanthology.org/2023.emnlp-main.824/

- **Link**: https://aclanthology.org/2023.emnlp-main.824/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

State-of-the-art vision-language models (VLMs) still have limited performance in structural knowledge extraction, such as relations between objects. In this work, we present ViStruct, a training framework to learn VLMs for effective visual structural knowledge extraction. Two novel designs are incorporated. First, we propose to leverage the inherent structure of programming language to depict visual structural information. This approach enables explicit and consistent representation of visual structural information of multiple granularities, such as concepts, relations, and events, in a well-organized structured format. Second, we introduce curriculum-based learning for VLMs to progressively comprehend visual structures, from fundamental visual concepts to intricate event structures. Our intuition is that lower-level knowledge may contribute to complex visual structure understanding. Furthermore, we compile and release a collection of datasets tailored for visual structural knowledge extraction. We adopt a weakly-supervised approach to directly generate visual event structures from captions for ViStruct training, capitalizing on abundant image-caption pairs from the web. In experiments, we evaluate ViStruct on visual structure prediction tasks, demonstrating its effectiveness in improving the understanding of visual structures. The code will be made public to facilitate future research.

</details>

---

## 27. Can We Edit Multimodal Large Language Models?

- [ ] Can We Edit Multimodal Large Language Models? | https://aclanthology.org/2023.emnlp-main.856/

- **Link**: https://aclanthology.org/2023.emnlp-main.856/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we focus on editing multimodal Large Language Models (LLMs). Compared to editing single-modal LLMs, multimodal model editing is more challenging, which demands a higher level of scrutiny and careful consideration in the editing process. To facilitate research in this area, we construct a new benchmark, dubbed MMEdit, for editing multimodal LLMs and establishing a suite of innovative metrics for evaluation. We conduct comprehensive experiments involving various model editing baselines and analyze the impact of editing different components for multimodal LLMs. Empirically, we notice that previous baselines can implement editing multimodal LLMs to some extent, but the effect is still barely satisfactory, indicating the potential difficulty of this task. We hope that our work can provide the NLP community with insights.

</details>

---

## 28. Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs

- [ ] Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs | https://aclanthology.org/2023.emnlp-main.870/

- **Link**: https://aclanthology.org/2023.emnlp-main.870/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision and language models (VLMs) have demonstrated remarkable zero-shot (ZS) performance in a variety of tasks. However, recent works have shown that even the best VLMs struggle to capture aspects of compositional scene understanding, such as object attributes, relations, and action states. In contrast, obtaining structured annotations, such as scene graphs (SGs), that could improve these models is time-consuming and costly, and thus cannot be used on a large scale. Here we ask whether small SG datasets can provide sufficient information for enhancing structured understanding of pretrained VLMs. We show that it is indeed possible to improve VLMs when learning from SGs by integrating components that incorporate structured information into both visual and textual representations. For the visual side, we incorporate a special “SG Component” in the image transformer trained to predict SG information, while for the textual side, we utilize SGs to generate fine-grained captions that highlight different compositional aspects of the scene. Our method improves the performance of several popular VLMs on multiple VL datasets with only a mild degradation in ZS capabilities.

</details>

---

## 29. When are Lemons Purple? The Concept Association Bias of Vision-Language Models

- [ ] When are Lemons Purple? The Concept Association Bias of Vision-Language Models | https://aclanthology.org/2023.emnlp-main.886/

- **Link**: https://aclanthology.org/2023.emnlp-main.886/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models such as CLIP have shown impressive performance on zero-shot image classification and image-to-text retrieval. However, such performance does not realize in tasks that require a finer-grained correspondence between vision and language, such as Visual Question Answering (VQA). We investigate why this is the case, and report an interesting phenomenon of vision-language models, which we call the Concept Association Bias (CAB), as a potential cause of the difficulty of applying these models to VQA and similar tasks. We find that models with CAB tend to treat input as a bag of concepts and attempt to fill in the other missing concept crossmodally, leading to an unexpected zero-shot prediction. We demonstrate CAB by showing that CLIP’s zero-shot classification performance greatly suffers when there is a strong concept association between an object (e.g. eggplant) and an attribute (e.g. color purple). We also show that the strength of CAB predicts the performance on VQA. We observe that CAB is prevalent in vision-language models trained with contrastive losses, even when autoregressive losses are jointly employed. However, a model that solely relies on autoregressive loss seems to exhibit minimal or no signs of CAB.

</details>

---

## 30. UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning

- [ ] UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning | https://aclanthology.org/2023.emnlp-main.906/

- **Link**: https://aclanthology.org/2023.emnlp-main.906/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Charts are widely used for data analysis, providing visual representations and insights into complex data. To facilitate chart-based data analysis using natural language, several downstream tasks have been introduced recently such as chart question answering and chart summarization. However, existing methods for these tasks often rely on pretraining on language or vision-language tasks, neglecting the explicit modeling of chart structures (e.g., how chart elements are related to each other). To address this, we first build a large corpus of charts covering diverse topics and visual styles. We then present UniChart, a pretrained model for chart comprehension and reasoning. UniChart encodes the relevant text, data, and visual elements of charts and then uses a chart-grounded text decoder for text generation. We propose several chart-specific pretraining tasks that include: (i) low-level tasks to extract the visual elements (e.g., bars, lines) and data from charts, and (ii) high-level tasks to acquire chart understanding and reasoning skills. Our experiments demonstrate that pretraining UniChart on a large corpus with chart-specific objectives, followed by fine-tuning, yields state-of-the-art performance on four downstream tasks. Moreover, our model exhibits superior generalizability to unseen chart corpus, surpassing previous approaches that lack chart-specific objectives and utilize limited chart resources.

</details>

---

## 31. R2H: Building Multimodal Navigation Helpers that Respond to Help Requests

- [ ] R2H: Building Multimodal Navigation Helpers that Respond to Help Requests | https://aclanthology.org/2023.emnlp-main.915/

- **Link**: https://aclanthology.org/2023.emnlp-main.915/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Intelligent navigation-helper agents are critical as they can navigate users in unknown areas through environmental awareness and conversational ability, serving as potential accessibility tools for individuals with disabilities. In this work, we first introduce a novel benchmark, Respond to Help Requests (R2H), to promote the development of multi-modal navigation helpers capable of responding to requests for help, utilizing existing dialog-based embodied datasets. R2H mainly includes two tasks: (1) Respond to Dialog History (RDH), which assesses the helper agent’s ability to generate informative responses based on a given dialog history, and (2) Respond during Interaction (RdI), which evaluates the effectiveness and efficiency of the response during consistent cooperation with a task performer. Furthermore, we explore two approaches to construct the navigation-helper agent, including fine-tuning a novel task-oriented multi-modal response generation model that can see and respond, named SeeRee, and employing a multi-modal large language model in a zero-shot manner. Analysis of the task and method was conducted based on both automatic benchmarking and human evaluations.

</details>

---

## 32. Fine-grained Medical Vision-Language Representation Learning for Radiology Report Generation

- [ ] Fine-grained Medical Vision-Language Representation Learning for Radiology Report Generation | https://aclanthology.org/2023.emnlp-main.989/

- **Link**: https://aclanthology.org/2023.emnlp-main.989/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Given the input radiology images, the objective of radiology report generation is to produce accurate and comprehensive medical reports, which typically include multiple descriptive clinical sentences associated with different phenotypes. Most existing works have relied on a pre-trained vision encoder to extract the visual representations of the images. In this study, we propose a phenotype-driven medical vision-language representation learning framework to efficiently bridge the gap between visual and textual modalities for improved text-oriented generation. In contrast to conventional methods which learn medical vision-language representations by contrasting images with entire reports, our approach learns more fine-grained representations by contrasting images with each sentence within the reports. The learned fine-grained representations can be used to improve radiology report generation. The experiments on two widely-used datasets MIMIC-CXR and IU X-ray demonstrate that our method can achieve promising performances and substantially outperform the conventional vision-language representation learning methods.

</details>

---

## 33. SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities

- [ ] SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities | https://aclanthology.org/2023.findings-emnlp.1055/

- **Link**: https://aclanthology.org/2023.findings-emnlp.1055/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models are regarded as a crucial step towards Artificial General Intelligence (AGI) and have garnered significant interest with the emergence of ChatGPT. However, current speech-language models typically adopt the cascade paradigm, preventing inter-modal knowledge transfer. In this paper, we propose SpeechGPT, a large language model with intrinsic cross-modal conversational abilities, capable of perceiving and generating multi-modal content. With discrete speech representations, we construct SpeechInstruct, the first large-scale cross-modal speech instruction dataset. Additionally, we employ a three-stage training strategy that includes modality-adaptation pre-training, cross-modal instruction fine-tuning, and chain-of-modality instruction fine-tuning. The experimental results demonstrate that SpeechGPT has an impressive capacity to follow cross-modal human instructions and highlight the potential of handling multiple modalities with one model. Code and models are available inhttps://github.com/0nutation/SpeechGPT. Demos are shown inhttps://0nutation.github.io/SpeechGPT.github.io/.

</details>

---

## 34. UReader: UniversalOCR-free Visually-situated Language Understanding with Multimodal Large Language Model

- [ ] UReader: UniversalOCR-free Visually-situated Language Understanding with Multimodal Large Language Model | https://aclanthology.org/2023.findings-emnlp.187/

- **Link**: https://aclanthology.org/2023.findings-emnlp.187/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Text is ubiquitous in our visual world, conveying crucial information, such as in documents, websites, and everyday photographs. In this work, we propose UReader, a first exploration of universal OCR-free visually-situated language understanding based on the Multimodal Large Language Model (MLLM). By leveraging the shallow text recognition ability of the MLLM, we only finetuned 1.2% parameters and the training cost is much lower than previous work following domain-specific pretraining and finetuning paradigms. Concretely, UReader is jointly finetuned on a wide range of Visually-situated Language Understanding tasks via a unified instruction format. To enhance the visual text and semantic understanding, we further apply two auxiliary tasks with the same format, namely text reading and key points generation tasks. We design a shape-adaptive cropping module before the encoder-decoder architecture of MLLM to leverage the frozen low-resolution vision encoder for processing high-resolution images. Without downstream finetuning, our single model achieves state-of-the-art ocr-free performance in 8 out of 10 visually-situated language understanding tasks, across 5 domains: documents, tables, charts, natural images, and webpage screenshots. Codes and instruction-tuning datasets will be released.

</details>

---

## 35. Filling the Image Information Gap forVQA: Prompting Large Language Models to Proactively Ask Questions

- [ ] Filling the Image Information Gap forVQA: Prompting Large Language Models to Proactively Ask Questions | https://aclanthology.org/2023.findings-emnlp.189/

- **Link**: https://aclanthology.org/2023.findings-emnlp.189/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) demonstrate impressive reasoning ability and the maintenance of world knowledge not only in natural language tasks, but also in some vision-language tasks such as open-domain knowledge-based visual question answering (OK-VQA). As images are invisible to LLMs, researchers convert images to text to engage LLMs into the visual question reasoning procedure. This leads to discrepancies between images and their textual representations presented to LLMs, which consequently impedes final reasoning performance. To fill the information gap and better leverage the reasoning capability, we design a framework that enables LLMs to proactively ask relevant questions to unveil more details in the image, along with filters for refining the generated information. We validate our idea on OK-VQA and A-OKVQA. Our method continuously boosts the performance of baselines methods by an average gain of 2.15% on OK-VQA, and achieves consistent improvements across different LLMs.

</details>

---

## 36. Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models

- [ ] Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models | https://aclanthology.org/2023.findings-emnlp.226/

- **Link**: https://aclanthology.org/2023.findings-emnlp.226/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained and frozen LLMs can effectively map simple scene re-arrangement instructions to programs over a robot’s visuomotor functions through appropriate few-shot example prompting. To parse open-domain natural language and adapt to a user’s idiosyncratic procedures, not known during prompt engineering time, fixed prompts fall short. In this paper, we introduce HELPER, an embodied agent equipped with an external memory of language-program pairs that parses free-form human-robot dialogue into action programs through retrieval-augmented LLM prompting: relevant memories are retrieved based on the current dialogue, instruction, correction or VLM description, and used as in-context prompt examples for LLM querying. The memory is expanded during deployment to include pairs of user’s language and action plans, to assist future inferences and personalize them to the user’s language and routines. HELPER sets a new state-of-the-art in the TEACh benchmark in both Execution from Dialog History (EDH) and Trajectory from Dialogue (TfD), with 1.7x improvement over the previous SOTA for TfD. Our models, code and video results can be found in our project’s website: https://helper-agent-llm.github.io.

</details>

---

## 37. InteMATs: Integrating Granularity-Specific Multilingual Adapters for Cross-Lingual Transfer

- [ ] InteMATs: Integrating Granularity-Specific Multilingual Adapters for Cross-Lingual Transfer | https://aclanthology.org/2023.findings-emnlp.335/

- **Link**: https://aclanthology.org/2023.findings-emnlp.335/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multilingual language models (MLLMs) have achieved remarkable success in various cross-lingual transfer tasks. However, they suffer poor performance in zero-shot low-resource languages, particularly when dealing with longer contexts. Existing research mainly relies on full-model fine-tuning on large parallel datasets to enhance the cross-lingual alignment of MLLMs, which is computationally expensive. In this paper, we propose InteMATs, a novel approach that integrates multilingual adapters trained on texts of different levels of granularity. To achieve this, we curate a multilingual parallel dataset comprising 42 languages to pre-train sentence-level and document-level adapters under the contrastive learning framework. Extensive experiments demonstrate the effectiveness of InteMATs in improving the cross-lingual transfer performance of MLLMs, especially on low-resource languages. Finally, our comprehensive analyses and ablation studies provide a deep understanding of the high-quality representations derived by InteMATs.

</details>

---

## 38. Black-Box Tuning of Vision-Language Models with Effective Gradient Approximation

- [ ] Black-Box Tuning of Vision-Language Models with Effective Gradient Approximation | https://aclanthology.org/2023.findings-emnlp.356/

- **Link**: https://aclanthology.org/2023.findings-emnlp.356/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Parameter-efficient fine-tuning (PEFT) methods have provided an effective way for adapting large vision-language models to specific tasks or scenarios. Typically, they learn a very small scale of parameters for pre-trained models in a white-box formulation, which assumes model architectures to be known and parameters to be accessible. However, large models are often not open-source due to considerations of preventing abuse or commercial factors, hence posing a barrier to the deployment of white-box PEFT methods. To alleviate the dependence on model accessibility, we introduce collaborative black-box tuning (CBBT) for both textual prompt optimization and output feature adaptation for black-box models. Specifically, considering that the backpropagation gradients are blocked, we approximate the gradients of textual prompts by analyzing the predictions with perturbed prompts. Secondly, a lightweight adapter is deployed over the output feature of the inaccessible model, further facilitating the model adaptation process. Empowered with these designs, our CBBT is extensively evaluated on eleven downstream benchmarks and achieves remarkable improvements compared to existing black-box VL adaptation methods. Our code will be made publicly available.

</details>

---

## 39. Sparse Black-Box Multimodal Attack for Vision-Language Adversary Generation

- [ ] Sparse Black-Box Multimodal Attack for Vision-Language Adversary Generation | https://aclanthology.org/2023.findings-emnlp.384/

- **Link**: https://aclanthology.org/2023.findings-emnlp.384/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Deep neural networks have been widely applied in real-world scenarios, such as product restrictions on e-commerce and hate speech monitoring on social media, to ensure secure governance of various platforms. However, illegal merchants often deceive the detection models by adding large-scale perturbations to prohibited products, so as to earn illegal profits. Current adversarial attacks using imperceptible perturbations encounter challenges in simulating such adversarial behavior and evaluating the vulnerabilities of detection models to such perturbations. To address this issue, we propose a novel black-box multimodal attack, termed Sparse Multimodal Attack (SparseMA), which leverages sparse perturbations to simulate the adversarial behavior exhibited by illegal merchants in the black-box scenario. Moreover, SparseMA bridges the gap between images and texts by treating the separated image patches and text words uniformly in the discrete space. Extensive experiments demonstrate that SparseMA can identify the vulnerability of the model to different modalities, outperforming existing multimodal attacks and unimodal attacks. SparseMA, which is the first proposed method for black-box multimodal attacks to our knowledge, would be used as an effective tool for evaluating the robustness of multimodal models to different modalities.

</details>

---

## 40. Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks

- [ ] Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks | https://aclanthology.org/2023.findings-emnlp.40/

- **Link**: https://aclanthology.org/2023.findings-emnlp.40/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Foundation models or pre-trained models have substantially improved the performance of various language, vision, and vision-language understanding tasks. However, existing foundation models can only perform the best in one type of tasks, namely language, vision, or vision-language. It is still an open question whether it is possible to construct a general foundation model performing the best for all the understanding tasks. In this paper, we propose a new method for training the general foundation model, X-FM (the X-Foundation Model). X-FM has one language encoder, one vision encoder, and one fusion encoder, as well as a new training method. The training method includes two new techniques for learning X-FM from text, image, and image-text pair data. One is to stop gradients from the vision-language training when learning the language encoder. The other is to leverage the vision-language training to guide the learning of the vision encoder. Extensive experiments on benchmark datasets show that X-FM can significantly outperform existing general foundation models and perform better than or comparable to existing foundation models specifically for language, vision, or vision-language understanding. Code and pre-trained models are released at https://github.com/zhangxinsong-nlp/XFM.

</details>

---

## 41. MM-Reasoner: A Multi-Modal Knowledge-Aware Framework for Knowledge-Based Visual Question Answering

- [ ] MM-Reasoner: A Multi-Modal Knowledge-Aware Framework for Knowledge-Based Visual Question Answering | https://aclanthology.org/2023.findings-emnlp.437/

- **Link**: https://aclanthology.org/2023.findings-emnlp.437/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Thanks to the strong reasoning capabilities of Large Language Models (LLMs), recent approaches to knowledge-based visual question answering (KVQA) utilize LLMs with a global caption of an input image to answer a question. However, these approaches may miss key visual information that is not captured by the caption. Moreover, they cannot fully utilize the visual information required to answer the question. To address these issues, we introduce a new framework called Multi-Modal Knowledge-Aware Reasoner (MM-Reasoner) for KVQA. MM-Reasoner first utilizes a set of vision APIs, such as dense captioners, object detectors, and OCR, to extract detailed information from the image in textual format. Then, it prompts an LLM to extract query-specific knowledge from the extracted textual information to provide a rich representation that contains external knowledge, commonsense, explicit supporting facts, and rationales required for reasoning. Finally, the knowledge, query, and visual input are used to fine-tune a Vision-Language Model (VLM). At test time, MM-Reasoner uses the potential answers predicted by the VLM to iteratively update and optimize the prompt, refining its answer. Empirical studies show that MM-Reasoner achieves state-of-the-art performance on several KVQA datasets.

</details>

---

## 42. VIPHY: Probing “Visible” Physical Commonsense Knowledge

- [ ] VIPHY: Probing “Visible” Physical Commonsense Knowledge | https://aclanthology.org/2023.findings-emnlp.473/

- **Link**: https://aclanthology.org/2023.findings-emnlp.473/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have shown remarkable performance on visual reasoning tasks (e.g. attributes, location). While such tasks measure the requisite knowledge to ground and reason over a given visual instance, they do not, however, measure the ability of VLMs to retain and generalize such knowledge. In this work, we evaluate VLMs’ ability to acquire “visible” physical knowledge – the information that is easily accessible from images of static scenes, particularly along the dimensions of object color, size, and space. We build an automatic pipeline to derive a comprehensive knowledge resource for calibrating and probing these models. Our results indicate a severe gap between model and human performance across all three dimensions. Furthermore, we demonstrate that a caption pretrained LM significantly outperforms VLMs on both size and spatial tasks – highlighting that despite sufficient access to ground language with visual modality, they struggle to retain such knowledge.

</details>

---

## 43. Dataset Bias Mitigation in Multiple-Choice Visual Question Answering and Beyond

- [ ] Dataset Bias Mitigation in Multiple-Choice Visual Question Answering and Beyond | https://aclanthology.org/2023.findings-emnlp.576/

- **Link**: https://aclanthology.org/2023.findings-emnlp.576/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language (VL) understanding tasks evaluate models’ comprehension of complex visual scenes through multiple-choice questions. However, we have identified two dataset biases that models can exploit as shortcuts to resolve various VL tasks correctly without proper understanding. The first type of dataset bias is Unbalanced Matching bias, where the correct answer overlaps the question and image more than the incorrect answers. The second type of dataset bias is Distractor Similarity bias, where incorrect answers are overly dissimilar to the correct answer but significantly similar to other incorrect answers within the same sample. To address these dataset biases, we first propose Adversarial Data Synthesis (ADS) to generate synthetic training and debiased evaluation data. We then introduce Intra-sample Counterfactual Training (ICT) to assist models in utilizing the synthesized training data, particularly the counterfactual data, via focusing on intra-sample differentiation. Extensive experiments demonstrate the effectiveness of ADS and ICT in consistently improving model performance across different benchmarks, even in domain-shifted scenarios.

</details>

---

## 44. Pre-trained Speech Processing Models Contain Human-Like Biases that Propagate to Speech Emotion Recognition

- [ ] Pre-trained Speech Processing Models Contain Human-Like Biases that Propagate to Speech Emotion Recognition | https://aclanthology.org/2023.findings-emnlp.602/

- **Link**: https://aclanthology.org/2023.findings-emnlp.602/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Previous work has established that a person’s demographics and speech style affect how well speech processing models perform for them. But where does this bias come from? In this work, we present the Speech Embedding Association Test (SpEAT), a method for detecting bias in one type of model used for many speech tasks: pre-trained models. The SpEAT is inspired by word embedding association tests in natural language processing, which quantify intrinsic bias in a model’s representations of different concepts, such as race or valence—something’s pleasantness or unpleasantness—and capture the extent to which a model trained on large-scale socio-cultural data has learned human-like biases. Using the SpEAT, we test for six types of bias in 16 English speech models (including 4 models also trained on multilingual data), which come from the wav2vec 2.0, HuBERT, WavLM, and Whisper model families. We find that 14 or more models reveal positive valence (pleasantness) associations with abled people over disabled people, with European-Americans over African-Americans, with females over males, with U.S. accented speakers over non-U.S. accented speakers, and with younger people over older people. Beyond establishing that pre-trained speech models contain these biases, we also show that they can have real world effects. We compare biases found in pre-trained models to biases in downstream models adapted to the task of Speech Emotion Recognition (SER) and find that in 66 of the 96 tests performed (69%), the group that is more associated with positive valence as indicated by the SpEAT also tends to be predicted as speaking with higher valence by the downstream model. Our work provides evidence that, like text and image-based models, pre-trained speech based-models frequently learn human-like biases when trained on large-scale socio-cultural datasets. Our work also shows that bias found in pre-trained models can propagate to the downstream task of SER.

</details>

---

## 45. TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding

- [ ] TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding | https://aclanthology.org/2023.findings-emnlp.66/

- **Link**: https://aclanthology.org/2023.findings-emnlp.66/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale video-language pre-training has made remarkable strides in advancing video-language understanding tasks. However, the heavy computational burden of video encoding remains a formidable efficiency bottleneck, particularly for long-form videos. These videos contain massive visual tokens due to their inherent 3D properties and spatiotemporal redundancy, making it challenging to capture complex temporal and spatial relationships. To tackle this issue, we propose an efficient method called TEmporal-Spatial Token Aggregation (TESTA). TESTA condenses video semantics by adaptively aggregating similar frames, as well as similar patches within each frame. TESTA can reduce the number of visual tokens by 75% and thus accelerate video encoding. Building upon TESTA, we introduce a pre-trained video-language model equipped with a divided space-time token aggregation module in each video encoder block. We evaluate our model on five datasets for paragraph-to-video retrieval and long-form VideoQA tasks. Experimental results show that TESTA improves computing efficiency by 1.7 times, and achieves significant performance gains from its scalability in processing longer input frames, e.g., +13.7 R@1 on QuerYD and +6.5 R@1 on Condensed Movie.

</details>

---

## 46. ROME: Evaluating Pre-trained Vision-Language Models on Reasoning beyond Visual Common Sense

- [ ] ROME: Evaluating Pre-trained Vision-Language Models on Reasoning beyond Visual Common Sense | https://aclanthology.org/2023.findings-emnlp.683/

- **Link**: https://aclanthology.org/2023.findings-emnlp.683/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Humans possess a strong capability for reasoning beyond common sense. For example, given an unconventional image of a goldfish laying on the table next to an empty fishbowl, a human would effortlessly determine that the fish is not inside the fishbowl. The case, however, may be different for a vision-language model, whose reasoning could gravitate towards the common scenario that the fish is inside the bowl, despite the visual input. In this paper, we introduce a novel probing dataset named ROME (reasoning beyond commonsense knowledge) to evaluate whether the state-of-the-art pre-trained vision-language models have the reasoning capability to correctly interpret counter-intuitive content. ROME contains images that defy commonsense knowledge with regards to color, shape, material, size and positional relation. Experiments on the state-of-the-art pre-trained vision-language models reveal that most of these models are still largely incapable of interpreting counter-intuitive scenarios. We hope that ROME will spur further investigations on reasoning beyond commonsense knowledge in vision-language research.

</details>

---

## 47. IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models

- [ ] IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models | https://aclanthology.org/2023.findings-emnlp.755/

- **Link**: https://aclanthology.org/2023.findings-emnlp.755/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The field of vision-and-language (VL) understanding has made unprecedented progress with end-to-end large pre-trained VL models (VLMs). However, they still fall short in zero-shot reasoning tasks that require multi-step inferencing. To achieve this goal, previous works resort to a divide-and-conquer pipeline. In this paper, we argue that previous efforts have several inherent shortcomings: 1) They rely on domain-specific sub-question decomposing models. 2) They force models to predict the final answer even if the sub-questions or sub-answers provide insufficient information. We address these limitations via IdealGPT, a framework that iteratively decomposes VL reasoning using large language models (LLMs). Specifically, IdealGPT utilizes an LLM to generate sub-questions, a VLM to provide corresponding sub-answers, and another LLM to reason to achieve the final answer. These three modules perform the divide-and-conquer procedure iteratively until the model is confident about the final answer to the main question. We evaluate IdealGPT on multiple challenging VL reasoning tasks under a zero-shot setting. In particular, our IdealGPT outperforms the best existing GPT-4-like models by an absolute 10% on VCR and 15% on SNLI-VE. Code is available at https://github.com/Hxyou/IdealGPT.

</details>

---

## 48. Scaling Vision-Language Models with Sparse Mixture of Experts

- [ ] Scaling Vision-Language Models with Sparse Mixture of Experts | https://aclanthology.org/2023.findings-emnlp.758/

- **Link**: https://aclanthology.org/2023.findings-emnlp.758/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The field of natural language processing (NLP) has made significant strides in recent years, particularly in the development of large-scale vision-language models (VLMs). These models aim to bridge the gap between text and visual information, enabling a more comprehensive understanding of multimedia data. However, as these models become larger and more complex, they also become more challenging to train and deploy. One approach to addressing this challenge is the use of sparsely-gated mixture-of-experts (MoE) techniques, which divide the model into smaller, specialized sub-models that can jointly solve a task. In this paper, we explore the effectiveness of MoE in scaling vision-language models, demonstrating its potential to achieve state-of-the-art performance on a range of benchmarks over dense models of equivalent computational cost. Our research offers valuable insights into stabilizing the training of MoE models, understanding the impact of MoE on model interpretability, and balancing the trade-offs between compute performance when scaling VLMs. We hope our work will inspire further research into the use of MoE for scaling large-scale vision-language models and other multimodal machine learning applications.

</details>

---

## 49. Language Guided Visual Question Answering: Elevate Your Multimodal Language Model Using Knowledge-Enriched Prompts

- [ ] Language Guided Visual Question Answering: Elevate Your Multimodal Language Model Using Knowledge-Enriched Prompts | https://aclanthology.org/2023.findings-emnlp.809/

- **Link**: https://aclanthology.org/2023.findings-emnlp.809/

- **Conference**: EMNLP

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual question answering (VQA) is the task of answering questions about an image. The task assumes an understanding of both the image and the question to provide a natural language answer. VQA has gained popularity in recent years due to its potential applications in a wide range of fields, including robotics, education, and healthcare. In this paper, we focus on knowledge-augmented VQA, where answering the question requires commonsense knowledge, world knowledge, and reasoning about ideas and concepts not present in the image. We propose a multimodal framework that uses language guidance (LG) in the form of rationales, image captions, scene graphs, etc to answer questions more accurately. We benchmark our method on the multi-choice question-answering task of the A-OKVQA, Science-QA, VSR, and IconQA datasets using CLIP and BLIP models. We show that the use of language guidance is a simple but powerful and effective strategy for visual question answering. Our language guidance improves the performance of CLIP by 7.6% and BLIP-2 by 4.8% in the challenging A-OKVQA dataset. We also observe consistent improvement in performance on the Science-QA, VSR, and IconQA datasets when using the proposed language guidances. The implementation of LG-VQA is publicly available at https://github.com/declare-lab/LG-VQA.

</details>

---

