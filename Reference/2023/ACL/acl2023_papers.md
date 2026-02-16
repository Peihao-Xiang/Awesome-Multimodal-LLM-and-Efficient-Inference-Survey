# ACL 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_acl2023_papers.csv

## 1. LAVIS: A One-stop Library for Language-Vision Intelligence

- [ ] LAVIS: A One-stop Library for Language-Vision Intelligence | https://aclanthology.org/2023.acl-demo.3/

- **Link**: https://aclanthology.org/2023.acl-demo.3/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce LAVIS, an open-source deep learning library for LAnguage-VISion research and applications. LAVIS aims to serve as a one-stop comprehensive library that brings recent advancements in the language-vision field accessible for researchers and practitioners, as well as fertilizing future research and development. It features a unified interface to easily access state-of-the-art image-language, video-language models and common datasets. LAVIS supports training, evaluation and benchmarking on a rich variety of tasks, including multimodal classification, retrieval, captioning, visual question answering, dialogue and pre-training. In the meantime, the library is also highly extensible and configurable, facilitating future development and customization. In this technical report, we describe design principles, key components and functionalities of the library, and also present benchmarking results across common language-vision tasks.

</details>

---

## 2. Alfred: A System for Prompted Weak Supervision

- [ ] Alfred: A System for Prompted Weak Supervision | https://aclanthology.org/2023.acl-demo.46/

- **Link**: https://aclanthology.org/2023.acl-demo.46/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Alfred is the first system for programmatic weak supervision (PWS) that creates training data for machine learning by prompting. In contrast to typical PWS systems where weak supervision sources are programs coded by experts, Alfred enables users to encode their subject matter expertise via natural language prompts for language and vision-language models. Alfred provides a simple Python interface for the key steps of this emerging paradigm, with a high-throughput backend for large-scale data labeling. Users can quickly create, evaluate, and refine their prompt-based weak supervision sources; map the results to weak labels; and resolve their disagreements with a label model. Alfred enables a seamless local development experience backed by models served from self-managed computing clusters. It automatically optimizes the execution of prompts with optimized batching mechanisms. We find that this optimization improves query throughput by 2.9x versus a naive approach. We present two example use cases demonstrating Alfred on YouTube comment spam detection and pet breeds classification. Alfred is open source, available athttps://github.com/BatsResearch/alfred.

</details>

---

## 3. “Let’s not Quote out of Context”: Unified Vision-Language Pretraining for Context Assisted Image Captioning

- [ ] “Let’s not Quote out of Context”: Unified Vision-Language Pretraining for Context Assisted Image Captioning | https://aclanthology.org/2023.acl-industry.67/

- **Link**: https://aclanthology.org/2023.acl-industry.67/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Well-formed context aware image captions and tags in enterprise content such as marketing material are critical to ensure their brand presence and content recall. Manual creation and updates to ensure the same is non trivial given the scale and the tedium towards this task. We propose a new unified Vision-Language (VL) model based on the One For All (OFA) model, with a focus on context-assisted image captioning where the caption is generated based on both the image and its context. Our approach aims to overcome the context-independent (image and text are treated independently) nature of the existing approaches. We exploit context by pretraining our model with datasets of three tasks- news image captioning where the news article is the context, contextual visual entailment, and keyword extraction from the context. The second pretraining task is a new VL task, and we construct and release two datasets for the task with 1.1M and 2.2K data instances. Our system achieves state-of-the-art results with an improvement of up to 8.34 CIDEr score on the benchmark news image captioning datasets. To the best of our knowledge, ours is the first effort at incorporating contextual information in pretraining the models for the VL tasks.

</details>

---

## 4. KAFA: Rethinking Image Ad Understanding with Knowledge-Augmented Feature Adaptation of Vision-Language Models

- [ ] KAFA: Rethinking Image Ad Understanding with Knowledge-Augmented Feature Adaptation of Vision-Language Models | https://aclanthology.org/2023.acl-industry.74/

- **Link**: https://aclanthology.org/2023.acl-industry.74/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Image ad understanding is a crucial task with wide real-world applications. Although highly challenging with the involvement of diverse atypical scenes, real-world entities, and reasoning over scene-texts, how to interpret image ads is relatively under-explored, especially in the era of foundational vision-language models (VLMs) featuring impressive generalizability and adaptability. In this paper, we perform the first empirical study of image ad understanding through the lens of pre-trained VLMs. We benchmark and reveal practical challenges in adapting these VLMs to image ad understanding. We propose a simple feature adaptation strategy to effectively fuse multimodal information for image ads and further empower it with knowledge of real-world entities. We hope our study draws more attention to image ad understanding which is broadly relevant to the advertising industry.

</details>

---

## 5. CocaCLIP: Exploring Distillation of Fully-Connected Knowledge Interaction Graph for Lightweight Text-Image Retrieval

- [ ] CocaCLIP: Exploring Distillation of Fully-Connected Knowledge Interaction Graph for Lightweight Text-Image Retrieval | https://aclanthology.org/2023.acl-industry.8/

- **Link**: https://aclanthology.org/2023.acl-industry.8/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained text-image models with dual-encoder architectures (such as CLIP) are typically adopted for various vision-language applications, including text-image retrieval. However, these models are still less practical on edge devices or for real-time situations, due to the substantial indexing and inference time and the large consumption of computational resources. Although knowledge distillation techniques have been widely utilized for uni-modal model compression, how to expand them to the situation when the numbers of modalities and teachers/students are doubled has been rarely studied. In this paper, we conduct comprehensive experiments on this topic and propose the fully-Connected knowledge interaction graph (Coca) technique for cross-modal pre-training distillation. Based on our findings, the resulting CocaCLIP achieves SOTA performances on the widely-used Flickr30K and MSCOCO benchmarks under the lightweight setting. An industry application of our method on an e-commercial platform further demonstrates the significant effectiveness of CocaCLIP.

</details>

---

## 6. KG-FLIP: Knowledge-guided Fashion-domain Language-Image Pre-training forE-commerce

- [ ] KG-FLIP: Knowledge-guided Fashion-domain Language-Image Pre-training forE-commerce | https://aclanthology.org/2023.acl-industry.9/

- **Link**: https://aclanthology.org/2023.acl-industry.9/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Various Vision-Language Pre-training (VLP) models (e.g., CLIP, BLIP) have sprung up and dramatically advanced the benchmarks for public general-domain datasets (e.g., COCO, Flickr30k). Such models usually learn the cross-modal alignment from large-scale well-aligned image-text datasets without leveraging external knowledge. Adapting these models to downstream applications in specific domains like fashion requires fine-grained in-domain image-text corpus, which are usually less semantically aligned and in small scale that requires efficient pre-training strategies. In this paper, we propose a knowledge-guided fashion-domain language-image pre-training (FLIP) framework that focuses on learning fine-grained representations in e-commerce domain and utilizes external knowledge (i.e., product attribute schema), to improve the pre-training efficiency. Experiments demonstrate that FLIP outperforms previous state-of-the-art VLP models on Amazon data and on the Fashion-Gen dataset by large margins. FLIP has been successfully deployed in the Amazon catalog system to backfill missing attributes and improve the customer shopping experience.

</details>

---

## 7. TableVLM: Multi-modal Pre-training for Table Structure Recognition

- [ ] TableVLM: Multi-modal Pre-training for Table Structure Recognition | https://aclanthology.org/2023.acl-long.137/

- **Link**: https://aclanthology.org/2023.acl-long.137/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Tables are widely used in research and business, which are suitable for human consumption, but not easily machine-processable, particularly when tables are present in images. One of the main challenges to extracting data from images of tables is accurately recognizing table structures, especially for complex tables with cross rows and columns. In this study, we propose a novel multi-modal pre-training model for table structure recognition, named TableVLM.With a two-stream multi-modal transformer-based encoder-decoder architecture, TableVLM learns to capture rich table structure-related features by multiple carefully-designed unsupervised objectives inspired by the notion of masked visual-language modeling. To pre-train this model, we also created a dataset, called ComplexTable, which consists of 1,000K samples to be released publicly. Experiment results show that the model built on pre-trained TableVLM can improve the performance up to 1.97% in tree-editing-distance-score on ComplexTable.

</details>

---

## 8. Multi-modal Action Chain Abductive Reasoning

- [ ] Multi-modal Action Chain Abductive Reasoning | https://aclanthology.org/2023.acl-long.254/

- **Link**: https://aclanthology.org/2023.acl-long.254/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Abductive Reasoning, has long been considered to be at the core ability of humans, which enables us to infer the most plausible explanation of incomplete known phenomena in daily life. However, such critical reasoning capability is rarely investigated for contemporary AI systems under such limited observations. To facilitate this research community, this paper sheds new light onAbductive Reasoningby studying a new vision-language task,Multi-modalAction chain abductiveReasoning (MAR), together with a large-scaleAbductive Reasoningdataset: Given an incomplete set of language described events, MAR aims to imagine the most plausible event by spatio-temporal grounding in past video and then infer the hypothesis of subsequent action chain that can best explain the language premise. To solve this task, we propose a strong baseline model that realizes MAR from two perspectives: (i) we first introduce the transformer, which learns to encode the observation to imagine the plausible event with explicitly interpretable event grounding in the video based on the commonsense knowledge recognition ability. (ii) To complete the assumption of a follow-up action chain, we design a novel symbolic module that can complete strict derivation of the progressive action chain layer by layer. We conducted extensive experiments on the proposed dataset, and the experimental study shows that the proposed model significantly outperforms existing video-language models in terms of effectiveness on our newly created MAR dataset.

</details>

---

## 9. Cross-modal Attention Congruence Regularization for Vision-Language Relation Alignment

- [ ] Cross-modal Attention Congruence Regularization for Vision-Language Relation Alignment | https://aclanthology.org/2023.acl-long.298/

- **Link**: https://aclanthology.org/2023.acl-long.298/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite recent progress towards scaling up multimodal vision-language models, these models are still known to struggle on compositional generalization benchmarks such as Winoground. We find that a critical component lacking from current vision-language models is relation-level alignment: the ability to match directional semantic relations in text (e.g., ‘mug in grass’) with spatial relationships in the image (e.g., the position of the mug relative to the grass). To tackle this problem, we show that relation alignment can be enforced by encouraging the language attention from ‘mug’ to ‘grass’ (capturing the semantic relation ‘in’) to match the visual attention from the mug to the grass (capturing the corresponding physical relation). Tokens and their corresponding objects are softly identified using a weighted mean of cross-modal attention. We prove that this notion of soft cross-modal equivalence is equivalent to enforcing congruence between vision and language attention matrices under a ‘change of basis’ provided by the cross-modal attention matrix. Intuitively, our approach projects visual attention into the language attention space to calculate its divergence from the actual language attention, and vice versa. We apply our Cross-modal Attention Congruence Regularization (CACR) loss to fine-tune UNITER and improve its Winoground Group score by 5.75 points.

</details>

---

## 10. World-to-Words: Grounded Open Vocabulary Acquisition through Fast Mapping in Vision-Language Models

- [ ] World-to-Words: Grounded Open Vocabulary Acquisition through Fast Mapping in Vision-Language Models | https://aclanthology.org/2023.acl-long.31/

- **Link**: https://aclanthology.org/2023.acl-long.31/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The ability to connect language units to their referents in the physical world, referred to as grounding, is crucial to learning and understanding grounded meanings of words. While humans demonstrate fast mapping in new word learning, it remains unclear whether modern vision-language models can truly represent language with their grounded meanings, and how grounding may further bootstrap new word learning. To this end, we introduce Grounded Open Vocabulary Acquisition (GOVA) to examine grounding and bootstrapping in open-world language learning. As an initial attempt, we propose World-to-Words (W2W), a novel visually-grounded language model by pre-training on image-text pairs highlighting grounding as an objective. Through extensive experiments and analysis, we demonstrate that W2W is a more coherent and fast grounded word learner, and that the grounding ability acquired during pre-training helps the model to learn unseen words more rapidly and robustly.

</details>

---

## 11. Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training

- [ ] Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training | https://aclanthology.org/2023.acl-long.315/

- **Link**: https://aclanthology.org/2023.acl-long.315/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce Cross-View Language Modeling, a simple and effective pre-training framework that unifies cross-lingual and cross-modal pre-training with shared architectures and objectives. Our approach is motivated by a key observation that cross-lingual and cross-modal pre-training share the same goal of aligning two different views of the same object into a common semantic space. To this end, the cross-view language modeling framework considers both multi-modal data (i.e., image-caption pairs) and multi-lingual data (i.e., parallel sentence pairs) as two different views of the same object, and trains the model to align the two views by maximizing the mutual information between them with conditional masked language modeling and contrastive learning. We pre-train CCLM, a Cross-lingual Cross-modal Language Model, with the cross-view language modeling framework. Empirical results on IGLUE, a multi-lingual multi-modal benchmark, and two multi-lingual image-text retrieval datasets show that while conceptually simpler, CCLM significantly outperforms the prior state-of-the-art with an average absolute improvement of over 10%. Moreover, CCLM is the first multi-lingual multi-modal pre-trained model that surpasses the translate-test performance of representative English vision-language models by zero-shot cross-lingual transfer.

</details>

---

## 12. Towards a Common Understanding of Contributing Factors for Cross-Lingual Transfer in Multilingual Language Models: A Review

- [ ] Towards a Common Understanding of Contributing Factors for Cross-Lingual Transfer in Multilingual Language Models: A Review | https://aclanthology.org/2023.acl-long.323/

- **Link**: https://aclanthology.org/2023.acl-long.323/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In recent years, pre-trained Multilingual Language Models (MLLMs) have shown a strong ability to transfer knowledge across different languages. However, given that the aspiration for such an ability has not been explicitly incorporated in the design of the majority of MLLMs, it is challenging to obtain a unique and straightforward explanation for its emergence. In this review paper, we survey literature that investigates different factors contributing to the capacity of MLLMs to perform zero-shot cross-lingual transfer and subsequently outline and discuss these factors in detail. To enhance the structure of this review and to facilitate consolidation with future studies, we identify five categories of such factors. In addition to providing a summary of empirical evidence from past studies, we identify consensuses among studies with consistent findings and resolve conflicts among contradictory ones. Our work contextualizes and unifies existing research streams which aim at explaining the cross-lingual potential of MLLMs. This review provides, first, an aligned reference point for future research and, second, guidance for a better-informed and more efficient way of leveraging the cross-lingual capacity of MLLMs.

</details>

---

## 13. Unifying Cross-Lingual and Cross-Modal Modeling Towards Weakly Supervised Multilingual Vision-Language Pre-training

- [ ] Unifying Cross-Lingual and Cross-Modal Modeling Towards Weakly Supervised Multilingual Vision-Language Pre-training | https://aclanthology.org/2023.acl-long.327/

- **Link**: https://aclanthology.org/2023.acl-long.327/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multilingual Vision-Language Pre-training (VLP) is a promising but challenging topic due to the lack of large-scale multilingual image-text pairs. Existing works address the problem by translating English data into other languages, which is intuitive and the generated data is usually limited in form and scale. In this paper, we explore a more practical and scalable setting: weakly supervised multilingual VLP with only English image-text pairs and multilingual text corpora. We argue that the universal multilingual representation learned from texts allows the cross-modal interaction learned in English to be transferable to other languages. To this end, we propose a framework to effectively unify cross-lingual and cross-modal pre-training. For unified modeling on different data, we design an architecture with flexible modules to learn different interactions. Moreover, two unified tasks are introduced to efficiently guide the unified cross-lingual cross-modal learning. Extensive experiments demonstrate that our pre-trained model learns universal multilingual multimodal representations, allowing effective cross-lingual transfer on multimodal tasks. Code and models are available athttps://github.com/FudanDISC/weakly-supervised-mVLP.

</details>

---

## 14. Scene Graph as Pivoting: Inference-time Image-free Unsupervised Multimodal Machine Translation with Visual Scene Hallucination

- [ ] Scene Graph as Pivoting: Inference-time Image-free Unsupervised Multimodal Machine Translation with Visual Scene Hallucination | https://aclanthology.org/2023.acl-long.329/

- **Link**: https://aclanthology.org/2023.acl-long.329/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this work, we investigate a more realistic unsupervised multimodal machine translation (UMMT) setup, inference-time image-free UMMT, where the model is trained with source-text image pairs, and tested with only source-text inputs. First, we represent the input images and texts with the visual and language scene graphs (SG), where such fine-grained vision-language features ensure a holistic understanding of the semantics. To enable pure-text input during inference, we devise a visual scene hallucination mechanism that dynamically generates pseudo visual SG from the given textual SG. Several SG-pivoting based learning objectives are introduced for unsupervised translation training. On the benchmark Multi30K data, our SG-based method outperforms the best-performing baseline by significant BLEU scores on the task and setup, helping yield translations with better completeness, relevance and fluency without relying on paired images. Further in-depth analyses reveal how our model advances in the task setting.

</details>

---

## 15. A Multi-Modal Context Reasoning Approach for Conditional Inference on Joint Textual and Visual Clues

- [ ] A Multi-Modal Context Reasoning Approach for Conditional Inference on Joint Textual and Visual Clues | https://aclanthology.org/2023.acl-long.601/

- **Link**: https://aclanthology.org/2023.acl-long.601/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Conditional inference on joint textual and visual clues is a multi-modal reasoning task that textual clues provide prior permutation or external knowledge, which are complementary with visual content and pivotal to deducing the correct option. Previous methods utilizing pretrained vision-language models (VLMs) have achieved impressive performances, yet they show a lack of multimodal context reasoning capability, especially for text-modal information. To address this issue, we propose a Multi-modal Context Reasoning approach, named ModCR. Compared to VLMs performing reasoning via cross modal semantic alignment, it regards the given textual abstract semantic and objective image information as the pre-context information and embeds them into the language model to perform context reasoning. Different from recent vision-aided language models used in natural language processing, ModCR incorporates the multi-view semantic alignment information between language and vision by introducing the learnable alignment prefix between image and text in the pretrained language model. This makes the language model well-suitable for such multi-modal reasoning scenario on joint textual and visual clues. We conduct extensive experiments on two corresponding data sets and experimental results show significantly improved performance (exact gain by 4.8% on PMR test set) compared to previous strong baselines.

</details>

---

## 16. MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering

- [ ] MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering | https://aclanthology.org/2023.acl-long.714/

- **Link**: https://aclanthology.org/2023.acl-long.714/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual language data such as plots, charts, and infographics are ubiquitous in the human world. However, state-of-the-art vision-language models do not perform well on these data. We propose MatCha (Math reasoning and Chart derendering pretraining) to enhance visual language models’ capabilities in jointly modeling charts/plots and language data. Specifically, we propose several pretraining tasks that cover plot deconstruction and numerical reasoning which are the key capabilities in visual language modeling. We perform the MatCha pretraining starting from Pix2Struct, a recently proposed image-to-text visual language model. On standard benchmarks such as PlotQA and ChartQA, the MatCha model outperforms state-of-the-art methods by as much as nearly 20%. We also examine how well MatCha pretraining transfers to domains such as screenshots, textbook diagrams, and document figures and observe overall improvement, verifying the usefulness of MatCha pretraining on broader visual language tasks.

</details>

---

## 17. PuMer: Pruning and Merging Tokens for Efficient Vision Language Models

- [ ] PuMer: Pruning and Merging Tokens for Efficient Vision Language Models | https://aclanthology.org/2023.acl-long.721/

- **Link**: https://aclanthology.org/2023.acl-long.721/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision language (VL) models use Transformers to perform cross-modal interactions between the input text and image. These cross-modal interactions are computationally expensive and memory-intensive due to the quadratic complexity of processing the input image and text. We present PuMer: a token reduction framework that uses text-informed Pruning and modality-aware Merging strategies to progressively reduce the tokens of input image and text, improving model inference speed and reducing memory footprint. PuMer learns to keep salient image tokens related to the input text and merges similar textual and visual tokens by adding lightweight token reducer modules at several cross-modal layers in the VL model. Training PuMer is mostly the same as finetuning the original VL model but faster. Our evaluation for two vision language models on four downstream VL tasks shows PuMer increases inference throughput by up to 2x and reduces memory footprint by over 50% while incurring less than a 1% accuracy drop.

</details>

---

## 18. mCLIP: MultilingualCLIPvia Cross-lingual Transfer

- [ ] mCLIP: MultilingualCLIPvia Cross-lingual Transfer | https://aclanthology.org/2023.acl-long.728/

- **Link**: https://aclanthology.org/2023.acl-long.728/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language pretrained (VLP) models like CLIP have shown remarkable performance on various downstream cross-modal tasks. However, they are usually biased towards English due to the lack of sufficient non-English image-text pairs. Existing multilingual VLP methods often learn retrieval-inefficient single-stream models by translation-augmented non-English image-text pairs. In this paper, we introduce mCLIP, a retrieval-efficient dual-stream multilingual VLP model, trained by aligning the CLIP model and a Multilingual Text Encoder (MTE) through a novel Triangle Cross-modal Knowledge Distillation (TriKD) method. It is parameter-efficient as only two light projectors on the top of them are updated during distillation. Furthermore, to enhance the token- and sentence-level multilingual representation of the MTE, we propose to train it with machine translation and contrastive learning jointly before the TriKD to provide a better initialization. Empirical results show that mCLIP achieves new state-of-the-art performance for both zero-shot and finetuned multilingual image-text retrieval task.

</details>

---

## 19. Wukong-Reader: Multi-modal Pre-training for Fine-grained Visual Document Understanding

- [ ] Wukong-Reader: Multi-modal Pre-training for Fine-grained Visual Document Understanding | https://aclanthology.org/2023.acl-long.748/

- **Link**: https://aclanthology.org/2023.acl-long.748/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Unsupervised pre-training on millions of digital-born or scanned documents has shown promising advances in visual document understanding (VDU). While various vision-language pre-training objectives are studied in existing solutions, the document textline, as an intrinsic granularity in VDU, has seldom been explored so far. A document textline usually contains words that are spatially and semantically correlated, which can be easily obtained from OCR engines. In this paper, we propose Wukong-Reader, trained with new pre-training objectives to leverage the structural knowledge nested in document textlines. We introduce textline-region contrastive learning to achieve fine-grained alignment between the visual regions and texts of document textlines. Furthermore, masked region modeling and textline-grid matching are also designed to enhance the visual and layout representations of textlines. Experiments show that Wukong-Reader brings superior performance on various VDU tasks in both English and Chinese. The fine-grained alignment over textlines also empowers Wukong-Reader with promising localization ability.

</details>

---

## 20. ManagerTower: Aggregating the Insights of Uni-Modal Experts for Vision-Language Representation Learning

- [ ] ManagerTower: Aggregating the Insights of Uni-Modal Experts for Vision-Language Representation Learning | https://aclanthology.org/2023.acl-long.811/

- **Link**: https://aclanthology.org/2023.acl-long.811/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Two-Tower Vision-Language (VL) models have shown promising improvements on various downstream VL tasks. Although the most advanced work improves performance by building bridges between encoders, it suffers from ineffective layer-by-layer utilization of uni-modal representations and cannot flexibly exploit different levels of uni-modal semantic knowledge. In this work, we propose ManagerTower, a novel VL model architecture that gathers and combines the insights of pre-trained uni-modal experts at different levels. The managers introduced in each cross-modal layer can adaptively aggregate uni-modal semantic knowledge to facilitate more comprehensive cross-modal alignment and fusion. ManagerTower outperforms previous strong baselines both with and without Vision-Language Pre-training (VLP). With only 4M VLP data, ManagerTower achieves superior performances on various downstream VL tasks, especially 79.15% accuracy on VQAv2 Test-Std, 86.56% IR@1 and 95.64% TR@1 on Flickr30K. Code and checkpoints are available athttps://github.com/LooperXX/ManagerTower.

</details>

---

## 21. Vision Language Pre-training by Contrastive Learning with Cross-Modal Similarity Regulation

- [ ] Vision Language Pre-training by Contrastive Learning with Cross-Modal Similarity Regulation | https://aclanthology.org/2023.acl-long.819/

- **Link**: https://aclanthology.org/2023.acl-long.819/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we reconsider the problem of (partial) false negative samples from the Mutual Information (MI) Maximization perspective, the traditional contrastive loss (like InfoNCE loss) will equally push away the anchor of all positive samples and negative samples regardless of their possible semantic similarities. We theoretically show that InfoNCE loss will not only maximize the MI between the anchor and positive samples but minimize the MI between the anchor and false negative samples even though they share similar semantic which could provide a possible theoretical explanation for the observation of the existence of false negative samples in the cross-modal contrastive learning will decrease the downstream task performance of VLP models. Above analysis motivate us to propose the VLP model with a novel Semantic Awared Contrastive Learning framework named SACL where different negative samples are assigned with different contrastive weights according to the semantic similarity between them and the anchor.

</details>

---

## 22. Measuring Progress in Fine-grained Vision-and-Language Understanding

- [ ] Measuring Progress in Fine-grained Vision-and-Language Understanding | https://aclanthology.org/2023.acl-long.87/

- **Link**: https://aclanthology.org/2023.acl-long.87/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

While pretraining on large-scale image–text data from the Web has facilitated rapid progress on many vision-and-language (V&L) tasks, recent work has demonstrated that pretrained models lack “fine-grained” understanding, such as the ability to recognise relationships, verbs, and numbers in images. This has resulted in an increased interest in the community to either develop new benchmarks or models for such capabilities. To better understand and quantify progress in this direction, we investigate four competitive V&L models on four fine-grained benchmarks. Through our analysis, we find that X-VLM (Zeng et al., 2022) consistently outperforms other baselines, and that modelling innovations can impact performance more than scaling Web data, which even degrades performance sometimes. Through a deeper investigation of X-VLM, we highlight the importance of both novel losses and rich data sources for learning fine-grained skills. Finally, we inspect training dynamics, and discover that for some tasks, performance peaks early in training or significantly fluctuates, never converging.

</details>

---

## 23. A Neural Divide-and-Conquer Reasoning Framework for Image Retrieval from Linguistically Complex Text

- [ ] A Neural Divide-and-Conquer Reasoning Framework for Image Retrieval from Linguistically Complex Text | https://aclanthology.org/2023.acl-long.909/

- **Link**: https://aclanthology.org/2023.acl-long.909/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pretrained Vision-Language Models (VLMs) have achieved remarkable performance in image retrieval from text. However, their performance drops drastically when confronted with linguistically complex texts that they struggle to comprehend. Inspired by the Divide-and-Conquer algorithm and dual-process theory, in this paper, we regard linguistically complex texts as compound proposition texts composed of multiple simple proposition sentences and propose an end-to-end Neural Divide-and-Conquer Reasoning framework, dubbed NDCR. It contains three main components: 1) Divide: a proposition generator divides the compound proposition text into simple proposition sentences and produces their corresponding representations, 2) Conquer: a pretrained VLMs-based visual-linguistic interactor achieves the interaction between decomposed proposition sentences and images, 3) Combine: a neural-symbolic reasoner combines the above reasoning states to obtain the final solution via a neural logic reasoning approach. According to the dual-process theory, the visual-linguistic interactor and neural-symbolic reasoner could be regarded as analogical reasoning System 1 and logical reasoning System 2. We conduct extensive experiments on a challenging image retrieval from contextual descriptions data set. Experimental results and analyses indicate NDCR significantly improves performance in the complex image-text reasoning problem.

</details>

---

## 24. Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages

- [ ] Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages | https://aclanthology.org/2023.acl-short.32/

- **Link**: https://aclanthology.org/2023.acl-short.32/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pre-training (VLP) has advanced the performance of many vision-language tasks, such as image-text retrieval, visual entailment, and visual reasoning. The pre-training mostly utilizes lexical databases and image queries in English. Previous work has demonstrated that the pre-training in English does not transfer well to other languages in a zero-shot setting. However, multilingual pre-trained language models (MPLM) have excelled at a variety of single-modal language tasks. In this paper, we propose a simple yet efficient approach to adapt VLP to unseen languages using MPLM.We utilize a cross-lingual contextualised token embeddings alignment approach to train text encoders for non-English languages. Our approach does not require image input and primarily uses machine translation, eliminating the need for target language data. Our evaluation across three distinct tasks (image-text retrieval, visual entailment, and natural language visual reasoning) demonstrates that this approach outperforms the state-of-the-art multilingual vision-language models without requiring large parallel corpora. Our code is available athttps://github.com/Yasminekaroui/CliCoTea.

</details>

---

## 25. MetaVL: Transferring In-Context Learning Ability From Language Models to Vision-Language Models

- [ ] MetaVL: Transferring In-Context Learning Ability From Language Models to Vision-Language Models | https://aclanthology.org/2023.acl-short.43/

- **Link**: https://aclanthology.org/2023.acl-short.43/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale language models have shown the ability to adapt to a new task via conditioning on a few demonstrations (i.e., in-context learning). However, in the vision-language domain, most large-scale pre-trained vision-language (VL) models do not possess the ability to conduct in-context learning. How can we enable in-context learning for VL models? In this paper, we study an interesting hypothesis: can we transfer the in-context learning ability from the language domain to the VL domain? Specifically, we first meta-trains a language model to perform in-context learning on NLP tasks (as in MetaICL); then we transfer this model to perform VL tasks by attaching a visual encoder. Our experiments suggest that indeed in-context learning ability can be transferred cross modalities: our model considerably improves the in-context learning capability on VL tasks and can even compensate for the size of the model significantly. On VQA, OK-VQA, and GQA, our method could outperform the baseline model while having ~20 times fewer parameters.

</details>

---

## 26. Indirectly Supervised Natural Language Processing

- [ ] Indirectly Supervised Natural Language Processing | https://aclanthology.org/2023.acl-tutorials.5/

- **Link**: https://aclanthology.org/2023.acl-tutorials.5/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This tutorial targets researchers and practitioners who are interested in ML technologies for NLP from indirect supervision. In particular, we will present a diverse thread of indirect supervision studies that try to answer the following questions: (i) when and how can we provide supervision for a target task T, if all we have is data that corresponds to a “related” task T′? (ii) humans do not use exhaustive supervision; they rely on occasional feedback, and learn from incidental signals from various sources; how can we effectively incorporate such supervision in machine learning? (iii) how can we leverage multi-modal supervision to help NLP? To the end, we will discuss several lines of research that address those challenges, including (i) indirect supervision from T ′ that handles T with outputs spanning from a moderate size to an open space, (ii) the use of sparsely occurring and incidental signals, such as partial labels, noisy labels, knowledge-based constraints, and cross-domain or cross-task annotations—all having statistical associations with the task, (iii) principled ways to measure and understand why these incidental signals can contribute to our target tasks, and (iv) indirect supervision from vision-language signals. We will conclude the tutorial by outlining directions for further investigation.

</details>

---

## 27. KU-DMIS-MSRAatRadSum23: Pre-trained Vision-Language Model for Radiology Report Summarization

- [ ] KU-DMIS-MSRAatRadSum23: Pre-trained Vision-Language Model for Radiology Report Summarization | https://aclanthology.org/2023.bionlp-1.59/

- **Link**: https://aclanthology.org/2023.bionlp-1.59/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce CheXOFA, a new pre-trained vision-language model (VLM) for the chest X-ray domain. Our model is initially pre-trained on various multimodal datasets within the general domain before being transferred to the chest X-ray domain. Following a prominent VLM, we unify various domain-specific tasks into a simple sequence-to-sequence schema. It enables the model to effectively learn the required knowledge and skills from limited resources in the domain. Demonstrating superior performance on the benchmark datasets provided by the BioNLP shared task (Delbrouck et al., 2023), our model benefits from its training across multiple tasks and domains. With subtle techniques including ensemble and factual calibration, our system achieves first place on the RadSum23 leaderboard for the hidden test set.

</details>

---

## 28. Pragmatic Inference with aCLIPListener for Contrastive Captioning

- [ ] Pragmatic Inference with aCLIPListener for Contrastive Captioning | https://aclanthology.org/2023.findings-acl.120/

- **Link**: https://aclanthology.org/2023.findings-acl.120/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We propose a simple yet effective and robust method for contrastive captioning: generating discriminative captions that distinguish target images from very similar alternative distractor images. Our approach is built on a pragmatic inference procedure that formulates captioning as a reference game between a speaker, which produces possible captions describing the target, and a listener, which selects the target given the caption. Unlike previous methods that derive both speaker and listener distributions from a single captioning model, we leverage an off-the-shelf CLIP model to parameterize the listener. Compared with captioner-only pragmatic models, our method benefits from rich vision-language alignment representations from CLIP when reasoning over distractors. Like previous methods for discriminative captioning, our method uses a hyperparameter to control the tradeoff between the informativity (how likely captions are to allow a human listener to discriminate the target image) and the fluency of the captions. However, we find that our method is substantially more robust to the value of this hyperparameter than past methods, which allows us to automatically optimize the captions for informativity — outperforming past methods for discriminative captioning by 11% to 15% accuracy in human evaluations.

</details>

---

## 29. Retrieving Multimodal Prompts for Generative Visual Question Answering

- [ ] Retrieving Multimodal Prompts for Generative Visual Question Answering | https://aclanthology.org/2023.findings-acl.158/

- **Link**: https://aclanthology.org/2023.findings-acl.158/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent years have witnessed impressive results of pre-trained vision-language models on knowledge-intensive tasks such as visual question answering (VQA). Despite the recent advances in VQA, existing methods mainly adopt a discriminative formulation that predicts answers within a pre-defined label set, leading to easy overfitting on low-resource domains (e.g., medicine) and poor generalization under domain shift to another dataset. To tackle this limitation, we propose a novel generative model enhanced by multimodal prompt retrieval (MPR) that integrates retrieved prompts and multimodal features to generate answers in free text. Our generative model enables rapid zero-shot dataset adaptation to unseen data distributions and open-set answer labels across datasets. Our experiments on medical VQA tasks show that MPR outperforms its non-retrieval counterpart by up to 30% accuracy points in a few-shot domain adaptation setting.

</details>

---

## 30. Fusion or Defusion? Flexible Vision-and-Language Pre-Training

- [ ] Fusion or Defusion? Flexible Vision-and-Language Pre-Training | https://aclanthology.org/2023.findings-acl.316/

- **Link**: https://aclanthology.org/2023.findings-acl.316/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Existing approaches in the vision-and-language pre-training (VLP) paradigm mainly deploy either fusion-based encoders or dual-encoders, failing to achieve both effectiveness and efficiency in downstream multimodal tasks. In this paper, we build a flexible VLP model by incorporating cross-modal fusions into a dual-encoder architecture, where the introduced fusion modules can be easily decoupled from the dual encoder so as to switch the model to a fusion-free one. To better absorb cross-modal features from the fusion modules, we design a cross-modal knowledge transfer strategy along with other comprehensive pre-training tasks to guide the training process, which can further strengthen both the fusion-based and fusion-free representation learning. Extensive experiments conducted on various downstream vision-language tasks show that our proposed model is well-equipped with effectiveness as well as efficiency, demonstrating a superior performance compared with other strong VLP models.

</details>

---

## 31. Visually-Enhanced Phrase Understanding

- [ ] Visually-Enhanced Phrase Understanding | https://aclanthology.org/2023.findings-acl.363/

- **Link**: https://aclanthology.org/2023.findings-acl.363/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language pre-training has exhibited strong performance in various visual and textual understanding tasks. Recently, the textual encoders of multi-modal pre-trained models have been shown to generate high-quality textual representations, which often outperform models that are purely text-based, such as BERT. In this study, our objective is to utilize both textual and visual encoders of multi-modal pre-trained models to enhance language understanding tasks. We achieve this by generating an image associated with a textual prompt, thus enriching the representation of a phrase for downstream tasks. Results from experiments conducted on four benchmark datasets demonstrate that our proposed method, which leverages visually-enhanced text representations, significantly improves performance in the entity clustering task.

</details>

---

## 32. Transferring General Multimodal Pretrained Models to Text Recognition

- [ ] Transferring General Multimodal Pretrained Models to Text Recognition | https://aclanthology.org/2023.findings-acl.37/

- **Link**: https://aclanthology.org/2023.findings-acl.37/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper proposes a new method, OFA-OCR, to transfer multimodal pretrained models to text recognition. Specifically, we recast text recognition as image captioning and directly transfer a unified vision-language pretrained model to the end task. Without pretraining on large-scale annotated or synthetic text recognition data, OFA-OCR outperforms the baselines and achieves state-of-the-art performance in the Chinese text recognition benchmark. Additionally, we construct an OCR pipeline with OFA-OCR, and we demonstrate that it can achieve competitive performance with the product-level API.

</details>

---

## 33. XtremeCLIP: Extremely Parameter-efficient Tuning for Low-resource Vision Language Understanding

- [ ] XtremeCLIP: Extremely Parameter-efficient Tuning for Low-resource Vision Language Understanding | https://aclanthology.org/2023.findings-acl.397/

- **Link**: https://aclanthology.org/2023.findings-acl.397/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recently, Contrastive Visual-Language Pre-training (CLIP) has demonstrated remarkable capability in various Visual Language Understanding (VLU) tasks. Yet, most CLIP-based methods require tasks-specific designs and sufficient training data. In this paper, we introduce a simple yet efficient paradigm for low-resource VLU named XtremeCLIP, which involves very few trainable parameters to improve the generalization ability of the trained models. In our XtremeCLIP framework, we reformulate a series of VLU tasks as a unified open-book affinity-matching problem. Furthermore, to handle the insufficient supervised signals in small datasets, we adopt contrastive learning to utilize the implicit sorting information of ground-truth labels to provide more supervised cues. Extensive experiments over multiple datasets on visual entailment, visual question answering, and image classification show that XtremeCLIP consistently outperforms existing baselines in low-resource settings.

</details>

---

## 34. FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

- [ ] FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing | https://aclanthology.org/2023.findings-acl.398/

- **Link**: https://aclanthology.org/2023.findings-acl.398/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Textual scene graph parsing has become increasingly important in various vision-language applications, including image caption evaluation and image retrieval. However, existing scene graph parsers that convert image captions into scene graphs often suffer from two types of errors. First, the generated scene graphs fail to capture the true semantics of the captions or the corresponding images, resulting in a lack of faithfulness. Second, the generated scene graphs have high inconsistency, with the same semantics represented by different annotations. To address these challenges, we propose a novel dataset, which involves re-annotating the captions in Visual Genome (VG) using a new intermediate representation called FACTUAL-MR. FACTUAL-MR can be directly converted into faithful and consistent scene graph annotations. Our experimental results clearly demonstrate that the parser trained on our dataset outperforms existing approaches in terms of faithfulness and consistency. This improvement leads to a significant performance boost in both image caption evaluation and zero-shot image retrieval tasks. Furthermore, we introduce a novel metric for measuring scene graph similarity, which, when combined with the improved scene graph parser, achieves state-of-the-art (SOTA) results on multiple benchmark datasets for the aforementioned tasks.

</details>

---

## 35. A Multi-dimensional study on Bias in Vision-Language models

- [ ] A Multi-dimensional study on Bias in Vision-Language models | https://aclanthology.org/2023.findings-acl.403/

- **Link**: https://aclanthology.org/2023.findings-acl.403/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In recent years, joint Vision-Language (VL) models have increased in popularity and capability. Very few studies have attempted to investigate bias in VL models, even though it is a well-known issue in both individual modalities. This paper presents the first multi-dimensional analysis of bias in English VL models, focusing on gender, ethnicity, and age as dimensions. When subjects are input as images, pre-trained VL models complete a neutral template with a hurtful word 5% of the time, with higher percentages for female and young subjects. Bias presence in downstream models has been tested on Visual Question Answering. We developed a novel bias metric called the Vision-Language Association Test based on questions designed to elicit biased associations between stereotypical concepts and targets. Our findings demonstrate that pre-trained VL models contain biases that are perpetuated in downstream tasks.

</details>

---

## 36. UniFine: A Unified and Fine-grained Approach for Zero-shot Vision-Language Understanding

- [ ] UniFine: A Unified and Fine-grained Approach for Zero-shot Vision-Language Understanding | https://aclanthology.org/2023.findings-acl.49/

- **Link**: https://aclanthology.org/2023.findings-acl.49/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language tasks, such as VQA, SNLI-VE, and VCR are challenging because they require the model’s reasoning ability to understand the semantics of the visual world and natural language. Supervised methods working for vision-language tasks have been well-studied. However, solving these tasks in a zero-shot setting is less explored. Since Contrastive Language-Image Pre-training (CLIP) has shown remarkable zero-shot performance on image-text matching, previous works utilized its strong zero-shot ability by converting vision-language tasks into an image-text matching problem, and they mainly consider global-level matching (e.g., the whole image or sentence). However, we find visual and textual fine-grained information, e.g., keywords in the sentence and objects in the image, can be fairly informative for semantics understanding. Inspired by this, we propose a unified framework to take advantage of the fine-grained information for zero-shot vision-language learning, covering multiple tasks such as VQA, SNLI-VE, and VCR. Our experiments show that our framework outperforms former zero-shot methods on VQA and achieves substantial improvement on SNLI-VE and VCR. Furthermore, our ablation studies confirm the effectiveness and generalizability of our proposed method.

</details>

---

## 37. Deeply Coupled Cross-Modal Prompt Learning

- [ ] Deeply Coupled Cross-Modal Prompt Learning | https://aclanthology.org/2023.findings-acl.504/

- **Link**: https://aclanthology.org/2023.findings-acl.504/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal foundation models (e.g., CLIP) have excelled in zero-shot generalization. Prompt tuning involved in the knowledge transfer from foundation models to downstream tasks has gained significant attention recently. Existing prompt-tuning methods in cross-modal learning, however, either solely focus on language branch, or learn vision-language interaction in a shallow mechanism. In this context, we propose a Deeply coupled Cross-modal Prompt learning (DCP) method based on CLIP. DCP flexibly accommodates the interplay between vision and language with a Cross-Modal Prompt Attention (CMPA) mechanism, which enables the mutual exchange of respective representation through a well-connected multi-head attention progressively and strongly. We then conduct comprehensive few-shot learning experiments on 11 image classification datasets and analyze the robustness to domain shift as well. Thorough experimental analysis evidently demonstrates the superb few-shot generalization and compelling domain adaption capacity of a well-executed DCP.

</details>

---

## 38. Entropy-guided Vocabulary Augmentation of Multilingual Language Models for Low-resource Tasks

- [ ] Entropy-guided Vocabulary Augmentation of Multilingual Language Models for Low-resource Tasks | https://aclanthology.org/2023.findings-acl.548/

- **Link**: https://aclanthology.org/2023.findings-acl.548/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multilingual language models (MLLMs) like mBERTpromise to extend the benefits of NLP research to low-resource languages (LRLs). However, LRL words are under-represented in the wordpiece/subword vocabularies of MLLMs. This leads to many LRL words getting replaced by UNK, or concatenated from morphologically unrelated wordpieces, leading to low task accuracy. (Pre)-training MLLMs after including LRL documents is resource-intensive in terms of both human inputs and computational resources. In response, we propose EVALM (entropy-based vocabulary augmented language model), which uses a new task-cognizant measurement to detect the most vulnerable LRL words, whose wordpiece segmentations are undesirable. EVALM then provides reasonable initializations of their embeddings, followed by limited fine-tuning using the small LRL task corpus. Our experiments show significant performance improvements and also some surprising limits to such vocabulary augmentation strategies in various classification tasks for multiple diverse LRLs, as well as code-mixed texts. We will release the code and data to enable further research.

</details>

---

## 39. RC3: Regularized Contrastive Cross-lingual Cross-modal Pre-training

- [ ] RC3: Regularized Contrastive Cross-lingual Cross-modal Pre-training | https://aclanthology.org/2023.findings-acl.746/

- **Link**: https://aclanthology.org/2023.findings-acl.746/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multilingual vision-language (V&L) pre-training has achieved remarkable progress in learning universal representations across different modalities and languages. In spite of recent success, there still remain challenges limiting further improvements of V&L pre-trained models in multilingual settings. Particularly, current V&L pre-training methods rely heavily on strictly-aligned multilingual image-text pairs generated from English-centric datasets through machine translation. However, the cost of collecting and translating such strictly-aligned datasets is usually unbearable. In this paper, we propose Regularized Contrastive Cross-lingual Cross-modal (RC3) pre-training, which further exploits more abundant weakly-aligned multilingual image-text pairs. Specifically, we design a regularized cross-lingual visio-textual contrastive learning objective that constrains the representation proximity of weakly-aligned visio-textual inputs according to textual relevance. Besides, existing V&L pre-training approaches mainly deal with visual inputs by either region-of-interest (ROI) features or patch embeddings. We flexibly integrate the two forms of visual features into our model for pre-training and downstream multi-modal tasks. Extensive experiments on 5 downstream multi-modal tasks across 6 languages demonstrate the effectiveness of our proposed method over competitive contrast models with strong zero-shot capability.

</details>

---

## 40. EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning

- [ ] EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning | https://aclanthology.org/2023.findings-acl.873/

- **Link**: https://aclanthology.org/2023.findings-acl.873/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have achieved impressive results in a range of vision-language tasks. However, popular VLMs usually consist of hundreds of millions of parameters which brings challenges for fine-tuning and deployment in real-world applications due to space, memory, and latency constraints. In this work, we introduce a distilling then pruning framework to compress large vision-language models into smaller, faster, and more accurate ones. We first shrink the size ofa pre-trained large VLM and apply knowledge distillation in the vision-language pre-training stage to obtain a task-agnostic compact VLM. Then we propose a modal-adaptive pruning algorithm to automatically infer the importance of vision and language modalities for different downstream tasks and adaptively remove redundant structures and neurons in different encoders with controllable target sparsity. We apply our framework to train EfficientVLM, a fast and accurate vision-language model consisting of 6 vision layers, 3 text layers, and 3 cross-modal fusion layers, accounting for only 93 million parameters in total, which is 44.3% of the teacher model. EfficientVLM retains 98.4% performance of the teacher model and accelerates its inference speed by 2.2×. EfficientVLM achieves a large absolute improvement over previous SoTA efficient VLMs of similar sizes by a large margin on various vision-language tasks, including VQAv2 (+4.9%), NLVR2 (+5.6%), ITR (R@1 on TR +17.2%, on IR + 15.6% ) and COCO caption generation (CIDEr +6.5), demonstrating a large potential on training lightweight VLMs.

</details>

---

## 41. Images in Language Space: Exploring the Suitability of Large Language Models for Vision & Language Tasks

- [ ] Images in Language Space: Exploring the Suitability of Large Language Models for Vision & Language Tasks | https://aclanthology.org/2023.findings-acl.894/

- **Link**: https://aclanthology.org/2023.findings-acl.894/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large language models have demonstrated robust performance on various language tasks using zero-shot or few-shot learning paradigms. While being actively researched, multimodal models that can additionally handle images as input have yet to catch up in size and generality with language-only models. In this work, we ask whether language-only models can be utilised for tasks that require visual input – but also, as we argue, often require a strong reasoning component. Similar to some recent related work, we make visual information accessible to the language model using separate verbalisation models. Specifically, we investigate the performance of open-source, open-access language models against GPT-3 on five vision-language tasks when given textually-encoded visual information. Our results suggest that language models are effective for solving vision-language tasks even with limited samples. This approach also enhances the interpretability of a model’s output by providing a means of tracing the output back through the verbalised image content.

</details>

---

## 42. CMU’sIWSLT2023 Simultaneous Speech Translation System

- [ ] CMU’sIWSLT2023 Simultaneous Speech Translation System | https://aclanthology.org/2023.iwslt-1.20/

- **Link**: https://aclanthology.org/2023.iwslt-1.20/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper describes CMU’s submission to the IWSLT 2023 simultaneous speech translation shared task for translating English speech to both German text and speech in a streaming fashion. We first build offline speech-to-text (ST) models using the joint CTC/attention framework. These models also use WavLM front-end features and mBART decoder initialization. We adapt our offline ST models for simultaneous speech-to-text translation (SST) by 1) incrementally encoding chunks of input speech, re-computing encoder states for each new chunk and 2) incrementally decoding output text, pruning beam search hypotheses to 1-best after processing each chunk. We then build text-to-speech (TTS) models using the VITS framework and achieve simultaneous speech-to-speech translation (SS2ST) by cascading our SST and TTS models.

</details>

---

## 43. Rutgers Multimedia Image Processing Lab atSemEval-2023 Task-1: Text-Augmentation-based Approach for Visual Word Sense Disambiguation

- [ ] Rutgers Multimedia Image Processing Lab atSemEval-2023 Task-1: Text-Augmentation-based Approach for Visual Word Sense Disambiguation | https://aclanthology.org/2023.semeval-1.204/

- **Link**: https://aclanthology.org/2023.semeval-1.204/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper describes our system used in SemEval-2023 Task-1: Visual Word Sense Disambiguation (VWSD). The VWSD task is to identify the correct image that corresponds to an ambiguous target word given limited textual context. To reduce word ambiguity and enhance image selection, we proposed several text augmentation techniques, such as prompting, WordNet synonyms, and text generation. We experimented with different vision-language pre-trained models to capture the joint features of the augmented text and image. Our approach achieved the best performance using a combination of GPT-3 text generation and the CLIP model. On the multilingual test sets, our system achieved an average hit rate (at top-1) of 51.11 and a mean reciprocal rank of 65.69.

</details>

---

## 44. LTatSemEval-2023 Task 1: Effective Zero-Shot Visual Word Sense Disambiguation Approaches using External Knowledge Sources

- [ ] LTatSemEval-2023 Task 1: Effective Zero-Shot Visual Word Sense Disambiguation Approaches using External Knowledge Sources | https://aclanthology.org/2023.semeval-1.64/

- **Link**: https://aclanthology.org/2023.semeval-1.64/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The objective of the SemEval-2023 Task 1: Visual Word Sense Disambiguation (VWSD) is to identify the image illustrating the indented meaning of a target word and some minimal additional context. The omnipresence of textual and visual data in the task strongly suggests the utilization of the recent advances in multi-modal machine learning, i.e., pretrained visiolinguistic models (VLMs). Often referred to as foundation models due to their strong performance on many vision-language downstream tasks, these models further demonstrate powerful zero-shot capabilities. In this work, we utilize various pertained VLMs in a zero-shot fashion for multiple approaches using external knowledge sources to enrich the contextual information. Further, we evaluate our methods on the final test data and extensively analyze the suitability of different knowledge sources, the influence of training data, model sizes, multi-linguality, and different textual prompting strategies. Although we are not among the best-performing systems (rank 20 of 56), our experiments described in this work prove competitive results. Moreover, we aim to contribute meaningful insights and propel multi-modal machine learning tasks like VWSD.

</details>

---

## 45. Scalable Performance Analysis for Vision-Language Models

- [ ] Scalable Performance Analysis for Vision-Language Models | https://aclanthology.org/2023.starsem-1.26/

- **Link**: https://aclanthology.org/2023.starsem-1.26/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Joint vision-language models have shown great performance over a diverse set of tasks. However, little is known about their limitations, as the high dimensional space learned by these models makes it difficult to identify semantic errors. Recent work has addressed this problem by designing highly controlled probing task benchmarks. Our paper introduces a more scalable solution that relies on already annotated benchmarks. Our method consists of extracting a large set of diverse features from a vision-language benchmark and measuring their correlation with the output of the target model. We confirm previous findings that CLIP behaves like a bag of words model and performs better with nouns and verbs; we also uncover novel insights such as CLIP getting confused by concrete words. Our framework is available athttps://github.com/MichiganNLP/Scalable-VLM-Probingand can be used with other multimodal models and benchmarks.

</details>

---

## 46. PrecogIIITH@WASSA2023: Emotion Detection forUrdu-English Code-mixed Text

- [ ] PrecogIIITH@WASSA2023: Emotion Detection forUrdu-English Code-mixed Text | https://aclanthology.org/2023.wassa-1.58/

- **Link**: https://aclanthology.org/2023.wassa-1.58/

- **Conference**: ACL

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Code-mixing refers to the phenomenon of using two or more languages interchangeably within a speech or discourse context. This practice is particularly prevalent on social media platforms, and determining the embedded affects in a code-mixed sentence remains as a challenging problem. In this submission we describe our system for WASSA 2023 Shared Task on Emotion Detection in English-Urdu code-mixed text. In our system we implement a multiclass emotion detection model with label space of 11 emotions. Samples are code-mixed English-Urdu text, where Urdu is written in romanised form. Our submission is limited to one of the subtasks - Multi Class classification and we leverage transformer-based Multilingual Large Language Models (MLLMs), XLM-RoBERTa and Indic-BERT. We fine-tune MLLMs on the released data splits, with and without pre-processing steps (translation to english), for classifying texts into the appropriate emotion category. Our methods did not surpass the baseline, and our submission is ranked sixth overall.

</details>

---

