# AAAI 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_aaai2024_papers.csv

## 1. An Empirical Study of CLIP for Text-Based Person Search

- [ ] An Empirical Study of CLIP for Text-Based Person Search | https://ojs.aaai.org/index.php/AAAI/article/view/27801

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27801

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Text-based Person Search (TBPS) aims to retrieve the person images using natural language descriptions. Recently, Contrastive Language Image Pretraining (CLIP), a universal large cross-modal vision-language pre-training model, has remarkably performed over various cross-modal downstream tasks due to its powerful cross-modal semantic learning capacity. TPBS, as a fine-grained cross-modal retrieval task, is also facing the rise of research on the CLIP-based TBPS. In order to explore the potential of the visual-language pre-training model for downstream TBPS tasks, this paper makes the first attempt to conduct a comprehensive empirical study of CLIP for TBPS and thus contribute a straightforward, incremental, yet strong TBPS-CLIP baseline to the TBPS community. We revisit critical design considerations under CLIP, including data augmentation and loss function. The model, with the aforementioned designs and practical training tricks, can attain satisfactory performance without any sophisticated modules. Also, we conduct the probing experiments of TBPS-CLIP in model generalization and model compression, demonstrating the effectiveness of TBPS-CLIP from various aspects. This work is expected to provide empirical insights and highlight future CLIP-based TBPS research.

</details>

---

## 2. Prompt-Based Distribution Alignment for Unsupervised Domain Adaptation

- [ ] Prompt-Based Distribution Alignment for Unsupervised Domain Adaptation | https://ojs.aaai.org/index.php/AAAI/article/view/27830

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27830

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, despite the unprecedented success of large pre-trained visual-language models (VLMs) on a wide range of downstream tasks, the real-world unsupervised domain adaptation (UDA) problem is still not well explored. Therefore, in this paper, we first experimentally demonstrate that the unsupervised-trained VLMs can significantly reduce the distribution discrepancy between source and target domains, thereby improving the performance of UDA. However, a major challenge for directly deploying such models on downstream UDA tasks is prompt engineering, which requires aligning the domain knowledge of source and target domains, since the performance of UDA is severely influenced by a good domain-invariant representation. We further propose a Prompt-based Distribution Alignment (PDA) method to incorporate the domain knowledge into prompt learning. Specifically, PDA employs a two-branch prompt-tuning paradigm, namely base branch and alignment branch. The base branch focuses on integrating class-related representation into prompts, ensuring discrimination among different classes.  To further minimize domain discrepancy, for the alignment branch, we construct feature banks for both the source and target domains and propose image-guided feature tuning (IFT) to make the input attend to feature banks, which effectively integrates self-enhanced and cross-domain features into the model.  In this way, these two branches can be mutually promoted to enhance the adaptation of VLMs for UDA. We conduct extensive experiments on three benchmarks to demonstrate that our proposed PDA achieves state-of-the-art performance. The code is available at https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment.

</details>

---

## 3. Image Safeguarding: Reasoning with Conditional Vision Language Model and Obfuscating Unsafe Content Counterfactually

- [ ] Image Safeguarding: Reasoning with Conditional Vision Language Model and Obfuscating Unsafe Content Counterfactually | https://ojs.aaai.org/index.php/AAAI/article/view/27835

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27835

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Social media platforms are being increasingly used by malicious actors to share unsafe content, such as images depicting sexual activity, cyberbullying, and self-harm. Consequently, major platforms use artificial intelligence (AI) and human moderation to obfuscate such images to make them safer. Two critical needs for obfuscating unsafe images is that an accurate rationale for obfuscating image regions must be provided, and the sensitive regions should be obfuscated (e.g. blurring) for users' safety. This process involves addressing two key problems: (1) the reason for obfuscating unsafe images demands the platform to provide an accurate rationale that must be grounded in unsafe image-specific attributes, and (2) the unsafe regions in the image must be minimally obfuscated while still depicting the safe regions. In this work, we address these key issues by first performing visual reasoning by designing a visual reasoning model (VLM) conditioned on pre-trained unsafe image classifiers to provide an accurate rationale grounded in unsafe image attributes, and then proposing a counterfactual explanation algorithm that minimally identifies and obfuscates unsafe regions for safe viewing, by first utilizing an unsafe image classifier attribution matrix to guide segmentation for a more optimal subregion segmentation followed by an informed greedy search to determine the minimum number of subregions required to modify the classifier's output based on attribution score. Extensive experiments on uncurated data from social networks emphasize the efficacy of our proposed method. We make our code available at: https://github.com/SecureAIAutonomyLab/ConditionalVLM

</details>

---

## 4. Domain-Controlled Prompt Learning

- [ ] Domain-Controlled Prompt Learning | https://ojs.aaai.org/index.php/AAAI/article/view/27853

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27853

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large pre-trained vision-language models, such as CLIP, have shown remarkable generalization capabilities across various tasks when appropriate text prompts are provided. However, adapting these models to specific domains, like remote sensing images (RSIs), medical images, etc, remains unexplored and challenging. Existing prompt learning methods often lack domain-awareness or domain-transfer mechanisms, leading to suboptimal performance due to the misinterpretation of specific images in natural image patterns.  To tackle this dilemma, we proposed a Domain-Controlled Prompt Learning for the specific domains. Specifically, the large-scale specific domain foundation model (LSDM) is first introduced to provide essential specific domain knowledge. Using lightweight neural networks, we transfer this knowledge into domain biases, which control both the visual and language branches to obtain domain-adaptive prompts in a directly incorporating manner.  Simultaneously, to overcome the existing overfitting challenge, we propose a novel noisy-adding strategy, without extra trainable parameters, to help the model escape the suboptimal solution in a global domain oscillation manner. Experimental results show our method achieves state-of-the-art performance in specific domain image recognition datasets. Our code is available at https://github.com/caoql98/DCPL.

</details>

---

## 5. EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE

- [ ] EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE | https://ojs.aaai.org/index.php/AAAI/article/view/27872

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27872

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Building scalable vision-language models to learn from diverse, multimodal data remains an open challenge. In this paper, we introduce an Efficient Vision-languagE foundation model, namely EVE, which is one unified multimodal Transformer pre-trained solely by one unified pre-training task. Specifically, EVE encodes both vision and language within a shared Transformer network integrated with modality-aware sparse Mixture-of-Experts (MoE) modules, which capture modality-specific information by selectively switching to different experts. To unify pre-training tasks of vision and language, EVE performs masked signal modeling on image-text pairs to reconstruct masked signals, i.e., image pixels and text tokens, given visible signals. This simple yet effective pre-training objective accelerates training by 4x compared to the model pre-trained with Image-Text Contrastive and Image-Text Matching losses. Owing to the combination of the unified architecture and pre-training task, EVE is easy to scale up, enabling better downstream performance with fewer resources and faster training speed. Despite its simplicity, EVE achieves state-of-the-art performance on various vision-language downstream tasks, including visual question answering, visual reasoning, and image-text retrieval.

</details>

---

## 6. Weak Distribution Detectors Lead to Stronger Generalizability of Vision-Language Prompt Tuning

- [ ] Weak Distribution Detectors Lead to Stronger Generalizability of Vision-Language Prompt Tuning | https://ojs.aaai.org/index.php/AAAI/article/view/27918

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27918

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose a generalized method for boosting the generalization ability of pre-trained vision-language models (VLMs) while fine-tuning on downstream few-shot tasks. The idea is realized by exploiting out-of-distribution (OOD) detection to predict whether a sample belongs to a base distribution or a novel distribution and then using the score generated by a dedicated competition based scoring function to fuse the zero-shot and few-shot classifier. The fused classifier is dynamic, which will bias towards the zero-shot classifier if a sample is more likely from the distribution pre-trained on, leading to improved base-to-novel generalization ability. Our method is performed only in test stage, which is applicable to boost existing methods without time-consuming re-training. Extensive experiments show that even weak distribution detectors can still improve VLMs' generalization ability. Specifically, with the help of OOD detectors, the harmonic mean of CoOp and ProGrad increase by 2.6 and 1.5 percentage points over 11 recognition datasets in the base-to-novel setting.

</details>

---

## 7. Simple Image-Level Classification Improves Open-Vocabulary Object Detection

- [ ] Simple Image-Level Classification Improves Open-Vocabulary Object Detection | https://ojs.aaai.org/index.php/AAAI/article/view/27939

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27939

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Open-Vocabulary Object Detection (OVOD) aims to detect novel objects beyond a given set of base categories on which the detection model is trained. Recent OVOD methods focus on adapting the image-level pre-trained vision-language models (VLMs), such as CLIP, to a region-level object detection task via, eg., region-level knowledge distillation, regional prompt learning, or region-text pre-training, to expand the detection vocabulary. These methods have demonstrated remarkable performance in recognizing regional visual concepts, but they are weak in exploiting the VLMs' powerful global scene understanding ability learned from the billion-scale image-level text descriptions. This limits their capability in detecting hard objects of small, blurred, or occluded appearance from novel/base categories, whose detection heavily relies on contextual information. To address this, we propose a novel approach, namely Simple Image-level Classification for Context-Aware Detection Scoring (SIC-CADS), to leverage the superior global knowledge yielded from CLIP for complementing the current OVOD models from a global perspective. The core of SIC-CADS is a multi-modal multi-label recognition (MLR) module that learns the object co-occurrence-based contextual information from CLIP to recognize all possible object categories in the scene. These image-level MLR scores can then be utilized to refine the instance-level detection scores of the current OVOD models in detecting those hard objects. This is verified by extensive empirical results on two popular benchmarks, OV-LVIS and OV-COCO, which show that SIC-CADS achieves significant and consistent improvement when combined with different types of OVOD models. Further, SIC-CADS also improves the cross-dataset generalization ability on Objects365 and OpenImages. Code is available at https://github.com/mala-lab/SIC-CADS.

</details>

---

## 8. SoftCLIP: Softer Cross-Modal Alignment Makes CLIP Stronger

- [ ] SoftCLIP: Softer Cross-Modal Alignment Makes CLIP Stronger | https://ojs.aaai.org/index.php/AAAI/article/view/27955

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27955

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

During the preceding biennium, vision-language pre-training has achieved noteworthy success on several downstream tasks. Nevertheless, acquiring high-quality image-text pairs, where the pairs are entirely exclusive of each other, remains a challenging task, and noise exists in the commonly used datasets. To address this issue, we propose SoftCLIP, a novel approach that relaxes the strict one-to-one constraint and achieves a soft cross-modal alignment by introducing a softened target, which is generated from the fine-grained intra-modal self-similarity. The intra-modal guidance is indicative to enable two pairs have some local similarities and model many-to-many relationships between the two modalities. Besides, since the positive still dominates in the softened target distribution, we disentangle the negatives in the distribution to further boost the relation alignment with the negatives in the cross-modal learning. Extensive experiments demonstrate the effectiveness of SoftCLIP. In particular, on ImageNet zero-shot classification task, using CC3M/CC12M as pre-training dataset, SoftCLIP brings a top-1 accuracy improvement of 6.8%/7.2% over the CLIP baseline.

</details>

---

## 9. AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models

- [ ] AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/27963

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27963

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) such as MiniGPT-4 and LLaVA have demonstrated the capability of understanding images and achieved remarkable performance in various visual tasks. Despite their strong abilities in recognizing common objects due to extensive training datasets, they lack specific domain knowledge and have a weaker understanding of localized details within objects, which hinders their effectiveness in the Industrial Anomaly Detection (IAD) task. On the other hand, most existing IAD methods only provide anomaly scores and necessitate the manual setting of thresholds to distinguish between normal and abnormal samples, which restricts their practical implementation. In this paper, we explore the utilization of LVLM to address the IAD problem and propose AnomalyGPT, a novel IAD approach based on LVLM. We generate training data by simulating anomalous images and producing corresponding textual descriptions for each image. We also employ an image decoder to provide fine-grained semantic and design a prompt learner to fine-tune the LVLM using prompt embeddings. Our AnomalyGPT eliminates the need for manual threshold adjustments, thus directly assesses the presence and locations of anomalies. Additionally, AnomalyGPT supports multi-turn dialogues and exhibits impressive few-shot in-context learning capabilities. With only one normal shot, AnomalyGPT achieves the state-of-the-art performance with an accuracy of 86.1%, an image-level AUC of 94.1%, and a pixel-level AUC of 95.3% on the MVTec-AD dataset.

</details>

---

## 10. COMMA: Co-articulated Multi-Modal Learning

- [ ] COMMA: Co-articulated Multi-Modal Learning | https://ojs.aaai.org/index.php/AAAI/article/view/27997

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27997

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pretrained large-scale vision-language models such as CLIP have demonstrated excellent generalizability over a series of downstream tasks. However, they are sensitive to the variation of input text prompts and need a selection of prompt templates to achieve satisfactory performance. Recently, various methods have been proposed to dynamically learn the prompts as the textual inputs to avoid the requirements of laboring hand-crafted prompt engineering in the fine-tuning process. We notice that these methods are suboptimal in two aspects. First, the prompts of the vision and language branches in these methods are usually separated or uni-directionally correlated. Thus, the prompts of both branches are not fully correlated and may not provide enough guidance to align the representations of both branches. Second, it's observed that most previous methods usually achieve better performance on seen classes but cause performance degeneration on unseen classes compared to CLIP. This is because the essential generic knowledge learned in the pretraining stage is partly forgotten in the fine-tuning process. In this paper, we propose Co-Articulated Multi-Modal Learning (COMMA) to handle the above limitations. Especially, our method considers prompts from both branches to generate the prompts to enhance the representation alignment of both branches. Besides, to alleviate forgetting about the essential knowledge, we minimize the feature discrepancy between the learned prompts and the embeddings of hand-crafted prompts in the pre-trained CLIP in the late transformer layers. We evaluate our method across three representative tasks of generalization to novel classes, new target datasets and unseen domain shifts. Experimental results demonstrate the superiority of our method by exhibiting a favorable performance boost upon all tasks with high efficiency. Code is available at https://github.com/hulianyuyy/COMMA.

</details>

---

## 11. BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions

- [ ] BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions | https://ojs.aaai.org/index.php/AAAI/article/view/27999

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27999

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs), which extend Large Language Models (LLM) by incorporating visual understanding capability, have demonstrated significant advancements in addressing open-ended visual question-answering (VQA) tasks. However, these models cannot accurately interpret images infused with text, a common occurrence in real-world scenarios. Standard procedures for extracting information from images often involve learning a fixed set of query embeddings. These embeddings are designed to encapsulate image contexts and are later used as soft prompt inputs in LLMs. Yet, this process is limited to the token count, potentially curtailing the recognition of scenes with text-rich context. To improve upon them, the present study introduces BLIVA: an augmented version of InstructBLIP with Visual Assistant. BLIVA incorporates the query embeddings from InstructBLIP and also directly projects encoded patch embeddings into the LLM, a technique inspired by LLaVA. This approach assists the model to capture intricate details potentially missed during the query decoding process. Empirical evidence demonstrates that our model, BLIVA, significantly enhances performance in processing text-rich VQA benchmarks (up to 17.76% in OCR-VQA benchmark) and in undertaking general (not particularly text-rich) VQA benchmarks (up to 7.9% in Visual Spatial Reasoning benchmark), and achieved 17.72% overall improvement in a comprehensive multimodal LLM benchmark (MME), comparing to our baseline InstructBLIP. BLIVA demonstrates significant capability in decoding real-world images, irrespective of text presence. To demonstrate the broad industry applications enabled by BLIVA, we evaluate the model using a new dataset comprising YouTube thumbnails paired with question-answer sets across 11 diverse categories. For researchers interested in further exploration, our code and models are freely accessible at https://github.com/mlpc-ucsd/BLIVA.

</details>

---

## 12. Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-Modal Structured Representations

- [ ] Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-Modal Structured Representations | https://ojs.aaai.org/index.php/AAAI/article/view/28017

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28017

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language pre-training has achieved significant performance in multi-modal understanding and generation tasks. However, existing methods often perform poorly on image-text matching tasks that require structured representations, i.e., representations of objects, attributes, and relations. The models cannot make a distinction between "An astronaut rides a horse" and "A horse rides an astronaut". This is because they fail to fully leverage structured knowledge when learning multi-modal representations. In this paper, we present an end-to-end framework Structure-CLIP, which integrates Scene Graph Knowledge (SGK) to enhance multi-modal structured representations. Firstly, we use scene graphs to guide the construction of semantic negative examples, which results in an increased emphasis on learning structured representations. Moreover, a Knowledge-Enhance Encoder (KEE) is proposed to leverage SGK as input to further enhance structured representations. To verify the effectiveness of the proposed framework, we pre-train our model with the aforementioned approaches and conduct experiments on downstream tasks.  Experimental results demonstrate that Structure-CLIP achieves state-of-the-art (SOTA) performance on VG-Attribution and VG-Relation datasets, with 12.5% and 4.1% ahead of the multi-modal SOTA model respectively. Meanwhile, the results on MSCOCO indicate that Structure-CLIP significantly enhances the structured representations while maintaining the ability of general representations. Our code is available at https://github.com/zjukg/Structure-CLIP.

</details>

---

## 13. TiMix: Text-Aware Image Mixing for Effective Vision-Language Pre-training

- [ ] TiMix: Text-Aware Image Mixing for Effective Vision-Language Pre-training | https://ojs.aaai.org/index.php/AAAI/article/view/28025

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28025

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Self-supervised Multi-modal Contrastive Learning (SMCL) remarkably advances modern Vision-Language Pre-training (VLP) models by aligning visual and linguistic modalities. Due to noises in web-harvested text-image pairs, however, scaling up training data volume in SMCL presents considerable obstacles in terms of computational cost and data inefficiency. To improve data efficiency in VLP, we propose Text-aware Image Mixing (TiMix), which integrates mix-based data augmentation techniques into SMCL, yielding significant performance improvements without significantly increasing computational overhead. We provide a theoretical analysis of TiMix from a mutual information (MI) perspective, showing that mixed data samples for cross-modal contrastive learning implicitly serve as a regularizer for  the contrastive loss. The experimental results demonstrate that TiMix exhibits a comparable performance on downstream tasks, even with a reduced amount of training data and shorter training time, when benchmarked against existing methods. This work empirically and theoretically demonstrates the potential of data mixing for data-efficient and computationally viable VLP, benefiting broader VLP model adoption in practical scenarios. Our code is available on https://github.com/chaoyajiang/TiMiX/tree/main.

</details>

---

## 14. Transferable Video Moment Localization by Moment-Guided Query Prompting

- [ ] Transferable Video Moment Localization by Moment-Guided Query Prompting | https://ojs.aaai.org/index.php/AAAI/article/view/28028

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28028

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video moment localization stands as a crucial task within the realm of computer vision, entailing the identification of temporal moments in untrimmed videos that bear semantic relevance to the supplied natural language queries. This work delves into a relatively unexplored facet of the task: the transferability of video moment localization models. This concern is addressed by evaluating moment localization models within a cross-domain transfer setting. In this setup, we curate multiple datasets distinguished by substantial domain gaps. The model undergoes training on one of these datasets, while validation and testing are executed using the remaining datasets. To confront the challenges inherent in this scenario, we draw inspiration from the recently introduced large-scale pre-trained vision-language models. Our focus is on exploring how the strategic utilization of these resources can bolster the capabilities of a model designed for video moment localization. Nevertheless, the distribution of language queries in video moment localization usually diverges from the text used by pre-trained models, exhibiting distinctions in aspects such as length, content, expression, and more. To mitigate the gap, this work proposes a Moment-Guided Query Prompting (MGQP) method for video moment localization. Our key idea is to generate multiple distinct and complementary prompt primitives through stratification of the original queries. Our approach is comprised of a prompt primitive constructor, a multimodal prompt refiner, and a holistic prompt incorporator. We carry out extensive experiments on Charades-STA, TACoS, DiDeMo, and YouCookII datasets, and investigate the efficacy of the proposed method using various pre-trained models, such as CLIP, ActionCLIP, CLIP4Clip, and VideoCLIP. The experimental results demonstrate the effectiveness of our proposed method.

</details>

---

## 15. Delving into Multimodal Prompting for Fine-Grained Visual Classification

- [ ] Delving into Multimodal Prompting for Fine-Grained Visual Classification | https://ojs.aaai.org/index.php/AAAI/article/view/28034

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28034

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-grained visual classification (FGVC) involves categorizing fine subdivisions within a broader category, which poses challenges due to subtle inter-class discrepancies and large intra-class variations. However, prevailing approaches primarily focus on uni-modal visual concepts. Recent advancements in pre-trained vision-language models have demonstrated remarkable performance in various high-level vision tasks, yet the applicability of such models to FGVC tasks remains uncertain. In this paper, we aim to fully exploit the capabilities of cross-modal description to tackle FGVC tasks and propose a novel multimodal prompting solution, denoted as MP-FGVC, based on the contrastive language-image pertaining (CLIP) model. Our MP-FGVC comprises a multimodal prompts scheme and a multimodal adaptation scheme. The former includes Subcategory-specific Vision Prompt (SsVP) and Discrepancy-aware Text Prompt (DaTP), which explicitly highlights the subcategory-specific discrepancies from the perspectives of both vision and language. The latter aligns the vision and text prompting elements in a common semantic space, facilitating cross-modal collaborative reasoning through a Vision-Language Fusion Module (VLFM) for further improvement on FGVC. Moreover, we tailor a two-stage optimization strategy for MP-FGVC to fully leverage the pre-trained CLIP model and expedite efficient adaptation for FGVC. Extensive experiments conducted on four FGVC datasets demonstrate the effectiveness of our MP-FGVC.

</details>

---

## 16. Expediting Contrastive Language-Image Pretraining via Self-Distilled Encoders

- [ ] Expediting Contrastive Language-Image Pretraining via Self-Distilled Encoders | https://ojs.aaai.org/index.php/AAAI/article/view/28052

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28052

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision language pretraining (VLP) have been largely attributed to the large-scale data collected from the web. However, uncurated dataset contains weakly correlated image-text pairs, causing data inefficiency. To address the issue, knowledge distillation have been explored at the expense of extra image and text momentum encoders to generate teaching signals for misaligned image-text pairs. In this paper, our goal is to resolve the misalignment problem with an efficient distillation framework. To this end, we propose ECLIPSE: Expediting Contrastive Language-Image Pretraining with Self-distilled Encoders. ECLIPSE features a distinctive distillation architecture wherein a shared text encoder is utilized between an online image encoder and a momentum image encoder. This strategic design choice enables the distillation to operate within a unified projected space of text embedding, resulting in better performance. Based on the unified text embedding space, ECLIPSE compensates for the additional computational cost of the momentum image encoder by expediting the online image encoder. Through our extensive experiments, we validate that there is a sweet spot between expedition and distillation where the partial view from the expedited online image encoder interacts complementarily with the momentum teacher. As a result, ECLIPSE outperforms its counterparts while achieving substantial acceleration in inference speed.

</details>

---

## 17. LaViP: Language-Grounded Visual Prompting

- [ ] LaViP: Language-Grounded Visual Prompting | https://ojs.aaai.org/index.php/AAAI/article/view/28064

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28064

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a language-grounded visual prompting method to adapt the visual encoder of vision-language models for downstream tasks. By capitalizing on language integration, we devise a parameter-efficient strategy to adjust the input of the visual encoder, eliminating the need to modify or add to the model's parameters. Due to this design choice, our algorithm can operate even in black-box scenarios, showcasing adaptability in situations where access to the model's parameters is constrained. We will empirically demonstrate that, compared to prior art, grounding visual prompts with language enhances both the accuracy and speed of adaptation. Moreover, our algorithm excels in base-to-novel class generalization, overcoming limitations of visual prompting and exhibiting the capacity to generalize beyond seen classes. We thoroughly assess and evaluate our method across a variety of image recognition datasets, such as EuroSAT, UCF101, DTD, and CLEVR, spanning different learning situations, including few-shot adaptation, base-to-novel class generalization, and transfer learning.

</details>

---

## 18. Point2Real: Bridging the Gap between Point Cloud and Realistic Image for Open-World 3D Recognition

- [ ] Point2Real: Bridging the Gap between Point Cloud and Realistic Image for Open-World 3D Recognition | https://ojs.aaai.org/index.php/AAAI/article/view/28088

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28088

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recognition in open-world scenarios is an important and challenging field, where Vision-Language Pre-training paradigms have greatly impacted the 2D domain. This inspires a growing interest in introducing 2D pre-trained models, such as CLIP, into the 3D domain to enhance the ability of point cloud understanding. Considering the difference between discrete 3D point clouds and real-world 2D images, reducing the domain gap is crucial. Some recent works project point clouds onto a 2D plane to enable 3D zero-shot capabilities without training. However, this simplistic approach leads to an unclear or even distorted geometric structure, limiting the potential of 2D pre-trained models in 3D. To address the domain gap, we propose Point2Real, a training-free framework based on the realistic rendering technique to automate the transformation of the 3D point cloud domain into the Vision-Language domain. Specifically, Point2Real leverages a shape recovery module that devises an iterative ball-pivoting algorithm to convert point clouds into meshes, narrowing the gap in shape at first. To simulate photo-realistic images, a set of refined textures as candidates is applied for rendering, where the CLIP confidence is utilized to select the suitable one. Moreover, to tackle the viewpoint challenge, a heuristic multi-view adapter is implemented for feature aggregation, which exploits the depth surface as an effective indicator of view-specific discriminability for recognition. We conduct experiments on ModelNet10, ModelNet40, and ScanObjectNN datasets, and the results demonstrate that Point2Real outperforms other approaches in zero-shot and few-shot tasks by a large margin.

</details>

---

## 19. Adaptive Uncertainty-Based Learning for Text-Based Person Retrieval

- [ ] Adaptive Uncertainty-Based Learning for Text-Based Person Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/28101

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28101

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Text-based person retrieval aims at retrieving a specific pedestrian image from a gallery based on textual descriptions. The primary challenge is how to overcome the inherent heterogeneous modality gap in the situation of significant intra-class variation and minimal inter-class variation. Existing approaches commonly employ vision-language pre-training or attention mechanisms to learn appropriate cross-modal alignments from noise inputs. Despite commendable progress, current methods inevitably suffer from two defects: 1) Matching ambiguity, which mainly derives from unreliable matching pairs; 2) One-sided cross-modal alignments, stemming from the absence of exploring one-to-many correspondence, i.e., coarse-grained semantic alignment. These critical issues significantly deteriorate retrieval performance. To this end, we propose a novel framework termed Adaptive Uncertainty-based Learning (AUL) for text-based person retrieval from the uncertainty perspective. Specifically, our AUL framework consists of three key components: 1) Uncertainty-aware Matching Filtration that leverages Subjective Logic to effectively mitigate the disturbance of unreliable matching pairs and select high-confidence cross-modal matches for training; 2) Uncertainty-based Alignment Refinement, which not only simulates coarse-grained alignments by constructing uncertainty representations but also performs progressive learning to incorporate coarse- and fine-grained alignments properly; 3) Cross-modal Masked Modeling that aims at exploring more comprehensive relations between vision and language. Extensive experiments demonstrate that our AUL method consistently achieves state-of-the-art performance on three benchmark datasets in supervised, weakly supervised, and domain generalization settings. Our code is available at https://github.com/CFM-MSG/Code-AUL.

</details>

---

## 20. VLM2Scene: Self-Supervised Image-Text-LiDAR Learning with Foundation Models for Autonomous Driving Scene Understanding

- [ ] VLM2Scene: Self-Supervised Image-Text-LiDAR Learning with Foundation Models for Autonomous Driving Scene Understanding | https://ojs.aaai.org/index.php/AAAI/article/view/28121

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28121

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision and language foundation models (VLMs) have showcased impressive capabilities in 2D scene understanding. However, their latent potential in elevating the understanding of 3D autonomous driving scenes remains untapped. In this paper, we propose VLM2Scene, which exploits the potential of VLMs to enhance 3D self-supervised representation learning through our proposed image-text-LiDAR contrastive learning strategy. Specifically, in the realm of autonomous driving scenes, the inherent sparsity of LiDAR point clouds poses a notable challenge for point-level contrastive learning methods. This method often grapples with limitations tied to a restricted receptive field and the presence of noisy points. To tackle this challenge, our approach emphasizes region-level learning, leveraging regional masks without semantics derived from the vision foundation model. This approach capitalizes on valuable contextual information to enhance the learning of point cloud representations. First, we introduce Region Caption Prompts to generate fine-grained language descriptions for the corresponding regions, utilizing the language foundation model. These region prompts then facilitate the establishment of positive and negative text-point pairs within the contrastive loss framework. Second, we propose a Region Semantic Concordance Regularization, which involves a semantic-filtered region learning and a region semantic assignment strategy. The former aims to filter the false negative samples based on the semantic distance, and the latter mitigates potential inaccuracies in pixel semantics, thereby enhancing overall semantic consistency. Extensive experiments on representative autonomous driving datasets demonstrate that our self-supervised method significantly outperforms other counterparts. Codes are available at https://github.com/gbliao/VLM2Scene.

</details>

---

## 21. Weakly Supervised Open-Vocabulary Object Detection

- [ ] Weakly Supervised Open-Vocabulary Object Detection | https://ojs.aaai.org/index.php/AAAI/article/view/28127

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28127

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite weakly supervised object detection (WSOD) being a promising step toward evading strong instance-level annotations, its capability is confined to closed-set categories within a single training dataset. In this paper, we propose a novel weakly supervised open-vocabulary object detection framework, namely WSOVOD, to extend traditional WSOD to detect novel concepts and utilize diverse datasets with only image-level annotations. To achieve this, we explore three vital strategies, including dataset-level feature adaptation, image-level salient object localization, and region-level vision-language alignment. First, we perform data-aware feature extraction to produce an input-conditional coefficient, which is leveraged into dataset attribute prototypes to identify dataset bias and help achieve cross-dataset generalization. Second, a customized location-oriented weakly supervised region proposal network is proposed to utilize high-level semantic layouts from the category-agnostic segment anything model to distinguish object boundaries. Lastly, we introduce a proposal-concept synchronized multiple-instance network, i.e., object mining and refinement with visual-semantic alignment, to discover objects matched to the text embeddings of concepts. Extensive experiments on Pascal VOC and MS COCO demonstrate that the proposed WSOVOD achieves new state-of-the-art compared with previous WSOD methods in both close-set object localization and detection tasks. Meanwhile, WSOVOD enables cross-dataset and open-vocabulary learning to achieve on-par or even better performance than well-established fully-supervised open-vocabulary object detection (FSOVOD).

</details>

---

## 22. Stitching Segments and Sentences towards Generalization in Video-Text Pre-training

- [ ] Stitching Segments and Sentences towards Generalization in Video-Text Pre-training | https://ojs.aaai.org/index.php/AAAI/article/view/28202

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28202

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-training models have recently achieved remarkable results on various multi-modal downstream tasks. However, most of these models rely on contrastive learning or masking modeling to align global features across modalities, neglecting the local associations between video frames and text tokens. This limits the model’s ability to perform fine-grained matching and generalization, especially for tasks that selecting segments in long videos based on query texts. To address this issue, we propose a novel stitching and matching pre-text task for video-language pre-training that encourages fine-grained interactions between modalities. Our task involves stitching video frames or sentences into longer sequences and predicting the positions of cross-model queries in the stitched sequences. The individual frame and sentence representations are thus aligned via the stitching and matching strategy, encouraging the fine-grained interactions between videos and texts. in the stitched sequences for the cross-modal query. We conduct extensive experiments on various benchmarks covering text-to-video retrieval, video question answering, video captioning, and moment retrieval. Our results demonstrate that the proposed method significantly improves the generalization capacity of the video-text pre-training models.

</details>

---

## 23. Unifying Visual and Vision-Language Tracking via Contrastive Learning

- [ ] Unifying Visual and Vision-Language Tracking via Contrastive Learning | https://ojs.aaai.org/index.php/AAAI/article/view/28205

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28205

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Single object tracking aims to locate the target object in a video sequence according to the state specified by different modal references, including the initial bounding box (BBOX), natural language (NL), or both (NL+BBOX). Due to the gap between different modalities, most existing trackers are designed for single or partial of these reference settings and overspecialize on the specific modality. Differently, we present a unified tracker called UVLTrack, which can simultaneously handle all three reference settings (BBOX, NL, NL+BBOX) with the same parameters. The proposed UVLTrack enjoys several merits. First, we design a modality-unified feature extractor for joint visual and language feature learning and propose a multi-modal contrastive loss to align the visual and language features into a unified semantic space. Second, a modality-adaptive box head is proposed, which makes full use of the target reference to mine ever-changing scenario features dynamically from video contexts and distinguish the target in a contrastive way, enabling robust performance in different reference settings. Extensive experimental results demonstrate that UVLTrack achieves promising performance on seven visual tracking datasets, three vision-language tracking datasets, and three visual grounding datasets. Codes and models will be open-sourced at https://github.com/OpenSpaceAI/UVLTrack.

</details>

---

## 24. Bridging the Gap between 2D and 3D Visual Question Answering: A Fusion Approach for 3D VQA

- [ ] Bridging the Gap between 2D and 3D Visual Question Answering: A Fusion Approach for 3D VQA | https://ojs.aaai.org/index.php/AAAI/article/view/28222

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28222

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In 3D Visual Question Answering (3D VQA), the scarcity of fully annotated data and limited visual content diversity hampers the generalization to novel scenes and 3D concepts (e.g., only around 800 scenes are utilized in ScanQA and SQA dataset). Current approaches resort supplement 3D reasoning with 2D information. However, these methods face challenges: either they use top-down 2D views that introduce overly complex and sometimes question-irrelevant visual clues, or they rely on globally aggregated scene/image-level representations from 2D VLMs, losing the fine-grained vision-language correlations. To overcome these limitations, our approach utilizes question-conditional 2D view selection procedure, pinpointing semantically relevant 2D inputs for crucial visual clues. We then integrate this 2D knowledge into the 3D-VQA system via a two-branch Transformer structure. This structure, featuring a Twin-Transformer design, compactly combines 2D and 3D modalities and captures fine-grained correlations between modalities, allowing them mutually augmenting each other. Integrating proposed mechanisms above, we present BridgeQA, that offers a fresh perspective on multi-modal transformer-based architectures for 3D-VQA. Experiments validate that BridgeQA achieves state-of-the-art on 3D-VQA datasets and significantly outperforms existing solutions. Code is available at https://github.com/matthewdm0816/BridgeQA.

</details>

---

## 25. Data Adaptive Traceback for Vision-Language Foundation Models in Image Classification

- [ ] Data Adaptive Traceback for Vision-Language Foundation Models in Image Classification | https://ojs.aaai.org/index.php/AAAI/article/view/28249

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28249

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language foundation models have been incredibly successful in a wide range of downstream computer vision tasks using adaptation methods. However, due to the high cost of obtaining pre-training datasets, pairs with weak image-text correlation in the data exist in large numbers. We call them weak-paired samples. Due to the limitations of these weak-paired samples, the pre-training model are unable to mine all the knowledge from pre-training data. The existing adaptation methods do not consider the missing knowledge, which may lead to crucial task-related knowledge for the downstream tasks being ignored. To address this issue, we propose a new adaptation framework called Data Adaptive Traceback (DAT). Specifically, we utilize a zero-shot-based method to extract the most downstream task-related subset of the pre-training data to enable the downstream tasks. Furthermore, we adopt a pseudo-label-based semi-supervised technique to reuse the pre-training images and a vision-language contrastive learning method to address the confirmation bias issue in semi-supervised learning. We conduct extensive experiments that show our proposed DAT approach meaningfully improves various benchmark datasets’ performance over traditional adaptation methods by simply.

</details>

---

## 26. Mining Fine-Grained Image-Text Alignment for Zero-Shot Captioning via Text-Only Training

- [ ] Mining Fine-Grained Image-Text Alignment for Zero-Shot Captioning via Text-Only Training | https://ojs.aaai.org/index.php/AAAI/article/view/28260

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28260

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image captioning aims at generating descriptive and meaningful textual descriptions of images, enabling a broad range of vision-language applications. Prior works have demonstrated that harnessing the power of Contrastive Image Language Pre-training (CLIP) offers a promising approach to achieving zero-shot captioning, eliminating the need for expensive caption annotations. However, the widely observed modality gap in the latent space of CLIP harms the performance of zero-shot captioning by breaking the alignment between paired image-text features. To address this issue, we conduct an analysis on the CLIP latent space which leads to two findings. Firstly, we observe that the CLIP's visual feature of image subregions can achieve closer proximity to the paired caption due to the inherent information loss in text descriptions. In addition, we show that the modality gap between a paired image-text can be empirically modeled as a zero-mean Gaussian distribution. Motivated by the findings, we propose a novel zero-shot image captioning framework with text-only training to reduce the modality gap. In particular, we introduce a subregion feature aggregation to leverage local region information, which produces a compact visual representation for matching text representation. Moreover, we incorporate a noise injection and CLIP reranking strategy to boost captioning performance. We also extend our framework to build a zero-shot VQA pipeline, demonstrating its generality. Through extensive experiments on common captioning and VQA datasets such as MSCOCO, Flickr30k and VQAV2, we show that our method achieves remarkable performance improvements. Code is available at https://github.com/Artanic30/MacCap.

</details>

---

## 27. GroundVLP: Harnessing Zero-Shot Visual Grounding from Vision-Language Pre-training and Open-Vocabulary Object Detection

- [ ] GroundVLP: Harnessing Zero-Shot Visual Grounding from Vision-Language Pre-training and Open-Vocabulary Object Detection | https://ojs.aaai.org/index.php/AAAI/article/view/28278

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28278

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual grounding, a crucial vision-language task involving the understanding of the visual context based on the query expression, necessitates the model to capture the interactions between objects, as well as various spatial and attribute information. However, the annotation data of visual grounding task is limited due to its time-consuming and labor-intensive annotation process, resulting in the trained models being constrained from generalizing its capability to a broader domain. To address this challenge, we propose GroundVLP, a simple yet effective zero-shot method that harnesses visual grounding ability from the existing models trained from image-text pairs and pure object detection data, both of which are more conveniently obtainable and offer a broader domain compared to visual grounding annotation data. GroundVLP proposes a fusion mechanism that combines the heatmap from GradCAM and the object proposals of open-vocabulary detectors. We demonstrate that the proposed method significantly outperforms other zero-shot methods on RefCOCO/+/g datasets, surpassing prior zero-shot state-of-the-art by approximately 28% on the test split of RefCOCO and RefCOCO+. Furthermore, GroundVLP performs comparably to or even better than some non-VLP-based supervised models on the Flickr30k entities dataset. Our code is available at https://github.com/om-ai-lab/GroundVLP.

</details>

---

## 28. PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology

- [ ] PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology | https://ojs.aaai.org/index.php/AAAI/article/view/28308

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28308

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As advances in large language models (LLMs) and multimodal techniques continue to mature, the development of general-purpose multimodal large language models (MLLMs) has surged, offering significant applications in interpreting natural images. However, the field of pathology has largely remained untapped, particularly in gathering high-quality data and designing comprehensive model frameworks. To bridge the gap in pathology MLLMs, we present PathAsst, a multimodal generative foundation AI assistant to revolutionize diagnostic and predictive analytics in pathology. The development of PathAsst involves three pivotal steps:  data acquisition, CLIP model adaptation, and the training of PathAsst's multimodal generative capabilities. Firstly, we collect over 207K high-quality pathology image-text pairs from authoritative sources. Leveraging the advanced power of ChatGPT, we generate over 180K instruction-following samples. Furthermore, we devise additional instruction-following data specifically tailored for invoking eight pathology-specific sub-models we prepared, allowing the PathAsst to effectively collaborate with these models, enhancing its diagnostic ability. Secondly, by leveraging the collected data, we construct PathCLIP, a pathology-dedicated CLIP, to enhance PathAsst's capabilities in interpreting pathology images. Finally, we integrate PathCLIP with the Vicuna-13b and utilize pathology-specific instruction-tuning data to enhance the multimodal generation capacity of PathAsst and bolster its synergistic interactions with sub-models. The experimental results of PathAsst show the potential of harnessing AI-powered generative foundation model to improve pathology diagnosis and treatment processes. We open-source our dataset, as well as a comprehensive toolkit for extensive pathology data collection and preprocessing at https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology.

</details>

---

## 29. Compound Text-Guided Prompt Tuning via Image-Adaptive Cues

- [ ] Compound Text-Guided Prompt Tuning via Image-Adaptive Cues | https://ojs.aaai.org/index.php/AAAI/article/view/28311

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28311

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable generalization capabilities to downstream tasks. However, existing prompt tuning based frameworks need to parallelize learnable textual inputs for all categories, suffering from massive GPU memory consumption when there is a large number of categories in the target dataset. Moreover, previous works require to include category names within prompts, exhibiting subpar performance when dealing with ambiguous category names. To address these shortcomings, we propose Compound Text-Guided Prompt Tuning (TGP-T) that significantly reduces resource demand while achieving superior performance. We introduce text supervision to the optimization of prompts, which enables two benefits: 1) releasing the model reliance on the pre-defined category names during inference, thereby enabling more flexible prompt generation; 2) reducing the number of inputs to the text encoder, which decreases GPU memory consumption significantly. Specifically, we found that compound text supervisions, i.e., category-wise and content-wise, is highly effective, since they provide inter-class separability and capture intra-class variations, respectively. Moreover, we condition the prompt generation on visual features through a module called Bonder, which facilitates the alignment between prompts and visual features. Extensive experiments on few-shot recognition and domain generalization demonstrate that TGP-T achieves superior performance with consistently lower training costs. It reduces GPU memory usage by 93% and attains a 2.5% performance gain on 16-shot ImageNet. The code is available at https://github.com/EricTan7/TGP-T.

</details>

---

## 30. Semantic-Aware Data Augmentation for Text-to-Image Synthesis

- [ ] Semantic-Aware Data Augmentation for Text-to-Image Synthesis | https://ojs.aaai.org/index.php/AAAI/article/view/28315

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28315

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Data augmentation has been recently leveraged as an effective regularizer in various vision-language deep neural networks. However, in text-to-image synthesis (T2Isyn), current augmentation wisdom still suffers from the semantic mismatch between augmented paired data. Even worse, semantic collapse may occur when generated images are less semantically constrained. In this paper, we develop a novel Semantic-aware Data Augmentation (SADA) framework dedicated to T2Isyn. In particular, we propose to augment texts in the semantic space via an Implicit Textual Semantic Preserving Augmentation, in conjunction with a specifically designed Image Semantic Regularization Loss as Generated Image Semantic Conservation, to cope well with semantic mismatch and collapse. As one major contribution, we theoretically show that  Implicit Textual Semantic Preserving Augmentation can certify better text-image consistency while Image Semantic Regularization Loss regularizing the semantics of generated images would avoid semantic collapse and enhance image quality. Extensive experiments validate that SADA enhances text-image consistency and improves image quality significantly in T2Isyn models across various backbones. Especially, incorporating SADA during the tuning process of Stable Diffusion models also yields performance improvements.

</details>

---

## 31. VIGC: Visual Instruction Generation and Correction

- [ ] VIGC: Visual Instruction Generation and Correction | https://ojs.aaai.org/index.php/AAAI/article/view/28338

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28338

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The integration of visual encoders and large language models (LLMs) has driven recent progress in multimodal large language models (MLLMs). However, the scarcity of high-quality instruction-tuning data for vision-language tasks remains a challenge. The current leading paradigm, such as LLaVA, relies on language-only GPT-4 to generate data, which requires pre-annotated image captions and detection bounding boxes, suffering from understanding image details. A practical solution to this problem would be to utilize the available multimodal large language models to generate instruction data for vision-language tasks. However, it's worth noting that the currently accessible MLLMs are not as powerful as their LLM counterparts, as they tend to produce inadequate responses and generate false information. As a solution for addressing the current issue, this paper proposes the Visual Instruction Generation and Correction (VIGC) framework that enables multimodal large language models to generate instruction-tuning data and progressively enhance its quality on-the-fly. Specifically, Visual Instruction Generation (VIG) guides the vision-language model to generate diverse instruction-tuning data. To ensure generation quality, Visual Instruction Correction (VIC) adopts an iterative update mechanism to correct any inaccuracies in data produced by VIG, effectively reducing the risk of hallucination. Leveraging the diverse, high-quality data generated by VIGC, we finetune mainstream models and validate data quality based on various evaluations. Experimental results demonstrate that VIGC not only compensates for the shortcomings of language-only data generation methods, but also effectively enhances the benchmark performance. The models, datasets, and code are available at https://opendatalab.github.io/VIGC

</details>

---

## 32. Learning to Learn Better Visual Prompts

- [ ] Learning to Learn Better Visual Prompts | https://ojs.aaai.org/index.php/AAAI/article/view/28343

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28343

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning provides a low-cost way of adapting vision-language models (VLMs) for various downstream vision tasks without requiring updating the huge pre-trained parameters. Dispensing with the conventional manual crafting of prompts, the recent prompt tuning method of Context Optimization (CoOp) introduces adaptable vectors as text prompts. Nevertheless, several previous works point out that the CoOp-based approaches are easy to overfit to the base classes and hard to generalize to novel classes. In this paper, we reckon that the prompt tuning works well only in the base classes because of the limited capacity of the adaptable vectors. The scale of the pre-trained model is hundreds times the scale of the adaptable vector, thus the learned vector has a very limited ability to absorb the knowledge of novel classes. To minimize this excessive overfitting of textual knowledge on the base class, we view prompt tuning as learning to learn (LoL) and learn the prompt in the way of meta-learning, the training manner of dividing the base classes into many different subclasses could fully exert the limited capacity of prompt tuning and thus transfer it power to recognize the novel classes.  To be specific, we initially perform fine-tuning on the base class based on the CoOp method for pre-trained CLIP. Subsequently, predicated on the fine-tuned CLIP model, we carry out further fine-tuning in an N-way K-shot manner from the perspective of meta-learning on the base classes. We finally apply the learned textual vector and VLM for unseen classes.Extensive experiments on benchmark datasets validate the efficacy of our meta-learning-informed prompt tuning, affirming its role as a robust optimization strategy for VLMs.

</details>

---

## 33. ViLT-CLIP: Video and Language Tuning CLIP with Multimodal Prompt Learning and Scenario-Guided Optimization

- [ ] ViLT-CLIP: Video and Language Tuning CLIP with Multimodal Prompt Learning and Scenario-Guided Optimization | https://ojs.aaai.org/index.php/AAAI/article/view/28347

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28347

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language(V-L) models such as CLIP have demonstrated impressive Zero-Shot performance in many downstream tasks. Since adopting contrastive video-text pairs methods like CLIP to video tasks is limited by its high cost and scale, recent approaches focus on efficiently transferring the image-based CLIP to the video domain. A major finding is that fine-tuning the pre-trained model to achieve strong fully supervised performance leads to low zero shot, few shot, and base to novel generalization. Instead, freezing the backbone network to maintain generalization ability weakens fully supervised performance. Otherwise, no single prompt tuning branch consistently performs optimally. In this work, we proposed a multimodal prompt learning scheme that balances supervised and generalized performance. Our prompting approach contains three sections: 1) Independent prompt on both the vision and text branches to learn the language and visual contexts. 2) Inter-modal prompt mapping to ensure mutual synergy. 3) Reducing the discrepancy between the hand-crafted prompt (a video of a person doing [CLS]) and the learnable prompt, to alleviate the forgetting about essential video scenarios. Extensive validation of fully supervised, zero-shot, few-shot, base-to-novel generalization settings for video recognition indicates that the proposed approach achieves competitive performance with less commute cost.

</details>

---

## 34. A Multimodal, Multi-Task Adapting Framework for Video Action Recognition

- [ ] A Multimodal, Multi-Task Adapting Framework for Video Action Recognition | https://ojs.aaai.org/index.php/AAAI/article/view/28361

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28361

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, the rise of large-scale vision-language pretrained models like CLIP, coupled with the technology of Parameter-Efficient FineTuning (PEFT), has captured substantial attraction in video action recognition. Nevertheless, prevailing approaches tend to prioritize strong supervised performance at the expense of compromising the models' generalization capabilities during transfer. In this paper, we introduce a novel Multimodal, Multi-task CLIP adapting framework named M2-CLIP to address these challenges, preserving both high supervised performance and robust transferability.
Firstly, to enhance the individual modality architectures, we introduce multimodal adapters to both the visual and text branches. Specifically, we design a novel visual TED-Adapter, that performs global Temporal Enhancement and local temporal Difference modeling to improve the temporal representation capabilities of the visual encoder. Moreover, we adopt text encoder adapters to strengthen the learning of semantic label information.
Secondly, we design a multi-task decoder with a rich set of supervisory signals, including the original contrastive learning head, a cross-modal classification head, a cross-modal masked language modeling head, and a visual classification head. This multi-task decoder adeptly satisfies the need for strong supervised performance within a multimodal framework.
Experimental results validate the efficacy of our approach, demonstrating exceptional performance in supervised learning while maintaining strong generalization in zero-shot scenarios.

</details>

---

## 35. Learning Hierarchical Prompt with Structured Linguistic Knowledge for Vision-Language Models

- [ ] Learning Hierarchical Prompt with Structured Linguistic Knowledge for Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/28387

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28387

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has become a prevalent strategy for adapting vision-language foundation models to downstream tasks. As large language models (LLMs) have emerged, recent studies have explored the use of category-related descriptions as input to enhance prompt effectiveness. Nevertheless, conventional descriptions fall short of structured information that effectively represents the interconnections among entities or attributes linked to a particular category. To address this limitation and prioritize harnessing structured knowledge, this paper advocates for leveraging LLMs to build a graph for each description to model the entities and attributes describing the category, as well as their correlations. Preexisting prompt tuning methods exhibit inadequacies in managing this structured knowledge. Consequently, we propose a novel approach called Hierarchical Prompt Tuning (HPT), which enables simultaneous modeling of both structured and conventional linguistic knowledge. Specifically, we introduce a relationship-guided attention module to capture pair-wise associations among entities and attributes for low-level prompt learning. In addition, by incorporating high-level and global-level prompts modeling overall semantics, the proposed hierarchical structure forges cross-level interlinks and empowers the model to handle more complex and long-term relationships. Extensive experiments demonstrate that our HPT shows strong effectiveness and generalizes much better than existing SOTA methods. Our code is available at https://github.com/Vill-Lab/2024-AAAI-HPT.

</details>

---

## 36. SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing

- [ ] SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing | https://ojs.aaai.org/index.php/AAAI/article/view/28393

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28393

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Remote sensing imagery, despite its broad applications in helping achieve Sustainable Development Goals and tackle climate change, has not yet benefited from the recent advancements of versatile, task-agnostic vision language models (VLMs). A key reason is that the large-scale, semantically diverse image-text dataset required for developing VLMs is still absent for remote sensing images. Unlike natural images, remote sensing images and their associated text descriptions cannot be efficiently collected from the public Internet at scale. In this work, we bridge this gap by using geo-coordinates to automatically connect open, unlabeled remote sensing images with rich semantics covered in OpenStreetMap, and thus construct SkyScript, a comprehensive vision-language dataset for remote sensing images, comprising 2.6 million image-text pairs covering 29K distinct semantic tags. 
With continual pre-training on this dataset, we obtain a VLM that surpasses baseline models with a 6.2% average accuracy gain in zero-shot scene classification across seven benchmark datasets. It also demonstrates the ability of zero-shot transfer for fine-grained object attribute classification and cross-modal retrieval. We hope this dataset can support the advancement of VLMs for various multi-modal tasks in remote sensing, such as open-vocabulary classification, retrieval, captioning, and text-to-image synthesis.

</details>

---

## 37. Image as a Language: Revisiting Scene Text Recognition via Balanced, Unified and Synchronized Vision-Language Reasoning Network

- [ ] Image as a Language: Revisiting Scene Text Recognition via Balanced, Unified and Synchronized Vision-Language Reasoning Network | https://ojs.aaai.org/index.php/AAAI/article/view/28402

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28402

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Scene text recognition is inherently a vision-language task. However, previous works have predominantly focused either on extracting more robust visual features or designing better language modeling. How to effectively and jointly model vision and language to mitigate heavy reliance on a single modality remains a problem. In this paper, aiming to enhance vision-language reasoning in scene text recognition, we present a balanced, unified and synchronized vision-language reasoning network (BUSNet). Firstly, revisiting the image as a language by balanced concatenation along length dimension alleviates the issue of over-reliance on vision or language. Secondly, BUSNet learns an ensemble of unified external and internal vision-language model with shared weight by masked modality modeling (MMM). Thirdly, a novel vision-language reasoning module (VLRM) with synchronized vision-language decoding capacity is proposed. Additionally, BUSNet achieves improved performance through iterative reasoning, which utilizes the vision-language prediction as a new language input. Extensive experiments indicate that BUSNet achieves state-of-the-art performance on several mainstream benchmark datasets and more challenge datasets for both synthetic and real training data compared to recent outstanding methods. Code and dataset will be available at https://github.com/jjwei66/BUSNet.

</details>

---

## 38. p-Laplacian Adaptation for Generative Pre-trained Vision-Language Models

- [ ] p-Laplacian Adaptation for Generative Pre-trained Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/28415

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28415

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language models (VLMs) pre-trained on large corpora have demonstrated notable success across a range of downstream tasks. In light of the rapidly increasing size of pre-trained VLMs, parameter-efficient transfer learning (PETL) has garnered attention as a viable alternative to full fine-tuning. One such approach is the adapter, which introduces a few trainable parameters into the pre-trained models while preserving the original parameters during adaptation.
In this paper, we present a novel modeling framework that recasts adapter tuning after attention as a graph message passing process on attention graphs, where the projected query and value features and attention matrix constitute the node features and the graph adjacency matrix, respectively. Within this framework, tuning adapters in VLMs necessitates handling heterophilic graphs, owing to the disparity between the projected query and value space.
To address this challenge, we propose a new adapter architecture, p-adapter, which employs p-Laplacian message passing in Graph Neural Networks (GNNs). Specifically, the attention weights are re-normalized based on the features, and the features are then aggregated using the calibrated attention matrix, enabling the dynamic exploitation of information with varying frequencies in the heterophilic attention graphs.
We conduct extensive experiments on different pre-trained VLMs and multi-modal tasks, including visual question answering, visual entailment, and image captioning. The experimental results validate our method's significant superiority over other PETL methods. Our code is available at https://github.com/wuhy68/p-Adapter/.

</details>

---

## 39. Toward Open-Set Human Object Interaction Detection

- [ ] Toward Open-Set Human Object Interaction Detection | https://ojs.aaai.org/index.php/AAAI/article/view/28422

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28422

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This work is oriented toward the task of open-set Human Object Interaction (HOI) detection. The challenge lies in identifying completely new, out-of-domain relationships, as opposed to in-domain ones which have seen improvements in zero-shot HOI detection. To address this challenge, we introduce a simple Disentangled HOI Detection (DHD) model for detecting novel relationships by integrating an open-set object detector with a Visual Language Model (VLM). We utilize a disentangled image-text contrastive learning metric for training and connect the bottom-up visual features to text embeddings through lightweight unary and pair-wise adapters. Our model can benefit from the open-set object detector and the VLM to detect novel action categories and combine actions with novel object categories. We further present the VG-HOI dataset, a comprehensive benchmark with over 17k HOI relationships for open-set scenarios. Experimental results show that our model can detect unknown action classes and combine unknown object classes. Furthermore, it can generalize to over 17k HOI classes while being trained on just 600 HOI classes.

</details>

---

## 40. VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection

- [ ] VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection | https://ojs.aaai.org/index.php/AAAI/article/view/28423

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28423

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent contrastive language-image pre-training (CLIP) model has shown great success in a wide range of image-level tasks, revealing remarkable ability for learning powerful visual representations with rich semantics. An open and worthwhile problem is efficiently adapting such a strong model to the video domain and designing a robust video anomaly detector. In this work, we propose VadCLIP, a new paradigm for weakly supervised video anomaly detection (WSVAD) by leveraging the frozen CLIP model directly without any pre-training and fine-tuning process. Unlike current works that directly feed extracted features into the weakly supervised classifier for frame-level binary classification, VadCLIP makes full use of fine-grained associations between vision and language on the strength of CLIP and involves dual branch. One branch simply utilizes visual features for coarse-grained binary classification, while the other fully leverages the fine-grained language-image alignment. With the benefit of dual branch, VadCLIP achieves both coarse-grained and fine-grained video anomaly detection by transferring pre-trained knowledge from CLIP to WSVAD task. We conduct extensive experiments on two commonly-used benchmarks, demonstrating that VadCLIP achieves the best performance on both coarse-grained and fine-grained WSVAD, surpassing the state-of-the-art methods by a large margin. Specifically, VadCLIP achieves 84.51% AP and 88.02% AUC on XD-Violence and UCF-Crime, respectively. Code and features are released at https://github.com/nwpu-zxr/VadCLIP.

</details>

---

## 41. CLIM: Contrastive Language-Image Mosaic for Region Representation

- [ ] CLIM: Contrastive Language-Image Mosaic for Region Representation | https://ojs.aaai.org/index.php/AAAI/article/view/28428

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28428

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Detecting objects accurately from a large or open vocabulary necessitates the vision-language alignment on region representations. However, learning such a region-text alignment by obtaining high-quality box annotations with text labels or descriptions is expensive and infeasible. In contrast, collecting image-text pairs is simpler but lacks precise object location information to associate regions with texts. In this paper, we propose a novel approach called Contrastive Language-Image Mosaic (CLIM), which leverages large-scale image-text pairs effectively for aligning region and text representations. CLIM combines multiple images into a mosaicked image and treats each image as a ‘pseudo region’. The feature of each pseudo region is extracted and trained to be similar to the corresponding text embedding while dissimilar from others by a contrastive loss, enabling the model to learn the region-text alignment without costly box annotations. As a generally
applicable approach, CLIM consistently improves different open-vocabulary object detection methods that use caption supervision. Furthermore, CLIM can effectively enhance the region representation of vision-language models, thus providing stronger backbones for open-vocabulary object detectors. Our experimental results demonstrate that CLIM improves different baseline open-vocabulary object detectors by a large margin on both OV-COCO and OV-LVIS benchmarks. The code is available at https://github.com/wusize/CLIM.

</details>

---

## 42. Embracing Language Inclusivity and Diversity in CLIP through Continual Language Learning

- [ ] Embracing Language Inclusivity and Diversity in CLIP through Continual Language Learning | https://ojs.aaai.org/index.php/AAAI/article/view/28466

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28466

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While vision-language pre-trained models (VL-PTMs) have advanced multimodal research in recent years, their mastery in a few languages like English restricts their applicability in broader communities. To this end, there is an increasing interest in developing multilingual VL models via a joint-learning setup, which, however, could be unrealistic due to expensive costs and data availability. In this work, we propose to extend VL-PTMs' language capacity by continual language learning (CLL), where a model needs to update its linguistic knowledge incrementally without suffering from catastrophic forgetting (CF). We begin our study by introducing a model dubbed CLL-CLIP, which builds upon CLIP, a prevailing VL-PTM that has acquired image-English text alignment. Specifically, CLL-CLIP contains an expandable token embedding layer to handle linguistic differences. It solely trains token embeddings to improve memory stability and is optimized under cross-modal and cross-lingual objectives to learn the alignment between images and multilingual texts. To alleviate CF raised by covariate shift and lexical overlap, we further propose a novel approach that ensures the identical distribution of all token embeddings during initialization and regularizes token embedding learning during training. We construct a CLL benchmark covering 36 languages based on MSCOCO and XM3600 datasets and then evaluate multilingual image-text retrieval performance. Extensive experiments verify the effectiveness of CLL-CLIP and show that our approach can boost CLL-CLIP, e.g., by 6.7% in text-to-image average Recall@1 on XM3600, and improve various state-of-the-art methods consistently. Our code and data are available at https://github.com/yangbang18/CLFM.

</details>

---

## 43. How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection

- [ ] How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection | https://ojs.aaai.org/index.php/AAAI/article/view/28485

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28485

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Object detection (OD) in computer vision has made significant progress in recent years, transitioning from closed-set labels to open-vocabulary detection (OVD) based on large-scale vision-language pre-training (VLP). However, current evaluation methods and datasets are limited to testing generalization over object types and referral expressions, which do not provide a systematic, fine-grained, and accurate benchmark of OVD models' abilities. In this paper, we propose a new benchmark named OVDEval, which includes 9 sub-tasks and introduces evaluations on commonsense knowledge, attribute understanding, position understanding, object relation comprehension, and more. The dataset is meticulously created to provide hard negatives that challenge models' true understanding of visual and linguistic input. Additionally, we identify a problem with the popular Average Precision (AP) metric when benchmarking models on these fine-grained label datasets and propose a new metric called Non-Maximum Suppression Average Precision (NMS-AP) to address this issue. Extensive experimental results show that existing top OVD models all fail on the new tasks except for simple object types, demonstrating the value of the proposed dataset in pinpointing the weakness of current OVD models and guiding future research. Furthermore, the proposed NMS-AP metric is verified by experiments to provide a much more truthful evaluation of OVD models, whereas traditional AP metrics yield deceptive results. Data is available at https://github.com/om-ai-lab/OVDEval

</details>

---

## 44. CLIP-Gaze: Towards General Gaze Estimation via Visual-Linguistic Model

- [ ] CLIP-Gaze: Towards General Gaze Estimation via Visual-Linguistic Model | https://ojs.aaai.org/index.php/AAAI/article/view/28496

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28496

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Gaze estimation methods often experience significant performance degradation when evaluated across different domains, due to the domain gap between the testing and training data. Existing methods try to address this issue using various domain generalization approaches, but with little success because of the limited diversity of gaze datasets, such as appearance, wearable, and image quality. To overcome these limitations, we propose a novel framework called CLIP-Gaze that utilizes a pre-trained vision-language model to leverage its transferable knowledge. Our framework is the first to leverage the vision-and-language cross-modality approach for gaze estimation task. Specifically, we extract gaze-relevant feature by pushing it away from gaze-irrelevant features which can be flexibly constructed via language descriptions. To learn more suitable prompts, we propose a personalized context optimization method for text prompt tuning. Furthermore, we utilize the relationship among gaze samples to refine the distribution of gaze-relevant features, thereby improving the generalization capability of the gaze estimation model. Extensive experiments demonstrate the excellent performance of CLIP-Gaze over existing methods on four cross-domain evaluations.

</details>

---

## 45. RadOcc: Learning Cross-Modality Occupancy Knowledge through Rendering Assisted Distillation

- [ ] RadOcc: Learning Cross-Modality Occupancy Knowledge through Rendering Assisted Distillation | https://ojs.aaai.org/index.php/AAAI/article/view/28533

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28533

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

3D occupancy prediction is an emerging task that aims to estimate the occupancy states and semantics of 3D scenes using multi-view images. However, image-based scene perception encounters significant challenges in achieving accurate prediction due to the absence of geometric priors. In this paper, we address this issue by exploring cross-modal knowledge distillation in this task, i.e., we leverage a stronger multi-modal model to guide the visual model during training. In practice, we observe that directly applying features or logits alignment, proposed and widely used in bird's-eye-view (BEV) perception, does not yield satisfactory results. To overcome this problem, we introduce RadOcc, a Rendering assisted distillation paradigm for 3D Occupancy prediction. By employing differentiable volume rendering, we generate depth and semantic maps in perspective views and propose two novel consistency criteria between the rendered outputs of teacher and student models. Specifically, the depth consistency loss aligns the termination distributions of the rendered rays, while the semantic consistency loss mimics the intra-segment similarity guided by vision foundation models (VLMs). Experimental results on the nuScenes dataset demonstrate the effectiveness of our proposed method in improving various 3D occupancy prediction approaches, e.g., our proposed methodology enhances our baseline by 2.2% in the metric of mIoU and achieves 50% in Occ3D benchmark.

</details>

---

## 46. Vision-Language Pre-training with Object Contrastive Learning for 3D Scene Understanding

- [ ] Vision-Language Pre-training with Object Contrastive Learning for 3D Scene Understanding | https://ojs.aaai.org/index.php/AAAI/article/view/28559

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28559

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In recent years, vision language pre-training frameworks have made significant progress in natural language processing and computer vision, achieving remarkable performance improvement on various downstream tasks. However, when extended to point cloud data, existing works mainly focus on building task-specific models, and fail to extract universal 3D vision-language embedding that generalize well. We carefully investigate three common tasks in semantic 3D scene understanding, and derive key insights into the development of a pre-training model. Motivated by these observations, we propose a vision-language pre-training framework 3DVLP (3D vision-language pre-training with object contrastive learning), which transfers flexibly on 3D vision-language downstream tasks. 3DVLP takes visual grounding as the proxy task and introduces Object-level IoU-guided Detection (OID) loss to obtain high-quality proposals in the scene. Moreover, we design Object-level Cross-Contrastive alignment (OCC) task and Object-level Self-Contrastive learning (OSC) task to align the objects with descriptions and distinguish different objects in the scene, respectively. Extensive experiments verify the excellent performance of 3DVLP on three 3D vision-language tasks, reflecting its superiority in semantic 3D scene understanding. Code is available at https://github.com/iridescentttt/3DVLP.

</details>

---

## 47. S3A: Towards Realistic Zero-Shot Classification via Self Structural Semantic Alignment

- [ ] S3A: Towards Realistic Zero-Shot Classification via Self Structural Semantic Alignment | https://ojs.aaai.org/index.php/AAAI/article/view/28557

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28557

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained Vision Language Models (VLMs) have proven effective for zero-shot classification. Despite the success, most traditional VLMs-based methods are restricted by the assumption of partial source supervision or ideal target vocabularies, which rarely satisfy the open-world scenario. In this paper, we aim at a more challenging setting, Realistic Zero-Shot Classification, which assumes no annotation but instead a broad vocabulary. To address the new problem, we propose the Self Structural Semantic Alignment (S3A) framework, which extracts the structural semantic information from unlabeled data while simultaneously self-learning. Our S3A framework adopts a unique Cluster-Vote-Prompt-Realign (CVPR) algorithm, which iteratively groups unlabeled data to derive structural semantics for pseudo-supervision. Our CVPR algorithm includes iterative clustering on images, voting within each cluster to identify initial class candidates from the vocabulary, generating discriminative prompts with large language models to discern confusing candidates, and realigning images and the vocabulary as structural semantic alignment. Finally, we propose to self-train the CLIP image encoder with both individual and structural semantic alignment through a teacher-student learning strategy. Our comprehensive experiments across various generic and fine-grained benchmarks demonstrate that the S3A method substantially improves over existing VLMs-based approaches, achieving a more than 15% accuracy improvement over CLIP on average. Our codes, models, and prompts are publicly released at https://github.com/sheng-eatamath/S3A.

</details>

---

## 48. Concept-Guided Prompt Learning for Generalization in Vision-Language Models

- [ ] Concept-Guided Prompt Learning for Generalization in Vision-Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/28568

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28568

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pretraining (CLIP) model has exhibited remarkable efficacy in establishing cross-modal connections between texts and images, yielding impressive
performance across a broad spectrum of downstream applications through fine-tuning. However, for generalization tasks, the current fine-tuning methods for CLIP, such as CoOp and
CoCoOp, demonstrate relatively low performance on some fine-grained datasets. We recognize the underlying reason is that these previous methods only projected global features
into the prompt, neglecting the various visual concepts, such as colors, shapes, and sizes, which are naturally transferable
across domains and play a crucial role in generalization tasks. To address this issue, in this work, we propose
Concept-Guided Prompt Learning (CPL) for vision-language models. Specifically, we leverage the well-learned knowledge
of CLIP to create a visual concept cache to enable conceptguided prompting. In order to refine the text features, we further
develop a projector that transforms multi-level visual features into text features. We observe that this concept-guided
prompt learning approach is able to achieve enhanced consistency between visual and linguistic modalities. Extensive
experimental results demonstrate that our CPL method significantly improves generalization capabilities compared to
the current state-of-the-art methods.

</details>

---

## 49. No Head Left Behind – Multi-Head Alignment Distillation for Transformers

- [ ] No Head Left Behind – Multi-Head Alignment Distillation for Transformers | https://ojs.aaai.org/index.php/AAAI/article/view/28583

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28583

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Knowledge distillation aims at reducing model size without compromising much performance. Recent work has applied it to large vision-language (VL) Transformers, and has shown that attention maps in the multi-head attention modules of vision-language Transformers contain extensive intra-modal and cross-modal co-reference relations to be distilled. The standard approach is to apply a one-to-one attention map distillation loss, i.e. the Teacher's first attention head instructs the Student's first head, the second teaches the second, and so forth, but this only works when the numbers of attention heads in the Teacher and Student are the same. To remove this constraint, we propose a new Attention Map Alignment Distillation (AMAD) method for Transformers with multi-head attention, which works for a Teacher and a Student with different numbers of attention heads. Specifically, we soft-align different heads in Teacher and Student attention maps using a cosine similarity weighting. The Teacher head contributes more to the Student heads for which it has a higher similarity weight. Each Teacher head contributes to all the Student heads by minimizing the divergence between the attention activation distributions for the soft-aligned heads. No head is left behind. This distillation approach operates like cross-attention. We experiment on distilling VL-T5 and BLIP, and apply AMAD loss on their T5, BERT, and ViT sub-modules. We show, under vision-language setting, that AMAD outperforms conventional distillation methods on VQA-2.0, COCO captioning, and Multi30K translation datasets. We further show that even without VL pre-training, the distilled VL-T5 models outperform corresponding VL pre-trained VL-T5 models that are further fine-tuned by ground-truth signals, and that fine-tuning distillation can also compensate to some degree for the absence of VL pre-training for BLIP models.

</details>

---

## 50. SEER: Backdoor Detection for Vision-Language Models through Searching Target Text and Image Trigger Jointly

- [ ] SEER: Backdoor Detection for Vision-Language Models through Searching Target Text and Image Trigger Jointly | https://ojs.aaai.org/index.php/AAAI/article/view/28611

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28611

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper proposes SEER, a novel backdoor detection algorithm for vision-language models, addressing the gap in the literature on multi-modal backdoor detection. While backdoor detection in single-modal models has been well studied, the investigation of such defenses in multi-modal models remains limited. Existing backdoor defense mechanisms cannot be directly applied to multi-modal settings due to their increased complexity and search space explosion. In this paper, we propose to detect backdoors in vision-language models by jointly searching image triggers and malicious target texts in feature space shared by vision and language modalities. Our extensive experiments demonstrate that SEER can achieve over 92% detection rate on backdoor detection in vision-language models in various settings without accessing training data or knowledge of downstream tasks.

</details>

---

## 51. FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning

- [ ] FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning | https://ojs.aaai.org/index.php/AAAI/article/view/29007

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29007

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, foundation models have exhibited remarkable advancements in multi-modal learning. These models, equipped with millions (or billions) of parameters, typically require a substantial amount of data for finetuning. However, collecting and centralizing training data from diverse sectors becomes challenging due to distinct privacy regulations. Federated Learning (FL) emerges as a promising solution, enabling multiple clients to collaboratively train neural networks without centralizing their local data. To alleviate client computation burdens and communication overheads, previous works have adapted Parameter-efficient Finetuning (PEFT) methods for FL. Hereby, only a small fraction of the model parameters are optimized and communicated during federated communications. Nevertheless, most previous works have focused on a single modality and neglected one common phenomenon, i.e., the presence of data heterogeneity across the clients. Therefore, in this work, we propose a finetuning framework tailored to heterogeneous multi-modal FL, called Federated Dual-Aadapter Teacher (FedDAT). Specifically, our approach leverages a Dual-Adapter Teacher (DAT) to address data heterogeneity by regularizing the client local updates and applying Mutual Knowledge Distillation (MKD) for an efficient knowledge transfer. FedDAT is the first approach that enables an efficient distributed finetuning of foundation models for a variety of heterogeneous Vision-Language tasks. To demonstrate its effectiveness, we conduct extensive experiments on four multi-modality FL benchmarks with different types of data heterogeneity, where FedDAT substantially outperforms the existing centralized PEFT methods adapted for FL.

</details>

---

## 52. Make Prompts Adaptable: Bayesian Modeling for Vision-Language Prompt Learning with Data-Dependent Prior

- [ ] Make Prompts Adaptable: Bayesian Modeling for Vision-Language Prompt Learning with Data-Dependent Prior | https://ojs.aaai.org/index.php/AAAI/article/view/29037

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29037

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language pre-trained (VLP) models have become the backbone for many downstream tasks, but they are utilized as frozen model without learning. Prompt learning is a method to improve the pre-trained VLP model by adding a learnable context vector to the inputs of the text encoder. In a few-shot learning scenario of the downstream task, MLE training can lead the context vector to over-fit dominant image features in the training data. This overfitting can potentially harm the generalization ability, especially in the presence of a distribution shift between the training and test dataset.  This paper presents a Bayesian-based framework of prompt tuning, which could alleviate the over-fitting issues on few-shot learning application and increase the adaptability of prompts on unobserved instances. Specifically, modeling data-dependent prior enhances the adaptability of text features for both seen and unseen image features without the trade-off of performance between them. Based on the Bayesian framework, we utilize the Wasserstein gradient flow in the estimation of our target posterior distribution, which enables our prompt to be flexible in capturing the complex modes of image features. We demonstrate the effectiveness of our method on benchmark datasets for several experiments by showing statistically significant improvements on performance compared to existing methods.

</details>

---

## 53. Continual Vision-Language Retrieval via Dynamic Knowledge Rectification

- [ ] Continual Vision-Language Retrieval via Dynamic Knowledge Rectification | https://ojs.aaai.org/index.php/AAAI/article/view/29054

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29054

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent large-scale pre-trained models like CLIP have aroused great concern in vision-language tasks. However, when required to match image-text data collected in a streaming manner, namely Continual Vision-Language Retrieval (CVRL), their performances are still limited due to the catastrophic forgetting of the learned old knowledge. To handle this issue, advanced methods are proposed to distill the affinity knowledge between images and texts from the old model to the new one for anti-forgetting. Unfortunately, existing approaches neglect the impact of incorrect affinity, which prevents the balance between the anti-forgetting of old knowledge and the acquisition of new knowledge. Therefore, we propose a novel framework called Dynamic Knowledge Rectification (DKR) that simultaneously achieves incorrect knowledge filtering and rectification. Specifically, we first filter the incorrect affinity knowledge calculated by the old model on the new data. Then, a knowledge rectification method is designed to rectify the incorrect affinities while preserving the correct ones. In particular, for the new data that can only be correctly retrieved by the new model, we rectify them with the corresponding new affinity to protect them from negative transfer. Additionally, for those that can not be retrieved by either the old or the new model, we introduce paired ground-truth labels to promote the acquisition of both old and new knowledge. Extensive experiments on several benchmark datasets demonstrate the effectiveness of our DKR and its superiority against state-of-the-art methods.

</details>

---

## 54. Relax Image-Specific Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camouflaged Objects

- [ ] Relax Image-Specific Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camouflaged Objects | https://ojs.aaai.org/index.php/AAAI/article/view/29144

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29144

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Camouflaged object detection (COD) approaches heavily rely on pixel-level annotated datasets.  Weakly-supervised COD (WSCOD) approaches use sparse annotations like scribbles or points to reduce annotation efforts, but this can lead to decreased accuracy.  The Segment Anything Model (SAM) shows remarkable segmentation ability with sparse prompts like points. However, manual prompt is not always feasible, as it may not be accessible in real-world application. Additionally, it only provides localization information instead of semantic one, which can intrinsically cause ambiguity in interpreting targets. In this work, we aim to eliminate the need for manual prompt. The key idea is to employ Cross-modal Chains of Thought Prompting (CCTP) to reason visual prompts using the semantic information given by a generic text prompt. To that end, we introduce a test-time instance-wise adaptation mechanism called Generalizable SAM (GenSAM) to automatically generate and optimize visual prompts from the generic task prompt for WSCOD. In particular, CCTP maps a single generic text prompt onto image-specific consensus foreground and background heatmaps using vision-language models,  acquiring reliable visual prompts. Moreover, to test-time adapt the visual prompts, we further propose Progressive Mask Generation (PMG) to iteratively reweight the input image, guiding the model to focus on the targeted region in a coarse-to-fine manner. Crucially, all network parameters are fixed, avoiding the need for additional training. Experiments on three benchmarks demonstrate that GenSAM outperforms point supervision approaches and achieves comparable results to scribble supervision ones, solely relying on general task descriptions. Our codes is in https://github.com/jyLin8100/GenSAM.

</details>

---

## 55. DART: Dual-Modal Adaptive Online Prompting and Knowledge Retention for Test-Time Adaptation

- [ ] DART: Dual-Modal Adaptive Online Prompting and Knowledge Retention for Test-Time Adaptation | https://ojs.aaai.org/index.php/AAAI/article/view/29320

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29320

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As an up-and-coming area, CLIP-based pre-trained vision-language models can readily facilitate downstream tasks through the zero-shot or few-shot fine-tuning manners. However, they still face critical challenges in test-time generalization due to the shifts between the training and test data distributions, hindering the further improvement of the performance. To address this crucial problem, the latest works have introduced Test-Time Adaptation (TTA) techniques to CLIP which dynamically learn text prompts using only test samples. However, their limited learning capacity due to the overlook of visual modality information, and the underutilization of knowledge in previously seen test samples result in reduced performance. In this paper, we propose a novel Dual-modal Adaptive online prompting and knowledge ReTention method called DART to overcome these challenges. To increase the learning capacity, DART captures knowledge from each test sample by learning class-specific text prompts and instance-level image prompts. Additionally, to fully leverage the knowledge from previously seen test samples, DART utilizes dual-modal knowledge retention prompts to adaptively retain the acquired knowledge, thereby enhancing the predictions on subsequent test samples. Extensive experiments on various large-scale benchmarks demonstrate the effectiveness of our proposed DART against state-of-the-art methods.

</details>

---

## 56. Leveraging Diffusion Perturbations for Measuring Fairness in Computer Vision

- [ ] Leveraging Diffusion Perturbations for Measuring Fairness in Computer Vision | https://ojs.aaai.org/index.php/AAAI/article/view/29333

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29333

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Computer vision models have been known to encode harmful biases, leading to the potentially unfair treatment of historically marginalized groups, such as people of color. However, there remains a lack of datasets balanced along demographic traits that can be used to evaluate the downstream fairness of these models. In this work, we demonstrate that diffusion models can be leveraged to create such a dataset. We first use a diffusion model to generate a large set of images depicting various occupations. Subsequently, each image is edited using inpainting to generate multiple variants, where each variant refers to a different perceived race. Using this dataset, we benchmark several vision-language models on a multi-class occupation classification task. We find that images generated with non-Caucasian labels have a significantly higher occupation misclassification rate than images generated with Caucasian labels, and that several misclassifications are suggestive of racial biases. We measure a model’s downstream fairness by computing the standard deviation in the probability of predicting the true occupation label across the different identity groups. Using this fairness metric, we find significant disparities between the evaluated vision-and-language models. We hope that our work demonstrates the potential value of diffusion methods for fairness evaluations.

</details>

---

## 57. CLIP-Guided Federated Learning on Heterogeneity and Long-Tailed Data

- [ ] CLIP-Guided Federated Learning on Heterogeneity and Long-Tailed Data | https://ojs.aaai.org/index.php/AAAI/article/view/29416

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29416

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Federated learning (FL) provides a decentralized machine learning paradigm where a server collaborates with a group of clients to learn a global model without accessing the clients' data. User heterogeneity is a significant challenge for FL, which together with the class-distribution imbalance further enhances the difficulty of FL. Great progress has been made in large vision-language models, such as Contrastive Language-Image Pre-training (CLIP), which paves a new way for image classification and object recognition. Inspired by the success of CLIP on few-shot and zero-shot learning, we use CLIP to optimize the federated learning between server and client models under its vision-language supervision. It is promising to mitigate the user heterogeneity and class-distribution balance due to the powerful cross-modality representation and rich open-vocabulary prior knowledge. In this paper, we propose the CLIP-guided FL (CLIP2FL) method on heterogeneous and long-tailed data. In CLIP2FL, the knowledge of the off-the-shelf CLIP model is transferred to the client-server models, and a bridge is built between the client and server. Specifically, for client-side learning, knowledge distillation is conducted between client models and CLIP to improve the ability of client-side feature representation. For server-side learning, in order to mitigate the heterogeneity and class-distribution imbalance, we generate federated features to retrain the server model. A prototype contrastive learning with the supervision of the text encoder of CLIP is introduced to generate federated features depending on the client-side gradients, and they are used to retrain a balanced server classifier. Extensive experimental results on several benchmarks demonstrate that CLIP2FL achieves impressive performance and effectively deals with data heterogeneity and long-tail distribution. The code is available at https://github.com/shijiangming1/CLIP2FL.

</details>

---

## 58. MmAP: Multi-Modal Alignment Prompt for Cross-Domain Multi-Task Learning

- [ ] MmAP: Multi-Modal Alignment Prompt for Cross-Domain Multi-Task Learning | https://ojs.aaai.org/index.php/AAAI/article/view/29540

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29540

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-Task Learning (MTL) is designed to train multiple correlated tasks simultaneously, thereby enhancing the performance of individual tasks. Typically, a multi-task network structure consists of a shared backbone and task-specific decoders. However, the complexity of the decoders increases with the number of tasks. To tackle this challenge, we integrate the decoder-free vision-language model CLIP, which exhibits robust zero-shot generalization capability. Recently, parameter-efficient transfer learning methods have been extensively explored with CLIP for adapting to downstream tasks, where prompt tuning showcases strong potential. Nevertheless, these methods solely fine-tune a single modality (text or visual), disrupting the modality structure of CLIP. In this paper, we first propose Multi-modal Alignment Prompt (MmAP) for CLIP, which aligns text and visual modalities during fine-tuning process. Building upon MmAP, we develop an innovative multi-task prompt learning framework. On the one hand, to maximize the complementarity of tasks with high similarity, we utilize a gradient-driven task grouping method that partitions tasks into several disjoint groups and assign a group-shared MmAP to each group. On the other hand, to preserve the unique characteristics of each task, we assign an task-specific MmAP to each task. Comprehensive experiments on two large multi-task learning datasets demonstrate that our method achieves significant performance improvements compared to full fine-tuning while only utilizing approximately ~ 0.09% of trainable parameters.

</details>

---

## 59. Beyond Grounding: Extracting Fine-Grained Event Hierarchies across Modalities

- [ ] Beyond Grounding: Extracting Fine-Grained Event Hierarchies across Modalities | https://ojs.aaai.org/index.php/AAAI/article/view/29718

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29718

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Events describe happenings in our world that are of importance. Naturally, understanding events mentioned in multimedia content and how they are related forms an important way of comprehending our world. Existing literature can infer if events across textual and visual (video) domains are identical (via grounding) and thus, on the same semantic level. However, grounding fails to capture the intricate cross-event relations that exist due to the same events being referred to on many semantic levels. For example, the abstract event of "war'' manifests at a lower semantic level through subevents "tanks firing'' (in video) and airplane "shot'' (in text), leading to a hierarchical, multimodal relationship between the events.


In this paper, we propose the task of extracting event hierarchies from multimodal (video and text) data to capture how the same event manifests itself in different modalities at different semantic levels. This reveals the structure of events and is critical to understanding them. To support research on this task, we introduce the Multimodal Hierarchical Events (MultiHiEve) dataset. Unlike prior video-language datasets, MultiHiEve is composed of news video-article pairs, which makes it rich in event hierarchies. We densely annotate a part of the dataset to construct the test benchmark. We show the limitations of state-of-the-art unimodal and multimodal baselines on this task. Further, we address these limitations via a new weakly supervised model, leveraging only unannotated video-article pairs from MultiHiEve. We perform a thorough evaluation of our proposed method which demonstrates improved performance on this task and highlight opportunities for future research. Data: https://github.com/hayyubi/multihieve

</details>

---

## 60. Visual Instruction Tuning with Polite Flamingo

- [ ] Visual Instruction Tuning with Polite Flamingo | https://ojs.aaai.org/index.php/AAAI/article/view/29727

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29727

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent research has demonstrated that the multi-task fine-tuning of multi-modal Large Language Models (LLMs) using an assortment of annotated downstream vision-language datasets significantly enhances their performance. Yet, during this process, a side effect, which we termed as the "multi-modal alignment tax", surfaces. This side effect negatively impacts the model's ability to format responses appropriately - for instance, its "politeness" - due to the overly succinct and unformatted nature of raw annotations, resulting in reduced human preference. In this paper, we introduce Polite Flamingo, a multi-modal response rewriter that transforms raw annotations into a more appealing, "polite" format. Polite Flamingo is trained to reconstruct high-quality responses from their automatically distorted counterparts and is subsequently applied to a vast array of vision-language datasets for response rewriting. After rigorous filtering, we generate the PF-1M dataset and further validate its value by fine-tuning a multi-modal LLM with it. Combined with novel methodologies including U-shaped multi-stage tuning and multi-turn augmentation, the resulting model, Clever Flamingo, demonstrates its advantages in both multi-modal understanding and response politeness according to automated and human evaluations. Code and dataset are available at https://github.com/ChenDelong1999/polite-flamingo

</details>

---

## 61. CoPL: Contextual Prompt Learning for Vision-Language Understanding

- [ ] CoPL: Contextual Prompt Learning for Vision-Language Understanding | https://ojs.aaai.org/index.php/AAAI/article/view/29766

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29766

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in multimodal learning has resulted in powerful vision-language models, whose representations are generalizable across a variety of downstream tasks. Recently, their generalization ability has been further extended by incorporating trainable prompts, borrowed from the natural language processing literature. While such prompt learning techniques have shown impressive results, we identify that these prompts are trained based on global image features which limits itself in two aspects: First, by using global features, these prompts could be focusing less on the discriminative foreground image, resulting in poor generalization to various out-of-distribution test cases. Second, existing work weights all prompts equally whereas intuitively, prompts should be reweighed according to the semantics of the image. We address these as part of our proposed Contextual Prompt Learning (CoPL) framework, capable of aligning the prompts to
the localized features of the image. Our key innovations over earlier works include using local image features as part of the prompt learning process, and more crucially, learning to weight these prompts based on local features that are appropriate for the task at hand. This gives us dynamic prompts that are both aligned to local image features as well as aware of local contextual relationships. Our extensive set of experiments on a variety of standard and few-shot datasets show that our method produces substantially improved performance when compared to the current state of the art methods. We also demonstrate both few-shot and out-of-distribution performance to establish the utility of learning dynamic prompts that are aligned to local image features.

</details>

---

## 62. Detecting and Preventing Hallucinations in Large Vision Language Models

- [ ] Detecting and Preventing Hallucinations in Large Vision Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/29771

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29771

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuned Large Vision Language Models (LVLMs) have significantly advanced in generalizing across a diverse set of multi-modal tasks, especially for Visual Question Answering (VQA). However, generating detailed responses that are visually grounded is still a challenging task for these models. We find that even the current state-of-the-art LVLMs (InstructBLIP) still contain a staggering 30 percent of the hallucinatory text in the form of non-existent objects, unfaithful descriptions, and inaccurate relationships. To address this, we introduce M-HalDetect, a Multimodal Hallucination Detection Dataset that can be used to train and benchmark models for hallucination detection and prevention. M-HalDetect consists of 16k fine-grained annotations on VQA examples, making it the first comprehensive multi-modal hallucination detection dataset for detailed image descriptions. Unlike previous work that only consider object hallucination, we additionally annotate both entity descriptions and relationships that are unfaithful. To demonstrate the potential of this dataset for hallucination prevention, we optimize InstructBLIP through our novel Fine-grained Direct Preference Optimization (FDPO). We also train fine-grained multi-modal reward models from InstructBLIP and evaluate their effectiveness with best-of-n rejection sampling (RS). We perform human evaluation on both FDPO and rejection sampling, and find that they reduce hallucination rates in InstructBLIP by 41% and 55% respectively. We also find that our reward model generalizes to other multi-modal models, reducing hallucinations in LLaVA and mPLUG-OWL by 15% and 57% respectively, and has strong correlation with human evaluated accuracy scores. The dataset is available at https://github.com/hendryx-scale/mhal-detect.

</details>

---

## 63. READ-PVLA: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling

- [ ] READ-PVLA: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling | https://ojs.aaai.org/index.php/AAAI/article/view/29847

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29847

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fully fine-tuning pretrained large-scale transformer models has become a popular paradigm for video-language modeling tasks, such as temporal language grounding and video-language summarization. With a growing number of tasks and limited training data, such full fine-tuning approach leads to costly model storage and unstable training. To overcome these shortcomings, we introduce lightweight adapters to the pre-trained model and only update them at fine-tuning time. However, existing adapters fail to capture intrinsic temporal relations among video frames or textual words. Moreover, they neglect the preservation of critical task-related information that flows from the raw video-language input into the adapter’s low-dimensional space. To address these issues, we first propose a novel REcurrent ADapter (READ) that employs recurrent computation to enable temporal modeling capability. Second, we propose Partial Video-Language Alignment (PVLA) objective via the use of partial optimal transport to maintain task-related information flowing into our READ modules. We validate our READ-PVLA framework through extensive experiments where READ-PVLA significantly outperforms all existing fine-tuning strategies on multiple low-resource temporal language grounding and video-language summarization benchmarks.

</details>

---

## 64. InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions

- [ ] InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions | https://ojs.aaai.org/index.php/AAAI/article/view/29874

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29874

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We study the problem of completing various visual document understanding (VDU) tasks, e.g., question answering and information extraction, on real-world documents through human-written instructions. To this end, we propose InstructDoc, the first large-scale collection of 30 publicly available VDU datasets, each with diverse instructions in a unified format, which covers a wide range of 12 tasks and includes open document types/formats. Furthermore, to enhance the generalization performance on VDU tasks, we design a new instruction-based document reading and understanding model, InstructDr, that connects document images, image encoders, and large language models (LLMs) through a trainable bridging module. Experiments demonstrate that InstructDr can effectively adapt to new VDU datasets, tasks, and domains via given instructions and outperforms existing multimodal LLMs and ChatGPT without specific training.

</details>

---

## 65. Tackling Vision Language Tasks through Learning Inner Monologues

- [ ] Tackling Vision Language Tasks through Learning Inner Monologues | https://ojs.aaai.org/index.php/AAAI/article/view/29905

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/29905

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual language tasks such as Visual Question Answering (VQA) or Visual Entailment (VE) require AI models to comprehend and reason with both visual and textual content. Driven by the power of Large Language Models (LLMs), two prominent methods have emerged: (1) the hybrid integration between LLMs and Vision-Language Models (VLMs), where visual inputs are firstly converted into language descriptions by VLMs, serving as inputs for LLMs to generate final answer(s); (2) visual feature alignment in language space, where visual inputs are encoded as embeddings and projected to LLMs' language space via further supervised fine-tuning. The first approach provides light training costs and interpretability but is hard to be optimized in an end-to-end fashion. The second approach presents decent performance, but feature alignment usually requires large amounts of training data and lacks interpretability. 
To tackle this dilemma, we propose a novel approach, Inner Monologue Multi-Modal Optimization (IMMO), to solve complex vision language problems by simulating Inner Monologue, a cognitive process in which an individual engages in silent verbal communication with themselves. More specifically, we enable LLMs and VLMs to interact through natural language conversation (i.e., Inner Monologue) and propose to use a two-stage training process to learn how to do Inner Monologue (self-asking questions and answering questions). IMMO is evaluated on two popular tasks and achieves competitive performance with less training data when compared with state-of-the-art models while concurrently keeping the interpretability. The results suggest that by emulating the cognitive phenomenon of internal dialogue, our approach can enhance reasoning and explanation abilities, contributing to the more effective fusion of vision and language models. More importantly, instead of using predefined human-crafted monologues, IMMO learns this process within the deep learning models, broadening its potential applications across various AI challenges beyond vision and language tasks.

</details>

---

## 66. Visual Adversarial Examples Jailbreak Aligned Large Language Models

- [ ] Visual Adversarial Examples Jailbreak Aligned Large Language Models | https://ojs.aaai.org/index.php/AAAI/article/view/30150

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/30150

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Warning: this paper contains data, prompts, and model outputs that are offensive in nature.

Recently, there has been a surge of interest in integrating vision into Large Language Models (LLMs), exemplified by Visual Language Models (VLMs) such as Flamingo and GPT-4. This paper sheds light on the security and safety implications of this trend. First, we underscore that the continuous and high-dimensional nature of the visual input makes it a weak link against adversarial attacks, representing an expanded attack surface of vision-integrated LLMs. Second, we highlight that the versatility of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, extending the implications of security failures beyond mere misclassification. As an illustration, we present a case study in which we exploit visual adversarial examples to circumvent the safety guardrail of aligned LLMs with integrated vision. Intriguingly, we discover that a single visual adversarial example can universally jailbreak an aligned LLM, compelling it to heed a wide range of harmful instructions (that it otherwise would not) and generate harmful content that transcends the narrow scope of a `few-shot' derogatory corpus initially employed to optimize the adversarial example. Our study underscores the escalating adversarial risks associated with the pursuit of multimodality. Our findings also connect the long-studied adversarial vulnerabilities of neural networks to the nascent field of AI alignment. The presented attack suggests a fundamental adversarial challenge for AI alignment, especially in light of the emerging trend toward multimodality in frontier foundation models.

</details>

---

## 67. Adventures of Trustworthy Vision-Language Models: A Survey

- [ ] Adventures of Trustworthy Vision-Language Models: A Survey | https://ojs.aaai.org/index.php/AAAI/article/view/30275

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/30275

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, transformers have become incredibly popular in computer vision and vision-language tasks. This notable rise in their usage can be primarily attributed to the capabilities offered by attention mechanisms and the outstanding ability of transformers to adapt and apply themselves to a variety of tasks and domains. Their versatility and state-of-the-art performance have established them as indispensable tools for a wide array of applications. However, in the constantly changing landscape of machine learning, the assurance of the trustworthiness of transformers holds utmost importance. This paper conducts a thorough examination of vision-language transformers, employing three fundamental principles of responsible AI: Bias, Robustness, and Interpretability. The primary objective of this paper is to delve into the intricacies and complexities associated with the practical use of transformers, with the overarching goal of advancing our comprehension of how to enhance their reliability and accountability.

</details>

---

## 68. Multimodal Ensembling for Zero-Shot Image Classification

- [ ] Multimodal Ensembling for Zero-Shot Image Classification | https://ojs.aaai.org/index.php/AAAI/article/view/30551

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/30551

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Artificial intelligence has made significant progress in image classification, an essential task for machine perception to achieve human-level image understanding. Despite recent advances in vision-language fields, multimodal image classification is still challenging, particularly for the following two reasons. First, models with low capacity often suffer from underfitting and thus underperform on fine-grained image classification. Second, it is important to ensure high-quality data with rich cross-modal representations of each class, which is often difficult to generate. Here, we utilize ensemble learning to reduce the impact of these issues on pre-trained models. We aim to create a meta-model that combines the predictions of multiple open-vocabulary multimodal models trained on different data to create more robust and accurate predictions. By utilizing ensemble learning and multimodal machine learning, we will achieve higher prediction accuracies without any additional training or fine-tuning, meaning that this method is completely zero-shot.

</details>

---

## 69. Vision-Language Models for Robot Success Detection

- [ ] Vision-Language Models for Robot Success Detection | https://ojs.aaai.org/index.php/AAAI/article/view/30552

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/30552

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we use Vision-Language Models (VLMs) as a binary success detector given a robot observation and task description, formulated as a Visual Question Answering (VQA) problem. We fine-tune the open-source MiniGPT-4 VLM to detect success on robot trajectories from the Berkeley Bridge and Berkeley AUTOLab UR5 datasets. We find that while a handful of test distribution trajectories can train an accurate detector, transferring learning between different environments is challenging due to distribution shift. In addition, while our VLM is robust to language variations, it is less robust to visual variations. In the future, more powerful VLMs such as Gemini and GPT-4 have the potential to be more accurate and robust success detectors, and success detectors can provide a sparse binary reward to improve existing policies.

</details>

---

## 70. AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head

- [ ] AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head | https://ojs.aaai.org/index.php/AAAI/article/view/30570

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/30570

- **Conference**: AAAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have exhibited remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. Despite the recent success, current LLMs are not capable of processing complex audio information or conducting spoken conversations (like Siri or Alexa). In this work, we propose a multi-modal AI system named AudioGPT, which complements LLMs (i.e., ChatGPT) with 1) foundation models to process complex audio information and solve numerous understanding and generation tasks; and 2) the input/output interface (ASR, TTS) to support spoken dialogue. With an increasing demand to evaluate multi-modal LLMs of human intention understanding and cooperation with foundation models, we outline the principles and processes and test AudioGPT in terms of consistency, capability, and robustness. Experimental results demonstrate the capabilities of AudioGPT in solving 16 AI tasks with speech, music, sound, and talking head understanding and generation in multi-round dialogues, which empower humans to create rich and diverse audio content with unprecedented ease. Code can be found in https://github.com/AIGC-Audio/AudioGPT

</details>

---

