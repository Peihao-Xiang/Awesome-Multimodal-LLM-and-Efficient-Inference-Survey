# ICCV 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_iccv2023_papers.csv

## 1. CLIPTER: Looking at the Bigger Picture in Scene Text Recognition

- [ ] CLIPTER: Looking at the Bigger Picture in Scene Text Recognition | https://openaccess.thecvf.com/content/ICCV2023/html/Aberdam_CLIPTER_Looking_at_the_Bigger_Picture_in_Scene_Text_Recognition_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Aberdam_CLIPTER_Looking_at_the_Bigger_Picture_in_Scene_Text_Recognition_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Reading text in real-world scenarios often requires understanding the context surrounding it, especially when dealing with poor-quality text. However, current scene text recognizers are unaware of the bigger picture as they operate on cropped text images. In this study, we harness the representative capabilities of modern vision-language models, such as CLIP, to provide scene-level information to the crop-based recognizer. We achieve this by fusing a rich representation of the entire image, obtained from the vision-language model, with the recognizer word-level features via a gated cross-attention mechanism. This component gradually shifts to the context-enhanced representation, allowing for stable fine-tuning of a pretrained recognizer. We demonstrate the effectiveness of our model-agnostic framework, CLIPTER (CLIP TExt Recognition), on leading text recognition architectures and achieve state-of-the-art results across multiple benchmarks. Furthermore, our analysis highlights improved robustness to out-of-vocabulary words and enhanced generalization in low-data regimes.

</details>

---

## 2. VL-Match: Enhancing Vision-Language Pretraining with Token-Level and Instance-Level Matching

- [ ] VL-Match: Enhancing Vision-Language Pretraining with Token-Level and Instance-Level Matching | https://openaccess.thecvf.com/content/ICCV2023/html/Bi_VL-Match_Enhancing_Vision-Language_Pretraining_with_Token-Level_and_Instance-Level_Matching_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Bi_VL-Match_Enhancing_Vision-Language_Pretraining_with_Token-Level_and_Instance-Level_Matching_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pretraining (VLP) has significantly improved the performance of various vision-language tasks with the matching of images and texts. In this paper, we propose VL-Match, a Vision-Language framework with Enhanced Token-level and Instance-level Matching. At the token level, a Vision-Language Replaced Token Detection task is designed to boost the substantial interaction between text tokens and images, where the text encoder of VLP works as a generator to generate a corrupted text, and the multimodal encoder of VLP works as a discriminator to predict whether each text token in the corrupted text matches the image. At the instance level, in the Image-Text Matching task that judges whether an image-text pair is matched, we propose a novel bootstrapping method to generate hard negative text samples that are different from the positive ones only at the token level. In this way, we can force the network to detect fine-grained differences between images and texts. Notably, with a smaller amount of parameters, VL-Match significantly outperforms previous SOTA on all image-text retrieval tasks.

</details>

---

## 3. A Retrospect to Multi-prompt Learning across Vision and Language

- [ ] A Retrospect to Multi-prompt Learning across Vision and Language | https://openaccess.thecvf.com/content/ICCV2023/html/Chen_A_Retrospect_to_Multi-prompt_Learning_across_Vision_and_Language_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_A_Retrospect_to_Multi-prompt_Learning_across_Vision_and_Language_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The vision community is undergoing the unprecedented progress with the emergence of Vision-Language Pretraining Models (VLMs). Prompt learning plays as the holy grail of accessing VLMs since it enables their fast adaptation to downstream tasks with limited resources. Whereas existing research milling around single-prompt paradigms, rarely investigate the technical potential behind their multi-prompt learning counterparts. This paper aims to provide a principled retrospect for vision-language multi-prompt learning. We extend the recent constant modality gap phenomenon to learnable prompts and then, justify the superiority of vision-language transfer with multi-prompt augmentation, empirically and theoretically. In terms of this observation, we propose an Energy-based Multi-prompt Learning (EMPL) to generate multiple prompt embeddings by drawing instances from an energy-based distribution, which is implicitly defined by VLMs. So our EMPL is not only parameter-efficient but also rigorously lead to the balance between in-domain and out-of-domain open-vocabulary generalization. Comprehensive experiments have been conducted to justify our claims and the excellence of EMPL.

</details>

---

## 4. Exploring Open-Vocabulary Semantic Segmentation from CLIP Vision Encoder Distillation Only

- [ ] Exploring Open-Vocabulary Semantic Segmentation from CLIP Vision Encoder Distillation Only | https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Semantic segmentation is a crucial task in computer vision that involves segmenting images into semantically meaningful regions at the pixel level. However, existing approaches often rely on expensive human annotations as supervision for model training, limiting their scalability to large, unlabeled datasets. To address this challenge, we present ZeroSeg, a novel method that leverages the existing pretrained vision-language (VL) model (e.g. CLIP vision encoder) to train open-vocabulary zero-shot semantic segmentation models. Although acquired extensive knowledge of visual concepts, it is non-trivial to exploit knowledge from these VL models to the task of semantic segmentation, as they are usually trained at an image level. ZeroSeg overcomes this by distilling the visual concepts learned by VL models into a set of segment tokens, each summarizing a localized region of the target image. We evaluate ZeroSeg on multiple popular segmentation benchmarks, including PASCAL VOC 2012, PASCAL Context, and COCO, in a zero-shot manner Our approach achieves state-of-the-art performance when compared to other zero-shot segmentation methods under the same training data, while also performing competitively compared to strongly supervised methods. Finally, we also demonstrated the effectiveness of ZeroSeg on open-vocabulary segmentation, through both human studies and qualitative visualizations. The code is publicly available at https://github.com/facebookresearch/ZeroSeg

</details>

---

## 5. SINC: Self-Supervised In-Context Learning for Vision-Language Tasks

- [ ] SINC: Self-Supervised In-Context Learning for Vision-Language Tasks | https://openaccess.thecvf.com/content/ICCV2023/html/Chen_SINC_Self-Supervised_In-Context_Learning_for_Vision-Language_Tasks_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_SINC_Self-Supervised_In-Context_Learning_for_Vision-Language_Tasks_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large Pre-trained Transformers exhibit an intriguing capacity for in-context learning. Without gradient updates, these models can rapidly construct new predictors from demonstrations presented in the inputs. Recent works promote this ability in the vision-language domain by incorporating visual information into large language models that can already make in-context predictions. However, these methods could inherit issues in the language domain, such as template sensitivity and hallucination. Also, the scale of these language models raises a significant demand for computations, making learning and operating these models resource-intensive. To this end, we raise a question: "How can we enable in-context learning without relying on the intrinsic in-context ability of large language models?". To answer it, we propose a succinct and general framework, Self-supervised IN-Context learning (SINC), that introduces a meta-model to learn on self-supervised prompts consisting of tailored demonstrations. The learned models can be transferred to downstream tasks for making in-context predictions on-the-fly. Extensive experiments show that SINC outperforms gradient-based methods in various vision-language tasks under few-shot settings. Furthermore, the designs of SINC help us investigate the benefits of in-context learning across different tasks, and the analysis further reveals the essential components for the emergence of in-context learning in the vision-language domain.

</details>

---

## 6. Tem-Adapter: Adapting Image-Text Pretraining for Video Question Answer

- [ ] Tem-Adapter: Adapting Image-Text Pretraining for Video Question Answer | https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Tem-Adapter_Adapting_Image-Text_Pretraining_for_Video_Question_Answer_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Tem-Adapter_Adapting_Image-Text_Pretraining_for_Video_Question_Answer_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-trained models have shown remarkable success in guiding video question-answering (VideoQA) tasks. However, due to the length of video sequences, training large-scale video-based models incurs considerably higher costs than training image-based ones. This motivates us to leverage the knowledge from image-based pretraining, despite the obvious gaps between image and video domains. To bridge these gaps, in this paper, we propose Tem-Adapter, which enables the learning of temporal dynamics and complex semantics by a visual Temporal Aligner and a textual Semantic Aligner. Unlike conventional pretrained knowledge adaptation methods that only concentrate on the downstream task objective, the Temporal Aligner introduces an extra language-guided autoregressive task aimed at facilitating the learning of temporal dependencies, with the objective of predicting future states based on historical clues and language guidance that describes event progression. Besides, to reduce the semantic gap and adapt the textual representation for better event description, we introduce a Semantic Aligner that first designs a template to fuse question and answer pairs as event descriptions and then learns a Transformer decoder with the whole video sequence as guidance for refinement. We evaluate Tem-Adapter and different pre-train transferring methods on two VideoQA benchmarks, and the significant performance improvement demonstrates the effectiveness of our method.

</details>

---

## 7. UniT3D: A Unified Transformer for 3D Dense Captioning and Visual Grounding

- [ ] UniT3D: A Unified Transformer for 3D Dense Captioning and Visual Grounding | https://openaccess.thecvf.com/content/ICCV2023/html/Chen_UniT3D_A_Unified_Transformer_for_3D_Dense_Captioning_and_Visual_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_UniT3D_A_Unified_Transformer_for_3D_Dense_Captioning_and_Visual_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Performing 3D dense captioning and visual grounding requires a common and shared understanding of the underlying multimodal relationships. However, despite some previous attempts on connecting these two related tasks with highly task-specific neural modules, it remains understudied how to explicitly depict their shared nature to learn them simultaneously. In this work, we propose UniT3D, a simple yet effective fully unified transformer-based architecture for jointly solving 3D visual grounding and dense captioning. UniT3D enables learning a strong multimodal representation across the two tasks through a supervised joint pre-training scheme with bidirectional and seq-to-seq objectives. With a generic architecture design, UniT3D allows expanding the pre-training scope to more various training sources such as the synthesized data from 2D prior knowledge to benefit 3D vision-language tasks. Extensive experiments and analysis demonstrate that UniT3D obtains significant gains for 3D dense captioning and visual grounding.

</details>

---

## 8. Video Action Recognition with Attentive Semantic Units

- [ ] Video Action Recognition with Attentive Semantic Units | https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Video_Action_Recognition_with_Attentive_Semantic_Units_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Video_Action_Recognition_with_Attentive_Semantic_Units_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual-Language Models (VLMs) have significantly advanced video action recognition. Supervised by the semantics of action labels, recent works adapt the visual branch of VLMs to learn video representations. Despite the effectiveness proved by these works, we believe that the potential of VLMs has yet to be fully harnessed. In light of this, we exploit the semantic units (SU) hiding behind the action labels and leverage their correlations with fine-grained items in frames for more accurate action recognition. SUs are entities extracted from the language descriptions of the entire action set, including body parts, objects, scenes, and motions. To further enhance the alignments between visual contents and the SUs, we introduce a multi-region module (MRA) to the visual branch of the VLM. The MRA allows the perception of region-aware visual features beyond the original global feature. Our method adaptively attends to and selects relevant SUs with visual features of frames. With a cross-modal decoder, the selected SUs serve to decode spatiotemporal video representations. In summary, the SUs as the medium can boost discriminative ability and transferability. Specifically, in fully-supervised learning, our method achieved 87.8% top-1 accuracy on Kinetics-400. In K=2 few-shot experiments, our method surpassed the previous state-of-the-art by +7.1% and +15.0% on HMDB-51 and UCF-101, respectively.

</details>

---

## 9. ChartReader: A Unified Framework for Chart Derendering and Comprehension without Heuristic Rules

- [ ] ChartReader: A Unified Framework for Chart Derendering and Comprehension without Heuristic Rules | https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_ChartReader_A_Unified_Framework_for_Chart_Derendering_and_Comprehension_without_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_ChartReader_A_Unified_Framework_for_Chart_Derendering_and_Comprehension_without_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Charts are a powerful tool for visually conveying complex data, but their comprehension poses a challenge due to the diverse chart types and intricate components. Existing chart comprehension methods suffer from either heuristic rules or an over-reliance on OCR systems, resulting in suboptimal performance. To address these issues, we present ChartReader, a unified framework that seamlessly integrates chart derendering and comprehension tasks. Our approach includes a transformer-based chart component detection module and an extended pre-trained vision-language model for chart-to-X tasks. By learning the rules of charts automatically from annotated datasets, our approach eliminates the need for manual rule-making, reducing effort and enhancing accuracy. We also introduce a data variable replacement technique and extend the input and position embeddings of the pre-trained model for cross-task training. We evaluate ChartReader on Chart-to-Table, ChartQA, and Chart-to-Text tasks, demonstrating its superiority over existing methods. Our proposed framework can significantly reduce the manual effort involved in chart analysis, providing a step towards a universal chart understanding model. Moreover, our approach offers opportunities for plug-and-play integration with mainstream LLMs such as T5 and TaPas, extending their capability to chart comprehension tasks.

</details>

---

## 10. PRIOR: Prototype Representation Joint Learning from Medical Images and Reports

- [ ] PRIOR: Prototype Representation Joint Learning from Medical Images and Reports | https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_PRIOR_Prototype_Representation_Joint_Learning_from_Medical_Images_and_Reports_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_PRIOR_Prototype_Representation_Joint_Learning_from_Medical_Images_and_Reports_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive learning based vision-language joint pre-training has emerged as a successful representation learning strategy. In this paper, we present a prototype representation learning framework incorporating both global and local alignment between medical images and reports. In contrast to standard global multi-modality alignment methods, we employ a local alignment module for fine-grained representation. Furthermore, a cross-modality conditional reconstruction module is designed to interchange information across modalities in the training phase by reconstructing masked images and reports. For reconstructing long reports, a sentence-wise prototype memory bank is constructed, enabling the network to focus on low-level localized visual and high-level clinical linguistic features. Additionally, a non-auto-regressive generation paradigm is proposed for reconstructing non-sequential reports. Experimental results on five downstream tasks, including supervised classification, zero-shot classification, image-to-text retrieval, semantic segmentation, and object detection, show the proposed method outperforms other state-of-the-art methods across multiple datasets and under different dataset size settings. The code is available at https://github.com/QtacierP/PRIOR.

</details>

---

## 11. Distribution-Aware Prompt Tuning for Vision-Language Models

- [ ] Distribution-Aware Prompt Tuning for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Cho_Distribution-Aware_Prompt_Tuning_for_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Cho_Distribution-Aware_Prompt_Tuning_for_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have shown impressive performance on various downstream tasks by utilizing knowledge learned from large data. In general, the performance of VLMs on target tasks can be further improved by prompt tuning, which adds context to the input image or text. By leveraging data from target tasks, various prompt-tuning methods have been studied in the literature. A key to prompt tuning is the feature space alignment between two modalities via learnable vectors with model parameters fixed. We observed that the alignment becomes more effective when embeddings of each modality are 'well-arranged' in the latent space. Inspired by this observation, we proposed distribution-aware prompt tuning (DAPT) for vision-language models, which is simple yet effective. Specifically, the prompts are learned by maximizing inter-dispersion, the distance between classes, as well as minimizing the intra-dispersion measured by the distance between embeddings from the same class. Our extensive experiments on 11 benchmark datasets demonstrate that our method significantly improves generalizability. The code is available at https://github.com/mlvlab/DAPT.

</details>

---

## 12. PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization

- [ ] PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization | https://openaccess.thecvf.com/content/ICCV2023/html/Cho_PromptStyler_Prompt-driven_Style_Generation_for_Source-free_Domain_Generalization_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Cho_PromptStyler_Prompt-driven_Style_Generation_for_Source-free_Domain_Generalization_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In a joint vision-language space, a text feature (e.g., from "a photo of a dog") could effectively represent its relevant image features (e.g., from dog photos). Also, a recent study has demonstrated the cross-modal transferability phenomenon of this joint space. From these observations, we propose PromptStyler which simulates various distribution shifts in the joint space by synthesizing diverse styles via prompts without using any images to deal with source-free domain generalization. The proposed method learns to generate a variety of style features (from "a S* style of a") via learnable style word vectors for pseudo-words S*. To ensure that learned styles do not distort content information, we force style-content features (from "a S* style of a [class]") to be located nearby their corresponding content features (from "[class]") in the joint vision-language space. After learning style word vectors, we train a linear classifier using synthesized style-content features. PromptStyler achieves the state of the art on PACS, VLCS, OfficeHome and DomainNet, even though it does not require any images for training.

</details>

---

## 13. Dual Learning with Dynamic Knowledge Distillation for Partially Relevant Video Retrieval

- [ ] Dual Learning with Dynamic Knowledge Distillation for Partially Relevant Video Retrieval | https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Dual_Learning_with_Dynamic_Knowledge_Distillation_for_Partially_Relevant_Video_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Dual_Learning_with_Dynamic_Knowledge_Distillation_for_Partially_Relevant_Video_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Almost all previous text-to-video retrieval works assume that videos are pre-trimmed with short durations. However, in practice, videos are generally untrimmed containing much background content. In this work, we investigate the more practical but challenging Partially Relevant Video Retrieval (PRVR) task, which aims to retrieve partially relevant untrimmed videos with the query input. Particularly, we propose to address PRVR from a new perspective, i.e., distilling the generalization knowledge from the large-scale vision-language pre-trained model and transferring it to a task-specific PRVR network. To be specific, we introduce a Dual Learning framework with Dynamic Knowledge Distillation (DL-DKD), which exploits the knowledge of a large vision-language model as the teacher to guide a student model. During the knowledge distillation, an inheritance student branch is devised to absorb the knowledge from the teacher model. Considering that the large model may be of mediocre performance due to the domain gaps, we further develop an exploration student branch to take the benefits of task-specific information. By jointly training the above two branches in a dual-learning way, our model is able to selectively acquire appropriate knowledge from the teacher model while capturing the task-specific property. In addition, a dynamical knowledge distillation strategy is further devised to adjust the effect of each student branch learning during the training. Experiment results demonstrate that our proposed model achieves state-of-the-art performance on ActivityNet and TVR datasets for PRVR.

</details>

---

## 14. PODA: Prompt-driven Zero-shot Domain Adaptation

- [ ] PODA: Prompt-driven Zero-shot Domain Adaptation | https://openaccess.thecvf.com/content/ICCV2023/html/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Domain adaptation has been vastly investigated in computer vision but still requires access to target images at train time, which might be intractable in some uncommon conditions. In this paper, we propose the task of 'Prompt-driven Zero-shot Domain Adaptation', where we adapt a model trained on a source domain using only a general description in natural language of the target domain, i.e., a prompt. First, we leverage a pretrained contrastive vision-language model (CLIP) to optimize affine transformations of source features, steering them towards the target text embedding while preserving their content and semantics. To achieve this, we propose Prompt-driven Instance Normalization (PIN). Second, we show that these prompt-driven augmentations can be used to perform zero-shot domain adaptation for semantic segmentation. Experiments demonstrate that our method significantly outperforms CLIP-based style transfer baselines on several datasets for the downstream task at hand, even surpassing one-shot unsupervised domain adaptation. A similar boost is observed on object detection and image classification. The code is available at https://github.com/astra-vision/PODA .

</details>

---

## 15. RCA-NOC: Relative Contrastive Alignment for Novel Object Captioning

- [ ] RCA-NOC: Relative Contrastive Alignment for Novel Object Captioning | https://openaccess.thecvf.com/content/ICCV2023/html/Fan_RCA-NOC_Relative_Contrastive_Alignment_for_Novel_Object_Captioning_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Fan_RCA-NOC_Relative_Contrastive_Alignment_for_Novel_Object_Captioning_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce a novel approach to novel object captioning which employs relative contrastive learning to learn visual and semantic alignment. Our approach maximizes compatibility between regions and object tags in a contrastive manner. To set up a proper contrastive learning objective, for each image, we augment tags by leveraging the relative nature of positive and negative pairs obtained from foundation models such as CLIP. We then use the rank of each augmented tag in a list as a relative relevance label to contrast each top-ranked tag with a set of lower-ranked tags. This learning objective encourages the top-ranked tags to be more compatible with their image and text context than lower-ranked tags, thus improving the discriminative ability of the learned multi-modality representation. We evaluate our approach on two datasets and show that our proposed RCA-NOC approach outperforms state-of-the-art methods by a large margin, demonstrating its effectiveness in improving vision-language representation for novel object captioning.

</details>

---

## 16. Unsupervised Open-Vocabulary Object Localization in Videos

- [ ] Unsupervised Open-Vocabulary Object Localization in Videos | https://openaccess.thecvf.com/content/ICCV2023/html/Fan_Unsupervised_Open-Vocabulary_Object_Localization_in_Videos_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Fan_Unsupervised_Open-Vocabulary_Object_Localization_in_Videos_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we show that recent advances in video representation learning and pre-trained vision-language models allow for substantial improvements in self-supervised video object localization. We propose a method that first localizes objects in videos via a slot attention approach and then assigns text to the obtained slots. The latter is achieved by an unsupervised way to read localized semantic information from the pre-trained CLIP model. The resulting video object localization is entirely unsupervised apart from the implicit annotation contained in CLIP, and it is effectively the first unsupervised approach that yields good results on regular video benchmarks.

</details>

---

## 17. UATVR: Uncertainty-Adaptive Text-Video Retrieval

- [ ] UATVR: Uncertainty-Adaptive Text-Video Retrieval | https://openaccess.thecvf.com/content/ICCV2023/html/Fang_UATVR_Uncertainty-Adaptive_Text-Video_Retrieval_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Fang_UATVR_Uncertainty-Adaptive_Text-Video_Retrieval_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

With the explosive growth of web videos and emerging large-scale vision-language pre-training models, e.g., CLIP, retrieving videos of interest with text instructions has attracted increasing attention. A common practice is to transfer text-video pairs to the same embedding space and craft cross-modal interactions with certain entities in specific granularities for semantic correspondence. Unfortunately, the intrinsic uncertainties of optimal entity combinations in appropriate granularities for cross-modal queries are understudied, which is especially critical for modalities with hierarchical semantics, e.g., video, text, etc. In this paper, we propose an Uncertainty-Adaptive Text-Video Retrieval approach, termed UATVR, which models each look-up as a distribution matching procedure. Concretely, we add additional learnable tokens in the encoders to adaptively aggregate multi-grained semantics for flexible high-level reasoning. In the refined embedding space, we represent text-video pairs as probabilistic distributions where prototypes are sampled for matching evaluation. Comprehensive experiments on four benchmarks justify the superiority of our UATVR, which achieves new state-of-the-art results on MSR-VTT (50.8%), VATEX (64.5%), MSVD (49.7%), and DiDeMo (45.8%). The code is available at https://github.com/bofang98/UATVR.

</details>

---

## 18. Transferable Decoding with Visual Entities for Zero-Shot Image Captioning

- [ ] Transferable Decoding with Visual Entities for Zero-Shot Image Captioning | https://openaccess.thecvf.com/content/ICCV2023/html/Fei_Transferable_Decoding_with_Visual_Entities_for_Zero-Shot_Image_Captioning_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Fei_Transferable_Decoding_with_Visual_Entities_for_Zero-Shot_Image_Captioning_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Image-to-text generation aims to describe images using natural language. Recently, zero-shot image captioning based on pre-trained vision-language models (VLMs) and large language models (LLMs) has made significant progress. However, we have observed and empirically demonstrated that these methods are susceptible to modality bias induced by LLMs and tend to generate descriptions containing objects (entities) that do not actually exist in the image but frequently appear during training (i.e., object hallucination). In this paper, we propose ViECap, a transferable decoding model that leverages entity-aware decoding to generate descriptions in both seen and unseen scenarios. ViECap incorporates entity-aware hard prompts to guide LLMs' attention toward the visual entities present in the image, enabling coherent caption generation across diverse scenes. With entity-aware hard prompts, ViECap is capable of maintaining performance when transferring from in-domain to out-of-domain scenarios. Extensive experiments demonstrate that ViECap sets a new state-of-theart cross-domain (transferable) captioning and performs competitively in-domain captioning compared to previous VLMs-based zero-shot methods. Our code is available at: https://github.com/FeiElysia/ViECap

</details>

---

## 19. Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning

- [ ] Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning | https://openaccess.thecvf.com/content/ICCV2023/html/Feng_Diverse_Data_Augmentation_with_Diffusions_for_Effective_Test-time_Prompt_Tuning_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Feng_Diverse_Data_Augmentation_with_Diffusions_for_Effective_Test-time_Prompt_Tuning_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Benefiting from prompt tuning, recent years have witnessed the promising performance of pre-trained vision-language models, e.g., CLIP, on versatile downstream tasks. In this paper, we focus on a particular setting of learning adaptive prompts on the fly for each test sample from an unseen new domain, which is known as test-time prompt tuning (TPT). Existing TPT methods typically rely on data augmentation and confidence selection. However, conventional data augmentation techniques, e.g., random resized crops, suffers from the lack of data diversity, while entropy-based confidence selection alone is not sufficient to guarantee prediction fidelity. To address these issues, we propose a novel TPT method, named DiffTPT, which leverages pre-trained diffusion models to generate diverse and informative new data. Specifically, we incorporate augmented data by both conventional method and pre-trained stable diffusion to exploit their respective merits, improving the model's ability to adapt to unknown new test data. Moreover, to ensure the prediction fidelity of generated data, we introduce a cosine similarity-based filtration technique to select the generated data with higher similarity to the single test sample. Our experiments on test datasets with distribution shifts and unseen categories demonstrate that DiffTPT improves the zero-shot accuracy by an average of 5.13% compared to the state-of-the-art TPT method.

</details>

---

## 20. Towards Models that Can See and Read

- [ ] Towards Models that Can See and Read | https://openaccess.thecvf.com/content/ICCV2023/html/Ganz_Towards_Models_that_Can_See_and_Read_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Ganz_Towards_Models_that_Can_See_and_Read_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual Question Answering (VQA) and Image Captioning (CAP), which are among the most popular vision-language tasks, 
 have analogous scene-text versions that require reasoning from the text in the image. Despite their obvious resemblance, the two are treated independently and, as we show, yield task-specific methods that can either see or read, but not both. In this work, we conduct an in-depth analysis of this phenomenon and propose UniTNT, a Unified Text-Non-Text approach, which grants existing multimodal architectures scene-text understanding capabilities. Specifically, we treat scene-text information as an additional modality, fusing it with any pretrained encoder-decoder-based architecture via designated modules. Thorough experiments reveal that UniTNT leads to the first single model that successfully handles both task types. Moreover, we show that scene-text understanding capabilities can boost vision-language models' performance on general VQA and CAP by up to 2.69% and 0.6 CIDEr, respectively.

</details>

---

## 21. CLIPTrans: Transferring Visual Knowledge with Pre-trained Models for Multimodal Machine Translation

- [ ] CLIPTrans: Transferring Visual Knowledge with Pre-trained Models for Multimodal Machine Translation | https://openaccess.thecvf.com/content/ICCV2023/html/Gupta_CLIPTrans_Transferring_Visual_Knowledge_with_Pre-trained_Models_for_Multimodal_Machine_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Gupta_CLIPTrans_Transferring_Visual_Knowledge_with_Pre-trained_Models_for_Multimodal_Machine_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

There has been a growing interest in developing multimodal machine translation (MMT) systems that enhance neural machine translation (NMT) with visual knowledge. This problem setup involves using images as auxiliary information during training, and more recently, eliminating their use during inference. Towards this end, previous works face a challenge in training powerful MMT models from scratch due to the scarcity of annotated multilingual vision-language data, especially for low-resource languages. Simultaneously, there has been an influx of multilingual pre-trained models for NMT and multimodal pre-trained models for vision-language tasks, primarily in English, which have shown exceptional generalisation ability. However, these are not directly applicable to MMT since they do not provide aligned multimodal multilingual features for generative tasks. To alleviate this issue, instead of designing complex modules for MMT, we propose CLIPTrans, which simply adapts the independently pre-trained multimodal M-CLIP and the multilingual mBART. In order to align their embedding spaces, mBART is conditioned on the M-CLIP features by a prefix sequence generated through a lightweight mapping network. We train this in a two-stage pipeline which warms up the model with image captioning before the actual translation task. Through experiments, we demonstrate the merits of this framework and consequently push forward the state-of-the-art across standard benchmarks by an average of +2.67 BLEU. The code can be found at www.github.com/devaansh100/CLIPTrans.

</details>

---

## 22. AutoAD II: The Sequel - Who, When, and What in Movie Audio Description

- [ ] AutoAD II: The Sequel - Who, When, and What in Movie Audio Description | https://openaccess.thecvf.com/content/ICCV2023/html/Han_AutoAD_II_The_Sequel_-_Who_When_and_What_in_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Han_AutoAD_II_The_Sequel_-_Who_When_and_What_in_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Audio Description (AD) is the task of generating descriptions of visual content, at suitable time intervals, for the benefit of visually impaired audiences. For movies, this presents notable challenges -- AD must occur only during existing pauses in dialogue, should refer to characters by name, and ought to aid understanding of the storyline as a whole.
 To this end, we develop a new model for automatically generating movie AD, given CLIP visual features of the frames, the cast list, and the temporal locations of the speech; addressing all three of the `who', `when', and `what' questions: (i) who -- we introduce a character bank consisting of the character's name, the actor that played the part, and a CLIP feature of their face, for the principal cast of each movie, and demonstrate how this can be used to improve naming in the generated AD; (ii) when -- we investigate several models for determining whether an AD should be generated for a time interval or not, based on the visual content of the interval and its neighbours; and (iii) what -- we implement a new vision-language model for this task, that can ingest the proposals from the character bank, whilst conditioning on the visual features using cross-attention, and demonstrate how this improves over previous architectures for AD text generation in an apples-to-apples comparison.

</details>

---

## 23. CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos

- [ ] CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos | https://openaccess.thecvf.com/content/ICCV2023/html/Han_CHAMPAGNE_Learning_Real-world_Conversation_from_Large-Scale_Web_Videos_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Han_CHAMPAGNE_Learning_Real-world_Conversation_from_Large-Scale_Web_Videos_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual information is central to conversation: body gestures and physical behaviour, for example, contribute to meaning that transcends words alone. To date, however, most neural conversational models are limited to just text. We introduce CHAMPAGNE, a generative model of conversations that can account for visual contexts. To train CHAMPAGNE, we collect and release YTD-18M, a large-scale corpus of 18M video-based dialogues. YTD-18M is constructed from web videos: crucial to our data collection pipeline is a pretrained language model that converts error-prone automatic transcripts to a cleaner dialogue format while maintaining meaning.
 Human evaluation reveals that YTD-18M is more sensible and specific than prior resources (MMDialog, 1M dialogues), while maintaining visual-groundedness. Experiments demonstrate that 1) CHAMPAGNE learns to conduct conversation from YTD-18M; and 2) when fine-tuned, it achieves state-of-the-art results on four vision-language tasks focused on real-world conversations. We release data, models, and code.

</details>

---

## 24. Global Knowledge Calibration for Fast Open-Vocabulary Segmentation

- [ ] Global Knowledge Calibration for Fast Open-Vocabulary Segmentation | https://openaccess.thecvf.com/content/ICCV2023/html/Han_Global_Knowledge_Calibration_for_Fast_Open-Vocabulary_Segmentation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Han_Global_Knowledge_Calibration_for_Fast_Open-Vocabulary_Segmentation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in pre-trained vision-language models, such as CLIP, have enabled the segmentation of arbitrary concepts solely from textual inputs, a process commonly referred to as open-vocabulary semantic segmentation (OVS). However, existing OVS techniques confront a fundamental challenge: the trained classifier tends to overfit on the base classes observed during training, resulting in suboptimal generalization performance to unseen classes. To mitigate this issue, recent studies have proposed the use of an additional frozen pre-trained CLIP for classification. Nonetheless, this approach incurs heavy computational overheads as the CLIP vision encoder must be repeatedly forward-passed for each mask, rendering it impractical for real-world applications. To address this challenge, our objective is to develop a fast OVS model that can perform comparably or better without the extra computational burden of the CLIP image encoder during inference. To this end, we propose a core idea of preserving the generalizable representation when fine-tuning on known classes. Specifically, we introduce a text diversification strategy that generates a set of synonyms for each training category, which prevents the learned representation from collapsing onto specific known category names. Additionally, we employ a text-guided knowledge distillation method to preserve the generalizable knowledge of CLIP. Extensive experiments demonstrate that our proposed model achieves robust generalization performance across various datasets. Furthermore, we perform a preliminary exploration of open-vocabulary video segmentation and present a benchmark that can facilitate future open-vocabulary research in the video domain.

</details>

---

## 25. EgoTV: Egocentric Task Verification from Natural Language Task Descriptions

- [ ] EgoTV: Egocentric Task Verification from Natural Language Task Descriptions | https://openaccess.thecvf.com/content/ICCV2023/html/Hazra_EgoTV_Egocentric_Task_Verification_from_Natural_Language_Task_Descriptions_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Hazra_EgoTV_Egocentric_Task_Verification_from_Natural_Language_Task_Descriptions_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

To enable progress towards egocentric agents capable of understanding everyday tasks specified in natural language, we propose a benchmark and a synthetic dataset called Egocentric Task Verification (EgoTV). The goal in EgoTV is to verify the execution of tasks from egocentric videos based on the natural language description of these tasks. EgoTV contains pairs of videos and their task descriptions for multi-step tasks -- these tasks contain multiple sub-task decompositions, state changes, object interactions, and sub-task ordering constraints. In addition, EgoTV also provides abstracted task descriptions that contain only partial details about ways to accomplish a task. Consequently, EgoTV requires causal, temporal, and compositional reasoning of video and language modalities, which is missing in existing datasets. We also find that existing vision-language models struggle at such all round reasoning needed for task verification in EgoTV. Inspired by the needs of EgoTV, we propose a novel Neuro-Symbolic Grounding (NSG) approach that leverages symbolic representations to capture the compositional and temporal structure of tasks. We demonstrate NSG's capability towards task tracking and verification on our EgoTV dataset and a real-world dataset derived from CrossTask (CTV). We open-source the EgoTV and CTV datasets and the NSG model for future research on egocentric assistive agents.

</details>

---

## 26. A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance

- [ ] A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance | https://openaccess.thecvf.com/content/ICCV2023/html/Huang_A_Sentence_Speaks_a_Thousand_Images_Domain_Generalization_through_Distilling_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Huang_A_Sentence_Speaks_a_Thousand_Images_Domain_Generalization_through_Distilling_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Domain generalization studies the problem of training a model with samples from several domains (or distributions) and then testing the model with samples from a new, unseen domain. In this paper, we propose a novel approach for domain generalization that leverages recent advances in large vision-language models, specifically a CLIP teacher model, to train a smaller model that generalizes to unseen domains. The key technical contribution is a new type of regularization that requires the student's learned image representations to be close to the teacher's learned text representations obtained from encoding the corresponding text descriptions of images. We introduce two designs of the loss function, absolute and relative distance, which provide specific guidance on how the training process of the student model should be regularized. We evaluate our proposed method, dubbed RISE (Regularized Invariance with Semantic Embeddings), on various benchmark datasets, and show that it outperforms several state-of-the-art domain generalization methods. To our knowledge, our work is the first to leverage knowledge distillation using a large vision-language model for domain generalization. By incorporating text-based information, RISE improves the generalization capability of machine learning models.

</details>

---

## 27. CLIP2Point: Transfer CLIP to Point Cloud Classification with Image-Depth Pre-Training

- [ ] CLIP2Point: Transfer CLIP to Point Cloud Classification with Image-Depth Pre-Training | https://openaccess.thecvf.com/content/ICCV2023/html/Huang_CLIP2Point_Transfer_CLIP_to_Point_Cloud_Classification_with_Image-Depth_Pre-Training_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Huang_CLIP2Point_Transfer_CLIP_to_Point_Cloud_Classification_with_Image-Depth_Pre-Training_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-training across 3D vision and language remains under development because of limited training data. Recent works attempt to transfer vision-language (V-L) pre-training methods to 3D vision. However, the domain gap between 3D and images is unsolved, so that V-L pre-trained models are restricted in 3D downstream tasks. To address this issue, we propose CLIP2Point, an image-depth pre-training method by contrastive learning to transfer CLIP to the 3D domain, and adapt it to point cloud classification. We introduce a new depth rendering setting that forms a better visual effect, and then render 52,460 pairs of images and depth maps from ShapeNet for pre-training. The pre-training scheme of CLIP2Point combines cross-modality learning to enforce the depth features for capturing expressive visual and textual features and intra-modality learning to enhance the invariance of depth aggregation. Additionally, we propose a novel Gated Dual-Path Adapter (GDPA), i.e., a dual-path structure with global-view aggregators and gated fusion for downstream representative learning. It allows the ensemble of CLIP and CLIP2Point, tuning pre-training knowledge to downstream tasks in an efficient adaptation. Experimental results show that CLIP2Point is effective in transferring CLIP knowledge to 3D vision. Our CLIP2Point outperforms other 3D transfer learning and pre-training networks, achieving state-of-the-art results on zero-shot, few-shot, and fully-supervised classification.

</details>

---

## 28. ESTextSpotter: Towards Better Scene Text Spotting with Explicit Synergy in Transformer

- [ ] ESTextSpotter: Towards Better Scene Text Spotting with Explicit Synergy in Transformer | https://openaccess.thecvf.com/content/ICCV2023/html/Huang_ESTextSpotter_Towards_Better_Scene_Text_Spotting_with_Explicit_Synergy_in_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Huang_ESTextSpotter_Towards_Better_Scene_Text_Spotting_with_Explicit_Synergy_in_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In recent years, end-to-end scene text spotting approaches are evolving to the Transformer-based framework. While previous studies have shown the crucial importance of the intrinsic synergy between text detection and recognition, recent advances in Transformer-based methods usually adopt an implicit synergy strategy with shared query, which can not fully realize the potential of these two interactive tasks. In this paper, we argue that the explicit synergy considering distinct characteristics of text detection and recognition can significantly improve the performance text spotting. To this end, we introduce a new model named Explicit Synergy-based Text Spotting Transformer framework (ESTextSpotter), which achieves explicit synergy by modeling discriminative and interactive features for text detection and recognition within a single decoder. Specifically, we decompose the conventional shared query into task-aware queries for text polygon and content, respectively. Through the decoder with the proposed vision-language communication module, the queries interact with each other in an explicit manner while preserving discriminative patterns of text detection and recognition, thus improving performance significantly. Additionally, we propose a task-aware query initialization scheme to ensure stable training. Experimental results demonstrate that our model significantly outperforms previous state-of-the-art methods. Code is available at https://github.com/mxin262/ESTextSpotter.

</details>

---

## 29. Improving Diversity in Zero-Shot GAN Adaptation with Semantic Variations

- [ ] Improving Diversity in Zero-Shot GAN Adaptation with Semantic Variations | https://openaccess.thecvf.com/content/ICCV2023/html/Jeon_Improving_Diversity_in_Zero-Shot_GAN_Adaptation_with_Semantic_Variations_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Jeon_Improving_Diversity_in_Zero-Shot_GAN_Adaptation_with_Semantic_Variations_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Training deep generative models usually requires a large amount of data. To alleviate the data collection cost, the task of zero-shot GAN adaptation aims to reuse well-trained generators to synthesize images of an unseen target domain without any further training samples. Due to the data absence, the textual description of the target domain and the vision-language models, e.g., CLIP, are utilized to effectively guide the generator. However, with only a single representative text feature instead of real images, the synthesized images gradually lose diversity as the model is optimized, which is also known as mode collapse. To tackle the problem, we propose a novel method to find semantic variations of the target text in the CLIP space. Specifically, we explore diverse semantic variations based on the informative text feature of the target domain while regularizing the uncontrolled deviation of the semantic information. With the obtained variations, we design a novel directional moment loss that matches the first and second moments of image and text direction distributions. Moreover, we introduce elastic weight consolidation and a relation consistency loss to effectively preserve valuable content information from the source domain, e.g., appearances. Through extensive experiments, we demonstrate the efficacy of the proposed methods in ensuring sample diversity in various scenarios of zero-shot GAN adaptation. We also conduct ablation studies to validate the effect of each proposed component. Notably, our model achieves a new state-of-the-art on zero-shot GAN adaptation in terms of both diversity and quality.

</details>

---

## 30. BUS: Efficient and Effective Vision-Language Pre-Training with Bottom-Up Patch Summarization.

- [ ] BUS: Efficient and Effective Vision-Language Pre-Training with Bottom-Up Patch Summarization. | https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_BUS_Efficient_and_Effective_Vision-Language_Pre-Training_with_Bottom-Up_Patch_Summarization._ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_BUS_Efficient_and_Effective_Vision-Language_Pre-Training_with_Bottom-Up_Patch_Summarization._ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision Transformer (ViT) based Vision-Language Pretraining (VLP) models recently demonstrated impressive performance in various tasks. However, the lengthy visual token sequences used in these models can lead to inefficient and ineffective performance. Existing methods to address these issues lack textual guidance and may overlook crucial visual information related to the text, leading to the introduction of irrelevant information during cross-modal fusion and additional computational cost. In this paper, we propose a Bottom-Up Patch Summarization approach named BUS which is inspired by the Document Summarization Task in NLP to learn a concise visual summary of lengthy visual token sequences, guided by textual semantics. We introduce a Text-Semantic Aware Patch Selector (TAPS) in the ViT backbone to perform a coarse-grained selective visual summarization to over-determine the text-relevant patches, and a light Summarization Decoder to perform fine-grained abstractive summarization based on the selected patches, resulting in a further condensed representation sequence that highlights text-relevant visual semantic information. Such bottom-up process is both efficient and effective with higher performing. We evaluate our approach on various VL understanding and generation tasks and show competitive or better downstream task performance while boosting the efficiency by 50%. Additionally, our model achieves well-designed SOTA downstream task performance by increasing input image resolution without increasing computational costs compared to baselines.

</details>

---

## 31. Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models

- [ ] Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Kan_Knowledge-Aware_Prompt_Tuning_for_Generalizable_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Kan_Knowledge-Aware_Prompt_Tuning_for_Generalizable_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models, e.g., CLIP, working with manually designed prompts have demonstrated great effectiveness in transfer learning. Recently, learnable prompts achieve state-of-the-art performance, which however are prone to overfit to seen classes while failing to generalize to unseen classes. In this paper, we propose a Knowledge-Aware Prompt Tuning (KAPT) framework for vision-language models. Our approach takes the inspiration from human intelligence in which external knowledge is usually incorporated into recognizing novel categories of objects. Specifically, we design two complementary types of knowledge-aware prompts for the text encoder to leverage the distinctive characteristics of category-related external knowledge. The discrete prompt extracts the key information from descriptions of an object category, and the learned continuous prompt captures overall contexts. We further design an adaptation head for the visual encoder to aggregate salient attentive visual cues, which establishes discriminative and task-aware visual representations. We conduct extensive experiments on 11 widely-used benchmark datasets and the results verify the effectiveness in few-shot image classification, especially in generalizing to unseen categories. Compared with the state-of-the-art CoCoOp method, KAPT exhibits favorable performance and achieves an absolute gain of 3.22% on new classes and 2.57% in terms of harmonic mean.

</details>

---

## 32. LERF: Language Embedded Radiance Fields

- [ ] LERF: Language Embedded Radiance Fields | https://openaccess.thecvf.com/content/ICCV2023/html/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Humans describe the physical world using natural language to refer to specific 3D locations based on a vast range of properties: visual appearance, semantics, abstract associations, or actionable affordances. In this work we propose Language Embedded Radiance Fields (LERFs), a method for grounding language embeddings from off-the-shelf models like CLIP into NeRF, which enable these types of open-ended language queries in 3D. LERF learns a dense, multi-scale language field inside NeRF by volume rendering CLIP embeddings along training rays, supervising these embeddings across training views to provide multi-view consistency and smooth the underlying language field. After optimization, LERF can extract 3D relevancy maps for a broad range of language prompts interactively in real-time, which has potential use cases in robotics, understanding vision-language models, and interacting with 3D scenes. LERF enables pixel-aligned, zero-shot queries on the distilled 3D CLIP embeddings without relying on region proposals or masks, supporting long-tail open-vocabulary queries hierarchically across the volume. See the project website at: https://lerf.io

</details>

---

## 33. Lecture Presentations Multimodal Dataset: Towards Understanding Multimodality in Educational Videos

- [ ] Lecture Presentations Multimodal Dataset: Towards Understanding Multimodality in Educational Videos | https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Lecture_Presentations_Multimodal_Dataset_Towards_Understanding_Multimodality_in_Educational_Videos_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Lecture_Presentations_Multimodal_Dataset_Towards_Understanding_Multimodality_in_Educational_Videos_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Many educational videos use slide presentations, a sequence of visual pages that contain text and figures accompanied by spoken language, which are constructed and presented carefully in order to optimally transfer knowledge to students. Previous studies in multimedia and psychology attribute the effectiveness of lecture presentations to their multimodal nature. As a step toward developing vision-language models to aid in student learning as intelligent teacher assistants, we introduce the Lecture Presentations Multimodal (LPM) Dataset as a large-scale benchmark testing the capabilities of vision-and-language models in multimodal understanding of educational videos. Our dataset contains aligned slides and spoken language, for 180+ hours of video and 9000+ slides, with 10 lecturers from various subjects (e.g., computer science, dentistry, biology). We introduce three research tasks, (1) figure-to-text retrieval, (2) text-to-figure retrieval, and (3) generation of slide explanations, which are grounded in multimedia learning and psychology principles to test a vision-language model's understanding of multimodal content. We provide manual annotations to help implement these tasks and establish baselines on them. Comparing baselines and human student performances, we find that state-of-the-art vision-language models (zero-shot and fine-tuned) struggle in (1) weak crossmodal alignment between slides and spoken text, (2) learning novel visual mediums, (3) technical language, and (4) long-range sequences. We introduce PolyViLT, a novel multimodal transformer trained with a multi-instance learning loss that is more effective than current approaches for retrieval. We conclude by shedding light on the challenges and opportunities in multimodal understanding of educational presentation videos.

</details>

---

## 34. Read-only Prompt Optimization for Vision-Language Few-shot Learning

- [ ] Read-only Prompt Optimization for Vision-Language Few-shot Learning | https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Read-only_Prompt_Optimization_for_Vision-Language_Few-shot_Learning_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Read-only_Prompt_Optimization_for_Vision-Language_Few-shot_Learning_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In recent years, prompt tuning has proven effective in adapting pre-trained vision-language models to down- stream tasks. These methods aim to adapt the pre-trained models by introducing learnable prompts while keeping pre- trained weights frozen. However, learnable prompts can affect the internal representation within the self-attention module, which may negatively impact performance vari- ance and generalization, especially in data-deficient set- tings. To address these issues, we propose a novel ap- proach, Read-only Prompt Optimization (RPO). RPO lever- ages masked attention to prevent the internal representa- tion shift in the pre-trained model. Further, to facilitate the optimization of RPO, the read-only prompts are ini- tialized based on special tokens of the pre-trained model. Our extensive experiments demonstrate that RPO outper- forms CLIP and CoCoOp in base-to-new generalization and domain generalization while displaying better robust- ness. Also, the proposed method achieves better generaliza- tion on extremely data-deficient settings, while improving parameter efficiency and computational overhead. Code is
 available at https://github.com/mlvlab/RPO.

</details>

---

## 35. Efficient Adaptive Human-Object Interaction Detection with Concept-guided Memory

- [ ] Efficient Adaptive Human-Object Interaction Detection with Concept-guided Memory | https://openaccess.thecvf.com/content/ICCV2023/html/Lei_Efficient_Adaptive_Human-Object_Interaction_Detection_with_Concept-guided_Memory_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lei_Efficient_Adaptive_Human-Object_Interaction_Detection_with_Concept-guided_Memory_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Human Object Interaction (HOI) detection aims to localize and infer the relationships between a human and an object. Arguably, training supervised models for this task from scratch presents challenges due to the performance drop over rare classes and the high computational cost and time required to handle long-tailed distributions of HOIs in complex HOI scenes in realistic settings. This observation motivates us to design an HOI detector that can be trained even with long-tailed labeled data and can leverage existing knowledge from pre-trained models. Inspired by the powerful generalization ability of the large Vision-Language Models (VLM) on classification and retrieval tasks, we propose an efficient Adaptive HOI Detector with Concept-guided Memory (ADA-CM). ADA-CM has two operating modes. The first mode makes it tunable without learning new parameters in a training-free paradigm. Its second mode incorporates an instance-aware adapter mechanism that can further efficiently boost performance if updating a lightweight set of parameters can be afforded. Our proposed method achieves competitive results with state-of-the-art on the HICO-DET and V-COCO datasets with much less training time. Code can be found at https://github.com/ltttpku/ADA-CM.

</details>

---

## 36. Distilling Large Vision-Language Model with Out-of-Distribution Generalizability

- [ ] Distilling Large Vision-Language Model with Out-of-Distribution Generalizability | https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models have achieved outstanding performance, but their size and computational requirements make their deployment on resource-constrained devices and time-sensitive tasks impractical. Model distillation, the process of creating smaller, faster models that maintain the performance of larger models, is a promising direction towards the solution. This paper investigates the distillation of visual representations in large teacher vision-language models into lightweight student models using a small- or mid-scale dataset. Notably, this study focuses on open-vocabulary out-of-distribution (OOD) generalization, a challenging problem that has been overlooked in previous model distillation literature. We propose two principles from vision and language modality perspectives to enhance student's OOD generalization: (1) by better imitating teacher's visual representation space, and carefully promoting better coherence in vision-language alignment with the teacher; (2) by enriching the teacher's language representations with informative and finegrained semantic attributes to effectively distinguish between different labels. We propose several metrics and conduct extensive experiments to investigate their techniques. The results demonstrate significant improvements in zero-shot and few-shot student performance on open-vocabulary out-of-distribution classification, highlighting the effectiveness of our proposed approaches.

</details>

---

## 37. Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection

- [ ] Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection | https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Current methods for open-vocabulary object detection (OVOD) rely on a pre-trained vision-language model (VLM) to acquire the recognition ability. In this paper, we propose a simple yet effective framework to Distill the Knowledge from the VLM to a DETR-like detector, termed DK-DETR. Specifically, we present two ingenious distillation schemes named semantic knowledge distillation (SKD) and relational knowledge distillation (RKD). To utilize the rich knowledge from the VLM systematically, SKD transfers the semantic knowledge explicitly, while RKD exploits implicit relationship information between objects. Furthermore, a distillation branch including a group of auxiliary queries is added to the detector to mitigate the negative effect on base categories. Equipped with SKD and RKD on the distillation branch, DK-DETR improves the detection performance of novel categories significantly and avoids disturbing the detection of base categories. Extensive experiments on LVIS and COCO datasets show that DK-DETR surpasses existing OVOD methods under the setting that the base-category supervision is solely available. The code and models are available at https://github.com/hikvision-research/opera.

</details>

---

## 38. Gradient-Regulated Meta-Prompt Learning for Generalizable Vision-Language Models

- [ ] Gradient-Regulated Meta-Prompt Learning for Generalizable Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Li_Gradient-Regulated_Meta-Prompt_Learning_for_Generalizable_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Gradient-Regulated_Meta-Prompt_Learning_for_Generalizable_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning, a recently emerging paradigm, enables the powerful vision-language pre-training models to adapt to downstream tasks in a parameter- and data- efficient way, by learning the "soft prompts" to condition frozen pre-training models. Though effective, it is particularly problematic in the few-shot scenario, where prompt tuning performance is sensitive to the initialization and requires a time-consuming process to find a good initialization, thus restricting the fast adaptation ability of the pre-training models. In addition, prompt tuning could undermine the generalizability of the pre-training models, because the learnable prompt tokens are easy to overfit to the limited training samples. To address these issues, we introduce a novel Gradient-RegulAted Meta-prompt learning (GRAM) framework that jointly meta-learns an efficient soft prompt initialization for better adaptation and a lightweight gradient regulating function for strong cross-domain generalizability in a meta-learning paradigm using only the unlabeled image-text pre-training data. Rather than designing a specific prompt tuning method, our GRAM can be easily incorporated into various prompt tuning methods in a model-agnostic way, and comprehensive experiments show that GRAM brings about consistent improvement for them in several settings (i.e., few-shot learning, cross-domain generalization, cross-dataset generalization, etc.) over 11 datasets. Further, experiments show that GRAM enables the orthogonal methods of textual and visual prompt tuning to work in a mutually-enhanced way, offering better generalizability beyond the uni-modal prompt tuning methods.

</details>

---

## 39. Progressive Spatio-Temporal Prototype Matching for Text-Video Retrieval

- [ ] Progressive Spatio-Temporal Prototype Matching for Text-Video Retrieval | https://openaccess.thecvf.com/content/ICCV2023/html/Li_Progressive_Spatio-Temporal_Prototype_Matching_for_Text-Video_Retrieval_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Progressive_Spatio-Temporal_Prototype_Matching_for_Text-Video_Retrieval_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The performance of text-video retrieval has been significantly improved by vision-language cross-modal learning schemes. 
 The typical solution is to directly align the global video-level and sentence-level features during learning, which would ignore the intrinsic video-text relations, i.e., a text description only corresponds to a spatio-temporal part of videos. 
 Hence, the matching process should consider both fine-grained spatial content and various temporal semantic events.
 To this end, we propose a text-video learning framework with progressive spatio-temporal prototype matching. Specifically, the vanilla matching process is decomposed into two complementary phases: object-phrase prototype matching and event-sentence prototype matching. In the object-phrase prototype matching phase, a spatial prototype generation mechanism is developed to predict key patches or words, which are sparsely integrated into object or phrase prototypes. Importantly, optimizing the local alignment between object-phrase prototypes helps the model perceive spatial details. In the event-sentence prototype matching phase, we design a temporal prototype generation mechanism to associate intra-frame objects and interact inter-frame temporal relations. Such progressively generated event prototypes can reveal semantic diversity in videos for dynamic matching. Validated by comprehensive experiments, our method consistently outperforms the state-of-the-art methods on four video retrieval benchmarks.

</details>

---

## 40. Unmasked Teacher: Towards Training-Efficient Video Foundation Models

- [ ] Unmasked Teacher: Towards Training-Efficient Video Foundation Models | https://openaccess.thecvf.com/content/ICCV2023/html/Li_Unmasked_Teacher_Towards_Training-Efficient_Video_Foundation_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Unmasked_Teacher_Towards_Training-Efficient_Video_Foundation_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video Foundation Models (VFMs) have received limited exploration due to high computational costs and data scarcity. Previous VFMs rely on Image Foundation Models (IFMs), which face challenges in transferring to the video domain. Although VideoMAE has trained a robust ViT from limited data, its low-level reconstruction poses convergence difficulties and conflicts with high-level cross-modal alignment. This paper proposes a training-efficient method for temporal-sensitive VFMs that integrates the benefits of existing methods. To increase data efficiency, we mask out most of the low-semantics video tokens, but selectively align the unmasked tokens with IFM, which serves as the UnMasked Teacher (UMT). By providing semantic guidance, our method enables faster convergence and multimodal friendliness. With a progressive pre-training framework, our model can handle various tasks including scene-related, temporal-related, and complex video-language understanding. Using only public sources for pre-training in 6 days on 32 A100 GPUs, our scratch-built ViT-L/16 achieves state-of-the-art performances on various video tasks.

</details>

---

## 41. MAtch, eXpand and Improve: Unsupervised Finetuning for Zero-Shot Action Recognition with Language Knowledge

- [ ] MAtch, eXpand and Improve: Unsupervised Finetuning for Zero-Shot Action Recognition with Language Knowledge | https://openaccess.thecvf.com/content/ICCV2023/html/Lin_MAtch_eXpand_and_Improve_Unsupervised_Finetuning_for_Zero-Shot_Action_Recognition_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lin_MAtch_eXpand_and_Improve_Unsupervised_Finetuning_for_Zero-Shot_Action_Recognition_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large scale Vision-Language (VL) models have shown tremendous success in aligning representations between visual and text modalities. This enables remarkable progress in zero-shot recognition, image generation & editing, and many other exciting tasks. However, VL models tend to over-represent objects while paying much less attention to verbs, and require additional tuning on video data for best zero-shot action recognition performance. While previous work relied on large-scale, fully-annotated data, in this work we propose an unsupervised approach. We adapt a VL model for zero-shot and few-shot action recognition using a collection of unlabeled videos and an unpaired action dictionary. Based on that, we leverage Large Language Models and VL models to build a text bag for each unlabeled video via matching, text expansion and captioning. We use those bags in a Multiple Instance Learning setup to adapt an image-text backbone to video data. Although finetuned on unlabeled video data, our resulting models demonstrate high transferability to numerous unseen zero-shot downstream tasks, improving the base VL model performance by up to 14%, and even comparing favorably to fully-supervised baselines in both zero-shot and few-shot video recognition transfer. The code is provided in supplementary and will be released upon acceptance.

</details>

---

## 42. SMAUG: Sparse Masked Autoencoder for Efficient Video-Language Pre-Training

- [ ] SMAUG: Sparse Masked Autoencoder for Efficient Video-Language Pre-Training | https://openaccess.thecvf.com/content/ICCV2023/html/Lin_SMAUG_Sparse_Masked_Autoencoder_for_Efficient_Video-Language_Pre-Training_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lin_SMAUG_Sparse_Masked_Autoencoder_for_Efficient_Video-Language_Pre-Training_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-training is crucial for learning powerful multi-modal representation. However, it typically requires a massive amount of computation. In this paper, we develop SMAUG, an efficient pre-training framework for video-language models. The foundation component in SMAUG is masked autoencoders. Different from prior works which only mask textual inputs, our masking strategy considers both visual and textual modalities, providing a better cross-modal alignment and saving more pre-training costs. On top of that, we introduce a space-time token sparsification module, which leverages context information to further select only "important" spatial regions and temporal frames for pre-training. Coupling all these designs allows our method to enjoy both competitive performances on text-to-video retrieval and video question answering tasks, and much less pre-training costs by 1.9x or more. For example, our SMAUG only needs  50 NVIDIA A6000 GPU hours for pre-training to attain competitive performances on these two video-language tasks across six popular benchmarks.

</details>

---

## 43. UniVTG: Towards Unified Video-Language Temporal Grounding

- [ ] UniVTG: Towards Unified Video-Language Temporal Grounding | https://openaccess.thecvf.com/content/ICCV2023/html/Lin_UniVTG_Towards_Unified_Video-Language_Temporal_Grounding_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lin_UniVTG_Towards_Unified_Video-Language_Temporal_Grounding_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video Temporal Grounding (VTG), which aims to ground target clips from videos (such as consecutive intervals or disjoint shots) according to custom language queries (e.g., sentences or words), is key for video browsing on social media. Most methods in this direction develop task-specific models that are trained with type-specific labels, such as moment retrieval (time interval) and highlight detection (worthiness curve), which limits their abilities to generalize to various VTG tasks and labels. In this paper, we propose to Unify the diverse VTG labels and tasks, dubbed UniVTG, along three directions: Firstly, we revisit a wide range of VTG labels and tasks and define a unified formulation. Based on this, we develop data annotation schemes to create scalable pseudo supervision. Secondly, we develop an effective and flexible grounding model capable of addressing each task and making full use of each label. Lastly, thanks to the unified framework, we are able to unlock temporal grounding pretraining from large-scale diverse labels and develop stronger grounding abilities e.g., zero-shot grounding. Extensive experiments on three tasks (moment retrieval, highlight detection and video summarization) across seven datasets (QVHighlights, Charades-STA, TACoS, Ego4D, YouTube Highlights, TV-Sum, and QFVS) demonstrate the effectiveness and flexibility of our proposed framework. The codes are available at https://github.com/showlab/UniVTG.

</details>

---

## 44. Bird's-Eye-View Scene Graph for Vision-Language Navigation

- [ ] Bird's-Eye-View Scene Graph for Vision-Language Navigation | https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Birds-Eye-View_Scene_Graph_for_Vision-Language_Navigation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Birds-Eye-View_Scene_Graph_for_Vision-Language_Navigation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language navigation (VLN), which entails an agent to navigate 3D environments following human instructions, has shown great advances. However, current agents are built upon panoramic observations, which hinders their ability to perceive 3D scene geometry and easily leads to ambiguous selection of panoramic view. To address these limitations, we present a BEV Scene Graph (BSG), which leverages multi-step BEV representations to encode scene layouts and geometric cues of indoor environment under the supervision of 3D detection. During navigation, BSG builds a local BEV representation at each step and maintains a BEV-based global scene map, which stores and organizes all the online collected local BEV representations according to their topological relations. Based on BSG, the agent predicts a local BEV grid-level decision score and a global graph-level decision score, combined with a subview selection score on panoramic views, for more accurate action prediction. Our approach significantly outperforms state-of-the-art methods on REVERIE, R2R, and R4R, showing the potential of BEV perception in VLN.

</details>

---

## 45. Task-Oriented Multi-Modal Mutual Leaning for Vision-Language Models

- [ ] Task-Oriented Multi-Modal Mutual Leaning for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Long_Task-Oriented_Multi-Modal_Mutual_Leaning_for_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Long_Task-Oriented_Multi-Modal_Mutual_Leaning_for_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has become one of the most efficient paradigms for adapting large pre-trained vision-language models to downstream tasks. Current state-of-the-art methods, like CoOp and ProDA, tend to adopt soft prompts to learn an appropriate prompt for each specific task. Recent CoCoOp further boosts the base-to-new generalization performance via an image-conditional prompt. However, it directly fuses identical image semantics to prompts of different labels and significantly weakens the discrimination among different classes as shown in our experiments. Motivated by this observation, we first propose a class-aware text prompt (CTP) to enrich generated prompts with label-related image information. Unlike CoCoOp, CTP can effectively involve image semantics and avoid introducing extra ambiguities into different prompts. On the other hand, instead of reserving the complete image representations, we propose text-guided feature tuning (TFT) to make the image branch attend to class-related representation. A contrastive loss is employed to align such augmented text and image representations on downstream tasks. In this way, the image-to-text CTP and text-to-image TFT can be mutually promoted to enhance the adaptation of VLMs for downstream tasks. Extensive experiments demonstrate that our method outperforms the existing methods by a significant margin. Especially, compared to CoCoOp, we achieve an average improvement of 4.03% on new classes and 3.19% on harmonic-mean over eleven classification benchmarks.

</details>

---

## 46. Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models

- [ ] Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models | https://openaccess.thecvf.com/content/ICCV2023/html/Lu_Set-level_Guidance_Attack_Boosting_Adversarial_Transferability_of_Vision-Language_Pre-training_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Lu_Set-level_Guidance_Attack_Boosting_Adversarial_Transferability_of_Vision-Language_Pre-training_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training (VLP) models have shown vulnerability to adversarial examples in multimodal tasks. Furthermore, malicious adversaries can be deliberately transferred to attack other black-box models. However, existing work has mainly focused on investigating white-box attacks. In this paper, we present the first study to investigate the adversarial transferability of recent VLP models. We observe that existing methods exhibit much lower transferability, compared to the strong attack performance in white-box settings. The transferability degradation is partly caused by the under-utilization of cross-modal interactions. Particularly, unlike unimodal learning, VLP models rely heavily on cross-modal interactions and the multimodal alignments are many-to-many, e.g., an image can be described in various natural languages. To this end, we propose a highly transferable Set-level Guidance Attack (SGA) that thoroughly leverages modality interactions and incorporates alignment-preserving augmentation with cross-modal guidance. Experimental results demonstrate that SGA could generate adversarial examples that can strongly transfer across different VLP models on multiple downstream vision-language tasks. On image-text retrieval, SGA significantly enhances the attack success rate for transfer attacks from ALBEF to TCL by a large margin (at least 9.78% and up to 30.21%), compared to the state-of-the-art. Our code is available at https://github.com/Zoky-2020/SGA.

</details>

---

## 47. Spectrum-guided Multi-granularity Referring Video Object Segmentation

- [ ] Spectrum-guided Multi-granularity Referring Video Object Segmentation | https://openaccess.thecvf.com/content/ICCV2023/html/Miao_Spectrum-guided_Multi-granularity_Referring_Video_Object_Segmentation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Miao_Spectrum-guided_Multi-granularity_Referring_Video_Object_Segmentation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Current referring video object segmentation (R-VOS) techniques extract conditional kernels from encoded (low-resolution) vision-language features to segment the decoded high-resolution features. We discovered that this causes significant feature drift, which the segmentation kernels struggle to perceive during the forward computation. This negatively affects the ability of segmentation kernels. To address the drift problem, we propose a Spectrum-guided Multi-granularity (SgMg) approach, which performs direct segmentation on the encoded features and employs visual details to further optimize the masks. In addition, we propose Spectrum-guided Cross-modal Fusion (SCF) to perform intra-frame global interactions in the spectral domain for effective multimodal representation. Finally, we extend SgMg to perform multi-object R-VOS, a new paradigm that enables simultaneous segmentation of multiple referred objects in a video. This not only makes R-VOS faster, but also more practical. Extensive experiments show that SgMg achieves state-of-the-art performance on four video benchmark datasets, outperforming the nearest competitor by 2.8% points on Ref-YouTube-VOS. Our extended SgMg enables multi-object R-VOS, runs about 3 times faster while maintaining satisfactory performance. Code is available at https://github.com/bo-miao/SgMg.

</details>

---

## 48. Verbs in Action: Improving Verb Understanding in Video-Language Models

- [ ] Verbs in Action: Improving Verb Understanding in Video-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Momeni_Verbs_in_Action_Improving_Verb_Understanding_in_Video-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Momeni_Verbs_in_Action_Improving_Verb_Understanding_in_Video-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Understanding verbs is crucial to modelling how people and objects interact with each other and the environment through space and time. Recently, state-of-the-art video-language models based on CLIP have been shown to have limited verb understanding and to rely extensively on nouns, restricting their performance in real-world video applications that require action and temporal understanding. In this work, we improve verb understanding for CLIP-based video-language models by proposing a new Verb-Focused Contrastive (VFC) framework. This consists of two main components: (1) leveraging pretrained large language models (LLMs) to create hard negatives for cross-modal contrastive learning, together with a calibration strategy to balance the occurrence of concepts in positive and negative pairs; and (2) enforcing a fine-grained, verb phrase alignment loss. Our method achieves state-of-the-art results for zero-shot performance on three downstream tasks that focus on verb understanding, including video-text matching, video question-answering and video classification; while maintaining performance on noun-focused settings. To the best of our knowledge, this is the first work which proposes a method to alleviate the verb understanding problem, and does not simply highlight it. Our code is publicly available.

</details>

---

## 49. Unsupervised 3D Perception with 2D Vision-Language Distillation for Autonomous Driving

- [ ] Unsupervised 3D Perception with 2D Vision-Language Distillation for Autonomous Driving | https://openaccess.thecvf.com/content/ICCV2023/html/Najibi_Unsupervised_3D_Perception_with_2D_Vision-Language_Distillation_for_Autonomous_Driving_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Najibi_Unsupervised_3D_Perception_with_2D_Vision-Language_Distillation_for_Autonomous_Driving_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Closed-set 3D perception models trained on only a pre-defined set of object categories can be inadequate for safety critical applications such as autonomous driving where new object types can be encountered after deployment. In this paper, we present a multi-modal auto labeling pipeline capable of generating amodal 3D bounding boxes and tracklets for training models on open-set categories without 3D human labels. Our pipeline exploits motion cues inherent in point cloud sequences in combination with the freely available 2D image-text pairs to identify and track all traffic participants. Compared to the recent studies in this domain, which can only provide class-agnostic auto labels limited to moving objects, our method can handle both static and moving objects in the unsupervised manner and is able to output open-vocabulary semantic labels thanks to the proposed vision-language knowledge distillation. Experiments on the Waymo Open Dataset show that our approach outperforms the prior work by significant margins on various unsupervised 3D perception tasks.

</details>

---

## 50. Black Box Few-Shot Adaptation for Vision-Language Models

- [ ] Black Box Few-Shot Adaptation for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language (V-L) models trained with contrastive learning to align the visual and language modalities have been shown to be strong few-shot learners. Soft prompt learning is the method of choice for few-shot downstream adaption aiming to bridge the modality gap caused by the distribution shift induced by the new domain. While parameter-efficient, prompt learning still requires access to the model weights and can be computationally infeasible for large models with billions of parameters. To address these shortcomings, in this work, we describe a black-box method for V-L few-shot adaptation that (a) operates on pre-computed image and text features and hence works without access to the model's weights, (b) it is orders of magnitude faster at training time, (c) it is amenable to both supervised and unsupervised training, and (d) it can be even used to align image and text features computed from uni-modal models. To achieve this, we propose Linear Feature Alignment (LFA), a simple linear approach for V-L re-alignment in the target domain. LFA is initialized from a closed-form solution to a least-squares problem and then it is iteratively updated by minimizing a re-ranking loss. Despite its simplicity, our approach can even surpass soft-prompt learning methods as shown by extensive experiments on 11 image and 2 video datasets.

</details>

---

## 51. Teaching CLIP to Count to Ten

- [ ] Teaching CLIP to Count to Ten | https://openaccess.thecvf.com/content/ICCV2023/html/Paiss_Teaching_CLIP_to_Count_to_Ten_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Paiss_Teaching_CLIP_to_Count_to_Ten_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models, such as CLIP, learn robust representations of text and images, facilitating advances in many downstream tasks, including zero-shot classification and text-to-image generation. However, these models have several well-documented limitations. They fail to encapsulate compositional concepts, such as counting. To the best of our knowledge, this work is the first to extend CLIP to handle object counting. We introduce a simple yet effective method to improve the quantitative understanding of vision-language models, while maintaining their overall performance on common benchmarks.
 Our method automatically augments image captions to create hard negative samples that differ from the original captions by only the number of objects. For example, an image of three dogs can be contrasted with the negative caption "Six dogs playing in the yard". A dedicated loss encourages discrimination between the correct caption and its negative variant. 
 In addition, we introduce CountBench, a new benchmark for evaluating a model's understanding of object counting, and demonstrate significant improvement over baseline models on this task. Furthermore, we leverage our improved CLIP representations for text-conditioned image generation, and show that our model can produce specific counts of objects more reliably than existing ones.

</details>

---

## 52. EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone

- [ ] EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone | https://openaccess.thecvf.com/content/ICCV2023/html/Pramanick_EgoVLPv2_Egocentric_Video-Language_Pre-training_with_Fusion_in_the_Backbone_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Pramanick_EgoVLPv2_Egocentric_Video-Language_Pre-training_with_Fusion_in_the_Backbone_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-training (VLP) has become increasingly important due to its ability to generalize to various vision and language tasks. However, existing egocentric VLP frameworks utilize separate video and language encoders and learn task-specific cross-modal information only during fine-tuning, limiting the development of a unified system. In this work, we introduce the second generation of egocentric video-language pre-training (EgoVLPv2), a significant improvement from the previous generation, by incorporating cross-modal fusion directly into the video and language backbones. EgoVLPv2 learns strong video-text representation during pre-training and reuses the cross-modal attention modules to support different downstream tasks in a flexible and efficient manner, reducing fine-tuning costs. Moreover, our proposed fusion in the backbone strategy is more lightweight and compute-efficient than stacking additional fusion-specific layers. Extensive experiments on a wide range of VL tasks demonstrate the effectiveness of EgoVLPv2 by achieving consistent state-of-the-art performance over strong baselines across all downstream.

</details>

---

## 53. Decouple Before Interact: Multi-Modal Prompt Learning for Continual Visual Question Answering

- [ ] Decouple Before Interact: Multi-Modal Prompt Learning for Continual Visual Question Answering | https://openaccess.thecvf.com/content/ICCV2023/html/Qian_Decouple_Before_Interact_Multi-Modal_Prompt_Learning_for_Continual_Visual_Question_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Qian_Decouple_Before_Interact_Multi-Modal_Prompt_Learning_for_Continual_Visual_Question_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In the real world, a desirable Visual Question Answering model is expected to provide correct answers to new questions and images in a continual setting (recognized as CL-VQA). However, existing works formulate CLVQA from a vision-only or language-only perspective, and straightforwardly apply the uni-modal continual learning (CL) strategies to this multi-modal task, which is improper and suboptimal. On the one hand, such a partial formulation may result in limited evaluations. On the other hand, neglecting the interactions between modalities will lead to poor performance. To tackle these challenging issues, we propose a comprehensive formulation for CL-VQA from the perspective of multi-modal vision-language fusion. Based on our formulation, we further propose MulTi-Modal PRompt LearnIng with DecouPLing bEfore InTeraction (TRIPLET), a novel approach that builds on a pre-trained vision-language model and consists of decoupled prompts and prompt interaction strategies to capture the complex interactions between modalities. In particular, decoupled prompts contain learnable parameters that are decoupled w.r.t different aspects, and the prompt interaction strategies are in charge of modeling interactions between inputs and prompts. Additionally, we build two CL-VQA benchmarks for a more comprehensive evaluation. Extensive experiments demonstrate that our TRIPLET outperforms state-of-the-art methods in both uni-modal and multi-modal continual settings for CL-VQA.

</details>

---

## 54. Perceptual Grouping in Contrastive Vision-Language Models

- [ ] Perceptual Grouping in Contrastive Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Ranasinghe_Perceptual_Grouping_in_Contrastive_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Ranasinghe_Perceptual_Grouping_in_Contrastive_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in zero-shot image recognition suggest that vision-language models learn generic visual representations with a high degree of semantic information that may be arbitrarily probed with natural language phrases. Understanding an image, however, is not just about understanding what content resides within an image, but importantly, where that content resides. In this work we examine how well vision-language models are able to understand where objects reside within an image and group together visually related parts of the imagery. We demonstrate how contemporary vision and language representation learning models based on contrastive losses and large web-based data capture limited object localization information. We propose a minimal set of modifications that results in models that uniquely learn both semantic and spatial information. We measure this performance in terms of zero-shot image recognition, unsupervised bottom-up and top-down semantic segmentations, as well as robustness analyses. We find that the resulting model achieves state-of-the-art results in terms of unsupervised segmentation, and demonstrate that the learned representations are uniquely robust to spurious correlations in datasets designed to probe the causal behavior of vision models.

</details>

---

## 55. Waffling Around for Performance: Visual Classification with Random Words and Broad Concepts

- [ ] Waffling Around for Performance: Visual Classification with Random Words and Broad Concepts | https://openaccess.thecvf.com/content/ICCV2023/html/Roth_Waffling_Around_for_Performance_Visual_Classification_with_Random_Words_and_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Roth_Waffling_Around_for_Performance_Visual_Classification_with_Random_Words_and_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The visual classification performance of vision-language models such as CLIP has been shown to benefit from additional semantic knowledge from large language models (LLMs) such as GPT-3. In particular, averaging over LLM-generated class descriptors, e.g. "waffle, which has a round shape", can notably improve generalization performance. In this work, we critically study this behavior and propose WaffleCLIP, a framework for zero-shot visual classification which simply replaces LLM-generated descriptors with random character and word descriptors. Without querying external models, we achieve comparable performance gains on a large number of visual classification tasks. This allows WaffleCLIP to both serve as a low-cost 
 alternative, as well as a sanity check for any future LLM-based vision-language model extensions. We conduct an extensive experimental study on the impact and shortcomings of additional semantics introduced with LLM-generated descriptors, and showcase how - if available - semantic context is better leveraged by querying LLMs for high-level concepts, which we show can be done to jointly resolve potential class name ambiguities. Code is available here: https://github.com/ExplainableML/WaffleCLIP.

</details>

---

## 56. HiVLP: Hierarchical Interactive Video-Language Pre-Training

- [ ] HiVLP: Hierarchical Interactive Video-Language Pre-Training | https://openaccess.thecvf.com/content/ICCV2023/html/Shao_HiVLP_Hierarchical_Interactive_Video-Language_Pre-Training_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Shao_HiVLP_Hierarchical_Interactive_Video-Language_Pre-Training_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-Language Pre-training (VLP) has become one of the most popular research topics in deep learning. However, compared to image-language pre-training, VLP has lagged far behind due to the lack of large amounts of video-text pairs. In this work, we train a VLP model with a hybrid of image-text and video-text pairs, which significantly outperforms pre-training with only the video-text pairs. Besides, existing methods usually model the cross-modal interaction using cross-attention between single-scale visual tokens and textual tokens. These visual features are either of low resolutions lacking fine-grained information, or of high resolutions without high-level semantics. To address the issue, we propose Hierarchical interactive Video-Language Pre-training (HiVLP) that efficiently uses a hierarchical visual feature group for multi-modal cross-attention during pre-training. In the hierarchical framework, low-resolution features are learned with focus on more global high-level semantic information, while high-resolution features carry fine-grained details. As a result, HiVLP has the ability to effectively learn both the global and fine-grained representations to achieve better alignment between video and text inputs. Furthermore, we design a hierarchical multi-scale vision contrastive loss for self-supervised learning to boost the interaction between them. Experimental results show that HiVLP establishes new state-of-the-art results in three downstream tasks, text-video retrieval, video-text retrieval, and video captioning.

</details>

---

## 57. LoGoPrompt: Synthetic Text Images Can Be Good Visual Prompts for Vision-Language Models

- [ ] LoGoPrompt: Synthetic Text Images Can Be Good Visual Prompts for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Shi_LoGoPrompt_Synthetic_Text_Images_Can_Be_Good_Visual_Prompts_for_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Shi_LoGoPrompt_Synthetic_Text_Images_Can_Be_Good_Visual_Prompts_for_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Prompt engineering is a powerful tool used to enhance the performance of pre-trained models on downstream tasks. For example, providing the prompt "Let's think step by step" improved GPT-3's reasoning accuracy to 63% on MutiArith while prompting "a photo of" filled with a class name enables CLIP to achieve 80% zero-shot accuracy on ImageNet. While previous research has explored prompt learning for the visual modality, analyzing what constitutes a good visual prompt specifically for image recognition is limited. In addition, existing visual prompt tuning methods' generalization ability is worse than text-only prompting tuning. This paper explores our key insight: synthetic text images are good visual prompts for vision-language models! To achieve that, we propose our LoGoPrompt, which reformulates the classification objective to the visual prompt selection and addresses the chicken-and-egg challenge of first adding synthetic text images as class-wise visual prompts or predicting the class first. Without any trainable visual prompt parameters, experimental results on 16 datasets demonstrate that our method consistently outperforms state-of-the-art methods in few-shot learning, base-to-new generalization, and domain generalization. The code will be publicly available upon publication.

</details>

---

## 58. EdaDet: Open-Vocabulary Object Detection Using Early Dense Alignment

- [ ] EdaDet: Open-Vocabulary Object Detection Using Early Dense Alignment | https://openaccess.thecvf.com/content/ICCV2023/html/Shi_EdaDet_Open-Vocabulary_Object_Detection_Using_Early_Dense_Alignment_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Shi_EdaDet_Open-Vocabulary_Object_Detection_Using_Early_Dense_Alignment_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models such as CLIP have boosted the performance of open-vocabulary object detection, where the detector is trained on base categories but required to detect novel categories. Existing methods leverage CLIP's strong zero-shot recognition ability to align object-level embeddings with textual embeddings of categories. However, we observe that using CLIP for object-level alignment results in overfitting to base categories, i.e., novel categories most similar to base categories have particularly poor performance as they are recognized as similar base categories. In this paper, we first identify that the loss of critical fine-grained local image semantics hinders existing methods from attaining strong base-to-novel generalization. Then, we propose Early Dense Alignment (EDA) to bridge the gap between generalizable local semantics and object-level prediction. In EDA, we use object-level supervision to learn the dense-level rather than object-level alignment to maintain the local fine-grained semantics. Extensive experiments demonstrate our superior performance to competing approaches under the same strict setting and without using external training resources, i.e., improving the +8.4% novel box AP50 on COCO and +3.9% rare mask AP on LVIS.

</details>

---

## 59. What does CLIP know about a red circle? Visual prompt engineering for VLMs

- [ ] What does CLIP know about a red circle? Visual prompt engineering for VLMs | https://openaccess.thecvf.com/content/ICCV2023/html/Shtedritski_What_does_CLIP_know_about_a_red_circle_Visual_prompt_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Shtedritski_What_does_CLIP_know_about_a_red_circle_Visual_prompt_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale Vision-Language Models, such as CLIP, learn powerful image-text representations that have found numerous applications, from zero-shot classification to text-to-image generation. Despite that, their capabilities for solving novel discriminative tasks via prompting fall behind those of large language models, such as GPT-3. Here we explore the idea of visual prompt engineering for solving computer vision tasks beyond classification by editing in image space instead of text. In particular, we discover an emergent ability of CLIP, where, by simply drawing a red circle around an object, we can direct the model's attention to that region, while also maintaining global information. We show the power of this simple approach by achieving state-of-the-art in zero-shot referring expressions comprehension and strong performance in keypoint localization tasks. Finally, we draw attention to some potential ethical concerns of large language-vision models.

</details>

---

## 60. eP-ALM: Efficient Perceptual Augmentation of Language Models

- [ ] eP-ALM: Efficient Perceptual Augmentation of Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Shukor_eP-ALM_Efficient_Perceptual_Augmentation_of_Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Shukor_eP-ALM_Efficient_Perceptual_Augmentation_of_Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have so far impressed the world, with unprecedented capabilities that emerge in models at large scales. On the vision side, transformer models (i.e., ViT) are following the same trend, achieving the best performance on challenging benchmarks. With the abundance of such unimodal models, a natural question arises; do we need also to follow this trend to tackle multimodal tasks? In this work, we propose to rather direct effort to efficient adaptations of existing models, and propose to augment Language Models with perception. Existing approaches for adapting pretrained models for vision-language tasks still rely on several key components that hinder their efficiency. In particular, they still train a large number of parameters, rely on large multimodal pretraining, use encoders (e.g., CLIP) trained on huge image-text datasets, and add significant inference overhead. In addition, most of these approaches have focused on Zero-Shot and In Context Learning, with little to no effort on direct finetuning. We investigate the minimal computational effort needed to adapt unimodal models for multimodal tasks and propose a new challenging setup, alongside different approaches, that efficiently adapts unimodal pretrained models. 
 We show that by freezing more than 99% of total parameters, training only one linear projection layer, and prepending only one trainable token, our approach (dubbed eP-ALM) significantly outperforms other baselines on VQA and Captioning across Image, Video, and Audio modalities, following the proposed setup. The code is available here: https://github.com/mshukor/eP-ALM.

</details>

---

## 61. In-Style: Bridging Text and Uncurated Videos with Style Transfer for Text-Video Retrieval

- [ ] In-Style: Bridging Text and Uncurated Videos with Style Transfer for Text-Video Retrieval | https://openaccess.thecvf.com/content/ICCV2023/html/Shvetsova_In-Style_Bridging_Text_and_Uncurated_Videos_with_Style_Transfer_for_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Shvetsova_In-Style_Bridging_Text_and_Uncurated_Videos_with_Style_Transfer_for_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale noisy web image-text datasets have been proven to be efficient for learning robust vision-language models. However, to transfer them to the task of video retrieval, models still need to be fine-tuned on hand-curated paired text-video data to adapt to the diverse styles of video descriptions. To address this problem without the need for hand-annotated pairs, we propose a new setting, text-video retrieval with uncurated & unpaired data, that uses only text queries together with uncurated web videos during training without any paired text-video data. To this end, we propose an approach, In-Style, that learns the style of the text queries and transfers it to uncurated web videos. Moreover, to improve generalization, we show that one model can be trained with multiple text styles. To this end, we introduce a multi-style contrastive training procedure, that improves the generalizability over several datasets simultaneously. We evaluate our model on retrieval performance over multiple datasets to demonstrate the advantages of our style transfer framework on the new task of uncurated & unpaired text-video retrieval and improve state-of-the-art performance on zero-shot text-video retrieval.

</details>

---

## 62. Blending-NeRF: Text-Driven Localized Editing in Neural Radiance Fields

- [ ] Blending-NeRF: Text-Driven Localized Editing in Neural Radiance Fields | https://openaccess.thecvf.com/content/ICCV2023/html/Song_Blending-NeRF_Text-Driven_Localized_Editing_in_Neural_Radiance_Fields_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Song_Blending-NeRF_Text-Driven_Localized_Editing_in_Neural_Radiance_Fields_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Text-driven localized editing of 3D objects is particularly difficult as locally mixing the original 3D object with the intended new object and style effects without distorting the object's form is not a straightforward process. To address this issue, we propose a novel NeRF-based model, Blending-NeRF, which consists of two NeRF networks: pretrained NeRF and editable NeRF. Additionally, we introduce new blending operations that allow Blending-NeRF to properly edit target regions which are localized by text. By using a pretrained vision-language aligned model, CLIP, we guide Blending-NeRF to add new objects with varying colors and densities, modify textures, and remove parts of the original object. Our extensive experiments demonstrate that Blending-NeRF produces naturally and locally edited 3D objects from various text prompts.

</details>

---

## 63. FLIP: Cross-domain Face Anti-spoofing with Language Guidance

- [ ] FLIP: Cross-domain Face Anti-spoofing with Language Guidance | https://openaccess.thecvf.com/content/ICCV2023/html/Srivatsan_FLIP_Cross-domain_Face_Anti-spoofing_with_Language_Guidance_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Srivatsan_FLIP_Cross-domain_Face_Anti-spoofing_with_Language_Guidance_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Face anti-spoofing (FAS) or presentation attack detection is an essential component of face recognition systems deployed in security-critical applications. Existing FAS methods have poor generalizability to unseen spoof types, camera sensors, and environmental conditions. Recently, vision transformer (ViT) models have been shown to be effective for the FAS task due to their ability to capture long-range dependencies among image patches. However, adaptive modules or auxiliary loss functions are often required to adapt pre-trained ViT weights learned on large-scale datasets such as ImageNet. In this work, we first show that initializing ViTs with multimodal (e.g., CLIP) pre-trained weights improves generalizability for the FAS task, which is in line with the zero-shot transfer capabilities of vision-language pre-trained (VLP) models. We then propose a novel approach for robust cross-domain FAS by grounding visual representations with the help of natural language. Specifically, we show that aligning the image representation with an ensemble of class descriptions (based on natural language semantics) improves FAS generalizability in low-data regimes. Finally, we propose a multimodal contrastive learning strategy to boost feature generalization further and bridge the gap between source and target domains. Extensive experiments on three standard protocols demonstrate that our method significantly outperforms the state-of-the-art methods, achieving better zero-shot transfer performance than five-shot transfer of "adaptive ViTs". Code: https://github.com/koushiksrivats/FLIP

</details>

---

## 64. DIME-FM : DIstilling Multimodal and Efficient Foundation Models

- [ ] DIME-FM : DIstilling Multimodal and Efficient Foundation Models | https://openaccess.thecvf.com/content/ICCV2023/html/Sun_DIME-FM__DIstilling_Multimodal_and_Efficient_Foundation_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Sun_DIME-FM__DIstilling_Multimodal_and_Efficient_Foundation_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Foundation Models (VLFM), such as CLIP, ALIGN and Florence, are trained on large private datasets of image-caption pairs and achieve superior transferability and robustness on downstream tasks, but they are difficult to use in many practical applications due to their large size, high latency and fixed architectures. Unfortunately, recent works show training a small custom VLFM for resource-limited applications is currently very difficult using public and smaller-scale data. In this paper, we introduce a new distillation mechanism (DIME-FM) that allows us to transfer the knowledge contained in large VLFMs to smaller, customized foundation models using a relatively small amount of inexpensive, unpaired images and sentences. We transfer the knowledge from the pre-trained CLIP-ViT-L/14 model to a ViT-B/32 model, with only 40M public images and 28.4M unpaired public sentences. The resulting model "Distill-ViT-B/32" rivals the CLIP-ViT-B/32 model pre-trained on its private WiT dataset (400M image-text pairs): Distill-ViT-B/32 achieves similar results in terms of zero-shot and linear-probing performance on both ImageNet and the ELEVATER (20 image classification tasks) benchmarks. It also displays comparable robustness when evaluated on five datasets with natural distribution shifts from ImageNet.

</details>

---

## 65. Linear Spaces of Meanings: Compositional Structures in Vision-Language Models

- [ ] Linear Spaces of Meanings: Compositional Structures in Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Trager_Linear_Spaces_of_Meanings_Compositional_Structures_in_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Trager_Linear_Spaces_of_Meanings_Compositional_Structures_in_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We investigate compositional structures in data embeddings from pre-trained vision-language models (VLMs). Traditionally, compositionality has been associated with algebraic operations on embeddings of words from a pre-existing vocabulary. In contrast, we seek to approximate representations from an encoder as combinations of a smaller set of vectors in the embedding space. These vectors can be seen as "ideal words" for generating concepts directly within embedding space of the model. We first present a framework for understanding compositional structures from a geometric perspective. We then explain what these compositional structures entail probabilistically in the case of VLM embeddings, providing intuitions for why they arise in practice. Finally, we empirically explore these structures in CLIP's embeddings and we evaluate their usefulness for solving different vision-language tasks such as classification, debiasing, and retrieval. Our results show that simple linear algebraic operations on embedding vectors can be used as compositional and interpretable methods for regulating the behavior of VLMs.

</details>

---

## 66. SuS-X: Training-Free Name-Only Transfer of Vision-Language Models

- [ ] SuS-X: Training-Free Name-Only Transfer of Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Udandarao_SuS-X_Training-Free_Name-Only_Transfer_of_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Udandarao_SuS-X_Training-Free_Name-Only_Transfer_of_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) has emerged as a simple yet effective way to train large-scale vision-language models. CLIP demonstrates impressive zero-shot classification and retrieval performance on diverse downstream tasks. However, to leverage its full potential, fine-tuning still appears to be necessary. Fine-tuning the entire CLIP model can be resource-intensive and unstable. Moreover, recent methods that aim to circumvent this need for fine-tuning still require access to images from the target distribution. In this paper, we pursue a different approach and explore the regime of training-free "name-only transfer" in which the only knowledge we possess about downstream tasks comprises the names of downstream target categories. We propose a novel method, SuS-X, consisting of two key building blocks--"SuS" and "TIP-X", that requires neither intensive fine-tuning nor costly labelled data. SuS-X achieves state-of-the-art (SoTA) zero-shot classification results on 19 benchmark datasets. We further show the utility of TIP-X in the training-free few-shot setting, where we again achieve SoTA results over strong training-free baselines.

</details>

---

## 67. ProbVLM: Probabilistic Adapter for Frozen Vison-Language Models

- [ ] ProbVLM: Probabilistic Adapter for Frozen Vison-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models (VLMs) like CLIP successfully find correspondences between images and text. Through the standard deterministic mapping process, an image or a text sample is mapped to a single vector in the embedding space. This is problematic: as multiple samples (images or text) can abstract the same concept in the physical world, deterministic embeddings do not reflect the inherent ambiguity in the embedding space. We propose ProbVLM, a probabilistic adapter that estimates probability distributions for the embeddings of pre-trained VLMs via inter/intra-modal alignment in a post-hoc manner without needing large-scale datasets or computing. On four challenging datasets, i.e., COCO, Flickr, CUB, and Oxford-flowers, we estimate the multi-modal embedding uncertainties for two VLMs, i.e., CLIP and BLIP, quantify the calibration of embedding uncertainties in retrieval tasks and show that ProbVLM outperforms other methods. Furthermore, we propose active learning and model selection as two real-world downstream tasks for VLMs and show that the estimated uncertainty aids both tasks. Lastly, we present a novel technique for visualizing the embedding distributions using a large-scale pre-trained latent diffusion model.

</details>

---

## 68. ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data

- [ ] ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data | https://openaccess.thecvf.com/content/ICCV2023/html/Varma_ViLLA_Fine-Grained_Vision-Language_Representation_Learning_from_Real-World_Data_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Varma_ViLLA_Fine-Grained_Vision-Language_Representation_Learning_from_Real-World_Data_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs), such as CLIP and ALIGN, are generally trained on datasets consisting of image-caption pairs obtained from the web. However, real-world multimodal datasets, such as healthcare data, are significantly more complex: each image (e.g. X-ray) is often paired with text (e.g. physician report) that describes many distinct attributes occurring in fine-grained regions of the image. We refer to these samples as exhibiting high pairwise complexity, since each image-text pair can be decomposed into a large number of region-attribute pairings. The extent to which VLMs can capture fine-grained relationships between image regions and textual attributes when trained on such data has not been previously evaluated. The first key contribution of this work is to demonstrate through systematic evaluations that as the pairwise complexity of the training dataset increases, standard VLMs struggle to learn region-attribute relationships, exhibiting performance degradations of up to 37% on retrieval tasks. In order to address this issue, we introduce ViLLA as our second key contribution. ViLLA, which is trained to capture fine-grained region-attribute relationships from complex datasets, involves two components: (a) a lightweight, self-supervised mapping model to decompose image-text samples into region-attribute pairs, and (b) a contrastive VLM to learn representations from generated region-attribute pairs. We demonstrate with experiments across four domains (synthetic, product, medical, and natural images) that ViLLA outperforms comparable VLMs on fine-grained reasoning tasks, such as zero-shot object detection (up to 3.6 AP50 points on COCO and 0.6 mAP points on LVIS) and retrieval (up to 14.2 R-Precision points).

</details>

---

## 69. DREAMWALKER: Mental Planning for Continuous Vision-Language Navigation

- [ ] DREAMWALKER: Mental Planning for Continuous Vision-Language Navigation | https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

VLN-CE is a recently released embodied task, where AI agents need to navigate a freely traversable environment to reach a distant target location, given language instructions. It poses great challenges due to the huge space of possible strategies. Driven by the belief that the ability to anticipate the consequences of future actions is crucial for the emergence of intelligent and interpretable planning behavior, we propose Dreamwalker --- a world model based VLN-CE agent. The world model is built to summarize the visual, topological, and dynamic properties of the complicated continuous environment into a discrete, structured, and compact representation. Dreamwalker can simulate and evaluate possible plans entirely in such internal abstract world, before executing costly actions. As opposed to existing model-free VLN-CE agents simply making greedy decisions in the real world, which easily results in shortsighted behaviors, Dreamwalker is able to make strategic planning through large amounts of "mental experiments." Moreover, the imagined future scenarios reflect our agent's intention, making its decision-making process more transparent. Extensive experiments and ablation studies on VLN-CE dataset confirm the effectiveness of the proposed approach and outline fruitful directions for future work. Our code will be released.

</details>

---

## 70. Equivariant Similarity for Vision-Language Foundation Models

- [ ] Equivariant Similarity for Vision-Language Foundation Models | https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Equivariant_Similarity_for_Vision-Language_Foundation_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Equivariant_Similarity_for_Vision-Language_Foundation_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This study explores the concept of equivariance in vision-language foundation models (VLMs), focusing specifically on the multimodal similarity function that is not only the major training objective but also the core delivery to support downstream tasks. Unlike the existing image-text similarity objective which only categorizes matched pairs as similar and unmatched as dissimilar, equivariance also requires similarity to vary faithfully according to the semantic changes. This allows VLMs to generalize better to nuanced and unseen multimodal compositions. However, modeling equivariance is challenging as the ground truth of semantic change is difficult to collect. For example, given an image-text pair about a dog, it is unclear to what extent the similarity changes when the pixel is changed from dog to cat? To this end, we propose EqSim, a regularization loss that can be efficiently calculated from any two matched training pairs and easily pluggable into existing image-text retrieval fine-tuning. Meanwhile, to further diagnose the equivariance of VLMs, we present a new challenging benchmark EqBen. Compared to the existing evaluation sets, EqBen is the first to focus on "visual-minimal change". Extensive experiments show the lack of equivariance in current VLMs and validate the effectiveness of EqSim.

</details>

---

## 71. Too Large; Data Reduction for Vision-Language Pre-Training

- [ ] Too Large; Data Reduction for Vision-Language Pre-Training | https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper examines the problems of severe image-text misalignment and high redundancy in the widely-used large-scale Vision-Language Pre-Training (VLP) datasets. To address these issues, we propose an efficient and straightforward Vision-Language learning algorithm called TL;DR which aims to compress the existing large VLP data into a small, high-quality set. Our approach consists of two major steps. First, a codebook-based encoder-decoder captioner is developed to select representative samples. Second, a new caption is generated to complement the original captions for selected samples, mitigating the text-image misalignment problem while maintaining uniqueness. As the result, TL;DR enables us to reduce the large dataset into a small set of high-quality data, which can serve as an alternative pre-training dataset. This algorithm significantly speeds up the time-consuming pretraining process. Specifically, TL;DR can compress the mainstream VLP datasets at a high ratio, e.g., reduce well-cleaned CC3M dataset from 2.8M to 0.67M ( 24%) and noisy YFCC15M from 15M to 2.5M ( 16.7%). Extensive experiments with three popular VLP models over seven downstream tasks show that VLP model trained on the compressed dataset provided by TL;DR can perform similar or even better results compared with training on the full-scale dataset.

</details>

---

## 72. ViLTA: Enhancing Vision-Language Pre-training through Textual Augmentation

- [ ] ViLTA: Enhancing Vision-Language Pre-training through Textual Augmentation | https://openaccess.thecvf.com/content/ICCV2023/html/Wang_ViLTA_Enhancing_Vision-Language_Pre-training_through_Textual_Augmentation_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_ViLTA_Enhancing_Vision-Language_Pre-training_through_Textual_Augmentation_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training (VLP) methods are blossoming recently, and its crucial goal is to jointly learn visual and textual features via a transformer-based architecture, demonstrating promising improvements on a variety of vision-language tasks. Prior arts usually focus on how to align visual and textual features, but strategies for improving the robustness of model and speeding up model convergence are left insufficiently explored.
 
 In this paper, we propose a novel method ViLTA, comprising of two components to further facilitate the model to learn fine-grained representations among image-text pairs. For Masked Language Modeling (MLM), we propose a cross-distillation method to generate soft labels to enhance the robustness of model, which alleviates the problem of treating synonyms of masked words as negative samples in one-hot labels. For Image-Text Matching (ITM), we leverage the current language encoder to synthesize hard negatives based on the context of language input, encouraging the model to learn high-quality representations by increasing the difficulty of the ITM task. By leveraging the above techniques, our ViLTA can achieve better performance on various vision-language tasks. Extensive experiments on benchmark datasets demonstrate that the effectiveness of ViLTA and its promising potential for vision-language pre-training.

</details>

---

## 73. Grounded Image Text Matching with Mismatched Relation Reasoning

- [ ] Grounded Image Text Matching with Mismatched Relation Reasoning | https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Grounded_Image_Text_Matching_with_Mismatched_Relation_Reasoning_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Grounded_Image_Text_Matching_with_Mismatched_Relation_Reasoning_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces Grounded Image Text Matching with Mismatched Relation (GITM-MR), a novel visual-linguistic joint task that evaluates the relation understanding capabilities of transformer-based pre-trained models. GITM-MR requires a model to first determine if an expression describes an image, then localize referred objects or ground the mismatched parts of the text. We provide a benchmark for evaluating vision-language (VL) models on this task, with a focus on the challenging settings of limited training data and out-of-distribution sentence lengths. Our evaluation demonstrates that pre-trained VL models often lack data efficiency and length generalization ability. To address this, we propose the Relation-sensitive Correspondence Reasoning Network (RCRN), which incorporates relation-aware reasoning via bi-directional message propagation guided by language structure. Our RCRN can be interpreted as a modular program and delivers strong performance in terms of both length generalization and data efficiency. The code and data are available on https://github.com/SHTUPLUS/GITM-MR.

</details>

---

## 74. Open Set Video HOI detection from Action-Centric Chain-of-Look Prompting

- [ ] Open Set Video HOI detection from Action-Centric Chain-of-Look Prompting | https://openaccess.thecvf.com/content/ICCV2023/html/Xi_Open_Set_Video_HOI_detection_from_Action-Centric_Chain-of-Look_Prompting_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Xi_Open_Set_Video_HOI_detection_from_Action-Centric_Chain-of-Look_Prompting_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Human-Object Interaction (HOI) detection is essential for understanding and modeling real-world events. Existing works on HOI detection mainly focus on static images and a closed setting, where all HOI classes are provided in the training set. In comparison, detecting HOIs in videos in open set scenarios is more challenging. First, under open set circumstances, HOI detectors are expected to hold strong generalizability to recognize unseen HOIs not included in the training data. Second, accurately capturing temporal contextual information from videos is difficult, but it is crucial for detecting temporal-related actions
 such as open, close, pull, push. To this end, we propose ACoLP, a model of Action-centric Chain-of-Look Prompting for open set video HOI detection. ACoLP regards actions as the carrier of semantics in videos, which captures the essential semantic information across frames. To make the model generalizable on unseen classes, inspired by the chain-of-thought prompting in natural language processing, we introduce the chain-of-look prompting scheme that decomposes prompt generation from large-scale vision-language model into a series of intermediate visual reasoning steps. Consequently, our model captures
 complex visual reasoning processes underlying the HOI events in videos, providing essential guidance for detecting unseen classes. Extensive experiments on two video HOI datasets, VidHOI and CAD120, demonstrate that ACoLP achieves competitive performance compared with the state-of-the-art methods in the conventional closed setting, and outperforms existing methods by a large margin in the open set setting. Our code is avaliable at https://github.
 com/southnx/ACoLP.

</details>

---

## 75. Why Is Prompt Tuning for Vision-Language Models Robust to Noisy Labels?

- [ ] Why Is Prompt Tuning for Vision-Language Models Robust to Noisy Labels? | https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Why_Is_Prompt_Tuning_for_Vision-Language_Models_Robust_to_Noisy_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Why_Is_Prompt_Tuning_for_Vision-Language_Models_Robust_to_Noisy_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models such as CLIP learn a generic text-image embedding from large-scale training data. A vision-language model can be adapted to a new classification task through few-shot prompt tuning. We find that such prompt tuning process is highly robust to label noises. This intrigues us to study the key reasons contributing to the robustness of the prompt tuning paradigm. We conducted extensive experiments to explore this property and find the key factors are: 1. the fixed classname tokens provide a strong regularization to the optimization of the model, reducing gradients induced by the noisy samples; 2. the powerful pre-trained image-text embedding that is learned from diverse and generic web data provides strong prior knowledge for image classification. Further, we demonstrate that noisy zero-shot predictions from CLIP can be used to tune its own prompt, significantly enhancing prediction accuracy in the unsupervised setting.

</details>

---

## 76. CiT: Curation in Training for Effective Vision-Language Data

- [ ] CiT: Curation in Training for Effective Vision-Language Data | https://openaccess.thecvf.com/content/ICCV2023/html/Xu_CiT_Curation_in_Training_for_Effective_Vision-Language_Data_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Xu_CiT_Curation_in_Training_for_Effective_Vision-Language_Data_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models are generally applicable to many downstream tasks, but come at an exorbitant training cost that only large institutions can afford. This paper trades generality for efficiency and presents Curation in Training (CiT), a simple and efficient vision-text learning algorithm that couples a data objective into training. CiT automatically yields quality data to speed-up contrastive image-text training and alleviates the need for an offline data filtering pipeline, allowing broad data sources (including raw image-text pairs from the web). CiT contains two loops: an outer loop curating the training data and an inner loop consuming the curated training data. The text encoder connects the two loops. Given metadata for tasks of interest, e.g., class names, and a large pool of image-text pairs, CiT alternatively selects relevant training data from the pool by measuring the similarity of their text embeddings and embeddings of the metadata. In our experiments, we observe that CiT can speed up training by over an order of magnitude, especially if the raw data size is large.

</details>

---

## 77. Learning Concise and Descriptive Attributes for Visual Recognition

- [ ] Learning Concise and Descriptive Attributes for Visual Recognition | https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Learning_Concise_and_Descriptive_Attributes_for_Visual_Recognition_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Learning_Concise_and_Descriptive_Attributes_for_Visual_Recognition_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in foundation models present new opportunities for interpretable visual recognition -- one can first query Large Language Models (LLMs) to obtain a set of attributes that describe each class, then apply vision-language models to classify images via these attributes. Pioneering work shows that querying thousands of attributes can achieve performance competitive with image features. However, our further investigation on 8 datasets reveals that LLM-generated attributes in a large quantity perform almost the same as random words. This surprising finding suggests that significant noise may be present in these attributes. We hypothesize that there exist subsets of attributes that can maintain the classification performance with much smaller sizes, and propose a novel learning-to-search method to discover those concise sets of attributes. As a result, on the CUB dataset, our method achieves performance close to that of massive LLM-generated attributes (e.g., 10k attributes for CUB), yet using only 32 attributes in total to distinguish 200 bird species. Furthermore, our new paradigm demonstrates several additional benefits: higher interpretability and interactivity for humans, and the ability to summarize knowledge for a recognition task.

</details>

---

## 78. ALIP: Adaptive Language-Image Pre-Training with Synthetic Caption

- [ ] ALIP: Adaptive Language-Image Pre-Training with Synthetic Caption | https://openaccess.thecvf.com/content/ICCV2023/html/Yang_ALIP_Adaptive_Language-Image_Pre-Training_with_Synthetic_Caption_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_ALIP_Adaptive_Language-Image_Pre-Training_with_Synthetic_Caption_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) has significantly boosted the performance of various vision-language tasks by scaling up the dataset with image-text pairs collected from the web. However, the presence of intrinsic noise and unmatched image-text pairs in web data can potentially affect the performance of representation learning. To address this issue, we first utilize the OFA model to generate synthetic captions that focus on the image content. The generated captions contain complementary information that is beneficial for pre-training. Then, we propose an Adaptive Language-Image Pre-training (ALIP), a bi-path model that integrates supervision from both raw text and synthetic caption. As the core components of ALIP, the Language Consistency Gate (LCG) and Description Consistency Gate (DCG) dynamically adjust the weights of samples and image-text/caption pairs during the training process. Meanwhile, the adaptive contrastive loss can effectively reduce the impact of noise data and enhances the efficiency of pre-training data. We validate ALIP with experiments on different scales of models and pre-training datasets. Experiments results show that ALIP achieves state-of-the-art performance on multiple downstream tasks including zero-shot image-text retrieval and linear probe. To facilitate future research, the code and pre-trained models are released at https://github.com/deepglint/ALIP.

</details>

---

## 79. Attentive Mask CLIP

- [ ] Attentive Mask CLIP | https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Attentive_Mask_CLIP_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Attentive_Mask_CLIP_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In vision-language modeling, image token removal is an efficient augmentation technique to reduce the cost of encoding image features. The CLIP-style models, however, have been found to be negatively impacted by this technique. We hypothesize that removing a large portion of image tokens may inadvertently destroy the semantic information associated to a given text description, resulting in misaligned paired data in CLIP training. To address this issue, we propose an attentive token removal approach, which retains a small number of tokens that have a strong semantic correlation to the corresponding text description. The correlation scores are dynamically evaluated through an EMA-updated vision encoder. Our method, termed attentive mask CLIP, outperforms original CLIP and CLIP variant with random token removal while saving the training time. In addition, our approach also enables efficient multi-view contrastive learning. Experimentally, by training ViT-B on YFCC-15M dataset, our approach achieves 43.9% top-1 accuracy on ImageNet-1K zero-shot classification, 62.7/42.1 and 38.0/23.2 I2T/T2I retrieval accuracy on Flickr30K and MS COCO, outperforming SLIP by +1.1%,+5.5/+0.9, and +4.4/+1.3, respectively, while being 2.30x faster. An efficient version of our approach runs 1.16x faster than the plain CLIP model, while achieving significant gains of +5.3%, +11.3/+8.0, and +9.5/+4.9 on these benchmarks, respectively.

</details>

---

## 80. Implicit Neural Representation for Cooperative Low-light Image Enhancement

- [ ] Implicit Neural Representation for Cooperative Low-light Image Enhancement | https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Implicit_Neural_Representation_for_Cooperative_Low-light_Image_Enhancement_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Implicit_Neural_Representation_for_Cooperative_Low-light_Image_Enhancement_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The following three factors restrict the application of existing low-light image enhancement methods: unpredictable brightness degradation and noise, inherent gap between metric-favorable and visual-friendly versions, and the limited paired training data. To address these limitations, we propose an implicit Neural Representation method for Cooperative low-light image enhancement, dubbed NeRCo. It robustly recovers perceptual-friendly results in an unsupervised manner. Concretely, NeRCo unifies the diverse degradation factors of real-world scenes with a controllable fitting function, leading to better robustness. In addition, for the output results, we introduce semantic-orientated supervision with priors from the pre-trained vision-language model. Instead of merely following reference images, it encourages results to meet subjective expectations, finding more visual-friendly solutions. Further, to ease the reliance on paired data and reduce solution space, we develop a dual-closed-loop constrained enhancement module. It is trained cooperatively with other affiliated modules in a self-supervised manner. Finally, extensive experiments demonstrate the robustness and superior effectiveness of our proposed NeRCo. Our code is available at https://github.com/Ysz2022/NeRCo.

</details>

---

## 81. Learning Trajectory-Word Alignments for Video-Language Tasks

- [ ] Learning Trajectory-Word Alignments for Video-Language Tasks | https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Learning_Trajectory-Word_Alignments_for_Video-Language_Tasks_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Learning_Trajectory-Word_Alignments_for_Video-Language_Tasks_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In a video, an object usually appears as the trajectory, i.e., it spans over a few spatial but longer temporal patches, that contains abundant spatiotemporal contexts. However, modern Video-Language BERTs (VDL-BERTs) neglect this trajectory characteristic that they usually follow image-language BERTs (IL-BERTs) to deploy the patch-to-word (P2W) attention that may over-exploit trivial spatial contexts and neglect significant temporal contexts. To amend this, we propose a novel TW-BERT to learn Trajectory-Word alignment by a newly designed trajectory-to-word (T2W) attention for solving video-language tasks. Moreover, previous VDL-BERTs usually uniformly sample a few frames into the model while different trajectories have diverse graininess, i.e., some trajectories span longer frames and some span shorter, and using a few frames will lose certain useful temporal contexts. However, simply sampling more frames will also make pre-training infeasible due to the largely increased training burdens. To alleviate the problem, during the fine-tuning stage, we insert a novel Hierarchical Frame-Selector (HFS) module into the video encoder. HFS gradually selects the suitable frames conditioned on the text context for the later cross-modal encoder to learn better trajectory-word alignments. By the proposed T2W attention and HFS, our TW-BERT achieves SOTA performances on text-to-video retrieval tasks, and comparable performances on video question-answering tasks with some VDL-BERTs trained on much more data. The code will be available in the supplementary material.

</details>

---

## 82. HiTeA: Hierarchical Temporal-Aware Video-Language Pre-training

- [ ] HiTeA: Hierarchical Temporal-Aware Video-Language Pre-training | https://openaccess.thecvf.com/content/ICCV2023/html/Ye_HiTeA_Hierarchical_Temporal-Aware_Video-Language_Pre-training_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Ye_HiTeA_Hierarchical_Temporal-Aware_Video-Language_Pre-training_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-training has advanced the performance of various downstream video-language tasks. However, most previous methods directly inherit or adapt typical image-language pre-training paradigms to video-language pre-training, thus not fully exploiting the unique characteristic of video, i.e., temporal. In this paper, we propose a Hierarchical Temporal-Aware video-language pre-training framework, HiTeA, with two novel pre-training tasks for yielding temporal-aware multi-modal representation with cross-modal fine-grained temporal moment information and temporal contextual relations between video-text multi-modal pairs. First, we propose a cross-modal moment exploration task to explore moments in videos by mining the paired texts, which results in detailed video moment representation. Then, based on the learned detailed moment representations, the inherent temporal contextual relations are captured by aligning video-text pairs as a whole in different time resolutions with multi-modal temporal relation exploration task. Furthermore, we introduce the shuffling test to evaluate the temporal reliance of datasets and video-language pre-training models. We achieve state-of-the-art results on 15 well-established video-language understanding and generation tasks, especially on temporal-oriented datasets (e.g., SSv2-Template and SSv2-Label) with 8.6% and 11.1% improvement respectively. HiTeA also demonstrates strong generalization ability when directly transferred to downstream tasks in a zero-shot manner.

</details>

---

## 83. SLAN: Self-Locator Aided Network for Vision-Language Understanding

- [ ] SLAN: Self-Locator Aided Network for Vision-Language Understanding | https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_SLAN_Self-Locator_Aided_Network_for_Vision-Language_Understanding_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_SLAN_Self-Locator_Aided_Network_for_Vision-Language_Understanding_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Learning fine-grained interplay between vision and language contributes to a more accurate understanding for Vision-Language tasks. However, it remains challenging to extract key image regions according to the texts for semantic
 alignments. Most existing works are either limited by text-agnostic and redundant regions obtained with the frozen detectors, or failing to scale further due to their heavy reliance on scarce grounding (gold) data to pre-train detectors. To
 solve these problems, we propose Self-Locator Aided Network (SLAN) for vision-language understanding tasks without any extra gold data. SLAN consists of a region filter and a region adaptor to localize regions of interest conditioned
 on different texts. By aggregating vision-language information, the region filter selects key regions and the region adaptor updates their coordinates with text guidance. With detailed region-word alignments, SLAN can be easily generalized to many downstream tasks. It achieves fairly competitive results on five vision-language understanding tasks (e.g., 85.7% and 69.2% on COCO image-to-text and text-to-image retrieval, surpassing previous SOTA methods). SLAN also demonstrates strong zero-shot and fine-tuned transferability to two localization tasks.

</details>

---

## 84. Exploring Temporal Concurrency for Video-Language Representation Learning

- [ ] Exploring Temporal Concurrency for Video-Language Representation Learning | https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Exploring_Temporal_Concurrency_for_Video-Language_Representation_Learning_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Exploring_Temporal_Concurrency_for_Video-Language_Representation_Learning_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Paired video and language data is naturally temporal concurrency, which requires the modeling of the temporal dynamics within each modality and the temporal alignment across modalities simultaneously. However, most existing video-language representation learning methods only focus on discrete semantic alignment that encourages aligned semantics to be close in the latent space, or temporal context dependency that captures short-range coherence, failing in building the temporal concurrency. In this paper, we propose to learn video-language representations by modeling video-language pairs as Temporal Concurrent Processes (TCP) via a process-wised distance metric learning framework. Specifically, we employ the soft Dynamic Time Warping (DTW) to measure the distance between two processes across modalities and then optimize the DTW costs. Meanwhile, we further introduce a regularization term that enforces the embeddings of each modality approximating a stochastic process to guarantee the inherent dynamics. Experimental results on three benchmarks demonstrate that TCP stands as a state-of-the-art method for various video-language understanding tasks, including paragraph-to-video retrieval, video moment retrieval, and video question-answering. Code is available at https://github.com/hengRUC/TCP.

</details>

---

## 85. Multi-Event Video-Text Retrieval

- [ ] Multi-Event Video-Text Retrieval | https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Multi-Event_Video-Text_Retrieval_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Multi-Event_Video-Text_Retrieval_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-Text Retrieval (VTR) is a crucial multi-modal task in an era of massive video-text data on the Internet.
 A plethora of work characterized by using a two-stream Vision-Language model architecture that learns a joint representation of video-text pairs has become a prominent approach for the VTR task.
 However, these models operate under the assumption of bijective video-text correspondences and neglect a more practical scenario where video content usually encompasses multiple events, while texts like user queries or webpage metadata tend to be specific and correspond to single events.
 This establishes a gap between the previous training objective and real-world applications, leading to the potential performance degradation of earlier models during inference.
 In this study, we introduce the Multi-event Video-Text Retrieval (MeVTR) task, addressing scenarios in which each video contains multiple different events, as a niche scenario of the conventional Video-Text Retrieval Task. We present a simple model, Me-Retriever, which incorporates key event video representation and a new MeVTR loss for the MeVTR task. Comprehensive experiments show that this straightforward framework outperforms other models in the Video-to-Text and Text-to-Video tasks, effectively establishing a robust baseline for the MeVTR task. We believe this work serves as a strong foundation for future studies.
 Code is available at https://github.com/gengyuanmax/MeVTR.

</details>

---

## 86. Generative Prompt Model for Weakly Supervised Object Localization

- [ ] Generative Prompt Model for Weakly Supervised Object Localization | https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Generative_Prompt_Model_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Generative_Prompt_Model_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Weakly supervised object localization (WSOL) remains challenging when learning object localization models from image category labels. Conventional methods that discriminatively train activation models ignore representative yet less discriminative object parts. In this study, we propose a generative prompt model (GenPromp), defining the first generative pipeline to localize less discriminative object parts by formulating WSOL as a conditional image denoising procedure. During training, GenPromp converts image category labels to learnable prompt embeddings which are fed to a generative model to conditionally recover the input image with noise and learn representative embeddings. During inference, GenPromp combines the representative embeddings with discriminative embeddings (queried from an off-the-shelf vision-language model) for both representative and discriminative capacity. The combined embeddings are finally used to generate multi-scale high-quality attention maps, which facilitate localizing full object extent. Experiments on CUB-200-2011 and ILSVRC show that GenPromp respectively outperforms the best discriminative models by 5.2% and 5.6% (Top-1 Loc), setting a solid baseline for WSOL with the generative model. Code is available at https://github.com/callsys/GenPromp.

</details>

---

## 87. Unified Visual Relationship Detection with Vision and Language Models

- [ ] Unified Visual Relationship Detection with Vision and Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Unified_Visual_Relationship_Detection_with_Vision_and_Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Unified_Visual_Relationship_Detection_with_Vision_and_Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This work focuses on training a single visual relationship detector predicting over the union of label spaces from multiple datasets. Merging labels spanning different datasets could be challenging due to inconsistent taxonomies. The issue is exacerbated in visual relationship detection when second-order visual semantics are introduced between pairs of objects. To address this challenge, we propose UniVRD, a novel bottom-up method for Unified Visual Relationship Detection by leveraging vision and language models (VLMs). VLMs provide well-aligned image and text embeddings, where similar relationships are optimized to be close to each other for semantic unification. Our bottom-up design enables the model to enjoy the benefit of training with both object detection and visual relationship datasets. Empirical results on both human-object interaction detection and scene-graph generation demonstrate the competitive performance of our model. UniVRD achieves 38.07 mAP on HICO-DET, outperforming the current best bottom-up HOI detector by 14.26 mAP. More importantly, we show that our unified detector performs as well as dataset-specific models in mAP, and achieves further improvements when we scale up the model. Our code will be made publicly available on GitHub.

</details>

---

## 88. Unleashing Text-to-Image Diffusion Models for Visual Perception

- [ ] Unleashing Text-to-Image Diffusion Models for Visual Perception | https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Unleashing_Text-to-Image_Diffusion_Models_for_Visual_Perception_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Unleashing_Text-to-Image_Diffusion_Models_for_Visual_Perception_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models (DMs) have become the new trend of generative models and have demonstrated a powerful ability of conditional synthesis. Among those, text-to-image diffusion models pre-trained on large-scale image-text pairs are highly controllable by customizable prompts. Unlike the unconditional generative models that focus on low-level attributes and details, text-to-image diffusion models contain more high-level knowledge thanks to the vision-language pre-training. In this paper, we propose VPD (Visual Perception with pre-trained Diffusion models), a new framework that exploits the semantic information of a pre-trained text-to-image diffusion model in visual perception tasks. Instead of using the pre-trained denoising autoencoder in a diffusion-based pipeline, we simply use it as a backbone and aim to study how to take full advantage of the learned knowledge. Specifically, we prompt the denoising decoder with proper textual inputs and refine the text features with an adapter, leading to a better alignment to the pre-trained stage and making the visual contents interact with the text prompts. We also propose to utilize the cross-attention maps between the visual features and the text features to provide explicit guidance. Compared with other pre-training methods, we show that vision-language pre-trained diffusion models can be faster adapted to downstream visual perception tasks using the proposed VPD. Extensive experiments on semantic segmentation, referring image segmentation, and depth estimation demonstrate the effectiveness of our method. Notably, VPD attains 0.254 RMSE on NYUv2 depth estimation and 73.3% oIoU on RefCOCO-val referring image segmentation, establishing new records on these two benchmarks. Code is available at https://github.com/wl-zhao/VPD.

</details>

---

## 89. Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models

- [ ] Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Preventing_Zero-Shot_Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Preventing_Zero-Shot_Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Continual learning (CL) can help pre-trained vision-language models efficiently adapt to new or under-trained data distributions without re-training. Nevertheless, during the continual training of the Contrastive Language-Image Pre-training (CLIP) model, we observe that the model's zero-shot transfer ability significantly degrades due to catastrophic forgetting. Existing CL methods can mitigate forgetting by replaying previous data. However, since the CLIP dataset is private, replay methods cannot access the pre-training dataset. In addition, replaying data of previously learned downstream tasks can enhance their performance but comes at the cost of sacrificing zero-shot performance. To address this challenge, we propose a novel method ZSCL to prevent zero-shot transfer degradation in the continual learning of vision-language models in both feature and parameter space. In the feature space, a reference dataset is introduced for distillation between the current and initial models. The reference dataset should have semantic diversity but no need to be labeled, seen in pre-training, or matched image-text pairs. In parameter space, we prevent a large parameter shift by averaging weights during the training. We propose a more challenging Multi-domain Task Incremental Learning (MTIL) benchmark to evaluate different methods, where tasks are from various domains instead of class-separated in a single dataset. Our method outperforms other methods in the traditional class-incremental learning setting and the MTIL by 9.7% average score. Our code locates at https://github.com/Thunderbeee/ZSCL.

</details>

---

## 90. Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-Trained Vision-Language Models

- [ ] Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-Trained Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Regularized_Mask_Tuning_Uncovering_Hidden_Knowledge_in_Pre-Trained_Vision-Language_Models_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Regularized_Mask_Tuning_Uncovering_Hidden_Knowledge_in_Pre-Trained_Vision-Language_Models_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning and adapter tuning have shown great potential in transferring pre-trained vision-language models (VLMs) to various downstream tasks. In this work, we design a new type of tuning method, termed as regularized mask tuning, which masks the network parameters through a learnable selection. Inspired by neural pathways, we argue that the knowledge required by a downstream task already exists in the pre-trained weights but just gets concealed in the upstream pre-training stage. To bring the useful knowledge back into light, we first identify a set of parameters that are important to a given downstream task, then attach a binary mask to each parameter, and finally optimize these masks on the downstream data with the parameters frozen. When updating the mask, we introduce a novel gradient dropout strategy to regularize the parameter selection, in order to prevent the model from forgetting old knowledge and overfitting the downstream data. Experimental results on 11 datasets demonstrate the consistent superiority of our method over previous alternatives. It is noteworthy that we manage to deliver 18.73% performance improvement compared to the zero-shot CLIP via masking an average of only 2.56% parameters. Furthermore, our method is synergistic with most existing parameter-efficient tuning methods and can boost the performance on top of them. Code will be made publicly available.

</details>

---

## 91. 3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment

- [ ] 3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment | https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_3D-VisTA_Pre-trained_Transformer_for_3D_Vision_and_Text_Alignment_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_3D-VisTA_Pre-trained_Transformer_for_3D_Vision_and_Text_Alignment_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

3D vision-language grounding (3D-VL) is an emerging field that aims to connect the 3D physical world with natural language, which is crucial for achieving embodied intelligence. Current 3D-VL models rely heavily on sophisticated modules, auxiliary losses, and optimization tricks, which calls for a simple and unified model. In this paper, we propose 3D-VisTA, a pre-trained Transformer for 3D Vis ion and Text Alignment that can be easily adapted to various downstream tasks. 3D-VisTA simply utilizes self-attention layers for both single-modal modeling and multi-modal fusion without any sophisticated task-specific design. To further enhance its performance on 3D-VL tasks, we construct ScanScribe, the first large-scale 3D scene-text pairs dataset for 3D-VL pre-training. ScanScribe contains 2,995 RGB-D scans for 1,185 unique indoor scenes originating from ScanNet and 3R-Scan datasets, along with paired 278K scene descriptions generated from existing 3D-VL tasks, templates, and GPT-3. 3D-VisTA is pre-trained on ScanScribe via masked language/object modeling and scene-text matching. It achieves state-of-the-art results on various 3D-VL tasks, ranging from visual grounding and question answering to situated reasoning. Moreover, 3D-VisTA demonstrates superior data efficiency, obtaining strong performance even with limited annotations during downstream task fine-tuning.

</details>

---

## 92. CTP:Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation

- [ ] CTP:Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation | https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_CTPTowards_Vision-Language_Continual_Pretraining_via_Compatible_Momentum_Contrast_and_Topology_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_CTPTowards_Vision-Language_Continual_Pretraining_via_Compatible_Momentum_Contrast_and_Topology_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pretraining (VLP) has shown impressive results on diverse downstream tasks by offline training on large-scale datasets. Regarding the growing nature of real-world data, such an offline training paradigm on ever-expanding data is unsustainable, because models lack the continual learning ability to accumulate knowledge constantly. However, most continual learning studies are limited to uni-modal classification and existing multi-modal datasets cannot simulate continual non-stationary data stream scenarios. To support the study of Vision-Language Continual Pretraining (VLCP), we first contribute a comprehensive and unified benchmark dataset P9D which contains over one million product image-text pairs from 9 industries. The data from each industry as an independent task supports continual learning and conforms to the real-world long-tail nature to simulate pretraining on web data. We comprehensively study the characteristics and challenges of VLCP, and propose a new algorithm: Compatible momentum contrast with Topology Preservation, dubbed CTP. The compatible momentum model absorbs the knowledge of the current and previous-task models to flexibly update the modal feature. Moreover, Topology Preservation transfers the knowledge of embedding across tasks while preserving the flexibility of feature adjustment. The experimental results demonstrate our method not only achieves superior performance compared with other baselines but also does not bring an expensive training burden. Dataset and codes are available at https://github.com/KevinLight831/CTP.

</details>

---

## 93. Prompt-aligned Gradient for Prompt Tuning

- [ ] Prompt-aligned Gradient for Prompt Tuning | https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.html

- **Conference**: ICCV

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Thanks to the large pre-trained vision-language models (VLMs) like CLIP, we can craft a zero-shot classifier by discrete prompt design, e.g., the confidence score of an image being "[CLASS]" can be obtained by using the VLM provided similarity between the image and the prompt sentence "a photo of a [CLASS]". Furthermore, prompting shows great potential for fast adaptation of VLMs to downstream tasks if we fine-tune the soft prompts with few samples. However, we find a common failure that improper fine-tuning or learning with extremely few-shot samples may even under-perform the zero-shot prediction. Existing methods still address this problem by using traditional anti-overfitting techniques such as early stopping and data augmentation, which lack a principled solution specific to prompting. In this paper, we present Prompt-aligned Gradient, dubbed ProGrad to prevent prompt tuning from forgetting the general knowledge learned from VLMs. In particular, ProGrad only updates the prompt whose gradient is aligned (or non-conflicting) to the general knowledge, which is represented as the optimization direction offered by the predefined prompt predictions. Extensive experiments under the few-shot learning, domain generalization, base-to-new generalization and cross-dataset transfer settings demonstrate
 the stronger few-shot generalization ability of ProGrad over state-of-the-art prompt tuning methods. Codes and theoretical proof are in Appendix.

</details>

---

