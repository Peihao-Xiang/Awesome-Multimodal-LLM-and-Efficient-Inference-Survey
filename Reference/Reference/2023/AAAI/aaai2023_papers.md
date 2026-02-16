# AAAI 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_aaai2023_papers.csv

## 1. Tagging before Alignment: Integrating Multi-Modal Tags for Video-Text Retrieval

- [ ] Tagging before Alignment: Integrating Multi-Modal Tags for Video-Text Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/25113

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25113

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language alignment learning for video-text retrieval arouses a lot of attention in recent years. Most of the existing methods either transfer the knowledge of image-text pretraining model to video-text retrieval task without fully exploring the multi-modal information of videos, or simply fuse multi-modal features in a brute force manner without explicit guidance. In this paper, we integrate multi-modal information in an explicit manner by tagging, and use the tags as the anchors for better video-text alignment. Various pretrained experts are utilized for extracting the information of multiple modalities, including object, person, motion, audio, etc. To take full advantage of these information, we propose the TABLE (TAgging Before aLignmEnt) network, which consists of a visual encoder, a tag encoder, a text encoder, and a tag-guiding cross-modal encoder for jointly encoding multi-frame visual features and multi-modal tags information. Furthermore, to strengthen the interaction between video and text, we build a joint cross-modal encoder with the triplet input of [vision, tag, text] and perform two additional supervised tasks, Video Text Matching (VTM) and Masked Language Modeling (MLM). Extensive experimental results demonstrate that the TABLE model is capable of achieving State-Of-The-Art (SOTA) performance on various video-text retrieval benchmarks, including MSR-VTT, MSVD, LSMDC and DiDeMo.

</details>

---

## 2. Unifying Vision-Language Representation Space with Single-Tower Transformer

- [ ] Unifying Vision-Language Representation Space with Single-Tower Transformer | https://ojs.aaai.org/index.php/AAAI/article/view/25178

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25178

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive learning is a form of distance learning that aims to learn invariant features from two related representations. In this work, we explore the hypothesis that an image and caption can be regarded as two different views of the underlying mutual information, and train a model to learn a unified vision-language representation space that encodes both modalities at once in a modality-agnostic manner. We first identify difficulties in learning a one-tower model for vision-language pretraining (VLP), and propose One Representation (OneR) as a simple yet effective framework for our goal. We discover intriguing properties that distinguish OneR from the previous works that have modality-specific representation spaces such as zero-shot localization, text-guided visual reasoning and multi-modal retrieval, and present analyses to provide insights into this new form of multi-modal representation learning. Thorough evaluations demonstrate the potential of a unified modality-agnostic VLP framework.

</details>

---

## 3. Learning Semantic Alignment with Global Modality Reconstruction for Video-Language Pre-training towards Retrieval

- [ ] Learning Semantic Alignment with Global Modality Reconstruction for Video-Language Pre-training towards Retrieval | https://ojs.aaai.org/index.php/AAAI/article/view/25222

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25222

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-language pre-training for text-based video retrieval tasks is vitally important. Previous pre-training methods suffer from the semantic misalignments. The reason is that these methods ignore sequence alignments but focusing on critical token alignment. To alleviate the problem, we propose a video-language pre-training framework, termed videolanguage pre-training For lEarning sEmantic aLignments (FEEL), to learn semantic alignments at the sequence level. Specifically, the global modality reconstruction and the cross- modal self-contrasting method is utilized to learn the alignments at the sequence level better. Extensive experimental results demonstrate the effectiveness of FEEL on text-based video retrieval and text-based video corpus moment retrieval.

</details>

---

## 4. CLIP-ReID: Exploiting Vision-Language Model for Image Re-identification without Concrete Text Labels

- [ ] CLIP-ReID: Exploiting Vision-Language Model for Image Re-identification without Concrete Text Labels | https://ojs.aaai.org/index.php/AAAI/article/view/25225

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25225

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models like CLIP have recently shown superior performances on various downstream tasks, including image classification and segmentation. However, in fine-grained image re-identification (ReID), the labels are indexes, lacking concrete text descriptions. Therefore, it remains to be determined how such models could be applied to these tasks. This paper first finds out that simply fine-tuning the visual model initialized by the image encoder in CLIP, has already obtained competitive performances in various ReID tasks. Then we propose a two-stage strategy to facilitate a better visual representation. The key idea is to fully exploit the cross-modal description ability in CLIP through a set of learnable text tokens for each ID and give them to the text encoder to form ambiguous descriptions. In the first training stage, image and text encoders from CLIP keep fixed, and only the text tokens are optimized from scratch by the contrastive loss computed within a batch. In the second stage, the ID-specific text tokens and their encoder become static, providing constraints for fine-tuning the image encoder. With the help of the designed loss in the downstream task, the image encoder is able to represent data as vectors in the feature embedding accurately. The effectiveness of the proposed strategy is validated on several datasets for the person or vehicle ReID tasks. Code is available at https://github.com/Syliz517/CLIP-ReID.

</details>

---

## 5. Actional Atomic-Concept Learning for Demystifying Vision-Language Navigation

- [ ] Actional Atomic-Concept Learning for Demystifying Vision-Language Navigation | https://ojs.aaai.org/index.php/AAAI/article/view/25243

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25243

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Navigation (VLN) is a challenging task which requires an agent to align complex visual observations to language instructions to reach the goal position. Most existing VLN agents directly learn to align the raw directional features and visual features trained using one-hot labels to linguistic instruction features. However, the big semantic gap among these multi-modal inputs makes the alignment difficult and therefore limits the navigation performance. In this paper, we propose Actional Atomic-Concept Learning (AACL), which maps visual observations to actional atomic concepts for facilitating the alignment. Specifically, an actional atomic concept is a natural language phrase containing an atomic action and an object, e.g., ``go up stairs''. These actional atomic concepts, which serve as the bridge between observations and instructions, can effectively mitigate the semantic gap and simplify the alignment. AACL contains three core components: 1) a concept mapping module to map the observations to the actional atomic concept representations through the VLN environment and the recently proposed Contrastive Language-Image Pretraining (CLIP) model, 2) a concept refining adapter to encourage more instruction-oriented object concept extraction by re-ranking the predicted object concepts by CLIP, and 3) an observation co-embedding module which utilizes concept representations to regularize the observation representations. Our AACL establishes new state-of-the-art results on both fine-grained (R2R) and high-level (REVERIE and R2R-Last) VLN benchmarks. Moreover, the visualization shows that AACL significantly improves the interpretability in action decision. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/VLN-AACL.

</details>

---

## 6. Token Mixing: Parameter-Efficient Transfer Learning from Image-Language to Video-Language

- [ ] Token Mixing: Parameter-Efficient Transfer Learning from Image-Language to Video-Language | https://ojs.aaai.org/index.php/AAAI/article/view/25267

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25267

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Applying large scale pre-trained image-language model to video-language tasks has recently become a trend, which brings two challenges. One is how to effectively transfer knowledge from static images to dynamic videos, and the other is how to deal with the prohibitive cost of fully fine-tuning due to growing model size. Existing works that attempt to realize parameter-efficient image-language to video-language transfer learning can be categorized into two types: 1) appending a sequence of temporal transformer blocks after the 2D Vision Transformer (ViT), and 2) inserting a temporal block into the ViT architecture. While these two types of methods only require fine-tuning the newly added components, there are still many parameters to update, and they are only validated on a single video-language task. In this work, based on our analysis of the core ideas of different temporal modeling components in existing approaches, we propose a token mixing strategy to enable cross-frame interactions, which enables transferring from the pre-trained image-language model to video-language tasks through selecting and mixing a key set and a value set from the input video samples. As token mixing does not require the addition of any components or modules, we can directly partially fine-tune the pre-trained image-language model to achieve parameter-efficiency. We carry out extensive experiments to compare our proposed token mixing method with other parameter-efficient transfer learning methods. Our token mixing method outperforms other methods on both understanding tasks and generation tasks. Besides, our method achieves new records on multiple video-language tasks. The code is available at https://github.com/yuqi657/video_language_model.

</details>

---

## 7. Efficient End-to-End Video Question Answering with Pyramidal Multimodal Transformer

- [ ] Efficient End-to-End Video Question Answering with Pyramidal Multimodal Transformer | https://ojs.aaai.org/index.php/AAAI/article/view/25296

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25296

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a new method for end-to-end Video Question Answering (VideoQA), aside from the current popularity of using large-scale pre-training with huge feature extractors. We achieve this with a pyramidal multimodal transformer (PMT) model, which simply incorporates a learnable word embedding layer, a few convolutional and transformer layers. We use the anisotropic pyramid to fulfill video-language interactions across different spatio-temporal scales. In addition to the canonical pyramid, which includes both bottom-up and top-down pathways with lateral connections, novel strategies are proposed to decompose the visual feature stream into spatial and temporal sub-streams at different scales and implement their interactions with the linguistic semantics while preserving the integrity of local and global semantics. We demonstrate better or on-par performances with high computational efficiency against state-of-the-art methods on five VideoQA benchmarks. Our ablation study shows the scalability of our model that achieves competitive results for text-to-video retrieval by leveraging feature extractors with reusable pre-trained weights, and also the effectiveness of the pyramid. Code available at: https://github.com/Trunpm/PMT-AAAI23.

</details>

---

## 8. End-to-End Zero-Shot HOI Detection via Vision and Language Knowledge Distillation

- [ ] End-to-End Zero-Shot HOI Detection via Vision and Language Knowledge Distillation | https://ojs.aaai.org/index.php/AAAI/article/view/25385

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25385

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Most existing Human-Object Interaction (HOI) Detection methods rely heavily on full annotations with predefined HOI categories, which is limited in diversity and costly to scale further. We aim at advancing zero-shot HOI detection to detect both seen and unseen HOIs simultaneously. The fundamental challenges are to discover potential human-object pairs and identify novel HOI categories. To overcome the above challenges, we propose a novel End-to-end zero-shot HOI Detection (EoID) framework via vision-language knowledge distillation. We first design an Interactive Score module combined with a Two-stage Bipartite Matching algorithm to achieve interaction distinguishment for human-object pairs in an action-agnostic manner.
Then we transfer the distribution of action probability from the pretrained vision-language teacher as well as the seen ground truth to the HOI model to attain zero-shot HOI classification. Extensive experiments on HICO-Det dataset demonstrate that our model discovers potential interactive pairs and enables the recognition of unseen HOIs. Finally, our method outperforms the previous SOTA under various zero-shot settings. Moreover, our method is generalizable to large-scale object detection data to further scale up the action sets. The source code is available at: https://github.com/mrwu-mac/EoID.

</details>

---

## 9. Revisiting Classifier: Transferring Vision-Language Models for Video Recognition

- [ ] Revisiting Classifier: Transferring Vision-Language Models for Video Recognition | https://ojs.aaai.org/index.php/AAAI/article/view/25386

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25386

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Transferring knowledge from task-agnostic pre-trained deep models for downstream tasks is an important topic in computer vision research. Along with the growth of computational capacity, we now have open-source vision-language pre-trained models in large scales of the model architecture and amount of data. In this study, we focus on transferring knowledge for video classification tasks. Conventional methods randomly initialize the linear classifier head for vision classification, but they leave the usage of the text encoder for downstream visual recognition tasks undiscovered. In this paper, we revise the role of the linear classifier and replace the classifier with the different knowledge from pre-trained model. We utilize the well-pretrained language model to generate good semantic target for efficient transferring learning. The empirical study shows that our method improves both the performance and the training speed of video classification, with a negligible change in the model. Our simple yet effective tuning paradigm achieves state-of-the-art performance and efficient training on various video recognition scenarios, i.e., zero-shot, few-shot, general recognition. In particular, our paradigm achieves the state-of-the-art accuracy of 87.8% on Kinetics-400, and also surpasses previous methods by 20~50% absolute top-1 accuracy under zero-shot, few-shot settings on five video datasets. Code and models are available at https://github.com/whwu95/Text4Vis.

</details>

---

## 10. Semantics-Aware Dynamic Localization and Refinement for Referring Image Segmentation

- [ ] Semantics-Aware Dynamic Localization and Refinement for Referring Image Segmentation | https://ojs.aaai.org/index.php/AAAI/article/view/25428

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25428

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Referring image segmentation segments an image from a language expression. With the aim of producing high-quality masks, existing methods often adopt iterative learning approaches that rely on RNNs or stacked attention layers to refine vision-language features. Despite their complexity, RNN-based methods are subject to specific encoder choices, while attention-based methods offer limited gains. In this work, we introduce a simple yet effective alternative for progressively learning discriminative multi-modal features. The core idea of our approach is to leverage a continuously updated query as the representation of the target object and at each iteration, strengthen multi-modal features strongly correlated to the query while weakening less related ones. As the query is initialized by language features and successively updated by object features, our algorithm gradually shifts from being localization-centric to segmentation-centric. This strategy enables the incremental recovery of missing object parts and/or removal of extraneous parts through iteration. Compared to its counterparts, our method is more versatile—it can be plugged into prior arts straightforwardly and consistently bring improvements. Experimental results on the challenging datasets of RefCOCO, RefCOCO+, and G-Ref demonstrate its advantage with respect to the state-of-the-art methods.

</details>

---

## 11. Generalizing Multiple Object Tracking to Unseen Domains by Introducing Natural Language Representation

- [ ] Generalizing Multiple Object Tracking to Unseen Domains by Introducing Natural Language Representation | https://ojs.aaai.org/index.php/AAAI/article/view/25437

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25437

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Although existing multi-object tracking (MOT) algorithms have obtained competitive performance on various benchmarks, almost all of them train and validate models on the same domain. The domain generalization problem of MOT is hardly studied. To bridge this gap, we first draw the observation that the high-level information contained in natural language is domain invariant to different tracking domains. Based on this observation, we propose to introduce natural language representation into visual MOT models for boosting the domain generalization ability. However, it is infeasible to label every tracking target with a textual description. To tackle this problem, we design two modules, namely visual context prompting (VCP) and visual-language mixing (VLM). Specifically, VCP generates visual prompts based on the input frames. VLM joints the information in the generated visual prompts and the textual prompts from a pre-defined Trackbook to obtain instance-level pseudo textual description, which is domain invariant to different tracking scenes. Through training models on MOT17 and validating them on MOT20, we observe that the pseudo textual descriptions generated by our proposed modules improve the generalization performance of query-based trackers by large margins.

</details>

---

## 12. STOA-VLP: Spatial-Temporal Modeling of Object and Action for Video-Language Pre-training

- [ ] STOA-VLP: Spatial-Temporal Modeling of Object and Action for Video-Language Pre-training | https://ojs.aaai.org/index.php/AAAI/article/view/25483

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25483

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Although large-scale video-language pre-training models, which usually build a global alignment between the video and the text, have achieved remarkable progress on various downstream tasks, the idea of adopting fine-grained information during the pre-training stage is not well explored. In this work, we propose STOA-VLP, a pre-training framework that jointly models object and action information across spatial and temporal dimensions. More specifically, the model regards object trajectories across frames and multiple action features from the video as fine-grained features. Besides, We design two auxiliary tasks to better incorporate both kinds of information into the pre-training process of the video-language model. The first is the dynamic object-text alignment task, which builds a better connection between object trajectories and the relevant noun tokens. The second is the spatial-temporal action set prediction, which guides the model to generate consistent action features by predicting actions found in the text. Extensive experiments on three downstream tasks (video captioning, text-video retrieval, and video question answering) demonstrate the effectiveness of our proposed STOA-VLP (e.g. 3.7 Rouge-L improvements on MSR-VTT video captioning benchmark, 2.9% accuracy improvements on MSVD video question answering benchmark, compared to previous approaches).

</details>

---

## 13. Debiased Fine-Tuning for Vision-Language Models by Prompt Regularization

- [ ] Debiased Fine-Tuning for Vision-Language Models by Prompt Regularization | https://ojs.aaai.org/index.php/AAAI/article/view/25496

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25496

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present a new paradigm for fine-tuning large-scale vision-language pre-trained models on downstream task, dubbed Prompt Regularization (ProReg). Different from traditional fine-tuning which easily overfits to the downstream task data, ProReg uses the prediction by prompting the pretrained model to regularize the fine-tuning. The motivation is: by prompting the large model “a photo of a [CLASS]”, the fill-in answer is only dependent on the pretraining encyclopedic knowledge while independent of the task data distribution, which is usually biased. Specifically, given a training sample prediction during fine-tuning, we first calculate its Kullback-Leibler loss of the prompt prediction and Cross-Entropy loss of the ground-truth label, and then combine them with a proposed sample-wise adaptive trade- off weight, which automatically adjusts the transfer between the pretrained and downstream domains. On various out-of-distribution benchmarks, we show the consistently strong performance of ProReg compared with conventional fine-tuning, zero-shot prompt, prompt tuning, and other state-of-the-art methods.

</details>

---

## 14. Visually Grounded Commonsense Knowledge Acquisition

- [ ] Visually Grounded Commonsense Knowledge Acquisition | https://ojs.aaai.org/index.php/AAAI/article/view/25809

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/25809

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale commonsense knowledge bases empower a broad range of AI applications, where the automatic extraction of commonsense knowledge (CKE) is a fundamental and challenging problem. CKE from text is known for suffering from the inherent sparsity and reporting bias of commonsense in text. Visual perception, on the other hand, contains rich commonsense knowledge about real-world entities, e.g., (person, can_hold, bottle), which can serve as promising sources for acquiring grounded commonsense knowledge. In this work, we present CLEVER, which formulates CKE as a distantly supervised multi-instance learning problem, where models learn to summarize commonsense relations from a bag of images about an entity pair without any human annotation on image instances. To address the problem, CLEVER leverages vision-language pre-training models for deep understanding of each image in the bag, and selects informative instances from the bag to summarize commonsense entity relations via a novel contrastive attention mechanism. Comprehensive experimental results in held-out and human evaluation show that CLEVER can extract commonsense knowledge in promising quality, outperforming pre-trained language model-based methods by 3.9 AUC and 6.4 mAUC points. The predicted commonsense scores show strong correlation with human judgment with a 0.78 Spearman coefficient. Moreover, the extracted commonsense can also be grounded into images with reasonable interpretability. The data and codes can be obtained at https://github.com/thunlp/CLEVER.

</details>

---

## 15. Accommodating Audio Modality in CLIP for Multimodal Processing

- [ ] Accommodating Audio Modality in CLIP for Multimodal Processing | https://ojs.aaai.org/index.php/AAAI/article/view/26153

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/26153

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multimodal processing has attracted much attention lately especially with the success of pre-training. However, the exploration has mainly focused on vision-language pre-training, as introducing more modalities can greatly complicate model design and optimization. In this paper, we extend the state-of-the-art Vision-Language model CLIP to accommodate the audio modality for Vision-Language-Audio multimodal processing. Specifically, we apply inter-modal and intra-modal contrastive learning to explore the correlation between audio and other modalities in addition to the inner characteristics of the audio modality. Moreover, we further design an audio type token to dynamically learn different audio information type for different scenarios, as both verbal and nonverbal heterogeneous information is conveyed in general audios. Our proposed CLIP4VLA model is validated in different downstream tasks including video retrieval and video captioning, and achieves the state-of-the-art performance on the benchmark datasets of MSR-VTT, VATEX, and Audiocaps.The corresponding code and checkpoints will be released at https://github.com/ludanruan/CLIP4VLA.

</details>

---

## 16. BridgeTower: Building Bridges between Encoders in Vision-Language Representation Learning

- [ ] BridgeTower: Building Bridges between Encoders in Vision-Language Representation Learning | https://ojs.aaai.org/index.php/AAAI/article/view/26263

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/26263

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language (VL) models with the Two-Tower architecture have dominated visual-language representation learning in recent years. Current VL models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a deep cross-modal encoder, or feed the last-layer uni-modal representations from the deep pre-trained uni-modal encoders into the top cross-modal encoder. Both approaches potentially restrict vision-language representation learning and limit model performance. In this paper, we propose BridgeTower, which introduces multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations of different semantic levels of pre-trained uni-modal encoders in the cross-modal encoder. Pre-trained with only 4M images, BridgeTower achieves state-of-the-art performance on various downstream vision-language tasks. In particular, on the VQAv2 test-std set, BridgeTower achieves an accuracy of 78.73%, outperforming the previous state-of-the-art model METER by 1.09% with the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BridgeTower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets. Code and checkpoints are available at https://github.com/microsoft/BridgeTower.

</details>

---

## 17. Improving the Cross-Lingual Generalisation in Visual Question Answering

- [ ] Improving the Cross-Lingual Generalisation in Visual Question Answering | https://ojs.aaai.org/index.php/AAAI/article/view/26574

- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/26574

- **Conference**: AAAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

While several benefits were realized for multilingual vision-language pretrained models, recent benchmarks across various tasks and languages showed poor cross-lingual generalisation when multilingually pre-trained vision-language models are applied to non-English data, with a large gap between (supervised) English performance and (zero-shot) cross-lingual transfer. In this work, we explore the poor performance of these models on a zero-shot cross-lingual visual question answering (VQA) task, where models are fine-tuned on English visual-question data and evaluated on 7 typologically diverse languages. We improve cross-lingual transfer with three strategies: (1) we introduce a linguistic prior objective to augment the cross-entropy loss with a similarity-based loss to guide the model during training, (2) we learn a task-specific subnetwork that improves cross-lingual generalisation and reduces variance without model modification, (3) we augment training examples using synthetic code-mixing to promote alignment of embeddings between source and target languages. Our experiments on xGQA using the pretrained multilingual multimodal transformers UC2 and M3P demonstrates the consistent effectiveness of the proposed fine-tuning strategy for 7 languages, outperforming existing transfer methods with sparse models.

</details>

---

