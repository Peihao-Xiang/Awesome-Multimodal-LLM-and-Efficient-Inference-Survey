# CVPR 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_cvpr2023_papers.csv

## 1. WINNER: Weakly-Supervised hIerarchical decompositioN and aligNment for Spatio-tEmporal Video gRounding

- [ ] WINNER: Weakly-Supervised hIerarchical decompositioN and aligNment for Spatio-tEmporal Video gRounding | https://cvpr.thecvf.com/virtual/2023/poster/20964

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/20964

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Spatio-temporal video grounding aims to localize the aligned visual tube corresponding to a language query. Existing techniques achieve such alignment by exploiting dense boundary and bounding box annotations, which can be prohibitively expensive. To bridge the gap, we investigate the weakly-supervised setting, where models learn from easily accessible video-language data without annotations. We identify that intra-sample spurious correlations among video-language components can be alleviated if the model captures the decomposed structures of video and language data. In this light, we propose a novel framework, namely WINNER, for hierarchical video-text understanding. WINNER first builds the language decomposition tree in a bottom-up manner, upon which the structural attention mechanism and top-down feature backtracking jointly build a multi-modal decomposition tree, permitting a hierarchical understanding of unstructured videos. The multi-modal decomposition tree serves as the basis for multi-hierarchy language-tube matching. A hierarchical contrastive learning objective is proposed to learn the multi-hierarchy correspondence and distinguishment with intra-sample and inter-sample video-text decomposition structures, achieving video-language decomposition structure alignment. Extensive experiments demonstrate the rationality of our design and its effectiveness beyond state-of-the-art weakly supervised methods, even some supervised methods.

</details>

---

## 2. CLIPPING: Distilling CLIP-Based Models With a Student Base for Video-Language Retrieval

- [ ] CLIPPING: Distilling CLIP-Based Models With a Student Base for Video-Language Retrieval | https://cvpr.thecvf.com/virtual/2023/poster/20979

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/20979

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-training a vison-language model and then fine-tuning it on downstream tasks have become a popular paradigm. However, pre-trained vison-language models with the Transformer architecture usually take long inference time. Knowledge distillation has been an efficient technique to transfer the capability of a large model to a small one while maintaining the accuracy, which has achieved remarkable success in natural language processing. However, it faces many problems when applying KD to the multi-modality applications. In this paper, we propose a novel knowledge distillation method, named CLIPPING, where the plentiful knowledge of a large teacher model that has been fine-tuned for video-language tasks with the powerful pre-trained CLIP can be effectively transferred to a small student only at the fine-tuning stage. Especially, a new layer-wise alignment with the student as the base is proposed for knowledge distillation of the intermediate layers in CLIPPING, which enables the student’s layers to be the bases of the teacher, and thus allows the student to fully absorb the knowledge of the teacher. CLIPPING with MobileViT-v2 as the vison encoder without any vison-language pre-training achieves 88.1%-95.3% of the performance of its teacher on three video-language retrieval benchmarks, with its vison encoder being 19.5x smaller. CLIPPING also significantly outperforms a state-of-the-art small baseline (ALL-in-one-B) on the MSR-VTT dataset, obtaining relatively 7.4% performance gain, with 29% fewer parameters and 86.9% fewer flops. Moreover, CLIPPING is comparable or even superior to many large pre-training models.

</details>

---

## 3. CREPE: Can Vision-Language Foundation Models Reason Compositionally?

- [ ] CREPE: Can Vision-Language Foundation Models Reason Compositionally? | https://cvpr.thecvf.com/virtual/2023/poster/21031

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21031

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

A fundamental characteristic common to both human vision and natural language is their compositional nature. Yet, despite the performance gains contributed by large vision and language pretraining, we find that--across 7 architectures trained with 4 algorithms on massive datasets--they struggle at compositionality. To arrive at this conclusion, we introduce a new compositionality evaluation benchmark, CREPE, which measures two important aspects of compositionality identified by cognitive science literature: systematicity and productivity. To measure systematicity, CREPE consists of a test dataset containing over 370K image-text pairs and three different seen-unseen splits. The three splits are designed to test models trained on three popular training datasets: CC-12M, YFCC-15M, and LAION-400M. We also generate 325K, 316K, and 309K hard negative captions for a subset of the pairs. To test productivity, CREPE contains 17K image-text pairs with nine different complexities plus 278K hard negative captions with atomic, swapping, and negation foils. The datasets are generated by repurposing the Visual Genome scene graphs and region descriptions and applying handcrafted templates and GPT-3. For systematicity, we find that model performance decreases consistently when novel compositions dominate the retrieval set, with Recall@1 dropping by up to 9%. For productivity, models’ retrieval success decays as complexity increases, frequently nearing random chance at high complexity. These results hold regardless of model and training dataset size.

</details>

---

## 4. CLIP-S4: Language-Guided Self-Supervised Semantic Segmentation

- [ ] CLIP-S4: Language-Guided Self-Supervised Semantic Segmentation | https://cvpr.thecvf.com/virtual/2023/poster/21037

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21037

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Existing semantic segmentation approaches are often limited by costly pixel-wise annotations and predefined classes. In this work, we present CLIP-S^4 that leverages self-supervised pixel representation learning and vision-language models to enable various semantic segmentation tasks (e.g., unsupervised, transfer learning, language-driven segmentation) without any human annotations and unknown class information. We first learn pixel embeddings with pixel-segment contrastive learning from different augmented views of images. To further improve the pixel embeddings and enable language-driven semantic segmentation, we design two types of consistency guided by vision-language models: 1) embedding consistency, aligning our pixel embeddings to the joint feature space of a pre-trained vision-language model, CLIP; and 2) semantic consistency, forcing our model to make the same predictions as CLIP over a set of carefully designed target classes with both known and unknown prototypes. Thus, CLIP-S^4 enables a new task of class-free semantic segmentation where no unknown class information is needed during training. As a result, our approach shows consistent and substantial performance improvement over four popular benchmarks compared with the state-of-the-art unsupervised and language-driven semantic segmentation methods. More importantly, our method outperforms these methods on unknown class recognition by a large margin.

</details>

---

## 5. IFSeg: Image-Free Semantic Segmentation via Vision-Language Model

- [ ] IFSeg: Image-Free Semantic Segmentation via Vision-Language Model | https://cvpr.thecvf.com/virtual/2023/poster/21038

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21038

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language (VL) pre-training has recently gained much attention for its transferability and flexibility in novel concepts (e.g., cross-modality transfer) across various visual tasks. However, VL-driven segmentation has been under-explored, and the existing approaches still have the burden of acquiring additional training images or even segmentation annotations to adapt a VL model to downstream segmentation tasks. In this paper, we introduce a novel image-free segmentation task where the goal is to perform semantic segmentation given only a set of the target semantic categories, but without any task-specific images and annotations. To tackle this challenging task, our proposed method, coined IFSeg, generates VL-driven artificial image-segmentation pairs and updates a pre-trained VL model to a segmentation task. We construct this artificial training data by creating a 2D map of random semantic categories and another map of their corresponding word tokens. Given that a pre-trained VL model projects visual and text tokens into a common space where tokens that share the semantics are located closely, this artificially generated word map can replace the real image inputs for such a VL model. Through an extensive set of experiments, our model not only establishes an effective baseline for this novel task but also demonstrates strong performances compared to existing methods that rely on stronger supervision, such as task-specific images and segmentation masks. Code is available at https://github.com/alinlab/ifseg.

</details>

---

## 6. CLIP2: Contrastive Language-Image-Point Pretraining From Real-World Point Cloud Data

- [ ] CLIP2: Contrastive Language-Image-Point Pretraining From Real-World Point Cloud Data | https://cvpr.thecvf.com/virtual/2023/poster/21103

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21103

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training, benefiting from large-scale unlabeled text-image pairs, has demonstrated great performance in open-world vision understanding tasks. However, due to the limited Text-3D data pairs, adapting the success of 2D Vision-Language Models (VLM) to the 3D space remains an open problem. Existing works that leverage VLM for 3D understanding generally resort to constructing intermediate 2D representations for the 3D data, but at the cost of losing 3D geometry information. To take a step toward open-world 3D vision understanding, we propose Contrastive Language-Image-Point Cloud Pretraining (CLIP^2) to directly learn the transferable 3D point cloud representation in realistic scenarios with a novel proxy alignment mechanism. Specifically, we exploit naturally-existed correspondences in 2D and 3D scenarios, and build well-aligned and instance-based text-image-point proxies from those complex scenarios. On top of that, we propose a cross-modal contrastive objective to learn semantic and instance-level aligned point cloud representation. Experimental results on both indoor and outdoor scenarios show that our learned 3D representation has great transfer ability in downstream tasks, including zero-shot and few-shot 3D recognition, which boosts the state-of-the-art methods by large margins. Furthermore, we provide analyses of the capability of different representations in real scenarios and present the optional ensemble scheme.

</details>

---

## 7. You Need Multiple Exiting: Dynamic Early Exiting for Accelerating Unified Vision Language Model

- [ ] You Need Multiple Exiting: Dynamic Early Exiting for Accelerating Unified Vision Language Model | https://cvpr.thecvf.com/virtual/2023/poster/21102

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21102

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale transformer models bring significant improvements for various downstream vision language tasks with a unified architecture. The performance improvements come with increasing model size, resulting in slow inference speed and increased cost for severing. While some certain predictions benefit from the full complexity of the large-scale model, not all of input need the same amount of computation to conduct, potentially leading to computation resource waste. To handle this challenge, early exiting is proposed to adaptively allocate computational power in term of input complexity to improve inference efficiency. The existing early exiting strategies usually adopt output confidence based on intermediate layers as a proxy of input complexity to incur the decision of skipping following layers. However, such strategies cannot apply to encoder in the widely-used unified architecture with both encoder and decoder due to difficulty of output confidence estimation in the encoder. It is suboptimal in term of saving computation power to ignore the early exiting in encoder component. To handle this challenge, we propose a novel early exiting strategy for unified visual language models, which allows dynamically skip the layers in encoder and decoder simultaneously in term of input layer-wise similarities with multiple times of early exiting, namely MuE. By decomposing the image and text modalities in the encoder, MuE is flexible and can skip different layers in term of modalities, advancing the inference efficiency while minimizing performance drop. Experiments on the SNLI-VE and MS COCO datasets show that the proposed approach MuE can reduce inference time by up to 50% and 40% while maintaining 99% and 96% performance respectively.

</details>

---

## 8. Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks

- [ ] Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks | https://cvpr.thecvf.com/virtual/2023/poster/21115

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21115

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

A big convergence of language, vision, and multimodal pretraining is emerging. In this work, we introduce a general-purpose multimodal foundation model BEiT-3, which achieves excellent transfer performance on both vision and vision-language tasks. Specifically, we advance the big convergence from three aspects: backbone architecture, pretraining task, and model scaling up. We use Multiway Transformers for general-purpose modeling, where the modular architecture enables both deep fusion and modality-specific encoding. Based on the shared backbone, we perform masked “language” modeling on images (Imglish), texts (English), and image-text pairs (“parallel sentences”) in a unified manner. Experimental results show that BEiT-3 obtains remarkable performance on object detection (COCO), semantic segmentation (ADE20K), image classification (ImageNet), visual reasoning (NLVR2), visual question answering (VQAv2), image captioning (COCO), and cross-modal retrieval (Flickr30K, COCO).

</details>

---

## 9. Hierarchical Prompt Learning for Multi-Task Learning

- [ ] Hierarchical Prompt Learning for Multi-Task Learning | https://cvpr.thecvf.com/virtual/2023/poster/21143

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21143

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) can effectively transfer to various vision tasks via prompt learning. Real-world scenarios often require adapting a model to multiple similar yet distinct tasks. Existing methods focus on learning a specific prompt for each task, limiting the ability to exploit potentially shared information from other tasks. Naively training a task-shared prompt using a combination of all tasks ignores fine-grained task correlations. Significant discrepancies across tasks could cause negative transferring. Considering this, we present Hierarchical Prompt (HiPro) learning, a simple and effective method for jointly adapting a pre-trained VLM to multiple downstream tasks. Our method quantifies inter-task affinity and subsequently constructs a hierarchical task tree. Task-shared prompts learned by internal nodes explore the information within the corresponding task group, while task-individual prompts learned by leaf nodes obtain fine-grained information targeted at each task. The combination of hierarchical prompts provides high-quality content of different granularity. We evaluate HiPro on four multi-task learning datasets. The results demonstrate the effectiveness of our method.

</details>

---

## 10. DeAR: Debiasing Vision-Language Models With Additive Residuals

- [ ] DeAR: Debiasing Vision-Language Models With Additive Residuals | https://cvpr.thecvf.com/virtual/2023/poster/21167

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21167

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large pre-trained vision-language models (VLMs) reduce the time for developing predictive models for various vision-grounded language downstream tasks by providing rich, adaptable image and text representations. However, these models suffer from societal biases owing to the skewed distribution of various identity groups in the training data. These biases manifest as the skewed similarity between the representations for specific text concepts and images of people of different identity groups and, therefore, limit the usefulness of such models in real-world high-stakes applications. In this work, we present DeAR (Debiasing with Additive Residuals), a novel debiasing method that learns additive residual image representations to offset the original representations, ensuring fair output representations. In doing so, it reduces the ability of the representations to distinguish between the different identity groups. Further, we observe that the current fairness tests are performed on limited face image datasets that fail to indicate why a specific text concept should/should not apply to them. To bridge this gap and better evaluate DeAR, we introduce a new context-based bias benchmarking dataset - the Protected Attribute Tag Association (PATA) dataset for evaluating the fairness of large pre-trained VLMs. Additionally, PATA provides visual context for a diverse human population in different scenarios with both positive and negative connotations. Experimental results for fairness and zero-shot performance preservation using multiple datasets demonstrate the efficacy of our framework.

</details>

---

## 11. Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning

- [ ] Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning | https://cvpr.thecvf.com/virtual/2023/poster/21189

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21189

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Compositional Zero-Shot Learning (CZSL) aims to recognize novel concepts formed by known states and objects during training. Existing methods either learn the combined state-object representation, challenging the generalization of unseen compositions, or design two classifiers to identify state and object separately from image features, ignoring the intrinsic relationship between them. To jointly eliminate the above issues and construct a more robust CZSL system, we propose a novel framework termed Decomposed Fusion with Soft Prompt (DFSP), by involving vision-language models (VLMs) for unseen composition recognition. Specifically, DFSP constructs a vector combination of learnable soft prompts with state and object to establish the joint representation of them. In addition, a cross-modal decomposed fusion module is designed between the language and image branches, which decomposes state and object among language features instead of image features. Notably, being fused with the decomposed features, the image features can be more expressive for learning the relationship with states and objects, respectively, to improve the response of unseen compositions in the pair space, hence narrowing the domain gap between seen and unseen sets. Experimental results on three challenging benchmarks demonstrate that our approach significantly outperforms other state-of-the-art methods by large margins.

</details>

---

## 12. Towards Fast Adaptation of Pretrained Contrastive Models for Multi-Channel Video-Language Retrieval

- [ ] Towards Fast Adaptation of Pretrained Contrastive Models for Multi-Channel Video-Language Retrieval | https://cvpr.thecvf.com/virtual/2023/poster/21259

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21259

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multi-channel video-language retrieval require models to understand information from different channels (e.g. video+question, video+speech) to correctly link a video with a textual response or query. Fortunately, contrastive multimodal models are shown to be highly effective at aligning entities in images/videos and text, e.g., CLIP; text contrastive models are extensively studied recently for their strong ability of producing discriminative sentence embeddings, e.g., SimCSE. However, there is not a clear way to quickly adapt these two lines to multi-channel video-language retrieval with limited data and resources. In this paper, we identify a principled model design space with two axes: how to represent videos and how to fuse video and text information. Based on categorization of recent methods, we investigate the options of representing videos using continuous feature vectors or discrete text tokens; for the fusion method, we explore the use of a multimodal transformer or a pretrained contrastive text model. We extensively evaluate the four combinations on five video-language datasets. We surprisingly find that discrete text tokens coupled with a pretrained contrastive text model yields the best performance, which can even outperform state-of-the-art on the iVQA and How2QA datasets without additional training on millions of video-text data. Further analysis shows that this is because representing videos as text tokens captures the key visual information and text tokens are naturally aligned with text models that are strong retrievers after the contrastive pretraining process. All the empirical analysis establishes a solid foundation for future research on affordable and upgradable multimodal intelligence. The code will be released at https://github.com/XudongLinthu/upgradable-multimodal-intelligence to facilitate future research.

</details>

---

## 13. Side Adapter Network for Open-Vocabulary Semantic Segmentation

- [ ] Side Adapter Network for Open-Vocabulary Semantic Segmentation | https://cvpr.thecvf.com/virtual/2023/poster/21298

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21298

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a new framework for open-vocabulary semantic segmentation with the pre-trained vision-language model, named SAN. Our approach models the semantic segmentation task as a region recognition problem. A side network is attached to a frozen CLIP model with two branches: one for predicting mask proposals, and the other for predicting attention bias which is applied in the CLIP model to recognize the class of masks. This decoupled design has the benefit CLIP in recognizing the class of mask proposals. Since the attached side network can reuse CLIP features, it can be very light. In addition, the entire network can be trained end-to-end, allowing the side network to be adapted to the frozen CLIP model, which makes the predicted mask proposals CLIP-aware. Our approach is fast, accurate, and only adds a few additional trainable parameters. We evaluate our approach on multiple semantic segmentation benchmarks. Our method significantly outperforms other counterparts, with up to 18 times fewer trainable parameters and 19 times faster inference speed. We hope our approach will serve as a solid baseline and help ease future research in open-vocabulary semantic segmentation.

</details>

---

## 14. Seeing What You Miss: Vision-Language Pre-Training With Semantic Completion Learning

- [ ] Seeing What You Miss: Vision-Language Pre-Training With Semantic Completion Learning | https://cvpr.thecvf.com/virtual/2023/poster/21378

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21378

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Cross-modal alignment is essential for vision-language pre-training (VLP) models to learn the correct corresponding information across different modalities. For this purpose, inspired by the success of masked language modeling (MLM) tasks in the NLP pre-training area, numerous masked modeling tasks have been proposed for VLP to further promote cross-modal interactions. The core idea of previous masked modeling tasks is to focus on reconstructing the masked tokens based on visible context for learning local-to-local alignment. However, most of them pay little attention to the global semantic features generated for the masked data, resulting in a limited cross-modal alignment ability of global representations. Therefore, in this paper, we propose a novel Semantic Completion Learning (SCL) task, complementary to existing masked modeling tasks, to facilitate global-to-local alignment. Specifically, the SCL task complements the missing semantics of masked data by capturing the corresponding information from the other modality, promoting learning more representative global features which have a great impact on the performance of downstream tasks. Moreover, we present a flexible vision encoder, which enables our model to perform image-text and video-text multimodal tasks simultaneously. Experimental results show that our proposed method obtains state-of-the-art performance on various vision-language benchmarks, such as visual question answering, image-text retrieval, and video-text retrieval.

</details>

---

## 15. Leveraging per Image-Token Consistency for Vision-Language Pre-Training

- [ ] Leveraging per Image-Token Consistency for Vision-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/21382

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21382

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Most existing vision-language pre-training (VLP) approaches adopt cross-modal masked language modeling (CMLM) to learn vision-language associations. However, we find that CMLM is insufficient for this purpose according to our observations: (1) Modality bias: a considerable amount of masked tokens in CMLM can be recovered with only the language information, ignoring the visual inputs. (2) Under-utilization of the unmasked tokens: CMLM primarily focuses on the masked token but it cannot simultaneously leverage other tokens to learn vision-language associations. To handle those limitations, we propose EPIC (lEveraging Per Image-Token Consistency for vision-language pre-training). In EPIC, for each image-sentence pair, we mask tokens that are salient to the image (i.e., Saliency-based Masking Strategy) and replace them with alternatives sampled from a language model (i.e., Inconsistent Token Generation Procedure), and then the model is required to determine for each token in the sentence whether it is consistent with the image (i.e., Image-Token Consistency Task). The proposed EPIC method is easily combined with pre-training methods. Extensive experiments show that the combination of the EPIC method and state-of-the-art pre-training approaches, including ViLT, ALBEF, METER, and X-VLM, leads to significant improvements on downstream tasks. Our coude is released at https://github.com/gyhdog99/epic

</details>

---

## 16. Exploring Structured Semantic Prior for Multi Label Recognition With Incomplete Labels

- [ ] Exploring Structured Semantic Prior for Multi Label Recognition With Incomplete Labels | https://cvpr.thecvf.com/virtual/2023/poster/21424

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21424

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multi-label recognition (MLR) with incomplete labels is very challenging. Recent works strive to explore the image-to-label correspondence in the vision-language model, i.e., CLIP, to compensate for insufficient annotations. In spite of promising performance, they generally overlook the valuable prior about the label-to-label correspondence. In this paper, we advocate remedying the deficiency of label supervision for the MLR with incomplete labels by deriving a structured semantic prior about the label-to-label correspondence via a semantic prior prompter. We then present a novel Semantic Correspondence Prompt Network (SCPNet), which can thoroughly explore the structured semantic prior. A Prior-Enhanced Self-Supervised Learning method is further introduced to enhance the use of the prior. Comprehensive experiments and analyses on several widely used benchmark datasets show that our method significantly outperforms existing methods on all datasets, well demonstrating the effectiveness and the superiority of our method. Our code will be available at https://github.com/jameslahm/SCPNet.

</details>

---

## 17. Open-Vocabulary Semantic Segmentation With Mask-Adapted CLIP

- [ ] Open-Vocabulary Semantic Segmentation With Mask-Adapted CLIP | https://cvpr.thecvf.com/virtual/2023/poster/21485

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21485

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation aims to segment an image into semantic regions according to text descriptions, which may not have been seen during training. Recent two-stage methods first generate class-agnostic mask proposals and then leverage pre-trained vision-language models, e.g., CLIP, to classify masked regions. We identify the performance bottleneck of this paradigm to be the pre-trained CLIP model, since it does not perform well on masked images. To address this, we propose to finetune CLIP on a collection of masked image regions and their corresponding text descriptions. We collect training data by mining an existing image-caption dataset (e.g., COCO Captions), using CLIP to match masked image regions to nouns in the image captions. Compared with the more precise and manually annotated segmentation labels with fixed classes (e.g., COCO-Stuff), we find our noisy but diverse dataset can better retain CLIP’s generalization ability. Along with finetuning the entire model, we utilize the “blank” areas in masked images using a method we dub mask prompt tuning. Experiments demonstrate mask prompt tuning brings significant improvement without modifying any weights of CLIP, and it can further improve a fully finetuned model. In particular, when trained on COCO and evaluated on ADE20K-150, our best model achieves 29.6% mIoU, which is +8.5% higher than the previous state-of-the-art. For the first time, open-vocabulary generalist models match the performance of supervised specialist models in 2017 without dataset-specific adaptations.

</details>

---

## 18. Open-Vocabulary Point-Cloud Object Detection Without 3D Annotation

- [ ] Open-Vocabulary Point-Cloud Object Detection Without 3D Annotation | https://cvpr.thecvf.com/virtual/2023/poster/21495

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21495

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The goal of open-vocabulary detection is to identify novel objects based on arbitrary textual descriptions. In this paper, we address open-vocabulary 3D point-cloud detection by a dividing-and-conquering strategy, which involves: 1) developing a point-cloud detector that can learn a general representation for localizing various objects, and 2) connecting textual and point-cloud representations to enable the detector to classify novel object categories based on text prompting. Specifically, we resort to rich image pre-trained models, by which the point-cloud detector learns localizing objects under the supervision of predicted 2D bounding boxes from 2D pre-trained detectors. Moreover, we propose a novel de-biased triplet cross-modal contrastive learning to connect the modalities of image, point-cloud and text, thereby enabling the point-cloud detector to benefit from vision-language pre-trained models, i.e., CLIP. The novel use of image and vision-language pre-trained models for point-cloud detectors allows for open-vocabulary 3D object detection without the need for 3D annotations. Experiments demonstrate that the proposed method improves at least 3.03 points and 7.47 points over a wide range of baselines on the ScanNet and SUN RGB-D datasets, respectively. Furthermore, we provide a comprehensive analysis to explain why our approach works.

</details>

---

## 19. Q: How To Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!

- [ ] Q: How To Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images! | https://cvpr.thecvf.com/virtual/2023/poster/21499

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21499

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Finetuning a large vision language model (VLM) on a target dataset after large scale pretraining is a dominant paradigm in visual question answering (VQA). Datasets for specialized tasks such as knowledge-based VQA or VQA in non natural-image domains are orders of magnitude smaller than those for general-purpose VQA. While collecting additional labels for specialized tasks or domains can be challenging, unlabeled images are often available. We introduce SelTDA (Self-Taught Data Augmentation), a strategy for finetuning large VLMs on small-scale VQA datasets. SelTDA uses the VLM and target dataset to build a teacher model that can generate question-answer pseudolabels directly conditioned on an image alone, allowing us to pseudolabel unlabeled images. SelTDA then finetunes the initial VLM on the original dataset augmented with freshly pseudolabeled images. We describe a series of experiments showing that our self-taught data augmentation increases robustness to adversarially searched questions, counterfactual examples, and rephrasings, it improves domain generalization, and results in greater retention of numerical reasoning skills. The proposed strategy requires no additional annotations or architectural modifications, and is compatible with any modern encoder-decoder multimodal transformer. Code available at https://github.com/codezakh/SelTDA

</details>

---

## 20. Improving Commonsense in Vision-Language Models via Knowledge Graph Riddles

- [ ] Improving Commonsense in Vision-Language Models via Knowledge Graph Riddles | https://cvpr.thecvf.com/virtual/2023/poster/21514

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21514

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper focuses on analyzing and improving the commonsense ability of recent popular vision-language (VL) models. Despite the great success, we observe that existing VL-models still lack commonsense knowledge/reasoning ability (e.g., “Lemons are sour”), which is a vital component towards artificial general intelligence. Through our analysis, we find one important reason is that existing large-scale VL datasets do not contain much commonsense knowledge, which motivates us to improve the commonsense of VL-models from the data perspective. Rather than collecting a new VL training dataset, we propose a more scalable strategy, i.e., “Data Augmentation with kNowledge graph linearization for CommonsensE capability” (DANCE). It can be viewed as one type of data augmentation technique, which can inject commonsense knowledge into existing VL datasets on the fly during training. More specifically, we leverage the commonsense knowledge graph (e.g., ConceptNet) and create variants of text description in VL datasets via bidirectional sub-graph sequentialization. For better commonsense evaluation, we further propose the first retrieval-based commonsense diagnostic benchmark. By conducting extensive experiments on some representative VL-models, we demonstrate that our DANCE technique is able to significantly improve the commonsense ability while maintaining the performance on vanilla retrieval tasks.

</details>

---

## 21. HierVL: Learning Hierarchical Video-Language Embeddings

- [ ] HierVL: Learning Hierarchical Video-Language Embeddings | https://cvpr.thecvf.com/virtual/2023/poster/21523

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21523

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video-language embeddings are a promising avenue for injecting semantics into visual representations, but existing methods capture only short-term associations between seconds-long video clips and their accompanying text. We propose HierVL, a novel hierarchical video-language embedding that simultaneously accounts for both long-term and short-term associations. As training data, we take videos accompanied by timestamped text descriptions of human actions, together with a high-level text summary of the activity throughout the long video (as are available in Ego4D). We introduce a hierarchical contrastive training objective that encourages text-visual alignment at both the clip level and video level. While the clip-level constraints use the step-by-step descriptions to capture what is happening in that instant, the video-level constraints use the summary text to capture why it is happening, i.e., the broader context for the activity and the intent of the actor. Our hierarchical scheme yields a clip representation that outperforms its single-level counterpart, as well as a long-term video representation that achieves SotA results on tasks requiring long-term video modeling. HierVL successfully transfers to multiple challenging downstream tasks (in EPIC-KITCHENS-100, Charades-Ego, HowTo100M) in both zero-shot and fine-tuned settings.

</details>

---

## 22. Improving Visual Grounding by Encouraging Consistent Gradient-Based Explanations

- [ ] Improving Visual Grounding by Encouraging Consistent Gradient-Based Explanations | https://cvpr.thecvf.com/virtual/2023/poster/21569

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21569

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We propose a margin-based loss for tuning joint vision-language models so that their gradient-based explanations are consistent with region-level annotations provided by humans for relatively smaller grounding datasets. We refer to this objective as Attention Mask Consistency (AMC) and demonstrate that it produces superior visual grounding results than previous methods that rely on using vision-language models to score the outputs of object detectors. Particularly, a model trained with AMC on top of standard vision-language modeling objectives obtains a state-of-the-art accuracy of 86.49% in the Flickr30k visual grounding benchmark, an absolute improvement of 5.38% when compared to the best previous model trained under the same level of supervision. Our approach also performs exceedingly well on established benchmarks for referring expression comprehension where it obtains 80.34% accuracy in the easy test of RefCOCO+, and 64.55% in the difficult split. AMC is effective, easy to implement, and is general as it can be adopted by any vision-language model, and can use any type of region annotations.

</details>

---

## 23. Bidirectional Cross-Modal Knowledge Exploration for Video Recognition With Pre-Trained Vision-Language Models

- [ ] Bidirectional Cross-Modal Knowledge Exploration for Video Recognition With Pre-Trained Vision-Language Models | https://cvpr.thecvf.com/virtual/2023/poster/21577

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21577

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) pre-trained on large-scale image-text pairs have demonstrated impressive transferability on various visual tasks. Transferring knowledge from such powerful VLMs is a promising direction for building effective video recognition models. However, current exploration in this field is still limited. We believe that the greatest value of pre-trained VLMs lies in building a bridge between visual and textual domains. In this paper, we propose a novel framework called BIKE, which utilizes the cross-modal bridge to explore bidirectional knowledge: i) We introduce the Video Attribute Association mechanism, which leverages the Video-to-Text knowledge to generate textual auxiliary attributes for complementing video recognition. ii) We also present a Temporal Concept Spotting mechanism that uses the Text-to-Video expertise to capture temporal saliency in a parameter-free manner, leading to enhanced video representation. Extensive studies on six popular video datasets, including Kinetics-400 & 600, UCF-101, HMDB-51, ActivityNet and Charades, show that our method achieves state-of-the-art performance in various recognition scenarios, such as general, zero-shot, and few-shot video recognition. Our best model achieves a state-of-the-art accuracy of 88.6% on the challenging Kinetics-400 using the released CLIP model. The code is available at https://github.com/whwu95/BIKE.

</details>

---

## 24. Policy Adaptation From Foundation Model Feedback

- [ ] Policy Adaptation From Foundation Model Feedback | https://cvpr.thecvf.com/virtual/2023/poster/21581

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21581

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent progress on vision-language foundation models have brought significant advancement to building general-purpose robots. By using the pre-trained models to encode the scene and instructions as inputs for decision making, the instruction-conditioned policy can generalize across different objects and tasks. While this is encouraging, the policy still fails in most cases given an unseen task or environment. In this work, we propose Policy Adaptation from Foundation model Feedback (PAFF). When deploying the trained policy to a new task or a new environment, we first let the policy play with randomly generated instructions to record the demonstrations. While the execution could be wrong, we can use the pre-trained foundation models to provide feedback to relabel the demonstrations. This automatically provides new pairs of demonstration-instruction data for policy fine-tuning. We evaluate our method on a broad range of experiments with the focus on generalization on unseen objects, unseen tasks, unseen environments, and sim-to-real transfer. We show PAFF improves baselines by a large margin in all cases.

</details>

---

## 25. HOICLIP: Efficient Knowledge Transfer for HOI Detection With Vision-Language Models

- [ ] HOICLIP: Efficient Knowledge Transfer for HOI Detection With Vision-Language Models | https://cvpr.thecvf.com/virtual/2023/poster/21598

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21598

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Human-Object Interaction (HOI) detection aims to localize human-object pairs and recognize their interactions. Recently, Contrastive Language-Image Pre-training (CLIP) has shown great potential in providing interaction prior for HOI detectors via knowledge distillation. However, such approaches often rely on large-scale training data and suffer from inferior performance under few/zero-shot scenarios. In this paper, we propose a novel HOI detection framework that efficiently extracts prior knowledge from CLIP and achieves better generalization. In detail, we first introduce a novel interaction decoder to extract informative regions in the visual feature map of CLIP via a cross-attention mechanism, which is then fused with the detection backbone by a knowledge integration block for more accurate human-object pair detection. In addition, prior knowledge in CLIP text encoder is leveraged to generate a classifier by embedding HOI descriptions. To distinguish fine-grained interactions, we build a verb classifier from training data via visual semantic arithmetic and a lightweight verb representation adapter. Furthermore, we propose a training-free enhancement to exploit global HOI predictions from CLIP. Extensive experiments demonstrate that our method outperforms the state of the art by a large margin on various settings, e.g. +4.04 mAP on HICO-Det. The source code is available in https://github.com/Artanic30/HOICLIP.

</details>

---

## 26. MAP: Multimodal Uncertainty-Aware Vision-Language Pre-Training Model

- [ ] MAP: Multimodal Uncertainty-Aware Vision-Language Pre-Training Model | https://cvpr.thecvf.com/virtual/2023/poster/21611

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21611

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Multimodal semantic understanding often has to deal with uncertainty, which means the obtained messages tend to refer to multiple targets. Such uncertainty is problematic for our interpretation, including inter- and intra-modal uncertainty. Little effort has studied the modeling of this uncertainty, particularly in pre-training on unlabeled datasets and fine-tuning in task-specific downstream datasets. In this paper, we project the representations of all modalities as probabilistic distributions via a Probability Distribution Encoder (PDE) by utilizing sequence-level interactions. Compared to the exiting deterministic methods, such uncertainty modeling can convey richer multimodal semantic information and more complex relationships. Furthermore, we integrate uncertainty modeling with popular pre-training frameworks and propose suitable pre-training tasks: Distribution-based Vision-Language Contrastive learning (D-VLC), Distribution-based Masked Language Modeling (D-MLM), and Distribution-based Image-Text Matching (D-ITM). The fine-tuned models are applied to challenging downstream tasks, including image-text retrieval, visual question answering, visual reasoning, and visual entailment, and achieve state-of-the-art results.

</details>

---

## 27. FashionSAP: Symbols and Attributes Prompt for Fine-Grained Fashion Vision-Language Pre-Training

- [ ] FashionSAP: Symbols and Attributes Prompt for Fine-Grained Fashion Vision-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/21620

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21620

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Fashion vision-language pre-training models have shown efficacy for a wide range of downstream tasks. However, general vision-language pre-training models pay less attention to fine-grained domain features, while these features are important in distinguishing the specific domain tasks from general tasks. We propose a method for fine-grained fashion vision-language pre-training based on fashion Symbols and Attributes Prompt (FashionSAP) to model fine-grained multi-modalities fashion attributes and characteristics. Firstly, we propose the fashion symbols, a novel abstract fashion concept layer, to represent different fashion items and to generalize various kinds of fine-grained fashion features, making modelling fine-grained attributes more effective. Secondly, the attributes prompt method is proposed to make the model learn specific attributes of fashion items explicitly. We design proper prompt templates according to the format of fashion data. Comprehensive experiments are conducted on two public fashion benchmarks, i.e., FashionGen and FashionIQ, and FashionSAP gets SOTA performances for four popular fashion tasks. The ablation study also shows the proposed abstract fashion symbols, and the attribute prompt method enables the model to acquire fine-grained semantics in the fashion domain effectively. The obvious performance gains from FashionSAP provide a new baseline for future fashion task research.

</details>

---

## 28. Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective

- [ ] Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective | https://cvpr.thecvf.com/virtual/2023/poster/21633

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21633

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We aim at advancing blind image quality assessment (BIQA), which predicts the human perception of image quality without any reference information. We develop a general and automated multitask learning scheme for BIQA to exploit auxiliary knowledge from other tasks, in a way that the model parameter sharing and the loss weighting are determined automatically. Specifically, we first describe all candidate label combinations (from multiple tasks) using a textual template, and compute the joint probability from the cosine similarities of the visual-textual embeddings. Predictions of each task can be inferred from the joint distribution, and optimized by carefully designed loss functions. Through comprehensive experiments on learning three tasks - BIQA, scene classification, and distortion type identification, we verify that the proposed BIQA method 1) benefits from the scene classification and distortion type identification tasks and outperforms the state-of-the-art on multiple IQA datasets, 2) is more robust in the group maximum differentiation competition, and 3) realigns the quality annotations from different IQA datasets more effectively. The source code is available at https://github.com/zwx8981/LIQE.

</details>

---

## 29. Open-Vocabulary Attribute Detection

- [ ] Open-Vocabulary Attribute Detection | https://cvpr.thecvf.com/virtual/2023/poster/21647

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21647

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language modeling has enabled open-vocabulary tasks where predictions can be queried using any text prompt in a zero-shot manner. Existing open-vocabulary tasks focus on object classes, whereas research on object attributes is limited due to the lack of a reliable attribute-focused evaluation benchmark. This paper introduces the Open-Vocabulary Attribute Detection (OVAD) task and the corresponding OVAD benchmark. The objective of the novel task and benchmark is to probe object-level attribute information learned by vision-language models. To this end, we created a clean and densely annotated test set covering 117 attribute classes on the 80 object classes of MS COCO. It includes positive and negative annotations, which enables open-vocabulary evaluation. Overall, the benchmark consists of 1.4 million annotations. For reference, we provide a first baseline method for open-vocabulary attribute detection. Moreover, we demonstrate the benchmark’s value by studying the attribute detection performance of several foundation models.

</details>

---

## 30. A-Cap: Anticipation Captioning With Commonsense Knowledge

- [ ] A-Cap: Anticipation Captioning With Commonsense Knowledge | https://cvpr.thecvf.com/virtual/2023/poster/21652

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21652

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Humans possess the capacity to reason about the future based on a sparse collection of visual cues acquired over time. In order to emulate this ability, we introduce a novel task called Anticipation Captioning, which generates a caption for an unseen oracle image using a sparsely temporally-ordered set of images. To tackle this new task, we propose a model called A-CAP, which incorporates commonsense knowledge into a pre-trained vision-language model, allowing it to anticipate the caption. Through both qualitative and quantitative evaluations on a customized visual storytelling dataset, A-CAP outperforms other image captioning methods and establishes a strong baseline for anticipation captioning. We also address the challenges inherent in this task.

</details>

---

## 31. MaPLe: Multi-Modal Prompt Learning

- [ ] MaPLe: Multi-Modal Prompt Learning | https://cvpr.thecvf.com/virtual/2023/poster/21684

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21684

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language (V-L) models such as CLIP have shown excellent generalization ability to downstream tasks. However, they are sensitive to the choice of input text prompts and require careful selection of prompt templates to perform well. Inspired by the Natural Language Processing (NLP) literature, recent CLIP adaptation approaches learn prompts as the textual inputs to fine-tune CLIP for downstream tasks. We note that using prompting to adapt representations in a single branch of CLIP (language or vision) is sub-optimal since it does not allow the flexibility to dynamically adjust both representation spaces on a downstream task. In this work, we propose Multi-modal Prompt Learning (MaPLe) for both vision and language branches to improve alignment between the vision and language representations. Our design promotes strong coupling between the vision-language prompts to ensure mutual synergy and discourages learning independent uni-modal solutions. Further, we learn separate prompts across different early stages to progressively model the stage-wise feature relationships to allow rich context learning. We evaluate the effectiveness of our approach on three representative tasks of generalization to novel classes, new target datasets and unseen domain shifts. Compared with the state-of-the-art method Co-CoOp, MaPLe exhibits favorable performance and achieves an absolute gain of 3.45% on novel classes and 2.72% on overall harmonic-mean, averaged over 11 diverse image recognition datasets. Our code and pre-trained models are available at https://github.com/muzairkhattak/multimodal-prompt-learning.

</details>

---

## 32. Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks

- [ ] Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks | https://cvpr.thecvf.com/virtual/2023/poster/21740

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21740

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Despite the remarkable success of foundation models, their task-specific fine-tuning paradigm makes them inconsistent with the goal of general perception modeling. The key to eliminating this inconsistency is to use generalist models for general task modeling. However, existing attempts at generalist models are inadequate in both versatility and performance. In this paper, we propose Uni-Perceiver v2, which is the first generalist model capable of handling major large-scale vision and vision-language tasks with competitive performance. Specifically, images are encoded as general region proposals, while texts are encoded via a Transformer-based language model. The encoded representations are transformed by a task-agnostic decoder. Different tasks are formulated as a unified maximum likelihood estimation problem. We further propose an effective optimization technique named Task-Balanced Gradient Normalization to ensure stable multi-task learning with an unmixed sampling strategy, which is helpful for tasks requiring large batch-size training. After being jointly trained on various tasks, Uni-Perceiver v2 is capable of directly handling downstream tasks without any task-specific adaptation. Results show that Uni-Perceiver v2 outperforms all existing generalist models in both versatility and performance. Meanwhile, compared with the commonly-recognized strong baselines that require tasks-specific fine-tuning, Uni-Perceiver v2 achieves competitive performance on a broad range of vision and vision-language tasks.

</details>

---

## 33. Test of Time: Instilling Video-Language Models With a Sense of Time

- [ ] Test of Time: Instilling Video-Language Models With a Sense of Time | https://cvpr.thecvf.com/virtual/2023/poster/21817

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21817

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Modelling and understanding time remains a challenge in contemporary video understanding models. With language emerging as a key driver towards powerful generalization, it is imperative for foundational video-language models to have a sense of time. In this paper, we consider a specific aspect of temporal understanding: consistency of time order as elicited by before/after relations. We establish that seven existing video-language models struggle to understand even such simple temporal relations. We then question whether it is feasible to equip these foundational models with temporal awareness without re-training them from scratch. Towards this, we propose a temporal adaptation recipe on top of one such model, VideoCLIP, based on post-pretraining on a small amount of video-text data. We conduct a zero-shot evaluation of the adapted models on six datasets for three downstream tasks which require varying degrees of time awareness. We observe encouraging performance gains especially when the task needs higher time awareness. Our work serves as a first step towards probing and instilling a sense of time in existing video-language models without the need for data and compute-intense training from scratch.

</details>

---

## 34. Weakly Supervised Temporal Sentence Grounding With Uncertainty-Guided Self-Training

- [ ] Weakly Supervised Temporal Sentence Grounding With Uncertainty-Guided Self-Training | https://cvpr.thecvf.com/virtual/2023/poster/21822

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21822

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The task of weakly supervised temporal sentence grounding aims at finding the corresponding temporal moments of a language description in the video, given video-language correspondence only at video-level. Most existing works select mismatched video-language pairs as negative samples and train the model to generate better positive proposals that are distinct from the negative ones. However, due to the complex temporal structure of videos, proposals distinct from the negative ones may correspond to several video segments but not necessarily the correct ground truth. To alleviate this problem, we propose an uncertainty-guided self-training technique to provide extra self-supervision signals to guide the weakly-supervised learning. The self-training process is based on teacher-student mutual learning with weak-strong augmentation, which enables the teacher network to generate relatively more reliable outputs compared to the student network, so that the student network can learn from the teacher’s output. Since directly applying existing self-training methods in this task easily causes error accumulation, we specifically design two techniques in our self-training method: (1) we construct a Bayesian teacher network, leveraging its uncertainty as a weight to suppress the noisy teacher supervisory signals; (2) we leverage the cycle consistency brought by temporal data augmentation to perform mutual learning between the two networks. Experiments demonstrate our method’s superiority on Charades-STA and ActivityNet Captions datasets. We also show in the experiment that our self-training method can be applied to improve the performance of multiple backbone methods.

</details>

---

## 35. Language-Guided Audio-Visual Source Separation via Trimodal Consistency

- [ ] Language-Guided Audio-Visual Source Separation via Trimodal Consistency | https://cvpr.thecvf.com/virtual/2023/poster/21840

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21840

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We propose a self-supervised approach for learning to perform audio source separation in videos based on natural language queries, using only unlabeled video and audio pairs as training data. A key challenge in this task is learning to associate the linguistic description of a sound-emitting object to its visual features and the corresponding components of the audio waveform, all without access to annotations during training. To overcome this challenge, we adapt off-the-shelf vision-language foundation models to provide pseudo-target supervision via two novel loss functions and encourage a stronger alignment between the audio, visual and natural language modalities. During inference, our approach can separate sounds given text, video and audio input, or given text and audio input alone. We demonstrate the effectiveness of our self-supervised approach on three audio-visual separation datasets, including MUSIC, SOLOS and AudioSet, where we outperform state-of-the-art strongly supervised approaches despite not using object detectors or text labels during training. Finally, we also include samples of our separated audios in the supplemental for reference.

</details>

---

## 36. Turning a CLIP Model Into a Scene Text Detector

- [ ] Turning a CLIP Model Into a Scene Text Detector | https://cvpr.thecvf.com/virtual/2023/poster/21866

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21866

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The recent large-scale Contrastive Language-Image Pretraining (CLIP) model has shown great potential in various downstream tasks via leveraging the pretrained vision and language knowledge. Scene text, which contains rich textual and visual information, has an inherent connection with a model like CLIP. Recently, pretraining approaches based on vision language models have made effective progresses in the field of text detection. In contrast to these works, this paper proposes a new method, termed TCM, focusing on Turning the CLIP Model directly for text detection without pretraining process. We demonstrate the advantages of the proposed TCM as follows: (1) The underlying principle of our framework can be applied to improve existing scene text detector. (2) It facilitates the few-shot training capability of existing methods, e.g., by using 10% of labeled data, we significantly improve the performance of the baseline method with an average of 22% in terms of the F-measure on 4 benchmarks. (3) By turning the CLIP model into existing scene text detection methods, we further achieve promising domain adaptation ability. The code will be publicly released at https://github.com/wenwenyu/TCM.

</details>

---

## 37. VLPD: Context-Aware Pedestrian Detection via Vision-Language Semantic Self-Supervision

- [ ] VLPD: Context-Aware Pedestrian Detection via Vision-Language Semantic Self-Supervision | https://cvpr.thecvf.com/virtual/2023/poster/21872

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21872

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Detecting pedestrians accurately in urban scenes is significant for realistic applications like autonomous driving or video surveillance. However, confusing human-like objects often lead to wrong detections, and small scale or heavily occluded pedestrians are easily missed due to their unusual appearances. To address these challenges, only object regions are inadequate, thus how to fully utilize more explicit and semantic contexts becomes a key problem. Meanwhile, previous context-aware pedestrian detectors either only learn latent contexts with visual clues, or need laborious annotations to obtain explicit and semantic contexts. Therefore, we propose in this paper a novel approach via Vision-Language semantic self-supervision for context-aware Pedestrian Detection (VLPD) to model explicitly semantic contexts without any extra annotations. Firstly, we propose a self-supervised Vision-Language Semantic (VLS) segmentation method, which learns both fully-supervised pedestrian detection and contextual segmentation via self-generated explicit labels of semantic classes by vision-language models. Furthermore, a self-supervised Prototypical Semantic Contrastive (PSC) learning method is proposed to better discriminate pedestrians and other classes, based on more explicit and semantic contexts obtained from VLS. Extensive experiments on popular benchmarks show that our proposed VLPD achieves superior performances over the previous state-of-the-arts, particularly under challenging circumstances like small scale and heavy occlusion. Code is available at https://github.com/lmy98129/VLPD.

</details>

---

## 38. MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering

- [ ] MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering | https://cvpr.thecvf.com/virtual/2023/poster/21937

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21937

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recently, finetuning pretrained vision-language models (VLMs) has been a prevailing paradigm for achieving state-of-the-art performance in VQA. However, as VLMs scale, it becomes computationally expensive, storage inefficient, and prone to overfitting when tuning full model parameters for a specific task in low-resource settings. Although current parameter-efficient tuning methods dramatically reduce the number of tunable parameters, there still exists a significant performance gap with full finetuning. In this paper, we propose MixPHM, a redundancy-aware parameter-efficient tuning method that outperforms full finetuning in low-resource VQA. Specifically, MixPHM is a lightweight module implemented by multiple PHM-experts in a mixture-of-experts manner. To reduce parameter redundancy, we reparameterize expert weights in a low-rank subspace and share part of the weights inside and across MixPHM. Moreover, based on our quantitative analysis of representation redundancy, we propose Redundancy Regularization, which facilitates MixPHM to reduce task-irrelevant redundancy while promoting task-relevant correlation. Experiments conducted on VQA v2, GQA, and OK-VQA with different low-resource settings show that our MixPHM outperforms state-of-the-art parameter-efficient methods and is the only one consistently surpassing full finetuning.

</details>

---

## 39. 3D Concept Learning and Reasoning From Multi-View Images

- [ ] 3D Concept Learning and Reasoning From Multi-View Images | https://cvpr.thecvf.com/virtual/2023/poster/21949

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21949

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Humans are able to accurately reason in 3D by gathering multi-view observations of the surrounding world. Inspired by this insight, we introduce a new large-scale benchmark for 3D multi-view visual question answering (3DMV-VQA). This dataset is collected by an embodied agent actively moving and capturing RGB images in an environment using the Habitat simulator. In total, it consists of approximately 5k scenes, 600k images, paired with 50k questions. We evaluate various state-of-the-art models for visual reasoning on our benchmark and find that they all perform poorly. We suggest that a principled approach for 3D reasoning from multi-view images should be to infer a compact 3D representation of the world from the multi-view images, which is further grounded on open-vocabulary semantic concepts, and then to execute reasoning on these 3D representations. As the first step towards this approach, we propose a novel 3D concept learning and reasoning (3D-CLR) framework that seamlessly combines these components via neural fields, 2D pre-trained vision-language models, and neural reasoning operators. Experimental results suggest that our framework outperforms baseline models by a large margin, but the challenge remains largely unsolved. We further perform an in-depth analysis of the challenges and highlight potential future directions.

</details>

---

## 40. CapDet: Unifying Dense Captioning and Open-World Detection Pretraining

- [ ] CapDet: Unifying Dense Captioning and Open-World Detection Pretraining | https://cvpr.thecvf.com/virtual/2023/poster/21960

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21960

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Benefiting from large-scale vision-language pre-training on image-text pairs, open-world detection methods have shown superior generalization ability under the zero-shot or few-shot detection settings. However, a pre-defined category space is still required during the inference stage of existing methods and only the objects belonging to that space will be predicted. To introduce a “real” open-world detector, in this paper, we propose a novel method named CapDet to either predict under a given category list or directly generate the category of predicted bounding boxes. Specifically, we unify the open-world detection and dense caption tasks into a single yet effective framework by introducing an additional dense captioning head to generate the region-grounded captions. Besides, adding the captioning task will in turn benefit the generalization of detection performance since the captioning dataset covers more concepts. Experiment results show that by unifying the dense caption task, our CapDet has obtained significant performance improvements (e.g., +2.1% mAP on LVIS rare classes) over the baseline method on LVIS (1203 classes). Besides, our CapDet also achieves state-of-the-art performance on dense captioning tasks, e.g., 15.44% mAP on VG V1.2 and 13.98% on the VG-COCO dataset.

</details>

---

## 41. Open-Category Human-Object Interaction Pre-Training via Language Modeling Framework

- [ ] Open-Category Human-Object Interaction Pre-Training via Language Modeling Framework | https://cvpr.thecvf.com/virtual/2023/poster/21970

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21970

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Human-object interaction (HOI) has long been plagued by the conflict between limited supervised data and a vast number of possible interaction combinations in real life. Current methods trained from closed-set data predict HOIs as fixed-dimension logits, which restricts their scalability to open-set categories. To address this issue, we introduce OpenCat, a language modeling framework that reformulates HOI prediction as sequence generation. By converting HOI triplets into a token sequence through a serialization scheme, our model is able to exploit the open-set vocabulary of the language modeling framework to predict novel interaction classes with a high degree of freedom. In addition, inspired by the great success of vision-language pre-training, we collect a large amount of weakly-supervised data related to HOI from image-caption pairs, and devise several auxiliary proxy tasks, including soft relational matching and human-object relation prediction, to pre-train our model. Extensive experiments show that our OpenCat significantly boosts HOI performance, particularly on a broad range of rare and unseen categories.

</details>

---

## 42. NeRDi: Single-View NeRF Synthesis With Language-Guided Diffusion As General Image Priors

- [ ] NeRDi: Single-View NeRF Synthesis With Language-Guided Diffusion As General Image Priors | https://cvpr.thecvf.com/virtual/2023/poster/21989

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/21989

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

2D-to-3D reconstruction is an ill-posed problem, yet humans are good at solving this problem due to their prior knowledge of the 3D world developed over years. Driven by this observation, we propose NeRDi, a single-view NeRF synthesis framework with general image priors from 2D diffusion models. Formulating single-view reconstruction as an image-conditioned 3D generation problem, we optimize the NeRF representations by minimizing a diffusion loss on its arbitrary view renderings with a pretrained image diffusion model under the input-view constraint. We leverage off-the-shelf vision-language models and introduce a two-section language guidance as conditioning inputs to the diffusion model. This is essentially helpful for improving multiview content coherence as it narrows down the general image prior conditioned on the semantic and visual features of the single-view input image. Additionally, we introduce a geometric loss based on estimated depth maps to regularize the underlying 3D geometry of the NeRF. Experimental results on the DTU MVS dataset show that our method can synthesize novel views with higher quality even compared to existing methods trained on this dataset. We also demonstrate our generalizability in zero-shot NeRF synthesis for in-the-wild images.

</details>

---

## 43. Meta-Personalizing Vision-Language Models To Find Named Instances in Video

- [ ] Meta-Personalizing Vision-Language Models To Find Named Instances in Video | https://cvpr.thecvf.com/virtual/2023/poster/22007

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22007

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models (VLM) have shown impressive results for language-guided search applications. While these models allow category-level queries, they currently struggle with personalized searches for moments in a video where a specific object instance such as “My dog Biscuit” appears. We present the following three contributions to address this problem. First, we describe a method to meta-personalize a pre-trained VLM, i.e., learning how to learn to personalize a VLM at test time to search in video. Our method extends the VLM’s token vocabulary by learning novel word embeddings specific to each instance. To capture only instance-specific features, we represent each instance embedding as a combination of shared and learned global category features. Second, we propose to learn such personalization without explicit human supervision. Our approach automatically identifies moments of named visual instances in video using transcripts and vision-language similarity in the VLM’s embedding space. Finally, we introduce This-Is-My, a personal video instance retrieval benchmark. We evaluate our approach on This-Is-My and DeepFashion2 and show that we obtain a 15% relative improvement over the state of the art on the latter dataset.

</details>

---

## 44. Multi-Modal Representation Learning With Text-Driven Soft Masks

- [ ] Multi-Modal Representation Learning With Text-Driven Soft Masks | https://cvpr.thecvf.com/virtual/2023/poster/22083

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22083

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We propose a visual-linguistic representation learning approach within a self-supervised learning framework by introducing a new operation, loss, and data augmentation strategy. First, we generate diverse features for the image-text matching (ITM) task via soft-masking the regions in an image, which are most relevant to a certain word in the corresponding caption, instead of completely removing them. Since our framework relies only on image-caption pairs with no fine-grained annotations, we identify the relevant regions to each word by computing the word-conditional visual attention using multi-modal encoder. Second, we encourage the model to focus more on hard but diverse examples by proposing a focal loss for the image-text contrastive learning (ITC) objective, which alleviates the inherent limitations of overfitting and bias issues. Last, we perform multi-modal data augmentations for self-supervised learning via mining various examples by masking texts and rendering distortions on images. We show that the combination of these three innovations is effective for learning a pretrained model, leading to outstanding performance on multiple vision-language downstream tasks.

</details>

---

## 45. Open-Set Fine-Grained Retrieval via Prompting Vision-Language Evaluator

- [ ] Open-Set Fine-Grained Retrieval via Prompting Vision-Language Evaluator | https://cvpr.thecvf.com/virtual/2023/poster/22122

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22122

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Open-set fine-grained retrieval is an emerging challenge that requires an extra capability to retrieve unknown subcategories during evaluation. However, current works are rooted in the close-set scenarios, where all the subcategories are pre-defined, and make it hard to capture discriminative knowledge from unknown subcategories, consequently failing to handle the inevitable unknown subcategories in open-world scenarios. In this work, we propose a novel Prompting vision-Language Evaluator (PLEor) framework based on the recently introduced contrastive language-image pretraining (CLIP) model, for open-set fine-grained retrieval. PLEor could leverage pre-trained CLIP model to infer the discrepancies encompassing both pre-defined and unknown subcategories, called category-specific discrepancies, and transfer them to the backbone network trained in the close-set scenarios. To make pre-trained CLIP model sensitive to category-specific discrepancies, we design a dual prompt scheme to learn a vision prompt specifying the category-specific discrepancies, and turn random vectors with category names in a text prompt into category-specific discrepancy descriptions. Moreover, a vision-language evaluator is proposed to semantically align the vision and text prompts based on CLIP model, and reinforce each other. In addition, we propose an open-set knowledge transfer to transfer the category-specific discrepancies into the backbone network using knowledge distillation mechanism. A variety of quantitative and qualitative experiments show that our PLEor achieves promising performance on open-set fine-grained retrieval datasets.

</details>

---

## 46. Revisiting Multimodal Representation in Contrastive Learning: From Patch and Token Embeddings to Finite Discrete Tokens

- [ ] Revisiting Multimodal Representation in Contrastive Learning: From Patch and Token Embeddings to Finite Discrete Tokens | https://cvpr.thecvf.com/virtual/2023/poster/22172

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22172

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive learning-based vision-language pre-training approaches, such as CLIP, have demonstrated great success in many vision-language tasks. These methods achieve cross-modal alignment by encoding a matched image-text pair with similar feature embeddings, which are generated by aggregating information from visual patches and language tokens. However, direct aligning cross-modal information using such representations is challenging, as visual patches and text tokens differ in semantic levels and granularities. To alleviate this issue, we propose a Finite Discrete Tokens (FDT) based multimodal representation. FDT is a set of learnable tokens representing certain visual-semantic concepts. Both images and texts are embedded using shared FDT by first grounding multimodal inputs to FDT space and then aggregating the activated FDT representations. The matched visual and semantic concepts are enforced to be represented by the same set of discrete tokens by a sparse activation constraint. As a result, the granularity gap between the two modalities is reduced. Through both quantitative and qualitative analyses, we demonstrate that using FDT representations in CLIP-style models improves cross-modal alignment and performance in visual recognition and vision-language downstream tasks. Furthermore, we show that our method can learn more comprehensive representations, and the learned FDT capture meaningful cross-modal correspondence, ranging from objects to actions and attributes.

</details>

---

## 47. All in One: Exploring Unified Video-Language Pre-Training

- [ ] All in One: Exploring Unified Video-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/22225

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22225

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Mainstream Video-Language Pre-training models consist of three parts, a video encoder, a text encoder, and a video-text fusion Transformer. They pursue better performance via utilizing heavier unimodal encoders or multimodal fusion Transformers, resulting in increased parameters with lower efficiency in downstream tasks. In this work, we for the first time introduce an end-to-end video-language model, namely all-in-one Transformer, that embeds raw video and textual signals into joint representations using a unified backbone architecture. We argue that the unique temporal information of video data turns out to be a key barrier hindering the design of a modality-agnostic Transformer. To overcome the challenge, we introduce a novel and effective token rolling operation to encode temporal representations from video clips in a non-parametric manner. The careful design enables the representation learning of both video-text multimodal inputs and unimodal inputs using a unified backbone model. Our pre-trained all-in-one Transformer is transferred to various downstream video-text tasks after fine-tuning, including text-video retrieval, video-question answering, multiple choice and visual commonsense reasoning. State-of-the-art performances with the minimal model FLOPs on nine datasets demonstrate the superiority of our method compared to the competitive counterparts.

</details>

---

## 48. DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-Training via Word-Region Alignment

- [ ] DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-Training via Word-Region Alignment | https://cvpr.thecvf.com/virtual/2023/poster/22231

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22231

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper presents DetCLIPv2, an efficient and scalable training framework that incorporates large-scale image-text pairs to achieve open-vocabulary object detection (OVD). Unlike previous OVD frameworks that typically rely on a pre-trained vision-language model (e.g., CLIP) or exploit image-text pairs via a pseudo labeling process, DetCLIPv2 directly learns the fine-grained word-region alignment from massive image-text pairs in an end-to-end manner. To accomplish this, we employ a maximum word-region similarity between region proposals and textual words to guide the contrastive objective. To enable the model to gain localization capability while learning broad concepts, DetCLIPv2 is trained with a hybrid supervision from detection, grounding and image-text pair data under a unified data formulation. By jointly training with an alternating scheme and adopting low-resolution input for image-text pairs, DetCLIPv2 exploits image-text pair data efficiently and effectively: DetCLIPv2 utilizes 13× more image-text pairs than DetCLIP with a similar training time and improves performance. With 13M image-text pairs for pre-training, DetCLIPv2 demonstrates superior open-vocabulary detection performance, e.g., DetCLIPv2 with Swin-T backbone achieves 40.4% zero-shot AP on the LVIS benchmark, which outperforms previous works GLIP/GLIPv2/DetCLIP by 14.4/11.4/4.5% AP, respectively, and even beats its fully-supervised counterpart by a large margin.

</details>

---

## 49. Probabilistic Prompt Learning for Dense Prediction

- [ ] Probabilistic Prompt Learning for Dense Prediction | https://cvpr.thecvf.com/virtual/2023/poster/22240

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22240

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in deterministic prompt learning has become a promising alternative to various downstream vision tasks, enabling models to learn powerful visual representations with the help of pre-trained vision-language models. However, this approach results in limited performance for dense prediction tasks that require handling more complex and diverse objects, since a single and deterministic description cannot sufficiently represent the entire image. In this paper, we present a novel probabilistic prompt learning to fully exploit the vision-language knowledge in dense prediction tasks. First, we introduce learnable class-agnostic attribute prompts to describe universal attributes across the object class. The attributes are combined with class information and visual-context knowledge to define the class-specific textual distribution. Text representations are sampled and used to guide the dense prediction task using the probabilistic pixel-text matching loss, enhancing the stability and generalization capability of the proposed method. Extensive experiments on different dense prediction tasks and ablation studies demonstrate the effectiveness of our proposed method.

</details>

---

## 50. Visual-Language Prompt Tuning With Knowledge-Guided Context Optimization

- [ ] Visual-Language Prompt Tuning With Knowledge-Guided Context Optimization | https://cvpr.thecvf.com/virtual/2023/poster/22242

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22242

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning is an effective way to adapt the pretrained visual-language model (VLM) to the downstream task using task-related textual tokens. Representative CoOp-based works combine the learnable textual tokens with the class tokens to obtain specific textual knowledge. However, the specific textual knowledge has worse generalizable to the unseen classes because it forgets the essential general textual knowledge having a strong generalization ability. To tackle this issue, we introduce a novel Knowledge-guided Context Optimization (KgCoOp) to enhance the generalization ability of the learnable prompt for unseen classes. To remember the essential general knowledge, KgCoOp constructs a regularization term to ensure that the essential general textual knowledge can be embedded into the special textual knowledge generated by the learnable prompt. Especially, KgCoOp minimizes the discrepancy between the textual embeddings generated by learned prompts and the hand-crafted prompts. Finally, adding the KgCoOp upon the contrastive loss can make a discriminative prompt for both seen and unseen tasks. Extensive evaluation of several benchmarks demonstrates that the proposed Knowledge-guided Context Optimization is an efficient method for prompt tuning, i.e., achieves better performance with less training time.

</details>

---

## 51. ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding

- [ ] ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding | https://cvpr.thecvf.com/virtual/2023/poster/22250

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22250

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The recognition capabilities of current state-of-the-art 3D models are limited by datasets with a small number of annotated data and a pre-defined set of categories. In its 2D counterpart, recent advances have shown that similar problems can be significantly alleviated by employing knowledge from other modalities, such as language. Inspired by this, leveraging multimodal information for 3D modality could be promising to improve 3D understanding under the restricted data regime, but this line of research is not well studied. Therefore, we introduce ULIP to learn a unified representation of images, language, and 3D point clouds by pre-training with object triplets from the three modalities. To overcome the shortage of training triplets, ULIP leverages a pre-trained vision-language model that has already learned a common visual and textual space by training with massive image-text pairs. Then, ULIP learns a 3D representation space aligned with the common image-text space, using a small number of automatically synthesized triplets. ULIP is agnostic to 3D backbone networks and can easily be integrated into any 3D architecture. Experiments show that ULIP effectively improves the performance of multiple recent 3D backbones by simply pre-training them on ShapeNet55 using our framework, achieving state-of-the-art performance in both standard 3D classification and zero-shot 3D classification on ModelNet40 and ScanObjectNN. ULIP also improves the performance of PointMLP by around 3% in 3D classification on ScanObjectNN, and outperforms PointCLIP by 28.8% on top-1 accuracy for zero-shot 3D classification on ModelNet40. Our code and pre-trained models are released at https://github.com/salesforce/ULIP.

</details>

---

## 52. Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training

- [ ] Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/22259

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22259

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models trained with contrastive learning on large-scale noisy data are becoming increasingly popular for zero-shot recognition problems. In this paper we improve the following three aspects of the contrastive pre-training pipeline: dataset noise, model initialization and the training objective. First, we propose a straightforward filtering strategy titled Complexity, Action, and Text-spotting (CAT) that significantly reduces dataset size, while achieving improved performance across zero-shot vision-language tasks. Next, we propose an approach titled Concept Distillation to leverage strong unimodal representations for contrastive training that does not increase training complexity while outperforming prior work. Finally, we modify the traditional contrastive alignment objective, and propose an importance-sampling approach to up-sample the importance of hard-negatives without adding additional complexity. On an extensive zero-shot benchmark of 29 tasks, our Distilled and Hard-negative Training (DiHT) approach improves on 20 tasks compared to the baseline. Furthermore, for few-shot linear probing, we propose a novel approach that bridges the gap between zero-shot and few-shot performance, substantially improving over prior work. Models are available at github.com/facebookresearch/diht.

</details>

---

## 53. Hierarchical Semantic Correspondence Networks for Video Paragraph Grounding

- [ ] Hierarchical Semantic Correspondence Networks for Video Paragraph Grounding | https://cvpr.thecvf.com/virtual/2023/poster/22271

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22271

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Video Paragraph Grounding (VPG) is an essential yet challenging task in vision-language understanding, which aims to jointly localize multiple events from an untrimmed video with a paragraph query description. One of the critical challenges in addressing this problem is to comprehend the complex semantic relations between visual and textual modalities. Previous methods focus on modeling the contextual information between the video and text from a single-level perspective (i.e., the sentence level), ignoring rich visual-textual correspondence relations at different semantic levels, e.g., the video-word and video-paragraph correspondence. To this end, we propose a novel Hierarchical Semantic Correspondence Network (HSCNet), which explores multi-level visual-textual correspondence by learning hierarchical semantic alignment and utilizes dense supervision by grounding diverse levels of queries. Specifically, we develop a hierarchical encoder that encodes the multi-modal inputs into semantics-aligned representations at different levels. To exploit the hierarchical semantic correspondence learned in the encoder for multi-level supervision, we further design a hierarchical decoder that progressively performs finer grounding for lower-level queries conditioned on higher-level semantics. Extensive experiments demonstrate the effectiveness of HSCNet and our method significantly outstrips the state-of-the-arts on two challenging benchmarks, i.e., ActivityNet-Captions and TACoS.

</details>

---

## 54. Learning To Detect and Segment for Open Vocabulary Object Detection

- [ ] Learning To Detect and Segment for Open Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2023/poster/22323

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22323

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Open vocabulary object detection has been greately advanced by the recent development of vision-language pre-trained model, which helps recognizing the novel objects with only semantic categories. The prior works mainly focus on knowledge transferring to the object proposal classification and employ class-agnostic box and mask prediction. In this work, we propose CondHead, a principled dynamic network design to better generalize the box regression and mask segmentation for open vocabulary setting. The core idea is to conditionally parametrize the network heads on semantic embedding and thus the model is guided with class-specific knowledge to better detect novel categories. Specifically, CondHead is composed of two streams of network heads, the dynamically aggregated heads and dynamically generated heads. The former is instantiated with a set of static heads that are conditionally aggregated, these heads are optimized as experts and are expected to learn sophisticated prediction. The Latter is instantiated with dynamically generated parameters and encodes general class-specific information. With such conditional design, the detection model is bridged by the semantic embedding to offer strongly generalizable class-wise box and mask prediction. Our method brings significant improvement to the prior state-of-the-art open vocabulary object detection methods with very minor overhead, e.g., it surpasses a RegionClip model by 3.0 detection AP on novel categories, with only 1.1% more computation.

</details>

---

## 55. Video-Text As Game Players: Hierarchical Banzhaf Interaction for Cross-Modal Representation Learning

- [ ] Video-Text As Game Players: Hierarchical Banzhaf Interaction for Cross-Modal Representation Learning | https://cvpr.thecvf.com/virtual/2023/poster/22354

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22354

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Contrastive learning-based video-language representation learning approaches, e.g., CLIP, have achieved outstanding performance, which pursue semantic interaction upon pre-defined video-text pairs. To clarify this coarse-grained global interaction and move a step further, we have to encounter challenging shell-breaking interactions for fine-grained cross-modal learning. In this paper, we creatively model video-text as game players with multivariate cooperative game theory to wisely handle the uncertainty during fine-grained semantic interaction with diverse granularity, flexible combination, and vague intensity. Concretely, we propose Hierarchical Banzhaf Interaction (HBI) to value possible correspondence between video frames and text words for sensitive and explainable cross-modal contrast. To efficiently realize the cooperative game of multiple video frames and multiple text words, the proposed method clusters the original video frames (text words) and computes the Banzhaf Interaction between the merged tokens. By stacking token merge modules, we achieve cooperative games at different semantic levels. Extensive experiments on commonly used text-video retrieval and video-question answering benchmarks with superior performances justify the efficacy of our HBI. More encouragingly, it can also serve as a visualization tool to promote the understanding of cross-modal interaction, which may have a far-reaching impact on the community. Project page is available at https://jpthu17.github.io/HBI/.

</details>

---

## 56. Position-Guided Text Prompt for Vision-Language Pre-Training

- [ ] Position-Guided Text Prompt for Vision-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/22379

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22379

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pre-Training (VLP) has shown promising capabilities to align image and text pairs, facilitating a broad variety of cross-modal learning tasks. However, we observe that VLP models often lack the visual grounding/localization capability which is critical for many downstream tasks such as visual reasoning. In this work, we propose a novel Position-guided Text Prompt (PTP) paradigm to enhance the visual grounding ability of cross-modal models trained with VLP. Specifically, in the VLP phase, PTP divides the image into NxN blocks, and identifies the objects in each block through the widely used object detector in VLP. It then reformulates the visual grounding task into a fill-in-the-blank problem given a PTP by encouraging the model to predict the objects in the given blocks or regress the blocks of a given object, e.g. filling “P” or “O” in a PTP “The block P has a O”. This mechanism improves the visual grounding capability of VLP models and thus helps them better handle various downstream tasks. By introducing PTP into several state-of-the-art VLP frameworks, we observe consistently significant improvements across representative cross-modal learning model architectures and several benchmarks, e.g. zero-shot Flickr30K Retrieval (+4.8 in average recall@1) for ViLT baseline, and COCO Captioning (+5.3 in CIDEr) for SOTA BLIP baseline. Moreover, PTP achieves comparable results with object-detector based methods, and much faster inference speed since PTP discards its object detector for inference while the later cannot. Our code and pre-trained weight will be released.

</details>

---

## 57. Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning With Multimodal Models

- [ ] Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning With Multimodal Models | https://cvpr.thecvf.com/virtual/2023/poster/22391

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22391

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The ability to quickly learn a new task with minimal instruction - known as few-shot learning - is a central aspect of intelligent agents. Classical few-shot benchmarks make use of few-shot samples from a single modality, but such samples may not be sufficient to characterize an entire concept class. In contrast, humans use cross-modal information to learn new concepts efficiently. In this work, we demonstrate that one can indeed build a better visual dog classifier by reading about dogs and listening to them bark. To do so, we exploit the fact that recent multimodal foundation models such as CLIP are inherently cross-modal, mapping different modalities to the same representation space. Specifically, we propose a simple cross-modal adaptation approach that learns from few-shot examples spanning different modalities. By repurposing class names as additional one-shot training samples, we achieve SOTA results with an embarrassingly simple linear classifier for vision-language adaptation. Furthermore, we show that our approach can benefit existing methods such as prefix tuning and classifier ensembling. Finally, to explore other modalities beyond vision and language, we construct the first (to our knowledge) audiovisual few-shot benchmark and use cross-modal training to improve the performance of both image and audio classification. We hope our success can inspire future works to embrace cross-modality for even broader domains and tasks.

</details>

---

## 58. Delving Into Shape-Aware Zero-Shot Semantic Segmentation

- [ ] Delving Into Shape-Aware Zero-Shot Semantic Segmentation | https://cvpr.thecvf.com/virtual/2023/poster/22399

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22399

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Thanks to the impressive progress of large-scale vision-language pretraining, recent recognition models can classify arbitrary objects in a zero-shot and open-set manner, with a surprisingly high accuracy. However, translating this success to semantic segmentation is not trivial, because this dense prediction task requires not only accurate semantic understanding but also fine shape delineation and existing vision-language models are trained with image-level language descriptions. To bridge this gap, we pursue shape-aware zero-shot semantic segmentation in this study. Inspired by classical spectral methods in the image segmentation literature, we propose to leverage the eigen vectors of Laplacian matrices constructed with self-supervised pixel-wise features to promote shape-awareness. Despite that this simple and effective technique does not make use of the masks of seen classes at all, we demonstrate that it out-performs a state-of-the-art shape-aware formulation that aligns ground truth and predicted edges during training. We also delve into the performance gains achieved on different datasets using different backbones and draw several interesting and conclusive observations: the benefits of promoting shape-awareness highly relates to mask compactness and language embedding locality. Finally, our method sets new state-of-the-art performance for zero-shot semantic segmentation on both Pascal and COCO, with significant margins. Code and models will be accessed at https://github.com/Liuxinyv/SAZS.

</details>

---

## 59. WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

- [ ] WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | https://cvpr.thecvf.com/virtual/2023/poster/22426

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22426

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Visual anomaly classification and segmentation are vital for automating industrial quality inspection. The focus of prior research in the field has been on training custom models for each quality inspection task, which requires task-specific images and annotation. In this paper we move away from this regime, addressing zero-shot and few-normal-shot anomaly classification and segmentation. Recently CLIP, a vision-language model, has shown revolutionary generality with competitive zero/few-shot performance in comparison to full-supervision. But CLIP falls short on anomaly classification and segmentation tasks. Hence, we propose window-based CLIP (WinCLIP) with (1) a compositional ensemble on state words and prompt templates and (2) efficient extraction and aggregation of window/patch/image-level features aligned with text. We also propose its few-normal-shot extension WinCLIP+, which uses complementary information from normal images. In MVTec-AD (and VisA), without further tuning, WinCLIP achieves 91.8%/85.1% (78.1%/79.6%) AUROC in zero-shot anomaly classification and segmentation while WinCLIP+ does 93.1%/95.2% (83.8%/96.4%) in 1-normal-shot, surpassing state-of-the-art by large margins.

</details>

---

## 60. CLIP the Gap: A Single Domain Generalization Approach for Object Detection

- [ ] CLIP the Gap: A Single Domain Generalization Approach for Object Detection | https://cvpr.thecvf.com/virtual/2023/poster/22474

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22474

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Single Domain Generalization (SDG) tackles the problem of training a model on a single source domain so that it generalizes to any unseen target domain. While this has been well studied for image classification, the literature on SDG object detection remains almost non-existent. To address the challenges of simultaneously learning robust object localization and representation, we propose to leverage a pre-trained vision-language model to introduce semantic domain concepts via textual prompts. We achieve this via a semantic augmentation strategy acting on the features extracted by the detector backbone, as well as a text-based classification loss. Our experiments evidence the benefits of our approach, outperforming by 10% the only existing SDG object detection method, Single-DGOD[49], on their own diverse weather-driving benchmark.

</details>

---

## 61. Collaborative Static and Dynamic Vision-Language Streams for Spatio-Temporal Video Grounding

- [ ] Collaborative Static and Dynamic Vision-Language Streams for Spatio-Temporal Video Grounding | https://cvpr.thecvf.com/virtual/2023/poster/22483

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22483

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Spatio-Temporal Video Grounding (STVG) aims to localize the target object spatially and temporally according to the given language query. It is a challenging task in which the model should well understand dynamic visual cues (e.g., motions) and static visual cues (e.g., object appearances) in the language description, which requires effective joint modeling of spatio-temporal visual-linguistic dependencies. In this work, we propose a novel framework in which a static vision-language stream and a dynamic vision-language stream are developed to collaboratively reason the target tube. The static stream performs cross-modal understanding in a single frame and learns to attend to the target object spatially according to intra-frame visual cues like object appearances. The dynamic stream models visual-linguistic dependencies across multiple consecutive frames to capture dynamic cues like motions. We further design a novel cross-stream collaborative block between the two streams, which enables the static and dynamic streams to transfer useful and complementary information from each other to achieve collaborative reasoning. Experimental results show the effectiveness of the collaboration of the two streams and our overall framework achieves new state-of-the-art performance on both HCSTVG and VidSTG datasets.

</details>

---

## 62. Texts as Images in Prompt Tuning for Multi-Label Image Recognition

- [ ] Texts as Images in Prompt Tuning for Multi-Label Image Recognition | https://cvpr.thecvf.com/virtual/2023/poster/22520

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22520

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning has been employed as an efficient way to adapt large vision-language pre-trained models (e.g. CLIP) to various downstream tasks in data-limited or label-limited settings. Nonetheless, visual data (e.g., images) is by default prerequisite for learning prompts in existing methods. In this work, we advocate that the effectiveness of image-text contrastive learning in aligning the two modalities (for training CLIP) further makes it feasible to treat texts as images for prompt tuning and introduce TaI prompting. In contrast to the visual data, text descriptions are easy to collect, and their class labels can be directly derived. Particularly, we apply TaI prompting to multi-label image recognition, where sentences in the wild serve as alternatives to images for prompt tuning. Moreover, with TaI, double-grained prompt tuning (TaI-DPT) is further presented to extract both coarse-grained and fine-grained embeddings for enhancing the multi-label recognition performance. Experimental results show that our proposed TaI-DPT outperforms zero-shot CLIP by a large margin on multiple benchmarks, e.g., MS-COCO, VOC2007, and NUS-WIDE, while it can be combined with existing methods of prompting from images to improve recognition performance further. The code is released at https://github.com/guozix/TaI-DPT.

</details>

---

## 63. OVTrack: Open-Vocabulary Multiple Object Tracking

- [ ] OVTrack: Open-Vocabulary Multiple Object Tracking | https://cvpr.thecvf.com/virtual/2023/poster/22536

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22536

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The ability to recognize, localize and track dynamic objects in a scene is fundamental to many real-world applications, such as self-driving and robotic systems. Yet, traditional multiple object tracking (MOT) benchmarks rely only on a few object categories that hardly represent the multitude of possible objects that are encountered in the real world. This leaves contemporary MOT methods limited to a small set of pre-defined object categories. In this paper, we address this limitation by tackling a novel task, open-vocabulary MOT, that aims to evaluate tracking beyond pre-defined training categories. We further develop OVTrack, an open-vocabulary tracker that is capable of tracking arbitrary object classes. Its design is based on two key ingredients: First, leveraging vision-language models for both classification and association via knowledge distillation; second, a data hallucination strategy for robust appearance feature learning from denoising diffusion probabilistic models. The result is an extremely data-efficient open-vocabulary tracker that sets a new state-of-the-art on the large-scale, large-vocabulary TAO benchmark, while being trained solely on static images. The project page is at https://www.vis.xyz/pub/ovtrack/.

</details>

---

## 64. Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-Commerce

- [ ] Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-Commerce | https://cvpr.thecvf.com/virtual/2023/poster/22582

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22582

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper aims to establish a generic multi-modal foundation model that has the scalable capability to massive downstream applications in E-commerce. Recently, large-scale vision-language pretraining approaches have achieved remarkable advances in the general domain. However, due to the significant differences between natural and product images, directly applying these frameworks for modeling image-level representations to E-commerce will be inevitably sub-optimal. To this end, we propose an instance-centric multi-modal pretraining paradigm called ECLIP in this work. In detail, we craft a decoder architecture that introduces a set of learnable instance queries to explicitly aggregate instance-level semantics. Moreover, to enable the model to focus on the desired product instance without reliance on expensive manual annotations, two specially configured pretext tasks are further proposed. Pretrained on the 100 million E-commerce-related data, ECLIP successfully extracts more generic, semantic-rich, and robust representations. Extensive experimental results show that, without further fine-tuning, ECLIP surpasses existing methods by a large margin on a broad range of downstream tasks, demonstrating the strong transferability to real-world E-commerce applications.

</details>

---

## 65. EC2: Emergent Communication for Embodied Control

- [ ] EC2: Emergent Communication for Embodied Control | https://cvpr.thecvf.com/virtual/2023/poster/22593

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22593

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Embodied control requires agents to leverage multi-modal pre-training to quickly learn how to act in new environments, where video demonstrations contain visual and motion details needed for low-level perception and control, and language instructions support generalization with abstract, symbolic structures. While recent approaches apply contrastive learning to force alignment between the two modalities, we hypothesize better modeling their complementary differences can lead to more holistic representations for downstream adaption. To this end, we propose Emergent Communication for Embodied Control (EC^2), a novel scheme to pre-train video-language representations for few-shot embodied control. The key idea is to learn an unsupervised “language” of videos via emergent communication, which bridges the semantics of video details and structures of natural language. We learn embodied representations of video trajectories, emergent language, and natural language using a language model, which is then used to finetune a lightweight policy network for downstream control. Through extensive experiments in Metaworld and Franka Kitchen embodied benchmarks, EC^2 is shown to consistently outperform previous contrastive learning methods for both videos and texts as task inputs. Further ablations confirm the importance of the emergent language, which is beneficial for both video and language learning, and significantly superior to using pre-trained video captions. We also present a quantitative and qualitative analysis of the emergent language and discuss future directions toward better understanding and leveraging emergent communication in embodied tasks.

</details>

---

## 66. Few-Shot Learning With Visual Distribution Calibration and Cross-Modal Distribution Alignment

- [ ] Few-Shot Learning With Visual Distribution Calibration and Cross-Modal Distribution Alignment | https://cvpr.thecvf.com/virtual/2023/poster/22631

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22631

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models have inspired much research on few-shot learning. However, with only a few training images, there exist two crucial problems: (1) the visual feature distributions are easily distracted by class-irrelevant information in images, and (2) the alignment between the visual and language feature distributions is difficult. To deal with the distraction problem, we propose a Selective Attack module, which consists of trainable adapters that generate spatial attention maps of images to guide the attacks on class-irrelevant image areas. By messing up these areas, the critical features are captured and the visual distributions of image features are calibrated. To better align the visual and language feature distributions that describe the same object class, we propose a cross-modal distribution alignment module, in which we introduce a vision-language prototype for each class to align the distributions, and adopt the Earth Mover’s Distance (EMD) to optimize the prototypes. For efficient computation, the upper bound of EMD is derived. In addition, we propose an augmentation strategy to increase the diversity of the images and the text prompts, which can reduce overfitting to the few-shot training images. Extensive experiments on 11 datasets demonstrate that our method consistently outperforms prior arts in few-shot learning.

</details>

---

## 67. Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection

- [ ] Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2023/poster/22676

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22676

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary object detection aims to provide object detectors trained on a fixed set of object categories with the generalizability to detect objects described by arbitrary text queries. Previous methods adopt knowledge distillation to extract knowledge from Pretrained Vision-and-Language Models (PVLMs) and transfer it to detectors. However, due to the non-adaptive proposal cropping and single-level feature mimicking processes, they suffer from information destruction during knowledge extraction and inefficient knowledge transfer. To remedy these limitations, we propose an Object-Aware Distillation Pyramid (OADP) framework, including an Object-Aware Knowledge Extraction (OAKE) module and a Distillation Pyramid (DP) mechanism. When extracting object knowledge from PVLMs, the former adaptively transforms object proposals and adopts object-aware mask attention to obtain precise and complete knowledge of objects. The latter introduces global and block distillation for more comprehensive knowledge transfer to compensate for the missing relation information in object distillation. Extensive experiments show that our method achieves significant improvement compared to current methods. Especially on the MS-COCO dataset, our OADP framework reaches 35.6 mAP^N 50, surpassing the current state-of-the-art method by 3.3 mAP^N 50. Code is anonymously provided in the supplementary materials.

</details>

---

## 68. Adaptive Zone-Aware Hierarchical Planner for Vision-Language Navigation

- [ ] Adaptive Zone-Aware Hierarchical Planner for Vision-Language Navigation | https://cvpr.thecvf.com/virtual/2023/poster/22679

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22679

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The task of Vision-Language Navigation (VLN) is for an embodied agent to reach the global goal according to the instruction. Essentially, during navigation, a series of sub-goals need to be adaptively set and achieved, which is naturally a hierarchical navigation process. However, previous methods leverage a single-step planning scheme, i.e., directly performing navigation action at each step, which is unsuitable for such a hierarchical navigation process. In this paper, we propose an Adaptive Zone-aware Hierarchical Planner (AZHP) to explicitly divides the navigation process into two heterogeneous phases, i.e., sub-goal setting via zone partition/selection (high-level action) and sub-goal executing (low-level action), for hierarchical planning. Specifically, AZHP asynchronously performs two levels of action via the designed State-Switcher Module (SSM). For high-level action, we devise a Scene-aware adaptive Zone Partition (SZP) method to adaptively divide the whole navigation area into different zones on-the-fly. Then the Goal-oriented Zone Selection (GZS) method is proposed to select a proper zone for the current sub-goal. For low-level action, the agent conducts navigation-decision multi-steps in the selected zone. Moreover, we design a Hierarchical RL (HRL) strategy and auxiliary losses with curriculum learning to train the AZHP framework, which provides effective supervision signals for each stage. Extensive experiments demonstrate the superiority of our proposed method, which achieves state-of-the-art performance on three VLN benchmarks (REVERIE, SOON, R2R).

</details>

---

## 69. Task Residual for Tuning Vision-Language Models

- [ ] Task Residual for Tuning Vision-Language Models | https://cvpr.thecvf.com/virtual/2023/poster/22699

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22699

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models (VLMs) pre-trained on billion-level data have learned general visual representations and broad visual concepts. In principle, the well-learned knowledge structure of the VLMs should be inherited appropriately when being transferred to downstream tasks with limited data. However, most existing efficient transfer learning (ETL) approaches for VLMs either damage or are excessively biased towards the prior knowledge, e.g., prompt tuning (PT) discards the pre-trained text-based classifier and builds a new one while adapter-style tuning (AT) fully relies on the pre-trained features. To address this, we propose a new efficient tuning approach for VLMs named Task Residual Tuning (TaskRes), which performs directly on the text-based classifier and explicitly decouples the prior knowledge of the pre-trained models and new knowledge regarding a target task. Specifically, TaskRes keeps the original classifier weights from the VLMs frozen and obtains a new classifier for the target task by tuning a set of prior-independent parameters as a residual to the original one, which enables reliable prior knowledge preservation and flexible task-specific knowledge exploration. The proposed TaskRes is simple yet effective, which significantly outperforms previous ETL methods (e.g., PT and AT) on 11 benchmark datasets while requiring minimal effort for the implementation. Our code is available at https://github.com/geekyutao/TaskRes.

</details>

---

## 70. Distilling Vision-Language Pre-Training To Collaborate With Weakly-Supervised Temporal Action Localization

- [ ] Distilling Vision-Language Pre-Training To Collaborate With Weakly-Supervised Temporal Action Localization | https://cvpr.thecvf.com/virtual/2023/poster/22717

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22717

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Weakly-supervised temporal action localization (WTAL) learns to detect and classify action instances with only category labels. Most methods widely adopt the off-the-shelf Classification-Based Pre-training (CBP) to generate video features for action localization. However, the different optimization objectives between classification and localization, make temporally localized results suffer from the serious incomplete issue. To tackle this issue without additional annotations, this paper considers to distill free action knowledge from Vision-Language Pre-training (VLP), as we surprisingly observe that the localization results of vanilla VLP have an over-complete issue, which is just complementary to the CBP results. To fuse such complementarity, we propose a novel distillation-collaboration framework with two branches acting as CBP and VLP respectively. The framework is optimized through a dual-branch alternate training strategy. Specifically, during the B step, we distill the confident background pseudo-labels from the CBP branch; while during the F step, the confident foreground pseudo-labels are distilled from the VLP branch. As a result, the dual-branch complementarity is effectively fused to promote one strong alliance. Extensive experiments and ablation studies on THUMOS14 and ActivityNet1.2 reveal that our method significantly outperforms state-of-the-art methods.

</details>

---

## 71. DeltaEdit: Exploring Text-Free Training for Text-Driven Image Manipulation

- [ ] DeltaEdit: Exploring Text-Free Training for Text-Driven Image Manipulation | https://cvpr.thecvf.com/virtual/2023/poster/22724

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22724

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Text-driven image manipulation remains challenging in training or inference flexibility. Conditional generative models depend heavily on expensive annotated training data. Meanwhile, recent frameworks, which leverage pre-trained vision-language models, are limited by either per text-prompt optimization or inference-time hyper-parameters tuning. In this work, we propose a novel framework named DeltaEdit to address these problems. Our key idea is to investigate and identify a space, namely delta image and text space that has well-aligned distribution between CLIP visual feature differences of two images and CLIP textual embedding differences of source and target texts. Based on the CLIP delta space, the DeltaEdit network is designed to map the CLIP visual features differences to the editing directions of StyleGAN at training phase. Then, in inference phase, DeltaEdit predicts the StyleGAN’s editing directions from the differences of the CLIP textual features. In this way, DeltaEdit is trained in a text-free manner. Once trained, it can well generalize to various text prompts for zero-shot inference without bells and whistles. Extensive experiments verify that our method achieves competitive performances with other state-of-the-arts, meanwhile with much better flexibility in both training and inference. Code is available at https://github.com/Yueming6568/DeltaEdit

</details>

---

## 72. Clover: Towards a Unified Video-Language Alignment and Fusion Model

- [ ] Clover: Towards a Unified Video-Language Alignment and Fusion Model | https://cvpr.thecvf.com/virtual/2023/poster/22766

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22766

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Building a universal video-language model for solving various video understanding tasks (e.g., text-video retrieval, video question answering) is an open challenge to the machine learning field. Towards this goal, most recent works build the model by stacking uni-modal and cross-modal feature encoders and train it with pair-wise contrastive pre-text tasks. Though offering attractive generality, the resulted models have to compromise between efficiency and performance. They mostly adopt different architectures to deal with different downstream tasks. We find this is because the pair-wise training cannot well align and fuse features from different modalities. We then introduce Clover--a Correlated Video-Language pre-training method--towards a universal video-language model for solving multiple video understanding tasks with neither performance nor efficiency compromise. It improves cross-modal feature alignment and fusion via a novel tri-modal alignment pre-training task. Additionally, we propose to enhance the tri-modal alignment via incorporating learning from semantic masked samples and a new pair-wise ranking loss. Clover establishes new state-of-the-arts on multiple downstream tasks, including three retrieval tasks for both zero-shot and fine-tuning settings, and eight video question answering tasks. Codes and pre-trained models will be released at https://github.com/LeeYN-43/Clover.

</details>

---

## 73. An Empirical Study of End-to-End Video-Language Transformers With Masked Visual Modeling

- [ ] An Empirical Study of End-to-End Video-Language Transformers With Masked Visual Modeling | https://cvpr.thecvf.com/virtual/2023/poster/22798

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22798

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Masked visual modeling (MVM) has been recently proven effective for visual pre-training. While similar reconstructive objectives on video inputs (e.g., masked frame modeling) have been explored in video-language (VidL) pre-training, previous studies fail to find a truly effective MVM strategy that can largely benefit the downstream performance. In this work, we systematically examine the potential of MVM in the context of VidL learning. Specifically, we base our study on a fully end-to-end VIdeO-LanguagE Transformer (VIOLET), where the supervision from MVM training can be backpropagated to the video pixel space. In total, eight different reconstructive targets of MVM are explored, from low-level pixel values and oriented gradients to high-level depth maps, optical flow, discrete visual tokens, and latent visual features. We conduct comprehensive experiments and provide insights into the factors leading to effective MVM training, resulting in an enhanced model VIOLETv2. Empirically, we show VIOLETv2 pre-trained with MVM objective achieves notable improvements on 13 VidL benchmarks, ranging from video question answering, video captioning, to text-to-video retrieval.

</details>

---

## 74. LAVENDER: Unifying Video-Language Understanding As Masked Language Modeling

- [ ] LAVENDER: Unifying Video-Language Understanding As Masked Language Modeling | https://cvpr.thecvf.com/virtual/2023/poster/22799

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22799

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Unified vision-language frameworks have greatly advanced in recent years, most of which adopt an encoder-decoder architecture to unify image-text tasks as sequence-to-sequence generation. However, existing video-language (VidL) models still require task-specific designs in model architecture and training objectives for each task. In this work, we explore a unified VidL framework LAVENDER, where Masked Language Modeling (MLM) is used as the common interface for all pre-training and downstream tasks. Such unification leads to a simplified model architecture, where only a lightweight MLM head, instead of a decoder with much more parameters, is needed on top of the multimodal encoder. Surprisingly, experimental results show that this unified framework achieves competitive performance on 14 VidL benchmarks, covering video question answering, text-to-video retrieval and video captioning. Extensive analyses further demonstrate LAVENDER can (i) seamlessly support all downstream tasks with just a single set of parameter values when multi-task finetuned; (ii) generalize to various downstream tasks with limited training samples; and (iii) enable zero-shot evaluation on video question answering tasks.

</details>

---

## 75. GIVL: Improving Geographical Inclusivity of Vision-Language Models With Pre-Training Methods

- [ ] GIVL: Improving Geographical Inclusivity of Vision-Language Models With Pre-Training Methods | https://cvpr.thecvf.com/virtual/2023/poster/22811

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22811

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

A key goal for the advancement of AI is to develop technologies that serve the needs not just of one group but of all communities regardless of their geographical region. In fact, a significant proportion of knowledge is locally shared by people from certain regions but may not apply equally in other regions because of cultural differences. If a model is unaware of regional characteristics, it may lead to performance disparity across regions and result in bias against underrepresented groups. We propose GIVL, a Geographically Inclusive Vision-and-Language Pre-trained model. There are two attributes of geo-diverse visual concepts which can help to learn geo-diverse knowledge: 1) concepts under similar categories have unique knowledge and visual characteristics, 2) concepts with similar visual features may fall in completely different categories. Motivated by the attributes, we design new pre-training objectives Image-Knowledge Matching (IKM) and Image Edit Checking (IEC) to pre-train GIVL. Compared with similar-size models pre-trained with similar scale of data, GIVL achieves state-of-the-art (SOTA) and more balanced performance on geo-diverse V&L tasks. Code and data are released at https://github.com/WadeYin9712/GIVL.

</details>

---

## 76. VILA: Learning Image Aesthetics From User Comments With Vision-Language Pretraining

- [ ] VILA: Learning Image Aesthetics From User Comments With Vision-Language Pretraining | https://cvpr.thecvf.com/virtual/2023/poster/22829

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22829

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Assessing the aesthetics of an image is challenging, as it is influenced by multiple factors including composition, color, style, and high-level semantics. Existing image aesthetic assessment (IAA) methods primarily rely on human-labeled rating scores, which oversimplify the visual aesthetic information that humans perceive. Conversely, user comments offer more comprehensive information and are a more natural way to express human opinions and preferences regarding image aesthetics. In light of this, we propose learning image aesthetics from user comments, and exploring vision-language pretraining methods to learn multimodal aesthetic representations. Specifically, we pretrain an image-text encoder-decoder model with image-comment pairs, using contrastive and generative objectives to learn rich and generic aesthetic semantics without human labels. To efficiently adapt the pretrained model for downstream IAA tasks, we further propose a lightweight rank-based adapter that employs text as an anchor to learn the aesthetic ranking concept. Our results show that our pretrained aesthetic vision-language model outperforms prior works on image aesthetic captioning over the AVA-Captions dataset, and it has powerful zero-shot capability for aesthetic tasks such as zero-shot style classification and zero-shot IAA, surpassing many supervised baselines. With only minimal finetuning parameters using the proposed adapter module, our model achieves state-of-the-art IAA performance over the AVA dataset.

</details>

---

## 77. Context-Aware Alignment and Mutual Masking for 3D-Language Pre-Training

- [ ] Context-Aware Alignment and Mutual Masking for 3D-Language Pre-Training | https://cvpr.thecvf.com/virtual/2023/poster/22858

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22858

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

3D visual language reasoning plays an important role in effective human-computer interaction. The current approaches for 3D visual reasoning are task-specific, and lack pre-training methods to learn generic representations that can transfer across various tasks. Despite the encouraging progress in vision-language pre-training for image-text data, 3D-language pre-training is still an open issue due to limited 3D-language paired data, highly sparse and irregular structure of point clouds and ambiguities in spatial relations of 3D objects with viewpoint changes. In this paper, we present a generic 3D-language pre-training approach, that tackles multiple facets of 3D-language reasoning by learning universal representations. Our learning objective constitutes two main parts. 1) Context aware spatial-semantic alignment to establish fine-grained correspondence between point clouds and texts. It reduces relational ambiguities by aligning 3D spatial relationships with textual semantic context. 2) Mutual 3D-Language Masked modeling to enable cross-modality information exchange. Instead of reconstructing sparse 3D points for which language can hardly provide cues, we propose masked proposal reasoning to learn semantic class and mask-invariant representations. Our proposed 3D-language pre-training method achieves promising results once adapted to various downstream tasks, including 3D visual grounding, 3D dense captioning and 3D question answering. Our codes are available at https://github.com/leolyj/3D-VLP

</details>

---

## 78. Mobile User Interface Element Detection via Adaptively Prompt Tuning

- [ ] Mobile User Interface Element Detection via Adaptively Prompt Tuning | https://cvpr.thecvf.com/virtual/2023/poster/22879

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22879

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent object detection approaches rely on pretrained vision-language models for image-text alignment. However, they fail to detect the Mobile User Interface (MUI) element since it contains additional OCR information, which describes its content and function but is often ignored. In this paper, we develop a new MUI element detection dataset named MUI-zh and propose an Adaptively Prompt Tuning (APT) module to take advantage of discriminating OCR information. APT is a lightweight and effective module to jointly optimize category prompts across different modalities. For every element, APT uniformly encodes its visual features and OCR descriptions to dynamically adjust the representation of frozen category prompts. We evaluate the effectiveness of our plug-and-play APT upon several existing CLIP-based detectors for both standard and open-vocabulary MUI element detection. Extensive experiments show that our method achieves considerable improvements on two datasets. The datasets is available at github.com/antmachineintelligence/MUI-zh.

</details>

---

## 79. Learning Video Representations From Large Language Models

- [ ] Learning Video Representations From Large Language Models | https://cvpr.thecvf.com/virtual/2023/poster/22945

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22945

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We introduce LAVILA, a new approach to learning video-language representations by leveraging Large Language Models (LLMs). We repurpose pre-trained LLMs to be conditioned on visual input, and finetune them to create automatic video narrators. Our auto-generated narrations offer a number of advantages, including dense coverage of long videos, better temporal synchronization of the visual information and text, and much higher diversity of text. The video-language embedding learned contrastively with these narrations outperforms the previous state-of-the-art on multiple first-person and third-person video tasks, both in zero-shot and finetuned setups. Most notably, LAVILA obtains an absolute gain of 10.1% on EGTEA classification and 5.9% Epic-Kitchens-100 multi-instance retrieval benchmarks. Furthermore, LAVILA trained with only half the narrations from the Ego4D dataset outperforms models trained on the full set, and shows positive scaling behavior on increasing pre-training data and model size.

</details>

---

## 80. PLA: Language-Driven Open-Vocabulary 3D Scene Understanding

- [ ] PLA: Language-Driven Open-Vocabulary 3D Scene Understanding | https://cvpr.thecvf.com/virtual/2023/poster/22986

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/22986

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary scene understanding aims to localize and recognize unseen categories beyond the annotated label space. The recent breakthrough of 2D open-vocabulary perception is largely driven by Internet-scale paired image-text data with rich vocabulary concepts. However, this success cannot be directly transferred to 3D scenarios due to the inaccessibility of large-scale 3D-text pairs. To this end, we propose to distill knowledge encoded in pre-trained vision-language (VL) foundation models through captioning multi-view images from 3D, which allows explicitly associating 3D and semantic-rich captions. Further, to foster coarse-to-fine visual-semantic representation learning from captions, we design hierarchical 3D-caption pairs, leveraging geometric constraints between 3D scenes and multi-view images. Finally, by employing contrastive learning, the model learns language-aware embeddings that connect 3D and text for open-vocabulary tasks. Our method not only remarkably outperforms baseline methods by 25.8% ~ 44.7% hIoU and 14.5% ~ 50.4% hAP_{50} in open-vocabulary semantic and instance segmentation, but also shows robust transferability on challenging zero-shot domain transfer tasks. See the project website at https://dingry.github.io/projects/PLA.

</details>

---

## 81. Scaling Language-Image Pre-Training via Masking

- [ ] Scaling Language-Image Pre-Training via Masking | https://cvpr.thecvf.com/virtual/2023/poster/23014

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23014

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present Fast Language-Image Pre-training (FLIP), a simple and more efficient method for training CLIP. Our method randomly masks out and removes a large portion of image patches during training. Masking allows us to learn from more image-text pairs given the same wall-clock time and contrast more samples per iteration with similar memory footprint. It leads to a favorable trade-off between accuracy and training time. In our experiments on 400 million image-text pairs, FLIP improves both accuracy and speed over the no-masking baseline. On a large diversity of downstream tasks, FLIP dominantly outperforms the CLIP counterparts trained on the same data. Facilitated by the speedup, we explore the scaling behavior of increasing the model size, data size, or training length, and report encouraging results and comparisons. We hope that our work will foster future research on scaling vision-language learning.

</details>

---

## 82. Aligning Bag of Regions for Open-Vocabulary Object Detection

- [ ] Aligning Bag of Regions for Open-Vocabulary Object Detection | https://cvpr.thecvf.com/virtual/2023/poster/23036

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23036

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) learn to align vision and language representations on large-scale datasets, where each image-text pair usually contains a bag of semantic concepts. However, existing open-vocabulary object detectors only align region embeddings individually with the corresponding features extracted from the VLMs. Such a design leaves the compositional structure of semantic concepts in a scene under-exploited, although the structure may be implicitly learned by the VLMs. In this work, we propose to align the embedding of bag of regions beyond individual regions. The proposed method groups contextually interrelated regions as a bag. The embeddings of regions in a bag are treated as embeddings of words in a sentence, and they are sent to the text encoder of a VLM to obtain the bag-of-regions embedding, which is learned to be aligned to the corresponding features extracted by a frozen VLM. Applied to the commonly used Faster R-CNN, our approach surpasses the previous best results by 4.6 box AP 50 and 2.8 mask AP on novel categories of open-vocabulary COCO and LVIS benchmarks, respectively. Code and models are available at https://github.com/wusize/ovdet.

</details>

---

## 83. KD-DLGAN: Data Limited Image Generation via Knowledge Distillation

- [ ] KD-DLGAN: Data Limited Image Generation via Knowledge Distillation | https://cvpr.thecvf.com/virtual/2023/poster/23061

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23061

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Generative Adversarial Networks (GANs) rely heavily on large-scale training data for training high-quality image generation models. With limited training data, the GAN discriminator often suffers from severe overfitting which directly leads to degraded generation especially in generation diversity. Inspired by the recent advances in knowledge distillation (KD), we propose KD-GAN, a knowledge-distillation based generation framework that introduces pre-trained vision-language models for training effective data-limited image generation models. KD-GAN consists of two innovative designs. The first is aggregated generative KD that mitigates the discriminator overfitting by challenging the discriminator with harder learning tasks and distilling more generalizable knowledge from the pre-trained models. The second is correlated generative KD that improves the generation diversity by distilling and preserving the diverse image-text correlation within the pre-trained models. Extensive experiments over multiple benchmarks show that KD-GAN achieves superior image generation with limited training data. In addition, KD-GAN complements the state-of-the-art with consistent and substantial performance gains. Note that codes will be released.

</details>

---

## 84. Accelerating Vision-Language Pretraining With Free Language Modeling

- [ ] Accelerating Vision-Language Pretraining With Free Language Modeling | https://cvpr.thecvf.com/virtual/2023/poster/23085

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23085

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

The state of the arts in vision-language pretraining (VLP) achieves exemplary performance but suffers from high training costs resulting from slow convergence and long training time, especially on large-scale web datasets. An essential obstacle to training efficiency lies in the entangled prediction rate (percentage of tokens for reconstruction) and corruption rate (percentage of corrupted tokens) in masked language modeling (MLM), that is, a proper corruption rate is achieved at the cost of a large portion of output tokens being excluded from prediction loss. To accelerate the convergence of VLP, we propose a new pretraining task, namely, free language modeling (FLM), that enables a 100% prediction rate with arbitrary corruption rates. FLM successfully frees the prediction rate from the tie-up with the corruption rate while allowing the corruption spans to be customized for each token to be predicted. FLM-trained models are encouraged to learn better and faster given the same GPU time by exploiting bidirectional contexts more flexibly. Extensive experiments show FLM could achieve an impressive 2.5x pretraining time reduction in comparison to the MLM-based methods, while keeping competitive performance on both vision-language understanding and generation tasks.

</details>

---

## 85. Generalized Decoding for Pixel, Image, and Language

- [ ] Generalized Decoding for Pixel, Image, and Language | https://cvpr.thecvf.com/virtual/2023/poster/23105

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23105

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present X-Decoder, a generalized decoding model that can predict pixel-level segmentation and language tokens seamlessly. X-Decoder takes as input two types of queries: (i) generic non-semantic queries and (ii) semantic queries induced from text inputs, to decode different pixel-level and token-level outputs in the same semantic space. With such a novel design, X-Decoder is the first work that provides a unified way to support all types of image segmentation and a variety of vision-language (VL) tasks. Further, our design enables seamless interactions across tasks at different granularities and brings mutual benefits by learning a common and rich pixel-level visual-semantic understanding space, without any pseudo-labeling. After pretraining on a mixed set of a limited amount of segmentation data and millions of image-text pairs, X-Decoder exhibits strong transferability to a wide range of downstream tasks in both zero-shot and finetuning settings. Notably, it achieves (1) state-of-the-art results on open-vocabulary segmentation and referring segmentation on eight datasets; (2) better or competitive finetuned performance to other generalist and specialist models on segmentation and VL tasks; and (3) flexibility for efficient finetuning and novel task composition. Code, demo, video and visualization are available at: https://x-decoder-vl.github.io.

</details>

---

## 86. ImageBind: One Embedding Space To Bind Them All

- [ ] ImageBind: One Embedding Space To Bind Them All | https://cvpr.thecvf.com/virtual/2023/poster/23114

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23114

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

We present ImageBind, an approach to learn a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. We show that all combinations of paired data are not necessary to train such a joint embedding, and only image-paired data is sufficient to bind the modalities together. ImageBind can leverage recent large scale vision-language models, and extends their zero-shot capabilities to new modalities just by using their natural pairing with images. It enables novel emergent applications ‘out-of-the-box’ including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation. The emergent capabilities improve with the strength of the image encoder and we set a new state-of-the-art on emergent zero-shot recognition tasks across modalities, outperforming specialist supervised models. Finally, we show strong few-shot recognition results outperforming prior work, and that ImageBind serves as a new way to evaluate vision models for visual and non-visual tasks.

</details>

---

## 87. CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model

- [ ] CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model | https://cvpr.thecvf.com/virtual/2023/poster/23133

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23133

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Supervised crowd counting relies heavily on costly manual labeling, which is difficult and expensive, especially in dense scenes. To alleviate the problem, we propose a novel unsupervised framework for crowd counting, named CrowdCLIP. The core idea is built on two observations: 1) the recent contrastive pre-trained vision-language model (CLIP) has presented impressive performance on various downstream tasks; 2) there is a natural mapping between crowd patches and count text. To the best of our knowledge, CrowdCLIP is the first to investigate the vision-language knowledge to solve the counting problem. Specifically, in the training stage, we exploit the multi-modal ranking loss by constructing ranking text prompts to match the size-sorted crowd patches to guide the image encoder learning. In the testing stage, to deal with the diversity of image patches, we propose a simple yet effective progressive filtering strategy to first select the highly potential crowd patches and then map them into the language space with various counting intervals. Extensive experiments on five challenging datasets demonstrate that the proposed CrowdCLIP achieves superior performance compared to previous unsupervised state-of-the-art counting methods. Notably, CrowdCLIP even surpasses some popular fully-supervised methods under the cross-dataset setting. The source code will be available at https://github.com/dk-liang/CrowdCLIP.

</details>

---

## 88. FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks

- [ ] FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks | https://cvpr.thecvf.com/virtual/2023/poster/23158

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23158

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In the fashion domain, there exists a variety of vision-and-language (V+L) tasks, including cross-modal retrieval, text-guided image retrieval, multi-modal classification, and image captioning. They differ drastically in each individual input/output format and dataset size. It has been common to design a task-specific model and fine-tune it independently from a pre-trained V+L model (e.g., CLIP). This results in parameter inefficiency and inability to exploit inter-task relatedness. To address such issues, we propose a novel FAshion-focused Multi-task Efficient learning method for Vision-and-Language tasks (FAME-ViL) in this work. Compared with existing approaches, FAME-ViL applies a single model for multiple heterogeneous fashion tasks, therefore being much more parameter-efficient. It is enabled by two novel components: (1) a task-versatile architecture with cross-attention adapters and task-specific adapters integrated into a unified V+L model, and (2) a stable and effective multi-task training strategy that supports learning from heterogeneous data and prevents negative transfer. Extensive experiments on four fashion tasks show that our FAME-ViL can save 61.5% of parameters over alternatives, while significantly outperforming the conventional independently trained single-task models. Code is available at https://github.com/BrandonHanx/FAME-ViL.

</details>

---

## 89. MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining

- [ ] MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining | https://cvpr.thecvf.com/virtual/2023/poster/23245

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23245

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a simple yet effective framework MaskCLIP, which incorporates a newly proposed masked self-distillation into contrastive language-image pretraining. The core idea of masked self-distillation is to distill representation from a full image to the representation predicted from a masked image. Such incorporation enjoys two vital benefits. First, masked self-distillation targets local patch representation learning, which is complementary to vision-language contrastive focusing on text-related representation. Second, masked self-distillation is also consistent with vision-language contrastive from the perspective of training objective as both utilize the visual encoder for feature aligning, and thus is able to learn local semantics getting indirect supervision from the language. We provide specially designed experiments with a comprehensive analysis to validate the two benefits. Symmetrically, we also introduce the local semantic supervision into the text branch, which further improves the pretraining performance. With extensive experiments, we show that MaskCLIP, when applied to various challenging downstream tasks, achieves superior results in linear probing, finetuning, and zero-shot performance with the guidance of the language encoder. We will release the code and data after the publication.

</details>

---

## 90. Top-Down Visual Attention From Analysis by Synthesis

- [ ] Top-Down Visual Attention From Analysis by Synthesis | https://cvpr.thecvf.com/virtual/2023/poster/23262

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23262

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Current attention algorithms (e.g., self-attention) are stimulus-driven and highlight all the salient objects in an image. However, intelligent agents like humans often guide their attention based on the high-level task at hand, focusing only on task-related objects. This ability of task-guided top-down attention provides task-adaptive representation and helps the model generalize to various tasks. In this paper, we consider top-down attention from a classic Analysis-by-Synthesis (AbS) perspective of vision. Prior work indicates a functional equivalence between visual attention and sparse reconstruction; we show that an AbS visual system that optimizes a similar sparse reconstruction objective modulated by a goal-directed top-down signal naturally simulates top-down attention. We further propose Analysis-by-Synthesis Vision Transformer (AbSViT), which is a top-down modulated ViT model that variationally approximates AbS, and achieves controllable top-down attention. For real-world applications, AbSViT consistently improves over baselines on Vision-Language tasks such as VQA and zero-shot retrieval where language guides the top-down attention. AbSViT can also serve as a general backbone, improving performance on classification, semantic segmentation, and model robustness. Project page: https://sites.google.com/view/absvit.

</details>

---

## 91. Towards Universal Fake Image Detectors That Generalize Across Generative Models

- [ ] Towards Universal Fake Image Detectors That Generalize Across Generative Models | https://cvpr.thecvf.com/virtual/2023/poster/23276

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23276

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

With generative models proliferating at a rapid rate, there is a growing need for general purpose fake image detectors. In this work, we first show that the existing paradigm, which consists of training a deep network for real-vs-fake classification, fails to detect fake images from newer breeds of generative models when trained to detect GAN fake images. Upon analysis, we find that the resulting classifier is asymmetrically tuned to detect patterns that make an image fake. The real class becomes a ‘sink’ class holding anything that is not fake, including generated images from models not accessible during training. Building upon this discovery, we propose to perform real-vs-fake classification without learning; i.e., using a feature space not explicitly trained to distinguish real from fake images. We use nearest neighbor and linear probing as instantiations of this idea. When given access to the feature space of a large pretrained vision-language model, the very simple baseline of nearest neighbor classification has surprisingly good generalization ability in detecting fake images from a wide variety of generative models; e.g., it improves upon the SoTA by +15.07 mAP and +25.90% acc when tested on unseen diffusion and autoregressive models.

</details>

---

## 92. Mask-Free OVIS: Open-Vocabulary Instance Segmentation Without Manual Mask Annotations

- [ ] Mask-Free OVIS: Open-Vocabulary Instance Segmentation Without Manual Mask Annotations | https://cvpr.thecvf.com/virtual/2023/poster/23294

- **Link**: https://cvpr.thecvf.com/virtual/2023/poster/23294

- **Conference**: CVPR

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Existing instance segmentation models learn task-specific information using manual mask annotations from base (training) categories. These mask annotations require tremendous human effort, limiting the scalability to annotate novel (new) categories. To alleviate this problem, Open-Vocabulary (OV) methods leverage large-scale image-caption pairs and vision-language models to learn novel categories. In summary, an OV method learns task-specific information using strong supervision from base annotations and novel category information using weak supervision from image-captions pairs. This difference between strong and weak supervision leads to overfitting on base categories, resulting in poor generalization towards novel categories. In this work, we overcome this issue by learning both base and novel categories from pseudo-mask annotations generated by the vision-language model in a weakly supervised manner using our proposed Mask-free OVIS pipeline. Our method automatically generates pseudo-mask annotations by leveraging the localization ability of a pre-trained vision-language model for objects present in image-caption pairs. The generated pseudo-mask annotations are then used to supervise an instance segmentation model, freeing the entire pipeline from any labour-expensive instance-level annotations and overfitting. Our extensive experiments show that our method trained with just pseudo-masks significantly improves the mAP scores on the MS-COCO dataset and OpenImages dataset compared to the recent state-of-the-art methods trained with manual masks. Codes and models are provided in https://vibashan.github.io/ovis-web/.

</details>

---

