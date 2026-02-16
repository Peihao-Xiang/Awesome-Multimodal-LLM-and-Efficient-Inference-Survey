# ICCV 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_iccv2025_papers.csv

## 1. PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes

- [ ] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes | https://openaccess.thecvf.com/content/ICCV2025/html/Abdelreheem_PlaceIt3D_Language-Guided_Object_Placement_in_Real_3D_Scenes_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Abdelreheem_PlaceIt3D_Language-Guided_Object_Placement_in_Real_3D_Scenes_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce the task of Language-Guided Object Placement in Real 3D Scenes. Given a 3D reconstructed point-cloud scene, a 3D asset, and a natural-language instruction, the goal is to place the asset so that the instruction is satisfied. The task demands tackling four intertwined challenges: (a) one-to-many ambiguity in valid placements; (b) precise geometric and physical reasoning; (c) joint understanding across the scene, the asset, and language; and (d) robustness to noisy point clouds with no privileged metadata at test time. The first three challenges mirror the complexities of synthetic scene generation, while the metadata-free, noisy-scan scenario is inherited from language-guided 3D visual grounding. We inaugurate this task by introducing a benchmark and evaluation protocol, releasing a dataset for training multi-modal large language models (MLLMs), and establishing a first nontrivial baseline. We believe this challenging setup and benchmark will provide a foundation for evaluating and advancing MLLMs in 3D understanding.

</details>

---

## 2. Kestrel: 3D Multimodal LLM for Part-Aware Grounded Description

- [ ] Kestrel: 3D Multimodal LLM for Part-Aware Grounded Description | https://openaccess.thecvf.com/content/ICCV2025/html/Ahmed_Kestrel_3D_Multimodal_LLM_for_Part-Aware_Grounded_Description_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ahmed_Kestrel_3D_Multimodal_LLM_for_Part-Aware_Grounded_Description_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce Part-Aware Point Grounded Description (PaPGD), a challenging task aimed at advancing 3D multimodal learning for fine-grained, part-aware segmentation grounding and detailed explanation of 3D objects. Existing 3D datasets largely focus on either vision-only part segmentation or vision-language scene segmentation, lacking the fine-grained multimodal segmentation needed for robotic navigation and interaction in real-world environments. To address this gap, we present the 3DCoMPaT Grounded Instructions (3DCoMPaT-GrIn) Dataset, a comprehensive resource that pairs rich point cloud descriptions with corresponding part-level segmentation masks. This dataset encompasses extensive samples designed for both PaPGD and fine-grained single-part grounding tasks. To tackle the inherent challenges of grounding objects and generating grounded descriptions at the part level, we propose Kestrel, a part-aware 3D multimodal large language model that integrates an advanced language model for nuanced language comprehension with multi-level point feature propagation and query refinement mechanism to enhance spatial reasoning at the part level. The extensive experiments demonstrate that Kestrel effectively bridges the gap between part-aware language understanding and 3D segmentation grounding, paving the way for more robust and interpretable 3D object comprehension that meets the demands of real-world robotic applications.

</details>

---

## 3. ProJudge: A Multi-Modal Multi-Discipline Benchmark and Instruction-Tuning Dataset for MLLM-based Process Judges

- [ ] ProJudge: A Multi-Modal Multi-Discipline Benchmark and Instruction-Tuning Dataset for MLLM-based Process Judges | https://openaccess.thecvf.com/content/ICCV2025/html/Ai_ProJudge_A_Multi-Modal_Multi-Discipline_Benchmark_and_Instruction-Tuning_Dataset_for_MLLM-based_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ai_ProJudge_A_Multi-Modal_Multi-Discipline_Benchmark_and_Instruction-Tuning_Dataset_for_MLLM-based_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As multi-modal large language models (MLLMs) frequently exhibit errors when solving scientific problems, evaluating the validity of their reasoning processes is critical for ensuring reliability and uncovering fine-grained model weaknesses. Since human evaluation is laborious and costly, prompting MLLMs as automated process judges has become a common practice. However, the reliability of these model-based judges remains uncertain. To address this, we introduce ProJudgeBench, the first comprehensive benchmark specifically designed for evaluating abilities of MLLM-based process judges. ProJudgeBench comprises 2,400 test cases and 50,118 step-level labels, spanning four scientific disciplines with diverse difficulty levels and multi-modal content. In ProJudgeBench, each step is meticulously annotated by human experts for correctness, error type, and explanation, enabling a systematic evaluation of judges' capabilities to detect, classify and diagnose errors. Evaluation on ProJudgeBench reveals a significant performance gap between open-source and proprietary models. To bridge this gap, we further propose ProJudge-173k, a large-scale instruction-tuning dataset, and a Dynamic Dual-Phase fine-tuning strategy that encourages models to explicitly reason through problem-solving before assessing solutions. Both contributions significantly enhance the process evaluation capabilities of open-source models. This project is available at: https://projudge.github.io.

</details>

---

## 4. Towards Higher Effective Rank in Parameter-Efficient Fine-tuning using Khatri-Rao Product

- [ ] Towards Higher Effective Rank in Parameter-Efficient Fine-tuning using Khatri-Rao Product | https://openaccess.thecvf.com/content/ICCV2025/html/Albert_Towards_Higher_Effective_Rank_in_Parameter-Efficient_Fine-tuning_using_Khatri-Rao_Product_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Albert_Towards_Higher_Effective_Rank_in_Parameter-Efficient_Fine-tuning_using_Khatri-Rao_Product_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Parameter-efficient fine-tuning (PEFT) has become a standard for adapting large pre-trained models. While low-rank adaptation (LoRA) has achieved notable success, recent studies highlight its limitations when compared to full-rank variants, particularly when scaling to demanding tasks such as vision-language classification or common-sense reasoning.We propose to quantitavely compare full and rank-restricted PEFT methods using a spectrum-controlled matrix approximation benchmark. Our results validate LoRA's rank limitations when approximating matrix presenting highly decorrelated or high frequency features. We further show that full-rank methods can reduce LoRA's approximation error on these matrix types for an equal parameter count.Our evaluation then extends beyond synthetic tasks where we observe that LoRA's restricted work subspace can produce high norm updates, leading to over-fitting and poor out-of-distribution generalization. We address these limits by introducing KRAdapter, a novel PEFT algorithms that uses properties of the Kathri-Rao matrix product to produce weight matrices of higher effective rank and lower norm than related PEFT algorithms.We show the performance improvements of KRAdapter on vision-language models up to 1B parameters and 8B %32Bfor LLMs where we report from 20 to 25 points of accuracy improvements over LoRA when reasoning on commonsense tasks unseen during training. Crucially, KRAdapter maintains the favorable training speed and memory efficiency of LoRA, making it a practical and robust alternative to fine-tune billion-scale parameter models. Code for reproducing toy experiments is available in the supplementary and will be released upon acceptance.

</details>

---

## 5. NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection

- [ ] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Ansari_NegRefine_Refining_Negative_Label-Based_Zero-Shot_OOD_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ansari_NegRefine_Refining_Negative_Label-Based_Zero-Shot_OOD_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language Models like CLIP have enabled zero-shot OOD detection by leveraging both image and textual label information. Among these, negative label-based methods such as NegLabel and CSP have shown promising results by utilizing a lexicon of words to define negative labels for distinguishing OOD samples. However, these methods suffer from detecting in-distribution samples as OOD due to negative labels that are subcategories of in-distribution labels or proper nouns. They also face limitations in handling images that match multiple in-distribution and negative labels. We propose NegRefine, a novel negative label refinement framework for zero-shot OOD detection. By introducing a filtering mechanism to exclude subcategory labels and proper nouns from the negative label set and incorporating a multi-matching-aware scoring function that dynamically adjusts the contributions of multiple labels matching an image, NegRefine ensures a more robust separation between in-distribution and OOD samples. We evaluate NegRefine on large-scale benchmarks, including ImageNet-1K. The code is available at https://github.com/ah-ansari/NegRefine.

</details>

---

## 6. DASH: Detection and Assessment of Systematic Hallucinations of VLMs

- [ ] DASH: Detection and Assessment of Systematic Hallucinations of VLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Augustin_DASH_Detection_and_Assessment_of_Systematic_Hallucinations_of_VLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Augustin_DASH_Detection_and_Assessment_of_Systematic_Hallucinations_of_VLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) are prone to object hal- lucinations, where they erroneously indicate the presence of certain objects in an image. Existing benchmarks quantify hallucinations using relatively small, labeled datasets. However, this approach is i) insufficient to assess hallucinations that arise in open-world settings, where VLMs are widely used, and ii) inadequate for detecting systematic errors in VLMs. We propose DASH (Detection and Assess- ment of Systematic Hallucinations), an automatic, large-scale pipeline designed to identify systematic hallucinations of VLMs on real-world images in an open-world setting. A key component is DASH-OPT for image-based retrieval, where we optimize over the "natural image manifold" to generate images that mislead the VLM. The output of DASH consists of clusters of real and semantically similar images for which the VLM hallucinates an object. We apply DASH to PaliGemma and two LLaVA-NeXT models across 380 object classes and, in total, find more than 19k clusters with 950k images. We study the transfer of the identified systematic hallucinations to other VLMs and show that fine-tuning PaliGemma with the model-specific images obtained with DASH mitigates object hallucinations.

</details>

---

## 7. DisenQ: Disentangling Q-Former for Activity-Biometrics

- [ ] DisenQ: Disentangling Q-Former for Activity-Biometrics | https://openaccess.thecvf.com/content/ICCV2025/html/Azad_DisenQ_Disentangling_Q-Former_for_Activity-Biometrics_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Azad_DisenQ_Disentangling_Q-Former_for_Activity-Biometrics_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we address activity-biometrics, which involves identifying individuals across diverse set of activities. Unlike traditional person identification, this setting introduces additional challenges as identity cues become entangled with motion dynamics and appearance variations, making biometrics feature learning more complex. While additional visual data like pose and/or silhouette help, they often struggle from extraction inaccuracies. To overcome this, we propose a multimodal language-guided framework that replaces reliance on additional visual data with structured textual supervision. At its core, we introduce **DisenQ** (**Disen**tangling **Q**-Former), a unified querying transformer that disentangles biometrics, motion, and non-biometrics features by leveraging structured language guidance. This ensures identity cues remain independent of appearance and motion variations, preventing misidentifications. We evaluate our approach on three activity-based video benchmarks, achieving state-of-the-art performance. Additionally, we demonstrate strong generalization to complex real-world scenario with competitive performance on a traditional video-based identification benchmark, showing the effectiveness of our framework.

</details>

---

## 8. Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences

- [ ] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences | https://openaccess.thecvf.com/content/ICCV2025/html/Bahng_Cycle_Consistency_as_Reward_Learning_Image-Text_Alignment_without_Human_Preferences_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Bahng_Cycle_Consistency_as_Reward_Learning_Image-Text_Alignment_without_Human_Preferences_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning alignment between language and vision is a fundamental challenge, especially as multimodal data becomes increasingly detailed and complex. Existing methods often rely on collecting human or AI preferences, which can be costly and time-intensive. We propose an alternative approach that leverages cycle consistency as a supervisory signal. Given an image and generated text, we map the text back to image space using a text-to-image model and compute the similarity between the original image and its reconstruction. Analogously, for text-to-image generation, we measure the textual similarity between an input caption and its reconstruction through the cycle. We use the cycle consistency score to rank candidates and construct a preference dataset of 866K comparison pairs. The reward model trained on our dataset, CycleReward, outperforms state-of-the-art alignment metrics on detailed captioning, with superior inference-time scalability when used as a verifier for Best-of-N sampling, while maintaining speed and differentiability. Furthermore, performing DPO and Diffusion DPO using our dataset enhances performance across a wide range of vision-language tasks and text-to-image generation. Our dataset, model, and code are publicly released at https://cyclereward.github.io/.

</details>

---

## 9. Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models

- [ ] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Balazadeh_Physics_Context_Builders_A_Modular_Framework_for_Physical_Reasoning_in_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Balazadeh_Physics_Context_Builders_A_Modular_Framework_for_Physical_Reasoning_in_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Physical reasoning remains a significant challenge for Vision-Language Models (VLMs). This limitation arises from an inability to translate learned knowledge into predictions about physical behavior. Although continual fine-tuning can mitigate this issue, it is expensive for large models and impractical to perform repeatedly for every task. This necessitates the creation of modular and scalable ways to teach VLMs about physical reasoning. To that end, we introduce Physics Context Builders (PCBs), a modular framework where specialized smaller VLMs are fine-tuned to generate detailed physical scene descriptions. These can be used as physical contexts to enhance the reasoning capabilities of larger VLMs. PCBs enable the separation of visual perception from reasoning, allowing us to analyze their relative contributions to physical understanding. We perform experiments on CLEVRER and on Falling Tower, a stability detection dataset with both simulated and real-world scenes, to demonstrate that PCBs provide substantial performance improvements, increasing average accuracy by up to 13.8% on complex physical reasoning tasks. Notably, PCBs also show strong Sim2Real transfer, successfully generalizing from simulated training data to real-world scenes.

</details>

---

## 10. Understanding Museum Exhibits using Vision-Language Reasoning

- [ ] Understanding Museum Exhibits using Vision-Language Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Balauca_Understanding_Museum_Exhibits_using_Vision-Language_Reasoning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Balauca_Understanding_Museum_Exhibits_using_Vision-Language_Reasoning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Museums serve as repositories of cultural heritage and historical artifacts from diverse epochs, civilizations, and regions, preserving well-documented collections that encapsulate vast knowledge, which, when systematically structured into large-scale datasets, can train specialized models. Visitors engage with exhibits through curiosity and questions, making expert domain-specific models essential for interactive query resolution and gaining historical insights. Understanding exhibits from images requires analyzing visual features and linking them to historical knowledge to derive meaningful correlations. We facilitate such reasoning by (a) collecting and curating a large-scale dataset of 65M images and 200M question-answer pairs for exhibits from all around the world; (b) training large vision-language models (VLMs) on the collected dataset; (c) benchmarking their ability on five visual question answering tasks, specifically designed to reflect real-world inquiries and challenges observed in museum settings.The complete dataset is labeled by museum experts, ensuring the quality and the practical significance of the labels. We train two VLMs from different categories: BLIP with vision-language aligned embeddings, but lacking the expressive power of large language models, and the LLaVA model, a powerful instruction-tuned LLM enriched with vision-language reasoning capabilities. Through extensive experiments, we find that while both model types effectively answer visually grounded questions, large vision-language models excel in queries requiring deeper historical context and reasoning. We further demonstrate the necessity of fine-tuning models on large-scale domain-specific datasets by showing that our fine-tuned models significantly outperform current SOTA VLMs in answering questions related to specific attributes, highlighting their limitations in handling complex, nuanced queries. Our dataset, benchmarks, and source code will be made publicly available.

</details>

---

## 11. DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding

- [ ] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Bao_DynImg_Key_Frames_with_Visual_Prompts_are_Good_Representation_for_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Bao_DynImg_Key_Frames_with_Visual_Prompts_are_Good_Representation_for_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, the introduction of Multi-modal Large Language Models (MLLMs) into video understanding tasks has become increasingly prevalent. However, how to effectively integrate temporal information remains a critical research focus. Traditional approaches treat spatial and temporal information separately. Due to issues like motion blur, it is challenging to accurately represent the spatial information of rapidly moving objects. This can lead to temporally important regions being underemphasized during spatial feature extraction, which in turn hinders accurate spatio-temporal interaction and video understanding. To address this limitation, we propose an innovative video representation method called Dynamic-Image (DynImg). Specifically, we introduce a set of non-key frames as temporal prompts to highlight the spatial areas containing fast-moving objects. During the process of visual feature extraction, these prompts guide the model to pay additional attention to the fine-grained spatial features corresponding to these regions. Moreover, to maintain the correct sequence for DynImg, we employ a corresponding 4D video Rotary Position Embedding. This retains both the temporal and spatial adjacency of DynImg, helping MLLM understand the spatio-temporal order within this combined format. Experimental evaluations reveal that DynImg surpasses the state-of-the-art methods by approximately 2% across multiple video understanding benchmarks, proving the effectiveness of our temporal prompts in enhancing video comprehension.

</details>

---

## 12. Latte: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning

- [ ] Latte: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Bao_Latte_Collaborative_Test-Time_Adaptation_of_Vision-Language_Models_in_Federated_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Bao_Latte_Collaborative_Test-Time_Adaptation_of_Vision-Language_Models_in_Federated_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation with pre-trained vision-language models has gained increasing attention for addressing distribution shifts during testing. Among these approaches, memory-based algorithms stand out due to their training-free nature and ability to leverage historical test data. However, existing test-time adaptation methods are typically designed for a single domain with abundant data. In decentralized settings such as federated learning, applying these methods individually to each client suffers from limited test data, while directly sharing a single global memory via the server prevents proper personalization to each client's unique distribution. To address this, we propose Latte, a novel framework where each client maintains a local memory to store embeddings from its own historical test data and an external memory to store class prototypes from other relevant clients. During communication, each client retrieves prototypes from similar clients under the server's coordination to expand its memory. For local adaptation, Latte utilizes both embedding similarity and uncertainty to enhance model performance. Our theoretical analysis shows that Latte effectively leverages in-distribution clients while remaining robust to out-of-distribution clients. Extensive experiments on domain adaptation and corruption benchmarks validate that Latte achieves superior performance in decentralized settings, while introducing only negligible communication and computation costs. Our code is available at https://github.com/baowenxuan/Latte.

</details>

---

## 13. What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models

- [ ] What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Baraldi_What_Changed_Detecting_and_Evaluating_Instruction-Guided_Image_Edits_with_Multimodal_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Baraldi_What_Changed_Detecting_and_Evaluating_Instruction-Guided_Image_Edits_with_Multimodal_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Instruction-based image editing models offer increased personalization opportunities in generative tasks. However, properly evaluating their results is challenging, and most of the existing metrics lag in terms of alignment with human judgment and explainability. To tackle these issues, we introduce DICE (DIfference Coherence Estimator), a model designed to detect localized differences between the original and the edited image and to assess their relevance to the given modification request. DICE consists of two key components: a difference detector and a coherence estimator, both built on an autoregressive Multimodal Large Language Model (MLLM) and trained using a strategy that leverages self-supervision, distillation from inpainting networks, and full supervision. Through extensive experiments, we evaluate each stage of our pipeline, comparing different MLLMs within the proposed framework. We demonstrate that DICE effectively identifies coherent edits, effectively evaluating images generated by different editing models with a strong correlation with human judgment. We publicly release our source code, models, and data at https://aimagelab.github.io/DICE.

</details>

---

## 14. Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation

- [ ] Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Barsellotti_Talking_to_DINO_Bridging_Self-Supervised_Vision_Backbones_with_Language_for_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Barsellotti_Talking_to_DINO_Bridging_Self-Supervised_Vision_Backbones_with_Language_for_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-Vocabulary Segmentation (OVS) aims at segmenting images from free-form textual concepts without predefined training classes. While existing vision-language models such as CLIP can generate segmentation masks by leveraging coarse spatial information from Vision Transformers, they face challenges in spatial localization due to their global alignment of image and text features. Conversely, self-supervised visual models like DINO excel in fine-grained visual encoding but lack integration with language. To bridge this gap, we present Talk2DINO, a novel hybrid approach that combines the spatial accuracy of DINOv2 with the language understanding of CLIP. Our approach aligns the textual embeddings of CLIP to the patch-level features of DINOv2 through a learned mapping function without the need to fine-tune the underlying backbones. At training time, we exploit the attention maps of DINOv2 to selectively align local visual patches with textual embeddings. We show that the powerful semantic and localization abilities of Talk2DINO can enhance the segmentation process, resulting in more natural and less noisy segmentations, and that our approach can also effectively distinguish foreground objects from the background. Experimental results demonstrate that Talk2DINO achieves state-of-the-art performance across several unsupervised OVS benchmarks.

</details>

---

## 15. RadGPT: Constructing 3D Image-Text Tumor Datasets

- [ ] RadGPT: Constructing 3D Image-Text Tumor Datasets | https://openaccess.thecvf.com/content/ICCV2025/html/Bassi_RadGPT_Constructing_3D_Image-Text_Tumor_Datasets_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Bassi_RadGPT_Constructing_3D_Image-Text_Tumor_Datasets_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Cancers identified in CT scans are usually accompanied by detailed radiology reports, but publicly available CT datasets often lack these essential reports. This absence limits their usefulness for developing accurate report generation AI. To address this gap, we present AbdomenAtlas 3.0, the first public, high-quality abdominal CT dataset with detailed, expert-reviewed radiology reports. All reports are paired with per-voxel masks and they describe liver, kidney and pancreatic tumors. AbdomenAtlas 3.0 has 9,262 triplets of CT, mask and report--3,955 with tumors. These CT scans come from 17 public datasets. Besides creating the reports for these datasets, we expanded their number of tumor masks by 4.2x, identifying 3,011 new tumor cases. Notably, the reports in AbdomenAtlas 3.0 are more standardized, and generated faster than traditional human-made reports. They provide details like tumor size, location, attenuation and surgical resectability. These reports were created by 12 board-certified radiologists using our proposed RadGPT, a novel framework that converted radiologist-revised tumor segmentation masks into structured and narrative reports. Besides being a dataset creation tool, RadGPT can also become a fully-automatic, segmentation-assisted report generation method. We benchmarked this method and 5 state-of-the-art report generation vision-language models. Our results show that segmentation strongly improves tumor detection in AI-made reports.

</details>

---

## 16. TWIST & SCOUT: Grounding Multimodal LLM-Experts by Forget-Free Tuning

- [ ] TWIST & SCOUT: Grounding Multimodal LLM-Experts by Forget-Free Tuning | https://openaccess.thecvf.com/content/ICCV2025/html/Bhowmik_TWIST__SCOUT_Grounding_Multimodal_LLM-Experts_by_Forget-Free_Tuning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Bhowmik_TWIST__SCOUT_Grounding_Multimodal_LLM-Experts_by_Forget-Free_Tuning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spatial awareness is key to enable embodied multimodal AI systems. Yet, without vast amounts of spatial supervision, current Multimodal Large Language Models (MLLMs) struggle at this task. In this paper, we introduce TWIST & SCOUT, a framework that equips pre-trained MLLMs with visual grounding ability without forgetting their existing image and language understanding skills. To this end, we propose TWIST, a twin-expert stepwise tuning module that modifies the decoder of the language model using one frozen module pre-trained on image understanding tasks and another learnable one for visual grounding tasks. This allows the MLLM to retain previously learned knowledge and skills, while acquiring what is missing. To fine-tune the model effectively, we generate a high-quality synthetic dataset we call SCOUT, which mimics human reasoning in visual grounding. This dataset provides rich supervision signals, describing a step-by-step multimodal reasoning process, thereby simplifying the task of visual grounding. We evaluate our approach on several standard benchmark datasets, encompassing grounded image captioning, zero-shot localization, and visual grounding tasks. Our method consistently delivers strong performance across all tasks, while retaining the pre-trained image understanding capabilities.

</details>

---

## 17. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion

- [ ] Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion | https://openaccess.thecvf.com/content/ICCV2025/html/Bian_Prompt-driven_Transferable_Adversarial_Attack_on_Person_Re-Identification_with_Attribute-aware_Textual_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Bian_Prompt-driven_Transferable_Adversarial_Attack_on_Person_Re-Identification_with_Attribute-aware_Textual_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.

</details>

---

## 18. LLaVA-KD: A Framework of Distilling Multimodal Large Language Models

- [ ] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The success of Large Language Models (LLMs) has inspired the development of Multimodal Large Language Models (MLLMs) for unified understanding of vision and language. However, the increasing model size and computational complexity of large-scale MLLMs (l-MLLMs) limit their use in resource-constrained scenarios. Although small-scale MLLMs (s-MLLMs) are designed to reduce computational costs, they typically suffer from performance degradation.To mitigate this limitation, we propose a novel LLaVA-KD framework to transfer knowledge from l-MLLMs to s-MLLMs. Specifically, we introduce Multimodal Distillation (MDist) to transfer teacher model's robust representations across both visual and linguistic modalities, and Relation Distillation (RDist) to transfer teacher model's ability to capture visual token relationships.Additionally, we propose a three-stage training scheme to fully exploit the potential of the proposed distillation strategy: 1) Distilled Pre-Training to strengthen the alignment between visual-linguistic representations in s-MLLMs, 2) Supervised Fine-Tuning to equip the s-MLLMs with multimodal understanding capacity, and 3) Distilled Fine-Tuning to refine s-MLLM's knowledge.Our approach significantly improves s-MLLMs performance without altering the model architecture. Extensive experiments and ablation studies validate the effectiveness of each proposed component. Code will be available.

</details>

---

## 19. NAVER: A Neuro-Symbolic Compositional Automaton for Visual Grounding with Explicit Logic Reasoning

- [ ] NAVER: A Neuro-Symbolic Compositional Automaton for Visual Grounding with Explicit Logic Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Cai_NAVER_A_Neuro-Symbolic_Compositional_Automaton_for_Visual_Grounding_with_Explicit_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cai_NAVER_A_Neuro-Symbolic_Compositional_Automaton_for_Visual_Grounding_with_Explicit_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Grounding (VG) tasks, such as referring expression detection and segmentation tasks are important for linking visual entities to context, especially in complex reasoning tasks that require detailed query interpretation. This paper explores VG beyond basic perception, highlighting challenges for methods that require reasoning like human cognition. Recent advances in large language methods (LLMs) and Vision-Language methods (VLMs) have improved abilities for visual comprehension, contextual understanding, and reasoning. These methods are mainly split into end-to-end and compositional methods, with the latter offering more flexibility. Compositional approaches that integrate LLMs and foundation models show promising performance but still struggle with complex reasoning with language-based logical representations. To address these limitations, we propose NAVER, a compositional visual grounding method that integrates explicit probabilistic logic reasoning within a finite-state automaton, equipped with a self-correcting mechanism. This design improves robustness and interpretability in inference through explicit logic reasoning. Our results show that NAVER achieves SoTA performance comparing to recent end-to-end and compositional baselines. The code is available at https://github.com/ControlNet/NAVER.

</details>

---

## 20. Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training

- [ ] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training | https://openaccess.thecvf.com/content/ICCV2025/html/Cao_Boosting_Vision_Semantic_Density_with_Anatomy_Normality_Modeling_for_Medical_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cao_Boosting_Vision_Semantic_Density_with_Anatomy_Normality_Modeling_for_Medical_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training (VLP) has great potential for developing multifunctional and general medical diagnostic capabilities. However, aligning medical images with a low signal-to-noise ratio (SNR) to reports with a high SNR presents a semantic density gap, leading to visual alignment bias. In this paper, we propose boosting vision semantic density to improve alignment effectiveness. On the one hand, we enhance visual semantics through disease-level vision contrastive learning, which strengthens the model's ability to differentiate between normal and abnormal samples for each anatomical structure. On the other hand, we introduce an anatomical normality modeling method to model the distribution of normal samples for each anatomy, leveraging VQ-VAE for reconstructing normal vision embeddings in latent space. This process amplifies abnormal signals by leveraging distribution shifts in abnormal samples, enhancing the model's perception and discrimination of abnormal attributes. The enhanced visual representation effectively captures the diagnostic-relevant semantics, facilitating more efficient and accurate alignment with the diagnostic report. We conduct extensive experiments on two chest CT datasets, CT-RATE and Rad-ChestCT, and an abdominal CT dataset, MedVL-69K, and comprehensively evaluate the diagnosis performance across multiple tasks in the chest and abdominal CT scenarios, achieving state-of-the-art zero-shot performance. Notably, our method achieved an average AUC of 84.9% across 54 diseases in 15 organs, significantly surpassing existing methods. Additionally, we demonstrate the superior transfer learning capabilities of our pre-trained model.

</details>

---

## 21. IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark

- [ ] IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark | https://openaccess.thecvf.com/content/ICCV2025/html/Cao_IRGPT_Understanding_Real-world_Infrared_Image_with_Bi-cross-modal_Curriculum_on_Large-scale_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cao_IRGPT_Understanding_Real-world_Infrared_Image_with_Bi-cross-modal_Curriculum_on_Large-scale_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Real-world infrared imagery presents unique challenges for vision-language models due to the scarcity of aligned text data and domain-specific characteristics. Although existing methods have advanced the field, their reliance on synthetic infrared images generated through style transfer from visible images, which limits their ability to capture the unique characteristics of the infrared modality. To address this, we propose IRGPT, the first multi-modal large language model for real-world infrared images, built upon a large-scale InfraRed-Text Dataset (IR-TD) comprising over 260K authentic image-text pairs. The proposed IR-TD dataset contains real infrared images paired with meticulously handcrafted texts, where the initial drafts originated from two complementary processes: (1) LLM-generated descriptions of visible images, and (2) rule-based descriptions of annotations. Furthermore, we introduce a bi-cross-modal curriculum transfer learning strategy that systematically transfers knowledge from visible to infrared domains by considering the difficulty scores of both infrared-visible and infrared-text. Evaluated on a benchmark of 9 tasks (e.g., recognition, grounding), IRGPT achieves state-of-the-art performance even compared with larger-scale models.

</details>

---

## 22. MotionCtrl: A Real-time Controllable Vision-Language-Motion Model

- [ ] MotionCtrl: A Real-time Controllable Vision-Language-Motion Model | https://openaccess.thecvf.com/content/ICCV2025/html/Cao_MotionCtrl_A_Real-time_Controllable_Vision-Language-Motion_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cao_MotionCtrl_A_Real-time_Controllable_Vision-Language-Motion_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human motion generation involves synthesizing coherent human motion sequences conditioned on diverse multimodal inputs and holds significant potential for real-world applications. Despite recent advancements, existing vision-language-motion models (VLMMs) remain limited in achieving this goal. In this paper, we identify the lack of controllability as a critical bottleneck, where VLMMs struggle with diverse human commands, pose initialization, generation of long-term or unseen cases, and fine-grained control over individual body parts. To address these challenges, we introduce MotionCtrl, the first real-time, controllable VLMM with state-of-the-art performance. MotionCtrl achieves its controllability through training on HuMo100M, the largest human motion dataset to date, featuring over 5 million self-collected motions, 100 million multi-task instructional instances, and detailed part-level descriptions that address a long-standing gap in the field. Additionally, we propose a novel part-aware residual quantization technique for motion tokenization, enabling precise control over individual body parts during motion generation. Extensive experiments demonstrate MotionCtrl's superior performance across a wide range of motion benchmarks. Furthermore, we provide strategic design insights and a detailed time efficiency analysis to guide the development of practical motion generators.

</details>

---

## 23. Refer to Any Segmentation Mask Group With Vision-Language Prompts

- [ ] Refer to Any Segmentation Mask Group With Vision-Language Prompts | https://openaccess.thecvf.com/content/ICCV2025/html/Cao_Refer_to_Any_Segmentation_Mask_Group_With_Vision-Language_Prompts_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cao_Refer_to_Any_Segmentation_Mask_Group_With_Vision-Language_Prompts_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent image segmentation models have advanced to segment images into high-quality masks for visual entities, and yet they cannot provide comprehensive semantic understanding for complex queries based on both language and vision. This limitation reduces their effectiveness in applications that require user-friendly interactions driven by vision-language prompts. To bridge this gap, we introduce a novel task of omnimodal referring expression segmentation (ORES). In this task, a model produces a group of masks based on arbitrary prompts specified by text only or text plus reference visual entities. To address this new challenge, we propose a novel framework to "Refer to Any Segmentation Mask Group" (RAS), which augments segmentation models with complex multimodal interactions and comprehension via a mask-centric large multimodal model. For training and benchmarking ORES models, we create datasets MaskGroups-2M and MaskGroups-HQ to include diverse mask groups specified by text and reference entities. Through extensive evaluation, we demonstrate superior performance of RAS on our new ORES task, as well as classic referring expression segmentation (RES) and generalized referring expression segmentation (GRES) tasks. Project page: https://Ref2Any.github.io.

</details>

---

## 24. VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization

- [ ] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization | https://openaccess.thecvf.com/content/ICCV2025/html/Cao_VideoMiner_Iteratively_Grounding_Key_Frames_of_Hour-Long_Videos_via_Tree-based_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cao_VideoMiner_Iteratively_Grounding_Key_Frames_of_Hour-Long_Videos_via_Tree-based_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding hour-long videos with multi-modal large language models (MM-LLMs) enriches the landscape of human-centered AI applications. However, for end-to-end video understanding with LLMs, uniformly sampling video frames results in LLMs being overwhelmed by a vast amount of irrelevant information as video length increases. Existing hierarchical key frame extraction methods improve the accuracy of video understanding but still face two critical challenges. 1) How can the interference of extensive redundant information in long videos be mitigated? 2) How can a model dynamically adapt to complex hierarchical structures while accurately identifying key frames? To address these issues, we propose VideoMiner, which iteratively segments, captions, and clusters long videos, forming a hierarchical tree structure. The proposed VideoMiner progresses from long videos to events to frames while preserving temporal coherence, effectively addressing the first challenge. To precisely locate key frames, we introduce T-GRPO, a tree-based group relative policy optimization in reinforcement learning method that guides the exploration of the VideoMiner. The proposed T-GRPO is specifically designed for tree structures, integrating spatiotemporal information at the event level while being guided by the question, thus solving the second challenge. We achieve superior performance in all long-video understanding tasks and uncover several interesting insights. Our proposed T-GRPO surprisingly incentivizes the model to spontaneously generate a reasoning chain. Additionally, the designed tree growth auxin dynamically adjusts the expansion depth, obtaining accuracy and efficiency gains. The code is publicly available at https://github.com/caoxinye/VideoMiner.

</details>

---

## 25. HouseTour: A Virtual Real Estate A(I)gent

- [ ] HouseTour: A Virtual Real Estate A(I)gent | https://openaccess.thecvf.com/content/ICCV2025/html/Celen_HouseTour_A_Virtual_Real_Estate_AIgent_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Celen_HouseTour_A_Virtual_Real_Estate_AIgent_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce HouseTour, a method for spatially-aware 3D camera trajectory and natural language summary generation from a collection of images depicting an existing 3D space. Unlike existing vision-language models (VLMs), which struggle with geometric reasoning, our approach generates smooth video trajectories via a diffusion process constrained by known camera poses and integrates this information into the VLM for 3D-grounded descriptions. We synthesize the final video using 3D Gaussian splatting to render novel views along the trajectory. To support this task, we present the HouseTour dataset, which includes over 1,200 house-tour videos with camera poses, 3D reconstructions, and real estate descriptions. Experiments demonstrate that incorporating 3D camera trajectories into the text generation process improves performance over methods handling each task independently. We evaluate both individual and end-to-end performance, introducing a new joint metric. Our work enables automated, professional-quality video creation for real estate and touristic applications without requiring specialized expertise or equipment.

</details>

---

## 26. Hallucinatory Image Tokens: A Training-free EAZY Approach to Detecting and Mitigating Object Hallucinations in LVLMs

- [ ] Hallucinatory Image Tokens: A Training-free EAZY Approach to Detecting and Mitigating Object Hallucinations in LVLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Che_Hallucinatory_Image_Tokens_A_Training-free_EAZY_Approach_to_Detecting_and_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Che_Hallucinatory_Image_Tokens_A_Training-free_EAZY_Approach_to_Detecting_and_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite their remarkable potential, Large Vision-Language Models (LVLMs) still face challenges with object hallucination, a problem where their generated outputs mistakenly incorporate objects that do not actually exist. Although most works focus on addressing this issue within the language-model backbone, our work shifts the focus to the image input source, investigating how specific image tokens contribute to hallucinations. Our analysis reveals that a small subset of image tokens with high attention scores are the main drivers of object hallucination. By removing these hallucinatory image tokens (only 1.5% of all image tokens), the issue can be effectively mitigated. This finding holds consistently across different models. Building on this insight, we introduce \eazy, a novel, training-free method that automatically identifies and Eliminates hAllucinations by Zeroing out hallucinator Y image tokens. We utilize EAZY for unsupervised object hallucination detection, achieving a 15% improvement compared to previous methods. Additionally, EAZY demonstrates remarkable effectiveness in mitigating hallucinations while preserving model utility and seamlessly adapting to various LVLM architectures.

</details>

---

## 27. ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation

- [ ] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_ADIEE_Automatic_Dataset_Creation_and_Scorer_for_Instruction-Guided_Image_Editing_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_ADIEE_Automatic_Dataset_Creation_and_Scorer_for_Instruction-Guided_Image_Editing_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in instruction-guided image editing underscore the need for effective automated evaluation. While Vision-Language Models (VLMs) have been explored as judges, open-source models struggle with alignment, and proprietary models lack transparency and cost efficiency. Additionally, no public training datasets exist to fine-tune open-source VLMs, only small benchmarks with diverse evaluation schemes. To address this, we introduce ADIEE, an automated dataset creation approach which is then used to train a scoring model for instruction-guided image editing evaluation. We generate a large-scale dataset with over 100K samples and use it to fine-tune a LLaVA-NeXT-8B model modified to decode a numeric score from a custom token. The resulting scorer outperforms all open-source VLMs and Gemini-Pro 1.5 across all benchmarks, achieving a 0.0696 (+17.24%) gain in score correlation with human ratings on AURORA-Bench, and improving pair-wise comparison accuracy by 4.03% (+7.21%) on GenAI-Bench and 4.75% (+9.35%) on AURORA-Bench, respectively, compared to the state-of-the-art. The scorer can act as a reward model, enabling automated best edit selection and model fine-tuning. Notably, the proposed scorer can boost MagicBrush model's average evaluation score on ImagenHub from 5.90 to 6.43 (+8.98%). Our code and models are available at https://github.com/SherryXTChen/ADIEE.git.

</details>

---

## 28. Aligning Effective Tokens with Video Anomaly in Large Language Models

- [ ] Aligning Effective Tokens with Video Anomaly in Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding abnormal events in videos is a vital and challenging task that has garnered significant attention in a wide range of applications. Although current video understanding Multi-modal Large Language Models (MLLMs) are capable of analyzing general videos, they often struggle to handle anomalies due to the spatial and temporal sparsity of abnormal events, where the redundant information always leads to suboptimal outcomes. To address these challenges, exploiting the representation and generalization capabilities of Vison Language Models (VLMs) and Large Language Models (LLMs), we propose VA-GPT, a novel MLLM designed for summarizing and localizing abnormal events in various videos. Our approach efficiently aligns effective tokens between visual encoders and LLMs through two key proposed modules: Spatial Effective Token Selection (SETS) and Temporal Effective Token Generation (TETG). These modules enable our model to effectively capture and analyze both spatial and temporal information associated with abnormal events, resulting in more accurate responses and interactions. Furthermore, we construct an instruction-following dataset specifically for fine-tuning video-anomaly-aware MLLMs, and introduce a cross-domain evaluation benchmark based on XD-Violence dataset. Our proposed method outperforms existing state-of-the-art methods on various benchmarks.

</details>

---

## 29. CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games

- [ ] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_CombatVLA_An_Efficient_Vision-Language-Action_Model_for_Combat_Tasks_in_3D_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_CombatVLA_An_Efficient_Vision-Language-Action_Model_for_Combat_Tasks_in_3D_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Vision-Language-Action models (VLAs) have expanded the capabilities of embodied intelligence. However, significant challenges remain in real-time decision-making in complex 3D environments, which demand second-level responses, high-resolution perception, and tactical reasoning under dynamic conditions. To advance the field, we introduce CombatVLA, an efficient VLA model optimized for combat tasks in 3D action role-playing games(ARPGs). Specifically, our CombatVLA is a 3B model trained on video-action pairs collected by an action tracker, where the data is formatted as action-of-thought (AoT) sequences. Thereafter, CombatVLA seamlessly integrates into an action execution framework, allowing efficient inference through our truncated AoT strategy. Experimental results demonstrate that CombatVLA not only outperforms all existing models on the combat understanding benchmark but also achieves a 50-fold acceleration in game combat. Moreover, it has a higher task success rate than human players. We will release all resources, including the action tracker, dataset, model weights, training code, and action execution framework implementation at https://combatvla.github.io/.

</details>

---

## 30. CompCap: Improving Multimodal Large Language Models with Composite Captions

- [ ] CompCap: Improving Multimodal Large Language Models with Composite Captions | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_CompCap_Improving_Multimodal_Large_Language_Models_with_Composite_Captions_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_CompCap_Improving_Multimodal_Large_Language_Models_with_Composite_Captions_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

How well can Multimodal Large Language Models (MLLMs) understand composite images? Composite images (CIs) are synthetic visuals created by merging multiple visual elements, such as charts, posters, or screenshots, rather than being captured directly by a camera. While CIs are prevalent in real-world applications, recent MLLM developments have primarily focused on interpreting natural images (NIs). Our research reveals that current MLLMs face significant challenges in accurately understanding CIs, often struggling to extract information or perform complex reasoning based on these images. We find that existing training data for CIs are mostly formatted for question-answer tasks (e.g., in datasets like ChartQA and ScienceQA), while high-quality image-caption datasets, critical for robust vision-language alignment, are only available for NIs. To bridge this gap, we introduce Composite Captions (CompCap), a flexible framework that leverages Large Language Models (LLMs) and automation tools to synthesize CIs with accurate and detailed captions. Using CompCap, we curate CompCap-118K, a dataset containing 118K image-caption pairs across six CI types. We validate the effectiveness of CompCap-118K by supervised fine-tuning MLLMs of three sizes: xGen-MM-inst.-4B and LLaVA-NeXT-Vicuna-7B/13B. Empirical results show that CompCap-118K significantly enhances MLLMs' understanding of CIs, yielding average gains of 1.7%, 2.0%, and 2.9% across eleven benchmarks, respectively.

</details>

---

## 31. Engage for All: Making Ordinary Image Descriptions Appealing Again!

- [ ] Engage for All: Making Ordinary Image Descriptions Appealing Again! | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Engage_for_All_Making_Ordinary_Image_Descriptions_Appealing_Again_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Engage_for_All_Making_Ordinary_Image_Descriptions_Appealing_Again_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, multi-modal large language models (MLLMs) have been successfully adopted to generate humorous and engaging descriptions for internet memes. While, it is challenging for the same approaches to apply to ordinary images which lack of inherent funny or exaggerated contents. Thus, crafting appealing descriptions for ordinary image demands imaginative efforts to discover or create intriguing connections between words to image contents. To address this gap, we introduce AppealImage, a large-scale dataset consisting of ordinary images paired with appealing descriptions. AppealImage allows us to define four distinct tasks with quantitative metrics to enable objective evaluation. Subsequently, we propose CharmNet, an innovative framework designed to generate appealing descriptions for ordinary images. CharmNet combines instruction tuning with heuristic active learning, guided by a referee model. Experimental results demonstrate that CharmNet outperforms the state-of-the-art method by 11.4% in generating appealing descriptions. Furthermore, CharmNet delivers impressive performance across various creative applications, including visual storytelling and situational dialogue generation. These results highlight CharmNet's potential to enhance social media engagement and to empower strong brand presence in competitive markets.

</details>

---

## 32. Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection via Graph Score Propagation

- [ ] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection via Graph Score Propagation | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Exploiting_Vision_Language_Model_for_Training-Free_3D_Point_Cloud_OOD_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Exploiting_Vision_Language_Model_for_Training-Free_3D_Point_Cloud_OOD_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection in 3D point cloud data remains a challenge, particularly in applications where safe and robust perception is critical. While existing OOD detection methods have shown progress for 2D image data, extending these to 3D environments involves unique obstacles. This paper introduces a training-free framework that leverages Vision-Language Models (VLMs) for effective OOD detection in 3D point clouds. By constructing a graph based on class prototypes and testing data, we exploit the data manifold structure to enhancing the effectiveness of VLMs for 3D OOD detection. We propose a novel Graph Score Propagation (GSP) method that incorporates prompt clustering and self-training negative prompting to improve OOD scoring with VLM. Our method is also adaptable to few-shot scenarios, providing options for practical applications. We demonstrate that GSP consistently outperforms state-of-the-art methods across synthetic and real-world datasets for 3D point cloud OOD detection.

</details>

---

## 33. Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model

- [ ] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Interpretable_Zero-Shot_Learning_with_Locally-Aligned_Vision-Language_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Interpretable_Zero-Shot_Learning_with_Locally-Aligned_Vision-Language_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models (VLMs), such as CLIP, have achieved remarkable success in zero-shot learning (ZSL) by leveraging large-scale visual-text pair datasets. However, these methods often lack interpretability, as they compute the similarity between an entire query image and the embedded category words, making it difficult to explain their predictions. One approach to address this issue is to develop interpretable models by integrating language, where classifiers are built using discrete attributes, similar to human perception. This introduces a new challenge: how to effectively align local visual features with corresponding attributes based on pre-trained VLMs. To tackle this, we propose LaZSL, a locally-aligned vision-language model for interpretable ZSL. LaZSL employs local visual-semantic alignment via optimal transport to perform interaction between visual regions and their associated attributes, facilitating effective alignment and providing interpretable similarity without the need for additional training. Extensive experiments demonstrate that our method offers several advantages, including enhanced interpretability, improved accuracy, and strong domain generalization. Codes available at: https://github.com/shiming-chen/LaZSL.

</details>

---

## 34. LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents

- [ ] LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LVAgent_Long_Video_Understanding_by_Multi-Round_Dynamical_Collaboration_of_MLLM_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LVAgent_Long_Video_Understanding_by_Multi-Round_Dynamical_Collaboration_of_MLLM_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing MLLMs encounter significant challenges in modeling the temporal context within long videos. Currently, mainstream Agent-based methods use external tools to assist a single MLLM in answering long video questions. Despite such tool-based support, a solitary MLLM still offers only a partial understanding of long videos, resulting in limited performance. In order to better address long video tasks, we introduce LVAgent, the first framework enabling multi-round dynamic collaboration of MLLM agents in long video understanding. Our method consists of four key steps: 1) Selection: We pre-select appropriate agents from the model library to form optimal agent teams based on different tasks. 2) Perception: We design an effective retrieval scheme for long videos, improving the coverage of critical temporal segments while maintaining computational efficiency. 3) Action: Agents answer long video questions and exchange reasons. 4) Reflection: We evaluate each agent's performance in each round of discussion and optimize the agent team for dynamic collaboration. The agents iteratively refine their answers by multi-round dynamical collaboration of MLLM agents. LVAgent is the first agent system method that outperforms all closed-source models (like GPT-4o) and open-source models (like InternVL-2.5 and Qwen2-VL) in the long video understanding tasks. Our LVAgent achieves an accuracy of 80% on four mainstream long video understanding tasks. Notably, LVAgent improves accuracy by 13.3% on LongVideoBench. Code is available at https://github.com/64327069/LVAgent.

</details>

---

## 35. Leveraging Debiased Cross-modal Attention Maps and Code-based Reasoning for Zero-shot Referring Expression Comprehension

- [ ] Leveraging Debiased Cross-modal Attention Maps and Code-based Reasoning for Zero-shot Referring Expression Comprehension | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Leveraging_Debiased_Cross-modal_Attention_Maps_and_Code-based_Reasoning_for_Zero-shot_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Leveraging_Debiased_Cross-modal_Attention_Maps_and_Code-based_Reasoning_for_Zero-shot_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot Referring Expression Comprehension (REC) aims at locating an object described by a natural language query without training on task-specific datasets. Current approaches often utilize Vision-Language Models (VLMs) to perform region-text matching based on region proposals. However, this may downgrade their performance since VLMs often fail in relation understanding and isolated proposals inevitably lack global image context. To tackle these challenges, we first design a general formulation for code-based relation reasoning. It instructs Large Language Models (LLMs) to decompose complex relations and adaptively implement code for spatial and relation computation. Moreover, we directly extract region-text relevance from cross-modal attention maps in VLMs. Observing the inherent bias in VLMs, we further develop a simple yet effective bias deduction method, which enhances attention maps' capability to align text with the corresponding regions. Experimental results on four representative datasets demonstrate the SOTA performance of our method. On the RefCOCO dataset centered on spatial understanding, our method gets an average improvement of 10% over the previous zero-shot SOTA.

</details>

---

## 36. Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models

- [ ] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Multi-Cache_Enhanced_Prototype_Learning_for_Test-Time_Generalization_of_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Multi-Cache_Enhanced_Prototype_Learning_for_Test-Time_Generalization_of_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In zero-shot setting, test-time adaptation adjusts pre-trained models using unlabeled data from the test phase to enhance performance on unknown test distributions. Existing cache-enhanced TTA methods rely on a low-entropy criterion to select samples for prototype construction, assuming intra-class compactness. However, low-entropy samples may be unreliable under distribution shifts, and the resulting prototypes may not ensure compact intra-class distributions. This study identifies a positive correlation between cache-enhanced performance and intra-class compactness. Based on this observation, we propose a Multi-Cache enhanced Prototype-based Test-Time Adaptation (MCP) featuring three caches: an entropy cache for initializing prototype representations with low-entropy samples, an align cache for integrating visual and textual information to achieve compact intra-class distributions, and a negative cache for prediction calibration using high-entropy samples. We further developed MCP++, a framework incorporating cross-modal prototype alignment and residual learning, introducing prototype residual fine-tuning. Comparative and ablation experiments across 15 downstream tasks demonstrate that the proposed method and framework achieve state-of-the-art generalization performance.

</details>

---

## 37. One Last Attention for Your Vision-Language Model

- [ ] One Last Attention for Your Vision-Language Model | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_One_Last_Attention_for_Your_Vision-Language_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_One_Last_Attention_for_Your_Vision-Language_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pretrained vision-language models (VLMs), such as CLIP, achieve remarkable zero-shot performance, yet their downstream potential hinges on effective fine-tuning. Most adaptation methods typically focus on refining representation from separate modalities (text or vision) but neglect the critical role of their fused representations in the decision-making process, i.e., rational matrix that drives the final prediction. To bridge the gap, we propose a simple yet effective Rational Adaptaion (RAda) to explicitly exploit the final fused representation during fine-tuning. RAda employs a learned mask, obtained from a lightweight attention layer attached at the end of a VLM, to dynamically calibrate the contribution of each element in the rational matrix, enabling targeted adjustments to the final cross-modal interactions without incurring costly modifications to intermediate features. Experiments in different settings (i.e., updating, or freezing pretrained encoders in adaptation, and test-time training that can only access the unlabeled test data) show that RAda serves as a versatile fine-tuning technique, improving the baseline with minimal code and performing comparably against current arts in most settings.

</details>

---

## 38. RMultiplex200K: Toward Reliable Multimodal Process Supervision for Visual Language Models on Telecommunications

- [ ] RMultiplex200K: Toward Reliable Multimodal Process Supervision for Visual Language Models on Telecommunications | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_RMultiplex200K_Toward_Reliable_Multimodal_Process_Supervision_for_Visual_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_RMultiplex200K_Toward_Reliable_Multimodal_Process_Supervision_for_Visual_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Language Models (VLMs) have achieved remarkable success in many domains due to their ability to perform step-by-step reasoning. However, progress in the telecommunication (Telecom) domain remains limited, primarily due to the lack of high-quality datasets and domain-specific insights. In this paper, we introduce RMultiplex200K, a multimodal dataset designed to present step-wise reasoning rationales and correctness scores for real-world Telecom questions. This enables VLMs to engage in step-level reasoning and verification using multimodal information, thereby facilitating reliable problem-solving. RMultiplex200K is highly scalable as it is constructed without human annotations, relying instead on our automatic plan-based annotation (ApPA) method, which automatically synthesizes reasoning steps labeled with reward scores. With this dataset, we introduce TC-NAVIGATOR, a new mechanism for training multimodal process reward models to serve as reliable reasoning verifiers for VLMs. For instance, the Qwen-2-VL-72B and Llama-3.2-90B models, which initially achieve only 21.3% and 19.8% respectively on practice Telecom questions, reached 48.5% and 46.1% accuracy, respectively, after training with RMultiplex200K and verifying with TC-NAVIGATOR.

</details>

---

## 39. Rethinking Layered Graphic Design Generation with a Top-Down Approach

- [ ] Rethinking Layered Graphic Design Generation with a Top-Down Approach | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Rethinking_Layered_Graphic_Design_Generation_with_a_Top-Down_Approach_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Rethinking_Layered_Graphic_Design_Generation_with_a_Top-Down_Approach_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphic design is crucial for conveying ideas and messages. Designers usually organize their work into objects, backgrounds, and vectorized text layers to simplify editing. However, this workflow demands considerable expertise. With the rise of GenAI methods, an endless supply of high-quality graphic designs in pixel format has become more accessible, though these designs often lack editability. Despite this, non-layered designs still inspire human designers, influencing their choices in layouts and text styles, ultimately guiding the creation of layered designs. Motivated by this observation, we propose Accordion, a graphic design generation framework taking the first attempt to convert AI-generated designs into editable layered designs, meanwhile refining nonsensical AI-generated text with meaningful alternatives guided by user prompts. It is built around a vision language model (VLM) playing distinct roles in three curated stages: (1) reference creation, (2) design planning, and (3) layer generation. For each stage, we design prompts to guide the VLM in executing different tasks. Distinct from existing bottom-up methods (e.g., COLE and Open-COLE) that gradually generate elements to create layered designs, our approach works in a top-down manner by using the visually harmonious reference image as global guidance to decompose each layer. Additionally, it leverages multiple vision experts such as SAM and element removal models to facilitate the creation of graphic layers. We train our method using the in-house graphic design dataset Design39K, augmented with AI-generated design images coupled with refined ground truth created by a customized inpainting model. Experimental results and user studies by designers show that Accordion generates favorable results on the DesignIntention benchmark, including tasks such as text-to-template, adding text to background, and text de-rendering, and also excels in creating design variations.

</details>

---

## 40. Training-Free Class Purification for Open-Vocabulary Semantic Segmentation

- [ ] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Training-Free_Class_Purification_for_Open-Vocabulary_Semantic_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Training-Free_Class_Purification_for_Open-Vocabulary_Semantic_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning pre-trained vision-language models has emerged as a powerful approach for enhancing open-vocabulary semantic segmentation (OVSS). However, the substantial computational and resource demands associated with training on large datasets have prompted interest in training-free methods for OVSS. Existing training-free approaches primarily focus on modifying model architectures and generating prototypes to improve segmentation performance. However, they often neglect the challenges posed by class redundancy, where multiple categories are not present in the current test image, and visual-language ambiguity, where semantic similarities among categories create confusion in class activation. These issues can lead to suboptimal class activation maps and affinity-refined activation maps. Motivated by these observations, we propose FreeCP, a novel training-free class purification framework designed to address these challenges. FreeCP focuses on purifying semantic categories and rectifying errors caused by redundancy and ambiguity. The purified class representations are then leveraged to produce final segmentation predictions. We conduct extensive experiments across eight benchmarks to validate FreeCP's effectiveness. Results demonstrate that FreeCP, as a plug-and-play module, significantly boosts segmentation performance when combined with other OVSS methods.

</details>

---

## 41. AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction

- [ ] AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_AnimeGamer_Infinite_Anime_Life_Simulation_with_Next_Game_State_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_AnimeGamer_Infinite_Anime_Life_Simulation_with_Next_Game_State_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in image and video synthesis have opened up new promise in generative games. One particularly intriguing application is transforming characters from anime films into interactive, playable entities. This allows players to immerse themselves in the dynamic anime world as their favorite characters for life simulation through language instructions. Such games are defined as "infinite game" since they eliminate predetermined boundaries and fixed gameplay rules, where players can interact with the game world through open-ended language and experience ever-evolving storylines and environments. Recently, a pioneering approach for infinite anime life simulation employs large language models (LLMs) to translate multi-turn text dialogues into language instructions for image generation. However, it neglects historical visual context, leading to inconsistent gameplay. Furthermore, it only generates static images, failing to incorporate the dynamics necessary for an engaging gaming experience. In this work, we propose AnimeGamer, which is built upon Multimodal Large Language Models (MLLMs) to generate each game state, including dynamic animation shots that depict character movements and updates to character states, as illustrated in Figure 1. We introduce novel action-aware multimodal representations to represent animation shots, which can be decoded into high-quality video clips using a video diffusion model. By taking historical animation shot representations as context and predicting subsequent representations, AnimeGamer can generate games with contextual consistency and satisfactory dynamics. Extensive evaluations using both automated metrics and human evaluations demonstrate that AnimeGamer outperforms existing methods in various aspects of the gaming experience.

</details>

---

## 42. MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding

- [ ] MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_MCAM_Multimodal_Causal_Analysis_Model_for_Ego-Vehicle-Level_Driving_Video_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_MCAM_Multimodal_Causal_Analysis_Model_for_Ego-Vehicle-Level_Driving_Video_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Accurate driving behavior recognition and reasoning are critical for autonomous driving video understanding. However, existing methods often tend to dig out the shallow causal, fail to address spurious correlations across modalities, and ignore the ego-vehicle level causality modeling. To overcome these limitations, we propose a novel Multimodal Causal Analysis Model (MCAM) that constructs latent causal structures between visual and language modalities. Firstly, we design a multi-level feature extractor to capture long-range dependencies. Secondly, we design a causal analysis module that dynamically models driving scenarios using a directed acyclic graph (DAG) of driving states. Thirdly, we utilize a vision-language transformer to align critical visual features with their corresponding linguistic expressions. Extensive experiments on the BDD-X, and CoVLA datasets demonstrate that MCAM achieves SOTA performance in visual-language causal relationship learning. Furthermore, the model exhibits superior capability in capturing causal characteristics within video sequences, showcasing its effectiveness for autonomous driving applications. The code is available at https://github.com/SixCorePeach/MCAM

</details>

---

## 43. SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models

- [ ] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_SimpleVQA_Multimodal_Factuality_Evaluation_for_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_SimpleVQA_Multimodal_Factuality_Evaluation_for_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The increasing application of multi-modal large language models (MLLMs) across various sectors has spotlighted the essence of their output reliability and accuracy, particularly their ability to produce content grounded in factual information (e.g. common and domain-specific knowledge). In this work, we introduce SimpleVQA, the first comprehensive multi-modal benchmark to evaluate the factuality ability of MLLMs to answer natural language short questions. SimpleVQA is characterized by 7 key features: it is based on bilingual, it covers multiple tasks and multiple scenarios, ensures high quality and challenging queries, maintains static and timeless reference answers, and is straightforward to evaluate. Our approach involves categorizing visual question-answering items into 9 different tasks around objective events or common knowledge and situating these within 9 scenario domains. Rigorous quality control processes are implemented to guarantee high-quality, concise, and clear answers, facilitating evaluation with minimal variance via an LLM-as-a-judge scoring system. Using SimpleVQA, we perform a comprehensive assessment of leading 18 MLLMs and 8 text-only LLMs, delving into their image comprehension and text generation abilities by identifying and analyzing error cases.

</details>

---

## 44. Social Debiasing for Fair Multi-modal LLMs

- [ ] Social Debiasing for Fair Multi-modal LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_Social_Debiasing_for_Fair_Multi-modal_LLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_Social_Debiasing_for_Fair_Multi-modal_LLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) have dramatically advanced the research field and delivered powerful vision-language understanding capabilities. However, these models often inherit deep-rooted social biases from their training data, leading to uncomfortable responses with respect to attributes such as race and gender. This paper addresses the issue of social biases in MLLMs by i) introducing a comprehensive counterfactual dataset with multiple social concepts (CMSC), which complements existing datasets by providing 18 diverse and balanced social concepts; and ii) proposing a counter-stereotype debiasing (CSD) strategy that mitigates social biases in MLLMs by leveraging the opposites of prevalent stereotypes. CSD incorporates both a novel bias-aware data sampling method and a loss rescaling method, enabling the model to effectively reduce biases. We conduct extensive experiments with four prevalent MLLM architectures. The results demonstrate the advantage of the CMSC dataset and the edge of CSD strategy in reducing social biases compared to existing competing methods, without compromising the overall performance on general multi-modal reasoning benchmarks.

</details>

---

## 45. OV-SCAN: Semantically Consistent Alignment for Novel Object Discovery in Open-Vocabulary 3D Object Detection

- [ ] OV-SCAN: Semantically Consistent Alignment for Novel Object Discovery in Open-Vocabulary 3D Object Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Chow_OV-SCAN_Semantically_Consistent_Alignment_for_Novel_Object_Discovery_in_Open-Vocabulary_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chow_OV-SCAN_Semantically_Consistent_Alignment_for_Novel_Object_Discovery_in_Open-Vocabulary_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary 3D object detection for autonomous driving aims to detect novel objects beyond the predefined training label sets in point cloud scenes. Existing approaches achieve this by connecting traditional 3D object detectors with vision-language models (VLMs) to regress 3D bounding boxes for novel objects and perform open-vocabulary classification through cross-modal alignment between 3D and 2D features. However, achieving robust cross-modal alignment remains a challenge due to semantic inconsistencies when generating corresponding 3D and 2D feature pairs. To overcome this challenge, we present OV-SCAN, an Open-Vocabulary 3D framework that enforces Semantically Consistent Alignment for Novel object discovery. OV-SCAN employs two core strategies: discovering precise 3D annotations and filtering out low-quality or corrupted alignment pairs (arising from 3D annotation, occlusion-induced, or resolution-induced noise). Extensive experiments on the nuScenes dataset demonstrate that OV-SCAN achieves state-of-the-art performance.

</details>

---

## 46. AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs

- [ ] AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Chowdhury_AVTrustBench_Assessing_and_Enhancing_Reliability_and_Robustness_in_Audio-Visual_LLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chowdhury_AVTrustBench_Assessing_and_Enhancing_Reliability_and_Robustness_in_Audio-Visual_LLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of Multi-modal Large Language Models (MLLMs), several diagnostic benchmarks have recently been developed to assess these models' multimodal reasoning proficiency. However, these benchmarks are restricted to assessing primarily the visual aspect and do not examine the holistic audio-visual (AV) understanding. Moreover, currently, there are no benchmarks that investigate the capabilities of AVLLMs to calibrate their responses when presented with perturbed inputs. To this end, we introduce Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench), comprising 600K samples spanning over 9 meticulously crafted tasks, evaluating the capabilities of AVLLMs across three distinct dimensions: Adversarial Attack, Compositional Reasoning, and Modality-specific Dependency. Using our benchmark, we extensively evaluate 16 state-of-the-art AVLLMs. The findings reveal that the majority of existing models fall significantly short of achieving human like comprehension, offering valuable insights for future research directions. To alleviate the limitations in the existing approaches, we further propose a robust, model agnostic calibrated audio-visual preference optimization based training strategy CAVPref, obtaining a gain up to 30.19% across all 9 tasks.

</details>

---

## 47. AURELIA: Test-time Reasoning Distillation in Audio-Visual LLMs

- [ ] AURELIA: Test-time Reasoning Distillation in Audio-Visual LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Chowdhury_AURELIA_Test-time_Reasoning_Distillation_in_Audio-Visual_LLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chowdhury_AURELIA_Test-time_Reasoning_Distillation_in_Audio-Visual_LLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in reasoning optimization have greatly enhanced the performance of large language models (LLMs). However, existing work fails to address the complexities of audio-visual scenarios, underscoring the need for further research. In this paper, we introduce AURELIA, a novel actor-critic based audio-visual (AV) reasoning framework that distils structured, step-by-step reasoning into AVLLMs at test time, improving their ability to process complex multi-modal inputs without additional training or fine-tuning. To further advance AVLLM reasoning skills, we present AVReasonBench, a challenging benchmark comprising 4500 audio-visual questions, each paired with detailed step-by-step reasoning. Our benchmark spans six distinct tasks, including AV-GeoIQ, which evaluates AV reasoning combined with geographical and cultural knowledge. Evaluating 18 AVLLMs on AVReasonBench reveals significant limitations in their multi-modal reasoning capabilities. Using AURELIA, we achieve up to a 100% relative improvement, demonstrating its effectiveness. This performance gain highlights the potential of reasoning-enhanced data generation for advancing AVLLMs in real-world applications.

</details>

---

## 48. GraspCoT: Integrating Physical Property Reasoning for 6-DoF Grasping under Flexible Language Instructions

- [ ] GraspCoT: Integrating Physical Property Reasoning for 6-DoF Grasping under Flexible Language Instructions | https://openaccess.thecvf.com/content/ICCV2025/html/Chu_GraspCoT_Integrating_Physical_Property_Reasoning_for_6-DoF_Grasping_under_Flexible_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Chu_GraspCoT_Integrating_Physical_Property_Reasoning_for_6-DoF_Grasping_under_Flexible_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Flexible instruction-guided 6-DoF grasping is a significant yet challenging task for real-world robotic systems. Existing methods utilize the contextual understanding capabilities of the large language models (LLMs) to establish mappings between expressions and targets, allowing robots to comprehend users' intentions in the instructions. However, the LLM's knowledge about objects' physical properties remains underexplored despite its tight relevance to grasping. In this work, we propose GraspCoT, a 6-DoF grasp detection framework that integrates a Chain-of-Thought (CoT) reasoning mechanism oriented to physical properties, guided by auxiliary question-answering (QA) tasks. Particularly, we design a set of QA templates to enable hierarchical reasoning that includes three stages: target parsing, physical property analysis, and grasp action selection. Moreover, GraspCoT presents a unified multimodal LLM architecture, which encodes multi-view observations of 3D scenes into 3D-aware visual tokens, and then jointly embeds these visual tokens with CoT-derived textual tokens within LLMs to generate grasp pose predictions. Furthermore, we present IntentGrasp, a large-scale benchmark that fills the gap in public datasets for multi-object grasp detection under diverse and indirect verbal commands. Extensive experiments on IntentGrasp demonstrate the superiority of our method, with additional validation in real-world robotic applications confirming its practicality. The code is available at https://github.com/cxmomo/GraspCoT.

</details>

---

## 49. PixTalk: Controlling Photorealistic Image Processing and Editing with Language

- [ ] PixTalk: Controlling Photorealistic Image Processing and Editing with Language | https://openaccess.thecvf.com/content/ICCV2025/html/Conde_PixTalk_Controlling_Photorealistic_Image_Processing_and_Editing_with_Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Conde_PixTalk_Controlling_Photorealistic_Image_Processing_and_Editing_with_Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-guided image generation and editing is emerging as a fundamental problem in computer vision. However, most approaches lack control, and the generated results are far from professional photography quality standards. In this work, we propose the first approach that introduces language and explicit control into the image processing and editing pipeline. PixTalk is a vision-language multi-task image processing model, guided using text instructions. Our method is able to perform over 40 transformations --the most popular techniques in photography--, delivering results as professional photography editing software. Our model can process 12MP images on consumer GPUs in real-time (under 1 second). As part of this effort, we propose a novel dataset and benchmark for new research on multi-modal image processing and editing.

</details>

---

## 50. DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy

- [ ] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy | https://openaccess.thecvf.com/content/ICCV2025/html/Dai_DeRIS_Decoupling_Perception_and_Cognition_for_Enhanced_Referring_Image_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Dai_DeRIS_Decoupling_Perception_and_Cognition_for_Enhanced_Referring_Image_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring Image Segmentation (RIS) is a challenging task that aims to segment objects in an image based on natural language expressions. While prior studies have predominantly concentrated on improving vision-language interactions and achieving fine-grained localization, a systematic analysis of the fundamental bottlenecks in existing RIS frameworks remains underexplored. To bridge this gap, we propose DeRIS , a novel framework that decomposes RIS into two key components: perception and cognition . This modular decomposition facilitates a systematic analysis of the primary bottlenecks impeding RIS performance. Our findings reveal that the predominant limitation lies not in perceptual deficiencies, but in the insufficient multi-modal cognitive capacity of current models. To mitigate this, we propose a Loopback Synergy mechanism, which enhances the synergy between the perception and cognition modules, thereby enabling precise segmentation while simultaneously improving robust image-text comprehension. Additionally, we analyze and introduce a simple non-referent sample conversion data augmentation to address the long-tail distribution issue related to target existence judgement in general scenarios. Notably, DeRIS demonstrates inherent adaptability to both non- and multi-referents scenarios without requiring specialized architectural modifications, enhancing its general applicability. The codes and models are available at https://github.com/Dmmm1997/DeRIS

</details>

---

## 51. GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks

- [ ] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks | https://openaccess.thecvf.com/content/ICCV2025/html/Danish_GEOBench-VLM_Benchmarking_Vision-Language_Models_for_Geospatial_Tasks_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Danish_GEOBench-VLM_Benchmarking_Vision-Language_Models_for_Geospatial_Tasks_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While numerous recent benchmarks focus on evaluating generic Vision-Language Models (VLMs), they do not effectively address the specific challenges of geospatial applications.Generic VLM benchmarks are not designed to handle the complexities of geospatial data, an essential component for applications such as environmental monitoring, urban planning, and disaster management.Key challenges in the geospatial domain include temporal change detection, large-scale object counting, tiny object detection, and understanding relationships between entities in remote sensing imagery.To bridge this gap, we present GEOBench-VLM, a comprehensive benchmark specifically designed to evaluate VLMs on geospatial tasks, including scene understanding, object counting, localization, fine-grained categorization, segmentation, and temporal analysis. Our benchmark features over 10,000 manually verified instructions and spanning diverse visual conditions, object types, and scales.We evaluate several state-of-the-art VLMs to assess performance on geospatial-specific challenges. The results indicate that although existing VLMs demonstrate potential, they face challenges when dealing with geospatial-specific tasks, highlighting the room for further improvements. Notably, the best-performing LLaVa-OneVision achieves only 41.7% accuracy on MCQs, slightly more than GPT-4o, which is approximately double the random guess performance. Our benchmark will be publicly available.

</details>

---

## 52. Training-Free Personalization via Retrieval and Reasoning on Fingerprints

- [ ] Training-Free Personalization via Retrieval and Reasoning on Fingerprints | https://openaccess.thecvf.com/content/ICCV2025/html/Das_Training-Free_Personalization_via_Retrieval_and_Reasoning_on_Fingerprints_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Das_Training-Free_Personalization_via_Retrieval_and_Reasoning_on_Fingerprints_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have lead to major improvements in multimodal reasoning, yet they still struggle to understand user-specific concepts. Existing personalization methods address this limitation butheavily rely on training procedures, that can be either costly or unpleasant to individual users.We depart from existing work, and for the first time explore the training-free setting in the context of personalization. We propose a novel method, Retrieval and Reasoning for Personalization (R2P), leveraging internal knowledge of VLMs. First, we leverage VLMs to extract the concept fingerprint, i.e., key attributes uniquely defining the concept within its semantic class. When a query arrives, the most similar fingerprints are retrieved and scored via chain of thought reasoning. To reduce the risk of hallucinations, the scores are validated through cross-modal verification at the attribute level:in case of a discrepancy between the scores, R2P refines the concept association viapairwise multimodal matching, where the retrieved fingerprints and their images aredirectly compared with the query.We validate R2P on two publicly available benchmarks and a newly introduced dataset, Personal Concepts with Visual Ambiguity (PerVA), for concept identification highlighting challenges in visual ambiguity. R2P consistently outperforms state-of-the-art approaches on various downstream tasks across all benchmarks.

</details>

---

## 53. MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs

- [ ] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Daxberger_MM-Spatial_Exploring_3D_Spatial_Understanding_in_Multimodal_LLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Daxberger_MM-Spatial_Exploring_3D_Spatial_Understanding_in_Multimodal_LLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) excel at 2D visual understanding but remain limited in their ability to reason about 3D space. In this work, we leverage large-scale high-quality 3D scene data with open-set annotations to introduce 1) a novel supervised fine-tuning dataset and 2) a new evaluation benchmark, focused on indoor scenes. Our Cubify Anything VQA (CA-VQA) data covers diverse spatial tasks including spatial relationship prediction, metric size and distance estimation, and 3D grounding. We show that CA-VQA enables us to train MM-Spatial, a strong generalist MLLM that also achieves state-of-the-art performance on 3D spatial understanding benchmarks, including our own. We show how incorporating metric depth and multi-view inputs (provided in CA-VQA) can further improve 3D understanding, and demonstrate that data alone allows our model to achieve depth perception capabilities comparable to dedicated monocular depth estimation models.

</details>

---

## 54. Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images

- [ ] Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images | https://openaccess.thecvf.com/content/ICCV2025/html/Deng_Visual_Chronicles_Using_Multimodal_LLMs_to_Analyze_Massive_Collections_of_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Deng_Visual_Chronicles_Using_Multimodal_LLMs_to_Analyze_Massive_Collections_of_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a system using Multimodal LLMs (MLLMs) to analyze a large database with tens of millions of images captured at different times, with the aim of discovering patterns in temporal changes. Specifically, we aim to capture frequent co-occurring changes ("trends") across a city over a certain period. Unlike previous visual analyses, our analysis answers open-ended queries (e.g., "what are the frequent types of changes in the city?") without any predetermined target subjects or training labels. These properties cast prior learning-based or unsupervised visual analysis tools unsuitable. We identify MLLMs as a novel tool for their open-ended semantic understanding capabilities. Yet, our datasets are four orders of magnitude too large for an MLLM to ingest as context. So we introduce a bottom-up procedure that decomposes the massive visual analysis problem into more tractable sub-problems. We carefully design MLLM-based solutions to each sub-problem. During experiments and ablation studies with our system, we find it significantly outperforms baselines and is able to discover interesting trends from images captured in large cities (e.g., "addition of outdoor dining,", "overpass was painted blue," etc.).

</details>

---

## 55. DiffTell: A High-Quality Dataset for Describing Image Manipulation Changes

- [ ] DiffTell: A High-Quality Dataset for Describing Image Manipulation Changes | https://openaccess.thecvf.com/content/ICCV2025/html/Di_DiffTell_A_High-Quality_Dataset_for_Describing_Image_Manipulation_Changes_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Di_DiffTell_A_High-Quality_Dataset_for_Describing_Image_Manipulation_Changes_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The image difference captioning (IDC) task is to describe the distinctions between two images. However, existing datasets do not offer comprehensive coverage across all image-difference categories. In this work, we introduce a high-quality dataset, DiffTell with various types of image manipulations, including global image alterations, object-level changes, and text manipulations. The data quality is controlled by careful human filtering. Additionally, to scale up the data collection without prohibitive human labor costs, we explore the possibility of automatically filtering for quality control. We demonstrate that both traditional methods and recent multimodal large language models (MLLMs) exhibit performance improvements on the IDC task after training on the DiffTell dataset. Through extensive ablation studies, we provide a detailed analysis of the performance gains attributed to DiffTell. Experiments show DiffTell significantly enhances the availability of resources for IDC research, offering a more comprehensive foundation and benchmark for future investigations.

</details>

---

## 56. MM-IFEngine: Towards Multimodal Instruction Following

- [ ] MM-IFEngine: Towards Multimodal Instruction Following | https://openaccess.thecvf.com/content/ICCV2025/html/Ding_MM-IFEngine_Towards_Multimodal_Instruction_Following_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ding_MM-IFEngine_Towards_Multimodal_Instruction_Following_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Instruction Following (IF) ability measures how well Multi-modal Large Language Models (MLLMs) understand exactly what users are telling them and doing it right.Existing multimodal instruction following training data is scarce, the benchmarks are simple with atomic instructions, and the evaluation strategies are imprecise for tasks demanding exact output constraints.To address this, we present MM-IFEngine, an effective pipeline to generate high-quality image-instruction pairs.Our MM-IFEngine pipeline yields large-scale, diverse, and high-quality training data MM-IFInstruct-23k, which is suitable for Supervised Fine-Tuning (SFT) and extended as MM-IFDPO-23k for Direct Preference Optimization (DPO).We further introduce MM-IFEval, a challenging and diverse multi-modal instruction-following benchmark that includes (1) both textual constraints for output responses and visual constraints tied to the input images, and (2) a comprehensive evaluation pipeline incorporating rule-based assessment and LLM-as-a-Judge evaluation.We conduct SFT and DPO experiments and demonstrate that fine-tuning MLLMs on MM-IFInstruct-23k and MM-IFDPO-23k achieve notable gains on various IF benchmarks, such as MM-IFEval (+11.8%), MIA (+7.7%), and IFEval (+10.5%).

</details>

---

## 57. EVEv2: Improved Baselines for Encoder-Free Vision-Language Models

- [ ] EVEv2: Improved Baselines for Encoder-Free Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Diao_EVEv2_Improved_Baselines_for_Encoder-Free_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Diao_EVEv2_Improved_Baselines_for_Encoder-Free_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing encoder-free vision-language models (VLMs) are rapidly narrowing the performance gap with their encoder-based counterparts, highlighting the promising potential for unified multimodal systems with structural simplicity and efficient deployment. We systematically clarify the performance gap between VLMs using pre-trained vision encoders, discrete tokenizers, and minimalist visual layers from scratch, deeply excavating the under-examined characteristics of encoder-free VLMs. We develop efficient strategies for encoder-free VLMs that rival mainstream encoder-based ones. After an in-depth investigation, we launch EVEv2.0, a new and improved family of encoder-free VLMs. We show that: (i) Properly decomposing and hierarchically associating vision and language within a unified model reduces interference between modalities. (ii) A well-designed training strategy enables effective optimization for encoder-free VLMs. Through extensive evaluation, our EVEv2.0 represents a thorough study for developing a decoder-only architecture across modalities, demonstrating superior data efficiency and strong vision-reasoning capability. Code is publicly available at: https://github.com/baaivision/EVE.

</details>

---

## 58. Confound from All Sides, Distill with Resilience: Multi-Objective Adversarial Paths to Zero-Shot Robustness

- [ ] Confound from All Sides, Distill with Resilience: Multi-Objective Adversarial Paths to Zero-Shot Robustness | https://openaccess.thecvf.com/content/ICCV2025/html/Dong_Confound_from_All_Sides_Distill_with_Resilience_Multi-Objective_Adversarial_Paths_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Dong_Confound_from_All_Sides_Distill_with_Resilience_Multi-Objective_Adversarial_Paths_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adversarially robust knowledge distillation transfers the robustness of a large-scale teacher model to a lightweight student while preserving natural performance. However, foundation Vision-Language Models (VLMs) also demand the transfer of zero-shot inference capabilities. We find that standard robust distillation using untargeted adversarial examples fails to transfer out-of-distribution (zero-shot) robustness, as these adversaries primarily push inputs away from their original distribution, exploring a limited portion of the teacher's decision space and missing more diverse failure modes. A natural solution is to generate multiple targeted adversaries that traverse diverse paths across decision boundaries. Thus, these adversaries probe a broader region of the teacher's decision surface. However, naive targeted adversary optimization often converges to local optima within a single category's decision region, limiting the diversity. To address this, we propose a Multi-Objective Optimization (MOO)-based adversarial distillation framework that transfers robustness from large VLMs to lightweight ones by exploiting adversaries with two main objectives: misclassification and category-level adversarial diversity. Theoretically, we show that optimizing for diversity mitigates adversarial collapse into local optima, ensuring adversaries span multiple decision regions and capture the teacher's generalizable robust features. Extensive experiments demonstrate the superiority of our method over state-of-the-art adversarial learning across diverse scenarios.

</details>

---

## 59. INTER: Mitigating Hallucination in Large Vision-Language Models by Interaction Guidance Sampling

- [ ] INTER: Mitigating Hallucination in Large Vision-Language Models by Interaction Guidance Sampling | https://openaccess.thecvf.com/content/ICCV2025/html/Dong_INTER_Mitigating_Hallucination_in_Large_Vision-Language_Models_by_Interaction_Guidance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Dong_INTER_Mitigating_Hallucination_in_Large_Vision-Language_Models_by_Interaction_Guidance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucinations in large vision-language models (LVLMs) pose significant challenges for real-world applications, as LVLMs may generate responses that appear plausible yet remain inconsistent with the associated visual content. This issue rarely occurs in human cognition. We argue that this discrepancy arises from humans' ability to effectively leverage multimodal interaction information in data samples. Specifically, humans typically first gather multimodal information, analyze the interactions across modalities for understanding, and then express their understanding through language. Motivated by this observation, we conduct extensive experiments on popular LVLMs and obtained insights that surprisingly reveal human-like, though less pronounced, cognitive behavior of LVLMs on multimodal samples. Building on these findings, we further propose INTER: Interaction Guidance Sampling, a novel training-free algorithm that mitigate hallucinations without requiring additional data. Specifically, INTER explicitly guides LVLMs to effectively reapply their understanding of multimodal interaction information when generating responses, thereby reducing potential hallucinations. On six benchmarks including VQA and image captioning tasks, INTER achieves an average improvement of up to 3.4% on five LVLMs compared to the state-of-the-art decoding strategy. The codes are released on \href https://github.com/xxxxx313/INTER  Github .

</details>

---

## 60. LLM-assisted Entropy-based Adaptive Distillation for Unsupervised Fine-grained Visual Representation Learning

- [ ] LLM-assisted Entropy-based Adaptive Distillation for Unsupervised Fine-grained Visual Representation Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Dong_LLM-assisted_Entropy-based_Adaptive_Distillation_for_Unsupervised_Fine-grained_Visual_Representation_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Dong_LLM-assisted_Entropy-based_Adaptive_Distillation_for_Unsupervised_Fine-grained_Visual_Representation_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unsupervised Fine-grained Visual Represent Learning (FVRL) aims to learn discriminative features to distinguish subtle differences among visually similar categories without using labeled fine-grained data. Existing works, which typically learn representation from target data, often struggle to capture subtle inter-class variations due to the limited prior fine-grained knowledge. To alleviate it, this paper proposes LLM-assisted Entropy-based Adaptive Distillation (LEAD), a novel unsupervised FVRL framework that selectively distills fine-grained knowledge from a powerful teacher model built upon pre-trained models. Specifically, we first harness the powerful reasoning capabilities of Large Language Models (LLMs) to generate contextual knowledge of fine-grained category-aware descriptions, enriching semantic priors in the teacher model. These descriptions are then used to form a prototype-driven fine-grained classifier, which acts as an assistant to generate rich knowledge with a frozen vision-language model. Besides, to achieve effective knowledge transfer, we further introduce an entropy-based adaptive mechanism, which dynamically adjusts the distillation strength based on the information entropy to identify and prioritize valuable knowledge. Extensive experimental results on three fine-grained datasets demonstrate the effectiveness and efficiency of our proposed LEAD for unsupervised FVRL. Our source code is available at https://anonymous.4open.science/r/EAD-FFAB.

</details>

---

## 61. Robustifying Zero-Shot Vision Language Models by Subspaces Alignment

- [ ] Robustifying Zero-Shot Vision Language Models by Subspaces Alignment | https://openaccess.thecvf.com/content/ICCV2025/html/Dong_Robustifying_Zero-Shot_Vision_Language_Models_by_Subspaces_Alignment_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Dong_Robustifying_Zero-Shot_Vision_Language_Models_by_Subspaces_Alignment_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) enjoy strong zero-shot performance but are vulnerable to adversarial attacks posing security risks. Adversarially robust fine-tuning enhances zero-shot robustness on new datasets while preserving the natural performance of pre-trained VLMs. However, prior methods use sample-wise adversarial fine-tuning, neglecting the underlying second-order statistics that represent entire groups of samples. This leads to a feature-level discrepancy between clean and adversarial samples of their augmented variants. Thus, we propose to represent groups of samples as subspaces to capture distributions and turn the traditional sample-wise adversarial fine-tuning into its distributional counterpart. For each image, we build distributions from (i) a clean sample with its augmentations and (ii) their adversarial counterparts. For text, we build distributions from (iii) a clean prompt and its synonymous prompts and (iv) their adversarial counterparts. We then perform alignment between image and text subspaces, and "adversarial" subspaces are also aligned toward "clean" subspaces. Thus, all samples underlying these distributions (think infinite number) also get aligned, leading to generalizable robustness. Evaluations on 15 datasets are provided.

</details>

---

## 62. Teaching VLMs to Localize Specific Objects from In-context Examples

- [ ] Teaching VLMs to Localize Specific Objects from In-context Examples | https://openaccess.thecvf.com/content/ICCV2025/html/Doveh_Teaching_VLMs_to_Localize_Specific_Objects_from_In-context_Examples_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Doveh_Teaching_VLMs_to_Localize_Specific_Objects_from_In-context_Examples_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have shown remarkable capabilities across diverse visual tasks, including image recognition, video understanding, and Visual Question Answering (VQA) when explicitly trained for these tasks. Despite these advances, we find that present-day VLMs (including the proprietary GPT-4o) lack a fundamental cognitive ability: learning to localize specific objects in a scene by taking into account the context.In this work, we focus on the task of few-shot personalized localization, where a model is given a small set of annotated images (in-context examples) -- each with a category label and bounding box -- and is tasked with localizing the same object type in a query image. Personalized localization can be particularly important in cases of ambiguity of several related objects that can respond to a text or an object that is hard to describe with words.To provoke personalized localization abilities in models, we present a data-centric solution that fine-tunes them using carefully curated data from video object tracking datasets. By leveraging sequences of frames tracking the same object across multiple shots, we simulate instruction-tuning dialogues that promote context awareness. To reinforce this, we introduce a novel regularization technique that replaces object labels with pseudo-names, ensuring the model relies on visual context rather than prior knowledge. Our method significantly enhances few-shot localization performance of recent VLMs ranging from 7B to 72B in size, without sacrificing generalization, as demonstrated on several benchmarks tailored towards evaluating personalized localization abilities. This work is the first to explore and benchmark personalized few-shot localization for VLMs -- exposing critical weaknesses in present-day VLMs, and lays a foundation for future research in context-driven vision-language applications.

</details>

---

## 63. From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning

- [ ] From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Du_From_Easy_to_Hard_The_MIR_Benchmark_for_Progressive_Interleaved_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Du_From_Easy_to_Hard_The_MIR_Benchmark_for_Progressive_Interleaved_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-image Interleaved Reasoning aims to improve Multimodal Large Language Models' (MLLMs) ability to jointly comprehend and reason across multiple images and their associated textual contexts, introducing unique challenges beyond single-image or non-interleaved multi-image tasks.While current multi-image benchmarks overlook interleaved textual contexts and neglect distinct relationships between individual images and their associated texts, enabling models to reason over multi-image interleaved data may significantly enhance their comprehension of complex scenes and better capture cross-modal correlations.To bridge this gap, we introduce a novel benchmark MIR, requiring joint reasoning over multiple images accompanied by interleaved textual contexts to accurately associate image regions with corresponding texts and logically connect information across images.To enhance MLLMs' ability to comprehend multi-image interleaved data, we introduce reasoning steps for each instance within the benchmark and propose a stage-wise curriculum learning strategy. This strategy follows an "easy to hard" approach, progressively guiding models from simple to complex scenarios, thereby enhancing their ability to handle challenging tasks.Extensive experiments benchmarking multiple MLLMs demonstrate that our method significantly enhances models' reasoning performance on MIR and other established benchmarks, highlighting the challenges current MLLMs face with multi-image interleaved reasoning.We believe that MIR will encourage further research into multi-image interleaved reasoning, facilitating advancements in MLLMs' capability to handle complex inter-modal tasks.

</details>

---

## 64. DIH-CLIP: Unleashing the Diversity of Multi-Head Self-Attention for Training-Free Open-Vocabulary Semantic Segmentation

- [ ] DIH-CLIP: Unleashing the Diversity of Multi-Head Self-Attention for Training-Free Open-Vocabulary Semantic Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Duan_DIH-CLIP_Unleashing_the_Diversity_of_Multi-Head_Self-Attention_for_Training-Free_Open-Vocabulary_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Duan_DIH-CLIP_Unleashing_the_Diversity_of_Multi-Head_Self-Attention_for_Training-Free_Open-Vocabulary_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent Training-Free Open-Vocabulary Semantic Segmentation (TF-OVSS) leverages a pre-training vision-language model to segment images from open-set visual concepts without training and fine-tuning. The key of TF-OVSS is to improve the local spatial representation of CLIP by leveraging self-correlation maps, thus preserving its zero-sample capability and achieving open understanding. However, most TF-OVSS methods utilize the Multi-Head Self-Attention (MHSA) mechanism to generate self-correlation maps, neglecting the diversity among multiple heads. In this paper, we explore the diversity of MHSA, revealing that the contributions of single-head attention to the final results are varied and redundant. To address this issue, we introduce DIH-CLIP, a training-free CLIP model for open-vocabulary semantic segmentation. Specifically, we propose a Selective Head Attention (SHA) to replace the traditional MHSA in CLIP, which contains two key designs: (1) evaluating the diversity of multi-head attention via calculating information entropy scores of per head attention map and removing the redundant attention head with threshold; (2) transferring the local representation of single-head attention to the global CLIP feature to enhance the local spatial representation capability of CLIP. Furthermore, we embed SHA into the middle layers of CLIP to extract the plentiful details. Experiments on six benchmark datasets demonstrate the effectiveness of DIH-CLIP.

</details>

---

## 65. TruthPrInt: Mitigating Large Vision-Language Models Object Hallucination Via Latent Truthful-Guided Pre-Intervention

- [ ] TruthPrInt: Mitigating Large Vision-Language Models Object Hallucination Via Latent Truthful-Guided Pre-Intervention | https://openaccess.thecvf.com/content/ICCV2025/html/Duan_TruthPrInt_Mitigating_Large_Vision-Language_Models_Object_Hallucination_Via_Latent_Truthful-Guided_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Duan_TruthPrInt_Mitigating_Large_Vision-Language_Models_Object_Hallucination_Via_Latent_Truthful-Guided_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Object Hallucination (OH) has been acknowledged as one of the major trustworthy challenges in Large Vision-Language Models (LVLMs). Recent advancements in Large Language Models (LLMs) indicate that internal states, such as hidden states, encode the "overall truthfulness" of generated responses. However, it remains under-explored how internal states in LVLMs function and whether they could serve as "per-token" hallucination indicators, which is essential for mitigating OH. In this paper, we first conduct an in-depth exploration of LVLM internal states in relation to OH issues and discover that (1) LVLM internal states are high-specificity per-token indicators of hallucination behaviors. Moreover, (2) different LVLMs encode universal patterns of hallucinations in common latent subspaces, indicating that there exist "generic truthful directions" shared by various LVLMs. Based on these discoveries, we propose Truthful-Guided Pre-Intervention (TruthPrInt) that first learns the truthful direction of LVLM decoding and then applies truthful-guided inference-time intervention during LVLM decoding. We further propose ComnHallu to enhance both cross-LVLM and cross-data hallucination detection transferability by constructing and aligning hallucination latent subspaces. We evaluate TruthPrInt in extensive experimental settings, including in-domain and out-of-domain scenarios, over popular LVLMs and OH benchmarks. Experimental results indicate that TruthPrInt significantly outperforms state-of-the-art methods.

</details>

---

## 66. Discovering Divergent Representations between Text-to-Image Models

- [ ] Discovering Divergent Representations between Text-to-Image Models | https://openaccess.thecvf.com/content/ICCV2025/html/Dunlap_Discovering_Divergent_Representations_between_Text-to-Image_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Dunlap_Discovering_Divergent_Representations_between_Text-to-Image_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we investigate when and how visual representations learned by two different generative models diverge from each other. Specifically, given two text-to-image models, our goal is to discover visual attributes that appear in images generated by one model but not the other, along with the types of prompts that trigger these attribute differences. For example, 'flames' might appear in one model's outputs when given prompts expressing strong emotions, while the other model does not produce this attribute given the same prompts. We introduce CompCon (Comparing Concepts), an evolutionary search algorithm that discovers visual attributes more prevalent in one model's output than the other, and uncovers the prompt concepts linked to these visual differences. To evaluate CompCon's ability to find diverging representations, we create an automated data generation pipeline to produce ID^2, a dataset of 60 input-dependent differences, and compare our approach to several LLM- and VLM-powered baselines. Finally, we use CompCon to compare popular text-to-image models, finding divergent representations such as how PixArt depicts prompts mentioning loneliness with wet streets and Stable Diffusion 3.5 depicts African American people in media professions.

</details>

---

## 67. Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration

- [ ] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration | https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent works on accelerating Vision-Language Models achieve strong performance across a variety of vision-language tasks despite highly compressing visual information. In this work, we examine the popular acceleration approach of early pruning of visual tokens inside the language model. Surprisingly, we find that while strong performance is maintained across many tasks, it exhibits drastically different behavior for a subset of vision-centric tasks such as localization. Upon further investigation, we uncover a core issue with the acceleration approach where most tokens towards the top of the image are pruned away. Yet, on many benchmarks aiming to evaluate vision-centric capabilities, strong performance persists with the flawed pruning strategy, highlighting these benchmarks' limited ability to assess fine-grained visual capabilities. Based on these findings, we propose FEATHER (Fast and Effective Acceleration wiTH Ensemble cRiteria), a straightforward approach that resolves the discovered early-layer pruning issue and further enhances the preservation of relevant tokens via multistage pruning with early uniform sampling to ensure broad image coverage. With comparable computational savings, we find that FEATHER achieves more than 5x performance improvement on the vision-centric localization benchmarks compared to the original acceleration approach.

</details>

---

## 68. Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding

- [ ] Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Embodied_VideoAgent_Persistent_Memory_from_Egocentric_Videos_and_Embodied_Sensors_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Embodied_VideoAgent_Persistent_Memory_from_Egocentric_Videos_and_Embodied_Sensors_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper investigates the problem of understanding dynamic 3D scenes from egocentric observations, a key challenge in robotics and embodied AI. Unlike prior studies that explored this as long-form video understanding and utilized egocentric video only, we instead propose an LLM-based agent, Embodied VideoAgent, which constructs scene memory from both egocentric video and embodied sensory inputs (e.g. depth and pose sensing). We further introduce a VLM-based approach to automatically update the memory when actions or activities over objects are perceived. Embodied VideoAgent attains significant advantages over counterparts in challenging reasoning and planning tasks in 3D scenes, achieving gains of 6.5% on Ego4D-VQ3D, 2.6% on OpenEQA, and 15.3% on EnvQA. We have also demonstrated its potential in various embodied AI tasks including generating embodied interactions and perception for robot manipulation. The code and demo will be made public.

</details>

---

## 69. Semantic Equitable Clustering: A Simple and Effective Strategy for Clustering Vision Tokens

- [ ] Semantic Equitable Clustering: A Simple and Effective Strategy for Clustering Vision Tokens | https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Semantic_Equitable_Clustering_A_Simple_and_Effective_Strategy_for_Clustering_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Semantic_Equitable_Clustering_A_Simple_and_Effective_Strategy_for_Clustering_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Vision Transformer (ViT) has gained prominence for its superior relational modeling prowess. However, its global attention mechanism's quadratic complexity poses substantial computational burdens. A common remedy spatially groups tokens for self-attention, reducing computational requirements. Nonetheless, this strategy neglects semantic information in tokens, possibly scattering semantically-linked tokens across distinct groups, thus compromising the efficacy of self-attention intended for modeling inter-token dependencies. Motivated by these insights, we introduce a fast and balanced clustering method, named Semantic Equitable Clustering (SEC). SEC clusters tokens based on their global semantic relevance in an efficient, straightforward manner. In contrast to traditional clustering methods requiring multiple iterations, our method achieves token clustering in a single pass. Additionally, SEC regulates the number of tokens per cluster, ensuring a balanced distribution for effective parallel processing on current computational platforms without necessitating further optimization. Capitalizing on SEC, we propose a versatile vision backbone, SECViT. Comprehensive experiments in image classification, object detection, instance segmentation, and semantic segmentation validate to the effectiveness of SECViT. Remarkably, SECViT attains an impressive 84.3% image classification accuracy with only 27M parameters and 4.6G FLOPs, without the need for for additional supervision or data. Moreover, SEC can be conveniently and swiftly applied to multimodal large language models (MLLM), such as LLaVA, to serve as a vision language connector, effectively accelerating the model's efficiency while maintaining unchanged or better performance.

</details>

---

## 70. Test-Time Retrieval-Augmented Adaptation for Vision-Language Models

- [ ] Test-Time Retrieval-Augmented Adaptation for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Test-Time_Retrieval-Augmented_Adaptation_for_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Test-Time_Retrieval-Augmented_Adaptation_for_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have shown promise in test-time adaptation tasks due to their remarkable capabilities in understanding and reasoning about visual content through natural language descriptions. However, training VLMs typically demands substantial computational resources, and they often struggle to adapt efficiently to new domains or tasks. Additionally, dynamically estimating the test distribution from streaming data at test time remains a significant challenge. In this work, we propose a novel test-time retrieval-augmented adaptation (TT-RAA) method that enables VLMs to maintain high performance across diverse visual recognition tasks without the need for task-specific training or large computational overhead. During inference, TT-RAA employs a streaming mixture of Gaussian database (SMGD) to continuously estimate test distributions, requiring minimal storage. Then, TT-RAA retrieves the most relevant information from the SMGD, enhancing the original VLM outputs. A key limitation of CLIP-based VLMs is their inter-modal vision-language optimization, which does not optimize vision-space similarity, leading to larger intra-modal variance. To address this, we propose a multimodal retrieval augmentation module that transforms the SMGD into a unified multimodal space, enabling retrieval that aligns both vision and language modalities. Extensive experiments across both cross-domain and out-of-distribution benchmarks comprising fourteen datasets demonstrate TT-RAA's superior performance compared to state-of-the-art methods. Ablation studies and hyperparameter analyses further validate the effectiveness of the proposed modules. The source code of our work is available at https://github.com/xinqi-fan/TT-RAA.

</details>

---

## 71. Can Knowledge be Transferred from Unimodal to Multimodal? Investigating the Transitivity of Multimodal Knowledge Editing

- [ ] Can Knowledge be Transferred from Unimodal to Multimodal? Investigating the Transitivity of Multimodal Knowledge Editing | https://openaccess.thecvf.com/content/ICCV2025/html/Fang_Can_Knowledge_be_Transferred_from_Unimodal_to_Multimodal_Investigating_the_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fang_Can_Knowledge_be_Transferred_from_Unimodal_to_Multimodal_Investigating_the_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) contain a substantial amount of factual knowledge, which may become outdated or inaccurate over time. Consequently, various knowledge editing techniques have been proposed to update the knowledge encoded within these models. Previous approaches maintain modality consistency during both the editing and testing phases. However, in practical applications, it is desirable for knowledge to be transferable across different modalities, which can enhance the robustness of knowledge editing and potentially allow for cost-effective editing of multimodal knowledge using textual information. To address this, we introduce the concept of Transitivity of Multimodal Knowledge Editing (TMKE) and design corresponding evaluation criteria. Subsequently, we construct a corresponding TMKE Benchmark through an automated pipeline. We evaluate three MLLMs and five knowledge editing methods, uncovering limitations in the current models and methods concerning transitivity. Additionally, we analyze the intrinsic representations of the model during the editing process based on Knowledge Neurons to interpret the experimental phenomena.

</details>

---

## 72. Creation-MMBench: Assessing Context-Aware Creative Intelligence in MLLMs

- [ ] Creation-MMBench: Assessing Context-Aware Creative Intelligence in MLLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Fang_Creation-MMBench_Assessing_Context-Aware_Creative_Intelligence_in_MLLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fang_Creation-MMBench_Assessing_Context-Aware_Creative_Intelligence_in_MLLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Creativity is a fundamental aspect of intelligence, involving the ability to generate novel and appropriate solutions across diverse contexts. While Large Language Models (LLMs) have been extensively evaluated for their creative capabilities, the assessment of Multimodal Large Language Models (MLLMs) in this domain remains largely unexplored. To address this gap, we introduce Creation-MMBench, a multimodal benchmark specifically designed to evaluate the creative capabilities of MLLMs in real-world, image-based tasks. The benchmark comprises 765 test cases spanning 51 fine-grained tasks.To ensure rigorous evaluation, we define instance-specific evaluation criteria for each test case, guiding the assessment of both general response quality and factual consistency with visual inputs. Experimental results reveal that current open-source MLLMs significantly underperform compared to proprietary models in creative tasks. Furthermore, our analysis demonstrates that visual fine-tuning can negatively impact the base LLM's creative abilities.Creation-MMBench provides valuable insights for advancing MLLM creativity and establishes a foundation for future improvements in multimodal generative intelligence. Full data and evaluation code will be released soon.

</details>

---

## 73. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models

- [ ] One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models | https://openaccess.thecvf.com/content/ICCV2025/html/Fang_One_Perturbation_is_Enough_On_Generating_Universal_Adversarial_Perturbations_against_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fang_One_Perturbation_is_Enough_On_Generating_Universal_Adversarial_Perturbations_against_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the learned multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these attacks are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also susceptible to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC). In light that the pivotal multimodal alignment in VLP models is achieved via contrastive learning, we devise to turn this powerful weapon against VLP models themselves. I.e., we employ a malicious version of contrastive learning to train the proposed generator using our carefully crafted positive and negative image-text pairs. Once training is complete, the generator is able to produce universal perturbations that can essentially destroy the established alignment relationship in VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus fundamentally enhancing attack performance across various victim models and V+L tasks.

</details>

---

## 74. PUMA: Empowering Unified MLLM with Multi-granular Visual Generation

- [ ] PUMA: Empowering Unified MLLM with Multi-granular Visual Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Fang_PUMA_Empowering_Unified_MLLM_with_Multi-granular_Visual_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fang_PUMA_Empowering_Unified_MLLM_with_Multi-granular_Visual_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal foundation models have yielded significant progress in vision-language understanding. Initial attempts have also explored the potential of multimodal large language models for visual content generation. However, existing approaches face a trade-off between generation diversity and controllability, struggling to meet the varying granularity demands of different image generation tasks within a unified MLLM framework. In this work, we propose PUMA, emPowering Unified MLLM with Multi-grAnular visual generation, a novel paradigm that tackles the diversity-controllability trade-off. PUMA achieves this by unifying multi-granular visual features as both inputs and outputs of MLLMs, thus effectively meeting the distinct granularity needs for diverse generation and precise manipulation within a single framework. Following multimodal pretraining and instruction tuning, PUMA demonstrates remarkable capabilities in a wide range of multimodal tasks, including image understanding, diverse text-to-image generation, editing, inpainting, colorization, and conditional generation. This work marks a significant stride towards realizing truly unified MLLMs capable of seamlessly adapting to the diverse granularity demands and task requirements inherent in various visual tasks. The code and model will be released upon acceptance.

</details>

---

## 75. HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics

- [ ] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics | https://openaccess.thecvf.com/content/ICCV2025/html/Faure_HERMES_temporal-coHERent_long-forM_understanding_with_Episodes_and_Semantics_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Faure_HERMES_temporal-coHERent_long-forM_understanding_with_Episodes_and_Semantics_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-form video understanding presents unique challenges that extend beyond traditional short-video analysis approaches, particularly in capturing long-range dependencies, processing redundant information efficiently, and extracting high-level semantic concepts. To address these challenges, we propose a novel approach that more accurately reflects human cognition. This paper introduces HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics, featuring two versatile modules that can enhance existing video-language models or operate as a standalone system. Our Episodic COmpressor (ECO) efficiently aggregates representations from micro to semi-macro levels, reducing computational overhead while preserving temporal dependencies. Our Semantics ReTRiever (SeTR) enriches these representations with semantic information by focusing on broader context, dramatically reducing feature dimensionality while preserving relevant macro-level information. We demonstrate that these modules can be seamlessly integrated into existing SOTA models, consistently improving their performance while reducing inference latency by up to 43% and memory usage by 46%. As a standalone system, HERMES achieves state-of-the-art performance across multiple long-video understanding benchmarks in both zero-shot and fully-supervised settings. Our project page and code can be found at https://joslefaure.github.io/assets/html/hermes.html.

</details>

---

## 76. ATCTrack: Aligning Target-Context Cues with Dynamic Target States for Robust Vision-Language Tracking

- [ ] ATCTrack: Aligning Target-Context Cues with Dynamic Target States for Robust Vision-Language Tracking | https://openaccess.thecvf.com/content/ICCV2025/html/Feng_ATCTrack_Aligning_Target-Context_Cues_with_Dynamic_Target_States_for_Robust_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Feng_ATCTrack_Aligning_Target-Context_Cues_with_Dynamic_Target_States_for_Robust_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language tracking aims to locate the target object in the video sequence using a template patch and a language description provided in the initial frame. To achieve robust tracking, especially in complex long-term scenarios that reflect real-world conditions as recently highlighted by MGIT, it is essential not only to characterize the target features but also to utilize the context features related to the target. However, the visual and textual target-context cues derived from the initial prompts generally align only with the initial target state. Due to their dynamic nature, target states are constantly changing, particularly in complex long-term sequences. It is intractable for these cues to continuously guide Vision-Language Trackers (VLTs). Furthermore, for the text prompts with diverse expressions, our experiments reveal that existing VLTs struggle to discern which words pertain to the target or the context, complicating the utilization of textual cues. In this work, we present a novel tracker named ATCTrack, which can obtain multimodal cues Aligned with the dynamic target states through comprehensive Target-Context feature modeling, thereby achieving robust tracking. Specifically, (1) for the visual modality, we propose an effective temporal visual target-context modeling approach that provides the tracker with timely visual cues. (2) For the textual modality, we achieve precise target words identification solely based on textual content, and design an innovative context words calibration method to adaptively utilize auxiliary context words. (3) We conduct extensive experiments on mainstream benchmarks and ATCTrack achieves a new SOTA performance. The code and models will be released at: https://github.com/XiaokunFeng/ATCTrack.

</details>

---

## 77. Partially Matching Submap Helps: Uncertainty Modeling and Propagation for Text to Point Cloud Localization

- [ ] Partially Matching Submap Helps: Uncertainty Modeling and Propagation for Text to Point Cloud Localization | https://openaccess.thecvf.com/content/ICCV2025/html/Feng_Partially_Matching_Submap_Helps_Uncertainty_Modeling_and_Propagation_for_Text_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Feng_Partially_Matching_Submap_Helps_Uncertainty_Modeling_and_Propagation_for_Text_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text to point cloud cross-modal localization is a crucial vision-language task for future human-robot collaboration. Existing coarse-to-fine frameworks assume that each query text precisely corresponds to the center area of a submap, limiting their applicability in real-world scenarios. This work redefines the task under a more realistic assumption, relaxing the one-to-one retrieval constraint by allowing partially matching query text and submap pairs. To address this challenge, we augment datasets with partially matching submaps and introduce an uncertainty-aware framework. Specifically, we model cross-modal ambiguity in fine-grained location regression by integrating uncertainty scores, represented as 2D Gaussian distributions, to mitigate the impact of challenging samples. Additionally, we propose an uncertainty-aware similarity metric that enhances similarity assessment between query text and submaps by propagating uncertainty into coarse place recognition, enabling the model to learn discriminative features, effectively handle partially matching samples and improve task synergy. Extensive experiments on KITTI360Pose and CityRefer demonstrate that our method achieves state-of-the-art performance across both stages. Our code is available at https://github.com/Afoolbird/PMSH

</details>

---

## 78. UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence

- [ ] UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence | https://openaccess.thecvf.com/content/ICCV2025/html/Feng_UrbanLLaVA_A_Multi-modal_Large_Language_Model_for_Urban_Intelligence_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Feng_UrbanLLaVA_A_Multi-modal_Large_Language_Model_for_Urban_Intelligence_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Urban research involves a wide range of scenarios and tasks that require the understanding of multi-modal data, such as structured geospatial data, trajectory data, satellite image data, and street view image data. Current methods often focus on specific data types and lack a unified framework in urban field for processing them comprehensively. The recent success of multi-modal large language models (MLLMs) presents a promising opportunity to overcome this limitation. In this paper, we introduce UrbanLLaVA, a multi-modal large language model designed to process these four types of data simultaneously and achieve strong performance across diverse urban tasks compared with general MLLMs. In UrbanLLaVA, we first curate a diverse urban instruction dataset encompassing both single-modal and cross-modal urban data, spanning from location view to global view of urban environment. Additionally, we design an effective multi-stage training pipeline to ensure the training stability and compatibility across various urban tasks. We also extend existing benchmark for urban research to assess the performance of MLLMs across a wide range of urban tasks. Experimental results from three cities demonstrate that UrbanLLaVA outperforms open source and commercial MLLMs in both single-modal tasks and complex cross-modal tasks and shows robust generalization abilities across cities. UrbanLLaVA sheds lights for building the unified foundation model with powerful perception and reasoning abilities for general urban intelligence. Source codes and data are openly accessible to the research community via https://github.com/tsinghua-fib-lab/UrbanLLaVA.

</details>

---

## 79. FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models

- [ ] FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Fu_FrameFusion_Combining_Similarity_and_Importance_for_Video_Token_Reduction_on_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fu_FrameFusion_Combining_Similarity_and_Importance_for_Video_Token_Reduction_on_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The increasing demand to process long and high-resolution videos significantly burdens Large Vision-Language Models (LVLMs) due to the enormous number of visual tokens. Existing token reduction methods primarily prune tokens based on importance metrics, such as cumulative attention scores. However, even important tokens may exhibit high redundancy caused by similarity among adjacent video frames and repetitive visual elements. To address this limitation, we propose FrameFusion, a novel token reduction approach integrating similarity-based merging with importance-based pruning. We conduct a thorough study on token similarity characteristics, revealing three key insights: (1) spatially corresponding visual tokens between adjacent frames have higher cosine similarities compared to other token pairs; (2) high token similarities prominently decrease in deeper model layers; and (3) token similarity rankings are highly consistent across different layers. Guided by these observations, FrameFusion computes token similarities exclusively between corresponding visual tokens from adjacent frames, applies token merging at initial successive layers followed by pruning in deeper layers, and adopts a cascaded merging strategy to further enhance efficiency. We evaluate FrameFusion comprehensively across six diverse LVLMs, ranging from 2B to 72B parameters, using five video benchmarks encompassing video retrieval, question-answering, and spatial-temporal understanding tasks. Experiments show that FrameFusion reduces visual tokens by 70%, achieving 1.6-3.6x end-to-end speedups, with an average performance impact of less than 3%. Our code is available at: https://github.com/thu-nics/FrameFusion.

</details>

---

## 80. ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation

- [ ] ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Fu_ORION_A_Holistic_End-to-End_Autonomous_Driving_Framework_by_Vision-Language_Instructed_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Fu_ORION_A_Holistic_End-to-End_Autonomous_Driving_Framework_by_Vision-Language_Instructed_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

End-to-end (E2E) autonomous driving methods still struggle to make correct decisions in interactive closed-loop evaluation due to limited causal reasoning capability. Current methods attempt to leverage the powerful understanding and reasoning abilities of Vision-Language Models (VLMs) to resolve this dilemma. However, the problem is still open that few VLMs for E2E methods perform well in the closed-loop evaluation due to the gap between the semantic reasoning space and the purely numerical trajectory output in the action space. To tackle this issue, we propose ORION, a holistic E2E autonomous driving framework by vision-language instructed action generation.ORION uniquely combines a QT-Former to aggregate long-term history context, a Large Language Model (LLM) for driving scenario reasoning, and a generative planner for precision trajectory prediction. ORION further aligns the reasoning space and the action space to implement a unified E2E optimization for both visual question-answering (VQA) and planning tasks. Our method achieves an impressive closed-loop performance of 77.47 Driving Score (DS) and 54.62% Success Rate (SR) on the challenge Bench2Drive datasets, which outperforms state-of-the-art (SOTA) methods by a large margin of 14.28 DS and 28.08% SR.

</details>

---

## 81. 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation

- [ ] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation | https://openaccess.thecvf.com/content/ICCV2025/html/Gao_3D_Gaussian_Map_with_Open-Set_Semantic_Grouping_for_Vision-Language_Navigation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_3D_Gaussian_Map_with_Open-Set_Semantic_Grouping_for_Vision-Language_Navigation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language navigation (VLN) requires an agent to traverse complex 3D environments based on natural language instructions, necessitating a thorough scene understanding. While existing works equip agents with various scene representations to enhance spatial awareness, they often neglect the complex 3D geometry and rich semantics in VLN scenarios, limiting the ability to generalize across diverse and unseen environments. To address these challenges, this work proposes a 3D Gaussian Map that represents the environment as a set of differentiable 3D Gaussians and accordingly develops a navigation strategy for VLN. Specifically, Egocentric Scene Map is constructed online by initializing 3D Gaussians from sparse pseudo-lidar point clouds, providing informative geometric priors for scene understanding. Each Gaussian primitive is further enriched through Open-Set Semantic Grouping operation, which groups 3D Gaussians based on their membership in object instances or stuff categories within the open world, resulting in a unified 3D Gaussian Map. Building on this map, Multi-Level Action Prediction strategy, which combines spatial-semantic cues at multiple granularities, is designed to assist agents in decision-making. Extensive experiments conducted on three public benchmarks (i.e., R2R, R4R, and REVERIE) validate the effectiveness of our method.

</details>

---

## 82. Benchmarking Multimodal CoT Reward Model Stepwise by Visual Program

- [ ] Benchmarking Multimodal CoT Reward Model Stepwise by Visual Program | https://openaccess.thecvf.com/content/ICCV2025/html/Gao_Benchmarking_Multimodal_CoT_Reward_Model_Stepwise_by_Visual_Program_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_Benchmarking_Multimodal_CoT_Reward_Model_Stepwise_by_Visual_Program_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in reward signal usage for Large Language Models (LLMs) are remarkable. However, significant challenges exist when transitioning reward signal to the multimodal domain, including labor-intensive annotations, over-reliance on one-step rewards, and inadequate evaluation. To address these issues, we propose SVIP, a novel approach to train a step-level multi-dimensional Chain-of-Thought (CoT) reward model automatically. It generates code for solving visual tasks and transforms the analysis of code blocks into the evaluation of CoT step as training samples. Then, we train SVIP-Reward model using a multi-head attention mechanism called TriAtt-CoT. The advantages of SVIP-Reward are evident throughout the entire process of MLLM. We also introduce a benchmark for CoT reward model training and testing. Experimental results demonstrate that SVIP-Reward improves MLLM performance across training and inference-time scaling, yielding better results on benchmarks while reducing hallucinations and enhancing reasoning ability.

</details>

---

## 83. Causality-guided Prompt Learning for Vision-language Models via Visual Granulation

- [ ] Causality-guided Prompt Learning for Vision-language Models via Visual Granulation | https://openaccess.thecvf.com/content/ICCV2025/html/Gao_Causality-guided_Prompt_Learning_for_Vision-language_Models_via_Visual_Granulation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_Causality-guided_Prompt_Learning_for_Vision-language_Models_via_Visual_Granulation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has recently attracted much attention for adapting pre-trained vision-language models (e.g., CLIP) to downstream recognition tasks. However, most of the existing CLIP-based prompt learning methods only show a limited ability for handling fine-grained datasets. To address this issue, we propose a causality-guided text prompt learning method via visual granulation for CLIP, called CaPL, where the explored visual granulation technique could construct sets of visual granules for the text prompt to capture subtle discrepancies among different fine-grained classes through casual inference. The CaPL method contains the following two modules: (1) An attribute disentanglement module is proposed to decompose visual features into non-individualized attributes (shared by some classes) and individualized attributes (specific to single classes) using a Brownian Bridge Diffusion Model; (2) A granule learning module is proposed to construct visual granules by integrating the aforementioned attributes for recognition under two causal inference strategies. Thanks to the learned visual granules, more discriminative text prompt is expected to be learned. Extensive experimental results on 15 datasets demonstrate that our CaPL method significantly outperforms the state-of-the-art prompt learning methods, especially on fine-grained datasets. Code is available at https://github.com/GaoMY-521/CaPL_Code.

</details>

---

## 84. Knowledge Transfer from Interaction Learning

- [ ] Knowledge Transfer from Interaction Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Gao_Knowledge_Transfer_from_Interaction_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_Knowledge_Transfer_from_Interaction_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current visual foundation models (VFMs) face a fundamental limitation in transferring knowledge from vision language models (VLMs): while VLMs excel at modeling cross-modal interactions through unified representation spaces, existing VFMs predominantly adopt result-oriented paradigms that neglect the underlying interaction processes. This representational discrepancy leads to suboptimal knowledge transfer and limited generalization capabilities across vision tasks.We propose Learning from Interactions, a cognitive-inspired framework that bridges this gap by explicitly modeling interactions during visual understanding. Our key insight is that preserving the interaction dynamics captured by VLMs -- rather than just their final representations -- enables more effective knowledge transfer to downstream VFMs. The technical core involves two innovations: (1) Interaction Queries that maintain persistent relationships across network layers, and (2) interaction-based supervision derived from pre-trained VLMs' cross-modal attention patterns.Comprehensive experiments demonstrate consistent improvements across multiple benchmarks: achieving ~3.3% and +1.6 mAP/+2.4 AP^ mask  absolute gains on TinyImageNet classification and COCO detection/segmentation respectively, with minimal parameter overhead and faster convergence (7xspeedup). The framework particularly excels in cross-domain scenarios, delivering ~2.4% and ~9.3% zero-shot improvements on PACS and VLCS. Human evaluations confirm our approach's cognitive alignment, outperforming result-oriented methods by 2.7xin semantic consistency metrics.

</details>

---

## 85. MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning

- [ ] MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning | https://openaccess.thecvf.com/content/ICCV2025/html/Gao_MMAT-1M_A_Large_Reasoning_Dataset_for_Multimodal_Agent_Tuning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_MMAT-1M_A_Large_Reasoning_Dataset_for_Multimodal_Agent_Tuning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs), enhanced through agent tuning, have demonstrated remarkable capabilities in Chain-of-Thought (CoT) and tool utilization, significantly surpassing the performance of standalone models. However, the multimodal domain still lacks a large-scale, high-quality agent tuning dataset to unlock the full potential of multimodal large language models. To bridge this gap, we introduce MMAT-1M, the first million-scale multimodal agent tuning dataset designed to support CoT, reflection, and dynamic tool usage. Our dataset is constructed through a novel four-stage data engine: 1) We first curate publicly available multimodal datasets containing question-answer pairs; 2) Then, leveraging GPT-4o, we generate rationales for the original question-answer pairs and dynamically integrate API calls and Retrieval Augmented Generation (RAG) information through a multi-turn paradigm; 3) Furthermore, we refine the rationales through reflection to ensure logical consistency and accuracy, creating a multi-turn dialogue dataset with both Rationale and Reflection (RR); 4) Finally, to enhance efficiency, we optionally compress multi-turn dialogues into a One-turn Rationale and Reflection (ORR) format. By fine-tuning open-source multimodal models on the MMAT-1M, we observe significant performance gains. For instance, the InternVL2.5-8B-RR model achieves an average improvement of 2.7% across eight public benchmarks and 8.8% on the RAG benchmark Dyn-VQA, demonstrating the dataset's effectiveness in enhancing multimodal reasoning and tool-based capabilities. The dataset is publicly available at https://github.com/VIS-MPU-Agent/MMAT-1M.

</details>

---

## 86. ProbMED: A Probabilistic Framework for Medical Multimodal Binding

- [ ] ProbMED: A Probabilistic Framework for Medical Multimodal Binding | https://openaccess.thecvf.com/content/ICCV2025/html/Gao_ProbMED_A_Probabilistic_Framework_for_Medical_Multimodal_Binding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_ProbMED_A_Probabilistic_Framework_for_Medical_Multimodal_Binding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical decision-making requires integrating diverse medical information, from imaging to clinical narratives. These medical modalities are often acquired in a many-to-many manner. However, current medical vision-language pretraining models (Med-VLPMs) fail to directly account for this many-to-many mapping in their model training and embeddings. To address this, we present Probabilistic Modality-Enhanced Diagnosis (ProbMED), a multimodal Med-VLPM that employs probabilistic contrastive learning to model distributions over embeddings rather than deterministic estimates. ProbMED aligns four distinct modalities--chest X-rays, electrocardiograms, echocardiograms, and clinical text--into a unified probabilistic embedding space. We use InfoNCE loss with Hellinger distance to integrate inter-modality distributions. We introduce a probabilistic synthetic sampling loss that captures modality-specific mean and variance to improve intra-modality binding. Extensive experiments across 13 medical datasets demonstrate that our model outperforms current Med-VLPMs in cross-modality retrieval, zero-shot, and few-shot classification. We also demonstrate the robust integration of multiple modalities for prognostication, showing improved intra- and inter-medical modality binding.

</details>

---

## 87. CLIP-Adapted Region-to-Text Learning for Generative Open-Vocabulary Semantic Segmentation

- [ ] CLIP-Adapted Region-to-Text Learning for Generative Open-Vocabulary Semantic Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Ge_CLIP-Adapted_Region-to-Text_Learning_for_Generative_Open-Vocabulary_Semantic_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ge_CLIP-Adapted_Region-to-Text_Learning_for_Generative_Open-Vocabulary_Semantic_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, Open-Vocabulary Semantic Segmentation (OVSS) has been largely advanced. However, existing methods mostly rely on a pre-trained vision-language model (e.g., CLIP) and require a predefined set of classes to guide the semantic segmentation process during the inference. This not only narrows the application scenario but also constrains comprehension within a finite vocabulary. To overcome this, we reformulate OVSS as a text generation task and propose the CLIP-adapted Region-to-Text Network (CRTNet) that achieves vocabulary-free OVSS by generating category names and descriptions upon segmentation masks. The training process consists of two steps to ensure an accurate and detailed interpretation of the masked regions: (i) the initial step adapts CLIP visual features to mask-level proposal features using binarized masks extracted by a trained mask extractor, and (ii) the subsequent step involves selecting and aggregating these features to become text-aware by integrating CLIP text embeddings, effectively aligning visual data with corresponding linguistic data to facilitate region-to-text learning. Furthermore, we introduce a series of parsing and filtering techniques to integrate multiple sources of training data to improve the generalization ability of our model. Experiments demonstrate that our model not only excels in OVSS but also exhibits scalability and can be adapted to various foundation models (e.g., SAM) without being retrained.

</details>

---

## 88. Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining

- [ ] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining | https://openaccess.thecvf.com/content/ICCV2025/html/Ge_Iris_Breaking_GUI_Complexity_with_Adaptive_Focus_and_Self-Refining_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ge_Iris_Breaking_GUI_Complexity_with_Adaptive_Focus_and_Self-Refining_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Digital agents are increasingly employed to automate tasks in interactive digital environments such as web pages, software applications, and operating systems. While text-based agents built on Large Language Models (LLMs) often require frequent updates due to platform-specific APIs, visual agents leveraging Multimodal Large Language Models (MLLMs) offer enhanced adaptability by interacting directly with Graphical User Interfaces (GUIs). However, these agents face significant challenges in visual perception, particularly when handling high-resolution, visually complex digital environments. This paper introduces Iris, a foundational visual agent that addresses these challenges through two key innovations: Information-Sensitive Cropping (ISC) and Self-Refining Dual Learning (SRDL). ISC dynamically identifies and prioritizes visually dense regions using an edge detection algorithm, enabling efficient processing by allocating more computational resources to areas with higher information density. SRDL enhances the agent's ability to handle complex tasks by leveraging a dual-learning loop, where improvements in referring (describing UI elements) reinforce grounding (locating elements) and vice versa, all without requiring additional annotated data. Empirical evaluations demonstrate that Iris achieves state-of-the-art performance across multiple benchmarks with only 850K GUI annotations, outperforming methods using 10x more training data. These improvements further translate to significant gains in both web and OS agent downstream tasks.

</details>

---

## 89. V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding

- [ ] V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding | https://openaccess.thecvf.com/content/ICCV2025/html/Ge_V2PE_Improving_Multimodal_Long-Context_Capability_of_Vision-Language_Models_with_Variable_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ge_V2PE_Improving_Multimodal_Long-Context_Capability_of_Vision-Language_Models_with_Variable_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have shown promising capabilities in handling various multimodal tasks, yet they struggle in long-context scenarios, particularly tasks involving videos, high-resolution images, or lengthy image-text documents. In our work, we first conduct an empirical analysis of VLMs' long-context capabilities using our augmented long-context multimodal datasets. Our findings reveal that directly applying the positional encoding mechanism used for textual tokens to visual tokens is suboptimal, and VLM performance degrades sharply when the position encoding exceeds the model's context window. To address this, we propose Variable Visual Position Encoding (V2PE), a novel positional encoding approach that employs variable and smaller increments for visual tokens, enabling more efficient management of long multimodal sequences. Our experiments demonstrate the effectiveness of V2PE in enhancing VLMs' ability to effectively understand and reason over long multimodal contexts. We further integrate V2PE with our augmented long-context multimodal datasets to fine-tune the open-source VLMs. The fine-tuned model achieves strong performance on both standard and long-context multimodal tasks. Notably, when the sequence length of the training dataset is increased to 256K tokens, the model is capable of processing multimodal sequences up to 1M tokens, highlighting its potential for real-world long-context applications. We shall release the code, model weights, and datasets to facilitate further research.

</details>

---

## 90. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders

- [ ] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders | https://openaccess.thecvf.com/content/ICCV2025/html/Geng_SAUCE_Selective_Concept_Unlearning_in_Vision-Language_Models_with_Sparse_Autoencoders_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Geng_SAUCE_Selective_Concept_Unlearning_in_Vision-Language_Models_with_Sparse_Autoencoders_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.

</details>

---

## 91. ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones

- [ ] ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones | https://openaccess.thecvf.com/content/ICCV2025/html/Ghosh_ROADWork_A_Dataset_and_Benchmark_for_Learning_to_Recognize_Observe_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ghosh_ROADWork_A_Dataset_and_Benchmark_for_Learning_to_Recognize_Observe_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Perceiving and autonomously navigating through work zones is a challenging and under-explored problem. Open datasets for this long-tailed scenario are scarce. We propose the ROADWork dataset to learn to recognize, observe, analyze, and drive through work zones. State-of-the-art foundation models fail when applied to work zones. Fine-tuning models on our dataset significantly improves perception and navigation in work zones. With ROADWork, we discover new work zone images with higher precision (+32.5%) at a much higher rate (12.8x) around the world. Open-vocabulary methods fail too, whereas fine-tuned detectors improve performance (+32.2 AP).Vision-Language Models (VLMs) struggle to describe work zones, but fine-tuning substantially improves performance (+36.7 SPICE). Beyond fine-tuning, we show the value of simple techniques. Video label propagation provides additional gains (+2.6 AP) for instance segmentation. While reading work zone signs, composing a detector and text spotter via crop-scaling improves performance (+14.2% 1-NED). Composing work zone detections to provide context further reduces hallucinations (+3.9 SPICE) in VLMs. We predict navigational goals and compute drivable paths from work zone videos. Incorporating road work semantics ensures 53.6% goals have angular error (AE) < 0.5 (+9.9%) and 75.3% pathways have AE < 0.5 (+8.1%).

</details>

---

## 92. SAGI: Semantically Aligned and Uncertainty Guided AI Image Inpainting

- [ ] SAGI: Semantically Aligned and Uncertainty Guided AI Image Inpainting | https://openaccess.thecvf.com/content/ICCV2025/html/Giakoumoglou_SAGI_Semantically_Aligned_and_Uncertainty_Guided_AI_Image_Inpainting_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Giakoumoglou_SAGI_Semantically_Aligned_and_Uncertainty_Guided_AI_Image_Inpainting_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in generative AI have made text-guided image inpainting--adding, removing, or altering image regions using textual prompts--widely accessible. However, generating semantically correct photorealistic imagery, typically requires carefully-crafted prompts and iterative refinement by evaluating the realism of the generated content - tasks commonly performed by humans. To automate the generative process, we propose Semantically Aligned and Uncertainty Guided AI Image Inpainting (SAGI), a model-agnostic pipeline, to sample prompts from a distribution that closely aligns with human perception and to evaluate the generated content and discard instances that deviate from such a distribution, which we approximate using pretrained large language models and vision-language models. By applying this pipeline on multiple state-of-the-art inpainting models, we create the SAGI Dataset SAGI-D, currently the largest and most diverse dataset of AI-generated inpaintings, comprising over 95k inpainted images and a human-evaluated subset. Our experiments show that semantic alignment significantly improves image quality and aesthetics, while uncertainty guidance effectively identifies realistic manipulations -- human ability to distinguish inpainted images from real ones drops from 74% to 35% in terms of accuracy, after applying our pipeline. Moreover, using SAGI-D for training several image forensic approaches increases in-domain detection performance on average by 37.4% and out-of-domain generalization by 26.1% in terms of IoU, also demonstrating its utility in countering malicious exploitation of generative AI. Code and dataset are available at https://mever-team.github.io/SAGI/

</details>

---

## 93. ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models

- [ ] ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Gong_ZeroKey_Point-Level_Reasoning_and_Zero-Shot_3D_Keypoint_Detection_from_Large_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gong_ZeroKey_Point-Level_Reasoning_and_Zero-Shot_3D_Keypoint_Detection_from_Large_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We propose a novel zero-shot approach for keypoint detection on 3D shapes. Point-level reasoning on visual data is challenging as it requires precise localization capability, posing problems even for powerful models like DINO or CLIP. Traditional methods for 3D keypoint detection rely heavily on annotated 3D datasets and extensive supervised training, limiting their scalability and applicability to new categories or domains. In contrast, our method utilizes the rich knowledge embedded within Multi-Modal Large Language Models (MLLMs). Specifically, we demonstrate, for the first time, that pixel-level annotations used to train recent MLLMs can be exploited for both extracting and naming salient keypoints on 3D models without any ground truth labels or supervision. Experimental evaluations demonstrate that our approach achieves competitive performance on standard benchmarks compared to supervised methods, despite not requiring any 3D keypoint annotations during training. Our results highlight the potential of integrating language models for localized 3D shape understanding. This work opens new avenues for cross-modal learning and underscores the effectiveness of MLLMs in contributing to 3D computer vision challenges.

</details>

---

## 94. Referring Expression Comprehension for Small Objects

- [ ] Referring Expression Comprehension for Small Objects | https://openaccess.thecvf.com/content/ICCV2025/html/Goto_Referring_Expression_Comprehension_for_Small_Objects_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Goto_Referring_Expression_Comprehension_for_Small_Objects_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring expression comprehension (REC) aims to localize the target object described by a natural language expression.Recent advances in vision-language learning have led to significant performance improvements in REC tasks.However, localizing extremely small objects remains a considerable challenge despite its importance in real-world applications such as autonomous driving.To address this issue, we introduce a novel dataset and method for REC targeting small objects.First, we present the small object REC (SOREC) dataset, which consists of 100,000 pairs of referring expressions and corresponding bounding boxes for small objects in driving scenarios.Second, we propose the progressive-iterative zooming adapter (PIZA), an adapter module for parameter-efficient fine-tuning that enables models to progressively zoom in and localize small objects.In a series of experiments, we apply PIZA to GroundingDINO and demonstrate a significant improvement in accuracy on the SOREC dataset.Our dataset, codes and pre-trained models are publicly available on the project page.

</details>

---

## 95. A Token-level Text Image Foundation Model for Document Understanding

- [ ] A Token-level Text Image Foundation Model for Document Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Guan_A_Token-level_Text_Image_Foundation_Model_for_Document_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Guan_A_Token-level_Text_Image_Foundation_Model_for_Document_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, general visual foundation models (VFMs) have witnessed increasing adoption, particularly as image encoders for popular multi-modal large language models (MLLMs). However, without semantically fine-grained supervision, these models still encounter fundamental prediction errors in the context of downstream text-image-related tasks, i.e., perception, understanding and reasoning with images containing small and dense texts. To bridge this gap, we develop TokenFD, the first token-level visual foundation model specifically tailored for text-image-related tasks, designed to support a variety of traditional downstream applications. To facilitate the pretraining of TokenFD, we also devise a high-quality data production pipeline that constructs the first token-level image text dataset, TokenIT, comprising 20 million images and 1.8 billion token-mask pairs. Furthermore, leveraging this foundation with exceptional image-as-text capability, we seamlessly replace previous VFMs with TokenFD to construct a token-level visual-language MLLM, TokenVL, for VQA-based document understanding tasks. Finally, extensive experiments demonstrate the effectiveness of TokenFD and TokenVL. Code, demo, datasets, and weights are available at https://github.com/Token-family/TokenFD.

</details>

---

## 96. Cooperative Pseudo Labeling for Unsupervised Federated Classification

- [ ] Cooperative Pseudo Labeling for Unsupervised Federated Classification | https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Cooperative_Pseudo_Labeling_for_Unsupervised_Federated_Classification_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Cooperative_Pseudo_Labeling_for_Unsupervised_Federated_Classification_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unsupervised federated learning (UFL) aims to collaboratively train a global model across distributed clients without data sharing and label information. Previous UFL works have predominantly focused on representation learning and clustering tasks. Recently, vision language models (e.g., CLIP) have gained significant attention for their attractive zero-shot prediction capabilities. Leveraging this advancement, classification problems that were previously infeasible under the UFL paradigm now present new opportunities but remain largely unexplored. In this paper, we extend UFL to the classification problem with CLIP for the first time and propose a novel method, **Fed**erated **Co**operative **P**seudo **L**abeling (**FedCoPL**). Specifically, clients estimate and upload their pseudo label distribution, and the server adjusts and redistributes them to avoid global imbalance among categories. Moreover, we introduce a partial prompt aggregation protocol for effective collaboration and personalization. In particular, visual prompts containing general image features are aggregated at the server, while text prompts encoding personalized knowledge are retained locally. Extensive experiments on six datasets demonstrate the superior performance of our FedCoPL compared to baseline methods. Our code is available at https://github.com/krumpguo/FedCoPL.

</details>

---

## 97. IMG: Calibrating Diffusion Models via Implicit Multimodal Guidance

- [ ] IMG: Calibrating Diffusion Models via Implicit Multimodal Guidance | https://openaccess.thecvf.com/content/ICCV2025/html/Guo_IMG_Calibrating_Diffusion_Models_via_Implicit_Multimodal_Guidance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Guo_IMG_Calibrating_Diffusion_Models_via_Implicit_Multimodal_Guidance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Ensuring precise multimodal alignment between diffusion-generated images and input prompts has been a long-standing challenge. Earlier works finetune diffusion weight using high-quality preference data, which tends to be limited and difficult to scale up. Recent editing-based methods further refine local regions of generated images but may compromise overall image quality. In this work, we propose Implicit Multimodal Guidance (IMG), a novel re-generation-based multimodal alignment framework that requires no extra data or editing operations. Specifically, given a generated image and its prompt, IMG a) utilizes a multimodal large language model (MLLM) to identify misalignments; b) introduces an Implicit Aligner that manipulates diffusion conditioning features to reduce misalignments and enable re-generation; and c) formulates the re-alignment goal into a trainable objective, namely Iteratively Updated Preference Objective. Extensive qualitative and quantitative evaluations on SDXL, SDXL-DPO, and FLUX show that IMG outperforms existing alignment methods. Furthermore, IMG acts as a flexible plug-and-play adapter, seamlessly enhancing prior finetuning-based alignment methods. Our code is available at https://github.com/SHI-Labs/IMG-Multimodal-Diffusion-Alignment.

</details>

---

## 98. ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization

- [ ] ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization | https://openaccess.thecvf.com/content/ICCV2025/html/Guo_ImageGem_In-the-wild_Generative_Image_Interaction_Dataset_for_Generative_Model_Personalization_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Guo_ImageGem_In-the-wild_Generative_Image_Interaction_Dataset_for_Generative_Model_Personalization_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce ImageGem, a dataset for studying generative models that understand fine-grained individual preferences. We posit that a key challenge hindering the development of such a generative model is the lack of in-the-wild and fine-grained user preference annotations. Our dataset features real-world interaction data from 57K users, who collectively have built 242K customized LoRAs, written 3M text prompts, and created 5M generated images. With user preference annotations from our dataset, we were able to train better preference alignment models. In addition, leveraging individual user preference, we investigated the performance of retrieval models and a vision-language model on personalized image retrieval and generative model recommendation. Finally, we propose an end-to-end framework for editing customized diffusion models in a latent weight space to align with individual user preferences. Our results demonstrate that the ImageGem dataset enables, for the first time, a new paradigm for generative model personalization.

</details>

---

## 99. Integrating Visual Interpretation and Linguistic Reasoning for Geometric Problem Solving

- [ ] Integrating Visual Interpretation and Linguistic Reasoning for Geometric Problem Solving | https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Integrating_Visual_Interpretation_and_Linguistic_Reasoning_for_Geometric_Problem_Solving_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Integrating_Visual_Interpretation_and_Linguistic_Reasoning_for_Geometric_Problem_Solving_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current large vision-language models (LVLMs) typically employ a connector module to link visual features with text embeddings of large language models (LLMs) and use end-to-end training to achieve multi-modal understanding in a unified process. Well alignment needs high-quality pre-training data and a carefully designed training process. Current LVLMs face challenges when addressing complex vision-language reasoning tasks, with their reasoning capabilities notably lagging behind those of LLMs. This paper proposes a paradigm shift: instead of training end-to-end vision-language reasoning models, we advocate for developing a decoupled reasoning framework based on existing visual interpretation specialists and text-based reasoning LLMs. Our approach leverages (1) a dedicated vision-language model to transform the visual content of images into textual descriptions and (2) an LLM to perform reasoning according to the visual-derived text and the original question. This method presents a cost-efficient solution for multi-modal model development by optimizing existing models to work collaboratively, avoiding end-to-end development of vision-language models from scratch. By transforming images into language model-compatible text representations, it facilitates future low-cost and flexible upgrades to upcoming powerful LLMs. We introduce an outcome-rewarded joint-tuning strategy to optimize the cooperation between the visual interpretation and linguistic reasoning model. Evaluation results on vision-language benchmarks demonstrate that the decoupled reasoning framework outperforms recent LVLMs. Our approach yields particularly significant performance gains on visually intensive geometric mathematics problems. The code is available: https://github.com/guozix/DVLR.

</details>

---

## 100. SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency

- [ ] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency | https://openaccess.thecvf.com/content/ICCV2025/html/Guo_SCAN_Bootstrapping_Contrastive_Pre-training_for_Data_Efficiency_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Guo_SCAN_Bootstrapping_Contrastive_Pre-training_for_Data_Efficiency_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While contrastive pre-training is widely employed, its data efficiency problem has remained relatively under-explored thus far. Existing methods often rely on static coreset selection algorithms to pre-identify important data for training. However, this static nature renders them unable to dynamically track the data utility throughout pre-training, leading to subpar pre-trained models. To address this challenge, our paper introduces a novel dynamic bootstrapping dataset pruning method. It involves pruning data preparation followed by dataset mutation operations, both of which undergo iterative and dynamic updates. We apply this method to two prevalent contrastive pre-training frameworks: CLIP and MoCo, representing vision-language and vision-centric domains, respectively. In particular, we individually pre-train seven CLIP models on two large-scale image-text pair datasets, and two MoCo models on the ImageNet dataset, resulting in a total of 16 pre-trained models. With a data pruning rate of 30-35% across all 16 models, our method exhibits only marginal performance degradation (less than 1% on average) compared to corresponding models trained on the full dataset counterparts across various downstream datasets, and also surpasses several baselines with a large performance margin. Additionally, the byproduct from our method, i.e., coresets derived from the original datasets after pre-training, also demonstrates significant superiority in terms of downstream performance over other static coreset selection approaches. Code is available at https://github.com/guoyang9/SCAN.

</details>

---

## 101. TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision

- [ ] TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision | https://openaccess.thecvf.com/content/ICCV2025/html/Gupta_TOGA_Temporally_Grounded_Open-Ended_Video_QA_with_Weak_Supervision_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Gupta_TOGA_Temporally_Grounded_Open-Ended_Video_QA_with_Weak_Supervision_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We address the problem of video question answering (video QA) with temporal grounding in a weakly supervised setup, without any temporal annotations. Given a video and a question, we generate an open-ended answer grounded with the start and end time. For this task, we propose TOGA: a vision-language model for Temporally Grounded Open-Ended Video QA with Weak Supervision. We instruct-tune TOGA to jointly generate the answer and the temporal grounding. We operate in a weakly supervised setup where the temporal grounding annotations are not available.We generate pseudo labels for temporal grounding and ensure the validity of these labels by imposing a consistency constraint between the question of a grounding response and the response generated by a question referring to the same temporal segment. We notice that jointly generating the answers with the grounding improves performance on question answering as well as grounding.We evaluate TOGA on grounded QA and open-ended QA tasks. For grounded QA, we consider the NExT-GQA benchmark which is designed to evaluate weakly supervised grounded open-ended question answering.For open-ended QA, we consider the MSVD-QA and ActivityNet-QA benchmarks. We achieve state-of-the-art performance for both tasks on these benchmarks.

</details>

---

## 102. All in One: Visual-Description-Guided Unified Point Cloud Segmentation

- [ ] All in One: Visual-Description-Guided Unified Point Cloud Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Han_All_in_One_Visual-Description-Guided_Unified_Point_Cloud_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Han_All_in_One_Visual-Description-Guided_Unified_Point_Cloud_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unified segmentation of 3D point clouds is crucial for scene understanding, but is hindered by its sparse structure, limited annotations, and the challenge of distinguishing fine-grained object classes in complex environments. Existing methods often struggle to capture rich semantic and contextual information due to limited supervision and a lack of diverse multimodal cues, leading to suboptimal differentiation of classes and instances. To address these challenges, we propose VDG-Uni3DSeg, a novel framework that integrates pre-trained vision-language models (e.g., CLIP) and large language models (LLMs) to enhance 3D segmentation. By leveraging LLM-generated textual descriptions and reference images from the internet, our method incorporates rich multimodal cues, facilitating fine-grained class and instance separation. We further design a Semantic-Visual Contrastive Loss to align point features with multimodal queries and a Spatial Enhanced Module to model scene-wide relationships efficiently. Operating within a closed-set paradigm that utilizes multimodal knowledge generated offline, VDG-Uni3DSeg achieves state-of-the-art results in semantic, instance, and panoptic segmentation, offering a scalable and practical solution for 3D understanding.Our code is available at https://github.com/Hanzy1996/VDG-Uni3DSeg.

</details>

---

## 103. Unlearning the Noisy Correspondence Makes CLIP More Robust

- [ ] Unlearning the Noisy Correspondence Makes CLIP More Robust | https://openaccess.thecvf.com/content/ICCV2025/html/Han_Unlearning_the_Noisy_Correspondence_Makes_CLIP_More_Robust_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Han_Unlearning_the_Noisy_Correspondence_Makes_CLIP_More_Robust_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The data appetite for Vision-Language Models (VLMs) has continuously scaled up from the early millions to billions today, which faces an untenable trade-off with data quality and inevitably introduces Noisy Correspondence (NC) samples. Undoubtedly, such semantically unrelated data significantly impairs the performance of VLMs. Previous efforts mainly address this challenge by estimating refined alignment for more precise guidance. However, such resource-intensive pipelines that train VLMs from scratch struggle to meet realistic data demands. In this paper, we present a brand new perspective that seeks to directly eliminate the harmful effects of NC in pre-trained VLMs. Specifically, we propose NCU, a Noisy Correspondence Unlearning fine-tuning framework that efficiently enhances VLMs' robustness by forgetting learned noisy knowledge. The key to NCU is learning the hardest negative information, which can provide explicit unlearning direction for both false positives and false negatives. Such twin goals unlearning process can be formalized into one unified optimal transport objective for fast fine-tuning. We validate our approach with the prevailing CLIP model over various downstream tasks. Remarkably, NCU surpasses the robust pre-trained method on zero-shot transfer while with lower computational overhead. The code is available at https://github.com/hhc1997/NCU.

</details>

---

## 104. Vision-Language Neural Graph Featurization for Extracting Retinal Lesions

- [ ] Vision-Language Neural Graph Featurization for Extracting Retinal Lesions | https://openaccess.thecvf.com/content/ICCV2025/html/Hassan_Vision-Language_Neural_Graph_Featurization_for_Extracting_Retinal_Lesions_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hassan_Vision-Language_Neural_Graph_Featurization_for_Extracting_Retinal_Lesions_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retinopathy comprises a group of retinal disorders that can lead to severe visual impairments or blindness. The heterogeneous morphology of lesions poses a significant challenge in developing robust diagnostic systems. Supervised approaches rely on large labeled datasets and often struggle with generalization. To address these limitations, we propose an unsupervised vision-language neural graph featurization method. This method first segments fundus images into a set of superpixels via Simple Linear Iterative Clustering (SLIC). The superpixels are then decomposed into an undirected graph where each superpixel serves as a node, and spatially adjacent nodes are connected by edges. A Hamiltonian path systematically traverses the graph and iteratively updates and propagates node and edge latent space embeddings throughout the graph until convergence is achieved. Then, a normalized cut separates the converged embeddings into two clusters within a latent space that represent the lesion and healthy superpixels of the input scans. The lesion superpixels are further classified into lesion categories using a prompt-based zero-shot vision-language model. The proposed method is rigorously tested on four public datasets, dubbed ODIR, FIVES, BIOMISA, and IDRiD, achieving F1-scores of 0.89, 0.92, 0.93, and 0.92, respectively, with significant performance gains over state-of-the-art methods.

</details>

---

## 105. DexVLG: Dexterous Vision-Language-Grasp Model at Scale

- [ ] DexVLG: Dexterous Vision-Language-Grasp Model at Scale | https://openaccess.thecvf.com/content/ICCV2025/html/He_DexVLG_Dexterous_Vision-Language-Grasp_Model_at_Scale_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/He_DexVLG_Dexterous_Vision-Language-Grasp_Model_at_Scale_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As large models gain traction, vision-language models are enabling robots to tackle increasingly complex tasks. However, limited by the difficulty of data collection, progress has mainly focused on controlling simple gripper end-effectors. There is little research on functional grasping with large models for human-like dexterous hands. In this paper, we introduce DexVLG, a large Vision-Language-Grasp model for Dexterous grasp pose prediction aligned with language instructions using single-view RGBD input. To accomplish this, we generate a dataset of 170 million dexterous grasp poses mapped to semantic parts across 174,000 objects in simulation, paired with detailed part-level captions. This large-scale dataset, named DexGraspNet 3.0, is used to train a VLM with a flow-matching-based pose head producing instruction-aligned grasp poses for tabletop objects. To evaluate DexVLG's performance, we create benchmarks in simulations and conduct real-world experiments. Extensive experiments demonstrate DexVLG's strong zero-shot generalization capabilities, achieving an over 76% zero-shot execution success rate and state-of-the-art part-grasp accuracy in simulation, as well as successful part-aligned grasps on physical objects in real-world scenarios.

</details>

---

## 106. PlanGen: Towards Unified Layout Planning and Image Generation in Auto-Regressive Vision Language Models

- [ ] PlanGen: Towards Unified Layout Planning and Image Generation in Auto-Regressive Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/He_PlanGen_Towards_Unified_Layout_Planning_and_Image_Generation_in_Auto-Regressive_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/He_PlanGen_Towards_Unified_Layout_Planning_and_Image_Generation_in_Auto-Regressive_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose a unified layout planning and image generation model, PlanGen, which can pre-plan spatial layout conditions before generating images as shown in Figure 1. Unlike previous diffusion-based models that treat layout planning and layout-to-image as two separate models, PlanGen jointly models the two tasks into one autoregressive transformer using only next-token prediction. PlanGen integrates layout conditions into the model as context without requiring specialized encoding of local captions and bounding box coordinates, which provides significant advantages over the previous embed-and-pool operations on layout conditions, particularly when dealing with complex layouts. Unified prompting allows PlanGen to perform multitasking training related to layout, including layout planning, layout-to-image generation, image layout understanding, etc. In addition, PlanGen can be seamlessly expanded to layout-guided image manipulation thanks to the well-designed modeling, with teacher-forcing content manipulation policy and negative layout guidance. Extensive experiments verify the effectiveness of our PlanGen in multiple layout-related tasks, showing its great potential.

</details>

---

## 107. Progressive Distribution Bridging: Unsupervised Adaptation for Large-scale Pre-trained Models via Adaptive Auxiliary Data

- [ ] Progressive Distribution Bridging: Unsupervised Adaptation for Large-scale Pre-trained Models via Adaptive Auxiliary Data | https://openaccess.thecvf.com/content/ICCV2025/html/He_Progressive_Distribution_Bridging_Unsupervised_Adaptation_for_Large-scale_Pre-trained_Models_via_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/He_Progressive_Distribution_Bridging_Unsupervised_Adaptation_for_Large-scale_Pre-trained_Models_via_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pre-trained Vision-Language Models (VLMs) like CLIP have demonstrated promising zero-shot transfer capabilities to downstream tasks. However, their performance deteriorates when facing significant domain shifts. In this paper, we focus on cost-effective adaptation of large-scale pre-trained VLMs to unlabeled target domains. In this context, two prevalent paradigms show inherent limitations: Unsupervised Fine-Tuning (UFT) struggles with poor initial model performance, while Unsupervised Domain Adaptation (UDA) may suffer from adverse effects of inappropriate auxiliary source domain. To alleviate these limitations, we propose to adaptively construct more suitable auxiliary data from large-scale image-text pairs to facilitate unsupervised adaptation without any human annotations. Specifically, we introduce Progressive Distribution Bridging (PDB), which decomposes the challenging adaptation task into multiple simple steps through the construction of auxiliary data. To obtain such data, we design an efficient and controllable retrieval algorithm incorporating cascaded semantic filters and style controller to regulate the semantic category and domain style of retrieved data, respectively. Experimental results across 11 different domains from three standard UDA benchmarks demonstrate the effectiveness of our auxiliary data. Notably, on Office-Home, our method outperforms state-of-the-art UDA methods that rely on labeled source domains. The proposed method offers a more universal and cost-effective solution for adapting VLMs to unlabeled downstream tasks.

</details>

---

## 108. RareCLIP: Rarity-aware Online Zero-shot Industrial Anomaly Detection

- [ ] RareCLIP: Rarity-aware Online Zero-shot Industrial Anomaly Detection | https://openaccess.thecvf.com/content/ICCV2025/html/He_RareCLIP_Rarity-aware_Online_Zero-shot_Industrial_Anomaly_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/He_RareCLIP_Rarity-aware_Online_Zero-shot_Industrial_Anomaly_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models such as CLIP have made significant strides in zero-shot anomaly detection through prompt engineering. However, most existing methods typically process each test image individually, ignoring the practical rarity of abnormal patches in real-world scenarios. Although some batch-based approaches exploit the rarity by processing multiple samples concurrently, they generally introduce unacceptable latency for real-time applications. To mitigate these limitations, we propose RareCLIP, a novel online zero-shot anomaly detection framework that enables sequential image processing in real-time without requiring prior knowledge of the target domain. RareCLIP capitalizes on the zero-shot capabilities of CLIP and integrates a dynamic test-time rarity estimation mechanism. A key innovation of our framework is the introduction of a prototype patch feature memory bank, which aggregates representative features from historical observations and continuously updates their corresponding rarity measures. For each incoming image patch, RareCLIP computes a rarity score by aggregating the rarity measures of its nearest neighbors within the memory bank. Moreover, we introduce a prototype sampling strategy based on dissimilarity to enhance computational efficiency, as well as a similarity calibration strategy to enhance the robustness of rarity estimation. Extensive experiments demonstrate that RareCLIP attains state-of-the-art performance with 98.2% image-level AUROC on MVTec AD and 94.4% on VisA, while achieving a latency of 59.4 ms. Code is available at https://github.com/hjf02/RareCLIP.

</details>

---

## 109. ZipVL: Accelerating Vision-Language Models through Dynamic Token Sparsity

- [ ] ZipVL: Accelerating Vision-Language Models through Dynamic Token Sparsity | https://openaccess.thecvf.com/content/ICCV2025/html/He_ZipVL_Accelerating_Vision-Language_Models_through_Dynamic_Token_Sparsity_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/He_ZipVL_Accelerating_Vision-Language_Models_through_Dynamic_Token_Sparsity_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The efficiency of large vision-language models (LVLMs) is constrained by the computational bottleneck of the attention mechanism during the prefill phase and the memory bottleneck of fetching the key-value (KV) cache in the decoding phase, particularly in scenarios involving high-resolution images or videos. Visual content often exhibits substantial redundancy, resulting in highly sparse attention maps within LVLMs. This sparsity can be leveraged to accelerate attention computation or compress the KV cache through various approaches. However, most studies focus on addressing only one of these bottlenecks and do not adequately support dynamic adjustment of sparsity concerning distinct layers or tasks. In this paper, we present ZipVL, an efficient inference framework designed for LVLMs through a dynamic ratio allocation strategy of important tokens. This ratio is adaptively determined based on the layer-specific distribution of attention scores, rather than fixed hyper-parameters, thereby improving efficiency for less complex tasks while maintaining high performance for more challenging ones. Then we select important tokens based on their normalized attention scores and perform sparse attention mechanism solely on those important tokens, reducing the latency in the prefill phase. Tokens deemed less important will be discarded to reduce KV cache size, alleviating the memory bottleneck in the decoding phase. Our experiments demonstrate that ZipVL can accelerate the prefill phase by 2.3x and improve decoding throughput by 2.8x, with a minimal accuracy reduction of only 0.5% on VQAv2 benchmark over LLaVA-Next-13B model, effectively enhancing the generation efficiency of LVLMs.

</details>

---

## 110. Understanding Co-speech Gestures in-the-wild

- [ ] Understanding Co-speech Gestures in-the-wild | https://openaccess.thecvf.com/content/ICCV2025/html/Hegde_Understanding_Co-speech_Gestures_in-the-wild_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hegde_Understanding_Co-speech_Gestures_in-the-wild_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Co-speech gestures play a vital role in non-verbal communication. In this paper, we introduce a new framework for co-speech gesture understanding in the wild. Specifically, we propose three new tasks and benchmarks to evaluate a model's capability to comprehend gesture-speech-text associations: (i) gesture based retrieval, (ii) gestured word spotting, and (iii) active speaker detection using gestures. We present a new approach that learns a tri-modal video-gesture-speech-text representation to solve these tasks. By leveraging a combination of global phrase contrastive loss and local gesture-word coupling loss, we demonstrate that a strong gesture representation can be learned in a weakly supervised manner from videos in the wild. Our learned representations outperform previous methods, including large vision-language models (VLMs). Further analysis reveals that speech and text modalities capture distinct gesture related signals, underscoring the advantages of learning a shared tri-modal embedding space. The dataset, model, and code are available at: https://www.robots.ox.ac.uk/ vgg/research/jegal.

</details>

---

## 111. 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos

- [ ] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos | https://openaccess.thecvf.com/content/ICCV2025/html/Heidinger_2HandedAfforder_Learning_Precise_Actionable_Bimanual_Affordances_from_Human_Videos_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Heidinger_2HandedAfforder_Learning_Precise_Actionable_Bimanual_Affordances_from_Human_Videos_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

When interacting with objects, humans effectively reason about which regions of objects are viable for an intended action, i.e., the affordance regions of the object. They can also account for subtle differences in object regions based on the task to be performed and whether one or two hands need to be used. However, current vision-based affordance prediction methods often reduce the problem to naive object part segmentation. In this work, we propose a framework for extracting affordance data from human activity video datasets. Our extracted 2HANDS dataset contains precise object affordance region segmentations and affordance class-labels as narrations of the activity performed. The data also accounts for bimanual actions, i.e., two hands co-ordinating and interacting with one or more objects. We present a VLM-based affordance prediction model, 2HandedAfforder, trained on the dataset and demonstrate superior performance over baselines in affordance region segmentation for various activities. Finally, we show that our predicted affordance regions are actionable, i.e., can be used by an agent performing a task, through demonstration in robotic manipulation scenarios.

</details>

---

## 112. Improving Large Vision and Language Models by Learning from a Panel of Peers

- [ ] Improving Large Vision and Language Models by Learning from a Panel of Peers | https://openaccess.thecvf.com/content/ICCV2025/html/Hernandez_Improving_Large_Vision_and_Language_Models_by_Learning_from_a_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hernandez_Improving_Large_Vision_and_Language_Models_by_Learning_from_a_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional alignment methods for Large Vision and Language Models (LVLMs) primarily rely on human-curated preference data. Human-generated preference data is costly; machine-generated preference data is limited in quality; and self-supervised preference data often introduces hallucinations. To overcome these limitations, we propose a novel Panel-of-Peers learning framework inspired by collaborative learning among humans. This approach leverages a panel of LVLMs, each evaluating and learning from their collective outputs through an iterative self-improvement process. By simulating a peer review system, our models generate, assess, and refine outputs in response to a curated set of prompts, mimicking a classroom learning environment. We demonstrate that this methodology enhances model performance without requiring extensive human-labeled datasets. Our experiments show significant improvement across multiple benchmarks, demonstrating the potential of peer evaluations as a scalable alternative to self-supervised alignment. Notably, we show that Panel-of-Peers increases the average score on fifteen benchmarks from 48% to 57%.

</details>

---

## 113. Bias in Gender Bias Benchmarks: How Spurious Features Distort Evaluation

- [ ] Bias in Gender Bias Benchmarks: How Spurious Features Distort Evaluation | https://openaccess.thecvf.com/content/ICCV2025/html/Hirota_Bias_in_Gender_Bias_Benchmarks_How_Spurious_Features_Distort_Evaluation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hirota_Bias_in_Gender_Bias_Benchmarks_How_Spurious_Features_Distort_Evaluation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Gender bias in vision-language foundation models (VLMs) raises concerns about their safe deployment and is typically evaluated using benchmarks with gender annotations on real-world images. However, as these benchmarks often contain spurious correlations between gender and non-gender features, such as objects and backgrounds, we identify a critical oversight in gender bias evaluation: Do spurious features distort gender bias evaluation? To address this question, we systematically perturb non-gender features across four widely used benchmarks (COCO-gender, FACET, MIAP, and PHASE) and various VLMs to quantify their impact on bias evaluation. Our findings reveal that even minimal perturbations, such as masking just 10% of objects or weakly blurring backgrounds, can dramatically alter bias scores, shifting metrics by up to 175% in generative VLMs and 43% in CLIP variants. This suggests that current bias evaluations often reflect model responses to spurious features rather than gender bias, undermining their reliability. Since creating spurious feature-free benchmarks is fundamentally challenging, we recommend reporting bias metrics alongside feature-sensitivity measurements to enable a more reliable bias assessment.

</details>

---

## 114. 4D Visual Pre-training for Robot Learning

- [ ] 4D Visual Pre-training for Robot Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Hou_4D_Visual_Pre-training_for_Robot_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hou_4D_Visual_Pre-training_for_Robot_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

General visual representations learned from web-scale datasets for robotics have achieved great success in recent years, enabling data-efficient robot learning on manipulation tasks; yet these pre-trained representations are mostly on 2D images, neglecting the inherent 3D nature of the world. However, due to the scarcity of large-scale 3D data, it is still hard to extract a universal 3D representation from web datasets. Instead, we are seeking a general visual pre-training framework that could improve all 3D representations as an alternative. Our framework, called FVP, is a novel 4D Visual Pre-training framework for real-world robot learning. FVP frames the visual pre-training objective as a next-point-cloud-prediction problem, models the prediction model as a diffusion model, and pre-trains the model on the larger public datasets directly. Across twelve real-world manipulation tasks, FVP boosts the average success rate of 3D Diffusion Policy (DP3) for these tasks by 28%. The FVP pre-trained DP3 achieves state-of-the-art performance across imitation learning methods. Moreover, the efficacy of \ours adapts across various point cloud encoders and datasets. Finally, we apply FVP to the RDT-1B, a larger Vision-Language-Action robotic model, enhancing its performance on various robot tasks. Our project page is available at: https://4d-visual-pretraining.github.io/.

</details>

---

## 115. Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

- [ ] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy | https://openaccess.thecvf.com/content/ICCV2025/html/Hou_Dita_Scaling_Diffusion_Transformer_for_Generalist_Vision-Language-Action_Policy_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hou_Dita_Scaling_Diffusion_Transformer_for_Generalist_Vision-Language-Action_Policy_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While recent vision-language-action models trained on diverse robot datasets exhibit promising generalization capabilities with limited in-domain data, their reliance on compact action heads to predict discretized or continuous actions constrains adaptability to heterogeneous action spaces. We present Dita, a scalable framework that leverages Transformer architectures to directly denoise continuous action sequences through a unified multimodal diffusion process. Departing from prior methods that condition denoising on fused embeddings via shallow networks, Dita employs in-context conditioning--enabling fine-grained alignment between denoised actions and raw visual tokens from historical observations. This design explicitly models action deltas and environmental nuances. By capitalizing on the Transformer's scalability, Dita effectively unifies cross-embodiment datasets spanning varying camera perspectives, tasks, and action spaces. Evaluations across extensive benchmarks demonstrate state-of-the-art or comparative performance in simulation. Notably, Dita achieves robust real-world adaptation to environmental variances and complex long-horizon tasks through 10-shot finetuning, using only third-person camera inputs. The architecture establishes a versatile, lightweight and open-source baseline for generalist robot policy learning. The code and website are included in the supplementary materials.

</details>

---

## 116. Bilateral Collaboration with Large Vision-Language Models for Open Vocabulary Human-Object Interaction Detection

- [ ] Bilateral Collaboration with Large Vision-Language Models for Open Vocabulary Human-Object Interaction Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Hu_Bilateral_Collaboration_with_Large_Vision-Language_Models_for_Open_Vocabulary_Human-Object_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hu_Bilateral_Collaboration_with_Large_Vision-Language_Models_for_Open_Vocabulary_Human-Object_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open vocabulary Human-Object Interaction (HOI) detection is a challenging task that detects all <human, verb, object> triplets of interest in an image, even those that are not pre-defined in the training set. Existing approaches typically rely on output features generated by large Vision-Language Models (VLMs) to enhance the generalization ability of interaction representations. However, the visual features produced by VLMs are holistic and coarse-grained, which contradicts the nature of detection tasks. To address this issue, we propose a novel Bilateral Collaboration framework for open vocabulary HOI detection (BC-HOI). This framework includes an Attention Bias Guidance (ABG) component, which guides the VLM to produce fine-grained instance-level interaction features according to the attention bias provided by the HOI detector. It also includes a Large Language Model (LLM)-based Supervision Guidance (LSG) component, which provides fine-grained token-level supervision for the HOI detector by the LLM component of the VLM. LSG enhances the ability of ABG to generate high-quality attention bias. We conduct extensive experiments on two popular benchmarks: HICO-DET and V-COCO, consistently achieving superior performance in the open vocabulary and closed settings. The code will be released in Github.

</details>

---

## 117. GroundingSuite: Measuring Complex Multi-Granular Pixel Grounding

- [ ] GroundingSuite: Measuring Complex Multi-Granular Pixel Grounding | https://openaccess.thecvf.com/content/ICCV2025/html/Hu_GroundingSuite_Measuring_Complex_Multi-Granular_Pixel_Grounding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hu_GroundingSuite_Measuring_Complex_Multi-Granular_Pixel_Grounding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pixel grounding, encompassing tasks such as Referring Expression Segmentation (RES), has garnered considerable attention due to its potential for bridging the gap between vision and language modalities. However, advancements in this domain are currently constrained by limitations inherent in existing datasets, including limited object categories, insufficient textual diversity, and a scarcity of high-quality annotations. To mitigate these limitations, we introduce GroundingSuite, which comprises: (1) an automated data annotation framework leveraging multiple Vision-Language Model (VLM) agents; (2) a large-scale training dataset encompassing 9.56 million diverse referring expressions and their corresponding segmentations; and (3) a meticulously curated evaluation benchmark consisting of 3,800 images. The GroundingSuite dataset boosts model performance to state-of-the-art levels. Specifically, a cIoU of 68.9 on gRefCOCO and a gIoU of 55.3 on RefCOCOm. Moreover, the GroundingSuite annotation framework demonstrates superior efficiency compared to the current leading data annotation method, i.e., 4.5x faster than the GLaMM. Codes are available at: https://github.com/hustvl/GroundingSuite.

</details>

---

## 118. OphCLIP: Hierarchical Retrieval-Augmented Learning for Ophthalmic Surgical Video-Language Pretraining

- [ ] OphCLIP: Hierarchical Retrieval-Augmented Learning for Ophthalmic Surgical Video-Language Pretraining | https://openaccess.thecvf.com/content/ICCV2025/html/Hu_OphCLIP_Hierarchical_Retrieval-Augmented_Learning_for_Ophthalmic_Surgical_Video-Language_Pretraining_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hu_OphCLIP_Hierarchical_Retrieval-Augmented_Learning_for_Ophthalmic_Surgical_Video-Language_Pretraining_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pretraining (VLP) enables open-world generalization beyond predefined labels, a critical capability in surgery due to the diversity of procedures, instruments, and patient anatomies. However, applying VLP to ophthalmic surgery presents unique challenges, including limited vision-language data, intricate procedural workflows, and the need for hierarchical understanding, ranging from fine-grained surgical actions to global clinical reasoning. To address these, we introduce OphVL, a large-scale, hierarchically structured dataset containing over 375K video-text pairs, making it 15x larger than existing surgical VLP datasets. OphVL captures a diverse range of ophthalmic surgical attributes, including surgical phases, operations, actions, instruments, medications, disease causes, surgical objectives, and postoperative care recommendations. By aligning short clips with detailed narratives and full-length videos with structured titles, OphVL provides both fine-grained surgical details and high-level procedural context. Building on OphVL, we propose OphCLIP, a hierarchical retrieval-augmented VLP framework. OphCLIP leverages silent surgical videos as a knowledge base, retrieving semantically relevant content to enhance narrated procedure learning. This enables OphCLIP to integrate explicit linguistic supervision with implicit visual knowledge, improving ophthalmic workflow modeling. Evaluations across 11 benchmark datasets for surgical phase recognition and multi-instrument identification demonstrate OphCLIP's robust generalization and superior performance, establishing it as a foundation model for ophthalmic surgery.

</details>

---

## 119. SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation with Long- and Local-range Context Reasoning

- [ ] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation with Long- and Local-range Context Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Hu_SPADE_Spatial-Aware_Denoising_Network_for_Open-vocabulary_Panoptic_Scene_Graph_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Hu_SPADE_Spatial-Aware_Denoising_Network_for_Open-vocabulary_Panoptic_Scene_Graph_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Panoptic Scene Graph Generation (PSG) integrates instance segmentation with relation understanding to capture pixel-level structural relationships in complex scenes. Although recent approaches leveraging pre-trained vision-language models (VLMs) have significantly improved performance in the open-vocabulary setting, they commonly ignore the inherent limitations of VLMs in spatial relation reasoning, such as difficulty in distinguishing object relative positions, which results in suboptimal relation prediction.Motivated by the denoising diffusion model's inversion process in preserving the spatial structure of input images, we propose SPADE (SPatial-Aware Denoising-nEtwork) framework---a novel approach for open-vocabulary PSG. SPADE consists of two key steps: (1) inversion-guided calibration for the UNet adaption, and (2) spatial-aware context reasoning. In the first step, we calibrate a general pre-trained teacher diffusion model into a PSG-specific denoising network with cross-attention maps derived during inversion through a lightweight LoRA-based fine-tuning strategy. In the second step, we develop a spatial-aware relation graph transformer that captures both local and long-range contextual information, facilitating the generation of high-quality relation queries. Extensive experiments on benchmark PSG and Visual Genome datasets demonstrate that SPADE outperforms state-of-the-art methods in both closed-set and open-set scenarios, particularly excelling in spatial relationship prediction. The code is available at: https://anonymous.4open.science/r/SPADE-105F.

</details>

---

## 120. AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference

- [ ] AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference | https://openaccess.thecvf.com/content/ICCV2025/html/Huang_AirCache_Activating_Inter-modal_Relevancy_KV_Cache_Compression_for_Efficient_Large_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Huang_AirCache_Activating_Inter-modal_Relevancy_KV_Cache_Compression_for_Efficient_Large_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Visual Language Models (LVLMs) have gained significant attention due to their remarkable reasoning capabilities and proficiency in generalization. However, processing a large number of visual tokens and generating long-context outputs impose substantial computational overhead, leading to excessive demands for key-value (KV) cache. To address this critical bottleneck, we propose AirCache, a novel KV cache compression method aimed at accelerating LVLMs inference. This work systematically investigates the correlations between visual and textual tokens within the attention mechanisms of LVLMs. Our empirical analysis reveals considerable redundancy in cached visual tokens, wherein strategically eliminating these tokens preserves model performance while significantly accelerating context generation. We introduce an elite observation window for assessing the importance of visual components in the KV cache, focusing on stable inter-modal relevancy modeling with enhanced multi-perspective consistency. Additionally, we develop an adaptive layer-wise budget allocation strategy that capitalizes on the strength and skewness of token importance distribution, showcasing superior efficiency compared to uniform allocation. Comprehensive evaluations across multiple LVLMs and benchmarks demonstrate that our method achieves comparable performance to the full cache while retaining only 10% of visual KV cache, thereby reducing decoding latency by 29% to 66% across various batch size and prompt length of inputs. Notably, as cache retention rates decrease, our method exhibits increasing performance advantages over existing approaches.

</details>

---

## 121. Boosting MLLM Reasoning with Text-Debiased Hint-GRPO

- [ ] Boosting MLLM Reasoning with Text-Debiased Hint-GRPO | https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Boosting_MLLM_Reasoning_with_Text-Debiased_Hint-GRPO_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Boosting_MLLM_Reasoning_with_Text-Debiased_Hint-GRPO_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

MLLM reasoning has drawn widespread research for its excellent problem-solving capability. Current reasoning methods fall into two types: PRM, which supervises the intermediate reasoning steps, and ORM, which supervises the final results. Recently, DeepSeek-R1 has challenged the traditional view that PRM outperforms ORM, which demonstrates strong generalization performance using an ORM method (i.e., GRPO). However, current MLLM's GRPO algorithms still struggle to handle challenging and complex multimodal reasoning tasks (e.g., mathematical reasoning). In this work, we reveal two problems that impede the performance of GRPO on the MLLM: Low data utilization and Text-bias. Low data utilization refers to that GRPO cannot acquire positive rewards to update the MLLM on difficult samples, and text-bias is a phenomenon that the MLLM bypasses image condition and solely relies on text condition for generation after GRPO training. To tackle these problems, this work proposes Hint-GRPO that improves data utilization by adaptively providing hints for samples of varying difficulty, and text-bias calibration that mitigates text-bias by calibrating the token prediction logits with image condition in test-time. Experiment results on three base MLLMs across eleven datasets demonstrate that our proposed methods advance the reasoning capability of original MLLM by a large margin, exhibiting superior performance to existing MLLM reasoning methods. Our code is available at https://github.com/hqhQAQ/Hint-GRPO.

</details>

---

## 122. Deciphering Cross-Modal Alignment in Large Vision-Language Models via Modality Integration Rate

- [ ] Deciphering Cross-Modal Alignment in Large Vision-Language Models via Modality Integration Rate | https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Deciphering_Cross-Modal_Alignment_in_Large_Vision-Language_Models_via_Modality_Integration_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Deciphering_Cross-Modal_Alignment_in_Large_Vision-Language_Models_via_Modality_Integration_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The early stage of multi-modal pre-training plays a pivotal role in aligning two modalities for Large Vision-Language Models (LVLMs), while evaluating its training quality usually requires the costly supervised fine-tuning (SFT) stage to verify the downstream benchmark scores. Loss, perplexity, and in-context evaluation results are commonly used pre-training metrics for Large Language Models (LLMs), while we observed that these metrics are less indicative when quantifying the pre-trained LVLMs. Due to the lack of proper metrics, the research of LVLMs in the multi-modal fusion stage is hindered greatly, including the training data choice, efficient module design, etc.In this paper, we first present Modality Integration Rate (MIR), an effective, robust, and generalized metric to indicate the multi-modal alignment quality of LVLMs without SFT. This metric evaluates LVLM pre-training from the inter-modal distribution distance perspective, which is 1) Effective to represent the fusion quality and show a positive relation with the benchmark performance after SFT, 2) Robust toward different training/evaluation data, and 3) Generalize across training configurations and architecture choices. Complementing MIR, we further propose learnable Modality Calibration (MoCa), a lightweight module to narrow the modality gap at each language model layer during training. A series of experiments are conducted to explore the effectiveness of MIR and MoCa, demonstrating that MIR is highly indicative about training data selection, training strategy schedule, and model architecture design to get better pre-training results. The code is avaliable at \href https://github.com/shikiw/Modality-Integration-Rate  shikiw/Modality-Integration-Rate

</details>

---

## 123. MCID: Multi-aspect Copyright Infringement Detection for Generated Images

- [ ] MCID: Multi-aspect Copyright Infringement Detection for Generated Images | https://openaccess.thecvf.com/content/ICCV2025/html/Huang_MCID_Multi-aspect_Copyright_Infringement_Detection_for_Generated_Images_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Huang_MCID_Multi-aspect_Copyright_Infringement_Detection_for_Generated_Images_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of generative models, we can now create highly realistic images. This represents a significant technical breakthrough but also introduces new challenges for copyright protection. Previous methods for detecting copyright infringement in AI-generated images mainly depend on global similarity. However, real-world infringement often occurs only on certain attributes rather than being a global infringement. To address these challenges, we propose a novel Multi-aspect Copyright Infringement Detection (MCID) task, which encompasses various types of infringement, including content, style, structure, and intellectual property infringement. We further develop the Hybrid Infringement Detection Model (HIDM) to address the MCID task. By combining feature-based methods with VLMs, it enables the detection of various infringement types and provides interpretable results. To ensure the MCID task meets actual legal requirements, we construct a Large-Scale Copyright Dataset (LSCD) with clear author copyright ownership. Based on LSCD, we provide a benchmark annotated by legal experts for performance evaluation. Experimental results show that HIDM effectively detects various types of image copyright infringement and offers a more interpretable and superior solution compared to previous methods.

</details>

---

## 124. Mind the Gap: Preserving and Compensating for the Modality Gap in CLIP-Based Continual Learning

- [ ] Mind the Gap: Preserving and Compensating for the Modality Gap in CLIP-Based Continual Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Mind_the_Gap_Preserving_and_Compensating_for_the_Modality_Gap_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Mind_the_Gap_Preserving_and_Compensating_for_the_Modality_Gap_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Continual learning aims to enable models to learn sequentially from continuously incoming data while retaining performance on previously learned tasks. With the Contrastive Language-Image Pre-trained model (CLIP) exhibiting strong capabilities across various downstream tasks, there has been growing interest in leveraging CLIP for continual learning in such scenarios. Most existing works overlook the inherent modality gap in CLIP, a key factor in its generalization and adaptability. In this paper, we analyze the variations in the modality gap during the fine-tuning of vision-language pre-trained models. Our observations reveal that the modality gap effectively reflects the extent to which pre-trained knowledge is preserved. Based on these insights, we propose a simple yet effective method, MG-CLIP, that improves CLIP's performance in class-incremental learning. Our approach leverages modality gap preservation to mitigate forgetting and modality gap compensation to enhance the capacity for new data, introducing a novel modality-gap-based perspective for continual learning. Extensive experiments on multiple benchmarks demonstrate that our method outperforms existing approaches without requiring additional replay data. Our code is available at https://github.com/linlany/MindtheGap.

</details>

---

## 125. Seeing the Trees for the Forest: Rethinking Weakly-Supervised Medical Visual Grounding

- [ ] Seeing the Trees for the Forest: Rethinking Weakly-Supervised Medical Visual Grounding | https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual grounding (VG) is the capability to identify the specific regions in an image associated with a particular text description. In medical imaging, VG enhances interpretability by highlighting relevant pathological features corresponding to textual descriptions, improving model transparency and trustworthiness for wider adoption of deep learning models in clinical practice. Current models struggle to associate textual descriptions with disease regions due to inefficient attention mechanisms and a lack of fine-grained token representations. In this paper, we empirically demonstrate two key observations. First, current VLMs assign high norms to background tokens, diverting the model's attention from regions of disease. Second, the global tokens used for cross-modal learning are not representative of local disease tokens. This hampers identifying correlations between the text and disease tokens. To address this, we introduce simple, yet effective Disease-Aware Prompting (DAP) process, which uses the explainability map of a VLM to identify the appropriate image features. This simple strategy amplifies disease-relevant regions while suppressing background interference. Without any additional pixel-level annotations, DAP improves visual grounding accuracy by 20.74% compared to state-of-the-art methods across three major chest X-ray datasets.

</details>

---

## 126. Vision-Language Models Can't See the Obvious

- [ ] Vision-Language Models Can't See the Obvious | https://openaccess.thecvf.com/content/ICCV2025/html/Huynh_Vision-Language_Models_Cant_See_the_Obvious_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Huynh_Vision-Language_Models_Cant_See_the_Obvious_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present Saliency Benchmark (SalBench), a novel benchmark designed to assess the capability of Large Vision-Language Models (LVLM) in detecting visually salient features that are readily apparent to humans, such as a large circle amidst a grid of smaller ones. This benchmark focuses on low-level features including color, intensity, and orientation, which are fundamental to human visual processing. Our SalBench consists of images that highlight rare, unusual, or unexpected elements within scenes, and naturally draw human attention. It comprises three novel tasks for evaluating the perceptual capabilities of LVLM: Odd-One-Out Detection, Referring Odd-One-Out, and Visual Referring Odd-One-Out. We perform a comprehensive evaluation of state-of-the-art LVLM using SalBench and our findings reveal a surprising limitation: LVLM struggle to identify seemingly obvious visual anomalies, with even the advanced GPT-4o achieving only 47.6% accuracy on such a simple task. SalBench will be an important step in measuring the capabilities of LVLM that align with the subtle definition of human attention.

</details>

---

## 127. HumorDB: Can AI understand graphical humor?

- [ ] HumorDB: Can AI understand graphical humor? | https://openaccess.thecvf.com/content/ICCV2025/html/Jain_HumorDB_Can_AI_understand_graphical_humor_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jain_HumorDB_Can_AI_understand_graphical_humor_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant advancements in image segmentation and object detection, understanding complex scenes remains a significant challenge. Here, we focus on graphical humor as a paradigmatic example of image interpretation that requires elucidating the interaction of different scene elements in the context of prior cognitive knowledge. This paper introduces HumorDB, a novel, controlled, and carefully curated dataset designed to evaluate and advance visual humor understanding by AI systems. The dataset comprises diverse images spanning photos, cartoons, sketches, and AI-generated content, including minimally contrastive pairs where subtle edits differentiate between humorous and non-humorous versions. We evaluate humans, state-of-the-art vision models, and large vision-language models on three tasks: binary humor classification, funniness rating prediction, and pairwise humor comparison. The results reveal a gap between current AI systems and human-level humor understanding. While pretrained vision-language models perform better than vision-only models, they still struggle with abstract sketches and subtle humor cues. Analysis of attention maps shows that even when models correctly classify humorous images, they often fail to focus on the precise regions that make the image funny. Preliminary mechanistic interpretability studies and evaluation of model explanations provide initial insights into how different architectures process humor. Our results identify promising trends and current limitations, suggesting that an effective understanding of visual humor requires sophisticated architectures capable of detecting subtle contextual features and bridging the gap between visual perception and abstract reasoning.All the code and data are available here: https://anonymous.4open.science/r/HumorDB_-049A

</details>

---

## 128. Target Bias Is All You Need: Zero-Shot Debiasing of Vision-Language Models with Bias Corpus

- [ ] Target Bias Is All You Need: Zero-Shot Debiasing of Vision-Language Models with Bias Corpus | https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Target_Bias_Is_All_You_Need_Zero-Shot_Debiasing_of_Vision-Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Target_Bias_Is_All_You_Need_Zero-Shot_Debiasing_of_Vision-Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) like CLIP have shown remarkable zero-shot performance by aligning different modalities in the embedding space, enabling diverse applications from image editing to visual question answering (VQA). However, these models often inherit biases from their training data, resulting in performance disparities across specific subpopulations. Traditional debiasing methods for VLMs primarily focus on specific downstream tasks using labeled datasets, which we argue is insufficient given the broad applicability of VLMs. Specifically, these methods struggle with generalizability, transferability, and feasibility due to overfitting, limited task applicability, and regulatory constraints on the use of sensitive data, making them less practical in real-world scenarios. To address these challenges, we propose a novel task-agnostic method for learning debiased image embeddings in VLMs. Our approach does not require expensive annotated datasets or curated prompts for downstream tasks, while still preserving the inherent zero-shot capabilities of these models. Instead, we leverage easily accessible information: 1) a bias text corpus generated by a large language model, and 2) a generic unsupervised vision dataset. Our method disentangles the image embedding into bias and neutral components by applying centered kernel alignment (CKA) regularization to the text-vision representational similarity, using the bias text corpus over the generic vision dataset. Experimental results validate the effectiveness of our approach across multiple tasks, offering a practical and versatile solution to debiasing VLMs.

</details>

---

## 129. Towards Cross-modal Backward-compatible Representation Learning for Vision-Language Models

- [ ] Towards Cross-modal Backward-compatible Representation Learning for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Towards_Cross-modal_Backward-compatible_Representation_Learning_for_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Towards_Cross-modal_Backward-compatible_Representation_Learning_for_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Modern retrieval systems often struggle with upgrading to new and more powerful models due to the incompatibility of embeddings between the old and new models. This necessitates a costly process known as backfilling, which involves re-computing the embeddings for a large number of data samples. In vision, Backward-compatible Training (BT) has been proposed to ensure that the new model aligns with the old model's embeddings. This paper extends the concept of vision-only BT to the field of cross-modal retrieval, marking the first attempt to address Cross-modal BT (XBT). Our goal is to achieve backward-compatibility between Vision-Language Pretraining (VLP) models, such as CLIP, for the cross-modal retrieval task. To address XBT challenges, we propose an efficient solution: a projection module that maps the new model's embeddings to those of the old model. This module, pretrained solely with text data, significantly reduces the number of image-text pairs required for XBT learning, and, once it is pretrained, it avoids using the old model during training. Furthermore, we utilize parameter-efficient training strategies that improve efficiency and preserve the off-the-shelf new model's knowledge by avoiding any modifications. Experimental results on cross-modal retrieval datasets demonstrate the effectiveness of XBT and its potential to enable backfill-free upgrades when a new VLP model emerges.

</details>

---

## 130. Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation

- [ ] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Jeon_Exploiting_Domain_Properties_in_Language-Driven_Domain_Generalization_for_Semantic_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jeon_Exploiting_Domain_Properties_in_Language-Driven_Domain_Generalization_for_Semantic_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent domain generalized semantic segmentation (DGSS) studies have achieved notable improvements by distilling semantic knowledge from Vision-Language Models (VLMs). However, they overlook the semantic misalignment between visual and textual contexts, which arises due to the rigidity of a fixed context prompt learned on a single source domain. To this end, we present a novel domain generalization framework for semantic segmentation, namely Domain-aware Prompt-driven Masked Transformer (DPMFormer). Firstly, we introduce domain-aware prompt learning to facilitate semantic alignment between visual and textual cues. To capture various domain-specific properties with a single source dataset, we propose domain-aware contrastive learning along with the texture perturbation that diversifies the observable domains. Lastly, to establish a framework resilient against diverse environmental changes, we have proposed the domain-robust consistency learning which guides the model to minimize discrepancies of prediction from original and the augmented images. Through experiments and analyses, we demonstrate the superiority of the proposed framework, which establishes a new state-of-the-art on various DGSS benchmarks.

</details>

---

## 131. Instruction-based Image Editing with Planning, Reasoning, and Generation

- [ ] Instruction-based Image Editing with Planning, Reasoning, and Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Ji_Instruction-based_Image_Editing_with_Planning_Reasoning_and_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ji_Instruction-based_Image_Editing_with_Planning_Reasoning_and_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Editing images via instruction provides a natural way to generate interactive content, but it is a big challenge due to the higher requirement of scene understanding and generation. Prior work utilizes a chain of large language models, object segmentation models, and editing models for this task. However, the understanding models provide only single-modality ability, restricting the editing quality. We aim to bridge understanding and generation via a new multi-modality model that provides the intelligent abilities to instruction-based image editing models for more complex cases. To achieve this goal, we individually separate the instruction editing task with the multi-modality chain of thought prompts, i.e., Chain-of-Thought (CoT) planning, editing region reasoning, and editing. For Chain-of-Thought planning, the large language model could reason the appropriate sub-prompts considering the instruction provided and the ability of the editing network. For editing region reasoning, we train an instruction-based editing region generation network with a multi-modal large language model. Finally, a hint-guided instruction-based editing network is proposed for editing image generations based on the sizeable text-to-image diffusion model to accept the hints for generation. Extensive experiments demonstrate that our method has competitive editing abilities on complex real-world images. The source code will be publicly available.

</details>

---

## 132. A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets

- [ ] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets | https://openaccess.thecvf.com/content/ICCV2025/html/Jia_A_Visual_Leap_in_CLIP_Compositionality_Reasoning_through_Generation_of_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jia_A_Visual_Leap_in_CLIP_Compositionality_Reasoning_through_Generation_of_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) often struggle with compositional reasoning due to insufficient high-quality image-text data. To tackle this challenge, we propose a novel block-based diffusion approach that automatically generates counterfactual datasets without manual annotation. Our method utilizes large language models to identify entities and their spatial relationships. It then independently generates image blocks as "puzzle pieces" coherently arranged according to specified compositional rules. This process creates diverse, high-fidelity counterfactual image-text pairs with precisely controlled variations. In addition, we introduce a specialized loss function that differentiates inter-set from intra-set samples, enhancing training efficiency and reducing the need for negative samples. Experiments demonstrate that fine-tuning VLMs with our counterfactual datasets significantly improves visual reasoning performance. Our approach achieves state-of-the-art results across multiple benchmarks while using substantially less training data than existing methods.

</details>

---

## 133. From Imitation to Innovation: The Emergence of AI's Unique Artistic Styles and the Challenge of Copyright Protection

- [ ] From Imitation to Innovation: The Emergence of AI's Unique Artistic Styles and the Challenge of Copyright Protection | https://openaccess.thecvf.com/content/ICCV2025/html/Jia_From_Imitation_to_Innovation_The_Emergence_of_AIs_Unique_Artistic_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jia_From_Imitation_to_Innovation_The_Emergence_of_AIs_Unique_Artistic_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current legal frameworks consider AI-generated works eligible for copyright protection when they meet originality requirements and involve substantial human intellectual input. However, systematic legal standards and reliable evaluation methods for AI art copyrights are lacking. Through comprehensive analysis of legal precedents, we establish three essential criteria for determining distinctive artistic style: stylistic consistency, creative uniqueness, and expressive accuracy. To address these challenges, we introduce ArtBulb, an interpretable and quantifiable framework for AI art copyright judgment that combines a novel style description-based multimodal clustering method with multimodal large language models (MLLMs). We also present AICD, the first benchmark dataset for AI art copyright annotated by artists and legal experts. Experimental results demonstrate that ArtBulb outperforms existing models in both quantitative and qualitative evaluations. Our work aims to bridge the gap between the legal and technological communities and bring greater attention to the societal issue of AI art copyrights.

</details>

---

## 134. Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning

- [ ] Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Corvid_Improving_Multimodal_Large_Language_Models_Towards_Chain-of-Thought_Reasoning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Corvid_Improving_Multimodal_Large_Language_Models_Towards_Chain-of-Thought_Reasoning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have demonstrated exceptional performance in multimodal perception and understanding. However, leading open-source MLLMs exhibit significant limitations in complex and structured reasoning, particularly in tasks requiring deep reasoning for decision-making and problem-solving. In this work, we present Corvid, an MLLM with enhanced chain-of-thought (CoT) reasoning capabilities. Architecturally, Corvid incorporates a hybrid vision encoder for informative visual representation and a meticulously designed connector (GateMixer) to facilitate cross-modal alignment. To enhance Corvid's CoT reasoning capabilities, we introduce MCoT-Instruct-287K, a high-quality multimodal CoT instruction-following dataset, refined and standardized from diverse public reasoning sources. Leveraging this dataset, we fine-tune Corvid with a two-stage CoT-formatted training approach to progressively enhance its step-by-step reasoning abilities. Furthermore, we propose an effective inference-time scaling strategy that enables Corvid to mitigate over-reasoning and under-reasoning through self-verification. Extensive experiments demonstrate that Corvid outperforms existing o1-like MLLMs and state-of-the-art MLLMs with similar parameter scales, with notable strengths in mathematical reasoning and science problem-solving. Project page: https://mm-vl.github.io/corvid.

</details>

---

## 135. Multimodal LLM Guided Exploration and Active Mapping using Fisher Information

- [ ] Multimodal LLM Guided Exploration and Active Mapping using Fisher Information | https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Multimodal_LLM_Guided_Exploration_and_Active_Mapping_using_Fisher_Information_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Multimodal_LLM_Guided_Exploration_and_Active_Mapping_using_Fisher_Information_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present an active mapping system that could plan for long-horizon exploration goals and short-term actions with a 3D Gaussian Splatting (3DGS) representation. Existing methods either did not take advantage of recent developments in multimodal Large Language Models (LLM) or did not consider challenges in localization uncertainty which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based algorithm. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.

</details>

---

## 136. Referring to Any Person

- [ ] Referring to Any Person | https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Referring_to_Any_Person_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Referring_to_Any_Person_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Humans are undoubtedly the most important participants in computer vision, and the ability to detect any individual given a natural language description, a task we define as referring to any person, holds substantial practical value. However, we find that existing models generally fail to achieve real-world usability, and current benchmarks are limited by their focus on one-to-one referring, that hinder progress in this area. In this work, we revisit this task from three critical perspectives: task definition, dataset design, and model architecture. We first identify five aspects of referable entities and three distinctive characteristics of this task. Next, we introduce HumanRef, a novel dataset designed to tackle these challenges and better reflect real-world applications. From a model design perspective, we integrate a multimodal large language model with an object detection framework, constructing a robust referring model named RexSeek. Experimental results reveal that state-of-the-art models, which perform well on commonly used benchmarks like RefCOCO/+/g, struggle with HumanRef due to their inability to detect multiple individuals. In contrast, RexSeek not only excels in human referring but also generalizes effectively to common object referring, making it broadly applicable across various perception tasks.

</details>

---

## 137. Token-Efficient VLM: High-Resolution Image Understanding via Dynamic Region Proposal

- [ ] Token-Efficient VLM: High-Resolution Image Understanding via Dynamic Region Proposal | https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) excel at visual understanding by leveraging pretrained image encoders to generate visual tokens. However, they struggle with high-resolution images and zoomed-in regions due to the computational burden and token redundancy of uniform patch-based processing, often leading to the loss of critical details. To address these challenges, we propose Token-Efficient Vision Language Model (TEVA), a novel framework that detects key regions and applies dynamic patch sampling to efficiently capture fine-grained details while preserving global context. Our approach first identifies subject-oriented regions using an adaptive detection strategy. Then, a dynamic patch sampling mechanism selects and arranges patches at varying scales, ensuring efficient processing without increasing token count. Extensive experiments demonstrate that Token-Efficient Vision Language Model (TEVA) significantly enhances VLM performance in handling visual details, seamlessly integrating with various decoders and LLMs.

</details>

---

## 138. CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting

- [ ] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting | https://openaccess.thecvf.com/content/ICCV2025/html/Jiao_CLIP-GS_Unifying_Vision-Language_Representation_with_3D_Gaussian_Splatting_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jiao_CLIP-GS_Unifying_Vision-Language_Representation_with_3D_Gaussian_Splatting_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent works in 3D representation learning and multimodal pre-training have made remarkable progress. However, typically multimodal 3D models are only capable of handling point clouds. Compared to the emerging 3D representation technique, 3D Gaussian Splatting (3DGS), the spatially sparse point cloud cannot depict the texture information of 3D objects, resulting in inferior reconstruction capabilities. This limitation constrains the potential of point cloud-based 3D multimodal representation learning. In this paper, we present CLIP-GS, a novel multimodal representation learning framework grounded in 3DGS. We introduce the GS Tokenizer to generate serialized gaussian tokens, which are then processed through a series of transformer layers pre-initialized with weights from point cloud models, resulting in the 3DGS embeddings. CLIP-GS leverages contrastive loss between 3DGS and the visual-text embeddings of CLIP, and we introduce an image voting loss to guide the directionality and convergence of gradient optimization. Furthermore, we develop an efficient way to generate triplets of 3DGS, images, and text, facilitating CLIP-GS in learning unified multimodal representations. Leveraging the well-aligned multimodal representations, CLIP-GS demonstrates versatility and outperforms point cloud-based models on various 3D tasks, including multimodal retrieval, zero-shot, and few-shot classification.

</details>

---

## 139. From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning

- [ ] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning | https://openaccess.thecvf.com/content/ICCV2025/html/Jiao_From_Holistic_to_Localized_Local_Enhanced_Adapters_for_Efficient_Visual_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jiao_From_Holistic_to_Localized_Local_Enhanced_Adapters_for_Efficient_Visual_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Efficient Visual Instruction Fine-Tuning (EVIT) seeks to adapt Multimodal Large Language Models (MLLMs) to downstream tasks with minimal computational overhead. However, as task diversity and complexity increase, EVIT faces significant challenges in resolving data conflicts. To address this limitation, we propose the Dual Low-Rank Adaptation (Dual-LoRA), a holistic-to-local framework that enhances the adapter's capacity to address data conflict through dual structural optimization. Specifically, we utilize two subspaces: a skill space for stable, holistic knowledge retention, and a rank-rectified task space that locally activates the holistic knowledge. Additionally, we introduce Visual Cue Enhancement (VCE), a multi-level local feature aggregation module designed to enrich the vision-language projection with local details. Our approach is both memory- and time-efficient, requiring only 1.16xthe inference time of the standard LoRA method (with injection into the query and value projection layers), and just 73% of the inference time of a 4-expert LoRA-MoE. Extensive experiments on various downstream tasks and general MLLM benchmarks validate the effectiveness of our proposed methods.

</details>

---

## 140. Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models

- [ ] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Continual learning enables pre-trained generative vision-language models (VLMs) to incorporate knowledge from new tasks without retraining data from previous ones. Recent methods update a visual projector to translate visual information for new tasks, connecting pre-trained vision encoders with large language models. However, such adjustments may cause the models to prioritize visual inputs over language instructions, particularly learning tasks with repetitive types of textual instructions. To address the neglect of language instructions, we propose a novel framework that grounds the translation of visual information on instructions for language models. We introduce a mixture of visual projectors, each serving as a specialized visual-to-language translation expert based on the given instruction context to adapt to new tasks. To avoid using experts for irrelevant instruction contexts, we propose an expert recommendation strategy that reuses experts for tasks similar to those previously learned. Additionally, we introduce expert pruning to alleviate interference from the use of experts that cumulatively activated in previous tasks. Extensive experiments on diverse vision-language tasks demonstrate that our method outperforms existing continual learning approaches by generating instruction-following responses.

</details>

---

## 141. Feature Purification Matters: Suppressing Outlier Propagation for Training-Free Open-Vocabulary Semantic Segmentation

- [ ] Feature Purification Matters: Suppressing Outlier Propagation for Training-Free Open-Vocabulary Semantic Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Feature_Purification_Matters_Suppressing_Outlier_Propagation_for_Training-Free_Open-Vocabulary_Semantic_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Feature_Purification_Matters_Suppressing_Outlier_Propagation_for_Training-Free_Open-Vocabulary_Semantic_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Training-free open-vocabulary semantic segmentation has advanced with vision-language models like CLIP, which exhibit strong zero-shot abilities. However, CLIP's attention mechanism often wrongly emphasises specific image tokens, namely outliers, which results in irrelevant over-activation. Existing approaches struggle with these outliers that arise in intermediate layers and propagate through the model, ultimately degrading spatial perception. In this paper, we propose a Self-adaptive Feature Purifier framework (SFP) to suppress propagated outliers and enhance semantic representations for open-vocabulary semantic segmentation. Specifically, based on an in-depth analysis of attention responses between image and class tokens, we design a self-adaptive outlier mitigator to detect and mitigate outliers at each layer for propagated feature purification. In addition, we introduce a semantic-aware attention enhancer to augment attention intensity in semantically relevant regions, which strengthens the purified feature to focus on objects. Further, we introduce a hierarchical attention integrator to aggregate multi-layer attention maps to refine spatially coherent feature representations for final segmentation. Our proposed SFP enables robust outlier suppression and object-centric feature representation, leading to a more precise segmentation. Extensive experiments show that our method achieves state-of-the-art performance and surpasses existing methods by an average of 4.6% mIoU on eight segmentation benchmarks. The code will be released.

</details>

---

## 142. Details Matter for Indoor Open-vocabulary 3D Instance Segmentation

- [ ] Details Matter for Indoor Open-vocabulary 3D Instance Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Details_Matter_for_Indoor_Open-vocabulary_3D_Instance_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Details_Matter_for_Indoor_Open-vocabulary_3D_Instance_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unlike closed-vocabulary 3D instance segmentation that is often trained end-to-end, open-vocabulary 3D instance segmentation (OV-3DIS) often leverages vision-language models (VLMs) to generate 3D instance proposals and classify them. While various concepts have been proposed from existing research, we observe that these individual concepts are not mutually exclusive but complementary. In this paper, we propose a new state-of-the-art solution for OV-3DIS by carefully designing a recipe to combine the concepts together and refining them to address key challenges. Our solution follows the two-stage scheme: 3D proposal generation and instance classification. We employ robust 3D tracking-based proposal aggregation to generate 3D proposals and remove overlapped or partial proposals by iterative merging/removal. For the classification stage, we replace the standard CLIP model with Alpha-CLIP, which incorporates object masks as an alpha channel to reduce background noise and obtain object-centric representation. Additionally, we introduce the standardized maximum similarity (SMS) score to normalize text-to-proposal similarity, effectively filtering out false positives and boosting precision. Our framework achieves state-of-the-art performance on ScanNet200 and S3DIS across all AP and AR metrics, even surpassing an end-to-end closed-vocabulary method.

</details>

---

## 143. Zero-Shot Compositional Video Learning with Coding Rate Reduction

- [ ] Zero-Shot Compositional Video Learning with Coding Rate Reduction | https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Zero-Shot_Compositional_Video_Learning_with_Coding_Rate_Reduction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Zero-Shot_Compositional_Video_Learning_with_Coding_Rate_Reduction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose a novel zero-shot compositional video understanding method inspired by how young children efficiently learn new concepts and flexibly expand their existing knowledge framework. While recent large-scale visual language models (VLMs) have achieved remarkable advancements and demonstrated impressive performance improvements across various tasks, they require massive amounts of data and computational resources. However, despite their strong benchmark performance, they often fail to solve simple zero-shot composition tasks. Moreover, VLMs designed for video data demand even greater computational resources. We introduce a new video representation learning method inspired by human compositional learning to address these challenges. Specifically, we demonstrate that achieving zero-shot compositional learning requires effective representation learning that disentangles given data into meaningful semantic units. We propose a novel method that learns such disentangled representations based on an information-theoretic measure. By optimizing coding rate reduction, we successfully learn spatio-temporally disentangled features from videos, one of the most challenging data. Our approach significantly enhances compositional generalizability, demonstrating its effectiveness in zero-shot learning scenarios.

</details>

---

## 144. Dynamic Multi-Layer Null Space Projection for Vision-Language Continual Learning

- [ ] Dynamic Multi-Layer Null Space Projection for Vision-Language Continual Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Dynamic_Multi-Layer_Null_Space_Projection_for_Vision-Language_Continual_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Dynamic_Multi-Layer_Null_Space_Projection_for_Vision-Language_Continual_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLM) have emerged as a highly promising approach for Continual Learning (CL) due to their powerful generalized features. While adapter-based VLM can exploit both task-specific and task-agnostic features, current CL methods have largely overlooked the distinct and evolving parameter distributions in visual and language modalities, which are found crucial for effectively mitigating catastrophic forgetting.In this study, we find that the visual modality experiences a broader parameter distribution and greater variance during class increments than the textual modality, leading to higher vulnerability to forgetting.Consequently, we handle the branches of the two modalities asymmetrically. Specifically, we propose a Dynamic Multi-layer Null Space Projection (DMNSP) strategy and apply it only to the visual modality branch, while optimizing the language branch according to the original optimizer. DMNSP can restrict the update of visual parameters within the common subspace of multiple null spaces, further limiting the impact of non-zero residual terms. Simultaneously, combined with a dynamic projection coefficient, we can precisely control the magnitude of gradient projection to the null space, endowing the model with good stability and plasticity.Extensive experiments on TinyImageNet, CIFAR100 and ImageNet-R demonstrate that our method outperforms current approaches in accuracy and knowledge retention, setting a new standard for state-of-the-art performance in class incremental learning.

</details>

---

## 145. LEGION: Learning to Ground and Explain for Synthetic Image Detection

- [ ] LEGION: Learning to Ground and Explain for Synthetic Image Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Kang_LEGION_Learning_to_Ground_and_Explain_for_Synthetic_Image_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kang_LEGION_Learning_to_Ground_and_Explain_for_Synthetic_Image_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancements in generative technology have emerged as a double-edged sword. While offering powerful tools that enhance convenience, they also pose significant social concerns. As defenders, current synthetic image detection methods often lack artifact-level textual interpretability and are overly focused on image manipulation detection, and current datasets usually suffer from outdated generators and a lack of fine-grained annotations. In this paper, we introduce SynthScars, a high-quality and diverse dataset consisting of 12,236 fully synthetic images with human-expert annotations. It features 4 distinct image content types, 3 categories of artifacts, and fine-grained annotations covering pixel-level segmentation, detailed textual explanations, and artifact category labels. Furthermore, we propose LEGION (LEarning to Ground and explain for Synthetic Image detectiON), a multimodal large language model (MLLM)-based image forgery analysis framework that integrates artifact detection, segmentation, and explanation. Building upon this capability, we further explore LEGION as a controller, integrating it into image refinement pipelines to guide the generation of higher-quality and more realistic images. Extensive experiments show that LEGION outperforms existing methods across multiple benchmarks, particularly surpassing the second-best traditional expert on SynthScars by 3.31% in mIoU and 7.75% in F1 score. Moreover, the refined images generated under its guidance exhibit stronger alignment with human preferences. More information about LEGION can be found at https://opendatalab.github.io/LEGION.

</details>

---

## 146. Open-ended Hierarchical Streaming Video Understanding with Vision Language Models

- [ ] Open-ended Hierarchical Streaming Video Understanding with Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Hierarchical Streaming Video Understanding, a task that combines online temporal action localization with free-form description generation. Given the scarcity of datasets with hierarchical and fine-grained temporal annotations, we demonstrate that LLMs can effectively group atomic actions into higher-level events, enriching existing datasets.We then propose OpenHOUSE (Open-ended Hierarchical Online Understanding System for Events), which extends streaming action perception beyond action classification. OpenHOUSE features a specialized streaming module that accurately detects boundaries between closely adjacent actions, nearly doubling the performance of direct extensions of existing methods.We envision the future of streaming action perception in the integration of powerful generative models, with OpenHOUSE representing a key step in that direction.

</details>

---

## 147. SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference

- [ ] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference | https://openaccess.thecvf.com/content/ICCV2025/html/Khaki_SparseVILA_Decoupling_Visual_Sparsity_for_Efficient_VLM_Inference_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Khaki_SparseVILA_Decoupling_Visual_Sparsity_for_Efficient_VLM_Inference_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision language models have received increasing attention for their ability to integrate visual and textual understanding, with some capable of processing native-resolution images and long videos. While the capacity to process large visual data unlocks numerous downstream applications, it often introduces significant latency challenges, as the visual tokens dominate the resource consumption. In this work, we introduce SparseVILA, a novel method of query-aware token retrieval to dynamically accelerate the underlying LLM by pruning tokens in the prefill stage while attending to a sparse subset of visual tokens during the decoding phase. By decoupling the context and generation compression, we can migrate the majority of sparsity into the generation stage, enabling query-aware support for multi-turn conversation while achieving a 1.4x speedup on image benchmarks. This approach leads to +5.9% average accuracy improvements on image-centric benchmarks over previous works. Finally, SparseVILA enables efficient long-context/long-generation tasks by achieving a 3.6x and 1.7x speedup in prefill and decoding, respectively.

</details>

---

## 148. Analyzing Finetuning Representation Shift for Multimodal LLMs Steering

- [ ] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering | https://openaccess.thecvf.com/content/ICCV2025/html/Khayatan_Analyzing_Finetuning_Representation_Shift_for_Multimodal_LLMs_Steering_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Khayatan_Analyzing_Finetuning_Representation_Shift_for_Multimodal_LLMs_Steering_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal LLMs (MLLMs) have reached remarkable levels of proficiency in understanding multimodal inputs. However, understanding and interpreting the behavior of such complex models is a challenging task, not to mention the dynamic shifts that may occur during fine-tuning, or due to covariate shift between datasets. In this work, we apply concept-level analysis towards MLLM understanding. More specifically, we propose to map hidden states to interpretable visual and textual concepts. This enables us to more efficiently compare certain semantic dynamics, such as the shift from an original and fine-tuned model, revealing concept alteration and potential biases that may occur during fine-tuning. We also demonstrate the use of shift vectors to capture these concepts changes. These shift vectors allow us to recover fine-tuned concepts by applying simple, computationally inexpensive additive concept shifts in the original model. Finally, our findings also have direct applications for MLLM steering, which can be used for model debiasing as well as enforcing safety in MLLM output. All in all, we propose a novel, training-free, ready-to-use framework for MLLM behavior interpretability and control. Our implementation is publicly available.

</details>

---

## 149. CARIM: Caption-Based Autonomous Driving Scene Retrieval via Inclusive Text Matching

- [ ] CARIM: Caption-Based Autonomous Driving Scene Retrieval via Inclusive Text Matching | https://openaccess.thecvf.com/content/ICCV2025/html/Ki_CARIM_Caption-Based_Autonomous_Driving_Scene_Retrieval_via_Inclusive_Text_Matching_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ki_CARIM_Caption-Based_Autonomous_Driving_Scene_Retrieval_via_Inclusive_Text_Matching_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-video retrieval serves as a powerful tool for navigating vast video databases. This is particularly useful in autonomous driving to retrieve scenes from a text query to simulate and evaluate the driving system in desired scenarios. However, traditional ranking-based retrieval methods often return partial matches that do not satisfy all query conditions. To address this, we introduce Inclusive Text-to-Video Retrieval, which retrieves only videos that meet all specified conditions, regardless of additional irrelevant elements. We propose CARIM, a framework for driving scene retrieval that employs inclusive text matching. By utilizing Vision-Language Model (VLM) and Large Language Model (LLM) to generate compressed captions for driving scenes, we transform text-to-video retrieval into a more efficient text-to-text retrieval problem, eliminating modality mismatches and heavy annotation costs. We introduce a novel positive and negative data curation strategy and an attention-based scoring mechanism tailored for driving scene retrieval. Experimental results on the DRAMA dataset demonstrate that CARIM outperforms state-of-the-art retrieval methods, excelling in edge cases where traditional models fail.

</details>

---

## 150. ContextFace: Generating Facial Expressions from Emotional Contexts

- [ ] ContextFace: Generating Facial Expressions from Emotional Contexts | https://openaccess.thecvf.com/content/ICCV2025/html/Kim_ContextFace_Generating_Facial_Expressions_from_Emotional_Contexts_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kim_ContextFace_Generating_Facial_Expressions_from_Emotional_Contexts_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The task of generating 3D facial expressions given various situational contexts is important for applications such as virtual avatars or human-robot interactions. The task is, however, challenging not only because it requires a comprehensive understanding of emotion, expression and contexts, but also there rarely are datasets to support the task. We propose ContextFace, a Multi-modal Large Language Model (MLLM) fine-tuned to generate 3D facial expressions depending on complex situational contexts. To overcome the lack of datasets, we perform a context augmentation to existing emotion recognition datasets; we generate plausible situations and quotes from images and emotions to annotate the dataset. Next, we perform visual instruction tuning of MLLMs on context-augmented datasets to boost their capability of visual synthesis from emotions. Experiments show a superior performance of ContextFace in the zero-shot evaluation of contextual emotion recognition. A qualitative evaluation shows that our method generates expressions consistent with diverse contexts and performs complex emotion reasoning, e.g., speculative generation of expressions of occluded faces through interactive prompting.

</details>

---

## 151. CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models

- [ ] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Kim_CapeLLM_Support-Free_Category-Agnostic_Pose_Estimation_with_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kim_CapeLLM_Support-Free_Category-Agnostic_Pose_Estimation_with_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Category-agnostic pose estimation (CAPE) has traditionally relied on support images with annotated keypoints, a process that is often cumbersome and may fail to fully capture the necessary correspondences across diverse object categories. Recent efforts have explored the use of text queries, leveraging their enhanced stability and generalization capabilities. However, existing approaches often remain constrained by their reliance on support queries, their failure to fully utilize the rich priors embedded in pre-trained large language models, and the limitations imposed by their parametric distribution assumptions. To address these challenges, we introduce CapeLLM, the first multimodal large language model (MLLM) designed for CAPE. Our method only employs query image and detailed text descriptions as an input to estimate category-agnostic keypoints. Our method encompasses effective training strategies and carefully designed instructions for applying the MLLM to CAPE. Moreover, we propose an inference mechanism that further enhances the reasoning process for unseen keypoints. while flexibly modeling their underlying spatial distribution and uncertainty, allowing for adaptive refinement based on contextual cues. We conducted extensive experiments to apply the MLLM to CAPE effectively, focusing not only on the model architecture and prompt design but also on ensuring robustness across input variations. Our approach sets a new state-of-the-art on the MP-100 benchmark in the 1-shot and even 5-shot setting, marking a significant advancement in the field of category-agnostic pose estimation. Code is available https://github.com/Junhojuno/CapeLLM

</details>

---

## 152. Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing

- [ ] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing | https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Early_Timestep_Zero-Shot_Candidate_Selection_for_Instruction-Guided_Image_Editing_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Early_Timestep_Zero-Shot_Candidate_Selection_for_Instruction-Guided_Image_Editing_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite recent advances in diffusion models, achieving reliable image generation and editing results remains challenging due to the inherent diversity induced by stochastic noise in the sampling process. Particularly, instruction-guided image editing with diffusion models offers user-friendly editing capabilities, yet editing failures, such as background distortion, frequently occur across different attempts. Users often resort to trial and error, adjusting seeds or prompts to achieve satisfactory results, which is inefficient.While seed selection methods exist for Text-to-Image (T2I) generation, they depend on external verifiers, limiting their applicability, and evaluating multiple seeds increases computational complexity, reducing practicality.To address this, we first establish a new multiple-seed-based image editing baseline using background consistency scores, achieving Best-of-N performance without supervision. Building on this, we introduce ELECT (Early-timestep Latent Evaluation for Candidate Selection), a zero-shot framework that selects reliable seeds by estimating background mismatches at early diffusion timesteps, identfying the seed that retains the background while modifying only the foreground. ELECT ranks seed candidates by a background inconsistency score, filtering unsuitable samples early based on background consistency while fully preserving editability.Beyond standalone seed selection, ELECT integrates into instruction-guided editing pipelines and extends to Multimodal Large-Language Models (MLLMs) for joint seed + prompt selection, further improving results when seed selection alone is insufficient. Experiments show that ELECT reduces computational costs (by 41% on average and up to 61%) while improving background consistency and instruction adherence, achieving around 40% success rates in previously failed cases--without any external supervision or training.

</details>

---

## 153. Free2Guide: Training-Free Text-to-Video Alignment using Image LVLM

- [ ] Free2Guide: Training-Free Text-to-Video Alignment using Image LVLM | https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Free2Guide_Training-Free_Text-to-Video_Alignment_using_Image_LVLM_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Free2Guide_Training-Free_Text-to-Video_Alignment_using_Image_LVLM_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models have achieved impressive results in generative tasks for text-to-video (T2V) synthesis. However, achieving accurate text alignment in T2V generation remains challenging due to the complex temporal dependencies across frames. Existing reinforcement learning (RL)-based approaches to enhance text alignment often require differentiable reward functions trained for video, hindering their scalability and applicability. In this paper, we propose Free^2Guide, a novel gradient-free and training-free framework for aligning generated videos with text prompts. Specifically, leveraging principles from path integral control, Free^2Guide approximates guidance for diffusion models using non-differentiable reward functions, thereby enabling the integration of powerful black-box Large Vision-Language Models (LVLMs) as reward models. To enable image-trained LVLMs to assess text-to-video alignment, we leverage stitching between video frames and use system prompts to capture sequential attributions. Our framework supports the flexible ensembling of multiple reward models to synergistically enhance alignment without significant computational overhead. Experimental results confirm that Free^2Guide using image-trained LVLMs significantly improves text-to-video alignment, thereby enhancing the overall video quality. Our results and code are available at https://free2guide.github.io/

</details>

---

## 154. Fuzzy Contrastive Decoding to Alleviate Object Hallucination in Large Vision-Language Models

- [ ] Fuzzy Contrastive Decoding to Alleviate Object Hallucination in Large Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Fuzzy_Contrastive_Decoding_to_Alleviate_Object_Hallucination_in_Large_Vision-Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Fuzzy_Contrastive_Decoding_to_Alleviate_Object_Hallucination_in_Large_Vision-Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) often exhibit object hallucination, a phenomenon where models generate descriptions of non-existent objects within images. Prior methods have sought to mitigate this issue by adjusting model logits to reduce linguistic bias, but they often lack precise control over visual uncertainty, sometimes exacerbating hallucinations instead of mitigating them. To address this limitation, we propose a novel decoding strategy called fuzzy contrastive decoding (FuzzyCD) that uses Takagi-Sugeno fuzzy inference to refine hallucination control. FuzzyCD adaptively assigns weights to high-hallucination logits while mitigating unnecessary linguistic bias. Specifically, it transforms the log-probabilities of top-1 tokens from both standard and hallucination logits into a confidence linguistic fuzzy set. Through Takagi-Sugeno fuzzy inference, it dynamically adjusts hallucination logits to prevent the model from over-relying on spurious linguistic patterns. Experimental results on object hallucination datasets demonstrate that hallucination is mitigated by 11%p compared to conventional LVLMs. In-depth analyses highlight the effectiveness of FuzzyCD in enhancing the reliability of vision-language models.

</details>

---

## 155. Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation

- [ ] Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation | https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Leveraging_the_Power_of_MLLMs_for_Gloss-Free_Sign_Language_Translation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Leveraging_the_Power_of_MLLMs_for_Gloss-Free_Sign_Language_Translation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Sign language translation (SLT) is a challenging task that involves translating sign language images into spoken language. For SLT models to perform this task successfully, they must bridge the modality gap and identify subtle variations in sign language components to understand their meanings accurately. To address these challenges, we propose a novel gloss-free SLT framework called Multimodal Sign Language Translation (MMSLT), which leverages the representational capabilities of off-the-shelf multimodal large language models (MLLMs). Specifically, we use MLLMs to generate detailed textual descriptions of sign language components. Then, through our proposed multimodal-language pre-training module, we integrate these description features with sign video features to align them within the spoken sentence space. Our approach achieves state-of-the-art performance on benchmark datasets PHOENIX14T and CSL-Daily, highlighting the potential of MLLMs to be utilized effectively in SLT. Code is available at https://github.com/helpmeIamnewbie/MMSLT.

</details>

---

## 156. Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval

- [ ] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval | https://openaccess.thecvf.com/content/ICCV2025/html/Ko_Bidirectional_Likelihood_Estimation_with_Multi-Modal_Large_Language_Models_for_Text-Video_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ko_Bidirectional_Likelihood_Estimation_with_Multi-Modal_Large_Language_Models_for_Text-Video_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-Video Retrieval aims to find the most relevant text (or video) candidate given a video (or text) query from large-scale online databases. Recent work leverages multi-modal large language models (MLLMs) to improve retrieval, especially for long or complex query-candidate pairs. However, we observe that the naive application of MLLMs, i.e., retrieval based on candidate likelihood, introduces candidate prior bias, favoring candidates with inherently higher priors over those more relevant to the query. To this end, we propose a novel retrieval framework, Bidirectional Likelihood Estimation with MLLM (BLiM), which leverages both query and candidate likelihoods by training the model to generate text from a given video as well as video features from a given text. Furthermore, we introduce Candidate Prior Normalization (CPN), a simple yet effective training-free score calibration module designed to mitigate candidate prior bias in candidate likelihood. On four Text-Video Retrieval benchmarks, our BLiM equipped with CPN outperforms previous state-of-the-art models by 6.4 R@1 on average, effectively alleviating candidate prior bias and emphasizing query-candidate relevance. Our in-depth analysis across various multi-modal tasks beyond retrieval highlights the broad applicability of CPN which enhances visual understanding by reducing reliance on textual priors. Code is available at \href https://github.com/mlvlab/BLiM  https://github.com/mlvlab/BLiM .

</details>

---

## 157. Learning Interpretable Queries for Explainable Image Classification with Information Pursuit

- [ ] Learning Interpretable Queries for Explainable Image Classification with Information Pursuit | https://openaccess.thecvf.com/content/ICCV2025/html/Kolek_Learning_Interpretable_Queries_for_Explainable_Image_Classification_with_Information_Pursuit_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kolek_Learning_Interpretable_Queries_for_Explainable_Image_Classification_with_Information_Pursuit_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Information Pursuit (IP) is a recently introduced learning framework to construct classifiers that are interpretable-by-design. Given a set of task-relevant and interpretable data queries, IP selects a small subset of the most informative queries and makes predictions based on the gathered query-answer pairs. However, a key limitation of IP is its dependency on task-relevant interpretable queries, which typically require considerable data annotation and curation efforts. While previous approaches have explored using general-purpose large language models to generate these query sets, they rely on prompt engineering heuristics and often yield suboptimal query sets, resulting in a performance gap between IP and non-interpretable black-box predictors. In this work, we propose parameterizing IP queries as a learnable dictionary defined in the latent space of vision-language models such as CLIP. We formulate an optimization objective to learn IP queries and propose an alternating optimization algorithm that shares appealing connections with classic sparse dictionary learning algorithms. Our learned dictionary outperforms baseline methods based on handcrafted or prompted dictionaries across several image classification benchmarks.

</details>

---

## 158. Embodied Navigation with Auxiliary Task of Action Description Prediction

- [ ] Embodied Navigation with Auxiliary Task of Action Description Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Kondoh_Embodied_Navigation_with_Auxiliary_Task_of_Action_Description_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kondoh_Embodied_Navigation_with_Auxiliary_Task_of_Action_Description_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The field of multimodal robot navigation in indoor environments has garnered significant attention in recent years. However, as tasks and methods become more advanced, the action decision systems tend to become more complex and operate as black-boxes. For a reliable system, the ability to explain or describe its decisions is crucial; however, there tends to be a trade-off in that explainable systems cannot outperform non-explainable systems in terms of performance. In this paper, we propose incorporating the task of describing actions in language into the reinforcement learning of navigation as an auxiliary task. Existing studies have found it difficult to incorporate describing actions into reinforcement learning due to the absence of ground-truth data. We address this issue by leveraging knowledge distillation from pre-trained description generation models, such as vision-language models. We comprehensively evaluate our approach across various navigation tasks, demonstrating that it can describe actions while attaining high navigation performance. Furthermore, it achieves state-of-the-art performance in the particularly challenging multimodal navigation task of semantic audio-visual navigation.

</details>

---

## 159. VLR-Driver: Large Vision-Language-Reasoning Models for Embodied Autonomous Driving

- [ ] VLR-Driver: Large Vision-Language-Reasoning Models for Embodied Autonomous Driving | https://openaccess.thecvf.com/content/ICCV2025/html/Kong_VLR-Driver_Large_Vision-Language-Reasoning_Models_for_Embodied_Autonomous_Driving_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kong_VLR-Driver_Large_Vision-Language-Reasoning_Models_for_Embodied_Autonomous_Driving_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rise of embodied intelligence and multi-modal large language models has led to exciting advancements in the field of autonomous driving, establishing it as a prominent research focus in both academia and industry. However, when confronted with intricate and ambiguous traffic scenarios, the lack of logical reasoning and cognitive decision-making capabilities remains the primary challenge impeding the realization of embodied autonomous driving. Although Vision Language Models (VLMs) have enhanced the deep semantic understanding of autonomous driving systems, they exhibit notable limitations in decision explainability when handling rare and long-tail traffic scenarios. In this paper, we propose VLR-Driver, a novel multi-modal Vision-Language-Reasoning (VLR) framework based on Chain of Thought (CoT) for embodied autonomous driving. The framework employs a spatiotemporal CoT reasoning approach to recursively analyze potential safety risks and driving intentions of other agents, thereby delivering an efficient and transparent decision-making process. Furthermore, we construct a multi-modal reasoning-decision dataset to support the advancement of hierarchical reasoning of VLMs in autonomous driving. Closed-loop experiments conducted in CARLA demonstrate that the VLR-Driver significantly outperforms state-of-the-art end-to-end methods. Notably, key metrics such as driving score improved by 17.5%, while the success rate improved by 22.2%, offering a more transparent, reliable, and secure solution for autonomous driving systems.

</details>

---

## 160. RoboAnnotatorX: A Comprehensive and Universal Annotation Framework for Accurate Understanding of Long-horizon Robot Demonstration

- [ ] RoboAnnotatorX: A Comprehensive and Universal Annotation Framework for Accurate Understanding of Long-horizon Robot Demonstration | https://openaccess.thecvf.com/content/ICCV2025/html/Kou_RoboAnnotatorX_A_Comprehensive_and_Universal_Annotation_Framework_for_Accurate_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kou_RoboAnnotatorX_A_Comprehensive_and_Universal_Annotation_Framework_for_Accurate_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in robotics have produced numerous valuable large-scale demonstration datasets, yet their potential remains underutilized due to annotation limitations. Current datasets often suffer from sparse temporal annotations, and inconsistent labeling granularity, particularly for complex long-horizon demonstrations. Traditional manual annotation methods are expensive and poorly scalable while existing automated methods struggle with temporal coherence and semantic richness across extended demonstrations. For this, we propose RoboAnnotatorX, a reliable annotation tool that enhances multimodal large language model to generate high-quality, context-rich annotations for complex long-horizon demonstrations. Specifically, we introduce a multi-scale token-efficient encoder to maintain computational efficiency while simultaneously capturing fine-grained visual details and preserving temporal information by jointly integrating scene-level anchoring, clip-level temporal dynamics, and video-level global modeling. We further construct a comprehensive dataset RoboX-VQA that synthesizes diverse QA pairs from both real-world and simulated data, bridging the significant domain gap in robotics demonstrations. Moreover, we leverage a curriculum-inspired three-stage training to progressively develop capabilities from basic visual perception to sophisticated temporal reasoning. Extensive experiments demonstrate that RoboAnnotatorX significantly outperforms existing approaches in annotation quality and exhibits strong generalization across diverse robotic environments, helping unlock the full potential of existing robotic datasets.

</details>

---

## 161. ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition

- [ ] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition | https://openaccess.thecvf.com/content/ICCV2025/html/Kundu_ProbRes_Probabilistic_Jump_Diffusion_for_Open-World_Egocentric_Activity_Recognition_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kundu_ProbRes_Probabilistic_Jump_Diffusion_for_Open-World_Egocentric_Activity_Recognition_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-world egocentric activity recognition poses a fundamental challenge due to its unconstrained nature, requiring models to infer unseen activities from an expansive, partially observed search space. We introduce ProbRes, a Probabilistic Residual search framework based on jump-diffusion that efficiently navigates this space by balancing prior-guided exploration with likelihood-driven exploitation. Our approach integrates structured commonsense priors to construct a semantically coherent search space, adaptively refines predictions using Vision-Language Models (VLMs) and employs a stochastic search mechanism to locate high-likelihood activity labels while minimizing exhaustive enumeration efficiently. We systematically evaluate ProbRes across multiple openness levels (L0-L3), demonstrating its adaptability to increasing search space complexity. In addition to achieving state-of-the-art performance on benchmark datasets (GTEA Gaze, GTEA Gaze+, EPIC-Kitchens, and Charades-Ego), we establish a clear taxonomy for open-world recognition, delineating the challenges and methodological advancements necessary for egocentric activity understanding. Our results highlight the importance of structured search strategies, paving the way for scalable and efficient open-world activity recognition.

</details>

---

## 162. D-Attn: Decomposed Attention for Large Vision-and-Language Model

- [ ] D-Attn: Decomposed Attention for Large Vision-and-Language Model | https://openaccess.thecvf.com/content/ICCV2025/html/Kuo_D-Attn_Decomposed_Attention_for_Large_Vision-and-Language_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kuo_D-Attn_Decomposed_Attention_for_Large_Vision-and-Language_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-and-language models (LVLMs) have traditionally integrated visual and textual tokens by concatenating them into a single homogeneous input for large language models (LLMs), thereby maximally preserving the pre-trained language capabilities. However, this constrained architecture for visual and textual tokens restricts the design space for processing visual tokens, potentially leading to suboptimal performance and efficiency. In this paper, we propose Decomposed Attention (D-Attn), a more flexible attention architecture for LVLMs, which enables modification of visual token operations without affecting textual-to-textual attention. D-Attn decomposes the 1-D causal self-attention of LVLMs into visual-to-visual, textual-to-visual, and textual-to-textual attentions, and the visual and textual output tokens from the decomposed attentions are merged with a carefully derived weighting strategy, namely \alpha-weighting. Taking advantage of the flexibility, we are able to introduce two critical improvements in visual token processing while maintaining the capacity of pre-trained LLMs: 1) We rectify the biased positional encoding in textual-to-visual attention to boost visual understanding performance. 2) We diagonalize visual-to-visual attention to reduce computation complexity from \mathcal O (|V|^2) to \mathcal O (|V|) for |V| visual tokens without compromising performance. Extensive experiments and analysis validate the effectiveness of D-Attn, demonstrating significant improvements on multiple image benchmarks while significantly reducing computational costs (e.g., 5xfaster). Code will be available at https://github.com/bytedance/DecomposedAttention.

</details>

---

## 163. CE-FAM: Concept-Based Explanation via Fusion of Activation Maps

- [ ] CE-FAM: Concept-Based Explanation via Fusion of Activation Maps | https://openaccess.thecvf.com/content/ICCV2025/html/Kuroki_CE-FAM_Concept-Based_Explanation_via_Fusion_of_Activation_Maps_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Kuroki_CE-FAM_Concept-Based_Explanation_via_Fusion_of_Activation_Maps_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although saliency maps can highlight important regions to explain the reasoning behind image classification in artificial intelligence (AI), the meaning of these regions is left to the user's interpretation. In contrast, concept-based explanations decompose AI predictions into human-understandable concepts, clarifying their contributions. However, few methods can simultaneously reveal what concepts an image classifier learns, which regions are associated with them, and how they contribute to predictions.We propose a novel concept-based explanation method, Concept-based Explanation via Fusion of Activation Maps (CE-FAM). It employs a branched network that shares activation maps with an image classifier and learns to mimic the embeddings of a Vision and Language Model (VLM). The branch network predicts concepts in an image, and their corresponding regions are represented by a weighted sum of activation maps, with weights given by the gradients of the concept prediction scores. Their contributions are quantified based on their impact on the image classification score. Our method provides a general framework for identifying the concept regions and their contributions while leveraging VLM knowledge to handle arbitrary concepts without requiring an annotated dataset. Furthermore, we introduce a novel evaluation metric to assess the accuracy of the concept regions. Our qualitative and quantitative evaluations demonstrate our method outperforms existing approaches and excels in zero-shot inference for unseen concepts.

</details>

---

## 164. ViLU: Learning Vision-Language Uncertainties for Failure Prediction

- [ ] ViLU: Learning Vision-Language Uncertainties for Failure Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Lafon_ViLU_Learning_Vision-Language_Uncertainties_for_Failure_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lafon_ViLU_Learning_Vision-Language_Uncertainties_for_Failure_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reliable Uncertainty Quantification (UQ) and failure prediction remain open challenges for Vision-Language Models (VLMs). We introduce ViLU, a new Vision-Language Uncertainty quantification framework that contextualizes uncertainty estimates by leveraging all task-relevant textual representations. ViLU constructs an uncertainty-aware multi-modal representation by integrating the visual embedding, the predicted textual embedding, and an image-conditioned textual representation via cross-attention. Unlike traditional UQ methods based on loss prediction, ViLU trains an uncertainty predictor as a binary classifier to distinguish correct from incorrect predictions using a weighted binary cross-entropy loss, making it loss-agnostic. In particular, our proposed approach is well-suited for post-hoc settings, where only vision and text embeddings are available without direct access to the model itself. Extensive experiments on diverse datasets show the significant gains of our method compared to state-of-the-art failure prediction methods. We apply our method to standard classification datasets, such as ImageNet-1k, as well as large-scale image-caption datasets like CC12M and LAION-400M. Ablation studies highlight the critical role of our architecture and training in achieving effective uncertainty quantification.

</details>

---

## 165. When Schrodinger Bridge Meets Real-World Image Dehazing with Unpaired Training

- [ ] When Schrodinger Bridge Meets Real-World Image Dehazing with Unpaired Training | https://openaccess.thecvf.com/content/ICCV2025/html/Lan_When_Schrodinger_Bridge_Meets_Real-World_Image_Dehazing_with_Unpaired_Training_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lan_When_Schrodinger_Bridge_Meets_Real-World_Image_Dehazing_with_Unpaired_Training_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in unpaired dehazing, particularly those using GANs, show promising performance in processing real-world hazy images. However, these methods tend to face limitations due to the generator's limited transport mapping capability, which hinders the full exploitation of their effectiveness in unpaired training paradigms. To address these challenges, we propose DehazeSB, a novel unpaired dehazing framework based on the Schrodinger Bridge. By leveraging optimal transport (OT) theory, DehazeSB directly bridges the distributions between hazy and clear images. This enables optimal transport mappings from hazy to clear images in fewer steps, thereby generating high-quality results. To ensure the consistency of structural information and details in the restored images, we introduce detail-preserving regularization, which enforces pixel-level alignment between hazy inputs and dehazed outputs. Furthermore, we propose a novel prompt learning to leverage pre-trained CLIP models in distinguishing hazy images and clear ones, by learning a haze-aware vision-language alignment. Extensive experiments on multiple real-world datasets demonstrate our method's superiority.

</details>

---

## 166. Error Recognition in Procedural Videos using Generalized Task Graph

- [ ] Error Recognition in Procedural Videos using Generalized Task Graph | https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Error_Recognition_in_Procedural_Videos_using_Generalized_Task_Graph_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Error_Recognition_in_Procedural_Videos_using_Generalized_Task_Graph_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding user actions and their possible mistakes is essential for successful operation of task assistants. In this paper, we develop a unified framework for joint temporal action segmentation and error recognition (recognizing when and which type of error happens) in procedural task videos. We propose a Generalized Task Graph (GTG) whose nodes encode correct steps and background (task-irrelevant actions). We then develop a GTG-Video Alignment algorithm (GTG2Vid) to jointly segment videos into actions and detect frames containing errors. Given that it is infeasible to gather many videos and their annotations for different types of errors, we study a framework that only requires normal (error-free) videos during training. More specifically, we leverage large language models (LLMs) to obtain error descriptions and subsequently use video-language models (VLMs) to generate visually-aligned textual features, which we use for error recognition. We then propose an Error Recognition Module (ERM) to recognize the error frames predicted by GTG2Vid using the generated error features. By extensive experiments on two egocentric datasets of EgoPER and CaptainCook4D, we show that our framework outperforms other baselines on action segmentation, error detection and recognition.

</details>

---

## 167. MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models

- [ ] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Lee_MultiVerse_A_Multi-Turn_Conversation_Benchmark_for_Evaluating_Large_Vision_and_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lee_MultiVerse_A_Multi-Turn_Conversation_Benchmark_for_Evaluating_Large_Vision_and_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-and-Language Models (VLMs) have shown impressive capabilities on single-turn benchmarks, yet real-world applications often demand more intricate multi-turn dialogues. Existing multi-turn datasets (e.g, MMDU, ConvBench) only partially capture the breadth and depth of conversational scenarios encountered by users. In this work, we introduce MultiVerse, a novel multi-turn conversation benchmark featuring 647 dialogues--each averaging four turns--derived from a diverse set of 12 popular VLM evaluation benchmarks. With 484 tasks and 484 interaction goals, MultiVerse covers a wide range of topics, from factual knowledge and perception to advanced reasoning tasks such as mathematics and coding. To facilitate robust assessment, we propose a checklist-based evaluation method that leverages GPT-4o as the automated evaluator, measuring performance across 37 key aspects, including perceptual accuracy, linguistic clarity, and factual correctness. We evaluate 18 VLMs on MultiVerse, revealing that even the strongest models (e.g., GPT-4o) achieve only a 50% success rate in complex multi-turn conversations, highlighting the dataset's challenging nature. Notably, we find that providing full dialogue context significantly enhances performance for smaller or weaker models, emphasizing the importance of in-context learning. We believe MultiVerse is a landscape of evaluating multi-turn interaction abilities for VLMs.

</details>

---

## 168. PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior

- [ ] PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior | https://openaccess.thecvf.com/content/ICCV2025/html/Lee_PASTA_Part-Aware_Sketch-to-3D_Shape_Generation_with_Text-Aligned_Prior_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lee_PASTA_Part-Aware_Sketch-to-3D_Shape_Generation_with_Text-Aligned_Prior_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

A fundamental challenge in conditional 3D shape generation is to minimize the information loss and maximize the intention of user input. Existing approaches have predominantly focused on two types of isolated conditional signals, i.e., user sketches and text descriptions, each of which does not offer flexible control of the generated shape. In this paper, we introduce PASTA, the flexible approach that seamlessly integrates a user sketch and a text description for 3D shape generation. The key idea is to use text embeddings from a vision-language model to enrich the semantic representation of sketches. Specifically, these text-derived priors specify the part components of the object, compensating for missing visual cues from ambiguous sketches. In addition, we introduce ISG-Net which employs two types of graph convolutional networks: IndivGCN, which processes fine-grained details, and PartGCN, which aggregates these details into parts and refines the structure of objects. Extensive experiments demonstrate that PASTA outperforms existing methods in part-level editing and achieves state-of-the-art results in sketch-to-3D shape generation.

</details>

---

## 169. Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation

- [ ] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation | https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Perspective-Aware_Reasoning_in_Vision-Language_Models_via_Mental_Imagery_Simulation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Perspective-Aware_Reasoning_in_Vision-Language_Models_via_Mental_Imagery_Simulation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a framework for perspective-aware reasoning in vision-language models (VLMs) through mental imagery simulation. Perspective-taking, the ability to perceive an environment or situation from an alternative viewpoint, is a key benchmark for human-level visual understanding, essential for environmental interaction and collaboration with autonomous agents. Despite advancements in spatial reasoning within VLMs, recent research has shown that modern VLMs significantly lack perspective-aware reasoning capabilities and exhibit a strong bias toward egocentric interpretations. To bridge the gap between VLMs and human perception, we focus on the role of mental imagery, where humans perceive the world through abstracted representations that facilitate perspective shifts. Motivated by this, we propose a framework for perspective-aware reasoning, named Abstract Perspective Change (APC), that effectively leverages vision foundation models, such as object detection, segmentation, and orientation estimation, to construct scene abstractions and enable perspective changes. Our experiments on synthetic and real-image benchmarks, compared with various VLMs, demonstrate significant improvements in perspective-aware reasoning with our framework, further outperforming fine-tuned spatial reasoning models and novel-view-synthesis-based approaches.

</details>

---

## 170. HOLa: Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation

- [ ] HOLa: Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation | https://openaccess.thecvf.com/content/ICCV2025/html/Lei_HOLa_Zero-Shot_HOI_Detection_with_Low-Rank_Decomposed_VLM_Feature_Adaptation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lei_HOLa_Zero-Shot_HOI_Detection_with_Low-Rank_Decomposed_VLM_Feature_Adaptation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot human-object interaction (HOI) detection remains a challenging task, particularly in generalizing to unseen actions. Existing methods address this challenge by tapping Vision-Language Models (VLMs) to access knowledge beyond the training data. However, they either struggle to distinguish actions involving the same object or demonstrate limited generalization to unseen classes. In this paper, we introduce HOLa (Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation), a novel approach that both enhances generalization to unseen classes and improves action distinction. In training, HOLa decomposes VLM text features for given HOI classes via low-rank factorization, producing class-shared basis features and adaptable weights. These features and weights form a compact HOI representation that preserves shared information across classes, enhancing generalization to unseen classes. Subsequently, we refine action distinction by adapting weights for each HOI class and introducing human-object tokens to enrich visual interaction representations. To further distinguish unseen actions, we guide the weight adaptation with LLM-derived action regularization. Experimental results show that our method sets a new state-of-the-art across zero-shot HOI settings on HICO-DET, achieving an unseen-class mAP of 27.91 in the unseen-verb setting. Our code is available at https://github.com/ChelsieLei/HOLa.

</details>

---

## 171. The Scalability of Simplicity: Empirical Analysis of Vision-Language Learning with a Single Transformer

- [ ] The Scalability of Simplicity: Empirical Analysis of Vision-Language Learning with a Single Transformer | https://openaccess.thecvf.com/content/ICCV2025/html/Lei_The_Scalability_of_Simplicity_Empirical_Analysis_of_Vision-Language_Learning_with_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lei_The_Scalability_of_Simplicity_Empirical_Analysis_of_Vision-Language_Learning_with_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces SAIL, a single transformer unified multimodal large language model (MLLM) that integrates raw pixel encoding and language decoding within a singular architecture. Unlike existing modular MLLMs, which rely on a pre-trained vision transformer (ViT), SAIL eliminates the need for a separate vision encoder, presenting a more minimalist architecture design. Instead of introducing novel architectural components, SAIL adapts mix-attention mechanisms and multimodal rotary position embedding to better align with the distinct characteristics of visual and textual modalities. We systematically compare SAIL's properties-including scalability, cross-modal information flow patterns, and visual representation capabilities-with those of modular MLLMs. By scaling both training data and model size, SAIL achieves performance comparable to modular MLLMs. Notably, the removal of pretrained ViT components enhances SAIL's scalability and results in significantly different cross-modal information flow patterns. Moreover, SAIL demonstrates strong visual representation capabilities, achieving results on par with ViT-22B in vision tasks such as semantic segmentation. Code and models are available at https://github.com/bytedance/SAIL.

</details>

---

## 172. Open-Vocabulary HOI Detection with Interaction-aware Prompt and Concept Calibration

- [ ] Open-Vocabulary HOI Detection with Interaction-aware Prompt and Concept Calibration | https://openaccess.thecvf.com/content/ICCV2025/html/Lei_Open-Vocabulary_HOI_Detection_with_Interaction-aware_Prompt_and_Concept_Calibration_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lei_Open-Vocabulary_HOI_Detection_with_Interaction-aware_Prompt_and_Concept_Calibration_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open Vocabulary Human-Object Interaction (HOI) detection aims to detect interactions between humans and objects while generalizing to novel interaction classes beyond the training set. Current methods often rely on Vision and Language Models (VLMs) but face challenges due to suboptimal image encoders, as image-level pre-training does not align well with the fine-grained region-level interaction detection required for HOI. Additionally, effectively encoding textual descriptions of visual appearances remains difficult, limiting the model's ability to capture detailed HOI relationships. To address these issues, we propose INteraction-aware Prompting with Concept Calibration (INP-CC), an end-to-end open-vocabulary HOI detector that integrates interaction-aware prompts and concept calibration. Specifically, we propose an interaction-aware prompt generator that dynamically generates a compact set of prompts based on the input scene, enabling selective sharing among similar interactions. This approach directs the model's attention to key interaction patterns rather than generic image-level semantics, enhancing HOI detection. Furthermore, we refine HOI concept representations through language model-guided calibration, which helps distinguish diverse HOI concepts by investigating visual similarities across categories. A negative sampling strategy is also employed to improve inter-modal similarity modeling, enabling the model to better differentiate visually similar but semantically distinct actions. Extensive experimental results demonstrate that INP-CC significantly outperforms state-of-the-art models on the SWIG-HOI and HICO-DET datasets. Code is available at https://github.com/ltttpku/INP-CC.

</details>

---

## 173. AGO: Adaptive Grounding for Open World 3D Occupancy Prediction

- [ ] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Li_AGO_Adaptive_Grounding_for_Open_World_3D_Occupancy_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_AGO_Adaptive_Grounding_for_Open_World_3D_Occupancy_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-world 3D semantic occupancy prediction aims to generate a voxelized 3D representation from sensor inputs while recognizing both known and unknown objects. Transferring open-vocabulary knowledge from vision-language models (VLMs) offers a promising direction but remains challenging. However, methods based on VLM-derived 2D pseudo-labels with traditional supervision are limited by a predefined label space and lack general prediction capabilities. Direct alignment with pretrained image embeddings, on the other hand, often fails to achieve reliable performance because of inconsistent image and text representations in VLMs. To address these challenges, we propose AGO, a novel 3D occupancy prediction framework with adaptive grounding to handle diverse open-world scenarios. AGO first encodes surrounding images and class prompts into 3D and text embeddings, respectively, leveraging similarity-based grounding training with 3D pseudo-labels. Additionally, a modality adapter maps 3D embeddings into a space aligned with VLM-derived image embeddings, reducing modality gaps. Experiments on Occ3D-nuScenes show that AGO improves unknown object prediction in zero-shot and few-shot transfer while achieving state-of-the-art closed-world self-supervised performance, surpassing prior methods by 4.09 mIoU. Code is available at: https://github.com/EdwardLeeLPZ/AGO.

</details>

---

## 174. AIRA: Activation-Informed Low-Rank Adaptation for Large Models

- [ ] AIRA: Activation-Informed Low-Rank Adaptation for Large Models | https://openaccess.thecvf.com/content/ICCV2025/html/Li_AIRA_Activation-Informed_Low-Rank_Adaptation_for_Large_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_AIRA_Activation-Informed_Low-Rank_Adaptation_for_Large_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Low-Rank Adaptation (LoRA) is a widely used method for efficiently fine-tuning large models by introducing low-rank matrices into weight updates. However, existing LoRA techniques fail to account for activation information, such as outliers, which significantly impact model performance. This omission leads to suboptimal adaptation and slower convergence. To address this limitation, we present Activation-Informed Low-Rank Adaptation (AIRA), a novel approach that integrates activation information into initialization, training, and rank assignment to enhance model performance. Specifically, AIRA introduces: (1) Outlier-weighted SVD decomposition to reduce approximation errors in low-rank weight initialization, (2) Outlier-driven dynamic rank assignment using offline optimization for better layer-wise adaptation, and (3) Activation-informed training to amplify updates on significant weights. This cascaded activation-informed paradigm enables faster convergence and fewer fine-tuned parameters while maintaining high performance. Extensive experiments on multiple large models demonstrate that AIRA outperforms state-of-the-art LoRA variants, achieving superior performance-efficiency trade-offs in vision-language instruction tuning, few-shot learning, and image generation. Codes are available at https://github.com/lliai/LoRA-Zoo.

</details>

---

## 175. Advancing Textual Prompt Learning with Anchored Attributes

- [ ] Advancing Textual Prompt Learning with Anchored Attributes | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Advancing_Textual_Prompt_Learning_with_Anchored_Attributes_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Advancing_Textual_Prompt_Learning_with_Anchored_Attributes_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Textual-based prompt learning methods primarily employ multiple learnable soft prompts and hard class tokens in a cascading manner as text inputs, aiming to align image and text (category) spaces for downstream tasks. However, current training is restricted to aligning images with predefined known categories and cannot be associated with unknown categories. In this work, we propose utilizing universal attributes as a bridge to enhance the alignment between images and unknown categories. Specifically, we introduce an Attribute-anchored Textual Prompt learning method for vision-language models, named ATPrompt. This approach expands the learning space of soft prompts from the original one-dimensional category level into the multi-dimensional attribute level by incorporating multiple attribute tokens into the learnable soft prompts. Through this modification, we transform the text prompt from a category-centric form to an attribute-category hybrid form. Additionally, we introduce a straightforward differentiable attribute search method to identify representative and suitable attributes for downstream tasks. As an easy-to-use plug-in technique, ATPrompt can seamlessly replace the existing basic prompt format in textual-based methods, providing general improvements at a negligible computational cost. Extensive experiments across 11 datasets validate the effectiveness of our method. Code is publicly available at https://github.com/zhengli97/ATPrompt.

</details>

---

## 176. Benefit From Seen: Enhancing Open-Vocabulary Object Detection by Bridging Visual and Textual Co-Occurrence Knowledge

- [ ] Benefit From Seen: Enhancing Open-Vocabulary Object Detection by Bridging Visual and Textual Co-Occurrence Knowledge | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Benefit_From_Seen_Enhancing_Open-Vocabulary_Object_Detection_by_Bridging_Visual_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Benefit_From_Seen_Enhancing_Open-Vocabulary_Object_Detection_by_Bridging_Visual_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-Vocabulary Object Detection (OVOD) aims to localize and recognize objects from both known and novel categories. However, existing methods rely heavily on internal knowledge from Vision-Language Models (VLMs), restricting their generalization to unseen categories due to limited contextual understanding. To address this, we propose CODet, a plug-and-play framework that enhances OVOD by integrating object co-occurrence ---- a form of external contextual knowledge pervasive in real-world scenes. Specifically, CODet extracts visual co-occurrence patterns from images, aligns them with textual dependencies validated by Large Language Models (LLMs), and injects contextual co-occurrence pseudo-labels as external knowledge to guide detection. Without architectural changes, CODet consistently improves five state-of-the-art VLM-based detectors across two benchmarks, achieving notable gains (up to +2.3 AP on novel categories). Analyses further confirm its ability to encode meaningful contextual guidance, advancing open-world perception by bridging visual and textual co-occurrence knowledge.

</details>

---

## 177. Breaking the Encoder Barrier for Seamless Video-Language Understanding

- [ ] Breaking the Encoder Barrier for Seamless Video-Language Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Breaking_the_Encoder_Barrier_for_Seamless_Video-Language_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Breaking_the_Encoder_Barrier_for_Seamless_Video-Language_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Most Video-Large Language Models (Video-LLMs) adopt an encoder-decoder framework, where a vision encoder extracts frame-wise features for processing by a language model. However, this approach incurs high computational costs, introduces resolution biases, and struggles to capture fine-grained multimodal interactions. To overcome these limitations, we propose ELVA, an encoder-free Video-LLM that directly models nuanced video-language interactions without relying on a vision encoder. ELVA employs token merging to construct a bottom-up hierarchical representation and incorporates a video guidance supervisor for direct spatiotemporal representation learning. Additionally, a hybrid-resolution mechanism strategically integrates high- and low-resolution frames as inputs to achieve an optimal balance between performance and efficiency. With only 7M publicly available video-text pairs, ELVA achieves competitive performance compared to encoder-based Video-LLMs while reducing FLOPs by up to 95% and inference latency by 92%, offering a scalable and efficient solution for real-time video understanding.

</details>

---

## 178. Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios

- [ ] Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of generative models, highly realistic image synthesis has posed new challenges to digital security and media credibility. Although AI-generated image detection methods have partially addressed these concerns, a substantial research gap remains in evaluating their performance under complex real-world conditions. This paper introduces the Real-World Robustness Dataset (RRDataset) for comprehensive evaluation of detection models across three dimensions: 1) Scenario Generalization - RRDataset encompasses high-quality images from seven major scenarios (War & Conflict, Disasters & Accidents, Political & Social Events, Medical & Public Health, Culture & Religion, Labor & Production, and everyday life), addressing existing dataset gaps from a content perspective. 2) Internet Transmission Robustness - examining detector performance on images that have undergone multiple rounds of sharing across various social media platforms.3) Re-digitization Robustness - assessing model effectiveness on images altered through four distinct re-digitization methods.We benchmarked 17 detectors and 10 vision-language models (VLMs) on RRDataset and conducted a large-scale human study involving 192 participants to investigate human few-shot learning capabilities in detecting AI-generated images. The benchmarking results reveal the limitations of current AI detection methods under real-world conditions and underscore the importance of drawing on human adaptability to develop more robust detection algorithms. Our dataset is publicly available at: https://zenodo.org/records/14963880.

</details>

---

## 179. CoA-VLA: Improving Vision-Language-Action Models via Visual-Text Chain-of-Affordance

- [ ] CoA-VLA: Improving Vision-Language-Action Models via Visual-Text Chain-of-Affordance | https://openaccess.thecvf.com/content/ICCV2025/html/Li_CoA-VLA_Improving_Vision-Language-Action_Models_via_Visual-Text_Chain-of-Affordance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_CoA-VLA_Improving_Vision-Language-Action_Models_via_Visual-Text_Chain-of-Affordance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Robot foundation models, particularly Vision-Language-Action (VLA) models, have garnered significant attention for their ability to enhance robot policy learning, greatly improving robot's generalization and robustness. OpenAI's recent model, O1, showcased impressive capabilities in solving complex problems by utilizing extensive reasoning chains. This prompts an important question: can robot models achieve better performance in multi-task, complex environments by reviewing prior observations and then providing task-specific reasoning to guide action prediction?In this paper, we introduce Chain-of-Affordance (CoA-VLA), a novel approach to scaling robot models by incorporating reasoning in the format of sequential robot affordances to facilitate task completion. Specifically, we prompt the model to consider the following four types of affordances before taking action: (1) object affordance-- what object to manipulate and where it is; (2) grasp affordance -- the specific object part to grasp; (3) spatial affordance -- the optimal space to place the object; and (4) movement affordance-- the collision-free path for movement. We further transform each affordance into two prompting formats: visual affordance and textual affordance. We introduce a novel vision-language co-injection module that integrates this knowledge into the policy network. This allows the robot to leverage essential contextual information during action inference, resulting in improved precision and robustness. Our experiments demonstrate that CoA-VLA outperforms state-of-the-art robot foundation models, including OpenVLA and Octo, on a variety of tasks. Furthermore, CoA-VLA exhibits strong generalization capabilities, including recognizing unseen object poses, identifying free space, and avoiding obstacles in novel environments.

</details>

---

## 180. FiVE-Bench: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models

- [ ] FiVE-Bench: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models | https://openaccess.thecvf.com/content/ICCV2025/html/Li_FiVE-Bench_A_Fine-grained_Video_Editing_Benchmark_for_Evaluating_Emerging_Diffusion_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_FiVE-Bench_A_Fine-grained_Video_Editing_Benchmark_for_Evaluating_Emerging_Diffusion_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Numerous text-to-video (T2V) editing methods have emerged recently, but the lack of a standardized benchmark for fair evaluation has led to inconsistent claims and an inability to assess model sensitivity to hyperparameters. Fine-grained video editing is crucial for enabling precise, object-level modifications while maintaining context and temporal consistency. To address this, we introduce FiVE-Bench, a Fine-grained Video Editing Benchmark for evaluating emerging diffusion and rectified flow models. Our benchmark includes 74 real-world videos and 26 generated videos, featuring 6 fine-grained editing types, 420 object-level editing prompt pairs, and their corresponding masks. Additionally, we adapt the latest rectified flow (RF) T2V generation models--Pyramid-Flow [??] and Wan2.1 [??]--by introducing FlowEdit [??], resulting in training-free and inversion-free video editing models Pyramid-Edit and Wan-Edit. We compare five diffusion methods with our two RF methods on the proposed FiVE-Bench, evaluating them across 15 metrics. These metrics include background preservation, text-video similarity, temporal consistency, and generated video quality. To further enhance object-level evaluation, we introduce FiVE-Acc, a novel metric leveraging Vision-Language Models (VLMs) to assess the success of fine-grained video editing. Experimental results demonstrate that RF-based editing significantly outperforms diffusion-based methods, with Wan-Edit achieving the best overall performance and exhibiting the least sensitivity to hyperparameters. More video demo available on the website: https://sites.google.com/view/five-benchmark.

</details>

---

## 181. Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving

- [ ] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Fine-Grained_Evaluation_of_Large_Vision-Language_Models_in_Autonomous_Driving_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Fine-Grained_Evaluation_of_Large_Vision-Language_Models_in_Autonomous_Driving_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing benchmarks for Vision-Language Model (VLM) in autonomous driving (AD) primarily assess interpretability through open-form visual question answering (QA) within coarse-grained tasks, which remain insufficient to assess capabilities in complex driving scenarios. To this end, we introduce VLADBench, a challenging and fine-grained benchmark featuring close-form QAs that progress from static foundational knowledge and elements to advanced reasoning for dynamic on-road situations. The elaborate VLADBench spans 5 key domains: Traffic Knowledge Understanding, General Element Recognition, Traffic Graph Generation, Target Attribute Comprehension, and Ego Decision-Making and Planning. These domains are further broken down into 11 secondary aspects and 29 tertiary tasks for a granular evaluation. A thorough assessment of general and domain-specific (DS) VLMs on this benchmark reveals both their strengths and critical limitations in AD contexts. To further exploit the cognitive and reasoning interactions among the 5 domains for AD understanding, we start from a small-scale VLM and train the DS models on individual domain datasets (collected from 1.4M DS QAs across public sources). The experimental results demonstrate that the proposed benchmark provides a crucial step toward a more comprehensive assessment of VLMs in AD, paving the way for the development of more cognitively sophisticated and reasoning-capable AD systems. The benchmark is available at https://github.com/Depth2World/VLADBench.

</details>

---

## 182. Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models

- [ ] Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Few-Shot_Image_Quality_Assessment_via_Adaptation_of_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Few-Shot_Image_Quality_Assessment_via_Adaptation_of_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image Quality Assessment (IQA) remains an unresolved challenge in computer vision due to complex distortions, diverse image content, and limited data availability. Existing Blind IQA (BIQA) methods largely rely on extensive human annotations, which are labor-intensive and costly due to the demanding nature of creating IQA datasets. To reduce this dependency, we propose the Gradient-Regulated Meta-Prompt IQA Framework (GRMP-IQA), designed to efficiently adapt the visual-language pre-trained model, CLIP, to IQA tasks, achieving high accuracy even with limited data. GRMP-IQA consists of two core modules: (i) Meta-Prompt Pre-training Module and (ii) Quality-Aware Gradient Regularization. The Meta Prompt Pre-training Module leverages a meta-learning paradigm to pre-train soft prompts with shared meta-knowledge across different distortions, enabling rapid adaptation to various IQA tasks. On the other hand, the Quality-Aware Gradient Regularization is designed to adjust the update gradients during fine-tuning, focusing the model's attention on quality-relevant features and preventing overfitting to semantic information. Extensive experiments on standard BIQA datasets demonstrate the superior performance to the state-of-the-art BIQA methods under limited data setting. Notably, utilizing just 20% of the training data, GRMP-IQA is competitive with most existing fully supervised BIQA approaches. Our code is available via https://github.com/LXDxmu/GRMP-IQA.

</details>

---

## 183. Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation

- [ ] Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Images_as_Noisy_Labels_Unleashing_the_Potential_of_the_Diffusion_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Images_as_Noisy_Labels_Unleashing_the_Potential_of_the_Diffusion_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, open-vocabulary semantic segmentation has garnered growing attention. Most current methods leverage vision-language models like CLIP to recognize unseen categories through their zero-shot capabilities. However, CLIP struggles to establish potential spatial dependencies among scene objects due to its holistic pre-training objective, causing sub-optimal results. In this paper, we propose a DEnoising learning framework based on the Diffusion model for Open-vocabulary semantic Segmentation, called DEDOS, which is aimed at constructing the scene skeleton. Motivation stems from the fact that diffusion models incorporate not only the visual appearance of objects but also embed rich scene spatial priors. Our core idea is to view images as labels embedded with "noise"--non-essential details for perceptual tasks--and to disentangle the intrinsic scene prior from the diffusion feature during the denoising process of the images. Specifically, to fully harness the scene prior knowledge of the diffusion model, we introduce learnable proxy queries during the denoising process. Meanwhile, we leverage the robustness of CLIP features to texture shifts as supervision, guiding proxy queries to focus on constructing the scene skeleton and avoiding interference from texture information in the diffusion feature space. Finally, we enhance spatial understanding within CLIP features using proxy queries, which also serve as an interface for multi-level interaction between text and visual modalities. Extensive experiments validate the effectiveness of our method, experimental results on five standard benchmarks have shown that DEDOS achieves state-of-the-art performance. We will make the code publicly available.

</details>

---

## 184. Information Density Principle for MLLM Benchmarks

- [ ] Information Density Principle for MLLM Benchmarks | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Information_Density_Principle_for_MLLM_Benchmarks_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Information_Density_Principle_for_MLLM_Benchmarks_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the emergence of Multimodal Large Language Models (MLLMs), hundreds of benchmarks have been developed to ensure the reliability of MLLMs in downstream tasks. However, the evaluation mechanism itself may not be reliable. For developers of MLLMs, questions remain about which benchmark to use and whether the test results meet their requirements. Therefore, we propose a critical principle of Information Density, which examines **how much insight a benchmark can provide for the development of MLLMs.** We characterize it from four key dimensions: (1) Fallacy, (2) Difficulty, (3) Redundancy, (4) Diversity. Through a comprehensive analysis of more than 10,000 samples, we measured the information density of 19 MLLM benchmarks. Experiments show that using the latest benchmarks in testing can provide more insight compared to previous ones, but there is still room for improvement in their information density. We hope this principle can promote the development and application of future MLLM benchmarks.

</details>

---

## 185. Intermediate Connectors and Geometric Priors for Language-Guided Affordance Segmentation on Unseen Object Categories

- [ ] Intermediate Connectors and Geometric Priors for Language-Guided Affordance Segmentation on Unseen Object Categories | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Intermediate_Connectors_and_Geometric_Priors_for_Language-Guided_Affordance_Segmentation_on_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Intermediate_Connectors_and_Geometric_Priors_for_Language-Guided_Affordance_Segmentation_on_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language-guided Affordance Segmentation (LASO) aims to identify actionable object regions based on text instructions. At the core of its practicality is learning generalizable affordance knowledge that captures functional regions across diverse objects. However, current LASO solutions struggle to extend learned affordances to object categories that are not encountered during training. Scrutinizing these designs, we identify limited generalizability on unseen categories, stemming from (1) underutilized generalizable patterns in the intermediate layers of both 3D and text backbones, which impedes the formation of robust affordance knowledge, and (2) the inability to handle substantial variability in affordance regions across object categories due to a lack of structural knowledge of the target region.Towards this, we introduce a GeneraLized frAmework on uNseen CategoriEs (GLANCE), incorporating two key components: a cross-modal connector that links intermediate stages of the text and 3D backbones to enrich pointwise embeddings with affordance concepts, and a VLM-guided query generator that provides affordance priors by extracting a few 3D key points based on the intra-view reliability and cross-view consistency of their multi-view segmentation masks. Extensive experiments on two benchmark datasets demonstrate that GLANCE outperforms state-of-the-art methods (SoTAs), with notable improvements in generalization to unseen categories. Our code is available at https://anonymous.4open.science/r/GLANCE.

</details>

---

## 186. LLM Thought Divergence and Convergence for Dialogue-Based Image Generation Control

- [ ] LLM Thought Divergence and Convergence for Dialogue-Based Image Generation Control | https://openaccess.thecvf.com/content/ICCV2025/html/Li_LLM_Thought_Divergence_and_Convergence_for_Dialogue-Based_Image_Generation_Control_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_LLM_Thought_Divergence_and_Convergence_for_Dialogue-Based_Image_Generation_Control_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generative AI (GenAI), which revolutionized both computer vision and natural language processing, has drawn continuous attention recently. Benefits from GenAI with the evolution of large language models (LLMs), the image generation task evolved from prompt-based to dialogue-based, which takes the real-world human intent expressed through conversations. When breaking this task into multiple steps, the best pathway of analyzing the dialogues is not determined, such as whether the objects or prompted template should be focused on the first step of dialogues analyzing. Thus, a multi-chain reasoning is requested to decompose this application beyond a pure chain-of-thought structure. After the divergent process, the question comes to how to converge the thinking chain that leads to the best matched image, which requires a new evaluation method to lead the thinking process. To address these challenges, we propose the LLM Thought Divergence and Convergence (LTDC) framework, which simulates human cognitive processes through three phases: (1) The Step-by-Step Thought process decomposes dialogue-based image generation tasks into sequential thinking chains using LLMs; (2) The Image Generation process creates image prompts following these thought instructions and produces corresponding images; (3) The Evaluation process aligns the coherence between generated images and dialogues through a multi-modal LLM, guiding the selection of optimal thinking chains. Evaluated on VisDial, our LTE framework achieves a 4.87% improvement in CLIP similarity, demonstrating the effectiveness in generating images with higher semantic fidelity.

</details>

---

## 187. Language Decoupling with Fine-grained Knowledge Guidance for Referring Multi-object Tracking

- [ ] Language Decoupling with Fine-grained Knowledge Guidance for Referring Multi-object Tracking | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring Multi-Object Tracking (RMOT) aims to detect and track specific objects based on natural language expressions. Previous methods typically rely on sentence-level vision-language alignment, often failing to exploit fine-grained linguistic cues that are crucial for distinguishing objects with similar characteristics. Notably, these cues play distinct roles at different tracking stages and should be leveraged accordingly to provide more explicit guidance. In this work, we propose DKGTrack, a novel RMOT method that enhances language comprehension for precise object tracking by decoupling language expressions into localized descriptions and motion states. To improve the accuracy of language-guided object identification, we introduce a Static Semantic Enhancement (SSE) module, which enhances region-level vision-language alignment through hierarchical cross-modal feature interaction, providing more discriminative object representations for tracking. Furthermore, we propose a Motion Perception Alignment (MPA) module that explicitly aligns object queries with motion descriptions, enabling accurate object trajectory prediction across frames. Experimental results on multiple RMOT benchmarks demonstrate the effectiveness of our method, which achieves competitive performance in challenging tracking scenarios.

</details>

---

## 188. MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling

- [ ] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling | https://openaccess.thecvf.com/content/ICCV2025/html/Li_MaTVLM_Hybrid_Mamba-Transformer_for_Efficient_Vision-Language_Modeling_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_MaTVLM_Hybrid_Mamba-Transformer_for_Efficient_Vision-Language_Modeling_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the advancement of RNN models with linear complexity, the quadratic complexity challenge of transformers has the potential to be overcome. Notably, the emerging Mamba-2 has demonstrated competitive performance, bridging the gap between RNN models and transformers. However, due to sequential processing and vanishing gradients, RNN models struggle to capture long-range dependencies, limiting contextual understanding. This results in slow convergence, high resource demands, and poor performance on downstream understanding and complex reasoning tasks. In this work, we present MaTVLM, a method for distilling pre-trained vision-language models (VLMs) into an efficient Mamba-Transformer hybrid architecture. Specifically, we construct this hybrid architecture by replacing a portion of the transformer decoder layers in the pre-trained VLM with Mamba-2 layers. Building on this design, we employ a single-stage distillation process, incorporating a clever initialization strategy, leveraging the inherent relationship between attention mechanisms and Mamba-2, and initialize Mamba-2 with corresponding attention weights, which notably accelerates convergence. With the pre-trained VLM serving as the teacher model, this distillation process further boosts both convergence speed and model performance. Furthermore, we investigate the impact of differential distillation loss within our training framework. We evaluate MaTVLM on multiple benchmarks, demonstrating competitive performance against the teacher model and existing VLMs while surpassing both Mamba-based VLMs and models of comparable parameter scales. Remarkably, MaTVLM achieves up to 4.3 times faster inference than the teacher model while reducing GPU memory consumption by 27.5%, all without compromising performance. Code and models are released at https://github.com/hustvl/MaTVLM.

</details>

---

## 189. OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography

- [ ] OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography | https://openaccess.thecvf.com/content/ICCV2025/html/Li_OracleFusion_Assisting_the_Decipherment_of_Oracle_Bone_Script_with_Structurally_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_OracleFusion_Assisting_the_Decipherment_of_Oracle_Bone_Script_with_Structurally_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As one of the earliest ancient languages, Oracle Bone Script (**OBS**) encapsulates the cultural records and intellectual expressions of ancient civilizations. Despite the discovery of approximately 4,500 OBS characters, only about 1,600 have been deciphered. The remaining undeciphered ones, with their complex structure and abstract imagery, pose significant challenges for interpretation. To address these challenges, this paper proposes a novel two-stage semantic typography framework, named **OracleFusion**. In the first stage, this approach leverages the Multimodal Large Language Model (MLLM) with enhanced Spatial Awareness Reasoning (SAR) to analyze the glyph structure of the OBS character and perform visual localization of key components. In the second stage, we introduce Oracle Structural Vector Fusion (**OSVF**), incorporating glyph structure constraints and glyph maintenance constraints to ensure the accurate generation of semantically enriched vector fonts. This approach preserves the objective integrity of the glyph structure, offering visually enhanced representations that assist experts in deciphering OBS. Extensive qualitative and quantitative experiments demonstrate that OracleFusion outperforms state-of-the-art baseline models in terms of semantics, visual appeal, and glyph maintenance, significantly enhancing both readability and aesthetic quality. Furthermore, OracleFusion provides expert-like insights on unseen oracle characters, making it a valuable tool for advancing the decipherment of OBS.

</details>

---

## 190. STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?

- [ ] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding? | https://openaccess.thecvf.com/content/ICCV2025/html/Li_STI-Bench_Are_MLLMs_Ready_for_Precise_Spatial-Temporal_World_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_STI-Bench_Are_MLLMs_Ready_for_Precise_Spatial-Temporal_World_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The use of Multimodal Large Language Models (MLLMs) as an end-to-end solution for Embodied AI and Autonomous Driving has become a prevailing trend. While MLLMs have been extensively studied for visual semantic understanding tasks, their ability to perform precise and quantitative spatial-temporal understanding in real-world applications remains largely unexamined, leading to uncertain prospects. To address this gap, we introduce ST-Bench, a benchmark designed to evaluate MLLMs' spatial-temporal understanding through challenging tasks such as estimating and predicting the appearance, pose, displacement, and motion of objects. Our benchmark encompasses a wide range of robot and vehicle operations across desktop, indoor, and outdoor scenarios. The extensive experiments reveals that the state-of-the-art MLLMs still struggle in real-world spatial-temporal understanding, especially in tasks requiring precise distance estimation and motion analysis.

</details>

---

## 191. SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining

- [ ] SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining | https://openaccess.thecvf.com/content/ICCV2025/html/Li_SceneSplat_Gaussian_Splatting-based_Scene_Understanding_with_Vision-Language_Pretraining_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_SceneSplat_Gaussian_Splatting-based_Scene_Understanding_with_Vision-Language_Pretraining_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recognizing arbitrary or previously unseen categories is essential for comprehensive real-world 3D scene understanding. Currently, all existing methods rely on 2D or textual modalities during training, or together at inference. This highlights a clear absence of a model capable of processing 3D data alone for learning semantics end-to-end, along with the necessary data to train such a model. Meanwhile, 3D Gaussian Splatting (3DGS) has emerged as the de facto standard for 3D scene representation across various vision tasks. However, effectively integrating semantic reasoning into 3DGS in a generalizable fashion remains an open challenge.To address these limitations we introduce SceneSplat, to our knowledge the first large-scale 3D indoor scene understanding approach that operates natively on 3DGS. Furthermore, we propose a self-supervised learning scheme that unlocks rich 3D feature learning from unlabeled scenes. In order to power the proposed methods, we introduce SceneSplat-7K, the first large-scale 3DGS dataset for indoor scenes, comprising of 6868 scenes derived from 7 established datasets like ScanNet, Matterport3D, etc. Generating SceneSplat-7K required computational resources equivalent to 119 GPU-days on an L4 GPU, enabling standardized benchmarking for 3DGS-based reasoning for indoor scenes.Our exhaustive experiments on SceneSplat-7K demonstrate the significant benefit of the proposed methods over the established baselines. Our code, model, and datasets will be released to facilitate further research.

</details>

---

## 192. Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection

- [ ] Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Synthesizing_Near-Boundary_OOD_Samples_for_Out-of-Distribution_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Synthesizing_Near-Boundary_OOD_Samples_for_Out-of-Distribution_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models have exhibited remarkable abilities in detecting out-of-distribution (OOD) samples. However, some challenging OOD samples, which lie close to in-distribution (InD) data in image feature space, can still lead to misclassification. The emergence of foundation models like diffusion models and multimodal large language models (MLLMs) offers a potential solution to this issue. In this work, we propose SynOOD, a novel approach that harnesses foundation models to generate synthetic, challenging OOD data for fine-tuning CLIP models, thereby enhancing boundary-level discrimination between InD and OOD samples. Our method uses an iterative in-painting process guided by contextual prompts from MLLMs to produce nuanced, boundary-aligned OOD samples. These samples are refined through noise adjustments based on gradients from OOD scores like the energy score, effectively sampling from the InD/OOD boundary. With these carefully synthesized images, we fine-tune the CLIP image encoder and negative label features derived from the text encoder to strengthen connections between near-boundary OOD samples and a set of negative labels. Finally, SynOOD achieves state-of-the-art performance on the large-scale ImageNet benchmark, with minimal increases in parameters and runtime. Our approach significantly surpasses existing methods, improving AUROC by 2.80% and reducing FPR95 by 11.13%. Codes are available in https://github.com/Jarvisgivemeasuit/SynOOD.

</details>

---

## 193. SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing

- [ ] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing | https://openaccess.thecvf.com/content/ICCV2025/html/Li_SuperEdit_Rectifying_and_Facilitating_Supervision_for_Instruction-Based_Image_Editing_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_SuperEdit_Rectifying_and_Facilitating_Supervision_for_Instruction-Based_Image_Editing_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Due to the challenges of manually collecting accurate editing data, existing datasets are typically constructed using various automated methods, leading to noisy supervision signals caused by the mismatch between editing instructions and original-edited image pairs. Recent efforts attempt to improve editing models through generating higher-quality edited images, pre-training on recognition tasks, or introducing vision-language models (VLMs) but fail to resolve this fundamental issue. In this paper, we offer a novel solution by constructing more effective editing instructions for given image pairs. This includes rectifying the editing instructions to better align with the original-edited image pairs and using contrastive editing instructions to further enhance their effectiveness. Specifically, we find that editing models exhibit specific generation attributes at different inference steps, independent of the text. Based on these prior attributes, we define a unified guide for VLMs to rectify editing instructions. However, there are some challenging editing scenarios that cannot be resolved solely with rectified instructions. To this end, we further construct contrastive supervision signals with positive and negative instructions and introduce them into the model training using triplet loss, thereby further facilitating supervision effectiveness. Our method does not require the VLM modules or pre-training tasks used in previous work, offering a more direct and efficient way to provide better supervision signals, and providing a novel, simple, and effective solution for instruction-based image editing. Results on multiple benchmarks demonstrate that our method significantly outperforms existing approaches. Compared with previous SOTA SmartEdit, we achieve 9.19% improvements on the Real-Edit benchmark with 30xless training data and 13xsmaller model size. All data and models are open-sourced on \href https://github.com/bytedance/SuperEdit  Github  for future research.

</details>

---

## 194. Token Activation Map to Visually Explain Multimodal LLMs

- [ ] Token Activation Map to Visually Explain Multimodal LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Token_Activation_Map_to_Visually_Explain_Multimodal_LLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Token_Activation_Map_to_Visually_Explain_Multimodal_LLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are broadly empowering various fields. Despite their advancements, the explainability of MLLMs remains less explored, hindering deeper understanding, model credibility, and effective visualization. Unlike conventional vision models (e.g., CNNs, ViTs, CLIP) that produce a single output, MLLMs generate sequences of tokens progressively, where each generated token depends on the previous context. Therefore, earlier context tokens can introduce redundant activations that interfere with the explanation of later tokens beyond their original information. Existing studies often overlook this issue, but our observations reveal that these redundant correlations can significantly hurt the reliability of explanations. To address this, we propose an estimated causal inference method to mitigate the interference of context to achieve high-quality MLLM explanation, with a novel rank Gaussian filter to further reduce activation noises. We term this method Token Activation Map (TAM) to highlight the consideration of interactions between tokens. TAM also indicates that it excels at explaining multiple tokens of MLLM, which is different from the Class Activation Map (CAM) for a single prediction. Our TAM method significantly outperforms existing SoTA methods, showcasing high-quality visualization results that can be utilized for various scenarios, such as object localization, failure case analysis, video visualization, MLLMs visual comparison, and model understanding (e.g., color, shape, action, location, visual reasoning, multi-turn conversation, etc). The code is available at github.com/xmed-lab/TAM.

</details>

---

## 195. Towards Long-Horizon Vision-Language-Action System: Reasoning, Acting and Memory

- [ ] Towards Long-Horizon Vision-Language-Action System: Reasoning, Acting and Memory | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Towards_Long-Horizon_Vision-Language-Action_System_Reasoning_Acting_and_Memory_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Towards_Long-Horizon_Vision-Language-Action_System_Reasoning_Acting_and_Memory_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language-Action (VLA) is crucial for autonomous decision-making in embodied systems. While current methods have advanced single-skill abilities, their short-horizon capability limits applicability in real-world scenarios. To address this challenge, we innovatively propose MindExplore, a general hierarchical VLA system with cross-skill for long-horizon tasks in highly dynamic sand. The key insight is to iteratively align the knowledge domain of task planning and action execution. Thus, this task-oriented action enables outstanding generalization across a wide range of real-world scenarios. In the reasoning layer, task-specific chains of thought (CoT) are designed for planning long-horizon task sequences and providing meta-action signals. In the acting layer, a simple but powerful Mixture of Policy Experts strategy is built inspired by signals and multimodal inputs for adaptively selecting skill experts and generating closed-loop action sequences. Also, it integrates a lightweight Multimodal Diffusion Policy (MMDP) to enhance spatial perception by fusing multi-visual modality features. Besides, the pioneering memory mechanism establishes feedback between the reasoning and acting layers, facilitating adaptive execution of long-horizon tasks and real-time replanning. Notably, we create SandGo-1k and SandThink-21k, the first expert-level multimodal embodied dataset and CoT dataset tailored for sandy environments. At a high execution frequency of 30 FPS, MindExplore is 3.01 times more successful than existing methods in unstructured and dynamic environments.

</details>

---

## 196. UIPro: Unleashing Superior Interaction Capability For GUI Agents

- [ ] UIPro: Unleashing Superior Interaction Capability For GUI Agents | https://openaccess.thecvf.com/content/ICCV2025/html/Li_UIPro_Unleashing_Superior_Interaction_Capability_For_GUI_Agents_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_UIPro_Unleashing_Superior_Interaction_Capability_For_GUI_Agents_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Building autonomous agents that perceive and operate graphical user interfaces (GUIs) like humans has long been a vision in the field of artificial intelligence. Central to these agents is the capability for GUI interaction, which involves GUI understanding and planning capabilities. Existing methods have tried developing GUI agents based on the multi-modal comprehension ability of vision-language models (VLMs). However, the limited scenario, insufficient size, and heterogeneous action spaces hinder the progress of building generalist GUI agents. To resolve these issues, this paper proposes UIPro, a novel generalist GUI agent trained with extensive multi-platform and multi-task GUI interaction data, coupled with a unified action space. We first curate a comprehensive dataset encompassing 20.6 million GUI understanding tasks to pre-train UIPro, granting it a strong GUI grounding capability, which is key to downstream GUI agent tasks. Subsequently, we establish a unified action space to harmonize heterogeneous GUI agent task datasets and produce a merged dataset to foster the action prediction ability of UIPro via continued fine-tuning. Experimental results demonstrate UIPro's superior performance across multiple GUI task benchmarks on various platforms, highlighting the effectiveness of our approach.

</details>

---

## 197. Unbiased Region-Language Alignment for Open-Vocabulary Dense Prediction

- [ ] Unbiased Region-Language Alignment for Open-Vocabulary Dense Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Unbiased_Region-Language_Alignment_for_Open-Vocabulary_Dense_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Unbiased_Region-Language_Alignment_for_Open-Vocabulary_Dense_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs), such as CLIP, have demonstrated impressive zero-shot recognition capability, but still underperform in dense prediction tasks. Self-distillation recently is emerging as a promising approach for fine-tuning VLMs to better adapt to local regions without requiring extensive annotations. However, previous state-of-the-art approaches often suffer from significant `foreground bias', where models tend to wrongly identify background regions as foreground objects. To alleviate this issue, we propose DenseVLM, a framework designed to learn unbiased region-language alignment from powerful pre-trained VLM representations. To alleviate this issue, we propose DenseVLM, a framework designed to learn unbiased region-language alignment from powerful pre-trained VLM representations. DenseVLM leverages the pre-trained VLM to retrieve categories for unlabeled regions and then decouples the interference between foreground and background features. We show that DenseVLM can directly replace the original VLM in open-vocabulary object detection and image segmentation methods, leading to notable performance improvements. Furthermore, it exhibits promising zero-shot scalability when training on more extensive and diverse datasets. Our code is publicly available https://github.com/HVision-NKU/DenseVLM.

</details>

---

## 198. Unveiling the Invisible: Reasoning Complex Occlusions Amodally with AURA

- [ ] Unveiling the Invisible: Reasoning Complex Occlusions Amodally with AURA | https://openaccess.thecvf.com/content/ICCV2025/html/Li_Unveiling_the_Invisible_Reasoning_Complex_Occlusions_Amodally_with_AURA_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Unveiling_the_Invisible_Reasoning_Complex_Occlusions_Amodally_with_AURA_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Amodal segmentation aims to infer the complete shape of occluded objects, even when the occluded region's appearance is unavailable. However, current amodal segmentation methods lack the capability to interact with users through text input and struggle to understand or reason about implicit and complex purposes. While methods like LISA integrate multi-modal large language models (LLMs) with segmentation for reasoning tasks, they are limited to predicting only visible object regions and face challenges in handling complex occlusion scenarios. To address these limitations, we propose a novel task named amodal reasoning segmentation, aiming to predict the complete amodal shape of occluded objects while providing answers with elaborations based on user text input. We develop a generalizable dataset generation pipeline and introduce a new dataset focusing on daily life scenarios, encompassing diverse real-world occlusions. Furthermore, we present AURA (Amodal Understanding and Reasoning Assistant), a novel model with advanced global and spatial-level designs specifically tailored to handle complex occlusions. Extensive experiments validate AURA's effectiveness on the proposed dataset. The code, model, and dataset are released on this page: https://zhixuanli.github.io/projects/li2025aura/index.html.

</details>

---

## 199. Describe Anything: Detailed Localized Image and Video Captioning

- [ ] Describe Anything: Detailed Localized Image and Video Captioning | https://openaccess.thecvf.com/content/ICCV2025/html/Lian_Describe_Anything_Detailed_Localized_Image_and_Video_Captioning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lian_Describe_Anything_Detailed_Localized_Image_and_Video_Captioning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating detailed and accurate descriptions for specific regions in images and videos remains a fundamental challenge for vision-language models. We introduce the Describe Anything Model (DAM), a model designed for detailed localized captioning (DLC). DAM preserves both local details and global context through two key innovations: a focal prompt, which ensures high-resolution encoding of targeted regions, and a localized vision backbone, which integrates precise localization with its broader context. To tackle the scarcity of high-quality DLC data, we propose a Semi-supervised learning (SSL)-based Data Pipeline (DLC-SDP). DLC-SDP starts with existing segmentation datasets and expands to unlabeled web images using SSL. We introduce DLC-Bench, a benchmark designed to evaluate DLC without relying on reference captions. DAM sets new state-of-the-art on 7 benchmarks spanning keyword-level, phrase-level, and detailed multi-sentence localized image and video captioning.

</details>

---

## 200. ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations

- [ ] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations | https://openaccess.thecvf.com/content/ICCV2025/html/Liang_ReferDINO_Referring_Video_Object_Segmentation_with_Visual_Grounding_Foundations_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liang_ReferDINO_Referring_Video_Object_Segmentation_with_Visual_Grounding_Foundations_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring video object segmentation (RVOS) aims to segment target objects throughout a video based on a text description. This is challenging as it involves deep vision-language understanding, pixel-level dense prediction and spatiotemporal reasoning. Despite notable progress in recent years, existing methods still exhibit a noticeable gap when considering all these aspects. In this work, we propose ReferDINO, a strong RVOS model that inherits region-level vision-language alignment from foundational visual grounding models, and is further endowed with pixel-level dense perception and cross-modal spatiotemporal reasoning. In detail, ReferDINO integrates two key components: 1) a grounding-guided deformable mask decoder that utilizes location prediction to progressively guide mask prediction through differentiable deformation mechanisms; 2) an object-consistent temporal enhancer that injects pretrained time-varying text features into inter-frame interaction to capture object-aware dynamic changes. Moreover, a confidence-aware query pruning strategy is designed to accelerate object decoding without compromising model performance. Extensive experimental results on five benchmarks demonstrate that our ReferDINO significantly outperforms previous methods (e.g., +3.9% (\mathcal J &\mathcal F ) on Ref-YouTube-VOS) with real-time inference speed (51 FPS).

</details>

---

## 201. Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models

- [ ] Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancements in Vision Language Models (VLMs) have prompted the development of multi-modal medical assistant systems. Despite this progress, current models still have inherent probabilistic uncertainties, often producing erroneous or unverified responses--an issue with serious implications in medical applications. Existing methods aim to enhance the performance of Medical Vision Language Model (MedVLM) by adjusting model structure, fine-tuning with high-quality data, or through preference fine-tuning. However, these training-dependent strategies are costly and still lack sufficient alignment with clinical expertise. To address these issues, we propose an expert-in-the-loop framework named Expert-Controlled Classifier-Free Guidance (Expert-CFG) to align MedVLM with clinical expertise without additional training. This framework introduces an uncertainty estimation strategy to identify unreliable outputs. It then retrieves relevant references to assist experts in highlighting key terms and applies classifier-free guidance to refine the token embeddings of MedVLM, ensuring that the adjusted outputs are correct and align with expert highlights. Evaluations across three medical visual question answering benchmarks demonstrate that the proposed Expert-CFG, with 4.2B parameters and limited expert annotations, outperforms state-of-the-art models with 13B parameters. The results demonstrate the feasibility of deploying such a system in resource-limited settings for clinical use.

</details>

---

## 202. WSI-LLaVA: A Multimodal Large Language Model for Whole Slide Image

- [ ] WSI-LLaVA: A Multimodal Large Language Model for Whole Slide Image | https://openaccess.thecvf.com/content/ICCV2025/html/Liang_WSI-LLaVA_A_Multimodal_Large_Language_Model_for_Whole_Slide_Image_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liang_WSI-LLaVA_A_Multimodal_Large_Language_Model_for_Whole_Slide_Image_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in computational pathology have introduced whole slide image (WSI)-level multimodal large language models (MLLMs) for automated pathological analysis. However, current WSI-level MLLMs face two critical challenges: limited explainability in their decision-making process and insufficient attention to morphological features crucial for accurate diagnosis. To address these challenges, we first introduce WSI-Bench, a large-scale morphology-aware benchmark containing 180k VQA pairs from 9,850 WSIs across 30 cancer types, specifically designed to evaluate MLLMs' understanding of morphological characteristics crucial for accurate diagnosis. To the best of our knowledge, WSI-Bench presents the first benchmarking systematically evaluate morphological understanding capabilities in WSI analysis. To enhance the model explainability, we present WSI-LLaVA, an MLLM framework for gigapixel WSI understanding with a three-stage training strategy, which can provide detailed morphological findings to explain its final answer. For more precise model assessment in pathological contexts, we develop two specialized WSI metrics: WSI-Precision and WSI-Relevance. Extensive evaluation on WSI-Bench reveals both the capabilities and limitations of current WSI MLLMs in morphological analysis and various pathology tasks, while demonstrating WSI-LLaVA's superior performance across all capabilities.

</details>

---

## 203. Background Invariance Testing According to Semantic Proximity

- [ ] Background Invariance Testing According to Semantic Proximity | https://openaccess.thecvf.com/content/ICCV2025/html/Liao_Background_Invariance_Testing_According_to_Semantic_Proximity_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liao_Background_Invariance_Testing_According_to_Semantic_Proximity_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In many applications, machine-learned (ML) models are required to hold some invariance qualities, such as rotation, size, and intensity invariance. Among these, testing for background invariance presents a significant challenge due to the vast and complex data space it encompasses. To evaluate invariance qualities, we first use a visualization-based testing framework which allows human analysts to assess and make informed decisions about the invariance properties of ML models. We show that such informative testing framework is preferred as ML models with the same global statistics (e.g., accuracy scores) can behave differently and have different visualized testing patterns. However, such human analysts might not lead to consistent decisions without a systematic sampling approach to select representative testing suites. In this work, we present a technical solution for selecting background scenes according to their semantic proximity to a target image that contains a foreground object being tested. We construct an ontology for storing knowledge about relationships among different objects using association analysis. This ontology enables an efficient and meaningful search for background scenes of different semantic distances to a target image, enabling the selection of a test suite that is both diverse and reasonable. Compared with other testing techniques, e.g., random sampling, nearest neighbors, or other sampled test suites by visual-language models (VLMs), our method achieved a superior balance between diversity and consistency of human annotations, thereby enhancing the reliability and comprehensiveness of background invariance testing.

</details>

---

## 204. ImageGen-CoT: Enhancing Text-to-Image In-context Learning with Chain-of-Thought Reasoning

- [ ] ImageGen-CoT: Enhancing Text-to-Image In-context Learning with Chain-of-Thought Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Liao_ImageGen-CoT_Enhancing_Text-to-Image_In-context_Learning_with_Chain-of-Thought_Reasoning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liao_ImageGen-CoT_Enhancing_Text-to-Image_In-context_Learning_with_Chain-of-Thought_Reasoning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we study the problem of Text-to-Image In-Context Learning (T2I-ICL). While Unified Multimodal LLMs (MLLMs) have advanced rapidly in recent years, they struggle with contextual reasoning in T2I-ICL scenarios. To address this limitation, we propose a novel framework that incorporates a reasoning chain called ImageGen-CoT prior to image generation. To avoid generating ineffective reasoning steps, we develop an automatic pipeline to curate a high-quality ImageGen-CoT dataset. We then fine-tune MLLMs using this dataset to enhance their contextual reasoning capabilities. To further enhance performance, we explore test-time scale-up strategies and propose a novel hybrid scaling approach. This approach first generates multiple reasoning chains and then produces multiple images for each chain via sampling. Extensive experiments demonstrate the effectiveness of our proposed method. Notably, fine-tuning with the ImageGen-CoT dataset leads to a substantial 80% performance gain for SEED-X on T2I-ICL tasks. See our project page at https://ImageGen-CoT.github.io/. Code will be open-sourced.

</details>

---

## 205. LangBridge: Interpreting Image as a Combination of Language Embeddings

- [ ] LangBridge: Interpreting Image as a Combination of Language Embeddings | https://openaccess.thecvf.com/content/ICCV2025/html/Liao_LangBridge_Interpreting_Image_as_a_Combination_of_Language_Embeddings_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liao_LangBridge_Interpreting_Image_as_a_Combination_of_Language_Embeddings_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent years have witnessed remarkable advances in Large Vision-Language Models (LVLMs), which have achieved human-level performance across various complex vision-language tasks. Following LLaVA's paradigm, mainstream LVLMs typically employ a shallow MLP for visual-language alignment through a two-stage training process: pretraining for cross-modal alignment followed by instruction tuning. While this approach has proven effective, the underlying mechanisms of how MLPs bridge the modality gap remain poorly understood. Although some research has explored how LLMs process transformed visual tokens, few studies have investigated the fundamental alignment mechanism. Furthermore, the MLP adapter requires retraining whenever switching LLM backbones. To address these limitations, we first investigate the working principles of MLP adapters and discover that they learn to project visual embeddings into subspaces spanned by corresponding text embeddings progressively. Based on this insight, we propose LangBridge, a novel adapter that explicitly maps visual tokens to linear combinations of LLM vocabulary embeddings. This innovative design enables pretraining-free adapter transfer across different LLMs while maintaining performance. Our experimental results demonstrate that a LangBridge adapter pre-trained on Qwen2-0.5B can be directly applied to larger models such as LLaMA3-8B or Qwen2.5-14B while maintaining competitive performance. Overall, LangBridge enables interpretable vision-language alignment by grounding visual representations in LLM vocab embedding, while its plug-and-play design ensures efficient reuse across multiple LLMs with nearly no performance degradation. See our project page at https://curryx-001.github.io/LangBridge.github.io/.

</details>

---

## 206. ChartCap: Mitigating Hallucination of Dense Chart Captioning

- [ ] ChartCap: Mitigating Hallucination of Dense Chart Captioning | https://openaccess.thecvf.com/content/ICCV2025/html/Lim_ChartCap_Mitigating_Hallucination_of_Dense_Chart_Captioning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lim_ChartCap_Mitigating_Hallucination_of_Dense_Chart_Captioning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating accurate, informative, and hallucination-free captions for charts remains challenging for vision language models, primarily due to the lack of large-scale, high-quality datasets of real-world charts. However, existing real-world chart datasets suffer from the inclusion of extraneous information that cannot be inferred from the chart and failure to sufficiently capture structural elements and key insights. Therefore, we introduce ChartCap, a large-scale dataset of 565K real-world chart images paired with type-specific, dense captions that exclude extraneous information and highlight both structural elements and key insights in detail. To build ChartCap, we design a four-stage pipeline that generates captions using only the discernible data from the chart and employ a cycle consistency-based human verification, which accelerates quality control without sacrificing accuracy. Additionally, we propose a novel metric, the Visual Consistency Score, which evaluates caption quality by measuring the similarity between the chart regenerated from a caption and the original chart, independent of reference captions. Extensive experiments confirms that models fine-tuned on ChartCap consistently generate more accurate and informative captions with reduced hallucinations, surpassing both open-source and proprietary models and even human-annotated captions.

</details>

---

## 207. INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance

- [ ] INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance | https://openaccess.thecvf.com/content/ICCV2025/html/Lin_INS-MMBench_A_Comprehensive_Benchmark_for_Evaluating_LVLMs_Performance_in_Insurance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lin_INS-MMBench_A_Comprehensive_Benchmark_for_Evaluating_LVLMs_Performance_in_Insurance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) and Multimodal Large Language Models (MLLMs) have demonstrated outstanding performance in various general multimodal applications and have shown increasing promise in specialized domains. However, their potential in the insurance domain--characterized by diverse application scenarios and rich multimodal data--remains largely underexplored. To date, there is no systematic review of multimodal tasks, nor a benchmark specifically designed to assess the capabilities of LVLMs in insurance. This gap hinders the development of LVLMs within the insurance industry. This study systematically reviews and categorizes multimodal tasks for 4 representative types of insurance: auto, property, health, and agricultural. We introduce INS-MMBench, the first hierarchical benchmark tailored for the insurance domain. INS-MMBench encompasses 22 fundamental tasks, 12 meta-tasks and 5 scenario tasks, enabling a comprehensive and progressive assessment from basic capabilities to real-world use cases. We benchmark 11 leading LVLMs, including closed-source models such as GPT-4o and open-source models like LLaVA. Our evaluation validates the effectiveness of INS-MMBench and offers detailed insights into the strengths and limitations of current LVLMs on a variety of insurance-related multimodal tasks. We hope that INS-MMBench will accelerate the integration of LVLMs into the insurance industry and foster interdisciplinary research. Our dataset and evaluation code are available at https://github.com/FDU-INS/INS-MMBench.

</details>

---

## 208. Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning

- [ ] Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Aligning_Vision_to_Language_Annotation-Free_Multimodal_Knowledge_Graph_Construction_for_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Aligning_Vision_to_Language_Annotation-Free_Multimodal_Knowledge_Graph_Construction_for_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at https://github.com/Wings-Of-Disaster/VaLiK.

</details>

---

## 209. Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning

- [ ] Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Constructing_Ophthalmic_MLLM_for_Positioning-diagnosis_Collaboration_Through_Clinical_Cognitive_Chain_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Constructing_Ophthalmic_MLLM_for_Positioning-diagnosis_Collaboration_Through_Clinical_Cognitive_Chain_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) demonstrate significant potential in the field of medical diagnosis. However, they face critical challenges in specialized domains such as ophthalmology, particularly the fragmentation of annotation granularity and inconsistencies in clinical reasoning logic, which hinder precise cross-modal understanding. This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system. Fundus-Engine automates localization and leverages MLLM-based semantic expansion to integrate global disease classification, local object detection, and fine-grained feature analysis within a single fundus image. Additionally, by constructing a clinically aligned cognitive chain, it guides the model to generate interpretable reasoning paths. FundusExpert, fine-tuned with instruction data from FundusGen, achieves the best performance in ophthalmic question-answering tasks, surpassing the average accuracy of the 40B MedRegA by 26.6%. It also excels in zero-shot report generation tasks, achieving a clinical consistency of 77.0%, significantly outperforming GPT-4o's 47.6%. Furthermore, we reveal a scaling law between data quality and model capability (L \propto N^ 0.068 ), demonstrating that the cognitive alignment annotations in FundusGen enhance data utilization efficiency. By integrating region-level localization with diagnostic reasoning chains, our work develops a scalable, clinically-aligned MLLM and explores a pathway toward bridging the visual-language gap in specific MLLMs. Our project can be found at https://github.com/MeteorElf/FundusExpert.

</details>

---

## 210. Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow

- [ ] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Flow4Agent_Long-form_Video_Understanding_via_Motion_Prior_from_Optical_Flow_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Flow4Agent_Long-form_Video_Understanding_via_Motion_Prior_from_Optical_Flow_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-form video understanding has always been a challenging problem due to the significant redundancy in both temporal and spatial contents. This challenge is further exacerbated by the limited context length of Multimodal Large Language Models (MLLMs). To address this issue, many previous works have attempted to extract key video information, where the "key" is typically semantic-aware and heavily dependent on the CLIP model as prior. In this paper, we propose Flow4Agent, a novel framework that pioneeringly incorporates motion priors from optical flow to facilitate LLM-based long video understanding. Flow4Agent mitigates the redundancy in long videos at both temporal and spatial levels through two core modules: Temporal Granularity Optimization (TGO) adaptively refines frame-level hierarchies, which first leverages coarse flow priors to group similar visual contents and then applies semantic priors to filter out highly irrelevant scene information. Motion Token Pruning (MTP) further refines the intra-frame visual representations, pruning high-redundancy video tokens using fine-grained optical flow information. Extensive experiments demonstrate that our Flow4Agent outperforms existing methods across a wide range of video MLLM benchmarks, especially for hour-level video understanding tasks, achieving 64.7% on Video-MME, 71.4% on MLVU and 60.4% on LongVideoBench.

</details>

---

## 211. GLEAM: Enhanced Transferable Adversarial Attacks for Vision-Language Pre-training Models via Global-Local Transformations

- [ ] GLEAM: Enhanced Transferable Adversarial Attacks for Vision-Language Pre-training Models via Global-Local Transformations | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_GLEAM_Enhanced_Transferable_Adversarial_Attacks_for_Vision-Language_Pre-training_Models_via_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_GLEAM_Enhanced_Transferable_Adversarial_Attacks_for_Vision-Language_Pre-training_Models_via_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training (VLP) models leverage large-scale cross-modal pre-training to align vision and text modalities, achieving impressive performance on tasks like image-text retrieval and visual grounding. However, these models are highly vulnerable to adversarial attacks, raising critical concerns about their robustness and reliability in safety-critical applications. Existing black-box attack methods are limited by insufficient data augmentation mechanisms or the disruption of global semantic structures, leading to poor adversarial transferability. To address these challenges, we propose the Global-Local Enhanced Adversarial Multimodal attack (GLEAM), a unified framework for generating transferable adversarial examples in vision-language tasks. GLEAM introduces a local feature enhancement module that achieves diverse local deformations while maintaining global semantic and geometric integrity. It also incorporates a global distribution expansion module, which expands feature space coverage through dynamic transformations. Additionally, a cross-modal feature alignment module leverages intermediate adversarial states to guide text perturbations. This enhances cross-modal consistency and adversarial text transferability. Extensive experiments on Flickr30K and MSCOCO datasets show that GLEAM outperforms state-of-the-art methods, with over 10%-30% higher attack success rates in image-text retrieval tasks and over 30% improved transferability on large models like Claude 3.5 Sonnet and GPT-4o. GLEAM provides a robust tool for exposing vulnerabilities in VLP models and offers valuable insights into designing more secure and reliable vision-language systems. Our code is available at https://github.com/LuckAlex/GLEAM.

</details>

---

## 212. GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis

- [ ] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_GEMeX_A_Large-Scale_Groundable_and_Explainable_Medical_VQA_Benchmark_for_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_GEMeX_A_Large-Scale_Groundable_and_Explainable_Medical_VQA_Benchmark_for_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical Visual Question Answering (Med-VQA) combines computer vision and natural language processing to automatically answer clinical inquiries about medical images. However, current Med-VQA datasets exhibit two significant limitations: (1) they often lack visual and textual explanations for answers, hindering comprehension for patients and junior doctors; (2) they typically offer a narrow range of question formats, inadequately reflecting the diverse requirements in practical scenarios. These limitations pose significant challenges to the development of a reliable and user-friendly Med-VQA system. To address these challenges, we introduce a large-scale, Groundable, and Explainable Medical VQA benchmark for chest X-ray diagnosis (GEMeX), featuring several innovative components: (1) a multi-modal explainability mechanism that offers detailed visual and textual explanations for each question-answer pair, thereby enhancing answer comprehensibility; (2) four question types--open-ended, closed-ended, single-choice, and multiple-choice--to better reflect practical needs. With 151,025 images and 1,605,575 questions, GEMeX is the currently largest chest X-ray VQA dataset. Evaluation of 12 representative large vision language models (LVLMs) on GEMeX reveals suboptimal performance, underscoring the dataset's complexity. Meanwhile, we propose a strong model by fine-tuning an existing LVLM on the GEMeX training set. The substantial performance improvement showcases the dataset's effectiveness. The benchmark is available at www.med-vqa.com/GEMeX.

</details>

---

## 213. Keyframe-oriented Vision Token Pruning: Enhancing Efficiency of Large Vision Language Models on Long-Form Video Processing

- [ ] Keyframe-oriented Vision Token Pruning: Enhancing Efficiency of Large Vision Language Models on Long-Form Video Processing | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Keyframe-oriented_Vision_Token_Pruning_Enhancing_Efficiency_of_Large_Vision_Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Keyframe-oriented_Vision_Token_Pruning_Enhancing_Efficiency_of_Large_Vision_Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) demonstrate strong capabilities in jointly processing visual and textual data. However, they often incur substantial computational overhead due to redundant visual information, particularly in long-form video scenarios. Existing approaches predominantly focus on either vision token pruning, which may overlook spatio-temporal dependencies, or keyframe selection, which identifies informative frames but discards others, thus disrupting contextual continuity. In this work, we propose KVTP (Keyframe-oriented Vision Token Pruning), a novel framework that overcomes the drawbacks of token pruning and keyframe selection. By adaptively assigning pruning rates based on frame relevance to the query, KVTP effectively retains essential contextual information while significantly reducing redundant computation. To thoroughly evaluate the long-form video understanding capacities of VLMs, we curated and reorganized subsets from VideoMME, EgoSchema, and NextQA into a unified benchmark named SparseKV-QA that highlights real-world scenarios with sparse but crucial events. Our experiments with VLMs of various scales show that KVTP can reduce token usage by 80% without compromising spatiotemporal and contextual consistency, significantly cutting computation while maintaining the performance. These results demonstrate our approach's effectiveness in efficient long-video processing, facilitating more scalable VLM deployment.

</details>

---

## 214. METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models

- [ ] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_METEOR_Multi-Encoder_Collaborative_Token_Pruning_for_Efficient_Vision_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_METEOR_Multi-Encoder_Collaborative_Token_Pruning_for_Efficient_Vision_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision encoders serve as the cornerstone of multimodal understanding. Single-encoder architectures like CLIP exhibit inherent constraints in generalizing across diverse multimodal tasks, while recent multi-encoder fusion methods introduce prohibitive computational overhead to achieve superior performance using complementary visual representations from multiple vision encoders. To address this, we propose a progressive pruning framework, namely Multi-Encoder collaboraTivE tOken pRuning (METEOR), that eliminates redundant visual tokens across the encoding, fusion, and decoding stages for multi-encoder MLLMs. For multi-vision encoding, we discard redundant tokens within each encoder via a rank guided collaborative token assignment strategy. Subsequently, for multi-vision fusion, we combine the visual features from different encoders while reducing cross-encoder redundancy with cooperative pruning. Finally, we propose an adaptive token pruning method in the LLM decoding stage to further discard irrelevant tokens based on the text prompts with dynamically adjusting pruning ratios for specific task demands. To our best knowledge, this is the first successful attempt that achieves an efficient multi-encoder based vision language model with multi-stage pruning strategies. Extensive experiments on 11 benchmarks demonstrate the effectiveness of our proposed approach. Compared with EAGLE, a typical multi-encoder MLLMs, METEOR reduces 76% visual tokens with only 0.3% performance drop in average. The code is available at https://github.com/YuchenLiu98/METEOR.

</details>

---

## 215. Probabilistic Prototype Calibration of Vision-language Models for Generalized Few-shot Semantic Segmentation

- [ ] Probabilistic Prototype Calibration of Vision-language Models for Generalized Few-shot Semantic Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Probabilistic_Prototype_Calibration_of_Vision-language_Models_for_Generalized_Few-shot_Semantic_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Probabilistic_Prototype_Calibration_of_Vision-language_Models_for_Generalized_Few-shot_Semantic_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generalized Few-Shot Semantic Segmentation (GFSS) aims to extend a segmentation model to novel classes with only a few annotated examples while maintaining performance on base classes. Recently, pretrained vision-language models (VLMs) such as CLIP have been leveraged in GFSS to improve generalization on novel classes through multi-modal prototypes learning. However, existing prototype-based methods are inherently deterministic, limiting the adaptability of learned prototypes to diverse samples, particularly for novel classes with scarce annotations. To address this, our work propose Probabilistic Prototype Calibration Network (PPCN) - a probabilistic modeling framework over multi-modal prototypes from the pretrained CLIP, thus providing more adaptive prototype learning for GFSS. Specifically, PPCN first introduces a prototype calibration mechanism, which refines frozen textual prototypes with learnable visual calibration prototypes, leading to a more discriminative and adaptive representation. Furthermore, unlike deterministic prototype learning techniques, PPCN introduces distribution regularization over these calibration prototypes. This probabilistic formulation ensures structured and uncertainty-aware prototype learning, effectively mitigating overfitting to limited novel class data while enhancing generalization. Extensive experimental results on PASCAL-5^i and COCO-20^i datasets demonstrate that our proposed PPCN significantly outperforms state-of-the-art approaches across both GFSS and class-incremental setting. The source code will be released publicly.

</details>

---

## 216. Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation

- [ ] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Stepping_Out_of_Similar_Semantic_Space_for_Open-Vocabulary_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Stepping_Out_of_Similar_Semantic_Space_for_Open-Vocabulary_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary segmentation aims to achieve segmentation of arbitrary categories given unlimited text inputs as guidance. To achieve this, recent works have focused on developing various technical routes to exploit the potential of large-scale pre-trained vision-language models and have made significant progress on existing benchmarks. However, we find that existing test sets are limited in measuring the models' comprehension of "open-vocabulary" concepts, as their semantic space closely resembles the training space, even with many overlapping categories. To this end, we present a new benchmark named OpenBench that differs significantly from the training semantics. It is designed to better assess the model's ability to understand and segment a wide range of real-world concepts. When testing existing methods on OpenBench, we find that their performance diverges from the conclusions drawn on existing test sets. In addition, we propose a method named OVSNet to improve the segmentation performance for diverse and open scenarios. Through elaborate fusion of heterogeneous features and cost-free expansion of the training space, OVSNet achieves state-of-the-art results on both existing datasets and our proposed OpenBench. Corresponding analysis demonstrate the soundness and effectiveness of our proposed benchmark and method.

</details>

---

## 217. Visual-RFT: Visual Reinforcement Fine-Tuning

- [ ] Visual-RFT: Visual Reinforcement Fine-Tuning | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Visual-RFT_Visual_Reinforcement_Fine-Tuning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Visual-RFT_Visual_Reinforcement_Fine-Tuning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reinforcement Fine-Tuning (RFT) in Large Reasoning Models like OpenAI o1 learns from feedback on its answers, which is especially useful in applications when fine-tuning data is scarce.Recent open-source work like DeepSeek-R1 demonstrates that reinforcement learning with verifiable reward is possibly one key direction in reproducing o1.While the R1-style model has demonstrated success in language models, its application in multi-modal domains remains under-explored.This work introduces Visual Reinforcement Fine-Tuning (Visual-RFT), which further extends the application areas of RFT on visual tasks.Specifically, Visual-RFT first uses Large Vision-Language Models (LVLMs) to generate multiple responses containing reasoning tokens and final answers for each input, and then uses our proposed visual perception verifiable reward functions to update the model via the policy optimization algorithm such as Group Relative Policy Optimization (GRPO).We design different verifiable reward functions for different perception tasks, such as the Intersection over Union (IoU) reward for object detection.Experimental results on fine-grained image classification, few-shot object detection, reasoning grounding, as well as open-vocabulary object detection benchmarks show the competitive performance and advanced generalization ability of Visual-RFT compared with Supervised Fine-tuning (SFT).For example, Visual-RFT improves accuracy by 24.3% over the baseline in one-shot fine-grained image classification with around 100 samples.In few-shot object detection, Visual-RFT also exceeds the baseline by 21.0 on COCO's 4-shot setting and 15.4 on LVIS.Our Visual-RFT represents a paradigm shift in fine-tuning LVLMs, offering a data-efficient, reward-driven approach that enhances reasoning and adaptability for domain-specific tasks.

</details>

---

## 218. When Lighting Deceives: Exposing Vision-Language Models' Illumination Vulnerability Through Illumination Transformation Attack

- [ ] When Lighting Deceives: Exposing Vision-Language Models' Illumination Vulnerability Through Illumination Transformation Attack | https://openaccess.thecvf.com/content/ICCV2025/html/Liu_When_Lighting_Deceives_Exposing_Vision-Language_Models_Illumination_Vulnerability_Through_Illumination_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_When_Lighting_Deceives_Exposing_Vision-Language_Models_Illumination_Vulnerability_Through_Illumination_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have achieved remarkable success in various tasks, yet their robustness to real-world illumination variations remains largely unexplored. To bridge this gap, we propose Illumination Transformation Attack (ITA), the first framework to systematically assess VLMs' robustness against illumination changes. However, there still exist two key challenges: (1) how to model global illumination with fine-grained control to achieve diverse lighting conditions and (2) how to ensure adversarial effectiveness while maintaining naturalness. To address the first challenge, we innovatively decompose global illumination into multiple parameterized point light sources based on the illumination rendering equation. This design enables us to model more diverse lighting variations that previous methods could not capture. Then, by integrating these parameterized lighting variations with physics-based lighting reconstruction techniques, we could precisely render such light interactions in the original scenes, finally meeting the goal of fine-grained lighting control. For the second challenge, by controlling illumination through the lighting reconstrution model's latent space rather than direct pixel manipulation, we inherently preserve physical lighting priors. Furthermore, to prevent potential reconstruction artifacts, we design additional perceptual constraints for maintaining visual consistency with original images and diversity constraints for avoiding light source convergence. Extensive experiments demonstrate that our ITA could significantly reduce the performance of advanced VLMs, e.g., LLaVA-1.6, while possessing competitive naturalness, exposing VLMS' critical illuminiation vulnerabilities.

</details>

---

## 219. LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs

- [ ] LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Lou_LLaVA-SP_Enhancing_Visual_Representation_with_Visual_Spatial_Tokens_for_MLLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lou_LLaVA-SP_Enhancing_Visual_Representation_with_Visual_Spatial_Tokens_for_MLLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The architecture of multimodal large language models (MLLMs) commonly connects a vision encoder, often based on CLIP-ViT, to a large language model. While CLIP-ViT works well for capturing global image features, it struggles to model local relationships between adjacent patches, leading to weaker visual representation, which in turn affects the detailed understanding ability of MLLMs. To solve this, we propose LLaVA-SP, which only adds six spatial visual tokens to the original visual tokens to enhance the visual representation. Our approach offers three key advantages: 1) We propose a novel Projector, which uses convolutional kernels to derive visual spatial tokens from ViT patch features, simulating two visual spatial ordering approaches: "from central region to global" and "from abstract to specific". Then, a cross-attention mechanism is applied to fuse fine-grained visual information, enriching the overall visual representation. 2) We present two model variants: LLaVA-SP-Cropping, which focuses on detail features through progressive cropping, and LLaVA-SP-Pooling, which captures global semantics through adaptive pooling, enabling the model to handle diverse visual understanding tasks. 3) Extensive experiments show that LLaVA-SP, fine-tuned with LoRA, achieves significant performance improvements across various multimodal benchmarks, outperforming the state-of-the-art LLaVA-1.5 model in multiple tasks with nearly identical inference latency. The code and models are available at https://github.com/CnFaker/LLaVA-SP.

</details>

---

## 220. B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens

- [ ] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens | https://openaccess.thecvf.com/content/ICCV2025/html/Lu_B-VLLM_A_Vision_Large_Language_Model_with_Balanced_Spatio-Temporal_Tokens_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lu_B-VLLM_A_Vision_Large_Language_Model_with_Balanced_Spatio-Temporal_Tokens_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Vision Large Language Models (VLLMs) with integrated vision encoders have shown promising performance in vision understanding. They encode visual content into sequences of visual tokens, enabling joint processing of visual and textual data. However, understanding videos, especially long videos, remains a challenge as the rapid growth of visual tokens during video encoding risks exceeding VLLMs' context window length and significantly escalates computational cost. To restrict the number of visual tokens, existing VLLMs either: (1) uniformly downsample videos into a fixed number of frames or (2) reducing the number of visual tokens encoded from each frame. We argue that the former neglects temporal dynamics in videos, while the latter fails to preserve spatial details within individual frame. In this work, we propose Balanced-VLLM (B-VLLM), a novel VLLM framework designed to model task relevant spatio-temporal cues, while restricting the number of visual tokens within the VLLM's context window length. Central to our framework is a text-conditioned adaptive frame selection module that dynamically identifies task-relevant frames, which are further de-duplicated with a temporal frame token merging strategy.The visual tokens of these frames then undergo spatial token sampling and an optional spatial token merging strategy for granular control against the token budget. Experiments demonstrate the effectiveness of B-VLLM in balancing the number of frames and visual tokens, moreover, our proposed method introduce 10% performance gain on MVBench. Our code will be publicly available.

</details>

---

## 221. Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection

- [ ] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Dynamic-DINO_Fine-Grained_Mixture_of_Experts_Tuning_for_Real-time_Open-Vocabulary_Object_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Dynamic-DINO_Fine-Grained_Mixture_of_Experts_Tuning_for_Real-time_Open-Vocabulary_Object_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Mixture of Experts (MoE) architecture has excelled in Large Vision-Language Models (LVLMs), yet its potential in real-time open-vocabulary object detectors, which also leverage large-scale vision-language datasets but smaller models, remains unexplored. This work investigates this domain, revealing intriguing insights. In the shallow layers, experts tend to cooperate with diverse peers to expand the search space. While in the deeper layers, fixed collaborative structures emerge, where each expert maintains 2-3 fixed partners and distinct expert combinations are specialized in processing specific patterns. Concretely, we propose Dynamic-DINO, which extends Grounding DINO 1.5 Edge from a dense model to a dynamic inference framework via an efficient MoE-Tuning strategy. Additionally, we design a granularity decomposition mechanism to decompose the Feed-Forward Network (FFN) of base model into multiple smaller expert networks, expanding the subnet search space. To prevent performance degradation at the start of fine-tuning, we further propose a pre-trained weight allocation strategy for the experts, coupled with a specific router initialization. During inference, only the input-relevant experts are activated to form a compact subnet. Experiments show that, pretrained with merely 1.56M open-source data, Dynamic-DINO outperforms Grounding DINO 1.5 Edge, pretrained on the private Grounding20M dataset.

</details>

---

## 222. FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection

- [ ] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Lu_FA_Forced_Prompt_Learning_of_Vision-Language_Models_for_Out-of-Distribution_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lu_FA_Forced_Prompt_Learning_of_Vision-Language_Models_for_Out-of-Distribution_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs) have advanced out-of-distribution (OOD) detection recently. However, existing CLIP-based methods often focus on learning OOD-related knowledge to improve OOD detection, showing limited generalization or reliance on external large-scale auxiliary datasets. In this study, instead of delving into the intricate OOD-related knowledge, we propose an innovative CLIP-based framework based on Forced prompt leArning (FA), designed to make full use of the In-Distribution (ID) knowledge and ultimately boost the effectiveness of OOD detection. Our key insight is to learn a prompt (i.e., forced prompt) that contains more diversified and richer descriptions of the ID classes beyond the textual semantics of class labels. Specifically, it promotes better discernment for ID images, by forcing more notable semantic similarity between ID images and the learnable forced prompt. Moreover, we introduce a forced coefficient, encouraging the forced prompt to learn more comprehensive and nuanced descriptions of the ID classes. In this way, FA is capable of achieving notable improvements in OOD detection, even when trained without any external auxiliary datasets, while maintaining an identical number of trainable parameters as CoOp. Extensive empirical evaluations confirm our method consistently outperforms current state-of-the-art methods. Code is available at https://github.com/0xFAFA/FA.

</details>

---

## 223. GenieBlue: Integrating both Linguistic and Multimodal Capabilities for Large Language Models on Mobile Devices

- [ ] GenieBlue: Integrating both Linguistic and Multimodal Capabilities for Large Language Models on Mobile Devices | https://openaccess.thecvf.com/content/ICCV2025/html/Lu_GenieBlue_Integrating_both_Linguistic_and_Multimodal_Capabilities_for_Large_Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lu_GenieBlue_Integrating_both_Linguistic_and_Multimodal_Capabilities_for_Large_Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have enabled their deployment on mobile devices. However, challenges persist in maintaining strong language capabilities and ensuring hardware compatibility, both of which are crucial for user experience and practical deployment efficiency. In our deployment process, we observe that existing MLLMs often face performance degradation on pure language tasks, and the current NPU platforms on smartphones do not support the MoE architecture, which is commonly used to preserve pure language capabilities during multimodal training. To address these issues, we systematically analyze methods to maintain pure language capabilities during the training of MLLMs, focusing on both training data and model architecture aspects. Based on these analyses, we propose GenieBlue, an efficient MLLM structural design that integrates both linguistic and multimodal capabilities for LLMs on mobile devices. GenieBlue freezes the original LLM parameters during MLLM training to maintain pure language capabilities. It acquires multimodal capabilities by duplicating specific transformer blocks for full fine-tuning and integrating lightweight LoRA modules. This approach preserves language capabilities while achieving comparable multimodal performance through extensive training. Deployed on smartphone NPUs, GenieBlue demonstrates efficiency and practicality for applications on mobile devices.

</details>

---

## 224. Hierarchical Divide-and-Conquer Grouping for Classification Adaptation of Pre-Trained Models

- [ ] Hierarchical Divide-and-Conquer Grouping for Classification Adaptation of Pre-Trained Models | https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Hierarchical_Divide-and-Conquer_Grouping_for_Classification_Adaptation_of_Pre-Trained_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Hierarchical_Divide-and-Conquer_Grouping_for_Classification_Adaptation_of_Pre-Trained_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing adaptation methods of pre-trained vision-language models like CLIP often rely on base-class samples during fine-tuning, introducing systematic biases that distort decision boundaries and degrade performance on novel classes. In this work, we break new ground by proposing a hierarchical divide-and-conquer framework that addresses classification bias at its root. Our method first segregates the label space into base and novel subspaces, ensuring domain separation. Subsequently, it employs text-embedding clustering within each subspace to decompose ambiguous intra-domain classes into disentangled, fine-grained clusters. This two-stage grouping strategy not only alleviates class confusion but also enables domain-specific model training in isolated subspaces, fostering specialized learning without overfitting base categories. Experiments on three classification benchmarks reveal that our approach achieves state-of-the-art performance, surpassing the second-best competitor by 10% average accuracy.

</details>

---

## 225. ReAL-AD: Towards Human-Like Reasoning in End-to-End Autonomous Driving

- [ ] ReAL-AD: Towards Human-Like Reasoning in End-to-End Autonomous Driving | https://openaccess.thecvf.com/content/ICCV2025/html/Lu_ReAL-AD_Towards_Human-Like_Reasoning_in_End-to-End_Autonomous_Driving_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Lu_ReAL-AD_Towards_Human-Like_Reasoning_in_End-to-End_Autonomous_Driving_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

End-to-end autonomous driving has emerged as a promising approach to unify perception, prediction, and planning within a single framework, reducing information loss and improving adaptability. However, existing methods often rely on fixed and sparse trajectory supervision, limiting their ability to capture the hierarchical reasoning process that human drivers naturally employ. To bridge this gap, we propose ReAL-AD, a Reasoning-Augmented Learning framework that structures decision-making in autonomous driving based on the three-tier human cognitive model: Driving Strategy, Driving Decision, and Driving Operation, where Vision-Language Models (VLMs) are incorporated to enhance situational awareness and structured reasoning across these levels. Specifically, we introduce: (1) the Strategic Reasoning Injector, which formulates high-level driving strategies by interpreting complex traffic contexts from VLM-generated insights; (2) the Tactical Reasoning Integrator, which refines strategic intent into interpretable tactical choices such as lane changes, overtaking, and speed adjustments; and (3) the Hierarchical Trajectory Decoder, which progressively translates tactical decisions into precise control actions for smooth and human-like trajectory execution. Extensive evaluations show that integrating our framework improves planning accuracy and safety by over 30%, making end-to-end autonomous driving more interpretable and aligned with human-like hierarchical reasoning.

</details>

---

## 226. CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model

- [ ] CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model | https://openaccess.thecvf.com/content/ICCV2025/html/Luo_CalliReader_Contextualizing_Chinese_Calligraphy_via_an_Embedding-Aligned_Vision-Language_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Luo_CalliReader_Contextualizing_Chinese_Calligraphy_via_an_Embedding-Aligned_Vision-Language_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chinese calligraphy, a UNESCO Heritage, remains computationally challenging due to visual ambiguity and cultural complexity. Existing AI systems fail to contextualize their intricate scripts, because of limited annotated data and poor visual-semantic alignment. We propose CalliReader, a vision-language model (VLM) that solves the Chinese Calligraphy Contextualization (CC^2) problem through three innovations: (1) character-wise slicing for precise character extraction and sorting, (2) CalliAlign for visual-text token compression and alignment, (3) embedding instruction tuning (e-IT) for improving alignment and addressing data scarcity. We also build CalliBench, the first benchmark for full-page calligraphic contextualization, addressing three critical issues in previous OCR and VQA approaches: fragmented context, shallow reasoning, and hallucination. Extensive experiments including user studies have been conducted to verify our CalliReader's superiority to other state-of-the-art methods and even human professionals in page-level calligraphy recognition and interpretation, achieving higher accuracy while reducing hallucination. Comparisons with reasoning models highlight the importance of accurate recognition as a prerequisite for reliable comprehension. Quantitative analyses validate CalliReader's efficiency; evaluations on document and real-world benchmarks confirm its robust generalization ability.

</details>

---

## 227. Dual-Process Image Generation

- [ ] Dual-Process Image Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Luo_Dual-Process_Image_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Luo_Dual-Process_Image_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prior methods for controlling image generation are limited in their ability to be taught new tasks. In contrast, vision-language models, or VLMs, can learn tasks in-context and produce the correct outputs for a given input. We propose a dual-process distillation scheme that allows feed-forward image generators to learn new tasks from deliberative VLMs. Our scheme uses a VLM to rate the generated images and backpropagates this gradient to update the weights of the image generator. Our general framework enables a wide variety of new control tasks through the same text-and-image based interface. We showcase a handful of applications of this technique for different types of control signals, such as commonsense inferences and visual prompts. With our method, users can implement multimodal controls for properties such as color palette, line weight, horizon position, and relative depth within a matter of minutes.

</details>

---

## 228. Visual Test-time Scaling for GUI Agent Grounding

- [ ] Visual Test-time Scaling for GUI Agent Grounding | https://openaccess.thecvf.com/content/ICCV2025/html/Luo_Visual_Test-time_Scaling_for_GUI_Agent_Grounding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Luo_Visual_Test-time_Scaling_for_GUI_Agent_Grounding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce RegionFocus, a visual test-time scaling approach for Vision Language Model Agents. Understanding webpages is challenging due to the visual complexity of GUI images and the large number of interface elements, making accurate action selection difficult. Our approach dynamically zooms in on relevant regions, reducing background clutter and improving grounding accuracy. To support this process, we propose an image-as-map mechanism that visualizes key landmarks at each step, providing a transparent action record and enables the agent to effectively choose among action candidates. Even with a simple region selection strategy, we observe significant performance gains of 28+% on Screenspot-pro and 24+% on WebVoyager benchmarks on top of two state-of-the-art open vision language model agents, UI-TARS-72B and Qwen2.5-VL-72B, highlighting the effectiveness of visual test-time scaling in interactive settings. We achieve a new state-of-the-art grounding performance of 61.6% on the ScreenSpot-Pro benchmark by applying RegionFocus to a Qwen2.5-VL-72B model. Our code is publicly available at https://github.com/tiangeluo/RegionFocus.

</details>

---

## 229. When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning

- [ ] When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning | https://openaccess.thecvf.com/content/ICCV2025/html/Luo_When_Large_Vision-Language_Model_Meets_Large_Remote_Sensing_Imagery_Coarse-to-Fine_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Luo_When_Large_Vision-Language_Model_Meets_Large_Remote_Sensing_Imagery_Coarse-to-Fine_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Efficient vision-language understanding of large Remote Sensing Images (RSIs) is meaningful but challenging. Current Large Vision-Language Models (LVLMs) typically employ limited pre-defined grids to process images, leading to information loss when handling gigapixel RSIs. Conversely, using unlimited grids significantly increases computational costs. To preserve image details while reducing computational complexity, we propose a text-guided token pruning method with Dynamic Image Pyramid (DIP) integration. Our method introduces: (i) a Region Focus Module (RFM) that leverages text-aware region localization capability to identify critical vision tokens, and (ii) a coarse-to-fine image tile selection and vision token pruning strategy based on DIP, which is guided by RFM outputs and avoids directly processing the entire large imagery. Additionally, existing benchmarks for evaluating LVLMs' perception ability on large RSI suffer from limited question diversity and constrained image sizes. We construct a new benchmark named LRS-VQA, which contains 7,333 QA pairs across 8 categories, with image length up to 27,328 pixels. Our method outperforms existing high-resolution strategies on four datasets using the same data. Moreover, compared to existing token reduction methods, our approach demonstrates higher efficiency under high-resolution settings.

</details>

---

## 230. GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers

- [ ] GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_GenHancer_Imperfect_Generative_Models_are_Secretly_Strong_Vision-Centric_Enhancers_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_GenHancer_Imperfect_Generative_Models_are_Secretly_Strong_Vision-Centric_Enhancers_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The synergy between generative and discriminative models receives growing attention. While discriminative Contrastive Language-Image Pre-Training (CLIP) excels in high-level semantics, it struggles with perceiving fine-grained visual details. Generally, to enhance representations, generative models take CLIP's visual features as conditions for reconstruction. However, the underlying principle remains underexplored. In this work, we empirically found that visually perfect generations are not always optimal for representation enhancement. The essence lies in effectively extracting fine-grained knowledge from generative models while mitigating irrelevant information. To explore critical factors, we delve into three aspects: (1) Conditioning mechanisms: We found that even a small number of local tokens can drastically reduce the difficulty of reconstruction, leading to collapsed training. We thus conclude that utilizing only global visual tokens as conditions is the most effective strategy. (2) Denoising configurations: We observed that end-to-end training introduces extraneous information. To address this, we propose a two-stage training strategy to prioritize learning useful visual knowledge. Additionally, we demonstrate that lightweight denoisers can yield remarkable improvements. (3) Generation paradigms: We explore both continuous and discrete denoisers with desirable outcomes, validating the versatility of our method. Through our in-depth explorations, we have finally arrived at an effective method, namely GenHancer, which consistently outperforms prior arts on the MMVP-VLM benchmark, e.g., 6.0% on OpenAICLIP. The enhanced CLIP can be further plugged into multimodal large language models for better vision-centric performance. All the models and codes are made publicly available.

</details>

---

## 231. HPSv3: Towards Wide-Spectrum Human Preference Score

- [ ] HPSv3: Towards Wide-Spectrum Human Preference Score | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Evaluating text-to-image generation models requires alignment with human perception, yet existing human-centric metrics are constrained by limited data coverage, suboptimal feature extraction, and inefficient loss functions. To address these challenges, we introduce Human Preference Score v3 (HPSv3). (1) We release HPDv3, the first wide-spectrum human preference dataset integrating 1.08M text-image pairs and 1.17M annotated pairwise comparisons from state-of-the-art generative models and low to high-quality real-world images. (2) We introduce a VLM-based preference model trained using an uncertainty-aware ranking loss for fine-grained ranking. Besides, we propose Chain-of-Human-Preference (CoHP), an iterative image refinement method that enhances quality without extra data, using HPSv3 to select the best image at each step. Extensive experiments demonstrate that HPSv3 serves as a robust metric for wide-spectrum image evaluation, and CoHP offers an efficient and human-aligned approach to improve image generation quality. Code and dataset is available at https://mizzenai.github.io/HPSv3/

</details>

---

## 232. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models

- [ ] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Heuristic-Induced_Multimodal_Risk_Distribution_Jailbreak_Attack_for_Multimodal_Large_Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Heuristic-Induced_Multimodal_Risk_Distribution_Jailbreak_Attack_for_Multimodal_Large_Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective jailbreak attacks poses unique challenges, especially given the highly constrained adversarial capabilities in real-world deployment scenarios. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which is black-box and consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to distribute harmful semantics into multiple modalities to effectively circumvent the single-modality protection mechanisms of MLLMs. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps MLLMs reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. HIMRD achieves an average attack success rate (ASR) of 90% across seven open-source MLLMs and an average ASR of around 68% in three closed-source MLLMs. HIMRD reveals cross-modal security vulnerabilities in current MLLMs and underscores the imperative for developing defensive strategies to mitigate such emerging risks. Code is available at https://github.com/MaTengSYSU/HIMRD-jailbreak.

</details>

---

## 233. Multimodal Prompt Alignment for Facial Expression Recognition

- [ ] Multimodal Prompt Alignment for Facial Expression Recognition | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Multimodal_Prompt_Alignment_for_Facial_Expression_Recognition_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Multimodal_Prompt_Alignment_for_Facial_Expression_Recognition_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has been widely adopted to efficiently adapt vision-language models (VLMs) like CLIP for various downstream tasks. Despite their success, current VLM-based facial expression recognition (FER) methods struggle to capture fine-grained textual-visual relationships, which are essential for distinguishing subtle differences between facial expressions. To address this challenge, we propose a multimodal prompt alignment framework for FER, called MPA-FER, that provides fine-grained semantic guidance to the learning process of prompted visual features, resulting in more precise and interpretable representations. Specifically, we introduce a multi-granularity hard prompt generation strategy that utilizes a large language model (LLM) like ChatGPT to generate detailed descriptions for each facial expression. The LLM-based external knowledge is injected into the soft prompts by minimizing the feature discrepancy between the soft prompts and the hard prompts. To preserve the generalization abilities of the pretrained CLIP model, our approach incorporates prototype-guided visual feature alignment, ensuring that the prompted visual features from the frozen image encoder align closely with class-specific prototypes. Additionally, we propose a cross-modal global-local alignment module that focuses on expression-relevant facial features, further improving the alignment between textual and visual features. Extensive experiments demonstrate our framework outperforms state-of-the-art methods on three FER benchmark datasets, while retaining the benefits of the pretrained model and minimizing computational costs.

</details>

---

## 234. ReMP-AD: Retrieval-enhanced Multi-modal Prompt Fusion for Few-Shot Industrial Visual Anomaly Detection

- [ ] ReMP-AD: Retrieval-enhanced Multi-modal Prompt Fusion for Few-Shot Industrial Visual Anomaly Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Industrial visual inspection is crucial for detecting defects in manufactured products, but it traditionally relies on human operators, leading to inefficiencies. Industrial Visual Anomaly Detection (IVAD) has emerged as a promising solution, with methods such as zero-shot, few-shot, and reconstruction-based techniques. However, zero-shot methods struggle with subtle anomalies, and reconstruction-based methods fail to capture fine-grained details. Few-shot methods, which use limited samples and prompts, offer a more efficient approach. Despite their promise, challenges remain in managing intra-class variation among references and in effectively extracting more representative anomaly features.This paper presents Retrieval-enhanced Multi-modal Prompt Fusion Anomaly Detection (ReMP-AD), a framework that introduces Intra-Class Token Retrieval (ICTR) to reduce noise in the memory bank and Vision-Language Prior Fusion (VLPF) to guide the encoder in capturing more distinctive and relevant features of anomalies. Experiments on the VisA and MVTec-AD datasets demonstrate that ReMP-AD outperforms existing methods, achieving 97.8%/94.1% performance in 4-shot anomaly segmentation and classification. Our approach also shows strong results on the PCB-Bank dataset, highlighting its effectiveness in few-shot industrial anomaly detection. Code is available at https://github.com/cshcma/ReMP-AD.git

</details>

---

## 235. Unknown Text Learning for CLIP-based Few-Shot Open-set Recognition

- [ ] Unknown Text Learning for CLIP-based Few-Shot Open-set Recognition | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Unknown_Text_Learning_for_CLIP-based_Few-Shot_Open-set_Recognition_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Unknown_Text_Learning_for_CLIP-based_Few-Shot_Open-set_Recognition_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, vision-language models (e.g., CLIP) with prompt learning have shown great potential in few-shot learning. However, an open issue remains for the effective extension of CLIP-based models to few-shot open-set recognition (FSOR), which requires classifying known classes and detecting unknown samples using a few known samples. The core challenge is that unknown samples and their textual descriptions are unavailable. To address this, we propose an Unknown Text Learning (UTL) method for CLIP-based FSOR tasks with only known samples. Specifically, UTL involves two key components, i.e., universal unknown words optimization (U^ 2 WO) and unknown label smoothing (ULS). Specifically, U^ 2 WO constructs the universal space of unknown words with basis vectors and characterizes unknown text based on a linear combination of those basis vectors. To efficiently learn unknown text without unknown samples, ULS is presented to perform contrast learning between unknown text and known samples by regulating the label of unknown classes to a small constant, which flexibly empowers unknown text to be non-matching with and confused on known visual samples. In addition, our UTL incorporates an additional context for known classes to mitigate conflicts of context optimization between known and unknown classes. UTL effectively regularizes the predicted probability by integrating learnable unknown text. Experimental results on various benchmarks show that our UTL is superior to its counterparts while achieving state-of-the-art performance.

</details>

---

## 236. VisionMath: Vision-Form Mathematical Problem-Solving

- [ ] VisionMath: Vision-Form Mathematical Problem-Solving | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_VisionMath_Vision-Form_Mathematical_Problem-Solving_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_VisionMath_Vision-Form_Mathematical_Problem-Solving_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mathematical problems in real-world scenarios are often presented in a purely vision-form, where textual problem statement and accompanying math figures, e.g., geometry figures and functional graphs, are integrated into a single image. This vision-form problem-solving task requires precise comprehension and reasoning on both textual and graphical elements in the images, posing significant challenge to current Multimodal Large Language Models (MLLMs), which process text and math figures in isolation. In this work, we propose VisionMath, the first exploration for vision-form mathematical problem-solving model, which employs a three-stage progressive multimodal reasoning alignment strategy to systematically enhance task-specific capabilities. Building upon a LLM proficient in unimodal mathematical reasoning, VisionMath first establishes foundational OCR capabilities through capturing rendered mathematical problem images. Subsequently, the model develops comprehensive understanding of figure structures and properties via learning from figure descriptions and mathematical educational videos. Finally, the model's reasoning capacity is activated using carefully constructed visual-form problem-solving datasets VisionMath-IT with chain-of-thought annotations. For comprehensive evaluation, we construct multilingual benchmarks covering diverse problem types, including geometry, algebra, function problems in both English and Chinese. Our model weights, data and code will be made available at https://github.com/mengqiDyangge/VisionMath.

</details>

---

## 237. X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation

- [ ] X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation | https://openaccess.thecvf.com/content/ICCV2025/html/Ma_X2I_Seamless_Integration_of_Multimodal_Understanding_into_Diffusion_Transformer_via_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_X2I_Seamless_Integration_of_Multimodal_Understanding_into_Diffusion_Transformer_via_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-image (T2I) models are well known for their ability to produce highly realistic images, while multimodal large language models (MLLMs) are renowned for their proficiency in understanding and integrating multiple modalities. However, currently there is no straightforward and efficient framework to transfer the multimodal comprehension abilities of MLLMs to T2I models to enable them to understand multimodal inputs. In this paper, we propose the X2I framework, which endows Diffusion Transformer (DiT) models with the capability to comprehend various modalities, including multilingual text, screenshot documents, images, videos, and audio. X2I is trained using merely 100K English corpus with 160 GPU hours. Building on the DiT teacher model, we adopt an innovative distillation method to extract the inference capabilities of the teacher model and design a lightweight AlignNet structure to serve as an intermediate bridge. Compared to the teacher model, X2I shows a decrease in performance degradation of less than 1% while gaining various multimodal understanding abilities. Furthermore, it is applicable for LoRA training in the context of image-text to image generation, filling a void in the industry in this area. We further design a simple LightControl to enhance the fidelity of instructional image editing. Finally, extensive experiments demonstrate the effectiveness, efficiency, multifunctionality, and transferability of our X2I.

</details>

---

## 238. CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers

- [ ] CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers | https://openaccess.thecvf.com/content/ICCV2025/html/Mallis_CAD-Assistant_Tool-Augmented_VLLMs_as_Generic_CAD_Task_Solvers_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Mallis_CAD-Assistant_Tool-Augmented_VLLMs_as_Generic_CAD_Task_Solvers_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We propose CAD-Assistant, a general-purpose CAD agent for AI-assisted design. Our approach is based on a powerful Vision and Large Language Model (VLLM) as a planner and a tool-augmentation paradigm using CAD-specific tools. CAD-Assistant addresses multimodal user queries by generating actions that are iteratively executed on a Python interpreter equipped with the FreeCAD software, accessed via its Python API. Our framework is able to assess the impact of generated CAD commands on geometry and adapts subsequent actions based on the evolving state of the CAD design. We consider a wide range of CAD-specific tools including a sketch image parameterizer, rendering modules, a 2D cross-section generator, and other specialized routines. CAD-Assistant is evaluated on multiple CAD benchmarks, where it outperforms VLLM baselines and supervised task-specific methods. Beyond existing benchmarks, we qualitatively demonstrate the potential of tool-augmented VLLMs as general-purpose CAD solvers across diverse workflows.

</details>

---

## 239. Controlling Multimodal LLMs via Reward-guided Decoding

- [ ] Controlling Multimodal LLMs via Reward-guided Decoding | https://openaccess.thecvf.com/content/ICCV2025/html/Manas_Controlling_Multimodal_LLMs_via_Reward-guided_Decoding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Manas_Controlling_Multimodal_LLMs_via_Reward-guided_Decoding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As Multimodal Large Language Models (MLLMs) gain widespread applicability, it is becoming increasingly desirable to adapt them for diverse user needs. In this paper, we study the adaptation of MLLMs through controlled decoding. To achieve this, we introduce the first method for reward-guided decoding of MLLMs and demonstrate its application in improving their visual grounding. Our method involves building reward models for visual grounding and using them to guide the MLLM's decoding process. Concretely, we build two separate reward models to independently control the degree of object precision and recall in the model's output. Our approach enables on-the-fly controllability of an MLLM's inference process in two ways: first, by giving control over the relative importance of each reward function during decoding, allowing a user to dynamically trade off object precision for recall in image captioning tasks; second, by giving control over the breadth of the search during decoding, allowing the user to control the trade-off between the amount of test-time compute and the degree of visual grounding. We evaluate our method on standard object hallucination benchmarks, showing that it provides significant controllability over MLLM inference, while consistently outperforming existing hallucination mitigation methods.

</details>

---

## 240. Visual Modality Prompt for Adapting Vision-Language Object Detectors

- [ ] Visual Modality Prompt for Adapting Vision-Language Object Detectors | https://openaccess.thecvf.com/content/ICCV2025/html/Medeiros_Visual_Modality_Prompt_for_Adapting_Vision-Language_Object_Detectors_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Medeiros_Visual_Modality_Prompt_for_Adapting_Vision-Language_Object_Detectors_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The zero-shot performance of object detectors degrades when tested on different modalities, such as infrared and depth. While recent work has explored image translation techniques to adapt detectors to new modalities, these methods are limited to a single modality and traditional detectors. Recently, vision-language detectors (VLDs), such as YOLO-World and Grounding DINO, have shown promising zero-shot capabilities; however, they have not yet been adapted for other visual modalities. Traditional fine-tuning approaches compromise the zero-shot capabilities of the detectors. The visual prompt strategies commonly used for classification with vision-language models apply the same linear prompt translation to each image, making them less effective. To address these limitations, we propose ModPrompt, a visual prompt strategy to adapt VLDs to new modalities without degrading zero-shot performance. In particular, an encoder-decoder visual prompt strategy is proposed, further enhanced by the integration of inference-friendly modality prompt decoupled residual, facilitating a more robust adaptation. Empirical benchmarking results show our method for modality adaptation on YOLO-World and Grounding DINO for challenging infrared (LLVIP, FLIR) and depth (NYUv2) datasets, achieving performance comparable to full fine-tuning while preserving the model's zero-shot capability. Our code is available at https://github.com/heitorrapela/ModPrompt.

</details>

---

## 241. Auxiliary Prompt Tuning of Vision-Language Models for Few-Shot Out-of-Distribution Detection

- [ ] Auxiliary Prompt Tuning of Vision-Language Models for Few-Shot Out-of-Distribution Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Miao_Auxiliary_Prompt_Tuning_of_Vision-Language_Models_for_Few-Shot_Out-of-Distribution_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Miao_Auxiliary_Prompt_Tuning_of_Vision-Language_Models_for_Few-Shot_Out-of-Distribution_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in CLIP-based out-of-distribution (OOD) detection have shown promising results via regularization on prompt tuning, leveraging background features extracted from a few in-distribution (ID) samples as proxies for OOD features.However, these methods suffer from an inherent limitation: a lack of diversity in the extracted OOD features from the few-shot ID data.To address this issue, we propose to leverage external datasets as auxiliary outlier data (i.e., pseudo OOD samples) to extract rich, diverse OOD features, with the features from not only background regions but also foreground object regions, thereby supporting more discriminative prompt tuning for OOD detection. We further introduce Auxiliary Prompt Tuning (APT), a novel framework that can be used as a plug-in module to enable existing prompt tuning-based methods to utilize the auxiliary data for more accurate OOD detection.There are two key challenges of utilizing those auxiliary data in prompt tuning, including I) foreground-background decomposition of unlabeled auxiliary data with diverse outlying objects and II) optimization of foreground OOD features. APT tackles challenge I with an adaptive logit-based Kullback-Leibler divergence method and challenge II by constructing foreground-background pairs for each foreground region to enable effective exploitation of foreground OOD features. Extensive experiments on standard and hard OOD benchmarks show that APT achieves state-of-the-art performance, obtaining significant improvements in challenging scenarios, e.g., hard OOD and 1-shot detection.

</details>

---

## 242. FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation

- [ ] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation | https://openaccess.thecvf.com/content/ICCV2025/html/Miao_FedVLA_Federated_Vision-Language-Action_Learning_with_Dual_Gating_Mixture-of-Experts_for_Robotic_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Miao_FedVLA_Federated_Vision-Language-Action_Learning_with_Dual_Gating_Mixture-of-Experts_for_Robotic_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language-Action (VLA) models have significantly advanced robotic manipulation by enabling robots to interpret language instructions for task execution. However, training these models often relies on large-scale user-specific data, raising concerns about privacy and security, which in turn limits their broader adoption. To address this, we propose FedVLA, the first federated VLA learning framework, enabling distributed model training that preserves data privacy without compromising performance. Our framework integrates task-aware representation learning, adaptive expert selection, and expert-driven federated aggregation, enabling efficient and privacy-preserving training of VLA models. Specifically, we introduce an Instruction-Oriented Scene-Parsing mechanism, which decomposes and enhances object-level features based on task instructions, improving contextual understanding. To effectively learn diverse task patterns, we design a Dual Gating Mixture-of-Experts (DGMoE) mechanism, where not only input tokens but also self-aware experts adaptively decide their activation. Finally, we propose an Expert-Driven Aggregation strategy at the federated server, where model aggregation is guided by activated experts, ensuring effective cross-client knowledge transfer. Extensive simulations and real-world robotic experiments demonstrate the effectiveness of our proposals. Notably, DGMoE significantly improves computational efficiency compared to its vanilla counterpart, while FedVLA achieves task success rates comparable to centralized training, effectively preserving data privacy.

</details>

---

## 243. Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting

- [ ] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting | https://openaccess.thecvf.com/content/ICCV2025/html/Miao_Towards_Scalable_Spatial_Intelligence_via_2D-to-3D_Data_Lifting_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Miao_Towards_Scalable_Spatial_Intelligence_via_2D-to-3D_Data_Lifting_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spatial intelligence is emerging as a transformative frontier in AI, yet it remains constrained by the scarcity of large-scale 3D datasets. Unlike the abundant 2D imagery, acquiring 3D data typically requires specialized sensors and laborious annotation. In this work, we present a scalable pipeline that converts single-view images into comprehensive, scale- and appearance-realistic 3D representations -- including point clouds, camera poses, depth maps, and pseudo-RGBD -- via integrated depth estimation, camera calibration, and scale calibration. Our method bridges the gap between the vast repository of imagery and the increasing demand for spatial scene understanding. By automatically generating authentic, scale-aware 3D data from images, we significantly reduce data collection costs and open new avenues for advancing spatial intelligence. We release two generated spatial datasets, i.e., COCO-3D and Objects365-v2-3D, and demonstrate through extensive experiments that our generated data can benefit various 3D tasks, ranging from fundamental perception to MLLM-based reasoning. These results validate our pipeline as an effective solution for developing AI systems capable of perceiving, understanding, and interacting with physical environments.

</details>

---

## 244. Vision-Language Interactive Relation Mining for Open-Vocabulary Scene Graph Generation

- [ ] Vision-Language Interactive Relation Mining for Open-Vocabulary Scene Graph Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Min_Vision-Language_Interactive_Relation_Mining_for_Open-Vocabulary_Scene_Graph_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Min_Vision-Language_Interactive_Relation_Mining_for_Open-Vocabulary_Scene_Graph_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

To promote the deployment of scenario understanding in the real world, Open-Vocabulary Scene Graph Generation (OV-SGG) has attracted much attention recently, aiming to generalize beyond the limited number of relation categories labeled during training and detect those unseen relations during inference. Towards OV-SGG, one feasible solution is to leverage the large-scale pre-trained vision-language models (VLMs) containing plentiful category-level content to capture accurate correspondences between images and text. However, due to the lack of quadratic relation-aware knowledge in VLMs, directly using the category-level correspondence in the base dataset could not sufficiently represent generalized relations involved in open world. Therefore, designing an effective open-vocabulary relation mining framework is challenging and meaningful. To this end, we propose a novel Vision-Language Interactive Relation Mining model (VL-IRM) for OV-SGG, which explores learning generalized relation-aware knowledge through multi-modal interaction. Specifically, first, to enhance the generalization of the relation text to visual content, we present a generative relation model to make the text modality explore possible open-ended relations based on visual content. Then, we employ visual modality to guide the relation text for spatial and semantic extension. Extensive experiments demonstrate the superior OV-SGG performance of our method.

</details>

---

## 245. Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features

- [ ] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features | https://openaccess.thecvf.com/content/ICCV2025/html/Mitra_Enhancing_Few-Shot_Vision-Language_Classification_with_Large_Multimodal_Model_Features_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Mitra_Enhancing_Few-Shot_Vision-Language_Classification_with_Large_Multimodal_Model_Features_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generative Large Multimodal Models (LMMs) like LLaVA and Qwen-VL excel at a wide variety of vision-language (VL) tasks. Despite strong performance, LMMs' generative outputs are not specialized for vision-language classification tasks (i.e., tasks with vision-language inputs and discrete labels) such as image classification and multiple-choice VQA. One key challenge in utilizing LMMs for these tasks is the extraction of useful features from generative LMMs. To overcome this, we propose an approach that leverages multimodal feature extraction from the LMM's latent space. Toward this end, we present Sparse Attention Vectors (SAVs)---a finetuning-free method that leverages sparse attention head activations (fewer than 5% of the heads) in LMMs as strong feature representations. With only few-shot examples, SAVs demonstrate state-of-the-art performance compared to a variety of few-shot and finetuned baselines on a collection of vision-language classification tasks. Our experiments also imply that SAVs can scale in performance with additional examples and generalize to similar tasks, establishing SAVs as both effective and robust multimodal feature representations.

</details>

---

## 246. Gaze-Language Alignment for Zero-Shot Prediction of Visual Search Targets from Human Gaze Scanpaths

- [ ] Gaze-Language Alignment for Zero-Shot Prediction of Visual Search Targets from Human Gaze Scanpaths | https://openaccess.thecvf.com/content/ICCV2025/html/Mondal_Gaze-Language_Alignment_for_Zero-Shot_Prediction_of_Visual_Search_Targets_from_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Mondal_Gaze-Language_Alignment_for_Zero-Shot_Prediction_of_Visual_Search_Targets_from_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Decoding human intent from eye gaze during a visual search task has become an increasingly important capability within augmented and virtual reality systems. However, gaze target prediction models used within such systems are constrained by the predefined target categories found within available gaze data, limiting their generalizability to novel categories and their usefulness within real-world, interactive systems. In this work, we present the Gaze-Language Alignment Model (GLAM), a vision-language model that can generalize gaze target predictions to novel categories of search targets lacking gaze annotation. To do so, GLAM uses a novel gaze encoder to encode foveal and peripheral information of a gaze scanpath. The resultant gaze embeddings are aligned with language embeddings of large language model-generated search descriptions for associated target categories using a novel contrastive learning strategy called Gaze-Language Alignment Decomposition (GLAD). When used to train GLAM in a zero-shot setup, GLAD surpassed naive contrastive learning strategies by nearly one-third in target prediction accuracy, even outperforming a fully supervised baseline. Moreover, in a fully supervised setup, GLAM outperformed previous methods in target prediction accuracy, regardless of the training strategy used.

</details>

---

## 247. PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection

- [ ] PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection | https://openaccess.thecvf.com/content/ICCV2025/html/Molahasani_PRISM_Reducing_Spurious_Implicit_Biases_in_Vision-Language_Models_with_LLM-Guided_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Molahasani_PRISM_Reducing_Spurious_Implicit_Biases_in_Vision-Language_Models_with_LLM-Guided_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Projection-based Reduction of Implicit Spurious bias in vision-language Models (PRISM), a new data-free and task-agnostic solution for bias mitigation in VLMs like CLIP. VLMs often inherit and amplify biases in their training data, leading to skewed predictions.PRISM is designed to debias VLMs without relying on predefined bias categories or additional external data. It operates in two stages: first, an LLM is prompted with simple class prompts to generate scene descriptions that contain spurious correlations. Next, PRISM uses our novel contrastive-style debiasing loss to learn a projection that maps the embeddings onto a latent space that minimizes spurious correlations while preserving the alignment between image and text embeddings. Extensive experiments demonstrate that PRISM outperforms current debiasing methods on the commonly used Waterbirds and CelebA datasets We make our code public at: https://github.com/MahdiyarMM/PRISM.

</details>

---

## 248. Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation

- [ ] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation | https://openaccess.thecvf.com/content/ICCV2025/html/Mrabah_Sparsity_Outperforms_Low-Rank_Projections_in_Few-Shot_Adaptation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Mrabah_Sparsity_Outperforms_Low-Rank_Projections_in_Few-Shot_Adaptation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adapting Vision-Language Models (VLMs) to new domains with few labeled samples remains a significant challenge due to severe overfitting and computational constraints. State-of-the-art solutions, such as low-rank reparameterization, mitigate these issues but often struggle with generalization and require extensive hyperparameter tuning. In this paper, a novel Sparse Optimization (SO) framework is proposed. Unlike low-rank approaches that typically constrain updates to a fixed subspace, our SO method leverages high sparsity to dynamically adjust very few parameters. We introduce two key paradigms. First, we advocate for local sparsity and global density, which updates a minimal subset of parameters per iteration while maintaining overall model expressiveness. As a second paradigm, we advocate for local randomness and global importance, which sparsifies the gradient using random selection while pruning the first moment based on importance. This combination significantly mitigates overfitting and ensures stable adaptation in low-data regimes. Extensive experiments on 11 diverse datasets show that SO achieves state-of-the-art few-shot adaptation performance while reducing memory overhead.

</details>

---

## 249. MINERVA: Evaluating Complex Video Reasoning

- [ ] MINERVA: Evaluating Complex Video Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Nagrani_MINERVA_Evaluating_Complex_Video_Reasoning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Nagrani_MINERVA_Evaluating_Complex_Video_Reasoning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal LLMs are turning their focus to video benchmarks, however most video benchmarks only provide outcome supervision, with no intermediate or interpretable reasoning steps. This makes it challenging to assess if models are truly able to combine perceptual and temporal information to reason about videos, or simply get the correct answer by chance or by exploiting linguistic biases. To remedy this, we provide a new video reasoning dataset called MINERVA for modern multimodal models. Each question in the dataset comes with 5 answer choices, as well as detailed, hand-crafted reasoning traces. Our dataset is multimodal, diverse in terms of video domain and length, and consists of complex multi-step questions. Extensive benchmarking shows that our dataset provides a challenge for frontier open-source and proprietary models. We perform fine-grained error analysis to identify common failure modes across various models, and create a taxonomy of reasoning errors. We use this to explore both human and LLM-as-a-judge methods for scoring video reasoning traces, and find that failure modes are primarily related to temporal localization, followed by visual perception errors, as opposed to logical or completeness errors. The dataset, along with questions, answer candidates and reasoning traces is publicly available under https://github.com/google-deepmind/neptune?tab=readme-ov-file#minerva  https://github.com/google-deepmind/neptune?tab=readme-ov-file\#minerva.

</details>

---

## 250. LV-MAE: Learning Long Video Representations through Masked-Embedding Autoencoders

- [ ] LV-MAE: Learning Long Video Representations through Masked-Embedding Autoencoders | https://openaccess.thecvf.com/content/ICCV2025/html/Naiman_LV-MAE_Learning_Long_Video_Representations_through_Masked-Embedding_Autoencoders_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Naiman_LV-MAE_Learning_Long_Video_Representations_through_Masked-Embedding_Autoencoders_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we introduce long-video masked-embedding autoencoders (LV-MAE), a self-supervised learning framework for long video representation.Our approach treats short- and long-span dependencies as two separate tasks.Such decoupling allows for a more intuitive video processing where short-span spatiotemporal primitives are first encoded and are then used to capture long-range dependencies across consecutive video segments. To achieve this, we leverage advanced off-the-shelf multimodal encoders to extract representations from short segments within the long video, followed by pre-training a masked-embedding autoencoder capturing high-level interactions across segments.LV-MAE is highly efficient to train and enables the processing of much longer videos by alleviating the constraint on the number of input frames.Furthermore, unlike existing methods that typically pre-train on short-video datasets, our approach offers self-supervised pre-training using long video samples (e.g., 20+ minutes video clips) at scale.Using LV-MAE representations, we achieve state-of-the-art results on three long-video benchmarks -- LVU, COIN, and Breakfast -- employing only a simple classification head for either attentive or linear probing.Finally, to assess LV-MAE pre-training and visualize its reconstruction quality, we leverage the video-language aligned space of short video representations to monitor LV-MAE through video-text retrieval.Our code will be made available upon publication.

</details>

---

## 251. SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion

- [ ] SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion | https://openaccess.thecvf.com/content/ICCV2025/html/Nassar_SmolDocling_An_ultra-compact_vision-language_model_for_end-to-end_multi-modal_document_conversion_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Nassar_SmolDocling_An_ultra-compact_vision-language_model_for_end-to-end_multi-modal_document_conversion_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce SmolDocling, an ultra-compact vision-language model targeting end-to-end document conversion. Our model comprehensively processes entire pages by generating DocTags, a new universal markup format that captures all page elements in their full context with location. Unlike existing approaches that rely on large foundational models, or ensemble solutions that rely on handcrafted pipelines of multiple specialized models, SmolDocling offers an end-to-end conversion for accurately capturing content, structure and spatial location of document elements in a 256M parameters vision-language model. SmolDocling exhibits robust performance in correctly reproducing document features such as code listings, tables, equations, charts, lists, and more across a diverse range of document types including business documents, academic papers, technical reports, patents, and forms -- significantly extending beyond the commonly observed focus on scientific papers. Additionally, we contribute novel publicly sourced datasets for charts, tables, equations, and code recognition.Experimental results demonstrate that SmolDocling competes with other Vision Language Models that are up to 27 times larger in size, while reducing computational requirements substantially. The model weights and datasets are available at: https://huggingface.co/ds4sd

</details>

---

## 252. Enhancing Spatial Reasoning in Multimodal Large Language Models through Reasoning-based Segmentation

- [ ] Enhancing Spatial Reasoning in Multimodal Large Language Models through Reasoning-based Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Ning_Enhancing_Spatial_Reasoning_in_Multimodal_Large_Language_Models_through_Reasoning-based_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ning_Enhancing_Spatial_Reasoning_in_Multimodal_Large_Language_Models_through_Reasoning-based_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in point cloud perception have demonstrated remarkable progress in scene understanding through vision-language alignment leveraging large language models (LLMs). However, existing methods may still encounter challenges in handling complex instructions that require accurate spatial reasoning, even if the 3D point cloud data provides detailed spatial cues such as size and position for identifying the targets. To tackle this issue, we propose Relevant Reasoning Segmentation (R^2S), a reasoning-based segmentation framework. The framework emulates human cognitive processes by decomposing spatial reasoning into two sequential stages: first identifying relevant elements, then processing instructions guided by their associated visual priors. Furthermore, acknowledging the inadequacy of existing datasets in complex reasoning tasks, we introduce 3D ReasonSeg, a reasoning-based segmentation dataset comprising 25,185 training samples and 3,966 validation samples with precise annotations. Both quantitative and qualitative experiments demonstrate that the R^2S and 3D ReasonSeg effectively endow 3D point cloud perception with stronger spatial reasoning capabilities, and we hope that they can serve as a new baseline and benchmark for future work.

</details>

---

## 253. The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models

- [ ] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Niss_The_Inter-Intra_Modal_Measure_A_Predictive_Lens_on_Fine-Tuning_Outcomes_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Niss_The_Inter-Intra_Modal_Measure_A_Predictive_Lens_on_Fine-Tuning_Outcomes_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The fine-tuning of large vision-language foundation models remains an underexplored area, particularly regarding its impact on learning gains and catastrophic forgetting. Inspired by the significance of modality gaps in contrastive dual-encoders, we introduce the Inter-Intra Modal Measure (IIMM)--a predictive metric that quantifies the relationship between intra-modal image embedding similarity and inter-modal misalignment. Through extensive empirical analysis across four state-of-the-art vision-language models and five fine-tuning techniques, we establish a strong linear relationship: tasks with higher IIMM scores yield greater in-domain performance improvements but suffer from more pronounced out-of-domain degradation, with some parameter-efficient fine-tuning (PEFT) methods exhibiting severe forgetting. Compared to existing transferability measures, the IIMM demonstrates significantly stronger predictive power for accuracy changes post fine-tuning in dual-encoder models. Moreover, we provide a theoretical bound, proving that changes in IIMM are limited by the Wasserstein distance between pre- and post-fine-tuning embedding distributions, ensuring its stability and robustness as a predictive measure. With only a single forward pass of the target data, practitioners can leverage this key insight to evaluate the degree to which a model can be expected to improve following fine-tuning. When combined with prior knowledge of a model's performance across diverse tasks, the IIMM further enhances transferability predictions for novel tasks, offering a lightweight yet effective tool for guiding model adaptation strategies. Our code is provided at https://github.com/mit-ll/IIMM.

</details>

---

## 254. Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling

- [ ] Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling | https://openaccess.thecvf.com/content/ICCV2025/html/Niu_Enhancing_Adversarial_Transferability_by_Balancing_Exploration_and_Exploitation_with_Gradient-Guided_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Niu_Enhancing_Adversarial_Transferability_by_Balancing_Exploration_and_Exploitation_with_Gradient-Guided_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adversarial attacks present a critical challenge to deep neural networks' robustness, particularly in transfer scenarios across different model architectures. However, the transferability of adversarial attacks faces a fundamental dilemma between Exploitation (maximizing attack potency) and Exploration (enhancing cross-model generalization). Traditional momentum-based methods over-prioritize Exploitation, i.e., higher loss maxima for attack potency but weakened generalization (narrow loss surface). Conversely, recent methods with inner-iteration sampling over-prioritize Exploration, i.e., flatter loss surfaces for cross-model generalization but weakened attack potency (suboptimal local maxima). To resolve this dilemma, we propose a simple yet effective Gradient-Guided Sampling (GGS), which harmonizes both objectives through guiding sampling along the gradient ascent direction to improve both sampling efficiency and stability. Specifically, based on MI-FGSM, GGS introduces inner-iteration random sampling and guides the sampling direction using the gradient from the previous inner-iteration (the sampling's magnitude is determined by a random distribution). This mechanism encourages adversarial examples to reside in balanced regions with both flatness for cross-model generalization and higher local maxima for strong attack potency. Comprehensive experiments across multiple DNN architectures and multimodal large language models (MLLMs) demonstrate the superiority of our method over state-of-the-art transfer attacks. Code is made available at https://github.com/anuin-cat/GGS.

</details>

---

## 255. ChatReID: Open-ended Interactive Person Retrieval via Hierarchical Progressive Tuning for Vision Language Models

- [ ] ChatReID: Open-ended Interactive Person Retrieval via Hierarchical Progressive Tuning for Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Niu_ChatReID_Open-ended_Interactive_Person_Retrieval_via_Hierarchical_Progressive_Tuning_for_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Niu_ChatReID_Open-ended_Interactive_Person_Retrieval_via_Hierarchical_Progressive_Tuning_for_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Person re-identification (Re-ID) is a crucial task in computer vision, aiming to recognize individuals across non-overlapping camera views. While recent advanced vision-language models (VLMs) excel in logical reasoning and multi-task generalization, their applications in Re-ID tasks remain limited. They either struggle to perform accurate matching based on identity-relevant features or assist image-dominated branches as auxiliary semantics. In this paper, we propose a novel framework ChatReID, that shifts the focus towards a text-side-dominated retrieval paradigm, enabling flexible and interactive re-identification. To integrate the reasoning abilities of language models into Re-ID pipelines, We first present a large-scale instruction dataset, which contains more than 8 million prompts to promote the model fine-tuning. Next. we introduce a hierarchical progressive tuning strategy, which endows Re-ID ability through three stages of tuning, i.e., from person attribute understanding to fine-grained image retrieval and to multi-modal task reasoning.Extensive experiments across ten popular benchmarks demonstrate that ChatReID outperforms existing methods, achieving state-of-the-art performance in all Re-ID tasks. More experiments demonstrate that ChatReID not only has the ability to recognize fine-grained details but also to integrate them into a coherent reasoning process.

</details>

---

## 256. Region-aware Anchoring Mechanism for Efficient Referring Visual Grounding

- [ ] Region-aware Anchoring Mechanism for Efficient Referring Visual Grounding | https://openaccess.thecvf.com/content/ICCV2025/html/Ouyang_Region-aware_Anchoring_Mechanism_for_Efficient_Referring_Visual_Grounding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ouyang_Region-aware_Anchoring_Mechanism_for_Efficient_Referring_Visual_Grounding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring Visual Grounding (RVG) tasks revolve around utilizing vision-language interactions to incorporate object information from language expressions, thereby enabling targeted object detection or segmentation within images. Transformer-based methods have enabled effective interaction through attention mechanisms, achieving notable performance in RVG tasks. However, existing strategies for RVG, which involve direct interaction between visual and linguistic features, face three key challenges: (i) tendency to focus on a single target, (ii) insufficient control over linguistic noise, and (iii) high computational cost. To address these challenges, we propose a Region-aware Anchoring Mechanism (RaAM) that mediates vision-language interactions. In RaAM, region-aware anchors engage in alternating interactions with vision and language modalities, acting as indicators for object presence across different regions within the image. RaAM (i) directs attention to multiple target regions for better localization, (ii) reduces cross-modal redundancy by using anchors as buffers, and (iii) lowers time complexity. In addition, we design region and pixel level loss functions to enhance object presence assessment and edge precision. We evaluate our RaAM-RVG on four benchmark datasets and integrate RaAM into various models by replacing their interaction design. Results show that RaAM outperforms state-of-the-art methods with lower computational cost.

</details>

---

## 257. ICE-Bench: A Unified and Comprehensive Benchmark for Image Creating and Editing

- [ ] ICE-Bench: A Unified and Comprehensive Benchmark for Image Creating and Editing | https://openaccess.thecvf.com/content/ICCV2025/html/Pan_ICE-Bench_A_Unified_and_Comprehensive_Benchmark_for_Image_Creating_and_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Pan_ICE-Bench_A_Unified_and_Comprehensive_Benchmark_for_Image_Creating_and_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image generation has witnessed significant advancements in the past few years. However, evaluating the performance of image generation models remains a formidable challenge. In this paper, we propose ICE-Bench, a unified and comprehensive benchmark designed to rigorously assess image generation models. Its comprehensiveness could be summarized in the following key features: (1) Coarse-to-Fine Tasks: We systematically deconstruct image generation into four task categories: No-ref/Ref Image Creating/Editing, based on the presence or absence of source images and reference images. And further decompose them into 31 fine-grained tasks covering a broad spectrum of image generation requirements, culminating in a comprehensive benchmark. (2) Multi-dimensional Metrics: The evaluation framework assesses image generation capabilities across 6 dimensions: aesthetic quality, imaging quality, prompt following, source consistency, reference consistency, and controllability. 11 metrics are introduced to support the multi-dimensional evaluation. Notably, we introduce VLLM-QA, an innovative metric designed to assess the success of image editing by leveraging large models. (3) Hybrid Data: The data comes from real scenes and virtual generation, which effectively improves data diversity and alleviates the bias problem in model evaluation. Through ICE-Bench, we conduct a thorough analysis of existing generation models, revealing both the challenging nature of our benchmark and the gap between current model capabilities and real-world generation requirements. To foster further advancements in the field, we will open-source ICE-Bench, including its dataset, evaluation code, and models, thereby providing a valuable resource for the research community.

</details>

---

## 258. Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP

- [ ] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP | https://openaccess.thecvf.com/content/ICCV2025/html/Park_Know_No_Better_A_Data-Driven_Approach_for_Enhancing_Negation_Awareness_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Park_Know_No_Better_A_Data-Driven_Approach_for_Enhancing_Negation_Awareness_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While CLIP has significantly advanced multimodal understanding by bridging vision and language, the inability to grasp negation -- such as failing to differentiate concepts like "parking" from "no parking" -- poses substantial challenges.By analyzing the data used in the public CLIP model's pre-training, we posit this limitation stems from a lack of negation-inclusive data.To address this, we introduce data generation pipelines that employ a large language model (LLM) and a multimodal LLM to produce negation-inclusive captions.Fine-tuning CLIP with data generated from our pipelines, we develop NegationCLIP, which enhances negation awareness while preserving the generality.Moreover, to enable a comprehensive evaluation of negation understanding, we propose NegRefCOCOg--a benchmark tailored to test VLMs' ability to interpret negation across diverse expressions and positions within a sentence.Experiments on various CLIP architectures validate the effectiveness of our data generation pipelines in enhancing CLIP's ability to perceive negation accurately.Additionally, NegationCLIP's enhanced negation awareness has practical applications across various multimodal tasks, demonstrated by performance gains in text-to-image generation and referring image segmentation.

</details>

---

## 259. NuPlanQA: A Large-Scale Dataset and Benchmark for Multi-View Driving Scene Understanding in Multi-Modal Large Language Models

- [ ] NuPlanQA: A Large-Scale Dataset and Benchmark for Multi-View Driving Scene Understanding in Multi-Modal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Park_NuPlanQA_A_Large-Scale_Dataset_and_Benchmark_for_Multi-View_Driving_Scene_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Park_NuPlanQA_A_Large-Scale_Dataset_and_Benchmark_for_Multi-View_Driving_Scene_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in multi-modal large language models (MLLMs) have demonstrated strong performance across various domains; however, their ability to comprehend driving scenes remains less proven. The complexity of driving scenarios, which includes multi-view information, poses significant challenges for existing MLLMs. In this paper, we introduce NuPlanQA-Eval, a multi-view, multi-modal evaluation benchmark for driving scene understanding. To further support generalization to multi-view driving scenarios, we also propose NuPlanQA-1M, a large-scale dataset comprising 1M real-world visual question-answering (VQA) pairs. For context-aware analysis of traffic scenes, we categorize our dataset into nine subtasks across three core skills: Road Environment Perception, Spatial Relations Recognition, and Ego-Centric Reasoning. Furthermore, we present BEV-LLM, integrating Bird's-Eye-View (BEV) features from multi-view images into MLLMs. Our evaluation results reveal key challenges that existing MLLMs face in driving scene-specific perception and spatial reasoning from ego-centric perspectives. In contrast, BEV-LLM demonstrates remarkable adaptability to this domain, outperforming other models in six of the nine subtasks. These findings highlight how BEV integration enhances multi-view MLLMs while also identifying key areas that require further refinement for effective adaptation to driving scenes. NuPlanQA is available at https://github.com/sungyeonparkk/NuPlanQA.

</details>

---

## 260. Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control

- [ ] Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control | https://openaccess.thecvf.com/content/ICCV2025/html/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Deep neural network (DNN)-based policy models, such as vision-language-action (VLA) models, excel at automating complex decision-making from multi-modal inputs. However, scaling these models greatly increases computational overhead, complicating deployment in resource-constrained settings like robot manipulation and autonomous driving. To address this, we propose Saliency-Aware Quantized Imitation Learning (\method), which combines quantization-aware training with a selective loss-weighting strategy for mission-critical states. By identifying these states via saliency scores and emphasizing them in the training loss, \method preserves decision fidelity under low-bit precision. We validate \method's generalization capability across extensive simulation benchmarks with environment variations, real-world tasks, and cross-domain tasks (self-driving, physics simulation), consistently recovering full-precision performance. Notably, a 4-bit weight-quantized VLA model for robotic manipulation achieves up to 2.5xspeedup and 2.5xenergy savings on an edge GPU with minimal accuracy loss. These results underline \method's potential for efficiently deploying large IL-based policy models on resource-limited devices.

</details>

---

## 261. Mitigating Object Hallucinations via Sentence-Level Early Intervention

- [ ] Mitigating Object Hallucinations via Sentence-Level Early Intervention | https://openaccess.thecvf.com/content/ICCV2025/html/Peng_Mitigating_Object_Hallucinations_via_Sentence-Level_Early_Intervention_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Peng_Mitigating_Object_Hallucinations_via_Sentence-Level_Early_Intervention_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have revolutionized cross-modal understanding but continue to struggle with hallucinations - fabricated content contradicting visual inputs. Existing hallucination mitigation methods either incur prohibitive computational costs or introduce distribution mismatches between training data and model outputs. We identify a critical insight: hallucinations predominantly emerge at the early stages of text generation and propagate through subsequent outputs. To address this, we propose SENTINEL (Sentence-level Early iNtervention Through IN-domain prEference Learning), a framework that eliminates dependency on human annotations. Specifically, we first bootstrap high-quality in-domain preference pairs by iteratively sampling model outputs, validating object existence through cross-checking with two open-vocabulary detectors, and classifying sentences into hallucinated/non-hallucinated categories. Subsequently, we use context-coherent positive samples and hallucinated negative samples to iteratively build context-aware preference data. Finally, we train models using a context-aware preference loss (C-DPO) that emphasizes discriminative learning at the sentence level where hallucinations initially manifest. Experimental results show that SENTINEL can reduce hallucinations by 90% over the original model and outperforms the previous state-of-the-art method on both the hallucination benchmarks and general capabilities benchmarks, manifesting its superiority and generalization ability. The models, datasets, and code are available at https://github.com/pspdada/SENTINEL.

</details>

---

## 262. ROVI: A VLM-LLM Re-Captioned Dataset for Open-Vocabulary Instance-Grounded Text-to-Image Generation

- [ ] ROVI: A VLM-LLM Re-Captioned Dataset for Open-Vocabulary Instance-Grounded Text-to-Image Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Peng_ROVI_A_VLM-LLM_Re-Captioned_Dataset_for_Open-Vocabulary_Instance-Grounded_Text-to-Image_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Peng_ROVI_A_VLM-LLM_Re-Captioned_Dataset_for_Open-Vocabulary_Instance-Grounded_Text-to-Image_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present ROVI, a high-quality synthetic dataset for instance-grounded text-to-image generation, created by labeling 1M curated web images. Our key innovation is a strategy called re-captioning, focusing on the pre-detection stage, where a VLM (Vision-Language Model) generates comprehensive visual descriptions that are then processed by an LLM (Large Language Model) to extract a flat list of potential categories for OVDs (Open-Vocabulary Detectors) to detect. This approach yields a global prompt inherently linked to instance annotations while capturing secondary visual elements humans typically overlook. Evaluations show that ROVI exceeds existing detection datasets in image quality and resolution while containing two orders of magnitude more categories with an open-vocabulary nature. For demonstrative purposes, a text-to-image model GLIGEN trained on ROVI significantly outperforms state-of-the-art alternatives in instance grounding accuracy, prompt fidelity, and aesthetic quality. Our dataset and reproducible pipeline are available at https://github.com/CihangPeng/ROVI.

</details>

---

## 263. MissRAG: Addressing the Missing Modality Challenge in Multimodal Large Language Models

- [ ] MissRAG: Addressing the Missing Modality Challenge in Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Pipoli_MissRAG_Addressing_the_Missing_Modality_Challenge_in_Multimodal_Large_Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Pipoli_MissRAG_Addressing_the_Missing_Modality_Challenge_in_Multimodal_Large_Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Multimodal Large Language Models (MLLMs) have emerged as a leading framework for enhancing the ability of Large Language Models (LLMs) to interpret non-linguistic modalities. Despite their impressive capabilities, the robustness of MLLMs under conditions where one or more modalities are missing remains largely unexplored. In this paper, we investigate the extent to which MLLMs can maintain performance when faced with missing modality inputs. Moreover, we propose a novel framework to mitigate the aforementioned issue called retrieval-augmented generation for missing modalities (MissRAG). It consists of a novel multimodal RAG technique alongside a tailored prompt engineering strategy designed to enhance model robustness by mitigating the impact of absent modalities while preventing the burden of additional instruction tuning. To demonstrate the effectiveness of our techniques, we conduct comprehensive evaluations across five diverse datasets, covering tasks such as audio-visual question answering, audio-visual captioning, and multimodal sentiment analysis. Our source code is available at https://github.com/aimagelab/MissRAG.

</details>

---

## 264. CAPTURE: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting

- [ ] CAPTURE: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting | https://openaccess.thecvf.com/content/ICCV2025/html/Pothiraj_CAPTURE_Evaluating_Spatial_Reasoning_in_Vision_Language_Models_via_Occluded_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Pothiraj_CAPTURE_Evaluating_Spatial_Reasoning_in_Vision_Language_Models_via_Occluded_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recognizing and reasoning about occluded (partially or fully hidden) objects is vital to understanding visual scenes, as occlusions frequently occur in real-world environments and act as obstacles for spatial comprehension. To test models' ability to reason about multiple occluded objects, we introduce a novel task, Counting Amodally for Patterns Through Unseen REgions (CAPTURe), which requires a model to count objects arranged in a pattern by inferring how the pattern continues behind an occluder (an object which blocks parts of the scene). CAPTURe requires both recognizing visual patterns and reasoning, making it a useful testbed for evaluating vision-language models (VLMs) on whether they understand occluded patterns and possess spatial understanding skills. By requiring models to reason about occluded objects, CAPTURe also tests VLMs' ability to form world models that would allow them to fill in missing information. CAPTURe consists of two parts: (1) CAPTURe-real, with manually filtered images of real objects in patterns and (2) CAPTURe-synthetic, a controlled diagnostic with generated patterned images. We evaluate four strong VLMs (GPT-4o, Intern-VL2, Molmo, and Qwen2-VL) on CAPTURe, finding that models struggle to count on both occluded and unoccluded patterns. Crucially, we find that models perform worse with occlusion, suggesting that VLMs are also deficient in inferring unseen spatial relationships: even the strongest VLMs like GPT-4o fail to count with occlusion. In contrast, we find that humans achieve very little error on CAPTURe. We also find that providing auxiliary information of occluded object locations increases performance, underscoring that the model error comes both from an inability to handle occlusion as well as difficulty in counting in images.

</details>

---

## 265. NeuralSVG: An Implicit Representation for Text-to-Vector Generation

- [ ] NeuralSVG: An Implicit Representation for Text-to-Vector Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Polaczek_NeuralSVG_An_Implicit_Representation_for_Text-to-Vector_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Polaczek_NeuralSVG_An_Implicit_Representation_for_Text-to-Vector_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vector graphics are essential in design, providing artists with a versatile medium for creating resolution-independent and highly editable visual content. Recent advancements in vision-language and diffusion models have fueled interest in text-to-vector graphics generation. However, existing approaches often suffer from over-parameterized outputs or treat the layered structure -- a core feature of vector graphics -- as a secondary goal, diminishing their practical use. Recognizing the importance of layered SVG representations, we propose NeuralSVG, an implicit neural representation for generating vector graphics from text prompts. Inspired by Neural Radiance Fields (NeRFs), NeuralSVG encodes the entire scene into the weights of a small MLP network, optimized using Score Distillation Sampling (SDS). To encourage a layered structure in the generated SVG, we introduce a dropout-based regularization technique that strengthens the standalone meaning of each shape. We additionally demonstrate that utilizing a neural representation provides an added benefit of inference-time control, enabling users to dynamically adapt the generated SVG based on user-provided inputs, all with a single learned representation. Through extensive qualitative and quantitative evaluations, we demonstrate that NeuralSVG outperforms existing methods in generating structured and flexible SVG.

</details>

---

## 266. Enrich and Detect: Video Temporal Grounding with Multimodal LLMs

- [ ] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Pramanick_Enrich_and_Detect_Video_Temporal_Grounding_with_Multimodal_LLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Pramanick_Enrich_and_Detect_Video_Temporal_Grounding_with_Multimodal_LLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce ED-VTG, a method for fine-grained video temporal grounding utilizing multi-modal large language models. Our approach harnesses the capabilities of multimodal LLMs to jointly process text and video, in order to effectively localize natural language queries in videos through a two-stage process. Rather than being directly grounded, language queries are initially transformed into enriched sentences that incorporate missing details and cues to aid in grounding. In the second stage, these enriched queries are grounded, using a lightweight decoder, which specializes at predicting accurate boundaries conditioned on contextualized representations of the enriched queries. To mitigate noise and reduce the impact of hallucinations, our model is trained with a multiple-instance-learning objective that dynamically selects the optimal version of the query for each training sample. We demonstrate state-of-the-art results across various benchmarks in temporal video grounding and paragraph grounding settings. Experiments reveal that our method significantly outperforms all previously proposed LLM-based temporal grounding approaches and is either superior or comparable to specialized models, while maintaining a clear advantage against them in zero-shot evaluation scenarios.

</details>

---

## 267. Trust but Verify: Programmatic VLM Evaluation in the Wild

- [ ] Trust but Verify: Programmatic VLM Evaluation in the Wild | https://openaccess.thecvf.com/content/ICCV2025/html/Prabhu_Trust_but_Verify_Programmatic_VLM_Evaluation_in_the_Wild_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Prabhu_Trust_but_Verify_Programmatic_VLM_Evaluation_in_the_Wild_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) frequently hallucinate responses to visual queries, undermining their reliability for critical applications. However, quantifying the effect of such hallucinations in free-form responses to open-ended queries requires visually verifying each claim within the response, which is highly challenging. We propose Programmatic VLM Evaluation (PROVE), a new benchmarking paradigm for evaluating VLM responses to open-ended queries. To construct PROVE, we provide a large language model with a high-fidelity scene-graph representation constructed from a detailed image caption, and prompt it to generate i) diverse and challenging question-answer (QA) pairs that test a range of image understanding capabilities, and ii) programs that can be executed over the scene graph object to verify each QA pair. We thus construct a benchmark of 10.6k challenging but grounded visual QA pairs. Next, we propose a scene graph-based evaluation framework to programmatically measure both the helpfulness and truthfulness of a free-form model response without relying on subjective LLM judgments. We extensively benchmark a range of VLMs on PROVE, and uncover a concerning tradeoff where models that provide more helpful responses often hallucinate more, whereas truthful models tend to be less informative. PROVE serves as a foundation for developing next-generation VLMs that balance helpfulness with truthfulness. A snapshot of our dataset is available at https://prove-explorer-anon.netlify.app/.

</details>

---

## 268. Lumina-Image 2.0: A Unified and Efficient Image Generative Framework

- [ ] Lumina-Image 2.0: A Unified and Efficient Image Generative Framework | https://openaccess.thecvf.com/content/ICCV2025/html/Qin_Lumina-Image_2.0_A_Unified_and_Efficient_Image_Generative_Framework_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qin_Lumina-Image_2.0_A_Unified_and_Efficient_Image_Generative_Framework_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Lumina-Image 2.0, an advanced text-to-image (T2I) model that surpasses previous state-of-the-art methods across multiple benchmarks. Lumina-Image 2.0 is characterized by two key features: (1) Unification - it adopts a unified architecture (Unified Next-DiT) that treats text and image tokens as a joint sequence, enabling natural cross-modal interactions and allowing seamless task expansion. Besides, since high-quality captioners can provide semantically well-aligned text-image training pairs, we introduce a unified captioning system, Unified Captioner (UniCap), which can generate detailed and accurate multilingual captions for our model. This not only accelerates model convergence, but also enhances prompt adherence, multi-granularity prompt handling, and task expansion with customized prompt templates. (2)Efficiency - to improve the efficiency of our proposed model, we develop multi-stage progressive training strategies to optimize our model, alongside inference-time acceleration strategies without compromising image quality. We evaluate our model on academic benchmarks and T2I arenas, with results confirming that it matches or exceeds existing state-of-the-art models across various metrics, highlighting the effectiveness of our methods. We have released our training details, code, and models at https://github.com/Alpha-VLLM/Lumina-Image-2.0.

</details>

---

## 269. Benchmarking Multimodal Large Language Models Against Image Corruptions

- [ ] Benchmarking Multimodal Large Language Models Against Image Corruptions | https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Benchmarking_Multimodal_Large_Language_Models_Against_Image_Corruptions_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Benchmarking_Multimodal_Large_Language_Models_Against_Image_Corruptions_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have made significant strides in visual and language tasks. However, despite their impressive performance on standard datasets, these models encounter considerable robustness challenges when processing corrupted images, raising concerns about their reliability in safety-critical applications. To address this issue, we introduce the MLLM-IC benchmark, specifically designed to assess the performance of MLLMs under image corruption scenarios. MLLM-IC offers a more comprehensive evaluation of corruption robustness compared to existing benchmarks, enabling a multi-dimensional assessment of various MLLM capabilities across a broad range of corruption types. It includes 40 distinct corruption types and 34 low-level multimodal capabilities, each organized into a three-level hierarchical structure. Notably, it is the first corruption robustness benchmark designed to facilitate the evaluation of fine-grained MLLM capabilities. We further evaluate several prominent MLLMs and derive valuable insights into their characteristics. We believe the MLLM-IC benchmark will provide crucial insights into the robustness of MLLMs in handling corrupted images and contribute to the development of more resilient MLLMs.

</details>

---

## 270. Web Artifact Attacks Disrupt Vision Language Models

- [ ] Web Artifact Attacks Disrupt Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Qraitem_Web_Artifact_Attacks_Disrupt_Vision_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qraitem_Web_Artifact_Attacks_Disrupt_Vision_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) (e.g., CLIP, LLaVA) are trained on large-scale, lightly curated web datasets, leading them to learn unintended correlations between semantic concepts and unrelated visual signals. These associations degrade model accuracy by causing predictions to rely on incidental patterns rather than genuine visual understanding. Prior work has weaponized these correlations as an attack vector to manipulate model predictions, such as inserting a deceiving class text onto the image in a "typographic" attack. These attacks succeed due to VLMs' text-heavy bias--a result of captions that echo visible words rather than describing content. However, this attack has focused solely on text that matches the target class exactly, overlooking a broader range of correlations, including non-matching text and graphical symbols, which arise from the abundance of branding content in web-scale data. To address this gap, we introduce "artifact-based" attacks: a novel class of manipulations that mislead models using both non-matching text and graphical elements. Unlike typographic attacks, these artifacts are not predefined, making them simultaneously harder to defend against and more challenging to find. We address this by framing artifact attacks as a search problem and demonstrate their effectiveness across five datasets, with some artifacts reinforcing each other to reach 100% attack success rates. These attacks transfer across models with up to 90% effectiveness, making it possible to attack unseen models. To defend against these attacks, we extend prior work's artifact aware prompting to the graphical setting. We see a moderate reduction of success rates of up to 15% relative to standard prompts, suggesting a promising direction for enhancing model robustness. Code: https://github.com/mqraitem/Web-Artifact-Attacks

</details>

---

## 271. DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup

- [ ] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup | https://openaccess.thecvf.com/content/ICCV2025/html/Qu_DictAS_A_Framework_for_Class-Generalizable_Few-Shot_Anomaly_Segmentation_via_Dictionary_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qu_DictAS_A_Framework_for_Class-Generalizable_Few-Shot_Anomaly_Segmentation_via_Dictionary_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent vision-language models (e.g., CLIP) have demonstrated remarkable class-generalizable ability to unseen classes in few-shot anomaly segmentation (FSAS), leveraging supervised prompt learning or fine-tuning on seen classes. However, their cross-category generalization largely depends on prior knowledge of real seen anomaly samples. In this paper, we propose a novel framework, namely DictAS, which enables a unified model to detect visual anomalies in unseen object categories without any retraining on the target data, only employing a few normal reference images as visual prompts. The insight behind DictAS is to transfer dictionary lookup capabilities to the FSAS task for unseen classes via self-supervised learning, instead of merely memorizing the normal and abnormal feature patterns from the training set. Specifically, DictAS mainly consists of three components: (1) **Dictionary Construction** - to simulate the index and content of a real dictionary using features from normal reference images. (2) **Dictionary Lookup** - to retrieve queried region features from the dictionary via a sparse lookup strategy. When a query feature cannot be retrieved, it is classified as an anomaly. (3) **Query Discrimination Regularization**- to enhance anomaly discrimination by making abnormal features harder to retrieve from the dictionary. To achieve this, Contrastive Query Constraint and Text Alignment Constraint are further proposed. Extensive experiments on seven public industrial and medical datasets demonstrate that DictAS consistently outperforms state-of-the-art FSAS methods.

</details>

---

## 272. Spatial Preference Rewarding for MLLMs Spatial Understanding

- [ ] Spatial Preference Rewarding for MLLMs Spatial Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Spatial_Preference_Rewarding_for_MLLMs_Spatial_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Spatial_Preference_Rewarding_for_MLLMs_Spatial_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models(MLLMs) have demonstrated promising spatial understanding capabilities, such as referencing and grounding object descriptions. Despite their successes, MLLMs still fall short in fine-grained spatial perception abilities, such as generating detailed region descriptions or accurately localizing objects. Additionally, they often fail to respond to the user's requirements for desired fine-grained spatial understanding. This issue might arise because existing approaches primarily focus on tuning MLLMs to model pre-annotated instruction data to inject spatial knowledge, without direct supervision of MLLMs' actual responses. We address this issue by SPR, a Spatial Preference Rewarding(SPR) approach that enhances MLLMs' spatial capabilities by rewarding MLLMs' detailed responses with precise object localization over vague or inaccurate responses. With randomly selected image regions and region descriptions from MLLMs, SPR introduces semantic and localization scores to comprehensively evaluate the text quality and localization quality in MLLM-generated descriptions. We also refine the MLLM descriptions with better localization accuracy and pair the best-scored refinement with the initial descriptions of the lowest score for direct preference optimization, thereby enhancing fine-grained alignment with visual input. Extensive experiments over standard referring and grounding benchmarks show that SPR improves MLLM spatial understanding capabilities effectively with minimal overhead in training. Data and code will be released at https://github.com/hanqiu-hq/SPR

</details>

---

## 273. Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma?

- [ ] Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma? | https://openaccess.thecvf.com/content/ICCV2025/html/Qu_Does_Your_Vision-Language_Model_Get_Lost_in_the_Long_Video_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qu_Does_Your_Vision-Language_Model_Get_Lost_in_the_Long_Video_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rise of Large Vision-Language Models (LVLMs) has significantly advanced video understanding. However, efficiently processing long videos remains a challenge due to the "Sampling Dilemma": low-density sampling risks missing critical information, while high-density sampling introduces redundancy. To address this issue, we introduce LSDBench, the first benchmark designed to evaluate LVLMs on long-video tasks by constructing high Necessary Sampling Density (NSD) questions--where NSD represents the minimum sampling density required to accurately answer a given question. LSDBench focuses on dense, short-duration actions to rigorously assess the sampling strategies employed by LVLMs. To tackle the challenges posed by high-NSD questions, we propose a novel Reasoning-Driven Hierarchical Sampling (RHS) framework, which combines global localization of question-relevant cues with local dense sampling for precise inference. Additionally, we develop a lightweight Semantic-Guided Frame Selector to prioritize informative frames, enabling RHS to achieve comparable or superior performance with significantly fewer sampled frames. Together, our LSDBench and RHS framework address the unique challenges of high-NSD long-video tasks, setting a new standard for evaluating and improving LVLMs in this domain.

</details>

---

## 274. Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions

- [ ] Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions | https://openaccess.thecvf.com/content/ICCV2025/html/Qu_Hate_in_Plain_Sight_On_the_Risks_of_Moderating_AI-Generated_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qu_Hate_in_Plain_Sight_On_the_Risks_of_Moderating_AI-Generated_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in text-to-image diffusion models have enabled the creation of a new form of digital art: optical illusions---visual tricks that create different perceptions of reality. However, adversaries may misuse such techniques to generate hateful illusions, which embed specific hate messages into harmless scenes and disseminate them across web communities. In this work, we take the first step toward investigating the risks of scalable hateful illusion generation and the potential for bypassing current content moderation models. Specifically, we generate 1,860 optical illusions using Stable Diffusion and ControlNet, conditioned on 62 hate messages. Of these, 1,571 are hateful illusions that successfully embed hate messages, either overtly or subtly, forming the Hateful Illusion dataset. Using this dataset, we evaluate the performance of six moderation classifiers and nine vision language models (VLMs) in identifying hateful illusions. Experimental results reveal significant vulnerabilities in existing moderation models: the detection accuracy falls below 0.245 for moderation classifiers and below 0.102 for VLMs. We further identify a critical limitation in their vision encoders, which mainly focus on surface-level image details while overlooking the secondary layer of information, i.e., hidden messages. To address this risk, we explore preliminary mitigation measures and identify the most effective approaches from the perspectives of image transformations and training-level strategies.

</details>

---

## 275. IGD: Instructional Graphic Design with Multimodal Layer Generation

- [ ] IGD: Instructional Graphic Design with Multimodal Layer Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Qu_IGD_Instructional_Graphic_Design_with_Multimodal_Layer_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qu_IGD_Instructional_Graphic_Design_with_Multimodal_Layer_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphic design visually conveys information and data by creating and combining text, images and graphics. Two-stage methods that rely primarily on layout generation lack creativity and intelligence, making graphic design still labor-intensive. Existing diffusion-based methods generate non-editable graphic design files at image level with poor legibility in visual text rendering, which prevents them from achieving satisfactory and practical automated graphic design. In this paper, we propose Instructional Graphic Designer (IGD) to swiftly generate multimodal layers with editable flexibility with only natural language instructions. IGD adopts a new paradigm that leverages parametric rendering and image asset generation. First, we develop a design platform and establish a standardized format for multi-scenario design files, thus laying the foundation for scaling up data. Second, IGD utilizes the multimodal understanding and reasoning capabilities of MLLM to accomplish attribute prediction, sequencing and layout of layers. It also employs a diffusion model to generate image content for assets. By enabling end-to-end training, IGD architecturally supports scalability and extensibility in complex graphic design tasks. The superior experimental results demonstrate that IGD offers a new solution for graphic design.

</details>

---

## 276. ReCoT: Reflective Self-Correction Training for Mitigating Confirmation Bias in Large Vision-Language Models

- [ ] ReCoT: Reflective Self-Correction Training for Mitigating Confirmation Bias in Large Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Qu_ReCoT_Reflective_Self-Correction_Training_for_Mitigating_Confirmation_Bias_in_Large_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Qu_ReCoT_Reflective_Self-Correction_Training_for_Mitigating_Confirmation_Bias_in_Large_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision-Language Models (LVLMs) have greatly improved their ability to understand both visual and text information. However, a common problem in LVLMs is confirmation bias, where models tend to repeat previous assumptions and follow earlier viewpoints instead of reflecting and correcting themselves. This problem is more common in smaller-scale LVLMs, as they are usually fine-tuned with training data that is mostly positive, focusing on generating coherent dialogue. To address this issue, we introduce ReCoT, a method designed to mitigate confirmation bias in smaller-scale LVLMs through Reflective Self-Correction Training.The method follows a two-stage SFT-DPO paradigm: the first SFT stage aims to cultivate the model's reflective correction abilities, while the DPO stage focuses on enhancing the consistency between answers and reflections. Specifically, we construct dialogue-based reflective samples, which serve as adversarial samples during SFT. In this process, the model is initially presented with a potentially incorrect answer, followed by a reflection and correction phase to generate the final answer. To enhance answer-reflection consistency, we propose the consistency direct preference optimization. To comprehensively evaluate the effectiveness of our ReCoT, we introduce a set of novel metrics to measure the accuracy of the reflection and correction process. Extensive experiments show that ReCoT enables LVLM to engage in robust self-reflection and error correction and reduce confirmation bias.

</details>

---

## 277. TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models

- [ ] TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Rahmanzadehgervi_TAB_Transformer_Attention_Bottlenecks_enable_User_Intervention_and_Debugging_in_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Rahmanzadehgervi_TAB_Transformer_Attention_Bottlenecks_enable_User_Intervention_and_Debugging_in_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-head self-attention (MHSA) is a key component of Transformers, a widely popular architecture in both language and vision. Multiple heads intuitively enable different parallel processes over the same input. Yet, they also obscure the attribution of each input patch to the output of a model. We propose a novel 1-head Transformer Attention Bottleneck (TAB) layer, inserted after the traditional MHSA architecture, to serve as an attention bottleneck for interpretability and intervention. Unlike standard self-attention, TAB constrains the total attention over all patches to \in [0, 1]. That is, when the total attention is 0, no visual information is propagated further into the network, and the vision-language model (VLM) would default to a generic, image-independent response. To demonstrate the advantages of TAB, we train VLMs with TAB to perform image-difference captioning. Over three datasets, our models perform similarly to baseline VLMs in captioning but the bottleneck is superior in localizing changes and in identifying when no changes occur. TAB is the first architecture to enable users to debug by editing attention, which often produces expected outputs by VLMs.

</details>

---

## 278. CuRe: Cultural Gaps in the Long Tail of Text-to-Image Systems

- [ ] CuRe: Cultural Gaps in the Long Tail of Text-to-Image Systems | https://openaccess.thecvf.com/content/ICCV2025/html/Rege_CuRe_Cultural_Gaps_in_the_Long_Tail_of_Text-to-Image_Systems_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Rege_CuRe_Cultural_Gaps_in_the_Long_Tail_of_Text-to-Image_Systems_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Popular text-to-image (T2I) systems are trained on web-scraped data, which is heavily Amero and Euro-centric, underrepresenting the cultures of the Global South. To analyze these biases, we introduce CuRe, a novel and scalable benchmarking and scoring suite for cultural representativeness that leverages the marginal utility of attribute specification to T2I systems as a proxy for human judgments. Our CuRe benchmark dataset has a novel categorical hierarchy built from the crowdsourced Wikimedia knowledge graph, with 300 cultural artifacts across 32 cultural subcategories grouped into six broad cultural axes (food, art, fashion, architecture, celebrations, and people). Our dataset's categorical hierarchy enables CuRe scorers to evaluate T2I systems by analyzing their response to increasing the informativeness of text conditioning, enabling fine-grained cultural comparisons. We empirically observe much stronger correlations of our class of scorers to human judgments of perceptual similarity, image-text alignment, and cultural diversity across image encoders (SigLIP 2, AIMV2 and DINOv2), multimodal language models (OpenCLIP, SigLIP 2, Gemini 2.0 Flash) and state-of-the-art text-to-image systems, including three variants of Stable Diffusion (1.5, XL, 3.5 Large), FLUX.1 [dev], Ideogram 2.0, and DALL-E 3. The code and dataset is open-sourced and available at https://aniketrege.github.io/cure/.

</details>

---

## 279. Multi-modal Segment Anything Model for Camouflaged Scene Segmentation

- [ ] Multi-modal Segment Anything Model for Camouflaged Scene Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Ren_Multi-modal_Segment_Anything_Model_for_Camouflaged_Scene_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ren_Multi-modal_Segment_Anything_Model_for_Camouflaged_Scene_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Camouflaged scenes, where objects blend seamlessly into their environments, pose significant challenges to both human observers and computer vision systems. These objects match the background in color, texture, and shape, making them difficult to detect. To this end, we propose leveraging the Segment Anything Model (SAM) to tackle this challenging task effectively. Specifically, we propose how to exploit SAM without requiring any manual prompts by proposing several ideas. At the core of our method lies the rich information extracted through multi-modal prompts. At first, we generate an image caption using the BLIP model and obtain its text embedding through the use of a text encoder. We then generate a visual embedding through the vision encoder of the BLIP model and use both as inputs to SAM to provide additional semantic information about the image. Finally, we propose a couple of architectural novelties, a) we effectively integrate the multi-modal information in SAM through a multi-level adapter and b) we replace the dense embedding of SAM with the image embedding of its image encoder. Our method achieves new state-of-the-art performance in 11 out of 12 metrics in three benchmark datasets for camouflaged detection. Additionally, our method can be successfully adapted to other tasks such as medical image segmentation performing on par or even outperforming the state-of-the-art methods. Our code is available in https://github.com/ic-qialanqian/Vision-Language-SAM.

</details>

---

## 280. AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?

- [ ] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations? | https://openaccess.thecvf.com/content/ICCV2025/html/Ruan_AdvDreamer_Unveils_Are_Vision-Language_Models_Truly_Ready_for_Real-World_3D_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ruan_AdvDreamer_Unveils_Are_Vision-Language_Models_Truly_Ready_for_Real-World_3D_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have exhibited remarkable generalization capabilities, yet their robustness in dynamic real-world scenarios remains largely unexplored. To systematically evaluate VLMs' robustness to real-world 3D variations, we propose AdvDreamer, the first framework capable of generating physically reproducible Adversarial 3D Transformation (Adv-3DT) samples from single-view observations. In AdvDreamer, we integrate three key innovations: Firstly, to characterize real-world 3D variations with limited prior knowledge precisely, we design a zero-shot Monocular Pose Manipulation pipeline built upon generative 3D priors. Secondly, to ensure the visual quality of worst-case Adv-3DT samples, we propose Naturalness Reward Model that provides continuous naturalness regularization during adversarial optimization, effectively preventing convergence to hallucinated or unnatural elements. Thirdly, to enable systematic evaluation across diverse VLM architectures and visual-language tasks, we introduce the Inverse Semantic Probability loss as the adversarial optimization objective, which solely operates in the fundamental visual-textual alignment space. Based on the captured Adv-3DT samples with high aggressiveness and transferability, we establish MM3DTBench, the first VQA benchmark dataset tailored to evaluate VLM robustness under challenging 3D variations. Extensive evaluations of representative VLMs with varying architectures reveal that real-world 3D variations can pose severe threats to model performance across various tasks.

</details>

---

## 281. VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models

- [ ] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models | https://openaccess.thecvf.com/content/ICCV2025/html/Ruan_VLRMBench_A_Comprehensive_and_Challenging_Benchmark_for_Vision-Language_Reward_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ruan_VLRMBench_A_Comprehensive_and_Challenging_Benchmark_for_Vision-Language_Reward_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although large visual-language models (LVLMs) have demonstrated strong performance in multimodal tasks, errors may occasionally arise due to biases during the reasoning process. Recently, reward models (RMs) have become increasingly pivotal in the reasoning process. Specifically, process RMs evaluate each reasoning step, outcome RMs focus on the assessment of reasoning results, and critique RMs perform error analysis on the entire reasoning process, followed by corrections. However, existing benchmarks for vision-language RMs (VLRMs) typically assess only a single aspect of their capabilities (e.g., distinguishing between two answers), thus limiting the all-round evaluation and restricting the development of RMs in the visual-language domain. To address this gap, we propose a comprehensive and challenging benchmark, dubbed as VLRMBench, encompassing 12,634 questions. VLRMBench is constructed based on three distinct types of datasets, covering mathematical reasoning, hallucination understanding, and multi-image understanding. We design 12 tasks across three major categories, focusing on evaluating VLRMs in the aspects of process understanding, outcome judgment, and critique generation. Extensive experiments are conducted on 21 open-source models and 5 advanced closed-source models, highlighting the challenges posed by VLRMBench. For instance, in the `Forecasting Future', a binary classification task, the advanced GPT-4o achieves only a 76.0% accuracy. The code is available at https://github.com/JCruan519/VLRMBench.

</details>

---

## 282. From Panels to Prose: Generating Literary Narratives from Comics

- [ ] From Panels to Prose: Generating Literary Narratives from Comics | https://openaccess.thecvf.com/content/ICCV2025/html/Sachdeva_From_Panels_to_Prose_Generating_Literary_Narratives_from_Comics_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sachdeva_From_Panels_to_Prose_Generating_Literary_Narratives_from_Comics_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Comics have long been a popular form of storytelling, offering visually engaging narratives that captivate audiences worldwide. However, the visual nature of comics presents a significant barrier for visually impaired readers, limiting their access to these engaging stories. In this work, we provide a pragmatic solution to this accessibility challenge by developing an automated system that generates text-based literary narratives from manga comics. Our approach aims to create an evocative and immersive prose that not only conveys the original narrative but also captures the depth and complexity of characters, their interactions, and the vivid settings in which they reside.To this end we make the following contributions: (1) We present a unified model, Magiv3, that excels at various functional tasks pertaining to comic understanding, such as localising panels, characters, texts, and speech-bubble tails, performing OCR, grounding characters etc. (2) We release human-annotated captions for over 3300 Japanese comic panels, along with character grounding annotations, and benchmark large vision-language models in their ability to understand comic images. (3) Finally, we demonstrate how integrating large vision-language models with Magiv3, can generate seamless literary narratives that allows visually impaired audiences to engage with the depth and richness of comic storytelling. Our code, trained model and dataset annotations are publicly available.

</details>

---

## 283. Generate, Transduct, Adapt: Iterative Transduction with VLMs

- [ ] Generate, Transduct, Adapt: Iterative Transduction with VLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Saha_Generate_Transduct_Adapt_Iterative_Transduction_with_VLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Saha_Generate_Transduct_Adapt_Iterative_Transduction_with_VLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Transductive zero-shot learning with vision-language models leverages image-image similarities within the dataset to achieve better classification accuracy compared to the inductive setting. However, there is little work that explores the structure of the language space in this context. We propose GTA-CLIP, a novel technique that incorporates supervision from language models for joint transduction in language and vision spaces. Our approach is iterative and consists of three steps: (i) incrementally exploring the attribute space by querying language models, (ii) an attribute-augmented transductive inference procedure, and (iii) fine-tuning the language and vision encoders based on inferred labels within the dataset. Through experiments with CLIP encoders, we demonstrate that GTA-CLIP, yields an average performance improvement of 9.5% and 4.0% across 12 datasets and 3 encoders, over CLIP and transductive CLIP respectively in the zero-shot setting. We also observe similar improvements in a few-shot setting. We present ablation studies that demonstrate the value of each step and visualize how the vision and language spaces evolve over iterations driven by the transductive learning.

</details>

---

## 284. CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning

- [ ] CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning | https://openaccess.thecvf.com/content/ICCV2025/html/Saito_CaptionSmiths_Flexibly_Controlling_Language_Pattern_in_Image_Captioning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Saito_CaptionSmiths_Flexibly_Controlling_Language_Pattern_in_Image_Captioning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

An image captioning model flexibly switching its language pattern, e.g., descriptiveness and length, should be useful since it can be applied to diverse applications. However, despite the dramatic improvement in generative vision-language models, fine-grained control over the properties of generated captions is not easy due to two reasons: (i) existing models are not given the properties as a condition during training and (ii) existing models cannot smoothly transition its language pattern from one state to the other. Given this challenge, we propose a new approach, CaptionSmiths, to acquire a single captioning model that can handle diverse language patterns. First, our approach quantifies three properties of each caption, length, descriptiveness, and uniqueness of a word, as continuous scalar values, without human annotation. Given the values, we represent the conditioning via interpolation between two endpoint vectors corresponding to the extreme states, e.g., one for a very short caption and one for a very long caption. Empirical results demonstrate that the resulting model can smoothly change the properties of the output captions and show higher lexical alignment than baselines. For instance, CaptionSmiths reduces the error in controlling caption length by 506% despite better lexical alignment. Code will be available on https://github.com/omronsinicx/captionsmiths.

</details>

---

## 285. MAVias: Mitigate any Visual Bias

- [ ] MAVias: Mitigate any Visual Bias | https://openaccess.thecvf.com/content/ICCV2025/html/Sarridis_MAVias_Mitigate_any_Visual_Bias_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sarridis_MAVias_Mitigate_any_Visual_Bias_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mitigating biases in computer vision models is an essential step towards trustworthy artificial intelligence systems. Existing bias mitigation methods are limited to predefined biases, preventing their use in visual datasets where multiple, possibly unknown biases exist. To address this limitation, we introduce MAVias, an open-set bias mitigation approach that leverages foundation models to discover spurious associations between visual attributes and target classes. MAVias first captures a wide variety of visual features in natural language via a foundation image tagging model, and then leverages a large language model to select visual features that define the target class, resulting in a set of language-coded potential visual biases. It then translates these biases into vision-language embeddings and introduces an in-processing bias mitigation approach to prevent the model from encoding information related to them. Experiments on diverse datasets, including CelebA, Waterbirds, ImageNet, and UrbanCars, show that MAVias effectively detects and mitigates a wide range of biases in visual recognition tasks, outperforming current state-of-the-art.

</details>

---

## 286. Global and Local Entailment Learning for Natural World Imagery

- [ ] Global and Local Entailment Learning for Natural World Imagery | https://openaccess.thecvf.com/content/ICCV2025/html/Sastry_Global_and_Local_Entailment_Learning_for_Natural_World_Imagery_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sastry_Global_and_Local_Entailment_Learning_for_Natural_World_Imagery_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning the hierarchical structure of data in vision-language models is a significant challenge. Previous works have attempted to address this challenge by employing entailment learning. However, these approaches fail to model the transitive nature of entailment explicitly, which establishes the relationship between order and semantics within a representation space. In this work, we introduce Radial Cross-Modal Embeddings (RCME), a framework that enables the explicit modeling of transitivity-enforced entailment. Our proposed framework optimizes for the partial order of concepts within vision-language models. By leveraging our framework, we develop a hierarchical vision-language foundation model capable of representing the hierarchy in the Tree of Life. Our experiments on hierarchical species classification and hierarchical retrieval tasks demonstrate the enhanced performance of our models compared to the existing state-of-the-art models. Our code and models are open-sourced at https://vishu26.github.io/RCME.

</details>

---

## 287. Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning

- [ ] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Shan_Geminio_Language-Guided_Gradient_Inversion_Attacks_in_Federated_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shan_Geminio_Language-Guided_Gradient_Inversion_Attacks_in_Federated_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Foundation models that bridge vision and language have made significant progress. While they have inspired many life-enriching applications, their potential for abuse in creating new threats remains largely unexplored. In this paper, we reveal that vision-language models (VLMs) can be weaponized to enhance gradient inversion attacks (GIAs) in federated learning (FL), where an FL server attempts to reconstruct private data samples from gradients shared by victim clients. Despite recent advances, existing GIAs struggle to reconstruct high-resolution images when the victim has a large local data batch. One promising direction is to focus reconstruction on valuable samples rather than the entire batch, but current methods lack the flexibility to target specific data of interest. To address this gap, we propose Geminio, the first approach to transform GIAs into semantically meaningful, targeted attacks. It enables a brand new privacy attack experience: attackers can describe, in natural language, the data they consider valuable, and Geminio will prioritize reconstruction to focus on those high-value samples. This is achieved by leveraging a pretrained VLM to guide the optimization of a malicious global model that, when shared with and optimized by a victim, retains only gradients of samples that match the attacker-specified query. Geminio can be launched at any FL round and has no impact on normal training (i.e., the FL server can steal clients' data while still producing a high-utility ML model as in benign scenarios). Extensive experiments demonstrate its effectiveness in pinpointing and reconstructing targeted samples, with high success rates across complex datasets and large batch sizes with resilience against defenses.

</details>

---

## 288. Growing a Twig to Accelerate Large Vision-Language Models

- [ ] Growing a Twig to Accelerate Large Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Shao_Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shao_Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) have demonstrated remarkable capabilities in open-world multimodal understanding, yet their high computational overheads pose great challenges for practical deployment. Some recent works have proposed methods to accelerate VLMs by pruning redundant visual tokens guided by the attention maps of VLM's early layers. Despite the success of these token pruning methods, they still suffer from two major shortcomings: (i) considerable accuracy drop due to insensitive attention signals in early layers, and (ii) limited speedup when generating long responses (e.g., 30 tokens). To address the limitations above, we present TwigVLM---a simple and general architecture by "growing" a lightweight twig upon an early layer of the base VLM. Compared with most existing VLM acceleration methods purely based on visual token pruning, our TwigVLM not only achieves better accuracy retention by employing a twig-guided token pruning (TTP) strategy, but also yields higher generation speed by utilizing a self-speculative decoding (SSD) strategy. Taking LLaVA-1.5-7B as the base VLM, experimental results show that TwigVLM preserves 96% of the original performance after pruning 88.9% of visual tokens and achieves 154% speedup in generating long responses, delivering significantly better performance in terms of both accuracy and speed over the state-of-the-art VLM acceleration methods.

</details>

---

## 289. Multi-Modal Multi-Task Unified Embedding Model (M3T-UEM): A Task-Adaptive Representation Learning Framework

- [ ] Multi-Modal Multi-Task Unified Embedding Model (M3T-UEM): A Task-Adaptive Representation Learning Framework | https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_Multi-Modal_Multi-Task_Unified_Embedding_Model_M3T-UEM_A_Task-Adaptive_Representation_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_Multi-Modal_Multi-Task_Unified_Embedding_Model_M3T-UEM_A_Task-Adaptive_Representation_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present Multi-Modal Multi-Task Unified Embedding Model (M3T-UEM), a framework that advances vision-language matching and retrieval by leveraging a large language model (LLM) backbone. While concurrent LLM-based approaches like VLM2VEC, MM-Embed, NV-Embed, and MM-GEM have demonstrated impressive capabilities in multi-modal and multi-task scenarios, our work introduces novel mechanisms for task-adaptive learning and embedding extraction that further enhance the potential of LLM-based retrieval systems. Our key technical contribution lies in the development of a task-aware contrastive learning framework with an automated Bayesian weighing mechanism. This approach provides a principled way to balance multiple tasks during training, departing from conventional contrastive learning strategies. We further enhance the framework through a multiple-token summarization strategy and an auxiliary language modeling objective, which together significantly improve retrieval performance.Comprehensive experiments on M-BEIR and ICinW benchmarks demonstrate M3T-UEM's effectiveness, showing competitive or superior performance compared to both traditional encoder-based methods and recent LLM-based approaches. Furthermore, we demonstrate particular strengths in handling compositional conceptual changes and multilingual scenarios owing to the incorporation of an LLM backbone where the method drastically outperforms CLIP in zero-shot settings, often by orders of magnitude.

</details>

---

## 290. AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs

- [ ] AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Shen_AutoComPose_Automatic_Generation_of_Pose_Transition_Descriptions_for_Composed_Pose_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shen_AutoComPose_Automatic_Generation_of_Pose_Transition_Descriptions_for_Composed_Pose_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Composed pose retrieval (CPR) enables users to search for human poses by specifying a reference pose and a transition description, but progress in this field is hindered by the scarcity and inconsistency of annotated pose transitions. Existing CPR datasets rely on costly human annotations or heuristic-based rule generation, both of which limit scalability and diversity. In this work, we introduce AutoComPose, the first framework that leverages multimodal large language models (MLLMs) to automatically generate rich and structured pose transition descriptions. Our method enhances annotation quality by structuring transitions into fine-grained body part movements and introducing swapped/mirrored variations, while a cyclic consistency constraint ensures logical coherence between forward and reverse transitions. To advance CPR research, we construct and release two dedicated benchmarks, AIST-CPR and PoseFixCPR, supplementing prior datasets with enhanced attributes. Extensive experiments demonstrate that training retrieval models with AutoComPose yields superior performance over human-annotated and heuristic-based methods, significantly reducing annotation costs while improving retrieval quality. Our work pioneers the automatic annotation of pose transitions, establishing a scalable foundation for future CPR research.

</details>

---

## 291. Online Reasoning Video Segmentation with Just-in-Time Digital Twins

- [ ] Online Reasoning Video Segmentation with Just-in-Time Digital Twins | https://openaccess.thecvf.com/content/ICCV2025/html/Shen_Online_Reasoning_Video_Segmentation_with_Just-in-Time_Digital_Twins_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shen_Online_Reasoning_Video_Segmentation_with_Just-in-Time_Digital_Twins_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reasoning segmentation (RS) aims to identify and segment objects of interest based on implicit text queries. As such, RS is a catalyst for embodied AI agents, enabling them to interpret high-level commands without requiring explicit step-by-step guidance. However, current RS approaches rely heavily on the visual perception capabilities of multimodal large language models (LLMs), leading to several major limitations. First, they struggle with queries that require multiple steps of reasoning or those that involve complex spatial/temporal relationships. Second, they necessitate LLM fine-tuning, which may require frequent updates to maintain compatibility with contemporary LLMs and may increase risks of catastrophic forgetting during fine-tuning. Finally, being primarily designed for static images or offline video processing, they scale poorly to online video data. To address these limitations, we propose an agent framework that disentangles perception and reasoning for online video RS without LLM fine-tuning. Our innovation is the introduction of a just-in-time digital twin concept, where -- given an implicit query -- an LLM plans the construction of a low-level scene representation from high-level video using specialist vision models. We refer to this approach to creating a digital twin as "just-in-time" because the LLM planner will anticipate the need for specific information and only request this limited subset instead of always evaluating every specialist model. The LLM then performs reasoning on this digital twin representation to identify target objects. To evaluate our approach, we introduce a new comprehensive video reasoning segmentation benchmark comprising 200 videos with 895 implicit text queries. The benchmark spans three reasoning categories (semantic, spatial, and temporal) with three different reasoning chain complexity. Experimental results demonstrate that our method performs best across all reasoning categories, suggesting that our just-in-time digital twin can bridge the gap between high-level reasoning and low-level perception in embodied AI. Benchmark is available at https://github.com/yiqings/jitbench/.

</details>

---

## 292. LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation

- [ ] LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Shenaj_LoRA.rar_Learning_to_Merge_LoRAs_via_Hypernetworks_for_Subject-Style_Conditioned_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shenaj_LoRA.rar_Learning_to_Merge_LoRAs_via_Hypernetworks_for_Subject-Style_Conditioned_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in image generation models have enabled personalized image creation with both user-defined subjects (content) and styles. Prior works achieved personalization by merging corresponding low-rank adapters (LoRAs) through optimization-based methods, which are computationally demanding and unsuitable for real-time use on resource-constrained devices like smartphones. To address this, we introduce LoRA.rar, a method that not only improves image quality but also achieves a remarkable speedup of over 4000x in the merging process. We collect a dataset of style and subject LoRAs and pre-train a hypernetwork on a diverse set of content-style LoRA pairs, learning an efficient merging strategy that generalizes to new, unseen content-style pairs, enabling fast, high-quality personalization. Moreover, we identify limitations in existing evaluation metrics for content-style quality and propose a new protocol using multimodal large language models (MLLMs) for more accurate assessment. Our method significantly outperforms the current state of the art in both content and style fidelity, as validated by MLLM assessments and human evaluations.

</details>

---

## 293. Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation

- [ ] Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Shi_Harnessing_Vision_Foundation_Models_for_High-Performance_Training-Free_Open_Vocabulary_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shi_Harnessing_Vision_Foundation_Models_for_High-Performance_Training-Free_Open_Vocabulary_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While CLIP has advanced open-vocabulary predictions, its performance on semantic segmentation remains suboptimal. This shortfall primarily stems from its spatial-invariant semantic features and constrained resolution. While previous adaptations addressed spatial invariance semantic by modifying the self-attention in CLIP's image encoder, the issue of limited resolution remains unexplored. Different from previous segment-then-splice methods that segment sub-images via a sliding window and splice the results, we introduce a splice-then-segment paradigm that incorporates Segment-Anything Model (SAM) to tackle the resolution issue since SAM excels at extracting fine-grained semantic correlations from high-resolution images. Specifically, we introduce Trident, a training-free framework that first splices features extracted by CLIP and DINO from sub-images, then leverages SAM's encoder to create a correlation matrix for global aggregation, enabling a broadened receptive field. Besides, we propose a refinement strategy for CLIP's coarse segmentation outputs by transforming them into prompts for SAM. Trident achieves a significant improvement in the mIoU across eight popular benchmarks compared with the current SOTA. Furthermore, it can also be utilized to generate visual prompts that enhance the performance of Large Vision-Language Models (LVLMs). Code is available at https://github.com/YuHengsss/Trident.

</details>

---

## 294. OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving

- [ ] OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving | https://openaccess.thecvf.com/content/ICCV2025/html/Shimomura_OD-RASE_Ontology-Driven_Risk_Assessment_and_Safety_Enhancement_for_Autonomous_Driving_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shimomura_OD-RASE_Ontology-Driven_Risk_Assessment_and_Safety_Enhancement_for_Autonomous_Driving_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although autonomous driving systems demonstrate high perception performance, they still face limitations when handling rare situations or complex road structures. Since existing road infrastructures are designed for human drivers, safety improvements are typically introduced only after accidents occur. This reactive approach poses a significant challenge for autonomous systems, which require proactive risk mitigation. To address this issue, we propose OD-RASE, a framework for enhancing the safety of autonomous driving systems by detecting road structures that cause traffic accidents and connecting these findings to infrastructure development. First, we formalize an ontology based on specialized domain knowledge of road traffic systems. In parallel, we generate infrastructure improvement proposals using a large-scale visual language model (LVLM) and use ontology-driven data filtering to enhance their reliability. This process automatically annotates improvement proposals on pre-accident road images, leading to the construction of a new dataset. Furthermore, we introduce the Baseline approach (OD-RASE model), which leverages LVLM and a diffusion model to produce both infrastructure improvement proposals and generated images of the improved road environment. Our experiments demonstrate that ontology-driven data filtering enables highly accurate prediction of accident-causing road structures and the corresponding improvement plans. We believe that this work contributes to the overall safety of traffic environments and marks an important step toward the broader adoption of autonomous driving systems.

</details>

---

## 295. AgroBench: Vision-Language Model Benchmark in Agriculture

- [ ] AgroBench: Vision-Language Model Benchmark in Agriculture | https://openaccess.thecvf.com/content/ICCV2025/html/Shinoda_AgroBench_Vision-Language_Model_Benchmark_in_Agriculture_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Shinoda_AgroBench_Vision-Language_Model_Benchmark_in_Agriculture_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Precise automated understanding of agricultural tasks such as disease identification is essential for the sustainable crop production. Recent advances in vision-language models (VLMs) are expected to further expand the range of agricultural tasks by facilitating human-model interaction through easy, text-based communication. Here, we introduce AgroBench (Agronomist AI Benchmark), a benchmark for evaluating VLM models across seven agricultural topics, covering key areas in agricultural engineering and relevant to real-world farming. Unlike recent agricultural VLM benchmarks, AgroBench is annotated by expert agronomists. Our AgroBench covers a state-of-the-art range of categories, including 203 crop categories and 682 disease categories, to thoroughly evaluate VLM capabilities. In our evaluation on AgroBench, we reveal that VLMs have room for improvement in fine-grained identification tasks. Notably, in weed identification, most open-source VLMs perform close to random. With our wide range of topics and expert-annotated categories, we analyze the types of errors made by VLMs and suggest potential pathways for future VLM development. Our dataset and code are available at https://dahlian00.github.io/AgroBenchPage/.

</details>

---

## 296. FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models

- [ ] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Singha_FedMVP_Federated_Multimodal_Visual_Prompt_Tuning_for_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Singha_FedMVP_Federated_Multimodal_Visual_Prompt_Tuning_for_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In federated learning, textual prompt tuning adapts Vision-Language Models (e.g., CLIP) by tuning lightweight input tokens (or prompts) on local client data, while keeping network weights frozen. After training, only the prompts are shared by the clients with the central server for aggregation. However, textual prompt tuning suffers from overfitting to known concepts, limiting its generalizability to unseen concepts. To address this limitation, we propose Multimodal Visual Prompt Tuning (FedMVP) that conditions the prompts on multimodal contextual information - derived from the input image and textual attribute features of a class. At the core of FedMVP is a PromptFormer module that synergistically aligns textual and visual features through a cross-attention mechanism. The dynamically generated multimodal visual prompts are then input to the frozen vision encoder of CLIP, and trained with a combination of CLIP similarity loss and a consistency loss. Extensive evaluation on 20 datasets, spanning three generalization settings, demonstrates that FedMVP not only preserves performance on in-distribution classes and domains, but also displays higher generalizability to unseen classes and domains, surpassing state-of-the-art methods by a notable margin of +1.57% - 2.26%. Code is available at https://github.com/mainaksingha01/FedMVP.

</details>

---

## 297. Calibrating MLLM-as-a-judge via Multimodal Bayesian Prompt Ensembles

- [ ] Calibrating MLLM-as-a-judge via Multimodal Bayesian Prompt Ensembles | https://openaccess.thecvf.com/content/ICCV2025/html/Slyman_Calibrating_MLLM-as-a-judge_via_Multimodal_Bayesian_Prompt_Ensembles_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Slyman_Calibrating_MLLM-as-a-judge_via_Multimodal_Bayesian_Prompt_Ensembles_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are increasingly used to evaluate text-to-image (TTI) generation systems, providing automated judgments based on visual and textual context. However, these "judge" models often suffer from biases, overconfidence, and inconsistent performance across diverse image domains. While prompt ensembling has shown promise for mitigating these issues in unimodal, text-only settings, our experiments reveal that standard ensembling methods fail to generalize effectively for TTI tasks. To address these limitations, we propose a new multimodal-aware method called **M**ultimodal **M**ixture-of-**B**ayesian Prompt Ensembles (MMB). Our approach uses a Bayesian prompt ensemble approach augmented by image clustering, allowing the judge to dynamically assign prompt weights based on the visual characteristics of each sample. We show that MMB improves accuracy in pairwise preference judgments and greatly enhances calibration, making it easier to gauge the judge's true uncertainty. In evaluations on two TTI benchmarks, HPSv2 and MJBench, MMB outperforms existing baselines in alignment with human annotations and calibration across varied image content. Our findings highlight the importance of multimodal-specific strategies for judge calibration and suggest a promising path forward for reliable large-scale TTI evaluation.

</details>

---

## 298. Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images

- [ ] Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images | https://openaccess.thecvf.com/content/ICCV2025/html/Song_Normal_and_Abnormal_Pathology_Knowledge-Augmented_Vision-Language_Model_for_Anomaly_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Song_Normal_and_Abnormal_Pathology_Knowledge-Augmented_Vision-Language_Model_for_Anomaly_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Anomaly detection in computational pathology aims to identify rare and scarce anomalies where disease-related data are often limited or missing. Existing anomaly detection methods, primarily designed for industrial settings, face limitations in pathology due to computational constraints, diverse tissue structures, and lack of interpretability. To address these challenges, we propose Ano-NAViLa, a Normal and Abnormal pathology knowledge-augmented Vision-Language model for Anomaly detection in pathology images. Ano-NAViLa is built on a pre-trained vision-language model with a lightweight trainable MLP. By incorporating both normal and abnormal pathology knowledge, Ano-NAViLa enhances accuracy and robustness to variability in pathology images and provides interpretability through image-text associations. Evaluated on two lymph node datasets from different organs, Ano-NAViLa achieves the state-of-the-art performance in anomaly detection and localization, outperforming competing models.

</details>

---

## 299. Riemannian-Geometric Fingerprints of Generative Models

- [ ] Riemannian-Geometric Fingerprints of Generative Models | https://openaccess.thecvf.com/content/ICCV2025/html/Song_Riemannian-Geometric_Fingerprints_of_Generative_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Song_Riemannian-Geometric_Fingerprints_of_Generative_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent breakthroughs and rapid integration of generative models (GMs) have sparked interest in the problem of model attribution and their fingerprints. For instance, service providers need reliable methods of authenticating their models to protect their IP, while users and law enforcement seek to verify the source of generated content for accountability and trust. In addition, a growing threat of model collapse is arising, as more model-generated data are being fed back into sources (e.g., YouTube) that are often harvested for training ("regurgitative training"), heightening the need to differentiate synthetic from human data. Yet, a gap still exists in understanding generative models' fingerprints, we believe, stemming from the lack of a formal framework that can define, represent, and analyze the fingerprints in a principled way. To address this gap, we take a geometric approach and propose a new definition of artifact and fingerprint of generative models using Riemannian geometry, which allows us to leverage the rich theory of differential geometry. Our new definition generalizes previous work (Song et al, 2024) to non-Euclidean manifolds by learning Riemannian metrics from data and replacing the Euclidean distances and nearest-neighbor search with geodesic distances and kNN-based Riemannian center of mass. We apply our theory to a new gradient-based algorithm for computing the fingerprints in practice. Results show that it is more effective in distinguishing a large array of generative models, spanning across 4 different datasets in 2 different resolutions (64x64, 256x256), 27 model architectures, and 2 modalities (Vision, Vision-Language). Using our proposed definition can significantly improve the performance on model attribution, as well as a generalization to unseen datasets, model types, and modalities, suggesting its efficacy in practice.

</details>

---

## 300. Weakly-Supervised Learning of Dense Functional Correspondences

- [ ] Weakly-Supervised Learning of Dense Functional Correspondences | https://openaccess.thecvf.com/content/ICCV2025/html/Stojanov_Weakly-Supervised_Learning_of_Dense_Functional_Correspondences_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Stojanov_Weakly-Supervised_Learning_of_Dense_Functional_Correspondences_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Establishing dense correspondences across image pairs is essential for tasks such as shape reconstruction and robot manipulation. In the challenging setting of matching across different categories, the function of an object, i.e., the effect that an object can cause on other objects, can guide how correspondences should be established. This is because object parts that enable specific functions often share similarities in shape and appearance. We derive the definition of dense functional correspondence based on this observation and propose a weakly-supervised learning paradigm to tackle the prediction task. The main insight behind our approach is that we can leverage vision-language models to pseudo-label multi-view images to obtain functional parts. We then integrate this with dense contrastive learning from pixel correspondences to distill both functional and spatial knowledge into a new model that can establish dense functional correspondence. Further, we curate synthetic and real evaluation datasets as task benchmarks. Our results demonstrate the advantages of our approach over baseline solutions consisting of off-the-shelf self-supervised image representations and grounded vision language models.

</details>

---

## 301. ART: Adaptive Relation Tuning for Generalized Relation Prediction

- [ ] ART: Adaptive Relation Tuning for Generalized Relation Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Sudhakaran_ART_Adaptive_Relation_Tuning_for_Generalized_Relation_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sudhakaran_ART_Adaptive_Relation_Tuning_for_Generalized_Relation_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual relation detection (VRD) is the task of identifying the relationships between objects in a scene. VRD models trained solely on relation detection data struggle to generalize beyond the relations on which they are trained. While prompt tuning has been used to adapt vision-language models (VLMs) for VRD, it uses handcrafted prompts and struggles with novel or complex relations. We argue that instruction tuning offers a more effective solution by fine-tuning VLMs on diverse instructional data. We thus introduce ART, an Adaptive Relation Tuning framework that adapts VLMs for VRD through instruction tuning and strategic instance selection. By converting VRD datasets into an instruction-tuning format and employing an adaptive sampling algorithm, ART directs the VLM to focus on informative relations while maintaining generalizability. Specifically, we focus on the relation classification, where subject-object boxes are given and the model predicts the predicate between them. We tune on a held-in set and evaluate across multiple held-out datasets of varying complexity. Our approach strongly improves over its baselines and can infer unseen relation concepts, a capability absent in mainstream VRD methods. We demonstrate ART's practical value by using the predicted relations for segmenting complex scenes.

</details>

---

## 302. CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieval

- [ ] CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieval | https://openaccess.thecvf.com/content/ICCV2025/html/Sun_CoTMR_Chain-of-Thought_Multi-Scale_Reasoning_for_Training-Free_Zero-Shot_Composed_Image_Retrieval_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_CoTMR_Chain-of-Thought_Multi-Scale_Reasoning_for_Training-Free_Zero-Shot_Composed_Image_Retrieval_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images by integrating information from a composed query (reference image and modification text) without training samples. Existing methods primarily combine caption models and Large Language Models (LLMs) to generate target captions from composed queries but face various issues such as incompatibility, visual information loss, and insufficient reasoning. In this work, we propose CoTMR, a training-free framework with novel Chain-of-thought (CoT) and Multi-scale Reasoning. Instead of relying on caption models for modality transformation, CoTMR directly employs the Large Vision-Language Model (LVLM) to achieve unified understanding and reasoning of composed queries. To enhance reasoning reliability, we devise CIRCoT, which guides the LVLM to perform step-by-step reasoning by following predefined subtasks. Additionally, while most existing approaches focus solely on global-level reasoning, CoTMR introduces fine-grained predictions about the presence or absence of key elements at the object scale for more comprehensive reasoning. Furthermore, we design a Multi-Grained Scoring (MGS) mechanism, which integrates CLIP similarity scores of the above reasoning outputs with candidate images to realize precise retrieval. Extensive experiments demonstrate that our CoTMR not only drastically outperforms previous methods across four prominent benchmarks but also offers appealing interpretability.

</details>

---

## 303. Multimodal Large Language Model-Guided ISP Hyperparameter Optimization with Dynamic Preference Learning

- [ ] Multimodal Large Language Model-Guided ISP Hyperparameter Optimization with Dynamic Preference Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Multimodal_Large_Language_Model-Guided_ISP_Hyperparameter_Optimization_with_Dynamic_Preference_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Multimodal_Large_Language_Model-Guided_ISP_Hyperparameter_Optimization_with_Dynamic_Preference_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The image signal processing (ISP) pipeline is responsible for converting the RAW images collected from the sensor into high-quality RGB images. It contains a series of image processing modules and associated ISP hyperparameters. Recent learning-based approaches aim to automate ISP hyperparameter optimization using solely image data. However, their unimodal nature limits their ability to capture richer contextual information, reducing robustness and adaptability across diverse application scenarios. To address this limitation, we propose a Multimodal Large Language Model (MLLM)-guided ISP hyperparameter optimization framework, which integrates textual insights generated by MLLMs into the optimization process. By incorporating both high-level semantic cues and low-level image quality descriptors, our method enhances contextual understanding and task adaptability. Additionally, we introduce a Dynamic Pair Generation (DPG) refinement strategy based on Direct Preference Optimization (DPO), facilitating efficient preference alignment without the need for extensive human-labeled data. This two-stage framework not only improves the directional consistency of optimization but also significantly reduces the computational and data preparation overhead. We validate our proposed methods on both high-level and low-level vision tasks, demonstrating superior performance compared to existing methods.

</details>

---

## 304. Structured Policy Optimization: Enhance Large Vision-Language Model via Self-referenced Dialogue

- [ ] Structured Policy Optimization: Enhance Large Vision-Language Model via Self-referenced Dialogue | https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Structured_Policy_Optimization_Enhance_Large_Vision-Language_Model_via_Self-referenced_Dialogue_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Structured_Policy_Optimization_Enhance_Large_Vision-Language_Model_via_Self-referenced_Dialogue_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Preference optimization algorithms typically enhance LLM response quality by leveraging human feedback on multiple answers given a fixed instruction. However, these methods often lack capturing the dynamic nature of conversational exchanges. For large vision-language models (LVLMs), direct preference optimization (DPO) can over-emphasize linguistic nuances while overlooking visual context. To address this challenge, we introduce structured policy optimization (SPO) -- a novel preference optimization method that simultaneously aligns preference instructions, responses, and dialogue interactions to improve multi-modal understanding and reasoning capabilities. The efficacy of SPO is attributed to one key design:treating the questioning and answering as a sequential action and binding them through a trajectory reward. This reward formulation better aligns with real-world dialogue studies and eliminates the need for fixed instructions. We evaluate our models on interleaved benchmarks, including image, multi-image, and video-based understanding and reasoning tasks. Experimental results show that the proposed SPO fine-tuning LVLM with multi-modal preference data can align with human preference more efficiently than DPO.

</details>

---

## 305. Visual Intention Grounding for Egocentric Assistants

- [ ] Visual Intention Grounding for Egocentric Assistants | https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Visual_Intention_Grounding_for_Egocentric_Assistants_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Visual_Intention_Grounding_for_Egocentric_Assistants_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual grounding associates textual descriptions with objects in an image. Conventional methods target third-person image inputs and named object queries. In applications such as AI assistants, the perspective shifts -- inputs are egocentric, and objects may be referred to implicitly through needs and intentions. To bridge this gap, we introduce EgoIntention, the first dataset for egocentric visual intention grounding. EgoIntention challenges multimodal LLMs to 1) understand and ignore unintended contextual objects and 2) reason about uncommon object functionalities. Benchmark results show that current models misidentify context objects and lack affordance understanding in egocentric views. We also propose Reason-to-Ground (RoG) instruction tuning; it enables hybrid training with normal descriptions and egocentric intentions with a chained intention reasoning and object grounding mechanism. RoG significantly outperforms naive finetuning and hybrid training on EgoIntention, while maintaining or slightly improving naive description grounding. This advancement enables unified visual grounding for egocentric and exocentric visual inputs while handling explicit object queries and implicit human intentions.

</details>

---

## 306. X-Prompt: Generalizable Auto-Regressive Visual Learning with In-Context Prompting

- [ ] X-Prompt: Generalizable Auto-Regressive Visual Learning with In-Context Prompting | https://openaccess.thecvf.com/content/ICCV2025/html/Sun_X-Prompt_Generalizable_Auto-Regressive_Visual_Learning_with_In-Context_Prompting_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_X-Prompt_Generalizable_Auto-Regressive_Visual_Learning_with_In-Context_Prompting_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in large language models have enabled task prompting for open-ended text generation. In the vision domain, a longstanding goal is developing models capable of general visual learning, encompassing tasks such as image generation, editing, low-level processing, and dense perception. Although recent efforts have aimed at building vision foundation models that support prompting, significant challenges remain, particularly in accurately comprehending visual prompts and addressing the ambiguity inherent in textual prompts. To address this, we introduce X-Prompt, a purely auto-regressive large vision-language model designed for generalizable visual learning via in-context prompting. X-Prompt can process visual and textual prompts as context, enabling precise task interpretation and accurate execution. A novel prompt-token fusion mechanism effectively extracts relevant task information from complex prompts while significantly reducing the token length. Additionally, a unified training strategy for text and image prediction enhances task awareness, enabling seamless adaptation to open-ended prompts. Extensive experiments demonstrate that X-Prompt effectively interprets in-context prompts and exhibits generalization across both in-domain and out-of-domain visual tasks, paving the way for future advancements in general visual learning.

</details>

---

## 307. From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment

- [ ] From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment | https://openaccess.thecvf.com/content/ICCV2025/html/Suo_From_Trial_to_Triumph_Advancing_Long_Video_Understanding_via_Visual_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Suo_From_Trial_to_Triumph_Advancing_Long_Video_Understanding_via_Visual_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large language models (MLLMs) show remarkable ability in video understanding. Nevertheless, understanding long videos remains challenging as the models can only process a finite number of frames in a single inference, potentially omitting crucial visual information. To address the challenge, we propose generating multiple predictions through visual context sampling, followed by a scoring mechanism to select the final prediction. Specifically, we devise a bin-wise sampling strategy that enables MLLMs to generate diverse answers based on various combinations of keyframes, thereby enriching the visual context. To determine the final prediction from the sampled answers, we employ a self-reward by linearly combining three scores: (1) a frequency score indicating the prevalence of each option, (2) a marginal confidence score reflecting the inter-intra sample certainty of MLLM predictions, and (3) a reasoning score for different question types, including clue-guided answering for global questions and temporal self-refocusing for local questions. The frequency score ensures robustness through majority correctness, the confidence-aligned score reflects prediction certainty, and the typed-reasoning score addresses cases with sparse key visual information using tailored strategies. Experiments show that this approach covers the correct answer for a high percentage of long video questions, on seven datasets show that our method improves the performance of three MLLMs.

</details>

---

## 308. Pruning All-Rounder: Rethinking and Improving Inference Efficiency for Large Vision Language Models

- [ ] Pruning All-Rounder: Rethinking and Improving Inference Efficiency for Large Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Suo_Pruning_All-Rounder_Rethinking_and_Improving_Inference_Efficiency_for_Large_Vision_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Suo_Pruning_All-Rounder_Rethinking_and_Improving_Inference_Efficiency_for_Large_Vision_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although Large Vision-Language Models (LVLMs) have achieved impressive results, their high computational costs pose a significant barrier to wide application. To enhance inference efficiency, most existing approaches can be categorized as parameter-dependent or token-dependent strategies to reduce computational demands. However, parameter-dependent methods require retraining LVLMs to recover performance while token-dependent strategies struggle to consistently select the most relevant tokens. In this paper, we systematically analyze the above challenges and provide a series of valuable insights for inference acceleration. Based on these findings, we propose a novel framework, the Pruning All-Rounder (PAR). Different from previous works, PAR develops a meta-router to adaptively organize pruning flows across both tokens and layers. With a self-supervised learning manner, our method achieves a superior balance between performance and efficiency. Notably, PAR is highly flexible, offering multiple pruning versions to address a range of acceleration scenarios. The code for this work is publicly available at https://github.com/ASGO-MM/Pruning-All-Rounder.

</details>

---

## 309. Collaborative Instance Object Navigation: Leveraging Uncertainty-Awareness to Minimize Human-Agent Dialogues

- [ ] Collaborative Instance Object Navigation: Leveraging Uncertainty-Awareness to Minimize Human-Agent Dialogues | https://openaccess.thecvf.com/content/ICCV2025/html/Taioli_Collaborative_Instance_Object_Navigation_Leveraging_Uncertainty-Awareness_to_Minimize_Human-Agent_Dialogues_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Taioli_Collaborative_Instance_Object_Navigation_Leveraging_Uncertainty-Awareness_to_Minimize_Human-Agent_Dialogues_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language-driven instance object navigation assumes that a human initiates the task by providing a detailed description of the target to the embodied agent. While this description is crucial for distinguishing the target from other visually similar instances, providing it prior to navigation can be demanding for humans. We thus introduce Collaborative Instance object Navigation (CoIN), a new task setting where the agent actively resolves uncertainties about the target instance during navigation in natural, template-free and open-ended dialogues with the human, minimizing user input. We propose a novel training-free method, Agent-user Interaction with UncerTainty Awareness (AIUTA), which operates independently from the navigation policy, and focuses on the human-agent interaction reasoning using Vision-Language Models (VLMs) and Large Language Models (LLMs). First, upon object detection, a Self-Questioner model initiates internal self-dialogues within the agent to obtain a complete and accurate observation with a novel uncertainty estimation technique. Then, an Interaction Trigger module determines whether to ask a question to the human, continue, or halt navigation. For evaluation, we introduce CoIN-Bench, with a curated dataset designed for challenging multi-instance scenarios. CoIN-Bench supports both online evaluation with humans and reproducible experiments with simulated user-agent interactions. On CoIN-Bench, we show that AIUTA serves as a competitive baseline, whereas existing language-driven instance navigation methods struggle in multi-instance scenes.

</details>

---

## 310. BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models

- [ ] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Tang_BASIC_Boosting_Visual_Alignment_with_Intrinsic_Refined_Embeddings_in_Multimodal_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tang_BASIC_Boosting_Visual_Alignment_with_Intrinsic_Refined_Embeddings_in_Multimodal_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mainstream Multimodal Large Language Models (MLLMs) achieve visual understanding by using a vision projector to bridge well-pretrained vision encoders and large language models (LLMs). The inherent gap between visual and textual modalities makes the embeddings from the vision projector critical for visual comprehension. However, current alignment approaches treat visual embeddings as contextual cues and merely apply auto-regressive supervision to textual outputs, neglecting the necessity of introducing equivalent direct visual supervision, which hinders the potential finer alignment of visual embeddings. In this paper, based on our analysis of the refinement process of visual embeddings in the LLM's shallow layers, we propose BASIC, a method that utilizes refined visual embeddings within the LLM as supervision to directly guide the projector in generating initial visual embeddings. Specifically, the guidance is conducted from two perspectives: (i) optimizing embedding directions by reducing angles between initial and supervisory embeddings in semantic space; (ii) improving semantic matching by minimizing disparities between the logit distributions of both visual embeddings. Without additional supervisory models or artificial annotations, BASIC significantly improves the performance of MLLMs across a wide range of benchmarks, demonstrating the effectiveness of our introduced direct visual supervision.

</details>

---

## 311. AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images

- [ ] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images | https://openaccess.thecvf.com/content/ICCV2025/html/Tang_AcZeroTS_Active_Learning_for_Zero-shot_Tissue_Segmentation_in_Pathology_Images_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tang_AcZeroTS_Active_Learning_for_Zero-shot_Tissue_Segmentation_in_Pathology_Images_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Tissue segmentation in pathology images is crucial for computer-aided diagnostics of human cancers. Traditional tissue segmentation models rely heavily on large-scale labeled datasets, where every tissue type must be annotated by experts. However, due to the complexity of tumor micro-environment, collecting annotations for all possible tissue types is challenging, which makes the traditional methods ineffective in segmenting unseen tissue types with zero training samples. With the rapid development of vision-language models (VLMs), recent studies extend their powerful zero-shot capabilities to pixel-level segmentation tasks, where the model is trained only on seen classes but can perform tissue segmentation on both seen and unseen categories in the testing phase. However, these VLM-based zero-shot segmentation models still require substantial annotation efforts on seen classes. To attach desirable segmentation performance on both seen and unseen categories with limited labeled data, we propose AcZeroTS, a novel active learning framework for zero-shot tissue segmentation in pathology images. Specifically, AcZeroTS is built on a VLM-based prototype-guided zero-shot segmentation model called ProZS. We introduce a novel active selection criterion to select the most valuable samples for annotation on seen classes, which not only considers both uncertainty and diversity of unlabeled samples, but also ensures that the generated prototypes of ProZS can effectively summarize both seen and unseen classes during inference. We evaluate our method on two pathology datasets (TNBC and HPBC) as well as a natural dataset (Pascal VOC 2012), and the experimental results demonstrate the superiority of our method in comparison with the existing studies.

</details>

---

## 312. FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging

- [ ] FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging | https://openaccess.thecvf.com/content/ICCV2025/html/Tang_FinMMR_Make_Financial_Numerical_Reasoning_More_Multimodal_Comprehensive_and_Challenging_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tang_FinMMR_Make_Financial_Numerical_Reasoning_More_Multimodal_Comprehensive_and_Challenging_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present FinMMR, a novel bilingual multimodal benchmark tailored to evaluate the reasoning capabilities of multimodal large language models (MLLMs) in financial numerical reasoning tasks. Compared to existing benchmarks, our work introduces three significant advancements. (1) Multimodality: We meticulously transform existing financial reasoning benchmarks, and construct novel questions from the latest Chinese financial research reports. FinMMR comprises 4.3K questions and 8.7K images spanning 14 categories, including tables, bar charts, and ownership structure charts. (2) Comprehensiveness: FinMMR encompasses 14 financial subdomains, including corporate finance, banking, and industry analysis, significantly exceeding existing benchmarks in financial domain knowledge breadth. (3) Challenge: Models are required to perform multi-step precise numerical reasoning by integrating financial knowledge with the understanding of complex financial images and text. The best-performing MLLM achieves only 51.4% accuracy on Hard problems. We believe that FinMMR will drive advancements in enhancing the reasoning capabilities of MLLMs in real-world scenarios.

</details>

---

## 313. How Can Objects Help Video-Language Understanding?

- [ ] How Can Objects Help Video-Language Understanding? | https://openaccess.thecvf.com/content/ICCV2025/html/Tang_How_Can_Objects_Help_Video-Language_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tang_How_Can_Objects_Help_Video-Language_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Do we still need to represent objects explicitly in multimodal large language models (MLLMs)? To one extreme, pre-trained encoders convert images into visual tokens, with which objects and spatiotemporal relationships may be implicitly modeled. To the other extreme, image captions by themselves provide strong empirical performances for understanding tasks, despite missing fine-grained spatiotemporal information. To answer this question, we introduce ObjectMLLM, a framework capable of leveraging arbitrary computer vision algorithm to extract and integrate structured visual representation. Through extensive evaluations on six video question answering benchmarks, we confirm that explicit integration of object-centric representation remains necessary. Surprisingly, we observe that the simple approach of quantizing the continuous, structured object information and representing them as plain text performs the best, offering a data-efficient approach to integrate other visual perception modules into MLLM design. Our code and models are released at https://github.com/brown-palm/ObjectMLLM.

</details>

---

## 314. RoboPearls: Editable Video Simulation for Robot Manipulation

- [ ] RoboPearls: Editable Video Simulation for Robot Manipulation | https://openaccess.thecvf.com/content/ICCV2025/html/Tao_RoboPearls_Editable_Video_Simulation_for_Robot_Manipulation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tao_RoboPearls_Editable_Video_Simulation_for_Robot_Manipulation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of generalist robot manipulation policies has seen significant progress, driven by large-scale demonstration data across diverse environments. However, the high cost and inefficiency of collecting real-world demonstrations hinder the scalability of data acquisition. While existing simulation platforms enable controlled environments for robotic learning, the challenge of bridging the sim-to-real gap remains. To address these challenges, we propose RoboPearls, an editable video simulation framework for robotic manipulation. Built on 3D Gaussian Splatting (3DGS), RoboPearls enables the construction of photo-realistic, view-consistent simulations from demonstration videos, and supports a wide range of simulation operators, including various object manipulations, powered by advanced modules like Incremental Semantic Distillation (ISD) and 3D regularized NNFM Loss (3D-NNFM). Moreover, by incorporating large language models (LLMs), RoboPearls automates the simulation production process in a user-friendly manner through flexible command interpretation and execution. Furthermore, RoboPearls employs a vision-language model (VLM) to analyze robotic learning issues to close the simulation loop for performance enhancement. To demonstrate the effectiveness of RoboPearls, we conduct extensive experiments on multiple datasets and scenes, including RLBench, COLOSSEUM, Ego4D, Open X-Embodiment, and a real-world robot, which demonstrate our satisfactory simulation performance.

</details>

---

## 315. SplatTalk: 3D VQA with Gaussian Splatting

- [ ] SplatTalk: 3D VQA with Gaussian Splatting | https://openaccess.thecvf.com/content/ICCV2025/html/Thai_SplatTalk_3D_VQA_with_Gaussian_Splatting_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Thai_SplatTalk_3D_VQA_with_Gaussian_Splatting_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language-guided 3D scene understanding is important for advancing applications in robotics, AR/VR, and human-computer interaction, enabling models to comprehend and interact with 3D environments through natural language. While 2D vision-language models (VLMs) have achieved remarkable success in 2D VQA tasks, progress in the 3D domain has been significantly slower due to the complexity of 3D data and the high cost of manual annotations. In this work, we introduce SplatTalk, a novel method that uses a generalizable 3D Gaussian Splatting (3DGS) framework to produce 3D tokens suitable for direct input into a pretrained LLM, enabling effective zero-shot 3D visual question answering (3D VQA) for scenes with only posed images. During experiments on multiple benchmarks, our approach outperforms both 3D models trained specifically for the task and previous 2D-LMM-based models utilizing only images (our setting), while achieving competitive performance with state-of-the-art 3D LMMs that additionally utilize 3D inputs.

</details>

---

## 316. CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting

- [ ] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting | https://openaccess.thecvf.com/content/ICCV2025/html/Tian_CCL-LGS_Contrastive_Codebook_Learning_for_3D_Language_Gaussian_Splatting_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tian_CCL-LGS_Contrastive_Codebook_Learning_for_3D_Language_Gaussian_Splatting_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in 3D reconstruction techniques and vision-language models have fueled significant progress in 3D semantic understanding, a capability critical to robotics, autonomous driving, and virtual/augmented reality. However, methods that rely on 2D priors are prone to a critical challenge: cross-view semantic inconsistencies induced by occlusion, image blur, and view-dependent variations. These inconsistencies, when propagated via projection supervision, deteriorate the quality of 3D Gaussian semantic fields and introduce artifacts in the rendered outputs. To mitigate this limitation, we propose CCL-LGS, a novel framework that enforces view-consistent semantic supervision by integrating multi-view semantic cues. Specifically, our approach first employs a zero-shot tracker to align a set of SAM-generated 2D masks and reliably identify their corresponding categories. Next, we utilize CLIP to extract robust semantic encodings across views. Finally, our Contrastive Codebook Learning (CCL) module distills discriminative semantic features by enforcing intra-class compactness and inter-class distinctiveness. In contrast to previous methods that directly apply CLIP to imperfect masks, our framework explicitly resolves semantic conflicts while preserving category discriminability. Extensive experiments demonstrate that CCL-LGS outperforms previous state-of-the-art methods. Our project page is available at https://epsilontl.github.io/CCL-LGS/.

</details>

---

## 317. LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching

- [ ] LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching | https://openaccess.thecvf.com/content/ICCV2025/html/Tian_LLM-enhanced_Action-aware_Multi-modal_Prompt_Tuning_for_Image-Text_Matching_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tian_LLM-enhanced_Action-aware_Multi-modal_Prompt_Tuning_for_Image-Text_Matching_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Driven by large-scale contrastive vision-language pre-trained models such as CLIP, recent advancements in the image-text matching task have achieved remarkable success in representation learning. Due to image-level visual-language alignment, CLIP falls short in understanding fine-grained details such as object attributes and spatial relationships between objects. Recent efforts have attempted to compel CLIP to acquire structured visual representations by introducing prompt learning to achieve object-level alignment. While achieving promising results, they still lack the capability to perceive actions, which are crucial for describing the states or relationships between objects. Therefore, we propose to endow CLIP with fine-grained action-level understanding by introducing an LLM-enhanced action-aware multi-modal prompt-tuning method, incorporating the action-related external knowledge generated by large language models (LLMs). Specifically, we design an action triplet prompt and an action state prompt to exploit compositional semantic knowledge and state-related causal knowledge implicitly stored in LLMs. Subsequently, we propose an adaptive interaction module to aggregate attentive visual features conditioned on action-aware prompted knowledge for establishing discriminative and action-aware visual representations, which further improves the performance. Comprehensive experimental results on two benchmark datasets demonstrate the effectiveness of our method.

</details>

---

## 318. MMCR: Benchmarking Cross-Source Reasoning in Scientific Papers

- [ ] MMCR: Benchmarking Cross-Source Reasoning in Scientific Papers | https://openaccess.thecvf.com/content/ICCV2025/html/Tian_MMCR_Benchmarking_Cross-Source_Reasoning_in_Scientific_Papers_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tian_MMCR_Benchmarking_Cross-Source_Reasoning_in_Scientific_Papers_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fully comprehending scientific papers by machines reflects a high level of Artificial General Intelligence, requiring the ability to reason across fragmented and heterogeneous sources of information, presenting a complex and practically significant challenge. While Vision-Language Models (VLMs) have made remarkable strides in various tasks, particularly those involving reasoning with evidence source from single image or text page, their ability to use cross-source information for reasoning remains an open problem. This work presents MMCR, a high-difficulty benchmark designed to evaluate VLMs' capacity for reasoning with cross-source information from scientific papers. The benchmark comprises 276 high-quality questions, meticulously annotated by humans across 7 subjects and 10 task types. Experiments with 18 VLMs demonstrate that cross-source reasoning presents a substantial challenge for existing models. Notably, even the top-performing model, GPT-4o, achieved only 48.55% overall accuracy, with just 20% accuracy in multi-table comprehension tasks, while the second-best model, Qwen2.5-VL-72B, reached 39.86% overall accuracy. These results highlight the pressing need to develop VLMs capable of effectively utilizing cross-source information for reasoning.

</details>

---

## 319. Head2Body: Body Pose Generation from Multi-sensory Head-mounted Inputs

- [ ] Head2Body: Body Pose Generation from Multi-sensory Head-mounted Inputs | https://openaccess.thecvf.com/content/ICCV2025/html/Tran_Head2Body_Body_Pose_Generation_from_Multi-sensory_Head-mounted_Inputs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tran_Head2Body_Body_Pose_Generation_from_Multi-sensory_Head-mounted_Inputs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating body pose from head-mounted, egocentric inputs is essential for immersive VR/AR and assistive technologies, as it supports more natural interactions. However, the task is challenging due to limited visibility of body parts in first-person views and the sparseness of sensory data, with only a single device placed on the head. To address these challenges, we introduce Head2Body, a novel framework for body pose estimation that effectively combines IMU and visual data. First, we introduce a pre-trained IMU encoder, trained on over 1,700 hours of head-IMU data from wearable eyeglasses, to better capture detailed temporal motion cues given limited labeled egocentric pose data. For visual processing, we leverage large vision-language models (LVLMs) to segment body parts that appear sporadically in video frames to improve visual feature extraction. To better guide the pose generation process with sparse signals from only head-mounted devices, we incorporates a Vector Quantized Variational Autoencoder (VQ-VAE) to represent poses as discrete tokens, which capture high-frequency motion patterns and provide a more structured representation of body pose. Our experiments demonstrate the effectiveness of the proposed approach, yielding 6-13% gains over state-of-the-art baselines on four datasets: AMASS, KinPoly, GIMO, and EgoExo4D. By capturing subtle temporal dynamics and leveraging complementary sensory data, our approach advances accurate egocentric body pose estimation and sets a new benchmark for multi-modal, first-person motion tracking.

</details>

---

## 320. More Reliable Pseudo-labels, Better Performance: A Generalized Approach to Single Positive Multi-label Learning

- [ ] More Reliable Pseudo-labels, Better Performance: A Generalized Approach to Single Positive Multi-label Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Tran_More_Reliable_Pseudo-labels_Better_Performance_A_Generalized_Approach_to_Single_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Tran_More_Reliable_Pseudo-labels_Better_Performance_A_Generalized_Approach_to_Single_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-label learning is a challenging computer vision task that requires assigning multiple categories to each image. However, fully annotating large-scale datasets is often impractical due to high costs and effort, motivating the study of learning from partially annotated data. In the extreme case of Single Positive Multi-Label Learning (SPML), each image is provided with only one positive label, while all other labels remain unannotated. Traditional SPML methods that treat missing labels as unknown or negative tend to yield inaccuracies and false negatives, and integrating various pseudo-labeling strategies can introduce additional noise. To address these challenges, we propose the Generalized Pseudo-Label Robust Loss (GPR Loss), a novel loss function that effectively learns from diverse pseudo-labels while mitigating noise. Complementing this, we introduce a simple yet effective Dynamic Augmented Multi-focus Pseudo-labeling (DAMP) technique. Together, these contributions form the Adaptive and Efficient Vision-Language Pseudo-Labeling (AEVLP) framework. Extensive experiments on four benchmark datasets demonstrate that our framework significantly advances multi-label classification, achieving state-of-the-art results.

</details>

---

## 321. ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models

- [ ] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Wan_ONLY_One-Layer_Intervention_Sufficiently_Mitigates_Hallucinations_in_Large_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wan_ONLY_One-Layer_Intervention_Sufficiently_Mitigates_Hallucinations_in_Large_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent Large Vision-Language Models (LVLMs) have introduced a new paradigm for understanding and reasoning about image input through textual responses. Although they have achieved remarkable performance across a range of multi-modal tasks, they face the persistent challenge of hallucination, which introduces practical weaknesses and raises concerns about their reliable deployment in real-world applications. Existing work has explored contrastive decoding approaches to mitigate this issue, where the output of the original LVLM is compared and contrasted with that of a perturbed version. However, these methods require two or more queries that slow down LVLM response generation, making them less suitable for real-time applications. To overcome this limitation, we propose ONLY, a training-free decoding approach that requires only a single query and a one-layer intervention during decoding, enabling efficient real-time deployment. Specifically, we enhance textual outputs by selectively amplifying crucial textual information using a text-to-visual entropy ratio for each token. Extensive experimental results demonstrate that our ONLY approach consistently outperforms state-of-the-art methods across various benchmarks while requiring minimal implementation effort and computational cost. Code is available at https://github.com/zifuwan/ONLY.

</details>

---

## 322. BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning

- [ ] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_BabyVLM_Data-Efficient_Pretraining_of_VLMs_Inspired_by_Infant_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_BabyVLM_Data-Efficient_Pretraining_of_VLMs_Inspired_by_Infant_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human infants rapidly develop visual reasoning skills from minimal input, suggesting that developmentally inspired pretraining could significantly enhance the efficiency of vision-language models (VLMs). Although recent efforts have leveraged infant-inspired datasets like SAYCam, existing evaluation benchmarks remain misaligned--they are either too simplistic, narrowly scoped, or tailored for large-scale pretrained models. Additionally, training exclusively on SAYCam overlooks the broader, diverse input from which infants naturally learn. To address these limitations, we propose BabyVLM, a novel framework comprising diverse in-domain evaluation benchmarks and a synthetic training dataset created via child-directed transformations of existing datasets. We demonstrate that VLMs trained with our synthetic dataset achieve superior performance on BabyVLM tasks compared to models trained solely on SAYCam or general-purpose data of the SAYCam size. BabyVLM thus provides a robust, developmentally aligned evaluation tool and illustrates how compact models trained on carefully curated data can generalize effectively, opening pathways toward data-efficient vision-language learning paradigms.

</details>

---

## 323. Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval

- [ ] Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Describe_Adapt_and_Combine_Empowering_CLIP_Encoders_for_Open-set_3D_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Describe_Adapt_and_Combine_Empowering_CLIP_Encoders_for_Open-set_3D_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-set 3D object retrieval (3DOR) is an emerging task aiming to retrieve 3D objects of unseen categories beyond the training set. Existing methods typically utilize all modalities (i.e., voxels, point clouds, multi-view images) and train specific backbones before fusion. However, they still struggle to produce generalized representations due to insufficient 3D training data. Being contrastively pre-trained on web-scale image-text pairs, CLIP inherently produces generalized representations for a wide range of downstream tasks. Building upon it, we present a simple yet effective framework named Describe, Adapt and Combine (DAC) by taking only multi-view images for open-set 3DOR. DAC innovatively synergizes a CLIP model with a multi-modal large language model (MLLM) to learn generalized 3D representations, where the MLLM is used for dual purposes. First, it describes the seen category information to align with CLIP's training objective for adaptation during training. Second, it provides external hints about unknown objects complementary to visual cues during inference. To improve the synergy, we introduce an Additive-Bias Low-Rank adaptation (AB-LoRA), which alleviates overfitting and further enhances the generalization to unseen categories. With only multi-view images, DAC significantly surpasses prior arts by an average of +10.01% mAP on four open-set 3DOR datasets. Moreover, its generalization is also validated on image-based and cross-dataset setups. Code is available at https://github.com/wangzhichuan123/DAC.

</details>

---

## 324. Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM

- [ ] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Dynamic-VLM_Simple_Dynamic_Visual_Token_Compression_for_VideoLLM_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Dynamic-VLM_Simple_Dynamic_Visual_Token_Compression_for_VideoLLM_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The application of Large Vision-Language Models (LVLMs) for analyzing images and videos is an exciting and rapidly evolving field. In recent years, we've seen significant growth in high-quality image-text datasets for fine-tuning image understanding, but there is still a lack of comparable datasets for videos. Additionally, many VideoLLMs are extensions of single-image VLMs, which may not efficiently handle the complexities of longer videos. In this study, we introduce a large-scale synthetic dataset created from proprietary models, using carefully designed prompts to tackle a wide range of questions. We also explore a dynamic visual token compression architecture that strikes a balance between computational efficiency and performance. Our proposed Dynamic-VLM achieves state-of-the-art results across various video tasks and shows impressive generalization, setting new baselines in multi-image understanding. Notably, Dynamic-VLM delivers an absolute improvement of 2.7% over LLaVA-OneVision on VideoMME and 10.7% on MuirBench.

</details>

---

## 325. Enhancing Numerical Prediction of MLLMs with Soft Labeling

- [ ] Enhancing Numerical Prediction of MLLMs with Soft Labeling | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Enhancing_Numerical_Prediction_of_MLLMs_with_Soft_Labeling_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Enhancing_Numerical_Prediction_of_MLLMs_with_Soft_Labeling_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The optimality of using the de facto cross-entropy loss with one-hot target distribution (hard labeling) is questioned when training (Multimodal) Large Language Models (LLMs/MLLMs). Although it is reasonable for language token prediction, which is a typical multi-class classification problem in discrete space, it is suboptimal for task like numerical prediction, which is a typical regression problem in continuous space. However, enabling regression in LLMs/MLLMs will complicate the training and next-token prediction paradigm at inference. Instead, to address this challenge, we propose a novel loss design, called soft labeling, which smooths the target probability distribution, enabling predictions to be penalized according to their distance to the target. This is similar to regression loss, which penalizes more on the further predictions in the continuous space, but will not change the model architecture and the next-token prediction paradigm of LLMs/MLLMs. We demonstrate the efficacy of soft labeling through extensive experiments on visual grounding, object counting, and chart understanding, achieving state-of-the-art performance on multiple benchmarks without bells and whistles. Soft labeling can be applied in any LLM/MLLM.

</details>

---

## 326. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics

- [ ] Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Exploring_the_Adversarial_Vulnerabilities_of_Vision-Language-Action_Models_in_Robotics_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Exploring_the_Adversarial_Vulnerabilities_of_Vision-Language-Action_Models_in_Robotics_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. Despite their significant capabilities, VLA models introduce new attack surfaces. This paper systematically evaluates their robustness. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.

</details>

---

## 327. FOLDER: Accelerating Multi-Modal Large Language Models with Enhanced Performance

- [ ] FOLDER: Accelerating Multi-Modal Large Language Models with Enhanced Performance | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_FOLDER_Accelerating_Multi-Modal_Large_Language_Models_with_Enhanced_Performance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_FOLDER_Accelerating_Multi-Modal_Large_Language_Models_with_Enhanced_Performance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Multi-modal Large Language Models (MLLMs) have shown remarkable effectiveness for multi-modal tasks due to their ability of cross-modal understanding. However, processing long sequences of visual tokens extracted from visual backbones poses challenges for deployment in real-time applications. To address this issue, we introduce FOLDER, a simple yet effective plug-and-play module designed to reduce the length of the visual token sequence, mitigating computational and memory demands during both training and inference. Through a comprehensive analysis of the token reduction process in the vision encoder, we analyze the information loss introduced by different reduction strategies and develop FOLDER to preserve key information while removing visual redundancy. We show the effectiveness of FOLDER by integrating it into the visual backbone of various MLLMs, significantly accelerating the inference phase. Furthermore, we evaluate its utility as a training accelerator or even performance booster for MLLMs. FOLDER achieves comparable or even better performance than the original models, while dramatically reducing complexity by removing up to 70% of visual tokens. Our code is available at https://github.com/anakin-skywalker-Joseph/Folder.

</details>

---

## 328. Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text

- [ ] Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Fix-CLIP_Dual-Branch_Hierarchical_Contrastive_Learning_via_Synthetic_Captions_for_Better_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Fix-CLIP_Dual-Branch_Hierarchical_Contrastive_Learning_via_Synthetic_Captions_for_Better_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

CLIP has shown promising performance across many short-text tasks in a zero-shot manner. However, limited by the input length of the text encoder, CLIP struggles on under-stream tasks with long-text inputs (>77 tokens). To remedy this issue, we propose FIX-CLIP, which includes three novel modules: (1) A dual-branch training pipeline that aligns short and long texts with masked and raw images, respectively, which boosts the long-text representation while preserving the short-text ability. (2) Multiple learnable regional prompts with unidirectional masks in Transformer layers for regional information extraction. (3) A hierarchical feature alignment module in the intermediate encoder layers to promote the consistency of multi-scale features. Furthermore, we collect 30M images and utilize existing MLLMs to synthesize long-text captions for training. Extensive experiments show that FIX-CLIP achieves state-of-the-art performance on both long-text and short-text retrieval benchmarks. For downstream applications, we reveal that FIX-CLIP's text encoder delivers promising performance in a plug-and-play manner for diffusion models with long-text input. The code is available at https://github.com/bcwang-sjtu/Fix-CLIP.

</details>

---

## 329. Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference

- [ ] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Free-MoRef_Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Free-MoRef_Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video Multimodal Large Language Models (Video-MLLM) have achieved remarkable advancements in video understanding tasks. However, constrained by the context length limitation in the underlying LLMs, existing Video-MLLMs typically exhibit suboptimal performance on long video scenarios. To understand extended input frames, common solutions span token compression and streaming inference techniques, which sacrifice feature granularity or inference efficiency. Differently, to efficiently achieve comprehensive understanding of longer frame inputs, we draw ideas from MoE and propose a training-free approach Free-MoRef, which instantly multiplexes the context perception capabilities of Video-MLLMs within one inference pass. Specifically, Free-MoRef reconstructs the vision tokens into several short sequences as multi-references. Subsequently, we introduce MoRef-attention, which gathers clues from the multi-reference chunks in parallel to summarize unified query activations. After the shadow layers in LLMs, a reference fusion step is derived to compose a final mixed reasoning sequence with key tokens from parallel chunks, which compensates the cross-reference vision interactions that are neglected in MoRef-attention. By splitting and fusing the long vision token sequences, Free-MoRef achieves improved performance under much lower computing costs in reasoning multiplexed context length, demonstrating strong efficiency and effectiveness. Experiments on VideoMME, MLVU, LongVideoBench show that Free-MoRef achieves full perception of 2xto 8xlonger input frames without compression on a single A100 GPU while keeping instant responses, thereby bringing significant performance gains, even surpassing dedicatedly trained long-video-MLLMs. Codes are available at https://github.com/wkfdb/Free-MoRef

</details>

---

## 330. How Do Multimodal Large Language Models Handle Complex Multimodal Reasoning? Placing Them in An Extensible Escape Game

- [ ] How Do Multimodal Large Language Models Handle Complex Multimodal Reasoning? Placing Them in An Extensible Escape Game | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_How_Do_Multimodal_Large_Language_Models_Handle_Complex_Multimodal_Reasoning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_How_Do_Multimodal_Large_Language_Models_Handle_Complex_Multimodal_Reasoning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancing of Multimodal Large Language Models (MLLMs) has spurred interest in complex multimodal reasoning tasks in the real-world and virtual environment, which require coordinating multiple abilities, including visual perception, visual reasoning, spatial awareness, and target deduction. However, existing evaluations primarily assess the final task completion, often degrading assessments to isolated abilities such as visual grounding and visual question answering. Less attention is given to comprehensively and quantitatively analyzing reasoning process in multimodal environments, which is crucial for understanding model behaviors and underlying reasoning mechanisms beyond merely task success. To address this, we introduce MM-Escape, an extensible benchmark for investigating multimodal reasoning, inspired by real-world escape games. MM-Escape emphasizes intermediate model behaviors alongside final task completion. To achieve this, we develop EscapeCraft, a customizable and open environment that enables models to engage in free-form exploration for assessing multimodal reasoning. Extensive experiments show that MLLMs, regardless of scale, can successfully complete the simplest room escape tasks, with some exhibiting human-like exploration strategies. Yet, performance dramatically drops as task difficulty increases. Moreover, we observe that models severely suffer from accidental success, and that performance bottlenecks vary across models, revealing distinct failure modes and limitations in their multimodal reasoning abilities, such as repetitive trajectories without adaptive exploration, getting stuck in corners due to poor visual spatial awareness, and ineffective use of acquired props, such as the key. We hope our work sheds light on new challenges in multimodal reasoning, and uncovers potential improvements in MLLMs capabilities.

</details>

---

## 331. ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance

- [ ] ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_ILLUME_Illuminating_Your_LLMs_to_See_Draw_and_Self-Enhance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_ILLUME_Illuminating_Your_LLMs_to_See_Draw_and_Self-Enhance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce ILLUME, a unified multimodal large language model (MLLM) that seamlessly integrates multimodal understanding and generation capabilities within a single large language model through a unified next-token prediction formulation.To address the large dataset size typically required for image-text alignment, we propose to enhance data efficiency through the design of a vision tokenizer that incorporates semantic information and a progressive multi-stage training procedure. This approach reduces the dataset size to just 15M for pretraining -- over four times fewer than what is typically needed -- while achieving competitive or even superior performance with existing unified MLLMs, such as Janus. Additionally, to promote synergistic enhancement between understanding and generation capabilities, which is under-explored in previous works, we introduce a novel self-enhancing multimodal alignment scheme. This scheme supervises the MLLM to self-assess the consistency between text descriptions and self-generated images, facilitating the model to interpret images more accurately and avoid unrealistic and incorrect predictions caused by misalignment in image generation. Based on our extensive experiments, our proposed ILLUME stands out and competes with state-of-the-art unified MLLMs and specialized models across various benchmarks for multimodal understanding, generation, and editing.

</details>

---

## 332. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves

- [ ] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_IDEATOR_Jailbreaking_and_Benchmarking_Large_Vision-Language_Models_Using_Themselves_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_IDEATOR_Jailbreaking_and_Benchmarking_Large_Vision-Language_Models_Using_Themselves_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks--techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.

</details>

---

## 333. Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs

- [ ] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Instruction-Oriented_Preference_Alignment_for_Enhancing_Multi-Modal_Comprehension_Capability_of_MLLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Instruction-Oriented_Preference_Alignment_for_Enhancing_Multi-Modal_Comprehension_Capability_of_MLLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Preference alignment has emerged as an effective strategy to enhance the performance of Multimodal Large Language Models (MLLMs) following supervised fine-tuning. While existing preference alignment methods predominantly target hallucination factors, they overlook the factors essential for multi-modal comprehension capabilities, often narrowing their improvements on hallucination mitigation. To bridge this gap, we propose Instruction-oriented Preference Alignment (IPA), a scalable framework designed to automatically construct alignment preferences grounded in instruction fulfillment efficacy. Our method involves an automated preference construction coupled with a dedicated verification process that identifies instruction-oriented factors, avoiding significant variability in response representations. Additionally, IPA incorporates a progressive preference collection pipeline, further recalling challenging samples through model self-evolution and reference-guided refinement. Experiments conducted on Qwen2VL-7B demonstrate IPA's effectiveness across multiple benchmarks, including hallucination evaluation, visual question answering, and text understanding tasks, highlighting its capability to enhance general comprehension.

</details>

---

## 334. Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation

- [ ] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Is_Less_More_Exploring_Token_Condensation_as_Training-free_Test-time_Adaptation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Is_Less_More_Exploring_Token_Condensation_as_Training-free_Test-time_Adaptation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pretraining (CLIP) excels at learning generalizable image representations but often falls short in zero-shot inference on certain downstream datasets. Test-time adaptation (TTA) mitigates this issue by adjusting components like normalization layers or context prompts, yet it typically requires large batch sizes and extensive augmentations, leading to high computational costs. This raises a key question: Can VLMs' performance drop in specific test cases be mitigated through efficient, training-free approaches? To explore the solution, we investigate token condensation (TC) techniques, originally designed to enhance vision transformer efficiency by refining token usage during inference. We observe that informative tokens improve visual-text alignment in VLMs like CLIP on unseen datasets. However, existing TC methods often fail to maintain in-distribution performance when reducing tokens, prompting us to ask: How can we transform TC into an effective "free-lunch" adaptation strategy for VLMs? To address this, we propose Token Condensation as Adaptation (TCA), a training-free adaptation method that takes a step beyond standard TC. Rather than passively discarding tokens, TCA condenses token representation by introducing reservoir-based domain anchor tokens for information-preserving token reduction and logit correction. TCA achieves up to a 21.4% performance improvement over the strongest baseline on cross-dataset benchmark and the CIFAR-100-Corrupted dataset while reducing GFLOPs by 12.2% to 48.9%, with minimal hyperparameter dependency on both CLIP and SigLIP series. Code is available at https://github.com/Jo-wang/TCA.

</details>

---

## 335. LVBench: An Extreme Long Video Understanding Benchmark

- [ ] LVBench: An Extreme Long Video Understanding Benchmark | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LVBench_An_Extreme_Long_Video_Understanding_Benchmark_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LVBench_An_Extreme_Long_Video_Understanding_Benchmark_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in multimodal large language models has markedly enhanced the understanding of short videos (typically under one minute), and several evaluation datasets have emerged accordingly. However, these advancements fall short of meeting the demands of real-world applications such as embodied intelligence for long-term decision-making, in-depth movie reviews and discussions, and live sports commentary, all of which require comprehension of long videos spanning several hours. To address this gap, we introduce LVBench, a benchmark specifically designed for long video understanding. Our dataset comprises publicly sourced videos and encompasses a diverse set of tasks aimed at long video comprehension and information extraction. LVBench is designed to challenge multimodal models to demonstrate long-term memory and extended comprehension capabilities. Our extensive evaluations reveal that current multimodal models still underperform on these demanding long video understanding tasks. Through LVBench, we aim to spur the development of more advanced models capable of tackling the complexities of long video comprehension.

</details>

---

## 336. Mamba-3VL: Taming State Space Model for 3D Vision Language Learning

- [ ] Mamba-3VL: Taming State Space Model for 3D Vision Language Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Mamba-3VL_Taming_State_Space_Model_for_3D_Vision_Language_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Mamba-3VL_Taming_State_Space_Model_for_3D_Vision_Language_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D vision-language (3D-VL) reasoning, connecting natural language with 3D physical world, represents a milestone in advancing spatial intelligence. While transformer-based methods dominate 3D-VL research, their quadratic complexity and simplistic positional embedding mechanisms severely limits effective modeling of long-range 3D-VL dependencies and spatial relationships in 3D-VL tasks. State Space Models (SSM) have emerged as promising linear-complexity alternatives for sequential data processing, while inherent selection mechanism offers notable capability for spatial modeling. Despite its potential, straightforward adoption of Mamba to 3D-VL tasks encounters two obstacles: (1) how to perceive the position of 3D objects and understand complex spatial relationships, and (2) how to achieve thorough synergies of multi-modal features. In this paper, we propose Mamba-3VL, a pioneering 3D-VL framework to model complex intra- and inter-modality correlations and enhance spatial relation reasoning, while guaranteeing top-tier performance, high efficiency, and generalization potential for 3D-VL tasks. Specifically, Mamba Mixer explicitly models 3D-VL interaction via channel twisting and relation-prioritized spatial scanning policy. It maximally retain spatial relation of object-centric features. To further provide precise spatial encoding for mamba, we develop Instance-aware Dynamic Position Adapter (IDPA) to dynamically adjust instance-specific positional embeddings and enhance local spatial relation of 3D objects. Extensive results validate Mamba-3VL trumps other competitors on seven 3D-VL benchmarks and showcases versatile potentials for challenging Embodied AI tasks.

</details>

---

## 337. Open-Vocabulary Octree-Graph for 3D Scene Understanding

- [ ] Open-Vocabulary Octree-Graph for 3D Scene Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Open-Vocabulary_Octree-Graph_for_3D_Scene_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Open-Vocabulary_Octree-Graph_for_3D_Scene_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary 3D scene understanding is indispensable for embodied agents. Recent works leverage pretrained vision-language models (VLMs) for object segmentation and project them to point clouds to build 3D maps. Despite progress, a point cloud is a set of unordered coordinates that requires substantial storage space and can not directly convey occupancy information or spatial relation, making existing methods inefficient for downstream tasks, e.g., path planning and complex text-based object retrieval. To address these issues, we propose Octree-Graph, a novel scene representation for open-vocabulary 3D scene understanding. Specifically, a Chronological Group-wise Segment Merging (CGSM) strategy and an Instance Feature Aggregation (IFA) algorithm are first designed to get 3D instances and corresponding semantic features. Subsequently, an adaptive-octree structure is developed that stores semantics and depicts the occupancy of an object adjustably according to its shape. Finally, the Octree-Graph is constructed where each adaptive-octree acts as a graph node, and edges describe the spatial relations among nodes. Extensive experiments on various tasks are conducted on several widely-used datasets, demonstrating the versatility and effectiveness of our method. Code is available at https://github.com/yifeisu/OV-Octree-Graph.

</details>

---

## 338. OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM

- [ ] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_OrderChain_Towards_General_Instruct-Tuning_for_Stimulating_the_Ordinal_Understanding_Ability_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_OrderChain_Towards_General_Instruct-Tuning_for_Stimulating_the_Ordinal_Understanding_Ability_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the remarkable progress of multimodal large language models (MLLMs), they continue to face challenges in achieving competitive performance on ordinal regression (OR; a.k.a. ordinal classification). To address this issue, this paper presents OrderChain, a novel and general prompting paradigm that improves the ordinal understanding ability of MLLMs by specificity and commonality modeling. Specifically, our OrderChain consists of a set of task-aware prompts to facilitate the specificity modeling of diverse OR tasks and a new range optimization Chain-of-Thought (RO-CoT), which learns a commonality way of thinking about OR tasks by uniformly decomposing them into multiple small-range optimization subtasks. Further, we propose a category recursive division (CRD) method to generate instruction candidate category prompts to support RO-CoT automatic optimization. Comprehensive experiments show that LLaVA model with our OrderChain improves baseline LLaVA significantly on diverse OR datasets, e.g., from 47.5% to 93.2% accuracy on the Adience dataset for age estimation, and from 30.0% to 85.7% accuracy on the Diabetic Retinopathy dataset. Notably, LLaVA with our OrderChain also remarkably outperforms state-of-the-art methods by 27% on accuracy and 0.24 on MAE on the Adience dataset. To our best knowledge, our OrderChain is the first work that augments MLLMs for OR tasks, and the effectiveness is witnessed across a spectrum of OR datasets. Project Page: https://order-chain.github.io/.

</details>

---

## 339. Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness

- [ ] Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Ross3D_Reconstructive_Visual_Instruction_Tuning_with_3D-Awareness_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Ross3D_Reconstructive_Visual_Instruction_Tuning_with_3D-Awareness_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of Large Multimodal Models (LMMs) for 2D images and videos has spurred efforts to adapt these models for interpreting 3D scenes. However, the absence of large-scale 3D vision-language datasets has posed a significant obstacle. To address this issue, typical approaches focus on injecting 3D awareness into 2D LMMs by designing 3D input-level scene representations. This work provides a new perspective. We introduce reconstructive visual instruction tuning with 3D-awareness (ROSS3D), which integrates 3D-aware visual supervision into the training procedure. Specifically, it incorporates cross-view and global-view reconstruction. The former requires reconstructing masked views by aggregating overlapping information from other views. The latter aims to aggregate information from all available views to recover Bird's-Eye-View images, contributing to a comprehensive overview of the entire scene. Empirically, ROSS3D achieves state-of-the-art performance across various 3D scene understanding benchmarks. More importantly, our semi-supervised experiments demonstrate significant potential in leveraging large amounts of unlabeled 3D vision-only data.

</details>

---

## 340. SAMPLE: Semantic Alignment through Temporal-Adaptive Multimodal Prompt Learning for Event-Based Open-Vocabulary Action Recognition

- [ ] SAMPLE: Semantic Alignment through Temporal-Adaptive Multimodal Prompt Learning for Event-Based Open-Vocabulary Action Recognition | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SAMPLE_Semantic_Alignment_through_Temporal-Adaptive_Multimodal_Prompt_Learning_for_Event-Based_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SAMPLE_Semantic_Alignment_through_Temporal-Adaptive_Multimodal_Prompt_Learning_for_Event-Based_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary action recognition (OVAR) extends recognition systems to identify unseen action categories. While large-scale vision-language models (VLMs) like CLIP have enabled OVAR in image domains, their adaptation to event data remains underexplored. Event cameras offer high temporal resolution and inherent privacy preservation, making them suitable for capturing fine-grained motion dynamics. However, leveraging event data for OVAR presents challenges: 1) bridging the domain gap between static image-based models and event streams, and 2) preserving the generalization capabilities of pretrained VLMs in open-vocabulary settings. In this paper, we propose SAMPLE, a lightweight adaptation of VLMs for event-based action recognition, balancing supervised and open-vocabulary performance. We introduce a Temporal-Adaptive Multimodal Prompt Learning strategy that can be divided into: 1) Unimodal prompt on both the event and text branches to learn the data distribution 2) Event-Text cross-modal prompt for representation space alignment 3) Temporal-Adaptive prompt to model temporal dependencies across event data. Extensive evaluations demonstrate that SAMPLE outperforms prior methods across fully supervised, few-shot, base-to-novel and zero-shot settings. Notably, in zero-shot scenarios, SAMPLE achieves gains of +15.46%, +29.76%, and +23.79% on SeAct, DVS128Gesture, and PAF respectively with less commute cost. Our codes are released at https://github.com/JingWang-self/SAMPLE.

</details>

---

## 341. SHIFT: Smoothing Hallucinations by Information Flow Tuning for Multimodal Large Language Models

- [ ] SHIFT: Smoothing Hallucinations by Information Flow Tuning for Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SHIFT_Smoothing_Hallucinations_by_Information_Flow_Tuning_for_Multimodal_Large_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SHIFT_Smoothing_Hallucinations_by_Information_Flow_Tuning_for_Multimodal_Large_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) are prone to hallucinations, which pose significant risks in their applications. Most existing hallucination detection methods rely on internal probabilities or external knowledge, and they are limited to identifying hallucinations at the sentence or passage level. In this paper, we introduce the first token-level, zero-resource hallucination detection framework, leveraging a novel approach inspired by the Mad Libs game. This method assesses the reliability of the input text by evaluating the consistency of information before and after the game. Building on this framework, we also propose an innovative automated hallucination generation technique and introduce a high-quality hallucination dataset, HalluWiki. Extensive experiments demonstrate that our approach achieves over 90% detection accuracy across different levels, establishing a new frontier in hallucination detection for LLMs.

</details>

---

## 342. Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension

- [ ] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Scaling_Inference-Time_Search_with_Vision_Value_Model_for_Improved_Visual_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Scaling_Inference-Time_Search_with_Vision_Value_Model_for_Improved_Visual_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant advancements in vision-language models (VLMs), there lacks effective approaches to enhance response quality by scaling inference-time computation. This capability is known to be a core step towards the self-improving models in recent large language model studies. In this paper, we present Vision Value Model (VisVM) that can guide VLM inference-time search to generate responses with better visual comprehension. Specifically, VisVM not only evaluates the generated sentence quality in the current search step, but also anticipates the quality of subsequent sentences that may result from the current step, thus providing a long-term value. In this way, VisVM steers VLMs away from generating sentences prone to hallucinations or insufficient detail, thereby producing higher quality responses. Experimental results demonstrate that VisVM-guided search significantly enhances VLMs' ability to generate descriptive captions with richer visual details and fewer hallucinations, compared with greedy decoding and search methods with other visual reward signals. Furthermore, we find that self-training the model with the VisVM-guided captions improve VLM's performance across a wide range of multimodal benchmarks, indicating the potential for developing self-improving VLMs.

</details>

---

## 343. SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning

- [ ] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual instruction tuning (VIT) enables multimodal large language models (MLLMs) to effectively handle a wide range of vision tasks by framing them as language-based instructions. Building on this, continual visual instruction tuning (CVIT) extends the capability of MLLMs to incrementally learn new tasks, accommodating evolving functionalities. While prior work has advanced CVIT through the development of new benchmarks and approaches to mitigate catastrophic forgetting, these efforts largely follow traditional continual learning paradigms, neglecting the unique challenges specific to CVIT. We identify a dual form of catastrophic forgetting in CVIT, where MLLMs not only forget previously learned visual understanding but also experience a decline in instruction following abilities as they acquire new tasks. To address this, we introduce the Separable Mixture of Low-Rank Adaptation (SMoLoRA) framework, which employs separable routing through two distinct modules--one for visual understanding and another for instruction following. This dual-routing design enables specialized adaptation in both domains, preventing forgetting while improving performance. Furthermore, we propose a new CVIT benchmark that goes beyond existing benchmarks by additionally evaluating a model's ability to generalize to unseen tasks and handle diverse instructions across various tasks. Extensive experiments demonstrate that SMoLoRA outperforms existing methods in mitigating dual forgetting, improving generalization to unseen tasks, and ensuring robustness in following diverse instructions.

</details>

---

## 344. SITE: towards Spatial Intelligence Thorough Evaluation

- [ ] SITE: towards Spatial Intelligence Thorough Evaluation | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SITE_towards_Spatial_Intelligence_Thorough_Evaluation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SITE_towards_Spatial_Intelligence_Thorough_Evaluation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spatial intelligence (SI) represents a cognitive ability encompassing the visualization, manipulation, and reasoning about spatial relationships, underpinning disciplines from neuroscience to robotics. We introduce SITE, a benchmark dataset towards SI Thorough Evaluation in a standardized format of multi-choice visual question-answering, designed to assess large vision-language models' spatial intelligence across diverse visual modalities (single-image, multi-image, and video) and SI factors (figural to environmental scales, spatial visualization and orientation, intrinsic and extrinsic, static and dynamic). Our approach to curating the benchmark combines a bottom-up survey of existing datasets and a top-down strategy drawing upon three classification systems in cognitive science, which prompt us to design two novel types of tasks about view-taking and dynamic scenes. Extensive experiments reveal that leading models fall behind human experts, especially in spatial orientation, a fundamental SI factor. Moreover, we demonstrate a positive correlation between a model's spatial reasoning proficiency and its performance on an embodied AI task.

</details>

---

## 345. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks

- [ ] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Safeguarding_Vision-Language_Models_Mitigating_Vulnerabilities_to_Gaussian_Noise_in_Perturbation-based_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Safeguarding_Vision-Language_Models_Mitigating_Vulnerabilities_to_Gaussian_Noise_in_Perturbation-based_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.

</details>

---

## 346. SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

- [ ] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SparseMM_Head_Sparsity_Emerges_from_Visual_Concept_Responses_in_MLLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SparseMM_Head_Sparsity_Emerges_from_Visual_Concept_Responses_in_MLLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are commonly derived by extending pre-trained Large Language Models (LLMs) with visual capabilities. In this work, we investigate how MLLMs process visual inputs by analyzing their attention mechanisms. We reveal a surprising sparsity phenomenon: only a small subset (approximately less than 5%) of attention heads in LLMs actively contribute to visual understanding, termed visual heads. To identify these heads efficiently, we design a training-free framework that quantifies head-level visual relevance through targeted response analysis. Building on this discovery, we introduce SparseMM, a KV-Cache optimization strategy that allocates asymmetric computation budgets to heads in LLMs based on their visual scores, leveraging the sparity of visual heads for accelerating the inference of MLLMs. Compared with prior KV-Cache acceleration methods that ignore the particularity of visual, SparseMM prioritizes stress and retaining visual semantics during decoding. Extensive evaluations across mainstream multimodal benchmarks demonstrate that SparseMM achieves superior accuracy-efficiency trade-offs. Notably, SparseMM delivers 1.38x real-time acceleration and 52% memory reduction during generation while maintaining performance parity on efficiency test. Our project is open sourced at https://github.com/CR400AF-A/SparseMM.

</details>

---

## 347. Semantic Discrepancy-aware Detector for Image Forgery Identification

- [ ] Semantic Discrepancy-aware Detector for Image Forgery Identification | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Semantic_Discrepancy-aware_Detector_for_Image_Forgery_Identification_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Semantic_Discrepancy-aware_Detector_for_Image_Forgery_Identification_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of image generation techniques, robust forgery detection has become increasingly imperative to ensure the trustworthiness of digital media. Recent research indicates that the learned semantic concepts of pre-trained models are critical for identifying fake images. However, the misalignment between the forgery and semantic concept spaces hinders the model's forgery detection performance. To address this problem, we propose a novel Semantic Discrepancy-aware Detector (SDD) that leverages reconstruction learning to align the two spaces at a fine-grained visual level. By exploiting the conceptual knowledge embedded in the pre-trained vision-language model, we specifically design a semantic token sampling module to mitigate the space shifts caused by features irrelevant to both forgery traces and semantic concepts. A concept-level forgery discrepancy learning module, built upon a visual reconstruction paradigm, is proposed to strengthen the interaction between visual semantic concepts and forgery traces, effectively capturing discrepancies under the concepts' guidance. Finally, the low-level forgery feature enhancemer integrates the learned concept-level forgery discrepancies to minimize redundant forgery information. Experiments conducted on two standard image forgery datasets demonstrate the efficacy of the proposed SDD, which achieves superior results compared to existing methods. The code is available at https://github.com/wzy1111111/SSD.

</details>

---

## 348. Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown

- [ ] Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Taming_the_Untamed_Graph-Based_Knowledge_Retrieval_and_Reasoning_for_MLLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Taming_the_Untamed_Graph-Based_Knowledge_Retrieval_and_Reasoning_for_MLLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The real value of knowledge lies not just in its accumulation, but in its potential to be harnessed effectively to conquer the unknown. Although recent multimodal large language models (MLLMs) exhibit impressing multimodal capabilities, they often fail in rarely encountered domain-specific tasks due to limited relevant knowledge. To explore this, we adopt visual game cognition as a testbed and select "Monster Hunter: World" as the target to construct a multimodal knowledge graph (MH-MMKG), which incorporates multi-modalities and intricate entity relations. We also design a series of challenging queries based on MH-MMKG to evaluate the models' ability for complex knowledge retrieval and reasoning. Furthermore, we propose a multi-agent retriever that enables a model to autonomously search relevant knowledge without additional training. Experimental results show that our approach significantly enhances the performance of MLLMs, providing a new perspective on multimodal knowledge-augmented reasoning and laying a solid foundation for future research.

</details>

---

## 349. Toward Fair and Accurate Cross-Domain Medical Image Segmentation: A VLM-Driven Active Domain Adaptation Paradigm

- [ ] Toward Fair and Accurate Cross-Domain Medical Image Segmentation: A VLM-Driven Active Domain Adaptation Paradigm | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Toward_Fair_and_Accurate_Cross-Domain_Medical_Image_Segmentation_A_VLM-Driven_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Toward_Fair_and_Accurate_Cross-Domain_Medical_Image_Segmentation_A_VLM-Driven_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fairness in AI-assisted medical image analysis is crucial for equitable healthcare, but is often neglected, especially in prevalent cross-domain scenarios (diverse demographics and imaging protocols). Effective and equitable deployment of AI models in these scenarios is critical, yet traditional Unsupervised Domain Adaptation (UDA) methods exhibit limited improvements. Emerging Active Domain Adaptation (ADA) approaches offer more effective enhancements, but all ignore fairness issues. Therefore, in this work, we propose the first fairness-aware ADA paradigm that simultaneously achieves both enhanced fairness and superior overall performance. Our method leverages the multimodal alignment capability of Vision-Language Models (VLMs): By performing medical images (vision) and sensitive attributes (language) learning, VLM inherently captures semantic correlations between visual features and protected attributes, enabling explicit attributes representation. Building on this foundation, we further devise an attribute-aware strategy (FairAP), which dynamically adapts to diverse patient demographics to promote equitable and high-quality outcomes by considering both Attribute and Polysemy. Extensive experiments on the FairDomain benchmark demonstrate that our method significantly reduces bias and maintains leading performance, outperforming existing UDA and ADA methods. This work pioneers a VLM-driven ADA paradigm for fair cross-domain medical segmentation, offering a blueprint for effective and equitable AI deployment in clinical practice. Code will be on (https://github.com/whq-xxh/Fair-AP).

</details>

---

## 350. Towards Annotation-Free Evaluation: KPAScore for Human Keypoint Detection

- [ ] Towards Annotation-Free Evaluation: KPAScore for Human Keypoint Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Towards_Annotation-Free_Evaluation_KPAScore_for_Human_Keypoint_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Towards_Annotation-Free_Evaluation_KPAScore_for_Human_Keypoint_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human keypoint detection is fundamental in computer vision, with applications in pose estimation and action recognition. However, existing evaluation metrics (e.g., OKS, PCP, PDJ) rely on human-annotated ground truth, a labor-intensive process that increases costs, limits scalability. To address this, we propose KPAScore (KeyPoint-Answering Score), an annotation-free metric independent of ground truth. It evaluates keypoint detection using a two-stage VLM-based question-answering process: first, the VLM identifies the presence of keypoints within the image, and second, visual prompts are introduced to query the likelihood of each keypoint being accurately localized within a predefined boundary. To validate the rationale behind KPAScore, we propose KPUBench (KeyPoint Understanding Benchmark), which comprehensively evaluates the VLM's ability to determine keypoint presence and localization. Extensive experiments demonstrate KPAScore's effectiveness from three perspectives: consistency to keypoint variation, correlation with traditional metrics, alignment with human perception. We hope KPAScore will reduce reliance on manual annotations, facilitating broader adoption of keypoint detection in real-world applications.

</details>

---

## 351. VISO: Accelerating In-orbit Object Detection with Language-Guided Mask Learning and Sparse Inference

- [ ] VISO: Accelerating In-orbit Object Detection with Language-Guided Mask Learning and Sparse Inference | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VISO_Accelerating_In-orbit_Object_Detection_with_Language-Guided_Mask_Learning_and_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VISO_Accelerating_In-orbit_Object_Detection_with_Language-Guided_Mask_Learning_and_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In-orbit object detection is essential for Earth observation missions on satellites equipped with GPUs. A promising approach is to use pre-trained vision-language modeling (VLM) to enhance its open-vocabulary capability. However, adopting it on satellites poses two challenges: (1) satellite imagery differs substantially from natural images, and (2) satellites' embedded GPUs are insufficient for complex models' inference. We reveal their lack of a crucial prior: in-orbit detection involves identifying a set of known objects within a cluttered yet monotonous background. Motivated by this observation, we propose VISO, a Vision-language Instructed Satellite Object detection model that focuses on object-specific features while suppressing irrelevant regions through language-guided mask learning. After pre-training on a large-scale satellite dataset with 3.4M region-text pairs, VISO enhances object-text alignment and object-centric features to improve detection accuracy. Also, VISO suppresses irrelevant regions, enabling highly sparse inference to accelerate speed on satellites. Extensive experiments show that VISO without sparsity outperforms state-of-the-art (SOTA) VLMs in zero-shot detection by increasing 34.1% AP and reducing 27xFLOPs, and surpasses specialist models in supervised object detection and object referring by improving 2.3% AP. When sparsifying VISO to a comparable AP, FLOPs can be greatly reduced by up to 8.5x. Real-world tests reveal that VISO achieves a 2.8-4.8xFPS speed-up on satellites' embedded GPUs.

</details>

---

## 352. VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers

- [ ] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VQ-VLA_Improving_Vision-Language-Action_Models_via_Scaling_Vector-Quantized_Action_Tokenizers_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VQ-VLA_Improving_Vision-Language-Action_Models_via_Scaling_Vector-Quantized_Action_Tokenizers_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce an innovative vector quantization based action tokenizer built upon the largest-scale action trajectory dataset to date, leveraging over 100 times more data than previous approaches. This extensive dataset enables our tokenizer to capture rich spatiotemporal dynamics, resulting in a model that not only accelerates inference but also generates smoother and more coherent action outputs. Once trained, the tokenizer can be seamlessly adapted to a wide range of downstream tasks in a zero-shot manner, from short-horizon reactive behaviors to long-horizon planning. A key finding of our work is that the domain gap between synthetic and real action trajectories is marginal, allowing us to effectively utilize a vast amount of synthetic data during training without compromising real-world performance. To validate our approach, we conducted extensive experiments in both simulated environments and on real robotic platforms. The results demonstrate that as the volume of synthetic trajectory data increases, the performance of our tokenizer on downstream tasks improves significantly-most notably, achieving up to a 30% higher success rate on two real-world tasks in long-horizon scenarios. These findings highlight the potential of our action tokenizer as a robust and scalable solution for real-time embodied intelligence systems, paving the way for more efficient and reliable robotic control in diverse application domains.

</details>

---

## 353. VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges

- [ ] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VideoLLaMB_Long_Streaming_Video_Understanding_with_Recurrent_Memory_Bridges_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VideoLLaMB_Long_Streaming_Video_Understanding_with_Recurrent_Memory_Bridges_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large-scale video-language models have shown significant potential for real-time planning and detailed interactions. However, their high computational demands and the scarcity of annotated datasets limit their practicality for academic researchers. In this work, we introduce VideoLLaMB, a novel and efficient framework for long video understanding that leverages recurrent memory bridges and temporal memory tokens to enable seamless encoding of entire video sequences with preserved semantic continuity. Central to our approach is a SceneTiling algorithm that segments videos into coherent semantic units, facilitating robust understanding across tasks without requiring additional training. VideoLLaMB achieves state-of-the-art performance, surpassing existing models by 4.2 points on four VideoQA benchmarks and by 2.06 points on egocentric planning tasks. Notably, it maintains strong performance under extreme video length scaling (up to 8x) and excels at fine-grained frame retrieval on our proposed Needle in a Video Haystack (NIAVH) benchmark. With linear GPU memory scaling, VideoLLaMB processes up to 320 frames using a single Nvidia A100 GPU, despite being trained on only 16 frames--offering an unprecedented balance of accuracy, scalability, and cost-effectiveness. This makes it highly accessible and practical for the academic community.

</details>

---

## 354. GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training

- [ ] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training | https://openaccess.thecvf.com/content/ICCV2025/html/Wei_GTR_Guided_Thought_Reinforcement_Prevents_Thought_Collapse_in_RL-based_VLM_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wei_GTR_Guided_Thought_Reinforcement_Prevents_Thought_Collapse_in_RL-based_VLM_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reinforcement learning with verifiable outcome rewards (RLVR) has effectively scaled up chain-of-thought (CoT) reasoning in large language models (LLMs). Yet, its efficacy in training vision-language model (VLM) agents for goal-directed action reasoning in visual environments is less established. This work investigates this problem through extensive experiments on complex card games, such as 24 points, and embodied tasks from ALFWorld. We find that when rewards are based solely on action outcomes, RL fails to incentivize CoT reasoning in VLMs, instead leading to a phenomenon we termed thought collapse, characterized by a rapid loss of diversity in the agent's thoughts, state-irrelevant and incomplete reasoning, and subsequent invalid actions, resulting in negative rewards. To counteract thought collapse, we highlight the necessity of process guidance and propose an automated corrector that evaluates and refines the agent's reasoning at each RL step. This simple and scalable GTR (Guided Thought Reinforcement) framework trains reasoning and action simultaneously without the need for dense, per-step human labeling. Our experiments demonstrate that GTR significantly enhances the performance and generalization of the LLaVA-7B model across various visual environments, achieving 3-5 times higher task success rates compared to SoTA models with notably smaller model sizes.

</details>

---

## 355. HQ-CLIP: Leveraging Large Vision-Language Models to Create High-Quality Image-Text Datasets and CLIP Models

- [ ] HQ-CLIP: Leveraging Large Vision-Language Models to Create High-Quality Image-Text Datasets and CLIP Models | https://openaccess.thecvf.com/content/ICCV2025/html/Wei_HQ-CLIP_Leveraging_Large_Vision-Language_Models_to_Create_High-Quality_Image-Text_Datasets_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wei_HQ-CLIP_Leveraging_Large_Vision-Language_Models_to_Create_High-Quality_Image-Text_Datasets_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large-scale but noisy image-text pair data have paved the way for the success of Contrastive Language-Image Pretraining (CLIP). As the foundation vision encoder, CLIP in turn serves as the cornerstone for most large vision-language models (LVLMs). This interdependence naturally raises an interesting question: Can we reciprocally leverage LVLMs to enhance the quality of image-text pair data, thereby opening the possibility of a self-reinforcing cycle for continuous improvement? In this work, we take a significant step toward this vision by introducing an LVLM-driven data refinement pipeline. Our framework leverages LVLMs to process images and their raw alt-text, generating four complementary textual formulas: long positive descriptions, long negative descriptions, short positive tags, and short negative tags. Applying this pipeline to the curated DFN-Large dataset yields VLM-150M, a refined dataset enriched with multi-grained annotations. Based on this dataset, we further propose a training paradigm that extends conventional contrastive learning by incorporating negative descriptions and short tags as additional supervised signals. The resulting model, namely HQ-CLIP, demonstrates remarkable improvements across diverse benchmarks. Within a comparable training data scale, our approach achieves state-of-the-art performance in zero-shot classification, cross-modal retrieval, and fine-grained visual understanding tasks. In retrieval benchmarks, HQ-CLIP even surpasses standard CLIP models trained on the DFN-2B dataset, which contains 10xmore training data than ours. All code, data, and models will be made publicly available to support further research.

</details>

---

## 356. InstructSeg: Unifying Instructed Visual Segmentation with Multi-modal Large Language Models

- [ ] InstructSeg: Unifying Instructed Visual Segmentation with Multi-modal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Wei_InstructSeg_Unifying_Instructed_Visual_Segmentation_with_Multi-modal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wei_InstructSeg_Unifying_Instructed_Visual_Segmentation_with_Multi-modal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Boosted by Multi-modal Large Language Models (MLLMs), text-guided universal segmentation models for the image and video domains have made rapid progress recently. However, these methods are often developed separately for specific domains, overlooking the similarities in task settings and solutions across these two areas. In this paper, we define the union of referring segmentation and reasoning segmentation at both the image and video levels as Instructed Visual Segmentation (IVS). Correspondingly, we propose InstructSeg, an end-to-end segmentation pipeline equipped with MLLMs for IVS. Specifically, we employ an object-aware video perceiver to extract temporal and object information from reference frames, facilitating comprehensive video understanding. Additionally, we introduce vision-guided multi-granularity text fusion to better integrate global and detailed text information with fine-grained visual guidance. By leveraging multi-task and end-to-end training, InstructSeg demonstrates superior performance across diverse image and video segmentation tasks, surpassing both segmentation specialists and MLLM-based methods with a single model.

</details>

---

## 357. Passing the Driving Knowledge Test

- [ ] Passing the Driving Knowledge Test | https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Passing_the_Driving_Knowledge_Test_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Passing_the_Driving_Knowledge_Test_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

If a Large Language Model (LLM) were to take a driving knowledge test today, would it pass? Beyond standard spatial and visual question-answering (QA) tasks on current autonomous driving benchmarks, driving knowledge tests require a complete understanding of all traffic rules, signage, and right-of-way principles. To pass this test, human drivers must discern various edge cases that rarely appear in real-world datasets. In this work, we present DriveQA, an extensive open-source text and vision-based benchmark that exhaustively covers traffic regulations and scenarios. Through our experiments using DriveQA, we show that (1) state-of-the-art LLMs and Multimodal LLMs (MLLMs) perform well on basic traffic rules but exhibit significant weaknesses in numerical reasoning and complex right-of-way scenarios, traffic sign variations, and spatial layouts, (2) fine-tuning on DriveQA improves accuracy across multiple categories, particularly in regulatory sign recognition and intersection decision-making, (3) controlled variations in DriveQA-V provide insights into model sensitivity to environmental factors such as lighting, perspective, distance, and weather conditions, and (4) pretraining on DriveQA enhances downstream driving task performance, leading to improved results on real-world datasets such as nuScenes and BDD, while also demonstrating that models can internalize text and synthetic traffic knowledge to generalize effectively across downstream QA tasks.

</details>

---

## 358. VisNumBench: Evaluating Number Sense of Multimodal Large Language Models

- [ ] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Weng_VisNumBench_Evaluating_Number_Sense_of_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Weng_VisNumBench_Evaluating_Number_Sense_of_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Can Multimodal Large Language Models (MLLMs) develop an intuitive number sense similar to humans? Targeting this problem, we introduce Visual Number Benchmark (VisNumBench) to evaluate the number sense abilities of MLLMs across a wide range of visual numerical tasks. VisNumBench consists of about 1,900 multiple-choice question-answer pairs derived from both synthetic and real-world visual data, covering seven visual numerical attributes and four types of visual numerical estimation tasks. Our experiments on VisNumBench led to the following key findings:(i) The 17 MLLMs we tested--including open-source models such as Qwen2.5-VL and InternVL2.5, as well as proprietary models like GPT-4o and Gemini 2.0 Flash--perform significantly below human levels in number sense-related tasks. (ii) Multimodal mathematical models and multimodal chain-of-thought (CoT) models did not exhibit significant improvements in number sense abilities. (iii) Stronger MLLMs with larger parameter sizes and broader general abilities demonstrate modest gains in number sense abilities. We believe VisNumBench will serve as a valuable resource for the research community, encouraging further advancements in enhancing MLLMs' number sense abilities.

</details>

---

## 359. FDPT: Federated Discrete Prompt Tuning for Black-Box Visual-Language Models

- [ ] FDPT: Federated Discrete Prompt Tuning for Black-Box Visual-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Wu_FDPT_Federated_Discrete_Prompt_Tuning_for_Black-Box_Visual-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wu_FDPT_Federated_Discrete_Prompt_Tuning_for_Black-Box_Visual-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

General-purpose Vision-Language Models (VLMs) have driven major advancements in multimodal AI. Fine-tuning these models with task-specific data enhances adaptability to various downstream tasks but suffers from privacy risks. While potential solutions like federated learning can address user data privacy concerns, model protection is also essential. Other methods that rely on black-box VLM APIs usually require the access of prediction logits, leaving them open to inversion attacks. Moreover, addressing the challenges of tuning complexity and data transmission efficiency in federated VLM scenarios is also crucial. To address these challenges, we propose FDPT--a federated discrete prompt tuning method utilizing black-box VLMs. During client optimization stage, FDPT employs an agent-driven framework leveraging large language models (LLMs) with enhanced reasoning capacities to systematically optimize discrete prompt representations, and also utilizes feedback mechanisms and chain of thought to enhance prediction accuracy. Importantly, it performs optimization by relying not on the predicted logic vectors output by LLMs but on textual results, avoiding reverse attack risks. During global aggregation stage, We mimic human electoral activities by employing evolutionary computation methods underpinned by semantic similarity computation to implement enhanced zero-order optimization for acquiring representative global tokens, thereby achieving knowledge aggregation. FDPT significantly outperforms nine state-of-the-art methods in image classification and visual question-answering, reducing communication overhead while generating highly transferable optimized prompts. Additionally, it exhibits improved robustness to data heterogeneity.

</details>

---

## 360. Hierarchical Variational Test-Time Prompt Generation for Zero-Shot Generalization

- [ ] Hierarchical Variational Test-Time Prompt Generation for Zero-Shot Generalization | https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Hierarchical_Variational_Test-Time_Prompt_Generation_for_Zero-Shot_Generalization_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Hierarchical_Variational_Test-Time_Prompt_Generation_for_Zero-Shot_Generalization_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models like CLIP have demonstrated strong zero-shot generalization, making them valuable for various downstream tasks through prompt learning. However, existing test-time prompt tuning methods, such as entropy minimization, treat both text and visual prompts as fixed learnable parameters, limiting their adaptability to unseen domains. In contrast, we propose Hierarchical Variational Test-Time Prompt Generation, a novel approach where both text and visual prompts are dynamically generated via a HyperTransformer at inference time. This enables the model to produce data-specific prompts for each modality, significantly improving generalization. To further address template sensitivity and distribution shifts, we introduce variational prompt generation, leveraging variational inference to mitigate biases introduced by different prompt templates and data augmentations. Additionally, our hierarchical variational prompt generation conditions prompts at each layer on those from previous layers, allowing the model to capture deeper contextual dependencies and refine prompt interactions for robust adaptation. Extensive experiments on domain generalization benchmarks demonstrate that our method significantly outperforms existing prompt-learning techniques, achieving state-of-the-art zero-shot accuracy while maintaining efficiency.

</details>

---

## 361. MUNBa: Machine Unlearning via Nash Bargaining

- [ ] MUNBa: Machine Unlearning via Nash Bargaining | https://openaccess.thecvf.com/content/ICCV2025/html/Wu_MUNBa_Machine_Unlearning_via_Nash_Bargaining_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wu_MUNBa_Machine_Unlearning_via_Nash_Bargaining_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions.To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions.To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point.Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective.We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation.Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving.Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.

</details>

---

## 362. Mixture-of-Scores: Robust Image-Text Data Valuation via Three Lines of Code

- [ ] Mixture-of-Scores: Robust Image-Text Data Valuation via Three Lines of Code | https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Mixture-of-Scores_Robust_Image-Text_Data_Valuation_via_Three_Lines_of_Code_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Mixture-of-Scores_Robust_Image-Text_Data_Valuation_via_Three_Lines_of_Code_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Evaluating the quality of image-text pairs is essential for data processing in vision-language pre-training. Most metrics currently use off-the-shelf models, like CLIP-Score, to score pairs based on feature similarity. However, we find that different scoring models often produce inconsistent quality scores for the same data. This disparity impacts data processing results, leading to variations in datasets and, consequently, in model performance when trained on these datasets. Notably, no single quality score excels across all tasks, as each has biases toward specific concepts, resulting in complementary effects on model performance. This complicates the selection of scoring models. In this paper, we analyze these disparities and propose a method called Mixture-of-Scores (MoS). This approach integrates various quality scores into a robust ensemble score, effectively mitigating biases. It can be implemented easily in just three lines of code. Our extensive experiments show that MoS outperforms existing single quality scores across multiple vision-language tasks and benchmarks. We aim to offer new insights and practical tools to help the community navigate the challenges of scoring model selection.

</details>

---

## 363. RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping

- [ ] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping | https://openaccess.thecvf.com/content/ICCV2025/html/Wu_RAGNet_Large-scale_Reasoning-based_Affordance_Segmentation_Benchmark_towards_General_Grasping_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wu_RAGNet_Large-scale_Reasoning-based_Affordance_Segmentation_Benchmark_towards_General_Grasping_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

General robotic grasping systems require accurate object affordance perception in diverse open-world scenarios following human instructions. However, current studies suffer from the problem of lacking reasoning-based large-scale affordance prediction data, leading to considerable concern about open-world effectiveness. To address this limitation, we build a large-scale grasping-oriented affordance segmentation benchmark with human-like instructions, named RAGNet. It contains 273k images, 180 categories, and 26k reasoning instructions. The images cover diverse embodied data domains, such as wild, robot, ego-centric, and even simulation data. They are carefully annotated with an affordance map, while the difficulty of language instructions is largely increased by removing their category name and only providing functional descriptions. Furthermore, we propose a comprehensive affordance-based grasping framework, named AffordanceNet, which consists of a VLM pre-trained on our massive affordance data and a grasping network that conditions an affordance map to grasp the target. Extensive experiments on affordance segmentation benchmarks and real-robot manipulation tasks show that our model has a powerful open-world generalization ability. Our data and code is available at this link.

</details>

---

## 364. VSP: Diagnosing the Dual Challenges of Perception and Reasoning in Spatial Planning Tasks for MLLMs

- [ ] VSP: Diagnosing the Dual Challenges of Perception and Reasoning in Spatial Planning Tasks for MLLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Wu_VSP_Diagnosing_the_Dual_Challenges_of_Perception_and_Reasoning_in_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wu_VSP_Diagnosing_the_Dual_Challenges_of_Perception_and_Reasoning_in_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models are an exciting emerging class of language models (LMs) that have merged classic LM capabilities with those of image processing systems. However, how these capabilities integrate is often not intuitive and warrants direct investigation. One understudied capability in MLLMs is visual spatial planning---the ability to comprehend the spatial arrangements of objects and devise action plans to achieve desired outcomes in visual scenes. It is unclear why MLLMs fall short on these tasks generally considered easy for humans, given their successes across other diverse scenarios. To this end, we introduce VSP, a benchmark that 1) evaluates the spatial planning capability in MLLMs in general, and 2) diagnoses this capability via finer-grained sub-tasks, including perception and reasoning, and measure the capabilities of models through these sub-tasks. Our evaluation confirms that both open-source and private MLLMs fail to generate effective plans for even simple spatial planning tasks. Evaluations on the fine-grained analytical tasks further reveal fundamental deficiencies in the models' visual perception and bottlenecks in reasoning abilities, explaining their worse performance in the general spatial planning tasks. Our work illuminates future directions for improving MLLMs' abilities in spatial planning. Our benchmark is publicly available.

</details>

---

## 365. Visual Textualization for Image Prompted Object Detection

- [ ] Visual Textualization for Image Prompted Object Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Visual_Textualization_for_Image_Prompted_Object_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Visual_Textualization_for_Image_Prompted_Object_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We propose VisTex-OVLM, a novel image prompted object detection method that introduces visual textualization ---- a process that projects a few visual exemplars into the text feature space to enhance Object-level Vision-Language Models' (OVLMs) capability in detecting rare categories that are difficult to describe textually and nearly absent from their pre-training data, while preserving their pre-trained object-text alignment. Specifically, VisTex-OVLM leverages multi-scale textualizing blocks and a multi-stage fusion strategy to integrate visual information from visual exemplars, generating textualized visual tokens that effectively guide OVLMs alongside text prompts. Unlike previous methods, our method maintains the original architecture of OVLM, maintaining its generalization capabilities while enhancing performance in few-shot settings. VisTex-OVLM demonstrates superior performance across open-set datasets which have minimal overlap with OVLM's pre-training data and achieves state-of-the-art results on few-shot benchmarks PASCAL VOC and MSCOCO . The code will be released at VisTex-OVLM.

</details>

---

## 366. Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation

- [ ] Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation | https://openaccess.thecvf.com/content/ICCV2025/html/Xia_Bootstrapping_Grounded_Chain-of-Thought_in_Multimodal_LLMs_for_Data-Efficient_Model_Adaptation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xia_Bootstrapping_Grounded_Chain-of-Thought_in_Multimodal_LLMs_for_Data-Efficient_Model_Adaptation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in interpreting images using natural language. However, without using large-scale datasets for retraining, these models are difficult to adapt to specialized vision tasks, e.g., chart understanding. This problem is caused by a mismatch between pre-training and downstream datasets: pre-training datasets primarily concentrate on scenes and objects but contain limited information about specialized, non-object images, such as charts and tables. In this paper, we share an interesting finding that training an MLLM with chain-of-thought (CoT) reasoning data can facilitate model adaptation in specialized vision tasks, especially under data-limited regimes. However, we identify a critical issue within CoT data distilled from pre-trained MLLMs, i.e., the data often contains multiple factual errors in the reasoning steps. To address the problem, we propose Grounded Chain-of-Thought (GCoT), a simple bootstrapping-based approach that aims to inject grounding information (i.e., bounding boxes) into CoT data, essentially making the reasoning steps more faithful to input images. We evaluate our approach on five specialized vision tasks, which cover a variety of visual formats including charts, tables, receipts, and reports. The results demonstrate that under data-limited regimes our approach significantly improves upon fine-tuning and distillation.

</details>

---

## 367. Exploring The Visual Feature Space for Multimodal Neural Decoding

- [ ] Exploring The Visual Feature Space for Multimodal Neural Decoding | https://openaccess.thecvf.com/content/ICCV2025/html/Xia_Exploring_The_Visual_Feature_Space_for_Multimodal_Neural_Decoding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xia_Exploring_The_Visual_Feature_Space_for_Multimodal_Neural_Decoding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The intrication of brain signals drives research that leverages multimodal AI to align brain modalities with visual and textual data for explainable descriptions. However, most existing studies are limited to coarse interpretations, lacking essential details on object descriptions, locations, attributes, and their relationships. This leads to imprecise and ambiguous reconstructions when using such cues for visual decoding. To address this, we analyze different choices of vision feature spaces from pre-trained visual components within Multimodal Large Language Models (MLLMs) and introduce a zero-shot multimodal brain decoding method that interacts with these models to decode across multiple levels of granularities. To assess a model's ability to decode fine details from brain signals, we propose the Multi-Granularity Brain Detail Understanding Benchmark (MG-BrainDub). This benchmark includes two key tasks: detailed descriptions and salient question-answering, with metrics highlighting key visual elements like objects, attributes, and relationships. Our approach enhances neural decoding precision and supports more accurate neuro-decoding applications.

</details>

---

## 368. Advancing Visual Large Language Model for Multi-granular Versatile Perception

- [ ] Advancing Visual Large Language Model for Multi-granular Versatile Perception | https://openaccess.thecvf.com/content/ICCV2025/html/Xiang_Advancing_Visual_Large_Language_Model_for_Multi-granular_Versatile_Perception_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xiang_Advancing_Visual_Large_Language_Model_for_Multi-granular_Versatile_Perception_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Perception is a fundamental task in the field of computer vision, encompassing a diverse set of subtasks that can be systematically categorized into four distinct groups based on two dimensions: prediction type and instruction type. Notably, existing researches often focus solely on a limited subset of these potential combinations, which constrains their applicability and versatility across various contexts. In response to this challenge, we present MVL-LM, a Multi-granular and Versatile Perception framework incorporating Visual Large Language Model. Our framework is designed to integrate both word-based and sentence-based perception tasks alongside box and mask predictions within a single architecture. MVL-LM features an innovative multi-granularity decoder in conjunction with a CoT-inspired dataset unification strategy, enabling seamless supervised fine-tuning across a wide spectrum of tasks, including but not limited to panoptic segmentation, detection, grounding, and referring expression segmentation. Furthermore, we introduce a query enhancement strategy aimed at harnessing the decoding and generative capabilities inherent in VLLMs. Extensive experiments conducted across a range of benchmarks in both word-based and sentence-based perception tasks substantiate the efficacy of our framework. The code will be available at https://github.com/xiangwentao666/MVP-LM.

</details>

---

## 369. MIEB: Massive Image Embedding Benchmark

- [ ] MIEB: Massive Image Embedding Benchmark | https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_MIEB_Massive_Image_Embedding_Benchmark_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_MIEB_Massive_Image_Embedding_Benchmark_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image representations are often evaluated through disjointed, task-specific protocols, leading to a fragmented understanding of model capabilities. For instance, it is unclear whether an image embedding model adept at clustering images is equally good at retrieving relevant images given a piece of text. We introduce the Massive Image Embedding Benchmark (MIEB) to evaluate the performance of image and image-text embedding models across the broadest spectrum to date. MIEB spans 38 languages across 130 individual tasks, which we group into 8 high-level categories. We benchmark 50 models across our benchmark, finding that no single method dominates across all task categories. We reveal hidden capabilities in advanced vision models such as their accurate visual representation of texts, and their yet limited capabilities in interleaved encodings and matching images and texts in the presence of confounders. We also show that the performance of vision encoders on MIEB correlates highly with their performance when used in multimodal large language models. Our code, dataset, and leaderboard are publicly available at https://github.com/embeddings-benchmark/mteb.

</details>

---

## 370. RoboTron-Sim: Improving Real-World Driving via Simulated Hard-Case

- [ ] RoboTron-Sim: Improving Real-World Driving via Simulated Hard-Case | https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_RoboTron-Sim_Improving_Real-World_Driving_via_Simulated_Hard-Case_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_RoboTron-Sim_Improving_Real-World_Driving_via_Simulated_Hard-Case_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Collecting real-world data for rare high-risk scenarios, long-tailed driving events, and complex interactions remains challenging, leading to poor performance of existing autonomous driving systems in these critical situations. In this paper, we propose RoboTron-Sim that improves real-world driving in critical situations by utilizing simulated hard cases. First, we develop a simulated dataset called Hard-case Augmented Synthetic Scenarios (HASS), which covers 13 high-risk edge-case categories, as well as balanced environmental conditions such as day/night and sunny/rainy. Second, we introduce Scenario-aware Prompt Engineering (SPE) and an Image-to-Ego Encoder (I2E Encoder) to enable multimodal large language models to effectively learn real-world challenging driving skills from HASS, via adapting to environmental deviations and hardware differences between real-world and simulated scenarios. Extensive experiments on nuScenes show that RoboTron-Sim improves driving performance in challenging scenarios by  50%, achieving state-of-the-art results in real-world open-loop planning. Qualitative results further demonstrate the effectiveness of RoboTron-Sim in better managing rare high-risk driving scenarios.

</details>

---

## 371. VideoAuteur: Towards Long Narrative Video Generation

- [ ] VideoAuteur: Towards Long Narrative Video Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_VideoAuteur_Towards_Long_Narrative_Video_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_VideoAuteur_Towards_Long_Narrative_Video_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent video generation models have shown promising results in producing high-quality video clips lasting several seconds. However, these models face challenges in generating long sequences that convey clear and informative events, limiting their ability to support coherent narrations. In this paper, we present a large-scale cooking video dataset designed to advance long-form narrative generation in the cooking domain. We validate the quality of our proposed dataset in terms of visual fidelity and textual caption accuracy using state-of-the-art Vision-Language Models (VLMs) and video generation models, respectively. We further introduce a Long Narrative Video Director to enhance both visual and semantic coherence in generated videos and emphasize the role of aligning visual embeddings to achieve improved overall video quality. Our method demonstrates substantial improvements in generating visually detailed and semantically aligned keyframes, supported by finetuning techniques that integrate text and image embeddings within the video generation process. Codes and data will be made publicly available.

</details>

---

## 372. Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data and Metric Perspectives

- [ ] Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data and Metric Perspectives | https://openaccess.thecvf.com/content/ICCV2025/html/Xie_Are_VLMs_Ready_for_Autonomous_Driving_An_Empirical_Study_from_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xie_Are_VLMs_Ready_for_Autonomous_Driving_An_Empirical_Study_from_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language Models (VLMs) have fueled interest in autonomous driving applications, particularly for interpretable decision-making. However, the assumption that VLMs provide visually grounded and reliable driving explanations remains unexamined. To address this, we introduce DriveBench, a benchmark evaluating 12 VLMs across 17 settings, covering 19,200 images, 20,498 QA pairs, and four key driving tasks. Our findings reveal that VLMs often generate plausible responses from general knowledge or textual cues rather than true visual grounding, especially under degraded or missing visual inputs. This behavior, concealed by dataset imbalances and insufficient evaluation metrics, poses significant risks in safety-critical scenarios like autonomous driving. We further observe that VLMs possess inherent corruption-awareness but only explicitly acknowledge these issues when directly prompted. Given the challenges and inspired by the inherent corruption awareness, we propose Robust Agentic Utilization (RAU), leveraging VLMs' corruption awareness and agentic planning with external tools to enhance perception reliability for downstream tasks. Our study challenges existing evaluation paradigms and provides a roadmap toward more robust and interpretable autonomous driving systems.

</details>

---

## 373. MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding

- [ ] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding | https://openaccess.thecvf.com/content/ICCV2025/html/Xie_MUSE-VL_Modeling_Unified_VLM_through_Semantic_Discrete_Encoding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xie_MUSE-VL_Modeling_Unified_VLM_through_Semantic_Discrete_Encoding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce MUSE-VL, a Unified Vision-Language Model through Semantic discrete Encoding for multimodal understanding and generation. Recently, the research community has begun exploring unified models for visual generation and understanding. However, existing vision tokenizers (e.g., VQGAN) only consider low-level information, which makes it difficult to align with language tokens. This results in high training complexity and necessitates a large amount of training data to achieve optimal performance.Additionally, their performance is still far from dedicated understanding models. This paper proposes Semantic Discrete Encoding (SDE), which effectively aligns the information of visual tokens and language tokens by adding semantic constraints to the visual tokenizer. This greatly reduces the amount of training data and improves the performance of the unified model. With the same LLM size, our method improved the understanding performance by 4.8% compared to the previous SOTA Emu3 and surpassed the dedicated understanding model LLaVA-NeXT 34B by 3.7%. For visual generation, our model achieves a FID score of 7.73 on MJHQ-30k, surpassing the existing unified models.

</details>

---

## 374. MVGBench: a Comprehensive Benchmark for Multi-view Generation Models

- [ ] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models | https://openaccess.thecvf.com/content/ICCV2025/html/Xie_MVGBench_a_Comprehensive_Benchmark_for_Multi-view_Generation_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xie_MVGBench_a_Comprehensive_Benchmark_for_Multi-view_Generation_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We propose MVGBench, a comprehensive benchmark for multi-view image generation models (MVGs) that evaluates 3D consistency in geometry and texture, image quality, and semantics (using vision language models). Recently, MVGs have been the main driving force in 3D object creation. However, existing metrics compare generated images against ground truth target views, which is not suitable for generative tasks where multiple solutions exist while differing from ground truth. Furthermore, different MVGs are trained on different view angles, synthetic data and specific lightings robustness to these factors and generalization to real data are rarely evaluated thoroughly. Without a rigorous evaluation protocol, it is also unclear what design choices contribute to the progress of MVGs. MVGBench evaluates three different aspects: best setup performance, generalization to real data and robustness. Instead of comparing against ground truth, we introduce a novel 3D self-consistency metric which compares 3D reconstructions from disjoint generated multi-views. We systematically compare 12 existing MVGs on 4 different curated real and synthetic datasets. With our analysis, we identify important limitations of existing methods specially in terms of robustness and generalization, and we find the most critical design choices. Using the discovered best practices, we propose ViFiGen, a method that outperforms all evaluated MVGs on 3D consistency. Our benchmark suite and pretrained models are released.

</details>

---

## 375. Region-based Cluster Discrimination for Visual Representation Learning

- [ ] Region-based Cluster Discrimination for Visual Representation Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Xie_Region-based_Cluster_Discrimination_for_Visual_Representation_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xie_Region-based_Cluster_Discrimination_for_Visual_Representation_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning visual representations is foundational for a broad spectrum of downstream tasks. Although recent vision-language contrastive models, such as CLIP and SigLIP, have achieved impressive zero-shot performance via large-scale vision-language alignment, their reliance on global representations constrains their effectiveness for dense prediction tasks, such as grounding, OCR, and segmentation. To address this gap, we introduce Region-Aware Cluster Discrimination (RICE), a novel method that enhances region-level visual and OCR capabilities. We first construct a billion-scale candidate region dataset and propose a Region Transformer layer to extract rich regional semantics. We further design a unified region cluster discrimination loss that jointly supports object and OCR learning within a single classification framework, enabling efficient and scalable distributed training on large-scale data. Extensive experiments show that RICE consistently outperforms previous methods on tasks, including segmentation, dense detection, and visual perception for Multimodal Large Language Models (MLLMs). The pre-trained models have been released at https://github.com/deepglint/MVT.

</details>

---

## 376. Shot-by-Shot: Film-Grammar-Aware Training-Free Audio Description Generation

- [ ] Shot-by-Shot: Film-Grammar-Aware Training-Free Audio Description Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Xie_Shot-by-Shot_Film-Grammar-Aware_Training-Free_Audio_Description_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xie_Shot-by-Shot_Film-Grammar-Aware_Training-Free_Audio_Description_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Our objective is the automatic generation of Audio Descriptions (ADs) for edited video material, such as movies and TV series. To achieve this, we propose a two-stage framework that leverages "shots" as the fundamental units of video understanding. This includes extending temporal context to neighboring shots and incorporating film grammar devices, such as shot scales and thread structures, to guide AD generation. Our method is compatible with both open-source and proprietary Visual-Language Models (VLMs), integrating expert knowledge from add-on modules without requiring additional training of the VLMs. We achieve state-of-the-art performance among all prior training-free approaches and even surpass fine-tuned methods on several benchmarks. To evaluate the quality of predicted ADs, we introduce a new evaluation measure -- an action score -- specifically targeted to assessing this important aspect of AD. Additionally, we propose a novel evaluation protocol that treats automatic frameworks as AD generation assistants and asks them to generate multiple candidate ADs for selection.

</details>

---

## 377. AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction

- [ ] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Xing_AID_Adapting_Image2Video_Diffusion_Models_for_Instruction-guided_Video_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xing_AID_Adapting_Image2Video_Diffusion_Models_for_Instruction-guided_Video_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-guided video prediction (TVP) involves predicting the motion of future frames from the initial frame according to an instruction, which has wide applications in virtual reality, robotics, and content creation. Previous TVP methods make significant breakthroughs by adapting Stable Diffusion for this task. However, they struggle with frame consistency and temporal stability primarily due to the limited scale of video datasets. We observe that pretrained Image2Video diffusion models possess good video dynamics priors but lack fine-grained textual control. Hence, transferring pretrained models to leverage their video dynamic priors while injecting fine-grained control to generate controllable videos is both a meaningful and challenging task. To achieve this, we introduce the Multi-Modal Large Language Model (MLLM) to predict future video states based on initial frames and text instructions. More specifically, we design a dual query transformer (DQFormer) architecture, which integrates the instructions and frames into the conditional embeddings for future frame prediction. Additionally, we develop Temporal and Spatial Adapters that can quickly transfer general video diffusion models to specific scenarios with minimal training costs. Experimental results show that our method significantly outperforms state-of-the-art techniques on four datasets: Something Something V2, Epic Kitchen-100, Bridge Data, and UCF-101. Notably, AID achieves 91.2% and 55.5% FVD improvements on Bridge and SSv2 respectively, demonstrating its effectiveness in various domains.

</details>

---

## 378. Automated Red Teaming for Text-to-Image Models through Feedback-Guided Prompt Iteration with Vision-Language Models

- [ ] Automated Red Teaming for Text-to-Image Models through Feedback-Guided Prompt Iteration with Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Automated_Red_Teaming_for_Text-to-Image_Models_through_Feedback-Guided_Prompt_Iteration_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Automated_Red_Teaming_for_Text-to-Image_Models_through_Feedback-Guided_Prompt_Iteration_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-image models have achieved remarkable progress in generating high-quality images from textual prompts, yet their potential for misuse like generating unsafe content remains a critical concern. Existing safety mechanisms, such as filtering and fine-tuning, remain insufficient in preventing vulnerabilities exposed by adversarial prompts. To systematically evaluate these weaknesses, we propose an automated red-teaming framework, Feedback-Guided Prompt Iteration (FGPI), which utilizes a Vision-Language Model (VLM) as the red-teaming agent following a feedback-guide-rewrite paradigm for iterative prompt optimization. The red-teaming VLM analyzes prompt-image pairs based on evaluation results, provides feedback and modification strategies to enhance adversarial effectiveness while preserving safety constraints, and iteratively improves prompts. To enable this functionality, we construct a multi-turn conversational VQA dataset with over 6,000 instances, covering seven attack types and facilitating the fine-tuning of the red-teaming VLM. Extensive experiments demonstrate the effectiveness of our approach, achieving over 90% attack success rate within five iterations while maintaining prompt stealthiness and safety. The experiments also validate the adaptability, diversity, transferability, and explainability of FGPI. The source code and dataset are available at https://github.com/Weiww-Xu/FGPI.

</details>

---

## 379. Bringing RNNs Back to Efficient Open-Ended Video Understanding

- [ ] Bringing RNNs Back to Efficient Open-Ended Video Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Bringing_RNNs_Back_to_Efficient_Open-Ended_Video_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Bringing_RNNs_Back_to_Efficient_Open-Ended_Video_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The challenge of long video understanding lies in its high computational complexity and prohibitive memory cost, since the memory and computation required by transformer-based LLMs scale quadratically with input sequence length. We propose AuroraLong to address this challenge by replacing the LLM component in MLLMs with a linear RNN language model that handles input sequence of arbitrary length with constant-size hidden states. To further increase throughput and efficiency, we combine visual token merge with linear RNN models by reordering the visual tokens by their sizes in ascending order. Despite having only 2B parameters and being trained exclusively on public data, AuroraLong achieves performance comparable to Transformer-based models of similar size trained on private datasets across multiple video benchmarks. This demonstrates the potential of efficient, linear RNNs to democratize long video understanding by lowering its computational entry barrier. To our best knowledge, we are the first to use a linear RNN based LLM backbone in a LLaVA-like model for open-ended video understanding.

</details>

---

## 380. ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning

- [ ] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_ChartPoint_Guiding_MLLMs_with_Grounding_Reflection_for_Chart_Reasoning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_ChartPoint_Guiding_MLLMs_with_Grounding_Reflection_for_Chart_Reasoning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have emerged as powerful tools for chart comprehension. However, they heavily rely on extracted content via OCR, which leads to numerical hallucinations when chart textual annotations are sparse. While existing methods focus on scaling instructions, they fail to address the fundamental challenge, i.e., reasoning with visual perception. In this paper, we identify a critical observation: MLLMs exhibit weak grounding in chart elements and proportional relationships, as evidenced by their inability to localize key positions to match their reasoning. To bridge this gap, we propose PointCoT, which integrates reflective interaction into chain-of-thought reasoning in charts. By prompting MLLMs to generate bounding boxes and re-render charts based on location annotations, we establish connections between textual reasoning steps and visual grounding regions. We further introduce an automated pipeline to construct ChartPoint-SFT-62k, a dataset featuring 19.2K high-quality chart samples with step-by-step CoT, bounding box, and re-rendered visualizations. Leveraging this data, we develop two instruction-tuned models, ChartPoint_Q2 and ChartPoint_Q2.5, which outperform state-of-the-art across several chart benchmarks, e.g., +5.04% on ChartBench.

</details>

---

## 381. LLaVA-CoT: Let Vision Language Models Reason Step-by-Step

- [ ] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_LLaVA-CoT_Let_Vision_Language_Models_Reason_Step-by-Step_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_LLaVA-CoT_Let_Vision_Language_Models_Reason_Step-by-Step_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models have demonstrated substantial advancements in reasoning capabilities. However, current Vision-Language Models (VLMs) often struggle to perform systematic and structured reasoning, especially when handling complex visual question-answering tasks. In this work, we introduce LLaVA-CoT, a large VLM designed to conduct autonomous multistage reasoning. Unlike chain-of-thought prompting, LLaVA-CoT independently engages in sequential stages of summarization, visual interpretation, logical reasoning, and conclusion generation. This structured approach enables LLaVA-CoT to achieve marked improvements on reasoning-intensive tasks. To accomplish this, we construct the LLaVA-CoT-100k dataset, integrating samples from various visual question answering sources and providing structured reasoning annotations. Besides, we propose a test-time stage-wise retracing search method (SWIRES), which enables effective and efficient test-time scaling. Remarkably, with only 100k training samples and test-time scaling, LLaVA-CoT not only outperforms its base model by 9.4% on a wide range of multimodal reasoning benchmarks, but also surpasses the performance of larger and even closed-source models, such as Gemini-1.5-pro, GPT-4o-mini, and Llama-3.2-90B-Vision-Instruct. The code, dataset, and pre-trained weights are publicly available at https://github.com/PKU-YuanGroup/LLaVA-CoT.

</details>

---

## 382. Learning to Inference Adaptively for Multimodal Large Language Models

- [ ] Learning to Inference Adaptively for Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Learning_to_Inference_Adaptively_for_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Learning_to_Inference_Adaptively_for_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown impressive capabilities in visual reasoning, yet come with substantial computational cost, limiting their deployment in resource-constrained settings. Despite recent effort on improving the efficiency of MLLMs, prior solutions fall short in responding to varying runtime conditions, in particular changing resource availability (e.g., contention due to the execution of other programs on the device). To bridge this gap, we introduce AdaLLaVA, an adaptive inference framework that learns to dynamically reconfigure operations in an MLLM during inference, accounting for the input data and a latency budget. We conduct extensive experiments across benchmarks involving question-answering, reasoning, and hallucination. Our results show that AdaLLaVA effectively adheres to input latency budget, achieving varying accuracy and latency tradeoffs at runtime. Further, we demonstrate that AdaLLaVA adapts to both input latency and content, can be integrated with token selection for enhanced efficiency, and generalizes across MLLMs.

</details>

---

## 383. MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs

- [ ] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_MC-Bench_A_Benchmark_for_Multi-Context_Visual_Grounding_in_the_Era_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_MC-Bench_A_Benchmark_for_Multi-Context_Visual_Grounding_in_the_Era_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While multimodal large language models (MLLMs) have demonstrated extraordinary vision-language understanding capabilities, their abilities to solve instance-level visual-language problems beyond a single image warrant further exploration. To assess these unproven abilities of MLLMs, this paper proposes a new visual grounding task called multi-context visual grounding, which aims to localize instances of interest across multiple images based on open-ended text prompts. In order to facilitate this research, we construct a new dataset MC-Bench that features 2K high-quality and manually annotated samples. Each sample consists of an instance-level labeled image pair and a corresponding text prompt that indicates the target instances in the images. These text prompts are highly open-ended and follow three distinct styles, covering 20 practical skills. We benchmark over 20 state-of-the-art MLLMs and foundation models with potential multi-context visual grounding capabilities, along with our developed simple yet effective agentic baseline and a finetuned baseline by multi-context instruction tuning. Our evaluation reveals a non-trivial performance gap between existing MLLMs and humans, along with some insightful observations that suggest potential future directions. We hope that MC-Bench and our empirical findings encourage the research community to further advance the untapped potentials of MLLMs in instance-level tasks, particularly in multi-image contexts. Project page: https://xuyunqiu.github.io/MC-Bench.

</details>

---

## 384. OURO: A Self-Bootstrapped Framework for Enhancing Multimodal Scene Understanding

- [ ] OURO: A Self-Bootstrapped Framework for Enhancing Multimodal Scene Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_OURO_A_Self-Bootstrapped_Framework_for_Enhancing_Multimodal_Scene_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_OURO_A_Self-Bootstrapped_Framework_for_Enhancing_Multimodal_Scene_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large models have made significant progress, yet fine-grained understanding of complex scenes remains a challenge. High-quality, large-scale vision-language datasets are essential for addressing this issue. However, existing methods often rely on labor-intensive manual annotations or closed-source models with optimal performance, making large-scale data collection costly. To overcome these limitations, we propose a self-bootstrapped training pipeline that leverages the model's own multimodal capabilities to recursively refine its understanding. By decomposing existing multimodal data into localized sub-regions and generating hierarchical scene descriptions and multi-faceted question-answer pairs, we construct a dataset based on 1.4M image-task instances. We further utilize this dataset to train the base model, significantly enhancing its ability to interpret complex visual scenes and perform various vision-related tasks. Our OURO model, fine-tuned on Qwen2-VL-7B-Instruct using LoRA, achieves substantial improvements over both the base model and similarly-sized counterparts across multiple multimodal benchmarks. Our self-bootstrapped training pipeline offers a novel paradigm for the continuous improvement of multimodal models. Code and datasets are available at https://github.com/tinnel123666888/OURO.git.

</details>

---

## 385. Perceiving and Acting in First-Person: A Dataset and Benchmark for Egocentric Human-Object-Human Interactions

- [ ] Perceiving and Acting in First-Person: A Dataset and Benchmark for Egocentric Human-Object-Human Interactions | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Perceiving_and_Acting_in_First-Person_A_Dataset_and_Benchmark_for_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Perceiving_and_Acting_in_First-Person_A_Dataset_and_Benchmark_for_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning action models from real-world human-centric interaction datasets is important towards building general-purpose intelligent assistants with efficiency. However, most existing datasets only offer specialist interaction category and ignore that AI assistants perceive and act based on first-person acquisition. We urge that both the generalist interaction knowledge and egocentric modality are indispensable. In this paper, we embed the manual-assisted task into a vision-language-action framework, where the assistant provides services to the instructor following egocentric vision and commands. With our hybrid RGB-MoCap system, pairs of assistants and instructors engage with multiple objects and the scene following GPT-generated scripts. Under this setting, we accomplish InterVLA, the first large-scale human-object-human interaction dataset with 11.4 hours and 1.2M frames of multimodal data, spanning 2 egocentric and 5 exocentric videos, accurate human/object motions and verbal commands. Furthermore, we establish novel benchmarks on egocentric human motion estimation, interaction synthesis, and interaction prediction with comprehensive analysis. We believe that our InterVLA testbed and the benchmarks will foster future works on building AI agents in the physical world.

</details>

---

## 386. Feature Decomposition-Recomposition in Large Vision-Language Model for Few-Shot Class-Incremental Learning

- [ ] Feature Decomposition-Recomposition in Large Vision-Language Model for Few-Shot Class-Incremental Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Xue_Feature_Decomposition-Recomposition_in_Large_Vision-Language_Model_for_Few-Shot_Class-Incremental_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Xue_Feature_Decomposition-Recomposition_in_Large_Vision-Language_Model_for_Few-Shot_Class-Incremental_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Few-Shot Class-Incremental Learning (FSCIL) focuses on incrementally learning novel classes using only a limited number of samples from novel classes, which faces dual challenges: catastrophic forgetting of previously learned classes and over-fitting to novel classes with few available samples. Recent advances in large pre-trained vision-language models (VLMs), such as CLIP, provide rich feature representations that generalize well across diverse classes. Therefore, freezing the pre-trained backbone and aggregating class features as prototypes becomes an intuitive and effective way to mitigate catastrophic forgetting.However, this strategy fails to address the overfitting challenge, and the prototypes of novel classes exhibit semantic bias due to the few samples per class. To address these limitations, we propose a semantic Feature Decomposition-Recomposition (FDR)  method based on VLMs. Firstly, we decompose the CLIP features into semantically distinct segments guided by text keywords from base classes. Then, these segments are adaptively recomposed at the attribute level given text descriptions, forming calibrated prototypes for novel classes. The recomposition process operates linearly at the attribute level but induces nonlinear adjustments across the entire prototype. This fine-grained and non-linear recomposition inherits the generalization capabilities of VLMs and the adaptive recomposition ability of base classes, leading to enhanced performance in FSCIL. Extensive experiments demonstrate our method's effectiveness, particularly in 1-shot scenarios where it achieves improvements between 6.70% and 19.66% for novel classes over state-of-the-art baselines on CUB200.

</details>

---

## 387. Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology

- [ ] Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology | https://openaccess.thecvf.com/content/ICCV2025/html/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of vision-language models has transformed medical AI, enabling unprecedented advances in diagnostic capability and clinical applications. However, progress in dermatology has lagged behind other medical domains due to the lack of standard image-text pairs. Existing dermatological datasets are limited in both scale and depth, offering only single-label annotations across a narrow range of diseases instead of rich textual descriptions, and lacking the crucial clinical context needed for real-world applications. To address these limitations, we present Derm1M, the first large-scale vision-language dataset for dermatology, comprising 1,029,761 image-text pairs. Built from diverse educational resources and structured around a standard ontology collaboratively developed by experts, Derm1M provides comprehensive coverage for over 390 skin conditions across four hierarchical levels and 130 clinical concepts with rich contextual information such as medical history, symptoms, and skin tone. To demonstrate Derm1M's potential in advancing both AI research and clinical application, we pretrained a series of CLIP-like models, collectively called DermLIP, on this dataset. The DermLIP family significantly outperforms state-of-the-art foundation models on eight diverse datasets across multiple tasks, including zero-shot skin disease classification, clinical and artifacts concept identification, few-shot/full-shot learning, and cross-modal retrieval. Our dataset and code are available at https://github.com/SiyuanYan1/Derm1M.

</details>

---

## 388. AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning

- [ ] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_AR-VRM_Imitating_Human_Motions_for_Visual_Robot_Manipulation_with_Analogical_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_AR-VRM_Imitating_Human_Motions_for_Visual_Robot_Manipulation_with_Analogical_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Robot Manipulation (VRM) aims to enable a robot to follow natural language instructions based on robot states and visual observations, and therefore requires costly multi- modal data. To compensate for the deficiency of robot data, existing approaches have employed vision-language pre- training with large-scale data. However, they either utilize web data that differs from robotic tasks, or train the model in an implicit way (e.g., predicting future frames at the pixel level), thus showing limited generalization ability under in- sufficient robot data. In this paper, we propose to learn from large-scale human action video datasets in an explicit way (i.e., imitating human actions from hand keypoints), introduc- ing Visual Robot Manipulation with Analogical Reasoning (AR-VRM). To acquire action knowledge explicitly from hu- man action videos, we propose a keypoint Vision-Language Model (VLM) pretraining scheme, enabling the VLM to learn human action knowledge and directly predict human hand keypoints. During fine-tuning on robot data, to facilitate the robotic arm in imitating the action patterns of human motions, we first retrieve human action videos that perform similar manipulation tasks and have similar historical obser- vations, and then learn the Analogical Reasoning (AR) map between human hand keypoints and robot components. Tak- ing advantage of focusing on action keypoints instead of irrel- evant visual cues, our method achieves leading performance on the CALVIN benchmark and real-world experiments. In few-shot scenarios, our AR-VRM outperforms previous meth- ods by large margins, underscoring the effectiveness of explicitly imitating human actions under data scarcity. Code available at https://github.com/idejie/ar.

</details>

---

## 389. CLIPSym: Delving into Symmetry Detection with CLIP

- [ ] CLIPSym: Delving into Symmetry Detection with CLIP | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_CLIPSym_Delving_into_Symmetry_Detection_with_CLIP_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_CLIPSym_Delving_into_Symmetry_Detection_with_CLIP_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Symmetry is one of the most fundamental geometric cues in computer vision, and detecting it has been an ongoing challenge. With the recent advances in vision-language models, i.e., CLIP, we investigate whether a pre-trained CLIP model can aid symmetry detection by leveraging the additional symmetry cues found in the natural image descriptions. We propose CLIPSym, which leverages CLIP's image and language encoders and a rotation-equivariant decoder based on a hybrid of Transformer and G-Convolution to detect rotation and reflection symmetries. To fully utilize CLIP's language encoder, we have developed a novel prompting technique called Semantic-Aware Prompt Grouping (SAPG), which aggregates a diverse set of frequent object-based prompts to better integrate the semantic cues for symmetry detection. Empirically, we show that CLIPSym outperforms the current state-of-the-art on three standard symmetry detection datasets (DENDI, SDRW, and LDRS). Finally, we conduct detailed ablations verifying the benefits of CLIP's pre-training, the proposed equivariant decoder, and the SAPG technique.

</details>

---

## 390. Effective Training Data Synthesis for Improving MLLM Chart Understanding

- [ ] Effective Training Data Synthesis for Improving MLLM Chart Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Effective_Training_Data_Synthesis_for_Improving_MLLM_Chart_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Effective_Training_Data_Synthesis_for_Improving_MLLM_Chart_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Being able to effectively read scientific plots, or chart understanding, is a central part toward building effective agents for science. However, existing multimodal large language models (MLLMs), especially open-source ones, are still falling behind with a typical success rate of 30%-50% on challenging benchmarks. Previous studies on fine-tuning MLLMs with synthetic charts are often restricted by their inadequate similarity to the real charts, which could compromise model training and performance on complex real-world charts. In this study, we show that modularizing chart generation and diversifying visual details improves chart understanding capabilities. In particular, we design a five-step data synthesis pipeline, where we separate data and function creation for single plot generation, condition the generation of later subplots on earlier ones for multi-subplot figures, visually diversify the generated figures, filter out low quality data, and finally generate the question-answer (QA) pairs with GPT-4o. This approach allows us to streamline the generation of fine-tuning datasets and introduce the effective chart dataset (ECD), which contains 10k+ chart images and 300k+ QA pairs, covering 25 topics and featuring 250+ chart type combinations with high visual complexity. We show that ECD consistently improves the performance of various MLLMs on a range of real-world and synthetic test sets. Code, data and models are available at: https://github.com/yuweiyang-anu/ECD.

</details>

---

## 391. Medical World Model

- [ ] Medical World Model | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Medical_World_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Medical_World_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Providing effective treatment and making informed decisions are essential goals of modern medicine and clinical care.We are interested in simulating disease dynamics for clinical decision-making, leveraging recent advances in large generative models.To this end, we introduce the Medical World Model (MeWM), the first world model in medicine that predicts future disease states based on clinical decisions. MeWM comprises (i) vision-language models to serve as policy models, and (ii) tumor generative models as dynamics models. The policy model generates action plans, such as clinical treatments, while the dynamics model simulates tumor progression or regression under given treatment conditions. Building on this, we propose the inverse dynamics model that applies survival analysis to the simulated post-treatment tumor, enabling the evaluation of treatment efficacy and the selection of the optimal clinical action plan. As a result, the proposed MeWM simulates disease dynamics by synthesizing post-treatment tumors, with state-of-the-art specificity in Turing tests evaluated by radiologists. Simultaneously, its inverse dynamics model outperforms medical-specialized GPTs in optimizing individualized treatment protocols across all metrics.Notably, MeWM improves clinical decision-making for interventional physicians, boosting F1-score in selecting the optimal TACE protocol by 13%, paving the way for future integration of medical world models as the second readers.

</details>

---

## 392. R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization

- [ ] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_R1-Onevision_Advancing_Generalized_Multimodal_Reasoning_through_Cross-Modal_Formalization_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_R1-Onevision_Advancing_Generalized_Multimodal_Reasoning_through_Cross-Modal_Formalization_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models have demonstrated remarkable reasoning capability in complex textual tasks. However, multimodal reasoning, which requires integrating visual and textual information, remains a significant challenge. Existing visual-language models often struggle to effectively analyze and reason visual content, resulting in suboptimal performance on complex reasoning tasks. Moreover, the absence of comprehensive benchmarks hinders the accurate assessment of multimodal reasoning capabilities. In this paper, we introduce R1-Onevision, a multimodal reasoning model designed to bridge the gap between visual perception and deep reasoning. To achieve this, we propose a cross-modal reasoning pipeline that transforms images into formal textual representations, enabling precise language-based reasoning. Leveraging this pipeline, we construct the R1-Onevision dataset which provides detailed, step-by-step multimodal reasoning annotations across diverse domains. We further develop the R1-Onevision model through supervised fine-tuning and reinforcement learning to cultivate advanced reasoning and robust generalization abilities. To comprehensively evaluate multimodal reasoning performance across different grades, we introduce R1-Onevision-Bench, a benchmark aligned with human educational stages, covering exams from junior high school to university and beyond. Experimental results show that R1-Onevision achieves state-of-the-art performance, outperforming models such as GPT-4o and Qwen2.5-VL on multiple challenging multimodal reasoning benchmarks. Code, dataset and benchmark are available at https://github.com/Fancy-MLLM/R1-Onevision

</details>

---

## 393. VFlowOpt: A Token Pruning Framework for LMMs with Visual Information Flow-Guided Optimization

- [ ] VFlowOpt: A Token Pruning Framework for LMMs with Visual Information Flow-Guided Optimization | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VFlowOpt_A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VFlowOpt_A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs) excel in visual-language tasks by leveraging numerous visual tokens for fine-grained visual information, but this token redundancy results in significant computational costs. Previous research aimed at reducing visual tokens during inference typically leverages importance maps derived from attention scores among vision-only tokens or vision-language tokens to prune tokens across one or multiple pruning stages. Despite this progress, pruning frameworks and strategies remain simplistic and insufficiently explored, often resulting in substantial performance degradation. In this paper, we propose VFlowOpt, a token pruning framework that introduces an importance map derivation process and a progressive pruning module with a recycling mechanism. The hyperparameters of its pruning strategy are further optimized by a visual information flow-guided method. Specifically, we compute an importance map for image tokens based on their attention-derived context relevance and patch-level information entropy. We then decide which tokens to retain or prune and aggregate the pruned ones as recycled tokens to avoid potential information loss. Finally, we apply a visual information flow-guided method that regards the last token in the LMM as the most representative signal of text-visual interactions. This method minimizes the discrepancy between token representations in LMMs with and without pruning, thereby enabling superior pruning strategies tailored to different LMMs. Experiments demonstrate that VFlowOpt can prune 90% of visual tokens while maintaining comparable performance, leading to an 89% reduction in KV-Cache memory and 3.8xfaster inference.

</details>

---

## 394. VCA: Video Curious Agent for Long Video Understanding

- [ ] VCA: Video Curious Agent for Long Video Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VCA_Video_Curious_Agent_for_Long_Video_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VCA_Video_Curious_Agent_for_Long_Video_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long video understanding poses unique challenges due to its temporal complexity and low information density. Recent works address this task by sampling numerous frames or incorporating auxiliary tools using LLMs, both of which result in high computational costs. In this work, we introduce a curiosity-driven video agent with self-exploration capability, dubbed as "VCA". Built upon VLMs, VCA autonomously navigates video segments and efficiently builds a comprehensive understanding of complex video sequences.Instead of directly sampling frames, VCA employs a tree-search structure to explore video segments and collect frames. Rather than relying on external feedback or reward, VCA leverages VLM's self-generated intrinsic reward to guide its exploration, enabling it to capture the most crucial information for reasoning. Experimental results on multiple long video benchmarks demonstrate our approach's superior effectiveness and efficiency.

</details>

---

## 395. VLIPP: Towards Physically Plausible Video Generation with Vision and Language Informed Physical Prior

- [ ] VLIPP: Towards Physically Plausible Video Generation with Vision and Language Informed Physical Prior | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VLIPP_Towards_Physically_Plausible_Video_Generation_with_Vision_and_Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VLIPP_Towards_Physically_Plausible_Video_Generation_with_Vision_and_Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video diffusion models (VDMs) have advanced significantly in recent years, enabling the generation of highly realistic videos and drawing the attention of the community in their potential as world simulators. However, despite their capabilities, VDMs often fail to produce physically plausible videos due to an inherent lack of understanding of physics, resulting in incorrect dynamics and event sequences. To address this limitation, we propose a novel two-stage image-to-video generation framework that explicitly incorporates physics with vision and language informed physical prior. In the first stage, we employ a Vision Language Model (VLM) as a coarse-grained motion planner, integrating chain-of-thought and physics-aware reasoning to predict a rough motion trajectories/changes that approximate real-world physical dynamics while ensuring the inter-frame consistency. In the second stage, we use the predicted motion trajectories/changes to guide the video generation of a VDM. As the predicted motion trajectories/changes are rough, noise is added during inference to provide freedom to the VDM in generating motion with more fine details. Extensive experimental results demonstrate that our framework can produce physically plausible motion, and comparative evaluations highlight the notable superiority of our approach over existing methods. More video results and code are available on our Project Page: https://madaoer.github.io/projects/physically_plausible_video_generation/.

</details>

---

## 396. Verbalized Representation Learning for Interpretable Few-Shot Generalization

- [ ] Verbalized Representation Learning for Interpretable Few-Shot Generalization | https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Verbalized_Representation_Learning_for_Interpretable_Few-Shot_Generalization_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Verbalized_Representation_Learning_for_Interpretable_Few-Shot_Generalization_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Humans recognize objects after observing only a few examples, a remarkable capability enabled by their inherent language understanding of the real-world environment. Developing verbalized and interpretable representation can significantly improve model generalization in low-data settings. In this work, we propose Verbalized Representation Learning (VRL), a novel approach for automatically extracting human-interpretable features for object recognition using few-shot data. Our method uniquely captures inter-class differences and intra-class commonalities in the form of natural language by employing a Vision-Language Model (VLM) to identify key discriminative features between different classes and shared characteristics within the same class. These verbalized features are then mapped to numeric vectors through the VLM. The resulting feature vectors can be further utilized to train and infer with downstream classifiers. Experimental results show that, at the same model scale, VRL achieves a 24% absolute improvement over prior state-of-the-art methods while using 95% less data and a smaller model. Furthermore, compared to human-labeled attributes, the features learned by VRL exhibit a 20% absolute gain when used for downstream classification tasks.

</details>

---

## 397. MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI

- [ ] MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI | https://openaccess.thecvf.com/content/ICCV2025/html/Yao_MMReason_An_Open-Ended_Multi-Modal_Multi-Step_Reasoning_Benchmark_for_MLLMs_Toward_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yao_MMReason_An_Open-Ended_Multi-Modal_Multi-Step_Reasoning_Benchmark_for_MLLMs_Toward_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reasoning plays a crucial role in advancing Multimodal Large Language Models (MLLMs) toward Artificial General Intelligence.However, existing MLLM benchmarks often fall short in precisely and comprehensively evaluating long-chain reasoning abilities from three key aspects: (1) lack of difficulty and diversity, (2) susceptibility to guessability and memorization, (3) inadequate assessment of intermediate reasoning steps.To fill this gap, we introduce **MMReason**, a new benchmark designed to precisely and comprehensively evaluate MLLM long-chain reasoning capability with diverse, open-ended, challenging questions.First, we curate challenging questions requiring multi-step reasoning from various fields (i.e., 6 disciplines) and multiple difficulty levels (i.e., from pre-university to university, and from foundational to competition tiers).Second, these questions are reformulated into an open-ended format and filtered using a multi-model voting technique to eliminate shortcut cases related to guessing and memorization, ensuring robust reasoning evaluations.Third, we annotate the questions with detailed step-by-step solutions, and design a reference-based ternary scoring mechanism to reliably assess intermediate reasoning steps.With MMReason, we benchmark popular leading MLLMs and provide an in-depth analysis of their reasoning capabilities.We hope MMReason will serve as a valuable resource for advancing MLLM reasoning research.

</details>

---

## 398. GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields

- [ ] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields | https://openaccess.thecvf.com/content/ICCV2025/html/Yasuki_GeoProg3D_Compositional_Visual_Reasoning_for_City-Scale_3D_Language_Fields_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yasuki_GeoProg3D_Compositional_Visual_Reasoning_for_City-Scale_3D_Language_Fields_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of 3D language fields has enabled intuitive interactions with 3D scenes via natural language. However, existing approaches are typically limited to small-scale environments, lacking the scalability and compositional reasoning capabilities necessary for large, complex urban settings. To overcome these limitations, we propose GeoProg3D, a visual programming framework that enables natural language-driven interactions with city-scale high-fidelity 3D scenes. GeoProg3D consists of two key components: (i) a Geography-aware City-scale 3D Language Field (GCLF) that leverages a memory-efficient hierarchical 3D model to handle large-scale data, integrated with geographic information for efficiently filtering vast urban spaces using directional cues, distance measurements, elevation data, and landmark references; and (ii) Geographical Vision APIs (GV-APIs), specialized geographic vision tools such as area segmentation and object detection. Our framework employs large language models (LLMs) as reasoning engines to dynamically combine GV-APIs and operate GCLF, effectively supporting diverse geographic vision tasks. To assess performance in city-scale reasoning, we introduce GeoEval3D, a comprehensive benchmark dataset containing 952 query-answer pairs across five challenging tasks: grounding, spatial reasoning, comparison, counting, and measurement. Experiments demonstrate that GeoProg3D significantly outperforms existing 3D language fields and vision-language models across multiple tasks. To our knowledge, GeoProg3D is the first framework enabling compositional geographic reasoning in high-fidelity city-scale 3D environments via natural language.

</details>

---

## 399. ATAS: Any-to-Any Self-Distillation for Enhanced Open-Vocabulary Dense Prediction

- [ ] ATAS: Any-to-Any Self-Distillation for Enhanced Open-Vocabulary Dense Prediction | https://openaccess.thecvf.com/content/ICCV2025/html/Yeo_ATAS_Any-to-Any_Self-Distillation_for_Enhanced_Open-Vocabulary_Dense_Prediction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yeo_ATAS_Any-to-Any_Self-Distillation_for_Enhanced_Open-Vocabulary_Dense_Prediction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models such as CLIP have recently propelled open-vocabulary dense prediction tasks by enabling recognition of a broad range of visual concepts. However, CLIP still struggles with fine-grained, region-level understanding, hindering its effectiveness on these dense prediction tasks. We identify two pivotal factors required to address this limitation: semantic coherence and fine-grained vision-language alignment. Current adaptation methods often improve fine-grained alignment at the expense of semantic coherence, and often rely on extra modules or supervised fine-tuning. To overcome these issues, we propose Any-to-Any Self-Distillation (ATAS), a novel approach that simultaneously enhances semantic coherence and fine-grained alignment by leveraging a model's own knowledge across all representation levels. Unlike prior methods, ATAS uses only unlabeled images and an internal self-distillation process to refine CLIP's representations, preserving local semantic consistency while sharpening local detail recognition. On open-vocabulary object detection and semantic segmentation benchmarks, ATAS achieves substantial performance gains, outperforming baseline CLIP models. These results validate the effectiveness of our approach and underscore the importance of jointly maintaining semantic coherence and fine-grained alignment for advanced open-vocabulary dense prediction.

</details>

---

## 400. ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail

- [ ] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail | https://openaccess.thecvf.com/content/ICCV2025/html/Yeshwanth_ExCap3D_Expressive_3D_Scene_Understanding_via_Object_Captioning_with_Varying_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yeshwanth_ExCap3D_Expressive_3D_Scene_Understanding_via_Object_Captioning_with_Varying_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating text descriptions of objects in 3D indoor scenes is an important building block of embodied understanding. Existing methods do this by describing objects at a single level of detail, which often does not capture fine-grained details such as varying textures, materials, and shapes of the parts of objects.We propose the task of expressive 3D captioning: given an input 3D scene, describe objects at multiple levels of detail: a high-level object description, and a low-level description of the properties of its parts.To produce such captions, we present ExCap3D, an expressive 3D captioning model which takes as input a 3D scan, and for each detected object in the scan, generates a fine-grained collective description of the parts of the object, along with an object-level description conditioned on the part-level description.We design ExCap3D to encourage semantic consistency between the generated text descriptions, as well as textual similarity in the latent space, to further increase the quality of the generated captions.To enable this task, we generated the ExCap3D Dataset by leveraging a visual-language model (VLM) for multi-view captioning. ExCap3D Dataset contains captions on the ScanNet++ dataset with varying levels of detail,comprising 190k text descriptions of 34k 3D objects in 947 indoor scenes.Our experiments show that the object- and part-level of detail captions generated by ExCap3D are of higher quality than those produced by state-of-the-art methods, with a Cider score improvement of 17% and 126% for object- and part-level details respectively. Our code, dataset and models will be made publicly available.

</details>

---

## 401. Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation

- [ ] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation | https://openaccess.thecvf.com/content/ICCV2025/html/Ying_Towards_Omnimodal_Expressions_and_Reasoning_in_Referring_Audio-Visual_Segmentation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Ying_Towards_Omnimodal_Expressions_and_Reasoning_in_Referring_Audio-Visual_Segmentation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Referring audio-visual segmentation (RAVS) has recently seen significant advancements, yet challenges remain in integrating multimodal information and deeply understanding and reasoning about audiovisual content. To extend the boundaries of RAVS and facilitate future research in this field, we propose Omnimodal Referring Audio-Visual Segmentation (OmniAVS), a new dataset containing 2,104 videos and 61,095 multimodal referring expressions. OmniAVS stands out with three key innovations: (1) 8 types of multimodal expressions that flexibly combine text, speech, sound, and visual cues; (2) an emphasis on understanding audio content beyond just detecting their presence; and (3) the inclusion of complex reasoning and world knowledge in expressions. Furthermore, we introduce Omnimodal Instructed Segmentation Assistant (OISA), to address the challenges of multimodal reasoning and fine-grained understanding of audiovisual content in OmniAVS. OISA uses MLLM to comprehend complex cues and perform reasoning-based segmentation. Extensive experiments show that OISA outperforms existing methods on OmniAVS and achieves competitive results on other related tasks.

</details>

---

## 402. Dynamic Group Detection using VLM-augmented Temporal Groupness Graph

- [ ] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph | https://openaccess.thecvf.com/content/ICCV2025/html/Yokoyama_Dynamic_Group_Detection_using_VLM-augmented_Temporal_Groupness_Graph_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yokoyama_Dynamic_Group_Detection_using_VLM-augmented_Temporal_Groupness_Graph_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper proposes dynamic human group detection in videos. For detecting complex groups, not only the local appearance features of in-group members but also the global context of the scene are important. Such local and global appearance features in each frame are extracted using a Vision-Language Model (VLM) augmented for group detection in our method. For further improvement, the group structure should be consistent over time. While previous methods are stabilized on the assumption that groups are not changed in a video, our method detects dynamically changing groups by global optimization using a graph with all frames' groupness probabilities estimated by our groupness-augmented CLIP features. Our experimental results demonstrate that our method outperforms state-of-the-art group detection methods on public datasets. Code: https://github.com/irajisamurai/VLM-GroupDetection.git

</details>

---

## 403. LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition

- [ ] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition | https://openaccess.thecvf.com/content/ICCV2025/html/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Transformers (ViTs) have revolutionized large-scale visual modeling, yet remain underexplored in face recognition (FR) where CNNs still dominate. We identify a critical bottleneck: CNN-inspired training paradigms fail to unlock ViT's potential, leading to suboptimal performance and convergence instability.To address this challenge, we propose LVFace, a ViT-based FR model that integrates Progressive Cluster Optimization (PCO) to achieve superior results. Specifically, PCO sequentially applies negative class sub-sampling (NCS) for robust and fast feature alignment from random initialization, feature expectation penalties for centroid stabilization, performing cluster boundary refinement through full-batch training without NCS constraints. LVFace establishes a new state-of-the-art face recognition baseline, surpassing leading approaches such as UniFace and TopoFR across multiple benchmarks. Extensive experiments demonstrate that LVFace delivers consistent performance gains, while exhibiting scalability to large-scale datasets and compatibility with mainstream VLMs and LLMs. Notably, LVFace secured 1st place in the ICCV 2021 Masked Face Recognition (MFR)-Ongoing Challenge (March 2025), proving its efficacy in real-world scenarios. Project is available at https://github.com/bytedance/LVFace.

</details>

---

## 404. Auto-Controlled Image Perception in MLLMs via Visual Perception Tokens

- [ ] Auto-Controlled Image Perception in MLLMs via Visual Perception Tokens | https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Auto-Controlled_Image_Perception_in_MLLMs_via_Visual_Perception_Tokens_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Auto-Controlled_Image_Perception_in_MLLMs_via_Visual_Perception_Tokens_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In MLLMs, Visual perception refers to the process by which MLLMs encode visual inputs, such as images, and align them with the text embedding space. Currently, MLLMs still lack the capability to autonomously control their own visual perception processes. For example, they cannot selectively re-encode specific regions of an image or focus on information related to specific object categories. In this work, we propose the concept of Visual Perception Token, aiming to empower MLLM with a mechanism to control its visual perception processes. We design two types of Visual Perception Tokens, termed the Region Selection Token and the Vision Re-Encoding Token. MLLMs autonomously generate these tokens, just as they generate natural language tokens, and use them to trigger additional visual perception process. The Region Selection Token explicitly identifies regions of interest that require further processing, while the Vision Re-Encoding Token utilizes its hidden states to guide an additional vision encoding process. Extensive experiments highlight the effectiveness of these tokens in enhancing spatial reasoning, fine-grained understanding, Text/OCR-related VQA, and a wide range of other visual tasks. On average, the introduction of Visual Perception Tokens improves the performance of a 2B model by 30.9%, increasing its score from 0.572 to 0.749, and even outperforms a 7B parameter model by 20.0% (from 0.624).

</details>

---

## 405. DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding

- [ ] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Yu_DocThinker_Explainable_Multimodal_Large_Language_Models_with_Rule-based_Reinforcement_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yu_DocThinker_Explainable_Multimodal_Large_Language_Models_with_Rule-based_Reinforcement_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in document understanding. However, their reasoning processes remain largely black-box, making it difficult to ensure reliability and trustworthiness, especially in high-stakes domains such as legal, financial, and medical document analysis. Existing methods use fixed Chain-of-Thought (CoT) reasoning with supervised fine-tuning (SFT) but suffer from catastrophic forgetting, poor adaptability, and limited generalization across domain tasks. In this paper, we propose DocThinker, a rule-based Reinforcement Learning (RL) framework for dynamic inference-time reasoning. Instead of relying on static CoT templates, DocThinker autonomously refines reasoning strategies via policy learning, generating explainable intermediate results, including structured reasoning processes, rephrased questions, regions of interest (RoI) supporting the answer, and the final answer. By integrating multi-objective rule-based rewards and KL-constrained optimization, our method mitigates catastrophic forgetting and enhances both adaptability and transparency. Extensive experiments on multiple benchmarks demonstrate that DocThinker significantly improves generalization while producing more explainable and human-understandable reasoning steps. Our findings highlight RL as a powerful alternative for enhancing explainability and adaptability in MLLM-based document understanding. Code will be available at https://github.com/wenwenyu/DocThinker.

</details>

---

## 406. Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness

- [ ] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness | https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Mastering_Collaborative_Multi-modal_Data_Selection_A_Focus_on_Informativeness_Uniqueness_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Mastering_Collaborative_Multi-modal_Data_Selection_A_Focus_on_Informativeness_Uniqueness_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuning fine-tunes pre-trained Multi-modal Large Language Models (MLLMs) to handle real-world tasks. However, the rapid expansion of visual instruction datasets introduces data redundancy, leading to excessive computational costs. We propose a collaborative framework, DataTailor, which leverages three key principles--informativeness, uniqueness, and representativeness--for effective data selection. We argue that a valuable sample should be informative of the task, non-redundant, and represent the sample distribution (i.e., not an outlier). We further propose practical ways to score against each principle, which automatically adapts to a given dataset without tedious hyperparameter tuning. Comprehensive experiments on various benchmarks demonstrate that DataTailor achieves 101.3% of the performance of full-data fine-tuning with only 15% of the data, significantly reducing computational costs while maintaining superior results. This exemplifies the "Less is More" philosophy in MLLM development. The code and data is available in this URL.

</details>

---

## 407. Multi-View Slot Attention Using Paraphrased Texts for Face Anti-Spoofing

- [ ] Multi-View Slot Attention Using Paraphrased Texts for Face Anti-Spoofing | https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Multi-View_Slot_Attention_Using_Paraphrased_Texts_for_Face_Anti-Spoofing_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Multi-View_Slot_Attention_Using_Paraphrased_Texts_for_Face_Anti-Spoofing_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent face anti-spoofing (FAS) methods have shown remarkable cross-domain performance by employing vision-language models like CLIP. However, existing CLIP-based FAS models do not fully exploit CLIP's patch embedding tokens, failing to detect critical spoofing clues. Moreover, these models rely on a single text prompt per class (e.g, 'live' or 'fake'), which limits generalization. To address these issues, we propose MVP-FAS, a novel framework incorporating two key modules: Multi-View Slot attention (MVS) and Multi-Text Patch Alignment (MTPA). Both modules utilize multiple paraphrased texts to generate generalized features and reduce dependence on domain-specific text. MVS extracts local detailed spatial features and global context from patch embeddings by leveraging diverse texts with multiple perspectives. MTPA aligns patches with multiple text representations to improve semantic robustness. Extensive experiments demonstrate that MVP-FAS achieves superior generalization performance, outperforming previous state-of-the-art methods on cross-domain datasets.

</details>

---

## 408. VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation

- [ ] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Yu_VEGGIE_Instructional_Editing_and_Reasoning_Video_Concepts_with_Grounded_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yu_VEGGIE_Instructional_Editing_and_Reasoning_Video_Concepts_with_Grounded_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While recent video diffusion models enable video editing, unifying diverse instructional editing tasks (e.g., add, remove, modify) under a single framework remains a significant challenge. In this paper, we introduce VEGGIE, a Video Editor with Grounded Generation from Instructions, a simple end-to-end framework that unifies video concept editing, grounding, and reasoning based on diverse user instructions. Specifically, given a video and text query, VEGGIE first utilises an MLLM to interpret user intentions in instructions and ground them to the video contexts, generating frame-specific grounded task queries for pixel-space responses. A diffusion model then renders these plans and generates edited videos that align with user intent. To support diverse tasks and complex instructions, we employ a curriculum learning strategy: first aligning the MLLM and video diffu- sion model with large-scale instructional image editing data, followed by end-to-end fine-tuning on high-quality multitask video data. Additionally, we introduce a novel data synthe- sis pipeline to generate paired instructional video editing data for model training. It transforms static image data into diverse, high-quality video editing samples by leveraging Image-to-Video models to inject dynamics. VEGGIE shows strong performance in instructional video editing with different editing skills, outperforming the best instructional baseline as a versatile model, while other models struggle with multi-tasking. VEGGIE also excels in video object grounding and reasoning segmentation, where other baselines fail. We further reveal how the multiple tasks help each other and highlight promising applications like zero-shot multimodal instructional and in-context video editing.

</details>

---

## 409. VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos

- [ ] VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos | https://openaccess.thecvf.com/content/ICCV2025/html/Yu_VRBench_A_Benchmark_for_Multi-Step_Reasoning_in_Long_Narrative_Videos_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yu_VRBench_A_Benchmark_for_Multi-Step_Reasoning_in_Long_Narrative_Videos_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present VRBench, the first long narrative video benchmark crafted for evaluating large models' multi-step reasoning capabilities, addressing limitations in existing evaluations that overlook temporal reasoning and procedural validity. It comprises 960 long videos (with an average duration of 1.6 hours), along with 8,243 human-labeled multi-step question-answering pairs and 25,106 reasoning steps with timestamps. These videos are curated via a multi-stage filtering process including expert inter-rater reviewing to prioritize plot coherence. We develop a human-AI collaborative framework that generates coherent reasoning chains, each requiring multiple temporally grounded steps, spanning seven types (e.g., event attribution, implicit inference). VRBench designs a multi-phase evaluation pipeline that assesses models at both the outcome and process levels. Apart from the MCQs for the final results, we propose a progress-level LLM-guided scoring metric to evaluate the quality of the reasoning chain from multiple dimensions comprehensively. Through extensive evaluations of 12 LLMs and 19 VLMs on VRBench, we undertake a thorough analysis and provide valuable insights that advance the field of multi-step reasoning.

</details>

---

## 410. ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers

- [ ] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers | https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_ShortV_Efficient_Multimodal_Large_Language_Models_by_Freezing_Visual_Tokens_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_ShortV_Efficient_Multimodal_Large_Language_Models_by_Freezing_Visual_Tokens_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) suffer from high computational costs due to their massive size and the large number of visual tokens. In this paper, we investigate layer-wise redundancy in MLLMs by introducing a novel metric, Layer Contribution (LC), which quantifies the impact of a layer's transformations on visual and text tokens, respectively. The calculation of LC involves measuring the divergence in model output that results from removing the layer's transformations on the specified tokens. Our pilot experiment reveals that many layers of MLLMs exhibit minimal contribution during the processing of visual tokens. Motivated by this observation, we propose ShortV, a training-free method that leverages LC to identify ineffective layers, and freezes visual token updates in these layers. Experiments show that ShortV can freeze visual token in approximately 60% of the MLLM layers, thereby dramatically reducing computational costs related to updating visual tokens. For example, it achieves a 50% reduction in FLOPs on LLaVA-NeXT-13B while maintaining superior performance. The code is publicly available at https://github.com/icip-cas/ShortV.

</details>

---

## 411. WalkVLM: Aid Visually Impaired People Walking by Vision Language Model

- [ ] WalkVLM: Aid Visually Impaired People Walking by Vision Language Model | https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_WalkVLM_Aid_Visually_Impaired_People_Walking_by_Vision_Language_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_WalkVLM_Aid_Visually_Impaired_People_Walking_by_Vision_Language_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Approximately 200 million individuals around the world suffer from varying degrees of visual impairment, making it crucial to leverage AI technology to offer walking assistance for these people.With the recent progress of vision-language models (VLMs), applying VLMs to offer walking guidance has become popular. However, the existing methods of walking guidance are mainly based on self-curated question-answering datasets that are not publicly accessible, without a standardized benchmark for training or evaluation. Moreover, walking assistance often requires real-time streaming video analysis and the generation of concise yet informative reminders, making VLMs struggle due to excessive responses and low efficiency in inferences. In this paper, we introduce the first large-scale dataset dedicated to walking assistance, comprising 12,000 video-annotation pairs, to provide a unified benchmark for training and evaluating systems to help visually-impaired individuals walk. Furthermore, a WalkVLM model is proposed, which employs chain of thought for hierarchical planning to generate concise but informative reminders and utilizes temporal-aware adaptive prediction to reduce the temporal redundancy of reminders. Finally, we have established a solid benchmark for blind walking task and verified the advantages of WalkVLM in stream video processing for this task compared to other VLMs.

</details>

---

## 412. Zero-Shot Vision Encoder Grafting via LLM Surrogates

- [ ] Zero-Shot Vision Encoder Grafting via LLM Surrogates | https://openaccess.thecvf.com/content/ICCV2025/html/Yue_Zero-Shot_Vision_Encoder_Grafting_via_LLM_Surrogates_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Yue_Zero-Shot_Vision_Encoder_Grafting_via_LLM_Surrogates_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) typically pair a modestly sized vision encoder with a large language model (LLM), e.g., Llama-70B, making the decoder the primary computational burden during training.To reduce costs, a promising strategy is to first train the vision encoder using a small language model before transferring it to the large one.We construct small "surrogate models" that share the same embedding space and representation language as the large target LLM by directly inheriting its shallow layers.Vision encoders trained on the surrogate can then be directly transferred to the larger model, a process we call zero-shot grafting -- when plugged directly into the full-size target LLM, the grafted pair surpasses the encoder-surrogate pair and, on some benchmarks, even performs on par with full decoder training with the target LLM.Furthermore, our surrogate training approach reduces overall VLM training costs by  45% when using Llama-70B as the decoder.

</details>

---

## 413. MOSCATO: Predicting Multiple Object State Change Through Actions

- [ ] MOSCATO: Predicting Multiple Object State Change Through Actions | https://openaccess.thecvf.com/content/ICCV2025/html/Zameni_MOSCATO_Predicting_Multiple_Object_State_Change_Through_Actions_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zameni_MOSCATO_Predicting_Multiple_Object_State_Change_Through_Actions_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce MOSCATO: a new benchmark for predicting the evolving states of multiple objects through long procedural videos with multiple actions. While prior work in object state prediction has typically focused on a single object undergoing one or a few state changes, real-world tasks require tracking many objects whose states evolve over multiple actions. Given the high cost of gathering framewise object-state labels for many videos, we develop a weakly-supervised multiple object state prediction framework, which only uses action labels during training. Specifically, we propose a novel Pseudo-Label Acquisition (PLA) pipeline that integrates large language models, vision-language models, and action segment annotations to generate fine-grained, per-frame object-state pseudo-labels for training a Multiple Object State Prediction (MOSP) network. We further devise a State-Action Interaction (SAI) module that explicitly models the correlations between actions and object states, thereby improving MOSP. To facilitate comprehensive evaluation, we create the MOSCATO benchmark by augmenting three egocentric video datasets with framewise object-state annotations. Experiments show that our multi-stage pseudo-labeling approach and SAI module significantly boost performance over zero-shot VLM baselines and naive extensions of existing methods, underscoring the importance of holistic action-state modeling for fine-grained procedural video understanding.

</details>

---

## 414. 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding

- [ ] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Zemskova_3DGraphLLM_Combining_Semantic_Graphs_and_Large_Language_Models_for_3D_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zemskova_3DGraphLLM_Combining_Semantic_Graphs_and_Large_Language_Models_for_3D_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

A 3D scene graph represents a compact scene model by capturing both the objects present and the semantic relationships between them, making it a promising structure for robotic applications. To effectively interact with users, an embodied intelligent agent should be able to answer a wide range of natural language queries about the surrounding 3D environment. Large Language Models (LLMs) are beneficial solutions for user-robot interaction due to their natural language understanding and reasoning abilities. Recent methods for learning scene representations have shown that adapting these representations to the 3D world can significantly improve the quality of LLM responses. However, existing methods typically rely only on geometric information, such as object coordinates, and overlook the rich semantic relationships between objects. In this work, we propose 3DGraphLLM, a method for constructing a learnable representation of a 3D scene graph that explicitly incorporates semantic relationships. This representation is used as input to LLMs for performing 3D vision-language tasks. In our experiments on popular ScanRefer, Multi3DRefer, ScanQA, Sqa3D, and Scan2cap datasets, we demonstrate that our approach outperforms baselines that do not leverage semantic relationships between objects. The code is publicly available at https://github.com/CognitiveAISystems/3DGraphLLM.

</details>

---

## 415. AVAM: a Universal Training-free Adaptive Visual Anchoring Embedded into Multimodal Large Language Model for Multi-image Question Answering

- [ ] AVAM: a Universal Training-free Adaptive Visual Anchoring Embedded into Multimodal Large Language Model for Multi-image Question Answering | https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_AVAM_a_Universal_Training-free_Adaptive_Visual_Anchoring_Embedded_into_Multimodal_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_AVAM_a_Universal_Training-free_Adaptive_Visual_Anchoring_Embedded_into_Multimodal_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of Multimodal Large Language Models (MLLMs) has driven significant progress in Visual Question Answering (VQA), evolving from Single to Multi Image VQA (MVQA). However, the increased number of images in MVQA inevitably introduces substantial visual redundancy that is irrelevant to question answering, negatively impacting both accuracy and efficiency. To address this issue, existing methods lack flexibility in controlling the number of compressed visual tokens and tend to produce discrete visual fragments, which hinder MLLMs' ability to comprehend images holistically. In this paper, we propose a straightforward yet universal Adaptive Visual Anchoring strategy, which can be seamlessly integrated into existing MLLMs, offering significant accuracy improvements through adaptive compression. Meanwhile, to balance the results derived from both global and compressed visual input, we further introduce a novel collaborative decoding mechanism, enabling optimal performance. Extensive experiments validate the effectiveness of our method, demonstrating consistent performance improvements across various MLLMs. The code will be publicly available.

</details>

---

## 416. Factorized Learning for Temporally Grounded Video-Language Models

- [ ] Factorized Learning for Temporally Grounded Video-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Factorized_Learning_for_Temporally_Grounded_Video-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Factorized_Learning_for_Temporally_Grounded_Video-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent video-language models have shown great potential for video understanding, but still struggle with accurate temporal grounding for event-level perception. We observe that two main factors in video understanding (i.e., temporal grounding and textual response) form a logical hierarchy: accurate temporal evidence grounding lays the foundation for reliable textual response. However, existing works typically handle these two tasks in a coupled manner without a clear logical structure, leading to sub-optimal objectives. We address this from a factorized learning perspective. We first propose D2VLM, a framework that decouples the learning of these two tasks while also emphasizing their inherent dependency. We adopt a "grounding then answering with evidence referencing" paradigm and introduce evidence tokens for evidence grounding, which emphasize event-level visual semantic capture beyond the focus on timestamp representation in existing works. To further facilitate the learning of these two tasks, we introduce a novel factorized preference optimization (FPO) algorithm. Unlike standard preference optimization, FPO explicitly incorporates probabilistic temporal grounding modeling into the optimization objective, enabling preference learning for both temporal grounding and textual response. We also construct a synthetic dataset to address the lack of suitable datasets for factorized preference learning with explicit temporal grounding. Experiments on various tasks demonstrate the clear advantage of our approach. Our source code is available at https://github.com/nusnlp/d2vlm.

</details>

---

## 417. Skip-Vision: Efficient and Scalable Acceleration of Vision-Language Models via Adaptive Token Skipping

- [ ] Skip-Vision: Efficient and Scalable Acceleration of Vision-Language Models via Adaptive Token Skipping | https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Skip-Vision_Efficient_and_Scalable_Acceleration_of_Vision-Language_Models_via_Adaptive_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Skip-Vision_Efficient_and_Scalable_Acceleration_of_Vision-Language_Models_via_Adaptive_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Transformer-based models have driven significant advancements in Multimodal Large Language Models (MLLMs), yet their computational costs surge drastically when scaling resolution, training data, and model parameters. A key bottleneck stems from the proliferation of visual tokens required for fine-grained image understanding. We propose Skip-Vision, a unified framework addressing both training and inference inefficiencies in vision-language models. On top of conventional token compression approaches, our method introduces two complementary acceleration strategies. For training acceleration, we observe that Feed-Forward Network (FFN) computations on visual tokens induce marginal feature updates. This motivates our Skip-FFN strategy, which bypasses FFN layers for redundant visual tokens. For inference acceleration, we design a selective KV-cache removal mechanism that prunes the skipped key-value pairs during decoding while preserving model performance. Experimental results demonstrate that Skip-Vision reduces training time by up to 35%, inference FLOPs by 75%, and latency by 45%, while achieving comparable or superior performance to existing methods. Our work provides a practical solution for scaling high-performance MLLMs with enhanced efficiency.

</details>

---

## 418. Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models

- [ ] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Visual-Oriented_Fine-Grained_Knowledge_Editing_for_MultiModal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Visual-Oriented_Fine-Grained_Knowledge_Editing_for_MultiModal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing knowledge editing works for MultiModal Large Language Models primarily focus on text-oriented, coarse-grained scenarios, where modifying textual content alone is sufficient. As a result, they fail to capture the unique challenges of multimodal editing, particularly when visual information is central to knowledge representation. In this paper, we introduce a visual-oriented, fine-grained multimodal knowledge editing task that targets precise modifications in images containing multiple interacting entities. To support this, we propose the Fine-Grained Visual Knowledge Editing (FGVEdit) benchmark, designed to evaluate the accuracy and effectiveness of multimodal editing at a granular level. To address this challenge, we present the Multimodal Scope Classifier-based Knowledge Editor (MSCKE), a new framework that leverages a multimodal scope classifier to integrate both textual and visual information. By accurately identifying and updating knowledge localized within images, MSCKE ensures precise editing while preserving unrelated content. Extensive experiments on the FGVEdit benchmark highlight the complexity of this new task and demonstrate that existing methods struggle with fine-grained multimodal editing. Our results highlight MSCKE as a scalable and promising framework for advancing multimodal knowledge editing. Code is available at https://github.com/zeng-zhen/FGVEdit.

</details>

---

## 419. Text2Outfit: Controllable Outfit Generation with Multimodal Language Models

- [ ] Text2Outfit: Controllable Outfit Generation with Multimodal Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zhai_Text2Outfit_Controllable_Outfit_Generation_with_Multimodal_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhai_Text2Outfit_Controllable_Outfit_Generation_with_Multimodal_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing outfit recommendation frameworks focus on outfit compatibility prediction and complementary item retrieval. We present a text-driven outfit generation framework, Text2Outfit, which generates outfits controlled by text prompts. Our framework supports two forms of outfit recommendation: 1) Text-to-outfit generation, where the prompt includes the specification for each outfit item (e.g., product features), and the model retrieves items that match the prompt and are stylistically compatible. 2) Seed-to-outfit generation, where the prompt includes the specification for a seed item, and the model both predicts which product types the outfit should include (referred to as composition generation) and retrieves the remaining items to build an outfit. We develop a large language model (LLM) framework that learns the cross-modal mapping between text and image set, and predicts a set of embeddings and compositions to retrieve outfit items. We devise an attention masking mechanism in LLM to handle the alignment between text descriptions and image tokens from different categories. We conduct experiments on the Polyvore dataset and evaluate the quality of the generated outfits from two perspectives: 1) feature matching for outfit items, and 2) outfit visual compatibility. The results demonstrate that our approach significantly outperforms the baseline methods in text to outfit generation.

</details>

---

## 420. Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring

- [ ] Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring | https://openaccess.thecvf.com/content/ICCV2025/html/Zhan_Griffon_v2_Advancing_Multimodal_Perception_with_High-Resolution_Scaling_and_Visual-Language_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhan_Griffon_v2_Advancing_Multimodal_Perception_with_High-Resolution_Scaling_and_Visual-Language_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision Language Models have achieved fine-grained object perception, but the limitation of image resolution remains a significant obstacle to surpassing the performance of task-specific experts in complex and dense scenarios. Such limitation further restricts the model's potential to achieve nuanced visual and language referring in domains such as GUI Agents, counting, etc. To address this issue, we introduce a unified high-resolution generalist model, Griffon v2, enabling flexible object referring with visual and textual prompts. To efficiently scale up image resolution, we design a simple and lightweight down-sampling projector to overcome the input tokens constraint in Large Language Models. This design inherently preserves the complete contexts and fine details and significantly improves multimodal perception ability, especially for small objects. Building upon this, we further equip the model with visual-language co-referring capabilities through a plug-and-play visual tokenizer. It enables user-friendly interaction with flexible target images, free-form texts, and even coordinates. Experiments demonstrate that Griffon v2 can localize objects of interest with visual and textual referring, achieve state-of-the-art performance on REC and phrase grounding, and outperform expert models in object detection, object counting, and REG. Data and codes are released at https://github.com/jefferyZhan/Griffon.

</details>

---

## 421. 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining

- [ ] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_2.5_Years_in_Class_A_Multimodal_Textbook_for_Vision-Language_Pretraining_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_2.5_Years_in_Class_A_Multimodal_Textbook_for_Vision-Language_Pretraining_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Compared to image-text pair data, interleaved corpora enable Vision-Language Models (VLMs) to understand the world more naturally like humans. However, such existing datasets are crawled from webpage, facing challenges like low knowledge density, loose image-text relations, and poor logical coherence between images. On the other hand, the internet hosts vast instructional videos (e.g., online geometry courses) that are widely used by humans to learn foundational subjects, yet these valuable resources remain underexplored in VLM training. In this paper, we introduce a high-quality multimodal textbook corpus with richer foundational knowledge for VLM pretraining. It collects over 2.5 years of instructional videos, totaling 22,000 class hours. We first use an LLM-proposed taxonomy to systematically gather instructional videos. Then we progressively extract and refine visual (keyframes), audio (ASR), and textual knowledge (OCR) from the videos, and organize as an image-text interleaved corpus based on temporal order. Compared to its counterparts, our video-centric textbook offers more coherent context, richer knowledge, and better image-text alignment. Experiments demonstrate its superb pretraining performance, particularly in knowledge- and reasoning-intensive tasks like ScienceQA and MathVista. Moreover, VLMs pre-trained on our textbook exhibit outstanding interleaved context awareness, leveraging visual and textual cues in their few-shot context for task solving. Code and dataset are available on https://multimodal-interleaved-textbook.github.io.

</details>

---

## 422. Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection

- [ ] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Adaptive_Prompt_Learning_via_Gaussian_Outlier_Synthesis_for_Out-of-distribution_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Adaptive_Prompt_Learning_via_Gaussian_Outlier_Synthesis_for_Out-of-distribution_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Out-of-distribution (OOD) detection aims to distinguish whether detected objects belong to known categories or not. Existing methods extract OOD samples from In-distribution (ID) data to regularize the model's decision boundaries. However, the decision boundaries are not adequately regularized because the model does not have sufficient knowledge about the distribution of OOD data. To address the above issue, we propose an Adaptive Prompt Learning framework via Gaussian Outlier Synthesis (APLGOS) for OOD detection. Specifically, we leverage the Vision-Language Model (VLM) to initialize learnable ID prompts by sampling standardized results from pre-defined Q&A pairs. Region-level prompts are synthesised in low-likelihood regions of class-conditional gaussian distributions. These prompts are then utilized to initialize learnable OOD prompts and optimized with adaptive prompt learning. Also, OOD pseudo-samples are synthesised via gaussian outlier synthesis. The aforementioned methodology regularizes the model to learn more compact decision boundaries for ID and OOD categories. Extensive experiments show that APLGOS achieves state-of-the-art performance with less ID data on four mainstream datasets.

</details>

---

## 423. Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs

- [ ] Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (VLMs) generally contain significantly more visual tokens than their textual counterparts, resulting in a considerable computational burden. Recent efforts have been made to tackle this issue by pruning visual tokens early within the language model. Most existing works use attention scores between text and visual tokens to assess the importance of visual tokens. However, in this study, we first analyze the text-visual attention in the language model and find that this score is not an ideal indicator for visual token pruning. Based on the analysis, We propose VisPruner, a plug-and-play method that utilizes visual cues for more effective token pruning in LVLMs. Specifically, we first use visual attention to select a limited number of significant tokens. Then, we remove duplicate tokens from the remaining ones based on their similarity. By retaining diverse tokens alongside the initially selected important tokens, we maximally preserve the visual information of the input image. Experimental results demonstrate that our VisPruner sustains strong performance across various VLM architectures and reduction ratios, significantly outperforming existing methods based on text-visual attention. Notably, without any training, VisPruner can reduce the FLOPs of LLaVA-1.5-7B by 91% and inference latency by 75%, while maintaining comparable performance. Our code is available at https://github.com/Theia-4869/VisPruner.

</details>

---

## 424. Beyond Training: Dynamic Token Merging for Zero-Shot Video Understanding

- [ ] Beyond Training: Dynamic Token Merging for Zero-Shot Video Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Training_Dynamic_Token_Merging_for_Zero-Shot_Video_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Training_Dynamic_Token_Merging_for_Zero-Shot_Video_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have opened new avenues for video understanding. However, achieving high performance in zero-shot video tasks remains challenging. Traditional video processing methods rely heavily on fine-tuning to capture nuanced spatial-temporal details, which incurs significant data and computation costs. In contrast, training-free approaches, though efficient, often lack robustness in preserving context-rich features across complex video content. To this end, we propose DyTo, a novel dynamic token merging framework for zero-shot video understanding that adaptively optimizes token efficiency while preserving crucial scene details. DyTo integrates hierarchical frame selection and bipartite token merging strategy to dynamically cluster key frames and selectively compress token sequences, striking a balance between computational efficiency with semantic richness. Extensive experiments across multiple benchmarks demonstrate the effectiveness of DyTo. Our method not only sets a new state-of-the-art for zero-shot video understanding when applied to image-trained MLLMs, but also further boosts the performance of models already fine-tuned on video data. Code is available at https://github.com/Jam1ezhang/DYTO.

</details>

---

## 425. Efficient Visual Place Recognition Through Multimodal Semantic Knowledge Integration

- [ ] Efficient Visual Place Recognition Through Multimodal Semantic Knowledge Integration | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual place recognition is crucial for autonomous navigation and robotic mapping. Current methods struggle with perceptual aliasing and computational inefficiency. We present SemVPR, a novel approach integrating multimodal semantic knowledge into VPR. By leveraging a pre-trained vision-language model as a teacher during the training phase, SemVPR learns local visual and semantic descriptors simultaneously, effectively mitigating perceptual aliasing through semantic-aware aggregation without extra inference cost. The proposed nested descriptor learning strategy generates a series of ultra-compact global descriptors, reduced by approximately compared to state-of-the-art methods, in a coarse-to-fine manner, eliminating the need for offline dimensionality reduction or training multiple models. Extensive experiments across various VPR benchmarks demonstrate that SemVPR consistently outperforms state-of-the-art methods with significantly lower computational costs, rendering its feasibility for latency-sensitive scenarios in real-world applications.

</details>

---

## 426. Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance

- [ ] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Egocentric_Action-aware_Inertial_Localization_in_Point_Clouds_with_Vision-Language_Guidance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Egocentric_Action-aware_Inertial_Localization_in_Point_Clouds_with_Vision-Language_Guidance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a novel inertial localization framework named Egocentric Action-aware Inertial Localization (EAIL), which leverages egocentric action cues from head-mounted IMU signals to localize the target individual within a 3D point cloud. Human inertial localization is challenging due to IMU sensor noise that causes trajectory drift over time. The diversity of human actions further complicates IMU signal processing by introducing various motion patterns. Nevertheless, we observe that some actions captured by the head-mounted IMU correlate with spatial environmental structures (e.g., bending down to look inside an oven, washing dishes next to a sink), thereby serving as spatial anchors to compensate for the localization drift. The proposed EAIL framework learns such correlations via hierarchical multi-modal alignment with vision-language guidance. By assuming that the 3D point cloud of the environment is available, it contrastively learns modality encoders that align short-term egocentric action cues in IMU signals with local environmental features in the point cloud. The learning process is enhanced using concurrently collected vision and language signals to improve multimodal alignment. The learned encoders are then used in reasoning the IMU data and the point cloud over time and space to perform inertial localization. Interestingly, these encoders can further be utilized to recognize the corresponding sequence of actions as a by-product. Extensive experiments demonstrate the effectiveness of the proposed framework over state-of-the-art inertial localization and inertial action recognition baselines.

</details>

---

## 427. Enhancing Zero-shot Object Counting via Text-guided Local Ranking and Number-evoked Global Attention

- [ ] Enhancing Zero-shot Object Counting via Text-guided Local Ranking and Number-evoked Global Attention | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Enhancing_Zero-shot_Object_Counting_via_Text-guided_Local_Ranking_and_Number-evoked_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Enhancing_Zero-shot_Object_Counting_via_Text-guided_Local_Ranking_and_Number-evoked_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-guided zero-shot object counting leverages vision-language models (VLMs) to count objects of an arbitrary class given by a text prompt. Existing approaches for this challenging task only utilize local patch-level features to fuse with text feature, ignoring the important influence of the global image-level feature. In this paper, we propose a universal strategy that can exploit both local patch-level features and global image-level feature simultaneously. Specifically, to improve the localization ability of VLMs, we propose Text-guided Local Ranking. Depending on the prior knowledge that foreground patches have higher similarity with the text prompt, a new local-text rank loss is designed to increase the differences between the similarity scores of foreground and background patches which push foreground and background patches apart. To enhance the counting ability of VLMs, Number-evoked Global Attention is introduced to first align global image-level feature with multiple number-conditioned text prompts. Then, the one with the highest similarity is selected to compute cross-attention with the global image-level feature. Through extensive experiments on widely used datasets and methods, the proposed approach has demonstrated superior advancements in performance, generalization, and scalability. Furthermore, to better evaluate text-guided zero-shot object counting methods, we propose a dataset named ZSC-8K, which is larger and more challenging, to establish a new benchmark. Codes and ZSC-8K dataset will be available.

</details>

---

## 428. FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers

- [ ] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_FALCON_Resolving_Visual_Redundancy_and_Fragmentation_in_High-resolution_Multimodal_Large_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_FALCON_Resolving_Visual_Redundancy_and_Fragmentation_in_High-resolution_Multimodal_Large_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The incorporation of high-resolution visual input equips multimodal large language models (MLLMs) with enhanced visual perception capabilities for real-world tasks. However, most existing high-resolution MLLMs rely on a cropping-based approach to process images, which leads to fragmented visual encoding and a sharp increase in redundant tokens. To tackle these issues, we propose the FALCON model. FALCON introduces a novel visual register technique to simultaneously: 1) Eliminate redundant tokens at the stage of visual encoding. To directly address the visual redundancy present in the output of vision encoder, we propose a Register-based Representation Compacting (ReCompact) mechanism. This mechanism introduces a set of learnable visual registers designed to adaptively aggregate essential information while discarding redundancy. It enables the encoder to produce a more compact visual representation with a minimal number of output tokens, thus eliminating the need for an additional compression module. 2) Ensure continuity in visual encoding. To address the potential encoding errors caused by fragmented visual inputs, we develop a Register Interactive Attention (ReAtten) module. This module facilitates effective and efficient information exchange across sub-images by enabling interactions between visual registers. It ensures the continuity of visual semantics throughout the encoding. We conduct comprehensive experiments with FALCON on high-resolution benchmarks across a wide range of scenarios. FALCON demonstrates superior performance with a remarkable 9-fold reduction in visual tokens.

</details>

---

## 429. Flash-VStream: Efficient Real-Time Understanding for Long Video Streams

- [ ] Flash-VStream: Efficient Real-Time Understanding for Long Video Streams | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Flash-VStream_Efficient_Real-Time_Understanding_for_Long_Video_Streams_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Flash-VStream_Efficient_Real-Time_Understanding_for_Long_Video_Streams_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Benefiting from the advances in large language models and cross-modal alignment, existing multimodal large language models have achieved prominent performance in image and short video understanding. However, the understanding of long videos is still challenging, as their long-context nature results in significant computational and memory overhead. Most existing work treats long videos in the same way as short videos, which is inefficient for real-world applications and hard to generalize to even longer videos. To address these issues, we propose Flash-VStream, an efficient video language model capable of processing extremely long videos and responding to user queries in real time. Particularly, we design a Flash Memory module, containing a low-capacity context memory to aggregate long-context temporal information and model the distribution of information density, and a high-capacity augmentation memory to retrieve detailed spatial information based on this distribution. Compared to existing models, Flash-VStream achieves significant reductions in inference latency. Extensive experiments on long video benchmarks and comprehensive video benchmarks, i.e., EgoSchema, MLVU, LVBench, MVBench and Video-MME, demonstrate the state-of-the-art performance and outstanding efficiency of our method. Code is available at https://github.com/IVGSZ/Flash-VStream.

</details>

---

## 430. FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers

- [ ] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_FreeCus_Free_Lunch_Subject-driven_Customization_in_Diffusion_Transformers_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_FreeCus_Free_Lunch_Subject-driven_Customization_in_Diffusion_Transformers_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In light of recent breakthroughs in text-to-image (T2I) generation, particularly with diffusion transformers (DiT), subject-driven technologies are increasingly being employed for high-fidelity customized production that preserves subject identity from reference inputs, enabling thrilling design workflows and engaging entertainment. Existing alternatives typically require either per-subject optimization via trainable text embeddings or training specialized encoders for subject feature extraction on large-scale datasets. Such dependencies on training procedures fundamentally constrain their practical applications. More importantly, current methodologies fail to fully leverage the inherent zero-shot potential of modern diffusion transformers (e.g., the Flux series) for authentic subject-driven synthesis. To bridge this gap, we propose FreeCus, a genuinely training-free framework that activates DiT's capabilities through three key innovations: 1) We introduce a pivotal attention sharing mechanism that captures the subject's layout integrity while preserving crucial editing flexibility. 2) Through a straightforward analysis of DiT's dynamic shifting, we propose an upgraded variant that significantly improves fine-grained feature extraction. 3) We further integrate advanced Multimodal Large Language Models (MLLMs) to enrich cross-modal semantic representations. Extensive experiments reflect that our method successfully unlocks DiT's zero-shot ability for consistent subject synthesis across diverse contexts, achieving state-of-the-art or comparable results compared to approaches that require additional training. Notably, our framework demonstrates seamless compatibility with existing inpainting pipelines and control modules, facilitating more compelling experiences. Our code is available at: https://github.com/Monalissaa/FreeCus.

</details>

---

## 431. HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?

- [ ] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding? | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_HRScene_How_Far_Are_VLMs_from_Effective_High-Resolution_Image_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_HRScene_How_Far_Are_VLMs_from_Effective_High-Resolution_Image_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-resolution image (HRI) understanding aims to process images with a large number of pixels, such as pathological images and agricultural aerial images, both of which can exceed 1 million pixels. Vision Large Language Models (VLMs) can allegedly handle HRIs, however, there is a lack of a comprehensive benchmark for VLMs to evaluate HRI understanding. To address this gap, we introduce HRScene, a novel unified benchmark for HRI understanding with rich scenes. HRScene incorporates 25 real-world datasets and 2 synthetic diagnostic datasets with resolutions ranging from 1,024 times 1,024 to 35,503 times 26,627. HRScene is collected and re-annotated by 10 graduate-level annotators, covering 25 scenarios, ranging from microscopic to radiology images, street views, long-range pictures, and telescope images. It includes HRIs of real-world objects, scanned documents, and composite multi-image. The two diagnostic evaluation datasets are synthesized by combining the target image with the gold answer and distracting images in different orders, assessing how well models utilize regions in HRI. We conduct extensive experiments involving 28 VLMs, including Gemini 2.0 Flash and GPT-4o. Experiments on HRScene show that current VLMs achieve an average accuracy of around 50% on real-world tasks, revealing significant gaps in HRI understanding. Results on synthetic datasets reveal that VLMs struggle to effectively utilize HRI regions, showing significant Regional Divergence and lost-in-middle, shedding light on future research.

</details>

---

## 432. Layer-wise Vision Injection with Disentangled Attention for Efficient LVLMs

- [ ] Layer-wise Vision Injection with Disentangled Attention for Efficient LVLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Layer-wise_Vision_Injection_with_Disentangled_Attention_for_Efficient_LVLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Layer-wise_Vision_Injection_with_Disentangled_Attention_for_Efficient_LVLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Benefiting from recent advancements in large language models and modality alignment techniques, existing Large Vision-Language Models(LVLMs) have achieved prominent performance across a wide range of scenarios. However, the excessive computational complexity limits the widespread use of these models in practical applications. We argue that one main bottleneck in computational complexity is caused by the involvement of redundant vision sequences in model computation. This is inspired by a reassessment of the efficiency of vision and language information transmission in the language decoder of LVLMs. Then, we propose a novel vision-language interaction mechanism called Layer-wise Vision Injection with Disentangled Attention (LVIDA). In LVIDA, only the language sequence undergoes full forward propagation, while the vision sequence interacts with the language at specific stages within each language decoder layer. It is striking that our approach significantly reduces computational complexity with minimal performance loss. Specifically, LVIDA achieves approximately a 10x reduction in the computational cost of the language decoder across multiple LVLM models while maintaining comparable performance. Project Page: https://xuange923.github.io/LVIDA/

</details>

---

## 433. Learning Beyond Still Frames: Scaling Vision-Language Models with Video

- [ ] Learning Beyond Still Frames: Scaling Vision-Language Models with Video | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Learning_Beyond_Still_Frames_Scaling_Vision-Language_Models_with_Video_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Learning_Beyond_Still_Frames_Scaling_Vision-Language_Models_with_Video_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-quality image-text data is critical in enhancing Vision-Language Models (VLMs), but traditional image-based pretraining approaches face limitations. These methods are resource-intensive, relying on curated, high-quality interleaved data that is costly and challenging to collect at scale. Additionally, while such datasets improve static image-text understanding, they fail to develop the temporal and motion comprehension needed for video understanding. To address these gaps, we propose incorporating video pretraining into VLMs to improve the model's ability to capture temporal dynamics and general visual perception, which requires reconciling spatial redundancy with strict temporal causality. Therefore, we propose Causal Hierarchical Aggregation to separate computation-heavy spatial encoding from lightweight temporal propagation and construct hierarchical receptive fields at varying granularities. As we scale video context to more than 100 billion tokens, our method excels in high throughput and state-of-the-art performances on both Image and Video understanding, as shown in Figure 1, providing a scalable solution to enhance multimodal learning in dynamic contexts.

</details>

---

## 434. Learning Visual Proxy for Compositional Zero-Shot Learning

- [ ] Learning Visual Proxy for Compositional Zero-Shot Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Learning_Visual_Proxy_for_Compositional_Zero-Shot_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Learning_Visual_Proxy_for_Compositional_Zero-Shot_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions by leveraging knowledge from seen compositions. Existing methods typically align textual prototypes with visual features using Vision-Language Models (VLMs), but they face two key limitations: (1) modality gaps hinder the ability to distinguish semantically similar attribute-object pairs, and (2) textual prototypes alone lack the fine-grained visual cues needed for accurate recognition. To address these challenges, we propose Visual Proxy Learning, a method that reduces modality gaps and enhances compositional generalization by initializing visual proxies for attributes, objects, and their compositions from text representations and optimizing the visual space to better capture fine-grained visual cues. To further strengthen cross-modal understanding, we introduce Cross-Modal Joint Learning (CMJL), which enforces consistency between text-image embeddings and fine-grained visual representations. This dual strategy improves generalization to unseen compositions and enhances the discrimination of similar pairs. Extensive experiments demonstrate that our method achieves state-of-the-art performance in closed-world settings and competitive results in open-world scenarios across four CZSL benchmarks, validating its effectiveness in improving compositional generalization.

</details>

---

## 435. Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis

- [ ] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Oasis_One_Image_is_All_You_Need_for_Multimodal_Instruction_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Oasis_One_Image_is_All_You_Need_for_Multimodal_Instruction_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The success of multi-modal large language models (MLLMs) has been largely attributed to the large-scale training data. However, the training data of many MLLMs is unavailable due to privacy concerns. The expensive and labor-intensive process of collecting multi-modal data further exacerbates the problem. Is it possible to synthesize multi-modal training data automatically without compromising diversity and quality? In this paper, we propose a new method, Oasis, to synthesize high-quality multi-modal data with only images. Oasis breaks through traditional methods by prompting only images to the MLLMs, thus extending the data diversity by a large margin. Our method features a delicate quality control method which ensures the data quality. We collected over 500k data and conducted incremental experiments on LLaVA-NeXT. Extensive experiments demonstrate that our method can significantly improve the performance of MLLMs. The image-based synthesis also allows us to focus on the specific-domain ability of MLLMs. Code and data will be publicly available.

</details>

---

## 436. Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs

- [ ] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Q-Frame_Query-aware_Frame_Selection_and_Multi-Resolution_Adaptation_for_Video-LLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Q-Frame_Query-aware_Frame_Selection_and_Multi-Resolution_Adaptation_for_Video-LLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated significant success in visual understanding tasks. However, challenges persist in adapting these models for video comprehension due to the large volume of data and temporal complexity. Existing Video-LLMs using uniform frame sampling often struggle to capture the query-related crucial spatiotemporal clues of videos effectively. In this paper, we introduce Q-Frame, a novel approach for adaptive frame selection and multi-resolution scaling tailored to the video's content and the specific query. Q-Frame employs a training-free, plug-and-play strategy generated by a text-image matching network like CLIP, utilizing the Gumbel-Max trick for efficient frame selection. Q-Frame allows Video-LLMs to process more frames without exceeding computational limits, thereby preserving critical temporal and spatial information. We demonstrate Q-Frame's effectiveness through extensive experiments on benchmark datasets, including MLVU, LongVideoBench, and Video-MME, illustrating its superiority over existing methods and its applicability across various video understanding tasks.

</details>

---

## 437. RANKCLIP: Ranking-Consistent Language-Image Pretraining

- [ ] RANKCLIP: Ranking-Consistent Language-Image Pretraining | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_RANKCLIP_Ranking-Consistent_Language-Image_Pretraining_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_RANKCLIP_Ranking-Consistent_Language-Image_Pretraining_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Self-supervised contrastive learning models, such as CLIP, have set new benchmarks for vision-language models in many downstream tasks. However, their dependency on rigid one-to-one mappings overlooks the complex and often multifaceted relationships between and within texts and images. To this end, we introduce RankCLIP, a novel pretraining method that extends beyond the rigid one-to-one matching framework of CLIP and its variants. By extending the traditional pair-wise loss to list-wise, and leveraging both in-modal and cross-modal ranking consistency, RankCLIP improves the alignment process, enabling it to capture the nuanced many-to-many relationships between and within each modality. Through comprehensive experiments, we demonstrate the effectiveness of RankCLIP in various downstream tasks, notably achieving significant gains in zero-shot classifications over state-of-the-art methods, underscoring the importance of this enhanced learning process.

</details>

---

## 438. R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization

- [ ] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_R1-VL_Learning_to_Reason_with_Multimodal_Large_Language_Models_via_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_R1-VL_Learning_to_Reason_with_Multimodal_Large_Language_Models_via_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are.In this work, we aim to enhance the MLLMs' reasoning ability beyond passively imitating positive reasoning paths. To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective and dense step-wise rewarding. Specifically, StepGRPO introduces two novel rule-based reasoning rewards: Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR). StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRAR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy. With the proposed step-wise reward mechanisms, we introduce R1-VL, a series of of MLLMs with outstanding capabilities in step-by-step reasoning. Extensive experiments over 8 benchmarks demonstrate the superiority of the proposed StepGRPO.

</details>

---

## 439. RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation

- [ ] RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_RoBridge_A_Hierarchical_Architecture_Bridging_Cognition_and_Execution_for_General_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_RoBridge_A_Hierarchical_Architecture_Bridging_Cognition_and_Execution_for_General_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Operating robots in open-ended scenarios with diverse tasks is a crucial research and application direction in robotics. While recent progress in natural language processing and large multimodal models has enhanced robots' ability to understand complex instructions, robot manipulation still faces the procedural skill dilemma and the declarative skill dilemma in open environments. Existing methods often compromise cognitive and executive capabilities. To address these challenges, in this paper, we propose RoBridge, a hierarchical intelligent architecture for general robotic manipulation. It consists of a high-level cognitive planner (HCP) based on a large-scale pre-trained vision-language model (VLM), an invariant operable representation (IOR) serving as a symbolic bridge, and a generalist embodied agent (GEA). RoBridge maintains the declarative skill of VLM and unleashes the procedural skill of reinforcement learning, effectively bridging the gap between cognition and execution. RoBridge demonstrates significant performance improvements over existing baselines, achieving a 75% success rate on new tasks and an 83% average success rate in sim-to-real generalization using only five real-world data samples per task. This work represents a significant step towards integrating cognitive reasoning with physical execution in robotic systems, offering a new paradigm for general robotic manipulation.

</details>

---

## 440. Scaling Omni-modal Pretraining with Multimodal Context: Advancing Universal Representation Learning Across Modalities

- [ ] Scaling Omni-modal Pretraining with Multimodal Context: Advancing Universal Representation Learning Across Modalities | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Scaling_Omni-modal_Pretraining_with_Multimodal_Context_Advancing_Universal_Representation_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Scaling_Omni-modal_Pretraining_with_Multimodal_Context_Advancing_Universal_Representation_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This work introduces Multimodal Context (MiCo), a scalable pretraining framework designed to advance omni-modal intelligence--an AI system capable of understanding and learning from multiple modalities to achieve universal representation learning. MiCo allows for efficient scaling of both the number of modalities and the volume of data, along with model parameters, during the pretraining phase. We evaluate the pretrained models across a diverse set of tasks, including: (i) single-modality perception benchmarks covering 10 distinct modalities, (ii) 25 cross-modal tasks spanning retrieval, question-answering, and captioning, and (iii) 18 large-scale multimodal language model benchmarks. MiCo consistently delivers state-of-the-art results, setting 37 new benchmarks across these tasks. The pretrained models, along with the collected datasets and codebase, will be made publicly available to support the development of omni-modal intelligence and broader research in multimodal learning.

</details>

---

## 441. Trade-offs in Image Generation: How Do Different Dimensions Interact?

- [ ] Trade-offs in Image Generation: How Do Different Dimensions Interact? | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Trade-offs_in_Image_Generation_How_Do_Different_Dimensions_Interact_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Trade-offs_in_Image_Generation_How_Do_Different_Dimensions_Interact_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Model performance in text-to-image (T2I) and image-to-image (I2I) generation often depends on multiple aspects, including quality, alignment, diversity, and robustness. However, models' complex trade-offs among these dimensions have been rarely explored due to (1) the lack of datasets that allow fine-grained quantification of these trade-offs, and (2) using a single metric for multiple dimensions. To address this gap, we introduce TRIG-Bench (Trade-offs in Image Generation), which spans 10 dimensions (Realism, Originality, Aesthetics, Content, Relation, Style, Knowledge, Ambiguity, Toxicity and Bias), contains over 40,200 samples, and covers 132 Pairwise Dimensional Subsets. Furthermore, we develop TRIGScore,a VLM-as-judge metric that automatically adapts to various dimensions. Based on this, we evaluate 14 cutting-edge models across T2I and I2I tasks. In addition, we propose the Relation Recognition System and generate the Dimension Trade-off Map (DTM), which visualizes model-specific capability trade-offs. Our experiments demonstrate that DTM consistently provides a comprehensive understanding of the trade-offs between dimensions for each type of generation models. Notably, after fine-tuning on DTM, the model's dimension-specific impact is mitigated, and overall performance is enhanced. Code is available at: https://github.com/fesvhtr/TRIG.

</details>

---

## 442. UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement

- [ ] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_UPRE_Zero-Shot_Domain_Adaptation_for_Object_Detection_via_Unified_Prompt_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_UPRE_Zero-Shot_Domain_Adaptation_for_Object_Detection_via_Unified_Prompt_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot domain adaptation (ZSDA) presents substantial challenges due to the lack of images in the target domain. Previous approaches leverage Vision-Language Models (VLMs) to tackle this challenge, exploiting their zero-shot learning capabilities. However, these methods primarily address domain distribution shifts and overlook the misalignment between the detection task and VLMs, which rely on manually crafted prompts. To overcome these limitations, we propose the unified prompt and representation enhancement (UPRE) framework, which jointly optimizes both textual prompts and visual representations. Specifically, our approach introduces a multi-view domain prompt that combines linguistic domain priors with detection-specific knowledge, and a visual representation enhancement module that produces domain style variations. Furthermore, we introduce multi-level enhancement strategies, including relative domain distance and positive-negative separation, which align multi-modal representations at the image level and capture diverse visual representations at the instance level, respectively. Extensive experiments conducted on nine benchmark datasets demonstrate the superior performance of our framework in ZSDA detection scenarios.

</details>

---

## 443. Unified Multimodal Understanding via Byte-Pair Visual Encoding

- [ ] Unified Multimodal Understanding via Byte-Pair Visual Encoding | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Unified_Multimodal_Understanding_via_Byte-Pair_Visual_Encoding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Unified_Multimodal_Understanding_via_Byte-Pair_Visual_Encoding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have made significant progress in vision-language understanding, yet effectively aligning different modalities remains a fundamental challenge. We present a framework that unifies multimodal understanding by applying byte-pair encoding to visual tokens. Unlike conventional approaches that rely on modality-specific encoders, our method directly incorporates structural information into visual tokens, mirroring successful tokenization strategies in text-only language models. We introduce a priority-guided encoding scheme that considers both frequency and spatial consistency, coupled with a multi-stage training procedure based on curriculum-driven data composition. These enhancements enable the transformer model to better capture cross-modal relationships and reason with visual information. Comprehensive experiments demonstrate improved performance across diverse vision-language tasks. By bridging the gap between visual and textual representations, our approach contributes to the advancement of more capable and efficient multimodal foundation models.

</details>

---

## 444. VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks

- [ ] VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VLABench_A_Large-Scale_Benchmark_for_Language-Conditioned_Robotics_Manipulation_with_Long-Horizon_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VLABench_A_Large-Scale_Benchmark_for_Language-Conditioned_Robotics_Manipulation_with_Long-Horizon_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

General-purposed embodied agents are designed to understand the users' natural instructions or intentions and act precisely to complete universal tasks. Recently, methods based on foundation models especially Vision-Language-Action models (VLAs) have shown a substantial potential to solve language-conditioned manipulation (LCM) tasks well. However, existing benchmarks do not adequately meet the needs of VLAs and relative algorithms. To better define such general-purpose tasks in the context of LLMs and advance the research in VLAs, we present VLABench, an open-source benchmark for evaluating universal LCM task learning. VLABench provides 100 carefully designed categories of tasks, with strong randomization in each category of task and a total of 2000+ objects. VLABench stands out from previous benchmarks in four key aspects: 1) tasks requiring world knowledge and common sense transfer, 2) natural language instructions with implicit human intentions rather than templates, 3) long-horizon tasks demanding multi-step reasoning, and 4) evaluation of both action policies and language model capabilities. The benchmark assesses multiple competencies including understanding of mesh&texture, spatial relationship, semantic instruction, physical laws, knowledge transfer and reasoning, etc. To support the downstream finetuning, we provide high-quality training data collected via an automated framework incorporating heuristic skills and prior information. The experimental results indicate that both the current state-of-the-art pretrained VLAs and the workflow based on VLMs face challenges in our tasks.

</details>

---

## 445. VLDrive: Vision-Augmented Lightweight MLLMs for Efficient Language-grounded Autonomous Driving

- [ ] VLDrive: Vision-Augmented Lightweight MLLMs for Efficient Language-grounded Autonomous Driving | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VLDrive_Vision-Augmented_Lightweight_MLLMs_for_Efficient_Language-grounded_Autonomous_Driving_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VLDrive_Vision-Augmented_Lightweight_MLLMs_for_Efficient_Language-grounded_Autonomous_Driving_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in language-grounded autonomous driving have been significantly promoted by the sophisticated cognition and reasoning capabilities of large language models (LLMs). However, current LLM-based approaches encounter critical challenges: (1) Failure analysis reveals that frequent collisions and obstructions, stemming from limitations in visual representations, remain primary obstacles to robust driving performance. (2) The substantial parameters of LLMs pose considerable deployment hurdles. To address these limitations, we introduce VLDrive, a novel approach featuring a lightweight MLLM architecture with enhanced vision components. VLDrive achieves compact visual tokens through innovative strategies, including cycle-consistent dynamic visual pruning and memory-enhanced feature aggregation. Furthermore, we propose a distance-decoupled instruction attention mechanism to improve joint visual-linguistic feature learning, particularly for long-range visual tokens. Extensive experiments conducted in the CARLA simulator demonstrate VLDrive's effectiveness. Notably, VLDrive achieves state-of-the-art driving performance while reducing parameters by 81% (from 7B to 1.3B), yielding substantial driving score improvements of 15.4%, 16.8%, and 7.6% at tiny, short, and long distances, respectively, in closed-loop evaluations. Code is available at https://github.com/ReaFly/VLDrive.

</details>

---

## 446. VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning

- [ ] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VTimeCoT_Thinking_by_Drawing_for_Video_Temporal_Grounding_and_Reasoning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VTimeCoT_Thinking_by_Drawing_for_Video_Temporal_Grounding_and_Reasoning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, video question answering based on multimodal large language models (MLLM) has garnered considerable attention, due to the benefits from the substantial advancements in LLMs. However, these models have a notable deficiency in the domains of video temporal grounding and reasoning, posing challenges to the development of effective real-world video understanding systems. Inspired by how humans use video players to interact with the progress bar for video comprehension, we introduce VTimeCoT, a simple yet effective training-free framework, designed for high-performance video grounding and reasoning. The proposed framework incorporates two novel visual tools of the progress bar: a plug-and-play progress bar integration tool and a high-efficiency highlighting tool. In addition, to address the limitations of conventional text-based chain-of-thought (CoT) approaches, we introduce a visuotemporal CoT process that integrates cross-modality reasoning across both video and text. Our approach demonstrates significant performance improvements on both Qwen2VL-7B and GPT4o baselines in tasks of video temporal grounding and reasoning-based question answering. Finally, we showcase that the proposed framework achieves a compositional and interpretable reasoning process. Project page: \href https://vtimecot.github.io  https://vtimecot.github.io .

</details>

---

## 447. VideoAds for Fast-Paced Video Understanding

- [ ] VideoAds for Fast-Paced Video Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VideoAds_for_Fast-Paced_Video_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VideoAds_for_Fast-Paced_Video_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Advertisement videos serve as a rich and valuable source of purpose-driven information, encompassing high-quality visual, textual, and contextual cues designed to engage viewers. They are often more complex than general videos of similar duration due to their structured narratives and rapid scene transitions, posing significant challenges to multi-modal large language models (MLLMs). In this work, we introduce VideoAds, the first dataset tailored for benchmarking the performance of MLLMs on advertisement videos. VideoAds comprises well-curated advertisement videos with complex temporal structures, accompanied by manually annotated diverse questions across three core tasks: visual finding, video summary, and visual reasoning. We propose a quantitative measure to compare VideoAds against existing benchmarks in terms of video complexity. Through extensive experiments, we find that Qwen2.5-VL-72B, an open-source MLLM, achieves 73.35% accuracy on VideoAds, outperforms GPT-4o (66.82%) and Gemini-1.5 Pro (69.66%); the two proprietary models especially fall behind the open-source model in video summarization and reasoning, but perform the best in visual finding. Gemini-2.5 Pro leads with an accuracy of 80.04%. Notably, human experts easily achieve a remarkable accuracy of 94.27%. These results underscore the necessity of advancing MLLMs' temporal modeling capabilities and highlight VideoAds as a potentially pivotal benchmark for future research in understanding video that requires high FPS sampling. The dataset and evaluation code will be publicly available at https://videoadsbenchmark.netlify.app.

</details>

---

## 448. Context-Aware Academic Emotion Dataset and Benchmark

- [ ] Context-Aware Academic Emotion Dataset and Benchmark | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Context-Aware_Academic_Emotion_Dataset_and_Benchmark_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Context-Aware_Academic_Emotion_Dataset_and_Benchmark_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Academic emotion analysis plays a crucial role in evaluating students' engagement and cognitive states during the learning process. This paper addresses the challenge of automatically recognizing academic emotions through facial expressions in real-world learning environments. While significant progress has been made in facial expression recognition for basic emotions, academic emotion recognition remains underexplored, largely due to the scarcity of publicly available datasets. To bridge this gap, we introduce RAER, a novel dataset comprising approximately 2,700 video clips collected from around 140 students in diverse, natural learning contexts such as classrooms, libraries, laboratories, and dormitories, covering both classroom sessions and individual study. Each clip was annotated independently by approximately ten annotators using two distinct sets of academic emotion labels with varying granularity, enhancing annotation consistency and reliability. To our knowledge, RAER is the first dataset capturing diverse natural learning scenarios. Observing that annotators naturally consider context cues--such as whether a student is looking at a phone or reading a book--alongside facial expressions, we propose CLIP-CAER (CLIP-based Context-aware Academic Emotion Recognition). Our method utilizes learnable text prompts within the vision-language model CLIP to effectively integrate facial expression and context cues from videos. Experimental results demonstrate that CLIP-CAER substantially outperforms state-of-the-art video-based facial expression recognition methods, which are primarily designed for basic emotions, emphasizing the crucial role of context in accurately recognizing academic emotions. Project page: https://zgsfer.github.io/CAER

</details>

---

## 449. p-MoD: Building Mixture-of-Depths MLLMs via Progressive Ratio Decay

- [ ] p-MoD: Building Mixture-of-Depths MLLMs via Progressive Ratio Decay | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_p-MoD_Building_Mixture-of-Depths_MLLMs_via_Progressive_Ratio_Decay_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_p-MoD_Building_Mixture-of-Depths_MLLMs_via_Progressive_Ratio_Decay_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the remarkable performance of multimodal large language models (MLLMs) across diverse tasks, the substantial training and inference costs impede their advancement. In this paper, we propose p-MoD, an efficient MLLM architecture that significantly reduces training and inference costs while maintaining model performance. The majority of computation in MLLMs stems from the overwhelming volume of vision tokens processed by the transformer-based LLM. Accordingly, we leverage the Mixture-of-Depths (MoD) mechanism, where each LLM layer selects essential vision tokens to process while skipping redundant ones. However, integrating MoD into MLLMs is non-trivial. To address the challenges of training and inference stability as well as limited training data, we adapt the MoD module with two novel designs: tanh-gated weight normalization (TanhNorm) and symmetric token reweighting (STRing). Moreover, we observe that vision tokens exhibit higher redundancy in deeper layers and thus design a progressive ratio decay (PRD) strategy, which gradually reduces the token retention ratio layer by layer, employing a shifted cosine schedule. This crucial design fully unleashes the potential of MoD, significantly boosting the efficiency and performance of our models. Extensive experiments on two baseline models across 15 benchmarks show that our model matches or even surpasses the performance of corresponding baselines, while requiring only 55.6% TFLOPs and 53.7% KV cache storage during inference, and 77.7% GPU hours during training.

</details>

---

## 450. DisCo: Towards Distinct and Coherent Visual Encapsulation in Video MLLMs

- [ ] DisCo: Towards Distinct and Coherent Visual Encapsulation in Video MLLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_DisCo_Towards_Distinct_and_Coherent_Visual_Encapsulation_in_Video_MLLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_DisCo_Towards_Distinct_and_Coherent_Visual_Encapsulation_in_Video_MLLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In video Multimodal Large Language Models (video MLLMs), the visual encapsulation process plays a pivotal role in converting video contents into representative tokens for LLM input. While linear projectors are widely employed for encapsulation, they introduce semantic indistinctness and temporal incoherence when applied to videos. Conversely, the structure of resamplers shows promise in tackling these challenges, but an effective solution remains unexplored. Drawing inspiration from resampler structures, we introduce DisCo, a novel visual encapsulation method designed to yield semantically distinct and temporally coherent visual tokens for video MLLMs. DisCo integrates two key components: (1) A Visual Concept Discriminator (VCD) module, assigning unique semantics for visual tokens by associating them in pair with discriminative concepts in the video. (2) A Temporal Focus Calibrator (TFC) module, ensuring consistent temporal focus of visual tokens to video elements across every video frame. Through extensive experiments on multiple video MLLM frameworks, we demonstrate that DisCo remarkably outperforms previous state-of-the-art methods across a variety of video understanding benchmarks, while also achieving higher token efficiency thanks to the reduction of semantic indistinctness.

</details>

---

## 451. Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency

- [ ] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Jailbreaking_Multimodal_Large_Language_Models_via_Shuffle_Inconsistency_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Jailbreaking_Multimodal_Large_Language_Models_via_Shuffle_Inconsistency_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have achieved impressive performance and have been put into practical use in commercial applications, but they still have potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety mechanisms and discover MLLMs' potential risks. Existing MLLMs' jailbreak methods often bypass the model's safety mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research, we empirically find that there exists a Shuffle Inconsistency between MLLMs' comprehension ability and safety ability for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand the shuffled harmful text-image instructions well. However, they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization method to select the most harmful shuffled inputs based on the feedback of the toxic judge model. A series of experiments show that SI-Attack can effectively improve the attack's performance on three benchmarks for both open-source and closed-source MLLMs.

</details>

---

## 452. HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding

- [ ] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_HIS-GPT_Towards_3D_Human-In-Scene_Multimodal_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_HIS-GPT_Towards_3D_Human-In-Scene_Multimodal_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We propose a new task to benchmark human-in-scene understanding for embodied agents: Human-In-Scene Question Answering (HIS-QA). Given a human motion within a 3D scene, HIS-QA requires the agent to comprehend human states and behaviors, reason about its surrounding environment, and answer human-related questions within the scene. To support this new task, we present HIS-Bench, a multimodal benchmark that systematically evaluates HIS understanding across a broad spectrum, from basic perception to commonsense reasoning and planning. Our evaluation of various vision-language models on HIS-Bench reveals significant limitations in their ability to handle HIS-QA tasks. To this end, we propose HIS-GPT, the first foundation model for HIS understanding. HIS-GPT integrates 3D scene context and human motion dynamics into large language models while incorporating specialized mechanisms to capture human-scene interactions. Extensive experiments demonstrate that HIS-GPT sets a new state-of-the-art on HIS-QA tasks. We hope this work inspires future research of human behavior analysis in 3D scenes, advancing embodied AI and world models.

</details>

---

## 453. One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models

- [ ] One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_One_Object_Multiple_Lies_A_Benchmark_for_Cross-task_Adversarial_Attack_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_One_Object_Multiple_Lies_A_Benchmark_for_Cross-task_Adversarial_Attack_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unified vision-language models (VLMs) have recently shown remarkable progress, enabling a single model to flexibly address diverse tasks through different instructions within a shared computational architecture. This instruction-based control mechanism creates unique security challenges, as adversarial inputs must remain effective across multiple task instructions that may be unpredictably applied to process the same malicious content. In this paper, we introduce CrossVLAD, a new benchmark dataset carefully curated from MSCOCO with GPT-4-assisted annotations for systematically evaluating cross-task adversarial attacks on unified VLMs. CrossVLAD centers on the object-change objective--consistently manipulating a target object's classification across four downstream tasks--and proposes a novel success rate metric that measures simultaneous misclassification across all tasks, providing a rigorous evaluation of adversarial transferability. To tackle this challenge, we present CRAFT (Cross-task Region-based Attack Framework with Token-alignment), an efficient region-centric attack method. Extensive experiments on Florence-2 and other popular unified VLMs demonstrate that our method outperforms existing approaches in both overall cross-task attack performance and targeted object-change success rates, highlighting its effectiveness in adversarially influencing unified VLMs across diverse tasks.

</details>

---

## 454. PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting

- [ ] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_PhysSplat_Efficient_Physics_Simulation_for_3D_Scenes_via_MLLM-Guided_Gaussian_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_PhysSplat_Efficient_Physics_Simulation_for_3D_Scenes_via_MLLM-Guided_Gaussian_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in 3D generation models have opened new possibilities for simulating dynamic 3D object movements and customizing behaviors, yet creating this content remains challenging. Current methods often require manual assignment of precise physical properties for simulations or rely on video generation models to predict them, which is computationally intensive. In this paper, we rethink the usage of multi-modal large language model (MLLM) in physics-based simulation, and present PhysSplat, a physics-based approach that efficiently endows static 3D objects with interactive dynamics. We begin with detailed scene reconstruction and object-level 3D open-vocabulary segmentation, progressing to multi-view image in-painting. Inspired by human visual reasoning, we propose MLLM-based Physical Property Perception (MLLM-P3) to predict the mean physical properties of objects in a zero-shot manner. The Material Property Distribution Prediction model (MPDP) then estimates physical property distributions via geometry-conditioned probabilistic sampling of MLLM-P3 outputs, reformulating the problem as probability distribution estimation to reduce computational costs. Finally, we simulate objects in 3D scenes with particles sampled via the Physical-Geometric Adaptive Sampling (PGAS) strategy, efficiently capturing complex deformations and significantly reducing computational costs. Extensive experiments and user studies demonstrate that our PhysSplat achieves more realistic motion than state-of-the-art methods within 2 minutes on a single GPU.

</details>

---

## 455. Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information

- [ ] Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Pi-GPS_Enhancing_Geometry_Problem_Solving_by_Unleashing_the_Power_of_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Pi-GPS_Enhancing_Geometry_Problem_Solving_by_Unleashing_the_Power_of_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Geometry problem solving has garnered increasing attention due to its potential applications in intelligent education field. Inspired by the observation that text often introduces ambiguities that diagrams can clarify, this paper presents Pi-GPS, a novel framework that unleashes the power of diagrammatic information to resolve textual ambiguities, an aspect largely overlooked in prior research. Specifically, we design a micro module comprising a rectifier and verifier: the rectifier employs MLLMs to disambiguate text based on the diagrammatic context, while the verifier ensures the rectified output adherence to geometric rules, mitigating model hallucinations. Additionally, we explore the impact of LLMs in theorem predictor based on the disambiguated formal language. Empirical results demonstrate that Pi-GPS surpasses state-of-the-art models, achieving a nearly 10% improvement on Geometry3K over prior neural-symbolic approaches. We hope this work highlights the significance of resolving textual ambiguity in multimodal mathematical reasoning, a crucial factor limiting performance.

</details>

---

## 456. Training-free Generation of Temporally Consistent Rewards from VLMs

- [ ] Training-free Generation of Temporally Consistent Rewards from VLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Training-free_Generation_of_Temporally_Consistent_Rewards_from_VLMs_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Training-free_Generation_of_Temporally_Consistent_Rewards_from_VLMs_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language models (VLMs) have significantly improved performance in embodied tasks such as goal decomposition and visual comprehension. However, providing accurate rewards for robotic manipulation without fine-tuning VLMs remains challenging due to the absence of domain-specific robotic knowledge in pre-trained datasets and high computational costs that hinder real-time applicability. To address this, we propose T2-VLM, a novel training-free, temporally consistent framework that generates accurate rewards through tracking the status changes in VLM-derived subgoals. Specifically, our method first queries the VLM to establish spatially aware subgoals and an initial completion estimate before each round of interaction. We then employ a Bayesian tracking algorithm to update the goal completion status dynamically, using subgoal hidden states to generate structured rewards for reinforcement learning (RL) agents. This approach enhances long-horizon decision-making and improves failure recovery capabilities with RL. Extensive experiments indicate that T2-VLM achieves state-of-the-art performance in two robot manipulation benchmarks, demonstrating superior reward accuracy with reduced computation consumption. We believe our approach not only advances reward generation techniques but also contributes to the broader field of embodied AI. Project website: https://t2-vlm.github.io/.

</details>

---

## 457. Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization

- [ ] Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization | https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Unsupervised_Visual_Chain-of-Thought_Reasoning_via_Preference_Optimization_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Unsupervised_Visual_Chain-of-Thought_Reasoning_via_Preference_Optimization_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chain-of-thought (CoT) reasoning greatly improves the interpretability and problem-solving abilities of multimodal large language models (MLLMs). However, existing approaches are focused on text CoT, limiting their ability to leverage visual cues. Visual CoT remains underexplored, and the only work is based on supervised fine-tuning (SFT) that relies on extensive labeled bounding-box data and is hard to generalize to unseen cases. In this paper, we introduce Unsupervised Visual CoT (UV-CoT), a novel framework for image-level CoT reasoning via preference optimization. UV-CoT performs preference comparisons between model-generated bounding boxes (one is preferred and the other is dis-preferred), eliminating the need for bounding-box annotations. We get such preference data by introducing an automatic data generation pipeline. Given an image, our target MLLM (e.g., LLaVA-1.5-7B) generates seed bounding boxes using a template prompt and then answers the question using each bounded region as input. An evaluator MLLM (e.g., OmniLLM-12B) ranks the responses, and these rankings serve as supervision to train the target MLLM with UV-CoT by minimizing negative log-likelihood losses. By emulating human perception--identifying key regions and reasoning based on them--UV-CoT can improve visual comprehension, particularly in spatial reasoning tasks where textual descriptions alone fall short. Our experiments on six datasets demonstrate the superiority of UV-CoT, compared to the state-of-the-art textual and visual CoT methods. Our zero-shot testing on three unseen datasets shows the strong generalization of UV-CoT. The implementation code is available in the Appendix.

</details>

---

## 458. Hierarchical Cross-modal Prompt Learning for Vision-Language Models

- [ ] Hierarchical Cross-modal Prompt Learning for Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Hierarchical_Cross-modal_Prompt_Learning_for_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Hierarchical_Cross-modal_Prompt_Learning_for_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained Vision-Language Models (VLMs) such as CLIP have shown excellent generalization abilities. However, adapting these large-scale models to downstream tasks while preserving their generalization capabilities remains challenging. Although prompt learning methods have shown promise, they suffer from two fundamental bottlenecks that limit generalization: (a) modality isolation, and (b) hierarchical semantic decay. To address these limitations, we propose HiCroPL, a Hierarchical Cross-modal Prompt Learning framework that establishes bidirectional knowledge flow between text and vision modalities, enabling them to refine their semantics mutually. HiCroPL routes knowledge flows by leveraging the complementary strengths of text and vision. In early layers, text prompts inject relatively clear semantics into visual prompts through a hierarchical knowledge mapper, enhancing the representation of low-level visual semantics. In later layers, visual prompts encoding specific task-relevant objects flow back to refine text prompts, enabling deeper alignment. Crucially, our hierarchical knowledge mapper allows representations at multi-scales to be fused, ensuring that deeper representations retain transferable shallow semantics thereby enhancing generalization. We further introduce a lightweight layer-specific knowledge proxy to enable efficient cross-modal interactions. Extensive evaluations across four tasks demonstrate HiCroPL's superior performance, achieving state-of-the-art results on 11 benchmarks with significant improvements. Code is available at: https://github.com/zzeoZheng/HiCroPL.

</details>

---

## 459. ViLLa: Video Reasoning Segmentation with Large Language Model

- [ ] ViLLa: Video Reasoning Segmentation with Large Language Model | https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_ViLLa_Video_Reasoning_Segmentation_with_Large_Language_Model_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_ViLLa_Video_Reasoning_Segmentation_with_Large_Language_Model_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent efforts in video reasoning segmentation (VRS) integrate large language models (LLMs) with perception models to localize and track objects via textual instructions, achieving barely satisfactory results in simple scenarios. However, they struggled to discriminate and deduce the objects from user queries in more real-world scenes featured by long durations, multiple objects, rapid motion, and heavy occlusions. In this work, we analyze the underlying causes of these limitations, and present **ViLLa**: **Vi**deo reasoning segmentation with **L**arge **La**nguage Model. Remarkably, our ViLLa manages to tackle these challenges through multiple core innovations: (1) a context synthesizer that dynamically encodes the user intent with video contexts for accurate reasoning, resolving ambiguities in complex queries, and (2) a hierarchical temporal synchronizer that disentangles multi-object interactions across complex temporal scenarios by modelling multi-object interactions at local and global temporal scales. To enable efficient processing of long videos, ViLLa incorporates (3) a key segment sampler that adaptively partitions long videos into shorter but semantically dense segments for less redundancy. What's more, to promote research in this unexplored area, we construct a VRS benchmark, **VideoReasonSeg**, featuring different complex scenarios. Our model also exhibits impressive state-of-the-art results on VideoReasonSeg, Ref-YouTube-VOS, Ref-DAVIS17, MeViS, and ReVOS. Both quantitative and qualitative experiments demonstrate that our method effectively enhances video reasoning segmentation capabilities for multimodal LLMs.

</details>

---

## 460. Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context

- [ ] Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context | https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Why_LVLMs_Are_More_Prone_to_Hallucinations_in_Longer_Responses_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Why_LVLMs_Are_More_Prone_to_Hallucinations_in_Longer_Responses_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have made significant progress in recent years but are also prone to hallucination issues. They exhibit more hallucinations in longer, free-form responses, often attributed to accumulated uncertainties. In this paper, we ask: Does increased hallucination result solely from length-induced errors, or is there a deeper underlying mechanism? After a series of preliminary experiments and findings, we suggest that the risk of hallucinations is not caused by length itself but by the increased reliance on context for coherence and completeness in longer responses. Building on these insights, we propose a novel "induce-detect-suppress" framework that actively induces hallucinations through deliberately designed contexts, leverages induced instances for early detection of high-risk cases, and ultimately suppresses potential object-level hallucinations during actual decoding. Our approach achieves consistent, significant improvements across all benchmarks, demonstrating its efficacy. The strong detection and improved hallucination mitigation not only validate our framework but, more importantly, re-validate our hypothesis on context. Rather than solely pursuing performance gains, this study aims to provide new insights and serves as a first step toward a deeper exploration of hallucinations in LVLMs' longer responses.

</details>

---

## 461. AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

- [ ] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning | https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_AIM_Adaptive_Inference_of_Multi-Modal_LLMs_via_Token_Merging_and_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_AIM_Adaptive_Inference_of_Multi-Modal_LLMs_via_Token_Merging_and_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that our method substantially reduces computation load (e.g., a 7-fold reduction in FLOPs) while preserving the performance of video and image LLMs. Further, at a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., +4.6 on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs. Our code is available at https://github.com/LaVi-Lab/AIM.

</details>

---

## 462. Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition

- [ ] Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition | https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Lyra_An_Efficient_and_Speech-Centric_Framework_for_Omni-Cognition_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Lyra_An_Efficient_and_Speech-Centric_Framework_for_Omni-Cognition_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As Multi-modal Large Language Models (MLLMs) evolve, expanding beyond single-domain capabilities is essential to meet the demands for more versatile and efficient AI. However, previous omni-models have insufficiently explored speech, neglecting its integration with multi-modality. We introduce Lyra, an efficient MLLM that enhances multi-modal abilities, including advanced long speech comprehension, sound understanding, cross-modality efficiency, and seamless speech interaction. To achieve efficiency and speech-centric capabilities, Lyra employs three strategies: (1) leveraging existing open-source large models and a proposed multi-modality LoRA to reduce training costs and data requirements; (2) using a latent multi-modality regularizer and extractor to strengthen the relationship between speech and other modalities, thereby enhancing model performance; and (3) constructing a high-quality, extensive dataset that includes 1.5M multi-modal (language, vision, audio) data samples and 12K long speech samples, enabling Lyra to handle complex long speech inputs and achieve more robust omni-cognition. Compared to other omni-methods, Lyra achieves state-of-the-art performance on various vision-language, vision-speech, and speech-language benchmarks, while also using fewer computational resources and less training data. All code, data, and models will be available to the public.

</details>

---

## 463. Zero-Shot Composed Image Retrieval via Dual-Stream Instruction-Aware Distillation

- [ ] Zero-Shot Composed Image Retrieval via Dual-Stream Instruction-Aware Distillation | https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Zero-Shot_Composed_Image_Retrieval_via_Dual-Stream_Instruction-Aware_Distillation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Zero-Shot_Composed_Image_Retrieval_via_Dual-Stream_Instruction-Aware_Distillation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Composed Image Retrieval (CIR) targets the retrieval of images conditioned on a reference image and a textual modification, but constructing labeled triplets (reference image, textual modification, target image) is inherently challenging. Existing Zero-Shot CIR (ZS-CIR) approaches often rely on well-aligned vision-language models (VLMs) to combine visual and textual inputs, or use large language models (LLMs) for richer modification understanding. While LLM-based methods excel in capturing textual details, they are computationally costly, slow to infer, and often restricted by proprietary constraints. In this paper, we argue that the superior performance of LLM-based ZS-CIR methods primarily stems from their capacity to follow instructions, an aspect largely missing in more efficient projection-based models built upon VLMs. To bridge this gap, we introduce DistillCIR, a dual-stream distillation framework that transfers LLMs' instruction-following capability into compact, projection-based architectures. By synthesizing triplet data with an LLM and incorporating a novel reasoning process, DistillCIR learns both composed retrieval and instruction awareness. In addition, we train an open-source multimodal LLM on the generated data, and further distill its instruction-aware embeddings into the projection-based model. Without any reliance on LLMs at inference, DistillCIR significantly surpasses state-of-the-art ZS-CIR methods in both performance and efficiency, offering a promising direction for instruction-aware, lightweight CIR.

</details>

---

## 464. AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

- [ ] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of AI-generated content (AIGC) technology has led to the misuse of highly realistic AI-generated images (AIGI) in spreading misinformation, posing a threat to public information security. Although existing AIGI detection techniques are generally effective, they face two issues: 1) a lack of human-verifiable explanations, and 2) a lack of generalization in the latest generation technology. To address these issues, we introduce a large-scale and comprehensive dataset, Holmes-Set, which includes the Holmes-SFTSet, an instruction-tuning dataset with explanations on whether images are AI-generated, and the Holmes-DPOSet, a human-aligned preference dataset. Our work introduces an efficient data annotation method called the Multi-Expert Jury, enhancing data generation through structured MLLM explanations and quality control via cross-model evaluation, expert defect filtering, and human preference modification. In addition, we propose Holmes Pipeline, a meticulously designed three-stage training framework comprising visual expert pre-training, supervised fine-tuning, and direct preference optimization. Holmes Pipeline adapts multimodal large language models (MLLMs) for AIGI detection while generating human-verifiable and human-aligned explanations, ultimately yielding our model AIGI-Holmes. During the inference stage, we introduce a collaborative decoding strategy that integrates the model perception of the visual expert with the semantic reasoning of MLLMs, further enhancing the generalization capabilities. Extensive experiments on three benchmarks validate the effectiveness of our AIGI-Holmes.

</details>

---

## 465. Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs

- [ ] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Are_They_the_Same_Exploring_Visual_Correspondence_Shortcomings_of_Multimodal_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Are_They_the_Same_Exploring_Visual_Correspondence_Shortcomings_of_Multimodal_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLM) have shown a strong ability in visual perception, reasoning abilities, and vision-language understanding. However, the visual matching ability of MLLMs is rarely studied, despite finding the visual correspondence of objects is essential in computer vision. Our research reveals that the matching capabilities in recent MLLMs still exhibit systematic shortcomings, even with current strong MLLMs models, GPT-4o. In particular, we construct a Multimodal Visual Matching (MMVM) benchmark to fairly benchmark over 30 different MLLMs. The MMVM benchmark is built from 15 open-source datasets and Internet videos with manual annotation. In addition, we have designed an automatic annotation pipeline to generate the MMVM SFT dataset, including 220K visual matching data with reasoning annotation. To our knowledge, this is the first MLLMs dataset and benchmark for the MLLM community. Finally, we present CoLVA, a novel contrastive MLLM with two novel technical designs: fine-grained vision expert with object-level contrastive learning and instruction augmentation strategy. The former learns instance discriminative tokens, while the latter further improves instruction following ability. CoLVA-InternVL2-4B achieves an overall accuracy (OA) of 49.80% on the MMVM benchmark, surpassing GPT-4o and the best open-source MLLM, Qwen2VL-72B, by 7.15% and 11.72% OA, respectively. These results demonstrate the effectiveness of our MMVM SFT dataset and our novel technical designs. Code, benchmark, dataset, and models will be released.

</details>

---

## 466. AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting

- [ ] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_AutoOcc_Automatic_Open-Ended_Semantic_Occupancy_Annotation_via_Vision-Language_Guided_Gaussian_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_AutoOcc_Automatic_Open-Ended_Semantic_Occupancy_Annotation_via_Vision-Language_Guided_Gaussian_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Obtaining high-quality 3D semantic occupancy from raw sensor data remains an essential yet challenging task, often requiring extensive manual labeling. In this work, we propose AutoOcc, a vision-centric automated pipeline for open-ended semantic occupancy annotation that integrates differentiable Gaussian splatting guided by vision-language models. We formulate the open-ended semantic 3D occupancy reconstruction task to automatically generate scene occupancy by combining attention maps from vision-language models and foundation vision models. We devise semantic-aware Gaussians as intermediate geometric descriptors and propose a cumulative Gaussian-to-voxel splatting algorithm that enables effective and efficient occupancy annotation. Our framework outperforms existing automated occupancy annotation methods without human labels. AutoOcc also enables open-ended semantic occupancy auto-labeling, achieving robust performance in both static and dynamically complex scenarios.

</details>

---

## 467. DOGR: Towards Versatile Visual Document Grounding and Referring

- [ ] DOGR: Towards Versatile Visual Document Grounding and Referring | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_DOGR_Towards_Versatile_Visual_Document_Grounding_and_Referring_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_DOGR_Towards_Versatile_Visual_Document_Grounding_and_Referring_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With recent advances in Multimodal Large Language Models (MLLMs), grounding and referring capabilities have gained increasing attention for achieving detailed understanding and flexible user interaction. However, these capabilities still remain underdeveloped in visual document understanding due to the scarcity of fine-grained datasets and comprehensive benchmarks. To fill this gap, we propose the **DO**cument **G**rounding and **R**eferring data engine (**DOGR-Engine**), which generates two types of high-quality fine-grained document data: (1) multi-granular parsing data to improve text localization and recognition, and (2) instruction-tuning data to activate MLLMs' grounding and referring capabilities in dialogue and reasoning. Using the DOGR-Engine, we construct **DOGR-Bench**, a benchmark covering seven grounding and referring tasks across three document types (chart, poster, and PDF document), offering a comprehensive evaluation of fine-grained document understanding. Leveraging the generated data, we further develop **DOGR**, a strong baseline model that excels in text localization and recognition, while precisely grounds and refers to key textual information during conversation and reasoning, thereby advancing document understanding to a finer granularity and enable flexible interaction paradigms. Our code, data, and model are open-sourced at https://github.com/zyinan99/DOGR.

</details>

---

## 468. External Knowledge Injection for CLIP-Based Class-Incremental Learning

- [ ] External Knowledge Injection for CLIP-Based Class-Incremental Learning | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_External_Knowledge_Injection_for_CLIP-Based_Class-Incremental_Learning_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_External_Knowledge_Injection_for_CLIP-Based_Class-Incremental_Learning_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Class-Incremental Learning (CIL) enables learning systems to continuously adapt to evolving data streams. With the advancement of pre-training, leveraging pre-trained vision-language models (e.g., CLIP) offers a promising starting point for CIL. However, CLIP makes decisions by matching visual embeddings to class names, overlooking the rich contextual information conveyed through language. For instance, the concept of "cat" can be decomposed into features like tail, fur, and face for recognition.Besides, since the model is continually updated, these detailed features are overwritten in CIL, requiring external knowledge for compensation.In this paper, we introduce ExterNal knowledGe INjEction (ENGINE) for CLIP-based CIL. To enhance knowledge transfer from outside the dataset, we propose a dual-branch injection tuning framework that encodes informative knowledge from both visual and textual modalities. The visual branch is enhanced with data augmentation to enrich the visual features, while the textual branch leverages GPT-4 to rewrite discriminative descriptors. In addition to this on-the-fly knowledge injection, we also implement post-tuning knowledge by re-ranking the prediction results during inference. With the injected knowledge, the model can better capture informative features for downstream tasks as data evolves. Extensive experiments demonstrate the state-of-the-art performance of ENGINE. Code is available at: https://github.com/LAMDA-CL/ICCV25-ENGINE

</details>

---

## 469. Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving

- [ ] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Hints_of_Prompt_Enhancing_Visual_Representation_for_Multimodal_LLMs_in_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Hints_of_Prompt_Enhancing_Visual_Representation_for_Multimodal_LLMs_in_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In light of the dynamic nature of autonomous driving environments and stringent safety requirements, general MLLMs combined with CLIP alone often struggle to accurately represent driving-specific scenarios, particularly in complex interactions and long-tail cases. To address this, we propose the Hints of Prompt (HoP) framework, which introduces three key enhancements: Affinity hint to emphasize instance-level structure by strengthening token-wise connections, Semantic hint to incorporate high-level information relevant to driving-specific cases, such as complex interactions among vehicles and traffic signs, and Question hint to align visual features with the query context, focusing on question-relevant regions. These hints are fused through a Hint Fusion module, enriching visual representations by capturing driving-related representations with limited domain data, ensuring faster adaptation to driving scenarios. Extensive experiments confirm the effectiveness of the HoP framework, showing that it significantly outperforms previous state-of-the-art methods in all key metrics.

</details>

---

## 470. LIRA: Reasoning Reconstruction via Multimodal Large Language Models

- [ ] LIRA: Reasoning Reconstruction via Multimodal Large Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_LIRA_Reasoning_Reconstruction_via_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_LIRA_Reasoning_Reconstruction_via_Multimodal_Large_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing language instruction-guided online 3D reconstruction systems mainly rely on explicit instructions or queryable maps, showing inadequate capability to handle implicit and complex instructions. In this paper, we first introduce a reasoning reconstruction task. This task inputs an implicit instruction involving complex reasoning and an RGB-D sequence, and outputs incremental 3D reconstruction of instances that conform to the instruction. To handle this task, we propose LIRA: Language Instructed Reconstruction Assistant. It leverages a multimodal large language model to actively reason about the implicit instruction and obtain instruction-relevant 2D candidate instances and their attributes. Then, candidate instances are back-projected into the incrementally reconstructed 3D geometric map, followed by instance fusion and target instance inference. In LIRA, to achieve higher instance fusion quality, we propose TIFF, a Text-enhanced Instance Fusion module operating within Fragment bounding volume, which is learning-based and fuses multiple keyframes simultaneously. Since the evaluation system for this task is not well established, we propose a benchmark ReasonRecon comprising the largest collection of scene-instruction data samples involving implicit reasoning. Experiments demonstrate that LIRA outperforms existing methods in the reasoning reconstruction task and is capable of running in real time. Code and benchmark are available at https://github.com/zhen6618/LIRA.

</details>

---

## 471. Learnable Retrieval Enhanced Visual-Text Alignment and Fusion for Radiology Report Generation

- [ ] Learnable Retrieval Enhanced Visual-Text Alignment and Fusion for Radiology Report Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Learnable_Retrieval_Enhanced_Visual-Text_Alignment_and_Fusion_for_Radiology_Report_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Learnable_Retrieval_Enhanced_Visual-Text_Alignment_and_Fusion_for_Radiology_Report_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automated radiology report generation is essential for improving diagnostic efficiency and reducing the workload of medical professionals. However, existing methods face significant challenges, such as disease class imbalance and insufficient cross-modal fusion. To address these issues, we propose the learnable Retrieval Enhanced Visual-Text Alignment and Fusion (REVTAF) framework, which effectively tackles both class imbalance and visual-text fusion in report generation. REVTAF incorporates two core components: (1) a Learnable Retrieval Enhancer (LRE) that utilizes semantic hierarchies from hyperbolic space and intra-batch context through a ranking-based metric. LRE adaptively retrieves the most relevant reference reports, enhancing image representations, particularly for underrepresented (tail) class inputs; and (2) a fine-grained visual-text alignment and fusion strategy that ensures consistency across multi-source cross-attention maps for precise alignment. This component further employs an optimal transport-based cross-attention mechanism to dynamically integrate task-relevant textual knowledge for improved report generation. By combining adaptive retrieval with multi-source alignment and fusion, REVTAF achieves fine-grained visual-text integration under weak image-report level supervision while effectively mitigating data imbalance issues. The experiments demonstrate that REVTAF outperforms state-of-the-art methods, achieving an average improvement of 7.4% on the MIMIC-CXR dataset and 2.9% on the IU X-Ray dataset. Comparisons with mainstream multimodal LLMs (e.g., GPT-series models), further highlight its superiority in radiology report generation.

</details>

---

## 472. Multimodal LLMs as Customized Reward Models for Text-to-Image Generation

- [ ] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Multimodal_LLMs_as_Customized_Reward_Models_for_Text-to-Image_Generation_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce LLaVA-Reward, an efficient reward model designed to automatically evaluate text-to-image (T2I) generations across multiple perspectives, leveraging pretrained multimodal large language models (MLLMs). Existing MLLM-based approaches require instruction-following data for supervised fine-tuning and evaluate generation quality on analyzing text response, which is time-consuming and difficult to train. To address this problem, we propose LLaVA-Reward, which directly utilizes the hidden states of MLLMs given text-image pairs. To enhance the bidirectional interaction between visual and textual representations in decoder-only MLLMs, we further propose adding a Skip-connection Cross Attention (SkipCA) module. This design enhances text-image correlation reasoning by connecting early-layer visual features with later-layer hidden representations. In addition, LLaVA-Reward supports different types of preference data for efficient fine-tuning, including paired preference data and unpaired data. We train LLaVA-Reward on four evaluation perspectives: text-image alignment, fidelity/artifact, safety, and overall ranking. Empirical results demonstrate that LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations.

</details>

---

## 473. OV3D-CG: Open-vocabulary 3D Instance Segmentation with Contextual Guidance

- [ ] OV3D-CG: Open-vocabulary 3D Instance Segmentation with Contextual Guidance | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_OV3D-CG_Open-vocabulary_3D_Instance_Segmentation_with_Contextual_Guidance_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_OV3D-CG_Open-vocabulary_3D_Instance_Segmentation_with_Contextual_Guidance_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary 3D instance segmentation (OV-3DIS), which aims to segment and classify objects beyond predefined categories, is a critical capability for embodied AI applications. Existing methods rely on pre-trained 2D foundation models, focusing on instance-level features while overlooking contextual relationships, limiting their ability to generalize to rare or ambiguous objects. To address these limitations, we propose an OV-3DIS framework guided by contextual information. First, we employ a Class-agnostic Proposal Module, integrating a pre-trained 3D segmentation model with a SAM-guided segmenter to extract robust 3D instance masks. Subsequently, we design a Semantic Reasoning Module, which selects the best viewpoint for each instance and constructs three 2D context-aware representations. The representations are processed using Multimodal Large Language Models with Chain-of-Thought prompting to enhance semantic inference. Notably, our method outperforms state-of-the-art methods on the ScanNet200 and Replica datasets, demonstrating superior open-vocabulary segmentation capabilities. Moreover, preliminary implementation in real-world scenarios verifies our method's robustness and accuracy, highlighting its potential for embodied AI tasks such as object-driven navigation. Our project page is at: https://vipl-vsu.github.io/OV3D-CG/.

</details>

---

## 474. VLM4D: Towards Spatiotemporal Awareness in Vision Language Models

- [ ] VLM4D: Towards Spatiotemporal Awareness in Vision Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_VLM4D_Towards_Spatiotemporal_Awareness_in_Vision_Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_VLM4D_Towards_Spatiotemporal_Awareness_in_Vision_Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) have shown remarkable capabilities in integrating linguistic and visual reasoning but remain fundamentally limited in understanding dynamic spatiotemporal interactions. Humans effortlessly track and reason about object movements, rotations, and perspective shifts--abilities essential for robust dynamic real-world understanding yet notably lacking in current VLMs. In this paper, we introduce VLM4D, the first benchmark specifically designed to evaluate the spatiotemporal reasoning capabilities of VLMs. Our benchmark comprises diverse real-world and synthetic videos accompanied by carefully curated question-answer pairs emphasizing translational and rotational motions, perspective awareness, and motion continuity. Through comprehensive evaluations of state-of-the-art open and closed-source VLMs, we identify significant performance gaps compared to human baselines, highlighting fundamental deficiencies in existing models. Extensive analysis reveals that VLMs struggle particularly with integrating multiple visual cues and maintaining temporal coherence. We further explore promising directions, such as leveraging 4D feature field reconstruction and targeted spatiotemporal supervised fine-tuning, demonstrating their effectiveness in enhancing spatiotemporal comprehension. Our work aims to encourage deeper exploration into improving VLMs' spatial and temporal grounding, paving the way towards more capable and reliable visual intelligence for dynamic environments.

</details>

---

## 475. 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding

- [ ] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding | https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_4D-Bench_Benchmarking_Multi-modal_Large_Language_Models_for_4D_Object_Understanding_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_4D-Bench_Benchmarking_Multi-modal_Large_Language_Models_for_4D_Object_Understanding_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated impressive 2D image/video understanding capabilities.However, there are no publicly standardized benchmarks to assess the abilities of MLLMs in understanding the 4D objects.In this paper, we introduce 4D-Bench, the first benchmark to evaluate the capabilities of MLLMs in 4D object understanding, featuring tasks in 4D object Question Answering (4D object QA) and 4D object captioning.4D-Bench provides 4D objects with diverse categories, high-quality annotations, and tasks necessitating multi-view spatial-temporal understanding, different from existing 2D image/video-based benchmarks.With 4D-Bench, we evaluate a wide range of open-source and closed-source MLLMs.The results from the 4D object captioning experiment indicate that MLLMs generally exhibit weaker temporal understanding compared to their appearance understanding, notably, while open-source models approach closed-source performance in appearance understanding, they show larger performance gaps in temporal understanding.4D object QA yields surprising findings: even with simple single-object videos, MLLMs perform poorly, with state-of-the-art GPT-4o achieving only 63% accuracy compared to the human baseline of 91%.These findings highlight a substantial gap in 4D object understanding and the need for further advancements in MLLMs.

</details>

---

## 476. Dynamic Multimodal Prototype Learning in Vision-Language Models

- [ ] Dynamic Multimodal Prototype Learning in Vision-Language Models | https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Dynamic_Multimodal_Prototype_Learning_in_Vision-Language_Models_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Dynamic_Multimodal_Prototype_Learning_in_Vision-Language_Models_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the increasing attention to pre-trained vision-language models (VLMs), e.g., CLIP, substantial efforts have been devoted to many downstream tasks, especially in test-time adaptation (TTA). However, previous works focus on learning prototypes only in the textual modality while overlooking the ambiguous semantics in class names. These ambiguities lead to textual prototypes that are insufficient to capture visual concepts, resulting in limited performance. To address this issue, we introduce **ProtoMM**, a training-free framework that constructs multimodal prototypes to adapt VLMs during the test time. By viewing the prototype as a discrete distribution over the textual descriptions and visual particles, ProtoMM has the ability to combine the multimodal features for comprehensive prototype learning. More importantly, the visual particles are dynamically updated as the testing stream flows. This allows our multimodal prototypes to continually learn from the data, enhancing their generalizability in unseen scenarios. In addition, we quantify the importance of the prototypes and test images by formulating their semantic distance as an optimal transport problem. Extensive experiments on 15 zero-shot benchmarks demonstrate the effectiveness of our method, achieving a 1.03% average accuracy improvement over state-of-the-art methods on ImageNet and its variant datasets.

</details>

---

## 477. Evading Data Provenance in Deep Neural Networks

- [ ] Evading Data Provenance in Deep Neural Networks | https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Evading_Data_Provenance_in_Deep_Neural_Networks_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Evading_Data_Provenance_in_Deep_Neural_Networks_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Modern over-parameterized deep models are highly data-dependent, with large scale general-purpose and domain-specific datasets serving as the bedrock for rapid advancements. However, many datasets are proprietary or contain sensitive information, making unrestricted model training problematic. In the open world where data thefts cannot be fully prevented, Dataset Ownership Verification (DOV) has emerged as a promising method to protect copyright by detecting unauthorized model training and tracing illicit activities. Due to its diversity and superior stealth, evading DOV is considered extremely challenging. However, this paper identifies that previous studies have relied on oversimplistic evasion attacks for evaluation, leading to a false sense of security. We introduce a unified evasion framework, in which a teacher model first learns from the copyright dataset and then transfers task-relevant yet identifier-independent domain knowledge to a surrogate student using an out-of-distribution (OOD) dataset as the intermediary. Leveraging Vision-Language Models and Large Language Models, we curate the most informative and reliable subsets from the OOD gallery set as the final transfer set, and propose selectively transferring task-oriented knowledge to achieve a better trade-off between generalization and evasion effectiveness. Experiments across diverse datasets covering eleven DOV methods demonstrate our approach simultaneously eliminates all copyright identifiers and significantly outperforms nine state-of-the-art evasion attacks in both generalization and effectiveness, with moderate computational overhead. As a proof of concept, we reveal key vulnerabilities in current DOV methods, highlighting the need for long-term development to enhance practicality.

</details>

---

## 478. Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection

- [ ] Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Fine-grained_Abnormality_Prompt_Learning_for_Zero-shot_Anomaly_Detection_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Fine-grained_Abnormality_Prompt_Learning_for_Zero-shot_Anomaly_Detection_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current zero-shot anomaly detection (ZSAD) methods show remarkable success in prompting large pre-trained vision-language models to detect anomalies in a target dataset without using any dataset-specific training or demonstration. However, these methods often focus on crafting/learning prompts that capture only coarse-grained semantics of abnormality, e.g., high-level semantics like 'damaged', 'imperfect', or 'defective' objects. They therefore have limited capability in recognizing diverse abnormality details that deviate from these general abnormal patterns in various ways. To address this limitation, we propose FAPrompt, a novel framework designed to learn Fine-grained Abnormality Prompts for accurate ZSAD. To this end, a novel Compound Abnormality Prompt learning (CAP) module is introduced in FAPrompt to learn a set of complementary, decomposed abnormality prompts, where abnormality prompts are enforced to model diverse abnormal patterns derived from the same normality semantic. On the other hand, the fine-grained abnormality patterns can be different from one dataset to another. To enhance the cross-dataset generalization, another novel module, namely Data-dependent Abnormality Prior learning (DAP), is introduced in FAPrompt to learn a sample-wise abnormality prior from abnormal features of each test image to dynamically adapt the abnormality prompts to individual test images. Comprehensive experiments on 19 real-world datasets, covering both industrial defects and medical anomalies, demonstrate that FAPrompt substantially outperforms state-of-the-art methods in both image- and pixel-level ZSAD tasks. Code is available at https:// github.com/ mala-lab/ FAPrompt.

</details>

---

## 479. LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities

- [ ] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities | https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_LLaVA-3D_A_Simple_yet_Effective_Pathway_to_Empowering_LMMs_with_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_LLaVA-3D_A_Simple_yet_Effective_Pathway_to_Empowering_LMMs_with_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Multimodal Models (LMMs) have greatly enhanced their proficiency in 2D visual understanding tasks, enabling them to effectively process and understand images and videos. However, the development of LMMs with 3D scene understanding capabilities has been hindered by the lack of large-scale 3D vision-language datasets and powerful 3D encoders. In this paper, we introduce a simple yet effective framework called LLaVA-3D. Leveraging the strong 2D visual understanding priors from LLaVA, our LLaVA-3D efficiently adapts LLaVA for 3D scene understanding without compromising 2D understanding capabilities. To achieve this, we utilize the 3D position embeddings to enhance the 2D CLIP Patches with 3D spatial context information and construct 3D patches. By integrating the 3D position embeddings into 2D LMMs and employing joint 2D and 3D vision-language instruction tuning, we establish a unified architecture for both 2D visual understanding and 3D scene understanding. In contrast to previous 3D LMMs, LLaVA-3D supports decoding accurate 3D spatial perception outputs, e.g., 3D bounding boxes, directly from these 3D patches, without relying on the time-consuming off-the-shelf 3D segmentors. Experimental results show that LLaVA-3D converges 3.5x faster than existing 3D LMMs when trained on 3D vision-language datasets. Moreover, LLaVA-3D not only achieves state-of-the-art performance across various 3D tasks but also maintains comparable 2D visual understanding and vision-language conversation capabilities with LLaVA.

</details>

---

## 480. Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation

- [ ] Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation | https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Move_to_Understand_a_3D_Scene_Bridging_Visual_Grounding_and_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Move_to_Understand_a_3D_Scene_Bridging_Visual_Grounding_and_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Embodied scene understanding requires not only comprehending visual-spatial information that has been observed but also determining where to explore next in the 3D physical world. Existing 3D Vision-Language (3D-VL) models primarily focus on grounding objects in static observations from 3D reconstruction, such as meshes and point clouds, but lack the ability to actively perceive and explore their environment. To address this limitation, we introduce Move to Understand (MTU3D), a unified framework that integrates active perception with 3D vision-language learning, enabling embodied agents to effectively explore and understand their environment. This is achieved by three key innovations 1) Online query-based representation learning, enabling direct spatial memory construction from RGB-D frames, eliminating the need for explicit 3D reconstruction. 2) A unified objective for grounding and exploration that represents unexplored locations as frontier queries and jointly optimizes object grounding and frontier selection. 3) End-to-end trajectory learning that combines Vision-Language-Exploration pre-training over a million diverse trajectories collected from both simulated and real-world RGB-D sequences. Extensive evaluations across various embodied navigation and question-answering benchmarks show that MTU3D outperforms state-of-the-art reinforcement learning and modular navigation approaches by 14%, 23%, 9%, and 2% in success rate on HM3D-OVON, GOAT-Bench, SG3D, and A-EQA, respectively. MTU3D's versatility enables navigation using diverse input modalities, including categories, language descriptions, and reference images. Additionally, we deploy it on a real robot to demonstrate its effectiveness in handling real-world data. These findings highlight the importance of bridging visual grounding and exploration for embodied intelligence.

</details>

---

## 481. PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation

- [ ] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation | https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_PASG_A_Closed-Loop_Framework_for_Automated_Geometric_Primitive_Extraction_and_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_PASG_A_Closed-Loop_Framework_for_Automated_Geometric_Primitive_Extraction_and_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The fragmentation between high-level task semantics and low-level geometric features remains a persistent challenge in robotic manipulation. While vision-language models (VLMs) have shown promise in generating affordance-aware visual representations, the lack of semantic grounding in canonical spaces and reliance on manual annotations severely limit their ability to capture dynamic semantic-affordance relationships. To address these, we propose Primitive-Aware Semantic Grounding (PASG), a closed-loop framework that introduces: (1) Automatic primitive extraction through geometric feature aggregation, enabling cross-category detection of keypoints and axes; (2) VLM-driven semantic anchoring that dynamically couples geometric primitives with functional affordances and task-relevant description; (3) A spatial-semantic reasoning benchmark and a fine-tuned VLM (Qwen2.5VL-PA). We demonstrate PASG's effectiveness in practical robotic manipulation tasks across diverse scenarios, achieving performance comparable to manual annotations. PASG achieves a finer-grained semantic-affordance understanding of objects, establishing a unified paradigm for bridging geometric primitives with task semantics in robotic manipulation.

</details>

---

## 482. Dataset Distillation via Vision-Language Category Prototype

- [ ] Dataset Distillation via Vision-Language Category Prototype | https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Dataset_Distillation_via_Vision-Language_Category_Prototype_ICCV_2025_paper.html

- **Link**: https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Dataset_Distillation_via_Vision-Language_Category_Prototype_ICCV_2025_paper.html

- **Conference**: ICCV

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Dataset distillation (DD) condenses large datasets into compact yet informative substitutes, preserving performance comparable to the original dataset while reducing storage, transmission costs, and computational consumption. However, previous DD methods mainly focus on distilling information from images, often overlooking the semantic information inherent in the data. The disregard for context hinders the model's generalization ability, particularly in tasks involving complex datasets, which may result in illogical outputs or the omission of critical objects. In this study, we integrate vision-language methods into DD by introducing text prototypes to distill language information and collaboratively synthesize data with image prototypes, thereby enhancing dataset distillation performance. Notably, the text prototypes utilized in this study are derived from descriptive text information generated by an open-source vision-language model. This framework demonstrates broad applicability across datasets without pre-existing text descriptions, expanding the potential of dataset distillation beyond traditional image-based approaches. Compared to other methods, the proposed approach generates logically coherent images containing target objects, achieving state-of-the-art validation performance and demonstrating robust generalization. Source code and generated data are available in https://github.com/zou-yawen/Dataset-Distillation-via-Vision-Language-Category-Prototype/.

</details>

---

