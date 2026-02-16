# ICLR 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_iclr2025_papers.csv

## 1. Do You Keep an Eye on What I Ask? Mitigating Multimodal Hallucination via Attention-Guided Ensemble Decoding

- [ ] Do You Keep an Eye on What I Ask? Mitigating Multimodal Hallucination via Attention-Guided Ensemble Decoding | https://iclr.cc/virtual/2025/poster/27655

- **Link**: https://iclr.cc/virtual/2025/poster/27655

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision-Language Models (LVLMs) have significantly expanded their utility in tasks like image captioning and visual question answering. However, they still struggle with object hallucination, where models generate descriptions that inaccurately reflect the visual content by including nonexistent objects or misrepresenting existing ones. While previous methods, such as data augmentation and training-free approaches, strive to tackle this issue, they still encounter scalability challenges and often depend on additional external modules. In this work, we propose Ensemble Decoding (ED), a novel strategy that splits the input image into sub-images and combines logit distributions by assigning weights through the attention map. Furthermore, we introduce ED adaptive plausibility constraint to calibrate logit distribution and FastED, a variant designed for speed-critical applications. Extensive experiments across hallucination benchmarks demonstrate that our proposed method achieves state-of-the-art performance, validating the effectiveness of our approach.

</details>

---

## 2. Semantic Temporal Abstraction via Vision-Language Model Guidance for Efficient Reinforcement Learning

- [ ] Semantic Temporal Abstraction via Vision-Language Model Guidance for Efficient Reinforcement Learning | https://iclr.cc/virtual/2025/poster/27665

- **Link**: https://iclr.cc/virtual/2025/poster/27665

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Extracting temporally extended skills can significantly improve the efficiency of reinforcement learning (RL) by breaking down complex decision-making problems with sparse rewards into simpler subtasks and enabling more effective credit assignment. However, existing abstraction methods either discover skills in an unsupervised manner, which often lacks semantic information and leads to erroneous or scattered skill extraction results, or require substantial human intervention. In this work, we propose to leverage the extensive knowledge in pretrained Vision-Language Models (VLMs) to progressively guide the latent space after vector quantization to be more semantically meaningful through relabeling each skill. This approach, termed V ision-l an guage model guided T emporal A bstraction ( VanTA ), facilitates the discovery of more interpretable and task-relevant temporal segmentations from offline data without the need for extensive manual intervention or heuristics. By leveraging the rich information in VLMs, our method can significantly outperform existing offline RL approaches that depend only on limited training data. From a theory perspective, we demonstrate that stronger internal sequential correlations within each sub-task, induced by VanTA, effectively reduces suboptimality in policy learning. We validate the effectiveness of our approach through extensive experiments on diverse environments, including Franka Kitchen, Minigrid, and Crafter. These experiments show that our method surpasses existing approaches in long-horizon offline reinforcement learning scenarios with both proprioceptive and visual observations.

</details>

---

## 3. Intervening Anchor Token: Decoding Strategy in Alleviating Hallucinations for MLLMs

- [ ] Intervening Anchor Token: Decoding Strategy in Alleviating Hallucinations for MLLMs | https://iclr.cc/virtual/2025/poster/27678

- **Link**: https://iclr.cc/virtual/2025/poster/27678

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) offer a powerful mechanism for interpreting visual information. However, they often suffer from hallucinations, which impede the real-world usage of these models. Existing methods attempt to alleviate this issue by designing special decoding strategies that penalize the summary tokens. However, these methods lack analysis of the relationship between hallucination and summarization mechanism of LLMs. Interestingly, we find that penalizing summary tokens is not necessary: merely intervening the query-key parameters variance, without costing extra inference time, still alleviates hallucinations. Specifically, we explore the causes of hallucinations by analyzing localized self-attention patterns called ``anchor" tokens and define the attention localization degree of the model as token propagation probabilities. Our analysis reveals that over-propagation of anchor tokens occurs when the distribution of eigenvalues of the query and key matrices has a non-zero mean and a polarized variance, leading to excessive dependence on anchor tokens while neglecting vision information and describes the image content with hallucination. Based on the observation, we propose a versatile plug-and-play decoding strategy, Dynamic Token Propagation Mechanism (TAME), to alleviate excessive propagation by dynamically intervening the eigenspectrum variance of the attention weight, thereby alleviating hallucinations without relying on complex decoding strategies. Extensive experiments reveal a correlation between the eigenspectrum and hallucinations across various MLLMs, and show that TAME reduces the percentage of hallucinated objects.

</details>

---

## 4. VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents

- [ ] VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents | https://iclr.cc/virtual/2025/poster/27679

- **Link**: https://iclr.cc/virtual/2025/poster/27679

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-augmented generation (RAG) is an effective technique that enables large language models (LLMs) to utilize external knowledge sources for generation.  However, current RAG systems are solely based on text, rendering it impossible to utilize vision information like layout and images that play crucial roles in real-world multi-modality documents.  In this paper, we introduce VisRAG, which tackles this issue by establishing a vision-language model (VLM)-based RAG pipeline.  In this pipeline, instead of first parsing the document to obtain text, the document is directly embedded using a VLM as an image and then retrieved to enhance the generation of a VLM. Compared to traditional text-based RAG, VisRAG maximizes the retention and utilization of the data information in the original documents, eliminating the information loss introduced during the parsing process. We collect both open-source and synthetic data to train the retriever in VisRAG and explore a variety of generation methods. Experiments demonstrate that VisRAG outperforms traditional RAG in both the retrieval and generation stages, achieving a 20–40% end-to-end performance gain over traditional text-based RAG pipeline. Further analysis reveals that VisRAG is efficient in utilizing training data and demonstrates strong generalization capability, positioning it as a promising solution for RAG on multi-modality documents. Our code and data are available at https://github.com/openbmb/visrag.

</details>

---

## 5. Language-Image Models with 3D Understanding

- [ ] Language-Image Models with 3D Understanding | https://iclr.cc/virtual/2025/poster/27715

- **Link**: https://iclr.cc/virtual/2025/poster/27715

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have shown incredible capabilities in a variety of 2D vision and language tasks. We extend MLLMs’ perceptual capabilities to ground and reason about images in 3-dimensional space. To that end, we first develop a large-scale pretraining dataset for 2D and 3D called LV3D by combining multiple existing 2D and 3D recognition datasets under a common task formulation: as multi-turn question-answering. Next, we introduce a new MLLM named CUBE-LLM and pre-train it on LV3D. We show that pure data scaling makes a strong 3D perception capability without 3D specific architectural design or training objective. CUBE-LLM exhibits intriguing properties similar to LLMs: (1) CUBE-LLM can apply chain-of-thought prompting to improve 3D understanding from 2D context information. (2) CUBE-LLM can follow complex and diverse instructions and adapt to versatile input and output formats. (3) CUBE-LLM can be visually prompted such as 2D box or a set of candidate 3D boxes from specialists. Our experiments on outdoor benchmarks demonstrate that CUBE-LLM significantly outperforms existing baselines by 21.3 points of AP-BEV on the Talk2Car dataset for 3D grounded reasoning and 17.7 points on the DriveLM dataset for complex reasoning about driving scenarios, respectively. CUBE-LLM also shows competitive results in general MLLM benchmarks such as refCOCO for 2D grounding with (87.0) average score, as well as visual question answering benchmarks such as VQAv2, GQA, SQA, POPE, etc. for complex reasoning.

</details>

---

## 6. Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment

- [ ] Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment | https://iclr.cc/virtual/2025/poster/27739

- **Link**: https://iclr.cc/virtual/2025/poster/27739

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite their significant advancements, Multimodal Large Language Models(MLLMs) often generate factually inaccurate information, referred to as hallucination.In this work, we address object hallucinations in MLLMs, where informationis generated about an object not present in the input image. We introduce Data-augmentedPhrase-level Alignment (DPA), a novel loss which can be applied toinstruction-tuned off-the-shelf MLLMs to mitigate hallucinations, while preservingtheir general vision-language capabilities. To fine-tune MLLMs with DPA, we firstgenerate a set of 'hallucinated' and 'correct' response pairs through generative dataaugmentation by selectively altering the ground-truth information of the correctresponses at a phrase level. The DPA loss is then used to train MLLMs to reducethe likelihood of hallucinated phrases compared to the correct ones. Our thoroughevaluation on various benchmarks confirms the effectiveness of DPA in mitigatinghallucination while retaining the out-of-the-box performance of the MLLMs ongeneral tasks. For instance, MLLMs finetuned with DPA, which we refer to as HallucinationAttenuated Language and Vision Assistant (HALVA), improve F1 by upto 13.4% on hallucination visual question-answering and reduce the hallucinationrate by up to 4.2% on image description tasks.

</details>

---

## 7. Correlating instruction-tuning (in multimodal models) with vision-language processing (in the brain)

- [ ] Correlating instruction-tuning (in multimodal models) with vision-language processing (in the brain) | https://iclr.cc/virtual/2025/poster/27766

- **Link**: https://iclr.cc/virtual/2025/poster/27766

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Transformer-based language models, though not explicitly trained to mimic brain recordings, have demonstrated surprising alignment with brain activity. Progress in these models—through increased size, instruction-tuning, and multimodality—has led to better representational alignment with neural data. Recently, a new class of instruction-tuned multimodal LLMs (MLLMs) have emerged, showing remarkable zero-shot capabilities in open-ended multimodal vision tasks. However, it is unknown whether MLLMs, when prompted with natural instructions, lead to better brain alignment and effectively capture instruction-specific representations. To address this, we first investigate the brain alignment, i.e., measuring the degree of predictivity of neural visual activity using text output response embeddings from MLLMs as participants engage in watching natural scenes. Experiments with 10 different instructions (like image captioning, visual question answering, etc.) show that  MLLMs exhibit significantly better brain alignment than vision-only models and perform comparably to non-instruction-tuned multimodal models like CLIP. We also find that while these MLLMs are effective at generating high-quality responses suitable to the task-specific instructions, not all instructions are relevant for brain alignment. Further, by varying instructions, we make the MLLMs encode instruction-specific visual concepts related to the input image. This analysis shows that MLLMs effectively capture count-related and recognition-related concepts, demonstrating strong alignment with brain activity. Notably, the majority of the explained variance of the brain encoding models is shared between MLLM embeddings of image captioning and other instructions. These results indicate that enhancing MLLMs' ability to capture more task-specific information could allow for better differentiation between various types of instructions, and hence improve their precision in predicting brain responses.

</details>

---

## 8. Mixture of Experts Made Personalized: Federated Prompt Learning for Vision-Language Models

- [ ] Mixture of Experts Made Personalized: Federated Prompt Learning for Vision-Language Models | https://iclr.cc/virtual/2025/poster/27771

- **Link**: https://iclr.cc/virtual/2025/poster/27771

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Federated prompt learning benefits federated learning with CLIP-like Vision-Language Model's (VLM's) robust representation learning ability through prompt learning. However, current federated prompt learning methods are habitually restricted to the traditional FL paradigm, where the participating clients are generally only allowed to download a single globally aggregated model from the server. While justifiable for training full-sized models under federated settings, in this work, we argue that this paradigm is ill-suited for lightweight prompts. By facilitating the clients to download multiple pre-aggregated prompts as fixed non-local experts, we propose Personalized Federated Mixture of Adaptive Prompts (pFedMoAP), a novel FL framework that personalizes the prompt learning process through the lens of Mixture of Experts (MoE). pFedMoAP implements a local attention-based gating network that learns to generate enhanced text features for better alignment with local image data, benefiting from both local and downloaded non-local adaptive prompt experts. Extensive experiments on 9 datasets under various federated settings demonstrate the efficacy of the proposed pFedMoAP algorithm. The code is available at https://github.com/ljaiverson/pFedMoAP.

</details>

---

## 9. SANER: Annotation-free Societal Attribute Neutralizer for Debiasing CLIP

- [ ] SANER: Annotation-free Societal Attribute Neutralizer for Debiasing CLIP | https://iclr.cc/virtual/2025/poster/27802

- **Link**: https://iclr.cc/virtual/2025/poster/27802

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large-scale vision-language models, such as CLIP, are known to contain societal bias regarding protected attributes (e.g., gender, age). This paper aims to address the problems of societal bias in CLIP. Although previous studies have proposed to debias societal bias through adversarial learning or test-time projecting, our comprehensive study of these works identifies two critical limitations: 1) loss of attribute information when it is explicitly disclosed in the input and 2) use of the attribute annotations during debiasing process. To mitigate societal bias in CLIP and overcome these limitations simultaneously, we introduce a simple-yet-effective debiasing method called SANER (societal attribute neutralizer) that eliminates attribute information from CLIP text features only of attribute-neutral descriptions. Experimental results show that SANER, which does not require attribute annotations and preserves original information for attribute-specific descriptions, demonstrates superior debiasing ability than the existing methods.

</details>

---

## 10. SPORTU: A Comprehensive Sports Understanding Benchmark for Multimodal Large Language Models

- [ ] SPORTU: A Comprehensive Sports Understanding Benchmark for Multimodal Large Language Models | https://iclr.cc/virtual/2025/poster/27805

- **Link**: https://iclr.cc/virtual/2025/poster/27805

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are advancing the ability to reason about complex sports scenarios by integrating textual and visual information. To comprehensively evaluate their capabilities, we introduce SPORTU, a benchmark designed to assess MLLMs across multi-level sports reasoning tasks. SPORTU comprises two key components: SPORTU-text, featuring 900 multiple-choice questions with human-annotated explanations for rule comprehension and strategy understanding. This component focuses on testing models' ability to reason about sports solely through question-answering (QA), without requiring visual inputs; SPORTU-video, consisting of 1,701 slow-motion video clips across 7 different sports and 12,048 QA pairs, designed to assess multi-level reasoning, from simple sports recognition to complex tasks like foul detection and rule application. We evaluated four prevalent LLMs mainly utilizing few-shot learning paradigms supplemented by chain-of-thought (CoT) prompting on the SPORTU-text part. GPT-4o achieves the highest accuracy of 71\%, but still falls short of human-level performance, highlighting room for improvement in rule comprehension and reasoning. The evaluation for the SPORTU-video part includes 6 proprietary and 8 open-source MLLMs. Experiments show that models fall short on hard tasks that require deep reasoning and rule-based understanding. GPT-4o performs the best with only 57.8\% accuracy on the hard task, showing large room for improvement. We hope that SPORTU will serve as a critical step toward evaluating models' capabilities in sports understanding and reasoning. The dataset is available at https://github.com/chili-lab/SPORTU .

</details>

---

## 11. Failures to Find Transferable Image Jailbreaks Between Vision-Language Models

- [ ] Failures to Find Transferable Image Jailbreaks Between Vision-Language Models | https://iclr.cc/virtual/2025/poster/27813

- **Link**: https://iclr.cc/virtual/2025/poster/27813

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways.In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs.We conducted a large-scale empirical study to assess the transferability of gradient-based universal image "jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release.Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain.When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak  successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors.Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM.Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of "highly-similar" VLMs.These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.

</details>

---

## 12. BlueSuffix: Reinforced Blue Teaming for Vision-Language Models Against Jailbreak Attacks

- [ ] BlueSuffix: Reinforced Blue Teaming for Vision-Language Models Against Jailbreak Attacks | https://iclr.cc/virtual/2025/poster/27811

- **Link**: https://iclr.cc/virtual/2025/poster/27811

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we focus on black-box defense for VLMs against jailbreak attacks.Existing black-box defense methods are either unimodal or bimodal. Unimodal methods enhance either the vision or language module of the VLM, while bimodal methods robustify the model through text-image representation realignment. However, these methods suffer from two limitations: 1) they fail to fully exploit the cross-modal information, or 2) they degrade the model performance on benign inputs.To address these limitations, we propose a novel blue-team method BlueSuffix that defends target VLMs against jailbreak attacks without compromising its performance under black-box setting. BlueSuffix includes three key components: 1) a visual purifier against jailbreak images, 2) a textual purifier against jailbreak texts, and 3) a blue-team suffix generator using reinforcement fine-tuning for enhancing cross-modal robustness. We empirically show on four VLMs (LLaVA, MiniGPT-4, InstructionBLIP, and Gemini) and four safety benchmarks (Harmful Instruction, AdvBench, MM-SafetyBench, and RedTeam-2K) that BlueSuffix outperforms the baseline defenses by a significant margin. Our BlueSuffix opens up a promising direction for defending VLMs against jailbreak attacks. Code is available at https://github.com/Vinsonzyh/BlueSuffix.

</details>

---

## 13. A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation

- [ ] A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation | https://iclr.cc/virtual/2025/poster/27817

- **Link**: https://iclr.cc/virtual/2025/poster/27817

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This work tackles the information loss bottleneck of vector-quantization (VQ) autoregressive image generation by introducing a novel model architecture called the 2-Dimensional Autoregression (DnD) Transformer. The DnD-Transformer predicts more codes for an image by introducing a new direction, model depth , along with the sequence length. Compared to 1D autoregression and previous work using similar 2D image decomposition such as RQ-Transformer, the DnD-Transformer is an end-to-end model that can generate higher quality images with the same backbone model size and sequence length, opening a new optimization perspective for autoregressive image generation. Furthermore, our experiments reveal that the DnD-Transformer's potential extends beyond generating natural images. It can even generate images with rich text and graphical elements in a self-supervised manner, demonstrating an understanding of these combined modalities. This has not been previously demonstrated for popular vision generative models such as diffusion models, showing a spark of vision-language intelligence when trained solely on images. Code, datasets and models are open at https://github.com/chenllliang/DnD-Transformer.

</details>

---

## 14. Aligning Visual Contrastive learning models via Preference Optimization

- [ ] Aligning Visual Contrastive learning models via Preference Optimization | https://iclr.cc/virtual/2025/poster/27829

- **Link**: https://iclr.cc/virtual/2025/poster/27829

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive learning models have demonstrated impressive abilities to capture semantic similarities by aligning representations in the embedding space. However, their performance can be limited by the quality of the training data and its inherent biases. While Preference Optimization (PO) methods such as Reinforcement  Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have been applied to align generative models with human preferences, their use in contrastive learning has yet to be explored. This paper introduces a novel method for training contrastive learning models using different PO methods to break down complex concepts. Our method systematically aligns model behavior with desired preferences, enhancing performance on the targeted task. In particular, we focus on enhancing model robustness against typographic attacks and inductive biases, commonly seen in contrastive vision-language models like CLIP. Our experiments demonstrate that models trained using PO outperform standard contrastive learning techniques while retaining their ability to handle adversarial challenges and maintain accuracy on other downstream tasks. This makes our method well-suited for tasks requiring fairness, robustness, and alignment with specific preferences. We evaluate our method for tackling typographic attacks on images and explore its ability to disentangle gender concepts and mitigate gender bias, showcasing the versatility of our approach.

</details>

---

## 15. VLMaterial: Procedural Material Generation with Large Vision-Language Models

- [ ] VLMaterial: Procedural Material Generation with Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/27856

- **Link**: https://iclr.cc/virtual/2025/poster/27856

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Procedural materials, represented as functional node graphs, are ubiquitous in computer graphics for photorealistic material appearance design. They allow users to perform intuitive and precise editing to achieve desired visual appearances. However, creating a procedural material given an input image requires professional knowledge and significant effort. In this work, we leverage the ability to convert procedural materials into standard Python programs and fine-tune a large pre-trained vision-language model (VLM) to generate such programs from input images. To enable effective fine-tuning, we also contribute an open-source procedural material dataset and propose to perform program-level augmentation by prompting another pre-trained large language model (LLM). Through extensive evaluation, we show that our method outperforms previous methods on both synthetic and real-world examples.

</details>

---

## 16. Tree of Attributes Prompt Learning for Vision-Language Models

- [ ] Tree of Attributes Prompt Learning for Vision-Language Models | https://iclr.cc/virtual/2025/poster/27861

- **Link**: https://iclr.cc/virtual/2025/poster/27861

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt learning has proven effective in adapting vision language models for downstream tasks. However, existing methods usually append learnable prompt tokens solely with the category names to obtain textual features, which fails to fully leverage the rich context indicated in the category name. To address this issue, we propose the Tree of Attributes Prompt learning (TAP), which first instructs LLMs to generate a tree of attributes with a “concept - attribute - description” structure for each category, and then learn the hierarchy with vision and text prompt tokens. Unlike existing methods that merely augment category names with a set of unstructured descriptions, our approach essentially distills structured knowledge graphs associated with class names from LLMs. Furthermore, our approach introduces text and vision prompts designed to explicitly learn the corresponding visual attributes, effectively serving as domain experts. Additionally, the general and diverse descriptions generated based on the class names may be wrong or absent in the specific given images. To address this misalignment, we further introduce a vision-conditional pooling module to extract instance-specific text features. Extensive experimental results demonstrate that our approach outperforms state-of-the-art methods on the zero-shot base-to-novel generalization, cross-dataset transfer, as well as few-shot classification across 11 diverse datasets. Code is available at https://github.com/HHenryD/TAP.

</details>

---

## 17. LongVILA: Scaling Long-Context Visual Language Models for Long Videos

- [ ] LongVILA: Scaling Long-Context Visual Language Models for Long Videos | https://iclr.cc/virtual/2025/poster/27865

- **Link**: https://iclr.cc/virtual/2025/poster/27865

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-context capability is critical for multi-modal foundation models, especially for long video understanding. We introduce LongVILA, a full-stack solution for long-context visual-language models by co-designing the algorithm and system. For model training, we upgrade existing VLMs to support long video understanding by incorporating two additional stages, i.e., long context extension and long video supervised fine-tuning. However, training on long video is computationally and memory intensive. We introduce the long-context Multi-Modal Sequence Parallelism (MM-SP) system that efficiently parallelizes long video training and inference, enabling 2M context length training on 256 GPUs without any gradient checkpointing. LongVILA efficiently extends the number of video frames of VILA from 8 to 2048, achieving 99.8% accuracy in 6,000-frame (more than 1 million tokens) video needle-in-a-haystack. LongVILA-7B demonstrates strong accuracy on 9 popular video benchmarks, e.g., 65.1% VideoMME with subtitle. Besides, MM-SP is 2.1x - 5.7x faster than ring style sequence parallelism and 1.1x - 1.4x faster than Megatron with a hybrid context and tensor parallelism. Moreover, it seamlessly integrates with Hugging Face Transformers.

</details>

---

## 18. Youku Dense Caption: A Large-scale Chinese Video Dense Caption Dataset and Benchmarks

- [ ] Youku Dense Caption: A Large-scale Chinese Video Dense Caption Dataset and Benchmarks | https://iclr.cc/virtual/2025/poster/27881

- **Link**: https://iclr.cc/virtual/2025/poster/27881

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the explosive growth of video content, video captions have emerged as a crucial tool for video comprehension, significantly enhancing the ability to understand and retrieve information from videos. However, most publicly available dense video captioning datasets are in English, resulting in a scarcity of large-scale and high-quality Chinese dense video captioning datasets. To address this gap within the Chinese community and to promote the advancement of Chinese multi-modal models, we develop the first, large-scale, and high-quality Chinese dense video captioning dataset, named Youku Dense Caption. This dataset is sourced from Youku, a prominent Chinese video-sharing website. Youku Dense Caption includes 31,466 complete short videos annotated by 311,921 Chinese captions. To the best of our knowledge, it is currently the largest publicly available dataset for fine-grained Chinese video descriptions. Additionally, we establish several benchmarks for Chinese video-language tasks based on the Youku Dense Caption, including retrieval, grounding, and generation tasks. Extensive experiments and evaluations are conducted on existing state-of-the-art multi-modal models, demonstrating the dataset's utility and the potential for further research.

</details>

---

## 19. Routing Experts: Learning to Route Dynamic Experts in Existing Multi-modal Large Language Models

- [ ] Routing Experts: Learning to Route Dynamic Experts in Existing Multi-modal Large Language Models | https://iclr.cc/virtual/2025/poster/27884

- **Link**: https://iclr.cc/virtual/2025/poster/27884

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, mixture of experts (MoE) has become a popular paradigm for achieving the trade-off between modal capacity and efficiency of multimodal large language models (MLLMs). Different from previous efforts, we are dedicated to exploring the dynamic experts in existing MLLMs and showing that a standard MLLM can also be a mixture of experts. However, achieving this target is still notoriously challenging. The well-trained MLLMs are more accustomed to the fixed pathway and a drastic change in its inference manner also greatly impedes its performance. To address these issues, we propose a novel dynamic expert routing method for existing MLLMs, termed Routing Experts (RoE), which can achieve example-dependent optimal path routing without obvious structure tweaks. Meanwhile, a new structure sparsity regularization is also introduced to force the well-trained MLLMs to learn more short-cut pathways. In addition, we also address the alignment of the training and inference of MLLMs in terms of network routing. To validate RoE, we apply it to a set of existing MLLMs, including LLaVA-1.5, LLaVA-HR and VILA, and conduct extensive experiments on a bunch of VL benchmarks. The experiment results not only show the effectiveness of our RoE in improving MLLMs' efficiency, but also yield obvious advantages over MoE-LLaVA in both performance and speed, e.g.,  an average performance gain of 3.3% on 5 benchmarks while being 1.61 times faster. Our code is anonymously released at https://github.com/DoubtedSteam/RoE

</details>

---

## 20. Text4Seg: Reimagining Image Segmentation as Text Generation

- [ ] Text4Seg: Reimagining Image Segmentation as Text Generation | https://iclr.cc/virtual/2025/poster/27892

- **Link**: https://iclr.cc/virtual/2025/poster/27892

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown exceptional capabilities in vision-language tasks; however, effectively integrating image segmentation into these models remains a significant challenge. In this paper, we introduce Text4Seg, a novel text-as-mask paradigm that casts image segmentation as a text generation problem, eliminating the need for additional decoders and significantly simplifying the segmentation process. Our key innovation is semantic descriptors, a new textual representation of segmentation masks where each image patch is mapped to its corresponding text label. This unified representation allows seamless integration into the auto-regressive training pipeline of MLLMs for easier optimization. We demonstrate that representing an image with $16\times16$ semantic descriptors yields competitive segmentation performance. To enhance efficiency, we introduce the Row-wise Run-Length Encoding (R-RLE), which compresses redundant text sequences, reducing the length of semantic descriptors by 74\% and accelerating inference by $3\times$, without compromising performance. Extensive experiments across various vision tasks, such as referring expression segmentation and comprehension, show that Text4Seg achieves state-of-the-art performance on multiple datasets by fine-tuning different MLLM backbones. Our approach provides an efficient, scalable solution for vision-centric tasks within the MLLM framework.

</details>

---

## 21. ImagineNav: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination

- [ ] ImagineNav: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination | https://iclr.cc/virtual/2025/poster/27914

- **Link**: https://iclr.cc/virtual/2025/poster/27914

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual navigation is an essential skill for home-assistance robots, providing the object-searching ability to accomplish long-horizon daily tasks. Many recent approaches use Large Language Models (LLMs) for commonsense inference to improve exploration efficiency. However, the planning process of LLMs is limited within texts and it is difficult to represent the spatial occupancy and geometry layout only by texts. Both are important for making rational navigation decisions. In this work, we seek to unleash the spatial perception and planning ability of Vision-Language Models (VLMs), and explore whether the VLM, with only on-board camera captured RGB/RGB-D stream inputs, can efficiently finish the visual navigation tasks in a mapless manner. We achieve this by developing the imagination-powered navigation framework ImagineNav, which imagines the future observation images at valuable robot views and translates the complex navigation planning process into a rather simple best-view image selection problem for VLM. To generate appropriate candidate robot views for imagination, we introduce the Where2Imagine module, which is distilled to align with human navigation habits. Finally, to reach the VLM preferred views, an off-the-shelf point-goal navigation policy is utilized. Empirical experiments on the challenging open-vocabulary object navigation benchmarks demonstrates the superiority of our proposed system.

</details>

---

## 22. The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs

- [ ] The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs | https://iclr.cc/virtual/2025/poster/27924

- **Link**: https://iclr.cc/virtual/2025/poster/27924

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) have exhibited impressive capability. However, recently many deficiencies of MLLMs have been found compared to human intelligence, $\textit{e.g.}$, hallucination. To drive the MLLMs study, the community dedicated efforts to building larger benchmarks with complex tasks. In this paper, we propose benchmarking an essential but usually overlooked intelligence: $\textbf{association}$, a human's basic capability to link observation and prior practice memory. To comprehensively investigate MLLM's performance on the association, we formulate the association task and devise a standard benchmark based on adjective and verb semantic concepts. Instead of costly data annotation and curation, we propose a convenient $\textbf{annotation-free}$ construction method transforming the general dataset for our association tasks. Simultaneously, we devise a rigorous data refinement process to eliminate confusion in the raw dataset. Building on this database, we establish three levels of association tasks: single-step, synchronous, and asynchronous associations. Moreover, we conduct a comprehensive investigation into the MLLMs' zero-shot association capabilities, addressing multiple dimensions, including three distinct memory strategies, both open-source and closed-source MLLMs, cutting-edge Mixture-of-Experts (MoE) models, and the involvement of human experts. Our systematic investigation shows that current open-source MLLMs consistently exhibit poor capability in our association tasks, even the currently state-of-the-art GPT-4V(vision) also has a significant gap compared to humans. We believe our benchmark would pave the way for future MLLM studies.  $\textit{Our data and code are available at:} https://mvig-rhos.com/llm_inception.

</details>

---

## 23. Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning

- [ ] Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning | https://iclr.cc/virtual/2025/poster/27935

- **Link**: https://iclr.cc/virtual/2025/poster/27935

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Value-based reinforcement learning (RL) can in principle learn effective policies for a wide range of multi-turn problems, from games to dialogue to robotic control, including via offline RL from static previously collected datasets. However, despite the widespread use of policy gradient methods to train large language models for single turn tasks (e.g., question answering), value-based methods for multi-turn RL in an off-policy or offline setting have proven particularly challenging to scale to the setting of large language models. This setting requires effectively leveraging pretraining, scaling to large architectures with billions of parameters, and training on large datasets, all of which represent major challenges for current value-based RL methods. In this work, we propose a novel offline RL algorithm that addresses these drawbacks, casting Q-learning as a modified supervised fine-tuning (SFT) problem where the probabilities of tokens directly translate to Q-values. In this way we obtain an algorithm that smoothly transitions from maximizing the likelihood of the data during pretraining to learning a near-optimal Q-function during finetuning. Our algorithm has strong theoretical foundations, enjoying performance bounds similar to state-of-the-art Q-learning methods, while in practice utilizing an objective that closely resembles SFT. Because of this, our approach can enjoy the full benefits of the pretraining of language models, without the need to reinitialize any weights before RL finetuning, and without the need to initialize new heads for predicting values or advantages. Empirically, we evaluate our method on both pretrained LLMs and VLMs, on a variety of tasks including both natural language dialogue and robotic manipulation and navigation from images.

</details>

---

## 24. ScImage: How good are multimodal large language models at scientific text-to-image generation?

- [ ] ScImage: How good are multimodal large language models at scientific text-to-image generation? | https://iclr.cc/virtual/2025/poster/27964

- **Link**: https://iclr.cc/virtual/2025/poster/27964

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (LLMs) have demonstrated impressive capabilities in generating high-quality images from textual instructions. However, their performance in generating scientific images—a critical application for accelerating scientific progress—remains underexplored. In this work, we address this gap by introducing ScImage, a benchmark designed to evaluate the multimodal capabilities of LLMs in generating scientific images from textual descriptions. ScImage assesses three key dimensions of understanding: spatial, numeric, and attribute comprehension, as well as their combinations, focusing on the relationships between scientific objects (e.g., squares, circles). We evaluate seven models, GPT-4o, Llama, AutomaTikZ, Dall-E, StableDiffusion, GPT-o1 and Qwen2.5-Coder-Instruct using two modes of output generation: code-based outputs (Python, TikZ) and direct raster image generation. Additionally, we examine four different input languages: English, German, Farsi, and Chinese. Our evaluation, conducted with 11 scientists across three criteria (correctness, relevance, and scientific accuracy), reveals that while GPT4-o produces outputs of decent quality for simpler prompts involving individual dimensions such as spatial, numeric, or attribute understanding in isolation, all models face challenges in this task, especially for more complex prompts. ScImage is available: huggingface.co/datasets/casszhao/ScImage

</details>

---

## 25. LLaVA-MoD: Making LLaVA Tiny via MoE-Knowledge Distillation

- [ ] LLaVA-MoD: Making LLaVA Tiny via MoE-Knowledge Distillation | https://iclr.cc/virtual/2025/poster/27973

- **Link**: https://iclr.cc/virtual/2025/poster/27973

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce LLaVA-MoD, a novel framework designed to enable the efficient training of small-scale Multimodal Language Models ($s$-MLLM) distilling knowledge from large-scale MLLM ($l$-MLLM). Our approach tackles two fundamental challenges in MLLM distillation. First, we optimize the network structure of $s$-MLLM by integrating a sparse Mixture of Experts (MoE) architecture into the language model, striking a balance between computational efficiency and model expressiveness. Second, we propose a progressive knowledge transfer strategy for comprehensive knowledge transfer. This strategy begins with mimic distillation, where we minimize the Kullback-Leibler (KL) divergence between output distributions to enable $s$-MLLM to emulate $s$-MLLM's understanding. Following this, we introduce preference distillation via Preference Optimization (PO), where the key lies in treating $l$-MLLM as the reference model. During this phase, the $s$-MLLM's ability to discriminate between superior and inferior examples is significantly enhanced beyond $l$-MLLM, leading to a better $s$-MLLM that surpasses $l$-MLLM, particularly in hallucination benchmarks.Extensive experiments demonstrate that LLaVA-MoD surpasses existing works across various benchmarks while maintaining a minimal activated parameters and low computational costs. Remarkably, LLaVA-MoD-2B surpasses Qwen-VL-Chat-7B with an average gain of 8.8\%, using merely $0.3\%$ of the training data and 23\% trainable parameters. The results underscore LLaVA-MoD's ability to effectively distill comprehensive knowledge from its teacher model, paving the way for developing efficient MLLMs.

</details>

---

## 26. Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models

- [ ] Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models | https://iclr.cc/virtual/2025/poster/27997

- **Link**: https://iclr.cc/virtual/2025/poster/27997

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive vision-language models (VLMs), like CLIP, have gained popularity for their versatile applicability to various downstream tasks. Despite their successes in some tasks, like zero-shot object recognition, they perform surprisingly poor on other tasks, like attribute recognition. Previous work has attributed these challenges to the modality gap, a separation of image and text in the shared representation space, and to a bias towards objects over other factors, such as attributes. In this analysis paper, we investigate both phenomena thoroughly. We evaluated off-the-shelf VLMs and while the gap's influence on performance is typically overshadowed by other factors, we find indications that closing the gap indeed leads to improvements. Moreover, we find that, contrary to intuition, only few embedding dimensions drive the gap and that the embedding spaces are differently organized. To allow for a clean study of object bias, we introduce a definition and a corresponding measure of it. Equipped with this tool, we find that object bias does not lead to worse performance on other concepts, such as attributes per se. However, why do both phenomena, modality gap and object bias, emerge in the first place? To answer this fundamental question and uncover some of the inner workings of contrastive VLMs, we conducted experiments that allowed us to control the amount of shared information between the modalities. These experiments revealed that the driving factor behind both the modality gap and the object bias, is an information imbalance between images and captions, and unveiled an intriguing connection between the modality gap and entropy of the logits.

</details>

---

## 27. Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology

- [ ] Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology | https://iclr.cc/virtual/2025/poster/28016

- **Link**: https://iclr.cc/virtual/2025/poster/28016

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Histopathology Whole-Slide Images (WSIs) provide an important tool to assess cancer prognosis in computational pathology (CPATH). While existing survival analysis (SA) approaches have made exciting progress, they are generally limited to adopting highly-expressive network architectures and only coarse-grained patient-level labels to learn visual prognostic representations from gigapixel WSIs. Such learning paradigm suffers from critical performance bottlenecks, when facing present scarce training data and standard multi-instance learning (MIL) framework in CPATH. To overcome it, this paper, for the first time, proposes a new Vision-Language-based SA ( VLSA ) paradigm. Concretely, (1) VLSA is driven by pathology VL foundation models. It no longer relies on high-capability networks and shows the advantage of data efficiency . (2) In vision-end, VLSA encodes textual prognostic prior and then employs it as auxiliary signals to guide the aggregating of visual prognostic features at instance level, thereby compensating for the weak supervision in MIL. Moreover, given the characteristics of SA, we propose i) ordinal survival prompt learning to transform continuous survival labels into textual prompts; and ii) ordinal incidence function as prediction target to make SA compatible with VL-based prediction. Notably, VLSA's predictions can be interpreted intuitively by our Shapley values-based method. The extensive experiments on five datasets confirm the effectiveness of our scheme. Our VLSA could pave a new way for SA in CPATH by offering weakly-supervised MIL an effective means to learn valuable prognostic clues from gigapixel WSIs. Our source code is available at https://github.com/liupei101/VLSA.

</details>

---

## 28. Training-Free Diffusion Model Alignment with Sampling Demons

- [ ] Training-Free Diffusion Model Alignment with Sampling Demons | https://iclr.cc/virtual/2025/poster/28034

- **Link**: https://iclr.cc/virtual/2025/poster/28034

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Aligning diffusion models with user preferences has been a key challenge.Existing methods for aligning diffusion models either require retraining or are limited to differentiable reward functions.To address these limitations, we propose a stochastic optimization approach, dubbed Demon , to guide the denoising process at inference time without backpropagation through reward functions or model retraining.Our approach works by controlling noise distribution in denoising steps to concentrate density on regions corresponding to high rewards through stochastic optimization.We provide comprehensive theoretical and empirical evidence to support and validate our approach, including experiments that use non-differentiable sources of rewards such as Visual-Language Model (VLM) APIs and human judgements.To the best of our knowledge, the proposed approach is the first inference-time, backpropagation-free preference alignment method for diffusion models.Our method can be easily integrated with existing diffusion models without further training.Our experiments show that the proposed approach significantly improves the average aesthetics scores for text-to-image generation. Implementation is available at this URL .

</details>

---

## 29. Backdooring Vision-Language Models with Out-Of-Distribution Data

- [ ] Backdooring Vision-Language Models with Out-Of-Distribution Data | https://iclr.cc/virtual/2025/poster/28044

- **Link**: https://iclr.cc/virtual/2025/poster/28044

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of Vision-Language Models (VLMs) represents a significant advancement in integrating computer vision with Large Language Models (LLMs) to generate detailed text descriptions from visual inputs. Despite their growing importance, the security of VLMs, particularly against backdoor attacks, is under explored. Moreover, prior works often assume attackers have access to the original training data, which is often unrealistic. In this paper, we address a more practical and challenging scenario where attackers must rely solely on Out-Of-Distribution (OOD) data. We introduce VLOOD (Backdoor Vision-Language Models using Out-of-Distribution Data), a novel approach with two key contributions: (1) demonstrating backdoor attacks on VLMs in complex image-to-text tasks while minimizing degradation of the original semantics under poisoned inputs, and (2) proposing innovative techniques for backdoor injection without requiring any access to the original training data. Our evaluation on image captioning and visual question answering (VQA) tasks confirms the effectiveness of VLOOD, revealing a critical security vulnerability in VLMs and laying the foundation for future research on securing multimodal models against sophisticated threats.

</details>

---

## 30. Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models

- [ ] Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/28052

- **Link**: https://iclr.cc/virtual/2025/poster/28052

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While recent Large Vision-Language Models (LVLMs) have shown remarkable performance in multi-modal tasks, they are prone to generating hallucinatory text responses that do not align with the given visual input, which restricts their practical applicability in real-world scenarios. In this work, inspired by the observation that the text-to-image generation process is the inverse of image-conditioned response generation in LVLMs, we explore the potential of leveraging text-to-image generative models to assist in mitigating hallucinations in LVLMs. We discover that generative models can offer valuable self-feedback for mitigating hallucinations at both the response and token levels. Building on this insight, we introduce self-correcting Decoding with Generative Feedback (DeGF), a novel training-free algorithm that incorporates feedback from text-to-image generative models into the decoding process to effectively mitigate hallucinations in LVLMs. Specifically, DeGF generates an image from the initial response produced by LVLMs, which acts as an auxiliary visual reference and provides self-feedback to verify and correct the initial response through complementary or contrastive decoding. Extensive experimental results validate the effectiveness of our approach in mitigating diverse types of hallucinations, consistently surpassing state-of-the-art methods across six benchmarks. Code is available at https://github.com/zhangce01/DeGF.

</details>

---

## 31. MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos

- [ ] MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos | https://iclr.cc/virtual/2025/poster/28053

- **Link**: https://iclr.cc/virtual/2025/poster/28053

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Language Language Models (MLLMs) demonstrate the emerging abilities of "world models"---interpreting and reasoning about complex real-world dynamics. To assess these abilities, we posit videos are the ideal medium, as they encapsulate rich representations of real-world dynamics and causalities. To this end, we introduce MMWorld, a new benchmark for multi-discipline, multi-faceted multimodal video understanding. MMWorld distinguishes itself from previous video understanding benchmarks with two unique advantages: (1) multi-discipline, covering various disciplines that often require domain expertise for comprehensive understanding; (2) multi-faceted reasoning, including explanation, counterfactual thinking, future prediction, etc. MMWorld consists of a human-annotated dataset to evaluate MLLMs with questions about the whole videos and a synthetic dataset to analyze MLLMs within a single modality of perception. Together, MMWorld encompasses 1,910 videos across seven broad disciplines and 69 subdisciplines, complete with 6,627 question-answer pairs and associated captions. The evaluation includes 4 proprietary and 11 open-source MLLMs, which struggle on MMWorld (e.g., GPT-4o performs the best with only 62.5% accuracy), showing large room for improvement. Further ablation studies reveal other interesting findings such as models' different skill sets from humans. We hope MMWorld can serve as an essential step towards world model evaluation in videos.

</details>

---

## 32. Diffusion Feedback Helps CLIP See Better

- [ ] Diffusion Feedback Helps CLIP See Better | https://iclr.cc/virtual/2025/poster/28060

- **Link**: https://iclr.cc/virtual/2025/poster/28060

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP), which excels at abstracting open-world representations across domains and modalities, has become a foundation for a variety of vision and multimodal tasks. However, recent studies reveal that CLIP has severe visual shortcomings, such as which can hardly distinguish orientation, quantity, color, structure, etc. These visual shortcomings also limit the perception capabilities of multimodal large language models (MLLMs) built on CLIP. The main reason could be that the image-text pairs used to train CLIP are inherently biased, due to the lack of the distinctiveness of the text and the diversity of images. In this work, we present a simple post-training approach for CLIP models, which largely overcomes its visual shortcomings via a self-supervised diffusion process. We introduce DIVA, which uses the DIffusion model as a Visual Assistant for CLIP. Specifically, DIVA leverages generative feedback from text-to-image diffusion models to optimize CLIP representations, with only images (without corresponding text). We demonstrate that DIVA improves CLIP's performance on the challenging MMVP-VLM benchmark which assesses fine-grained visual abilities to a large extent (e.g., 3-7%), and enhances the performance of MLLMs and vision models on multimodal understanding and segmentation tasks. Extensive evaluation on 29 image classification and retrieval benchmarks confirms that our framework preserves CLIP's strong zero-shot capabilities. The code is publicly available at https://github.com/baaivision/DIVA.

</details>

---

## 33. Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate

- [ ] Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate | https://iclr.cc/virtual/2025/poster/28079

- **Link**: https://iclr.cc/virtual/2025/poster/28079

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have seen significant progress but continue to struggle with persistent reasoning mistakes.Previous methods of self-reflection have been proven limited due to the models’ inherent fixed thinking patterns. While Multi-Agent Debate (MAD) attempts to mitigate this by incorporating multiple agents, it often employs the same reasoning methods, even though assigning different personas to models. This leads to a "fixed mental set", where models rely on homogeneous thought processes without exploring alternative perspectives.In this paper, we introduce Diverse Multi-Agent Debate (DMAD), a method that encourages agents to think with distinct reasoning approaches. By leveraging diverse problem-solving strategies, each agent can gain insights from different perspectives, refining its responses through discussion and collectively arriving at the optimal solution. DMAD effectively breaks the limitations of fixed mental sets. We evaluate DMAD against various prompting techniques, including self-reflection and traditional MAD, across multiple benchmarks using both LLMs and Multimodal LLMs. Our experiments show that DMAD consistently outperforms other methods, delivering better results than MAD in fewer rounds. Code is available at https://github.com/MraDonkey/DMAD.

</details>

---

## 34. C-CLIP: Multimodal Continual Learning for Vision-Language Model

- [ ] C-CLIP: Multimodal Continual Learning for Vision-Language Model | https://iclr.cc/virtual/2025/poster/28116

- **Link**: https://iclr.cc/virtual/2025/poster/28116

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal pre-trained models like CLIP need large image-text pairs for training but often struggle with domain-specific tasks. Since retraining with specialized and historical data incurs significant memory and time costs, it is important to continually learn new domains in the open world while preserving original performance. However, current continual learning research mainly focuses on single-modal scenarios, and the evaluation criteria are insufficient without considering image-text matching performance and the forgetting of zero-shot performance. This work introduces image-caption datasets from various domains and establishes a multimodal vision-language continual learning benchmark. Then, a novel framework named C-CLIP is proposed, which not only prevents forgetting but also enhances new task learning impressively. Comprehensive experiments demonstrate that our method has strong continual learning ability across different domain image-text datasets, and has little forgetting of the original capabilities of zero-shot prediction,  significantly outperforming existing methods.

</details>

---

## 35. MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models

- [ ] MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models | https://iclr.cc/virtual/2025/poster/28145

- **Link**: https://iclr.cc/virtual/2025/poster/28145

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Artificial Intelligence (AI) has demonstrated significant potential in healthcare, particularly in disease diagnosis and treatment planning. Recent progress in Medical Large Vision-Language Models (Med-LVLMs) has opened up new possibilities for interactive diagnostic tools. However, these models often suffer from factual hallucination, which can lead to incorrect diagnoses. Fine-tuning and retrieval-augmented generation (RAG) have emerged as methods to address these issues. However, the amount of high-quality data and distribution shifts between training data and deployment data limit the application of fine-tuning methods. Although RAG is lightweight and effective, existing RAG-based approaches are not sufficiently general to different medical domains and can potentially cause misalignment issues, both between modalities and between the model and the ground truth. In this paper, we propose a versatile multimodal RAG system, MMed-RAG, designed to enhance the factuality of Med-LVLMs. Our approach introduces a domain-aware retrieval mechanism, an adaptive retrieved contexts selection, and a provable RAG-based preference fine-tuning strategy. These innovations make the RAG process sufficiently general and reliable, significantly improving alignment when introducing retrieved contexts. Experimental results across five medical datasets (involving radiology, ophthalmology, pathology) on medical VQA and report generation demonstrate that MMed-RAG can achieve an average improvement of 43.8% in factual accuracy in the factual accuracy of Med-LVLMs.

</details>

---

## 36. Articulate-Anything:  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model

- [ ] Articulate-Anything:  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model | https://iclr.cc/virtual/2025/poster/28149

- **Link**: https://iclr.cc/virtual/2025/poster/28149

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Interactive 3D simulated objects are crucial in AR/VR, animations, and robotics, driving immersive experiences and advanced automation.However, creating these articulated objects requires extensive human effort and expertise, limiting their broader applications. To overcome this challenge, we present Articulate-Anything, a system that automates the articulation of diverse, complex objects from many input modalities, including text, images, and videos. Articulate-Anything leverages vision-language models (VLMs) to generate code that can be compiled into an interactable digital twin for use in standard 3D simulators. Our system exploits existing 3D asset datasets via a mesh retrieval mechanism, along with an actor-critic system that iteratively proposes, evaluates, and refines solutions for articulating the objects, self-correcting errors to achieve a robust out- come. Qualitative evaluations demonstrate Articulate-Anything's capability to articulate complex and even ambiguous object affordances by leveraging rich grounded inputs. In extensive quantitative experiments on the standard PartNet-Mobility dataset, Articulate-Anything substantially outperforms prior work, increasing the success rate from 8.7-11.6\% to 75\% and setting a new bar for state-of-art performance.  We further showcase the utility of our generated assets by using them to train robotic policies for fine-grained manipulation tasks that go beyond basic pick and place.

</details>

---

## 37. VCR: A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text

- [ ] VCR: A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text | https://iclr.cc/virtual/2025/poster/28154

- **Link**: https://iclr.cc/virtual/2025/poster/28154

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Visual Caption Restoration (VCR), a novel vision-language task that challenges models to accurately restore partially obscured texts using pixel-level hints within images through complex reasoning. This task stems from the observation that text embedded in images intrinsically differs from common visual elements and text due to the need to align the modalities of vision, text, and text embedded in images. While many works incorporate text into images for visual question answering, they mostly rely on OCR or masked language modeling, reducing the task to text-based processing. However, text-based processing becomes ineffective in VCR as accurate text restoration depends on the combined information from provided images, context, and subtle cues from the tiny, exposed areas of masked texts. We develop a pipeline to generate synthetic images for the VCR task using image-caption pairs, with adjustable caption visibility to control the task difficulty. With this pipeline, we construct VCR-WIKI for VCR using Wikipedia images with captions, including 2.11M English and 346K Chinese training entities, plus 5K validation and 5K test entities in both languages, each in easy and hard configurations. We also make a hidden test set, VCR-HIDDEN, to avoid potential overfitting on VCR-WIKI. Our results reveal that current vision-language models significantly lag behind human performance in the VCR task, and merely fine-tuning the models on our dataset does not lead to notable improvements. We release VCR-WIKI and the data construction code to facilitate future research.

</details>

---

## 38. Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models

- [ ] Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/28166

- **Link**: https://iclr.cc/virtual/2025/poster/28166

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucination remains a significant challenge in Large Vision-Language Models (LVLMs). To alleviate this issue, some methods, known as contrastive decoding, induce hallucinations by manually disturbing the raw vision or instruction inputs and then mitigate them by contrasting the outputs of the original and disturbed LVLMs. However, these holistic input disturbances sometimes induce potential noise and also double the inference cost. To tackle these issues, we propose a simple yet effective method named $\textit{Self-Introspective Decoding}$ (SID). Our empirical investigations reveal that pre-trained LVLMs can introspectively assess the importance of vision tokens based on preceding vision and text (both instruction and generated) tokens. Leveraging this insight, we develop the Context and Text-aware Token Selection (CT$^2$S) strategy, which preserves only the least important vision tokens after the early decoder layers, thereby adaptively amplify vision-and-text association hallucinations during auto-regressive decoding. This strategy ensures that multimodal knowledge absorbed in the early decoder layers induces multimodal contextual rather than aimless hallucinations, and significantly reduces computation burdens. Subsequently, the original token logits subtract the amplified fine-grained hallucinations, effectively alleviating hallucinations without compromising the LVLMs' general ability. Extensive experiments illustrate SID generates less-hallucination and higher-quality texts across various metrics, without much additional computation cost.

</details>

---

## 39. Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology

- [ ] Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology | https://iclr.cc/virtual/2025/poster/28193

- **Link**: https://iclr.cc/virtual/2025/poster/28193

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Developing agents capable of navigating to a target location based on language instructions and visual information, known as vision-language navigation (VLN), has attracted widespread interest. Most research has focused on ground-based agents, while UAV-based VLN remains relatively underexplored. Recent efforts in UAV vision-language navigation predominantly adopt ground-based VLN settings, relying on predefined discrete action spaces and neglecting the inherent disparities in agent movement dynamics and the complexity of navigation tasks between ground and aerial environments. To address these disparities and challenges, we propose solutions from three perspectives: platform, benchmark, and methodology. To enable realistic UAV trajectory simulation in VLN tasks, we propose the OpenUAV platform,  which features diverse environments, realistic flight control, and extensive algorithmic support. We further construct a target-oriented VLN dataset consisting of approximately 12k trajectories on this platform, serving as the first dataset specifically designed for realistic UAV VLN tasks. To tackle the challenges posed by complex aerial environments, we propose an assistant-guided UAV object search benchmark called UAV-Need-Help, which provides varying levels of guidance information to help UAVs better accomplish realistic VLN tasks. We also propose a UAV navigation LLM that, given multi-view images, task descriptions, and assistant instructions, leverages the multimodal understanding capabilities of the MLLM to jointly process visual and textual information, and performs hierarchical trajectory generation. The evaluation results of our method significantly outperform the baseline models, while there remains a considerable gap between our results and those achieved by human operators, underscoring the challenge presented by the UAV-Need-Help task.

</details>

---

## 40. Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning

- [ ] Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning | https://iclr.cc/virtual/2025/poster/28198

- **Link**: https://iclr.cc/virtual/2025/poster/28198

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While large language models (LLMs) have integrated images, adapting them to graphs remains challenging, limiting their applications in materials and drug design. This difficulty stems from the need for coherent autoregressive generation across texts and graphs. To address this, we introduce Llamole, the first multimodal LLM capable of interleaved text and graph generation, enabling molecular inverse design with retrosynthetic planning. Llamole integrates a base LLM with the Graph Diffusion Transformer and Graph Neural Networks for multi-conditional molecular generation and reaction inference within texts, while the LLM, with enhanced molecular understanding, flexibly controls activation among the different graph modules. Additionally, Llamole integrates A* search with LLM-based cost functions for efficient retrosynthetic planning. We create benchmarking datasets and conduct extensive experiments to evaluate Llamole against in-context learning and supervised fine-tuning. Llamole significantly outperforms 14 adapted LLMs across 12 metrics for controllable molecular design and retrosynthetic planning. Code and model at https://github.com/liugangcode/Llamole.

</details>

---

## 41. PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration

- [ ] PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration | https://iclr.cc/virtual/2025/poster/28208

- **Link**: https://iclr.cc/virtual/2025/poster/28208

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) like CLIP have attracted substantial attention in pathology, serving as backbones for applications such as zero-shot image classification and Whole Slide Image (WSI) analysis. Additionally, they can function as vision encoders when combined with large language models (LLMs) to support broader capabilities. Current efforts to train pathology VLMs rely on pathology image-text pairs from platforms like PubMed, YouTube, and Twitter, which provide limited, unscalable data with generally suboptimal image quality. In this work, we leverage large-scale WSI datasets like TCGA to extract numerous high-quality image patches. We then train a large multimodal model (LMM) to generate captions for extracted images, creating PathGen-1.6M, a dataset containing 1.6 million high-quality image-caption pairs. Our approach involves multiple agent models collaborating to extract representative WSI patches, generating and refining captions to obtain high-quality image-text pairs. Extensive experiments show that integrating these generated pairs with existing datasets to train a pathology-specific CLIP model, PathGen-CLIP, significantly enhances its ability to analyze pathological images, with substantial improvements across nine pathology-related zero-shot image classification tasks and three whole-slide image tasks. Furthermore, we construct 200K instruction-tuning data based on PathGen-1.6M and integrate PathGen-CLIP with the Vicuna LLM to create more powerful multimodal models through instruction tuning. Overall, we provide a scalable pathway for high-quality data generation in pathology, paving the way for next-generation general pathology models. Our dataset, code, and model are open-access at https://github.com/PathFoundation/PathGen-1.6M.

</details>

---

## 42. Interleaved Scene Graphs for Interleaved Text-and-Image Generation Assessment

- [ ] Interleaved Scene Graphs for Interleaved Text-and-Image Generation Assessment | https://iclr.cc/virtual/2025/poster/28211

- **Link**: https://iclr.cc/virtual/2025/poster/28211

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Many real-world user queries (e.g. "How do to make egg fried rice?" ) could benefit from systems capable of generating responses with both textual steps with accompanying images, similar to a cookbook.Models designed to generate interleaved text and images face challenges in ensuring consistency within and across these modalities.To address these challenges, we present ISG, a comprehensive evaluation framework for interleaved text-and-image generation. ISG leverages a scene graph structure to capture relationships between text and image blocks, evaluating responses on four levels of granularity: holistic, structural, block-level, and image-specific. This multi-tiered evaluation allows for a nuanced assessment of consistency, coherence, and accuracy, and provides interpretable question-answer feedback.In conjunction with ISG, we introduce a benchmark, ISG-Bench, encompassing 1,150 samples across 8 categories and 21 subcategories. This benchmark dataset includes complex language-vision dependencies and golden answers to evaluate models effectively on vision-centric tasks such as style transfer, a challenging area for current models. Using ISG-Bench, we demonstrate that recent unified vision-language models perform poorly on generating interleaved content. While compositional approaches that combine separate language and image models show a 111% improvement over unified models at the holistic level, their performance remains suboptimal at both block and image levels.To facilitate future work, we develop ISG-Agent, a baseline agent employing a "plan-execute-refine" pipeline to invoke tools, achieving a 122% performance improvement.

</details>

---

## 43. TULIP: Token-length Upgraded CLIP

- [ ] TULIP: Token-length Upgraded CLIP | https://iclr.cc/virtual/2025/poster/28217

- **Link**: https://iclr.cc/virtual/2025/poster/28217

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We address the challenge of representing long captions in vision-language models, such as CLIP. By design these models are limited by fixed, absolute positional encodings, restricting inputs to a maximum of 77 tokens and hindering performance on tasks requiring longer descriptions. Although recent work has attempted to overcome this limit, their proposed approaches struggle to model token relationships over longer distances and simply extend to a fixed new token length. Instead, we propose a generalizable method, named TULIP, able to upgrade the token length to any length for CLIP-like models. We do so by improving the architecture with relative position encodings, followed by a training procedure that (i) distills the original CLIP text encoder into an encoder with relative position encodings and (ii) enhances the model for aligning longer captions with images. By effectively encoding captions longer than the default 77 tokens, our model outperforms baselines on cross-modal tasks such as retrieval and text-to-image generation. The code repository is available at https://github.com/ivonajdenkoska/tulip.

</details>

---

## 44. Locality Alignment Improves Vision-Language Models

- [ ] Locality Alignment Improves Vision-Language Models | https://iclr.cc/virtual/2025/poster/28233

- **Link**: https://iclr.cc/virtual/2025/poster/28233

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs) have seen growing adoption in recent years, but many still struggle with basic spatial reasoning errors. We hypothesize that this is due to VLMs adopting pre-trained vision backbones, specifically vision transformers (ViTs) trained with image-level supervision and minimal inductive biases. Such models may fail to encode the class contents at each position in the image, and our goal is to resolve this with a vision backbone that effectively captures both local and global image semantics. Our main insight is that we do not require new supervision to learn this capability – pre-trained models contain significant knowledge of local semantics that we can extract and use for scalable self-supervision. We propose a new efficient post-training stage for ViTs called locality alignment and a novel fine-tuning procedure called MaskEmbed that uses a masked reconstruction loss to learn semantic contributions for each image patch. We first evaluate locality alignment with a vision-only benchmark, finding that it improves a model’s performance at patch-level semantic segmentation, especially for strong backbones trained with image-caption pairs (e.g., CLIP and SigLIP). We then train a series of VLMs with and without locality alignment, and show that locality-aligned backbones improve performance across a range of benchmarks, particularly ones that involve spatial understanding (e.g., RefCOCO, OCID-Ref, TallyQA, VSR, AI2D). Overall, we demonstrate that we can efficiently learn local semantic extraction via a locality alignment stage, and that this procedure benefits VLM training recipes that use off-the-shelf vision backbones.

</details>

---

## 45. econSG: Efficient and Multi-view Consistent Open-Vocabulary 3D Semantic Gaussians

- [ ] econSG: Efficient and Multi-view Consistent Open-Vocabulary 3D Semantic Gaussians | https://iclr.cc/virtual/2025/poster/28248

- **Link**: https://iclr.cc/virtual/2025/poster/28248

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The primary focus of most recent works on open-vocabulary neural fields is extracting precise semantic featuresfrom the VLMs and then consolidating them efficiently into a multi-view consistent 3D neural fieldsrepresentation. However, most existing works over-trusted SAM to regularize image-level CLIP without any further refinement. Moreover, several existing works improved efficiency by dimensionality reduction of semantic features from 2D VLMs before fusing with 3DGS semantic fields, which inevitably leads to multi-view inconsistency. In this work, we propose econSG for open-vocabulary semantic segmentation with 3DGS. Our econSG consists of: 1) A Confidence-region Guided Regularization (CRR) that mutually refines SAM and CLIP to get the best of both worlds for precise semantic features with complete and precise boundaries. 2) A low dimensional contextual space to enforce 3D multi-view consistency while improving computational efficiency by fusing backprojected multi-view 2D features and follow by dimensional reduction directly on the fused 3D features instead of operating on each 2D view separately. Our econSG show state-of-the-art performance on four benchmark datasets compared to the existing methods. Furthermore, we are also the most efficient training among all the methods.

</details>

---

## 46. VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning

- [ ] VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning | https://iclr.cc/virtual/2025/poster/28264

- **Link**: https://iclr.cc/virtual/2025/poster/28264

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have become a powerful tool for integrating visual and textual information. Despite their exceptional performance on visual understanding benchmarks, measuring their ability to reason abstractly across multiple images remains a significant challenge. To address this, we introduce VOILA, a large-scale, open-ended, dynamic benchmark designed to evaluate MLLMs' perceptual understanding and abstract relational reasoning. VOILA employs an analogical mapping approach in the visual domain, requiring models to generate an image that completes an analogy between two given image pairs, reference and application, without relying on predefined choices. Our experiments demonstrate that the analogical reasoning tasks in VOILA present a challenge to MLLMs. Through multi-step analysis, we reveal that current MLLMs struggle to comprehend inter-image relationships and exhibit limited capabilities in high-level relational reasoning. Notably, we observe that performance improves when following a multi-step strategy of least-to-most prompting. Comprehensive evaluations on open-source models and GPT-4o show that on text-based answers, the best accuracy for challenging scenarios is 13% (LLaMa 3.2) and even for simpler tasks is only 29% (GPT-4o), while human performance is significantly higher at 70% across both difficulty levels.

</details>

---

## 47. Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision

- [ ] Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision | https://iclr.cc/virtual/2025/poster/28265

- **Link**: https://iclr.cc/virtual/2025/poster/28265

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language model (LM) post-training relies on two stages of human supervision: task demonstrations for supervised finetuning (SFT), followed by preference comparisons for reinforcement learning from human feedback (RLHF). As LMs become more capable, the tasks they are given become harder to supervise. Will post-training remain effective under unreliable supervision? To test this, we simulate unreliable demonstrations and comparison feedback using small LMs and time-constrained humans. We find that in the presence of unreliable supervision, SFT still retains some effectiveness, but DPO (a common RLHF algorithm) fails to improve the model beyond SFT. To address this, we propose iterative label refinement (ILR) as an alternative to RLHF. ILR improves the SFT data by using comparison feedback to decide whether human demonstrations should be replaced by model-generated alternatives, then retrains the model via SFT on the updated data. SFT+ILR outperforms SFT+DPO on several tasks with unreliable supervision (math, coding, and safe instruction-following). Our findings suggest that as LMs are used for complex tasks where human supervision is unreliable, RLHF may no longer be the best use of human comparison feedback; instead, it is better to direct feedback towards improving the training data rather than continually training the model . Our code and data are available at https://github.com/helloelwin/iterative-label-refinement.

</details>

---

## 48. $\gamma-$MoD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models

- [ ] $\gamma-$MoD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models | https://iclr.cc/virtual/2025/poster/28266

- **Link**: https://iclr.cc/virtual/2025/poster/28266

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the significant progress in multimodal large language models (MLLMs), their high computational cost remains a barrier to real-world deployment. Inspired by the mixture of depths (MoDs) in natural language processing, we aim to address this limitation from the perspective of ``activated tokens''. Our key insight is that if most tokens are redundant for the layer computation, then can be skipped directly via the MoD layer. However, directly converting the dense layers of MLLMs to MoD layers leads to substantial performance degradation. To address this issue,  we propose an innovative MoD adaptation strategy for existing MLLMs called $\gamma$-MoD.  In $\gamma$-MoD,   a novel metric is proposed to guide the deployment of MoDs in the MLLM, namely rank of attention maps (ARank). Through ARank, we can effectively identify which layer is redundant and should be replaced with the MoD layer.  Based on ARank,  we further propose two novel designs to maximize the computational sparsity of MLLM while maintaining its performance, namely  shared vision-language router and  masked routing learning.   With these designs, more than 90% dense layers of the MLLM can be effectively converted to the MoD ones. To validate our method, we apply it to three popular MLLMs, and conduct extensive experiments on 9 benchmark datasets.  Experimental results not only validate the significant efficiency benefit of $\gamma$-MoD to existing MLLMs but also confirm its generalization ability on various MLLMs.  For example, with a minor performance drop,  i.e., -1.5%, $\gamma$-MoD can reduce the training and inference time of LLaVA-HR by 31.0% and 53.2%, respectively.

</details>

---

## 49. G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model

- [ ] G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model | https://iclr.cc/virtual/2025/poster/28274

- **Link**: https://iclr.cc/virtual/2025/poster/28274

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have shown remarkable proficiency in human-level reasoning and generation capabilities, which encourages extensive research on their application in mathematical problem solving. However, current work has been largely focused on text-based mathematical problems, with limited investigation in problems involving multi-modal geometric information. Addressing this gap, we aim to enable LLMs to solve geometric problems by understanding image input. We first identify the limitations of current Multimodal Large Language Models (MLLMs) in this area: they struggle to accurately comprehend basic geometric elements and their relationships. To address these challenges, we leverage the inherent attribute of logical structure compactness in geometric figures, utilizing text-only Large Language Models (LLMs) to curate a comprehensive multimodal geometry dataset. This dataset, named Geo170k, contains more than 170K geometric image-caption and question-answer pairs. Utilizing the Geo170k dataset, we introduce G-LLaVA, a model that demonstrates exceptional performance in solving geometric problems. It significantly outperforms GPT4-V on the geometry task of MathVista benchmark with only 7B parameters.

</details>

---

## 50. mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models

- [ ] mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models | https://iclr.cc/virtual/2025/poster/28277

- **Link**: https://iclr.cc/virtual/2025/poster/28277

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models have demonstrated remarkable capabilities in executing instructions for a variety of single-image tasks. Despite this progress, significant challenges remain in modeling long image sequences. In this work, we introduce the versatile multi-modal large language model, mPLUG-Owl3, which enhances the capability for long image-sequence understanding in scenarios that incorporate retrieved image-text knowledge, multimodal in-context examples, and lengthy videos. Specifically, we propose novel hyper attention blocks to efficiently integrate vision and language into a common language-guided semantic space, thereby facilitating the processing of extended multi-image scenarios. We conduct evaluations on 21 benchmarks that cover single/multi-image, and short/long video understanding. mPLUG-Owl3 achieves competitive performance with the state-of-the-art methods while reducing inference time and memory usage by 87.8\% and 48.5\% in average. Moreover, we propose a Distractor Resistance evaluation to assess the ability of models to maintain focus amidst distractions. mPLUG-Owl3 also demonstrates outstanding performance in distractor resistance on ultra-long visual sequence inputs. We hope that mPLUG-Owl3 can contribute to the development of more efficient and powerful multimodal large language models.

</details>

---

## 51. TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data

- [ ] TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data | https://iclr.cc/virtual/2025/poster/28289

- **Link**: https://iclr.cc/virtual/2025/poster/28289

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision and language assistants have enabled new capabilities for interpreting natural images. These approaches have recently been adapted to earth observation data, but they are only able to handle single image inputs, limiting their use for many real-world tasks. In this work, we develop a new vision and language assistant called TEOChat that can engage in conversations about temporal sequences of earth observation data. To train TEOChat, we curate an instruction-following dataset composed of many single image and temporal tasks including building change and damage assessment, semantic change detection, and temporal scene classification. We show that TEOChat can perform a wide variety of spatial and temporal reasoning tasks, substantially outperforming previous vision and language assistants, and even achieving comparable or better performance than several specialist models trained to perform specific tasks. Furthermore, TEOChat achieves impressive zero-shot performance on a change detection and change question answering dataset, outperforms GPT-4o and Gemini 1.5 Pro on multiple temporal tasks, and exhibits stronger single image capabilities than a comparable single image instruction-following model on scene classification, visual question answering, and captioning. We publicly release our data, models, and code at https://github.com/ermongroup/TEOChat .

</details>

---

## 52. FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models

- [ ] FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models | https://iclr.cc/virtual/2025/poster/28315

- **Link**: https://iclr.cc/virtual/2025/poster/28315

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of generative AI is a double-edged sword, which not only facilitates content creation but also makes image manipulation easier and more difficult to detect. Although current image forgery detection and localization (IFDL) methods are generally effective, they tend to face two challenges: \textbf{1)} black-box nature with unknown detection principle, \textbf{2)} limited generalization across diverse tampering methods (e.g., Photoshop, DeepFake, AIGC-Editing). To address these issues, we propose the explainable IFDL task and design FakeShield, a multi-modal framework capable of evaluating image authenticity, generating tampered region masks, and providing a judgment basis based on pixel-level and image-level tampering clues. Additionally, we leverage GPT-4o to enhance existing IFDL datasets, creating the Multi-Modal Tamper Description dataSet (MMTD-Set) for training FakeShield's tampering analysis capabilities. Meanwhile, we incorporate a Domain Tag-guided Explainable Forgery Detection Module (DTE-FDM) and a Multi-modal Forgery Localization Module (MFLM) to address various types of tamper detection interpretation and achieve forgery localization guided by detailed textual descriptions. Extensive experiments demonstrate that FakeShield effectively detects and localizes various tampering techniques, offering an explainable and superior solution compared to previous IFDL methods. The code is available at https://github.com/zhipeixu/FakeShield.

</details>

---

## 53. Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models

- [ ] Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models | https://iclr.cc/virtual/2025/poster/28323

- **Link**: https://iclr.cc/virtual/2025/poster/28323

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have shown remarkable abilities in various visual understanding tasks. However, MLLMs still struggle with fine-grained visual recognition (FGVR), which aims to identify subordinate-level categories from images. This can negatively impact more advanced capabilities of MLLMs, such as object-centric visual question answering and reasoning. In our study, we revisit three quintessential capabilities of MLLMs for FGVR, including object information extraction, category knowledge reserve, object-category alignment, and position of the root cause as a misalignment problem. To address this issue, we present Finedefics, an MLLM that enhances the model's FGVR capability by incorporating informative attribute descriptions of objects into the training phase. We employ contrastive learning on object-attribute pairs and attribute-category pairs simultaneously and use examples from similar but incorrect categories as hard negatives, naturally bringing representations of visual objects and category names closer. Extensive evaluations across multiple popular FGVR datasets demonstrate that Finedefics outperforms existing MLLMs of comparable parameter sizes, showcasing its remarkable efficacy. The code is available at https://github.com/PKU-ICST-MIPL/Finedefics_ICLR2025 .

</details>

---

## 54. ColPali: Efficient Document Retrieval with Vision Language Models

- [ ] ColPali: Efficient Document Retrieval with Vision Language Models | https://iclr.cc/virtual/2025/poster/28336

- **Link**: https://iclr.cc/virtual/2025/poster/28336

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents -often through lengthy and brittle processes-, they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG).To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark $\textit{ViDoRe}$, composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings. The inherent complexity and performance shortcomings of modern systems motivate a new concept; doing document retrieval by directly embedding the images of the document pages. We release $\textit{ColPali}$, a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, $\textit{ColPali}$ largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable. We release models, data, code and benchmarks under open licenses at https://hf.co/vidore.

</details>

---

## 55. Large (Vision) Language Models are Unsupervised In-Context Learners

- [ ] Large (Vision) Language Models are Unsupervised In-Context Learners | https://iclr.cc/virtual/2025/poster/28335

- **Link**: https://iclr.cc/virtual/2025/poster/28335

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in large language and vision-language models have enabled zero-shot inference, allowing models to solve new tasks without task-specific training. Various adaptation techniques such as prompt engineering, In-Context Learning (ICL), and supervised fine-tuning can further enhance the model’s performance on a downstream task, but they require substantial manual effort to construct effective prompts or labeled examples. In this work, we introduce a joint inference framework for fully unsupervised adaptation, eliminating the need for manual prompt engineering and labeled examples. Unlike zero-shot inference, which makes independent predictions, the joint inference makes predictions simultaneously for all inputs in a given task. Since direct joint inference involves computationally expensive optimization, we develop efficient approximation techniques, leading to two unsupervised adaptation methods: unsupervised fine-tuning and unsupervised ICL. We demonstrate the effectiveness of our methods across diverse tasks and models, including language-only Llama-3.1 on natural language processing tasks, reasoning-oriented Qwen2.5-Math on grade school math problems, vision-language OpenFlamingo on vision tasks, and the API-only access GPT-4o model on massive multi-discipline tasks. Our experiments demonstrate substantial improvements over the standard zero-shot approach, including 39% absolute improvement on the challenging GSM8K math reasoning dataset. Remarkably, despite being fully unsupervised, our framework often performs on par with supervised approaches that rely on ground truth labels.

</details>

---

## 56. SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes

- [ ] SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes | https://iclr.cc/virtual/2025/poster/28347

- **Link**: https://iclr.cc/virtual/2025/poster/28347

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Self-supervised pre-trained audio networks have seen widespread adoption in real-world systems, particularly in multi-modal large language models. These networks are often employed in a frozen state, under the assumption that the self-supervised pre-training has sufficiently equipped them to handle real-world audio. However, a critical question remains: how well do these models actually perform in real-world conditions, where audio is typically polyphonic and complex, involving multiple overlapping sound sources? Current audio self-supervised learning (SSL) methods are often benchmarked on datasets predominantly featuring monophonic audio, such as environmental sounds, and speech. As a result, the ability of SSL models to generalize to polyphonic audio, a common characteristic in natural scenarios, remains underexplored. This limitation raises concerns about the practical robustness of SSL models in more realistic audio settings. To address this gap, we introduce Self-Supervised Learning from Audio Mixtures (SSLAM), a novel direction in audio SSL research,  designed to improve the model’s ability to learn from polyphonic data while maintaining strong performance on monophonic data. We thoroughly evaluate SSLAM on standard audio SSL benchmark datasets which are predominantly monophonic and conduct a comprehensive comparative analysis against state-of-the-art (SOTA) methods using a range of high-quality, publicly available polyphonic datasets. SSLAM not only improves model performance on polyphonic audio, but also maintains or exceeds performance on standard audio SSL benchmarks. Notably, it achieves up to a 3.9% improvement on the AudioSet-2M(AS-2M), reaching a mean average precision (mAP) of 50.2. For polyphonic datasets, SSLAM sets new SOTA in both linear evaluation and fine-tuning regimes with performance improvements of up to 9.1%(mAP). These results demonstrate SSLAM's effectiveness in both polyphonic and monophonic soundscapes, significantly enhancing the performance of audio SSL models. Code and pre-trained models are available at https://github.com/ta012/SSLAM.

</details>

---

## 57. Revealing and Reducing Gender Biases in Vision and Language Assistants (VLAs)

- [ ] Revealing and Reducing Gender Biases in Vision and Language Assistants (VLAs) | https://iclr.cc/virtual/2025/poster/28356

- **Link**: https://iclr.cc/virtual/2025/poster/28356

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained large language models (LLMs) have been reliably integrated with visual input for multimodal tasks. The widespread adoption of instruction-tuned image-to-text vision-language assistants (VLAs) like LLaVA and InternVL necessitates evaluating gender biases. We study gender bias in 22 popular open-source VLAs with respect to personality traits, skills, and occupations. Our results show that VLAs replicate human biases likely present in the data, such as real-world occupational imbalances. Similarly, they tend to attribute more skills and positive personality traits to women than to men, and we see a consistent tendency to associate negative personality traits with men. To eliminate the gender bias in these models, we find that fine-tuning-based debiasing methods achieve the best trade-off between debiasing and retaining performance on downstream tasks. We argue for pre-deploying gender bias assessment in VLAs and motivate further development of debiasing strategies to ensure equitable societal outcomes. Code is available at https://github.com/ExplainableML/vla-gender-bias.

</details>

---

## 58. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation

- [ ] Show-o: One Single Transformer to Unify Multimodal Understanding and Generation | https://iclr.cc/virtual/2025/poster/28376

- **Link**: https://iclr.cc/virtual/2025/poster/28376

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a unified transformer, i.e., Show-o, that unifies multimodal understanding and generation. Unlike fully autoregressive models, Show-o unifies autoregressive and (discrete) diffusion modeling to adaptively handle inputs and outputs of various and mixed modalities. The unified model flexibly supports a wide range of vision-language tasks including visual question-answering, text-to-image generation, text-guided inpainting/extrapolation, and mixed-modality generation. Across various benchmarks, it demonstrates comparable or superior performance to existing individual models with an equivalent or larger number of parameters tailored for understanding or generation. This significantly highlights its potential as a next-generation foundation model.

</details>

---

## 59. ChartMoE: Mixture of Diversely Aligned Expert Connector for Chart Understanding

- [ ] ChartMoE: Mixture of Diversely Aligned Expert Connector for Chart Understanding | https://iclr.cc/virtual/2025/poster/28378

- **Link**: https://iclr.cc/virtual/2025/poster/28378

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automatic chart understanding is crucial for content comprehension and document parsing. Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in chart understanding through domain-specific alignment and fine-tuning. However, current MLLMs still struggle to provide faithful data and reliable analysis only based on charts. To address it, we propose ChartMoE, which employs the Mixture of Expert (MoE) architecture to replace the traditional linear projector to bridge the modality gap. Specifically, we train several linear connectors through distinct alignment tasks, which are utilized as the foundational initialization parameters for different experts. Additionally, we introduce ChartMoE-Align, a dataset with nearly 1 million chart-table-JSON-code quadruples to conduct three alignment tasks (chart-table/JSON/code). Combined with the vanilla connector, we initialize different experts diversely and adopt high-quality knowledge learning to further refine the MoE connector and LLM parameters. Extensive experiments demonstrate the effectiveness of the MoE connector and our initialization strategy, e.g., ChartMoE improves the accuracy of the previous state-of-the-art from 80.48% to 84.64% on the ChartQA benchmark.

</details>

---

## 60. Large-scale and Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding

- [ ] Large-scale and Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding | https://iclr.cc/virtual/2025/poster/28403

- **Link**: https://iclr.cc/virtual/2025/poster/28403

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Artificial intelligence (AI) shows great potential in assisting radiologists to improve the efficiency and accuracy of medical image interpretation and diagnosis. However, a versatile AI model requires large-scale data and comprehensive annotations, which are often impractical in medical settings. Recent studies leverage radiology reports as a naturally high-quality supervision for medical images, using contrastive language-image pre-training (CLIP) to develop language-informed models for radiological image interpretation. Nonetheless, these approaches typically contrast entire images with reports, neglecting the local associations between imaging regions and report sentences, which may undermine model performance and interoperability. In this paper, we propose a fine-grained vision-language model (fVLM) for anatomy-level CT image interpretation. Specifically, we explicitly match anatomical regions of CT images with corresponding descriptions in radiology reports and perform contrastive pre-training for each anatomy individually. Fine-grained alignment, however, faces considerable false-negative challenges, mainly from the abundance of anatomy-level healthy samples and similarly diseased abnormalities, leading to ambiguous patient-level pairings. To tackle this issue, we propose identifying false negatives of both normal and abnormal samples and calibrating contrastive learning from patient-level to disease-aware pairing. We curated the largest CT dataset to date, comprising imaging and report data from 69,086 patients, and conducted a comprehensive evaluation of 54 major and important disease (including several most deadly cancers) diagnosis tasks across 15 main anatomies. Experimental results demonstrate the substantial potential of fVLM in versatile medical image interpretation. In the zero-shot classification task, we achieved an average AUC of 81.3% on 54 diagnosis tasks, surpassing CLIP and supervised methods by 12.9% and 8.0%, respectively. Additionally, on the publicly available CT-RATE and Rad-ChestCT benchmarks, our fVLM outperformed the current state-of-the-art methods with absolute AUC gains of 7.4% and 4.8%, respectively.

</details>

---

## 61. TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning

- [ ] TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning | https://iclr.cc/virtual/2025/poster/28421

- **Link**: https://iclr.cc/virtual/2025/poster/28421

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated impressive performance in short video understanding. However, understanding long-form videos still remains challenging for MLLMs. This paper proposes TimeSuite, a collection of new designs to adapt the existing short-form video MLLMs for long video understanding, including a simple yet efficient framework to process long video sequence, a high-quality video dataset for grounded tuning of MLLMs, and a carefully-designed instruction tuning task to explicitly incorporate the grounding supervision in the traditional QA format. Specifically, based on VideoChat, we propose our long-video MLLM, coined as VideoChat-T, by implementing a token shuffling to compress long video tokens and introducing Temporal Adaptive Position Encoding (TAPE) to enhance the temporal awareness of visual representation. Meanwhile, we introduce the TimePro, a comprehensive grounding-centric instruction tuning dataset composed of 9 tasks and 349k high-quality grounded annotations. Notably, we design a new instruction tuning task type, called Temporal Grounded Caption, to peform detailed video descriptions with the corresponding time stamps prediction. This explicit temporal location prediction will guide MLLM to correctly attend on the visual content when generating description, and thus reduce the hallucination risk caused by the LLMs. Experimental results demonstrate that our TimeSuite provides a successful solution to enhance the long video understanding capability of short-form MLLM, achieving improvement of 5.6% and 6.8% on the benchmarks of Egoschema and VideoMME, respectively. In addition, VideoChat-T exhibits robust zero-shot temporal grounding capabilities, significantly outperforming the existing state-of-the-art MLLMs. After fine-tuning, it performs on par with the traditional supervised expert models.

</details>

---

## 62. Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist

- [ ] Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist | https://iclr.cc/virtual/2025/poster/28416

- **Link**: https://iclr.cc/virtual/2025/poster/28416

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Exceptional mathematical reasoning ability is one of the key features that demonstrate the power of large language models (LLMs). How to comprehensively define and evaluate the mathematical abilities of LLMs, and even reflect the user experience in real-world scenarios, has emerged as a critical issue. Current benchmarks predominantly concentrate on problem-solving capabilities, presenting a substantial risk of model overfitting and fails to accurately measure the genuine mathematical reasoning abilities. In this paper, we argue that if a model really understands a problem, it should be robustly and readily applied across a diverse array of tasks. To this end, we introduce MathCheck, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently. MathCheck includes multiple mathematical reasoning tasks and robustness tests to facilitate a comprehensive evaluation of both mathematical reasoning ability and behavior testing. Utilizing MathCheck, we develop MathCheck-GSM and MathCheck-GEO to assess mathematical textual reasoning and multi-modal reasoning capabilities, respectively, serving as upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K. We adopt MathCheck-GSM and MathCheck-GEO to evaluate over 26 LLMs and 17 multi-modal LLMs, assessing their comprehensive mathematical reasoning abilities. Our results demonstrate that while frontier LLMs like GPT-4o continue to excel in various abilities on the checklist, many other model families exhibit a significant decline. Further experiments indicate that, compared to traditional math benchmarks, MathCheck better reflects true mathematical abilities and represents mathematical intelligence more linearly, thereby supporting our design. Using MathCheck, we can also efficiently conduct informative behavior analysis to deeply investigate models. Finally, we show that our proposed checklist paradigm can easily extend to other reasoning tasks for their comprehensive evaluation.

</details>

---

## 63. OS-ATLAS: Foundation Action Model for Generalist GUI Agents

- [ ] OS-ATLAS: Foundation Action Model for Generalist GUI Agents | https://iclr.cc/virtual/2025/poster/28423

- **Link**: https://iclr.cc/virtual/2025/poster/28423

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing efforts in building GUI agents heavily rely on the availability of robust commercial Vision-Language Models (VLMs) such as GPT-4o and GeminiProVision. Practitioners are often reluctant to use open-source VLMs due to their significant performance lag compared to their closed-source counterparts, particularly in GUI grounding and Out-Of-Distribution (OOD) scenarios. To facilitate future research in this area, we developed OS-Atlas—a foundational GUI action model that excels at GUI grounding and OOD agentic tasks through innovations in both data and modeling.We have invested significant engineering effort in developing an open-source toolkit for synthesizing GUI grounding data across multiple platforms, including Windows, Linux, MacOS, Android, and the web. Leveraging this toolkit, we are releasing the largest open-source cross-platform GUI grounding corpus to date, which contains over 13 million GUI elements. This dataset, combined with innovations in model training, provides a solid foundation for OS-Atlas to understand GUI screenshots and generalize to unseen interfaces.Through extensive evaluation across six benchmarks spanning three different platforms (mobile, desktop, and web), OS-Atlas demonstrates significant performance improvements over previous state-of-the-art models. Our evaluation also uncovers valuable insights into continuously improving and scaling the agentic capabilities of open-source VLMs.

</details>

---

## 64. Towards Semantic Equivalence of Tokenization in Multimodal LLM

- [ ] Towards Semantic Equivalence of Tokenization in Multimodal LLM | https://iclr.cc/virtual/2025/poster/28428

- **Link**: https://iclr.cc/virtual/2025/poster/28428

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in processing vision-language tasks. One of the crux of MLLMs lies in vision tokenization, which involves efficiently transforming input visual signals into feature representations that are most beneficial for LLMs. However, existing vision tokenizers, essential for semantic alignment between vision and language, remain problematic. Existing methods aggressively fragment visual input, corrupting the visual semantic integrity. To address this, this paper proposes a novel dynamic Semantic-Equivalent Vision Tokenizer (SeTok), which groups visual features into semantic units via a dynamic clustering algorithm, flexibly determining the number of tokens based on image complexity. The resulting vision tokens effectively preserve semantic integrity and capture both low-frequency and high-frequency visual features. The proposed MLLM (Setokim) equipped with SeTok significantly demonstrates superior performance across various tasks, as evidenced by our experimental results.

</details>

---

## 65. MMR: A Large-scale Benchmark Dataset for Multi-target and Multi-granularity Reasoning Segmentation

- [ ] MMR: A Large-scale Benchmark Dataset for Multi-target and Multi-granularity Reasoning Segmentation | https://iclr.cc/virtual/2025/poster/28436

- **Link**: https://iclr.cc/virtual/2025/poster/28436

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The fusion of Large Language Models (LLMs) with vision models is pioneering new possibilities in user-interactive vision-language tasks. A notable application is reasoning segmentation, where models generate pixel-level segmentation masks by comprehending implicit meanings in human instructions. However, seamless human-AI interaction demands more than just object-level recognition; it requires understanding both objects and the functions of their detailed parts, particularly in multi-target scenarios. For example, when instructing a robot to \textit{“turn on the TV"}, there could be various ways to accomplish this command. Recognizing multiple objects capable of turning on the TV, such as the TV itself or a remote control (multi-target), provides more flexible options and aids in finding the optimized scenario. Furthermore, understanding specific parts of these objects, like the TV's button or the remote's button (part-level), is important for completing the action. Unfortunately, current reasoning segmentation datasets predominantly focus on a single target object-level reasoning, which limits the detailed recognition of an object's parts in multi-target contexts. To address this gap, we construct a large-scale dataset called Multi-target and Multi-granularity Reasoning (MMR). MMR comprises 194K complex and implicit instructions that consider multi-target, object-level, and part-level aspects, based on pre-existing image-mask sets. This dataset supports diverse and context-aware interactions by hierarchically providing object and part information. Moreover, we propose a straightforward yet effective framework for multi-target, object-level, and part-level reasoning segmentation. Experimental results on MMR show that the proposed method can reason effectively in multi-target and multi-granularity scenarios, while the existing reasoning segmentation model still has room for improvement. The dataset is available at \url{https://github.com/jdg900/MMR}.

</details>

---

## 66. Is Your Video Language Model a Reliable Judge?

- [ ] Is Your Video Language Model a Reliable Judge? | https://iclr.cc/virtual/2025/poster/28480

- **Link**: https://iclr.cc/virtual/2025/poster/28480

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As video language models (VLMs) gain more applications in various scenarios,the need for robust and scalable evaluation of their performance becomes increasingly critical. The traditional human expert-based evaluation of VLMs has limitations in consistency and scalability, which sparked interest in automatic methodssuch as employing VLMs to evaluate VLMs. However, the reliability of VLMs asjudges remains underexplored. Existing methods often rely on a single VLM asthe evaluator. However, this approach can be unreliable or biased because such amodel may lack the ability to fully understand the content and may have inherentbiases, ultimately compromising evaluation reliability. A remedy is to apply theprinciple of collective thoughts, aggregating evaluations from multiple VLMs toenhance reliability. This study investigates the efficacy of such approaches, particularly when the pool of judges includes both reliable and unreliable models. Ourfindings reveal that incorporating collective judgments from such a mixed pooldoes not necessarily improve the accuracy of the final evaluation. The inclusion ofless reliable judges can introduce noise, undermining the overall reliability of theoutcomes. To explore the factors that impact evaluation reliability, we fine-tunean underperforming VLM judge, Video-LLaVA, and observe that improved understanding ability alone is insufficient to make VLM judges more reliable. Thesefindings stress the limitations of collective thought approaches and highlight theneed for more advanced methods that can account for the reliability of individualmodels. Our study promotes the development of more reliable evaluation methodsfor VLMs

</details>

---

## 67. CG-Bench: Clue-grounded Question Answering Benchmark for Long Video Understanding

- [ ] CG-Bench: Clue-grounded Question Answering Benchmark for Long Video Understanding | https://iclr.cc/virtual/2025/poster/28509

- **Link**: https://iclr.cc/virtual/2025/poster/28509

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The existing video understanding benchmarks for multimodal large language models (MLLMs) mainly focus on short videos. The few benchmarks for long video understanding often rely on multiple-choice questions (MCQs). Due to the limitations of MCQ evaluations and the advanced reasoning abilities of MLLMs, models can often answer correctly by combining short video insights with elimination, without truly understanding the content. To bridge this gap, we introduce CG-Bench, a benchmark for clue-grounded question answering in long videos. CG-Bench emphasizes the model's ability to retrieve relevant clues, enhancing evaluation credibility. It includes 1,219 manually curated videos organized into 14 primary, 171 secondary, and 638 tertiary categories, making it the largest benchmark for long video analysis. The dataset features 12,129 QA pairs in three question types: perception, reasoning, and hallucination. To address the limitations of MCQ-based evaluation, we develop two novel clue-based methods: clue-grounded white box and black box evaluations, assessing whether models generate answers based on accurate video understanding. We evaluated multiple closed-source and open-source MLLMs on CG-Bench. The results show that current models struggle significantly with long videos compared to short ones, and there is a notable gap between open-source and commercial models. We hope CG-Bench will drive the development of more reliable and capable MLLMs for long video comprehension.

</details>

---

## 68. Agent S: An Open Agentic Framework that Uses Computers Like a Human

- [ ] Agent S: An Open Agentic Framework that Uses Computers Like a Human | https://iclr.cc/virtual/2025/poster/28525

- **Link**: https://iclr.cc/virtual/2025/poster/28525

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present Agent S, an open agentic framework that enables autonomous interaction with computers through Graphical User Interface (GUI), aimed at transforming human-computer interaction by automating complex, multi-step tasks. Agent S addresses three key challenges in automating computer tasks: acquiring domain-specific knowledge, planning over long task horizons, and handling dynamic, non-uniform interfaces. To this end, Agent S introduces experience-augmented hierarchical planning, which learns from external knowledge search and internal experience retrieval at multiple levels, facilitating efficient task planning and subtask execution. In addition, it employs an Agent-Computer Interface (ACI) to better elicit the reasoning and control capabilities of GUI agents based on Multimodal Large Language Models (MLLMs). Evaluation on the OSWorld benchmark shows that Agent S outperforms the baseline by 9.37\% on success rate (an 83.6\% relative improvement) and achieves a new state-of-the-art. Comprehensive analysis highlights the effectiveness of individual components and provides insights for future improvements. Furthermore, Agent S demonstrates broad generalizability to different operating systems on a newly-released WindowsAgentArena benchmark. Code available at https://github.com/simular-ai/Agent-S.

</details>

---

## 69. Enhancing Cognition and Explainability of Multimodal Foundation Models with Self-Synthesized Data

- [ ] Enhancing Cognition and Explainability of Multimodal Foundation Models with Self-Synthesized Data | https://iclr.cc/virtual/2025/poster/28527

- **Link**: https://iclr.cc/virtual/2025/poster/28527

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs), or Vision-Language Models (VLMs), have shown impressive capabilities in a wide range of visual tasks. However, they often struggle with fine-grained visual reasoning, failing to identify domain-specific objectives and provide justifiable explanations for their predictions. To address the above challenge, we propose a novel visual rejection sampling framework to improve the cognition and explainability of LMMs using self-synthesized data. Specifically, visual fine-tuning requires images, queries, and target answers. Our approach begins by synthesizing interpretable answers that include human-verifiable visual features. These features are based on expert-defined concepts, and carefully selected based on their alignment with the image content. After each round of fine-tuning, we apply a reward model-free filtering mechanism to select the highest-quality interpretable answers for the next round of tuning. This iterative process of synthetic data generation and fine-tuning progressively improves the model's ability to generate accurate and reasonable explanations. Experimental results demonstrate the effectiveness of our method in improving both the accuracy and explainability of specialized visual classification tasks.

</details>

---

## 70. Do Vision & Language Decoders use Images and Text equally? How Self-consistent are their Explanations?

- [ ] Do Vision & Language Decoders use Images and Text equally? How Self-consistent are their Explanations? | https://iclr.cc/virtual/2025/poster/28530

- **Link**: https://iclr.cc/virtual/2025/poster/28530

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision and language model (VLM) decoders are currently the best-performing architectures on multimodal tasks. Next to answers, they are able to produce natural language explanations, either in post-hoc or CoT settings. However, it is not clear to what extent they are using the input vision and text modalities when generating answers or explanations.In this work, we investigate if VLMs rely on their input modalities differently when they produce explanations as opposed to answers.We also evaluate the self-consistency of VLM decoders in both post-hoc and CoT explanation settings, by extending existing unimodal tests and measures to VLM decoders. We find that most tested VLMs are less self-consistent than LLMs. Text contributions in all tested VL decoders are more important than image contributions in all examined tasks. However, when comparing explanation generation to answer generation, the contributions of images are significantly stronger for generating explanations compared to answers. This difference is even larger in CoT compared to post-hoc explanations.Lastly, we provide an up-to-date benchmarking of state-of-the-art VL decoders on the VALSE benchmark, which before was restricted to VL encoders. We find that the tested VL decoders still struggle with most phenomena tested by VALSE. We will make our code publicly available.

</details>

---

## 71. Prompt as Knowledge Bank: Boost Vision-language model via Structural Representation for  zero-shot medical detection

- [ ] Prompt as Knowledge Bank: Boost Vision-language model via Structural Representation for  zero-shot medical detection | https://iclr.cc/virtual/2025/poster/28543

- **Link**: https://iclr.cc/virtual/2025/poster/28543

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot medical detection can further improve detection performance without relying on annotated medical images even upon the fine-tuned model, showing great clinical value. Recent studies leverage grounded vision-language models (GLIP) to achieve this by using detailed disease descriptions as prompts for the target disease name during the inference phase.  However, these methods typically treat prompts as equivalent context to the target name, making it difficult to assign specific disease knowledge based on visual information, leading to a coarse alignment between images and target descriptions. In this paper, we propose StructuralGLIP, which introduces an auxiliary branch to encode prompts into a latent knowledge bank layer-by-layer, enabling more context-aware and fine-grained alignment. Specifically, in each layer, we select highly similar features from both the image representation and the knowledge bank, forming structural representations that capture nuanced relationships between image patches and target descriptions. These features are then fused across modalities to further enhance detection performance.Extensive experiments demonstrate that StructuralGLIP achieves a +4.1\% AP improvement over prior state-of-the-art methods across seven zero-shot medical detection benchmarks, and consistently improves fine-tuned models by +3.2\% AP on endoscopy image datasets.

</details>

---

## 72. Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives

- [ ] Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives | https://iclr.cc/virtual/2025/poster/28553

- **Link**: https://iclr.cc/virtual/2025/poster/28553

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have sparked a new wave of AI applications; however, their substantial computational costs and memory demands pose significant challenges to democratizing access to LLMs for a broader audience. Singular Value Decomposition (SVD), a technique studied for decades, offers a hardware-independent and flexibly tunable solution for LLM compression. In this paper, we present new directions using SVD: we first theoretically analyze the optimality of truncating weights and truncating activations, then we further identify three key issues on SVD-based LLM compression, including (1) How can we determine the optimal truncation position for each weight matrix in LLMs? (2) How can we efficiently update the weight matrices based on truncation position? (3) How can we address the inherent "injection" nature that results in the information loss of the SVD? We propose an effective approach, Dobi-SVD , to tackle the three issues. First, we propose a differentiable truncation-value learning mechanism, along with gradient-robust backpropagation, enabling the model to adaptively find the optimal truncation positions. Next, we utilize the Eckart-Young-Mirsky theorem to derive a theoretically optimal weight update formula through rigorous mathematical analysis. Lastly, by observing and leveraging the quantization-friendly nature of matrices after SVD decomposition, we reconstruct a mapping between truncation positions and memory requirements, establishing a bijection from truncation positions to memory. Experimental results show that with a 40\% parameter-compression rate, our method achieves a perplexity of 9.07 on the Wikitext2 dataset with the compressed LLama-7B model, a 78.7\% improvement over the state-of-the-art SVD for LLM compression method. We emphasize that Dobi-SVD is the first to achieve such a high-ratio LLM compression with minimal performance drop.  We also extend our Dobi-SVD to VLM compression, achieving a 20\% increase in throughput with minimal performance degradation. We hope that the inference speedup—up to 12.4x on 12GB NVIDIA Titan Xp GPUs and 3x on 80GB A100 GPUs for LLMs, and 1.2x on 80GB A100 GPUs for VLMs—will bring significant benefits to the broader community such as robotics.

</details>

---

## 73. Boosting the visual interpretability of CLIP via adversarial fine-tuning

- [ ] Boosting the visual interpretability of CLIP via adversarial fine-tuning | https://iclr.cc/virtual/2025/poster/28566

- **Link**: https://iclr.cc/virtual/2025/poster/28566

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

CLIP has achieved great success in visual representation learning and is becoming an important plug-in component for many large multi-modal models like LLaVA and DALL-E. However, the lack of interpretability caused by the intricate image encoder architecture and training process restricts its wider use in high-stake decision making applications. In this work, we propose an unsupervised adversarial fine-tuning (AFT) with norm-regularization to enhance the visual interpretability of CLIP. We provide theoretical analysis showing that AFT has implicit regularization that enforces the image encoder to encode the input features sparsely, directing the network's focus towards meaningful features. Evaluations by both feature attribution techniques and network dissection offer convincing evidence that the visual interpretability of CLIP has significant improvements. With AFT, the image encoder prioritizes pertinent input features, and the neuron within the encoder exhibits better alignment with human-understandable concepts. Moreover, these effects are generalizable to out-of-distribution datasets and can be transferred to downstream tasks. Additionally, AFT enhances the visual interpretability of derived large vision-language models that incorporate the pre-trained CLIP an integral component. The code of this paper is available at the CLIP_AFT GitHub repository .

</details>

---

## 74. MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?

- [ ] MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans? | https://iclr.cc/virtual/2025/poster/28598

- **Link**: https://iclr.cc/virtual/2025/poster/28598

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Comprehensive evaluation of Multimodal Large Language Models (MLLMs) has recently garnered widespread attention in the research community. However, we observe that existing benchmarks present several common barriers that make it difficult to measure the significant challenges that models face in the real world, including: 1) small data scale leads to a large performance variance; 2) reliance on model-based annotations results in restricted data quality; 3) insufficient task difficulty, especially caused by the limited image resolution. To tackle these issues, we introduce MME-RealWorld. Specifically, we collect more than $300$ K images from public datasets and the Internet, filtering $13,366$ high-quality images for annotation. This involves the efforts of professional $25$ annotators and $7$ experts in MLLMs, contributing to $29,429$ question-answer pairs that cover $43$ subtasks across $5$ real-world scenarios, extremely challenging even for humans. As far as we know, **MME-RealWorld is the largest manually annotated benchmark to date, featuring the highest resolution and a targeted focus on real-world applications**. We further conduct a thorough evaluation involving $29$ prominent MLLMs, such as GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet. Our results show that even the most advanced models struggle with our benchmarks, where none of them reach 60\% accuracy. The challenges of perceiving high-resolution images and understanding complex real-world scenarios remain urgent issues to be addressed. The data and evaluation code are released in our Project Page.

</details>

---

## 75. Learning Interleaved Image-Text Comprehension in Vision-Language Large Models

- [ ] Learning Interleaved Image-Text Comprehension in Vision-Language Large Models | https://iclr.cc/virtual/2025/poster/28632

- **Link**: https://iclr.cc/virtual/2025/poster/28632

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The swift progress of Multi-modal Large Models (MLLMs) has showcased their impressive ability to tackle tasks blending vision and language.Yet, most current models and benchmarks cater to scenarios with a narrow scope of visual and textual contexts.These models often fall short when faced with complex comprehension tasks, which involve navigating through a plethora of irrelevant and potentially misleading information in both text and image forms.To bridge this gap, we introduce a new, more demanding task known as Interleaved Image-Text Comprehension (IITC).This task challenges models to discern and disregard superfluous elements in both images and text to accurately answer questions and to follow intricate instructions to pinpoint the relevant image.In support of this task, we further craft a new VEGA dataset, tailored for the IITC task on scientific content, and devised a subtask, Image-Text Association (ITA), to refine image-text correlation skills.Our evaluation of four leading closed-source models, as well as various open-source models using VEGA, underscores the rigorous nature of IITC.Even the most advanced models, such as Gemini-1.5-pro and GPT4V, only achieved modest success.By employing a multi-task, multi-scale post-training strategy, we have set a robust baseline for MLLMs on the IITC task, attaining an $85.8\%$ accuracy rate in image association and a $0.508$ Rouge score. These results validate the effectiveness of our dataset in improving MLLMs capabilities for nuanced image-text comprehension.

</details>

---

## 76. Attribute-based Visual Reprogramming for Vision-Language Models

- [ ] Attribute-based Visual Reprogramming for Vision-Language Models | https://iclr.cc/virtual/2025/poster/28650

- **Link**: https://iclr.cc/virtual/2025/poster/28650

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

*Visual reprogramming* (VR) reuses pre-trained vision models for downstream image classification tasks by adding trainable noise patterns to inputs. When applied to vision-language models (e.g., CLIP), existing VR approaches follow the same pipeline used in vision models (e.g., ResNet, ViT), where ground-truth class labels are inserted into fixed text templates to guide the optimization of VR patterns. This label-based approach, however, overlooks the rich information and diverse attribute-guided textual representations that CLIP can exploit, which may lead to the misclassification of samples. In this paper, we propose ***Attr**ibute-based **V**isual **R**eprogramming* (AttrVR) for CLIP, utilizing ***des**criptive **attr**ibutes* (DesAttrs) and ***dist**inctive **attr**ibutes* (DistAttrs), which respectively represent common and unique feature descriptions for different classes. Besides, as images of the same class may reflect different attributes after VR, AttrVR iteratively refines patterns using the $k$-nearest DesAttrs and DistAttrs for each image sample, enabling more dynamic and sample-specific optimization. Theoretically, AttrVR is shown to reduce intra-class variance and increase inter-class separation. Empirically, it achieves superior performance in 12 downstream tasks for both ViT-based and ResNet-based CLIP. The success of AttrVR facilitates more effective integration of VR from unimodal vision models into vision-language models. Our code is available at https://github.com/tmlr-group/AttrVR.

</details>

---

## 77. PerturboLLaVA: Reducing Multimodal Hallucinations with Perturbative Visual Training

- [ ] PerturboLLaVA: Reducing Multimodal Hallucinations with Perturbative Visual Training | https://iclr.cc/virtual/2025/poster/28657

- **Link**: https://iclr.cc/virtual/2025/poster/28657

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper aims to address the challenge of hallucinations in Multimodal Large Language Models (MLLMs)  particularly for dense image captioning tasks. To tackle the challenge, we identify the current lack of a metric that finely measures the caption quality in concept level. We hereby introduce HalFscore, a novel metric built upon the language graph and is designed to evaluate both the  accuracy and completeness of dense captions at agranular level. Additionally, we identify the root cause of hallucination as the model's over-reliance on its language prior. To address this, we propose PerturboLLaVA, which reduces the model's reliance on the language prior by incorporating adversarially perturbed text during training. This method enhances the model's focus on visual inputs, effectively reducing hallucinations and producing accurate, image-grounded descriptions without incurring additional computational overhead.  PerturboLLaVA significantly improves the fidelity of generated captions, outperforming existing approaches in handling multimodal hallucinations and achieving improved performance across general multimodal benchmarks.

</details>

---

## 78. Noisy Test-Time Adaptation in Vision-Language Models

- [ ] Noisy Test-Time Adaptation in Vision-Language Models | https://iclr.cc/virtual/2025/poster/28661

- **Link**: https://iclr.cc/virtual/2025/poster/28661

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time adaptation (TTA) aims to address distribution shifts between source and target data by relying solely on target data during testing. In open-world scenarios, models often encounter noisy samples, i.e., samples outside the in-distribution (ID) label space. Leveraging the zero-shot capability of pre-trained vision-language models (VLMs), this paper introduces Zero-Shot Noisy TTA (ZS-NTTA), focusing on adapting the model to target data with noisy samples during test-time in a zero-shot manner. In the preliminary study, we reveal that existing TTA methods suffer from a severe performance decline under ZS-NTTA, often lagging behind even the frozen model. We conduct comprehensive experiments to analyze this phenomenon, revealing that the negative impact of unfiltered noisy data outweighs the benefits of clean data during model updating. In addition, as these methods adopt the adapting classifier to implement ID classification and noise detection sub-tasks, the ability of the model in both sub-tasks is largely hampered. Based on this analysis, we propose a novel framework that decouples the classifier and detector, focusing on developing an individual detector while keeping the classifier (including the backbone) frozen. Technically, we introduce the Adaptive Noise Detector (AdaND), which utilizes the frozen model's outputs as pseudo-labels to train a noise detector for detecting noisy samples effectively. To address clean data streams, we further inject Gaussian noise during adaptation, preventing the detector from misclassifying clean samples as noisy. Beyond the ZS-NTTA, AdaND can also improve the zero-shot out-of-distribution (ZS-OOD) detection ability of VLMs. Extensive experiments show that our method outperforms in both ZS-NTTA and ZS-OOD detection. On ImageNet, AdaND achieves a notable improvement of $8.32\%$ in harmonic mean accuracy ($\text{Acc}_\text{H}$) for ZS-NTTA and $9.40\%$ in FPR95 for ZS-OOD detection, compared to state-of-the-art methods. Importantly, AdaND is computationally efficient and comparable to the model-frozen method. The code is publicly available at: https://github.com/tmlr-group/ZS-NTTA.

</details>

---

## 79. REMEDY: Recipe Merging Dynamics in Large Vision-Language Models

- [ ] REMEDY: Recipe Merging Dynamics in Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/28692

- **Link**: https://iclr.cc/virtual/2025/poster/28692

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Model merging has emerged as a powerful technique for combining task-specific vision models into a unified and multi-functional model. Previous methods represented by task arithmetic, have demonstrated effectiveness and scalability in this domain. When large vision-language models (LVLMs) arise with model size scaling up, this design becomes challenging to fuse different instruction-tuned LVLMs for generalization enhancement. The large scale and multi-modal nature of LVLMs present unique obstacles, including constructing reusable and modular components to accommodate the multi-component architecture of LVLMs and the requirement for dynamic fusion based on multi-modal input tokens. To address these challenges, we propose the \textbf{RE}cipe \textbf{ME}rging \textbf{DY}namics (REMEDY) method, a scalable and flexible paradigm for model merging in LVLMs. We first define reusable modules termed \textit{recipes} including the projector and shallow LLM layers, enhancing visual-language understanding. Then, we introduce a modality-aware allocator dynamically generates weights in a one-shot manner based on input relevance to existing recipes, enabling efficient cross-modal knowledge integration. REMEDY thus offers an adaptive solution for LVLMs to tackle both seen (i.e., multi-task learning) and unseen (i.e., zero-shot generalization) tasks. Experimental results demonstrate that our method consistently improves performance on both seen and unseen tasks, underscoring the effectiveness of REMEDY in diverse multi-modal scenarios.

</details>

---

## 80. LLaRA: Supercharging Robot Learning Data for Vision-Language Policy

- [ ] LLaRA: Supercharging Robot Learning Data for Vision-Language Policy | https://iclr.cc/virtual/2025/poster/28695

- **Link**: https://iclr.cc/virtual/2025/poster/28695

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have recently been leveraged to generate robotic actions, forming Vision-Language-Action (VLA) models. However, directly adapting a pretrained VLM for robotic control remains challenging, particularly when constrained by a limited number of robot demonstrations. In this work, we introduce LLaRA: Large Language and Robotics Assistant, a framework that formulates robot action policy as visuo-textual conversations and enables an efficient transfer of a pretrained VLM into a powerful VLA, motivated by the success of visual instruction tuning in Computer Vision. First, we present an automated pipeline to generate conversation-style instruction tuning data for robots from existing behavior cloning datasets, aligning robotic actions with image pixel coordinates. Further, we enhance this dataset in a self-supervised manner by defining six auxiliary tasks, without requiring any additional action annotations. We show that a VLM finetuned with a limited amount of such datasets can produce meaningful action decisions for robotic control. Through experiments across multiple simulated and real-world tasks, we demonstrate that LLaRA achieves state-of-the-art performance while preserving the generalization capabilities of large language models. The code, datasets, and pretrained models are available at https://github.com/LostXine/LLaRA.

</details>

---

## 81. Can We Talk Models Into Seeing the World Differently?

- [ ] Can We Talk Models Into Seeing the World Differently? | https://iclr.cc/virtual/2025/poster/28696

- **Link**: https://iclr.cc/virtual/2025/poster/28696

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unlike traditional vision-only models, vision language models (VLMs) offer an intuitive way to access visual content through language prompting by combining a large language model (LLM) with a vision encoder. However, both the LLM and the vision encoder come with their own set of biases, cue preferences, and shortcuts, which have been rigorously studied in uni-modal models. A timely question is how such (potentially misaligned) biases and cue preferences behave under multi-modal fusion in VLMs. As a first step towards a better understanding, we investigate a particularly well-studied vision-only bias - the texture vs. shape bias and the dominance of local over global information. As expected, we find that VLMs inherit this bias to some extent from their vision encoders. Surprisingly, the multi-modality alone proves to have important effects on the model behavior, i.e., the joint training and the language querying change the way visual cues are processed. While this direct impact of language-informed training on a model's visual perception is intriguing, it raises further questions on our ability to actively steer a model's output so that its prediction is based on particular visual cues of the user's choice. Interestingly, VLMs have an inherent tendency to recognize objects based on shape information, which is different from what a plain vision encoder would do. Further active steering towards shape-based classifications through language prompts is however limited. In contrast, active VLM steering towards texture-based decisions through simple natural language prompts is often more successful.

</details>

---

## 82. MM-EMBED: UNIVERSAL MULTIMODAL RETRIEVAL WITH MULTIMODAL LLMS

- [ ] MM-EMBED: UNIVERSAL MULTIMODAL RETRIEVAL WITH MULTIMODAL LLMS | https://iclr.cc/virtual/2025/poster/28720

- **Link**: https://iclr.cc/virtual/2025/poster/28720

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

State-of-the-art retrieval models typically address a straightforward search scenario, in which retrieval tasks are fixed (e.g., finding a passage to answer a specific question) and only a single modality is supported for both queries and retrieved results. This paper introduces techniques for advancing information retrieval with multimodal large language models (MLLMs), enabling a broader search scenario, termed universal multimodal retrieval, where multiple modalities and diverse retrieval tasks are accommodated. To this end, we first study fine-tuning an MLLM as a bi-encoder retriever on 10 datasets with 16 retrieval tasks. Our empirical results show that the fine-tuned MLLM retriever is capable of understanding challenging queries, composed of both text and image, but it underperforms compared to a smaller CLIP retriever in cross-modal retrieval tasks due to the modality bias exhibited by MLLMs. To address the issue, we propose modality-aware hard negative mining to mitigate the modality bias exhibited by MLLM retrievers. Second, we propose continuously fine-tuning the universal multimodal retriever to enhance its text retrieval capability while preserving multimodal retrieval capability. As a result, our model, MM-Embed, achieves state-of-the-art performance on the multimodal retrieval benchmark M-BEIR, which spans multiple domains and tasks, while also surpassing the state-of-the-art text retrieval model, NV-Embed-v1, on the MTEB retrieval benchmark. Finally, we explore prompting the off-the-shelf MLLMs as zero-shot rerankers to refine the ranking of the candidates from the multimodal retriever. We find that, through prompt-and-reranking, MLLMs can further improve multimodal retrieval when the user queries (e.g., text-image composed queries) are more complex and challenging to understand. These findings also pave the way for advancing universal multimodal retrieval in the future. We release the model weights at: https://huggingface.co/nvidia/MM-Embed.

</details>

---

## 83. Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

- [ ] Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification | https://iclr.cc/virtual/2025/poster/28727

- **Link**: https://iclr.cc/virtual/2025/poster/28727

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have achieved remarkable success in vision understanding, reasoning, and interaction. However, the inference computation and memory increase progressively with the generation of output tokens during decoding, directly affecting the efficacy of MLLMs. Existing methods attempt to reduce the vision context redundancy to achieve efficient MLLMs. Unfortunately, the efficiency benefits of the vision context reduction in the prefill stage gradually diminish during the decoding stage. To address this problem, we proposed a dynamic vision-language context sparsification framework Dynamic-LLaVA, which dynamically reduces the redundancy of vision context in the prefill stage and decreases the memory and computation overhead of the generated language context during decoding. Dynamic-LLaVA designs a tailored sparsification inference scheme for different inference modes, i.e., prefill, decoding with and without KV cache, to achieve efficient inference of MLLMs. In practice, Dynamic-LLaVA can reduce computation consumption by $\sim$75\% in the prefill stage. Meanwhile, throughout the entire generation process of MLLMs, Dynamic-LLaVA reduces the $\sim$50\% computation consumption under decoding without KV cache, while saving $\sim$50\% GPU memory overhead when decoding with KV cache, due to the vision-language context sparsification. Extensive experiments also demonstrate that Dynamic-LLaVA achieves efficient inference for MLLMs with negligible understanding and generation ability degradation or even performance gains compared to the full-context inference baselines. Code is available at https://github.com/Osilly/dynamic_llava.

</details>

---

## 84. Causal Graphical Models for Vision-Language Compositional Understanding

- [ ] Causal Graphical Models for Vision-Language Compositional Understanding | https://iclr.cc/virtual/2025/poster/28753

- **Link**: https://iclr.cc/virtual/2025/poster/28753

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent work has empirically shown that Vision-Language Models (VLMs) struggleto fully understand the compositional properties of the human language, usuallymodeling an image caption as a “bag of words”. As a result, they performpoorly on compositional tasks, which require a deeper understanding of the differententities of a sentence (subject, verb, etc.) jointly with their mutual relationshipsin order to be solved. In this paper, we model the dependency relationsamong textual and visual tokens using a Causal Graphical Model (CGM), built usinga dependency parser, and we train a decoder conditioned by the VLM visualencoder. Differently from standard autoregressive or parallel predictions, our decoder’sgenerative process is partially-ordered following the CGM structure. Thisstructure encourages the decoder to learn only the main causal dependencies ina sentence discarding spurious correlations. Using extensive experiments on fivecompositional benchmarks, we show that our method significantly outperformsall the state-of-the-art compositional approaches by a large margin, and it also improvesover methods trained using much larger datasets. Our model weights and code are publicly available.

</details>

---

## 85. GEVRM: Goal-Expressive Video Generation Model For Robust Visual Manipulation

- [ ] GEVRM: Goal-Expressive Video Generation Model For Robust Visual Manipulation | https://iclr.cc/virtual/2025/poster/28764

- **Link**: https://iclr.cc/virtual/2025/poster/28764

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid development of embodied artificial intelligence, significant progress has been made in vision-language-action (VLA) models for general robot decision-making. However, the majority of existing VLAs fail to account for the inevitable external perturbations encountered during deployment. These perturbations introduce unforeseen state information to the VLA, resulting in inaccurate actions and consequently, a significant decline in generalization performance. The classic internal model control (IMC) principle demonstrates that a closed-loop system with an internal model that includes external input signals can accurately track the reference input and effectively offset the disturbance. We propose a novel closed-loop VLA method GEVRM that integrates the IMC principle to enhance the robustness of robot visual manipulation. The text-guided video generation model in GEVRM can generate highly expressive future visual planning goals. Simultaneously, we evaluate perturbations by simulating responses, which are called internal embeddings and optimized through prototype contrastive learning. This allows the model to implicitly infer and distinguish perturbations from the external environment. The proposed GEVRM achieves state-of-the-art performance on both standard and perturbed CALVIN benchmarks and shows significant improvements in realistic robot tasks.

</details>

---

## 86. HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation

- [ ] HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation | https://iclr.cc/virtual/2025/poster/28776

- **Link**: https://iclr.cc/virtual/2025/poster/28776

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large foundation models have shown strong open-world generalization to complex problems in vision and language, but similar levels of generalization have yet to be achieved in robotics. One fundamental challenge is the lack of robotic data, which are typically obtained through expensive on-robot operation. A promising remedy is to leverage cheaper, off-domain data such as action-free videos, hand-drawn sketches, or simulation data. In this work, we posit that hierarchical vision-language-action (VLA) models can be more effective in utilizing off-domain data than standard monolithic VLA models that directly finetune vision-language models (VLMs) to predict actions.In particular, we study a class of hierarchical VLA models, where the high-level VLM is finetuned to produce a coarse 2D path indicating the desired robot end-effector trajectory given an RGB image and a task description. The intermediate 2D path prediction is then served as guidance to the low-level, 3D-aware control policy capable of precise manipulation. Doing so alleviates the high-level VLM from fine-grained action prediction, while reducing the low-level policy's burden on complex task-level reasoning.We show that, with the hierarchical design, the high-level VLM can transfer across significant domain gaps between the off-domain finetuning data and real-robot testing scenarios, including differences in embodiments, dynamics, visual appearances, and task semantics, etc.In the real-robot experiments, we observe an average of 20% improvement in success rate across seven different axes of generalization over OpenVLA, representing a 50% relative gain.Visual results are provided at: https://hamster-robot.github.io/

</details>

---

## 87. SCBench: A KV Cache-Centric Analysis of Long-Context Methods

- [ ] SCBench: A KV Cache-Centric Analysis of Long-Context Methods | https://iclr.cc/virtual/2025/poster/28803

- **Link**: https://iclr.cc/virtual/2025/poster/28803

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-context Large Language Models (LLMs) have enabled numerous downstream applications but also introduced significant challenges related to computational and memory efficiency. To address these challenges, optimizations for long-context inference have been developed, centered around the KV cache. However, existing benchmarks often evaluate in single-request, neglecting the full lifecycle of the KV cache in real-world use. This oversight is particularly critical, as KV cache reuse has become widely adopted in LLMs inference frameworks, such as vLLM and SGLang, as well as by LLM providers, including OpenAI, Microsoft, Google, and Anthropic. To address this gap, we introduce SCBENCH (SharedContextBENCH), a comprehensive benchmark for evaluating long-context methods from a KV cache centric perspective: 1) KV cache generation, 2) KV cache compression, 3) KV cache retrieval, and 4) KV cache loading. Specifically, SCBench uses test examples with shared context, ranging 12 tasks with two shared context modes, covering four categories of long-context capabilities: string retrieval, semantic retrieval, global information, and multi-task. With SCBench, we provide an extensive KV cache-centric analysis of eight categories long-context solutions, including Gated Linear RNNs (Codestal-Mamba), Mamba-Attention hybrids (Jamba-1.5-Mini), and efficient methods such as sparse attention, KV cache dropping, quantization, retrieval, loading, and prompt compression. The evaluation is conducted on six Transformer-based long-context LLMs: Llama-3.1-8B/70B, Qwen2.5-72B/32B, Llama-3-8B-262K, and GLM-4-9B. Our findings show that sub-O(n) memory methods suffer in multi-turn scenarios, while sparse encoding with O(n) memory and sub-O(n^2) pre-filling computation perform robustly. Dynamic sparsity yields more expressive KV caches than static patterns, and layer-level sparsity in hybrid architectures reduces memory usage with strong performance. Additionally, we identify attention distribution shift issues in long-generation scenarios.

</details>

---

## 88. DocMIA: Document-Level Membership Inference Attacks against DocVQA Models

- [ ] DocMIA: Document-Level Membership Inference Attacks against DocVQA Models | https://iclr.cc/virtual/2025/poster/28823

- **Link**: https://iclr.cc/virtual/2025/poster/28823

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Document Visual Question Answering (DocVQA) has introduced a new paradigm for end-to-end document understanding, and quickly became one of the standard benchmarks for multimodal LLMs. Automating document processing workflows, driven by DocVQA models, presents significant potential for many business sectors. However, documents tend to contain highly sensitive information, raising concerns about privacy risks associated with training such DocVQA models. One significant privacy vulnerability, exploited by the membership inference attack, is the possibility for an adversary to determine if a particular record was part of the model's training data. In this paper, we introduce two novel membership inference attacks tailored specifically to DocVQA models. These attacks are designed for two different adversarial scenarios: a white-box setting, where the attacker has full access to the model architecture and parameters, and a black-box setting, where only the model's outputs are available. Notably, our attacks assume the adversary lacks access to auxiliary datasets, which is more realistic in practice but also more challenging. Our unsupervised methods outperform existing state-of-the-art membership inference attacks across a variety of DocVQA models and datasets, demonstrating their effectiveness and highlighting the privacy risks in this domain.

</details>

---

## 89. Black Sheep in the Herd: Playing with Spuriously Correlated Attributes for Vision-Language Recognition

- [ ] Black Sheep in the Herd: Playing with Spuriously Correlated Attributes for Vision-Language Recognition | https://iclr.cc/virtual/2025/poster/28838

- **Link**: https://iclr.cc/virtual/2025/poster/28838

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Few-shot adaptation for Vision-Language Models (VLMs) presents a dilemma: balancing in-distribution accuracy with out-of-distribution generalization. Recent research has utilized low-level concepts such as visual attributes to enhance generalization. However, this study reveals that VLMs overly rely on a small subset of attributes on decision-making, which co-occur with the category but are not inherently part of it, termed spuriously correlated attributes. This biased nature of VLMs results in poor generalization. To address this, 1) we first propose Spurious Attribute Probing (SAP), identifying and filtering out these problematic attributes to significantly enhance the generalization of existing attribute-based methods; 2) We introduce Spurious Attribute Shielding (SAS), a plug-and-play module that mitigates the influence of these attributes on prediction, seamlessly integrating into various Parameter-Efficient Fine-Tuning (PEFT) methods. In experiments, SAP and SAS significantly enhance accuracy on distribution shifts across 11 datasets and 3 generalization tasks without compromising downstream performance, establishing a new state-of-the-art benchmark.

</details>

---

## 90. Vision Language Models are In-Context Value Learners

- [ ] Vision Language Models are In-Context Value Learners | https://iclr.cc/virtual/2025/poster/28853

- **Link**: https://iclr.cc/virtual/2025/poster/28853

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Predicting temporal progress from visual trajectories is important for intelligent robots that can learn, adapt, and improve. However, learning such progress estimator, or temporal value function, across different tasks and domains requires both a large amount of diverse data and methods which can scale and generalize. To address these challenges, we present Generative Value Learning (GVL), a universal value function estimator that leverages the world knowledge embedded in vision-language models (VLMs) to predict task progress. Naively asking a VLM to predict values for a video sequence performs poorly due to the strong temporal correlation between successive frames. Instead, GVL poses value estimation as a temporal ordering problem over shuffled video frames; this seemingly more challenging task encourages VLMs to more fully exploit their underlying semantic and temporal grounding capabilities to differentiate frames based on their perceived task progress, consequently producing significantly better value predictions. Without any robot or task specific training, GVL can in-context zero-shot and few-shot predict effective values for more than 300 distinct real-world tasks across diverse robot platforms, including challenging bimanual manipulation tasks. Furthermore, we demonstrate that GVL permits flexible multi-modal in-context learning via examples from heterogeneous tasks and embodiments, such as human videos. The generality of GVL enables various downstream applications pertinent to visuomotor policy learning, including dataset filtering, success detection, and value-weighted regression -- all without any model training or finetuning.

</details>

---

## 91. BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games

- [ ] BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games | https://iclr.cc/virtual/2025/poster/28856

- **Link**: https://iclr.cc/virtual/2025/poster/28856

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) and Vision Language Models (VLMs) possess extensive knowledge and exhibit promising reasoning abilities, however, they still struggle to perform well in complex, dynamic environments. Real-world tasks require handling intricate interactions, advanced spatial reasoning, long-term planning, and continuous exploration of new strategies—areas in which we lack effective methodologies for comprehensively evaluating these capabilities. To address this gap, we introduce BALROG, a novel benchmark designed to assess the agentic capabilities of LLMs and VLMs through a diverse set of challenging games. Our benchmark incorporates a range of existing reinforcement learning environments with varying levels of difficulty, including tasks that are solvable by non-expert humans in seconds to extremely challenging ones that may take years to master (e.g., the NetHack Learning Environment). We devise fine-grained metrics to measure performance and conduct an extensive evaluation of several popular open-source and closed-source LLMs and VLMs. Our findings indicate that while current models achieve partial success in the easier games, they struggle significantly with more challenging tasks. Notably, we observe severe deficiencies in vision-based decision-making, as several models perform worse when visual representations of the environments are provided. We release BALROG as an open and user-friendly benchmark to facilitate future research and development in the agentic community. Code and Leaderboard at balrogai.com

</details>

---

## 92. Teaching Human Behavior Improves Content Understanding Abilities Of VLMs

- [ ] Teaching Human Behavior Improves Content Understanding Abilities Of VLMs | https://iclr.cc/virtual/2025/poster/28866

- **Link**: https://iclr.cc/virtual/2025/poster/28866

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Communication is defined as " Who says what to whom with what effect." A message from a communicator generates downstream receiver effects, also known as behavior. Receiver behavior, being a downstream effect of the message, carries rich signals about it. Even after carrying signals about the message, the behavior signal is often ignored while training vision-language models. We show that training VLMs on receiver behavior can actually help improve their content-understanding abilities. We demonstrate that training VLMs to predict receiver behaviors, such as likes, comments, and replay graphs, which are available at scale, enhances the VLM's performance across a broad range of downstream content understanding tasks. We show this performance increase over 6 types of behavior, 46 different tasks covering image, video, text, and audio over 26 benchmark datasets across both zero-shot and fine-tuning settings, outperforming many supervised baselines on diverse tasks ranging from emotion recognition to captioning by up to 150%. We note that since receiver behavior, such as likes, comments, and replay graphs, is collected by default on the internet and does not need any human annotations to be useful, the performance improvement we get after training on this data is essentially free lunch.  We also release BLIFT , our Behaviour-LLaVA IFT dataset comprising 730k images and videos with their receiver behavior collected from multiple platforms on which we train our models to achieve this. The dataset and code are available at behavior-in-the-wild.github.io/behavior-llava .

</details>

---

## 93. MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models

- [ ] MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/28892

- **Link**: https://iclr.cc/virtual/2025/poster/28892

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual preference alignment involves training Large Vision-Language Models (LVLMs) to predict human preferences between visual inputs. This is typically achieved by using labeled datasets of chosen/rejected pairs and employing optimization algorithms like direct preference optimization (DPO).Existing visual alignment methods, primarily designed for single-image scenarios, struggle to effectively handle the complexity of multi-image tasks due to the scarcity of diverse training data and the high cost of annotating chosen/rejected pairs.We present Multi-Image Augmented Direct Preference Optimization (MIA-DPO), a visual preference alignment approach that effectively handles multi-image inputs.MIA-DPO mitigates the scarcity of diverse multi-image training data by extending single-image data with unrelated images arranged in grid collages or pic-in-pic formats, significantly reducing the costs associated with multi-image data annotations.Our observation reveals that attention values of LVLMs vary considerably across different images. We use attention values to identify and filter out rejected responses the model may have mistakenly focused on.Our attention-aware selection for constructing the chosen/rejected pairs without relying on (i) human annotation, (ii) extra data, and (iii) external models or APIs.MIA-DPO is compatible with various architectures and outperforms existing methods on five multi-image benchmarks, achieving an average performance boost of 3.0% on LLaVA-v1.5 and 4.3% on the recent InternLM-XC2.5.Moreover, MIA-DPO has a minimal effect on the model's ability to understand single images.

</details>

---

## 94. How Does Vision-Language Adaptation Impact the Safety of Vision Language Models?

- [ ] How Does Vision-Language Adaptation Impact the Safety of Vision Language Models? | https://iclr.cc/virtual/2025/poster/28921

- **Link**: https://iclr.cc/virtual/2025/poster/28921

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language adaptation (VL adaptation) transforms Large Language Models (LLMs) into Large Vision-Language Models (LVLMs) for multimodal tasks, but this process often compromises the inherent safety capabilities embedded in the original LLMs. Despite potential harmfulness due to weakened safety measures, in-depth analysis on the effects of VL adaptation on safety remains under-explored. This study examines how VL adaptation influences safety and evaluates the impact of safety fine-tuning methods. Our analysis reveals that safety degradation occurs during VL adaptation, even when the training data is safe. While safety tuning techniques like supervised fine-tuning with safety datasets or reinforcement learning from human feedback mitigate some risks, they still lead to safety degradation and a reduction in helpfulness due to over-rejection issues. Further analysis of internal model weights suggests that VL adaptation may impact certain safety-related layers, potentially lowering overall safety levels. Additionally, our findings demonstrate that the objectives of VL adaptation and safety tuning are divergent, which often results in their simultaneous application being suboptimal. To address this, we suggest the weight merging approach as an optimal solution effectively reducing safety degradation while maintaining helpfulness. These insights help guide the development of more reliable and secure LVLMs for real-world applications.

</details>

---

## 95. Transformer-Squared: Self-adaptive LLMs

- [ ] Transformer-Squared: Self-adaptive LLMs | https://iclr.cc/virtual/2025/poster/28974

- **Link**: https://iclr.cc/virtual/2025/poster/28974

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse tasks. We introduce Transformer-Squared, a novel self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices. During inference, Transformer-Squared employs a two-pass mechanism: first, a dispatch system identifies the task properties, and then task-specific 'expert' vectors, trained using reinforcement learning, are dynamically mixed to obtain targeted behavior for the incoming prompt. Our method consistently outperforms ubiquitous approaches such as LoRA, with fewer parameters and greater efficiency. Furthermore, Transformer-Squared demonstrates versatility across different LLM architectures and modalities, including vision-language tasks. Transformer-Squared represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of LLMs, paving the way for truly dynamic, self-organizing AI systems.

</details>

---

## 96. Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling

- [ ] Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling | https://iclr.cc/virtual/2025/poster/28988

- **Link**: https://iclr.cc/virtual/2025/poster/28988

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The majority of online continual learning (CL) advocates single-epoch training and imposes restrictions on the size of replay memory. However, single-epoch training would incur a different amount of computations per CL algorithm, and the additional storage cost to store logit or model in addition to replay memory is largely ignored in calculating the storage budget. Arguing different computational and storage budgets hinder fair comparison among CL algorithms in practice, we propose to use floating point operations (FLOPs) and total memory size in Byte as a metric for computational and memory budgets, respectively, to compare and develop CL algorithms in the same ‘total resource budget.’ To improve a CL method in a limited total budget, we propose adaptive layer freezing that does not update the layers for less informative batches to reduce computational costs with a negligible loss of accuracy. In addition, we propose a memory retrieval method that allows the model to learn the same amount of knowledge as using random retrieval in fewer iterations. Empirical validations on the CIFAR-10/100, CLEAR-10/100, and ImageNet-1K datasets demonstrate that the proposed approach outperforms the state-of-the-art methods within the same total budget. Furthermore, we validate its effectiveness in the Multi-modal Concept incremental Learning setup with multimodal large language models, such as LLaVA-1.5-7B. Code is available at https://github.com/snumprlab/budgeted-cl.

</details>

---

## 97. Morphing Tokens Draw Strong Masked Image Models

- [ ] Morphing Tokens Draw Strong Masked Image Models | https://iclr.cc/virtual/2025/poster/29006

- **Link**: https://iclr.cc/virtual/2025/poster/29006

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Masked image modeling (MIM) has emerged as a promising approach for pre-training Vision Transformers (ViTs). MIMs predict masked tokens token-wise to recover target signals that are tokenized from images or generated by pre-trained models like vision-language models. While using tokenizers or pre-trained models is viable, they often offer spatially inconsistent supervision even for neighboring tokens, hindering models from learning discriminative representations. Our pilot study identifies spatial inconsistency in supervisory signals and suggests that addressing it can improve representation learning. Building upon this insight, we introduce Dynamic Token Morphing (DTM), a novel method that dynamically aggregates tokens while preserving context to generate contextualized targets, thereby likely reducing spatial inconsistency. DTM is compatible with various SSL frameworks; we showcase significantly improved MIM results, barely introducing extra training costs. Our method facilitates MIM training by using more spatially consistent targets, resulting in improved training trends as evidenced by lower losses. Experiments on ImageNet-1K and ADE20K demonstrate DTM's superiority, which surpasses complex state-of-the-art MIM methods. Furthermore, the evaluation of transfer learning on downstream tasks like iNaturalist, along with extensive empirical studies, supports DTM's effectiveness.

</details>

---

## 98. TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models

- [ ] TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models | https://iclr.cc/virtual/2025/poster/29025

- **Link**: https://iclr.cc/virtual/2025/poster/29025

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Causal language models have demonstrated remarkable capabilities, but their size poses significant challenges for deployment in resource-constrained environments. Knowledge distillation, a widely-used technique for transferring knowledge from a large teacher model to a small student model, presents a promising approach for model compression.A significant remaining issue lies in the major differences between teacher and student models, namely the substantial capacity gap, mode averaging, and mode collapse, which pose barriers during distillation.To address these issues, we introduce $\textit{Temporally Adaptive Interpolated Distillation (TAID)}$, a novel knowledge distillation approach that dynamically interpolates student and teacher distributions through an adaptive intermediate distribution, gradually shifting from the student's initial distribution towards the teacher's distribution. We provide a theoretical analysis demonstrating TAID's ability to prevent mode collapse and empirically show its effectiveness in addressing the capacity gap while balancing mode averaging and mode collapse.Our comprehensive experiments demonstrate TAID's superior performance across various model sizes and architectures in both instruction tuning and pre-training scenarios. Furthermore, we showcase TAID's practical impact by developing two state-of-the-art compact foundation models: $\texttt{TAID-LLM-1.5B}$ for language tasks and $\texttt{TAID-VLM-2B}$ for vision-language tasks.These results demonstrate TAID's effectiveness in creating high-performing and efficient models, advancing the development of more accessible AI technologies.

</details>

---

## 99. VL-ICL Bench: The Devil in the Details of Multimodal In-Context Learning

- [ ] VL-ICL Bench: The Devil in the Details of Multimodal In-Context Learning | https://iclr.cc/virtual/2025/poster/29026

- **Link**: https://iclr.cc/virtual/2025/poster/29026

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) famously exhibit emergent in-context learning (ICL) - the ability to rapidly adapt to new tasks using few-shot examples provided as a prompt, without updating the model's weights. Built on top of LLMs, vision large language models (VLLMs) have advanced significantly in areas such as recognition, reasoning, and grounding. However, investigations into multimodal ICL have predominantly focused on few-shot visual question answering (VQA), and image captioning, which we will show neither exploit the strengths of ICL, nor test its limitations. The broader capabilities and limitations of multimodal ICL remain under-explored. In this study, we introduce a comprehensive benchmark VL-ICL Bench for multimodal in-context learning, encompassing a broad spectrum of tasks that involve both images and text as inputs and outputs, and different types of challenges, from {perception to reasoning and long context length}. We evaluate the abilities of state-of-the-art VLLMs against this benchmark suite, revealing their diverse strengths and weaknesses, and showing that even the most advanced models, such as GPT-4, find the tasks challenging. By highlighting a range of new ICL tasks, and the associated strengths and limitations of existing models, we hope that our dataset will inspire future work on enhancing the in-context learning capabilities of VLLMs, as well as inspire new applications that leverage VLLM ICL. Project page is at https://ys-zong.github.io/VL-ICL/

</details>

---

## 100. Towards Interpreting Visual Information Processing in Vision-Language Models

- [ ] Towards Interpreting Visual Information Processing in Vision-Language Models | https://iclr.cc/virtual/2025/poster/29034

- **Link**: https://iclr.cc/virtual/2025/poster/29034

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) are powerful tools for processing and understanding text and images. We study the processing of visual tokens in the language model component of LLaVA, a prominent VLM. Our approach focuses on analyzing the localization of object information, the evolution of visual token representations across layers, and the mechanism of integrating visual information for predictions. Through ablation studies, we demonstrated that object identification accuracy drops by over 70\% when object-specific tokens are removed. We observed that visual token representations become increasingly interpretable in the vocabulary space across layers, suggesting an alignment with textual tokens corresponding to image content. Finally, we found that the model extracts object information from these refined representations at the last token position for prediction, mirroring the process in text-only language models for factual association tasks. These findings provide crucial insights into how VLMs process and integrate visual information, bridging the gap between our understanding of language and vision models, and paving the way for more interpretable and controllable multimodal systems.

</details>

---

## 101. CertainlyUncertain: A Benchmark and Metric for Multimodal Epistemic and Aleatoric Awareness

- [ ] CertainlyUncertain: A Benchmark and Metric for Multimodal Epistemic and Aleatoric Awareness | https://iclr.cc/virtual/2025/poster/29051

- **Link**: https://iclr.cc/virtual/2025/poster/29051

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The ability to acknowledge the inevitable uncertainty in their knowledge and reasoning is a prerequisite for AI systems to be truly truthful and reliable. In this paper, we present a taxonomy of uncertainty specific to vision-language AI systems, distinguishing between epistemic uncertainty (arising from a lack of information) and aleatoric uncertainty (due to inherent unpredictability), and further explore finer categories within. Based on this taxonomy, we synthesize a benchmark dataset, CertainlyUncertain, featuring 178K visual question answering (VQA) samples as contrastive pairs. This is achieved by 1) inpainting images to make previously answerable questions into unanswerable ones; and 2) using image captions to prompt large language models for both answerable and unanswerable questions. Additionally, we introduce a new metric confidence-weighted accuracy, that is well correlated with both accuracy and calibration error, to address the shortcomings of existing metrics. Despite the recent rapid progress in vision-language models (VLMs), evaluations on our benchmark show that they perform poorly in uncertain scenarios. Further experiments demonstrate that supervised fine-tuning with CertainlyUncertain enhances the performance of VLMs, and reduces the calibration error. These improvements extend beyond our benchmark to existing refusal-oriented datasets and show positive results on reducing hallucinations, while maintaining performance on standard VQA benchmarks. Our work underscores the importance of addressing uncertainty in vision-language AI systems to improve their reliability and trustworthiness in real-world applications.

</details>

---

## 102. Fine-Grained Verifiers: Preference Modeling as Next-token Prediction in Vision-Language Alignment

- [ ] Fine-Grained Verifiers: Preference Modeling as Next-token Prediction in Vision-Language Alignment | https://iclr.cc/virtual/2025/poster/29059

- **Link**: https://iclr.cc/virtual/2025/poster/29059

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The recent advancements in large language models (LLMs) and pre-trained vision models have accelerated the development of vision-language large models (VLLMs), enhancing the interaction between visual and linguistic modalities. Despite their notable success across various domains, VLLMs face challenges in modality alignment, which can lead to issues like hallucinations and unsafe content generation. Current alignment techniques often rely on coarse feedback and external datasets, limiting scalability and performance. In this paper, we propose FiSAO (Fine-Grained Self-Alignment Optimization), a novel self-alignment method that utilizes the model’s own visual encoder as a fine-grained verifier to improve vision-language alignment without the need for additional data. By leveraging token-level feedback from the vision encoder, FiSAO significantly improves vision-language alignment, even surpassing traditional preference tuning methods that require additional data. Through both theoretical analysis and experimental validation, we demonstrate that FiSAO effectively addresses the misalignment problem in VLLMs, marking the first instance of token-level rewards being applied to such models.  Our code is avaliable at \url{https://anonymous.4open.science/r/FISAO-57F0/}.

</details>

---

## 103. $\mathbb{X}$-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs

- [ ] $\mathbb{X}$-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs | https://iclr.cc/virtual/2025/poster/29076

- **Link**: https://iclr.cc/virtual/2025/poster/29076

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Learning good representations involves capturing the diverse ways in which data samples relate. Contrastive loss—an objective matching related samples—underlies methods from self-supervised to multimodal learning. Contrastive losses, however, can be viewed more broadly as modifying a similarity graph to indicate how samples should relate in the embedding space.  This view reveals a shortcoming in contrastive learning: the similarity graph is binary, as only one sample is the related positive sample. Crucially, similarities \textit{across} samples are ignored. Based on this observation, we revise the standard contrastive loss to explicitly encode how a sample relates to others. We experiment with this new objective, called $\mathbb{X}$-Sample Contrastive, to train vision models based on similarities in class or text caption descriptions.Our study spans three scales: ImageNet-1k with 1 million, CC3M with 3 million, and CC12M with 12 million samples. The representations learned via our objective outperform both contrastive self-supervised and vision-language models trained on the same data across a range of tasks. When training on CC12M, we outperform CLIP by $0.6\%$ on both ImageNet and ImageNet Real. Our objective appears to work particularly well in lower-data regimes, with gains over CLIP of $17.2\%$ on ImageNet and $18.0\%$ on ImageNet Real when training with CC3M. Finally, our objective encourages the model to learn representations that separate objects from their attributes and backgrounds, with gains of $3.3$-$5.6$\% over CLIP on ImageNet9. The proposed method takes a step towards developing richer learning objectives for understanding sample relations in foundation models.

</details>

---

## 104. Are Large Vision Language Models Good Game Players?

- [ ] Are Large Vision Language Models Good Game Players? | https://iclr.cc/virtual/2025/poster/29074

- **Link**: https://iclr.cc/virtual/2025/poster/29074

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision Language Models (LVLMs) have demonstrated remarkable abilities in understanding and reasoning about both visual and textual information. However, existing evaluation methods for LVLMs, primarily based on benchmarks like Visual Question Answering and image captioning, often fail to capture the full scope of LVLMs' capabilities. These benchmarks are limited by issues such as inadequate assessment of detailed visual perception, data contamination, and a lack of focus on multi-turn reasoning. To address these challenges, we propose LVLM-Playground, a game-based evaluation framework designed to provide a comprehensive assessment of LVLMs' cognitive and reasoning skills in structured environments. LVLM-Playground uses a set of games to evaluate LVLMs on four core tasks: Perceiving, Question Answering, Rule Following, and End-to-End Playing, with each target task designed to assess specific abilities, including visual perception, reasoning, decision-making, etc. Based on this framework, we conduct extensive experiments that explore the limitations of current LVLMs, such as handling long structured outputs and perceiving detailed and dense elements. Code and data are publicly available at https://github.com/xinke-wang/LVLM-Playground.

</details>

---

## 105. Multimodal Unsupervised Domain Generalization by Retrieving Across the Modality Gap

- [ ] Multimodal Unsupervised Domain Generalization by Retrieving Across the Modality Gap | https://iclr.cc/virtual/2025/poster/29088

- **Link**: https://iclr.cc/virtual/2025/poster/29088

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Domain generalization (DG) is an important problem that learns a model which generalizes to unseen test domains leveraging one or more source domains, under the assumption of shared label spaces. However, most DG methods assume access to abundant source data in the target label space, a requirement that proves overly stringent for numerous real-world applications, where acquiring the same label space as the target task is prohibitively expensive. For this setting, we tackle the multimodal version of the unsupervised domain generalization (MUDG) problem, which uses a large task-agnostic unlabeled source dataset during finetuning. Our framework does not explicitly assume any relationship between the source dataset and target task. Instead, it relies only on the premise that the source dataset can be accurately and efficiently searched in a joint vision-language space. We make three contributions in the MUDG setting. Firstly, we show theoretically that cross-modal approximate nearest neighbor search suffers from low recall due to the large distance between text queries and the image centroids used for coarse quantization. Accordingly, we propose paired k-means, a simple clustering algorithm that improves nearest neighbor recall by storing centroids in query space instead of image space. Secondly, we propose an adaptive text augmentation scheme for target labels designed to improve zero-shot accuracy and diversify retrieved image data. Lastly, we present two simple but effective components to further improve downstream target accuracy. We compare against state-of-the-art name-only transfer, source-free DG and zero-shot (ZS) methods on their respective benchmarks and show consistent improvement in accuracy on 20 diverse datasets. Code is available: https://github.com/Chris210634/mudg

</details>

---

## 106. Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want

- [ ] Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want | https://iclr.cc/virtual/2025/poster/29098

- **Link**: https://iclr.cc/virtual/2025/poster/29098

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present the Draw-and-Understand framework, exploring how to integrate visual prompting understanding capabilities into Multimodal Large Language Models (MLLMs). Visual prompts allow users to interact through multi-modal instructions, enhancing the models' interactivity and fine-grained image comprehension. In this framework, we propose a general architecture adaptable to different pre-trained MLLMs, enabling it to recognize various types of visual prompts (such as points, bounding boxes, and free-form shapes) alongside language understanding. Additionally, we introduce MDVP-Instruct-Data, a multi-domain dataset featuring 1.2 million image-visual prompt-text triplets, including natural images, document images, scene text images, mobile/web screenshots, and remote sensing images. Building on this dataset, we introduce MDVP-Bench, a challenging benchmark designed to evaluate a model's ability to understand visual prompting instructions. The experimental results demonstrate that our framework can be easily and effectively applied to various MLLMs, such as SPHINX-X and LLaVA. After training with MDVP-Instruct-Data and image-level instruction datasets, our models exhibit impressive multimodal interaction capabilities and pixel-level understanding, while maintaining their image-level visual perception performance.

</details>

---

## 107. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs

- [ ] Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs | https://iclr.cc/virtual/2025/poster/29107

- **Link**: https://iclr.cc/virtual/2025/poster/29107

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs).However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in anonymous github page \url{https://github.com/Benchmark-Dysca/Dysca}.

</details>

---

## 108. Adapting Multi-modal Large Language Model to Concept Drift From Pre-training Onwards

- [ ] Adapting Multi-modal Large Language Model to Concept Drift From Pre-training Onwards | https://iclr.cc/virtual/2025/poster/29128

- **Link**: https://iclr.cc/virtual/2025/poster/29128

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) frequently face challenges from concept drift when dealing with real-world streaming data, wherein distributions change unpredictably. This mainly includes gradual drift due to long-tailed data and sudden drift from Out-Of-Distribution (OOD) data, both of which have increasingly drawn the attention of the research community. While these issues have been extensively studied in the individual domain of vision or language, their impacts on MLLMs in concept drift settings remain largely underexplored. In this paper, we reveal the susceptibility and vulnerability of Vision-Language (VL) models to significant biases arising from gradual drift and sudden drift, particularly in the pre-training. To effectively address these challenges, we propose a unified framework that extends concept drift theory to the multi-modal domain, enhancing the adaptability of the VL model to unpredictable distribution changes. Additionally, a T-distribution based drift adapter is proposed to effectively mitigate the bias induced by the gradual drift, which also facilitates the model in distinguishing sudden distribution changes through explicit distribution modeling. Extensive experiments demonstrate our method enhances the efficiency and accuracy of image-text alignment in the pre-training of VL models, particularly in the concept drift scenario. Moreover, various downstream tasks exhibit significant improvements in our model's ability to adapt to the long-tailed open world. Furthermore, we create a set of multi-modal datasets called OpenMMlo, specifically tailored for the long-tailed open-world setting, to validate our findings. To foster the development of the multi-modal community, we have made both OpenMMlo datasets and our code publicly available at: https://github.com/XiaoyuYoung/ConceptDriftMLLMs.

</details>

---

## 109. TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies

- [ ] TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies | https://iclr.cc/virtual/2025/poster/29130

- **Link**: https://iclr.cc/virtual/2025/poster/29130

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although large vision-language-action (VLA) models pretrained on extensive robot datasets offer promising generalist policies for robotic learning, they still struggle with spatial-temporal dynamics in interactive robotics, making them less effective in handling complex tasks, such as manipulation. In this work, we introduce visual trace prompting, a simple yet effective approach to facilitate VLA models’ spatial-temporal awareness for action prediction by encoding state-action trajectories visually. We develop a new TraceVLA model by finetuningOpenVLA on our own collected dataset of 150K robot manipulation trajectories using visual trace prompting. Evaluations of TraceVLA across 137 configurations in SimplerEnv and 4 tasks on a physical WidowX robot demonstrate state-of-the-art performance, outperforming OpenVLA by 10% on SimplerEnv and 3.5x on real-robot tasks and exhibiting robust generalization across diverse embodiments and scenarios. To further validate the effectiveness and generality of our method, we present a compact VLA model based on 4B Phi-3-Vision, pretrained on the Open-X-Embodiment and finetuned on our dataset, rivals the 7B OpenVLA baseline while significantly improving inference efficiency.

</details>

---

## 110. Mitigate the Gap: Improving Cross-Modal Alignment in CLIP

- [ ] Mitigate the Gap: Improving Cross-Modal Alignment in CLIP | https://iclr.cc/virtual/2025/poster/29170

- **Link**: https://iclr.cc/virtual/2025/poster/29170

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language--Image Pre-training (CLIP) has manifested remarkable improvements in zero-shot classification and cross-modal vision-language tasks. Yet, from a geometrical point of view, the CLIP embedding space has been found to have a pronounced modality gap. This gap renders the embedding space overly sparse and disconnected, with different modalities being densely distributed in distinct subregions of the hypersphere. In this work, we propose AlignCLIP, in order to improve the alignment between text and image embeddings, and thereby reduce the modality gap. AlignCLIP increases the cross-modal alignment, and yields gains across several zero-shot and fine-tuning downstream evaluations by sharing the learnable parameters between the modality encoders and a semantically-regularized separation objective function on the uni-modal embeddings. The source code and model checkpoints for reproducing our experiments are available at https://github.com/sarahESL/AlignCLIP.

</details>

---

## 111. Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages

- [ ] Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages | https://iclr.cc/virtual/2025/poster/29188

- **Link**: https://iclr.cc/virtual/2025/poster/29188

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite recent advances in multimodal large language models (MLLMs), their development has predominantly focused on English- and western-centric datasets and tasks, leaving most of the world's languages and diverse cultural contexts underrepresented.  This paper introduces PANGEA, a multilingual multimodal LLM trained on PANGEAINS, a diverse 6M instruction dataset spanning 39 languages. PANGEAINS features: 1) high-quality English instructions, 2) carefully machine-translated instructions, and 3) culturally relevant multimodal tasks to ensure cross-cultural coverage. To rigorously assess models' capabilities, we introduce PANGEABENCH, a holistic evaluation suite encompassing 14 datasets covering 47 languages. Results show that PANGEA significantly outperforms existing open-source models in multilingual settings and diverse cultural contexts. Ablation studies further reveal the importance of English data proportions, language popularity, and the number of multimodal training samples on overall performance.  We fully open-source our data, code, and trained checkpoints, to facilitate the development of inclusive and robust multilingual MLLMs, promoting equity and accessibility across a broader linguistic and cultural spectrum.

</details>

---

## 112. One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning

- [ ] One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning | https://iclr.cc/virtual/2025/poster/29190

- **Link**: https://iclr.cc/virtual/2025/poster/29190

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Anomaly detection methods under the 'one-for-all' paradigm aim to develop a unified model capable of detecting anomalies across multiple classes. However, these approaches typically require a large number of normal samples for model training, which may not always be feasible in practice. Few-shot anomaly detection methods can address scenarios with limited data but often require a tailored model for each class, struggling within the 'one-for-one' paradigm. In this paper, we first proposed the one-for-all few-shot anomaly detection method with the assistance of vision-language model. Different from previous CLIP-based methods learning fix prompts for each class, our method learn a class-shared prompt generator to adaptively generate suitable prompt for each instance. The prompt generator is trained by aligning the prompts with the visual space and utilizing guidance from general textual descriptions of normality and abnormality. Furthermore, we address the mismatch problem of the memory bank within one-for-all paradigm. Extensive experimental results on MVTec and VisA demonstrate the superiority of our method in few-shot anomaly detection task under the one-for-all paradigm.

</details>

---

## 113. TLDR: Token-Level Detective Reward Model for Large Vision Language Models

- [ ] TLDR: Token-Level Detective Reward Model for Large Vision Language Models | https://iclr.cc/virtual/2025/poster/29193

- **Link**: https://iclr.cc/virtual/2025/poster/29193

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although reward models have been successful in improving multimodal large language models, the reward models themselves remain brutal and contain minimal information. Notably, existing reward models only mimic human annotations by assigning only one feedback to any text, no matter how long the text is. In the realm of multimodal language models, where models are required to process both images and texts, a naive reward model may learn implicit biases toward texts and become less grounded in images. In this paper, we propose a T oken- L evel D etective R eward Model ( TLDR ) to provide fine-grained annotations to each text token. We first introduce a perturbation-based method to generate synthetic hard negatives and their token-level labels to train TLDR models. Then we show the rich usefulness of TLDR models both in assisting off-the-shelf models to self-correct their generations, and in serving as a hallucination evaluation tool. We show that TLDR automatically trains a token-level likelihood optimization, and can improve the base model's performance significantly. Finally, we show that TLDR models can significantly speed up human annotation by 3 times to acquire a broader range of high-quality vision language data.

</details>

---

## 114. Needle In A Video Haystack: A Scalable  Synthetic Evaluator for Video MLLMs

- [ ] Needle In A Video Haystack: A Scalable  Synthetic Evaluator for Video MLLMs | https://iclr.cc/virtual/2025/poster/29224

- **Link**: https://iclr.cc/virtual/2025/poster/29224

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video understanding is a crucial next step for multimodal large language models (MLLMs).Various benchmarks are introduced for better evaluating the MLLMs.Nevertheless, current video benchmarks are still inefficient for evaluating video models during iterative development due to the high cost of constructing datasets and the difficulty in isolating specific skills.In this paper, we propose VideoNIAH (Video Needle in A Haystack), a benchmark construction framework through synthetic video generation. VideoNIAH decouples video content from their query-responses by inserting unrelated visual 'needles' into original videos. The framework automates the generation of query-response pairs using predefined rules, minimizing manual labor.  The queries focus on specific aspects of video understanding, enabling more skill-specific evaluations. The separation between video content and the queries also allow for increased video variety and evaluations across different lengths.Utilizing VideoNIAH, we compile a video benchmark, VNBench, which includes tasks such as retrieval, ordering, and counting to evaluate three key aspects of video understanding: temporal perception, chronological ordering, and spatio-temporal coherence. We conduct a comprehensive evaluation of both proprietary and open-source models, uncovering significant differences in their video understanding capabilities across various tasks. Additionally, we perform an in-depth analysis of the test results and model configurations. Based on these findings, we provide some advice for improving video MLLM training, offering valuable insights to guide future research and model development.

</details>

---

## 115. COMBO: Compositional World Models for Embodied Multi-Agent Cooperation

- [ ] COMBO: Compositional World Models for Embodied Multi-Agent Cooperation | https://iclr.cc/virtual/2025/poster/29260

- **Link**: https://iclr.cc/virtual/2025/poster/29260

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we investigate the problem of embodied multi-agent cooperation, where decentralized agents must cooperate given only egocentric views of the world. To effectively plan in this setting, in contrast to learning world dynamics in a single-agent scenario, we must simulate world dynamics conditioned on an arbitrary number of agents' actions given only partial egocentric visual observations of the world. To address this issue of partial observability, we first train generative models to estimate the overall world state given partial egocentric observations. To enable accurate simulation of multiple sets of actions on this world state, we then propose to learn a compositional world model for multi-agent cooperation by factorizing the naturally composable joint actions of multiple agents and compositionally generating the video conditioned on the world state. By leveraging this compositional world model, in combination with Vision Language Models to infer the actions of other agents, we can use a tree search procedure to integrate these modules and facilitate online cooperative planning. We evaluate our methods on three challenging benchmarks with 2-4 agents. The results show our compositional world model is effective and the framework enables the embodied agents to cooperate efficiently with different agents across various tasks and an arbitrary number of agents, showing the promising future of our proposed methods. More videos can be found at \url{https://embodied-agi.cs.umass.edu/combo/}.

</details>

---

## 116. Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders

- [ ] Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders | https://iclr.cc/virtual/2025/poster/29276

- **Link**: https://iclr.cc/virtual/2025/poster/29276

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The ability to accurately interpret complex visual information is a crucial topic of multimodal large language models (MLLMs). Recent work indicates that enhanced visual perception significantly reduces hallucinations and improves performance on resolution-sensitive tasks, such as optical character recognition and document analysis. A number of recent MLLMs achieve this goal using a mixture of vision encoders. Despite their success, there is a lack of systematic comparisons and detailed ablation studies addressing critical aspects, such as expert selection and the integration of multiple vision experts. This study provides an extensive exploration of the design space for MLLMs using a mixture of vision encoders and resolutions. Our findings reveal several underlying principles common to various existing strategies, leading to a streamlined yet effective design approach. We discover that simply concatenating visual tokens from a set of complementary vision encoders is as effective as more complex mixing architectures or strategies. We additionally introduce Pre-Alignment to bridge the gap between vision-focused encoders and language tokens, enhancing model coherence. The resulting family of MLLMs, Eagle, surpasses other leading open-source models on major MLLM benchmarks.

</details>

---

## 117. See It from My Perspective: How Language Affects Cultural Bias in Image Understanding

- [ ] See It from My Perspective: How Language Affects Cultural Bias in Image Understanding | https://iclr.cc/virtual/2025/poster/29299

- **Link**: https://iclr.cc/virtual/2025/poster/29299

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) can respond to queries about images in many languages. However, beyond language, culture affects how we see things.  For example, individuals from Western cultures focus more on the central figure in an image while individuals from East Asian cultures attend more to scene context (Nisbett 2001).  In this work, we characterize the Western bias of VLMs in image understanding and investigate the role that language plays in this disparity. We evaluate VLMs across subjective and objective visual tasks with culturally diverse images and annotations. We find that VLMs perform better on the Western split than on the East Asian split of each task.  Through controlled experimentation, we trace one source of this bias in image understanding to the lack of diversity in language model construction. While inference in a language nearer to a culture can lead to reductions in bias, we show it is much more effective when that language was well-represented during text-only pre-training. Interestingly, this yields bias reductions even when prompting in English. Our work highlights the importance of richer representation of all languages in building equitable VLMs.

</details>

---

## 118. 3D-SPATIAL MULTIMODAL MEMORY

- [ ] 3D-SPATIAL MULTIMODAL MEMORY | https://iclr.cc/virtual/2025/poster/29300

- **Link**: https://iclr.cc/virtual/2025/poster/29300

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present 3D Spatial MultiModal Memory (M3), a multimodal memory system designed to retain information about medium-sized static scenes through video sources for visual perception. By integrating 3D Gaussian Splatting techniques with foundation models, M3 builds a multimodal memory capable of rendering feature representations across granularities, encompassing a wide range of knowledge. In our exploration, we identify two key challenges in previous works on feature splatting: (1) computational constraints in storing high-dimensional features for each Gaussian primitive, and (2) misalignment or information loss between distilled features and foundation model features. To address these challenges, we propose M3 with key components of principal scene components and Gaussian memory attention, enabling efficient training and inference. To validate M3, we conduct comprehensive quantitative evaluations of feature similarity and downstream tasks, as well as qualitative visualizations to highlight the pixel trace of Gaussian memory attention. Our approach encompasses a diverse range of foundation models, including vision-language models (VLMs), perception models, and large multimodal and language models (LMMs/LLMs). Furthermore, to demonstrate real-world applicability, we deploy M3’s feature field in indoor scenes on a quadruped robot. Notably, we claim that M3 is the first work to address the core compression challenges in 3D feature distillation.

</details>

---

## 119. COAT: Compressing Optimizer states and Activations for Memory-Efficient FP8 Training

- [ ] COAT: Compressing Optimizer states and Activations for Memory-Efficient FP8 Training | https://iclr.cc/virtual/2025/poster/29297

- **Link**: https://iclr.cc/virtual/2025/poster/29297

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

FP8 training has emerged as a promising method for improving training efficiency. Existing frameworks accelerate training by applying FP8 computation to linear layers while leaving optimizer states and activations in higher precision, which fails to fully optimize memory usage. This paper introduces COAT ( C ompressing O ptimizer States and A ctivations for FP8 T raining), a novel FP8 training framework designed to significantly reduce memory footprint when training large models. COAT addresses current limitations through two key innovations: (1) Dynamic Range Expansion , which aligns optimizer state distributions more closely with the FP8 representation range, thereby reducing quantization error, and (2) Mixed-Granularity Activation Quantization , which optimizes activation memory using a combination of per-tensor and per-group quantization strategies. Experiments demonstrate that COAT effectively reduces end-to-end training memory footprint by 1.54× compared to BF16 while achieving nearly lossless performance across various tasks, such as Large Language Model pretraining and fine-tuning and Vision Language Model training. COAT also achieves a 1.43× end-to-end training speedup compared to BF16, performing on par with or surpassing TransformerEngine's speedup. COAT enables efficient full-parameter training of large models on fewer GPUs, and facilitates doubling the batch size in distributed training settings, providing a practical solution for scaling large-scale model training. Code will be released upon publication.

</details>

---

## 120. Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping

- [ ] Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping | https://iclr.cc/virtual/2025/poster/29328

- **Link**: https://iclr.cc/virtual/2025/poster/29328

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across multimodal tasks such as visual perception and reasoning, leading to good performance on various multimodal evaluation benchmarks. However, these benchmarks keep a static nature and overlap with the pre-training data, resulting in fixed complexity constraints and data contamination issues. This raises the concern regarding the validity of the evaluation. To address these two challenges, we introduce a dynamic multimodal evaluation protocol called Vision-Language Bootstrapping (VLB). VLB provides a robust and comprehensive assessment for LVLMs with reduced data contamination and flexible complexity. To this end, VLB dynamically generates new visual question-answering samples through a multimodal bootstrapping module that modifies both images and language, while ensuring that newly generated samples remain consistent with the original ones by a judge module. By composing various bootstrapping strategies, VLB offers dynamic variants of existing benchmarks with diverse complexities, enabling the evaluation to co-evolve with the ever-evolving capabilities of LVLMs. Extensive experimental results across multiple benchmarks, including SEEDBench, MMBench, and MME, show that VLB significantly reduces data contamination and exposes performance limitations of LVLMs.

</details>

---

## 121. LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior

- [ ] LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior | https://iclr.cc/virtual/2025/poster/29340

- **Link**: https://iclr.cc/virtual/2025/poster/29340

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present LARP, a novel video tokenizer designed to overcome limitations in current video tokenization methods for autoregressive (AR) generative models. Unlike traditional patchwise tokenizers that directly encode local visual patches into discrete tokens, LARP introduces a holistic tokenization scheme that gathers information from the visual content using a set of learned holistic queries. This design allows LARP to capture more global and semantic representations, rather than being limited to local patch-level information. Furthermore, it offers flexibility by supporting an arbitrary number of discrete tokens, enabling adaptive and efficient tokenization based on the specific requirements of the task. To align the discrete token space with downstream AR generation tasks, LARP integrates a lightweight AR transformer as a training-time prior model that predicts the next token on its discrete latent space. By incorporating the prior model during training, LARP learns a latent space that is not only optimized for video reconstruction but is also structured in a way that is more conducive to autoregressive generation. Moreover, this process defines a sequential order for the discrete tokens, progressively pushing them toward an optimal configuration during training, ensuring smoother and more accurate AR generation at inference time. Comprehensive experiments demonstrate LARPs strong performance, achieving state-of-the-art FVD on the UCF101 class-conditional video generation benchmark. LARP enhances the compatibility of AR models with videos and opens up the potential to build unified high-fidelity multimodal large language models (MLLMs). Project page:  https://hywang66.github.io/larp/

</details>

---

## 122. MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models

- [ ] MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/29339

- **Link**: https://iclr.cc/virtual/2025/poster/29339

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The capability to process multiple images is crucial for Large Vision-Language Models (LVLMs) to develop a more thorough and nuanced understanding of a scene. Recent multi-image LVLMs have begun to address this need. However, their evaluation has not kept pace with their development. To fill this gap, we introduce the Multimodal Multi-image Understanding (MMIU) benchmark, a comprehensive evaluation suite designed to assess LVLMs across a wide range of multi-image tasks. MMIU encompasses 7 types of multi-image relationships, 52 tasks, 77K images, and 11K meticulously curated multiple-choice questions, making it the most extensive benchmark of its kind. Our evaluation of nearly 30 popular LVLMs, including both open-source and proprietary models, reveals significant challenges in multi-image comprehension, particularly in tasks involving spatial understanding. Even the most advanced models, such as GPT-4o, achieve only 55.7\% accuracy on MMIU. Through multi-faceted analytical experiments, we identify key performance gaps and limitations, providing valuable insights for future model and data improvements. We aim for MMIU to advance the frontier of LVLM research and development. We release the data and code at https://github.com/MMIUBenchmark/MMIU.

</details>

---

## 123. Anyprefer: An Agentic Framework for Preference Data Synthesis

- [ ] Anyprefer: An Agentic Framework for Preference Data Synthesis | https://iclr.cc/virtual/2025/poster/29342

- **Link**: https://iclr.cc/virtual/2025/poster/29342

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-quality preference data is essential for aligning foundation models with human values through preference learning. However, manual annotation of such data is often time-consuming and costly. Recent methods often adopt a self-rewarding approach, where the target model generates and annotates its own preference data, but this can lead to inaccuracies since the reward model shares weights with the target model, thereby amplifying inherent biases. To address these issues, we propose Anyprefer, a framework designed to synthesize high-quality preference data for aligning the target model. Anyprefer frames the data synthesis process as a cooperative two-player Markov Game, where the target model and the judge model collaborate together. Here, a series of external tools are introduced to assist the judge model in accurately rewarding the target model’s responses, mitigating biases in the rewarding process. In addition, a feedback mechanism is introduced to optimize prompts for both models, enhancing collaboration and improving data quality. The synthesized data is compiled into a new preference dataset, Anyprefer-V1, consisting of 58K high-quality preference pairs. Extensive experiments show that Anyprefer significantly improves model alignment performance across four main applications, covering 21 datasets, achieving average improvements of 18.55% in five natural language generation datasets, 3.66% in nine vision-language understanding datasets, 30.05% in three medical image analysis datasets, and 16.00% in four visuo-motor control tasks.

</details>

---

## 124. Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model

- [ ] Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model | https://iclr.cc/virtual/2025/poster/29390

- **Link**: https://iclr.cc/virtual/2025/poster/29390

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Mixture-of-Experts (MoE) has gained increasing attention in studying Large Vision-Language Models (LVLMs). It uses a sparse model to replace the dense model, achieving comparable performance while activating fewer parameters during inference, thus significantly reducing the inference cost. Existing MoE methods in LVLM encourage different experts to specialize in different tokens, and they usually employ a router to predict the routing of each token. However, the router is not optimized concerning distinct parameter optimization directions generated from tokens within an expert. This may lead to severe interference between tokens within an expert. To address this problem, we propose to use the token-level gradient analysis to Solving Token Gradient Conflict (STGC) in this paper. Specifically, we first use token-level gradients to identify conflicting tokens in experts. After that, we add a regularization loss tailored to encourage conflicting tokens routing from their current experts to other experts, for reducing interference between tokens within an expert. Our method can serve as a plug-in for diverse LVLM methods, and extensive experimental results demonstrate its effectiveness. demonstrate its effectiveness. The code will be publicly available at https://github.com/longrongyang/STGC.

</details>

---

## 125. Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent

- [ ] Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent | https://iclr.cc/virtual/2025/poster/29392

- **Link**: https://iclr.cc/virtual/2025/poster/29392

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Retrieval Augmented Generation (mRAG) plays an important role in mitigating the “hallucination” issue inherent in multimodal large language models (MLLMs). Although promising, existing heuristic mRAGs typically predefined fixed retrieval processes, which causes two issues: (1) Non-adaptive Retrieval Queries. (2) Overloaded Retrieval Queries. However, these flaws cannot be adequately reflected by current knowledge-seeking visual question answering (VQA) datasets, since the most required knowledge can be readily obtained with a standard two-step retrieval. To bridge the dataset gap, we first construct Dyn-VQA dataset, consisting of three types of ``dynamic'' questions, which require complex knowledge retrieval strategies variable in query, tool, and time: (1) Questions with rapidly changing answers. (2) Questions requiring multi-modal knowledge. (3) Multi-hop questions. Experiments on Dyn-VQA reveal that existing heuristic mRAGs struggle to provide sufficient and precisely relevant knowledge for dynamic questions due to their rigid retrieval processes. Hence, we further propose the first self-adaptive planning agent for multimodal retrieval, OmniSearch . The underlying idea is to emulate the human behavior in question solution which dynamically decomposes complex multimodal questions into sub-question chains with retrieval action. Extensive experiments prove the effectiveness of our OmniSearch, also provide direction for advancing mRAG. Code and dataset will be open-sourced.

</details>

---

## 126. Latent Action Pretraining from Videos

- [ ] Latent Action Pretraining from Videos | https://iclr.cc/virtual/2025/poster/29409

- **Link**: https://iclr.cc/virtual/2025/poster/29409

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce Latent Action Pretraining for general Action models (LAPA), the first unsupervised method for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels. Existing Vision-Language-Action models require action labels typically collected by human teleoperators during pretraining, which significantly limits possible data sources and scale. In this work, we propose a method to learn from internet-scale videos that do not have robot action labels. We first train an action quantization model leveraging VQ-VAE-based objective to learn discrete latent actions between image frames, then pretrain a latent VLA model to predict these latent actions from observations and task descriptions, and finally finetune the VLA on small-scale robot manipulation data to map from latent to robot actions. Experimental results demonstrate that our method significantly outperforms existing techniques that train robot manipulation policies from large-scale videos. Furthermore, it outperforms the state-of-the-art VLA model trained with robotic action labels on real-world manipulation tasks that require language conditioning, generalization to unseen objects, and semantic generalization to unseen instructions. Training only on human manipulation videos also shows positive transfer, opening up the potential for leveraging web-scale data for robotics foundation models.

</details>

---

## 127. Cross the Gap:  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion

- [ ] Cross the Gap:  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion | https://iclr.cc/virtual/2025/poster/29411

- **Link**: https://iclr.cc/virtual/2025/poster/29411

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained multi-modal Vision-Language Models like CLIP are widely used off-the-shelf for a variety of applications. In this paper, we show that the common practice of individually exploiting the text or image encoders of these powerful multi-modal models is highly suboptimal for intra-modal tasks like image-to-image retrieval. We argue that this is inherently due to the CLIP-style inter-modal contrastive loss that does not enforce any intra-modal constraints, leading to what we call intra-modal misalignment. To demonstrate this, we leverage two optimization-based modality inversion techniques that map representations from their input modality to the complementary one without any need for auxiliary data or additional trained adapters. We empirically show that, in the intra-modal tasks of image-to-image and text-to-text retrieval, approaching these tasks inter-modally significantly improves performance with respect to intra-modal baselines on more than fifteen datasets. Additionally, we demonstrate that approaching a native inter-modal task (e.g. zero-shot image classification) intra-modally decreases performance, further validating our findings. Finally, we show that incorporating an intra-modal term in the pre-training objective or narrowing the modality gap between the text and image feature embedding spaces helps reduce the intra-modal misalignment. The code is publicly available at: https://github.com/miccunifi/Cross-the-Gap.

</details>

---

## 128. DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models

- [ ] DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models | https://iclr.cc/virtual/2025/poster/29415

- **Link**: https://iclr.cc/virtual/2025/poster/29415

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancements in Vision-Language Models (VLMs) have shown great potential in tackling mathematical reasoning tasks that involve visual context. Unlike humans who can reliably apply solution steps to similar problems with minor modifications, we found that state-of-the-art VLMs like GPT-4o can consistently fail in these scenarios, revealing limitations in their mathematical reasoning capabilities. In this paper, we investigate the mathematical reasoning robustness in VLMs and evaluate how well these models perform under different variants of the same question, such as changes in visual numerical values or function graphs.While several vision-based math benchmarks have been developed to assess VLMs' problem-solving capabilities, these benchmarks contain only static sets of problems and cannot easily evaluate mathematical reasoning robustness.To fill this gap, we introduce DynaMath , a dynamic visual math benchmark designed for in-depth assessment of VLMs. DynaMath includes 501 high-quality, multi-topic seed questions, each represented as a Python program . Those programs are carefully designed and annotated to enable the automatic generation of a much larger set of concrete questions, including many different types of visual and textual variations. DynaMath allows us to evaluate the generalization ability of VLMs, by assessing their performance under varying input conditions of a seed question. We evaluated 14 state-of-the-art VLMs with 5,010 generated concrete questions (10 per seed question). Our results show that the worst-case model accuracy, defined as the percentage of correctly answered seed questions in all 10 variants, is significantly lower than the average-case accuracy. In addition, many models show high consistency in answering these questions -- the incorrectness of a certain variant of a seed question is not only due to inherent randomness. Our analysis emphasizes the need to study the robustness of VLMs' reasoning abilities, and DynaMath provides valuable insights to guide the development of more reliable models for mathematical reasoning.

</details>

---

## 129. RA-TTA: Retrieval-Augmented Test-Time Adaptation for Vision-Language Models

- [ ] RA-TTA: Retrieval-Augmented Test-Time Adaptation for Vision-Language Models | https://iclr.cc/virtual/2025/poster/29434

- **Link**: https://iclr.cc/virtual/2025/poster/29434

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) are known to be susceptible to distribution shifts between pre-training data and test data, and test-time adaptation (TTA) methods for VLMs have been proposed to mitigate the detrimental impact of the distribution shifts.  However, the existing methods solely rely on the internal knowledge encoded within the model parameters, which are constrained to pre-training data.  To complement the limitation of the internal knowledge, we propose Retrieval-Augmented-TTA (RA-TTA) for adapting VLMs to test distribution using external knowledge obtained from a web-scale image database.  By fully exploiting the bi-modality of VLMs, RA-TTA adaptively retrieves proper external images for each test image to refine VLMs' predictions using the retrieved external images, where fine-grained text descriptions are leveraged to extend the granularity of external knowledge. Extensive experiments on 17 datasets demonstrate that the proposed RA-TTA outperforms the state-of-the-art methods by 3.01-9.63\% on average.

</details>

---

## 130. MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models

- [ ] MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models | https://iclr.cc/virtual/2025/poster/29447

- **Link**: https://iclr.cc/virtual/2025/poster/29447

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing multimodal retrieval benchmarks primarily focus on evaluating whether models can retrieve and utilize external textual knowledge for question answering. However, there are scenarios where retrieving visual information is either more beneficial or easier to access than textual data. In this paper, we introduce a multimodal retrieval-augmented generation benchmark, MRAG-Bench, in which we systematically identify and categorize scenarios where visually augmented knowledge is better than textual knowledge, for instance, more images from varying viewpoints.MRAG-Bench consists of 16,130 images and 1,353 human-annotated multiple-choice questions across 9 distinct scenarios. With MRAG-Bench, we conduct an evaluation of 10 open-source and 4 proprietary large vision-language models (LVLMs). Our results show that all LVLMs exhibit greater improvements when augmented with images compared to textual knowledge, confirming that MRAG-Bench is vision-centric. Additionally, we conduct extensive analysis with MRAG-Bench, which offers valuable insights into retrieval-augmented LVLMs. Notably, the top-performing model, GPT-4o, faces challenges in effectively leveraging retrieved knowledge, achieving only a 5.82\% improvement with ground-truth information, in contrast to a 33.16\% improvement observed in human participants. These findings highlight the importance of MRAG-Bench in encouraging the community to enhance LVLMs' ability to utilize retrieved visual knowledge more effectively.

</details>

---

## 131. Mitigating Spurious Correlations in Zero-Shot Multimodal Models

- [ ] Mitigating Spurious Correlations in Zero-Shot Multimodal Models | https://iclr.cc/virtual/2025/poster/29449

- **Link**: https://iclr.cc/virtual/2025/poster/29449

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal models or Vision Language Models (VLMs) have reshaped the paradigm in machine learning, offering zero-shot capabilities that require no additional training when adapted to new classification tasks. However, despite their advancements, spurious correlations still exist in VLMs. Existing approaches to tackle this issue often require target label annotations, contradicting the principle of zero-shot classification, or they primarily focus on a single modality, risking misalignment between text and image modalities. Others rely on extensive domain knowledge or large language models (LLMs) to characterize spurious features, making the performance sensitive to the generated prompts and undermining zero-shot capability. In response, we propose a new solution that tackles spurious correlations in VLMs within the zero-shot setting. Our approach utilizes a translation operation that preserves the latent space distribution to address issues of spurious correlations. In particular, our method is grounded in and inspired by a theoretical analysis, which identifies that the optimal translation directions are along the spurious vector. As VLMs unify two modalities, we compute spurious vectors from the text prompts and guide the translation for image embeddings, aligning the requirements for the fusion of different modalities in VLMs. We conducted experiments on benchmark datasets, which have shown significant improvements in worst-group accuracy. Additionally, our visualizations of VLMs further demonstrate the effectiveness of this intervention.

</details>

---

## 132. Information Theoretic Text-to-Image Alignment

- [ ] Information Theoretic Text-to-Image Alignment | https://iclr.cc/virtual/2025/poster/29462

- **Link**: https://iclr.cc/virtual/2025/poster/29462

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models for Text-to-Image (T2I) conditional generation have recently achievedtremendous success. Yet, aligning these models with user’s intentions still involves alaborious trial-and-error process, and this challenging alignment problem has attractedconsiderable attention from the research community. In this work, instead of relying onfine-grained linguistic analyses of prompts, human annotation, or auxiliary vision-languagemodels, we use Mutual Information (MI) to guide model alignment. In brief, our methoduses self-supervised fine-tuning and relies on a point-wise MI estimation between promptsand images to create a synthetic fine-tuning set for improving model alignment. Ouranalysis indicates that our method is superior to the state-of-the-art, yet it only requiresthe pre-trained denoising network of the T2I model itself to estimate MI, and a simplefine-tuning strategy that improves alignment while maintaining image quality. Code available at https://github.com/Chao0511/mitune.

</details>

---

## 133. Exploring the Design Space of Visual Context Representation in Video MLLMs

- [ ] Exploring the Design Space of Visual Context Representation in Video MLLMs | https://iclr.cc/virtual/2025/poster/29477

- **Link**: https://iclr.cc/virtual/2025/poster/29477

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video Multimodal Large Language Models~(MLLMs) have shown remarkable capability of understanding the video semantics on various downstream tasks. Despite the advancements, there is still a lack of systematic research on visual context representation, which refers to the scheme to select frames from a video and further select the tokens from a frame. In this paper, we explore the design space for visual context representation, and aim to improve the performance of video MLLMs by finding more effective representation schemes. Firstly, we formulate the task of visual context representation as a constrained optimization problem, and model the language modeling loss as a function of the number of frames and the number of embeddings (or tokens) per frame, given the maximum visual context window size. Then, we explore the scaling effects in frame selection and token selection respectively, and fit the corresponding function curve by conducting extensive empirical experiments. We examine the effectiveness of typical selection strategies and present empirical findings to determine the two factors. Furthermore, we study the joint effect of frame selection and token selection, and derive the optimal formula for determining the two factors. We demonstrate that the derived optimal settings show alignment with the best-performed results of empirical experiments. The data and code are available at: https://github.com/RUCAIBox/Opt-Visor.

</details>

---

## 134. MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding

- [ ] MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding | https://iclr.cc/virtual/2025/poster/29509

- **Link**: https://iclr.cc/virtual/2025/poster/29509

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce MuirBench, a comprehensive benchmark that focuses on robust multi-image understanding capabilities of multimodal LLMs. MuirBench consists of 12 diverse multi-image tasks (e.g., scene understanding, ordering) that involve 10 categories of multi-image relations (e.g., multiview, temporal relations). Comprising 11,264 images and 2,600 multiple-choice questions, MuirBench is created in a pairwise manner, where each standard instance is paired with an unanswerable variant that has minimal semantic differences, in order for a reliable assessment. Evaluated upon 20 recent multi-modal LLMs, our results reveal that even the best-performing models like GPT-4o and Gemini Pro find it challenging to solve MuirBench, achieving 68.0% and 49.3% in accuracy. Open-source multimodal LLMs trained on single images can hardly generalize to multi-image questions, hovering below 33.3% in accuracy. These results highlight the importance of MuirBench in encouraging the community to develop multimodal LLMs that can look beyond a single image, suggesting potential pathways for future improvements.

</details>

---

## 135. Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models

- [ ] Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models | https://iclr.cc/virtual/2025/poster/29536

- **Link**: https://iclr.cc/virtual/2025/poster/29536

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal models highlight the value of rewritten captions for improving performance, yet key challenges remain. For example, while synthetic captions often provide superior quality and image-text alignment, it is not clear whether they can fully replace AltTexts: the role of synthetic captions and their interaction with original web-crawled AltTexts in pre-training is still not well understood. Moreover, different multimodal foundation models may have unique preferences for specific caption formats, but efforts to identify the optimal captions for each model remain limited. In this work, we propose a novel, controllable, and scalable captioning pipeline designed to generate diverse caption formats tailored to various multimodal models. By examining short synthetic captions (SSC) and descriptive synthetic captions (DSC) as case studies, we systematically explore their effects and interactions with AltTexts across models such as CLIP, multimodal LLMs, and diffusion models. Our findings reveal that a hybrid approach that keeps both synthetic captions and AltTexts can outperform the use of synthetic captions alone, improving both alignment and performance, with each model demonstrating preferences for particular caption formats. This comprehensive analysis provides valuable insights into optimizing captioning strategies, thereby advancing the pre-training of multimodal foundation models.

</details>

---

## 136. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification

- [ ] CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification | https://iclr.cc/virtual/2025/poster/29543

- **Link**: https://iclr.cc/virtual/2025/poster/29543

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we aim to build an adversarially robust zero-shot image classifier that can accurately and efficiently classify unseen examples while defending against unforeseen adversarial attacks, addressing critical challenges in real-world safety-sensitive scenarios. To achieve this, we focus on two key challenges: zero-shot classification and defense against unforeseen attacks. We ground our work on CLIP, a vision-language pre-trained model to perform zero-shot classification. To defend against unforeseen attacks, we adopt a purification approach, as it is independent of specific attack types. We then define a purification risk as the KL divergence between the joint distributions of the purification and attack process. The derived lower bound of purification risk inspires us to explore purification in CLIP's multi-modal latent space. We propose a CLIP-based purification method called CLIPure, which has two variants: CLIPure-Diff , which models image likelihood with a generative process of its latent vector, and CLIPure-Cos , which models the likelihood based on the similarity between embeddings of the image and a blank template. As far as we know, CLIPure is the first purification method in latent space and CLIPure-Cos is the first purification method not relying on generative models, substantially improving defense efficiency. Extensive experimental results show that the robustness achieved by CLIPure is within a small gap of clean accuracy, outperforming SOTA robustness by a large margin, e.g., from 71.7\% to 91.1\% on CIFAR10, from 59.6\% to 72.6\% on ImageNet, and 108\% relative improvements of average robustness on the 13 datasets over previous SOTA, with only 14\% extra inference cost and no additional training.

</details>

---

## 137. ClawMachine: Learning to Fetch Visual Tokens for Referential Comprehension

- [ ] ClawMachine: Learning to Fetch Visual Tokens for Referential Comprehension | https://iclr.cc/virtual/2025/poster/29545

- **Link**: https://iclr.cc/virtual/2025/poster/29545

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Aligning vision and language concepts at a finer level remains an essential topic of multimodal large language models (MLLMs), particularly for tasks such as referring and grounding. Existing methods, such as proxy encoding and geometry encoding genres, incorporate additional syntax to encode spatial information, imposing extra burdens when communicating between language with vision modules. In this study, we propose ClawMachine, offering a new methodology that explicitly notates each entity using token collectives —groups of visual tokens that collaboratively represent higher-level semantics. A hybrid perception mechanism is also explored to perceive and understand scenes from both discrete and continuous spaces. Our method unifies the prompt and answer of visual referential tasks without using additional syntax. By leveraging a joint vision-language vocabulary, ClawMachine integrates referring and grounding in an auto-regressive manner, demonstrating great potential with scaled up pre-training data. Experiments show that ClawMachine achieves superior performance on scene-level and referential understanding tasks with higher efficiency. It also exhibits the potential to integrate multi-source information for complex visual reasoning, which is beyond the capability of many MLLMs. Our code is available at https://github.com/martian422/ClawMachine.

</details>

---

## 138. VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks

- [ ] VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks | https://iclr.cc/virtual/2025/poster/29555

- **Link**: https://iclr.cc/virtual/2025/poster/29555

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Embedding models play a crucial role in a variety of downstream tasks, including semantic similarity, information retrieval, and clustering. While there has been a surge of interest in developing universal text embedding models that generalize across tasks (e.g., MTEB), progress in learning universal multimodal embedding models has been comparatively slow, despite their importance and practical applications. In this work, we explore the potential of building universal multimodal embeddings capable of handling a broad range of downstream tasks. Our contributions are twofold: (1) we propose MMEB (Massive Multimodal Embedding Benchmark), which covers four meta-tasks (classification, visual question answering, multimodal retrieval, and visual grounding) and 36 datasets, including 20 training datasets and 16 evaluation datasets spanning both in-distribution and out-of-distribution tasks, and (2) VLM2Vec (Vision-Language Model → Vector), a contrastive training framework that transforms any vision-language model into an embedding model through contrastive training on MMEB. Unlike previous models such as CLIP and BLIP, which encode text and images independently without task-specific guidance, VLM2Vec can process any combination of images and text while incorporating task instructions to generate a fixed-dimensional vector. We develop a series of VLM2Vec models based on state-of-the-art VLMs, including Phi-3.5-V, LLaVA-1.6, and Qwen2-VL, and evaluate them on MMEB’s benchmark. With LoRA tuning, VLM2Vec achieves a 10% to 20% improvement over existing multimodal embedding models on MMEB’s evaluation sets. Our findings reveal that VLMs are surprisingly strong embedding models.

</details>

---

## 139. ZooProbe: A Data Engine for Evaluating, Exploring, and Evolving Large-scale Training Data for Multimodal LLMs

- [ ] ZooProbe: A Data Engine for Evaluating, Exploring, and Evolving Large-scale Training Data for Multimodal LLMs | https://iclr.cc/virtual/2025/poster/29564

- **Link**: https://iclr.cc/virtual/2025/poster/29564

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are thriving through continuous fine-tuning by LLMs. Driven by the law that "scale is everything", MLLMs expand their training sets during version iterations. In this paper, we propose a large-scale training data engine built around an evaluating-exploring-evolving (E3) loop. Evaluating the data provides insights into its characteristics. Exploring quality rules helps identify which data enhances training. Together, these processes facilitate the systematic evolution of new, high-quality data. With the E3 loop, we introduce ZooProbe, an efficient data engine for MLLMs. First, the problem of data expansion is formalized as a tree of sampling and growth. ZooProbe introduces a small-scale model *zoo* to obtain comprehensive evaluations for child datasets. From multiple perspectives, visual, textual, and multimodal models cover over 50 dimensions of intrinsic and meta attributes, such as object and topic distribution, and higher-level properties, like annotation quality and scene complexity. ZooProbe constructs based on A$^\star$ search, modeling the heuristic function as a quality estimate from data evaluation results. It dynamically explores the rule of data quality based on the model state of the *probe* datasets. Additionally, it evolves new targeted data with identified high-quality rules. We also develop an extra heuristic quality ranker with the data utilized and discarded during the expansion. Our experiments show that ZooProbe significantly breaks the scaling law in multimodal instruction fine-tuning at scales of 260$k$ and below.ZooProbe generates high-quality data that accelerates MLLM training and enhances performance, automating the evolution of large-scale training data.

</details>

---

## 140. Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models

- [ ] Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models | https://iclr.cc/virtual/2025/poster/29566

- **Link**: https://iclr.cc/virtual/2025/poster/29566

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Sparse Mixture of Experts (SMoE) has been widely employed to enhance the efficiency of training and inference for Transformer-based foundational models, yielding promising results. However, the performance of SMoE heavily depends on the choice of hyper-parameters, such as the number of experts and the number of experts to be activated (referred to as top-$k$), resulting in significant computational overhead due to the extensive model training by searching over various hyper-parameter configurations. As a remedy, we introduce the Dynamic Mixture of Experts (DynMoE) technique. DynMoE incorporates (1) a novel gating method that enables each token to automatically determine the number of experts to activate. (2) An adaptive process automatically adjusts the number of experts during training. Extensive numerical results across Vision, Language, and Vision-Language tasks demonstrate the effectiveness of our approach to achieve competitive performance compared to GMoE for vision and language tasks, and MoE-LLaVA for vision-language tasks, while maintaining efficiency by activating fewer parameters. Our code is available at \url{https://github.com/LINs-lab/DynMoE}.

</details>

---

## 141. 3D Vision-Language Gaussian Splatting

- [ ] 3D Vision-Language Gaussian Splatting | https://iclr.cc/virtual/2025/poster/29604

- **Link**: https://iclr.cc/virtual/2025/poster/29604

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in 3D reconstruction methods and vision-language models have propelled the development of multi-modal 3D scene understanding, which has vital applications in robotics, autonomous driving, and virtual/augmented reality. However, current multi-modal scene understanding approaches have naively embedded semantic representations into 3D reconstruction methods without striking a balance between visual and language modalities, which leads to unsatisfying semantic rasterization of translucent or reflective objects, as well as over-fitting on color modality. To alleviate these limitations, we propose a solution that adequately handles the distinct visual and semantic modalities, i.e., a 3D vision-language Gaussian splatting model for scene understanding, to put emphasis on the representation learning of language modality. We propose a novel cross-modal rasterizer, using modality fusion along with a smoothed semantic indicator for enhancing semantic rasterization. We also employ a camera-view blending technique to improve semantic consistency between existing and synthesized views, thereby effectively mitigating over-fitting. Extensive experiments demonstrate that our method achieves state-of-the-art performance in open-vocabulary semantic segmentation, surpassing existing methods by a significant margin.

</details>

---

## 142. Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion

- [ ] Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion | https://iclr.cc/virtual/2025/poster/29612

- **Link**: https://iclr.cc/virtual/2025/poster/29612

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In computer vision tasks, features often come from diverse representations, domains (e.g., indoor and outdoor), and modalities (e.g., text, images, and videos). Effectively fusing these features is essential for robust performance, especially with the availability of powerful pre-trained models like vision-language models. However, common fusion methods, such as concatenation, element-wise operations, and non-linear techniques, often fail to capture structural relationships, deep feature interactions, and suffer from inefficiency or misalignment of features across domains or modalities. In this paper, we shift from high-dimensional feature space to a lower-dimensional, interpretable graph space by constructing relationship graphs that encode feature relationships at different levels, e.g., clip, frame, patch, token, etc. To capture deeper interactions, we expand graphs through iterative graph relationship updates and introduce a learnable graph fusion operator to integrate these expanded relationships for more effective fusion. Our approach is relationship-centric, operates in a homogeneous space, and is mathematically principled, resembling element-wise relationship score aggregation via multilinear polynomials. We demonstrate the effectiveness of our graph-based fusion method on video anomaly detection, showing strong performance across multi-representational, multi-modal, and multi-domain feature fusion tasks.

</details>

---

## 143. Reflexive Guidance: Improving OoDD in Vision-Language Models via Self-Guided Image-Adaptive Concept Generation

- [ ] Reflexive Guidance: Improving OoDD in Vision-Language Models via Self-Guided Image-Adaptive Concept Generation | https://iclr.cc/virtual/2025/poster/29663

- **Link**: https://iclr.cc/virtual/2025/poster/29663

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the recent emergence of foundation models trained on internet-scale data and demonstrating remarkable generalization capabilities, such foundation models have become more widely adopted, leading to an expanding range of application domains. Despite this rapid proliferation, the trustworthiness of foundation models remains underexplored. Specifically, the out-of-distribution detection (OoDD) capabilities of large vision-language models (LVLMs), such as GPT-4o, which are trained on massive multi-modal data, have not been sufficiently addressed. The disparity between their demonstrated potential and practical reliability raises concerns regarding the safe and trustworthy deployment of foundation models. To address this gap, we evaluate and analyze the OoDD capabilities of various proprietary and open-source LVLMs. Our investigation contributes to a better understanding of how these foundation models represent confidence scores through their generated natural language responses. Furthermore, we propose a self-guided prompting approach, termed Reflexive Guidance (ReGuide), aimed at enhancing the OoDD capability of LVLMs by leveraging self-generated image-adaptive concept suggestions. Experimental results demonstrate that our ReGuide enhances the performance of current LVLMs in both image classification and OoDD tasks. The lists of sampled images, along with the prompts and responses for each sample are available at https://github.com/daintlab/ReGuide.

</details>

---

## 144. A Simple Framework for Open-Vocabulary Zero-Shot Segmentation

- [ ] A Simple Framework for Open-Vocabulary Zero-Shot Segmentation | https://iclr.cc/virtual/2025/poster/29668

- **Link**: https://iclr.cc/virtual/2025/poster/29668

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot classification capabilities naturally arise in models trained within a vision-language contrastive framework. Despite their classification prowess, these models struggle in dense tasks like zero-shot open-vocabulary segmentation. This deficiency is often attributed to the absence of localization cues in captions and the intertwined nature of the learning process, which encompasses both image/text representation learning and cross-modality alignment. To tackle these issues, we propose SimZSS, a $\textbf{Sim}$ple framework for open-vocabulary $\textbf{Z}$ero-$\textbf{S}$hot $\textbf{S}$egmentation. The method is founded on two key principles: i) leveraging frozen vision-only models that exhibit spatial awareness while exclusively aligning the text encoder and ii) exploiting the discrete nature of text and linguistic knowledge to pinpoint local concepts within captions. By capitalizing on the quality of the visual representations, our method requires only image-caption pair datasets and adapts to both small curated and large-scale noisy datasets. When trained on COCO Captions across 8 GPUs, SimZSS achieves state-of-the-art results on 7 out of 8 benchmark datasets in less than 15 minutes. Our code and pretrained models are publicly available at https://github.com/tileb1/simzss.

</details>

---

## 145. Is Your Multimodal Language Model Oversensitive to Safe Queries?

- [ ] Is Your Multimodal Language Model Oversensitive to Safe Queries? | https://iclr.cc/virtual/2025/poster/29670

- **Link**: https://iclr.cc/virtual/2025/poster/29670

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Humans are prone to cognitive distortions — biased thinking patterns that lead to exaggerated responses to specific stimuli, albeit in very different contexts.This paper demonstrates that advanced Multimodal Large Language Models (MLLMs) exhibit similar tendencies.While these models are designed to respond queries under safety mechanism, they sometimes reject harmless queries in the presence of certain visual stimuli, disregarding the benign nature of their contexts.As the initial step in investigating this behavior, we identify three representative types of stimuli that trigger the oversensitivity of existing MLLMs: $\textbf{\textit{Exaggerated Risk}}$, $\textbf{\textit{Negated Harm}}$, and $\textbf{\textit{Counterintuitive Interpretation}}$.To systematically evaluate MLLMs' oversensitivity to these stimuli, we propose the $\textbf{M}$ultimodal $\textbf{O}$ver$\textbf{S}$en$\textbf{S}$itivity $\textbf{Bench}$mark (MOSSBench).This toolkit consists of 300 manually collected benign multimodal queries, cross-verified by third-party reviewers (AMT).Empirical studies using MOSSBench on 20 MLLMs reveal several insights:(1). Oversensitivity is prevalent among SOTA MLLMs, with refusal rates reaching up to $\textbf{76}$\% for harmless queries.(2). Safer models are more oversensitive: increasing safety may inadvertently raise caution and conservatism in the model’s responses.(3). Different types of stimuli tend to cause errors at specific stages — perception, intent reasoning, and safety judgement — in the response process of MLLMs.These findings highlight the need for refined safety mechanisms that balance caution with contextually appropriate responses, improving the reliability of MLLMs in real-world applications.

</details>

---

## 146. ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time

- [ ] ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time | https://iclr.cc/virtual/2025/poster/29674

- **Link**: https://iclr.cc/virtual/2025/poster/29674

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have become essential backbones for multi-modal intelligence, yet significant safety challenges limit their real-world application. While textual inputs can often be effectively safeguarded, adversarial visual inputs can often easily bypass VLM defense mechanisms. Existing defense methods are either resource-intensive, requiring substantial data and compute, or fail to simultaneously ensure safety and usefulness in responses. To address these limitations, we propose a novel two-phase inference-time alignment framework, **E**valuating **T**hen **A**ligning (ETA): i) Evaluating input visual contents and output responses to establish a robust safety awareness in multimodal settings, and ii) Aligning unsafe behaviors at both shallow and deep levels by conditioning the VLMs' generative distribution with an interference prefix and performing sentence-level best-of-$N$ to search the most harmless and helpful generation paths. Extensive experiments show that ETA outperforms baseline methods in terms of harmlessness, helpfulness, and efficiency, reducing the unsafe rate by 87.5\% in cross-modality attacks and achieving 96.6\% win-ties in GPT-4 helpfulness evaluation.

</details>

---

## 147. VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning

- [ ] VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning | https://iclr.cc/virtual/2025/poster/29691

- **Link**: https://iclr.cc/virtual/2025/poster/29691

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Broadly intelligent agents should form task-specific abstractions that selectively expose the essential elements of a task, while abstracting away the complexity of the raw sensorimotor space. In this work, we present Neuro-Symbolic Predicates, a first-order abstraction language that combines the strengths of symbolic and neural knowledge representations. We outline an online algorithm for inventing such predicates and learning abstract world models. We compare our approach to hierarchical reinforcement learning, vision-language model planning, and symbolic predicate invention approaches, on both in- and out-of-distribution tasks across five simulated robotic domains. Results show that our approach offers better sample complexity, stronger out-of-distribution generalization, and improved interpretability.

</details>

---

## 148. SegLLM: Multi-round Reasoning Segmentation with Large Language Models

- [ ] SegLLM: Multi-round Reasoning Segmentation with Large Language Models | https://iclr.cc/virtual/2025/poster/29732

- **Link**: https://iclr.cc/virtual/2025/poster/29732

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present SegLLM, a novel multi-round interactive reasoning segmentation model that enhances LLM-based segmentation by exploiting conversational memory of both visual and textual outputs. By leveraging a mask-aware multimodal LLM, SegLLM re-integrates previous segmentation results into its input stream, enabling it to reason about complex user intentions and segment objects in relation to previously identified entities, including positional, interactional, and hierarchical relationships, across multiple interactions. This capability allows SegLLM to respond to visual and text queries in a chat-like manner. Evaluated on the newly curated MRSeg benchmark, SegLLM outperforms existing methods in multi- round interactive reasoning segmentation by over 20%. Additionally, we observed that training on multi-round reasoning segmentation data enhances performance on standard single-round referring segmentation and localization tasks, resulting in a 5.5% increase in cIoU for referring expression segmentation and a 4.5% improvement in Acc@0.5 for referring expression localization.

</details>

---

## 149. Should VLMs be Pre-trained with Image Data?

- [ ] Should VLMs be Pre-trained with Image Data? | https://iclr.cc/virtual/2025/poster/29734

- **Link**: https://iclr.cc/virtual/2025/poster/29734

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained LLMs that are further trained with image data perform well on vision-language tasks. While adding images during a second training phase effectively unlocks this capability, it is unclear how much of a gain or loss this two-step pipeline gives over VLMs which integrate images earlier into the training process. To investigate this, we train models spanning various datasets, scales, image-text ratios, and amount of pre-training done before introducing vision tokens.We then fine-tune these models and evaluate their downstream performance on a suite of vision-language and text-only tasks.We find that pre-training with a mixture of image and text data allows models to perform better on vision-language tasks while maintaining strong performance on text-only evaluations.On an average of 6 diverse tasks, we find that for a 1B model, introducing visual tokens 80\% of the way through pre-training results in a 2\% average improvement over introducing visual tokens to a fully pre-trained model.

</details>

---

## 150. LLM-wrapper: Black-Box Semantic-Aware Adaptation of Vision-Language Models for Referring Expression Comprehension

- [ ] LLM-wrapper: Black-Box Semantic-Aware Adaptation of Vision-Language Models for Referring Expression Comprehension | https://iclr.cc/virtual/2025/poster/29739

- **Link**: https://iclr.cc/virtual/2025/poster/29739

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have demonstrated remarkable capabilities in various open-vocabulary tasks, yet their zero-shot performance lags behind task-specific fine-tuned models, particularly in complex tasks like Referring Expression Comprehension (REC). Fine-tuning usually requires ‘white-box’ access to the model’s architecture and weights, which is not always feasible due to proprietary or privacy concerns. In this work, we propose LLM-wrapper, a method for ‘black-box’ adaptation of VLMs for the REC task using Large Language Models (LLMs). LLM-wrapper capitalizes on the reasoning abilities of LLMs, improved with a light fine-tuning, to select the most relevant bounding box to match the referring expression, from candidates generated by a zero-shot black-box VLM. Our approach offers several advantages: it enables the adaptation of closed-source models without needing access to their internal workings, it is versatile and works with any VLM, transfers to new VLMs, and it allows for the adaptation of an ensemble of VLMs. We evaluate LLM-wrapper on multiple datasets using different VLMs and LLMs, demonstrating significant performance improvements and highlighting the versatility of our method. While LLM-wrapper is not meant to directly compete with standard white-box fine-tuning, it offers a practical and effective alternative for black-box VLM adaptation. The code will be open-sourced.

</details>

---

## 151. Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models

- [ ] Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models | https://iclr.cc/virtual/2025/poster/29769

- **Link**: https://iclr.cc/virtual/2025/poster/29769

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the broader context of deep learning, Multimodal Large Language Models have achieved significant breakthroughs by leveraging powerful Large Language Models as a backbone to align different modalities into the language space. A prime exemplification is the development of Video Large Language Models (Video-LLMs). While numerous advancements have been proposed to enhance the video understanding capabilities of these models, they are predominantly trained on questions generated directly from video content. However, in real-world scenarios, users often pose questions that extend beyond the informational scope of the video, highlighting the need for Video-LLMs to assess the relevance of the question. We demonstrate that even the best-performing Video-LLMs fail to reject unfit questions-not necessarily due to a lack of video understanding, but because they have not been trained to identify and refuse such questions. To address this limitation, we propose alignment for answerability, a framework that equips Video-LLMs with the ability to evaluate the relevance of a question based on the input video and appropriately decline to answer when the question exceeds the scope of the video, as well as an evaluation framework with a comprehensive set of metrics designed to measure model behavior before and after alignment. Furthermore, we present a pipeline for creating a dataset specifically tailored for alignment for answerability, leveraging existing video-description paired datasets.

</details>

---

## 152. Understanding Long Videos with Multimodal Language Models

- [ ] Understanding Long Videos with Multimodal Language Models | https://iclr.cc/virtual/2025/poster/29788

- **Link**: https://iclr.cc/virtual/2025/poster/29788

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have allowed recent LLM-based approaches to achieve excellent performance on long-video understanding benchmarks. We investigate how extensive world knowledge and strong reasoning skills of underlying LLMs influence this strong performance. Surprisingly, we discover that LLM-based approaches can yield surprisingly good accuracy on long-video tasks with limited video information, sometimes even with no video-specific information. Building on this, we explore injecting video-specific information into an LLM-based framework. We utilize off-the-shelf vision tools to extract three object-centric information modalities from videos, and then leverage natural language as a medium for fusing this information. Our resulting Multimodal Video Understanding (MVU) framework demonstrates state-of-the-art performance across multiple video understanding benchmarks. Strong performance also on robotics domain tasks establishes its strong generality. Code: github.com/kahnchana/mvu

</details>

---

## 153. ADAM: An Embodied Causal Agent in Open-World Environments

- [ ] ADAM: An Embodied Causal Agent in Open-World Environments | https://iclr.cc/virtual/2025/poster/29794

- **Link**: https://iclr.cc/virtual/2025/poster/29794

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In open-world environments like Minecraft, existing agents face challenges in continuously learning structured knowledge, particularly causality. These challenges stem from the opacity inherent in black-box models and an excessive reliance on prior knowledge during training, which impair their interpretability and generalization capability. To this end, we introduce ADAM, An emboDied causal Agent in Minecraft, which can autonomously navigate the open world, perceive multimodal context, learn causal world knowledge, and tackle complex tasks through lifelong learning. ADAM is empowered by four key components: 1) an interaction module, enabling the agent to execute actions while recording the interaction processes; 2) a causal model module, tasked with constructing an ever-growing causal graph from scratch, which enhances interpretability and reduces reliance on prior knowledge; 3) a controller module, comprising a planner, an actor, and a memory pool, using the learned causal graph to accomplish tasks; 4) a perception module, powered by multimodal large language models, enabling ADAM to perceive like a human player. Extensive experiments show that ADAM constructs a nearly perfect causal graph from scratch, enabling efficient task decomposition and execution with strong interpretability. Notably, in the modified Minecraft game where no prior knowledge is available, ADAM excels with remarkable robustness and generalization capability. ADAM pioneers a novel paradigm that integrates causal methods and embodied agents synergistically. Our project page is at https://opencausalab.github.io/ADAM.

</details>

---

## 154. SPA-BENCH: A COMPREHENSIVE BENCHMARK FOR SMARTPHONE AGENT EVALUATION

- [ ] SPA-BENCH: A COMPREHENSIVE BENCHMARK FOR SMARTPHONE AGENT EVALUATION | https://iclr.cc/virtual/2025/poster/29820

- **Link**: https://iclr.cc/virtual/2025/poster/29820

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Smartphone agents are increasingly important for helping users control devices efficiently, with (Multimodal) Large Language Model (MLLM)-based approaches emerging as key contenders. Fairly comparing these agents is essential but challenging, requiring a varied task scope, the integration of agents with different implementations, and a generalisable evaluation pipeline to assess their strengths and weaknesses. In this paper, we present SPA-Bench, a comprehensive SmartPhone Agent Benchmark designed to evaluate (M)LLM-based agents in an interactive environment that simulates real-world conditions. SPA-Bench offers three key contributions: (1) A diverse set of tasks covering system and third-party apps in both English and Chinese, focusing on features commonly used in daily routines; (2) A plug-and-play framework enabling real-time agent interaction with Android devices, integrating over ten agents with the flexibility to add more; (3) A novel evaluation pipeline that automatically assesses agent performance across multiple dimensions, encompassing seven metrics related to task completion and resource consumption. Our extensive experiments across tasks and agents reveal challenges like interpreting mobile user interfaces, action grounding, memory retention, and execution costs. We propose future research directions to ease these difficulties, moving closer to real-world smartphone agent applications.

</details>

---

## 155. Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary Resolution

- [ ] Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary Resolution | https://iclr.cc/virtual/2025/poster/29834

- **Link**: https://iclr.cc/virtual/2025/poster/29834

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual data comes in various forms, ranging from small icons of just a few pixels to long videos spanning hours. Existing multi-modal LLMs usually standardize these diverse visual inputs to fixed-resolution images or patches for visual encoders and yield similar numbers of tokens for LLMs. This approach is non-optimal for multimodal understanding and inefficient for processing inputs with long and short visual contents. To solve the problem, we propose Oryx, a unified multimodal architecture for the spatial-temporal understanding of images, videos, and multi-view 3D scenes. Oryx offers an on-demand solution to seamlessly and efficiently process visual inputs with arbitrary spatial sizes and temporal lengths through two core innovations: 1) a pre-trained OryxViT model that can encode images at any resolution into LLM-friendly visual representations; 2) a dynamic compressor module that supports 1x to 16x compression on visual tokens by request. These designs enable Oryx to accommodate extremely long visual contexts, such as videos, with lower resolution and high compression while maintaining high recognition precision for tasks like document understanding with native resolution and no compression. Beyond the architectural improvements, enhanced data curation and specialized training on long-context retrieval and spatial-aware data help Oryx achieve strong capabilities in image, video, and 3D multimodal understanding simultaneously.

</details>

---

## 156. Weighted Multi-Prompt Learning with Description-free Large Language Model Distillation

- [ ] Weighted Multi-Prompt Learning with Description-free Large Language Model Distillation | https://iclr.cc/virtual/2025/poster/29892

- **Link**: https://iclr.cc/virtual/2025/poster/29892

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in pre-trained Vision Language Models (VLM) have shown promising potential for effectively adapting to downstream tasks through prompt learning , without the need for additional annotated paired datasets.To supplement the text information in VLM trained on correlations with vision data, new approaches leveraging Large Language Models (LLM) in prompts have been proposed, enhancing robustness to unseen and diverse data.Existing methods typically extract text-based responses (i.e., descriptions ) from LLM to incorporate into prompts; however, this approach suffers from high variability and low reliability.In this work, we propose De scription-free Mul ti-prompt Learning( DeMul ), a novel method that eliminates the process of extracting descriptions and instead directly distills knowledge from LLM into prompts.By adopting a description-free approach, prompts can encapsulate richer semantics while still being represented as continuous vectors for optimization, thereby eliminating the need for discrete pre-defined templates.Additionally, in a multi-prompt setting, we empirically demonstrate the potential of prompt weighting in reflecting the importance of different prompts during training.Experimental results show that our approach achieves superior performance across 11 recognition datasets.

</details>

---

## 157. Discovering Clone Negatives via Adaptive Contrastive Learning for Image-Text Matching

- [ ] Discovering Clone Negatives via Adaptive Contrastive Learning for Image-Text Matching | https://iclr.cc/virtual/2025/poster/29908

- **Link**: https://iclr.cc/virtual/2025/poster/29908

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we identify a common yet challenging issue in image-text matching, i.e., clone negatives: negative image-text pairs that semantically resemble positive pairs, leading to ambiguous and sub-optimal matching outcomes. To tackle this issue, we propose Adaptive Contrastive Learning (AdaCL), which introduces two margin parameters along with a modulating anchor to dynamically strengthen the compactness between positives and mitigate the influence of clone negatives. The modulating anchor is selected based on the distribution of negative samples without the need for explicit training, allowing for progressive tuning and advanced in-batch supervision. Extensive experiments across several tasks demonstrate the effectiveness of AdaCL in image-text matching. Furthermore, we extend AdaCL to weakly-supervised image-text matching by replacing human-annotated descriptions with automatically generated captions, thereby increasing the number of potential clone negatives. AdaCL maintains robustness in this setting, alleviating the reliance on crowd-sourced annotations and laying a foundation for scalable vision-language contrastive learning.

</details>

---

## 158. MAVIS: Mathematical Visual Instruction Tuning with an Automatic Data Engine

- [ ] MAVIS: Mathematical Visual Instruction Tuning with an Automatic Data Engine | https://iclr.cc/virtual/2025/poster/29918

- **Link**: https://iclr.cc/virtual/2025/poster/29918

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) have recently showcased superior proficiency in general visual scenarios. However, we identify their mathematical capabilities remain under-explored with three areas to be improved: visual encoding of math diagrams, diagram-language alignment, and chain-of-thought (CoT) reasoning. This draws forth an urgent demand for an effective training paradigm and a large-scale, comprehensive dataset with detailed CoT rationales, which is challenging to collect and costly to annotate manually. To tackle this issue, we propose MAVIS, a MAthematical VISual instruction tuning pipeline for MLLMs, featuring an automatic data engine to efficiently create mathematical visual datasets.We design the data generation process to be entirely independent of human intervention or GPT API usage, while ensuring the diagram-caption correspondence, question-answer correctness, and CoT reasoning quality. With this approach, we curate two datasets, MAVIS-Caption (558K diagram-caption pairs) and MAVIS-Instruct (834K visual math problems with CoT rationales), and propose four progressive stages for training MLLMs from scratch.First, we utilize MAVIS-Caption to fine-tune a math-specific vision encoder (CLIP-Math) through contrastive learning, tailored for improved diagram visual encoding. Second, we also leverage MAVIS-Caption to align the CLIP-Math with a large language model (LLM) by a projection layer, enhancing vision-language alignment in mathematical domains. Third, we adopt MAVIS-Instruct to perform the instruction tuning for robust problem-solving skills, and term the resulting model as MAVIS-7B. Fourth, we apply Direct Preference Optimization (DPO) to enhance the CoT capabilities of our model, further refining its step-wise reasoning performance.On various mathematical benchmarks, our MAVIS-7B achieves leading results among open-source MLLMs, e.g., surpassing other 7B models by +9.3% and the second-best LLaVA-NeXT (110B) by +6.9%, demonstrating the effectiveness of our method.

</details>

---

## 159. LICORICE: Label-Efficient Concept-Based Interpretable Reinforcement Learning

- [ ] LICORICE: Label-Efficient Concept-Based Interpretable Reinforcement Learning | https://iclr.cc/virtual/2025/poster/29920

- **Link**: https://iclr.cc/virtual/2025/poster/29920

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in reinforcement learning (RL) have predominantly leveraged neural network policies for decision-making, yet these models often lack interpretability, posing challenges for stakeholder comprehension and trust. Concept bottleneck models offer an interpretable alternative by integrating human-understandable concepts into policies. However, prior work assumes that concept annotations are readily available during training. For RL, this requirement poses a significant limitation: it necessitates continuous real-time concept annotation, which either places an impractical burden on human annotators or incurs substantial costs in API queries and inference time when employing automated labeling methods. To overcome this limitation, we introduce a novel training scheme that enables RL agents to efficiently learn a concept-based policy by only querying annotators to label a small set of data. Our algorithm, LICORICE, involves three main contributions: interleaving concept learning and RL training, using an ensemble to actively select informative data points for labeling, and decorrelating the concept data. We show how LICORICE reduces human labeling efforts to 500 or fewer concept labels in three environments, and 5000 or fewer in two more complex environments, all at no cost to performance. We also explore the use of VLMs as automated concept annotators, finding them effective in some cases but imperfect in others. Our work significantly reduces the annotation burden for interpretable RL, making it more practical for real-world applications that necessitate transparency. Our code is released.

</details>

---

## 160. Re-Aligning Language to Visual Objects with an Agentic Workflow

- [ ] Re-Aligning Language to Visual Objects with an Agentic Workflow | https://iclr.cc/virtual/2025/poster/29934

- **Link**: https://iclr.cc/virtual/2025/poster/29934

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language-based object detection (LOD) aims to align visual objects with language expressions. A large amount of paired data is utilized to improve LOD model generalizations. During the training process, recent studies leverage vision-language models (VLMs) to automatically generate human-like expressions for visual objects, facilitating training data scaling up. In this process, we observe that VLM hallucinations bring inaccurate object descriptions (e.g., object name, color, and shape) to deteriorate VL alignment quality. To reduce VLM hallucinations, we propose an agentic workflow controlled by an LLM to re-align language to visual objects via adaptively adjusting image and text prompts. We name this workflow Real-LOD, which includes planning, tool use, and reflection steps. Given an image with detected objects and VLM raw language expressions, Real-LOD reasons its state automatically and arranges action based on our neural symbolic designs (i.e., planning). The action will adaptively adjust the image and text prompts and send them to VLMs for object re-description (i.e., tool use). Then, we use another LLM to analyze these refined expressions for feedback (i.e., reflection). These steps are conducted in a cyclic form to gradually improve language descriptions for re-aligning to visual objects. We construct a dataset that contains a tiny amount of 0.18M images with re-aligned language expression and train a prevalent LOD model to surpass existing LOD methods by around 50% on the standard benchmarks. Our Real-LOD workflow, with automatic VL refinement, reveals a potential to preserve data quality along with scaling up data quantity, which further improves LOD performance from a data-alignment perspective.

</details>

---

## 161. Do Egocentric Video-Language Models Truly Understand Hand-Object Interactions?

- [ ] Do Egocentric Video-Language Models Truly Understand Hand-Object Interactions? | https://iclr.cc/virtual/2025/poster/29951

- **Link**: https://iclr.cc/virtual/2025/poster/29951

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Egocentric video-language pretraining is a crucial step in advancing the understanding of hand-object interactions in first-person scenarios. Despite successes on existing testbeds, we find that current EgoVLMs can be easily misled by simple modifications, such as changing the verbs or nouns in interaction descriptions, with models struggling to distinguish between these changes. This raises the question: "Do EgoVLMs truly understand hand-object interactions?'' To address this question, we introduce a benchmark called $\textbf{EgoHOIBench}$, revealing the performance limitation of current egocentric models when confronted with such challenges. We attribute this performance gap to insufficient fine-grained supervision and the greater difficulty EgoVLMs experience in recognizing verbs compared to nouns. To tackle these issues, we propose a novel asymmetric contrastive objective named $\textbf{EgoNCE++}$. For the video-to-text objective, we enhance text supervision by generating negative captions using large language models or leveraging pretrained vocabulary for HOI-related word substitutions. For the text-to-video objective, we focus on preserving an object-centric feature space that clusters video representations based on shared nouns. Extensive experiments demonstrate that EgoNCE++ significantly enhances EgoHOI understanding, leading to improved performance across various EgoVLMs in tasks such as multi-instance retrieval, action recognition, and temporal understanding. Our code is available at https://github.com/xuboshen/EgoNCEpp.

</details>

---

## 162. Chain-of-region: Visual Language Models Need  Details for Diagram Analysis

- [ ] Chain-of-region: Visual Language Models Need  Details for Diagram Analysis | https://iclr.cc/virtual/2025/poster/29954

- **Link**: https://iclr.cc/virtual/2025/poster/29954

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Language Models (VLMs) like GPT-4V have broadened the scope of LLM applications, yet they face significant challenges in accurately processing visual details, particularly in scientific diagrams. This paper explores the necessity of meticulous visual detail collection and region decomposition for enhancing the performance of VLMs in scientific diagram analysis. We propose a novel approach that combines traditional computer vision techniques with VLMs to systematically decompose diagrams into discernible visual elements and aggregate essential metadata. Our method employs techniques in OpenCV library to identify and label regions, followed by a refinement process using shape detection and region merging algorithms, which are particularly suited to the structured nature of scientific diagrams. This strategy not only improves the granularity and accuracy of visual information processing but also extends the capabilities of VLMs beyond their current limitations. We validate our approach through a series of experiments that demonstrate enhanced performance in diagram analysis tasks, setting a new standard for integrating visual and language processing in a multimodal context.

</details>

---

## 163. DistRL: An Asynchronous Distributed Reinforcement Learning Framework for On-Device Control Agent

- [ ] DistRL: An Asynchronous Distributed Reinforcement Learning Framework for On-Device Control Agent | https://iclr.cc/virtual/2025/poster/30000

- **Link**: https://iclr.cc/virtual/2025/poster/30000

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

On-device control agents, especially on mobile devices, are responsible for operating mobile devices to fulfill users' requests, enabling seamless and intuitive interactions. Integrating Multimodal Large Language Models (MLLMs) into these agents enhances their ability to understand and execute complex commands, thereby improving user experience. However, fine-tuning MLLMs for on-device control presents significant challenges due to limited data availability and inefficient online training processes. This paper introduces DistRL, a novel framework designed to enhance the efficiency of online RL fine-tuning for mobile device control agents. DistRL employs centralized training and decentralized data acquisition to ensure efficient fine-tuning in the context of dynamic online interactions. Additionally, the framework is backed by our tailor-made RL algorithm, which effectively balances exploration with the prioritized utilization of collected data to ensure stable and robust training. Our experiments show that, on average, DistRL delivers a 3$\times$ improvement in training efficiency and enables training data collection 2.4$\times$ faster than the leading synchronous multi-machine methods. Notably, after training, DistRL achieves a 20\% relative improvement in success rate compared to state-of-the-art methods on general Android tasks from an open benchmark, significantly outperforming existing approaches while maintaining the same training time. These results validate DistRL as a scalable and efficient solution, offering substantial improvements in both training efficiency and agent performance for real-world, in-the-wild device control tasks.

</details>

---

## 164. Can LLMs Understand Time Series Anomalies?

- [ ] Can LLMs Understand Time Series Anomalies? | https://iclr.cc/virtual/2025/poster/30008

- **Link**: https://iclr.cc/virtual/2025/poster/30008

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have gained popularity in time series forecasting, but their potential for anomaly detection remains largely unexplored. Our study investigates whether LLMs can understand and detect anomalies in time series data, focusing on zero-shot and few-shot scenarios. Inspired by conjectures about LLMs' behavior from time series forecasting research, we formulate key hypotheses about LLMs' capabilities in time series anomaly detection. We design and conduct principled experiments to test each of these hypotheses. Our investigation reveals several surprising findings about LLMs for time series: (1) LLMs understand time series better as images rather than as text, (2) LLMs do not demonstrate enhanced performance when prompted to engage in explicit reasoning about time series analysis. (3) Contrary to common beliefs, LLMs' understanding of time series do not stem from their repetition biases or arithmetic abilities. (4) LLMs' behaviors and performance in time series analysis vary significantly across different models. This study provides the first comprehensive analysis of contemporary LLM capabilities in time series anomaly detection. Our results suggest that while LLMs can understand trivial time series anomalies (we have no evidence that they can understand more subtle real-world anomalies), many common conjectures based on their reasoning capabilities do not hold. All synthetic dataset generators, final prompts, and evaluation scripts have been made available in https://github.com/rose-stl-lab/anomllm.

</details>

---

## 165. Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering

- [ ] Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering | https://iclr.cc/virtual/2025/poster/30013

- **Link**: https://iclr.cc/virtual/2025/poster/30013

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucination poses a challenge to the deployment of large vision-language models (LVLMs) in applications. Unlike in large language models (LLMs), hallucination in LVLMs often arises from misalignments between visual inputs and textual outputs. This paper investigates the underlying mechanisms of hallucination, focusing on the unique structure of LVLMs that distinguishes them from LLMs. We identify that hallucinations often arise from the sensitivity of text decoders to vision inputs, a natural phenomenon when image encoders and text decoders are pre-trained separately. Inspired by this, we introduce Visual and Textual Intervention (VTI), a novel technique designed to reduce hallucinations by steering latent space representations during inference to enhance the stability of vision features. As a task-agnostic test-time intervention, VTI can be easily applied to any problem without additional training costs. Extensive experiments demonstrate that it can effectively reduce hallucinations and outperform baseline methods across multiple metrics, highlighting the critical role of vision feature stability in LVLMs.

</details>

---

## 166. Sketch2Diagram: Generating Vector Diagrams from Hand-Drawn Sketches

- [ ] Sketch2Diagram: Generating Vector Diagrams from Hand-Drawn Sketches | https://iclr.cc/virtual/2025/poster/30027

- **Link**: https://iclr.cc/virtual/2025/poster/30027

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We address the challenge of automatically generating high-quality vector diagrams from hand-drawn sketches. Vector diagrams are essential for communicating complex ideas across various fields, offering flexibility and scalability. While recent research has progressed in generating diagrams from text descriptions, converting hand-drawn sketches into vector diagrams remains largely unexplored due to the lack of suitable datasets. To address this gap, we introduce SketikZ, a dataset comprising 3,231 pairs of hand-drawn sketches and thier corresponding TikZ codes as well as reference diagrams.Our evaluations reveal the limitations of state-of-the-art vision and language models (VLMs), positioning SketikZ as a key benchmark for future research in sketch-to-diagram conversion.Along with SketikZ, we present ImgTikZ, an image-to-TikZ model that integrates a 6.7B parameter code-specialized open-source large language model (LLM) with a pre-trained vision encoder. Despite its relatively compact size, ImgTikZ performs comparably to GPT-4o.This success is driven by using our two data augmentation techniques and a multi-candidate inference strategy.Our findings open promising directions for future research in sketch-to-diagram conversion and broader image-to-code generation tasks. SketikZ is publicly available.

</details>

---

## 167. Ranking-aware adapter for text-driven image ordering with CLIP

- [ ] Ranking-aware adapter for text-driven image ordering with CLIP | https://iclr.cc/virtual/2025/poster/30043

- **Link**: https://iclr.cc/virtual/2025/poster/30043

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language models (VLMs) have made significant progress in downstream tasks that require quantitative concepts such as facial age estimation and image quality assessment, enabling VLMs to explore applications like image ranking and retrieval. However, existing studies typically focus on the reasoning based on a single image and heavily depend on text prompting, limiting their ability to learn comprehensive understanding from multiple images. To address this, we propose an effective yet efficient approach that reframes the CLIP model into a learning-to-rank task and introduces a lightweight adapter to augment CLIP for text-guided image ranking. Specifically, our approach incorporates learnable prompts to adapt to new instructions for ranking purposes and an auxiliary branch with ranking-aware attention, leveraging text-conditioned visual differences for additional supervision in image ranking. Our ranking-aware adapter consistently outperforms fine-tuned CLIPs on various tasks and achieves competitive results compared to state-of-the-art models designed for specific tasks like facial age estimation and image quality assessment. Overall, our approach primarily focuses on ranking images with a single instruction, which provides a natural and generalized way of learning from visual differences across images, bypassing the need for extensive text prompts tailored to individual tasks.

</details>

---

## 168. YOLO-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary

- [ ] YOLO-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary | https://iclr.cc/virtual/2025/poster/30052

- **Link**: https://iclr.cc/virtual/2025/poster/30052

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Identifying and localizing objects within images is a fundamental challenge, and numerous efforts have been made to enhance model accuracy by experimenting with diverse architectures and refining training strategies. Nevertheless, a prevalent limitation in existing models is overemphasizing the current input while ignoring the information from the entire dataset. We introduce an innovative $\textbf{R}etriever-\textbf{D}ictionary$ (RD) module to address this issue. This architecture enables YOLO-based models to efficiently retrieve features from a Dictionary that contains the insight of the dataset, which is built by the knowledge from  Visual Models (VM), Large Language Models (LLM), or Visual Language Models (VLM). The flexible RD enables the model to incorporate such explicit knowledge that enhances the ability to benefit multiple tasks, specifically, segmentation, detection, and classification, from pixel to image level. The experiments show that using the RD significantly improves model performance, achieving more than a 3\% increase in mean Average Precision for object detection with less than a 1\% increase in model parameters. Beyond 1-stage object detection models, the RD module improves the effectiveness of 2-stage models and DETR-based architectures, such as Faster R-CNN and Deformable DETR. Code is released at https://github.com/henrytsui000/YOLO.

</details>

---

## 169. Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images

- [ ] Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images | https://iclr.cc/virtual/2025/poster/30074

- **Link**: https://iclr.cc/virtual/2025/poster/30074

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have demonstrated remarkable image understanding and dialogue capabilities, allowing them to handle a variety of visual question answering tasks. However, their widespread availability raises concerns about unauthorized usage and copyright infringement, where users or individuals can develop their own LVLMs by fine-tuning published models. In this paper, we propose a novel method called Parameter Learning Attack (PLA) for tracking the copyright of LVLMs without modifying the original model. Specifically, we construct adversarial images through targeted attacks against the original model, enabling it to generate specific outputs. To ensure these attacks remain effective on potential fine-tuned models to trigger copyright tracking, we allow the original model to learn the trigger images by updating parameters in the opposite direction during the adversarial attack process. Notably, the proposed method can be applied after the release of the original model, thus not affecting the model’s performance and behavior. To simulate real-world applications, we fine-tune the original model using various strategies across diverse datasets, creating a range of models for copyright verification. Extensive experiments demonstrate that our method can more effectively identify the original copyright of fine-tuned models compared to baseline methods. Therefore, this work provides a powerful tool for tracking copyrights and detecting unlicensed usage of LVLMs.

</details>

---

## 170. VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation

- [ ] VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation | https://iclr.cc/virtual/2025/poster/30076

- **Link**: https://iclr.cc/virtual/2025/poster/30076

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language-action models (VLAs) have recently become highly prevalent in robot manipulation due to its end-to-end architecture and impressive performance. However, current VLAs are limited to processing human instructions in textual form, neglecting the more natural speech modality for human interaction. A typical approach of incorporating speech modality into VLA necessitates a separate speech recognition system to transcribe spoken instructions into text. Such a cascading pipeline raises two major concerns for robotic systems. First, the entire model grows in size and complexity, potentially resulting in redundant computations and increased memory consumption. Second, the transcription procedure would lose non-semantic information in the raw speech, such as voiceprint, which is crucial for a robot to successfully understand and complete customized tasks. To this end, we propose VLAS, the fisrt end-to-end policy model that seamlessly integrates speech modality for robot manipulation. We present a three-stage speech instruction tuning strategy leveraging multimodal datasets, including our manually curated SQA and CSI datasets. Furthermore, to facilitate personalized operations, we develop a voice retrieval-augmented generation (RAG) approach to enhance the robot's performance in tasks requiring individual-specific knowledge. Experimental results show that the proposed VLAS, following either textual or speech instructions, can achieve performance comparable to traditional VLAs on the CALVIN benchmark. In addition, we created a benchmark consisting of customization tasks, where our VLAS demonstrates absolute superiority by fully leveraging the auxiliary information in speech.

</details>

---

## 171. MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents

- [ ] MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents | https://iclr.cc/virtual/2025/poster/30075

- **Link**: https://iclr.cc/virtual/2025/poster/30075

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

MLLM agents demonstrate potential for complex embodied tasks by retrieving multimodal task-relevant trajectory data. However, current retrieval methods primarily focus on surface-level similarities of textual or visual cues in trajectories, neglecting their effectiveness for the specific task at hand. To address this issue, we propose a novel method, MART, which enhances the performance of embodied agents by utilizing interaction data to fine-tune an MLLM retriever based on preference learning, such that the retriever fully considers the effectiveness of trajectories and prioritize them for unseen tasks. We also introduce Trajectory Abstraction, a mechanism that leverages MLLMs' summarization capabilities to represent trajectories with fewer tokens while preserving key information, enabling agents to better comprehend milestones in the trajectory. Experimental results across various environments demonstrate our method significantly improves task success rates in unseen scenes compared to baseline methods. This work presents a new paradigm for multimodal retrieval in embodied agents, by fine-tuning a general-purpose MLLM as the retriever to assess trajectory effectiveness. All the code for benchmark tasks, simulator modifications and the MLLM retriever is available at https://github.com/PKU-RL/MART.

</details>

---

## 172. DAMO: Decoding by Accumulating Activations Momentum for Mitigating Hallucinations in Vision-Language Models

- [ ] DAMO: Decoding by Accumulating Activations Momentum for Mitigating Hallucinations in Vision-Language Models | https://iclr.cc/virtual/2025/poster/30108

- **Link**: https://iclr.cc/virtual/2025/poster/30108

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (VLMs) exhibit significant potential in multimodal tasks but often struggle with hallucinations—responses that are plausible yet visually ungrounded. In this work, we investigate the layer-wise prediction tendencies of VLMs and conduct an in-depth analysis of their decoding mechanism. We observe that VLMs tend to ``overthink'' during the final stages of decoding, making significant prediction shifts in the last few layers often favoring incorrect results, which leads to a surge in hallucinative outputs. Leveraging this localized pattern, we propose a novel decoding strategy inspired by the momentum analogy used in gradient descent-based optimizers. Our method enforces decoding consistency across layers in an adaptive manner during forward passes—an under-explored approach in existing works. This strategy significantly improves the reliability and performance of VLMs in various multimodal tasks, while introducing only negligible efficiency overhead.

</details>

---

## 173. AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation

- [ ] AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation | https://iclr.cc/virtual/2025/poster/30106

- **Link**: https://iclr.cc/virtual/2025/poster/30106

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Robotic manipulation in open-world settings requires not only task execution but also the ability to detect and learn from failures. While recent advances in vision-language models (VLMs) and large language models (LLMs) have improved robots' spatial reasoning and problem-solving abilities, they still struggle with failure recognition, limiting their real-world applicability. We introduce AHA, an open-source VLM designed to detect and reason about failures in robotic manipulation using natural language. By framing failure detection as a free-form reasoning task, AHA identifies failures and provides detailed, adaptable explanations across different robots, tasks, and environments. We fine-tuned AHA using FailGen, a scalable framework that generates the first large-scale dataset of robotic failure trajectories, the AHA dataset. FailGen achieves this by procedurally perturbing successful demonstrations from simulation. Despite being trained solely on the AHA dataset, AHA generalizes effectively to real-world failure datasets, robotic systems, and unseen tasks. It surpasses the second-best model (GPT-4o in-context learning) by 10.3% and exceeds the average performance of six compared models including five state-of-the-art VLMs by 35.3% across multiple metrics and datasets. We integrate AHA into three manipulation frameworks that utilize LLMs/VLMs for reinforcement learning, task and motion planning, and zero-shot trajectory generation. AHA’s failure feedback enhances these policies' performances by refining dense reward functions, optimizing task planning, and improving sub-task verification, boosting task success rates by an average of 21.4% across all three tasks compared to GPT-4 models. Project page: https://aha-vlm.github.io

</details>

---

## 174. Cyclic Contrastive Knowledge Transfer for Open-Vocabulary Object Detection

- [ ] Cyclic Contrastive Knowledge Transfer for Open-Vocabulary Object Detection | https://iclr.cc/virtual/2025/poster/30109

- **Link**: https://iclr.cc/virtual/2025/poster/30109

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In pursuit of detecting unstinted objects that extend beyond predefined categories, prior arts of open-vocabulary object detection (OVD) typically resort to pretrained vision-language models (VLMs) for base-to-novel category generalization. However, to mitigate the misalignment between upstream image-text pretraining and downstream region-level perception, additional supervisions are indispensable, e.g., image-text pairs or pseudo annotations generated via self-training strategies. In this work, we propose CCKT-Det trained without any extra supervision. The proposed framework constructs a cyclic and dynamic knowledge transfer from language queries and visual region features extracted from VLMs, which forces the detector to closely align with the visual-semantic space of VLMs. Specifically, 1) we prefilter and inject semantic priors to guide the learning of queries, and 2) introduce a regional contrastive loss to improve the awareness of queries on novel objects. CCKT-Det can consistently improve performance as the scale of VLMs increases, all while requiring the detector at a moderate level of computation overhead. Comprehensive experimental results demonstrate that our method achieves performance gain of +2.9% and +10.2% AP_{50} over previous state-of-the-arts on the challenging COCO benchmark, both without and with a stronger teacher model.

</details>

---

## 175. MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection

- [ ] MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection | https://iclr.cc/virtual/2025/poster/30120

- **Link**: https://iclr.cc/virtual/2025/poster/30120

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the field of industrial inspection, Multimodal Large Language Models (MLLMs) have a high potential to renew the paradigms in practical applications due to their robust language capabilities and generalization abilities. However, despite their impressive problem-solving skills in many domains, MLLMs' ability in industrial anomaly detection has not been systematically studied. To bridge this gap, we present MMAD, a full-spectrum MLLM benchmark in industrial Anomaly Detection. We defined seven key subtasks of MLLMs in industrial inspection and designed a novel pipeline to generate the MMAD dataset with 39,672 questions for 8,366 industrial images. With MMAD, we have conducted a comprehensive, quantitative evaluation of various state-of-the-art MLLMs. The commercial models performed the best, with the average accuracy of GPT-4o models reaching 74.9\%. However, this result falls far short of industrial requirements. Our analysis reveals that current MLLMs still have significant room for improvement in answering questions related to industrial anomalies and defects. We further explore two training-free performance enhancement strategies to help models improve in industrial scenarios, highlighting their promising potential for future research. The code and data are available at https://github.com/jam-cc/MMAD.

</details>

---

## 176. RetroInText: A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning

- [ ] RetroInText: A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning | https://iclr.cc/virtual/2025/poster/30129

- **Link**: https://iclr.cc/virtual/2025/poster/30129

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Development of robust and effective strategies for retrosynthetic planning requires a deep understanding of the synthesis process. A critical step in achieving this goal is accurately identifying synthetic intermediates. Current machine learning-based methods often overlook the valuable context from the overall route, focusing only on predicting reactants from the product, requiring cost annotations for every reaction step, and ignoring the multi-faced nature of molecular, resulting in inaccurate synthetic route predictions. Therefore, we introduce RetroInText, an advanced end-to-end framework based on a multimodal Large Language Model (LLM), featuring in-context learning with TEXT descriptions of synthetic routes. First, RetroInText including ChatGPT presents detailed descriptions of the reaction procedure. It learns the distinct compound representations in parallel with corresponding molecule encoders to extract multi-modal representations including 3D features. Subsequently, we propose an attention-based mechanism that offers a fusion module to complement these multi-modal representations with in-context learning and a fine-tuned language model for a single-step model. As a result, RetroInText accurately represents and effectively captures the complex relationship between molecules and the synthetic route. In experiments on the USPTO pathways dataset RetroBench, RetroInText outperforms state-of-the-art methods, achieving up to a 5% improvement in Top-1 test accuracy, particularly for long synthetic routes. These results demonstrate the superiority of RetroInText by integrating with context information over routes. They also demonstrate its potential for advancing pathway design and facilitating the development of organic chemistry. Code is available at https://github.com/guofei-tju/RetroInText.

</details>

---

## 177. What Makes a Maze Look Like a Maze?

- [ ] What Makes a Maze Look Like a Maze? | https://iclr.cc/virtual/2025/poster/30138

- **Link**: https://iclr.cc/virtual/2025/poster/30138

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

A unique aspect of human visual understanding is the ability to flexibly interpret abstract concepts: acquiring lifted rules explaining what they symbolize, grounding them across familiar and unfamiliar contexts, and making predictions or reasoning about them. While off-the-shelf vision-language models excel at making literal interpretations of images (e.g., recognizing object categories such as tree branches), they still struggle to make sense of such visual abstractions (e.g., how an arrangement of tree branches may form the walls of a maze). To address this challenge, we introduce Deep Schema Grounding (DSG), a framework that leverages explicit structured representations of visual abstractions for grounding and reasoning. At the core of DSG are schemas—dependency graph descriptions of abstract concepts that decompose them into more primitive-level symbols. DSG uses large language models to extract schemas, then hierarchically grounds concrete to abstract components of the schema onto images with vision-language models. The grounded schema is used to augment visual abstraction understanding. We systematically evaluate DSG and different methods in reasoning on our new Visual Abstractions Benchmark, which consists of diverse, real-world images of abstract concepts and corresponding question-answer pairs labeled by humans. We show that DSG significantly improves the abstract visual reasoning performance of vision-language models, and is a step toward human-aligned understanding of visual abstractions.

</details>

---

## 178. MedTrinity-25M: A Large-scale Multimodal Dataset with Multigranular Annotations for Medicine

- [ ] MedTrinity-25M: A Large-scale Multimodal Dataset with Multigranular Annotations for Medicine | https://iclr.cc/virtual/2025/poster/30141

- **Link**: https://iclr.cc/virtual/2025/poster/30141

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces MedTrinity-25M, a comprehensive, large-scale multimodal dataset for medicine, covering over 25 million images across 10 modalities with multigranular annotations for more than 65 diseases. These multigranular annotations encompass both global information, such as modality and organ detection, and local information like ROI analysis, lesion texture, and region-wise correlations. Unlike the existing multimodal datasets, which are limited by the availability of image-text pairs, we have developed the first automated pipeline that scales up multimodal data by generating multigranular visual and textual annotations in the form of image-ROI-description triplets without the need for any paired text descriptions. Specifically, data from over 30 different sources have been collected, preprocessed, and grounded using domain-specific expert models to identify ROIs related to abnormal regions. We then build a comprehensive knowledge base and prompt multimodal large language models to perform retrieval-augmented generation with the identified ROIs as guidance, resulting in multigranular textual descriptions. Compared to existing datasets, MedTrinity-25M provides the most enriched annotations, supporting a comprehensive range of multimodal tasks such as captioning and report generation, as well as vision-centric tasks like classification and segmentation. We propose LLaVA-Tri by pretraining LLaVA on MedTrinity-25M, achieving state-of-the-art performance on VQA-RAD, SLAKE, and PathVQA, surpassing representative SOTA multimodal large language models. Furthermore, MedTrinity-25M can also be utilized to support large-scale pre-training of multimodal medical AI models, contributing to the development of future foundation models in the medical domain. We will make our dataset available. The dataset is publicly available at https://yunfeixie233.github.io/MedTrinity-25M/.

</details>

---

## 179. Harnessing Webpage UIs for Text-Rich Visual Understanding

- [ ] Harnessing Webpage UIs for Text-Rich Visual Understanding | https://iclr.cc/virtual/2025/poster/30176

- **Link**: https://iclr.cc/virtual/2025/poster/30176

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-rich visual understanding—the ability to interpret both textual content and visual elements within a scene—is crucial for multimodal large language models (MLLMs) to effectively interact with structured environments. We propose leveraging webpage UIs as a naturally structured and diverse data source to enhance MLLMs’ capabilities in this area. Existing approaches, such as rule-based extraction, multimodal model captioning, and rigid HTML parsing, are hindered by issues like noise, hallucinations, and limited generalization. To overcome these challenges, we introduce MultiUI, a dataset of 7.3 million samples spanning various UI types and tasks, structured using enhanced accessibility trees and task taxonomies. By scaling multimodal instructions from web UIs through LLMs, our dataset enhances generalization beyond web domains, significantly improving performance in document understanding, GUI comprehension, grounding, and advanced agent tasks. This demonstrates the potential of structured web data to elevate MLLMs’ proficiency in processing text-rich visual environments and generalizing across domains.

</details>

---

## 180. Multimodal Situational Safety

- [ ] Multimodal Situational Safety | https://iclr.cc/virtual/2025/poster/30187

- **Link**: https://iclr.cc/virtual/2025/poster/30187

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are rapidly evolving, demonstrating impressive capabilities as multimodal assistants that interact with both humans and their environments. However, this increased sophistication introduces significant safety concerns. In this paper, we present the first evaluation and analysis of a novel safety challenge termed Multimodal Situational Safety, which explores how safety considerations vary based on the specific situation in which the user or agent is engaged. We argue that for an MLLM to respond safely—whether through language or action—it often needs to assess the safety implications of a language query within its corresponding visual context.To evaluate this capability, we develop the Multimodal Situational Safety benchmark (MSSBench) to assess the situational safety performance of current MLLMs. The dataset comprises 1,960 language query-image pairs, half of which the image context is safe, and the other half is unsafe. We also develop an evaluation framework that analyzes key safety aspects, including explicit safety reasoning, visual understanding, and, crucially, situational safety reasoning. Our findings reveal that current MLLMs struggle with this nuanced safety problem in the instruction-following setting and struggle to tackle these situational safety challenges all at once, highlighting a key area for future research. Furthermore, we develop multi-agent pipelines to coordinately solve safety challenges, which shows consistent improvement in safety over the original MLLM response.

</details>

---

## 181. SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding

- [ ] SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding | https://iclr.cc/virtual/2025/poster/30198

- **Link**: https://iclr.cc/virtual/2025/poster/30198

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the significant advancements of Large Vision-Language Models (LVLMs) on established benchmarks, there remains a notable gap in suitable evaluation regarding their applicability in the emerging domain of long-context streaming video understanding. Current benchmarks for video understanding typically emphasize isolated single-instance text inputs and fail to evaluate the capacity to sustain temporal reasoning throughout the entire duration of video streams. To address these limitations, we introduce SVBench, a pioneering benchmark with temporal multi-turn question-answering chains specifically designed to thoroughly assess the capabilities of streaming video understanding of current LVLMs. We design a semi-automated annotation pipeline to obtain 49,979 Question-Answer (QA) pairs of 1,353 streaming videos, which includes generating QA chains that represent a series of consecutive multi-turn dialogues over video segments and constructing temporal linkages between successive QA chains. Our experimental results, obtained from 14 models in dialogue and streaming evaluations, reveal that while the closed-source GPT-4o outperforms others, most open-source LVLMs struggle with long-context streaming video understanding. We also construct a StreamingChat model, which significantly outperforms open-source LVLMs on our SVBench and achieves comparable performance on diverse vision-language benchmarks. We expect SVBench to advance the research of streaming video understanding by providing a comprehensive and in-depth analysis of current LVLMs. Our benchmark and model can be accessed at https://yzy-bupt.github.io/SVBench.

</details>

---

## 182. MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models

- [ ] MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/30211

- **Link**: https://iclr.cc/virtual/2025/poster/30211

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Interleaved multimodal comprehension and generation, enabling models to produce and interpret both images and text in arbitrary sequences, have become a pivotal area in multimodal learning. Despite significant advancements, the evaluation of this capability remains insufficient. Existing benchmarks suffer from limitations in data scale, scope, and evaluation depth, while current evaluation metrics are often costly or biased, lacking in reliability for practical applications. To address these challenges, we introduce MMIE, a large-scale knowledge-intensive benchmark for evaluating interleaved multimodal comprehension and generation in Large Vision-Language Models (LVLMs). MMIE comprises 20K meticulously curated multimodal queries, spanning 3 categories, 12 fields, and 102 subfields, including mathematics, coding, physics, literature, health, and arts. It supports both interleaved inputs and outputs, offering a mix of multiple-choice and open-ended question formats to evaluate diverse competencies. Moreover, we propose a reliable automated evaluation metric, leveraging a scoring model fine-tuned with human-annotated data and systematic evaluation criteria, aimed at reducing bias and improving evaluation accuracy. Extensive experiments demonstrate the effectiveness of our benchmark and metrics in providing a comprehensive evaluation of interleaved LVLMs. Specifically, we evaluate eight LVLMs, revealing that even the best models show significant room for improvement, with most achieving only moderate results. We believe MMIE will drive further advancements in the development of interleaved LVLMs.

</details>

---

## 183. RandLoRA: Full rank parameter-efficient fine-tuning of large models

- [ ] RandLoRA: Full rank parameter-efficient fine-tuning of large models | https://iclr.cc/virtual/2025/poster/30212

- **Link**: https://iclr.cc/virtual/2025/poster/30212

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Low-Rank Adaptation (LoRA) and its variants have shown impressive results in reducing the number of trainable parameters and memory requirements of large transformer networks while maintaining fine-tuning performance. The low-rank nature of the weight update inherently limits the representation power of fine-tuned models, however, thus potentially compromising performance on complex tasks. This raises a critical question: when a performance gap between LoRA and standard fine-tuning is observed, is it due to the reduced number of train-able parameters or the rank deficiency? This paper aims to answer this question by introducing RandLoRA, a parameter-efficient method that performs full-rank updates using a learned linear combinations of low-rank, non-trainable random matrices. Our method limits the number of trainable parameters by restricting optimization to diagonal scaling matrices applied to the fixed random matrices. This allows us to effectively overcome the low-rank limitations while maintaining parameter and memory efficiency during training. Through extensive experimen-tation across vision, language, and vision-language benchmarks, we systematically evaluate the limitations of LoRA and existing random basis methods. Our findings reveal that full-rank updates are beneficial across vision and language tasks individually, and even more so for vision-language tasks, where RandLoRA significantly reduces—and sometimes eliminates—the performance gap between standard fine-tuning and LoRA, demonstrating its efficacy.

</details>

---

## 184. MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning

- [ ] MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning | https://iclr.cc/virtual/2025/poster/30222

- **Link**: https://iclr.cc/virtual/2025/poster/30222

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present MM1.5, a new family of multimodal large language models (MLLMs) designed to enhance capabilities in text-rich image understanding, visual referring and grounding, and multi-image reasoning. Building upon the MM1 architecture, MM1.5 adopts a data-centric approach to model training, systematically exploring the impact of diverse data mixtures across the entire model training lifecycle. This includes high-quality OCR data and synthetic captions for continual pre-training, as well as an optimized visual instruction-tuning data mixture for supervised fine-tuning. Our models range from 1B to 30B parameters, encompassing both dense and mixture-of-experts (MoE) variants, and demonstrate that careful data curation and training strategies can yield strong performance even at small scales (1B and 3B). Additionally, we introduce two specialized variants: MM1.5-Video, designed for video understanding, and MM1.5-UI, tailored for mobile UI understanding. Through extensive empirical studies and ablations, we provide detailed insights into the training processes and decisions that inform our final designs, offering valuable guidance for future research in MLLM development.

</details>

---

## 185. World Model on Million-Length Video And Language With Blockwise RingAttention

- [ ] World Model on Million-Length Video And Language With Blockwise RingAttention | https://iclr.cc/virtual/2025/poster/30229

- **Link**: https://iclr.cc/virtual/2025/poster/30229

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Enabling long-context understanding remains a key challenge in scaling existing sequence models -- a crucial component in developing generally intelligent models that can process and operate over long temporal horizons that potentially consist of millions of tokens. In this paper, we aim to address these challenges by providing a comprehensive exploration of the full development process for producing 1M context language models and video-language models, setting new benchmarks in language retrieval and new capabilities in long video understanding. We detail our long context data curation process, progressive context extension from 4K to 1M tokens, and present an efficient open-source implementation for scalable training on long sequences. Additionally, we open-source a family of 7B parameter models capable of processing long text documents and videos exceeding 1M tokens.

</details>

---

## 186. VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration

- [ ] VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration | https://iclr.cc/virtual/2025/poster/30231

- **Link**: https://iclr.cc/virtual/2025/poster/30231

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have demonstrated impressive performance across a versatile set of tasks. A key challenge in accelerating VLMs is storing and accessing the large Key-Value (KV) cache that encodes long visual contexts, such as images or videos. While existing KV cache compression methods are effective for Large Language Models (LLMs), directly migrating them to VLMs yields suboptimal accuracy and speedup. To bridge the gap, we propose VL-Cache, a novel KV cache compression recipe tailored for accelerating VLM inference. In this paper, we first investigate the unique sparsity pattern of VLM attention by distinguishing visual and text tokens in prefill and decoding phases. Based on these observations, we introduce a layer-adaptive sparsity-aware cache budget allocation method that effectively distributes the limited cache budget across different layers, further reducing KV cache size without compromising accuracy. Additionally, we develop a modality-aware token scoring policy to better evaluate the token importance. Empirical results on multiple benchmark datasets demonstrate that retaining only 10% of KV cache achieves accuracy comparable to that with full cache. In a speed benchmark, our method accelerates end-to-end latency of generating 100 tokens by up to 2.33x and speeds up decoding by up to 7.08x, while reducing the memory footprint of KV cache in GPU by 90%.

</details>

---

## 187. MediConfusion: Can you trust your AI radiologist? Probing the reliability of multimodal medical foundation models

- [ ] MediConfusion: Can you trust your AI radiologist? Probing the reliability of multimodal medical foundation models | https://iclr.cc/virtual/2025/poster/30242

- **Link**: https://iclr.cc/virtual/2025/poster/30242

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have tremendous potential to improve the accuracy, availability, and cost-effectiveness of healthcare by providing automated solutions or serving as aids to medical professionals. Despite promising first steps in developing medical MLLMs in the past few years, their capabilities and limitations are not well understood. Recently, many benchmark datasets have been proposed that test the general medical knowledge of such models across a variety of medical areas. However, the systematic failure modes and vulnerabilities of such models are severely underexplored with most medical benchmarks failing to expose the shortcomings of existing models in this safety-critical domain. In this paper, we introduce MediConfusion, a challenging medical Visual Question Answering (VQA) benchmark dataset, that probes the failure modes of medical MLLMs from a vision perspective. We reveal that state-of-the-art models are easily confused by image pairs that are otherwise visually dissimilar and clearly distinct for medical experts. Strikingly, all available models (open-source or proprietary) achieve performance below random guessing on MediConfusion, raising serious concerns about the reliability of existing medical MLLMs for healthcare deployment. We also extract common patterns of model failure that may help the design of a new generation of more trustworthy and reliable MLLMs in healthcare.

</details>

---

## 188. Unhackable Temporal Reward for Scalable Video MLLMs

- [ ] Unhackable Temporal Reward for Scalable Video MLLMs | https://iclr.cc/virtual/2025/poster/30267

- **Link**: https://iclr.cc/virtual/2025/poster/30267

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the pursuit of superior video-processing MLLMs, we have encountered a perplexing paradox: the “anti-scaling law”, where more data and larger models lead to worse performance. This study unmasks the culprit: “temporal hacking”, a phenomenon where models shortcut by fixating on select frames, missing the full video narrative. In this work, we systematically establish a comprehensive theory of temporal hacking, defining it from a reinforcement learning perspective, introducing the Temporal Perplexity (TPL) score to assess this misalignment, and proposing the Unhackable Temporal Rewarding (UTR) framework to mitigate the temporal hacking. Both theoretically and empirically, TPL proves to be a reliable indicator of temporal modeling quality, correlating strongly with frame activation patterns. Extensive experiments reveal that UTR not only counters temporal hacking but significantly elevates video comprehension capabilities. This work not only advances video-AI systems but also illuminates the critical importance of aligning proxy rewards with true objectives in MLLM development.

</details>

---

## 189. Bridging Compressed Image Latents and Multimodal Large Language Models

- [ ] Bridging Compressed Image Latents and Multimodal Large Language Models | https://iclr.cc/virtual/2025/poster/30277

- **Link**: https://iclr.cc/virtual/2025/poster/30277

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents the first-ever study of adapting compressed image latents to suit the needs of downstream vision tasks that adopt Multimodal Large Language Models (MLLMs). MLLMs have extended the success of large language models to modalities (e.g. images) beyond text, but their billion scale hinders deployment on resource-constrained end devices. While cloud-hosted MLLMs could be available, transmitting raw, uncompressed images captured by end devices to the cloud requires an efficient image compression system. To address this, we focus on emerging neural image compression and propose a novel framework with a lightweight transform-neck and a surrogate loss to adapt compressed image latents for MLLM-based vision tasks. Given the huge scale of MLLMs, our framework excludes the entire downstream MLLM except part of its visual encoder from training our system. This stands out from most existing coding for machine approaches that involve downstream networks in training and thus could be impractical when the networks are MLLMs. The proposed framework is general in that it is applicable to various MLLMs, neural image codecs, and multiple application scenarios, where the neural image codec can be (1) pre-trained for human perception without updating, (2) fully updated for joint human and machine perception, or (3) fully updated for only machine perception. Extensive experiments on different neural image codecs and various MLLMs show that our method achieves great rate-accuracy performance with much less complexity.

</details>

---

## 190. MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science

- [ ] MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science | https://iclr.cc/virtual/2025/poster/30279

- **Link**: https://iclr.cc/virtual/2025/poster/30279

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained on extensive text and image corpora, current Multi-Modal Large Language Models (MLLM) have shown strong capabilities in general visual reasoning tasks. However, their performance is still lacking in physical domains that require understanding diagrams with complex physical structures and quantitative analysis based on multi-modal information. To address this, we develop a new framework, named M ulti-Modal Scientific Re A soning with P hysics Perception and S imulation ( MAPS ) based on an MLLM. MAPS decomposes expert-level multi-modal reasoning task into physical diagram understanding via a Physical Perception Model (PPM) and reasoning with physical knowledge via a simulator. The PPM module is obtained by fine-tuning a visual language model using carefully designed synthetic data with paired physical diagrams and corresponding simulation language descriptions. At the inference stage, MAPS integrates the simulation language description of the input diagram provided by PPM and results obtained through a Chain-of-Simulation process with MLLM to derive the underlying rationale and the final answer. Validated using our collected college-level circuit analysis problems, MAPS significantly improves reasoning accuracy of MLLM and outperforms all existing models. The results confirm MAPS offers a promising direction for enhancing multi-modal scientific reasoning ability of MLLMs. We will release our code, model and dataset used for our experiments upon publishing of this paper.

</details>

---

## 191. ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning

- [ ] ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning | https://iclr.cc/virtual/2025/poster/30295

- **Link**: https://iclr.cc/virtual/2025/poster/30295

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autonomous agents have demonstrated significant potential in automating complex multistep decision-making tasks. However, even state-of-the-art vision-language models (VLMs), such as GPT-4o, still fall short of human-level performance, particularly in intricate web environments and long-horizon planning tasks. To address these limitations, we introduce Reflective Monte Carlo Tree Search (R-MCTS), a novel test-time algorithm designed to enhance the ability of AI agents, e.g., powered by GPT-4o, to explore decision space on the fly.R-MCTS extends traditional MCTS by 1) incorporating contrastive reflection, allowing agents to learn from past interactions and dynamically improve their search efficiency; and 2) using multi-agent debate to provide reliable state evaluation. Moreover, we improve the agent's performance by fine-tuning GPT-4o through self-learning, using R-MCTS generated tree traversals without any human-provided labels. On the challenging VisualWebArena benchmark, our GPT-4o-based R-MCTS agent achieves a 6% to 30% relative improvement across various tasks compared to the previous state-of-the-art. Additionally, we show that the knowledge gained from test-time search can be effectively transferred back to GPT-4o via fine-tuning. The fine-tuned GPT-4o matches 97\% of R-MCTS's performance while reducing compute usage by a factor of four at test time. Furthermore, qualitative results reveal that the fine-tuned GPT-4o model demonstrates the ability to explore the environment, evaluate a state, and backtrack to viable ones when it detects that the current state cannot lead to success. Moreover, our work demonstrates the compute scaling properties in both training - data collection with R-MCTS - and testing time. These results suggest a promising research direction to enhance VLMs' reasoning and planning capabilities for agentic applications via test-time search and self-learning.

</details>

---

## 192. Natural Language Inference Improves Compositionality in Vision-Language Models

- [ ] Natural Language Inference Improves Compositionality in Vision-Language Models | https://iclr.cc/virtual/2025/poster/30305

- **Link**: https://iclr.cc/virtual/2025/poster/30305

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Compositional reasoning in Vision-Language Models (VLMs) remains challenging as these models often struggle to relate objects, attributes, and spatial relationships. Recent methods aim to address these limitations by relying on the semantics of the textual description, using Large Language Models (LLMs) to break them down into subsets of questions and answers. However, these methods primarily operate on the surface level, failing to incorporate deeper lexical understanding while introducing incorrect assumptions generated by the LLM. In response to these issues, we present Caption Expansion with Contradictions and Entailments (CECE), a principled approach that leverages Natural Language Inference (NLI) to generate entailments and contradictions from a given premise. CECE produces lexically diverse sentences while maintaining their core meaning. Through extensive experiments, we show that CECE enhances interpretability and reduces overreliance on biased or superficial features. By balancing CECE along the original premise, we achieve significant improvements over previous methods without requiring additional fine-tuning, producing state-of-the-art results on benchmarks that score agreement with human judgments for image-text alignment, and achieving an increase in performance on Winoground of $+19.2\%$ (group score) and $+12.9\%$ on EqBen (group score) over the best prior work (finetuned with targeted data).

</details>

---

## 193. CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning

- [ ] CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning | https://iclr.cc/virtual/2025/poster/30334

- **Link**: https://iclr.cc/virtual/2025/poster/30334

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have shown broad effectiveness due to extensive training that aligns visual inputs with corresponding language responses. However, this conclusive alignment training causes models to overlook essential visual reasoning, leading to failures in handling detailed visual tasks and producing unfaithful responses. Drawing inspiration from human cognition in solving visual problems (e.g., marking, zoom in), this paper introduces Chain of Manipulations, a mechanism that enables VLMs to tackle problems step-by-step with evidence. After training, models can solve various visual problems by eliciting intrinsic manipulations (e.g., grounding, zoom in) with results (e.g., boxes, image) actively without relying external tools, while also allowing users to trace error causes. In this paper, we study the comprehensive methodology that includes: (1) a flexible design of manipulations based on extensive analysis, (2) an efficient automated data generation pipeline, (3) a compatible VLM architecture capable of multi-turn, multi-image, and (4) a model training process for versatile capabilities. With the design, we also manually annotate 6K high-quality samples for challenging graphical mathematical problems. Our trained model, CogCoM, equipped with this mechanism and 17B parameters, achieves SOTA performance across 9 benchmarks in 4 categories, demonstrating its effectiveness while maintaining interpretability. Code, model, and data are available at https://github.com/THUDM/CogCoM.

</details>

---

## 194. Proxy Denoising for Source-Free Domain Adaptation

- [ ] Proxy Denoising for Source-Free Domain Adaptation | https://iclr.cc/virtual/2025/poster/30349

- **Link**: https://iclr.cc/virtual/2025/poster/30349

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Source-Free Domain Adaptation (SFDA) aims to adapt a pre-trained source model to an unlabeled target domain with no access to the source data. Inspired by the success of large Vision-Language (ViL) models in many applications, the latest research has validated ViL's benefit for SFDA by using their predictions as pseudo supervision. However, we observe that ViL's supervision could be noisy and inaccurate at an unknown rate, potentially introducing additional negative effects during adaption. To address this thus-far ignored challenge, we introduce a novel Proxy Denoising ( ProDe ) approach. The key idea is to leverage the ViL model as a proxy to facilitate the adaptation process towards the latent domain-invariant space. Concretely, we design a proxy denoising mechanism to correct ViL's predictions. This is grounded on a proxy confidence theory that models the dynamic effect of proxy's divergence against the domain-invariant space during adaptation. To capitalize the corrected proxy, we further derive a mutual knowledge distilling regularization. Extensive experiments show that ProDe significantly outperforms the current state-of-the-art alternatives under both conventional closed-set setting and the more challenging open-set, partial-set, generalized SFDA, multi-target, multi-source, and test-time settings. Our code and data are available at https://github.com/tntek/source-free-domain-adaptation.

</details>

---

## 195. Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs

- [ ] Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs | https://iclr.cc/virtual/2025/poster/30358

- **Link**: https://iclr.cc/virtual/2025/poster/30358

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its significant impact on improving text generation quality.

</details>

---

## 196. Adapt-$\infty$: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection

- [ ] Adapt-$\infty$: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection | https://iclr.cc/virtual/2025/poster/30379

- **Link**: https://iclr.cc/virtual/2025/poster/30379

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual instruction datasets from various distributors are released at different times and often contain a significant number of semantically redundant text-image pairs, depending on their task compositions (i.e., skills) or reference sources. This redundancy greatly limits the efficient deployment of continually adaptable multimodal large language models, hindering their ability to refine existing skills and acquire new competencies over time. To address this, we reframe the problem of lifelong Instruction Tuning (LiIT) via data selection, where the model automatically selects beneficial samples to learn from earlier and new datasets based on the current state of acquired knowledge in the model. Based on empirical analyses that show that selecting the best data subset using a static importance measure is often ineffective for multi-task datasets with evolving distributions, we propose Adapt-$\infty$, a new multi-way and adaptive data selection approach that dynamically balances sample efficiency and effectiveness during LiIT. We first construct pseudo-skill clusters by grouping gradient-based sample vectors. Next, we select the best-performing data selector for each skill cluster from a pool of selector experts, including our newly proposed scoring function, Image Grounding score. This data selector samples a subset of the most important samples from each skill cluster for training. To prevent the continuous increase in the size of the dataset pool during LIT, which would result in excessive computation, we further introduce a cluster-wise permanent data pruning strategy to remove the most semantically redundant samples from each cluster, keeping computational requirements manageable. We validate the effectiveness and efficiency of Adapt-$\infty$ over a sequence of various multimodal instruction tuning datasets with various tasks, including (Knowledge) VQA, multilingual, grounding, reasoning, language-only, and multi-image comprehension tasks. Training with samples selected by Adapt-$\infty$ alleviates catastrophic forgetting, especially for rare tasks, and promotes forward transfer across the continuum using only a fraction of the original datasets.

</details>

---

## 197. Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection

- [ ] Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection | https://iclr.cc/virtual/2025/poster/30380

- **Link**: https://iclr.cc/virtual/2025/poster/30380

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Out-of-Distribution (OOD) detection, aiming to distinguish outliers from known categories, has gained prominence in practical scenarios. Recently, the advent of vision-language models (VLM) has heightened interest in enhancing OOD detection for VLM through few-shot tuning. However, existing methods mainly focus on optimizing global prompts, ignoring refined utilization of local information with regard to outliers. Motivated by this, we freeze global prompts and introduce Local-Prompt, a novel coarse-to-fine tuning paradigm to emphasize regional enhancement with local prompts. Our method comprises two integral components: global prompt guided negative augmentation and local prompt enhanced regional regularization. The former utilizes frozen, coarse global prompts as guiding cues to incorporate negative augmentation, thereby leveraging local outlier knowledge. The latter employs trainable local prompts and a regional regularization to capture local information effectively, aiding in outlier identification. We also propose regional-related metric to empower the enrichment of OOD detection. Moreover, since our approach explores enhancing local prompts only, it can be seamlessly integrated with trained global prompts during inference to boost the performance. Comprehensive experiments demonstrate the effectiveness and potential of our method. Notably, our method reduces average FPR95 by 5.17% against state-of-the-art method in 4-shot tuning on challenging ImageNet-1k dataset, even outperforming 16-shot results of previous methods.

</details>

---

## 198. EMMA: Empowering Multi-modal Mamba with Structural and Hierarchical Alignment

- [ ] EMMA: Empowering Multi-modal Mamba with Structural and Hierarchical Alignment | https://iclr.cc/virtual/2025/poster/30381

- **Link**: https://iclr.cc/virtual/2025/poster/30381

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mamba-based architectures have shown to be a promising new direction for deep learning models owing to their competitive performance and sub-quadratic deployment speed. However, current Mamba multi-modal large language models (MLLM) are insufficient in extracting visual features, leading to imbalanced cross-modal alignment between visual and textural latents, negatively impacting performance on multi-modal tasks. In this work, we propose Empowering Multi-modal Mamba with Structural and Hierarchical Alignment (EMMA), which enables the MLLM to extract fine-grained visual information. Specifically, we propose a pixel-wise alignment module to autoregressively optimize the learning and processing of spatial image-level features along with textual tokens, enabling structural alignment at the image level. In addition, to prevent the degradation of visual information during the cross-model alignment process, we propose a multi-scale feature fusion (MFF) module to combine multi-scale visual features from intermediate layers, enabling hierarchical alignment at the feature level. Extensive experiments are conducted across a variety of multi-modal benchmarks. Our model shows lower latency than other Mamba-based MLLMs and is nearly four times faster than transformer-based MLLMs of similar scale during inference. Due to better cross-modal alignment, our model exhibits lower degrees of hallucination and enhanced sensitivity to visual details, which manifests in superior performance across diverse multi-modal benchmarks. Code provided at https://github.com/xingyifei2016/EMMA.

</details>

---

## 199. Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models

- [ ] Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models | https://iclr.cc/virtual/2025/poster/30385

- **Link**: https://iclr.cc/virtual/2025/poster/30385

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (LLMs) are pivotal in revolutionizing customer support and operations by integrating multiple modalities such as text, images, and audio. Federated Prompt Learning (FPL) is a recently proposed approach that combines pre-trained multimodal LLMs such as vision-language models with federated learning to create personalized, privacy-preserving AI systems. However, balancing the competing goals of personalization, generalization, and privacy remains a significant challenge. Over-personalization can lead to overfitting, reducing generalizability, while stringent privacy measures, such as differential privacy, can hinder both personalization and generalization. In this paper, we propose a Differentially Private Federated Prompt Learning (DP-FPL) approach to tackle this challenge by leveraging a low-rank factorization scheme to capture generalization while maintaining a residual term that preserves expressiveness for personalization. To ensure privacy, we introduce a novel method where we apply local differential privacy to the two low-rank components of the local prompt, and global differential privacy to the global prompt. Our approach mitigates the impact of privacy noise on the model performance while balancing the tradeoff between personalization and generalization. Extensive experiments demonstrate the effectiveness of our approach over other benchmarks.

</details>

---

## 200. Lumina-T2X: Scalable Flow-based Large Diffusion Transformer for Flexible Resolution Generation

- [ ] Lumina-T2X: Scalable Flow-based Large Diffusion Transformer for Flexible Resolution Generation | https://iclr.cc/virtual/2025/poster/30398

- **Link**: https://iclr.cc/virtual/2025/poster/30398

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Sora unveils the potential of scaling Diffusion Transformer (DiT) for generating photorealistic images and videos at arbitrary resolutions, aspect ratios, and durations, yet it still lacks sufficient implementation details. In this paper, we introduce the Lumina-T2X family -- a series of Flow-based Large Diffusion Transformers (Flag-DiT) equipped with zero-initialized attention, as a simple and scalable generative framework that can be adapted to various modalities, e.g., transforming noise into images, videos, multi-view 3D objects, or audio clips conditioned on text instructions. By tokenizing the latent spatial-temporal space and incorporating learnable placeholders such as |[nextline]| and |[nextframe]| tokens, Lumina-T2X seamlessly unifies the representations of different modalities across various spatial-temporal resolutions. Advanced techniques like RoPE, KQ-Norm, and flow matching enhance the stability, flexibility, and scalability of Flag-DiT, enabling models of Lumina-T2X to scale up to 7 billion parameters and extend the context window to 128K tokens. This is particularly beneficial for creating ultra-high-definition images with our Lumina-T2I model and long 720p videos with our Lumina-T2V model. Remarkably, Lumina-T2I, powered by a 5-billion-parameter Flag-DiT, requires only 35% of the training computational costs of a 600-million-parameter naive DiT (PixArt-alpha), indicating that increasing the number of parameters significantly accelerates convergence of generative models without compromising visual quality. Our further comprehensive analysis underscores Lumina-T2X's preliminary capability in resolution extrapolation, high-resolution editing, generating consistent 3D views, and synthesizing videos with seamless transitions. All code and checkpoints of Lumina-T2X are released at https://github.com/Alpha-VLLM/Lumina-T2X to further foster creativity, transparency, and diversity in the generative AI community.

</details>

---

## 201. ToVE: Efficient Vision-Language Learning via Knowledge Transfer from Vision Experts

- [ ] ToVE: Efficient Vision-Language Learning via Knowledge Transfer from Vision Experts | https://iclr.cc/virtual/2025/poster/30410

- **Link**: https://iclr.cc/virtual/2025/poster/30410

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language (VL) learning requires extensive visual perception capabilities, such as fine-grained object recognition and spatial perception. Recent works typically rely on training huge models on massive datasets to develop these capabilities. As a more efficient alternative, this paper proposes a new framework that Transfers the knowledge from a hub of Vision Experts (ToVE) for efficient VL learning, leveraging pre-trained vision expert models to promote visual perception capability. Specifically, building on a frozen CLIP image encoder that provides vision tokens for image-conditioned language generation, ToVE introduces a hub of multiple vision experts and a token-aware gating network that dynamically routes expert knowledge to vision tokens. In the transfer phase, we propose a "residual knowledge transfer" strategy, which not only preserves the generalizability of the vision tokens but also allows selective detachment of low-contributing experts to improve inference efficiency. Further, we explore to merge these expert knowledge to a single CLIP encoder, creating a knowledge-merged CLIP that produces more informative vision tokens without expert inference during deployment. Experiment results across various VL tasks demonstrate that the proposed ToVE achieves competitive performance with two orders of magnitude fewer training data.

</details>

---

## 202. DynaPrompt: Dynamic Test-Time Prompt Tuning

- [ ] DynaPrompt: Dynamic Test-Time Prompt Tuning | https://iclr.cc/virtual/2025/poster/30415

- **Link**: https://iclr.cc/virtual/2025/poster/30415

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Test-time prompt tuning enhances zero-shot generalization of vision-language models but tends to ignore the relatedness among test samples during inference. Online test-time prompt tuning provides a simple way to leverage the information in previous test samples, albeit with the risk of prompt collapse due to error accumulation. To enhance test-time prompt tuning, we propose DynaPrompt, short for dynamic test-time prompt tuning, exploiting relevant data distribution information while reducing error accumulation. Built on an online prompt buffer, DynaPrompt adaptively selects and optimizes the relevant prompts for each test sample during tuning. Specifically, we introduce a dynamic prompt selection strategy based on two metrics: prediction entropy and probability difference. For unseen test data information, we develop dynamic prompt appending, which allows the buffer to append new prompts and delete the inactive ones. By doing so, the prompts are optimized to exploit beneficial information on specific test data, while alleviating error accumulation. Experiments on fourteen datasets demonstrate the effectiveness of dynamic test-time prompt tuning.

</details>

---

## 203. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials

- [ ] AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials | https://iclr.cc/virtual/2025/poster/30416

- **Link**: https://iclr.cc/virtual/2025/poster/30416

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interface (GUI) agents hold great potential for automating complex tasks across diverse digital environments, from web applications to desktop software. However, the development of such agents is hindered by the lack of high-quality, multi-step trajectory data required for effective training. Existing approaches rely on expensive and labor-intensive human annotation, making them unsustainable at scale. To address this challenge, we propose AgentTrek, a scalable data synthesis pipeline that generates high-quality web agent trajectories by leveraging web tutorials. Our method automatically gathers tutorial-like texts from the internet, transforms them into task goals with step-by-step instructions, and employs a visual-language model (VLM) agent to simulate their execution in a real digital environment. A VLM-based evaluator ensures the correctness of the generated trajectories. We demonstrate that training GUI agents with these synthesized trajectories significantly improves their grounding and planning performance over the current models. Moreover, our approach is more cost-efficient compared to traditional human annotation methods. This work underscores the potential of guided replay with web tutorials as a viable strategy for large-scale GUI agent training, paving the way for more capable and autonomous digital agents.

</details>

---

## 204. Have the VLMs Lost Confidence? A Study of Sycophancy in VLMs

- [ ] Have the VLMs Lost Confidence? A Study of Sycophancy in VLMs | https://iclr.cc/virtual/2025/poster/30427

- **Link**: https://iclr.cc/virtual/2025/poster/30427

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the study of LLMs, sycophancy represents a prevalent hallucination that poses significant challenges to these models. Specifically, LLMs often fail to adhere to original correct responses, instead blindly agreeing with users' opinions, even when those opinions are incorrect or malicious. However, research on sycophancy in visual language models (VLMs) has been scarce. In this work, we extend the exploration of sycophancy from LLMs to VLMs, introducing the MM-SY benchmark to evaluate this phenomenon. We present evaluation results from multiple representative models, addressing the gap in sycophancy research for VLMs. To mitigate sycophancy, we propose a synthetic dataset for training and employ methods based on prompts, supervised fine-tuning, and DPO. Our experiments demonstrate that these methods effectively alleviate sycophancy in VLMs. Additionally, we probe VLMs to assess the semantic impact of sycophancy and analyze the attention distribution of visual tokens. Our findings indicate that the ability to prevent sycophancy is predominantly observed in higher layers of the model. The lack of attention to image knowledge in these higher layers may contribute to sycophancy, and enhancing image attention at high layers proves beneficial in mitigating this issue.

</details>

---

## 205. MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs

- [ ] MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs | https://iclr.cc/virtual/2025/poster/30449

- **Link**: https://iclr.cc/virtual/2025/poster/30449

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have experienced rapid progress in visual recognition tasks in recent years. Given their potential integration into many critical applications, it is important to understand the limitations of their visual perception. In this work, we study whether MLLMs can perceive small visual details as effectively as large ones when answering questions about images. We observe that their performance is very sensitive to the size of the visual subject of the question, and further show that this effect is in fact causal by conducting an intervention study. Next, we study the attention patterns of MLLMs when answering visual questions, and intriguingly find that they consistently know where to look, even when they provide the wrong answer. Based on these findings, we then propose training-free visual intervention methods that leverage the internal knowledge of any MLLM itself, in the form of attention and gradient maps, to enhance its perception of small visual details. We evaluate our proposed methods on two widely-used MLLMs and seven visual question answering benchmarks and show that they can significantly improve MLLMs' accuracy without requiring any training. Our results elucidate the risk of applying MLLMs to visual recognition tasks concerning small details and indicate that visual intervention using the model's internal state is a promising direction to mitigate this risk. Our code is available at: https://github.com/saccharomycetes/mllms_know.

</details>

---

## 206. DSBench: How Far Are Data Science Agents from Becoming Data Science Experts?

- [ ] DSBench: How Far Are Data Science Agents from Becoming Data Science Experts? | https://iclr.cc/virtual/2025/poster/30458

- **Link**: https://iclr.cc/virtual/2025/poster/30458

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) have demonstrated impressive language/vision reasoning abilities, igniting the recent trend of building agents for targeted applications such as shopping assistants or AI software engineers. Recently, many data science benchmarks have been proposed to investigate their performance in the data science domain. However, existing data science benchmarks still fall short when compared to real-world data science applications due to their simplified settings. To bridge this gap, we introduce DSBench, a comprehensive benchmark designed to evaluate data science agents with realistic tasks. This benchmark includes 466 data analysis tasks and 74 data modeling tasks, sourced from Eloquence and Kaggle competitions. DSBench offers a realistic setting by encompassing long contexts, multimodal task backgrounds, reasoning with large data files and multi-table structures, and performing end-to-end data modeling tasks. Our evaluation of state-of-the-art LLMs, LVLMs, and agents shows that they struggle with most tasks, with the best agent solving only 34.12% of data analysis tasks and achieving a 34.74% Relative Performance Gap (RPG). These findings underscore the need for further advancements in developing more practical, intelligent, and autonomous data science agents.

</details>

---

## 207. Probabilistic Language-Image Pre-Training

- [ ] Probabilistic Language-Image Pre-Training | https://iclr.cc/virtual/2025/poster/30478

- **Link**: https://iclr.cc/virtual/2025/poster/30478

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) embed aligned image-text pairs into a joint space but often rely on deterministic embeddings, assuming a one-to-one correspondence between images and texts. This oversimplifies real-world relationships, which are inherently many-to-many, with multiple captions describing a single image and vice versa. We introduce Probabilistic Language-Image Pre-training (ProLIP), the first probabilistic VLM pre-trained on a billion-scale image-text dataset using only probabilistic objectives, achieving a strong zero-shot capability (e.g., 74.6% ImageNet zero-shot accuracy with ViT-B/16). ProLIP efficiently estimates uncertainty by an ``uncertainty token'' without extra parameters. We also introduce a novel inclusion loss that enforces distributional inclusion relationships between image-text pairs and between original and masked inputs. Experiments demonstrate that, by leveraging uncertainty estimates, ProLIP benefits downstream tasks and aligns with intuitive notions of uncertainty, e.g., shorter texts being more uncertain and more general inputs including specific ones. Utilizing text uncertainties, we further improve ImageNet accuracy from 74.6% to 75.8% (under a few-shot setting), supporting the practical advantages of our probabilistic approach. The code is available at https://github.com/naver-ai/prolip

</details>

---

## 208. MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs

- [ ] MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs | https://iclr.cc/virtual/2025/poster/30477

- **Link**: https://iclr.cc/virtual/2025/poster/30477

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current multimodal misinformation detection (MMD) methods often assume a single source and type of forgery for each sample, which is insufficient for real-world scenarios where multiple forgery sources coexist. The lack of a benchmark for mixed-source misinformation has hindered progress in this field. To address this, we introduce MMFakeBench, the first comprehensive benchmark for mixed-source MMD. MMFakeBench includes 3 critical sources: textual veracity distortion, visual veracity distortion, and cross-modal consistency distortion, along with 12 sub-categories of misinformation forgery types. We further conduct an extensive evaluation of 6 prevalent detection methods and 15 Large Vision-Language Models (LVLMs) on MMFakeBench under a zero-shot setting. The results indicate that current methods struggle under this challenging and realistic mixed-source MMD setting. Additionally, we propose MMD-Agent, a novel approach to integrate the reasoning, action, and tool-use capabilities of LVLM agents, significantly enhancing accuracy and generalization. We believe this study will catalyze future research into more realistic mixed-source multimodal misinformation and provide a fair evaluation of misinformation detection methods.

</details>

---

## 209. Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model

- [ ] Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model | https://iclr.cc/virtual/2025/poster/30482

- **Link**: https://iclr.cc/virtual/2025/poster/30482

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have revolutionized machine learning by leveraging large pre-trained models to tackle various downstream tasks. Although label, training, and data efficiency have improved, many state-of-the-art VLMs still require task-specific hyperparameter tuning and fail to fully exploit test samples. To overcome these challenges, we propose a graph-based approach for label-efficient adaptation and inference. Our method dynamically constructs a graph over text prompts, few-shot examples, and test samples, using label propagation for inference without task-specific tuning. Unlike existing zero-shot label propagation techniques, our approach requires no additional unlabeled support set and effectively leverages the test sample manifold through dynamic graph expansion. We further introduce a context-aware feature re-weighting mechanism to improve task adaptation accuracy. Additionally, our method supports efficient graph expansion, enabling real-time inductive inference. Extensive evaluations on downstream tasks, such as fine-grained categorization and out-of-distribution generalization, demonstrate the effectiveness of our approach. The source code is available at https://github.com/Yushu-Li/ECALP.

</details>

---

## 210. Digi-Q: Learning VLM Q-Value Functions for Training Device-Control Agents

- [ ] Digi-Q: Learning VLM Q-Value Functions for Training Device-Control Agents | https://iclr.cc/virtual/2025/poster/30502

- **Link**: https://iclr.cc/virtual/2025/poster/30502

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While a number of existing approaches for building foundation model agents rely on prompting or fine-tuning with human demonstrations, it is not sufficient in dynamic environments (e.g., mobile device control). On-policy reinforcement learning (RL) should address these limitations, but collecting actual rollouts in an environment is often undesirable in truly open-ended agentic problems such as mobile device control or interacting with humans, where each unit of interaction is associated with a cost. In such scenarios, a method for policy learning that can utilize off-policy experience by learning a trained action-value function is much more effective. In this paper, we develop an approach, called Digi-Q, to train VLM-based action-value Q-functions which are then used to extract the agent policy. We study our approach in the mobile device control setting. Digi-Q trains the Q-function using offline temporal-difference (TD) learning, on top of frozen, intermediate-layer features of a VLM. Compared to fine-tuning the whole VLM, this approach saves us compute and enhances scalability. To make the VLM features amenable for representing the Q-function, we need to employ an initial phase of fine-tuning to amplify coverage over actionable information needed for value function. Once trained, we use this Q-function via a Best-of-N policy extraction operator that imitates the best action out of multiple candidate actions from the current policy as ranked by the value function, enabling policy improvement without environment interaction. Digi-Q outperforms several prior methods on user-scale device control tasks in Android-in-the-Wild, attaining 21.2% improvement over prior best-performing method. In some cases, our Digi-Q ap-proach already matches state-of-the-art RL methods that require interaction. The project is open-sourced at https://github.com/DigiRL-agent/digiq

</details>

---

## 211. Class Distribution-induced Attention Map for Open-vocabulary Semantic Segmentations

- [ ] Class Distribution-induced Attention Map for Open-vocabulary Semantic Segmentations | https://iclr.cc/virtual/2025/poster/30516

- **Link**: https://iclr.cc/virtual/2025/poster/30516

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-vocabulary semantic segmentation is a challenging task that assigns seen or unseen class labels to individual pixels. While recent works with vision-language models (VLMs) have shown promising results in zero-shot semantic segmentation, they still struggle to accurately localize class-related objects. In this work, we argue that CLIP-based prior works yield patch-wise noisy class predictions while having highly correlated class distributions for each object. Then, we propose Class Distribution-induced Attention Map, dubbed CDAM, that is generated by the Jensen-Shannon divergence between class distributions of two patches that belong to the same (class) object. This CDAM can be used for open-vocabulary semantic segmentation by integrating it into the final layer of CLIP to enhance the capability to accurately localize desired classes. Our class distribution-induced attention scheme can easily work with multi-scale image patches as well as augmented text prompts for further enhancing attention maps. By exploiting class distribution, we also propose robust entropy-based background thresholding for the inference of semantic segmentation. Interestingly, the core idea of our proposed method does not conflict with other prior arts in zero-shot semantic segmentation, thus can be synergetically used together, yielding substantial improvements in performance across popular semantic segmentation benchmarks.

</details>

---

## 212. Animate Your Thoughts: Reconstruction of Dynamic Natural Vision from Human Brain Activity

- [ ] Animate Your Thoughts: Reconstruction of Dynamic Natural Vision from Human Brain Activity | https://iclr.cc/virtual/2025/poster/30544

- **Link**: https://iclr.cc/virtual/2025/poster/30544

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reconstructing human dynamic vision from brain activity is a challenging task with great scientific significance.  Although prior video reconstruction methods have made substantial progress, they still suffer from several limitations, including: (1) difficulty in simultaneously reconciling semantic (e.g. categorical descriptions), structure (e.g. size and color), and consistent motion information (e.g. order of frames); (2) low temporal resolution of fMRI, which poses a challenge in decoding multiple frames of video dynamics from a single fMRI frame; (3) reliance on video generation models, which introduces ambiguity regarding whether the dynamics observed in the reconstructed videos are genuinely derived from fMRI data or are hallucinations from generative model. To overcome these limitations,  we propose a two-stage model named Mind-Animator. During the fMRI-to-feature stage, we decouple semantic, structure, and motion features from fMRI. Specifically, we employ fMRI-vision-language tri-modal contrastive learning to decode semantic feature from fMRI and design a sparse causal attention mechanism for decoding multi-frame video motion features through a next-frame-prediction task. In the feature-to-video stage, these features are integrated into videos using an inflated Stable Diffusion, effectively eliminating external video data interference.  Extensive experiments on multiple video-fMRI datasets demonstrate that our model achieves state-of-the-art performance. Comprehensive visualization analyses further elucidate the interpretability of our model from a neurobiological perspective.  Project page: https://mind-animator-design.github.io/.

</details>

---

## 213. ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer

- [ ] ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer | https://iclr.cc/virtual/2025/poster/30543

- **Link**: https://iclr.cc/virtual/2025/poster/30543

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Diffusion models have emerged as a powerful generative technology and have been found to be applicable in various scenarios. Most existing foundational diffusion models are primarily designed for text-guided visual generation and do not support multi-modal conditions, which are essential for many visual editing tasks. This limitation prevents these foundational diffusion models from serving as a unified model in the field of visual generation, like GPT-4 in the natural language processing field. In this work, we propose ACE, an All-round Creator and Editor, which achieves comparable performance compared to those expert models in a wide range of visual generation tasks. To achieve this goal, we first introduce a unified condition format termed Long-context Condition Unit (LCU), and propose a novel Transformer-based diffusion model that uses LCU as input, aiming for joint training across various generation and editing tasks. Furthermore, we propose an efficient data collection approach to address the issue of the absence of available training data. It involves acquiring pairwise images with synthesis-based or clustering-based pipelines and supplying these pairs with accurate textual instructions by leveraging a fine-tuned multi-modal large language model. To comprehensively evaluate the performance of our model, we establish a benchmark of manually annotated pairs data across a variety of visual generation tasks. The extensive experimental results demonstrate the superiority of our model in visual generation fields. Thanks to the all-in-one capabilities of our model, we can easily build a multi-modal chat system that responds to any interactive request for image creation using a single model to serve as the backend, avoiding the cumbersome pipeline typically employed in visual agents.

</details>

---

## 214. Understanding and Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention

- [ ] Understanding and Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention | https://iclr.cc/virtual/2025/poster/30556

- **Link**: https://iclr.cc/virtual/2025/poster/30556

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) exhibit impressive capabilities in complex visual tasks but are prone to hallucination, especially in open-ended generation tasks. This paper explores why LVLMs tend to hallucinate and how to mitigate it. First, we conduct causal mediation analysis through counterfactual edits on specific modules in LVLMs. Our results disclose that Multi-Head Attention (MHA) modules contribute more to the probability of generating hallucination words than multi-layer perceptron modules. We then identify specific heads that are responsible for hallucination, referred to as hallucination heads. Second, we examine the behavior of hallucination heads. We find that they are concentrated in the middle and deeper layers, displaying a strong attention bias toward text tokens. Further, we show that the attention patterns of certain hallucination heads exhibit greater similarity to the base language model and change slowly during the instruction tuning process. Finally, we propose two simple yet effective methods to mitigate hallucination: one is training-free and can be applied directly during decoding, while the other involves fine-tuning. Both methods are targeted for hallucination heads to reduce their reliance on text tokens. Notably, our methods achieve up to 1.7x reduction in hallucination rate for the LLaVA-v1.5-7B model in COCO captioning task, outperforming existing baselines. Overall, our findings suggest that hallucinations in LVLMs are likely to stem from certain modules, and targeted interventions can effectively mitigate these issues.

</details>

---

## 215. Generating CAD Code with Vision-Language Models for 3D Designs

- [ ] Generating CAD Code with Vision-Language Models for 3D Designs | https://iclr.cc/virtual/2025/poster/30576

- **Link**: https://iclr.cc/virtual/2025/poster/30576

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generative AI has transformed the fields of Design and Manufacturing by providingefficient and automated methods for generating and modifying 3D objects. Oneapproach involves using Large Language Models (LLMs) to generate Computer-Aided Design (CAD) scripting code, which can then be executed to render a 3Dobject; however, the resulting 3D object may not meet the specified requirements.Testing the correctness of CAD generated code is challenging due to the complexityand structure of 3D objects (e.g., shapes, surfaces, and dimensions) that are notfeasible in code. In this paper, we introduce CADCodeVerify, a novel approach toiteratively verify and improve 3D objects generated from CAD code. Our approachworks by producing ameliorative feedback by prompting a Vision-Language Model(VLM) to generate and answer a set of validation questions to verify the generatedobject and prompt the VLM to correct deviations. To evaluate CADCodeVerify, weintroduce, CADPrompt, the first benchmark for CAD code generation, consisting of200 natural language prompts paired with expert-annotated scripting code for 3Dobjects to benchmark progress. Our findings show that CADCodeVerify improvesVLM performance by providing visual feedback, enhancing the structure of the 3Dobjects, and increasing the success rate of the compiled program. When applied toGPT-4, CADCodeVerify achieved a 7.30% reduction in Point Cloud distance and a5.0% improvement in success rate compared to prior work.

</details>

---

## 216. Lightweight Neural App Control

- [ ] Lightweight Neural App Control | https://iclr.cc/virtual/2025/poster/30577

- **Link**: https://iclr.cc/virtual/2025/poster/30577

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces a novel mobile phone control architecture, Lightweight Multi-modal App Control (LiMAC), for efficient interactions and control across various Android apps. LiMAC  takes as input a textual goal and a sequence of past mobile observations, such as screenshots and corresponding UI trees, to generate precise actions. To address the computational constraints inherent to smartphones, we introduce a small Action Transformer (AcT) integrated with a fine-tuned vision-language model (VLM) for real-time decision-making and task execution.  We evaluate LiMAC on two open-source mobile control datasets, demonstrating the superior performance of our small-form-factor approach against fine-tuned versions of open-source VLMs, such as Florence2 and Qwen2-VL. It also significantly outperforms prompt engineering baselines utilising closed-source foundation models like GPT-4o. More specifically, LiMAC increases the overall action accuracy by up to 19% compared to fine-tuned VLMs, and up to 42% compared to prompt-engineering baselines.

</details>

---

## 217. FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model

- [ ] FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model | https://iclr.cc/virtual/2025/poster/30597

- **Link**: https://iclr.cc/virtual/2025/poster/30597

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We aim to develop a model-based planning framework for world models that can be scaled with increasing model and data budgets for general-purpose manipulation tasks with only language and vision inputs. To this end, we present FLow-CentrIc generative Planning (FLIP), a model-based planning algorithm on visual space that features three key modules: 1) a multi-modal flow generation model as the general-purpose action proposal module; 2) a flow-conditioned video generation model as the dynamics module; and 3) a vision-language representation learning model as the value module. Given an initial image and language instruction as the goal, FLIP can progressively search for long-horizon flow and video plans that maximize the discounted return to accomplish the task. FLIP is able to synthesize long-horizon plans across objects, robots, and tasks with image flows as the general action representation, and the dense flow information also provides rich guidance for long-horizon video generation. In addition, the synthesized flow and video plans can guide the training of low-level control policies for robot execution. Experiments on diverse benchmarks demonstrate that FLIP can improve both the success rates and quality of long-horizon video plan synthesis and has the interactive world model property, opening up wider applications for future works. Video demos are on our website: https://nus-lins-lab.github.io/flipweb/.

</details>

---

## 218. Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality

- [ ] Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality | https://iclr.cc/virtual/2025/poster/30629

- **Link**: https://iclr.cc/virtual/2025/poster/30629

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have emerged as a central focus in both industry and academia, but often suffer from biases introduced by visual and language priors, which can lead to multimodal hallucination. These biases arise from the visual encoder and the Large Language Model (LLM) backbone, affecting the attention mechanism responsible for aligning multimodal inputs. Existing decoding-based mitigation methods focus on statistical correlations and overlook the causal relationships between attention mechanisms and model output, limiting their effectiveness in addressing these biases. To tackle this issue, we propose a causal inference framework termed CausalMM that applies structural causal modeling to MLLMs, treating modality priors as a confounder between attention mechanisms and output. Specifically, by employing backdoor adjustment and counterfactual reasoning at both the visual and language attention levels, our method mitigates the negative effects of modality priors and enhances the alignment of MLLM's inputs and outputs, with a maximum score improvement of 65.3% on 6 VLind-Bench indicators and 164 points on MME Benchmark compared to conventional methods. Extensive experiments validate the effectiveness of our approach while being a plug-and-play solution. Our code is available at: https://github.com/The-Martyr/CausalMM.

</details>

---

## 219. Sample then Identify: A General Framework for Risk Control and Assessment in Multimodal Large Language Models

- [ ] Sample then Identify: A General Framework for Risk Control and Assessment in Multimodal Large Language Models | https://iclr.cc/virtual/2025/poster/30685

- **Link**: https://iclr.cc/virtual/2025/poster/30685

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) exhibit promising advancements across various tasks, yet they still encounter significant trustworthiness issues. Prior studies apply Split Conformal Prediction (SCP) in language modeling to construct prediction sets with statistical guarantees. However, these methods typically rely on internal model logits or are restricted to multiple-choice settings, which hampers their generalizability and adaptability in dynamic, open-ended environments. In this paper, we introduce TRON , a t wo-step framework for r isk c o ntrol and assessme n t, applicable to any MLLM that supports sampling in both open-ended and closed-ended scenarios. TRON comprises two main components: (1) a novel conformal score to sample response sets of minimum size, and (2) a nonconformity score to identify high-quality responses based on self-consistency theory, controlling the error rates by two specific risk levels. Furthermore, we investigate semantic redundancy in prediction sets within open-ended contexts for the first time, leading to a promising evaluation metric for MLLMs based on average set size. Our comprehensive experiments across four Video Question-Answering (VideoQA) datasets utilizing eight MLLMs show that TRON achieves desired error rates bounded by two user-specified risk levels. Additionally, deduplicated prediction sets maintain adaptiveness while being more efficient and stable for risk assessment under different risk levels.

</details>

---

## 220. Multi-Reward as Condition for Instruction-based Image Editing

- [ ] Multi-Reward as Condition for Instruction-based Image Editing | https://iclr.cc/virtual/2025/poster/30691

- **Link**: https://iclr.cc/virtual/2025/poster/30691

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-quality training triplets (instruction, original image, edited image) are essential for instruction-based image editing. Predominant training datasets (e.g., InsPix2Pix) are created using text-to-image generative models (e.g., Stable Diffusion, DALL-E) which are not trained for image editing. Accordingly, these datasets suffer from inaccurate instruction following, poor detail preserving, and generation artifacts. In this paper, we propose to address the training data quality issue with multi-perspective reward data instead of refining the ground-truth image quality. 1) we first design a quantitative metric system based on best-in-class LVLM (Large Vision Language Model), i.e., GPT-4o in our case, to evaluate the generation quality from 3 perspectives, namely, instruction following, detail preserving, and generation quality. For each perspective, we collected quantitative score in $0\sim 5$ and text descriptive feedback on the specific failure points in ground-truth edited images, resulting in a high-quality editing reward dataset, i.e., RewardEdit20K. 2) We further proposed a novel training framework to seamlessly integrate the metric output, regarded as multi-reward, into editing models to learn from the imperfect training triplets. During training, the reward scores and text descriptions are encoded as embeddings and fed into both the latent space and the U-Net of the editing models as auxiliary conditions. During inference, we set these additional conditions to the highest score with no text description for failure points, to aim at the best generation outcome. 3) We also build a challenging evaluation benchmark with real-world images/photos and diverse editing instructions, named as Real-Edit. Experiments indicate that our multi-reward conditioned model outperforms its no-reward counterpart on two popular editing pipelines, i.e., InsPix2Pix and SmartEdit. Code is released at https://github.com/bytedance/Multi-Reward-Editing.

</details>

---

## 221. Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency

- [ ] Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency | https://iclr.cc/virtual/2025/poster/30720

- **Link**: https://iclr.cc/virtual/2025/poster/30720

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language pre-training models, such as CLIP, have demonstrated strong capability in rapidly adapting to downstream tasks through fine-tuning, and have been widely applied across various tasks. However, when the downstream tasks are constrained by limited image-text paired data, CLIP struggles to effectively address the domain gap between the pre-training and the target tasks. To address this limitation, we propose a novel semi-supervised CLIP training method coined SemiCLIP that leverages a small amount of image-text pairs alongside a large volume of images without text descriptions to enhance CLIP’s cross-modal alignment. To effectively utilize unlabeled images, we introduce semantic concept mining to improve task-specific visual representations by matching images with relevant concepts mined from labeled data. Leveraging matched semantic concepts, we construct learnable surrogate captions for unlabeled images and optimize a trapezoidal consistency to regulate the geometric structure of image-text pairs in the representation space. Experimental results demonstrate that our approach significantly improves the adaptability of CLIP in target tasks with limited labeled data, achieving gains ranging from 1.72\% -- 6.58\% for zero-shot classification accuracy and 2.32\% -- 3.23\% for image-text retrieval performance on standard benchmarks. The source code is available at https://github.com/Gank0078/SemiCLIP.

</details>

---

## 222. Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations

- [ ] Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations | https://iclr.cc/virtual/2025/poster/30724

- **Link**: https://iclr.cc/virtual/2025/poster/30724

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We investigate the internal representations of vision-language models (VLMs) to address hallucinations, a persistent challenge despite advances in model size and training. We project VLMs’ internal image representations to their language vocabulary and observe more confident output probabilities on real objects than hallucinated objects. We additionally use these output probabilities to spatially localize real objects. Building on this approach, we introduce a knowledge erasure algorithm that removes hallucinations by linearly orthogonalizing image features with respect to hallucinated object features. We show that targeted edits to a model’s latent representations can reduce hallucinations by up to 25.7% on the COCO2014 dataset while preserving performance. Our findings demonstrate how a deeper understanding of VLMs’ latent representations can enhance reliability and enable novel capabilities, such as zero-shot segmentation.

</details>

---

## 223. Data Pruning by Information Maximization

- [ ] Data Pruning by Information Maximization | https://iclr.cc/virtual/2025/poster/30725

- **Link**: https://iclr.cc/virtual/2025/poster/30725

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present InfoMax, a novel data pruning method, also known as coreset selection, designed to maximize the information content of selected samples while minimizing redundancy. By doing so, InfoMax enhances the overall informativeness of the coreset. The information of individual samples is measured by importance scores, which capture their influence or difficulty in model learning. To quantify redundancy, we use pairwise sample similarities, based on the premise that similar samples contribute similarly to the learning process.We formalize the coreset selection problem as a discrete quadratic programming (DQP) task, with the objective of maximizing the total information content, represented as the sum of individual sample contributions minus the redundancies introduced by similar samples within the coreset.To ensure practical scalability, we introduce an efficient gradient-based solver, complemented by sparsification techniques applied to the similarity matrix and dataset partitioning strategies. This enables InfoMax to seamlessly scale to datasets with millions of samples. Extensive experiments demonstrate the superior performance of InfoMax in various data pruning tasks, including image classification, vision-language pre-training, and instruction tuning for large language models.

</details>

---

## 224. Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities

- [ ] Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities | https://iclr.cc/virtual/2025/poster/30786

- **Link**: https://iclr.cc/virtual/2025/poster/30786

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spatial expressions in situated communication can be ambiguous, as their meanings vary depending on the frames of reference (FoR) adopted by speakers and listeners. While spatial language understanding and reasoning by vision-language models (VLMs) have gained increasing attention, potential ambiguities in these models are still under-explored. To address this issue, we present the COnsistent Multilingual Frame Of Reference Test (COMFORT), an evaluation protocol to systematically assess the spatial reasoning capabilities of VLMs. We evaluate nine state-of-the-art VLMs using COMFORT. Despite showing some alignment with English conventions in resolving ambiguities, our experiments reveal significant shortcomings of VLMs: notably, the models (1) exhibit poor robustness and consistency, (2) lack the flexibility to accommodate multiple FoRs, and (3) fail to adhere to language-specific or culture-specific conventions in cross-lingual tests, as English tends to dominate other languages. With a growing effort to align vision-language models with human cognitive intuitions, we call for more attention to the ambiguous nature and cross-cultural diversity of spatial reasoning.

</details>

---

## 225. See What You Are Told: Visual Attention Sink in Large Multimodal Models

- [ ] See What You Are Told: Visual Attention Sink in Large Multimodal Models | https://iclr.cc/virtual/2025/poster/30795

- **Link**: https://iclr.cc/virtual/2025/poster/30795

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal models (LMMs) "see" images by leveraging the attention mechanism between text and visual tokens in the transformer decoder. Ideally, these models should focus on key visual information relevant to the text token. However, recent findings indicate that LMMs have an extraordinary tendency to consistently allocate high attention weights to specific visual tokens, even when these tokens are irrelevant to the corresponding text. In this study, we investigate the property behind the appearance of these irrelevant visual tokens and examine their characteristics. Our findings show that this behavior arises due to the massive activation of certain hidden state dimensions, which resembles the attention sink found in language models. Hence, we refer to this phenomenon as the visual attention sink. In particular, our analysis reveals that removing the irrelevant visual sink tokens does not impact model performance, despite receiving high attention weights. Consequently, we recycle the attention to these tokens as surplus resources, redistributing the attention budget to enhance focus on the image. To achieve this, we introduce Visual Attention Redistribution (VAR), a method that redistributes attention in image-centric heads, which we identify as innately focusing on visual information. VAR can be seamlessly applied across different LMMs to improve performance on a wide range of tasks, including general vision-language tasks, visual hallucination tasks, and vision-centric tasks, all without the need for additional training, models, or inference steps. Experimental results demonstrate that VAR enables LMMs to process visual information more effectively by adjusting their internal attention mechanisms, offering a new direction to enhancing the multimodal capabilities of LMMs.

</details>

---

## 226. CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs

- [ ] CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs | https://iclr.cc/virtual/2025/poster/30804

- **Link**: https://iclr.cc/virtual/2025/poster/30804

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) still struggle with hallucinations despite their impressive capabilities. Recent studies have attempted to mitigate this by applying Direct Preference Optimization (DPO) to multimodal scenarios using preference pairs from text-based responses. However, our analysis of representation distributions reveals that multimodal DPO struggles to align image and text representations and to distinguish between hallucinated and non-hallucinated descriptions. To address these challenges,In this work, we propose a Cross-modal Hierarchical Direct Preference Optimization (CHiP) to address these limitations.We introduce a visual preference optimization module within the DPO framework, enabling MLLMs to learn from both textual and visual preferences simultaneously. Furthermore, we propose a hierarchical textual preference optimization module that allows the model to capture preferences at multiple granular levels, including response, segment, and token levels. We evaluate CHiP through both quantitative and qualitative analyses, with results across multiple benchmarks demonstrating its effectiveness in reducing hallucinations. On the Object HalBench dataset, CHiP outperforms DPO in hallucination reduction, achieving improvements of 52.7% and 55.5% relative points based on the base model Muffin and LLaVA models, respectively. We make all our datasets and code publicly available.

</details>

---

## 227. Modality-Specialized Synergizers for Interleaved Vision-Language Generalists

- [ ] Modality-Specialized Synergizers for Interleaved Vision-Language Generalists | https://iclr.cc/virtual/2025/poster/30822

- **Link**: https://iclr.cc/virtual/2025/poster/30822

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language Models (VLMs) have led to the emergence of Vision-Language Generalists (VLGs) capable of understanding and generating both text and images. However, seamlessly generating an arbitrary sequence of text and images remains a challenging task for the current VLGs. One primary limitation lies in applying a unified architecture and the same set of parameters to simultaneously model discrete text tokens and continuous image features. Recent works attempt to tackle this fundamental problem by introducing modality-aware expert models. However, they employ identical architectures to process both text and images, disregarding the intrinsic inductive biases in these two modalities. In this work, we introduce Modality-Specialized Synergizers (MoSS), a novel design that efficiently optimizes existing unified architectures of VLGs with modality-specialized adaptation layers, i.e., a Convolutional LoRA for modeling the local priors of image patches and a Linear LoRA for processing sequential text. This design enables more effective modeling of modality-specific features while maintaining the strong cross-modal integration gained from pretraining. In addition, to improve the instruction-following capability on interleaved text-and-image generation, we introduce LeafInstruct, the first open-sourced interleaved instruction tuning dataset comprising 184,982 high-quality instances on more than 10 diverse domains. Extensive experiments show that VLGs integrated with MoSS achieve state-of-the-art performance, significantly surpassing baseline VLGs in complex interleaved generation tasks. Furthermore, our method exhibits strong generalizability on different VLGs.

</details>

---

## 228. MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs

- [ ] MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs | https://iclr.cc/virtual/2025/poster/30836

- **Link**: https://iclr.cc/virtual/2025/poster/30836

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Effective evaluation of Multimodal Large Language Models (MLLMs) is essential for understanding their capabilities and limitations. In this paper, we introduce MIA-Bench, a benchmark designed to assess MLLMs’ ability to strictly adhere to complex instructions. Our benchmark comprises a diverse set of 400 image-prompt pairs, each crafted to challenge the models’ compliance with layered instructions in generating accurate and contextually appropriate responses. Evaluation results from a wide array of state-of-the-art MLLMs reveal significant variations in performance, highlighting areas for improvement in instruction fidelity. Additionally, we create extra training data and explore supervised fine-tuning and direct preference optimization to enhance the models’ ability to strictly follow instructions without compromising performance on other tasks. We hope this benchmark not only serves as a tool for measuring MLLM adherence to instructions, but also guides future developments in MLLM training methods.

</details>

---

## 229. Mini-Monkey: Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image Pyramid

- [ ] Mini-Monkey: Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image Pyramid | https://iclr.cc/virtual/2025/poster/30854

- **Link**: https://iclr.cc/virtual/2025/poster/30854

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, scaling images to high resolution has received much attention in multimodal large language models (MLLMs). Most existing practices adopt a sliding-window-style cropping strategy to adapt to resolution increase. Such a cropping strategy, however, can easily cut off objects and connected regions, which introduces semantic discontinuity and therefore impedes MLLMs from recognizing small or irregularly shaped objects or text, leading to a phenomenon we call the semantic sawtooth effect. This effect is particularly evident in lightweight MLLMs. To address this issue, we introduce a Complementary Image Pyramid (CIP), a simple, effective, and plug-and-play solution designed to mitigate semantic discontinuity during high-resolution image processing. In particular, CIP dynamically constructs an image pyramid to provide complementary semantic information for the cropping-based MLLMs, enabling it rich acquire semantics at all levels. Furthermore, we introduce a Scale Compression Mechanism (SCM) to reduce the additional computational overhead by compressing the redundant visual tokens. Our experiments demonstrate that CIP can consistently enhance the performance across diverse architectures (e.g., MiniCPM-V-2, InternVL2, and LLaVA-OneVision), various model capacity (1B$\rightarrow$8B), and different usage configurations (training-free and fine-tuning). Leveraging the proposed CIP and SCM, we introduce a lightweight MLLM, Mini-Monkey, which achieves remarkable performance in both general multimodal understanding and document understanding. On the OCRBench, the 2B-version Mini-Monkey even surpasses the 8B model InternVL2-8B by 12 score. Additionally, training Mini-Monkey is cheap, requiring only eight RTX 3090 GPUs. Code and models are available at https://github.com/Yuliang-Liu/Monkey.

</details>

---

## 230. Divergence-enhanced Knowledge-guided Context Optimization for Visual-Language Prompt Tuning

- [ ] Divergence-enhanced Knowledge-guided Context Optimization for Visual-Language Prompt Tuning | https://iclr.cc/virtual/2025/poster/30858

- **Link**: https://iclr.cc/virtual/2025/poster/30858

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt tuning vision-language models like CLIP has shown great potential in learning transferable representations for various downstream tasks. The main issue is how to mitigate the over-fitting problem on downstream tasks with limited training samples. While knowledge-guided context optimization has been proposed by constructing consistency constraints to handle catastrophic forgetting in the pre-trained backbone, it also introduces a bias toward pre-training. This paper proposes a novel and simple Divergence-enhanced Knowledge-guided Prompt Tuning (DeKg) method to address this issue.The key insight is that the bias toward pre-training can be alleviated by encouraging the independence between the learnable and the crafted prompt. Specifically, DeKg employs the Hilbert-Schmidt Independence Criterion (HSIC) to regularize the learnable prompts, thereby reducing their dependence on prior general knowledge, and enabling divergence induced by target knowledge. Comprehensive evaluations demonstrate that DeKg serves as a plug-and-play module can seamlessly integrate with existing knowledge-guided context optimization methods and achieves superior performance in three challenging benchmarks. We make our code available at https://github.com/cnunlp/DeKg.

</details>

---

## 231. Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering

- [ ] Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering | https://iclr.cc/virtual/2025/poster/30879

- **Link**: https://iclr.cc/virtual/2025/poster/30879

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

For vision-language models (VLMs), understanding the dynamic properties of objects and their interactions in 3D scenes from videos is crucial for effective reasoning about high-level temporal and action semantics. Although humans are adept at understanding these properties by constructing 3D and temporal (4D) representations of the world, current video understanding models struggle to extract these dynamic semantics, arguably because these models use cross-frame reasoning without underlying knowledge of the 3D/4D scenes.In this work, we introduce DynSuperCLEVR , the first video question answering dataset that focuses on language understanding of the dynamic properties of 3D objects. We concentrate on three physical concepts— velocity , acceleration , and collisions —within 4D scenes. We further generate three types of questions, including factual queries, future predictions, and counterfactual reasoning that involve different aspects of reasoning on these 4D dynamic properties.To further demonstrate the importance of explicit scene representations in answering these 4D dynamics questions, we propose NS-4DPhysics , a N eural- S ymbolic VideoQA model integrating Physics prior for 4D dynamic properties with explicit scene representation of videos. Instead of answering the questions directly from the video text input, our method first estimates the 4D world states with a 3D generative model powered by a physical prior, and then uses neural symbolic reasoning to answer the questions based on the 4D world states.Our evaluation on all three types of questions in DynSuperCLEVR shows that previous video question answering models and large multimodal models struggle with questions about 4D dynamics, while our NS-4DPhysics significantly outperforms previous state-of-the-art models.

</details>

---

## 232. Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters

- [ ] Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters | https://iclr.cc/virtual/2025/poster/30880

- **Link**: https://iclr.cc/virtual/2025/poster/30880

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have demonstrated strong capabilities across various visual understanding and reasoning tasks, driven by incorporating image representations into the token inputs of Large Language Models (LLMs). However, their real-world deployment is often constrained by high latency during inference due to the substantial compute required by the LLM to process the large number of input tokens, predominantly arising from the image. To reduce inference costs, one can either downsize the LLM or reduce the number of input tokens needed to represent the image, the latter of which has been the focus of many recent efforts around token compression. However, it is unclear what the optimal trade-off is given a fixed inference budget. We first characterize this optimal trade-off between the number of visual tokens and LLM parameters by establishing scaling laws that capture variations in performance with these two factors. Our results reveal a surprising trend: for visual reasoning tasks, the inference-optimal behavior in VLMs is achieved by using the largest LLM that fits within the inference budget while minimizing visual token count - often to a single token. While the token reduction literature has mainly focused on maintaining base model performance by modestly reducing the token count (e.g., $5-10\times$), our results indicate that the compute-optimal inference regime requires operating under even higher token compression ratios. Based on these insights, we take the first steps toward designing token compression algorithms tailored for high-compression settings, utilizing prompt-based compression of tokens. Our work underscores the performance and efficiency benefits of operating in low visual token regimes and the importance of developing tailored token reduction algorithms for such conditions.

</details>

---

## 233. UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting

- [ ] UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting | https://iclr.cc/virtual/2025/poster/30882

- **Link**: https://iclr.cc/virtual/2025/poster/30882

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multi-modal 3D pre-training methods have shown promising efficacy in learning joint representations of text, images, and point clouds. However, adopting point clouds as 3D representation fails to fully capture the intricacies of the 3D world and exhibits a noticeable gap between the discrete points and the dense 2D pixels of images. To tackle this issue, we propose UniGS, integrating 3D Gaussian Splatting (3DGS) into multi-modal pre-training to enhance the 3D representation. We first rely on the 3DGS representation to model the 3D world as a collection of 3D Gaussians with color and opacity, incorporating all the information of the 3D scene while establishing a strong connection with 2D images. Then, to achieve Language-Image-3D pertaining, UniGS starts with a pretrained vision-language model to establish a shared visual and textual space through extensive real-world image-text pairs. Subsequently, UniGS employs a 3D encoder to align the optimized 3DGS with the Language-Image representations to learn unified multi-modal representations. To facilitate the extraction of global explicit 3D features by the 3D encoder and achieve better cross-modal alignment, we additionally introduce a novel Gaussian-Aware Guidance module that guides the learning of fine-grained representations of the 3D domain. Through extensive experiments across the Objaverse, ABO, MVImgNet and SUN RGBD datasets with zero-shot classification, text-driven retrieval and open-world understanding tasks, we demonstrate the effectiveness of UniGS in learning a more general and stronger aligned multi-modal representation. Specifically, UniGS achieves leading results across different 3D tasks with remarkable improvements over previous SOTA, Uni3D, including on zero-shot classification (+9.36%), text-driven retrieval (+4.3%) and open-world understanding (+7.92%).

</details>

---

## 234. GeoX: Geometric Problem Solving Through Unified Formalized Vision-Language Pre-training

- [ ] GeoX: Geometric Problem Solving Through Unified Formalized Vision-Language Pre-training | https://iclr.cc/virtual/2025/poster/30888

- **Link**: https://iclr.cc/virtual/2025/poster/30888

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite their proficiency in general tasks, Multi-modal Large Language Models (MLLMs) struggle with automatic Geometry Problem Solving (GPS), which demands understanding diagrams, interpreting symbols, and performing complex reasoning. This limitation arises from their pre-training on natural images and texts, along with the lack of automated verification in the problem-solving process. Besides, current geometric specialists are limited by their task-specific designs, making them less effective for broader geometric problems. To this end, we present GeoX, a multi-modal large model focusing on geometric understanding and reasoning tasks. Given the significant differences between geometric diagram-symbol and natural image-text, we introduce unimodal pre-training to develop a diagram encoder and symbol decoder, enhancing the understanding of geometric images and corpora. Furthermore, we introduce geometry-language alignment, an effective pre-training paradigm that bridges the modality gap between unimodal geometric experts. We propose a Generator-And-Sampler Transformer (GS-Former) to generate discriminative queries and eliminate uninformative representations from unevenly distributed geometric signals. Finally, GeoX benefits from visual instruction tuning, empowering it to take geometric images and questions as input and generate verifiable solutions. Experiments show that GeoX outperforms both generalists and geometric specialists on publicly recognized benchmarks, such as GeoQA, UniGeo, Geometry3K, and PGPS9k. Our data and code will be released soon to accelerate future research on automatic GPS.

</details>

---

## 235. MMEgo: Towards Building Egocentric Multimodal LLMs for Video QA

- [ ] MMEgo: Towards Building Egocentric Multimodal LLMs for Video QA | https://iclr.cc/virtual/2025/poster/30907

- **Link**: https://iclr.cc/virtual/2025/poster/30907

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This research aims to comprehensively explore building a multimodal foundation model for egocentric video understanding.To achieve this goal, we work on three fronts. First, as there is a lack of QA data for egocentric video understanding, we automatically generate 7M high-quality QA samples for egocentric videos ranging from 30 seconds to one hour long in Ego4D based on human-annotated data.This is one of the largest egocentric QA datasets.Second, we contribute a challenging egocentric QA benchmark with 629 videos and 7,026 questions to evaluate the models' ability in recognizing and memorizing visual details across videos of varying lengths. We introduce a new de-biasing evaluation method to help mitigate the unavoidable language bias present in the models being evaluated.Third, we propose a specialized multimodal architecture featuring a novel ``Memory Pointer Prompting" mechanism. This design includes a global glimpse step to gain an overarching understanding of the entire video and identify key visual information, followed by a fallback step that utilizes the key visual information to generate responses. This enables the model to more effectively comprehend extended video content.With the data, benchmark, and model, we build MM-Ego, an egocentric multimodal LLM that shows powerful performance on egocentric video understanding.

</details>

---

## 236. Painting with Words: Elevating Detailed Image Captioning with Benchmark and Alignment Learning

- [ ] Painting with Words: Elevating Detailed Image Captioning with Benchmark and Alignment Learning | https://iclr.cc/virtual/2025/poster/30910

- **Link**: https://iclr.cc/virtual/2025/poster/30910

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image captioning has long been a pivotal task in visual understanding, with recent advancements in vision-language models (VLMs) significantly enhancing the ability to generate detailed image captions. However, the evaluation of detailed image captioning remains underexplored due to outdated evaluation metrics and coarse annotations. In this paper, we introduce DeCapBench along with a novel metric, DCScore, specifically designed for detailed captioning tasks. DCScore evaluates hallucinations and fine-grained comprehensiveness by deconstructing responses into the smallest self-sufficient units, termed primitive information units, and assessing them individually. Our evaluation shows that DCScore aligns more closely with human judgment than other rule-based or model-based metrics. Concurrently, DeCapBench exhibits a high correlation with VLM arena results on descriptive tasks, surpassing existing benchmarks for vision-language models. Additionally, we present an automatic fine-grained feedback collection method, FeedQuill, for preference optimization based on our advanced metric, demonstrating robust generalization capabilities across auto-generated preference data. Extensive experiments on multiple VLMs demonstrate that our method not only significantly reduces hallucinations but also enhances performance across various benchmarks, achieving superior detail captioning performance while surpassing GPT-4o.

</details>

---

## 237. How to Probe: Simple Yet Effective Techniques for Improving Post-hoc Explanations

- [ ] How to Probe: Simple Yet Effective Techniques for Improving Post-hoc Explanations | https://iclr.cc/virtual/2025/poster/30968

- **Link**: https://iclr.cc/virtual/2025/poster/30968

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Post-hoc importance attribution methods are a popular tool for “explaining” Deep Neural Networks (DNNs) and are inherently based on the assumption that the explanations can be applied independently of how the models were trained. Contrarily, in this work we bring forward empirical evidence that challenges this very notion. Surprisingly, we discover a strong dependency on and demonstrate that the training details of a pre-trained model’s classification layer (<10% of model parameters) play a crucial role, much more than the pre-training scheme itself. This is of high practical relevance: (1) as techniques for pre-training models are becoming increasingly diverse, understanding the interplay between these techniques and attribution methods is critical; (2) it sheds light on an important yet overlooked assumption of post-hoc attribution methods which can drastically impact model explanations and how they are interpreted eventually. With this finding we also present simple yet effective adjustments to the classification layers, that can significantly enhance the quality of model explanations. We validate our findings across several visual pre-training frameworks (fully-supervised, self-supervised, contrastive vision-language training) and analyse how they impact explanations for a wide range of attribution methods on a diverse set of evaluation metrics.

</details>

---

## 238. MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation

- [ ] MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation | https://iclr.cc/virtual/2025/poster/30978

- **Link**: https://iclr.cc/virtual/2025/poster/30978

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) frequently exhibit hallucination phenomena, but the underlying reasons remain poorly understood. In this paper, we present an empirical analysis and find that, although MLLMs incorrectly generate the objects in the final output, they are actually able to recognize visual objects in the preceding layers. We speculate that this may be due to the strong knowledge priors of the language model suppressing the visual information, leading to hallucinations. Motivated by this, we propose a novel dynamic correction decoding method for MLLMs DeCo, which adaptively selects the appropriate preceding layers and proportionally integrates knowledge into the final layer to adjust the output logits. Note that DeCo is model agnostic and can be seamlessly incorporated with various classic decoding strategies and applied to different MLLMs. We evaluate DeCo on widely-used benchmarks, demonstrating that it can reduce hallucination rates by a large margin compared to baselines, highlighting its potential to mitigate hallucinations. Code is available at https://github.com/zjunlp/DeCo.

</details>

---

## 239. Cached Multi-Lora Composition for Multi-Concept Image Generation

- [ ] Cached Multi-Lora Composition for Multi-Concept Image Generation | https://iclr.cc/virtual/2025/poster/30998

- **Link**: https://iclr.cc/virtual/2025/poster/30998

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Low-Rank Adaptation (LoRA) has emerged as a widely adopted technique in text-to-image models, enabling precise rendering of multiple distinct elements, such as characters and styles, in multi-concept image generation. However, current approaches face significant challenges when composing these LoRAs for multi-concept image generation, particularly as the number of LoRAs increases, resulting in diminished generated image quality. In this paper, we initially investigate the role of LoRAs in the denoising process through the lens of the Fourier frequency domain.Based on the hypothesis that applying multiple LoRAs could lead to "semantic conflicts", we have conducted empirical experiments and find that certain LoRAs amplify high-frequency features such as edges and textures, whereas others mainly focus on low-frequency elements, including the overall structure and smooth color gradients.Building on these insights, we devise a frequency domain based sequencing strategy to determine the optimal order in which LoRAs should be integrated during inference. This strategy offers a methodical and generalizable solution compared to the naive integration commonly found in existing LoRA fusion techniques.To fully leverage our proposed LoRA order sequence determination method in multi-LoRA composition tasks, we introduce a novel, training-free framework, Cached Multi-LoRA (CMLoRA), designed to efficiently integrate multiple LoRAs while maintaining cohesive image generation.With its flexible backbone for multi-LoRA fusion and a non-uniform caching strategy tailored to individual LoRAs, CMLoRA has the potential to reduce semantic conflicts in LoRA composition and improve computational efficiency.Our experimental evaluations demonstrate that CMLoRA outperforms state-of-the-art training-free LoRA fusion methods by a significant margin -- it achieves an average improvement of $2.19$% in CLIPScore, and $11.25%$% in MLLM win rate compared to LoraHub, LoRA Composite, and LoRA Switch.

</details>

---

## 240. Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models

- [ ] Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models | https://iclr.cc/virtual/2025/poster/31036

- **Link**: https://iclr.cc/virtual/2025/poster/31036

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language alignment in Large Vision-Language Models (LVLMs) successfully enables LLMs to understand visual input. However, we find that existing vision-language alignment methods fail to transfer the existing safety mechanism for text in LLMs to vision, which leads to vulnerabilities in toxic image. To explore the cause of this problem, we give the insightful explanation of where and how the safety mechanism of LVLMs operates and conduct comparative analysis between text and vision. We find that the hidden states at the specific transformer layers play a crucial role in the successful activation of safety mechanism, while the vision-language alignment at hidden states level in current methods is insufficient. This results in a semantic shift for input images compared to text in hidden states, therefore misleads the safety mechanism. To address this, we propose a novel Text-Guided vision-language Alignment method (TGA) for LVLMs. TGA retrieves the texts related to input vision and uses them to guide the projection of vision into the hidden states space in LLMs. Experiments show that \textbf{TGA} not only successfully transfers the safety mechanism for text in basic LLMs to vision in vision-language alignment for LVLMs without any safety fine-tuning on the visual modality but also maintains the general performance on various vision tasks (Safe and Good). Code is in supplemental material and will be released on GitHub after acceptance.

</details>

---

## 241. An Information Criterion for Controlled Disentanglement of Multimodal Data

- [ ] An Information Criterion for Controlled Disentanglement of Multimodal Data | https://iclr.cc/virtual/2025/poster/31055

- **Link**: https://iclr.cc/virtual/2025/poster/31055

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal representation learning seeks to relate and decompose information inherent in multiple modalities. By disentangling modality-specific information from information that is shared across modalities, we can improve interpretability and robustness and enable downstream tasks such as the generation of counterfactual outcomes. Separating the two types of information is challenging since they are often deeply entangled in many real-world applications. We propose $\textbf{Disentangled}$ $\textbf{S}$elf-$\textbf{S}$upervised $\textbf{L}$earning (DisentangledSSL), a novel self-supervised approach for learning disentangled representations. We present a comprehensive analysis of the optimality of each disentangled representation, particularly focusing on the scenario not covered in prior work where the so-called $\textit{Minimum Necessary Information}$ (MNI) point is not attainable. We demonstrate that \algo successfully learns shared and modality-specific features on multiple synthetic and real-world datasets and consistently outperforms baselines on various downstream tasks, including prediction tasks for vision-language data, as well as molecule-phenotype retrieval tasks for biological data.

</details>

---

## 242. Compositional Entailment Learning for Hyperbolic Vision-Language Models

- [ ] Compositional Entailment Learning for Hyperbolic Vision-Language Models | https://iclr.cc/virtual/2025/poster/31060

- **Link**: https://iclr.cc/virtual/2025/poster/31060

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image-text representation learning forms a cornerstone in vision-language models, where pairs of images and textual descriptions are contrastively aligned in a shared embedding space. Since visual and textual concepts are naturally hierarchical, recent work has shown that hyperbolic space can serve as a high-potential manifold to learn vision-language representation with strong downstream performance. In this work, for the first time we show how to fully leverage the innate hierarchical nature of hyperbolic embeddings by looking beyond individual image-text pairs. We propose Compositional Entailment Learning for hyperbolic vision-language models. The idea is that an image is not only described by a sentence but is itself a composition of multiple object boxes, each with their own textual description. Such information can be obtained freely by extracting nouns from sentences and using openly available localized grounding models. We show how to hierarchically organize images, image boxes, and their textual descriptions through contrastive and entailment-based objectives. Empirical evaluation on a hyperbolic vision-language model trained with millions of image-text pairs shows that the proposed compositional learning approach outperforms conventional Euclidean CLIP learning, as well as recent hyperbolic alternatives, with better zero-shot and retrieval generalization and clearly stronger hierarchical performance.

</details>

---

## 243. CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion

- [ ] CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion | https://iclr.cc/virtual/2025/poster/31072

- **Link**: https://iclr.cc/virtual/2025/poster/31072

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite impressive advancements in recent multimodal reasoning approaches, they are still limited in flexibility and efficiency, as these models typically process only a few fixed modality inputs and require updates to numerous parameters. This paper tackles these critical challenges and proposes CREMA, a generalizable, highly efficient, and modular modality-fusion framework that can incorporate many new modalities to enhance video reasoning. We first augment multiple informative modalities (such as optical flow, 3D point cloud, audio, thermal heatmap, and touch map) from given videos without extra human annotation by leveraging sensors or existing pre-trained models. Next, we introduce a query transformer with multiple parameter-efficient modules associated with each accessible modality. It projects diverse modality features to the LLM token embedding space, allowing the model to integrate different data types for response generation. Furthermore, we propose a novel progressive multimodal fusion design supported by a lightweight fusion module and modality-sequential training strategy. It helps compress information across various assisting modalities, maintaining computational efficiency in the LLM while improving performance. We validate our method on seven video-language reasoning tasks assisted by diverse modalities, including conventional VideoQA and Video-Audio/3D/Touch/Thermal QA, and achieve better/equivalent performance against strong multimodal LLMs, including OneLLM, BLIP-2, and SeViLA while reducing over 90% trainable parameters. We provide extensive analyses of CREMA, including the impact of each modality on reasoning domains, the design of the fusion module, and example visualizations.

</details>

---

## 244. From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities

- [ ] From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities | https://iclr.cc/virtual/2025/poster/31074

- **Link**: https://iclr.cc/virtual/2025/poster/31074

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models have made significant strides in integrating visual and textual information, yet they often struggle with effectively aligning these modalities. We introduce a novel image tokenizer that bridges this gap by applying the principle of Byte-Pair Encoding (BPE) to visual data. Unlike conventional approaches that rely on separate visual encoders, our method directly incorporates structural prior information into image tokens, mirroring the successful tokenization strategies used in text-only Large Language Models. This innovative approach enables Transformer models to more effectively learn and reason across modalities. Through theoretical analysis and extensive experiments, we demonstrate that our BPE Image Tokenizer significantly enhances MLLMs' multimodal understanding capabilities, even with limited training data. Leveraging this method, we develop Being-VL-0, a model that demonstrates superior performance across various benchmarks and shows promising scalability, potentially paving the way for more efficient and capable multimodal foundation models. For further details, visit our website https://github.com/BeingBeyond/Being-VL-0.

</details>

---

## 245. Visual Description Grounding Reduces Hallucinations and Boosts Reasoning in LVLMs

- [ ] Visual Description Grounding Reduces Hallucinations and Boosts Reasoning in LVLMs | https://iclr.cc/virtual/2025/poster/31078

- **Link**: https://iclr.cc/virtual/2025/poster/31078

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) often produce responses that misalign with factual information, a phenomenon known as hallucinations. While hallucinations are well-studied, the exact causes behind them remain underexplored. In this paper, we first investigate the root causes of hallucinations in LVLMs. Our findings reveal that existing mitigation techniques primarily reduce hallucinations for visual recognition prompts—those that require simple descriptions of visual elements—but fail for cognitive prompts that demand deliberate reasoning. We identify the core issue as a lack of true visual perception in LVLMs: although they can accurately recognize visual elements, they struggle to fully interpret these elements in the context of the input prompt and effectively link this recognition to their internal knowledge, which is critical for reasoning. To address this gap, we introduce Visual Description Grounded Decoding (VDGD), a simple, robust, and training-free method designed to enhance visual perception and improve reasoning capabilities in LVLMs. VDGD works by first generating a detailed description of the image and appending it as a prefix to the instruction. During response generation, tokens are sampled based on their KL divergence to the description, favoring candidates with lower divergence. Experimental results on multiple visual reasoning benchmarks and LVLMs demonstrate that VDGD consistently outperforms existing baselines  2% - 33%. Finally, we introduce VaLLu, a benchmark designed for comprehensive evaluation of the cognitive capabilities of LVLMs.

</details>

---

## 246. An Intelligent Agentic System for Complex Image Restoration Problems

- [ ] An Intelligent Agentic System for Complex Image Restoration Problems | https://iclr.cc/virtual/2025/poster/31076

- **Link**: https://iclr.cc/virtual/2025/poster/31076

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Real-world image restoration (IR) is inherently complex and often requires combining multiple specialized models to address diverse degradations. Inspired by human problem-solving, we propose AgenticIR, an agentic system that mimics the human approach to image processing by following five key stages: Perception, Scheduling, Execution, Reflection, and Rescheduling. AgenticIR leverages large language models (LLMs) and vision-language models (VLMs) that interact via text generation to dynamically operate a toolbox of IR models. We fine-tune VLMs for image quality analysis and employ LLMs for reasoning, guiding the system step by step. To compensate for LLMs' lack of specific IR knowledge and experience, we introduce a self-exploration method, allowing the LLM to observe and summarize restoration results into referenceable documents. Experiments demonstrate AgenticIR's potential in handling complex IR tasks, representing a promising path toward achieving general intelligence in visual processing.

</details>

---

## 247. NL-Eye: Abductive NLI For Images

- [ ] NL-Eye: Abductive NLI For Images | https://iclr.cc/virtual/2025/poster/31098

- **Link**: https://iclr.cc/virtual/2025/poster/31098

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Will a Visual Language Model (VLM)-based bot warn us about slipping if it detects a wet floor? Recent VLMs have demonstrated impressive capabilities, yet their ability to infer outcomes and causes remains underexplored. To address this, we introduce NL-Eye, a benchmark designed to assess VLMs' visual abductive reasoning skills. NL-Eye adapts the abductive Natural Language Inference (NLI) task to the visual domain, requiring models to evaluate the plausibility of hypothesis images based on a premise image and explain their decisions. NL-Eye consists of 350 carefully curated triplet examples (1,050 images) spanning diverse reasoning categories: physical, functional, logical, emotional, cultural, and social. The data curation process involved two steps—writing textual descriptions and generating images using text-to-image models, both requiring substantial human involvement to ensure high-quality and challenging scenes. Our experiments show that VLMs struggle significantly on NL-Eye, often performing at random baseline levels, while humans excel in both plausibility prediction and explanation quality. This demonstrates a deficiency in the abductive reasoning capabilities of modern VLMs. NL-Eye represents a crucial step toward developing VLMs capable of robust multimodal reasoning for real-world applications, including accident-prevention bots and generated video verification.

</details>

---

## 248. MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks

- [ ] MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks | https://iclr.cc/virtual/2025/poster/31110

- **Link**: https://iclr.cc/virtual/2025/poster/31110

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present MEGA-Bench, an evaluation suite that scales multimodal evaluation to over 500 real-world tasks, to address the highly heterogeneous daily use cases of end users.Our objective is to optimize for a set of high-quality data samples that cover a highly diverse and rich set of multimodal tasks, while enabling cost-effective and accurate model evaluation.In particular, we collected 505 realistic tasks encompassing over 8,000 samples from 16 expert annotators to extensively cover the multimodal task space. Instead of unifying these problems into standard multi-choice questions (like MMMU, MM-Bench, and MMT-Bench), we embrace a wide range of output formats like numbers, phrases, code, \LaTeX, coordinates, JSON, free-form, etc. To accommodate these formats, we developed over 40 metrics to evaluate these tasks. Unlike existing benchmarks, MEGA-Bench offers a fine-grained capability report across multiple dimensions (e.g., application, input type, output format, skill), allowing users to interact with and visualize model capabilities in depth. We evaluate a wide variety of frontier vision-language models on MEGA-Bench to understand their capabilities across these dimensions.

</details>

---

## 249. Language-Assisted Feature Transformation for Anomaly Detection

- [ ] Language-Assisted Feature Transformation for Anomaly Detection | https://iclr.cc/virtual/2025/poster/31115

- **Link**: https://iclr.cc/virtual/2025/poster/31115

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces LAFT, a novel feature transformation method designed to incorporate user knowledge and preferences into anomaly detection using natural language. Accurately modeling the boundary of normality is crucial for distinguishing abnormal data, but this is often challenging due to limited data or the presence of nuisance attributes. While unsupervised methods that rely solely on data without user guidance are common, they may fail to detect anomalies of specific interest. To address this limitation, we propose Language-Assisted Feature Transformation (LAFT), which leverages the shared image-text embedding space of vision-language models to transform visual features according to user-defined requirements. Combined with anomaly detection methods, LAFT effectively aligns visual features with user preferences, allowing anomalies of interest to be detected. Extensive experiments on both toy and real-world datasets validate the effectiveness of our method.

</details>

---

## 250. ZIP: An Efficient Zeroth-order Prompt Tuning for Black-box Vision-Language Models

- [ ] ZIP: An Efficient Zeroth-order Prompt Tuning for Black-box Vision-Language Models | https://iclr.cc/virtual/2025/poster/31142

- **Link**: https://iclr.cc/virtual/2025/poster/31142

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have introduced various approaches for prompt-tuning black-box vision-language models, referred to as black-box prompt-tuning (BBPT). While BBPT has demonstrated considerable potential, it is often found that many existing methods require an excessive number of queries (i.e., function evaluations), which poses a significant challenge in real-world scenarios where the number of allowed queries is limited. To tackle this issue, we propose Zeroth-order Intrinsic-dimensional Prompt-tuning (ZIP), a novel approach that enables efficient and robust prompt optimization in a purely black-box setting. The key idea of ZIP is to reduce the problem dimensionality and the variance of zeroth-order gradient estimates, such that the training is done fast with far less queries. We achieve this by re-parameterizing prompts in low-rank representations and designing intrinsic-dimensional clipping of estimated gradients. We evaluate ZIP on 13+ vision-language tasks in standard benchmarks and show that it achieves an average improvement of approximately 6% in few-shot accuracy and 48% in query efficiency compared to the best-performing alternative BBPT methods, establishing a new state of the art. Our ablation analysis further shows that the proposed clipping mechanism is robust and nearly optimal, without the need to manually select the clipping threshold, matching the result of expensive hyperparameter search.

</details>

---

## 251. Open-Vocabulary Customization from CLIP via Data-Free Knowledge Distillation

- [ ] Open-Vocabulary Customization from CLIP via Data-Free Knowledge Distillation | https://iclr.cc/virtual/2025/poster/31193

- **Link**: https://iclr.cc/virtual/2025/poster/31193

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models such as CLIP have demonstrated strong zero-shot performance, but their considerable size and inefficient inference limit customizable deployment for users. While knowledge distillation is a solution, it still requires the original data, which is not always available due to copyrights and privacy concerns. For many users seeking open-vocabulary customization, Data-Free Knowledge Distillation (DFKD) emerges as a promising direction. Upon rethinking DFKD, we find that existing methods fail on CLIP due to their heavy reliance on BatchNorm layers, which are unexpectedly unusable in CLIP. Based on our findings, we adopt image-text matching to achieve DFKD for CLIP, enabling customization based on arbitrary class texts. This involves (i) inversing a surrogate dataset from CLIP based on text prompts; and (ii) distilling a student model from CLIP using the surrogate dataset. Specifically, we introduce style dictionary diversification to enhance the diversity of synthetic images. To prevent uncontrollable semantics introduced by diversification, we propose a class consistency maintaining strategy to ensure the consistency of synthetic images. Based on synthetic images with various styles, we further propose meta knowledge distillation to train the student model with good generalization ability. Moreover, we introduce a simple yet effective method to enable customization based on few example images. Comprehensive experiments showcase the superiority of our approach across twelve customized tasks, achieving a 9.33\% improvement compared to existing DFKD methods.

</details>

---

## 252. Feast Your Eyes:  Mixture-of-Resolution Adaptation for Multimodal Large Language Models

- [ ] Feast Your Eyes:  Mixture-of-Resolution Adaptation for Multimodal Large Language Models | https://iclr.cc/virtual/2025/poster/31217

- **Link**: https://iclr.cc/virtual/2025/poster/31217

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In existing multimodal large language models (MLLMs), image resolution plays a significant role for  granular visual recognition.  However, directly increasing image resolution leads to expensive computational cost for MLLMs.  In this paper, we  reveal that a combination of low- and high-resolution visual features can efficiently mitigate this shortcoming.  Based on this principle, we propose a novel and efficient method for MLLMs, termed Mixture-of-Resolution Adaptation (MRA). In particular, MRA adopts two visual pathways for  images of different resolutions, where  high-resolution visual information is embedded into the low-resolution pathway via the novel mixture-of-resolution adapters (MR-Adapters). This design also   greatly  reduces the input sequence length of MLLMs. To validate MRA, we apply it to a recent MLLM called LLaVA, and term the new model LLaVA-HR. We conduct extensive  experiments on 17 vision-language (VL) tasks, which show that LLaVA-HR outperforms existing MLLMs on 15 VL tasks, e.g., +5.2\% on TextVQA.  More importantly,    both training and inference  of LLaVA-HR remain efficient with MRA, e.g.,  20 training hours and  faster inference speed  than LLaVA-NeXT.  Source codes are  released at: https://github.com/luogen1996/LLaVA-HR.

</details>

---

## 253. Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset

- [ ] Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset | https://iclr.cc/virtual/2025/poster/31229

- **Link**: https://iclr.cc/virtual/2025/poster/31229

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Machine unlearning has emerged as an effective strategy for forgetting specific information in the training data. However, with the increasing integration of visual data, privacy concerns in Vision Language Models (VLMs) remain underexplored. To address this, we introduce Facial Identity Unlearning Benchmark (FIUBench), a novel VLM unlearning benchmark designed to robustly evaluate the effectiveness of unlearning algorithms under the Right to be Forgotten setting. Specifically, we formulate the VLM unlearning task via constructing the Fictitious Facial Identity VQA dataset and apply a two-stage evaluation pipeline that is designed to precisely control the sources of information and their exposure levels. In terms of evaluation, since VLM supports various forms of ways to ask questions with the same semantic meaning,  we also provide robust evaluation metrics including membership inference attacks and carefully designed adversarial privacy attacks to evaluate the performance of algorithms.  Through the evaluation of four baseline VLM unlearning algorithms within FIUBench, we find that all methods remain limited in their unlearning performance, with significant trade-offs between model utility and forget quality. Furthermore, our findings also highlight the importance of privacy attacks for robust evaluations. We hope FIUBench will drive progress in developing more effective VLM unlearning algorithms.

</details>

---

## 254. Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage

- [ ] Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage | https://iclr.cc/virtual/2025/poster/31249

- **Link**: https://iclr.cc/virtual/2025/poster/31249

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of large language models (LLMs) prompts the development of multi-modal agents, which are used as a controller to call external tools, providing a feasible way to solve practical tasks. In this paper, we propose a multi-modal agent tuning method that automatically generates multi-modal tool-usage data and tunes a vision-language model (VLM) as the controller for powerful tool-usage reasoning. To preserve the data quality, we prompt the GPT-4o mini model to generate queries, files, and trajectories, followed by query-file and trajectory verifiers. Based on the data synthesis pipeline, we collect the MM-Traj dataset that contains 20K tasks with trajectories of tool usage. Then, we develop the T3-Agent via Trajectory Tuning on VLMs for Tool usage using MM-Traj. Evaluations on the GTA and GAIA benchmarks show that the T3-Agent consistently achieves improvements on two popular VLMs: MiniCPM-V-8.5B and Qwen2-VL-7B, which outperforms untrained VLMs by 20%, showing the effectiveness of the proposed data synthesis pipeline, leading to high-quality data for tool-usage capabilities.

</details>

---

## 255. VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation

- [ ] VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation | https://iclr.cc/virtual/2025/poster/31274

- **Link**: https://iclr.cc/virtual/2025/poster/31274

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

VILA-U is a Unified foundation model that integrates Video, Image, Language understanding and generation. Traditional visual language models (VLMs) use separate modules for understanding and generating visual content, which can lead to misalignment and increased complexity. In contrast, VILA-U employs a single autoregressive next-token prediction framework for both tasks, eliminating the need for additional components like diffusion models. This approach not only simplifies the model but also achieves near state-of-the-art performance in visual language understanding and generation. The success of VILA-U is attributed to two main factors: the unified vision tower that aligns discrete visual tokens with textual inputs during pretraining, which enhances visual perception, and autoregressive image generation can achieve similar quality as diffusion models with high-quality dataset. This allows VILA-U to perform comparably to more complex models using a fully token-based autoregressive framework.

</details>

---

## 256. Mechanistic Interpretability Meets Vision Language Models: Insights and Limitations

- [ ] Mechanistic Interpretability Meets Vision Language Models: Insights and Limitations | https://iclr.cc/virtual/2025/poster/31330

- **Link**: https://iclr.cc/virtual/2025/poster/31330

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision language models (VLMs), such as GPT-4o, have rapidly evolved, demonstrating impressive capabilities across diverse tasks. However, much of the progress in this field has been driven by engineering efforts, with a limited understanding of how these models work. The lack of scientific insight poses challenges to further enhancing their robustness, generalization, and interpretability, especially in high-stakes settings. In this work, we systematically review the use of mechanistic interpretability methods to foster a more scientific and transparent understanding of VLMs. Specifically, we examine five prominent techniques: probing, activation patching, logit lens, sparse autoencoders, and automated explanation. We summarize the key insights these methods provide into how VLMs process information and make decisions. We also discuss critical challenges and limitations that must be addressed to further advance the field.

</details>

---

## 257. AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models

- [ ] AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models | https://iclr.cc/virtual/2025/poster/31476

- **Link**: https://iclr.cc/virtual/2025/poster/31476

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Classifiers built upon vision-language models such as CLIP have shown remarkable zero-shot performance across a broad range of image classification tasks. Prior work has studied different ways of automatically creating descriptor sets for every class based on prompt templates, ranging from manually engineered templates over templates obtained from a large language model to templates built from random words and characters. Up until now, deriving zero-shot classifiers from the respective encoded class descriptors has remained nearly unchanged, i.e., classify to the class that maximizes cosine similarity between its averaged encoded class descriptors and the image encoding. However, weighing all class descriptors equally can be suboptimal when certain descriptors match visual clues on a given image better than others. In this work, we propose AutoCLIP, a method for auto-tuning zero-shot classifiers. AutoCLIP tunes per-image weights to each prompt template at inference time, based on statistics of class descriptor-image similarities. AutoCLIP is fully unsupervised, has only a minor additional computation overhead, and can be easily implemented in few lines of code. We show that AutoCLIP outperforms baselines across a broad range of vision-language models, datasets, and prompt templates consistently and by up to 3 percent point accuracy.

</details>

---

## 258. Enhancing Vision-Language Model with Unmasked Token Alignment

- [ ] Enhancing Vision-Language Model with Unmasked Token Alignment | https://iclr.cc/virtual/2025/poster/31500

- **Link**: https://iclr.cc/virtual/2025/poster/31500

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive pre-training on image-text pairs, exemplified by CLIP, becomes a standard technique for learning multi-modal visual-language representations. Although CLIP has demonstrated remarkable performance, training it from scratch on noisy web-scale datasets is computationally demanding. On the other hand, mask-then-predict pre-training approaches, like Masked Image Modeling (MIM), offer efficient self-supervised learning for single-modal representations. This paper introduces $\textbf{U}$nmasked $\textbf{T}$oken $\textbf{A}$lignment ($\textbf{UTA}$), a method that leverages existing CLIP models to further enhance its vision-language representations. UTA trains a Vision Transformer (ViT) by aligning unmasked visual tokens to the corresponding image tokens from a frozen CLIP vision encoder, which automatically aligns the ViT model with the CLIP text encoder. The pre-trained ViT can be directly applied for zero-shot evaluation even without training on image-text pairs. Compared to MIM approaches, UTA does not suffer from training-finetuning inconsistency and is much more training-efficient by avoiding using the extra $\mathrm{[MASK]}$ tokens. Extensive experimental results demonstrate that UTA can enhance CLIP models and outperform existing MIM methods on various uni- and multi-modal benchmarks.

</details>

---

## 259. Exposing and Addressing Cross-Task Inconsistency in Unified Vision-Language Models

- [ ] Exposing and Addressing Cross-Task Inconsistency in Unified Vision-Language Models | https://iclr.cc/virtual/2025/poster/31504

- **Link**: https://iclr.cc/virtual/2025/poster/31504

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As general purpose vision models get increasingly effective at a wide set of tasks, it is imperative that they be consistent across the tasks they support. Inconsistent AI models are considered brittle and untrustworthy by human users and are more challenging to incorporate into larger systems that take dependencies on their outputs. Measuring consistency between very heterogeneous tasks that might include outputs in different modalities is challenging since it is difficult to determine if the predictions are consistent with one another. As a solution, we introduce a benchmark dataset, CocoCON, where we create contrast sets by modifying test instances for multiple tasks in small but semantically meaningful ways to change the gold label and outline metrics for measuring if a model is consistent by ranking the original and perturbed instances across tasks. We find that state-of-the-art vision-language models suffer from a surprisingly high degree of inconsistent behavior across tasks, especially for more heterogeneous tasks. To alleviate this issue, we propose a rank correlation-based auxiliary training objective, computed over large automatically created cross-task contrast sets, that improves the multi-task consistency of large unified models while retaining their original accuracy on downstream tasks.

</details>

---

## 260. $F^3Set$: Towards Analyzing Fast, Frequent, and Fine-grained Events from Videos

- [ ] $F^3Set$: Towards Analyzing Fast, Frequent, and Fine-grained Events from Videos | https://iclr.cc/virtual/2025/poster/32051

- **Link**: https://iclr.cc/virtual/2025/poster/32051

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Analyzing Fast, Frequent, and Fine-grained ($F^3$) events presents a significant challenge in video analytics and multi-modal LLMs. Current methods struggle to identify events that satisfy all the $F^3$ criteria with high accuracy due to challenges such as motion blur and subtle visual discrepancies. To advance research in video understanding, we introduce $F^3Set$, a benchmark that consists of video datasets for precise $F^3$ event detection. Datasets in $F^3Set$ are characterized by their extensive scale and comprehensive detail, usually encompassing over 1,000 event types with precise timestamps and supporting multi-level granularity. Currently, $F^3Set$ contains several sports datasets, and this framework may be extended to other applications as well. We evaluated popular temporal action understanding methods on $F^3Set$, revealing substantial challenges for existing techniques. Additionally, we propose a new method, $F^3ED$, for $F^3$ event detections, achieving superior performance. The dataset, model, and benchmark code are available at https://github.com/F3Set/F3Set.

</details>

---

## 261. SafeWatch: An Efficient Safety-Policy Following Video Guardrail Model with Transparent Explanations

- [ ] SafeWatch: An Efficient Safety-Policy Following Video Guardrail Model with Transparent Explanations | https://iclr.cc/virtual/2025/poster/32048

- **Link**: https://iclr.cc/virtual/2025/poster/32048

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rise of generative AI and rapid growth of high-quality video generation, video guardrails have become more crucial than ever to ensure safety and security across platforms. Current video guardrails, however, are either overly simplistic, relying on pure classification models trained on simple policies with limited unsafe categories, which lack detailed explanations, or prompting multimodal large language models (MLLMs) with long safety guidelines, which are inefficient and impractical for guardrailing real-world content. To bridge this gap, we propose SafeWatch, an efficient MLLM-based video guardrail model designed to follow customized safety policies and provide multi-label video guardrail outputs with content-specific explanations in a zero-shot manner. In particular, unlike traditional MLLM-based guardrails that encode all safety policies autoregressively, causing inefficiency and bias, SafeWatch uniquely encodes each policy chunk in parallel and eliminates their position bias such that all policies are attended simultaneously with equal importance. In addition, to improve efficiency and accuracy, SafeWatch incorporates a policy-aware visual token pruning algorithm that adaptively selects the most relevant video tokens for each policy, discarding noisy or irrelevant information. This allows for more focused, policy-compliant guardrail with significantly reduced computational overhead. Considering the limitations of existing video guardrail benchmarks, we propose SafeWatch-Bench, a large-scale video guardrail benchmark comprising over 2M videos spanning six safety categories which covers over 30 tasks to ensure a comprehensive coverage of all potential safety scenarios. We have conducted extensive experiments, showing that SafeWatch outperforms all SOTA video guardrails on SafeWatch-Bench by 28.2%, and achieves a 13.6% improvement on existing benchmarks, all while reducing inference costs by an average of 10%. SafeWatch also demonstrates strong policy-following abilities and outperforms previous SOTAs by 5.6% and 15.6% in zero-shot generalizability to new policies and new prompting tasks. Additionally, both LLM-as-a-judge and human evaluators confirm the high quality of the explanations provided by SafeWatch. Our project is open-sourced at https://safewatch-aiguard.github.io.

</details>

---

## 262. Personalized Visual Instruction Tuning

- [ ] Personalized Visual Instruction Tuning | https://iclr.cc/virtual/2025/poster/32055

- **Link**: https://iclr.cc/virtual/2025/poster/32055

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have demonstrated significant progress; however, these models exhibit a notable limitation, which we refer to as "face blindness." Specifically, they can engage in general conversations but fail to conduct personalized dialogues targeting at specific individuals. This deficiency hinders the application of MLLMs in personalized settings, such as tailored visual assistants on mobile devices, or domestic robots that need to recognize members of the family. In this paper, we introduce Personalized Visual Instruction Tuning (PVIT), a novel data curation and training framework designed to enable MLLMs to identify target individuals within an image and engage in personalized and coherent dialogues. Our approach involves the development of a sophisticated pipeline that autonomously generates training data containing personalized conversations. This pipeline leverages the capabilities of various visual experts, image generation models, and (multi-modal) large language models. To evaluate the personalized potential of MLLMs, we present a benchmark called P-Bench, which encompasses various question types with different levels of difficulty. The experiments demonstrate a substantial personalized performance enhancement after fine-tuning with our curated dataset.

</details>

---

## 263. Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents

- [ ] Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents | https://iclr.cc/virtual/2025/poster/32062

- **Link**: https://iclr.cc/virtual/2025/poster/32062

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are transforming the capabilities of graphical user interface (GUI) agents, facilitating their transition from controlled simulations to complex, real-world applications across various platforms. However, the effectiveness of these agents hinges on the robustness of their grounding capability. Current GUI agents predominantly utilize text-based representations such as HTML or accessibility trees, which, despite their utility, often introduce noise, incompleteness, and increased computational overhead. In this paper, we advocate a human-like embodiment for GUI agents that perceive the environment entirely visually and directly perform pixel-level operations on the GUI. The key is visual grounding models that can accurately map diverse referring expressions of GUI elements to their coordinates on the GUI across different platforms. We show that a simple recipe, which includes web-based synthetic data and slight adaptation of the LLaVA architecture, is surprisingly effective for training such visual grounding models. We collect the largest dataset for GUI visual grounding so far, containing 10M GUI elements and their referring expressions over 1.3M screenshots, and use it to train UGround, a strong universal visual grounding model for GUI agents. Empirical results on six benchmarks spanning three categories (grounding, offline agent, and online agent) show that 1) UGround substantially outperforms existing visual grounding models for GUI agents, by up to 20\% absolute, and 2) agents with UGround outperform state-of-the-art agents, despite the fact that existing agents use additional text-based input while ours only uses visual perception. These results provide strong support for the feasibility and promises of GUI agents that navigate the digital world as humans do.

</details>

---

## 264. OmniCorpus: A Unified Multimodal Corpus of 10 Billion-Level Images Interleaved with Text

- [ ] OmniCorpus: A Unified Multimodal Corpus of 10 Billion-Level Images Interleaved with Text | https://iclr.cc/virtual/2025/poster/32063

- **Link**: https://iclr.cc/virtual/2025/poster/32063

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image-text interleaved data, consisting of multiple images and texts arranged in a natural document format, aligns with the presentation paradigm of internet data and closely resembles human reading habits. Recent studies have shown that such data aids multimodal in-context learning and maintains the capabilities of large language models during multimodal fine-tuning. However, the limited scale and diversity of current image-text interleaved data restrict the development of multimodal large language models. In this paper, we introduce OmniCorpus, a 10 billion-scale image-text interleaved dataset. Using an efficient data engine, we filter and extract large-scale high-quality documents, which contain 8.6 billion images and 1,696 billion text tokens. Compared to counterparts (e.g., MMC4, OBELICS), our dataset 1) has 15 times larger scales while maintaining good data quality; 2) features more diverse sources, including both English and non-English websites as well as video-centric websites; 3) is more flexible, easily degradable from an image-text interleaved format to pure text corpus and image-text pairs. Through comprehensive analysis and experiments, we validate the quality, usability, and effectiveness of the proposed dataset. We hope this could provide a solid data foundation for future multimodal model research.

</details>

---

## 265. BadRobot: Jailbreaking Embodied LLM Agents in the Physical World

- [ ] BadRobot: Jailbreaking Embodied LLM Agents in the Physical World | https://iclr.cc/virtual/2025/poster/32071

- **Link**: https://iclr.cc/virtual/2025/poster/32071

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Embodied AI represents systems where AI is integrated into physical entities. Multimodal Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, the first attack paradigm designed to jailbreak robotic manipulation, making embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. We emphasize that addressing this emerging vulnerability is crucial for the secure deployment of LLMs in robotics.Warning: This paper contains harmful AI-generated language and aggressive actions.

</details>

---

## 266. Interpretable Bilingual Multimodal Large Language Model for Diverse Biomedical Tasks

- [ ] Interpretable Bilingual Multimodal Large Language Model for Diverse Biomedical Tasks | https://iclr.cc/virtual/2025/poster/32076

- **Link**: https://iclr.cc/virtual/2025/poster/32076

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Several medical Multimodal Large Languange Models (MLLMs) have been developed to address tasks involving visual images with textual instructions across various medical modalities, achieving impressive results. Most current medical generalist models are region-agnostic, treating the entire image as a holistic representation. However, they struggle to identify which specific regions they are focusing on when generating a sentence.To mimic the behavior of doctors, who typically begin by reviewing the entire image before concentrating on specific regions for a thorough evaluation, we aim to enhance the capability of medical MLLMs in understanding anatomical regions within entire medical scans.To achieve it, we first formulate \textbf{Region-Centric tasks} and construct a \textbf{large-scale dataset, MedRegInstruct,} to incorporate regional information into training. Combining our collected dataset with other medical multimodal corpora for training, we propose a \textbf{Region-Aware medical MLLM, MedRegA}, which is the first bilingual generalist medical AI system to simultaneously handle image-level and region-level medical vision-language tasks across a broad range of modalities. Our MedRegA not only enables three region-centric tasks, but also achieves the best performance for visual question answering, report generation and medical image classification over 8 modalities, showcasing significant versatility. Experiments demonstrate that our model can not only accomplish powerful performance across various medical vision-language tasks in bilingual settings, but also recognize and detect structures in multimodal medical scans, boosting the interpretability and user interactivity of medical MLLMs. The codes and model will be made publicly available.

</details>

---

## 267. Measuring And Improving Engagement of Text-to-Image Generation Models

- [ ] Measuring And Improving Engagement of Text-to-Image Generation Models | https://iclr.cc/virtual/2025/poster/32082

- **Link**: https://iclr.cc/virtual/2025/poster/32082

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in text-to-image generation have achieved impressive aesthetic quality, making these models usable for both personal and commercial purposes. However, in the fields of marketing and advertising, images are often created to be more engaging, as reflected in user behaviors such as increasing clicks, likes, and purchases, in addition to being aesthetically pleasing. To this end, we introduce the challenge of optimizing the image generation process for improved viewer engagement. In order to study image engagement and utility in real-world marketing scenarios, we collect EngagingImageNet , the first large-scale dataset of images, along with associated user engagement metrics. Further, we find that existing image evaluation metrics like aesthetics, CLIPScore, PickScore, ImageReward, etc. are unable to capture viewer engagement. To address the lack of reliable metrics for assessing image utility, we use the EngagingImageNet dataset to train EngageNet , an engagement-aware Vision Language Model (VLM) that predicts viewer engagement of images by leveraging contextual information about the tweet content, enterprise details, and posting time. We then explore methods to enhance the engagement of text-to-image models, making initial strides in this direction. These include conditioning image generation on improved prompts, supervised fine-tuning of stable diffusion on high-performing images, and reinforcement learning to align stable diffusion with EngageNet -based reward signals, all of which lead to the generation of images with higher viewer engagement. Finally, we propose the Engagement Arena , to benchmark text-to-image models based on their ability to generate engaging images, using EngageNet as the evaluator, thereby encouraging the research community to measure further advances in the engagement of text-to-image modeling. These contributions provide a new pathway for advancing utility-driven image generation, with significant implications for the commercial application of image generation. We have released our code and dataset on behavior-in-the-wild.github.io/image-engagement .

</details>

---

## 268. GUI-World: A Video Benchmark and Dataset for Multimodal GUI-oriented Understanding

- [ ] GUI-World: A Video Benchmark and Dataset for Multimodal GUI-oriented Understanding | https://iclr.cc/virtual/2025/poster/32086

- **Link**: https://iclr.cc/virtual/2025/poster/32086

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Multimodal Large Language Models (MLLMs) have been used as agents to control keyboard and mouse inputs by directly perceiving the Graphical User Interface (GUI) and generating corresponding commands.However, current agents primarily demonstrate strong understanding capabilities in static environments and are mainly applied to relatively simple domains, such as Web or mobile interfaces.We argue that a robust GUI agent should be capable of perceiving temporal information on the GUI, including dynamic Web content and multi-step tasks.Additionally, it should possess a comprehensive understanding of various GUI scenarios, including desktop software and multi-window interactions.To this end, this paper introduces a new dataset, termed GUI-World, which features meticulously crafted Human-MLLM annotations, extensively covering six GUI scenarios and eight types of GUI-oriented questions in three formats.We evaluate the capabilities of current state-of-the-art MLLMs, including Image LLMs and Video LLMs, in understanding various types of GUI content, especially dynamic and sequential content. Our findings reveal that current models struggle with dynamic GUI content without manually annotated keyframes or operation history. On the other hand, Video LLMs fall short in all GUI-oriented tasks given the sparse GUI video dataset. Therefore, we take the initial step of leveraging a fine-tuned Video LLM, GUI-Vid, as a GUI-oriented assistant, demonstrating an improved understanding of various GUI tasks. However, due to the limitations in the performance of base LLMs, we conclude that using video LLMs as GUI agents remains a significant challenge. We believe our work provides valuable insights for future research in dynamic GUI content understanding. All the dataset and code are publicly available at: https://gui-world.github.io.

</details>

---

## 269. Unleashing the Potential of Vision-Language Pre-Training for 3D Zero-Shot Lesion Segmentation via Mask-Attribute Alignment

- [ ] Unleashing the Potential of Vision-Language Pre-Training for 3D Zero-Shot Lesion Segmentation via Mask-Attribute Alignment | https://iclr.cc/virtual/2025/poster/32087

- **Link**: https://iclr.cc/virtual/2025/poster/32087

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in medical vision-language pre-training models have driven significant progress in zero-shot disease recognition. However, transferring image-level knowledge to pixel-level tasks, such as lesion segmentation in 3D CT scans, remains a critical challenge. Due to the complexity and variability of pathological visual characteristics, existing methods struggle to align fine-grained lesion features not encountered during training with disease-related textual representations. In this paper, we present Malenia, a novel multi-scale lesion-level mask-attribute alignment framework, specifically designed for 3D zero-shot lesion segmentation. Malenia improves the compatibility between mask representations and their associated elemental attributes, explicitly linking the visual features of unseen lesions with the extensible knowledge learned from previously seen ones. Furthermore, we design a Cross-Modal Knowledge Injection module to enhance both visual and textual features with mutually beneficial information, effectively guiding the generation of segmentation results. Comprehensive experiments across three datasets and 12 lesion categories validate the superior performance of Malenia.

</details>

---

## 270. PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding

- [ ] PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding | https://iclr.cc/virtual/2025/poster/32088

- **Link**: https://iclr.cc/virtual/2025/poster/32088

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding the physical world is a fundamental challenge in embodied AI, critical for enabling agents to perform complex tasks and operate safely in real-world environments. While Vision-Language Models (VLMs) have shown great promise in reasoning and task planning for embodied agents, their ability to comprehend physical phenomena remains extremely limited.To close this gap, we introduce PhysBench, a comprehensive benchmark designed to evaluate VLMs' physical world understanding capability across a diverse set of tasks. PhysBench contains 10,002 entries of interleaved video-image-text data, categorized into four major domains: physical object properties, physical object relationships, physical scene understanding, and physics-based dynamics, further divided into 19 subclasses and 8 distinct capability dimensions.Our extensive experiments, conducted on 75 representative VLMs, reveal that while these models excel in common-sense reasoning, they struggle with understanding the physical world---likely due to the absence of physical knowledge in their training data and the lack of embedded physical priors.To tackle the shortfall, we introduce PhysAgent, a novel framework that combines the generalization strengths of VLMs with the specialized expertise of vision models, significantly enhancing VLMs' physical understanding across a variety of tasks, including an 18.4\% improvement on GPT-4o.Furthermore, our results demonstrate that enhancing VLMs' physical world understanding capabilities can help embodied agents such as MOKA.We believe that PhysBench and PhysAgent offer valuable insights and contribute to bridging the gap between VLMs and physical world understanding. Project Page is here

</details>

---

## 271. Grounding Multimodal Large Language Model in GUI World

- [ ] Grounding Multimodal Large Language Model in GUI World | https://iclr.cc/virtual/2025/poster/32092

- **Link**: https://iclr.cc/virtual/2025/poster/32092

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have accelerated the development of Graphical User Interface (GUI) agents capable of automating complex tasks across digital platforms. However, precise GUI element grounding remains a key challenge for accurate interaction and generalization. In this work, we present an effective GUI grounding framework, which includes an automated data collection engine that gathers extensive GUI screenshots and annotations to ensure broad generalization. We also propose a lightweight and flexible GUI grounding module designed to efficiently localize UI elements by pre-training on the collected data, and introduce a novel method to integrate this module with MLLMs for the effective execution of GUI tasks. Our approach demonstrates superior performance in task accuracy and adaptability, as validated by benchmarks such as ScreenSpot, MiniWob, AITW, and Mind2Web.

</details>

---

## 272. IDA-VLM: Towards Movie Understanding via ID-Aware Large Vision-Language Model

- [ ] IDA-VLM: Towards Movie Understanding via ID-Aware Large Vision-Language Model | https://iclr.cc/virtual/2025/poster/32091

- **Link**: https://iclr.cc/virtual/2025/poster/32091

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of Large Vision-Language models (LVLMs) has demonstrated a spectrum of emergent capabilities. Nevertheless, current models only focus on the visual content of a single scenario, while their ability to associate instances across different scenes has not yet been explored, which is essential for understanding complex visual content, such as movies with multiple characters and intricate plots. Towards movie understanding, a critical initial step for LVLMs is to unleash the potential of character identities memory and recognition across multiple visual scenarios. To achieve the goal, we propose visual instruction tuning with ID reference and develop an ID-Aware Large Vision-Language Model, IDA-VLM. Furthermore, our research introduces a novel benchmark MM-ID, to examine LVLMs on instance IDs memory and recognition across four dimensions: matching, location, question-answering, and captioning. Our findings highlight the limitations of existing LVLMs in recognizing and associating instance identities with ID reference. This paper paves the way for future artificial intelligence systems to possess multi-identity visual inputs, thereby facilitating the comprehension of complex visual narratives like movies.

</details>

---

## 273. Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms

- [ ] Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms | https://iclr.cc/virtual/2025/poster/32099

- **Link**: https://iclr.cc/virtual/2025/poster/32099

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Building a generalist model for user interface (UI) understanding is challenging due to various foundational issues, such as platform diversity, resolution variation, and data limitation. In this paper, we introduce Ferret-UI 2, a multimodal large language model (MLLM) designed for universal UI understanding across a wide range of platforms, including iPhone, Android, iPad, Webpage, and AppleTV. Building on the foundation of Ferret-UI, Ferret-UI 2 introduces three key innovations: support for multiple platform types, high-resolution perception through adaptive scaling, and advanced task training data generation powered by GPT-4o with set-of-mark visual prompting. These advancements enable Ferret-UI 2 to perform complex, user-centered interactions, making it highly versatile and adaptable for the expanding diversity of platform ecosystems. Extensive empirical experiments on referring, grounding, user-centric advanced tasks (comprising 9 subtasks $\times$ 5 platforms), GUIDE next-action prediction dataset, and GUI-World multi-platform benchmark demonstrate that Ferret-UI 2 significantly outperforms Ferret-UI, and also shows strong cross-platform transfer capabilities.

</details>

---

## 274. TAU-106K: A New Dataset for Comprehensive Understanding of Traffic Accident

- [ ] TAU-106K: A New Dataset for Comprehensive Understanding of Traffic Accident | https://iclr.cc/virtual/2025/poster/32101

- **Link**: https://iclr.cc/virtual/2025/poster/32101

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated impressive performance in general visual understanding tasks. However, their potential for high-level, fine-grained comprehension, such as anomaly understanding, remains unexplored. Focusing on traffic accidents, a critical and practical scenario within anomaly understanding, we investigate the advanced capabilities of MLLMs and propose TABot, a multimodal MLLM specialized for accident-related tasks. To facilitate this, we first construct TAU-106K, a large-scale multimodal dataset containing 106K traffic accident videos and images collected from academic benchmarks and public platforms. The dataset is meticulously annotated through a video-to-image annotation pipeline to ensure comprehensive and high-quality labels. Building upon TAU-106K, we train TABot using a two-step approach designed to integrate multi-granularity tasks, including accident recognition, spatial-temporal grounding, and an auxiliary description task to enhance the model's understanding of accident elements. Extensive experiments demonstrate TABot's superior performance in traffic accident understanding, highlighting not only its capabilities in high-level anomaly comprehension but also the robustness of the TAU-106K benchmark. Our code and data will be available at https://github.com/cool-xuan/TABot.

</details>

---

## 275. SV-RAG: LoRA-Contextualizing Adaptation of  MLLMs for Long Document Understanding

- [ ] SV-RAG: LoRA-Contextualizing Adaptation of  MLLMs for Long Document Understanding | https://iclr.cc/virtual/2025/poster/32102

- **Link**: https://iclr.cc/virtual/2025/poster/32102

- **Conference**: ICLR

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have recently shown great progress in text-rich image understanding, yet they still struggle with complex, multi-page visually-rich documents. Traditional methods using document parsers for retrieval-augmented generation suffer from performance and efficiency limitations, while directly presenting all pages to MLLMs leads to inefficiencies, especially with lengthy ones.  In this work, we present a novel framework named S elf- V isual R etrieval- A ugmented G eneration (SV-RAG), which can broaden horizons of any MLLM to support long-document understanding. We demonstrate that MLLMs themselves can be an effective multimodal retriever to fetch relevant pages and then answer user questions based on these pages. SV-RAG is implemented with two specific MLLM adapters, one for evidence page retrieval and the other for question answering. Empirical results show state-of-the-art performance on public benchmarks, demonstrating the effectiveness of SV-RAG.

</details>

---

