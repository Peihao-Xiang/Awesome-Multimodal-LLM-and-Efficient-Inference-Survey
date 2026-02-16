# EMNLP 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_emnlp2025_papers.csv

## 1. AMCrawl: AnArabic Web-Scale Dataset of Interleaved Image-Text Documents and Image-Text Pairs

- [ ] AMCrawl: AnArabic Web-Scale Dataset of Interleaved Image-Text Documents and Image-Text Pairs | https://aclanthology.org/2025.arabicnlp-main.37/

- **Link**: https://aclanthology.org/2025.arabicnlp-main.37/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present the Arabic Multimodal Crawl (AMCrawl), the first native-based Arabic multimodal dataset to our knowledge, derived from the Common Crawl corpus and rigorously filtered for quality and safety. Image-text pair datasets are the standard choice for pretraining multimodal large language models. However, they are often derived from image alt-text metadata, which is typically brief and context-poor, disconnecting images from their broader meaning. Although significant advances have been made in building interleaved image-text datasets for English, such as the OBELICS dataset, a substantial gap remains for native Arabic content. Our processing covered 8.6 million Arabic web pages, yielding 5.8 million associated images and 1.3 billion text tokens. The final dataset includes interleaved image-text documents and question-answer pairs, featuring 2.8 million high-quality interleaved documents and 5 million QA pairs. Alongside the dataset, we release the complete pipeline and code, ensuring reproducibility and encouraging further research and development. To demonstrate the effectiveness of AMCrawl, we introduce a publicly available native Arabic Vision Language model, trained with 13 billion parameters. These models achieve competitive results when benchmarked against publicly available datasets. AMCrawl bridges a critical gap in Arabic multimodal resources, providing a robust foundation for developing Arabic multimodal large language models and fostering advancements in this underrepresented area. Code: github.com/shahad-aboukozzana/AMCrawl

</details>

---

## 2. A-SEA3𝐋-QA: A Fully Automated Self-Evolving, Adversarial Workflow forArabic Long-Context Question-Answer Generation

- [ ] A-SEA3𝐋-QA: A Fully Automated Self-Evolving, Adversarial Workflow forArabic Long-Context Question-Answer Generation | https://aclanthology.org/2025.arabicnlp-main.9/

- **Link**: https://aclanthology.org/2025.arabicnlp-main.9/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present an end-to-end, self-evolving adversarial workflow for long-context Question-Answer (QA) Generation in Arabic. By orchestrating multiple specialized LVLMs: a question generator, an evaluator, and a swarm of answer generators, our system iteratively refines its own performance without any human intervention. Starting from raw, multi-page Arabic documents across diverse domains, the question generator produces fine-grained, context-aware queries to be tackled by the answer generator swarm, and the evaluator assesses and feeds back quality metrics. This closed-loop cycle enables continuous learning: low-confidence outputs trigger automated re-generation and model updates, progressively enhancing question difficulty and relevance. Moreover, we set the quality metrics as a tunable hyperparameter, enabling question generation at controllable and customizable difficulty levels. We releaseAraLongBench, a large-scale Arabic benchmark of single- and multi-page challenges spanning hundreds of pages, and demonstrate that our self-evolving workflow substantially outperform static pipelines, markedly boosting the long-context comprehension capabilities of leading Arabic Large Vision Language Models (LVLMs). Lastly, we also meticulously architect a fully automated agentic workflow for long-context Arabic document collection.

</details>

---

## 3. BitMar: Low-Bit Multimodal Fusion with Episodic Memory for Edge Devices

- [ ] BitMar: Low-Bit Multimodal Fusion with Episodic Memory for Edge Devices | https://aclanthology.org/2025.babylm-main.11/

- **Link**: https://aclanthology.org/2025.babylm-main.11/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Cross-attention transformers and other multimodal vision-language models excel at grounding and generation; however, their extensive, full-precision backbones make it challenging to deploy them on edge devices. Memory-augmented architectures enhance the utilization of past context; however, most works rarely pair them with aggressive edge-oriented quantization. We introduce BitMar, a quantized multimodal transformer that proposes an external human-like episodic memory for effective image-text generation on hardware with limited resources. BitMar utilizes 1.58-bit encoders, one for text (BitNet-style) and one for vision (DiNOv2-based), to create compact embeddings that are combined and used to query a fixed-size key-value episodic memory. During vector retrieval, the BitNet decoder applies per‐layer conditioning, which increases the contextual relevance of generated content. The decoder also employs attention sinks with a sliding‐window mechanism to process long or streaming inputs under tight memory budgets. The combination of per-layer conditioning and sliding-window attention achieves a strong quality–speed trade–off, delivering competitive captioning and multimodal understanding at low latency with a small model footprint. These characteristics make BitMar well-suited for edge deployment.

</details>

---

## 4. Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling

- [ ] Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling | https://aclanthology.org/2025.babylm-main.15/

- **Link**: https://aclanthology.org/2025.babylm-main.15/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Training vision-language models on cognitively-plausible amounts of data requires rethinking how models integrate multimodal information. Within the constraints of the Vision track for the BabyLM Challenge 2025, we propose a lightweight decoder-based architecture with (1) token-wise dynamic gating for adaptive fusion of linguistic and visual cues, (2) feature modulation and channel attention to maximise the utility of limited visual information and (3) auxiliary contrastive objectives for visual grounding. Evaluation on five benchmarks (BLiMP, BLiMP Supplement, EWoK, Winoground and VQA) shows competitive or superior performance to multimodal baselines. More notably, our dynamic gate discovers interpretable patterns without explicit supervision, favouring visual cues for content words and linguistic cues for function words. While we identify limitations in the Challenge constraints, such as the information bottleneck created by global image embeddings and training instability from the dataset split, our findings establish dynamic gating as a powerful tool for efficient multimodal learning, offering both interpretability and performance even under severe constraints.

</details>

---

## 5. Model Merging to Maintain Language-Only Performance in Developmentally Plausible Multimodal Models

- [ ] Model Merging to Maintain Language-Only Performance in Developmentally Plausible Multimodal Models | https://aclanthology.org/2025.babylm-main.5/

- **Link**: https://aclanthology.org/2025.babylm-main.5/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

State-of-the-art vision-and-language models consist of many parameters and learn from enormous datasets, surpassing the amounts of linguistic data that children are exposed to as they acquire a language. This paper presents our approach to the multimodal track of the BabyLM challenge addressing this discrepancy. We develop language-only and multimodal models in low-resource settings using developmentally plausible datasets, with our multimodal models outperforming previous BabyLM baselines. One finding in the multimodal language model literature is that these models tend to underperform inlanguage-onlytasks. Therefore, we focus on maintaining language-only abilities in multimodal models. To this end, we experiment withmodel merging, where we fuse the parameters of multimodal models with those of language-only models using weighted linear interpolation. Our results corroborate the findings that multimodal models underperform in language-only benchmarks that focus on grammar, and model merging with text-only models can help alleviate this problem to some extent, while maintaining multimodal performance.

</details>

---

## 6. Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in MultilingualLLMs

- [ ] Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in MultilingualLLMs | https://aclanthology.org/2025.blackboxnlp-1.2/

- **Link**: https://aclanthology.org/2025.blackboxnlp-1.2/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We explore Cross-lingual Backdoor ATtacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare and high-occurring tokens serving as specific, effective triggers. Our findings reveal a critical vulnerability that affects the model’s architecture, leading to a concealed backdoor effect during the information flow. Our code and data are publicly available at https://github.com/himanshubeniwal/X-BAT.

</details>

---

## 7. OpenRLHF: A Ray-based Easy-to-use, Scalable and High-performanceRLHFFramework

- [ ] OpenRLHF: A Ray-based Easy-to-use, Scalable and High-performanceRLHFFramework | https://aclanthology.org/2025.emnlp-demos.48/

- **Link**: https://aclanthology.org/2025.emnlp-demos.48/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) fine-tuned via Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) significantly improve the alignment of human-AI values and further raise the upper bound of AI capabilities, particularly in reasoning-intensive, long-context Chain-of-Thought (long-CoT) tasks. However, existing RLHF (or RLVR) frameworks commonly face challenges such as inference bottlenecks and complexity barriers, restricting their accessibility for newcomers. To bridge this gap, we introduceOpenRLHF, a user-friendly, scalable, and easy-to-learn open-source RLHF framework built upon Ray, vLLM, DeepSpeed, and HuggingFace Transformers, featuring a simplified design, clear code structure, and comprehensive documentation to facilitate entry for researchers and practitioners. Experimental results show that OpenRLHF achieves superior training efficiency with speedups ranging from 1.22× to 1.68× across different model sizes compared to state-of-the-art frameworks, while requiring significantly fewer lines of code for implementation. OpenRLHF is publicly available athttps://github.com/OpenRLHF/OpenRLHF, and has already been adopted by leading institutions to accelerate RLHF research and learning.

</details>

---

## 8. PresentAgent: Multimodal Agent for Presentation Video Generation

- [ ] PresentAgent: Multimodal Agent for Presentation Video Generation | https://aclanthology.org/2025.emnlp-demos.58/

- **Link**: https://aclanthology.org/2025.emnlp-demos.58/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present PresentAgent, a multimodal agent that transforms long-form documents into narrated presentation videos. While existing approaches are limited to generating static slides or text summaries, our method advances beyond these limitations by producing fully synchronized visual and spoken content that closely mimics human-style presentations. To achieve this integration, PresentAgent employs a modular pipeline that systematically segments the input document, plans and renders slide-style visual frames, generates contextual spoken narration with large language models and Text-to-Speech models, and seamlessly composes the final video with precise audio-visual alignment. Given the complexity of evaluating such multimodal outputs, we introduce PresentEval, a unified assessment framework powered by Vision-Language Models that comprehensively scores videos across three critical dimensions: content fidelity, visual clarity, and audience comprehension through prompt-based evaluation. Our experimental validation on a curated dataset of 30 document–presentation pairs demonstrates that PresentAgent approaches human-level quality across all evaluation metrics. These results highlight the significant potential of controllable multimodal agents in transforming static textual materials into dynamic, effective, and accessible presentation formats.

</details>

---

## 9. From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models withVLM-Lens

- [ ] From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models withVLM-Lens | https://aclanthology.org/2025.emnlp-demos.68/

- **Link**: https://aclanthology.org/2025.emnlp-demos.68/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce VLM-Lens, a toolkit designed to enable systematic benchmarking, analysis, and interpretation of vision-language models (VLMs) by supporting the extraction of intermediate outputs from any layer during the forward pass of open-source VLMs. VLM-Lens provides a unified, YAML-configurable interface that abstracts away model-specific complexities and supports user-friendly operation across diverse VLMs. It currently supports 16 state-of-the-art base VLMs and their over 30 variants, and is extensible to accommodate new models without changing the core logic.The toolkit integrates easily with various interpretability and analysis methods. We demonstrate its usage with two simple analytical experiments, revealing systematic differences in the hidden representations of VLMs across layers and target concepts. VLM-Lens is released as an open-sourced project to accelerate community efforts in understanding and improving VLMs.

</details>

---

## 10. RCI: A Score for Evaluating Global and Local Reasoning in Multimodal Benchmarks

- [ ] RCI: A Score for Evaluating Global and Local Reasoning in Multimodal Benchmarks | https://aclanthology.org/2025.emnlp-industry.10/

- **Link**: https://aclanthology.org/2025.emnlp-industry.10/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have achieved impressive results on vision-language benchmarks, yet it remains unclear whether these benchmarks assess genuine global reasoning or allow success via localized visual cues. Existing evaluation methods do not explicitly measure this distinction, hindering effective dataset curation and real-world focused model development.We introduce Region Comprehension Index (RCI), the first model-based score to directly quantify a dataset’s reliance on global versus local visual information. RCI systematically compares reference-model performance on image patches versus full images, revealing if tasks require holistic image understanding or can be solved with partial or localized visual cues.When applying RCI to 13 widely used multimodal benchmarks, we observed that most of them favor localized reasoning and exhibit significant spatial biases, indicating potential risks in real-world applications. RCI equips researchers & practitioners with an actionable tool for diagnosing & mitigating these biases, enabling the construction of datasets and benchmarks to foster the development of robust, enterprise-ready multimodal systems.

</details>

---

## 11. Deploying TinyLVLMJudges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices

- [ ] Deploying TinyLVLMJudges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices | https://aclanthology.org/2025.emnlp-industry.134/

- **Link**: https://aclanthology.org/2025.emnlp-industry.134/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) with only 7B parameters have shown promise as automated judges in chart comprehension tasks. However, tiny models (<=2B parameters) still perform poorly as judges, limiting their real-world use in resource-constrained settings. To address this, we propose two approaches to ensure cost‐efficient evaluation: (i) multi-criteria prompting, which combines separate evaluation criteria into a single query, and (ii) domain‐adaptive transfer learning, in which we fine‐tune a 2B‐parameter VLM on synthetic judgments in a chart dataset to create theChartJudge. Experiments show that multi-criteria prompting exposes robustness gaps, which led to a huge drop in performance for 7B models, including specialized LVLM judges like LLaVA‐Critic. In addition, we find that our tiny LVLM (ChartJudge) can effectively transfer knowledge from one dataset to another to make it a more specialized model. Our fine-grained analysis across chart types and query complexities offers actionable insights into trade-offs between model size, prompt design, and transferability, enabling scalable, low-cost evaluation for chart reasoning tasks. Our code and the data will be made publicly available.

</details>

---

## 12. PCRI: Measuring Context Robustness in Multimodal Models for Enterprise Applications

- [ ] PCRI: Measuring Context Robustness in Multimodal Models for Enterprise Applications | https://aclanthology.org/2025.emnlp-industry.14/

- **Link**: https://aclanthology.org/2025.emnlp-industry.14/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The reliability of Multimodal Large Language Models (MLLMs) in real-world settings is often undermined by sensitivity to irrelevant or distracting visual context, an aspect not captured by existing evaluation metrics. We introduce the Patch Context Robustness Index (PCRI), the first systematic and interpretable score for quantifying MLLM robustness to variations in visual context granularity, measuring performance changes between localized image patches and full-image input.Applying PCRI to 19 state-of-the-art MLLMs across 15 vision-language benchmarks, we find that most leading models remain brittle to background noise, with only a few, such as InternVL2-26B and Qwen2VL-72B, demonstrating consistent robustness across tasks. PCRI analysis also highlights how different model architectures handle and integrate visual context, offering actionable diagnostic insight for both researchers and practitioners.PCRI enables rigorous comparison of context robustness, supporting principled model selection and guiding the development of future architectures and training strategies for robust, real-world deployment.

</details>

---

## 13. Generating Spatial Knowledge Graphs from Automotive Diagrams for Question Answering

- [ ] Generating Spatial Knowledge Graphs from Automotive Diagrams for Question Answering | https://aclanthology.org/2025.emnlp-industry.157/

- **Link**: https://aclanthology.org/2025.emnlp-industry.157/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Answering “Where is the X button?” with “It’s next to the Y button” is unhelpful if the user knows neither location. Useful answers require obvious landmarks as a reference point. We address this by generating from a vehicle dashboard diagram a spatial knowledge graph (SKG) that shows the spatial relationship between a dashboard component and its nearby landmarks and using the SKG to help answer questions. We evaluate three distinct generation pipelines (Per-Attribute, Per-Component, and a Single-Shot baseline) to create the SKG using Large Vision-Language Models (LVLMs). On a new 65-vehicle dataset, we demonstrate that a decomposed Per-Component pipeline is the most effective strategy for generating a high-quality SKG; the graph produced by this method, when evaluated with a novel Significance score, identifies landmarks achieving 71.3% agreement with human annotators. This work enables downstream QA systems to provide more intuitive, landmark-based answers.

</details>

---

## 14. PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models

- [ ] PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models | https://aclanthology.org/2025.emnlp-industry.169/

- **Link**: https://aclanthology.org/2025.emnlp-industry.169/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the field of urban planning, existing Vision-Language Models (VLMs) frequently fail to effectively analyze planning maps, which are critical for urban planners and educational contexts. Planning maps require specialized understanding of spatial configurations, regulatory requirements, and multi-scale analysis.To address this challenge, we introduce PlanGPT-VL, the first domain-specific VLM tailored for urban planning maps. PlanGPT-VL employs three innovations:(1) PlanAnno-V framework for high-quality VQA data synthesis,(2) Critical Point Thinking (CPT) to reduce hallucinations through structured verification, and(3) PlanBench-V benchmark for systematic evaluation.Evaluation on PlanBench-V shows that PlanGPT-VL outperforms general-purpose VLMs on planning map interpretation tasks, with our 7B model achieving performance comparable to larger 72B models.

</details>

---

## 15. Distilling Cross-Modal Knowledge into Domain-Specific Retrievers for Enhanced Industrial Document Understanding

- [ ] Distilling Cross-Modal Knowledge into Domain-Specific Retrievers for Enhanced Industrial Document Understanding | https://aclanthology.org/2025.emnlp-industry.173/

- **Link**: https://aclanthology.org/2025.emnlp-industry.173/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-Augmented Generation (RAG) has shown strong performance in open-domain tasks, but its effectiveness in industrial domains is limited by a lack of domain understanding and document structural elements (DSE) such as tables, figures, charts, and formula.To address this challenge, we propose an efficient knowledge distillation framework that transfers complementary knowledge from both Large Language Models (LLMs) and Vision-Language Models (VLMs) into a compact domain-specific retriever.Extensive experiments and analysis on real-world industrial datasets from shipbuilding and electrical equipment domains demonstrate that the proposed framework improves both domain understanding and visual-structural retrieval, outperforming larger baselines while requiring significantly less computational complexity.

</details>

---

## 16. From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding

- [ ] From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding | https://aclanthology.org/2025.emnlp-industry.185/

- **Link**: https://aclanthology.org/2025.emnlp-industry.185/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid growth of online video content, especially on short video platforms, has created a growing demand for efficient video editing techniques that can condense long-form videos into concise and engaging clips. Existing automatic editing methods predominantly rely on textual cues from ASR transcripts and end-to-end segment selection, often neglecting the rich visual context and leading to incoherent outputs. In this paper, we propose a Human-Inspired automatic video editing framework (HIVE) that leverages multimodal narrative understanding to address these limitations. Our approach incorporates character extraction, dialogue analysis, and narrative summarization through multimodal large language models, enabling a holistic understanding of the video content. To further enhance coherence, we apply scene-level segmentation and decompose the editing process into three subtasks: highlight detection, opening/ending selection, and pruning of irrelevant content. To facilitate research in this area, we introduce DramaAD, a novel benchmark dataset comprising over 2500 short drama episodes and 500 professionally edited advertisement clips. Experimental results demonstrate that our framework consistently outperforms existing baselines across both general and advertisement-oriented editing tasks, significantly narrowing the quality gap between automatic and human-edited videos.

</details>

---

## 17. CAPSTONE: ComposableAttribute‐Prompted Scene Translation forZero‐Shot Vision–Language Reasoning

- [ ] CAPSTONE: ComposableAttribute‐Prompted Scene Translation forZero‐Shot Vision–Language Reasoning | https://aclanthology.org/2025.emnlp-industry.190/

- **Link**: https://aclanthology.org/2025.emnlp-industry.190/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Interpreting visual scenes with high-level reasoning is essential for many real-world applications, such as autonomous systems andcontent moderation, but training and maintaining Vision–Language Models (VLMs) remains resource-intensive and opaque. In this work, we present CAPSTONE, a lightweight, modular framework designed for industrial settings. Instead of relying on multimodal training or fine-tuning large models, CAPSTONE transforms outputs from off-the-shelf vision models into structured text prompts that can be interpreted by a frozen Large Language Model (LLM). This plug-and-play architecture enables reasoning over visual input without access to raw pixels, dramatically reducing computational cost and complexity. On the POPE dataset, our system, using a 7B LLM, outperforms several fully trained VLMs in zero-shot evaluations, while on the VSR benchmark, the 4B model achieves competitive results, together demonstrating strong generalization without retraining. CAPSTONE offers a scalable and interpretable alternative for companies looking to integrate visual reasoning capabilities without the burden of full-scale VLM pipelines.

</details>

---

## 18. Generating Fine Details of Entity Interactions

- [ ] Generating Fine Details of Entity Interactions | https://aclanthology.org/2025.emnlp-industry.37/

- **Link**: https://aclanthology.org/2025.emnlp-industry.37/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent text-to-image models excel at generating high-quality object-centric images from instructions. However, images should also encapsulate rich interactions between objects, where existing models often fall short, likely due to limited training data and benchmarks for rare interactions. This paper explores a novel application of Multimodal Large Language Models (MLLMs) to benchmark and enhance the generation of interaction-rich images.We introduce InterActing-1000, an interaction-focused dataset with 1000 LLM-generated fine-grained prompts for image generation covering (1) functional and action-based interactions, (2) multi-subject interactions, and (3) compositional spatial relationships.To address interaction-rich generation challenges, we propose a decomposition-augmented refinement procedure. Our approach, DetailScribe, leverages LLMs to decompose interactions into finer-grained concepts, uses an MLLM to critique generated images, and applies targeted refinements with a partial diffusion denoising process. Automatic and human evaluations show significantly improved image quality, demonstrating the potential of enhanced inference strategies. Our dataset and code are available at https://detailscribe.github.io/.

</details>

---

## 19. VENUS: AVLLM-driven Video Content Discovery System for Real Application Scenarios

- [ ] VENUS: AVLLM-driven Video Content Discovery System for Real Application Scenarios | https://aclanthology.org/2025.emnlp-industry.4/

- **Link**: https://aclanthology.org/2025.emnlp-industry.4/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video Content Discovery (VCD) is to identify the specific videos defined by a certain pre-specified text policy (or constraint), which plays a crucial role in building a healthy and high-quality Web content ecology. Currently, related works typically employ multiple classifiers or similarity-based systems to support VCD. However, these approaches are difficult to manage, lack generalization power, and suffer from low performance. To tackle these problems, this paper presents a new Vision-Language Large Model (VLLM)-driven VCD system called VENUS (the abbreviation of Video contENt UnderStander). Concretely, we first develop an automatic policy-guided sequential annotator (APSA) to generate high-quality, VCD-specific, and reasoning-equipped instruct-tuning data for model training, then extend the VLLM inference to support VCD better. Following that, we construct a real VCD test set called VCD-Bench, which includes a total of 13 policies and 57K videos. Furthermore, to evaluate its practical efficacy, we deploy VENUS in three different real scenarios. Extensive experiments on both the VCD-Bench and public evaluation datasets for various VCD-related tasks demonstrate the superiority of VENUS over existing baselines.

</details>

---

## 20. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for EnhancedLLMContent Moderation

- [ ] Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for EnhancedLLMContent Moderation | https://aclanthology.org/2025.emnlp-industry.46/

- **Link**: https://aclanthology.org/2025.emnlp-industry.46/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the Graph of Attacks with Pruning (GAP) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP’s superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning.

</details>

---

## 21. DASR: Distributed Adaptive Scene Recognition - A Multi-Agent Cloud-Edge Framework for Language-Guided Scene Detection

- [ ] DASR: Distributed Adaptive Scene Recognition - A Multi-Agent Cloud-Edge Framework for Language-Guided Scene Detection | https://aclanthology.org/2025.emnlp-industry.57/

- **Link**: https://aclanthology.org/2025.emnlp-industry.57/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The increasing complexity of modern driving systems demands efficient collection and analysis of specific driving scenarios that are crucial for system development and validation. Current approaches either rely on massive data collection followed by manual filtering, or rigid threshold-based recording systems that often miss important edge cases. In this paper, we present Distributed Adaptive Scene Recognition (DASR), a novel multi-agent cloud-edge framework for language-guided scene detection in connected vehicles. Our system leverages the complementary strengths of cloud-based large language models and edge-deployed vision language models to intelligently identify and preserve relevant driving scenarios while optimizing limited on-vehicle buffer storage. The cloud-based LLM serves as an intelligent coordinator that analyzes developer prompts to determine which specialized tools and sensor data streams should be incorporated, while the edge-deployed VLM efficiently processes video streams in real time to make relevant decisions. Extensive experiments across multiple driving datasets demonstrate that our framework achieves superior performance compared to larger baseline models, with exceptional performance on complex driving tasks requiring sophisticated reasoning. DASR also shows strong generalization capabilities on out-of-distribution datasets and significantly reduces storage requirements (28.73 %) compared to baseline methods.

</details>

---

## 22. Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning

- [ ] Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning | https://aclanthology.org/2025.emnlp-industry.66/

- **Link**: https://aclanthology.org/2025.emnlp-industry.66/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video Temporal Grounding (VTG) aims to localize relevant temporal segments in videos given natural language queries. Despite recent progress with large vision-language models (LVLMs) and instruction-tuning, existing approaches often suffer from limited temporal awareness and poor generalization. In this work, we introduce a two-stage training framework that integrates supervised fine-tuning with reinforcement learning (RL) to improve both the accuracy and robustness of VTG models. Our approach first leverages high-quality curated cold-start data for SFT initialization, followed by difficulty-controlled RL to further enhance temporal localization and reasoning abilities. Comprehensive experiments on multiple VTG benchmarks demonstrate that our method consistently outperforms existing models, particularly in challenging and open-domain scenarios. We conduct an in-depth analysis of training strategies and dataset curation, highlighting the importance of both high-quality cold-start data and difficulty-controlled RL. To facilitate further research and industrial adoption, we release all intermediate datasets, models, and code to the community.

</details>

---

## 23. Reasoning-Enhanced Domain-Adaptive Pretraining of Multimodal Large Language Models for Short Video Content Governance

- [ ] Reasoning-Enhanced Domain-Adaptive Pretraining of Multimodal Large Language Models for Short Video Content Governance | https://aclanthology.org/2025.emnlp-industry.77/

- **Link**: https://aclanthology.org/2025.emnlp-industry.77/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Short video platforms are evolving rapidly, making the identification of inappropriate content increasingly critical.Existing approaches typically train separate and small classification models for each type of issue, which requires extensive human-labeled data and lacks cross-issue generalization.We propose a reasoning-enhanced multimodal large language model (MLLM) pretraining paradigm for unified inappropriate content detection. To address the distribution gap between short video content and the original pretraining data of MLLMs, as well as the complex issue definitions, we introduce three targeted pretraining tasks:(1)Caption, to enhance the MLLM’s perception of video details;(2)Visual Question Answering (VQA), to deepen the MLLM’s understanding of issue definitions and annotation guidelines;(3)Chain-of-Thought (CoT), to enhance the MLLM’s reasoning capability.Experimental results show that our pretraining approach significantly improves the MLLM’s performance in both zero-shot and supervised fine-tuning (SFT) settings.In addition, our pretrained model demonstrates strong generalization capabilities to emergent, previously unseen issues.

</details>

---

## 24. Learning fromLLMAgents: In-Context Generative Models for Text Casing inE-Commerce Ads

- [ ] Learning fromLLMAgents: In-Context Generative Models for Text Casing inE-Commerce Ads | https://aclanthology.org/2025.emnlp-industry.79/

- **Link**: https://aclanthology.org/2025.emnlp-industry.79/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

E-commerce ad platforms enforce content policies and review created ads before publication, with casing requirements playing a critical role in maintaining readability and brand consistency. Existing NER-based transformer models have been widely used for casing correction, but they process characters independently in a classification-based manner, failing to capture sentence level contextual dependencies, making them less reliable when handling unseen or ad-specific terms, e.g., brand names. LLMs like ChatGPT offer better generalization to proper nouns, but they are expensive and have high latency. Besides, generative model can suffer from hallucination. To address these challenges, we propose a two-stage approach: (1) an LLM-based Agent leveraging Chain-of-Actions (CoA) to enforce casing policies while accurately handling ads-specific terms, such as brand names, and (2) a lightweight generative model that preserves the LLM Agent’s knowledge while significantly reducing latency and costs. We design a novel in-context decoding strategy, which avoids hallucinations. Our approach outperforms NER-based methods and achieves near-LLM Agent performance, making it a scalable and efficient solution for real-world ad compliance automation.

</details>

---

## 25. I-SEE: An Instruction-tuned,SOP-Enhanced Quality Evaluator for Product Content

- [ ] I-SEE: An Instruction-tuned,SOP-Enhanced Quality Evaluator for Product Content | https://aclanthology.org/2025.emnlp-industry.96/

- **Link**: https://aclanthology.org/2025.emnlp-industry.96/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-quality content is critical for driving customer satisfaction and conversions across digital platforms and e-commerce. Ensuring that essential information is complete, accurate, and aligned with customer expectations presents a significant challenge at scale. Existing approaches to content evaluation often treat all information uniformly, without prioritizing based on customer relevance, and rely heavily on manual prompt design to encode domain expertise into Large Language Models (LLMs). We present ISEE, a unified framework that addresses these limitations through three core innovations: (1) automated identification of customer-impacting features by synthesizing signals from search behavior, queries, and feedback, enabling targeted content improvements; (2) an instruction-tuned multimodal LLM trained to reliably follow structured operational guidelines, reducing dependence on manual prompt engineering; and (3) robust zero-shot generalization to new product content, features and SOPs via targeted instruction tuning. Evaluated across 20 product categories and 150 product specific features, ISEE achieves 90% precision at 78% recall in detecting content inconsistencies, outperforming much larger (> 200B parameters) models while using a compact 12B architecture.

</details>

---

## 26. Balanced Multi-Factor In-Context Learning for Multilingual Large Language Models

- [ ] Balanced Multi-Factor In-Context Learning for Multilingual Large Language Models | https://aclanthology.org/2025.emnlp-main.1016/

- **Link**: https://aclanthology.org/2025.emnlp-main.1016/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multilingual large language models (MLLMs) are able to leverage in-context learning (ICL) to achieve high performance by leveraging cross-lingual knowledge transfer without parameter updates. However, their effectiveness is highly sensitive to example selection, particularly in multilingual settings. Based on the findings of existing work, three key factors influence multilingual ICL: (1) semantic similarity, (2) linguistic alignment, and (3) language-specific performance. However, existing approaches address these factors independently, without explicitly disentangling their combined impact, leaving optimal example selection underexplored. To address this gap, we propose balanced multi-factor ICL (BMF-ICL), a method that quantifies and optimally balances these factors for improved example selection. Experiments on mCSQA and TYDI across four MLLMs demonstrate that BMF-ICL outperforms existing methods. Further analysis highlights the importance of incorporating all three factors and the importance of selecting examples from multiple languages.

</details>

---

## 27. MR. Judge: Multimodal Reasoner as a Judge

- [ ] MR. Judge: Multimodal Reasoner as a Judge | https://aclanthology.org/2025.emnlp-main.1021/

- **Link**: https://aclanthology.org/2025.emnlp-main.1021/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The paradigm of using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) as evaluative judges has emerged as an effective approach in RLHF and inference-time scaling. In this work, we propose Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering general-purpose MLLMs judges with strong reasoning capabilities. Instead of directly assigning scores for each response, we formulate the judgement process as a reasoning-inspired multiple-choice problem. Specifically, the judge model first conducts deliberate reasoning covering different aspects of the responses and eventually selects the best response from them. This reasoning process not only improves the interpretibility of the judgement, but also greatly enhances the performance of MLLM judges. To cope with the lack of questions with scored responses, we propose the following strategy to achieve automatic annotation: 1) Reverse Response Candidates Synthesis: starting from a supervised fine-tuning (SFT) dataset, we treat the original response as the best candidate and prompt the MLLM to generate plausible but flawed negative candidates. 2) Text-based reasoning distillation: we carefully design a data synthesis pipeline for distilling the reasoning capability from a text-based reasoning model, which is adopted to enable the MLLM judges to regain complex reasoning ability via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge is effective across a wide range of tasks. Specifically, our MR. Judge-7B surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet during inference-time scaling by up to 7.7%.

</details>

---

## 28. Pointing to a Llama and Call it a Camel: On the Sycophancy of Multimodal Large Language Models

- [ ] Pointing to a Llama and Call it a Camel: On the Sycophancy of Multimodal Large Language Models | https://aclanthology.org/2025.emnlp-main.1020/

- **Link**: https://aclanthology.org/2025.emnlp-main.1020/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have demonstrated extraordinary capabilities in conducting conversations based on image inputs. However, we observe that MLLMs exhibit a pronounced form of visual sycophantic behavior. While similar behavior has also been noted in text-based large language models (LLMs), it becomes significantly more prominent when MLLMs process image inputs. We refer to this phenomenon as the “sycophantic modality gap.” To better understand this issue, we further analyze the factors that contribute to the exacerbation of this gap. To mitigate the visual sycophantic behavior, we first experiment with naive supervised fine-tuning to help the MLLM resist misleading instructions from the user. However, we find that this approach also makes the MLLM overly resistant to corrective instructions (i.e., stubborn even if it is wrong). To alleviate this trade-off, we propose Sycophantic Reflective Tuning (SRT), which enables the MLLM to engage in reflective reasoning, allowing it to determine whether a user’s instruction is misleading or corrective before drawing a conclusion. After applying SRT, we observe a significant reduction in sycophantic behavior toward misleading instructions, without resulting in excessive stubbornness when receiving corrective instructions.

</details>

---

## 29. How Do Large Vision-Language Models See Text in Image? Unveiling the Distinctive Role ofOCRHeads

- [ ] How Do Large Vision-Language Models See Text in Image? Unveiling the Distinctive Role ofOCRHeads | https://aclanthology.org/2025.emnlp-main.1032/

- **Link**: https://aclanthology.org/2025.emnlp-main.1032/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant advancements in Large Vision Language Models (LVLMs), a gap remains, particularly regarding their interpretability and how they locate and interpret textual information within images. In this paper, we explore various LVLMs to identify the specific heads responsible for recognizing text from images, which we term the Optical Character Recognition Head (OCR Head). Our findings regarding these heads are as follows: (1) Less Sparse: Unlike previous retrieval heads, a large number of heads are activated to extract textual information from images. (2) Qualitatively Distinct: OCR heads possess properties that differ significantly from general retrieval heads, exhibiting low similarity in their characteristics. (3) Statically Activated: The frequency of activation for these heads closely aligns with their OCR scores. We validate our findings in downstream tasks by applying Chain-of-Thought (CoT) to both OCR and conventional retrieval heads and by masking these heads. We also demonstrate that redistributing sink-token values within the OCR heads improves performance. These insights provide a deeper understanding of the internal mechanisms LVLMs employ in processing embedded textual information in images.

</details>

---

## 30. Explainability and Interpretability of Multilingual Large Language Models: A Survey

- [ ] Explainability and Interpretability of Multilingual Large Language Models: A Survey | https://aclanthology.org/2025.emnlp-main.1033/

- **Link**: https://aclanthology.org/2025.emnlp-main.1033/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multilingual large language models (MLLMs) demonstrate state-of-the-art capabilities across diverse cross-lingual and multilingual tasks. Their complex internal mechanisms, however, often lack transparency, posing significant challenges in elucidating their internal processing of multilingualism, cross-lingual transfer dynamics and handling of language-specific features. This paper addresses this critical gap by presenting a survey of current explainability and interpretability methods specifically for MLLMs. To our knowledge, it is the first comprehensive review of its kind. Existing literature is categorised according to the explainability techniques employed, the multilingual tasks addressed, the languages investigated and available resources. The survey further identifies key challenges, distils core findings and outlines promising avenues for future research within this rapidly evolving domain.

</details>

---

## 31. WebInject: Prompt Injection Attack to Web Agents

- [ ] WebInject: Prompt Injection Attack to Web Agents | https://aclanthology.org/2025.emnlp-main.104/

- **Link**: https://aclanthology.org/2025.emnlp-main.104/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

</details>

---

## 32. MMAG: Multimodal Learning for Mucus Anomaly Grading in Nasal Endoscopy via Semantic Attribute Prompting

- [ ] MMAG: Multimodal Learning for Mucus Anomaly Grading in Nasal Endoscopy via Semantic Attribute Prompting | https://aclanthology.org/2025.emnlp-main.1066/

- **Link**: https://aclanthology.org/2025.emnlp-main.1066/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Accurate grading of rhinitis severity in nasal endoscopy relies heavily on the characterization of key secretion types, notably clear nasal discharge (CND) and purulent nasal secretion (PUS). However, both exhibit ambiguous appearance and high structural variability, posing challenges to automated grading under weak supervision. To address this, we propose Multimodal Learning for Mucus Anomaly Grading (MMAG), which integrates structured prompts with rank-aware vision-language modeling for joint detection and grading. Attribute prompts are constructed from clinical descriptors (e.g., secretion type, severity, location) and aligned with multi-level visual features via a dual-branch encoder. During inference, the model localizes mucus anomalies and maps the input image to severity-specific prompts (e.g., “moderate pus”), projecting them into a rank-aware feature space for progressive similarity scoring.Extensive evaluations on CND and PUS datasets show that our method achieves consistent gains over Baseline, improving AUC by 6.31% and 4.79%, and F1 score by 12.85% and 6.03%, respectively.This framework enables interpretable, annotation-efficient, and semantically grounded assessment of rhinitis severity based on mucus anomalies.

</details>

---

## 33. DCP: Dual-Cue Pruning for Efficient Large Vision-Language Models

- [ ] DCP: Dual-Cue Pruning for Efficient Large Vision-Language Models | https://aclanthology.org/2025.emnlp-main.1074/

- **Link**: https://aclanthology.org/2025.emnlp-main.1074/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) achieve remarkable performance in multimodal tasks but suffer from high computational costs due to the large number of visual tokens. Existing pruning methods either apply after visual tokens enter the LLM or perform pre-pruning based solely on visual attention. Both fail to balance efficiency and semantic alignment, as post-pruning incurs redundant computation, while visual-only pre-pruning overlooks multimodal relevance.To address this limitation, we propose Dual-Cue Pruning (DCP), a novel cross-modal pruning framework that jointly considers textual semantics and visual self-attention. DCP consists of a text-aware computation module, which employs a gradient-weighted attention mechanism to enhance text-visual alignment, and an image-aware computation module, which utilizes deep-layer self-attention distributions to retain essential structural information. By integrating both cues, DCP adaptively selects the most informative visual tokens, achieving efficient inference acceleration while maintaining strong task performance. Experimental results show that DCP can retain only 25% of the visual tokens, with a minimal performance degradation of only 0.063% on LLaVA-1.5-13B, demonstrating its effectiveness in balancing efficiency and accuracy.

</details>

---

## 34. Leveraging Large Models to Evaluate Novel Content: A Case Study on Advertisement Creativity

- [ ] Leveraging Large Models to Evaluate Novel Content: A Case Study on Advertisement Creativity | https://aclanthology.org/2025.emnlp-main.1072/

- **Link**: https://aclanthology.org/2025.emnlp-main.1072/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Evaluating creativity is challenging, even for humans, not only because of its subjectivity but also because it involves complex cognitive processes. Inspired by work in marketing, we attempt to break down visual advertisement creativity into atypicality and originality. With fine-grained human annotations on these dimensions, we propose a suite of tasks specifically for such a subjective problem. We also evaluate the alignment between state-of-the-art (SoTA) vision language models (VLMs) and humans on our proposed benchmark, demonstrating both the promises and challenges of using VLMs for automatic creativity assessment.

</details>

---

## 35. Unveiling the Response of Large Vision-Language Models to Visually Absent Tokens

- [ ] Unveiling the Response of Large Vision-Language Models to Visually Absent Tokens | https://aclanthology.org/2025.emnlp-main.1092/

- **Link**: https://aclanthology.org/2025.emnlp-main.1092/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) generate contextually relevant responses by jointly interpreting visual and textual inputs. However, our finding reveals they often mistakenly perceive text inputs lacking visual evidence as being part of the image, leading to erroneous responses. In light of this finding, we probe whether LVLMs possess an internal capability to determine if textual concepts are grounded in the image, and discover a specific subset of Feed-Forward Network (FFN) neurons, termed Visual Absence-aware (VA) neurons, that consistently signal the visual absence through a distinctive activation pattern. Leveraging these patterns, we develop a detection module that systematically classifies whether an input token is visually grounded. Guided by its prediction, we propose a method to refine the outputs by reinterpreting question prompts or replacing the detected absent tokens during generation. Extensive experiments show that our method effectively mitigates the models’ tendency to falsely presume the visual presence of text input and its generality across various LVLMs.

</details>

---

## 36. RICO: Improving Accuracy and Completeness in Image Recaptioning via Visual Reconstruction

- [ ] RICO: Improving Accuracy and Completeness in Image Recaptioning via Visual Reconstruction | https://aclanthology.org/2025.emnlp-main.1105/

- **Link**: https://aclanthology.org/2025.emnlp-main.1105/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image recaptioning is widely used to generate training datasets with enhanced quality for various multimodal tasks. Existing recaptioning methods typically rely on powerful multimodal large language models (MLLMs) to enhance textual descriptions, but often suffer from inaccuracies due to hallucinations and incompleteness caused by missing fine-grained details. To address these limitations, we propose RICO, a novel framework that refines captions through visual reconstruction. Specifically, we leverage a text-to-image model to reconstruct a caption into a reference image, and prompt an MLLM to identify discrepancies between the original and reconstructed images to refine the caption. This process is performed iteratively, further progressively promoting the generation of more faithful and comprehensive descriptions. To mitigate the additional computational cost induced by the iterative process, we introduce RICO-Flash, which learns to generate captions like RICO using DPO. Extensive experiments demonstrate that our approach significantly improves caption accuracy and completeness, outperforms most baselines by approximately 10% on both CapsBench and CompreCap.

</details>

---

## 37. Puzzled by Puzzles: When Vision-Language Models Can’t Take a Hint

- [ ] Puzzled by Puzzles: When Vision-Language Models Can’t Take a Hint | https://aclanthology.org/2025.emnlp-main.1101/

- **Link**: https://aclanthology.org/2025.emnlp-main.1101/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Rebus puzzles, visual riddles that encode language through imagery, spatial arrangement, and symbolic substitution, pose a unique challenge to current vision-language models (VLMs). Unlike traditional image captioning or question answering tasks, rebus solving requires multimodal abstraction, symbolic reasoning, and a grasp of cultural, phonetic and linguistic puns. In this short paper, we investigate the capacity of contemporary VLMs to interpret and solve rebus puzzles by constructing a hand-generated and annotated benchmark of diverse english-language rebus puzzles, ranging from simple pictographic substitutions to spatially-dependent cues (“head” over “heels”). We analyze how different VLMs perform, and our findings reveal that while VLMs exhibit some surprising capabilities in decoding simple visual clues, they struggle significantly with tasks requiring abstract reasoning, lateral thinking, and understanding visual metaphors.

</details>

---

## 38. Seeing Culture: A Benchmark for Visual Reasoning and Grounding

- [ ] Seeing Culture: A Benchmark for Visual Reasoning and Grounding | https://aclanthology.org/2025.emnlp-main.1131/

- **Link**: https://aclanthology.org/2025.emnlp-main.1131/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal vision-language models (VLMs) have made substantial progress in various tasks that require a combined understanding of visual and textual content, particularly in cultural understanding tasks, with the emergence of new cultural datasets. However, these datasets frequently fall short of providing cultural reasoning while underrepresenting many cultures.In this paper, we introduce the Seeing Culture Benchmark (SCB), focusing on cultural reasoning with a novel approach that requires VLMs to reason on culturally rich images in two stages: i) selecting the correct visual option with multiple-choice visual question answering (VQA), and ii) segmenting the relevant cultural artifact as evidence of reasoning. Visual options in the first stage are systematically organized into three types: those originating from the same country, those from different countries, or a mixed group. Notably, all options are derived from a singular category for each type. Progression to the second stage occurs only after a correct visual option is chosen. The SCB benchmark comprises 1,065 images that capture 138 cultural artifacts across five categories from seven Southeast Asia countries, whose diverse cultures are often overlooked, accompanied by 3,178 questions, of which 1,093 are unique and meticulously curated by human annotators. Our evaluation of various VLMs reveals the complexities involved in cross-modal cultural reasoning and highlights the disparity between visual reasoning and spatial grounding in culturally nuanced scenarios. The SCB serves as a crucial benchmark for identifying these shortcomings, thereby guiding future developments in the field of cultural reasoning. https://github.com/buraksatar/SeeingCulture

</details>

---

## 39. Follow the Flow: Fine-grained Flowchart Attribution with Neurosymbolic Agents

- [ ] Follow the Flow: Fine-grained Flowchart Attribution with Neurosymbolic Agents | https://aclanthology.org/2025.emnlp-main.1144/

- **Link**: https://aclanthology.org/2025.emnlp-main.1144/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Flowcharts are a critical tool for visualizing decision-making processes. However, their non-linear structure and complex visual-textual relationships make it challenging to interpret them using LLMs, as vision-language models frequently hallucinate nonexistent connections and decision paths when analyzing these diagrams. This leads to compromised reliability for automated flowchart processing in critical domains such as logistics, health, and engineering. We introduce the task of Fine-grained Flowchart Attribution, which traces specific components grounding a flowchart referring LLM response. Flowchart Attribution ensures the verifiability of LLM predictions and improves explainability by linking generated responses to the flowchart’s structure. We propose FlowPathAgent, a neurosymbolic agent that performs fine-grained post hoc attribution through graph-based reasoning. It first segments the flowchart, then converts it into a structured symbolic graph, and then employs an agentic approach to dynamically interact with the graph, to generate attribution paths. Additionally, we present FlowExplainBench, a novel benchmark for evaluating flowchart attributions across diverse styles, domains, and question types. Experimental results show that FlowPathAgent mitigates visual hallucinations in LLM answers over flowchart QA, outperforming strong baselines by 10–14% on our proposed FlowExplainBench dataset.

</details>

---

## 40. WildDoc: How Far Are We from Achieving Comprehensive and Robust Document Understanding in the Wild?

- [ ] WildDoc: How Far Are We from Achieving Comprehensive and Robust Document Understanding in the Wild? | https://aclanthology.org/2025.emnlp-main.1172/

- **Link**: https://aclanthology.org/2025.emnlp-main.1172/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced capabilities in Document Understanding. However, prevailing benchmarks like DocVQA and ChartQA predominantly comprise scanned or digital documents, inadequately reflecting the intricate challenges posed by diverse real-world scenarios such as variable illumination and physical distortions. This paper introduces WildDoc, the inaugural benchmark designed specifically for assessing document understanding in natural environments. WildDoc incorporates a diverse set of manually captured document images reflecting real-world conditions and leverages document sources from established benchmarks to facilitate comprehensive comparisons with digital or scanned documents. Further, to rigorously evaluate model robustness, each document is captured four times under different conditions. Evaluations of state-of-the-art MLLMs on WildDoc expose substantial performance declines and underscore the models’ inadequate robustness compared to traditional benchmarks, highlighting the unique challenges posed by real-world document understanding.

</details>

---

## 41. Fooling theLVLMJudges: Visual Biases inLVLM-Based Evaluation

- [ ] Fooling theLVLMJudges: Visual Biases inLVLM-Based Evaluation | https://aclanthology.org/2025.emnlp-main.1182/

- **Link**: https://aclanthology.org/2025.emnlp-main.1182/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, large vision–language models (LVLMs) have emerged as the preferred tools for judging text–image alignment, yet their robustness along the visual modality remains underexplored. This work is the first study to address a key research question: Can adversarial visual manipulations systematically fool LVLM judges into assigning unfairly inflated scores? We define potential image-induced biases within the context of T2I evaluation and examine how these biases affect the evaluations of LVLM judges. Moreover, we introduce a novel, fine-grained, multi-domain meta-evaluation benchmark named FRAME, which is deliberately constructed to exhibit diverse score distributions. By introducing the defined biases into the benchmark, we reveal that all tested LVLM judges exhibit vulnerability across all domains, consistently inflating scores for manipulated images. Further analysis reveals that combining multiple biases amplifies their effects, and pairwise evaluations are similarly susceptible. Moreover, we observe that visual biases persist despite prompt-based mitigation strategies, highlighting the vulnerability of current LVLM evaluation systems and underscoring the urgent need for more robust LVLM judges.

</details>

---

## 42. ClimateViz: A Benchmark for Statistical Reasoning and Fact Verification on Scientific Charts

- [ ] ClimateViz: A Benchmark for Statistical Reasoning and Fact Verification on Scientific Charts | https://aclanthology.org/2025.emnlp-main.1196/

- **Link**: https://aclanthology.org/2025.emnlp-main.1196/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scientific fact-checking has largely focused on textual and tabular sources, neglecting scientific charts—a primary medium for conveying quantitative evidence and supporting statistical reasoning in research communication. We introduce ClimateViz, the first large-scale benchmark for scientific fact-checking grounded in real-world, expert-curated scientific charts. ClimateViz comprises 49,862 claims paired with 2,896 visualizations, each labeled as support, refute, or not enough information. To enable interpretable verification, each instance includes structured knowledge graph explanations that capture statistical patterns, temporal trends, spatial comparisons, and causal relations. We conduct a comprehensive evaluation of state-of-the-art multimodal large language models, including proprietary and open-source ones, under zero-shot and few-shot settings. Our results show that current models struggle to perform fact-checking when statistical reasoning over charts is required: even the best-performing systems, such as Gemini 2.5 and InternVL 2.5, achieve only 76.2–77.8% accuracy in label-only output settings, which is far below human performance (89.3% and 92.7%). While few-shot prompting yields limited improvements, explanation-augmented outputs significantly enhance performance in some closed-source models, notably o3 and Gemini 2.5.

</details>

---

## 43. Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization

- [ ] Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization | https://aclanthology.org/2025.emnlp-main.121/

- **Link**: https://aclanthology.org/2025.emnlp-main.121/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of large Vision Language Models (VLMs) has broadened the scope and capabilities of single-modal Large Language Models (LLMs) by integrating visual modalities, thereby unlocking transformative cross-modal applications in a variety of real-world scenarios. Despite their impressive performance, VLMs are prone to significant hallucinations, particularly in the form of cross-modal inconsistencies. Building on the success of Reinforcement Learning from Human Feedback (RLHF) in aligning LLMs, recent advancements have focused on applying direct preference optimization (DPO) on carefully curated datasets to mitigate these issues. Yet, such approaches typically introduce preference signals in a brute-force manner, neglecting the crucial role of visual information in the alignment process. In this paper, we introduce Re-Align, a novel alignment framework that leverages image retrieval to construct a dual-preference dataset, effectively incorporating both textual and visual preference signals. We further introduce rDPO, an extension of the standard direct preference optimization that incorporates an additional visual preference objective during fine-tuning. Our experimental results demonstrate that Re-Align not only mitigates hallucinations more effectively than previous methods but also yields significant performance gains in general visual question-answering (VQA) tasks. Moreover, we show that Re-Align maintains robustness and scalability across a wide range of VLM sizes and architectures. This work represents a significant step forward in aligning multimodal LLMs, paving the way for more reliable and effective cross-modal applications.

</details>

---

## 44. RAcQUEt: Unveiling the Dangers of Overlooked Referential Ambiguity in VisualLLMs

- [ ] RAcQUEt: Unveiling the Dangers of Overlooked Referential Ambiguity in VisualLLMs | https://aclanthology.org/2025.emnlp-main.1206/

- **Link**: https://aclanthology.org/2025.emnlp-main.1206/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Ambiguity resolution is key to effective communication. While humans effortlessly address ambiguity through conversational grounding strategies, the extent to which current language models can emulate these strategies remains unclear. In this work, we examine referential ambiguity in image-based question answering by introducing RAcQUEt, a carefully curated dataset targeting distinct aspects of ambiguity. Through a series of evaluations, we reveal significant limitations and problems of overconfidence of state-of-the-art large multimodal language models in addressing ambiguity in their responses. The overconfidence issue becomes particularly relevant for RAcQUEt-BIAS, a subset designed to analyze a critical yet underexplored problem: failing to address ambiguity leads to stereotypical, socially biased responses. Our results underscore the urgency of equipping models with robust strategies to deal with uncertainty without resorting to undesirable stereotypes.

</details>

---

## 45. Grounded Semantic Role Labelling from Synthetic Multimodal Data for Situated Robot Commands

- [ ] Grounded Semantic Role Labelling from Synthetic Multimodal Data for Situated Robot Commands | https://aclanthology.org/2025.emnlp-main.1212/

- **Link**: https://aclanthology.org/2025.emnlp-main.1212/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding natural language commands in situated Human-Robot Interaction (HRI) requires linking linguistic input to perceptual context. Traditional symbolic parsers lack the flexibility to operate in complex, dynamic environments. We introduce a novel Multimodal Grounded Semantic Role Labelling (G-SRL) framework that combines frame semantics with perceptual grounding, enabling robots to interpret commands via multimodal logical forms. Our approach leverages modern Visual Language Models (VLLMs), which jointly process text and images, and is supported by an automated pipeline that generates high-quality training data. Structured command annotations are converted into photorealistic scenes via LLM-guided prompt engineering and diffusion models, then rigorously validated through object detection and visual question answering. The pipeline produces over 11,000 image-command pairs (3,500+ manually validated), while approaching the quality of manually curated datasets at significantly lower cost.

</details>

---

## 46. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection

- [ ] Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection | https://aclanthology.org/2025.emnlp-main.1215/

- **Link**: https://aclanthology.org/2025.emnlp-main.1215/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems.Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

</details>

---

## 47. Socratic-MCTS: Test-Time Visual Reasoning by Asking the Right Questions

- [ ] Socratic-MCTS: Test-Time Visual Reasoning by Asking the Right Questions | https://aclanthology.org/2025.emnlp-main.1230/

- **Link**: https://aclanthology.org/2025.emnlp-main.1230/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent research in vision-language models (VLMs) has centered around the possibility of equipping them with implicit long-form chain-of-thought reasoning—akin to the success observed in language models—via distillation and reinforcement learning. But what about the non-reasoning models already trained and deployed across the internet? Should we simply abandon them, or is there hope for a search mechanism that can elicit hidden knowledge and induce long reasoning traces— without any additional training or supervision? In this paper, we explore this possibility using a Monte Carlo Tree Search (MCTS)-inspired algorithm, which injects subquestion–subanswer pairs into the model’s output stream. We show that framing reasoning as a search process—where subquestions act as latent decisions within a broader inference trajectory—helps the model “connect the dots” between fragmented knowledge and produce extended reasoning traces in non-reasoning models. We evaluate our method across three benchmarks and observe consistent improvements. Notably, our approach yields a 2% overall improvement on MMMU-PRO, including a significant 9% gain in Liberal Arts.

</details>

---

## 48. VisFinEval: A Scenario-DrivenChinese Multimodal Benchmark for Holistic Financial Understanding

- [ ] VisFinEval: A Scenario-DrivenChinese Multimodal Benchmark for Holistic Financial Understanding | https://aclanthology.org/2025.emnlp-main.1229/

- **Link**: https://aclanthology.org/2025.emnlp-main.1229/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) hold great promise for automating complex financial analysis. To comprehensively evaluate their capabilities, we introduce VisFinEval, the first large-scale Chinese benchmark that spans the full front-middle-back office lifecycle of financial tasks. VisFinEval comprises 15,848 annotated question–answer pairs drawn from eight common financial image modalities (e.g., K-line charts, financial statements, official seals), organized into three hierarchical scenario depths: Financial Knowledge & Data Analysis, Financial Analysis & Decision Support, and Financial Risk Control & Asset Optimization. We evaluate 21 state-of-the-art MLLMs in a zero-shot setting. The top model, Qwen-VL-max, achieves an overall accuracy of 76.3%, outperforming non-expert humans but trailing financial experts by over 14 percentage points. Our error analysis uncovers six recurring failure modes—including cross-modal misalignment, hallucinations, and lapses in business-process reasoning—that highlight critical avenues for future research. VisFinEval aims to accelerate the development of robust, domain-tailored MLLMs capable of seamlessly integrating textual and visual financial information. The data and the code are available at https://github.com/SUFE-AIFLM-Lab/VisFinEval.

</details>

---

## 49. Grounding Multilingual MultimodalLLMs With Cultural Knowledge

- [ ] Grounding Multilingual MultimodalLLMs With Cultural Knowledge | https://aclanthology.org/2025.emnlp-main.1232/

- **Link**: https://aclanthology.org/2025.emnlp-main.1232/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models excel in high-resource settings, but often misinterpret long-tail cultural entities and underperform in low-resource languages. To address this gap, we propose a data-centric approach that directly grounds MLLMs in cultural knowledge. Leveraging a large scale knowledge graph from Wikidata, we collect images that represent culturally significant entities, and generate synthetic multilingual visual question answering data. The resulting dataset, CulturalGround, comprises 22 million high-quality, culturally-rich VQA pairs spanning 42 countries and 39 languages. We train an open-source MLLM CulturalPangea on CulturalGround, interleaving standard multilingual instruction-tuning data to preserve general abilities. Cultural-Pangea achieves state-of-the-art performance among open models on various culture-focused multilingual multimodal benchmarks, outperforming prior models by an average of +5.0%without degrading results on mainstream vision–language tasks. Our findings show that our targeted, culturally grounded approach could substantially narrow the cultural gap in MLLMs and offer a practical path towards globally inclusive multimodal systems.

</details>

---

## 50. MERMAID: Multi-perspective Self-reflective Agents with Generative Augmentation for Emotion Recognition

- [ ] MERMAID: Multi-perspective Self-reflective Agents with Generative Augmentation for Emotion Recognition | https://aclanthology.org/2025.emnlp-main.1252/

- **Link**: https://aclanthology.org/2025.emnlp-main.1252/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have demonstrated strong performance across diverse multimodal tasks, achieving promising outcomes. However, their application to emotion recognition in natural images remains underexplored. MLLMs struggle to handle ambiguous emotional expressions and implicit affective cues, whose capability is crucial for affective understanding but largely overlooked. To address these challenges, we propose MERMAID, a novel multi-agent framework that integrates a multi-perspective self-reflection module, an emotion-guided visual augmentation module, and a cross-modal verification module. These components enable agents to interact across modalities and reinforce subtle emotional semantics, thereby enhancing emotion recognition and supporting autonomous performance. Extensive experiments show that MERMAID outperforms existing methods, achieving absolute accuracy gains of 8.70%–27.90% across diverse benchmarks and exhibiting greater robustness in emotionally diverse scenarios.

</details>

---

## 51. Hanfu-Bench: A Multimodal Benchmark on Cross-Temporal Cultural Understanding and Transcreation

- [ ] Hanfu-Bench: A Multimodal Benchmark on Cross-Temporal Cultural Understanding and Transcreation | https://aclanthology.org/2025.emnlp-main.1251/

- **Link**: https://aclanthology.org/2025.emnlp-main.1251/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Culture is a rich and dynamic domain that evolves across both geography and time. However, existing studies on cultural understanding with vision-language models (VLMs) primarily emphasize geographic diversity, often overlooking the critical temporal dimensions. To bridge this gap, we introduce Hanfu-Bench, a novel, expert-curated multimodal dataset. Hanfu, a traditional garment spanning ancient Chinese dynasties, serves as a representative cultural heritage that reflects the profound temporal aspects of Chinese culture while remaining highly popular in Chinese contemporary society. Hanfu-Bench comprises two core tasks: cultural visual understanding and cultural image transcreation. The former task examines temporal-cultural feature recognition based on single- or multi-image inputs through multiple-choice visual question answering, while the latter focuses on transforming traditional attire into modern designs through cultural element inheritance and modern context adaptation. Our evaluation shows that closed VLMs perform comparably to non-experts on visual cutural understanding but fall short by 10% to human experts, while open VLMs lags further behind non-experts. For the transcreation task, multi-faceted human evaluation indicates that the best-performing model achieves a success rate of only 42%. Our benchmark provides an essential testbed, revealing significant challenges in this new direction of temporal cultural understanding and creative adaptation.

</details>

---

## 52. Hidden in Plain Sight: Reasoning in Underspecified and Misspecified Scenarios for MultimodalLLMs

- [ ] Hidden in Plain Sight: Reasoning in Underspecified and Misspecified Scenarios for MultimodalLLMs | https://aclanthology.org/2025.emnlp-main.1255/

- **Link**: https://aclanthology.org/2025.emnlp-main.1255/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are increasingly deployed in open-ended, real-world environments where inputs are messy, underspecified, and not always trustworthy. Unlike curated benchmarks, these settings frequently involve instructions that reference missing objects or contradictory facts, rely on ambiguous cues, or request infeasible actions. In such cases, success hinges not merely on task execution, but on the model’s ability to detect when something is silently wrong. This paper presents a systematic analysis of how current MLLMs handle such underspecified and misspecified scenarios: cases where flaws must be inferred from context rather than explicitly stated. Using a curated diagnostic suite spanning four categories of real-world failure modes, we evaluate nine MLLMs, including o3 and GPT-4o, and find that models often fail to surface hidden issues, even when they possess the necessary perceptual and reasoning skills. Explicit prompting reveals that the underlying capabilities exist but are frequently suppressed in favor of user compliance.We further show that simple inference-time interventions, such as cautious persona prompting and, in particular, requiring a clarifying question, can substantially recover performance. Our findings highlight a persistent gap between reasoning competence and behavioral compliance in current MLLMs, and suggest practical strategies for making these systems more trustworthy in underconstrained environments.

</details>

---

## 53. Beyond Text: Unveiling Privacy Vulnerabilities in Multi-modal Retrieval-Augmented Generation

- [ ] Beyond Text: Unveiling Privacy Vulnerabilities in Multi-modal Retrieval-Augmented Generation | https://aclanthology.org/2025.emnlp-main.1259/

- **Link**: https://aclanthology.org/2025.emnlp-main.1259/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Retrieval-Augmented Generation (MRAG) systems enhance LMMs by integrating external multimodal databases, but introduce unexplored privacy vulnerabilities. While text-based RAG privacy risks have been studied, multimodal data presents unique challenges. We provide the first systematic analysis of MRAG privacy vulnerabilities across vision-language and speech-language modalities. Using a novel compositional structured prompt attack in a black-box setting, we demonstrate how attackers can extract private information by manipulating queries. Our experiments reveal that LMMs can both directly generate outputs resembling retrieved content and produce descriptions that indirectly expose sensitive information, highlighting the urgent need for robust privacy-preserving MRAG techniques.

</details>

---

## 54. Pixels Versus Priors: Controlling Knowledge Priors in Vision-Language Models through Visual Counterfacts

- [ ] Pixels Versus Priors: Controlling Knowledge Priors in Vision-Language Models through Visual Counterfacts | https://aclanthology.org/2025.emnlp-main.1262/

- **Link**: https://aclanthology.org/2025.emnlp-main.1262/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) perform well on tasks such as visual question answering, but it remains unclear whether their reasoning relies more on memorized world knowledge or on the visual information present in the input image. To investigate this, we introduce Visual CounterFact, a new dataset of visually-realistic counterfactuals that put world knowledge priors (e.g, red strawberry) into direct conflict with visual input (e.g, blue strawberry). Using Visual CounterFact, we show that model predictions initially reflect memorized priors, but shift toward visual evidence in mid-to-late layers. This dynamic reveals a competition between the two modalities, with visual input ultimately overriding priors during evaluation. To control this behavior, we propose Pixels Versus Priors (PvP) steering vectors, a mechanism for controlling model outputs toward either world knowledge or visual input through activation-level interventions. On average, PvP successfully shifts 99.3% of color and 80.8% of size predictions from priors to counterfactuals. Together, these findings offer new tools for interpreting and controlling factual behavior in multimodal models.

</details>

---

## 55. WebMMU: A Benchmark for Multimodal Multilingual Website Understanding and Code Generation

- [ ] WebMMU: A Benchmark for Multimodal Multilingual Website Understanding and Code Generation | https://aclanthology.org/2025.emnlp-main.1276/

- **Link**: https://aclanthology.org/2025.emnlp-main.1276/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present WebMMU, a multilingual benchmark that evaluates three core web tasks: (1) website visual question answering, (2) code editing involving HTML/CSS/JavaScript, and (3) mockup-to-code generation. Unlike prior benchmarks that treat these tasks separately, WebMMU unifies them using expert-annotated, real-world web data to assess models’ abilities in complex multi-step reasoning, precise element grounding, and functional UI comprehension and coding. Our evaluation shows that while multimodal large language models (MLLMs) perform well on basic information extraction, they struggle with reasoning and grounding, editing code to preserve functionality, and generating design-to-code that maintains hierarchy and supports multilingual content. These findings reveal key limitations in current MLLMs and underscore the need for improved multimodal and cross-lingual reasoning to build future web agents capable of automating diverse web development tasks.

</details>

---

## 56. Corrupted but Not Broken: Understanding and Mitigating the Negative Impacts of Corrupted Data in Visual Instruction Tuning

- [ ] Corrupted but Not Broken: Understanding and Mitigating the Negative Impacts of Corrupted Data in Visual Instruction Tuning | https://aclanthology.org/2025.emnlp-main.1317/

- **Link**: https://aclanthology.org/2025.emnlp-main.1317/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Instruction Tuning (VIT) aims to enhance Multimodal Large Language Models (MLLMs), yet its effectiveness is often compromised by corrupted datasets with issues such as hallucinated content, incorrect responses, and poor OCR quality. Previous approaches to address these challenges have focused on refining datasets through high-quality data collection or rule-based filtering that can be costly or limited in scope. In this paper, we conduct a systematic investigation into the impact of corrupted data on MLLMs and discover that, although corrupted data degrade model performance, such adverse effects are largely reversible, and MLLMs arecorrupted but not broken. Specifically, we find that disabling a small subset of parameters can almost fully restore performance. Moreover, corrupted MLLMs inherently possess the capability to differentiate between clean and corrupted samples, facilitating dataset cleaning without external intervention. Building on these insights, we introduce a corruption-robust training paradigm that significantly surpasses existing strategies for mitigating the effects of corrupted data.

</details>

---

## 57. Jigsaw-Puzzles: From Seeing to Understanding to Reasoning in Vision-Language Models

- [ ] Jigsaw-Puzzles: From Seeing to Understanding to Reasoning in Vision-Language Models | https://aclanthology.org/2025.emnlp-main.1320/

- **Link**: https://aclanthology.org/2025.emnlp-main.1320/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spatial reasoning is a core component of human cognition, enabling individuals to perceive, comprehend, and interact with the physical world. It relies on a nuanced understanding of spatial structures and inter-object relationships, serving as the foundation for complex reasoning and decision-making. To investigate whether current vision-language models (VLMs) exhibit similar capability, we introduce Jigsaw-Puzzles, a novel benchmark consisting of 1,100 carefully curated real-world images with high spatial complexity. Based on this dataset, we design five tasks to rigorously evaluate VLMs’ spatial perception, structural understanding, and reasoning capabilities, while deliberately minimizing reliance on domain-specific knowledge to better isolate and assess the general spatial reasoning capability. We conduct a comprehensive evaluation across 24 state-of-the-art VLMs. The results show that even the strongest model, Gemini-2.5-Pro, achieves only 77.14% overall accuracy and performs particularly poorly on the Order Generation task, with only 30.00% accuracy, far below the 90%+ performance achieved by human participants. This persistent gap underscores the need for continued progress, positioning Jigsaw-Puzzles as a challenging and diagnostic benchmark for advancing spatial reasoning research in VLMs. Our project page is at https://zesen01.github.io/jigsaw-puzzles.

</details>

---

## 58. Zero-shot Multimodal Document Retrieval via Cross-modal Question Generation

- [ ] Zero-shot Multimodal Document Retrieval via Cross-modal Question Generation | https://aclanthology.org/2025.emnlp-main.1324/

- **Link**: https://aclanthology.org/2025.emnlp-main.1324/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Rapid advances in Multimodal Large Language Models (MLLMs) have extended information retrieval beyond text, enabling access to complex real-world documents that combine both textual and visual content. However, most documents are private, either owned by individuals or confined within corporate silos, and current retrievers struggle when faced with unseen domains or languages. To address this gap, we introduce PREMIR, a simple yet effective framework that leverages the broad knowledge of an MLLM to generate cross-modal pre-questions (preQs) before retrieval. Unlike earlier multimodal retrievers that embed entire documents as a single vector, PREMIR leverages preQs, decomposed from documents into finer token-level representations across modalities, enabling richer contextual understanding. Experiments show that PREMIR achieves state-of-the-art performance on out-of-distribution benchmarks, including closed-domain and multilingual settings, outperforming strong baselines across all metrics. We confirm the contribution of each component through in-depth ablation studies, and qualitative analyses of the generated preQs further highlight the framework’s robustness in real-world settings.

</details>

---

## 59. DiMo-GUI: Advancing Test-time Scaling inGUIGrounding via Modality-Aware Visual Reasoning

- [ ] DiMo-GUI: Advancing Test-time Scaling inGUIGrounding via Modality-Aware Visual Reasoning | https://aclanthology.org/2025.emnlp-main.1334/

- **Link**: https://aclanthology.org/2025.emnlp-main.1334/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Grounding natural language queries in graphical user interfaces (GUIs) poses unique challenges due to the diversity of visual elements, spatial clutter, and the ambiguity of language. In this paper, we introduce DiMo-GUI, a training-free framework for GUI grounding that leverages two core strategies: dynamic visual grounding and modality-aware optimization. Instead of treating the GUI as a monolithic image, our method splits the input into textual elements and iconic elements, allowing the model to reason over each modality independently using general-purpose vision-language models. When predictions are ambiguous or incorrect, DiMo-GUI dynamically focuses attention by generating candidate focal regions centered on the model’s initial predictions and incrementally zooms into subregions to refine the grounding result. This hierarchical refinement process helps disambiguate visually crowded layouts without the need for additional training or annotations. We evaluate our approach on standard GUI grounding benchmarks and demonstrate consistent improvements over baseline inference pipelines, highlighting the effectiveness of combining modality separation with region-focused reasoning.

</details>

---

## 60. VLA-Mark: A cross modal watermark for large vision-language alignment models

- [ ] VLA-Mark: A cross modal watermark for large vision-language alignment models | https://aclanthology.org/2025.emnlp-main.1342/

- **Link**: https://aclanthology.org/2025.emnlp-main.1342/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models demand watermarking solutions that protect intellectual property without compromising multimodal coherence. Existing text watermarking methods disrupt visual-textual alignment through biased token selection and static strategies, leaving semantic-critical concepts vulnerable. We propose VLA-Mark, a vision-aligned framework that embeds detectable watermarks while preserving semantic fidelity through cross-modal coordination. Our approach integrates multiscale visual-textual alignment metrics, combining localized patch affinity, global semantic coherence, and contextual attention patterns, to guide watermark injection without model retraining. An entropy-sensitive mechanism dynamically balances watermark strength and semantic preservation, prioritizing visual grounding during low-uncertainty generation phases. Experiments show 7.4% lower PPL and 26.6% higher BLEU than conventional methods, with near-perfect detection (98.8% AUC). The framework demonstrates 96.1% attack resilience against attacks such as paraphrasing and synonym substitution, while maintaining text-visual consistency, establishing new standards for quality-preserving multimodal watermarking.

</details>

---

## 61. COCO-Tree: Compositional Hierarchical Concept Trees for Enhanced Reasoning in Vision-Language Models

- [ ] COCO-Tree: Compositional Hierarchical Concept Trees for Enhanced Reasoning in Vision-Language Models | https://aclanthology.org/2025.emnlp-main.135/

- **Link**: https://aclanthology.org/2025.emnlp-main.135/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Compositional reasoning remains a persistent weakness of modern vision language models (VLMs): they often falter when a task hinges on understanding how multiple objects, attributes, and relations interact within an image. Multiple research works have attempted to improve compositionality performance by creative tricks such as improving prompt structure, chain of thought reasoning, etc. A more recent line of work attempts to impart additional reasoning in VLMs using well-trained Large Language Models (LLMs), which are far superior in linguistic understanding than VLMs to compensate for the limited linguistic prowess of VLMs. However, these approaches are either resource-intensive or do not provide an interpretable reasoning process. In this paper, we present “COCO-Tree” - a novel approach that augments VLM outputs with carefully designed neurosymbolic concept trees learned from LLMs to improve VLM’s linguistic reasoning. COCO-Tree’s beam search-inspired reasoning process boosts compositionality performance and provides a rationale behind VLM predictions. Empirical results on four compositionality benchmarks, Winoground, EqBench, ColorSwap, and SugarCrepe, in seven different open-source VLMs with varying sizes, demonstrate that COCO-Tree significantly improves compositional generalization by 5-10% over baselines.

</details>

---

## 62. 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark

- [ ] 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark | https://aclanthology.org/2025.emnlp-main.1353/

- **Link**: https://aclanthology.org/2025.emnlp-main.1353/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Though Large Vision-Language Models (LVLMs) are being actively explored in medicine, their ability to conduct complex real-world telemedicine consultations combining accurate diagnosis with professional dialogue remains underexplored. This paper presents3MDBench(MedicalMultimodalMulti-agentDialogueBenchmark), an open-source framework for simulating and evaluating LVLM-driven telemedical consultations. 3MDBench simulates patient variability through temperament-based Patient Agent and evaluates diagnostic accuracy and dialogue quality via Assessor Agent. It includes 2996 cases across 34 diagnoses from real-world telemedicine interactions, combining textual and image-based data. The experimental study compares diagnostic strategies for widely used open and closed-source LVLMs. We demonstrate that multimodal dialogue with internal reasoning improves F1 score by 6.5% over non-dialogue settings, highlighting the importance of context-aware, information-seeking questioning. Moreover, injecting predictions from a diagnostic convolutional neural network into the LVLM’s context boosts F1 by up to 20%. Source code is available athttps://github.com/univanxx/3mdbench.

</details>

---

## 63. Enhancing Large Vision-Language Models with Ultra-Detailed Image Caption Generation

- [ ] Enhancing Large Vision-Language Models with Ultra-Detailed Image Caption Generation | https://aclanthology.org/2025.emnlp-main.1357/

- **Link**: https://aclanthology.org/2025.emnlp-main.1357/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-quality image captions are essential for improving modality alignment and visual understanding in Large Vision-Language Models (LVLMs). However, the scarcity of ultra-detailed image caption data limits further advancements. This paper presents a systematic pipeline for generating high-quality, ultra-detailed image captions, encompassing both pre-processing and post-processing stages. In the pre-processing stage, we classify and deduplicate images, extract visual information using expert tools, and leverage GPT-4o with structured prompts to generate initial captions. To enhance comprehensiveness, we introduce an expansion strategy based on Large Language Models (LLMs), defining eight descriptive dimensions to refine and extend captions, which serve as seed data for training a proprietary captioner model. In the post-processing stage, we incorporate human error-correction annotations and an active learning-inspired approach to refine low-quality samples. Using high-quality corrected data, we apply Direct Preference Optimization (DPO) and develop a critic-rewrite pipeline, training a sentence-level critic model to mitigate hallucinations. Experimental results demonstrate that our ultra-detailed captions significantly enhance LVLMs’ perception and cognitive abilities across multiple vision-language benchmarks. The code and dataset are available at https://github.com/yuzeng0-0/UltraCaption.

</details>

---

## 64. iVISPAR— An Interactive Visual-Spatial Reasoning Benchmark forVLMs

- [ ] iVISPAR— An Interactive Visual-Spatial Reasoning Benchmark forVLMs | https://aclanthology.org/2025.emnlp-main.1359/

- **Link**: https://aclanthology.org/2025.emnlp-main.1359/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) are known to struggle with spatial reasoning and visual alignment. To help overcome these limitations, we introduce iVISPAR, an interactive multimodal benchmark designed to evaluate the spatial reasoning capabilities of VLMs acting as agents. iVISPAR is based on a variant of the sliding tile puzzle—a classic problem that demands logical planning, spatial awareness, and multi-step reasoning. The benchmark supports visual 3D, 2D, and text-based input modalities, enabling comprehensive assessments of VLMs’ planning and reasoning skills. We evaluate a broad suite of state-of-the-art open-source and closed-source VLMs, comparing their performance while also providing optimal path solutions and a human baseline to assess the task’s complexity and feasibility for humans. Results indicate that while VLMs perform better on 2D tasks compared to 3D or text-based settings, they struggle with complex spatial configurations and consistently fall short of human performance, illustrating the persistent challenge of visual alignment. This underscores critical gaps in current VLM capabilities, highlighting their limitations in achieving human-level cognition. Project website: https://microcosm.ai/ivispar.

</details>

---

## 65. Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance

- [ ] Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance | https://aclanthology.org/2025.emnlp-main.1367/

- **Link**: https://aclanthology.org/2025.emnlp-main.1367/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language-Action (VLA) models have made substantial progress by leveraging the robust capabilities of Visual Language Models (VLMs). However, VLMs’ significant parameter size and autoregressive (AR) decoding nature impose considerable computational demands on VLA models. While Speculative Decoding (SD) has shown efficacy in accelerating Large Language Models (LLMs) by incorporating efficient drafting and parallel verification, allowing multiple tokens to be generated in one forward pass, its application to VLA models remains unexplored. This work introduces Spec-VLA, an SD framework designed to accelerate VLA models. Due to the difficulty of the action prediction task and the greedy decoding mechanism of the VLA models, the direct application of the advanced SD framework to the VLA prediction task yields a minor speed improvement. To boost the generation speed, we propose an effective mechanism to relax acceptance utilizing the relative distances represented by the action tokens of the VLA model. Empirical results across diverse test scenarios affirm the effectiveness of the Spec-VLA framework, and further analysis substantiates the impact of our proposed strategies, which enhance the acceptance length by 44%, achieving1.42×speedup compared with the OpenVLA baseline, without compromising the success rate. The success of the Spec-VLA framework highlights the potential for broader application of speculative execution in VLA prediction scenarios.

</details>

---

## 66. CAVE: Detecting and Explaining Commonsense Anomalies in Visual Environments

- [ ] CAVE: Detecting and Explaining Commonsense Anomalies in Visual Environments | https://aclanthology.org/2025.emnlp-main.1379/

- **Link**: https://aclanthology.org/2025.emnlp-main.1379/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Humans can naturally identify, reason about, and explain anomalies in their environment. In computer vision, this long-standing challenge remains limited to industrial defects or unrealistic, synthetically generated anomalies, failing to capture the richness and unpredictability of real-world anomalies. In this work, we introduce CAVE, the first benchmark of real-world visual anomalies. CAVE supports three open-ended tasks: anomaly description, explanation, and justification; with fine-grained annotations for visual grounding and categorizing anomalies based on their visual manifestations, their complexity, severity, and commonness. These annotations draw inspiration from cognitive science research on how humans identify and resolve anomalies, providing a comprehensive framework for evaluating Vision-Language Models (VLMs) in detecting and understanding anomalies. We show that state-of-the-art VLMs struggle with visual anomaly perception and commonsense reasoning, even with advanced prompting strategies. By offering a realistic and cognitively grounded benchmark, CAVE serves as a valuable resource for advancing research in anomaly detection and commonsense reasoning in VLMs.

</details>

---

## 67. SemVink: AdvancingVLMs’ Semantic Understanding of Optical Illusions via Visual Global Thinking

- [ ] SemVink: AdvancingVLMs’ Semantic Understanding of Optical Illusions via Visual Global Thinking | https://aclanthology.org/2025.emnlp-main.1381/

- **Link**: https://aclanthology.org/2025.emnlp-main.1381/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) excel in semantic tasks but falter at a core human capability: detecting hidden content in optical illusions or AI-generated images through perceptual adjustments like zooming. We introduce HC-Bench, a benchmark of 112 images with hidden texts, objects, and illusions, revealing that leading VLMs achieve near-zero accuracy (0–5.36%) even with explicit prompting. Humans resolve such ambiguities instinctively, yet VLMs fail due to an overreliance on high-level semantics. Strikingly, we propose SemVink (Semantic Visual Thinking) by simply scaling images to low resolutions, which unlocks over 99% accuracy by eliminating redundant visual noise. This exposes a critical architectural flaw: VLMs prioritize abstract reasoning over low-level visual operations crucial for real-world robustness. Our work urges a shift toward hybrid models integrating multi-scale processing, bridging the gap between computational vision and human cognition for applications in medical imaging, security, and beyond.

</details>

---

## 68. Randomized Smoothing Meets Vision-Language Models

- [ ] Randomized Smoothing Meets Vision-Language Models | https://aclanthology.org/2025.emnlp-main.1396/

- **Link**: https://aclanthology.org/2025.emnlp-main.1396/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Randomized smoothing (RS) is one of the prominent techniques to ensure the correctness of machine learning models, where point-wise robustness certificates can be derived analytically. While RS is well understood for classification, its application to generative models is unclear, since their outputs are sequences rather than labels. We resolve this by connecting generative outputs to an oracle classification task and showing that RS can still be enabled: the final response can be classified as a discrete action (e.g., service-robot commands in VLAs), as harmful vs. harmless (content moderation or toxicity detection in VLMs), or even applying oracles to cluster answers into semantically equivalent ones. Provided that the error rate for the oracle classifier comparison is bounded, we develop the theory that associates the number of samples with the corresponding robustness radius. We further derive improved scaling laws analytically relating the certified radius and accuracy to the number of samples, showing that the earlier result of 2 to 3 orders of magnitude fewer samples sufficing with minimal loss remains valid even under weaker assumptions. Together, these advances make robustness certification both well-defined and computationally feasible for state-of-the-art VLMs, as validated against recent jailbreak-style adversarial attacks.

</details>

---

## 69. Gradient-Attention Guided Dual-Masking Synergetic Framework for Robust Text-based Person Retrieval

- [ ] Gradient-Attention Guided Dual-Masking Synergetic Framework for Robust Text-based Person Retrieval | https://aclanthology.org/2025.emnlp-main.14/

- **Link**: https://aclanthology.org/2025.emnlp-main.14/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although Contrastive Language-Image Pre-training (CLIP) exhibits strong performance across diverse vision tasks, its application to person representation learning faces two critical challenges: (i) the scarcity of large-scale annotated vision-language data focused on person-centric images, and (ii) the inherent limitations of global contrastive learning, which struggles to maintain discriminative local features crucial for fine-grained matching while remaining vulnerable to noisy text tokens. This work advances CLIP for person representation learning through synergistic improvements in data curation and model architecture. First, we develop a noise-resistant data construction pipeline that leverages the in-context learning capabilities of MLLMs to automatically filter and caption web-sourced images. This yields WebPerson, a large-scale dataset of 5M high-quality person-centric image-text pairs. Second, we introduce the GA-DMS (Gradient-Attention Guided Dual-Masking Synergetic) framework, which improves cross-modal alignment by adaptively masking noisy textual tokens based on the gradient-attention similarity score. Additionally, we incorporate masked token prediction objectives that compel the model to predict informative text tokens, enhancing fine-grained semantic representation learning. Extensive experiments show that GA-DMS achieves state-of-the-art performance across multiple benchmarks. The data and pre-trained models are released at https://github.com/Multimodal-Representation-Learning-MRL/GA-DMS.

</details>

---

## 70. GLIMPSE: Do Large Vision-Language Models Truly Think With Videos or Just Glimpse at Them?

- [ ] GLIMPSE: Do Large Vision-Language Models Truly Think With Videos or Just Glimpse at Them? | https://aclanthology.org/2025.emnlp-main.1415/

- **Link**: https://aclanthology.org/2025.emnlp-main.1415/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing video benchmarks often resemble image-based benchmarks, with question types like “What actions does the person perform throughout the video?” or “What color is the woman’s dress in the video?” For these, models can often answer by scanning just a few key frames, without deep temporal reasoning. This limits our ability to assess whether large vision-language models (LVLMs) can truly think with videos rather than perform superficial frame-level analysis. To address this, we introduce , a benchmark specifically designed to evaluate whether LVLMs can genuinely think with videos. Unlike prior benchmarks, emphasizes comprehensive video understanding beyond static image cues. It consists of 3,269 videos and over 4,342 highly visual-centric questions across 11 categories, including Trajectory Analysis, Temporal Reasoning, and Forensics Detection. All questions are carefully crafted by human annotators and require watching the entire video and reasoning over full video context—this is what we mean by thinking with video. These questions cannot be answered by scanning selected frames or relying on text alone. In human evaluations, achieves 94.82% accuracy, but current LVLMs face significant challenges. Even the best-performing model, GPT-o3, reaches only 66.43%, highlighting that LVLMs still struggle to move beyond surface-level reasoning to truly think with videos. We publicly release our benchmark and code at https://github.com/aiming-lab/GLIMPSE.

</details>

---

## 71. VLP: Vision-Language Preference Learning for Embodied Manipulation

- [ ] VLP: Vision-Language Preference Learning for Embodied Manipulation | https://aclanthology.org/2025.emnlp-main.1444/

- **Link**: https://aclanthology.org/2025.emnlp-main.1444/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reward engineering is one of the key challenges in Reinforcement Learning (RL). Preference-based RL effectively addresses this issue by learning from human feedback. However, it is both time-consuming and expensive to collect human preference labels. In this paper, we propose a novelVision-LanguagePreference learning framework, namedVLP, which learns a vision-language preference model to provide feedback for embodied manipulation tasks. To achieve this, we define three types of language-conditioned preferences and construct a vision-language preference dataset, which contains versatile implicit preference orders. The model learns to extract language-related features, and then serves as a predictor in various downstream tasks. The policy can be learned according to the annotated labels via reward learning or direct policy optimization. Extensive empirical results on simulated embodied manipulation tasks demonstrate that our method provides accurate preferences and generalizes to unseen tasks and unseen language instructions, outperforming the baselines by a large margin and shifting the burden from continuous, per-task human annotation to one-time, per-domain data collection.

</details>

---

## 72. EGOILLUSION: Benchmarking Hallucinations in Egocentric Video Understanding

- [ ] EGOILLUSION: Benchmarking Hallucinations in Egocentric Video Understanding | https://aclanthology.org/2025.emnlp-main.1446/

- **Link**: https://aclanthology.org/2025.emnlp-main.1446/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable performance in complex multimodal tasks. While MLLMs excel at visual perception and reasoning in third-person and egocentric videos, they are prone to hallucinations, generating coherent yet inaccurate responses. We present EGOILLUSION, a first benchmark to evaluate MLLM hallucinations in egocentric videos. EGOILLUSION comprises 1,400 videos paired with 8,000 human-annotated open and closed-ended questions designed to trigger hallucinations in both visual and auditory cues in egocentric videos. Evaluations across ten MLLMs reveal significant challenges, including powerful models like GPT-4o and Gemini, achieving only 59% accuracy. EGOILLUSION lays the foundation in developing robust benchmarks to evaluate the effectiveness of MLLMs and spurs the development of better egocentric MLLMs with reduced hallucination rates. Our benchmark will be open-sourced for reproducibility

</details>

---

## 73. SimpleDoc:Multi‐Modal Document Understanding withDual‐Cue Page Retrieval and Iterative Refinement

- [ ] SimpleDoc:Multi‐Modal Document Understanding withDual‐Cue Page Retrieval and Iterative Refinement | https://aclanthology.org/2025.emnlp-main.1443/

- **Link**: https://aclanthology.org/2025.emnlp-main.1443/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Document Visual Question Answering (DocVQA) is a practical yet challenging task, which is to ask questions based on documents while referring to multiple pages and different modalities of information, e.g., images and tables. To handle multi-modality, recent methods follow a similar Retrieval Augmented Generation (RAG) pipeline, but utilize Visual Language Models (VLMs) based embedding model to embed and retrieve relevant pages as images, and generate answers with VLMs that can accept an image as input. In this paper, we introduce SimpleDoc, a lightweight yet powerful retrieval - augmented framework for DocVQA. It boosts evidence page gathering by first retrieving candidates through embedding similarity and then filtering and re-ranking these candidates based on page summaries. A single VLM-based reasoner agent repeatedly invokes this dual-cue retriever, iteratively pulling fresh pages into a working memory until the question is confidently answered. SimpleDoc outperforms previous baselines by 3.2% on average on 4 DocVQA datasets with much fewer pages retrieved. Our code is available at https://github.com/ag2ai/SimpleDoc.

</details>

---

## 74. QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models

- [ ] QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models | https://aclanthology.org/2025.emnlp-main.1445/

- **Link**: https://aclanthology.org/2025.emnlp-main.1445/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Multimodal Large Language Models (MLLMs) encounter two key issues in multi-image contexts: (1) a lack of fine-grained perception across disparate images, and (2) a diminished capability to effectively reason over and synthesize information from multiple visual inputs. However, while various prompting methods aim to describe visual content, many existing studies focus primarily on single-image settings or specific, constrained scenarios. This leaves a critical gap in understanding and addressing how MLLMs tackle more general and complex multi-image reasoning tasks. Thus, we first extensively investigate how current prompting methods perceive fine-grained visual details and process visual information when dealing with multiple images. Our findings reveal that existing prompting methods fall short in attending to needed clues and seamlessly integrating perception and reasoning. Inspired by the findings, we propose a new zero-shot prompting method, Question-Guided Chain-of-Captions (QG-CoC), a generalized prompting approach that effectively handles problems with an arbitrary number of images. We evaluate our method on various open-source and closed-source MLLMs for multi-image and single-image benchmarks. Experimental results indicate that QG-CoC demonstrates competitive performance across tasks and exhibits robust improvements in the challenging scenarios where existing prompting methods fail.

</details>

---

## 75. Multi-Frequency Contrastive Decoding: Alleviating Hallucinations for Large Vision-Language Models

- [ ] Multi-Frequency Contrastive Decoding: Alleviating Hallucinations for Large Vision-Language Models | https://aclanthology.org/2025.emnlp-main.1452/

- **Link**: https://aclanthology.org/2025.emnlp-main.1452/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large visual-language models (LVLMs) have demonstrated remarkable performance in visual-language tasks. However, object hallucination remains a significant challenge for LVLMs. Existing studies attribute object hallucinations in LVLMs mainly to linguistic priors and data biases. We further explore the causes of object hallucinations from the perspective of frequency domain and reveal that insufficient frequency information in images amplifies these linguistic priors, increasing the likelihood of hallucinations. To mitigate this issue, we propose the Multi-Frequency Contrastive Decoding (MFCD) method, a simple yet trainingfree approach that removes the hallucination distribution in the original output distribution, which arises from LVLMs neglecting the high-frequency information or low-frequency information in the image input. Without compromising the general capabilities of LVLMs, the proposed MFCD effectively mitigates the object hallucinations in LVLMs. Our experiments demonstrate that MFCD significantly mitigates object hallucination across diverse large-scale vision-language models, without requiring additional training or external tools. In addition, MFCD can be applied to various LVLMs without modifying model architecture or requiring additional training, demonstrating its generality and robustness. Codes are available at https://github.com/liubq-dev/mfcd.

</details>

---

## 76. MAviS: A Multimodal Conversational Assistant For Avian Species

- [ ] MAviS: A Multimodal Conversational Assistant For Avian Species | https://aclanthology.org/2025.emnlp-main.1455/

- **Link**: https://aclanthology.org/2025.emnlp-main.1455/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fine-grained understanding and species-specific, multimodal question answering are vital for advancing biodiversity conservation and ecological monitoring. However, existing multimodal large language models (MM-LLMs) face challenges when it comes to specialized topics like avian species, making it harder to provide accurate and contextually relevant information in these areas. To address this limitation, we introduce the **MAviS-Dataset**, a large-scale multimodal avian species dataset that integrates image, audio, and text modalities for over 1,000 bird species, comprising both pretraining and instruction-tuning subsets enriched with structured question–answer pairs. Building on the MAviS-Dataset, we introduce **MAviS-Chat**, a multimodal LLM that supports audio, vision, and text designed for fine-grained species understanding, multimodal question answering, and scene-specific description generation. Finally, for quantitative evaluation, we present **MAviS-Bench**, a benchmark of over 25,000 Q&A pairs designed to assess avian species-specific perceptual and reasoning abilities across modalities. Experimental results show that MAviS-Chat outperforms the baseline MiniCPM-o-2.6 by a large margin, achieving state-of-the-art open-source results and demonstrating the effectiveness of our instruction-tuned MAviS-Dataset. Our findings highlight the necessity of domain-adaptive MM-LLMs for ecological applications. Our code, training data, evaluation benchmark, and models are available at https://github.com/yevheniia-uv/MAviS.

</details>

---

## 77. Detecting Knowledge Boundary of Vision Large Language Models by Sampling-Based Inference

- [ ] Detecting Knowledge Boundary of Vision Large Language Models by Sampling-Based Inference | https://aclanthology.org/2025.emnlp-main.1458/

- **Link**: https://aclanthology.org/2025.emnlp-main.1458/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the advancements made in Vision Large Language Models (VLLMs), like text Large Language Models (LLMs), they have limitations in addressing questions that require real-time information or are knowledge-intensive. Indiscriminately adopting Retrieval Augmented Generation (RAG) techniques is an effective yet expensive way to enable models to answer queries beyond their knowledge scopes. To mitigate the dependence on retrieval and simultaneously maintain, or even improve, the performance benefits provided by retrieval, we propose a method to detect the knowledge boundary of VLLMs, allowing for more efficient use of techniques like RAG. Specifically, we propose a method with two variants that fine-tune a VLLM on an automatically constructed dataset for boundary identification. Experimental results on various types of Visual Question Answering datasets show that our method successfully depicts a VLLM’s knowledge boundary, based on which we are able to reduce indiscriminate retrieval while maintaining or improving the performance. In addition, we show that the knowledge boundary identified by our method for one VLLM can be used as a surrogate boundary for other VLLMs. Code will be released at https://github.com/Chord-Chen-30/VLLM-KnowledgeBoundary

</details>

---

## 78. From Charts to Fair Narratives: Uncovering and Mitigating Geo-Economic Biases in Chart-to-Text

- [ ] From Charts to Fair Narratives: Uncovering and Mitigating Geo-Economic Biases in Chart-to-Text | https://aclanthology.org/2025.emnlp-main.1472/

- **Link**: https://aclanthology.org/2025.emnlp-main.1472/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Charts are very common for exploring dataand communicating insights, but extracting key takeaways from charts and articulating them in natural language can be challenging. The chart-to-text task aims to automate this process by generating textual summaries of charts. While with the rapid advancement of large Vision-Language Models (VLMs), we have witnessed great progress in this domain, little to no attention has been given to potential biases in their outputs. This paper investigates how VLMs can amplify geo-economic biases when generating chart summaries, potentially causing societal harm. Specifically, we conduct a large-scale evaluation of geo-economic biases in VLM-generated chart summaries across 6,000 chart-country pairs from six widely used proprietary and open-source models to understand how a country’s economic status influences the sentiment of generated summaries. Our analysis reveals that existing VLMs tend to produce more positive descriptions for high-income countries compared to middle- or low-income countries, even when country attribution is the only variable changed. We also find that models such as GPT-4o-mini, Gemini-1.5-Flash, and Phi-3.5 exhibit varying degrees of bias. We further explore inference-time prompt-based debiasing techniques using positive distractors but find them only partially effective, underscoring the complexity of the issue and the need for more robust debiasing strategies. Our code and dataset are available at <redacted>.

</details>

---

## 79. M2Edit: Locate and Edit Multi-Granularity Knowledge in Multimodal Large Language Model

- [ ] M2Edit: Locate and Edit Multi-Granularity Knowledge in Multimodal Large Language Model | https://aclanthology.org/2025.emnlp-main.1478/

- **Link**: https://aclanthology.org/2025.emnlp-main.1478/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal knowledge editing is an important method for modifying outdated or incorrect knowledge in Multimodal Large Language Models (MLLMs). However, existing datasets for multimodal knowledge editing lack multi-granularity knowledge. In this paper, we present a more realistic dataset called M2Edit, which includes three distinct types of knowledge: entity, relation, and action. Additionally, existing knowledge editing methods for MLLMs lack the ability to handle multi-granularity knowledge and generalize to multimodal data. To address these limitations, we propose the multimodal knowledge editing method MLE. This approach identifies key knowledge layers within different components and collaboratively edits the various components of MLLMs. As a result, we observe significant improvements in visual generality performance, ranging from 4.8 to 10.8, and achieve the best overall performance on knowledge data of different granularities.

</details>

---

## 80. MemeIntel: Explainable Detection of Propagandistic and Hateful Memes

- [ ] MemeIntel: Explainable Detection of Propagandistic and Hateful Memes | https://aclanthology.org/2025.emnlp-main.1539/

- **Link**: https://aclanthology.org/2025.emnlp-main.1539/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The proliferation of multimodal content on social media presents significant challenges in understanding and moderating complex, context-dependent issues such as misinformation, hate speech, and propaganda. While efforts have been made to develop resources and propose new methods for automatic detection, limited attention has been given to label detection and the generation of explanation-based rationales for predicted labels. To address this challenge, we introduce MemeXplain, an explanation-enhanced dataset for propaganda memes in Arabic and hateful memes in English, making it the first large-scale resource for these tasks. To solve these tasks, we propose a novel multi-stage optimization approach and train Vision-Language Models (VLMs). Our results demonstrate that this approach significantly improves performance over the base model for both label detection and explanation generation, outperforming the current state-of-the-art with an absolute improvement of approximately 3% on ArMeme and 7% on Hateful Memes. For reproducibility and future research, we aim to make the MemeXplain dataset and scripts publicly available.

</details>

---

## 81. Are Vision-Language Models Safe in the Wild? A Meme-Based Benchmark Study

- [ ] Are Vision-Language Models Safe in the Wild? A Meme-Based Benchmark Study | https://aclanthology.org/2025.emnlp-main.1555/

- **Link**: https://aclanthology.org/2025.emnlp-main.1555/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Rapid deployment of vision-language models (VLMs) magnifies safety risks, yet most evaluations rely on artificial images. This study asks: How safe are current VLMs when confronted with meme images that ordinary users share? To investigate this question, we introduce MemeSafetyBench, a 50,430-instance benchmark pairing real meme images with both harmful and benign instructions. Using a comprehensive safety taxonomy and LLM-based instruction generation, we assess multiple VLMs across single and multi-turn interactions. We investigate how real-world memes influence harmful outputs, the mitigating effects of conversational context, and the relationship between model scale and safety metrics. Our findings demonstrate that VLMs are more vulnerable to meme-based harmful prompts than to synthetic or typographic images. Memes significantly increase harmful responses and decrease refusals compared to text-only inputs. Though multi-turn interactions provide partial mitigation, elevated vulnerability persists. These results highlight the need for ecologically valid evaluations and stronger safety mechanisms. MemeSafetyBench is publicly available at https://github.com/oneonlee/Meme-Safety-Bench.

</details>

---

## 82. Image Embedding Sampling Method for Diverse Captioning

- [ ] Image Embedding Sampling Method for Diverse Captioning | https://aclanthology.org/2025.emnlp-main.156/

- **Link**: https://aclanthology.org/2025.emnlp-main.156/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image Captioning for state-of-the-art VLMs has significantly improved over time; however, this comes at the cost of increased computational complexity, making them less accessible for resource-constrained applications such as mobile devices and assistive technologies. Alternatively, comparably smaller VLMs prioritize high-level scene descriptions, overlooking finer details that contribute to a richer understanding of an image. In this paper, we introduce a training-free framework that enhances caption diversity and informativeness by explicitly attending to distinct image regions using a comparably small VLM, BLIP, as the backbone. Our approach leverages structured segmentation to produce hierarchical representations that capture both global and localized semantics. Without requiring additional model training, we demonstrate that our method allows smaller VLMs to achieve performance comparable to larger models in terms of image-caption alignment, semantic integrity, and diversity. We evaluate our framework on MSCOCO, Flickr30k, and Nocaps test datasets, achieving a Div-2 score of 0.735, 0.750, and 0.748 for each dataset, respectively, while maintaining strong image-caption relevancy and semantic integrity with the human-annotated captions. Our code is available athttps://github.com/xfactlab/HBoP.

</details>

---

## 83. CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models

- [ ] CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models | https://aclanthology.org/2025.emnlp-main.1561/

- **Link**: https://aclanthology.org/2025.emnlp-main.1561/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability. Extending LLMs to incorporate visual inputs, large vision-language models (LVLMs) have shown impressive performance in tasks such as recognition and visual question answering (VQA). Despite increasing interest in the utility of LLMs in causal reasoning tasks such as causal discovery and counterfactual reasoning, there has been relatively little work showcasing the abilities of LVLMs on visual causal reasoning tasks. We take this opportunity to formally introduce a comprehensive causal reasoning benchmark for multi-modal in-context learning from LVLMs. Our CausalVLBench encompasses three representative tasks: causal structure inference, intervention target prediction, and counterfactual prediction. We evaluate the ability of state-of-the-art open-source LVLMs on our causal reasoning tasks across three causal representation learning datasets and demonstrate their fundamental strengths and weaknesses. We hope that our benchmark elucidates the drawbacks of existing vision-language models and motivates new directions and paradigms in improving the visual causal reasoning abilities of LVLMs.

</details>

---

## 84. UnifiedVisual: A Framework for Constructing Unified Vision-Language Datasets

- [ ] UnifiedVisual: A Framework for Constructing Unified Vision-Language Datasets | https://aclanthology.org/2025.emnlp-main.1572/

- **Link**: https://aclanthology.org/2025.emnlp-main.1572/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unified vision large language models (VLLMs) have recently achieved impressive advancements in both multimodal understanding and generation, powering applications such as visual question answering and text-guided image synthesis. However, progress in unified VLLMs remains constrained by the lack of datasets that fully exploit the synergistic potential between these two core abilities. Existing datasets typically address understanding and generation in isolation, thereby limiting the performance of unified VLLMs. To bridge this critical gap, we introduce a novel dataset construction framework,UnifiedVisual, and presentUnifiedVisual-240K, a high-quality dataset meticulously designed to facilitate mutual enhancement between multimodal understanding and generation. UnifiedVisual-240K seamlessly integrates diverse visual and textual inputs and outputs, enabling comprehensive cross-modal reasoning and precise text-to-image alignment. Our dataset encompasses a wide spectrum of tasks and data sources, ensuring rich diversity and addressing key shortcomings of prior resources. Extensive experiments demonstrate that models trained on UnifiedVisual-240K consistently achieve strong performance across a wide range of tasks. Notably, these models exhibit significant mutual reinforcement between multimodal understanding and generation, further validating the effectiveness of our framework and dataset. We believe UnifiedVisual represents a new growth point for advancing unified VLLMs and unlocking their full potential.

</details>

---

## 85. ProLongVid: A Simple but Strong Baseline for Long-context Video Instruction Tuning

- [ ] ProLongVid: A Simple but Strong Baseline for Long-context Video Instruction Tuning | https://aclanthology.org/2025.emnlp-main.1570/

- **Link**: https://aclanthology.org/2025.emnlp-main.1570/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video understanding is essential for multimodal large language models (MLLMs) to interact effectively with users and the real world. However, analyzing long videos remains a major challenge due to the lack of high-quality video instruction data and effective training strategies. In this paper, we introduce a simple yet effective baseline for long-context video understanding, including dataset construction and training recipes. We curate a large-scale video instruction dataset with over 1M samples, encompassing videos from a few seconds to several minutes across diverse sources, without any human annotations. Additionally, we propose a progressive video instruction tuning strategy that incrementally increases input context length, enabling better utilization of videos of varying durations. Comprehensive experiments demonstrate that our dataset significantly outperforms existing video instruction datasets for fine-tuning MLLMs. Furthermore, our training approach establishes a strong video MLLM baseline, surpassing previous open-source models on video benchmarks and outperforming proprietary models like GPT-4V and GPT-4o-mini on VideoMME, even with a compact 7B model.

</details>

---

## 86. Is Cognition Consistent with Perception? Assessing and Mitigating Multimodal Knowledge Conflicts in Document Understanding

- [ ] Is Cognition Consistent with Perception? Assessing and Mitigating Multimodal Knowledge Conflicts in Document Understanding | https://aclanthology.org/2025.emnlp-main.1574/

- **Link**: https://aclanthology.org/2025.emnlp-main.1574/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have shown impressive capabilities in document understanding, a rapidly growing research area with significant industrial demand. As a multimodal task, document understanding requires models to possess both perceptual and cognitive abilities. However, due to different types of annotation noise in training, current MLLMs often face conflicts between perception and cognition. Taking a document VQA task (cognition) as an example, an MLLM might generate answers that do not match the corresponding visual content identified by its OCR (perception). This conflict suggests that the MLLM might struggle to establish an intrinsic connection between the information it “sees” and what it “understands”. Such conflicts challenge the intuitive notion that cognition is consistent with perception, hindering the performance and explainability of MLLMs. In this paper, we define the conflicts between cognition and perception as Cognition and Perception (C&P) knowledge conflicts, a form of multimodal knowledge conflicts, and systematically assess them with a focus on document understanding. Our analysis reveals that even GPT-4o, a leading MLLM, achieves only 75.26% C&P consistency. To mitigate the C&P knowledge conflicts, we propose a novel method called Multimodal Knowledge Consistency Fine-tuning. Our method reduces C&P knowledge conflicts across all tested MLLMs and enhances their performance in both cognitive and perceptual tasks.

</details>

---

## 87. MMDocIR: Benchmarking Multimodal Retrieval for Long Documents

- [ ] MMDocIR: Benchmarking Multimodal Retrieval for Long Documents | https://aclanthology.org/2025.emnlp-main.1576/

- **Link**: https://aclanthology.org/2025.emnlp-main.1576/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal document retrieval aims to identify and retrieve various forms of multimodal content, such as figures, tables, charts, and layout information from extensive documents. Despite its increasing popularity, there is a notable lack of a comprehensive and robust benchmark to effectively evaluate the performance of systems in such tasks. To address this gap, this work introduces a new benchmark, named MMDocIR, that encompasses two distinct tasks: page-level and layout-level retrieval. The former evaluates the performance of identifying the most relevant pages within a long document, while the later assesses the ability of detecting specific layouts, providing a more fine-grained measure than whole-page analysis. A layout refers to a variety of elements, including textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring 1,685 questions annotated by experts and 173,843 questions with bootstrapped labels, making it a valuable resource in multimodal document retrieval for both training and evaluation. Through rigorous experiments, we demonstrate that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR training set effectively enhances the performance of multimodal document retrieval and (iii) text retrievers leveraging VLM-text significantly outperforms retrievers relying on OCR-text.

</details>

---

## 88. Waste-Bench: A Comprehensive Benchmark for EvaluatingVLLMs in Cluttered Environments

- [ ] Waste-Bench: A Comprehensive Benchmark for EvaluatingVLLMs in Cluttered Environments | https://aclanthology.org/2025.emnlp-main.1578/

- **Link**: https://aclanthology.org/2025.emnlp-main.1578/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Language Models (LLMs) have paved the way for VisionLarge Language Models (VLLMs) capable ofperforming a wide range of visual understand-ing tasks. While LLMs have demonstrated impressive performance on standard naturalimages, their capabilities have not been thoroughly explored in cluttered datasets where there is complex environment having deformedshaped objects. In this work, we introduce a novel dataset specifically designed for waste classification in real-world scenarios, character-ized by complex environments and deformed shaped objects. Along with this dataset, we present an in-depth evaluation approach to rig-orously assess the robustness and accuracy of VLLMs. The introduced dataset and comprehensive analysis provide valuable insights intothe performance of VLLMs under challenging conditions. Our findings highlight the critical need for further advancements in VLLM’s ro-bustness to perform better in complex enviroments. The dataset and code for our experiments are available at https://github.com/aliman80/wastebench.

</details>

---

## 89. DELOC: Document Element Localizer

- [ ] DELOC: Document Element Localizer | https://aclanthology.org/2025.emnlp-main.1585/

- **Link**: https://aclanthology.org/2025.emnlp-main.1585/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Editing documents and PDFs using natural language instructions is desirable for many reasons – ease of use, increasing accessibility to non-technical users, and for creativity. To do this automatically, a system needs to first understand the user’s intent and convert this to an executable plan or command, and then the system needs to identify or localize the elements that the user desires to edit. While there exist methods that can accomplish these tasks, a major bottleneck in these systems is the inability to ground the spatial edit location effectively. We address this gap through our proposed system, DELOC (Document Element LOCalizer). DELOC adapts the grounding capabilities of existing Multimodal Large Language Model (MLLM) from natural images to PDFs. This adaptation involves two novel contributions: 1) synthetically generating PDF-grounding instruction tuning data from partially annotated datasets; and 2) synthetic data cleaning via Code-NLI, an NLI-inspired process to clean data using generated Python code. The effectiveness of DELOC is apparent in the >3x zero-shot improvement it achieves over the next best Multimodal LLM, GPT-4o.

</details>

---

## 90. SHIFT: Selected Helpful Informative Frame for Video-guided Machine Translation

- [ ] SHIFT: Selected Helpful Informative Frame for Video-guided Machine Translation | https://aclanthology.org/2025.emnlp-main.161/

- **Link**: https://aclanthology.org/2025.emnlp-main.161/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video-guided Machine Translation (VMT) aims to improve translation quality by integrating contextual information from paired short video clips. Mainstream VMT approaches typically incorporate multimodal information by uniformly sampling frames from the input videos. However, this paradigm frequently incurs significant computational overhead and introduces redundant multimodal content, which degrades both efficiency and translation quality. To tackle these challenges, we propose SHIFT (Selected Helpful Informative Frame for Translation). It is a lightweight, plug-and-play framework designed for VMT with Multimodal Large Language Models (MLLMs). SHIFT adaptively selects a single informative key frame when visual context is necessary; otherwise, it relies solely on textual input. This process is guided by a dedicated clustering module and a selector module. Experimental results demonstrate that SHIFT enhances the performance of MLLMs on the VMT task while simultaneously reducing computational cost, without sacrificing generalization ability. The code will be released upon acceptance.

</details>

---

## 91. ProReason: Multi-Modal Proactive Reasoning with Decoupled Eyesight and Wisdom

- [ ] ProReason: Multi-Modal Proactive Reasoning with Decoupled Eyesight and Wisdom | https://aclanthology.org/2025.emnlp-main.1614/

- **Link**: https://aclanthology.org/2025.emnlp-main.1614/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have witnessed significant progress on visual understanding tasks. However, they often prioritize language knowledge over image information on visual reasoning tasks, incurring performance degradation. To tackle this issue, we first identify the drawbacks of existing solutions (i.e., limited multi-modal reasoning capacities, and insufficient and irrelevant visual descriptions). We then decompose visual reasoning process into two stages: proactive visual perception (i.e., eyesight) and textual reasoning (i.e., wisdom), and introduce a novel visual reasoning framework named ProReason. This framework features decoupled vision-reasoning capabilities and multi-run proactive perception. Briefly, given a multi-modal question, ProReason iterates proactive information collection and reasoning until the answer can be concluded with necessary and sufficient visual descriptions. Notably, the disassociation of capabilities allows seamless integration of existing large language models (LLMs) to compensate for the reasoning deficits of LVLMs. Our extensive experiments demonstrate that ProReason outperforms existing multi-step reasoning frameworks on various benchmarks for both open-source and closed-source models, with the average performance gain reaching 13.2%. Besides, the integration of LLMs allows ProReason to produce high-quality visual reasoning data, which empowers ProReason-distilled models (i.e., ProReason-VL and ProReason-Q3) to achieve superior performance in downstream tasks. Our insights into existing solutions and the decoupled perspective for feasible integration of LLMs illuminate future research on visual reasoning techniques, especially LLM-assisted ones. The code is available at https://github.com/lian-tian-mo-zun/Pro_Reason.

</details>

---

## 92. When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using SmallVLMs

- [ ] When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using SmallVLMs | https://aclanthology.org/2025.emnlp-main.1613/

- **Link**: https://aclanthology.org/2025.emnlp-main.1613/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (L-VLMs) have demonstrated remarkable performance in various vision and language tasks, including Visual Question Answering (VQA). However, their high computational cost makes them impractical for resource-constrained settings and inference-heavy applications. In contrast, Small Vision-Language Models (S-VLMs) offer efficiency but suffer from a significant performance gap compared to their larger counterparts. In this work, we introduce the Model Parity Aligner (MPA), a novel framework designed to systematically improve S-VLMs by leveraging unlabeled images and effective knowledge transfer from L-VLMs. Instead of traditional knowledge distillation methods that rely on labeled training data, MPA employs a strategic parity-based approach that precisely identifies the knowledge disparities between S-VLMs and L-VLMs, and optimizes training by targeting only these disparities. We conduct extensive experiments on four diverse VQA benchmarks, namely TextVQA, ST-VQA, ChartQA, and OKVQA, each of which required specialized reasoning capabilities such as text recognition, chart interpretation, and commonsense and factual understanding. Our results demonstrate that MPA consistently enhances the performance of S-VLM on all benchmarks, reducing the performance gap while maintaining computational efficiency. We shall make our code and MPA-aligned models publicly available upon acceptance of this work.

</details>

---

## 93. VEHME: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions

- [ ] VEHME: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions | https://aclanthology.org/2025.emnlp-main.1619/

- **Link**: https://aclanthology.org/2025.emnlp-main.1619/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automatically assessing handwritten mathematical solutions is an important problem in educational technology with practical applications, but remains a significant challenge due to the diverse formats, unstructured layouts, and symbolic complexity of student work. To address this challenge, we introduce VEHME-aVision-Language Model forEvaluatingHandwrittenMathematicsExpressions—designed to assess open-form handwritten math responses with high accuracy and interpretable reasoning traces. VEHME integrates a two-phase training pipeline: (i) supervised fine-tuning using structured reasoning data, and (ii) reinforcement learning that aligns model outputs with multi-dimensional grading objectives, including correctness, reasoning depth, and error localization. To enhance spatial understanding, we propose an Expression-Aware Visual Prompting Module, trained on our synthesized multi-line math expressions dataset to robustly guide attention in visually heterogeneous inputs. Evaluated on AIHub and FERMAT datasets, VEHME achieves state-of-the-art performance among open-source models and approaches the accuracy of proprietary systems, demonstrating its potential as a scalable and accessible tool for automated math assessment. Our training and experiment code is publicly available at our GitHub repository.

</details>

---

## 94. VideoPASTA: 7KPreference Pairs That Matter for Video-LLMAlignment

- [ ] VideoPASTA: 7KPreference Pairs That Matter for Video-LLMAlignment | https://aclanthology.org/2025.emnlp-main.1647/

- **Link**: https://aclanthology.org/2025.emnlp-main.1647/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduceVideoPASTA(PreferenceAlignment withSpatio-Temporal-Cross FrameAdversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully crafted adversarial examples that deliberately violate spatial, temporal, or cross-frame relationships. With only 7,020 preference pairs and Direct Preference Optimization, VideoPASTA enables models to learn robust representations that capture fine-grained spatial details and long-range temporal dynamics. Experiments demonstrate that VideoPASTA is model agnostic and significantly improves performance, for example, achieving gains of up to + 3.8 percentage points on LongVideoBench, +4.1 on VideoMME, and +4.0 on MVBench, when applied to various state-of-the-art Video-LLMs. These results demonstrate that targeted alignment, rather than massive pretraining or architectural modifications, effectively addresses core video-language challenges. Notably, VideoPASTA achieves these improvements without any human annotation or captioning, relying solely on 32-frame sampling. This efficiency makes our approach a scalable plug-and-play solution that seamlessly integrates with existing models while preserving their original capabilities.

</details>

---

## 95. VISaGE: Understanding Visual Generics and Exceptions

- [ ] VISaGE: Understanding Visual Generics and Exceptions | https://aclanthology.org/2025.emnlp-main.1655/

- **Link**: https://aclanthology.org/2025.emnlp-main.1655/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Vision Language Models (VLMs) learn conceptual representations, in the form of generalized knowledge, during training, they are typically used to analyze individual instances. When evaluation instances are atypical, this paradigm results in tension between two priors in the model. The first is a pragmatic prior that the textual and visual input are both relevant, arising from VLM finetuning on congruent inputs; the second is a semantic prior that the conceptual representation is generally true for instances of the category. In order to understand how VLMs trade off these priors, we introduce a new evaluation dataset, VISaGE, consisting of both typical and exceptional images. In carefully balanced experiments, we show that conceptual understanding degrades when the assumption of congruency underlying the pragmatic prior is violated with incongruent images. This effect is stronger than the effect of the semantic prior when querying about individual instances

</details>

---

## 96. MakingVLMs More Robot-Friendly: Self-Critical Distillation of Low-Level Procedural Reasoning

- [ ] MakingVLMs More Robot-Friendly: Self-Critical Distillation of Low-Level Procedural Reasoning | https://aclanthology.org/2025.emnlp-main.1658/

- **Link**: https://aclanthology.org/2025.emnlp-main.1658/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have shown promise in robotic procedural planning, yet their human-centric reasoning often omits the low-level, grounded details needed for robotic execution. Vision-language models (VLMs) offer a path toward more perceptually grounded plans, but current methods either rely on expensive, large-scale models or are constrained to narrow simulation settings. We introduce SelfReVision, a lightweight and scalable self-improvement framework for vision-language procedural planning. SelfReVision enables small VLMs to iteratively critique, revise, and verify their own plans, without external supervision or teacher models, drawing inspiration from chain-of-thought prompting and self-instruct paradigms. Through this self-distillation loop, models generate higher-quality, execution-ready plans that can be used both at inference and for continued fine-tuning. Using models varying from 3B to 72B, our results show that SelfReVision not only boosts performance over weak base VLMs but also outperforms models 100X the size, yielding improved control in downstream embodied tasks.

</details>

---

## 97. GUI-Bee: AlignGUIAction Grounding to Novel Environments via Autonomous Exploration

- [ ] GUI-Bee: AlignGUIAction Grounding to Novel Environments via Autonomous Exploration | https://aclanthology.org/2025.emnlp-main.1688/

- **Link**: https://aclanthology.org/2025.emnlp-main.1688/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interface (GUI) action grounding, mapping language instructions to actionable elements on GUI screens, is important for assisting users in interactive tutorials, task automation, accessibility support, etc. Most recent works of GUI action grounding use large GUI datasets to fine-tune Multimodal Large Language Models (MLLMs). However, the fine-tuning data is inherently limited to specific GUI environments, leading to significant performance degradation in novel environments due to the generalization challenges in the GUI domain. Therefore, we argue that GUI action grounding models should be further aligned with novel environments before deployment to optimize their performance. To address this, we first propose GUI-Bee, an MLLM-based autonomous agent, to collect high-quality, environment-specific data through exploration and then continuously fine-tune GUI grounding models with the collected data. To ensure the GUI action grounding models generalize to various screens within the target novel environment after the continuous fine-tuning, we equip GUI-Bee with a novel Q-value-Incentive In-Context Reinforcement Learning (Q-ICRL) algorithm that optimizes exploration efficiency and exploration data quality. In the experiment, we introduce NovelScreenSpot to test how well the data can help align GUI action grounding models to novel environments. Furthermore, we conduct an ablation study to validate the Q-ICRL method in enhancing the efficiency of GUI-Bee.

</details>

---

## 98. Taking Notes Brings Focus? Towards Multi-Turn Multimodal Dialogue Learning

- [ ] Taking Notes Brings Focus? Towards Multi-Turn Multimodal Dialogue Learning | https://aclanthology.org/2025.emnlp-main.1690/

- **Link**: https://aclanthology.org/2025.emnlp-main.1690/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs), built on large-scale pre-trained vision towers and language models, have shown great capabilities in multimodal understanding. However, most existing MLLMs are trained on single-turn vision question-answering tasks, which do not accurately reflect real-world human conversations. In this paper, we introduce MMDiag, a new large-scale multi-turn multimodal dialogue dataset. This dataset is collaboratively generated through deliberately designed rules and GPT assistance, featuring complex dialogues with contextual dependencies that force models to track, ground, and recall information across multiple turns and disparate visual regions. MMDiag serves as a strong benchmark for multi-turn multimodal dialogue learning and brings more challenges to the grounding and reasoning capabilities of MLLMs. Further, inspired by human vision processing we present DiagNote, equipped with multimodal grounding and reasoning capabilities. DiagNote adopts a novel dual-module architecture that explicitly separates reasoning from grounding: a reasoning module (Deliberate) performs step-by-step Chain-of-Thought, while a grounding module (Gaze) provides precise visual focus by predicting bounding box annotations. These modules interact iteratively, enabling DiagNote to dynamically refine its understanding. We empirically demonstrate the advantages of DiagNote in both grounding and jointly processing and reasoning with vision and language information over existing MLLMs.

</details>

---

## 99. KRETA: A Benchmark forKorean Reading and Reasoning in Text-RichVQAAttuned to Diverse Visual Contexts

- [ ] KRETA: A Benchmark forKorean Reading and Reasoning in Text-RichVQAAttuned to Diverse Visual Contexts | https://aclanthology.org/2025.emnlp-main.1696/

- **Link**: https://aclanthology.org/2025.emnlp-main.1696/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding and reasoning over text within visual contexts poses a significant challenge for Vision-Language Models (VLMs), given the complexity and diversity of real-world scenarios. To address this challenge, text-rich Visual Question Answering (VQA) datasets and benchmarks have emerged for high-resource languages like English. However, a critical gap persists for low-resource languages such as Korean, where the lack of comprehensive benchmarks hinders robust model evaluation and comparison. To bridge this gap, we introduce KRETA, a benchmark for Korean Reading and rEasoning in Text-rich VQA Attuned to diverse visual contexts. KRETA facilitates an in-depth evaluation of both visual text understanding and reasoning capabilities, while also supporting a multifaceted assessment across 15 domains and 26 image types. Additionally, we introduce a semi-automated VQA generation pipeline specifically optimized for text-rich settings, leveraging refined stepwise image decomposition and a rigorous seven-metric evaluation protocol to ensure data quality. While KRETA is tailored for Korean, we hope our adaptable and extensible pipeline will facilitate the development of similar benchmarks in other languages, thereby accelerating multilingual VLM research. The code and dataset for KRETA are available at [https://github.com/tabtoyou/KRETA](https://github.com/tabtoyou/KRETA).

</details>

---

## 100. Benchmarking and MitigatingMCQASelection Bias of Large Vision-Language Models

- [ ] Benchmarking and MitigatingMCQASelection Bias of Large Vision-Language Models | https://aclanthology.org/2025.emnlp-main.1703/

- **Link**: https://aclanthology.org/2025.emnlp-main.1703/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have achieved strong performance on vision-language tasks, particularly Visual Question Answering (VQA). While prior work has explored unimodal biases in VQA, the problem of selection bias in Multiple-Choice Question Answering (MCQA), where models may favor specific option tokens (e.g., “A”) or positions, remains underexplored. In this paper, we investigate both the presence and nature of selection bias in LVLMs through fine-grained MCQA benchmarks spanning easy, medium, and hard difficulty levels, defined by the semantic similarity of the options. We further propose an inference-time logit-level debiasing method that estimates an ensemble bias vector from general and contextual prompts and applies confidence-adaptive corrections to the model’s output. Our method mitigates bias without retraining and is compatible with frozen LVLMs. Extensive experiments across several state-of-the-art models reveal consistent selection biases that intensify with task difficulty, and show that our mitigation approach significantly reduces bias while improving accuracy in challenging settings. This work offers new insights into the limitations of LVLMs in MCQA and presents a practical approach to improve their robustness in fine-grained visual reasoning. Datasets and code are available at: https://github.com/Atabuzzaman/Selection-Bias-of-LVLMs

</details>

---

## 101. Image Difference Captioning via Adversarial Preference Optimization

- [ ] Image Difference Captioning via Adversarial Preference Optimization | https://aclanthology.org/2025.emnlp-main.1713/

- **Link**: https://aclanthology.org/2025.emnlp-main.1713/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image Difference Captioning (IDC) aims to generate natural language descriptions that highlight subtle differences between two visually similar images. While recent advances leverage pre-trained vision-language models to align fine-grained visual differences with textual semantics, existing supervised approaches often overfit to dataset-specific language patterns and fail to capture accurate preferences on IDC, which often indicates fine-grained and context-aware distinctions. To address these limitations, we propose an adversarial direct preference optimization (ADPO) framework for IDC, which formulates IDC as a preference optimization problem under the Bradley-Terry-Luce model, directly aligning the captioning policy with pairwise difference preferences via Direct Preference Optimization (DPO). To model more accurate and diverse IDC preferences, we introduce an adversarially trained hard negative retriever that selects counterfactual captions, This results in a minimax optimization problem, which we solve via policy-gradient reinforcement learning, enabling the policy and retriever to improve jointly. Experiments on benchmark IDC datasets show that our approach outperforms existing baselines, especially in generating fine-grained and accurate difference descriptions.

</details>

---

## 102. Shallow Focus, Deep Fixes: Enhancing Shallow Layers Vision Attention Sinks to Alleviate Hallucination inLVLMs

- [ ] Shallow Focus, Deep Fixes: Enhancing Shallow Layers Vision Attention Sinks to Alleviate Hallucination inLVLMs | https://aclanthology.org/2025.emnlp-main.174/

- **Link**: https://aclanthology.org/2025.emnlp-main.174/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) demonstrate excellent abilities for understanding visual information, while the hallucination remains. Albeit image tokens constitute the majority of the MLLMs input, the relation between image tokens and hallucinations is still unexplored. In this paper, we analyze the attention score distribution of image tokens across layers and attention heads in models, revealing an intriguing but common phenomenon: most hallucinations are closely linked to the attention sink patterns of image tokens attention matrix, where shallow layers exhibit dense sinks and deep layers exhibit the sparse. We further explore the attention heads of different layers, finding: heads with high-density attention sink of the image part act positively in mitigating hallucinations. Inspired by these findings, we propose a training-free approach called Enhancing Vision Attention Sinks (EVAS) to facilitate the convergence of the image token attention sink within shallow layers. Specifically, EVAS identifies the attention heads that emerge as the densest visual sink in shallow layers and extracts its attention matrix, which is then broadcast to other heads of the same layer, thereby strengthing the layer’s focus on the image itself. Extensive empirical results of various MLLMs illustrate the superior performance of the proposed EVAS, demonstrating its effectiveness and generality.

</details>

---

## 103. Concept-pedia: a Wide-coverage Semantically-annotated Multimodal Dataset

- [ ] Concept-pedia: a Wide-coverage Semantically-annotated Multimodal Dataset | https://aclanthology.org/2025.emnlp-main.1745/

- **Link**: https://aclanthology.org/2025.emnlp-main.1745/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language Models (VLMs), such as CLIP and SigLIP, have become the de facto standard for multimodal tasks, serving as essential building blocks for recent Multimodal Large Language Models, including LLaVA and PaliGemma. However, current evaluations for VLMs remain heavily anchored to ImageNet. In this paper, we question whether ImageNet’s coverage is still sufficiently challenging for modern VLMs, and investigate the impact of adding novel and varied concept categories, i.e., semantically grouped fine-grained synsets. To this end, we introduce Concept-pedia, a novel, large-scale, semantically-annotated multimodal resource covering more than 165,000 concepts. Leveraging a language-agnostic, automatic annotation pipeline grounded in Wikipedia, Concept-pedia expands the range of visual concepts, including diverse abstract categories. Building on Concept-pedia, we also present a manually-curated Visual Concept Recognition evaluation benchmark, Concept-10k, that spans thousands of concepts across a wide range of categories. Our experiments show that current models, although excelling on ImageNet, struggle with Concept-10k. Not only do these findings highlight a persistent bias toward ImageNet-centric concepts, but they also underscore the urgent need for more representative benchmarks. By offering a broader and semantically richer testbed, Concept-10k aims to support the development of multimodal systems that better generalize to the complexities of real-world visual concepts.

</details>

---

## 104. EduVidQA: Generating and Evaluating Long-form Answers to Student Questions based on Lecture Videos

- [ ] EduVidQA: Generating and Evaluating Long-form Answers to Student Questions based on Lecture Videos | https://aclanthology.org/2025.emnlp-main.1760/

- **Link**: https://aclanthology.org/2025.emnlp-main.1760/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As digital platforms redefine educational paradigms, ensuring interactivity remains vital for effective learning. This paper explores using Multimodal Large Language Models (MLLMs) to automatically respond to student questions from online lectures - a novel question answering task of real world significance. We introduce the EduVidQA Dataset with 5252 question-answer pairs (both synthetic and real-world) from 296 computer science videos covering diverse topics and difficulty levels. To understand the needs of the dataset and task evaluation, we empirically study the qualitative preferences of students, which we provide as an important contribution to this line of work. Our benchmarking experiments consist of 6 state-of-the-art MLLMs, through which we study the effectiveness of our synthetic data for finetuning, as well as showing the challenging nature of the task. We evaluate the models using both text-based and qualitative metrics, thus showing a nuanced perspective of the models’ performance, which is paramount to future work. This work not only sets a benchmark for this important problem, but also opens exciting avenues for future research in the field of Natural Language Processing for Education.

</details>

---

## 105. MemeReaCon: Probing Contextual Meme Understanding in Large Vision-Language Models

- [ ] MemeReaCon: Probing Contextual Meme Understanding in Large Vision-Language Models | https://aclanthology.org/2025.emnlp-main.176/

- **Link**: https://aclanthology.org/2025.emnlp-main.176/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Memes have emerged as a popular form of multimodal online communication, where their interpretation heavily depends on the specific context in which they appear. Current approaches predominantly focus on isolated meme analysis, either for harmful content detection or standalone interpretation, overlooking a fundamental challenge: the same meme can express different intents depending on its conversational context. This oversight creates an evaluation gap: although humans intuitively recognize how context shapes meme interpretation, Large Vision Language Models (LVLMs) can hardly understand context-dependent meme intent. To address this critical limitation, we introduce MemeReaCon, a novel benchmark specifically designed to evaluate how LVLMs understand memes in their original context. We collected memes from five different Reddit communities, keeping each meme’s image, the post text, and user comments together. We carefully labeled how the text and meme work together, what the poster intended, how the meme is structured, and how the community responded. Our tests with leading LVLMs show a clear weakness: models either fail to interpret critical information in the contexts, or overly focus on visual details while overlooking communicative purpose. MemeReaCon thus serves both as a diagnostic tool exposing current limitations and as a challenging benchmark to drive development toward more sophisticated LVLMs of the context-aware understanding.

</details>

---

## 106. CHURRO: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition

- [ ] CHURRO: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition | https://aclanthology.org/2025.emnlp-main.1763/

- **Link**: https://aclanthology.org/2025.emnlp-main.1763/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Accurate text recognition for historical documents can greatly advance the study and preservation of cultural heritage. Existing vision-language models (VLMs), however, are designed for modern, standardized texts and are not equipped to read the diverse languages and scripts, irregular layouts, and frequent degradation found in historical materials.This paper presents CHURRO, a 3B-parameter open-weight VLM specialized for historical text recognition. The model is trained on CHURRO-DS, the largest historical text recognition dataset to date. CHURRO-DS unifies 155 historical corpora comprising 99,491 pages, spanning 22 centuries of textual heritage across 46 language clusters, including historical variants and dead languages.We evaluate several open-weight and closed VLMs and optical character recognition (OCR) systems on CHURRO-DS and find that CHURRO outperforms all other VLMs. On the CHURRO-DS test set, CHURRO achieves 82.3% (printed) and 70.1% (handwritten) normalized Levenshtein similarity, surpassing the second-best model, Gemini 2.5 Pro, by 1.4% and 6.5%, respectively, while being 15.5 times more cost-effective.By releasing the model and dataset, we aim to enable community-driven research to improve the readability of historical texts and accelerate scholarship.

</details>

---

## 107. The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology

- [ ] The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology | https://aclanthology.org/2025.emnlp-main.1768/

- **Link**: https://aclanthology.org/2025.emnlp-main.1768/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

According to the U.S. National Institutes of Health, more than 3.4 million children experience speech disorders that require clinical intervention. The number of speech-language pathologists (SLPs) is roughly 20 times fewer than the number of affected children, highlighting a significant gap in children’s care and a pressing need for technological support that improves the productivity of SLPs. State-of-the-art multimodal language models (MLMs) show promise for supporting SLPs, but their use remains underexplored largely due to a limited understanding of their performance in high-stakes clinical settings. To address this gap, we collaborate with domain experts to develop a taxonomy of real-world use cases of MLMs in speech-language pathologies. Building on this taxonomy, we introduce the first comprehensive benchmark for evaluating MLM across five core use cases, each containing 1,000 manually annotated data points. This benchmark includes robustness and sensitivity tests under various settings, including background noise, speaker gender, and accent. Our evaluation of 15 state-of-the-art MLMs reveals that no single model consistently outperforms others across all tasks. Notably, we find systematic disparities, with models performing better on male speakers, and observe that chain-of-thought prompting can degrade performance on classification tasks with large label spaces and narrow decision boundaries. Furthermore, we study fine-tuning MLMs on domain-specific data, achieving improvements of over 30% compared to base models. These findings highlight both the potential and limitations of current MLMs for speech-language pathology applications, underscoring the need for further research and targeted development.

</details>

---

## 108. Eliciting Implicit Acoustic Styles from Open-domain Instructions to Facilitate Fine-grained Controllable Generation of Speech

- [ ] Eliciting Implicit Acoustic Styles from Open-domain Instructions to Facilitate Fine-grained Controllable Generation of Speech | https://aclanthology.org/2025.emnlp-main.182/

- **Link**: https://aclanthology.org/2025.emnlp-main.182/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper focuses on generating speech with the acoustic style that meets users’ needs based on their open-domain instructions. To control the style, early work mostly relies on pre-defined rules or templates. The control types and formats are fixed in a closed domain, making it hard to meet diverse needs of users. One solution is to resort to instructions in free text to guide the generation. Current work mainly studies the instructions that clearly specify the acoustic styles, such as low pitch and fast speed. However, the instructions are complex, some even vague and abstract, such as “Generate a voice of a woman who is heartbroken due to a breakup. It is hard to infer this implicit style by traditional matching-based methods. To address this problem, we propose a new controllable model. It first utilizes multimodal LLMs with knowledge-augmented techniques to infer the desired speech style from the instructions. The powerful language understanding ability of LLMs can help us better elicit the implicit style factors from the instruction. By using these factors as a control condition, we design a diffusion-based generator adept at finely adjusting speech details. That offers higher flexibility to meet complex users’ needs. Next, we verify the output speech from three aspects, i.e., consistency of decoding state, mel-spectrogram, and instruction style. This verified feedback can inversely optimize the generator. Extensive experiments are conducted on three popular datasets. The results show the effectiveness and good controllability of our approach.

</details>

---

## 109. Seeing Through Words, Speaking Through Pixels: Deep Representational Alignment Between Vision and Language Models

- [ ] Seeing Through Words, Speaking Through Pixels: Deep Representational Alignment Between Vision and Language Models | https://aclanthology.org/2025.emnlp-main.1806/

- **Link**: https://aclanthology.org/2025.emnlp-main.1806/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies show that deep vision-only and language-only models—trained on disjoint modalities—nonetheless project their inputs into a partially aligned representational space. Yet we still lack a clear picture of _where_ in each network this convergence emerges, _what_ visual or linguistic cues support it, _whether_ it captures human preferences in many-to-many image-text scenarios, and _how_ aggregating exemplars of the same concept affects alignment. Here, we systematically investigate these questions. We find that alignment peaks in mid-to-late layers of both model types, reflecting a shift from modality-specific to conceptually shared representations. This alignment is robust to appearance-only changes but collapses when semantics are altered (e.g., object removal or word-order scrambling), highlighting that the shared code is truly semantic. Moving beyond the one-to-one image-caption paradigm, a forced-choice “Pick-a-Pic” task shows that human preferences for image-caption matches are mirrored in the embedding spaces across all vision-language model pairs. This pattern holds bidirectionally when multiple captions correspond to a single image, demonstrating that models capture fine-grained semantic distinctions akin to human judgments. Surprisingly, averaging embeddings across exemplars amplifies alignment rather than blurring detail. Together, our results demonstrate that unimodal networks converge on a shared semantic code that aligns with human judgments and strengthens with exemplar aggregation.

</details>

---

## 110. Language-to-Space Programming for Training-Free 3DVisual Grounding

- [ ] Language-to-Space Programming for Training-Free 3DVisual Grounding | https://aclanthology.org/2025.emnlp-main.191/

- **Link**: https://aclanthology.org/2025.emnlp-main.191/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

3D visual grounding (3DVG) is challenging due to the need to understand 3D spatial relations. While supervised approaches have achieved superior performance, they are constrained by the scarcity and high annotation costs of 3D vision-language datasets. Training-free approaches based on LLMs/VLMs eliminate the need for large-scale training data, but they either incur prohibitive grounding time and token costs or have unsatisfactory accuracy. To address the challenges, we introduce a novel method for training-free 3D visual grounding, namely **La**nguage-to-**S**pace **P**rogramming (LaSP). LaSP introduces LLM-generated codes to analyze 3D spatial relations among objects, along with a pipeline that evaluates and optimizes the codes automatically. Experimental results demonstrate that LaSP achieves 52.9% accuracy on the Nr3D benchmark, ranking among the best training-free methods. Moreover, it substantially reduces the grounding time and token costs, offering a balanced trade-off between performance and efficiency.

</details>

---

## 111. F2TEval: Human-Aligned Multi-Dimensional Evaluation for Figure-to-Text Task

- [ ] F2TEval: Human-Aligned Multi-Dimensional Evaluation for Figure-to-Text Task | https://aclanthology.org/2025.emnlp-main.195/

- **Link**: https://aclanthology.org/2025.emnlp-main.195/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Figure-to-Text (F2T) tasks aim to convert structured figure information into natural language text, serving as a bridge between visual perception and language understanding.However, existing evaluation methods remain limited: 1) Reference-based methods can only capture shallow semantic similarities and rely on costly labeled reference text; 2) Reference-free methods depend on multimodal large language models, which suffer from low efficiency and instruction sensitivity; 3) Existing methods provide only sample-level evaluations, lacking interpretability and alignment with expert-level multi-dimensional evaluation criteria.Accordingly, we propose F2TEval, a five-dimensional reference-free evaluation method aligned with expert criteria, covering faithfulness, completeness, conciseness, logicality, and analysis, to support fine-grained evaluation. We design a lightweight mixture-of-experts model that incorporates independent scoring heads and applies the Hilbert-Schmidt Independence Criterion to optimize the disentanglement of scoring representations across dimensions. Furthermore, we construct F2TBenchmark, a human-annotated benchmark dataset covering 21 chart types and 35 application domains, to support research on F2T evaluation. Experimental results demonstrate our model’s superior performance and efficiency, outperforming Gemini-2.0 and Claude-3.5 with only 0.9B parameters.

</details>

---

## 112. SmartBench: Is YourLLMTruly a GoodChinese Smartphone Assistant?

- [ ] SmartBench: Is YourLLMTruly a GoodChinese Smartphone Assistant? | https://aclanthology.org/2025.emnlp-main.194/

- **Link**: https://aclanthology.org/2025.emnlp-main.194/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have become integral to daily life, especially advancing as intelligent assistants through on-device deployment on smartphones. However, existing LLM evaluation benchmarks predominantly focus on objective tasks like mathematics and coding in English, which do not necessarily reflect the practical use cases of on-device LLMs in real-world mobile scenarios, especially for Chinese users. To address these gaps, we introduce **SmartBench**, the first benchmark designed to evaluate the capabilities of on-device LLMs in Chinese mobile contexts. We analyze functionalities provided by representative smartphone manufacturers and divide them into five categories: text summarization, text Q&A, information extraction, content creation, and notification management, further detailed into 20 specific tasks. For each task, we construct high-quality datasets comprising 50 to 200 question-answer pairs that reflect everyday mobile interactions, and we develop automated evaluation criteria tailored for these tasks. We conduct comprehensive evaluations of on-device LLMs and MLLMs using SmartBench and also assess their performance after quantized deployment on real smartphone NPUs. Our contributions provide a standardized framework for evaluating on-device LLMs in Chinese, promoting further development and optimization in this critical area. Code and data will be available at https://github.com/vivo-ai-lab/SmartBench.

</details>

---

## 113. ModRWKV: Transformer Multimodality in Linear Time

- [ ] ModRWKV: Transformer Multimodality in Linear Time | https://aclanthology.org/2025.emnlp-main.204/

- **Link**: https://aclanthology.org/2025.emnlp-main.204/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Currently, most multimodal studies are based on large language models (LLMs) with quadratic-complexity Transformer architectures. While linear models like RNNs enjoy low inference costs, their application has been largely limited to the text-only modality. This work explores the capabilities of modern RNN architectures in multimodal contexts. We propose ModRWKV—a decoupled multimodal framework built upon the RWKV7 architecture as its LLM backbone—which achieves multi-source information fusion through dynamically adaptable heterogeneous modality encoders. We designed the multimodal modules in ModRWKV with an extremely lightweight architecture and, through extensive experiments, identified a configuration that achieves an optimal balance between performance and computational efficiency. ModRWKV leverages the pretrained weights of the RWKV7 LLM for initialization, which significantly accelerates multimodal training. Comparative experiments with different pretrained checkpoints further demonstrate that such initialization plays a crucial role in enhancing the model’s ability to understand multimodal signals. Supported by extensive experiments, we conclude that modern RNN architectures present a viable alternative to Transformers in the domain of multimodal large language models (MLLMs). Furthermore, we identify the optimal configuration of the ModRWKV architecture through systematic exploration.

</details>

---

## 114. Multimedia Event Extraction withLLMKnowledge Editing

- [ ] Multimedia Event Extraction withLLMKnowledge Editing | https://aclanthology.org/2025.emnlp-main.205/

- **Link**: https://aclanthology.org/2025.emnlp-main.205/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal event extraction task aims to identify event types and arguments from visual and textual representations related to events. Due to the high cost of multimedia training data, previous methods mainly focused on weakly alignment of excellent unimodal encoders. However, they ignore the conflict between event understanding and image recognition, resulting in redundant feature perception affecting the understanding of multimodal events. In this paper, we propose a multimodal event extraction strategy with a multi-level redundant feature selection mechanism, which enhances the event understanding ability of multimodal large language models by leveraging knowledge editing techniques, and requires no additional parameter optimization work. Extensive experiments show that our method outperforms the state-of-the-art (SOTA) baselines on the M2E2 benchmark. Compared with the highest baseline, we achieve a 34% improvement of precision on event extraction and a 11% improvement of F1 on argument extraction.

</details>

---

## 115. FinRAGBench-V: A Benchmark for MultimodalRAGwith Visual Citation in the Financial Domain

- [ ] FinRAGBench-V: A Benchmark for MultimodalRAGwith Visual Citation in the Financial Domain | https://aclanthology.org/2025.emnlp-main.211/

- **Link**: https://aclanthology.org/2025.emnlp-main.211/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-Augmented Generation (RAG) plays a vital role in the financial domain, powering applications such as real-time market analysis, trend forecasting, and interest rate computation. However, most existing RAG research in finance focuses predominantly on textual data, overlooking the rich visual content in financial documents, resulting in the loss of key analytical insights. To bridge this gap, we present FinRAGBench-V, a comprehensive visual RAG benchmark tailored for finance. This benchmark effectively integrates multimodal data and provides visual citation to ensure traceability. It includes a bilingual retrieval corpus with 60,780 Chinese and 51,219 English pages, along with a high-quality, human-annotated question-answering (QA) dataset spanning heterogeneous data types and seven question categories. Moreover, we introduce RGenCite, an RAG baseline that seamlessly integrates visual citation with generation. Furthermore, we propose an automatic citation evaluation method to systematically assess the visual citation capabilities of Multimodal Large Language Models (MLLMs). Extensive experiments on RGenCite underscore the challenging nature of FinRAGBench-V, providing valuable insights for the development of multimodal RAG systems in finance.

</details>

---

## 116. BannerAgency: Advertising Banner Design with MultimodalLLMAgents

- [ ] BannerAgency: Advertising Banner Design with MultimodalLLMAgents | https://aclanthology.org/2025.emnlp-main.214/

- **Link**: https://aclanthology.org/2025.emnlp-main.214/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Advertising banners are critical for capturing user attention and enhancing advertising campaign effectiveness. Creating aesthetically pleasing banner designs while conveying the campaign messages is challenging due to the large search space involving multiple design elements. Additionally, advertisers need multiple sizes for different displays and various versions to target different sectors of audiences. Since design is intrinsically an iterative and subjective process, flexible editability is also in high demand for practical usage. While current models have served as assistants to human designers in various design tasks, they typically handle only segments of the creative design process or produce pixel-based outputs that limit editability. This paper introduces a training-free framework for fully automated banner ad design creation, enabling frontier multimodal large language models (MLLMs) to streamline the production of effective banners with minimal manual effort across diverse marketing contexts. We present BannerAgency, an MLLM agent system that collaborates with advertisers to understand their brand identity and banner objectives, generates matching background images, creates blueprints for foreground design elements, and renders the final creatives as editable components in Figma or SVG formats rather than static pixels. To facilitate evaluation and future research, we introduce BannerRequest400, a benchmark featuring 100 unique logos paired with 400 diverse banner requests. Through quantitative and qualitative evaluations, we demonstrate the framework’s effectiveness, emphasizing the quality of the generated banner designs, their adaptability to various banner requests, and their strong editability enabled by this component-based approach.

</details>

---

## 117. Chain-of-Talkers (CoTalk): Fast Human Annotation of Dense Image Captions

- [ ] Chain-of-Talkers (CoTalk): Fast Human Annotation of Dense Image Captions | https://aclanthology.org/2025.emnlp-main.221/

- **Link**: https://aclanthology.org/2025.emnlp-main.221/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While densely annotated image captions significantly facilitate the learning of robust vision-language alignment, methodologies for systematically optimizing human annotation efforts remain underexplored. We introduce Chain-of-Talkers (CoTalk), an AI-in-the-loop methodology designed to maximize the number of annotated samples and improve their comprehensiveness under fixed budget constraints (e.g., total human annotation time). The framework is built upon two key insights. First, sequential annotation reduces redundant workload compared to conventional parallel annotation, as subsequent annotators only need to annotate the “residual”—the missing visual information that previous annotations have not covered. Second, humans process textual input faster by reading while outputting annotations with much higher throughput via talking; thus a multimodal interface enables optimized efficiency. We evaluate our framework from two aspects: intrinsic evaluations that assess the comprehensiveness of semantic units, obtained by parsing detailed captions into object-attribute trees and analyzing their effective connections; extrinsic evaluation measures the practical usage of the annotated captions in facilitating vision-language alignment. Experiments with eight participants show our Chain-of-Talkers (CoTalk) improves annotation speed (0.42 vs. 0.30 units/sec) and retrieval performance (41.13% vs. 40.52%) over the parallel method.

</details>

---

## 118. ChartMind: A Comprehensive Benchmark for Complex Real-world Multimodal Chart Question Answering

- [ ] ChartMind: A Comprehensive Benchmark for Complex Real-world Multimodal Chart Question Answering | https://aclanthology.org/2025.emnlp-main.226/

- **Link**: https://aclanthology.org/2025.emnlp-main.226/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chart question answering (CQA) has become a critical multimodal task for evaluating the reasoning capabilities of vision-language models. While early approaches have shown promising performance by focusing on visual features or leveraging large-scale pre-training, most existing evaluations rely on rigid output formats and objective metrics, thus ignoring the complex, real-world demands of practical chart analysis. In this paper, we introduce ChartMind, a new benchmark designed for complex CQA tasks in real-world settings. ChartMind covers seven task categories, incorporates multilingual contexts, supports open-domain textual outputs, and accommodates diverse chart formats, bridging the gap between real-world applications and traditional academic benchmarks. Furthermore, we propose a context-aware yet model-agnostic framework, ChartLLM, that focuses on extracting key contextual elements, reducing noise, and enhancing the reasoning accuracy of multimodal large language models. Extensive evaluations on ChartMind and three representative public benchmarks with 14 mainstream multimodal models show our framework significantly outperforms the previous three common CQA paradigms: instruction-following, OCR-enhanced, and chain-of-thought, highlighting the importance of flexible chart understanding for real-world CQA. These findings suggest new directions for developing more robust chart reasoning in future research.

</details>

---

## 119. Pruning the Paradox: HowCLIP’s Most Informative Heads Enhance Performance While Amplifying Bias

- [ ] Pruning the Paradox: HowCLIP’s Most Informative Heads Enhance Performance While Amplifying Bias | https://aclanthology.org/2025.emnlp-main.229/

- **Link**: https://aclanthology.org/2025.emnlp-main.229/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

CLIP is one of the most popular foundation models and is heavily used for many vision-language tasks, yet little is known about its inner workings. As CLIP is increasingly deployed in real-world applications, it is becoming even more critical to understand its limitations and embedded social biases to mitigate potentially harmful downstream consequences. However, the question of what internal mechanisms drive both the impressive capabilities as well as problematic shortcomings of CLIP has largely remained unanswered. To bridge this gap, we study the conceptual consistency of text descriptions for attention heads in CLIP-like models. Specifically, we propose Concept Consistency Score (CCS), a novel interpretability metric that measures how consistently individual attention heads in CLIP models align with specific concepts. Our soft-pruning experiments reveal that high CCS heads are critical for preserving model performance, as pruning them leads to a significantly larger performance drop than pruning random or low CCS heads. Notably, we find that high CCS heads capture essential concepts and play a key role in out-of-domain detection, concept-specific reasoning, and video-language understanding. Moreover, we prove that high CCS heads learn spurious correlations which amplify social biases. These results position CCS as a powerful interpretability metric exposing the paradox of performance and social biases in CLIP models.

</details>

---

## 120. MIO: A Foundation Model on Multimodal Tokens

- [ ] MIO: A Foundation Model on Multimodal Tokens | https://aclanthology.org/2025.emnlp-main.255/

- **Link**: https://aclanthology.org/2025.emnlp-main.255/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce MIO, a novel foundation model built on multimodal tokens, capable of understanding and generating speech, text, images, and videos in an end-to-end, autoregressive manner. While the emergence of large language models (LLMs) and multimodal large language models (MM-LLMs) propels advancements in artificial general intelligence through their versatile capabilities, they still lack true any-to-any understanding and generation. Recently, the release of GPT-4o has showcased the remarkable potential of any-to-any LLMs for complex real-world tasks, enabling omnidirectional input and output across images, speech, and text. However, it is closed-source and does not support the generation of multimodal interleaved sequences. To address this gap, we present MIO, which is trained on a mixture of discrete tokens across four modalities using causal multimodal modeling. MIO undergoes a four-stage training process: (1) alignment pre-training, (2) interleaved pre-training, (3) speech-enhanced pre-training, and (4) comprehensive supervised fine-tuning on diverse textual, visual, and speech tasks. Our experimental results indicate that MIO exhibits competitive, and in some cases superior, performance compared to previous dual-modal baselines, any-to-any model baselines, and even modality-specific baselines. Moreover, MIO demonstrates advanced capabilities inherent to its any-to-any feature, such as interleaved video-text generation, chain-of-visual-thought reasoning, visual guideline generation, instructional image editing, etc.

</details>

---

## 121. AssoCiAm: A Benchmark for Evaluating Association Thinking while Circumventing Ambiguity

- [ ] AssoCiAm: A Benchmark for Evaluating Association Thinking while Circumventing Ambiguity | https://aclanthology.org/2025.emnlp-main.263/

- **Link**: https://aclanthology.org/2025.emnlp-main.263/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have garnered significant attention, offering a promising pathway toward artificial general intelligence (AGI). Among the essential capabilities required for AGI, creativity has emerged as a critical trait for MLLMs, with association serving as its foundation. Association reflects a model’s ability to think creatively, making it vital to evaluate and understand. While several frameworks have been proposed to assess associative ability, they often overlook the inherent ambiguity in association tasks, which arises from the divergent nature of associations and undermines the reliability of evaluations. To address this issue, we decompose ambiguity into two types—internal ambiguity and external ambiguity—and introduce AssoCiAm, a benchmark designed to evaluate associative ability while circumventing the ambiguity through a hybrid computational method. We then conduct extensive experiments on MLLMs, revealing a strong positive correlation between cognition and association. Additionally, we observe that the presence of ambiguity in the evaluation process causes MLLMs’ behavior to become more random-like. Finally, we validate the effectiveness of our method in ensuring more accurate and reliable evaluations. See Project Page for the data and codes.

</details>

---

## 122. Chat-Driven Text Generation and Interaction for Person Retrieval

- [ ] Chat-Driven Text Generation and Interaction for Person Retrieval | https://aclanthology.org/2025.emnlp-main.266/

- **Link**: https://aclanthology.org/2025.emnlp-main.266/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-based person search (TBPS) enables the retrieval of person images from large-scale databases using natural language descriptions, offering critical value in surveillance applications. However, a major challenge lies in the labor-intensive process of obtaining high-quality textual annotations, which limits scalability and practical deployment. To address this, we introduce two complementary modules: Multi-Turn Text Generation (MTG) and Multi-Turn Text Interaction (MTI). MTG generates rich pseudo-labels through simulated dialogues with MLLMs, producing fine-grained and diverse visual descriptions without manual supervision. MTI refines user queries at inference time through dynamic, dialogue-based reasoning, enabling the system to interpret and resolve vague, incomplete, or ambiguous descriptions—characteristics often seen in real-world search scenarios. Together, MTG and MTI form a unified and annotation-free framework that significantly improves retrieval accuracy, robustness, and usability. Extensive evaluations demonstrate that our method achieves competitive or superior results while eliminating the need for manual captions, paving the way for scalable and practical deployment of TBPS systems.

</details>

---

## 123. ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model

- [ ] ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model | https://aclanthology.org/2025.emnlp-main.273/

- **Link**: https://aclanthology.org/2025.emnlp-main.273/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Humans possess a unified cognitive ability to perceive, comprehend, and interact with the physical world. Why can’t large language models replicate this holistic understanding? Through a systematic analysis of existing training paradigms in vision-language-action models (VLA), we identify two key challenges:spurious forgetting, where robot training overwrites crucial visual-text alignments, andtask interference, where competing control and understanding tasks degrade performance when trained jointly. To overcome these limitations, we propose ChatVLA, a novel framework featuring Phased Alignment Training, which incrementally integrates multimodal data after initial control mastery, and a Mixture-of-Experts architecture to minimize task interference. ChatVLA demonstrates competitive performance on visual question-answering datasets and significantly surpasses state-of-the-art vision-language-action (VLA) methods on multimodal understanding benchmarks. Notably, it achieves a six times higher performance on MMMU and scores 47.2% on MMStar with a more parameter-efficient design than ECoT. Furthermore, ChatVLA demonstrates superior performance on 25 real-world robot manipulation tasks compared to existing VLA methods like OpenVLA. Our findings highlight the potential of our unified framework for achieving both robust multimodal understanding and effective robot control.

</details>

---

## 124. CLIP-MoE: Towards Building Mixture of Experts forCLIPwith Diversified Multiplet Upcycling

- [ ] CLIP-MoE: Towards Building Mixture of Experts forCLIPwith Diversified Multiplet Upcycling | https://aclanthology.org/2025.emnlp-main.275/

- **Link**: https://aclanthology.org/2025.emnlp-main.275/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) has become a cornerstone in multimodal intelligence. However, recent studies discovered that CLIP can only encode one aspect of the feature space, leading to substantial information loss and indistinctive features. To mitigate this issue, this paper introduces a novel strategy that fine-tunes a series of complementary CLIP models and transforms them into a CLIP-MoE. Specifically, we propose a model-agnostic Diversified Multiplet Upcycling (DMU) framework for CLIP. Instead of training multiple CLIP models from scratch, DMU leverages a pre-trained CLIP and fine-tunes it into a diverse set with highly cost-effective multistage contrastive learning, thus capturing distinct feature subspaces efficiently. To fully exploit these fine-tuned models while minimizing computational overhead, we transform them into a CLIP-MoE, which dynamically activates a subset of CLIP experts, achieving an effective balance between model capacity and computational cost. Comprehensive experiments demonstrate the superior performance of CLIP-MoE across various zero-shot retrieval, zero-shot image classification tasks, and downstream Multimodal Large Language Model (MLLM) benchmarks when used as a vision encoder. Code is available at https://github.com/OpenSparseLLMs/CLIP-MoE.

</details>

---

## 125. Seeing More, Saying More: Lightweight Language Experts are Dynamic Video Token Compressors

- [ ] Seeing More, Saying More: Lightweight Language Experts are Dynamic Video Token Compressors | https://aclanthology.org/2025.emnlp-main.28/

- **Link**: https://aclanthology.org/2025.emnlp-main.28/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large video-language models have revolutionized video understanding tasks. However, their efficiency is significantly constrained by processing high volumes of visual tokens. Existing token compression strategies apply a fixed compression ratio, ignoring the variability in semantic density among different video clips. Consequently, this lead to inadequate representation of information-rich clips due to insufficient tokens and unnecessary computation on static or content-poor ones. To address this, we propose LangDC, a Language-aware Dynamic Token Compressor. LangDC leverages a lightweight language model to describe video clips, converting them into soft caption tokens as visual representations. Trained with our proposed semantic density-aware supervision, LangDC aims to 1) cover key visual cues necessary for downstream task reasoning and 2) dynamically adjust compression ratios based on scene richness, reflected by descriptions length. Our design mimics how humans dynamically express what they see: complex scenes (seeing more) elicit more detailed language to convey nuances (saying more), whereas simpler scenes are described with fewer words. Experimental results show that our method reduces FLOPs by 49% compared to VideoGPT+ while maintaining competitive performance. Furthermore, qualitative results demonstrate our approach adaptively adjusts the token compression ratio based on video segment richness. Code will be released once acceptance.

</details>

---

## 126. TRUST-VL: An Explainable News Assistant for General Multimodal Misinformation Detection

- [ ] TRUST-VL: An Explainable News Assistant for General Multimodal Misinformation Detection | https://aclanthology.org/2025.emnlp-main.284/

- **Link**: https://aclanthology.org/2025.emnlp-main.284/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal misinformation, encompassing textual, visual, and cross-modal distortions, poses an increasing societal threat that is amplified by generative AI. Existing methods typically focus on a single type of distortion and struggle to generalize to unseen scenarios. In this work, we observe that different distortion types share common reasoning capabilities while also requiring task-specific skills. We hypothesize that joint training across distortion types facilitates knowledge sharing and enhances the model’s ability to generalize. To this end, we introduce TRUST-VL, a unified and explainable vision-language model for general multimodal misinformation detection. TRUST-VL incorporates a novel Question-Aware Visual Amplifier module, designed to extract task-specific visual features. To support training, we also construct TRUST-Instruct, a large-scale instruction dataset containing 198K samples featuring structured reasoning chains aligned with human fact-checking workflows. Extensive experiments on both in-domain and zero-shot benchmarks demonstrate that TRUST-VL achieves state-of-the-art performance, while also offering strong generalization and interpretability.

</details>

---

## 127. Diagram-Driven Course Questions Generation

- [ ] Diagram-Driven Course Questions Generation | https://aclanthology.org/2025.emnlp-main.305/

- **Link**: https://aclanthology.org/2025.emnlp-main.305/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Question Generation (VQG) research focuses predominantly on natural images while neglecting the diagram, which is a critical component in educational materials. To meet the needs of pedagogical assessment, we propose the Diagram-Driven Course Questions Generation (DDCQG) task and construct DiagramQG, a comprehensive dataset with 15,720 diagrams and 25,798 questions across 37 subjects and 371 courses. Our approach employscourseandinputtext constraints to generate course-relevant questions about specific diagram elements. We reveal three challenges of DDCQG: domain-specific knowledge requirements across courses, long-tail distribution in course coverage, and high information density in diagrams. To address these, we propose the Hierarchical Knowledge Integration framework (HKI-DDCQG), which utilizes trainable CLIP for identifying relevant diagram patches, leverages frozen vision-language models for knowledge extraction, and generates questions with trainable T5. Experiments demonstrate that HKI-DDCQG outperforms existing models on DiagramQG while maintaining strong generalizability across natural image datasets, establishing a strong baseline for DDCQG.

</details>

---

## 128. VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models

- [ ] VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models | https://aclanthology.org/2025.emnlp-main.312/

- **Link**: https://aclanthology.org/2025.emnlp-main.312/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emergence of Multimodal Large Reasoning Models (MLRMs) has enabled sophisticated visual reasoning capabilities by integrating reinforcement learning and Chain-of-Thought (CoT) supervision. However, while these enhanced reasoning capabilities improve performance, they also introduce new and underexplored safety risks. In this work, we systematically investigate the security implications of advanced visual reasoning in MLRMs. Our analysis reveals a fundamental trade-off: as visual reasoning improves, models become more vulnerable to jailbreak attacks. Motivated by this critical finding, we introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA combines targeted visual attention masking with a two-stage reasoning induction strategy to precisely control harmful outputs. Extensive experiments demonstrate VisCRA’s significant effectiveness, achieving high attack success rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking, 68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical insight: the very capability that empowers MLRMs — their visual reasoning — can also serve as an attack vector, posing significant security risks. Warning: This paper contains unsafe examples.

</details>

---

## 129. MAKAR: a Multi-Agent framework based Knowledge-Augmented Reasoning for Grounded Multimodal Named Entity Recognition

- [ ] MAKAR: a Multi-Agent framework based Knowledge-Augmented Reasoning for Grounded Multimodal Named Entity Recognition | https://aclanthology.org/2025.emnlp-main.311/

- **Link**: https://aclanthology.org/2025.emnlp-main.311/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Grounded Multimodal Named Entity Recognition (GMNER), which aims to extract textual entities, their types, and corresponding visual regions from image-text data, has become a critical task in multimodal information extraction. However, existing methods face two major challenges. First, they fail to address the semantic ambiguity caused by polysemy and the long-tail distribution of datasets. Second, unlike visual grounding which provides descriptive phrases, entity grounding only offers brief entity names which carry less semantic information. Current methods lack sufficient semantic interaction between text and image, hindering accurate entity-visual region matching. To tackle these issues, we propose MAKAR, a Multi-Agent framework based Knowledge-Augmented Reasoning, comprising three agents: Knowledge Enhancement, Entity Correction, and Entity Reasoning Grounding. Specifically, in the named entity recognition phase, the Knowledge Enhancement Agent leverages a Multimodal Large Language Model (MLLM) as an implicit knowledge base to enhance ambiguous image-text content with its internal knowledge. For samples with low-confidence entity boundaries and types, the Entity Correction Agent uses web search tools to retrieve and summarize relevant web content, thereby correcting entities using both internal and external knowledge. In the entity grounding phase, the Entity Reasoning Grounding Agent utilizes multi-step Chain-of-Thought reasoning to perform grounding for each entity. Extensive experiments show that MAKAR achieves state-of-the-art performance on two benchmark datasets. Code is available at: https://github.com/Nikol-coder/MAKAR.

</details>

---

## 130. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization

- [ ] PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization | https://aclanthology.org/2025.emnlp-main.32/

- **Link**: https://aclanthology.org/2025.emnlp-main.32/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.

</details>

---

## 131. Audio-centric Video Understanding Benchmark without Text Shortcut

- [ ] Audio-centric Video Understanding Benchmark without Text Shortcut | https://aclanthology.org/2025.emnlp-main.333/

- **Link**: https://aclanthology.org/2025.emnlp-main.333/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Audio often serves as an auxiliary modality in video understanding tasks of audio-visual large language models (LLMs), merely assisting in the comprehension of visual information. However, a thorough understanding of videos significantly depends on auditory information, as audio offers critical context, emotional cues, and semantic meaning that visual data alone often lacks. This paper proposes an audio-centric video understanding benchmark (AVUT) to evaluate the video comprehension capabilities of multimodal LLMs with a particular focus on auditory information. AVUT introduces a suite of carefully designed audio-centric tasks, holistically testing the understanding of both audio content and audio-visual interactions in videos. Moreover, this work points out the text shortcut problem that largely exists in other benchmarks where the correct answer can be found from question text alone without needing videos. AVUT addresses this problem by proposing a answer permutation-based filtering mechanism.A thorough evaluation across a diverse range of open-source and proprietary multimodal LLMs is performed, followed by the analyses of deficiencies in audio-visual LLMs. Demos and data are available at https://github.com/lark-png/AVUT.

</details>

---

## 132. FlightGPT: Towards Generalizable and InterpretableUAVVision-and-Language Navigation with Vision-Language Models

- [ ] FlightGPT: Towards Generalizable and InterpretableUAVVision-and-Language Navigation with Vision-Language Models | https://aclanthology.org/2025.emnlp-main.338/

- **Link**: https://aclanthology.org/2025.emnlp-main.338/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Unmanned Aerial Vehicle (UAV) Vision-and-Language Navigation (VLN) is vital for applications such as disaster response, logistics delivery, and urban inspection. However, existing methods often struggle with insufficient multimodal fusion, weak generalization, and poor interpretability. To address these challenges, we propose FlightGPT, a novel UAV VLN framework built upon Vision-Language Models (VLMs) with powerful multimodal perception capabilities. We design a two-stage training pipeline: first, Supervised Fine-Tuning (SFT) using high-quality demonstrations to improve initialization and structured reasoning; then, Group Relative Policy Optimization (GRPO) algorithm, guided by a composite reward that considers goal accuracy, reasoning quality, and format compliance, to enhance generalization and adaptability. Furthermore, FlightGPT introduces a Chain-of-Thought (CoT)-based reasoning mechanism to improve decision interpretability. Extensive experiments on the city-scale dataset CityNav demonstrate that FlightGPT achieves state-of-the-art performance across all scenarios, with a 9.22% higher success rate than the strongest baseline in unseen environments. Our implementation is publicly available.

</details>

---

## 133. ZoomEye: Enhancing MultimodalLLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration

- [ ] ZoomEye: Enhancing MultimodalLLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration | https://aclanthology.org/2025.emnlp-main.335/

- **Link**: https://aclanthology.org/2025.emnlp-main.335/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in vision-language understanding. Recently, with the integration of test-time scaling techniques, these models have also shown strong potential in visual reasoning. However, most existing reasoning approaches remain text-level in nature: MLLMs are prompted to explore various combinations of textual tokens via their underlying language model, while the visual input remains fixed throughout the reasoning process. This paradigm limits the model’s ability to fully exploit rich visual information, particularly when dealing with images containing numerous fine-grained elements. In such cases, vision-level reasoning becomes crucial—where models dynamically zoom into specific regions of the image to gather detailed visual cues necessary for accurate decision-making. In this paper, we propose Zoom Eye, a training-free, model-agnostic tree search algorithm tailored for vision-level reasoning. Zoom Eye treats an image as a hierarchical tree structure, where each child node represents a zoomed-in sub-region of its parent, and the root corresponds to the full image. The algorithm enables MLLMs to simulate human-like zooming behavior by navigating from root to leaf nodes in search of task-relevant visual evidence. We experiment on a series of elaborate high-resolution benchmarks and the results demonstrate that Zoom Eye not only consistently improves the performance of a series of MLLMs with large margin (e.g., InternVL2.5-8B increases by 15.71% and 17.69% on HR-Bench) but also enables small 3-8B MLLMs to outperform strong large models such as GPT-4o.

</details>

---

## 134. Multimodal Language Models See Better When They Look Shallower

- [ ] Multimodal Language Models See Better When They Look Shallower | https://aclanthology.org/2025.emnlp-main.339/

- **Link**: https://aclanthology.org/2025.emnlp-main.339/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) typically extract visual features from the final layers of a pretrained Vision Transformer (ViT). This widespread deep-layer bias, however, is largely driven by empirical convention rather than principled analysis. While prior studies suggest that different ViT layers capture different types of information—shallower layers focusing on fine visual details and deeper layers aligning more closely with textual semantics, the impact of this variation on MLLM performance remains underexplored. We present the first comprehensive study of visual layer selection for MLLMs, analyzing representation similarity across ViT layers to establish shallow, middle, and deep layer groupings. Through extensive evaluation of MLLMs (1.4B–7B parameters) across 10 benchmarks encompassing 60+ tasks, we find that while deep layers excel in semantic-rich tasks like OCR, shallow and middle layers significantly outperform them on fine-grained visual tasks including counting, positioning, and object localization. Building on these insights, we propose a lightweight feature fusion method that strategically incorporates shallower layers, achieving consistent improvements over both single-layer and specialized fusion baselines. Our work offers the first principled study of visual layer selection in MLLMs, showing that MLLMs can often see better when they look shallower.

</details>

---

## 135. ViLBench: A Suite for Vision-Language Process Reward Modeling

- [ ] ViLBench: A Suite for Vision-Language Process Reward Modeling | https://aclanthology.org/2025.emnlp-main.344/

- **Link**: https://aclanthology.org/2025.emnlp-main.344/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Process-supervised reward models serve as a fine-grained function that provides detailed step-wise feedback to model responses, facilitating effective selection of reasoning trajectories for complex tasks. Despite its advantages, evaluation on PRMs remains less explored, especially in the multimodal domain. To address this gap, this paper first benchmarks current vision large language models (VLLMs) as two types of reward models: output reward models (ORMs) and process reward models (PRMs) on multiple vision-language benchmarks, which reveal that neither ORM nor PRM consistently outperforms across all tasks, and superior VLLMs do not necessarily yield better rewarding performance. To further advance evaluation, we introduce ViLBench, a vision-language benchmark designed to require intensive process reward signals. Notably, OpenAI’s GPT-4o with Chain-of-Thought (CoT) achieves only 27.3% accuracy, challenging current VLLMs. Lastly, we preliminarily showcase a promising pathway towards bridging the gap between general VLLMs and reward models—by collecting 73.6K vision-language process reward data using an enhanced tree-search algorithm, our 3B model is able to achieve an average improvement of 3.3% over standard CoT and up to 2.5% compared to its untrained counterpart on ViLBench by selecting OpenAI o1’s generations. We will release our code, model, and data at https://ucsc-vlaa.github.io/ViLBench.

</details>

---

## 136. LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts

- [ ] LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts | https://aclanthology.org/2025.emnlp-main.368/

- **Link**: https://aclanthology.org/2025.emnlp-main.368/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Redundancy of visual tokens in multi-modal large language models (MLLMs) significantly reduces their computational efficiency. Recent approaches, such as resamplers and summarizers, have sought to reduce the number of visual tokens, but at the cost of visual reasoning ability. To address this, we propose LEO-Mini, a novel MLLM that significantly reduces the number of visual tokens and simultaneously boosts visual reasoning capabilities. For efficiency, LEO-Mini incorporates CoTR, a novel token reduction module to consolidate a large number of visual tokens into a smaller set of tokens, using the similarity between visual tokens, text tokens, and a compact learnable query. For effectiveness, to scale up the model’s ability with minimal computational overhead, LEO-Mini employs MMoE, a novel mixture of multi-modal experts module. MMoE employs a set of LoRA experts with a novel router to switch between them based on the input text and visual tokens instead of only using the input hidden state. MMoE also includes a general LoRA expert that is always activated to learn general knowledge for LLM reasoning. For extracting richer visual features, MMoE employs a set of vision experts trained on diverse domain-specific data. To demonstrate LEO-Mini’s improved efficiency and performance, we evaluate it against existing efficient MLLMs on various benchmark vision-language tasks.

</details>

---

## 137. SpecVLM: Enhancing Speculative Decoding of VideoLLMs via Verifier-Guided Token Pruning

- [ ] SpecVLM: Enhancing Speculative Decoding of VideoLLMs via Verifier-Guided Token Pruning | https://aclanthology.org/2025.emnlp-main.366/

- **Link**: https://aclanthology.org/2025.emnlp-main.366/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning.Building on our novel finding that the draft model’s speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens to enable efficient speculation without sacrificing accuracy. To achieve this, we performs a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes remaining redundant ones in a spatially uniform manner.Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68×decoding speedup for LLaVA-OneVision-72B and 2.11×speedup for Qwen2.5-VL-32B. Code is available at https://github.com/zju-jiyicheng/SpecVLM.

</details>

---

## 138. AesBiasBench: Evaluating Bias and Alignment in Multimodal Language Models for Personalized Image Aesthetic Assessment

- [ ] AesBiasBench: Evaluating Bias and Alignment in Multimodal Language Models for Personalized Image Aesthetic Assessment | https://aclanthology.org/2025.emnlp-main.386/

- **Link**: https://aclanthology.org/2025.emnlp-main.386/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are increasingly applied in Personalized Image Aesthetic Assessment (PIAA) as a scalable alternative to expert evaluations. However, their predictions may reflect subtle biases influenced by demographic factors such as gender, age, and education. In this work, we propose AesBiasBench, a benchmark designed to evaluate MLLMs along two complementary dimensions: (1) stereotype bias, quantified by measuring variations in aesthetic evaluations across demographic groups; and (2) alignment between model outputs and genuine human aesthetic preferences. Our benchmark covers three subtasks (Aesthetic Perception, Assessment, Empathy) and introduces structured metrics (IFD, NRD, AAS) to assess both bias and alignment. We evaluate 19 MLLMs, including proprietary models (e.g., GPT-4o, Claude-3.5-Sonnet) and open-source models (e.g., InternVL-2.5, Qwen2.5-VL). Results indicate that smaller models exhibit stronger stereotype biases, whereas larger models align more closely with human preferences. Incorporating identity information often exacerbates bias, particularly in emotional judgments. These findings underscore the importance of identity-aware evaluation frameworks in subjective vision-language tasks.

</details>

---

## 139. SURE: Safety Understanding and Reasoning Enhancement for Multimodal Large Language Models

- [ ] SURE: Safety Understanding and Reasoning Enhancement for Multimodal Large Language Models | https://aclanthology.org/2025.emnlp-main.384/

- **Link**: https://aclanthology.org/2025.emnlp-main.384/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) demonstrate impressive capabilities by integrating visual and textual information. However, the incorporation of visual modalities also introduces new and complex safety risks, rendering even the most advanced models vulnerable to sophisticated jailbreak attacks. This paper first analyzes the impact of inserting safety reasoning prompt on various aspects of the model. We find that this external method can help the model resist jailbreak attacks to some extent, but the model still fails to distinguish specific semantic scenarios, resulting in a significantly increased refusal rate for benign queries. Inspired by this, we propose a novel training framework,SURE(Safety Understanding and Reasoning Enhancement for Multimodal Large Language Models), designed to help models internalize chain-of-thought-based safety decision-making capabilities. Extensive experiments demonstrate that SURE significantly improves model safety while effectively avoiding over-defense, achieving a good balance between safety and generality. Finally, we create a large-scale multimodal safety reasoning dataset, MLLM-SCoT-Plus, to facilitate research on safety alignment in multimodal models.Our code and the dataset are publicly available athttps://github.com/hfutml/SURE.

</details>

---

## 140. TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration

- [ ] TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration | https://aclanthology.org/2025.emnlp-main.39/

- **Link**: https://aclanthology.org/2025.emnlp-main.39/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision–language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input ICL sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens oftask mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we presentTACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures ICL sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a novel and valuable perspective for interpreting and improving multimodal ICL.

</details>

---

## 141. UnCo: Uncertainty-Driven Collaborative Framework of Large and Small Models for Grounded MultimodalNER

- [ ] UnCo: Uncertainty-Driven Collaborative Framework of Large and Small Models for Grounded MultimodalNER | https://aclanthology.org/2025.emnlp-main.388/

- **Link**: https://aclanthology.org/2025.emnlp-main.388/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Grounded Multimodal Named Entity Recognition (GMNER) is a new information extraction task. It requires models to extract named entities and ground them to real-world visual objects. Previous methods, relying on domain-specific fine-tuning, struggle with unseen multimodal entities due to limited knowledge and generalization. Recently, multimodal large language models (MLLMs) have demonstrated strong open-set abilities. However, their performance is hindered by the lack of in-domain knowledge due to costly training for GMNER datasets. To address these limitations, we propose **UnCo**, a two-stage Uncertainty-driven Collaborative framework that leverages the complementary strengths of small fine-tuned models and MLLMs. Specifically, **in stage one**, we equip the small model with a unified uncertainty estimation (UE) for multimodal entities. This enables the small model to express"I do not know"when recognizing unseen entities beyond its capabilities. Predictions with high uncertainty are then filtered and delegated to the MLLM. **In stage two**, an Uncertainty-aware Hierarchical Correction mechanism guides the MLLM to refine uncertain predictions using its open-domain knowledge. Ultimately, UnCo effectively retains the in-domain knowledge of small models while utilizing the capabilities of MLLMs to handle unseen samples. Extensive experiments demonstrate UnCo’s effectiveness on two GMNER benchmarks.

</details>

---

## 142. DiscoSG: Towards Discourse-Level Text Scene Graph Parsing through Iterative Graph Refinement

- [ ] DiscoSG: Towards Discourse-Level Text Scene Graph Parsing through Iterative Graph Refinement | https://aclanthology.org/2025.emnlp-main.398/

- **Link**: https://aclanthology.org/2025.emnlp-main.398/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) generate discourse-level, multi-sentence visual descriptions, challenging text scene graph parsers built for single-sentence caption-to-graph mapping. Current approaches typically merge sentence-level parsing outputs for discourse input, often missing phenomena like cross-sentence coreference, resulting in fragmented graphs and degraded downstream VLM task performance. We introduce a new task, Discourse-level text Scene Graph parsing (DiscoSG), and release DiscoSG-DS, a dataset of 400 expert-annotated and 8,430 synthesised multi-sentence caption-graph pairs. Each caption averages 9 sentences, and each graph contains at least 3×more triples than those in existing datasets. Fine-tuning GPT-4o on DiscoSG-DS yields over 40% higher SPICE than the strongest sentence-merging baseline. However, its high inference cost and licensing restrict open-source use, and smaller fine-tuned open-source models (e.g., Flan-T5) perform poorly on dense graph generation. To bridge this gap, we propose DiscoSG-Refiner, which drafts a base graph using a seed parser and iteratively refines it with a second model, improving robustness for complex graph generation. Using two small fine-tuned Flan-T5-Base models, DiscoSG-Refiner improves SPICE by ~30% over the baseline while achieving86×faster inference than GPT-4o. It also delivers consistent gains on downstream VLM tasks, including discourse-level caption evaluation and hallucination detection, outperforming alternative parsers. Code and data are available at https://github.com/ShaoqLin/DiscoSG .

</details>

---

## 143. Automating Steering for Safe Multimodal Large Language Models

- [ ] Automating Steering for Safe Multimodal Large Language Models | https://aclanthology.org/2025.emnlp-main.41/

- **Link**: https://aclanthology.org/2025.emnlp-main.41/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model’s internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.

</details>

---

## 144. AbsVis – Benchmarking How Humans and Vision-Language Models “See” Abstract Concepts in Images

- [ ] AbsVis – Benchmarking How Humans and Vision-Language Models “See” Abstract Concepts in Images | https://aclanthology.org/2025.emnlp-main.417/

- **Link**: https://aclanthology.org/2025.emnlp-main.417/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Abstract concepts like mercy and peace often lack clear visual grounding, and thus challenge humans and models to provide suitable image representations. To address this challenge, we introduce AbsVis – a dataset of 675 images annotated with 14,175 concept–explanation attributions from humans and two Vision-Language Models (VLMs: Qwen and LLaVA), where each concept is accompanied by a textual explanation. We compare human and VLM attributions in terms of diversity, abstractness, and alignment, and find that humans attribute more varied concepts. AbsVis also includes 2,680 human preference judgments evaluating the quality of a subset of these annotations, showing that overlapping concepts (attributed by both humans and VLMs) are most preferred. Explanations clarify and strengthen the perceived attributions, both from humans and VLMs. Explanations clarify and strengthen the perceived attributions, both from human and VLMs. Finally, we show that VLMs can approximate human preferences and use them to fine-tune VLMs via Direct Preference Optimization (DPO), yielding improved alignments with preferred concept–explanation pairs.

</details>

---

## 145. Does Acceleration Cause Hidden Instability in Vision Language Models? Uncovering Instance-Level Divergence Through a Large-Scale Empirical Study

- [ ] Does Acceleration Cause Hidden Instability in Vision Language Models? Uncovering Instance-Level Divergence Through a Large-Scale Empirical Study | https://aclanthology.org/2025.emnlp-main.425/

- **Link**: https://aclanthology.org/2025.emnlp-main.425/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) are powerful yet computationally intensive for widespread practical deployments. To address such challenge without costly re-training, post-training acceleration techniques like quantization and token reduction are extensively explored. However, current acceleration evaluations primarily target minimal overall performance degradation, overlooking a crucial question: does the accelerated model still give the same answers to the same questions as it did before acceleration? This is vital for stability-centered industrial applications where consistently correct answers for specific, known situations are paramount, such as in AI-based disease diagnosis. We systematically investigate this for accelerated VLMs, testing four leading models (LLaVA-1.5, LLaVA-Next, Qwen2-VL, Qwen2.5-VL) with eight acceleration methods on ten multi-modal benchmarks. Our findings are stark: despite minimal aggregate performance drops, accelerated models changed original answers up to 20% of the time. Critically, up to 6.5% of these changes converted correct answers to incorrect. Input perturbations magnified these inconsistencies, and the trend is confirmed by case studies with the medical VLM LLaVA-Med. This research reveals a significant oversight in VLM acceleration, stressing an urgent need for instance-level stability checks to ensure trustworthy real-world deployment.

</details>

---

## 146. DocReRank: Single-Page Hard Negative Query Generation for Training Multi-ModalRAGRerankers

- [ ] DocReRank: Single-Page Hard Negative Query Generation for Training Multi-ModalRAGRerankers | https://aclanthology.org/2025.emnlp-main.436/

- **Link**: https://aclanthology.org/2025.emnlp-main.436/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Rerankers play a critical role in multimodal Retrieval-Augmented Generation (RAG) by refining ranking of an initial set of retrieved documents. Rerankers are typically trained using hard negative mining, whose goal is to select pages for each query which rank high, but are actually irrelevant. However, this selection process is typically passive and restricted to what the retriever can find in the available corpus, leading to several inherent limitations. These include: limited diversity, negative examples which are often not hard enough, low controllability, and frequent false negatives which harm training. Our paper proposes an alternative approach: Single-Page Hard Negative Query Generation, which goes the other way around. Instead of retrieving negative pages per query, we generate hard negative queries per page. Using an automated LLM-VLM pipeline, and given a page and its positive query, we create hard negatives by rephrasing the query to be as similar as possible in form and context, yet not answerable from the page. This paradigm enables fine-grained control over the generated queries, resulting in diverse, hard, and targeted negatives. It also supports efficient false negative verification. Our experiments show that rerankers trained with data generated using our approach outperform existing models and significantly improve retrieval performance.

</details>

---

## 147. VELA: AnLLM-Hybrid-as-a-Judge Approach for Evaluating Long Image Captions

- [ ] VELA: AnLLM-Hybrid-as-a-Judge Approach for Evaluating Long Image Captions | https://aclanthology.org/2025.emnlp-main.438/

- **Link**: https://aclanthology.org/2025.emnlp-main.438/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this study, we focus on the automatic evaluation of long and detailed image captions generated by multimodal Large Language Models (MLLMs). Most existing automatic evaluation metrics for image captioning are primarily designed for short captions and are not suitable for evaluating long captions. Moreover, recent LLM-as-a-Judge approaches suffer from slow inference due to their reliance on autoregressive inference and early fusion of visual information. To address these limitations, we propose VELA, an automatic evaluation metric for long captions developed within a novel LLM-Hybrid-as-a-Judge framework. Furthermore, we propose LongCap-Arena, a benchmark specifically designed for evaluating metrics for long captions. This benchmark comprises 7,805 images, the corresponding human-provided long reference captions and long candidate captions, and 32,246 human judgments from three distinct perspectives: Descriptiveness, Relevance, and Fluency. We demonstrated that VELA outperformed existing metrics and achieved superhuman performance on LongCap-Arena.

</details>

---

## 148. Language-Guided Temporal Token Pruning for EfficientVideoLLMProcessing

- [ ] Language-Guided Temporal Token Pruning for EfficientVideoLLMProcessing | https://aclanthology.org/2025.emnlp-main.451/

- **Link**: https://aclanthology.org/2025.emnlp-main.451/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) struggle with long-form videos due to the quadratic complexity of attention mechanisms. We propose Language-Guided Temporal Token Pruning (LGTTP), which leverages temporal cues from queries to adaptively prune video tokens, preserving contextual continuity while reducing computational overhead. Unlike uniform pruning or keyframe selection, LGTTP retains higher token density in temporally relevant segments. Our model-agnostic framework integrates with TimeChat and LLaVA-Video, achieving a 65% reduction in computation while preserving 97-99% of the original performance. On QVHighlights, LGTTP improves HIT@1 by +9.5%, and on Charades-STA, it retains 99.6% of R@1. It excels on queries with explicit temporal markers and remains effective across general video understanding tasks.

</details>

---

## 149. HVGuard: Utilizing Multimodal Large Language Models for Hateful Video Detection

- [ ] HVGuard: Utilizing Multimodal Large Language Models for Hateful Video Detection | https://aclanthology.org/2025.emnlp-main.456/

- **Link**: https://aclanthology.org/2025.emnlp-main.456/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid growth of video platforms has transformed information dissemination and led to an explosion of multimedia content. However, this widespread reach also introduces risks, as some users exploit these platforms to spread hate speech, which is often concealed through complex rhetoric, making hateful video detection a critical challenge. Existing detection methods rely heavily on unimodal analysis or simple feature fusion, struggling to capture cross-modal interactions and reason through implicit hate in sarcasm and metaphor. To address these limitations, we propose HVGuard, the first reasoning-based hateful video detection framework with multimodal large language models (MLLMs). Our approach integrates Chain-of-Thought (CoT) reasoning to enhance multimodal interaction modeling and implicit hate interpretation. Additionally, we design a Mixture-of-Experts (MoE) network for efficient multimodal fusion and final decision-making. The framework is modular and extensible, allowing flexible integration of different MLLMs and encoders. Experimental results demonstrate that HVGuard outperforms all existing advanced detection tools, achieving an improvement of 6.88% to 13.13% in accuracy and 9.21% to 34.37% in M-F1 on two public datasets covering both English and Chinese.

</details>

---

## 150. Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models

- [ ] Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models | https://aclanthology.org/2025.emnlp-main.470/

- **Link**: https://aclanthology.org/2025.emnlp-main.470/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in text-only “slow-thinking” reasoning have prompted efforts to transfer this capability to vision-language models (VLMs), for training visual reasoning models (VRMs). However, such transfer faces critical challenges: Effective “slow thinking” in VRMs requires visual reflection, the ability to check the reasoning process based on visual information. Through quantitative analysis, we observe that current VRMs exhibit limited visual reflection, as their attention to visual information diminishes rapidly with longer generated responses. To address this challenge, we propose a new VRM Reflection-V, which enhances visual reflection based on reasoning data construction for cold-start and reward design for reinforcement learning (RL). Firstly, we construct vision-centered reasoning data by leveraging an agent that interacts between VLMs and reasoning LLMs, enabling cold-start learning of visual reflection patterns. Secondly, a visual attention based reward model is employed during RL to encourage reasoning based on visual information. Therefore, Reflection-V demonstrates significant improvements across multiple visual reasoning benchmarks. Furthermore, Reflection-V maintains a stronger and more consistent reliance on visual information during visual reasoning, indicating effective enhancement in visual reflection capabilities.

</details>

---

## 151. TCPO: Thought-Centric Preference Optimization for Effective Embodied Decision-making

- [ ] TCPO: Thought-Centric Preference Optimization for Effective Embodied Decision-making | https://aclanthology.org/2025.emnlp-main.484/

- **Link**: https://aclanthology.org/2025.emnlp-main.484/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Using effective generalization capabilities of vision language models (VLMs) in context-specific dynamic tasks for embodied artificial intelligence remains a significant challenge. Although supervised fine-tuned models can better align with the real physical world, they still exhibit sluggish responses and hallucination issues in dynamically changing environments, necessitating further alignment. Existing post-SFT methods, reliant on reinforcement learning and chain-of-thought (CoT) approaches, are constrained by sparse rewards and action-only optimization, resulting in low sample efficiency, poor consistency, and model degradation. To address these issues, this paper proposes Thought-Centric Preference Optimization (TCPO) for effective embodied decision-making. Specifically, TCPO introduces a stepwise preference-based optimization approach, transforming sparse reward signals into richer step sample pairs. It emphasizes the alignment of the model’s intermediate reasoning process, mitigating the problem of model degradation. Moreover, by incorporating Action Policy Consistency Constraint (APC), it further imposes consistency constraints on the model output. Experiments in the ALFWorld environment demonstrate an average success rate of **26.67%**, achieving a **6%** improvement over RL4VLM and validating the effectiveness of our approach in mitigating model degradation after fine-tuning. These results highlight the potential of integrating preference-based learning techniques with CoT processes to enhance the decision-making capabilities of vision-language models in embodied agents.

</details>

---

## 152. Reimagining Safety Alignment with An Image

- [ ] Reimagining Safety Alignment with An Image | https://aclanthology.org/2025.emnlp-main.485/

- **Link**: https://aclanthology.org/2025.emnlp-main.485/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) excel in diverse applications but face dual challenges: generating harmful content under jailbreak attacks and over-refusing benign queries due to rigid safety mechanisms. These issues severely affect the application of LLMs, especially in the medical and education fields. Existing approaches can be divided into three types: contrastive decoding, activation manipulation, and prompting strategies. However, all these approaches face challenges like inefficiency, fragility, or architectural constraints,ultimately failing to strike a balance between safety and usability. These problems are more obvious in multimodal large language models (MLLMs), especially in terms of heightened over-refusal in cross-modal tasks and new security risks arising from expanded attack surfaces. We propose Magic Image, an optimization-driven visual prompt framework that enhances security and reduces over-refusal at the same time. The Magic Image is optimized using gradients derived from harmful/benign training samples. Using the magic image can modify the model’s original safety alignment, maintaining robust safety while reducing unnecessary denials. Experiments demonstrate its effectiveness in preserving model performance and improving safety-responsiveness balance across datasets, including unseen data, offering a practical solution for reliable MLLM deployment.

</details>

---

## 153. Visual Contextual Attack: JailbreakingMLLMs with Image-Driven Context Injection

- [ ] Visual Contextual Attack: JailbreakingMLLMs with Image-Driven Context Injection | https://aclanthology.org/2025.emnlp-main.487/

- **Link**: https://aclanthology.org/2025.emnlp-main.487/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments.Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack.VisCo fabricates contextual dialogue using four distinct vision-focused strategies, dynamically generating auxiliary images when necessary to construct a vision-centric jailbreak scenario.To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which achieves a toxicity score of 2.48 and an ASR of 22.2%. Code: https://github.com/Dtc7w3PQ/Visco-Attack.

</details>

---

## 154. CROP: Contextual Region-Oriented Visual Token Pruning

- [ ] CROP: Contextual Region-Oriented Visual Token Pruning | https://aclanthology.org/2025.emnlp-main.492/

- **Link**: https://aclanthology.org/2025.emnlp-main.492/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current VLM-based VQA methods often process entire images, leading to excessive visual tokens that include redundant information irrelevant to the posed question. This abundance of unnecessary image details creates numerous visual tokens, drastically increasing memory and computational requirements in VLMs. To address this, we propose Contextual Region-Oriented Visual Token Pruning (CROP), a novel framework to compress visual tokens through a two-step process: Localization and Pruning. Specifically, CROP first employs an efficient model to identify the contextual region relevant to the input query. Subsequently, two distinct strategies are introduced for pruning: (1) Pre-LLM Compression (PLC), which adaptively compresses different image regions with varying ratios, and (2) Inner-LLM Pruning (ILP), a training-free method that prunes tokens within early LLM layers guided by the identified contextual region. Extensive experiments on a wide range of VQA tasks demonstrate that CROP significantly outperforms existing visual token pruning methods and achieves state-of-the-art performance.

</details>

---

## 155. AIKnows Where You Are: Exposure, Bias, and Inference in Multimodal Geolocation withKoreaGEO

- [ ] AIKnows Where You Are: Exposure, Bias, and Inference in Multimodal Geolocation withKoreaGEO | https://aclanthology.org/2025.emnlp-main.501/

- **Link**: https://aclanthology.org/2025.emnlp-main.501/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in vision-language models (VLMs) have enabled accurate image-based geolocation, raising serious concerns about location privacy risks in everyday social media posts. Yet, a systematic evaluation of such risks is still lacking: existing benchmarks show coarse granularity, linguistic bias, and a neglect of multimodal privacy risks. To address these gaps, we introduce KoreaGEO, the first fine-grained, multimodal, and privacy-aware benchmark for geolocation, built on Korean street views. The benchmark covers four socio-spatial clusters and nine place types with rich contextual annotations and two captioning styles that simulate real-world privacy exposure. To evaluate mainstream VLMs, we design a three-path protocol spanning image-only, functional-caption, and high-risk-caption inputs, enabling systematic analysis of localization accuracy, spatial bias, and reasoning behavior. Results show that input modality exerts a stronger influence on localization precision and privacy exposure than model scale or architecture, with high-risk captions substantially boosting accuracy. Moreover, they highlight structural prediction biases toward core cities.

</details>

---

## 156. Stop Looking for “Important Tokens” in Multimodal Language Models: Duplication Matters More

- [ ] Stop Looking for “Important Tokens” in Multimodal Language Models: Duplication Matters More | https://aclanthology.org/2025.emnlp-main.505/

- **Link**: https://aclanthology.org/2025.emnlp-main.505/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision tokens in multimodal large language models often dominate huge computational overhead due to their excessive length compared to linguistic modality. Abundant recent methods aim to solve this problem with token pruning, which first defines an importance criterion for tokens and then prunes the unimportant vision tokens during inference. However, in this paper, we show that the importance is not an ideal indicator to decide whether a token should be pruned. Surprisingly, it usually results in inferior performance than random token pruning and leading to incompatibility to efficient attention computation operators. Instead, we propose DART (Duplication-Aware Reduction of Tokens), which prunes tokens based on its duplication with other tokens, leading to significant and training-free acceleration. Concretely, DART selects a small subset of pivot tokens and then retains the tokens with low duplication to the pivots, ensuring minimal information loss during token pruning. Experiments demonstrate that DART can prune 88.9% vision tokens while maintaining comparable performance, leading to a 1.99×and 2.99×speed-up in total time and prefilling stage, respectively, with good compatibility to efficient attention operators.

</details>

---

## 157. Probing Logical Reasoning ofMLLMs in Scientific Diagrams

- [ ] Probing Logical Reasoning ofMLLMs in Scientific Diagrams | https://aclanthology.org/2025.emnlp-main.542/

- **Link**: https://aclanthology.org/2025.emnlp-main.542/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We examine how multimodal large language models (MLLMs) perform logical inference grounded in visual information. We first construct a dataset of food web/chain images, along with questions that follow seven structured templates with progressively more complex reasoning involved. We show that complex reasoning about entities in the images remains challenging (even with elaborate prompts) and that visual information is underutilized.

</details>

---

## 158. Static or Dynamic: Towards Query-Adaptive Token Selection for Video Question Answering

- [ ] Static or Dynamic: Towards Query-Adaptive Token Selection for Video Question Answering | https://aclanthology.org/2025.emnlp-main.545/

- **Link**: https://aclanthology.org/2025.emnlp-main.545/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video question answering benefits from the rich information in videos, enabling various applications. However, the large volume of tokens generated from long videos presents challenges to memory efficiency and model performance. To alleviate this, existing works propose to compress video inputs, but often overlook the varying importance of static and dynamic information across different queries, leading to inefficient token usage within limited budgets. We propose a novel token selection strategy, explore-then-select, that adaptively adjusts static and dynamic information based on question requirements. Our framework first explores different token allocations between key frames, which preserve spatial details, and delta frames, which capture temporal changes. Then it employs a query-aware attention-based metric to select the optimal token combination without model updates. Our framework is plug-and-play and can be seamlessly integrated within diverse video language models. Extensive experiments show that our method achieves significant performance improvements (up to 5.8%) on multiple video question answering benchmarks. Our code is available at *https://github.com/ANDgate99/Explore-Then-Select*.

</details>

---

## 159. Can Vision-Language Models Solve Visual Math Equations?

- [ ] Can Vision-Language Models Solve Visual Math Equations? | https://aclanthology.org/2025.emnlp-main.547/

- **Link**: https://aclanthology.org/2025.emnlp-main.547/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite strong performance in visual understanding and language-based reasoning, Vision-Language Models (VLMs) struggle with tasks requiring integrated perception and symbolic computation. We study this limitation through visual equation solving, where mathematical equations are embedded in images, variables are represented by object icons, and coefficients must be inferred by counting. While VLMs perform well on textual equations, they fail on visually grounded counterparts. To understand this gap, we decompose the task into coefficient counting and variable recognition, and find that counting is the primary bottleneck, even when recognition is accurate. We also observe that composing recognition and reasoning introduces additional errors, highlighting challenges in multi-step visual reasoning. Finally, as equation complexity increases, symbolic reasoning itself becomes a limiting factor. These findings reveal key weaknesses in current VLMs and point toward future improvements in visually grounded mathematical reasoning.

</details>

---

## 160. SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration inMLLMs

- [ ] SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration inMLLMs | https://aclanthology.org/2025.emnlp-main.55/

- **Link**: https://aclanthology.org/2025.emnlp-main.55/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities by integrating visual and textual inputs, yet modality alignment remains one of the most challenging aspects. Current MLLMs typically rely on simple adapter architectures and pretraining approaches to bridge vision encoders with large language models (LLM), guided by image-level supervision. We identify this paradigm often leads to suboptimal alignment between modalities, significantly constraining the LLM’s ability to properly interpret and reason with visual features particularly for smaller language models. To address this fundamental limitation, we propose Supervised Embedding Alignment (SEA), a token-level supervision alignment method that enables more precise visual-text alignment during pretraining. SEA introduces minimal computational overhead while preserving language capabilities and substantially improving cross-modal understanding. Our comprehensive analyses reveal critical insights into the adapter’s role in multimodal integration, and extensive experiments demonstrate that SEA consistently improves performance across various model sizes, with smaller models benefiting the most (average performance gain of 7.61% for Gemma-2B). This work establishes a foundation for developing more effective alignment strategies for future multimodal systems.

</details>

---

## 161. LATTE: Learning to Think with Vision Specialists

- [ ] LATTE: Learning to Think with Vision Specialists | https://aclanthology.org/2025.emnlp-main.564/

- **Link**: https://aclanthology.org/2025.emnlp-main.564/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While open-source vision-language models perform well on simple question-answering, they still struggle with complex questions that require both perceptual and reasoning capabilities. We propose LATTE, a family of vision-language models that have LeArned to Think wiTh vision spEcialists. By offloading perception to state-of-the-art vision models, our approach enables vision-language models to focus solely on reasoning over high-quality perceptual information. To train LATTE, we synthesize and filter a large dataset of 293K multi-modal reasoning traces over perceptual outputs of vision specialists. LATTE trained on this data achieves significant 4-5% gains over baselines across 6 benchmarks covering both perception and reasoning abilities. Ablation studies reveal that the effectiveness of multi-modal reasoning traces depends on the data sources, formats, and quality of thoughts.

</details>

---

## 162. SUA: Stealthy Multimodal Large Language Model Unlearning Attack

- [ ] SUA: Stealthy Multimodal Large Language Model Unlearning Attack | https://aclanthology.org/2025.emnlp-main.565/

- **Link**: https://aclanthology.org/2025.emnlp-main.565/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) trained on massive data may memorize sensitive personal information and photos, posing serious privacy risks. To mitigate this, MLLM unlearning methods are proposed, which fine-tune MLLMs to reduce the “forget” sensitive information. However, it remains unclear whether the knowledge has been truly forgotten or just hidden in the model. Therefore, we propose to study a novel problem of LLM unlearning attack, which aims to recover the unlearned knowledge of an unlearned LLM. To achieve the goal, we propose a novel framework Stealthy Unlearning Attack (SUA) framework that learns a universal noise pattern. When applied to input images, this noise can trigger the model to reveal unlearned content. While pixel-level perturbations may be visually subtle, they can be detected in the semantic embedding space, making such attacks vulnerable to potential defenses. To improve stealthiness, we introduce an embedding alignment loss that minimizes the difference between the perturbed and denoised image embeddings, ensuring the attack is semantically unnoticeable. Experimental results show that SUA can effectively recover unlearned information from MLLMs. Furthermore, the learned noise generalizes well: a single perturbation trained on a subset of samples can reveal forgotten content in unseen images. This indicates that knowledge reappearance is not an occasional failure, but a consistent behavior.

</details>

---

## 163. Towards Statistical Factuality Guarantee for Large Vision-Language Models

- [ ] Towards Statistical Factuality Guarantee for Large Vision-Language Models | https://aclanthology.org/2025.emnlp-main.576/

- **Link**: https://aclanthology.org/2025.emnlp-main.576/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Advancements in Large Vision-Language Models (LVLMs) have demonstrated impressive performance in image-conditioned text generation; however, hallucinated outputs–text that misaligns with the visual input–pose a major barrier to their use in safety-critical applications. We introduce ConfLVLM, a conformal-prediction-based framework that achieves finite-sample distribution-free statistical guarantees to the factuality of LVLM output. Taking each generated detail as a hypothesis, ConfLVLM statistically tests factuality via efficient heuristic uncertainty measures to filter out unreliable claims. We conduct extensive experiments covering three representative application domains: general scene understanding, medical radiology report generation, and document understanding. Remarkably, ConfLVLM reduces the error rate of claims generated by LLaVa-1.5 for scene descriptions from 87.8% to 10.0% by filtering out erroneous claims with a 95.3% true positive rate. Our results further show that ConfLVLM is highly flexible, and can be applied to any black-box LVLMs paired with any uncertainty measure for any image-conditioned free-form text generation task while providing a rigorous guarantee on controlling hallucination risk.

</details>

---

## 164. CoMMIT: Coordinated Multimodal Instruction Tuning

- [ ] CoMMIT: Coordinated Multimodal Instruction Tuning | https://aclanthology.org/2025.emnlp-main.582/

- **Link**: https://aclanthology.org/2025.emnlp-main.582/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuning in multimodal large language models (MLLMs) generally involves cooperative learning between a backbone LLM and a feature encoder of non-text input modalities. The major challenge is how to efficiently find the synergy between the two modules so that LLMs can adapt their reasoning abilities to downstream tasks while feature encoders can adjust to provide more task-specific information about its modality. In this paper, we analyze the MLLM instruction tuning from both theoretical and empirical perspectives, where we find the unbalanced learning between the feature encoder and the LLM can cause problems of oscillation and biased learning that lead to sub-optimal convergence. Inspired by our findings, we propose a Multimodal Balance Coefficient that enables quantitative measurement of the balance of learning. Based on this, we further design a dynamic learning scheduler that better coordinates the learning between the LLM and feature encoder, alleviating the problems of oscillation and biased learning. In addition, we introduce an auxiliary regularization on the gradient to promote updating with larger step sizes, which potentially allows for a more accurate estimation of the proposed MultiModal Balance Coefficient and further improves the training sufficiency. Our proposed approach is agnostic to the architecture of LLM and feature encoder, so it can be generically integrated with various MLLMs. We conduct experiments on multiple downstream tasks with various MLLMs, demonstrating that the proposed method is more effective than the baselines in MLLM instruction tuning.

</details>

---

## 165. CEMTM: Contextual Embedding-based Multimodal Topic Modeling

- [ ] CEMTM: Contextual Embedding-based Multimodal Topic Modeling | https://aclanthology.org/2025.emnlp-main.590/

- **Link**: https://aclanthology.org/2025.emnlp-main.590/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce CEMTM, a context-enhanced multimodal topic model designed to infer coherent and interpretable topic structures from both short and long documents containing text and images. CEMTM builds on fine-tuned large vision language models (LVLMs) to obtain contextualized embeddings, and employs a distributional attention mechanism to weight token-level contributions to topic inference. A reconstruction objective aligns topic-based representations with the document embedding, encouraging semantic consistency across modalities. Unlike existing approaches, CEMTM can process multiple images per document without repeated encoding and maintains interpretability through explicit word-topic and document-topic distributions. Extensive experiments on six multimodal benchmarks show that CEMTM consistently outperforms unimodal and multimodal baselines, achieving a remarkable average LLM score of 2.61. Further analysis shows its effectiveness in downstream few-shot retrieval and its ability to capture visually grounded semantics in complex domains such as scientific articles.

</details>

---

## 166. SilVar: Speech-Driven Multimodal Model for Reasoning Visual Question Answering and Object Localization

- [ ] SilVar: Speech-Driven Multimodal Model for Reasoning Visual Question Answering and Object Localization | https://aclanthology.org/2025.emnlp-main.589/

- **Link**: https://aclanthology.org/2025.emnlp-main.589/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Language Models have demonstrated remarkable capabilities across various tasks, including visual question answering and image captioning. However, most models rely on text-based instructions, limiting their effectiveness in natural human-machine interactions. Moreover, the quality of language models primarily depends on reasoning and prompting techniques, such as chain-of-thought, which remain underexplored when using speech instructions. To address these challenges, we proposeSilVar, an end-to-end multimodal model that leverages speech instructions for reasoning-based visual question answering. Additionally, we investigate reasoning techniques at different levels, including conversational, simple, and complex speech instructions. SilVar is built upon CLIP, Whisper, and LLaMA 3.1-8B, enabling more intuitive interactions by allowing users to provide verbal or text-based instructions. To this end, we introduce a new dataset designed to challenge models with speech-based reasoning tasks for object localization. This dataset enhances the model’s ability to process and explain visual scenes from spoken input, moving beyond simple object recognition to reasoning-based interactions. To our knowledge, SilVar is the first open-source, speech-driven VLM. We believe SilVar will inspire the next generation of multimodal reasoning models, advancing toward expert artificial general intelligence.

</details>

---

## 167. Modeling Bottom-up Information Quality during Language Processing

- [ ] Modeling Bottom-up Information Quality during Language Processing | https://aclanthology.org/2025.emnlp-main.592/

- **Link**: https://aclanthology.org/2025.emnlp-main.592/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contemporary theories model language processing as integrating both top-down expectations and bottom-up inputs. One major prediction of such models is that the quality of the bottom-up inputs modulates ease of processing—noisy inputs should lead to difficult and effortful comprehension. We test this prediction in the domain of reading. First, we propose an information-theoretic operationalization for the “quality” of bottom-up information as the mutual information (MI) between visual information and word identity. We formalize this prediction in a mathematical model of reading as a Bayesian update. Second, we test our operationalization by comparing participants’ reading times in conditions where words’ information quality has been reduced, either by occluding their top or bottom half, with full words. We collect data in English and Chinese. We then use multimodal language models to estimate the mutual information between visual inputs and words. We use these data to estimate the specific effect of reduced information quality on reading times. Finally, we compare how information is distributed across visual forms. In English and Chinese, the upper half contains more information about word identity than the lower half. However, the asymmetry is more pronounced in English, a pattern which is reflected in the reading times.

</details>

---

## 168. D-CoDe: Scaling Image-PretrainedVLMs to Video via Dynamic Compression and Question Decomposition

- [ ] D-CoDe: Scaling Image-PretrainedVLMs to Video via Dynamic Compression and Question Decomposition | https://aclanthology.org/2025.emnlp-main.597/

- **Link**: https://aclanthology.org/2025.emnlp-main.597/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video large language models (Vid-LLMs), which excel in diverse video-language tasks, can be effectively constructed by adapting image-pretrained vision-language models (VLMs). However, this adaptation remains challenging, as it requires processing dense and temporally extended visual inputs that exceed the capacity of image-based models. This paper identifies the perception bottleneck and token overload as key challenges in extending image-based VLMs to the video domain. To address these issues, we propose D-CoDe, a training-free adaptation framework that incorporates dynamic compression and question decomposition. Specifically, dynamic compression alleviates the perception bottleneck through adaptive selection of representative frames and content-aware aggregation of spatial tokens, thereby reducing redundancy while preserving informative content. In parallel, question decomposition mitigates token overload by reformulating the original query into sub-questions, guiding the model to focus on distinct aspects of the video and enabling more comprehensive understanding. Experiments demonstrate that D-CoDe effectively improves video understanding across various benchmarks. Furthermore, strong performance on the challenging long-video benchmark highlights the potential of D-CoDe in handling complex video-language tasks. Code is available at https://github.com/hukcc/D-CoDe.

</details>

---

## 169. ChartGaze: Enhancing Chart Understanding inLVLMs with Eye-Tracking Guided Attention Refinement

- [ ] ChartGaze: Enhancing Chart Understanding inLVLMs with Eye-Tracking Guided Attention Refinement | https://aclanthology.org/2025.emnlp-main.607/

- **Link**: https://aclanthology.org/2025.emnlp-main.607/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Charts are a crucial visual medium for communicating and representing information. While Large Vision-Language Models (LVLMs) have made progress on chart question answering (CQA), the task remains challenging, particularly when models attend to irrelevant regions of the chart. In this work, we present ChartGaze, a new eye-tracking dataset that captures human gaze patterns during chart reasoning tasks. Through a systematic comparison of human and model attention, we find that LVLMs often diverge from human gaze, leading to reduced interpretability and accuracy. To address this, we propose a gaze-guided attention refinement that aligns image-text attention with human fixations. Our approach improves both answer accuracy and attention alignment, yielding gains of up to 2.56 percentage points across multiple models. These results demonstrate the promise of incorporating human gaze to enhance both the reasoning quality and interpretability of chart-focused LVLMs.

</details>

---

## 170. ICG: Improving Cover Image Generation viaMLLM-based Prompting and Personalized Preference Alignment

- [ ] ICG: Improving Cover Image Generation viaMLLM-based Prompting and Personalized Preference Alignment | https://aclanthology.org/2025.emnlp-main.617/

- **Link**: https://aclanthology.org/2025.emnlp-main.617/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in multimodal large language models (MLLMs) and diffusion models (DMs) have opened new possibilities for AI-generated content. Yet, personalized cover image generation remains underexplored, despite its critical role in boosting user engagement on digital platforms. We propose ICG, a novel framework that integrates MLLM-based prompting with personalized preference alignment to generate high-quality, contextually relevant covers. ICG extracts semantic features from item titles and reference images via meta tokens, refines them with user embeddings, and injects the resulting personalized context into the diffusion model. To address the lack of labeled supervision, we adopt a multi-reward learning strategy that combines public aesthetic and relevance rewards with a personalized preference model trained from user behavior. Unlike prior pipelines relying on handcrafted prompts and disjointed modules, ICG employs an adapter to bridge MLLMs and diffusion models for end-to-end training. Experiments demonstrate that ICG significantly improves image quality, semantic fidelity, and personalization, leading to stronger user appeal and offline recommendation accuracy in downstream tasks. As a plug-and-play adapter bridging MLLMs and diffusion models, ICG is compatible with common checkpoints and requires no ground-truth labels during optimization.

</details>

---

## 171. Improving Clustering with Positive Pairs Generated fromLLM-Driven Labels

- [ ] Improving Clustering with Positive Pairs Generated fromLLM-Driven Labels | https://aclanthology.org/2025.emnlp-main.613/

- **Link**: https://aclanthology.org/2025.emnlp-main.613/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional unsupervised clustering methods, which often rely on contrastive training of embedders, suffer from a lack of label knowledge, resulting in suboptimal performance. Furthermore, the presence of potential false negatives can destabilize the training process. Hence, we propose to improve clustering with Positive Pairs generated from LLM-driven Labels (PPLL). In the proposed framework, LLM is initially employed to cluster the data and generate corresponding mini-cluster labels. Subsequently, positive pairs are constructed based on these labels, and an embedder is trained using BYOL to obviate the need for negative pairs. Following training, the acquired label knowledge is integrated into K-means clustering. This framework enables the integration of label information throughout the training and inference processes, while mitigating the reliance on negative pairs. Additionally, it generates interpretable labels for improved understanding of clustering results. Empirical evaluations on a range of datasets demonstrate that our proposed framework consistently surpasses state-of-the-art baselines, achieving superior performance, robustness, and computational efficiency for diverse text clustering applications.

</details>

---

## 172. Mitigating Hallucinations in Vision-Language Models through Image-Guided Head Suppression

- [ ] Mitigating Hallucinations in Vision-Language Models through Image-Guided Head Suppression | https://aclanthology.org/2025.emnlp-main.631/

- **Link**: https://aclanthology.org/2025.emnlp-main.631/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite their remarkable progress in multimodal understanding tasks, large vision language models (LVLMs) often suffer from “hallucination”, generating texts misaligned with the visual context. Existing methods aimed at reducing hallucinations through inference time intervention incur a significant increase in latency. To mitigate this, we present **SPIN**, a task-agnostic attention-guided head suppression strategy that can be seamlessly integrated during inference **without incurring any significant compute or latency overhead**. We investigate whether hallucination in LVLMs can be linked to specific model components. Our analysis suggests that hallucinations can be attributed to a dynamic subset of attention heads in each layer. Leveraging this insight, for each text query token, we selectively suppress attention heads that exhibit low attention to image tokens, keeping the top-k attention heads intact. Extensive evaluations on visual question answering and image description tasks demonstrate the efficacy of SPIN in reducing hallucination scores up to **2.7x** while maintaining F1, and improving throughput by **1.8x** compared to existing alternatives.

</details>

---

## 173. ProcWorld: Benchmarking Large Model Planning in Reachability-Constrained Environments

- [ ] ProcWorld: Benchmarking Large Model Planning in Reachability-Constrained Environments | https://aclanthology.org/2025.emnlp-main.635/

- **Link**: https://aclanthology.org/2025.emnlp-main.635/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce ProcWorld, a large-scale benchmark for partially observable embodied spatial reasoning and long-term planning with large language models (LLM) and vision language models (VLM). ProcWorld features a wide range of challenging embodied navigation and object manipulation tasks, covering 16 task types, 5,000 rooms, and over 10 million evaluation trajectories with diverse data distribution. ProcWorld supports configurable observation modes, ranging from text-only descriptions to vision-only observations. It enables text-based actions to control the agent following language instructions. ProcWorld has presented significant challenges for LLMs and VLMs: (1) active information gathering given partial observations for disambiguation; (2) simultaneous localization and decision-making by tracking the spatio-temporal state-action distribution; (3) constrained reasoning with dynamic states subject to physical reachability. Our extensive evaluation of 15 foundation models and 5 reasoning algorithms (with over 1 million rollouts) indicates larger models perform better. However, ProcWorld remains highly challenging for existing state-of-the-art models and in-context learning methods due to constrained reachability and the need for combinatorial spatial reasoning.

</details>

---

## 174. MovieCORE:COgnitiveREasoning in Movies

- [ ] MovieCORE:COgnitiveREasoning in Movies | https://aclanthology.org/2025.emnlp-main.66/

- **Link**: https://aclanthology.org/2025.emnlp-main.66/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces MovieCORE, a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content. Unlike existing datasets that focus on surface-level comprehension, MovieCORE emphasizes questions that engage System-2 thinking while remaining specific to the video material. We present an innovative agentic brainstorming approach, utilizing multiple large language models (LLMs) as thought agents to generate and refine high-quality question-answer pairs. To evaluate dataset quality, we develop a set of cognitive tests assessing depth, thought-provocation potential, and syntactic complexity. We also propose a comprehensive evaluation scheme for assessing VQA model performance on deeper cognitive tasks. To address the limitations of existing video-language models (VLMs), we introduce an agentic enhancement module, Agentic Choice Enhancement (ACE), which improves model reasoning capabilities post-training by 25%. Our work contributes to advancing movie understanding in AI systems and provides valuable insights into the capabilities and limitations of current VQA models when faced with more challenging, nuanced questions about cinematic content. Our project page, dataset and code can be found at https://joslefaure.github.io/assets/html/moviecore.html.

</details>

---

## 175. Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for MultimodalLLMs

- [ ] Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for MultimodalLLMs | https://aclanthology.org/2025.emnlp-main.665/

- **Link**: https://aclanthology.org/2025.emnlp-main.665/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Multimodal Large Language Models (MLLMs) have enhanced their versatility as they integrate a growing number of modalities. Considering the heavy cost of training MLLMs, it is efficient to reuse the existing ones and extend them to more modalities through Modality-incremental Continual Learning (MCL). The exploration of MCL is in its early stages. In this work, we dive into the causes of performance degradation in MCL. We uncover that it suffers not only from forgetting as in traditional continual learning, but also from misalignment between the modality-agnostic and modality-specific components. To this end, we propose an elegantly simple MCL paradigm called “MErge then ReAlign” (MERA) to address both forgetting and misalignment. MERA avoids introducing heavy model budgets or modifying model architectures, hence is easy to deploy and highly reusable in the MLLM community. Extensive experiments demonstrate the impressive performance of MERA, holding an average of 99.84% Backward Relative Gain when extending to four modalities, achieving nearly lossless MCL performance. Our findings underscore the misalignment issue in MCL. More broadly, our work showcases how to adjust different components of MLLMs during continual learning.

</details>

---

## 176. DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models’ Understanding onIndian Culture

- [ ] DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models’ Understanding onIndian Culture | https://aclanthology.org/2025.emnlp-main.68/

- **Link**: https://aclanthology.org/2025.emnlp-main.68/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce DRISHTIKON, a first-of-its-kind multimodal and multilingual benchmark centered exclusively on Indian culture, designed to evaluate the cultural understanding of generative AI systems. Unlike existing benchmarks with a generic or global scope, DRISHTIKON offers deep, fine-grained coverage across India’s diverse regions, spanning 15 languages, covering all states and union territories, and incorporating over 64,000 aligned text-image pairs. The dataset captures rich cultural themes including festivals, attire, cuisines, art forms, and historical heritage amongst many more. We evaluate a wide range of vision-language models (VLMs), including open-source small and large models, proprietary systems, reasoning-specialized VLMs, and Indic-focused models—across zero-shot and chain-of-thought settings. Our results expose key limitations in current models’ ability to reason over culturally grounded, multimodal inputs, particularly for low-resource languages and less-documented traditions. DRISHTIKON fills a vital gap in inclusive AI research, offering a robust testbed to advance culturally aware, multimodally competent language technologies.

</details>

---

## 177. Astra: Efficient Transformer Architecture and Contrastive Dynamics Learning for Embodied Instruction Following

- [ ] Astra: Efficient Transformer Architecture and Contrastive Dynamics Learning for Embodied Instruction Following | https://aclanthology.org/2025.emnlp-main.688/

- **Link**: https://aclanthology.org/2025.emnlp-main.688/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language-action models have gained significant attention for their ability to model multimodal sequences in embodied instruction following tasks. However, most existing models rely on causal attention, which we find suboptimal for processing sequences composed of interleaved segments from different modalities. In this paper, we introduce Astra, a novel Transformer architecture featuring trajectory attention and learnable action queries, designed to efficiently process segmented multimodal trajectories and predict actions for imitation learning. Furthermore, we propose a contrastive dynamics learning objective to enhance the model’s understanding of environment dynamics and multimodal alignment, complementing the primary behavior cloning objective. Through extensive experiments on three large-scale robot manipulation benchmarks, Astra demonstrates substantial performance improvements over previous models.

</details>

---

## 178. Unmasking Deceptive Visuals: Benchmarking Multimodal Large Language Models on Misleading Chart Question Answering

- [ ] Unmasking Deceptive Visuals: Benchmarking Multimodal Large Language Models on Misleading Chart Question Answering | https://aclanthology.org/2025.emnlp-main.695/

- **Link**: https://aclanthology.org/2025.emnlp-main.695/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Misleading visualizations, which manipulate chart representations to support specific claims, can distort perception and lead to incorrect conclusions. Despite decades of research, they remain a widespread issue, posing risks to public understanding and raising safety concerns for AI systems involved in data-driven communication. While recent multimodal large language models (MLLMs) show strong chart comprehension abilities, their capacity to detect and interpret misleading charts remains unexplored. We introduce Misleading ChartQA benchmark, a large-scale multimodal dataset designed to evaluate MLLMs on misleading chart reasoning. It contains 3,026 curated examples spanning 21 misleader types and 10 chart types, each with standardized chart code, CSV data, multiple-choice questions, and labeled explanations, validated through iterative MLLM checks and exhausted expert human review. We benchmark 24 state-of-the-art MLLMs, analyze their performance across misleader types and chart formats, and propose a novel region-aware reasoning pipeline that enhances model accuracy. Our work lays the foundation for developing MLLMs that are robust, trustworthy, and aligned with the demands of responsible visual communication.

</details>

---

## 179. TVQACML: Benchmarking Text-Centric Visual Question Answering in MultilingualChinese Minority Languages

- [ ] TVQACML: Benchmarking Text-Centric Visual Question Answering in MultilingualChinese Minority Languages | https://aclanthology.org/2025.emnlp-main.705/

- **Link**: https://aclanthology.org/2025.emnlp-main.705/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-Centric Visual Question Answering (TEC-VQA) is a critical research area that requires semantic interactions between objects and scene texts. However, most existing TEC-VQA benchmarks focus on high-resource languages like English and Chinese. Although few works expanding multilingual QA pairs in non-text-centric VQA datasets through translation, which encounters a substantial “visual-textual misalignment” problem when applied to TEC-VQA. Moreover, the open-source nature of these benchmarks and the broad sources of training data for MLLMs have inevitably led to benchmark contamination, resulting in unreliable evaluation results. To alleviate this issue, we propose a contamination-free and more challenging TEC-VQA benchmark called Text-Centric Visual Question Answering in Multilingual Chinese Minority Languages(TVQACML), which involves eight languages, including Standard Chinese, Korean, and six minority languages. TVQACML supports a wide range of tasks, such as Text Recognition, Scene Text-Centric VQA, Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten Mathematical Expression Recognition (HMER), featuring 32,000 question-answer pairs across 8,000 images. Extensive experiments on TVQACML across multiple MLLMs demonstrate the effectiveness of evaluating the MLLMs and enhancing multilingual TEC-VQA performance with fine-tuning.

</details>

---

## 180. Transparent and Coherent Procedural Mistake Detection

- [ ] Transparent and Coherent Procedural Mistake Detection | https://aclanthology.org/2025.emnlp-main.706/

- **Link**: https://aclanthology.org/2025.emnlp-main.706/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Procedural mistake detection (PMD) is a challenging problem of classifying whether a human user (observed through egocentric video) has successfully executed a task (specified by a procedural text). Despite significant recent efforts, machine performance in the wild remains nonviable, and the reasoning processes underlying this performance are opaque. As such, we extend PMD to require generating visual self-dialog rationales to inform decisions. Given the impressive, mature image understanding capabilities observed in recent vision-and-language models (VLMs), we curate a suitable benchmark dataset for PMD based on individual frames. As our reformulation enables unprecedented transparency, we leverage a natural language inference (NLI) model to formulate two automated metrics for the coherence of generated rationales. We establish baselines for this reframed task, showing that VLMs struggle off-the-shelf, but with some trade-offs, their accuracy, coherence, and efficiency can be improved by incorporating these metrics into common inference and fine-tuning methods. Lastly, our multi-faceted metrics visualize common outcomes, highlighting areas for further improvement.

</details>

---

## 181. MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval

- [ ] MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval | https://aclanthology.org/2025.emnlp-main.708/

- **Link**: https://aclanthology.org/2025.emnlp-main.708/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Document Understanding is a foundational AI capability with broad applications, and Document Question Answering (DocQA) is a key evaluation task. Traditional methods convert the document into text for processing by Large Language Models (LLMs), but this process strips away critical multi-modal information like figures. While Large Vision-Language Models (LVLMs) address this limitation, their constrained input size makes multi-page document comprehension infeasible. Retrieval-augmented generation (RAG) methods mitigate this by selecting relevant pages, but they rely solely on semantic relevance, ignoring logical connections between pages and the query, which is essential for reasoning and accurate answers.To this end, we proposeMoLoRAG, a logic-aware retrieval framework for multi-modal, multi-page document understanding. By constructing a page graph that captures contextual relationships between pages, a lightweight VLM performs graph traversal to retrieve relevant pages, including those with logical connections often overlooked. This approach combines semantic and logical relevance to deliver more accurate retrieval. After retrieval, the top-Kpages are fed into arbitrary LVLMs for question answering. To enhance flexibility, MoLoRAG offers two variants: a training-free solution for easy deployment and a fine-tuned version to improve logical relevance checking. Experiments on four DocQA datasets demonstrate average improvements of 9.68% in accuracy over LVLM direct inference and 7.44% in retrieval precision over baselines. Codes and datasets are released at https://github.com/WxxShirley/MoLoRAG.

</details>

---

## 182. Vision-Free Retrieval: Rethinking Multimodal Search with Textual Scene Descriptions

- [ ] Vision-Free Retrieval: Rethinking Multimodal Search with Textual Scene Descriptions | https://aclanthology.org/2025.emnlp-main.709/

- **Link**: https://aclanthology.org/2025.emnlp-main.709/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Contrastively-trained Vision-Language Models (VLMs), such as CLIP, have become the standard approach for learning discriminative vision-language representations. However, these models often exhibit shallow language understanding, manifesting bag-of-words behaviour. These limitations are reinforced by their dual-encoder design, which induces amodality gap. Additionally, the reliance on vast web-collected data corpora for training makes the process computationally expensive and introduces significant privacy concerns. To address these limitations, in this work, we challenge the necessity of vision encoders for retrieval tasks by introducing avision-free, single-encoderretrieval pipeline. Departing from the traditional text-to-image retrieval paradigm, we migrate to a text-to-text paradigm with the assistance of VLLM-generated structured image descriptions. We demonstrate that this paradigm shift has significant advantages, including a substantial reduction of the modality gap, improved compositionality, and better performance on short and long caption queries, all attainable with only two hours of calibration on two GPUs. Additionally, substituting raw images with textual descriptions introduces a more privacy-friendly alternative for retrieval. To further assess generalisation and address some of the shortcomings of prior compositionality benchmarks, we release two benchmarks derived from Flickr30k and COCO, containing diverse compositional queries made of short captions, which we coin subFlickr and subCOCO. Our vision-free retriever matches and often surpasses traditional multimodal models. Importantly, our approach achieves state-of-the-art zero-shot performance on multiple retrieval and compositionality benchmarks, with models as small as 0.3B parameters.

</details>

---

## 183. Retrieval Enhanced Feedback via In-context Neural Error-book

- [ ] Retrieval Enhanced Feedback via In-context Neural Error-book | https://aclanthology.org/2025.emnlp-main.711/

- **Link**: https://aclanthology.org/2025.emnlp-main.711/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Language Models (LLMs) have significantly improved reasoning capabilities, with in-context learning (ICL) emerging as a key technique for adaptation without retraining. While previous works have focused on leveraging correct examples, recent research highlights the importance of learning from errors to enhance performance. However, existing methods lack a structured framework for analyzing and mitigating errors, particularly in Multimodal Large Language Models (MLLMs), where integrating visual and textual inputs adds complexity. To address this issue, we propose REFINE: Retrieval-Enhanced Feedback via In-context Neural Error-book, a teacher-student framework that systematically structures errors and provides targeted feedback. REFINE introduces three systematic queries to construct structured feedback—Feed-Target, Feed-Check, and Feed-Path—to enhance multimodal reasoning by prioritizing relevant visual information, diagnosing critical failure points, and formulating corrective actions. Unlike prior approaches that rely on redundant retrievals, REFINE optimizes structured feedback retrieval, improving inference efficiency, token usage, and scalability. Our results demonstrate substantial speedup, reduced computational costs, and successful generalization, highlighting REFINE’s potential for enhancing multimodal reasoning.

</details>

---

## 184. VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search

- [ ] VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search | https://aclanthology.org/2025.emnlp-main.72/

- **Link**: https://aclanthology.org/2025.emnlp-main.72/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models have made significant progress on many perception-focused tasks. However, their progress on reasoning-focused tasks remains limited due to the lack of high-quality and diverse training data. In this work, we aim to address the scarcity of reasoning-focused multimodal datasets. We propose VisualWebInstruct, a novel approach that leverages search engines to create a diverse and high-quality dataset spanning multiple disciplines, including mathematics, physics, finance, and chemistry, etc. Starting with a meticulously selected set of 30,000 seed images, we employ Google Image Search to identify websites containing similar images. We collect and process HTML data from over 700K unique URLs. Through a pipeline of content extraction, filtering, and synthesis, we construct a dataset of approximately 900K question-answer (QA) pairs, with 40% consisting of visual QA pairs and the remaining comprising text-based QA pairs. Models fine-tuned on VisualWebInstruct demonstrate significant performance improvements: (1) fine-tuning on Llava-OV results in 10-20 absolute points improvement across benchmarks, and (2) fine-tuning from MAmmoTH-VL yields a 5 absolute points gain across benchmarks. Our best model, MAmmoTH-VL2, achieves the best known performance with SFT without RL within the 10B parameter class on MMMU-Pro (40.7), MathVerse (42.6), and DynaMath (55.7). These results highlight the effectiveness of our dataset in enhancing the reasoning capabilities of vision-language models for complex multimodal tasks.

</details>

---

## 185. SHARP: Steering Hallucination inLVLMs via Representation Engineering

- [ ] SHARP: Steering Hallucination inLVLMs via Representation Engineering | https://aclanthology.org/2025.emnlp-main.725/

- **Link**: https://aclanthology.org/2025.emnlp-main.725/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite their impressive capabilities, Large Vision-Language Models (LVLMs) frequently generate responses that are plausible but incorrect or unsupported—commonly referred to as hallucinations. In this study, we investigate whether different types of hallucinations are reflected in the model’s internal representations by probing their encoded features. We focus on two key causes of hallucination in multimodal reasoning: (1) over-reliance on textual priors and (2) preference for user prompts over conflicting visual evidence—factors identified in prior work as frequent and impactful. Our probing results reveal that hallucinations exhibit distinguishable representational patterns, suggesting the potential for a representation-level approach to characterize and mitigate them. Motivated by these findings, we propose Steering HAllucination via RePresentation Engineering (SHARP), a representation-level intervention framework that modulates hallucination-related features during inference. SHARP identifies functional representations responsible for prior-driven biases and visual-context conflicts, and jointly adjusts the model’s internal activations in real time. We evaluate our approach extensively on three large vision-language models across multiple benchmarks. Experimental results demonstrate that SHARP effectively reduces hallucinations while preserving the performance and generalization capabilities of LVLMs.

</details>

---

## 186. Advancing Fine-Grained Visual Understanding with Multi-Scale Alignment in Multi-Modal Models

- [ ] Advancing Fine-Grained Visual Understanding with Multi-Scale Alignment in Multi-Modal Models | https://aclanthology.org/2025.emnlp-main.721/

- **Link**: https://aclanthology.org/2025.emnlp-main.721/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have achieved remarkable success in fine-grained visual understanding across a range of tasks. However, they often encounter significant challenges due to inadequate alignment for fine-grained knowledge, which restricts their ability to accurately capture local details and attain a comprehensive global perception. While recent advancements have focused on aligning object expressions with grounding information, they typically lack explicit integration of object images, which contain affluent information beyond mere texts or coordinates. To bridge this gap, we introduce a novel fine-grained visual knowledge alignment method that effectively aligns and integrates multi-scale knowledge of objects, including texts, coordinates, and images. This innovative method is underpinned by our multi-scale fine-grained enhancement data synthesis pipeline, which provides over 300K essential training data to enhance alignment and improve overall performance. Furthermore, we present TinyGroundingGPT, a series of compact models optimized for high-level alignments. With a scale of approximately 3B parameters, TinyGroundingGPT achieves outstanding results in grounding tasks while delivering performance comparable to larger MLLMs in complex visual scenarios.

</details>

---

## 187. Seeing is Believing, but How Much? A Comprehensive Analysis of Verbalized Calibration in Vision-Language Models

- [ ] Seeing is Believing, but How Much? A Comprehensive Analysis of Verbalized Calibration in Vision-Language Models | https://aclanthology.org/2025.emnlp-main.74/

- **Link**: https://aclanthology.org/2025.emnlp-main.74/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Uncertainty quantification is essential for assessing the reliability and trustworthiness of modern AI systems. Among existing approaches, verbalized uncertainty, where models express their confidence through natural language, has emerged as a lightweight and interpretable solution in large language models (LLMs). However, its effectiveness in vision-language models (VLMs) remains insufficiently studied. In this work, we conduct a comprehensive evaluation of verbalized confidence in VLMs, spanning three model categories, four task domains, and three evaluation scenarios. Our results show that current VLMs often display notable miscalibration across diverse tasks and settings. Notably, visual reasoning models (i.e., thinking with images) consistently exhibit better calibration, suggesting that modality-specific reasoning is critical for reliable uncertainty estimation. To further address calibration challenges, we introduce Visual Confidence-Aware Prompting, a two-stage prompting strategy that improves confidence alignment in multimodal settings. Overall, our study highlights the inherent miscalibration in VLMs across modalities. More broadly, our findings underscore the fundamental importance of modality alignment and model faithfulness in advancing reliable multimodal systems.

</details>

---

## 188. WISE: Weak-Supervision-Guided Step-by-Step Explanations for MultimodalLLMs in Image Classification

- [ ] WISE: Weak-Supervision-Guided Step-by-Step Explanations for MultimodalLLMs in Image Classification | https://aclanthology.org/2025.emnlp-main.741/

- **Link**: https://aclanthology.org/2025.emnlp-main.741/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown promise in visual-textual reasoning, with Multimodal Chain-of-Thought (MCoT) prompting significantly enhancing interpretability. However, existing MCoT methods rely on rationale-rich datasets and largely focus on inter-object reasoning, overlooking the intra-object understanding crucial for image classification. To address this gap, we propose WISE, a Weak-supervision-guided Step-by-step Explanation method that augments any image classification dataset with MCoTs by reformulating the concept-based representations from Concept Bottleneck Models (CBMs) into concise, interpretable reasoning chains under weak supervision. Experiments across ten datasets show that our generated MCoTs not only improve interpretability by 37% but also lead to gains in classification accuracy when used to fine-tune MLLMs. Our work bridges concept-based interpretability and generative MCoT reasoning, providing a generalizable framework for enhancing MLLMs in fine-grained visual understanding.

</details>

---

## 189. MIRROR: Multimodal Cognitive Reframing Therapy for Rolling with Resistance

- [ ] MIRROR: Multimodal Cognitive Reframing Therapy for Rolling with Resistance | https://aclanthology.org/2025.emnlp-main.751/

- **Link**: https://aclanthology.org/2025.emnlp-main.751/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have explored the use of large language models (LLMs) in psychotherapy; however, text-based cognitive behavioral therapy (CBT) models often struggle with client resistance, which can weaken therapeutic alliance. To address this, we propose a multimodal approach that incorporates nonverbal cues, which allows the AI therapist to better align its responses with the client’s negative emotional state.Specifically, we introduce a new synthetic dataset, Mirror (Multimodal Interactive Rolling with Resistance), which is a novel synthetic dataset that pairs each client’s statements with corresponding facial images. Using this dataset, we train baseline vision language models (VLMs) so that they can analyze facial cues, infer emotions, and generate empathetic responses to effectively manage client resistance.These models are then evaluated in terms of both their counseling skills as a therapist, and the strength of therapeutic alliance in the presence of client resistance. Our results demonstrate that Mirror significantly enhances the AI therapist’s ability to handle resistance, which outperforms existing text-based CBT approaches.Human expert evaluations further confirm the effectiveness of our approach in managing client resistance and fostering therapeutic alliance.

</details>

---

## 190. MUCAR: Benchmarking Multilingual Cross-Modal Ambiguity Resolution for Multimodal Large Language Models

- [ ] MUCAR: Benchmarking Multilingual Cross-Modal Ambiguity Resolution for Multimodal Large Language Models | https://aclanthology.org/2025.emnlp-main.760/

- **Link**: https://aclanthology.org/2025.emnlp-main.760/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated significant advances across numerous vision-language tasks. Due to their strong performance in image-text alignment, MLLMs can effectively understand image-text pairs with clear meanings. However, effectively resolving the inherent ambiguities in natural language and visual contexts remains challenging. Existing multimodal benchmarks typically overlook linguistic and visual ambiguities, relying mainly on unimodal context for disambiguation and thus failing to exploit the mutual clarification potential between modalities. To bridge this gap, we introduce MUCAR, a novel and challenging benchmark designed explicitly for evaluating multimodal ambiguity resolution across multilingual and cross-modal scenarios. MUCAR includes: (1) a multilingual dataset where ambiguous textual expressions are uniquely resolved by corresponding visual contexts, and (2) a dual-ambiguity dataset that systematically pairs ambiguous images with ambiguous textual contexts, with each combination carefully constructed to yield a single, clear interpretation through mutual disambiguation. Extensive evaluations involving 19 state-of-the-art multimodal models—encompassing both open-source and proprietary architectures—reveal substantial gaps compared to human-level performance, highlighting the need for future research into more sophisticated cross-modal ambiguity comprehension methods, further pushing the boundaries of multimodal reasoning.

</details>

---

## 191. Let’s Play Across Cultures: A Large Multilingual, Multicultural Benchmark for Assessing Language Models’ Understanding of Sports

- [ ] Let’s Play Across Cultures: A Large Multilingual, Multicultural Benchmark for Assessing Language Models’ Understanding of Sports | https://aclanthology.org/2025.emnlp-main.769/

- **Link**: https://aclanthology.org/2025.emnlp-main.769/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language Models (LMs) are primarily evaluated on globally popular sports, often overlooking regional and indigenous sporting traditions. To address this gap, we introduceCultSportQA, a benchmark designed to assess LMs’ understanding of traditional sports across 60 countries and 6 continents, encompassing four distinct cultural categories. The dataset features 33,000 multiple-choice questions (MCQs) across text and image modalities, categorized into primarily three key types: history-based, rule-based, and scenario-based. To evaluate model performance, we employ zero-shot, few-shot, and chain-of-thought (CoT) prompting across a diverse set of Large Language Models (LLMs), Small Language Models (SLMs), and Multimodal Large Language Models (MLMs). By providing a comprehensive multilingual and multicultural sports benchmark,CultSportQAestablishes a new standard for assessing AI’s ability to understand and reason about traditional sports. The dataset will be publicly available, fostering research in culturally aware AI systems.

</details>

---

## 192. Task-Aware Resolution Optimization for Visual Large Language Models

- [ ] Task-Aware Resolution Optimization for Visual Large Language Models | https://aclanthology.org/2025.emnlp-main.795/

- **Link**: https://aclanthology.org/2025.emnlp-main.795/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Real-world vision-language applications demand varying levels of perceptual granularity. However, most existing visual large language models (VLLMs), such as LLaVA, pre-assume a fixed resolution for downstream tasks, which leads to subpar performance. To address this problem, we first conduct a comprehensive and pioneering investigation into the resolution preferences of different vision-language tasks, revealing a correlation between resolution preferences with (1) image complexity, and (2) uncertainty variance of the VLLM at different image input resolutions. Building on this insight, we propose an empirical formula to determine the optimal resolution for a given vision-language task, accounting for these two factors as the zeroth-order and first-order terms in the Taylor expansion on a given image input. Second, based on rigorous experiments, we propose a novel parameter-efficient fine-tuning technique to extend the visual input resolution of pre-trained VLLMs to the identified optimal resolution. Extensive experiments on various vision-language tasks validate the effectiveness of our method.

</details>

---

## 193. Boosting Multi-modal Keyphrase Prediction with Dynamic Chain-of-Thought in Vision-Language Models

- [ ] Boosting Multi-modal Keyphrase Prediction with Dynamic Chain-of-Thought in Vision-Language Models | https://aclanthology.org/2025.emnlp-main.798/

- **Link**: https://aclanthology.org/2025.emnlp-main.798/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal keyphrase prediction (MMKP) aims to advance beyond text-only methods by incorporating multiple modalities of input information to produce a set of conclusive phrases. Traditional multi-modal approaches have been proven to have significant limitations in handling the challenging absence and unseen scenarios. Additionally, we identify shortcomings in existing benchmarks that overestimate model capability due to significant overlap in training tests. In this work, we propose leveraging vision-language models (VLMs) for the MMKP task. Firstly, we use two widely-used strategies,e.g., zero-shot and supervised fine-tuning (SFT) to assess the lower bound performance of VLMs. Next, to improve the complex reasoning capabilities of VLMs, we adopt Fine-tune-CoT, which leverages high-quality CoT reasoning data generated by a teacher model to finetune smaller models. Finally, to address the “overthinking” phenomenon, we propose a dynamic CoT strategy which adaptively injects CoT data during training, allowing the model to flexibly leverage its reasoning capabilities during the inference stage. We evaluate the proposed strategies on various datasets and the experimental results demonstrate the effectiveness of the proposed approaches. The code is available at https://github.com/bytedance/DynamicCoT.

</details>

---

## 194. Out of Sight, Not Out of Context? Egocentric Spatial Reasoning inVLMs Across Disjoint Frames

- [ ] Out of Sight, Not Out of Context? Egocentric Spatial Reasoning inVLMs Across Disjoint Frames | https://aclanthology.org/2025.emnlp-main.816/

- **Link**: https://aclanthology.org/2025.emnlp-main.816/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

An embodied AI assistant operating on egocentric video must integrate spatial cues across time - for instance, determining where an object A, glimpsed a few moments ago lies relative to an object B encountered later. We introduce Disjoint-3DQA , a generative QA benchmark that evaluates this ability of VLMs by posing questions about object pairs that are not co-visible in the same frame. We evaluated seven state-of-the-art VLMs and found that models lag behind human performance by 28%, with steeper declines in accuracy (60% → 30 %) as the temporal gap widens. Our analysis further reveals that providing trajectories or bird’s-eye-view projections to VLMs results in only marginal improvements, whereas providing oracle 3D coordinates leads to a substantial 20% performance increase. This highlights a core bottleneck of multi-frame VLMs in constructing and maintaining 3D scene representations over time from visual signals. Disjoint-3DQA therefore sets a clear, measurable challenge for long-horizon spatial reasoning and aims to catalyze future research at the intersection of vision, language, and embodied AI.

</details>

---

## 195. MULTIGUARD: An Efficient Approach forAISafety Moderation Across Languages and Modalities

- [ ] MULTIGUARD: An Efficient Approach forAISafety Moderation Across Languages and Modalities | https://aclanthology.org/2025.emnlp-main.819/

- **Link**: https://aclanthology.org/2025.emnlp-main.819/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose OMNIGUARD, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. OMNIGUARD improves harmful prompt classification accuracy by 11.57% over the strongest baseline in a multilingual setting, by 20.44% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, OMNIGUARD is also very efficient (≈ 120× faster than the next fastest baseline). Code and data are available at https://github.com/vsahil/OmniGuard

</details>

---

## 196. POINTS-Reader: Distillation-Free Adaptation of Vision-Language Models for Document Conversion

- [ ] POINTS-Reader: Distillation-Free Adaptation of Vision-Language Models for Document Conversion | https://aclanthology.org/2025.emnlp-main.82/

- **Link**: https://aclanthology.org/2025.emnlp-main.82/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

High-quality labeled data is essential for training accurate document conversion models, particularly in domains with complex formats such as tables, formulas, and multi-column text. However, manual annotation is both costly and time-consuming, while automatic labeling using existing models often lacks accuracy in handling such challenging scenarios. Consequently, training student models by distilling outputs from teacher models can significantly limit their performance in real-world applications. In this paper, we propose a fully automated, distillation-free framework comprising two stages for constructing high-quality document extraction datasets and models capable of handling diverse document formats and layouts. In the first stage, we introduce a method for generating large-scale, diverse synthetic data, which enables a model to extract key elements in a unified format with strong initial performance. In the second stage, we present a self-improvement approach that further adapts the model, initially trained on synthetic data, to real-world documents. Specifically, we first use the fine-tuned model to annotate real documents, then apply a suite of filtering strategies to verify annotation quality, and finally retrain the model on the verified dataset. By iteratively repeating this process, we progressively enhance both the model’s conversion capabilities and the quality of the generated data. We train a public POINTS-1.5 model to obtainPOINTS-Reader, which surpasses many existing public and proprietary models of comparable or larger size. Our model will be made publicly available.

</details>

---

## 197. LVLMs are Bad at Overhearing Human Referential Communication

- [ ] LVLMs are Bad at Overhearing Human Referential Communication | https://aclanthology.org/2025.emnlp-main.849/

- **Link**: https://aclanthology.org/2025.emnlp-main.849/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

During spontaneous conversations, speakers collaborate on novel referring expressions, which they can then re-use in subsequent conversations. Understanding such referring expressions is an important ability for an embodied agent, so that it can carry out tasks in the real world. This requires integrating and understanding language, vision, and conversational interaction. We study the capabilities of seven state-of-the-art Large Vision Language Models (LVLMs) as overhearers to a corpus of spontaneous conversations between pairs of human discourse participants engaged in a collaborative object-matching task. We find that such a task remains challenging for current LVLMs and they all fail to show a consistent performance improvement as they overhear more conversations from the same discourse participants repeating the same task for multiple rounds. We release our corpus and code for reproducibility and to facilitate future research.

</details>

---

## 198. WildScore: BenchmarkingMLLMs in-the-Wild Symbolic Music Reasoning

- [ ] WildScore: BenchmarkingMLLMs in-the-Wild Symbolic Music Reasoning | https://aclanthology.org/2025.emnlp-main.853/

- **Link**: https://aclanthology.org/2025.emnlp-main.853/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various vision-language tasks. However, their reasoning abilities in the multimodal symbolic music domain remain largely unexplored.We introduce WildScore, the first in-the-wild multimodal symbolic music reasoning and analysis benchmark, designed to evaluate MLLMs’ capacity to interpret real-world music scores and answer complex musicological queries. Each instance in WildScore is sourced from genuine musical compositions and accompanied by authentic user-generated questions and discussions, capturing the intricacies of practical music analysis. To facilitate a comprehensive evaluation, we propose a systematic taxonomy,comprising both high-level and fine-grained musicological ontologies. Furthermore, we frame complex music reasoning as multiple-choice question answering,enabling controlled and scalable assessment of MLLMs’ symbolic music understanding. Empirical benchmarking of state-of-the-art MLLMs on WildScore reveals intriguing patterns in their visual-symbolic reasoning, uncovering both promising directions and persistent challenges for MLLMs in symbolic music reasoning and analysis.We release the dataset and code.

</details>

---

## 199. Prototypical Human-AICollaboration Behaviors fromLLM-Assisted Writing in the Wild

- [ ] Prototypical Human-AICollaboration Behaviors fromLLM-Assisted Writing in the Wild | https://aclanthology.org/2025.emnlp-main.852/

- **Link**: https://aclanthology.org/2025.emnlp-main.852/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As large language models (LLMs) are used in complex writing workflows, users engage in multi-turn interactions to steer generations to better fit their needs. Rather than passively accepting output, users actively refine, explore, and co-construct text. We conduct a large scale analysis of this collaborative behavior for users engaged in writing tasks in the wild with two popular AI assistants, Bing Copilot and WildChat. Our analysis goes beyond simple task classification or satisfaction estimation common in prior work and instead characterizes how users interact with LLMs through the course of a session. We identify prototypical behaviors in how users interact with LLMs in prompts following their original request. We refer to these as Prototypical Human AI Collaboration Behaviors (PATHs) and find that a small group of PATHs explain a majority of the variation seen in user-LLM interaction. These PATHs span users revising intents, exploring texts, posing questions, adjusting style or injecting new content. Next, we find statistically significant correlations between specific writing intents and PATHs, revealing how users’ intents shape their collaboration behaviors. We conclude by discussing the implications of our findings on LLM alignment.

</details>

---

## 200. V-SEAM: Visual Semantic Editing and Attention Modulating for Causal Interpretability of Vision-Language Models

- [ ] V-SEAM: Visual Semantic Editing and Attention Modulating for Causal Interpretability of Vision-Language Models | https://aclanthology.org/2025.emnlp-main.880/

- **Link**: https://aclanthology.org/2025.emnlp-main.880/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in causal interpretability have extended from language models to vision-language models (VLMs), seeking to reveal their internal mechanisms through input interventions. While textual interventions often target semantics, visual interventions typically rely on coarse pixel-level perturbations, limiting semantic insights on multimodal integration. In this study, we introduce V-SEAM, a novel framework that combines **V**isual **S**emantic **E**diting and **A**ttention **M**odulating for causal interpretation of VLMs. V-SEAM enables concept-level visual manipulations and identifies attention heads with positive or negative contributions to predictions across three semantic levels: objects, attributes, and relationships. We observe that positive heads are often shared within the same semantic level but vary across levels, while negative heads tend to generalize broadly. Finally, we introduce an automatic method to modulate key head embeddings, demonstrating enhanced performance for both LLAVA and InstructBLIP across three diverse VQA benchmarks. Our data and code are released at: https://github.com/petergit1/V-SEAM.

</details>

---

## 201. Structured Preference Optimization for Vision-Language Long-Horizon Task Planning

- [ ] Structured Preference Optimization for Vision-Language Long-Horizon Task Planning | https://aclanthology.org/2025.emnlp-main.884/

- **Link**: https://aclanthology.org/2025.emnlp-main.884/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing vision-language planning methods perform well on short-horizon tasks but struggle with long-horizon reasoning in dynamic environments due to the difficulty of training models to generate high-quality reasoning processes. To address this, we propose Structured Preference Optimization (SPO), a framework that enhances reasoning and action selection for long-horizon task planning through structured evaluation and optimized training. SPO introduces: 1) Structured Preference Evaluation and Optimization, which evaluates reasoning chains across task relevance, historical consistency (as part of textual coherence), and image awareness (alignment with visual observations) to construct high-quality preference pairs; and 2) Curriculum-Guided Progressive Learning, enabling the model to adapt from simple to complex tasks, thereby improving generalization and robustness. To advance research in vision-language long-horizon task planning, we introduce ExtendaBench, a comprehensive benchmark covering 1,509 tasks across VirtualHome and Habitat 2.0, categorized into ultra-short, short, medium, and long tasks. Experimental results demonstrate that SPO significantly improves reasoning quality and final decision accuracy, outperforming prior methods on long-horizon tasks and underscoring the effectiveness of preference-driven optimization in vision-language task planning. Specifically, SPO achieves a +5.98% GCR and +4.68% SR improvement in VirtualHome and a +3.30% GCR and +2.11% SR improvement in Habitat over the best-performing baselines.

</details>

---

## 202. CLLMate: A Multimodal Benchmark for Weather and Climate Events Forecasting

- [ ] CLLMate: A Multimodal Benchmark for Weather and Climate Events Forecasting | https://aclanthology.org/2025.emnlp-main.886/

- **Link**: https://aclanthology.org/2025.emnlp-main.886/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Forecasting weather and climate events is crucial for making appropriate measures to mitigate environmental hazards and minimize losses. However, existing environmental forecasting research focuses narrowly on predicting numerical meteorological variables (e.g., temperature), neglecting the translation of these variables into actionable textual narratives of events and their consequences. To bridge this gap, we proposed Weather and Climate Event Forecasting (WCEF), a new task that leverages numerical meteorological raster data and textual event data to predict weather and climate events. This task is challenging to accomplish due to difficulties in aligning multimodal data and the lack of supervised datasets. To address these challenges, we present CLLMate, the first multimodal dataset for WCEF, using 26,156 environmental news articles aligned with ERA5 reanalysis data. We systematically benchmark 32 existing models on CLLMate, including closed-source, open-source, and our fine-tuned models. Our experiments reveal the advantages and limitations of existing MLLMs and the value of CLLMate for the training and benchmarking of the WCEF task. The dataset is available at https://github.com/hobolee/CLLMate.

</details>

---

## 203. MemeArena: Automating Context-Aware Unbiased Evaluation of Harmfulness Understanding for Multimodal Large Language Models

- [ ] MemeArena: Automating Context-Aware Unbiased Evaluation of Harmfulness Understanding for Multimodal Large Language Models | https://aclanthology.org/2025.emnlp-main.890/

- **Link**: https://aclanthology.org/2025.emnlp-main.890/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The proliferation of memes on social media necessitates the capabilities of multimodal Large Language Models (mLLMs) to effectively understand multimodal harmfulness. Existing evaluation approaches predominantly focus on mLLMs’ detection accuracy for binary classification tasks, which often fail to reflect the in-depth interpretive nuance of harmfulness across diverse contexts. In this paper, we propose MemeArena, an agent-based arena-style evaluation framework that provides a context-aware and unbiased assessment for mLLMs’ understanding of multimodal harmfulness. Specifically, MemeArena simulates diverse interpretive contexts to formulate evaluation tasks that elicit perspective-specific analyses from mLLMs. By integrating varied viewpoints and reaching consensus among evaluators, it enables fair and unbiased comparisons of mLLMs’ abilities to interpret multimodal harmfulness. Extensive experiments demonstrate that our framework effectively reduces the evaluation biases of judge agents, with judgment results closely aligning with human preferences, offering valuable insights into reliable and comprehensive mLLM evaluations in multimodal harmfulness understanding. Our code and data are publicly available at https://github.com/Lbotirx/MemeArena.

</details>

---

## 204. ViPE: Visual Perception in Parameter Space for Efficient Video-Language Understanding

- [ ] ViPE: Visual Perception in Parameter Space for Efficient Video-Language Understanding | https://aclanthology.org/2025.emnlp-main.897/

- **Link**: https://aclanthology.org/2025.emnlp-main.897/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing video-language models (Video-LLMs) typically rely on concatenating visual tokens with textual inputs for joint modeling. However, this token-level alignment leads to significant inefficiency, especially when scaling to long videos with dense visual inputs. In this work, we propose a video-to-parameter efficiency paradigm named ViPE that eliminates redundant visual tokens by transforming video content into visual perceptual weights, which are directly injected into the LLM’s parameters. ViPE consists of a visual injection module that compresses video features into a small set of perceptual queries using a hierarchical merge strategy, and a visual perception module that integrates the resulting representations into the LLM through a lightweight LoRA-like mechanism. ViPE achieves performance comparable to token-based baselines such as LLaVA, while reducing FLOPs by 85% and inference time by up to 65%, demonstrating a highly efficient and scalable solution for video understanding.

</details>

---

## 205. BANMIME: Misogyny Detection with Metaphor Explanation onBangla Memes

- [ ] BANMIME: Misogyny Detection with Metaphor Explanation onBangla Memes | https://aclanthology.org/2025.emnlp-main.900/

- **Link**: https://aclanthology.org/2025.emnlp-main.900/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Detecting misogyny in multimodal content remains a notable challenge, particularly in culturally conservative and low-resource contexts like Bangladesh. While existing research has explored hate speech and general meme classification, the nuanced identification of misogyny in Bangla memes, rich in metaphor, humor, and visual-textual interplay, remains severely underexplored. To address this gap, we introduce BanMiMe, the first comprehensive Bangla misogynistic meme dataset comprising 2,000 culturally grounded samples where each meme includes misogyny labels, humor categories, metaphor localization, and detailed human-written explanations. We benchmark the various performance of open and closed-source vision-language models (VLMs) under zero-shot and prompt-based settings and evaluate their capacity for both classification and explanation generation. Furthermore, we systematically explore multiple fine-tuning strategies, including standard, data-augmented, and Chain-of-Thought (CoT) supervision. Our results demonstrate that CoT-based fine-tuning consistently enhances model performance, both in terms of accuracy and in generating meaningful explanations. We envision BanMiMe as a foundational resource for advancing explainable multimodal moderation systems in low-resource and culturally sensitive settings.

</details>

---

## 206. Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time

- [ ] Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time | https://aclanthology.org/2025.emnlp-main.901/

- **Link**: https://aclanthology.org/2025.emnlp-main.901/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Multimodal Large Language Models (MLLMs) have gained significant attention across various domains. However, their widespread adoption has also raised serious safety concerns.In this paper, we uncover a new safety risk of MLLMs: the output preference of MLLMs can be arbitrarily manipulated by carefully optimized images. Such attacks often generate contextually relevant yet biased responses that are neither overtly harmful nor unethical, making them difficult to detect. Specifically, we introduce a novel method, **P**reference **Hi**jacking (**Phi**), for manipulating the MLLM response preferences using a preference hijacked image. Our method works at inference time and requires no model modifications. Additionally, we introduce a universal hijacking perturbation – a transferable component that can be embedded into different images to hijack MLLM responses toward any attacker-specified preferences. Experimental results across various tasks demonstrate the effectiveness of our approach. The code for Phi is accessible at https://github.com/Yifan-Lan/Phi.

</details>

---

## 207. Retrieval-augmentedGUIAgents with Generative Guidelines

- [ ] Retrieval-augmentedGUIAgents with Generative Guidelines | https://aclanthology.org/2025.emnlp-main.902/

- **Link**: https://aclanthology.org/2025.emnlp-main.902/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

GUI agents powered by vision-language models (VLMs) show promise in automating complex digital tasks. However, their effectiveness in real-world applications is often limited by scarce training data and the inherent complexity of these tasks, which frequently require long-tailed knowledge covering rare, unseen scenarios. We propose RAG-GUI , a lightweight VLM that leverages web tutorials at inferencetime. RAG-GUI is first warm-started via supervised finetuning (SFT) and further refined through self-guided rejection sampling fine-tuning (RSF). Designed to be model-agnostic, RAG-GUI functions as a generic plug-in that enhances any VLM-based agent. Evaluatedacross three distinct tasks, it consistently outperforms baseline agents and surpasses other inference baselines by 2.6% to 13.3% acrosstwo model sizes, demonstrating strong generalization and practical plug-and-play capabilities in real-world scenarios.

</details>

---

## 208. VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models

- [ ] VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models | https://aclanthology.org/2025.emnlp-main.908/

- **Link**: https://aclanthology.org/2025.emnlp-main.908/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This research investigates both explicit and implicit social biases exhibited by Vision-Language Models (VLMs). The key distinction between these bias types lies in the level of awareness: explicit bias refers to conscious, intentional biases, while implicit bias operates subconsciously. To analyze explicit bias, we directly pose questions to VLMs related to gender and racial differences: (1) Multiple-choice questions based on a given image (e.g., “What is the education level of the person in the image?”) (2) Yes-No comparisons using two images (e.g., “Is the person in the first image more educated than the person in the second image?”) For implicit bias, we design tasks where VLMs assist users but reveal biases through their responses: (1) Image description tasks: Models are asked to describe individuals in images, and we analyze disparities in textual cues across demographic groups. (2) Form completion tasks: Models draft a personal information collection form with 20 attributes, and we examine correlations among selected attributes for potential biases. We evaluate Gemini-1.5, GPT-4V, GPT-4o, LLaMA-3.2-Vision and LLaVA-v1.6. Our code and data are publicly available at https://github.com/uscnlp-lime/VisBias.

</details>

---

## 209. AISees YourLocation—But With A Bias Toward The Wealthy World

- [ ] AISees YourLocation—But With A Bias Toward The Wealthy World | https://aclanthology.org/2025.emnlp-main.910/

- **Link**: https://aclanthology.org/2025.emnlp-main.910/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual-Language Models (VLMs) have shown remarkable performance across various tasks, particularly in recognizing geographic information from images. However, VLMs still show regional biases in this task. To systematically evaluate these issues, we introduce a benchmark consisting of 1,200 images paired with detailed geographic metadata. Evaluating four VLMs, we find that while these models demonstrate the ability to recognize geographic information from images, achieving up to 53.8% accuracy in city prediction, they exhibit significant biases. Specifically, performance is substantially higher for economically developed and densely populated regions compared to less developed (-12.5%) and sparsely populated (-17.0%) areas. Moreover, regional biases of frequently over-predicting certain locations remain. For instance, they consistently predict Sydney for images taken in Australia, shown by the low entropy scores for these countries. The strong performance of VLMs also raises privacy concerns, particularly for users who share images online without the intent of being identified. Our code and dataset are publicly available at https://github.com/uscnlp-lime/FairLocator.

</details>

---

## 210. Iterative Prompt Refinement for Safer Text-to-Image Generation

- [ ] Iterative Prompt Refinement for Safer Text-to-Image Generation | https://aclanthology.org/2025.emnlp-main.913/

- **Link**: https://aclanthology.org/2025.emnlp-main.913/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-Image (T2I) models have made remarkable progress in generating images from text prompts, but their output quality and safety still depend heavily on how prompts are phrased. Existing safety methods typically refine prompts using large language models (LLMs), but they overlook the images produced, which can result in unsafe outputs or unnecessary changes to already safe prompts. To address this, we propose an iterative prompt refinement algorithm that uses Vision Language Models (VLMs) to analyze both the input prompts and the generated images. By leveraging visual feedback, our method refines prompts more effectively, improving safety while maintaining user intent and reliability comparable to existing LLM-based approaches. Additionally, we introduce a new dataset labeled with both textual and visual safety signals using off-the-shelf multi-modal LLM, enabling supervised fine-tuning. Experimental results demonstrate that our approach produces safer outputs without compromising alignment with user intent, offering a practical solution for generating safer T2I content.\textcolor{red}{WARNING: This paper contains examples of harmful or inappropriate images generated by models.}

</details>

---

## 211. Exploring Response Uncertainty inMLLMs: An Empirical Evaluation under Misleading Scenarios

- [ ] Exploring Response Uncertainty inMLLMs: An Empirical Evaluation under Misleading Scenarios | https://aclanthology.org/2025.emnlp-main.916/

- **Link**: https://aclanthology.org/2025.emnlp-main.916/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have recently achieved state-of-the-art performance on tasks ranging from visual question answering to video understanding. However, existing studies have concentrated mainly on visual–textual misalignment, leaving largely unexplored the MLLMs’ ability to preserve an originally correct answer when confronted with misleading information. We reveal a response uncertainty phenomenon: across nine standard datasets, twelve state-of-the-art open-source MLLMs overturn a previously correct answer in 65% of cases after receiving a single deceptive cue. To systematically quantify this vulnerability, we propose a two-stage evaluation pipeline: (1) elicit each model’s original response on unperturbed inputs; (2) injectexplicit(false-answer hints) andimplicit(contextual contradictions) misleading instructions, and compute themisleading rate—the fraction of correct-to-incorrect flips. Leveraging the most susceptible examples, we curate the Multimodal Uncertainty Benchmark (MUB), a collection of image–question pairs stratified into low, medium, and high difficulty based on how many of twelve state-of-the-art MLLMs they mislead. Extensive evaluation on twelve open-source and five closed-source models reveals a high uncertainty: average misleading rates exceed 86%, with explicit cues over 67.19% and implicit cues over 80.67%. To reduce the misleading rate, we then fine-tune all open-source MLLMs on a compact 2,000-sample mixed-instruction dataset, reducing misleading rates to 6.97% (explicit) and 32.77% (implicit), boosting consistency by nearly 29.37% on highly deceptive inputs, and slightly improving accuracy on standard benchmarks.

</details>

---

## 212. UI-Hawk: Unleashing the Screen Stream Understanding for MobileGUIAgents

- [ ] UI-Hawk: Unleashing the Screen Stream Understanding for MobileGUIAgents | https://aclanthology.org/2025.emnlp-main.920/

- **Link**: https://aclanthology.org/2025.emnlp-main.920/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interface (GUI) agents are expected to precisely operate on the screens of digital devices. Existing GUI agents merely depend on current visual observations and plain-text action history, ignoring the significance of history screens. To mitigate this issue, we propose **UI-Hawk**, a multi-modal GUI agent specially designed to process screen streams encountered during GUI navigation. UI-Hawk incorporates a history-aware visual encoder to handle the screen sequences. To acquire a better understanding of screen streams, we select four fundamental tasks—UI grounding, UI referring, screen question answering, and screen summarization. We further propose a curriculum learning strategy to subsequently guide the model from fundamental tasks to advanced screen-stream comprehension.Along with the efforts above, we have also created a benchmark FunUI to quantitatively evaluate the fundamental screen understanding ability of MLLMs. Extensive experiments on FunUI and GUI navigation benchmarks consistently validate that screen stream understanding is essential for GUI tasks.Our code and data are now available at https://github.com/IMNearth/UIHawk.

</details>

---

## 213. Med-VRAgent: A Framework for Medical Visual Reasoning-Enhanced Agents

- [ ] Med-VRAgent: A Framework for Medical Visual Reasoning-Enhanced Agents | https://aclanthology.org/2025.emnlp-main.939/

- **Link**: https://aclanthology.org/2025.emnlp-main.939/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) achieve promising results in medical reasoning but struggle with hallucinations, vague descriptions, Inconsistent logic and poor localization. To address this, we propose a agent framework named Medical Visual Reasoning Agent (Med-VRAgent). The approach is based on Visual Guidance and Self-Reward paradigms and Monte Carlo Tree Search (MCTS). By combining the Visual Guidance with tree search, Med-VRAgent improves the medical visual reasoning capabilities of VLMs. We use the trajectories collected by Med-RAgent as feedback to further improve the performance by fine-tuning the VLMs with the proximal policy optimization (PPO) objective. Experiments on multiple medical VQA benchmarks demonstrate that our method outperforms existing approaches.

</details>

---

## 214. PunMemeCN: A Benchmark to Explore Vision-Language Models’ Understanding ofChinese Pun Memes

- [ ] PunMemeCN: A Benchmark to Explore Vision-Language Models’ Understanding ofChinese Pun Memes | https://aclanthology.org/2025.emnlp-main.944/

- **Link**: https://aclanthology.org/2025.emnlp-main.944/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pun memes, which combine wordplay with visual elements, represent a popular form of humor in Chinese online communications. Despite their prevalence, current Vision-Language Models (VLMs) lack systematic evaluation in understanding and applying these culturally-specific multimodal expressions. In this paper, we introduce PunMemeCN, a novel benchmark designed to assess VLMs’ capabilities in processing Chinese pun memes across three progressive tasks: pun meme detection, sentiment analysis, and chat-driven meme response. PunMemeCN consists of 1,959 Chinese memes (653 pun memes and 1,306 non-pun memes) with comprehensive annotations of punchlines, sentiments, and explanations, alongside 2,008 multi-turn chat conversations incorporating these memes. Our experiments indicate that state-of-the-art VLMs struggle with Chinese pun memes, particularly with homophone wordplay, even with Chain-of-Thought prompting. Notably, punchlines in memes can effectively conceal potentially harmful content from AI detection. These findings underscore the challenges in cross-cultural multimodal understanding and highlight the need for culture-specific approaches to humor comprehension in AI systems.

</details>

---

## 215. METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding

- [ ] METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding | https://aclanthology.org/2025.emnlp-main.954/

- **Link**: https://aclanthology.org/2025.emnlp-main.954/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Video Large Language Models (VLLMs) have significantly enhanced their ability to understand video content. Nonetheless, processing long videos remains challenging due to high computational demands and the redundancy present in the visual data. In this work, we proposeMETok, a training-free,Multi-stageEvent-basedToken compression framework designed to accelerate VLLMs’ inference while preserving accuracy. METok progressively eliminates redundant visual tokens across three critical stages: (1) event-aware compression during vision encoding, (2) hierarchical token pruning in the prefilling stage based on semantic alignment and event importance, and (3) a decoding-stage KV Cache optimization that further reduces memory consumption. Our experiments on diverse video benchmarks demonstrate that METok achieves an optimal trade-off between efficiency and accuracy by dynamically selecting informative visual tokens. For instance, equipping LongVA-7B with METok realizes an 80.6% FLOPs reduction and 93.5% KV Cache memory savings, all while maintaining comparable or even superior accuracy.

</details>

---

## 216. Small Models, Big Results: Achieving Superior Intent Extraction through Decomposition

- [ ] Small Models, Big Results: Achieving Superior Intent Extraction through Decomposition | https://aclanthology.org/2025.emnlp-main.949/

- **Link**: https://aclanthology.org/2025.emnlp-main.949/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding user intents from UI interaction trajectories remains a challenging, yet crucial, frontier in intelligent agent development. While massive, datacenter-based, multi-modal large language models (MLLMs) possess greater capacity to handle the complexities of such sequences, smaller models which can run on-device to provide a privacy-preserving, low-cost, and low-latency user experience, struggle with accurate intent inference. We address these limitations by introducing a novel decomposed approach: first, we perform structured interaction summarization, capturing key information from each user action. Second, we perform intent extraction using a fine-tuned model operating on the aggregated summaries. This method improves intent understanding in resource-constrained models, even surpassing the base performance of large MLLMs.

</details>

---

## 217. SheetDesigner:MLLM-Powered Spreadsheet Layout Generation with Rule-Based and Vision-Based Reflection

- [ ] SheetDesigner:MLLM-Powered Spreadsheet Layout Generation with Rule-Based and Vision-Based Reflection | https://aclanthology.org/2025.emnlp-main.957/

- **Link**: https://aclanthology.org/2025.emnlp-main.957/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spreadsheets are critical to data-centric tasks, with rich, structured layouts that enable efficient information transmission. Given the time and expertise required for manual spreadsheet layout design, there is an urgent need for automated solutions.However, existing automated layout models are ill-suited to spreadsheets, as they often (1) treat components as axis-aligned rectangles with continuous coordinates, overlooking the inherently discrete, grid-based structure of spreadsheets; and (2) neglect interrelated semantics, such as data dependencies and contextual links, unique to spreadsheets. In this paper, we first formalize the spreadsheet layout generation task, supported by a seven-criterion evaluation protocol and a dataset of 3,326 spreadsheets. We then introduceSheetDesigner, a zero-shot and training-free framework using Multimodal Large Language Models (MLLMs) that combines rule and vision reflection for component placement and content population. SheetDesigner outperforms five baselines by at least 22.6%. We further find that through vision modality, MLLMs handle overlap and balance well but struggle with alignment, necessitates hybrid rule and visual reflection strategies. Our codes and data is available at Github.

</details>

---

## 218. RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language

- [ ] RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language | https://aclanthology.org/2025.emnlp-main.96/

- **Link**: https://aclanthology.org/2025.emnlp-main.96/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal question answering (QA) often requires identifying which video, audio, or sensor tokens are relevant to the question. Yet modality disagreements are common: off-camera speech, background noise, or motion outside the field of view often mislead fusion models that weight all streams equally. We present RAVEN, a unified QA architecture whose core is QuART, a query-conditioned cross-modal gating module that assigns scalar relevance scores to each token across modalities, enabling the model to amplify informative signals and suppress distractors before fusion. RAVEN is trained through a three-stage pipeline comprising unimodal pretraining, query-aligned fusion, and disagreement-oriented fine-tuning - each stage targeting a distinct challenge in multi-modal reasoning: representation quality, cross-modal relevance, and robustness to modality mismatch. To support training and evaluation, we release AVS-QA, a dataset of 300K synchronized Audio-Video-Sensor streams paired with automatically generated question-answer pairs. Experimental results on seven multi-modal QA benchmarks - including egocentric and exocentric tasks - show that RAVEN achieves up to 14.5% and 8.0% gains in accuracy compared to state-of-the-art multi-modal large language models, respectively. Incorporating sensor data provides an additional 16.4% boost, and the model remains robust under modality corruption, outperforming SOTA baselines by 50.23%. Our code and dataset are available at https://github.com/BASHLab/RAVEN.

</details>

---

## 219. VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient MultimodalLLMs

- [ ] VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient MultimodalLLMs | https://aclanthology.org/2025.emnlp-main.955/

- **Link**: https://aclanthology.org/2025.emnlp-main.955/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have achieved strong performance across vision-language tasks, but suffer from significant computational overhead due to the quadratic growth of attention computations with the number of multimodal tokens. Though efforts have been made to prune tokens in MLLMs, *they lack a fundamental understanding of how MLLMs process and fuse multimodal information*. Through systematic analysis, we uncover a three-stage cross-modal interaction process: (1) Shallow layers recognize task intent, with visual tokens acting as passive attention sinks; (2) Cross-modal fusion occurs abruptly in middle layers, driven by a few critical visual tokens; (3) Deep layers discard vision tokens, focusing solely on linguistic refinement. Based on these findings, we propose *VisiPruner*, a training-free pruning framework that reduces **99.9%** of vision-related attention computations and **62.8%** of FLOPs while maintaining performance. It significantly outperforms existing token pruning methods and generalizes across diverse MLLMs. Beyond pruning, our insights further provide actionable guidelines for training efficient MLLMs by aligning model architecture with its intrinsic layer-wise processing dynamics.

</details>

---

## 220. Cache-of-Thought: Master-Apprentice Framework for Cost-Effective Vision Language Model Reasoning

- [ ] Cache-of-Thought: Master-Apprentice Framework for Cost-Effective Vision Language Model Reasoning | https://aclanthology.org/2025.emnlp-main.97/

- **Link**: https://aclanthology.org/2025.emnlp-main.97/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have achieved remarkable success in a wide range of vision applications of increasing complexity and scales, yet choosing the right VLM model size involves a trade-off between response quality and cost. While smaller VLMs are cheaper to run, they typically produce responses only marginally better than random guessing on benchmarks such as MMMU. In this paper, we proposeCache of Thought (CoT), a master–apprentice framework for collaborative inference between large and small VLMs. CoT manages high-quality query results from large VLMs (master) in a cache, which are then selected via a novel multi-modal retrieval and in-context learning to aid the performance of small VLMs (apprentice). We extensively evaluate CoT on various widely-recognized and challenging general reasoning benchmarks, and show that CoT increases overall reasoning performance by up to 7.7% under the same budget, and specifically boosts the reasoning performance of apprentice VLMs by up to 36.6%. Our code is available athttps://github.com/UIUC-MONET/Cache-of-Thoughts.

</details>

---

## 221. Shared Path: Unraveling Memorization in MultilingualLLMs through Language Similarities

- [ ] Shared Path: Unraveling Memorization in MultilingualLLMs through Language Similarities | https://aclanthology.org/2025.emnlp-main.978/

- **Link**: https://aclanthology.org/2025.emnlp-main.978/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present the first comprehensive study of Memorization in Multilingual Large Language Models (MLLMs), analyzing 95 languages using models across diverse model scales, architectures, and memorization definitions. As MLLMs are increasingly deployed, understanding their memorization behavior has become critical. Yet prior work has focused primarily on monolingual models, leaving multilingual memorization underexplored, despite the inherently long-tailed nature of training corpora. We find that the prevailing assumption, that memorization is highly correlated with training data availability, fails to fully explain memorization patterns in MLLMs. We hypothesize that treating languages in isolation — ignoring their similarities — obscures the true patterns of memorization. To address this, we propose a novel graph-based correlation metric that incorporates language similarity to analyze cross-lingual memorization. Our analysis reveals that among similar languages, those with fewer training tokens tend to exhibit higher memorization, a trend that only emerges when cross-lingual relationships are explicitly modeled. These findings underscore the importance of a language-aware perspective in evaluating and mitigating memorization vulnerabilities in MLLMs. This also constitutes empirical evidence that language similarity both explains Memorization in MLLMs and underpins Cross-lingual Transferability, with broad implications for multilingual NLP.

</details>

---

## 222. Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization

- [ ] Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization | https://aclanthology.org/2025.emnlp-main.982/

- **Link**: https://aclanthology.org/2025.emnlp-main.982/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Visual Language Models (LVLMs) have demonstrated impressive capabilities across multiple tasks. However, their trustworthiness is often challenged by hallucinations, which can be attributed to the modality misalignment and the inherent hallucinations of their underlying Large Language Models (LLMs) backbone. Existing preference alignment methods focus on aligning model responses with human preferences while neglecting image-text modality alignment, resulting in over-reliance on LLMs and hallucinations. In this paper, we propose Entity-centric Multimodal Preference Optimization (EMPO), which achieves enhanced modality alignment than existing human preference alignment methods. Besides, to overcome the scarcity of high-quality multimodal preference data, we utilize open-source instruction datasets to automatically construct high-quality preference data across three aspects: image, instruction, and response. Experiments on two human preference datasets and five multimodal hallucination benchmarks demonstrate the effectiveness of EMPO, e.g., reducing hallucination rates by 80.4% on Object HalBench and 52.6% on MM HalBench, thereby enhancing the trustworthiness of LVLMs. The code and dataset will be made publicly available.

</details>

---

## 223. InfiniBench: A Benchmark for Large Multi-Modal Models in Long-Form Movies andTVShows

- [ ] InfiniBench: A Benchmark for Large Multi-Modal Models in Long-Form Movies andTVShows | https://aclanthology.org/2025.emnlp-main.984/

- **Link**: https://aclanthology.org/2025.emnlp-main.984/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding long-form videos, such as movies and TV episodes ranging from tens of minutes to two hours, remains a significant challenge for multi-modal models. Existing benchmarks often fail to test the full range of cognitive skills needed to process these temporally rich and narratively complex inputs. Therefore, we introduce InfiniBench, a comprehensive benchmark designed to evaluate the capabilities of models in long video understanding rigorously.InfiniBench offers:(1) Over 1,000 hours of video content, with an average video length of 53 minutes.(2) The largest set of question-answer pairs for long video comprehension, totaling around 87.7 K.(3) Eight diverse skills that span both grounding-based (e.g., scene transitions, character actions) and reasoning-based (e.g., deep context understanding, multi-event linking).(4) Rich annotation formats, including both multiple-choice and open-ended questions.We conducted an in-depth evaluation across both commercial (GPT-4o, Gemini 2.0 Flash) and most recent open-source vision-language models, such as Qwen2.5-VL, InternVL3.0). Results reveal that:(1) Models struggle across the board: Even the best model, GPT-4o, achieves only 47.1% on grounding-based skills, with most models performing near or just above random chance.(2) Strong reliance on world knowledge: Models achieve surprisingly high scores using only metadata (e.g., video titles), highlighting a tendency to rely on pre-trained knowledge rather than actual visual or temporal understanding.(3) Multi-Modal Importance: When provided with full video and subtitle context, however, models show substantial improvements, confirming the critical role of multimodal input in video understanding.Our findings underscore the inherent challenges in long-video comprehension and point to the need for substantial advancements in both grounding and reasoning capabilities in MLLMs.

</details>

---

## 224. Looking Beyond Text: Reducing Language Bias in Large Vision-Language Models via Multimodal Dual-Attention and Soft-Image Guidance

- [ ] Looking Beyond Text: Reducing Language Bias in Large Vision-Language Models via Multimodal Dual-Attention and Soft-Image Guidance | https://aclanthology.org/2025.emnlp-main.995/

- **Link**: https://aclanthology.org/2025.emnlp-main.995/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have achieved impressive results in vision-language tasks. However, Therefore, we propose LACING, designed to address such bias with Mu̲Ltimodal Du̲Al-attention Me̲Chan̲Ism (MDA) a̲Nd Soft-Image̲Guidance (SIG). Specifically, MDA adopts aparallel dual-attention mechanismthat constructs separate attention for visual and text inputs to enhance integration of visual inputs across model. SIG uses alearnable soft visual promptduring training and inference to replace visual inputs, designed to compel LVLMs to prioritize text inputs during inference. Experiments across different model architectures and scales demonstrate that LACING effectively debiases LVLMs from their language bias, enhancing visual comprehension and reducing hallucinations without additional resources.

</details>

---

## 225. Treble CounterfactualVLMs: A Causal Approach to Hallucination

- [ ] Treble CounterfactualVLMs: A Causal Approach to Hallucination | https://aclanthology.org/2025.findings-emnlp.1000/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1000/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) excel at tasks such as image captioning and visual question answering but frequently produce hallucinated outputs that deviate from the actual visual input or prompt. While prior work links hallucination to biases in data or representation, their causal origins remain unclear. We propose a causal framework to analyze and mitigate hallucination in VLMs. Our key hypothesis is that hallucinations arise from unintended direct influences of the vision or text modality that bypass the intended multi-modal fusion. To examine this, we construct a causal graph of the VLM and use counterfactual analysis to estimate the Natural Direct Effect (NDE) of each modality and their interaction. By systematically identifying and suppressing these direct effects, we encourage outputs that are more faithfully grounded in true cross-modal reasoning. Our approach consists of three steps: (1) designing structural causal graphs to distinguish correct fusion pathways from spurious modality shortcuts, (2) estimating modality-specific and cross-modal NDE using perturbed image representations, hallucinated text embeddings, and degraded visual inputs, and (3) implementing a test-time intervention module to dynamically adjust the model’s dependence on each modality. Experimental results demonstrate that our method significantly reduces hallucination while preserving task performance, providing a robust and interpretable framework for improving VLM reliability.

</details>

---

## 226. MaskCD: MitigatingLVLMHallucinations by Image Head Masked Contrastive Decoding

- [ ] MaskCD: MitigatingLVLMHallucinations by Image Head Masked Contrastive Decoding | https://aclanthology.org/2025.findings-emnlp.1025/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1025/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have shown remarkable performance in visual-language understanding for downstream multimodal tasks. While their capabilities are improving, problems emerge simultaneously. Among those problems, the hallucinations have attracted much attention, which stands for the phenomenon where LVLMs generate contradictory content to their input visual and text contents. Many approaches have been proposed to deal with this issue, such as contrastive decoding and attention manipulation. However, contrastive decoding methods struggle in constructing appropriate contrastive samples, and attention manipulation methods are highly sensitive, lacking stability. In this work, we propose image head Masked Contrastive Decoding (MaskCD). Our approach utilizes the “image heads” in LVLMs, masking them to construct contrastive samples for contrastive decoding. We evaluated MaskCD on LLaVA-1.5-7b and Qwen-VL-7b, using various benchmarks such as CHAIR, POPE, AMBER and MME. The results demonstrate that MaskCD effectively alleviates the phenomenon of hallucinations and retains the general capabilities of LVLMs. Corresponding resources could be found at: https://github.com/Deng-Jingyuan/MaskCD.

</details>

---

## 227. A Structured Framework for Evaluating and Enhancing Interpretive Capabilities of MultimodalLLMs in Culturally Situated Tasks

- [ ] A Structured Framework for Evaluating and Enhancing Interpretive Capabilities of MultimodalLLMs in Culturally Situated Tasks | https://aclanthology.org/2025.findings-emnlp.103/

- **Link**: https://aclanthology.org/2025.findings-emnlp.103/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This study aims to test and evaluate the capabilities and characteristics of current mainstream Visual Language Models (VLMs) in generating critiques for traditional Chinese painting. To achieve this, we first developed a quantitative framework for Chinese painting critique. This framework was constructed by extracting multi-dimensional evaluative features covering evaluative stance, feature focus, and commentary quality from human expert critiques using a zero-shot classification model. Based on these features, several representative critic personas were defined and quantified. This framework was then employed to evaluate selected VLMs such as Llama, Qwen, or Gemini. The experimental design involved persona-guided prompting to assess the VLM’s ability to generate critiques from diverse perspectives. Our findings reveal the current performance levels, strengths, and areas for improvement of VLMs in the domain of art critique, offering insights into their potential and limitations in complex semantic understanding and content generation tasks.

</details>

---

## 228. Tales of Morality: Comparing Human- andLLM-Generated Moral Stories from Visual Cues

- [ ] Tales of Morality: Comparing Human- andLLM-Generated Moral Stories from Visual Cues | https://aclanthology.org/2025.findings-emnlp.1029/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1029/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Do moral values align between images, the stories humans write about them, and the narratives generated by large language models (LLMs)? This question matters because stories are central to how humans communicate moral values, yet little is known about how people and LLMs perform this task in a multimodal (text and image) setting. We present a systematic comparison of moral values represented in human- and LLM-generated narratives based on images annotated by humans for moral content. Our analysis shows that while human stories reflect a balanced distribution of moral foundations and coherent narrative arcs, LLMs disproportionately emphasize the Care foundation and often lack emotional resolution. Even with moral conditioning, these biases persist in LLMs. We introduce a novel dataset and framework for evaluating moral storytelling in vision-language models, highlighting key challenges in aligning AI with human moral reasoning across cultures.

</details>

---

## 229. RealBench: AChinese Multi-image Understanding Benchmark Close to Real-world Scenarios

- [ ] RealBench: AChinese Multi-image Understanding Benchmark Close to Real-world Scenarios | https://aclanthology.org/2025.findings-emnlp.1039/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1039/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While various multimodal multi-image evaluation datasets have been emerged, but these datasets are primarily based on English, and there has yet to be a Chinese multi-image dataset. To fill this gap, we introduce RealBench, the first Chinese multimodal multi-image dataset, which contains 9393 samples and 69910 images. RealBench distinguishes itself by incorporating real user-generated content, ensuring high relevance to real-world applications. Additionally, the dataset covers a wide variety of scenes, image resolutions, and image structures, further increasing the difficulty of multi-image understanding. Ultimately, we conduct a comprehensive evaluation of RealBench using 21 multimodal LLMs of different sizes, including closed-source models that support multi-image inputs as well as open-source visual and video models. The experimental results indicate that even the most powerful closed-source models still face challenges when handling multi-image Chinese scenarios. Moreover, there remains a noticeable performance gap of around 71.8% on average between open-source visual/video models and closed-source models. These results show that RealBench provides an important research foundation for further exploring multi-image understanding capabilities in the Chinese context. Our datasets will be publicly available.

</details>

---

## 230. Train a Unified Multimodal Data Quality Classifier with Synthetic Data

- [ ] Train a Unified Multimodal Data Quality Classifier with Synthetic Data | https://aclanthology.org/2025.findings-emnlp.104/

- **Link**: https://aclanthology.org/2025.findings-emnlp.104/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Multimodal Large Language Models (MLLMs) are continually pre-trained on a mixture of image-text caption data and interleaved document data, while the high-quality data filtering towards image-text interleaved document data is under-explored. We propose to train an efficient MLLM as a Unified Mulitmodal Data Quality Classifier to Filter both high-quality image-text caption and interleaved data (UniFilter). To address the challenge of collecting diverse labeled multimodal data, we introduce a semi-synthetic approach that leverages readily available raw images and generates corresponding text across four quality levels. This method enables efficient creation of sample-score pairs for both caption and interleaved document data to train UniFilter. We apply UniFilter to curate high-quality caption data from DataComp caption dataset and interleaved data from the OBELICS image-text interleaved dataset. MLLMs pre-trained on the filtered data demonstrate significantly enhanced capabilities compared to those trained on baseline-filtered data, achieving stronger zero-shot reasoning and in-context learning capabilities. After visual supervised fine-tuning, these UniFilter-induced MLLMs achieve stronger performance on various benchmarks, highlighting the downstream benefits of high-quality multimodal pre-training.

</details>

---

## 231. Knowing More, Acting Better: Hierarchical Representation for Embodied Decision-Making

- [ ] Knowing More, Acting Better: Hierarchical Representation for Embodied Decision-Making | https://aclanthology.org/2025.findings-emnlp.1042/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1042/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Modern embodied AI uses multimodal large language models (MLLMs) as policy models, predicting actions from final-layer hidden states. This widely adopted approach, however, assumes that monolithic last-layer representations suffice for decision-making—a structural simplification at odds with decades of cognitive science, which highlights the importance of distributed, hierarchical processing for perception and action. Addressing this foundational asymmetry, we introduce a hierarchical action probing method that explicitly aggregates representations from all layers, mirroring the brain’s multi-level organization. Experiments reveal that early layers facilitate spatial grounding, middle layers support contextual integration, and later layers enable abstract generalization—which shows MLLMs inherently encode distributed action-relevant structures. These layer-wise features are integrated by a lightweight probe for spatial reasoning and contextual understanding, without costly backbone fine-tuning. This hierarchical solution shows significant improvements over standard last-layer embodied models in physical simulators, achieving a 46.6% success rate and a 62.5% gain in spatial reasoning tasks. These findings challenge conventional assumptions in embodied AI, establishing hierarchical probing as a principled alternative grounded in both cognitive theory and empirical evidence.

</details>

---

## 232. Self-Improvement in Multimodal Large Language Models: A Survey

- [ ] Self-Improvement in Multimodal Large Language Models: A Survey | https://aclanthology.org/2025.findings-emnlp.105/

- **Link**: https://aclanthology.org/2025.findings-emnlp.105/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in self-improvement for Large Language Models (LLMs) have efficiently enhanced model capabilities without significantly increasing costs, particularly in terms of human effort. While this area is still relatively young, its extension to the multimodal domain holds immense potential for leveraging diverse data sources and developing more general self-improving models. This survey is the first to provide a comprehensive overview of self-improvement in Multimodal LLMs (MLLMs). We provide a structured overview of the current literature and discuss methods from three perspectives: 1) data collection, 2) data organization, and 3) model optimization, to facilitate the further development of self-improvement in MLLMs. We also include commonly used evaluations and downstream applications. Finally, we conclude by outlining open challenges and future research directions.

</details>

---

## 233. OVFact: Measuring and Improving Open-Vocabulary Factuality for Long Caption Models

- [ ] OVFact: Measuring and Improving Open-Vocabulary Factuality for Long Caption Models | https://aclanthology.org/2025.findings-emnlp.1058/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1058/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) often struggle to generate long and factual captions. However, traditional measures for hallucination and factuality are not well suited for evaluating longer, more diverse captions and in settings where ground-truth human-annotated captions are unavailable. We introduce OVFact, a novel method for measuring caption factuality of long captions that leverages open-vocabulary visual grounding and tool-based verification without depending on human annotations. Our method improves agreement with human judgements and captures both caption descriptiveness (recall) and factual precision in the same metric. Furthermore, unlike previous metrics, our reference-free method design enables new applications towards factuality-based data filtering. We observe models trained on an OVFact-filtered (2.5-5x less) subset of a large-scale, noisy (VLM-generated) pretraining set meaningfully improve factuality precision without sacrificing caption descriptiveness across a range of downstream long caption benchmarks.

</details>

---

## 234. DoLVLMs Know What They Know? A Systematic Study of Knowledge Boundary Perception inLVLMs

- [ ] DoLVLMs Know What They Know? A Systematic Study of Knowledge Boundary Perception inLVLMs | https://aclanthology.org/2025.findings-emnlp.1081/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1081/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) demonstrate strong visual question answering (VQA) capabilities but are shown to hallucinate. A reliable model should perceive its knowledge boundaries—knowing what it knows and what it does not. This paper investigates LVLMs’ perception of their knowledge boundaries by evaluating three types of confidence signals: probabilistic confidence, answer consistency-based confidence, and verbalized confidence. Experiments on three LVLMs across three VQA datasets show that, although LVLMs possess a reasonable perception level, there is substantial room for improvement. Among the three confidence, probabilistic and consistency-based signals are more reliable indicators, while verbalized confidence often leads to overconfidence. To enhance LVLMs’ perception, we adapt several established confidence calibration methods from Large Language Models (LLMs) and propose three effective methods. Additionally, we compare LVLMs with their LLM counterparts, finding that jointly processing visual and textual inputs decreases question-answering performance but reduces confidence, resulting in improved perception level compared to LLMs.

</details>

---

## 235. Token Preference Optimization with Self-Calibrated Visual-Anchored Rewards for Hallucination Mitigation

- [ ] Token Preference Optimization with Self-Calibrated Visual-Anchored Rewards for Hallucination Mitigation | https://aclanthology.org/2025.findings-emnlp.1076/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1076/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Direct Preference Optimization (DPO) has been demonstrated to be highly effective in mitigating hallucinations in Large Vision Language Models (LVLMs) by aligning their outputs more closely with human preferences. Despite the recent progress, existing methods suffer from two drawbacks: 1) Lack of scalable token-level rewards; and 2) Neglect of visual-anchored tokens. To this end, we propose a novel Token Preference Optimization model with self-calibrated rewards (dubbed as TPO), which adaptively attends to visual correlated tokens without fine-grained annotations. Specifically, we introduce a token-level visual-anchored reward as the difference of the logistic distributions of generated tokens conditioned on the raw image and the corrupted one. In addition, to highlight the informative visual-anchored tokens, a visual-aware training objective is proposed to enhance more accurate token-level optimization. Extensive experimental results have manifested the state-of-the-art performance of the proposed TPO. For example, by building on top of LLaVA and Qwen, our TPO boosts the performance absolute improvement for hallucination benchmarks.

</details>

---

## 236. MTabVQA: Evaluating Multi-Tabular Reasoning of Language Models in Visual Space

- [ ] MTabVQA: Evaluating Multi-Tabular Reasoning of Language Models in Visual Space | https://aclanthology.org/2025.findings-emnlp.1083/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1083/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have demonstrated remarkable capabilities in interpreting visual layouts and text. However, a significant challenge remains in their ability to interpret robustly and reason over multi-tabular data presented as images, a common occurrence in real-world scenarios like web pages and digital documents. Existing benchmarks typically address single tables or non-visual data (text/structured). This leaves a critical gap: they don’t assess the ability to parse diverse table images, correlate information across them, and perform multi-hop reasoning on the combined visual data. To bridge this evaluation gap, we introduce MTabVQA, a novel benchmark specifically designed for multi-tabular visual question answering. MTabVQA comprises 3,745 complex question-answer pairs that necessitate multi-hop reasoning across several visually rendered table images. We provide extensive benchmark results for state-of-the-art VLMs on MTabVQA, revealing significant performance limitations. We further investigate post-training techniques to enhance these reasoning abilities and release MTabVQA-Instruct, a large-scale instruction-tuning dataset. Our experiments show that fine-tuning VLMs with MTabVQA-Instruct substantially improves their performance on visual multi-tabular reasoning. Code and dataset are available online: .

</details>

---

## 237. Intelligent Document Parsing: Towards End-to-end Document Parsing via Decoupled Content Parsing and Layout Grounding

- [ ] Intelligent Document Parsing: Towards End-to-end Document Parsing via Decoupled Content Parsing and Layout Grounding | https://aclanthology.org/2025.findings-emnlp.1088/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1088/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the daily work, vast amounts of documents are stored in pixel-based formats such as images and scanned PDFs, posing challenges for efficient database management and data processing. Existing methods often fragment the parsing process into the pipeline of separated subtasks on the layout element level, resulting in incomplete semantics and error propagation. Even though models based on multi-modal large language models (MLLMs) mitigate the issues to some extent, they also suffer from absent or sub-optimal grounding ability for visual information. To address these challenges, we introduce the Intelligent Document Parsing (IDP) framework, an end-to-end document parsing framework leveraging the vision-language priors of MLLMs, equipped with an elaborately designed document representation and decoding mechanism to decouple the content parsing and layout grounding to fully activate the potential of MLLMs for document parsing. Experimental results demonstrate that the IDP method surpasses existing methods, significantly advancing MLLM-based document parsing.

</details>

---

## 238. All-in-one: Understanding and Generation in Multimodal Reasoning with theMAIABenchmark

- [ ] All-in-one: Understanding and Generation in Multimodal Reasoning with theMAIABenchmark | https://aclanthology.org/2025.findings-emnlp.1091/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1091/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce MAIA (Multimodal AI Assessment), a native-Italian benchmark designed for fine-grained investigation of the reasoning abilities of visual language models on videos. MAIA differs from other available video benchmarks for its design, its reasoning categories, the metric it uses, and the language and culture of the videos. MAIA evaluates Vision Language Models (VLMs) on two aligned tasks: a visual statement verification task, and an open-ended visual question-answering task, both on the same set of video-related questions. It considers twelve reasoning categories that aim to disentangle language and vision relations by highlighting the role of the visual input. Thanks to its carefully taught design, it evaluates VLMs’ consistency and visually grounded natural language comprehension and generation simultaneously through an aggregated metric revealing low results that highlight models’ fragility. Last but not least, the video collection has been carefully selected to reflect the Italian culture, and the language data are produced by native-speakers.Data available at *[GitHub](https://github.com/Caput97/MAIA-Multimodal_AI_Assessment.git).*

</details>

---

## 239. Attack as Defense: Safeguarding Large Vision-Language Models from Jailbreaking by Adversarial Attacks

- [ ] Attack as Defense: Safeguarding Large Vision-Language Models from Jailbreaking by Adversarial Attacks | https://aclanthology.org/2025.findings-emnlp.1095/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1095/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adversarial vulnerabilities in vision-language models pose a critical challenge to the reliability of large language systems, where typographic manipulations and adversarial perturbations can effectively bypass language model defenses. We introduce Attack as Defense (AsD), the first approach to proactively defend at the cross-modality level, embedding protective perturbations in vision to disrupt attacks before they propagate to the language model. By leveraging the semantic alignment between vision and language, AsD enhances adversarial robustness through model perturbations and system-level prompting. Unlike prior work that focuses on text-stage defenses, our method integrates visual defenses to reinforce prompt-based protections, mitigating jailbreaking attacks across benchmarks. Experiments on the LLaVA-1.5 show that AsD reduces attack success rates from 56.7% to 12.6% for typographic attacks and from 89.0% to 47.5% for adversarial perturbations. Further analysis reveals that the key bottleneck in vision-language security lies not in isolated model vulnerabilities, but in cross-modal interactions, where adversarial cues in the vision model fail to consistently activate the defense mechanisms of the language model.

</details>

---

## 240. The Role of Model Confidence on Bias Effects in Measured Uncertainties for Vision-Language Models

- [ ] The Role of Model Confidence on Bias Effects in Measured Uncertainties for Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.1104/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1104/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the growing adoption of Large Language Models (LLMs) for open-ended tasks, accurately assessing epistemic uncertainty, which reflects a model’s lack of knowledge, has become crucial to ensuring reliable outcomes. However, quantifying epistemic uncertainty in such tasks is challenging due to the presence of aleatoric uncertainty, which arises from multiple valid answers. While bias can introduce noise into epistemic uncertainty estimation, it may also reduce noise from aleatoric uncertainty. To investigate this trade-off, we conduct experiments on Visual Question Answering (VQA) tasks and find that mitigating prompt-introduced bias improves uncertainty quantification in GPT-4o. Building on prior work showing that LLMs tend to copy input information when model confidence is low, we further analyze how these prompt biases affect measured epistemic and aleatoric uncertainty across varying bias-free confidence levels with GPT-4o and Qwen2-VL. We find that all considered biases have greater effects in both uncertainties when bias-free model confidence is lower. Moreover, lower bias-free model confidence is associated with greater bias-induced underestimation of epistemic uncertainty, resulting in overconfident estimates, whereas it has no significant effect on the direction of bias effect in aleatoric uncertainty estimation. These distinct effects deepen our understanding of bias mitigation for uncertainty quantification and potentially inform the development of more advanced techniques.

</details>

---

## 241. The Security Threat of Compressed Projectors in Large Vision-Language Models

- [ ] The Security Threat of Compressed Projectors in Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.1111/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1111/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The choice of a suitable visual language projector (VLP) is critical to the successful training of large visual language models (LVLMs). Mainstream VLPs can be broadly categorized into compressed and uncompressed projectors, and each offers distinct advantages in performance and computational efficiency. However, their security implications have not been thoroughly examined. Our comprehensive evaluation reveals significant differences in their security profiles: compressed projectors exhibit substantial vulnerabilities, allowing adversaries to successfully compromise LVLMs even with minimal knowledge of structure information. In stark contrast, uncompressed projectors demonstrate robust security properties and do not introduce additional vulnerabilities. These findings provide critical guidance for researchers in selecting optimal VLPs that enhance the security and reliability of visual language models. The code is available athttps://github.com/btzyd/TCP.

</details>

---

## 242. CoViPAL: Layer-wise Contextualized Visual Token Pruning for Large Vision-Language Models

- [ ] CoViPAL: Layer-wise Contextualized Visual Token Pruning for Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.1127/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1127/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) process multimodal inputs consisting of text tokens and vision tokens extracted from images or videos. Due to the rich visual information, a single image can generate thousands of vision tokens, leading to high computational costs during the prefilling stage and significant memory overhead during decoding. Existing methods attempt to prune redundant vision tokens, revealing substantial redundancy in visual representations. However, these methods often struggle in shallow layers due to the lack of sufficient contextual information. We argue that many visual tokens are inherently redundant even in shallow layers and can be safely and effectively pruned with appropriate contextual signals. In this work, we propose CoViPAL, a layer-wise contextualized visual token pruning method that employs a Plug-and-Play Pruning Module (PPM) to predict and remove redundant vision tokens before they are processed by the LVLM. The PPM is lightweight, model-agnostic, and operates independently of the LVLM architecture, ensuring seamless integration with various models. Extensive experiments on multiple benchmarks demonstrate that CoViPAL outperforms training-free pruning methods under equal token budgets and surpasses training-based methods with comparable supervision. CoViPAL offers a scalable and efficient solution to improve inference efficiency in LVLMs without compromising accuracy.

</details>

---

## 243. Decoupled Proxy Alignment: Mitigating Language Prior Conflict for Multimodal Alignment inMLLMs

- [ ] Decoupled Proxy Alignment: Mitigating Language Prior Conflict for Multimodal Alignment inMLLMs | https://aclanthology.org/2025.findings-emnlp.1142/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1142/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have gained significant attention due to their impressive ability to integrate vision and language modalities. Recent advancements in MLLMs have primarily focused on improving performance through high-quality datasets, novel architectures, and optimized training strategies. However, in this paper, we identify a previously overlooked issue,language prior conflict, a mismatch between the inherent language priors of large language models (LLMs) and the language priors in training datasets. This conflict leads to suboptimal vision-language alignment, as MLLMs are prone to adapting to the language style of training samples. To address this issue, we propose a novel training method calledDecoupled Proxy Alignment (DPA). DPA introduces two key innovations: (1) the use of a proxy LLM during pretraining to decouple the vision-language alignment process from language prior interference, and (2) dynamic loss adjustment based on visual relevance to strengthen optimization signals for visually relevant tokens. Extensive experiments demonstrate that DPA significantly mitigates the language prior conflict, achieving superior alignment performance across diverse datasets, model families, and scales. Our method not only improves the effectiveness of MLLM training but also shows exceptional generalization capabilities, making it a robust approach for vision-language alignment.

</details>

---

## 244. How Jailbreak Defenses Work and Ensemble? A Mechanistic Investigation

- [ ] How Jailbreak Defenses Work and Ensemble? A Mechanistic Investigation | https://aclanthology.org/2025.findings-emnlp.1160/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1160/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Jailbreak attacks, where harmful prompts bypass generative models’ built-in safety, raise serious concerns about model vulnerability. While many defense methods have been proposed, the trade-offs between safety and helpfulness, and their application to Large Vision-Language Models (LVLMs), are not well understood. This paper systematically examines jailbreak defenses by reframing the standard generation task as a binary classification problem to assess model refusal tendencies for both harmful and benign queries. We identify two key defense mechanisms:safety shift, which increases refusal rates across all queries, andharmfulness discrimination, which improves the model’s ability to differentiate between harmful and benign inputs. Using these mechanisms, we develop two ensemble defense strategies—inter-mechanism and intra-mechanism ensembles—to balance safety and helpfulness. Experiments on the MM-SafetyBench and MOSSBench datasets with LLaVA-1.5 models show that these strategies effectively improve model safety or optimize the trade-off between safety and helpfulness.

</details>

---

## 245. EmoGist: Efficient In-Context Learning for Visual Emotion Understanding

- [ ] EmoGist: Efficient In-Context Learning for Visual Emotion Understanding | https://aclanthology.org/2025.findings-emnlp.116/

- **Link**: https://aclanthology.org/2025.findings-emnlp.116/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple descriptions of emotion labels, by analyzing the clusters of example images belonging to each label. At test time, we retrieve a version of description based on the cosine similarity of test image to cluster centroids, and feed it together with the test image to a fast LVLM for classification. Through our experiments, we show that EmoGist allows up to 12 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset.

</details>

---

## 246. Visual Self-Refinement for Autoregressive Models

- [ ] Visual Self-Refinement for Autoregressive Models | https://aclanthology.org/2025.findings-emnlp.1161/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1161/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autoregressive models excel in sequential modeling and have proven to be effective for vision-language data. However, the spatial nature of visual signals conflicts with the sequential dependencies of next-token prediction, leading to suboptimal results. This work proposes a plug-and-play refinement module to enhance the complex spatial correspondence modeling within the generated visual sequence. This module operates as a post-pretraining step tojointly refine all generated tokens of autoregressive model, enhancing vision-language modeling under a shared sequential prediction framework. By leveraging global context and relationship across the tokens, our method mitigates the error accumulation issue within the sequential generation. Experiments demonstrate that the proposed method improves the generation quality, enhancing the model’s ability to produce semantically consistent results.

</details>

---

## 247. LongLLaVA: Scaling Multi-modalLLMs to 1000 Images Efficiently via a Hybrid Architecture

- [ ] LongLLaVA: Scaling Multi-modalLLMs to 1000 Images Efficiently via a Hybrid Architecture | https://aclanthology.org/2025.findings-emnlp.1168/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1168/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Expanding the long-context capabilities of Multi-modal Large Language Models (MLLMs) is critical for advancing video understanding and high-resolution image analysis. Achieving this requires systematic improvements in model architecture, data construction, and training strategies, particularly to address challenges such as performance degradation with increasing image counts and high computational costs. In this paper, we propose a hybrid architecture that integrates Mamba and Transformer blocks, introduce data construction methods that capture both temporal and spatial dependencies, and employ a progressive training strategy. Our released model, LongLLaVA (Long-Context Large Language and Vision Assistant), demonstrates an effective balance between efficiency and performance. LongLLaVA achieves competitive results across various benchmarks while maintaining high throughput and low memory consumption. Notably, it can process nearly one thousand images on a single A100 80GB GPU, underscoring its potential for a wide range of multi-modal applications.

</details>

---

## 248. Exploring and Detecting Self-disclosure in Multi-modal posts onChinese Social Media

- [ ] Exploring and Detecting Self-disclosure in Multi-modal posts onChinese Social Media | https://aclanthology.org/2025.findings-emnlp.1173/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1173/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Self-disclosure can provide psychological comfort and social support, but it also carries the risk of unintentionally revealing sensitive information, leading to serious privacy concerns. Research on self-disclosure in Chinese multimodal contexts remains limited, lacking high-quality corpora, analysis, and methods for detection. This work focuses on self-disclosure behaviors on Chinese multimodal social media platforms and constructs a high-quality text-image corpus to address this critical data gap. We systematically analyze the distribution of self-disclosure types, modality preferences, and their relationship with user intent, uncovering expressive patterns unique to the Chinese multimodal context. We also fine-tune five multimodal large language models to enhance self-disclosure detection in multimodal scenarios. Among these models, the Qwen2.5-omni-7B achieved a strong performance, with a partial span F1 score of 88.2%. This study provides a novel research perspective on multimodal self-disclosure in the Chinese context.

</details>

---

## 249. PVTNL: Prompting Vision Transformers with Natural Language for Generalizable Person Re-identification

- [ ] PVTNL: Prompting Vision Transformers with Natural Language for Generalizable Person Re-identification | https://aclanthology.org/2025.findings-emnlp.1181/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1181/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Domain generalization person re-identification (DG-ReID) aims to train models on source domains and generalize to unseen target domains.While patch-based Vision Transformers have achieved success in capturing fine-grained visual features, they often overlook global semantic structure and suffer from feature entanglement, leading to overfitting across domains. Meanwhile, natural language provides high-level semantic abstraction but lacks spatial precision for fine-grained alignment.We propose PVTNL (Prompting Vision Transformers with Natural Language), a novel framework for generalizable person re-identification. PVTNL leverages the pre-trained vision-language model BLIP to extract aligned visual and textual embeddings. Specifically, we utilize body-part cues to segment images into semantically coherent regions and align them with corresponding natural language descriptions. These region-level textual prompts are encoded and injected as soft prompts into the Vision Transformer to guide localized feature learning. Notably, our language module is retained during inference, enabling persistent semantic grounding that enhances cross-domain generalization.Extensive experiments on standard DG-ReID benchmarks demonstrate that PVTNL achieves state-of-the-art performance. Ablation studies further confirm the effectiveness of body-part-level alignment, soft language prompting, and the benefit of preserving language guidance at inference time.

</details>

---

## 250. CaMMT: Benchmarking Culturally Aware Multimodal Machine Translation

- [ ] CaMMT: Benchmarking Culturally Aware Multimodal Machine Translation | https://aclanthology.org/2025.findings-emnlp.1220/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1220/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Translating cultural content poses challenges for machine translation systems due to the differences in conceptualizations between cultures, where language alone may fail to convey sufficient context to capture region-specific meanings. In this work, we investigate whether images can act as cultural context in multimodal translation. We introduce CaMMT, a human-curated benchmark of over 5,800 triples of images along with parallel captions in English and regional languages. Using this dataset, we evaluate five Vision Language Models (VLMs) in text-only and text+image settings. Through automatic and human evaluations, we find that visual context generally improves translation quality, especially in handling Culturally-Specific Items (CSIs), disambiguation, and correct gender marking. By releasing CaMMT, our objective is to support broader efforts to build and evaluate multimodal translation systems that are better aligned with cultural nuance and regional variations.

</details>

---

## 251. Mitigating Visual Knowledge Forgetting inMLLMInstruction-tuning via Modality-decoupled Gradient Descent

- [ ] Mitigating Visual Knowledge Forgetting inMLLMInstruction-tuning via Modality-decoupled Gradient Descent | https://aclanthology.org/2025.findings-emnlp.123/

- **Link**: https://aclanthology.org/2025.findings-emnlp.123/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent MLLMs have demonstrated strong visual understanding and reasoning after large-scale multimodal pre-training. However, instruction-tuning is typically text-driven with limited visual supervision, leading to significant visual forgetting and degradation of pre-trained visual knowledge. Existing fine-tuning and continual learning methods compress visual representations and emphasize task alignment over visual retention, failing to address this challenge. We present a novel perspective using effective rank to quantify the loss of visual representation richness, framing visual forgetting as excessive compression under the information bottleneck principle. To address this, we propose modality-decoupled gradient descent (MDGD), which regulates gradient updates to preserve the effective rank of visual features and explicitly disentangles visual learning from task-specific alignment. We further introduce a memory-efficient fine-tuning variant using gradient masking for parameter-efficient adaptation. Extensive experiments show that MDGD effectively mitigates visual forgetting across downstream tasks and models, maintaining pre-trained visual knowledge while supporting strong task adaptation.

</details>

---

## 252. NUMINA: A Natural Understanding Benchmark for Multi-dimensional Intelligence and Numerical Reasoning Abilities

- [ ] NUMINA: A Natural Understanding Benchmark for Multi-dimensional Intelligence and Numerical Reasoning Abilities | https://aclanthology.org/2025.findings-emnlp.1229/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1229/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in 2D multimodal large language models (MLLMs) have significantly improved performance in vision-language tasks. However, extending these capabilities to 3D environments remains a distinct challenge due to the complexity of spatial reasoning. Nevertheless, existing 3D benchmarks often lack fine-grained numerical reasoning task annotations, limiting MLLMs’ ability to perform precise spatial measurements and complex numerical reasoning. To address this gap, we introduce NUMINA, the first Natural Understanding benchmark for Multi-dimensional Intelligence and Numerical reasoning Abilities to enhance multimodal indoor perceptual understanding. NUMINA features multi-scale annotations and various question-answer pairs, generated using NUMINA-Flow, an automated annotation pipeline that integrates LLM rewriting and rule-based self-verification. We evaluate the performance of various state-of-the-art LLMs on NUMINA following the Chat-Scene framework, demonstrating that current LLMs struggle with multimodal numerical reasoning, particularly in performing precise computations such as distance and volume estimation, highlighting the need for further advancements in 3D models. The dataset and source codes can be obtained from https://github.com/fengshun124/NUMINA.

</details>

---

## 253. MoMentS: A Comprehensive Multimodal Benchmark for Theory of Mind

- [ ] MoMentS: A Comprehensive Multimodal Benchmark for Theory of Mind | https://aclanthology.org/2025.findings-emnlp.1230/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1230/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding Theory of Mind is essential for building socially intelligent multimodal agents capable of perceiving and interpreting human behavior. We introduce MoMentS (Multimodal Mental States), a comprehensive benchmark designed to assess the ToM capabilities of multimodal large language models (LLMs) through realistic, narrative-rich scenarios presented in short films. MoMentS includes over 2,300 multiple-choice questions spanning seven distinct ToM categories. The benchmark features long video context windows and realistic social interactions that provide deeper insight into characters’ mental states. We evaluate several MLLMs and find that although vision generally improves performance, models still struggle to integrate it effectively. For audio, models that process dialogues as audio do not consistently outperform transcript-based inputs. Our findings highlight the need to improve multimodal integration and point to open challenges that must be addressed to advance AI’s social understanding.

</details>

---

## 254. Lost in Embeddings: Information Loss in Vision–Language Models

- [ ] Lost in Embeddings: Information Loss in Vision–Language Models | https://aclanthology.org/2025.findings-emnlp.1235/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1235/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision–language models (VLMs) often process visual inputs through a pretrained vision encoder, followed by a projection into the language model’s embedding space via a connector component. While crucial for modality fusion, the potential information loss induced by this projection step and its direct impact on model capabilities remain understudied. We introduce two complementary approaches to examine and quantify this loss by analyzing the latent representation space. First, we evaluate semantic information preservation by analyzing changes in k-nearest neighbor relationships between image representations, before and after projection. Second, we directly measure information loss by reconstructing visual embeddings from the projected representation, localizing loss at an image patch level. Experiments reveal that connectors substantially distort the local geometry of visual representations, with k-nearest neighbors diverging by 40–60% post-projection, correlating with degradation in retrieval performance. The patch-level embedding reconstruction provides interpretable insights for model behavior on visually grounded question-answering tasks, finding that areas of high information loss reliably predict instances where models struggle.

</details>

---

## 255. PathoHR: Hierarchical Reasoning for Vision-Language Models in Pathology

- [ ] PathoHR: Hierarchical Reasoning for Vision-Language Models in Pathology | https://aclanthology.org/2025.findings-emnlp.124/

- **Link**: https://aclanthology.org/2025.findings-emnlp.124/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Accurate analysis of pathological images is essential for automated tumor diagnosis but remains challenging due to high structural similarity and subtle morphological variations in tissue images. Current vision-language (VL) models often struggle to capture the complex reasoning required for interpreting structured pathological reports. To address these limitations, we propose PathoHR-Bench, a novel benchmark designed to evaluate VL models’ abilities in hierarchical semantic understanding and compositional reasoning within the pathology domain. Results of this benchmark reveal that existing VL models fail to effectively model intricate cross-modal relationships, hence limiting their applicability in clinical setting. To overcome this, we further introduce a pathology-specific VL training scheme that generates enhanced and perturbed samples for multimodal contrastive learning. Experimental evaluations demonstrate that our approach achieves state-of-the-art performance on PathoHR-Bench and six additional pathology datasets, highlighting its effectiveness in fine-grained pathology representation.

</details>

---

## 256. mrCAD: Multimodal Communication to Refine Computer-aided Designs

- [ ] mrCAD: Multimodal Communication to Refine Computer-aided Designs | https://aclanthology.org/2025.findings-emnlp.1248/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1248/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In collaborative creation tasks, people steer artifacts towards specific goals by _refining_ them with _multimodal_ communication over multiple rounds of interaction. In contrast, generative AI excels at creating artifacts in a single turn but can struggle to make precise refinements that match our design intent. To close this gap, we present mrCAD, a dataset of multi-turn interactions in which pairs of humans iteratively created and refined computer-aided designs (CADs). In each game, a _Designer sent instructions to a _Maker_, explaining how to create and subsequently refine a CAD to match a target design that only the _Designer_ could see. mrCAD consists of 6,082 communication games, 15,163 instruction-execution rounds, played between 1,092 pairs of human players. Crucially, _Designers_ had access to two communication modalities – text and drawing. Analysis finds that players relied more on text in refinement than in initial generation instructions, and used different linguistic elements for refinement than for generation. We also find that state-of-the-art VLMs are better at following generation instructions than refinement instructions. These results lay the foundation for modeling multi-turn, multimodal communication not captured in prior datasets.

</details>

---

## 257. Evaluating Fairness in Large Vision-Language Models Across Diverse Demographic Attributes and Prompts

- [ ] Evaluating Fairness in Large Vision-Language Models Across Diverse Demographic Attributes and Prompts | https://aclanthology.org/2025.findings-emnlp.1251/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1251/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have recently achieved significant progress, demonstrating strong capabilities in open-world visual understanding. However, it is not yet clear how LVLMs address demographic biases in real life, especially the disparities across attributes such as gender, skin tone, age and race. In this paper, We empirically investigate visual fairness in several mainstream LVLMs by auditing their performance disparities across demographic attributes using public fairness benchmark datasets (e.g., FACET, UTKFace). Our fairness evaluation framework employs direct and single-choice question prompt on visual question-answering/classification tasks. Despite advancements in visual understanding, our zero-shot prompting results show that both open-source and closed-source LVLMs continue to exhibit fairness issues across different prompts and demographic groups. Furthermore, we propose a potential multi-modal Chain-of-thought (CoT) based strategy for unfairness mitigation, applicable to both open-source and closed-source LVLMs. This approach enhances transparency and offers a scalable solution for addressing fairness, providing a solid foundation for future research and practical efforts in unfairness mitigation. The dataset and code used in this study are publicly available at this GitHub Repository.

</details>

---

## 258. VIBE: Can aVLMRead the Room?

- [ ] VIBE: Can aVLMRead the Room? | https://aclanthology.org/2025.findings-emnlp.1252/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1252/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding human social behavior such as recognizing emotions and the social dynamics causing them is an important and challenging problem. While LLMs have made remarkable advances, they are limited to the textual domain and cannot account for the major role that non-verbal cues play in understanding social situations. Vision Language Models (VLMs) can potentially account for this gap, however their ability to make correct inferences over such social cues has received little attention. In this paper, we explore the capabilities of VLMs at social reasoning. We identify a previously overlooked limitation in VLMs: the Visual Social-Pragmatic Inference gap. To target this gap, we propose a new task for VLMs: Visual Social-Pragmatic Inference. We construct a high quality dataset to test the abilities of a VLM for this task and benchmark the performance of several VLMs on it.

</details>

---

## 259. Pearl: A Multimodal Culturally-AwareArabic Instruction Dataset

- [ ] Pearl: A Multimodal Culturally-AwareArabic Instruction Dataset | https://aclanthology.org/2025.findings-emnlp.1254/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1254/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mainstream large vision-language models (LVLMs) inherently encode cultural biases, highlighting the need for diverse multimodal datasets. To address this gap, we introduce PEARL, a large-scale Arabic multimodal dataset and benchmark explicitly designed for cultural understanding. Constructed through advanced agentic workflows and extensive human-in-the-loop annotations by 37 annotators from across the Arab world, PEARL comprises over 309K multimodal examples spanning ten culturally significant domains covering all Arab countries. We further provide two robust evaluation benchmarks (PEARL and PEARL-LITE) along with a specialized subset (PEARL-X) explicitly developed to assess nuanced cultural variations. Comprehensive evaluations on state-of-the-art open and proprietary LVLMs demonstrate that reasoning-centric instruction alignment substantially improves models’ cultural grounding compared to conventional scaling methods. PEARL establishes a foundational resource for advancing culturally-informed multimodal modeling research. All datasets and benchmarks are publicly available.

</details>

---

## 260. Looking Beyond the Pixels: Evaluating Visual Metaphor Understanding inVLMs

- [ ] Looking Beyond the Pixels: Evaluating Visual Metaphor Understanding inVLMs | https://aclanthology.org/2025.findings-emnlp.1257/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1257/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual metaphors are a complex vision–language phenomenon that requires both perceptual and conceptual reasoning to understand. They provide a valuable test of a model’s ability to interpret visual input and reason about it with creativity and coherence. We introduce ImageMet, a visual metaphor dataset, featuring 2177 synthetic and 350 human-annotated images. We benchmark several SOTA VLMs on two tasks: Visual Metaphor Captioning (VMC) and Visual Metaphor VQA (VM-VQA). We establish strong baselines by fine-tuning on ImageMet, which yields substantial performance gains in VMC (+4.67% SBERT-Similarity, +4.84% task-specific metric) and VM-VQA (+9.3% Accuracy on average). Additionally, we introduce a task-specific CoT prompting strategy that outperforms standard few-shot baselines (+1.99% in VMC, +5.21% in VM-VQA). We observe that despite strong performance on the VMC task, VLMs still significantly lag behind humans in understanding visual metaphors, indicating that their success often relies on learned associations rather than genuine analytical reasoning. We note that this gap is often obscured in metaphor captioning tasks where the automatic metrics correlate only moderately at best with human judgment (Pearson r < 0.6), highlighting the need for careful, holistic evaluation of the visual metaphor understanding of the models.

</details>

---

## 261. ProcVQA: Benchmarking the Effects of Structural Properties in Mined Process Visualizations on Vision–Language Model Performance

- [ ] ProcVQA: Benchmarking the Effects of Structural Properties in Mined Process Visualizations on Vision–Language Model Performance | https://aclanthology.org/2025.findings-emnlp.1266/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1266/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models have shown both impressive capabilities and notable failures in data visualization understanding tasks, but we have limited understanding on how specific properties within a visualization type affect model performance. We present ProcVQA, a benchmark designed to analyze how VLM performance can be affected by structure type and structural density of visualizations depicting frequent patterns mined from sequence data. ProcVQA consists of mined process visualizations spanning three structure types (linear sequences, tree, graph) with varying levels of structural density (quantified using the number of nodes and edges), with expert-validated QA pairs on these visualizations. We evaluate 21 proprietary and open-source models on the dataset on two major tasks: visual data extraction (VDE) and visual question answering (VQA) (with four categories of questions). Our analysis reveals three key findings. First, models exhibit steep performance drops on multi-hop reasoning, with question type and structure type impacting the degradation. Second, structural density strongly affects VDE performance: hallucinations and extraction errors increase with edge density, even in frontier models. Third, extraction accuracy does not necessarily translate into strong reasoning ability. By isolating structural factors through controlled visualization generation, ProcVQA enables precise identification of VLM limitations. ProcVQA is available at: https://github.com/kzintas/ProcVQA.

</details>

---

## 262. UnderstandingGUIAgent Localization Biases through Logit Sharpness

- [ ] UnderstandingGUIAgent Localization Biases through Logit Sharpness | https://aclanthology.org/2025.findings-emnlp.1268/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1268/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have enabled GUI agents to interact with operating systems by grounding language into spatial actions. Despite their promising performance, these models frequently exhibit hallucinations—systematic localization errors that compromise reliability. We propose a fine-grained evaluation framework that categorizes model predictions into four distinct types, revealing nuanced failure modes beyond traditional accuracy metrics. To better quantify model uncertainty, we introduce the Peak Sharpness Score (PSS), a metric that evaluates the alignment between semantic continuity and logits distribution in coordinate prediction. Building on this insight, we further propose Context-Aware Cropping, a training-free technique that improves model performance by adaptively refining input context. Extensive experiments demonstrate that our framework and methods provide actionable insights and enhance the interpretability and robustness of GUI agent behavior.

</details>

---

## 263. HomoGraphAdapter: A Homogeneous Graph Neural Network as an Effective Adapter for Vision-Language Models

- [ ] HomoGraphAdapter: A Homogeneous Graph Neural Network as an Effective Adapter for Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.1270/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1270/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs), such as CLIP, have exhibited significant advancements in recognizing visual concepts through natural language guidance. However, adapting these models to downstream tasks remains challenging. Existing adaptation methods either overlook the structural knowledge between the text and image modalities or create overly complex graphs containing redundant information for alignment, leading to suboptimal classification performance and increased computational overhead. This paper proposes a novel adapter-tuning methodology named Homogeneous Graph Adapter (HomoGraphAdapter), which transforms diverse textual and visual descriptions into a unified set of node representations and establishes edges between nodes for inter-modal and cross-modal semantic alignment. We leverage a straightforward homogeneous Graph Neural Network (GNN) to adapt positive and negative classifiers across text and image modalities. The classifiers comprehensively enhance the performance for few-shot classification and OOD generalization. Compared with the SOTA approach HeGraphAdapter, HomoGraphAdapter improves classification accuracy by an average of 1.51% for 1-shot and 0.74% for 16-shot on 11 datasets, while also reducing both precomputation time and training time.

</details>

---

## 264. Anatomy of a Feeling: Narrating Embodied Emotions via Large Vision-Language Models

- [ ] Anatomy of a Feeling: Narrating Embodied Emotions via Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.1276/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1276/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The embodiment of emotional reactions from body parts contains rich information about our affective experiences. We propose a framework that utilizes state-of-the-art large vision language models (LVLMs) to generate Embodied LVLM Emotion Narratives (ELENA). These are well-defined, multi-layered text outputs, primarily comprising descriptions that focus on the salient body parts involved in emotional reactions. We also employ attention maps and observe that contemporary models exhibit a persistent bias towards the facial region. Despite this limitation, we observe that our employed framework can effectively recognize embodied emotions in face-masked images, outperforming baselines without any fine-tuning. ELENA opens a new trajectory for embodied emotion analysis across the modality of vision and enriches modeling in an affect-aware setting.

</details>

---

## 265. Zero-Shot Fine-Grained Image Classification Using Large Vision-Language Models

- [ ] Zero-Shot Fine-Grained Image Classification Using Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.1280/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1280/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated impressive performance on vision-language reasoning tasks. However, their potential for zero-shot fine-grained image classification, a challenging task requiring precise differentiation between visually similar categories, remains underexplored. We present a novel method that transforms zero-shot fine-grained image classification into a visual question-answering framework, leveraging LVLMs’ comprehensive understanding capabilities rather than relying on direct class name generation. We enhance model performance through a novel attention intervention technique. We also address a key limitation in existing datasets by developing more comprehensive and precise class description benchmarks. We validate the effectiveness of our method through extensive experimentation across multiple fine-grained image classification benchmarks. Our proposed method consistently outperforms the current state-of-the-art (SOTA) approach, demonstrating both the effectiveness of our method and the broader potential of LVLMs for zero-shot fine-grained classification tasks. Code and Datasets: https://github.com/Atabuzzaman/Fine-grained-classification

</details>

---

## 266. SteerVLM: Robust Model Control through Lightweight Activation Steering for Vision Language Models

- [ ] SteerVLM: Robust Model Control through Lightweight Activation Steering for Vision Language Models | https://aclanthology.org/2025.findings-emnlp.1285/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1285/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This work introduces SteerVLM, a lightweight steering module designed to guide Vision-Language Models (VLMs) towards outputs that better adhere to desired instructions. Our approach learns from the latent embeddings of paired prompts encoding target and converse behaviors to dynamically adjust activations connecting the language modality with image context. This allows for fine-grained, inference-time control over complex output semantics without modifying model weights while preserving performance on off-target tasks. Our steering module requires learning parameters equal to 0.14% of the original VLM’s size. Our steering module gains model control through dimension-wise activation modulation and adaptive steering across layers without requiring pre-extracted static vectors or manual tuning of intervention points. Furthermore, we introduce VNIA (Visual Narrative Intent Alignment), a multimodal dataset specifically created to facilitate the development and evaluation of VLM steering techniques. Our method outperforms existing intervention techniques on steering and hallucination mitigation benchmarks for VLMs and proposes a robust solution for multimodal model control through activation engineering.

</details>

---

## 267. GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning

- [ ] GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning | https://aclanthology.org/2025.findings-emnlp.1284/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1284/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces GeoChain, a large-scale benchmark for evaluating step-by-step geographic reasoning in multimodal large language models (MLLMs). Leveraging 1.46 million Mapillary street-level images, GeoChain pairs each image with a 21-step chain-of-thought (CoT) question sequence (over 30 million Q&A pairs). These sequences guide models from coarse attributes to fine-grained localization across four reasoning categories - visual, spatial, cultural, and precise geolocation - annotated by difficulty. Images are also enriched with semantic segmentation (150 classes) and a visual locatability score. Our benchmarking of frontier MLLMs on a diverse 2,088-image subset reveals consistent challenges: models frequently exhibit weaknesses in visual grounding, display erratic reasoning, and struggle to achieve accurate localization, especially as the reasoning complexity escalates. GeoChain offers a robust diagnostic methodology, critical for fostering significant advancements in complex geographic reasoning within MLLMs.

</details>

---

## 268. RG-VQA: Leveraging Retriever-Generator Pipelines for Knowledge Intensive Visual Question Answering

- [ ] RG-VQA: Leveraging Retriever-Generator Pipelines for Knowledge Intensive Visual Question Answering | https://aclanthology.org/2025.findings-emnlp.1306/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1306/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose a method to improve the reasoning capabilities of Visual Question Answering (VQA) systems by integrating Dense Passage Retrievers (DPRs) with Vision Language Models (VLMs). While recent works focus on the application of knowledge graphs and chain-of-thought reasoning, we recognize that the complexity of graph neural networks and end-to-end training remain significant challenges. To address these issues, we introduce **R**elevance **G**uided **VQA** (**RG-VQA**), a retriever-generator pipeline that uses DPRs to efficiently extract relevant information from structured knowledge bases. Our approach ensures scalability to large graphs without significant computational overhead. Experiments on the ScienceQA dataset show that RG-VQA achieves state-of-the-art performance, surpassing human accuracy and outperforming GPT-4 by more than . This demonstrates the effectiveness of RG-VQA in boosting the reasoning capabilities of VQA systems and its potential for practical applications.

</details>

---

## 269. BannerBench: Benchmarking Vision Language Models for Multi-Ad Selection with Human Preferences

- [ ] BannerBench: Benchmarking Vision Language Models for Multi-Ad Selection with Human Preferences | https://aclanthology.org/2025.findings-emnlp.1311/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1311/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Web banner advertisements, which are placed on websites to guide users to a targeted landing page (LP), are still often selected manually because human preferences are important in selecting which ads to deliver. To automate this process, we propose a new benchmark, BannerBench, to evaluate the human preference-driven banner selection process using vision-language models (VLMs). This benchmark assesses the degree of alignment with human preferences in two tasks: a ranking task and a best-choice task, both using sets of five images derived from a single LP. Our experiments show that VLMs are moderately correlated with human preferences on the ranking task. In the best-choice task, most VLMs perform close to chance level across various prompting strategies. These findings suggest that although VLMs have a basic understanding of human preferences, most of them struggle to pinpoint a single suitable option from many candidates.

</details>

---

## 270. TIU-Bench: A Benchmark for Evaluating Large Multimodal Models on Text-rich Image Understanding

- [ ] TIU-Bench: A Benchmark for Evaluating Large Multimodal Models on Text-rich Image Understanding | https://aclanthology.org/2025.findings-emnlp.1318/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1318/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-rich images are ubiquitous in real-world applications, serving as a critical medium for conveying complex information and facilitating accessibility.Despite recent advances driven by Multimodal Large Language Models (MLLMs), existing benchmarks suffer from limited scale, fragmented scenarios, and evaluation protocols that fail to fully capture holistic image understanding.To address these gaps, we present TIU-Bench, a large-scale, multilingual benchmark comprising over 100,000 full-image annotations and 22,000 rigorously validated question-answer (QA) pairs that span 18 subtasks across diverse real-world scenarios.TIU-Bench introduces a novel full-image structured output format that jointly models geometric, textual, and relational information, enabling fine-grained evaluation of perception and reasoning capabilities. Furthermore, we propose a two-stage understanding framework named T2TIU, which first generates a structured representation of the entire image and subsequently conducts reasoning on this representation to address complex visual-textual queries.Extensive experiments on 10 state-of-the-art generative models highlight the challenges and opportunities in advancing text-rich image understanding.Our benchmark and framework provide a comprehensive platform for developing and evaluating next-generation multimodal AI systems.

</details>

---

## 271. Leveraging Unpaired Feedback for Long-TermLLM-based Recommendation Tuning

- [ ] Leveraging Unpaired Feedback for Long-TermLLM-based Recommendation Tuning | https://aclanthology.org/2025.findings-emnlp.1332/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1332/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Most recommender systems focus on short-term objectives such as click-through rate, often at the expense of long-term user satisfaction. This can lead to echo chambers, where users are repeatedly exposed to redundant content. While recent efforts integrate Large Language Models (LLMs) into recommendation, they typically inherit this short-sighted focus. In this work, we highlight unpaired feedback—implicit signals such as continued engagement (positive) or silent disengagement (negative) that lack explicit contrastive labels—as a key challenge for long-term recommendation. Effectively learning from such feedback is crucial for improving LLM-based recommenders in dynamic user environments. To this end, we propose ULRec (Unpaired Feedback for Long-Term LLM-based Recommendation Tuning), a simple framework that fine-tunes LLMs using both positive and negative unpaired feedback. ULRec leverages the KTO algorithm to incorporate these signals without requiring paired supervision. Despite its simplicity, ULRec consistently improves long-term recommendation performance, demonstrating the value of modeling unpaired user feedback.

</details>

---

## 272. QEVA: A Reference-Free Evaluation Metric for Narrative Video Summarization with Multimodal Question Answering

- [ ] QEVA: A Reference-Free Evaluation Metric for Narrative Video Summarization with Multimodal Question Answering | https://aclanthology.org/2025.findings-emnlp.1340/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1340/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video-to-text summarization remains underexplored in terms of comprehensive evaluation methods. Traditional n-gram overlap-based metrics and recent large language model (LLM)-based approaches depend heavily on human-written reference summaries, limiting their practicality and sensitivity to nuanced semantic aspects. In this paper, we propose QEVA, a reference-free metric evaluating candidate summaries directly against source videos through multimodal question answering. QEVA assesses summaries along three clear dimensions: Coverage, Factuality, and Temporal Coherence. We also introduce MLVU(VS)-Eval, a new annotated benchmark derived from the MLVU dataset, comprising 800 summaries generated from 200 videos using state-of-the-art video-language multimodal models. This dataset establishes a transparent and consistent framework for evaluation. Experimental results demonstrate that QEVA shows higher correlation with human judgments compared to existing approaches, as measured by Kendall’s𝜏b,𝜏c, and Spearman’s𝜌. We hope that our benchmark and metric will facilitate meaningful progress in video-to-text summarization research and provide valuable insights for the development of future evaluation methods.

</details>

---

## 273. AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering

- [ ] AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering | https://aclanthology.org/2025.findings-emnlp.1350/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1350/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical Multimodal Large Language Models (Med-MLLMs) have shown great promise in medical visual question answering (Med-VQA). However, when deployed in low-resource settings where abundant labeled data are unavailable, existing Med-MLLMs commonly fail due to their medical reasoning capability bottlenecks: (i) the intrinsic reasoning bottleneck that ignores the details from the medical image; (ii) the extrinsic reasoning bottleneck that fails to incorporate specialized medical knowledge. To address those limitations, we propose AMANDA, a training-free agentic framework that performs medical knowledge augmentation via LLM agents. Specifically, our intrinsic medical knowledge augmentation focuses on coarse-to-fine question decomposition for comprehensive diagnosis, while extrinsic medical knowledge augmentation grounds the reasoning process via biomedical knowledge graph retrieval. Extensive experiments across eight Med-VQA benchmarks demonstrate substantial improvements in both zero-shot and few-shot Med-VQA settings. The code is available athttps://github.com/REAL-Lab-NU/AMANDA.

</details>

---

## 274. Mixed Signals: DecodingVLMs’ Reasoning and Underlying Bias in Vision-Language Conflict

- [ ] Mixed Signals: DecodingVLMs’ Reasoning and Underlying Bias in Vision-Language Conflict | https://aclanthology.org/2025.findings-emnlp.1351/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1351/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have demonstrated impressive performance by effectively integrating visual and textual information to solve complex tasks. However, it is not clear how these models reason over the visual and textual data together, nor how the flow of information between modalities is structured. In this paper, we examine how VLMs reason by analyzing their biases when confronted with scenarios that present conflicting image and text cues—a common occurrence in real-world applications. To uncover the extent and nature of these biases, we build upon existing benchmarks to create five datasets containing mismatched image-text pairs, covering topics in mathematics, science, and visual descriptions. Our analysis shows that VLMs favor text in simpler queries but shift toward images as query complexity increases. This bias correlates with model scale, with the difference between the percentage of image- and text-preferred responses ranging from +56.8% (image favored) to -85.1% (text favored), depending on the task and model. In addition, we explore three mitigation strategies: simple prompt modifications, modifications that explicitly instruct models on how to handle conflicting information (akin to chain-of-thought prompting), and a task decomposition strategy that analyzes each modality separately before combining their results. Our findings indicate that the effectiveness of these strategies in identifying and mitigating bias varies significantly and is closely linked to the model’s overall performance on the task and the specific modality in question. We released our dataset and code.

</details>

---

## 275. Mitigating Hallucination in Large Vision-Language Models through Aligning Attention Distribution to Information Flow

- [ ] Mitigating Hallucination in Large Vision-Language Models through Aligning Attention Distribution to Information Flow | https://aclanthology.org/2025.findings-emnlp.1352/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1352/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Due to the unidirectional masking mechanism, Decoder-Only models propagate information from left to right. LVLMs (Large Vision-Language Models) follow the same architecture, with visual information gradually integrated into semantic representations during forward propagation. Through systematic analysis, we observe that over 80% of the visual information is absorbed into the semantic representations. However, the model’s attention still predominantly focuses on the visual representations. This misalignment between the attention distribution and the actual information flow undermines the model’s visual understanding ability and contributes to hallucinations.To address this issue, we enhance the model’s visual understanding by leveraging the core information embedded in semantic representations. Specifically, we identify attention heads that focus on core semantic representations based on their attention distributions. Then, through a two-stage optimization paradigm, we propagate the advantages of these attention heads across the entire model, aligning the attention distribution with the actual information flow.We evaluate our method on three image captioning benchmarks using five different LVLMs,demonstrating its effectiveness in significantly reducing hallucinations. Further experiments reveal a trade-off between reduced hallucinations and richer details. Notably, our method allows for manual adjustment of the model’s conservativeness, enabling flexible control to meet diverse real-world requirements.

</details>

---

## 276. On the Fine-Grained Planning Abilities ofVLMWeb Agents

- [ ] On the Fine-Grained Planning Abilities ofVLMWeb Agents | https://aclanthology.org/2025.findings-emnlp.1382/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1382/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have shown promise as web agents, yet their planning—the ability to devise strategies or action sequences to complete tasks—remains understudied. While prior works focus on VLM’s perception and overall success rates (i.e., goal completion), fine-grained investigation of their planning has been overlooked. To address this gap, we examine VLMs’ capability to (1) understand temporal relationships within web contexts, and (2) assess plans of actions across diverse scenarios. We design four simple yet effective tests to delve into these nuanced aspects around planning. Our results across nineteen VLMs reveal that these models exhibit limited performance in the aforementioned skills and are not reliable to function as web agents. To facilitate future work, we release our planning evaluations and data, providing a foundation for advancing the future research in this area.

</details>

---

## 277. STA-CoT: Structured Target-Centric Agentic Chain-of-Thought for Consistent Multi-Image Geological Reasoning

- [ ] STA-CoT: Structured Target-Centric Agentic Chain-of-Thought for Consistent Multi-Image Geological Reasoning | https://aclanthology.org/2025.findings-emnlp.1386/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1386/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reliable multi-image geological reasoning is essential for automating expert tasks in remote-sensing mineral exploration, yet remains challenging for multimodal large language models (MLLMs) due to the need for locating target areas, accurate cross-image referencing, and consistency over long reasoning chains. We propose STA-CoT, a Structured Target-centric Agentic Chain-of-Thought framework that orchestrates planning, execution, and verification agents to decompose, ground, and iteratively refine reasoning steps over geological and hyperspectral image sets. By aligning each reasoning step to specific image target areas and enforcing consistency through agentic verification and majority voting, STA-CoT robustly mitigates tool errors, long-chain inconsistencies, and error propagation. We rigorously evaluate STA-CoT on MineBench, a dedicated benchmark for multi-image mineral exploration, demonstrating substantial improvements over existing multimodal chain-of-thought and agentic baselines. Our results establish STA-CoT as a reliable and robust solution for consistent multi-image geological reasoning, advancing automated scientific discovery in mineral exploration.

</details>

---

## 278. GeoPQA: Bridging the Visual Perception Gap inMLLMs for Geometric Reasoning

- [ ] GeoPQA: Bridging the Visual Perception Gap inMLLMs for Geometric Reasoning | https://aclanthology.org/2025.findings-emnlp.1400/

- **Link**: https://aclanthology.org/2025.findings-emnlp.1400/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in reinforcement learning (RL) have enhanced the reasoning abilities of large language models (LLMs), yet the impact on multimodal LLMs (MLLMs) is limited. Particularly in vision-intensive tasks like geometric reasoning, MLLMs hallucinate frequently, leading to inaccurate reasoning. We attribute this to the perceptual bottleneck in MLLMs, which caps the benefits of reasoning training. To quantify this, we design a Geo-Perception Question-Answering (GeoPQA) benchmark, targeting basic geometric concepts and spatial relationships. Experiments on GeoPQA reveal significant shortcomings of MLLMs in visual perception, constraining RL reward signals for training. To address this bottleneck, we propose a two-stage RL training framework by first enhancing the visual perception of geometric structures, then fostering reasoning capabilities. Applied to Qwen2.5-VL-3B-Instruct, our two-stage training improves geometric reasoning by 9.7% and problem-solving by 9.1%, compared to the direct reasoning training approach. Our method also generalizes to other vision-intensive domains like figure understanding, highlighting the importance of perceptual grounding in effective MLLM reasoning.

</details>

---

## 279. Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation

- [ ] Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation | https://aclanthology.org/2025.findings-emnlp.140/

- **Link**: https://aclanthology.org/2025.findings-emnlp.140/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present an Audio-Visual Language Model (AVLM) for expressive speech generation by integrating full-face visual cues into a pre-trained expressive speech model. We explore multiple visual encoders and multimodal fusion strategies during pre-training to identify the most effective integration approach. Subsequent fine-tuning on emotion recognition and expressive dialogue tasks yields substantial gains over speech-only baselines (e.g.,+5F1 in emotion recognition). AVLM highlights the value of expressive visual information in guiding speech generation and offers a foundation for end-to-end multimodal conversational systems.

</details>

---

## 280. Automating eHMIAction Design withLLMs for Automated Vehicle Communication

- [ ] Automating eHMIAction Design withLLMs for Automated Vehicle Communication | https://aclanthology.org/2025.findings-emnlp.148/

- **Link**: https://aclanthology.org/2025.findings-emnlp.148/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The absence of explicit communication channels between automated vehicles (AVs) and other road users requires the use of external Human-Machine Interfaces (eHMIs) to convey messages effectively in uncertain scenarios. Currently, most eHMI studies employ predefined text messages and manually designed actions to perform these messages, which limits the real-world deployment of eHMIs, where adaptability in dynamic scenarios is essential. Given the generalizability and versatility of large language models (LLMs), they could potentially serve as automated action designers for the message-action design task. To validate this idea, we make three contributions: (1) We propose a pipeline that integrates LLMs and 3D renderers, using LLMs as action designers to generate executable actions for controlling eHMIs and rendering action clips. (2) We collect a user-rated Action-Design Scoring dataset comprising a total of 320 action sequences for eight intended messages and four representative eHMI modalities. The dataset validates that LLMs can translate intended messages into actions close to a human level, particularly for reasoning-enabled LLMs. (3) We introduce two automated raters, Action Reference Score (ARS) and Vision-Language Models (VLMs), to benchmark 18 LLMs, finding that the VLM aligns with human preferences yet varies across eHMI modalities. The source code, prompts, Blender scenarios, and rendered clips are available at https://github.com/ApisXia/AutoActionDesign.

</details>

---

## 281. PreGenie: An Agentic Framework for High-quality Visual Presentation Generation

- [ ] PreGenie: An Agentic Framework for High-quality Visual Presentation Generation | https://aclanthology.org/2025.findings-emnlp.165/

- **Link**: https://aclanthology.org/2025.findings-emnlp.165/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual presentations are vital for effective communication. Early attempts to automate their creation using deep learning often faced issues such as poorly organized layouts, inaccurate text summarization, and a lack of image understanding, leading to mismatched visuals and text. These limitations restrict their application in formal contexts like business and scientific research. To address these challenges, we propose PreGenie, an agentic and modular framework powered by multimodal large language models (MLLMs) for generating high-quality visual presentations.PreGenie is built on the Slidev presentation framework, where slides are rendered from Markdown code. It operates in two stages: (1) Analysis and Initial Generation, which summarizes multimodal input and generates initial code, and (2) Review and Re-generation, which iteratively reviews intermediate code and rendered slides to produce final, high-quality presentations. Each stage leverages multiple MLLMs that collaborate and share information. Comprehensive experiments demonstrate that PreGenie excels in multimodal understanding, outperforming existing models in both aesthetics and content consistency, while aligning more closely with human design preferences.

</details>

---

## 282. On Domain-Adaptive Post-Training for Multimodal Large Language Models

- [ ] On Domain-Adaptive Post-Training for Multimodal Large Language Models | https://aclanthology.org/2025.findings-emnlp.17/

- **Link**: https://aclanthology.org/2025.findings-emnlp.17/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Adapting general multimodal large language models (MLLMs) to specific domains, such as scientific and industrial fields, is highly significant in promoting their practical applications. This paper systematically investigates domain adaptation of MLLMs via post-training, focusing on data synthesis, training pipeline, and task evaluation. (1) **Data Synthesis**: Using only open-source models, we develop a generate-then-filter pipeline that curates diverse visual instruction tasks based on domain-specific image-caption pairs. The resulting data surpass the data synthesized by manual rules or strong closed-source models in enhancing domain-specific performance. (2) **Training Pipeline**: Unlike general MLLMs that typically adopt a two-stage training paradigm, we find that a single-stage approach is more effective for domain adaptation. (3) **Task Evaluation**: We conduct extensive experiments in high-impact domains such as biomedicine, food, and remote sensing, by post-training a variety of MLLMs and then evaluating MLLM performance on various domain-specific tasks. Finally, we fully open-source our models, code, and data to encourage future research in this area.

</details>

---

## 283. Distill Visual Chart Reasoning Ability fromLLMs toMLLMs

- [ ] Distill Visual Chart Reasoning Ability fromLLMs toMLLMs | https://aclanthology.org/2025.findings-emnlp.172/

- **Link**: https://aclanthology.org/2025.findings-emnlp.172/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Solving complex chart Q&A tasks requires advanced visual reasoning abilities in multimodal large language models (MLLMs), including recognizing key information from visual inputs and conducting reasoning over it. While fine-tuning MLLMs for reasoning is critical, collecting and annotating charts and questions is expensive, hard to scale, and often results in low-quality annotations. To address this, we propose Code-as-Intermediary Translation (CIT), a cost-effective, efficient and scalable data synthesis method for distilling visual reasoning abilities from LLMs to MLLMs. The code serves as an intermediary that translates visual chart representations into textual representations, enabling language models to understand cross-modal information and generate reasoning chains accordingly. In this way, we can employ text-based synthesizing techniques to expand chart-plotting code and generate high-quality Q&A pairs for training models. This produces ReachQA, a dataset containing 3k reasoning-intensive charts and 20k Q&A pairs to enhance both recognition and reasoning abilities of MLLMs. Experiments show that models fine-tuned with ReachQA not only perform well on chart-related tasks but also show performance gains on general reasoning benchmarks.

</details>

---

## 284. DocAssistant: Integrating Key-region Reading and Step-wise Reasoning for Robust Document Visual Question Answering

- [ ] DocAssistant: Integrating Key-region Reading and Step-wise Reasoning for Robust Document Visual Question Answering | https://aclanthology.org/2025.findings-emnlp.187/

- **Link**: https://aclanthology.org/2025.findings-emnlp.187/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding the multimodal documents is essential for accurately extracting relevant evidence and using it for reasoning. Existing document understanding models struggle to focus on key information and tend to generate answers straightforwardly, ignoring evidence from source documents and lacking interpretability. In this work, we improve the visual encoder to focus on key information relevant to the question and address the shortcomings of existing document visual question-answering datasets to provide the model with the ability to answer questions step-wise, dubbed DocAssistant. Specifically, for the visual side, we propose an effective vision-language adaptation that fuses text into visual encoders without compromising the performance of the original model. For the language side, we use Multimodal Large Language Models (MLLMs) as data generators and checkers to produce high-quality step-wise question-and-answer pairs for document images. We then use the generated high-quality data to train our enhanced model, specifically designed to solve complex questions that require reasoning or multi-hop question answering. The experimental results demonstrate the effectiveness of the model.

</details>

---

## 285. Beyond Spurious Signals: Debiasing Multimodal Large Language Models via Counterfactual Inference and Adaptive Expert Routing

- [ ] Beyond Spurious Signals: Debiasing Multimodal Large Language Models via Counterfactual Inference and Adaptive Expert Routing | https://aclanthology.org/2025.findings-emnlp.205/

- **Link**: https://aclanthology.org/2025.findings-emnlp.205/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown substantial capabilities in integrating visual and textual information, yet frequently rely on spurious correlations, undermining their robustness and generalization in complex multimodal reasoning tasks. This paper addresses the critical challenge of superficial correlation bias in MLLMs through a novel causal mediation-based debiasing framework. Specially, we distinguishing core semantics from spurious textual and visual contexts via counterfactual examples to activate training-stage debiasing and employ a Mixture-of-Experts (MoE) architecture with dynamic routing to selectively engages modality-specific debiasing experts. Empirical evaluation on multimodal sarcasm detection and sentiment analysis tasks demonstrates that our framework significantly surpasses unimodal debiasing strategies and existing state-of-the-art models.

</details>

---

## 286. ReLoop: “Seeing Twice and Thinking Backwards” via Closed-loop Training to Mitigate Hallucinations in Multimodal understanding

- [ ] ReLoop: “Seeing Twice and Thinking Backwards” via Closed-loop Training to Mitigate Hallucinations in Multimodal understanding | https://aclanthology.org/2025.findings-emnlp.222/

- **Link**: https://aclanthology.org/2025.findings-emnlp.222/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Multimodal Large Language Models (MLLMs) have achieved remarkable progress in open-ended visual question answering, they remain vulnerable to hallucinations. These are outputs that contradict or misrepresent input semantics, posing a critical challenge to the reliability and factual consistency. Existing methods often rely on external verification or post-hoc correction, lacking an internal mechanism to validate outputs directly during training. To bridge this gap, we propose ReLoop, a unified closed-loop training framework that encourages multimodal consistency for cross-modal understanding in MLLMs. ReLoop adopts a ring-shaped structure that integrates three complementary consistency feedback mechanisms, obliging MLLMs to “seeing twice and thinking backwards”. Specifically, ReLoop employs the frozen Consistency Feedback Plugin (CFP), comprising semantic reconstruction, visual description, and an attention supervision module for attention alignment. These components collectively enforce semantic reversibility, visual consistency, and interpretable attention, enabling the model to correct its outputs during training. Extensive evaluations and analyses demonstrate the effectiveness of ReLoop in reducing hallucination rates across multiple benchmarks, establishing a robust method for hallucination mitigation in MLLMs. We will release our source code and data in the camera-ready version. The code is available at: https://github.com/ZiyanHuang11/Reloop-hallucinations.

</details>

---

## 287. Sparkle: Mastering Basic Spatial Capabilities in Vision Language Models Elicits Generalization to Spatial Reasoning

- [ ] Sparkle: Mastering Basic Spatial Capabilities in Vision Language Models Elicits Generalization to Spatial Reasoning | https://aclanthology.org/2025.findings-emnlp.217/

- **Link**: https://aclanthology.org/2025.findings-emnlp.217/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) excel in many downstream tasks but struggle with spatial reasoning, which is crucial for navigation and interaction with physical environments. Specifically, many spatial reasoning tasks rely on fundamental two-dimensional (2D) capabilities, yet our evaluation shows that state-of-the-art VLMs often produce implausible or incorrect solutions for composite spatial problems, including simple pathfinding tasks that humans solve effortlessly at a glance. To address this, we explore an effective approach to enhance 2D spatial reasoning in VLMs by training them solely on basic spatial capabilities. We first disentangle 2D spatial reasoning into three core components: direction comprehension, distance estimation, and localization. Our central hypothesis is that mastering these basic capabilities will significantly boost performance on more complex spatial tasks requiring advanced reasoning and combinatorial problem-solving, as well as generalize to real-world visual-spatial scenarios. To test this hypothesis, we introduce Sparkle, a framework that generates synthetic data to provide targeted supervision for VLMs across these three basic spatial capabilities, producing an instruction dataset for each capability. Our experiments demonstrate that VLMs fine-tuned with Sparkle achieve substantial improvements, not only on basic tasks but also in generalizing to composite and out-of-distribution real-world spatial reasoning tasks. These findings highlight that enhancing basic spatial capabilities through synthetic generalization effectively improves complex spatial reasoning, offering insights into systematic strategies for boosting VLMs’ spatial understanding. Source codes of Sparkle are available at https://github.com/YihongT/Sparkle.

</details>

---

## 288. A Survey on Training-free Alignment of Large Language Models

- [ ] A Survey on Training-free Alignment of Large Language Models | https://aclanthology.org/2025.findings-emnlp.238/

- **Link**: https://aclanthology.org/2025.findings-emnlp.238/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The alignment of large language models (LLMs) aims to ensure their outputs adhere to human values, ethical standards, and legal norms. Traditional alignment methods often rely on resource-intensive fine-tuning (FT), which may suffer from knowledge degradation and face challenges in scenarios where the model accessibility or computational resources are constrained. In contrast, training-free (TF) alignment techniques—leveraging in-context learning, decoding-time adjustments, and post-generation corrections—offer a promising alternative by enabling alignment without heavily retraining LLMs, making them adaptable to both open-source and closed-source environments. This paper presents the first systematic review of TF alignment methods, categorizing them by stages of **pre-decoding**, **in-decoding**, and **post-decoding**. For each stage, we provide a detailed examination from the viewpoint of LLMs and multimodal LLMs (MLLMs), highlighting their mechanisms and limitations. Furthermore, we identify key challenges and future directions, paving the way for more inclusive and effective TF alignment techniques. By synthesizing and organizing the rapidly growing body of research, this survey offers a guidance for practitioners and advances the development of safer and more reliable LLMs.

</details>

---

## 289. CIVET: Systematic Evaluation of Understanding inVLMs

- [ ] CIVET: Systematic Evaluation of Understanding inVLMs | https://aclanthology.org/2025.findings-emnlp.239/

- **Link**: https://aclanthology.org/2025.findings-emnlp.239/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Vision-Language Models (VLMs) have achieved competitive performance in various tasks, their comprehension of the underlying structure and semantics of a scene remains understudied. To investigate the understanding of VLMs, we study their capability regarding object properties and relations in a controlled and interpretable manner. To this scope, we introduce CIVET, a novel and extensible framework for systematiCevaluatIonVia controllEd sTimuli. CIVET addresses the lack of standardized systematic evaluation for assessing VLMs’ understanding, enabling researchers to test hypotheses with statistical rigor. With CIVET, we evaluate five state-of-the-art VLMs on exhaustive sets of stimuli, free from annotation noise, dataset-specific biases, and uncontrolled scene complexity. Our findings reveal that 1) current VLMs can accurately recognize only a limited set of basic object properties; 2) their performance heavily depends on the position of the object in the scene; 3) they struggle to understand basic relations among objects. Furthermore, a comparative evaluation with human annotators reveals that VLMs still fall short of achieving human-level accuracy.

</details>

---

## 290. Can MultimodalLLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization

- [ ] Can MultimodalLLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization | https://aclanthology.org/2025.findings-emnlp.235/

- **Link**: https://aclanthology.org/2025.findings-emnlp.235/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Materials characterization is fundamental to acquiring materials information, revealing the processing-microstructure-property relationships that guide material design and optimization. While multimodal large language models (MLLMs) have recently shown promise in generative and predictive tasks within materials science, their capacity to understand real-world characterization imaging data remains underexplored. To bridge this gap, we presentMatCha, the first benchmark for materials characterization image understanding, comprising 1,500 questions that demand expert-level domain expertise. MatCha encompasses four key stages of materials research comprising 21 distinct tasks, each designed to reflect authentic challenges faced by materials scientists. Our evaluation of state-of-the-art MLLMs on MatCha reveals a significant performance gap compared to human experts. These models exhibit degradation when addressing questions requiring higher-level expertise and sophisticated visual perception. Simple few-shot and chain-of-thought prompting struggle to alleviate these limitations. These findings highlight that existing MLLMs still exhibit limited adaptability to real-world materials characterization scenarios. We hope MatCha will facilitate future research in areas such as new material discovery and autonomous scientific agents. MatCha is available at https://github.com/FreedomIntelligence/MatCha.

</details>

---

## 291. End-to-End Optimization for Multimodal Retrieval-Augmented Generation via Reward Backpropagation

- [ ] End-to-End Optimization for Multimodal Retrieval-Augmented Generation via Reward Backpropagation | https://aclanthology.org/2025.findings-emnlp.24/

- **Link**: https://aclanthology.org/2025.findings-emnlp.24/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Retrieval-Augmented Generation (MM-RAG) has emerged as a promising approach for enhancing the reliability and factuality of large vision-language models (LVLMs). While end-to-end loss backpropagation is infeasible due to non-differentiable operations during the forward process, current methods primarily focus on component-level optimizations, necessitate extensive component-specific training datasets and suffer from a gap between local and global optimization objectives. In this paper, we propose a new paradigm that backpropagates global rewards from the system output to each component and then transforms these rewards into specific local losses, enabling each component to perform gradient descent and thus ensuring end-to-end optimization. Specifically, we first insert two lightweight multimodal components, a query translator and an adaptive reranker, to address the heterogeneity of multimodal knowledge and the varying knowledge demands for different questions, and then tune only these inserted components using our proposed paradigm to integrate the entire system. Our method achieves SOTA performance on multiple knowledge-intensive multimodal benchmarks with high training efficiency, relying exclusively on supervised signals from an external reward model. Experimental results and our detailed analysis of the evolution of components during training collectively reveal the advantages and considerable potential of this paradigm as a promising direction for MM-RAG research.

</details>

---

## 292. FakeSV-VLM: TamingVLMfor Detecting Fake Short-Video News via Progressive Mixture-Of-Experts Adapter

- [ ] FakeSV-VLM: TamingVLMfor Detecting Fake Short-Video News via Progressive Mixture-Of-Experts Adapter | https://aclanthology.org/2025.findings-emnlp.257/

- **Link**: https://aclanthology.org/2025.findings-emnlp.257/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present FakeSV-VLM in this paper, a new VLM-based framework for detecting fake news on short video platforms. Despite significant efforts to combat this issue due to the severe threat that fake news videos pose to public information security, existing methods still fall short in detection accuracy, often due to lack of knowledge to verify the news is real or not. However, large Vision Language Models (VLMs) have absorbed extensive real-world knowledge from massive multimodal datasets. Motivated by this, we adapt advanced VLMs for fake news detection in short videos. Upon close examination of news samples, we observe that short video samples can be categorized into four distinct scenarios: both video and text are real (for real samples), or both are fake, or either the video or text is fake (for fake samples). Inspired by this insight, we design four experts tailored to handle each scenario and integrate them into VLM via Mixture of Experts. Specifically, we develop the Progressive MoE Adapter (PMOE) module where detection experts first provide an initial analysis, followed by attribution experts for a comprehensive diagnosis, leading to a robust decision. Additionally, we also note the fake news videos often show inconsistency between two modalities. Consequently, we further design the Alignment-driven Event Checking (ADEC) module, which perceives the fake news by capturing the inconsistency between different modalities. Extensive experiments on two benchmark datasets, FakeSV and FakeTT, verify the superiority of our model. It significantly outperforms current state-of-the-art models by +3.32% and +5.02%, establishing a new benchmark in the field.

</details>

---

## 293. ProPy: Building Interactive Prompt Pyramids uponCLIPfor Partially Relevant Video Retrieval

- [ ] ProPy: Building Interactive Prompt Pyramids uponCLIPfor Partially Relevant Video Retrieval | https://aclanthology.org/2025.findings-emnlp.28/

- **Link**: https://aclanthology.org/2025.findings-emnlp.28/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Partially Relevant Video Retrieval (PRVR) is a practical yet challenging task that involves retrieving videos based on queries relevant to only specific segments. While existing works follow the paradigm of developing models to process unimodal features, powerful pretrained vision-language models like CLIP remain underexplored in this field. To bridge this gap, we propose ProPy, a model with systematic architectural adaption of CLIP specifically designed for PRVR. Drawing insights from the semantic relevance of multi-granularity events, ProPy introduces two key innovations: (1) A Prompt Pyramid, a hierarchical structure that organizes event prompts to capture semantics at multiple granularity levels, and (2) An Ancestor-Descendant Interaction Mechanism built on the pyramid that enables dynamic semantic interaction among events. With these designs, ProPy achieves SOTA performance on three public datasets, outperforming previous models by significant margins. We will release all code and checkpoints to facilitate further research.

</details>

---

## 294. What Media Frames Reveal About Stance: A Dataset and Study about Memes in Climate Change Discourse

- [ ] What Media Frames Reveal About Stance: A Dataset and Study about Memes in Climate Change Discourse | https://aclanthology.org/2025.findings-emnlp.286/

- **Link**: https://aclanthology.org/2025.findings-emnlp.286/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Media framing refers to the emphasis on specific aspects of perceived reality to shape how an issue is defined and understood. Its primary purpose is to shape public perceptions often in alignment with the authors’ opinions and stances. However, the interaction between stance and media frame remains largely unexplored. In this work, we apply an interdisciplinary approach to conceptualize and computationally explore this interaction with internet memes on climate change. We curate CLIMATEMEMES, the first dataset of climate-change memes annotated with both stance and media frames, inspired by research in communication science. CLIMATEMEMES includes 1,184 memes sourced from 47 subreddits, enabling analysis of frame prominence over time and communities, and sheds light on the framing preferences of different stance holders. We propose two meme understanding tasks: stance detection and media frame detection. We evaluate LLaVA-NeXT and Molmo in various setups, and report the corresponding results on their LLM backbone. Human captions consistently enhance performance. Synthetic captions and human-corrected OCR also help occasionally. Our findings highlight that VLMs perform well on stance, but struggle on frames, where LLMs outperform VLMs. Finally, we analyze VLMs’ limitations in handling nuanced frames and stance expressions on climate change internet memes.

</details>

---

## 295. MCiteBench: A Multimodal Benchmark for Generating Text with Citations

- [ ] MCiteBench: A Multimodal Benchmark for Generating Text with Citations | https://aclanthology.org/2025.findings-emnlp.318/

- **Link**: https://aclanthology.org/2025.findings-emnlp.318/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have advanced in integrating diverse modalities but frequently suffer from hallucination. A promising solution to mitigate this issue is to generate text with citations, providing a transparent chain for verification. However, existing work primarily focuses on generating citations for text-only content, leaving the challenges of multimodal scenarios largely unexplored. In this paper, we introduce MCiteBench, the first benchmark designed to assess the ability of MLLMs to generate text with citations in multimodal contexts. Our benchmark comprises data derived from academic papers and review-rebuttal interactions, featuring diverse information sources and multimodal content. Experimental results reveal that MLLMs struggle to ground their outputs reliably when handling multimodal input. Further analysis uncovers a systematic modality bias and reveals how models internally rely on different sources when generating citations, offering insights into model behavior and guiding future directions for multimodal citation tasks.

</details>

---

## 296. Beyond Single Frames: CanLMMs Comprehend Implicit Narratives in Comic Strip?

- [ ] Beyond Single Frames: CanLMMs Comprehend Implicit Narratives in Comic Strip? | https://aclanthology.org/2025.findings-emnlp.342/

- **Link**: https://aclanthology.org/2025.findings-emnlp.342/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs) have demonstrated strong performance on vision-language benchmarks, yet current evaluations predominantly focus on single-image reasoning. In contrast, real-world scenarios always involve understanding sequences of images. A typical scenario is comic strips understanding, which requires models to perform nuanced visual reasoning beyond surface-level recognition. To address this gap, we introduce STRIPCIPHER , a benchmark designed to evaluate the model ability on understanding implicit narratives in silent comics. STRIPCIPHER is a high-quality, human-annotated dataset featuring fine-grained annotations and comprehensive coverage of varying difficulty levels. It comprises three tasks: visual narrative comprehension, contextual frame prediction, and temporal narrative reordering. % , covering various difficulty. Notably, evaluation results on STRIPCIPHER reveals a significant gap between current LMMs and human performance—e.g., GPT-4o achieves only 23.93% accuracy in the reordering task, 56.07% below human levels. These findings underscore the limitations of current LMMs in implicit visual narrative understanding and highlight opportunities for advancing sequential multimodal reasoning.

</details>

---

## 297. Bridging the Dynamic Perception Gap: Training-Free Draft Chain-of-Thought for Dynamic Multimodal Spatial Reasoning

- [ ] Bridging the Dynamic Perception Gap: Training-Free Draft Chain-of-Thought for Dynamic Multimodal Spatial Reasoning | https://aclanthology.org/2025.findings-emnlp.349/

- **Link**: https://aclanthology.org/2025.findings-emnlp.349/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While chains-of-thought (CoT) have advanced complex reasoning in multimodal large language models (MLLMs), existing methods remain confined to text or static visual domains, often faltering in dynamic spatial reasoning tasks. To bridge this gap, we present GRASSLAND, a novel maze navigation benchmark designed to evaluate dynamic spatial reasoning. Our experiments show that augmenting textual reasoning chains with dynamic visual drafts, overlaid on input images, significantly outperforms conventional approaches, offering new insights into spatial reasoning in evolving environments. To generalize this capability, we propose D2R (Dynamic Draft-Augmented Reasoning), a training-free framework that seamlessly integrates textual CoT with corresponding visual drafts into MLLMs. Extensive evaluations demonstrate that D2R consistently enhances performance across diverse tasks, establishing a robust baseline for dynamic spatial reasoning without requiring model fine-tuning.

</details>

---

## 298. Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in MultimodalLLMs

- [ ] Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in MultimodalLLMs | https://aclanthology.org/2025.findings-emnlp.372/

- **Link**: https://aclanthology.org/2025.findings-emnlp.372/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although multimodal large language models (MLLMs) have achieved impressive performance, the multimodal instruction tuning stage often causes catastrophic forgetting of the base LLM’s language ability, even in strong models like Llama3. To address this, we propose Locate-then-Merge, a training-free parameter fusion framework that first locates important parameters and then selectively merges them. We further introduce Neuron-Fusion, a neuron-level strategy that preserves the influence of neurons with large parameter shifts—neurons likely responsible for newly acquired visual capabilities—while attenuating the influence of neurons with smaller changes that likely encode general-purpose language skills. This design enables better retention of visual adaptation while mitigating language degradation. Experiments on 13 benchmarks across both language and visual tasks show that Neuron-Fusion consistently outperforms existing model merging methods. Further analysis reveals that our method effectively reduces context hallucination in generation.

</details>

---

## 299. Seeing Race, Feeling Bias: Emotion Stereotyping in Multimodal Language Models

- [ ] Seeing Race, Feeling Bias: Emotion Stereotyping in Multimodal Language Models | https://aclanthology.org/2025.findings-emnlp.386/

- **Link**: https://aclanthology.org/2025.findings-emnlp.386/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) are increasingly used to predict human emotions, but previous studies show that these models reproduce gendered emotion stereotypes. Emotion stereotypes are also tightly tied to race and skin tone (consider for example the trope of the angry black woman), but previous work has thus far overlooked this dimension. In this paper, we address this gap by introducing the first large-scale multimodal study of racial, gender, and skin-tone bias in emotion attribution, revealing how modality (text, images) and their combination shape emotion stereotypes in Multimodal LLMs (MLLMs). We evaluate four open-source MLLMs using 2.1K emotion-related events paired with 400 neutral face images across three different prompt strategies. Our findings reveal varying biases in MLLMs representations of different racial groups: models reproduce racial stereotypes across modalities, with textual cues being particularly noticeable. Models also reproduce colourist trends, with darker skin tones showing more skew. Our research highlights the need for future rigorous evaluation and mitigation strategies that account for race, colorism, and gender in MLLMs.

</details>

---

## 300. AdaptMerge: Inference Time Adaptive Visual and Language-Guided Token Merging for Efficient Large Multimodal Models

- [ ] AdaptMerge: Inference Time Adaptive Visual and Language-Guided Token Merging for Efficient Large Multimodal Models | https://aclanthology.org/2025.findings-emnlp.387/

- **Link**: https://aclanthology.org/2025.findings-emnlp.387/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Large Multimodal Models (LMMs) have showcased impressive visual understanding and vision-language reasoning capabilities, yet their computational cost hinders practical deployment, especially in resource-constrained settings. A key bottleneck is the large number of visual tokens generated by its vision encoders, which increases latency and memory demands. Existing token reduction methods often require costly fine-tuning or apply fixed token reduction ratios, ignoring image complexity and vision-language interactions. We propose AdaptMerge, a training-free, inference-time token merging strategy that adaptively reduces visual tokens by leveraging feature diversity and language-guided relevance. By dynamically adjusting to image complexity and ensuring multimodal coherence, AdaptMerge significantly lowers floating-point operations while improving performance. Extensive experiments on Google’s latest Gemma 3 models (4B and 12B parameters) across four challenging benchmarks demonstrate that AdaptMerge outperforms state-of-the-art token reduction techniques, achieving both reduced computational costs and improved performance, thereby providing a practical pathway to more efficient LMMs.

</details>

---

## 301. Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities inMLLM-Powered MobileGUIAgents

- [ ] Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities inMLLM-Powered MobileGUIAgents | https://aclanthology.org/2025.findings-emnlp.411/

- **Link**: https://aclanthology.org/2025.findings-emnlp.411/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior, enhancing effectiveness and utility. Extensive results show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7% on three attack objectives, and shows stealthiness with only 1% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1%.

</details>

---

## 302. GeoDANO: GeometricVLMwith Domain Agnostic Vision Encoder

- [ ] GeoDANO: GeometricVLMwith Domain Agnostic Vision Encoder | https://aclanthology.org/2025.findings-emnlp.414/

- **Link**: https://aclanthology.org/2025.findings-emnlp.414/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce GeoDANO, a geometric vision-language model (VLM) with a domain-agnostic vision encoder, for solving plane geometry problems. Although VLMs have been employed for solving geometry problems, their ability to recognize geometric features remains insufficiently analyzed. To address this gap, we propose a benchmark that evaluates the recognition of visual geometric features, including primitives such as dots and lines, and relations such as orthogonality. Our preliminary study shows that vision encoders often used in general-purpose VLMs, e.g., OpenCLIP, fail to detect these features and struggle to generalize across domains. To overcome the limitation, we develop GeoCLIP, a CLIP-based model trained on synthetic geometric diagram–caption pairs. Benchmark results show that GeoCLIP outperforms existing vision encoders in recognizing geometric features. We then propose our VLM, GeoDANO, which augments GeoCLIP with a domain adaptation strategy for unseen diagram styles. GeoDANO outperforms specialized methods for plane geometry problems and GPT-4o on MathVerse. The implementation is available at https://github.com/ml-postech/GeoDANO.

</details>

---

## 303. FairCoT: Enhancing Fairness in Text-to-Image Generation via Chain of Thought Reasoning with Multimodal Large Language Models

- [ ] FairCoT: Enhancing Fairness in Text-to-Image Generation via Chain of Thought Reasoning with Multimodal Large Language Models | https://aclanthology.org/2025.findings-emnlp.42/

- **Link**: https://aclanthology.org/2025.findings-emnlp.42/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the domain of text-to-image generative models, biases inherent in training datasets often propagate into generated content, posing significant ethical challenges, particularly in socially sensitive contexts. We introduce FairCoT, a novel framework that enhances fairness in text-to-image models through Chain-of-Thought (CoT) reasoning within multimodal generative large language models. FairCoT employs iterative CoT refinement to systematically mitigate biases, and dynamically adjusts textual prompts in real time, ensuring diverse and equitable representation in generated images. By integrating iterative reasoning processes, FairCoT addresses the limitations of zero-shot CoT in sensitive scenarios, balancing creativity with ethical responsibility. Experimental evaluations across popular text-to-image systems—including DALL-E and various Stable Diffusion variants—demonstrate that FairCoT significantly enhances fairness and diversity without sacrificing image quality or semantic fidelity. By combining robust reasoning, lightweight deployment, and extensibility to multiple models, FairCoT represents a promising step toward more socially responsible and transparent AI-driven content generation.

</details>

---

## 304. VLMIs a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training

- [ ] VLMIs a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training | https://aclanthology.org/2025.findings-emnlp.432/

- **Link**: https://aclanthology.org/2025.findings-emnlp.432/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language Models (VLMs) have demonstrated remarkable capabilities in processing and generating content across multiple data modalities. However, a significant drawback of VLMs is their reliance on static training data, leading to outdated information and limited contextual awareness. This static nature hampers their ability to provide accurate and up-to-date responses, particularly in dynamic or rapidly evolving contexts. To address these limitations, we propose RagVL, a novel framework with knowledge-enhanced reranking and noise-injected training. We instruction-tune the VLM with a simple yet effective instruction template to induce its ranking ability and serve it as a reranker to precisely filter the top-k retrieved images. For generation, we inject visual noise during training at the data and token levels to enhance the generator’s robustness. Extensive experiments on four datasets verify the effectiveness of our method. Code and models are available at https://anonymous.4open.science/r/RagVL-F694.

</details>

---

## 305. Improving Alignment inLVLMs with Debiased Self-Judgment

- [ ] Improving Alignment inLVLMs with Debiased Self-Judgment | https://aclanthology.org/2025.findings-emnlp.436/

- **Link**: https://aclanthology.org/2025.findings-emnlp.436/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancements in Large Language Models (LLMs) and Large Visual-Language Models (LVLMs) have opened up new opportunities for integrating visual and linguistic modalities. Yet, challenges remain in aligning these modalities effectively, causing issues such as hallucinations, where generated outputs are not grounded in the visual input, and safety concerns in the application of LVLMs across various domains. Existing alignment methods, such as instruction tuning and preference tuning, often rely on external datasets, human annotations, or complex post-processing, which limit scalability and introduce additional costs. To address these challenges, we propose a novel approach that generates the debiased self-judgment score, a self-evaluation metric created internally by the model without relying on external resources. This enables the model to autonomously improve alignment. Our method enhances both decoding strategies and preference tuning processes, resulting in improved alignment, reduced hallucinations, and enhanced safety. Empirical results show that our approach significantly outperforms traditional methods, offering a more effective solution for aligning LVLMs.

</details>

---

## 306. Watermarking for Factuality: Guiding Vision-Language Models Toward Truth via Tri-layer Contrastive Decoding

- [ ] Watermarking for Factuality: Guiding Vision-Language Models Toward Truth via Tri-layer Contrastive Decoding | https://aclanthology.org/2025.findings-emnlp.444/

- **Link**: https://aclanthology.org/2025.findings-emnlp.444/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have recently shown promising results on various multimodal tasks, even achieving human-comparable performance in certain cases. Nevertheless, LVLMs remain prone to hallucinations–they often rely heavily on a single modality or memorize training data without properly grounding their outputs. To address this, we propose a training-free, tri-layer contrastive decoding with watermarking, which proceeds in three steps: (1) select a mature layer and an amateur layer among the decoding layers, (2) identify a pivot layer using a watermark-related question to assess whether the layer is visually well-grounded, and (3) apply tri-layer contrastive decoding to generate the final output. Experiments on public benchmarks such as POPE, MME and AMBER demonstrate that our method achieves state-of-the-art performance in reducing hallucinations in LVLMs and generates more visually grounded responses.

</details>

---

## 307. DAPE-BR: Distance-Aware Positional Encoding for Mitigating Object Hallucination inLVLMs

- [ ] DAPE-BR: Distance-Aware Positional Encoding for Mitigating Object Hallucination inLVLMs | https://aclanthology.org/2025.findings-emnlp.459/

- **Link**: https://aclanthology.org/2025.findings-emnlp.459/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision–Language Models (LVLMs) have garnered substantial interest owing to their impressive ability to interpret visual inputs and converse with users.Nevertheless, LVLMs still suffer from object hallucination – generating descriptions for objects that are absent from the image, which undermines reliability and hinders real-world deployment. We propose DAPE-BR, a positional-alignment scheme that (i) preserves the pretrained weight order while globally—- visual–text distances, (ii) embeds an isotropic fused patch-distance metric, and (iii) applies a patch-distance causal mask to enforce spatial causality. Extensive experiments on POPE, MMStar and SQA show that DAPE-BR consistently reduces hallucinations and boosts.

</details>

---

## 308. Unveiling Multimodal Processing: Exploring Activation Patterns in MultimodalLLMs for Interpretability and Efficiency

- [ ] Unveiling Multimodal Processing: Exploring Activation Patterns in MultimodalLLMs for Interpretability and Efficiency | https://aclanthology.org/2025.findings-emnlp.478/

- **Link**: https://aclanthology.org/2025.findings-emnlp.478/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent Multimodal Large Language Models (MLLMs) have achieved remarkable advancements, yet their internal mechanisms for concurrently processing diverse modalities like text, image, and audio remain largely opaque. In this paper, we propose a methodology to convert dense MLLMs into fine-grained Mixture-of-Experts (MoE) architectures. This allows us to visually investigate their multimodal activation patterns through expert activation frequency heatmaps. Conducting comprehensive experiments on representative MLLMs, we analyze the similarities and differences in internal neuron activations when handling distinct modalities. Specifically, we examine the distribution of high-frequency activated experts, the distinct roles of high-frequency (e.g., fundamental logic) and low-frequency (e.g., domain-specific concepts) multimodal shared experts, and the prevalence and localization of modality-specific experts. Furthermore, we explore leveraging these discovered activation discrepancies to guide sparse activation and model pruning. Experimental results demonstrate that our approach substantially outperforms random expert pruning and can achieve comparable or even superior performance to the original unpruned models while utilizing significantly fewer active parameters. Our work not only sheds light on the multimodal processing mechanisms within MLLMs but also provides a practical pathway toward developing more interpretable and efficient multimodal systems.

</details>

---

## 309. INREACT: An Inspire-Then-Reinforce Training Framework For MultimodalGUIAgent

- [ ] INREACT: An Inspire-Then-Reinforce Training Framework For MultimodalGUIAgent | https://aclanthology.org/2025.findings-emnlp.486/

- **Link**: https://aclanthology.org/2025.findings-emnlp.486/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interface (GUI) interaction, which aims to develop an intelligent GUI agent that executes user instructions to perform tasks such as installing applications by controlling digital devices, has gained significant attention due to its practical value. Although current advanced multimodal large language models (LLMs) provide GUI agents with robust perception and reasoning capabilities, they often struggle with the precise localization of small elements. To tackle this problem, we propose InReAct, a multimodal GUI agent framework that unifies observing, thinking, and acting for precise and interpretable decision-making. It is trained via a two-stage process: curriculum learning to progressively build perception, grounding, and reasoning abilities, followed by reinforcement learning to refine pixel-level grounding with an outcome-based reward. We introduce a rule-based reward function that jointly optimizes action-type selection and pixel-level localization accuracy. Experimental results on multiple datasets demonstrate the superiority of InReAct in both grounding and navigation tasks.

</details>

---

## 310. Extracting Conceptual Spaces fromLLMs Using Prototype Embeddings

- [ ] Extracting Conceptual Spaces fromLLMs Using Prototype Embeddings | https://aclanthology.org/2025.findings-emnlp.493/

- **Link**: https://aclanthology.org/2025.findings-emnlp.493/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Conceptual spaces represent entities and concepts using cognitively meaningful dimensions, typically referring to perceptual features. Such representations are widely used in cognitive science and have the potential to serve as a cornerstone for explainable AI. Unfortunately, they have proven notoriously difficult to learn, although recent LLMs appear to capture the required perceptual features to a remarkable extent. Nonetheless, practical methods for extracting the corresponding conceptual spaces are currently still lacking. While various methods exist for extracting embeddings from LLMs, extracting conceptual spaces also requires us to encode the underlying features. In this paper, we propose a strategy in which features (e.g. sweetness) are encoded by embedding the description of a corresponding prototype (e.g. a very sweet food). To improve this strategy, we fine-tune the LLM to align the prototype embeddings with the corresponding conceptual space dimensions. Our empirical analysis finds this approach to be highly effective.

</details>

---

## 311. FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts

- [ ] FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts | https://aclanthology.org/2025.findings-emnlp.494/

- **Link**: https://aclanthology.org/2025.findings-emnlp.494/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have become powerful and widely adopted in some practical applications.However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most MLLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks.In our work, we discover that by using flowcharts with partially harmful information, MLLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack.Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets.The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts.These flowcharts are then combined with a benign textual prompt to execute the jailbreak attack on MLLMs.Our evaluations on Advbench show that FC-Attack attains an attack success rate of up to 96% via images and up to 78% via videos across multiple MLLMs.Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. We also find that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style.To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.

</details>

---

## 312. DongbaMIE: A Multimodal Information Extraction Dataset for Evaluating Semantic Understanding of Dongba Pictograms

- [ ] DongbaMIE: A Multimodal Information Extraction Dataset for Evaluating Semantic Understanding of Dongba Pictograms | https://aclanthology.org/2025.findings-emnlp.51/

- **Link**: https://aclanthology.org/2025.findings-emnlp.51/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Dongba pictographic is the only pictographic script still in use in the world. Its pictorial ideographic features carry rich cultural and contextual information. However, due to the lack of relevant datasets, research on semantic understanding of Dongba hieroglyphs has progressed slowly. To this end, we constructed DongbaMIE - the first dataset focusing on multimodal information extraction of Dongba pictographs. The dataset consists of images of Dongba hieroglyphic characters and their corresponding semantic annotations in Chinese. It contains 23,530 sentence-level and 2,539 paragraph-level high-quality text-image pairs. The annotations cover four semantic dimensions: object, action, relation and attribute. Systematic evaluation of mainstream multimodal large language models shows that the models are difficult to perform information extraction of Dongba hieroglyphs efficiently under zero-shot and few-shot learning. Although supervised fine-tuning can improve the performance, accurate extraction of complex semantics is still a great challenge at present.

</details>

---

## 313. Leveraging the Cross-Domain & Cross-Linguistic Corpus for Low ResourceNMT: A Case Study OnBhili-Hindi-English Parallel Corpus

- [ ] Leveraging the Cross-Domain & Cross-Linguistic Corpus for Low ResourceNMT: A Case Study OnBhili-Hindi-English Parallel Corpus | https://aclanthology.org/2025.findings-emnlp.508/

- **Link**: https://aclanthology.org/2025.findings-emnlp.508/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The linguistic diversity of India poses significant machine translation challenges, especially for underrepresented tribal languages like Bhili, which lack high-quality linguistic resources. This paper addresses the gap by introducing Bhili-Hindi-English Parallel Corpus (BHEPC), the first and largest parallel corpus worldwide comprising 110,000 meticulously curated sentences across Bhili, Hindi, and English. The corpus was created with the assistance of expert human translators. BHEPC spans critical domains such as education, administration, and news, establishing a valuable benchmark for research in low resource machine translation. To establish a comprehensive Bhili Machine Translation benchmark, we evaluated a wide range of proprietary and open-source Multilingual Large Language Models (MLLMs) on bidirectional translation tasks between English/Hindi and Bhili. Comprehensive evaluation demonstrates that the fine-tuned NLLB-200 distilled 600M variant model outperforms others, highlighting the potential of multilingual models in low resource scenarios. Furthermore, we investigated the generative translation capabilities of multilingual LLMs on BHEPC using in-context learning, assessing performance under cross-domain generalization and quantifying distributional divergence. This work bridges a critical resource gap and promotes inclusive natural language processing technologies for low-resource and marginalized languages globally.

</details>

---

## 314. DivScene: Towards Open-Vocabulary Object Navigation with Large Vision Language Models in Diverse Scenes

- [ ] DivScene: Towards Open-Vocabulary Object Navigation with Large Vision Language Models in Diverse Scenes | https://aclanthology.org/2025.findings-emnlp.513/

- **Link**: https://aclanthology.org/2025.findings-emnlp.513/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have achieved significant progress in tasks like visual question answering and document understanding. However, their potential to comprehend embodied environments and navigate within them remains underexplored. In this work, we first study the challenge of open-vocabulary object navigation by introducing DivScene, a large-scale dataset with 4,614 houses across 81 scene types and 5,707 kinds of target objects. Our dataset provides a much greater diversity of target objects and scene types than existing datasets, enabling a comprehensive task evaluation. We evaluated various methods with LVLMs and LLMs on our dataset and found that current models still fall short of open-vocab object navigation ability. Then, we fine-tuned LVLMs to predict the next action with CoT explanations. We observe that LVLM’s navigation ability can be improved substantially with only BFS-generated shortest paths without any human supervision, surpassing GPT-4o by over 20% in success rates.

</details>

---

## 315. Multifaceted Evaluation of Audio-Visual Capability forMLLMs: Effectiveness, Efficiency, Generalizability and Robustness

- [ ] Multifaceted Evaluation of Audio-Visual Capability forMLLMs: Effectiveness, Efficiency, Generalizability and Robustness | https://aclanthology.org/2025.findings-emnlp.54/

- **Link**: https://aclanthology.org/2025.findings-emnlp.54/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have recently achieved great success in processing and understanding information from diverse modalities (e.g., text, audio, and visual signals). Despite their growing popularity, there remains a lack of comprehensive evaluation measuring the audio-visual capabilities of these models, especially in diverse scenarios (e.g., distribution shifts and adversarial attacks). In this paper, we present a multifaceted evaluation of the audio-visual capability of MLLMs, focusing on four key dimensions: effectiveness, efficiency, generalizability, and robustness. Through extensive experiments, we find that MLLMs exhibit strong zero-shot and few-shot generalization abilities, enabling them to achieve great performance with limited data. However, their success relies heavily on the vision modality, which impairs performance when visual input is corrupted or missing. Additionally, while MLLMs are susceptible to adversarial samples, they demonstrate greater robustness compared to traditional models. The experimental results and our observations provide new insights into the audio-visual capabilities of MLLMs, highlighting areas for improvement and offering guidance for future research.

</details>

---

## 316. MediVLM: A Vision Language Model for Radiology Report Generation from Medical Images

- [ ] MediVLM: A Vision Language Model for Radiology Report Generation from Medical Images | https://aclanthology.org/2025.findings-emnlp.544/

- **Link**: https://aclanthology.org/2025.findings-emnlp.544/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating radiology reports from medical images has garnered sufficient attention in the research community. While existing methods have demonstrated promise, they often tend to generate reports that are factually incomplete and inconsistent, fail to focus on informative regions within an image, and impose strong annotation assumptions, such as bounding box annotations, image level annotations (which can be challenging to obtain) for model training. In this paper, we propose MediVLM, a vision language model (VLM) for radiology report generation from medical images. The proposed model consists of a pre-trained object detector to extract the salient anatomical regions from the images, an image encoder, a text encoder, a module to align the visual and text representations, a cross attention layer to fuse the two representations and finally, a transformer based decoder to generate the final report. MediVLM can generate radiology reports even when no reports are available for training; this is an extremely useful feature, as curating such reports is a labor-intensive task. Further, it computes a severity score (depicting the seriousness of a patient’s medical condition) from the generated radiology reports, which can be used to prioritize patients who need immediate medical attention. Our extensive empirical analyses on three benchmark datasets corroborate the promise and potential of our method against competing baselines. Our code is open-sourcedin our project webpage at: https://sites.google.com/view/medivlm/home

</details>

---

## 317. AdDriftBench: A Benchmark for Detecting Data Drift and Label Drift in Short Video Advertising

- [ ] AdDriftBench: A Benchmark for Detecting Data Drift and Label Drift in Short Video Advertising | https://aclanthology.org/2025.findings-emnlp.545/

- **Link**: https://aclanthology.org/2025.findings-emnlp.545/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the commercialization of short video platforms (SVPs), the demand for compliance auditing of advertising content has grown rapidly. The rise of large vision-language models (VLMs) offers new opportunities for automating ad content moderation. However, short video advertising scenarios present unique challenges due todata drift (DD)andlabel drift (LD). DD refers to rapid shifts in data distribution caused by advertisers to evade platform review mechanisms. LD arises from the evolving and increasingly standardized review guidelines of SVPs, which effectively alter the classification boundaries over time. Despite the significance of these phenomena, there is currently a lack of benchmark tools designed to evaluate model performance under such conditions. To address this gap, we proposeAdDriftBench (ADB). The ADB dataset consists of 3,480 short video ads, including 2,280 examples labeled under data drift scenarios, designed to evaluate the generalization capabilities of VLMs under rapidly shifting content distributions. An additional 1,200 examples represent label drift scenarios, aimed at assessing VLMs’ abilities in instruction following and fine-grained semantic understanding under varying auditing standards. Through extensive experiments on 16 open-source VLMs, we find that current models perform moderately in short video advertising contexts, particularly in handling fine-grained semantics and adapting to shifting instructions. Our dataset will be made publicly available.

</details>

---

## 318. ViFT: Towards Visual Instruction-Free Fine-tuning for Large Vision-Language Models

- [ ] ViFT: Towards Visual Instruction-Free Fine-tuning for Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.547/

- **Link**: https://aclanthology.org/2025.findings-emnlp.547/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual instruction tuning has become the predominant technology in eliciting the multimodal task-solving capabilities of large vision-language models (LVLMs). Despite the success, as visual instructions require images as the input, it would leave the gap in inheriting the task-solving capabilities from the backbone LLMs, and make it costly to collect a large-scale high-quality dataset. To address it, we propose ViFT, a visual instruction-free fine-tuning framework for LVLMs. In ViFT, we only require the text-only instructions and image caption data during training, to separately learn the task-solving and visual perception abilities. During inference, we extract and combine the representations of the text and image inputs, for fusing the two abilities to fulfill multimodal tasks. Experimental results demonstrate that ViFT can achieve state-of-the-art performance on several downstream benchmarks, with rather less training data. Our code and data will be publicly released.

</details>

---

## 319. Two Steps from Hell: Compositionality on ChemicalLMs

- [ ] Two Steps from Hell: Compositionality on ChemicalLMs | https://aclanthology.org/2025.findings-emnlp.55/

- **Link**: https://aclanthology.org/2025.findings-emnlp.55/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper investigates compositionality in chemical language models (ChemLLMs). We introduce STEPS, a benchmark with compositional questions that reflect intricate chemical structures and reactions, to evaluate models’ understanding of chemical language. Our approach focuses on identifying and analyzing compositional patterns within chemical data, allowing us to evaluate how well existing LLMs can handle complex queries. Experiments with state-of-the-art ChemLLMs show significant performance drops in compositional tasks, highlighting the need for models that move beyond pattern recognition. By creating and sharing this benchmark, we aim to enhance the development of more capable chemical LLMs and provide a resource for future research on compositionality in chemical understanding.

</details>

---

## 320. NLKI: A Lightweight Natural Language Knowledge Integration Framework for Improving SmallVLMs in CommonsenseVQATasks

- [ ] NLKI: A Lightweight Natural Language Knowledge Integration Framework for Improving SmallVLMs in CommonsenseVQATasks | https://aclanthology.org/2025.findings-emnlp.557/

- **Link**: https://aclanthology.org/2025.findings-emnlp.557/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Commonsense visual–question answering often hinges on knowledge that is missing from the image or the question. Small vision-language models (sVLMs) such as ViLT, VisualBERT, and FLAVA therefore lag behind their larger generative counterparts. To study the effect of careful commonsense knowledge integration on sVLMs, we present an end-to-end framework (NLKI) that (i) retrieves natural language facts, (ii) prompts an LLM to craft natural language explanations, and (iii) feeds both signals to sVLMs across two commonsense VQA datasets (CRIC, AOKVQA) and a visual-entailment dataset (e-SNLI-VE). Facts retrieved using a fine-tuned ColBERTv2 and an object information-enriched prompt yield explanations that largely cut down hallucinations while lifting the end-to-end answer accuracy by up to 7% (across three datasets), making FLAVA and other models in NLKI match or exceed medium-sized VLMs such as Qwen-2 VL-2B and SmolVLM-2.5B. As these benchmarks contain 10–25% label noise, additional finetuning using noise-robust losses (such as symmetric cross-entropy and generalised cross-entropy) adds another 2.5% in CRIC and 5.5% in AOKVQA. Our findings expose when LLM-based commonsense knowledge beats retrieval from commonsense knowledge bases, how noise-aware training stabilises small models in the context of external knowledge augmentation, and why parameter-efficient commonsense reasoning is now within reach for 250M models.

</details>

---

## 321. Both Text and Images Leaked! A Systematic Analysis of Data Contamination in MultimodalLLM

- [ ] Both Text and Images Leaked! A Systematic Analysis of Data Contamination in MultimodalLLM | https://aclanthology.org/2025.findings-emnlp.556/

- **Link**: https://aclanthology.org/2025.findings-emnlp.556/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of multimodal large language models (MLLMs) has significantly enhanced performance across benchmarks. However, data contamination — partial/entire benchmark data is included in the model’s training set — poses critical challenges for fair evaluation. Existing detection methods for unimodal large language models (LLMs) are inadequate for MLLMs due to multimodal data complexity and multi-phase training. We systematically analyze multimodal data contamination using our analytical framework, MM-DETECT, which defines two contamination categories — unimodal and cross-modal — and effectively quantifies contamination severity across multiple-choice and caption-based Visual Question Answering tasks. Evaluations on twelve MLLMs and five benchmarks reveal significant contamination, particularly in proprietary models and older benchmarks. Crucially, contamination sometimes originates during unimodal pre-training rather than solely from multimodal fine-tuning. Our insights refine contamination understanding, guiding evaluation practices and improving multimodal model reliability.

</details>

---

## 322. DesignCLIP: Multimodal Learning withCLIPfor Design Patent Understanding

- [ ] DesignCLIP: Multimodal Learning withCLIPfor Design Patent Understanding | https://aclanthology.org/2025.findings-emnlp.553/

- **Link**: https://aclanthology.org/2025.findings-emnlp.553/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the field of design patent analysis, traditional tasks such as patent classification and patent image retrieval heavily depend on the image data. However, patent images—typically consisting of sketches with abstract and structural elements of an invention—often fall short in conveying comprehensive visual context and semantic information. This inadequacy can lead to ambiguities in evaluation during prior art searches. Recent advancements in vision-language models, such as CLIP, offer promising opportunities for more reliable and accurate AI-driven patent analysis. In this work, we leverage CLIP models to develop a unified framework DesignCLIP for design patent applications with a large-scale dataset of U.S. design patents. To address the unique characteristics of patent data, DesignCLIP incorporates class-aware classification and contrastive learning, utilizing generated detailed captions for patent images and multi-views image learning. We validate the effectiveness of DesignCLIP across various downstream tasks, including patent classification and patent retrieval. Additionally, we explore multimodal patent retrieval, which provides the potential to enhance creativity and innovation in design by offering more diverse sources of inspiration. Our experiments show that DesignCLIP consistently outperforms baseline and SOTA models in the patent domain on all tasks. Our findings underscore the promise of multimodal approaches in advancing patent analysis. The codebase is available here: https://github.com/AI4Patents/DesignCLIP.

</details>

---

## 323. 3D-Aware Vision-Language Models Fine-Tuning with Geometric Distillation

- [ ] 3D-Aware Vision-Language Models Fine-Tuning with Geometric Distillation | https://aclanthology.org/2025.findings-emnlp.562/

- **Link**: https://aclanthology.org/2025.findings-emnlp.562/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have shown remarkable performance on diverse visual and linguistic tasks, yet they remain fundamentally limited in their understanding of 3D spatial structures.We propose Geometric Distillation, a lightweight, annotation-free fine-tuning framework that injects human-inspired geometric cues into pretrained VLMs without modifying their architecture.By distilling (1) sparse correspondences, (2) relative depth relations, and (3) dense cost volumes from off-the-shelf 3D foundation models (e.g., MASt3R, VGGT), our method shapes representations to be geometry-aware while remaining compatible with natural image–text inputs.Through extensive evaluations on 3D vision-language reasoning and 3D perception benchmarks, our method consistently outperforms prior approaches, achieving improved 3D spatial reasoning with significantly lower computational cost.Our work demonstrates a scalable and efficient path to bridge 2D-trained VLMs with 3D understanding, opening up wider use in spatially grounded multimodal tasks.

</details>

---

## 324. AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving

- [ ] AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving | https://aclanthology.org/2025.findings-emnlp.564/

- **Link**: https://aclanthology.org/2025.findings-emnlp.564/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduceAgentThink, a pioneering unified framework that, for the first time, integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink’s core innovations include:(i) Structured Data Generation, by establishing an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios;(ii) A Two-stage Training Pipeline, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and(iii) Agent-style Tool-Usage Evaluation, introducing a novel multi-tool assessment protocol to rigorously evaluate the model’s tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate AgentThink significantly boosts overall reasoning scores by53.91%and enhances answer accuracy by33.54%, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models.

</details>

---

## 325. SteeringLVLMs via Sparse Autoencoder for Hallucination Mitigation

- [ ] SteeringLVLMs via Sparse Autoencoder for Hallucination Mitigation | https://aclanthology.org/2025.findings-emnlp.572/

- **Link**: https://aclanthology.org/2025.findings-emnlp.572/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have achieved remarkable performance on multimodal tasks. However, they still suffer from hallucinations, generating text inconsistent with visual input, posing significant risks in real-world applications. Existing approaches to address this issue focus on incorporating external knowledge bases, alignment training, or decoding strategies, all of which require substantial computational cost and time. Recent works try to explore more efficient alternatives by adjusting LVLMs’ internal representations. Although promising, these methods may cause hallucinations to be insufficiently suppressed or lead to excessive interventions that negatively affect normal semantics. In this work, we leverage sparse autoencoders (SAEs) to identify semantic directions closely associated with faithfulness or hallucination, extracting more precise and disentangled hallucination-related representations. Our analysis demonstrates that interventions along the identified faithful direction can mitigate hallucinations, while those along the hallucinatory direction can exacerbate them. Building on these insights, we propose **S**teering LVLMs via **S**AE **L**atent Directions (SSL), a plug-and-play method based on SAE-derived latent directions to mitigate hallucinations in LVLMs. Extensive experiments demonstrate that SSL significantly outperforms existing decoding approaches in mitigating hallucinations, while maintaining transferability across different model architectures with negligible additional time overhead. The code is available at [https://github.com/huazhenglin2003/SSL](https://github.com/huazhenglin2003/SSL).

</details>

---

## 326. On the Perception Bottleneck ofVLMs for Chart Understanding

- [ ] On the Perception Bottleneck ofVLMs for Chart Understanding | https://aclanthology.org/2025.findings-emnlp.573/

- **Link**: https://aclanthology.org/2025.findings-emnlp.573/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chart understanding requires models to effectively analyze and reason about numerical data, textual elements, and complex visual components. Our observations reveal that the perception capabilities of existing large vision-language models (LVLMs) constitute a critical bottleneck in this process. In this study, we delve into this perception bottleneck by decomposing it into two components: the vision encoder bottleneck, where the visual representation may fail to encapsulate the correct information, and the extraction bottleneck, where the language model struggles to extract the necessary information from the provided visual representations. Through comprehensive experiments, we find that (1) the information embedded within visual representations is substantially richer than what is typically captured by linear extractors, such as the widely used retrieval accuracy metric; (2) While instruction tuning effectively enhances the extraction capability of LVLMs, the vision encoder remains a critical bottleneck, demanding focused attention and improvement. Therefore, we further enhance the visual encoder to mitigate the vision encoder bottleneck under a contrastive learning framework. Empirical results demonstrate that our approach significantly mitigates the perception bottleneck and improves the ability of LVLMs to comprehend charts.

</details>

---

## 327. ARXSA: A General Negative Feedback Control Theory in Vision-Language Models

- [ ] ARXSA: A General Negative Feedback Control Theory in Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.591/

- **Link**: https://aclanthology.org/2025.findings-emnlp.591/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Transformer model has been increasingly applied across various domains, driven by the self-attention mechanism, which offers robust data processing capabilities and has substantially contributed to the advancement of the model. In the self-attention mechanism, three core matrices from the same data batch are computed together to determine correlations between input elements. Drawing inspiration from the efficiency and stability conferred by negative feedback structures in predictive control systems, the concept of vertical training was introduced to integrate data from multiple batches. Accordingly, this paper proposes an autoregressive with exogenous inputs (ARX) approach for the self-attention mechanism, transforming the Encoder block into a negative feedback predictive control system. A network architecture based on this method is also proposed, enabling the autoregressive with exogenous inputs for self-attention to transmit data from batches at previous time points. The effectiveness of the proposed approach is validated through comparative experimental evaluations.

</details>

---

## 328. Can youSPLICEit together? A Human Curated Benchmark for Probing Visual Reasoning inVLMs

- [ ] Can youSPLICEit together? A Human Curated Benchmark for Probing Visual Reasoning inVLMs | https://aclanthology.org/2025.findings-emnlp.604/

- **Link**: https://aclanthology.org/2025.findings-emnlp.604/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we introduce SPLICE, a human-curated benchmark derived from the COIN instructional video dataset, designed to probe event-based reasoning across multiple dimensions: temporal, causal, spatial, contextual, and general knowledge. SPLICE includes 3,381 human-filtered videos spanning 12 categories and 180 sub-categories, such as sports, engineering, and housework. These videos are segmented into a total of 11,423 event clips. We evaluate both human participants and state-of-the-art vision-language models (VLMs) on the task of rearranging these clips into coherent event sequences to assess visual reasoning capabilities. Results reveal a significant gap: VLMs struggle to match human performance. While human-annotated textual descriptions improve model accuracy, they do not affect human performance, suggesting that models rely more on language priors than on visual understanding. Even with annotations, VLMs fall short of human-level reasoning, underscoring persistent challenges in visual reasoning. A deeper analysis across sub-categories shows that VLMs perform relatively better on videos where temporal and causal reasoning are dominant, compared to those where contextual and spatial reasoning are dominant. They also perform better on everyday tasks than on specialized ones.

</details>

---

## 329. Making Every Step Effective: Jailbreaking Large Vision-Language Models Through HierarchicalKVEqualization

- [ ] Making Every Step Effective: Jailbreaking Large Vision-Language Models Through HierarchicalKVEqualization | https://aclanthology.org/2025.findings-emnlp.618/

- **Link**: https://aclanthology.org/2025.findings-emnlp.618/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In the realm of large vision-language models (LVLMs), adversarial jailbreak attacks serve as a red-teaming approach to identify safety vulnerabilities of these models and their associated defense mechanisms. However, we identify a critical limitation: not every adversarial optimization step leads to a positive outcome, and indiscriminately accepting optimization results at each step may reduce the overall attack success rate. To address this challenge, we introduce HKVE (Hierarchical Key-Value Equalization), an innovative jailbreaking framework that selectively accepts gradient optimization results based on the distribution of attention scores across different layers, ensuring that every optimization step positively contributes to the attack. Extensive experiments demonstrate HKVE’s significant effectiveness, achieving attack success rates of 75.08% on MiniGPT4, 85.84% on LLaVA and 81.00% on Qwen-VL, substantially outperforming existing methods by margins of 20.43%, 21.01% and 26.43% respectively. Furthermore, making every step effective not only leads to an increase in attack success rate but also allows for a reduction in the number of iterations, thereby lowering computational costs.

</details>

---

## 330. Attribution and Application of Multiple Neurons in Multimodal Large Language Models

- [ ] Attribution and Application of Multiple Neurons in Multimodal Large Language Models | https://aclanthology.org/2025.findings-emnlp.625/

- **Link**: https://aclanthology.org/2025.findings-emnlp.625/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance across various tasks. However, the internal mechanisms by which they interpret and integrate cross-modal information remain insufficiently understood. In this paper, to address the limitations of prior studies that could only identify neurons corresponding to single-token and rely on the vocabulary of LLMs, we propose a novel method to identify multimodal neurons in Transformer-based MLLMs. Then we introduce fuzzy set theory to model the complex relationship between neurons and semantic concepts and to characterize how multiple neurons collaboratively contribute to semantic concepts. Through both theoretical analysis and empirical validation, we demonstrate the effectiveness of our method and present some meaningful findings. Furthermore, by modulating neuron activation values based on the constructed fuzzy sets, we enhance performance on the Visual Question Answering (VQA) task, showing the practical value of our approach in downstream applications in MLLMs.

</details>

---

## 331. Do What? Teaching Vision-Language-Action Models to Reject the Impossible

- [ ] Do What? Teaching Vision-Language-Action Models to Reject the Impossible | https://aclanthology.org/2025.findings-emnlp.635/

- **Link**: https://aclanthology.org/2025.findings-emnlp.635/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, Vision-Language-Action (VLA) models have demonstrated strong performance on a range of robotic tasks. These models rely on multimodal inputs, with language instructions playing a crucial role-not only in predicting actions, but also in robustly interpreting user intent, even when the requests are impossible to fulfill. In this work, we investigate how VLAs can recognize, interpret, and respond to false-premise instructions-natural language commands that reference objects or conditions absent from the environment. We propose — Instruct-Verify-and-Act (IVA) — a unified framework that (i) detects when an instruction cannot be executed due to a false premise, (ii) engages in language-based clarification or correction, and (iii) grounds plausible alternatives in perception and action. Towards this end, we construct a large-scale instruction tuning setup with structured language prompts and train a VLA model capable of handling both accurate and erroneous requests. Our approach leverages a contextually augmented, semi-synthetic dataset containing paired positive and false-premise instructions, enabling robust detection and natural language correction. Our experiments show that IVA can improves false premise detection accuracy by 58.89% over baselines, while increasing successful responses in false-premise scenarios by 27.89%.

</details>

---

## 332. Exploring and Evaluating Multimodal Knowledge Reasoning Consistency of Multimodal Large Language Models

- [ ] Exploring and Evaluating Multimodal Knowledge Reasoning Consistency of Multimodal Large Language Models | https://aclanthology.org/2025.findings-emnlp.639/

- **Link**: https://aclanthology.org/2025.findings-emnlp.639/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, multimodal large language models (MLLMs) have achieved significant breakthroughs, enhancing understanding across text and vision. However, current MLLMs still face challenges in effectively integrating knowledge across these modalities during multimodal knowledge reasoning, leading to inconsistencies in reasoning outcomes. To systematically explore this issue, we propose four evaluation tasks and construct a new dataset. We conduct a series of experiments on this dataset to analyze and compare the extent of consistency degradation in multimodal knowledge reasoning within MLLMs. Based on the experimental results, we identify factors contributing to the observed degradation in consistency. Our research provides new insights into the challenges of multimodal knowledge reasoning and offers valuable guidance for future efforts aimed at improving MLLMs.

</details>

---

## 333. Mitigating Object Hallucinations inMLLMs via Multi-Frequency Perturbations

- [ ] Mitigating Object Hallucinations inMLLMs via Multi-Frequency Perturbations | https://aclanthology.org/2025.findings-emnlp.64/

- **Link**: https://aclanthology.org/2025.findings-emnlp.64/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, multimodal large language models (MLLMs) have demonstrated remarkable performance in visual-language tasks. However, the authenticity of the responses generated by MLLMs is often compromised by object hallucinations. We identify that a key cause of these hallucinations is the model’s over-susceptibility to image frequency features in detecting objects. In this paper, we introduce Multi-Frequency Perturbations (MFP), a simple, cost-effective, and pluggable adversarial training method that leverages both low-frequency and high-frequency features of images to perturb visual feature representations and explicitly suppress redundant frequency-domain features during inference, thereby mitigating hallucinations. Experimental results demonstrate that our method significantly mitigates object hallucinations across various model architectures. Furthermore, as a training-time method, MFP can be combined with inference-time methods to achieve state-of-the-art performance on the CHAIR benchmark.

</details>

---

## 334. Curr-ReFT: Overcoming Training Bottlenecks in Small-scale Vision-Language Models via Curriculum Reinforcement Finetuning

- [ ] Curr-ReFT: Overcoming Training Bottlenecks in Small-scale Vision-Language Models via Curriculum Reinforcement Finetuning | https://aclanthology.org/2025.findings-emnlp.643/

- **Link**: https://aclanthology.org/2025.findings-emnlp.643/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

State-of-the-art vision-language models (VLMs) require massive scaling that limits practical deployment. Small-scale VLMs offer a practical alternative but face out-of-domain (OOD) collapse when trained with traditional supervised fine-tuning (SFT). Through GeneralPoints experiments, we identify that OOD collapse is due to SFT’s tendency to induce visual hallucinations under distribution shifts, whereas Reinforcement Learning’s (RL) bidirectional reward-driven mechanism with iterative error correction refines visual perception. Although RL-based post-training effectively mitigates OOD degradation, it faces a critical sparse reward dilemma in complex visual reasoning tasks. To this end, we propose Curriculum Reinforcement Finetuning (Curr-ReFT), comprising two sequential stages: (1) Structured Curriculum Reinforcement Learning, which progressively evolves task formats and reward functions to match models’ growing capabilities; and (2) Rejected Sampling-based Self-improvement, which maintains the fundamental capabilities of VLMs through selective learning from high-quality examples. Extensive experiments demonstrate that Curr-ReFT achieves state-of-the-art performance across various visual tasks in both in- and out-of-domain settings and benchmarks.

</details>

---

## 335. AMIA: Automatic Masking and Joint Intention Analysis MakesLVLMs Robust Jailbreak Defenders

- [ ] AMIA: Automatic Masking and Joint Intention Analysis MakesLVLMs Robust Jailbreak Defenders | https://aclanthology.org/2025.findings-emnlp.651/

- **Link**: https://aclanthology.org/2025.findings-emnlp.651/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce AMIA, a lightweight, inference-only defense for Large Vision–Language Models (LVLMs) that (1) Automatically Masks a small set of text-irrelevant image patches to disrupt adversarial perturbations, and (2) conducts joint Intention Analysis to uncover and mitigate hidden harmful intents before response generation. Without any retraining, AMIA improves defense success rates across diverse LVLMs and jailbreak benchmarks from an average of 52.4% to 81.7%, preserves general utility with only a 2% average accuracy drop, and incurs only modest inference overhead. Ablation confirms that both masking and intention analysis are essential for robust safety–utility trade-off. Our code will be released.

</details>

---

## 336. Language-Informed Synthesis of Rational Agent Models for Grounded Theory-of-Mind Reasoning On-the-fly

- [ ] Language-Informed Synthesis of Rational Agent Models for Grounded Theory-of-Mind Reasoning On-the-fly | https://aclanthology.org/2025.findings-emnlp.654/

- **Link**: https://aclanthology.org/2025.findings-emnlp.654/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Drawing real world social inferences usually requires taking into account information from multiple modalities. Language is a particularly powerful source of information in social settings, especially in novel situations where language can provide both abstract information about the environment dynamics and concrete specifics about an agent that cannot be easily visually observed. In this paper, we propose Language-Informed Rational Agent Synthesis (LIRAS), a framework for drawing context-specific social inferences that integrate linguistic and visual inputs. LIRAS frames multimodal social reasoning as a process of constructing structured but situation-specific agent and environment representations – leveraging multimodal language models to parse language and visual inputs into unified symbolic representations, over which a Bayesian inverse planning engine can be run to produce granular probabilistic judgments. On a range of existing and new social reasoning tasks derived from cognitive science experiments, we find that our model (instantiated with a comparatively lightweight VLM) outperforms ablations and state-of-the-art models in capturing human judgments across all domains.

</details>

---

## 337. FESTA: Functionally Equivalent Sampling for Trust Assessment of MultimodalLLMs

- [ ] FESTA: Functionally Equivalent Sampling for Trust Assessment of MultimodalLLMs | https://aclanthology.org/2025.findings-emnlp.657/

- **Link**: https://aclanthology.org/2025.findings-emnlp.657/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We proposeFunctionallyEquivalentSampling forTrustAssessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.

</details>

---

## 338. MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models

- [ ] MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.656/

- **Link**: https://aclanthology.org/2025.findings-emnlp.656/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Speculative decoding significantly accelerates language model inference by enabling a lightweight draft model to propose multiple tokens that a larger target model verifies simultaneously. However, applying this technique to vision-language models (VLMs) presents two fundamental challenges: small language models that could serve as efficient drafters lack the architectural components to process visual inputs, and their token predictions fail to match those of VLM target models that consider visual context. We introduce Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (MASSV), which transforms existing small language models into effective multimodal drafters through a two-phase approach. MASSV first connects the target VLM’s vision encoder to the draft model via a lightweight trainable projector, then applies self-distilled visual instruction tuning using responses generated by the target VLM to align token predictions. Comprehensive experiments across the Qwen2.5-VL and Gemma3 model families demonstrate that MASSV increases accepted length by up to 30% and delivers end-to-end inference speedups of up to 1.46x compared to conventional text-only drafting baselines on visually-grounded tasks.

</details>

---

## 339. Language-Specific Layer Matters: Efficient Multilingual Enhancement for Large Vision-Language Models

- [ ] Language-Specific Layer Matters: Efficient Multilingual Enhancement for Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.666/

- **Link**: https://aclanthology.org/2025.findings-emnlp.666/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have demonstrated exceptional capabilities in understanding visual information with human languages but also exhibit an imbalance in multilingual capabilities. In this work, we delve into the multilingual working pattern of LVLMs and identify a salient correlation between the multilingual understanding ability of LVLMs and language-specific neuron activations in shallow layers. Building on this insight, we introduce PLAST, a training recipe that achieves efficient multilingual enhancement for LVLMs by Precise LAnguage Specific layers fine-Tuning. PLAST first identifies layers involved in multilingual understanding by monitoring language-specific neuron activations. These layers are then precisely fine-tuned with question-translation pairs to achieve multilingual alignment. Our empirical results on MMBench and MMMB demonstrate that PLAST effectively improves the multilingual capabilities of LVLMs and achieves significant efficiency with only 14% of the parameters tuned. Further analysis reveals that PLAST facilitates the language-specific visual information engagement in shallow layers.

</details>

---

## 340. From Grounding to Manipulation: Case Studies of Foundation Model Integration in Embodied Robotic Systems

- [ ] From Grounding to Manipulation: Case Studies of Foundation Model Integration in Embodied Robotic Systems | https://aclanthology.org/2025.findings-emnlp.69/

- **Link**: https://aclanthology.org/2025.findings-emnlp.69/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Foundation models (FMs) are increasingly applied to bridge language and action in embodied agents, yet the operational characteristics of different integration strategies remain under-explored—especially for complex instruction following and versatile action generation in changing environments. We investigate three paradigms for robotic systems: end-to-end vision-language-action models (VLAs) that implicitly unify perception and planning, and modular pipelines using either vision-language models (VLMs) or multimodal large language models (MLLMs). Two case studies frame the comparison: instruction grounding, which probs fine-grained language understanding and cross-modal disambiguation; and object manipulation, which targets skill transfer via VLA finetuning. Our experiments reveal trade-offs in system scale, generalization and data efficiency. These findings indicate design lessons for language-driven physical agents and point to challenges and opportunities for FM-powered robotics in real-world conditions.

</details>

---

## 341. Flexible Thinking for Multimodal Emotional Support Conversation via Reinforcement Learning

- [ ] Flexible Thinking for Multimodal Emotional Support Conversation via Reinforcement Learning | https://aclanthology.org/2025.findings-emnlp.70/

- **Link**: https://aclanthology.org/2025.findings-emnlp.70/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Emotional Support Conversation (ESC) systems aim to alleviate user distress. However, current Chain-of-Thought based ESC methods often employ rigid, text-only reasoning, limiting adaptability in dynamic, multimodal interactions and introducing reasoning noise that degrades support quality. To address this, we introduce “Flexible Thinking” for multimodal ESC, enabling models to adaptively select contextually relevant thinking aspects: Visual Scene, Emotion, Situation, and Response Strategy. We first construct training data by manually curating flexible thinking demonstrations on the MESC dataset, then using a Multimodal Large Language Model to synthesize these processes for the full training set. Then, we propose FIRES, a framework integrating Supervised Fine-Tuning (SFT) for initial learning with Reinforcement Learning for refinement. This two-stage approach helps FIRES transcend SFT’s generalization limits and, crucially, directly links thinking processes to response quality via tailored rewards, moving beyond imitating potentially imperfect synthetic data. Experiments on MESC and EMOTyDA datasets demonstrate FIRES’s effectiveness and generalizability in fostering higher-quality emotional support responses through adaptive reasoning.

</details>

---

## 342. DocMMIR: A Framework for Document Multi-modal Information Retrieval

- [ ] DocMMIR: A Framework for Document Multi-modal Information Retrieval | https://aclanthology.org/2025.findings-emnlp.705/

- **Link**: https://aclanthology.org/2025.findings-emnlp.705/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of unsupervised representation learning and large-scale pre-trained vision-language models has significantly improved cross-modal retrieval tasks. However, existing multi-modal information retrieval (MMIR) studies lack a comprehensive exploration of document-level retrieval and suffer from the absence of cross-domain datasets at this granularity. To address this limitation, we introduceDocMMIR, a novel multi-modal document retrieval framework designed explicitly to unify diverse document formats and domains—including Wikipedia articles, scientific papers (arXiv), and presentation slides—within a comprehensive retrieval scenario. We construct a large-scale cross-domain multimodal dataset, comprising450Ktraining,19.2Kvalidation, and19.2Ktest documents, serving as both a benchmark to reveal the shortcomings of existing MMIR models and a training set for further improvement. The dataset systematically integrates textual and visual information. Our comprehensive experimental analysis reveals substantial limitations in current state-of-the-art MLLMs (CLIP, BLIP2, SigLIP-2, ALIGN) when applied to our tasks, with only CLIP (ViT-L/14) demonstrating reasonable zero-shot performance. Through systematic investigation of cross-modal fusion strategies and loss function selection on the CLIP (ViT-L/14) model, we develop an optimised approach that achieves a+31%improvement in MRR@10 metrics from zero-shot baseline to fine-tuned model. Our findings offer crucial insights and practical guidance for future development in unified multimodal document retrieval tasks.

</details>

---

## 343. ChartM3: A Multi-Stage Code-Driven Pipeline for Constructing Multi-Dimensional and Multi-Step Visual Reasoning Data in Chart Comprehension

- [ ] ChartM3: A Multi-Stage Code-Driven Pipeline for Constructing Multi-Dimensional and Multi-Step Visual Reasoning Data in Chart Comprehension | https://aclanthology.org/2025.findings-emnlp.701/

- **Link**: https://aclanthology.org/2025.findings-emnlp.701/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Complex chart understanding tasks demand advanced visual recognition and reasoning capabilities from multimodal large language models (MLLMs). However, current research provides limited coverage of complex chart scenarios and computation-intensive reasoning tasks prevalent in real-world applications. This study proposes an automated multi-stage code-driven pipeline for systematically generating visual reasoning datasets to address these limitations. The pipeline integrates retrieval-augmented generation (RAG) to retrieve professional chart templates and employs chain-of-thought (CoT) strategies to generate reasoning codes that simulate real data distributions, thereby driving chart rendering and question-related statistical computations. Through model-based evaluation, the pipeline enhances chart diversity and data quality. Using this framework, we construct ChartM3, a multi-dimensional and multi-step dataset containing 38K charts and 142K Q&A pairs for training, along with 2,871 high-quality evaluation samples for enabling practical performance assessment. Supervised fine-tuning (SFT) and reinforcement learning (RL) experiments demonstrate that our dataset significantly improves reasoning capabilities and cross-domain generalization performance, enabling smaller models to achieve performance comparable to larger-scale models in complex chart comprehension.

</details>

---

## 344. A Closer Look at Bias and Chain-of-Thought Faithfulness of Large (Vision) Language Models

- [ ] A Closer Look at Bias and Chain-of-Thought Faithfulness of Large (Vision) Language Models | https://aclanthology.org/2025.findings-emnlp.723/

- **Link**: https://aclanthology.org/2025.findings-emnlp.723/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chain-of-thought (CoT) reasoning enhances performance of large language models, but questions remain about whether these reasoning traces faithfully reflect the internal processes of the model. We present the first comprehensive study of CoT faithfulness in large vision-language models (LVLMs), investigating how both text-based and previously unexplored image-based biases affect reasoning and bias articulation. Our work introduces a novel, fine-grained evaluation pipeline for categorizing bias articulation patterns, enabling significantly more precise analysis of CoT reasoning than previous methods. This framework reveals critical distinctions in how models process and respond to different types of biases, providing new insights into LVLM CoT faithfulness. Our findings reveal that subtle image-based biases are rarely articulated compared to explicit text-based ones, even in models specialized for reasoning. Additionally, many models exhibit a previously unidentified phenomenon we term “inconsistent” reasoning - correctly reasoning before abruptly changing answers, serving as a potential canary for detecting biased reasoning from unfaithful CoTs. We then apply the same evaluation pipeline to revisit CoT faithfulness in LLMs across various levels of implicit cues. Our findings reveal that current language-only reasoning models continue to struggle with articulating cues that are not overtly stated.

</details>

---

## 345. Adversary-AwareDPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training

- [ ] Adversary-AwareDPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training | https://aclanthology.org/2025.findings-emnlp.735/

- **Link**: https://aclanthology.org/2025.findings-emnlp.735/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Safety alignment is critical in pre-trained large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. Unlike LLM, the current safety alignment of VLMs is often achieved with post-hoc safety fine-tuning. However, these methods are less effective to white-box attacks. To address this, we proposeAdversary-aware DPO (ADPO), a novel training framework that explicitly considers adversary.Adversary-aware DPO (ADPO)integrates adversarial training into DPO to enhance the safety alignment of VLMs under worst-case adversarial perturbations.ADPOintroduces two key components: (1) an adversarial-trained reference model that generates human-preferred responses under worst-case perturbations, and (2) an adversary-aware DPO loss that generates winner-loser pairs accounting for adversarial distortions. By combining these innovations,ADPOensures that VLMs remain robust and reliable even in the presence of sophisticated jailbreak attacks. Extensive experiments demonstrate thatADPOoutperforms baselines in terms of both safety alignment and general utility of VLMs.

</details>

---

## 346. Mitigating Hallucinations in Large Vision-Language Models by Self-Injecting Hallucinations

- [ ] Mitigating Hallucinations in Large Vision-Language Models by Self-Injecting Hallucinations | https://aclanthology.org/2025.findings-emnlp.746/

- **Link**: https://aclanthology.org/2025.findings-emnlp.746/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) suffer from serious hallucination problems, where the model-generated responses are inconsistent with the visual inputs. Existing hallucination mitigation methods are mainly based on preference alignment and require external human annotations or auxiliary models for preference data collection, which increase costs and limit sustainable improvement. To tackle these challenges, we propose Autonomous Preference Alignment via Self-Injection (APASI), a novel and generalizable method that mitigates hallucinations without external dependencies. APASI leverages the target LVLM to self-inject hallucinations into a generated response, creating a pair of responses with varying preference levels. During the self-injection process, the dis-preferred response is generated based on three key observations of hallucinations, ensuring it simulates real hallucination patterns. This fidelity offers an accurate learning signal for hallucination mitigation. Moreover, APASI incorporates an iterative alignment training strategy combined with curriculum learning to periodically update the preference data with increasing challenge, enabling stable and continuous enhancement of the LVLM. Extensive experiments across six benchmarks show that APASI not only effectively mitigates hallucinations for three baseline models but also achieves comparable or even superior performance to alignment-based methods with external dependency, thereby demonstrating its effectiveness and generalization capability.

</details>

---

## 347. InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning

- [ ] InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning | https://aclanthology.org/2025.findings-emnlp.766/

- **Link**: https://aclanthology.org/2025.findings-emnlp.766/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-training on large, high-quality datasets is essential for improving the reasoning abilities of Large Language Models (LLMs), particularly in specialized fields like mathematics. However, the field of Multimodal LLMs (MLLMs) lacks a comprehensive, open-source dataset for mathematical reasoning. To fill this gap, we present InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents. It consists of 24 million web pages, 85 million image URLs, and 40 billion text tokens, all carefully extracted and filtered from CommonCrawl. We outline our data collection and processing pipeline in detail. Models trained on InfiMM-WebMath-40B demonstrate strong performance in both text-only and multimodal settings, setting a new state-of-the-art on multimodal math benchmarks such as MathVerse and We-Math.

</details>

---

## 348. Zero-Shot Defense Against Toxic Images via Inherent Multimodal Alignment inLVLMs

- [ ] Zero-Shot Defense Against Toxic Images via Inherent Multimodal Alignment inLVLMs | https://aclanthology.org/2025.findings-emnlp.767/

- **Link**: https://aclanthology.org/2025.findings-emnlp.767/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have made significant strides in multimodal comprehension, thanks to extensive pre-training and fine-tuning on large-scale visual datasets. However, despite their robust textual safety mechanisms, they remain vulnerable to harmful visual inputs. Existing safeguards—typically relying on pre-filtering or fine-tuning—incur high costs and diminish overall utility. To address this critical vulnerability, we introduce SafeCLIP, a lightweight method that leverages LVLMs’ inherent multimodal alignment for zero-shot toxic image detection. By projecting CLIP’s discarded CLS token into its text space and matching it with toxic descriptors, SafeCLIP detects harmful content without any architectural changes—adding minimal latency and enabling dynamic safety corrections during inference and fine-tuning. Experiments show that SafeCLIP achieves a 66.9% defense success rate with only 3.2% false positive rate and 7.2% overhead. In contrast, state-of-the-art methods achieve 52.9% success but have a 10.7% false positive rate and 210% overhead. Our work demonstrates that leveraging inherent multimodal alignment can yield efficient, low-cost LVLM safety. Code is available atanonymous.4open.science/r/safeclip-2C01.

</details>

---

## 349. BcQLM: Efficient Vision-Language Understanding with DistilledQ-Gated Cross-Modal Fusion

- [ ] BcQLM: Efficient Vision-Language Understanding with DistilledQ-Gated Cross-Modal Fusion | https://aclanthology.org/2025.findings-emnlp.780/

- **Link**: https://aclanthology.org/2025.findings-emnlp.780/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As multimodal large language models (MLLMs) advance, their large-scale architectures pose challenges for deployment in resource-constrained environments. In the age of large models, where energy efficiency, computational scalability and environmental sustainability are paramount, the development of lightweight and high-performance models is critical for real-world applications. As such, we propose a lightweight MLLM framework for end-to-end visual question answering. Our proposed approach centres on BreezeCLIP, a compact yet powerful vision-language encoder optimised for efficient multimodal understanding. With only 1.2 billion parameters overall, our model significantly reduces computational cost while achieving performance comparable to standard-size MLLMs. Experiments conducted on multiple datasets further validate its effectiveness in balancing accuracy and efficiency. The modular and extensible design enables generalisation to broader multimodal tasks. The proposed lightweight vision-language framework is denoted as BcQLM (BreezeCLIP-enhanced Q-Gated Multimodal Language Model). It offers a promising path toward deployable MLLMs under practical hardware constraints. The source code is available athttps://github.com/thico0224/BcQLM.

</details>

---

## 350. MDSEval: A Meta-Evaluation Benchmark for Multimodal Dialogue Summarization

- [ ] MDSEval: A Meta-Evaluation Benchmark for Multimodal Dialogue Summarization | https://aclanthology.org/2025.findings-emnlp.794/

- **Link**: https://aclanthology.org/2025.findings-emnlp.794/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Dialogue Summarization (MDS) is a critical task with wide-ranging applications. To support the development of effective MDS models, robust automatic evaluation methods are essential for reducing both cost and human effort. However, such methods require a strong meta-evaluation benchmark grounded in human annotations. In this work, we introduce MDSEval, the first meta-evaluation benchmark for MDS, consisting image-sharing dialogues, corresponding summaries, and human judgments across eight well-defined quality aspects. To ensure data quality and richfulness, we propose a novel filtering framework leveraging Mutually Exclusive Key Information (MEKI) across modalities. Our work is the first to identify and formalize key evaluation dimensions specific to MDS. Finally, we benchmark state-of-the-art modal evaluation methods, revealing their limitations in distinguishing summaries from advanced MLLMs and their susceptibility to various bias.

</details>

---

## 351. LLM-empowered Dynamic Prompt Routing for Vision-Language Models Tuning under Long-Tailed Distributions

- [ ] LLM-empowered Dynamic Prompt Routing for Vision-Language Models Tuning under Long-Tailed Distributions | https://aclanthology.org/2025.findings-emnlp.799/

- **Link**: https://aclanthology.org/2025.findings-emnlp.799/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Pre-trained vision-language models (VLMs), such as CLIP, have demonstrated impressive capability in visual tasks, but their fine-tuning often suffers from bias in class-imbalanced scenes. Recent works have introduced large language models (LLMs) to enhance VLM fine-tuning withsupplementaryy semantic information. However, they often overlook inherent class imbalance in VLMs’ pre-training, which may lead to bias accumulation in downstream tasks. To address this problem, this paper proposes a Multi-dimensional Dynamic Prompt Routing (MDPR) framework. MDPR constructs a comprehensive knowledge base for classes, spanning multiple visual-semantic dimensions. During fine-tuning, the dynamic routing mechanism aligns global visual classes, retrieves optimal prompts, and balances fine-grained semantics, yielding stable predictions through logits fusion. Extensive experiments on long-tailed benchmarks, including CIFAR-LT, ImageNet-LT, and Places-LT, demonstrate that MDPR achieves comparable results with current SOTA methods. Ablation studies further confirm the effectiveness of our semantic library for tail classes and show that our dynamic routing operates with a slight increase in computational overhead, making MDPR a flexible and efficient enhancement for VLM fine-tuning under data imbalance. The codes are available in https://github.com/Sha843/MDPR.

</details>

---

## 352. One More Modality: DoesAbstractMeaningRepresentation Benefit Visual Question Answering?

- [ ] One More Modality: DoesAbstractMeaningRepresentation Benefit Visual Question Answering? | https://aclanthology.org/2025.findings-emnlp.82/

- **Link**: https://aclanthology.org/2025.findings-emnlp.82/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Question Answering (VQA) requires a vision-language model to reason over both visual and textual inputs to answer questions about images. In this work, we investigate whether incorporating explicit semantic information, in the form of Abstract Meaning Representation (AMR) graphs, can enhance model performance—particularly in low-resource settings where training data is limited. We augment two vision-language models, LXMERT and BLIP-2, with sentence- and document-level AMRs and evaluate their performance under both full and reduced training data conditions. Our findings show that in well-resourced settings, models (in particular the smaller LXMERT) are negatively impacted by incorporating AMR without specialized training. However, in low-resource settings, AMR proves beneficial: LXMERT achieves up to a 13.1% relative gain using sentence-level AMRs. These results suggest that while addition of AMR can lower the performance in some settings, in a low-resource setting AMR can serve as a useful semantic prior, especially for lower-capacity models trained on limited data.

</details>

---

## 353. X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding

- [ ] X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding | https://aclanthology.org/2025.findings-emnlp.822/

- **Link**: https://aclanthology.org/2025.findings-emnlp.822/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Long-form egocentric video understanding provides rich contextual information and unique insights into long-term human behaviors, holding significant potential for applications in embodied intelligence, long-term activity analysis, and personalized assistive technologies. However, existing benchmark datasets primarily focus on single, short (e.g., minutes to tens of minutes) to moderately long videos, leaving a substantial gap in evaluating extensive, ultra-long egocentric video recordings. To address this, we introduce X-LeBench, a novel benchmark dataset meticulously designed to fill this gap by focusing on tasks requiring a comprehensive understanding of extremely long egocentric video recordings. Our X-LeBench develops a life-logging simulation pipeline that produces realistic, coherent daily plans aligned with real-world video data. This approach enables the flexible integration of synthetic daily plans with real-world footage from Ego4D—a massive-scale egocentric video dataset covers a wide range of daily life scenarios—resulting in 432 simulated video life logs spanning from 23 minutes to 16.4 hours. The evaluations of several baseline systems and multimodal large language models (MLLMs) reveal their poor performance across the board, highlighting the inherent challenges of long-form egocentric video understanding, such as temporal localization and reasoning, context aggregation, and memory retention, and underscoring the need for more advanced models.

</details>

---

## 354. LLMs Can Compensate for Deficiencies in Visual Representations

- [ ] LLMs Can Compensate for Deficiencies in Visual Representations | https://aclanthology.org/2025.findings-emnlp.825/

- **Link**: https://aclanthology.org/2025.findings-emnlp.825/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder.

</details>

---

## 355. CanVLMs Recall Factual Associations From Visual References?

- [ ] CanVLMs Recall Factual Associations From Visual References? | https://aclanthology.org/2025.findings-emnlp.850/

- **Link**: https://aclanthology.org/2025.findings-emnlp.850/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Through a controlled study, we identify a systematic deficiency in the multimodal grounding of Vision Language Models (VLMs). While VLMs can recall factual associations when provided a textual reference to an entity, their ability to do so is significantly diminished when the reference is visual instead. Forcing VLMs to rely on image representations of an entity halves their ability to recall factual knowledge, suggesting that VLMs struggle to link their internal knowledge of an entity with its image representation. We show that such linking failures are correlated with the expression of distinct patterns in model internal states, and that probes on these internal states achieve over 92% accuracy at flagging cases where the VLM response is unreliable. These probes can be applied, without retraining, to identify when a VLM will fail to correctly answer a question that requires an understanding of multimodal input. When used to facilitate selective prediction on a visual question answering task, the probes increase coverage by 7.87% (absolute) while also reducing the risk of error by 0.9% (absolute). Addressing the systematic, detectable deficiency is an important avenue in language grounding, and we provide informed recommendations for future directions.

</details>

---

## 356. Debating for Better Reasoning in Vision-Language Models

- [ ] Debating for Better Reasoning in Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.853/

- **Link**: https://aclanthology.org/2025.findings-emnlp.853/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As Large Language Models (LLMs) gain expertise across diverse domains and modalities, scalable oversight becomes increasingly challenging, particularly when their capabilities may surpass human evaluators. Debate has emerged as a promising mechanism for enabling such oversight. We extend the debate paradigm to a multimodal setting, exploring its potential for blind models to supervise and enhance the performance of sighted ones. We focus on visual question answering (VQA), where two “sighted” expert vision-language models debate an answer, while a “blind” (text-only) judge adjudicates based solely on the quality of the arguments. In our framework, the experts only defend answers aligned with their beliefs, thereby obviating the need for explicit role-playing and concentrating the debate on instances of expert disagreement. Experiments on several multimodal tasks demonstrate that the debate framework consistently outperforms individual expert models. Moreover, judgments from blind LLMs can be used to instil reasoning capabilities in vision-language models through fine-tuning.

</details>

---

## 357. MRFD: Multi-Region Fusion Decoding with Self-Consistency for Mitigating Hallucinations inLVLMs

- [ ] MRFD: Multi-Region Fusion Decoding with Self-Consistency for Mitigating Hallucinations inLVLMs | https://aclanthology.org/2025.findings-emnlp.858/

- **Link**: https://aclanthology.org/2025.findings-emnlp.858/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have shown strong performance across multimodal tasks. However, they often produce hallucinations—text that is inconsistent with visual input, due to the limited ability to verify information in different regions of the image. To address this, we propose **Multi-Region Fusion Decoding (MRFD)**, a training-free decoding method that improves factual grounding by modeling inter-region consistency. MRFD identifies salient regions using cross-attention, generates initial responses for each, and computes reliability weights based on Jensen-Shannon Divergence (JSD) among the responses. These weights guide a consistency-aware fusion of per-region predictions, using region-aware prompts inspired by Chain-of-Thought reasoning. Experiments across multiple LVLMs and benchmarks show that MRFD significantly reduces hallucinations and improves response factuality without requiring model updates.

</details>

---

## 358. Captioning for Text-Video Retrieval via Dual-Group Direct Preference Optimization

- [ ] Captioning for Text-Video Retrieval via Dual-Group Direct Preference Optimization | https://aclanthology.org/2025.findings-emnlp.869/

- **Link**: https://aclanthology.org/2025.findings-emnlp.869/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In text-video retrieval, auxiliary captions are often used to enhance video understanding, bridging the gap between the modalities. While recent advances in multi-modal large language models (MLLMs) have enabled strong zero-shot caption generation, we observe that such captions tend to be generic and indistinguishable across visually similar videos, limiting their utility for fine-grained retrieval. Moreover, conventional captioning approaches are typically evaluated using language generation metrics, such as BLEU, which are not typically tailored for retrieval tasks that require making discriminative distinctions between candidates. To address this, we proposeCaRe-DPO, a retrieval framework that directly optimizes caption generation using retrieval relevance scores. At its core is Dual-Group Direct Preference Optimization (DG-DPO), a novel learning strategy that supervises captioning by modeling preferences across groups of distinct video and caption pairs. In addition, we present an MLLM-based retrieval model that incorporates role-embeddings to better distinguish between textual inputs with different functional roles, such as an auxiliary caption and a text query. Through extensive experiments, we demonstrate that CaRe-DPO significantly enhances retrieval performance by effectively leveraging auxiliary knowledge to generate fine-grained captions for retrieval. Code is available at https://github.com/mlvlab/CaReDPO.

</details>

---

## 359. VisualEDU: A Benchmark for Assessing Coding and Visual Comprehension through Educational Problem-Solving Video Generation

- [ ] VisualEDU: A Benchmark for Assessing Coding and Visual Comprehension through Educational Problem-Solving Video Generation | https://aclanthology.org/2025.findings-emnlp.889/

- **Link**: https://aclanthology.org/2025.findings-emnlp.889/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generating logically coherent video from text (T2V) for reasoning-intensive tasks like mathematical problem-solving presents a significant challenge for Vision-Language Models (VLMs). Therefore, we introduce VisualEDU, a benchmark based on Manim package to rigorously evaluate VLM capabilities in producing coherent, step-by-step video solutions for educational purposes, with a framework that integrates meta-prompt learning, visual and code feedback, and a modular drawing toolkit to enhance output quality. Novel metrics for temporal consistency, logical correctness, and visual clarity are proposed, and extensive experiments across nine VLMs reveal that while advanced proprietary models show promise, all struggle significantly with increasing task complexity (e.g., the performances of Claude-3.7-Sonnet and GPT-4o are below 56% on difficult tasks ), highlighting limitations in code generation, visual feedback correction and precise tool invocation. VisualEDU offers a robust platform for systematic T2V assessment in reasoning-intensive domains and guides future VLM improvements in this area.

</details>

---

## 360. FigEx: Aligned Extraction of Scientific Figures and Captions

- [ ] FigEx: Aligned Extraction of Scientific Figures and Captions | https://aclanthology.org/2025.findings-emnlp.899/

- **Link**: https://aclanthology.org/2025.findings-emnlp.899/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automatic understanding of figures in scientific papers is challenging since they often contain subfigures and subcaptions in complex layouts. In this paper, we propose FigEx, a vision-language model to extract aligned pairs of subfigures and subcaptions from scientific papers. We also release BioSci-Fig, a curated dataset of 7,174 compound figures with annotated subfigure bounding boxes and aligned subcaptions. On BioSci-Fig, FigEx improves subfigure detectionAPbover Grounding DINO by 0.023 and boosts caption separation BLEU over Llama-2-13B by 0.465. The source code is available at: https://github.com/Huang-AI4Medicine-Lab/FigEx.

</details>

---

## 361. PATIMT-Bench: A Multi-Scenario Benchmark for Position-Aware Text Image Machine Translation in Large Vision-Language Models

- [ ] PATIMT-Bench: A Multi-Scenario Benchmark for Position-Aware Text Image Machine Translation in Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.900/

- **Link**: https://aclanthology.org/2025.findings-emnlp.900/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text Image Machine Translation (TIMT) aims to translate texts embedded within an image into another language. Current TIMT studies primarily focus on providing translations for all the text within an image, while neglecting to provide bounding boxes and covering limited scenarios. In this work, we extend traditional TIMT into position-aware TIMT (PATIMT), aiming to support fine-grained and layout-preserving translation, which holds great practical value but remains largely unexplored. This task comprises two key sub-tasks: region-specific translation and full-image translation with grounding. To support existing models on PATIMT and conduct fair evaluation, we construct the PATIMT benchmark (PATIMT-Bench), which consists of 10 diverse real-world scenarios. Specifically, we introduce an Adaptive Image OCR Refinement Pipeline, which adaptively selects appropriate OCR tools based on scenario and refines the results of text-rich images. To ensure evaluation reliability, we further construct a test set, which contains 1,200 high-quality instances manually annotated and reviewed by human experts. After fine-tuning on our data, compact Large Vision-Language Models (LVLMs) achieve state-of-the-art performance on both sub-tasks. Experimental results also highlight the scalability and generalizability of our training data.

</details>

---

## 362. Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability

- [ ] Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability | https://aclanthology.org/2025.findings-emnlp.90/

- **Link**: https://aclanthology.org/2025.findings-emnlp.90/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in processing both visual and textual information. However, the critical challenge of alignment between visual and textual representations is not fully understood. This survey presents a comprehensive examination of alignment and misalignment in LVLMs through an explainability lens. We first examine the fundamentals of alignment, exploring its representational and behavioral aspects, training methodologies, and theoretical foundations. We then analyze misalignment phenomena across three semantic levels: object, attribute, and relational misalignment. Our investigation reveals that misalignment emerges from challenges at multiple levels: the data level, the model level, and the inference level. We provide a comprehensive review of existing mitigation strategies, categorizing them into parameter-frozen and parameter-tuning approaches. Finally, we outline promising future research directions, emphasizing the need for standardized evaluation protocols and in-depth explainability studies.

</details>

---

## 363. Self-adaptive Dataset Construction for Real-World Multimodal Safety Scenarios

- [ ] Self-adaptive Dataset Construction for Real-World Multimodal Safety Scenarios | https://aclanthology.org/2025.findings-emnlp.912/

- **Link**: https://aclanthology.org/2025.findings-emnlp.912/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are rapidly evolving, presenting increasingly complex safety challenges. However, current dataset construction methods, which are risk-oriented, fail to cover the growing complexity of real-world multimodal safety scenarios (RMS). And due to the lack of a unified evaluation metric, their overall effectiveness remains unproven. This paper introduces a novel image-oriented self-adaptive dataset construction method for RMS, which starts with images and end constructing paired text and guidance responses. Using the image-oriented method, we automatically generate an RMS dataset comprising 35,610 image–text pairs with guidance responses. Additionally, we introduce a standardized safety dataset evaluation metric: fine-tuning a safety judge model and evaluating its capabilities on other safety datasets. Extensive experiments on various tasks demonstrate the effectiveness of the proposed image-oriented pipeline. The results confirm the scalability and effectiveness of the image-oriented approach, offering a new perspective for the construction of real-world multimodal safety datasets.

</details>

---

## 364. Does Visual Grounding Enhance the Understanding of Embodied Knowledge in Large Language Models?

- [ ] Does Visual Grounding Enhance the Understanding of Embodied Knowledge in Large Language Models? | https://aclanthology.org/2025.findings-emnlp.920/

- **Link**: https://aclanthology.org/2025.findings-emnlp.920/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant progress in multimodal language models (LMs), it remains unclear whether visual grounding enhances their understanding of embodied knowledge compared to text-only models. To address this question, we propose a novel embodied knowledge understanding benchmark based on the perceptual theory from psychology, encompassing visual, auditory, tactile, gustatory, olfactory external senses, and interoception. The benchmark assesses the models’ perceptual abilities across different sensory modalities through vector comparison and question-answering tasks with over 1,700 questions. By comparing 30 state-of-the-art LMs, we surprisingly find that vision-language models (VLMs) do not outperform text-only models in either task. Moreover, the models perform significantly worse in the visual dimension compared to other sensory dimensions. Further analysis reveals that the vector representations are easily influenced by word form and frequency, and the models struggle to answer questions involving spatial perception and reasoning. Our findings underscore the need for more effective integration of embodied knowledge in LMs to enhance their understanding of the physical world.

</details>

---

## 365. CCL-XCoT: An Efficient Cross-Lingual Knowledge Transfer Method for Mitigating Hallucination Generation

- [ ] CCL-XCoT: An Efficient Cross-Lingual Knowledge Transfer Method for Mitigating Hallucination Generation | https://aclanthology.org/2025.findings-emnlp.93/

- **Link**: https://aclanthology.org/2025.findings-emnlp.93/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multilingual Large Language Models (MLLMs) demonstrate strong generalization across languages, yet they remain prone to hallucinations, especially in low-resource languages, due to training data imbalances. These hallucinations, which include inaccurate or fabricated outputs, are particularly problematic in domain-specific generation tasks (Chataigner et al., 2024). To address this challenge, we propose CCL-XCoT (Curriculum-based Contrastive Learning-based Cross-lingual Chain-of-Thought), a two-stage fine-tuning framework for mitigating hallucination in MLLMs. Our approach first enhances cross-lingual semantic alignment through curriculum-based contrastive learning combined with next-token prediction during continued pre-training. Building on this foundation, we then introduce a cross-lingual Chain-of-Thought (XCoT) prompting strategy during instruction fine-tuning, which guides the model to reason in a high-resource language before generating answers in the target low-resource language. Experimental results show that CCL-XCoT reduces hallucination rates by up to 62% and substantially improves factual knowledge transfer across language pairs, without relying on external retrieval or multi-model ensembles.

</details>

---

## 366. Tracing Training Footprints: A Calibration Approach for Membership Inference Attacks Against Multimodal Large Language Models

- [ ] Tracing Training Footprints: A Calibration Approach for Membership Inference Attacks Against Multimodal Large Language Models | https://aclanthology.org/2025.findings-emnlp.931/

- **Link**: https://aclanthology.org/2025.findings-emnlp.931/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the increasing scale of training data for Multimodal Large Language Models (MLLMs) and the lack of data details, there is growing concern about privacy breaches and data security issues. Under black-box access, exploring effective Membership Inference Attacks (MIA) has garnered increasing attention. In real-world applications, where most samples are non-members, the issue of non-members being over-represented in the data manifold, leading to misclassification as member samples, becomes more prominent. This has motivated recent work to focus on developing effective difficulty calibration strategies, producing promising results. However, these methods only consider text-only input during calibration, and their effectiveness is diminished when migrated to MLLMs due to the presence of visual embeddings. To address the above problem, we propose PC-MMIA, focusing on visual instruction fine-tuning data. PC-MMIA is based on the idea that tokens located in poorly generalized local manifolds can better reflect traces of member samples that have been trained. By employing bidirectional perturbation of image embeddings to capture tokens critical to MIA and assigning them different weights, we achieve difficulty calibration. Experimental results demonstrate that our proposed method surpasses existing methods.

</details>

---

## 367. Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models

- [ ] Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.936/

- **Link**: https://aclanthology.org/2025.findings-emnlp.936/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Object hallucinations in Large Vision-Language Models (LVLMs) significantly impede their real-world applicability. As the primary component for accurately interpreting visual information, the choice of visual encoder is pivotal. We hypothesize that the diverse training paradigms employed by different visual encoders instill them with distinct inductive biases, which leads to their diverse hallucination performances. Existing benchmarks typically focus on coarse-grained hallucination detection and fail to capture the diverse hallucinations elaborated in our hypothesis. To systematically analyze these effects, we introduce VHBench-10, a comprehensive benchmark for evaluating LVLMs across ten fine-grained hallucination categories. Our evaluations confirm encoders exhibit unique hallucination characteristics. Building on these insights and the suboptimality of simple feature fusion, we propose VisionWeaver, a novel Context-Aware Routing Network. It employs global visual features to generate routing signals, dynamically aggregating visual features from multiple specialized experts. Comprehensive experiments confirm the effectiveness of VisionWeaver in significantly reducing hallucinations and improving overall model performance. Our code and benchmark are available at https://github.com/whwangovo/VisionWeaver.

</details>

---

## 368. PhysicsArena: The First Multimodal Physics Reasoning Benchmark Exploring Variable, Process, and Solution Dimensions

- [ ] PhysicsArena: The First Multimodal Physics Reasoning Benchmark Exploring Variable, Process, and Solution Dimensions | https://aclanthology.org/2025.findings-emnlp.937/

- **Link**: https://aclanthology.org/2025.findings-emnlp.937/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in diverse reasoning tasks, yet their application to complex physics reasoning remains underexplored. Physics reasoning presents unique challenges, requiring grounding in physical conditions and the interpretation of multimodal information. Current physics benchmarks are limited, often focusing on text-only inputs or solely on problem-solving, thereby overlooking the critical intermediate steps of variable identification and process formulation. To address these limitations, we introduce **PhysicsArena, the first multimodal physics reasoning benchmark designed to holistically evaluate MLLMs across three critical dimensions: variable identification, physical process formulation, and solution derivation.** PhysicsArena aims to provide a comprehensive platform for assessing and advancing the multimodal physics reasoning abilities of MLLMs.

</details>

---

## 369. VIVA+: Human-Centered Situational Decision-Making

- [ ] VIVA+: Human-Centered Situational Decision-Making | https://aclanthology.org/2025.findings-emnlp.944/

- **Link**: https://aclanthology.org/2025.findings-emnlp.944/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) show promising results for embodied agents in operating meaningfully in complex, human-centered environments. Yet, evaluating their capacity for nuanced, human-like reasoning and decision-making remains challenging. In this work, we introduce VIVA+, a cognitively grounded benchmark for evaluating the reasoning and decision-making of MLLMs in human-centered situations. VIVA+ consists of 1,317 real-world situations paired with 6,373 multiple-choice questions, targeting three core abilities for decision-making: (1) Foundational Situation Comprehension, (2) Context-Driven Action Justification, and (3) Reflective Reasoning. Together, these dimensions provide a systematic framework for assessing a model’s ability to perceive, reason, and act in socially meaningful ways. We evaluate the latest commercial and open-source models on VIVA+, where we reveal distinct performance patterns and highlight significant challenges. We further explore targeted training and multi-step reasoning strategies, which yield consistent performance improvements. Finally, our in-depth analysis highlights current model limitations and provides actionable insights for advancing MLLMs toward more robust, context-aware, and socially adept decision-making in real-world settings.

</details>

---

## 370. MAFMO: Multi-modal Adaptive Fusion with Meta-template Optimization for Vision-Language Models

- [ ] MAFMO: Multi-modal Adaptive Fusion with Meta-template Optimization for Vision-Language Models | https://aclanthology.org/2025.findings-emnlp.953/

- **Link**: https://aclanthology.org/2025.findings-emnlp.953/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models like CLIP demonstrate exceptional generalization capabilities but face significant adaptation challenges due to parameter scale, prompt sensitivity, and cross-modal alignment difficulties. Existing approaches primarily focus on single-modality adjustments, leading to suboptimal alignment and limited generalization. We introduce MAFMO, a plug-and-play framework comprising: (1) a Harmonic Cross-Modal Adapter enabling efficient cross-modal knowledge transfer; (2) a Meta-Template Optimization module dynamically generating input-dependent templates; and (3) a Cross-Modal Knowledge Synthesis mechanism preserving critical structural relationships during adaptation. Extensive experiments across multiple fine-grained visual recognition benchmarks demonstrate MAFMO consistently improves existing methods’ performance on both novel classes and harmonic mean, while maintaining robustness under various challenging conditions with minimal computational overhead.

</details>

---

## 371. Data or Language Supervision: What MakesCLIPBetter thanDINO?

- [ ] Data or Language Supervision: What MakesCLIPBetter thanDINO? | https://aclanthology.org/2025.findings-emnlp.98/

- **Link**: https://aclanthology.org/2025.findings-emnlp.98/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

CLIP outperforms self-supervised models like DINO as vision encoders for vision-language models (VLMs), but it remains unclear whether this advantage stems from CLIP’s language supervision or its much larger training data. To disentangle these factors, we pre-train CLIP and DINO under controlled settings—using the same architecture, dataset, and training configuration—achieving similar ImageNet accuracy. Embedding analysis shows that CLIP captures high-level semantics (e.g., object categories, text), while DINO is more responsive to low-level features like colors and styles. When integrated into VLMs and evaluated on 20 VQA benchmarks, CLIP excels at text-intensive tasks, while DINO slightly outperforms on vision-centric ones. Variants of language supervision (e.g., sigmoid loss, pre-trained language encoders) yield limited gains. Our findings provide scientific insights into vision encoder design and its impact on VLM performance.

</details>

---

## 372. Towards an Automated Framework to Audit Youth Safety onTikTok

- [ ] Towards an Automated Framework to Audit Youth Safety onTikTok | https://aclanthology.org/2025.hcinlp-1.9/

- **Link**: https://aclanthology.org/2025.hcinlp-1.9/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper investigates the effectiveness of TikTok’s enforcement mechanisms for limiting the exposure of harmful content to youth accounts. We collect over 7000 videos, classify them as harmful vs not-harmful, and then simulate interactions using age-specific sockpuppet accounts through both passive and active engagement strategies. We also evaluate the performance of large language (LLMs) and vision-language models (VLMs) in detecting harmful content, identifying key challenges in precision and scalability. Preliminary results show minimal differences in content exposure between adult and youth accounts, raising concerns about the platform’s age-based moderation. These findings suggest that the platform needs to strengthen youth safety measures and improve transparency in content moderation.

</details>

---

## 373. MobileA3gent: Training MobileGUIAgents Using Decentralized Self-Sourced Data from Diverse Users

- [ ] MobileA3gent: Training MobileGUIAgents Using Decentralized Self-Sourced Data from Diverse Users | https://aclanthology.org/2025.hcinlp-1.8/

- **Link**: https://aclanthology.org/2025.hcinlp-1.8/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The advancement of mobile GUI agents has opened new opportunities for automating tasks on mobile devices. Training these agents requires large-scale high-quality data, which is prohibitively expensive when relying on human labor. Given the vast population of global mobile phone users, if automated data collection from them becomes feasible, the resulting data volume and the subsequently trained mobile agents could reach unprecedented levels. Nevertheless, two major challenges arise: (1) extracting user instructions without human intervention and (2) utilizing distributed user data while preserving privacy.To tackle these challenges, we propose MobileA3gent, a collaborative framework that trains mobile GUI Agents using decentralized self-sourced data from diverse users. The framework comprises two components, each targeting a specific challenge: (1) Auto-Annotation, which enables the automatic collection of high-quality datasets during users’ routine phone usage with minimal cost. (2) FedVLM-A, which enhances federated VLM training under non-IID distributions by incorporating adapted global aggregation based on both episode-level and step-level variability. Extensive experiments prove that MobileA3gent achieves superior performance over traditional approaches at only 1% of the cost, highlighting its potential for real-world applications. Our code is publicly available at: https://anonymous.4open.science/r/MobileA3gent-Anonymous.

</details>

---

## 374. CHECK-MAT: Probing the Mathematical Reasoning and Rubric-Alignment of Vision-Language Models on Handwritten Solutions

- [ ] CHECK-MAT: Probing the Mathematical Reasoning and Rubric-Alignment of Vision-Language Models on Handwritten Solutions | https://aclanthology.org/2025.mathnlp-main.6/

- **Link**: https://aclanthology.org/2025.mathnlp-main.6/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The application of contemporary NLP models for inference over mathematical text remains a critical and under-explored area. While Vision-Language Models (VLMs) have shown promise, a significant gap exists in their ability to perform nuanced, rubric-based assessment of handwritten mathematical arguments, a task requiring the joint interpretation of visual, textual, and symbolic modalities. This paper directly addresses the need for robust evaluation tasks in this domain. This paper introduces CHECK-MAT, a new benchmark and methodology for the automated, rubric-based assessment of handwritten mathematical solutions using Vision-Language Models (VLMs). Composed of 122 real-world solutions from a high-stakes national exam, CHECK-MAT evaluates the capacity of VLMs to emulate expert graders by identifying logical flaws and applying detailed grading rubrics. Our systematic evaluation of seven state-of-the-art VLMs serves as a direct instance of probing the mathematical understanding of state-of-the-art models. We reveal key limitations in their ability to parse complex notation and align with human grading rubrics, which we frame as a challenge in understanding the linguistic analysis of mathematical discourse. Our work contributes a robust benchmark to the NLP community and offers critical insights for developing models with more sophisticated mathematical reasoning capabilities. You can find code in https://github.com/Karifannaa/Auto-check-EGE-math.

</details>

---

## 375. Mind the (Language) Gap: Towards Probing Numerical and Cross-Lingual Limits ofLVLMs

- [ ] Mind the (Language) Gap: Towards Probing Numerical and Cross-Lingual Limits ofLVLMs | https://aclanthology.org/2025.mrl-main.38/

- **Link**: https://aclanthology.org/2025.mrl-main.38/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce MMCRICBENCH-3K, a benchmark for Visual Question Answering (VQA) on cricket scorecards, designed to evaluate large vision-language models (LVLMs) on complex numerical and cross-lingual reasoning over semi-structured tabular images. MMCRICBENCH-3K comprises 1,463 synthetically generated scorecard images from ODI, T20, and Test formats, accompanied by 1,500 English QA pairs. It includes two subsets: MMCRICBENCH-E-1.5K, featuring English scorecards, and MMCRICBENCH-H1.5K, containing visually similar Hindi scorecards, with all questions and answers kept in English to enable controlled cross-script evaluation. The task demands reasoning over structured numerical data, multi-image context, and implicit domain knowledge. Empirical results show that even state-of-the-art LVLMs, such as GPT-4o and Qwen2.5VL, struggle on the English subset despite it being their primary training language and exhibit a further drop in performance on the Hindi subset. This reveals key limitations in structure-aware visual text understanding, numerical reasoning, and cross-lingual generalization. The dataset is publicly available via Hugging Face at https://huggingface.co/ datasets/DIALab/MMCricBench, to promote LVLM research in this direction.

</details>

---

## 376. Bridging Multimodal and Video Summarization: A Unified Survey

- [ ] Bridging Multimodal and Video Summarization: A Unified Survey | https://aclanthology.org/2025.newsum-main.11/

- **Link**: https://aclanthology.org/2025.newsum-main.11/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal summarization (MMS) and video summarization (VS) have traditionally evolved in separate communities—natural language processing (NLP) and computer vision (CV), respectively. MMS focuses on generating textual summaries from inputs such as text, images, or audio, while VS emphasizes selecting key visual content. With the recent rise of vision-language models (VLMs), these once-disparate tasks are converging under a unified framework that integrates visual and linguistic understanding.In this survey, we provide a unified perspective that bridges MMS and VS. We formalize the task landscape, review key datasets and evaluation metrics, and categorize major modeling approaches into new taxonomy. In addition, we highlight core challenges and outline future directions toward building general-purpose multimodal summarization systems. By synthesizing insights from both NLP and CV communities, this survey aims to establish a coherent foundation for advancing this rapidly evolving field.

</details>

---

## 377. DisCoCLIP: A Distributional Compositional Tensor Network Encoder for Vision-Language Understanding

- [ ] DisCoCLIP: A Distributional Compositional Tensor Network Encoder for Vision-Language Understanding | https://aclanthology.org/2025.starsem-1.25/

- **Link**: https://aclanthology.org/2025.starsem-1.25/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent vision–language models excel at large-scale image–text alignment but often neglect the compositional structure of language, leading to failures on tasks that hinge on word order and predicate–argument structure. We introduce DisCoCLIP, a multimodal encoder that combines a frozen CLIP vision transformer with a novel tensor network text encoder that explicitly encodes syntactic structure. Sentences are parsed with a Combinatory Categorial Grammar parser to yield distributional word tensors whose contractions mirror the sentence’s grammatical derivation. To keep the model efficient, high-order tensors are factorized with tensor decompositions, reducing parameter count from tens of millions to under one million. Trained end-to-end with a self-supervised contrastive loss, DisCoCLIP markedly improves sensitivity to verb semantics and word order: it raises CLIP’s SVO-Probes verb accuracy from 77.6% to 82.4%, boosts ARO attribution and relation scores by over 9% and 4%, and achieves 93.7% on a newly introduced SVO-Swap benchmark. These results demonstrate that embedding explicit linguistic structure via tensor networks yields interpretable, parameter-efficient representations that substantially improve compositional reasoning in vision–language tasks.

</details>

---

## 378. Evaluating Compositional Generalisation inVLMs and Diffusion Models

- [ ] Evaluating Compositional Generalisation inVLMs and Diffusion Models | https://aclanthology.org/2025.starsem-1.9/

- **Link**: https://aclanthology.org/2025.starsem-1.9/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

A fundamental aspect of the semantics of natural language is that novel meanings can be formed from the composition of previously known parts.Vision-language models (VLMs) have made significant progress in recent years, however, there is evidence that they are unable to perform this kind of composition. For example, given an image of a red cube and a blue cylinder, a VLM such as CLIP is likely to incorrectly label the image as a red cylinder or a blue cube, indicating it represents the image as a ‘bag-of-words’ and fails to capture compositional semantics. Diffusion models have recently gained significant attention for their impressive generative abilities, and zero-shot classifiers based on diffusion models have been shown to perform competitively with CLIP in certain compositional tasks. We explore whether the generative Diffusion Classifier has improved compositional generalisation abilities compared to discriminative models. We assess three models—Diffusion Classifier, CLIP, and ViLT—on their ability to bind objects with attributes and relations in both zero-shot learning (ZSL) and generalised zero-shot learning (GZSL) settings. Our results show that the Diffusion Classifier and ViLT perform well at concept binding tasks, but that all models struggle significantly with the relational GZSL task, underscoring the broader challenges VLMs face with relational reasoning. Analysis of CLIP embeddings suggests that the difficulty may stem from overly similar representations of relational concepts such as left and right. Code and dataset are available at [link redacted for anonymity].

</details>

---

## 379. Template-Based Text-to-Image Alignment for Language Accessibility A Study on Visualizing Text Simplifications

- [ ] Template-Based Text-to-Image Alignment for Language Accessibility A Study on Visualizing Text Simplifications | https://aclanthology.org/2025.tsar-1.1/

- **Link**: https://aclanthology.org/2025.tsar-1.1/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Individuals with intellectual disabilities often have difficulties in comprehending complex texts. While many text-to-image models prioritize photorealism over cognitive accessibility it is not clear how visual illustrations relate to text simplifications TS generated from them. This paper presents a structured vision language model VLM prompting framework for generating cognitively accessible images from simplified texts. We designed five prompt templates i.e. Basic Object Focus Contextual Scene Educational Layout Multi-Level Detail and Grid Layout each following distinct spatial arrangements while adhering to accessibility constraints such as object count limits spatial separation and content restrictions. Using 400 sentence-level TS pairs from four established text simplification datasets OneStopEnglish SimPA Wikipedia ASSET we conducted a two-phase evaluation Phase 1 assessed template effectiveness with CLIP similarity scores and Phase 2 involved expert annotation of generated images across ten visual styles by four accessibility specialists. Results show that the Basic Object Focus template achieved the highest semantic alignment indicating that visual minimalism enhances accessibility. Expert evaluation further identified Retro style as the most accessible and Wikipedia as the most effective text source. Inter-annotator agreement varied across dimensions with Text Simplicity showing strong reliability and Image Quality proving more subjective. Overall our framework offers practical guidelines for accessible content creation and underscores the importance of structured prompting in AI-generated visual accessibility tools.

</details>

---

## 380. HALLUCINOGEN: Benchmarking Hallucination in Implicit Reasoning within Large Vision Language Models

- [ ] HALLUCINOGEN: Benchmarking Hallucination in Implicit Reasoning within Large Vision Language Models | https://aclanthology.org/2025.uncertainlp-main.10/

- **Link**: https://aclanthology.org/2025.uncertainlp-main.10/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in complex multimodal tasks. However, these models still suffer from hallucinations, particularly when required to implicitly recognize or infer diverse visual entities from images for complex vision-language tasks. To address this challenge, we propose HALLUCINOGEN, a novel visual question answering (VQA) benchmark that employs contextual reasoning prompts as hallucination attacks to evaluate the extent of hallucination in state-of-the-art LVLMs. Our benchmark provides a comprehensive study of the implicit reasoning capabilities of these models by first categorizing visual entities based on the ease of recognition in an image as either salient (prominent, visibly recognizable objects such as a car) or latent entities (such as identifying a disease from a chest X-ray), which are not readily visible and require domain knowledge or contextual reasoning for accurate inference. Next, we design hallucination attacks for both types of entities to assess hallucinations in LVLMs while performing various vision-language tasks, such as locating or reasoning about specific entities within an image, where models must perform implicit reasoning by verifying the existence of the queried entity within the image before generating responses. Finally, our extensive evaluations of eleven LVLMs, including powerful open-source models (like LLaMA-3.2 and DeepSeek-V2), commercial models like Gemini, and two hallucination mitigation strategies across multiple datasets, demonstrate that current LVLMs remain susceptible to hallucination attacks.

</details>

---

## 381. Can Vision-Language Models Infer Speaker’s Ignorance? The Role of Visual and Linguistic Cues

- [ ] Can Vision-Language Models Infer Speaker’s Ignorance? The Role of Visual and Linguistic Cues | https://aclanthology.org/2025.uncertainlp-main.25/

- **Link**: https://aclanthology.org/2025.uncertainlp-main.25/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This study investigates whether vision-language models (VLMs) can perform pragmatic inference, focusing on ignorance implicatures, utterances that imply the speaker’s lack of precise knowledge. To test this, we systematically manipulated contextual cues: the visually depicted situation (visual cue) and QUD-based linguistic prompts (linguistic cue). When only visual cues were provided, three state-of-the-art VLMs (GPT-4o, Gemini 1.5 Pro, and Claude 3.5 sonnet) produced interpretations largely based on the lexical meaning of the modified numerals. When linguistic cues were added to enhance contextual informativeness, Claude exhibited more human-like inference by integrating both types of contextual cues. In contrast, GPT and Gemini favored precise, literal interpretations. Although the influence of contextual cues increased, they treated each contextual cue independently and aligned them with semantic features rather than engaging in context-driven reasoning. These findings suggest that although the models differ in how they handle contextual cues, Claude’s ability to combine multiple cues may signal emerging pragmatic competence in multimodal models.

</details>

---

## 382. Seeing Symbols, Missing Cultures: Probing Vision-Language Models’ Reasoning on Fire Imagery and Cultural Meaning

- [ ] Seeing Symbols, Missing Cultures: Probing Vision-Language Models’ Reasoning on Fire Imagery and Cultural Meaning | https://aclanthology.org/2025.winlp-main.1/

- **Link**: https://aclanthology.org/2025.winlp-main.1/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) often appearculturally competent but rely on superficial pat.tern matching rather than genuine cultural understanding. We introduce a diagnostic framework to probe VLM reasoning on fire-themedcultural imagery through both classification andexplanation analysis. Testing multiple modelson Western festivals, non-Western traditions.and emergency scenes reveals systematic biases: models correctly identify prominent Western festivals but struggle with underrepresentedcultural events, frequently offering vague labelsor dangerously misclassifying emergencies ascelebrations. These failures expose the risksof symbolic shortcuts and highlight the needfor cultural evaluation beyond accuracy metrics to ensure interpretable and fair multimodalsystems.

</details>

---

## 383. A Simple Data Augmentation Strategy for Text-in-Image ScientificVQA

- [ ] A Simple Data Augmentation Strategy for Text-in-Image ScientificVQA | https://aclanthology.org/2025.winlp-main.17/

- **Link**: https://aclanthology.org/2025.winlp-main.17/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scientific visual question answering poses significant challenges for vision-language models due to the complexity of scientific figures and their multimodal context. Traditional approaches treat the figure and accompanying text (e.g., questions and answer options) as separate inputs. EXAMS-V introduced a new paradigm by embedding both visual and textual content into a single image. However, even state-of-the-art proprietary models perform poorly on this setup in zero-shot settings, underscoring the need for task-specific fine-tuning. To address the scarcity of training data in this “text-in-image” format, we synthesize a new dataset by converting existing separate image-text pairs into unified images. Fine-tuning a small multilingual multimodal model on a mix of our synthetic data and EXAMS-V yields notable gains across 13 languages, demonstrating strong average improvements and cross-lingual transfer.

</details>

---

## 384. Brown Like Chocolate: How Vision-Language Models Associate Skin Tone with Food Colors

- [ ] Brown Like Chocolate: How Vision-Language Models Associate Skin Tone with Food Colors | https://aclanthology.org/2025.winlp-main.32/

- **Link**: https://aclanthology.org/2025.winlp-main.32/

- **Conference**: EMNLP

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We investigate how Vision-Language Models (VLMs) leverage visual features when making analogical comparisons about people. Using synthetic images of individuals varying in skin tone and nationality, we prompt GPT and Gemini models to make analogical associations with desserts and drinks. Results reveal that VLMs systematically associate darker-skinned individuals with brown-colored food items, with GPT showing stronger associations than Gemini. These patterns are amplified in Thai versus English prompts, suggesting language-dependent encoding of visual stereotypes. The associations persist across manipulation checks including position swapping and clothing changes, though presenting individuals alone yields divergent language-specific patterns. This work reveals concerning associations in VLMs’ visual reasoning that vary by language, with important implications for multilingual deployment.

</details>

---

