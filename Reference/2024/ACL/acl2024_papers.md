# ACL 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_acl2024_papers.csv

## 1. OpenVNA: A Framework for Analyzing the Behavior of Multimodal Language Understanding System under Noisy Scenarios

- [ ] OpenVNA: A Framework for Analyzing the Behavior of Multimodal Language Understanding System under Noisy Scenarios | https://aclanthology.org/2024.acl-demos.2/

- **Link**: https://aclanthology.org/2024.acl-demos.2/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present OpenVNA, an open-source framework designed for analyzing the behavior of multimodal language understanding systems under noisy conditions. OpenVNA serves as an intuitive toolkit tailored for researchers, facilitating convenience batch-level robustness evaluation and on-the-fly instance-level demonstration. It primarily features a benchmark Python library for assessing global model robustness, offering high flexibility and extensibility, thereby enabling customization with user-defined noise types and models. Additionally, a GUI-based interface has been developed to intuitively analyze local model behavior. In this paper, we delineate the design principles and utilization of the created library and GUI-based web platform. Currently, OpenVNA is publicly accessible athttps://github.com/thuiar/OpenVNA, with a demonstration video available athttps://youtu.be/0Z9cW7RGct4.

</details>

---

## 2. LEGENT: Open Platform for Embodied Agents

- [ ] LEGENT: Open Platform for Embodied Agents | https://aclanthology.org/2024.acl-demos.32/

- **Link**: https://aclanthology.org/2024.acl-demos.32/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite advancements in Large Language Models (LLMs) and Large Multimodal Models (LMMs), their integration into language-grounded, human-like embodied agents remains incomplete, hindering complex real-life task performance in 3D environments. Existing integrations often feature limited open-sourcing, challenging collective progress in this field. We introduce LEGENT, an open, scalable platform for developing embodied agents using LLMs and LMMs. LEGENT offers a dual approach: a rich 3D environment with interactive, communicable, and actionable agents, paired with a user-friendly interface, and a sophisticated data generation pipeline utilizing advanced algorithms to exploit supervision from simulated worlds at scale. In our experiments, an embryonic vision-language-action model trained on LEGENT-generated data surpasses GPT-4V in embodied tasks, showcasing promising generalization capabilities. The demo video is available at the following link https://video.legent.ai.

</details>

---

## 3. VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval

- [ ] VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval | https://aclanthology.org/2024.acl-long.175/

- **Link**: https://aclanthology.org/2024.acl-long.175/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal retrieval becomes increasingly popular in practice. However, the existing retrievers are mostly text-oriented, which lack the capability to process visual information. Despite the presence of vision-language models like CLIP, the current methods are severely limited in representing the text-only and image-only data. In this work, we present a new embedding model VISTA for universal multi-modal retrieval. Our work brings forth threefold technical contributions. Firstly, we introduce a flexible architecture which extends a powerful text encoder with the image understanding capability by introducing visual token embeddings. Secondly, we develop two data generation strategies, which bring high-quality composed image-text to facilitate the training of the embedding model. Thirdly, we introduce a multi-stage training algorithm, which first aligns the visual token embedding with the text encoder using massive weakly labeled data, and then develops multi-modal representation capability using the generated composed image-text data. In our experiments, VISTA achieves superior performances across a variety of multi-modal retrieval tasks in both zero-shot and supervised settings. Our model, data, and source code are available at https://github.com/FlagOpen/FlagEmbedding.

</details>

---

## 4. Unified Hallucination Detection for Multimodal Large Language Models

- [ ] Unified Hallucination Detection for Multimodal Large Language Models | https://aclanthology.org/2024.acl-long.178/

- **Link**: https://aclanthology.org/2024.acl-long.178/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite significant strides in multimodal tasks, Multimodal Large Language Models (MLLMs) are plagued by the critical issue of hallucination. The reliable detection of such hallucinations in MLLMs has, therefore, become a vital aspect of model evaluation and the safeguarding of practical application deployment. Prior research in this domain has been constrained by a narrow focus on singular tasks, an inadequate range of hallucination categories addressed, and a lack of detailed granularity. In response to these challenges, our work expands the investigative horizons of hallucination detection. We present a novel meta-evaluation benchmark, MHaluBench, meticulously crafted to facilitate the evaluation of advancements in hallucination detection methods. Additionally, we unveil a novel unified multimodal hallucination detection framework, UNIHD, which leverages a suite of auxiliary tools to validate the occurrence of hallucinations robustly. We demonstrate the effectiveness of UNIHD through meticulous evaluation and comprehensive analysis. We also provide strategic insights on the application of specific tools for addressing various categories of hallucinations.

</details>

---

## 5. Expedited Training of Visual Conditioned Language Generation via Redundancy Reduction

- [ ] Expedited Training of Visual Conditioned Language Generation via Redundancy Reduction | https://aclanthology.org/2024.acl-long.19/

- **Link**: https://aclanthology.org/2024.acl-long.19/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduceEVLGen, a streamlined framework designed for the pre-training of visually conditioned language generation models with high computational demands, utilizing frozen pre-trained large language models (LLMs). The conventional approach in vision-language pre-training (VLP) typically involves a two-stage optimization process: an initial resource-intensive phase dedicated to general-purpose vision-language representation learning, focused on extracting and consolidating relevant visual features. This is followed by a subsequent phase that emphasizes end-to-end alignment between visual and linguistic modalities. Our novel one-stage, single-loss framework bypasses the computationally demanding first training stage by gradually merging similar visual tokens during training, while avoiding model collapse caused by single-stage training of BLIP-2 type models. The gradual merging process effectively condenses visual information while preserving semantic richness, resulting in rapid convergence without compromising performance. Our experimental findings demonstrate that our approach accelerates the training of vision-language models by a factor of 5 without a noticeable impact on overall performance. Furthermore, we illustrate that our models significantly narrow the performance gap to current vision-language models using only 1/10 of the data. Finally, we showcase how our image-text models can seamlessly adapt to video-conditioned language generation tasks through novel soft attentive temporal token contextualizing modules. Code: https://github.com/yiren-jian/EVLGen

</details>

---

## 6. In-context Mixing (ICM): Code-mixed Prompts for MultilingualLLMs

- [ ] In-context Mixing (ICM): Code-mixed Prompts for MultilingualLLMs | https://aclanthology.org/2024.acl-long.228/

- **Link**: https://aclanthology.org/2024.acl-long.228/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a simple and effective prompting technique called in-context mixing (ICM) for effective in-context learning (ICL) with multilingual large language models (MLLMs). With ICM, we modify the few-shot examples within ICL prompts to be intra-sententially code-mixed by randomly swapping content words in the target languages with their English translations. We observe that ICM prompts yield superior performance in NLP tasks such as disfluency correction, grammar error correction and text simplification that demand a close correspondence between the input and output sequences. Significant improvements are observed mainly for low-resource languages that are under-represented during the pretraining and finetuning of MLLMs. We present an extensive set of experiments to analyze when ICM is effective and what design choices contribute towards its effectiveness. ICM works consistently and significantly better than other prompting techniques across models of varying capacity such as mT0-XXL, BloomZ and GPT-4.

</details>

---

## 7. Exploiting Intrinsic Multilateral Logical Rules for Weakly Supervised Natural Language Video Localization

- [ ] Exploiting Intrinsic Multilateral Logical Rules for Weakly Supervised Natural Language Video Localization | https://aclanthology.org/2024.acl-long.247/

- **Link**: https://aclanthology.org/2024.acl-long.247/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Weakly supervised natural language video localization (WS-NLVL) aims to retrieve the moment corresponding to a language query in a video with only video-language pairs utilized during training. Despite great success, existing WS-NLVL methods seldomly consider the complex temporal relations enclosing the language query (e.g., between the language query and sub-queries decomposed from it or its synonymous query), yielding illogical predictions. In this paper, we propose a novel plug-and-play method, Intrinsic Multilateral Logical Rules, namely IMLR, to exploit intrinsic temporal relations and logical rules for WS-NLVL. Specifically, we formalize queries derived from the original language query as the nodes of a directed graph, i.e., intrinsic temporal relation graph (ITRG), and the temporal relations between them as the edges. Instead of directly prompting a pre-trained language model, a relation-guided prompting method is introduced to generate ITRG in a hierarchical manner. We customize four types of multilateral temporal logical rules (i.e., identity, inclusion, synchronization, and succession) from ITRG and utilize them to train our model. Experiments demonstrate the effectiveness and superiority of our method on the Charades-STA and ActivityNet Captions datasets.

</details>

---

## 8. Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences

- [ ] Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences | https://aclanthology.org/2024.acl-long.25/

- **Link**: https://aclanthology.org/2024.acl-long.25/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated proficiency in handling a variety of visual-language tasks. However, current MLLM benchmarks are predominantly designed to evaluate reasoning based on static information about a single image, and the ability of modern MLLMs to extrapolate from image sequences, which is essential for understanding our ever-changing world, has been less investigated. To address this challenge, this paper introduces Mementos, a new benchmark designed to assess MLLMs’ sequential image reasoning abilities. Mementos features 4,761 diverse image sequences with varying lengths. We also employ a GPT-4 assisted method to evaluate MLLM reasoning performance. Through a careful evaluation of nine recent MLLMs on Mementos, including GPT-4V and Gemini, we find that they struggle to accurately describe dynamic information about given image sequences, often leading to hallucinations/misrepresentations of objects and their corresponding behaviors. Our quantitative analysis and case studies identify three key factors impacting MLLMs’ sequential image reasoning: the correlation between object and behavioral hallucinations, the influence of co-occurring behaviors, and the compounding impact of behavioral hallucinations.

</details>

---

## 9. RelayAttention for Efficient Large Language Model Serving with Long System Prompts

- [ ] RelayAttention for Efficient Large Language Model Serving with Long System Prompts | https://aclanthology.org/2024.acl-long.270/

- **Link**: https://aclanthology.org/2024.acl-long.270/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A practical large language model (LLM) service may involve a long system prompt, which specifies the instructions, examples, and knowledge documents of the task and is reused across requests. However, the long system prompt causes throughput/latency bottlenecks as the cost of generating the next token grows w.r.t the sequence length. This paper aims to improve the efficiency of LLM services that involve long system prompts. Our key observation is that handling these system prompts requires heavily redundant memory accesses in existing causal attention computation algorithms. Specifically, for batched requests, the cached hidden states (i.e., key-value pairs) of system prompts are transferred from off-chip DRAM to on-chip SRAM multiple times, each corresponding to an individual request. To eliminate such a redundancy, we propose RelayAttention, an attention algorithm that allows reading these hidden states from DRAM exactly once for a batch of input tokens. RelayAttention is a free lunch: it maintains the generation quality while requiring no model retraining, as it is based on a mathematical reformulation of causal attention. We have observed significant performance improvements to a production-level system, vLLM, through integration with RelayAttention. The improvements are even more profound with longer system prompts.

</details>

---

## 10. OLIVE: Object Level In-Context Visual Embeddings

- [ ] OLIVE: Object Level In-Context Visual Embeddings | https://aclanthology.org/2024.acl-long.282/

- **Link**: https://aclanthology.org/2024.acl-long.282/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent generalist vision-language models (VLMs) have demonstrated impressive reasoning capabilities across diverse multimodal tasks. However, these models still struggle with fine-grained object-level understanding and grounding. In terms of modeling, existing VLMs implicitly align text tokens with image patch tokens, which is ineffective for embedding alignment at the same granularity and inevitably introduces noisy spurious background features. Additionally, these models struggle when generalizing to unseen visual concepts and may not be reliable for domain-specific tasks without further fine-tuning. To address these limitations, we propose a novel method to prompt large language models with in-context visual object vectors, thereby enabling controllable object-level reasoning. This eliminates the necessity of fusing a lengthy array of image patch features and significantly speeds up training. Furthermore, we propose region-level retrieval using our object representations, facilitating rapid adaptation to new objects without additional training. Our experiments reveal that our method achieves competitive referring object classification and captioning performance, while also offering zero-shot generalization and robustness to visually challenging contexts.

</details>

---

## 11. Eliciting Better Multilingual Structured Reasoning fromLLMs through Code

- [ ] Eliciting Better Multilingual Structured Reasoning fromLLMs through Code | https://aclanthology.org/2024.acl-long.281/

- **Link**: https://aclanthology.org/2024.acl-long.281/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The development of large language models (LLM) has shown progress on reasoning, though studies have largely considered either English or simple reasoning tasks. To address this, we introduce a multilingual structured reasoning and explanation dataset, termed xSTREET, that covers four tasks across six languages. xSTREET exposes a gap in base LLM performance between English and non-English reasoning tasks.We then propose two methods to remedy this gap, building on the insight that LLMs trained on code are better reasoners. First, at training time, we augment a code dataset with multilingual comments using machine translation while keeping program code as-is. Second, at inference time, we bridge the gap between training and inference by employing a prompt structure that incorporates step-by-step code primitives to derive new facts and find a solution. Our methods show improved multilingual performance on xSTREET, most notably on the scientific commonsense reasoning subtask. Furthermore, the models show no regression on non-reasoning tasks, thus demonstrating our techniques maintain general-purpose abilities.

</details>

---

## 12. SparseFlow: Accelerating Transformers by Sparsifying Information Flows

- [ ] SparseFlow: Accelerating Transformers by Sparsifying Information Flows | https://aclanthology.org/2024.acl-long.323/

- **Link**: https://aclanthology.org/2024.acl-long.323/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Transformers have become the de-facto standard for natural language processing. However, dense information flows within transformers pose significant challenges for real-time and resource-constrained devices, as computational complexity grows quadratically with sequence length. To counteract such dense information flows, we propose SparseFlow, a novel efficient method designed to sparsify the dense pathways of token representations across all transformer blocks. To this end, SparseFlow parameterizes the information flows linking token representations to transformer blocks. These parameterized information flows are optimized to be sparse, allowing only the salient information to pass through into the blocks. To validate the efficacy of SparseFlow, we conduct comprehensive experiments across diverse benchmarks (understanding and generation), scales (ranging from millions to billions), architectures (including encoders, decoders, and seq-to-seq models), and modalities (such as language-only and vision-language). The results convincingly demonstrate that sparsifying the dense information flows leads to substantial speedup gains without compromising task accuracy. For instance, SparseFlow reduces computational costs by half on average, without a significant loss in accuracy.

</details>

---

## 13. UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion

- [ ] UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion | https://aclanthology.org/2024.acl-long.335/

- **Link**: https://aclanthology.org/2024.acl-long.335/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing text-to-image diffusion models primarily generate images from text prompts. However, the inherent conciseness of textual descriptions poses challenges in faithfully synthesizing images with intricate details, such as specific entities or scenes. This paper presents UNIMO-G, a simple multimodal conditional diffusion framework that operates on multimodal prompts with interleaved textual and visual inputs, which demonstrates a unified ability for both text-driven and subject-driven image generation. UNIMO-G comprises two core components: a Multimodal Large Language Model (MLLM) for encoding multimodal prompts, and a conditional denoising diffusion network for generating images based on the encoded multimodal input. We leverage a two-stage training strategy to effectively train the framework: firstly pre-training on large-scale text-image pairs to develop conditional image generation capabilities, and then instruction tuning with multimodal prompts to achieve unified image generation proficiency. A well-designed data processing pipeline involving language grounding and image segmentation is employed to construct multi-modal prompts. UNIMO-G excels in both text-to-image generation and zero-shot subject-driven synthesis, and is notably effective in generating high-fidelity images from complex multimodal prompts involving multiple image entities.

</details>

---

## 14. GroundingGPT: Language Enhanced Multi-modal Grounding Model

- [ ] GroundingGPT: Language Enhanced Multi-modal Grounding Model | https://aclanthology.org/2024.acl-long.360/

- **Link**: https://aclanthology.org/2024.acl-long.360/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) have demonstrated remarkable performance across various tasks. However, these models often prioritize capturing global information and overlook the importance of perceiving local information. This limitation hinders their ability to effectively understand fine-grained details and handle grounding tasks that necessitate nuanced comprehension. Although some recent works have made strides in this, they have primarily focused on single-modality inputs. Therefore, we proposeGroundingGPT, an end-to-end language enhanced multi-modal grounding model. It is designed to perform fine-grained grounding tasks for three modalities: image, video and audio. To enhance the model’s performance, we adopt a coarse-to-fine training strategy, utilizing a three-stage training approach to progressively enhance the model’s semantic awareness and fine-grained understanding capabilities. Additionally, we employ a diversified stage-specific dataset construction pipeline, developing a multi-modal, multi-granularity dataset tailored for training the model in different stages. Extensive experiments conducted on multiple multi-modal benchmarks demonstrate that our model achieves impressive fine-grained understanding of multi-modal inputs on grounding tasks while maintaining or improving its global comprehension capabilities. Our code, model, and dataset are available at https://github.com/lzw-lzw/GroundingGPT.

</details>

---

## 15. Muffin orChihuahua? Challenging Multimodal Large Language Models with MultipanelVQA

- [ ] Muffin orChihuahua? Challenging Multimodal Large Language Models with MultipanelVQA | https://aclanthology.org/2024.acl-long.370/

- **Link**: https://aclanthology.org/2024.acl-long.370/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multipanel images, commonly seen as web screenshots, posters, etc., pervade our daily lives. These images, characterized by their composition of multiple subfigures in distinct layouts, effectively convey information to people. Toward building advanced multimodal AI applications, such as agents that understand complex scenes and navigate through webpages, the skill of multipanel visual reasoning is essential, and a comprehensive evaluation of models in this regard is important. Therefore, we introduce Multipanel Visual Question Answering (MultipanelVQA), a novel benchmark comprising 6,600 triplets of questions, answers, and multipanel images that specifically challenge models in comprehending multipanel images. Our evaluation shows that questions in the MultipanelVQA benchmark pose significant challenges to the state-of-the-art Multimodal Large Language Models (MLLMs) tested, even though humans can attain approximately 99% accuracy on these questions. Distinctively, the MultipanelVQA benchmark features synthetically generated multipanel images specifically crafted to isolate and assess the impact of various factors, such as the layout, on MLLMs’ multipanel image comprehension abilities. As a result, in addition to benchmarking the capabilities of MLLMs in understanding multipanel images, we analyze various factors of the multipanel image that affect MLLMs’ performance with synthetic data and offer insights for enhancement.

</details>

---

## 16. Multimodal Instruction Tuning with Conditional Mixture ofLoRA

- [ ] Multimodal Instruction Tuning with Conditional Mixture ofLoRA | https://aclanthology.org/2024.acl-long.38/

- **Link**: https://aclanthology.org/2024.acl-long.38/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable proficiency in diverse tasks across different domains, with an increasing focus on improving their zero-shot generalization capabilities for unseen multimodal tasks. Multimodal instruction tuning has emerged as a successful strategy for achieving zero-shot generalization by fine-tuning pre-trained models on diverse multimodal tasks through instructions. As MLLMs grow in complexity and size, the need for parameter-efficient fine-tuning methods like Low-Rank Adaption (LoRA), which fine-tunes with a minimal set of parameters, becomes essential. However, applying LoRA in multimodal instruction tuning presents the challenge of task interference, which leads to performance degradation, especially when dealing with a broad array of multimodal tasks. To address this, this paper introduces a novel approach that integrates multimodal instruction tuning with Conditional Mixture-of-LoRA (MixLoRA). It innovates upon LoRA by dynamically constructing low-rank adaptation matrices tailored to the unique demands of each input instance, aiming to mitigate task interference. Experimental results on various multimodal evaluation datasets indicate that MixLoRA not only outperforms the conventional LoRA with the same or even higher ranks, demonstrating its efficacy and adaptability in diverse multimodal tasks.

</details>

---

## 17. PrivLM-Bench: A Multi-level Privacy Evaluation Benchmark for Language Models

- [ ] PrivLM-Bench: A Multi-level Privacy Evaluation Benchmark for Language Models | https://aclanthology.org/2024.acl-long.4/

- **Link**: https://aclanthology.org/2024.acl-long.4/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of language models (LMs) brings unprecedented accessibility and usage for both models and users. On the one hand, powerful LMs achieve state-of-the-art performance over numerous downstream NLP tasks. On the other hand, more and more attention is paid to unrestricted model accesses that may bring malicious privacy risks of data leakage. To address these issues, many recent works propose privacy-preserving language models (PPLMs) with differential privacy (DP). Unfortunately, different DP implementations make it challenging for a fair comparison among existing PPLMs. In this paper, we present PrivLM-Bench, a multi-perspective privacy evaluation benchmark to empirically and intuitively quantify the privacy leakage of LMs. Instead of only reporting DP parameters, PrivLM-Bench sheds light on the neglected inference data privacy during actual usage. PrivLM-Bench first clearly defines multi-faceted privacy objectives. Then, PrivLM-Bench constructs a unified pipeline to perform private fine-tuning. Lastly, PrivLM-Bench performs existing privacy attacks on LMs with pre-defined privacy objectives as the empirical evaluation results. The empirical attack results are used to fairly and intuitively evaluate the privacy leakage of various PPLMs. We conduct extensive experiments on three datasets of GLUE for mainstream LMs.

</details>

---

## 18. Advancement in Graph Understanding: A Multimodal Benchmark and Fine-Tuning of Vision-Language Models

- [ ] Advancement in Graph Understanding: A Multimodal Benchmark and Fine-Tuning of Vision-Language Models | https://aclanthology.org/2024.acl-long.404/

- **Link**: https://aclanthology.org/2024.acl-long.404/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Graph data organizes complex relationships and interactions between objects, facilitating advanced analysis and decision-making across different fields. In this paper, we propose a new paradigm for interactive and instructional graph data understanding and reasoning.Instead of adopting complex graph neural models or heuristic graph-to-text instruction design, we leverage Vision-Language Models (VLMs) to encode the graph images with varying structures across different domains. This paper first evaluates the capabilities of public VLMs in graph learning from multiple aspects. Then it introduces a novel instruction-following dataset for multimodal graph understanding and reasoning in English and Chinese. Besides, by fine-tuning MiniGPT-4 and LLaVA on our dataset, we achieved an accuracy increase of 5%-15% compared to baseline models, with the best-performing model attaining scores comparable to Gemini in GPT-asissted Evaluation. This research not only showcases the potential of integrating VLMs with graph data but also opens new avenues for advancements in graph data understanding.

</details>

---

## 19. Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment

- [ ] Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment | https://aclanthology.org/2024.acl-long.411/

- **Link**: https://aclanthology.org/2024.acl-long.411/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Evaluating and Rethinking the current landscape of Large Multimodal Models (LMMs), we observe that widely-used visual-language projection approaches (e.g., Q-former or MLP) focus on the alignment of image-text descriptions yet ignore the visual knowledge-dimension alignment, i.e., connecting visuals to their relevant knowledge. Visual knowledge plays a significant role in analyzing, inferring, and interpreting information from visuals, helping improve the accuracy of answers to knowledge-based visual questions. In this paper, we mainly explore improving LMMs with visual-language knowledge alignment, especially aimed at challenging knowledge-based visual question answering (VQA). To this end, we present a Cognitive Visual-Language Mapper (CVLM), which contains a pretrained Visual Knowledge Aligner (VKA) and a Fine-grained Knowledge Adapter (FKA) used in the multimodal instruction tuning stage. Specifically, we design the VKA based on the interaction between a small language model and a visual encoder, training it on collected image-knowledge pairs to achieve visual knowledge acquisition and projection. FKA is employed to distill the fine-grained visual knowledge of an image and inject it into Large Language Models (LLMs). We conduct extensive experiments on knowledge-based VQA benchmarks and experimental results show that CVLM significantly improves the performance of LMMs on knowledge-based VQA (average gain by 5.0%). Ablation studies also verify the effectiveness of VKA and FKA, respectively.

</details>

---

## 20. EXAMS-V: A Multi-Discipline Multilingual Multimodal Exam Benchmark for Evaluating Vision Language Models

- [ ] EXAMS-V: A Multi-Discipline Multilingual Multimodal Exam Benchmark for Evaluating Vision Language Models | https://aclanthology.org/2024.acl-long.420/

- **Link**: https://aclanthology.org/2024.acl-long.420/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce EXAMS-V, a new challenging multi-discipline multimodal multilingual exam benchmark for evaluating vision language models. It consists of 20,932 multiple-choice questions across 20 school disciplines covering natural science, social science, and other miscellaneous studies, e.g., religion, fine arts, business, etc. EXAMS-V includes a variety of multimodal features such as text, images, tables, figures, diagrams, maps, scientific symbols, and equations. The questions come in 11 languages from 7 language families. Unlike existing benchmarks, EXAMS-V is uniquely curated by gathering school exam questions from various countries, with a variety of education systems. This distinctive approach calls for intricate reasoning across diverse languages and relies on region-specific knowledge. Solving the problems in the dataset requires advanced perception and joint reasoning over the text and the visual content in the image. Our evaluation results demonstrate that this is a challenging dataset, which is difficult even for advanced vision–text models such as GPT-4V and Gemini; this underscores the inherent complexity of the dataset and its significance as a future benchmark.

</details>

---

## 21. Chain-of-Exemplar: Enhancing Distractor Generation for Multimodal Educational Question Generation

- [ ] Chain-of-Exemplar: Enhancing Distractor Generation for Multimodal Educational Question Generation | https://aclanthology.org/2024.acl-long.432/

- **Link**: https://aclanthology.org/2024.acl-long.432/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multiple-choice questions (MCQs) are important in enhancing concept learning and student engagement for educational purposes. Despite the multimodal nature of educational content, current methods focus mainly on text-based inputs and often neglect the integration of visual information. In this work, we study the problem of multimodal educational question generation, which aims at generating subject-specific educational questions with plausible yet incorrect distractors based on multimodal educational content. To tackle this problem, we introduce a novel framework, named Chain-of-Exemplar (CoE), which utilizes multimodal large language models (MLLMs) with Chain-of-Thought reasoning to improve the generation of challenging distractors. Furthermore, CoE leverages three-stage contextualized exemplar retrieval to retrieve exemplary questions as guides for generating more subject-specific educational questions. Experimental results on the ScienceQA benchmark demonstrate the superiority of CoE in both question generation and distractor generation over existing methods across various subjects and educational levels.

</details>

---

## 22. MemeGuard: AnLLMandVLM-based Framework for Advancing Content Moderation via Meme Intervention

- [ ] MemeGuard: AnLLMandVLM-based Framework for Advancing Content Moderation via Meme Intervention | https://aclanthology.org/2024.acl-long.439/

- **Link**: https://aclanthology.org/2024.acl-long.439/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the digital world, memes present a unique challenge for content moderation due to their potential to spread harmful content. Although detection methods have improved, proactive solutions such as intervention are still limited, with current research focusing mostly on text-based content, neglecting the widespread influence of multimodal content like memes. Addressing this gap, we presentMemeGuard, a comprehensive framework leveraging Large Language Models (LLMs) and Visual Language Models (VLMs) for meme intervention.MemeGuardharnesses a specially fine-tuned VLM,VLMeme, for meme interpretation, and a multimodal knowledge selection and ranking mechanism (MKS) for distilling relevant knowledge. This knowledge is then employed by a general-purpose LLM to generate contextually appropriate interventions. Another key contribution of this work is theInterveningCyberbullying inMultimodalMemes (ICMM)dataset, a high-quality, labeled dataset featuring toxic memes and their corresponding human-annotated interventions. We leverageICMMto testMemeGuard, demonstrating its proficiency in generating relevant and effective responses to toxic memes. redDisclaimer:This paper contains harmful content that may be disturbing to some readers.

</details>

---

## 23. M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought

- [ ] M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought | https://aclanthology.org/2024.acl-long.446/

- **Link**: https://aclanthology.org/2024.acl-long.446/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Chain-of-Thought (MCoT) requires models to leverage knowledge from both textual and visual modalities for step-by-step reasoning, which gains increasing attention. Nevertheless, the current MCoT benchmark still faces some challenges: (1) absence of visual modal reasoning, (2) single-step visual modal reasoning, and (3) domain missing, thereby hindering the development of MCoT. Motivated by this, we introduce a novel benchmark (M3CoT) to address the above challenges, advancing the multi-domain, multi-step, and multi-modal CoT. Additionally, we conduct a thorough evaluation involving abundant MCoT approaches on Vision Large Language Models (VLLMs). In addition, we highlight that the current VLLMs still struggle to correctly reason in M3CoT and there is a large gap between VLLMs and human performance in M3CoT, despite their superior results on previous MCoT benchmarks. To our knowledge, we take the first meaningful step toward the multi-domain, multi-step, and multi-modal scenario in MCoT. We hope that M3CoT will serve as a valuable resource, providing a pioneering foundation in multi-domain, multi-step, multi-modal chain-of-thought research.

</details>

---

## 24. Direct Metric Optimization for Image Captioning through Reward-Weighted Augmented Data Utilization

- [ ] Direct Metric Optimization for Image Captioning through Reward-Weighted Augmented Data Utilization | https://aclanthology.org/2024.acl-long.453/

- **Link**: https://aclanthology.org/2024.acl-long.453/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While image captioning is an essential field of vision language models (VLM), a lack of continuity between the learning objective and final performance metrics of VLMs complicates their training and optimization. Reinforcement learning (RL) can directly optimize such metrics, but it is accompanied by a significant computational cost, making it difficult to apply to recent large-scale VLMs. In this paper, we propose Direct Metric Optimization (DMO), which is a lightweight final-metric-optimizing training method. We replace the computationally expensive exploration process in RL with an offline, diverse text data augmentation and show that self-supervised training on reward-weighted augmented data leads to direct and stable metric optimization. Our experiments demonstrate that DMO achieves performance comparable to those of the state-of-the-art RL method while saving hundreds of times more model forwarding iterations and greater amounts of computation time. This suggests that DMO constitutes a promising alternative for metric optimization in the era of large-scale VLMs.

</details>

---

## 25. DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding

- [ ] DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding | https://aclanthology.org/2024.acl-long.463/

- **Link**: https://aclanthology.org/2024.acl-long.463/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Enterprise documents such as forms, receipts, reports, and other such records, often carry rich semantics at the intersection of textual and spatial modalities. The visual cues offered by their complex layouts play a crucial role in comprehending these documents effectively. In this paper, we present DocLLM, a lightweight extension to traditional large language models (LLMs) for reasoning over visual documents, taking into account both textual semantics and spatial layout. Our model differs from existing multimodal LLMs by avoiding expensive image encoders and focuses exclusively on bounding box information to incorporate the spatial layout structure. Specifically, the cross-alignment between text and spatial modalities is captured by decomposing the attention mechanism in classical transformers to a set of disentangled matrices. Furthermore, we devise a pre-training objective that learns to infill text segments. This approach allows us to address irregular layouts and heterogeneous content frequently encountered in visual documents. The pre-trained model is fine-tuned using a large-scale instruction dataset, covering four core document intelligence tasks. We demonstrate that our solution outperforms SotA LLMs on 14 out of 16 datasets across all tasks, and generalizes well to 4 out of 5 previously unseen datasets.

</details>

---

## 26. Multimodal Table Understanding

- [ ] Multimodal Table Understanding | https://aclanthology.org/2024.acl-long.493/

- **Link**: https://aclanthology.org/2024.acl-long.493/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although great progress has been made by previous table understanding methods including recent approaches based on large language models (LLMs), they rely heavily on the premise that given tables must be converted into a certain text sequence (such as Markdown or HTML) to serve as model input. However, it is difficult to access such high-quality textual table representations in some real-world scenarios, and table images are much more accessible. Therefore, how to directly understand tables using intuitive visual information is a crucial and urgent challenge for developing more practical applications. In this paper, we propose a new problem, multimodal table understanding, where the model needs to generate correct responses to various table-related requests based on the given table image. To facilitate both the model training and evaluation, we construct a large-scale dataset named MMTab, which covers a wide spectrum of table images, instructions and tasks. On this basis, we develop Table-LLaVA, a generalist tabular multimodal large language model (MLLM), which significantly outperforms recent open-source MLLM baselines on 23 benchmarks under held-in and held-out settings.

</details>

---

## 27. MM-SAP: A Comprehensive Benchmark for Assessing Self-Awareness of Multimodal Large Language Models in Perception

- [ ] MM-SAP: A Comprehensive Benchmark for Assessing Self-Awareness of Multimodal Large Language Models in Perception | https://aclanthology.org/2024.acl-long.498/

- **Link**: https://aclanthology.org/2024.acl-long.498/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in visual perception and understanding. However, these models also suffer from hallucinations, which limit their reliability as AI systems. We believe that these hallucinations are partially due to the models’ struggle with understanding what they can and cannot perceive from images, a capability we refer to as self-awareness in perception. Despite its importance, this aspect of MLLMs has been overlooked in prior studies. In this paper, we aim to define and evaluate the self-awareness of MLLMs in perception. To do this, we first introduce the knowledge quadrant in perception, which helps define what MLLMs know and do not know about images. Using this framework, we propose a novel benchmark, the Self-Awareness in Perception for MLLMs (MM-SAP), specifically designed to assess this capability. We apply MM-SAP to a variety of popular MLLMs, offering a comprehensive analysis of their self-awareness and providing detailed insights. The experiment results reveal that current MLLMs possess limited self-awareness capabilities, pointing to a crucial area for future advancement in the development of trustworthy MLLMs. Code and data are available at https://github.com/YHWmz/MM-SAP.

</details>

---

## 28. VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks

- [ ] VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks | https://aclanthology.org/2024.acl-long.50/

- **Link**: https://aclanthology.org/2024.acl-long.50/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Autonomous agents capable of planning, reasoning, and executing actions on the web offer a promising avenue for automating computer tasks. However, the majority of existing benchmarks primarily focus on text-based agents, neglecting many natural tasks that require visual information to effectively solve. Given that most computer interfaces cater to human perception, visual information often augments textual data in ways that text-only models struggle to harness effectively. To bridge this gap, we introduce VisualWebArena, a benchmark designed to assess the performance of multimodal web agents on *realistic visually grounded tasks*. VisualWebArena comprises of a set of diverse and complex web-based tasks that evaluate various capabilities of autonomous multimodal agents. To perform on this benchmark, agents need to accurately process image-text inputs, interpret natural language instructions, and execute actions on websites to accomplish user-defined objectives. We conduct an extensive evaluation of state-of-the-art LLM-based autonomous agents, including several multimodal models. Through extensive quantitative and qualitative analysis, we identify several limitations of text-only LLM agents, and reveal gaps in the capabilities of state-of-the-art multimodal language agents. VisualWebArena provides a framework for evaluating multimodal autonomous language agents, and offers insights towards building stronger autonomous agents for the web.

</details>

---

## 29. Synchronized Video Storytelling: Generating Video Narrations with Structured Storyline

- [ ] Synchronized Video Storytelling: Generating Video Narrations with Structured Storyline | https://aclanthology.org/2024.acl-long.513/

- **Link**: https://aclanthology.org/2024.acl-long.513/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Video storytelling is engaging multimedia content that utilizes video and its accompanying narration to share a story and attract the audience, where a key challenge is creating narrations for recorded visual scenes. Previous studies on dense video captioning and video story generation have made some progress. However, in practical applications, we typically require synchronized narrations for ongoing visual scenes. In this work, we introduce a new task of Synchronized Video Storytelling, which aims to generate synchronous and informative narrations for videos. These narrations, associated with each video clip, should relate to the visual content, integrate relevant knowledge, and have an appropriate word count corresponding to the clip’s duration. Specifically, a structured storyline is beneficial to guide the generation process, ensuring coherence and integrity. To support the exploration of this task, we introduce a new benchmark dataset E-SyncVidStory with rich annotations. Since existing Multimodal LLMs are not effective in addressing this task in one-shot or few-shot settings, we propose a framework named VideoNarrator that can generate a storyline for input videos and simultaneously generate narrations with the guidance of the generated or predefined storyline. We further introduce a set of evaluation metrics to thoroughly assess the generation. Both automatic and human evaluations validate the effectiveness of our approach. Our dataset, codes, and evaluations will be released.

</details>

---

## 30. Fine-Grained Image-Text Alignment in Medical Imaging Enables Explainable Cyclic Image-Report Generation

- [ ] Fine-Grained Image-Text Alignment in Medical Imaging Enables Explainable Cyclic Image-Report Generation | https://aclanthology.org/2024.acl-long.514/

- **Link**: https://aclanthology.org/2024.acl-long.514/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-grained vision-language models (VLM) have been widely used for inter-modality local alignment between the predefined fixed patches and textual words. However, in medical analysis, lesions exhibit varying sizes and positions, and using fixed patches may cause incomplete representations of lesions. Moreover, these methods provide explainability by using heatmaps to show the general image areas potentially associated with texts rather than specific regions, making their explanations not explicit and specific enough. To address these issues, we propose a novel Adaptive patch-word Matching (AdaMatch) model to correlate chest X-ray (CXR) image regions with words in medical reports and apply it to CXR-report generation to provide explainability for the generation process. AdaMatch exploits the fine-grained relation between adaptive patches and words to provide explanations of specific image regions with corresponding words. To capture the abnormal regions of varying sizes and positions, we introduce an Adaptive Patch extraction (AdaPatch) module to acquire adaptive patches for these regions adaptively. Aiming to provide explicit explainability for the CXR-report generation task, we propose an AdaMatch-based bidirectional LLM for Cyclic CXR-report generation (AdaMatch-Cyclic). It employs AdaMatch to obtain the keywords for CXR images and ‘keypatches’ for medical reports as hints to guide CXR-report generation. Extensive experiments on two publicly available CXR datasets validate the effectiveness of our method and its superior performance over existing methods. Source code will be released.

</details>

---

## 31. AnyGPT: Unified MultimodalLLMwith Discrete Sequence Modeling

- [ ] AnyGPT: Unified MultimodalLLMwith Discrete Sequence Modeling | https://aclanthology.org/2024.acl-long.521/

- **Link**: https://aclanthology.org/2024.acl-long.521/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages.We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs.Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/.

</details>

---

## 32. Tuning Large Multimodal Models for Videos using Reinforcement Learning fromAIFeedback

- [ ] Tuning Large Multimodal Models for Videos using Reinforcement Learning fromAIFeedback | https://aclanthology.org/2024.acl-long.52/

- **Link**: https://aclanthology.org/2024.acl-long.52/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large language models have influenced the development of video large multimodal models (VLMMs). Previous approaches for VLMMs involve Supervised Fine-Tuning (SFT) with instruction-tuned datasets, integrating LLM with visual encoders, and additional learnable parameters. Here, aligning video with text, and vice versa, remains a challenge, primarily due to the insufficient quality and quantity of multimodal instruction-tune data compared to that of text-only. This discrepancy often results in alignments that poorly ground the video content. To address this, we present a novel alignment strategy that employs a multimodal AI system equipped with Reinforcement Learning from AI Feedback (RLAIF), providing self-preference feedback to refine itself and facilitating the alignment of video and text modalities. Our approach uniquely integrates detailed video descriptions as context into a multimodal AI system during the preference feedback generation to enrich the understanding of video content, a process we call context-aware reward modeling. Empirical evaluations on various video benchmarks demonstrate that our VLM-RLAIF outperforms existing approaches, including the SFT model. We commit to open-sourcing our code, models, and datasets to foster further research in this area.

</details>

---

## 33. AbsInstruct: Eliciting Abstraction Ability fromLLMs through Explanation Tuning with Plausibility Estimation

- [ ] AbsInstruct: Eliciting Abstraction Ability fromLLMs through Explanation Tuning with Plausibility Estimation | https://aclanthology.org/2024.acl-long.55/

- **Link**: https://aclanthology.org/2024.acl-long.55/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Abstraction ability is crucial in human intelligence, which can also benefit various tasks in NLP study. Existing work shows that LLMs are deficient in abstract ability, and how to improve it remains unexplored. In this work, we design the framework AbsInstruct to enhance LLMs’ abstraction ability through instruction tuning. The framework builds instructions with in-depth explanations to assist LLMs in capturing the underlying rationale of abstraction. Meanwhile, we introduce a plausibility estimator to select instructions that are more consistent with the abstraction knowledge of LLMs to be aligned. Then, our framework combines abstraction instructions with general-purpose ones to build a hybrid dataset. Extensive experiments and analyses demonstrate that our framework can considerably enhance LLMs’ abstraction ability with strong generalization performance while maintaining their general instruction-following abilities.

</details>

---

## 34. CODIS: Benchmarking Context-dependent Visual Comprehension for Multimodal Large Language Models

- [ ] CODIS: Benchmarking Context-dependent Visual Comprehension for Multimodal Large Language Models | https://aclanthology.org/2024.acl-long.573/

- **Link**: https://aclanthology.org/2024.acl-long.573/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have demonstrated promising results in a variety of tasks that combine vision and language. As these models become more integral to research and applications, conducting comprehensive evaluations of their capabilities has grown increasingly important. However, most existing benchmarks fail to consider that, in certain situations, images need to be interpreted within a broader context. In this work, we introduce a new benchmark, named as CODIS, designed to assess the ability of models to use context provided in free-form text to enhance visual comprehension. Our findings indicate that MLLMs consistently fall short of human performance on this benchmark. Further analysis confirms that these models struggle to effectively extract and utilize contextual information to improve their understanding of images. This underscores the pressing need to enhance the ability of MLLMs to comprehend visuals in a context-dependent manner.

</details>

---

## 35. Exploring Chain-of-Thought for Multi-modal Metaphor Detection

- [ ] Exploring Chain-of-Thought for Multi-modal Metaphor Detection | https://aclanthology.org/2024.acl-long.6/

- **Link**: https://aclanthology.org/2024.acl-long.6/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Metaphors are commonly found in advertising and internet memes. However, the free form of internet memes often leads to a lack of high-quality textual data. Metaphor detection demands a deep interpretation of both textual and visual elements, requiring extensive common-sense knowledge, which poses a challenge to language models. To address these challenges, we propose a compact framework called C4MMD, which utilizes aChain-of-Thought(CoT) methodforMulti-modalMetaphorDetection. Specifically, our approach designs a three-step process inspired by CoT that extracts and integrates knowledge from Multi-modal Large Language Models(MLLMs) into smaller ones. We also developed a modality fusion architecture to transform knowledge from large models into metaphor features, supplemented by auxiliary tasks to improve model performance. Experimental results on the MET-MEME dataset demonstrate that our method not only effectively enhances the metaphor detection capabilities of small models but also outperforms existing models. To our knowledge, this is the first systematic study leveraging MLLMs in metaphor detection tasks. The code for our method is publicly available athttps://github.com/xyz189411yt/C4MMD.

</details>

---

## 36. Browse and Concentrate: Comprehending Multimodal Content via Prior-LLMContext Fusion

- [ ] Browse and Concentrate: Comprehending Multimodal Content via Prior-LLMContext Fusion | https://aclanthology.org/2024.acl-long.605/

- **Link**: https://aclanthology.org/2024.acl-long.605/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the bloom of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs) that incorporate LLMs with pre-trained vision models have recently demonstrated impressive performance across diverse vision-language tasks. However, they fall short to comprehend context involving multiple images. A primary reason for this shortcoming is that the visual features for each images are encoded individually by frozen encoders before feeding into the LLM backbone, lacking awareness of other images and the multimodal instructions. We term this issue as prior-LLM modality isolation and propose a two phase paradigm, browse-and-concentrate, to enable in-depth multimodal context fusion prior to feeding the features into LLMs. This paradigm initially “browses” through the inputs for essential insights, and then revisits the inputs to “concentrate” on crucial details, guided by these insights, to achieve a more comprehensive understanding of the multimodal inputs. Additionally, we develop training strategies specifically to enhance the understanding of multi-image inputs. Our method markedly boosts the performance on 7 multi-image scenarios, contributing to increments on average accuracy by 2.13% and 7.60% against strong MLLMs baselines with 3B and 11B LLMs, respectively.

</details>

---

## 37. Model Composition for Multimodal Large Language Models

- [ ] Model Composition for Multimodal Large Language Models | https://aclanthology.org/2024.acl-long.606/

- **Link**: https://aclanthology.org/2024.acl-long.606/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent developments in Multimodal Large Language Models (MLLMs) have shown rapid progress, moving towards the goal of creating versatile MLLMs that understand inputs from various modalities. However, existing methods typically rely on joint training with paired multimodal instruction data, which is resource-intensive and challenging to extend to new modalities. In this paper, we propose a new paradigm through the model composition of existing MLLMs to create a new model that retains the modal understanding capabilities of each original model. Our basic implementation, NaiveMC, demonstrates the effectiveness of this paradigm by reusing modality encoders and merging LLM parameters. Furthermore, we introduce DAMC to address parameter interference and mismatch issues during the merging process, thereby enhancing the model performance. To facilitate research in this area, we propose MCUB, a benchmark for assessing ability of MLLMs to understand inputs from diverse modalities. Experiments on this benchmark and four other multimodal understanding tasks show significant improvements over baselines, proving that model composition can create a versatile model capable of processing inputs from multiple modalities.

</details>

---

## 38. Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond

- [ ] Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond | https://aclanthology.org/2024.acl-long.639/

- **Link**: https://aclanthology.org/2024.acl-long.639/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent advancements in generative language models have demonstrated their ability to memorize knowledge from documents and recall knowledge to respond to user queries effectively. Building upon this capability, we propose to enable multimodal large language models (MLLMs) to memorize and recall images within their parameters. Given a user query for visual content, the MLLM is anticipated to “recall” the relevant image from its parameters as the response. Achieving this target presents notable challenges, including inbuilt visual memory and visual recall schemes within MLLMs. To address these challenges, we introduce a generative cross-modal retrieval framework, which assigns unique identifier strings to represent images and involves two training steps: learning to memorize and learning to retrieve. The first step focuses on training the MLLM to memorize the association between images and their respective identifiers. The latter step teaches the MLLM to generate the corresponding identifier of the target image, given the textual query input. By memorizing images in MLLMs, we introduce a new paradigm to cross-modal retrieval, distinct from previous discriminative approaches. The experiments demonstrate that the generative paradigm performs effectively and efficiently even with large-scale image candidate sets.

</details>

---

## 39. Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models

- [ ] Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models | https://aclanthology.org/2024.acl-long.648/

- **Link**: https://aclanthology.org/2024.acl-long.648/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Though advanced in understanding visual information with human languages, Large Vision-Language Models (LVLMs) still suffer from multimodal hallucinations. A natural concern is that during multimodal interaction, the generated hallucinations could influence the LVLMs’ subsequent generation. Thus, we raise a question:When presented with a query relevant to the previously generated hallucination, will LVLMs be misled and respond incorrectly, even though the ground visual information exists?To answer this, we propose a framework called\\textitMMHalSnowballto evaluate LVLMs’ behaviors when encountering generated hallucinations, where LVLMs are required to answer specific visual questions within a curated hallucinatory conversation. Crucially, our experiment shows that the performance of open-source LVLMs drops by at least31\\%, indicating that LVLMs are prone to accept the generated hallucinations and make false claims that they would not have supported without distractions. We term thisMultimodal Hallucination Snowballing. To mitigate this issue, we further propose a training-free method calledResidual Visual Decoding,where we revise the output distribution of LVLMs with the one derived from the residual visual input, providing models with direct access to the visual information. Experiments show that our method can mitigate more than24\\%of the snowballed multimodal hallucination while maintaining capabilities.

</details>

---

## 40. VisDiaHalBench: A Visual Dialogue Benchmark For Diagnosing Hallucination in Large Vision-Language Models

- [ ] VisDiaHalBench: A Visual Dialogue Benchmark For Diagnosing Hallucination in Large Vision-Language Models | https://aclanthology.org/2024.acl-long.658/

- **Link**: https://aclanthology.org/2024.acl-long.658/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the significant success of large vision-language models (LVLMs), some studies have revealed that LVLMs suffer from the hallucination problem, where the LVLMs’ response contains descriptions of non-existent objects. Although various benchmarks have been proposed to investigate this problem, they mostly focus on single-turn evaluation and overlook the hallucination raised by textual inputs. To investigate the hallucination problem of LVLMs when given long-term misleading textual history, we propose a novel visual dialogue hallucination evaluation benchmark VisDiaHalBench. The benchmark consists of samples with five-turn questions about an edited image and its original version. VisDiaHalBench differs from previous hallucination benchmarks in the following three points: 1) The questions and answers are unambiguously grounded by annotated scene graphs. 2) The images are uncommonly edited to inspect the visual model and common-object hallucination in LLMs. 3) The carefully designed dialogue refers a same object in different turns to assess the image consistency and influence of history for LVLMs. The detailed analysis of several state-of-the-art LVLMs across image consistency, visual understanding, history influence, and other dimensions reveals their substantial performance gap with single-turn VQA tasks. The benchmark is released in: https://github.com/qingxingcao/VisDiaHalBench

</details>

---

## 41. VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation

- [ ] VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation | https://aclanthology.org/2024.acl-long.663/

- **Link**: https://aclanthology.org/2024.acl-long.663/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the rapidly advancing field of conditional image generation research, challenges such as limited explainability lie in effectively evaluating the performance and capabilities of various models. This paper introduces VIEScore, a Visual Instruction-guided Explainable metric for evaluating any conditional image generation tasks. VIEScore leverages general knowledge from Multimodal Large Language Models (MLLMs) as the backbone and does not require training or fine-tuning. We evaluate VIEScore on seven prominent tasks in conditional image tasks and found: (1) VIEScore (GPT4-o) achieves a high Spearman correlation of 0.4 with human evaluations, while the human-to-human correlation is 0.45. (2) VIEScore (with open-source MLLM) is significantly weaker than GPT-4o and GPT-4v in evaluating synthetic images. (3) VIEScore achieves a correlation on par with human ratings in the generation tasks but struggles in editing tasks. With these results, we believe VIEScore shows its great potential to replace human judges in evaluating image synthesis tasks.

</details>

---

## 42. Peacock: A Family ofArabic Multimodal Large Language Models and Benchmarks

- [ ] Peacock: A Family ofArabic Multimodal Large Language Models and Benchmarks | https://aclanthology.org/2024.acl-long.689/

- **Link**: https://aclanthology.org/2024.acl-long.689/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have proven effective in a wide range of tasks that require complex reasoning and linguistic comprehension. However, due to a lack of high-quality multimodal resources in languages other than English, the success of MLLMs remains relatively limited to English-based settings. This poses significant challenges in developing comparable models for other languages, even those with large speaker populations, such as Arabic. To alleviate this challenge, we introduce a comprehensive family of Arabic MLLMs, dubbed *Peacock*, with strong vision and language capabilities. Through comprehensive qualitative and quantitative analysis, we demonstrate the solid performance of our models on various visual reasoning tasks and further show their emerging dialectal potential. Additionally, we introduce *Henna*, a new benchmark specifically designed for assessing MLLMs on aspects related to Arabic culture, setting the first stone for culturally-aware Arabic MLLMs. The GitHub repository for the *Peacock* project is available at [https://github.com/UBC-NLP/peacock](https://github.com/UBC-NLP/peacock).

</details>

---

## 43. Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks

- [ ] Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks | https://aclanthology.org/2024.acl-long.690/

- **Link**: https://aclanthology.org/2024.acl-long.690/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multistep instructions, such as recipes and how-to guides, greatly benefit from visual aids, such as a series of images that accompany the instruction steps. While Large Language Models (LLMs) have become adept at generating coherent textual steps, Large Vision/Language Models (LVLMs) are less capable of generating accompanying image sequences. The most challenging aspect is that each generated image needs to adhere to the relevant textual step instruction, as well as be visually consistent with earlier images in the sequence. To address this problem, we propose an approach for generating consistent image sequences, which integrates a Latent Diffusion Model (LDM) with an LLM to transform the sequence into a caption to maintain the semantic coherence of the sequence. In addition, to maintain the visual coherence of the image sequence, we introduce a copy mechanism to initialise reverse diffusion processes with a latent vector iteration from a previously generated image from a relevant step. Both strategies will condition the reverse diffusion process on the sequence of instruction steps and tie the contents of the current image to previous instruction steps and corresponding images. Experiments show that the proposed approach is preferred by humans in 46.6% of the cases against 26.6% for the second best method. In addition, automatic metrics showed that the proposed method maintains semantic coherence and visual consistency across steps in both domains.

</details>

---

## 44. REFINESUMM: Self-RefiningMLLMfor Generating a Multimodal Summarization Dataset

- [ ] REFINESUMM: Self-RefiningMLLMfor Generating a Multimodal Summarization Dataset | https://aclanthology.org/2024.acl-long.743/

- **Link**: https://aclanthology.org/2024.acl-long.743/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) excel at synthesizing key information from diverse sources. However, generating accurate and faithful multimodal summaries is challenging, primarily due to the lack of appropriate multimodal datasets for fine-tuning that meaningfully integrate textual and visual modalities. To address this gap, we present a new dataset designed specifically for image-text multimodal summarization, harnessing the capabilities of state-of-the-art MLLMs. We generate summaries from Wikipedia sections and corresponding images and evaluate them across text-based, visual and multimodal dimensions, employing reference-free metrics. To refine the dataset, we: (1) Filter the MLLM-generated summaries by training a critic model on human annotations and using its predictions to remove low-quality summaries; (2) Fine-tune the MLLM with the filtered high-quality summaries; (3) Use the fine-tuned model in turn to regenerate the summaries. This self-refinement process significantly improves summary quality, as measured by human judgements and automatic multimodal metrics, resulting in a valuable dataset for multimodal summarization research. The dataset is publicly available at https://github.com/amazon-science/refinesumm.

</details>

---

## 45. Multi-modal Preference Alignment Remedies Degradation of Visual Instruction Tuning on Language Models

- [ ] Multi-modal Preference Alignment Remedies Degradation of Visual Instruction Tuning on Language Models | https://aclanthology.org/2024.acl-long.765/

- **Link**: https://aclanthology.org/2024.acl-long.765/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models (MLLMs) are expected to support multi-turn queries of interchanging image and text modalities in production. However, the current MLLMs trained with visual-question-answering (VQA) datasets could suffer from degradation, as VQA datasets lack the diversity and complexity of the original text instruction datasets with which the underlying language model was trained. To address this degradation, we first collect a lightweight, 5k-sample VQA preference dataset where answers were annotated by Gemini for five quality metrics in a granular fashion and investigate standard Supervised Fine-tuning, rejection sampling, Direct Preference Optimization (DPO) and SteerLM algorithms. Our findings indicate that with DPO, we can surpass the instruction-following capabilities of the language model, achieving a 6.73 score on MT-Bench, compared to Vicuna’s 6.57 and LLaVA’s 5.99. This enhancement in textual instruction-following capability correlates with boosted visual instruction performance (+4.9% on MM-Vet, +6% on LLaVA-Bench), with minimal alignment tax on visual knowledge benchmarks compared to the previous RLHF approach. In conclusion, we propose a distillation-based multi-modal alignment model with fine-grained annotations on a small dataset that restores and boosts MLLM’s language capability after visual instruction tuning.

</details>

---

## 46. DeVAn: Dense Video Annotation for Video-Language Models

- [ ] DeVAn: Dense Video Annotation for Video-Language Models | https://aclanthology.org/2024.acl-long.772/

- **Link**: https://aclanthology.org/2024.acl-long.772/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present a novel human annotated dataset for evaluating the ability for visual-language models to generate both short and long descriptions for real-world video clips, termedDeVAn(Dense Video Annotation). The dataset contains 8.5K YouTube video clips of 20-60 seconds in duration and covers a wide range of topics and interests. Each video clip is independently annotated by 5 human annotators, producing both captions (1 sentence) and summaries (3-10 sentences). Given any video selected from the dataset and its corresponding ASR information, we evaluate visual-language models on either caption or summary generation that is grounded in both the visual and auditory content of the video. Additionally, models are also evaluated on caption- and summary-based retrieval tasks, where the summary-based retrieval task requires the identification of a target video givenexcerptsof a given summary. Given the novel nature of the paragraph-length video summarization task, we compared different existing evaluation metrics and their alignment with human preferences and found that model-based evaluation metrics provide more semantically-oriented and human-aligned evaluation. Finally, we benchmarked a wide range of current video-language models on DeVAn, and we aim for DeVAn to serve as a useful evaluation set in the age of large language models and complex multi-modal tasks. Code is available at https://github.com/TK-21st/DeVAn.

</details>

---

## 47. MultimodalArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models

- [ ] MultimodalArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models | https://aclanthology.org/2024.acl-long.775/

- **Link**: https://aclanthology.org/2024.acl-long.775/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) excel across diverse tasks involving concrete images from natural scenes. However, their ability to interpret abstract figures, such as geometry shapes and scientific plots, remains limited due to a scarcity of training datasets in scientific domains.To fill this gap, we introduce Multimodal ArXiv, consisting of ArXivCap and ArXivQA, for enhancing LVLMs scientific comprehension.ArXivCap is a figure-caption dataset comprising 6.4M images and 3.9M captions, sourced from 572K ArXiv papers spanning various scientific domains.Drawing from ArXivCap, we introduce ArXivQA, a question-answering dataset generated by prompting GPT-4V based on scientific figures. ArXivQA greatly enhances open-sourced LVLMs’ mathematical reasoning capabilities, achieving a 10.4% absolute accuracy gain on a multimodal mathematical reasoning benchmark.Furthermore, employing ArXivCap, we devise four vision-to-text tasks for benchmarking LVLMs.Evaluation results with state-of-the-art LVLMs underscore their struggle with the nuanced semantics of academic figures, while domain-specific training yields substantial performance gains.Our error analysis uncovers misinterpretations of visual context, recognition errors, and the production of overly simplified captions by current LVLMs, shedding light on future improvements.

</details>

---

## 48. SceMQA: A Scientific College Entrance Level Multimodal Question Answering Benchmark

- [ ] SceMQA: A Scientific College Entrance Level Multimodal Question Answering Benchmark | https://aclanthology.org/2024.acl-short.11/

- **Link**: https://aclanthology.org/2024.acl-short.11/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The paper introduces SceMQA, a novel benchmark for scientific multimodal question answering at the college entrance level. It addresses a critical educational phase often overlooked in existing benchmarks, spanning high school to pre-college levels. SceMQA focuses on core science subjects including Mathematics, Physics, Chemistry, and Biology. It features a blend of multiple-choice and free-response formats, ensuring a comprehensive evaluation of AI models’ abilities. Additionally, our benchmark provides specific knowledge points for each problem and detailed explanations for each answer. SceMQA also uniquely presents problems with identical contexts but varied questions to facilitate a more thorough and accurate assessment of reasoning capabilities. In the experiment, we evaluate both open-source and close-source state-of-the-art Multimodal Large Language Models (MLLMs), across various experimental settings. The results show that further research and development are needed in developing more capable MLLM, as highlighted by only 50% to 60% accuracy achieved by the strongest models.

</details>

---

## 49. EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models

- [ ] EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models | https://aclanthology.org/2024.acl-short.33/

- **Link**: https://aclanthology.org/2024.acl-short.33/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent rapid development of Large Vision-Language Models (LVLMs) has indicated their potential for embodied tasks. However, the critical skill of spatial understanding in embodied environments has not been thoroughly evaluated, leaving the gap between current LVLMs and qualified embodied intelligence unknown. Therefore, we construct EmbSpatial-Bench, a benchmark for evaluating embodied spatial understanding of LVLMs. The benchmark is automatically derived from embodied scenes and covers 6 spatial relationships from an egocentric perspective. Experiments expose the insufficient capacity of current LVLMs (even GPT-4V). We further present EmbSpatial-SFT, an instruction-tuning dataset designed to improve LVLMs’ embodied spatial understanding.

</details>

---

## 50. MTP: A Dataset for Multi-Modal Turning Points in Casual Conversations

- [ ] MTP: A Dataset for Multi-Modal Turning Points in Casual Conversations | https://aclanthology.org/2024.acl-short.30/

- **Link**: https://aclanthology.org/2024.acl-short.30/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Detecting critical moments, such as emotional outbursts or changes in decisions during conversations, is crucial for understanding shifts in human behavior and their consequences. Our work introduces a novel problem setting focusing on these moments as turning points (TPs), accompanied by a meticulously curated, high-consensus, human-annotated multi-modal dataset. We provide precise timestamps, descriptions, and visual-textual evidence high-lighting changes in emotions, behaviors, perspectives, and decisions at these turning points. We also propose a framework, TPMaven, utilizing state-of-the-art vision-language models to construct a narrative from the videos and large language models to classify and detect turning points in our multi-modal dataset. Evaluation results show that TPMaven achieves an F1-score of 0.88 in classification and 0.61 in detection, with additional explanations aligning with human expectations.

</details>

---

## 51. Naming, Describing, and Quantifying Visual Objects in Humans andLLMs

- [ ] Naming, Describing, and Quantifying Visual Objects in Humans andLLMs | https://aclanthology.org/2024.acl-short.50/

- **Link**: https://aclanthology.org/2024.acl-short.50/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While human speakers use a variety of different expressions when describing the same object in an image, giving rise to a distribution of plausible labels driven by pragmatic constraints, the extent to which current Vision & Language Large Language Models (VLLMs) can mimic this crucial feature of language use is an open question. This applies to common, everyday objects, but it is particularly interesting for uncommon or novel objects for which a category label may be lacking or fuzzy. Furthermore, similar patterns of variation are observed among human speakers for highly context-sensitive expressions, such as the quantifiers ‘few’ or ‘most’. In our work, we evaluate VLLMs (FROMAGe, BLIP-2, LLaVA) on three categories (nouns, attributes, and quantifiers) where humans show great subjective variability concerning the distribution over plausible labels, using datasets and resources mostly under-explored in previous work. Our results reveal mixed evidence on the ability of VLLMs to capture human naming preferences at generation time: while some models are good at mimicking human distributions for nouns and attributes, all of them fail to assign quantifiers, a task that requires more accurate, high-level reasoning.

</details>

---

## 52. Cross-Modal Projection in MultimodalLLMs Doesn’t Really Project Visual Attributes to Textual Space

- [ ] Cross-Modal Projection in MultimodalLLMs Doesn’t Really Project Visual Attributes to Textual Space | https://aclanthology.org/2024.acl-short.60/

- **Link**: https://aclanthology.org/2024.acl-short.60/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) like LLaVA and GPT-4(V) enable general-purpose conversations about images with the language modality. As off-the-shelf MLLMs may have limited capabilities on images from domains like dermatology and agriculture, they must be fine-tuned to unlock domain-specific applications. The prevalent architecture of current open-source MLLMs comprises two major modules: an image-language (cross-modal) projection network and a large language model. It is desirable to understand the roles of these two modules in modeling domain-specific visual attributes to inform the design of future models and streamline the interpretability efforts on the current models. To this end, via experiments on 4 datasets and under 2 fine-tuning settings, we find that as the MLLM is fine-tuned, it indeed gains domain-specific visual capabilities, but the updates do not lead to the projection extracting relevant domain-specific visual attributes. Our results indicate that the domain-specific visual attributes are modeled by the LLM, even when only the projection is fine-tuned. Through this study, we offer a potential reinterpretation of the role of cross-modal projections in MLLM architectures.

</details>

---

## 53. Towards Artwork Explanation in Large-scale Vision Language Models

- [ ] Towards Artwork Explanation in Large-scale Vision Language Models | https://aclanthology.org/2024.acl-short.65/

- **Link**: https://aclanthology.org/2024.acl-short.65/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale Vision-Language Models (LVLMs) output text from images and instructions, demonstrating advanced capabilities in text generation and comprehension. However, it has not been clarified to what extent LVLMs understand the knowledge necessary for explaining images, the complex relationships between various pieces of knowledge, and how they integrate these understandings into their explanations. To address this issue, we propose a new task: the artwork explanation generation task, along with its evaluation dataset and metric for quantitatively assessing the understanding and utilization of knowledge about artworks. This task is apt for image description based on the premise that LVLMs are expected to have pre-existing knowledge of artworks, which are often subjects of wide recognition and documented information.It consists of two parts: generating explanations from both images and titles of artworks, and generating explanations using only images, thus evaluating the LVLMs’ language-based and vision-based knowledge.Alongside, we release a training dataset for LVLMs to learn explanations that incorporate knowledge about artworks.Our findings indicate that LVLMs not only struggle with integrating language and visual information but also exhibit a more pronounced limitation in acquiring knowledge from images alone. The datasets ExpArt=Explain Artworks are available at https://huggingface.co/datasets/naist-nlp/ExpArt

</details>

---

## 54. Don’t Buy it! Reassessing the Ad Understanding Abilities of Contrastive Multimodal Models

- [ ] Don’t Buy it! Reassessing the Ad Understanding Abilities of Contrastive Multimodal Models | https://aclanthology.org/2024.acl-short.77/

- **Link**: https://aclanthology.org/2024.acl-short.77/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image-based advertisements are complex multimodal stimuli that often contain unusual visual elements and figurative language. Previous research on automatic ad understanding has reported impressive zero-shot accuracy of contrastive vision-and-language models (VLMs) on an ad-explanation retrieval task. Here, we examine the original task setup and show that contrastive VLMs can solve it by exploiting grounding heuristics. To control for this confound, we introduce TRADE, a new evaluation test set with adversarial grounded explanations. While these explanations look implausible to humans, we show that they “fool” four different contrastive VLMs. Our findings highlight the need for an improved operationalisation of automatic ad understanding that truly evaluates VLMs’ multimodal reasoning abilities. We make our code and TRADE available at https://github.com/dmg-illc/trade.

</details>

---

## 55. MoExtend: Tuning New Experts for Modality and Task Extension

- [ ] MoExtend: Tuning New Experts for Modality and Task Extension | https://aclanthology.org/2024.acl-srw.53/

- **Link**: https://aclanthology.org/2024.acl-srw.53/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) excel in various tasks but are primarily trained on text data, limiting their application scope. Expanding LLM capabilities to include vision-language understanding is vital, yet training them on multimodal data from scratch is challenging and costly. Existing instruction tuning methods, e.g., LLAVA, often connects a pretrained CLIP vision encoder and LLMs via fully fine-tuning LLMs to bridge the modality gap. However, full fine-tuning is plagued by catastrophic forgetting, i.e., forgetting previous knowledge, and high training costs particularly in the era of increasing tasks and modalities. To solve this issue, we introduce MoExtend, an effective framework designed to streamline the modality adaptation and extension of Mixture-of-Experts (MoE) models. MoExtend seamlessly integrates new experts into pre-trained MoE models, endowing them with novel knowledge without the need to tune pretrained models such as MoE and vision encoders. This approach enables rapid adaptation and extension to new modal data or tasks, effectively addressing the challenge of accommodating new modalities within LLMs. Furthermore, MoExtend avoids tuning pretrained models, thus mitigating the risk of catastrophic forgetting. Experimental results demonstrate the efficacy and efficiency of MoExtend in enhancing the multimodal capabilities of LLMs, contributing to advancements in multimodal AI research.

</details>

---

## 56. Foundation Model for Biomedical Graphs: Integrating Knowledge Graphs and Protein Structures to Large Language Models

- [ ] Foundation Model for Biomedical Graphs: Integrating Knowledge Graphs and Protein Structures to Large Language Models | https://aclanthology.org/2024.acl-srw.58/

- **Link**: https://aclanthology.org/2024.acl-srw.58/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Transformer model has been a de-facto standard in natural language processing. Its adaptations in other fields such as computer vision showed promising results that this architecture is a powerful neural network in representation learning regardless of the data type. This recent success has led to research in multimodal Large Language Model (LLM), which enabled us to new types of tasks and applications with multiple data types. However, multimodal LLM in the biomedical domain is primarily limited to images, text, and/or sequence data. Here I propose to work on multimodal LLM architecture for biomedical graphs such as protein structure and chemical molecules. The research hypothesis is based on the fact that clinicians and researchers in computational biology and clinical research take advantage of various information for their decision-making process. Therefore, an AI model being able to handle multiple data types should boost its ability to use diverse knowledge for improved performances in clinical applications.

</details>

---

## 57. Vulnerabilities of Large Language Models to Adversarial Attacks

- [ ] Vulnerabilities of Large Language Models to Adversarial Attacks | https://aclanthology.org/2024.acl-tutorials.5/

- **Link**: https://aclanthology.org/2024.acl-tutorials.5/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This tutorial serves as a comprehensive guide on the vulnerabilities of Large Language Models (LLMs) to adversarial attacks, an interdisciplinary field that blends perspectives from Natural Language Processing (NLP) and Cybersecurity. As LLMs become more complex and integrated into various systems, understanding their security attributes is crucial. However, current research indicates that even safety-aligned models are not impervious to adversarial attacks that can result in incorrect or harmful outputs. The tutorial first lays the foundation by explaining safety-aligned LLMs and concepts in cybersecurity. It then categorizes existing research based on different types of learning architectures and attack methods. We highlight the existing vulnerabilities of unimodal LLMs, multi-modal LLMs, and systems that integrate LLMs, focusing on adversarial attacks designed to exploit weaknesses and mislead AI systems. Finally, the tutorial delves into the potential causes of these vulnerabilities and discusses potential defense mechanisms.

</details>

---

## 58. Vision Language Models for Spreadsheet Understanding: Challenges and Opportunities

- [ ] Vision Language Models for Spreadsheet Understanding: Challenges and Opportunities | https://aclanthology.org/2024.alvr-1.10/

- **Link**: https://aclanthology.org/2024.alvr-1.10/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper explores capabilities of Vision Language Models on spreadsheet comprehension. We propose three self-supervised challenges with corresponding evaluation metrics to comprehensively evaluate VLMs on Optical Character Recognition (OCR), spatial perception, and visual format recognition. Additionally, we utilize the spreadsheet table detection task to assess the overall performance of VLMs by integrating these challenges. To probe VLMs more finely, we propose three spreadsheet-to-image settings: column width adjustment, style change, and address augmentation. We propose variants of prompts to address the above tasks in different settings. Notably, to leverage the strengths of VLMs in understanding text rather than two-dimensional positioning, we propose to decode cell values on the four boundaries of the table in spreadsheet boundary detection. Our findings reveal that VLMs demonstrate promising OCR capabilities but produce unsatisfactory results due to cell omission and misalignment, and they notably exhibit insufficient spatial and format recognition skills, motivating future work to enhance VLMs’ spreadsheet data comprehension capabilities using our methods to generate extensive spreadsheet-image pairs in various settings.

</details>

---

## 59. Improving Vision-Language Cross-Lingual Transfer with Scheduled Unfreezing

- [ ] Improving Vision-Language Cross-Lingual Transfer with Scheduled Unfreezing | https://aclanthology.org/2024.alvr-1.13/

- **Link**: https://aclanthology.org/2024.alvr-1.13/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large-scale pretraining of vision-language (VL) models brought dramatic improvements across numerous tasks, from visual question-answering to cross-modal retrieval but these gains are mostly limited to English. Massively multilingual VL encoder models (mVLMs) hold promise for other languages: after fine-tuning on only English task data, they can perform the task in other languages in what is termed zero-shot cross-lingual transfer (ZS-XLT). Still, ZS-XLT sees a large performance gap to English, especially for low-resource languages. In this work, we reduce this gap with a fine-tuning strategy known asScheduled Unfreezing(SUF): instead of updating all parameters from the start, we begin with the top layer(s) of the vision-language encoder and gradually unfreeze (i.e., update) its layers top to bottom. SUF forces reliance on encoder’s representations from higher layers: the fact that in multilingual models these representations encode higher-level semantics rather than low-level language-specific idiosyncrasies, we hypothesize, should render SUF beneficial for ZS-XLT. Experiments with two mVLMs (UC2 & CCLM) on three downstream tasks (xGQA, XVNLI, xFlickrCo) show that SUF brings consistent gains in ZS-XLT, especially for visual Q&A (xGQA) by up to 10 points.

</details>

---

## 60. VerbCLIP: Improving Verb Understanding in Vision-Language Models with Compositional Structures

- [ ] VerbCLIP: Improving Verb Understanding in Vision-Language Models with Compositional Structures | https://aclanthology.org/2024.alvr-1.17/

- **Link**: https://aclanthology.org/2024.alvr-1.17/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Verbs describe the dynamics of interactions between people, objects, and their environments. They play a crucial role in language formation and understanding. Nonetheless, recent vision-language models like CLIP predominantly rely on nouns and have a limited account of verbs. This limitation affects their performance in tasks requiring action recognition and scene understanding. In this work, we introduce VerbCLIP, a verb-centric vision-language model which learns meanings of verbs based on a compositional approach to statistical machine learning. Our methods significantly outperform CLIP in zero-shot performance on the VALSE, VL-Checklist, and SVO-Probes datasets, with improvements of +2.38%, +3.14%, and +1.47%, without fine-tuning. Fine-tuning resulted in further improvements, with gains of +2.85% and +9.2% on the VALSE and VL-Checklist datasets.

</details>

---

## 61. Evolutionary Reward Design and Optimization with Multimodal Large Language Models

- [ ] Evolutionary Reward Design and Optimization with Multimodal Large Language Models | https://aclanthology.org/2024.alvr-1.18/

- **Link**: https://aclanthology.org/2024.alvr-1.18/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Designing reward functions is a pivotal yet challenging task for Reinforcement Learning (RL) practices, often demanding domain expertise and substantial effort. Recent studies have explored the utilization of Large Language Models (LLMs) to generate reward functions via evolutionary search techniques. However, these approaches overlook the potential of multimodal information, such as images and videos. In particular, prior methods predominantly rely on numerical feedback from the RL environment for doing evolution, neglecting the incorporation of visual data obtained during training. This study introduces a novel approach by employing Multimodal Large Language Models (MLLMs) to craft reward functions tailored for various RL tasks. The methodology involves providing MLLM with the RL environment’s code alongside its image as context and task information to generate reward candidates. Then, the chosen agent undergoes training, and the numerical feedback from the environment, along with the recorded video of the top-performing policy, is provided as feedback to the MLLM. By employing an iterative feedback mechanism through evolutionary search, MLLM consistently refines the reward function to maximize accuracy. Testing on two different agents points to the preeminence of our approach over previous methodology, which themselves outperformed 83% of reward functions designed by human experts.

</details>

---

## 62. mBLIP: Efficient Bootstrapping of Multilingual Vision-LLMs

- [ ] mBLIP: Efficient Bootstrapping of Multilingual Vision-LLMs | https://aclanthology.org/2024.alvr-1.2/

- **Link**: https://aclanthology.org/2024.alvr-1.2/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Modular vision-language models (Vision-LLMs) align pretrained image encoders with (frozen) large language models (LLMs) and post-hoc condition LLMs to ‘understand’ the image input. With the abundance of readily available high-quality English image-text data as well as strong monolingual English LLMs, the research focus has been on English-only Vision-LLMs. Multilingual vision-language models are still predominantly obtained via expensive end-to-end pretraining, resulting in comparatively smaller models, trained on limited multilingual image data supplemented with text-only multilingual corpora. We present mBLIP, the first Vision-LLM leveraging multilingual LLMs, which we obtain in a computationally efficient manner on consumer-level hardware. To this end, were-alignan image encoder previously tuned to an English LLM to a new, multilingual LLM using only a few million multilingual training examples derived from a mix of vision-and-language tasks, which we obtain by machine-translating high-quality English data to 95 languages. On the IGLUE benchmark and XM3600, mBLIP yields results competitive with state-of-the-art models and it greatly outperforms strong English-only Vision-LLMs like Llava 1.5. We release our model, code, and train data athttps://github.com/gregor-ge/mBLIP.

</details>

---

## 63. VideoCoT: A Video Chain-of-Thought Dataset with Active Annotation Tool

- [ ] VideoCoT: A Video Chain-of-Thought Dataset with Active Annotation Tool | https://aclanthology.org/2024.alvr-1.8/

- **Link**: https://aclanthology.org/2024.alvr-1.8/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are flourishing, but mainly focus on images with less attention than videos, especially in sub-fields such as prompt engineering, video chain-of-though (CoT), and instruction tuning on videos. Therefore, we try to explore the collection of CoT datasets in videos to lead to video OpenQA and improve the reasoning ability of MLLMs. Unfortunately, making such video CoT datasets is not an easy task. Given that human annotation is too cumbersome and expensive, while machine-generated is not reliable due to the hallucination issue, we develop an automatic annotation tool that combines machine and human experts, under the active learning paradigm. Active learning is an interactive strategy between the model and human experts, in this way, the workload of human labeling can be reduced and the quality of the dataset can be guaranteed. With the help of the automatic annotation tool, we strive to contribute three datasets, namely VideoCoT, TopicQA, TopicCoT. Furthermore, we propose a simple but effective benchmark based on the collected datasets, which exploits CoT to maximize the complex reasoning capabilities of MLLMs. Extensive experiments demonstrate the effectiveness our solution, and we will release our source codes and datasets to facilitate the research community.

</details>

---

## 64. Enhancing Conceptual Understanding in Multimodal Contrastive Learning through Hard Negative Samples

- [ ] Enhancing Conceptual Understanding in Multimodal Contrastive Learning through Hard Negative Samples | https://aclanthology.org/2024.alvr-1.9/

- **Link**: https://aclanthology.org/2024.alvr-1.9/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current vision-language models leveraging contrastive learning often face limitations in developing fine-grained conceptual understanding. This is due to random negative samples during pretraining, causing almost exclusively very dissimilar concepts to be compared in the loss function. Consequently, the models struggle with fine-grained semantic differences. To address this problem, we introduce a novel pretraining method incorporating synthetic hard negative text examples. The hard negatives replace terms corresponding to visual concepts, leading to a more fine-grained visual and textual concept alignment. Further, we introduce InpaintCOCO, a new challenging dataset for assessing the fine-grained alignment of colors, objects, and sizes in vision-language models. We created the dataset using generative inpainting from COCO images by changing the visual concepts so that the images no longer match their original captions. Our results show significant improvements in fine-grained concept understanding across various vision-language datasets, including our InpaintCOCO dataset.

</details>

---

## 65. Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models

- [ ] Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models | https://aclanthology.org/2024.alvr-1.4/

- **Link**: https://aclanthology.org/2024.alvr-1.4/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Object hallucination poses a significant challenge in vision-language (VL) models, often leading to the generation of nonsensical or unfaithful responses with non-existent objects. However, the absence of a general measurement for evaluating object hallucination in VL models has hindered our understanding and ability to mitigate this issue. In this work, we present NOPE (Negative Object Presence Evaluation), a novel benchmark designed to assess object hallucination in VL models through visual question answering (VQA). We propose a cost-effective and scalable approach utilizing large language models to generate 29.5k synthetic negative pronoun (NegP) data of high quality for NOPE. We extensively investigate the performance of 10 state-of-the-art VL models in discerning the non-existence of objects in visual questions, where the ground truth answers are denoted as (e.g., “none”). Additionally, we evaluate their standard performance on visual questions on 9 other VQA datasets. Through our experiments, we demonstrate that no VL model is immune to the vulnerability of object hallucination, as all models achieve accuracy below 10% onNegP. Furthermore, we uncover that lexically diverse visual questions, question types with large scopes, and scene-relevant objects capitalize the risk of object hallucination in VL models.

</details>

---

## 66. Dallah: A Dialect-Aware Multimodal Large Language Model forArabic

- [ ] Dallah: A Dialect-Aware Multimodal Large Language Model forArabic | https://aclanthology.org/2024.arabicnlp-1.27/

- **Link**: https://aclanthology.org/2024.arabicnlp-1.27/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements have significantly enhanced the capabilities of Multimodal Large Language Models (MLLMs) in generating and understanding image-to-text content. Despite these successes, progress is predominantly limited to English due to the scarcity of high-quality multimodal resources in other languages. This limitation impedes the development of competitive models in languages such as Arabic. To alleviate this situation, we introduce an efficient Arabic multimodal assistant, dubbed ***Dallah***, that utilizes an advanced language model based on LLaMA-2 to facilitate multimodal interactions. ***Dallah*** demonstrates state-of-the-art performance in Arabic MLLMs. Through fine-tuning six Arabic dialects, ***Dallah*** showcases its capability to handle complex dialectal interactions incorporating both textual and visual elements. The model excels in two benchmark tests: one evaluating its performance on Modern Standard Arabic (MSA) and another specifically designed to assess dialectal responses. Beyond its robust performance in multimodal interaction tasks, ***Dallah*** has the potential to pave the way for further development of dialect-aware Arabic MLLMs.

</details>

---

## 67. XrayGPT: Chest Radiographs Summarization using Large Medical Vision-Language Models

- [ ] XrayGPT: Chest Radiographs Summarization using Large Medical Vision-Language Models | https://aclanthology.org/2024.bionlp-1.35/

- **Link**: https://aclanthology.org/2024.bionlp-1.35/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The latest breakthroughs in large language models (LLMs) and vision-language models (VLMs) have showcased promising capabilities toward performing a wide range of tasks. Such models are typically trained on massive datasets comprising billions of image-text pairs with diverse tasks. However, their performance on task-specific domains, such as radiology, is still under-explored. While few works have recently explored LLMs-based conversational medical models, they mainly focus on text-based analysis. In this paper, we introduce XrayGPT, a conversational medical vision-language (VLMs) model that can analyze and answer open-ended questions about chest radiographs. Specifically, we align both medical visual encoder with a fine-tuned LLM to possess visual conversation abilities, grounded in an understanding of radiographs and medical knowledge. For improved alignment of chest radiograph data, we generate ~217k interactive and high-quality summaries from free-text radiology reports. Extensive experiments are conducted to validate the merits of XrayGPT. To conduct an expert evaluation, certified medical doctors evaluated the output of our XrayGPT on a test subset and the results reveal that more than 70% of the responses are scientifically accurate, with an average score of 4/5. We hope our simple and effective method establishes a solid baseline, facilitating future research toward automated analysis and summarization of chest radiographs. Code, models, and instruction sets will be publicly released.

</details>

---

## 68. Optimizing Multimodal Large Language Models for Detection of Alcohol Advertisements via Adaptive Prompting

- [ ] Optimizing Multimodal Large Language Models for Detection of Alcohol Advertisements via Adaptive Prompting | https://aclanthology.org/2024.bionlp-1.42/

- **Link**: https://aclanthology.org/2024.bionlp-1.42/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Adolescents exposed to advertisements promoting addictive substances exhibit a higher likelihood of subsequent substance use. The predominant source for youth exposure to such advertisements is through online content accessed via smartphones. Detecting these advertisements is crucial for establishing and maintaining a safer online environment for young people. In our study, we utilized Multimodal Large Language Models (MLLMs) to identify addictive substance advertisements in digital media. The performance of MLLMs depends on the quality of the prompt used to instruct the model. To optimize our prompts, an adaptive prompt engineering approach was implemented, leveraging a genetic algorithm to refine and enhance the prompts. To evaluate the model’s performance, we augmented the RICO dataset, consisting of Android user interface screenshots, by superimposing alcohol ads onto them. Our results indicate that the MLLM can detect advertisements promoting alcohol with a 0.94 accuracy and a 0.94 F1 score.

</details>

---

## 69. iHealth-Chile-1 atRRG24: In-context Learning and Finetuning of a Large Multimodal Model for Radiology Report Generation

- [ ] iHealth-Chile-1 atRRG24: In-context Learning and Finetuning of a Large Multimodal Model for Radiology Report Generation | https://aclanthology.org/2024.bionlp-1.52/

- **Link**: https://aclanthology.org/2024.bionlp-1.52/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents the approach of the iHealth-Chile-1 team for the shared task of Large-Scale Radiology Report Generation at the BioNLP workshop, inspired by progress in large multimodal models for processing images and text. In this work, we leverage LLaVA, a Visual-Language Model (VLM), composed of a vision-encoder, a vision-language connector or adapter, and a large language model able to process text and visual embeddings. We achieve our best result by enriching the input prompt of LLaVA with the text output of a simpler report generation model. With this enriched-prompt technique, we improve our results in 4 of 5 metrics (BLEU-4, Rouge-L, BertScore and F1-RadGraph,), only doing in-context learning. Moreover, we provide details about different architecture settings, fine-tuning strategies, and dataset configurations.

</details>

---

## 70. SICARatRRG2024:GPUPoor’s Guide to Radiology Report Generation

- [ ] SICARatRRG2024:GPUPoor’s Guide to Radiology Report Generation | https://aclanthology.org/2024.bionlp-1.55/

- **Link**: https://aclanthology.org/2024.bionlp-1.55/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Radiology report generation (RRG) aims to create free-text radiology reports from clinical imaging. Our solution employs a lightweight multimodal language model (MLLM) enhanced with a two-stage post-processing strategy, utilizing a Large Language Model (LLM) to boost diagnostic accuracy and ensure patient safety. We introduce the “First, Do No Harm” SafetyNet, which incorporates Xraydar, an advanced X-ray classification model, to cross-verify the model outputs and specifically address false negatives from the MLLM. This comprehensive approach combines the efficiency of lightweight models with the robustness of thorough post-processing techniques, offering a reliable solution for radiology report generation. Our system achieved fourth place on the F1-Radgraph metric for findings generation in the Radiology Report Generation Shared Task (RRG24).

</details>

---

## 71. e-HealthCSIROatRRG24: Entropy-Augmented Self-Critical Sequence Training for Radiology Report Generation

- [ ] e-HealthCSIROatRRG24: Entropy-Augmented Self-Critical Sequence Training for Radiology Report Generation | https://aclanthology.org/2024.bionlp-1.8/

- **Link**: https://aclanthology.org/2024.bionlp-1.8/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The core novelty of our approach lies in the addition of entropy regularisation to self-critical sequence training. This helps maintain a higher entropy in the token distribution, preventing overfitting to common phrases and ensuring a broader exploration of the vocabulary during training, which is essential for handling the diversity of the radiology reports in the RRG24 datasets. We apply this to a multimodal language model with RadGraph as the reward. Additionally, our model incorporates several other aspects. We use token type embeddings to differentiate between findings and impression section tokens, as well as image embeddings. To handle missing sections, we employ special tokens. We also utilise an attention mask with non-causal masking for the image embeddings and a causal mask for the report token embeddings.

</details>

---

## 72. What does Kiki look like? Cross-modal associations between speech sounds and visual shapes in vision-and-language models

- [ ] What does Kiki look like? Cross-modal associations between speech sounds and visual shapes in vision-and-language models | https://aclanthology.org/2024.cmcl-1.17/

- **Link**: https://aclanthology.org/2024.cmcl-1.17/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Humans have clear cross-modal preferences when matching certain novel words to visual shapes. Evidence suggests that these preferences play a prominent role in our linguistic processing, language learning, and the origins of signal-meaning mappings. With the rise of multimodal models in AI, such as vision-and-language (VLM) models, it becomes increasingly important to uncover the kinds of visio-linguistic associations these models encode and whether they align with human representations. Informed by experiments with humans, we probe and compare four VLMs for a well-known human cross-modal preference, the bouba-kiki effect. We do not find conclusive evidence for this effect but suggest that results may depend on features of the models, such as architecture design, model size, and training details. Our findings inform discussions on the origins of the bouba-kiki effect in human cognition and future developments of VLMs that align well with human cross-modal associations.

</details>

---

## 73. Evaluating Semantic Relations in Predicting Textual Labels for Images of Abstract and Concrete Concepts

- [ ] Evaluating Semantic Relations in Predicting Textual Labels for Images of Abstract and Concrete Concepts | https://aclanthology.org/2024.cmcl-1.18/

- **Link**: https://aclanthology.org/2024.cmcl-1.18/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study investigates the performance of SigLIP, a state-of-the-art Vision-Language Model (VLM), in predicting labels for images depicting 1,278 concepts. Our analysis across 300 images per concept shows that the model frequently predicts the exact user-tagged labels, but similarly, it often predicts labels that are semantically related to the exact labels in various ways: synonyms, hypernyms, co-hyponyms, and associated words, particularly for abstract concepts. We then zoom into the diversity of the user tags of images and word associations for abstract versus concrete concepts. Surprisingly, not only abstract but also concrete concepts exhibit significant variability, thus challenging the traditional view that representations of concrete concepts are less diverse.

</details>

---

## 74. Evaluating Vision-Language Models on Bistable Images

- [ ] Evaluating Vision-Language Models on Bistable Images | https://aclanthology.org/2024.cmcl-1.2/

- **Link**: https://aclanthology.org/2024.cmcl-1.2/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Bistable images, also known as ambiguous or reversible images, present visual stimuli that can be seen in two distinct interpretations, though not simultaneously, by the observer. In this study, we conduct the most extensive examination of vision-language models using bistable images to date. We manually gathered a dataset of 29 bistable images, along with their associated labels, and subjected them to 121 different manipulations in brightness, resolution, tint, and rotation. We evaluated twelve different models in both classification and generative tasks across six model architectures. Our findings reveal that, with the exception of models from the Idefics family and LLaVA1.5-13b, there is a pronounced preference for one interpretation over another among the models, and minimal variance under image manipulations, with few exceptions on image rotations. Additionally, we compared the models’ preferences with humans, noting that the models do not exhibit the same continuity biases as humans and often diverge from human initial interpretations. We also investigated the influence of variations in prompts and the use of synonymous labels, discovering that these factors significantly affect model interpretations more than image manipulations showing a higher influence of the language priors on bistable image interpretations compared to image-text training data. All code and data is open sourced.

</details>

---

## 75. VALOR-EVAL: Holistic Coverage and Faithfulness Evaluation of Large Vision-Language Models

- [ ] VALOR-EVAL: Holistic Coverage and Faithfulness Evaluation of Large Vision-Language Models | https://aclanthology.org/2024.findings-acl.105/

- **Link**: https://aclanthology.org/2024.findings-acl.105/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) suffer from hallucination issues, wherein the models generate plausible-sounding but factually incorrect outputs, undermining their reliability. A comprehensive quantitative evaluation is necessary to identify and understand the extent of hallucinations in these models. However, existing benchmarks are often limited in scope, focusing mainly on object hallucinations. Furthermore, current evaluation methods struggle to effectively address the subtle semantic distinctions between model outputs and reference data, as well as the balance between hallucination and informativeness. To address these issues, we introduce a multi-dimensional benchmark covering objects, attributes, and relations, with challenging images selected based on associative biases. Moreover, we propose a large language model (LLM)-based two-stage evaluation framework that generalizes the popular CHAIR metric and incorporates both faithfulness and coverage into the evaluation. Experiments on 10 established LVLMs demonstrate that our evaluation metric is more comprehensive and better correlated with humans than existing work when evaluating on our challenging human-annotated benchmark dataset. Our work also highlights the critical balance between faithfulness and coverage of model outputs, and encourages future works to address hallucinations in LVLMs while keeping their outputs informative.

</details>

---

## 76. SoMeLVLM: A Large Vision Language Model for Social Media Processing

- [ ] SoMeLVLM: A Large Vision Language Model for Social Media Processing | https://aclanthology.org/2024.findings-acl.140/

- **Link**: https://aclanthology.org/2024.findings-acl.140/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The growth of social media, characterized by its multimodal nature, has led to the emergence of diverse phenomena and challenges, which calls for an effective approach to uniformly solve automated tasks. The powerful Large Vision Language Models make it possible to handle a variety of tasks simultaneously, but even with carefully designed prompting methods, the general domain models often fall short in aligning with the unique speaking style and context of social media tasks. In this paper, we introduce a Large Vision Language Model for Social Media Processing (SoMeLVLM), which is a cognitive framework equipped with five key capabilities including knowledge & comprehension, application, analysis, evaluation, and creation. SoMeLVLM is designed to understand and generate realistic social media behavior. We have developed a 654k multimodal social media instruction-tuning dataset to support our cognitive framework and fine-tune our model. Our experiments demonstrate that SoMeLVLM achieves state-of-the-art performance in multiple social media tasks. Further analysis shows its significant advantages over baselines in terms of cognitive abilities.

</details>

---

## 77. Episodic Memory Retrieval fromLLMs: A Neuromorphic Mechanism to Generate Commonsense Counterfactuals for Relation Extraction

- [ ] Episodic Memory Retrieval fromLLMs: A Neuromorphic Mechanism to Generate Commonsense Counterfactuals for Relation Extraction | https://aclanthology.org/2024.findings-acl.146/

- **Link**: https://aclanthology.org/2024.findings-acl.146/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have achieved satisfactory performance in counterfactual generation. However, confined by the stochastic generation process of LLMs, there often are misalignments between LLMs and humans which hinder LLMs from handling complex tasks like relation extraction. As a result, LLMs may generate commonsense-violated counterfactuals like ‘eggs were produced by a box’. To bridge this gap, we propose to mimick the episodic memory retrieval, the working mechanism of human hippocampus, to align LLMs’ generation process with that of humans. In this way, LLMs can derive experience from their extensive memory, which keeps in line with the way humans gain commonsense. We then implement two central functions in the hippocampus, i.e., pattern separation and pattern completion, to retrieve the episodic memory from LLMs and generate commonsense counterfactuals for relation extraction. Experimental results demonstrate the improvements of our framework over existing methods in terms of the quality of counterfactuals.

</details>

---

## 78. LANS: A Layout-Aware Neural Solver for Plane Geometry Problem

- [ ] LANS: A Layout-Aware Neural Solver for Plane Geometry Problem | https://aclanthology.org/2024.findings-acl.153/

- **Link**: https://aclanthology.org/2024.findings-acl.153/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Geometry problem solving (GPS) is a challenging mathematical reasoning task requiring multi-modal understanding, fusion, and reasoning. Existing neural solvers take GPS as a vision-language task but are short in the representation of geometry diagrams that carry rich and complex layout information. In this paper, we propose a layout-aware neural solver named LANS, integrated with two new modules: multimodal layout-aware pre-trained language module (MLA-PLM) and layout-aware fusion attention (LA-FA). MLA-PLM adopts structural-semantic pre-training (SSP) to implement global relationship modeling, and point-match pre-training (PMP) to achieve alignment between visual points and textual points. LA-FA employs a layout-aware attention mask to realize point-guided cross-modal fusion for further boosting layout awareness of LANS. Extensive experiments on datasets Geometry3K and PGPS9K validate the effectiveness of the layout-aware modules and superior problem-solving performance of our LANS solver, over existing symbolic and neural solvers. We have made our code and data publicly available.

</details>

---

## 79. Red Teaming Visual Language Models

- [ ] Red Teaming Visual Language Models | https://aclanthology.org/2024.findings-acl.198/

- **Link**: https://aclanthology.org/2024.findings-acl.198/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

VLMs (Vision-Language Models) extend the capabilities of LLMs (Large Language Models) to accept multimodal inputs. Since it has been verified that LLMs can be induced to generate harmful or inaccurate content through specific test cases (termed as Red Teaming), how VLMs perform in similar scenarios, especially with their combination of textual and visual inputs, remains a question. To explore this problem, we present a novel red teaming dataset RTVLM, which encompasses 12 subtasks (e.g., image misleading, multi-modal jailbreaking, face fairness, etc) under 4 primary aspects (faithfulness, privacy, safety, fairness). Our RTVLM is the first red teaming dataset to benchmark current VLMs in terms of these 4 different aspects. Detailed analysis shows that 10 prominent open-sourced VLMs struggle with the red teaming in different degrees and have up to 31% performance gap with GPT-4V. Additionally, we simply apply red teaming alignment to LLaVA-v1.5 with Supervised Fine-tuning (SFT) using RTVLM, and this bolsters the models’ performance with 10% in RTVLM test set, 13% in MM-hallu, and without noticeable decline in MM-Bench, overpassing other LLaVA-based models in similar size with regular alignment data. This reveals that current open-sourced VLMs still lack red teaming alignment. Our code and datasets will be open-sourced.

</details>

---

## 80. ImplicitAVE: An Open-Source Dataset and MultimodalLLMs Benchmark for Implicit Attribute Value Extraction

- [ ] ImplicitAVE: An Open-Source Dataset and MultimodalLLMs Benchmark for Implicit Attribute Value Extraction | https://aclanthology.org/2024.findings-acl.20/

- **Link**: https://aclanthology.org/2024.findings-acl.20/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing datasets for attribute value extraction (AVE) predominantly focus on explicit attribute values while neglecting the implicit ones, lack product images, are often not publicly available, and lack an in-depth human inspection across diverse domains. To address these limitations, we present ImplicitAVE, the first, publicly available multimodal dataset for implicit attribute value extraction. ImplicitAVE, sourced from the MAVE dataset, is carefully curated and expanded to include implicit AVE and multimodality, resulting in a refined dataset of 68k training and 1.6k testing data across five domains. We also explore the application of multimodal large language models (MLLMs) to implicit AVE, establishing a comprehensive benchmark for MLLMs on the ImplicitAVE dataset. Six recent MLLMs with eleven variants are evaluated across diverse settings, revealing that implicit value extraction remains a challenging task for MLLMs. The contributions of this work include the development and release of ImplicitAVE, and the exploration and benchmarking of various MLLMs for implicit AVE, providing valuable insights and potential future research directions. Dataset and code are available at https://github.com/HenryPengZou/ImplicitAVE.

</details>

---

## 81. Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives

- [ ] Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives | https://aclanthology.org/2024.findings-acl.217/

- **Link**: https://aclanthology.org/2024.findings-acl.217/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Humans use multiple senses to comprehend the environment. Vision and language are two of the most vital senses since they allow us to easily communicate our thoughts and perceive the world around us. There has been a lot of interest in creating video-language understanding systems with human-like senses since a video-language pair can mimic both our linguistic medium and visual environment with temporal dynamics. In this survey, we review the key tasks of these systems and highlight the associated challenges. Based on the challenges, we summarize their methods from model architecture, model training, and data perspectives. We also conduct performance comparison among the methods, and discuss promising directions for future research.

</details>

---

## 82. Tables as Texts or Images: Evaluating the Table Reasoning Ability ofLLMs andMLLMs

- [ ] Tables as Texts or Images: Evaluating the Table Reasoning Ability ofLLMs andMLLMs | https://aclanthology.org/2024.findings-acl.23/

- **Link**: https://aclanthology.org/2024.findings-acl.23/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Tables contrast with unstructured text data by its structure to organize the information.In this paper, we investigate the efficiency of various LLMs in interpreting tabular data through different prompting strategies and data formats. Our analysis extends across six benchmarks for table-related tasks such as question-answering and fact-checking. We pioneer in the assessment of LLMs’ performance on image-based table representation. Specifically, we compare five text-based and three image-based table representations, revealing the influence of representation and prompting on LLM performance. We hope our study provides researchers insights into optimizing LLMs’ application in table-related tasks.

</details>

---

## 83. Autonomous Workflow for Multimodal Fine-Grained Training Assistants Towards Mixed Reality

- [ ] Autonomous Workflow for Multimodal Fine-Grained Training Assistants Towards Mixed Reality | https://aclanthology.org/2024.findings-acl.240/

- **Link**: https://aclanthology.org/2024.findings-acl.240/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Autonomous artificial intelligence (AI) agents have emerged as promising protocols for automatically understanding the language-based environment, particularly with the exponential development of large language models (LLMs). However, a fine-grained, comprehensive understanding of multimodal environments remains under-explored. This work designs an autonomous workflow tailored for integrating AI agents seamlessly into extended reality (XR) applications for fine-grained training. We present a demonstration of a multimodal fine-grained training assistant for LEGO brick assembly in a pilot XR environment. Specifically, we design a cerebral language agent that integrates LLM with memory, planning, and interaction with XR tools and a vision-language agent, enabling agents to decide their actions based on past experiences. Furthermore, we introduce LEGO-MRTA, a multimodal fine-grained assembly dialogue dataset synthesized automatically in the workflow served by a commercial LLM. This dataset comprises multimodal instruction manuals, conversations, XR responses, and vision question answering. Last, we present several prevailing open-resource LLMs as benchmarks, assessing their performance with and without fine-tuning on the proposed dataset. We anticipate that the broader impact of this workflow will advance the development of smarter assistants for seamless user interaction in XR environments, fostering research in both AI and HCI communities.

</details>

---

## 84. Your Vision-Language Model Itself Is a Strong Filter: Towards High-Quality Instruction Tuning with Data Selection

- [ ] Your Vision-Language Model Itself Is a Strong Filter: Towards High-Quality Instruction Tuning with Data Selection | https://aclanthology.org/2024.findings-acl.246/

- **Link**: https://aclanthology.org/2024.findings-acl.246/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Data selection in instruction tuning emerges as a pivotal process for acquiring high-quality data and training instruction-following large language models (LLMs), but it is still a new and unexplored research area for vision-language models (VLMs). Existing data selection approaches on LLMs either rely on single unreliable scores, or use downstream tasks for selection, which is time-consuming and can lead to potential over-fitting on the chosen evaluation datasets. To address this challenge, we introduce a novel dataset selection method, Self-Filter, that utilizes the VLM itself as a filter. This approach is inspired by the observation that VLMs benefit from training with the most challenging instructions. Self-Filter operates in two stages. In the first stage, we devise a scoring network to evaluate the difficulty of training instructions, which is co-trained with the VLM. In the second stage, we use the trained score net to measure the difficulty of each instruction, select the most challenging samples, and penalize similar samples to encourage diversity. Comprehensive experiments on LLaVA and MiniGPT-4 show that Self-Filter can reach better results compared to full data settings with merely about 15% samples, and can achieve superior performance against competitive baselines.

</details>

---

## 85. Multi-Modal Retrieval For Large Language Model Based Speech Recognition

- [ ] Multi-Modal Retrieval For Large Language Model Based Speech Recognition | https://aclanthology.org/2024.findings-acl.262/

- **Link**: https://aclanthology.org/2024.findings-acl.262/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Retrieval is a widely adopted approach for improving language models leveraging external information. As the field moves towards multi-modal large language models, it is important to extend the pure text based methods to incorporate other modalities in retrieval as well for applications across the wide spectrum of machine learning tasks and data types. In this work, we propose multi-modal retrieval with two approaches: kNN-LM and cross-attention techniques. We demonstrate the effectiveness of our retrieval approaches empirically by applying them to automatic speech recognition tasks with access to external information. Under this setting, we show that speech-based multi-modal retrieval outperforms text based retrieval, and yields up to improvement in word error rate over the multi-modal language model baseline. Furthermore, we achieve state-of-the-art recognition results on the Spoken-Squad question answering dataset.

</details>

---

## 86. InfiMM: Advancing Multimodal Understanding with an Open-Sourced Visual Language Model

- [ ] InfiMM: Advancing Multimodal Understanding with an Open-Sourced Visual Language Model | https://aclanthology.org/2024.findings-acl.27/

- **Link**: https://aclanthology.org/2024.findings-acl.27/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this work, we present InfiMM, an advanced Multimodal Large Language Model that adapts to intricate vision-language tasks. InfiMM, inspired by the Flamingo architecture, distinguishes itself through the utilization of large-scale training data, comprehensive training strategies, and diverse large language models. This approach ensures the preservation of Flamingo’s foundational strengths while simultaneously introducing augmented capabilities. Empirical evaluations across a variety of benchmarks underscore InfiMM’s remarkable capability in multimodal understanding. The code can be found at: https://anonymous.4open.science/r/infimm-zephyr-F60C/.

</details>

---

## 87. On the Language Encoder of Contrastive Cross-modal Models

- [ ] On the Language Encoder of Contrastive Cross-modal Models | https://aclanthology.org/2024.findings-acl.293/

- **Link**: https://aclanthology.org/2024.findings-acl.293/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastive cross-modal models such as CLIP and CLAP aid various vision-language (VL) and audio-language (AL) tasks. However, there has been limited investigation of and improvement in their language encoder – the central component of encoding natural language descriptions of image/audio into vector representations. We extensively evaluate how unsupervised and supervised sentence embedding training affect language encoder quality and cross-modal task performance. In VL pretraining, we found that sentence embedding training enhances language encoder quality and aids in cross-modal tasks, improving contrastive VL models such as CyCLIP. Sentence embedding training benefits AL tasks when the amount of training data is large. We analyze the representation spaces to understand the strengths of sentence embedding training, and find that it improves text-space uniformity, at the cost of decreased cross-modal alignment.

</details>

---

## 88. MLeVLM: Improve Multi-level Progressive Capabilities based on Multimodal Large Language Model for Medical Visual Question Answering

- [ ] MLeVLM: Improve Multi-level Progressive Capabilities based on Multimodal Large Language Model for Medical Visual Question Answering | https://aclanthology.org/2024.findings-acl.296/

- **Link**: https://aclanthology.org/2024.findings-acl.296/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Medical visual question answering (MVQA) requires in-depth understanding of medical images and questions to provide reliable answers. We summarize multi-level progressive capabilities that models need to focus on in MVQA: recognition, details, diagnosis, knowledge, and reasoning. Existing MVQA models tend to ignore the above capabilities due to unspecific data and plain architecture. To address these issues, this paper proposes Multi-level Visual Language Model (MLeVLM) for MVQA. On the data side, we construct a high-quality multi-level instruction dataset MLe-VQA via GPT-4, which covers multi-level questions and answers as well as reasoning processes from visual clues to semantic cognition. On the architecture side, we propose a multi-level feature alignment module, including attention-based token selector and context merger, which can efficiently align features at different levels from visual to semantic. To better evaluate the model’s capabilities, we manually construct a multi-level MVQA evaluation benchmark named MLe-Bench. Extensive experiments demonstrate the effectiveness of our constructed multi-level instruction dataset and the multi-level feature alignment module. It also proves that MLeVLM outperforms existing medical multimodal large language models.

</details>

---

## 89. MIKE: A New Benchmark for Fine-grained Multimodal Entity Knowledge Editing

- [ ] MIKE: A New Benchmark for Fine-grained Multimodal Entity Knowledge Editing | https://aclanthology.org/2024.findings-acl.298/

- **Link**: https://aclanthology.org/2024.findings-acl.298/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal knowledge editing represents a critical advancement in enhancing the capabilities of Multimodal Large Language Models (MLLMs). Despite its potential, current benchmarks predominantly focus on coarse-grained knowledge, leaving the intricacies of fine-grained (FG) multimodal entity knowledge largely unexplored. This gap presents a notable challenge, as FG entity recognition is pivotal for the practical deployment and effectiveness of MLLMs in diverse real-world scenarios. To bridge this gap, we introduce MIKE, a comprehensive benchmark and dataset specifically designed for the FG multimodal entity knowledge editing. MIKE encompasses a suite of tasks tailored to assess different perspectives, including Vanilla Name Answering, Entity-Level Caption, and Complex-Scenario Recognition. In addition, a new form of knowledge editing, Multi-step Editing, is introduced to evaluate the editing efficiency. Through our extensive evaluations, we demonstrate that the current state-of-the-art methods face significant challenges in tackling our proposed benchmark, underscoring the complexity of FG knowledge editing in MLLMs. Our findings spotlight the urgent need for novel approaches in this domain, setting a clear agenda for future research and development efforts within the community.

</details>

---

## 90. Mitigating Hallucinations in Large Vision-Language Models (LVLMs) via Language-Contrastive Decoding (LCD)

- [ ] Mitigating Hallucinations in Large Vision-Language Models (LVLMs) via Language-Contrastive Decoding (LCD) | https://aclanthology.org/2024.findings-acl.359/

- **Link**: https://aclanthology.org/2024.findings-acl.359/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) are an extension of Large Language Models (LLMs) that facilitate processing both image and text inputs, expanding AI capabilities. However, LVLMs struggle with object hallucinations due to their reliance on text cues and learned object co-occurrence biases. While most research quantifies these hallucinations, mitigation strategies are still lacking. Our study introduces a Language Contrastive Decoding (LCD) algorithm that adjusts LVLM outputs based on LLM distribution confidence levels, effectively reducing object hallucinations. We demonstrate the advantages of LCD in leading LVLMs, showing up to %4 improvement in POPE F1 scores and up to %36 reduction in CHAIR scores on the COCO validation set, while also improving captioning quality scores. Our method effectively improves LVLMs without needing complex post-processing or retraining, and is easily applicable to different models. Our findings highlight the potential of further exploration of LVLM-specific decoding algorithms.

</details>

---

## 91. Exploring Spatial Schema Intuitions in Large Language and Vision Models

- [ ] Exploring Spatial Schema Intuitions in Large Language and Vision Models | https://aclanthology.org/2024.findings-acl.365/

- **Link**: https://aclanthology.org/2024.findings-acl.365/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the ubiquity of large language models (LLMs) in AI research, the question of embodiment in LLMs remains underexplored, distinguishing them from embodied systems in robotics where sensory perception directly informs physical action.Our investigation navigates the intriguing terrain of whether LLMs, despite their non-embodied nature, effectively capture implicit human intuitions about fundamental, spatial building blocks of language. We employ insights from spatial cognitive foundations developed through early sensorimotor experiences, guiding our exploration through the reproduction of three psycholinguistic experiments. Surprisingly, correlations between model outputs and human responses emerge, revealing adaptability without a tangible connection to embodied experiences. Notable distinctions include polarized language model responses and reduced correlations in vision language models. This research contributes to a nuanced understanding of the interplay between language, spatial experiences, and the computations made by large language models.Project Website: https://cisnlp.github.io/Spatial_Schemas/

</details>

---

## 92. Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models

- [ ] Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models | https://aclanthology.org/2024.findings-acl.37/

- **Link**: https://aclanthology.org/2024.findings-acl.37/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in large language models (LLMs) have promoted generative error correction (GER) for automatic speech recognition (ASR), which aims to predict the ground-truth transcription from the decoded N-best hypotheses. Thanks to the strong language generation ability of LLMs and rich information in the N-best list, GER shows great effectiveness in enhancing ASR results. However, it still suffers from two limitations: 1) LLMs are unaware of the source speech during GER, which may lead to results that are grammatically correct but violate the source speech content, 2) N-best hypotheses usually only vary in a few tokens, making it redundant to send all of them for GER, which could confuse LLM about which tokens to focus on and thus lead to increased miscorrection. In this paper, we propose ClozeGER, a new paradigm for ASR generative error correction. First, we introduce a multimodal LLM (i.e., SpeechGPT) to receive source speech as extra input to improve the fidelity of correction output. Then, we reformat GER as a cloze test with logits calibration to remove the input information redundancy and simplify GER with clear instructions. Experiments show that ClozeGER achieves a new breakthrough over vanilla GER on 9 popular ASR datasets.

</details>

---

## 93. MM-SOC: Benchmarking Multimodal Large Language Models in Social Media Platforms

- [ ] MM-SOC: Benchmarking Multimodal Large Language Models in Social Media Platforms | https://aclanthology.org/2024.findings-acl.370/

- **Link**: https://aclanthology.org/2024.findings-acl.370/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Social media platforms are hubs for multimodal information exchange, encompassing text, images, and videos, making it challenging for machines to comprehend the information or emotions associated with interactions in online spaces. Multimodal Large Language Models (MLLMs) have emerged as a promising solution to address these challenges, yet struggle with accurately interpreting human emotions and complex contents like misinformation. This paper introduces MM-Soc, a comprehensive benchmark designed to evaluate MLLMs’ understanding of multimodal social media content. MM-Soc compiles prominent multimodal datasets and incorporates a novel large-scale YouTube tagging dataset, targeting a range of tasks from misinformation detection, hate speech detection, and social context generation. Through our exhaustive evaluation on ten size-variants of four open-source MLLMs, we have identified significant performance disparities, highlighting the need for advancements in models’ social understanding capabilities. Our analysis reveals that, in a zero-shot setting, various types of MLLMs generally exhibit difficulties in handling social media tasks. However, MLLMs demonstrate performance improvements post fine-tuning, suggesting potential pathways for improvement.

</details>

---

## 94. DoLVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning

- [ ] DoLVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning | https://aclanthology.org/2024.findings-acl.41/

- **Link**: https://aclanthology.org/2024.findings-acl.41/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Advances in large vision-language models (LVLMs) have led to significant progress in generating natural language descriptions for visual contents. These powerful models are known for producing texts that are factually inconsistent with the visual input. While some efforts mitigate such inconsistencies in natural image captioning, the factuality of generated captions for structured visuals, such as charts, has not received as much scrutiny. This work introduces a comprehensive typology of factual errors in generated chart captions. A large-scale human annotation effort provides insight into the error patterns in captions generated by various models, ultimately forming the foundation of a dataset, CHOCOLATE. Our analysis reveals that even advanced models like GPT-4V frequently produce captions laced with factual inaccuracies. To combat this, we establish the task of Chart Caption Factual Error Correction and introduce CHARTVE, a visual entailment model that outperforms current LVLMs in evaluating caption factuality. Furthermore, we propose C2TFEC, an interpretable two-stage framework that excels at correcting factual errors. This work inaugurates a new domain in factual error correction for chart captions, presenting a novel evaluation metric, and demonstrating an effective approach to ensuring the factuality of generated chart captions. The code and data as well as the continuously updated benchmark can be found at: https://khuangaf.github.io/CHOCOLATE/.

</details>

---

## 95. RAP: Efficient Text-Video Retrieval with Sparse-and-Correlated Adapter

- [ ] RAP: Efficient Text-Video Retrieval with Sparse-and-Correlated Adapter | https://aclanthology.org/2024.findings-acl.427/

- **Link**: https://aclanthology.org/2024.findings-acl.427/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Text-Video Retrieval (TVR) aims to align relevant video content with natural language queries. To date, most of the state-of-the-art TVR methods learn image-to-video transfer learning based on the large-scale pre-trained vision-language models (e.g., CLIP). However, fully fine-tuning these pre-trained models for TVR incurs prohibitively expensive computation cost. To this end, we propose to conduct efficient text-video Retrieval with a salient-and-correlated AdaPter (RAP), i.e., fine-tuning the pre-trained model with a few parameterized layers. To accommodate the text-video scenario, we equip our RAP with two indispensable characteristics including temporal sparsity and correlation. Specifically, we propose a low-rank modulation module to refine the per-image features from frozen CLIP backbone, which accentuates silent frames within the video features while alleviating temporal redundancy. Besides, we introduce an asynchronous self-attention mechanism which firstly selects top responsive visual patch and augments the correlation modeling between them with learnable temporal and patch offsets. Extensive experiments on four TVR datasets demonstrate that our RAP achieves superior or comparable performance compared to the fully fine-tuned counterpart and other parameter-efficient finetuning methods.

</details>

---

## 96. BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models

- [ ] BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models | https://aclanthology.org/2024.findings-acl.433/

- **Link**: https://aclanthology.org/2024.findings-acl.433/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal reasoning stands as a pivotal capability for large vision-language models (LVLMs). The integration with Domain-Specific Languages (DSL), offering precise visual representations, equips these models with the opportunity to execute more accurate reasoning in complex and professional domains. However, the vanilla Chain-of-Thought (CoT) prompting method faces challenges in effectively leveraging the unique strengths of visual and DSL representations, primarily due to their differing reasoning mechanisms. Additionally, it often falls short in addressing critical steps in multi-step reasoning tasks. To mitigate these challenges, we introduce the Bi-Modal Behavioral Alignment (BBA) prompting method, designed to maximize the potential of DSL in augmenting complex multi-modal reasoning tasks. This method initiates by guiding LVLMs to create separate reasoning chains for visual and DSL representations. Subsequently, it aligns these chains by addressing any inconsistencies, thus achieving a cohesive integration of behaviors from different modalities. Our experiments demonstrate that BBA substantially improves the performance of GPT-4V(ision) on geometry problem solving (28.34%→34.22%), chess positional advantage prediction (42.08%→46.99%) and molecular property prediction (77.47%→83.52%).

</details>

---

## 97. ChartAssistant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning

- [ ] ChartAssistant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning | https://aclanthology.org/2024.findings-acl.463/

- **Link**: https://aclanthology.org/2024.findings-acl.463/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Charts play a vital role in data visualization, understanding data patterns, and informed decision-making. However, their unique combination of graphical elements (e.g., bars, lines) and textual components (e.g., labels, legends) poses challenges for general-purpose multimodal models. While vision-language models trained on chart data excel in comprehension, they struggle with generalization. To address these challenges, we propose ChartAssistant, a chart-based vision-language model for universal chart comprehension and reasoning. ChartAssistant leverages ChartSFT, a comprehensive dataset covering diverse chart-related tasks with basic (e.g. bars and pies) and specialized (e.g. radars, and bubbles) chart types. It undergoes a two-stage training process, starting with pre-training on chart-to-table parsing to align chart and text, followed by multitask instruction-following fine-tuning. This approach enables ChartAssistant to achieve competitive performance across various chart tasks. Experimental results demonstrate significant performance gains over the state-of-the-art UniChart and ChartLlama methods, especially outperforming them on real-world chart data with zero-shot setting. The code and data are available at https://github.com/OpenGVLab/ChartAst.

</details>

---

## 98. GAOKAO-MM: AChinese Human-Level Benchmark for Multimodal Models Evaluation

- [ ] GAOKAO-MM: AChinese Human-Level Benchmark for Multimodal Models Evaluation | https://aclanthology.org/2024.findings-acl.521/

- **Link**: https://aclanthology.org/2024.findings-acl.521/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The Large Vision-Language Models (LVLMs) have demonstrated great abilities in image perception and language understanding. However, existing datasets either focus solely on primary perception abilities and commonsense knowledge, or have a low level of text comprehension difficulty, which are insufficient to reflect the comprehensive capabilities of LVLMs, particularly in terms of Chinese language proficiency. We propose GAOKAO-MM, a multimodal benchmark based on the Chinese College Entrance Examination (GAOKAO), comprising of 8 subjects and 12 types of images, such as diagrams, function graphs, maps and photos. GAOKAO-MM derives from native Chinese context and sets human-level requirements for the model’s abilities, including perception, understanding, knowledge and reasoning. We evaluate 10 LVLMs and find that the accuracies of all of them are lower than 50%, with GPT-4-Vision (48.1%), Qwen-VL-Plus (41.2%) and Gemini-Pro-Vision (35.1%) ranking in the top three positions. The results of our multi-dimension analysis indicate that LVLMs have moderate distance towards Artificial General Intelligence (AGI) and provide insights facilitating the development of multilingual LVLMs. The dataset and evaluation code are available through: https://github.com/OpenMOSS/GAOKAO-MM

</details>

---

## 99. MEEL: Multi-Modal Event Evolution Learning

- [ ] MEEL: Multi-Modal Event Evolution Learning | https://aclanthology.org/2024.findings-acl.528/

- **Link**: https://aclanthology.org/2024.findings-acl.528/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Event Reasoning (MMER) endeavors to endow machines with the ability to comprehend intricate event relations across diverse data modalities. MMER is fundamental and underlies a wide broad of applications. Despite extensive instruction fine-tuning, current multi-modal large language models still fall short in such ability. The disparity stems from that existing models are insufficient to capture underlying principles governing event evolution in various scenarios. In this paper, we introduce Multi-Modal Event Evolution Learning (MEEL) to enable the model to grasp the event evolution mechanism yielding advanced MMER ability. Specifically, we commence with the design of event diversification to gather seed events from a rich spectrum of scenarios. Subsequently, we employ ChatGPT to generate evolving graphs for these seed events. We propose an instruction encapsulation process that formulates the evolving graphs into instruction-tuning data, aligning the comprehension of event reasoning to humans. Finally, we observe that models trained in this way are still struggling to fully comprehend event evolution. In such a case, we propose the guiding discrimination strategy, in which models are trained to discriminate the improper evolution direction. We collect and curate a benchmark M-EV2 for MMER. Extensive experiments on M-EV2 validate the effectiveness of our approach, showcasing competitive performance in open-source multi-modal LLMs.

</details>

---

## 100. CoCo-Agent: A Comprehensive CognitiveMLLMAgent for SmartphoneGUIAutomation

- [ ] CoCo-Agent: A Comprehensive CognitiveMLLMAgent for SmartphoneGUIAutomation | https://aclanthology.org/2024.findings-acl.539/

- **Link**: https://aclanthology.org/2024.findings-acl.539/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have shown remarkable potential as human-like autonomous language agents to interact with real-world environments, especially for graphical user interface (GUI) automation.However, those GUI agents require comprehensive cognition including exhaustive perception and reliable action response.We propose a Comprehensive Cognitive LLM Agent, CoCo-Agent, with two novel approaches, comprehensive environment perception (CEP) and conditional action prediction (CAP), to systematically improve the GUI automation performance. First, CEP facilitates the GUI perception through different aspects and granularity, including screenshots and complementary detailed layouts for the visual channel and historical actions for the textual channel.Second, CAP decomposes the action prediction into sub-problems: determining the action type and then identifying the action target conditioned on the action type.With our technical design, our agent achieves state-of-the-art performance on AITW and META-GUI benchmarks, showing promising abilities in realistic scenarios. Code is available athttps://github.com/xbmxb/CoCo-Agent.

</details>

---

## 101. Question-Instructed Visual Descriptions for Zero-Shot Video Answering

- [ ] Question-Instructed Visual Descriptions for Zero-Shot Video Answering | https://aclanthology.org/2024.findings-acl.555/

- **Link**: https://aclanthology.org/2024.findings-acl.555/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present Q-ViD, a simple approach for video question answering (video QA), that unlike prior methods, which are based on complex architectures, computationally expensive pipelines or use closed models like GPTs, Q-ViD relies on a single instruction-aware open vision-language model (InstructBLIP) to tackle videoQA using frame descriptions. Specifically, we create captioning instruction prompts that rely on the target questions about the videos and leverage InstructBLIP to obtain video frame captions that are useful to the task at hand. Subsequently, we form descriptions of the whole video using the question-dependent frame captions, and feed that information, along with a question-answering prompt, to a large language model (LLM). The LLM is our reasoning module, and performs the final step of multiple-choice QA. Our simple Q-ViD framework achieves competitive or even higher performances than current state of the art models on a diverse range of videoQA benchmarks, including NExT-QA, STAR, How2QA, TVQA and IntentQA.

</details>

---

## 102. Visual Hallucinations of Multi-modal Large Language Models

- [ ] Visual Hallucinations of Multi-modal Large Language Models | https://aclanthology.org/2024.findings-acl.573/

- **Link**: https://aclanthology.org/2024.findings-acl.573/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual hallucination (VH) means that a multi-modal LLM (MLLM) imagines incorrect details about an image in visual question answering. Existing studies find VH instances only in existing image datasets, which results in biased understanding of MLLMs’ performance under VH due to limited diversity of such VH instances. In this work, we propose a tool called VHTest to generate a diverse set of VH instances. Specifically, VHTest finds some initial VH instances in existing image datasets (e.g., COCO), generates a text description for each VH mode, and uses a text-to-image generative model (e.g., DALL-E-3) to generate VH images based on the text descriptions. We collect a benchmark dataset with 1,200 VH instances in 8 VH modes using VHTest. We find that existing MLLMs such as GPT-4, LLaVA-1.5, and MiniGPT-v2 hallucinate for a large fraction of the instances in our benchmark. Moreover, we find that fine-tuning an MLLM using our benchmark dataset reduces its likelihood to hallucinate without sacrificing its performance on other benchmarks. Our benchmarks are publicly available: https://github.com/wenhuang2000/VHTest.

</details>

---

## 103. Leveraging Entity Information for Cross-Modality Correlation Learning: The Entity-Guided Multimodal Summarization

- [ ] Leveraging Entity Information for Cross-Modality Correlation Learning: The Entity-Guided Multimodal Summarization | https://aclanthology.org/2024.findings-acl.587/

- **Link**: https://aclanthology.org/2024.findings-acl.587/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid increase in multimedia data has spurred advancements in Multimodal Summarization with Multimodal Output (MSMO), which aims to produce a multimodal summary that integrates both text and relevant images. The inherent heterogeneity of content within multimodal inputs and outputs presents a significant challenge to the execution of MSMO. Traditional approaches typically adopt a holistic perspective on coarse image-text data or individual visual objects, overlooking the essential connections between objects and the entities they represent. To integrate the fine-grained entity knowledge, we propose an Entity-Guided Multimodal Summarization model (EGMS). Our model, building on BART, utilizes dual multimodal encoders with shared weights to process text-image and entity-image information concurrently. A gating mechanism then combines visual data for enhanced textual summary generation, while image selection is refined through knowledge distillation from a pre-trained vision-language model. Extensive experiments on public MSMO dataset validate the superiority of the EGMS method, which also prove the necessity to incorporate entity information into MSMO problem.

</details>

---

## 104. Fair Federated Learning with Biased Vision-Language Models

- [ ] Fair Federated Learning with Biased Vision-Language Models | https://aclanthology.org/2024.findings-acl.595/

- **Link**: https://aclanthology.org/2024.findings-acl.595/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing literature that integrates CLIP into federated learning (FL) largely ignores the inherent group unfairness within CLIP and its ethical implications on FL applications. Furthermore, such CLIP bias may be amplified in FL, due to the unique issue of data heterogeneity across clients. However, in identity-sensitive FL applications, model fairness (i.e., group fairness) is imperative for model development. Therefore, this work explores a critical question ignored by the existing literature: how can we build a fair FL framework using biased pre-trained VLMs (e.g., CLIP)? To address this problem, we propose a fairness-aware adaptation framework tailored for VLM (e.g., CLIP) in the context of FL, namedFairFederatedDeepVisiualPrompting orFF-DVP. As implied by its name, trains a fair FL model with fairness-aware deep visual prompting (DVP). Moreover, incorporates modality-fused classification heads to learn client-specific knowledge and fairness constraints. These modules explicitly addresses a unique bias in FL, namely the bias triggered by data heterogeneity. We show that can be readily extended to prevailing parameter-efficient fine-tuning methods (e.g., adapter or LoRA) for debiasing. To the best of our knowledge, is the first to leverage biased VLMs for building fair FL frameworks. Extensive results on human face attribute recognition (FAR) applications suggest that effectively improves model fairness and training convergence, outperforming state-of-the-art baselines.

</details>

---

## 105. SpeechGuard: Exploring the Adversarial Robustness of Multi-modal Large Language Models

- [ ] SpeechGuard: Exploring the Adversarial Robustness of Multi-modal Large Language Models | https://aclanthology.org/2024.findings-acl.596/

- **Link**: https://aclanthology.org/2024.findings-acl.596/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.

</details>

---

## 106. An Empirical Study on Parameter-Efficient Fine-Tuning forMultiModal Large Language Models

- [ ] An Empirical Study on Parameter-Efficient Fine-Tuning forMultiModal Large Language Models | https://aclanthology.org/2024.findings-acl.598/

- **Link**: https://aclanthology.org/2024.findings-acl.598/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) fine-tuned with multimodal instruction-following data have demonstrated formidable capabilities in multimodal tasks. However, fine-tuning all parameters of MLLMs has become challenging due to the rapid growth of the overall model’s parameters. To address this issue, we study Parameter-Efficient Fine-Tuning (PEFT) methods for MLLMs. We aim to identify effective methods for enhancing performance in scenarios where only a limited number of parameters are trained. This paper conducts empirical studies that employ four widely used PEFT methods to fine-tune the LLM component of open-source MLLMs. We present a comprehensive analysis that encompasses various aspects, including the impact of PEFT methods on various models, parameters and location of PEFT module, fine-tuning data scale, model stability based on PEFT method, MLLM’s generalization, and hallucination. We evaluated four PEFT methods on seven datasets from two different categories, unseen and seen datasets. Across all experiments, we show that the adapter is the best-performing PEFT method in various aspects. At the same time, fine-tuning the connector layers leads to improved performance in most MLLMs.

</details>

---

## 107. Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers

- [ ] Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers | https://aclanthology.org/2024.findings-acl.60/

- **Link**: https://aclanthology.org/2024.findings-acl.60/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding the internal mechanisms by which multi-modal large language models (LLMs) interpret different modalities and integrate cross-modal representations is becoming increasingly critical for continuous improvements in both academia and industry. In this paper, we propose a novel method to identify key neurons for interpretability — how multi-modal LLMs bridge visual and textual concepts for captioning. Our method improves conventional works upon efficiency and applied range by removing needs of costly gradient computation. Based on those identified neurons, we further design a multi-modal knowledge editing method, beneficial to mitigate sensitive words or hallucination. For rationale of our design, we provide theoretical assumption. For empirical evaluation, we have conducted extensive quantitative and qualitative experiments. The results not only validate the effectiveness of our methods, but also offer insightful findings that highlight three key properties of multi-modal neurons: sensitivity, specificity and causal-effect, to shed light for future research.

</details>

---

## 108. ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning

- [ ] ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning | https://aclanthology.org/2024.findings-acl.619/

- **Link**: https://aclanthology.org/2024.findings-acl.619/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Charts provide visual representations of data and are widely used for analyzing information, addressing queries, and conveying insights to others. Various chart-related downstream tasks have emerged recently, such as question-answering and summarization. A common strategy to solve these tasks is to fine-tune various models originally trained on vision tasks language. However, such task-specific models are not capable of solving a wide range of chart-related tasks, constraining their real-world applicability. To overcome these challenges, we introduce ChartInsruct: a novel chart-specific vision-language Instruction-following dataset comprising 191K instructions generated with 71K charts. We then present two distinct systems for instruction tuning on such datasets: (1) an end-to-end model that connects a vision encoder for chart understanding with a LLM; and (2) a pipeline model that employs a two-step approach to extract chart data tables and input them into the LLM. In experiments on four downstream tasks, we first show the effectiveness of our model–achieving a new set of state-of-the-art results. Further evaluation shows that our instruction-tuning approach supports a wide array of real-world chart comprehension and reasoning scenarios, thereby expanding the scope and applicability of our models to new kinds of tasks.

</details>

---

## 109. PCA-Bench: Evaluating Multimodal Large Language Models in Perception-Cognition-Action Chain

- [ ] PCA-Bench: Evaluating Multimodal Large Language Models in Perception-Cognition-Action Chain | https://aclanthology.org/2024.findings-acl.64/

- **Link**: https://aclanthology.org/2024.findings-acl.64/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present PCA-Bench, a multimodal decision-making benchmark for evaluating the integrated capabilities of Multimodal Large Language Models (MLLMs). Departing from previous benchmarks focusing on simplistic tasks and individual model capability, PCA-Bench introduces three complex scenarios: autonomous driving, domestic robotics, and open-world games. Given task instructions and diverse contexts, the model is required to seamlessly integrate multiple capabilities of Perception, Cognition, and Action in a reasoning chain to make accurate decisions. Moreover, PCA-Bench features error localization capabilities, scrutinizing model inaccuracies in areas such as perception, knowledge, or reasoning. This enhances the reliability of deploying MLLMs. To balance accuracy and efficiency in evaluation, we propose PCA-Eval, an automatic evaluation protocol, and assess 10 prevalent MLLMs. The results reveal significant performance disparities between open-source models and powerful proprietary models like GPT-4 Vision. To address this, we introduce Embodied-Instruction-Evolution (EIE), an automatic framework for synthesizing instruction tuning examples in multimodal embodied environments. EIE generates 7,510 training examples in PCA-Bench and enhances the performance of open-source MLLMs, occasionally surpassing GPT-4 Vision (+3% in decision accuracy), thereby validating the effectiveness of EIE. Our findings suggest that robust MLLMs like GPT4-Vision show promise for decision-making in embodied agents, opening new avenues for MLLM research. All benchmark data and evaluation code are made public.

</details>

---

## 110. ViCor: Bridging Visual Understanding and Commonsense Reasoning with Large Language Models

- [ ] ViCor: Bridging Visual Understanding and Commonsense Reasoning with Large Language Models | https://aclanthology.org/2024.findings-acl.640/

- **Link**: https://aclanthology.org/2024.findings-acl.640/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In our work, we explore the synergistic capabilities of pre-trained vision-and-language models (VLMs) and large language models (LLMs) on visual commonsense reasoning (VCR) problems. We find that VLMs and LLMs-based decision pipelines are good at different kinds of VCR problems. Pre-trained VLMs exhibit strong performance for problems involving understanding the literal visual content, which we noted as visual commonsense understanding (VCU). For problems where the goal is to infer conclusions beyond image content, which we noted as visual commonsense inference (VCI), VLMs face difficulties, while LLMs, given sufficient visual evidence, can use commonsense to infer the answer well. We empirically validate this by letting LLMs classify VCR problems into these two categories and show the significant difference between VLM and LLM with image caption decision pipelines on two subproblems. Moreover, we identify a challenge with VLMs’ passive perception, which may miss crucial context information, leading to incorrect reasoning by LLMs. Based on these, we suggest a collaborative approach, named ViCor, where pre-trained LLMs serve as problem classifiers to analyze the problem category, then either use VLMs to answer the question directly or actively instruct VLMs to concentrate on and gather relevant visual elements to support potential commonsense inferences. We evaluate our framework on two VCR benchmark datasets and outperform all other methods without in-domain fine-tuning.

</details>

---

## 111. CoLLaVO: Crayon Large Language and Vision mOdel

- [ ] CoLLaVO: Crayon Large Language and Vision mOdel | https://aclanthology.org/2024.findings-acl.66/

- **Link**: https://aclanthology.org/2024.findings-acl.66/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The remarkable success of Large Language Models (LLMs) and instruction tuning drives the evolution of Vision Language Models (VLMs) towards a versatile general-purpose model. Yet, it remains unexplored whether current VLMs genuinely possess quality object-level image understanding capabilities determined from ‘what objects are in the image?’ or ‘which object corresponds to a specified bounding box?’. Our findings reveal that the image understanding capabilities of current VLMs are strongly correlated with their zero-shot performance on vision language (VL) tasks. This suggests that prioritizing basic image understanding is crucial for VLMs to excel at VL tasks. To enhance object-level image understanding, we propose Crayon Large Language and Vision mOdel (CoLLaVO), which incorporates instruction tuning with Crayon Prompt as a new visual prompt tuning scheme based on panoptic color maps. Furthermore, we present a learning strategy of Dual QLoRA to preserve object-level image understanding without forgetting it during visual instruction tuning, thereby achieving a significant leap in numerous VL benchmarks in a zero-shot setting.

</details>

---

## 112. ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos

- [ ] ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos | https://aclanthology.org/2024.findings-acl.663/

- **Link**: https://aclanthology.org/2024.findings-acl.663/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In an era of rapidly evolving internet technology, the surge in multimodal content, including videos, has expanded the horizons of online communication. However, the detection of toxic content in this diverse landscape, particularly in low-resource code-mixed languages, remains a critical challenge. While substantial research has addressed toxic content detection in textual data, the realm of video content, especially in non-English languages, has been relatively underexplored. This paper addresses this research gap by introducing a benchmark dataset, the first of its kind, consisting of 931 videos with 4021 code-mixed Hindi-English utterances collected from YouTube. Each utterance within this dataset has been meticulously annotated for toxicity, severity, and sentiment labels. We have developed an advanced Multimodal Multitask framework built for Toxicity detection in Video Content by leveraging Language Models (LMs), crafted for the primary objective along with the additional tasks of conducting sentiment and severity analysis. ToxVidLM incorporates three key modules – the Encoder module, Cross-Modal Synchronization module, and Multitask module – crafting a generic multimodal LM customized for intricate video classification tasks. Our experiments reveal that incorporating multiple modalities from the videos substantially enhances the performance of toxic content detection by achieving an Accuracy and Weighted F1 score of 94.29% and 94.35%, respectively.

</details>

---

## 113. Enhancing Adverse Drug Event Detection with Multimodal Dataset: Corpus Creation and Model Development

- [ ] Enhancing Adverse Drug Event Detection with Multimodal Dataset: Corpus Creation and Model Development | https://aclanthology.org/2024.findings-acl.667/

- **Link**: https://aclanthology.org/2024.findings-acl.667/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The mining of adverse drug events (ADEs) is pivotal in pharmacovigilance, enhancing patient safety by identifying potential risks associated with medications, facilitating early detection of adverse events, and guiding regulatory decision-making. Traditional ADE detection methods are reliable but slow, not easily adaptable to large-scale operations, and offer limited information. With the exponential increase in data sources like social media content, biomedical literature, and Electronic Medical Records (EMR), extracting relevant ADE-related information from these unstructured texts is imperative. Previous ADE mining studies have focused on text-based methodologies, overlooking visual cues, limiting contextual comprehension, and hindering accurate interpretation. To address this gap, we present a MultiModal Adverse Drug Event (MMADE) detection dataset, merging ADE-related textual information with visual aids. Additionally, we introduce a framework that leverages the capabilities of LLMs and VLMs for ADE detection by generating detailed descriptions of medical images depicting ADEs, aiding healthcare professionals in visually identifying adverse events. Using our MMADE dataset, we showcase the significance of integrating visual cues from images to enhance overall performance. This approach holds promise for patient safety, ADE awareness, and healthcare accessibility, paving the way for further exploration in personalized healthcare.

</details>

---

## 114. Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation

- [ ] Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation | https://aclanthology.org/2024.findings-acl.672/

- **Link**: https://aclanthology.org/2024.findings-acl.672/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Assessing long-form responses generated by Vision-Language Models (VLMs) is challenging. It not only requires checking whether the VLM follows the given instruction but also verifying whether the text output is properly grounded on the given image. Inspired by the recent approach of evaluating LMs with LMs, in this work, we propose to evaluate VLMs with VLMs. For this purpose, we present a new feedback dataset called the Perception Collection, encompassing 15K customized score rubrics that users might care about during assessment. Using the Perception Collection, we train Prometheus-Vision, the first open-source VLM evaluator model that can understand the user-defined score criteria during evaluation. Prometheus-Vision shows the highest Pearson correlation with human evaluators and GPT-4V among open-source models, showing its effectiveness for transparent and accessible evaluation of VLMs. We open-source our code, dataset, and model.

</details>

---

## 115. Visualizing Dialogues: Enhancing Image Selection through Dialogue Understanding with Large Language Models

- [ ] Visualizing Dialogues: Enhancing Image Selection through Dialogue Understanding with Large Language Models | https://aclanthology.org/2024.findings-acl.700/

- **Link**: https://aclanthology.org/2024.findings-acl.700/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

For dialogue systems, the utilization of multimodal dialogue responses, as opposed to relying solely on text-only responses, offers the capability to describe different concepts through various modalities. This enhances the effectiveness of communication and elevates the overall conversational experience. However, current methods for dialogue-to-image retrieval are constrained by the capabilities of the pre-trained vision language models (VLMs). They struggle to accurately extract key information from conversations and are unable to handle long-turn conversations. In this paper, we leverage the reasoning capabilities of large language models (LLMs) to predict the potential features that may be present in the images to be shared, based on the dialogue context. This approach allows us to obtain succinct and precise descriptors, thereby improving the performance of text-image retrieval. Experimental results shows that our method outperforms previous approaches significantly in terms of Recall@k.

</details>

---

## 116. MatPlotAgent: Method and Evaluation forLLM-Based Agentic Scientific Data Visualization

- [ ] MatPlotAgent: Method and Evaluation forLLM-Based Agentic Scientific Data Visualization | https://aclanthology.org/2024.findings-acl.701/

- **Link**: https://aclanthology.org/2024.findings-acl.701/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Scientific data visualization plays a crucial role in research by enabling the direct display of complex information and assisting researchers in identifying implicit patterns. Despite its importance, the use of Large Language Models (LLMs) for scientific data visualization remains rather unexplored. In this study, we introduce MatPlotAgent, an efficient model-agnostic LLM agent framework designed to automate scientific data visualization tasks. Leveraging the capabilities of both code LLMs and multi-modal LLMs, MatPlotAgent consists of three core modules: query understanding, code generation with iterative debugging, and a visual feedback mechanism for error correction. To address the lack of benchmarks in this field, we present MatPlotBench, a high-quality benchmark consisting of 100 human-verified test cases. Additionally, we introduce a scoring approach that utilizes GPT-4V for automatic evaluation. Experimental results demonstrate that MatPlotAgent can improve the performance of various LLMs, including both commercial and open-source models. Furthermore, the proposed evaluation method shows a strong correlation with human-annotated scores.

</details>

---

## 117. MM-LLMs: Recent Advances inMultiModal Large Language Models

- [ ] MM-LLMs: Recent Advances inMultiModal Large Language Models | https://aclanthology.org/2024.findings-acl.738/

- **Link**: https://aclanthology.org/2024.findings-acl.738/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the past year, MultiModal Large Language Models (MM-LLMs) have undergone substantial advancements, augmenting off-the-shelf LLMs to support MM inputs or outputs via cost-effective training strategies. The resulting models not only preserve the inherent reasoning and decision-making capabilities of LLMs but also empower a diverse range of MM tasks. In this paper, we provide a comprehensive survey aimed at facilitating further research of MM-LLMs. Initially, we outline general design formulations for model architecture and training pipeline. Subsequently, we introduce a taxonomy encompassing 126 MM-LLMs, each characterized by its specific formulations. Furthermore, we review the performance of selected MM-LLMs on mainstream benchmarks and summarize key training recipes to enhance the potency of MM-LLMs. Finally, we explore promising directions for MM-LLMs while concurrently maintaining a [real-time tracking website](https://mm-llms.github.io/) for the latest developments in the field. We hope that this survey contributes to the ongoing advancement of the MM-LLMs domain.

</details>

---

## 118. Selective “Selective Prediction”: Reducing Unnecessary Abstention in Vision-Language Reasoning

- [ ] Selective “Selective Prediction”: Reducing Unnecessary Abstention in Vision-Language Reasoning | https://aclanthology.org/2024.findings-acl.767/

- **Link**: https://aclanthology.org/2024.findings-acl.767/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Selective prediction minimizes incorrect predictions from vision-language models (VLMs) by allowing them to abstain from answering when uncertain. However, when deploying a vision-language system with low tolerance for inaccurate predictions, selective prediction may be over-cautious and abstain too frequently, even on many correct predictions. We introduce ReCoVERR, an inference-time algorithm to reduce the over-abstention of a selective vision-language system without increasing the error rate of the system’s predictions. When the VLM makes a low-confidence prediction, instead of abstaining ReCoVERR tries to find relevant clues in the image that provide additional evidence for the prediction. ReCoVERR uses an LLM to pose related questions to the VLM, collects high-confidence evidences, and if enough evidence confirms the prediction the system makes a prediction instead of abstaining. ReCoVERR enables three VLMs (BLIP2, InstructBLIP and LLaVA-1.5) to answer up to 20% more questions on the VQAv2 and A-OKVQA tasks without decreasing system accuracy, thus improving overall system reliability. Our code is available at https://github.com/tejas1995/ReCoVERR.

</details>

---

## 119. FinTral: A Family ofGPT-4 Level Multimodal Financial Large Language Models

- [ ] FinTral: A Family ofGPT-4 Level Multimodal Financial Large Language Models | https://aclanthology.org/2024.findings-acl.774/

- **Link**: https://aclanthology.org/2024.findings-acl.774/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce FinTral, a suite of state-of-the-art multimodal large language models (LLMs) built upon the Mistral-7b model and tailored for financial analysis. FinTral integrates textual, numerical, tabular, and image data. We enhance FinTral with domain-specific pretraining, instruction fine-tuning, and RLAIF training by exploiting a large collection of textual and visual datasets we curate for this work. We also introduce an extensive benchmark featuring nine tasks and 25 datasets for evaluation, including hallucinations in the financial domain. Our FinTral model trained with direct preference optimization employing advanced Tools and Retrieval methods, dubbed FinTral-DPO-T&R, demonstrates an exceptional zero-shot performance. It outperforms ChatGPT-3.5 in all tasks and surpasses GPT-4 in five out of nine tasks, marking a significant advancement in AI-driven financial technology. We also demonstrate that FinTral has the potential to excel in real-time analysis and decision-making in diverse financial contexts.

</details>

---

## 120. Aligning Large Multimodal Models with Factually AugmentedRLHF

- [ ] Aligning Large Multimodal Models with Factually AugmentedRLHF | https://aclanthology.org/2024.findings-acl.775/

- **Link**: https://aclanthology.org/2024.findings-acl.775/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMM) are built across modalities and the misalignment between two modalities can result in “hallucination”, generating textual outputs that are not grounded by the multimodal information in context. To address the multimodal misalignment issue, we adapt the Reinforcement Learning from Human Feedback (RLHF) from the text domain to the vision-language alignment, where human annotators are asked to compare two responses and pinpoint the more hallucinated one, and the vision-language model is trained to maximize the simulated human rewards. We propose a new alignment algorithm called Factually Augmented RLHF that augments the reward model with additional factual information such as image captions and ground-truth multi-choice options, which alleviates the reward hacking phenomenon in RLHF and further improves the performance. We also enhance the GPT-4-generated training data (for vision instruction tuning) with previously available human-written image-text pairs to improve the general capabilities of our model. To evaluate the proposed approach in real-world scenarios, we develop a new evaluation benchmark MMHAL-BENCH with a special focus on penalizing hallucinations. As the first LMM trained with RLHF, our approach achieves remarkable improvement on the LLaVA-Bench dataset with the 96% performance level of the text-only GPT-4 (while previous best methods can only achieve the 87% level), and an improvement of 60% on MMHAL-BENCH over other baselines.

</details>

---

## 121. FlowVQA: Mapping Multimodal Logic in Visual Question Answering with Flowcharts

- [ ] FlowVQA: Mapping Multimodal Logic in Visual Question Answering with Flowcharts | https://aclanthology.org/2024.findings-acl.78/

- **Link**: https://aclanthology.org/2024.findings-acl.78/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing benchmarks for visual question answering lack in visual grounding and complexity, particularly in evaluating spatial reasoning skills. We introduce FlowVQA, a novel benchmark aimed at assessing the capabilities of visual question-answering multimodal language models in reasoning with flowcharts as visual contexts. FlowVQA comprises 2,272 carefully generated and human-verified flowchart images from three distinct content sources, along with 22,413 diverse question-answer pairs, to test a spectrum of reasoning tasks, including information localization, decision-making, and logical progression. We conduct a thorough baseline evaluation on a suite of both open-source and proprietary multimodal language models using various strategies, followed by an analysis of directional bias. The results underscore the benchmark’s potential as a vital tool for advancing the field of multimodal modeling, providing a focused and challenging environment for enhancing model performance in visual and logical reasoning tasks.

</details>

---

## 122. Light Up the Shadows: Enhance Long-Tailed Entity Grounding with Concept-Guided Vision-Language Models

- [ ] Light Up the Shadows: Enhance Long-Tailed Entity Grounding with Concept-Guided Vision-Language Models | https://aclanthology.org/2024.findings-acl.793/

- **Link**: https://aclanthology.org/2024.findings-acl.793/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-Modal Knowledge Graphs (MMKGs) have proven valuable for various downstream tasks. However, scaling them up is challenging because building large-scale MMKGs often introduces mismatched images (i.e., noise). Most entities in KGs belong to the long tail, meaning there are few images of them available online. This scarcity makes it difficult to determine whether a found image matches the entity. To address this, we draw on the Triangle of Reference Theory and suggest enhancing vision-language models with concept guidance. Specifically, we introduce COG, a two-stage framework with COncept-Guided vision-language models. The framework comprises a Concept Integration module, which effectively identifies image-text pairs of long-tailed entities, and an Evidence Fusion module, which offers explainability and enables human verification. To demonstrate the effectiveness of COG, we create a dataset of 25k image-text pairs of long-tailed entities. Our comprehensive experiments show that COG not only improves the accuracy of recognizing long-tailed image-text pairs compared to baselines but also offers flexibility and explainability.

</details>

---

## 123. The Revolution of Multimodal Large Language Models: A Survey

- [ ] The Revolution of Multimodal Large Language Models: A Survey | https://aclanthology.org/2024.findings-acl.807/

- **Link**: https://aclanthology.org/2024.findings-acl.807/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Connecting text and visual modalities plays an essential role in generative intelligence. For this reason, inspired by the success of large language models, significant research efforts are being devoted to the development of Multimodal Large Language Models (MLLMs). These models can seamlessly integrate visual and textual modalities, while providing a dialogue-based interface and instruction-following capabilities. In this paper, we provide a comprehensive review of recent visual-based MLLMs, analyzing their architectural choices, multimodal alignment strategies, and training techniques. We also conduct a detailed analysis of these models across a wide range of tasks, including visual grounding, image generation and editing, visual understanding, and domain-specific applications. Additionally, we compile and describe training datasets and evaluation benchmarks, conducting comparisons among existing models in terms of performance and computational requirements. Overall, this survey offers a comprehensive overview of the current state of the art, laying the groundwork for future MLLMs.

</details>

---

## 124. ChartCheck: Explainable Fact-Checking over Real-World Chart Images

- [ ] ChartCheck: Explainable Fact-Checking over Real-World Chart Images | https://aclanthology.org/2024.findings-acl.828/

- **Link**: https://aclanthology.org/2024.findings-acl.828/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Whilst fact verification has attracted substantial interest in the natural language processing community, verifying misinforming statements against data visualizations such as charts has so far been overlooked. Charts are commonly used in the real-world to summarize and com municate key information, but they can also be easily misused to spread misinformation and promote certain agendas. In this paper, we introduce ChartCheck, a novel, large-scale dataset for explainable fact-checking against real-world charts, consisting of 1.7k charts and 10.5k human-written claims and explanations. We systematically evaluate ChartCheck using vision-language and chart-to-table models, and propose a baseline to the community. Finally, we study chart reasoning types and visual attributes that pose a challenge to these models.

</details>

---

## 125. Recognizing Everything from All Modalities at Once: Grounded Multimodal Universal Information Extraction

- [ ] Recognizing Everything from All Modalities at Once: Grounded Multimodal Universal Information Extraction | https://aclanthology.org/2024.findings-acl.863/

- **Link**: https://aclanthology.org/2024.findings-acl.863/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the field of information extraction (IE), tasks across a wide range of modalities and their combinations have been traditionally studied in isolation, leaving a gap in deeply recognizing and analyzing cross-modal information. To address this, this work for the first time introduces the concept of grounded Multimodal Universal Information Extraction (MUIE), providing a unified task framework to analyze any IE tasks over various modalities, along with their fine-grained groundings. To tackle MUIE, we tailor a multimodal large language model (MLLM), Reamo, capable of extracting and grounding information from all modalities, i.e., recognizing everything from all modalities at once. Reamo is updated via varied tuning strategies, equipping it with powerful capabilities for information recognition and fine-grained multimodal grounding. To address the absence of a suitable benchmark for grounded MUIE, we curate a high-quality, diverse, and challenging test set, which encompasses IE tasks across 9 common modality combinations with the corresponding multimodal groundings. The extensive comparison of Reamo with existing MLLMs integrated into pipeline approaches demonstrates its advantages across all evaluation dimensions, establishing a strong benchmark for the follow-up research. Our resources are publicly released at https://haofei.vip/MUIE.

</details>

---

## 126. Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data

- [ ] Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data | https://aclanthology.org/2024.findings-acl.864/

- **Link**: https://aclanthology.org/2024.findings-acl.864/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The remarkable multimodal capabilities demonstrated by OpenAI’s GPT-4 have sparked significant interest in the development of multimodal Large Language Models (LLMs). A primary research objective of such models is to align visual and textual modalities effectively while comprehending human instructions.Current methodologies often rely on annotations derived from benchmark datasets to construct image-dialogue datasets for training purposes, akin to instruction tuning in LLMs. However, these datasets often exhibit domain bias, potentially constraining the generative capabilities of the models. In an effort to mitigate these limitations, we propose a novel data collection methodology that synchronously synthesizes images and dialogues for visual instruction tuning. This approach harnesses the power of generative models, marrying the abilities of ChatGPT and text-to-image generative models to yield a diverse and controllable dataset with varied image content. This not only provides greater flexibility compared to existing methodologies but also significantly enhances several model capabilities. Our research includes comprehensive experiments conducted on various datasets using the open-source LLAVA model as a testbed for our proposed pipeline. Our results underscore marked enhancements across more than ten commonly assessed capabilities.

</details>

---

## 127. BloomVQA: Assessing Hierarchical Multi-modal Comprehension

- [ ] BloomVQA: Assessing Hierarchical Multi-modal Comprehension | https://aclanthology.org/2024.findings-acl.885/

- **Link**: https://aclanthology.org/2024.findings-acl.885/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose a novel VQA dataset, BloomVQA, to facilitate comprehensive evaluation of large vision-language models on comprehension tasks. Unlike current benchmarks that often focus on fact-based memorization and simple reasoning tasks without theoretical grounding, we collect multiple-choice samples based on picture stories that reflect different levels of comprehension, as laid out in Bloom’s Taxonomy, a classic framework for learning assessment widely adopted in education research. Our data maps to a novel hierarchical graph representation which enables automatic data augmentation and novel measures characterizing model consistency. We perform graded evaluation and reliability analysis on recent multi-modal models. In comparison to low-level tasks, we observe decreased performance on tasks requiring advanced comprehension and cognitive skills with up to 38.0% drop in VQA accuracy. In comparison to earlier models, GPT-4V demonstrates improved accuracy over all comprehension levels and also shows a tendency of bypassing visual inputs especially for higher-level tasks. Current models also show consistency patterns misaligned with human comprehension in various scenarios, demonstrating the need for improvement based on theoretically-grounded criteria. The dataset can be accessed at https://huggingface.co/datasets/ygong/BloomVQA.

</details>

---

## 128. Vision-Flan: Scaling Human-Labeled Tasks in Visual Instruction Tuning

- [ ] Vision-Flan: Scaling Human-Labeled Tasks in Visual Instruction Tuning | https://aclanthology.org/2024.findings-acl.905/

- **Link**: https://aclanthology.org/2024.findings-acl.905/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite vision-language models’ (VLMs) remarkable capabilities as versatile visual assistants, two substantial challenges persist within the existing VLM frameworks: (1) lacking task diversity in pretraining and visual instruction tuning, and (2) annotation error and bias in GPT-4 synthesized instruction tuning data. Both challenges lead to issues such as poor generalizability, hallucination, and catastrophic forgetting. To address these challenges, we construct Vision-Flan, the most diverse publicly available visual instruction tuning dataset to date, comprising 187 diverse tasks and 1,664,261 instances sourced from academic datasets, and each task is accompanied by an expert-written instruction. In addition, we propose a two-stage instruction tuning framework, in which VLMs are firstly finetuned on Vision-Flan and further tuned on GPT-4 synthesized data. We find this two-stage tuning framework significantly outperforms the traditional single-stage visual instruction tuning framework and achieves the state-of-the-art performance across a wide range of multi-modal evaluation benchmarks. Finally, we conduct in-depth analyses to understand visual instruction tuning and our findings reveal that: (1) GPT-4 synthesized data does not substantially enhance VLMs’ capabilities but rather modulates the model’s responses to human-preferred formats; (2) A minimal quantity (e.g., 1,000) of GPT-4 synthesized data can effectively align VLM responses with human-preference; (3) Visual instruction tuning mainly helps large-language models (LLMs) to understand visual features.

</details>

---

## 129. Embodied Language Learning: Opportunities, Challenges, and Future Directions

- [ ] Embodied Language Learning: Opportunities, Challenges, and Future Directions | https://aclanthology.org/2024.findings-acl.908/

- **Link**: https://aclanthology.org/2024.findings-acl.908/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While large language and vision-language models showcase impressive capabilities, they face a notable limitation: the inability to connect language with the physical world. To bridge this gap, research has focused on embodied language learning, where the language learner is situated in the world, perceives it, and interacts with it. This article explores the current standing of research in embodied language learning, highlighting opportunities and discussing common challenges. Lastly, it identifies existing gaps from the perspective of language understanding research within the embodied world and suggests potential future directions.

</details>

---

## 130. Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding

- [ ] Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding | https://aclanthology.org/2024.findings-acl.937/

- **Link**: https://aclanthology.org/2024.findings-acl.937/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) are increasingly adept at generating contextually detailed and coherent responses from visual inputs. However, their application in multimodal decision-making and open-ended generation is hindered by a notable rate of hallucinations, where generated text inaccurately represents the visual contents. To address this issue, this paper introduces the Instruction Contrastive Decoding (ICD) method, a novel approach designed to reduce hallucinations during LVLM inference. Our method is inspired by our observation that what we call disturbance instructions significantly exacerbate hallucinations in multimodal fusion modules. ICD contrasts distributions from standard and instruction disturbance, thereby increasing alignment uncertainty and effectively subtracting hallucinated concepts from the original distribution. Through comprehensive experiments on discriminative benchmarks (POPE and MME) and a generative benchmark (LLaVa-Bench), we demonstrate that ICD significantly mitigates both object-level and attribute-level hallucinations. Moreover, our method not only addresses hallucinations but also significantly enhances the general perception and recognition capabilities of LVLMs.

</details>

---

## 131. Visual In-Context Learning for Large Vision-Language Models

- [ ] Visual In-Context Learning for Large Vision-Language Models | https://aclanthology.org/2024.findings-acl.940/

- **Link**: https://aclanthology.org/2024.findings-acl.940/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In Large Visual Language Models (LVLMs), the efficacy of In-Context Learning (ICL) remains limited by challenges in cross-modal interactions and representation disparities. To overcome these challenges, we introduce a novel Visual In-Context Learning (VICL) method comprising Visual Demonstration Retrieval, Intent-Oriented Image Summarization, and Intent-Oriented Demonstration Composition. Our approach retrieves images via ”Retrieval & Rerank” paradigm, summarises images with task intent and task-specific visual parsing, and composes language-based demonstrations that reduce token count and alleviate cross-modal interaction problem. Experimental evaluations on five visual reasoning datasets demonstrate the effectiveness of our method. Moreover, our extensive experiments leverage information flow analysis to elucidate the effectiveness of our method, and investigate the impact of length and position of demonstrations for LVLM. The use of in-context unlearning further shows promise in resetting specific model knowledge without retraining.

</details>

---

## 132. ContextBLIP: Doubly Contextual Alignment for Contrastive Image Retrieval from Linguistically Complex Descriptions

- [ ] ContextBLIP: Doubly Contextual Alignment for Contrastive Image Retrieval from Linguistically Complex Descriptions | https://aclanthology.org/2024.findings-acl.961/

- **Link**: https://aclanthology.org/2024.findings-acl.961/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image retrieval from contextual descriptions (IRCD) aims to identify an image within a set of minimally contrastive candidates based on linguistically complex text. Despite the success of VLMs, they still significantly lag behind human performance in IRCD. The main challenges lie in aligning key contextual cues in two modalities, where these subtle cues are concealed in tiny areas of multiple contrastive images and within the complex linguistics of textual descriptions. This motivates us to propose ContextBLIP, a simple yet effective method that relies on a doubly contextual alignment scheme for challenging IRCD. Specifically, 1) our model comprises a multi-scale adapter, a matching loss, and a text-guided masking loss. The adapter learns to capture fine-grained visual cues. The two losses enable iterative supervision for the adapter, gradually highlighting the focal patches of a single image to the key textual cues. We term such a way as intra-contextual alignment. 2) Then, ContextBLIP further employs an inter-context encoder to learn dependencies among candidates, facilitating alignment between the text to multiple images. We term this step as inter-contextual alignment. Consequently, the nuanced cues concealed in each modality can be effectively aligned. Experiments on two benchmarks show the superiority of our method. We observe that ContextBLIP can yield comparable results with GPT-4V, despite involving about 7,500 times fewer parameters.

</details>

---

## 133. ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation

- [ ] ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation | https://aclanthology.org/2024.findings-acl.99/

- **Link**: https://aclanthology.org/2024.findings-acl.99/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces the ColorSwap dataset, designed to assess and improve the proficiency of multimodal models in matching objects with their colors. The dataset is comprised of 2,000 unique image-caption pairs, grouped into 1,000 examples. Each example includes a caption-image pair, along with a “color-swapped” pair. We follow the Winoground schema: the two captions in an example have the same words, but the color words have been rearranged to modify different objects. The dataset was created through a novel blend of automated caption and image generation with humans in the loop. We evaluate image-text matching (ITM) and visual language models (VLMs) and find that even the latest ones are still not robust at this task. GPT-4V and LLaVA score 72% and 42% on our main VLM metric, although they may improve with more advanced prompting techniques. On the main ITM metric, contrastive models such as CLIP and SigLIP perform close to chance (at 12% and 30%, respectively), although the non-contrastive BLIP ITM model is stronger (87%). We also find that finetuning on fewer than 2,000 examples yields significant performance gains on this out-of-distribution word-order understanding task.

</details>

---

## 134. Vision-Language Models under Cultural and Inclusive Considerations

- [ ] Vision-Language Models under Cultural and Inclusive Considerations | https://aclanthology.org/2024.hucllm-1.5/

- **Link**: https://aclanthology.org/2024.hucllm-1.5/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision Language Models can be used to assist visually impaired individuals by describing images they capture in their daily lives. Current evaluation datasets may not reflect the diverse cultural user backgrounds nor the situational context of this use case. To address this problem, we create a survey to determine caption preferences and propose a culture-centric evaluation benchmark by filtering VizWiz, an existing dataset with images taken by people who are blind. We then evaluate different models and prompts, investigating their reliability as visual assistants. While the evaluation results for state-of-the-art models seem promising, we identified some weak spots such as hallucinations and problems with conventional evaluation metrics. Our survey, data, code, and model outputs will be publicly available.

</details>

---

## 135. CMU’sIWSLT2024 Simultaneous Speech Translation System

- [ ] CMU’sIWSLT2024 Simultaneous Speech Translation System | https://aclanthology.org/2024.iwslt-1.20/

- **Link**: https://aclanthology.org/2024.iwslt-1.20/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper describes CMU’s submission to the IWSLT 2024 Simultaneous Speech Translation (SST) task for translating English speech to German text in a streaming manner. Our end-to-end speech-to-text (ST) system integrates the WavLM speech encoder, a modality adapter, and the Llama2-7B-Base model as the decoder. We employ a two-stage training approach: initially, we align the representations of speech and text, followed by full fine-tuning. Both stages are trained on MuST-c v2 data with cross-entropy loss. We adapt our offline ST model for SST using a simple fixed hold-n policy. Experiments show that our model obtains an offline BLEU score of 31.1 and a BLEU score of 29.5 under 2 seconds latency on the MuST-C-v2 tst-COMMON.

</details>

---

## 136. TransformingLLMs into Cross-modal and Cross-lingual Retrieval Systems

- [ ] TransformingLLMs into Cross-modal and Cross-lingual Retrieval Systems | https://aclanthology.org/2024.iwslt-1.4/

- **Link**: https://aclanthology.org/2024.iwslt-1.4/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) are trained on text-only data that go far beyond the languages with paired speech and text data. At the same time, Dual Encoder (DE) based retrieval systems project queries and documents into the same embedding space and have demonstrated their success in retrieval and bi-text mining. To match speech and text in many languages, we propose using LLMs to initialize multi-modal DE retrieval systems. Unlike traditional methods, our system doesn’t require speech data during LLM pre-training and can exploit LLM’s multilingual text understanding capabilities to match speech and text in languages unseen during retrieval training. Our multi-modal LLM-based retrieval system is capable of matching speech and text in 102 languages despite only training on 21 languages. Our system outperforms previous systems trained explicitly on all 102 languages. We achieve a 10% absolute improvement in Recall@1 averaged across these languages. Additionally, our model demonstrates cross-lingual speech and text matching, which is further enhanced by readily available machine translation data.

</details>

---

## 137. Mol2Lang-VLM: Vision- and Text-Guided Generative Pre-trained Language Models for Advancing Molecule Captioning through Multimodal Fusion

- [ ] Mol2Lang-VLM: Vision- and Text-Guided Generative Pre-trained Language Models for Advancing Molecule Captioning through Multimodal Fusion | https://aclanthology.org/2024.langmol-1.12/

- **Link**: https://aclanthology.org/2024.langmol-1.12/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces Mol2Lang-VLM, an enhanced method for refining generative pre-trained language models for molecule captioning using multimodal features to achieve more accurate caption generation. Our approach leverages the encoder and decoder blocks of the Transformer-based architecture by introducing third sub-layers into both. Specifically, we insert sub-layers in the encoder to fuse features from SELFIES strings and molecular images, while the decoder fuses features from SMILES strings and their corresponding descriptions. Moreover, cross multi-head attention is employed instead of common multi-head attention to enable the decoder to attend to the encoder’s output, thereby integrating the encoded contextual information for better and more accurate caption generation. Performance evaluation on the CheBI-20 and L+M-24 benchmark datasets demonstrates Mol2Lang-VLM’s superiority, achieving higher accuracy and quality in caption generation compared to existing methods. Our code and pre-processed data are available at https://github.com/nhattruongpham/mol-lang-bridge/tree/mol2lang/.

</details>

---

## 138. Knowledge Graph Extraction from Total Synthesis Documents

- [ ] Knowledge Graph Extraction from Total Synthesis Documents | https://aclanthology.org/2024.langmol-1.9/

- **Link**: https://aclanthology.org/2024.langmol-1.9/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Knowledge graphs (KGs) have emerged as a powerful tool for organizing and integrating complex information, making it a suitable format for scientific knowledge. However, translating scientific knowledge into KGs is challenging as a wide variety of styles and elements to present data and ideas is used. Although efforts for KG extraction (KGE) from scientific documents exist, evaluation remains challenging and field-dependent; and existing benchmarks do not focuse on scientific information. Furthermore, establishing a general benchmark for this task is challenging as not all scientific knowledge has a ground-truth KG representation, making any benchmark prone to ambiguity. Here we propose Graph of Organic Synthesis Benchmark (GOSyBench), a benchmark for KG extraction from scientific documents in chemistry, that leverages the native KG-like structure of synthetic routes in organic chemistry. We develop KG-extraction algorithms based on LLMs (GPT-4, Claude, Mistral) and VLMs (GPT-4o), the best of which reaches 73% recovery accuracy and 59% precision, leaving a lot of room for improvement. We expect GOSyBench can serve as a valuable resource for evaluating and advancing KGE methods in the scientific domain, ultimately facilitating better organization, integration, and discovery of scientific knowledge.

</details>

---

## 139. Text-Guided Alternative Image Clustering

- [ ] Text-Guided Alternative Image Clustering | https://aclanthology.org/2024.repl4nlp-1.13/

- **Link**: https://aclanthology.org/2024.repl4nlp-1.13/

- **Conference**: ACL

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Traditional image clustering techniques only find a single grouping within visual data. In particular, they do not provide a possibility to explicitly define multiple types of clustering. This work explores the potential of large vision-language models to facilitate alternative image clustering. We propose Text-Guided Alternative Image Consensus Clustering (TGAICC), a novel approach that leverages user-specified interests via prompts to guide the discovery of diverse clusterings. To achieve this, it generates a clustering for each prompt, groups them using hierarchical clustering, and then aggregates them using consensus clustering. TGAICC outperforms image- and text-based baselines on four alternative image clustering benchmark datasets. Furthermore, using count-based word statistics, we are able to obtain text-based explanations of the alternative clusterings. In conclusion, our research illustrates how contemporary large vision-language models can transform explanatory data analysis, enabling the generation of insightful, customizable, and diverse image clusterings.

</details>

---

