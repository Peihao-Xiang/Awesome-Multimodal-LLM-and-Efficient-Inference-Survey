# EMNLP 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_emnlp2024_papers.csv

## 1. Text2Afford: Probing Object Affordance Prediction abilities of Language Models solely from Text

- [ ] Text2Afford: Probing Object Affordance Prediction abilities of Language Models solely from Text | https://aclanthology.org/2024.conll-1.27/

- **Link**: https://aclanthology.org/2024.conll-1.27/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We investigate the knowledge of object affordances in pre-trained language models (LMs) and pre-trained Vision-Language models (VLMs).A growing body of literature shows that PTLMs fail inconsistently and non-intuitively, demonstrating a lack of reasoning and grounding. To take a first step toward quantifying the effect of grounding (or lack thereof), we curate a novel and comprehensive dataset of object affordances – Text2Afford, characterized by 15 affordance classes. Unlike affordance datasets collected in vision and language domains, we annotate in-the-wild sentences with objects and affordances. Experimental results reveal that PTLMs exhibit limited reasoning abilities when it comes to uncommon object affordances. We also observe that pre-trained VLMs do not necessarily capture object affordances effectively. Through few-shot fine-tuning, we demonstrate improvement in affordance knowledge in PTLMs and VLMs. Our research contributes a novel dataset for language grounding tasks, and presents insights into LM capabilities, advancing the understanding of object affordances.

</details>

---

## 2. Image-conditioned human language comprehension and psychometric benchmarking of visual language models

- [ ] Image-conditioned human language comprehension and psychometric benchmarking of visual language models | https://aclanthology.org/2024.conll-1.34/

- **Link**: https://aclanthology.org/2024.conll-1.34/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language model (LLM)s’ next-word predictions have shown impressive performance in capturing human expectations during real-time language comprehension. This finding has enabled a line of research on psychometric benchmarking of LLMs against human language-comprehension data in order to reverse-engineer humans’ linguistic subjective probability distributions and representations. However, to date, this work has exclusively involved unimodal (language-only) comprehension data, whereas much human language use takes place in rich multimodal contexts. Here we extend psychometric benchmarking to visual language models (VLMs). We develop a novel experimental paradigm,Image-Conditioned Maze Reading, in which participants first view an image and then read a text describing an image within the Maze paradigm, yielding word-by-word reaction-time measures with high signal-to-noise ratio and good localization of expectation-driven language processing effects. We find a large facilitatory effect of correct image context on language comprehension, not only for words such as concrete nouns that are directly grounded in the image but even for ungrounded words in the image descriptions. Furthermore, we find that VLM surprisal captures most to all of this effect. We use these findings to benchmark a range of VLMs, showing that models with lower perplexity generally have better psychometric performance, but that among the best VLMs tested perplexity and psychometric performance dissociate. Overall, our work offers new possibilities for connecting psycholinguistics with multimodal LLMs for both scientific and engineering goals.

</details>

---

## 3. A Multimodal Large Language Model “Foresees” Objects Based on Verb Information but Not Gender

- [ ] A Multimodal Large Language Model “Foresees” Objects Based on Verb Information but Not Gender | https://aclanthology.org/2024.conll-1.32/

- **Link**: https://aclanthology.org/2024.conll-1.32/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study employs the classical psycholinguistics paradigm, the visual world eye-tracking paradigm (VWP), to explore the predictive capabilities of LLAVA, a multimodal large language model (MLLM), and compare them with human anticipatory gaze behaviors. Specifically, we examine the attention weight distributions of LLAVA when presented with visual displays and English sentences containing verb and gender cues. Our findings reveal that LLAVA, like humans, can predictively attend to objects relevant to verbs, but fails to demonstrate gender-based anticipatory attention. Layer-wise analysis indicates that the middle layers of the model are more related to predictive attention than the early or late layers. This study is pioneering in applying psycholinguistic paradigms to compare the multimodal predictive attention of humans and MLLMs, revealing both similarities and differences between them.

</details>

---

## 4. BattleAgent: Multi-modal Dynamic Emulation on Historical Battles to Complement Historical Analysis

- [ ] BattleAgent: Multi-modal Dynamic Emulation on Historical Battles to Complement Historical Analysis | https://aclanthology.org/2024.emnlp-demo.18/

- **Link**: https://aclanthology.org/2024.emnlp-demo.18/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presentsBattleAgent, a detailed emulation demonstration system that combines the Large Vision-Language Model (VLM) and Multi-Agent System (MAS). This novel system aims to emulate complex dynamic interactions among multiple agents, as well as between agents and their environments, over a period of time. The emulation showcases the current capabilities of agents, featuring fine-grained multi-modal interactions between agents and landscapes. It develops customizable agent structures to meet specific situational requirements, for example, a variety of battle-related activities like scouting and trench digging. These components collaborate to recreate historical events in a lively and comprehensive manner. This methodology holds the potential to substantially improve visualization of historical events and deepen our understanding of historical events especially from the perspective of decision making. The data and code for this project are accessible athttps://github.com/agiresearch/battleagentand the demo is accessible athttps://drive.google.com/file/d/1I5B3KWiYCSSP1uMiPGNmXlTmild-MzRJ/view?usp=sharing.

</details>

---

## 5. AutoTrain: No-code training for state-of-the-art models

- [ ] AutoTrain: No-code training for state-of-the-art models | https://aclanthology.org/2024.emnlp-demo.44/

- **Link**: https://aclanthology.org/2024.emnlp-demo.44/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

With the advancements in open-source models, training(or finetuning) models on custom datasets has become a crucial part of developing solutions which are tailored to specific industrial or open-source applications. Yet, there is no single tool which simplifies the process of training across different types of modalities or tasks.We introduce AutoTrain(aka AutoTrain Advanced)—an open-source, no code tool/library which can be used to train (or finetune) models for different kinds of tasks such as: large language model (LLM) finetuning, text classification/regression, token classification, sequence-to-sequence task, finetuning of sentence transformers, visual language model (VLM) finetuning, image classification/regression and even classification and regression tasks on tabular data. AutoTrain Advanced is an open-source library providing best practices for training models on custom datasets. The library is available at https://github.com/huggingface/autotrain-advanced. AutoTrain can be used in fully local mode or on cloud machines and works with tens of thousands of models shared on Hugging Face Hub and their variations.

</details>

---

## 6. LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit

- [ ] LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit | https://aclanthology.org/2024.emnlp-industry.12/

- **Link**: https://aclanthology.org/2024.emnlp-industry.12/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large language models (LLMs) are propelling us toward artificial general intelligence with their remarkable emergent abilities and reasoning capabilities. However, the substantial computational and memory requirements limit the widespread adoption. Quantization, a key compression technique, can effectively mitigate these demands by compressing and accelerating LLMs, albeit with potential risks to accuracy. Numerous studies have aimed to minimize the accuracy loss associated with quantization. However, their quantization configurations vary from each other and cannot be fairly compared. In this paper, we present LLMC, a plug-and-play compression toolkit, to fairly and systematically explore the impact of quantization. LLMC integrates dozens of algorithms, models, and hardware, offering high extensibility from integer to floating-point quantization, from LLM to vision-language (VLM) model, from fixed-bit to mixed precision, and from quantization to sparsification. Powered by this versatile toolkit, our benchmark covers three key aspects: calibration data, algorithms (three strategies), and data formats, providing novel insights and detailed analyses for further research and practical guidance for users. Our toolkit is available at https://github.com/ModelTC/llmc.

</details>

---

## 7. ScaleLLM: A Resource-FrugalLLMServing Framework by Optimizing End-to-End Efficiency

- [ ] ScaleLLM: A Resource-FrugalLLMServing Framework by Optimizing End-to-End Efficiency | https://aclanthology.org/2024.emnlp-industry.22/

- **Link**: https://aclanthology.org/2024.emnlp-industry.22/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have surged in popularity and are extensively used in commercial applications, where the efficiency of model serving is crucial for the user experience. Most current research focuses on optimizing individual sub-procedures, e.g. local inference and communication, however, there is no comprehensive framework that provides a holistic system view for optimizing LLM serving in an end-to-end manner. In this work, we conduct a detailed analysis to identify major bottlenecks that impact end-to-end latency in LLM serving systems. Our analysis reveals that a comprehensive LLM serving endpoint must address a series of efficiency bottlenecks that extend beyond LLM inference. We then propose ScaleLLM, an optimized system for resource-efficient LLM serving. Our extensive experiments reveal that reveal that with 64 concurrent requests on Mixtral 8x7B, ScaleLLM achieves a 4.3× speed up over vLLM and outperforms state-of-the-arts with 1.5× higher throughput.

</details>

---

## 8. FastAdaSP: Multitask-Adapted Efficient Inference for Large Speech Language Model

- [ ] FastAdaSP: Multitask-Adapted Efficient Inference for Large Speech Language Model | https://aclanthology.org/2024.emnlp-industry.33/

- **Link**: https://aclanthology.org/2024.emnlp-industry.33/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this study, we aim to explore Multitask Speech Language Model (SpeechLM) efficient inference via token reduction. Unlike other modalities such as vision or text, speech has unique temporal dependencies, making previous efficient inference works on other modalities not directly applicable. Furthermore, methods for efficient SpeechLM inference on long sequence and sparse signals remain largely unexplored. In this work, we propose FastAdaSP, a weighted token merging framework specifically designed for various speech-related tasks to improve the trade-off between efficiency and performance. Experimental results on WavLLM and Qwen-Audio show that our method achieves the state-of-the-art (SOTA) efficiency-performance trade-off compared with other baseline methods. Specifically, FastAdaSP achieved 7x memory efficiency and 1.83x decoding throughput without any degradation on tasks like Emotion Recognition (ER) and Spoken Question Answering (SQA).

</details>

---

## 9. IPL: Leveraging Multimodal Large Language Models for Intelligent Product Listing

- [ ] IPL: Leveraging Multimodal Large Language Models for Intelligent Product Listing | https://aclanthology.org/2024.emnlp-industry.52/

- **Link**: https://aclanthology.org/2024.emnlp-industry.52/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Unlike professional Business-to-Consumer (B2C) e-commerce platforms (e.g., Amazon), Consumer-to-Consumer (C2C) platforms (e.g., Facebook marketplace) are mainly targeting individual sellers who usually lack sufficient experience in e-commerce. Individual sellers often struggle to compose proper descriptions for selling products. With the recent advancement of Multimodal Large Language Models (MLLMs), we attempt to integrate such state-of-the-art generative AI technologies into the product listing process. To this end, we develop IPL, an Intelligent Product Listing tool tailored to generate descriptions using various product attributes such as category, brand, color, condition, etc. IPL enables users to compose product descriptions by merely uploading photos of the selling product. More importantly, it can imitate the content style of our C2C platform Xianyu. This is achieved by employing domain-specific instruction tuning on MLLMs, and by adopting the multi-modal Retrieval-Augmented Generation (RAG) process. A comprehensive empirical evaluation demonstrates that the underlying model of IPL significantly outperforms the base model in domain-specific tasks while producing less hallucination. IPL has been successfully deployed in our production system, where 72% of users have their published product listings based on the generated content, and those product listings are shown to have a quality score 5.6% higher than those without AI assistance.

</details>

---

## 10. Generating Vehicular Icon Descriptions and Indications Using Large Vision-Language Models

- [ ] Generating Vehicular Icon Descriptions and Indications Using Large Vision-Language Models | https://aclanthology.org/2024.emnlp-industry.83/

- **Link**: https://aclanthology.org/2024.emnlp-industry.83/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

To enhance a question-answering system for automotive drivers, we tackle the problem of automatic generation of icon image descriptions. The descriptions can match the driver’s query about the icon appearing on the dashboard and tell the driver what is happening so that they may take an appropriate action. We use three state-of-the-art large vision-language models to generate both visual and functional descriptions based on the icon image and its context information in the car manual. Both zero-shot and few-shot prompts are used. We create a dataset containing over 400 icons with their ground-truth descriptions and use it to evaluate model-generated descriptions across several performance metrics. Our evaluation shows that two of these models (GPT-4o and Claude 3.5) performed well on this task, while the third model (LLaVA-NEXT) performs poorly.

</details>

---

## 11. Investigating and Mitigating Object Hallucinations in Pretrained Vision-Language (CLIP) Models

- [ ] Investigating and Mitigating Object Hallucinations in Pretrained Vision-Language (CLIP) Models | https://aclanthology.org/2024.emnlp-main.1016/

- **Link**: https://aclanthology.org/2024.emnlp-main.1016/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have achieved impressive performance, yet research has pointed out a serious issue with object hallucinations within these models. However, there is no clear conclusion as to which part of the model these hallucinations originate from. In this paper, we present an in-depth investigation into the object hallucination problem specifically within the CLIP model, which serves as the backbone for many state-of-the-art vision-language systems. We unveil that even in isolation, the CLIP model is prone to object hallucinations, suggesting that the hallucination problem is not solely due to the interaction between vision and language modalities. To address this, we propose a counterfactual data augmentation method by creating negative samples with a variety of hallucination issues. We demonstrate that our method can effectively mitigate object hallucinations for CLIP model, and we show the the enhanced model can be employed as a visual encoder, effectively alleviating the object hallucination issue in LVLMs.

</details>

---

## 12. PALM: Few-Shot Prompt Learning for Audio Language Models

- [ ] PALM: Few-Shot Prompt Learning for Audio Language Models | https://aclanthology.org/2024.emnlp-main.1030/

- **Link**: https://aclanthology.org/2024.emnlp-main.1030/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Audio-Language Models (ALMs) have recently achieved remarkable success in zero-shot audio recognition tasks, which match features of audio waveforms with class-specific text prompt features, inspired by advancements in Vision-Language Models (VLMs). Given the sensitivity of zero-shot performance to the choice of hand-crafted text prompts, many prompt learning techniques have been developed for VLMs. We explore the efficacy of these approaches in ALMs and propose a novel method, Prompt Learning in Audio Language Models (PALM), which optimizes the feature space of the text encoder branch. Unlike existing methods that work in the input space, our approach results in greater training efficiency. We demonstrate the effectiveness of our approach on 11 audio recognition datasets, encompassing a variety of speech-processing tasks, and compare the results with three baselines in a few-shot learning setup. Our method is either on par with or outperforms other approaches while being computationally less demanding. Our code is publicly available athttps://asif-hanif.github.io/palm/.

</details>

---

## 13. HELPD: Mitigating Hallucination ofLVLMs by Hierarchical Feedback Learning with Vision-enhanced Penalty Decoding

- [ ] HELPD: Mitigating Hallucination ofLVLMs by Hierarchical Feedback Learning with Vision-enhanced Penalty Decoding | https://aclanthology.org/2024.emnlp-main.105/

- **Link**: https://aclanthology.org/2024.emnlp-main.105/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have shown remarkable performance on many visual-language tasks. However, these models still suffer frommultimodal hallucination, which means the generation of objects or content that violates the images. Many existing work detects hallucination by directly judging whether an object exists in an image, overlooking the association between the object and semantics. To address this issue, we propose Hierarchical Feedback Learning with Vision-enhanced Penalty Decoding (HELPD). This framework incorporates hallucination feedback at both object and sentence semantic levels. Remarkably, even with a marginal degree of training, this approach can alleviate over 15% of hallucination. Simultaneously, HELPD penalizes the output logits according to the image attention window to avoid being overly affected by generated text. HELPD can be seamlessly integrated with any LVLMs. Our experiments demonstrate that the proposed framework yields favorable results across multiple hallucination benchmarks. It effectively mitigates hallucination for different LVLMs and concurrently improves their text generation quality.

</details>

---

## 14. TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning

- [ ] TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning | https://aclanthology.org/2024.emnlp-main.1059/

- **Link**: https://aclanthology.org/2024.emnlp-main.1059/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

It is challenging for models to understand complex, multimodal content such as television clips, and this is in part because video-language models often rely on single-modality reasoning and lack interpretability. To combat these issues we propose TV-TREES, the first multimodal entailment tree generator. TV-TREES serves as an approach to video understanding that promotes interpretable joint-modality reasoning by searching for trees of entailment relationships between simple text-video evidence and higher-level conclusions that prove question-answer pairs. We also introduce the task of multimodal entailment tree generation to evaluate reasoning quality. Our method’s performance on the challenging TVQA benchmark demonstrates interpretable, state-of-the-art zero-shot performance on full clips, illustrating that multimodal entailment tree generation can be a best-of-both-worlds alternative to black-box systems.

</details>

---

## 15. TopViewRS: Vision-Language Models as Top-View Spatial Reasoners

- [ ] TopViewRS: Vision-Language Models as Top-View Spatial Reasoners | https://aclanthology.org/2024.emnlp-main.106/

- **Link**: https://aclanthology.org/2024.emnlp-main.106/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Top-view perspective denotes a typical way in which humans read and reason over different types of maps, and it is vital for localization and navigation of humans as well as of ‘non-human’ agents, such as the ones backed by large Vision-Language Models (VLMs). Nonetheless, spatial reasoning capabilities of modern VLMs in this setup remain unattested and underexplored. In this work, we study their capability to understand and reason over spatial relations from the top view. The focus on top view also enables controlled evaluations at different granularity of spatial reasoning; we clearly disentangle different abilities (e.g., recognizing particular objects versus understanding their relative positions). We introduce the TopViewRS (Top-View Reasoning in Space) dataset, consisting of 11,384 multiple-choice questions with either realistic or semantic top-view map as visual input. We then use it to study and evaluate VLMs across 4 perception and reasoning tasks with different levels of complexity. Evaluation of 10 representative open- and closed-source VLMs reveals the gap of more than 50% compared to average human performance, and it is even lower than the random baseline in some cases. Although additional experiments show that Chain-of-Thought reasoning can boost model capabilities by 5.82% on average, the overall performance of VLMs remains limited. Our findings underscore the critical need for enhanced model capability in top-view spatial reasoning and set a foundation for further research towards human-level proficiency of VLMs in real-world multimodal tasks.

</details>

---

## 16. Preserving Multi-Modal Capabilities of Pre-trainedVLMs for Improving Vision-Linguistic Compositionality

- [ ] Preserving Multi-Modal Capabilities of Pre-trainedVLMs for Improving Vision-Linguistic Compositionality | https://aclanthology.org/2024.emnlp-main.1062/

- **Link**: https://aclanthology.org/2024.emnlp-main.1062/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we propose a new method to enhance compositional understanding in pre-trained vision and language models (VLMs) without sacrificing performance in zero-shot multi-modal tasks. Traditional fine-tuning approaches often improve compositional reasoning at the cost of degrading multi-modal capabilities, primarily due to the use of global hard negative (HN) loss, which contrasts global representations of images and texts. This global HN loss pushes HN texts that are highly similar to the original ones, damaging the model’s multi-modal representations. To overcome this limitation, we propose Fine-grained Selective Calibrated CLIP (FSC-CLIP), which integrates local hard negative loss and selective calibrated regularization. These innovations provide fine-grained negative supervision while preserving the model’s representational integrity. Our extensive evaluations across diverse benchmarks for both compositionality and multi-modal tasks show that FSC-CLIP not only achieves compositionality on par with state-of-the-art models but also retains strong multi-modal capabilities. Code is available at: https://github.com/ytaek-oh/fsc-clip.

</details>

---

## 17. GRIZAL: Generative Prior-guided Zero-Shot Temporal Action Localization

- [ ] GRIZAL: Generative Prior-guided Zero-Shot Temporal Action Localization | https://aclanthology.org/2024.emnlp-main.1061/

- **Link**: https://aclanthology.org/2024.emnlp-main.1061/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Zero-shot temporal action localization (TAL) aims to temporally localize actions in videos without prior training examples. To address the challenges of TAL, we offer GRIZAL, a model that uses multimodal embeddings and dynamic motion cues to localize actions effectively. GRIZAL achieves sample diversity by using large-scale generative models such as GPT-4 for generating textual augmentations and DALL-E for generating image augmentations. Our model integrates vision-language embeddings with optical flow insights, optimized through a blend of supervised and self-supervised loss functions. On ActivityNet, Thumos14 and Charades-STA datasets, GRIZAL greatly outperforms state-of-the-art zero-shot TAL models, demonstrating its robustness and adaptability across a wide range of video content. We will make all the models and code publicly available by open-sourcing them.

</details>

---

## 18. FoodieQA: A Multimodal Dataset for Fine-Grained Understanding ofChinese Food Culture

- [ ] FoodieQA: A Multimodal Dataset for Fine-Grained Understanding ofChinese Food Culture | https://aclanthology.org/2024.emnlp-main.1063/

- **Link**: https://aclanthology.org/2024.emnlp-main.1063/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Food is a rich and varied dimension of cultural heritage, crucial to both individuals and social groups. To bridge the gap in the literature on the often-overlooked regional diversity in this domain, we introduce FoodieQA, a manually curated, fine-grained image-text dataset capturing the intricate features of food cultures across various regions in China. We evaluate vision–language Models (VLMs) and large language models (LLMs) on newly collected, unseen food images and corresponding questions. FoodieQA comprises three multiple-choice question-answering tasks where models need to answer questions based on multiple images, a single image, and text-only descriptions, respectively. While LLMs excel at text-based question answering, surpassing human accuracy, the open-sourced VLMs still fall short by 41% on multi-image and 21% on single-image VQA tasks, although closed-weights models perform closer to human levels (within 10%). Our findings highlight that understanding food and its cultural implications remains a challenging and under-explored direction.

</details>

---

## 19. IntCoOp: Interpretability-Aware Vision-Language Prompt Tuning

- [ ] IntCoOp: Interpretability-Aware Vision-Language Prompt Tuning | https://aclanthology.org/2024.emnlp-main.1092/

- **Link**: https://aclanthology.org/2024.emnlp-main.1092/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image-text contrastive models such as CLIP learn transferable and robust representations for zero-shot transfer to a variety of downstream tasks. However, to obtain strong downstream performances, prompts need to be carefully curated, which can be a tedious engineering task. To address the issue of manual prompt engineering, prompt-tuning is used where a set of contextual vectors are learned by leveraging information from the training data. Despite their effectiveness, existing prompt-tuning frameworks often lack interpretability, thus limiting their ability to understand the compositional nature of images. In this work, we first identify that incorporating compositional attributes (e.g., a “green” tree frog) in the design of manual prompts can significantly enhance image-text alignment scores. Building upon this observation, we propose a novel and interpretable prompt-tuning method named IntCoOp, which learns to jointly align attribute-level inductive biases and class embeddings during prompt-tuning. To assess the effectiveness of our approach, we evaluate IntCoOp across two representative tasks in a few-shot learning setup: generalization to novel classes, and unseen domain shifts. Through extensive experiments across 10 downstream datasets on CLIP, we find that introducing attribute-level inductive biases leads to superior performance against state-of-art prompt tuning frameworks. Notably, in a 16-shot setup, IntCoOp improves CoOp by 7.35% in average performance across 10 diverse datasets.

</details>

---

## 20. Self-Bootstrapped Visual-Language Model for Knowledge Selection and Question Answering

- [ ] Self-Bootstrapped Visual-Language Model for Knowledge Selection and Question Answering | https://aclanthology.org/2024.emnlp-main.110/

- **Link**: https://aclanthology.org/2024.emnlp-main.110/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While large pre-trained visual-language models have shown promising results on traditional visual question answering benchmarks, it is still challenging for them to answer complex VQA problems which requires diverse world knowledge. Motivated by the research of retrieval-augmented generation in the field of natural language processing, we use Dense Passage Retrieval (DPR) to retrieve related knowledge to help the model answer questions. However, DPR conduct retrieving in natural language space, which may not ensure comprehensive acquisition of image information. Thus, the retrieved knowledge is not truly conducive to helping answer the question, affecting the performance of the overall system. To address this issue, we propose a novel framework that leverages the visual-language model to select the key knowledge retrieved by DPR and answer questions. The framework consists of two modules: Selector and Answerer, where both are initialized by the MLLM and parameter-efficiently finetuned by self-bootstrapping: find key knowledge in the retrieved knowledge documents using the Selector, and then use them to finetune the Answerer to predict answers; obtain the pseudo-labels of key knowledge documents based on the predictions of the Answerer and weak supervision labels, and then finetune the Selector to select key knowledge; repeat. Our framework significantly enhances the performance of the baseline on the challenging open-domain Knowledge-based VQA benchmark, OK-VQA, achieving a state-of-the-art accuracy of 62.83%.

</details>

---

## 21. Error Analysis of Multilingual Language Models in Machine Translation: A Case Study ofEnglish-Amharic Translation

- [ ] Error Analysis of Multilingual Language Models in Machine Translation: A Case Study ofEnglish-Amharic Translation | https://aclanthology.org/2024.emnlp-main.1102/

- **Link**: https://aclanthology.org/2024.emnlp-main.1102/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multilingual large language models (mLLMs) have significantly advanced machine translation, yet challenges remain for low-resource languages like Amharic. This study evaluates the performance of state-of-the-art mLLMs, specifically NLLB-200 (NLLB3.3, NLLB1.3 Distilled1.3, NLB600) and M2M (M2M1.2B, M2M418), in English-Amharic bidirectional translation using the Lesan AI dataset. We employed both automatic and human evaluation methods to analyze translation errors. Automatic evaluation used BLEU, METEOR, chrF, and TER metrics, while human evaluation assessed translation quality at both word and sentence levels. Sentence-level accuracy was rated by annotators on a scale from 0 to 5, and word-level quality was evaluated using Multidimensional Quality Metrics. Our findings indicate that the NLLB3.3B model consistently outperformed other mLLMs across all evaluation methods. Common error included mistranslation, omission, untranslated segments, and additions, with mistranslation being particularly common. Punctuation and spelling errors were rare in our experiment.

</details>

---

## 22. Whiteboard-of-Thought: Thinking Step-by-Step Across Modalities

- [ ] Whiteboard-of-Thought: Thinking Step-by-Step Across Modalities | https://aclanthology.org/2024.emnlp-main.1117/

- **Link**: https://aclanthology.org/2024.emnlp-main.1117/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

When presented with questions involving visual thinking, humans naturally switch reasoning modalities, often forming mental images or drawing visual aids. Large language models have shown promising results in arithmetic and symbolic reasoning by expressing intermediate reasoning in text as a chain of thought, yet struggle to extend this capability to answer text queries that are easily solved by visual reasoning, even with extensive multimodal pretraining. We introduce a simple method,whiteboard-of-thoughtprompting, to unlock the visual reasoning capabilities of multimodal large language models across modalities. Whiteboard-of-thought prompting provides multimodal large language models with a metaphorical ‘whiteboard’ to draw out reasoning steps as images, then returns these images back to the model for further processing. We find this can be accomplished with no demonstrations or specialized modules, instead leveraging models’ existing ability to write code with libraries such as Matplotlib and Turtle. This simple approach shows state-of-the-art results on four difficult natural language tasks that involve visual and spatial reasoning. We identify multiple settings where GPT-4o using chain-of-thought fails dramatically, including more than one where it achieves 0% accuracy, while whiteboard-of-thought enables up to 92% accuracy in these same settings. We present a detailed exploration of where the technique succeeds as well as its sources of error.

</details>

---

## 23. Self-Training Large Language and Vision Assistant for Medical Question Answering

- [ ] Self-Training Large Language and Vision Assistant for Medical Question Answering | https://aclanthology.org/2024.emnlp-main.1119/

- **Link**: https://aclanthology.org/2024.emnlp-main.1119/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have shown significant potential in assisting medical diagnosis by leveraging extensive biomedical datasets. However, the advancement of medical image understanding and reasoning critically depends on building high-quality visual instruction data, which is costly and labor-intensive to obtain, particularly in the medical domain. To mitigate this data-starving issue, we introduce Self-Training Large Language and Vision Assistant for Medical (STLLaVA-Med). The proposed method is designed to train a policy model (an LVLM) capable of auto-generating medical visual instruction data to improve data efficiency, guided through Direct Preference Optimization (DPO). Specifically, a more powerful and larger LVLM (e.g., GPT-4o) is involved as a biomedical expert to oversee the DPO fine-tuning process on the auto-generated data, encouraging the policy model to align efficiently with human preferences. We validate the efficacy and data efficiency of STLLaVA-Med across three major medical Visual Question Answering (VQA) benchmarks, demonstrating competitive zero-shot performance with the utilization of only 9% of the medical data.

</details>

---

## 24. TinyChart: Efficient Chart Understanding with Program-of-Thoughts Learning and Visual Token Merging

- [ ] TinyChart: Efficient Chart Understanding with Program-of-Thoughts Learning and Visual Token Merging | https://aclanthology.org/2024.emnlp-main.112/

- **Link**: https://aclanthology.org/2024.emnlp-main.112/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Charts are important for presenting and explaining complex data relationships. Recently, multimodal large language models (MLLMs) have shown remarkable capabilities in chart understanding. However, the sheer size of these models limits their use in resource-constrained environments. In this paper, we present TinyChart, an efficient MLLM for chart understanding with only 3B parameters. TinyChart overcomes two key challenges in efficient chart understanding: (1) reduce the burden of learning numerical computations through Program-of-Thoughts (PoT) learning, which trains the model to generate Python programs for numerical calculations, and (2) reduce lengthy vision feature sequences through Vision Token Merging, which gradually merges most similar vision tokens. Extensive experiments demonstrate that our 3B TinyChart achieves SOTA performance on various chart understanding benchmarks including ChartQA, Chart-to-Text, Chart-to-Table, OpenCQA, and ChartX. It outperforms several chart-understanding MLLMs with up to 13B parameters, and close-sourced MLLM GPT-4V on ChartQA, with higher throughput during inference due to a smaller model scale and more efficient vision encoding.

</details>

---

## 25. Enhancing Advanced Visual Reasoning Ability of Large Language Models

- [ ] Enhancing Advanced Visual Reasoning Ability of Large Language Models | https://aclanthology.org/2024.emnlp-main.114/

- **Link**: https://aclanthology.org/2024.emnlp-main.114/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language (VL) research have sparked new benchmarks for complex visual reasoning, challenging models’ advanced reasoning ability. Traditional Vision-Language models (VLMs) perform well in visual perception tasks while struggling with complex reasoning scenarios. Conversely, Large Language Models (LLMs) demonstrate robust text reasoning capabilities; however, they lack visual acuity. To bridge this gap, we propose **C**omplex **V**isual **R**easoning **L**arge **L**anguage **M**odels (**CVR-LLM**), capitalizing on VLMs’ visual perception proficiency and LLMs’ extensive reasoning capability. Unlike recent multimodal large language models (MLLMs) that require a projection layer, our approach transforms images into detailed, context-aware descriptions using an iterative self-refinement loop and leverages LLMs’ text knowledge for accurate predictions without extra training. We also introduce a novel multi-modal in-context learning (ICL) methodology to enhance LLMs’ contextual understanding and reasoning. Additionally, we introduce Chain-of-Comparison (CoC), a step-by-step comparison technique enabling contrasting various aspects of predictions. Our CVR-LLM presents the first comprehensive study across a wide array of complex visual reasoning tasks and achieves SOTA performance among all.

</details>

---

## 26. No Culture Left Behind:ArtELingo-28, a Benchmark ofWikiArt with Captions in 28 Languages

- [ ] No Culture Left Behind:ArtELingo-28, a Benchmark ofWikiArt with Captions in 28 Languages | https://aclanthology.org/2024.emnlp-main.1165/

- **Link**: https://aclanthology.org/2024.emnlp-main.1165/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Research in vision and language has made considerable progress thanks to benchmarks such as COCO. COCO captions focused on unambiguous facts in English; ArtEmis introduced subjective emotions and ArtELingo introduced some multilinguality (Chinese and Arabic). However we believe there should be more multilinguality. Hence, we present ArtELingo-28, a vision-language benchmark that spans 28 languages and encompasses approximately 200,000 annotations (140 annotations per image). Traditionally, vision research focused on unambiguous class labels, whereas ArtELingo-28 emphasizes diversity of opinions over languages and cultures. The challenge is to build machine learning systems that assign emotional captions to images. Baseline results will be presented for three novel conditions: Zero-Shot, Few-Shot and One-vs-All Zero-Shot. We find that cross-lingual transfer is more successful for culturally-related languages. Data and code will be made publicly available.

</details>

---

## 27. Retrieval-enriched zero-shot image classification in low-resource domains

- [ ] Retrieval-enriched zero-shot image classification in low-resource domains | https://aclanthology.org/2024.emnlp-main.1186/

- **Link**: https://aclanthology.org/2024.emnlp-main.1186/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Low-resource domains, characterized by scarce data and annotations, present significant challenges for language and visual understanding tasks, with the latter much under-explored in the literature. Recent advancements in Vision-Language Models (VLM) have shown promising results in high-resource domains but fall short in low-resource concepts that are under-represented (e.g. only a handful of images per category) in the pre-training set. We tackle the challenging task of zero-shot low-resource image classification from a novel perspective. By leveraging a retrieval-based strategy, we achieve this in a training-free fashion. Specifically, our method, named CoRE (Combination of Retrieval Enrichment), enriches the representation of both query images and class prototypes by retrieving relevant textual information from large web-crawled databases. This retrieval-based enrichment significantly boosts classification performance by incorporating the broader contextual information relevant to the specific class. We validate our method on a newly established benchmark covering diverse low-resource domains, including medical imaging, rare plants, and circuits. Our experiments demonstrate that CoRE outperforms existing state-of-the-art methods that rely on synthetic data generation and model fine-tuning.

</details>

---

## 28. Show and Guide: Instructional-Plan Grounded Vision and Language Model

- [ ] Show and Guide: Instructional-Plan Grounded Vision and Language Model | https://aclanthology.org/2024.emnlp-main.1191/

- **Link**: https://aclanthology.org/2024.emnlp-main.1191/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Guiding users through complex procedural plans is an inherently multimodal task in which having visually illustrated plan steps is crucial to deliver an effective plan guidance. However, existing works on plan-following language models (LMs) often are not capable of multimodal input and output. In this work, we present MM-PlanLLM, the first multimodal LLM designed to assist users in executing instructional tasks by leveraging both textual plans and visual information. Specifically, we bring cross-modality through two key tasks: Conversational Video Moment Retrieval, where the model retrieves relevant step-video segments based on user queries, and Visually-Informed Step Generation, where the model generates the next step in a plan, conditioned on an image of the user’s current progress. MM-PlanLLM is trained using a novel multitask-multistage approach, designed to gradually expose the model to multimodal instructional-plans semantic layers, achieving strong performance on both multimodal and textual dialogue in a plan-grounded setting. Furthermore, we show that the model delivers cross-modal temporal and plan-structure representations aligned between textual plan steps and instructional video moments.

</details>

---

## 29. An Empirical Analysis on Spatial Reasoning Capabilities of Large Multimodal Models

- [ ] An Empirical Analysis on Spatial Reasoning Capabilities of Large Multimodal Models | https://aclanthology.org/2024.emnlp-main.1195/

- **Link**: https://aclanthology.org/2024.emnlp-main.1195/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs) have achieved strong performance across a range of vision and language tasks. However, their spatial reasoning capabilities are under-investigated. In this paper, we construct a novel VQA dataset, Spatial-MM, to comprehensively study LMMs’ spatial understanding and reasoning capabilities. Our analyses on object-relationship and multi-hop reasoning reveal several important findings. Firstly, bounding boxes and scene graphs, even synthetic ones, can significantly enhance LMMs’ spatial reasoning. Secondly, LMMs struggle more with questions posed from the human perspective than the camera perspective about the image. Thirdly, chain of thought (CoT) prompting does not improve model performance on complex multi-hop questions involving spatial relations. Moreover, spatial reasoning steps are much less accurate than non-spatial ones across MLLMs. Lastly, our perturbation analysis on GQA-spatial reveals that LMMs are much stronger at basic object detection than complex spatial reasoning. We believe our new benchmark dataset and in-depth analyses can spark further research on LMMs spatial reasoning.

</details>

---

## 30. Tag-grounded Visual Instruction Tuning with Retrieval Augmentation

- [ ] Tag-grounded Visual Instruction Tuning with Retrieval Augmentation | https://aclanthology.org/2024.emnlp-main.120/

- **Link**: https://aclanthology.org/2024.emnlp-main.120/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite recent advances in the general visual instruction-following ability of Multimodal Large Language Models (MLLMs), they still struggle with critical problems when required to provide a precise and detailed response to a visual instruction: (1) failure to identify novel objects or entities, (2) mention of non-existent objects, and (3) neglect of object’s attributed details. Intuitive solutions include improving the size and quality of data or using larger foundation models. They show effectiveness in mitigating these issues, but at an expensive cost of collecting a vast amount of new data and introducing a significantly larger model. Standing at the intersection of these approaches, we examine the three object-oriented problems from the perspective of the image-to-text mapping process by the multimodal connector. In this paper, we first identify the limitations of multimodal connectors stemming from insufficient training data. Driven by this, we propose to enhance the mapping with retrieval-augmented tag tokens, which contain rich object-aware information such as object names and attributes. With our Tag-grounded visual instruction tuning with retrieval Augmentation (TUNA), we outperform baselines that share the same language model and training data on 12 benchmarks. Furthermore, we show the zero-shot capability of TUNA when provided with specific datastores.

</details>

---

## 31. Adversarial Text Generation using Large Language Models for Dementia Detection

- [ ] Adversarial Text Generation using Large Language Models for Dementia Detection | https://aclanthology.org/2024.emnlp-main.1222/

- **Link**: https://aclanthology.org/2024.emnlp-main.1222/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Although large language models (LLMs) excel in various text classification tasks, regular prompting strategies (e.g., few-shot prompting) do not work well with dementia detection via picture description. The challenge lies in the language marks for dementia are unclear, and LLM may struggle with relating its internal knowledge to dementia detection. In this paper, we present an accurate and interpretable classification approach by Adversarial Text Generation (ATG), a novel decoding strategy that could relate dementia detection with other tasks. We further develop a comprehensive set of instructions corresponding to various tasks and use them to guide ATG, achieving the best accuracy of 85%, >10% improvement compared to the regular prompting strategies. In addition, we introduce feature context, a human-understandable text that reveals the underlying features of LLM used for classifying dementia. From feature contexts, we found that dementia detection can be related to tasks such as assessing attention to detail, language, and clarity with specific features of the environment, character, and other picture content or language-related features. Future work includes incorporating multi-modal LLMs to interpret speech and picture information.

</details>

---

## 32. Towards Difficulty-Agnostic Efficient Transfer Learning for Vision-Language Models

- [ ] Towards Difficulty-Agnostic Efficient Transfer Learning for Vision-Language Models | https://aclanthology.org/2024.emnlp-main.124/

- **Link**: https://aclanthology.org/2024.emnlp-main.124/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) like CLIP have demonstrated remarkable applicability across a variety of downstream tasks, including zero-shot image classification. Recently, the use of prompts or adapters for efficient transfer learning (ETL) has gained significant attention for effectively adapting to downstream tasks. However, previous studies have overlooked the challenge of varying transfer difficulty of downstream tasks. In this paper, we empirically analyze how each ETL method behaves with respect to transfer difficulty. Our observations indicate that utilizing vision prompts and text adapters is crucial for adaptability and generalizability in domains with high difficulty. Also, by applying an adaptive ensemble approach that integrates task-adapted VLMs with pre-trained VLMs and strategically leverages more general knowledge in low-difficulty and less in high-difficulty domains, we consistently enhance performance across both types of domains. Based on these observations, we propose an adaptive ensemble method that combines visual prompts and text adapters with pre-trained VLMs, tailored by transfer difficulty, to achieve optimal performance for any target domain. Upon experimenting with extensive benchmarks, our method consistently outperforms all baselines, particularly on unseen tasks, demonstrating its effectiveness.

</details>

---

## 33. SimLLM: Detecting Sentences Generated by Large Language Models Using Similarity between the Generation and its Re-generation

- [ ] SimLLM: Detecting Sentences Generated by Large Language Models Using Similarity between the Generation and its Re-generation | https://aclanthology.org/2024.emnlp-main.1246/

- **Link**: https://aclanthology.org/2024.emnlp-main.1246/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models have emerged as a significant phenomenon due to their ability to produce natural text across various applications. However, the proliferation of generated text raises concerns regarding its potential misuse in fraudulent activities such as academic dishonesty, spam dissemination, and misinformation propagation. Prior studies have detected the generation of non-analogous text, which manifests numerous differences between original and generated text. We have observed that the similarity between the original text and its generation is notably higher than that between the generated text and its subsequent regeneration. To address this, we propose a novel approach named SimLLM, aimed at estimating the similarity between an input sentence and its generated counterpart to detect analogous machine-generated sentences that closely mimic human-written ones. Our empirical analysis demonstrates SimLLM’s superior performance compared to existing methods.

</details>

---

## 34. MIBench: Evaluating Multimodal Large Language Models over Multiple Images

- [ ] MIBench: Evaluating Multimodal Large Language Models over Multiple Images | https://aclanthology.org/2024.emnlp-main.1250/

- **Link**: https://aclanthology.org/2024.emnlp-main.1250/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Built on the power of LLMs, numerous multimodal large language models (MLLMs) have recently achieved remarkable performance on various vision-language tasks. However, most existing MLLMs and benchmarks primarily focus on single-image input scenarios, leaving the performance of MLLMs when handling realistic multiple images underexplored. Although a few benchmarks consider multiple images, their evaluation dimensions and samples are very limited. In this paper, we propose a new benchmark MIBench, to comprehensively evaluate fine-grained abilities of MLLMs in multi-image scenarios. Specifically, MIBench categorizes the multi-image abilities into three scenarios: multi-image instruction (MII), multimodal knowledge-seeking (MKS) and multimodal in-context learning (MIC), and constructs 13 tasks with a total of 13K annotated samples. During data construction, for MII and MKS, we extract correct options from manual annotations and create challenging distractors to obtain multiple-choice questions. For MIC, to enable an in-depth evaluation, we set four sub-tasks and transform the original datasets into in-context learning formats. We evaluate several open-source and closed-source MLLMs on the proposed MIBench. The results reveal that although current models excel in single-image tasks, they exhibit significant shortcomings when faced with multi-image inputs, such as limited fine-grained perception, multi-image reasoning and in-context learning abilities. The annotated data of MIBench is available at https://huggingface.co/datasets/StarBottle/MIBench.

</details>

---

## 35. CELLO: Causal Evaluation of Large Vision-Language Models

- [ ] CELLO: Causal Evaluation of Large Vision-Language Models | https://aclanthology.org/2024.emnlp-main.1247/

- **Link**: https://aclanthology.org/2024.emnlp-main.1247/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Causal reasoning is fundamental to human intelligence and crucial for effective decision-making in real-world environments. Despite recent advancements in large vision-language models (LVLMs), their ability to comprehend causality remains unclear. Previous work typically focuses on commonsense causality between events and/or actions, which is insufficient for applications like embodied agents and lacks the explicitly defined causal graphs required for formal causal reasoning. To overcome these limitations, we introduce a fine-grained and unified definition of causality involving interactions between humans and/or objects. Building on the definition, we construct a novel dataset, CELLO, consisting of 14,094 causal questions across all four levels of causality: discovery, association, intervention, and counterfactual. This dataset surpasses traditional commonsense causality by including explicit causal graphs that detail the interactions between humans and objects. Extensive experiments on CELLO reveal that current LVLMs still struggle with causal reasoning tasks, but they can benefit significantly from our proposed CELLO-CoT, a causally inspired chain-of-thought prompting strategy. Both quantitative and qualitative analyses from this study provide valuable insights for future research. Our project page is at https://github.com/OpenCausaLab/CELLO.

</details>

---

## 36. By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting

- [ ] By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting | https://aclanthology.org/2024.emnlp-main.133/

- **Link**: https://aclanthology.org/2024.emnlp-main.133/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have demonstrated exceptional abilities across various domains. However, utilizing LLMs for ubiquitous sensing applications remains challenging as existing text-prompt methods show significant performance degradation when handling long sensor data sequences. In this paper, we propose a visual prompting approach for sensor data using multimodal LLMs (MLLMs). Specifically, we design a visual prompt that directs MLLMs to utilize visualized sensor data alongside descriptions of the target sensory task. Additionally, we introduce a visualization generator that automates the creation of optimal visualizations tailored to a given sensory task, eliminating the need for prior task-specific knowledge. We evaluated our approach on nine sensory tasks involving four sensing modalities, achieving an average of 10% higher accuracy compared to text-based prompts and reducing token costs by 15.8 times. Our findings highlight the effectiveness and cost-efficiency of using visual prompts with MLLMs for various sensory tasks. The source code is available at https://github.com/diamond264/ByMyEyes.

</details>

---

## 37. VIVA: A Benchmark for Vision-Grounded Decision-Making with Human Values

- [ ] VIVA: A Benchmark for Vision-Grounded Decision-Making with Human Values | https://aclanthology.org/2024.emnlp-main.137/

- **Link**: https://aclanthology.org/2024.emnlp-main.137/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces VIVA, a benchmark for VIsion-grounded decision-making driven by human VAlues. While most large vision-language models (VLMs) focus on physical-level skills, our work is the first to examine their multimodal capabilities in leveraging human values to make decisions under a vision-depicted situation. VIVA contains 1,062 images depicting diverse real-world situations and the manually annotated decisions grounded in them. Given an image there, the model should select the most appropriate action to address the situation and provide the relevant human values and reason underlying the decision. Extensive experiments based on VIVA show the limitation of VLMs in using human values to make multimodal decisions. Further analyses indicate the potential benefits of exploiting action consequences and predicted human values.

</details>

---

## 38. African orEuropean Swallow? Benchmarking Large Vision-Language Models for Fine-Grained Object Classification

- [ ] African orEuropean Swallow? Benchmarking Large Vision-Language Models for Fine-Grained Object Classification | https://aclanthology.org/2024.emnlp-main.154/

- **Link**: https://aclanthology.org/2024.emnlp-main.154/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent Large Vision-Language Models (LVLMs) demonstrate impressive abilities on numerous image understanding and reasoning tasks. The task of fine-grained object classification (e.g., distinction betweenanimal species), however, has been probed insufficiently, despite its downstream importance. We fill this evaluation gap by creating FOCI (Fine-grainedObjectClassIfication), a difficult multiple-choice benchmark for fine-grained object classification, from existing object classification datasets: (1) multiple-choice avoids ambiguous answers associated with casting classification as open-ended QA task; (2) we retain classification difficulty by mining negative labels with a CLIP model. FOCI complements five popular classification datasets with four domain-specific subsets from ImageNet-21k. We benchmark 12 public LVLMs on and show that it tests for acomplementary skillto established image understanding and reasoning benchmarks. Crucially, CLIP models exhibit dramatically better performance than LVLMs. Since the image encoders of LVLMs come from these CLIP models, this points to inadequate alignment for fine-grained object distinction between the encoder and the LLM and warrants (pre)training data with more fine-grained annotation. We release our code atANONYMIZED.

</details>

---

## 39. Does Object Grounding Really Reduce Hallucination of Large Vision-Language Models?

- [ ] Does Object Grounding Really Reduce Hallucination of Large Vision-Language Models? | https://aclanthology.org/2024.emnlp-main.159/

- **Link**: https://aclanthology.org/2024.emnlp-main.159/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have recently dramatically pushed the state of the art in image captioning and many image understanding tasks (e.g., visual question answering). LVLMs, however, oftenhallucinateand produce captions that mention concepts that cannot be found in the image. These hallucinations erode the trustworthiness of LVLMs and are arguably among the main obstacles to their ubiquitous adoption. Recent work suggests that addition of grounding objectives—those that explicitly align image regions or objects to text spans—reduces the amount of LVLM hallucination. Although intuitive, this claim is not empirically justified as the reduction effects have been established, we argue, with flawed evaluation protocols that (i) rely on data (i.e., MSCOCO) that has been extensively used in LVLM training and (ii) measure hallucination via question answering rather than open-ended caption generation.In this work, in contrast, we offer the first systematic analysis of the effect of fine-grained object grounding on LVLM hallucination under an evaluation protocol that more realistically captures LVLM hallucination in open generation. Our extensive experiments over three backbone LLMs reveal that grounding objectives have little to no effect on object hallucination in open caption generation.

</details>

---

## 40. With Ears to See and Eyes to Hear: Sound Symbolism Experiments with Multimodal Large Language Models

- [ ] With Ears to See and Eyes to Hear: Sound Symbolism Experiments with Multimodal Large Language Models | https://aclanthology.org/2024.emnlp-main.167/

- **Link**: https://aclanthology.org/2024.emnlp-main.167/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, Large Language Models (LLMs) and Vision Language Models (VLMs) have demonstrated aptitude as potential substitutes for human participants in experiments testing psycholinguistic phenomena. However, an understudied question is to what extent models that only have access to vision and text modalities are able to implicitly understand sound-based phenomena via abstract reasoning from orthography and imagery alone. To investigate this, we analyse the ability of VLMs and LLMs to demonstrate sound symbolism (i.e., to recognise a non-arbitrary link between sounds and concepts) as well as their ability to “hear” via the interplay of the language and vision modules of open and closed-source multimodal models. We perform multiple experiments, including replicating the classic Kiki-Bouba and Mil-Mal shape and magnitude symbolism tasks and comparing human judgements of linguistic iconicity with that of LLMs. Our results show that VLMs demonstrate varying levels of agreement with human labels, and more task information may be required for VLMs versus their human counterparts forin silicoexperimentation. We additionally see through higher maximum agreement levels that Magnitude Symbolism is an easier pattern for VLMs to identify than Shape Symbolism, and that an understanding of linguistic iconicity is highly dependent on model size.

</details>

---

## 41. DocKD: Knowledge Distillation fromLLMs for Open-World Document Understanding Models

- [ ] DocKD: Knowledge Distillation fromLLMs for Open-World Document Understanding Models | https://aclanthology.org/2024.emnlp-main.185/

- **Link**: https://aclanthology.org/2024.emnlp-main.185/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual document understanding (VDU) is a challenging task that involves understanding documents across various modalities (text and image) and layouts (forms, tables, etc.). This study aims to enhance generalizability of small VDU models by distilling knowledge from LLMs. We identify that directly prompting LLMs often fails to generate informative and useful data. In response, we present a new framework (called DocKD) that enriches the data generation process by integrating external document knowledge. Specifically, we provide an LLM with various document elements like key-value pairs, layouts, and descriptions, to elicit open-ended answers. Our experiments show that DocKD produces high-quality document annotations and surpasses the direct knowledge distillation approach that does not leverage external document knowledge. Moreover, student VDU models trained with solely DocKD-generated data is not only comparable to those trained with human-annotated data on in-domain tasks but also significantly excel them on out-of-domain tasks.

</details>

---

## 42. Fine-Grained Prediction of Reading Comprehension from Eye Movements

- [ ] Fine-Grained Prediction of Reading Comprehension from Eye Movements | https://aclanthology.org/2024.emnlp-main.198/

- **Link**: https://aclanthology.org/2024.emnlp-main.198/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Can human reading comprehension be assessed from eye movements in reading? In this work, we address this longstanding question using large-scale eyetracking data. We focus on a cardinal and largely unaddressed variant of this question: predicting reading comprehension of a single participant for a single question from their eye movements over a single paragraph. We tackle this task using a battery of recent models from the literature, and three new multimodal language models. We evaluate the models in two different reading regimes: ordinary reading and information seeking, and examine their generalization to new textual items, new participants, and the combination of both. The evaluations suggest that the task is highly challenging, and highlight the importance of benchmarking against a strong text-only baseline. While in some cases eye movements provide improvements over such a baseline, they tend to be small. This could be due to limitations of current modelling approaches, limitations of the data, or because eye movement behavior does not sufficiently pertain to fine-grained aspects of reading comprehension processes. Our study provides an infrastructure for making further progress on this question.

</details>

---

## 43. Decompose and Compare Consistency: MeasuringVLMs’ Answer Reliability via Task-Decomposition Consistency Comparison

- [ ] Decompose and Compare Consistency: MeasuringVLMs’ Answer Reliability via Task-Decomposition Consistency Comparison | https://aclanthology.org/2024.emnlp-main.211/

- **Link**: https://aclanthology.org/2024.emnlp-main.211/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite tremendous advancements, current state-of-the-art Vision-Language Models (VLMs) are still far from perfect. They tend to hallucinate and may generate biased responses. In such circumstances, having a way to assess the reliability of a given response generated by a VLM is quite useful. Existing methods, such as estimating uncertainty using answer likelihoods or prompt-based confidence generation, often suffer from overconfidence. Other methods use self-consistency comparison but are affected by confirmation biases. To alleviate these, we propose Decompose and Compare Consistency (DeCC) for reliability measurement. By comparing the consistency between the direct answer generated using the VLM’s internal reasoning process, and the indirect answers obtained by decomposing the question into sub-questions and reasoning over the sub-answers produced by the VLM, DeCC measures the reliability of VLM’s direct answer. Experiments across six vision-language tasks with three VLMs show DeCC’s reliability estimation achieves better correlation with task accuracy compared to the existing methods.

</details>

---

## 44. VGBench: Evaluating Large Language Models on Vector Graphics Understanding and Generation

- [ ] VGBench: Evaluating Large Language Models on Vector Graphics Understanding and Generation | https://aclanthology.org/2024.emnlp-main.213/

- **Link**: https://aclanthology.org/2024.emnlp-main.213/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the realm of vision models, the primary mode of representation is using pixels to rasterize the visual world. Yet this is not always the best or unique way to represent visual content, especially for designers and artists who depict the world using geometry primitives such as polygons. Vector graphics (VG), on the other hand, offer a textual representation of visual content, which can be more concise and powerful for content like cartoons, sketches and scientific figures. Recent studies have shown promising results on processing vector graphics with capable Large Language Models (LLMs). However, such works focus solely on qualitative results, understanding, or a specific type of vector graphics. We propose VGBench, a comprehensive benchmark for LLMs on handling vector graphics through diverse aspects, including (a) both visual understanding and generation, (b) evaluation of various vector graphics formats, (c) diverse question types, (d) wide range of prompting techniques, (e) under multiple LLMs and (f) comparison with VLMs on rasterized representations. Evaluating on our collected 4279 understanding and 5845 generation samples, we find that LLMs show strong capability on both aspects while exhibiting less desirable performance on low-level formats (SVG). Both data and evaluation pipeline will be open-sourced.

</details>

---

## 45. M2PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning

- [ ] M2PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning | https://aclanthology.org/2024.emnlp-main.218/

- **Link**: https://aclanthology.org/2024.emnlp-main.218/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) demonstrate remarkable performance across a wide range of domains, with increasing emphasis on enhancing their zero-shot generalization capabilities for unseen tasks across various modalities. Instruction tuning has emerged as an effective strategy for achieving zero-shot generalization by finetuning pretrained models on diverse multimodal tasks. As the scale of MLLMs continues to grow, parameter-efficient finetuning becomes increasingly critical. However, most existing parameter-efficient approaches focus only on single modalities and often overlook the multimodal characteristics during finetuning. In this work, we introduce a novel Multimodal Prompt Tuning (M2PT) approach for efficient instruction tuning of MLLMs. M2PT effectively integrates visual and textual prompts into the vision encoder and language processor respectively during finetuning, facilitating the extraction and alignment of features across modalities. Empirical results on various multimodal evaluation datasets demonstrate the superior performance of our approach compared to several state-of-the-art baselines. A comprehensive set of ablation studies validates the effectiveness of our prompt design and the efficiency of our approach.

</details>

---

## 46. Visual Prompting inLLMs for Enhancing Emotion Recognition

- [ ] Visual Prompting inLLMs for Enhancing Emotion Recognition | https://aclanthology.org/2024.emnlp-main.257/

- **Link**: https://aclanthology.org/2024.emnlp-main.257/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision Large Language Models (VLLMs) are transforming the intersection of computer vision and natural language processing; however, the potential of using visual prompts for emotion recognition in these models remains largely unexplored and untapped. Traditional methods in VLLMs struggle with spatial localization and often discard valuable global context. We propose a novel Set-of-Vision prompting (SoV) approach that enhances zero-shot emotion recognition by using spatial information, such as bounding boxes and facial landmarks, to mark targets precisely. SoV improves accuracy in face count and emotion categorization while preserving the enriched image context. Through comprehensive experimentation and analysis of recent commercial or open-source VLLMs, we evaluate the SoV model’s ability to comprehend facial expressions in natural environments. Our findings demonstrate the effectiveness of integrating spatial visual prompts into VLLMs for improving emotion recognition performance.

</details>

---

## 47. World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering

- [ ] World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering | https://aclanthology.org/2024.emnlp-main.265/

- **Link**: https://aclanthology.org/2024.emnlp-main.265/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Vision-Language Models (VLMs) and the scarcity of high-quality multi-modal alignment data have inspired numerous researches on synthetic VLM data generation. The conventional norm in VLM data construction uses a mixture of specialists in caption and OCR, or stronger VLM APIs and expensive human annotation.In this paper, we present World to Code (W2C), a meticulously curated multi-modal data construction pipeline that organizes the final generation output into a Python code format. The pipeline leverages the VLM itself to extract cross-modal information via different prompts and filter the generated outputs again via a consistency filtering strategy. Experiments have demonstrated the high quality ofW2Cby improving various existing visual question answering and visual grounding benchmarks across different VLMs. Further analysis also demonstrates that the new code parsing ability of VLMs presents better cross-modal equivalence than the commonly used detail caption ability. Our code is available at https://github.com/foundation-multimodal-models/World2Code.

</details>

---

## 48. RWKV-CLIP: A Robust Vision-Language Representation Learner

- [ ] RWKV-CLIP: A Robust Vision-Language Representation Learner | https://aclanthology.org/2024.emnlp-main.276/

- **Link**: https://aclanthology.org/2024.emnlp-main.276/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastive Language-Image Pre-training (CLIP) has significantly improved performance in various vision-language tasks by expanding the dataset with image-text pairs obtained from the web. This paper further explores CLIP from the perspectives of data and model architecture. To mitigate the impact of the noise data and enhance the quality of large-scale image-text data crawled from the internet, we introduce a diverse description generation framework that can leverage Large Language Models (LLMs) to combine and refine information from web-based image-text pairs, synthetic captions, and detection tags. Additionally, we propose RWKV-CLIP, the first RWKV-driven vision-language representation learning model that combines the effective parallel training of transformers with the efficient inference of RNNs. Extensive experiments across different model scales and pre-training datasets demonstrate that RWKV-CLIP is a robust vision-language representation learner and it achieves state-of-the-art performance across multiple downstream tasks, including linear probing, zero-shot classification, and zero-shot image-text retrieval. To facilitate future research, the code and pre-trained models are released at https://github.com/deepglint/RWKV-CLIP.

</details>

---

## 49. From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis

- [ ] From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis | https://aclanthology.org/2024.emnlp-main.284/

- **Link**: https://aclanthology.org/2024.emnlp-main.284/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We explore multi-step reasoning in vision-language models (VLMs). The problem is challenging, as reasoning data consisting of multiple steps of visual and language processing are barely available. To overcome the challenge, we first introduce a least-to-most visual reasoning paradigm, which interleaves steps of decomposing a question into sub-questions and invoking external tools for resolving sub-questions. Based on the paradigm, we further propose a novel data synthesis approach that can automatically create questions and multi-step reasoning paths for an image in a bottom-up manner. Our approach divides the complex synthesis task into a few simple sub-tasks, and (almost entirely) relies on open-sourced models to accomplish the sub-tasks. Therefore, the entire synthesis process is reproducible and cost-efficient, and the synthesized data is quality guaranteed. With the approach, we construct 50k visual reasoning examples. Then, we develop a visual reasoner through supervised fine-tuning, which is capable of generally enhancing the reasoning abilities of a wide range of existing VLMs in a plug-and-play fashion. Extensive experiments indicate that the visual reasoner can consistently and significantly improve four VLMs on four VQA benchmarks.

</details>

---

## 50. Concept-skill Transferability-based Data Selection for Large Vision-Language Models

- [ ] Concept-skill Transferability-based Data Selection for Large Vision-Language Models | https://aclanthology.org/2024.emnlp-main.291/

- **Link**: https://aclanthology.org/2024.emnlp-main.291/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuning, or supervised finetuning on extensive task-specific data, is necessary for Large Vision-Language Models (LVLMs) to generalize well across a broad range of vision-language (VL) tasks. However, training on large VL datasets can become prohibitively expensive. In this work, we introduce COINCIDE, an effective and scalable data selection technique that uses a small model as a reference model to select visual instruction tuning data for efficient finetuning of a target LVLM, focusing on diversity and transferability. Specifically, we cluster the training data using internal activations from a small model, which identifies VL concept-skill compositions needed by a target LVLM. We then sample data from these diverse clusters by considering their density and transferability, or the ability to transfer well to other concept-skill compositions. This approach ensures the diversity of these compositions, which is vital for LVLM generalization. Extensive experiments demonstrate that COINCIDE achieves superior performance and data selection efficiency against 8 strong baselines on two distinct datasets: LLaVA-1.5 and Vision-Flan. Using only 20% of the LLaVA-1.5 dataset, COINCIDE achieves performance comparable to the LVLM finetuned on the whole dataset, with 70% reduction of the wall-clock running time. On the Vision-Flan dataset, our method achieves superior results with only 16.7% of the training data.

</details>

---

## 51. How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?

- [ ] How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning? | https://aclanthology.org/2024.emnlp-main.305/

- **Link**: https://aclanthology.org/2024.emnlp-main.305/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The increase in parameter size of multimodal large language models (MLLMs) introduces significant capabilities, particularly multimodal in-context learning, where MLLMs enhance task performance without updating pre-trained parameters. However, this effectiveness hinges on the appropriate selection of in-context examples, a process currently biased towards visual data, overlooking textual information. More importantly, the area of supervised retrievers for retrieval of multimodal in-context learning, crucial for optimal in-context example selection, continues to be investigated. Our study provides an in-depth evaluation of the impact of textual information on the unsupervised selection of in-context examples in multimodal contexts, uncovering a notable sensitivity of retriever performance to the employed modalities. Based on the above finding, we introduce a novel supervised MLLM prompt retriever MSIER that leverages a trained retriever based on MLLM’s confidence to select examples, which enhances multimodal in-context learning efficiency. This approach is validated through extensive testing across three different tasks, demonstrating the method’s effectiveness. Additionally, we investigate the influence of modalities on our supervised retrieval method’s training and explore the transferability of the supervised prompt retriever. This exploration paves the way for future advancements, highlighting the potential for refined in-context learning in MLLMs through the strategic use of multimodal data. The public code is available at https://github.com/NUS-HPC-AI-Lab/Multimodal-ICL-Retriever.

</details>

---

## 52. To Preserve or To Compress: An In-Depth Study of Connector Selection in Multimodal Large Language Models

- [ ] To Preserve or To Compress: An In-Depth Study of Connector Selection in Multimodal Large Language Models | https://aclanthology.org/2024.emnlp-main.325/

- **Link**: https://aclanthology.org/2024.emnlp-main.325/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In recent years, multimodal large language models (MLLMs) have attracted widespread attention from both industry and academia. Based on the integration position, MLLMs can be categorized into external and internal fusion architectures, with the former being more predominant. However, there remains considerable debate on how to construct the optimal external fusion MLLM architecture, especially regarding the performance of different connectors on tasks with varying granularities. This paper systematically investigates the impact of connectors on MLLM performance. Specifically, we classify connectors into feature-preserving and feature-compressing types. Utilizing a unified classification standard, we categorize sub-tasks from three comprehensive benchmarks, MMBench, MME, and SEED-Bench, into three task types: coarse-grained perception, fine-grained perception, and reasoning, and evaluate the performance from this perspective. Our findings reveal significant performance differences between different types of connectors across various tasks, offering essential guidance for MLLM architecture design and advancing the understanding of MLLM architecture optimization.

</details>

---

## 53. Benchmarking Vision Language Models for Cultural Understanding

- [ ] Benchmarking Vision Language Models for Cultural Understanding | https://aclanthology.org/2024.emnlp-main.329/

- **Link**: https://aclanthology.org/2024.emnlp-main.329/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Foundation models and vision-language pre-training have notably advanced Vision Language Models (VLMs), enabling multimodal processing of visual and linguistic data. However, their performance has been typically assessed on general scene understanding - recognizing objects, attributes, and actions - rather than cultural comprehension. This study introduces CulturalVQA, a visual question-answering benchmark aimed at assessing VLM’s geo-diverse cultural understanding. We curate a diverse collection of 2,378 image-question pairs with 1-5 answers per question representing cultures from 11 countries across 5 continents. The questions probe understanding of various facets of culture such as clothing, food, drinks, rituals, and traditions. Benchmarking VLMs on CulturalVQA, including GPT-4V and Gemini, reveals disparity in their level of cultural understanding across regions, with strong cultural understanding capabilities for North America while significantly weaker capabilities for Africa. We observe disparity in their performance across cultural facets too, with clothing, rituals, and traditions seeing higher performances than food and drink. These disparities help us identify areas where VLMs lack cultural understanding and demonstrate the potential of CulturalVQA as a comprehensive evaluation set for gauging VLM progress in understanding diverse cultures.

</details>

---

## 54. Analyzing Key Factors Influencing Emotion Prediction Performance ofVLLMs in Conversational Contexts

- [ ] Analyzing Key Factors Influencing Emotion Prediction Performance ofVLLMs in Conversational Contexts | https://aclanthology.org/2024.emnlp-main.331/

- **Link**: https://aclanthology.org/2024.emnlp-main.331/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Emotional intelligence (EI) in artificial intelligence (AI), which refers to the ability of an AI to understand and respond appropriately to human emotions, has emerged as a crucial research topic. Recent studies have shown that large language models (LLMs) and vision large language models (VLLMs) possess EI and the ability to understand emotional stimuli in the form of text and images, respectively. However, factors influencing the emotion prediction performance of VLLMs in real-world conversational contexts have not been sufficiently explored. This study aims to analyze the key elements affecting the emotion prediction performance of VLLMs in conversational contexts systematically. To achieve this, we reconstructed the MELD dataset, which is based on the popular TV series Friends, and conducted experiments through three sub-tasks: overall emotion tone prediction, character emotion prediction, and contextually appropriate emotion expression selection. We evaluated the performance differences based on various model architectures (e.g., image encoders, modality alignment, and LLMs) and image scopes (e.g., entire scene, person, and facial expression). In addition, we investigated the impact of providing persona information on the emotion prediction performance of the models and analyzed how personality traits and speaking styles influenced the emotion prediction process. We conducted an in-depth analysis of the impact of various other factors, such as gender and regional biases, on the emotion prediction performance of VLLMs. The results revealed that these factors significantly influenced the model performance.

</details>

---

## 55. Quantifying the Gaps Between Translation and Native Perception in Training for Multimodal, Multilingual Retrieval

- [ ] Quantifying the Gaps Between Translation and Native Perception in Training for Multimodal, Multilingual Retrieval | https://aclanthology.org/2024.emnlp-main.335/

- **Link**: https://aclanthology.org/2024.emnlp-main.335/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

There is a scarcity of multilingual vision-language models that properly account for the perceptual differences that are reflected in image captions across languages and cultures. In this work, through a multimodal, multilingual retrieval case study, we quantify the existing lack of model flexibility. We empirically show performance gaps between training on captions that come from native German perception and captions that have been either machine-translated or human-translated from English into German. To address these gaps, we further propose and evaluate caption augmentation strategies. While we achieve mean recall improvements (+1.3), gaps still remain, indicating an open area of future work for the community.

</details>

---

## 56. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection

- [ ] Video-LLaVA: Learning United Visual Representation by Alignment Before Projection | https://aclanthology.org/2024.emnlp-main.342/

- **Link**: https://aclanthology.org/2024.emnlp-main.342/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Model (LVLM) has enhanced the performance of various downstream tasks in visual-language understanding. Most existing approaches encode images and videos into separate feature spaces, which are then fed as inputs to large language models. However, due to the lack of unified tokenization for images and videos, namely misalignment before projection, it becomes challenging for a Large Language Model (LLM) to learn multi-modal interactions from several poor projection layers.In this work, we unify visual representation into the language feature space to advance the foundational LLM towards a unified LVLM. As a result, we establish a simple but robust LVLM baseline, Video-LLaVA, which learns from a mixed dataset of images and videos, mutually enhancing each other.As a result, Video-LLaVA outperforms Video-ChatGPT by 5.8%, 9.9%, 18.6%, and 10.1% on MSRVTT, MSVD, TGIF, and ActivityNet, respectively. Additionally, our Video-LLaVA also achieves superior performances on a broad range of 9 image benchmarks.Notably, extensive experiments demonstrate that Video-LLaVA mutually benefits images and videos within a unified visual representation, outperforming models designed specifically for images or videos. We aim for this work to provide modest insights into the multi-modal inputs for the LLM.

</details>

---

## 57. Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models

- [ ] Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models | https://aclanthology.org/2024.emnlp-main.356/

- **Link**: https://aclanthology.org/2024.emnlp-main.356/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in instruction-tuned Large Vision-Language Models (LVLMs) have imbued the models with the ability to generate high-level, image-grounded explanations with ease. While such capability is largely attributed to the rich world knowledge contained within the Large Language Models (LLMs), our work reveals their shortcomings in fine-grained visual categorization (FGVC) across six different benchmark settings. Most recent state-of-the-art LVLMs such as LLaVa-1.5, InstructBLIP and GPT-4V not only severely deteriorate in terms of classification performance, e.g., average drop of 65.58 in EM for Stanford Dogs for LLaVA-1.5, but also struggle to generate descriptive visual attributes based on a concept that appears within an input image despite their prominent zero-shot image captioning ability. In-depth analyses show that instruction-tuned LVLMs suffer from modality gap, showing discrepancy when given textual and visual inputs that correspond to the same concept. In an effort to further the community’s endeavor in this direction, we propose a multiple granularity attribute-centric benchmark and training mixture, Finer, which aims to establish a ground to evaluate LVLMs’ fine-grained visual comprehension ability and provide significantly improved explainability.

</details>

---

## 58. VLFeedback: A Large-ScaleAIFeedback Dataset for Large Vision-Language Models Alignment

- [ ] VLFeedback: A Large-ScaleAIFeedback Dataset for Large Vision-Language Models Alignment | https://aclanthology.org/2024.emnlp-main.358/

- **Link**: https://aclanthology.org/2024.emnlp-main.358/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As large vision-language models (LVLMs) evolve rapidly, the demand for high-quality and diverse data to align these models becomes increasingly crucial. However, the creation of such data with human supervision proves costly and time-intensive. In this paper, we investigate the efficacy of AI feedback to scale supervision for aligning LVLMs. We introduce VLFeedback, the first large-scale vision-language feedback dataset, comprising over 82K multi-modal instructions and comprehensive rationales generated by off-the-shelf models without human annotations. To evaluate the effectiveness of AI feedback for vision-language alignment, we train Silkie, an LVLM fine-tuned via direct preference optimization on VLFeedback. Silkie showcases exceptional performance regarding helpfulness, visual faithfulness, and safety metrics. It outperforms its base model by 6.9% and 9.5% in perception and cognition tasks, reduces hallucination issues on MMHal-Bench, and exhibits enhanced resilience against red-teaming attacks. Furthermore, our analysis underscores the advantage of AI feedback, particularly in fostering preference diversity to deliver more comprehensive improvements. Our dataset, training code and models are available athttps://vlf-silkie.github.io.

</details>

---

## 59. UOUO: Uncontextualized Uncommon Objects for Measuring Knowledge Horizons of Vision Language Models

- [ ] UOUO: Uncontextualized Uncommon Objects for Measuring Knowledge Horizons of Vision Language Models | https://aclanthology.org/2024.emnlp-main.369/

- **Link**: https://aclanthology.org/2024.emnlp-main.369/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Smaller-scale Vision-Language Models (VLMs) often claim to perform on par with larger models in general-domain visual grounding and question-answering benchmarks while offering advantages in computational efficiency and storage. However, their ability to handle rare objects, which fall into the long tail of data distributions, is less understood. To rigorously evaluate this aspect, we introduce the “Uncontextualized Uncommon Objects” (UOUO) benchmark. This benchmark focuses on systematically testing VLMs with both large and small parameter counts on rare and specialized objects. Our comprehensive analysis reveals that while smaller VLMs maintain competitive performance on common datasets, they significantly underperform on tasks involving uncommon objects. We also propose an advanced, scalable pipeline for data collection and cleaning, ensuring the UOUO benchmark provides high-quality, challenging instances. These findings highlight the need to consider long-tail distributions when assessing the true capabilities of VLMs. Code and project details for UOUO can be found at https://zoezheng126.github.io/UOUO-Website/.

</details>

---

## 60. Unifying Multimodal Retrieval via Document Screenshot Embedding

- [ ] Unifying Multimodal Retrieval via Document Screenshot Embedding | https://aclanthology.org/2024.emnlp-main.373/

- **Link**: https://aclanthology.org/2024.emnlp-main.373/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the real world, documents are organized in different formats and varied modalities. Traditional retrieval pipelines require tailored document parsing techniques and content extraction modules to prepare input for indexing. This process is tedious, prone to errors, and has information loss. To this end, we propose Document Screenshot Embedding (DSE), a novel retrieval paradigm that regards document screenshots as a unified input format, which does not require any content extraction preprocess and preserves all the information in a document (e.g., text, image and layout). DSE leverages a large vision-language model to directly encode document screenshots into dense representations for retrieval. To evaluate our method, we first craft the dataset of Wiki-SS, a 1.3M Wikipedia web page screenshots as the corpus to answer the questions from the Natural Questions dataset. In such a text-intensive document retrieval setting, DSE shows competitive effectiveness compared to other text retrieval methods relying on parsing. For example, DSE outperforms BM25 by 17 points in top-1 retrieval accuracy. Additionally, in a mixed-modality task of slide retrieval, DSE significantly outperforms OCR text retrieval methods by over 15 points in nDCG@10. These experiments show that DSE is an effective document retrieval paradigm for diverse types of documents. Model checkpoints, code, and Wiki-SS collection will be released.

</details>

---

## 61. From Local Concepts to Universals: Evaluating the Multicultural Understanding of Vision-Language Models

- [ ] From Local Concepts to Universals: Evaluating the Multicultural Understanding of Vision-Language Models | https://aclanthology.org/2024.emnlp-main.385/

- **Link**: https://aclanthology.org/2024.emnlp-main.385/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite recent advancements in vision-language models, their performance remains suboptimal on images from non-western cultures due to underrepresentation in training datasets. Various benchmarks have been proposed to test models’ cultural inclusivity. Still, they have limited coverage of cultures and do not adequately assess cultural diversity across universal and culture-specific local concepts. To address these limitations, we introduce the GlobalRG benchmark, comprising two challenging tasks: retrieval across universals and cultural visual grounding. The former task entails retrieving culturally diverse images for universal concepts from 50 countries, while the latter aims at grounding culture-specific concepts within images from 15 countries. Our evaluation across a wide range of models reveals that the performance varies significantly across cultures – underscoring the necessity for enhancing multicultural understanding in vision-language models.

</details>

---

## 62. MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model

- [ ] MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model | https://aclanthology.org/2024.emnlp-main.387/

- **Link**: https://aclanthology.org/2024.emnlp-main.387/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Projecting visual features into word embedding space has become a significant fusion strategy adopted by Multimodal Large Language Models (MLLMs). However, its internal mechanisms have yet to be explored. Inspired by multilingual research, we identify domain-specific neurons in multimodal large language models. Specifically, we investigate the distribution of domain-specific neurons and the mechanism of how MLLMs process features from diverse domains. Furthermore, we propose a three-stage framework for language model modules in MLLMs when handling projected image features, and verify this hypothesis using logit lens. Extensive experiments indicate that while current MLLMs exhibit Visual Question Answering (VQA) capability, they may not fully utilize domain-specific information. Manipulating domain-specific neurons properly will result in a 10% change of accuracy at most, shedding light on the development of cross-domain, all-encompassing MLLMs in the future. The source code is available at https://anonymous.4open.science/r/MMNeuron.

</details>

---

## 63. Beyond Embeddings: The Promise of Visual Table in Visual Reasoning

- [ ] Beyond Embeddings: The Promise of Visual Table in Visual Reasoning | https://aclanthology.org/2024.emnlp-main.391/

- **Link**: https://aclanthology.org/2024.emnlp-main.391/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual representation learning has been a cornerstone in computer vision, involving typical forms such as visual embeddings, structural symbols, and text-based representations. Despite the success of CLIP-type visual embeddings, they often lack access to world knowledge critical for visual reasoning. In this work, we propose Visual Table, a novel form of visual representation tailored for visual reasoning. Visual tables are constructed as hierarchical descriptions of visual scenes, featuring a scene description and multiple object-centric descriptions covering categories, attributes, and knowledge. Thanks to the structural and textual formats, visual tables offer unique properties over mere visual embeddings, such as explainability and controllable editing. Furthermore, they deliver instance-level world knowledge and detailed attributes that are essential for visual reasoning. To create visual tables, we develop a generator trained on the dataset with collected, small-scale annotations. Extensive results on 11 visual reasoning benchmarks demonstrate that the generated visual tables significantly outperform previous structural and text-based representations. Moreover, they consistently enhance state-of-the-art multi-modal large language models across diverse benchmarks, showcasing their potential for advancing visual reasoning tasks. Our code is available at https://github.com/LaVi-Lab/Visual-Table.

</details>

---

## 64. Towards Injecting Medical Visual Knowledge into MultimodalLLMs at Scale

- [ ] Towards Injecting Medical Visual Knowledge into MultimodalLLMs at Scale | https://aclanthology.org/2024.emnlp-main.418/

- **Link**: https://aclanthology.org/2024.emnlp-main.418/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of multimodal large language models (MLLMs), such as GPT-4V, has led to significant advancements. However, these models still face challenges in medical multimodal capabilities due to limitations in the quantity and quality of medical vision-text data, stemming from data privacy concerns and high annotation costs. While pioneering approaches utilize PubMed’s large-scale, de-identified medical image-text pairs to address these limitations, they often fall short due to inherent data noise. To tackle this, we refined medical image-text pairs from PubMed and employed MLLMs (GPT-4V) in an ‘unblinded’ capacity to denoise and reformat the data, resulting in the creation of the **PubMedVision** dataset with 1.3 million medical VQA samples. Our validation demonstrates that: (1) PubMedVision can significantly enhance the medical multimodal capabilities of MLLMs, showing significant improvement in benchmarks including the MMMU Health & Medicine track; (2) manual checks by medical experts and empirical results validate the superior data quality of our dataset compared to other data construction methods. Using PubMedVision, we train a 34B medical MLLM **HuatuoGPT-Vision**, which shows superior performance in medical multimodal scenarios among open-source MLLMs. Our code and data are available at https://github.com/FreedomIntelligence/HuatuoGPT-Vision.

</details>

---

## 65. Divide and Conquer Radiology Report Generation via Observation Level Fine-grained Pretraining and Prompt Tuning

- [ ] Divide and Conquer Radiology Report Generation via Observation Level Fine-grained Pretraining and Prompt Tuning | https://aclanthology.org/2024.emnlp-main.433/

- **Link**: https://aclanthology.org/2024.emnlp-main.433/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The automation of radiology report generation (RRG) holds immense potential to alleviate radiologists’ workloads and improve diagnostic accuracy. Despite advancements in image captioning and vision-language pretraining, RRG remains challenging due to the lengthy and complex nature of radiology reports. In this work, we proposes the Divide and Conquer Radiology Report Generation (DCRRG) model, which breaks down full-text radiology reports into concise observation descriptions. This approach enables the model to capture fine-grained representations from each observation through a two-stage process: an encoding stage focusing on observation prediction tasks to learn fine-grained representations, and a decoding stage for integrating these descriptions into cohesive and comprehensive radiology reports. Experimental results on two benchmark datasets demonstrate that DCRRG achieves significant improvements across all evaluated metrics, underscoring its capability to generate semantically coherent and clinically accurate radiology reports.

</details>

---

## 66. SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information

- [ ] SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information | https://aclanthology.org/2024.emnlp-main.434/

- **Link**: https://aclanthology.org/2024.emnlp-main.434/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have become pivotal at the intersection of computer vision and natural language processing. However, the full potential of LVLMs’ Retrieval-Augmented Generation (RAG) capabilities remains underutilized. Existing works either focus solely on the text modality or are limited to specific tasks. Moreover, most LVLMs struggle to selectively utilize retrieved information and are sensitive to irrelevant or misleading references. To address these challenges, we propose a self-refinement framework designed to teach LVLMs toSelectivelyUtilizeRetrieved Information (SURf). Specifically, when given questions that are incorrectly answered by the LVLM backbone, we obtain references that help correct the answers (positive references) and those that do not (negative references). We then fine-tune the LVLM backbone using a combination of these positive and negative references. Our experiments across three tasks and seven datasets demonstrate that our framework significantly enhances LVLMs’ ability to effectively utilize retrieved multimodal references and improves their robustness against irrelevant or misleading information. The source code is available at https://anonymous.4open.science/r/SURf-6433.

</details>

---

## 67. DAMRO: Dive into the Attention Mechanism ofLVLMto Reduce Object Hallucination

- [ ] DAMRO: Dive into the Attention Mechanism ofLVLMto Reduce Object Hallucination | https://aclanthology.org/2024.emnlp-main.439/

- **Link**: https://aclanthology.org/2024.emnlp-main.439/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the great success of Large Vision-Language Models (LVLMs), they inevitably suffer from hallucination. As we know, both the visual encoder and the Large Language Model (LLM) decoder in LVLMs are Transformer-based, allowing the model to extract visual information and generate text outputs via attention mechanisms. We find that the attention distribution of LLM decoder on image tokens is highly consistent with the visual encoder and both distributions tend to focus on particular background tokens rather than the referred objects in the image. We attribute to the unexpected attention distribution to an inherent flaw in the visual encoder itself, which misguides LLMs to over emphasize the redundant information and generate object hallucination. To address the issue, we propose DAMRO, a novel training-free strategy that **D**ive into **A**ttention **M**echanism of LVLM to **R**educe **O**bject Hallucination. Specifically, our approach employs classification token (CLS) of ViT to filter out high-attention tokens scattered in the background and then eliminate their influence during decoding stage. We evaluate our method on LVLMs including LLaVA-1.5, LLaVA-NeXT and InstructBLIP, using various benchmarks such as POPE, CHAIR, MME and GPT-4V Aided Evaluation. The results demonstrate that our approach significantly reduces the impact of these outlier tokens, thus effectively alleviating the hallucination of LVLMs.

</details>

---

## 68. GeoGPT4V: Towards Geometric Multi-modal Large Language Models with Geometric Image Generation

- [ ] GeoGPT4V: Towards Geometric Multi-modal Large Language Models with Geometric Image Generation | https://aclanthology.org/2024.emnlp-main.44/

- **Link**: https://aclanthology.org/2024.emnlp-main.44/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models have seen widespread adoption in math problem-solving, yet for geometry problems, which often necessitate visual aids even for humans, the most advanced multi-modal models still struggle to effectively utilize image information. High-quality data is crucial for enhancing the geometric capabilities of multi-modal models, yet existing open-source datasets and related efforts are either too challenging for direct model learning or suffer from misalignment between text and images. To overcome this issue, we introduce a novel pipeline that leverages GPT-4 and GPT-4V to generate relatively basic geometry problems with aligned text and images, facilitating model learning. We have produced a dataset of 4.9K geometry problems and combined it with 19K open-source data to form our GeoGPT4V dataset. Experimental results demonstrate that the GeoGPT4V dataset significantly improves the geometry performance of various models on the MathVista and MathVision benchmarks. The code is available at https://anonymous.4open.science/r/GeoGPT4V-08B2.

</details>

---

## 69. Bridging Modalities: Enhancing Cross-Modality Hate Speech Detection with Few-Shot In-Context Learning

- [ ] Bridging Modalities: Enhancing Cross-Modality Hate Speech Detection with Few-Shot In-Context Learning | https://aclanthology.org/2024.emnlp-main.445/

- **Link**: https://aclanthology.org/2024.emnlp-main.445/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The widespread presence of hate speech on the internet, including formats such as text-based tweets and multimodal memes, poses a significant challenge to digital platform safety. Recent research has developed detection models tailored to specific modalities; however, there is a notable gap in transferring detection capabilities across different formats. This study conducts extensive experiments using few-shot in-context learning with large language models to explore the transferability of hate speech detection between modalities. Our findings demonstrate that text-based hate speech examples can significantly enhance the classification accuracy of vision-language hate speech. Moreover, text-based demonstrations outperform vision-language demonstrations in few-shot learning settings. These results highlight the effectiveness of cross-modality knowledge transfer and offer valuable insights for improving hate speech detection systems.

</details>

---

## 70. MIND: Multimodal Shopping Intention Distillation from Large Vision-language Models forE-commerce Purchase Understanding

- [ ] MIND: Multimodal Shopping Intention Distillation from Large Vision-language Models forE-commerce Purchase Understanding | https://aclanthology.org/2024.emnlp-main.446/

- **Link**: https://aclanthology.org/2024.emnlp-main.446/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Improving user experience and providing personalized search results in E-commerce platforms heavily rely on understanding purchase intention. However, existing methods for acquiring large-scale intentions bank on distilling large language models with human annotation for verification. Such an approach tends to generate product-centric intentions, overlook valuable visual information from product images, and incurs high costs for scalability. To address these issues, we introduce MIND, a multimodal framework that allows Large Vision-Language Models (LVLMs) to infer purchase intentions from multimodal product metadata and prioritize human-centric ones. Using Amazon Review data, we apply MIND and create a multimodal intention knowledge base, which contains 1,264,441 intentions derived from 126,142 co-buy shopping records across 107,215 products. Extensive human evaluations demonstrate the high plausibility and typicality of our obtained intentions and validate the effectiveness of our distillation framework and filtering mechanism. Further experiments reveal the positive downstream benefits that MIND brings to intention comprehension tasks and highlight the importance of multimodal generation and role-aware filtering. Additionally, MIND shows robustness to different prompts and superior generation quality compared to previous methods.

</details>

---

## 71. Efficient Vision-Language pre-training via domain-specific learning for human activities

- [ ] Efficient Vision-Language pre-training via domain-specific learning for human activities | https://aclanthology.org/2024.emnlp-main.454/

- **Link**: https://aclanthology.org/2024.emnlp-main.454/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Current Vision-Language (VL) models owe their success to large-scale pre-training on web-collected data, which in turn requires high-capacity architectures and large compute resources for training. We posit that when the downstream tasks are known in advance, which is in practice common, the pretraining process can be aligned to the downstream domain, leading to more efficient and accurate models, while shortening the pretraining step. To this end, we introduce a domain-aligned pretraining strategy that, without additional data collection, improves the accuracy on a domain of interest, herein, that of human activities, while largely preserving the generalist knowledge. At the core of our approach stands a new LLM-based method that, provided with a simple set of concept seeds, produces a concept hierarchy with high coverage of the target domain.The concept hierarchy is used to filter a large-scale web-crawled dataset and, then, enhance the resulting instances with targeted synthetic labels. We study in depth how to train such approaches and their resulting behavior. We further show generalization to video-based data by introducing a fast adaptation approach for transitioning from a static (image) model to a dynamic one (i.e. with temporal modeling). On the domain of interest, our approach significantly outperforms models trained on up to60×more samples and between10-100×shorter training schedules for image retrieval, video retrieval and action recognition. Code will be released.

</details>

---

## 72. mDPO: Conditional Preference Optimization for Multimodal Large Language Models

- [ ] mDPO: Conditional Preference Optimization for Multimodal Large Language Models | https://aclanthology.org/2024.emnlp-main.460/

- **Link**: https://aclanthology.org/2024.emnlp-main.460/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Direct preference optimization (DPO) has shown to be an effective method for large language model (LLM) alignment. Recent works have attempted to apply DPO to multimodal scenarios but have found it challenging to achieve consistent improvement. Through a comparative experiment, we identify the unconditional preference problem in multimodal preference optimization, where the model overlooks the image condition. To address this problem, we propose mDPO, a multimodal DPO objective that prevents the over-prioritization of language-only preferences by also optimizing image preference. Moreover, we introduce a reward anchor that forces the reward to be positive for chosen responses, thereby avoiding the decrease in their likelihood—an intrinsic problem of relative preference optimization. Experiments on two multimodal LLMs of different sizes and three widely used benchmarks demonstrate that mDPO effectively addresses the unconditional preference problem in multimodal preference optimization and significantly improves model performance, particularly in reducing hallucination.

</details>

---

## 73. Pelican: Correcting Hallucination in Vision-LLMs via Claim Decomposition and Program of Thought Verification

- [ ] Pelican: Correcting Hallucination in Vision-LLMs via Claim Decomposition and Program of Thought Verification | https://aclanthology.org/2024.emnlp-main.470/

- **Link**: https://aclanthology.org/2024.emnlp-main.470/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Visual Language Models (LVLMs) struggle with hallucinations in visual instruction following task(s). These issues hinder their trustworthiness and real-world applicability. We propose Pelican – a novel framework designed to detect and mitigate hallucinations through claim verification. Pelican first decomposes the visual claim into a chain of sub-claims based on first-order predicates. These sub-claims consists of (predicate, question) pairs and can be conceptualized as nodes of a computational graph. We then use use Program-of-Thought prompting to generate Python code for answering these questions through flexible composition of external tools. Pelican improves over prior work by introducing (1) intermediate variables for precise grounding of object instances, and (2) shared computation for answering the sub-question to enable adaptive corrections and inconsistency identification. We finally use reasoning abilities of LLM to verify the correctness of the the claim by considering the consistency and confidence of the (question, answer) pairs from each sub-claim. Our experiments demonstrate consistent performance improvements over various baseline LVLMs and existing hallucination mitigation approaches across several benchmarks.

</details>

---

## 74. Read Anywhere Pointed: Layout-awareGUIScreen Reading with Tree-of-Lens Grounding

- [ ] Read Anywhere Pointed: Layout-awareGUIScreen Reading with Tree-of-Lens Grounding | https://aclanthology.org/2024.emnlp-main.533/

- **Link**: https://aclanthology.org/2024.emnlp-main.533/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interfaces (GUIs) are central to our interaction with digital devices and growing efforts have been made to build models for various GUI understanding tasks. However, these efforts largely overlook an important GUI-referring task: screen reading based on user-indicated points, which we name the Screen Point-and-Read (ScreenPR) task. Currently, this task is predominantly handled by rigid accessible screen reading tools, in great need of new models driven by advancements in Multimodal Large Language Models (MLLMs). In this paper, we propose a Tree-of-Lens (ToL) agent, utilizing a novel ToL grounding mechanism, to address the ScreenPR task. Based on the input point coordinate and the corresponding GUI screenshot, our ToL agent constructs a Hierarchical Layout Tree. Based on the tree, our ToL agent not only comprehends the content of the indicated area but also articulates the layout and spatial relationships between elements. Such layout information is crucial for accurately interpreting information on the screen, distinguishing our ToL agent from other screen reading tools. We also thoroughly evaluate the ToL agent against other baselines on a newly proposed ScreenPR benchmark, which includes GUIs from mobile, web, and operating systems. Last but not least, we test the ToL agent on mobile GUI navigation tasks, demonstrating its utility in identifying incorrect actions along the path of agent execution trajectories. Code and data: https://screen-point-and-read.github.io.

</details>

---

## 75. IfCLIPCould Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions

- [ ] IfCLIPCould Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions | https://aclanthology.org/2024.emnlp-main.547/

- **Link**: https://aclanthology.org/2024.emnlp-main.547/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent works often assume that Vision-Language Model (VLM) representations are based on visual attributes like shape. However, it is unclear to what extent VLMs prioritize this information to represent concepts. We propose Extract and Explore (EX2), a novel approach to characterize textual features that are important for VLMs. EX2 uses reinforcement learning to align a large language model with VLM preferences and generates descriptions that incorporate features that are important for the VLM. Then, we inspect the descriptions to identify features that contribute to VLM representations. Using EX2, we find that spurious descriptions have a major role in VLM representations despite providing no helpful information, e.g., Click to enlarge photo of CONCEPT. More importantly, among informative descriptions, VLMs rely significantly on non-visual attributes like habitat (e.g., North America) to represent visual concepts. Also, our analysis reveals that different VLMs prioritize different attributes in their representations. Overall, we show that VLMs do not simply match images to scene descriptions and that non-visual or even spurious descriptions significantly influence their representations.

</details>

---

## 76. Efficient Temporal Extrapolation of Multimodal Large Language Models with Temporal Grounding Bridge

- [ ] Efficient Temporal Extrapolation of Multimodal Large Language Models with Temporal Grounding Bridge | https://aclanthology.org/2024.emnlp-main.556/

- **Link**: https://aclanthology.org/2024.emnlp-main.556/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite progress in multimodal large language models (MLLMs), the challenge of interpreting long-form videos in response to linguistic queries persists, largely due to the inefficiency in temporal grounding and limited pre-trained context window size. In this work, we introduce Temporal Grounding Bridge (TGB), a novel framework that bootstraps MLLMs with advanced temporal grounding capabilities and broadens their contextual scope. Our framework significantly enhances the temporal capabilities of current MLLMs through three key innovations: an efficient multi-span temporal grounding algorithm applied to low-dimension temporal features projected from flow; a multimodal length extrapolation training paradigm that utilizes low-dimension temporal features to extend the training context window size; and a bootstrapping framework that bridges our model with pluggable MLLMs without requiring annotation. We validate TGB across seven video benchmarks and demonstrate substantial performance improvements compared with prior MLLMs. Notably, our model, initially trained on sequences of four frames, effectively handles sequences up to 16 longer without sacrificing performance, highlighting its scalability and effectiveness in real-world applications. Our code is publicly available.

</details>

---

## 77. InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance

- [ ] InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance | https://aclanthology.org/2024.emnlp-main.585/

- **Link**: https://aclanthology.org/2024.emnlp-main.585/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As large language models (LLMs) rapidly evolve, they are increasingly being customized through fine-tuning to suit the specific needs of various applications. A critical aspect of this advancement is the alignment process, which ensures that these models perform tasks in ways that align with human values and expectations. Current alignment methods, such as direct preference optimization (DPO) and reinforcement learning from human feedback (RLHF), focus primarily on alignment during training phase. However, these methods often involve complex and resource-intensive training processes, posing significant challenge for their implementation. Therefore, we proposeInferAligner, a simple yet effective method for harmlessness alignment during inference phase. InferAligner decouples harmlessness from helpfulness. During the training phase, it focuses solely on enhancing the target model’s capabilities on downstream tasks. In the inference phase, it utilizes safety steering vectors extracted from the aligned model to guide the target model towards harmlessness alignment. Experimental results show that our method can be very effectively applied to domain-specific models in finance, medicine, and mathematics, as well as to multimodal large language models (MLLMs) such as LLaVA. It significantly diminishes the attack success rate (ASR) of both harmful instructions and jailbreak instructions, while maintaining almost unchanged performance in downstream tasks.

</details>

---

## 78. ImageInWords: Unlocking Hyper-Detailed Image Descriptions

- [ ] ImageInWords: Unlocking Hyper-Detailed Image Descriptions | https://aclanthology.org/2024.emnlp-main.6/

- **Link**: https://aclanthology.org/2024.emnlp-main.6/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite the longstanding adage ”an image is worth a thousand words,” generating accurate hyper-detailed image descriptions remains unsolved. Trained on short web-scraped image-text, vision-language models often generate incomplete descriptions with visual inconsistencies. We address this via a novel data-centric approach with ImageInWords (IIW), a carefully designed human-in-the-loop framework for curating hyper-detailed image descriptions. Human evaluations on IIW data show major gains compared to recent datasets (+66%) and GPT-4V (+48%) across comprehensiveness, specificity, hallucinations, and more. We also show that fine-tuning with IIW data improves these metrics by +31% against models trained with prior work, even with only 9k samples. Lastly, we evaluate IIW models with text-to-image generation and vision-language reasoning tasks. Our generated descriptions result in the highest fidelity images, and boost compositional reasoning by up to 6% on ARO, SVO-Probes, and Winoground datasets. We release the IIW-Eval benchmark with human judgement labels, object and image-level annotations from our framework, and existing image caption datasets enriched via IIW-model.

</details>

---

## 79. Kiss up, Kick down: Exploring Behavioral Changes in Multi-modal Large Language Models with Assigned Visual Personas

- [ ] Kiss up, Kick down: Exploring Behavioral Changes in Multi-modal Large Language Models with Assigned Visual Personas | https://aclanthology.org/2024.emnlp-main.609/

- **Link**: https://aclanthology.org/2024.emnlp-main.609/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study is the first to explore whether multi-modal large language models (LLMs) can align their behaviors with visual personas, addressing a significant gap in the literature that predominantly focuses on text-based personas. We developed a novel dataset of 5K fictional avatar images for assignment as visual personas to LLMs, and analyzed their negotiation behaviors based on the visual traits depicted in these images, with a particular focus on aggressiveness. The results indicate that LLMs assess the aggressiveness of images in a manner similar to humans and output more aggressive negotiation behaviors when prompted with an aggressive visual persona. Interestingly, the LLM exhibited more aggressive negotiation behaviors when the opponent’s image appeared less aggressive than their own, and less aggressive behaviors when the opponent’s image appeared more aggressive.

</details>

---

## 80. Large Language Models Know What is Key Visual Entity: AnLLM-assisted Multimodal Retrieval forVQA

- [ ] Large Language Models Know What is Key Visual Entity: AnLLM-assisted Multimodal Retrieval forVQA | https://aclanthology.org/2024.emnlp-main.613/

- **Link**: https://aclanthology.org/2024.emnlp-main.613/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Visual question answering (VQA) tasks, often performed by visual language model (VLM), face challenges with long-tail knowledge. Recent retrieval-augmented VQA (RA-VQA) systems address this by retrieving and integrating external knowledge sources. However, these systems still suffer from redundant visual information irrelevant to the question during retrieval. To address these issues, in this paper, we propose LLM-RA, a novel method leveraging the reasoning capability of a large language model (LLM) to identify key visual entities, thus minimizing the impact of irrelevant information in the query of retriever. Furthermore, key visual entities are independently encoded for multimodal joint retrieval, preventing cross-entity interference. Experimental results demonstrate that our method outperforms other strong RA-VQA systems. In two knowledge-intensive VQA benchmarks, our method achieves the new state-of-the-art performance among those with similar scale of parameters and even performs comparably to models with 1-2 orders larger parameters.

</details>

---

## 81. RULE: Reliable MultimodalRAGfor Factuality in Medical Vision Language Models

- [ ] RULE: Reliable MultimodalRAGfor Factuality in Medical Vision Language Models | https://aclanthology.org/2024.emnlp-main.62/

- **Link**: https://aclanthology.org/2024.emnlp-main.62/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent emergence of Medical Large Vision Language Models (Med-LVLMs) has enhanced medical diagnosis. However, current Med-LVLMs frequently encounter factual issues, often generating responses that do not align with established medical facts. Retrieval-Augmented Generation (RAG), which utilizes external knowledge, can improve the factual accuracy of these models but introduces two major challenges. First, limited retrieved contexts might not cover all necessary information, while excessive retrieval can introduce irrelevant and inaccurate references, interfering with the model’s generation. Second, in cases where the model originally responds correctly, applying RAG can lead to an over-reliance on retrieved contexts, resulting in incorrect answers. To address these issues, we propose RULE, which consists of two components. First, we introduce a provably effective strategy for controlling factuality risk through the calibrated selection of the number of retrieved contexts. Second, based on samples where over-reliance on retrieved contexts led to errors, we curate a preference dataset to fine-tune the model, balancing its dependence on inherent knowledge and retrieved contexts for generation. We demonstrate the effectiveness of RAFE on three medical VQA datasets, achieving an average improvement of 20.8% in factual accuracy.

</details>

---

## 82. TroL: Traversal of Layers for Large Language and Vision Models

- [ ] TroL: Traversal of Layers for Large Language and Vision Models | https://aclanthology.org/2024.emnlp-main.633/

- **Link**: https://aclanthology.org/2024.emnlp-main.633/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language and vision models (LLVMs) have been driven by the generalization power of large language models (LLMs) and the advent of visual instruction tuning. Along with scaling them up directly, these models enable LLVMs to showcase powerful vision language (VL) performances by covering diverse tasks via natural language instructions. However, existing open-source LLVMs that perform comparably to closed-source LLVMs such as GPT-4V are often considered too large (e.g., 26B, 34B, and 110B parameters), having a larger number of layers. These large models demand costly, high-end resources for both training and inference. To address this issue, we present a new efficient LLVM family with 1.8B, 3.8B, and 7B LLM model sizes, Traversal of Layers (TroL), which enables the reuse of layers in a token-wise manner. This layer traversing technique simulates the effect of looking back and retracing the answering stream while increasing the number of forward propagation layers without physically adding more layers. We demonstrate that TroL employs a simple layer traversing approach yet efficiently outperforms the open-source LLVMs with larger model sizes and rivals the performances of the closed-source LLVMs with substantial sizes.

</details>

---

## 83. Repairs in a Block World: A New Benchmark for Handling User Corrections with Multi-Modal Language Models

- [ ] Repairs in a Block World: A New Benchmark for Handling User Corrections with Multi-Modal Language Models | https://aclanthology.org/2024.emnlp-main.643/

- **Link**: https://aclanthology.org/2024.emnlp-main.643/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In dialogue, the addressee may initially misunderstand the speaker and respond erroneously, often prompting the speaker to correct the misunderstanding in the next turn with a Third Position Repair (TPR). The ability to process and respond appropriately to such repair sequences is thus crucial in conversational AI systems. In this paper, we first collect, analyse, and publicly release BlockWorld-Repairs: a dataset of multi-modal TPR sequences in an instruction-following manipulation task that is, by design, rife with referential ambiguity. We employ this dataset to evaluate several state-of-the-art Vision and Language Models (VLM) across multiple settings, focusing on their capability to process and accurately respond to TPRs and thus recover from miscommunication. We find that, compared to humans, all models significantly underperform in this task. We then show that VLMs can benefit from specialised losses targeting relevant tokens during fine-tuning, achieving better performance and generalising better to new scenarios. Our results suggest that these models are not yet ready to be deployed in multi-modal collaborative settings where repairs are common, and highlight the need to design training regimes and objectives that facilitate learning from interaction. Our code and data are available at www.github.com/JChiyah/blockworld-repairs

</details>

---

## 84. EFUF: Efficient Fine-Grained Unlearning Framework for Mitigating Hallucinations in Multimodal Large Language Models

- [ ] EFUF: Efficient Fine-Grained Unlearning Framework for Mitigating Hallucinations in Multimodal Large Language Models | https://aclanthology.org/2024.emnlp-main.67/

- **Link**: https://aclanthology.org/2024.emnlp-main.67/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have attracted increasing attention in the past few years, but they may still generate descriptions that include objects not present in the corresponding images, a phenomenon known as object hallucination. To eliminate hallucinations, existing methods manually annotate paired responses with and without hallucinations, and then employ various alignment algorithms to improve the alignment capability between images and text. However, they not only demand considerable computation resources during the finetuning stage but also require expensive human annotation to construct paired data needed by the alignment algorithms. To address these issues, we propose an efficient fine-grained unlearning framework (EFUF), which performs gradient ascent utilizing three tailored losses to eliminate hallucinations without paired data. Extensive experiments show that our method consistently reduces hallucinations while preserving the generation quality with modest computational overhead. Our code and datasets will be publicly available.

</details>

---

## 85. Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress?

- [ ] Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? | https://aclanthology.org/2024.emnlp-main.677/

- **Link**: https://aclanthology.org/2024.emnlp-main.677/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Several recent works seek to develop foundation models specifically for medical applications, adapting general-purpose large language models (LLMs) and vision-language models (VLMs) via continued pretraining on publicly available biomedical corpora. These works typically claim that such domain-adaptive pretraining (DAPT) improves performance on downstream medical tasks, such as answering medical licensing exam questions. In this paper, we compare seven public “medical” LLMs and two VLMs against their corresponding base models, arriving at a different conclusion: all medical VLMs and nearly all medical LLMs fail to consistently improve over their base models in the zero-/few-shot prompting regime for medical question-answering (QA) tasks. For instance, across the tasks and model pairs we consider in the 3-shot setting, medical LLMs only outperform their base models in 12.1% of cases, reach a (statistical) tie in 49.8% of cases, and are significantly worse than their base models in the remaining 38.2% of cases. Our conclusions are based on (i) comparing each medical model head-to-head, directly against the corresponding base model; (ii) optimizing the prompts for each model separately; and (iii) accounting for statistical uncertainty in comparisons. While these basic practices are not consistently adopted in the literature, our ablations show that they substantially impact conclusions. Our findings suggest that state-of-the-art general-domain models may already exhibit strong medical knowledge and reasoning capabilities, and offer recommendations to strengthen the conclusions of future studies.

</details>

---

## 86. UNICORN: A Unified Causal Video-Oriented Language-Modeling Framework for Temporal Video-Language Tasks

- [ ] UNICORN: A Unified Causal Video-Oriented Language-Modeling Framework for Temporal Video-Language Tasks | https://aclanthology.org/2024.emnlp-main.722/

- **Link**: https://aclanthology.org/2024.emnlp-main.722/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The great success of large language models has encouraged the development of large multimodal models, with a focus on image-language interaction. Despite promising results in various image-language downstream tasks, it is still challenging and unclear how to extend the capabilities of these models to the more complex video domain, especially when dealing with explicit temporal signals. To address the problem in existing large multimodal models, in this paper we adopt visual instruction tuning to build a unified causal video-oriented language modeling framework, named UNICORN. Specifically, we collect a comprehensive dataset under the instruction-following format, and instruction-tune the model accordingly. Experimental results demonstrate that without customized training objectives and intensive pre-training, UNICORN can achieve comparable or better performance on established temporal video-language tasks including moment retrieval, video paragraph captioning and dense video captioning. Moreover, the instruction-tuned model can be used to automatically annotate internet videos with temporally-aligned captions. Compared to commonly used ASR captions, we show that training on our generated captions improves the performance of video-language models on both zero-shot and fine-tuning settings. Source code can be found at https://github.com/xyh97/UNICORN.

</details>

---

## 87. Shaking UpVLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling

- [ ] Shaking UpVLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling | https://aclanthology.org/2024.emnlp-main.793/

- **Link**: https://aclanthology.org/2024.emnlp-main.793/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study explores replacing Transformers in Visual Language Models (VLMs) with Mamba, a recent structured state space model (SSM) that demonstrates promising performance in sequence modeling. We test models up to 3B parameters under controlled conditions, showing that Mamba-based VLMs outperforms Transformers-based VLMs in captioning, question answering, and reading comprehension. However, we find that Transformers achieve greater performance in visual grounding and the performance gap widens with scale. We explore two hypotheses to explain this phenomenon: 1) the effect of task-agnostic visual encoding on the updates of the hidden states, and 2) the difficulty in performing visual grounding from the perspective of in-context multimodal retrieval. Our results indicate that a task-aware encoding yields minimal performance gains on grounding, however, Transformers significantly outperform Mamba at in-context multimodal retrieval. Overall, Mamba shows promising performance on tasks where the correct output relies on a summary of the image but struggles when retrieval of explicit information from the context is required.

</details>

---

## 88. Vision-Language Model Fine-Tuning via Simple Parameter-Efficient Modification

- [ ] Vision-Language Model Fine-Tuning via Simple Parameter-Efficient Modification | https://aclanthology.org/2024.emnlp-main.797/

- **Link**: https://aclanthology.org/2024.emnlp-main.797/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in fine-tuning Vision-Language Models (VLMs) have witnessed the success of prompt tuning and adapter tuning, while the classic model fine-tuning on inherent parameters seems to be overlooked. It is believed that fine-tuning the parameters of VLMs with few-shot samples corrupts the pre-trained knowledge since fine-tuning the CLIP model even degrades performance. In this paper, we revisit this viewpoint, and propose a new perspective: fine-tuning the specific parameters instead of all will uncover the power of classic model fine-tuning on VLMs. Through our meticulous study, we propose ClipFit, a simple yet effective method to fine-tune CLIP without introducing any overhead of extra parameters. We demonstrate that by only fine-tuning the specific bias terms and normalization layers, ClipFit can improve the performance of zero-shot CLIP by 7.27% average harmonic mean accuracy. Lastly, to understand how fine-tuning in CLIPFit affects the pre-trained models, we conducted extensive experimental analyses w.r.t. changes in internal parameters and representations. We found that low-level text bias layers and the first layer normalization layer change much more than other layers. The code will be released.

</details>

---

## 89. Interpretable Composition Attribution Enhancement for Visio-linguistic Compositional Understanding

- [ ] Interpretable Composition Attribution Enhancement for Visio-linguistic Compositional Understanding | https://aclanthology.org/2024.emnlp-main.810/

- **Link**: https://aclanthology.org/2024.emnlp-main.810/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Contrastively trained vision-language models such as CLIP have achieved remarkable progress in vision and language representation learning. Despite the promising progress, their proficiency in compositional reasoning over attributes and relations (e.g., distinguishing between “the car is underneath the person” and “the person is underneath the car”) remains notably inadequate. We investigate the cause for this deficient behavior is the composition attribution issue, where the attribution scores (e.g., attention scores or GradCAM scores) for relations (e.g., underneath) or attributes (e.g., red) in the text are substantially lower than those for object terms. In this work, we show such issue is mitigated via a novel framework called CAE (Composition Attribution Enhancement). This generic framework incorporates various interpretable attribution methods to encourage the model to pay greater attention to composition words denoting relationships and attributes within the text. Detailed analysis shows that our approach enables the models to adjust and rectify the attribution of the texts. Extensive experiments across seven benchmarks reveal that our framework significantly enhances the ability to discern intricate details and construct more sophisticated interpretations of combined visual and linguistic elements.

</details>

---

## 90. ActPlan-1K: Benchmarking the Procedural Planning Ability of Visual Language Models in Household Activities

- [ ] ActPlan-1K: Benchmarking the Procedural Planning Ability of Visual Language Models in Household Activities | https://aclanthology.org/2024.emnlp-main.833/

- **Link**: https://aclanthology.org/2024.emnlp-main.833/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models(LLMs) have been adopted to process textual task description and accomplish procedural planning in embodied AI tasks because of their powerful reasoning ability. However, there is still lack of study on how vision language models(VLMs) behave when multi-modal task inputs are considered. Counterfactual planning that evaluates the model’s reasoning ability over alternative task situations are also under exploited. In order to evaluate the planning ability of both multi-modal and counterfactual aspects, we propose ActPlan-1K. ActPlan-1K is a multi-modal planning benchmark constructed based on ChatGPT and household activity simulator iGibson2. The benchmark consists of 153 activities and 1,187 instances. Each instance describing one activity has a natural language task description and multiple environment images from the simulator. The gold plan of each instance is action sequences over the objects in provided scenes. Both the correctness and commonsense satisfaction are evaluated on typical VLMs. It turns out that current VLMs are still struggling at generating human-level procedural plans for both normal activities and counterfactual activities. We further provide automatic evaluation metrics by finetuning over BLEURT model to facilitate future research on our benchmark.

</details>

---

## 91. FineCops-Ref: A new Dataset and Task for Fine-Grained Compositional Referring Expression Comprehension

- [ ] FineCops-Ref: A new Dataset and Task for Fine-Grained Compositional Referring Expression Comprehension | https://aclanthology.org/2024.emnlp-main.864/

- **Link**: https://aclanthology.org/2024.emnlp-main.864/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Referring Expression Comprehension (REC) is a crucial cross-modal task that objectively evaluates the capabilities of language understanding, image comprehension, and language-to-image grounding. Consequently, it serves as an ideal testing ground for Multi-modal Large Language Models (MLLMs). In pursuit of this goal, we have established a new REC dataset characterized by two key features: Firstly, it is designed with controllable varying levels of difficulty, necessitating multi-level fine-grained reasoning across object categories, attributes, and multi-hop relationships. Secondly, it includes negative text and images created through fine-grained editing and generation based on existing data, thereby testing the model’s ability to correctly reject scenarios where the target object is not visible in the image—an essential aspect often overlooked in existing datasets and approaches. Utilizing this high-quality dataset, we conducted comprehensive evaluations of both state-of-the-art specialist models and MLLMs. Our findings indicate that there remains a significant gap in achieving satisfactory grounding performance. We anticipate that our dataset will inspire new approaches to enhance visual reasoning and develop more advanced cross-modal interaction strategies, ultimately unlocking the full potential of MLLMs.

</details>

---

## 92. ERVQA: A Dataset to Benchmark the Readiness of Large Vision Language Models in Hospital Environments

- [ ] ERVQA: A Dataset to Benchmark the Readiness of Large Vision Language Models in Hospital Environments | https://aclanthology.org/2024.emnlp-main.873/

- **Link**: https://aclanthology.org/2024.emnlp-main.873/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The global shortage of healthcare workers has demanded the development of smart healthcare assistants, which can help monitor and alert healthcare workers when necessary. We examine the healthcare knowledge of existing Large Vision Language Models (LVLMs) via the Visual Question Answering (VQA) task in hospital settings through expert annotated open-ended questions. We introduce the Emergency Room Visual Question Answering (ERVQA) dataset, consisting of <image, question, answer> triplets covering diverse emergency room scenarios, a seminal benchmark for LVLMs. By developing a detailed error taxonomy and analyzing answer trends, we reveal the nuanced nature of the task. We benchmark state-of-the-art open-source and closed LVLMs using traditional and adapted VQA metrics: Entailment Score and CLIPScore Confidence. Analyzing errors across models, we infer trends based on properties like decoder type, model size, and in-context examples. Our findings suggest the ERVQA dataset presents a highly complex task, highlighting the need for specialized, domain-specific solutions.

</details>

---

## 93. Images Speak Louder than Words: Understanding and Mitigating Bias in Vision-Language Model from a Causal Mediation Perspective

- [ ] Images Speak Louder than Words: Understanding and Mitigating Bias in Vision-Language Model from a Causal Mediation Perspective | https://aclanthology.org/2024.emnlp-main.878/

- **Link**: https://aclanthology.org/2024.emnlp-main.878/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) pre-trained on extensive datasets can inadvertently learn biases by correlating gender information with specific objects or scenarios. Current methods, which focus on modifying inputs and monitoring changes in the model’s output probability scores, often struggle to comprehensively understand bias from the perspective of model components. We propose a framework that incorporates causal mediation analysis to measure and map the pathways of bias generation and propagation within VLMs. Our framework is applicable to a wide range of vision-language and multimodal tasks. In this work, we apply it to the object detection task and implement it on the GLIP model. This approach allows us to identify the direct effects of interventions on model bias and the indirect effects of interventions on bias mediated through different model components. Our results show that image features are the primary contributors to bias, with significantly higher impacts than text features, specifically accounting for 32.57% and 12.63% of the bias in the MSCOCO and PASCAL-SENTENCE datasets, respectively. Notably, the image encoder’s contribution surpasses that of the text encoder and the deep fusion encoder. Further experimentation confirms that contributions from both language and vision modalities are aligned and non-conflicting. Consequently, focusing on blurring gender representations within the image encoder which contributes most to the model bias, reduces bias efficiently by 22.03% and 9.04% in the MSCOCO and PASCAL-SENTENCE datasets, respectively, with minimal performance loss or increased computational demands.

</details>

---

## 94. UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation

- [ ] UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation | https://aclanthology.org/2024.emnlp-main.89/

- **Link**: https://aclanthology.org/2024.emnlp-main.89/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The fashion domain encompasses a variety of real-world multimodal tasks, including multimodal retrieval and multimodal generation. The rapid advancements in artificial intelligence generated content, particularly in technologies like large language models for text generation and diffusion models for visual generation, have sparked widespread research interest in applying these multimodal models in the fashion domain. However, tasks that use embeddings, such as image-to-text or text-to-image retrieval, have been largely ignored from this perspective due to the diverse nature of the multimodal fashion domain. And current research on multi-task single models lack focus on image generation. In this work, we present UniFashion, a unified framework that simultaneously tackles the challenges of multimodal generation and retrieval tasks within the fashion domain, integrating image generation with retrieval tasks and text generation tasks. UniFashion unifies embedding and generative tasks by integrating a diffusion model and LLM, enabling controllable and high-fidelity generation. Our model significantly outperforms previous single-task state-of-the-art models across diverse fashion tasks, and can be readily adapted to manage complex vision-language tasks. This work demonstrates the potential learning synergy between multimodal generation and retrieval, offering a promising direction for future research in the fashion domain.

</details>

---

## 95. MLLM-Protector: EnsuringMLLM’s Safety without Hurting Performance

- [ ] MLLM-Protector: EnsuringMLLM’s Safety without Hurting Performance | https://aclanthology.org/2024.emnlp-main.895/

- **Link**: https://aclanthology.org/2024.emnlp-main.895/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. This paper investigates the novel challenge of defending MLLMs against such attacks. Compared to large language models (LLMs), MLLMs include an additional image modality. We discover that images act as a “foreign language” that is not considered during safety alignment, making MLLMs more prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover all possible scenarios. This vulnerability is exacerbated by the fact that most state-of-the-art MLLMs are fine-tuned on limited image-text pairs that are much fewer than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during safety fine-tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy that solves two subtasks: 1) identifying harmful responses via a lightweight harm detector, and 2) transforming harmful responses into harmless ones via a detoxifier. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the original performance of MLLMs. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.

</details>

---

## 96. CorrSynth - A Correlated Sampling Method for Diverse Dataset Generation fromLLMs

- [ ] CorrSynth - A Correlated Sampling Method for Diverse Dataset Generation fromLLMs | https://aclanthology.org/2024.emnlp-main.899/

- **Link**: https://aclanthology.org/2024.emnlp-main.899/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have demonstrated remarkable performance in diverse tasks using zero-shot and few-shot prompting. Even though their capabilities of data synthesis have been studied well in recent years, the generated data suffers from a lack of diversity, less adherence to the prompt, and potential biases that creep into the data from the generator model. In this work, we tackle the challenge of generating datasets with high diversity, upon which a student model is trained for downstream tasks. Taking the route of decoding-time guidance-based approaches, we propose CorrSynth, which generates data that is more diverse and faithful to the input prompt using a correlated sampling strategy. Further, our method overcomes the complexity drawbacks of some other guidance-based techniques like classifier-based guidance. With extensive experiments, we show the effectiveness of our approach and substantiate our claims. In particular, we perform intrinsic evaluation to show the improvements in diversity. Our experiments show that CorrSynth improves both student metrics and intrinsic metrics upon competitive baselines across four datasets, showing the innate advantage of our method.

</details>

---

## 97. The Instinctive Bias: Spurious Images lead to Illusion inMLLMs

- [ ] The Instinctive Bias: Spurious Images lead to Illusion inMLLMs | https://aclanthology.org/2024.emnlp-main.904/

- **Link**: https://aclanthology.org/2024.emnlp-main.904/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have recently experienced remarkable progress, where the advent of multi-modal large language models (MLLMs) has endowed LLMs with visual capabilities, leading to impressive performances in various multi-modal tasks. However, those powerful MLLMs such as GPT-4V still fail spectacularly when presented with certain image and text inputs. In this paper, we identify a typical class of inputs that baffles MLLMs, which consist of images that are highly relevant but inconsistent with answers, causing MLLMs to suffer from visual illusion. To quantify the effect, we propose CorrelationQA, the first benchmark that assesses the visual illusion level given spurious images. This benchmark contains 7,308 text-image pairs across 13 categories. Based on the proposed CorrelationQA, we conduct a thorough analysis on 9 mainstream MLLMs, illustrating that they universally suffer from this instinctive bias to varying degrees. We hope that our curated benchmark and evaluation results aid in better assessments of the MLLMs’ robustness in the presence of misleading images. The code and datasets are available at https://github.com/MasaiahHan/CorrelationQA.

</details>

---

## 98. MAR: Matching-Augmented Reasoning for Enhancing Visual-based Entity Question Answering

- [ ] MAR: Matching-Augmented Reasoning for Enhancing Visual-based Entity Question Answering | https://aclanthology.org/2024.emnlp-main.91/

- **Link**: https://aclanthology.org/2024.emnlp-main.91/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A multimodal large language model MLLMs may struggle with answering visual-based (personal) entity questions (VEQA), such as ”who is A?” or ”who is A that B is talking to?” for various reasons, e.g., the absence of the name of A in the caption or the inability of MLLMs to recognize A, particularly for less common entities. Furthermore, even if the MLLMs can identify A, it may refrain from answering due to privacy concerns. In this paper, we introduce a novel method called Matching-Augmented Reasoning (MAR) to enhance VEQA. Given a collection of visual objects with captions, MAR preprocesses each object individually, identifying faces, names, and their alignments within the object. It encodes this information and stores their vector representations in vector databases. When handling VEQA, MAR retrieves matching faces and names and organizes these entities into a matching graph. MAR then derives the answer to the query by reasoning over this matching graph. Extensive experiments show that MAR significantly improves VEQA compared with the state-of-the-art methods using MLLMs.

</details>

---

## 99. ***YesBut***: A High-Quality Annotated Multimodal Dataset for evaluating Satire Comprehension capability of Vision-Language Models

- [ ] ***YesBut***: A High-Quality Annotated Multimodal Dataset for evaluating Satire Comprehension capability of Vision-Language Models | https://aclanthology.org/2024.emnlp-main.937/

- **Link**: https://aclanthology.org/2024.emnlp-main.937/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Understanding satire and humor is a challenging task for even current Vision-Language models. In this paper, we propose the challenging tasks of Satirical Image Detection (detecting whether an image is satirical), Understanding (generating the reason behind the image being satirical), and Completion (given one half of the image, selecting the other half from 2 given options, such that the complete image is satirical) and release a high-quality dataset ***YesBut***, consisting of 2547 images, 1084 satirical and 1463 non-satirical, containing different artistic styles, to evaluate those tasks. Each satirical image in the dataset depicts a normal scenario, along with a conflicting scenario which is funny or ironic. Despite the success of current Vision-Language Models on multimodal tasks such as Visual QA and Image Captioning, our benchmarking experiments show that such models perform poorly on the proposed tasks on the ***YesBut*** Dataset in Zero-Shot Settings w.r.t both automated as well as human evaluation. Additionally, we release a dataset of 119 real, satirical photographs for further research.

</details>

---

## 100. On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning

- [ ] On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning | https://aclanthology.org/2024.emnlp-main.944/

- **Link**: https://aclanthology.org/2024.emnlp-main.944/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in language and vision assistants have showcased impressive capabilities but suffer from a lack of transparency, limiting broader research and reproducibility. While open-source models handle general image tasks effectively, they face challenges with the high computational demands of complex visually-situated text understanding. Such tasks often require increased token inputs and large vision modules to harness high-resolution information. Striking a balance between model size and data importance remains an open question. This study aims to redefine the design of vision-language models by identifying key components and creating efficient models with constrained inference costs. By strategically formulating datasets, optimizing vision modules, and enhancing supervision techniques, we achieve significant improvements in inference throughput while maintaining high performance. Extensive experiments across models ranging from 160M to 13B parameters offer insights into model optimization.We will fully open-source our codebase, models, and datasets at https://github.com/naver-ai/elva.

</details>

---

## 101. Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models

- [ ] Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models | https://aclanthology.org/2024.emnlp-main.947/

- **Link**: https://aclanthology.org/2024.emnlp-main.947/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite recent advances demonstrating vision- language models’ (VLMs) abilities to describe complex relationships among objects in images using natural language, their capability to quantitatively reason about object sizes and distances remains underexplored. In this work, we introduce a manually annotated benchmark of 241 questions across five categories specifically designed for quantitative spatial reasoning, and systematically investigate the performance of SoTA VLMs on this task. Our analysis reveals that questions involving reasoning about distances between objects are particularly challenging for SoTA VLMs; however, some VLMs perform significantly better at this task than others, with an almost 40 points gap between the two best performing models. We also make the surprising observation that the success rate of the top-performing VLM increases by 19 points when a reasoning path using a reference object emerges naturally in the response. Inspired by this observation, we develop a zero-shot prompting technique, SpatialPrompt, that encourages VLMs to answer quantitative spatial questions using references objects as visual cues. Specifically, we demonstrate that instruct- ing VLMs to use reference objects in their reasoning paths significantly improves their quantitative spatial reasoning performance, bypassing the need for external data, architectural modifications, or fine-tuning. Remarkably, by solely using SpatialPrompt, Gemini 1.5 Pro, GPT-4V, and GPT-4o improve by 56.2, 28.5, and 6.7 points on average in Q-Spatial Bench without the need for more data, model architectural modifications, or fine-tuning.

</details>

---

## 102. Granular Privacy Control for Geolocation with Vision Language Models

- [ ] Granular Privacy Control for Geolocation with Vision Language Models | https://aclanthology.org/2024.emnlp-main.957/

- **Link**: https://aclanthology.org/2024.emnlp-main.957/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) are rapidly advancing in their capability to answer information-seeking questions. As these models are widely deployed in consumer applications, they could lead to new privacy risks due to emergent abilities to identify people in photos, geolocate images, etc. As we demonstrate, somewhat surprisingly, current open-source and proprietary VLMs are very capable image geolocators, making widespread geolocation with VLMs an immediate privacy risk, rather than merely a theoretical future concern. As a first step to address this challenge, we develop a new benchmark, GPTGeoChat, to test the capability of VLMs to moderate geolocation dialogues with users. We collect a set of 1,000 image geolocation conversations between in-house annotators and GPT-4v, which are annotated with the granularity of location information revealed at each turn. Using this new dataset we evaluate the ability of various VLMs to moderate GPT-4v geolocation conversations by determining when too much location information has been revealed. We find that custom fine-tuned models perform on par with prompted API-based models when identifying leaked location information at the country or city level, however fine-tuning on supervised data appears to be needed to accurately moderate finer granularities, such as the name of a restaurant or building.

</details>

---

## 103. Predicate Debiasing in Vision-Language Models Integration for Scene Graph Generation Enhancement

- [ ] Predicate Debiasing in Vision-Language Models Integration for Scene Graph Generation Enhancement | https://aclanthology.org/2024.emnlp-main.97/

- **Link**: https://aclanthology.org/2024.emnlp-main.97/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Scene Graph Generation (SGG) provides basic language representation of visual scenes, requiring models to grasp complex and diverse semantics between objects. This complexity and diversity in SGG leads to underrepresentation, where parts of triplet labels are rare or even unseen during training, resulting in imprecise predictions. To tackle this, we propose integrating the pretrained Vision-language Models to enhance representation. However, due to the gap between pretraining and SGG, direct inference of pretrained VLMs on SGG leads to severe bias, which stems from the imbalanced predicates distribution in the pretraining language set. To alleviate the bias, we introduce a novel LM Estimation to approximate the unattainable predicates distribution. Finally, we ensemble the debiased VLMs with SGG models to enhance the representation, where we design a certainty-aware indicator to score each sample and dynamically adjust the ensemble weights. Our training-free method effectively addresses the predicates bias in pretrained VLMs, enhances SGG’s representation, and significantly improve the performance.

</details>

---

## 104. FromLLMs toMLLMs: Exploring the Landscape of Multimodal Jailbreaking

- [ ] FromLLMs toMLLMs: Exploring the Landscape of Multimodal Jailbreaking | https://aclanthology.org/2024.emnlp-main.973/

- **Link**: https://aclanthology.org/2024.emnlp-main.973/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has exposed vulnerabilities to various adversarial attacks. This paper provides a comprehensive overview of jailbreaking research targeting both LLMs and MLLMs, highlighting recent advancements in evaluation benchmarks, attack techniques and defense strategies. Compared to the more advanced state of unimodal jailbreaking, multimodal domain remains underexplored. We summarize the limitations and potential research directions of multimodal jailbreaking, aiming to inspire future research and further enhance the robustness and security of MLLMs.

</details>

---

## 105. Can Large Language Models Enhance Predictions of Disease Progression? Investigating Through Disease Network Link Prediction

- [ ] Can Large Language Models Enhance Predictions of Disease Progression? Investigating Through Disease Network Link Prediction | https://aclanthology.org/2024.emnlp-main.980/

- **Link**: https://aclanthology.org/2024.emnlp-main.980/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have made significant strides in various tasks, yet their effectiveness in predicting disease progression remains relatively unexplored. To fill this gap, we use LLMs and employ advanced graph prompting and Retrieval-Augmented Generation (RAG) to predict disease comorbidity within disease networks. Specifically, we introduce a disease Comorbidity prediction model using LLM, named ComLLM, which leverages domain knowledge to enhance the prediction performance. Based on the comprehensive experimental results, ComLLM consistently outperforms conventional models, such as Graph Neural Networks, achieving average area under the curve (AUC) improvements of 10.70% and 6.07% over the best baseline models in two distinct disease networks. ComLLM is evaluated across multiple settings for disease progression prediction, employing various prompting strategies, including zero-shot, few-shot, Chain-of-Thought, graph prompting and RAG. Our results show that graph prompting and RAG enhance LLM performance in disease progression prediction tasks. ComLLM exhibits superior predictive capabilities and serves as a proof-of-concept for LLM-based systems in disease progression prediction, highlighting its potential for broad applications in healthcare.

</details>

---

## 106. From Descriptive Richness to Bias: Unveiling the Dark Side of Generative Image Caption Enrichment

- [ ] From Descriptive Richness to Bias: Unveiling the Dark Side of Generative Image Caption Enrichment | https://aclanthology.org/2024.emnlp-main.986/

- **Link**: https://aclanthology.org/2024.emnlp-main.986/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have enhanced the capacity of vision-language models to caption visual text. This generative approach to image caption enrichment further makes textual captions more descriptive, improving alignment with the visual context. However, while many studies focus on the benefits of generative caption enrichment (GCE), are there any negative side effects? We compare standard-format captions and recent GCE processes from the perspectives of gender bias and hallucination, showing that enriched captions suffer from increased gender bias and hallucination. Furthermore, models trained on these enriched captions amplify gender bias by an average of 30.9% and increase hallucination by 59.5%. This study serves as a caution against the trend of making captions more descriptive.

</details>

---

## 107. In-Context Compositional Generalization for Large Vision-Language Models

- [ ] In-Context Compositional Generalization for Large Vision-Language Models | https://aclanthology.org/2024.emnlp-main.996/

- **Link**: https://aclanthology.org/2024.emnlp-main.996/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent work has revealed that in-context learning for large language models exhibits compositional generalization capacity, which can be enhanced by selecting in-context demonstrations similar to test cases to provide contextual information. However, how to exhibit in-context compositional generalization (ICCG) of large vision-language models (LVLMs) is non-trival. Due to the inherent asymmetry between visual and linguistic modalities, ICCG in LVLMs faces an inevitable challenge—redundant information on the visual modality. The redundant information affects in-context learning from two aspects: (1) Similarity calculation may be dominated by redundant information, resulting in sub-optimal demonstration selection. (2) Redundant information in in-context demonstrations brings misleading contextual information to in-context learning. To alleviate these problems, we propose a demonstration selection method to achieve ICCG for LVLMs, by considering two key factors of demonstrations: content and structure, from a multimodal perspective. Specifically, we design a diversity-coverage-based matching score to select demonstrations with maximum coverage, and avoid selecting demonstrations with redundant information via their content redundancy and structural complexity. We build a GQA-ICCG dataset to simulate the ICCG setting, and conduct experiments on GQA-ICCG and the VQA v2 dataset. Experimental results demonstrate the effectiveness of our method.

</details>

---

## 108. Game on Tree: Visual Hallucination Mitigation via Coarse-to-Fine View Tree and Game Theory

- [ ] Game on Tree: Visual Hallucination Mitigation via Coarse-to-Fine View Tree and Game Theory | https://aclanthology.org/2024.emnlp-main.998/

- **Link**: https://aclanthology.org/2024.emnlp-main.998/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) may produce outputs that are unfaithful to reality, also known as visual hallucinations (VH), which hinders their application in multimodal understanding and decision-making. In this work, we introduce a novel plug-and-play train-free decoding algorithm named Game and Tree based Hallucination Mitigation (GTHM), designed for mitigating VH. GTHM is inspired by empirical observations that the fuzziness of multi-granularity view perception exacerbates VH. Based on this, GTHM leverages visual information to construct a coarse-to-fine visual view tree (CFTree) that organizes visual objects, attributes, and relationships in a hierarchical manner. Additionally, we innovatively model the optimal visual-token matching process on the CFTree as the cooperative game. Specifically, we define the Tree-based Shapley Value (TSV) for each visual view on the CFTree to assess its significant contribution to the overall visual understanding, thereby determining the optimal visual granularity. Subsequently, we utilize the TSV as guidance to implement adaptive weight contrastive decoding to achieve vision-aware decoding. Extensive experiments on four popular benchmarks confirm the effectiveness of our GTHM in alleviating VH across different LVLM families without additional training or post-processing. Our code is published at https://github.com/mengchuang123/GTHM.

</details>

---

## 109. RAGAR, Your Falsehood Radar:RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models

- [ ] RAGAR, Your Falsehood Radar:RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models | https://aclanthology.org/2024.fever-1.29/

- **Link**: https://aclanthology.org/2024.fever-1.29/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The escalating challenge of misinformation, particularly in political discourse, requires advanced fact-checking solutions; this is even clearer in the more complex scenario of multimodal claims. We tackle this issue using a multimodal large language model in conjunction with retrieval-augmented generation (RAG), and introduce two novel reasoning techniques: Chain of RAG (CoRAG) and Tree of RAG (ToRAG). They fact-check multimodal claims by extracting both textual and image content, retrieving external information, and reasoning subsequent questions to be answered based on prior evidence. We achieve a weighted F1-score of 0.85, surpassing a baseline reasoning technique by 0.14 points. Human evaluation confirms that the vast majority of our generated fact-check explanations contain all information from gold standard data.

</details>

---

## 110. Enhancing Fine-Grained Image Classifications via Cascaded Vision Language Models

- [ ] Enhancing Fine-Grained Image Classifications via Cascaded Vision Language Models | https://aclanthology.org/2024.findings-emnlp.102/

- **Link**: https://aclanthology.org/2024.findings-emnlp.102/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-grained image classification, especially in zero-/few-shot scenarios, poses a considerable challenge for vision-language models (VLMs) like CLIP, which often struggle to differentiate between semantically similar classes due to insufficient supervision for fine-grained tasks. On the other hand, Large Vision Language Models (LVLMs) have demonstrated remarkable capabilities in tasks like Visual Question Answering (VQA) but remain underexplored in the context of fine-grained image classification. This paper presents CascadeVLM, a novel framework that harnesses the complementary strengths of both CLIP-like and LVLMs VLMs to tackle these challenges. Using granular knowledge effectively in LVLMs and integrating a cascading approach, CascadeVLM dynamically allocates samples using an entropy threshold, balancing computational efficiency with classification accuracy. Experiments on multiple fine-grained datasets, particularly the Stanford Cars dataset, show that CascadeVLM outperforms existing models, achieving 92% accuracy. Our results highlight the potential of combining VLM and LVLM for robust, efficient and interpretable fine-grained image classification, offering new insights into their synergy.

</details>

---

## 111. Visual Question Decomposition on Multimodal Large Language Models

- [ ] Visual Question Decomposition on Multimodal Large Language Models | https://aclanthology.org/2024.findings-emnlp.107/

- **Link**: https://aclanthology.org/2024.findings-emnlp.107/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Question decomposition has emerged as an effective strategy for prompting Large Language Models (LLMs) to answer complex questions. However, while existing methods primarily focus on unimodal language models, the question decomposition capability of Multimodal Large Language Models (MLLMs) has yet to be explored. To this end, this paper explores visual question decomposition on MLLMs. Specifically, we introduce a systematic evaluation framework including a dataset and several evaluation criteria to assess the quality of the decomposed sub-questions, revealing that existing MLLMs struggle to produce high-quality sub-questions. To address this limitation, we propose a specific finetuning dataset, DecoVQA+, for enhancing the model’s question decomposition capability. Aiming at enabling models to perform appropriate selective decomposition, we propose an efficient finetuning pipeline. The finetuning pipeline consists of our proposed dataset and a training objective for selective decomposition. Finetuned MLLMs demonstrate significant improvements in the quality of sub-questions and the policy of selective question decomposition. Additionally, the models also achieve higher accuracy with selective decomposition on VQA benchmark datasets.

</details>

---

## 112. SnapNTell: Enhancing Entity-Centric Visual Question Answering with Retrieval Augmented MultimodalLLM

- [ ] SnapNTell: Enhancing Entity-Centric Visual Question Answering with Retrieval Augmented MultimodalLLM | https://aclanthology.org/2024.findings-emnlp.14/

- **Link**: https://aclanthology.org/2024.findings-emnlp.14/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-extended LLMs have made significant strides in Visual Question Answering (VQA). Despite these advancements, VLLMs still encounter substantial difficulties in handling queries involving long-tail entities, with a tendency to produce erroneous or hallucinated responses. In this work, we introduce a novel evaluative benchmark namedSnapNTell, specifically tailored for entity-centric VQA. This task aims to test the models’ capabilities in identifying entities and providing detailed, entity-specific knowledge. We have developed theSnapNTell Dataset, distinct from traditional VQA datasets: (1) It encompasses a wide range of categorized entities, each represented by images and explicitly named in the answers; (2) It features QA pairs that require extensive knowledge for accurate responses. The dataset is organized into 22 major categories, containing 7,568 unique entities in total. For each entity, we curated 10 illustrative images and crafted 10 knowledge-intensive QA pairs. To address this novel task, we devised a scalable, efficient, and transparent retrieval-augmented multimodal LLM. Our approach markedly outperforms existing methods on the SnapNTell dataset, achieving a 66.5% improvement in the BELURT score.

</details>

---

## 113. MM-ChatAlign: A Novel Multimodal Reasoning Framework based on Large Language Models for Entity Alignment

- [ ] MM-ChatAlign: A Novel Multimodal Reasoning Framework based on Large Language Models for Entity Alignment | https://aclanthology.org/2024.findings-emnlp.148/

- **Link**: https://aclanthology.org/2024.findings-emnlp.148/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal entity alignment (MMEA) integrates multi-source and cross-modal knowledge graphs, a crucial yet challenging task for data-centric applications.Traditional MMEA methods derive the visual embeddings of entities and combine them with other modal data for alignment by embedding similarity comparison.However, these methods are hampered by the limited comprehension of visual attributes and deficiencies in realizing and bridging the semantics of multimodal data. To address these challenges, we propose MM-ChatAlign, a novel framework that utilizes the visual reasoning abilities of MLLMs for MMEA.The framework features an embedding-based candidate collection module that adapts to various knowledge representation strategies, effectively filtering out irrelevant reasoning candidates. Additionally, a reasoning and rethinking module, powered by MLLMs, enhances alignment by efficiently utilizing multimodal information.Extensive experiments on four MMEA datasets demonstrate MM-ChatAlign’s superiority and underscore the significant potential of MLLMs in MMEA tasks.The source code is available at https://github.com/jxh4945777/MMEA/.

</details>

---

## 114. PSLM: Parallel Generation of Text and Speech withLLMs for Low-Latency Spoken Dialogue Systems

- [ ] PSLM: Parallel Generation of Text and Speech withLLMs for Low-Latency Spoken Dialogue Systems | https://aclanthology.org/2024.findings-emnlp.151/

- **Link**: https://aclanthology.org/2024.findings-emnlp.151/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal language models that process both text and speech have a potential for applications in spoken dialogue systems. However, current models face two major challenges in response generation latency: (1) generating a spoken response requires the prior generation of a written response, and (2) speech sequences are significantly longer than text sequences. This study addresses these issues by extending the input and output sequences of the language model to support the parallel generation of text and speech. Our experiments on spoken question answering tasks demonstrate that our approach improves latency while maintaining the quality of response content. Additionally, we show that latency can be further reduced by generating speech in multiple sequences. Demo samples are available at https://rinnakk.github.io/research/publications/PSLM.

</details>

---

## 115. mPLUG-DocOwl 1.5: Unified Structure Learning forOCR-free Document Understanding

- [ ] mPLUG-DocOwl 1.5: Unified Structure Learning forOCR-free Document Understanding | https://aclanthology.org/2024.findings-emnlp.175/

- **Link**: https://aclanthology.org/2024.findings-emnlp.175/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Structure information is critical for understanding the semantics of text-rich images, such as documents, tables, and charts. Existing Multimodal Large Language Models (MLLMs) for Visual Document Understanding are equipped with text recognition ability but lack general structure understanding abilities for text-rich document images. In this work, we emphasize the importance of structure information in Visual Document Understanding and propose Unified Structure Learning to boost the performance of MLLMs. Based on publicly available text-rich images, we build a comprehensive training set DocStruct4M to support structure-aware parsing tasks and multi-grained text localization tasks across 5 domains: document, webpage, table, chart, and natural image. To better encode structure information, we design a simple and effective vision-to-text module H-Reducer, which can not only maintain the layout information but also reduce the length of visual features by merging horizontal adjacent patches through convolution, enabling the LLM to understand high-resolution images more efficiently. Our model DocOwl 1.5 achieves state-of-the-art performance on 10 visual document understanding benchmarks. All codes, models, and datasets are publicly available at https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5.

</details>

---

## 116. Granular Entity Mapper: Advancing Fine-grained Multimodal Named Entity Recognition and Grounding

- [ ] Granular Entity Mapper: Advancing Fine-grained Multimodal Named Entity Recognition and Grounding | https://aclanthology.org/2024.findings-emnlp.183/

- **Link**: https://aclanthology.org/2024.findings-emnlp.183/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Named Entity Recognition and Grounding (MNERG) aims to extract paired textual and visual entities from texts and images. It has been well explored through a two-step paradigm: initially identifying potential visual entities using object detection methods and then aligning the extracted textual entities with their corresponding visual entities. However, when it comes to fine-grained MNERG, the long-tailed distribution of textual entity categories and the performance of object detectors limit the effectiveness of traditional methods. Specifically, more detailed classification leads to many low-frequency categories, and existing object detection methods often fail to pinpoint subtle regions within images. To address these challenges, we propose the Granular Entity Mapper (GEM) framework. Firstly, we design a multi-granularity entity recognition module, followed by a reranking module based on the Multimodal Large Language Model (MLLM) to incorporate hierarchical information of entity categories, visual cues, and external textual resources collectively for accurate fine-grained textual entity recognition. Then, we utilize a pre-trained Large Visual Language Model (LVLM) as an implicit visual entity grounder that directly deduces relevant visual entity regions from the entire image without the need for bounding box training. Experimental results on the GMNER and FMNERG datasets demonstrate that our GEM framework achieves state-of-the-art results on the fine-grained content extraction task.

</details>

---

## 117. Are Large Vision Language Models up to the Challenge of Chart Comprehension and Reasoning

- [ ] Are Large Vision Language Models up to the Challenge of Chart Comprehension and Reasoning | https://aclanthology.org/2024.findings-emnlp.191/

- **Link**: https://aclanthology.org/2024.findings-emnlp.191/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Natural language is a powerful complementary modality of communication for data visualizations, such as bar and line charts. To facilitate chart-based reasoning using natural language, various downstream tasks have been introduced recently such as chart question answering, chart summarization, and fact-checking with charts. These tasks pose a unique challenge, demanding both vision-language reasoning and a nuanced understanding of chart data tables, visual encodings, and natural language instructions. Despite the recent success of Large Language Models (LLMs) across diverse NLP tasks, their abilities and limitations in the realm of data visualization remain under-explored, possibly due to their lack of multi-modal capabilities. To bridge the gap, this paper presents one of the first comprehensive evaluations of the recently developed large vision language models (LVLMs) for chart understanding and reasoning tasks. Our evaluation includes a comprehensive assessment of both closed and open-sourced LVLMs across five major chart reasoning tasks. Furthermore, we perform a qualitative evaluation of LVLMs’ performance on a diverse range of charts, aiming to provide a thorough analysis. Our findings reveal that while LVLMs demonstrate impressive abilities in generating fluent texts covering high-level data insights, they also encounter common problems like hallucinations, factual errors, and data bias. We highlight the key strengths and limitations of LVLMs in chart comprehension tasks, offering insights for future research

</details>

---

## 118. Precision or Recall? An Analysis of Image Captions for Training Text-to-Image Generation Model

- [ ] Precision or Recall? An Analysis of Image Captions for Training Text-to-Image Generation Model | https://aclanthology.org/2024.findings-emnlp.211/

- **Link**: https://aclanthology.org/2024.findings-emnlp.211/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Despite advancements in text-to-image models, generating images that precisely align with textual descriptions remains challenging due to misalignment in training data. In this paper, we analyze the critical role of caption precision and recall in text-to-image model training. Our analysis of human-annotated captions shows that both precision and recall are important for text-image alignment, but precision has a more significant impact. Leveraging these insights, we utilize Large Vision Language Models to generate synthetic captions for training. Models trained with these synthetic captions show similar behavior to those trained on human-annotated captions, underscores the potential for synthetic data in text-to-image training.

</details>

---

## 119. Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models

- [ ] Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models | https://aclanthology.org/2024.findings-emnlp.221/

- **Link**: https://aclanthology.org/2024.findings-emnlp.221/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in general-purpose or domain-specific multimodal large language models (LLMs) have witnessed remarkable progress for medical decision-making. However, they are designated for specific classification or generative tasks, and require model training or finetuning on large-scale datasets with sizeable parameters and tremendous computing, hindering their clinical utility across diverse resource-constrained scenarios in practice. In this paper, we propose a novel and lightweight framework Med-MoE (Mixture-of-Experts) that tackles both discriminative and generative multimodal medical tasks. The learning of Med-MoE consists of three steps: multimodal medical alignment, Instruction tuning and routing, and domain-specific MoE tuning. After aligning multimodal medical images with LLM tokens, we then enable the model for different multimodal medical tasks with instruction tuning, together with a trainable router tailored for expert selection across input modalities. Finally, the model is tuned by integrating the router with multiple domain-specific experts, which are selectively activated and further empowered by meta experts. Comprehensive experiments on both open- and close-end medical question answering (Med-VQA) and image classification tasks across datasets such as VQA-RAD, SLAKE and Path-VQA demonstrate that our model can achieve performance superior to or on par with state-of-the-art baselines, while only requiring approximately 30%-50% of activated model parameters. Extensive analysis and ablations corroborate the effectiveness and practical utility of our method.

</details>

---

## 120. LOOK-M: Look-Once Optimization inKVCache for Efficient Multimodal Long-Context Inference

- [ ] LOOK-M: Look-Once Optimization inKVCache for Efficient Multimodal Long-Context Inference | https://aclanthology.org/2024.findings-emnlp.235/

- **Link**: https://aclanthology.org/2024.findings-emnlp.235/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Long-context Multimodal Large Language Models (MLLMs) demand substantial computational resources for inference as the growth of their multimodal Key-Value (KV) cache, in response to increasing input lengths, challenges memory and time efficiency. Unlike single-modality LLMs that manage only textual contexts, the KV cache of long-context MLLMs includes representations from multiple images with temporal and spatial relationships and related textual contexts. The predominance of image tokens means traditional optimizations for LLMs’ KV caches are unsuitable for multimodal long-context settings, and no prior works have addressed this challenge.In this work, we introduce **LOOK-M**, a pioneering, fine-tuning-free approach that efficiently reduces the multimodal KV cache size while maintaining performance comparable to a full cache. We observe that during prompt prefill, the model prioritizes more textual attention over image features, and based on the multimodal interaction observation, a new proposed text-prior method is explored to compress the KV cache. Furthermore, to mitigate the degradation of image contextual information, we propose several compensatory strategies using KV pairs merging. **LOOK-M** demonstrates that with a significant reduction in KV Cache memory usage, such as reducing it by **80%** in some cases, it not only achieves approximately **1.3x** faster decoding but also maintains or even **enhances** performance across a variety of long context multimodal tasks.

</details>

---

## 121. M5 – A Diverse Benchmark to Assess the Performance of Large Multimodal Models Across Multilingual and Multicultural Vision-Language Tasks

- [ ] M5 – A Diverse Benchmark to Assess the Performance of Large Multimodal Models Across Multilingual and Multicultural Vision-Language Tasks | https://aclanthology.org/2024.findings-emnlp.250/

- **Link**: https://aclanthology.org/2024.findings-emnlp.250/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Since the release of ChatGPT, the field of Natural Language Processing has experienced rapid advancements, particularly in Large Language Models (LLMs) and their multimodal counterparts, Large Multimodal Models (LMMs). Despite their impressive capabilities, LLMs often exhibit significant performance disparities across different languages and cultural contexts, as demonstrated by various text-only benchmarks. However, current research lacks such benchmarks for multimodal visio-linguistic settings. This work fills this gap by introducing M5, the first comprehensive benchmark designed to evaluate LMMs on diverse vision-language tasks within a multilingual and multicultural context. M5 includes eight datasets covering five tasks and 41 languages, with a focus on underrepresented languages and culturally diverse images. Furthermore, we introduce two novel datasets, M5-VGR and M5-VLOD, including a new Visio-Linguistic Outlier Detection task, in which all evaluated open-source models fail to significantly surpass the random baseline. Through extensive evaluation and analyses, we highlight substantial task-agnostic performance disparities between high- and low-resource languages. Moreover, we show that larger models do not necessarily outperform smaller ones in a multilingual setting.

</details>

---

## 122. Reference-free Hallucination Detection for Large Vision-Language Models

- [ ] Reference-free Hallucination Detection for Large Vision-Language Models | https://aclanthology.org/2024.findings-emnlp.262/

- **Link**: https://aclanthology.org/2024.findings-emnlp.262/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have made significant progress in recent years. While LVLMs exhibit excellent ability in language understanding, question answering, and conversations of visual inputs, they are prone to producing hallucinations. While several methods are proposed to evaluate the hallucinations in LVLMs, most are reference-based and depend on external tools, which complicates their practical application. To assess the viability of alternative methods, it is critical to understand whether the reference-free approaches, which do not rely on any external tools, can efficiently detect hallucinations. Therefore, we initiate an exploratory study to demonstrate the effectiveness of different reference-free solutions in detecting hallucinations in LVLMs. In particular, we conduct an extensive study on three kinds of techniques: uncertainty-based, consistency-based, and supervised uncertainty quantification methods on four representative LVLMs across two different tasks. The empirical results show that the reference-free approaches are capable of effectively detecting non-factual responses in LVLMs, with the supervised uncertainty quantification method outperforming the others, achieving the best performance across different settings.

</details>

---

## 123. WavLLM: Towards Robust and Adaptive Speech Large Language Model

- [ ] WavLLM: Towards Robust and Adaptive Speech Large Language Model | https://aclanthology.org/2024.findings-emnlp.263/

- **Link**: https://aclanthology.org/2024.findings-emnlp.263/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in large language models (LLMs) have expanded their scope in natural language processing (NLP) to encompass multimodal functions. However, integrating listening capabilities effectively remains a significant challenge for generalization and complex auditory task execution. In this work, we introduce WavLLM, a robust and adaptive speech large language model featuring dual encoders—a Whisper encoder for semantics and a WavLM encoder for speaker characteristics. Within the two-stage curriculum learning framework, WavLLM first builds its foundational capabilities by optimizing on mixed elementary single tasks, followed by advanced multi-task training on more complex tasks such as combinations of the elementary tasks. To enhance the flexibility and adherence to different tasks and instructions, a prompt-aware LoRA weight adapter is introduced in the second advanced multi-task training stage. We validate the proposed model on universal speech benchmarks and also apply it to specialized speech-question-answer (SQA) dataset, and speech Chain-of-Thought (CoT) evaluation set. Experiments demonstrate that the proposed model achieves state-of-the-art performance across a range of speech tasks on the same model size, exhibiting robust generalization capabilities in executing complex tasks using CoT approach. The codes, models, audio samples, and SQA evaluation set can be accessed athttps://github.com/microsoft/SpeechT5/tree/main/WavLLM.

</details>

---

## 124. Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation

- [ ] Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation | https://aclanthology.org/2024.findings-emnlp.269/

- **Link**: https://aclanthology.org/2024.findings-emnlp.269/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study presents a novel evaluation framework for the Vision-Language Navigation (VLN) task. It aims to diagnose current models for various instruction categories at a finer-grained level. The framework is structured around the context-free grammar (CFG) of the task. The CFG serves as the basis for the problem decomposition and the core premise of the instruction categories design. We propose a semi-automatic method for CFG construction with the help of Large-Language Models (LLMs). Then, we induct and generate data spanning five principal instruction categories (i.e. direction change, landmark recognition, region recognition, vertical movement, and numerical comprehension). Our analysis of different models reveals notable performance discrepancies and recurrent issues. The stagnation of numerical comprehension, heavy selective biases over directional concepts, and other interesting findings contribute to the development of future language-guided navigation systems. The project is now available at https://zehao-wang.github.io/navnuances.

</details>

---

## 125. Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models

- [ ] Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models | https://aclanthology.org/2024.findings-emnlp.268/

- **Link**: https://aclanthology.org/2024.findings-emnlp.268/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have demonstrated impressive reasoning capabilities, particularly in textual mathematical problem-solving. However, existing open-source image instruction fine-tuning datasets, containing limited question-answer pairs per image, do not fully exploit visual information to enhance the multimodal mathematical reasoning capabilities of Multimodal LLMs (MLLMs). To bridge this gap, we address the lack of high-quality, diverse multimodal mathematical datasets by collecting 40K high-quality images with question-answer pairs from 24 existing datasets and synthesizing 320K new pairs, creating the MathV360K dataset, which enhances both the breadth and depth of multimodal mathematical questions. We introduce Math-LLaVA, a LLaVA-1.5-based model fine-tuned with MathV360K. This novel approach significantly improves the multimodal mathematical reasoning capabilities of LLaVA-1.5, achieving a 19-point increase and comparable performance to GPT-4V on MathVista’s minitest split, and yielding leading performance on Math-V and MathVerse. Furthermore, Math-LLaVA demonstrates enhanced generalizability, showing substantial improvements on the MMMU benchmark. Our research highlights the importance of dataset diversity and synthesis in advancing MLLMs’ mathematical reasoning abilities. The code and data are available at:https://github.com/HZQ950419/Math-LLaVA.

</details>

---

## 126. Geneverse: A Collection of Open-source Multimodal Large Language Models for Genomic and Proteomic Research

- [ ] Geneverse: A Collection of Open-source Multimodal Large Language Models for Genomic and Proteomic Research | https://aclanthology.org/2024.findings-emnlp.277/

- **Link**: https://aclanthology.org/2024.findings-emnlp.277/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The applications of large language models (LLMs) are promising for biomedical and healthcare research. Despite the availability of open-source LLMs trained using a wide range of biomedical data, current research on the applications of LLMs to genomics and proteomics is still limited. To fill this gap, we propose a collection of finetuned LLMs and multimodal LLMs (MLLMs), known as Geneverse, for three novel tasks in genomic and proteomic research. The models in Geneverse are trained and evaluated based on domain-specific datasets, and we use advanced parameter-efficient finetuning techniques to achieve the model adaptation for tasks including the generation of descriptions for gene functions, protein function inference from its structure, and marker gene selection from spatial transcriptomic data. We demonstrate that adapted LLMs and MLLMs perform well for these tasks and may outperform closed-source large-scale models based on our evaluations focusing on both truthfulness and structural correctness. All of the training strategies and base models we used are freely accessible. Our codes can be found athttps://github.com/HelloWorldLTY/Geneverse.

</details>

---

## 127. CONSTRUCTURE: BenchmarkingCONceptSTRUCTUreREasoning for Multimodal Large Language Models

- [ ] CONSTRUCTURE: BenchmarkingCONceptSTRUCTUreREasoning for Multimodal Large Language Models | https://aclanthology.org/2024.findings-emnlp.285/

- **Link**: https://aclanthology.org/2024.findings-emnlp.285/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown promising results in various tasks, but their ability to perceive the visual world with deep, hierarchical understanding similar to humans remains uncertain. To address this gap, we introduce CONSTRUCTURE, a novel concept-level benchmark to assess MLLMs’ hierarchical concept understanding and reasoning abilities. Our goal is to evaluate MLLMs across four key aspects: 1) Understanding atomic concepts at different levels of abstraction; 2) Performing upward abstraction reasoning across concepts; 3) Achieving downward concretization reasoning across concepts; and 4) Conducting multi-hop reasoning between sibling or common ancestor concepts. Our findings indicate that even state-of-the-art multimodal models struggle with concept structure reasoning (e.g., GPT-4o averages a score of 62.1%). We summarize key findings of MLLMs in concept structure reasoning evaluation. Morever, we provide key insights from experiments using CoT prompting and fine-tuning to enhance their abilities.

</details>

---

## 128. FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models

- [ ] FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models | https://aclanthology.org/2024.findings-emnlp.290/

- **Link**: https://aclanthology.org/2024.findings-emnlp.290/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce FaithScore (Faithfulness to Atomic Image Facts Score), a reference-free and fine-grained evaluation metric that measures the faithfulness of the generated free-form answers from large vision-language models (LVLMs). The FaithScore evaluation first identifies sub-sentences containing descriptive statements that need to be verified, then extracts a comprehensive list of atomic facts from these sub-sentences, and finally conducts consistency verification between fine-grained atomic facts and the input image. Meta-evaluation demonstrates that our metric highly correlates with human judgments of faithfulness. We collect two benchmark datasets (i.e. LLaVA-1k and MSCOCO-Cap) for evaluating LVLMs instruction-following hallucinations. We measure hallucinations in state-of-the-art LVLMs with FaithScore on the datasets. Results reveal that current systems are prone to generate hallucinated content unfaithful to the image, which leaves room for future improvements. We hope our metric FaithScore can help evaluate future LVLMs in terms of faithfulness and provide insightful advice for enhancing LVLMs’ faithfulness.

</details>

---

## 129. Losing Visual Needles in Image Haystacks: Vision Language Models are Easily Distracted in Short and Long Contexts

- [ ] Losing Visual Needles in Image Haystacks: Vision Language Models are Easily Distracted in Short and Long Contexts | https://aclanthology.org/2024.findings-emnlp.312/

- **Link**: https://aclanthology.org/2024.findings-emnlp.312/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We present LoCoVQA, a dynamic benchmark generator for evaluating long-context reasoning in vision language models (VLMs). LoCoVQA augments test examples for mathematical reasoning, VQA, and character recognition tasks with increasingly long visual contexts composed of both in-distribution and out-of-distribution distractor images.Across these tasks, a diverse set of VLMs rapidly lose performance as the visual context length grows, often exhibiting a striking logarithmic decay trend. This test assesses how well VLMs can ignore irrelevant information when answering queries—a task that is quite easy for language models (LMs) in the text domain—demonstrating that current state-of-the-art VLMs lack this essential capability for many long-context applications.

</details>

---

## 130. Unveiling the Invisible: Captioning Videos with Metaphors

- [ ] Unveiling the Invisible: Captioning Videos with Metaphors | https://aclanthology.org/2024.findings-emnlp.366/

- **Link**: https://aclanthology.org/2024.findings-emnlp.366/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Metaphors are a common communication tool used in our day-to-day life. The detection and generation of metaphors in textual form have been studied extensively but metaphors in other forms have been under-explored. Recent studies have shown that Vision-Language (VL) models cannot understand visual metaphors in memes and adverts. As of now, no probing studies have been done that involve complex language phenomena like metaphors with videos. Hence, we introduce a new VL task of describing the metaphors present in the videos in our work. To facilitate this novel task, we construct and release a manually created dataset with 705 videos and 2115 human-written captions, along with a new metric called Average Concept Distance (ACD), to automatically evaluate the creativity of the metaphors generated. We also propose a novel low-resource video metaphor captioning system: GIT-LLaVA, which obtains comparable performance to SoTA video language models on the proposed task. We perform a comprehensive analysis of existing video language models on this task and publish our dataset, models, and benchmark results to enable further research.

</details>

---

## 131. VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning withLLMs

- [ ] VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning withLLMs | https://aclanthology.org/2024.findings-emnlp.384/

- **Link**: https://aclanthology.org/2024.findings-emnlp.384/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In the video-language domain, recent works in leveraging zero-shot Large Language Model-based reasoning for video understanding have become competitive challengers to previous end-to-end models. However, long video understanding presents unique challenges due to the complexity of reasoning over extended timespans, even for zero-shot LLM-based approaches. The challenge of information redundancy in long videos prompts the question of what specific information is essential for large language models (LLMs) and how to leverage them for complex spatial-temporal reasoning in long-form video analysis. We propose a framework VideoINSTA , i.e. INformative Spatial-TemporAl Reasoning for zero-shot long-form video understanding.VideoINSTA contributes (1) a zero-shot framework for long video understanding using LLMs; (2) an event-based temporalreasoning and content-based spatial reasoning approach for LLMs to reason over spatial-temporal information in videos; (3) a self-reflective information reasoning scheme based on information sufficiency and prediction confidence while balancing temporal factors.Our model significantly improves the state-of-the-art on three long video question-answering benchmarks: EgoSchema, NextQA, and IntentQA, and the open question answering dataset ActivityNetQA. Code is released: https://github.com/mayhugotong/VideoINSTA.

</details>

---

## 132. MMCode: Benchmarking Multimodal Large Language Models for Code Generation with Visually Rich Programming Problems

- [ ] MMCode: Benchmarking Multimodal Large Language Models for Code Generation with Visually Rich Programming Problems | https://aclanthology.org/2024.findings-emnlp.42/

- **Link**: https://aclanthology.org/2024.findings-emnlp.42/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Programming often involves converting detailed and complex specifications into code, a process during which developers typically utilize visual aids to more effectively convey concepts. While recent developments in Large Multimodal Models have demonstrated remarkable abilities in visual reasoning and mathematical tasks, there is little work on investigating whether these models can effectively interpret visual elements for code generation. To this end, we present MMCode, the first multi-modal coding dataset for evaluating algorithmic problem-solving skills in visually rich contexts. MMCode contains 3,548 questions and 6,620 images collected from real-world programming challenges harvested from 10 code competition websites, presenting significant challenges due to the extreme demand for reasoning abilities. Our experiment results show that current state-of-the-art models struggle to solve these problems. The results highlight the lack of powerful vision-code models, and we hope MMCode can serve as an inspiration for future works in this domain. The data and code are publicly available.

</details>

---

## 133. Difficult Task Yes but Simple Task No: Unveiling the Laziness in MultimodalLLMs

- [ ] Difficult Task Yes but Simple Task No: Unveiling the Laziness in MultimodalLLMs | https://aclanthology.org/2024.findings-emnlp.442/

- **Link**: https://aclanthology.org/2024.findings-emnlp.442/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) demonstrate a strong understanding of the real world and can even handle complex tasks. However, they still fail on some straightforward visual question-answering (VQA) problems. This paper dives deeper into this issue, revealing that models tend to err when answering easy questions (e.g., Yes/No questions) about an image, even though they can correctly describe it.We refer to this model behavior discrepancy between difficult and simple questions as model laziness.To systematically investigate model laziness, we manually construct LazyBench, a benchmark that includes Yes/No, multiple choice, short answer questions, and image description tasks that are related to the same subjects in the images.Based on LazyBench. we observe that laziness widely exists in current advanced MLLMs (e.g., GPT-4o, Gemini-1.5-pro, Claude 3, LLaVA-1.5, LLaVA-1.6, and QWen-VL). We also analyzed the failure cases of LLaVA-1.5-13B on the VQA-v2 benchmark and discovered that about half of these failures are due to the model’s laziness. This further highlights the importance of ensuring that the model fully utilizes its capability.To this end, we conduct a preliminary exploration of how to mitigate laziness and find that chain of thought can effectively avoid this issue. The data can be accessed at https://github.com/Akutagawa1998/LazyBench.

</details>

---

## 134. MACAROON: Training Vision-Language Models To Be Your Engaged Partners

- [ ] MACAROON: Training Vision-Language Models To Be Your Engaged Partners | https://aclanthology.org/2024.findings-emnlp.454/

- **Link**: https://aclanthology.org/2024.findings-emnlp.454/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs), while proficient in following instructions and responding to diverse questions, invariably generate detailed responses even when questions are ambiguous or unanswerable, leading to hallucinations and bias issues. Thus, it is essential for LVLMs to proactively engage with humans to ask for clarifications or additional information for better responses. In this study, we aim to shift LVLMs from passive answer providers to proactive engaged partners. We begin by establishing a three-tiered hierarchy for questions of invalid, ambiguous, and personalizable nature to measure the proactive engagement capabilities of LVLMs. Utilizing this hierarchy, we create PIE, (ProactIve Engagement Evaluation) through GPT-4o and human annotators, consisting of 853 questions across six distinct, fine-grained question types that are verified by human annotators and accompanied with well-defined metrics. Our evaluations on indicate poor performance of existing LVLMs, with the best-performing open-weights model only achieving an Aggregate Align Rate (AAR) of 0.28. In response, we introduce MACAROON, self-iMaginAtion for ContrAstive pReference OptimizatiON, which instructs LVLMs to autonomously generate contrastive response pairs for unlabeled questions given the task description and human-crafted criteria. Then, the self-imagined data is formatted for conditional reinforcement learning. Experimental results show MACAROON effectively improves LVLMs’ capabilities to be proactively engaged (0.84 AAR) while maintaining comparable performance on general tasks.

</details>

---

## 135. IsGPT-4V(ision) All You Need for Automating Academic Data Visualization? Exploring Vision-Language Models’ Capability in Reproducing Academic Charts

- [ ] IsGPT-4V(ision) All You Need for Automating Academic Data Visualization? Exploring Vision-Language Models’ Capability in Reproducing Academic Charts | https://aclanthology.org/2024.findings-emnlp.485/

- **Link**: https://aclanthology.org/2024.findings-emnlp.485/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While effective data visualization is crucial to present complex information in academic research, its creation demands significant expertise in both data management and graphic design. We explore the potential of using Vision-Language Models (VLMs) in automating the creation of data visualizations by generating code templates from existing charts. As the first work to systematically investigate this task, we first introduce AcademiaChart, a dataset comprising 2525 high-resolution data visualization figures with captions from a variety of AI conferences, extracted directly from source codes. We then conduct large-scale experiments with six state-of-the-art (SOTA) VLMs, including both closed-source and open-source models. Our findings reveal that SOTA closed-source VLMs can indeed be helpful in reproducing charts. On the contrary, open-source ones are only effective at reproducing much simpler charts but struggle with more complex ones. Interestingly, the application of Chain-of-Thought (CoT) prompting significantly enhances the performance of the most advanced model, GPT-4-V, while it does not work as well for other models. These results underscore the potential of VLMs in data visualization while also highlighting critical areas that need improvement for broader application.

</details>

---

## 136. AutoHallusion: Automatic Generation of Hallucination Benchmarks for Vision-Language Models

- [ ] AutoHallusion: Automatic Generation of Hallucination Benchmarks for Vision-Language Models | https://aclanthology.org/2024.findings-emnlp.493/

- **Link**: https://aclanthology.org/2024.findings-emnlp.493/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) are prone to hallucinations, where certain contextual cues in an image can trigger the language module to produce overconfident and incorrect reasoning about abnormal or hypothetical objects. While some benchmarks have been developed to investigate LVLM hallucinations, they often rely on hand-crafted corner cases whose failure patterns may not generalize well. Additionally, fine-tuning on these examples could undermine their validity. To address this, we aim to scale up the number of cases through an automated approach, reducing human bias in crafting such corner cases. This motivates the development of AutoHallusion, the first automated benchmark generation approach that employs several key strategies to create a diverse range of hallucination examples. Our generated visual-question pairs pose significant challenges to LVLMs, requiring them to overcome contextual biases and distractions to arrive at correct answers. AutoHallusion enables us to create new benchmarks at the minimum cost and thus overcomes the fragility of hand-crafted benchmarks. It also reveals common failure patterns and reasons, providing key insights to detect, avoid, or control hallucinations. Comprehensive evaluations of top-tier LVLMs, e.g., GPT-4V(ision), Gemini Pro Vision, Claude 3, and LLaVA-1.5, show a 97.7% and 98.7% success rate of hallucination induction on synthetic and real-world datasets of AutoHallusion, paving the way for a long battle against hallucinations. The codebase and data can be accessed at https://github.com/wuxiyang1996/AutoHallusion

</details>

---

## 137. Infrared-LLaVA: Enhancing Understanding of Infrared Images in Multi-Modal Large Language Models

- [ ] Infrared-LLaVA: Enhancing Understanding of Infrared Images in Multi-Modal Large Language Models | https://aclanthology.org/2024.findings-emnlp.501/

- **Link**: https://aclanthology.org/2024.findings-emnlp.501/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Expanding the understanding capabilities of multi-modal large language models (MLLMs) for infrared modality is a challenge due to the single-modality nature and limited amount of training data. Existing methods typically construct a uniform embedding space for cross-modal alignment and leverage abundant visual image data to indirectly understand infrared images. However, they ignore the supervisory signals of infrared-modality-specific attributes, which may lead to biased understanding of infrared images. To address this issue, we propose a debating multi-agent generation system which transfers knowledge from visible images to generate infrared image-text pairs and infrared instruction data. Moreover, we construct an infrared question-answering benchmark based on common infrared tasks. Experimental results from incremental fine-tuning on existing models and our Infrared-LLaVA-7B trained from scratch on infrared data demonstrate the effectiveness of the generated data and the feasibility of the generation approach.

</details>

---

## 138. Exploring the Capability of MultimodalLLMs with Yonkoma Manga: TheYManga Dataset and Its Challenging Tasks

- [ ] Exploring the Capability of MultimodalLLMs with Yonkoma Manga: TheYManga Dataset and Its Challenging Tasks | https://aclanthology.org/2024.findings-emnlp.506/

- **Link**: https://aclanthology.org/2024.findings-emnlp.506/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Yonkoma Manga, characterized by its four-panel structure, presents unique challenges due to its rich contextual information and strong sequential features. To address the limitations of current multimodal large language models (MLLMs) in understanding this type of data, we create a novel dataset named YManga from the Internet. After filtering out low-quality content, we collect a dataset of 1,015 yonkoma strips, containing 10,150 human annotations. We then define three challenging tasks for this dataset: panel sequence detection, generation of the author’s creative intention, and description generation for masked panels. These tasks progressively introduce the complexity of understanding and utilizing such image-text data. To the best of our knowledge, YManga is the first dataset specifically designed for yonkoma manga strips understanding. Extensive experiments conducted on this dataset reveal significant challenges faced by current multimodal large language models. Our results show a substantial performance gap between models and humans across all three tasks.

</details>

---

## 139. MMedAgent: Learning to Use Medical Tools with Multi-modal Agent

- [ ] MMedAgent: Learning to Use Medical Tools with Multi-modal Agent | https://aclanthology.org/2024.findings-emnlp.510/

- **Link**: https://aclanthology.org/2024.findings-emnlp.510/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-Modal Large Language Models (MLLMs), despite being successful, exhibit limited generality and often fall short when compared to specialized models. Recently, LLM-based agents have been developed to address these challenges by selecting appropriate specialized models as tools based on user inputs. However, such advancements have not been extensively explored within the medical domain. To bridge this gap, this paper introduces the first agent explicitly designed for the medical field, namedMulti-modalMedicalAgent(MMedAgent). We curate an instruction-tuning dataset comprising six medical tools solving seven tasks across five modalities, enabling the agent to choose the most suitable tools for a given task. Comprehensive experiments demonstrate that MMedAgent achieves superior performance across a variety of medical tasks compared to state-of-the-art open-source methods and even the closed-source model, GPT-4o. Furthermore, MMedAgent exhibits efficiency in updating and integrating new medical tools.

</details>

---

## 140. Can Textual Unlearning Solve Cross-Modality Safety Alignment?

- [ ] Can Textual Unlearning Solve Cross-Modality Safety Alignment? | https://aclanthology.org/2024.findings-emnlp.574/

- **Link**: https://aclanthology.org/2024.findings-emnlp.574/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent studies reveal that integrating new modalities into large language models (LLMs), such as vision-language models (VLMs), creates a new attack surface that bypasses existing safety training techniques like supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF). While further SFT and RLHF-based safety training can be conducted in multi-modal settings, collecting multi-modal training datasets poses a significant challenge. Inspired by the structural design of recent multi-modal models, where all input modalities are ultimately fused into the language space, we explore whether unlearning solely in the textual domain can be effective for cross-modality safety alignment. Our empirical evaluation across seven datasets demonstrates promising transferability — textual unlearning in VLMs significantly reduces the Attack Success Rate (ASR) to less than 8% and in some cases, even as low as nearly 2% for both text-based and vision-text-based attacks, alongside preserving the utility. Moreover, our experiments show that unlearning with a multi-modal dataset offers no potential benefits but incurs significantly increased computational demands.

</details>

---

## 141. Pruning Multilingual Large Language Models for Multilingual Inference

- [ ] Pruning Multilingual Large Language Models for Multilingual Inference | https://aclanthology.org/2024.findings-emnlp.580/

- **Link**: https://aclanthology.org/2024.findings-emnlp.580/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multilingual large language models (MLLMs), trained on multilingual balanced data, demonstrate better zero-shot learning performance in non-English languages compared to large language models trained on English-dominant data. However, the disparity in performance between English and non-English languages remains a challenge yet to be fully addressed. This study introduces a promising direction for enhancing non-English performance through a specialized pruning approach. Specifically, we prune MLLMs using bilingual sentence pairs from English and other languages and empirically demonstrate that this pruning strategy can enhance the MLLMs’ performance in non-English language.

</details>

---

## 142. Employing Glyphic Information forChinese Event Extraction with Vision-Language Model

- [ ] Employing Glyphic Information forChinese Event Extraction with Vision-Language Model | https://aclanthology.org/2024.findings-emnlp.58/

- **Link**: https://aclanthology.org/2024.findings-emnlp.58/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

As a complex task that requires rich information input, features from various aspects have been utilized in event extraction. However, most of the previous works ignored the value of glyph, which could contain enriched semantic information and can not be fully expressed by the pre-trained embedding in hieroglyphic languages like Chinese. We argue that, compared with combining the sophisticated textual features, glyphic information from visual modality could provide us with extra and straight semantic information in extracting events. Motivated by this, we propose a glyphic multi-modal Chinese event extraction model with hieroglyphic images to capture the intra- and inter-character morphological structure from the sequence. Extensive experiments build a new state-of-the-art performance in the ACE2005 Chinese and KBP Eval 2017 dataset, which underscores the effectiveness of our proposed glyphic event extraction model, and more importantly, the glyphic feature can be obtained at nearly zero cost.

</details>

---

## 143. VPL: Visual Proxy Learning Framework for Zero-Shot Medical Image Diagnosis

- [ ] VPL: Visual Proxy Learning Framework for Zero-Shot Medical Image Diagnosis | https://aclanthology.org/2024.findings-emnlp.583/

- **Link**: https://aclanthology.org/2024.findings-emnlp.583/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models like CLIP, utilizing class proxies derived from class name text features, have shown a notable capability in zero-shot medical image diagnosis which is vital in scenarios with limited disease databases or labeled samples. However, insufficient medical text precision and the modal disparity between text and vision spaces pose challenges for such paradigm. We show analytically and experimentally that enriching medical texts with detailed descriptions can markedly enhance the diagnosis performance, with the granularity and phrasing of these enhancements having a crucial impact on CLIP’s understanding of medical images; and learning proxies within the vision domain can effectively circumvent the modal gap issue. Based on our analysis, we propose a medical visual proxy learning framework comprising two key components: a text refinement module that create high quality medical text descriptions, and a stable Sinkhorn algorithm for an efficient generation of pseudo labels which further guide the visual proxy learning. Our method elevates the Vanilla CLIP inference by supplying meticulously crafted clues to leverage CLIP’s existing interpretive power and using the feature of refined texts to bridge the vision-text gap. The effectiveness and robustness of our method are clearly demonstrated through extensive experiments. Notably, our method outperforms the state-of-the-art zero-shot medical image diagnosis by a significant margin, ranging from 1.69% to 15.31% on five datasets covering various diseases, confirming its immense potential in zero-shot diagnosis across diverse medical applications.

</details>

---

## 144. Unleashing the Potentials of Likelihood Composition for Multi-modal Language Models

- [ ] Unleashing the Potentials of Likelihood Composition for Multi-modal Language Models | https://aclanthology.org/2024.findings-emnlp.594/

- **Link**: https://aclanthology.org/2024.findings-emnlp.594/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Model fusing has always been an important topic, especially in an era where large language models (LLM) and multi-modal language models (MLM) with different architectures, parameter sizes and training pipelines, are being created all the time. In this work, we propose a post-hoc framework, aiming at fusing heterogeneous models off-the-shell, which we calllikelihood composition, and the basic idea is to compose multiple models’ likelihood distribution when doing a multi-choice visual-question-answering task. Here the core concept,likelihood, is actually the log-probability of the candidate answer. Inlikelihood composition, we introduce some basic operations:debias,highlight,majority-voteandensemble. By combining (composing) these basic elements, we get the mixed composition methods:mix-composition. Through conducting comprehensive experiments on 9 VQA datasets and 10 MLMs, we prove the effectiveness ofmix-compositioncompared with simpleensembleormajority-votemethods. In this framework, people can propose new basic composition methods and combine them to get the new mixed composition methods. We hope our proposedlikelihood compositioncan provide a new perspective of fusing heterogeneous models and inspire the exploration under this framework.

</details>

---

## 145. PRESTO: Progressive Pretraining Enhances Synthetic Chemistry Outcomes

- [ ] PRESTO: Progressive Pretraining Enhances Synthetic Chemistry Outcomes | https://aclanthology.org/2024.findings-emnlp.597/

- **Link**: https://aclanthology.org/2024.findings-emnlp.597/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have seen growing adoption across various scientific disciplines. These advancements encourage the investigation of molecule-text modeling within synthetic chemistry, a field dedicated to designing and conducting chemical reactions to synthesize new compounds with desired properties and applications. Current approaches, however, often neglect the critical role of multi-molecule graph interaction in understanding chemical reactions, leading to suboptimal performance in synthetic chemistry tasks. This study introduces PRESTO (Progressive Pretraining Enhances Synthetic Chemistry Outcomes), a new framework that bridges the molecule-text modality gap by integrating a comprehensive benchmark of pretraining strategies and dataset configurations. It progressively improves multimodal LLMs through cross-modal alignment and multi-graph understanding. Our extensive experiments demonstrate that PRESTO offers competitive results in downstream synthetic chemistry tasks. The code can be found at https://github.com/IDEA-XL/PRESTO.

</details>

---

## 146. MobileVLM: A Vision-Language Model for Better Intra- and Inter-UIUnderstanding

- [ ] MobileVLM: A Vision-Language Model for Better Intra- and Inter-UIUnderstanding | https://aclanthology.org/2024.findings-emnlp.599/

- **Link**: https://aclanthology.org/2024.findings-emnlp.599/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recently, mobile AI agents based on VLMs have been gaining increasing attention. These works typically utilize VLM as a foundation, fine-tuning it with instruction-based mobile datasets. However, these VLMs are typically pre-trained on general-domain data, which often results in a lack of fundamental capabilities specific to the mobile domain. Therefore, they may struggle to recognize specific UI elements and understand intra-UI fine-grained information. In addition, the current fine-tuning task focuses on interacting with the most relevant element for the given instruction. These fine-tuned VLMs may still ignore the relationships between UI pages, neglect the roles of elements in page transitions and lack inter-UI understanding. To address issues, we propose a VLM called MobileVLM, which includes two additional pre-training stages to enhance both intra- and inter-UI understanding. We defined four UI-based pre-training tasks, enabling the model to better perceive fine-grained elements and capture page transition actions. To address the lack of mobile pre-training data, we built a large Chinese mobile dataset Mobile3M from scratch, which contains 3 million UI pages, and real-world transition actions, forming a directed graph structure. Experimental results show MobileVLM excels on both our test set and public mobile benchmarks, outperforming existing VLMs.

</details>

---

## 147. Introducing Spatial Information and a Novel Evaluation Scheme for Open-Domain Live Commentary Generation

- [ ] Introducing Spatial Information and a Novel Evaluation Scheme for Open-Domain Live Commentary Generation | https://aclanthology.org/2024.findings-emnlp.606/

- **Link**: https://aclanthology.org/2024.findings-emnlp.606/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper focuses on the task of open-domain live commentary generation. Compared to domain-specific work in this task, this setting proved particularly challenging due to the absence of domain-specific features. Aiming to bridge this gap, we integrate spatial information by proposing an utterance generation model with a novel spatial graph that is flexible to deal with the open-domain characteristics of the commentaries and significantly improves performance. Furthermore, we propose a novel evaluation scheme, more suitable for live commentary generation, that uses LLMs to automatically check whether generated utterances address essential aspects of the video via the answerability of questions extracted directly from the videos using LVLMs. Our results suggest that using a combination of our answerability score and a standard machine translation metric is likely a more reliable way to evaluate the performance in this task.

</details>

---

## 148. BiasDora: Exploring Hidden Biased Associations in Vision-Language Models

- [ ] BiasDora: Exploring Hidden Biased Associations in Vision-Language Models | https://aclanthology.org/2024.findings-emnlp.611/

- **Link**: https://aclanthology.org/2024.findings-emnlp.611/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Existing works examining Vision-Language Models (VLMs) for social biases predominantly focus on a limited set of documented bias associations, such as gender-profession or race-crime. This narrow scope often overlooks a vast range of unexamined implicit associations, restricting the identification and, hence, mitigation of such biases. We address this gap by probing VLMs to (1) uncover hidden, implicit associations across 9 bias dimensions. We systematically explore diverse input and output modalities and (2) demonstrate how biased associations vary in their negativity, toxicity, and extremity. Our work (3) identifies subtle and extreme biases that are typically not recognized by existing methodologies. We make the **D**ataset **o**f **r**etrieved **a**ssociations (**Dora**) publicly available.

</details>

---

## 149. Multimodal Misinformation Detection by Learning from Synthetic Data with MultimodalLLMs

- [ ] Multimodal Misinformation Detection by Learning from Synthetic Data with MultimodalLLMs | https://aclanthology.org/2024.findings-emnlp.613/

- **Link**: https://aclanthology.org/2024.findings-emnlp.613/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Detecting multimodal misinformation, especially in the form of image-text pairs, is crucial. Obtaining large-scale, high-quality real-world fact-checking datasets for training detectors is costly, leading researchers to use synthetic datasets generated by AI technologies. However, the generalizability of detectors trained on synthetic data to real-world scenarios remains unclear due to the distribution gap. To address this, we propose learning from synthetic data for detecting real-world multimodal misinformation through two model-agnostic data selection methods that match synthetic and real-world data distributions. Experiments show that our method enhances the performance of a small MLLM (13B) on real-world fact-checking datasets, enabling it to even surpass GPT-4V.

</details>

---

## 150. Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models

- [ ] Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models | https://aclanthology.org/2024.findings-emnlp.633/

- **Link**: https://aclanthology.org/2024.findings-emnlp.633/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning pre-trained Vision-Language Models (VLMs) has shown remarkable capabilities in medical image and textual depiction synergy. Nevertheless, many pre-training datasets are restricted by patient privacy concerns, potentially containing noise that can adversely affect downstream performance. Moreover, the growing reliance on multi-modal generation exacerbates this issue because of its susceptibility to adversarial attacks. To investigate how VLMs trained on adversarial noisy data perform on downstream medical tasks, we first craft noisy upstream datasets using multi-modal adversarial attacks. Through our comprehensive analysis, we unveil that moderate noise enhances model robustness and transferability, but increasing noise levels negatively impact downstream task performance. To mitigate this issue, we propose rectify adversarial noise (RAN) framework, a recipe designed to effectively defend adversarial attacks and rectify the influence of upstream noise during fine-tuning.

</details>

---

## 151. AlanaVLM: A Multimodal EmbodiedAIFoundation Model for Egocentric Video Understanding

- [ ] AlanaVLM: A Multimodal EmbodiedAIFoundation Model for Egocentric Video Understanding | https://aclanthology.org/2024.findings-emnlp.649/

- **Link**: https://aclanthology.org/2024.findings-emnlp.649/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

AI personal assistants deployed via robots or wearables require embodied understanding to collaborate with humans effectively. However, current Vision-Language Models (VLMs) primarily focus on third-person view videos, neglecting the richness of egocentric perceptual experience. To address this gap, we propose three key contributions. First, we introduce the Egocentric Video Understanding Dataset (EVUD) for training VLMs on video captioning and question answering tasks specific to egocentric videos. Second, we present , a 7B parameter VLM trained using parameter-efficient methods on EVUD. Finally, we evaluate ‘s capabilities on OpenEQA, a challenging benchmark for embodied video question answering. Our model achieves state-of-the-art performance, outperforming open-source models including strong Socratic models using GPT-4 as a planner by 3.6%.Additionally, we outperform Claude 3 and Gemini Pro Vision 1.0 and showcase competitive results compared to Gemini Pro 1.5 and GPT-4V, even surpassing the latter in spatial reasoning. This research paves the way for building efficient VLMs that can be deployed in robots or wearables, leveraging embodied video understanding to collaborate seamlessly with humans in everyday tasks, contributing to the advancement of next-generation Embodied AI.

</details>

---

## 152. A Unified Framework and Dataset for Assessing Societal Bias in Vision-Language Models

- [ ] A Unified Framework and Dataset for Assessing Societal Bias in Vision-Language Models | https://aclanthology.org/2024.findings-emnlp.66/

- **Link**: https://aclanthology.org/2024.findings-emnlp.66/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) have gained widespread adoption in both industry and academia. In this study, we propose a unified framework for systematically evaluating gender, race, and age biases in VLMs with respect to professions. Our evaluation encompasses all supported inference modes of the recent VLMs, including image-to-text, text-to-text, text-to-image, and image-to-image. We create a synthetic, high-quality dataset comprising text and images that intentionally obscure gender, race, and age distinctions across various professions. The dataset includes action-based descriptions of each profession and serves as a benchmark for evaluating societal biases in vision-language models (VLMs). In our benchmarking of popular vision-language models (VLMs), we observe that different input-output modalities result in distinct bias magnitudes and directions. We hope our work will help guide future progress in improving VLMs to learn socially unbiased representations. We will release our data and code.

</details>

---

## 153. HSDreport: Heart Sound Diagnosis with Echocardiography Reports

- [ ] HSDreport: Heart Sound Diagnosis with Echocardiography Reports | https://aclanthology.org/2024.findings-emnlp.664/

- **Link**: https://aclanthology.org/2024.findings-emnlp.664/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Heart sound auscultation holds significant importance in the diagnosis of congenital heart disease. However, existing methods for Heart Sound Diagnosis (HSD) tasks are predominantly limited to a few fixed categories, framing the HSD task as a rigid classification problem that does not fully align with medical practice and offers only limited information to physicians. Besides, such methods do not utilize echocardiography reports, the gold standard in the diagnosis of related diseases. To tackle this challenge, we introduce HSDreport, a new benchmark for HSD, which mandates the direct utilization of heart sounds obtained from auscultation to predict echocardiography reports. This benchmark aims to merge the convenience of auscultation with the comprehensive nature of echocardiography reports. First, we collect a new dataset for this benchmark, comprising 2,275 heart sound samples along with their corresponding reports. Subsequently, we develop a knowledge-aware query-based transformer to handle this task. The intent is to leverage the capabilities of medically pre-trained models and the internal knowledge of large language models (LLMs) to address the task’s inherent complexity and variability, thereby enhancing the robustness and scientific validity of the method. Furthermore, our experimental results indicate that our method significantly outperforms traditional HSD approaches and existing multimodal LLMs in detecting key abnormalities in heart sounds.

</details>

---

## 154. VGA: VisionGUIAssistant - Minimizing Hallucinations through Image-Centric Fine-Tuning

- [ ] VGA: VisionGUIAssistant - Minimizing Hallucinations through Image-Centric Fine-Tuning | https://aclanthology.org/2024.findings-emnlp.68/

- **Link**: https://aclanthology.org/2024.findings-emnlp.68/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (VLMs) have already been applied to the understanding of Graphical User Interfaces (GUIs) and have achieved notable results. However, existing VLMs often overly rely on internal text-based knowledge while neglecting visual inputs. This imbalance may lead models to produce answers that do not align with the visual content in GUI comprehension tasks. Such inaccuracies are termed as ‘hallucinations’ where models generate incorrect or illogical responses upon visual verification against GUI elements. These errors result in misinterpretations and diminish the model’s practical utility in applied settings. To address these issues, we introduce VGA, a fine-tuned model designed for comprehensive GUI understanding. Our model aims to balance attention image and text to enhance interpretation and reduce hallucinations. We construct a Vision Question Answering (VQA) dataset of 63.8k high-quality examples with our propose *Referent Method*, focusing on response with visual content of images. We then design a two-stage fine-tuning method to enhance both the model’s accuracy to extract information from image content and alignment with human intent. Experiments show that our approach enhances the model’s ability to extract information from images and achieves state-of-the-art results in GUI understanding tasks. https://github.com/Linziyang1999/VGA-visual-GUI-assistant

</details>

---

## 155. ChartInsights: Evaluating Multimodal Large Language Models for Low-Level Chart Question Answering

- [ ] ChartInsights: Evaluating Multimodal Large Language Models for Low-Level Chart Question Answering | https://aclanthology.org/2024.findings-emnlp.710/

- **Link**: https://aclanthology.org/2024.findings-emnlp.710/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Chart question answering (ChartQA) tasks play a critical role in interpreting and extracting insights from visualization charts. While recent advancements in multimodal large language models (MLLMs) like GPT-4o have shown promise in high-level ChartQA tasks, such as chart captioning, their effectiveness in low-level ChartQA tasks (*e.g.*, identifying correlations) remains underexplored.In this paper, we address this gap by evaluating MLLMs on low-level ChartQA using a newly curated dataset, *ChartInsights*, which consists of 22,347 (chart, task, query, answer) covering 10 data analysis tasks across 7 chart types. We systematically evaluate 19 advanced MLLMs, including 12 open-source and 7 closed-source models. The average accuracy rate across these models is 39.8%, with GPT-4o achieving the highest accuracy at 69.17%.To further explore the limitations of MLLMs in low-level ChartQA, we conduct experiments that alter visual elements of charts (*e.g.*, changing color schemes, adding image noise) to assess their impact on the task effectiveness. Furthermore, we propose a new textual prompt strategy, *Chain-of-Charts*, tailored for low-level ChartQA tasks, which boosts performance by 14.41%, achieving an accuracy of 83.58%. Finally, incorporating a visual prompt strategy that directs attention to relevant visual elements further improves accuracy to 84.32%.

</details>

---

## 156. Large Language Models Are Challenged by Habitat-Centered Reasoning

- [ ] Large Language Models Are Challenged by Habitat-Centered Reasoning | https://aclanthology.org/2024.findings-emnlp.763/

- **Link**: https://aclanthology.org/2024.findings-emnlp.763/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper we perform a novel in-depth evaluation of text-only and multimodal LLMs’ abilities to reason about object *habitats* or conditions on how objects are situated in their environments that affect the types of behaviors (or *affordances*) that can be enacted upon them. We present a novel curated multimodal dataset of questions about object habitats and affordances, which are formally grounded in the underlying lexical semantics literature, with multiple images from various sources that depict the scenario described in the question. We evaluate 16 text-only and multimodal LLMs on this challenging data. Our findings indicate that while certain LLMs can perform reasonably well on reasoning about affordances, there appears to be a consistent low upper bound on habitat-centered reasoning performance. We discuss how the formal semantics of habitats in fact predicts this behavior and propose this as a challenge to the community.

</details>

---

## 157. V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization

- [ ] V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization | https://aclanthology.org/2024.findings-emnlp.775/

- **Link**: https://aclanthology.org/2024.findings-emnlp.775/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) suffer from hallucination, resulting in misalignment between the output textual response and the input visual content. Recent research indicates that the over-reliance on the Large Language Model (LLM) backbone, as one cause of the LVLM hallucination, inherently introduces bias from language priors, leading to insufficient context attention to the visual inputs.We tackle this issue of hallucination by mitigating such over-reliance through preference learning. We propose Vision-guided Direct Preference Optimization (V-DPO) to enhance visual context learning at training time. To interpret the effectiveness and generalizability of V-DPO on different types of training data, we construct a synthetic dataset containing both response- and image-contrast preference pairs, compared against existing human-annotated hallucination samples. Our approach achieves significant improvements compared with baseline methods across various hallucination benchmarks. Our analysis indicates that V-DPO excels in learning from image-contrast preference data, demonstrating its superior ability to elicit and understand nuances of visual context. Our code is publicly available at https://github.com/YuxiXie/V-DPOhttps://github.com/YuxiXie/V-DPO.

</details>

---

## 158. Exploring the Potential of MultimodalLLMwith Knowledge-Intensive MultimodalASR

- [ ] Exploring the Potential of MultimodalLLMwith Knowledge-Intensive MultimodalASR | https://aclanthology.org/2024.findings-emnlp.776/

- **Link**: https://aclanthology.org/2024.findings-emnlp.776/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal large language models (MLLMs) have made significant progress in integrating information across various modalities, yet real-world applications in educational and scientific domains remain challenging. This paper introduces the Multimodal Scientific ASR (MS-ASR) task, which focuses on transcribing scientific conference videos by leveraging visual information from slides to enhance the accuracy of technical terminologies. Realized that traditional metrics like WER fall short in assessing performance accurately, prompting the proposal of severity-aware WER (SWER) that considers the content type and severity of ASR errors. We propose the Scientific Vision Augmented ASR (SciVASR) framework as a baseline method, enabling MLLMs to improve transcript quality through post-editing. Evaluations of state-of-the-art MLLMs, including GPT-4o, show a 45% improvement over speech-only baselines, highlighting the importance of multimodal information integration.

</details>

---

## 159. Why doLLaVAVision-Language Models Reply to Images inEnglish?

- [ ] Why doLLaVAVision-Language Models Reply to Images inEnglish? | https://aclanthology.org/2024.findings-emnlp.783/

- **Link**: https://aclanthology.org/2024.findings-emnlp.783/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We uncover a surprising multilingual bias occurring in a popular class of multimodal vision-language models (VLMs). Including an image in the query to a LLaVA-style VLM significantly increases the likelihood of the model returning an English response, regardless of the language of the query. This paper investigates the causes of this loss with a two-pronged approach that combines extensive ablation of the design space with a mechanistic analysis of the models’ internal representations of image and text inputs. Both approaches indicate that the issue stems in the language modeling component of the LLaVA model. Statistically, we find that switching the language backbone for a bilingual language model has the strongest effect on reducing this error. Mechanistically, we provide compelling evidence that visual inputs are not mapped to a similar space as text ones, and that intervening on intermediary attention layers can reduce this bias. Our findings provide important insights to researchers and engineers seeking to understand the crossover between multimodal and multilingual spaces, and contribute to the goal of developing capable and inclusive VLMs for non-English contexts.

</details>

---

## 160. Multilingual Synopses of Movie Narratives: A Dataset for Vision-Language Story Understanding

- [ ] Multilingual Synopses of Movie Narratives: A Dataset for Vision-Language Story Understanding | https://aclanthology.org/2024.findings-emnlp.788/

- **Link**: https://aclanthology.org/2024.findings-emnlp.788/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Story video-text alignment, a core task in computational story understanding, aims to align video clips with corresponding sentences in their descriptions. However, progress on the task has been held back by the scarcity of manually annotated video-text correspondence and the heavy concentration on English narrations of Hollywood movies. To address these issues, in this paper, we construct a large-scale multilingual video story dataset named Multilingual Synopses of Movie Narratives (M-SyMoN), containing 13,166 movie summary videos from 7 languages, as well as manual annotation of fine-grained video-text correspondences for 101.5 hours of video. Training on the human annotated data from SyMoN outperforms the SOTA methods by 15.7 and 16.2 percentage points on Clip Accuracy and Sentence IoU scores, respectively, demonstrating the effectiveness of the annotations. As benchmarks for future research, we create 6 baseline approaches with different multilingual training strategies, compare their performance in both intra-lingual and cross-lingual setups, exemplifying the challenges of multilingual video-text alignment. The dataset is released at:https://github.com/insundaycathy/M-SyMoN

</details>

---

## 161. MVP-Bench: Can Large Vision-Language Models Conduct Multi-level Visual Perception Like Humans?

- [ ] MVP-Bench: Can Large Vision-Language Models Conduct Multi-level Visual Perception Like Humans? | https://aclanthology.org/2024.findings-emnlp.789/

- **Link**: https://aclanthology.org/2024.findings-emnlp.789/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Humans perform visual perception at multiple levels, including low-level object recognition and high-level semantic interpretation such as behavior understanding. Subtle differences in low-level details can lead to substantial changes in high-level perception. For example, substituting the shopping bag held by a person with a gun suggests violent behavior, implying criminal or violent activity. Despite significant advancements in various multimodal tasks, Large Visual Language Models (LVLMs) remain unexplored in their capabilities to conduct such multi-level visual perceptions.To investigate the perception gap between LVLMs and humans, we introduce MVP-Bench, the first visual–language benchmark systematically evaluating both low- and high-level visual perception of LVLMs. We construct MVP-Bench across natural and synthetic images to investigate how manipulated content influences model perception. Using MVP-Bench, we diagnose the visual perception of 10 open-source and 2 closed-source LVLMs, showing that high-level perception tasks significantly challenge existing LVLMs. The state-of-the-art GPT-4o only achieves an accuracy of 56% on Yes/No questions, compared with 74% in low-level scenarios. Furthermore, the performance gap between natural and manipulated images indicates that current LVLMs do not generalize in understanding the visual semantics of synthetic images as humans do.

</details>

---

## 162. MultiSkill: Evaluating Large Multimodal Models for Fine-grained Alignment Skills

- [ ] MultiSkill: Evaluating Large Multimodal Models for Fine-grained Alignment Skills | https://aclanthology.org/2024.findings-emnlp.81/

- **Link**: https://aclanthology.org/2024.findings-emnlp.81/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We propose MultiSkill, an evaluation protocol that assesses large multimodal models (LMMs) across multiple fine-grained skills for alignment with human values. Recent LMMs have shown various intriguing abilities, such as solving graph theory problems and explaining visual jokes. However, existing multimodal benchmarks have mainly focused on coarse-grained evaluation (e.g., accuracy), without considering the skill composition required by specific instructions. To this end, we present MultiSkill, designed to decompose coarse-level scoring to a fine-grained skill set-level scoring tailored to each instruction. MultiSkill defines five core vision-language capabilities and divides into 12 skills that are necessary to align with user instructions. For evaluation metrics on specific skills, we propose an LMM-based evaluator for open-ended outputs. Based on the diverse instructions collected from 66 datasets spanning 10 domains, we compare multiple representative open-source and proprietary LMMs and find a high correlation between model-based and human-based evaluations. Our experiments underscore the importance of fine-grained evaluation in providing a holistic view of model performance and enhancing the reliability of the evaluation.

</details>

---

## 163. Query-based Cross-Modal Projector Bolstering Mamba MultimodalLLM

- [ ] Query-based Cross-Modal Projector Bolstering Mamba MultimodalLLM | https://aclanthology.org/2024.findings-emnlp.827/

- **Link**: https://aclanthology.org/2024.findings-emnlp.827/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The Transformer’s quadratic complexity with input length imposes an unsustainable computational load on large language models (LLMs). In contrast, the Selective Scan Structured State-Space Model, or Mamba, addresses this computational challenge effectively. This paper explores a query-based cross-modal projector designed to bolster Mamba’s efficiency for vision-language modeling by compressing visual tokens based on input through the cross-attention mechanism. This innovative projector also removes the need for manually designing the 2D scan order of original image features when converting them into an input sequence for Mamba LLM. Experimental results across various vision-language understanding benchmarks show that the proposed cross-modal projector enhances Mamba-based multimodal LLMs, boosting both performance and throughput.

</details>

---

## 164. Semantic Token Reweighting for Interpretable and Controllable Text Embeddings inCLIP

- [ ] Semantic Token Reweighting for Interpretable and Controllable Text Embeddings inCLIP | https://aclanthology.org/2024.findings-emnlp.837/

- **Link**: https://aclanthology.org/2024.findings-emnlp.837/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

A text encoder within Vision-Language Models (VLMs) like CLIP plays a crucial role in translating textual input into an embedding space shared with images, thereby facilitating the interpretative analysis of vision tasks through natural language. Despite the varying significance of different textual elements within a sentence depending on the context, efforts to account for variation of importance in constructing text embeddings have been lacking. We propose a framework of Semantic Token Reweighting to build Interpretable text embeddings (SToRI), which incorporates controllability as well. SToRI refines the text encoding process in CLIP by differentially weighting semantic elements based on contextual importance, enabling finer control over emphasis responsive to data-driven insights and user preferences. The efficacy of SToRI is demonstrated through comprehensive experiments on few-shot image classification and image retrieval tailored to user preferences.

</details>

---

## 165. MATE: Meet At The Embedding - Connecting Images with Long Texts

- [ ] MATE: Meet At The Embedding - Connecting Images with Long Texts | https://aclanthology.org/2024.findings-emnlp.90/

- **Link**: https://aclanthology.org/2024.findings-emnlp.90/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

While advancements in Vision Language Models (VLMs) have significantly improved the alignment of visual and textual data, these models primarily focus on aligning images with short descriptive captions. This focus limits their ability to handle complex text interactions, particularly with longer texts such as lengthy captions or documents, which have not been extensively explored yet. In this paper, we introduce Meet At The Embedding (MATE), a novel approach that combines the capabilities of VLMs with Large Language Models (LLMs) to overcome this challenge without the need for additional image-long text pairs. Specifically, we replace the text encoder of the VLM with a pretrained LLM-based encoder that excels in understanding long texts. To bridge the gap between VLM and LLM, MATE incorporates a projection module that is trained in a multi-stage manner. It starts by aligning the embeddings from the VLM text encoder with those from the LLM using extensive text pairs. This module is then employed to seamlessly align image embeddings closely with LLM embeddings. We propose two new cross-modal retrieval benchmarks to assess the task of connecting images with long texts (lengthy captions / documents). Extensive experimental results demonstrate that MATE effectively connects images with long texts, uncovering diverse semantic relationships.

</details>

---

## 166. Advancing Vision-Language Models with Adapter Ensemble Strategies

- [ ] Advancing Vision-Language Models with Adapter Ensemble Strategies | https://aclanthology.org/2024.findings-emnlp.921/

- **Link**: https://aclanthology.org/2024.findings-emnlp.921/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

CLIP revolutes vision-language pretraining by using contrastive learning on paired web data. However, the sheer size of these pretrained models makes full-model finetuning exceedingly costly. One common solution is the “adapter”, which finetunes a few additional parameters while freezing the backbone. It harnesses the heavy-duty backbone while offering a light finetuning for small downstream tasks. This synergy prompts us to explore the potential of augmenting large-scale backbones with traditional machine learning techniques. Often employed in traditional fields and overlooked in the large-scale era, these techniques could provide valuable enhancements. Herein, we delve into the “adapter ensembles” in the realm of large-scale pretrained vision-language models. We begin with a proof-of-concept study to establish the efficacy of combining multiple adapters. We then present extensive evidence showing these ensembles excel in a variety of settings, particularly when employing a Multi-Scale Attention (MSA) approach thoughtfully integrated into the ensemble framework. We further incorporate the LoRA to mitigate the additional parameter burden. We focus on vision-language retrieval, using different backbones under constraints of minimal data, parameters, and finetuning budgets. This research paves the way for a synergistic blend of traditional, yet effective, strategies with modern large-scale networks.

</details>

---

## 167. Grounding Partially-Defined Events in Multimodal Data

- [ ] Grounding Partially-Defined Events in Multimodal Data | https://aclanthology.org/2024.findings-emnlp.934/

- **Link**: https://aclanthology.org/2024.findings-emnlp.934/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

How are we able to learn about complex current events just from short snippets of video? While natural language enables straightforward ways to represent under-specified, partially observable events, visual data does not facilitate analogous methods and, consequently, introduces unique challenges in event understanding. With the growing prevalence of vision-capable AI agents, these systems must be able to model events from collections of unstructured video data. To tackle robust event modeling in multimodal settings, we introduce a multimodal formulation for partially-defined events and cast the extraction of these events as a three-stage span retrieval task. We propose a corresponding benchmark for this task, MultiVENT-G, that consists of 14.5 hours of densely annotated current event videos and 1,168 text documents, containing 22.8K labeled event-centric entities. We propose a collection of LLM-driven approaches to the task of multimodal event analysis, and evaluate them on MultiVENT-G. Results illustrate the challenges that abstract event understanding poses and demonstrates promise in event-centric video-language systems.

</details>

---

## 168. MiRAGeNews: Multimodal RealisticAI-Generated News Detection

- [ ] MiRAGeNews: Multimodal RealisticAI-Generated News Detection | https://aclanthology.org/2024.findings-emnlp.959/

- **Link**: https://aclanthology.org/2024.findings-emnlp.959/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The proliferation of inflammatory or misleading “fake” news content has become increasingly common in recent years. Simultaneously, it has become easier than ever to use AI tools to generate photorealistic images depicting any scene imaginable. Combining these two—AI-generated fake news content—is particularly potent and dangerous. To combat the spread of AI-generated fake news, we propose the MiRAGeNews Dataset, a dataset of 12,500 high-quality real and AI-generated image-caption pairs from state-of-the-art generators. We find that our dataset poses a significant challenge to humans (60% F-1) and state-of-the-art multi-modal LLMs (< 24% F-1). Using our dataset we train a multi-modal detector (MiRAGe) that improves by +5.1% F-1 over state-of-the-art baselines on image-caption pairs from out-of-domain image generators and news publishers. We release our code and data to aid future work on detecting AI-generated content.

</details>

---

## 169. Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective

- [ ] Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective | https://aclanthology.org/2024.findings-emnlp.960/

- **Link**: https://aclanthology.org/2024.findings-emnlp.960/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Language Models (LLMs) have facilitated the development of Multimodal LLMs (MLLMs). Despite their impressive capabilities, MLLMs often suffer from over-reliance on unimodal biases (e.g., language bias and vision bias), leading to incorrect answers in complex multimodal tasks. To investigate this issue, we propose a causal framework to interpret the biases in Visual Question Answering (VQA) problems. Within this framework, we conduct an in-depth causal analysis to assess the causal effect of these biases on MLLM predictions. Based on the analysis, we introduce 1) a novel MORE dataset with 12,000 challenging VQA instances requiring multi-hop reasoning and overcoming unimodal biases. 2) a causality-enhanced agent framework CAVE that guides models to comprehensively integrate information from different modalities and mitigate biases. Our experiments show that MLLMs perform poorly on MORE, indicating strong unimodal biases and limited semantic understanding. However, when integrated with our CAVE, promising improvements in reasoning and bias mitigation can be seen. These findings provide important insights for the development of more robust MLLMs and contribute to the broader goal of advancing multimodal AI systems capable of deeper understanding and reasoning. Our project page is at https://github.com/OpenCausaLab/MORE.

</details>

---

## 170. TransferCVLM: Transferring Cross-Modal Knowledge for Vision-Language Modeling

- [ ] TransferCVLM: Transferring Cross-Modal Knowledge for Vision-Language Modeling | https://aclanthology.org/2024.findings-emnlp.975/

- **Link**: https://aclanthology.org/2024.findings-emnlp.975/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Recent large vision-language multimodal models pre-trained with huge amount of image-text pairs show remarkable performances in downstream tasks. However, the multimodal pre-training has limitations in terms of resources and training time when it comes to obtaining new models that surpass existing models. To overcome these issues, we propose TransferCVLM, a method of efficient knowledge transfer that integrates pre-trained uni-modal models (and cross-modal fusion-encoder) into a combined vision-language model (CVLM), without pre-training the CVLM with large amount of multimodal data, and then for each task application, fine-tunes the CVLM and transfers the multimodal knowledge of a teacher vision-language model to the CVLM by using knowledge distillation techniques. We demonstrate that 1) the fine-tuned CVLM performs comparable to other vision-language models of similar size, that 2) the multimodal knowledge transfer consistently enhances the CVLM, and the knowledge-transferred CVLM composed of large-size unimodal models outperforms the teacher multimodal model in most of downstream tasks, and that 3) TransferCVLM can also be used for model compression when using small-size unimodal models. We estimate that the training of TransferCVLM takes only 6% of pre-training of other vision-language models. Our code is available at https://github.com/DMCB-GIST/TransferCVLM.

</details>

---

## 171. Unraveling the Truth: DoVLMs really Understand Charts? A Deep Dive into Consistency and Robustness

- [ ] Unraveling the Truth: DoVLMs really Understand Charts? A Deep Dive into Consistency and Robustness | https://aclanthology.org/2024.findings-emnlp.973/

- **Link**: https://aclanthology.org/2024.findings-emnlp.973/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Chart question answering (CQA) is a crucial area of Visual Language Understanding. However, the robustness and consistency of current Visual Language Models (VLMs) in this field remain under-explored. This paper evaluates state-of-the-art VLMs on comprehensive datasets, developed specifically for this study, encompassing diverse question categories and chart formats. We investigate two key aspects: 1) the models’ ability to handle varying levels of chart and question complexity, and 2) their robustness across different visual representations of the same underlying data. Our analysis reveals significant performance variations based on question and chart types, highlighting both strengths and weaknesses of current models. Additionally, we identify areas for improvement and propose future research directions to build more robust and reliable CQA systems. This study sheds light on the limitations of current models and paves the way for future advancements in the field.

</details>

---

## 172. Personalized Video Comment Generation

- [ ] Personalized Video Comment Generation | https://aclanthology.org/2024.findings-emnlp.979/

- **Link**: https://aclanthology.org/2024.findings-emnlp.979/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Generating personalized responses, particularly in the context of video, poses a unique challenge for language models. This paper introduces the novel task ofPersonalized Video Comment Generation(PVCG), aiming to predict user comments tailored to both the input video and the user’s comment history, where the user is unseen during the model training process. Unlike existing video captioning tasks that ignores the personalization in the text generation process, we introduce PerVidCom, a new dataset specifically collected for this novel task with diverse personalized comments from YouTube. Recognizing the limitations of existing captioning metrics for evaluating this task, we propose a new automatic metric based on Large Language Models (LLMs) with few-shot in-context learning, named FICL-Score, specifically measuring quality from the aspects of emotion, language style and content relevance. We verify the proposed metric with human evaluations. We establish baselines using prominent Multimodal LLMs (MLLMs), analyze their performance discrepancies through extensive evaluation, and identifies directions for future improvement on this important task. Our research opens up a new direction of personalizing MLLMs and paves the way for future research.

</details>

---

## 173. Improving Adversarial Robustness in Vision-Language Models with Architecture and Prompt Design

- [ ] Improving Adversarial Robustness in Vision-Language Models with Architecture and Prompt Design | https://aclanthology.org/2024.findings-emnlp.990/

- **Link**: https://aclanthology.org/2024.findings-emnlp.990/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have seen a significant increase in both research interest and real-world applications across various domains, including healthcare, autonomous systems, and security. However, their growing prevalence demands higher reliability and safety including robustness to adversarial attacks. We systematically examine the possibility of incorporating adversarial robustness through various model design choices. We explore the effects of different vision encoders, the resolutions of vision encoders, and the size and type of language models. Additionally, we introduce novel, cost-effective approaches to enhance robustness through prompt engineering. By simply suggesting the possibility of adversarial perturbations or rephrasing questions, we demonstrate substantial improvements in model robustness against strong image-based attacks such as Auto-PGD. Our findings provide important guidelines for developing more robust VLMs, particularly for deployment in safety-critical environments where reliability and security are paramount. These insights are crucial for advancing the field of VLMs, ensuring they can be safely and effectively utilized in a wide range of applications.

</details>

---

## 174. Reference-Based Metrics Are Biased Against Blind and Low-Vision Users’ Image Description Preferences

- [ ] Reference-Based Metrics Are Biased Against Blind and Low-Vision Users’ Image Description Preferences | https://aclanthology.org/2024.nlp4pi-1.26/

- **Link**: https://aclanthology.org/2024.nlp4pi-1.26/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image description generation models are sophisticated Vision-Language Models which promise to make visual content, such as images, non-visually accessible through linguistic descriptions. While these systems can benefit all, their primary motivation tends to lie in allowing blind and low-vision (BLV) users access to increasingly visual (online) discourse. Well-defined evaluation methods are crucial for steering model development into socially useful directions. In this work, we show that the most popular evaluation metrics (reference-based metrics) are biased against BLV users and therefore potentially stifle useful model development. Reference-based metrics assign quality scores based on the similarity to human-generated ground-truth descriptions and are widely accepted as neutrally representing the needs of all users. However, we find that these metrics are more strongly correlated with sighted participant ratings than BLV ratings, and we explore factors which appear to mediate this finding: description length, the image’s context of appearance, and the number of reference descriptions available. These findings suggest that there is a need for developing evaluation methods that are established based on specific downstream user groups, and they highlight the importance of reflecting on emerging biases against minorities in the development of general-purpose automatic metrics.

</details>

---

## 175. ARMADA: Attribute-Based Multimodal Data Augmentation

- [ ] ARMADA: Attribute-Based Multimodal Data Augmentation | https://aclanthology.org/2024.wikinlp-1.17/

- **Link**: https://aclanthology.org/2024.wikinlp-1.17/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In Multimodal Language Models (MLMs), the cost of manually annotating high-quality image-text pair data for fine-tuning and alignment is extremely high. While existing multimodal data augmentation frameworks propose ways to augment image-text pairs, they either suffer from semantic inconsistency between texts and images, or generate unrealistic images, causing knowledge gap with real world examples. To address these issues, we propose Attribute-based Multimodal Data Augmentation (ARMADA), a novel multimodal data augmentation method via knowledge-guided manipulation of visual attributes of the mentioned entities. Specifically, we extract entities and their visual attributes from the original text data, then search for alternative values for the visual attributes under the guidance of knowledge bases (KBs) and large language models (LLMs). We then utilize an image-editing model to edit the images with the extracted attributes. ARMADA is a novel multimodal data generation framework that: (i) extracts knowledge-grounded attributes from symbolic KBs for semantically consistent yet distinctive image-text pair generation, (ii) generates visually similar images of disparate categories using neighboring entities in the KB hierarchy, and (iii) uses the commonsense knowledge of LLMs to modulate auxiliary visual attributes such as backgrounds for more robust representation of original entities. Our empirical results over four downstream tasks demonstrate the efficacy of our framework to produce high-quality data and enhance the model performance. This also highlights the need to leverage external knowledge proxies for enhanced interpretability and real-world grounding.

</details>

---

## 176. Benchmarking Visually-Situated Translation of Text in Natural Images

- [ ] Benchmarking Visually-Situated Translation of Text in Natural Images | https://aclanthology.org/2024.wmt-1.115/

- **Link**: https://aclanthology.org/2024.wmt-1.115/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

We introduce a benchmark, Vistra, for visually-situated translation of English text in natural images to four target languages. We describe the dataset construction and composition. We benchmark open-source and commercial OCR and MT models on Vistra, and present both quantitative results and a taxonomy of common OCR error classes with their effect on downstream MT. Finally, we assess direct image-to-text translation with a multimodal LLM, and show that it is able in some cases but not yet consistently to disambiguate possible translations with visual context. We show that this is an unsolved and challenging task even for strong commercial models. We hope that the creation and release of this benchmark which is the first of its kind for these language pairs will encourage further research in this direction.

</details>

---

## 177. Brotherhood atWMT2024: LeveragingLLM-Generated Contextual Conversations for Cross-Lingual Image Captioning

- [ ] Brotherhood atWMT2024: LeveragingLLM-Generated Contextual Conversations for Cross-Lingual Image Captioning | https://aclanthology.org/2024.wmt-1.81/

- **Link**: https://aclanthology.org/2024.wmt-1.81/

- **Conference**: EMNLP

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we describe our system under the team name Brotherhood for the English-to-Lowres Multi-Modal Translation Task. We participate in the multi-modal translation tasks for English-Hindi, English-Hausa, English-Bengali, and English-Malayalam language pairs. We present a method leveraging multi-modal Large Language Models (LLMs), specifically GPT-4o and Claude 3.5 Sonnet, to enhance cross-lingual image captioning without traditional training or fine-tuning.Our approach utilizes instruction-tuned prompting to generate rich, contextual conversations about cropped images, using their English captions as additional context. These synthetic conversations are then translated into the target languages. Finally, we employ a weighted prompting strategy, balancing the original English caption with the translated conversation to generate captions in the target language.This method achieved competitive results, scoring 37.90 BLEU on the English-Hindi Challenge Set and ranking first and second for English-Hausa on the Challenge and Evaluation Leaderboards, respectively. We conduct additional experiments on a subset of 250 images, exploring the trade-offs between BLEU scores and semantic similarity across various weighting schemes.

</details>

---

