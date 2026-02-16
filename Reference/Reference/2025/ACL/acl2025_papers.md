# ACL 2025 Papers

> ☐ 勾选论文后，可用脚本导出 selected_acl2025_papers.csv

## 1. FORG3D: Flexible Object Rendering for Generating Vision-Language Spatial Reasoning Data from 3DScenes

- [ ] FORG3D: Flexible Object Rendering for Generating Vision-Language Spatial Reasoning Data from 3DScenes | https://aclanthology.org/2025.acl-demo.36/

- **Link**: https://aclanthology.org/2025.acl-demo.36/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce FORG3D, a 3D rendering toolkit developed with Blender and Python, which synthesizes vision-language data for two primary purposes: (1) supporting human cognitive experiments that require fine-grained control over material and (2) analyzing and improving the visual reasoning capabilities of large vision-language models. The toolkit provides flexible and precise control over object placement, orientation, inter-object distances, and camera configurations while automatically generating detailed spatial metadata. Additionally, it includes a built-in feature for integrating AI-generated backgrounds, enhancing the realism of synthetic scenes. FORG3D is publicly available at https://github.com/compling-wat/FORG3D, and a video demonstration is available at https://www.youtube.com/watch?v=QvIqib_PU8A.

</details>

---

## 2. FlagEval-Arena: A Side-by-Side Comparative Evaluation Platform for Large Language Models and Text-DrivenAIGC

- [ ] FlagEval-Arena: A Side-by-Side Comparative Evaluation Platform for Large Language Models and Text-DrivenAIGC | https://aclanthology.org/2025.acl-demo.56/

- **Link**: https://aclanthology.org/2025.acl-demo.56/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce FlagEval-Arena, an evaluation platform for side-by-side comparisons of large language models and text-driven AIGC systems.Compared with the well-known LM Arena (LMSYS Chatbot Arena), we reimplement our own framework with the flexibility to introduce new mechanisms or features. Our platform enables side-by-side evaluation not only for language models or vision-language models, but also text-to-image or text-to-video synthesis. We specifically target at Chinese audience with a more focus on the Chinese language, more models developed by Chinese institutes, and more general usage beyond the technical community. As a result, we currently observe very interesting differences from usual results presented by LM Arena. Our platform is available via this URL:https://flageval.baai.org/#/arena.

</details>

---

## 3. FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation

- [ ] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation | https://aclanthology.org/2025.acl-demo.6/

- **Link**: https://aclanthology.org/2025.acl-demo.6/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible at https://github.com/flageval-baai/FlagEvalMM, with a demonstration video available at https://youtu.be/L7EtacjoM0k.

</details>

---

## 4. MIRA: Empowering One-TouchAIServices on Smartphones withMLLM-based Instruction Recommendation

- [ ] MIRA: Empowering One-TouchAIServices on Smartphones withMLLM-based Instruction Recommendation | https://aclanthology.org/2025.acl-industry.103/

- **Link**: https://aclanthology.org/2025.acl-industry.103/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of generative AI technologies is driving the integration of diverse AI-powered services into smartphones, transforming how users interact with their devices. To simplify access to predefined AI services, this paper introduces MIRA, a pioneering framework for task instruction recommendation that enables intuitive one-touch AI tasking on smartphones. With MIRA, users can long-press on images or text objects to receive contextually relevant instruction recommendations for executing AI tasks. Our work introduces three key innovations: 1) A multimodal large language model (MLLM)-based recommendation pipeline with structured reasoning to extract key entities, infer user intent, and generate precise instructions; 2) A template-augmented reasoning mechanism that integrates high-level reasoning templates, enhancing task inference accuracy; 3) A prefix-tree-based constrained decoding strategy that restricts outputs to predefined instruction candidates, ensuring coherence and intent alignment. Through evaluation using a real-world annotated datasets and a user study, MIRA has demonstrated substantial improvements in recommendation accuracy. The encouraging results highlight MIRA’s potential to revolutionize the way users engage with AI services on their smartphones, offering a more seamless and efficient experience.

</details>

---

## 5. EcoDoc: A Cost-Efficient Multimodal Document Processing System for Enterprises UsingLLMs

- [ ] EcoDoc: A Cost-Efficient Multimodal Document Processing System for Enterprises UsingLLMs | https://aclanthology.org/2025.acl-industry.109/

- **Link**: https://aclanthology.org/2025.acl-industry.109/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Enterprises are increasingly adopting Generative AI applications to extract insights from large volumes of multimodal documents in domains such as finance, law, healthcare, and industry. These documents contain structured and unstructured data (images, charts, handwritten texts, etc.) requiring robust AI systems for effective retrieval and comprehension. Recent advancements in Retrieval-Augmented Generation (RAG) frameworks and Vision-Language Models (VLMs) have improved retrieval performance on multimodal documents by processing pages as images. However, large-scale deployment remains challenging due to the high cost of LLM API usage and the slower inference speed of image-based processing of pages compared to text-based processing. To address these challenges, we propose EcoDoc, a cost-effective multimodal document processing system that dynamically selects the processing modalities for each page as an image or text based on page characteristics and query intent. Our experimental evaluation on TAT-DQA and DocVQA benchmarks shows that EcoDoc reduces average query processing latency by up to 2.29×and cost by up to 10×, without compromising accuracy.

</details>

---

## 6. Arctic-TILT. Business Document Understanding at Sub-Billion Scale

- [ ] Arctic-TILT. Business Document Understanding at Sub-Billion Scale | https://aclanthology.org/2025.acl-industry.20/

- **Link**: https://aclanthology.org/2025.acl-industry.20/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The vast portion of workloads employing LLMs involves answering questions grounded on PDF or scanned content. We introduce the Arctic-TILT achieving accuracy on par with models 1000×its size on these use cases. It can be finetuned and deployed on a single 24GB GPU, lowering operational costs while processing rich documents with up to 400k tokens. The model establishes state-of-the-art results on seven diverse Document Understanding benchmarks, as well as provides reliable confidence scores and quick inference, essential for processing files in large-scale or time-sensitive enterprise environments. We release Arctic-TILT weights and an efficient vLLM-based implementation on a permissive license.

</details>

---

## 7. LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences

- [ ] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences | https://aclanthology.org/2025.acl-industry.22/

- **Link**: https://aclanthology.org/2025.acl-industry.22/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have transformed image captioning, shifting from concise captions to detailed descriptions. We introduce LOTUS, a leaderboard for evaluating detailed captions, addressing three main gaps in existing evaluations: lack of standardized criteria, bias-aware assessments, and user preference considerations. LOTUS comprehensively evaluates various aspects, including caption quality (e.g., alignment, descriptiveness), risks (e.g., hallucination), and societal biases (e.g., gender bias) while enabling preference-oriented evaluations by tailoring criteria to diverse user preferences. Our analysis of recent LVLMs reveals no single model excels across all criteria, while correlations emerge between caption detail and bias risks. Preference-oriented evaluations demonstrate that optimal model selection depends on user priorities.

</details>

---

## 8. LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions

- [ ] LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions | https://aclanthology.org/2025.acl-industry.29/

- **Link**: https://aclanthology.org/2025.acl-industry.29/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Anomaly Detection (AD) focuses on detecting samples that differ from the standard pattern, making it a vital tool in process control. Logical anomalies may appear visually normal yet violate predefined constraints on object presence, arrangement, or quantity, depending on reasoning and explainability. We introduce LogicQA, a framework that enhances AD by providing industrial operators with explanations for logical anomalies. LogicQA compiles automatically generated questions into a checklist and collects responses to identify violations of logical constraints. LogicQA is training-free, annotation-free, and operates in a few-shot setting. We achieve state-of-the-art (SOTA) Logical AD performance on public benchmarks, MVTec LOCO AD, with an AUROC of 87.6% and anF1-max of 87.0% along with the explanations of anomalies. Also, our approach has shown outstanding performance on semiconductor SEM corporate data, further validating its effectiveness in industrial applications.

</details>

---

## 9. RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning

- [ ] RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning | https://aclanthology.org/2025.acl-industry.3/

- **Link**: https://aclanthology.org/2025.acl-industry.3/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Advertisement (Ad) video violation detection is critical for ensuring platform compliance, but existing methods struggle with precise temporal grounding, noisy annotations, and limited generalization. We propose RAVEN, a novel framework that integrates curriculum reinforcement learning with multimodal large language models (MLLMs) to enhance reasoning and cognitive capabilities for violation detection. RAVEN employs a progressive training strategy, combining precisely and coarsely annotated data, and leverages Group Relative Policy Optimization (GRPO) to develop emergent reasoning abilities without explicit reasoning annotations. Multiple hierarchical sophisticated reward mechanism ensures precise temporal grounding and consistent category prediction. Experiments on industrial datasets and public benchmarks show that RAVEN achieves superior performances in violation category accuracy and temporal interval localization. We also design a pipeline to deploy the RAVEN on the online Ad services, and online A/B testing further validates its practical applicability, with significant improvements in precision and recall. RAVEN also demonstrates strong generalization, mitigating the catastrophic forgetting issue associated with supervised fine-tuning.

</details>

---

## 10. Filter-And-Refine: AMLLMBased Cascade System for Industrial-Scale Video Content Moderation

- [ ] Filter-And-Refine: AMLLMBased Cascade System for Industrial-Scale Video Content Moderation | https://aclanthology.org/2025.acl-industry.62/

- **Link**: https://aclanthology.org/2025.acl-industry.62/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Effective content moderation is essential for video platforms to safeguard user experience and uphold community standards. While traditional video classification models effectively handle well-defined moderation tasks, they struggle with complicated scenarios such as implicit harmful content and contextual ambiguity. Multimodal large language models (MLLMs) offer a promising solution to these limitations with their superior cross-modal reasoning and contextual understanding. However, two key challenges hinder their industrial adoption. First, the high computational cost of MLLMs makes full-scale deployment impractical. Second, adapting generative models for discriminative classification remains an open research problem. In this paper, we first introduce an efficient method to transform a generative MLLM into a multimodal classifier using minimal discriminative training data. To enable industry-scale deployment, we then propose a router-ranking cascade system that integrates MLLMs with a lightweight router model. Offline experiments demonstrate that our MLLM-based approach improves F1 score by 66.50% over traditional classifiers while requiring only 2% of the fine-tuning data. Online evaluations show that our system increases automatic content moderation volume by 41%, while the cascading deployment reduces computational cost to only 1.5% of direct full-scale deployment.

</details>

---

## 11. MathAgent: Leveraging a Mixture-of-Math-Agent Framework for Real-World Multimodal Mathematical Error Detection

- [ ] MathAgent: Leveraging a Mixture-of-Math-Agent Framework for Real-World Multimodal Mathematical Error Detection | https://aclanthology.org/2025.acl-industry.7/

- **Link**: https://aclanthology.org/2025.acl-industry.7/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mathematical error detection in educational settings presents a significant challenge for Multimodal Large Language Models (MLLMs), requiring a sophisticated understanding of both visual and textual mathematical content along with complex reasoning capabilities. Though effective in mathematical problem-solving, MLLMs often struggle with the nuanced task of **identifying and categorizing student errors in multimodal mathematical contexts**. Therefore, we introduce **MathAgent, a novel Mixture-of-Math-Agent framework** specifically designed to address these challenges. Our approach decomposes error detection into three phases with specialized agents: an image-text consistency validator, a visual semantic interpreter, and an integrative error analyzer. This architecture enables more accurate processing of multimodal mathematical content by explicitly modeling the relationships between multimodal problems and student solution steps. We evaluate MathAgent on real-world educational data, demonstrating approximately 5% higher accuracy in error step identification and 3% improvement in error categorization compared to baseline models. Furthermore, MathAgent has been successfully deployed in an educational platform serving over one million K-12 students, achieving nearly 90% student satisfaction while generating significant cost savings by reducing manual error detection.

</details>

---

## 12. sudo rm -rf agentic_security

- [ ] sudo rm -rf agentic_security | https://aclanthology.org/2025.acl-industry.75/

- **Link**: https://aclanthology.org/2025.acl-industry.75/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2tox Offense), a novel attack framework that systematically bypasses refusal-trained safeguards in commercial computer-use agents, such as Claude for Computer Use. The core mechanism, Detox2tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24.41% (with no refinement), and up to 41.33% (by its iterative refinement) in Claude for Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs.

</details>

---

## 13. Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?

- [ ] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning? | https://aclanthology.org/2025.acl-industry.83/

- **Link**: https://aclanthology.org/2025.acl-industry.83/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Charts are ubiquitous as they help people understand and reason with data. Recently, various downstream tasks, such as chart question answering, chart2text, and fact-checking, have emerged. Large Vision-Language Models (LVLMs) show promise in tackling these tasks, but their evaluation is costly and time-consuming, limiting real-world deployment. While using LVLMs as judges to assess chart comprehension capabilities of other LVLMs could streamline evaluation processes, challenges like proprietary datasets, restricted access to powerful models, and evaluation costs hinder their adoption in industrial settings. To this end, we present a comprehensive evaluation of 13 open-source LVLMs as judges for diverse chart comprehension and reasoning tasks. We design both pairwise and pointwise evaluation tasks covering criteria like factual correctness, informativeness, and relevancy. Additionally, we analyze LVLM judges based on format adherence, positional consistency, length bias, and instruction-following. We focus on cost-effective LVLMs (<10B parameters) suitable for both research and commercial use, following a standardized evaluation protocol and rubric to measure the LVLM judge accuracy. Experimental results reveal notable variability: while some open LVLM judges achieve GPT-4-level evaluation performance (about 80% agreement with GPT-4 judgments), others struggle (below ~10% agreement). Our findings highlight that state-of-the-art open-source LVLMs can serve as cost-effective automatic evaluators for chart-related tasks, though biases such as positional preference and length bias persist.

</details>

---

## 14. Deep Temporal Reasoning in Video Language Models: A Cross-Linguistic Evaluation of Action Duration and Completion through Perfect Times

- [ ] Deep Temporal Reasoning in Video Language Models: A Cross-Linguistic Evaluation of Action Duration and Completion through Perfect Times | https://aclanthology.org/2025.acl-long.1000/

- **Link**: https://aclanthology.org/2025.acl-long.1000/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human perception of events is intrinsically tied to distinguishing between completed (perfect and telic) and ongoing (durative) actions, a process mediated by both linguistic structure and visual cues. In this work, we introduce the Perfect Times dataset, a novel, quadrilingual (English, Italian, Russian, and Japanese) multiple-choice question-answering benchmark designed to assess video-language models (VLMs) on temporal reasoning. By pairing everyday activity videos with event completion labels and perfectivity-tailored distractors, our dataset probes whether models truly comprehend temporal dynamics or merely latch onto superficial markers. Experimental results indicate that state-of-the-art models, despite their success on text-based tasks, struggle to mirror human-like temporal and causal reasoning grounded in video. This study underscores the necessity of integrating deep multimodal cues to capture the nuances of action duration and completion within temporal and causal video dynamics, setting a new standard for evaluating and advancing temporal reasoning in VLMs.

</details>

---

## 15. AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations

- [ ] AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations | https://aclanthology.org/2025.acl-long.1008/

- **Link**: https://aclanthology.org/2025.acl-long.1008/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

State-of-the-art multimodal web agents, powered by Multimodal Large Language Models (MLLMs), can autonomously execute many web tasks by processing user instructions and interacting with graphical user interfaces (GUIs). Current strategies for building web agents rely on (i) the generalizability of underlying MLLMs and their steerability via prompting, and (ii) large-scale fine-tuning of MLLMs on web-related tasks. However, web agents still struggle to automate tasks on unseen websites and domains, limiting their applicability to enterprise-specific and proprietary platforms. Beyond generalization from large-scale pre-training and fine-tuning, we propose building agents for few-shot adaptability using human demonstrations. We introduce the AdaptAgent framework that enables both proprietary and open-weights multimodal web agents to adapt to new websites and domains using few human demonstrations (up to 2). Our experiments on two popular benchmarks — Mind2Web & VisualWebArena — show that using in-context demonstrations (for proprietary models) or meta-adaptation demonstrations (for meta-learned open-weights models) boosts task success rate by 3.36% to 7.21% over non-adapted state-of-the-art models, corresponding to a relative increase of 21.03% to 65.75%. Furthermore, our additional analyses (a) show the effectiveness of multimodal demonstrations over text-only ones, (b) illuminate how different meta-learning data selection strategies influence the agent’s generalization, and (c) demonstrate how the number of few-shot examples affects the web agent’s success rate. Our results offer a complementary axis for developing widely applicable multimodal web agents beyond large-scale pre-training and fine-tuning, emphasizing few-shot adaptability.

</details>

---

## 16. VF-Eval: Evaluating MultimodalLLMs for Generating Feedback onAIGCVideos

- [ ] VF-Eval: Evaluating MultimodalLLMs for Generating Feedback onAIGCVideos | https://aclanthology.org/2025.acl-long.1027/

- **Link**: https://aclanthology.org/2025.acl-long.1027/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, multimodal large language models (MLLMs) have been extensively explored in video question answering. However, most existing assessments focus on natural videos, overlooking synthetic videos (e.g., AI-generated content). Meanwhile, some works in video generation rely on MLLMs to evaluate the quality of generated videos, but the capabilities of MLLMs on AIGC videos remain largely underexplored. To address this, we propose a new benchmark, VQ-Eval, which introduces four tasks—coherence validation, error awareness, error type detection, and reasoning evaluation—to comprehensively evaluate the abilities of MLLMs on AIGC videos. We evaluate 13 frontier MLLMs on VQ-Eval and find that even the best-performing model, GPT-4.1, struggles to achieve consistently good performance across all tasks. This highlights the challenging nature of our benchmark. Additionally, to investigate the practical applications of VQ-Eval in improving video generation, we design a re-prompt pipeline, demonstrating that aligning MLLMs more closely with human feedback can benefit the video generation.

</details>

---

## 17. CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships?

- [ ] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships? | https://aclanthology.org/2025.acl-long.1033/

- **Link**: https://aclanthology.org/2025.acl-long.1033/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are renowned for their superior instruction-following and reasoning capabilities across diverse problem domains. However, existing benchmarks primarily focus on assessing factual and logical correctness in downstream tasks, with limited emphasis on evaluating MLLMs’ ability to interpret pragmatic cues and intermodal relationships. To address this gap, we assess the competency of MLLMs in performing Multimodal Discourse Analysis (MDA) using Coherence Relations. Our benchmark, CORDIAL, encompasses a broad spectrum of Coherence Relations across 3 different discourse domains at varying levels of granularity. Through our experiments on 10+ MLLMs employing different prompting strategies, we show that even top models like Gemini 1.5 Pro and GPT-4o fail to match the performance of simple classifier-based baselines. This study emphasizes the need to move beyond similarity-based metrics and adopt a discourse-driven framework for evaluating MLLMs, providing a more nuanced assessment of their capabilities. The benchmark and code are available at: https://aashish2000.github.io/CORDIAL/.

</details>

---

## 18. World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning

- [ ] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning | https://aclanthology.org/2025.acl-long.1044/

- **Link**: https://aclanthology.org/2025.acl-long.1044/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in large vision-language models (LVLMs) have shown promise for embodied task planning, yet they struggle with fundamental challenges like dependency constraints and efficiency. Existing approaches either solely optimize action selection or directly leverage pre-trained models as world models during inference, overlooking the benefits of learning to model the world as a way to enhance planning capabilities. We propose Dual Preference Optimization (D2PO), a new learning framework that jointly optimizes state prediction and action selection through preference learning, enabling LVLMs to understand environment dynamics for better planning. To automatically collect trajectories and stepwise preference data without human annotation, we introduce a tree search mechanism for extensive exploration via trial-and-error. Extensive experiments on VoTa-Bench demonstrate that our D2PO-based method significantly outperforms existing methods and GPT-4o when applied to Qwen2-VL (7B), LLaVA-1.6 (7B), and LLaMA-3.2 (11B), achieving superior task success rates with more efficient execution paths.

</details>

---

## 19. VisuoThink: EmpoweringLVLMReasoning with Multimodal Tree Search

- [ ] VisuoThink: EmpoweringLVLMReasoning with Multimodal Tree Search | https://aclanthology.org/2025.acl-long.1053/

- **Link**: https://aclanthology.org/2025.acl-long.1053/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision-Language Models have showcased remarkable capabilities. However, they often falter when confronted with complex reasoning tasks that humans typically address through visual aids and deliberate, step-by-step thinking. While existing methods have explored text-based slow thinking or rudimentary visual assistance, they fall short of capturing the intricate, interleaved nature of human visual-verbal reasoning processes. To overcome these limitations and inspired by the mechanisms of slow thinking in human cognition, we introduce VisuoThink, a novel framework that seamlessly integrates visuospatial and linguistic domains. VisuoThink facilitates multimodal slow thinking by enabling progressive visual-textual reasoning and incorporates test-time scaling through look-ahead tree search. Extensive experiments demonstrate that VisuoThink significantly enhances reasoning capabilities via inference-time scaling, even without fine-tuning, achieving state-of-the-art performance in tasks involving geometry and spatial reasoning.

</details>

---

## 20. AutomatedCADModeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models

- [ ] AutomatedCADModeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models | https://aclanthology.org/2025.acl-long.1054/

- **Link**: https://aclanthology.org/2025.acl-long.1054/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Designing complex computer-aided design (CAD) models is often time-consuming due to challenges such as computational inefficiency and the difficulty of generating precise models. We propose a novel language-guided framework for industrial design automation to address these issues, integrating large language models (LLMs) with computer-automated design (CAutoD).Through this framework, CAD models are automatically generated from parameters and appearance descriptions, supporting the automation of design tasks during the detailed CAD design phase. Our approach introduces three key innovations: (1) a semi-automated data annotation pipeline that leverages LLMs and vision-language large models (VLLMs) to generate high-quality parameters and appearance descriptions; (2) a Transformer-based CAD generator (TCADGen) that predicts modeling sequences via dual-channel feature aggregation; (3) an enhanced CAD modeling generation model, called CADLLM, that is designed to refine the generated sequences by incorporating the confidence scores from TCADGen. Experimental results demonstrate that the proposed approach outperforms traditional methods in both accuracy and efficiency, providing a powerful tool for automating industrial workflows and generating complex CAD models from textual prompts.The code is available at https://jianxliao.github.io/cadllm-page/

</details>

---

## 21. Knowledge Image Matters: Improving Knowledge-Based Visual Reasoning with Multi-Image Large Language Models

- [ ] Knowledge Image Matters: Improving Knowledge-Based Visual Reasoning with Multi-Image Large Language Models | https://aclanthology.org/2025.acl-long.1063/

- **Link**: https://aclanthology.org/2025.acl-long.1063/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We revisit knowledge-based visual reasoning (KB-VR) in light of modern advances in multimodal large language models (MLLMs), and make the following contributions: (i) We propose Visual Knowledge Card (VKC) – a novel image that incorporates not only internal visual knowledge (e.g., scene-aware information) detected from the raw image, but also external world knowledge (e.g., attribute or object knowledge) produced by a knowledge generator; (ii) We present VKC-based Multi-Image Reasoning (VKC-MIR) – a four-stage pipeline which harnesses a state-of-the-art scene perception engine to construct an initial VKC (Stage-1), a powerful LLM to generate relevant domain knowledge (Stage-2), an excellent image editing toolkit to introduce generated knowledge into the updated VKC (Stage-3), and finally, an emerging multi-image MLLM to solve the VKC-enhanced task (Stage-4). By performing experiments on three popular KB-VR benchmarks, our approach achieves new state-of-the-art results compared to previous top-performing models.

</details>

---

## 22. GUICourse: From General Vision Language Model to VersatileGUIAgent

- [ ] GUICourse: From General Vision Language Model to VersatileGUIAgent | https://aclanthology.org/2025.acl-long.1065/

- **Link**: https://aclanthology.org/2025.acl-long.1065/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Utilizing Graphic User Interfaces (GUIs) for human-computer interaction is essential for accessing various digital tools. Recent advancements in Vision Language Models (VLMs) reveal significant potential for developing versatile agents that assist humans in navigating GUIs. However, current VLMs face challenges related to fundamental abilities, such as OCR and grounding, as well as a lack of knowledge about GUI elements functionalities and control methods. These limitations hinder their effectiveness as practical GUI agents. To address these challenges, we introduce GUICourse, a series of datasets for training visual-based GUI agents using general VLMs. First, we enhance the OCR and grounding capabilities of VLMs using the GUIEnv dataset. Next, we enrich the GUI knowledge of VLMs using the GUIAct and GUIChat datasets. Our experiments demonstrate that even a small-sized GUI agent (with 3.1 billion parameters) performs effectively on both single-step and multi-step GUI tasks. We further finetune our GUI agents on other GUI tasks with different action spaces (AITW and Mind2Web), and the results show that our agents are better than their baseline VLMs. Additionally, we analyze the impact of OCR and grounding capabilities through an ablation study, revealing a positive correlation with GUI navigation ability.

</details>

---

## 23. Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLMCollaboration

- [ ] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLMCollaboration | https://aclanthology.org/2025.acl-long.1066/

- **Link**: https://aclanthology.org/2025.acl-long.1066/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

To create culturally inclusive vision-language models (VLMs), developing a benchmark that tests their ability to address culturally relevant questions is essential. Existing approaches typically rely on human annotators, making the process labor-intensive and creating a cognitive burden in generating diverse questions. To address this, we propose a semi-automated framework for constructing cultural VLM benchmarks, specifically targeting multiple-choice QA. This framework combines human-VLM collaboration, where VLMs generate questions based on guidelines, a small set of annotated examples, and relevant knowledge, followed by a verification process by native speakers. We demonstrate the effectiveness of this framework through the creation of K-Viscuit, a dataset focused on Korean culture. Our experiments on this dataset reveal that open-source models lag behind proprietary ones in understanding Korean culture, highlighting key areas for improvement. We also present a series of further analyses, including human evaluation, augmenting VLMs with external knowledge, and the evaluation beyond multiple-choice QA. Our dataset is available at https://huggingface.co/datasets/ddehun/k-viscuit.

</details>

---

## 24. Caution for the Environment: MultimodalLLMAgents are Susceptible to Environmental Distractions

- [ ] Caution for the Environment: MultimodalLLMAgents are Susceptible to Environmental Distractions | https://aclanthology.org/2025.acl-long.1087/

- **Link**: https://aclanthology.org/2025.acl-long.1087/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper investigates the faithfulness of multimodal large language model (MLLM) agents in a graphical user interface (GUI) environment, aiming to address the research question of whether multimodal GUI agents can be distracted by environmental context. A general scenario is proposed where both the user and the agent are benign, and the environment, while not malicious, contains unrelated content. A wide range of MLLMs are evaluated as GUI agents using a simulated dataset, following three working patterns with different levels of perception. Experimental results reveal that even the most powerful models, whether generalist agents or specialist GUI agents, are susceptible to distractions. While recent studies predominantly focus on the helpfulness of agents, our findings first indicate that these agents are prone to environmental distractions. Furthermore, we implement an adversarial environment injection and analyze the approach to improve faithfulness, calling for a collective focus on this important topic.

</details>

---

## 25. Automatic Evaluation for Text-to-image Generation: Task-decomposed Framework, Distilled Training, and Meta-evaluation Benchmark

- [ ] Automatic Evaluation for Text-to-image Generation: Task-decomposed Framework, Distilled Training, and Meta-evaluation Benchmark | https://aclanthology.org/2025.acl-long.1088/

- **Link**: https://aclanthology.org/2025.acl-long.1088/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Driven by the remarkable progress in diffusion models, text-to-image generation has achieved substantial advancements, underscoring the urgent need for robust automatic quality assessment. This task is inherently complex, requiring evaluations that range from object presence and attribute correctness to relational consistency and visual fidelity. Consequently, current state-of-the-art MLLM-based approaches often rely on powerful commercial models such as GPT-4o, which offer superior reasoning and instruction-following capabilities but are not universally accessible. In contrast, while open-source MLLMs demonstrate promising skills in vision and language understanding, they underperform in comprehensive image quality assessment.To address these challenges, we propose a task decomposition evaluation framework based on GPT-4o to automatically construct a specialized training dataset, breaking down the multifaceted evaluation process into simpler sub-tasks and thus reducing learning complexity. Building on this dataset, we design novel training strategies to distill GPT-4o’s evaluation capabilities into a7Bopen-source MLLM, MiniCPM-V-2.6, enabling it to better follow instructions across diverse assessment criteria. Furthermore, to reliably and comprehensively assess prior works and our proposed model, we manually annotate a meta-evaluation benchmark that includes chain-of-thought explanations alongside quality scores for generated images.Experimental results demonstrate that our distilled open-source MLLM significantly outperforms the current state-of-the-art GPT-4o-base baseline, VIEScore, with over 4.6% improvement in Spearman and Kendall correlations with human judgments.

</details>

---

## 26. ChartLens: Fine-grained Visual Attribution in Charts

- [ ] ChartLens: Fine-grained Visual Attribution in Charts | https://aclanthology.org/2025.acl-long.1094/

- **Link**: https://aclanthology.org/2025.acl-long.1094/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The growing capabilities of multimodal large language models (MLLMs) have advanced tasks like chart understanding. However, these models often suffer from hallucinations, where generated text sequences conflict with the provided visual data. To address this, we introduce Post-Hoc Visual Attribution for Charts, which identifies fine-grained chart elements that validate a given chart-associated response. We propose ChartLens, a novel chart attribution algorithm that uses segmentation-based techniques to identify chart objects and employs set-of-marks prompting with MLLMs for fine-grained visual attribution. Additionally, we present ChartVA-Eval, a benchmark with synthetic and real-world charts from diverse domains like finance, policy, and economics, featuring fine-grained attribution annotations. Our evaluations show that ChartLens improves fine-grained attributions by 26-66%.

</details>

---

## 27. MMRC: A Large-Scale Benchmark for Understanding Multimodal Large Language Model in Real-World Conversation

- [ ] MMRC: A Large-Scale Benchmark for Understanding Multimodal Large Language Model in Real-World Conversation | https://aclanthology.org/2025.acl-long.1096/

- **Link**: https://aclanthology.org/2025.acl-long.1096/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent multimodal large language models (MLLMs) have demonstrated significant potential in open-ended conversation, generating more accurate and personalized responses. However, their abilities to memorize, recall, and reason in sustained interactions within real-world scenarios remain underexplored. This paper introduces MMRC, a Multi-Modal Real-world Conversation benchmark for evaluating six core open-ended abilities of MLLMs: information extraction, multi-turn reasoning, information update, image management, memory recall, and answer refusal. With data collected from real-world scenarios, MMRC comprises 5,120 conversations and 28,720 corresponding manually labeled questions, posing a significant challenge to existing MLLMs. Evaluations on 20 MLLMs in MMRC indicate an accuracy drop during open-ended interactions. We identify four common failure patterns: long-term memory degradation, inadequacies in updating factual knowledge, accumulated assumption of error propagation, and reluctance to “say no.” To mitigate these issues, we propose a simple yet effective NOTE-TAKING strategy, which can record key information from the conversation and remind the model during its responses, enhancing conversational capabilities. Experiments across six MLLMs demonstrate significant performance improvements.

</details>

---

## 28. Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues

- [ ] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues | https://aclanthology.org/2025.acl-long.112/

- **Link**: https://aclanthology.org/2025.acl-long.112/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Nonverbal communication is integral to human interaction, with gestures, facial expressions, and body language conveying critical aspects of intent and emotion. However, existing large language models (LLMs) fail to effectively incorporate these nonverbal elements, limiting their capacity to create fully immersive conversational experiences. We introduce MARS, a multimodal language model designed to understand and generate nonverbal cues alongside text, bridging this gap in conversational AI.Our key innovation is VENUS, a large-scale dataset comprising annotated videos with time-aligned text, facial expressions, and body language.Leveraging VENUS, we train MARS with a next-token prediction objective, combining text with vector-quantized nonverbal representations to achieve multimodal understanding and generation within a unified framework.Based on various analyses of the VENUS datasets, we validate its substantial scale and high effectiveness. Our quantitative and qualitative results demonstrate that MARS successfully generates text and nonverbal languages, corresponding to conversational input.Our dataset and code are available at https://github.com/winston1214/nonverbal-conversation.

</details>

---

## 29. Does the Emotional Understanding ofLVLMs Vary Under High-Stress Environments and Across Different Demographic Attributes?

- [ ] Does the Emotional Understanding ofLVLMs Vary Under High-Stress Environments and Across Different Demographic Attributes? | https://aclanthology.org/2025.acl-long.1130/

- **Link**: https://aclanthology.org/2025.acl-long.1130/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

According to psychological and neuroscientific research, a high-stress environment can restrict attentional resources and intensify negative affect, thereby impairing the ability to understand emotions. Furthermore, demographic attributes such as race, gender, and age group have been repeatedly reported to cause significant differences in emotional expression and recognition. This study is the first to systematically verify whether these psychological findings observed in humans also apply to the latest Large Vision Language Models (LVLMs). We constructed low-stress versus high-stress environments and generated an image dataset (a total of 540 images) that combines race, gender, and age group. Based on this, we applied the Pretend prompt technique to induce LVLMs to interpret others’ emotions from the standpoint of the assigned environment and persona. An analysis of the models’ emotional understanding ability, using EQ-Bench-based metrics, revealed that (1) under high-stress environments, the accuracy of emotion understanding significantly declined in most LVLMs, and (2) performance disparities were confirmed across race, gender, and age group. These findings suggest that the effects of high-stress and demographic attributes identified in human research may also be reflected in LVLMs.

</details>

---

## 30. FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning

- [ ] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning | https://aclanthology.org/2025.acl-long.1138/

- **Link**: https://aclanthology.org/2025.acl-long.1138/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Real-world decision-making often requires integrating and reasoning over information from multiple modalities. While recent multimodal large language models (MLLMs) have shown promise in such tasks, their ability to perform multi-hop reasoning across diverse sources remains insufficiently evaluated. Existing benchmarks, such as MMQA, face challenges due to (1) data contamination and (2) a lack of complex queries that necessitate operations across more than two modalities, hindering accurate performance assessment. To address this, we present Financial Cross-Modal Multi-Hop Reasoning (FCMR), a benchmark created to analyze the reasoning capabilities of MLLMs by urging them to combine information from textual reports, tables, and charts within the financial domain. FCMR is categorized into three difficulty levels—Easy, Medium, and Hard—facilitating a step-by-step evaluation. In particular, problems at the Hard level require precise cross-modal three-hop reasoning and are designed to prevent the disregard of any modality. Experiments on this new benchmark reveal that even state-of-the-art MLLMs struggle, with the best-performing model (Claude 3.5 Sonnet) achieving only 30.4% accuracy on the most challenging tier. We also conduct analysis to provide insights into the inner workings of the models, including the discovery of a critical bottleneck in the information retrieval phase.

</details>

---

## 31. Finding Needles in Images: Can Multi-modalLLMs Locate Fine Details?

- [ ] Finding Needles in Images: Can Multi-modalLLMs Locate Fine Details? | https://aclanthology.org/2025.acl-long.1152/

- **Link**: https://aclanthology.org/2025.acl-long.1152/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Multi-modal Large Language Models (MLLMs) have shown impressive capabilities in document understanding tasks, their ability to locate and reason about fine-grained details within complex documents remains understudied. Consider searching a restaurant menu for a specific nutritional detail or identifying a disclaimer in a lengthy newspaper article — tasks that demand careful attention to small but significant details within a broader narrative, akin to Finding Needles in Images (NiM). To address this gap, we introduce NiM-Benchmark, a carefully curated benchmark spanning diverse real-world documents including newspapers, menus, and lecture images, specifically designed to evaluate MLLMs’ capability in these intricate tasks. Building on this, we further propose Spot-IT, a simple yet effective approach that enhances MLLMs capability through intelligent patch selection and Gaussian attention, motivated from how humans zoom and focus when searching documents. Our extensive experiments reveal both the capabilities and limitations of current MLLMs in handling fine-grained document understanding tasks, while demonstrating the effectiveness of our approach. Spot-IT achieves significant improvements over baseline methods, particularly in scenarios requiring precise detail extraction from complex layouts.

</details>

---

## 32. Inference Compute-Optimal Video Vision Language Models

- [ ] Inference Compute-Optimal Video Vision Language Models | https://aclanthology.org/2025.acl-long.117/

- **Link**: https://aclanthology.org/2025.acl-long.117/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This work investigates the optimal allocation of inference compute across three key scaling factors in video vision language models: language model size, frame count, and the number of visual tokens per frame. While prior works typically focuses on optimizing model efficiency or improving performance without considering resource constraints, we instead identify optimal model configuration under fixed inference compute budgets. We conduct large-scale training sweeps and careful parametric modeling of task performance to identify the inference compute-optimal frontier. Our experiments reveal how task performance depends on scaling factors and finetuning data size, as well as how changes in data size shift the compute-optimal frontier. These findings translate to practical tips for selecting these scaling factors.

</details>

---

## 33. Asclepius: A Spectrum Evaluation Benchmark for Medical Multi-Modal Large Language Models

- [ ] Asclepius: A Spectrum Evaluation Benchmark for Medical Multi-Modal Large Language Models | https://aclanthology.org/2025.acl-long.1178/

- **Link**: https://aclanthology.org/2025.acl-long.1178/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The significant breakthroughs of Medical Multi-Modal Large Language Models (Med-MLLMs) renovate modern healthcare with robust information synthesis and medical decision support. However, these models are often evaluated on benchmarks that are unsuitable for the Med-MLLMs due to the intricate nature of the real-world diagnostic frameworks, which encompass diverse medical specialties and involve complex clinical decisions. Thus, a clinically representative benchmark is highly desirable for credible Med-MLLMs evaluation. To this end, we introduce Asclepius, a novel Med-MLLM benchmark that comprehensively assesses Med-MLLMs in terms of: distinct medical specialties (cardiovascular, gastroenterology, etc.) and different diagnostic capacities (perception, disease analysis, etc.). Grounded in 3 proposed core principles, Asclepius ensures a comprehensive evaluation by encompassing 15 medical specialties, stratifying into 3 main categories and 8 sub-categories of clinical tasks, and exempting overlap with the existing VQA dataset. We further provide an in-depth analysis of 6 Med-MLLMs and compare them with 3 human specialists, providing insights into their competencies and limitations in various medical contexts. Our work not only advances the understanding of Med-MLLMs’ capabilities but also sets a precedent for future evaluations and the safe deployment of these models in clinical environments.

</details>

---

## 34. InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning

- [ ] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning | https://aclanthology.org/2025.acl-long.1179/

- **Link**: https://aclanthology.org/2025.acl-long.1179/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal foundation models, particularly in the domains of language and vision, have significantly advanced various tasks, including robotics, autonomous driving, information retrieval, and grounding. However, many of these models perceive objects as indivisible, overlooking the components that constitute them. Understanding these components and their associated affordances provides valuable insights into an object’s functionality, which is fundamental for performing a wide range of tasks. In this work, we introduce a novel real-world benchmark, InstructPart, comprising hand-labeled part segmentation annotations and task-oriented instructions to evaluate the performance of current models in understanding and executing part-level tasks within everyday contexts. Through our experiments, we demonstrate that task-oriented part segmentation remains a challenging problem, even for state-of-the-art Vision-Language Models (VLMs). In addition to our benchmark, we introduce a simple baseline that achieves a twofold performance improvement through fine-tuning with our dataset. With our dataset and benchmark, we aim to facilitate research on task-oriented part segmentation and enhance the applicability of VLMs across various domains, including robotics, virtual reality, information retrieval, and other related fields. Project website: https://zifuwan.github.io/InstructPart/.

</details>

---

## 35. Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models

- [ ] Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models | https://aclanthology.org/2025.acl-long.118/

- **Link**: https://aclanthology.org/2025.acl-long.118/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Aligned representations across languages is a desired property in multilingual large language models (mLLMs), as alignment can improve performance in cross-lingual tasks. Typically alignment requires fine-tuning a model, which is computationally expensive, and sizable language data, which often may not be available. A data-efficient alternative to fine-tuning is model interventions — a method for manipulating model activations to steer generation into the desired direction. We analyze the effect of a popular intervention (finding experts) on the alignment of cross-lingual representations in mLLMs. We identify the neurons to manipulate for a given language and introspect the embedding space of mLLMs pre- and post-manipulation. We show that modifying the mLLM’s activations changes its embedding space such that cross-lingual alignment is enhanced. Further, we show that the changes to the embedding space translate into improved downstream performance on retrieval tasks, with up to 2x improvements in top-1 accuracy on cross-lingual retrieval.

</details>

---

## 36. OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval

- [ ] OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval | https://aclanthology.org/2025.acl-long.1198/

- **Link**: https://aclanthology.org/2025.acl-long.1198/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language retrieval-augmented generation (RAG) has become an effective approach for tackling Knowledge-Based Visual Question Answering (KB-VQA), which requires external knowledge beyond the visual content presented in images. The effectiveness of Vision-language RAG systems hinges on multimodal retrieval, which is inherently challenging due to the diverse modalities and knowledge granularities in both queries and knowledge bases. Existing methods have not fully tapped into the potential interplay between these elements. We propose a multimodal RAG system featuring a coarse-to-fine, multi-step retrieval that harmonizes multiple granularities and modalities to enhance efficacy. Our system begins with a broad initial search aligning knowledge granularity for cross-modal retrieval, followed by a multimodal fusion reranking to capture the nuanced multimodal information for top entity selection. A text reranker then filters out the most relevant fine-grained section for augmented generation. Extensive experiments on the InfoSeek and Encyclopedic-VQA benchmarks show our method achieves state-of-the-art retrieval performance and highly competitive answering results, underscoring its effectiveness in advancing KB-VQA systems. Our code can be found at https://github.com/ChaoLinAViy/OMGM.

</details>

---

## 37. Retrospective Learning from Interactions

- [ ] Retrospective Learning from Interactions | https://aclanthology.org/2025.acl-long.1200/

- **Link**: https://aclanthology.org/2025.acl-long.1200/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-turn interactions between large language models (LLMs) and users naturally include implicit feedback signals. If an LLM responds in an unexpected way to an instruction, the user is likely to signal it by rephrasing the request, expressing frustration, or pivoting to an alternative task. Such signals are task-independent and occupy a relatively constrained subspace of language, allowing the LLM to identify them even if it fails on the actual task. We introduce ReSpect, a method to learn from such signals in past interactions via retrospection without additional annotations. We deploy ReSpect in a new multimodal interaction scenario, where humans instruct a multimodal LLM to solve an abstract reasoning task with a combinatorial solution space. Through thousands of interactions with humans, we show how ReSpect gradually improves task completion rate from 31% to 82%, all without any external annotation.

</details>

---

## 38. SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings

- [ ] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings | https://aclanthology.org/2025.acl-long.1212/

- **Link**: https://aclanthology.org/2025.acl-long.1212/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have serious security vulnerabilities. While safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM’s security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at https://github.com/ZeroNLP/SEA.

</details>

---

## 39. Mind the Gesture: EvaluatingAISensitivity to Culturally Offensive Non-Verbal Gestures

- [ ] Mind the Gesture: EvaluatingAISensitivity to Culturally Offensive Non-Verbal Gestures | https://aclanthology.org/2025.acl-long.1218/

- **Link**: https://aclanthology.org/2025.acl-long.1218/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Gestures are an integral part of non-verbal communication, with meanings that vary across cultures, and misinterpretations that can have serious social and diplomatic consequences. As AI systems become more integrated into global applications, ensuring they do not inadvertently perpetuate cultural offenses is critical. To this end, we introduce Multi-Cultural Set of Inappropriate Gestures and Nonverbal Signs (MC-SIGNS), a dataset of 288 gesture-country pairs annotated for offensiveness, cultural significance, and contextual factors across 25 gestures and 85 countries. Through systematic evaluation using MC-SIGNS, we uncover critical limitations: text-to-image (T2I) systems exhibit strong US-centric biases, performing better at detecting offensive gestures in US contexts than in non-US ones; large language models (LLMs) tend to over-flag gestures as offensive; and vision-language models (VLMs) default to US-based interpretations when responding to universal concepts like wishing someone luck, frequently suggesting culturally inappropriate gestures. These findings highlight the urgent need for culturally-aware AI safety mechanisms to ensure equitable global deployment of AI technologies.

</details>

---

## 40. Response Wide Shut? Surprising Observations in Basic Vision Language Model Capabilities

- [ ] Response Wide Shut? Surprising Observations in Basic Vision Language Model Capabilities | https://aclanthology.org/2025.acl-long.1241/

- **Link**: https://aclanthology.org/2025.acl-long.1241/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language Models (VLMs) have emerged as general-purpose tools for addressing a variety of complex computer vision problems. Such models have been shown to be highly capable, but, at the same time, lacking some basic visual understanding skills. In this paper, we set out to understand the limitations of SoTA VLMs on fundamental visual tasks (object classification, spatial understanding, and ability to delineate individual object instances through counting), by constructing a series of tests that probe which components of design, specifically, may be lacking. Importantly, we go significantly beyond the current benchmarks, which simply measure the final performance of VLM response, by also comparing and contrasting it to the performance of probes trained directly on features obtained from the visual encoder, intermediate vision-language projection and LLM-decoder output. In doing so, we uncover shortcomings in VLMs and make a number of important observations about their capabilities, robustness and how they process visual information. We hope our insights will guide progress in further improving VLMs.

</details>

---

## 41. EffiVLM-BENCH: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models

- [ ] EffiVLM-BENCH: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models | https://aclanthology.org/2025.acl-long.1242/

- **Link**: https://aclanthology.org/2025.acl-long.1242/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have achieved remarkable success, yet their significant computational demands hinder practicaldeployment. While efforts to improve LVLM efficiency are growing, existing methods lack comprehensive evaluation across diverse backbones, benchmarks, and metrics. In this work, we systematically evaluate mainstream acceleration techniques for LVLMs, categorized into token and parameter compression. We introduce EffiVLM-BENCH, a unified framework for assessing not only absolute performance but also generalization and loyalty, while exploring Pareto-optimal trade-offs. Our extensive experiments and in-depth analyses offer insights into optimal strategies for accelerating LVLMs. We open-source code and recipes for EffiVLM-BENCH to foster future research.

</details>

---

## 42. Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users

- [ ] Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users | https://aclanthology.org/2025.acl-long.1260/

- **Link**: https://aclanthology.org/2025.acl-long.1260/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper explores the effectiveness of Multimodal Large Language models (MLLMs) as assistive technologies for visually impaired individuals. We conduct a user survey to identify adoption patterns and key challenges users face with such technologies. Despite a high adoption rate of these models, our findings highlight concerns related to contextual understanding, cultural sensitivity, and complex scene understanding, particularly for individuals who may rely solely on them for visual interpretation. Informed by these results, we collate five user-centred tasks with image and video inputs, including a novel task on Optical Braille Recognition. Our systematic evaluation of twelve MLLMs reveals that further advancements are necessary to overcome limitations related to cultural context, multilingual support, Braille reading comprehension, assistive object recognition, and hallucinations. This work provides critical insights into the future direction of multimodal AI for accessibility, underscoring the need for more inclusive, robust, and trustworthy visual assistance technologies.

</details>

---

## 43. RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection

- [ ] RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection | https://aclanthology.org/2025.acl-long.1279/

- **Link**: https://aclanthology.org/2025.acl-long.1279/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have demonstrated remarkable capabilities in various domains, including radiology report generation. Previous approaches have attempted to utilize multimodal LLMs for this task, enhancing their performance through the integration of domain-specific knowledge retrieval. However, these approaches often overlook the knowledge already embedded within the LLMs, leading to redundant information integration. To address this limitation, we propose Radar, a framework for enhancing radiology report generation with supplementary knowledge injection. Radar improves report generation by systematically leveraging both the internal knowledge of an LLM and externally retrieved information. Specifically, it first extracts the model’s acquired knowledge that aligns with expert image-based classification outputs. It then retrieves relevant supplementary knowledge to further enrich this information. Finally, by aggregating both sources, Radar generates more accurate and informative radiology reports. Extensive experiments on MIMIC-CXR, CheXpert-Plus, and IU X-ray demonstrate that our model outperforms state-of-the-art LLMs in both language quality and clinical accuracy

</details>

---

## 44. Make Imagination Clearer! Stable Diffusion-based Visual Imagination for Multimodal Machine Translation

- [ ] Make Imagination Clearer! Stable Diffusion-based Visual Imagination for Multimodal Machine Translation | https://aclanthology.org/2025.acl-long.1289/

- **Link**: https://aclanthology.org/2025.acl-long.1289/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual information has been introduced for enhancing machine translation (MT), and its effectiveness heavily relies on the availability of large amounts of bilingual parallel sentence pairs with manual image annotations. In this paper, we introduce a stable diffusion-based imagination network into a multimodal large language model (MLLM) to explicitly generate an image for each source sentence, thereby advancing the multimodel MT. Particularly, we build heuristic feedback with reinforcement learning to ensure the consistency of the generated image with the source sentence without the supervision of visual information, which breaks the high-cost bottleneck of image annotation in MT. Furthermore, the proposed method enables imaginative visual information to be integrated into text-only MT in addition to multimodal MT. Experimental results show that our model significantly outperforms existing multimodal MT and text-only MT, especially achieving an average improvement of more than 14 BLEU points on Multi30K and MSCOCO multimodal MT benchmarks.

</details>

---

## 45. AdvancingSMoEfor Continuous Domain Adaptation ofMLLMs: Adaptive Router and Domain-Specific Loss

- [ ] AdvancingSMoEfor Continuous Domain Adaptation ofMLLMs: Adaptive Router and Domain-Specific Loss | https://aclanthology.org/2025.acl-long.1290/

- **Link**: https://aclanthology.org/2025.acl-long.1290/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have explored Continual Instruction Tuning (CIT) in Multimodal Large Language Models (MLLMs), with a primary focus on Task-incremental CIT, where MLLMs are required to continuously acquire new tasks. However, the more practical and challenging Domain-incremental CIT, focused on the continual adaptation of MLLMs to new domains, remains underexplored. In this paper, we propose a new Sparse Mixture of Expert (SMoE) based method for domain-incremental CIT in MLLMs. During training, we learn a domain-specific SMoE module for each new domain in every FFN sub-layer of MLLMs, preventing catastrophic forgetting caused by inter-domain conflicts. Moreover, we equip the SMoE module with a domain-specific autoregressive loss (DSAL), which is used to identify the most suitable SMoE module for processing each test instruction during inference. To further enhance the SMoE module’s ability to learn domain knowledge, we design an adaptive threshold-based router (AT-Router) that allocates computing resources (experts) to instruction tokens based on their importance. Finally, we establish a new benchmark to evaluate the efficacy of our method and advance future research. Extensive experiments show that our method consistently outperforms all competitive baselines.

</details>

---

## 46. Exploring Multimodal Relation Extraction of Hierarchical Tabular Data with Multi-task Learning

- [ ] Exploring Multimodal Relation Extraction of Hierarchical Tabular Data with Multi-task Learning | https://aclanthology.org/2025.acl-long.1298/

- **Link**: https://aclanthology.org/2025.acl-long.1298/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Relation Extraction (RE) is a key task in table understanding, aiming to extract semantic relations between columns. However, complex tables with hierarchical headers are hard to obtain high-quality textual formats (e.g., Markdown) for input under practical scenarios like webpage screenshots and scanned documents, while table images are more accessible and intuitive. Besides, existing works overlook the need of mining relations among multiple columns rather than just the semantic relation between two specific columns in real-world practice. In this work, we explore utilizing Multimodal Large Language Models (MLLMs) to address RE in tables with complex structures. We creatively extend the concept of RE to include calculational relations, enabling multi-task learning of both semantic and calculational RE for mutual reinforcement. Specifically, we reconstruct table images into graph structure based on neighboring nodes to extract graph-level visual features. Such feature enhancement alleviates the insensitivity of MLLMs to the positional information within table images. We then propose a Chain-of-Thought distillation framework with self-correction mechanism to enhance MLLMs’ reasoning capabilities without increasing parameter scale. Our method significantly outperforms most baselines on wide datasets. Additionally, we release a benchmark dataset for calculational RE in complex tables.

</details>

---

## 47. LPOI: Listwise Preference Optimization for Vision Language Models

- [ ] LPOI: Listwise Preference Optimization for Vision Language Models | https://aclanthology.org/2025.acl-long.1302/

- **Link**: https://aclanthology.org/2025.acl-long.1302/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Aligning large VLMs with human preferences is a challenging task, as methods like RLHF and DPO often overfit to textual information or exacerbate hallucinations. Although augmenting negative image samples partially addresses these pitfalls, no prior work has employed listwise preference optimization for VLMs, due to the complexity and cost of constructing listwise image samples. In this work, we propose LPOI, the first object-aware listwise preference optimization developed for reducing hallucinations in VLMs. LPOI identifies and masks a critical object in the image, and then interpolates the masked region between the positive and negative images to form a sequence of incrementally more complete images. The model is trained to rank these images in ascending order of object visibility, effectively reducing hallucinations while retaining visual fidelity. LPOI requires no extra annotations beyond standard pairwise preference data, as it automatically constructs the ranked lists through object masking and interpolation. Comprehensive experiments on MMHalBench, AMBER, and Object HalBench confirm that LPOI outperforms existing preference optimization methods in reducing hallucinations and enhancing VLM performance.

</details>

---

## 48. “Give MeBF16 or Give Me Death”? Accuracy-Performance Trade-Offs inLLMQuantization

- [ ] “Give MeBF16 or Give Me Death”? Accuracy-Performance Trade-Offs inLLMQuantization | https://aclanthology.org/2025.acl-long.1304/

- **Link**: https://aclanthology.org/2025.acl-long.1304/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the popularity of large language model (LLM) quantization for inference acceleration, significant uncertainty remains regarding the accuracy-performance trade-offs associated with various quantization formats. We present a comprehensive empirical study of quantized accuracy, evaluating popular quantization formats (FP8, INT8, INT4) across academic benchmarks and real-world tasks, on the entire Llama-3.1 model family. Additionally, our study examines the difference in text generated by quantized models versus their uncompressed counterparts. Beyond benchmarks, we also present a couple of quantization improvements which allowed us to obtain state-of-the-art accuracy recovery results. Our investigation, encompassing over 500,000 individual evaluations, yields several key findings: (1) FP8 weight and activation quantization (W8A8-FP) is lossless across all model scales, (2) INT8 weight and activation quantization (W8A8-INT), when properly tuned, incurs surprisingly low 1-3% accuracy degradation, and (3) INT4 weight-only quantization (W4A16-INT) is competitive with 8-bit integer weight and activation quantization. To address the question of the “best” format for a given deployment environment, we conduct inference performance analysis using the popular open-source vLLM framework on various GPU architectures. We find that W4A16 offers the best cost-efficiency for synchronous deployments, and for asynchronous deployment on mid-tier GPUs. At the same time, W8A8 formats excel in asynchronous deployment of mid and large-size models on high-end GPUs. Our results provide a first set of practical guidelines for deploying quantized LLMs across different scales and performance requirements.

</details>

---

## 49. Walk in Others’ Shoes with a Single Glance: Human-Centric Visual Grounding with Top-View Perspective Transformation

- [ ] Walk in Others’ Shoes with a Single Glance: Human-Centric Visual Grounding with Top-View Perspective Transformation | https://aclanthology.org/2025.acl-long.1306/

- **Link**: https://aclanthology.org/2025.acl-long.1306/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual perspective-taking, an ability to envision others’ perspectives from a single self-perspective, is vital in human-robot interactions. Thus, we introduce a human-centric visual grounding task and a dataset to evaluate this ability. Recent advances in vision-language models (VLMs) have shown potential for inferring others’ perspectives, yet are insensitive to information differences induced by slight perspective changes. To address this problem, we propose a top-view enhanced perspective transformation (TEP) method, which decomposes the transition from robot to human perspectives through an abstract top-view representation. It unifies perspectives and facilitates the capture of information differences from diverse perspectives. Experimental results show that TEP improves performance by up to 18%, exhibits perspective-taking abilities across various perspectives, and generalizes effectively to robotic and dynamic scenarios.

</details>

---

## 50. FOCUS: Evaluating Pre-trained Vision-Language Models on Underspecification Reasoning

- [ ] FOCUS: Evaluating Pre-trained Vision-Language Models on Underspecification Reasoning | https://aclanthology.org/2025.acl-long.1337/

- **Link**: https://aclanthology.org/2025.acl-long.1337/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Humans possess a remarkable ability to interpret underspecified ambiguous statements by inferring their meanings from contexts such as visual inputs. This ability, however, may not be as developed in recent pre-trained vision-language models (VLMs). In this paper, we introduce a novel probing dataset called FOCUS to evaluate whether state-of-the-art VLMs have this ability. FOCUS consists of underspecified sentences paired with image contexts and carefully designed probing questions. Our experiments reveal that VLMs still fall short in handling underspecification even when visual inputs that can help resolve the ambiguities are available. To further support research in underspecification, FOCUS will be released for public use. We hope this dataset will inspire further research on the reasoning and contextual understanding capabilities of VLMs.

</details>

---

## 51. Sightation Counts: Leveraging Sighted User Feedback in Building aBLV-aligned Dataset of Diagram Descriptions

- [ ] Sightation Counts: Leveraging Sighted User Feedback in Building aBLV-aligned Dataset of Diagram Descriptions | https://aclanthology.org/2025.acl-long.1338/

- **Link**: https://aclanthology.org/2025.acl-long.1338/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Often, the needs and visual abilities differ between the annotator group and the end user group. Generating detailed diagram descriptions for blind and low-vision (BLV) users is one such challenging domain. Sighted annotators could describe visuals with ease, but existing studies have shown that direct generations by them are costly, bias-prone, and somewhat lacking by BLV standards. In this study, we ask sighted individuals to assess—rather than produce—diagram descriptions generated by vision-language models (VLM) that have been guided with latent supervision via a multi-pass inference. The sighted assessments prove effective and useful to professional educators who are themselves BLV and teach visually impaired learners. We release Sightation, a collection of diagram description datasets spanning 5k diagrams and 137k samples for completion, preference, retrieval, question answering, and reasoning training purposes and demonstrate their fine-tuning potential in various downstream tasks.

</details>

---

## 52. CheXalign: Preference fine-tuning in chestX-ray interpretation models without human feedback

- [ ] CheXalign: Preference fine-tuning in chestX-ray interpretation models without human feedback | https://aclanthology.org/2025.acl-long.1342/

- **Link**: https://aclanthology.org/2025.acl-long.1342/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Radiologists play a crucial role in translating medical images into actionable reports. However, the field faces staffing shortages and increasing workloads. While automated approaches using vision-language models (VLMs) show promise as assistants, they require exceptionally high accuracy. Most current VLMs in radiology rely solely on supervised fine-tuning. Meanwhile, additional preference fine-tuning in the post-training pipeline has become standard practice in the general domain. The challenge in radiology lies in the prohibitive cost of obtaining radiologist feedback at scale. To address this challenge, we propose an automated pipeline for preference feedback, focusing on chest X-ray radiology report generation (RRG). Specifically, our method leverages publicly available datasets containing pairs of images and radiologist-written reference reports with reference-based metrics, or Judges, eliminating the need for *additional radiologist feedback*. We investigate reward overoptimization via length exploitation in this setting and introduce a length-controlled version of the GREEN score. Our best-performing setup achieves state-of-the-art CheXbert scores on the MIMIC-CXR dataset for the RRG task while on average maintaining robust performance across six additional image perception and reasoning tasks.

</details>

---

## 53. Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains

- [ ] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains | https://aclanthology.org/2025.acl-long.1347/

- **Link**: https://aclanthology.org/2025.acl-long.1347/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) achieve remarkable success in single-image tasks. However, real-world scenarios often involve intricate multi-image inputs, leading to a notable performance decline as models struggle to disentangle critical information scattered across complex visual features. In this work, we propose Focus-Centric Visual Chain, a novel paradigm that enhances VLMs’ perception, comprehension, and reasoning abilities in multi-image scenarios. To facilitate this paradigm, we propose Focus-Centric Data Synthesis, a scalable bottom-up approach for synthesizing high-quality data with elaborate reasoning paths. Through this approach, We construct VISC-150K, a large-scale dataset with reasoning data in the form of Focus-Centric Visual Chain, specifically designed for multi-image tasks. Experimental results on seven multi-image benchmarks demonstrate that our method achieves average performance gains of 3.16% and 2.24% across two distinct model architectures, without compromising the general vision-language capabilities. Our study represents a significant step toward more robust and capable vision-language systems that can handle complex visual scenarios.

</details>

---

## 54. NusaAksara: A Multimodal and Multilingual Benchmark for PreservingIndonesian Indigenous Scripts

- [ ] NusaAksara: A Multimodal and Multilingual Benchmark for PreservingIndonesian Indigenous Scripts | https://aclanthology.org/2025.acl-long.1377/

- **Link**: https://aclanthology.org/2025.acl-long.1377/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Indonesia is rich in languages and scripts. However, most NLP progress has been made using romanized text. In this paper, we present NusaAksara, a novel public benchmark for Indonesian languages that includes their original scripts. Our benchmark covers both text and image modalities and encompasses diverse tasks such as image segmentation, OCR, transliteration, translation, and language identification. Our data is constructed by human experts through rigorous steps. NusaAksara covers 8 scripts across 7 languages, including low-resource languages not commonly seen in NLP benchmarks. Although unsupported by Unicode, the Lampung script is included in this dataset. We benchmark our data across several models, from LLMs and VLMs such as GPT-4o, Llama 3.2, and Aya 23 to task-specific systems such as PP-OCR and LangID, and show that most NLP technologies cannot handle Indonesia’s local scripts, with many achieving near-zero performance.

</details>

---

## 55. Reviving Cultural Heritage: A Novel Approach for Comprehensive Historical Document Restoration

- [ ] Reviving Cultural Heritage: A Novel Approach for Comprehensive Historical Document Restoration | https://aclanthology.org/2025.acl-long.1402/

- **Link**: https://aclanthology.org/2025.acl-long.1402/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Historical documents represent an invaluable cultural heritage, yet have undergone significant degradation over time through tears, water erosion, and oxidation. Existing Historical Document Restoration (HDR) methods primarily focus on single modality or limited-size restoration, failing to meet practical needs. To fill this gap, we present a full-page HDR dataset (FPHDR) and a novel automated HDR solution (AutoHDR). Specifically, FPHDR comprises 1,633 real and 6,543 synthetic images with character-level and line-level locations, as well as character annotations in different damage grades. AutoHDR mimics historians’ restoration workflows through a three-stage approach: OCR-assisted damage localization, vision-language context text prediction, and patch autoregressive appearance restoration. The modular architecture of AutoHDR enables seamless human-machine collaboration, allowing for flexible intervention and optimization at each restoration stage. Experiments demonstrate AutoHDR’s remarkable performance in HDR. When processing severely damaged documents, our system improves OCR accuracy from 46.83% to 84.05%, with further enhancement to 94.25% through human-machine collaboration. We believe this work represents a significant advancement in automated historical document restoration and contributes substantially to cultural heritage preservation. The model and dataset are available at https://github.com/SCUT-DLVCLab/AutoHDR.

</details>

---

## 56. Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models

- [ ] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models | https://aclanthology.org/2025.acl-long.1411/

- **Link**: https://aclanthology.org/2025.acl-long.1411/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) excel at extracting and reasoning about information from images. Yet, their capacity to leverage internal knowledge about specific entities remains underexplored. This work investigates the disparity in model performance when answering factual questions about an entity described in text versus depicted in an image. Our results reveal a significant accuracy drop — reaching 18% for some models — when the entity is presented visually instead of textually. To study this gap we present PopVQA, a dataset which allows separating entity recognition and question answering, and use it to benchmark several models. We hypothesize that this decline arises from limitations in how information flows from image tokens to query tokens. Thus, we use mechanistic interpretability tools to reveal that, although image tokens are preprocessed by the vision encoder, meaningful information flow from these tokens occurs only in the much deeper layers. Furthermore, critical image processing happens in the language model’s middle layers, allowing few layers for consecutive reasoning, highlighting a potential inefficiency in how the model utilizes its layers for reasoning. These insights shed light on the internal mechanics of VLMs and offer pathways for enhancing their reasoning capabilities. PopVQA can be found at https://huggingface.co/datasets/idoco/PopVQA.

</details>

---

## 57. FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation

- [ ] FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation | https://aclanthology.org/2025.acl-long.1426/

- **Link**: https://aclanthology.org/2025.acl-long.1426/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have experienced rapid development in recent years. However, in the financial domain, there is a notable lack of effective and specialized multimodal evaluation datasets. To advance the development of MLLMs in the finance domain, we introduce FinMME, encompassing more than 11,000 high-quality financial research samples across 18 financial domains and 6 asset classes, featuring 10 major chart types and 21 subtypes. We ensure data quality through 20 annotators and carefully designed validation mechanisms. Additionally, we develop FinScore, an evaluation system incorporating hallucination penalties and multi-dimensional capability assessment to provide an unbiased evaluation. Extensive experimental results demonstrate that even state-of-the-art models like GPT-4o exhibit unsatisfactory performance on FinMME, highlighting its challenging nature. The benchmark exhibits high robustness with prediction variations under different prompts remaining below 1%, demonstrating superior reliability compared to existing datasets. Our dataset and evaluation protocol are available at https://huggingface.co/datasets/luojunyu/FinMME and https://github.com/luo-junyu/FinMME.

</details>

---

## 58. Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model

- [ ] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model | https://aclanthology.org/2025.acl-long.143/

- **Link**: https://aclanthology.org/2025.acl-long.143/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Most Large Vision-Language Models (LVLMs) to date are trained predominantly on English data, which makes them struggle to understand non-English input and fail to generate output in the desired target language. Existing efforts mitigate these issues by adding multilingual training data, but do so in a largely ad-hoc manner, lacking insight into how different training mixes tip the scale for different groups of languages. In this work, we present a comprehensive investigation into the training strategies for massively multilingual LVLMs. First, we conduct a series of multi-stage experiments spanning 13 downstream vision-language tasks and 43 languages, systematically examining: (1) the number of training languages that can be included without degrading English performance and (2) optimal language distributions of pre-training as well as (3) instruction-tuning data. Further, we (4) investigate how to improve multilingual text-in-image understanding, and introduce a new benchmark for the task. Surprisingly, our analysis reveals that one can (i) include as many as 100 training languages simultaneously (ii) with as little as 25-50% of non-English data, to greatly improve multilingual performance while retaining strong English performance. We further find that (iii) including non-English OCR data in pre-training and instruction-tuning is paramount for improving multilingual text-in-image understanding. Finally, we put all our findings together and train , a 100-language LVLM, offering state-of-the-art performance in an evaluation covering 14 tasks and 56 languages.

</details>

---

## 59. METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling

- [ ] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling | https://aclanthology.org/2025.acl-long.1452/

- **Link**: https://aclanthology.org/2025.acl-long.1452/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chart generation aims to generate code to produce charts satisfying the desired visual properties, e.g., texts, layout, color, and type. It has great potential to empower the automatic professional report generation in financial analysis, research presentation, education, and healthcare. In this work, we build a vision-language model (VLM) based multi-agent framework for effective automatic chart generation. Generating high-quality charts requires both strong visual design skills and precise coding capabilities that embed the desired visual properties into code. Such a complex multi-modal reasoning process is difficult for direct prompting of VLMs. To resolve these challenges, we propose METAL, a multi-agent framework that decomposes the task of chart generation into the iterative collaboration among specialized agents. METAL achieves a 5.2% improvement in the F1 score over the current best result in the chart generation task. Additionally, METAL improves chart generation performance by 11.33% over Direct Prompting with LLaMA-3.2-11B.Furthermore, the METAL framework exhibits the phenomenon of test-time scaling: its performance increases monotonically as the logarithm of computational budget grows from 512 to 8192 tokens.

</details>

---

## 60. VISA: Retrieval Augmented Generation with Visual Source Attribution

- [ ] VISA: Retrieval Augmented Generation with Visual Source Attribution | https://aclanthology.org/2025.acl-long.1456/

- **Link**: https://aclanthology.org/2025.acl-long.1456/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generation with source attribution is important for enhancing the verifiability of retrieval-augmented generation (RAG) systems. However, existing approaches in RAG primarily link generated content to document-level references, making it challenging for users to locate evidence among multiple content-rich retrieved documents. To address this challenge, we propose Retrieval-Augmented Generation with Visual Source Attribution (VISA), a novel approach that combines answer generation with visual source attribution. Leveraging large vision-language models (VLMs), VISA identifies the evidence and highlights the exact regions that support the generated answers with bounding boxes in the retrieved document screenshots. To evaluate its effectiveness, we curated two datasets: Wiki-VISA, based on crawled Wikipedia webpage screenshots, and Paper-VISA, derived from PubLayNet and tailored to the medical domain. Experimental results demonstrate the effectiveness of VISA for visual source attribution on documents’ original look, as well as highlighting the challenges for improvement.

</details>

---

## 61. ConInstruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities

- [ ] ConInstruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities | https://aclanthology.org/2025.acl-long.146/

- **Link**: https://aclanthology.org/2025.acl-long.146/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing attacks against multimodal language models often communicate instruction through text, either as an explicit malicious instruction or a crafted generic prompt, and accompanied by a toxic image. In contrast, here we exploit the capabilities of MLLMs in following non-textual instruction, i.e., an adversarial image or audio, namely Con Instruction. It is a novel gray-box attack method that generates adversarial images or audio to convey specific harmful instructions to MLLMs. We also find that combining our adversarial examples with certain non-empty text inputs amplifies attack success, while appending these after malicious text has limited effects. To evaluate whether an attack is successful, we introduce a new attack response categorization (ARC) that considers the response quality and relevancy concerning the malicious instruction. The results show that Con Instruction effectively bypasses the safety mechanisms in various visual and audio-language models, including LLaVA-v1.5, InternVL, Qwen-VL, and Qwen-Audio, across two standard benchmarks: AdvBench and SafeBench. Specifically, our method achieves the highest attack success rates, reaching 81.3% and 86.6% on LLaVA-v1.5 (13B). We show that larger models are more susceptible toCon Instruction, contrasting observations in their underlying LLMs. On the defense side, we explore various methods against our attacks and find substantial gaps among existing techniques. The code will be made available upon publication.

</details>

---

## 62. Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images

- [ ] Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images | https://aclanthology.org/2025.acl-long.1462/

- **Link**: https://aclanthology.org/2025.acl-long.1462/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent studies have shown that Large Vision-Language Models (VLMs) tend to neglect image content and over-rely on language-model priors, resulting in errors in visually grounded tasks and hallucinations. We hypothesize that this issue arises because existing VLMs are not explicitly trained to generate texts that are accurately grounded in fine-grained image details. To enhance visual feedback during VLM training, we propose S-VCO (Symmetrical Visual Contrastive Optimization), a novel finetuning objective that steers the model toward capturing important visual details and aligning them with corresponding text tokens. To further facilitate this detailed alignment, we introduce MVC, a paired image-text dataset built by automatically filtering and augmenting visual counterfactual data to challenge the model with hard contrastive cases involving Minimal Visual Contrasts. Experiments show that our method consistently improves VLM performance across diverse benchmarks covering various abilities and domains, achieving up to a 22% reduction in hallucinations, and significant gains in vision-centric and general tasks. Notably, these improvements become increasingly pronounced in benchmarks with higher visual dependency. In short, S-VCO offers a significant enhancement of VLM’s visually-dependent task performance while retaining or even improving the model’s general abilities.

</details>

---

## 63. Predicting Implicit Arguments in Procedural Video Instructions

- [ ] Predicting Implicit Arguments in Procedural Video Instructions | https://aclanthology.org/2025.acl-long.1467/

- **Link**: https://aclanthology.org/2025.acl-long.1467/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Procedural texts help AI enhance reasoning about context and action sequences. Transforming these into Semantic Role Labeling (SRL) improves understanding of individual steps by identifying predicate-argument structure like verb,what,where/with. Procedural instructions are highly elliptic, for instance, (i) add cucumber to the bowl and (ii) add sliced tomatoes, the second step’s where argument is inferred from the context, referring to where the cucumber was placed. Prior SRL benchmarks often miss implicit arguments, leading to incomplete understanding. To address this, we introduce Implicit-VidSRL, a dataset that necessitates inferring implicit and explicit arguments from contextual information in multimodal cooking procedures. Our proposed dataset benchmarks multimodal models’ contextual reasoning, requiring entity tracking through visual changes in recipes. We study recent multimodal LLMs and reveal that they struggle to predict implicit arguments of what and where/with from multi-modal procedural data given the verb. Lastly, we propose iSRL-Qwen2-VL, which achieves a 17% relative improvement in F1-score for what-implicit and a 14.7% for where/with-implicit semantic roles over GPT-4o.

</details>

---

## 64. Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning

- [ ] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning | https://aclanthology.org/2025.acl-long.1482/

- **Link**: https://aclanthology.org/2025.acl-long.1482/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across diverse tasks. Despite great success, recent studies show that LVLMs encounter substantial limitations when engaging with visual graphs. To study the reason behind these limitations, we propose VGCure, a comprehensive benchmark covering 22 tasks for examining the fundamental graph understanding and reasoning capacities of LVLMs. Extensive evaluations conducted on 14 LVLMs reveal that LVLMs are weak in basic graph understanding and reasoning tasks, particularly those concerning relational or structurally complex information. Based on this observation, we propose a structure-aware fine-tuning framework to enhance LVLMs with structure learning abilities through three self-supervised learning tasks. Experiments validate the effectiveness of our method in improving LVLMs’ performance on fundamental and downstream graph learning tasks, as well as enhancing their robustness against complex visual graphs.

</details>

---

## 65. ISR: Self-Refining Referring Expressions for Entity Grounding

- [ ] ISR: Self-Refining Referring Expressions for Entity Grounding | https://aclanthology.org/2025.acl-long.1483/

- **Link**: https://aclanthology.org/2025.acl-long.1483/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Entity grounding, a crucial task in constructing multimodal knowledge graphs, aims to align entities from knowledge graphs with their corresponding images. Unlike conventional visual grounding tasks that use referring expressions (REs) as inputs, entity grounding relies solely on entity names and types, presenting a significant challenge. To address this, we introduce a novel **I**terative **S**elf-**R**efinement (**ISR**) scheme to enhance the multimodal large language model’s capability to generate high quality REs for the given entities as explicit contextual clues. This training scheme, inspired by human learning dynamics and human annotation processes, enables the MLLM to iteratively generate and refine REs by learning from successes and failures, guided by outcome rewards from a visual grounding model. This iterative cycle of self-refinement avoids overfitting to fixed annotations and fosters continued improvement in referring expression generation. Extensive experiments demonstrate that our methods surpasses other methods in entity grounding, highlighting its effectiveness, robustness and potential for broader applications.

</details>

---

## 66. Activating Distributed Visual Region withinLLMs for Efficient and Effective Vision-Language Training and Inference

- [ ] Activating Distributed Visual Region withinLLMs for Efficient and Effective Vision-Language Training and Inference | https://aclanthology.org/2025.acl-long.1484/

- **Link**: https://aclanthology.org/2025.acl-long.1484/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) typically learn visual capacity through visual instruction tuning, involving updates to both a projector and their LLM backbones. Inspired by the concept of a visual region in the human brain, we investigate the existence of an analogousvisual regionwithin LLMs that functions as a cognitive core, and explore the potential of efficient training of LVLMs via selective layers tuning. Using Bunny-Llama-3-8B-V for detailed analysis and other three LVLMs for validation across diverse visual and textual tasks, we find that selectively updating 25% of LLMs layers, when sparsely and uniformly distributed, can preserve nearly 99% of visual performance and maintain or improve textual task results, while effectively reducing training time. Based on this targeted training approach, we further propose a novel visual region-based pruning paradigm, removing non-critical layers outside the visual region, which can achieve minimal performance loss. This study offers an effective and efficient strategy for LVLM training and inference by activating a layer-wise visual region within LLMs, which proves consistently effective across different models.

</details>

---

## 67. Multi-Modality Expansion and Retention forLLMs through Parameter Merging and Decoupling

- [ ] Multi-Modality Expansion and Retention forLLMs through Parameter Merging and Decoupling | https://aclanthology.org/2025.acl-long.1491/

- **Link**: https://aclanthology.org/2025.acl-long.1491/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning Large Language Models (LLMs) with multimodal encoders on modality-specific data expands the modalities that LLMs can handle, leading to the formation of Multimodal LLMs (MLLMs). However, this paradigm heavily relies on resource-intensive and inflexible fine-tuning from scratch with new multimodal data. In this paper, we propose MMER (Multi-modality Expansion and Retention), a training-free approach that integrates existing MLLMs for effective multimodal expansion while retaining their original performance. Specifically, MMER reuses MLLMs’ multimodal encoders while merging their LLM parameters. By comparing original and merged LLM parameters, MMER generates binary masks to approximately separate LLM parameters for each modality. These decoupled parameters can independently process modality-specific inputs, reducing parameter conflicts and preserving original MLLMs’ fidelity. MMER can also mitigate catastrophic forgetting by applying a similar process to MLLMs fine-tuned on new tasks. Extensive experiments show significant improvements over baselines, proving that MMER effectively expands LLMs’ multimodal capabilities while retaining 99% of the original performance, and also markedly mitigates catastrophic forgetting.

</details>

---

## 68. HintsOfTruth: A Multimodal Checkworthiness Detection Dataset with Real and Synthetic Claims

- [ ] HintsOfTruth: A Multimodal Checkworthiness Detection Dataset with Real and Synthetic Claims | https://aclanthology.org/2025.acl-long.1510/

- **Link**: https://aclanthology.org/2025.acl-long.1510/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Misinformation can be countered with fact-checking, but the process is costly and slow. Identifying checkworthy claims is the first step, where automation can help scale fact-checkers’ efforts. However, detection methods struggle with content that is (1) multimodal, (2) from diverse domains, and (3) synthetic. We introduce HintsOfTruth, a public dataset for multimodal checkworthiness detection with 27K real-world and synthetic image/claim pairs. The mix of real and synthetic data makes this dataset unique and ideal for benchmarking detection methods. We compare fine-tuned and prompted Large Language Models (LLMs). We find that well-configured lightweight text-based encoders perform comparably to multimodal models but the former only focus on identifying non-claim-like content. Multimodal LLMs can be more accurate but come at a significant computational cost, making them impractical for large-scale applications. When faced with synthetic data, multimodal models perform more robustly.

</details>

---

## 69. A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models

- [ ] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models | https://aclanthology.org/2025.acl-long.1514/

- **Link**: https://aclanthology.org/2025.acl-long.1514/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current vision-language models (VLMs) understand complex vision-text tasks by extracting overall semantic information from large-scale cross-modal associations. However, extracting from large-scale cross-modal associations often smooths out semantic details and requires large computations, limiting multimodal fine-grained understanding performance and efficiency. To address this issue, this paper proposes a detail-oriented prompt learning (DoPL) method for vision-language models to implement fine-grained multi-modal semantic alignment with merely 0.25M trainable parameters. According to the low-entropy information concentration theory, DoPL explores shared interest tokens from text-vision correlations and transforms them into alignment weights to enhance text prompt and vision prompt via detail-oriented prompt generation. It effectively guides the current frozen layer to extract fine-grained text-vision alignment cues. Furthermore, DoPL constructs detail-oriented prompt generation for each frozen layer to implement layer-by-layer localization of fine-grained semantic alignment, achieving precise understanding in complex vision-text tasks. DoPL performs well in parameter-efficient fine-grained semantic alignment with only 0.12% tunable parameters for vision-language models. The state-of-the-art results over the previous parameter-efficient fine-tuning methods and full fine-tuning approaches on six benchmarks demonstrate the effectiveness and efficiency of DoPL in complex multi-modal tasks.

</details>

---

## 70. UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces

- [ ] UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces | https://aclanthology.org/2025.acl-long.1558/

- **Link**: https://aclanthology.org/2025.acl-long.1558/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large multimodal models exhibit remarkable intelligence, yet their embodied cognitive abilities during motion in open-ended urban aerial spaces remain to be explored. We introduce a benchmark to evaluate whether video-large language models (Video-LLMs) can naturally process continuous first-person visual observations like humans, enabling recall, perception, reasoning, and navigation. We have manually control drones to collect 3D embodied motion video data from real-world cities and simulated environments, resulting in 1.5k video clips. Then we design a pipeline to generate 5.2k multiple-choice questions. Evaluations of 17 widely-used Video-LLMs reveal current limitations in urban embodied cognition. Correlation analysis provides insight into the relationships between different tasks, showing that causal reasoning has a strong correlation with recall, perception, and navigation, while the abilities for counterfactual and associative reasoning exhibit lower correlation with other tasks. We also validate the potential for Sim-to-Real transfer in urban embodiment through fine-tuning.

</details>

---

## 71. ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities

- [ ] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities | https://aclanthology.org/2025.acl-long.1560/

- **Link**: https://aclanthology.org/2025.acl-long.1560/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional fixed test datasets fall short in evaluating the open-ended capabilities of foundation models. To address this, we propose ONEBench (OpeN-Ended Benchmarking), a new paradigm that consolidates individual evaluation datasets into a unified, ever-expanding sample pool. ONEBench enables custom benchmarks for specific capabilities while reusing and aggregating samples, mitigating overfitting and dataset bias for broader capability assessment. It reframes model evaluation as selecting and aggregating sample-level tests.Transitioning from task-specific benchmarks to ONEBench introduces two challenges: heterogeneity (aggregating diverse metrics) and incompleteness(comparing models tested on different data subsets). To address these, we propose an aggregation algorithm that ensures identifiability (asymptotically recovering ground-truth scores) and rapid convergence, enabling accurate model comparisons with relatively little data. On homogenous datasets, our algorithm produces rankings that highly correlate with average scores. Moreover, it remains robust to over 95% missing measurements, reducing evaluation costs by up to 20x with minimal impact on rankings. We introduce ONEBench-LLM for language models and ONEBench-LMM for vision-language models, unifying evaluations across these domains, and enabling targeted model testing across diverse capabilities.

</details>

---

## 72. Logic-Regularized Verifier Elicits Reasoning fromLLMs

- [ ] Logic-Regularized Verifier Elicits Reasoning fromLLMs | https://aclanthology.org/2025.acl-long.1567/

- **Link**: https://aclanthology.org/2025.acl-long.1567/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Verifiers are crucial components for enhancing modern LLMs’ reasoning capability. Typical verifiers require resource-intensive supervised dataset construction, which is costly and faces limitations in data diversity. In this paper, we propose LOVER, an unsupervised verifier regularized by logical rules. LOVER treats the verifier as a binary latent variable, utilizing internal activations and enforcing three logical constraints on multiple reasoning paths: negation consistency, intra-group consistency, and inter-group consistency (grouped by the final answer). By incorporating logical rules as priors, LOVER can leverage unlabeled examples and is directly compatible with any off-the-shelf LLMs. Experiments on 10 datasets demonstrate that LOVER significantly outperforms unsupervised baselines, achieving performance comparable to the supervised verifier (reaching its 95% level on average).

</details>

---

## 73. CoRe-MMRAG: Cross-Source Knowledge Reconciliation for MultimodalRAG

- [ ] CoRe-MMRAG: Cross-Source Knowledge Reconciliation for MultimodalRAG | https://aclanthology.org/2025.acl-long.1583/

- **Link**: https://aclanthology.org/2025.acl-long.1583/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Retrieval-Augmented Generation (MMRAG) has been introduced to enhance Multimodal Large Language Models by incorporating externally retrieved multimodal knowledge, but it introduces two challenges: Parametric-Retrieved Knowledge Inconsistency (PRKI), where discrepancies between parametric and retrieved knowledge create uncertainty in determining reliability, and Visual-Textual Knowledge Inconsistency (VTKI), where misalignment between visual and textual sources disrupts entity representation. To address these challenges, we proposeCross-source knowledgeReconciliation forMultiModalRAG(CoRe-MMRAG), a novel end-to-end framework that effectively reconciles inconsistencies across knowledge sources. CoRe-MMRAG follows a four-stage pipeline: it first generates an internal response from parametric knowledge, then selects the most relevant multimodal evidence via joint similarity assessment, generates an external response, and finally integrates both to produce a reliable answer. Additionally, a specialized training paradigm enhances knowledge source discrimination, multimodal integration, and unified answer generation. Experiments on KB-VQA benchmarks show that CoRe-MMRAG achieves substantial improvements over baseline methods, achieving 5.6% and 9.3% performance gains on InfoSeek and Encyclopedic-VQA, respectively. We release code and data at https://github.com/TyangJN/CoRe-MMRAG.

</details>

---

## 74. Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings fromLLMs

- [ ] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings fromLLMs | https://aclanthology.org/2025.acl-long.159/

- **Link**: https://aclanthology.org/2025.acl-long.159/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Extracting sentence embeddings from large language models (LLMs) is a promising direction, as LLMs have demonstrated stronger semantic understanding capabilities. Previous studies typically focus on prompt engineering to elicit sentence embeddings from LLMs by prompting the model to encode sentence information into the embedding of the last token.However, LLMs are mostly decoder-only models with causal attention and the earlier tokens in the sentence cannot attend to the latter tokens, resulting in biased encoding of sentence information and cascading effects on the final decoded token.To this end, we propose a novel Token Prepending (TP) technique that prepends each layer’s decoded sentence embedding to the beginning of the sentence in the next layer’s input, allowing earlier tokens to attend to the complete sentence information under the causal attention mechanism.The proposed TP technique is a plug-and-play and training-free technique, which means it can be seamlessly integrated with various prompt-based sentence embedding methods and autoregressive LLMs.Extensive experiments on various Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our proposed TP technique can significantly improve the performance of existing prompt-based sentence embedding methods across different LLMs, while incurring negligible additional inference cost.

</details>

---

## 75. Scalable Vision Language Model Training via High Quality Data Curation

- [ ] Scalable Vision Language Model Training via High Quality Data Curation | https://aclanthology.org/2025.acl-long.1595/

- **Link**: https://aclanthology.org/2025.acl-long.1595/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduceSAIL-VL(ScAlable Vision Language Model TraIning via High QuaLity Data Curation), an open-source vision language model (VLM) series achieving state-of-the-art (SOTA) performance in 2B and 8B parameters. The following three key improvements contribute to SAIL-VL’s leading performance: (1) Scalable high-quality visual understanding data construction: We implement a data construction pipeline to enable hundred-million-scale high-quality recaption data annotation. The resulted dataset SAIL-Caption is validated to be of the highest data quality compared with opensource datasets. (2) Scalable Pretraining with High-Quality Visual Understanding Data: We scale SAIL-VL’s pretraining budget up to 655B tokens and show that even a 2B VLM benefits from scaled up training data sizes, exhibiting logarithmic data size scaling laws in benchmark performance. (3) Scalable SFT via data quantity and complexity scaling: We curate a high-quality SFT dataset collection with leading data quantity scaling effectiveness and demonstrate that training with progressively higher-complexity data surpasses baseline one-stage training by a large margin. SAIL-VL series models achieve the highest average score in 18 widely used VLM benchmarks in our evaluation, with the 2B model takes the top position over VLMs of comparable sizes on OpenCompass 2024 (https://rank.opencompass.org.cn/leaderboard-multimodal), demonstrating robust visual comprehension abilities. SAIL-VL series models are released at HuggingFace (https://huggingface.co/BytedanceDouyinContent).

</details>

---

## 76. Design Choices for Extending the Context Length of Visual Language Models

- [ ] Design Choices for Extending the Context Length of Visual Language Models | https://aclanthology.org/2025.acl-long.1603/

- **Link**: https://aclanthology.org/2025.acl-long.1603/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Language Models (VLMs) demonstrate impressive capabilities in processing multimodal inputs, yet applications such as visual agents, which require handling multiple images and high-resolution videos, demand enhanced long-range modeling. Moreover, existing open-source VLMs lack systematic exploration into extending their context length, and commercial models often provide limited details. To tackle this, we aim to establish an effective solution that enhances long context performance of VLMs while preserving their capacities in short context scenarios. Towards this goal, we make the best design choice through extensive experiment settings from data curation to context window extending and utilizing: (1) we analyze data sources and length distributions to construct ETVLM - a data recipe to balance the performance across scenarios; (2) we examine existing position extending methods, identify their limitations and propose M-RoPE++ as an enhanced approach; we also choose to solely instruction-tune the backbone with mixed-source data; (3) we discuss how to better utilize extended context windows and propose hybrid-resolution training. Built on the Qwen-VL series model, we propose Giraffe, which is effectively extended to 128K lengths. Evaluated on extensive long context VLM benchmarks such as VideoMME and Viusal Haystacks, our Giraffe achieves state-of-the-art performance among similarly sized open-source long VLMs and is competitive with commercial model GPT-4V. We will open-source the code, data, and models.

</details>

---

## 77. Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models

- [ ] Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models | https://aclanthology.org/2025.acl-long.162/

- **Link**: https://aclanthology.org/2025.acl-long.162/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Evaluating Video Language Models (VLMs) is a challenging task. Due to its transparency, Multiple-Choice Question Answering (MCQA) is widely used to measure the performance of these models through accuracy. However, existing MCQA benchmarks fail to capture the full reasoning capabilities of VLMs due to selection bias, when models disproportionately favor certain answer options based on positional patterns observed during training. In this work, we conduct a comprehensive empirical analysis of several VLM architectures across major datasets designed to assess complex video-focused reasoning. We identify where the bias is most pronounced and demonstrate to what extent model responses reflect genuine understanding of video content and related questions, as opposed to reliance on arbitrary patterns or superficial cues, such as answer position. By decomposing the MCQA task and adapting fairness bias metrics to VLMs, we introduce a post-processing calibration technique BOLD to balance this bias. Our results show that reducing selection bias improves not only debiasing metrics but also overall model performance, including Accuracy and F1 Mean score. Our method, by suppressing “blind guessing”, offers a more cost- and time-effective approach to mitigating selection bias compared to existing techniques. This study represents the first focused investigation of selection bias in video-to-text LLM-powered models.

</details>

---

## 78. Cracking the Code of Hallucination inLVLMs with Vision-aware Head Divergence

- [ ] Cracking the Code of Hallucination inLVLMs with Vision-aware Head Divergence | https://aclanthology.org/2025.acl-long.175/

- **Link**: https://aclanthology.org/2025.acl-long.175/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have made substantial progress in integrating large language models (LLMs) with visual inputs, enabling advanced multimodal reasoning. Despite their success, a persistent challenge is hallucination—where generated text fails to accurately reflect visual content—undermining both accuracy and reliability. Existing methods focus on alignment training or decoding refinements but primarily address symptoms at the generation stage without probing the underlying causes. In this work, we investigate the internal mechanisms driving hallucination in LVLMs, with an emphasis on the multi-head attention module. Specifically, we introduce Vision-aware Head Divergence (VHD), a metric that quantifies the sensitivity of attention head outputs to visual context. Based on this, our findings reveal the presence of vision-aware attention heads that are more attuned to visual information; however, the model’s overreliance on its prior language patterns is closely related to hallucinations. Building on these insights, we propose Vision-aware Head Reinforcement (VHR), a training-free approach to mitigate hallucination by enhancing the role of vision-aware attention heads. Extensive experiments demonstrate that our method achieves superior performance compared to state-of-the-art approaches in mitigating hallucinations, while maintaining high efficiency with negligible additional time overhead. The code is available at https://github.com/jinghan1he/VHR.

</details>

---

## 79. Progressive Multimodal Reasoning via Active Retrieval

- [ ] Progressive Multimodal Reasoning via Active Retrieval | https://aclanthology.org/2025.acl-long.180/

- **Link**: https://aclanthology.org/2025.acl-long.180/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). AR-MCTS follows the MCTS algorithm and heuristically integrates an active retrieval mechanism during the expansion stage to automatically acquire high-quality step-wise reasoning annotations. Moreover, we further introduce curriculum training objectives to progressively align with a process reward model, ultimately achieving process-level multimodal reasoning verification. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of AR-MCTS. Further analysis demonstrates that it can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning.

</details>

---

## 80. Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions

- [ ] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions | https://aclanthology.org/2025.acl-long.182/

- **Link**: https://aclanthology.org/2025.acl-long.182/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In visual question answering (VQA) context, users often pose ambiguous questions to visual language models (VLMs) due to varying expression habits. Existing research addresses such ambiguities primarily by rephrasing questions. These approaches neglect the inherently interactive nature of user interactions with VLMs, where ambiguities can be clarified through user feedback. However, research on interactive clarification faces two major challenges: (1) Benchmarks are absent to assess VLMs’ capacity for resolving ambiguities through interaction; (2) VLMs are trained to prefer answering rather than asking, preventing them from seeking clarification. To overcome these challenges, we introduce ClearVQA benchmark, which targets three common categories of ambiguity in VQA context, and encompasses various VQA scenarios. Furthermore, we propose an automated pipeline to generate ambiguity-clarification question pairs, enabling VLMs to ask reasonable clarification questions and generate more accurate and specific answers based on user feedback, as demonstrated by experimental results.

</details>

---

## 81. VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism

- [ ] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism | https://aclanthology.org/2025.acl-long.199/

- **Link**: https://aclanthology.org/2025.acl-long.199/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have shown exceptional performance in multimodal tasks, but their effectiveness in complex visual reasoning is still constrained, especially when employing Chain-of-Thought prompting techniques. In this paper, we propose VReST, a novel training-free approach that enhances Reasoning in LVLMs through Monte Carlo Tree Search and Self-Reward mechanisms. VReST meticulously traverses the reasoning landscape by establishing a search tree, where each node encapsulates a reasoning step, and each path delineates a comprehensive reasoning sequence. Our innovative multimodal Self-Reward mechanism assesses the quality of reasoning steps by integrating the utility of sub-questions, answer correctness, and the relevance of vision-language clues, all without the need for additional models. VReST surpasses current prompting methods and secures state-of-the-art performance across three multimodal mathematical reasoning benchmarks. Furthermore, it substantiates the efficacy of test-time scaling laws in multimodal tasks, offering a promising direction for future research.

</details>

---

## 82. Meta-Reflection: A Feedback-Free Reflection Learning Framework

- [ ] Meta-Reflection: A Feedback-Free Reflection Learning Framework | https://aclanthology.org/2025.acl-long.201/

- **Link**: https://aclanthology.org/2025.acl-long.201/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the remarkable capabilities of large language models (LLMs) in natural language understanding and reasoning, they often display undesirable behaviors, such as generating hallucinations and unfaithful reasoning. A prevalent strategy to mitigate these issues is the use of reflection, which refines responses through an iterative process. However, while promising, reflection heavily relies on high-quality external feedback and requires iterative multi-agent inference processes, thus hindering its practical application. In this paper, we propose Meta-Reflection, a novel feedback-free reflection mechanism that necessitates only a single inference pass without external feedback. Motivated by the human ability to remember and retrieve reflections from past experiences when encountering similar problems, Meta-Reflection integrates reflective insights into a codebook, allowing the historical insights to be stored, retrieved, and used to guide LLMs in problem-solving. To thoroughly investigate and evaluate the practicality of Meta-Reflection in real-world scenarios, we introduce an industrial e-commerce benchmark named E-commerce Customer Intent Detection. Extensive experiments conducted on both public datasets and the ECID benchmark highlight the effectiveness and efficiency of our proposed approach. Project is available at https://github.com/DCDmllm/Meta-Reflection

</details>

---

## 83. Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models

- [ ] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models | https://aclanthology.org/2025.acl-long.205/

- **Link**: https://aclanthology.org/2025.acl-long.205/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have shown impressive progress by integrating visual perception with linguistic understanding to produce contextually grounded outputs. Despite these advancements achieved, LVLMs still suffer from the hallucination problem, e.g., they tend to produce content that does not exist in the input images. Our investigation suggests that such hallucinations often stem from the deficiencies in fine-grained comprehension on the visual aspect, particularly when visual scenes exhibit appearance or semantic similarities (e.g., bicycle vs. motorcycles, baseball bat vs. baseball). In this work, we show such hallucination is naturally mitigated via a novel method called visual evidence prompting, utilizing small visual models to complement the LVLMs. While traditional visual models are not adept at interacting with humans, they excel at perceiving the fine-grained image contents. By symbolizing the professional outputs of domain-expert models as prompts, the LVLM generalists are able to refer to these evidences as visual knowledge to generate more precise answers. Detailed analysis shows that visual evidence enables models to adjust and rectify the attribution and attention on the images, reducing visual confusion by suppressing false activation while enhancing correct ones. Extensive experiments and in-depth analysis demonstrate the effectiveness of our method. We hope our straightforward but insightful work enhances the comprehension of hallucination in LVLMs and offers valuable perspectives on addressing such challenges.

</details>

---

## 84. AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness

- [ ] AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness | https://aclanthology.org/2025.acl-long.213/

- **Link**: https://aclanthology.org/2025.acl-long.213/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The proliferation of multimodal memes in the social media era demands that multimodal Large Language Models (mLLMs) effectively understand meme harmfulness. Existing benchmarks for assessing mLLMs on harmful meme understanding rely on accuracy-based, model-agnostic evaluations using static datasets. These benchmarks are limited in their ability to provide up-to-date and thorough assessments, as online memes evolve dynamically. To address this, we propose AdamMeme, a flexible, agent-based evaluation framework that adaptively probes the reasoning capabilities of mLLMs in deciphering meme harmfulness. Through multi-agent collaboration, AdamMeme provides comprehensive evaluations by iteratively updating the meme data with challenging samples, thereby exposing specific limitations in how mLLMs interpret harmfulness. Extensive experiments show that our framework systematically reveals the varying performance of different target mLLMs, offering in-depth, fine-grained analyses of model-specific weaknesses. Our code is available at https://github.com/Lbotirx/AdamMeme.

</details>

---

## 85. Towards Text-Image Interleaved Retrieval

- [ ] Towards Text-Image Interleaved Retrieval | https://aclanthology.org/2025.acl-long.214/

- **Link**: https://aclanthology.org/2025.acl-long.214/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current multimodal information retrieval studies mainly focus on single-image inputs, which limits real-world applications involving multiple images and text-image interleaved content. In this work, we introduce the text-image interleaved retrieval (TIIR) task, where the query and document are interleaved text-image sequences, and the model is required to understand the semantics from the interleaved context for effective retrieval. We construct a TIIR benchmark based on naturally interleaved wikiHow tutorials, where a specific pipeline is designed to generate interleaved queries. To explore the task, we adapt several off-the-shelf retrievers and build a dense baseline by interleaved multimodal large language model (MLLM). We then propose a novel Matryoshka Multimodal Embedder (MME), which compresses the number of visual tokens at different granularity, to address the challenge of excessive visual tokens in MLLM-based TIIR models. Experiments demonstrate that simple adaption of existing models does not consistently yield effective results. Our MME achieves significant improvements over the baseline by substantially fewer visual tokens. We provide extensive analysis and will release the dataset and code to facilitate future research.

</details>

---

## 86. Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding

- [ ] Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding | https://aclanthology.org/2025.acl-long.222/

- **Link**: https://aclanthology.org/2025.acl-long.222/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite existing multimodal language models showing impressive performance on the video understanding task, extremely long videos still pose significant challenges to language model’s context length, memory consumption, and computational complexity. To address these issues, we propose a vision-language model named Sophia for long video understanding, which can efficiently handle hour-scale long videos. First, we employ a Shot-adaptive Frame Pruning technique, which naturally segments long videos into multiple camera shots, to more sharply identify and focus on the frames relevant to the query. Additionally, we introduce a Hierarchical Attention mechanism to effectively model the long-term temporal dependencies between video frames, which achieves a time and space complexity of O(N) w.r.t. the input sequence length N while theoretically maintaining the global modeling efficiency. Experimentally, our Sophia exhibits competitive performance compared to existing video understanding baselines across various benchmarks for long video understanding with reduced time and memory consumption. The model code and weights are available at https://huggingface.co/Tao-tse/Sophia.

</details>

---

## 87. Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal LongCoTReasoning

- [ ] Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal LongCoTReasoning | https://aclanthology.org/2025.acl-long.257/

- **Link**: https://aclanthology.org/2025.acl-long.257/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Language Models (LLMs) have demonstrated enhanced reasoning capabilities, evolving from Chain-of-Thought (CoT) prompting to advanced, product-oriented solutions like OpenAI o1. During our re-implementation of this model, we noticed that in multimodal tasks requiring visual input (e.g., geometry problems), Multimodal LLMs (MLLMs) struggle to maintain focus on the visual information, in other words, MLLMs suffer from a gradual decline in attention to visual information as reasoning progresses, causing text-over-relied outputs. To investigate this, we ablate image inputs during long-chain reasoning. Concretely, we truncate the reasoning process midway, then re-complete the reasoning process with the input image removed. We observe only a ~2 accuracy drop on MathVista’s test-hard subset, revealing the model’s textual outputs dominate the following reasoning process. Motivated by this, we propose Take-along Visual Conditioning (TVC), a strategy that shifts image input to critical reasoning stages and compresses redundant visual tokens via dynamic pruning. This methodology helps the model retain attention to the visual components throughout the reasoning. Our approach achieves state-of-the-art performance on average across five mathematical reasoning benchmarks (+3.4% vs previous sota), demonstrating the effectiveness of TVC in enhancing multimodal reasoning systems. The project page is available athttps://sun-hailong.github.io/projects/TVC.

</details>

---

## 88. OS-Genesis: AutomatingGUIAgent Trajectory Construction via Reverse Task Synthesis

- [ ] OS-Genesis: AutomatingGUIAgent Trajectory Construction via Reverse Task Synthesis | https://aclanthology.org/2025.acl-long.277/

- **Link**: https://aclanthology.org/2025.acl-long.277/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interface (GUI) agents powered by Vision-Language Models (VLMs) have demonstrated human-like computer control capability. Despite their utility in advancing digital automation, the development of such agents faces a critical bottleneck: collecting high-quality trajectory data for training. Common practices for collecting such data rely on human supervision or synthetic data generation through executing pre-defined tasks, which are either resource-intensive or unable to guarantee data quality. Further, these approaches exhibit significant gaps between the generated data and online environments, alongside limited data diversity. To address this issue, we introduce OS-Genesis, a novel GUI data synthesis pipeline that overcomes the challenges above. Unlike prior methods that rely on preset tasks, OS-Genesis reverse engineers the GUI trajectory construction process. Agents first perceive environments and perform step-level interactions, then retrospectively derive high-quality tasks to enable trajectory-level exploration. A trajectory reward model is then employed to ensure the quality of the generated trajectories. We demonstrate that training GUI agents with OS-Genesis significantly improves their performance on highly challenging online benchmarks. In-depth analysis further validates OS-Genesis’s cost-effectiveness and its superior data quality and diversity compared to existing synthesis methods.

</details>

---

## 89. GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge forGUIAgent

- [ ] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge forGUIAgent | https://aclanthology.org/2025.acl-long.282/

- **Link**: https://aclanthology.org/2025.acl-long.282/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

GUI automation faces critical challenges in dynamic environments. MLLMs suffer from two key issues: misinterpreting UI components and outdated knowledge. Traditional fine-tuning methods are costly for app-specific knowledge updates. We propose GUI-explorer, a training-free GUI agent that incorporates two fundamental mechanisms:(1) Autonomous Exploration of Function-aware Trajectory. To comprehensively cover all application functionalities, we design aFunction-aware Task Goal Generatorthat automatically constructs exploration goals by analyzing GUI structural information (e.g., screenshots and activity hierarchies). This enables systematic exploration to collect diverse trajectories.(2) Unsupervised Mining of Transition-aware Knowledge. To establish precise screen-operation logic, we develop aTransition-aware Knowledge Extractorthat extracts effective screen-operation logic through unsupervised analysis the state transition of structured interaction triples (observation, action, outcome). This eliminates the need for human involvement in knowledge extraction. With a task success rate of 53.7% on SPA-Bench and 47.4% on AndroidWorld, GUI-explorer shows significant improvements over SOTA agents. It requires no parameter updates for new apps. GUI-explorer is open-sourced and publicly available at https://github.com/JiuTian-VL/GUI-explorer.

</details>

---

## 90. Enhancing Multimodal Continual Instruction Tuning withBranchLoRA

- [ ] Enhancing Multimodal Continual Instruction Tuning withBranchLoRA | https://aclanthology.org/2025.acl-long.287/

- **Link**: https://aclanthology.org/2025.acl-long.287/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Continual Instruction Tuning (MCIT) aims to finetune Multimodal Large Language Models (MLLMs) to continually align with human intent across sequential tasks. Existing approaches often rely on the Mixture-of-Experts (MoE) LoRA framework to preserve previous instruction alignments. However, these methods are prone to Catastrophic Forgetting (CF), as they aggregate all LoRA blocks via simple summation, which compromises performance over time. In this paper, we identify a critical parameter inefficiency in the MoELoRA framework within the MCIT context. Based on this insight, we propose BranchLoRA, an asymmetric framework to enhance both efficiency and performance. To mitigate CF, we introduce a flexible tuning-freezing mechanism within BranchLoRA, enabling branches to specialize in intra-task knowledge while fostering inter-task collaboration. Moreover, we incrementally incorporate task-specific routers to ensure an optimal branch distribution over time, rather than favoring the most recent task. To streamline inference, we introduce a task selector that automatically routes test inputs to the appropriate router without requiring task identity. Extensive experiments on the latest MCIT benchmark demonstrate that BranchLoRA significantly outperforms MoELoRA and maintains its superiority across various MLLM sizes.

</details>

---

## 91. mPLUG-DocOwl2: High-resolution Compressing forOCR-free Multi-page Document Understanding

- [ ] mPLUG-DocOwl2: High-resolution Compressing forOCR-free Multi-page Document Understanding | https://aclanthology.org/2025.acl-long.291/

- **Link**: https://aclanthology.org/2025.acl-long.291/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodel Large Language Models(MLLMs) have achieved promising OCR-free Document Understanding performance by increasing the supported resolution of document images. However, this comes at the cost of generating thousands of visual tokens for a single document image, leading to excessive GPU memory and slower inference times, particularly in multi-page document comprehension. In this work, to address these challenges, we propose a High-resolution DocCompressor module to compress each high-resolution document image into 324 tokens, guided by low-resolution global visual features. With this compression module, to strengthen multi-page document comprehension ability and balance both token efficiency and question-answering performance, we develop the DocOwl2 under a three-stage training framework: Single-image Pretraining, Multi-image Continue-pretraining, and Multi-task Finetuning. DocOwl2 sets a new state-of-the-art across multi-page document understanding benchmarks and reduces first token latency by more than 50%. Compared to single-image MLLMs trained on similar data, our DocOwl2 achieves comparable single-page understanding performance with less than 20% of the visual tokens. Our codes, models, and data will be publicly available.

</details>

---

## 92. Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models

- [ ] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models | https://aclanthology.org/2025.acl-long.295/

- **Link**: https://aclanthology.org/2025.acl-long.295/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Generative models such as Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) trained on massive datasets can lead them to memorize and inadvertently reveal sensitive information, raising ethical and privacy concerns. While some prior works have explored this issue in the context of LLMs, it presents a unique challenge for MLLMs due to the entangled nature of knowledge across modalities, making comprehensive unlearning more difficult. To address this challenge, we propose Modality Aware Neuron Unlearning (MANU), a novel unlearning framework for MLLMs designed to selectively clip neurons based on their relative importance to the targeted forget data, curated for different modalities. Specifically, MANU consists of two stages: important neuron selection and selective pruning. The first stage identifies and collects the most influential neurons across modalities relative to the targeted forget knowledge, while the second stage is dedicated to pruning those selected neurons. MANU effectively isolates and removes the neurons that contribute most to the forget data within each modality, while preserving the integrity of retained knowledge. Our experiments conducted across various MLLM architectures illustrate that MANU can achieve a more balanced and comprehensive unlearning in each modality without largely affecting the overall model utility.

</details>

---

## 93. Can Multimodal Large Language Models Understand Spatial Relations?

- [ ] Can Multimodal Large Language Models Understand Spatial Relations? | https://aclanthology.org/2025.acl-long.31/

- **Link**: https://aclanthology.org/2025.acl-long.31/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Spatial relation reasoning is a crucial task for multimodal large language models (MLLMs) to understand the objective world. However, current benchmarks have issues like relying on bounding boxes, ignoring perspective substitutions, or allowing questions to be answered using only the model’s prior knowledge without image understanding. To address these issues, we introduce SpatialMQA, a human-annotated spatial relation reasoning benchmark based on COCO2017, which enables MLLMs to focus more on understanding images in the objective world. To ensure data quality, we design a well-tailored annotation procedure, resulting in SpatialMQA consisting of 5,392 samples. Based on this benchmark, a series of closed- and open-source MLLMs are implemented and the results indicate that the current state-of-the-art MLLM achieves only 48.14% accuracy, far below the human-level accuracy of 98.40%. Extensive experimental analyses are also conducted, suggesting the future research directions. The benchmark and codes are available at https://huggingface.co/datasets/liuziyan/SpatialMQA.

</details>

---

## 94. Evaluating Multimodal Large Language Models on Video Captioning viaMonteCarlo Tree Search

- [ ] Evaluating Multimodal Large Language Models on Video Captioning viaMonteCarlo Tree Search | https://aclanthology.org/2025.acl-long.323/

- **Link**: https://aclanthology.org/2025.acl-long.323/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video captioning can be used to assess the video understanding capabilities of Multimodal Large Language Models (MLLMs).However, existing benchmarks and evaluation protocols suffer from crucial issues, such as inadequate or homogeneous creation of key points, exorbitant cost of data creation, and limited evaluation scopes. To address these issues, we propose an automatic framework, named AutoCaption, which leverages Monte Carlo Tree Search (MCTS) to construct numerous and diverse descriptive sentences (i.e., key points) that thoroughly represent video content in an iterative way. This iterative captioning strategy enables the continuous enhancement of video details such as actions, objects’ attributes, environment details, etc. We apply AutoCaption to curate MCTS-VCB, a fine-grained video caption benchmark covering video details, thereby enabling a comprehensive evaluation of MLLMs on the video captioning task. We evaluate more than 20 open- and closed-source MLLMs of varying sizes on MCTS-VCB. Results show that MCTS-VCB can effectively and comprehensively evaluate the video captioning capability, with Gemini-1.5-Pro achieving the highest F1 score of 71.2. Interestingly, we fine-tune InternVL2.5-8B with the AutoCaption-generated data, which helps the model achieve an overall improvement of 25.0% on MCTS-VCB and 16.3% on DREAM-1K, further demonstrating the effectiveness of AutoCaption. The code and data are available athttps://github.com/tjunlp-lab/MCTS-VCB.

</details>

---

## 95. AlignMMBench: EvaluatingChinese Multimodal Alignment in Large Vision-Language Models

- [ ] AlignMMBench: EvaluatingChinese Multimodal Alignment in Large Vision-Language Models | https://aclanthology.org/2025.acl-long.327/

- **Link**: https://aclanthology.org/2025.acl-long.327/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Evaluating the alignment capabilities of large Vision-Language Models (VLMs) is essential for determining their effectiveness as helpful assistants. However, existing benchmarks primarily focus on basic abilities using nonverbal methods, such as yes-no and multiple-choice questions. In this paper, we address this gap by introducing AlignMMBench, which provides more nuanced evaluations of alignment capabilities and is the first benchmark specifically designed for Chinese visual contexts. This benchmark is meticulously curated from real-world scenarios and internet sources, encompassing thirteen specific tasks across three categories, and includes both single-turn and multi-turn dialogue scenarios. Incorporating a prompt rewrite strategy, AlignMMBench encompasses 1,054 images and 4,978 question-answer pairs. To facilitate the evaluation pipeline, we develop CritiqueVLM, a rule-calibrated evaluator that exceeds GPT-4’s evaluation ability. Additionally, we measure the “alignment score”, a quantitative metric designed to assess the robustness and stability of models across diverse prompts. Finally, we evaluate the performance of representative VLMs on AlignMMBench, offering insights into the capabilities and limitations of different VLM architectures. The evaluation code and data are available at https://github.com/THUDM/AlignMMBench.

</details>

---

## 96. TrimLLM: Progressive Layer Dropping for Domain-SpecificLLMs

- [ ] TrimLLM: Progressive Layer Dropping for Domain-SpecificLLMs | https://aclanthology.org/2025.acl-long.33/

- **Link**: https://aclanthology.org/2025.acl-long.33/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Specializing large language models (LLMs) for local deployment in domain-specific use cases is necessary for strong performance while meeting latency and privacy constraints. However, conventional task-specific adaptation approaches do not show simultaneous memory saving and inference speedup at deployment time. Practical compression techniques like quantization and pruning require dedicated hardware or kernel support to achieve measured inference speedup. We develop TrimLLM based on the layer-wise specialization phenomenon we empirically observed and verified on contemporary LLMs. TrimLLM reduces the depth of LLMs via progressive layer dropping. We show it retains LLMs’ capacity in specific domains and achieves inference speedup irrespective of hardware and deep learning frameworks. We evaluated TrimLLM on LLMs of various sizes for inference; models adapted on medical, legal, and financial datasets all demonstrate2.1 - 5.7×inference speedup on consumer GPUs and up to3.1×speedup on A100 when compared to state-of-the-art model compression algorithms, with no loss in accuracy at50∼ 60% model compression ratio.

</details>

---

## 97. DRAG: DistillingRAGforSLMs fromLLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation

- [ ] DRAG: DistillingRAGforSLMs fromLLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation | https://aclanthology.org/2025.acl-long.358/

- **Link**: https://aclanthology.org/2025.acl-long.358/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-Augmented Generation (RAG) methods have proven highly effective for tasks requiring factual consistency and robust knowledge retrieval. However, large-scale RAG systems consume significant computational resources and are prone to generating “hallucinated” content from Humans. In this work, we introduce DRAG, a novel framework for distilling RAG knowledge from large-scale Language Models (LLMs) into small LMs (SLMs). Our approach leverages evidence- and knowledge graph–based distillation, ensuring that the distilled model retains critical factual knowledge while significantly reducing model size and computational cost. By aligning the smaller model’s predictions with a structured knowledge graph and ranked evidence, DRAG effectively mitigates hallucinations and improves factual accuracy. We further present a case demonstrating how our framework mitigates user privacy risks and introduce a corresponding benchmark. Experimental evaluations on multiple benchmarks demonstrate that our method outperforms the prior competitive RAG methods like MiniRAG for SLMs by up to 27.7% using the same models, preserving high-level efficiency and reliability. With DRAG, we provide a practical and resource-efficient roadmap to deploying enhanced retrieval and generation capabilities in small-size LLMs. Code is available at https://github.com/VILA-Lab/DRAG.

</details>

---

## 98. ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation

- [ ] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation | https://aclanthology.org/2025.acl-long.363/

- **Link**: https://aclanthology.org/2025.acl-long.363/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in chart understanding tasks. However, interpreting charts with textual descriptions often leads to information loss, as it fails to fully capture the dense information embedded in charts. In contrast, parsing charts into code provides lossless representations that can effectively contain all critical details. Although existing open-source MLLMs have achieved success in chart understanding tasks, they still face two major challenges when applied to chart-to-code tasks: (1) Low executability and poor restoration of chart details in the generated code and (2) Lack of large-scale and diverse training data. To address these challenges, we proposeChartCoder, the first dedicated chart-to-code MLLM, which leverages Code LLMs as the language backbone to enhance the executability of the generated code. Furthermore, we introduceChart2Code-160k, the first large-scale and diverse dataset for chart-to-code generation, and propose theSnippet-of-Thought (SoT)method, which transforms direct chart-to-code generation data into step-by-step generation. Experiments demonstrate that ChartCoder, with only 7B parameters, surpasses existing open-source MLLMs on chart-to-code benchmarks, achieving superior chart restoration and code excitability. Our code is available athttps://github.com/thunlp/ChartCoder.

</details>

---

## 99. OSAgents: A Survey onMLLM-based Agents for Computer, Phone and Browser Use

- [ ] OSAgents: A Survey onMLLM-based Agents for Computer, Phone and Browser Use | https://aclanthology.org/2025.acl-long.369/

- **Link**: https://aclanthology.org/2025.acl-long.369/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The dream to create AI assistants as capable and versatile as the fictional J.A.R.V.I.S from Iron Man has long captivated imaginations. With the evolution of multi-modal large language models ((M)LLMs), this dream is closer to reality, as (M)LLM-based Agents using computers, mobile phones and web browsers by operating within the environments and interfaces (e.g., Graphical User Interface (GUI) and Command Line Interface (CLI)) provided by operating systems (OS) to automate tasks have significantly advanced. This paper presents a comprehensive survey on these advanced agents, designated as OS Agents. We begin by elucidating the fundamentals of OS Agents, exploring their key components and capabilities. We then examine methodologies for constructing OS Agents, focusing on domain-specific foundation models and agent frameworks. A detailed review of evaluation metrics and benchmarks highlights how OS Agents are assessed across diverse platforms and tasks. Finally, we discuss current challenges and identify promising directions for future research. An open-source GitHub repository is maintained as a dynamic resource to foster further innovation in this field.

</details>

---

## 100. VLM2-Bench: A Closer Look at How WellVLMs Implicitly Link Explicit Matching Visual Cues

- [ ] VLM2-Bench: A Closer Look at How WellVLMs Implicitly Link Explicit Matching Visual Cues | https://aclanthology.org/2025.acl-long.372/

- **Link**: https://aclanthology.org/2025.acl-long.372/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visually linking matching cues is a crucial ability in daily life, such as identifying the same person in multiple photos based on their cues, even without knowing who they are. Despite the extensive knowledge that vision-language models (VLMs) possess, it remains largely unexplored whether they are capable of performing this fundamental task. To address this, we introduce VLM2-Bench, a benchmark designed to assess whether VLMs can Visually Link Matching cues, with 9 subtasks and over 3,000 test cases. Comprehensive evaluation across twelve VLMs, along with further analysis of various language-side and vision-side prompting methods, leads to a total of eight key findings. We identify critical challenges in models’ ability to link visual cues, highlighting a significant performance gap. Based on these insights, we advocate for (i) enhancing core visual capabilities to improve adaptability and reduce reliance on prior knowledge, (ii) establishing clearer principles for integrating language-based reasoning in vision-centric tasks to prevent unnecessary biases, and (iii) shifting vision-text training paradigms toward fostering models’ ability to independently structure and infer relationships among visual cues.

</details>

---

## 101. ActiView: Evaluating Active Perception Ability for Multimodal Large Language Models

- [ ] ActiView: Evaluating Active Perception Ability for Multimodal Large Language Models | https://aclanthology.org/2025.acl-long.376/

- **Link**: https://aclanthology.org/2025.acl-long.376/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Active perception, a crucial human capability, involves setting a goal based on the current understanding of the environment and performing actions to achieve that goal. Despite significant efforts in evaluating Multimodal Large Language Models (MLLMs), active perception has been largely overlooked. To address this gap, we propose a novel benchmark named ActiView to evaluate active perception in MLLMs. We focus on a specialized form of Visual Question Answering (VQA) that eases and quantifies the evaluation yet challenging for existing MLLMs. Meanwhile, intermediate reasoning behaviors of models are also discussed. Given an image, we restrict the perceptual field of a model, requiring it to actively zoom or shift its perceptual field based on reasoning to answer the question successfully. We conduct extensive evaluation over 30 models, including proprietary and open-source models, and observe that restricted perceptual fields play a significant role in enabling active perception. Results reveal a significant gap in the active perception capability of MLLMs, indicating that this area deserves more attention. We hope that ActiView could help develop methods for MLLMs to understand multimodal inputs in more natural and holistic ways.

</details>

---

## 102. A Text is Worth Several Tokens: Text Embedding fromLLMs Secretly Aligns Well with The Key Tokens

- [ ] A Text is Worth Several Tokens: Text Embedding fromLLMs Secretly Aligns Well with The Key Tokens | https://aclanthology.org/2025.acl-long.379/

- **Link**: https://aclanthology.org/2025.acl-long.379/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text embeddings from large language models (LLMs) have achieved excellent results in tasks such as information retrieval, semantic textual similarity, etc. In this work, we show an interesting finding: when feeding a text into the LLM-based embedder, the obtained text embedding will be able to be aligned with the key tokens in the input text. We first fully analyze this phenomenon on eight LLM-based embedders and show that this phenomenon is universal and is not affected by model architecture, training strategy, and embedding method. With a deeper analysis, we find that the main change in embedding space between these embedders and their LLM backbones is in the first principal component. By adjusting the first principal component, we can align text embedding with the key tokens. Finally, we give several examples to demonstrate the vast application potential of this finding: (1) we propose a simple and practical sparse retrieval method based on the aligned tokens, which can achieve 80% of the dense retrieval effect of the same model while reducing the computation significantly; (2) we show that our findings provide a novel perspective to help understand novel technologies (e.g., instruction-following embedding) and fuzzy concepts (e.g., semantic relatedness vs. similarity) in this field.

</details>

---

## 103. AXIS: Efficient Human-Agent-Computer Interaction withAPI-FirstLLM-Based Agents

- [ ] AXIS: Efficient Human-Agent-Computer Interaction withAPI-FirstLLM-Based Agents | https://aclanthology.org/2025.acl-long.381/

- **Link**: https://aclanthology.org/2025.acl-long.381/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have enabled LLM-based agents to directly interact with application user interfaces (UIs), enhancing agents’ performance in complex tasks. However, these agents often suffer from high latency and low reliability due to the extensive sequential UI interactions. To address this issue, we propose AXIS, a novel LLM-based agents framework that prioritize actions through application programming interfaces (APIs) over UI actions. This framework also facilitates the creation and expansion of APIs through automated exploration of applications. Our experiments on Microsoft Word demonstrate that AXIS reduces task completion time by 65%-70% and cognitive workload by 38%-53%, while maintaining accuracy of 97%-98% compared to humans. Our work contributes to a new human-agent-computer interaction (HACI) framework and explores a fresh UI design principle for application providers to turn applications into agents in the era of LLMs, paving the way towards an agent-centric operating system (Agent OS). The code and dataset will be available at https://aka.ms/haci_axis.

</details>

---

## 104. VQAGuider: Guiding Multimodal Large Language Models to Answer Complex Video Questions

- [ ] VQAGuider: Guiding Multimodal Large Language Models to Answer Complex Video Questions | https://aclanthology.org/2025.acl-long.385/

- **Link**: https://aclanthology.org/2025.acl-long.385/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Complex video question-answering (VQA) requires in-depth understanding of video contents including object and action recognition as well as video classification and summarization, which exhibits great potential in emerging applications in education and entertainment, etc. Multimodal large language models (MLLMs) may accomplish this task by grasping the intention of a question and decomposing it to a series of visual recognition sub-tasks to find out the answer with the help of an agent. To tackle this task, we first collect a new dedicated Complex VQA dataset named CVQA and then propose VQAGuider, an innovative framework planning a few atomic visual recognition tools by video-related API matching. VQAGuider facilitates a deep engagement with video content and precise responses to complex video-related questions by MLLMs, which is beyond aligning visual and language features for simple VQA tasks. Our experiments demonstrate VQAGuider is capable of navigating the complex VQA tasks by MLLMs and improves the accuracy by 29.6% and 17.2% on CVQA and the existing VQA datasets, respectively, highlighting its potential in advancing MLLMs’s capabilities in video understanding.

</details>

---

## 105. SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data

- [ ] SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data | https://aclanthology.org/2025.acl-long.387/

- **Link**: https://aclanthology.org/2025.acl-long.387/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) work well in tasks ranging from image captioning to visual question answering (VQA), yet they struggle with spatial reasoning, a key skill for understanding our physical world that humans excel at. We find that spatial relations are generally rare in widely used VL datasets, with only a few being well represented, while most form a long tail of underrepresented relations. This gap leaves VLMs ill-equipped to handle diverse spatial relationships. To bridge it, we construct a synthetic VQA dataset focused on spatial reasoning generated from hyper-detailed image descriptions in Localized Narratives, DOCCI, and PixMo-Cap. Our dataset consists of 455k samples containing 3.4 million QA pairs. Trained on this dataset, our Spatial-Reasoning Enhanced (SpaRE) VLMs show strong improvements on spatial reasoning benchmarks, achieving up to a 49% performance gain on the What’s Up benchmark, while maintaining strong results on general tasks. Our work narrows the gap between human and VLM spatial reasoning and makes VLMs more capable in real-world tasks such as robotics and navigation. We plan to share our code and dataset in due course.

</details>

---

## 106. R2-MultiOmnia: Leading Multilingual Multimodal Reasoning via Self-Training

- [ ] R2-MultiOmnia: Leading Multilingual Multimodal Reasoning via Self-Training | https://aclanthology.org/2025.acl-long.402/

- **Link**: https://aclanthology.org/2025.acl-long.402/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reasoning is an intricate process that transcends both language and vision; yet, despite its inherently modality-agnostic nature, develop-ing effective multilingual and multimodal reasoning capabilities remains a substantial challenge for Multimodal Large Language Models (MLLMs). They struggle to activate complex reasoning behaviours, delivering step-wise explanation, questioning and reflection, particularly in multilingual settings where high-quality supervision across languages is lacking. Recent works have introduced eclectic strategies to enhance MLLMs’ reasoning; however, they remain related to a single language.To make MLLMs’ reasoning capabilities aligned among languages and improve modality performances, we propose R2-MultiOmnia, a modular approach that instructs the models to abstract key elements of the reasoning process and then refine reasoning trajectories via self-correction. Specifically, we instruct the models producing multimodal synthetic resources by bridging modalities and then self-improving their capabilities. To stabilise learning and the reasoning processes structure, we propose Curriculum Learning Reasoning Stabilisation with structured output rewards to gradually refine the models’ capabilities to learn and deliver robust reasoning processes. Experiments show that R2-MultiOmnia improves multimodal reasoning, gets aligned performances among the languages approaching strong models.

</details>

---

## 107. VLSBench: Unveiling Visual Leakage in Multimodal Safety

- [ ] VLSBench: Unveiling Visual Leakage in Multimodal Safety | https://aclanthology.org/2025.acl-long.405/

- **Link**: https://aclanthology.org/2025.acl-long.405/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Safety concerns of Multimodal large language models (MLLMs) have gradually become an important problem in various applications. Surprisingly, previous works indicate a counterintuitive phenomenon that using textual unlearning to align MLLMs achieves comparable safety performances with MLLMs aligned with image-text pairs. To explain such a phenomenon, we discover aVisualSafetyInformationLeakage(VSIL)problem in existing multimodal safety benchmarks,i.e., the potentially risky content in the image has been revealed in the textual query. Thus, MLLMs can easily refuse these sensitive image-text pairs according to textual queries only, leading tounreliable cross-modality safety evaluation of MLLMs. We also conduct a further comparison experiment between textual alignment and multimodal alignment to highlight this drawback. To this end, we constructVisualLeaklessSafetyBench(VLSBench)with 2.2k image-text pairs through an automated data pipeline. Experimental results indicate that VLSBench poses a significant challenge to both open-source and close-source MLLMs,i.e., LLaVA, Qwen2-VL and GPT-4o. Besides, we empirically compare textual and multimodal alignment methods on VLSBench and find that textual alignment is effective enough for multimodal safety scenarios with VSIL, while multimodal alignment is preferable for safety scenarios without VSIL.

</details>

---

## 108. Revisiting ClassicalChinese Event Extraction with Ancient Literature Information

- [ ] Revisiting ClassicalChinese Event Extraction with Ancient Literature Information | https://aclanthology.org/2025.acl-long.414/

- **Link**: https://aclanthology.org/2025.acl-long.414/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The research on classical Chinese event extraction trends to directly graft the complex modeling from English or modern Chinese works, neglecting the utilization of the unique characteristic of this language. We argue that, compared with grafting the sophisticated methods from other languages, focusing on classical Chinese’s inimitable source of __Ancient Literature__ could provide us with extra and comprehensive semantics in event extraction. Motivated by this, we propose a Literary Vision-Language Model (VLM) for classical Chinese event extraction, integrating with literature annotations, historical background and character glyph to capture the inner- and outer-context information from the sequence. Extensive experiments build a new state-of-the-art performance in the GuwenEE, CHED datasets, which underscores the effectiveness of our proposed VLM, and more importantly, these unique features can be obtained precisely at nearly zero cost.

</details>

---

## 109. Attacking Vision-Language Computer Agents via Pop-ups

- [ ] Attacking Vision-Language Computer Agents via Pop-ups | https://aclanthology.org/2025.acl-long.411/

- **Link**: https://aclanthology.org/2025.acl-long.411/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing their tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques, such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack. Code is available at [this link](https://github.com/SALT-NLP/PopupAttack).

</details>

---

## 110. Focus on What Matters: Enhancing Medical Vision-Language Models with Automatic Attention Alignment Tuning

- [ ] Focus on What Matters: Enhancing Medical Vision-Language Models with Automatic Attention Alignment Tuning | https://aclanthology.org/2025.acl-long.460/

- **Link**: https://aclanthology.org/2025.acl-long.460/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical Large Vision-Language Models (Med-LVLMs) often exhibit suboptimal attention distribution on visual inputs, leading to hallucinated or inaccurate outputs. Existing methods primarily rely on inference-time interventions, which are limited in attention adaptation or require additional supervision. To address this, we propose A3Tune, a novel fine-tuning framework for Automatic Attention Alignment Tuning. ATune leverages zero-shot weak labels from SAM, refines them into prompt-aware labels using BioMedCLIP, and then selectively modifies visually-critical attention heads to improve alignment while minimizing interference. Additionally, we introduce a A3MoE module, enabling adaptive parameter selection for attention tuning across diverse prompts and images. Extensive experiments on medical VQA and report generation benchmarks show that A3Tune outperforms state-of-the-art baselines, achieving enhanced attention distributions and performance in Med-LVLMs.

</details>

---

## 111. Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition in Social Media Contexts

- [ ] Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition in Social Media Contexts | https://aclanthology.org/2025.acl-long.472/

- **Link**: https://aclanthology.org/2025.acl-long.472/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The recent progress in Vision-Language Models (VLMs) has broadened the scope of multimodal applications. However, evaluations often remain limited to functional tasks, neglecting abstract dimensions such as personality traits and human values. To address this gap, we introduce Value-Spectrum, a novel Visual Question Answering (VQA) benchmark aimed at assessing VLMs based on Schwartz’s value dimensions that capture core human values guiding people’s preferences and actions. We design a VLM agent pipeline to simulate video browsing and construct a vector database comprising over 50,000 short videos from TikTok, YouTube Shorts, and Instagram Reels. These videos span multiple months and cover diverse topics, including family, health, hobbies, society, technology, etc. Benchmarking on Value-Spectrum highlights notable variations in how VLMs handle value-oriented content. Beyond identifying VLMs’ intrinsic preferences, we also explore the ability of VLM agents to adopt specific personas when explicitly prompted, revealing insights into the adaptability of the model in role-playing scenarios. These findings highlight the potential of Value-Spectrum as a comprehensive evaluation set for tracking VLM preferences in value-based tasks and abilities to simulate diverse personas. The complete code and data are available at https://github.com/Jeremyyny/Value-Spectrum.

</details>

---

## 112. AligningVLMAssistants with Personalized Situated Cognition

- [ ] AligningVLMAssistants with Personalized Situated Cognition | https://aclanthology.org/2025.acl-long.484/

- **Link**: https://aclanthology.org/2025.acl-long.484/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) aligned with general human objectives, such as being harmless and hallucination-free, have become valuable assistants of humans in managing visual tasks. However, people with diversified backgrounds have different cognition even in the same situation. Consequently, they may have personalized expectations for VLM assistants. This highlights the urgent need to align VLM assistants with personalized situated cognition for real-world assistance. To study this problem, we first simplify it by characterizing individuals based on the sociological concept of Role-Set. Then, we propose to evaluate the individuals’ actions to examine whether the personalized alignment is achieved. Further, we construct a benchmark named PCogAlignBench, which includes 18k instances and 20 individuals with different Role-Sets. Finally, we present a framework called PCogAlign, which constructs a cognition-aware and action-based reward model for personalized alignment. Experimental results and human evaluations demonstrate the reliability of the PCogAlignBench and the effectiveness of our proposed PCogAlign. We will open-source the constructed benchmark and code after being accepted.

</details>

---

## 113. CADReview: Automatically ReviewingCADPrograms with Error Detection and Correction

- [ ] CADReview: Automatically ReviewingCADPrograms with Error Detection and Correction | https://aclanthology.org/2025.acl-long.489/

- **Link**: https://aclanthology.org/2025.acl-long.489/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Computer-aided design (CAD) is crucial in prototyping 3D objects through geometric instructions (i.e., CAD programs). In practical design workflows, designers often engage in time-consuming reviews and refinements of these prototypes by comparing them with reference images. To bridge this gap, we introduce the CAD review task to automatically detect and correct potential errors, ensuring consistency between the constructed 3D objects and reference images. However, recent advanced multimodal large language models (MLLMs) struggle to recognize multiple geometric components and perform spatial geometric operations within the CAD program, leading to inaccurate reviews. In this paper, we propose the CAD program repairer (ReCAD) framework to effectively detect program errors and provide helpful feedback on error correction. Additionally, we create a dataset, CADReview, consisting of over 20K program-image pairs, with diverse errors for the CAD review task. Extensive experiments demonstrate that our ReCAD significantly outperforms existing MLLMs, which shows great potential in design applications.

</details>

---

## 114. PunchBench: BenchmarkingMLLMs in Multimodal Punchline Comprehension

- [ ] PunchBench: BenchmarkingMLLMs in Multimodal Punchline Comprehension | https://aclanthology.org/2025.acl-long.49/

- **Link**: https://aclanthology.org/2025.acl-long.49/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal punchlines, which involve humor or sarcasm conveyed in image-caption pairs, are a popular way of communication on online multimedia platforms. With the rapid development of multimodal large language models (MLLMs), it is essential to assess their ability to effectively comprehend these punchlines. However, existing benchmarks on punchline comprehension suffer from three major limitations: 1) language shortcuts that allow models to solely rely on text, 2) lack of question diversity, and 3) narrow focus on a specific domain of multimodal content (e.g., cartoon). To address these limitations, we introduce a multimodal **Punch**line comprehension **Bench**mark, named **PunchBench**, which is tailored for accurate and comprehensive evaluation of punchline comprehension. To enhance the evaluation accuracy, we generate synonymous and antonymous captions by modifying original captions, which mitigates the impact of shortcuts in the captions. To provide a comprehensive evaluation, PunchBench incorporates diverse question formats and image-captions from various domains. On this basis, we conduct extensive evaluations and reveal a significant gap between state-of-the-art MLLMs and humans in punchline comprehension. To improve punchline comprehension, we propose Simple-to-Complex Chain-of-Question (SC-CoQ) strategy, enabling the models to incrementally address complicated questions by first mastering simple ones. SC-CoQ effectively enhances the performance of various MLLMs on PunchBench, surpassing in-context learning and chain-of-thought.

</details>

---

## 115. Exploring How GenerativeMLLMs Perceive More ThanCLIPwith the Same Vision Encoder

- [ ] Exploring How GenerativeMLLMs Perceive More ThanCLIPwith the Same Vision Encoder | https://aclanthology.org/2025.acl-long.499/

- **Link**: https://aclanthology.org/2025.acl-long.499/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent research has shown that CLIP models struggle with visual reasoning tasks that require grounding compositionality, understanding spatial relationships, or capturing fine-grained details. One natural hypothesis is that the CLIP vision encoder does not embed essential information for these tasks. However, we find that this is not always the case: The encoder gathers query-relevant visual information, while CLIP fails to extract it. In particular, we show that another branch of Vision-Language Models (VLMs), Generative Multimodal Large Language Models (MLLMs), achieve significantly higher accuracy than CLIP in many of these tasks using the *same* vision encoder and weights, indicating that these Generative MLLMs *perceive more*—as they extract and utilize visual information more effectively. We conduct a series of controlled experiments and reveal that their success is attributed to multiple key design choices, including patch tokens, position embeddings, and prompt-based weighting. On the other hand, enhancing the training data alone or applying a stronger text encoder does not suffice to solve the task, and additional text tokens offer little benefit. Interestingly, we find that fine-grained visual reasoning is not exclusive to generative models trained by an autoregressive loss: When converted into CLIP-like encoders by contrastive finetuning, these MLLMs still outperform CLIP under the same cosine similarity-based evaluation protocol. Our study highlights the importance of VLM architectural choices and suggests directions for improving the performance of CLIP-like contrastive VLMs.

</details>

---

## 116. EfficientQAT: Efficient Quantization-Aware Training for Large Language Models

- [ ] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models | https://aclanthology.org/2025.acl-long.498/

- **Link**: https://aclanthology.org/2025.acl-long.498/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) are crucial in modern natural language processing and artificial intelligence. However, they face challenges in managing their significant memory requirements. Although quantization-aware training (QAT) offers a solution by reducing memory consumption through low-bit representations with minimal accuracy loss, it is impractical due to substantial training resources. To address this, we propose Efficient Quantization-Aware Training (EfficientQAT), a more feasible QAT algorithm. EfficientQAT involves two consecutive phases: Block-wise training of all parameters (Block-AP) and end-to-end training of quantization parameters (E2E-QP). To the best of our knowledge, Block-AP is the first method to enable direct training of all parameters in a block-wise manner, reducing accuracy loss in low-bit scenarios by enhancing the solution space during optimization. E2E-QP then trains only the quantization parameters (step sizes) end-to-end, further improving the performance of quantized models by considering interactions among all sub-modules. Extensive experiments demonstrate that EfficientQAT outperforms previous quantization methods across a range of models, including base LLMs, instruction-tuned LLMs, and multimodal LLMs, with scales from 7B to 70B parameters at various quantization bits. For instance, EfficientQAT obtains a 2-bit Llama-2-70B model on a single A100-80GB GPU in 41 hours, with less than 3 points accuracy degradation compared to the full precision (69.48 vs. 72.41). Code is available at https://github.com/OpenGVLab/EfficientQAT.

</details>

---

## 117. HAIC: Improving Human Action Understanding and Generation with Better Captions for Multi-modal Large Language Models

- [ ] HAIC: Improving Human Action Understanding and Generation with Better Captions for Multi-modal Large Language Models | https://aclanthology.org/2025.acl-long.501/

- **Link**: https://aclanthology.org/2025.acl-long.501/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent Multi-modal Large Language Models (MLLMs) have made great progress in video understanding. However, their performance on videos involving human actions is still limited by the lack of high-quality data. To address this, we introduce a two-stage data annotation pipeline. First, we design strategies to accumulate videos featuring clear human actions from the Internet. Second, videos are annotated in a standardized caption format that uses human attributes to distinguish individuals and chronologically details their actions and interactions. Through this pipeline, we curate two datasets, namely HAICTrain and HAICBench. **HAICTrain** comprises 126K video-caption pairs generated by Gemini-Pro and verified for training purposes. Meanwhile, **HAICBench** includes 412 manually annotated video-caption pairs and 2,000 QA pairs, for a comprehensive evaluation of human action understanding. Experimental results demonstrate that training with HAICTrain not only significantly enhances human understanding abilities across 4 benchmarks, but can also improve text-to-video generation results. Both the HAICTrain and HAICBench will be made open-source to facilitate further research.

</details>

---

## 118. Uni-Retrieval: A Multi-Style Retrieval Framework forSTEM’s Education

- [ ] Uni-Retrieval: A Multi-Style Retrieval Framework forSTEM’s Education | https://aclanthology.org/2025.acl-long.502/

- **Link**: https://aclanthology.org/2025.acl-long.502/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In AI-facilitated teaching, leveraging various query styles to interpret abstract text descriptions is crucial for ensuring high-quality teaching. However, current retrieval models primarily focus on natural text-image retrieval, making them insufficiently tailored to educational scenarios due to the ambiguities in the retrieval process. In this paper, we propose a diverse expression retrieval task tailored to educational scenarios, supporting retrieval based on multiple query styles and expressions. We introduce the STEM Education Retrieval Dataset (SER), which contains over 24,000 query pairs of different styles, and the Uni-Retrieval, an efficient and style-diversified retrieval vision-language model based on prompt tuning. Uni-Retrieval extracts query style features as prototypes and builds a continuously updated Prompt Bank containing prompt tokens for diverse queries. This bank can updated during test time to represent domain-specific knowledge for different subject retrieval scenarios. Our framework demonstrates scalability and robustness by dynamically retrieving prompt tokens based on prototype similarity, effectively facilitating learning for unknown queries. Experimental results indicate that Uni-Retrieval outperforms existing retrieval models in most retrieval tasks.

</details>

---

## 119. AutoGUI: ScalingGUIGrounding with Automatic Functionality Annotations fromLLMs

- [ ] AutoGUI: ScalingGUIGrounding with Automatic Functionality Annotations fromLLMs | https://aclanthology.org/2025.acl-long.510/

- **Link**: https://aclanthology.org/2025.acl-long.510/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

User interface understanding with vision-language models (VLMs) has received much attention due to its potential for enhancing software automation.However, existing datasets used to build UI-VLMs either only contain large-scale context-free element annotations or contextualized functional descriptions for elements at a small scale.In this work, we propose theAutoGUIpipeline for automatically annotating UI elements with detailed functionality descriptions at scale.Specifically, we leverage large language models (LLMs) to infer element functionality by comparing UI state changes before and after simulated interactions. To improve annotation quality, we propose LLM-aided rejection and verification, eliminating invalid annotations without human labor.We construct a high-quality AutoGUI-704k dataset using the proposed pipeline, featuring diverse and detailed functionality annotations that are hardly provided by previous datasets.Human evaluation shows that we achieve annotation correctness comparable to a trained human annotator. Extensive experiments show that our dataset remarkably enhances VLM’s UI grounding capabilities and exhibits significant scaling effects. We also show the interesting potential use of our dataset in UI agent tasks. Please view our project at https://autogui-project.github.io/.

</details>

---

## 120. MCS-Bench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Classical Studies

- [ ] MCS-Bench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Classical Studies | https://aclanthology.org/2025.acl-long.515/

- **Link**: https://aclanthology.org/2025.acl-long.515/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid development of Multimodal Large Language Models (MLLMs), their potential in Chinese Classical Studies (CCS), a field which plays a vital role in preserving and promoting China’s rich cultural heritage, remains largely unexplored due to the absence of specialized benchmarks. To bridge this gap, we propose MCS-Bench, the first-of-its-kind multimodal benchmark specifically designed for CCS across multiple subdomains. MCS-Bench spans seven core subdomains (Ancient Chinese Text, Calligraphy, Painting, Oracle Bone Script, Seal, Cultural Relic, and Illustration), with a total of 45 meticulously designed tasks. Through extensive evaluation of 37 representative MLLMs, we observe that even the top-performing model (InternVL2.5-78B) achieves an average score below 50, indicating substantial room for improvement. Our analysis reveals significant performance variations across different tasks and identifies critical challenges in areas such as Optical Character Recognition (OCR) and cultural context interpretation. MCS-Bench not only establishes a standardized baseline for CCS-focused MLLM research but also provides valuable insights for advancing cultural heritage preservation and innovation in the Artificial General Intelligence (AGI) era. Data and code will be publicly available.

</details>

---

## 121. Large Language and Protein Assistant for Protein-Protein Interactions Prediction

- [ ] Large Language and Protein Assistant for Protein-Protein Interactions Prediction | https://aclanthology.org/2025.acl-long.554/

- **Link**: https://aclanthology.org/2025.acl-long.554/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Predicting the types and affinities of protein-protein interactions (PPIs) is crucial for understanding biological processes and developing novel therapeutic approaches. While encoding proteins themselves is essential, PPI networks can also provide rich prior knowledge for these predictive tasks. However, existing methods oversimplify the problem of PPI prediction in a semi-supervised manner when utilizing PPI networks, limiting their practical application. Furthermore, how to effectively use the rich prior knowledge of PPI networks for novel proteins not present in the network remains an unexplored issue. Additionally, due to inflexible architectures, most of existing methods cannot handle complexes containing an flexible number of proteins. To overcome these limitations, we introduce LLaPA (Large Language and Protein Assistant), a multimodal large language model that integrates proteins and PPI networks. LLaPA offers a more rational approach to utilizing PPI networks for PPI prediction and can fully exploit the information of PPI networks for unseen proteins. Through natural language instructions, LLaPA can accept flexible number of protein sequences and has the potential to perform various protein tasks. Experiments show that LLaPA achieves state-of-the-art performance in multi-label PPI (mPPI) type prediction and is capable of predicting the binding affinity between multiple interacting proteins based on sequence data.

</details>

---

## 122. Defining and Evaluating Visual Language Models’ Basic Spatial Abilities: A Perspective from Psychometrics

- [ ] Defining and Evaluating Visual Language Models’ Basic Spatial Abilities: A Perspective from Psychometrics | https://aclanthology.org/2025.acl-long.567/

- **Link**: https://aclanthology.org/2025.acl-long.567/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The Theory of Multiple Intelligences underscores the hierarchical nature of cognitive capabilities. To advance Spatial Artificial Intelligence, we pioneer a psychometric framework defining five Basic Spatial Abilities (BSAs) in Visual Language Models (VLMs): Spatial Perception, Spatial Relation, Spatial Orientation, Mental Rotation, and Spatial Visualization. Benchmarking 13 mainstream VLMs through nine validated psychometric experiments reveals significant gaps versus humans, with three key findings: 1) VLMs mirror human hierarchies (strongest in 2D orientation, weakest in 3D rotation) with independent BSAs; 2) Many smaller models surpass larger counterparts, with Qwen leading and InternVL2 lagging; 3) Interventions like CoT and few-shot training show limits from architectural constraints, while ToT demonstrates the most effective enhancement. Identified barriers include weak geometry encoding and missing dynamic simulation. By linking Psychometrics to VLMs, we provide a comprehensive BSA evaluation benchmark, a methodological perspective for embodied AI development, and a cognitive science-informed roadmap for achieving human-like spatial intelligence.

</details>

---

## 123. SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation

- [ ] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation | https://aclanthology.org/2025.acl-long.568/

- **Link**: https://aclanthology.org/2025.acl-long.568/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current vision-language models may grasp basic spatial cues and simple directions (e.g. left, right, front, back), but struggle with the multi-dimensional spatial reasoning necessary for human-like understanding and real-world applications. To address this gap, we develop SPHERE (Spatial Perception and Hierarchical Evaluation of REasoning), a hierarchical evaluation framework supported by a new human-annotated dataset. SPHERE systematically probes models across increasing levels of complexity, from fundamental skills to multi-skill integration and high-level reasoning that combines spatial, visual, and logical understanding. Benchmark evaluation of state-of-the-art models reveals significant deficiencies, especially in reasoning about distance and proximity, understanding both egocentric and allocentric perspectives, and applying spatial logic in physical contexts. These findings expose critical blind spots in existing models and underscore the need for more advanced spatial reasoning techniques, driving the development of vision-language models that align more closely with human spatial cognition.

</details>

---

## 124. LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating

- [ ] LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating | https://aclanthology.org/2025.acl-long.57/

- **Link**: https://aclanthology.org/2025.acl-long.57/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision language models (LVLMs) have improved the document understanding capabilities remarkably, enabling the handling of complex document elements, longer contexts, and a wider range of tasks. However, existing document understanding benchmarks have been limited to handling only a small number of pages and fail to provide a comprehensive analysis of layout elements locating. In this paper, we first define three primary task categories: Long Document Understanding, numerical Reasoning, and cross-element Locating, and then propose a comprehensive benchmark—LongDocURL—integrating above three primary tasks and comprising 20 sub-tasks categorized based on different primary tasks and answer evidences. Furthermore, we develop a semi-automated construction pipeline and collect 2,325 high-quality question-answering pairs, covering more than 33,000 pages of documents, significantly outperforming existing benchmarks. Subsequently, we conduct comprehensive evaluation experiments on both open-source and closed- source models across 26 different configurations, revealing critical performance gaps in this field. The code and data: https://github.com/dengc2023/LongDocURL.

</details>

---

## 125. Agri-CM3: AChinese Massive Multi-modal, Multi-level Benchmark for Agricultural Understanding and Reasoning

- [ ] Agri-CM3: AChinese Massive Multi-modal, Multi-level Benchmark for Agricultural Understanding and Reasoning | https://aclanthology.org/2025.acl-long.576/

- **Link**: https://aclanthology.org/2025.acl-long.576/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) integrating images, text, and speech can provide farmers with accurate diagnoses and treatment of pests and diseases, enhancing agricultural efficiency and sustainability. However, existing benchmarks lack comprehensive evaluations, particularly in multi-level reasoning, making it challenging to identify model limitations. To address this issue, we introduce Agri-CM3, an expert-validated benchmark assessing MLLMs’ understanding and reasoning in agricultural management. It includes 3,939 images and 15,901 multi-level multiple-choice questions with detailed explanations. Evaluations of 45 MLLMs reveal significant gaps. Even GPT-4o achieves only 63.64% accuracy, falling short in fine-grained reasoning tasks. Analysis across three reasoning levels and seven compositional abilities highlights key challenges in accuracy and cognitive understanding. Our study provides insights for advancing MLLMs in agricultural management, driving their development and application. Code and data are available athttps://github.com/HIT-Kwoo/Agri-CM3.

</details>

---

## 126. GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art

- [ ] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art | https://aclanthology.org/2025.acl-long.583/

- **Link**: https://aclanthology.org/2025.acl-long.583/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

***Video Comment Art*** enhances user engagement by providing creative content that conveys humor, satire, or emotional resonance, requiring a nuanced and comprehensive grasp of cultural and contextual subtleties. Although Multimodal Large Language Models (MLLMs) and Chain-of-Thought (CoT) have demonstrated strong reasoning abilities in STEM tasks (e.g. mathematics and coding), they still struggle to generate creative expressions such as resonant jokes and insightful satire. Moreover, existing benchmarks are constrained by their limited modalities and insufficient categories, hindering the exploration of comprehensive creativity in video-based Comment Art creation. To address these limitations, we introduce **GODBench**, a novel benchmark that integrates video and text modalities to systematically evaluate MLLMs’ abilities to compose Comment Art. Furthermore, inspired by the propagation patterns of waves in physics, we propose **Ripple of Thought (RoT)**, a multi-step reasoning framework designed to enhance the creativity of MLLMs. Extensive experiments on GODBench reveal that existing MLLMs and CoT methods still face significant challenges in understanding and generating creative video comments. In contrast, RoT provides an effective approach to improving creative composing, highlighting its potential to drive meaningful advancements in MLLM-based creativity.

</details>

---

## 127. Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation

- [ ] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation | https://aclanthology.org/2025.acl-long.606/

- **Link**: https://aclanthology.org/2025.acl-long.606/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Document Image Machine Translation (DIMT) aims to translate text within document images, facing generalization challenges due to limited training data and the complex interplay between visual and textual information. To address these challenges, we introduce M4Doc, a novel single-to-mix Modality alignment framework leveraging Multimodal Large Language Models (MLLMs). M4Doc aligns an imageonly encoder with the multimodal representations of an MLLM, pre-trained on large-scale document image datasets. This alignment enables a lightweight DIMT model to learn crucial visual-textual correlations during training. During inference, M4Doc bypasses the MLLM, maintaining computational efficiency while benefiting from its multimodal knowledge. Comprehensive experiments demonstrate substantial improvements in translation quality, especially in cross-domain generalization and challenging document image scenarios. The code will be released upon acceptance.

</details>

---

## 128. MakingLLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning

- [ ] MakingLLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning | https://aclanthology.org/2025.acl-long.610/

- **Link**: https://aclanthology.org/2025.acl-long.610/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have achieved significant success in Speech-to-Text Translation (S2TT) tasks. While most existing research has focused on English-centric translation directions, the exploration of many-to-many translation is still limited by the scarcity of parallel data. To address this, we propose a three-stage curriculum learning strategy that leverages the machine translation capabilities of large language models and adapts them to S2TT tasks, enabling effective learning in low-resource settings. We trained MLLMs with varying parameter sizes (3B, 7B, and 32B) and evaluated the proposed strategy using the FLEURS and CoVoST-2 datasets. Experimental results show that the proposed strategy achieves state-of-the-art average performance in15×14language pairs, requiring fewer than 10 hours of speech data per language to achieve competitive results. The source code and models are released athttps://github.com/yxduir/LLM-SRT.

</details>

---

## 129. Redundancy Principles forMLLMs Benchmarks

- [ ] Redundancy Principles forMLLMs Benchmarks | https://aclanthology.org/2025.acl-long.612/

- **Link**: https://aclanthology.org/2025.acl-long.612/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid iteration of Multi-modality Large Language Models (MLLMs) and the evolving demands of the field, the number of benchmarks produced annually has surged into the hundreds. The rapid growth has inevitably led to significant redundancy among benchmarks. Therefore, it is crucial to take a step back and critically assess the current state of redundancy and propose targeted principles for constructing effective MLLM benchmarks. In this paper, we focus on redundancy from three key perspectives: 1) Redundancy of benchmark capability dimensions, 2) Redundancy in the number of test questions, and 3) Cross-benchmark redundancy within specific domains. Through the comprehensive analysis over hundreds of MLLMs’ performance across more than 20 benchmarks, we aim to quantitatively measure the level of redundancy lies in existing MLLM evaluations, provide valuable insights to guide the future development of MLLM benchmarks, and offer strategies to refine and address redundancy issues effectively.

</details>

---

## 130. Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention

- [ ] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention | https://aclanthology.org/2025.acl-long.634/

- **Link**: https://aclanthology.org/2025.acl-long.634/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated impressive capabilities in multimodal understanding, but they frequently suffer from hallucination - generating content inconsistent with visual inputs. In this work, we explore a novel perspective on hallucination mitigation by examining the intermediate activations of LVLMs during generation. Our investigation reveals that hallucinated content manifests as distinct, identifiable patterns in the model’s hidden state space. Motivated by this finding, we propose Activation Steering Decoding (ASD), a training-free approach that mitigates hallucination through targeted intervention in the model’s intermediate activations. ASD operates by first identifying directional patterns of hallucination in the activation space using a small calibration set, then employing a contrast decoding mechanism that computes the difference between positive and negative steering predictions. This approach effectively suppresses hallucination patterns while preserving the model’s general capabilities. Extensive experiments demonstrate that our method significantly reduces hallucination across multiple benchmarks while maintaining performance on general visual understanding tasks. Notably, our approach requires no model re-training or architectural modifications, making it readily applicable to existing deployed models.

</details>

---

## 131. Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback

- [ ] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback | https://aclanthology.org/2025.acl-long.636/

- **Link**: https://aclanthology.org/2025.acl-long.636/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing Medical Large Vision-Language Models (Med-LVLMs), encapsulating extensive medical knowledge, demonstrate excellent capabilities in understanding medical images. However, there remain challenges in visual localization in medical images, which is crucial for abnormality detection and interpretation. To address these issues, we propose a novel UMed-LVLM designed to unveil medical abnormalities. Specifically, we collect a Medical Abnormalities Unveiling (MAU) dataset and propose a two-stage training method for UMed-LVLM training. To collect MAU dataset, we propose a prompt method utilizing the GPT-4V to generate diagnoses based on identified abnormal areas in medical images. Moreover, the two-stage training method includes Abnormal-Aware Instruction Tuning and Abnormal-Aware Rewarding, comprising Relevance Reward, Abnormal Localization Reward and Vision Relevance Reward. Experimental results demonstrate that our UMed-LVLM significantly outperforms existing Med-LVLMs in identifying and understanding medical abnormalities, achieving a 58% improvement over the baseline. In addition, this work shows that enhancing the abnormality detection capabilities of Med-LVLMs significantly improves their understanding of medical images and generalization capability. Our code and data release at URL.

</details>

---

## 132. MapNav: A Novel Memory Representation via Annotated Semantic Maps forVLM-based Vision-and-Language Navigation

- [ ] MapNav: A Novel Memory Representation via Annotated Semantic Maps forVLM-based Vision-and-Language Navigation | https://aclanthology.org/2025.acl-long.638/

- **Link**: https://aclanthology.org/2025.acl-long.638/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language navigation (VLN) is a key task in Embodied AI, requiring agents to navigate diverse and unseen environments while following natural language instructions. Traditional approaches rely heavily on historical observations as spatio-temporal contexts for decision making, leading to significant storage and computational overhead. In this paper, we introduce MapNav, a novel end-to-end VLN model that leverages Annotated Semantic Map (ASM) to replace historical frames. Specifically, our approach constructs a top-down semantic map at the start of each episode and update it at each timestep, allowing for precise object mapping and structured navigation information. Then, we enhance this map with explicit textual labels for key regions, transforming abstract semantics into clear navigation cues and generate our ASM. MapNav agent using the constructed ASM as input, and use the powerful end-to-end capabilities of VLM to empower VLN. Extensive experiments demonstrate that MapNav achieves state-of-the-art (SOTA) performance in both simulated and real-world environments, validating the effectiveness of our method. We will release our ASM generation source code and dataset to ensure reproducibility, contributing valuable resources to the field. We believe that our proposed MapNav can be used as a new memory representation method in VLN, paving the way for future research in this field.

</details>

---

## 133. Exploring Compositional Generalization of MultimodalLLMs for Medical Imaging

- [ ] Exploring Compositional Generalization of MultimodalLLMs for Medical Imaging | https://aclanthology.org/2025.acl-long.639/

- **Link**: https://aclanthology.org/2025.acl-long.639/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical imaging provides essential visual insights for diagnosis, and multimodal large language models (MLLMs) are increasingly utilized for its analysis due to their strong generalization capabilities; however, the underlying factors driving this generalization remain unclear. Current research suggests that multi-task training outperforms single-task as different tasks can benefit each other, but they often overlook the internal relationships within these tasks. To analyze this phenomenon, we attempted to employ **compositional generalization** (CG), which refers to the models’ ability to understand novel combinations by recombining learned elements, as a guiding framework. Since medical images can be precisely defined by **M**odality, **A**natomical area, and **T**ask, naturally providing an environment for exploring CG, we assembled 106 medical datasets to create **Med-MAT** for comprehensive experiments. The experiments confirmed that MLLMs can use CG to understand unseen medical images and identified CG as one of the main drivers of the generalization observed in multi-task training. Additionally, further studies demonstrated that CG effectively supports datasets with limited data and confirmed that MLLMs can achieve CG across classification and detection tasks, underscoring its broader generalization potential. Med-MAT is available at https://github.com/FreedomIntelligence/Med-MAT.

</details>

---

## 134. CLAIM: Mitigating Multilingual Object Hallucination in Large Vision-Language Models with Cross-Lingual Attention Intervention

- [ ] CLAIM: Mitigating Multilingual Object Hallucination in Large Vision-Language Models with Cross-Lingual Attention Intervention | https://aclanthology.org/2025.acl-long.640/

- **Link**: https://aclanthology.org/2025.acl-long.640/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have demonstrated impressive multimodal abilities but remain prone to multilingual object hallucination, with a higher likelihood of generating responses inconsistent with the visual input when utilizing queries in non-English languages compared to English. Most existing approaches to address these rely on pretraining or fine-tuning, which are resource-intensive. In this paper, inspired by observing the disparities in cross-modal attention patterns across languages, we propose Cross-Lingual Attention Intervention for Mitigating multilingual object hallucination (CLAIM) in LVLMs, a novel near training-free method by aligning attention patterns. CLAIM first identifies language-specific cross-modal attention heads, then estimates language shift vectors from English to the target language, and finally intervenes in the attention outputs during inference to facilitate cross-lingual visual perception capability alignment. Extensive experiments demonstrate that CLAIM achieves an average improvement of 13.56% (up to 30% in Spanish) on the POPE and 21.75% on the hallucination subsets of the MME benchmark across various languages. Further analysis reveals that multilingual attention divergence is most prominent in intermediate layers, highlighting their critical role in multilingual scenarios.

</details>

---

## 135. Cultivating Gaming Sense for Yourself: MakingVLMs Gaming Experts

- [ ] Cultivating Gaming Sense for Yourself: MakingVLMs Gaming Experts | https://aclanthology.org/2025.acl-long.643/

- **Link**: https://aclanthology.org/2025.acl-long.643/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Developing agents capable of fluid gameplay in first/third-person games without API access remains a critical challenge in Artificial General Intelligence (AGI). Recent efforts leverage Vision Language Models (VLMs) as direct controllers, frequently pausing the game to analyze screens and plan action through language reasoning. However, this inefficient paradigm fundamentally restricts agents to basic and non-fluent interactions: relying on isolated VLM reasoning for each action makes it impossible to handle tasks requiring high reactivity (e.g., FPS shooting) or dynamic adaptability (e.g., ACT combat). To handle this, we propose a paradigm shift in gameplay agent design: instead of direct control, VLM serves as a developer, creating specialized execution modules tailored for tasks like shooting and combat. These modules handle real-time game interactions, elevating VLM to a high-level developer. Building upon this paradigm, we introduce GameSense, a gameplay agent framework where VLM develops task-specific game sense modules by observing task execution and leveraging vision tools and neural network training pipelines. These modules encapsulate action-feedback logic, ranging from direct action rules to neural network-based decisions. Experiments demonstrate that our framework is the first to achieve fluent gameplay in diverse genres, including ACT, FPS, and Flappy Bird, setting a new benchmark for game-playing agents.

</details>

---

## 136. LLaSE-G1: Incentivizing Generalization Capability forLLaMA-based Speech Enhancement

- [ ] LLaSE-G1: Incentivizing Generalization Capability forLLaMA-based Speech Enhancement | https://aclanthology.org/2025.acl-long.651/

- **Link**: https://aclanthology.org/2025.acl-long.651/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area.

</details>

---

## 137. MadaKV: Adaptive Modality-PerceptionKVCache Eviction for Efficient Multimodal Long-Context Inference

- [ ] MadaKV: Adaptive Modality-PerceptionKVCache Eviction for Efficient Multimodal Long-Context Inference | https://aclanthology.org/2025.acl-long.652/

- **Link**: https://aclanthology.org/2025.acl-long.652/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces MadaKV, a modality-adaptive key-value (KV) cache eviction strategy designed to enhance the efficiency of multimodal large language models (MLLMs) in long-context inference. In multimodal scenarios, attention heads exhibit varying preferences for different modalities, resulting in significant disparities in modality importance across attention heads. Traditional KV cache eviction methods, which are tailored for unimodal settings, fail to capture modality-specific information, thereby yielding suboptimal performance. MadaKV addresses these challenges through two key components: modality preference adaptation and hierarchical compression compensation. By dynamically sensing modality information within attention heads and adaptively retaining critical tokens, MadaKV achieves substantial reductions in KV cache memory footprint and model inference decoding latency (1.3 to 1.5 times improvement) while maintaining high accuracy across various multimodal long-context tasks. Extensive experiments on representative MLLMs and the MileBench benchmark demonstrate the effectiveness of MadaKV compared to existing KV cache eviction methods.

</details>

---

## 138. Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons

- [ ] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons | https://aclanthology.org/2025.acl-long.661/

- **Link**: https://aclanthology.org/2025.acl-long.661/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multilingual language models (MLLMs) have demonstrated remarkable abilities to transfer knowledge across languages, despite being trained without explicit cross-lingual supervision. We analyze the parameter spaces of three MLLMs to study how their representations evolve during pre-training, observing patterns consistent with compression: models initially form language-specific representations, which gradually converge into cross-lingual abstractions as training progresses. Through probing experiments, we observe a clear transition from uniform language identification capabilities across layers to more specialized layer functions. For deeper analysis, we focus on neurons that encode distinct semantic concepts. By tracing their development during pre-training, we show how they gradually align across languages. Notably, we identify specific neurons that emerge as increasingly reliable predictors for the same concepts across languages. This alignment manifests concretely in generation: once an MLLM exhibits cross-lingual generalization according to our measures, we can select concept-specific neurons identified from, e.g., Spanish text and manipulate them to guide token predictions. Remarkably, rather than generating Spanish text, the model produces semantically coherent English text. This demonstrates that cross-lingually aligned neurons encode generalized semantic representations, independent of the original language encoding.

</details>

---

## 139. HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model

- [ ] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model | https://aclanthology.org/2025.acl-long.666/

- **Link**: https://aclanthology.org/2025.acl-long.666/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Instruction tuning is widely used to enhance a pre-trained Multimodal Large Language Model (MLLM) to understand and follow human instructions by training it on a curated set of task-specific dataset. However, it is infeasible to collect all possible instruction datasets simultaneously in real-world scenarios. Thus, enabling MLLM with continual instruction tuning is essential for maintaining their adaptability. However, existing methods often trade off memory efficiency for performance gains, significantly compromising overall efficiency. In this paper, we propose a task-specific expansion and task-general fusion framework based on the variations in Centered Kernel Alignment (CKA) similarity across different model layers when trained on diverse datasets. Furthermore, we analyze the information leakage present in the existing benchmark and propose a new and more challenging benchmark to rationally evaluate the performance of different methods. Comprehensive experiments showcase a significant performance improvement of our method compared to existing state-of-the-art methods. Our code will be public available.

</details>

---

## 140. TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition

- [ ] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition | https://aclanthology.org/2025.acl-long.669/

- **Link**: https://aclanthology.org/2025.acl-long.669/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Parameter-Efficient Fine-Tuning (PEFT) methods like Low-Rank Adaptation (LoRA) effectively address resource constraints during fine-tuning, their performance often falls short, especially in multidimensional task scenarios. To address this issue, one straightforward solution is to introduce task-specific LoRA as domain experts, leveraging the modeling of multiple capabilities of experts and thus enhancing the general capability of multi-task learning.Although promising, these additional components often add complexity to the training and inference process, contravening the efficiency that PEFT is designed to deliver. Considering this, we introduce an innovative PEFT method, **TeamLoRA**, consisting of a collaboration and competition module for LoRA experts, thus achieving the right balance of effectiveness and efficiency:**(i)** For *collaboration*, we introduce a novel knowledge sharing and organization mechanism designed to optimize hierarchical learning while enhancing the efficiency of model training and inference.**(ii)** For *competition*, we propose leveraging a game-theoretic interaction mechanism for experts, encouraging experts to transfer their domain-specific knowledge while facing diverse downstream tasks, thus enhancing the performance.By doing so, TeamLoRA elegantly connects the experts as a “*Team*” with internal collaboration and competition, enabling a faster and more accurate PEFT paradigm. Meanwhile, we curate a **Comprehensive Multi-Task Evaluation (CME)** benchmark to thoroughly assess the capability of multi-task learning. Experiments conducted on our CME and other benchmarks indicate the effectiveness and efficiency of TeamLoRA. Our project is available at https://github.com/DCDmllm/TeamLoRA.

</details>

---

## 141. HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models

- [ ] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models | https://aclanthology.org/2025.acl-long.679/

- **Link**: https://aclanthology.org/2025.acl-long.679/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical Vision-Language Models (Med-VLMs) have achieved success across various tasks, yet most existing methods overlook the modality misalignment issue that can lead to untrustworthy responses in clinical settings. In this paper, we propose Hierarchical Self-Contrastive Rewarding (HSCR), a novel approach that addresses two critical challenges in Med-VLM alignment: 1) Cost-effective generation of high-quality preference data; 2) Capturing nuanced and context-aware preferences for improved alignment. HSCR first leverages the inherent capability of Med-VLMs to generate dispreferred responses with higher sampling probability. By analyzing output logit shifts after visual token dropout, we identify modality-coupled tokens that induce misalignment and derive an implicit alignment reward function. This function guides token replacement with hallucinated ones during decoding, producing high-quality dispreferred data. Furthermore, HSCR introduces a multi-level preference optimization strategy, which extends beyond traditional adjacent-level optimization by incorporating nuanced implicit preferences, leveraging relative quality in dispreferred data to capture subtle alignment cues for more precise and context-aware optimization. Extensive experiments across multiple medical tasks, including Med-VQA, medical image captioning and instruction following, demonstrate that HSCR not only enhances zero-shot performance but also significantly improves modality alignment and trustworthiness with just 2,000 training entries. Code is released on https://github.com/jiangsongtao/HSCR.

</details>

---

## 142. MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale

- [ ] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale | https://aclanthology.org/2025.acl-long.680/

- **Link**: https://aclanthology.org/2025.acl-long.680/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Open-source multimodal large language models (MLLMs) have shown significant potential in a broad range of tasks. However, their reasoning capabilities remain constrained by existing instruction-tuning datasets, which were predominately repurposed from academic datasets such as VQA, AI2D, and ChartQA. These datasets target simplistic tasks, and only provide phrase-level answers without any intermediate rationales.To address these challenges, we introduce a scalable and cost-effective method to construct a large-scale multimodal instruction-tuning dataset with rich intermediate rationales designed to elicit CoT reasoning. Using only open models, we create a dataset containing 12M instruction-response pairs to cover diverse reasoning-intensive tasks.Experiments demonstrate that training MLLMs on our dataset not only significantly improves reasoning capabilities, achieving state-of-the-art performance on benchmarks such as MathVerse (+8.1%), MMMU-Pro (+7%), and MuirBench (+13.3%), but also gains improvements of up to 4% on non-reasoning-based benchmarks.

</details>

---

## 143. Emma-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning

- [ ] Emma-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning | https://aclanthology.org/2025.acl-long.695/

- **Link**: https://aclanthology.org/2025.acl-long.695/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Traditional reinforcement learning-based robotic control methods are often task-specific and fail to generalize across diverse environments or unseen objects and instructions. Visual Language Models (VLMs) demonstrate strong scene understanding and planning capabilities but lack the ability to generate actionable policies tailored to specific robotic embodiments. To address this, Visual-Language-Action (VLA) models have emerged, yet they face challenges in long-horizon spatial reasoning and grounded task planning. In this work, we propose the Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning, EMMA-X. EMMA-X leverages our constructed hierarchical embodiment dataset based on BridgeV2, containing 60,000 robot manipulation trajectories auto-annotated with grounded task reasoning and spatial guidance. Additionally, we introduce a trajectory segmentation strategy based on gripper states and motion trajectories, which can help mitigate hallucination in grounding subtask reasoning generation. Experimental results demonstrate that EMMA-X achieves superior performance over competitive baselines, particularly in real-world robotic tasks requiring spatial reasoning.

</details>

---

## 144. CanMLLMs Understand the Deep Implication BehindChinese Images?

- [ ] CanMLLMs Understand the Deep Implication BehindChinese Images? | https://aclanthology.org/2025.acl-long.700/

- **Link**: https://aclanthology.org/2025.acl-long.700/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As the capabilities of Multimodal Large Language Models (MLLMs) improve, the need for higher-order evaluation of them is increasing. However, there is a lack of work evaluating MLLM for higher-order perception and understanding of Chinese visual content. To address this, we introduce the CII-Bench, which aims to assess MLLMs’ such capabilities for Chinese images. To ensure the authenticity of the Chinese context, images in CII-Bench are sourced from the Chinese Internet and manually reviewed, with corresponding answers also manually crafted. Additionally, CII-Bench incorporates images that represent Chinese traditional culture, such as famous Chinese traditional paintings, which can deeply reflect the model’s understanding of Chinese traditional culture. Through experiments on multiple MLLMs using CII-Bench, significant findings emerged. There is a large gap between MLLMs and humans in performance. The highest MLLM accuracy is 64.4%, while the human average is 78.2% and the peak is 81.0%. MLLMs perform poorly on traditional culture images, indicating limitations in understanding high-level semantics and lacking a deep knowledge base of Chinese traditional culture. Moreover, most models have higher accuracy when image emotion hints are added to the prompts. We believe CII-Bench will help MLLMs better understand Chinese semantics and specific images, and move forward the development of expert artificial general intelligence (AGI). Our project is publicly available at https://cii-bench.github.io.

</details>

---

## 145. EAGLE: Expert-Guided Self-Enhancement for Preference Alignment in Pathology Large Vision-Language Model

- [ ] EAGLE: Expert-Guided Self-Enhancement for Preference Alignment in Pathology Large Vision-Language Model | https://aclanthology.org/2025.acl-long.711/

- **Link**: https://aclanthology.org/2025.acl-long.711/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision Language Models (LVLMs) show promise for pathological diagnosis, yet their application in clinical settings faces critical challenges of multimodal hallucination and biased responses. While preference alignment methods have proven effective in general domains, acquiring high-quality preference data for pathology remains challenging due to limited expert resources and domain complexity. In this paper, we propose EAGLE (Expert-guided self-enhancement for preference Alignment in patholoGy Large vision-languagE model), a novel framework that systematically integrates medical expertise into preference alignment. EAGLE consists of three key stages: initialization through supervised fine-tuning, self-preference creation leveraging expert prompting and medical entity recognition, and iterative preference following-tuning. The self-preference creation stage uniquely combines expert-verified chosen sampling with expert-guided rejected sampling to generate high-quality preference data, while the iterative tuning process continuously refines both data quality and model performance. Extensive experiments demonstrate that EAGLE significantly outperforms existing pathological LVLMs, effectively reducing hallucination and bias while maintaining pathological accuracy. The source code is available at https://github.com/meidandz/EAGLE.

</details>

---

## 146. RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought

- [ ] RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought | https://aclanthology.org/2025.acl-long.715/

- **Link**: https://aclanthology.org/2025.acl-long.715/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal Large Language Models (MLLMs) have demonstrated remarkable reasoning capability while lack explicit mechanisms for visual grounding and segmentation, creating a gap between cognitive reasoning and visual perception. To bridge this gap, we introduce Reasoning Segmentation via Visual Prompting (RSVP), a novel framework that unifies multi-step multimodal reasoning with grounded visual understanding. RSVP is a two-stage structuralized framework that integrates reasoning-driven localization with segmentation refinement. In the reasoning stage, RSVP employs multimodal chain-of-thought visual prompts to help MLLMs understand queries and infer targets, generating interpretable region proposals that enhance visual grounding. In segmentation stage, RSVP refines these proposals with a Vision-Language Segmentation Module (VLSM), seamlessly integrates textual and visual cues to produce precise segmentation masks. By explicitly modelling the interaction between multimodal reasoning and segmentation, RSVP introduces a new paradigm for interpretable reasoning segmentation. It exploits MLLMs’ inherent localization capabilities, enabling the models to not only reason about objects but also generate structured visual representations. Our extensive experiments demonstrate that RSVP achieves state-of-the-art performance, surpasses state-of-the-art methods by up to +6.5 gIoU and +9.2 cIoU on ReasonSeg, and achieves 49.7 mAP on SegInW under zero-shot settings. These results validate RSVP as an effective and scalable framework for integrating cognitive reasoning with structured visual understanding.

</details>

---

## 147. Can Vision-Language Models Evaluate Handwritten Math?

- [ ] Can Vision-Language Models Evaluate Handwritten Math? | https://aclanthology.org/2025.acl-long.720/

- **Link**: https://aclanthology.org/2025.acl-long.720/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Vision-Language Models (VLMs) have opened new possibilities in automatic grading of handwritten student responses, particularly in mathematics. However, a comprehensive study to test the ability of VLMs to evaluate and reason over handwritten content remains absent. To address this gap, we introduce FERMAT, a benchmark designed to assess VLMs’ ability to detect, localize and correct errors in handwritten mathematical content. FERMAT spans four key error dimensions - computational, conceptual, notational, and presentation - and comprises over 2,200 handwritten math solutions derived from 609 manually curated problems from grades 7-12 with intentionally introduced perturbations. Using FERMAT we benchmark nine VLMs across three tasks: error detection, localization, and correction. Our results reveal significant shortcomings in current VLMs in reasoning over handwritten text, with Gemini-1.5-Pro achieving the highest error correction rate (77%). We also observed that some models struggle with processing handwritten content, as their accuracy improves when handwritten inputs are replaced with printed text or images. These findings highlight the limitations of current VLMs and reveal new avenues for improvement. We will release FERMAT and all the associated resources in the open-source to drive further research.

</details>

---

## 148. HiddenDetect: Detecting Jailbreak Attacks against Multimodal Large Language Models via Monitoring Hidden States

- [ ] HiddenDetect: Detecting Jailbreak Attacks against Multimodal Large Language Models via Monitoring Hidden States | https://aclanthology.org/2025.acl-long.724/

- **Link**: https://aclanthology.org/2025.acl-long.724/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that HiddenDetect surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code and data will be released publicly.

</details>

---

## 149. LLaVASteering: Visual Instruction Tuning with 500x Fewer Parameters through Modality Linear Representation-Steering

- [ ] LLaVASteering: Visual Instruction Tuning with 500x Fewer Parameters through Modality Linear Representation-Steering | https://aclanthology.org/2025.acl-long.739/

- **Link**: https://aclanthology.org/2025.acl-long.739/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) enhance visual tasks by integrating visual representations into large language models (LLMs). The textual modality, inherited from LLMs, enables instruction following and in-context learning, while the visual modality boosts downstream task performance through rich semantic content, spatial information, and grounding capabilities. These modalities work synergistically across various visual tasks. Our research reveals a persistent imbalance between these modalities, with text often dominating output generation during visual instruction tuning, regardless of using full or parameter-efficient fine-tuning (PEFT). We found that re-balancing these modalities can significantly reduce trainable parameters, inspiring further optimization of visual instruction tuning. To this end, we introduce Modality Linear Representation-Steering (MoReS), which re-balances intrinsic modalities by steering visual representations through linear transformations in the visual subspace across each model layer. We validated our approach by developing LLaVA Steering, a suite of models using MoReS. Results show that LLaVA Steering requires, on average, 500 times fewer trainable parameters than LoRA while maintaining comparable performance across three visual benchmarks and eight visual question-answering tasks. Finally, we introduce the LLaVA Steering Factory, a platform that enables rapid customization of MLLMs with a component-based architecture, seamlessly integrating state-of-the-art models and evaluating intrinsic modality imbalance. This open-source project facilitates a deeper understanding of MLLMs within the research community.

</details>

---

## 150. Jailbreak Large Vision-Language Models Through Multi-Modal Linkage

- [ ] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage | https://aclanthology.org/2025.acl-long.74/

- **Link**: https://aclanthology.org/2025.acl-long.74/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of Large Vision-Language Models (VLMs), concerns about their ‌potential misuse and abuse have grown rapidly. Prior research has exposed VLMs’ vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, current jailbreak methods often fail against cutting-edge models such as GPT-4o. We attribute this to the over-exposure of harmful content and the absence of stealthy malicious guidance. In this work, we introduce a novel jailbreak framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML employs an encryption-decryption process across text and image modalities to mitigate the over-exposure of malicious information. To covertly align the model’s output with harmful objectives, MML leverages a technique we term evil alignment, framing the attack within the narrative context of a video game development scenario. Extensive experiments validate the effectiveness of MML. Specifically, MML jailbreaks GPT-4o with attack success rates of 99.40% on SafeBench, 98.81% on MM-SafeBench, and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML.

</details>

---

## 151. VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service

- [ ] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service | https://aclanthology.org/2025.acl-long.781/

- **Link**: https://aclanthology.org/2025.acl-long.781/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have demonstrated great potential in real-world applications. While existing research primarily focuses on improving their accuracy, the efficiency remains underexplored. Given the real-time demands of many applications and the high inference overhead of VLMs, efficiency robustness is a critical issue. However, previous studies evaluate efficiency robustness under unrealistic assumptions, requiring access to the model architecture and parameters—an impractical scenario in ML-as-a-service settings, where VLMs are deployed via inference APIs. To address this gap, we propose VLMInferSlow, a novel approach for evaluating VLM efficiency robustness in a realistic black-box setting. VLMInferSlow incorporates fine-grained efficiency modeling tailored to VLM inference and leverages zero-order optimization to search for adversarial examples. Experimental results show that VLMInferSlow generates adversarial images with imperceptible perturbations, increasing the computational cost by up to 128.47%. We hope this research raises the community’s awareness about the efficiency robustness of VLMs.

</details>

---

## 152. The Alternative Annotator Test forLLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators withLLMs

- [ ] The Alternative Annotator Test forLLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators withLLMs | https://aclanthology.org/2025.acl-long.782/

- **Link**: https://aclanthology.org/2025.acl-long.782/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The “LLM-as-an-annotator” and “LLM-as-a-judge” paradigms employ Large Language Models (LLMs) as annotators, judges, and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure, the Alternative Annotator Test (alt-test), that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM annotators and judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming the open-source LLMs we examine, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.

</details>

---

## 153. MMBoundary: AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration

- [ ] MMBoundary: AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration | https://aclanthology.org/2025.acl-long.802/

- **Link**: https://aclanthology.org/2025.acl-long.802/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, multimodal large language models (MLLMs) have made significant progress but continue to face inherent challenges in multimodal reasoning, which requires multi-level (e.g., perception, reasoning) and multi-granular (e.g., multi-step reasoning chain) advanced inferencing. Prior work on estimating model confidence tends to focus on the overall response for training and calibration, but fails to assess confidence in each reasoning step, leading to undesirable hallucination snowballing. In this work, we present MMBoundary, a novel framework that advances the knowledge boundary awareness of MLLMs through reasoning step confidence calibration. To achieve this, we propose to incorporate complementary textual and cross-modal self-rewarding signals to estimate confidence at each step of the MLLM reasoning process. In addition to supervised fine-tuning MLLM on this set of self-rewarding confidence estimation signal for initial confidence expression warm-up, we introduce a reinforcement learning stage with multiple reward functions for further aligning model knowledge and calibrating confidence at each reasoning step, enhancing reasoning chain self-correction. Empirical results show that MMBoundary significantly outperforms existing methods across diverse domain datasets and metrics, achieving an average of 7.5% reduction in multimodal confidence calibration errors and up to 8.3% improvement in task performance.

</details>

---

## 154. Improve Vision Language Model Chain-of-thought Reasoning

- [ ] Improve Vision Language Model Chain-of-thought Reasoning | https://aclanthology.org/2025.acl-long.82/

- **Link**: https://aclanthology.org/2025.acl-long.82/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Chain-of-thought (CoT) reasoning in vision language models (VLMs) is crucial for improving interpretability and trustworthiness. However, current training recipes often relying on datasets dominated by short annotations with minimal rationales. In this work, we show that training VLM on short answers leads to poor generalization on reasoning tasks that require more detailed explanations. To address this limitation, we propose a two-stage post-training strategy that extends the usage of short answer data for enhanced CoT reasoning. First, we augment short answers with CoT reasoning generated by GPT-4o, enhancing the VLM’s CoT capabilities through fine-tuning. Second, we leverage short answers as outcome rewards for reinforcement learning. Specifically, short answers are used as correctness indicators to construct positive (correct) and negative (incorrect) pairs from model-generated reasoning chains. These pairs are then used to calibrate the model’s reasoning via Direct Preference Optimization. Our experiments show significant improvements in CoT reasoning on benchmark datasets, along with enhanced generalization to direct answer prediction. This work provides a critical data resource for VLM CoT training and demonstrates the effectiveness of outcome rewards for multimodal models post-training.

</details>

---

## 155. Can’t See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for MultimodalLLMs

- [ ] Can’t See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for MultimodalLLMs | https://aclanthology.org/2025.acl-long.832/

- **Link**: https://aclanthology.org/2025.acl-long.832/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have expanded the capabilities of traditional language models by enabling interaction through both text and images. However, ensuring the safety of these models remains a significant challenge, particularly in accurately identifying whether multimodal content is safe or unsafe—a capability we termsafety awareness. In this paper, we introduce MMSafeAware, the first comprehensive multimodal safety awareness benchmark designed to evaluate MLLMs across 29 safety scenarios with 1,500 carefully curated image-prompt pairs. MMSafeAware includes both unsafe and over-safety subsets to assess models’ abilities to correctly identify unsafe content and avoid over-sensitivity that can hinder helpfulness. Evaluating nine widely used MLLMs using MMSafeAware reveals that current models are not sufficiently safe and often overly sensitive; for example, GPT-4V misclassifies 36.1% of unsafe inputs as safe and 59.9% of benign inputs as unsafe. We further explore three methods to improve safety awareness—prompting-based approaches, visual contrastive decoding, and vision-centric reasoning fine-tuning—but find that none achieve satisfactory performance. Our findings highlight the profound challenges in developing MLLMs with robust safety awareness, underscoring the need for further research in this area. All the code and data will be publicly available to facilitate future research.

</details>

---

## 156. Movie101v2: Improved Movie Narration Benchmark

- [ ] Movie101v2: Improved Movie Narration Benchmark | https://aclanthology.org/2025.acl-long.836/

- **Link**: https://aclanthology.org/2025.acl-long.836/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automatic movie narration aims to generate video-aligned plot descriptions to assist visually impaired audiences. Unlike standard video captioning, it involves not only describing key visual details but also inferring plots that unfold across multiple movie shots, presenting distinct and complex challenges. To advance this field, we introduce Movie101v2, a large-scale, bilingual dataset with enhanced data quality specifically designed for movie narration. Revisiting the task, we propose breaking down the ultimate goal of automatic movie narration into three progressive stages, offering a clear roadmap with corresponding evaluation metrics. Based on our new benchmark, we baseline a range of large vision-language models and conduct an in-depth analysis of the challenges in movie narration generation. Our findings highlight that achieving applicable movie narration generation is a fascinating goal that requires significant research.

</details>

---

## 157. Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation

- [ ] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation | https://aclanthology.org/2025.acl-long.855/

- **Link**: https://aclanthology.org/2025.acl-long.855/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Reasoning about images with rich text, such as charts and documents, is a critical application of vision-language models (VLMs). However, VLMs often struggle in these domains due to the scarcity of diverse text-rich vision-language data. To address this challenge, we present CoSyn, a framework that leverages the coding capabilities of text-only large language models (LLMs) to automatically create synthetic text-rich multimodal data. Given input text describing a target domain (e.g., “nutrition fact labels”), CoSyn prompts an LLM to generate code (Python, HTML, LaTeX, etc.) for rendering synthetic images. With the underlying code as textual representations of the synthetic images, CoSyn can generate high-quality instruction-tuning data, again relying on a text-only LLM. Using CoSyn, we constructed a dataset comprising 400K images and 2.7M rows of vision-language instruction-tuning data. Comprehensive experiments on seven benchmarks demonstrate that models trained on our synthetic data achieve state-of-the-art performance among competitive open-source models, including Llama 3.2, and surpass proprietary models such as GPT-4V and Gemini 1.5 Flash. Furthermore, CoSyn can produce synthetic pointing data, enabling VLMs to ground information within input images, showcasing its potential for developing multimodal agents capable of acting in real-world environments.

</details>

---

## 158. Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents

- [ ] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents | https://aclanthology.org/2025.acl-long.857/

- **Link**: https://aclanthology.org/2025.acl-long.857/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As Multimodal Large Language Models (MLLMs) advance, multimodal agents show promise in real-world tasks like web navigation and embodied intelligence. However, due to limitations in a lack of external feedback, these agents struggle with self-correction and generalization. A promising approach is to use reward models as external feedback, but there is no clear on how to select reward models for agents. Thus, there is an urgent need to build a reward bench targeted at agents. To address these challenges, we propose Agent-RewardBench, a benchmark designed to evaluate reward modeling ability in MLLMs. The benchmark is characterized by three key features: (1) Multiple dimensions and real-world agent scenarios evaluation. It covers perception, planning, and safety with 7 scenarios; (2) Step-level reward evaluation. It allows for the assessment of agent capabilities at the individual steps of a task, providing a more granular view of performance during the planning process; and (3) Appropriately difficulty and high-quality. We carefully sample from 10 diverse models, difficulty control to maintain task challenges, and manual verification to ensure the integrity of the data. Experiments demonstrate that even state-of-the-art multimodal models show limited performance, highlighting the need for specialized training in agent reward modeling. Code is available at github.

</details>

---

## 159. Open-World Attribute Mining forE-Commerce Products with Multimodal Self-Correction Instruction Tuning

- [ ] Open-World Attribute Mining forE-Commerce Products with Multimodal Self-Correction Instruction Tuning | https://aclanthology.org/2025.acl-long.85/

- **Link**: https://aclanthology.org/2025.acl-long.85/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In e-commerce, effective product Attribute Mining (AM) is essential for improving product features and aiding consumer decisions. However, current AM methods often focus on extracting attributes from unimodal text, underutilizing multimodal data. In this paper, we propose a novel framework called Multimodal Self-Correction Instruction Tuning (MSIT) to mine new potential attributes from both images and text with Multimodal Large Language Models. The tuning process involves two datasets: Attribute Generation Tuning Data (AGTD) and Chain-of-Thought Tuning Data (CTTD). AGTD is constructed utilizing in-context learning with a small set of seed attributes, aiding the MLLM in accurately extracting attribute-value pairs from multimodal information. To introduce explicit reasoning and improve the extraction in accuracy, we construct CTTD, which incorporates a structured 5-step reasoning process for self-correction. Finally, we employ a 3-stage inference process to filter out redundant attributes and sequentially validate each generated attribute. Comprehensive experimental results on two datasets show that MSIT outperforms state-of-the-art methods. We will release our code and data in the near future.

</details>

---

## 160. Takin-VC: Expressive Zero-Shot Voice Conversion via Adaptive Hybrid Content Encoding and Enhanced Timbre Modeling

- [ ] Takin-VC: Expressive Zero-Shot Voice Conversion via Adaptive Hybrid Content Encoding and Enhanced Timbre Modeling | https://aclanthology.org/2025.acl-long.87/

- **Link**: https://aclanthology.org/2025.acl-long.87/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Expressive zero-shot voice conversion (VC) is a critical and challenging task that aims to transform the source timbre into an arbitrary unseen speaker while preserving the original content and expressive qualities. Despite recent progress in zero-shot VC, there remains considerable potential for improvements in speaker similarity and speech naturalness. Moreover, existing zero-shot VC systems struggle to fully reproduce paralinguistic information in highly expressive speech, such as breathing, crying, and emotional nuances, limiting their practical applicability. To address these issues, we propose Takin-VC, a novel expressive zero-shot VC framework via adaptive hybrid content encoding and memory-augmented context-aware timbre modeling. Specifically, we introduce an innovative hybrid content encoder that incorporates an adaptive fusion module, capable of effectively integrating quantized features of the pre-trained WavLM and HybridFormer in an implicit manner, so as to extract precise linguistic features while enriching paralinguistic elements. For timbre modeling, we propose advanced memory-augmented and context-aware modules to generate high-quality target timbre features and fused representations that seamlessly align source content with target timbre. To enhance real-time performance, we advocate a conditional flow matching model to reconstruct the Mel-spectrogram of the source speech. Experimental results show that our Takin-VC consistently surpasses state-of-the-art VC systems, achieving notable improvements in terms of speech naturalness, speech expressiveness, and speaker similarity, while offering enhanced inference speed.

</details>

---

## 161. Insight Over Sight: Exploring the Vision-Knowledge Conflicts in MultimodalLLMs

- [ ] Insight Over Sight: Exploring the Vision-Knowledge Conflicts in MultimodalLLMs | https://aclanthology.org/2025.acl-long.872/

- **Link**: https://aclanthology.org/2025.acl-long.872/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper explores the problem of commonsense level vision-knowledge conflict in Multimodal Large Language Models (MLLMs), where visual information contradicts model’s internal commonsense knowledge. To study this issue, we introduce an automated framework, augmented with human-in-the-loop quality control, to generate inputs designed to simulate and evaluate these conflicts in MLLMs. Using this framework, we have crafted a diagnostic benchmark consisting of 374 original images and 1,122 high-quality question-answer (QA) pairs. The benchmark covers two aspects of conflict and three question types, providing a thorough assessment tool. We apply this benchmark to assess the conflict-resolution capabilities of nine representative MLLMs from various model families. Our results indicate an evident over-reliance on parametric knowledge for approximately 20% of all queries, especially among Yes-No and action-related problems. Based on these findings, we evaluate the effectiveness of existing approaches to mitigating the conflicts and compare them to our “Focus-on-Vision” prompting strategy. Despite some improvement, the vision-knowledge conflict remains unresolved and can be further scaled through our data construction framework. Our proposed framework, benchmark, and analysis contribute to the understanding and mitigation of vision-knowledge conflicts in MLLMs.

</details>

---

## 162. The Impact of Auxiliary Patient Data on Automated ChestX-Ray Report Generation and How to Incorporate It

- [ ] The Impact of Auxiliary Patient Data on Automated ChestX-Ray Report Generation and How to Incorporate It | https://aclanthology.org/2025.acl-long.9/

- **Link**: https://aclanthology.org/2025.acl-long.9/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This study investigates the integration of diverse patient data sources into multimodal language models for automated chest X-ray (CXR) report generation. Traditionally, CXR report generation relies solely on data from a patient’s CXR exam, overlooking valuable information from patient electronic health records. Utilising the MIMIC-CXR and MIMIC-IV-ED datasets, we investigate the use of patient data from emergency department (ED) records — such as vital signs measured and medicines reconciled during an ED stay — for CXR report generation, with the aim of enhancing diagnostic accuracy. We also investigate conditioning CXR report generation on the clinical history section of radiology reports, which has been overlooked in the literature. We introduce a novel approach to transform these heterogeneous data sources into patient data embeddings that prompt a multimodal language model (CXRMate-ED). Our comprehensive evaluation indicates that using a broader set of patient data significantly enhances diagnostic accuracy. The model, training code, and dataset are publicly available.

</details>

---

## 163. OmniAlign-V: Towards Enhanced Alignment ofMLLMs with Human Preference

- [ ] OmniAlign-V: Towards Enhanced Alignment ofMLLMs with Human Preference | https://aclanthology.org/2025.acl-long.906/

- **Link**: https://aclanthology.org/2025.acl-long.906/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in open-source multi-modal large language models (MLLMs) have primarily focused on enhancing foundational capabilities, leaving a significant gap in human preference alignment. This paper introduces OmniAlign-V, a comprehensive dataset of 200K high-quality training samples featuring diverse images, complex questions, and varied response formats to improve MLLMs’ alignment with human preferences. We also present MM-AlignBench, a human-annotated benchmark specifically designed to evaluate MLLMs’ alignment with human values. Experimental results show that finetuning MLLMs with OmniAlign-V, using Supervised Fine-Tuning (SFT) or Direct Preference Optimization (DPO), significantly enhances human preference alignment while maintaining or enhancing performance on standard VQA benchmarks, preserving their fundamental capabilities.

</details>

---

## 164. Crowdsource, Crawl, or Generate? CreatingSEA-VL, a Multicultural Vision-Language Dataset forSoutheastAsia

- [ ] Crowdsource, Crawl, or Generate? CreatingSEA-VL, a Multicultural Vision-Language Dataset forSoutheastAsia | https://aclanthology.org/2025.acl-long.916/

- **Link**: https://aclanthology.org/2025.acl-long.916/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite Southeast Asia’s (SEA) extraordinary linguistic and cultural diversity, the region remains significantly underrepresented in vision-language (VL) research, resulting in AI models that inadequately capture SEA cultural nuances. To fill this gap, we present SEA-VL, an open-source initiative dedicated to developing culturally relevant high-quality datasets for SEA languages. By involving contributors from SEA countries, SEA-VL ensures better cultural relevance and diversity, fostering greater inclusivity of underrepresented languages and cultural depictions in VL research. Our methodology employed three approaches: community-driven crowdsourcing with SEA contributors, automated image crawling, and synthetic image generation. We evaluated each method’s effectiveness in capturing cultural relevance. We found that image crawling achieves approximately ~85% cultural relevance while being more cost- and time-efficient than crowdsourcing, whereas synthetic image generation failed to accurately reflect SEA cultural nuances and contexts. Collectively, we gathered 1.28 million SEA culturally relevant images, more than 50 times larger than other existing datasets. This work bridges the representation gap in SEA, establishes a foundation for developing culturally aware AI systems for this region, and provides a replicable framework for addressing representation gaps in other underrepresented regions.

</details>

---

## 165. MegaPairs: Massive Data Synthesis for Universal Multimodal Retrieval

- [ ] MegaPairs: Massive Data Synthesis for Universal Multimodal Retrieval | https://aclanthology.org/2025.acl-long.935/

- **Link**: https://aclanthology.org/2025.acl-long.935/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the rapidly growing demand for multimodal retrieval, progress in this field remains severely constrained by a lack of training data. In this paper, we introduce MegaPairs, a novel data synthesis method that leverages vision language models (VLMs) and open-domain images, together with a massive synthetic dataset generated from this method. Our empirical analysis shows that MegaPairs generates high-quality data, enabling the multimodal retriever to significantly outperform the baseline model trained on 70×more data from existing datasets. Moreover, since MegaPairs solely relies on general image corpora and open-source VLMs, it can be easily scaled up, enabling continuous improvements in retrieval performance. In this stage, we produced more than 26 million training instances and trained several models of varying sizes using this data. These new models achieve state-of-the-art zero-shot performance across 4 popular composed image retrieval (CIR) benchmarks and the highest overall performance on the 36 datasets provided by MMEB. They also demonstrate notable performance improvements with additional downstream fine-tuning. Our code, synthesized dataset, and pre-trained models are publicly available at https://github.com/VectorSpaceLab/MegaPairs.

</details>

---

## 166. Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension

- [ ] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension | https://aclanthology.org/2025.acl-long.987/

- **Link**: https://aclanthology.org/2025.acl-long.987/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the impressive performance of multilingual large language models (mLLMs) in various natural language processing tasks, their ability to understand procedural texts, particularly those with culture-specific content, remains largely unexplored. Texts describing cultural procedures, including rituals, traditional craftsmanship, and social etiquette, require an inherent understanding of cultural context, presenting a significant challenge for mLLMs. In this work, we introduce CAPTex, a benchmark designed to evaluate mLLMs’ ability to process and reason over culturally diverse procedural texts in multiple languages. Using a range of evaluation methods, we find that (1) mLLMs struggle with culturally contextualized procedural content, particularly in low-resource languages; (2) performance varies across cultural domains, with some proving more difficult than others; and (3) models perform better on multiple-choice tasks presented in conversational formats than on direct questions. These results highlight the current limitations of mLLMs and emphasize the need for culturally informed benchmarks like CAPTex to support more accurate and inclusive language understanding.

</details>

---

## 167. Do Multimodal Large Language Models Truly See What We Point At? Investigating Indexical, Iconic, and Symbolic Gesture Comprehension

- [ ] Do Multimodal Large Language Models Truly See What We Point At? Investigating Indexical, Iconic, and Symbolic Gesture Comprehension | https://aclanthology.org/2025.acl-short.40/

- **Link**: https://aclanthology.org/2025.acl-short.40/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Understanding hand gestures is essential for human communication, yet it remains unclear how well multimodal large language models (MLLMs) comprehend them. In this paper, we examine MLLMs’ ability to interpret indexical gestures, which require external referential grounding, in comparison to iconic gestures, which depict imagery, and symbolic gestures, which are conventionally defined. We hypothesize that MLLMs, lacking real-world referential understanding, will struggle significantly with indexical gestures. To test this, we manually annotated five gesture type labels to 925 gesture instances from the Miraikan SC Corpus and analyzed gesture descriptions generated by state-of-the-art MLLMs, including GPT-4o. Our findings reveal a consistent weakness across models in interpreting indexical gestures, suggesting that MLLMs rely heavily on linguistic priors or commonsense knowledge rather than grounding their interpretations in visual or contextual cues.

</details>

---

## 168. Fast or Slow? Integrating Fast Intuition and Deliberate Thinking for Enhancing Visual Question Answering

- [ ] Fast or Slow? Integrating Fast Intuition and Deliberate Thinking for Enhancing Visual Question Answering | https://aclanthology.org/2025.acl-short.41/

- **Link**: https://aclanthology.org/2025.acl-short.41/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) still struggle with complex reasoning tasks in Visual Question Answering (VQA). While current methods have advanced by incorporating visual prompts, our study uncovers critical limitations: these approaches indiscriminately annotate all detected objects for every visual question, generating excessive visual markers that degrade task performance. This issue stems primarily from a lack of focus on key visual elements, raising two important questions: Are all objects equally important, and do all questions require visual prompts? Motivated by Dual Process Theory, which distinguishes between instinctive and deliberate cognitive modes in human reasoning, we propose FOCUS, a plug-and-play approach that dynamically adapts to the complexity of questions, combining fast intuitive judgments with deliberate analytical reasoning to enhance the vision-language reasoning capability of the MLLM. For straightforward questions, FOCUS supports efficient zero-shot reasoning. For more complex tasks, it employs the conceptualizing before observation strategy to highlight critical elements. Extensive experiments on four benchmarks—ScienceQA, TextQA, VizWiz, and MME—demonstrate that FOCUS consistently improves the performance of both open-source and black-box MLLMs, achieving significant gains across all datasets. Ablation studies further validate the importance of combining diverse cognitive strategies with refined visual information for superior performance. Code will be released.

</details>

---

## 169. Advancing Sequential Numerical Prediction in Autoregressive Models

- [ ] Advancing Sequential Numerical Prediction in Autoregressive Models | https://aclanthology.org/2025.acl-short.44/

- **Link**: https://aclanthology.org/2025.acl-short.44/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autoregressive models have become the de facto choice for sequence generation tasks, but standard approaches treat digits as independent tokens and apply cross-entropy loss, overlooking the coherent structure of numerical sequences. This paper introducesNumericalTokenIntegrityLoss(NTIL)to address this gap. NTIL operates at two levels: (1) token-level, where it extends the Earth Mover’s Distance (EMD) to preserve ordinal relationships between numerical values, and (2) sequence-level, where it penalizes the overall discrepancy between the predicted and actual sequences. This dual approach improves numerical prediction and integrates effectively with LLMs/MLLMs. Extensive experiments show significant performance improvements with NTIL.

</details>

---

## 170. ConECTDataset: Overcoming Data Scarcity in Context-AwareE-CommerceMT

- [ ] ConECTDataset: Overcoming Data Scarcity in Context-AwareE-CommerceMT | https://aclanthology.org/2025.acl-short.7/

- **Link**: https://aclanthology.org/2025.acl-short.7/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Neural Machine Translation (NMT) has improved translation by using Transformer-based models, but it still struggles with word ambiguity and context. This problem is especially important in domain-specific applications, which often have problems with unclear sentences or poor data quality. Our research explores how adding information to models can improve translations in the context of e-commerce data. To this end we create ConECT– a new Czech-to-Polish e-commerce product translation dataset coupled with images and product metadata consisting of 11,400 sentence pairs. We then investigate and compare different methods that are applicable to context-aware translation. We test a vision-language model (VLM), finding that visual context aids translation quality. Additionally, we explore the incorporation of contextual information into text-to-text models, such as the product’s category path or image descriptions. The results of our study demonstrate that the incorporation of contextual information leads to an improvement in the quality of machine translation. We make the new dataset publicly available.

</details>

---

## 171. Transferring Textual Preferences to Vision-Language Understanding through Model Merging

- [ ] Transferring Textual Preferences to Vision-Language Understanding through Model Merging | https://aclanthology.org/2025.acl-short.72/

- **Link**: https://aclanthology.org/2025.acl-short.72/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) perform outstandingly across various multimodal tasks. However, their ability to evaluate generated content remains limited, and training vision-language reward models (VLRMs) with preference data is computationally expensive. This paper explores a training-free alternative by merging text-based reward models (RMs) with LVLMs to create VLRMs. Our approach shows that integrating these models leads to improved performance over LVLMs’ scoring and text-based RMs, offering an efficient method for incorporating textual preferences into LVLMs.

</details>

---

## 172. WinSpot:GUIGrounding Benchmark with Multimodal Large Language Models

- [ ] WinSpot:GUIGrounding Benchmark with Multimodal Large Language Models | https://aclanthology.org/2025.acl-short.85/

- **Link**: https://aclanthology.org/2025.acl-short.85/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Graphical User Interface (GUI) automation relies on accurate GUI grounding. However, obtaining large-scale, high-quality labeled data remains a key challenge, particularly in desktop environments like Windows Operating System (OS). Existing datasets primarily focus on structured web-based elements, leaving a gap in real-world GUI interaction data for non-web applications. To address this, we introduce a new framework that leverages LLMs to generate large-scale GUI grounding data, enabling automated and scalable labeling across diverse interfaces. To ensure high accuracy and reliability, we manually validated and refined 5,000 GUI coordinate-instruction pairs, creating WinSpot—the first benchmark specifically designed for GUI grounding tasks in Windows environments. WinSpot provides a high-quality dataset for training and evaluating visual GUI agents, establishing a foundation for future research in GUI automation across diverse and unstructured desktop environments.

</details>

---

## 173. Chart Question Answering from Real-World Analytical Narratives

- [ ] Chart Question Answering from Real-World Analytical Narratives | https://aclanthology.org/2025.acl-srw.50/

- **Link**: https://aclanthology.org/2025.acl-srw.50/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a new dataset for chart question answering (CQA) constructed from visualization notebooks. The dataset features real-world, multi-view charts paired with natural language questions grounded in analytical narratives. Unlike prior benchmarks, our data reflects ecologically valid reasoning workflows. Benchmarking state-of-the-art multimodal large language models reveals a significant performance gap, with GPT-4.1 achieving an accuracy of 69.3%, underscoring the challenges posed by this more authentic CQA setting.

</details>

---

## 174. DRUM: Learning Demonstration Retriever for LargeMUlti-modal Models

- [ ] DRUM: Learning Demonstration Retriever for LargeMUlti-modal Models | https://aclanthology.org/2025.acl-srw.83/

- **Link**: https://aclanthology.org/2025.acl-srw.83/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, large language models (LLMs) have demonstrated impressive capabilities in dealing with new tasks with the help of in-context learning (ICL). In the study of Large Vision-Language Models (LVLMs), when implementing ICL, researchers usually adopt the naive strategies like fixed demonstrations across different samples, or selecting demonstrations directly via a visual-language embedding model. These methods do not guarantee the configured demonstrations fit the need of the LVLMs. To address this issue, we propose a novel framework, demonstration retriever for large multi-modal model (DRUM), which fine-tunes the CLIP embedding model to better meet the LVLM’s needs. First, we discuss the retrieval strategies for a visual-language task, assuming an embedding model is given. And we propose to concate the image and text embeddings to enhance the retrieval performance. Second, we propose to re-rank the the embedding model’s retrieved demonstrations via the LVLM’s feedbacks, and calculate a list-wise ranking loss for training the embedding model. Third, we propose an iterative demonstration mining strategy to improve the training of the embedding model. Through extensive experiments on 3 types of visual-language tasks, 7 benchmark datasets, our DRUM framework is proven to be effective in boosting the LVLM’s in-context learning performance via retrieving more proper demonstrations.

</details>

---

## 175. Challenging MultimodalLLMs withAfrican Standardized Exams: A DocumentVQAEvaluation

- [ ] Challenging MultimodalLLMs withAfrican Standardized Exams: A DocumentVQAEvaluation | https://aclanthology.org/2025.africanlp-1.22/

- **Link**: https://aclanthology.org/2025.africanlp-1.22/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite rapid advancements in multimodal large language models (MLLMs), their ability to process low-resource African languages in document-based visual question answering (VQA) tasks remains limited. This paper evaluates three state-of-the-art MLLMs—GPT-4o, Claude-3.5 Haiku, and Gemini-1.5 Pro—on WAEC/NECO standardized exam questions in Yoruba, Igbo, and Hausa. We curate a dataset of multiple-choice questions from exam images and compare model accuracies across two prompting strategies: (1) using English prompts for African language questions, and (2) using native-language prompts. While GPT-4o achieves over 90% accuracy for English, performance drops below 40% for African languages, highlighting severe data imbalance in model training. Notably, native-language prompting improves accuracy for most models, yet no system approaches human-level performance, which reaches over 50% in Yoruba, Igbo, and Hausa. These findings emphasize the need for diverse training data, fine-tuning, and dedicated benchmarks that address the linguistic intricacies of African languages in multimodal tasks, paving the way for more equitable and effective AI systems in education.

</details>

---

## 176. Testing Spatial Intuitions of Humans and Large Language and Multimodal Models in Analogies

- [ ] Testing Spatial Intuitions of Humans and Large Language and Multimodal Models in Analogies | https://aclanthology.org/2025.analogyangle-1.9/

- **Link**: https://aclanthology.org/2025.analogyangle-1.9/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Language and Vision-Language Models exhibit impressive language capabilities akin to human reasoning. However, unlike humans who acquire language through embodied, interactive experiences, these models learn from static datasets without real-world interaction. This difference raises questions about how they conceptualize abstract notions and whether their reasoning aligns with human cognition. We investigate spatial conceptualizations of LLMs and VLMs by conducting analogy prompting studies with LLMs, VLMs, and human participants. We assess their ability to generate and interpret analogies for spatial concepts. We quantitatively compare the analogies produced by each group, examining the impact of multimodal inputs and reasoning mechanisms. Our findings indicate that generative models can produce and interpret analogies but differ significantly from human reasoning in their abstraction of spatial concepts - variability influenced by input modality, model size, and prompting methods, with analogy-based prompts not consistently enhancing alignment. Contributions include a methodology for probing generative models through analogies; a comparative analysis of analogical reasoning among models, and humans; and insights into the effect of multimodal inputs on reasoning.

</details>

---

## 177. MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct

- [ ] MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct | https://aclanthology.org/2025.findings-acl.1009/

- **Link**: https://aclanthology.org/2025.findings-acl.1009/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of Multimodal Large Language Models (MLLMs) has seen significant progress, driven by increasing demands across various fields (e.g., multimodal agents, embodied intelligence). While model-driven approaches aim to enhance MLLM capabilities through diverse architectures, their performance gains have become increasingly marginal. In contrast, data-driven methods, which scale up image-text instruction datasets, have proven more effective but face challenges related to limited data diversity and complexity. The absence of high-quality instruction data remains a major bottleneck in MLLM development. To address this issue, we propose , a novel multimodal instruction data evolution framework. This framework iteratively enhances data quality through a refined combination of fine-grained perception, cognitive reasoning, and interaction evolution, generating a more complex and diverse image-text instruction dataset that significantly improves MLLM capabilities. Starting with an initial dataset, SEED-163K, we employ to systematically expand instruction diversity, extend visual reasoning steps to improve cognitive abilities, and extract fine-grained visual details to enhance understanding and robustness. To rigorously evaluate our approach, we conduct extensive qualitative analysis and quantitative experiments across 13 vision-language tasks. Compared to baseline models trained on the original seed dataset, our method achieves an average accuracy improvement of 3.1 percentage points. Moreover, our approach attains state-of-the-art (SOTA) performance in nine tasks while using significantly less data than existing state-of-the-art models.

</details>

---

## 178. Multimodal Large Language Models for Text-rich Image Understanding: A Comprehensive Review

- [ ] Multimodal Large Language Models for Text-rich Image Understanding: A Comprehensive Review | https://aclanthology.org/2025.findings-acl.1023/

- **Link**: https://aclanthology.org/2025.findings-acl.1023/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The recent emergence of Multi-modal Large Language Models (MLLMs) has introduced a new dimension to the Text-rich Image Understanding (TIU) field, with models demonstrating impressive and inspiring performance. However, their rapid evolution and widespread adoption have made it increasingly challenging to keep up with the latest advancements. To address this, we present a systematic and comprehensive survey to facilitate further research on TIU MLLMs. Initially, we outline the timeline, architecture, and pipeline of nearly all TIU MLLMs. Then, we review the performance of selected models on mainstream benchmarks. Finally, we explore promising directions, challenges, and limitations within the field.

</details>

---

## 179. Unveiling the Lack ofLVLMRobustness to Fundamental Visual Variations: Why and Path Forward

- [ ] Unveiling the Lack ofLVLMRobustness to Fundamental Visual Variations: Why and Path Forward | https://aclanthology.org/2025.findings-acl.1037/

- **Link**: https://aclanthology.org/2025.findings-acl.1037/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision Language Models (LVLMs) have shown impressive performance on various vision-language tasks. However, while objects in natural scenes inevitably exhibit visual variations in position, scale, orientation, and context due to changes in viewpoint and environment, the robustness of LVLMs to these fundamental visual variations remains largely unexplored. To address this gap, we introduce V²R-Bench, a comprehensive benchmark framework for evaluating Visual Variation Robustness of LVLMs, which encompasses automated evaluation dataset generation and principled metrics for thorough robustness assessment. Through extensive evaluation of 13 LVLMs, we reveal a surprising vulnerability to visual variations, affecting even advanced models that excel at complex vision-language tasks yet significantly underperform on simple tasks like object recognition. Interestingly, these models exhibit a distinct visual position bias that contradicts theories of effective receptive fields and demonstrate a human-like visual acuity threshold. To identify the source of these vulnerabilities, we propose a systematic framework for component-level analysis, featuring a novel visualization approach for aligned visual features. Results show that these vulnerabilities stem from error accumulation in the pipeline architecture and inadequate multimodal alignment. Complementary experiments with synthetic data further demonstrate that these limitations are fundamentally architectural challenges, underscoring the need for architectural innovations in future LVLM designs.

</details>

---

## 180. InImageTrans: MultimodalLLM-based Text Image Machine Translation

- [ ] InImageTrans: MultimodalLLM-based Text Image Machine Translation | https://aclanthology.org/2025.findings-acl.1039/

- **Link**: https://aclanthology.org/2025.findings-acl.1039/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have shown remarkable capabilities across various downstream tasks. However, when MLLMs are transferred to the text image machine translation (TiMT) task, preliminary experiments reveal that MLLMs suffer from serious repetition and omission hallucinations. To alleviate these issues, this paper first designs an efficient MLLM named InImageTrans for TiMT and then proposes a simple and effective method named multi-conditional direct preference optimization (mcDPO) for advancing the TiMT. Particularly, the proposed mcDPO not only guides the MLLM in rejecting repetition output by creating text output preference pairs automatically, but also guides the MLLM in paying more attention to text information in images by creating image input preference pairs. Furthermore, we build a high-quality benchmark called MCiT for comprehensively evaluating the TiMT capabilities of InImageTrans. Experimental results show that the proposed method significantly outperforms existing open-source MLLMs on MCiT.

</details>

---

## 181. LLMVoX: Autoregressive Streaming Text-to-Speech Model for AnyLLM

- [ ] LLMVoX: Autoregressive Streaming Text-to-Speech Model for AnyLLM | https://aclanthology.org/2025.findings-acl.1051/

- **Link**: https://aclanthology.org/2025.findings-acl.1051/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in speech-to-speech dialogue systems leverage LLMs for multimodal interactions, yet they remain hindered by fine-tuning requirements, high computational overhead, and text-speech misalignment. Existing speech-enabled LLMs often degrade conversational quality by modifying the LLM, thereby compromising its linguistic capabilities. In contrast, we propose LLMVoX, a lightweight 30M-parameter, LLM-agnostic, autoregressive streaming TTS system that generates high-quality speech with low latency, while fully preserving the capabilities of the base LLM. Our approach achieves a significantly lower Word Error Rate compared to speech-enabled LLMs, while operating at comparable latency. By decoupling speech synthesis from LLM processing via a multi-queue token streaming system, LLMVoX enables seamless, infinite-length dialogues. Its plug-and-play design also facilitates extension to various tasks with different backbones. Furthermore, LLMVoX generalizes to new languages with minimal dataset adaptation, attaining a low Character Error Rate on an Arabic speech task. Evaluations demonstrate that LLMVoX matches or surpasses existing speech-enabled LLMs in both speech quality and latency, while maintaining the original linguistic strengths of the LLM. Additionally, we have integrated LLMVoX with a Vision-Language Model to create an omni-model with speech, text, and vision capabilities, without requiring additional multimodal training.

</details>

---

## 182. HarnessingPDFData for ImprovingJapanese Large Multimodal Models

- [ ] HarnessingPDFData for ImprovingJapanese Large Multimodal Models | https://aclanthology.org/2025.findings-acl.108/

- **Link**: https://aclanthology.org/2025.findings-acl.108/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Multimodal Models (LMMs) have demonstrated strong performance in English, but their effectiveness in Japanese remains limited due to the lack of high-quality training data. Current Japanese LMMs often rely on translated English datasets, restricting their ability to capture Japan-specific cultural knowledge. To address this, we explore the potential of Japanese PDF data as a training resource, an area that remains largely underutilized. We introduce a fully automated pipeline that leverages pretrained models to extract image-text pairs from PDFs through layout analysis, OCR, and vision-language pairing, removing the need for manual annotation. Additionally, we construct instruction data from extracted image-text pairs to enrich the training data. To evaluate the effectiveness of PDF-derived data, we train Japanese LMMs and assess their performance on the Japanese LMM Benchmark. Our results demonstrate substantial improvements, with performance gains ranging from 2.1% to 13.8% on Heron-Bench. Further analysis highlights the impact of PDF-derived data on various factors, such as model size and language models, reinforcing its value as a multimodal resource for Japanese LMMs.

</details>

---

## 183. C²RBench: AChinese Complex Reasoning Benchmark for Large Language Models

- [ ] C²RBench: AChinese Complex Reasoning Benchmark for Large Language Models | https://aclanthology.org/2025.findings-acl.1083/

- **Link**: https://aclanthology.org/2025.findings-acl.1083/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have achieved remarkable progress in autonomous reasoning, evolving from basic text processing to sophisticated multimodal reasoning, a critical capability for general-purpose AI assistants. However, existing benchmarks usually fail to adequately capture the intricate multi-step reasoning demands inherent in real-world scenarios. To bridge this gap, we propose **C²RBench**: a **C**hinese **C**omplex **R**easoning **Bench**mark for evaluating multi-step, multimodal advanced reasoning capability of LLMs. C²RBench comprises 1,115 carefully curated Chinese tasks, which are organized into eight domain-specific subsets, each meticulously designed to mirror real-world challenges. This hierarchical benchmark features three difficulty tiers based on the number of reasoning steps required (average 8.44 steps per task), significantly exceeding existing benchmarks in cognitive complexity. Extensive evaluations of 20 LLMs (including DeepSeek-R1) and 24 multimodal large language models (MLLMs) on C²RBench reveal critical performance gaps: GPT-4.1 achieves only 52.11% accuracy, indicating substantial room for improvement. The dataset and evaluation code are publicly available.

</details>

---

## 184. VideoRAG: Retrieval-Augmented Generation over Video Corpus

- [ ] VideoRAG: Retrieval-Augmented Generation over Video Corpus | https://aclanthology.org/2025.findings-acl.1096/

- **Link**: https://aclanthology.org/2025.findings-acl.1096/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-Augmented Generation (RAG) is a powerful strategy for improving the factual accuracy of models by retrieving external knowledge relevant to queries and incorporating it into the generation process. However, existing approaches primarily focus on text, with some recent advancements considering images, and they largely overlook videos, a rich source of multimodal knowledge capable of representing contextual details more effectively than any other modality. Also, while very recent studies explore the use of videos in response generation, they either predefine query-associated videos without retrieval or convert videos into textual descriptions, losing multimodal richness. To tackle these, we introduce VideoRAG, a novel framework that not only dynamically retrieves videos based on their relevance with queries but also utilizes both visual and textual information. The operation of VideoRAG is powered by recent Large Video Language Models (LVLMs), which enable the direct processing of video content to represent it for retrieval and the seamless integration of retrieved videos jointly with queries for response generation. Also, inspired by that the context size of LVLMs may not be sufficient to process all frames in extremely long videos and not all frames are equally important, we introduce a video frame selection mechanism to extract the most informative subset of frames, along with a strategy to extract textual information from videos (as it can aid the understanding of video content) when their subtitles are not available. We experimentally validate the effectiveness of VideoRAG, showcasing that it is superior to relevant baselines. Our code is available at https://github.com/starsuzi/VideoRAG.

</details>

---

## 185. CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents

- [ ] CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents | https://aclanthology.org/2025.findings-acl.1113/

- **Link**: https://aclanthology.org/2025.findings-acl.1113/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The development of autonomous agents increasingly relies on Multimodal Language Models (MLMs) to perform tasks described in natural language with GUI environments, such as websites, desktop computers, or mobile phones. Existing benchmarks for MLM agents in interactive environments are limited by their focus on a single environment, lack of detailed and generalized evaluation methods, and thecomplexities of constructing tasks and evaluators. To overcome these limitations, we introduce CRAB, the first cross-environment agent benchmark framework, incorporating a graph-based fine-grained evaluation method and an efficient task generation method. Our framework supports multiple devices and can be easily extended to any environment with a Python interface. Leveraging CRAB, we develope CRAB Benchmark-v0 comprising 120 tasks in computer desktop and mobile phone environments. We evaluated 6 advanced MLMs using different single and multi-agent system configurations on this benchmark. The experimental results demonstrate that the single agent with GPT-4o achieves the best completion ratio of 38.01%.

</details>

---

## 186. READoc: A Unified Benchmark for Realistic Document Structured Extraction

- [ ] READoc: A Unified Benchmark for Realistic Document Structured Extraction | https://aclanthology.org/2025.findings-acl.1128/

- **Link**: https://aclanthology.org/2025.findings-acl.1128/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Document Structured Extraction (DSE) aims to extract structured content from raw documents. Despite the emergence of numerous DSE systems, their unified evaluation remains inadequate, significantly hindering the field’s advancement. This problem is largely attributed to existing benchmark paradigms, which exhibit fragmented and localized characteristics. To offer a thorough evaluation of DSE systems, we introduce a novel benchmark named READoc, which defines DSE as a realistic task of converting unstructured PDFs into semantically rich Markdown. The READoc dataset is derived from 3,576 diverse and real-world documents from arXiv, GitHub, and Zenodo. In addition, we develop a DSE Evaluation S3uite comprising Standardization, Segmentation and Scoring modules, to conduct a unified evaluation of state-of-the-art DSE approaches. By evaluating a range of pipeline tools, expert visual models, and general Vision-Language Models, we identify the gap between current work and the unified, realistic DSE objective for the first time. We aspire that READoc will catalyze future research in DSE, fostering more comprehensive and practical solutions.

</details>

---

## 187. KITAB-Bench: A Comprehensive Multi-Domain Benchmark forArabicOCRand Document Understanding

- [ ] KITAB-Bench: A Comprehensive Multi-Domain Benchmark forArabicOCRand Document Understanding | https://aclanthology.org/2025.findings-acl.1135/

- **Link**: https://aclanthology.org/2025.findings-acl.1135/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the growing adoption of Retrieval-Augmented Generation (RAG) in document processing, robust text recognition has become increasingly critical for knowledge extraction. While OCR (Optical Character Recognition) for English and other languages benefits from large datasets and well-established benchmarks, Arabic OCR faces unique challenges due to its cursive script, right-to-left text flow, and complex typographic and calligraphic features. We present KITAB-Bench, a comprehensive Arabic OCR benchmark that fills the gaps in current evaluation systems. Our benchmark comprises 8,809 samples across 9 major domains and 36 subdomains, encompassing diverse document types including handwritten text, structured tables, and specialized coverage of 21 chart types for business intelligence. Our findings show that modern vision language models (such as GPT-4o, Gemini, and Qwen) outperform traditional OCR approaches (such as EasyOCR, PaddleOCR, and Surya) by an average of 60% in the character error rate (CER). Furthermore, we highlight significant limitations of current Arabic OCR models, particularly in PDF-to-Markdown conversion, where the best model Gemini-2.0-Flash achieves only 65% accuracy. This underscores the challenges of accurately recognizing Arabic text, including issues with complex fonts, numeral recognition errors, word elongation, and table structure detection. This work establishes a rigorous evaluation framework that can drive improvements in Arabic document analysis methods and bridge the performance gap with English OCR technologies.

</details>

---

## 188. Creating a Lens ofChinese Culture: A Multimodal Dataset forChinese Pun Rebus Art Understanding

- [ ] Creating a Lens ofChinese Culture: A Multimodal Dataset forChinese Pun Rebus Art Understanding | https://aclanthology.org/2025.findings-acl.1155/

- **Link**: https://aclanthology.org/2025.findings-acl.1155/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (VLMs) have demonstrated remarkable abilities in understanding everyday content. However, their performance in the domain of art, particularly culturally rich art forms, remains less explored. As a pearl of human wisdom and creativity, art encapsulates complex cultural narratives and symbolism. In this paper, we offer the Pun Rebus Art Dataset, a multimodal dataset for art understanding deeply rooted in traditional Chinese culture. We focus on three primary tasks: identifying salient visual elements, matching elements with their symbolic meanings, and explanations for the conveyed messages. Our evaluation reveals that state-of-the-art VLMs struggle with these tasks, often providing biased and hallucinated explanations and showing limited improvement through in-context learning. By releasing the Pun Rebus Art Dataset, we aim to facilitate the development of VLMs that can better understand and interpret culturally specific content, promoting greater inclusiveness beyond English-based corpora. The dataset and evaluation code are available at [this link](https://github.com/zhang-tuo-pdf/Pun-Rebus-Art-Benchmark).

</details>

---

## 189. Explain then Rank: Scale Calibration of Neural Rankers Using Natural Language Explanations fromLLMs

- [ ] Explain then Rank: Scale Calibration of Neural Rankers Using Natural Language Explanations fromLLMs | https://aclanthology.org/2025.findings-acl.1167/

- **Link**: https://aclanthology.org/2025.findings-acl.1167/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In search settings, calibrating the scores during the ranking process to quantities such as click-through rates or relevance levels enhances a system’s usefulness and trustworthiness for downstream users. While previous research has improved this notion of calibration for low complexity learning-to-rank models, the larger data demands and parameter count specific to modern neural text rankers produce unique obstacles that hamper the efficacy of methods intended for the learning-to-rank setting.This paper proposes exploiting large language models (LLMs) to provide relevance and uncertainty signals for these neural text rankers to produce scale-calibrated scores through Monte Carlo sampling of natural language explanations (NLEs). Our approach transforms the neural ranking task from ranking textual query-document pairs to ranking corresponding synthesized NLEs. Comprehensive experiments on two popular document ranking datasets show that the NLE-based calibration approach consistently outperforms past calibration methods and LLM-based methods for ranking, calibration, and query performance prediction tasks.

</details>

---

## 190. Texts or Images? A Fine-grained Analysis on the Effectiveness of Input Representations and Models for Table Question Answering

- [ ] Texts or Images? A Fine-grained Analysis on the Effectiveness of Input Representations and Models for Table Question Answering | https://aclanthology.org/2025.findings-acl.117/

- **Link**: https://aclanthology.org/2025.findings-acl.117/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In table question answering (TQA), tables are encoded as either texts or images. Prior work suggests that passing images of tables to multi-modal large language models (MLLMs) performs comparably to using textual input with large language models (LLMs). However, the lack of controlled setups limits fine-grained distinctions between these approaches. In this paper, we conduct the first controlled study on the effectiveness of several combinations of table representations and model types from two perspectives: question complexity and table size. We build a new benchmark based on existing TQA datasets. In a systematic analysis of seven pairs of MLLMs and LLMs, we find that the best combination of table representation and model varies across setups. We propose FRES, a method selecting table representations dynamically, and observe a 10% average performance improvement compared to using both representations indiscriminately.

</details>

---

## 191. JustKIDDIN’ : Knowledge Infusion and Distillation for Detection ofINdecent Memes

- [ ] JustKIDDIN’ : Knowledge Infusion and Distillation for Detection ofINdecent Memes | https://aclanthology.org/2025.findings-acl.1184/

- **Link**: https://aclanthology.org/2025.findings-acl.1184/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Detecting toxicity in online multimodal environments, such as memes, remains a challenging task due to the complex contextual connections across modalities (e.g., text and visual), which demand both common-sense reasoning and contextual awareness. To bridge this gap, we propose a hybrid neurosymbolic framework that unifies (1) distillation of implicit contextual knowledge (e.g., sarcasm, cultural references) from Large Vision-Language Models (LVLMs) and (2) infusion of explicit relational semantics through sub-graphs from Knowledge Graphs (KGs). Experimental results on two benchmark datasets show the superior performance of our approach,Knowledge-Infused Distilled Vision-Language Model (KID-VLM), over the state-of-the-art baselines across AUC and F1, with improvements of 0.5%, and 10.6%, respectively, in HatefulMemes Benchmark across variants. Further, KID-VLM demonstrates better generalizability and achieves the best performance across all baselines in the HarMeme Dataset with a 6.3% and 3.2% in F1 and AUC.Given the contextual complexity of the toxicity detection, KID-VLM showcases the significance of learning compact models (~500M parameters) from both explicit (i.e., KG) and implicit (i.e., LVLMs) contextual cues incorporated through a hybrid neurosymbolic approach. Our codes and pretrained models are publicly available.

</details>

---

## 192. FREE: Fast and Robust Vision Language Models with Early Exits

- [ ] FREE: Fast and Robust Vision Language Models with Early Exits | https://aclanthology.org/2025.findings-acl.1209/

- **Link**: https://aclanthology.org/2025.findings-acl.1209/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In recent years, Vision-Language Models (VLMs) have shown remarkable performance improvements in Vision-Language tasks. However, their large size poses challenges for real-world applications where inference latency is a concern. To tackle this issue, we propose employing Early Exit (EE) strategies in VLMs. However, training exit classifiers in VLMs is challenging, particularly with limited labeled training data. To address this, we introduce FREE, an adversarial training approach within a GAN-based framework. Here, each exit consists of a transformer layer and a classifier. The transformer layer is adversarially trained to produce feature representations similar to the final layer, while a feature classifier serves as the discriminator. Our method focuses on performing input-adaptive inference that increases inference speed with minimal drop in performance. Experimental results demonstrate the effectiveness of our approach in enhancing accuracy and model robustness by mitigating overthinking and the phenomenon of mid-crisis that we highlight. We experimentally validate that our method speeds up the inference process by more than1.51×while retaining comparable performance. The anonymized source code is available at https://github.com/Div290/BLIPEE.

</details>

---

## 193. ImprovingMLLM’s Document Image Machine Translation via Synchronously Self-reviewing ItsOCRProficiency

- [ ] ImprovingMLLM’s Document Image Machine Translation via Synchronously Self-reviewing ItsOCRProficiency | https://aclanthology.org/2025.findings-acl.1213/

- **Link**: https://aclanthology.org/2025.findings-acl.1213/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown strong performance in document image tasks, especially Optical Character Recognition (OCR). However, they struggle with Document Image Machine Translation (DIMT), which requires handling both cross-modal and cross-lingual challenges. Previous efforts to enhance DIMT capability through Supervised Fine-Tuning (SFT) on the DIMT dataset often result in the forgetting of the model’s existing monolingual abilities, such as OCR. To address these challenges, we introduce a novel fine-tuning paradigm, named Synchronously Self-Reviewing (SSR) its OCR proficiency, inspired by the concept “Bilingual Cognitive Advantage”. Specifically, SSR prompts the model to generate OCR text before producing translation text, which allows the model to leverage its strong monolingual OCR ability while learning to translate text across languages. Comprehensive experiments demonstrate the proposed SSR learning helps mitigate catastrophic forgetting, improving the generalization ability of MLLMs on both OCR and DIMT tasks. The code will be released upon acceptance.

</details>

---

## 194. Revisiting 3DLLMBenchmarks: Are We Really Testing 3DCapabilities?

- [ ] Revisiting 3DLLMBenchmarks: Are We Really Testing 3DCapabilities? | https://aclanthology.org/2025.findings-acl.1222/

- **Link**: https://aclanthology.org/2025.findings-acl.1222/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we identify the “2D-Cheating” problem in 3D LLM evaluation, where these tasks might be easily solved by VLMs with rendered images of point clouds, exposing ineffective evaluation of 3D LLMs’ unique 3D capabilities. We test VLM performance across multiple 3D LLM benchmarks and, using this as a reference, propose principles for better assessing genuine 3D understanding. We also advocate explicitly separating 3D abilities from 1D or 2D aspects when evaluating 3D LLMs.

</details>

---

## 195. SceneGram: Conceptualizing and Describing Tangrams in Scene Context

- [ ] SceneGram: Conceptualizing and Describing Tangrams in Scene Context | https://aclanthology.org/2025.findings-acl.1229/

- **Link**: https://aclanthology.org/2025.findings-acl.1229/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Research on reference and naming suggests that humans can come up with very different ways of conceptualizing and referring to the same object, e.g. the same abstract tangram shape can be a “crab”, “sink” or “space ship”. Another common assumption in cognitive science is that scene context fundamentally shapes our visual perception of objects and conceptual expectations. This paper contributes SceneGram, a dataset of human references to tangram shapes placed in different scene contexts, allowing for systematic analyses of the effect of scene context on conceptualization. Based on this data, we analyze references to tangram shapes generated by multimodal LLMs, showing that these models do not account for the richness and variability of conceptualizations found in human references.

</details>

---

## 196. RedundancyLens: Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-OnlyMLLMs

- [ ] RedundancyLens: Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-OnlyMLLMs | https://aclanthology.org/2025.findings-acl.1233/

- **Link**: https://aclanthology.org/2025.findings-acl.1233/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current Multimodal Large Language Model (MLLM) architectures face a critical tradeoff between performance and efficiency: decoder-only architectures achieve higher performance but lower efficiency, while cross-attention-based architectures offer greater efficiency but lower performance. The key distinction lies in how visual tokens are processed. Decoder-only architectures apply self-attention and FFN operations on visual tokens, while cross-attention architectures skip these computations. To investigate whether redundancy exists in this computationally expensive process, we propose a training-free framework for analyzing trained MLLMs. It consists of Probe-Activated Dynamic FFN and Hollow Attention, which enable adjustable reductions in computations for visual tokens, as well as a Layer Ranking Algorithm that prioritizes layers for these reductions. Extensive experiments demonstrate substantial, structured, and clustered redundancy unique to decoder-only MLLMs, offering valuable insights for future MLLM architecture design. Furthermore, by leveraging our reduction framework as a training-free inference acceleration approach, we achieve performance comparable to or better than state-of-the-art methods while remaining compatible with them. Code is available at https://github.com/L-Hugh/RedundancyLens.

</details>

---

## 197. Are Multimodal Large Language Models Pragmatically Competent Listeners in Simple Reference Resolution Tasks?

- [ ] Are Multimodal Large Language Models Pragmatically Competent Listeners in Simple Reference Resolution Tasks? | https://aclanthology.org/2025.findings-acl.1236/

- **Link**: https://aclanthology.org/2025.findings-acl.1236/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We investigate the linguistic abilities of multimodal large language models in reference resolution tasks featuring simple yet abstract visual stimuli, such as color patches and color grids. Although the task may not seem challenging for today’s language models, being straightforward for human dyads, we consider it to be a highly relevant probe of the pragmatic capabilities of MLLMs. Our results and analyses indeed suggest that basic pragmatic capabilities, such as context-dependent interpretation of color descriptions, still constitute major challenges for state-of-the-art MLLMs.

</details>

---

## 198. Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?

- [ ] Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences? | https://aclanthology.org/2025.findings-acl.1248/

- **Link**: https://aclanthology.org/2025.findings-acl.1248/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces the TempVS benchmark, which focuses on temporal grounding and reasoning capabilities of Multimodal Large Language Models (MLLMs) in image sequences. TempVS consists of three main tests (i.e., event relation inference, sentence ordering and image ordering), each accompanied with a basic grounding test. TempVS requires MLLMs to rely on both visual and linguistic modalities to understand the temporal order of events. We evaluate 38 state-of-the-art MLLMs, demonstrating that models struggle to solve TempVS, with a substantial performance gap compared to human capabilities. We also provide fine-grained insights that suggest promising directions for future research. Our TempVS benchmark data and code are available at https://github.com/yjsong22/TempVS.

</details>

---

## 199. CanVLMs Actually See and Read? A Survey on Modality Collapse in Vision-Language Models

- [ ] CanVLMs Actually See and Read? A Survey on Modality Collapse in Vision-Language Models | https://aclanthology.org/2025.findings-acl.1256/

- **Link**: https://aclanthology.org/2025.findings-acl.1256/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) integrate textual and visual information, enabling the model to process visual inputs and leverage visual information to generate predictions. Such models are demanding for tasks such as visual question answering, image captioning, and visual grounding. However, some recent work found that VLMs often rely heavily on textual information, ignoring visual information, but are still able to achieve competitive performance in vision-language (VL) tasks. This survey reviews modality collapse analysis work to provide insights into the reason for this unintended behavior. It also reviews probing studies for fine-grained vision-language understanding, presenting current findings on information encoded in VL representations and highlighting potential directions for future research.

</details>

---

## 200. Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation

- [ ] Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation | https://aclanthology.org/2025.findings-acl.127/

- **Link**: https://aclanthology.org/2025.findings-acl.127/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Self-supervised learning (SSL) has reduced the reliance on expensive labeling in speech technologies by learning meaningful representations from unannotated data. Since most SSL-based downstream tasks prioritize content information in speech, ideal representations should disentangle content from unwanted variations like speaker characteristics in the SSL representations. However, removing speaker information often degrades other speech components, and existing methods either fail to fully disentangle speaker identity or require resource-intensive models. In this paper, we propose a novel disentanglement method that linearly decomposes SSL representations into speaker-specific and speaker-independent components, effectively generating speaker disentangled representations. Comprehensive experiments show that our approach achieves speaker independence and as such, when applied to content-driven tasks such as voice conversion, our representations yield significant improvements over state-of-the-art methods.

</details>

---

## 201. WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts

- [ ] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts | https://aclanthology.org/2025.findings-acl.1280/

- **Link**: https://aclanthology.org/2025.findings-acl.1280/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Documents are fundamental to preserving and disseminating information, often incorporating complex layouts, tables, and charts that pose significant challenges for automatic document understanding (DU). While vision-language large models (VLLMs) have demonstrated improvements across various tasks, their effectiveness in processing long-context vision inputs remains unclear. This paper introduces WikiMixQA, a benchmark comprising 1,000 multiple-choice questions (MCQs) designed to evaluate cross-modal reasoning over tables and charts extracted from 4,000 Wikipedia pages spanning seven distinct topics. Unlike existing benchmarks, WikiMixQA emphasizes complex reasoning by requiring models to synthesize information from multiple modalities. We evaluate 12 state-of-the-art vision-language models, revealing that while proprietary models achieve ~70% accuracy when provided with direct context, their performance deteriorates significantly when retrieval from long documents is required. Among these, GPT-4-o is the only model exceeding 50% accuracy in this setting, whereas open-source models perform considerably worse, with a maximum accuracy of 27%. These findings underscore the challenges of long-context, multi-modal reasoning and establish WikiMixQA as a crucial benchmark for advancing document understanding research.

</details>

---

## 202. From Perception to Reasoning: Enhancing Vision-Language Models for MobileUIUnderstanding

- [ ] From Perception to Reasoning: Enhancing Vision-Language Models for MobileUIUnderstanding | https://aclanthology.org/2025.findings-acl.1295/

- **Link**: https://aclanthology.org/2025.findings-acl.1295/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Accurately grounding visual and textual elements within mobile user interfaces (UIs) remains a significant challenge for Vision-Language Models (VLMs). Visual grounding, a critical task in this domain, involves identifying the most relevant UI element or region based on a natural language query—a process that requires both precise perception and context-aware reasoning. In this work, we present - **MoUI**, a light-weight mobile UI understanding model trained on **MoIT**, an instruction-tuning dataset specifically tailored for mobile screen understanding and grounding, designed to bridge the gap between user intent and visual semantics. Complementing this dataset, we also present a human-annotated reasoning benchmark **MoIQ** that rigorously evaluates complex inference capabilities over mobile UIs. To harness these resources effectively, we propose a two-stage training approach that separately addresses perception and reasoning tasks, leading to stronger perception capabilities and improvement in reasoning abilities. Through extensive experiments, we demonstrate that our MoUI models achieve significant gains in accuracy across all perception tasks and _state-of-the-art_ results on public reasoning benchmark **ComplexQA (78%) and our MoIQ (49%)**. We will be open-sourcing our dataset, code, and models to foster further research and innovation in the field.

</details>

---

## 203. Can Hallucination Correction Improve Video-Language Alignment?

- [ ] Can Hallucination Correction Improve Video-Language Alignment? | https://aclanthology.org/2025.findings-acl.1314/

- **Link**: https://aclanthology.org/2025.findings-acl.1314/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models often generate hallucinated content that is not grounded in its visual inputs. While prior work focuses on mitigating hallucinations, we instead explore leveraging hallucination correction as a training objective to improve video-language alignment. We introduce HACA, a self-training framework learning to correct hallucinations in descriptions that do not align with the video content. By identifying and correcting inconsistencies, HACA enhances the model’s ability to align video and textual representations for spatio-temporal reasoning. Our experimental results show consistent gains in video-caption binding and text-to-video retrieval tasks, demonstrating that hallucination correction-inspired tasks serve as an effective strategy for improving vision and language alignment.

</details>

---

## 204. Cautious Next Token Prediction

- [ ] Cautious Next Token Prediction | https://aclanthology.org/2025.findings-acl.1318/

- **Link**: https://aclanthology.org/2025.findings-acl.1318/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Next token prediction paradigm has been prevailing for autoregressive models in the era of LLMs. The current default sampling choice for popular LLMs is temperature scaling together with nucleus sampling to balance diversity and coherence. Nevertheless, such approach leads to inferior performance in various NLP tasks when the model is not certain about testing questions. To this end, we propose a brand new training-free decoding strategy, dubbed as Cautious Next Token Prediction (CNTP). In the decoding process, if the model has comparatively high prediction entropy at a certain step, we sample multiple trials starting from the step independently and stop when encountering any punctuation. Then we select the trial with the lowest perplexity score viewed as the most probable and reliable trial path given the model’s capacity. The trial number is negatively correlated with the prediction confidence, i.e., the less confident the model is, the more trials it should sample. This is consistent with human beings’ behaviour: when feeling uncertain or unconfident, one tends to think more creatively, exploring multiple thinking paths, to cautiously select the path one feels most confident about. Extensive experiments on both LLMs and MLLMs show that our proposed CNTP approach outperforms existing standard decoding strategies consistently by a clear margin. Moreover, the integration of CNTP with self consistency can further improve over vanilla self consistency. We believe our proposed CNTP has the potential to become one of the default choices for LLM decoding. Code is available at https://github.com/wyzjack/CNTP.

</details>

---

## 205. Blinded by Context: Unveiling the Halo Effect ofMLLMinAIHiring

- [ ] Blinded by Context: Unveiling the Halo Effect ofMLLMinAIHiring | https://aclanthology.org/2025.findings-acl.1338/

- **Link**: https://aclanthology.org/2025.findings-acl.1338/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This study investigates the halo effect in AI-driven hiring evaluations using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Through experiments with hypothetical job applications, we examined how these models’ evaluations are influenced by non-job-related information, including extracurricular activities and social media images. By analyzing models’ responses to Likert-scale questions across different competency dimensions, we found that AI models exhibit significant halo effects, particularly in image-based evaluations, while text-based assessments showed more resistance to bias. The findings demonstrate that supplementary multimodal information can substantially influence AI hiring decisions, highlighting potential risks in AI-based recruitment systems.

</details>

---

## 206. Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation

- [ ] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation | https://aclanthology.org/2025.findings-acl.1342/

- **Link**: https://aclanthology.org/2025.findings-acl.1342/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Internal world models (WMs) enable agents to understand the world’s state and predict transitions, serving as the basis for advanced deliberative reasoning.Recent large Vision-Language Models (VLMs), such as GPT-4o and Gemini, exhibit potential as general-purpose WMs. While the latest studies have evaluated and shown limitations in specific capabilities such as visual understanding, a systematic evaluation of VLMs’ fundamental WM abilities remains absent. Drawing on comparative psychology and cognitive science, we propose a two-stage framework that assesses **perception** (visual, spatial, temporal, quantitative, and motion) and **prediction** (mechanistic simulation, transitive inference, compositional inference) to provide an atomic evaluation of VLMs as WMs. Guided by this framework, we introduce **WM-ABench**, a large-scale benchmark comprising 23 fine-grained evaluation dimensions across 6 diverse simulated environments with controlled counterfactual simulations. Through 660 experiments on 15 latest commercial and open-source VLMs, we find that these models exhibit striking limitations in basic world modeling abilities. For instance, all models perform at near-random accuracy when distinguishing motion trajectories. Additionally, they lack disentangled understanding—e.g., they tend to believe blue objects move faster than green ones. More rich results and analyses reveal significant gaps between VLMs and human-level world modeling.

</details>

---

## 207. M2-TabFact: Multi-Document Multi-Modal Fact Verification with Visual and Textual Representations of Tabular Data

- [ ] M2-TabFact: Multi-Document Multi-Modal Fact Verification with Visual and Textual Representations of Tabular Data | https://aclanthology.org/2025.findings-acl.1345/

- **Link**: https://aclanthology.org/2025.findings-acl.1345/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Tabular data is used to store information in many real-world systems ranging from finance to healthcare. However, such structured data is often communicated to humans in visually interpretable formats (e.g. charts and textual paragraphs), making it imperative that fact-checking models should be able to reason over multiple pieces of structured evidence presented across different modalities. In this paper, we propose Multi-Document Multi-Modal Table-based Fact Verification (M2-TabFact), a challenging fact verification task that requires jointly reasoning over visual and textual representations of structured data. We design an automatic data generation pipeline that converts existing tabular data into descriptive visual and textual evidence. We then use Large Language Models to generate complex claims that depend on multi-document, multi-modal evidence. In total, we create 8,856 pairs of complex claims and multi-modal evidence through this procedure and systematically evaluate M2-TabFact with a set of strong vision-language models (VLM). We find that existing VLMs have large gaps in fact verification performance compared to humans. Moreover, we find that they are imbalanced when it comes to their ability to handle reason about different modalities, and currently struggle to reason about information extracted from multiple documents.

</details>

---

## 208. EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations

- [ ] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations | https://aclanthology.org/2025.findings-acl.1367/

- **Link**: https://aclanthology.org/2025.findings-acl.1367/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in large language models and vision-language models have led to growing interest in explainable evaluation metrics for image captioning. However, these metrics generate explanations without standardized criteria, and the overall quality of the generated explanations remains unverified. In this paper, we propose EXPERT, a reference-free evaluation metric that provides structured explanations based on three fundamental criteria: fluency, relevance, and descriptiveness. By constructing large-scale datasets of high-quality structured explanations, we develop a two-stage evaluation template to effectively supervise a vision-language model for both scoring and explanation generation. EXPERT achieves state-of-the-art results on benchmark datasets while providing significantly higher-quality explanations than existing metrics, as validated through comprehensive human evaluation. Our code and datasets are available at https://github.com/hjkim811/EXPERT.

</details>

---

## 209. Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning

- [ ] Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning | https://aclanthology.org/2025.findings-acl.137/

- **Link**: https://aclanthology.org/2025.findings-acl.137/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Disentanglement of visual features of primitives (i.e., attributes and objects) has shown exceptional results in Compositional Zero-shot Learning (CZSL). However, due to the feature divergence of an attribute (resp. object) when combined with different objects (resp. attributes), it is challenging to learn disentangled primitive features that are general across different compositions. To this end, we propose the solution ofcross-composition feature disentanglement, which takes multiple primitive-sharing compositions as inputs and constrains the disentangled primitive features to be general across these compositions. More specifically, we leverage a compositional graph to define the overall primitive-sharing relationships between compositions, and build a task-specific architecture upon the recently successful large pre-trained vision-language model (VLM) CLIP, with dual cross-composition disentangling adapters (called L-Adapter and V-Adapter) inserted into CLIP’s frozen text and image encoders, respectively. Evaluation on three popular CZSL benchmarks shows that our proposed solution significantly improves the performance of CZSL, and its components have been verified by solid ablation studies. Our code and data are available at: https://github.com/zhurunkai/DCDA.

</details>

---

## 210. Can Vision Language Models Understand Mimed Actions?

- [ ] Can Vision Language Models Understand Mimed Actions? | https://aclanthology.org/2025.findings-acl.1372/

- **Link**: https://aclanthology.org/2025.findings-acl.1372/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Non-verbal communication (NVC) is an integral part of human language, but it has been overlooked in natural language processing research. Studying NVC in general is challenging because of its high variance in interpretation among individuals and cultures, but mime—the theatrical technique of suggesting intent using only gesture, expression, and movement—is a subset of NVC with much lower human interpretation variance. As a gateway for evaluating vision-language models on their understanding of NVC, we propose Mime Identification-based Multimodal Evaluation (MIME), a gesture recognition task built upon a novel corpus of mimed activity comprising 86 unique gestures with a variety of perturbations applied to the avatar, background, and viewpoint for evaluating recognition robustness. We find that both open-weight and API-based vision-language models perform significantly worse than humans at identifying mimed gestures in MIME, motivating the need for increased research for instilling more robust understanding of human actions for VLMs.

</details>

---

## 211. MMRefine: Unveiling the Obstacles to Robust Refinement in Multimodal Large Language Models

- [ ] MMRefine: Unveiling the Obstacles to Robust Refinement in Multimodal Large Language Models | https://aclanthology.org/2025.findings-acl.1378/

- **Link**: https://aclanthology.org/2025.findings-acl.1378/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces MMRefine, a MultiModal Refinement benchmark designed to evaluate the error refinement capabilities of Multimodal Large Language Models (MLLMs). As the emphasis shifts toward enhancing reasoning during inference, MMRefine provides a framework that evaluates MLLMs’ abilities to detect and correct errors across six distinct scenarios beyond just comparing final accuracy before and after refinement. Furthermore, the benchmark analyzes the refinement performance by categorizing errors into six error types.Experiments with various open and closed MLLMs reveal bottlenecks and factors impeding refinement performance, highlighting areas for improvement in effective reasoning enhancement. Our code and dataset are publicly available athttps://github.com/naver-ai/MMRefine.

</details>

---

## 212. ProgressiveLoRAfor Multimodal Continual Instruction Tuning

- [ ] ProgressiveLoRAfor Multimodal Continual Instruction Tuning | https://aclanthology.org/2025.findings-acl.143/

- **Link**: https://aclanthology.org/2025.findings-acl.143/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Continual Instruction Tuning (MCIT) empowers Multimodal Large Language Models (MLLMs) to adapt to ever-evolving requirements without continuous costly retraining. However, MCIT faces challenges in mitigating Catastrophic Forgetting (CF) and enhancing Knowledge Transfer (KT). Existing works combine Mixture-of-Expert (MoE) and LoRA to address these. However, using a fixed number of shared LoRA blocks across tasks can lead to the overwriting of acquired knowledge, making MLLMs harder to handle CF and KT. Therefore, we propose the **Prog**ressive **LoRA** framework (ProgLoRA), which contains a progressive LoRA pool and trains a new LoRA block for each incremental task to reduce knowledge interference. Specifically, ProgLoRA has two key mechanisms: task-aware allocation for effectively leveraging acquired knowledge at current task and task recall for realigning the model with learned tasks. Additionally, considering different application scenarios, we design a static ProgLoRA for the more idealized basic setting and a dynamic ProgLoRA for the more realistic challenging setting. Experiments on the latest MCIT benchmark demonstrate that ProgLoRA outperforms existing approaches.

</details>

---

## 213. Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation

- [ ] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation | https://aclanthology.org/2025.findings-acl.151/

- **Link**: https://aclanthology.org/2025.findings-acl.151/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Counterfactual reasoning is crucial for robust video understanding but remains underexplored in existing multimodal benchmarks. In this paper, we introduce **COVER** (**CO**unterfactual **V**id**E**o **R**easoning), a multidimensional multimodal benchmark that systematically evaluates MLLMs across the abstract-concrete and perception-cognition dimensions. Beyond prior multimodal benchmarks, COVER decomposes complex queries into structured sub-questions, enabling fine-grained reasoning analysis. Experiments on commercial and open-source models reveal a strong correlation between sub-question accuracy and counterfactual reasoning performance, highlighting the role of structured inference in video understanding. Furthermore, our results suggest a key insight: enhancing the reasoning capability of models is essential for improving the robustness of video understanding. COVER establishes a new standard for assessing MLLMs’ logical reasoning abilities in dynamic environments. Our work is available at https://github.com/gongyifan-hash/COVER-Benchmark.

</details>

---

## 214. VSCBench: Bridging the Gap in Vision-Language Model Safety Calibration

- [ ] VSCBench: Bridging the Gap in Vision-Language Model Safety Calibration | https://aclanthology.org/2025.findings-acl.158/

- **Link**: https://aclanthology.org/2025.findings-acl.158/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of vision-language models (VLMs) has brought a lot of attention to their safety alignment. However, existing methods have primarily focused on model undersafety, where the model responds to hazardous queries, while neglecting oversafety, where the model refuses to answer safe queries. In this paper, we introduce the concept of safety calibration, which systematically addresses both undersafety and oversafety. Specifically, we present VSCBench, a novel dataset of 3,600 image-text pairs that are visually or textually similar but differ in terms of safety, which is designed to evaluate safety calibration across image-centric and text-centric scenarios. Based on our benchmark, we evaluate safety calibration across eleven widely used VLMs. Our extensive experiments revealed major issues with both undersafety and oversafety. We further investigated four approaches to improve the model’s safety calibration. We found that even though some methods effectively calibrated the models’ safety problems, these methods also lead to the degradation of models’ utility. This trade-off underscores the urgent need for advanced calibration methods, and our benchmark provides a valuable tool for evaluating future approaches.

</details>

---

## 215. Detecting and Mitigating Challenges in Zero-Shot Video Summarization with VideoLLMs

- [ ] Detecting and Mitigating Challenges in Zero-Shot Video Summarization with VideoLLMs | https://aclanthology.org/2025.findings-acl.16/

- **Link**: https://aclanthology.org/2025.findings-acl.16/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video summarization aims to generate a condensed textual version of an original video. Summaries may consist of either plain text or a shortlist of salient events, possibly including temporal or spatial references. Video Large Language Models (VLLMs) exhibit impressive zero-shot capabilities in video analysis. However, their performance varies significantly according to the LLM prompt, the characteristics of the video, and the properties of the training data and LLM architecture.In this work, we thoroughly evaluate the zero-shot summarization performance of four state-of-the-art open-source VLLMs specifically designed to address spatial and temporal reasoning. In light of the detected summarization issues, we propose different cost-effective mitigation strategies, based on Chain-of-Thought prompting, that involve the injection of knowledge extracted by external, lightweight models. To perform the VLLM evaluation, we design a new video summarization benchmark consisting of 100 videos with varying characteristics in terms of domain, duration, and spatio-temporal properties. Videos are manually annotated by three independent human experts with plain text, event-based, and spatio-temporal summaries. The experimental evaluation shows that VLLMs significantly benefit from prompting a list of recognized actions, whereas injecting automatically recognized objects and scene changes respectively improve spatially contextualized and event-based summaries in specific cases.

</details>

---

## 216. MANBench: Is Your Multimodal Model Smarter than Human?

- [ ] MANBench: Is Your Multimodal Model Smarter than Human? | https://aclanthology.org/2025.findings-acl.178/

- **Link**: https://aclanthology.org/2025.findings-acl.178/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of Multimodal Large Language Models (MLLMs) has ignited discussions regarding their potential to surpass human performance in multimodal tasks. In response, we introduce MANBench (Multimodal Ability Norms Benchmark), a bilingual benchmark (English and Chinese) comprising 1,314 questions across nine tasks, spanning knowledge-based and non-knowledge-based domains. MANBench emphasizes intuitive reasoning, seamless cross-modal integration, and real-world complexity, providing a rigorous evaluation framework.Through extensive human experiments involving diverse participants, we compared human performance against state-of-the-art MLLMs. The results indicate that while MLLMs excel in tasks like Knowledge and Text-Image Understanding, they struggle with deeper cross-modal reasoning tasks such as Transmorphic Understanding, Image Consistency, and Multi-image Understanding. Moreover, both humans and MLLMs face challenges in highly complex tasks like Puzzles and Spatial Imagination.MANBench highlights the strengths and limitations of MLLMs, revealing that even advanced models fall short of achieving human-level performance across many domains. We hope MANBench will inspire efforts to bridge the gap between MLLMs and human multimodal capabilities. The code and dataset are available at https://github.com/micdz/MANBench/.

</details>

---

## 217. mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus

- [ ] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus | https://aclanthology.org/2025.findings-acl.180/

- **Link**: https://aclanthology.org/2025.findings-acl.180/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (mLLMs) are trained on a large amount of text-image data. While most mLLMs are trained on caption-like data only, Alayrac et al. (2022) showed that additionally training them on interleaved sequences of text and images can lead to the emergence of in-context learning capabilities. However, the dataset they used, M3W, is not public and is only in English. There have been attempts to reproduce their results but the released datasets are English-only. In contrast, current multilingual and multimodal datasets are either composed of caption-like only or medium-scale or fully private data. This limits mLLM research for the 7,000 other languages spoken in the world. We therefore introduce mOSCAR, to the best of our knowledge the first large-scale multilingual and multimodal document corpus crawled from the web. It covers 163 languages, 303M documents, 200B tokens and 1.15B images. We carefully conduct a set of filtering and evaluation steps to make sure mOSCAR is sufficiently safe, diverse and of good quality. We additionally train two types of multilingual model to prove the benefits of mOSCAR: (1) a model trained on a subset of mOSCAR and captioning data and (2) a model trained on captioning data only. The model additionally trained on mOSCAR shows a strong boost in few-shot learning performance across various multilingual image-text tasks and benchmarks, confirming previous findings for English-only mLLMs. The dataset will be made publicly accessible under the Creative Commons CC BY 4.0 license.

</details>

---

## 218. ChartEdit: How Far AreMLLMs From Automating Chart Analysis? EvaluatingMLLMs’ Capability via Chart Editing

- [ ] ChartEdit: How Far AreMLLMs From Automating Chart Analysis? EvaluatingMLLMs’ Capability via Chart Editing | https://aclanthology.org/2025.findings-acl.185/

- **Link**: https://aclanthology.org/2025.findings-acl.185/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Although multimodal large language models (MLLMs) show promise in generating chart rendering code, editing charts via code presents a greater challenge. This task demands MLLMs to integrate chart understanding and reasoning capacities, which are labor-intensive. While many MLLMs claim such editing capabilities, current evaluations rely on limited case studies, highlighting the urgent need for a comprehensive evaluation framework.In this work, we propose ChartEdit, a new high-quality benchmark designed for chart editing tasks. This benchmark comprises 1,405 diverse editing instructions applied to 233 real-world charts, with each instruction-chart instance having been manually annotated and validated for accuracy. Utilizing ChartEdit, we evaluate the performance of 10 mainstream MLLMs across two types of experiments at both the code and chart levels.The results suggest that large-scale models can generate code to produce images that partially match the reference images.However, their ability to generate accurate edits according to the instructions remains limited. The state-of-the-art (SOTA) model achieves a score of only 59.96, highlighting significant challenges in precise modification. In contrast, small-scale models, including chart-domain models, struggle both with following editing instructions and generating overall chart images, underscoring the need for further development in this area. Code is available athttps://github.com/xxlllz/ChartEdit.

</details>

---

## 219. Unraveling and Mitigating Safety Alignment Degradation of Vision-Language Models

- [ ] Unraveling and Mitigating Safety Alignment Degradation of Vision-Language Models | https://aclanthology.org/2025.findings-acl.186/

- **Link**: https://aclanthology.org/2025.findings-acl.186/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The safety alignment ability of Vision-Language Models (VLMs) is prone to be degraded by the integration of the vision module compared to its LLM backbone. We investigate this phenomenon, dubbed as “safety alignment degradation” in this paper, and show that the challenge arises from the representation gap that emerges when introducing vision modality to VLMs. In particular, we show that the representations of multi-modal inputs shift away from that of text-only inputs which represent the distribution that the LLM backbone is optimized for. At the same time, the safety alignment capabilities, initially developed within the textual embedding space, do not successfully transfer to this new multi-modal representation space. To reduce safety alignment degradation, we introduce Cross-Modality Representation Manipulation (CMRM), an inference time representation intervention method for recovering the safety alignment ability that is inherent in the LLM backbone of VLMs, while simultaneously preserving the functional capabilities of VLMs. The empirical results show that our framework significantly recovers the alignment ability that is inherited from the LLM backbone with minimal impact on the fluency and linguistic capabilities of pre-trained VLMs even without additional training. Specifically, the unsafe rate of LLaVA-7B on multi-modal input can be reduced from 61.53% to as low as 3.15% with only inference-time intervention.

</details>

---

## 220. SignAlignLM: Integrating Multimodal Sign Language Processing into Large Language Models

- [ ] SignAlignLM: Integrating Multimodal Sign Language Processing into Large Language Models | https://aclanthology.org/2025.findings-acl.190/

- **Link**: https://aclanthology.org/2025.findings-acl.190/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Deaf and Hard-of-Hearing (DHH) users increasingly utilize Large Language Models (LLMs), yet face significant challenges due to these models’ limited understanding of sign language grammar, multimodal sign inputs, and Deaf cultural contexts. Further, current approaches that try to address these limitations, frequently reduce sign language processing (SLP) to traditional translation tasks, neglecting the multimodal and linguistic complexity inherent in signed languages. In this paper, we present an empirical investigation informed by learning theory into natively integrating sign language support within LLMs, directly addressing the documented needs of DHH users. We introduce the first text-based and multimodal LLMs capable of sign language processing called SignAlignLM, and propose new prompting and fine-tuning strategies incorporating sign linguistic rules and conventions. We show that LLMs can be generalized interfaces for both spoken and signed languages if trained with a multitasking paradigm. Our code and model checkpoints are open-source.

</details>

---

## 221. NegVQA: Can Vision Language Models Understand Negation?

- [ ] NegVQA: Can Vision Language Models Understand Negation? | https://aclanthology.org/2025.findings-acl.191/

- **Link**: https://aclanthology.org/2025.findings-acl.191/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Negation is a fundamental linguistic phenomenon that can entirely reverse the meaning of a sentence. As vision language models (VLMs) continue to advance and are deployed in high-stakes applications, assessing their ability to comprehend negation becomes essential. To address this, we introduce NegVQA, a visual question answering (VQA) benchmark consisting of 7,379 two-choice questions covering diverse negation scenarios and image-question distributions. We construct NegVQA by leveraging large language models to generate negated versions of questions from existing VQA datasets. Evaluating 20 state-of-the-art VLMs across seven model families, we find that these models struggle significantly with negation, exhibiting a substantial performance drop compared to their responses to the original questions. Furthermore, we uncover a U-shaped scaling trend, where increasing model size initially degrades performance on NegVQA before leading to improvements. Our benchmark reveals critical gaps in VLMs’ negation understanding and offers insights into future VLM development. Project page available at https://yuhui-zh15.github.io/NegVQA/.

</details>

---

## 222. Beyond Perception: Evaluating Abstract Visual Reasoning through Multi-Stage Task

- [ ] Beyond Perception: Evaluating Abstract Visual Reasoning through Multi-Stage Task | https://aclanthology.org/2025.findings-acl.2/

- **Link**: https://aclanthology.org/2025.findings-acl.2/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current Multimodal Large Language Models (MLLMs) excel in general visual reasoning but remain underexplored in Abstract Visual Reasoning (AVR), which demands higher-order reasoning to identify abstract rules beyond simple perception. Existing AVR benchmarks focus on single-step reasoning, emphasizing the end result but neglecting the multi-stage nature of reasoning process. Past studies found MLLMs struggle with these benchmarks, but it doesn’t explain how they fail. To address this gap, we introduce MultiStAR, a Multi-Stage AVR benchmark, based on RAVEN, designed to assess reasoning across varying levels of complexity. Additionally, existing metrics like accuracy only focus on the final outcomes while do not account for the correctness of intermediate steps. Therefore, we propose a novel metric, MSEval, which considers the correctness of intermediate steps in addition to the final outcomes. We conduct comprehensive experiments on MultiStAR using 17 representative close-source and open-source MLLMs. The results reveal that while existing MLLMs perform adequately on basic perception tasks, they continue to face challenges in more complex rule detection stages. The dataset and code will be available after acceptance.

</details>

---

## 223. Vision Language Model Helps Private Information De-Identification in Vision Data

- [ ] Vision Language Model Helps Private Information De-Identification in Vision Data | https://aclanthology.org/2025.findings-acl.236/

- **Link**: https://aclanthology.org/2025.findings-acl.236/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Language Models (VLMs) have gained significant popularity due to their remarkable ability. While various methods exist to enhance privacy in text-based applications, privacy risks associated with visual inputs remain largely overlooked such as Protected Health Information (PHI) in medical images. To tackle this problem, two key tasks: accurately localizing sensitive text and processing it to ensure privacy protection should be performed. To address this issue, we introduce VisShield (Vision Privacy Shield), an end-to-end framework designed to enhance the privacy awareness of VLMs. Our framework consists of two key components: a specialized instruction-tuning dataset OPTIC (Optical Privacy Text Instruction Collection) and a tailored training methodology. The dataset provides diverse privacy-oriented prompts that guide VLMs to perform targeted Optical Character Recognition (OCR) for precise localization of sensitive text, while the training strategy ensures effective adaptation of VLMs to privacy-preserving tasks. Specifically, our approach ensures that VLMs recognize privacy-sensitive text and output precise bounding boxes for detected entities, allowing for effective masking of sensitive information. Extensive experiments demonstrate that our framework significantly outperforms existing approaches in handling private information, paving the way for privacy-preserving applications in vision-language models.

</details>

---

## 224. Unveiling Privacy Risks in Multi-modal Large Language Models: Task-specific Vulnerabilities and Mitigation Challenges

- [ ] Unveiling Privacy Risks in Multi-modal Large Language Models: Task-specific Vulnerabilities and Mitigation Challenges | https://aclanthology.org/2025.findings-acl.237/

- **Link**: https://aclanthology.org/2025.findings-acl.237/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Privacy risks in text-only Large Language Models (LLMs) are well studied, particularly their tendency to memorize and leak sensitive information. However, Multi-modal Large Language Models (MLLMs), which process both text and images, introduce unique privacy challenges that remain underexplored. Compared to text-only models, MLLMs can extract and expose sensitive information embedded in images, posing new privacy risks. We reveal that some MLLMs are susceptible to privacy breaches, leaking sensitive data embedded in images or stored in memory. Specifically, in this paper, we (1) introduce MM-Privacy, a comprehensive dataset designed to assess privacy risks across various multi-modal tasks and scenarios, where we define Disclosure Risks and Retention Risks. (2) systematically evaluate different MLLMs using MM-Privacy and demonstrate how models leak sensitive data across various tasks, and (3) provide additional insights into the role of task inconsistency in privacy risks, emphasizing the urgent need for mitigation strategies. Our findings highlight privacy concerns in MLLMs, underscoring the necessity of safeguards to prevent data exposure. Part of our dataset and code can be found here.

</details>

---

## 225. MM-R3: On (In-)Consistency of Vision-Language Models (VLMs)

- [ ] MM-R3: On (In-)Consistency of Vision-Language Models (VLMs) | https://aclanthology.org/2025.findings-acl.246/

- **Link**: https://aclanthology.org/2025.findings-acl.246/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the advent of LLMs and variants, a flurry of research has emerged, analyzing the performance of such models across an array of tasks. While most studies focus on evaluating the capabilities of state-of-the-art (SoTA) Vision Language Models (VLMs) through task accuracy (e.g., visual question answering, grounding), our work explores the related but complementary aspect ofconsistency– the ability of a VLM to produce semantically similar or identical responses to semantically similar queries. We note that consistency is a fundamental prerequisite (necessary but not sufficient condition) for robustness and trust in VLMs. Armed with this perspective, we propose the MM-Rbenchmark, which allows us to analyze performance, in terms of consistency and accuracy, of SoTA VLMs on three tasks: Question Rephrasing, Image Restyling, and Context Reasoning. Our analysis reveals that consistency does not always align with accuracy, indicating that models with higher accuracy are not necessarily more consistent, and vice versa. Furthermore, we propose a simple yet effective mitigation strategy in the form of an adapter module trained to minimize inconsistency across prompts. With our proposed strategy, we are able to achieve absolute improvements of 5.7% and 12.5%, on average on widely used VLMs such as BLIP-2 and LLaVa 1.5M in terms of consistency over their existing counterparts.

</details>

---

## 226. Shadow-Activated Backdoor Attacks on Multimodal Large Language Models

- [ ] Shadow-Activated Backdoor Attacks on Multimodal Large Language Models | https://aclanthology.org/2025.findings-acl.248/

- **Link**: https://aclanthology.org/2025.findings-acl.248/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper delves into a novel backdoor attack scenario, aiming to uncover potential security risks associated with Multimodal Large Language Models (MLLMs) during multi-round open-ended conversations with users. In the practical use of MLLMs, users have full control over the interaction process with the model, such as using their own collected photos and posing arbitrary open-ended questions. Traditional backdoor attacks that rely on adding external triggers are less applicable. To this end, we introduce a new shadow-activated backdoor attacking paradigm in this paper, wherein attacks implicitly inject malicious content into the responses of MLLMs when the responses explicitly relate to the shadowed object, i.e., without any triggers. To facilitate the shadow-activated backdoor attack, we present a novel framework named BadMLLM to achieve the desired behaviors by constructing a poisoned dataset using GPT-4 Vision and implementing an attention-regularized tuning strategy to address the semantic discontinuity between the original response and the inserted promotion. Extensive experimental results conducted on five MLLMs, three objects, and two types of promotion slogans have demonstrated impressive performance in achieving both efficacy and utility goals, thereby highlighting the significant potential risks concealed within MLLMs.

</details>

---

## 227. Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding

- [ ] Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding | https://aclanthology.org/2025.findings-acl.249/

- **Link**: https://aclanthology.org/2025.findings-acl.249/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have achieved remarkable progress in multimodal tasks, yet they often struggle with visual arithmetic, seemingly simple capabilities like object counting or length comparison, which are essential for relevant complex tasks like chart understanding and geometric reasoning. In this work, we first investigate the root causes of this deficiency through a suite of probing tasks focusing on basic visual arithmetic. Our analysis reveals that while pre-trained vision encoders typically capture sufficient information, the text decoder often fails to decode it correctly for arithmetic reasoning. To address this, we propose CogAlign, a novel post-training strategy inspired by Piaget’s theory of cognitive development. CogAlign trains VLMs to recognize invariant properties under visual transformations. We demonstrate that this approach significantly improves the performance of three diverse VLMs on our proposed probing tasks. Furthermore, CogAlign enhances performance by an average of 4.6% on CHOCOLATE and 2.9% on MATH-VISION, outperforming or matching supervised fine-tuning methods while requiring only 60% less training data. These results highlight the effectiveness and generalizability of CogAlign in improving fundamental visual arithmetic capabilities and their transfer to downstream tasks.

</details>

---

## 228. AdaV: Adaptive Text-visual Redirection for Vision-Language Models

- [ ] AdaV: Adaptive Text-visual Redirection for Vision-Language Models | https://aclanthology.org/2025.findings-acl.258/

- **Link**: https://aclanthology.org/2025.findings-acl.258/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The success of Vision-Language Models (VLMs) often relies on high-resolution schemes that preserve image details, while these approaches also generate an excess of visual tokens, leading to a substantial decrease in model efficiency. A typical VLM includes a visual encoder, a text encoder, and an LLM. Recent studies suggest pruning visual tokens based on visual and textual priors to accelerate VLMs without additional training costs. However, these methods often overlook prompt semantics or suffer from biased self-attention in the LLM. Inspired by the efficient mechanisms of the human brain for multimodal understanding, we introduce AdaV, a novel training-free visual token pruning method. By emulating the neural pathways that preprocess visual and auditory information before the reasoning stage, we shift text-guided visual attention redirection to the pre-LLM stage, which reduces biased token pruning and enhances model robustness with a limited visual token budget. A Self-adaptive Cross-modality Attention Redirection (SCAR) module is further proposed that effectively merges and redirects visual attention with text-to-image attention. Extensive experiments on seven challenging benchmarks demonstrate that our AdaV achieves SOTA performance in training-free VLM acceleration and can be plug-and-play on various VLMs. We plan to open-source the code upon publication.

</details>

---

## 229. AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding

- [ ] AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding | https://aclanthology.org/2025.findings-acl.283/

- **Link**: https://aclanthology.org/2025.findings-acl.283/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have revolutionized video understanding, yet are still limited by context length when processing long videos. Recent methods compress videos by leveraging visual redundancy uniformly, yielding promising results. Nevertheless, our quantitative analysis shows that redundancy varies significantly across time and model layers, necessitating a more flexible compression strategy. We propose **AdaReTaKe**, a training-free method that flexibly reduces visual redundancy by allocating compression ratios among time and layers with theoretical guarantees. Integrated into state-of-the-art MLLMs, AdaReTaKe improves processing capacity from 256 to 2048 frames while preserving critical information. Experiments on VideoMME, MLVU, LongVideoBench, and LVBench datasets demonstrate that AdaReTaKe outperforms existing methods by 2.3% and 2.8% for 7B and 72B models, respectively, with even greater improvements of 5.9% and 6.0% on the longest LVBench.

</details>

---

## 230. Multimodal Causal Reasoning Benchmark: Challenging Multimodal Large Language Models to Discern Causal Links Across Modalities

- [ ] Multimodal Causal Reasoning Benchmark: Challenging Multimodal Large Language Models to Discern Causal Links Across Modalities | https://aclanthology.org/2025.findings-acl.288/

- **Link**: https://aclanthology.org/2025.findings-acl.288/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have showcased exceptional Chain-of-Thought (CoT) reasoning ability in complex textual inference tasks including causal reasoning. However, will these causalities remain straightforward when crucial hints hide in visual details? If not, what factors might influence cross-modal generalization? Whether we can effectively enhance their capacity for robust causal inference across both text and vision? Motivated by these, we introduce **MuCR** - a novel **Mu**ltimodal **C**ausal **R**easoning benchmark that leverages synthetic siamese images and text pairs to challenge MLLMs. Additionally, we develop tailored metrics from multiple perspectives, including image-level match, phrase-level understanding, and sentence-level explanation, to comprehensively assess MLLMs’ comprehension abilities. Our experiments reveal that current MLLMs fall short in multimodal causal reasoning compared to their performance in purely textual settings. Additionally, we find that identifying visual cues across images is key to effective cross-modal generalization. Finally, we propose the **VcCoT** strategy that better highlights visual cues, and our results confirm its efficacy in enhancing multimodal causal reasoning.

</details>

---

## 231. VCD: A Dataset for Visual Commonsense Discovery in Images

- [ ] VCD: A Dataset for Visual Commonsense Discovery in Images | https://aclanthology.org/2025.findings-acl.290/

- **Link**: https://aclanthology.org/2025.findings-acl.290/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual commonsense plays a vital role in understanding and reasoning about the visual world. While commonsense knowledge bases like ConceptNet provide structured collections of general facts, they lack visually grounded representations. Scene graph datasets like Visual Genome, though rich in object-level descriptions, primarily focus on directly observable information and lack systematic categorization of commonsense knowledge. We present Visual Commonsense Dataset (VCD), a large-scale dataset containing over 100,000 images and 14 million object-commonsense pairs that bridges this gap. VCD introduces a novel three-level taxonomy for visual commonsense, integrating both Seen (directly observable) and Unseen (inferrable) commonsense across Property, Action, and Space aspects. Each commonsense is represented as a triple where the head entity is grounded to object bounding boxes in images, enabling scene-dependent and object-specific visual commonsense representation. To demonstrate VCD’s utility, we develop VCM, a generative model that combines a vision-language model with instruction tuning to discover diverse visual commonsense from images. Extensive evaluations demonstrate both the high quality of VCD and its value as a resource for advancing visually grounded commonsense understanding and reasoning. Our dataset and code will be released on https://github.com/NUSTM/VCD.

</details>

---

## 232. ProMedTS: A Self-Supervised, Prompt-Guided Multimodal Approach for Integrating Medical Text and Time Series

- [ ] ProMedTS: A Self-Supervised, Prompt-Guided Multimodal Approach for Integrating Medical Text and Time Series | https://aclanthology.org/2025.findings-acl.308/

- **Link**: https://aclanthology.org/2025.findings-acl.308/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large language models (LLMs) have shown remarkable performance in vision-language tasks, but their application in the medical field remains underexplored, particularly for integrating structured time series data with unstructured clinical notes. In clinical practice, dynamic time series data, such as lab test results, capture critical temporal patterns, while clinical notes provide rich semantic context. Merging these modalities is challenging due to the inherent differences between continuous signals and discrete text. To bridge this gap, we introduce ProMedTS, a novel self-supervised multimodal framework that employs prompt-guided learning to unify these heterogeneous data types. Our approach leverages lightweight anomaly detection to generate anomaly captions that serve as prompts, guiding the encoding of raw time series data into informative prompt embeddings. These prompt embeddings are aligned with textual representations in a shared latent space, preserving fine-grained temporal nuances alongside semantic insights. Furthermore, our framework incorporates tailored self-supervised objectives to enhance both intra- and inter-modal alignment. We evaluate ProMedTS on disease diagnosis tasks using real-world datasets, and the results demonstrate that our method consistently outperforms state-of-the-art approaches.

</details>

---

## 233. Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models

- [ ] Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models | https://aclanthology.org/2025.findings-acl.322/

- **Link**: https://aclanthology.org/2025.findings-acl.322/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucination issues continue to affect multimodal large language models (MLLMs), with existing research mainly addressing object-level or attribute-level hallucinations, neglecting the more complex relation hallucinations that require advanced reasoning. Current benchmarks for relation hallucinations lack detailed evaluation and effective mitigation, and their datasets often suffer from biases due to systematic annotation processes. To address these challenges, we introduce Reefknot, a comprehensive benchmark targeting relation hallucinations, comprising over 20,000 real-world samples. We provide a systematic definition of relation hallucinations, integrating perceptive and cognitive perspectives, and construct a relation-based corpus using the Visual Genome scene graph dataset. Our comparative evaluation reveals significant limitations in current MLLMs’ ability to handle relation hallucinations. Additionally, we propose a novel confidence-based mitigation strategy, which reduces the hallucination rate by an average of 9.75% across three datasets, including Reefknot. Our work offers valuable insights for achieving trustworthy multimodal intelligence. The dataset and code are released at https://github.com/JackChen-seu/Reefknot.

</details>

---

## 234. Advancing General Multimodal Capability of Vision-language Models with Pyramid-descent Visual Position Encoding

- [ ] Advancing General Multimodal Capability of Vision-language Models with Pyramid-descent Visual Position Encoding | https://aclanthology.org/2025.findings-acl.327/

- **Link**: https://aclanthology.org/2025.findings-acl.327/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language Models (VLMs) have shown remarkable capabilities in advancing general artificial intelligence, yet the irrational encoding of visual positions persists in inhibiting the models’ comprehensive perception performance across different levels of granularity. In this work, we propose Pyramid-descent Visual Position Encoding (PyPE), a novel approach designed to enhance the perception of visual tokens within VLMs. By assigning visual position indexes from the periphery to the center and expanding the central receptive field incrementally, PyPE addresses the limitations of traditional raster-scan methods and mitigates the long-term decay effects induced by Rotary Position Embedding (RoPE). Our method reduces the relative distance between interrelated visual elements and instruction tokens, promoting a more rational allocation of attention weights and allowing for a multi-granularity perception of visual elements and countering the over-reliance on anchor tokens. Extensive experimental evaluations demonstrate that PyPE consistently improves the general capabilities of VLMs across various sizes. Code is available at https://anonymous.4open.science/r/PyPE-34EE.

</details>

---

## 235. EssayJudge: A Multi-Granular Benchmark for Assessing Automated Essay Scoring Capabilities of Multimodal Large Language Models

- [ ] EssayJudge: A Multi-Granular Benchmark for Assessing Automated Essay Scoring Capabilities of Multimodal Large Language Models | https://aclanthology.org/2025.findings-acl.329/

- **Link**: https://aclanthology.org/2025.findings-acl.329/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automated Essay Scoring (AES) plays a crucial role in educational assessment by providing scalable and consistent evaluations of writing tasks. However, traditional AES systems face three major challenges: (i) reliance on handcrafted features that limit generalizability, (ii) difficulty in capturing fine-grained traits like coherence and argumentation, and (iii) inability to handle multimodal contexts. In the era of Multimodal Large Language Models (MLLMs), we propose **EssayJudge**, the **first multimodal benchmark to evaluate AES capabilities across lexical-, sentence-, and discourse-level traits**. By leveraging MLLMs’ strengths in trait-specific scoring and multimodal context understanding, EssayJudge aims to offer precise, context-rich evaluations without manual feature engineering, addressing longstanding AES limitations. Our experiments with 18 representative MLLMs reveal gaps in AES performance compared to human evaluation, particularly in discourse-level traits, highlighting the need for further advancements in MLLM-based AES research. Our dataset and code will be available upon acceptance.

</details>

---

## 236. Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks

- [ ] Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks | https://aclanthology.org/2025.findings-acl.331/

- **Link**: https://aclanthology.org/2025.findings-acl.331/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While Vision-Language Models (VLMs) have shown remarkable abilities, they invariably generate flawed responses. Self-correction that instructs models to refine their outputs presents a promising solution to this issue. Previous studies have mainly concentrated on Large Language Models (LLMs), while the self-correction abilities of VLMs, particularly concerning both visual and linguistic information, remain largely unexamined. This study investigates the self-correction capabilities of VLMs during both inference and fine-tuning stages. We introduce a Self-Correction Learning (SCL) approach that enables VLMs to learn from their self-generated self-correction data through Direct Preference Optimization (DPO) without relying on external feedback, facilitating self-improvement. Experimental results demonstrate that although VLMs struggle to self-correct effectively during iterative inference without additional fine-tuning and external feedback, they can enhance their performance and avoid previous mistakes through preference fine-tuning when their generated self-correction data are categorized into preferred and disfavored samples. This study emphasizes that self-correction is not merely a refinement process; rather, it should enhance models’ reasoning ability through additional training, enabling them to generate high-quality responses directly without further refinement.

</details>

---

## 237. InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model

- [ ] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model | https://aclanthology.org/2025.findings-acl.340/

- **Link**: https://aclanthology.org/2025.findings-acl.340/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite the promising performance of Large Vision Language Models (LVLMs) in visual understanding, they occasionally generate incorrect outputs. While reward models (RMs) with reinforcement learning or test-time scaling offer the potential for improving generation quality, a critical gap remains: publicly available multi-modal RMs for LVLMs are scarce, and the implementation details of proprietary models are often unclear. We bridge this gap with InternLM-XComposer2.5-Reward (IXC-2.5-Reward), a simple yet effective multi-modal reward model that aligns LVLMs with human preferences. To ensure the robustness and versatility of IXC-2.5-Reward, we set up a high-quality multi-modal preference corpus spanning text, image, and video inputs across diverse domains, such as instruction following, general understanding, text-rich documents, mathematical reasoning, and video understanding. IXC-2.5-Reward achieves excellent results on the latest multi-modal reward model benchmark and shows competitive performance on text-only reward model benchmarks. We further demonstrate three key applications of IXC-2.5-Reward: (1) Providing a supervisory signal for RL training. We integrate IXC-2.5-Reward with Proximal Policy Optimization (PPO) yields IXC-2.5-Chat, which shows consistent improvements in instruction following and multi-modal open-ended dialogue; (2) Selecting the best response from candidate responses for test-time scaling; and (3) Filtering outlier or noisy samples from existing image and video instruction tuning training data.

</details>

---

## 238. RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models

- [ ] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models | https://aclanthology.org/2025.findings-acl.341/

- **Link**: https://aclanthology.org/2025.findings-acl.341/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Object Navigation (ObjectNav) is a fundamental task in embodied artificial intelligence. Although significant progress has been made in semantic map construction and target direction prediction in current research, redundant exploration and exploration failures remain inevitable. A critical but underexplored direction is the timely termination of exploration to overcome these challenges. We observe a diminishing marginal effect between exploration steps and exploration rates and analyze the cost-benefit relationship of exploration. Inspired by this, we propose RATE-Nav, a Region-Aware Termination-Enhanced method. It includes a geometric predictive region segmentation algorithm and region-Based exploration estimation algorithm for exploration rate calculation. By leveraging the visual question answering capabilities of visual language models (VLMs) and exploration rates enables efficient termination.RATE-Nav achieves a success rate of 67.8% and an SPL of 31.3% on the HM3D dataset. And on the more challenging MP3D dataset, RATE-Nav shows approximately 10% improvement over previous zero-shot methods.

</details>

---

## 239. OS-Kairos: Adaptive Interaction forMLLM-PoweredGUIAgents

- [ ] OS-Kairos: Adaptive Interaction forMLLM-PoweredGUIAgents | https://aclanthology.org/2025.findings-acl.348/

- **Link**: https://aclanthology.org/2025.findings-acl.348/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Autonomous graphical user interface (GUI) agents powered by multimodal large language models have shown great promise. However, a critical yet underexplored issue persists:over-execution, where the agent executes tasks in a fully autonomous way, without adequate assessment of its action confidence to compromise an adaptive human-agent collaboration. This poses substantial risks in complex scenarios, such as those involving ambiguous user instructions, unexpected interruptions, and environmental hijacks. To address the issue, we introduceOS-Kairos, an adaptive GUI agent capable of predicting confidence levels at each interaction step and efficiently deciding whether to act autonomously or seek human intervention.OS-Kairosis developed through two key mechanisms: (i) collaborative probing that annotates confidence scores at each interaction step; (ii) confidence-driven interaction that leverages these confidence scores to elicit the ability of adaptive interaction. Experimental results show thatOS-Kairossubstantially outperforms existing models on our curated dataset featuring complex scenarios, as well as on established benchmarks such as AITZ and Meta-GUI, with 24.59%~87.29% improvements in task success rate.OS-Kairosfacilitates an adaptive human-agent collaboration, prioritizing effectiveness, generality, scalability, and efficiency for real-world GUI interaction. The dataset and codes are available at Anonymous.

</details>

---

## 240. Can We TrustAIDoctors? A Survey of Medical Hallucination in Large Language and Large Vision-Language Models

- [ ] Can We TrustAIDoctors? A Survey of Medical Hallucination in Large Language and Large Vision-Language Models | https://aclanthology.org/2025.findings-acl.350/

- **Link**: https://aclanthology.org/2025.findings-acl.350/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucination has emerged as a critical challenge for large language models (LLMs) and large vision-language models (LVLMs), particularly in high-stakes medical applications. Despite its significance, dedicated research on medical hallucination remains unexplored. In this survey, we first provide a unified perspective on medical hallucination for both LLMs and LVLMs, and delve into its causes. Subsequently, we review recent advancements in detecting, evaluating, and mitigating medical hallucinations, offering a comprehensive overview of evaluation benchmarks, metrics, and strategies developed to tackle this issue. Moreover, we delineate the current challenges and delve into new frontiers, thereby shedding light on future research. We hope this work coupled with open-source resources can provide the community with quick access and spur breakthrough research in medical hallucination.

</details>

---

## 241. Vision-aided Unsupervised Constituency Parsing with Multi-MLLMDebating

- [ ] Vision-aided Unsupervised Constituency Parsing with Multi-MLLMDebating | https://aclanthology.org/2025.findings-acl.353/

- **Link**: https://aclanthology.org/2025.findings-acl.353/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a novel framework for vision-aided unsupervised constituency parsing (VUCP), leveraging multimodal large language models (MLLMs) pre-trained on diverse image-text or video-text data. Unlike previous methods requiring explicit cross-modal alignment, our approach eliminates this need by using pre-trained models like Qwen-VL and VideoLLaVA, which seamlessly handle multimodal inputs. We introduce two multi-agent debating mechanisms—consensus-driven (CD) and round-driven (RD)—to enable cooperation between models with complementary strengths. Extensive experiments demonstrate that our approach achieves state-of-the-art performance on both image-text and video-text datasets for VUCP, improving robustness and accuracy.

</details>

---

## 242. TAMP: Token-Adaptive Layerwise Pruning in Multimodal Large Language Models

- [ ] TAMP: Token-Adaptive Layerwise Pruning in Multimodal Large Language Models | https://aclanthology.org/2025.findings-acl.359/

- **Link**: https://aclanthology.org/2025.findings-acl.359/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have shown remarkable versatility in understanding diverse multimodal data and tasks. However, these capabilities come with an increased model scale. While post-training pruning reduces model size in unimodal models, its application to MLLMs often yields limited success. Our analysis discovers that conventional methods fail to account for the unique token attributes across layers and modalities inherent to MLLMs. Inspired by this observation, we propose TAMP, a simple yet effective pruning framework tailored for MLLMs, featuring two key components: (1) Diversity-Aware Sparsity, which adjusts sparsity ratio per layer based on diversities among multimodal output tokens, preserving more parameters in high-diversity layers; and (2) Adaptive Multimodal Input Activation, which identifies representative multimodal input tokens using attention scores to guide unstructured weight pruning. We validate our method on two state-of-the-art MLLMs: LLaVA-NeXT, designed for vision-language tasks, and VideoLLaMA2, capable of processing audio, visual, and language modalities. Empirical experiments across various multimodal evaluation benchmarks demonstrate that each component of our approach substantially outperforms existing pruning techniques. Our code is available at https://github.com/G-JWLee/TAMP

</details>

---

## 243. RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for EvaluatingLLM-Based Table Analysis

- [ ] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for EvaluatingLLM-Based Table Analysis | https://aclanthology.org/2025.findings-acl.371/

- **Link**: https://aclanthology.org/2025.findings-acl.371/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce **RealHiTBench**, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using **25** state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based agent that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs’ perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at https://github.com/cspzyy/RealHiTBench.

</details>

---

## 244. MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models

- [ ] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models | https://aclanthology.org/2025.findings-acl.375/

- **Link**: https://aclanthology.org/2025.findings-acl.375/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in Machine Unlearning (MU) has introduced solutions for the selective removal of private or sensitive information encoded within deep neural networks. Nonetheless, MU for Multimodal Large Language Models (MLLMs) remains in its nascent phase. Therefore, we propose to **reformulate the task of multimodal MU in the era of MLLMs**, which aims to erase only the visual patterns associated with a given entity while preserving the corresponding textual knowledge encoded within the original parameters of the language model backbone. Furthermore, we **develop a novel geometry-constrained gradient ascent method MMUnlearner**. It updates the weights of MLLMs with a weight saliency map jointly restricted by the remaining concepts and textual knowledge during unlearning, thereby preserving parameters essential for non-target knowledge. Extensive experiments demonstrate that MMUnlearner surpasses baselines that finetuning MLLMs with VQA data directly through Gradient Ascent (GA) or Negative Preference Optimization (NPO), across all evaluation dimensions. Our code will be released upon acceptance.

</details>

---

## 245. Generating Questions, Answers, and Distractors for Videos: Exploring Semantic Uncertainty of Object Motions

- [ ] Generating Questions, Answers, and Distractors for Videos: Exploring Semantic Uncertainty of Object Motions | https://aclanthology.org/2025.findings-acl.376/

- **Link**: https://aclanthology.org/2025.findings-acl.376/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Video Question-Answer-Distractors (QADs) show promising values for assessing the performance of systems in perceiving and comprehending multimedia content. Given the significant cost and labor demands of manual annotation, existing large-scale Video QADs benchmarks are typically generated automatically using video captions. Since video captions are incomplete representations of visual content and susceptible to error propagation, direct generation of QADs from video is crucial. This work first leverages a large vision-language model for video QADs generation. To enhance the consistency and diversity of the generated QADs, we propose utilizing temporal motion to describe the video objects. In addition, We design a selection mechanism that chooses diverse temporal object motions to generate diverse QADs focusing on different objects and interactions, maximizing overall semantic uncertainty for a given video. Evaluation on the NExT-QA and Perception Test benchmarks demonstrates that the proposed approach significantly improves both the consistency and diversity of QADs generated by a range of large vision-language models, thus highlighting its effectiveness and generalizability.

</details>

---

## 246. Self-Rewarding Large Vision-Language Models for Optimizing Prompts in Text-to-Image Generation

- [ ] Self-Rewarding Large Vision-Language Models for Optimizing Prompts in Text-to-Image Generation | https://aclanthology.org/2025.findings-acl.383/

- **Link**: https://aclanthology.org/2025.findings-acl.383/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-to-image models are powerful for producing high-quality images based on given text prompts, but crafting these prompts often requires specialized vocabulary. To address this, existing methods train rewriting models with supervision from large amounts of manually annotated data and trained aesthetic assessment models. To alleviate the dependence on data scale for model training and the biases introduced by trained models, we propose a novel prompt optimization framework, designed to rephrase a simple user prompt into a sophisticated prompt to a text-to-image model. Specifically, we employ the large vision language models (LVLMs) as the solver to rewrite the user prompt, and concurrently, employ LVLMs as a reward model to score the aesthetics and alignment of the images generated by the optimized prompt. Instead of laborious human feedback, we exploit the prior knowledge of the LVLM to provide rewards, i.e., AI feedback. Simultaneously, the solver and the reward model are unified into one model and iterated in reinforcement learning to achieve self-improvement by giving a solution and judging itself. Results on two popular datasets demonstrate that our method outperforms other strong competitors.

</details>

---

## 247. Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models

- [ ] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models | https://aclanthology.org/2025.findings-acl.394/

- **Link**: https://aclanthology.org/2025.findings-acl.394/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the increasing size of Large Vision-Language Models (LVLMs), network pruning techniques aimed at compressing models for deployment in resource-constrained environments have garnered significant attention. However, we observe that pruning often leads to a degradation in safety performance. To address this issue, we present a novel and lightweight approach, termed Hierarchical Safety Realignment (HSR). HSR operates by first quantifying the contribution of each attention head to safety, identifying the most critical ones, and then selectively restoring neurons directly within these attention heads that play a pivotal role in maintaining safety. This process hierarchically realigns the safety of pruned LVLMs, progressing from the attention head level to the neuron level. We validate HSR across various models and pruning strategies, consistently achieving notable improvements in safety performance. To our knowledge, this is the first work explicitly focused on restoring safety in LVLMs post-pruning.

</details>

---

## 248. MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering

- [ ] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering | https://aclanthology.org/2025.findings-acl.404/

- **Link**: https://aclanthology.org/2025.findings-acl.404/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Text-Centric Visual Question Answering (TEC-VQA) in its proper format not only facilitates human-machine interaction in text-centric visual environments but also serves as a de facto gold proxy to evaluate AI models in the domain of text-centric scene understanding. Nonetheless, most existing TEC-VQA benchmarks focus on high-resource languages like English and Chinese. Despite pioneering works expanding multilingual QA pairs in non-text-centric VQA datasets through translation engines, the translation-based protocol encounters a substantial “visual-textual misalignment” problem when applied to TEC-VQA. Specifically, it prioritizes the text in question-answer pairs while disregarding the visual text present in images. Moreover, it fails to address complexities related to nuanced meaning, contextual distortion, language bias, and question-type diversity. In this work, we tackle multilingual TEC-VQA by introducing MTVQA, the first benchmark featuring high-quality human expert annotations across 9 diverse languages, consisting of 6,778 question-answer pairs across 2,116 images. Further, by comprehensively evaluating numerous state-of-the-art Multimodal Large Language Models (MLLMs), including Qwen2.5-VL, InternVL-2.5, GPT-4o, GPT-4V, Claude3, and Gemini, on the MTVQA benchmark, it is evident that there is still a large room for performance improvement (InternVL-2.5 scoring 32.2 versus 79.7 for human performance), underscoring the value of MTVQA. By providing a dataset with nuanced multilingual annotations, MTVQA aims to set a new standard for benchmarks, fostering advancements in multilingual visual text comprehension.

</details>

---

## 249. Flow2Code: Evaluating Large Language Models for Flowchart-based Code Generation Capability

- [ ] Flow2Code: Evaluating Large Language Models for Flowchart-based Code Generation Capability | https://aclanthology.org/2025.findings-acl.425/

- **Link**: https://aclanthology.org/2025.findings-acl.425/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

While large language models (LLMs) show promise in code generation, existing benchmarks neglect the flowchart-based code generation. To promote further research on flowchart-based code generation, this work presents Flow2Code, a novel benchmark for flowchart-based code generation evaluation. The evaluation dataset spans 15 programming languages and includes 5,622 code segments paired with 16,866 flowcharts of three types: code, UML, and pseudocode. Extensive experiments with 13 multimodal LLMs reveal that current LLMs can not generate code based on flowcharts perfectly. Besides, experiment results show that the supervised fine-tuning technique contributes greatly to the models’ performance. The dataset will be publicly available.

</details>

---

## 250. Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models

- [ ] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models | https://aclanthology.org/2025.findings-acl.430/

- **Link**: https://aclanthology.org/2025.findings-acl.430/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite significant advancements in Large Vision-Language Models, Object Hallucination (OH) remains a persistent challenge. Building upon prior studies on contrastive decoding that address this issue without requiring additional model training, we introduce RVCD (Retrieval Visual Contrastive Decoding), an advanced method to suppress OH. RVCD leverages both negative and positive images at the logit level, explicitly referencing AI-generated images designed to represent a single concept. Our approach demonstrates substantial improvements over existing decoding-based methods.

</details>

---

## 251. mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data

- [ ] mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data | https://aclanthology.org/2025.findings-acl.433/

- **Link**: https://aclanthology.org/2025.findings-acl.433/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal embedding models have gained significant attention for their ability to map data from different modalities, such as text and images, into a unified representation space. However, the limited labeled multimodal data often hinders embedding performance. Recent approaches have leveraged data synthesis to address this problem, yet the quality of synthetic data remains a critical bottleneck. In this work, we identify three criteria for high-quality synthetic multimodal data. First, broad scope ensures that the generated data covers diverse tasks and modalities, making it applicable to various downstream scenarios. Second, robust cross-modal alignment makes different modalities semantically consistent. Third, high fidelity ensures that the synthetic data maintains realistic details to enhance its reliability. Guided by these principles, we synthesize datasets that: (1) cover a wide range of tasks, modality combinations, and languages, (2) are generated via a deep thinking process within a single pass of a multimodal large language model, and (3) incorporate real-world images with accurate and relevant texts, ensuring fidelity through self-evaluation and refinement. Leveraging these high-quality synthetic and labeled datasets, we train a multimodal multilingual E5 model mmE5. Extensive experiments demonstrate that mmE5 achieves state-of-the-art performance on the MMEB Benchmark and superior multilingual performance on the XTD benchmark. Our codes, datasets, and models are released in https://github.com/haon-chen/mmE5.

</details>

---

## 252. Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models

- [ ] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models | https://aclanthology.org/2025.findings-acl.448/

- **Link**: https://aclanthology.org/2025.findings-acl.448/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have exhibited impressive capabilities across various visual tasks, yet they remain hindered by the persistent challenge of hallucinations. To address this critical issue, we propose Mixture of Decoding (MoD), a novel approach for hallucination mitigation that dynamically adapts decoding strategies by evaluating the correctness of the model’s attention on image tokens. Specifically, MoD measures the consistency between outputs generated from the original image tokens and those derived from the model’s attended image tokens, to distinguish the correctness aforementioned. If the outputs are consistent, indicating correct attention, MoD employs a complementary strategy to amplify critical information. Conversely, if the outputs are inconsistent, suggesting erroneous attention, MoD utilizes a contrastive strategy to suppress misleading information. Extensive experiments demonstrate that MoD significantly outperforms existing decoding methods across multiple mainstream benchmarks, effectively mitigating hallucinations in LVLMs. Code is available at https://github.com/xlchen0205/MoD.

</details>

---

## 253. From Specific-MLLMs to Omni-MLLMs: A Survey onMLLMs Aligned with Multi-modalities

- [ ] From Specific-MLLMs to Omni-MLLMs: A Survey onMLLMs Aligned with Multi-modalities | https://aclanthology.org/2025.findings-acl.453/

- **Link**: https://aclanthology.org/2025.findings-acl.453/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

To tackle complex tasks in real-world scenarios, more researchers are focusing on Omni-MLLMs, which aim to achieve omni-modal understanding and generation. Beyond the constraints of any specific non-linguistic modality, Omni-MLLMs map various non-linguistic modalities into the embedding space of LLMs and enable the interaction and understanding of arbitrary combinations of modalities within a single model. In this paper, we systematically investigate relevant research and provide a comprehensive survey of Omni-MLLMs. Specifically, we first explain the four core components of Omni-MLLMs for unified multi-modal modeling with a meticulous taxonomy that offers novel perspectives. Then, we introduce the effective integration achieved through two-stage training and discuss the corresponding datasets as well as evaluation. Furthermore, we summarize the main challenges of current Omni-MLLMs and outline future directions. We hope this paper serves as an introduction for beginners and promotes the advancement of related research. Resources have been made publicly availableat https://github.com/threegold116/Awesome-Omni-MLLMs.

</details>

---

## 254. Align2LLaVA: Cascaded Human and Large Language Model Preference Alignment for Multi-modal Instruction Curation

- [ ] Align2LLaVA: Cascaded Human and Large Language Model Preference Alignment for Multi-modal Instruction Curation | https://aclanthology.org/2025.findings-acl.458/

- **Link**: https://aclanthology.org/2025.findings-acl.458/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in Multi-modal Large Language Models (MLLMs), such as LLaVA-series models, are driven by massive machine-generated instruction-following data tuning. Such automatic instruction collection pipelines, however, inadvertently introduce significant variability in data quality. This paper introduces a novel instruction curation algorithm, derived from two unique perspectives, human and LLM preference alignment, to compress this vast corpus of machine-generated multimodal instructions to a compact and high-quality form: (i) For human preference alignment, we have collected a machine-generated multimodal instruction dataset and established a comprehensive set of both subjective and objective criteria to guide the data quality assessment critically from human experts. By doing so, a reward model was trained on the annotated dataset to internalize the nuanced human understanding of instruction alignment. (ii) For LLM preference alignment, given the instruction selected by the reward model, we propose leveraging the inner LLM used in MLLM to align the writing style of visual instructions with that of the inner LLM itself, resulting in LLM-aligned instruction improvement. Extensive experiments demonstrate that we can maintain or even improve model performance by compressing synthetic multimodal instructions by up to 90%. Impressively, by aggressively reducing the training instructions from 158k to 14k (9× smaller), our model consistently outperforms its full-size dataset counterpart across various MLLM benchmarks. Our project is available at https://github.com/DCDmllm/Align2LLaVA.

</details>

---

## 255. LIME: Less Is More forMLLMEvaluation

- [ ] LIME: Less Is More forMLLMEvaluation | https://aclanthology.org/2025.findings-acl.474/

- **Link**: https://aclanthology.org/2025.findings-acl.474/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are measured on numerous benchmarks like image captioning, visual question answer, and reasoning. However, these benchmarks often include overly simple or uninformative samples, making it difficult to effectively distinguish the performance of different MLLMs. Additionally, evaluating models across many benchmarks creates a significant computational burden. To address these issues, we propose LIME (Less Is More for MLLM Evaluation), a refined and efficient benchmark curated using a semi-automated pipeline. This pipeline filters out uninformative samples and eliminates answer leakage by focusing on tasks that require image-based understanding. Our experiments show that LIME reduces the number of samples by 76% and evaluation time by 77%, while it can more effectively distinguish different models’ abilities. Notably, we find that traditional automatic metrics like CIDEr are insufficient for evaluating MLLMs’ captioning performance, and excluding the caption task score yields a more accurate reflection of overall model performance. All code and data are available at https://anonymous.4open.science/r/LIME-49CD

</details>

---

## 256. MHALO: EvaluatingMLLMs as Fine-grained Hallucination Detectors

- [ ] MHALO: EvaluatingMLLMs as Fine-grained Hallucination Detectors | https://aclanthology.org/2025.findings-acl.478/

- **Link**: https://aclanthology.org/2025.findings-acl.478/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Hallucination remains a critical challenge for multimodal large language models (MLLMs), undermining their reliability in real-world applications. While fine-grained hallucination detection (FHD) holds promise for enhancing high-quality vision-language data construction and model alignment through enriched feedback signals, automated solutions for this task have yet to be systematically explored. Inspired by the concept of “MLLM as a Judge”, we introduce MHALO, the first comprehensive benchmark specifically designed for evaluating MLLMs’ capability in performing token-level FHD. Our benchmark encompasses 12 distinct hallucination types spanning both multimodal perception and reasoning domains. Through extensive evaluations of 9 selected MLLMs, we reveal substantial performance limitations, with the leading model achieving an averageF1IoUof only 40.59%. To address this limitation, we develop HaloDet-4B, a specialized model trained on our curated training data, which significantly outperforms existing models. We hope the benchmark can provide valuable insights for future research on hallucination mitigation in MLLMs. The code and dataset will be publicly available.

</details>

---

## 257. CMIE: CombiningMLLMInsights with External Evidence for Explainable Out-of-Context Misinformation Detection

- [ ] CMIE: CombiningMLLMInsights with External Evidence for Explainable Out-of-Context Misinformation Detection | https://aclanthology.org/2025.findings-acl.487/

- **Link**: https://aclanthology.org/2025.findings-acl.487/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have demonstrated impressive capabilities in visual reasoning and text generation. While previous studies have explored the application of MLLM for detecting out-of-context (OOC) misinformation, our empirical analysis reveals two persisting challenges of this paradigm. Evaluating the representative GPT-4o model on direct reasoning and evidence augmented reasoning, results indicate that MLLM struggle to capture the deeper relationships—specifically, cases in which the image and text are not directly connected but are associated through underlying semantic links. Moreover, noise in the evidence further impairs detection accuracy.To address these challenges, we propose CMIE, a novel OOC misinformation detection framework that incorporates a Coexistence Relationship Generation (CRG) strategy and an Association Scoring (AS) mechanism. CMIE identifies the underlying coexistence relationships between images and text, and selectively utilizes relevant evidence to enhance misinformation detection. Experimental results demonstrate that our approach outperforms existing methods.

</details>

---

## 258. GIMMICK: Globally Inclusive Multimodal Multitask Cultural Knowledge Benchmarking

- [ ] GIMMICK: Globally Inclusive Multimodal Multitask Cultural Knowledge Benchmarking | https://aclanthology.org/2025.findings-acl.500/

- **Link**: https://aclanthology.org/2025.findings-acl.500/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) have recently gained attention due to their distinctive performance and broad applicability. While it has been previously shown that their efficacy in usage scenarios involving non-Western contexts falls short, existing studies are limited in scope, covering just a narrow range of cultures, focusing exclusively on a small number of cultural aspects, or evaluating a limited selection of models on a single task only. Towards globally inclusive LVLM research, we introduce GIMMICK, an extensive multimodal benchmark designed to assess a broad spectrum of cultural knowledge across 144 countries representing six global macro-regions. GIMMICK comprises six tasks built upon three new datasets that span 728 unique cultural events or facets on which we evaluated 20 LVLMs and 11 LLMs, including five proprietary and 26 open-weight models of all sizes. We systematically examine (1) regional cultural biases, (2) the influence of model size, (3) input modalities, and (4) external cues. Our analyses reveal strong biases toward Western cultures across models and tasks and highlight strong correlations between model size and performance, as well as the effectiveness of multimodal input and external geographic cues. We further find that models have more knowledge of tangible than intangible aspects (e.g., food vs. rituals) and that they excel in recognizing broad cultural origins but struggle with a more nuanced understanding.

</details>

---

## 259. R-VLM: Region-Aware Vision Language Model for PreciseGUIGrounding

- [ ] R-VLM: Region-Aware Vision Language Model for PreciseGUIGrounding | https://aclanthology.org/2025.findings-acl.501/

- **Link**: https://aclanthology.org/2025.findings-acl.501/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual agent models for automating human activities on Graphical User Interfaces (GUIs) have emerged as a promising research direction, driven by advances in large Vision Language Models (VLMs). A critical challenge in GUI automation is the precise grounding of interface elements across diverse platforms. Existing vision-only GUI agents directly ground elements from large and cluttered screenshots, requiring them to process substantial irrelevant information that compromises their accuracy. In addition, these approaches typically employ basic cross-entropy loss for learning grounding objectives, which fails to effectively capture grounding quality compared to established object detection metrics like Intersection-over-Union (IoU). To address these issues, we introduce R-VLM, a novel GUI grounding approach that leverages zoomed-in region proposals for precise element localization. We also propose an IoU-aware objective function that facilitates model convergence toward high IoU predictions. Our approach bridges the gap between VLMs and conventional object detection techniques, improving the state-of-the-art grounding accuracy by 13% across diverse GUI platforms on the GUI grounding benchmarks ScreenSpot and AgentStudio. In addition, our R-VLM approach shows 3.2-9.7% absolute accuracy improvements in GUI navigation tasks on the AITW and Mind2Web benchmarks.

</details>

---

## 260. MMXU: A Multi-Modal and Multi-X-ray Understanding Dataset for Disease Progression

- [ ] MMXU: A Multi-Modal and Multi-X-ray Understanding Dataset for Disease Progression | https://aclanthology.org/2025.findings-acl.508/

- **Link**: https://aclanthology.org/2025.findings-acl.508/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large vision-language models (LVLMs) have shown great promise in medical applications, particularly in visual question answering (MedVQA) and diagnosis from medical images. However, existing datasets and models often fail to consider critical aspects of medical diagnostics, such as the integration of historical records and the analysis of disease progression over time. In this paper, we introduce MMXU (Multimodal and MultiX-ray Understanding), a novel dataset for MedVQA that focuses on identifying changes in specific regions between two patient visits. Unlike previous datasets that primarily address single-image questions, MMXU enables multi-image questions, incorporating both current and historical patient data. We demonstrate the limitations of current LVLMs in identifying disease progression on MMXU-test, even those that perform well on traditional benchmarks. To address this, we propose a MedRecord-Augmented Generation (MAG) approach, incorporating both global and regional historical records.Our experiments show that integrating historical records significantly enhances diagnostic accuracy by at least 20%, bridging the gap between current LVLMs and human expert performance. Additionally, we fine-tune models with MAG on MMXU-dev, which demonstrates notable improvements. We hope this work could illuminate the avenue of advancing the use of LVLMs in medical diagnostics by emphasizing the importance of historical context in interpreting medical images.Our dataset is released at github.

</details>

---

## 261. Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models

- [ ] Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models | https://aclanthology.org/2025.findings-acl.512/

- **Link**: https://aclanthology.org/2025.findings-acl.512/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The recent advancement of Multimodal Large Language Models (MLLMs) has significantly improved their fine-grained perception of single images and general comprehension across multiple images. However, existing MLLMs still face challenges in achieving precise grounding in complex multi-image scenarios. To address this, we first explore a Chain-of-Thought (CoT) framework that integrates single-image grounding with multi-image comprehension. While partially effective, it remains unstable and struggles to capture abstract visual information due to its non-end-to-end nature. Therefore, we introduce Migician, the first multi-image grounding model capable of performing free-form and accurate grounding across multiple images. To support this, we present the MGrounding-630k dataset, which comprises data for several multi-image grounding tasks derived from existing datasets, along with newly generated free-form grounding instruction-following data. Furthermore, we propose MIG-Bench, a comprehensive benchmark specifically designed for evaluating multi-image grounding capabilities. Experimental results demonstrate that our model achieves significantly superior multi-image grounding capabilities, outperforming the best existing MLLMs by 24.94% and even surpassing much larger 70B models. Our code, model, dataset, and benchmark are fully open-sourced at https://migician-vg.github.io/.

</details>

---

## 262. Self-play through Computational Runtimes improves Chart Reasoning

- [ ] Self-play through Computational Runtimes improves Chart Reasoning | https://aclanthology.org/2025.findings-acl.559/

- **Link**: https://aclanthology.org/2025.findings-acl.559/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) achieve impressive zero-shot performance on multimodal reasoning tasks. Typically, best reported performance is achieved with a zero- or a few-shot prompt. We observe that asking the model to take other routes of solving the same task, such as through code generation, hurts performance. Furthermore, training sets are typically no longer useful for improving model performance through few-shot learning, due to their use in training. Indeed, we observe that auto-prompting techniques such as DSPy (CITATION), when applied on training sets, do not produce few-shot examples that further improve validation performance. Further, when used in conjunction with program-of-thought, performance becomes even worse.Our work overcomes these limitations by introducing a novel self-play programming interface which leverages the ability of VLMs to first generate code to decompose a complex visual reasoning task in sub-tasks, then use itself, or other models, as a tool to solve decomposed tasks. Our approach enables DSPy to not suffer from performance drops, when applied iteratively on training sets. Furthermore, it outperforms zero-shot baselines on difficult chart reasoning benchmarks. We report the performance of our approach on ChartQA, PlotQA and ChartFC. This enables large models, such as Gemini or GPT to autonomously learn how to use themselves as tools and iteratively improve without the need for additional data.

</details>

---

## 263. ProBench: Judging Multimodal Foundation Models on Open-ended Multi-domain Expert Tasks

- [ ] ProBench: Judging Multimodal Foundation Models on Open-ended Multi-domain Expert Tasks | https://aclanthology.org/2025.findings-acl.568/

- **Link**: https://aclanthology.org/2025.findings-acl.568/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Solving expert-level multimodal tasks is a key milestone in general intelligence. As the capabilities of multimodal large language models (MLLMs) continue to evolve, evaluation of frontier multimodal intelligence becomes necessary yet challenging. In this work, we introduce ProBench, a benchmark of open-ended user queries encapsulating professional expertise and advanced reasoning. ProBench consists of 4,000 high-quality samples independently collected from professionals based on their productivity demands. It spans across 10 fields and 56 sub-fields, including science, arts, humanities, coding, mathematics, and creative writing. Experimentally, we evaluate and compare 24 latest models using MLLM-as-a-Judge. Our results reveal that although the best open-source models rival the proprietary ones, they all face significant challenges in visual perception, textual understanding, domain knowledge, and advanced reasoning. Our benchmark is publicly accessible atTBC.

</details>

---

## 264. MUSE: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles

- [ ] MUSE: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles | https://aclanthology.org/2025.findings-acl.58/

- **Link**: https://aclanthology.org/2025.findings-acl.58/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Current conversational recommendation systems focus predominantly on text. However, real-world recommendation settings are generally multimodal, causing a significant gap between existing research and practical applications. To address this issue, we propose Muse, the first multimodal conversational recommendation dataset. Muse comprises 83,148 utterances from 7,000 conversations centered around the Clothing domain. Each conversation contains comprehensive multimodal interactions, rich elements, and natural dialogues. Data in Muse are automatically synthesized by a multi-agent framework powered by multimodal large language models (MLLMs). It innovatively derives user profiles from real-world scenarios rather than depending on manual design and history data for better scalability, and then it fulfills conversation simulation and optimization. Both human and LLM evaluations demonstrate the high quality of conversations in Muse. Additionally, fine-tuning experiments on three MLLMs demonstrate Muse’s learnable patterns for recommendations and responses, confirming its value for multimodal conversational recommendation. Our dataset and codes are available at https://anonymous.4open.science/r/Muse-0086.

</details>

---

## 265. Listen, Watch, and Learn to Feel: Retrieval-Augmented Emotion Reasoning for Compound Emotion Generation

- [ ] Listen, Watch, and Learn to Feel: Retrieval-Augmented Emotion Reasoning for Compound Emotion Generation | https://aclanthology.org/2025.findings-acl.590/

- **Link**: https://aclanthology.org/2025.findings-acl.590/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The ability to comprehend human emotion using multimodal large language models (MLLMs) is essential for advancing human-AI interaction and multimodal sentiment analysis. While psychology theory-based human annotations have contributed to multimodal emotion tasks, the subjective nature of emotional perception often leads to inconsistent annotations, limiting the robustness of current models. Addressing these challenges requires more fine-grained methods and evaluation frameworks. In this paper, we propose the Retrieval-Augmented Emotion Reasoning (RAER) framework, a plug-and-play module that enhances MLLMs’ ability to tackle compound and context-rich emotion tasks. To systematically evaluate model performance, we introduce the Stimulus-Armed Bandit (SAB) framework, designed to benchmark emotional reasoning capabilities. Additionally, we construct the Compound Emotion QA dataset, an AI-generated multimodal dataset aimed at strengthening emotion understanding in MLLMs. Experimental results demonstrate the effectiveness of RAER across both traditional benchmarks and SAB evaluations, highlighting its potential to enhance emotional intelligence in multimodal AI systems.

</details>

---

## 266. Large Language Models Are Natural Video Popularity Predictors

- [ ] Large Language Models Are Natural Video Popularity Predictors | https://aclanthology.org/2025.findings-acl.597/

- **Link**: https://aclanthology.org/2025.findings-acl.597/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Predicting video popularity is often framed as a supervised learning task, relying heavily on meta-information and aggregated engagement data. However, video popularity is shaped by complex cultural and social factors that such approaches often overlook. We argue that Large Language Models (LLMs), with their deep contextual awareness, can better capture these nuances. To bridge the gap between pixel-based video data and token-based LLMs, we convert frame-level visuals into sequential text representations using Vision-Language Models. This enables LLMs to process multimodal content—titles, frame-based descriptions, and captions—capturing both engagement intensity (view count) and geographic spread (number of countries where a video trends). On 13,639 popular videos, a supervised neural network using content embeddings achieves 80% accuracy, while our LLM-based approach reaches 82% without fine-tuning. Combining the neural network’s predictions with the LLM further improves accuracy to 85.5%. Moreover, the LLM generates interpretable, attribute-based explanations for its predictions. Manual validations confirm the quality of these hypotheses and address concerns about hallucinations in the video-to-text conversion process. Overall, our findings suggest that LLMs, equipped with text-based multimodal representations, offer a powerful, interpretable, and data-efficient solution for tasks requiring rich contextual insight, such as video popularity prediction.

</details>

---

## 267. A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges

- [ ] A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges | https://aclanthology.org/2025.findings-acl.614/

- **Link**: https://aclanthology.org/2025.findings-acl.614/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Mathematical reasoning, a core aspect of human cognition, is vital across many domains, from educational problem-solving to scientific advancements. As artificial general intelligence (AGI) progresses, integrating large language models (LLMs) with mathematical reasoning tasks is becoming increasingly significant. This survey provides **the first comprehensive analysis of mathematical reasoning in the era of multimodal large language models (MLLMs)**. We review over 200 studies published since 2021, and examine the state-of-the-art developments in Math-LLMs, with a focus on multimodal settings. We categorize the field into three dimensions: benchmarks, methodologies, and challenges. In particular, we explore multimodal mathematical reasoning pipeline, as well as the role of (M)LLMs and the associated methodologies. Finally, we identify five major challenges hindering the realization of AGI in this domain, offering insights into the future direction for enhancing multimodal reasoning capabilities. This survey serves as a critical resource for the research community in advancing the capabilities of LLMs to tackle complex multimodal reasoning tasks.

</details>

---

## 268. VAQUUM: Are Vague Quantifiers Grounded in Visual Data?

- [ ] VAQUUM: Are Vague Quantifiers Grounded in Visual Data? | https://aclanthology.org/2025.findings-acl.619/

- **Link**: https://aclanthology.org/2025.findings-acl.619/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vague quantifiers such as “a few” and “many” are influenced by various contextual factors, including the number of objects present in a given context. In this work, we evaluate the extent to which vision-and-language models (VLMs) are compatible with humans when producing or judging the appropriateness of vague quantifiers in visual contexts. We release a novel dataset, VAQUUM, containing 20,300 human ratings on quantified statements across a total of 1089 images. Using this dataset, we compare human judgments and VLM predictions using three different evaluation methods. Our findings show that VLMs, like humans, are influenced by object counts in vague quantifier use. However, we find significant inconsistencies across models in different evaluation settings, suggesting that judging and producing vague quantifiers rely on two different processes. We release our dataset and code at https://github.com/hughmee/vaquum.

</details>

---

## 269. FIHA: Automated Fine-grained Hallucinations Evaluations in Large Vision Language Models withDavidson Scene Graphs

- [ ] FIHA: Automated Fine-grained Hallucinations Evaluations in Large Vision Language Models withDavidson Scene Graphs | https://aclanthology.org/2025.findings-acl.622/

- **Link**: https://aclanthology.org/2025.findings-acl.622/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid development of Large Vision-Language Models (LVLMs) often comes with widespread hallucination issues, making cost-effective and comprehensive assessments increasingly vital. Current approaches mainly rely on costly annotations and are not comprehensive – in terms of evaluating all aspects, such as relations, attributes, and dependencies between aspects. Therefore, we introduce the FIHA (automated Fine-graIned Hallucination evAluation in LVLMs), which could access LVLMs hallucination in an LLM-free and annotation-free way and model the dependency between different types of hallucinations. FIHA can generate Q&A pairs on any image dataset at minimal cost, enabling hallucination assessment from both image and caption. Based on this approach, we introduce a benchmark called FIHA-v1, which consists of diverse questions on various images from three datasets. Furthermore, we use the Davidson Scene Graph (DSG) to organize the structure among Q&A pairs, in which we can increase the reliability of the evaluation. We evaluate representative models using FIHA-v1, highlighting their limitations and challenges. We released our code and data at https://github.com/confidentzzzs/FIHA.

</details>

---

## 270. Forgotten Polygons: Multimodal Large Language Models are Shape-Blind

- [ ] Forgotten Polygons: Multimodal Large Language Models are Shape-Blind | https://aclanthology.org/2025.findings-acl.620/

- **Link**: https://aclanthology.org/2025.findings-acl.620/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite strong performance on vision-language tasks, Multimodal Large Language Models (MLLMs) struggle with mathematical problem-solving, with both open-source and state-of-the-art models falling short of human performance on visual-math benchmarks. To systematically examine visual-mathematical reasoning in MLLMs, we (1) evaluate their understanding of geometric primitives, (2) test multi-step reasoning, and (3) explore a potential solution to improve visual reasoning capabilities. Our findings reveal fundamental shortcomings in shape recognition, with top models achieving under 50% accuracy in identifying regular polygons. We analyze these failures through the lens of dual-process theory and show that MLLMs rely on System 1 (intuitive, memorized associations) rather than System 2 (deliberate reasoning). Consequently, MLLMs fail to count the sides of both familiar and novel shapes, suggesting they have neither learned the concept of “sides” nor effectively process visual inputs. Finally, we propose Visually Cued Chain-of-Thought (VC-CoT) prompting, which enhances multi-step mathematical reasoning by explicitly referencing visual annotations in diagrams, boosting GPT-4o’s accuracy on an irregular polygon side-counting task from 7% to 93%. Our findings suggest that System 2 reasoning in MLLMs remains an open problem, and visually-guided prompting is essential for successfully engaging visual reasoning.

</details>

---

## 271. GlyphPattern: An Abstract Pattern Recognition for Vision-Language Models

- [ ] GlyphPattern: An Abstract Pattern Recognition for Vision-Language Models | https://aclanthology.org/2025.findings-acl.63/

- **Link**: https://aclanthology.org/2025.findings-acl.63/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Models (VLMs) have made rapid progress in reasoning across visual and textual data. While VLMs perform well on vision tasks that they are trained on, our results highlight key challenges in abstract pattern recognition. We present GlyphPattern, a 954 item dataset that pairs 318 human-written descriptions of visual patterns from 40 writing systems with three visual presentation styles.GlyphPattern evaluates abstract pattern recognition in VLMs, requiring models to understand and judge natural language descriptions of visual patterns. GlyphPattern patterns are drawn from a large-scale cognitive science investigation of human writing systems; as a result, they are rich in spatial reference and compositionality. Our experiments show that GlyphPattern is challenging for state-of-the-art VLMs (GPT-4o achieves only 55% accuracy), with marginal gains from few-shot prompting. Our detailed analysis reveals errors at multiple levels, including visual processing, natural language understanding, and pattern generalization.

</details>

---

## 272. Forget the Token and Pixel: Rethinking Gradient Ascent for Concept Unlearning in Multimodal Generative Models

- [ ] Forget the Token and Pixel: Rethinking Gradient Ascent for Concept Unlearning in Multimodal Generative Models | https://aclanthology.org/2025.findings-acl.630/

- **Link**: https://aclanthology.org/2025.findings-acl.630/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Gradient Ascent (GA) has emerged as a promising approach for concept unlearning in Multimodal Generative Models (MGMs), such as Multimodal Large Language Models (MLLMs) and Stable Diffusion Models (SDMs). Despite its effectiveness in removing undesired knowledge, GA leads to severe utility degradation in MGMs. In this paper, we explore the mechanism behind this degradation by quantifying two distinct forms of knowledge in MGMs: (i) Conceptual Knowledge, which represents specific information about concepts; (ii) Natural Knowledge, which refers to the ability to produce coherent and logically structured outputs. Our analysis reveals that applying GA globally not only removes the targeted Conceptual Knowledge but also inadvertently diminishes Natural Knowledge, resulting in utility collapse. To address this issue, we propose Forget the Token and Pixel (FTTP), a novel approach that selectively applies GA to targeted Conceptual Knowledge while preserving Natural Knowledge through Gradient Descent (GD). FTTP eliminates the need for additional retain sets and a large number of training steps, thereby reducing computational resource costs. Extensive experiments demonstrate FTTP’s efficiency and superior utility-unlearning tradeoff for both text and image generation tasks. Our source code will be released in the near future.

</details>

---

## 273. AIGuard: A Benchmark and Lightweight Detection forE-commerceAIGCRisks

- [ ] AIGuard: A Benchmark and Lightweight Detection forE-commerceAIGCRisks | https://aclanthology.org/2025.findings-acl.643/

- **Link**: https://aclanthology.org/2025.findings-acl.643/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in AI-generated content (AIGC) have heightened concerns about harmful outputs, such as misinformation and malicious misuse.Existing detection methods face two key limitations:(1) lacking real-world AIGC scenarios and corresponding risk datasets, and(2) both traditional and multimodal large language models (MLLMs) struggle to detect risks in AIGC.Towards this end, we introduce **AIGuard**, the first benchmark for AIGC risk detection in real-world e-commerce. It includes 253,420 image-text pairs (i.e., the risk content and risk description) across four critical categories: *abnormal body*, *violating physical laws*, *misleading or illogical context*, and *harmful or problematic message*.To effectively detect these risks, we propose distilling text annotations into dense soft prompts and identifying risk content through image soft prompt matching during inference.Experiments on the benchmark show that this method achieves a 9.68% higher recall than leading multimodal models while using only 25% of the training resources and improving inference speed by 37.8 times.For further research, our benchmark and code are available at [https://github.com/wenh-zhang/aiguard-dataset](https://github.com/wenh-zhang/aiguard-dataset).

</details>

---

## 274. Express What You See: Can MultimodalLLMs Decode Visual Ciphers with Intuitive Semiosis Comprehension?

- [ ] Express What You See: Can MultimodalLLMs Decode Visual Ciphers with Intuitive Semiosis Comprehension? | https://aclanthology.org/2025.findings-acl.660/

- **Link**: https://aclanthology.org/2025.findings-acl.660/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Bridging the gap between visual and language remains a pivotal challenge for the multimodal community. Traditional VQA benchmarks encounter a modality gap and over-reliance on language priors, whereas human cognition excels at intuitive semiosis, associating abstract visual symbols to linguistic semantics. Inspired by this neurocognitive mechanism, we focus on emojis, the visual cipher conveying abstract textual semantics. Specifically, we propose a novel task of generating abstract linguistics from emoji sequence images, where such reasoning underpins critical applications in cryptography, thus challenging MLLMs’ reasoning of decoding complex semantics of visual ciphers. We introduce eWe-bench (Express What you SeE), assessing MLLMs’ capability of intuitive semiosis like humans. Our data construction framework ensures high visual sensitivity and data quality, which can be extended to future data enhancement. Evaluation results on advanced MLLMs highlight critical deficiencies in visual intuitive symbolic reasoning. We believe our interesting insights for advancing visual semiosis in MLLMs will pave the way for cryptographic analysis and high-level intuitive cognition intelligence of MLLMs.

</details>

---

## 275. Grounding Task Assistance with Multimodal Cues from a Single Demonstration

- [ ] Grounding Task Assistance with Multimodal Cues from a Single Demonstration | https://aclanthology.org/2025.findings-acl.663/

- **Link**: https://aclanthology.org/2025.findings-acl.663/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

A person’s demonstration often serves as a key reference for others learning the same task. However, RGB video, the dominant medium for representing these demonstrations, often fails to capture fine-grained contextual cues such as intent, safety-critical environmental factors, and subtle preferences embedded in human behavior. This sensory gap fundamentally limits the ability of Vision Language Models (VLMs) to reason about why actions occur and how they should adapt to individual users. To address this, we introduce MICA (Multimodal Interactive Contextualized Assistance), a framework that improves conversational agents for task assistance by integrating eye gaze and speech cues. MICA segments demonstrations into meaningful sub-tasks and extracts keyframes and captions that capture fine-grained intent and user-specific cues, enabling richer contextual grounding for visual question answering. Evaluations on questions derived from real-time chat-assisted task replication show that multimodal cues significantly improve response quality over frame-based retrieval. Notably, gaze cues alone achieves 93% of speech performance, and their combination yields the highest accuracy. Task type determines the effectiveness of implicit (gaze) vs. explicit (speech) cues, underscoring the need for adaptable multimodal models. These results highlight the limitations of frame-based context and demonstrate the value of multimodal signals for real-world AI task assistance.

</details>

---

## 276. 3DM: Distill, Dynamic Drop, and Merge for Debiasing Multi-modal Large Language Models

- [ ] 3DM: Distill, Dynamic Drop, and Merge for Debiasing Multi-modal Large Language Models | https://aclanthology.org/2025.findings-acl.722/

- **Link**: https://aclanthology.org/2025.findings-acl.722/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The rapid advancement of Multi-modal Language Models (MLLMs) has significantly enhanced performance in multimodal tasks, yet these models often exhibit inherent biases that compromise their reliability and fairness. Traditional debiasing methods face a trade-off between the need for extensive labeled datasets and high computational costs. Model merging, which efficiently combines multiple models into a single one, offers a promising alternative but its usage is limited to MLLMs with the same architecture. We propose 3DM, a novel framework integrating Distill, Dynamic Drop, and Merge to address these challenges. 3DM employs knowledge distillation to harmonize models with divergent architectures and introduces a dynamic dropping strategy that assigns parameter-specific drop rates based on their contributions to bias and overall performance. This approach preserves critical weights while mitigating biases, as validated on the MMSD2.0 sarcasm detection dataset. Our key contributions include architecture-agnostic merging, dynamic dropping, and the introduction of the Bias Ratio (BR) metric for systematic bias assessment. Empirical results demonstrate that 3DM outperforms existing methods in balancing debiasing and enhancing the overall performance, offering a practical and scalable solution for deploying fair and efficient MLLMs in real-world applications.

</details>

---

## 277. CapArena: Benchmarking and Analyzing Detailed Image Captioning in theLLMEra

- [ ] CapArena: Benchmarking and Analyzing Detailed Image Captioning in theLLMEra | https://aclanthology.org/2025.findings-acl.724/

- **Link**: https://aclanthology.org/2025.findings-acl.724/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Image captioning has been a longstanding challenge in vision-language research. With the rise of LLMs, modern Vision-Language Models (VLMs) generate detailed and comprehensive image descriptions. However, benchmarking the quality of such captions remains unresolved. This paper addresses two key questions: (1) How well do VLMs actually perform on image captioning, particularly compared to humans? We built CapArena, a platform with over 6000 pairwise caption battles and high-quality human preference votes. Our Arena-style evaluation marks a milestone, showing that leading models like GPT-4o achieve or even surpass human performance, while most open-source models lag behind. (2) Can automated metrics reliably assess caption quality? Using human annotations from CapArena, we evaluate traditional and recent captioning metrics, as well as VLM-as-a-Judge. Our analysis reveals that while some metrics (e.g., METEOR) show high caption-level agreement with humans, their systematic biases lead to inconsistencies in model ranking. In contrast, VLM-as-a-Judge demonstrates robust discernment at both the caption and model levels. Building on these insights, we release CapArena-Auto, an accurate and efficient automated benchmark for detailed captioning, achieving 93.4% correlation with human rankings at just $4 per test. All data and evaluation resources have been open-sourced.

</details>

---

## 278. UQ-Merge: Uncertainty Guided Multimodal Large Language Model Merging

- [ ] UQ-Merge: Uncertainty Guided Multimodal Large Language Model Merging | https://aclanthology.org/2025.findings-acl.73/

- **Link**: https://aclanthology.org/2025.findings-acl.73/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) have gained increasing popularity as a promising framework for leveraging the strong language reasoning capabilities in the vision-language domain. Given a wide range of MLLMs, model merging potentially offers a cheap way to aggregate their diverse knowledge into a single MLLM. However, directly plug-in existing model merging approaches often leads to suboptimal performance due to (1) inclusion of harmful models that have over-confident predictions in the target task; (2) the lack of specialized designs for vision-language inputs. To tackle these pain points, we conduct pioneering investigations to dissect the merging procedures and propose an uncertainty-guided MLLM merging algorithm,i.e.,UQ-Merge, whichi) identifies beneficial candidates for merging,ii) determines the merging order and the number of helpful candidates, andiii) performs appropriate merging. Within our framework, we consider uncertainty quantification on both text and vision inputs to examine the MLLM prediction confidence, and then decide whether and when a MLLM needs to be included. It is worth mentioning that our vision-language uncertainty quantification does not require access to sample labels, making it more practical in various scenarios. Extensive experiments consistently demonstrate the superior MLLM merging performance ofUQ-Mergein both held-in and held-out vision-language benchmarks. For example, compared to existing state-of-the-art merging methods,UQ-Mergebrings substantial performance improvements of up to 44.3% on average accuracy in 12 datasets. Codes are available at https://anonymous.4open.science/r/UQ-Merge-7CD7.

</details>

---

## 279. SafeEraser: Enhancing Safety in Multimodal Large Language Models through Multimodal Machine Unlearning

- [ ] SafeEraser: Enhancing Safety in Multimodal Large Language Models through Multimodal Machine Unlearning | https://aclanthology.org/2025.findings-acl.731/

- **Link**: https://aclanthology.org/2025.findings-acl.731/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As Multimodal Large Language Models (MLLMs) develop, their potential security issues have become increasingly prominent. **Machine Unlearning (MU)**, as an effective strategy for forgetting specific knowledge in training data, has been widely used in privacy protection. However, *MU for safety in MLLM has yet to be fully explored*. To address this issue, we propose , a safety unlearning benchmark for MLLMs, consisting of 3,000 images and 28.8K VQA pairs. We comprehensively evaluate unlearning methods from two perspectives: **_forget quality_** and **_model utility_**. Our findings show that existing MU methods struggle to maintain model performance while implementing the forget operation and often suffer from **_over-forgetting_**. Hence, we introduce **Prompt Decouple (PD) Loss** to alleviate over-forgetting through decouple prompt during unlearning process. To quantitatively measure over-forgetting mitigated by PD Loss, we propose a new metric called **Safe Answer Refusal Rate (SARR)**. Experimental results demonstrate that combining PD Loss with existing unlearning methods can effectively prevent over-forgetting and achieve a decrease of 79.5% in the SARR metric of LLaVA-7B and LLaVA-13B, while maintaining forget quality and model utility. Our code and dataset will be released upon acceptance. **Warning: This paper contains examples of harmful language and images, and reader discretion is recommended.**

</details>

---

## 280. MMSciBench: Benchmarking Language Models onChinese Multimodal Scientific Problems

- [ ] MMSciBench: Benchmarking Language Models onChinese Multimodal Scientific Problems | https://aclanthology.org/2025.findings-acl.755/

- **Link**: https://aclanthology.org/2025.findings-acl.755/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in large language models (LLMs) and vision-language models (LVLMs) have shown promise across many tasks, yet their scientific reasoning capabilities remain untested, particularly in multimodal settings. We present MMSciBench, a benchmark for evaluating mathematical and physical reasoning through text-only and text-image formats, with human-annotated difficulty levels, solutions with detailed explanations, and taxonomic mappings. Evaluation of state-of-the-art models reveals significant limitations, with even the best model achieving only 63.77% accuracy and particularly struggling with visual reasoning tasks. Our analysis exposes critical gaps in complex reasoning and visual-textual integration, establishing MMSciBench as a rigorous standard for measuring progress in multimodal scientific understanding. The code for MMSciBench is open-sourced at GitHub, and the dataset is available at Hugging Face.

</details>

---

## 281. Ponder & Press: Advancing VisualGUIAgent towards General Computer Control

- [ ] Ponder & Press: Advancing VisualGUIAgent towards General Computer Control | https://aclanthology.org/2025.findings-acl.76/

- **Link**: https://aclanthology.org/2025.findings-acl.76/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Most existing GUI agents typically depend on non-vision inputs like HTML source code or accessibility trees, limiting flexibility across diverse software environments and platforms. Current multimodal large language models (MLLMs), though excel at using vision to ground real-world objects, often struggle with accurately localizing GUI elements – a critical requirement for effective GUI automation – due to the semantic gap between real-world objects and GUI elements. In this work, we introduce Ponder & Press, a divide-and-conquer framework for general computer control that uses only visual input. Our approach combines a general-purpose MLLM as an ‘interpreter’, responsible for translating high-level user instructions into detailed action descriptions, with a GUI-specific MLLM as a ‘locator’ that precisely locates GUI elements for action placement. By leveraging a purely visual input, our agent offers a versatile, human-like interaction paradigm applicable to various applications. Ponder & Press locator outperforms existing models by +22.5% on the ScreenSpot GUI grounding benchmark. More offline and interactive agent benchmarks across various GUI environments – including web pages, desktop software, and mobile UIs – demonstrate that the Ponder & Press framework achieves state-of-the-art performance, highlighting the potential of visual GUI agents.

</details>

---

## 282. VADE: Visual Attention Guided Hallucination Detection and Elimination

- [ ] VADE: Visual Attention Guided Hallucination Detection and Elimination | https://aclanthology.org/2025.findings-acl.773/

- **Link**: https://aclanthology.org/2025.findings-acl.773/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision Language Models (VLMs) have achieved significant advancements in complex visual understanding tasks. However, VLMs are prone to hallucinations—generating outputs that lack alignment with visual content. This paper addresses hallucination detection in VLMs by leveraging the visual grounding information encoded in transformer attention maps. We identify three primary challenges in this approach: the elective nature of visual grounding for certain tokens, the high-dimensional and noisy nature of attention maps, and the dynamic sequence length of attention on previous tokens. To address these, we propose VADE, a novel sequence modelling approach to effectively learn complex sequential patterns from high-dimensional and noisy attention maps for fine-grained hallucination detection and mitigation. VADE achieves an average PR-AUC of 80% in hallucination detection on M-HalDetect across four different model architectures and an 5% improvement in hallucination mitigation on MSCOCO.

</details>

---

## 283. IntelliCockpitBench: A Comprehensive Benchmark to EvaluateVLMs for Intelligent Cockpit

- [ ] IntelliCockpitBench: A Comprehensive Benchmark to EvaluateVLMs for Intelligent Cockpit | https://aclanthology.org/2025.findings-acl.798/

- **Link**: https://aclanthology.org/2025.findings-acl.798/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The integration of sophisticated Vision-Language Models (VLMs) in vehicular systems is revolutionizing vehicle interaction and safety, performing tasks such as Visual Question Answering (VQA). However, a critical gap persists due to the lack of a comprehensive benchmark for multimodal VQA models in vehicular scenarios. To address this, we propose IntelliCockpitBench, a benchmark that encompasses diverse automotive scenarios. It includes images from front, side, and rear cameras, various road types, weather conditions, and interior views, integrating data from both moving and stationary states. Notably, all images and queries in the benchmark are verified for high levels of authenticity, ensuring the data accurately reflects real-world conditions. A sophisticated scoring methodology combining human and model-generated assessments enhances reliability and consistency. Our contributions include a diverse and authentic dataset for automotive VQA and a robust evaluation metric aligning human and machine assessments. All code and data can be found athttps://github.com/Lane315/IntelliCockpitBench.

</details>

---

## 284. Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?

- [ ] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem? | https://aclanthology.org/2025.findings-acl.802/

- **Link**: https://aclanthology.org/2025.findings-acl.802/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) have shown remarkable performance for cross-modal understanding and generation, yet still suffer from severe inference costs. Recently, abundant works have been proposed to solve this problem with token pruning, which identifies the redundant tokens in MLLMs and then prunes them to reduce the computation and KV storage costs, leading to significant acceleration without training. While these methods claim efficiency gains, critical questions about their fundamental design and evaluation remain unanswered: Why do many existing approaches underperform even compared to naive random token selection? Are attention-based scoring sufficient for reliably identifying redundant tokens? Is language information really helpful during token pruning? What makes a good trade-off between token importance and duplication? Are current evaluation protocols comprehensive and unbiased? The ignorance of previous research on these problems hinders the long-term development of token pruning. In this paper, we answer these questions one by one, providing insights into the design of future token pruning methods. Codes are available in the supplementary materials.

</details>

---

## 285. UI-E2I-Synth: AdvancingGUIGrounding with Large-Scale Instruction Synthesis

- [ ] UI-E2I-Synth: AdvancingGUIGrounding with Large-Scale Instruction Synthesis | https://aclanthology.org/2025.findings-acl.809/

- **Link**: https://aclanthology.org/2025.findings-acl.809/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in Large Vision-Language Models are accelerating the development of Graphical User Interface (GUI) agents that utilize human-like vision perception capabilities to enhance productivity on digital devices. Compared to approaches predicated on GUI metadata, which are platform-dependent and vulnerable to implementation variations, vision-based approaches offer broader applicability.In this vision-based paradigm, the GUI instruction grounding, which maps user instruction to the location of corresponding element on the given screenshot, remains a critical challenge, particularly due to limited public training dataset and resource-intensive manual instruction data annotation.In this paper, we delve into unexplored challenges in this task including element-to-screen ratio, unbalanced element type, and implicit instruction. To address these challenges, we introduce a large-scale data synthesis pipelineUI-E2I-Synthfor generating varying complex instruction datasets using GPT-4o instead of human annotators. Furthermore, we propose a new GUI instruction grounding benchmarkUI-I2E-Bench, which is designed to address the limitations of existing benchmarks by incorporating diverse annotation aspects.Our model, trained on the synthesized data, achieves superior performance in GUI instruction grounding, demonstrating the advancements of proposed data synthesis pipeline.The proposed benchmark, accompanied by extensive analyses, provides practical insights for future research in this domain. We will release our dataset and benchmark to facilitate further development of GUI instruction grounding community.

</details>

---

## 286. WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inWebUI-to-Code

- [ ] WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inWebUI-to-Code | https://aclanthology.org/2025.findings-acl.815/

- **Link**: https://aclanthology.org/2025.findings-acl.815/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

With the rapid advancement of Generative AI technology, Multimodal Large Language Models(MLLMs) have the potential to act as AI software engineers capable of executing complex web application development. Considering that the model requires a confluence of multidimensional sub-capabilities to address the challenges of various development phases, constructing a multi-view evaluation framework is crucial for accurately guiding the enhancement of development efficiency. However, existing benchmarks usually fail to provide an assessment of sub-capabilities and focus solely on webpage generation outcomes. In this work, we draw inspiration from the principles of software engineering and further propose WebUIBench, a benchmark systematically designed to evaluate MLLMs in four key areas: WebUI Perception, HTML Programming, WebUI-HTML Understanding, and WebUI-to-Code. WebUIBench comprises 21K high-quality question-answer pairs derived from over 0.7K real-world websites. The extensive evaluation of 29 mainstream MLLMs uncovers the skill characteristics and various weakness that models encountered during the development process.

</details>

---

## 287. Training Multi-ModalLLMs through Dialogue Planning forHRI

- [ ] Training Multi-ModalLLMs through Dialogue Planning forHRI | https://aclanthology.org/2025.findings-acl.837/

- **Link**: https://aclanthology.org/2025.findings-acl.837/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Grounded natural language understanding in Human-Robot Interaction (HRI) requires integrating linguistic, visual, and world knowledge to ensure effective task execution. We propose an approach that enhances Multi-Modal Large Language Models (MLLMs) with a novel explicit dialogue planning phase, allowing robotic agents to systematically refine their understanding of ambiguous commands through structured clarification steps. This reduces hallucinations and improves task feasibility.To evaluate this approach, we introduce a novel dataset of over 1,100 annotated dialogues in English and Italian, designed for fine-tuning and assessing Multi-Modal models in HRI scenarios. Experimental results show that dialogue planning improves response accuracy and quality, and contributes to cross-lingual generalisation, enabling models trained in one language to transfer effectively to another. To the best of our knowledge, this is the first application of structured, goal-driven, and explicit dialogue planning in Multi-Modal LLMs for grounded interaction.

</details>

---

## 288. MVL-SIB: A Massively Multilingual Vision-Language Benchmark for Cross-Modal Topical Matching

- [ ] MVL-SIB: A Massively Multilingual Vision-Language Benchmark for Cross-Modal Topical Matching | https://aclanthology.org/2025.findings-acl.838/

- **Link**: https://aclanthology.org/2025.findings-acl.838/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing multilingual vision-language (VL) benchmarks often only cover a handful of languages. Consequently, evaluations of large vision-language models (LVLMs) predominantly target high-resource languages, underscoring the need for evaluation data for low-resource languages. To address this limitation, we introduce MVL-SIB, a massively multilingual vision-language benchmark that evaluates both cross-modal and text-only topical matching across 205 languages – over 100 more than the most multilingual existing VL benchmarks encompass. We then benchmark a range of of open-weight LVLMs together with GPT-4o(-mini) on MVL-SIB. Our results reveal that LVLMs struggle in cross-modal topic matching in lower-resource languages, performing no better than chance on languages like N’Koo. Our analysis further reveals that VL support in LVLMs declines disproportionately relative to textual support for lower-resource languages, as evidenced by comparison of cross-modal and text-only topical matching performance. We further observe that open-weight LVLMs do not benefit from representing a topic with more than one image, suggesting that these models are not yet fully effective at handling multi-image tasks. By correlating performance on MVL-SIB with other multilingual VL benchmarks, we highlight that MVL-SIB serves as a comprehensive probe of multilingual VL understanding in LVLMs.

</details>

---

## 289. Can Medical Vision-Language Pre-training Succeed with Purely Synthetic Data?

- [ ] Can Medical Vision-Language Pre-training Succeed with Purely Synthetic Data? | https://aclanthology.org/2025.findings-acl.843/

- **Link**: https://aclanthology.org/2025.findings-acl.843/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Medical Vision-Language Pre-training (MedVLP) has made significant progress in enabling zero-shot tasks for medical image understanding. However, training MedVLP models typically requires large-scale datasets with paired, high-quality image-text data, which are scarce in the medical domain. Recent advancements in Large Language Models (LLMs) and diffusion models have made it possible to generate large-scale synthetic image-text pairs. This raises the question: Can MedVLP succeed using purely synthetic data? To address this, we use off-the-shelf generative models to create synthetic radiology reports and paired Chest X-ray (CXR) images, and propose an automated pipeline to build a diverse, high-quality synthetic dataset, enabling a rigorous study that isolates model and training settings, focusing entirely from the data perspective.Our results show that MedVLP models trained exclusively on synthetic data outperform those trained on real data by 3.8% in averaged AUC on zero-shot classification. Moreover, using a combination of synthetic and real data leads to a further improvement of 9.07%. Additionally, MedVLP models trained on synthetic or mixed data consistently outperform those trained on real data in zero-shot grounding, as well as in fine-tuned classification and segmentation tasks.Our analysis suggests MedVLP trained on well-designed synthetic data can outperform models trained on real datasets, which may be limited by low-quality samples and long-tailed distributions[^1].[^1]: All data and code will be released upon acceptance.

</details>

---

## 290. See the World, Discover Knowledge: AChinese Factuality Evaluation for Large Vision Language Models

- [ ] See the World, Discover Knowledge: AChinese Factuality Evaluation for Large Vision Language Models | https://aclanthology.org/2025.findings-acl.844/

- **Link**: https://aclanthology.org/2025.findings-acl.844/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

The evaluation of factual accuracy in large vision language models (LVLMs) has lagged behind their rapid development, making it challenging to fully reflect these models’ knowledge capacity and reliability. In this paper, we introduce the first factuality-based visual question-answering benchmark in Chinese, namedChineseSimpleVQA, aimed at assessing the visual factuality of LVLMs across 8 major topics and 56 subtopics. The key features of this benchmark include a focus on theChineselanguage,diverseknowledge types, amulti-hopquestion construction,high-qualitydata,staticconsistency, andeasy-to-evaluatethrough short answers. Moreover, we contribute a rigorous data construction pipeline and decouple the visual factuality into two parts: seeing the world (i.e., object recognition) and discovering knowledge. This decoupling allows us to analyze the capability boundaries and execution mechanisms of LVLMs. Subsequently, we evaluate 34 advanced open-source and closed-source models, revealing critical performance gaps within this field.

</details>

---

## 291. Argus: Benchmarking and Enhancing Vision-Language Models for 3DRadiology Report Generation

- [ ] Argus: Benchmarking and Enhancing Vision-Language Models for 3DRadiology Report Generation | https://aclanthology.org/2025.findings-acl.845/

- **Link**: https://aclanthology.org/2025.findings-acl.845/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Automatic radiology report generation holds significant potential to streamline the labor-intensive process of report writing by radiologists, particularly for 3D radiographs such as CT scans. While CT scans are critical for clinical diagnostics, they remain less explored compared to 2D radiographs. To date, there has been no comprehensive benchmark for 3D radiograph report generation (3DRRG), nor sufficient investigation into the optimal training strategies for Vision Language Models (VLMs) in this context, particularly with respect to vision encoder choices, visual token compression, and model scaling.In this work, we make two three contributions. We curate CT-3DRRG, the largest publicly available 3D CT-report dataset, establishing a robust and diverse benchmark for evaluating VLM performance on 3DRRG. Furthermore, we propose a comprehensive training recipe for building high-performing VLMs for 3DRRG, exploring key factors such as vision encoder pretraining strategies, visual token compression, and the impact of data & model scale. Guided by these findings, we introduce Argus, a state-of-the-art family of VLMs that achieve superior performance across different model sizes and input 3D medical image resolutions, efficiently processing high-resolution 3D images up to 512 × 512 × 256.

</details>

---

## 292. Mitigating Hallucination in Multimodal Large Language Model via Hallucination-targeted Direct Preference Optimization

- [ ] Mitigating Hallucination in Multimodal Large Language Model via Hallucination-targeted Direct Preference Optimization | https://aclanthology.org/2025.findings-acl.850/

- **Link**: https://aclanthology.org/2025.findings-acl.850/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs) are known to hallucinate, which limits their practical applications. Recent works have attempted to apply Direct Preference Optimization (DPO) to enhance the performance of MLLMs, but have shown inconsistent improvements in mitigating hallucinations. To address this issue more effectively, we introduce Hallucination-targeted Direct Preference Optimization (HDPO) to reduce hallucinations in MLLMs. Unlike previous approaches, our method tackles hallucinations from their diverse forms and causes. Specifically, we develop three types of preference pair data targeting the following causes of MLLM hallucinations: (1) insufficient visual capabilities, (2) long context generation, and (3) multimodal conflicts. Experimental results demonstrate that our method achieves superior performance across multiple hallucination evaluation datasets, surpassing most state-of-the-art (SOTA) methods and highlighting the potential of our approach. Ablation studies and in-depth analyses further confirm the effectiveness of our method and suggest the potential for further improvements through scaling up.

</details>

---

## 293. Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach

- [ ] Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach | https://aclanthology.org/2025.findings-acl.868/

- **Link**: https://aclanthology.org/2025.findings-acl.868/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Prompt trading has emerged as a significant intellectual property concern in recent years, where vendors entice users by showcasing sample images before selling prompt templates that can generate similar images. This work investigates a critical security vulnerability: attackers can steal prompt templates using only a limited number of sample images. To investigate this threat, we introducePrism, a prompt-stealing benchmark consisting of 50 templates and 450 images, organized into Easy and Hard difficulty levels. To identify the vulnerabity of VLMs to prompt stealing, we proposeEvoStealer, a novel template stealing method that operates without model fine-tuning by leveraging differential evolution algorithms. The system first initializes population sets using multimodal large language models (MLLMs) based on predefined patterns, then iteratively generates enhanced offspring through MLLMs. During evolution, EvoStealer identifies common features across offspring to derive generalized templates. Our comprehensive evaluation conducted across open-source (InternVL2-26B) and closed-source models (GPT-4o and GPT-4o-mini) demonstrates that EvoStealer’s stolen templates can reproduce images highly similar to originals and effectively generalize to other subjects, significantly outperforming baseline methods with an average improvement of over 10%. Moreover, our cost analysis reveals that EvoStealer achieves template stealing with negligible computational expenses. Our code and dataset are available at https://whitepagewu.github.io/evostealer-site.

</details>

---

## 294. MAGIC-VQA: Multimodal And Grounded Inference with Commonsense Knowledge for Visual Question Answering

- [ ] MAGIC-VQA: Multimodal And Grounded Inference with Commonsense Knowledge for Visual Question Answering | https://aclanthology.org/2025.findings-acl.872/

- **Link**: https://aclanthology.org/2025.findings-acl.872/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Visual Question Answering (VQA) necessitates models to reason effectively across visual and textual modalities. However, existing Large Vision-Language Models (LVLMs) often fall short in achieving human-like reasoning due to a lack of integrated commonsense knowledge, limiting their robustness and accuracy in real-world scenarios where both explicit facts and implicit understanding are crucial. To address this challenge, we present MAGIC-VQA: Multimodal And Grounded Inference with Commonsense Knowledge, a novel framework designed to enhance multimodal inference by integrating commonsense reasoning. MAGIC-VQA introduces a three-stage process: (1) Explicit Commonsense Knowledge Retrieval from external knowledge graphs, (2) By-Type Commonsense Knowledge Post-Processing to refine contextual relevance, and (3) Implicit Commonsense Knowledge Augmentation using a heterogeneous graph processed by a Graph Neural Network (GNN). These stages collectively enable nuanced, context-aware reasoning without extensive pre-training or intricate prompt tuning.Our MAGIC-VQA significantly improves comprehensive benchmark datasets, surpassing existing models in tasks requiring advanced commonsense reasoning. MAGIC-VQA establishes a robust pathway for integrating commonsense knowledge into VQA, bridging the gap between vision-language inputs and high-level reasoning for improved reliability and contextual accuracy.

</details>

---

## 295. VP-MEL: Visual Prompts Guided Multimodal Entity Linking

- [ ] VP-MEL: Visual Prompts Guided Multimodal Entity Linking | https://aclanthology.org/2025.findings-acl.880/

- **Link**: https://aclanthology.org/2025.findings-acl.880/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal entity linking (MEL), a task aimed at linking mentions within multimodal contexts to their corresponding entities in a knowledge base (KB), has attracted much attention due to its wide applications in recent years. However, existing MEL methods often rely on mention words as retrieval cues, which limits their ability to effectively utilize information from both images and text. This reliance causes MEL to struggle with accurately retrieving entities in certain scenarios, especially when the focus is on image objects or mention words are missing from the text. To solve these issues, we introduce a Visual Prompts guided Multimodal Entity Linking (VP-MEL) task. Given a text-image pair, VP-MEL aims to link a marked region (i.e., visual prompt) in an image to its corresponding entities in the knowledge base. To facilitate this task, we present a new dataset, VPWiki, specifically designed for VP-MEL. Furthermore, we propose a framework named IIER, which enhances visual feature extraction using visual prompts and leverages the pre-trained Detective-VLM model to capture latent information. Experimental results on the VPWiki dataset demonstrate that IIER outperforms baseline methods across multiple benchmarks for the VP-MEL task.

</details>

---

## 296. Libra: Leveraging Temporal Images for Biomedical Radiology Analysis

- [ ] Libra: Leveraging Temporal Images for Biomedical Radiology Analysis | https://aclanthology.org/2025.findings-acl.888/

- **Link**: https://aclanthology.org/2025.findings-acl.888/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Radiology report generation (RRG) requires advanced medical image analysis, effective temporal reasoning, and accurate text generation. While multimodal large language models (MLLMs) align with pre-trained vision encoders to enhance visual-language understanding, most existing methods rely on single-image analysis or rule-based heuristics to process multiple images, failing to fully leverage temporal information in multi-modal medical datasets. In this paper, we introduce **Libra**, a temporal-aware MLLM tailored for chest X-ray report generation. Libra combines a radiology-specific image encoder with a novel Temporal Alignment Connector (**TAC**), designed to accurately capture and integrate temporal differences between paired current and prior images. Extensive experiments on the MIMIC-CXR dataset demonstrate that Libra establishes a new state-of-the-art benchmark among similarly scaled MLLMs, setting new standards in both clinical relevance and lexical accuracy. All source code and data are publicly available at: https://github.com/X-iZhang/Libra.

</details>

---

## 297. MC-MKE: A Fine-Grained Multimodal Knowledge Editing Benchmark Emphasizing Modality Consistency

- [ ] MC-MKE: A Fine-Grained Multimodal Knowledge Editing Benchmark Emphasizing Modality Consistency | https://aclanthology.org/2025.findings-acl.896/

- **Link**: https://aclanthology.org/2025.findings-acl.896/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are prone to non-factual or outdated knowledge issues, highlighting the importance of knowledge editing. Many benchmark has been proposed for researching multimodal knowledge editing. However, previous benchmarks focus on limited scenarios due to the lack of rigorous definition of multimodal knowledge. To better evaluate multimodal knowledge editing, we propose a decomposed definition of multimodal knowledge. Following the decomposed definition of multimodal knowledge, we introduce three scenarios and a novel requirement modality consistency. We construct MC-MKE, a fine-grained **M**ultimodal **K**nowledge **E**diting benchmark emphasizing **M**odality **C**onsistency through strict data selection. We evaluate four multimodal knowledge editing methods on MC-MKE, revealing their limitations, particularly in terms of modality consistency. Our work highlights the challenges posed by multimodal knowledge editing and motivates further research in developing effective techniques for this task.

</details>

---

## 298. Look & Mark: Leveraging Radiologist Eye Fixations and Bounding boxes in Multimodal Large Language Models for ChestX-ray Report Generation

- [ ] Look & Mark: Leveraging Radiologist Eye Fixations and Bounding boxes in Multimodal Large Language Models for ChestX-ray Report Generation | https://aclanthology.org/2025.findings-acl.909/

- **Link**: https://aclanthology.org/2025.findings-acl.909/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent advancements in multimodal Large Language Models (LLMs) have significantly enhanced the automation of medical image analysis, particularly in generating radiology reports from chest X-rays (CXR). However, these models still suffer from hallucinations and clinically significant errors, limiting their reliability in real-world applications. In this study, we propose Look & Mark (L&M), a novel grounding fixation strategy that integrates radiologist eye fixations (Look) and bounding box annotations (Mark) into the LLM prompting framework. Unlike conventional fine-tuning, L&M leverages in-context learning to achieve substantial performance gains without retraining. When evaluated across multiple domain-specific and general-purpose models, L&M demonstrates significant gains, including a 1.2% improvement in overall metrics (A.AVG) for CXR-LLaVA compared to baseline prompting and a remarkable 9.2% boost for LLaVA-Med. General-purpose models also benefit from L&M combined with in-context learning, with LLaVA-OV achieving an 87.3% clinical average performance (C.AVG)—the highest among all models, even surpassing those explicitly trained for CXR report generation. Expert evaluations further confirm that L&M reduces clinically significant errors (by 0.43 average errors per report), such as false predictions and omissions, enhancing both accuracy and reliability. These findings highlight L&M’s potential as a scalable and efficient solution for AI-assisted radiology, paving the way for improved diagnostic workflows in low-resource clinical settings.

</details>

---

## 299. JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse

- [ ] JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse | https://aclanthology.org/2025.findings-acl.920/

- **Link**: https://aclanthology.org/2025.findings-acl.920/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recently, action-based decision-making in open-world environments has gained significant attention. Visual Language Action (VLA) models, pretrained on large-scale web datasets, have shown promise in decision-making tasks. However, previous work has primarily focused on action post-training, often neglecting enhancements to the foundation model itself. In response, we introduce Act from Visual Language Post-Training (ActVLP), a novel training paradigm. ActVLP distinctively enhances the foundation model prior to action-specific tuning by first post-training it on a curated set of environment-specific visual and linguistic tasks using self-supervised learning. This initial stage significantly improves the model’s capabilities in world knowledge, visual recognition, and spatial grounding. Subsequently, this strengthened VLM undergoes action post-training via imitation learning on trajectory datasets.Following this paradigm, we develop JARVIS-VLA, the first VLA model in Minecraft that can follow human instructions on over 1k different atomic tasks, including crafting, smelting, cooking, mining, and killing. Our experiments demonstrate that our ActVLP paradigm leads to a significant 40% improvement over the best agent baseline on a diverse set of atomic tasks. Furthermore, JARVIS-VLA surpasses traditional imitation learning-based policies in Minecraft, achieving state-of-the-art performance. We have open-sourced the code, models, and datasets to foster further research.The project page can be found athttps://craftjarvis.github.io/JarvisVLA.

</details>

---

## 300. Generative Frame Sampler for Long Video Understanding

- [ ] Generative Frame Sampler for Long Video Understanding | https://aclanthology.org/2025.findings-acl.921/

- **Link**: https://aclanthology.org/2025.findings-acl.921/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Despite recent advances in Video Large Language Models (VideoLLMs), effectively understanding long-form videos remains a significant challenge. Perceiving lengthy videos containing thousands of frames poses substantial computational burden. To mitigate this issue, this paper introduces Generative Frame Sampler (GenS), a plug-and-play module integrated with VideoLLMs to facilitate efficient lengthy video perception. Built upon a lightweight VideoLLM, GenS leverages its inherent vision-language capabilities to identify question-relevant frames. To facilitate effective retrieval, we construct GenS-Video-150K, a large-scale video instruction dataset with dense frame relevance annotations. Extensive experiments demonstrate that GenS consistently boosts the performance of various VideoLLMs, including open-source models (Qwen2-VL-7B, Aria-25B, LLaVA-Video-7B/72B) and proprietary assistants (GPT-4o, Gemini). When equipped with GenS, open-source VideoLLMs achieve impressive state-of-the-art results on long-form video benchmarks: LLaVA-Video-72B reaches 66.8 (+4.3) on LongVideoBench and 77.0 (+2.7) on MLVU, while Aria obtains 39.2 on HourVideo surpassing the Gemini-1.5-pro by 1.9 points.

</details>

---

## 301. VISIAR: EmpowerMLLMfor Visual Story Ideation

- [ ] VISIAR: EmpowerMLLMfor Visual Story Ideation | https://aclanthology.org/2025.findings-acl.945/

- **Link**: https://aclanthology.org/2025.findings-acl.945/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Ideation, the process of forming ideas from concepts, is a big part of the content creation process. However, the noble goal of helping visual content creators by suggesting meaningful sequences of visual assets from a limited collection is challenging. It requires a nuanced understanding of visual assets and the integration of open-world knowledge to support creative exploration. Despite its importance, this task has yet to be explored fully in existing literature. To fill this gap, we propose Visual Story Ideation, a novel and underexplored task focused on the automated selection and arrangement of visual assets into coherent sequences that convey expressive storylines.We also present VISIAR, Visual Ideation through Sequence Integration and Asset Rearrangement, a robust framework leveraging Multimodal Large Language Models (MLLMs), and a novel Story Graph mechanism. Our framework operates in three key stages: visual content understanding, candidate asset selection, and asset rearrangement via MLLMs. In addition, we curated a new benchmark dataset, called VTravel, to evaluate our methods both qualitatively and quantitatively.User studies and GPT-as-the-judge evaluation show that our approach surpasses GPT-4o based baseline by an average of 33.5% and 18.5% across three different metrics, demonstrating the effectiveness of our framework for generating compelling visual stories.

</details>

---

## 302. Biases Propagate in Encoder-based Vision-Language Models: A Systematic Analysis From Intrinsic Measures to Zero-shot Retrieval Outcomes

- [ ] Biases Propagate in Encoder-based Vision-Language Models: A Systematic Analysis From Intrinsic Measures to Zero-shot Retrieval Outcomes | https://aclanthology.org/2025.findings-acl.955/

- **Link**: https://aclanthology.org/2025.findings-acl.955/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

To build fair AI systems we need to understand how social-group biases intrinsic to foundational encoder-based vision-language models (VLMs) manifest in biases in downstream tasks. In this study, we demonstrate that intrinsic biases in VLM representations systematically “carry over” or propagate into zero-shot retrieval tasks, revealing how deeply rooted biases shape a model’s outputs. We introduce a controlled framework to measure this propagation by correlating (a) intrinsic measures of bias in the representational space with (b) extrinsic measures of bias in zero-shot text-to-image (TTI) and image-to-text (ITT) retrieval. Results show substantial correlations between intrinsic and extrinsic bias, with an average𝜌= 0.83±0.10. This pattern is consistent across 114 analyses, both retrieval directions, six social groups, and three distinct VLMs. Notably, we find that larger/better-performing models exhibit greater bias propagation, a finding that raises concerns given the trend towards increasingly complex AI models. Our framework introduces baseline evaluation tasks to measure the propagation of group and valence signals. Investigations reveal that underrepresented groups experience less robust propagation, further skewing their model-related outcomes.

</details>

---

## 303. MVTamperBench: Evaluating Robustness of Vision-Language Models

- [ ] MVTamperBench: Evaluating Robustness of Vision-Language Models | https://aclanthology.org/2025.findings-acl.963/

- **Link**: https://aclanthology.org/2025.findings-acl.963/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal Large Language Models (MLLMs), are recent advancement of Vision-Language Models (VLMs) that have driven major advances in video understanding. However, their vulnerability to adversarial tampering and manipulations remains underexplored. To address this gap, we introduceMVTamperBench, a benchmark that systematically evaluates MLLM robustness against five prevalent tampering techniques: rotation, masking, substitution, repetition, and dropping; based on real-world visual tampering scenarios such as surveillance interference, social media content edits, and misinformation injection. MVTamperBench comprises ~3.4K original videos, expanded into over ~17K tampered clips covering 19 distinct video manipulation tasks. This benchmark challenges models to detect manipulations in spatial and temporal coherence. We evaluate 45 recent MLLMs from 15+ model families. We reveal substantial variability in resilience across tampering types and show that larger parameter counts do not necessarily guarantee robustness. MVTamperBench sets a new benchmark for developing tamper-resilient MLLM in safety-critical applications, including detecting clickbait, preventing harmful content distribution, and enforcing policies on media platforms. We release all code, data, and benchmark to foster open research in trustworthy video understanding.

</details>

---

## 304. Multimodal Inconsistency Reasoning (MMIR): A New Benchmark for Multimodal Reasoning Models

- [ ] Multimodal Inconsistency Reasoning (MMIR): A New Benchmark for Multimodal Reasoning Models | https://aclanthology.org/2025.findings-acl.964/

- **Link**: https://aclanthology.org/2025.findings-acl.964/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Existing Multimodal Large Language Models (MLLMs) are predominantly trained and tested on consistent visual-textual inputs, leaving open the question of whether they can handle inconsistencies in real-world, layout-rich content. To bridge this gap, we propose the Multimodal Inconsistency Reasoning (MMIR) benchmark to assess MLLMs’ ability to detect and reason about semantic mismatches in artifacts such as webpages, presentation slides, and posters. MMIR comprises 534 challenging samples, each containing synthetically injected errors across five reasoning-heavy categories: Factual Contradiction, Identity Misattribution, Contextual Mismatch, Quantitative Discrepancy, and Temporal/Spatial Incoherence. We evaluate eight state-of-the-art MLLMs, showing that models with dedicated multimodal reasoning capabilities, such as o1, substantially outperform their counterparts while open-source models remain particularly vulnerable to inconsistency errors. Detailed error analyses further show that models excel in detecting inconsistencies confined to a single modality, particularly in text, but struggle with cross-modal conflicts and complex layouts. Probing experiments reveal that single-modality prompting, including Chain-of-Thought (CoT) and Set-of-Mark (SoM) methods, yields marginal gains, revealing a key bottleneck in cross-modal reasoning. Our findings highlight the need for advanced multimodal reasoning and point to future research on multimodal inconsistency.

</details>

---

## 305. Vision-Language Models Struggle to Align Entities across Modalities

- [ ] Vision-Language Models Struggle to Align Entities across Modalities | https://aclanthology.org/2025.findings-acl.965/

- **Link**: https://aclanthology.org/2025.findings-acl.965/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Cross-modal entity linking refers to the ability to align entities and their attributes across different modalities. While cross-modal entity linking is a fundamental skill needed for real-world applications such as multimodal code generation, fake news detection, or scene understanding, it has not been thoroughly studied in the literature. In this paper, we introduce a new task and benchmark to address this gap. Our benchmark, MATE, consists of 5.5k evaluation instances featuring visual scenes aligned with their textual representations. To evaluate cross-modal entity linking performance, we design a question-answering task that involves retrieving one attribute of an object in one modality based on a unique attribute of that object in another modality. We evaluate state-of-the-art Vision-Language Models (VLMs) and humans on this task, and find that VLMs struggle significantly compared to humans, particularly as the number of objects in the scene increases. Our analysis also shows that, while chain-of-thought prompting can improve VLM performance, models remain far from achieving human-level proficiency. These findings highlight the need for further research in cross-modal entity linking and show that MATE is a strong benchmark to support that progress.

</details>

---

## 306. ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering

- [ ] ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering | https://aclanthology.org/2025.findings-acl.978/

- **Link**: https://aclanthology.org/2025.findings-acl.978/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Charts are ubiquitous, as people often use them to analyze data, answer questions, and discover critical insights. However, performing complex analytical tasks with charts requires significant perceptual and cognitive effort. Chart Question Answering (CQA) systems automate this process by enabling models to interpret and reason with visual representations of data. However, existing benchmarks like ChartQA lack real-world diversity and have recently shown performance saturation with modern large vision-language models (LVLMs). To address these limitations, we introduce ChartQAPro, a new benchmark that includes 1,341 charts from 99 diverse sources, spanning various chart types—including infographics and dashboards—and featuring 1,948 questions in various types, such as multiple-choice, conversational, hypothetical, and unanswerable questions, to better reflect real-world challenges. Our evaluations with 21 models show a substantial performance drop for LVLMs on ChartQAPro; e.g., Claude Sonnet 3.5 scores 90.5% on ChartQA but only 55.81% on ChartQAPro, underscoring the complexity of chart reasoning. We complement our findings with detailed error analyses and ablation studies, identifying key challenges and opportunities for advancing LVLMs in chart understanding and reasoning. We release ChartQAPro at https://github.com/vis-nlp/ChartQAPro.

</details>

---

## 307. V-ALPHASOCIAL: Benchmark and Self-Reflective Chain-of-Thought Generation for Visual Social Commonsense Reasoning

- [ ] V-ALPHASOCIAL: Benchmark and Self-Reflective Chain-of-Thought Generation for Visual Social Commonsense Reasoning | https://aclanthology.org/2025.findings-acl.975/

- **Link**: https://aclanthology.org/2025.findings-acl.975/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Social commonsense reasoning naturally involves both the verbal and non-verbal cues of a social interaction. It is important for Large Vision-Language Models (VLMs) to leverage both textual and visual information in performing tasks like social understanding and reasoning. However, while current LLMs have shown good social reasoning capabilities in textual context, whether they can effectively incorporate visual information in social comprehension remains under-explored. To narrow the gap, we first construct and propose a benchmark: V-Social, featuring well-aligned text and visual content, tailored to assess visual social commonsense for multimodal foundation models. Through experimenting with V-Social, we find that even the most advanced VLM, GPT-4o, often falls short in social commonsense reasoning. This highlights the critical need to enhance the social grounding of VLMs. One major obstacle for improving this is the lack of high-quality data with good reasoning process. To overcome this obstacle, we introduce V-AlphaSocial, a novel method that generates high-quality chain-of-thought reasoning paths from unlabeled data. We design a visual reasoning reward model to improve VLM, and then iteratively refine both the VLM and the reward model. Our extensive analysis showcases how our method enhances social commonsense reasoning, proposing an effective approach that facilitates deeper exploration into field.

</details>

---

## 308. From Observation to Understanding: Front-Door Adjustments with Uncertainty Calibration for Enhancing Egocentric Reasoning inLVLMs

- [ ] From Observation to Understanding: Front-Door Adjustments with Uncertainty Calibration for Enhancing Egocentric Reasoning inLVLMs | https://aclanthology.org/2025.findings-acl.979/

- **Link**: https://aclanthology.org/2025.findings-acl.979/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Recent progress in large vision-language models (LVLMs) has shown substantial potential across a broad spectrum of third-person tasks. However, adapting these LVLMs to egocentric scenarios remains challenging due to their third-person training bias. Existing methods that adapt LVLMs for first-person tasks often overlook critical agent-environment interactions, limiting their ability to perform egocentric reasoning. To address these challenges, we propose a novel zero-shot paradigm termed Front-Door Adjustments with Uncertainty Calibration (FRUIT) to enhance the egocentric reasoning abilities of LVLMs by simulating human causal reasoning. Specifically, the FRUIT operates in two stages: observation and understanding. Unlike conventional prompting techniques, we formalize egocentric reasoning using a structural causal model. Then, we ground interaction regions and expand them into hierarchical visual cues, augmented with corresponding captions, to form the initial observations. To reduce noise in these observations, we employ uncertainty calibration to filter out unreliable information. These refined observations as mediators are then incorporated into the prompt template, guiding the model to understand semantics from a first-person perspective. Extensive experiments conducted on the EgoThink benchmark demonstrate that our FRUIT method consistently enhances the performance of existing LVLMs on six distinct tasks. Our code is available at https://github.com/Mrshenshen/FRUIT.

</details>

---

## 309. EgoNormia: Benchmarking Physical-Social Norm Understanding

- [ ] EgoNormia: Benchmarking Physical-Social Norm Understanding | https://aclanthology.org/2025.findings-acl.985/

- **Link**: https://aclanthology.org/2025.findings-acl.985/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Human activity is moderated by norms; however, supervision for normative reasoning is sparse, particularly where norms are physically- or socially-grounded. We thus present EgoNormia\lVert𝜖\rVert, comprising 1,853 (200 for EgoNormia-verified) multiple choice questions (MCQs) grounded within ego-centric videos of human interactions, enabling the evaluation and improvement of normative reasoning in vision-language models (VLMs). spans seven norm categories: safety, privacy, proxemics, politeness, cooperation, coordination/proactivity, and communication/legibility. To compile this dataset at scale, we propose a novel pipeline to generate grounded MCQs from raw egocentric video. Our work demonstrates that current state-of-the-art VLMs lack robust grounded norm understanding, scoring a maximum of 54% on EgoNormia and 58% on EgoNormia-verified, with performance across norm categories indicating significant risks of safety and privacy when VLMs are used in real-world agents. We additionally explore methods for improving normative understanding, demonstrating a naive retrieval-based generation (RAG) method using can enhance normative reasoning in VLMs.

</details>

---

## 310. Don’t Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models

- [ ] Don’t Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models | https://aclanthology.org/2025.findings-acl.99/

- **Link**: https://aclanthology.org/2025.findings-acl.99/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision Language Models (LVLMs) demonstrate strong capabilities in visual understanding and description, yet often suffer from hallucinations, attributing incorrect or misleading features to images. We observe that LVLMs disproportionately focus on a small subset of image tokens—termed blind tokens—which are typically irrelevant to the query (e.g., background or non-object regions). We hypothesize that such attention misalignment plays a key role in generating hallucinated responses. To mitigate this issue, we propose Attentional Vision Calibration (AvisC), a test-time approach that dynamically recalibrates the influence of blind tokens without modifying the underlying attention mechanism. AvisC first identifies blind tokens by analyzing layer-wise attention distributions over image tokens, then employs a contrastive decoding strategy to balance the influence of original and blind-token-biased logits. Experiments on standard benchmarks, including POPE, MME, and AMBER, demonstrate that AvisC effectively reduces hallucinations in LVLMs.

</details>

---

## 311. Strengths and Limitations of Word-Based Task Explainability in Vision Language Models: a Case Study on Biological Sex Biases in the Medical Domain

- [ ] Strengths and Limitations of Word-Based Task Explainability in Vision Language Models: a Case Study on Biological Sex Biases in the Medical Domain | https://aclanthology.org/2025.gebnlp-1.12/

- **Link**: https://aclanthology.org/2025.gebnlp-1.12/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) can achieve high accuracy in medical applications but can retain demographic biases from training data. While multiple works have identified the presence of these biases in many VLMs, it remains unclear how strong their impact at the inference level is. In this work, we study how well a task-level explainability method based on linear combinations of words can detect multiple types of biases, with a focus on medical image classification. By manipulating the training datasets with demographic and non-demographic biases, we show how the adopted approach can detect explicitly encoded biases but fails with implicitly encoded ones, particularly biological sex. Our results suggest that such a failure likely stems from misalignment between sex-describing features in image versus text modalities. Our findings highlight limitations in the evaluated explainability method for detecting implicit biases in medical VLMs.

</details>

---

## 312. Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations fromLLMs

- [ ] Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations fromLLMs | https://aclanthology.org/2025.gebnlp-1.18/

- **Link**: https://aclanthology.org/2025.gebnlp-1.18/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

One of the goals of fairness research in NLP is to measure and mitigate stereotypical biases that are propagated by NLP systems. However, such work tends to focus on single axes of bias (most often gender) and the English language. Addressing these limitations, we contribute the first study of multilingual intersecting country and gender biases, with a focus on occupation recommendations generated by large language models. We construct a benchmark of prompts in English, Spanish and German, where we systematically vary country and gender, using 25 countries and four pronoun sets. Then, we evaluate a suite of 5 Llama-based models on this benchmark, finding that LLMs encode significant gender and country biases. Notably, we find that even when models show parity for gender or country individually, intersectional occupational biases based on both country and gender persist. We also show that the prompting language significantly affects bias, and instruction-tuned models consistently demonstrate the lowest and most stable levels of bias. Our findings highlight the need for fairness researchers to use intersectional and multilingual lenses in their work.

</details>

---

## 313. Analyzing the Sensitivity of Vision Language Models in Visual Question Answering

- [ ] Analyzing the Sensitivity of Vision Language Models in Visual Question Answering | https://aclanthology.org/2025.gem-1.36/

- **Link**: https://aclanthology.org/2025.gem-1.36/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We can think of Visual Question Answering as a (multimodal) conversation between a human and an AI system. Here, we explore the sensitivity of Vision Language Models (VLMs) through the lens of cooperative principles of conversation proposed by Grice. Specifically, even when Grice’s maxims of conversation are flouted, humans typically do not have much difficulty in understanding the conversation even though it requires more cognitive effort. Here, we study if VLMs are capable of handling violations to Grice’s maxims in a manner that is similar to humans. Specifically, we add modifiers to human-crafted questions and analyze the response of VLMs to these modifiers. We use three state-of-the-art VLMs in our study, namely, GPT-4o, Claude-3.5-Sonnet and Gemini-1.5-Flash on questions from the VQA v2.0 dataset. Our initial results seem to indicate that the performance of VLMs consistently diminish with the addition of modifiers which indicates our approach as a promising direction to understand the limitations of VLMs.

</details>

---

## 314. Big Escape Benchmark: Evaluating Human-Like Reasoning in Language Models via Real-World Escape Room Challenges

- [ ] Big Escape Benchmark: Evaluating Human-Like Reasoning in Language Models via Real-World Escape Room Challenges | https://aclanthology.org/2025.gem-1.42/

- **Link**: https://aclanthology.org/2025.gem-1.42/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have recently demonstrated remarkable reasoning capabilities across a wide range of tasks. While many benchmarks have been developed on specific academic subjects, coding, or constrained visual tasks, they often fail to fully capture the breadth, diversity, and dynamic nature of real-world human reasoning. Further, the creation of high-quality, complex multimodal reasoning benchmarks typically requires significant manual effort and expert annotation, which is costly and time-consuming.To address these limitations, we introduce Big Escape Bench, a novel multimodal reasoning benchmark derived from popular reality shows and television programs. Big Escape Bench leverages unique characteristics of TV content, providing a rich source of challenging and realistic multimodal reasoning problems. Key advantages include: questions guaranteed to be human-solvable and of moderate difficulty; problems reflecting diverse, real-world scenarios and knowledge domains; high inherent quality due to content generated by professional program teams.Notably, we develop an automated pipeline to construct the data from these programs into a standardized benchmark format, significantly reducing the manual effort compared to traditional dataset construction. We have conducted extensive experiments to evaluate state-of-the-art (SOTA) LLMs and Multimodal Large Language Models (MLLMs) on Big Escape Bench. Our results reveal a surprising performance gap: while the questions are easily solved by human viewers (about 60% in accuracy), the performance of even the most advanced models (best 40.50% in accuracy) is significantly lower than human-level accuracy. This highlights that despite recent progress, MLLMs still face substantial challenges in robustly performing the kind of diverse, dynamic, and context-dependent reasoning that is trivial for humans in routine situations. Big Escape Bench serves as a valuable tool for identifying current limitations of MLLMs and fostering future research towards more human-like multimodal reasoning.

</details>

---

## 315. Coreference as an indicator of context scope in multimodal narrative

- [ ] Coreference as an indicator of context scope in multimodal narrative | https://aclanthology.org/2025.gem-1.67/

- **Link**: https://aclanthology.org/2025.gem-1.67/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We demonstrate that large multimodal language models differ substantially from humans in the distribution of coreferential expressions in a visual storytelling task. We introduce a number of metrics to quantify the characteristics of coreferential patterns in both human- and machine-written texts. Humans distribute coreferential expressions in a way that maintains consistency across texts and images, interleaving references to different entities in a highly varied way. Machines are less able to track mixed references, despite achieving perceived improvements in generation quality. Materials, metrics, and code for our study are available at https://github.com/GU-CLASP/coreference-context-scope.

</details>

---

## 316. MLAN: Language-Based Instruction Tuning Preserves and Transfers Knowledge in Multimodal Language Models

- [ ] MLAN: Language-Based Instruction Tuning Preserves and Transfers Knowledge in Multimodal Language Models | https://aclanthology.org/2025.knowllm-1.6/

- **Link**: https://aclanthology.org/2025.knowllm-1.6/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We present a novel visual instruction tuning strategy to improve the zero-shot task generalization of multimodal large language models by building a firm text-only knowledge base. Existing work lacks sufficient experimentation on the importance of each modality in the instruction tuning stage, often using a majority of vision-language data while keeping text-only data limited and fixing mixtures of modalities. By incorporating diverse text-only data in the visual instruction tuning stage, we vary vision-language data in various controlled experiments to investigate the importance of modality in visual instruction tuning. Our comprehensive evaluation shows that the text-heavy instruction tuning approach is able to perform on par with traditional vision-heavy mixtures on both modalities across 12 general datasets while using as low as half the total training tokens. We find that simply increasing sufficiently diverse text-only data enables transfer of instruction following ability and domain knowledge across modalities while being more efficient than the vision-language approach.

</details>

---

## 317. Quantifying Memorization and Parametric Response Rates in Retrieval-Augmented Vision-Language Models

- [ ] Quantifying Memorization and Parametric Response Rates in Retrieval-Augmented Vision-Language Models | https://aclanthology.org/2025.l2m2-1.10/

- **Link**: https://aclanthology.org/2025.l2m2-1.10/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) demonstrate remarkable capabilities in question answering (QA), but metrics for assessing their reliance on memorization versus retrieval remain underdeveloped. Moreover, while finetuned models are state-of-the-art on closed-domain tasks, general-purpose models like GPT-4o exhibit strong zero-shot performance. This raises questions about the trade-offs between memorization, generalization, and retrieval. In this work, we analyze the extent to which multimodal retrieval-augmented VLMs memorize training data compared to baseline VLMs. Using the WebQA benchmark, we contrast finetuned models with baseline VLMs on multihop retrieval and question answering, examining the impact of finetuning on data memorization. To quantify memorization in end-to-end retrieval and QA systems, we propose several proxy metrics by investigating instances where QA succeeds despite retrieval failing. In line with existing work, we find that finetuned models rely more heavily on memorization than retrieval-augmented VLMs, and achieve higher accuracy as a result (72% vs 52% on WebQA test set). Finally, we present the first empirical comparison of the parametric effect between text and visual modalities. Here, we find that image-based questions have parametric response rates that are consistently 15-25% higher than for text-based questions in the WebQA dataset. As such, our measures pose a challenge for future work, both to account for differences in model memorization across different modalities and more generally to reconcile memorization and generalization in joint Retrieval-QA tasks.

</details>

---

## 318. Beyond Words: Multilingual and Multimodal Red Teaming ofMLLMs

- [ ] Beyond Words: Multilingual and Multimodal Red Teaming ofMLLMs | https://aclanthology.org/2025.llmsec-1.15/

- **Link**: https://aclanthology.org/2025.llmsec-1.15/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Multimodal large language models (MLLMs) are increasingly deployed in real-world applications, yet their safety remains underexplored, particularly in multilingual and visual contexts. In this work, we present a systematic red teaming framework to evaluate MLLM safeguards using adversarial prompts translated into seven languages and delivered via four input modalities: plain text, jailbreak prompt + text, text rendered as an image, and jailbreak prompt + text rendered as an image. We find that rendering prompts as images increases attack success rates and reduces refusal rates, with the effect most pronounced in lower-resource languages such as Slovenian, Czech, and Valencian. Our results suggest that vision-based multilingual attacks expose a persistent gap in current alignment strategies, highlighting the need for robust multilingual and multimodal MLLM safety evaluation and mitigation of these risks. We make our code and data available.

</details>

---

## 319. MultiReflect: Multimodal Self-ReflectiveRAG-based Automated Fact-Checking

- [ ] MultiReflect: Multimodal Self-ReflectiveRAG-based Automated Fact-Checking | https://aclanthology.org/2025.magmar-1.1/

- **Link**: https://aclanthology.org/2025.magmar-1.1/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this work, we introduce MultiReflect, a novel multimodal self-reflective Retrieval Augmented Generation (RAG)-based automated fact-checking pipeline. MultiReflect is designed to address the challenges of rapidly outdated information, limitations in human query capabilities, and expert knowledge barriers in fact-checking. Our proposed pipeline leverages the latest advancements in Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) to enhance fact verification across text and images. Specifically, by integrating multimodal data processing with RAG-based evidence reflection, our system improves the accuracy of fact-checking by utilizing internet-sourced verification. We evaluate our results on the VERITE benchmarks and using several multimodal LLMs, outperforming baselines in binary classification.

</details>

---

## 320. CollEX– A Multimodal AgenticRAGSystem Enabling Interactive Exploration of Scientific Collections

- [ ] CollEX– A Multimodal AgenticRAGSystem Enabling Interactive Exploration of Scientific Collections | https://aclanthology.org/2025.magmar-1.2/

- **Link**: https://aclanthology.org/2025.magmar-1.2/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce CollEx, an innovative multimodal agentic Retrieval-Augmented Generation (RAG) system designed to enhance interactive exploration of extensive scientific collections. Given the overwhelming volume and inherent complexity of scientific collections, conventional search systems often lack necessary intuitiveness and interactivity, presenting substantial barriers for learners, educators, and researchers. CollEx addresses these limitations by employing state-of-the-art Large Vision-Language Models (LVLMs) as multimodal agents accessible through an intuitive chat interface. By abstracting complex interactions via specialized agents equipped with advanced tools, CollEx facilitates curiosity-driven exploration, significantly simplifying access to diverse scientific collections and records therein. Our system integrates textual and visual modalities, supporting educational scenarios that are helpful for teachers, pupils, students, and researchers by fostering independent exploration as well as scientific excitement and curiosity. Furthermore, CollEx serves the research community by discovering interdisciplinary connections and complementing visual data. We illustrate the effectiveness of our system through a proof-of-concept application containing over 64,000 unique records across 32 collections from a local scientific collection from a public university.

</details>

---

## 321. MT2ST: Adaptive Multi-Task to Single-Task Learning

- [ ] MT2ST: Adaptive Multi-Task to Single-Task Learning | https://aclanthology.org/2025.magmar-1.8/

- **Link**: https://aclanthology.org/2025.magmar-1.8/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We proposeMT2ST, a general and efficient framework for accelerating multi-task training by progressively transitioning to single-task optimization. Unlike conventional multi-task learning (MTL) or single-task fine-tuning (STL), MT2ST dynamically adjusts the training focus via two complementary strategies:Diminish, which gradually down-weights auxiliary losses, andSwitch, which explicitly switches to the primary task at a scheduled point. We demonstrate the effectiveness of MT2ST across three key paradigms: representation learning, transformers, and diffusion models, covering both unimodal (text/image) and multimodal (vision-language) tasks. Extensive experiments show that MT2ST significantly improves training efficiency—achieving up to 56% FLOPs compression—while maintaining or surpassing task performance. These results suggest MT2ST as a general-purpose solution for scalable and adaptive multi-task training. Although this work is general-purpose, it is especially suitable for multimodal settings such as VQA or vision-language retrieval, where auxiliary pretraining (e.g., masked language modeling or contrastive learning) often diverges from final objectives. We include a VQA case study and outline its efficiency for multimodal retrieval.

</details>

---

## 322. Multimodal Retrieval-Augmented Generation: Unified Information Processing Across Text, Image, Table, and Video Modalities

- [ ] Multimodal Retrieval-Augmented Generation: Unified Information Processing Across Text, Image, Table, and Video Modalities | https://aclanthology.org/2025.magmar-1.5/

- **Link**: https://aclanthology.org/2025.magmar-1.5/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Retrieval-augmented generation (RAG) is a powerful paradigm for leveraging external data to enhance the capabilities of large language models (LLMs). However, most existing RAG solutions are tailored for single-modality or limited multimodal scenarios, restricting their applicability in real-world contexts where diverse data sources—including text, tables, images, and videos—must be integrated seamlessly. In this work proposes a unifiedMultimodal Retrieval-augmented generation (mRAG)system designed to unify information processing across all four modalities. Our pipeline ingests and indexes data from PDFs and videos using tools like Amazon Textract, Transcribe, Langfuse, and multimodal LLMs (e.g., Claude 3.5 Sonnet) for structured extraction and semantic enrichment. The dataset includes text queries, table lookups, image-based questions, and videos. Evaluation with the Deepeval framework shows improved retrieval accuracy and response quality, especially for structured text and tables. While performance on image and video queries is lower, the multimodal integration framework remains robust, underscoring the value of unified pipelines for diverse data.

</details>

---

## 323. MakingLVLMs Look Twice: Contrastive Decoding with Contrast Images

- [ ] MakingLVLMs Look Twice: Contrastive Decoding with Contrast Images | https://aclanthology.org/2025.magmar-1.6/

- **Link**: https://aclanthology.org/2025.magmar-1.6/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Vision-Language Models (LVLMs) are becoming increasingly popular for text-vision tasks requiring cross-modal reasoning, but often struggle with fine-grained visual discrimination. This limitation is evident in recent benchmarks like NaturalBench and D3, where closed models such as GPT-4o achieve only 39.6%, and open-source models perform below random chance (25%). We introduce Contrastive decoding with Contrast Images (CoCI), which adjusts LVLM outputs by contrasting them against outputs for similar images (Contrast Images - CIs). CoCI demonstrates strong performance across three distinct supervision regimes. First, when using naturally occurring CIs in benchmarks with curated image pairs, we achieve improvements of up to 98.9% on NaturalBench, 69.5% on D3, and 37.6% on MMVP. Second, for scenarios with modest training data (~5k samples), we show that a lightweight neural classifier can effectively select CIs from similar images at inference time, improving NaturalBench performance by up to 36.8%. Third, for scenarios with no training data, we develop a caption-matching technique that selects CIs by comparing LVLM-generated descriptions of candidate images. Notably, on VQAv2, our method improves VQA performance even in pointwise evaluation settings without explicit contrast images. Our approach demonstrates the potential for enhancing LVLMs at inference time through different CI selection approaches, each suited to different data availability scenarios.

</details>

---

## 324. Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models

- [ ] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models | https://aclanthology.org/2025.nlp4pi-1.7/

- **Link**: https://aclanthology.org/2025.nlp4pi-1.7/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Phishing attacks represent a significant cybersecurity threat, necessitating adaptive detection techniques. This study explores few-shot Adaptive Linguistic Prompting (ALP) in detecting phishing webpages through the multimodal capabilities of state-of-the-art large language models (LLMs) such as GPT-4o and Gemini 1.5 Pro. ALP is a structured semantic reasoning method that guides LLMs to analyze textual deception by breaking down linguistic patterns, detecting urgency cues, and identifying manipulative diction commonly found in phishing content. By integrating textual, visual, and URL-based analysis, we propose a unified model capable of identifying sophisticated phishing attempts. Our experiments demonstrate that ALP significantly enhances phishing detection accuracy by guiding LLMs through structured reasoning and contextual analysis. The findings highlight the potential of ALP-integrated multimodal LLMs to advance phishing detection frameworks, achieving an F1-score of 0.93—surpassing traditional approaches. These results establish a foundation for more robust, interpretable, and adaptive linguistic-based phishing detection systems using LLMs.

</details>

---

## 325. VisTRA: Visual Tool-use Reasoning Analyzer for Small Object Visual Question Answering

- [ ] VisTRA: Visual Tool-use Reasoning Analyzer for Small Object Visual Question Answering | https://aclanthology.org/2025.realm-1.26/

- **Link**: https://aclanthology.org/2025.realm-1.26/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This study proposes VisTRA (Visual Tool-use Reasoning Analyzer), a framework for analyzing how Visual Language Models (VLMs) utilize tools in VQA tasks involving small objects in high-resolution images. While tools like object detection and zoom functionality are essential for small object VQA, their potential errors necessitate careful verification of outputs. Our framework provides systematic evaluation of VLMs’ tool-use capabilities through analysis of verification patterns. Using the V* bench dataset, we find that direct acceptance of tool outputs correlates with decreased VQA accuracy, while lower-performing models exhibit higher frequencies of cyclic verification loops. These findings offer insights for improving tool verification mechanisms in VLM architectures focused on small object detection tasks.

</details>

---

## 326. Hidden Forms: A Dataset to Fill Masked Interfaces from Language Commands

- [ ] Hidden Forms: A Dataset to Fill Masked Interfaces from Language Commands | https://aclanthology.org/2025.realm-1.7/

- **Link**: https://aclanthology.org/2025.realm-1.7/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper introduces Hidden Forms (hFORMS), a dataset of natural language commands paired with user interfaces with masked visual context. By obscuring specific UI elements, the dataset challenges Computer-Using Agents to parse natural language instructions and infer the correct bounding box locations by leveraging UI context. Furthermore, hFORMS contains three distinct masking strategies representing progressive difficulty levels. Additionally, we explore parameter-efficient fine-tuning approaches using Vision-Language models from the Llama and Qwen series, demonstrating that fine-tuning on mobile domains results in more than 5x improvement in zero-shot domain adaptation performance when identifying bounding boxes on the desktop and web domains.

</details>

---

## 327. SciVQA2025: Overview of the First Scientific Visual Question Answering Shared Task

- [ ] SciVQA2025: Overview of the First Scientific Visual Question Answering Shared Task | https://aclanthology.org/2025.sdp-1.18/

- **Link**: https://aclanthology.org/2025.sdp-1.18/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper provides an overview of the First Scientific Visual Question Answering (SciVQA) shared task conducted as part of the Fifth Scholarly Document Processing workshop (SDP 2025). SciVQA aims to explore the capabilities of current multimodal large language models (MLLMs) in reasoning over figures from scholarly publications for question answering (QA). The main focus of the challenge is on closed-ended visual and non-visual QA pairs. We developed the novel SciVQA benchmark comprising 3,000 images of figures and a total of 21,000 QA pairs. The shared task received seven submissions, with the best performing system achieving an average F1 score of approx. 0.86 across ROUGE-1, ROUGE-L, and BertScore metrics. Participating teams explored various fine-tuning and prompting strategies, as well as augmenting the SciVQA dataset with out-of-domain data and incorporating relevant context from source publications. The findings indicate that while MLLMs demonstrate strong performance on SciVQA, they face challenges in visual reasoning and still fall behind human judgments.

</details>

---

## 328. Visual Question Answering on Scientific Charts Using Fine-Tuned Vision-Language Models

- [ ] Visual Question Answering on Scientific Charts Using Fine-Tuned Vision-Language Models | https://aclanthology.org/2025.sdp-1.19/

- **Link**: https://aclanthology.org/2025.sdp-1.19/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Scientific charts often encapsulate the core findings of research papers, making the ability to answer questions about these charts highly valuable. This paper explores recent advancements in scientific chart visual question answering (VQA) enabled by large Vision Language Models (VLMs) and newly curated datasets. As part of the SciVQA shared task from the 5th Workshop on Scholarly Document Processing, we develop and evaluate multimodal Systems capable of answering diverse question types - including multiple-choice, yes/no, unanswerable, and infinite answer set questions - based on chart images extracted from scientific literature. We investigate the effects of zero-shot and one-shot prompting, as well as supervised fine-tuning (SFT), on the performance of Qwen2.5-VL models (7B and 32B variants). We also tried to include more training data from domain-specific datasets (SpiQA and ArXivQA). Our fine-tuned Qwen2.5-VL 32B model achieves a substantial improvement over the GPT-4o-mini baseline and reaches the 4th place in the shared task, highlighting the effectiveness of domain-specific fine-tuning. We published the code for the experiments.

</details>

---

## 329. ExpertNeurons atSciVQA-2025: Retrieval AugmentedVQAwith Vision Language Model (RAVQA-VLM)

- [ ] ExpertNeurons atSciVQA-2025: Retrieval AugmentedVQAwith Vision Language Model (RAVQA-VLM) | https://aclanthology.org/2025.sdp-1.20/

- **Link**: https://aclanthology.org/2025.sdp-1.20/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

We introduce RAVQA-VLM, a novel Retrieval-Augmented Generation (RAG) architecture with Vision Language Model for the SciVQA challenge, which targets closed-ended visual and nonvisual questions over scientific figures drawn from ACL Anthology and arXiv papers (Borisova and Rehm, 2025). Our system first encodes each input figure and its accompanying metadata (caption, figure ID, type) into dense embed- dings, then retrieves context passages from the full PDF of the source paper via a Dense Passage Retriever (Karpukhin et al., 2020). The extracted contexts are concatenated with the question and passed to a vision-capable generative backbone (e.g., Phi-3.5, Pixtral-12B, Mixtral-24B-small, InterVL-3-14B) fine-tuned on the 15.1K SciVQA training examples (Yang et al., 2023; Pramanick et al., 2024). We jointly optimize retrieval and generation end-to-end to minimize answer loss and mitigate hallucinations (Lewis et al., 2020; Rujun Han and Castelli, 2024). On the SciVQA test set, RAVQA-VLM achieves significant improvements over parametric only baselines, with relative gains of +5% ROUGE1 and +5% ROUGE-L, demonstrating the efficacy of RAG for multimodal scientific QA.

</details>

---

## 330. Coling-UniAatSciVQA2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models

- [ ] Coling-UniAatSciVQA2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models | https://aclanthology.org/2025.sdp-1.21/

- **Link**: https://aclanthology.org/2025.sdp-1.21/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper describes our system for the SciVQA 2025 Shared Task on Scientific Visual Question Answering. Our system employs an ensemble of two Multimodal Large Language Models and various few-shot example retrieval strategies. The model and few-shot setting are selected based on the figure and question type. We also select answers based on the models’ confidence levels. On the blind test data, our system ranks third out of seven with an average F1 score of 85.12 across ROUGE-1, ROUGE-L, and BERTS. Our code is publicly available.

</details>

---

## 331. Instruction-tunedQwenChart for Chart Question Answering

- [ ] Instruction-tunedQwenChart for Chart Question Answering | https://aclanthology.org/2025.sdp-1.22/

- **Link**: https://aclanthology.org/2025.sdp-1.22/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Charts, where information is delivered holistically by visual and textual features, represent a challenge when it comes to downstream tasks such as chart question answering, where both kinds of information contribute to the task. The standard approach is to decouple the task in two steps, first extracting information from the charts, or representing it as a table, text or code, and then a second reasoning step to output the answers. Today, the advancements in visual encoding of Visual Large Language Models (VLLM) have shown their capabilities to solve such complex tasks without using in-between representations of the charts or massive in-domain training. Our new instruction fine-tuned and chain-of-thought model QwenChart showed that even in a complex new benchmark such as SciVQA general models can achieve great performances with low-cost training, matching the capabilities that LLMs have showed in unimodal downstream tasks. An out-of-domain evaluation showed satisfactory results, albeit with an expected drop in performance.

</details>

---

## 332. Modgenix atSemEval-2025 Task 1: Context Aware Vision Language Ranking (CAViLR) for Multimodal Idiomaticity Understanding

- [ ] Modgenix atSemEval-2025 Task 1: Context Aware Vision Language Ranking (CAViLR) for Multimodal Idiomaticity Understanding | https://aclanthology.org/2025.semeval-1.106/

- **Link**: https://aclanthology.org/2025.semeval-1.106/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents CAViLR, a hybrid multimodal approach for SemEval-2025 Task 1. Our methodintegrates CLIP as a baseline with a Mixture of Experts (MoE) framework that dynamically selectsexpert models such as Pixtral-12B and Phi-3.5 based on input context. The approach addresseschallenges in both image ranking and image sequence prediction, improving the alignment of visualand textual semantics. Experimental results demonstrate that our hybrid model outperforms individualmodels. Future work will focus on refining expert selection and enhancing disambiguation strategiesfor complex idiomatic expressions.

</details>

---

## 333. daalft atSemEval-2025 Task 1: Multi-step Zero-shot Multimodal Idiomaticity Ranking

- [ ] daalft atSemEval-2025 Task 1: Multi-step Zero-shot Multimodal Idiomaticity Ranking | https://aclanthology.org/2025.semeval-1.19/

- **Link**: https://aclanthology.org/2025.semeval-1.19/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper presents a multi-step zero-shot system for SemEval-2025 Task 1 on Advancing Multimodal Idiomaticity Representation (AdMIRe). The system employs two state-of-the-art multimodal language models, Claude Sonnet 3.5 and OpenAI GPT-4o, to determine idiomaticity and rank images for relevance in both subtasks. A hybrid approach combining o1-preview for idiomaticity classification and GPT-4o for visual ranking produced the best overall results. The system demonstrates competitive performance on the English extended dataset for Subtask A, but faces challenges in cross-lingual transfer to Portuguese. Comparing Image+Text and Text-Only approaches reveals interesting trends and raises questions about the role of visual information in multimodal idiomaticity detection.

</details>

---

## 334. Mr. Snuffleupagus atSemEval-2025 Task 4: Unlearning Factual Knowledge fromLLMs Using AdaptiveRMU

- [ ] Mr. Snuffleupagus atSemEval-2025 Task 4: Unlearning Factual Knowledge fromLLMs Using AdaptiveRMU | https://aclanthology.org/2025.semeval-1.213/

- **Link**: https://aclanthology.org/2025.semeval-1.213/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, their tendency to memorize training data raises concerns regarding privacy, copyright compliance, and security, particularly in cases involving Personally Identifiable Information (PII). Effective machine unlearning techniques are essential to mitigate these risks, yet existing methods remain underdeveloped for LLMs due to their open-ended output space. In this work, we apply the Adaptive Representation Misdirection Unlearning (RMU) technique to unlearn sensitive information from LLMs. Through extensive experiments, we analyze the effects of unlearning across different decoder layers to determine the most effective regions for sensitive information removal. Our technique ranked 4th on the official leaderboard of both 1B parameter and 7B parameter models.

</details>

---

## 335. FJWU_Squad atSemEval-2025 Task 1: An Idiom Visual Understanding Dataset for Idiom Learning

- [ ] FJWU_Squad atSemEval-2025 Task 1: An Idiom Visual Understanding Dataset for Idiom Learning | https://aclanthology.org/2025.semeval-1.231/

- **Link**: https://aclanthology.org/2025.semeval-1.231/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Idiomatic expressions pose difficulties for Natural Language Processing (NLP) because they are noncompositional. In this paper, we propose the Idiom Visual Understanding Dataset (IVUD), a multimodal dataset for idiom understanding using visual and textual representation. For SemEval-2025 Task 1 (AdMIRe), we specifically addressed dataset augmentation using AI-synthesized images and human-directed prompt engineering. We compared the efficacy of vision- and text-based models in ranking images aligned with idiomatic phrases. The results identify the advantages of using multimodal context for enhanced idiom understanding, showcasing how vision-language models perform better than text-only approaches in the detection of idiomaticity.

</details>

---

## 336. PoliTo atSemEval-2025 Task 1: Beyond Literal Meaning: A Chain-of-Though Approach for Multimodal Idiomacity Understanding

- [ ] PoliTo atSemEval-2025 Task 1: Beyond Literal Meaning: A Chain-of-Though Approach for Multimodal Idiomacity Understanding | https://aclanthology.org/2025.semeval-1.269/

- **Link**: https://aclanthology.org/2025.semeval-1.269/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Idiomatic expressions present significant challenges for natural language understanding systems as their meaning often diverge from the literal interpretation. While prior works have focused on textual idiom detection, the role of visual content in reasoning about idiomaticity remains underexplored. This study introduces a Chain-of-Thought reasoning framework that enhances idiomatic comprehension by ranking images based on their relevance to a compound expression in context, requiring the system to distinguish between idiomatic and literal meanings.We comprehensively evaluate our approach by quantitatively analyzing the performance improvements achieved integrating textual and visual information in the ranking process through different prompting settings. Our empirical findings provide insights into the capabilities of visual Large Language Models to establish meaningful correlations between idiomatic content and its visual counterpart, suggesting promising directions for multimodal language understanding.

</details>

---

## 337. UCSCNLPT6 atSemEval-2025 Task 1: LeveragingLLMs andVLMs for Idiomatic Understanding

- [ ] UCSCNLPT6 atSemEval-2025 Task 1: LeveragingLLMs andVLMs for Idiomatic Understanding | https://aclanthology.org/2025.semeval-1.274/

- **Link**: https://aclanthology.org/2025.semeval-1.274/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Idiomatic expressions pose a significant challenge for natural language models due to their non-compositional nature. In this work, we address Subtask 1 of the SemEval-2025 Task 1 (ADMIRE), which requires distinguishing between idiomatic and literal usages of phrases and identify images that align with the relevant meaning.Our approach integrates large language models (LLMs) and vision-language models, and we show how different prompting techniques improve those models’ ability to identify and explain the meaning of idiomatic language.

</details>

---

## 338. HiTZ-Ixa atSemEval-2025 Task 1: Multimodal Idiomatic Language Understanding

- [ ] HiTZ-Ixa atSemEval-2025 Task 1: Multimodal Idiomatic Language Understanding | https://aclanthology.org/2025.semeval-1.293/

- **Link**: https://aclanthology.org/2025.semeval-1.293/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we present our approach to the AdMIRe (Advancing Multimodal Idiomaticity Representation) shared task, outlining the methodologies and strategies employed to tackle the challenges of idiomatic expressions in multimodal contexts. We discuss both successful and unsuccessful approaches, including the use of models of varying sizes and experiments involving zero- and few-shot learning. Our final submission, based on a zero-shot instruction-following vision-and-language model (VLM), achieved 13th place for the English test set and 1st place for the Portuguese test set on the preliminary leaderboard.We investigate the performance of open VLMs in this task, demonstrating that both large language models (LLMs) and VLMs exhibit strong capabilities in identifying idiomatic expressions. However, we also identify significant limitations in both model types, including instability and a tendency to generate hallucinated content, which raises concerns about their reliability in interpreting figurative language. Our findings emphasize the need for further advancements in multimodal models to improve their robustness and mitigate these issues.

</details>

---

## 339. SemEval-2025 Task 1:AdMIRe - Advancing Multimodal Idiomaticity Representation

- [ ] SemEval-2025 Task 1:AdMIRe - Advancing Multimodal Idiomaticity Representation | https://aclanthology.org/2025.semeval-1.330/

- **Link**: https://aclanthology.org/2025.semeval-1.330/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Idiomatic expressions present a unique challenge in NLP, as their meanings are often notdirectly inferable from their constituent words. Despite recent advancements in Large LanguageModels (LLMs), idiomaticity remains a significant obstacle to robust semantic representation.We present datasets and tasks for SemEval-2025 Task 1: AdMiRe (Advancing Multimodal Idiomaticity Representation), which challenges the community to assess and improve models’ ability to interpret idiomatic expressions in multimodal contexts and in multiple languages. Participants competed in two subtasks: ranking images based on their alignment with idiomatic or literal meanings, and predicting the next image in a sequence. The most effective methods achieved human-level performance by leveraging pretrained LLMs and vision-language models in mixture-of-experts settings, with multiple queries used to smooth over the weaknesses in these models’ representations of idiomaticity.

</details>

---

## 340. Table Understanding and (Multimodal)LLMs: A Cross-Domain Case Study on Scientific vs. Non-Scientific Data

- [ ] Table Understanding and (Multimodal)LLMs: A Cross-Domain Case Study on Scientific vs. Non-Scientific Data | https://aclanthology.org/2025.trl-1.10/

- **Link**: https://aclanthology.org/2025.trl-1.10/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Tables are among the most widely used tools for representing structured data in research, business, medicine, and education. Although LLMs demonstrate strong performance in downstream tasks, their efficiency in processing tabular data remains underexplored. In this paper, we investigate the effectiveness of both text-based and multimodal LLMs on table understanding tasks through a cross-domain and cross-modality evaluation. Specifically, we compare their performance on tables from scientific vs. non-scientific contexts and examine their robustness on tables represented as images vs. text. Additionally, we conduct an interpretability analysis to measure context usage and input relevance. We also introduce the TableEval benchmark, comprising 3017 tables from scholarly publications, Wikipedia, and financial reports, where each table is provided in five different formats: Image, Dictionary, HTML, XML, and LaTeX. Our findings indicate that while LLMs maintain robustness across table modalities, they face significant challenges when processing scientific tables.

</details>

---

## 341. RITT: A Retrieval-Assisted Framework with Image and Text Table Representations for Table Question Answering

- [ ] RITT: A Retrieval-Assisted Framework with Image and Text Table Representations for Table Question Answering | https://aclanthology.org/2025.trl-1.8/

- **Link**: https://aclanthology.org/2025.trl-1.8/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

Tables can be represented either as text or as images. Previous works on table question answering (TQA) typically rely on only one representation, neglecting the potential benefits of combining both. In this work, we explore integrating textual and visual table representations using multi-modal large language models (MLLMs) for TQA. Specifically, we propose RITT, a retrieval-assisted framework that first identifies the most relevant part of a table for a given question, then dynamically selects the optimal table representations based on the question type. Experiments demonstrate that our framework significantly outperforms the baseline MLLMs by an average of 13 Exact Match and surpasses two text-only state-of-the-art TQA methods on four TQA benchmarks, highlighting the benefits of leveraging both textual and visual table representations.

</details>

---

## 342. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt. Generation for EnhancedLLMContent Moderation

- [ ] Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt. Generation for EnhancedLLMContent Moderation | https://aclanthology.org/2025.woah-1.44/

- **Link**: https://aclanthology.org/2025.woah-1.44/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP’s superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods across various open and closed LLMs, with attack success rates of 96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning.

</details>

---

## 343. Benchmarking Table Extraction: MultimodalLLMs vs TraditionalOCR

- [ ] Benchmarking Table Extraction: MultimodalLLMs vs TraditionalOCR | https://aclanthology.org/2025.xllm-1.2/

- **Link**: https://aclanthology.org/2025.xllm-1.2/

- **Conference**: ACL

- **Year**: 2025

<details>
<summary><strong>Abstract</strong></summary>

This paper compares two approaches for table extraction from images: deep learning computer vision and Multimodal Large Language Models (MLLMs). Computer vision models for table extraction, such as the Table Transformer model (TATR), have enhanced the extraction of complex table structural layouts by leveraging deep learning for precise structural recognition combined with traditional Optical Character Recognition (OCR). Conversely, MLLMs, which process both text and image inputs, present a novel approach by potentially bypassing the limitations of TATR plus OCR methods altogether. Models such as GPT-4o, Phi-3 Vision, and Granite Vision 3.2 demonstrate the potential of MLLMs to analyze and interpret table images directly, offering enhanced accuracy and robust extraction capabilities. A state-of-the-art metric like Grid Table Similarity (GriTS) evaluated these methodologies, providing nuanced insights into structural and text content effectiveness. Utilizing the PubTables-1M dataset, a comprehensive and widely used benchmark in the field, this study highlights the strengths and limitations of each approach, setting the stage for future innovations in table extraction technologies. Deep learning computer vision techniques still have a slight edge when extracting table structural layout, but in terms of text cell content, MLLMs are far better.

</details>

---

