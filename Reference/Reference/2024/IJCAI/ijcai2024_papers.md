# IJCAI 2024 Papers

> ☐ 勾选论文后，可用脚本导出 selected_ijcai2024_papers.csv

## 1. Cross-modal Generation and Alignment via Attribute-guided Prompt for Unsupervised Text-based Person Retrieval

- [ ] Cross-modal Generation and Alignment via Attribute-guided Prompt for Unsupervised Text-based Person Retrieval | https://www.ijcai.org/proceedings/2024/116

- **Link**: https://www.ijcai.org/proceedings/2024/116

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Text-based Person Search aims to retrieve a specified person using a given text query. Current methods predominantly rely on paired labeled image-text data to train the cross-modality retrieval model, necessitating laborious and time-consuming labeling. In response to this challenge, we present the Cross-modal Generation and Alignment via Attribute-guided Prompt framework (GAAP) for fully unsupervised text-based person search, utilizing only unlabeled images. Our proposed GAAP framework consists of two key parts: Attribute-guided Prompt Caption Generation and Attribute-guided Cross-modal Alignment module. The Attribute-guided Prompt Caption Generation module generates pseudo text labels by feeding the attribute prompts into a large-scale pre-trained vision-language model. These synthetic texts are then meticulously selected through a sample selection, ensuring the reliability for subsequent fine-tuning. The Attribute-guided Cross-modal Alignment module encompasses three sub-modules for feature alignment across modalities. Firstly, Cross-Modal Center Alignment (CMCA) aligns the samples with different modality centroids. Subsequently, to address ambiguity arising from local attribute similarities, an Attribute-guided Image-Text Contrastive Learning module (AITC) is proposed to facilitate the alignment of relationships among different pairs by considering local attribute similarities. Lastly, the Attribute-guided Image-Text Matching (AITM) module is introduced to mitigate noise in pseudo captions by using the image-attribute matching score to soften the hard matching labels. Empirical results showcase the effectiveness of our method across various text-based person search datasets under the fully unsupervised setting.

</details>

---

## 2. C3L: Content Correlated Vision-Language Instruction Tuning Data Generation via Contrastive Learning

- [ ] C3L: Content Correlated Vision-Language Instruction Tuning Data Generation via Contrastive Learning | https://www.ijcai.org/proceedings/2024/128

- **Link**: https://www.ijcai.org/proceedings/2024/128

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-Language Instruction Tuning (VLIT) is a critical training phase for Large Vision-Language Models (LVLMs). With the improving capabilities of open-source LVLMs, researchers have increasingly turned to generate VLIT data by using open-source LVLMs and achieved significant progress. However, such data generation approaches are bottlenecked by the following challenges: 1) Since multi-modal models tend to be influenced by prior language knowledge, directly using LVLMs to generate VLIT data would inevitably lead to low content relevance between generated data and images. 2) To improve the ability of the models to generate VLIT data, previous methods have incorporated an additional training phase to boost the generative capacity. This process hurts the generalization of the models to unseen inputs (i.e., “exposure bias” problem). In this paper, we propose a new Content Correlated VLIT data generation  via Contrastive Learning (C3L). Specifically, we design a new content relevance module which enhances the content relevance between VLIT data and images by computing Image Instruction Correspondence Scores S(I2C). Moreover, a contrastive learning module is introduced to further boost the VLIT data generation capability of the LVLMs. A large number of automatic measures  on four benchmarks show the effectiveness of our method.

</details>

---

## 3. FineFMPL: Fine-grained Feature Mining Prompt Learning for Few-Shot Class Incremental Learning

- [ ] FineFMPL: Fine-grained Feature Mining Prompt Learning for Few-Shot Class Incremental Learning | https://www.ijcai.org/proceedings/2024/144

- **Link**: https://www.ijcai.org/proceedings/2024/144

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Few-Shot Class Incremental Learning (FSCIL) aims to continually learn new classes with few training samples without forgetting already learned old classes. Existing FSCIL methods generally fix the backbone network in incremental sessions to achieve a balance between suppressing forgetting old classes and learning new classes. However, the fixed backbone network causes insufficient learning of new classes from a few samples. Benefiting from the powerful visual and textual understanding ability of Vision-Language (VL) pre-training models, we propose a Fine-grained Feature Mining Prompt Learning (FineFMPL) approach to adapt the VL model to FSCIL, which comprehensively learns and memorizes fine-grained discriminative information of emerging classes. Concretely, the visual probe prompt is firstly proposed to guide the image encoder of VL model to extract global-level coarse-grained features and object-level fine-grained features, and visual prototypes are preserved based on image patch significance, which contains the discriminative characteristics exclusive to the class. Secondly, the textual context prompt is constructed by cross-modal mapping of visual prototypes, feeding into the text encoder of VL model to memorize the class information as textual prototypes. Finally, integrating visual and textual prototypes based on fine-grained feature mining into the model improves the recognition performance of all classes in FSCIL. Extensive experiments on three benchmark datasets demonstrate that our FineFMPL achieves new state-of-the-art. The code is available at https://github.com/PKU-ICST-MIPL/FineFMPL_IJCAI2024.

</details>

---

## 4. DTS-TPT: Dual Temporal-Sync Test-time Prompt Tuning for Zero-shot Activity Recognition

- [ ] DTS-TPT: Dual Temporal-Sync Test-time Prompt Tuning for Zero-shot Activity Recognition | https://www.ijcai.org/proceedings/2024/170

- **Link**: https://www.ijcai.org/proceedings/2024/170

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Finetuning the large vision-language models on video data with a set of learnable prompts has shown promising performance on zero-shot activity recognition but still requires extra video data and expensive training costs. Inspired by recent Test-time Prompt Tuning (TPT) on the image domain, this work attempts to extend TPT to video data for zero-shot activity recognition. However, monotonous spatial augmentation and short class names cannot meet the need to capture diverse and complicated semantics of human behavior during prompt tuning. To this end, this work proposes a Dual Temporal-Sync Test-time Prompt Tuning (DTS-TPT) framework for zero-shot activity recognition. DTS-TPT tunes the learnable prompts appended to text inputs on video feature sequences of different temporal scales in multiple steps during test time. In each tuning step, we minimize the semantic consistency among the predictions from video feature sequences randomly augmented via AugMix with both original class names and the corresponding description generated through LLM. Compared with the state-of-the-art methods, the proposed method improves the zero-shot top-1 accuracy by approximately 2% ~ 5% on popular benchmarks. The code is available at https://github.com/quhongyu/DTS-TPT.

</details>

---

## 5. 3D Vision and Language Pretraining with Large-Scale Synthetic Data

- [ ] 3D Vision and Language Pretraining with Large-Scale Synthetic Data | https://www.ijcai.org/proceedings/2024/172

- **Link**: https://www.ijcai.org/proceedings/2024/172

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

3D Vision-Language Pre-training (3D-VLP)  aims to provide a pre-train model which can bridge 3D scenes with natural language, which is an important technique for embodied intelligence.  However, current 3D-VLP datasets are hindered by limited scene-level diversity and insufficient fine-grained annotations (only 1.2K scenes and 280K textual annotations in ScanScribe), primarily due to the labor-intensive of collecting and annotating 3D scenes. To overcome these obstacles, we construct SynVL3D, a comprehensive synthetic scene-text corpus with 10K indoor scenes and 1M descriptions at object, view, and room levels, which has the advantages of diverse scene data, rich textual descriptions, multi-grained 3D-text associations, and low collection cost. Utilizing the rich annotations in SynVL3D, we pre-train a simple and unified Transformer for aligning 3D and language with multi-grained pretraining tasks. Moreover, we propose a synthetic-to-real domain adaptation in downstream task fine-tuning process to address the domain shift. Through extensive experiments, we verify the effectiveness of our model design by achieving state-of-the-art performance on downstream tasks including visual grounding, dense captioning, and question answering. Codes are available at: https://github.com/idejie/3DSyn

</details>

---

## 6. CIC: A Framework for Culturally-Aware Image Captioning

- [ ] CIC: A Framework for Culturally-Aware Image Captioning | https://www.ijcai.org/proceedings/2024/180

- **Link**: https://www.ijcai.org/proceedings/2024/180

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Image Captioning generates descriptive sentences from images using Vision-Language Pre-trained models (VLPs) such as BLIP, which has improved greatly. However, current methods lack the generation of detailed descriptive captions for the cultural elements depicted in the images, such as the traditional clothing worn by people from Asian cultural groups. In this paper, we propose a new framework, Culturally-aware Image Captioning (CIC), that generates captions and describes cultural elements extracted from cultural visual elements in images representing cultures. Inspired by methods combining visual modality and Large Language Models (LLMs) through appropriate prompts, our framework (1) generates questions based on cultural categories from images,  (2) extracts cultural visual elements from Visual Question Answering (VQA) using generated questions, and (3) generates culturally-aware captions using LLMs with the prompts. Our human evaluation conducted on 45 participants from 4 different cultural groups with a high understanding of the corresponding culture shows that our proposed framework generates more culturally descriptive captions when compared to the image captioning baseline based on VLPs. Resources can be found at https://shane3606.github.io/cic.

</details>

---

## 7. Towards Dynamic-Prompting Collaboration for Source-Free Domain Adaptation

- [ ] Towards Dynamic-Prompting Collaboration for Source-Free Domain Adaptation | https://www.ijcai.org/proceedings/2024/182

- **Link**: https://www.ijcai.org/proceedings/2024/182

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In domain adaptation, challenges such as data privacy constraints can impede access to source data, catalyzing the development of source-free domain adaptation (SFDA) methods. However, current approaches heavily rely on models trained on source data, posing the risk of overfitting and suboptimal generalization.This paper introduces a dynamic prompt learning paradigm that harnesses the power of large-scale vision-language models to enhance the semantic transfer of source models. Specifically, our approach fosters robust and adaptive collaboration between the source-trained model and the vision-language model, facilitating the reliable extraction of domain-specific information from unlabeled target data, while consolidating domain-invariant knowledge. Without the need for accessing source data, our method amalgamates the strengths inherent in both traditional SFDA approaches and vision-language models, formulating a collaborative framework for addressing SFDA challenges. Extensive experiments conducted on three benchmark datasets showcase the superiority of our framework over previous SOTA methods.

</details>

---

## 8. 3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset

- [ ] 3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset | https://www.ijcai.org/proceedings/2024/189

- **Link**: https://www.ijcai.org/proceedings/2024/189

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Evaluating the performance of Multi-modal Large Language Models (MLLMs), integrating both point cloud and language, presents significant challenges. The lack of a comprehensive assessment hampers determining whether these models truly represent advancements, thereby impeding further progress in the field. Current evaluations heavily rely on classification and caption tasks, falling short in providing a thorough assessment of MLLMs. A pressing need exists for a more sophisticated evaluation method capable of thoroughly analyzing the spatial understanding and expressive capabilities of these models. To address these issues, we introduce a scalable 3D benchmark, accompanied by a large-scale instruction-tuning dataset known as 3DBench, providing an extensible platform for a comprehensive evaluation of MLLMs. Specifically, we establish the benchmark that spans a wide range of spatial and semantic scales, from object-level to scene-level, addressing both perception and planning tasks. Furthermore, we present a rigorous pipeline for automatically constructing scalable 3D instruction-tuning datasets, covering 10 diverse multi-modal tasks with more than 0.23 million QA pairs generated in total. Thorough experiments evaluating trending MLLMs, comparisons against existing datasets, and variations of training protocols demonstrate the superiority of 3DBench, offering valuable insights into current limitations and potential research directions. Codes are available at https://github.com/Inshsang/3DBench.

</details>

---

## 9. ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning

- [ ] ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning | https://www.ijcai.org/proceedings/2024/193

- **Link**: https://www.ijcai.org/proceedings/2024/193

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Human-AI interactivity is a critical aspect that reflects the usability of Multimodal Large Language Models (MLLMs). However, existing end-to-end MLLMs only allow users to interact with them through language instructions, leading to the limitation of the interactive accuracy and efficiency. In this study, we present precise referring instructions that utilize diverse reference representations such as points and boxes as referring prompts to refer to the special region. This enables MLLMs to focus on the region of interest and achieve finer-grained interaction. Based on precise referring instruction, we propose ChatSpot, a unified end-to-end MLLM that supports diverse forms of interactivity including mouse clicks, drag-and-drop, and drawing boxes, which provides a more flexible and seamless interactive experience. We also construct a multi-grained vision-language instruction-following dataset based on existing datasets and GPT-4 generating. Furthermore, we design a series of evaluation tasks to assess the effectiveness of region recognition and interaction. Experimental results showcase ChatSpot's promising performance. Project page: https://github.com/Ahnsun/ChatSpot.

</details>

---

## 10. ABM: Attention before Manipulation

- [ ] ABM: Attention before Manipulation | https://www.ijcai.org/proceedings/2024/201

- **Link**: https://www.ijcai.org/proceedings/2024/201

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Vision-language models (VLMs) show promising generalization and zero-shot capabilities, offering a potential solution to the impracticality and cost of enabling robots to comprehend diverse human instructions and scene semantics in the real world. Existing approaches most directly integrate the semantic representations from pre-trained VLMs with policy learning. However, these methods are limited to the labeled data learned, resulting in poor generalization ability to unseen instructions and objects. To address the above limitation, we propose a simple method called "Attention before Manipulation" (ABM), which fully leverages the object knowledge encoded in CLIP to extract information about the target object in the image. It constructs an Object Mask Field, serving as a better representation of the target object for the model to separate visual grounding from action prediction and acquire specific manipulation skills effectively. We train ABM for 8 RLBench tasks and 2 real-world tasks via behavior cloning. Extensive experiments show that our method significantly outperforms the baselines in the zero-shot and compositional generalization experiment settings.

</details>

---

## 11. ScreenAI: A Vision-Language Model for UI and Infographics Understanding

- [ ] ScreenAI: A Vision-Language Model for UI and Infographics Understanding | https://www.ijcai.org/proceedings/2024/339

- **Link**: https://www.ijcai.org/proceedings/2024/339

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Screen user interfaces (UIs) and infographics, sharing similar visual language and design principles, play important roles in human communication and human-machine interaction.
We introduce ScreenAI, a vision-language model that specializes in UI and infographics understanding.
Our model improves upon the PaLI architecture with the flexible patching strategy of pix2struct and is trained on a unique mixture of datasets.
At the heart of this mixture is a novel screen annotation task in which the model has to identify the type and location of UI elements.
We use these text annotations to describe screens to Large Language Models and automatically generate question-answering (QA), UI navigation, and summarization training datasets at scale.
We run ablation studies to demonstrate the impact of these design choices.
At only 5B parameters, ScreenAI achieves new state-of-the-art results
on UI- and infographics-based tasks (Multipage DocVQA, WebSRC, and MoTIF), and new best-in-class performance on others (ChartQA, DocVQA, and InfographicVQA) compared to models of similar size.
Finally, we release three new datasets: one focused on the screen annotation task and two others focused on question answering.

</details>

---

## 12. Breaking Barriers of System Heterogeneity: Straggler-Tolerant Multimodal Federated Learning via Knowledge Distillation

- [ ] Breaking Barriers of System Heterogeneity: Straggler-Tolerant Multimodal Federated Learning via Knowledge Distillation | https://www.ijcai.org/proceedings/2024/419

- **Link**: https://www.ijcai.org/proceedings/2024/419

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Internet of Things (IoT) devices possess valuable yet private multimodal data, calling for a decentralized machine learning scheme. Though several multimodal federated learning (MFL) methods have been proposed, most of them merely overlook the system heterogeneity across IoT devices, resulting in the inadaptability to real world applications. Aiming at this, we conduct theoretical analysis and exploration experiments on straggler impacts and uncover the fact that stragglers caused by system heterogeneity are fatal to MFL, resulting in catastrophic time overhead. Motivated by this, we propose a novel Multimodal Federated Learning with Accelerated Knowledge Distillation (MFL-AKD) framework, which is the first attempt to integrate knowledge distillation to combat stragglers in complex multimodal federated scenarios. Concretely, given the pretrained large-scale vision-language models deployed in the central server, we apply a fast knowledge transfer mechanism to conduct early training of local models with part of the local data. The early-trained model is then enhanced through the distillation of the pretrained large model and further trained on the remaining data. Extensive experiments on two datasets for video moment retrieval and two datasets for image-text retrieval demonstrate that our method achieves superior results with high straggler robustness.

</details>

---

## 13. Integrating Vision-Language Semantic Graphs in Multi-View Clustering

- [ ] Integrating Vision-Language Semantic Graphs in Multi-View Clustering | https://www.ijcai.org/proceedings/2024/472

- **Link**: https://www.ijcai.org/proceedings/2024/472

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In recent years, a variety of graph learning-based multi-view clustering (MVC) methods have emerged. However, these methods continue to face challenges in extracting latent features from real-world data, particularly in scenarios involving high-resolution color images and high-dimensional features. This task is notably difficult in cases where images are visually similar yet semantically diverse. To address this issue, we present a novel large-scale pre-trained model for multi-view clustering, named Integrate Vision-Language Semantic Graphs in Multi-View Clustering (IVSGMV), which harnesses the capabilities of visual-language pre-training models to enhance clustering performance and confronts issues in the unsupervised tuning of pre-trained models for multi-view data. We introduce an effective unsupervised approach for creating semantic graphs from image multi-view datasets using pre-trained encoders. Our method addresses the inherent spatial noise and imbalance in these encoders by employing graph filters and a joint process that integrates both image node and edge features. Additionally, we demonstrate the application of our approach to multi-view image clustering on extensive datasets, notably the high-resolution MVImgNet, achieving an impressive 82% accuracy. Furthermore, our method extends the zero-shot capabilities of large-scale pre-trained models, resulting in good performance in clustering tasks on untrained multi-view datasets.

</details>

---

## 14. TAI++: Text as Image for Multi-Label Image Classification by Co-Learning Transferable Prompt

- [ ] TAI++: Text as Image for Multi-Label Image Classification by Co-Learning Transferable Prompt | https://www.ijcai.org/proceedings/2024/578

- **Link**: https://www.ijcai.org/proceedings/2024/578

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The recent introduction of prompt tuning based on pre-trained vision-language models has dramatically improved the performance of multi-label image classification. However, some existing strategies that have been explored still have drawbacks, i.e., either exploiting massive labeled visual data at a high cost or using text data only for text prompt tuning and thus failing to learn the diversity of visual knowledge. Hence, the application scenarios of these methods are limited. In this paper, we propose a pseudo-visual prompt (PVP) module for implicit visual prompt tuning to address this problem. Specifically, we first learn the pseudo-visual prompt for each category, mining diverse visual knowledge by the well-aligned space of pre-trained vision-language models. Then, a co-learning strategy with a dual-adapter module is designed to transfer visual knowledge from pseudo-visual prompt to text prompt, enhancing their visual representation abilities. Experimental results on VOC2007, MS-COCO, and NUSWIDE datasets demonstrate that our method can surpass state-of-the-art (SOTA) methods across various settings for multi-label image classification tasks. The code is available at https://github.com/njustkmg/PVP.

</details>

---

## 15. SGDCL: Semantic-Guided Dynamic Correlation Learning for Explainable Autonomous Driving

- [ ] SGDCL: Semantic-Guided Dynamic Correlation Learning for Explainable Autonomous Driving | https://www.ijcai.org/proceedings/2024/66

- **Link**: https://www.ijcai.org/proceedings/2024/66

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

By learning expressive representations, deep learning (DL) has revolutionized autonomous driving (AD). Despite significant advancements, the inherent opacity of DL models engenders public distrust, impeding their widespread adoption. For explainable autonomous driving, current studies primarily concentrate on extracting features from input scenes to predict driving actions and their corresponding explanations. However, these methods underutilize semantics and correlation information within actions and explanations (collectively called categories in this work), leading to suboptimal performance. To address this issue, we propose Semantic-Guided Dynamic Correlation Learning (SGDCL), a novel approach that effectively exploits semantic richness and dynamic interactions intrinsic to categories. SGDCL employs a semantic-guided learning module to obtain category-specific representations and a dynamic correlation learning module to adaptively capture intricate correlations among categories. Additionally, we introduce an innovative loss term to leverage fine-grained co-occurrence statistics of categories for refined regularization. We extensively evaluate SGDCL on two well-established benchmarks, demonstrating its superiority over seven state-of-the-art baselines and a large vision-language model. SGDCL significantly promotes explainable autonomous driving with up to 15.3% performance improvement and interpretable attention scores, bolstering public trust in AD.

</details>

---

## 16. GRASP: A Novel Benchmark for Evaluating Language GRounding and Situated Physics Understanding in Multimodal Language Models

- [ ] GRASP: A Novel Benchmark for Evaluating Language GRounding and Situated Physics Understanding in Multimodal Language Models | https://www.ijcai.org/proceedings/2024/696

- **Link**: https://www.ijcai.org/proceedings/2024/696

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This paper presents GRASP, a novel benchmark to evaluate the language grounding and physical understanding capabilities of video-based multimodal large language models (LLMs). This evaluation is accomplished via a two-tier approach leveraging Unity simulations. The first level tests for language grounding by assessing a model's ability to relate simple textual descriptions with visual information. The second level evaluates the model's understanding of "Intuitive Physics" principles, such as object permanence and continuity. In addition to releasing the benchmark, we use it to evaluate several state-of-the-art multimodal LLMs. Our evaluation reveals significant shortcomings in the language grounding and intuitive physics capabilities of these models. Although they exhibit at least some grounding capabilities, particularly for colors and shapes, these capabilities depend heavily on the prompting strategy. At the same time, all models perform below or at the chance level of 50% in the Intuitive Physics tests, while human subjects are on average 80% correct. These identified limitations underline the importance of using benchmarks like GRASP to monitor the progress of future models in developing these competencies.

</details>

---

## 17. ScreenAgent: A Vision Language Model-driven Computer Control Agent

- [ ] ScreenAgent: A Vision Language Model-driven Computer Control Agent | https://www.ijcai.org/proceedings/2024/711

- **Link**: https://www.ijcai.org/proceedings/2024/711

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Large Language Models (LLM) can invoke a variety of tools and APIs to complete complex tasks. The computer, as the most powerful and universal tool, could potentially be controlled by a trained LLM agent. Powered by the computer, we can hopefully build a more generalized agent to assist humans in various daily digital works. In this paper, we construct an environment for a Vision Language Model (VLM) agent to interact with a real computer screen. Within this environment, the agent can observe screenshots and manipulate the Graphical User Interface (GUI) by outputting mouse and keyboard actions. We also design an automated control pipeline that includes planning, acting, and reflecting phases, guiding the agent to continuously interact with the environment and complete multi-step tasks. Additionally, we construct the ScreenAgent Dataset, which collects screenshots and action sequences when completing daily computer tasks. Finally, we train a model, ScreenAgent, which achieves comparable computer control capabilities to GPT-4V and demonstrated more precise UI positioning capabilities. Our attempts could inspire further research on building a generalist LLM agent. The code and more detailed information are at https://github.com/niuzaisheng/ScreenAgent.

</details>

---

## 18. RealDex: Towards Human-like Grasping for Robotic Dexterous Hand

- [ ] RealDex: Towards Human-like Grasping for Robotic Dexterous Hand | https://www.ijcai.org/proceedings/2024/758

- **Link**: https://www.ijcai.org/proceedings/2024/758

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

In this paper, we introduce RealDex, a pioneering dataset capturing authentic dexterous hand grasping motions infused with human behavioral patterns, enriched by multi-view and multimodal visual data. Utilizing a teleoperation system, we seamlessly synchronize human-robot hand poses in real time. This collection of human-like motions is crucial for training dexterous hands to mimic human movements more naturally and precisely. RealDex holds immense promise in advancing humanoid robot for automated perception, cognition, and manipulation in real-world scenarios. Moreover, we introduce a cutting-edge dexterous grasping motion generation framework, which aligns with human experience and enhances real-world applicability through effectively utilizing Multimodal Large Language Models. Extensive experiments have demonstrated the superior performance of our method on RealDex and other open datasets. The dataset and associated code are available at https://4dvlab.github.io/RealDex_page/.

</details>

---

## 19. Unified Physical-Digital Face Attack Detection

- [ ] Unified Physical-Digital Face Attack Detection | https://www.ijcai.org/proceedings/2024/83

- **Link**: https://www.ijcai.org/proceedings/2024/83

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Face Recognition (FR) systems can suffer from physical (i.e., print photo) and digital (i.e., DeepFake) attacks. However, previous related work rarely considers both situations at the same time. This implies the deployment of multiple models and thus more computational burden. The main reasons for this lack of an integrated model are caused by two factors: (1) The lack of a dataset including both physical and digital attacks which the same ID covers the real face and all attack types; (2) Given the large intra-class variance between these two attacks, it is difficult to learn a compact feature space to detect both attacks simultaneously. To address these issues, we collect a Unified physical-digital Attack dataset, called UniAttackData. The dataset consists of 1,800 participations of 2 and 12 physical and digital attacks, respectively, resulting in a total of 28,706 videos. Then, we propose a Unified Attack Detection framework based on Vision-Language Models (VLMs), namely UniAttackDetection, which includes three main modules: the Teacher-Student Prompts (TSP) module, focused on acquiring unified and specific knowledge respectively; the Unified Knowledge Mining (UKM) module, designed to capture a comprehensive feature space; and the Sample-Level Prompt Interaction (SLPI) module, aimed at grasping sample-level semantics. These three modules seamlessly form a robust unified attack detection framework. Extensive experiments on UniAttackData and three other datasets demonstrate the superiority of our approach for unified face attack detection. Dataset link: https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/uniattackdatacvpr2024

</details>

---

## 20. KALE: An Artwork Image Captioning System Augmented with Heterogeneous Graph

- [ ] KALE: An Artwork Image Captioning System Augmented with Heterogeneous Graph | https://www.ijcai.org/proceedings/2024/848

- **Link**: https://www.ijcai.org/proceedings/2024/848

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Exploring the narratives conveyed by fine-art paintings is a challenge in image captioning, where the goal is to generate descriptions that not only precisely represent the visual content but also offer a in-depth interpretation of the artwork's meaning. The task is particularly complex for artwork images due to their diverse interpretations and varied aesthetic principles across different artistic schools and styles. In response to this, we present KALE (Knowledge-Augmented vision-Language model for artwork Elaborations), a novel approach that enhances existing vision-language models by integrating artwork metadata as additional knowledge. KALE incorporates the metadata in two ways: firstly as direct textual input, and secondly through a multimodal heterogeneous knowledge graph. To optimize the learning of graph representations, we introduce a new cross-modal alignment loss that maximizes the similarity between the image and its corresponding metadata. Experimental results demonstrate that KALE achieves strong performance (when evaluated with CIDEr, in particular) over existing state-of-the-art work across several artwork datasets. Source code of the project is available at https://github.com/Yanbei-Jiang/Artwork-Interpretation.

</details>

---

## 21. MuChin: A Chinese Colloquial Description Benchmark for Evaluating Language Models in the Field of Music

- [ ] MuChin: A Chinese Colloquial Description Benchmark for Evaluating Language Models in the Field of Music | https://www.ijcai.org/proceedings/2024/860

- **Link**: https://www.ijcai.org/proceedings/2024/860

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

The rapidly evolving multimodal Large Language Models (LLMs) urgently require new benchmarks to uniformly evaluate their performance on understanding and textually describing music. However, due to semantic gaps between Music Information Retrieval (MIR) algorithms and human understanding, discrepancies between professionals and the public, and low precision of annotations, existing music description datasets cannot serve as benchmarks. To this end, we present MuChin, the first open-source music description benchmark in Chinese colloquial language, designed to evaluate the performance of multimodal LLMs in understanding and describing music. We established the Caichong Music Annotation Platform (CaiMAP) that employs an innovative multi-person, multi-stage assurance method, and recruited both amateurs and professionals to ensure the precision of annotations and alignment with popular semantics. Utilizing this method, we built a large-scale, private dataset with multi-dimensional, high-precision music annotations, the Caichong Music Dataset (CaiMD), and carefully selected 1,000 high-quality entries to serve as the test set for MuChin. Based on MuChin, we analyzed the discrepancies between professionals and amateurs in terms of music description, and empirically demonstrated the effectiveness of CaiMD for fine-tuning LLMs. Ultimately, we employed MuChin to evaluate existing music understanding models on their ability to provide colloquial descriptions of music.

</details>

---

## 22. Safety of Multimodal Large Language Models on Images and Text

- [ ] Safety of Multimodal Large Language Models on Images and Text | https://www.ijcai.org/proceedings/2024/901

- **Link**: https://www.ijcai.org/proceedings/2024/901

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions. The relevant papers are collected at "https://github.com/isXinLiu/Awesome-MLLM-Safety".

</details>

---

## 23. CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning

- [ ] CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning | https://www.ijcai.org/proceedings/2024/92

- **Link**: https://www.ijcai.org/proceedings/2024/92

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

Multi-modal large language models(MLLMs) have achieved remarkable progress and demonstrated powerful knowledge comprehension and reasoning abilities. However, the mastery of domain-specific knowledge, which is essential for evaluating the intelligence of MLLMs, continues to be a challenge. Current multi-modal benchmarks for domain-specific knowledge concentrate on multiple-choice questions and are predominantly available in English, which imposes limitations on the comprehensiveness of the evaluation. To this end, we introduce CMMU, a novel benchmark for multi-modal and multi-type question understanding and reasoning in Chinese. CMMU consists of 3,603 questions in 7 subjects, covering knowledge from primary to high school. The questions can be categorized into 3 types: multiple-choice, multiple-response, and fill-in-the-blank, bringing greater challenges to MLLMs. In addition, we propose an evaluation strategy called Positional Error Variance for assessing multiple-choice questions. The strategy aims to perform a quantitative analysis of position bias. We evaluate seven open-source MLLMs along with GPT4-V, Gemini-Pro, and Qwen-VL-Plus. The results demonstrate that CMMU poses a significant challenge to the recent MLLMs. The data and code are available at https://github.com/FlagOpen/CMMU.

</details>

---

## 24. Integrating LLM, VLM, and Text-to-Image Models for Enhanced Information Graphics: A Methodology for Accurate and Visually Engaging Visualizations

- [ ] Integrating LLM, VLM, and Text-to-Image Models for Enhanced Information Graphics: A Methodology for Accurate and Visually Engaging Visualizations | https://www.ijcai.org/proceedings/2024/995

- **Link**: https://www.ijcai.org/proceedings/2024/995

- **Conference**: IJCAI

- **Year**: 2024

<details>
<summary><strong>Abstract</strong></summary>

This study presents an innovative approach to the creation of information graphics, where the accuracy of content and aesthetic appeal are of paramount importance. Traditional methods often struggle to balance these two aspects, particularly in complex visualizations like phylogenetic trees. Our methodology integrates the strengths of Large Language Models (LLMs), Vision Language Models (VLMs), and advanced text-to-image models to address this challenge. Initially, an LLM plans the layout and structure, employing Mermaid—a JavaScript-based tool that uses Markdown-like scripts for diagramming—to establish a precise and structured foundation. This structured script is crucial for ensuring data accuracy in the graphical representation. Following this, text-to-image models are employed to enhance the vector graphic generated by Mermaid, adding rich visual elements and enhancing overall aesthetic appeal. The integration of text-to-image models is a key innovation, enabling the creation of graphics that are not only informative but also visually captivating. Finally, a VLM performs quality control, ensuring that the visual enhancements align with the informational accuracy. This comprehensive approach effectively combines the accuracy of structured data representation, the creative potential of text-to-image models, and the validation capabilities of VLMs. The result is a new standard in information graphic creation, suitable for diverse applications ranging from education to scientific communication, where both information integrity and visual engagement are essential.

</details>

---

