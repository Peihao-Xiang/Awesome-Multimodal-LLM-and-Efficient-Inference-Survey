# IJCAI 2023 Papers

> ☐ 勾选论文后，可用脚本导出 selected_ijcai2023_papers.csv

## 1. Hierarchical Prompt Learning for Compositional Zero-Shot Recognition

- [ ] Hierarchical Prompt Learning for Compositional Zero-Shot Recognition | https://www.ijcai.org/proceedings/2023/163

- **Link**: https://www.ijcai.org/proceedings/2023/163

- **Conference**: IJCAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Compositional Zero-Shot Learning (CZSL) aims to imitate the powerful generalization ability of human beings to recognize novel compositions of known primitive concepts that correspond to a state and an object, e.g., purple apple. To fully capture the intra- and inter-class correlations between compositional concepts, in this paper, we propose to learn them in a hierarchical manner. Specifically, we set up three hierarchical embedding spaces that respectively model the states, the objects, and their compositions, which serve as three “experts” that can be combined in inference for more accurate predictions. We achieve this based on the recent success of large-scale pretrained vision-language models, e.g., CLIP, which provides a strong initial knowledge of image-text relationships. To better adapt this knowledge to CZSL, we propose to learn three hierarchical prompts by explicitly fixing the unrelated word tokens in the three embedding spaces. Despite its simplicity, our proposed method consistently yields superior performance over current state-of-the-art approaches on three widely-used CZSL benchmarks.

</details>

---

## 2. Prompt Learns Prompt: Exploring Knowledge-Aware Generative Prompt Collaboration For Video Captioning

- [ ] Prompt Learns Prompt: Exploring Knowledge-Aware Generative Prompt Collaboration For Video Captioning | https://www.ijcai.org/proceedings/2023/180

- **Link**: https://www.ijcai.org/proceedings/2023/180

- **Conference**: IJCAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Fine-tuning large vision-language models is a challenging task. Prompt tuning approaches have been introduced to learn fixed textual or visual prompts while freezing the pre-trained model in downstream tasks. Despite the effectiveness of prompt tuning, what do those learnable prompts learn remains unexplained. In this work, we explore whether prompts in the fine-tuning can learn knowledge-aware prompts from the pre-training, by designing two different sets of prompts in pre-training and fine-tuning phases respectively. Specifically, we present a Video-Language Prompt tuning (VL-Prompt) approach for video captioning, which first efficiently pre-train a video-language model to extract key information (e.g., actions and objects) with flexibly generated Knowledge-Aware Prompt (KAP). Then, we design a Video-Language Prompt (VLP) to transfer the knowledge from the knowledge-aware prompts and fine-tune the model to generate full captions. Experimental results show the superior performance of our approach over several state-of-the-art baselines. We further demonstrate that the video-language prompts are well learned from the knowledge-aware prompts.

</details>

---

## 3. Black-box Prompt Tuning for Vision-Language Model as a Service

- [ ] Black-box Prompt Tuning for Vision-Language Model as a Service | https://www.ijcai.org/proceedings/2023/187

- **Link**: https://www.ijcai.org/proceedings/2023/187

- **Conference**: IJCAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

In the scenario of Model-as-a-Service (MaaS), pre-trained models are usually released as inference APIs. Users are allowed to query those models with manually crafted prompts. Without accessing the network structure and gradient information, it's tricky to perform continuous prompt tuning on MaaS, especially for vision-language models (VLMs) considering cross-modal interaction. In this paper, we propose a black-box prompt tuning framework for VLMs to learn task-relevant prompts without back-propagation. In particular, the vision and language prompts are jointly optimized in the intrinsic parameter subspace with various evolution strategies. Different prompt variants are also explored to enhance the cross-model interaction. Experimental results show that our proposed black-box prompt tuning framework outperforms both hand-crafted prompt engineering and gradient-based prompt learning methods, which serves as evidence of its capability to train task-relevant prompts in a derivative-free manner.

</details>

---

## 4. Vision Language Navigation with Knowledge-driven Environmental Dreamer

- [ ] Vision Language Navigation with Knowledge-driven Environmental Dreamer | https://www.ijcai.org/proceedings/2023/204

- **Link**: https://www.ijcai.org/proceedings/2023/204

- **Conference**: IJCAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Vision-language navigation (VLN) requires an agent to perceive visual observation in a house scene and navigate step-by-step following natural language instruction. Due to the high cost of data annotation and data collection, current VLN datasets provide limited instruction-trajectory data samples. Learning vision-language alignment for VLN from limited data is challenging since visual observation and language instruction are both complex and diverse. Previous works only generate augmented data based on original scenes while failing to generate data samples from unseen scenes, which limits the generalization ability of the navigation agent. In this paper, we introduce the Knowledge-driven Environmental Dreamer (KED), a method that leverages the knowledge of the embodied environment and generates unseen scenes for a navigation agent to learn. Generating an unseen environment with texture consistency and structure consistency is challenging. To address this problem, we incorporate three knowledge-driven regularization objectives into the KED and adopt a reweighting mechanism for self-adaptive optimization. Our KED method is able to generate unseen embodied environments without extra annotations. We use KED to successfully generate 270 houses and 500K instruction-trajectory pairs. The navigation agent with the KED method outperforms the state-of-the-art methods on various VLN benchmarks, such as R2R, R4R, and RxR. Both qualitative and quantitative experiments prove that our proposed KED method is able to high-quality augmentation data with texture consistency and structure consistency.

</details>

---

## 5. From Association to Generation: Text-only Captioning by Unsupervised Cross-modal Mapping

- [ ] From Association to Generation: Text-only Captioning by Unsupervised Cross-modal Mapping | https://www.ijcai.org/proceedings/2023/481

- **Link**: https://www.ijcai.org/proceedings/2023/481

- **Conference**: IJCAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

With the development of Vision-Language Pre-training Models (VLPMs) represented by CLIP and ALIGN, significant breakthroughs have been achieved for association-based visual tasks such as image classification and image-text retrieval by the zero-shot capability of CLIP without fine-tuning. However, CLIP is hard to apply to generation-based tasks. This is due to the lack of decoder architecture and pre-training tasks for generation. Although previous works have created generation capacity for CLIP through additional language models, a modality gap between the CLIP representations of different modalities and the inability of CLIP to model the offset of this gap, which results in the failure of the concept to transfer across modes. To solve the problem, we try to map images/videos to the language modality and generate captions from the language modality. In this paper, we propose the K-nearest-neighbor Cross-modality Mapping (Knight), a zero-shot method from association to generation. With vision-free unsupervised training, Knight achieves state-of-the-art performance in zero-shot methods for image captioning and video captioning.

</details>

---

## 6. Core Challenges in Embodied Vision-Language Planning (Extended Abstract)

- [ ] Core Challenges in Embodied Vision-Language Planning (Extended Abstract) | https://www.ijcai.org/proceedings/2023/775

- **Link**: https://www.ijcai.org/proceedings/2023/775

- **Conference**: IJCAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in the areas of Multimodal Machine Learning and Artificial Intelligence (AI) have led to the development of challenging tasks at the intersection of Computer Vision, Natural Language Processing, and Robotics. Whereas many approaches and previous survey pursuits have characterised one or two of these dimensions, there has not been a holistic analysis at the center of all three. Moreover, even when combinations of these topics are considered, more focus is placed on describing, e.g., current architectural methods, as opposed to also illustrating high-level challenges and opportunities for the field. In this survey paper, we discuss Embodied Vision-Language Planning (EVLP) tasks, a family of prominent embodied navigation and manipulation problems that jointly leverage computer vision and natural language for interaction in physical environments. We propose a taxonomy to unify these tasks and provide an in-depth analysis and comparison of the new and current algorithmic approaches, metrics, simulators, and datasets used for EVLP tasks. Finally, we present the core challenges that we believe new EVLP works should seek to address, and we advocate for task construction that enables model generalisability and furthers real-world deployment.

</details>

---

## 7. Artificial Intelligence, Bias, and Ethics

- [ ] Artificial Intelligence, Bias, and Ethics | https://www.ijcai.org/proceedings/2023/799

- **Link**: https://www.ijcai.org/proceedings/2023/799

- **Conference**: IJCAI

- **Year**: 2023

<details>
<summary><strong>Abstract</strong></summary>

Although ChatGPT attempts to mitigate bias, when instructed to translate the gender-neutral Turkish sentences “O bir doktor. O bir hemşire” to English, the outcome is biased: “He is a doctor. She is a nurse.” In 2016, we have demonstrated that language representations trained via unsupervised learning automatically embed implicit biases documented in social cognition through the statistical regularities in language corpora. Evaluating embedding associations in language, vision, and multi-modal language-vision models reveals that large-scale sociocultural data is a source of implicit human biases regarding gender, race or ethnicity, skin color, ability, age, sexuality, religion, social class, and intersectional associations. The study of gender bias in language, vision, language-vision, and generative AI has highlighted the sexualization of women and girls in AI, while easily accessible generative AI models such as text-to-image generators amplify bias at scale. As AI increasingly automates tasks that determine life’s outcomes and opportunities, the ethics of AI bias has significant implications for human cognition, society, justice, and the future of AI. Thus, it is necessary to advance our understanding of the depth, prevalence, and complexities of bias in AI to mitigate it both in machines and society.

</details>

---

