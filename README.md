# MedLingua: Zero- and Few-Shot Prompting for Arabic Medical QA

<p align="center">
<!-- User: Replace with your project's banner image -->
<img src="https://placehold.co/800x200/dbeafe/3b82f6?text=MedLingua+Project" alt="MedLingua Project Banner">
</p>

#### [Fatimah Mohamed Emad Elden](https://scholar.google.com/citations?user=CfX6eA8AAAAJ&hl=ar), [Mumina Abukar](https://scholar.google.com/citations?user=YOUR_USER_ID)

#### **Cairo University & The University of South Wales**

[![Paper](https://img.shields.io/badge/arXiv-25XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/25XX.XXXXX) 
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/astral-fate/arahealthqa-2025)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://github.com/astral-fate/arahealthqa-2025/blob/main/LICENSE)

---

## 🏆 Overview

This repository contains the code and resources for the **MedLingua** team's submission to the **MedArabiQ2025 Shared Task (Track 2, Sub-Task 1: Multiple Choice Question Answering)**. Our approach centered on evaluating the zero-shot and few-shot capabilities of various Large Language Models (LLMs) on Arabic medical questions, as fine-tuning was not permitted.

We systematically tested a range of models, from general-purpose state-of-the-art LLMs like Google's **Gemini 2.5 Pro** to specialized medical models such as BiMediX2 and MedGemma. Our findings reveal that advanced, general-domain models significantly outperform specialized medical LLMs that are not optimized for Arabic. Our best-performing system, using Gemini 2.5 Pro, achieved an **accuracy of 74% on the blind test set**, securing the **third-highest accuracy** in the competition.

---

## 🔑 Key Findings

Our key contributions are as follows:

* **State-of-the-Art Prompting:** We demonstrate that modern general-purpose LLMs can achieve high performance on Arabic medical QA using only zero-shot and few-shot prompting techniques, without any task-specific fine-tuning.
* **Model Performance Gap:** Our core finding is the pronounced performance gap between large, multilingual general-purpose models (like Gemini 2.5 Pro) and available specialized medical LLMs. The former's superior understanding of Arabic proved to be a decisive advantage.
* **Robust System Architecture:** We developed a comprehensive pipeline featuring dynamic prompt engineering (including Chain-of-Thought) and a hierarchical, rule-based response parser to ensure accurate extraction of answers.
* **Competitive Results:** Our system ranked **3rd** in the competition, highlighting the effectiveness of leveraging the in-context learning abilities of modern LLMs in a zero-resource fine-tuning scenario.

---

## ⚙️ Methodology & System Architecture

Our methodology was built entirely on in-context learning. The process follows five key stages: Input Data, Prompt Engineering, LLM Inference, Hierarchical Response Parsing, and Final Output. A key component of our strategy was the use of **Chain-of-Thought (CoT)** prompting, where we instructed models to perform step-by-step reasoning before providing a final answer.

The high-level architecture of our system is illustrated below:
<p align="center">
  <img src="https://github.com/user-attachments/assets/e3ec85bc-fd65-451d-9849-beb888ef9db2" alt="System Architecture Diagram" width="800"/>
 
  <em>A high-level overview of our data processing pipeline, from input data to the final parsed output.</em>
</p>

---

## 📊 Results

Our experiments consistently showed that large, general-purpose LLMs outperformed specialized medical models. Our final submission, using **Gemini 2.5 Pro**, achieved **75% accuracy** on the blind test set.

### Blind Test Set Performance

| Model                           | Test Accuracy |
|:--------------------------------|:-------------:|
| **Gemini 2.5 Pro** |    **754** |
| colosseum 355b                  |      51%      |
| palmyra-med-70b-32k             |      49%      |
| OpenBioLLM 8B                   |      49%      |
| qwen3-235b-a22b                 |      41%      |
| MedGemma                        |      38%      |
| BioMistral                      |      37%      |
| BiMediX2                        |      37%      |
| deepseek-r1-distill-llama-70b   |      22%      |

### Development Set Performance (Top Models)

| Model's Name         | Fill in the Blank | Multiple Choice | Multiple Choice w/ Bias |
|:---------------------|:-----------------:|:---------------:|:-----------------------:|
| qwen/qwen2-32b       |        83%        |       69%       |           67%           |
| google/medgemma-27b-it |        76%        |       55%       |           46%           |
| colosseum 355b       |        75%        |       50%       |           45%           |

---

## 📜 License & Citation

This project is released under the MIT License. For more details, please refer to the `LICENSE` file.

⚠️ **Warning!** This work is intended for research purposes. It is not ready for clinical or commercial use. The outputs should not be used for medical diagnoses or treatment decisions without verification by a qualified healthcare professional.

If you use our work in your research, please cite our paper:

```bibtex
@inproceedings{elden2025medlingua,
      title={{MedLingua at MedArabiQ2025: Zero- and Few-Shot Prompting of Large Language Models for Arabic Medical QA}},
      author={Elden, Fatimah Mohamed Emad and Abukar, Mumina},
      year={2025},
      booktitle={Proceedings of the MedArabiQ2025 Shared Task},
      eprint={25XX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
