Shared Task Details

Track 2: MedArabiQ 2025

Motivation

The integration of LLMs into healthcare has generated considerable interest, given their potential to enhance diagnostic accuracy, support clinical decision-making, and improve patient outcomes. Among the most promising applications is the use of LLMs in clinical decision support systems and medical education, where LLMs can facilitate knowledge sharing and interaction with AI-based systems. Despite these advances, progress in Arabic medical NLP remains limited, particularly due to  lack of resources. This shared task is motivated by three key challenges:

Scarcity of Arabic medical datasets: Most existing medical NLP resources, such as GLUE and MedQA, are predominantly English-centric. Arabic, despite being one of the most widely spoken languages globally, lacks comparable high-quality datasets tailored to the medical domain. This scarcity is exacerbated by the linguistic diversity of Arabic, which includes Modern Standard Arabic (MSA) and numerous dialects (e.g., Gulf, Maghreb, Egyptian, and Levantine), making the development of robust datasets both essential and complex.

Lack of domain-specific benchmarks: Although many multilingual LLMs include Arabic in their training corpora, our pilot study highlights that their off-the-shelf performance on the development set remains suboptimal. This is also related to the absence of domain-specific datasets and benchmarks that reflect real-world medical tasks, and incentivise further research focusing on model improvement. Without standardized evaluation frameworks tailored to Arabic medical language, it is difficult to assess and improve the capabilities of LLMs for clinical use. Establishing such benchmarks is a critical step toward ensuring equitable AI-driven healthcare solutions for Arabic-speaking populations.

Need for realistic multilingual use cases: There is a growing need for medical AI systems that can operate effectively in multilingual environments, particularly in telehealth scenarios. Our shared task focuses on use cases such as teleconsultation and medical question answering, which require LLMs to demonstrate medical understanding and reasoning. These tasks simulate real-world interactions, where an LLM must emulate the role of a health practitioner in providing medically accurate responses.


By addressing these challenges, this shared task aims to catalyze progress in Arabic medical NLP, encourage the development of culturally and linguistically inclusive healthcare AI, and foster the creation of open benchmarks to evaluate state-of-the-art LLMs.


Data Collection and Creation

The development dataset consists of 700 samples. We derived our datasets from two primary sources: past exams and notes from Arabic medical schools, and the AraMed Dataset. We specifically selected data sources that were unlikely to have been included in prior training datasets. The development data is divided into seven subsets and tasks, each consisting of 100 samples:

Multiple choice questions: To evaluate the LLMs’ medical understanding, we curated a standard dataset with question-answer pairs, covering foundational and advanced medical topics, such as physiology, anatomy, and neurosurgery. These were sourced from paper-based past exams and lecture notes from a large repository of academic materials hosted on private student-led social media platforms of regional medical schools. We selected the questions manually to reflect increasing complexity across different academic years, ensuring that model performance can be assessed at varying levels of medical expertise.

Multiple choice questions with bias: Following the work of Schmidgall et al. (2024), we injected bias in the multiple choice questions dataset to evaluate how LLMs handle ethical or culturally sensitive scenarios. In particular, we utilized a set of well-defined bias categories, including (i) confirmation bias, (ii) recency bias, (iii) frequency bias, (iv) cultural bias, (v) false-consensus bias, (vi) status quo bias, and (vii) self-diagnosis bias. By manually injecting the bias, we ensured relevance to the unique linguistic and clinical challenges of Arabic healthcare contexts.

Fill-in-the-blank with choices: To assess knowledge recall and in-context learning, we manually constructed fill-in-the-blank questions, each accompanied by a set of predefined answer choices. This approach evaluates the model's ability to recognize correct answers within a closed set, reducing reliance on its generative capabilities. 

Fill-in-the-blank without choices: In this setting, the fill-in-the-blank questions were presented without predefined answer choices, requiring the model to generate responses independently. This evaluation measures the model’s ability to recall and generate accurate medical knowledge without additional information, emphasizing its reasoning and language generation capabilities. 

Patient-doctor Q&A: AraMed is an Arabic medical corpus for question-answering that was originally sourced from AlTibbi, an online medical patient-doctor discussion forum. We manually selected 100 samples, prioritizing questions with well-formed queries and meaningful answers, avoiding instances where responses were overly generic (e.g., “Consult a doctor” or “See a specialist'”). Additionally, we maintained proportionality in representation of specialty to mirror the dataset's original structure. We also incorporated information about the patient, specifically age and gender, when known, into the query.

Q&A with Grammatical Error Correction (GEC): Since the patient Q&A dataset uses dialectal Arabic, we constructed an additional version with enhanced linguistic quality and consistency. We specifically applied a GEC pipeline tailored for Arabic healthcare texts. 

Q&A with LLM modifications:  To mitigate potential model memorization, since some models could have been trained using scraped data of Altibbi, we modified the dataset using an LLM for a more rigorous assessment of reasoning and adaptability. In particular, we used GPT-4o to paraphrase our original questions using the prompt, “You are a helpful assistant that paraphrases text while keeping its meaning intact.” This approach ensured that the core medical concepts remained unchanged while reducing the likelihood of models relying on memorized content.


Task Description

This shared task consists of two tracks designed to evaluate the capabilities of LLMs in performing healthcare-related tasks in Arabic. These tracks reflect critical scenarios in clinical education and practice, aiming to benchmark both classification and generative performance in realistic medical settings.


Sub-task 1: Multiple-choice question & answering (classification)

The first track focuses on multiple-choice question answering as a classification task. Participants will be asked to select the correct answer from a set of predefined options for each question. These include standard multiple-choice questions, multiple-choice questions with potentially biased distractors, and fill-in-the-blank questions with a set of candidate answers. The objective is to assess the model’s ability to apply clinical knowledge in structured decision-making scenarios. Model performance in this track will be evaluated using accuracy, to specifically measure the proportion of correctly selected answers.


Sub-task 2: Fill-in-the-blank questions (generation)

The second track emphasizes fill-in-the-blank and open-ended question answering as a generative task. Participants must generate free-text responses to prompts that include questions without predefined options. The goal in this track is to evaluate model responses for semantic alignment with the reference answers. Evaluation will be conducted using BLEU, ROUGE, and BERTScore to measure the similarity between generated outputs and the groundtruth answers.


Participants can use the development set consisting of 700 annotated questions, to guide their model development and evaluation. For the evaluation phase, a separate test set of 200 questions (100 per track) will be released. Submissions will be managed and assessed via the CodaBench platform. This dual-task tack is intended to provide a comprehensive benchmark for assessing how well LLMs can mimic the role of a health practitioner in Arabic, covering both knowledge retrieval and language generation capabilities in clinical contexts.



# evalaution metric

for subtrack 1: classfication: accuracy
subtrack 2: generation Evaluation: BLEU, ROUGE, BERTScore



# Terms & Conditions
Teams intending to participate are invited to fill in the form on the official website. The full development set is available for download here. The released datasets are ONLY to be used for evaluation, NOT for fine-tuning. Participants are free to use external resources (be creative!) to enhance model performance.


# results on the final blind test 



| # | Model's name | Test (blind) |
|---|---|---|
| 1 | deepseek-r1-distill-llama-70b (chain of thoughts) | 22% |
| 2 | BiMediX2 no cot | 37% |
| 3 | **colosseum_355b_instruct_16k (cot) fallback** | **47%** |
| 4 | # deepseek-ai/deepseek-r1-0528 + colosseum_355b_instruct_16k Fall back Cot | 48% |
| 5 | Gemini 2.5 pro | 37% |
| 6 | BioMistral | 3%7 |
| 7 | OpenBioLLM 8B (after-presossing) | 49% |
| 8 | Gemini 2.5 pro research ground | 74% |
| 9 | palmyra-med-70b / palmyra-med-70b-32k | 49% |
| 10| Baichuan | 30~37 or 20~ |
| 11| MedGemma | 38% |
