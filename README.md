# AraHealthQA-2025

We are excited to invite researchers to participate in AraHealthQA 2025: Arabic Health Question Answering Shared Task! 

General Arabic Health QA (Track 2) is based on our recent work MedArabiQ:  https://www.arxiv.org/abs/2505.03427

There are two sub-tasks:
❓Multiple Choice Question Answering (Classification): https://www.codabench.org/competitions/8967/
❓Open-Ended Question Answering (Generation):   https://www.codabench.org/competitions/8740/

By developing robust systems for Arabic medical content, our work aims to support the advancement of accessible high-quality healthcare information for Arabic-speaking populations.

Stay tuned for the results and insights. We are proud to contribute to the growing field of Clinical NLP in Arabic, and excited about the collaboration and innovation this shared task will bring.

________________

📢 Exciting Opportunity ! 📢

We're thrilled to announce the AraHealthQA 2025 Shared Task, a new initiative designed to bridge the gap in Arabic medical question answering for Large Language Models (LLMs)! This is part of the الوسومات (هاشتاغ)#ArabicNLP2025, which is co-located with الوسومات (هاشتاغ)#EMNLP2025 in Suzhou, China.

Despite the immense potential of LLMs in healthcare, their effectiveness in the Arabic medical domain remains largely underexplored due to a lack of high-quality, specialized datasets and benchmarking efforts. AraHealthQA 2025 aims to address this by providing curated datasets and a structured evaluation framework for LLMs in Arabic medical applications.

Our task focuses on two important domains:
- General Health QA (MedArabiQ)
- Mental Health QA (MentalQA)

Learn more and participate here: 👉 https://sites.google.com/nyu.edu/arahealthqa-2025


__________

# AraHealthQA 2025 Shared task - Track 1 (Sub-task 1)

Track 1: MentalQA 2025
This track includes three sub-tasks hosted separately on Codabench. Please register individually for each sub-task you wish to participate in:

Sub-Task 1: Question Categorization
Classify questions into one or more categories like diagnosis, treatment, etc.(current page)

Sub-Task 2: Answer Categorization
Classify responses based on answer strategy (e.g., emotional support, information).

Sub-Task 3: Question Answering
Build models to generate mental health answers based on the question and context.

Motivation

The motivation for this shared task is rooted in the growing global recognition of mental health's importance and the escalating demand for accessible and reliable support systems. This need is particularly pronounced within the Arabic NLP landscape, where specific linguistic and cultural nuances present unique challenges and opportunities. Mental health resources often lack representation in languages other than English, and while efforts exist for other languages, Arabic remains understudied in this domain.

In particular, the motivations for the shared task are twofold: Social and NLP-specific.

Socially, there is an urgent and critical need for mental health support in the Arab world. While mental health concerns are pressing globally, Arabic-speaking communities face a pronounced shortage of culturally and linguistically tailored resources. This scarcity severely restricts access to essential information and support. By facilitating the creation of Arabic mental health resources, this shared task aims to bridge this critical gap. Specifically, the task promotes the development of effective, accurate, and culturally appropriate NLP tools for mental health. Such tools can empower individuals by reducing stigma, encouraging them to seek help, and enabling informed decisions regarding their well-being. Ultimately, these resources can significantly enhance mental health outcomes and generate meaningful positive social impact within Arabic-speaking communities.

From an NLP perspective, the shared task also aims to stimulate innovation and drive progress within the Arabic NLP field. By generating new resources and modeling opportunities, this task will encourage advancements across several NLP domains. Mental health presents unique linguistic and semantic challenges, thereby fostering developments in information retrieval (finding relevant and reliable information), semantic understanding (accurately interpreting meaning and intent), and answer generation (producing precise, informative, and culturally appropriate responses). By emphasizing accuracy, reliability, and cultural sensitivity, this shared task will contribute to the creation of more sophisticated and robust Arabic NLP systems, significantly advancing the field.

Dataset

This shared task leverages the MentalQA dataset, a newly constructed Arabic Question Answering dataset specifically designed for the mental health domain. MentalQA comprises a total of 500 questions and answers (Q&A) posts, including both question types and answer strategies. For the purpose of the shared task, the dataset will be divided into three subsets: 300 samples for training, 50 samples for development (Dev), and 150 samples for testing. The training set can be used to fine-tune large language models (LLMs) or serve as a base for few-shot learning approaches. The development set is intended for tuning model hyperparameters and evaluating performance, while the test set will be used for final evaluation to ensure fair benchmarking of models across participants.

The question categories include (Q):

(A) Diagnosis (questions about interpreting clinical findings)

(B) Treatment (questions about seeking treatments)

(C) Anatomy and Physiology (questions about basic medical knowledge)

(D) Epidemiology (questions about the course, prognosis, and etiology of diseases)

(E) Healthy Lifestyle (questions related to diet, exercise, and mood control)

(F) Provider Choices (questions seeking recommendations for medical professionals and facilities).

(Z). Other. Questions that do not fall under the above-mentioned categories.

The answer strategies include (A):

(1) Information (answers providing information, resources, etc.)

(2) Direct Guidance (answers providing suggestions, instructions, or advice)

(3) Emotional Support (answers providing approval, reassurance, or other forms of emotional support).

Tasks: The first two sub-tasks involve multi-label classification, while the third is a text generation task.

Sub-Task-1: Question Categorization: The first task focuses on classifying patient questions into one or more specific types (e.g., diagnosis, treatment, anatomy, etc.) to facilitate a deeper understanding of the user’s intent. This categorization enables the development of intelligent question-answering systems tailored to address diverse patient needs.

Sub-Task-2: Answer Categorization: The second task centers on classifying answers according to their strategies (e.g., informational, direct guidance, emotional support), ensuring the extraction of relevant and contextually appropriate information.

Sub-Task-3: Question Answering : The third task involves developing a question-answering model that leverages the classifications from the previous tasks. This task forms the basis for a robust question-answering system capable of providing specialized responses to a wide range of mental health-related inquiries.

Evaluation:

For Sub-Task 1 and Sub-Task 2 — both formulated as multi-label classification tasks — we will use weighted F1 and the Jaccard score.

For Sub-Task 3, we will evaluate system output using BERTScore

https://www.codabench.org/competitions/8559/

# AraHealthQA 2025 Shared task - Track 1 (sub-task 2)

Track 1: MentalQA 2025
This track includes three sub-tasks hosted separately on Codabench. Please register individually for each sub-task you wish to participate in:

Sub-Task 1: Question Categorization
Classify questions into one or more categories like diagnosis, treatment, etc.

Sub-Task 2: Answer Categorization
Classify responses based on answer strategy (e.g., emotional support, information).(current page)

Sub-Task 3: Question Answering
Build models to generate mental health answers based on the question and context.

Motivation

The motivation for this shared task is rooted in the growing global recognition of mental health's importance and the escalating demand for accessible and reliable support systems. This need is particularly pronounced within the Arabic NLP landscape, where specific linguistic and cultural nuances present unique challenges and opportunities. Mental health resources often lack representation in languages other than English, and while efforts exist for other languages, Arabic remains understudied in this domain.

In particular, the motivations for the shared task are twofold: Social and NLP-specific.

Socially, there is an urgent and critical need for mental health support in the Arab world. While mental health concerns are pressing globally, Arabic-speaking communities face a pronounced shortage of culturally and linguistically tailored resources. This scarcity severely restricts access to essential information and support. By facilitating the creation of Arabic mental health resources, this shared task aims to bridge this critical gap. Specifically, the task promotes the development of effective, accurate, and culturally appropriate NLP tools for mental health. Such tools can empower individuals by reducing stigma, encouraging them to seek help, and enabling informed decisions regarding their well-being. Ultimately, these resources can significantly enhance mental health outcomes and generate meaningful positive social impact within Arabic-speaking communities.

From an NLP perspective, the shared task also aims to stimulate innovation and drive progress within the Arabic NLP field. By generating new resources and modeling opportunities, this task will encourage advancements across several NLP domains. Mental health presents unique linguistic and semantic challenges, thereby fostering developments in information retrieval (finding relevant and reliable information), semantic understanding (accurately interpreting meaning and intent), and answer generation (producing precise, informative, and culturally appropriate responses). By emphasizing accuracy, reliability, and cultural sensitivity, this shared task will contribute to the creation of more sophisticated and robust Arabic NLP systems, significantly advancing the field.

Dataset

This shared task leverages the MentalQA dataset, a newly constructed Arabic Question Answering dataset specifically designed for the mental health domain. MentalQA comprises a total of 500 questions and answers (Q&A) posts, including both question types and answer strategies. For the purpose of the shared task, the dataset will be divided into three subsets: 300 samples for training, 50 samples for development (Dev), and 150 samples for testing. The training set can be used to fine-tune large language models (LLMs) or serve as a base for few-shot learning approaches. The development set is intended for tuning model hyperparameters and evaluating performance, while the test set will be used for final evaluation to ensure fair benchmarking of models across participants.

The question categories include (Q):

(A) Diagnosis (questions about interpreting clinical findings)

(B) Treatment (questions about seeking treatments)

(C) Anatomy and Physiology (questions about basic medical knowledge)

(D) Epidemiology (questions about the course, prognosis, and etiology of diseases)

(E) Healthy Lifestyle (questions related to diet, exercise, and mood control)

(F) Provider Choices (questions seeking recommendations for medical professionals and facilities).

(Z). Other. Questions that do not fall under the above-mentioned categories.

The answer strategies include (A):

(1) Information (answers providing information, resources, etc.)

(2) Direct Guidance (answers providing suggestions, instructions, or advice)

(3) Emotional Support (answers providing approval, reassurance, or other forms of emotional support).

Tasks: The first two sub-tasks involve multi-label classification, while the third is a text generation task.

Sub-Task-1: Question Categorization: The first task focuses on classifying patient questions into one or more specific types (e.g., diagnosis, treatment, anatomy, etc.) to facilitate a deeper understanding of the user’s intent. This categorization enables the development of intelligent question-answering systems tailored to address diverse patient needs.

Sub-Task-2: Answer Categorization: The second task centers on classifying answers according to their strategies (e.g., informational, direct guidance, emotional support), ensuring the extraction of relevant and contextually appropriate information.

Sub-Task-3: Question Answering : The third task involves developing a question-answering model that leverages the classifications from the previous tasks. This task forms the basis for a robust question-answering system capable of providing specialized responses to a wide range of mental health-related inquiries.

Evaluation:

For Sub-Task 1 and Sub-Task 2 — both formulated as multi-label classification tasks — we will use weighted F1 and the Jaccard score.

For Sub-Task 3, we will evaluate system output using BERTScore.


https://www.codabench.org/competitions/8730/

#  MentalQA 2025: AraHealthQA 2025 Shared task - Track 1 (sub-task 3) 

Track 1: MentalQA 2025
This track includes three sub-tasks hosted separately on Codabench. Please register individually for each sub-task you wish to participate in:

Sub-Task 1: Question Categorization
Classify questions into one or more categories like diagnosis, treatment, etc.

Sub-Task 2: Answer Categorization
Classify responses based on answer strategy (e.g., emotional support, information).

Sub-Task 3: Question Answering
Build models to generate mental health answers based on the question and context.(current page)

Motivation

The motivation for this shared task is rooted in the growing global recognition of mental health's importance and the escalating demand for accessible and reliable support systems. This need is particularly pronounced within the Arabic NLP landscape, where specific linguistic and cultural nuances present unique challenges and opportunities. Mental health resources often lack representation in languages other than English, and while efforts exist for other languages, Arabic remains understudied in this domain.

In particular, the motivations for the shared task are twofold: Social and NLP-specific.

Socially, there is an urgent and critical need for mental health support in the Arab world. While mental health concerns are pressing globally, Arabic-speaking communities face a pronounced shortage of culturally and linguistically tailored resources. This scarcity severely restricts access to essential information and support. By facilitating the creation of Arabic mental health resources, this shared task aims to bridge this critical gap. Specifically, the task promotes the development of effective, accurate, and culturally appropriate NLP tools for mental health. Such tools can empower individuals by reducing stigma, encouraging them to seek help, and enabling informed decisions regarding their well-being. Ultimately, these resources can significantly enhance mental health outcomes and generate meaningful positive social impact within Arabic-speaking communities.

From an NLP perspective, the shared task also aims to stimulate innovation and drive progress within the Arabic NLP field. By generating new resources and modeling opportunities, this task will encourage advancements across several NLP domains. Mental health presents unique linguistic and semantic challenges, thereby fostering developments in information retrieval (finding relevant and reliable information), semantic understanding (accurately interpreting meaning and intent), and answer generation (producing precise, informative, and culturally appropriate responses). By emphasizing accuracy, reliability, and cultural sensitivity, this shared task will contribute to the creation of more sophisticated and robust Arabic NLP systems, significantly advancing the field.

Dataset

This shared task leverages the MentalQA dataset, a newly constructed Arabic Question Answering dataset specifically designed for the mental health domain. MentalQA comprises a total of 500 questions and answers (Q&A) posts, including both question types and answer strategies. For the purpose of the shared task, the dataset will be divided into three subsets: 300 samples for training, 50 samples for development (Dev), and 150 samples for testing. The training set can be used to fine-tune large language models (LLMs) or serve as a base for few-shot learning approaches. The development set is intended for tuning model hyperparameters and evaluating performance, while the test set will be used for final evaluation to ensure fair benchmarking of models across participants.

The question categories include (Q):

(A) Diagnosis (questions about interpreting clinical findings)

(B) Treatment (questions about seeking treatments)

(C) Anatomy and Physiology (questions about basic medical knowledge)

(D) Epidemiology (questions about the course, prognosis, and etiology of diseases)

(E) Healthy Lifestyle (questions related to diet, exercise, and mood control)

(F) Provider Choices (questions seeking recommendations for medical professionals and facilities).

(Z). Other. Questions that do not fall under the above-mentioned categories.

The answer strategies include (A):

(1) Information (answers providing information, resources, etc.)

(2) Direct Guidance (answers providing suggestions, instructions, or advice)

(3) Emotional Support (answers providing approval, reassurance, or other forms of emotional support).

Tasks: The first two sub-tasks involve multi-label classification, while the third is a text generation task.

Sub-Task-1: Question Categorization: The first task focuses on classifying patient questions into one or more specific types (e.g., diagnosis, treatment, anatomy, etc.) to facilitate a deeper understanding of the user’s intent. This categorization enables the development of intelligent question-answering systems tailored to address diverse patient needs.

Sub-Task-2: Answer Categorization: The second task centers on classifying answers according to their strategies (e.g., informational, direct guidance, emotional support), ensuring the extraction of relevant and contextually appropriate information.

Sub-Task-3: Question Answering : The third task involves developing a question-answering model that leverages the classifications from the previous tasks. This task forms the basis for a robust question-answering system capable of providing specialized responses to a wide range of mental health-related inquiries.

Evaluation:

For Sub-Task 1 and Sub-Task 2 — both formulated as multi-label classification tasks — we will use weighted F1 and the Jaccard score.

For Sub-Task 3, we will evaluate system output using BERTScore.

https://www.codabench.org/competitions/8649/



## Multi-label question categorization: 

Dataset: https://github.com/hasanhuz/MentalQA/blob/main/sub_task1.zip



## Multi-label answer categorization:

Dataset: https://github.com/hasanhuz/MentalQA/blob/main/sub_task2.zip



## Patient-doctor question answering:

Dataset: https://github.com/hasanhuz/MentalQA/blob/main/sub_task3.zip

_________________________________

# AraHealthQA 2025 Shared task - Track 2 (Sub-task 1)


Track 2: MedArabiQ2025
This track includes two sub-tasks hosted separately on Codabench. Please register individually for each sub-task you wish to participate in:

Sub-Task 1: Multiple choice question & answering (classification) [current page]: Develop a model that can choose the correct answers from fixed sets of choices to Arabic medical questions.

Sub-Task 2: Open-ended question answering (generation): Develop a model that can correctly answer medical fill-in-the-blank or open-ended real-world clinical questions.

Motivation
LLMs hold great promise for healthcare by enhancing diagnostics, clinical decision support, and medical education, but progress in Arabic medical NLP remains limited due to resource scarcity. This shared task is motivated by three key challenges:

Scarcity of Arabic medical datasets: Arabic medical NLP lacks high-quality datasets, unlike English-centric resources like GLUE and MedQA, due to both limited availability and the linguistic complexity of Arabic's many dialects.

Lack of domain-specific benchmarks: Despite Arabic being included in many multilingual LLMs’ training data, our pilot study shows their off-the-shelf performance on medical tasks is suboptimal, largely due to the lack of domain-specific datasets and evaluation benchmarks—highlighting the urgent need for standardized frameworks to advance equitable AI healthcare for Arabic speakers.

Need for realistic multilingual use cases: There is an increasing demand for medical AI systems capable of functioning in multilingual settings, especially in telehealth, where tasks like teleconsultation and medical question answering require LLMs to reason and respond with clinical accuracy, simulating real practitioner interactions.

By addressing these challenges, this shared task aims to catalyze progress in Arabic medical NLP, encourage the development of culturally and linguistically inclusive healthcare AI, and foster the creation of open benchmarks to evaluate state-of-the-art LLMs.

Sub-Task 1: Multiple Choice Question Answering
Data Collection and Creation
The development dataset for this sub-task consists of 300 samples. We derived our datasets from past exams and notes from Arabic medical schools. We specifically selected data sources that were unlikely to have been included in prior training datasets. The development data is divided into three subsets and tasks, each consisting of 100 samples:

Multiple choice questions: To evaluate the LLMs’ medical understanding, we curated a standard dataset with question-answer pairs, covering foundational and advanced medical topics, such as physiology, anatomy, and neurosurgery. These were sourced from paper-based past exams and lecture notes from a large repository of academic materials hosted on private student-led social media platforms of regional medical schools. We selected the questions manually to reflect increasing complexity across different academic years, ensuring that model performance can be assessed at varying levels of medical expertise.

Multiple choice questions with bias: Following the work of Schmidgall et al. (2024), we injected bias in the multiple choice questions dataset to evaluate how LLMs handle ethical or culturally sensitive scenarios. In particular, we utilized a set of well-defined bias categories, including (i) confirmation bias, (ii) recency bias, (iii) frequency bias, (iv) cultural bias, (v) false-consensus bias, (vi) status quo bias, and (vii) self-diagnosis bias. By manually injecting the bias, we ensured relevance to the unique linguistic and clinical challenges of Arabic healthcare contexts.

Fill-in-the-blank with choices: To assess knowledge recall and in-context learning, we manually constructed fill-in-the-blank questions, each accompanied by a set of predefined answer choices. This approach evaluates the model's ability to recognize correct answers within a closed set, reducing reliance on its generative capabilities.

Task Description
This sub-task focuses on multiple-choice question answering as a classification task. Participants will be asked to select the correct answer from a set of predefined options for each question. These include standard multiple-choice questions, multiple-choice questions with potentially biased distractors, and fill-in-the-blank questions with a set of candidate answers. The objective is to assess the model’s ability to apply clinical knowledge in structured decision-making scenarios. Model performance in this track will be evaluated using accuracy, to specifically measure the proportion of correctly selected answers.

Participants can use the development set consisting of 300 annotated questions, to guide their model development and evaluation. For the evaluation phase, a separate test set of 100 questions will be released.

Alongside the other subtask, this track is intended to provide a comprehensive benchmark for assessing how well LLMs can mimic the role of a health practitioner in Arabic, covering both knowledge retrieval and language generation capabilities in clinical contexts.

https://www.codabench.org/competitions/8967/



## Multiple choice questions:

Dataset: https://github.com/nyuad-cai/MedArabiQ/blob/main/datasets/multiple-choice-questions.csv


## Multiple choice questions with bias

Dataset: https://github.com/nyuad-cai/MedArabiQ/blob/main/datasets/multiple-choice-withbias.csv



## Fill-in-the-blank with choices: 

Dataset: https://github.com/nyuad-cai/MedArabiQ/blob/main/datasets/fill-in-the-blank-choices.csv
______________________

# AraHealthQA 2025 Shared task - Track 2 (Sub-task 2)

Track 2: MedArabiQ2025
This track includes two sub-tasks hosted separately on Codabench. Please register individually for each sub-task you wish to participate in:

Sub-Task 1: Multiple choice question answering (classification)
Develop a model that can choose the correct answers from fixed sets of choices to Arabic medical questions.

Sub-Task 2: Open-ended question answering (generation) [current page]
Develop a model that can correctly answer medical fill-in-the-blank or open-ended real-world clinical questions.

Motivation
LLMs hold great promise for healthcare by enhancing diagnostics, clinical decision support, and medical education, but progress in Arabic medical NLP remains limited due to resource scarcity. This shared task is motivated by three key challenges:

Scarcity of Arabic medical datasets: Arabic medical NLP lacks high-quality datasets, unlike English-centric resources like GLUE and MedQA, due to both limited availability and the linguistic complexity of Arabic's many dialects.

Lack of domain-specific benchmarks: Despite Arabic being included in many multilingual LLMs’ training data, our pilot study shows their off-the-shelf performance on medical tasks is suboptimal, largely due to the lack of domain-specific datasets and evaluation benchmarks—highlighting the urgent need for standardized frameworks to advance equitable AI healthcare for Arabic speakers.

Need for realistic multilingual use cases: There is an increasing demand for medical AI systems capable of functioning in multilingual settings, especially in telehealth, where tasks like teleconsultation and medical question answering require LLMs to reason and respond with clinical accuracy, simulating real practitioner interactions.

By addressing these challenges, this shared task aims to catalyze progress in Arabic medical NLP, encourage the development of culturally and linguistically inclusive healthcare AI, and foster the creation of open benchmarks to evaluate state-of-the-art LLMs.

Sub-Task 2: Open-Ended Question Answering
Data Collection and Creation
The development dataset for this sub-task consists of 400 samples. We derived our datasets from two primary sources: past exams and notes from Arabic medical schools, and the AraMed dataset (Alasmari et al., 2024). We specifically selected data sources that were unlikely to have been included in prior training datasets. The development data is divided into four subsets and tasks, each consisting of 100 samples:

Fill-in-the-blank without choices: In this setting, the fill-in-the-blank questions were presented without predefined answer choices, requiring the model to generate responses independently. This evaluation measures the model’s ability to recall and generate accurate medical knowledge without additional information, emphasizing its reasoning and language generation capabilities.

Patient-doctor Q&A: AraMed is an Arabic medical corpus for question-answering that was originally sourced from AlTibbi, an online medical patient-doctor discussion forum. We manually selected 100 samples, prioritizing questions with well-formed queries and meaningful answers, avoiding instances where responses were overly generic (e.g., “Consult a doctor” or “See a specialist'”). Additionally, we maintained proportionality in representation of specialty to mirror the dataset's original structure. We also incorporated information about the patient, specifically age and gender, when known, into the query.

Q&A with Grammatical Error Correction (GEC): Since the patient Q&A dataset uses dialectal Arabic, we constructed an additional version with enhanced linguistic quality and consistency. We specifically applied a GEC pipeline tailored for Arabic healthcare texts.

Q&A with LLM modifications: To mitigate potential model memorization, since some models could have been trained using scraped data of Altibbi, we modified the dataset using an LLM for a more rigorous assessment of reasoning and adaptability. In particular, we used GPT-4o to paraphrase our original questions using the prompt, “You are a helpful assistant that paraphrases text while keeping its meaning intact.” This approach ensured that the core medical concepts remained unchanged while reducing the likelihood of models relying on memorized content.

Task Description
This subtask emphasizes fill-in-the-blank and open-ended question answering as a generative task. Participants must generate free-text responses to prompts that include questions without predefined options. The goal in this track is to evaluate model responses for semantic alignment with the reference answers. Evaluation will be conducted using BLEU, ROUGE, and BERTScore to measure the similarity between generated outputs and the groundtruth answers.

Participants can use the development set consisting of 400 annotated questions, to guide their model development and evaluation. For the evaluation phase, a separate test set of 100 questions will be released.

Alongside the other subtask, this track is intended to provide a comprehensive benchmark for assessing how well LLMs can mimic the role of a health practitioner in Arabic, covering both knowledge retrieval and language generation capabilities in clinical contexts.

For additional information, refer to the ori

https://www.codabench.org/competitions/8740/
## Fill-in-the-blank without choices

Dataset: https://github.com/nyuad-cai/MedArabiQ/blob/main/datasets/fill-in-the-blank-nochoices.csv



## Patient-doctor Q&A:

Dataset: https://github.com/nyuad-cai/MedArabiQ/blob/main/datasets/patient-doctor-qa.csv


## Q&A with Grammatical Error Correction (GEC)

Dataset: https://github.com/nyuad-cai/MedArabiQ/blob/main/datasets/patient-doctor-qa-gec.csv



## Q&A with LLM modifications

Dataset: https://github.com/nyuad-cai/MedArabiQ/blob/main/datasets/patient-doctor-qa-llm.csv

__________________________























