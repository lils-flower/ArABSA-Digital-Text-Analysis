# Arabic Aspect-Based Sentiment Analysis 🎓

This repository hosts the work done in the framework of the internship on digital text analysis conducted by Lily Foula in summer 2025. THe goal of the internship was to develop a Jupyter notebook for **Aspect-Based sentiment analysis** in Arabic. This was completed in collaboration with Prof. dr. phil. Lisa Maria Franke, Assistant professor, Department of languages and cultures, Centre for Arabic, Islamic and Middle Eastern Studies, Ghent University, with the support of the [Ghent Center for Digital Humanities](https://www.ghentcdh.ugent.be/) and the [Language and Translation Technology Team (LT3)](https://lt3.ugent.be/).

We decided to tackle this task by focusing our efforts on the development of aspect-based sentiment analysis workflows in two steps:
1.	Aspect extraction*.
2.	Sentiment analysis on the aspect and sentence columns.

*_We define an aspect as a unit in the sentence which can be both a named entity (a proper name) and a noun._

❗🧠 These Notebooks are not meant to reinvent the wheel. We simply want to build an infrastructure for scholars looking to perform aspect-based sentiment analysis on their corpus, and we do this by making step-wise code examples which can be freely used and adapted for your own purposes! 

🚀 Below we list the resources and Jupyter Notebooks we created for **aspect-based sentiment analysis** for Arabic.

## Annotated Training Dataset
For training this notebook, we used the annotated dataset from "Semeval-2016 task 5: Aspect based sentiment analysis", which is an Arabic dataset for Hotel reviews. We did not personally create nor annotate this dataset. This is the sample dataset you will find uploaded here!

References: 

Mohammad, A. S., Qwasmeh, O., Talafha, B., Al-Ayyoub, M., Jararweh, Y., & Benkhelifa, E. (2016, December). An enhanced framework for aspect-based sentiment analysis of Hotels' reviews: Arabic reviews case study. In 2016 11th International Conference for Internet Technology and Secured Transactions (ICITST) (pp. 98-103). IEEE.
Al-Smadi, M., Talafha, B., Al-Ayyoub, M., & Jararweh, Y. (2019). Using long short-term memory deep neural networks for aspect-based sentiment analysis of Arabic reviews. International Journal of Machine Learning and Cybernetics, 10(8), 2163-2175.
Pontiki, M., Galanis, D., Papageorgiou, H., Androutsopoulos, I., Manandhar, S., Mohammad, A. S., ... & Hoste, V. (2016, June). Semeval-2016 task 5: Aspect based sentiment analysis. In Proceedings of the 10th international workshop on semantic evaluation (SemEval-2016) (pp. 19-30).

[You can find the original GitHub repository hosting this dataset here.](https://github.com/msmadi/ABSA-Hotels)

Feel free to use your own annotated datset. 

###	**Aspect_flair**
Using the sample training dataset, we show you the following functionalities of the [Flair package](https://flairnlp.github.io/):
1.	Generate off-the-shelf entities for your corpus using [Flair’s off-the-shelf models](https://flairnlp.github.io/docs/category/tutorial-1-basic-tagging).
2.	Evaluate the results of your tagger on a small test set.
3.	Generate zero-shot entities for your corpus using Flair’s [TARSTAGGER](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md).
4.	Generate few-shot entities for your corpus by fine-tuning Flair’s TARSTAGGER.
5.	Create your own SequenceTagger on top of TARSTAGGER. 

## Training and evaluation (ABSA)
###	**Arabic ABSA**
This notebook shows you an approach to train an ABSA-system for your corpus in Arabic. A machine learning-based pipeline is developed in two steps: 
1)	The aspect extraction task is tackled by training a Flair-based sequence tagger using Huggingface models, and evaluating them on your gold-standard data using 5-fold cross-validation. 
2)	For the sentiment analysis task, our notebook shows you how to fine-tune HuggingFace’s embeddings on your gold standard aspects. These embeddings subsequently serve as input for diverse machine learning classification architectures, including SVM, AdaBoost, Random Forest, and MLP classifiers.


# Huggingface models
In this notebook, we used two Huggingface models.
The first model, the [AraBERT Model](aubmindlab/bert-base-arabertv2), is used in Task A which is aspect extraction.

Reference: 

@inproceedings{antoun2020arabert,
  title={AraBERT: Transformer-based Model for Arabic Language Understanding},
  author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
  booktitle={LREC 2020 Workshop Language Resources and Evaluation Conference 11--16 May 2020},
  pages={9}
}
The second model, the [Arabic Sentiment Model](Walid-Ahmed/arabic-sentiment-model), is used in Task B that is sentiment classification.

This notebook was adapted from the [GhentCDH CLSinfra English ABSA notebook](https://github.com/GhentCDH/CLSinfra/blob/main/Notebooks/ABSA_HF_English.ipynb).
For more information, visit their [GhentCDH CLSinfra project](https://github.com/GhentCDH/CLSinfra/tree/main).

## Other interesting sources to check! 🦾❗

Looking for other ways to apply ABSA on your corpus? Make sure to check out these incredibly interesting sources for Digital Humanists to experiment with!
* [LitBank](https://github.com/dbamman/litbank)
* [Python Programming for DH (YouTube)](https://www.youtube.com/@python-programming)
* [Gollie](https://hitz-zentroa.github.io/GoLLIE/): open-source Large Language Model trained to follow annotation guidelines.
* Bensoltane, R., & Zaki, T. (2022). Towards Arabic aspect-based sentiment analysis: A transfer learning-based approach. Social Network Analysis and Mining, 12, 1-16.


