# Mulitmodal Sentiment Analysis using Text and Image Data

## Overview

In this Project, we perform multimodal sentiment analysis on twitter data comprised of tweets containing both text and images [Mohammed, D. J., & Aleqabie, H. J. (2022, September). The Enrichment Of MVSA Twitter Data Via Caption-Generated Label Using Sentiment Analysis. In 2022 Iraqi International Conference on Communication and Information Technologies (IICCIT) (pp. 322-327). IEEE.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10010435&casa_token=FaL62LQyhXwAAAAA:J7OoPlBySzH0qCHs-1u7xeIzVGKIS-LP8Qcb7bliCM2IHciREp_bthjFuXNNgVKHT3ydxVmROQ?tag=1) to predict the sentiment behind the tweets. The sentiment is classified into three different categories: Positive, Neutral and Negative.

## Table of Contents
+ [Overview](#overview)
+ [Table of Contents](#table-of-contents)
+ [Datasets](#datasets)
+ [Model Architecture](#model-architecture)
+ [Preprocessing](#preprocessing)
+ [Training](#training)
+ [Evaluation](#evaluation)
+ [Usage](#usage)
+ [Dependencies](#dependencies)

## Datasets

The following dataset has been used for this project : [Mohammed, D. J., & Aleqabie, H. J. (2022, September). The Enrichment Of MVSA Twitter Data Via Caption-Generated Label Using Sentiment Analysis. In 2022 Iraqi International Conference on Communication and Information Technologies (IICCIT) (pp. 322-327). IEEE.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10010435&casa_token=FaL62LQyhXwAAAAA:J7OoPlBySzH0qCHs-1u7xeIzVGKIS-LP8Qcb7bliCM2IHciREp_bthjFuXNNgVKHT3ydxVmROQ?tag=1) which can be found [here](https://www.kaggle.com/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis).

## Model Architecture

## Preprocessing

### Text

The captions are corresponding labels are available in LabeledText.xlsx, feature engineering has been done to add the following feature columns:<br>

- Caption Length : _Indicating length of captions_
- Hashtags : _Extracting and collecting all the hashtags used in each tweet_
- Total Hashtags : _Showing the total number of hashtags in each tweets_

> The code to do this is available to run in `Scripts/Text/FeatureEng.py`, the engineered data is then saved as a csv file to `Data/Text/Engineered.csv`<br>

Afterwards, the embeddings are generated for captions and hashtags using TF-IDF approach and BERT.<br>

> The code to do this is in `Scripts/Text/Preprocess.py`, the function `tfidf_preprocessing()` and class `BERT_Embeddings` is present in `CustomFunctions.py` and then the embeddings are saved in `Data/Text/TF-IDF` and `Data/Text/BERT` along with target labels and number of captions.

## Training

## Evaluation

## Usage

## Dependencies

All the dependencies in the project are mentioned in __requirements.txt__ file. To install all dependencies run the following command in your terminal:<br>
```
pip install -r requirements.txt
```
