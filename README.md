# DLfM2018
Reproducible code for DLfM 2018 paper: An extended jingju solo singing voice dataset and its application on automatic assessment of singing pronunciation and overall quality at phoneme-level

## Phoneme embedding training feature extraction
Download the datasets, [nacta](https://doi.org/10.5281/zenodo.780559), [nacta_2017](https://doi.org/10.5281/zenodo.842229) and [primary school](https://doi.org/10.5281/zenodo.1244732).
You should at least download the wav and the annotation in textgrid format. 

Change the root paths in `src.filePath.py` to the your local paths of these 3 datasets.

Change `data_path_phone_embedding_model` in `src.filePath.py` to where you want to store the extracted features.

Do
```bash
python ./dataCollection/trainingSampleCollectionPhoneEmbedding.py
```
to extract features for ANOVA feature analysis.

You can also skip this step by directly downloading the extracted log-mel features `log-mel-scaler-keys-label-encoder.zip`
from [zenodo page](https://doi.org/10.5281/zenodo.1287251), then unzip it your local `data_path_phone_embedding_model`.

## Model training
When the train data is ready, we can run the below script to train pronunciation or overall quality embedding models

1. train pronunciation embedding classification model
```bash
python ./training_scripts/embedding_model_train_pronunciation.py -d <string: train_data_path> -o <string: model_output_path> -e <string: experiment>
```

2. train overall quality embedding classification model
```bash
python ./training_scripts/embedding_model_train_overall_quality.py -d <string: train_data_path> -o <string: model_output_path> -e <string: experiment>
```

* -d <string: train_data_path>: feature, scaler, feature dictionary keys, train validation split file
* -o <string: model_output_path>: where to save the output model
* -e <string: experiment>: 'baseline', 'attention', 'dense', 'cnn', '32_embedding', 'dropout', 'best_combination'

## Model evaluation
When the classification embedding model has been trained, we can run the below scripts to get the validation or test sets
evaluation results. Or you can download the pretrained embedding models `pretrained_embedding_models.zip`
from [zenodo page](https://doi.org/10.5281/zenodo.1287251), then unzip them into your embedding model path.

1. evaluate pronunciation embeddings
```bash
python ./evaluation/eval_embedding_pronunciation.py -d <string: val_test_data_path> -v <string: val_or_test> -e <string: experiment> -o <string: result_output_path> -m <string: model_path>
```

2. evaluate overall quality embeddings
```bash
python ./evaluation/eval_embedding_overall_quality.py -d <string: val_test_data_path> -v <string: val_or_test> -e <string: experiment> -o <string: result_output_path> -m <string: model_path>
```

* -d <string: val_test_data_path>: feature, scaler, feature dictionary keys
* -v <string: val_or_test>: "val" validaton or "test" test
* -e <string: experiment>: 'baseline', 'attention', 'dense', 'cnn', '32_embedding', 'dropout', 'best_combination'
* -o <string: result_output_path>: where to save the result csv
* -m <string: model_path>: embedding model path

## ANOVA feature analysis
The goal of doing ANVOA feature analysis is to find the most discriminant individual feature to separate 
Professional, amateur train and validation, and amateur test phoneme samples.

Download the datasets, [nacta](https://doi.org/10.5281/zenodo.780559), [nacta_2017](https://doi.org/10.5281/zenodo.842229) and [primary school](https://doi.org/10.5281/zenodo.1244732).
You should at least download the wav and the annotation in textgrid format. 

Change the root paths in `src.filePath.py` to the your local paths of these 3 datasets.

Change `phn_wav_path` in `src.filePath.py` to where you want to store the phoneme-level wav locally, which will be
used for the ANOVA feature analysis.

Do
```bash
python ./dataCollection/phoneme_wav_sample_collection.py
```
to extract phoneme-level wav for ANOVA feature analysis.

Then do
```bash
python ./ANOVA_exp/freesound_feature_extraction.py
```
to extract acoustic features by using freesound extractor. This step requires Essentia installed. Please
check this [link](http://essentia.upf.edu/documentation/installing.html) for the Essentia installation detail.

You can also skip the previous steps by directly downloading 
the acoustic features `anova_analysis_essentia_feature.zip` from [zenodo page](https://doi.org/10.5281/zenodo.1287251), then
unzip it into your local `phn_wav_path`.

Finally, do
```bash
python ./ANOVA_exp/anova_calculation.py
```
to calculate the ANOVA F-value, sort the feature according to its F-value and plot the feature distributions.

## t-SNE overall quality aspect embedding visualization for each phoneme
Do the step **Phoneme embedding training feature extraction** to extract log-mel features for professional and amateur recordings.

Do
```bash
python ./tsne_embedding_extractor.py -d <string: train_data_path> -m <string: embedding_model_path> -o <string: embedding_output_path> --dense <bool>
```
to calculate the classification model embeddings for overall quality aspect.

* -d <string: train_data_path>: feature, scaler, feature dictionary keys
* -m <string: embedding_model_path>: embedding model path
* -o <string: embedding_output_path>: output calculated embeddings path
* --dense <bool>: using 32 embedding dimension or not

We provide the calculated embeddings in `./eval/phone_embedding_classifier` path, so you can skip the previous step.

Do
```bash
python ./tsne_plot.py -e <string: input_embedding_path> --dense <bool>
```
to plot the t-SNE visualization for each phoneme.

* -e <string: input_embedding_path>: embedding path
* --dense <bool>: using 32 embeddings dimension or not