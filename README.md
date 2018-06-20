# DLfM2018
Reproducible code for DLfM 2018 paper: An extended jingju solo singing voice dataset and its application on automatic assessment of singing pronunciation and overall quality at phoneme-level

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
evaluation results.

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