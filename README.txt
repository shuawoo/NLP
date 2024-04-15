1. Group Member:
WU, WANG, WU, ZHAI

2. Description of the implemented classifier:
1) input and feature representation (roberta.py)
Input: We extract and combine text before offset_start, target, text after offset_end, aspect, and use the token [SEP] as separators. We tokenize the input with pretained RobertaTokenizer from roberta-base as features. We also try to add the attention mask in the model input, but find it does not help with prediction. 
Label: We encode the labels using LabelEncoder and inverse transform the encoded predicted labels after prediction.  

2) type of classification model (classifier.py) 
We use pretrained RobertaForSequenceClassification from roberta-base as our classifier, CrossEntropyLoss as the criterion and Adam as the optimizer. After tuning the hyperparameters, we find our best performance was achieved with a learning rate of 2e-5 and a batch size of 16.  

3) other resources (earlystopping.py)
We also use early stopping technique to avoid overfitting, when validation loss no longer decreases for 3 consecutive epochs, the model will early stop.  

3. Accuracy on the dev dataset:
Completed 5 runs.
Dev accs: [89.36, 88.3, 88.83, 89.63, 88.83]
Test accs: [-1, -1, -1, -1, -1]

Mean Dev Acc.: 88.99 (0.46)
Mean Test Acc.: -1.00 (0.00)

Exec time: 665.81 s. ( 133 per run )
