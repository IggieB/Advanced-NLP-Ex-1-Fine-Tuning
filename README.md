# Exercise 1 - Advanced Natural Language Processing 67664

This is the code base for ANLP HUJI course exercise 1. 
The code is aimed at fine tuning the pretrained model bert-base-uncased to perform paraphrase detection on the MRPC dataset from the GLUE benchmark.

### Install
``` pip install -r requirements.txt ```

### Fine-Tune 
Run:

```
python ex1.py 
--do_train - use the code in training mode
--max_train_samples <number of train samples> (optional, default is all)
--max_eval_samples <number of validation samples> (optional, default is all)
--lr <learning rate> (default is 2e-5)
--num_train_epochs <number of training epochs> (default is 3)
--batch_size <batch size> (default is 8)
```

### Predict
Run:
```
python ex1.py 
--do_predict - use the code in predict mode
--max_predict_samples <number of prediction samples> (optional, default is all)
--model_path <path to prediction model> (A must! Path example: "outputs/model_ep4_lr3e-05_bs16")
--predict_split - use validation, test or train dataset split for prediction (added for accuracy computation)
```

If you use --do_predict, a prediction.txt file will be generated, containing prediction results for all test samples.
