from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, TrainerCallback
import pandas as pd
from datasets import Dataset
from utils import compute_metrics
import os


def test(test_data_dir = './datasets/test.csv',labId = "reactionary_detection", ckpt_number = 1, model_name = "distilbert-base-uncased",sample_model_dir = ''):
    
    # Load test dataset
    df = pd.read_csv(test_data_dir)
    del df['dataset']
    test_dataset = Dataset.from_pandas(df)

    #Load Model
    if sample_model_dir:
        model_dir = sample_model_dir
    else:
        model_dir = f'./modelDir/{labId}/log_train/{model_name}'
      
    if sample_model_dir:
        ckpt_path =  os.path.join (model_dir, 'ckpt')
    else:
        ckpt_path =  os.path.join (model_dir, 'ckpt-'+str (ckpt_number))
      
    model = AutoModelForSequenceClassification.from_pretrained("/mnt/wsl/PHYSICALDRIVE0p1/toan/PVA/Textreactionary/modelDir/reactionary_detection/log_train/distilbert-base-uncased/ckpt-1")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/wsl/PHYSICALDRIVE0p1/toan/PVA/Textreactionary/modelDir/reactionary_detection/log_train/distilbert-base-uncased/ckpt-1")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    trainer = Trainer(
        model = model, # type: ignore
        eval_dataset = test_dataset, # type: ignore
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics, # type: ignore
    )
    result = trainer.evaluate()
    return {
        'test_acc': result['eval_accuracy'],
        'test_loss': result["eval_loss"],
        'model_checkpoint_number': ckpt_number or "Invalid"
    }

if __name__ == "__main__":
    print(test(ckpt_number=1))
    print(test(ckpt_number=2))
    print(test(ckpt_number=3))