from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import os


def infer(text = '',labId = "reactionary_detection", ckpt_number = 1, model_name = "distilbert-base-uncased",sample_model_dir = ''):

    #Load Model
    if sample_model_dir:
        model_dir = sample_model_dir
    else:
        model_dir = f'./modelDir/{labId}/log_train/{model_name}'
      
    if sample_model_dir:
        ckpt_path =  os.path.join (model_dir, 'ckpt')
    else:
        ckpt_path =  os.path.join (model_dir, 'ckpt-'+str (ckpt_number))

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    classifier = pipeline(task = 'text-classification', model = model,tokenizer = tokenizer)
    return classifier(text)

if __name__ == "__main__":
    print(infer("Đù má mày chúng mày muốn chết à ??"))