from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, TrainerCallback
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np
from copy import deepcopy

def load_train_valid_dataset(train_data_dir,val_size):
    df = pd.read_csv(train_data_dir)
    df = df[df['text'].notna()]
    del df['dataset']
    df_valid = df.sample(frac=val_size)
    df_train = df.drop(df_valid.index)
    train_dataset = Dataset.from_pandas(df_train)
    valid_dataset = Dataset.from_pandas(df_valid)
    return train_dataset, valid_dataset


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
    
class Model_Reactionary_Detection:
    def __init__(self, model_name = 'distilbert-base-uncased', train_data_dir = "./datasets/train.csv", val_size = 0.1):
        
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, id2label=self.id2label, label2id=self.label2id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset, self.valid_dataset = load_train_valid_dataset(train_data_dir,val_size)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def preprocess_function(self,examples):
        return self.tokenizer(examples["text"], truncation=True)
    
    def train_tt(self, learning_rate = 0.3, EPOCHS = 10, BS = 32):
        self.train_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        self.valid_dataset = self.valid_dataset.map(self.preprocess_function, batched=True)
        training_args = TrainingArguments(
            output_dir="./models",
            evaluation_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=BS,
            per_device_eval_batch_size=BS,
            learning_rate=learning_rate,
            num_train_epochs=1,
            lr_scheduler_type='constant',
            logging_strategy = "no",
            report_to=["tensorboard"],
            load_best_model_at_end=False,
            metric_for_best_model="accuracy"
        )
        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = self.train_dataset, # type: ignore
            eval_dataset = self.valid_dataset, # type: ignore
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics = compute_metrics, # type: ignore
        )
        trainer.add_callback(CustomCallback(trainer))
        for epoch in range(EPOCHS):
            trainer.train()
            trainer.save_model(f"./models/epoch_{epoch+1}")
            yield {
                "epoch" : epoch + 1,
                "train_accuracy" : trainer.state.log_history[0]["train_accuracy"],
                "train_loss": trainer.state.log_history[0]["train_loss"],
                "eval_accuracy" : trainer.state.log_history[1]["eval_accuracy"],
                "eval_loss": trainer.state.log_history[1]["eval_loss"]
            }
            
if __name__ == "__main__":
    model = Model_Reactionary_Detection()
    results = []
    results_generator = model.train_tt()
    for result in results_generator:
        results.append(result)
    print(results)