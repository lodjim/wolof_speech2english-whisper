from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import evaluate
from datasets import load_dataset,DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

class WhisperFinetuner:
    def __init__(self,model_base:str,path2dataset:str) -> None:
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_base)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_base, language="english", task="transcribe")
        #self.dataset = load_dataset('json', data_files=path2dataset)
        self.dataset = DatasetDict() 
        self.dataset['train'] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
        self.metric = evaluate.load("wer")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_base)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

    def prepare_dataset(self,batch):
        audio = batch["audio"]

        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch
    
    def train_model(self,output_dir:str,per_device_train_batch_size:int=16,lr:float=1e-5) -> None:
        new_dataset = self.dataset.map(self.prepare_dataset, remove_columns=self.dataset.column_names["train"], num_proc=4)
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,  # change to a repo name of your choice
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=lr,
            warmup_steps=500,
            max_steps=4000,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=True,
        )
        trainer = Seq2SeqTrainer(
        args=training_args,
        model=self.model,
        train_dataset=new_dataset,
        data_collator=data_collator,
        compute_metrics=self.compute_metrics,
        tokenizer=self.processor.feature_extractor,
        )
        trainer.train()

    def compute_metrics(self,pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}