import os
import random
import numpy as np
import torch
from transformers import set_seed, AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import json
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import argparse
import evaluate
import datasets
from evaluate import evaluator


# sklearnのF1スコアの評価用クラスとほとんど同じものです、
# これを実装した当時は、ここで評価用モジュールを定義・管理しないと
# マクロ平均かマイクロ平均かのどちらかしか実行できませんでした。
# 評価スコアの計算を一つのプログラム中で終わらせたかったのでこのような方法をとりました。
class F1(evaluate.Metric):
    average_mode = "micro"

    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"],
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, sample_weight=None):
        score = f1_score(
            references, predictions, labels=labels, pos_label=pos_label, average=self.average_mode, sample_weight=sample_weight
        )
        #return {"f1": float(score) if score.size == 1 else score}
        if isinstance(score, float):
            return {"f1": score}
        else:
            return {"f1": score}




f1_metric = F1()
f1_metric.average_mode = "micro"
acc_metric = evaluate.load("accuracy")
metric_compute_kwargs = {"average": "micro"}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels)


def set_seed_value(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed_value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', default='alabnii/jmedroberta-base-sentencepiece')
    parser.add_argument('--train_tsv_path', default='../../resources/dataset/seq_cls/ldcc_first_sentence_train.tsv')
    parser.add_argument('--valid_tsv_path', default='../../resources/dataset/seq_cls/ldcc_first_sentence_train.tsv')
    parser.add_argument('--trained_result_path', default='./jmedroberta_livedoor_classification')
    parser.add_argument('--checkpoint_output_path', default='./checkpoints')
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=5e-05, type=float)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--num_runs', default=5, type=int)
    args = parser.parse_args()

    train_dataset_tsv_path = args.train_tsv_path
    test_dataset_tsv_path = args.valid_tsv_path
    label_dic_file_path = os.path.join(args.trained_result_path, "label_dic.json")
    test_result_file_path = os.path.join(args.trained_result_path, "test_result.json")

    if args.seed != -1:
        print("seed value:", args.seed)
    seed = args.seed if args.seed != -1 else random.randint(0, 10000)

    train_df = pd.read_table(train_dataset_tsv_path)
    test_df = pd.read_table(test_dataset_tsv_path)

    labels = list(set(list(train_df["label"]) + list(test_df["label"])))
    num_labels = len(labels)
    label_dic = {labels[i]: i for i in range(num_labels)}
    label_mapping = {f"LABEL_{i}": i for i in range(num_labels)}
    print(label_dic)

    train_df["sentence"] = train_df['sentence'].map(lambda x: f"{x}")
    test_df["sentence"] = test_df['sentence'].map(lambda x: f"{x}")
    train_df["label"] = train_df['label'].map(label_dic)
    test_df["label"] = test_df['label'].map(label_dic)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    def tokenize_function(examples):
        return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512,   
        )

        #変更 return tokenizer(examples["sentence"], padding="max_length", truncation=True)
    
    test_tm_dataset = Dataset.from_pandas(test_df) 
    scores = {}
    for run in range(args.num_runs):
        if args.seed != -1:
            set_seed_value(seed)
        tm_dataset = Dataset.from_pandas(train_df)
        tokenized_datasets = tm_dataset.map(tokenize_function, batched=True)
        train_tm_dataset = tokenized_datasets# ["train"]
        # eval_tm_dataset = tokenized_datasets["test"]

        model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=num_labels)
        training_args = TrainingArguments(
            output_dir=os.path.join(args.checkpoint_output_path, f'run-{run}'),
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            seed=seed,
            # evaluation_strategy='epoch',
            # logging_strategy="epoch",
            # save_strategy="epoch",
            # load_best_model_at_end=True,
            # metric_for_best_model='f1',
            save_total_limit=1
        )
        if args.seed != -1:
            seed = random.randint(0, 10000)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tm_dataset,
            # eval_dataset=eval_tm_dataset,
            # compute_metrics=compute_metrics,

        )

        trainer.train()
        trainer.save_model(os.path.join(args.trained_result_path, f'run-{run}'))
        tokenizer.save_pretrained(os.path.join(args.trained_result_path, f'run-{run}'))

        with torch.no_grad():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #スコアを計算
            eval = evaluator("text-classification")
            accuracy_results = eval.compute(model_or_pipeline=trainer.model, tokenizer=tokenizer, label_mapping=label_mapping, device=device,
                                data=test_tm_dataset, metric=acc_metric, label_column="label", input_column="sentence", strategy="bootstrap"
                                )
            print(accuracy_results)
            f1_metric.average_mode = "micro"
            micro_results = eval.compute(model_or_pipeline=trainer.model, tokenizer=tokenizer, label_mapping=label_mapping, device=device,
                                data=test_tm_dataset, metric=f1_metric, label_column="label", input_column="sentence", strategy="bootstrap"
                                )
            print(micro_results)
            f1_metric.average_mode = "macro"
            macro_results = eval.compute(model_or_pipeline=trainer.model, tokenizer=tokenizer, label_mapping=label_mapping, device=device,
                                data=test_tm_dataset, metric=f1_metric, label_column="label", input_column="sentence", strategy="bootstrap"
                                )
            print(macro_results)
           
           
            scores[f"run-{run}"] = {}
            scores[f"run-{run}"]["f1_micro"] = micro_results
            scores[f"run-{run}"]["f1_marco"] = macro_results
            scores[f"run-{run}"]["accuracy"] = accuracy_results

        del model
        del trainer


    with open(test_result_file_path, mode="w", encoding="utf-8") as fp:
        json.dump(scores, fp, ensure_ascii=False)
    with open(label_dic_file_path, mode="w", encoding="utf-8") as fp:
        json.dump(label_dic, fp, ensure_ascii=False)

if __name__ == "__main__":
    main()
