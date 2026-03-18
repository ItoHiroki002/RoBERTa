import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import set_seed
from sentence_transformers import InputExample
from datasets import load_dataset

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

# LabelAccurcyLossを用いた学習に使うDataLoader用のデータ読み込み関数
def load_nli_dataset_for_LabelAccLoss(tsv_path: str)-> (list[InputExample], dict[str, int]):
    label_to_id: dict[str, int] = {"contradiction": 0, "entailment": 1, "neutral": 2}
    samples: list[InputExample] = []
    with open(tsv_path, mode="r", encoding="utf-8") as fp:
        for row in fp.readlines():
            rows = row.strip().split("\t")
            label_id = label_to_id[rows[0]]
            if ("\n" in rows[2]):
                raise Exception("改行コードが混じっています。")
            samples.append(InputExample(texts=[rows[1].replace(" ", ""), rows[2].replace(" ", "")], label=label_id))
    return samples, label_to_id


# MultopleNegativesRankingLossを用いた学習に使うNoDuplicateDataloader用のデータセットの読み込み関数
def load_nli_dataset_for_MNRLoss(tsv_path: str, nega_is_only_contradiction: bool = False)-> list[InputExample]:
    data_dict: dict[str, dict[str, set]] = {}

    def add_to_data_dict(sent1, sent2, label):
        if sent1 not in data_dict.keys():
            data_dict[sent1] = {"contradiction": set(), "entailment": set(), "neutral": set(), "nega": set()}
        data_dict[sent1][label].add(sent2)
        if label != "entailment": # contradictionとneutralをnegaに対応する要素として追加
            data_dict[sent1]["nega"].add(sent2)

    # NLIデータセットの読み込み
    with open(tsv_path, mode="r", encoding="utf-8") as fp:
        for row in fp.readlines():
            if row == "\n":
                continue
            rows = row.strip().split("\t")
            sent1 = rows[1].replace(" ", "")
            sent2 = rows[2].replace(" ", "")
            label = rows[0].replace(" ", "")
            add_to_data_dict(sent1, sent2, label)
            add_to_data_dict(sent2, sent1, label)

    samples: list[InputExample] = []
    nega_key = "contradiction" if nega_is_only_contradiction else "nega"
    for sent1, others in data_dict.items():        
        if len(others['entailment']) > 0 and len(others[nega_key]) > 0:
            samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others[nega_key]))]))
            samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others[nega_key]))]))
    return samples # 学習データとなる、3つの文の組のリスト


# TripletEvaluator用にデータを読み込む関数
def load_nil_dataset_for_TripletEvaluator(tsv_path: str, nega_is_only_contradiction: bool = False)-> dict[str, list[str]]:
    samples: list[InputExample] = load_nli_dataset_for_MNRLoss(tsv_path, nega_is_only_contradiction)
    triplet_lists: dict[str, list[str]] = {"anc":[], "pos":[], "neg":[]}

    for sample in samples:
        triplet_lists["anc"].append(sample.texts[0])
        triplet_lists["pos"].append(sample.texts[1])
        triplet_lists["neg"].append(sample.texts[2])
    return triplet_lists

def load_nli_dataset_for_ParaphraseMiningEvalutor(tsv_path):
    sentences_map: dict[id, str] = {} # id -> sent
    sentences_reverse_map: dict[str, int] = {} # sent -> id
    duplicates_list: list[tuple[int, int]] = [] # (id1, id2)

    def register(sent):
        if sent not in sentences_reverse_map:
            id = str(len(sentences_reverse_map))
            sentences_reverse_map[sent] = id
            sentences_map[id] = sent
            return id
        else:
          return sentences_reverse_map[sent]

    with open(tsv_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split("\t") for line in lines]
        rows = [[line[0], line[1].replace(" ", ""), line[2].replace(" ", "")] for line in lines]
        for row in rows:
            label = row[0] 
            sent1 = row[1]
            sent2 = row[2]
            ids = [register(sent) for sent in [sent1, sent2]]
            if label == "entailment":
                duplicates_list.append(tuple(ids))
    return sentences_map, duplicates_list


# STSベンチマーク用のEmbeddingSimiparityEvaluator用のデータ読み込み関数
def load_sts_benckmark_for_enbedding_similarity_evalator_with_hf(path: str = "shunk031/JGLUE", name="JSTS"):
    dataset_dict = load_dataset(path=path, name=name)
    dataset_samples: dict[str, list[InputExample]] = {}
    for key, dataset in dataset_dict.items():
        samples: list[InputExample] = []
        for data in dataset:
            score = float(data["label"]) / 5.0
            samples.append(InputExample(texts=[data['sentence1'], data['sentence2']], label=score))
        dataset_samples[key] = samples
    return dataset_samples


# MultopleNegativesRankingLossに重みをつけた学習に使うNoDuplicateDataloader用のデータセットの読み込み関数
def load_nli_dataset_for_WeightedMNRLoss(tsv_path: str, weight=0.3)-> list[InputExample]:
    data_dict: dict[str, dict[str, set]] = {}

    def add_to_data_dict(sent1, sent2, label):
        if sent1 not in data_dict.keys():
            data_dict[sent1] = {"contradiction": set(), "entailment": set(), "neutral": set(), "nega": set()}
        data_dict[sent1][label].add(sent2)
        if label != "entailment":
            data_dict[sent1]["nega"].add(sent2)

    with open(tsv_path, mode="r", encoding="utf-8") as fp:
        for row in fp.readlines():
            if row == "\n":
                continue
            rows = row.strip().split("\t")
            sent1 = rows[1].replace(" ", "")
            sent2 = rows[2].replace(" ", "")
            label = rows[0].replace(" ", "")
            add_to_data_dict(sent1, sent2, label)
            add_to_data_dict(sent2, sent1, label)

    samples: list[InputExample] = []
    for sent1, others in data_dict.items():
        if len(others['entailment']) > 0 and len(others["nega"]) > 0:
            text1 = random.choice(list(others["nega"]))
            text2 = random.choice(list(others["nega"]))
            samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), text1], label=(1.0 if text1 in list(others['contradiction']) else weight)))
            samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, text2], label=(1.0 if text2 in list(others['contradiction']) else weight)))
    return samples