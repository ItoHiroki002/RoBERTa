"""import json
from sentence_transformers.evaluation import SentenceEvaluator, InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import SoftmaxLoss
import torch
from torch import Tensor
import logging
from tqdm import tqdm, trange
from sentence_transformers.util import cos_sim, dot_score
import os
import numpy as np
from typing import List, Tuple, Dict, Set, Callable
import heapq
import json
from sklearn.metrics import auc


def load_from_json_for_sentence_evaluator(path: str, return_text_keys=["symptom"]):
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    with open(path, mode="r", encoding="utf-8") as fp:
        json_data = json.load(fp)

    for qk, qv in json_data.items():
        queries[qk] = qk
        for k, v_list in qv.items():
            relevant_docs_set = set()
            if k == "positive":
                for v in v_list:
                    relevant_docs_set.add(v["name"])
                relevant_docs[qk] = relevant_docs_set
            
            for v in v_list:
                if v["name"] in corpus.keys():
                    continue
                corpus[v["name"]] = "".join([v[key] for key in return_text_keys])

    return (queries, corpus, relevant_docs, len(corpus), [i+1 for i in range(len(corpus))])


def load_from_json(path: str, return_text_keys=["symptom"]):
    corpus: list[str] = []
    corpus_dict: dict[str, int] = {}
    true_idxes_dict: dict[str, list[int]] = {}

    with open(path, mode="r", encoding="utf-8") as fp:
        json_data = json.load(fp)

    corpus_set = set()
    for qk, qv in json_data.items():
        #true_idxes_dict[qk] = []
        query_text = qk
        for k, v_list in qv.items():
            for v in v_list:
                corpus_set.add("".join([v[key] for key in return_text_keys]))
    
    corpus = sorted(list(corpus_set))
    for i in range(len(corpus)):
        corpus_dict[corpus[i]] = i

    for qk, qv in json_data.items():
        true_idxes_dict[qk] = []
        for k, v_list in qv.items():
            relevant_docs_set = set()
            if k == "positive":
                for v in v_list:
                    relevant_docs_set.add("".join([v[key] for key in return_text_keys]))
                sorted_document = sorted(list(relevant_docs_set))
                true_idxes_dict[qk] = [ corpus_dict[text] for text in sorted_document]

    return corpus, true_idxes_dict


class RankingEvaluator:
    __doc__ = '\n    This class evaluates an Information Retrieval (IR) setting.\n\n    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures\n    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)\n    '

    def __init__(self, hyoka_data_path):
        (corpus, true_idxes_dict) = load_from_json(hyoka_data_path)
        self.max_k = len(corpus)
        self.corpus = corpus
        self.true_idxes_dict = true_idxes_dict

    # Recall@kの計算
    def recall_at_k(self, true_indices, ranked_indices, k):
        top_k = set(ranked_indices[:k].tolist())
        relevant_docs = set(true_indices)
        return len(relevant_docs.intersection(top_k)) / len(relevant_docs)

    # Precision@kの計算
    def precision_at_k(self, true_indices, ranked_indices, k):
        top_k = set(ranked_indices[:k].tolist())
        relevant_docs = set(true_indices)
        if len(top_k) != len(ranked_indices[:k]):
            print("Error presicion")
        return len(relevant_docs.intersection(top_k)) / len(top_k)

    # 各クエリにおける平均精度を計算
    def average_precision(self, true_indices, ranked_indices):
        ap_sum = 0
        hit_count = 0
        for k, index in enumerate(ranked_indices, start=1):
            if index in true_indices:
                hit_count += 1
                ap_sum += hit_count / k
        if hit_count == 0:
            return 0
        return ap_sum / len(true_indices)

    # MAPを計算する関数
    def calculate_map(self, scores_dict):
        ap_values = [score['average_precision'] for score in scores_dict.values()]
        return sum(ap_values) / len(ap_values) if ap_values else 0

    # PR曲線の点とAUCを計算する関数
    def calculate_pr_curve_and_auc(self, true_indices, ranked_indices):
        precision_values = []
        recall_values = []
        

        # 各kに対してprecisionとrecallを計算
        for k in range(1, len(ranked_indices) + 1):
            precision = self.precision_at_k(true_indices, ranked_indices, k)
            recall = self.recall_at_k(true_indices, ranked_indices, k)
            # print(precision, recall)
            precision_values.append(precision)
            recall_values.append(recall)
        
        # AUCの計算
        pr_auc = auc(recall_values, precision_values)

        # 結果を辞書に格納
        pr_curve_data = {
            "precision": precision_values,
            "recall": recall_values,
            "auc": pr_auc
        }

        return pr_curve_data


    def run_eval(self, model, output_path=None, epoch=-1, steps=-1, *args, **kwargs):
        with torch.no_grad():
            corpus_emb = model.encode(
                self.corpus,
                convert_to_numpy=False,
                convert_to_tensor=True
            )
            all_score = {}

            for (query, true_idx_list) in self.true_idxes_dict.items():
            # query は必ず自然言語であること！
                query_emb = model.encode(
                    query,
                    convert_to_numpy=False,
                    convert_to_tensor=True
                )

                cos_similarities = cos_sim(query_emb, corpus_emb)
                ranked_indices = cos_similarities.argsort(descending=True)[0]

                pr_result = self.calculate_pr_curve_and_auc(
                    true_idx_list,
                ranked_indices
                )

                ap = self.average_precision(true_idx_list, ranked_indices)
                all_score[query] = {
                    "ranking": ranked_indices.cpu().tolist(),
                    "precision": pr_result["precision"],
                    "recall": pr_result["recall"],
                    "auc": pr_result["auc"],
                    "average_precision": ap
                }

        return all_score
    """

import json
import torch
import numpy as np
from sentence_transformers.util import cos_sim
from sklearn.metrics import auc
from typing import Dict, List


def load_from_json(path: str, return_text_keys=["symptom"]):
    corpus: List[str] = []
    corpus_dict: Dict[str, int] = {}
    true_idxes_dict: Dict[str, List[int]] = {}

    with open(path, mode="r", encoding="utf-8") as fp:
        json_data = json.load(fp)

    corpus_set = set()
    for qk, qv in json_data.items():
        for k, v_list in qv.items():
            for v in v_list:
                corpus_set.add("".join([v[key] for key in return_text_keys]))

    corpus = sorted(list(corpus_set))
    for i, text in enumerate(corpus):
        corpus_dict[text] = i

    for qk, qv in json_data.items():
        true_idxes_dict[qk] = []
        if "positive" in qv:
            for v in qv["positive"]:
                text = "".join([v[key] for key in return_text_keys])
                true_idxes_dict[qk].append(corpus_dict[text])

    return corpus, true_idxes_dict


class RankingEvaluator:
    """
    Information Retrieval 評価用クラス
    Recall@k, MAP, PR-AUC を算出
    """

    def __init__(self, hyoka_data_path: str):
        corpus, true_idxes_dict = load_from_json(hyoka_data_path)
        self.corpus = corpus
        self.true_idxes_dict = true_idxes_dict
        self.max_k = len(corpus)

    def recall_at_k(self, true_indices, ranked_indices, k):
        if len(true_indices) == 0:
            return 0.0
        top_k = set(ranked_indices[:k])
        return len(set(true_indices) & top_k) / len(true_indices)

    def precision_at_k(self, true_indices, ranked_indices, k):
        top_k = set(ranked_indices[:k])
        if len(top_k) == 0:
            return 0.0
        return len(set(true_indices) & top_k) / len(top_k)

    def average_precision(self, true_indices, ranked_indices):
        if len(true_indices) == 0:
            return 0.0

        score = 0.0
        hit = 0
        for i, idx in enumerate(ranked_indices, start=1):
            if idx in true_indices:
                hit += 1
                score += hit / i
        return score / len(true_indices)

    def calculate_pr_curve_and_auc(self, true_indices, ranked_indices):
        precision_vals = []
        recall_vals = []

        for k in range(1, len(ranked_indices) + 1):
            precision_vals.append(
                self.precision_at_k(true_indices, ranked_indices, k)
            )
            recall_vals.append(
                self.recall_at_k(true_indices, ranked_indices, k)
            )

        # recall が単調増加になるように補正
        recall_vals = np.maximum.accumulate(recall_vals)

        pr_auc = auc(recall_vals, precision_vals)

        return {
            "precision": precision_vals,
            "recall": recall_vals,
            "auc": pr_auc
        }

    def run_eval(self, model):
        model.eval()
        scores = {}

        with torch.no_grad():
            corpus_emb = model.encode(
                self.corpus,
                convert_to_tensor=True,
                show_progress_bar=True
            )

            for query, true_idx_list in self.true_idxes_dict.items():
                query_emb = model.encode(
                    query,
                    convert_to_tensor=True
                )

                sims = cos_sim(query_emb, corpus_emb)[0]
                ranked_indices = sims.argsort(descending=True).cpu().tolist()

                pr_result = self.calculate_pr_curve_and_auc(
                    true_idx_list,
                    ranked_indices
                )

                scores[query] = {
                    "ranking": ranked_indices,
                    "precision": pr_result["precision"],
                    "recall": pr_result["recall"],
                    "auc": pr_result["auc"],
                    "average_precision": self.average_precision(
                        true_idx_list,
                        ranked_indices
                    )
                }

        return scores
