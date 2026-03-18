import sbert_train_util
import os
import unidic_lite
import time
import json
import torch
import random
import numpy as np
from transformers import set_seed
import argparse
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator, ParaphraseMiningEvaluator
from sentence_transformers import models, losses, datasets
from torch.utils.data import DataLoader, Dataset
import math
import json
import os
import transformers
import logging
import random
from datasets import load_dataset
import logging
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_path', default='alabnii/jmedroberta-base-sentencepiece')
parser.add_argument('--train_tsv_path', default='../empty_train.tsv')
parser.add_argument('--valid_tsv_path', default='../empty_valid.tsv')
parser.add_argument('--test_tsv_path', default="../empty_test.tsv")
parser.add_argument("--hyoka_data_path", default="../../resource/eval_dataset/hyoka.json")
parser.add_argument('--trained_result_path', default='./empty')
parser.add_argument('--checkpoint_output_path', default='./empty_checkpints')
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--learning_rate', default=2e-05, type=float)
parser.add_argument('--seed', default=-1, type=int)
args = parser.parse_args()


SEED = random.randint(-32768, 32767) if args.seed == -1 else args.seed

TRAIN_DATASET_PATH = args.train_tsv_path
VALID_DATASET_PATH = args.valid_tsv_path
TEST_DATASET_PATH = args.test_tsv_path
PRETRAINED_MODEL_PATH = args.pretrained_model_path
TRAINED_RESULT_PATH = args.trained_result_path
CHEKCKPOINTS_PATH = args.checkpoint_output_path
os.makedirs(CHEKCKPOINTS_PATH, exist_ok=True)

BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
USE_AMP = True
LEARNING_RATE = args.learning_rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sbert_train_util.set_seed_value(SEED)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

word_embedding_model = models.Transformer(PRETRAINED_MODEL_PATH)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_samples = sbert_train_util.load_nli_dataset_for_MNRLoss(
    TRAIN_DATASET_PATH, False)
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

sentences_map, duplicates_list = sbert_train_util.load_nli_dataset_for_ParaphraseMiningEvalutor(VALID_DATASET_PATH)
dev_evaluator = ParaphraseMiningEvaluator(sentences_map, duplicates_list, name="paramin-jsnli-dev")

evaluation_steps = math.floor(len(train_dataloader) * EPOCHS * 0.05)
warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * 0.1) # 
logging.info("Warmup-steps: {}".format(warmup_steps))


# 学習
model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=EPOCHS,
        evaluation_steps=evaluation_steps,
        optimizer_params= {'lr': LEARNING_RATE},
        warmup_steps=warmup_steps,
        output_path=TRAINED_RESULT_PATH,
        checkpoint_path=CHEKCKPOINTS_PATH,
        checkpoint_save_steps=len(train_dataloader),
        show_progress_bar=False
        )


# 学習後の検証
with torch.no_grad():
    logging.info("Read STSbenchmark dev dataset")
    #dataset_samples = sbert_train_util.load_sts_benckmark_for_enbedding_similarity_evalator_with_hf()
    checkpoint_models = [ (os.path.join(CHEKCKPOINTS_PATH, dir), dir) for dir in os.listdir(CHEKCKPOINTS_PATH) ]

    for checkpoint, count in checkpoint_models:
        model = SentenceTransformer(checkpoint)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dataset_samples["validation"], batch_size=BATCH_SIZE, name=f'jsts-test-{count}')
        test_evaluator(model, output_path=TRAINED_RESULT_PATH)

    model = SentenceTransformer(TRAINED_RESULT_PATH)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dataset_samples["validation"], batch_size=BATCH_SIZE, name='jsts-test-result')
    test_evaluator(model, output_path=TRAINED_RESULT_PATH)

    logging.info("Read NLI test dataset")
    test_triplet_lists = sbert_train_util.load_nil_dataset_for_TripletEvaluator(TEST_DATASET_PATH, False)

    for checkpoint, count in checkpoint_models:
        model = SentenceTransformer(checkpoint).to(device)
        test_evaluator = TripletEvaluator(test_triplet_lists["anc"], test_triplet_lists["pos"], test_triplet_lists["neg"], name=f"triplet-test-{count}", batch_size=BATCH_SIZE)
        test_evaluator(model, output_path=TRAINED_RESULT_PATH)

    model = SentenceTransformer(checkpoint).to(device)
    test_evaluator = TripletEvaluator(test_triplet_lists["anc"], test_triplet_lists["pos"], test_triplet_lists["neg"], name=f"triplet-test-result", batch_size=BATCH_SIZE)
    test_evaluator(model, output_path=TRAINED_RESULT_PATH)
