"""
import os
import argparse
import json
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForPreTraining, AutoTokenizer, AutoModel, set_seed

from transformers import (AutoTokenizer, AutoModel, BertForPreTraining, TextDatasetForNextSentencePrediction,
                          DataCollatorForWholeWordMask, DataCollatorForLanguageModeling, TrainingArguments, Trainer)



parser = argparse.ArgumentParser()
parser.add_argument('--before_train_model', default='alabnii/jmedroberta-base-sentencepiece')
parser.add_argument('--trained_model', default='result_model')
parser.add_argument("--batch_size", default="8")
parser.add_argument("--epochs", default="300")
parser.add_argument("--lr", default="5e-5")
parser.add_argument("--corpus", default="all_corpus.txt")

args = parser.parse_args()


ROOT_PATH = "./"
BEFORE_TRAIN_MODEL_PATH = args.before_train_model
TRAINED_MODEL_PATH = args.trained_model
CORPUS_FILE_PATH = args.corpus
CHECKPOINT_OUTPUT_DIR_PATH = os.path.join(ROOT_PATH, "output")


print(BEFORE_TRAIN_MODEL_PATH)
print(TRAINED_MODEL_PATH)

TRAIN_EPOCH = int(args.epochs)
TRAIN_BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.lr)
SHORT_SEQ_PROB = 0.1
NEXT_SENTENCE_PROBABILITY = 0.5
MLM_PROBABILITY = 0.15
MAX_TOKENS_NUM = 512
WARMUP_RATIO = 0.1
SEED_VALUE = 42

print(TRAIN_EPOCH)
print(TRAIN_BATCH_SIZE)
print(LEARNING_RATE)


# シード値等を固定するためのもの
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

set_seed_value(SEED_VALUE)


# NSPに関する事前学習用の学習データを作成するクラス
# TextDatasetForNextSentencePredictionの実装を参考に、必要な個所をオーバーロードする実装
class CustomDatasetForBertPreTraining(TextDatasetForNextSentencePrediction):
    def __init__(self, tokenizer, file_path, block_size, overwrite_cache, short_seq_probability, nsp_probability):
        super().__init__(tokenizer, file_path, block_size, overwrite_cache, short_seq_probability, nsp_probability)
    

    # これは、あなたが直接呼び出してはいけない関数です。
    def create_examples_from_document(self, document: list[list[int]], doc_index: int, block_size: int):
        max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index:
                                break
                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    if not (len(tokens_a) >= 1):
                        raise ValueError(f"Length of sequence a is {len(tokens_a)} which must be no less than 1")
                    if not (len(tokens_b) >= 1):
                        raise ValueError(f"Length of sequence b is {len(tokens_b)} which must be no less than 1")

                    len_check = target_seq_length - (len(tokens_a) + len(tokens_b))

                    if len_check < 0:
                        tokens_b = tokens_b[:len_check]

                    input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long)
                    }

                    self.examples.append(example)

                current_chunk = []
                current_length = 0
            i += 1



# 学習データを読み込む際、をMLMを使った事前学習向けに読み込むクラス
class DataCollatorForBertPreTraining(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability, whole_word_mask=True)
  
    def __call__(self, examples):
        mlm_and_nsp_input = super().__call__(examples)
        token_type_ids_list = [example["token_type_ids"].clone().detach() for example in examples]
        token_type_ids = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)
        next_sentence_label_list = [example["next_sentence_label"].clone().detach() for example in examples]
        attention_masks_list = [(example["input_ids"] != 0).int() for example in examples]
        attention_mask = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)
        next_sentence_label = torch.stack(next_sentence_label_list, dim=0)
        mlm_and_nsp_input["token_type_ids"] = token_type_ids
        mlm_and_nsp_input["next_sentence_label"] = next_sentence_label
        mlm_and_nsp_input["attention_mask"] = attention_mask
        return mlm_and_nsp_input



# トークナイザ、モデルの読み込み
model_name = BEFORE_TRAIN_MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(model_name, verbose=False)
model = BertForPreTraining.from_pretrained(model_name)

# NSP用にDatasetを作る
dataset = CustomDatasetForBertPreTraining(tokenizer=tokenizer, file_path=CORPUS_FILE_PATH, block_size=MAX_TOKENS_NUM, overwrite_cache=True, short_seq_probability=SHORT_SEQ_PROB, nsp_probability=NEXT_SENTENCE_PROBABILITY)

# Word Whole Masking用のData Collatorを使う
data_collator = DataCollatorForBertPreTraining(tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY)

# 中断していた時に最後に保存されたチェックポイントを保存する
output_dir = CHECKPOINT_OUTPUT_DIR_PATH
checkpoint_dir = None
if os.path.exists(output_dir):
    checkpoints = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) > 0:
        checkpoint_dir = os.path.join(output_dir, checkpoints[-1])

# 学習の設定
training_args = TrainingArguments(
    output_dir=CHECKPOINT_OUTPUT_DIR_PATH,
    overwrite_output_dir=True,
    num_train_epochs=TRAIN_EPOCH,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=100,
    learning_rate=LEARNING_RATE,
    warmup_ratio = WARMUP_RATIO,
    resume_from_checkpoint=checkpoint_dir,
    seed=SEED_VALUE,
)


# 事前学習時の損失の計算のみ実装したクラス
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        next_sentence_label = inputs.pop("next_sentence_label")
        outputs = model(**inputs, labels=labels, next_sentence_label=next_sentence_label)
        return outputs.loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# 学習の実行
trainer.train()

# 学習終了時の保存動作
trainer.save_state()
trainer.save_model(TRAINED_MODEL_PATH)
tokenizer.save_pretrained(TRAINED_MODEL_PATH)
with open(os.path.join(CHECKPOINT_OUTPUT_DIR_PATH, "trainer_state.json"), mode="r", encoding="utf-8") as fp:
    trainer_state = json.load(fp)
with open(os.path.join(TRAINED_MODEL_PATH, "trainer_state.json"), mode="w", encoding="utf-8") as fp:
    json.dump(trainer_state, fp, ensure_ascii=False)
    """
import os
import argparse
import json
import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed
)

# ====== 引数 ======
parser = argparse.ArgumentParser()
parser.add_argument('--before_train_model', default='alabnii/jmedroberta-base-sentencepiece')
parser.add_argument('--trained_model', default='result_model')
parser.add_argument("--batch_size", default="8")
parser.add_argument("--epochs", default="5")
parser.add_argument("--lr", default="5e-5")
parser.add_argument("--corpus", default="all_corpus.txt")
args = parser.parse_args()

# ====== パラメータ ======
BEFORE_TRAIN_MODEL_PATH = args.before_train_model
TRAINED_MODEL_PATH = args.trained_model
CORPUS_FILE_PATH = args.corpus
CHECKPOINT_OUTPUT_DIR_PATH = "./output"

TRAIN_EPOCH = int(args.epochs)
TRAIN_BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.lr)
MLM_PROBABILITY = 0.15
SEED_VALUE = 42

# ====== シード固定 ======
def set_seed_value(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    set_seed(seed_value)

set_seed_value(SEED_VALUE)

# ====== トークナイザとモデル読み込み ======
model_name = "../bert_preprocess/jmedroberta_bf_randinit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# tokenizer と model の vocab サイズを一致させる
model.resize_token_embeddings(len(tokenizer))

# ====== データセット ======
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=CORPUS_FILE_PATH,
    block_size=128,  # 512でもOK、GPUメモリと相談
)

# ====== データコラレーター ======
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY,
)

# ====== 学習設定 ======
training_args = TrainingArguments(
    output_dir=CHECKPOINT_OUTPUT_DIR_PATH,
    overwrite_output_dir=True,
    num_train_epochs=TRAIN_EPOCH,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=LEARNING_RATE,
    save_total_limit=5,
    seed=SEED_VALUE,
)

# ====== Trainer ======
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# ====== 学習実行 ======
trainer.train()

# ====== 保存 ======
trainer.save_model(TRAINED_MODEL_PATH)
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

# 学習履歴を保存
with open(os.path.join(CHECKPOINT_OUTPUT_DIR_PATH, "trainer_state.json"), "r", encoding="utf-8") as fp:
    trainer_state = json.load(fp)
with open(os.path.join(TRAINED_MODEL_PATH, "trainer_state.json"), "w", encoding="utf-8") as fp:
    json.dump(trainer_state, fp, ensure_ascii=False)

print("✅ Training finished successfully!")
