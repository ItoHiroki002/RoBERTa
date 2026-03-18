from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# データの読み込み関数
def load_data(filename):
    data = pd.read_csv(filename, sep="\t", header=None, names=["label", "text_a", "text_b"])
    examples = [InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"]) for index, row in data.iterrows()]
    return examples

# 学習用と検証用データの読み込み
train_examples = load_data("./causal_relation_cut_train.tsv")
dev_examples = load_data("./causal_relation_cut_valid.tsv")


model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=16)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)


model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=None, evaluation_steps=500)



from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch


# モデルとデータを受け取って、Softmaxの出力から予測結果等を返す関数
def get_predictions(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            texts = [example.texts for example in batch]
            labels = [example.label for example in batch]
            true_labels.extend(labels)

            # 予測
            logits = train_loss.forward(sentence_features=texts, labels=None)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            predictions.extend(preds)
    
    return np.array(predictions), np.array(true_labels)


predictions, true_labels = get_predictions(model, dev_dataloader)

# 精度、再現率、F1スコアの計算
precision = precision_score(true_labels, predictions, average='binary')
recall = recall_score(true_labels, predictions, average='binary')
f1 = f1_score(true_labels, predictions, average='binary')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
