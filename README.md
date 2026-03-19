# RoBERTa

## 注意点

resourcesとしてSharepointのResourcesフォルダを配置してください。
基本的なコードは2023伊藤さんのコードをもとに作成してます。

## 概要

- bert_preprocess: BERTの学習前に行う事前処理に関するスクリプトなど
- bert_pretrain: BERTの事前学習に関するスクリプトなど
- bert_finetune: 事前学習後のBERTのファインチューン(Sentence-BERT以外)に関するスクリプトなど
- sbert_train: Sentence-BERTの学習に関するスクリプトなど
- sbert_finetune: 学習後のSentence-BERTに対するファインチューンに関するコード
- resources: 学習に使用するリソース等 (SharePointのResourcesフォルダをそのまま配置する前提です。)
- other: その他

## 動作確認環境

- Python 3.10.x
- PyTorch 2.2.X
- Huggingface Transformers 4.39.X
- Sentence-Transformers 2.6.X
