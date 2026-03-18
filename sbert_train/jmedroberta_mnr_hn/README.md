---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:294912
- loss:MultipleNegativesRankingLoss
base_model: alabnii/jmedroberta-base-sentencepiece
widget:
- source_sentence: 彼女が歩きながら電話で話している女性。
  sentences:
  - 少年は母親にスマートフォンの使い方を教えます。
  - 男性に囲まれた果物屋台があります。
  - アフリカ系アメリカ人の女性が黒い財布を保持している携帯電話で話しながら通りを歩きます。
- source_sentence: 犬が何かを飛び越えます。
  sentences:
  - 森に落ちた木の上を犬が飛びます。
  - 犬が茂みの中を歩きます。
  - 消防士と警官は、自動車事故の現場に対応しています。
- source_sentence: 人々のグループが水の近くに集まっています。
  sentences:
  - 外に十代の若者たちがいます。
  - 一部の子供たちはレモネードを販売しています。
  - 一部の人々は水の近くにいます。
- source_sentence: 駅でバスに乗る人。
  sentences:
  - デポ駅でバスを急いで乗車する人々。
  - ヤンキーススタジアムで２人のライブクルーが演奏します。
  - 小さなタンクと大きな腹を持つ男は彼のショートパンツを保持しています。
- source_sentence: 若い女の子がバックグラウンドで群衆とポーズ
  sentences:
  - 群衆とポーズをとる少女。
  - テーブルに座っている数人の女性。
  - 若い男がお尻を人々に見せています。
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- average_precision
- f1
- precision
- recall
- threshold
model-index:
- name: SentenceTransformer based on alabnii/jmedroberta-base-sentencepiece
  results:
  - task:
      type: paraphrase-mining
      name: Paraphrase Mining
    dataset:
      name: paramin jsnli dev
      type: paramin-jsnli-dev
    metrics:
    - type: average_precision
      value: 0.12971354860093084
      name: Average Precision
    - type: f1
      value: 0.2367111809374182
      name: F1
    - type: precision
      value: 0.19324497648567765
      name: Precision
    - type: recall
      value: 0.3054054054054054
      name: Recall
    - type: threshold
      value: 0.7011634707450867
      name: Threshold
---

# SentenceTransformer based on alabnii/jmedroberta-base-sentencepiece

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [alabnii/jmedroberta-base-sentencepiece](https://huggingface.co/alabnii/jmedroberta-base-sentencepiece). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [alabnii/jmedroberta-base-sentencepiece](https://huggingface.co/alabnii/jmedroberta-base-sentencepiece) <!-- at revision 44781e53c5c626494360c67e0a677c25ce29c942 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '若い女の子がバックグラウンドで群衆とポーズ',
    '群衆とポーズをとる少女。',
    '若い男がお尻を人々に見せています。',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8547, 0.2329],
#         [0.8547, 1.0000, 0.1766],
#         [0.2329, 0.1766, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Paraphrase Mining

* Dataset: `paramin-jsnli-dev`
* Evaluated with [<code>ParaphraseMiningEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.ParaphraseMiningEvaluator) with these parameters:
  ```json
  {'add_transitive_closure': <function ParaphraseMiningEvaluator.add_transitive_closure at 0x7f97d943feb0>, 'max_pairs': 500000, 'top_k': 100}
  ```

| Metric                | Value      |
|:----------------------|:-----------|
| **average_precision** | **0.1297** |
| f1                    | 0.2367     |
| precision             | 0.1932     |
| recall                | 0.3054     |
| threshold             | 0.7012     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 294,912 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | sentence_2                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | string                                                                            |
  | details | <ul><li>min: 5 tokens</li><li>mean: 15.68 tokens</li><li>max: 52 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 16.08 tokens</li><li>max: 71 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 13.37 tokens</li><li>max: 74 tokens</li></ul> |
* Samples:
  | sentence_0                           | sentence_1                              | sentence_2                  |
  |:-------------------------------------|:----------------------------------------|:----------------------------|
  | <code>子供たちは夕暮れ時に桟橋から水に飛び込みます。</code> | <code>夕暮れ時に水に飛び込む子供たち。</code>           | <code>海で泳いでいる赤ちゃん。</code>   |
  | <code>ボートに乗った男</code>                | <code>男は大きな青いボートに座って、汚れた水を見ています。</code> | <code>自転車の男</code>          |
  | <code>女の子がトランポリンに乗っています。</code>      | <code>女の子がトランポリンでトリックをしています。</code>     | <code>女の子が椅子に座っています。</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 10
- `disable_tqdm`: True
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: True
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss | paramin-jsnli-dev_average_precision |
|:------:|:-----:|:-------------:|:-----------------------------------:|
| 0.1085 | 500   | 2.952         | -                                   |
| 0.2170 | 1000  | 1.6581        | -                                   |
| 0.3255 | 1500  | 1.3634        | -                                   |
| 0.4340 | 2000  | 1.1925        | -                                   |
| 0.5    | 2304  | -             | 0.0790                              |
| 0.5425 | 2500  | 1.1024        | -                                   |
| 0.6510 | 3000  | 1.0059        | -                                   |
| 0.7595 | 3500  | 0.95          | -                                   |
| 0.8681 | 4000  | 0.9158        | -                                   |
| 0.9766 | 4500  | 0.8772        | -                                   |
| 1.0    | 4608  | -             | 0.0980                              |
| 1.0851 | 5000  | 0.8274        | -                                   |
| 1.1936 | 5500  | 0.78          | -                                   |
| 1.3021 | 6000  | 0.7473        | -                                   |
| 1.4106 | 6500  | 0.7161        | -                                   |
| 1.5    | 6912  | -             | 0.1080                              |
| 1.5191 | 7000  | 0.69          | -                                   |
| 1.6276 | 7500  | 0.6479        | -                                   |
| 1.7361 | 8000  | 0.618         | -                                   |
| 1.8446 | 8500  | 0.6064        | -                                   |
| 1.9531 | 9000  | 0.5749        | -                                   |
| 2.0    | 9216  | -             | 0.1149                              |
| 2.0616 | 9500  | 0.5591        | -                                   |
| 2.1701 | 10000 | 0.523         | -                                   |
| 2.2786 | 10500 | 0.5126        | -                                   |
| 2.3872 | 11000 | 0.4951        | -                                   |
| 2.4957 | 11500 | 0.4854        | -                                   |
| 2.5    | 11520 | -             | 0.1157                              |
| 2.6042 | 12000 | 0.4638        | -                                   |
| 2.7127 | 12500 | 0.4412        | -                                   |
| 2.8212 | 13000 | 0.4313        | -                                   |
| 2.9297 | 13500 | 0.417         | -                                   |
| 3.0    | 13824 | -             | 0.1201                              |
| 3.0382 | 14000 | 0.4044        | -                                   |
| 3.1467 | 14500 | 0.3824        | -                                   |
| 3.2552 | 15000 | 0.3689        | -                                   |
| 3.3637 | 15500 | 0.3593        | -                                   |
| 3.4722 | 16000 | 0.3528        | -                                   |
| 3.5    | 16128 | -             | 0.1221                              |
| 3.5807 | 16500 | 0.3487        | -                                   |
| 3.6892 | 17000 | 0.3139        | -                                   |
| 3.7977 | 17500 | 0.3194        | -                                   |
| 3.9062 | 18000 | 0.31          | -                                   |
| 4.0    | 18432 | -             | 0.1237                              |
| 4.0148 | 18500 | 0.3002        | -                                   |
| 4.1233 | 19000 | 0.2875        | -                                   |
| 4.2318 | 19500 | 0.2734        | -                                   |
| 4.3403 | 20000 | 0.2653        | -                                   |
| 4.4488 | 20500 | 0.262         | -                                   |
| 4.5    | 20736 | -             | 0.1231                              |
| 4.5573 | 21000 | 0.2597        | -                                   |
| 4.6658 | 21500 | 0.2369        | -                                   |
| 4.7743 | 22000 | 0.2369        | -                                   |
| 4.8828 | 22500 | 0.2394        | -                                   |
| 4.9913 | 23000 | 0.2336        | -                                   |
| 5.0    | 23040 | -             | 0.1253                              |
| 5.0998 | 23500 | 0.221         | -                                   |
| 5.2083 | 24000 | 0.2087        | -                                   |
| 5.3168 | 24500 | 0.204         | -                                   |
| 5.4253 | 25000 | 0.2042        | -                                   |
| 5.5    | 25344 | -             | 0.1255                              |
| 5.5339 | 25500 | 0.2006        | -                                   |
| 5.6424 | 26000 | 0.185         | -                                   |
| 5.7509 | 26500 | 0.1814        | -                                   |
| 5.8594 | 27000 | 0.1854        | -                                   |
| 5.9679 | 27500 | 0.1778        | -                                   |
| 6.0    | 27648 | -             | 0.1277                              |
| 6.0764 | 28000 | 0.1718        | -                                   |
| 6.1849 | 28500 | 0.1586        | -                                   |
| 6.2934 | 29000 | 0.1575        | -                                   |
| 6.4019 | 29500 | 0.1604        | -                                   |
| 6.5    | 29952 | -             | 0.1256                              |
| 6.5104 | 30000 | 0.154         | -                                   |
| 6.6189 | 30500 | 0.1485        | -                                   |
| 6.7274 | 31000 | 0.1438        | -                                   |
| 6.8359 | 31500 | 0.1427        | -                                   |
| 6.9444 | 32000 | 0.1383        | -                                   |
| 7.0    | 32256 | -             | 0.1297                              |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.9.1+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->