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
- source_sentence: ピースサインを与える２人の女の子。
  sentences:
  - 女の子は中に座って、塗り絵を描きます。
  - 外の壁にアメリカ国旗のある店で歩く２人のアジアの女性。
  - この写真には２人の女の子がいます。
- source_sentence: 男がギターを弾く
  sentences:
  - ギターを弾く年上の男
  - 男がピアノを弾く
  - 消防士と警官は、自動車事故の現場に対応しています。
- source_sentence: チェックアウトラインで２人の男性の後ろに立っている老婦人と、新聞を読んでバックグラウンドでサングラスをかけている男性。
  sentences:
  - 男は屋外です。
  - 赤ちゃんは父親に微笑みます。
  - 老婦人がチェックアウトしています。
- source_sentence: 駅でバスに乗る人。
  sentences:
  - デポ駅でバスを急いで乗車する人々。
  - ヤンキーススタジアムで２人のライブクルーが演奏します。
  - 公務員が都市広場の記念碑の前でゴミの山を掃除している。
- source_sentence: 若い女の子がバックグラウンドで群衆とポーズ
  sentences:
  - 群衆とポーズをとる少女。
  - 結婚式を祝う人々のグループ。
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
      value: 0.12464146034177122
      name: Average Precision
    - type: f1
      value: 0.22568093385214008
      name: F1
    - type: precision
      value: 0.19169027384324835
      name: Precision
    - type: recall
      value: 0.2743243243243243
      name: Recall
    - type: threshold
      value: 0.7190684378147125
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
# tensor([[1.0000, 0.8653, 0.1934],
#         [0.8653, 1.0000, 0.2579],
#         [0.1934, 0.2579, 1.0000]])
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
  {'add_transitive_closure': <function ParaphraseMiningEvaluator.add_transitive_closure at 0x7f488ddcbeb0>, 'max_pairs': 500000, 'top_k': 100}
  ```

| Metric                | Value      |
|:----------------------|:-----------|
| **average_precision** | **0.1246** |
| f1                    | 0.2257     |
| precision             | 0.1917     |
| recall                | 0.2743     |
| threshold             | 0.7191     |

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
  | details | <ul><li>min: 5 tokens</li><li>mean: 15.69 tokens</li><li>max: 52 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 16.02 tokens</li><li>max: 71 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 13.39 tokens</li><li>max: 74 tokens</li></ul> |
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
- `num_train_epochs`: 20
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
- `num_train_epochs`: 20
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
| 0.1085 | 500   | 3.3081        | -                                   |
| 0.2170 | 1000  | 1.9002        | -                                   |
| 0.3255 | 1500  | 1.5534        | -                                   |
| 0.4340 | 2000  | 1.3422        | -                                   |
| 0.5425 | 2500  | 1.2394        | -                                   |
| 0.6510 | 3000  | 1.1374        | -                                   |
| 0.7595 | 3500  | 1.0777        | -                                   |
| 0.8681 | 4000  | 1.0284        | -                                   |
| 0.9766 | 4500  | 0.9796        | -                                   |
| 1.0    | 4608  | -             | 0.0896                              |
| 1.0851 | 5000  | 0.9443        | -                                   |
| 1.1936 | 5500  | 0.8782        | -                                   |
| 1.3021 | 6000  | 0.8498        | -                                   |
| 1.4106 | 6500  | 0.8045        | -                                   |
| 1.5191 | 7000  | 0.7894        | -                                   |
| 1.6276 | 7500  | 0.7484        | -                                   |
| 1.7361 | 8000  | 0.7343        | -                                   |
| 1.8446 | 8500  | 0.716         | -                                   |
| 1.9531 | 9000  | 0.694         | -                                   |
| 2.0    | 9216  | -             | 0.1078                              |
| 2.0616 | 9500  | 0.6762        | -                                   |
| 2.1701 | 10000 | 0.6296        | -                                   |
| 2.2786 | 10500 | 0.6157        | -                                   |
| 2.3872 | 11000 | 0.5811        | -                                   |
| 2.4957 | 11500 | 0.5644        | -                                   |
| 2.6042 | 12000 | 0.5508        | -                                   |
| 2.7127 | 12500 | 0.5261        | -                                   |
| 2.8212 | 13000 | 0.5123        | -                                   |
| 2.9297 | 13500 | 0.4984        | -                                   |
| 3.0    | 13824 | -             | 0.1197                              |
| 3.0382 | 14000 | 0.4736        | -                                   |
| 3.1467 | 14500 | 0.4463        | -                                   |
| 3.2552 | 15000 | 0.4317        | -                                   |
| 3.3637 | 15500 | 0.416         | -                                   |
| 3.4722 | 16000 | 0.4008        | -                                   |
| 3.5807 | 16500 | 0.396         | -                                   |
| 3.6892 | 17000 | 0.3725        | -                                   |
| 3.7977 | 17500 | 0.3619        | -                                   |
| 3.9062 | 18000 | 0.3607        | -                                   |
| 4.0    | 18432 | -             | 0.1201                              |
| 4.0148 | 18500 | 0.3352        | -                                   |
| 4.1233 | 19000 | 0.3215        | -                                   |
| 4.2318 | 19500 | 0.3102        | -                                   |
| 4.3403 | 20000 | 0.302         | -                                   |
| 4.4488 | 20500 | 0.288         | -                                   |
| 4.5573 | 21000 | 0.289         | -                                   |
| 4.6658 | 21500 | 0.2721        | -                                   |
| 4.7743 | 22000 | 0.2636        | -                                   |
| 4.8828 | 22500 | 0.2628        | -                                   |
| 4.9913 | 23000 | 0.2426        | -                                   |
| 5.0    | 23040 | -             | 0.1231                              |
| 5.0998 | 23500 | 0.2363        | -                                   |
| 5.2083 | 24000 | 0.2218        | -                                   |
| 5.3168 | 24500 | 0.2218        | -                                   |
| 5.4253 | 25000 | 0.2088        | -                                   |
| 5.5339 | 25500 | 0.2108        | -                                   |
| 5.6424 | 26000 | 0.1986        | -                                   |
| 5.7509 | 26500 | 0.1939        | -                                   |
| 5.8594 | 27000 | 0.1909        | -                                   |
| 5.9679 | 27500 | 0.1848        | -                                   |
| 6.0    | 27648 | -             | 0.1183                              |
| 6.0764 | 28000 | 0.1729        | -                                   |
| 6.1849 | 28500 | 0.16          | -                                   |
| 6.2934 | 29000 | 0.1633        | -                                   |
| 6.4019 | 29500 | 0.1508        | -                                   |
| 6.5104 | 30000 | 0.1528        | -                                   |
| 6.6189 | 30500 | 0.1484        | -                                   |
| 6.7274 | 31000 | 0.1411        | -                                   |
| 6.8359 | 31500 | 0.1416        | -                                   |
| 6.9444 | 32000 | 0.1307        | -                                   |
| 7.0    | 32256 | -             | 0.1246                              |


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