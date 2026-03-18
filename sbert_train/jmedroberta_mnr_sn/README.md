---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:311040
- loss:MultipleNegativesRankingLoss
base_model: alabnii/jmedroberta-base-sentencepiece
widget:
- source_sentence: サーフィンをしている男性は、約２０フィートの高さの波に乗ってトリックをしています。
  sentences:
  - 茶色の犬は敏ｇ性コンテストに参加しており、青いチューブに向かって走っています。
  - サーファーは非常に大きな波に乗っています。
  - サーファーは海で波を待っています。
- source_sentence: 乾燥した砂漠地帯でクリケットをする子どもたち
  sentences:
  - 屋外で遊ぶ子供たち
  - クリケットの試合を学ぶ３年生
  - 男性は腰の高さで木を切っています。
- source_sentence: 人はフードスタンドのそばにいます。
  sentences:
  - 結婚式の服を着た男女がキスをしています。
  - 黄色のシャツを着た女性がフードスタンドのそばに立っています。
  - 女性が食料品店から食料を受け取っています。
- source_sentence: 一人の男がメモに沿って続きます。
  sentences:
  - 音符を読みながらバイオリンを弾く男。
  - リフトの男性
  - 男が家で歌います。
- source_sentence: ピンクのシャツを着た少女は、女性の顔の鮮やかなピンクの壁画のそばを歩いて微笑んでいます。
  sentences:
  - 壁画のそばを歩く少女は微笑みます。
  - 少年は落書きを笑います。
  - 何かを見ている犬
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
      value: 0.10376767040844839
      name: Average Precision
    - type: f1
      value: 0.21516754850088182
      name: F1
    - type: precision
      value: 0.1715548413017276
      name: Precision
    - type: recall
      value: 0.2885135135135135
      name: Recall
    - type: threshold
      value: 0.7037822902202606
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
    'ピンクのシャツを着た少女は、女性の顔の鮮やかなピンクの壁画のそばを歩いて微笑んでいます。',
    '壁画のそばを歩く少女は微笑みます。',
    '少年は落書きを笑います。',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8269, 0.1037],
#         [0.8269, 1.0000, 0.1730],
#         [0.1037, 0.1730, 1.0000]])
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
  {'add_transitive_closure': <function ParaphraseMiningEvaluator.add_transitive_closure at 0x7f03b3f97eb0>, 'max_pairs': 500000, 'top_k': 100}
  ```

| Metric                | Value      |
|:----------------------|:-----------|
| **average_precision** | **0.1038** |
| f1                    | 0.2152     |
| precision             | 0.1716     |
| recall                | 0.2885     |
| threshold             | 0.7038     |

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

* Size: 311,040 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | sentence_2                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | string                                                                            |
  | details | <ul><li>min: 5 tokens</li><li>mean: 15.53 tokens</li><li>max: 45 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 15.89 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 13.72 tokens</li><li>max: 75 tokens</li></ul> |
* Samples:
  | sentence_0                       | sentence_1                                           | sentence_2                         |
  |:---------------------------------|:-----------------------------------------------------|:-----------------------------------|
  | <code>アジア人は野菜を見ています。</code>      | <code>２人の子供を持つアジアカップルは、スタンドで果物や野菜を見ています。</code>      | <code>健康的な食べ物について子供たちに教える親。</code> |
  | <code>汚れの上を走っている白い犬。</code>      | <code>走っている犬</code>                                  | <code>タイルの上を走っている犬。</code>         |
  | <code>茶色の犬がコンテストに参加しています。</code> | <code>茶色の犬は敏ｇ性コンテストに参加しており、青いチューブに向かって走っています。</code> | <code>黒犬が怠ｚｉによだれを垂らします。</code>     |
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
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss | paramin-jsnli-dev_average_precision |
|:------:|:-----:|:-------------:|:-----------------------------------:|
| 0.1029 | 500   | 3.0166        | -                                   |
| 0.2058 | 1000  | 1.7915        | -                                   |
| 0.3086 | 1500  | 1.4814        | -                                   |
| 0.4115 | 2000  | 1.3274        | -                                   |
| 0.5    | 2430  | -             | 0.0450                              |
| 0.5144 | 2500  | 1.2288        | -                                   |
| 0.6173 | 3000  | 1.1512        | -                                   |
| 0.7202 | 3500  | 1.0747        | -                                   |
| 0.8230 | 4000  | 1.041         | -                                   |
| 0.9259 | 4500  | 1.008         | -                                   |
| 1.0    | 4860  | -             | 0.0352                              |
| 1.0288 | 5000  | 0.9541        | -                                   |
| 1.1317 | 5500  | 0.9216        | -                                   |
| 1.2346 | 6000  | 0.8979        | -                                   |
| 1.3374 | 6500  | 0.8338        | -                                   |
| 1.4403 | 7000  | 0.8114        | -                                   |
| 1.5    | 7290  | -             | 0.0365                              |
| 1.5432 | 7500  | 0.7862        | -                                   |
| 1.6461 | 8000  | 0.7573        | -                                   |
| 1.7490 | 8500  | 0.7151        | -                                   |
| 1.8519 | 9000  | 0.7145        | -                                   |
| 1.9547 | 9500  | 0.6773        | -                                   |
| 2.0    | 9720  | -             | 0.0386                              |
| 2.0576 | 10000 | 0.6505        | -                                   |
| 2.1605 | 10500 | 0.637         | -                                   |
| 2.2634 | 11000 | 0.6274        | -                                   |
| 2.3663 | 11500 | 0.5839        | -                                   |
| 2.4691 | 12000 | 0.5736        | -                                   |
| 2.5    | 12150 | -             | 0.0401                              |
| 2.5720 | 12500 | 0.5696        | -                                   |
| 2.6749 | 13000 | 0.542         | -                                   |
| 2.7778 | 13500 | 0.5244        | -                                   |
| 2.8807 | 14000 | 0.521         | -                                   |
| 2.9835 | 14500 | 0.488         | -                                   |
| 3.0    | 14580 | -             | 0.0455                              |
| 3.0864 | 15000 | 0.4741        | -                                   |
| 3.1893 | 15500 | 0.475         | -                                   |
| 3.2922 | 16000 | 0.4561        | -                                   |
| 3.3951 | 16500 | 0.435         | -                                   |
| 3.4979 | 17000 | 0.4262        | -                                   |
| 3.5    | 17010 | -             | 0.0501                              |
| 3.6008 | 17500 | 0.4178        | -                                   |
| 3.7037 | 18000 | 0.3983        | -                                   |
| 3.8066 | 18500 | 0.399         | -                                   |
| 3.9095 | 19000 | 0.3862        | -                                   |
| 4.0    | 19440 | -             | 0.0658                              |
| 4.0123 | 19500 | 0.3586        | -                                   |
| 4.1152 | 20000 | 0.3591        | -                                   |
| 4.2181 | 20500 | 0.3619        | -                                   |
| 4.3210 | 21000 | 0.3392        | -                                   |
| 4.4239 | 21500 | 0.3312        | -                                   |
| 4.5    | 21870 | -             | 0.0681                              |
| 4.5267 | 22000 | 0.3259        | -                                   |
| 4.6296 | 22500 | 0.3174        | -                                   |
| 4.7325 | 23000 | 0.3031        | -                                   |
| 4.8354 | 23500 | 0.297         | -                                   |
| 4.9383 | 24000 | 0.2919        | -                                   |
| 5.0    | 24300 | -             | 0.0792                              |
| 5.0412 | 24500 | 0.2789        | -                                   |
| 5.1440 | 25000 | 0.2753        | -                                   |
| 5.2469 | 25500 | 0.276         | -                                   |
| 5.3498 | 26000 | 0.2585        | -                                   |
| 5.4527 | 26500 | 0.2536        | -                                   |
| 5.5    | 26730 | -             | 0.0795                              |
| 5.5556 | 27000 | 0.2555        | -                                   |
| 5.6584 | 27500 | 0.2458        | -                                   |
| 5.7613 | 28000 | 0.2368        | -                                   |
| 5.8642 | 28500 | 0.2413        | -                                   |
| 5.9671 | 29000 | 0.2263        | -                                   |
| 6.0    | 29160 | -             | 0.0885                              |
| 6.0700 | 29500 | 0.2156        | -                                   |
| 6.1728 | 30000 | 0.2224        | -                                   |
| 6.2757 | 30500 | 0.2154        | -                                   |
| 6.3786 | 31000 | 0.2046        | -                                   |
| 6.4815 | 31500 | 0.2011        | -                                   |
| 6.5    | 31590 | -             | 0.0887                              |
| 6.5844 | 32000 | 0.2054        | -                                   |
| 6.6872 | 32500 | 0.1933        | -                                   |
| 6.7901 | 33000 | 0.194         | -                                   |
| 6.8930 | 33500 | 0.1913        | -                                   |
| 6.9959 | 34000 | 0.179         | -                                   |
| 7.0    | 34020 | -             | 0.0959                              |
| 7.0988 | 34500 | 0.1788        | -                                   |
| 7.2016 | 35000 | 0.1829        | -                                   |
| 7.3045 | 35500 | 0.1723        | -                                   |
| 7.4074 | 36000 | 0.17          | -                                   |
| 7.5    | 36450 | -             | 0.0940                              |
| 7.5103 | 36500 | 0.1693        | -                                   |
| 7.6132 | 37000 | 0.1684        | -                                   |
| 7.7160 | 37500 | 0.1615        | -                                   |
| 7.8189 | 38000 | 0.1583        | -                                   |
| 7.9218 | 38500 | 0.1586        | -                                   |
| 8.0    | 38880 | -             | 0.1014                              |
| 8.0247 | 39000 | 0.1471        | -                                   |
| 8.1276 | 39500 | 0.1542        | -                                   |
| 8.2305 | 40000 | 0.1532        | -                                   |
| 8.3333 | 40500 | 0.1432        | -                                   |
| 8.4362 | 41000 | 0.1496        | -                                   |
| 8.5    | 41310 | -             | 0.0988                              |
| 8.5391 | 41500 | 0.145         | -                                   |
| 8.6420 | 42000 | 0.1435        | -                                   |
| 8.7449 | 42500 | 0.1362        | -                                   |
| 8.8477 | 43000 | 0.1378        | -                                   |
| 8.9506 | 43500 | 0.138         | -                                   |
| 9.0    | 43740 | -             | 0.1029                              |
| 9.0535 | 44000 | 0.1286        | -                                   |
| 9.1564 | 44500 | 0.1366        | -                                   |
| 9.2593 | 45000 | 0.134         | -                                   |
| 9.3621 | 45500 | 0.1285        | -                                   |
| 9.4650 | 46000 | 0.1264        | -                                   |
| 9.5    | 46170 | -             | 0.1033                              |
| 9.5679 | 46500 | 0.1333        | -                                   |
| 9.6708 | 47000 | 0.1262        | -                                   |
| 9.7737 | 47500 | 0.1243        | -                                   |
| 9.8765 | 48000 | 0.13          | -                                   |
| 9.9794 | 48500 | 0.1207        | -                                   |
| 10.0   | 48600 | -             | 0.1038                              |

</details>

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