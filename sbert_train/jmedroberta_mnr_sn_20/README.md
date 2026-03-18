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
- source_sentence: 二人の男が電気のこぎりで木を切った
  sentences:
  - アフリカ系アメリカ人の紳士がピアノを弾き、若い男の子に見られながらマイクに向かって歌っています。
  - 男性は腰の高さで木を切っています。
  - 男性は木をのこぎりで挽いています。
- source_sentence: ボウリング場の女性がボールをリリースしています。
  sentences:
  - ボウリングレーンでボウリングボールを解放する青いシャツを着た女性。
  - 男がテニスコートでボールを提供しています。
  - 安全眼鏡をかけている男性が自転車のタイヤを溶接します。
- source_sentence: ピンクのシャツを着た少女は、女性の顔の鮮やかなピンクの壁画のそばを歩いて微笑んでいます。
  sentences:
  - 老婦人が床に座っている間、犬はマットレスの上に横たわっています。
  - 壁画のそばを歩く少女は微笑みます。
  - 少年は落書きを笑います。
- source_sentence: 赤レンガの表面を歩く群衆。
  sentences:
  - 群衆はレンガの上を歩いています。
  - 見物人が見ている間、縞模様のシャツと青いズボンを着た少女がトランポリンに落ちます。
  - 群衆は抗議で歩いています。
- source_sentence: 男が楽器を演奏します。
  sentences:
  - 赤いドラムセットを演奏する男。
  - 主に黒い服を着た男が、巨大な三角形のギターのような楽器を持ち、その前に譜面台があります。
  - 人々のグループは外にあります。
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
      value: 0.10077085586428626
      name: Average Precision
    - type: f1
      value: 0.21395106715252474
      name: F1
    - type: precision
      value: 0.17400508044030483
      name: Precision
    - type: recall
      value: 0.2777027027027027
      name: Recall
    - type: threshold
      value: 0.7026666700839996
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
    '男が楽器を演奏します。',
    '赤いドラムセットを演奏する男。',
    '主に黒い服を着た男が、巨大な三角形のギターのような楽器を持ち、その前に譜面台があります。',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6424, 0.1863],
#         [0.6424, 1.0000, 0.0620],
#         [0.1863, 0.0620, 1.0000]])
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
  {'add_transitive_closure': <function ParaphraseMiningEvaluator.add_transitive_closure at 0x7f4fcd5b3eb0>, 'max_pairs': 500000, 'top_k': 100}
  ```

| Metric                | Value      |
|:----------------------|:-----------|
| **average_precision** | **0.1008** |
| f1                    | 0.214      |
| precision             | 0.174      |
| recall                | 0.2777     |
| threshold             | 0.7027     |

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
  | details | <ul><li>min: 5 tokens</li><li>mean: 15.53 tokens</li><li>max: 45 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 15.94 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 13.77 tokens</li><li>max: 75 tokens</li></ul> |
* Samples:
  | sentence_0                       | sentence_1                                           | sentence_2                         |
  |:---------------------------------|:-----------------------------------------------------|:-----------------------------------|
  | <code>アジア人は野菜を見ています。</code>      | <code>２人の子供を持つアジアカップルは、スタンドで果物や野菜を見ています。</code>      | <code>健康的な食べ物について子供たちに教える親。</code> |
  | <code>汚れの上を走っている白い犬。</code>      | <code>走っている犬</code>                                  | <code>別の犬を追いかける犬</code>            |
  | <code>茶色の犬がコンテストに参加しています。</code> | <code>茶色の犬は敏ｇ性コンテストに参加しており、青いチューブに向かって走っています。</code> | <code>茶色の犬がコンテストで優勝しています。</code>   |
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
<details><summary>Click to expand</summary>

| Epoch   | Step  | Training Loss | paramin-jsnli-dev_average_precision |
|:-------:|:-----:|:-------------:|:-----------------------------------:|
| 0.1029  | 500   | 3.3502        | -                                   |
| 0.2058  | 1000  | 2.0285        | -                                   |
| 0.3086  | 1500  | 1.6526        | -                                   |
| 0.4115  | 2000  | 1.4912        | -                                   |
| 0.5144  | 2500  | 1.3973        | -                                   |
| 0.6173  | 3000  | 1.2653        | -                                   |
| 0.7202  | 3500  | 1.2061        | -                                   |
| 0.8230  | 4000  | 1.1634        | -                                   |
| 0.9259  | 4500  | 1.0933        | -                                   |
| 1.0     | 4860  | -             | 0.0470                              |
| 1.0288  | 5000  | 1.0661        | -                                   |
| 1.1317  | 5500  | 1.0153        | -                                   |
| 1.2346  | 6000  | 0.9817        | -                                   |
| 1.3374  | 6500  | 0.9482        | -                                   |
| 1.4403  | 7000  | 0.9221        | -                                   |
| 1.5432  | 7500  | 0.9053        | -                                   |
| 1.6461  | 8000  | 0.8488        | -                                   |
| 1.7490  | 8500  | 0.8362        | -                                   |
| 1.8519  | 9000  | 0.8211        | -                                   |
| 1.9547  | 9500  | 0.7891        | -                                   |
| 2.0     | 9720  | -             | 0.0431                              |
| 2.0576  | 10000 | 0.7698        | -                                   |
| 2.1605  | 10500 | 0.7278        | -                                   |
| 2.2634  | 11000 | 0.7332        | -                                   |
| 2.3663  | 11500 | 0.6928        | -                                   |
| 2.4691  | 12000 | 0.6879        | -                                   |
| 2.5720  | 12500 | 0.6599        | -                                   |
| 2.6749  | 13000 | 0.6241        | -                                   |
| 2.7778  | 13500 | 0.6115        | -                                   |
| 2.8807  | 14000 | 0.591         | -                                   |
| 2.9835  | 14500 | 0.5699        | -                                   |
| 3.0     | 14580 | -             | 0.0456                              |
| 3.0864  | 15000 | 0.5445        | -                                   |
| 3.1893  | 15500 | 0.5238        | -                                   |
| 3.2922  | 16000 | 0.5318        | -                                   |
| 3.3951  | 16500 | 0.4965        | -                                   |
| 3.4979  | 17000 | 0.4955        | -                                   |
| 3.6008  | 17500 | 0.472         | -                                   |
| 3.7037  | 18000 | 0.4556        | -                                   |
| 3.8066  | 18500 | 0.4465        | -                                   |
| 3.9095  | 19000 | 0.4233        | -                                   |
| 4.0     | 19440 | -             | 0.0505                              |
| 4.0123  | 19500 | 0.4177        | -                                   |
| 4.1152  | 20000 | 0.3958        | -                                   |
| 4.2181  | 20500 | 0.3867        | -                                   |
| 4.3210  | 21000 | 0.3925        | -                                   |
| 4.4239  | 21500 | 0.3687        | -                                   |
| 4.5267  | 22000 | 0.3648        | -                                   |
| 4.6296  | 22500 | 0.3391        | -                                   |
| 4.7325  | 23000 | 0.3368        | -                                   |
| 4.8354  | 23500 | 0.331         | -                                   |
| 4.9383  | 24000 | 0.31          | -                                   |
| 5.0     | 24300 | -             | 0.0614                              |
| 5.0412  | 24500 | 0.3051        | -                                   |
| 5.1440  | 25000 | 0.2911        | -                                   |
| 5.2469  | 25500 | 0.2914        | -                                   |
| 5.3498  | 26000 | 0.2816        | -                                   |
| 5.4527  | 26500 | 0.2715        | -                                   |
| 5.5556  | 27000 | 0.2691        | -                                   |
| 5.6584  | 27500 | 0.2531        | -                                   |
| 5.7613  | 28000 | 0.2519        | -                                   |
| 5.8642  | 28500 | 0.2413        | -                                   |
| 5.9671  | 29000 | 0.2305        | -                                   |
| 6.0     | 29160 | -             | 0.0583                              |
| 6.0700  | 29500 | 0.2236        | -                                   |
| 6.1728  | 30000 | 0.2147        | -                                   |
| 6.2757  | 30500 | 0.2199        | -                                   |
| 6.3786  | 31000 | 0.207         | -                                   |
| 6.4815  | 31500 | 0.2074        | -                                   |
| 6.5844  | 32000 | 0.1974        | -                                   |
| 6.6872  | 32500 | 0.1935        | -                                   |
| 6.7901  | 33000 | 0.1861        | -                                   |
| 6.8930  | 33500 | 0.1759        | -                                   |
| 6.9959  | 34000 | 0.1773        | -                                   |
| 7.0     | 34020 | -             | 0.0701                              |
| 7.0988  | 34500 | 0.1691        | -                                   |
| 7.2016  | 35000 | 0.1659        | -                                   |
| 7.3045  | 35500 | 0.1697        | -                                   |
| 7.4074  | 36000 | 0.1572        | -                                   |
| 7.5103  | 36500 | 0.1596        | -                                   |
| 7.6132  | 37000 | 0.1468        | -                                   |
| 7.7160  | 37500 | 0.1492        | -                                   |
| 7.8189  | 38000 | 0.1456        | -                                   |
| 7.9218  | 38500 | 0.1333        | -                                   |
| 8.0     | 38880 | -             | 0.0805                              |
| 8.0247  | 39000 | 0.1354        | -                                   |
| 8.1276  | 39500 | 0.1286        | -                                   |
| 8.2305  | 40000 | 0.1289        | -                                   |
| 8.3333  | 40500 | 0.1245        | -                                   |
| 8.4362  | 41000 | 0.1219        | -                                   |
| 8.5391  | 41500 | 0.1209        | -                                   |
| 8.6420  | 42000 | 0.1135        | -                                   |
| 8.7449  | 42500 | 0.1145        | -                                   |
| 8.8477  | 43000 | 0.1109        | -                                   |
| 8.9506  | 43500 | 0.1059        | -                                   |
| 9.0     | 43740 | -             | 0.0826                              |
| 9.0535  | 44000 | 0.1005        | -                                   |
| 9.1564  | 44500 | 0.0989        | -                                   |
| 9.2593  | 45000 | 0.1025        | -                                   |
| 9.3621  | 45500 | 0.0968        | -                                   |
| 9.4650  | 46000 | 0.0951        | -                                   |
| 9.5679  | 46500 | 0.0946        | -                                   |
| 9.6708  | 47000 | 0.0911        | -                                   |
| 9.7737  | 47500 | 0.0875        | -                                   |
| 9.8765  | 48000 | 0.0869        | -                                   |
| 9.9794  | 48500 | 0.0837        | -                                   |
| 10.0    | 48600 | -             | 0.0861                              |
| 10.0823 | 49000 | 0.0834        | -                                   |
| 10.1852 | 49500 | 0.0785        | -                                   |
| 10.2881 | 50000 | 0.0826        | -                                   |
| 10.3909 | 50500 | 0.0756        | -                                   |
| 10.4938 | 51000 | 0.0773        | -                                   |
| 10.5967 | 51500 | 0.0724        | -                                   |
| 10.6996 | 52000 | 0.0741        | -                                   |
| 10.8025 | 52500 | 0.0717        | -                                   |
| 10.9053 | 53000 | 0.0681        | -                                   |
| 11.0    | 53460 | -             | 0.0886                              |
| 11.0082 | 53500 | 0.0693        | -                                   |
| 11.1111 | 54000 | 0.065         | -                                   |
| 11.2140 | 54500 | 0.0643        | -                                   |
| 11.3169 | 55000 | 0.0666        | -                                   |
| 11.4198 | 55500 | 0.0604        | -                                   |
| 11.5226 | 56000 | 0.0638        | -                                   |
| 11.6255 | 56500 | 0.0585        | -                                   |
| 11.7284 | 57000 | 0.0613        | -                                   |
| 11.8313 | 57500 | 0.059         | -                                   |
| 11.9342 | 58000 | 0.0544        | -                                   |
| 12.0    | 58320 | -             | 0.0934                              |
| 12.0370 | 58500 | 0.0549        | -                                   |
| 12.1399 | 59000 | 0.0532        | -                                   |
| 12.2428 | 59500 | 0.055         | -                                   |
| 12.3457 | 60000 | 0.0522        | -                                   |
| 12.4486 | 60500 | 0.0513        | -                                   |
| 12.5514 | 61000 | 0.0516        | -                                   |
| 12.6543 | 61500 | 0.0493        | -                                   |
| 12.7572 | 62000 | 0.0497        | -                                   |
| 12.8601 | 62500 | 0.0487        | -                                   |
| 12.9630 | 63000 | 0.0463        | -                                   |
| 13.0    | 63180 | -             | 0.0928                              |
| 13.0658 | 63500 | 0.0462        | -                                   |
| 13.1687 | 64000 | 0.044         | -                                   |
| 13.2716 | 64500 | 0.046         | -                                   |
| 13.3745 | 65000 | 0.0433        | -                                   |
| 13.4774 | 65500 | 0.0436        | -                                   |
| 13.5802 | 66000 | 0.044         | -                                   |
| 13.6831 | 66500 | 0.0426        | -                                   |
| 13.7860 | 67000 | 0.0418        | -                                   |
| 13.8889 | 67500 | 0.0394        | -                                   |
| 13.9918 | 68000 | 0.0407        | -                                   |
| 14.0    | 68040 | -             | 0.0951                              |
| 14.0947 | 68500 | 0.0387        | -                                   |
| 14.1975 | 69000 | 0.0379        | -                                   |
| 14.3004 | 69500 | 0.0391        | -                                   |
| 14.4033 | 70000 | 0.0351        | -                                   |
| 14.5062 | 70500 | 0.0386        | -                                   |
| 14.6091 | 71000 | 0.0354        | -                                   |
| 14.7119 | 71500 | 0.0376        | -                                   |
| 14.8148 | 72000 | 0.0361        | -                                   |
| 14.9177 | 72500 | 0.0343        | -                                   |
| 15.0    | 72900 | -             | 0.0947                              |
| 15.0206 | 73000 | 0.0346        | -                                   |
| 15.1235 | 73500 | 0.0339        | -                                   |
| 15.2263 | 74000 | 0.0333        | -                                   |
| 15.3292 | 74500 | 0.0329        | -                                   |
| 15.4321 | 75000 | 0.0315        | -                                   |
| 15.5350 | 75500 | 0.0327        | -                                   |
| 15.6379 | 76000 | 0.0313        | -                                   |
| 15.7407 | 76500 | 0.032         | -                                   |
| 15.8436 | 77000 | 0.0319        | -                                   |
| 15.9465 | 77500 | 0.0291        | -                                   |
| 16.0    | 77760 | -             | 0.0980                              |
| 16.0494 | 78000 | 0.0296        | -                                   |
| 16.1523 | 78500 | 0.0282        | -                                   |
| 16.2551 | 79000 | 0.0303        | -                                   |
| 16.3580 | 79500 | 0.0285        | -                                   |
| 16.4609 | 80000 | 0.0287        | -                                   |
| 16.5638 | 80500 | 0.0282        | -                                   |
| 16.6667 | 81000 | 0.0282        | -                                   |
| 16.7695 | 81500 | 0.0281        | -                                   |
| 16.8724 | 82000 | 0.0276        | -                                   |
| 16.9753 | 82500 | 0.027         | -                                   |
| 17.0    | 82620 | -             | 0.0974                              |
| 17.0782 | 83000 | 0.0262        | -                                   |
| 17.1811 | 83500 | 0.0254        | -                                   |
| 17.2840 | 84000 | 0.0269        | -                                   |
| 17.3868 | 84500 | 0.0253        | -                                   |
| 17.4897 | 85000 | 0.0265        | -                                   |
| 17.5926 | 85500 | 0.0253        | -                                   |
| 17.6955 | 86000 | 0.0258        | -                                   |
| 17.7984 | 86500 | 0.0251        | -                                   |
| 17.9012 | 87000 | 0.024         | -                                   |
| 18.0    | 87480 | -             | 0.1004                              |
| 18.0041 | 87500 | 0.0247        | -                                   |
| 18.1070 | 88000 | 0.0243        | -                                   |
| 18.2099 | 88500 | 0.024         | -                                   |
| 18.3128 | 89000 | 0.0245        | -                                   |
| 18.4156 | 89500 | 0.0221        | -                                   |
| 18.5185 | 90000 | 0.0245        | -                                   |
| 18.6214 | 90500 | 0.0226        | -                                   |
| 18.7243 | 91000 | 0.0243        | -                                   |
| 18.8272 | 91500 | 0.0236        | -                                   |
| 18.9300 | 92000 | 0.0222        | -                                   |
| 19.0    | 92340 | -             | 0.1008                              |

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