## Create Environment
```
conda create -n LLM_finetuning_assignment python=3.10 -y
conda activate LLM_finetuning_assignment
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install einops
pip install vllm==0.2.7
pip install sacrebleu rouge_score bert_score
```




Steps:

1. Create train/test split on alpaca dataset and save the data splits:
        python data_splits.py
2. Fine-tune models on alpaca data using the train split:
        python finetune_llama.py
        python finetune_mistral.py
        python finetune_phi.py
3. Run inference on the three fine-tuned models with different hyperparameters conduct automatic evaluations:
        sh hyperparameter_experiments.sh


Hyperparameter changes used were: 
- `[10, 25, 40, 75]` for `top_k`
- `[2, 3, 5, 10]` for `num_beams`
- `[0.0, 0.25, 0.5, 1.0]` for `temperature`



## Results


### Llama2-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human  |
|----------|----------|----------|----------|-------|
| top_k_50_num_beams_1_temp_0.0 | 0.104250 | 0.234846 | 0.835651 |  0.87 |
| top_k_50_num_beams_1_temp_0.5 | 0.093281 | 0.224190 | 0.834736 |  0.77 |
| top_k_10_num_beams_1_temp_0.8 | 0.088869 | 0.229596 | 0.842532 |  0.83 |
| top_k_40_num_beams_1_temp_0.8 | 0.087204 | 0.224284 | 0.843406 | 0.77  |
| top_k_50_num_beams_2_temp_0.8 | 0.098968 | 0.234223 | 0.836924 | 0.7  |
| top_k_50_num_beams_1_temp_0.25 | 0.102562 | 0.234169 | 0.834929 |  0.87 |
| top_k_50_num_beams_5_temp_0.8 | 0.113437 | 0.254784 | 0.834861 | 0.6  |
| top_k_50_num_beams_3_temp_0.8 | 0.104220 | 0.243178 | 0.837395 | 0.77  |
| top_k_75_num_beams_1_temp_0.8 | 0.085980 | 0.221993 | 0.844428 | 0.83  |
| top_k_25_num_beams_1_temp_0.8 | 0.086934 | 0.222303 | 0.843334 | 0.77  |
| top_k_50_num_beams_1_temp_1.0 | 0.071234 | 0.209478 | 0.841053 |  0.8 |
| top_k_50_num_beams_10_temp_0.8 | 0.112773 | 0.240020 | 0.782440 |  0.63 |





### Mistral-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human |
|----------|----------|----------|----------|---------|
| top_k_50_num_beams_1_temp_0.25 | 0.094395 | 0.223659 | 0.852543 | 0.87  |
| top_k_50_num_beams_1_temp_1.0 | 0.073075 | 0.209348 | 0.856090 |  0.87 |
| top_k_50_num_beams_1_temp_0.5 | 0.090261 | 0.215826 | 0.848454 |  0.83 |
| top_k_40_num_beams_1_temp_0.8 | 0.085135 | 0.221411 | 0.845139 | 0.77  |
| top_k_25_num_beams_1_temp_0.8 | 0.087344 | 0.212703 | 0.846659 | 0.73  |
| top_k_50_num_beams_10_temp_0.8 | 0.126070 | 0.332443 | 0.870765 |  0.87 |
| top_k_50_num_beams_1_temp_0.0 | 0.094592 | 0.229714 | 0.853408 | 0.87  |
| top_k_75_num_beams_1_temp_0.8 | 0.086474 | 0.209671 | 0.847688 |0.77   |
| top_k_10_num_beams_1_temp_0.8 | 0.086329 | 0.224557 | 0.851953 | 0.77  |
| top_k_50_num_beams_3_temp_0.8 | 0.119682 | 0.294829 | 0.868226 | 0.83  |
| top_k_50_num_beams_5_temp_0.8 | 0.121952 | 0.313187 | 0.871245 |  0.83 |
| top_k_50_num_beams_2_temp_0.8 | 0.112313 | 0.282406 | 0.865394 | 0.8  |

### Phi2-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human |
|----------|----------|----------|----------|---------|
| top_k_50_num_beams_1_temp_1.0 | 0.032650 | 0.139256 | 0.778045 | 0.6  |
| top_k_25_num_beams_1_temp_0.8 | 0.040128 | 0.152024 | 0.776562 | 0.7  |
| top_k_50_num_beams_3_temp_0.8 | 0.042344 | 0.153522 | 0.768393 | 0.6  |
| top_k_50_num_beams_10_temp_0.8 | 0.059236 | 0.182520 | 0.775320 | 0.5  |
| top_k_50_num_beams_2_temp_0.8 | 0.041136 | 0.149731 | 0.765922 | 0.6  |
| top_k_50_num_beams_5_temp_0.8 | 0.049616 | 0.169586 | 0.771494 | 0.6  |
| top_k_40_num_beams_1_temp_0.8 | 0.035122 | 0.146049 | 0.774651 | 0.7  |
| top_k_10_num_beams_1_temp_0.8 | 0.037112 | 0.150133 | 0.775434 | 0.6  |
| top_k_50_num_beams_1_temp_0.5 | 0.036085 | 0.146374 | 0.767870 | 0.7  |
| top_k_75_num_beams_1_temp_0.8 | 0.032411 | 0.143313 | 0.769573 |  0.8 |
| top_k_50_num_beams_1_temp_0.25 | 0.037701 | 0.144081 | 0.764685 |  0.6 |

## Discussion Section

**Write a discussion explaining the comparison between two models. Moreover, compare the metrics and discuss which metrics are more appropriate compared to human evaluation.**

Discussion


**Write another discussion explaining the how the hyperparameters effect on the different metrics of LLaMA, Mistral, Phi-2.**

Discussion
