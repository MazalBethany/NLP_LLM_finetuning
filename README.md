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
| top_k_50_num_beams_1_temp_0.0 | 0.105 | 0.235 | 0.834 |   | 
| top_k_50_num_beams_1_temp_0.5 | 0.093 | 0.225 | 0.833 |   |
| top_k_10_num_beams_1_temp_0.8 | 0.088 | 0.229 | 0.841 |   |
| top_k_40_num_beams_1_temp_0.8 | 0.089 | 0.224 | 0.844 |   |
| top_k_50_num_beams_2_temp_0.8 | 0.098 | 0.232 | 0.835 |   |
| top_k_50_num_beams_1_temp_0.25 | 0.103 | 0.232 | 0.839 |   |
| top_k_50_num_beams_5_temp_0.8 | 0.113 | 0.255 | 0.833 |  |
| top_k_50_num_beams_3_temp_0.8 | 0.104 | 0.243 | 0.838 |  |
| top_k_75_num_beams_1_temp_0.8 | 0.086 | 0.223 | 0.843 |  |
| top_k_25_num_beams_1_temp_0.8 | 0.087 | 0.222 | 0.843 |  |
| top_k_50_num_beams_1_temp_1.0 | 0.071 | 0.209 | 0.842 |  |
| top_k_50_num_beams_10_temp_0.8 | 0.113 | 0.240 | 0.783 |  |





### Mistral-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human |
|----------|----------|----------|----------|---------|
| top_k_50_num_beams_1_temp_0.25 | 0.094 | 0.225 | 0.853 |  |
| top_k_50_num_beams_1_temp_1.0 | 0.073 | 0.210 | 0.855 |  |
| top_k_50_num_beams_1_temp_0.5 | 0.091 | 0.214 | 0.848 |  |
| top_k_40_num_beams_1_temp_0.8 | 0.083 | 0.221 | 0.844 |  |
| top_k_25_num_beams_1_temp_0.8 | 0.087 | 0.212 | 0.847 |  |
| top_k_50_num_beams_10_temp_0.8 | 0.126 | 0.332 | 0.871 |  |
| top_k_50_num_beams_1_temp_0.0 | 0.096 | 0.231 | 0.854 |   |
| top_k_75_num_beams_1_temp_0.8 | 0.086 | 0.209 | 0.846 |  |
| top_k_10_num_beams_1_temp_0.8 | 0.085 | 0.223 | 0.851 |   |
| top_k_50_num_beams_3_temp_0.8 | 0.120 | 0.294 | 0.869 |   |
| top_k_50_num_beams_5_temp_0.8 | 0.121 | 0.314 | 0.870 |  |
| top_k_50_num_beams_2_temp_0.8 | 0.113 | 0.282 | 0.864 |  |

### Phi2-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human |
|----------|----------|----------|----------|---------|
| top_k_50_num_beams_1_temp_1.0 | 0.033 | 0.140 | 0.778 |  |
| top_k_25_num_beams_1_temp_0.8 | 0.041 | 0.152 | 0.775 |   |
| top_k_50_num_beams_3_temp_0.8 | 0.042 | 0.153 | 0.768 |  |
| top_k_50_num_beams_10_temp_0.8 | 0.059 | 0.183 | 0.774 |   |
| top_k_50_num_beams_2_temp_0.8 | 0.042 | 0.150 | 0.763 |  |
| top_k_50_num_beams_5_temp_0.8 | 0.049 | 0.170 | 0.771 |  |
| top_k_40_num_beams_1_temp_0.8 | 0.035 | 0.146 | 0.773 |  |
| top_k_10_num_beams_1_temp_0.8 | 0.037 | 0.151 | 0.775 | |
| top_k_50_num_beams_1_temp_0.5 | 0.036 | 0.147 | 0.766 |   |
| top_k_75_num_beams_1_temp_0.8 | 0.033 | 0.143 | 0.769 |  |
| top_k_50_num_beams_1_temp_0.25 | 0.037 | 0.144 | 0.763 |  |

## Discussion Section

**Write a discussion explaining the comparison between two models. Moreover, compare the metrics and discuss which metrics are more appropriate compared to human evaluation.**

Discussion


**Write another discussion explaining the how the hyperparameters effect on the different metrics of LLaMA, Mistral, Phi-2.**

Discussion
