# Accelerating Large Language Model decoding: When small models can help LLM.
SNU - Natural Language Processing, Final Project

2023 - Autumn

## TL;DR
The project aims to improve the generation speed of Large Language Models (LLMs) by using smaller models (drafters) to generate token drafts in parallel. These drafts are then accepted or replaced by the LLM. Two approaches are considered: using Knowledge Distillation to transfer the target model's distribution to smaller models and generating a pseudo-dataset to train non-neural models. Experiments will be conducted on summarization tasks using the CNN/DailyMail dataset, with the target model being FLAN-T5-XL and drafters including N-gram models, CNN, LSTM, and T5-small. Evaluation will compare acceptance rates between distilled and baseline models.

[Presentation Link](https://docs.google.com/presentation/d/15dF94japq756oelQDUEs5wlfMWQrBnCQ3_4ujOOS5fk/edit?usp=sharing)

## Members
종윤 (@ArtemisDicoTiar), 재석 (@medduk9871), 원표 (@lenscloth), 재진 (@jjkim0807), Romain (@RomainStorai)

## Problem Definition
Large Language Models (LLMs) are struggling on generation speed as it requires auto-regressive generation (sequential token by token generation). This is mainly caused by a memory bottleneck.
We can therefore use in parallel a condensed model of the target model (orginial LLM that we want to speed-up) that will draft tokens. 
The drafts can be accepted by the system or can be replaced by the LLM generation. It is then possible to generate in one step at least one token (no time lost) and at most $\gamma + 1$ tokens if we are lucky.

Our goal is to make the drafts the more likely to be accepted by the target model (tend to generate $\gamma + 1$ tokens), which enables the generation to be accelerated.
Therefore, in this project, we aim to infuse the target model distribution to the drafters to increase the possibility of drafter model generation acceptance rate.

## Possible Solutions
1. Use Knowledge Distillation to infuse the probability distribution output of the teacher model to smaller student models.
    * We will apply this techniques to CNN, LSTM and T5 small models.
2. Generate a pseudo-dataset that mimic the probability distributions of the teacher model, and train models on it.
    * We will apply this techniques to non-neural models. We will actually build ngrams on top of this method (since ngrams are very efficient models)/

## Experiment details
For each models, we will compare the $\alpha = E(\beta)$ paramater (with $\beta$ being the acceptance rate of the drafts) between a *distilled model* and a *baseline*.
The *baseline* is a model fine-tuned on the dataset, while the *distilled model* is a model after Knowledge Distillation.

__Distillation Process__: We use the Kullback-Leibler Divergence loss function.

$L_{KD} = D_{KL}( \log(\text{softmax}(input_{logits} / T)) / \text{softmax}(target_{logits} / T))$

### Target Task and Dataset
Summarization, [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)

### Models
* Target model
  * FLAN-T5-XL (finetuned model: [bbhattar/flan_t5_xl_cnn_dailymail](https://huggingface.co/bbhattar/flan_t5_xl_cnn_dailymail))
* Forgotten small models for Drafter
  * N-gram models (N: 1 ~ 4)
  * Convolutional Neural Network (CNN)
  * Long-Short Term Memory (LSTM)
  * [T5-small](https://huggingface.co/google/t5-v1_1-small) (approx. 60M parameters)

### Prerequisite

- Install Python 3.8 or higher
- Install Pipenv
  ```sh
  $ pip install pipenv
  ```
- Install python packages
  ```sh
  $ pipenv install
  ```

### How to run

- Train (exclude ngram)
  ```sh
  $ python run.py train \
      --drafter <drafter model> \
      --exp_name <experiment name> \
      --loss <loss value> \
      --kd_loss <knowledge distillation loss> \
      --kd_temp <knowledge distillation temperature> \
      --distill_method <distillation method> \
      run
  ```
- Evaluation
  ```sh
  $ python run.py eval \
      --ckpt_path <path to checkpoint> \
      --device 0 \
      run
  ```

Since NgramModel isn't a deep learning model, the process is different for the "training" (making of the ngrams)

- Generate the pseudo-dataset: Use the multi gpus generation script `./FastLLM/scripts/generate_peudodataset_multigpus.sh` after modifying the parameters in the script.
- Fit the ngram
  ```sh
  $ python run.py train \
      --drafter ngram \
      --exp_name <experiment name> \
      --ngram_n <size of gram> \
      --pseudo_dataset_path <path to the generated dataset OR do not include to train on original> \
      run
  ```

### Model Checkpoints
*Local checkpoints...*
* N-grams (1 to 4):
   * BaseLine (Built on dataset's labels): `/data/romsto/ngrams_ckpts/dataset/` 
   * Distilled (Built on pseudo dataset): `/data/romsto/ngrams_ckpts/pseudo_dataset/`
* CNN
   * BaseLine: `/data/jaeseok/data/nlp/models/`
   * Distilled: `/data/jaeseok/data/nlp/models/cnn_cnn-drafter-2023.pt` 
* LSTM
   * BaseLine: `/data/jaeseok/data/nlp/models/`
   * Distilled: `/data/jaeseok/data/nlp/models/lstm_lstm-drafter-2023.pt` 
* T5-Small
   * BaseLine: `/data/wppark/Workspace/FastLLM/t5small_baseline-drafter-2023.pt`
   * Distilled: `/data/wppark/Workspace/FastLLM/t5small_kd50-drafter-2023.pt`

### Model details

#### NgramModel
The `NgramModel` is a torch implementation of a naive n-gram model. It is used to compute the probability of the next token given the previous prefix. The model is built using a map of n-grams and their corresponding counts, which are used to calculate the probabilities.

The model uses Laplace smoothing to handle the case of unseen n-grams in the training data. The smoothing parameter can be adjusted during the model initialization.

The `NgramModel` class provides custom methods for saving and loading the model. The model is saved in a text format, which includes the n-gram counts and total counts. To load a saved model, you should input the path of the saved model in the `resume` parameter of the constructor.

The model also uses backoff models for handling cases where the prefix length is less than n-1. The backoff model is an (n-1)-gram model that is used when the current n-gram model cannot provide a prediction.

Please note that the `fit` method is not a training method in the traditional sense, as the model is not trainable. It simply builds the n-gram counts based on the given data.


## References
~~~
@article{nallapati2016abstractive,
  title={Abstractive text summarization using sequence-to-sequence rnns and beyond},
  author={Nallapati, Ramesh and Zhou, Bowen and Gulcehre, Caglar and Xiang, Bing and others},
  journal={arXiv preprint arXiv:1602.06023},
  year={2016}
}
~~~
~~~
@inproceedings{leviathan2023fast,
  title={Fast inference from transformers via speculative decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  booktitle={International Conference on Machine Learning},
  pages={19274--19286},
  year={2023},
  organization={PMLR}
}
~~~
~~~
@article{spector2023accelerating,
  title={Accelerating llm inference with staged speculative decoding},
  author={Spector, Benjamin and Re, Chris},
  journal={arXiv preprint arXiv:2308.04623},
  year={2023}
}
~~~
~~~
@article{zhang2023draft,
  title={Draft \& Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding},
  author={Zhang, Jun and Wang, Jue and Li, Huan and Shou, Lidan and Chen, Ke and Chen, Gang and Mehrotra, Sharad},
  journal={arXiv preprint arXiv:2309.08168},
  year={2023}
}
~~~
~~~
@article{kim2023big,
  title={Big little transformer decoder},
  author={Kim, Sehoon and Mangalam, Karttikeya and Malik, Jitendra and Mahoney, Michael W and Gholami, Amir and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2302.07863},
  year={2023}
}
~~~
~~~
@misc{gu_knowledge_2023,
	title = {Knowledge {Distillation} of {Large} {Language} {Models}},
	url = {http://arxiv.org/abs/2306.08543},
	doi = {10.48550/arXiv.2306.08543},
	publisher = {arXiv},
	author = {Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
	year = {2023},
}
~~~
~~~
@misc{hinton_distilling_2015,
	title = {Distilling the {Knowledge} in a {Neural} {Network}},
	url = {http://arxiv.org/abs/1503.02531},
	doi = {10.48550/arXiv.1503.02531},
	publisher = {arXiv},
	author = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
	year = {2015},
}
~~~
