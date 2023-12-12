# FastLLM: LLM needs forgotten Small LMs
SNU - Natural Language Processing, Final Project

2023 - Autumn

## TL;DR
[Presentation Link](https://docs.google.com/presentation/d/1aVl-7LN0Ryjw0jq_RnF0OuLRLqW1EHfgyeMqEQdj1Hw/edit?usp=drive_link)

## Members
종윤, 재석, 원표, 재진, Romain

## Problem Definition
Large Language Models (LLMs) are struggling on generation speed as it requires auto-regressively generate token by token.
To speed up the generation, condensed models of target model (original LLM) can be applied as drafter.
This drafter model output can be used as initial generation.
The draft generation can be accepted by the system or can be replaced by the LLM generation.

At this point, it is crucial to make drafter model generation to be accept in high rate so that the LLM will less involve on the generation, which decreases the generation time.
Therefore, in this project, we aim to infuse the target model distribution to the drafters to increase the possibility of drafter model generation acceptance rate.

## Possible Solutions
1. Target model generated pseudo dataset
    * The target model generated output is worked as a pseudo dataset for the drafter model to train.
3. Knowledge Distillation

## Target Task / Dataset
Summarization, [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)

## Experiment details
### Models
* Target model
  * FLAN-T5-XL (finetuned model: [bbhattar/flan_t5_xl_cnn_dailymail](https://huggingface.co/bbhattar/flan_t5_xl_cnn_dailymail))
* Forgotten small models for Drafter
  * N-gram models (N: 1 ~ 4)
  * Convolutional Neural Network
  * Long-Short Term Memory
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
* N-grams (1 to 4):
   * BaseLine (Built on dataset's labels): `/data/romsto/ngrams_ckpts/dataset/` 
   * Distilled (Built on pseudo dataset): `/data/romsto/ngrams_ckpts/pseudo_dataset/`
* CNN
   * BaseLine: 
   * Distilled: 
* LSTM
   * BaseLine: 
   * Distilled: `/data/jaeseok/data/nlp/models/lstm_lstm-drafter-2023-12-08_14-01-35.pt` 
* T5-Small
   * BaseLine: `/data/wppark/Workspace/FastLLM/t5small_baseline-drafter-2023-12-08_13-42-00.pt`
   * Distilled: `/data/wppark/Workspace/FastLLM/t5small_kd50-drafter-2023-12-08_13-36-59.pt`

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
