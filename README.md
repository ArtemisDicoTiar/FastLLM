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
  * N-gram models (N: 1 ~ 5)
    * (model size: N/A)
  * Convolutional Neural Network (model size: ...)
  * Long-Short Term Memory (model size: ...)
  * [T5-small](https://huggingface.co/google/t5-v1_1-small) (approx. 60M parameters)

### Prerequisite

- Install python 3.8 or higher
- Install python packages
  ```sh
  $ pip install -r requirements.txt
  ```

### How to run

- Train
  ```sh
  $ python ./train.py
  ```

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
