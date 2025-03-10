# Awesome-Speculative-Decoding

Reading list on speculative decoding.

**Table of Contents**

- [History & Origin](#history--origin)
- [Draft Models](#draft-models)
- [Retrieval-based Speculative Decoding](#retrieval-based-speculative-decoding)
- [Draft Tree Construction](#draft-tree-construction)
- [Verification Strategies](#verification-strategies)
- [Draft Length Control](#draft-length-control)
- [Citation](#citation)

## History & Origin

- "Fast Inference from Transformers via Speculative Decoding" [2022-11] [ICML 2023] [[paper](https://arxiv.org/abs/2211.17192)]

  > Experiments on: T5-11B, LaMDA-137B | WMT En-De, CNN/DM

- "Accelerating Large Language Model Decoding with Speculative Sampling" [2023-02] [[paper](https://arxiv.org/abs/2302.01318)]

  > Experiemnts on: Chinchilla-70B | XSum, HumanEval

## Draft Models

- "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" [2024-01] [ICML 2024] [[paper](https://arxiv.org/abs/2401.15077)]

  > The draft model autoregressively processes at the feature (hidden states before LM head) level and then derives tokens using the LM head of the target model
  >
  > Experiments on: Vicuna-7B/13B/33B, LLaMA2-Chat-7B/13B/70B, Mixtral-8x7B-Instruct | MT-Bench, HumanEval, GSM8K, Alpaca

    <p align="center">
    <img src="imgs/2401.15077-1.png" width="300"></img>
    </p>

- "GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding" [2024-02] [ICML 2024] [[paper](https://arxiv.org/abs/2402.02082)]

  > Each layer in the draft model attends to a corresponding layer in the target, counting from the top
  >
  > Experiments on: Vicuna-7B/13B/33B, Mistral-7B-Instruct-v0.1 | GSM8K, Finance-Alpaca, Spider, CodeSearchNet-Python

    <p align="center">
    <img src="imgs/2402.02082-1.png" width="600"></img>
    </p>

## Retrieval-based Speculative Decoding

- "Speculative Decoding for Multi-Sample Inference" [2025-03] [[paper](https://arxiv.org/abs/2503.05330)]

  > SD method tailored for multi-sample reasoning scenarios, such as self-consistency and Best-of-N sampling.
  >
  > "for any partial sequence on the path $i$, we use its $k$-token suffix as a query to search for matching prefixes in other paths"
  >
  > Experiments on: Qwen2.5-7B-Instruct, LLaMA-3-8B-Instruct | GSM8K, MATH

## Draft Tree Construction

- "GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding" [2024-02] [ICML 2024] [[paper](https://arxiv.org/abs/2402.02082)]

  > Expand each draft token to top-$k$ candidates where $k$ is a piecewise linear function of the top-1 confidence score $p$, set to 7, 5, 3, 1 for $p$ in (0, 0.3], (0.3, 0.6], (0.6, 0.8], (0.8, 1] respectively.
  >
  > Experiments on: Vicuna-7B/13B/33B | MT-Bench

    <p align="center">
    <img src="imgs/2402.02082-2.png" width="300"></img>
    </p>

## Verification Strategies

- "TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding" [2025-02] [[paper](https://arxiv.org/abs/2502.15197)]

  > Selects draft tokens for verification based on draft model's output probability
  >
  > Experiments on: Vicuna-33B-v1.3, LLaMA-3.1-70B/405B-Instruct | ShareGPT, Chatbot Arena, Domain Tough Questions

    <p align="center">
    <img src="imgs/2502.15197-1.png" width="300"></img>
    </p>

## Draft Length Control

- "Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models" [2024-05] [[paper](https://arxiv.org/abs/2405.04304)]

  > After generating a draft token, an FFN decides whether or not to continue drafting (input of FFN: top-10 draft model confidence, draft model entropy, token position)
  >
  > Experiments on: Vicuna-13B-v1.3, Starcoder-15B | CNN/DM, Alpaca, HumanEval, MBPP

- "Dynamic Depth Decoding: Faster Speculative Decoding for LLMs" [2024-08] [[paper](https://arxiv.org/abs/2409.00142)]

  > In the EAGLE framework, use the sum of the probabilities of all the sequences in the beam as a heuristic for whether or not to continue draft generation
  >
  > Experiments on: Vicuna-7B/13B, LLaMA2-Chat-7B/13B | MT-Bench

- "Draft Model Knows When to Stop: A Self-Verification Length Policy for Speculative Decoding" [2024-11] [[paper](https://arxiv.org/abs/2411.18462)]

  > After generating a draft token, the model decides whether or not to continue draft based on draft model entropy.
  >
  > Applies off-the-shelf to any autoregressive speculative decoding system without training.
  >
  > Experiments on: Pythia-6.9B, Vicuna-7B/13B-v1.3, LLaMA-3-70B, Qwen2.5-14B/32B, QwQ | MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DM, Natural Questions, MATH, GPQA, AIME

    <p align="center">
    <img src="imgs/2411.18462-1.png" width="600"></img>
    </p>

- "AdaEAGLE: Optimizing Speculative Decoding via Explicit Modeling of Adaptive Draft Structures" [2024-12] [[paper](https://arxiv.org/abs/2412.18910)]

  > In the EAGLE framework, train a 3-layer MLP on the penultimate prefix token's input embedding and last hidden states to predict next round's draft length.
  >
  > Experiments on: Vicuna-7B-v1.3 | MT-Bench, Alpaca, HumanEval, GSM8K, CNN/DM, Natural Questions

    <p align="center">
    <img src="imgs/2412.18910-1.png" width="600"></img>
    </p>

- "SpecServe: Efficient and SLO-Aware Large Language Model Serving with Adaptive Speculative Decoding" [2025-03] [[paper](https://arxiv.org/abs/2503.05096)]

  > 1. Adaptive drafter: incorporates efficiency estimation (based on historical data) into the drafting phase, achieving step-level speculative length control
  > 2. Confidence prior verifier: prioritizes the verification of tokens with high acceptance rates, achieving fine-grained request-level speculative length control
  > 3. SLO-aware Efficiency estimator: evaluates the efficiency of speculative decoding and achieves SLO (service level objective) awareness
  >
  > Experiments on: Vicuna-7B-v1.5, Vicuna-33B-v1.3, LLaMA-3.1-70B | MT-Bench, WMT14 De-En, CNN/DM, Natural Questions, GSM8K, DPR

## Citation

If you refer to this repo, please cite the following paper:

```
@misc{zhang2024draftmodelknowsstop,
      title={Draft Model Knows When to Stop: A Self-Verification Length Policy for Speculative Decoding},
      author={Ziyin Zhang and Jiahao Xu and Tian Liang and Xingyu Chen and Zhiwei He and Rui Wang and Zhaopeng Tu},
      year={2024},
      eprint={2411.18462},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.18462},
}
```
