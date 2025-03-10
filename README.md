# Awesome-Speculative-Decoding

Reading list on speculative decoding.

**Table of Contents**

- [History](#history)
- [Draft Models](#draft-models)
- [Retrieval-based Speculative Decoding](#retrieval-based-speculative-decoding)
- [Draft Length Control](#draft-length-control)
- [Citation]()

## History

## Draft Models

## Retrieval-based Speculative Decoding

- "Speculative Decoding for Multi-Sample Inference" [2025-03] [[paper](https://arxiv.org/abs/2503.05330)]

  > SD method tailored for multi-sample reasoning scenarios, such as self-consistency and Best-of-N sampling.
  >
  > "for any partial sequence on the path $i$, we use its $k$-token suffix as a query to search for matching prefixes in other paths"
  >
  > Experiments on: Qwen2.5-7B-Instruct, LLaMA-3-8B-Instruct, GSM8K, MATH

## Verification Strategies

- "TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding" [2025-02] [[paper](https://arxiv.org/abs/2502.15197)]

  > Selects draft tokens for verification based on draft model's output probability
  >
  > Experiments on: Vicuna-33B-v1.3, LLaMA-3.1-70B/405B-Instruct, ShareGPT, Chatbot Arena, Domain Tough Questions

    <p align="center">
    <img src="imgs/2502.15197-1.png" width="300"></img>
    </p>

## Draft Length Control

- "AdaEAGLE: Optimizing Speculative Decoding via Explicit Modeling of Adaptive Draft Structures" [2024-12] [[paper](https://arxiv.org/abs/2412.18910)]

  > Trains a 3-layer MLP on the penultimate prefix token's input embedding and last hidden states to predict next round's draft length.
  >
  > Experiments on: Vicuna-7B-v1.3, MT-Bench, Alpaca, HumanEval, GSM8K, CNN/DM, Natural Questions

    <p align="center">
    <img src="imgs/2412.18910-1.png" width="600"></img>
    </p>

- "SpecServe: Efficient and SLO-Aware Large Language Model Serving with Adaptive Speculative Decoding" [2025-03] [[paper](https://arxiv.org/abs/2503.05096)]

  > 1. Adaptive drafter: incorporates efficiency estimation (based on historical data) into the drafting phase, achieving step-level speculative length control
  > 2. Confidence prior verifier: prioritizes the verification of tokens with high acceptance rates, achieving fine-grained request-level speculative length control
  > 3. SLO-aware Efficiency estimator: evaluates the efficiency of speculative decoding and achieves SLO (service level objective) awareness
  >
  > Experiments on: Vicuna-7B-v1.5, Vicuna-33B-v1.3, LLaMA-3.1-70B, MT-Bench, WMT14 De-En, CNN/DM, Natural Questions, GSM8K, DPR

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
