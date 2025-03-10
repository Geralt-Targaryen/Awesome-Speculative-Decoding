# Awesome-Speculative-Decoding

Reading list on speculative decoding.

## History

## Draft Models

## Retrieval-based Speculative Decoding

- "Speculative Decoding for Multi-Sample Inference" [2025-03] [[paper](https://arxiv.org/abs/2503.05330)]

        SD method tailored for multi-sample reasoning scenarios, such as self-consistency and Best-of-N sampling.

        "for any partial sequence on the path $i$, we use its $k$-token suffix as a query to search for matching prefixes in other paths"

        Experiments on: Qwen2.5-7B-Instruct, LLaMA-3-8B-Instruct, GSM8K, MATH

## Verification Strategies

## Draft Length Control

- "SpecServe: Efficient and SLO-Aware Large Language Model Serving with Adaptive Speculative Decoding" [2025-03] [[paper](https://arxiv.org/abs/2503.05096)]

        1. Adaptive drafter: incorporates efficiency estimation (based on historical data) into the drafting phase, achieving step-level speculative length control
        2. Confidence prior verifier: prioritizes the verification of tokens with high acceptance rates, achieving fine-grained request-level speculative length control
        3. SLO-aware Efficiency estimator: evaluates the efficiency of speculative decoding and achieves SLO (service level objective) awareness

        Experiments on: Vicuna-7B-v1.5, Vicuna-33B-v1.3, LLaMA-3.1-70B, MT-Bench, WMT14 De-En, CNN/DM, Natural Questions, GSM8K, DPR
