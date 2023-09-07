# Adversarial-SimCSE

Implementation of SimCSE with adopting adversal training loss.

Experimented on Roberta-base with supervised learning setting

batch 512 w/o adv obtained from https://arxiv.org/pdf/2104.08821.pdf

The performance strongly based on the batch size, so increasing the batch size will definitely go over SIMCSE accuracy. Only increasing 40 batch size increased average performance +0.5 as show below comparing Batch 100 with adv and Batch 60 with adv

\[
\text{Loss} = \text{contrastive loss} + \text{weight} \times \text{adversarial loss}
\]

contrastive loss is computed as following NTX Loss with one postive pair and negative pair
Adversarial loss is computed by applying smoothness inducing adversarial regularizaation with adding pertubation to the embeddings.  https://arxiv.org/pdf/1911.03437v5.pdf 

| Configuration       | Test STSB | Test SICK | STS12  | STS13  | STS14  | STS15  | STS16  | AVG    | Best STSB |
|---------------------|-----------|-----------|--------|--------|--------|--------|--------|--------|-----------|
| Batch 512 w/o adv   | 0.8583    | 0.8050    | 0.7653 | 0.8521 | 0.8095 | 0.8603 | 0.8257 | 0.8253 | 0.8583    |
| Batch 100 with adv  | 0.8706    | 0.8078    | 0.7513 | 0.8515 | 0.8083 | 0.8613 | 0.8215 | 0.8246 | 0.8706    |
| Batch 60 with adv   | 0.8649    | 0.7786    | 0.7591 | 0.8466 | 0.8056 | 0.8617 | 0.8248 | 0.8202 | 0.8649    |
| Batch 60 w/o adv    | 0.8543    | 0.7737    | 0.7431 | 0.8091 | 0.7732 | 0.833  | 0.8004 | 0.7981 | 0.8543    |
