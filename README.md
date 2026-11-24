# UGP-Former

Compact PyTorch implementation of UGP-Former for pre-duty EEG fatigue-risk prediction.
The release includes the model, uncertainty inference, adaptive augmentation and training loss.

## Configuration

| Setting | Value |
| --- | --- |
| Probabilistic embedding | Kernels 5 and 25; 24 features per branch; input prior variance 0.001 |
| Confidence gate | Initial alpha 1.0 and beta 0.5 |
| PLG backbone | 7 blocks; hidden dimension 48; 4 attention heads; FFN dimension 96 (2 × hidden dimension) |
| MC inference | 20 passes through the classifier with dropout 0.3 |
| Adaptive augmentation | Controller 96 → 48 → 6 with GELU; 5 bands: 1–4, 4–8, 8–13, 13–30 and 30–45 Hz |
| Loss | Cross-entropy + 1.0 × KL consistency loss |

The embedding propagates variance using the first-order GELU derivative. In
`predict_with_uncertainty`, the backbone is evaluated once and only the classifier dropout
is sampled. The returned probabilities are the MC mean. Total uncertainty is the entropy
of that mean, data uncertainty is the mean entropy of the samples, and model uncertainty
is their difference (mutual information). There is no separate output-variance head.
The default model has 207,652 inference parameters, excluding the training-only augmenter.

The augmentation controller pools the feature mean and variance over time and outputs five
band attenuation strengths and one noise scale. It is updated together with the network
by the total loss.

The manuscript evaluation uses batch size 32, AdamW (learning rate 0.001, weight decay
0.01), 100 epochs and seeds 2021–2025. Each LOSO fold uses 28 training, 3 validation and
1 test participant; standardization is fitted on the training participants and the
checkpoint is selected by validation NLL.

## Quick check

```bash
pip install -r requirements.txt
python ugp_former.py
```

The command checks forward propagation, backpropagation and default MC inference.
