<!-- source-page: 1 -->
# Membership Inference Attacks From First Principles

Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr

1 Google Research  
2 University of Massachusetts Amherst

\* Authors ordered alphabetically.

## Abstract

A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g. `<= 0.1%`) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

![Figure 1](./figures/figure-p001-01.png)

Fig. 1: Comparing the true-positive rate vs. false-positive rate of prior membership inference attacks reveals a wide gap in effectiveness. An attack's average accuracy is not indicative of its performance at low FPRs. By extending on the most effective ideas, we improve membership inference attacks by 10x, for a non-overfit CIFAR-10 model (92% test accuracy).

## I. Introduction

Neural networks are now trained on increasingly sensitive datasets, and so it is necessary to ensure that trained models are privacy-preserving. In order to empirically verify if a model is in fact private, membership inference attacks have become the de facto standard because of their simplicity. A membership inference attack receives as input a trained model and an example from the data distribution, and predicts if that example was used to train the model.

Unfortunately, many prior membership inference attacks use an incomplete evaluation methodology that considers average-case success metrics such as accuracy or ROC-AUC. Those metrics aggregate an attack's success over an entire dataset and over all decision thresholds. Privacy, however, is not an average-case metric, and should not be evaluated as such. This paper argues that prior attacks often do not measure the worst-case privacy of machine learning models.

The paper's main contribution is to revisit membership inference from first principles. The authors argue that attacks should be evaluated by considering their true-positive rate (TPR) at low false-positive rates (FPR), because this is the regime that matters in security-sensitive settings. If an attack can reliably identify even a few users in a sensitive dataset, it has succeeded. Conversely, an attack that only looks good under aggregate metrics should not be considered successful.

When evaluated this way, most prior attacks fail in the low-FPR regime. Aggregate metrics such as AUC are often poorly correlated with low-FPR performance. The authors then introduce the Likelihood Ratio Attack (LiRA), which combines per-example difficulty estimates with a calibrated Gaussian likelihood model. They report that LiRA is roughly 10x stronger than prior work at low false-positive rates, and release code at `https://github.com/tensorflow/privacy/tree/master/research/mmi_lira_2021`.

<!-- source-page: 2 -->
## II. Background

This section introduces notation and situates membership inference among privacy attacks on trained models.

### A. Machine learning notation

A classification neural network `f_theta : X -> [0,1]^n` maps an input sample `x` to an `n`-class probability distribution, and `f(x)_y` denotes the probability assigned to class `y`. Given a dataset `D` sampled from an underlying distribution `D`, the notation `f_theta <- T(D)` means that the training algorithm `T` learns parameters `theta` from `D`.

Training proceeds with stochastic gradient descent:

```tex
\theta_{t+1} \leftarrow \theta_t - \eta \sum_{(x,y)\in B} \nabla_\theta \ell(f_\theta(x), y)
```

For classification, the main loss is cross-entropy:

```tex
\ell(f_\theta(x), y) = -\log(f_\theta(x)_y)
```

The paper focuses on realistically trained, well-generalizing models rather than intentionally weak or heavily overfit models. For the remainder of the paper, the authors train state-of-the-art classifiers with regularization, tuned learning rates, and strong data augmentation.

### B. Training data privacy

Training data privacy concerns attacks that leak information about individual training examples. The paper briefly distinguishes several families:

- `Privacy attacks`: attacks on the privacy of the training data, including training-data extraction, model inversion, and property inference.
- `Theory of memorization`: the view that membership inference is tied to a model's ability to memorize individual data points or labels.
- `Privacy-preserving training`: methods such as differentially private training, as well as heuristic defenses that attempt to reduce leakage.
- `Measuring training data privacy`: auditing schemes that use strong privacy attacks to quantify privacy leakage directly.

The authors argue that the most common practical measurement technique is still membership inference, and that stronger attacks are needed for that measurement to be trustworthy.

## III. Membership Inference Attacks

The objective of a membership inference attack is to predict whether a specific training example was, or was not, used as training data in a particular model. This section formalizes the task and introduces the evaluation methodology used throughout the paper.

<!-- source-page: 3 -->
### A. Definitions

The paper defines a standard security game between a challenger `C` and an adversary `A`:

1. The challenger samples a training dataset `D <- D` and trains a model `f_theta <- T(D)`.
2. The challenger flips a bit `b`. If `b = 0`, it samples a fresh point `(x,y) <- D` such that `(x,y) notin D`. Otherwise it samples a point `(x,y)` from the training set.
3. The challenger sends `(x,y)` to the adversary.
4. The adversary gets query access to the distribution `D` and to the model `f_theta`, and outputs a bit `b'`.
5. The adversary wins if `b' = b`.

For simplicity, the paper writes `A(x,y)` for the adversary's prediction on sample `(x,y)` when the model and distribution are clear from context. Many attacks first compute a real-valued membership confidence score `A'(x,y)` and then threshold it:

```tex
A(x,y) = 1[A'(x,y) > \tau]
```

As a first illustrative attack, the paper considers the LOSS attack of Yeom et al. Because models are trained to minimize the loss on their training examples, examples with lower loss are more likely to be members. Formally:

```tex
A_{\text{loss}}(x,y) = 1[-\ell(f(x), y) > \tau]
```

### B. Evaluating membership inference attacks

The paper argues that balanced attack accuracy is inadequate for privacy evaluation. Balanced accuracy is symmetric and treats false positives and false negatives equally, but practical privacy failures are driven by confident positive identifications. A method that correctly identifies a tiny, highly sensitive subset may be much more damaging than a method that performs slightly above chance on average.

Balanced accuracy is also an average-case metric. Two attacks can have the same balanced accuracy while having radically different privacy implications. The paper gives the example of an attack that perfectly identifies `0.1%` of users and guesses randomly on the rest, versus another attack that succeeds with `50.05%` probability on every user. The second may have the same balanced accuracy, but the first is far more useful to an adversary.

For the LOSS attack on the paper's CIFAR-10 model, balanced accuracy is about `60%`. That sounds strong, but the attack is effectively useless at confidently identifying members: at `0.1%` FPR it achieves `0%` TPR.

<!-- source-page: 4 -->
### ROC analysis

Instead of balanced accuracy, the paper recommends using the tradeoff between true-positive rate and false-positive rate. ROC curves characterize this tradeoff across all thresholds, and the authors emphasize reporting full ROC curves on logarithmic scales, as well as TPR at fixed low FPRs such as `0.1%` or `0.001%`.

![Figure 2](./figures/figure-p004-01.png)

Fig. 2: ROC curve for the LOSS baseline membership inference attack, shown with both linear scaling (left) and log-log scaling (right) to emphasize the low-FPR regime.

The paper notes that AUC can be especially misleading because it averages over all false-positive rates, including regimes that are irrelevant for privacy evaluation. In Figure 2, the LOSS attack does not beat random chance for any FPR below roughly `20%`, and is therefore ineffective at confidently breaching privacy.

## IV. The Likelihood Ratio Attack (LiRA)

### A. Membership inference as hypothesis testing

The paper reframes membership inference as a hypothesis test: distinguish a world where the target example `(x,y)` was included in training from one where it was not. Let `Q_in(x,y)` be the distribution of models trained on datasets that contain `(x,y)`, and `Q_out(x,y)` be the distribution of models trained on datasets that do not.

Given a target model `f` and example `(x,y)`, the adversary performs a likelihood-ratio test. In the paper's simplified one-dimensional instantiation, the statistic is the model loss on the target example, and the attack compares the likelihood of that observed loss under the `in` and `out` distributions.

```tex
\Lambda(f; x,y) = \frac{p(\ell(f(x), y)\mid \tilde{Q}_{\text{in}}(x,y))}{p(\ell(f(x), y)\mid \tilde{Q}_{\text{out}}(x,y))}
```

This yields a principled attack score rather than a single global loss threshold.

### B. Memorization and per-example hardness

![Figure 3](./figures/figure-p005-01.png)

Fig. 3: Some examples are easier to fit than others, and some have larger separability between their losses when they are members versus non-members. The paper trains 1024 models on random subsets of CIFAR-10 and plots losses for four examples.

![Figure 4](./figures/figure-p005-manual-02.png)

Fig. 4: The model's confidence, or its logarithm (the cross-entropy loss), are not normally distributed. Applying the logit function yields values that are approximately normal.

The authors show that examples vary along at least two axes: how easy they are to fit, and how much their member and non-member loss distributions separate. Some points are outliers, some are inherently hard, and these effects are not captured by a single global threshold. This explains why prior attacks that threshold only the observed loss can fail.

### C. Estimating the likelihood-ratio test with parametric modeling

LiRA estimates `Q_in` and `Q_out` using shadow models. The paper advocates a parametric Gaussian model for the transformed confidence values, which lets the attack estimate likelihoods robustly with fewer shadow models than nonparametric alternatives. The paper also emphasizes the importance of modeling each example separately.

<!-- source-page: 6 -->
### Algorithm 1: Online LiRA

The paper's online attack trains shadow models both with and without the target example and then compares the target model's observed confidence against the estimated Gaussian distributions:

```text
Require: model f, example (x,y), data distribution D
confs_in  = {}
confs_out = {}
repeat N times:
  sample a shadow dataset D_attack from D
  train f_in  on D_attack U {(x,y)}
  train f_out on D_attack \ {(x,y)} or on a shadow dataset not containing (x,y)
  add phi(f_in(x)_y)  to confs_in
  add phi(f_out(x)_y) to confs_out
estimate mu_in, mu_out, sigma_in^2, sigma_out^2
observe conf_obs = phi(f(x)_y)
return the likelihood ratio between N(mu_in, sigma_in^2) and N(mu_out, sigma_out^2)
```

The paper's offline variant avoids retraining models around each target point. It trains shadow models in advance, estimates only the `out` distribution, and switches to a one-sided hypothesis test:

```tex
A = 1 - \Pr[Z > \phi(f(x)_y)], \quad Z \sim \mathcal{N}(\mu_{\text{out}}, \sigma_{\text{out}}^2)
```

The online attack is stronger but more computationally expensive. The offline attack is cheaper and remains highly effective.

## V. Attack Evaluation

The paper evaluates LiRA across CIFAR-10, CIFAR-100, ImageNet, and WikiText-103, with additional appendix results on Purchase and Texas. The experiments focus on the low-FPR regime that the paper argues is the right operating point for privacy evaluation.

<!-- source-page: 7 -->
![Figure 5](./figures/figure-p007-01.png)

Fig. 5: Success rate of the online attack on CIFAR-10, CIFAR-100, ImageNet, and WikiText. All plots are generated with 256 shadow models, except ImageNet which uses 64.

![Figure 6](./figures/figure-p007-manual-02.png)

Fig. 6: Success rate of the offline attack on CIFAR-10, CIFAR-100, ImageNet, and WikiText. All plots are generated with 128 `OUT` shadow models, except ImageNet which uses 32. The paper also plots the online attack with the same number of shadow models for comparison.

### A. Online attack evaluation

Figure 5 presents the main results for the online attack. Across the four datasets, LiRA achieves true-positive rates ranging from around `0.1%` to `10%` at `0.001%` FPR, and substantially larger TPR at `0.1%` FPR. The authors note that even for complex datasets, the attack is practical because the underlying models can still be trained relatively quickly.

An important empirical finding is that low-FPR vulnerability is not explained solely by train-test gap. When comparing image datasets, average-case success correlates with generalization gap, but low-FPR success can behave differently. In particular, CIFAR-10 models can be easier to attack than ImageNet models at low FPR despite better apparent generalization.

### B. Offline attack evaluation

Figure 6 shows that the offline attack is only slightly weaker than the online attack. At `0.1%` FPR, the offline TPR is within roughly `20%` of the best online attack using the same number of shadow models.

### C. Re-evaluating prior membership inference attacks

The paper re-evaluates prior work under the same low-FPR protocol and finds that earlier methods perform far worse than their average-case numbers suggest. The strongest prior attack in this setting is the per-example thresholding approach of Sablayrolles et al., but LiRA still substantially outperforms it.

<!-- source-page: 8 -->
![Table I](./figures/table-p008-01.png)

Table I: Comparison of prior membership inference attacks under the same settings for well-generalizing models on CIFAR-10, CIFAR-100, and WikiText-103 using 256 shadow models. Accuracy is shown only for completeness; the paper argues it is not a meaningful privacy metric.

The paper reviews several ingredients of prior attacks:

- `Shadow models`: train an auxiliary classifier on outputs from many shadow models.
- `Multiple queries`: query the target several times under perturbations or augmentations.
- `Per-class hardness`: use class-dependent thresholds.
- `Per-example hardness`: learn a threshold specific to each example.

The paper finds that per-class thresholds do not materially improve low-FPR success. Per-example thresholds help more, but prior nonparametric versions are brittle in the low-FPR regime. LiRA improves on them by combining per-example modeling with a parametric Gaussian likelihood test.

<!-- source-page: 9 -->
![Figure 7](./figures/figure-p009-01.png)

Fig. 7: Attack true-positive rate versus model train-test gap for a variety of CIFAR-10 models.

![Table II](./figures/table-p009-01.png)

Table II: Breakdown of how various components build up from the simple LOSS attack to the paper's stronger offline and online attacks on CIFAR-10.

### D. Membership inference and overfitting

To understand the relationship between vulnerability and overfitting, the paper plots TPR at `0.1%` FPR against model train-test gap. While models with larger train-test gap are generally more vulnerable, models with identical gap can still differ by up to `100x` in vulnerability. In the appendix, the paper further shows that more accurate models can be more vulnerable than less accurate ones.

## VI. Ablation Study

The attack consists of several interacting pieces. The ablation study shows how each contributes to low-FPR performance.

<!-- source-page: 10 -->
![Figure 8](./figures/figure-p010-01.png)

Fig. 8: The best scoring metrics ensure the output distribution is approximately Gaussian, and the worst metrics are not easily modeled with a standard distribution.

![Figure 9](./figures/figure-p010-manual-02.png)

Fig. 9: Attack success rate increases as the number of shadow models increases, with diminishing returns after roughly 64 models.

### A. Logit scaling the loss function

The first step of the attack projects the model confidence to a logit scale so that the resulting statistic is closer to Gaussian. The paper considers both an unstable and a stable logit implementation and reports that the stable variant is more robust in practice. If logits are available directly, a hinge-loss-style statistic performs similarly well and is numerically simpler.

### B. Gaussian distribution fitting

A central advantage of LiRA is that it uses a parametric Gaussian model. This reduces the number of shadow models required and makes low-FPR extrapolation more stable than nonparametric density estimation.

<!-- source-page: 11 -->
![Table III](./figures/table-p011-01.png)

Table III: Querying on augmented versions of the image doubles the true-positive rate at low false-positive rates, with most benefits obtained from only two queries.

![Figure 10](./figures/figure-p011-01.png)

Fig. 10: The attack's success rate on CINIC-10 remains nearly unchanged when the shadow-model training data is disjoint from the target model's training data, so long as both datasets come from the same distribution.

### C. Number of queries

Models are often trained with data augmentations, so it is natural to query the target on augmented variants of the same input. The paper evaluates up to 162 augmentations on CIFAR-10 and finds that most of the gain comes from just two queries. Eighteen queries perform about as well as all 162.

### D. Disjoint datasets

The paper separates the target's training dataset `D_train` from the attacker's dataset `D_attack`. When `D_attack` is disjoint but drawn from the same distribution, performance barely changes. When the attacker trains shadow models on a different distribution, the attack weakens but still remains surprisingly effective.

### E. Mismatched training procedures

The paper next varies the target architecture, optimizer, and augmentation while letting the attacker guess these properties when training shadow models.

<!-- source-page: 12 -->
![Figure 11](./figures/figure-p012-01.png)

Fig. 11: The attack succeeds even when the adversary is uncertain about the target model's training setup. It performs best when the attacker guesses correctly, but remains strong under moderate mismatch.

![Figure 12](./figures/figure-p012-manual-02.png)

Fig. 12: The attack succeeds against real state-of-the-art CIFAR-10 models. The strongest results come from matching the target architecture, but even mismatched shadow architectures remain useful.

The paper reports that larger models tend to be more vulnerable than smaller ones, and that the attacker's guess of the data augmentation has the largest effect on final performance. The attack is still useful even when the attacker guesses architecture or optimizer incorrectly.

## VII. Additional Investigations

### A. Attacking real-world models

The experiments so far used models trained by the authors. The paper also attacks released CIFAR-10 models from Phan. It trains 256 shadow models by subsampling CIFAR-10 and evaluates two settings: one where the attacker knows the target architecture and another where the attacker uses a different architecture. The same qualitative lessons hold: matching helps, but mismatched shadows still provide a strong attack.

<!-- source-page: 13 -->
![Figure 13](./figures/figure-p013-01.png)

Fig. 13: Out-of-distribution training examples are less private.

### B. Why are some examples less private?

To study example-level privacy, the paper injects out-of-distribution and mislabeled examples into training and measures a privacy score based on how distinguishable the `in` and `out` distributions are for each example.

The main finding is that out-of-distribution or incorrectly labeled examples are substantially easier to detect as training members. Samples drawn from CINIC-10 and then inserted into CIFAR-10 are less private than normal CIFAR-10 examples, and intentionally mislabeled examples are even less private. Interpolations between those extremes land in between.

## VIII. Conclusion

Membership inference attacks should be evaluated in the low-FPR regime. LiRA is presented as a concrete attack that succeeds much more often than prior methods under that operating point. The paper argues that stronger attacks are necessary both for measuring privacy leakage and for understanding what current defenses do, and do not, protect against.

The conclusion raises several open questions for future work:

- Whether previously proposed defenses actually prevent low-FPR attacks.
- Whether attacks using less information, such as label-only methods, remain effective in the low-FPR regime.
- Whether stronger attacks can serve as better privacy metrics for real systems.

## Acknowledgements

The authors thank Thomas Steinke, Dave Evans, Reza Shokri, Sanghyun Hong, Alex Sablayrolles, Liwei Song, Matthias Lecuyer, and the anonymous reviewers for comments on drafts of the paper.

<!-- source-page: 14 -->
## References

The bibliography spans source pages 14 to 16 in the rendered page images. Those pages were reviewed manually during transcription, but the reference list itself is left in image form here to keep the Markdown concise and avoid transcription mistakes in dozens of long entries.

<!-- source-page: 15 -->
References continue on this page.

<!-- source-page: 16 -->
References continue on this page. The final entries are references `[70]` through `[73]`, followed by Appendix A.

## Appendix A. Additional Experiments

### A. Attacking DP-SGD

The paper studies membership inference against models trained with differentially private SGD. It varies the clipping norm and noise multiplier and reports both model accuracy and attack effectiveness.

![Table IV](./figures/table-p016-01.png)

Table IV: Accuracy of the models trained with DP-SGD on CIFAR-10 under different noise parameters.

![Figure 14](./figures/figure-p016-manual-01.png)

Fig. 14: Effectiveness of using DP-SGD against the paper's attack with different privacy budgets.

The appendix concludes that even very small amounts of DP noise can substantially reduce attack effectiveness, though this can also hurt model accuracy. Training with very small noise is already an effective defense in these experiments, despite corresponding to loose provable `epsilon` bounds.

<!-- source-page: 17 -->
### B. White-box attacks

Prior work suggested that membership inference becomes easier if the adversary has white-box access to the target model and can inspect gradients. The paper compares its confidence-based attack to variants that additionally use gradient norms.

![Figure 15](./figures/figure-p017-01.png)

Fig. 15: Comparison of the white-box attack using the paper's approach to the black-box setting.

The appendix reports that gradient norms can improve overall AUC, but do not help in the low-FPR regime that the paper cares about. Using model confidences remains preferable there.

## Appendix B. Additional Figures and Tables

### A. Attack performance versus model accuracy

![Figure 16](./figures/figure-p017-manual-02.png)

Fig. 16: Attack true-positive rate versus model test accuracy.

### B. Full ROC curves for Gaussian distribution fitting

![Figure 17](./figures/figure-p017-manual-03.png)

Fig. 17: Effect of varying the number of models trained on attack success rates. Estimating the mean per-example is useful, but with few shadow models it is much more effective to assign all examples the same variance.

<!-- source-page: 18 -->
### C. Comparison to prior work on additional datasets

Similarly to Figure 1 for CIFAR-10, the appendix compares the paper's attack against prior work on additional datasets: CIFAR-100, WikiText-103, Texas, and Purchase.

![Figure 18](./figures/figure-p018-01.png)

Fig. 18: ROC curve of prior membership inference attacks, compared to the paper's attack, on CIFAR-100.

![Figure 19](./figures/figure-p018-manual-02.png)

Fig. 19: ROC curve of prior membership inference attacks, compared to the paper's attack, on WikiText-103. Prior attacks that rely on model features are omitted because they were not designed for sequential models.

![Figure 20](./figures/figure-p018-manual-03.png)

Fig. 20: ROC curve of prior membership inference attacks, compared to the paper's attack, on the Texas dataset.

![Figure 21](./figures/figure-p018-manual-04.png)

Fig. 21: ROC curve of prior membership inference attacks, compared to the paper's attack, on the Purchase dataset.

<!-- source-page: 19 -->
### D. Attack ablations on additional datasets

The appendix repeats the ablation study on additional datasets. The same overall pattern holds: combining per-example thresholds, logit scaling, Gaussian likelihood estimation, and multiple queries yields the strongest attacks. On datasets such as WikiText-103, Texas, and Purchase, the models are trained without data augmentations, so query augmentation is not used there.

![Table V](./figures/table-p019-01.png)

Table V: Breakdown of how various components build up to obtain the paper's best attacks on CIFAR-100.

![Table VI](./figures/table-p019-02.png)

Table VI: Breakdown of how various components build up to obtain the paper's best attacks on WikiText-103.

![Table VII](./figures/table-p019-03.png)

Table VII: Breakdown of how various components build up to obtain the paper's best attacks on Texas.

![Table VIII](./figures/table-p019-04.png)

Table VIII: Breakdown of how various components build up to obtain the paper's best attacks on Purchase.

### E. Full ROC curves for mismatched training procedures

This appendix section plots full ROC curves for the experiments where the attacker must guess the target model's architecture, optimizer, and augmentation procedure.

<!-- source-page: 20 -->
![Figure 22](./figures/figure-p020-manual-01.png)

Fig. 22: Different architectures with momentum optimizer and mirror and shift as augmentation.

![Figure 23](./figures/figure-p020-manual-02.png)

Fig. 23: Different optimizers on WRN28-10 with mirror and shift as augmentation.

![Figure 24](./figures/figure-p020-manual-03.png)

Fig. 24: Different augmentations on WRN28-10 with momentum optimizer.
