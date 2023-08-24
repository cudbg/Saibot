# Saibot VLDB 2023

https://arxiv.org/abs/2307.00432

Recent data search platforms rely on ML task-based utility measures rather than metadata-based keyword search to search among a large corpus of datasets. Users submit a training dataset and  these platforms search for augmentations - join or union-compatible datasets - that, when used to augment the user's dataset, most improve model (e.g., linear regression) performance. Although effective, providers that manage personally identifiable data demand differential privacy (DP) guarantees before allow these platforms access to their data.   Unfortunately, making data search differentially private is nontrivial, as a single search can involve training and evaluating datasets hundreds or thousands of times, and quickly deplete available privacy budgets.  


We present Saibot, a differentially private data search platform that employs a new DP mechanism called FDP which calculates sufficient semi-ring statistics to support ML models over different combinations of datasets. These factorized sufficient statistics are privatized once by the data provider, and can be freely reused by the platform. This allows Saibot to scale to arbitrary numbers of datasets in the corpus and numbers of requests, while minimizing the amount that DP noise affects search results. We optimize the sensitivity of FDP for commmon augmentation operations, and analyze its properties with respect to linear regression models.  Specifically, we develop an unbiased estimator for many-to-many joins, prove its bounds, and develop  and optimization to redistribute DP noise to minimize the impact on the model. Our evaluation on a real-world dataset corpus of $329$ datasets demonstrates that Saibot can return augmentations that achieve model accuracy within 50-90% of non-private search, while the leading alternative DP mechanisms (LDP, GDP, SDP) are several orders of magnitude worse.


# Usage

Please see the demo.ipynb notebook for the example codes of Saibot.

The technical report is available in /tech.
