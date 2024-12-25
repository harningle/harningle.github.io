---
layout:       post
title:        "“Best” papers"
date:         2024-12-25
last_updated: 2024-12-25
tags:         [econ]
---

I left economics a few years ago but it keeps "amazing" me over and over. Here are the "best" (not necessarily econ) papers I've read in each year, i.e. my "best" paper awards.


## 2024: *feng shui* geomancer (算命先生)

**Hong, Justin Jihao, and Yuheng Zhao.** 2024. "The Costs of Leader Biases: Evidence from Superstitious Chinese Mayors." Paper Presented at [NBER Development Economics Program Meeting (2024 Fall)](https://www.nber.org/conferences/development-economics-fall-2024). [https://hjihao.github.io/justinhong_leader_jmp.pdf](https://hjihao.github.io/justinhong_leader_jmp.pdf).

> [...] when mapping local leaders' [...] unfavorable zones, we seek the assistance of established spatial astrologers and cross-validate their predictions.

> [...] zones perceived as unfavorable to mayors [...] have an average 2 percent lower GDP [...]


## 2023: Matray constant (篡改结果)

**Boissel, Charles, and Adrien Matray.** 2022. "Dividend Taxes and the Allocation of Capital." *American Economic Review* 112 (9): 2884--920. doi: [10.1257/aer.20210369](https://doi.org/10.1257/aer.20210369).

Matray divides pre-treatment DID coefficients by a factor of $1.8$ to make the unparallel trend parallel.


## 2023 honourable mention: PhD $\ll$ Master (别读博)

**Marini, Giulio, and Golo Henseke.** 2023. "Is a PhD Worth More Than a Master's in the UK Labour Market? The Role of Specialisation and Managerial Position." *Studies in Higher  Education* 48 (10): 1538--50.

> Realistically, a PhD holder takes no less than 10 years (even more than 20 years for, say, those who studied STEM) to recover the costs incurred from the pursuit of a PhD
> 
> <div style="text-align: right">—<a href="https://blogs.lse.ac.uk/impactofsocialsciences/2023/10/18/what-counts-for-more-in-the-uk-job-market-a-phd-or-a-masters/">LSE blog: What counts for more in the UK job market – a PhD or a Master’s?</a></div>

If my calculation is correct, you need 33 years to catch up if you do a PhD lol.


## 2022: masturbation (正太控)

**Karl, Andersson.** 2022. "I Am Not Alone -- We Are All Alone: Using Masturbation as an Ethnographic Method in Research on *shota* Subculture in Japan." *Qualitative Research*. doi: [10.1177/14687941221096600](https://doi.org/10.1177/14687941221096600
).

> I [...] read[...] the comics in the same way as my research participants had told me that they did it: while masturbating.

The paper was retracted. It can be found in [Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/b/b0/I_am_not_alone_(Andersson_paper).pdf).


## 2022 honourable mention: `torch.manual_seed(3407)` is all you need (调随机数种子)

**Picard, David.** 2023. "`torch.manual seed(3407)` Is All You Need: On the Influence of
Random Seeds in Deep Learning Architectures for Computer Vision." arXiv: [2109.08203](https://arxiv.org/abs/2109.08203).

> [I]t is surprisingly easy to find an outlier [random seed] that performs much better or much worse than the average.

I actually quite like this paper, not in a sarcastic way:

* even with "modern" optimisers, we still easily get stuck in a bad local optimal if we start with a bad random initialisation
* pre-training works but does not 100% solves the above problem
* some random seeds work better than others, but the difference is not very big though. A small improvement may thus not be a real improvement, but a lucky seed
* my prior is that many SOTAs in CV are created by p hacking random seeds...


## 2022 honourable mention: adverse incentive (打不过就开摆)

**Brown, Jennifer.** 2011. "Quitters Never Win: The (Adverse) Incentive Effects of Competing with Superstars." *Journal of Political Economy* 119 (5): 982--1013. doi: [10.1086/663306](https://doi.org/10.1086/663306).

If the skill differences are too big, then I may decide to give up since I won't be able to beat the opponent even if I try hard.
