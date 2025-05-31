---
layout:       post
title:        "“Best” papers"
date:         2024-12-25
last_updated: 2025-05-31
tags:         [econ]
---

I left economics a few years ago but it keeps "amazing" me over and over. Here are the "best" (not necessarily econ) papers I've read in each year, i.e. my "best" paper awards.


## 2025: I am Corning

**Toner-Rodgers, Aidan.** 2024. "Artificial Intelligence, Scientific Discovery, and Product Innovation." arXiv: [2412.17866](https://arxiv.org/pdf/2412.17866v1).

* "the lab [...] rollout [...] the model in May of 2022": wow, a first-year PhD student managed to convince a big firm to do a machine learning RCT before ChatGPT went popular
* how big is the firm? "1,018 scientists across 221 teams". Wow again. He managed to convince such a big firm as a first-year student
* "the effects on [...] patenting emerge after 5-6 months". Wow, five months to file a patent. Speed of light. No office politics. No corporate bureaucracy
* it's a sole author paper, with no RA, so of course there is a lot of work. He "fine-tune[d Claude 3.5]" to read the lab log. Claude 3.5 was [released in June 2024](https://en.wikipedia.org/wiki/Claude_(language_model)#Models), and the first draft of his paper was published November 2024. Wow that is MIT speed
* he even owned the domain [corningresearch.com](corningresearch.com) to fake his RCT! And apparently he [lost the dispute at WIPO](https://www.wipo.ch/amc/en/domains/decisions/pdf/2025/d2025-0410.pdf) from the real Corning
* [*QJE* R&R](https://web.archive.org/web/20250404234241/https://aidantr.github.io/)


## 2024: *feng shui* geomancer (算命先生)

**Hong, Justin Jihao, and Yuheng Zhao.** 2024. "The Costs of Leader Biases: Evidence from Superstitious Chinese Mayors." Paper Presented at [NBER Development Economics Program Meeting (2024 Fall)](https://www.nber.org/conferences/development-economics-fall-2024). [https://hjihao.github.io/justinhong_leader_jmp.pdf](https://hjihao.github.io/justinhong_leader_jmp.pdf).

> [...] when mapping local leaders' [...] unfavorable zones, we seek the assistance of established spatial astrologers and cross-validate their predictions.

> [...] zones perceived as unfavorable to mayors [...] have an average 2 percent lower GDP [...]


## 2023: Matray constant (篡改结果)

**Boissel, Charles, and Adrien Matray.** 2022. "Dividend Taxes and the Allocation of Capital." *American Economic Review* 112 (9): 2884--920. doi: [10.1257/aer.20210369](https://doi.org/10.1257/aer.20210369).

Matray divides pre-treatment DID coefficients by a factor of $1.8$ to make the unparallel trend parallel.


## 2023 honourable mention: PhD $\ll$ Master (别读博)

**Marini, Giulio, and Golo Henseke.** 2023. "Is a PhD Worth More Than a Master's in the UK Labour Market? The Role of Specialisation and Managerial Position." *Studies in Higher  Education* 48 (10): 1538--50. doi: [10.1080/03075079.2023.2254806](https://doi.org/10.1080/03075079.2023.2254806).

> Realistically, a PhD holder takes no less than 10 years (even more than 20 years for, say, those who studied STEM) to recover the costs incurred from the pursuit of a PhD
> 
> <div style="text-align: right">—<a href="https://blogs.lse.ac.uk/impactofsocialsciences/2023/10/18/what-counts-for-more-in-the-uk-job-market-a-phd-or-a-masters/">LSE blog: What counts for more in the UK job market – a PhD or a Master’s?</a></div>

If my reading is correct, you need 33 years to catch up if you do a PhD lol.


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

* this paper is very much consistent with my personal belief: if the skill differences are too big, then I may just give up, since I won't be able to beat the opponent even if I try hard
* I don't know how golf works, but in some multi-stage games, such "quitters"/"lose quickly" may be attributed to other concerns, e.g. save energy for the next stage. A very good example is that some badminton players lost game on purpose in 2012 London Olympics, so they would meet "easier" opponents in the next round and got disqualified eventually.[^bbc] I'm not saying the paper is wrong, but just it may not be easily apply to other fields
* and of course, not sure if it's reproducible at all (Babington et al., *J. Sport Econ.* [2020](https://doi.org/10.1177/152700251985940); Connolly and Rendleman, [2014](http://dx.doi.org/10.2139/ssrn.2533537))

[^bbc]: [https://www.bbc.co.uk/newsround/19075263](https://www.bbc.co.uk/newsround/19075263).


## 2021: sunlight *causes* collectivism (太阳光把我照瞎了所以只能依赖集体)

**Fredriksson, Per G., and Aatishya Mohanty.** 2021. "Sunlight and Culture." *Journal of Economic Behavior & Organization* 188: 757--82. doi: [10.1016/j.jebo.2021.05.033](https://doi.org/10.1016/j.jebo.2021.05.033).

> Populations exposed to [more sunlight] have higher incidence of [...] eye disease [...], rais[ing] the level of uncertainty avoidance [...] and [...] facilitating [...] collectivism.


## 2020: online exam cheating (作弊大师)

**Bilena, Eren, and Alexander Matros.** 2021. "Online Cheating amid COVID-19." *Journal of Economic Behavior & Organization* 182: 196--211. doi: [10.1016/j.jebo.2020.12.004](https://doi.org/10.1016/j.jebo.2020.12.004).

I've taken one and only one exam under such proctoring: my job application online assessment in 2022 for a securities firm in Shanghai. And I was convicted of cheating, because my mobile phone camera resolution was not high enough, so my computer screen wasn't very visible. Ridiculous.


## 2019: RCT intercourse (随机性交)

**Loewenstein, George, Tamar Krishnamurti, Jessica Kopsic, and Daniel McDonald.** 2015. "Does Increased Sexual Frequency Enhance Happiness?" *Journal of Economic Behavior & Organization* 116: 206--18. doi: [10.1016/j.jebo.2015.04.021](https://doi.org/10.1016/j.jebo.2015.04.021).

> We [...] randomly assigning some couples to increase their frequency of sex


## 2019 honourable mention: *I Ching* master (周易大师)

**Fisman, Ray, Wei Huang, Bo Ning, Yue Pan, Jiaping Qiu, and Yongxiang Wang.** 2023. "Superstition and Risk Taking: Evidence from 'Zodiac Year' Beliefs in China." *Management Science* 69 (9): 5174--88. doi: [10.1287/mnsc.2022.4594](https://doi.org/10.1287/mnsc.2022.4594).

> We focus on widely held beliefs in bad luck during one's "zodiac year", which occurs on a 12-year cycle around a person’s birth year, to study superstitions and risk taking.

I went to a seminar of this paper in 2019, and the author photoshopped a book with a title of "周易大师" (*I Ching* master) and the author being himself. It was very fun. I don't think it's a *Management Science* paper, but the seminar presentation was very good and very easy to follow.

One author of our [2024 best paper *feng shui* geomancer (算命先生)](#2024-feng-shui-geomancer-算命先生) is the student of one author of this paper.


## 2019 honourable mention: fairy (舞法天女)

**Ma, Chun, Fei Jiang, Feng-Tao Sheng, Yinchun Jiao, Guang-Jian Mei, and Feng Shi.** 2019. "Design and Catalytic Asymmetric Construction of Axially Chiral 3,3'-Bisindole Skeletons." *Angewandte Chemie* 58: 3014--20. doi: [10.1002/anie.201811177](https://doi.org/10.1002/anie.201811177).

<figure style="width: 50%; margin-left: auto; margin-right: auto;"><img src="/assets/images/angew_chem.png"></figure>

I thought chemistry would use more serious cover images...


## 2018: bridges (修桥)

**Brooks, Wyatt, and Kevin Donovan.** 2020. "Eliminating Uncertainty in Market Access: The Impact of New Bridges in Rural Nicaragua." *Econometrica* 88 (5): 1965--97. doi: [10.3982/ECTA15828](https://doi.org/10.3982/ECTA15828).

> We build bridges that eliminate this risk [that] unpredictable flash floods ... cut
[villages] off from outside food, labor and product markets for days or weeks at a time. 

A bridge costs $40,000, and they built six bridges.

In [old working paper version](https://economics.yale.edu/sites/default/files/nicaraguabridges_web.pdf), they wrote "[w]e build bridges that eliminate this risk", but the published version above writes "[w]e study the impact of new bridges that eliminate this risk." I love the old one. Sounds richer.


