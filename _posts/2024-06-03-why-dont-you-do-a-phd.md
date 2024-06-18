---
layout:       post
title:        "Why don't you do a Ph.D.?"
date:         2024-06-03
last_updated: 2024-06-19
tags:         econ
---


<p><font color="#828282">(Constantly updated)</font></p>

I decided not to do a Ph.D. and started job search very abruptly in early 2022. This is partly due to financial reasons. Since I'd never thought about industry before, everything on my CV was RA, and zero internship. So one very fair question that I got and keep getting in almost every job interview is "why don't you do a Ph.D.?"[^answer] Here are the real reasons:

[^answer]: If you are also considering quitting, make sure you have a good answer to this. My very personal suggestion is, make sure you *do* get some (good) offers. So your answer can be: "oh you know I do easily get these grad school offers, but I'm no longer interested." That is, convince people that Ph.D. is *within your constraint* and quitting is your *choice*, instead of Ph.D. being *beyond your capability*! The opposite would be: "no I don't like academia", but then the interviewer might see you as an incompetent/failed student.

* Matray constant $\mathscr{m} = 1.8$
    + feel free to shrink your DID pre-treatment coefficients by a factor of $1.8$ to get a nice pre-trend, and this is "unintentional and made in good faith"
    + [retraction](https://www.aeaweb.org/articles?id=10.1257/aer.113.7.2053), author's [webpage](https://web.archive.org/web/20240603021537/https://sites.google.com/view/adrienmatray/papers?authuser=0), author's responses ([1](https://www.dropbox.com/scl/fi/264cyt3m3e9frkldsvgch/statement_AER.pdf?rlkey=2i7isi0tp4iq170igz3tdrdsd&dl=0), [2](https://www.dropbox.com/s/jq1bmy6q39fmlcu/supplemental_note.pdf?dl=0), [3](https://www.dropbox.com/s/skenqoxmygjxfcr/retraction_note_for_website.pdf?dl=0), [4](https://www.dropbox.com/s/xfrdxsso8k1mmux/Matray_2023.pdf?dl=0)), [EJMR](https://www.econjobrumors.com/topic/the-great-matray-scam-2023)
* Vuillemey's lost regression
    + write but don't run the code, and then fake whatever numbers you want in the draft. Finally "original data and code that produced the published results" can be conveniently lost
    + [retraction](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13064), *JoF* [response](https://afajof.org/2021/07/answers-to-faq-about-the-recent-retraction-of-an-article-in-the-jf/), [Retraction Watch](https://retractionwatch.com/2021/07/08/researchers-forfeit-10000-award-when-papers-findings-cant-be-replicated/), [EJMR](https://www.econjobrumors.com/topic/jf-article-is-retracted)
* Giuliano and Spilimbergo's missing data
    + fabricate the findings until someone catches you. Then "the original codes and data sets" are again conveniently lost
    + [retraction](https://doi.org/10.1093/restud/rdac085), Bietenbeck and Thiemann's (*JAE* [2023](https://doi.org/10.1002/jae.2970)) replication study
* LaCour's never-existing data
    + you don't even need any data to write a field experiment paper! Run analysis on "simulated" data and get a *Science* paper!
    + [retraction](https://www.science.org/doi/10.1126/science.aac6638), author's [response](https://retractionwatch.com/wp-content/uploads/2015/05/LaCour_Response_05-29-2015.pdf), [Wikipedia](https://en.wikipedia.org/wiki/When_contact_changes_minds)
* Heshmati's imputation
    + no data? Excel autofill will help you!
    + [retraction](https://doi.org/10.1016/j.jclepro.2023.138092), [Retraction Watch](https://retractionwatch.com/2024/02/05/no-data-no-problem-undisclosed-tinkering-in-excel-behind-economics-paper/)
* Persson and Rossin-Slater's literature review
    + if you don't cite any prior studies, then you are "the first" one to study this question!
    + [Retraction Watch](https://retractionwatch.com/2016/05/26/economists-go-wild-over-overlooked-citations-in-preprint-on-prenatal-stress/)
* $\log(NAICS)$
    + oh don't forget you can take the log of categorical vars.
    + [JMP from Wayback Machine](https://web.archive.org/web/20150616021934/http://faculty.chicagobooth.edu/workshops/financelunch/pdf/siriwardanejmp.pdf)
* Reproducibility never there
    + 60% replication success rate is considered "relatively high" for experimental papers (Camerer et al., *Science* [2016](https://www.science.org/doi/10.1126/science.aaf0918))
    + if you do macro, luck you! 10% is the bar (McCullough et al., *JMCB* [2006](https://doi.org/10.1353/mcb.2006.0061))
        - a seminal example is Reinhart and Rogoff (*AER: P&P* [2010](https://doi.org/10.1257/aer.100.2.573)). Herndon et al. (*Cambridge J. Econ* [2014](https://doi.org/10.1093/cje/bet075)) invalidate the original paper in a very solid way
    + political economy? Maybe an even lower bar (Wiebe, *Research & Politics* [2024](https://doi.org/10.1177/20531680241229875))
        - I've personally looked into this promotion tournament literature as well. You can easily find (many) rawdata errors and/or totally unreasonable summary stat. in Chen and Kung (*QJE* [2019](https://doi.org/10.1093/qje/qjy027)), Kung and Chen (*APSR* Expression of Concern [2023](https://doi.org/10.1017/S0003055423000060)), Landry et al. (*CPS* [2018](https://doi.org/10.1177/0010414017730078)), Li et al. (*EJ* [2019](https://doi.org/10.1093/ej/uez018)), and Yao and Zhang (*JEG* [2015](https://doi.org/10.1007/s10887-015-9116-1))
* *p*-hacking is all you need
    + we always see $p = 0.049$ but never see $p = 0.051$ (Brodeur et al, *AER* [2016](http://dx.doi.org/10.1257/app.20150044); *AER* [2020](https://doi.org/10.1257/aer.20190687))
    + want to publish with null results? In your dreams! (Chopra et al., *EJ* [2024](https://doi.org/10.1093/ej/uead060))
* Oh you also need connections
    + Azoulay et al. (*QJE* [2010](https://doi.org/10.1162/qjec.2010.125.2.549); *AER* [2019](https://doi.org/10.1257/aer.20161574)), Brogaard et al. (*JFE* [2014](https://doi.org/10.1016/j.jfineco.2013.10.006)), Carrell et al. (*JPE* [Forthcoming](https://doi.org/10.1086/730208)), Colussi (*REStat* [2018](https://doi.org/10.1162/REST_a_00666)), Rubin and Rubin (*JPE* [2021](https://doi.org/10.1086/715021)), Zinovyeva and Bagues (*AEJ: Applied* [2015](https://doi.org/10.1257/app.20120337))
* Try to live as long as possible! As a paper can take ~20 years before getting published
    + Smith et al. (*QJE* [2021](https://doi.org/10.1093/restud/rdab001)): 24 years. Authors' story [here](https://x.com/LonesSmith/status/1369763777537847298)
    + Martinez-Bravo et al. (*AER* [2022](https://doi.org/10.1257/aer.20181249)): 15 years. Authors' story [here](https://mp.weixin.qq.com/s/7b6YHc5cRuPUO2KeamwHgA). One RA's story [here](https://weibo.com/7214391486/L2TAnq9QL)
        - fun fact: (at least) one RA of the paper already got tenured when the paper was accepted
* "Authors are in reverse alphabetical order"
    + find a very similar paper? Threaten the authors to add you as a coauthor
    + author's [statement](https://sites.google.com/site/oyvindthomassen/authorship)
* Mental health
    + MIT suicides [here](https://web.archive.org/web/20150716045000/http://web.mit.edu/~sdavies/www/mit-suicides/) and [here](https://thetech.com/2015/09/22/nickolaus-v135-n23), [Harvard suicide](https://www.thecrimson.com/article/1996/6/22/economics-doctoral-student-commits-suicide-pdmitry/), [LSE suicides](https://x.com/nunopgpalma/status/1535190269754220545), [Emmanuel Farhi](https://en.wikipedia.org/wiki/Emmanuel_Farhi), [Alan Krueger](https://en.wikipedia.org/wiki/Alan_Krueger#Death_and_legacy), [Martin Weitzman](https://en.wikipedia.org/wiki/Martin_Weitzman#Research)
    + Bolotnyy et al. (*JEL* [2022](https://doi.org/10.1257/jel.20201555)), Macchi et al. (*IZA Discussion Paper* [2023](https://elisamacchi.github.io/publication/econ-mental-health/)), Sahm (blog post [2020](https://web.archive.org/web/20200729023354/https://macromomblog.com/2020/07/29/economics-is-a-disgrace/) and the updated [version](https://web.archive.org/web/20240315205918/https://macromomblog.com/2020/07/29/economics-is-a-disgrace/))
* Publish or perish
    + only slightly more than half of the graduates from a top 5 school can publish at least one top 5 paper, ~12 years after graduation (Grove and Wu, *AER P&P* [2007](https://doi.org/10.1257/aer.97.2.506)) [students admitted in 1989]
    + a median graduate in top 30 US schools publishes ~0 *AER*-equivalent paper, six years after graduation (Conley and Önder, *JEP* [2014](http://doi.org/10.1257/jep.28.3.205)) [students graduating in 1986 - 2000]
    + the acceptance rate in top 5 is 6% (Card and DellaVigna, *JEL* [2013](https://doi.org/10.1257/jel.51.1.144)) [papers submitted in 2012]
    + one top 5 paper boosts up the tenure probability by 80% (Heckman and Moktan, *JEL* [2020](https://doi.org/10.1257/jel.20191574)) [tenure track professors in top 35 schools in 1996-2010]
    + this is even worse for females (Hengel, *EJ* [2022](https://doi.org/10.1093/ej/ueac032))
