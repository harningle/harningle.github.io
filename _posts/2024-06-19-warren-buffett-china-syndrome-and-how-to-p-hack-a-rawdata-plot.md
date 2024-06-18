---
layout:       post
title:        Warren Buffett, China Syndrome, and how to $p$-hack a rawdata plot
date:         2024-06-19
tags:         [econ, visualisation, p-hacking]
---

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/blob/main/p_hacking/index_line_chart.py">harningle/useful-scripts</a>)</font></p>


An extremely common and powerful way to motivate (or distort) a story is by (cherry picking) stylised facts. And this very post is motivated by a chart I saw recently on Twitter, comparing Warren Buffett's Berkshire Hathaway and S&P 500. I was a bit shocked: over the past 20 years, Buffett underperformed S&P 500.

<figure>
    <img src="https://pbs.twimg.com/media/GMrYPrPWwAAazrt?format=png">
    <p><i>Source</i>: <a href="https://twitter.com/MikeZaccardi/status/1786477995215011907">https://twitter.com/MikeZaccardi/status/1786477995215011907</a></p>
</figure>

## How does the "rawdata" look like

First of all, I try to collect the data behind this chart above. Berkshire Hathaway [website](https://www.berkshirehathaway.com/letters/2023ltr.pdf) has all we need. I plot the gains if we invested 1 dollar in <font color="#000080">Berkshire</font> and <font color="#800000">S&P 500</font> back in 1965, when Berkshire was founded.

<figure>
    <figure style="display: inline-block; width: 49.5%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/brk_spx.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) linear scale</p>
    </figure>
    <figure style="display: inline-block; width: 49.5%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/brk_spx_log.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) log scale</p>
    </figure>
    <p><i>Source</i>: <a href="https://www.berkshirehathaway.com/letters/2023ltr.pdf">Buffett's letter to shareholders of Berkshire Hathaway Inc., 2023, p.17</a></p>
</figure>

**Scales of Axes Matter a Lot, Visually.** Panel (A) shows the vanilla line chart, and <font color="#000080">Berkshire</font> completely destroys <font color="#800000">S&P 500</font>. However, simply changing $y$-axis to $\log$ scale can easily "reduce" the performance gap between the two in Panel (B). This trick is very common in econ papers, e.g. showing bar charts with $y$-axis *not* starting from $0$, stretching $y$-axis to make make coefficient plots nicer, transforming $x$-axis into [ticks with unequal intervals](https://junkcharts.typepad.com/junk_charts/2023/01/if-you-blink-youd-miss-this-axis-trick.html) etc.


## Cherry pick the sample period

Just to make sure there is no factual error, let's zoom into 2004-2023, and Panel (A) below gets effectively the same chart as that on Twitter. So no factual errors; the graph on Twitter does faithfully plot the original rawdata. More interestingly, I find *many* time spans where <font color="#800000">S&P 500</font> was more profitable than <font color="#000080">Berkshire</font>.

<figure>
    <figure style="display: inline-block; width: 49.5%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/brk_spx_2004_2023.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) 2004-2023</p>
    </figure>
    <figure style="display: inline-block; width: 49.5%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/brk_spx_2023.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) 2023 full year</p>
    </figure>
    <p><i>Source</i>: <a href="https://www.berkshirehathaway.com/letters/2023ltr.pdf">Buffett's letter to shareholders of Berkshire Hathaway Inc., 2023, p.17</a>, Yahoo Finance (<a href="https://finance.yahoo.com/quote/SPY/history/">S&P 500</a>, <a href="https://finance.yahoo.com/quote/BRK-A/history/">Berkshire Hathaway</a>)</p>
</figure>

**Even Rawdata Plots Are Carefully Picked.** Unlike [Matray constant](http://localhost:4000/2024/06/03/why-dont-you-do-a-phd.html), cherry picking can be totally legal and very natural. If you read

> "we look at the performance of xxx over the past 20 years", or
> 
> "last year, xxx outperformed yyy",

do you think these lines are natural or will you have any doubt? Keep in mind that everything we see are cherry picked. Cherry picking and $p$-hacking are not necessarily stilted, and the authors always try their best to make them flow as natural as possible, so as to cheat the referees and editors and get the paper published.


## The China Syndrome?

Cherry picking the sample period reminds me of the seminal paper by Autor et al. (*AER* [2013](https://doi.org/10.1257/aer.103.6.2121)). They basically blame China for unemployment in the US. The first figure in their paper indicates a strong negative correlation between US's imports from China and the employment rate in US. I successfully reproduce the chart (with some difference) from scratch, i.e. not using their replication package.[^mistake]

[^mistake]: I think I made a mistake somewhere in my replication. The trend/slope of my import penetration blue line is almost identical to the original paper, but the level is 10x smaller than theirs. Maybe a unit conversion mistake on my side. But I wasn't able to figure out where.

<figure>
    <figure style="display: inline-block; width: 51.5%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/adh2013.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) Figure 1 in Autor et al. (<i>AER</i> <a href="https://doi.org/10.1257/aer.103.6.2121">2013</a>)</p>
    </figure>
    <figure style="display: inline-block; width: 47.5%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/adh2013_own.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) our replication</p>
    </figure>
    <p style="margin-bottom: 0px;"><i>Source</i>: U.S. Bureau of Economic Analysis, U.S. Bureau of Labor Statistics, U.S. Census Bureau (Civilian Labor Force Level [<a href="https://fred.stlouisfed.org/series/CLF16OV">CLF16OV</a>], All Employees, Manufacturing [<a href="https://fred.stlouisfed.org/series/MANEMP">MANEMP</a>], Gross Domestic Product [<a href="https://fred.stlouisfed.org/series/GDP">GDP</a>], Imports of Goods and Services [<a href="https://fred.stlouisfed.org/series/IMPGS">IMPGS</a>], Exports of Goods and Services [<a href="https://fred.stlouisfed.org/series/EXPGS">EXPGS</a>], U.S. Imports of Goods by Customs Basis from China [<a href="https://fred.stlouisfed.org/series/IMPCH">IMPCH</a>]), retrieved from FRED, Federal Reserve Bank of St. Louis</p>
    <p><i>Notes</i>: U.S. Imports of Goods by Customs Basis from China [<a href="https://fred.stlouisfed.org/series/IMPCH">IMPCH</a>] is not seasonally adjusted, while other series are. I take a dirty and quick 12-month moving average to “remove” the seasonality, and then aggregate it to quarterly level to match the frequency of other series. The notes apply to figures below as well.</p>
</figure>

However, if we look at the entire time series of US employment, the rise in unemployment seems to have nothing to do with China; the downward trend has been there since day 1...

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/adh2013_own_manu.svg">
    <p style="margin-bottom: 0px;"><i>Source</i>: Same as above</p>
</figure>

Now the full picture. If I plot the entire time series of both US unemployment and import from China, do you still think there is any correlation between them?

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/p_hacking/figure/adh2013_own_full.svg">
    <p style="margin-bottom: 0px;"><i>Source</i>: Same as above</p>
</figure>

That being said, I still like Autor et al. (*AER* [2013](https://doi.org/10.1257/aer.103.6.2121)) very much. Wang et al. (NBER Working Paper [2018](https://www.nber.org/papers/w24886)) is also worth reading: after taking supply chain/general equilibrium into consideration, and imports from China actually *boost up* employment in US.
