---
layout:       post
title:        "Ridge plot"
date:         2025-10-30
tags:         [visualisation]
---

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/blob/main/ridge_plot/ridge_plot.py">harningle/useful-scripts</a>)</font></p>


We often want to visualise distributions by subgroup, e.g. income distribution by gender, or firm size distribution by firm age, etc. A usual way is to simply plot all distributions in the same figure, with each subgroup in different colours or different line types, like below.

<figure style="width: 60%">
    <img src="/assets/images/ridge_plot_bad.svg">
    <p><i>Source</i>: Segarra and Teruel (<i>JEBO</i><a href="https://doi.org/10.1016/j.jebo.2012.02.012">2012</a>), Figure 4</p>
</figure>

However, this is not very eye friendly. A fancier way is to use a ridge plot. The figure is much easier to digest, especially when the subgroups are time related. In the below example, we clearly see, as times passes, the median doesn't change a lot, but the density goes from one mode to multi modal.

<figure style="width: 35%">
    <img src="/assets/images/ridge_plot_good.png">
    <p><i>Source</i>: Voth and Yanagizawa-Drott (CEPR Working Paper <a href="https://cepr.org/publications/dp19219">2024</a>), Figure 9</p>
</figure>

To illustrate how to make a ridge plot, I take the data from "[how good is 'good'](https://yougov.co.uk/society/articles/21568-how-good-good)". We have a list of words, such as "great", "perfect", etc. For each word, we ask people to give a score from 0 to 10, where 10 means very good, and 0, very bad. In the end, the data we have look like

| Score                         | Good | Great | Perfect |
|-------------------------------|------|-------|---------|
| 0                             | 1    | 0     | 0       |
| 1                             | 0    | 0     | 0       |
| 2                             | 3    | 5     | 0       |
| ...                           | ...  | ...   | ...     |
| 9                             | 30   | 33    | 10      |
| 10                            | *25* | 35    | 80      |

, where *25* means 25 people give a score of 10 for the word "good".

For each word, we make a kernel density plot, and then arrange all the plots from top to bottom, i.e. put them in a 3-row 1-col. figure.

<figure style="width: 40%">
    <img src="/assets/images/ridge_plot_1.svg">
    <p><i>Notes</i>: Data behind this figure is synthetic, not the original data from <a href="https://yougov.co.uk/society/articles/21568-how-good-good">how good is "good"</a></p>
</figure>

The trick is then to reduce the vertical gaps between the subfigures, so visually the distribution overlaps with each other a bit. We can also fill the distribution with some semi-transparent colour, so that the figures look like 3D ish.

<figure style="width: 70%">
     <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ridge_plot/ridge_plot.svg">
    <p><i>Source</i>: <a href="https://d3nkl3psvxxpe9.cloudfront.net/documents/YouGov_Word_Sentiment.pdf">YouGov Word Sentiment</a> and <a href="https://d3nkl3psvxxpe9.cloudfront.net/documents/YouGov_Word_Sentiment_2.pdf">YouGov Word Sentiment 2</a></p>
</figure>
