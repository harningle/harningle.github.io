---
layout:       post
title:        "GPU rent prices"
date:         2026-09-28
tags:         [ml, visualisation]
---

I did a takehome exam for a job application a few weeks ago, and it needed a GPU, more powerful than what was sitting in my desktop. So I had to rent one. Now, a few weeks later, I'd like to write a short note just in case I need to rent anything in the future. I collect GPU prices from three main providers ([AutoDL](https://www.autodl.com/market/list), [Runpod](https://www.console.runpod.io/deploy), and [vast.ai](https://cloud.vast.ai/)) and provide some comparisons below.


## The same card is not the same price

Below is the price for each popular GPU from each provider. The faint bar behind each mark runs from the cheapest to the most expensive offer of the same card from the same provider.

<figure class="tex2jax_ignore">
    {% include gpu_rent/price_by_market.html %}
    <p style="margin-bottom: 0px;"><i>Source</i>: All prices were collected on 2026/09/15 evening. The prices on AutoDL are in Chinese Yuan, and I converted them to USD using <a href="https://data.bis.org/topics/XRU/BIS,WS_XRU,1.0/D.CN.CNY.A">BIS exchange rate</a> of 6.7105. All figures below use the same data.</p>
    <p><i>Notes</i>: One mark per provider per card, at the cheapest offer available at that moment. The faint bar behind it spans the cheapest to the most expensive offer for the same card from the same provider. On vast.ai, only single-GPU listings are counted, so "none" there means nothing was on offer one GPU at a time. RunPod has no bar because it quotes one price per card per cloud. The <i>x</i>-axis is in log.</p>
</figure>

**Between-Provider Price Difference Is Large.** E.g., an RTX 5090 has a median price of USD 0.75 on vast.ai, and USD 0.99 on Runpod secure. Such arbitrage is not free lunch. Many providers operate as a peer-to-peer marketplace. On paper it can be the same 5090, but one machine may be someone's personal gaming rig, while another one can come from a proper data centre. Runpod *secure* gives you better reliability, but at the cost of higher prices. Reliability does matter.[^unavailability] I would say the price difference is roughly the price of not having to care, plus other hardware differences. See the following point.

[^unavailability]: I personally experienced two disruptions on both AutoDL and vast.ai. On AutoDL, everything was running fine, and I shut down the instance. The next day, the instance was still there, but GPU was no longer available, so I had to back up everything and find another machine to continue the work. On vast.ai, the entire instance simply got terminated, and I lost everything.

**Within-Provider Price Difference Can Also Be Big.** The price is never GPU alone; when we rent, we rent an instance that also includes CPU, storage, etc. So of course the price of the same GPU can be different, depending on other hardware and setup. A cheap machine may have ridiculously slow network speed, slow disk, etc. Sometimes the region of the GPU also matters, e.g. almost all within-GPU price variation in AutoDL comes from region.


## What is the price of going faster?

Faster is better, but comes at a price. How big is the price? I use [llama.cpp CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013) as a measure of GPU performance. It's basically LLM inference speed: `tg128` is token generation (bandwidth-bound) and `pp512` is prompt processing (compute-bound).

<figure class="tex2jax_ignore">
    {% include gpu_rent/price_vs_speed.html %}
    <p style="margin-bottom: 0px;"><i>Source</i>: Prices are as above. Throughput is from <a href="https://github.com/ggml-org/llama.cpp/discussions/15013">ggml-org/llama.cpp#15013</a>. I took Llama 2 7B Q4_0 without flash attention, read on 2026/09/27. Three of the twelve (H200 SXM, B200, and H100 PCIe) come from runs posted in the discussion’s comments that have not yet been promoted to the main table. The figure doesn't include all GPUs because only a subset of them are in the scoreboard.</p>
    <p><i>Notes</i>: The dotted line is the cheapest offer at or above some performance. Faded marks are beaten by another cheaper and faster GPU. One mark is a GPU from a given provider, so a GPU sits at one height and scatters sideways. A faded mark is used to indicate a dominated GPU. I group GPUs into families, e.g. I assume RTX Pro 6000 Max-Q and RTX Pro 6000 Workstation have the same performance. Such a family carries a <code>~</code> prefix in the legend and in the label. The price axis is on log.</p>
</figure>

The gaming GPUs and RTX Pro 6000 sit at the frontier, if we only need *one* GPU. In some sense, this is a very unfair benchmark: this is a single stream on a single GPU, which is the worst possible case for an H100, and the best possible case for a 5090. Data centre cards earn their price on batched serving, on NVLink between multiple GPUs, on FP8, etc. Gaming GPUs have none of these. If you need one GPU, then almost surely a gaming GPU is the best. If you need multiple, then probably you need a different benchmark to decide which is the most cost-efficient.

Another caveat is training performance. The llama.cpp benchmark is inference only. The training speed may be different. I don't see a good benchmark that covers both gaming GPUs and data centre GPUs, so I can't plot a similar figure here.


## Can we even run it?

Speed is only half the question. The other half is whether the thing loads at all. A 5090 is the best tokens-per-dollar there, and it has 32GB, which means a 70B model is doomed to have "CUDA out of memory". So my last figure plots capacity: VRAM against price.

<figure class="tex2jax_ignore">
    {% include gpu_rent/price_vs_vram.html %}
    <p><i>Notes</i>: The dotted line is the cheapest offer at or above each capacity. Faded marks are beaten by another cheaper and larger GPU. The colour names the card, and the marker shape is the same as in the first figure, so a circle is vast.ai and a diamond AutoDL. Both axes are log.</p>
</figure>

VRAM per dollar is *not* monotone. It rises to a peak in the 80 to 96GB tier, at roughly 91GB per dollar per hour (an A800 80GB on AutoDL at USD 0.88, or an RTX Pro 6000 Max-Q on vast.ai at USD 1.06), and then falls off a cliff: a B200 gives 32GB per dollar and a B300 only 31. The cliff has an obvious reason: for these data centre cards, you are buying HBM, NVLink, etc. But if what you need is "a GPU that can run a 70B model", the 80 to 96GB tier is the sweet spot.

Put the two figures together, and the decision rule falls out backwards:

1. find the smallest capacity that fits your model. This is a hard constraint
2. at that capacity, take the cheapest offer, which is the frontier in the last figure
3. only then look at speed


## Caveats

**One Snapshot.** Everything here is 2026/09/15 evening. And the prices can change.

**Availability Is a Moment.** When I'm writing this, the deadline for ICLR 2027 has just passed. I heard some rumours that AutoDL almost ran out of GPUs when people were chasing the deadline. I don't have a time series to say anything, but just keep in mind that availability can be a real problem.

**AutoDL Behind the Great Wall.** For any given GPU, AutoDL is almost always the cheapest. The problem is its machines are located in China, so no internet access to GitHub, Hugging Face, Claude Code, etc.
