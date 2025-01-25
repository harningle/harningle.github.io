---
layout:       post
title:        "Rent prices in Geneva"
date:         2025-01-25
tags:         [visualisation]
---

I moved to Geneva at the beginning of this year and now find the rent prices are ridiculously high. I knew Geneva was expensive but didn't expect that expensive. My prior is mainly based on my renting experience in Zurich and London. I did my Master's in Zurich three years ago: a nice two-bedroom apartment for students cost only CHF 1,400 in total, everything included. My two-bedroom apartment in London cost GBP 2,000 last year, excluding heating and electricity. I thought Geneva would be slightly more expensive but I was wrong. Now I feel it's the time to update my prior for housing in Geneva.


**False Belief 1.** Geneva is more expensive than Zurich

Many people told me renting in Geneva was more expensive than in Zurich.[^rental-subsidy] But the truth is the opposite. Switzerland has pretty good data, from the very famous [MeteoSwiss](https://www.meteoswiss.admin.ch/), to the rent we are studying now.[^ci] Federal Statistical Office actually has a webpage for rents [here](https://www.bfs.admin.ch/bfs/en/home/statistics/construction-housing/dwellings/rented-dwellings.html). For example, in year 2022, the average rent in Zurich is CHF 1,654, and Geneva's figure is 1,504. So Geneva is expensive, but it's cheaper than Zurich, at least on average.

[^rental-subsidy]: This can have huge spatial heterogeneity. More than one friends told me that area near the international organisations was priced higher than they should have been, due to the [rental subsidy](https://info.undp.org/gssu/onlinetools/RentalSubsidy/RentalSubsidy.aspx). And this is the area I'm mainly searching for, so...

[^ci]: I would strongly recommend [MeteoSwiss](https://www.meteoswiss.admin.ch/). Its weather forecast even has confidence intervals! The visualisation on their webpage as well as mobile phone application is also among the best, if not the best.

<figure>
    <iframe src="https://datawrapper.dwcdn.net/8f22b4a69528c5be6e22f0967aa02965/6/" style="width: 100%; aspect-ratio: 4/3; border: 0"></iframe>
    <p><i>Source</i>: <a href="https://www.bfs.admin.ch/bfs/en/home/statistics/construction-housing/dwellings/rented-dwellings.html">https://www.bfs.admin.ch/bfs/en/home/statistics/construction-housing/dwellings/rented-dwellings.html</a></p>
</figure>


**False Belief 2.** Geneva is expensive after flat sharing

The truth is, if a friend and you can live together, Geneva is only a bit more expensive than London. The bar chart above looks at the "average" apartment. What does it mean? How many people can live in an "average" apartment? The map below partly addresses the second problem. It shows, for each canton and each year, how much a 3-4 room apartment costs on average. Generally speaking, two to three people can comfortably live in such apartments.[^room] In 2022, such an apartment only costs CHF 1,530 in the *canton* of Geneva, which translates to CHF 500 to 700 per bedroom. That is, if two people share a two-bedroom apartment, the rent is going to be round CHF 700 per month. This is not very unreasonable.

[^room]: It's not clear to me what a "room" is here. Funnily, there is no "legal" definition of a "room" in Switzerland now. Generally speaking, 12m<sup>2</sup> and 6m<sup>2</sup> are two thresholds. If a room is smaller than 6m<sup>2</sup>, it's not counted as a room. Between 6 and 12m<sup>2</sup>, it's 0.5 room. Above 12m<sup>2</sup> means a full room. This was defined in [DIN 283](https://www.irbnet.de/daten/rswb/82079080132.pdf) in 1951. However, it's a German law and it has already been [withdrawn](https://www.dinmedia.de/en/regulation/woflv/69686892). What's even funnier is that Geneva counts rooms differently from other cantons: it counts a kitchen as a room, while other cantons generally don't. The underlying data for the map is [Structural Survey](https://www.bfs.admin.ch/bfs/en/home/statistics/population/surveys/se.html), and the [questionnaire](https://dam-api.bfs.admin.ch/hub/api/dam/assets/33427128/master) doesn't mention any size requirements for a "room". All it says is that a kitchen or toilet does not qualify for a room. So I guess here 3-4 room apartment means one living room plus two to three bedrooms,

<figure>
    <iframe src="https://www.atlas.bfs.admin.ch/maps/13/de/17839_11973_175_169/27613.html" frameborder="0" style="width: 100%; aspect-ratio: 4/3; cursor: auto;"></iframe>
    <p><i>Source</i>: <a href="https://www.atlas.bfs.admin.ch/maps/13/de/17839_11973_175_169/27613.html">https://www.atlas.bfs.admin.ch/maps/13/de/17839_11973_175_169/27613.html</a></p>
</figure>

However, keep in mind that the CHF 1,500 for two people is not the *city* of Geneva, but the *canton* of Geneva, which includes less central places like Versoix. So the average rent in the central places, e.g. [Servette](https://maps.app.goo.gl/RZdzYf8XNTEL8Rgo8) (1202) or [Plainpalais](https://maps.app.goo.gl/BFUbNDksmwcqqm9bA) (1205), is going to be way higher than CHF 1,500. I don't know how big the premium of city vs suburb is. For London it's roughly 30% to 100% (zone 1 vs zone 2) in my experience. If we apply the same premium to Geneva, the rent per person will be around CHF 1,000 to 1,500. This is still high but not unacceptable. Thus, flat sharing can be a very budget-friendly option.


**False Belief 3.** CHF 1,500 to 2,000 is common for one person

People often tell me housing in Geneva is expensive and everyone is paying a lot. But really? I thought Geneva was more expensive than Zurich, but the data says no. It can well be the case that people hold wrong (first-order and/or second-order) beliefs on the average rent.[^pluralistic-ignorance] I searched a bit but couldn't find very good survey or census data on housing. The best publicly available data is Federal Population Census 2011, available at [IPUMS](https://international.ipums.org/international-action/sample_details/country/ch#tab_ch2011a). I get all people who work in Geneva and who live in Geneva, and plot how much they pay below. The average rent for a two-room apartment, which I assume is one bedroom plus one living room, is around CHF 1,100 to 1,200 in year 2011. From 2011 to 2025 January, the [rental price index](https://www.bfs.admin.ch/bfs/en/home/news/special-coverage/housing-rents.assetdetail.33787151.html) increases by roughly 16-17% in Switzerland. This 2011 number thus translates to CHF 1,300 to 1,400 in 2025.[^assumption]

[^pluralistic-ignorance]: This can be [pluralistic ignorance](https://en.wikipedia.org/wiki/Pluralistic_ignorance). E.g., only people paying a high rent will complain, and then other people hear their complaints and think all other people are paying a lot. So a common conversation can be: "I pay 1,000 for a studio. But you know what? I'm luck to get a cheap one. Everyone around me is complaining about the 2,000 rent, so be prepared for the high rent." But the reality may be 99% people paying CHF 1,000 and only 1% paying CHF 2,000.

[^assumption]: I'm assuming rents in Geneva increases at the speed of Switzerland country average. May or may not be true.

<figure>
    {% remote_include https://raw.githubusercontent.com/harningle/useful-scripts/refs/heads/main/geneva_rent/rent_distri.html %}
    <p style="margin-bottom: 0px;"><i>Notes</i>: The rawdata is <a href="https://international.ipums.org/international-action/sample_details/country/ch#tab_ch2011a">3.5% Federal Population Census 2011</a>. I drop households with missing room number or rent info. Households without canton info. are also dropped. Finally, only people who work in a canton and her departure location to the office is in the same canton is kept.</p>
    <p style="margin-bottom: 0px;"><i>Source</i>: Ruggles, Steven, Lara Cleveland, Rodrigo Lovaton, Sula Sarkar, Matthew Sobek, Derek Burk, Dan Ehrlich, Quinn Heimann, and Jane Lee. 2024. Integrated Public Use Microdata Series, International: Version 7.5 [dataset]. Minneapolis, MN: IPUMS. doi: <a href="https://www.ipums.org/projects/ipums-international/d020.v7.5">10.18128/D020.V7.5</a>.</p>
    <p><i>Acknowledgement</i>: I wish to acknowledge the statistical office that provided the underlying data making this research possible: Federal Statistical Office, Switzerland.</p>
</figure>

One advantage of the population census data here is that it partly address the city vs canton problem. We only look at people who live in a canton and who work in the same canton. If one wants to live very far and spend three hours on commute, so be it. I believe most people will live within a reasonable distance to their office, so the rent distribution is not necessarily biased towards the cheap end.

[foxai.ch](https://foxai.ch/tables) is another very good source for rent distribution. The [author](https://www.reddit.com/r/Switzerland/comments/1h2jekz/hi_reddit_i_made_a_map_to_visually_explore_the/) collects data from rental agencies or platforms, and produces the rent distribution at zip code level. This is based on very recent 2023 and 2024 data, much newer than 2011 census. Definitely worth checking before you sign the lease contract! However, the data sounds like flow, rather than stock. On one hand, this is the market we face: maybe people who come three years earlier find a flat of CHF 500, but all we have is what the régies supply now. The stock doesn't matter too much. On the other hand, there is always some possibility of lease takeover so stock sometimes can be redistributed...


## My sad story

I finally ended up with a one-living room one-bedroom apartment at the price of CHF 1,500, not including charges. It was a lease takeover. My flat search was not easy at all. I sent 41 flat visit requests in total, most of them were via Facebook groups and the others on the rent platforms like [Homegate](https://www.homegate.ch/en) or [immobilier.ch](https://www.immobilier.ch/en). I got 19 responses, and visited 17 of them (two were already gone by then time I asked). I applied to all of them, and got two offers. Interestingly, the two are in the same building. I accepted the cheaper one.
