---
layout:       post
title:        "Hot summer in Switzerland"
date:         2026-06-28
tags:         [stat]
---

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/tree/main/hot_summer">harningle/useful-scripts</a>)</font></p>

I just don't understand why we don't have AC in Switzerland...


## Heat-related death in Switzerland is of a similar magnitude to gun death in the US

I saw a post somewhere claiming that the heat-related death rate in Europe is roughly the same as the gun-violence death rate in the US. I was a bit surprised and checked the Swiss data. I know there can be plenty of measurement error, but it seems to be *true*![^measurement-error]

[^measurement-error]: I feel the heat-related death data should be OK, but the gun-death data not so much. It may suffer from underestimation.

<figure>
    {% remote_include https://raw.githubusercontent.com/harningle/useful-scripts/refs/heads/main/hot_summer/figures/us_gun_vs_ch_heat.html %}
    <p style="margin-bottom: 0px;"><i>Source</i>: CDC WONDER. 2018--2024. <i>Multiple Cause of Death, 2018-2024.</i> Atlanta, GA: United States Department of Health and Human Services. <a href="https://wonder.cdc.gov/mcd-icd10-expanded.html">https://wonder.cdc.gov/mcd-icd10-expanded.html</a>. Federal Office for the Environment. 2026. <i>Heat-Related Deaths.</i> Bern, Switzerland: Federal Department of Environment, Transport, Energy and Communications. <a href="https://www.bafu.admin.ch/en/indicators">https://www.bafu.admin.ch/en/indicators</a>.</p>
    <p><i>Notes</i>: The gun-death series includes the following ICD-10 codes: W32, W33, W34 (accidental), X93, X94, X95 (homicide), Y22, Y23, Y24 (undetermined intent), and Y35 (legal intervention). We exclude suicide here. The shade is the 95% confidence interval.</p>
</figure>


## It's getting hotter and hotter

Many people say Switzerland doesn't have AC because it wasn't hot in the past. Indeed. But it has been getting astonishingly hot recently in Geneva!

<figure>
    {% remote_include https://raw.githubusercontent.com/harningle/useful-scripts/refs/heads/main/hot_summer/figures/hot_days_heatmap.html %}
    <p style="margin-bottom: 0px;"><i>Source</i>: MeteoSwiss. 2026. <i>Climate Stations - Homogeneous Data Series.</i> Zurich, Switzerland: Federal Department of Home Affairs. <a href="https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-nbcn/items/gve?.asset=asset-ogd-nbcn_gve_d_historical-csv">https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-nbcn/items/gve?.asset=asset-ogd-nbcn_gve_d_historical-csv</a>.</p>
    <p><i>Notes</i>: The colour indicates the number of days with the highest temperature falling inside the temperature range in the year. This figure was generated on 2026/06/28, so we only have partial data for 2026; all previous years have full-year data.</p>
</figure>


## Switzerland is a net electricity exporter in the summer

One reason against AC is the environment. I don't think it makes sense electricity-wise. Switzerland has been a net electricity exporter during the summer for quite a few years.

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/hot_summer/figures/net_export_all_years.svg">
    <p style="margin-bottom: 0px;"><i>Source</i>: Swiss Federal Office of Energy. 2026. <i>Schweizerische Elektrizitätsbilanz - Monatswerte.</i> Bern, Switzerland: Federal Department of Environment, Transport, Energy and Communications. <a href="https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/electricity-statistics.html/">https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/electricity-statistics.html/</a>.</p>
    <p><i>Notes</i>: This figure plots each year's net export value since 1990. It overlays year on year with transparency. The more transparent a region, the fewer years have data there.</p>
</figure>

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/hot_summer/figures/production_consumption_all_years.svg">
    <p style="margin-bottom: 0px;"><i>Source</i>: Same as above.</p>
    <p><i>Notes</i>: Same as above, but for production and consumption.</p>
</figure>

You can argue much of the electricity comes from hydro, which eventually is from melting glaciers and snow, so not that green. You can argue that, beyond electricity, Switzerland is a net importer of energy overall. I agree AC is not green. But would you rather pollute the environment and enjoy a 28-degree room temperature, or sleep at 31 degrees and not use AC?[^31-not-hypothetical]

[^31-not-hypothetical]: This 31 degrees Celsius isn't hypothetical. My thermometer right now shows my bedroom is at 31 Celsius, at 2:20 a.m., with windows and door open.


## My tips

**Use Libraries.** Many libraries have AC or are water-cooled, and most are open to the public for free, even if you are not a student.[^genilac] I've personally verified that [Graduate Institute Library](https://www.graduateinstitute.ch/library), [Uni MAIL](https://www.unige.ch/biblio/en/infos/locations/mail/), and [Uni CMU](https://www.unige.ch/biblio/en/infos/locations/cmu/) are all cool. But note they may not be open at weekends as it's summer vacation.

[^genilac]: I very much like [GeniLac](https://ww2.sig-ge.ch/entreprises/offres-energies/thermique-renouvelable/reseau_genilac) from SIG. Basically they use cool water from Lake Geneva to cool down the buildings. Check their video [here](https://www.youtube.com/watch?v=GUUP2PeGx1Y).

**Avoid Local Heavy Compute.** If possible, move local compute, especially those machine learning ones, to some free cloud like Colab or Kaggle. That helps *a lot*. For me, the biggest heat generator is my GPU.

**Bernoulli Effect.** If you do feel outside is much cooler than your room, directing your fan to blow towards *outside* works way better than towards *inside*. Let the Bernoulli effect do its job.
