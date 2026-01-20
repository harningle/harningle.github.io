---
layout:       post
title:        "Geneva public transportation: Is it really punctual?"
date:         2026-01-20
tags:         [stat]
---

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/tree/main/sbb">harningle/useful-scripts</a>)</font></p>

Trains and public transportation in general are super good in Switzerland. I've lived in Zurich and Geneva for over three years. My experience with trains is great. However, the buses in Geneva are less good though: they often arrive and depart *too early*. Sometimes the timetable says the bus would depart at 10:30 but actually it leaves on 10:29. I have missed a lot of buses due to this. I'm going to see if the data is in line with my experience.


## Trains: Yes they are really punctual

[Actual data](https://data.opentransportdata.swiss/en/dataset/ist-daten-v2) lists all bus and train journeys every day, with detailed scheduled time, actual arrival and departure time, and stop location. To showcase the data, the figure below shows the distribution of train arrival delays in year 2025, by operator. Unsurprisingly, Deutsche Bahn (DB) is the worst, with a median delay of 100 seconds, and a super fat right tail.[^late-db-trains-blocked] Meanwhile, the median delay is only 31 seconds for SBB trains.

[^late-db-trains-blocked]: It's so bad that some trains from Germany to Switzerland, if delay too much, can be terminated in the border city, and passengers need to transfer to SBB trains (see SBB announcement [here](https://news.sbb.ch/medien/artikel/113076/einzelne-verbindungen-von-und-nach-deutschland-mit-umsteigen) and [here](https://news.sbb.ch/medien/artikel/136768/ic-stuttgart-zuerich-neue-massnahmen-fuer-puenktlichere-zuege)). I was once stuck on a DB train towards Cologne for over an hour for "vandalism". The last time I saw this word is on a GRE vocabulary book lol.

<figure style="width: 60%">
    <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/sbb/train_delay_by_operator.svg">
    <p style="margin-bottom: 0px;">><i>Source</i>: Real Time Data - CUS. 2025. Actual Data and Actual Data v2. Bern: Business office SKI. <a href="https://data.opentransportdata.swiss/en/dataset/ist-daten-v2">https://data.opentransportdata.swiss/en/dataset/ist-daten-v2</a>.</p>
    <p><i>Notes</i>: I only keep train stops within Switzerland, and filter out arrival status "Unbekannt" (unknown). I drop stations with a total train stop of less than 10,000 in 2025. I winsorise the delay that are (1) more than two minutes early, and (2) more than five minutes late. This is why we see some bunching in the left and right endpoints in the figure. Median delay in seconds are in parenthesis.</p>
</figure>


## My bus

There is one and only bus that directly connects my flat with my office, so it's my only commute option. I list the share of buses that are *early* in departure time below. I define "early" as it leaves one minute or earlier than the timetable. Every dot is a scheduled departure time, and for each scheduled departure, I averaged across the early departure across all weekdays in the whole year 2025. Take the scheduled departure of 9:49 for an example: out of ~200 weekdays in 2025, the bus departed early on 18.9% of the days. It's quite sizeable. If I leave around 8.30 a.m., I will miss the bus due to early departure once every four days. If I work from home and go to the office after lunch, then almost every other day I will miss the bus.

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/sbb/bus_early_departure_share.svg">
    <p style="margin-bottom: 0px;"><i>Source</i>: Real Time Data - CUS. 2025. Actual Data and Actual Data v2. Bern: Business office SKI. <a href="https://data.opentransportdata.swiss/en/dataset/ist-daten-v2">https://data.opentransportdata.swiss/en/dataset/ist-daten-v2</a>. TPG. 2026. Vos horaires par arrêt pour la ligne 22. <a href="https://www.tpg.ch/en/lignes/22">https://www.tpg.ch/en/lignes/22</a>.</p>
    <p><i>Notes</i>: I only keep the scheduled departures on non-holiday weekdays. The line is a cubic spline.</p>
</figure>

I guess I need to leave home earlier then...
