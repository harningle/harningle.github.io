---
layout:       post
title:        "rayshader: Maps in 3D"
date:         2026-09-03
tags:         visualisation
---

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/tree/main/3d_pop_distri">harningle/useful-scripts</a>)</font></p>

The 3D maps from [@researchremora](https://x.com/researchremora) look very nice. To me, they are visually more telling than the normal 2D density maps. This post tries to reproduce such maps using [rayshader](https://www.rayshader.com/).

<figure style="width: 99%">
    <figure style="display: inline-block; width: 56%; margin-bottom: 0;">
        <img src="https://pbs.twimg.com/media/FknLZnvWIBMKyQD?format=jpg&name=4096x4096">
        <p style="text-align: center; margin-bottom: 0;">(A) rayshader 3D map</p>
    </figure>
    <figure style="display: inline-block; width: 43%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/3d_pop_distri/2d.png">
        <p style="text-align: center; margin-bottom: 0;">(B) Usual 2D map</p>
    </figure>
    <p><i>Source</i>: <a href="https://x.com/researchremora/status/1606043431150592000">https://x.com/researchremora/status/1606043431150592000</a>, and Carnell, Edward J., Sam Tomlinson, and Stefan Reis. <i>UK Gridded Population at 1 Km Resolution for 2021 Based on Census 2021/2022 and Land Cover Map 2021</i>. Lancaster, UK: NERC EDS Environmental Information Data Centre. <a href="https://doi.org/10.5285/7beefde9-c520-4ddf-897a-0167e8918595">https://doi.org/10.5285/7beefde9-c520-4ddf-897a-0167e8918595</a>. The same source is used throughout the post.</p>
</figure>


## Tuning the map

Drawing maps using rayshader is easy, but making them beautiful isn't. A few parameters need tuning.

**Rescaling.** The raw population data is extremely skewed. It may look like: London = 1,000,000 and all countrysides = 10. When the two numbers are plotted on the same map, you will see a super tall London, and everything else is flat. So we need to take some transformation to make the distribution less skewed. Here I use $pop = pop^{0.9}$.

**Light Angle.** When we plot things in 3D with shade/shadow, we need to specify where the light comes from. Specifically, there are two parameters: `sunangle` and `anglebreaks`. `sunangle` controls where light originates. If we want the light to come from top left, and the shadow will be draw towards bottom right, then we need to set it to `315` (north west). `anglebreaks` specifies how high the light is and thus how long the shadow is. E.g., at noon the sunlight is perpendicular to the earth, and we have a very short shadow. When the azimuth angle is small, then the shadow is long. These two also need a bit trial and error.

**Camera.** Given the light and the object, we can take photo from difference angles/directions. They can be controlled in `rayshader::render_camera`. Again, just try a few different values and find your favourite.

This is my final output. No as good as the original twitter post, but looks nicer to be than the 2D map.

<figure style="width: 30%"><img src="https://github.com/harningle/useful-scripts/raw/main/3d_pop_distri/3d.png"></figure>
