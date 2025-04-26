---
layout:       post
title:        "Uniform Distribution after Normalisation"
date:         2025-04-26
tags:         [stats]
---

<p><font color="#828282">(This post is largely based on <a href="https://mathoverflow.net/q/368226">https://mathoverflow.net/q/368226</a> and <a href="https://www.zhihu.com/question/658588018/answer/3574225263">https://www.zhihu.com/question/658588018/answer/3574225263</a></font></p>

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/tree/main/random_after_normalisation/dirichlet.py">harningle/useful-scripts</a>)</font></p>

When writing the [synthetic control post]({% link _posts/2024-09-25-synthetic-control-in-python-part-2.md %}), I need to generate some random weights that add up to one. A more interesting question is, how can we generate *evenly*/*uniformally* i.i.d. random variables that add up to one? This is a very common problem, e.g. I want to generate some weights which should sum up to one, or I'm slicing a pie and all pieces together are the whole pie, and I want the weights or pieces to have a uniform distribution.

**TL;DR: Dirichlet distribution is the answer. Uniform distribution after normalisation is wrong.**


## Uniform distribution on a line

Formally, the goal is to uniformally sample $n$ random variables from the set ${\big\\{x_0, \cdots, x_{n - 1} \| \sum\limits_i x_i = 1\big\\}}$. Let's consider a low dimensional case first. If $n = 2$, the problem reduces to uniformally pick $x$ and $y$ so that $x + y = 1$. This is equivalent to picking a <font color="#800000">point</font> on a line of length one $1$, and the first segment is $x$ and the other segment is $y = 1 - x$.

<figure style="width: 50%">
    <img src="/assets/images/line_1.svg">
</figure>

As long as the <font color="#800000">point</font> is sampled from $\mathcal{U}([0, 1])$, $x$ and $y$ are also uniformally distributed. Proof is trivial. Assume the distance between <font color="#800000">point</font> and the left endpoint of the line is $a$. Then by construction $x = a$ and $y = 1 - a$. Now because $a$ is from a uniform distribution, $P(x = a) = P(y = 1 - a)$, as every number between $0$ and $1$, which includes $a$ and $1 - a$, has equal probability to be sampled. So $x$ and $y$ always have the same probability to be $a$ and $1 - a, \forall a\sim\mathcal{U}([0, 1])$, which is the definition of uniform distribution. $\blacksquare$

Does this "pick a point" method work for $n > 2$ cases? No. For $n = 3$, we need two points to split a line into three segments. Suppose the distance from the lefter point to the left endpoint is $a$, and that of the righter point to the left endpoint is $b$. Then $x_0 = a$, $x_1 = b - a$, and $x_2 = 1 - b$. We want $P(x_0 = a) = P(x_1 = b - a) = P(x_2 = 1 - b)$ holding for any $0\leq a < b\leq 1$. I don't know any (joint) distribution of $a$ and $b$ that can satisfy this requirement.[^dirichlet]

[^dirichlet]: Of course there is. See the final section.

<figure style="width: 50%">
    <img src="/assets/images/line_2.svg">
</figure>


## Uniform distribution on a simplex

Now let's look the same $n = 2$ case again, but in a more general way. We want $x + y = 1$. And in higher dimensional cases, we would want $x + y + z = 1$, or $x + y + z + w = 1$, etc. This "sum up to one" constraint is essentially: pick a point on a line on 2D plane, or a point on a 2D plane in 3D space, or a point inside a 3D cube in 4D space, etc. The advantage of this way is that if we can solve the 2D case, it should be relatively easy to scale up to higher dimensions.

<figure style="width: 50%">
    <img src="/assets/images/line_3.svg">
</figure>


### The wrong way: pick a point in hypercube and then rescale/normalise

A very straightforward but wrong way to do this is by normalisation. That is, we pick $x^\*$ and $y^\*$ according to a uniform distribution, and then let ${\color{maroon}u} = \dfrac{x^\*}{x^\* + y^\*}$ and ${\color{maroon}v} = \dfrac{y^\*}{x^\* + y^\*}$. E.g., we get two realisations from a uniform distribution, say $0.2$ and $0.6$. We then normalise them to $\dfrac{0.2}{0.2 + 0.6}$ and $\dfrac{0.6}{0.2 + 0.6}$, and $(0.25, 0.75)$ is the resulting point. This construction ensures ${\color{maroon}u} + {\color{maroon}v} = 1$, and "seemingly" as long as $x^\*$ and $y^\*$ are from uniform distribution, we are good. This style/form of construction is also very common in the real world. E.g., we want to compute the population-weighted average height of a country, based on city level data. We do $\bar h = \dfrac{pop_1}{pop_1 + pop_2}\times h_1 + \dfrac{pop_2}{pop_1 + pop_2}\times h_2$. This construction is very natural. But it's wrong for this uniform purpose.[^this-purpose]

[^this-purpose]: I'm not saying weighted average has anything wrong. Of course it's correct. However, we can't use the same form of construction to get a uniform distribution on a simplex here.

First, the empirics doesn't look right. We can repeat sampling $x^\*$ and $y^\*$ from a uniform distribution multiple times and plot the resulting points $\color{maroon}(u, v)$. They do lie on the line (as the result of normalisation of course), but the distribution of the points doesn't look like uniform at all: seems we are more likely to end up in the middle of the line, and less likely to be near the endpoints. This is even more evident in the 3D case.

<figure>
    <figure style="display: inline-block; width: 46.5%; margin-bottom: 0; margin-right: 4%">
        <img src="https://github.com/harningle/useful-scripts/raw/main/random_after_normalisation/figures/uniform_wrong.svg">
        <p style="text-align: center">2D case</p>
    </figure>
    <figure style="display: inline-block; width: 46.5%; margin-bottom: 0">
        <img src="https://github.com/harningle/useful-scripts/raw/main/random_after_normalisation/figures/uniform_wrong_3d.svg">
        <p style="text-align: center">3D case</p>
    </figure>
</figure>

Second, as we do normalisation, no theory guarantees the uniformity after normalisation/division. When we transform $x^\*$ to $\dfrac{x^\*}{x^\* + y^\*}$, we divide a random variable by the sum of two random variables. Usually a nonlinear transformations of some random variables don't result in the same distribution. In this case, we can prove it's not normal. Let $Z = \dfrac{X}{X + Y}$, where $X, Y\overset{i.i.d.}{\sim}\mathcal{U}([0, 1])$. The c.d.f.

$$
\begin{align*}
P(Z\leq z) &= P\Big (\dfrac{X}{X + Y}\leq z\Big ) \\
           &= P\Big (Y\geq \dfrac{1 - z}{z}X\Big ) \\
           &= \iint_{\frac{y}{x}\geq\frac{1 - z}{z}}\text{d}x\text{d}y
\end{align*}
$$

, where $x$ and $y$ are inside $[0, 1]$. This is the area under line $y = \dfrac{1 - x}{x}$ inside the unit square.

<figure>
    <figure style="display: inline-block; width: 46.5%; margin-bottom: 0; margin-right: 4%">
        <img src="/assets/images/integral_1.svg">
        <p style="text-align: center">$z < 0.5$</p>
    </figure>
    <figure style="display: inline-block; width: 46.5%; margin-bottom: 0;">
        <img src="/assets/images/integral_2.svg">
        <p style="text-align: center">$z > 0.5$</p>
    </figure>
</figure>

So the c.d.f. of $x = \dfrac{x^\*}{x^\* + y^\*}$ is

$$
F_X(x) = \left\{ \begin{align*}
&\frac{x}{2 - 2x},  & x\leq 0.5 \\
&\frac{3x - 1}{2x}, & x > 0.5
\end{align*} \right .
$$

. This is not a normal distribution. $\blacksquare$

<figure style="width: 75%">
    <img src="https://github.com/harningle/useful-scripts/raw/main/random_after_normalisation/figures/uniform_wrong_cdf.svg">
</figure>


## One correct answer: Dirichlet distribution

Proof will come soon...

<figure>
    <figure style="display: inline-block; width: 46.5%; margin-bottom: 0; margin-right: 4%">
        <img src="https://github.com/harningle/useful-scripts/raw/main/random_after_normalisation/figures/dirichlet.svg">
        <p style="text-align: center">2D case</p>
    </figure>
    <figure style="display: inline-block; width: 46.5%; margin-bottom: 0">
        <img src="https://github.com/harningle/useful-scripts/raw/main/random_after_normalisation/figures/dirichlet_3d.svg">
        <p style="text-align: center">3D case</p>
    </figure>
</figure>
