---
layout:       post
title:        "Synthetic control in Python"
date:         2024-08-20
tags:         [econ]
---

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/tree/main/synth">harningle/useful-scripts</a>)</font></p>

**TL;DR: CVXPY in Python is 800x faster than `synth` in Stata.**

Recently I was asked to do something using synthetic control in my day job. The official Stata commands [`synth`](https://web.stanford.edu/~jhain/synthpage.html) and [`sdid`](https://github.com/Daniel-Pailanir/sdid) from the authors are both outrageously slow. I don't know R very well, and apparently am not able to code in C properly, so Python is the only alternative. There are fortunately many existing Python implementations.[^existing-Python-implementations] However, some of them give slightly different results from the Stata ones. So I decided to write my own synthetic control, which should be close to Stata's estimation, and benchmark the speed/performance.

[^existing-Python-implementations]: Such as [d2cml-ai/synthdid.py](https://github.com/d2cml-ai/synthdid.py), Facure and Germano ([2022](https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html)), [MasaAsami/pysynthdid](https://github.com/MasaAsami/pysynthdid), [microsoft/SparseSC](https://github.com/microsoft/SparseSC), and [OscarEngelbrektson/SyntheticControlMethods](https://github.com/OscarEngelbrektson/SyntheticControlMethods).

I take the very seminal California tobacco control dataset from Abadie et al. (*JASA* [2010](https://doi.org/10.1198/jasa.2009.ap08746)). The dataset is a state level panel: for each year, in each state, we know the cigarette sales ($cigsale$), among other variables. The paper estimates the effect of a tobacco control program in California, which took place in 1988. The main idea is that we can evaluate this program by comparing $cigsale$ in California vs other states, before vs after 1988. Intuitively, some average of other states can be a nice counterfactual for California. E.g. simple DID would use a simple unweighted average of all other states. The synthetic control takes one step further to use a weighted average, e.g. California's without treatment counterfactual $\approx 0.7\ \times$ Connecticut $+\  0.3\ \times$ Utah. The question is how to find the appropriate weights, $0.7$ and $0.3$ in this example, to construct this weighted average. A straightforward idea, very similar to pre-trend in DID or matching estimators, is to find such a weighting scheme that before the treatment, the weighted average's $cigsale$ is very similar to California's $cigsale$.


## The math part

Formally, let's define what is "similar". We want the difference in $cigsale$ between California and the weighted average to be small, in all pre-treatment periods. In the dataset, we have California and 38 other states. Let $cigsale_{j, t}$ be the cigarette sales in state $j$ in year $t$. $j = 0$ denotes California and $j\geq 1$ are other states. So the weights 

$$\vec{w} = \{w_1, w_2, \cdots, w_{38}\}^T$$

is a $38\times 1$ column vector, where $w_j$ is the weight for state $j$. The weighted average in year $t$ is then simply

$$\widehat{cigsale_t} = \sum\limits_{j = 1}^{38} w_j cigsale_{j, t}$$

. The difference between this above weighted average and California in year $t$ is

$$cigsale_{0, t} - \widehat{cigsale_t} = cigsale_{0, t} - \sum\limits_{j = 1}^{38} w_j cigsale_{j, t}$$

. The dataset begins in year 1970 and the treatment happened in 1988, so we want the total differences in the 19 years to be small.

<figure>
    <img src="/assets/images/scm_loss.svg">
    <p><i>Notes</i>: Made with GeoGebra. These points are faked/solely for illustration, not actual data.</p>
</figure>

This looks extremely similar to usual OLS, so it's natural to define "similar" as

$$\sum\limits_{t = 1970}^{1988}\Bigg (cigsale_{0, t} - \sum\limits_{j = 1}^{38} w_j cigsale_{j, t}\Bigg )^2$$

. Or even easier in matrix form:

$$(X_0 - X_1\vec{w})^T(X_0 - X_1\vec{w})$$

, where $X_0 = \\{cigsale_{0, 1970}, \cdots, cigsale_{0, 1988}\\}^T$ is a $19\times 1$ column vector, $X_1$ is a $19\times 38$ matrix, where the $(p, q)$ entry is the $cigsale$ in state $q$ in year $p$.[^compare-with-ols]

[^compare-with-ols]: This is no fancier than OLS. In OLS we minimise $(y - X\beta)^T(y - X\beta)$. Here it's the same thing.

Now it's clear how to find the weights $\vec{w}$:

$$
\begin{align}
\min_{\vec{w}}\ & (X_0 - X_1\vec{w})^T(X_0 - X_1\vec{w}) \label{eq:optimisation} \\
\text{s.t.}\    & \sum_j w_j = 1 \nonumber \\
                & 0\leq w_j \leq 1\ \forall j \nonumber \\
\end{align}
$$

The objective function is quadratic and thus convex. Constrains are that the weights adds up to one, and each individual weight is bounded between $0$ and $1$, both of which are linear.[^convex-hull] So in the end it's a convex optimisation and we are guaranteed to find a global optimum. After getting the solution $w^*$, we can compare California and weighted average after 1988 to get the treatment effects:

$$
\begin{equation}ATT_t = cigsale_{0, t} - \sum\limits_{j = 1}^{38} w^*_j cigsale_{j, t}\label{eq:att}
\end{equation}$$

, for $t\geq 1989$.

[^convex-hull]: By "weighted average", we implicitly assume something like it's fine to have $0.1\ \times$ Ohio $+\ 0.9\ \times$ Alaska, but it's not fine to have $-1.8\ \times$ Seattle $+\ 15.4\ \times$ Arkansas. This is *not* trivial. There are good econometric reasons to use a weighted average instead of an arbitrary linear combination. It actually speaks to interpolation vs extrapolation, or overfitting, or regularisation. See discussion around Figure 2 in Abadie (*JEL* [2021](https://doi.org/10.1257/jel.20191450)) and Amjad et al. (*JMLR* [2018](https://doi.org/10.5555/3291125.3291147)).


## Python implementation

As the first step, let's get $X_0$ and $X_1$ from the rawdata. As said above, each row in $X$'s is a year and each column is a state. E.g., the $(2, 1)$ cell in $X_0$ is $cigsale$ in year $1970 + 2$ and state $1$, which is Arkansas.

```python
import pandas as pd

# Get data in pre-treatment periods, i.e. before 1988
df = pd.read_stata(
    'https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta'
)
df['year'] = df['year'].astype(int)
pre = df[df['year'] <= 1988][['state', 'year', 'cigsale']]

# Reshape to wide format: each column is a state and each row is a year
pre = pre \
    .set_index(['state', 'year']) \
    .unstack(level=0) \
    .stack(level=0, future_stack=True)
print(pre.shape)  # (19, 39)

# X_0 is the column for California
X_0 = pre['California'].values
print(X_0.shape)  # (19,)

# X_1 is the columns for the rest 38 states
X_1 = pre.drop(columns='California').values
print(X_1.shape)  # (19, 38)
```

Now set up the optimisation. We have two popular packages for such optimisation problem: [SciPy](https://github.com/scipy/scipy) and [CVXPY](https://github.com/cvxpy/cvxpy). SciPy is powerful and can solve most problems under the sun, while CVXPY, as the name "CVX" already states, is only for convex problems. I'm more familiar with SciPy so let's use it first.

```python
from functools import partial
from scipy.optimize import minimize

# Objective function to minimise
def loss(w, X_0, X_1):
    resid = X_0 - X_1 @ w
    return resid.T @ resid

# Constraint 1: all weights sum up to one
constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})

# Solve for the optimal weights
n = X_1.shape[1]  # #. of other states, which is 38
w_scipy = minimize(
    fun=partial(loss, X_0=X_0, X_1=X_1),
    x0=[0] * n,  # Initial guess of the solution. Set to zero's for simplicity
    constraints=constraints,
    bounds=[(0, 1)] * n  # Constraint 2: each weight is in [0, 1]
).x
```

We can set up the problem using CVXPY similarly:

```python
import cvxpy as cp

# 38*1 col. vector of weights
w = cp.Variable(n)

# Objective function. Equivalent to `resid.T @ resid` above
loss = cp.Minimize(cp.sum_squares(X_0 - X_1 @ w))

# Constraints of the weights
constraints = [
    cp.sum(w) == 1,
    w >= 0,
    w <= 1
]

# Solve
problem = cp.Problem(loss, constraints)
problem.solve()
# 52.12957126425125
w_cvxpy = w.value

# Save the results of both SciPy and CVXPY
python = pd.DataFrame({'scipy': w_scipy, 'cvxpy': w_cvxpy})
python.to_csv('data/python_synth.csv', index=False)
```

Finally, we need the solution from the authors' official Stata package:[^customV]

[^customV]: One thing is that we treat all years equally, i.e. each column in $X_1$ is treated equally. This does not necessarily have to be true. See the last section below. Also see Section 3.2 in Abadie (*JEL* [2021](https://doi.org/10.1257/jel.20191450)).

```stata
clear all
use "https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta"
xtset state year
#delimit ;
synth cigsale
      cigsale(1970) cigsale(1971) cigsale(1972) cigsale(1973) cigsale(1974)
      cigsale(1975) cigsale(1976) cigsale(1977) cigsale(1978) cigsale(1979)
      cigsale(1980) cigsale(1981) cigsale(1982) cigsale(1983) cigsale(1984)
      cigsale(1985) cigsale(1986) cigsale(1987) cigsale(1988)
  , trunit(3) trperiod(1989)
    customV(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
    keep("data/stata_synth.dta") replace ;
#delimit cr

// Save Stata results in .csv instead of .dta, so git is happy
use "data/stata_synth.dta", clear
export delimited "data/stata_synth.csv", delimit(",") replace
erase "data/stata_synth.dta"
```


## Same as Stata's estimates?

Now we plot the optimal weights solved from ($\ref{eq:optimisation}$) below. The big picture is more or less the same: the weights are sparse; only five states get a non-zero weight in the right panel below. However, within those five states, there is quite some difference. For example, the Stata package assigns **15%+ more** weights on Montana than Python implementations. Both SciPy and CVXPY gives almost identical numbers.

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/synth/figures/weights.svg">
</figure>

To figure out why the results are different, I first check if the optimisation is set up correctly. In principle, if we feed Stata's weights into the optimisation, we should:

1. satisfies all constraints, so that constraints are correct
1. get the same effect size/ATT as Stata, so that if we have the same weights, we will have same effect size
1. get the same loss as Stata, so that the loss function is correct

```python
import numpy as np

stata = pd.read_csv('data/stata_synth.csv')
w_stata = stata['_W_Weight'].values

# 1. constraints happy?
assert (w_stata >= 0).all() and (w_stata <= 1).all()

# 2. same ATT?
att_stata = stata['_Y_synthetic'].values
entire_period = df[['state', 'year', 'cigsale']]  # Get the full sample to wide
entire_period = entire_period \
    .set_index(['state', 'year']) \
    .unstack(level=0) \
    .stack(level=0, future_stack=True)
entire_period.drop(columns=['California'], inplace=True)
print(entire_period.shape)  # (31, 38)
att_scipy = entire_period.values @ w_stata
np.testing.assert_array_almost_equal(att_scipy, att_scipy)

# 3. same loss?
def loss(w, X_0, X_1):
    resid = X_0 - X_1 @ w
    return resid.T @ resid

temp = stata[stata['_time'] <= 1988]
loss_stata = ((temp['_Y_treated'] - temp['_Y_synthetic']) ** 2).sum()
print(loss_stata)  # 54.7451436809619
np.testing.assert_approx_equal(loss(w_stata, X_0, X_1), loss_stata)
```

Passed all three tests, so our optimisation setup has no problem. Now the only explanation is that Stata is wrong, i.e. not find the optimum... Recall the loss from CVXPY above: that is 52.xxx. Here Stata's loss is 54.xxx. Stata's loss is **5% higher**! After checking the documentation of `synth` in Stata, it seems to have a `nested` option that gives better convergence. After specifying that, Stata is much slower (as expected), but now we get almost identical weights! Great! Our code is correct!

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/synth/figures/weights_nested.svg">
</figure>


## Performance benchmark

Finally, we compare the speed of different implementations. Stata is too slow so I have to use log $y$ axis. All Python implementations are roughly 500x faster than the Stata one (, which is not surprising at all).[^c-also-slow] SciPy is the slowest among Python implementations. This is also expected as `scipy.optimize.minimize` aims at solving general optimisation problems, so it does not exploit the convexity of our problem. On the other side, CVXPY can only solve convex problems (, and in fact it first [checks](https://www.cvxpy.org/tutorial/dcp/index.html#dcp-problems) whether your problem is indeed convex before proceeding to solve it,) and utilise all maths that helps to solve it quickly. Therefore, CVXPY is roughly 2.5x quicker than SciPy.

[^c-also-slow]: One surprising fact is that even without `nested` option, Stata is still an order of magnitude slower. If I read the doc. correctly, `synth` uses a C++ plugin to get the weights when `nested` is not specified. It shouldn't be that slow. A reasonable speedup factor for usual `reg` is at least 10x, as documented by [Mauricio M. Cáceres Bravo](https://mcaceresb.github.io/stata/plugins/2017/02/15/writing-stata-plugins-example.html). I don't know why it does not achieve a good efficiency here.

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/synth/figures/timing.svg">
    <p><i>Notes</i>: This is the violin plot the amount of time used in each implementation, over 1,000 runs for each. Horizontal thick lines are the average run time.</p>
</figure>

We do manage to make SciPy slightly faster by providing the closed-form gradient vector. SciPy needs to compute the partial derivatives of objection function w.r.t. the weights.[^gradient] The SciPy default is to compute the numerical derivatives, i.e. take $x$ and $x + \Delta x$, compute $f(x + \Delta x) - f(x)$, and finally divide that by $\Delta x$, with a sufficiently small $\Delta x$. This is slow. If we provide the derivative $f'(x)$ directly, then SciPy can simply plug in $x$ and get the number. And indeed we do observe a doubled speed when the gradient, i.e. Jacobian matrix, is provided to SciPy. But after all, CVXPY should be the preferred method.

[^gradient]: Again, this is not unheard-of. In OLS, we want to minimise $RSS = (y - X\beta)^T(y - X\beta)$, and we also compute the derivatives $\displaystyle\frac{\text{d}RSS}{\text{d}\beta} = -2X^T(y - X\beta)$.
