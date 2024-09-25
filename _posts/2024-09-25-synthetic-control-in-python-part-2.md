---
layout:       post
title:        "Synthetic control in Python, part 2"
date:         2024-09-25
tags:         [econ]
---

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/tree/main/synth">harningle/useful-scripts</a>)</font></p>

There are several important features not implemented in the [last post]({% link _posts/2024-08-20-synthetic-control-in-python.md %}):

1. we do not only want to match outcomes, but also other covariates
1. when matching multiple covariates, how to assign proper "relative importance" of them?

There are good reasons to match more than the outcome. E.g., California is a rich state, so a good control should also be rich, and thus we want to match pre-treatment income $lnincome$ as well.[^not-match-outcome] Once we have more than one variable to match, a natural question is: are they equally important? Intuitively, the outcome $cigsale$ is more important than $lnincome$, since our research question is about $cigsale$. But how to define or find the "relative importance" across matching variables? Again, pre-trend! Among all possible relative importance, use the one that gives us the best pre-trend.

[^not-match-outcome]: To me there are ever better reasons for *not* matching *only* the outcome on a per-period basis. If we *only* match on the outcome, the matching perfectly coincides with the treatment: then the matching stops when the treatment begins. If we find an effect after treatment, is this "effect" the true causal effect, or is it the effect of our matching? This could give very misleading results: no pre-trend, by construction of the matching, and some effect, which we don't know what it is. Abadie and Gardeazabal (*AER* [2003](https://doi.org/10.1257/000282803321455188), Appendix B) also discourage matching the outcome only.


## The math part

Following the authors' Stata package, we match on (1) the average beer sale $beer$ in 1984-1988, the pre-treatment average (2) income $lnincome$, (3) cigarette price $retprice$, and (4) age composition $age15to24$, and also the oucome $cigsale$ in year (5) 1988, (6) 1980, and (7) 1975. Here we have seven variables to match. Recall the notations from the [previous post]({% link _posts/2024-08-20-synthetic-control-in-python.md %}).[^more-rigour] Now the variables being matched for California is

[^more-rigour]: See Abadie et al. (JASA [2010](https://dx.doi.org/10.1198/jasa.2009.ap08746), Section 2) for a proper mathematical derivation.

$$
X_0 = \{\overline{beer_{0, 1984, \cdots, 1988}}, \overline{lnincome_0}, \overline{retprice_0}, \overline{age15to24_0}, cigsale_{0, 1988}, cigsale_{0, 1980}, cigsale_{0, 1975}\}^T
$$

, which is a $7\times 1$ column vector. Similarly, $X_1$ for other states is a $7\times 38$ matrix, where each row is one of the seven variables, and each column is one of the other states. The weighted average of other states is still $X_1\vec{w}$. Since each of the seven variables carry different relative importance, the similarity between California and $X_1\vec{w}$ is also weighted by this relative importance:

$$
(X_0 - X_1\vec{w})^T V (X_0 - X_1\vec{w})
$$

.[^WLS] Here $V$ is a $7\times 7$ diagonal matrix, where $(i, i)$ is the relative importance of variable $i$. E.g., if we believe, in the similarity computation, $cigsale$ is more important than $lnincome$, then $V_{4, 4}$ (and also $V_{5, 5}$ and $V_{6, 6}$) should be bigger than $V_{1, 1}$. Again here the relative importance shouldn't be negative, i.e. $V$ is positive definite, and all of them sum up to one.

[^WLS]: This form is exactly the same as weighted least squares. Here the "weights" are the relative importance: important variables carry more weights in the similarity calculation. To avoid terminology confusion, I will call $V$ "relative importance" and reserve "weights" for the weights $\vec{w}$ in weighted average of other states.

Now we have two problems: how to find the relative importance $V$, and how to find the weighting scheme $\vec{w}$ for weighted average. An ideal $V$ should be that, using this relative importance, we solve for the weights $\vec{w}$ that make the weighted average of other states very similar to California, and this $\vec{w}$ will give us nice pre trend. Formally, for each possible $V$, we solve for $\vec{w}$:

$$
    \vec{w}^*(V) = \underset{\vec{w}}{\text{argmin}} (X_0 - X_1\vec{w})^T V (X_0 - X_1\vec{w})
$$

. Now using this weights $\vec{w}^*(V)$, the pre trend is

$$
\big (Y_0 - Y_1\vec{w}^*(V)\big )^T \big (Y_0 - Y_1\vec{w}^*(V) \big )
$$


, where $Y_0$ is a $19\times 1$ column vector for the time series of the outcome $cigsale$ in California. E.g., $(4, 0)$ element in $Y_0$ is the $cigsale$ in period $4$, which is year 1984, in California. $Y_1$ is a $19\times 38$ matrix, where each column is one of the other states, and each row is $cigsale$ in a year.

Now for each possible $V$, we have a pre trend. Among all possible $V$'s, we pick the $V$ with smallest pre trend, and this $V$ is the best relative importance. Combining the two, the final optimisation problem is:

$$
\begin{align}
    \min_V\         & (Y_0 - Y_1\vec{w}^*)^T (Y_0 - Y_1\vec{w}^*)\label{eq:loss}\\
    \text{s.t.}\    & \vec{w}^* = \underset{0\leq w_j \leq 1, \sum_j w_j = 1}{\text{argmin}} (X_0 - X_1\vec{w})^T V (X_0 - X_1\vec{w}) \label{eq:w}\\
                    & \sum_i v_i = 1 \nonumber \\
                    & 0\leq v_i \leq 1\ \forall i \nonumber \\
\end{align}
$$


However, although ($\ref{eq:w}$) is convex, the entire problem ($\ref{eq:loss}$) is not. There is no method that analytically guarantees to find the global minimum of a non convex problem. So the below Python implementation should make sure we don't stuck in bad local minimum.


## Python implementation

First of all, let's get $X_0$, $X_1$, $Y_0$, and $Y_1$.

```python
import pandas as pd

# Get data in pre-treatment periods, i.e. before 1988
df = pd.read_stata(
    'https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta'
)
df['year'] = df['year'].astype(int)
pre = df[df['year'] <= 1988][['state', 'year', 'cigsale', 'beer', 'lnincome',
                              'retprice', 'age15to24']]

# 1984-1988 avg. beer
beer = pre[pre['year'].between(1984, 1988)].groupby('state')['beer'].mean()

# Pre-treatment avg. income, cigarette price, and age structure
temp = pre.groupby('state')[['lnincome', 'retprice', 'age15to24']].mean()
temp['beer'] = beer
del beer
temp = temp[['beer', 'lnincome', 'retprice', 'age15to24']]

# Cigarette sales in 1988, 1980, and 1975
cigsale = pre[pre['year'].isin([1988, 1980, 1975])]
cigsale = cigsale[['state', 'year', 'cigsale']] \
    .set_index(['state', 'year']) \
    .unstack(level=0) \
    .stack(level=0, future_stack=True) \
    .iloc[::-1]  # Sort years in descending order, following Stata's synth...

# X_0 and X_1
pre = pd.concat([temp.T, cigsale])
temp = pre.values
avg = temp.mean(axis=1, keepdims=True)
std_dev = temp.std(axis=1, keepdims=True)
temp = (temp - avg) / std_dev
pre = pd.DataFrame(temp, index=pre.index, columns=pre.columns)
del temp, cigsale
X_0 = pre['California'].values
print(X_0.shape)  # (7,)
X_1 = pre.drop(columns='California').values
print(X_1.shape)  # (7, 38)
del pre

# Y_0 and Y_1
y = df[(df['year'] <= 1988)][['state', 'year', 'cigsale']] \
    .set_index(['state', 'year']) \
    .unstack(level=0) \
    .stack(level=0, future_stack=True)
Y_0 = y['California'].values
print(Y_0.shape)  # (19,)
Y_1 = y.drop(columns='California').values
print(Y_1.shape)  # (19, 38)
del y
```

The core is to set up the new optimisation. As this is not a convex problem, most algorithms generally cannot guarantee to converge to the global optimal. Anyway, we use SciPy default as the first attempt.

```python
import cvxpy as cp
from functools import partial
import json
import numpy as np
from scipy.optimize import LinearConstraint, minimize
import time

# For a given v, solve for w by eq. (2)
def solve_w(v, X_0, X_1, Y_0, Y_1):
    n = X_1.shape[1]
    w = cp.Variable(n)
    m = X_0.shape[0]
    V = np.zeros((m, m))
    np.fill_diagonal(V, v)
    similarity = cp.sum(V @ cp.square(X_0 - X_1 @ w))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 1
    ]
    problem = cp.Problem(cp.Minimize(similarity), constraints)
    problem.solve()
    return w.value

# Pre-trend to minimise
def loss(v, X_0, X_1, Y_0, Y_1):
    # First, get the w for this given v
    w = solve_w(v, X_0, X_1, Y_0, Y_1)

    # Now plug w into eq. (1) to get the pre trend
    resid = Y_0 - Y_1 @ w
    return resid.T @ resid

# Constraints: all relative importance sum up to one
m = X_1.shape[0]  # #. of matching variables, which is 7
constraints = [LinearConstraint(np.ones(m), 1, 1)]
"""
In the last post, we write the constraint as `{'type': 'eq', 'fun': lambda v:
v.sum() - 1}`. Here we use a linear constraint, which basically asks the dot
product between a row vector of ones and \diag{V}, which is equal to the sum of
\diag{V}, to be 1. They are (mathematically) equivalent. This format makes it
easier for our later experiments.
"""

# Solve for the optimal relative importance
np.random.seed(seed=42)
v = minimize(
    fun=partial(loss, X_0=X_0, X_1=X_1, Y_0=Y_0, Y_1=Y_1),
    x0=[1 / m] * m,      # Initial guess. Set to zero for simplicity
    constraints=constraints,
    bounds=[(0, 1)] * m  # Constraint: each importance is in [0, 1]
)
```

Now get the estimation from official Stata `synth`.

```stata
clear all
use "https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta"
xtset state year
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) ///
      cigsale(1980) cigsale(1975), trunit(3) trperiod(1989) nested allopt
```

The figure below plots the solved $\vec{w}$ and $V$ of our Python estimate and the official Stata package. They are very (!) different, and our loss is way much higher. It may be the case that SciPy is stuck in a local minimum while Stata finds the global or at least a better local minimum. So the next section will try various approaches to make sure our Python implementation converges to a minimum that is at least as good as the Stata one.

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/synth/figures/v_w_equal_guess.svg">
</figure>

## Journey to global minimum

The default `scipy.optimize.minimize` doesn't work well above. In order to find a better minimum, we try a few different settings below, and also try to reformulate the problem into a single level one mathematically using KKT.

### More initial guesses

One reason why SciPy fails is that our initial guess, which is set to be equal relative importance for simplicity, is bad.[^kaiming-initialisation] The opposite extreme is that you are so lucky that your initial guess is the global optimal, and SciPy would then definitely identify this global optimal. Therefore an easy fix is simply to try more initial guesses, and hope for some guesses being near the global optimal, so SciPy should be able to find it. Since all relative importance are bounded between $0$ and $1$ and sum up to $1$, we can sample from [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution#Support) as the initial guesses.

[^kaiming-initialisation]: The initial guess, or initialisation of weights, is very important in machine learning. There are many studies in this area, e.g. He et al. (*ICCV* [2015](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)). Picard ([2023](https://arxiv.org/abs/2109.08203)), one of my favourite papers, even finds the random seed used in initialisation matters *a lot*.

<figure style="width: 50%;">
    <img src="https://64.media.tumblr.com/9767309293e985358a71ba49358f762a/tumblr_mmygapyiOJ1sp8cfeo1_640.pnj">
    <p><i>Source</i>: <a href="https://daronacemoglufacts.tumblr.com/">Daron Acemoglu Facts</a></p>
</figure>

```python
np.random.seed(42)
guesses = np.random.dirichlet(alpha=np.ones(m), size=100)
losses = []
for g in guesses:
    try:
        v = minimize(
            fun=partial(loss, X_0=X_0, X_1=X_1, Y_0=Y_0, Y_1=Y_1),
            x0=g,
            constraints=constraints,
            bounds=[(0, 1)] * m
        )
        losses.append(v.fun)
    except:  # Sometimes our initial guess is so bad that CVXPY cannot solve w
        losses.append(np.nan)
```

Above we sample 100 realisations from Dirichlet distribution as the initial guess, and we plot the loss of each of them below. Fairly many of them (35%) converge to ok-ish local optimum, but the variance is big. This is slow as well. Can be a fallback plan, but not ideal.

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/synth/figures/initial_guess_loss.svg">
    <p><i>Notes</i>: Among all 100 trials, five fails to converge, and six are better than Stata.</p>
</figure>


### Make more sophisticated guesses

Are there "better" initial guess than others? Instead of blindly sampling guesses from Dirichlet distribution, can we only try a small amount of "better" guesses? Abadie et al. (*J. Stat. Softw.* [2011](https://doi.org/10.18637/jss.v042.i13)) proposed a regression-based guess.[^footnote] The intuition is straightforward. We have $lnincome$, $cigsale$, ..., and by matching them, the goal is to get a nice pre trend. Which variables should be more important in this sense? Of course $cigsale$, because it is effectively the pre trend. Therefore, if a matching variable is strongly correlated with the outcome, it should receive a larger relative importance. So we can simply regress $Y$ on $X$ to get the correlation, and then diagonalise the regression coefficients to get an initial guess of $V$.[^reg]

[^footnote]: This regression-based guess is only briefly mentioned in footnote 2 in Abadie et al. (*J. Stat. Softw.* [2011](https://doi.org/10.18637/jss.v042.i13)). The authors don't even explain what exactly it is in the paper or in their Stata and R package. (Or maybe I missed them in my reading somehow.) This works amazingly well! Check their R package [source code](https://cran.r-project.org/web/packages/Synth/index.html) for details.

[^reg]: The econometrics behind this is not straightforward to me. If it was me, I would simply run `reg cigsale beer age15to24 ...` and take the coefficients. The authors run $[Y_0, Y_1]$ on $[X_0, X_1]$. That is, the outcome is not a column vector as in most common regressions. It is $[Y_0, Y_1]$, a $19\times 39$ matrix in this case. So it's regressing a matrix on a matrix, instead of a column vector on a matrix. I don't understand the reason behind this, but simply follow the authors' code here.

```python
from sklearn.linear_model import LinearRegression

X = np.hstack((X_0[:, None], X_1))
Y = np.hstack((Y_0[:, None], Y_1))
reg = LinearRegression(fit_intercept=True).fit(X.T, Y.T)
coef = reg.coef_
v_0 = np.diag(coef.T @ coef)
v_0 = v_0 / v_0.sum()
np.random.seed(42)
v = minimize(
    fun=partial(loss, X_0=X_0, X_1=X_1, Y_0=Y_0, Y_1=Y_1),
    x0=v_0,
    constraints=constraints,
    bounds=[(0, 1)] * m
)
```


### Why constraint?

$V$ has to sum up to one. This is a linear constraint, and not many optimiser can handle constraints. Why not simply remove this constraint? Eq. ($\ref{eq:w}$) and ($\ref{eq:loss}$) are both homogeneous of degree 1 w.r.t. $V$. That is, $V$ and $2V$ give identical result. Therefore, whether $V$ adds up to one or five doesn't matter at all. By removing the constraint on $\sum v_i$, we have access to a broader set of optimisers, which were previously unavailable due to the existence of the constraint.[^bound-and-constraint] We can benchmark and pick the best optimiser. However, this is not necessarily a Pareto improvement. If there exists one global optimal $V^*$, then there are infinitely many global optimal due to homogeneity. I don't know if optimisers like such type of functions.

[^bound-and-constraint]: After removing $\sum v_i = 1$, we still have the bound constraint $0\leq v_i\leq 1$. That's fine. Bounds are easier for optimisers than general constraints. E.g., [L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) can handle bounds but not constraints.

```python
v = minimize(
    fun=partial(loss, X_0=X_0, X_1=X_1, Y_0=Y_0, Y_1=Y_1),
    x0=v_0,
    constraints=constraints,
    bounds=[(0, 1)] * m
)
v = v.x / v.x.sum()  # If we want, can easily normalise V to sum up to one
```


### Reformulate to single level optimisation

The optimisation is difficult partly because of the [bilevel/nested](https://en.wikipedia.org/wiki/Bilevel_optimization). nature: when the minimising of pre trend, the constraint has to maximise similarity. The good news is that the inner optimisation of solving $w$ is convex, so we can easily replace it with KKT and reformulate the entire problem into a single optimisation. First, the inner problem is:

$$
\begin{align*}
\min_{\vec{w}}\ & (X_0 - X_1\vec{w})^T V (X_0 - X_1\vec{w}) \\
\text{s.t.}\    & \sum_j w_j = 1 \\
                & 0\leq w_j \leq 1\ \forall j \\
\end{align*}
$$

The Lagrangian is

$$
\mathcal{L} = (X_0 - X_1\vec{w})^T V (X_0 - X_1\vec{w}) + \lambda (\sum_j w_j - 1) - \vec{\mu}^T\vec{w} + \vec{\nu}^T(\vec{w} - 1)
$$

The KKT conditions are then:

$$
\begin{align}
\nabla_\vec{w}\mathcal{L} = -2X_1^T V(X_0 - X_1\vec{w}) + \lambda \vec{1} - \vec{\mu} + \vec{\nu} &= 0 \label{eq:3} \\
\sum_j w_j - 1 &= 0 \label{eq:4} \\
0\leq w_j      &\leq 1 \label{eq:5} \\
\mu_j          &\geq 0 \label{eq:6} \\
\nu_j          &\geq 0 \label{eq:7} \\
\mu_j w_j      &= 0 \label{eq:8} \\
\nu_j(w_i - 1) &= 0\ \forall j\label{eq:9}
\end{align}
$$

, where ($\ref{eq:3}$) sets the derivative of Lagrangian w.r.t. $\vec{w}$ to zero, ($\ref{eq:4}$) and ($\ref{eq:5}$) are the constraints on $\vec{w}$, i.e. primary feasibility, ($\ref{eq:6}$) and ($\ref{eq:7}$) are for inequality constraints of $w_j\in [0, 1]$, i.e. dual feasibility, and finally ($\ref{eq:8}$) and ($\ref{eq:9}$) are complementary slackness. Now plug them back to the outer problem, we get the single level reformulation:

$$
\begin{align*}
\min_{V, \vec{w}, \lambda, \vec{\mu}, \vec{\nu}}\ & (Y_0 - Y_1\vec{w})^T(Y_0 - Y_1\vec{w}) \\
\text{s.t.}\    & 0\leq v_i \leq 1\ \forall i \\
                & \enclose{horizontalstrike}{\sum v_i = 1} \\
                & -2X_1^T V(X_0 - X_1\vec{w}) + \lambda \vec{1} - \vec{\mu} + \vec{\nu} = 0 \\
                & \sum w_j - 1 = 0 \\
                & 0\leq w_j      \leq 1 \\
                & \mu_j          \geq 0 \\
                & \nu_j          \geq 0 \\
                & \mu_j w_j      = 0 \\
                & \nu_j(w_i - 1) = 0\ \forall j
\end{align*}
$$

. Note that we may or may not want to include the constraint of $\sum v_i$, as explained [above](#why-constraint). This single level problem has a very nice quadratic target function, and a bunch of linear and nonlinear constraints. This is not necessarily easier than the original one. Now we are optimising over $\vec{w}, V, \lambda, \vec{\mu}$, and $\vec{\nu}$. It's $38 + 7 + 1 + 38 + 38 = 122$ variables! It's a very high dimensional non-convex problem. It's not clear whether it's easier to solve a bilevel problem with $7$ variables, or easier to solve a single level one with 100+ variables.

This problem can translate to SciPy easily, though very tedious. To speed up, we provide analytical Jacobian here.

```python
from scipy.optimize import NonlinearConstraint

n = X_1.shape[1]
m = X_0.shape[0]

# Initial guess for w, V, \lambda, \mu, \nu
initial_w = solve_w(v_0, X_0, X_1, Y_0, Y_1)
initial_V = v_0
initial_lambda = 1e-3
initial_mu = np.zeros(n)
initial_nu = np.zeros(n)

# Flatten the initial variables into a single array
"""
In this example, the first 38 elements are for \vec{w}, and the next 7 elements
are for relative importance V. Afterwards, it is Lagrangian multiplier \lambda.
Then we have \vec{\mu} and \vec{\nu}, each of which has 38 elements. So in the
end we are optimising over 38 + 7 + 1 + 38 + 38 = 122 numbers.
"""
initial_guess = np.concatenate([initial_w, initial_V, [initial_lambda],
                                initial_mu, initial_nu])

# Objective function: eq. (1)
def objective(x):
    w = x[:n]
    resid = Y_0 - Y_1 @ w
    return resid.T @ resid

def objective(x):
    w = x[:n]
    resid = Y_0 - Y_1 @ w
    return np.linalg.norm(resid)

def jac_objective(x):
    w = x[:n]
    j = np.zeros(x.shape)
    j[:n] = -2 * Y_1.T @ (Y_0 - Y_1 @ w)
    return j

def jac_objective(x):
    w = x[:n]
    j = np.zeros(x.shape)
    resid = Y_0 - Y_1 @ w
    j[:n] = -(resid.T @ Y_1) / np.linalg.norm(resid)
    return j

# \nabla L = 0: eq. (3)
def stationarity(x):
    w = x[:n]
    V = np.diag(x[n:n + m])
    lambda_ = x[n + m]
    mu = x[n + m + 1:n + m + 1 + n]
    nu = x[n + m + 1 + n:]
    grad = -2 * X_1.T @ V @ (X_0 - X_1 @ w) + lambda_ * np.ones(n) - mu + nu
    return grad

def jac_stationarity(x):
    w = x[:n]
    V = np.diag(x[n:n + m])
    J = np.zeros((n, len(x)))

    # w.r.t. w
    J[:, :n] = 2 * X_1.T @ V @ X_1

    # w.r.t. V
    for i in range(m):
        J[:, n + i] = -2 * X_1.T[:, i] * (X_0[i] - X_1[i] @ w)

    # w.r.t. \lambda
    J[:, n + m] = np.ones(n)

    # w.r.t. \mu and \nu
    J[:, n + m + 1:n + m + 1 + n] = -np.eye(n)
    J[:, n + m + 1 + n:] = np.eye(n)
    return J

# Eq. (8) and (9)
def complementary_slackness(x):
    w = x[:n]
    mu = x[n + m + 1:n + m + 1 + n]
    nu = x[n + m + 1 + n:]
    return np.concatenate([mu * w, nu * (w - 1)])

def jac_complementary_slackness(x):
    w = x[:n]
    mu = x[n + m + 1:n + m + 1 + n]
    nu = x[n + m + 1 + n:]

    J = np.zeros((2 * n, len(x)))

    # Derivatives of mu * w with respect to w and mu
    J[:n, :n] = np.diag(mu)
    J[:n, n + m + 1:n + m + 1 + n] = np.diag(w)

    # Derivatives of nu * (w - 1) with respect to w and nu
    J[n:, :n] = np.diag(nu)
    J[n:, n + m + 1 + n:] = np.diag(w - 1)
    return J

# \sum v_i = 1
v_sum = np.zeros(len(initial_guess))
v_sum[n:n + m] = 1

# 0 <= v_i <= 1
v_bounds = np.zeros((m, len(initial_guess)))
v_bounds[:, n:n + m] = np.eye(m)

# \sum w_i = 1: eq. (4)
w_sum = np.zeros(len(initial_guess))
w_sum[:n] = 1

# 0 <= w_j <= 1: eq. (5)
w_bounds = np.zeros((n, len(initial_guess)))
w_bounds[:, :n] = np.eye(n)

# Dual feasibility: eq. (6) and (7)
mu_nu_bounds = np.zeros((n * 2, len(initial_guess)))
mu_nu_bounds[:, n + m + 1:] = np.eye(n * 2)


# Constraints
constraints = [
    LinearConstraint(v_bounds, 0, 1),
    LinearConstraint(v_sum, 1, 1),
    NonlinearConstraint(stationarity, 0, 0, jac=jac_stationarity),
    LinearConstraint(w_sum, 1, 1),
    LinearConstraint(w_bounds, 0, 1),
    LinearConstraint(mu_nu_bounds, 0, np.inf),
    NonlinearConstraint(complementary_slackness, 0, 0,
                        jac=jac_complementary_slackness)
]

# Perform optimization
result = minimize(objective, initial_guess, method='trust-constr',
                  constraints=constraints,
                  jac=jac_objective,
                  options={'disp': True})

# Extract the solution
w = result.x[:n]
v = np.diag(result.x[n:n + m])
```


### RSS or RMSE?

This was motivated by an error/warning `Positive directional derivative for linesearch` when I ran the code above. I don't know the underlying algorithm or math behind this, but this seems to be related to the *scale*/*magnitude* of loss function.[^positive-directional-derivative-for-linesearch] Many optimisers are gradient-based, so we need a numerically well-behaved Jacobian. Basically, if the loss is like $1, 2, 1.3, 0.3, 2.5, 2.1$, it's fine. But the optimiser don't like $1, 200, 1.3, 0.000003, 250000, 2100$, even if it's simply a monotonic transformation of the first loss, because sometimes you get super large gradient. So instead of using $RSS$, we can use the square root of $RSS$, or $\ell^2$-norm of the residuals.[^RSS-RMSE] This will make the loss/gradient more stable.

[^positive-directional-derivative-for-linesearch]: See my answer at [Stack Overflow](https://stackoverflow.com/a/78917038/12867291).

[^RSS-RMSE]: I'm abusing the terminology a bit here. RSS, MSE, RMSE, $\ell^2$-norm, etc., are simple transformation of each other. What I mean here is simply taking the square root of $RSS$. See [here](https://stats.stackexchange.com/a/221845).

### Constraint strictness

If you read authors' Stata `synth`, there is an option called `margin` for constraint violation tolerance. Say we need $\sum w_j = 1$, it's fine to have $\sum w_j = 0.999999$, or maybe $\sum w_j = 1.003$. This makes the problem more flexible and the floating number optimisation easier, while keeping the constraint more or less the same. See discussions [here](https://or.stackexchange.com/q/3166) and [here](https://stackoverflow.com/q/61577559/12867291).

```python
# This is easy. Simply relax the lower and upper bounds of constraints a bit
constraints = [LinearConstraint(np.ones(m), 1 - 0.001, 1 + 0.001)]
```


### Taking stock

I benchmark all combinations of the above settings. I run each combination 100 times and take the final loss and time used. A good optimiser should find a good enough local minimum and be fast, and consistently be good across all runs.[^random-seed]

[^random-seed]: Many optimisers, such as the famous stochastic gradient descent, involve some randomness. Given the same initial guess and identical code, the solutions we get can be different across different runs, due to the randomness. So the 100 runs here are not only for getting a precise average run time, but also monitors the loss when we use different random seeds.

$$
\underset{\text{Initial guess}}{\underbrace{\begin{pmatrix}
    \text{equal} \\
    \text{reg. based } \\
    \text{try many}
\end{pmatrix}}}
\times
\underset{\text{formulation}}{\underbrace{\begin{pmatrix}
    \text{nested} \\
    \text{single}
\end{pmatrix}}}
\times
\underset{V\text{ sum up to one?}}{\underbrace{\begin{pmatrix}
    \text{w/} \sum v_i = 1 \\
    \text{w/o}
\end{pmatrix}}}
\times
\underset{\text{optimiser}}{\underbrace{\begin{pmatrix}
    \text{BFGS} \\
    \text{SLSQP} \\
    \cdots
\end{pmatrix}}}
\times
\underset{\text{loss}}{\underbrace{\begin{pmatrix}
    RSS \\
    \ell^2\text{-norm}
\end{pmatrix}}}
\times
\underset{\text{constraint strictness}}{\underbrace{\begin{pmatrix}
    \text{strict} \\
    \text{relax}
\end{pmatrix}}}
$$

The figure below plots the pooled average loss for each setting. Three observations:

1. we almost always have very unstable convergence: the top 1% percentile of loss can always get to 800+, while the Stata's loss is ~60
1. reformulating the problem to single level is not useful at all...
1. authors' regression-based initial guess is **AMAZING**! It's super stable and almost always get the best minimum!

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/main/synth/figures/loss_across_settings.svg">
    <p><i>Notes</i>: The bars plot the average loss for each setting, across all runs regardless of other settings. The whiskers are the top and bottom 2.5% percentiles. The worst 1% losses are dropped. Otherwise the scale of this plot would look very bad...</p>
</figure>

Let's further zoom in to the area where loss and run time are both good, and see which settings can get us there. We also plot Stata's performance in red dashed lines. It re-confirms the above finding: regression-based initial guess (in blue) is the best! Very fast and low loss, and consistently fast and low loss!

<figure>
    {% remote_include https://raw.githubusercontent.com/harningle/useful-scripts/refs/heads/main/synth/figures/scatter_loss_time.html %}  
    <p><i>Notes</i>: I only show settings with the best 1% percentile loss. Horizontal and vertical whiskers are 95% CI for speed (seconds) and loss (RSS). Settings using regression-based initial guess are plotted in blue, and others in green. Red dashed lines are Stata's performance, averaged across ten runs.</p>
</figure>

There are some other findings:

* using single-level reformulation is a bad idea. This makes the problem higher dimensional and more difficult to solve. None of the top 1% best losses are achieved using single-level reformulation
* when the initial guess is good, the type of loss, e.g. RSS vs RMSE, does not matter much. This is not surprising: if you are already close to the minimum, then the Jacobian should be small and relatively well-behaved. So scaling the loss does not help or hurt
* giving larger tolerance for constraint violation doesn't help or hurt as well. With a good initial guess, these things are really unimportant. The same holds for whether or not removing $\sum v_i = 1$
* SLSQP seems to be faster and more stable than BFGS and Nelder-Mead. But the difference is small

In short, use regression-based guess!


### Concluding remarks

The complete synthetic control is easy to implement, but to converge to a good local minimum is not easy. The authors' regression based initial guess is the best way to help us get there!
