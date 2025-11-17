---
layout:       post
title:        "Conjugate gradient method"
date:         2025-11-17
tags:         [econ, stat, ml]
---

<p><font color="#828282">(I borrowed a lot from Shewchuk [<a href="https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf">1994</a>], <a href="https://flat2010.github.io/2018/10/26/%E5%85%B1%E8%BD%AD%E6%A2%AF%E5%BA%A6%E6%B3%95%E9%80%9A%E4%BF%97%E8%AE%B2%E4%B9%89/">https://flat2010.github.io/2018/10/26/共轭梯度法通俗讲义</a>, <a href="https://zhuanlan.zhihu.com/p/375846690">https://zhuanlan.zhihu.com/p/375846690</a>, and <a href="https://zhuanlan.zhihu.com/p/98642663">https://zhuanlan.zhihu.com/p/98642663</a>)</font></p>

<p><font color="#828282">(Code available at <a href="https://github.com/harningle/useful-scripts/tree/main/ols_big/cgm">harningle/useful-scripts</a>)</font></p>

What if you want to run an OLS but the data is too big and thus slow? This post goes into one of the fastest ways to estimate OLS, and more generally to minimise a quadratic problem: conjugate gradient method. Compared to usual gradient descent, this method picks the descent direction in a more sophisticated way and thus achieves a faster convergence. I know very little about linear algebra, so I try to write this without knowledge of fancy linear algebra like Gram-Schmidt orthogonalisation.


## Setting: When inverse/normal equation is costly

Assume the input data is $X$ and the outcome $y$. We have $n$ rows of data, and $k$ features/explanatory variables, so $X$ is a $n\times k$ matrix, and $y$ is a $n\times 1$ column vector.[^psd] Let's assume both $n$ and $k$ are very big, i.e. a lot of data points and very high dimensional. We solve for $y = X\beta$. The normal equation is $\hat\beta = (X^TX)^{-1}X^T y$. The most expensive operation here is $(X^TX)^{-1}$, both time-wise and space-wise. We can probably optimise this a bit, by using fancy multiplications like [Strassen algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm) or LU or SVD decomposition for the inverse. But it can still be slow and RAM is not happy.

[^psd]: In a general system of linear equations $Ax = b$, to use conjugate gradient method, $A$ should be positive (semi-)definite. Otherwise there is generally no guarantee that we can find the optimal solution. However, it's fine to use all methods in this blog post to solve least square problems (as long as the rank condition holds). To see it, consider the normal equation $X^Ty = X^TX\beta$. Let $\tilde b = X^Ty$, $\tilde A = X^TX$, and $\tilde x = \beta$. Then the normal equation becomes $\tilde b = \tilde A \tilde x$. $\tilde A = X^TX$ is always positive definite when $X$ is not singular, so conjugate gradient method is fine. What matters is not $X$ but $X^TX$. See Stuetzle ([2001](https://sites.stat.washington.edu/wxs/Stat538-w03/conjugate-gradients.pdf)) for a rigourous treatment.


## (Stochastic) gradient descent

To save our memory, we may want to avoid matrix inversion. That is, not using the normal equation. In the first place, OLS is *least* squares, and it's an optimisation problem: find a $\beta$ that makes $\|\|y - X\beta\|\|_2$ smallest.[^rss] This is a very well-behaved optimisation: quadratic and no constraint. Therefore, gradient-based optimisers should work well.

[^rss]: I use RSS, $\ell^2$ norm, loss, and objective function interchangeably.


### Intuition

Intuitively, gradient descent tries to iteratively approach the minimum, where each iteration we take a step towards the direction of the gradient. To visualise it, assume $k = 2$, i.e. $y = \beta_0 + \beta_1 x$. In this case, RSS is $\displaystyle\sum\limits_{i = 0}^n (y_i - \beta_0 - \beta_1 x_i)^2$, which is a simple quadratic function w.r.t. $\vec\beta$.[^vector] We plot an example RSS in a 3D space as well as its contour in 2D plane below.

<figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/ssr_3d.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) 3D surface</p>
    </figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/ssr_contour.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) Contour</p>
    </figure>
    <p><i>Notes</i>: The dark red dot and dashed lines denote location of the solution. The blue line with arrows shows the $\beta_i$ in the iteration.</p>
</figure>

[^vector]: I sometimes write $\beta$ as $\beta$ and sometimes as $\vec\beta$. They refer to the same thing; notations here are not very consistent. Keep in mind that $\beta$ is a $k\times 1$ column vector, $y$ is a $n\times 1$ column vector, and $X$ is a $n\times k$ matrix.

Imagine we have an initial guess $\vec\beta_0$ somewhere, and we make small steps towards the gradient direction iteratively, we can eventually reach the optimal location.

<div align="center"><iframe width="80%" style="aspect-ratio: 16 / 9" src="https://www.youtube-nocookie.com/embed/SmZmBKc7Lrs?si=CQL3Jruc4vt3DSdg&amp;start=1128" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>


### Math

Formally let the RSS/loss be:

$$
\ell(\vec\beta) =  (y - X\vec\beta)^T(y - X\vec\beta)
$$

. The gradient is


$$
\nabla\ell(\vec\beta) = -2X^T(y - X\vec\beta)
$$

. Suppose we have an initial guess of $\beta = \beta_0$, with an associated RSS of $\ell(\beta_0)$. We can prove that the gradient points to the steepest ascent direction. To see this, let's take an arbitrary direction $\vec v$ with norm one. The directional derivative is then

$$
\nabla_v\ell = \lim\limits_{\varepsilon\to 0}\frac{\ell(\vec\beta + \varepsilon\vec{v}) - \ell(\vec\beta)}{\varepsilon}
$$

. We then Taylor expand $\ell(\vec\beta + \varepsilon\vec{v})$ to the first order:

$$
\nabla_v\ell = \lim\limits_{\varepsilon\to 0}\frac{\ell(\vec\beta) + \nabla\ell\cdot\varepsilon\vec{v} + o(\varepsilon\vec{v}) - \ell(\vec\beta)}{\varepsilon} = \lim\limits_{\varepsilon\to 0}\frac{\nabla\ell\cdot\varepsilon\vec{v} + o(\varepsilon\vec{v})}{\varepsilon} = \nabla\ell\cdot\vec{v}
$$

.[^higher-order-terms] Finally, applying Cauchy-Schwarz inequality, we have

[^higher-order-terms]: $o(\varepsilon\vec{v})$ is a higher order term of $\varepsilon\vec{v}$ and $\vec v$ is a unit vector, so after taking the limit of $\varepsilon\to 0$, it's equal to zero.

$$
|\nabla_v\ell| = |\nabla\ell\cdot\vec{v}| \leq |\nabla\ell||\vec v| = |\nabla\ell|\cdot 1 = |\nabla\ell|
$$

. Therefore, any directional derivative is (weakly) smaller in size than the gradient. The next thing is to show, the gradient direction is ascending. Re-use the Taylor expansion trick, we have

$$
\ell(\vec\beta + \varepsilon\nabla\ell) = \ell(\vec\beta) + \nabla\ell\cdot\varepsilon\nabla\ell + o(\varepsilon\nabla\ell) = \ell(\vec\beta) + \varepsilon||\nabla\ell||^2 + o(\varepsilon\nabla\ell)
$$

. When $\varepsilon\in\mathbb{R}^+$ is sufficiently small, $o(\varepsilon\nabla\ell)$ is basically zero, and $\varepsilon\|\|\nabla\ell\|\|^2$ is always (weakly) positive, so $\ell(\vec\beta + \varepsilon\nabla\ell) \geq \ell(\vec\beta)$. Therefore, the gradient direction is the steepest direction and it's ascending.$\blacksquare$

The iteration is then

$$
\beta_{i + 1} = \beta_i - \eta_i \nabla \ell (\beta_i) = \beta_i - \eta_i -2X^T(y - X\beta_i)
$$

. I'm not going to prove convergence here. There are plenty of good proof online. If you want to implement (batched) stochastic gradient descent, then in each iteration, instead of using the full $X$ and $y$, you compute the gradient using only one random row or a few random rows of data.


### Implementation

The Python implementation is very easy.

```python
import numpy as np

np.random.seed(42)

N = 1000
M = 2
X = np.random.rand(N, M)  # Some random X and y
Y = np.random.normal(size=N)

b = np.zeros(M)  # Initial guess of beta. Set to all zeros
lr = 0.0001      # Learning rate. A small number is good here
for i in range(n_iter):
    b -= lr * (-2 * X.T @ (Y - X @ b))  # Descend along the gradient direction

# The stochastic version would be like
b = np.zeros(M)
indices = list(range(N))
rng = np.random.default_rng(seed=42)
batch_size = 1

for i in range(n_iter):
    rng.shuffle(indices)
    for j in range(0, N, batch_size):
        X_batch = X[indices[j:j + batch_size]]
        Y_batch = Y[indices[j:j + batch_size]]
        b -= lr * (-2 * X_batch.T @ (Y_batch - X_batch @ b))
```

The convergence path is like:

<figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/gd_iter_3d.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) 3D surface</p>
    </figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/gd_iter_contour.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) Contour</p>
    </figure>
    <p><i>Notes</i>: The dark red dot and dashed lines denote location of the solution.</p>
</figure>


## Steepest descent

The problem with (stochastic) gradient descent is that the learning rate needs to be carefully chosen. In this example, simply setting learning rate to be 0.01 can easily break the convergence. There are many learning rate schedules that can easily fix this. But is there a mathematically optimal learning rate? Kind-of. Steepest descent.

Suppose we start from $\vec{\beta_0}$ and, after taking a few iterations, now arrive at $\vec{\beta_i}$. The gradient of the loss function at $\vec{\beta_i}$ is $\nabla \ell(\vec{\beta_i})$. Thus the next step is $\overrightarrow{\beta_{i + 1}} = \vec{\beta_i} + \eta_i \nabla \ell(\vec{\beta_i})$, with new loss being $\ell(\overrightarrow{\beta_{i + 1}})$. The intuition of steepest descent is to make sure, for every step, we take a step size/learning rate $\eta_i$ to minimise the new loss, i.e. greedy. So

$$
\begin{equation}
\eta_i = \arg \min \ell(\overrightarrow{\beta_{i + 1}}) = \arg \min \ell(\vec{\beta_i} + \eta_i \nabla \ell(\vec{\beta_i}))\label{eq:eta}
\end{equation}
$$

. This is a simple univariate function optimisation, and all we need is to set the first derivative $\ell'$ to zero:

$$\frac{\text{d}\ell}{\text{d}\eta_i} = 0$$

. I'm not good at matrix calculus, so this first derivative is not straightforward to me. Let's call $\vec{u}(\vec{\beta_i}) = \vec{\beta_i} + \eta_i \nabla \ell(\vec{\beta_i})$. First we have the learning rate $\eta_i$, and then this positive scalar number goes to $\vec{\beta_i} + \eta_i \nabla \ell(\vec{\beta_i})$, which is a $k\times 1$ column vector. Finally $\ell$ maps this vector to a scalar, which is our loss. That is, $\eta_i\in\mathbb{R}^+ \xrightarrow{\quad\vec{u}\quad} \mathbb{R}^k \xrightarrow{\quad\ell\quad} \mathbb{R}$. So the function is scalar to vector to scalar. By chain rule, we have

$$
\frac{\text{d}\ell}{\text{d}\eta_i} = \frac{\text{d}\ell}{\text{d}\vec{u}} \frac{\text{d}\vec{u}}{\text{d}\eta_i}
$$

. $\displaystyle\frac{\text{d}\ell}{\text{d}\vec{u}}$ is the derivative of a scalar function w.r.t. a vector, which is defined as $\displaystyle\Big (\frac{\partial\ell}{\partial u_1}, \frac{\partial\ell}{\partial u_2}, \cdots, \frac{\partial\ell}{\partial u_k} \Big )$, which is a $1\times k$ row vector. This is simply the gradient, but transposed from a column vector to a row vector layout. So $\displaystyle\frac{\text{d}\ell}{\text{d}\vec{u}} = (\nabla_u \ell)^T = \Big (\nabla \ell(\overrightarrow{\beta_{i + 1}})\Big )^T$. $\displaystyle\frac{\text{d}\vec{u}}{\text{d}\eta_i}$ is the derivative of a vector-valued function w.r.t. a scalar, and it follows the usual way: $\displaystyle\frac{\text{d}\vec{u}}{\text{d}\eta_i} = \frac{\text{d}}{\text{d}\eta_i}\vec{\beta_i} + \eta_i \nabla \ell(\vec{\beta_i}) = \nabla \ell(\vec{\beta_i})$. Plug them in, we have 

$$
\begin{align}
\frac{\text{d}\ell}{\text{d}\eta_i} = \frac{\text{d}\ell}{\text{d}\vec{u}} \frac{\text{d}\vec{u}}{\text{d}\eta_i} = \Big (\nabla \ell(\overrightarrow{\beta_{i + 1}})\Big )^T \nabla \ell(\vec{\beta_i}) = 0\label{eq:sd_orthogonal}
\end{align}
$$

. This is actually an interesting F.O.C.: the optimal $\eta$ is achieved when the gradient of this step is orthogonal to the gradient of the next step. We can illustrate this condition as below. In the left panel, the current gradient direction is the blue line, and we can take steps along this direction with different sizes. After taking this step, the gradients at the next step are shown in red. The right panel shows what would happen if we take a very small step. Then this step moves towards top right, and the next step will move towards bottom left. Isn't this red dashed line wasted? If the next gradient goes somehow in slightly opposite direction to the current gradient, then next move will cancel out some of our current move's contribution to loss. Similarly, if the next gradient shares a similar direction with the current gradient, then why we need two moves? Why not just make the current move farther so we finish everything in the current direction? So the ideal case is, we make the current move and max out everything in this direction, and the next move will explore something else in the completely new gradient/direction. Therefore, the optimal step size will make two gradients perpendicular to each other. This intuition is not new: e.g., in physics we often decompose a force into x- and y-components, i.e. [orthogonal components](https://en.wikipedia.org/wiki/Force#Combining_forces).

Now solve for $\eta_i$. $\ell$ is simply sum of residual squares, i.e. $\ell(\beta) = (y - X\beta)^T(y - X\beta)$, and $\nabla \ell = -2X^T(y - X\beta)$. So

$$
\begin{aligned}
\Big ({\color{green}\nabla\ell}(\overrightarrow{\beta_{ {\color{red}i} + {\color{blue}1}}})\Big )^T \nabla \ell(\vec{\beta_i}) & = \Big ({\color{green}-2X^T\big (y - X(}\overrightarrow{\beta_{ {\color{red}i} + {\color{blue}1}}}{\color{green}) \big )}\Big )^T \nabla \ell(\vec{\beta_i}) \\
 & = \Bigg ({\color{green}-2X^T\bigg (y - X\Big (}{\color{red}\vec\beta_i} + {\color{blue}\eta_i\nabla\ell(\vec\beta_i)}{\color{green}\Big ) \bigg )}\Bigg )^T \nabla \ell(\vec{\beta_i}) \\
 & = \Big ({\color{green}-2X^T(y - X}{\color{red}\vec\beta_i}{\color{green})} + {\color{green}2X^TX}{\color{blue}\eta_i\nabla\ell(\vec\beta_i)}\Big )^T \nabla \ell(\vec{\beta_i}) \\
 & = \Big ({\color{green}\nabla\ell (}{\color{red}\vec\beta_i}{\color{green})} + {\color{green}2X^TX}{\color{blue}\eta_i\nabla\ell(\vec\beta_i)}\Big )^T \nabla \ell(\vec{\beta_i}) \\
 & = \Big ({\color{green}\nabla\ell (}{\color{red}\vec\beta_i}{\color{green})}^T + {\color{green}2X^TX}{\color{blue}\eta_i\nabla\ell(\vec\beta_i}{\color{blue})}^T\Big ) \nabla \ell(\vec{\beta_i}) \\
 & = {\color{green}\nabla\ell (}{\color{red}\vec\beta_i}{\color{green})}^T \nabla \ell(\vec{\beta_i}) + {\color{green}2X^TX}{\color{blue}\eta_i\nabla\ell(\vec\beta_i}{\color{blue})}^T \nabla \ell(\vec{\beta_i})\\
 & = 0
\end{aligned}
$$

. Thus $\displaystyle\eta_i = -\frac{\nabla\ell(\vec\beta_i)^T\nabla\ell(\vec\beta_i)}{2\nabla\ell(\vec\beta_i)^T X^TX\nabla\ell(\vec\beta_i)}$. So the recurrence relation is:

$$
\beta_{i + 1} = \beta_i - \frac{\nabla\ell(\vec\beta_i)^T\nabla\ell(\vec\beta_i)}{2\nabla\ell(\vec\beta_i)^T X^TX\nabla\ell(\vec\beta_i)}\nabla\ell(\vec\beta_i)
$$


### Implementation

```python
for i in range(n_iter):
    grad = -2 * X.T @ (Y - X @ b)
    eta = -grad.T @ grad / (2 * grad.T @ X.T @ X @ grad)
    b += eta * grad
```

Now we have a much better convergence:

<figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/sd_iter_3d.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) 3D surface</p>
    </figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/sd_iter_contour.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) Contour</p>
    </figure>
    <p><i>Notes</i>: The dark red dot and dashed lines denote location of the solution. The blue line with arrows shows the $\beta_i$ in the iteration.</p>
</figure>

Previous in gradient descent with fixed learning rate, after 50 steps, we are not quite at the solution. Now with steepest descent, we reach the solution in 10 steps or so.


## Conjugate gradient method

Now after so much preparation, we are ready to start conjugate gradient method. If we look at the convergence path above, we are moving like <font color="#000080">zig zag</font> (, which is expected due to the orthogonality of gradients). But can we make it even faster by taking something like the <font color="orange">orange path</font> below? This is conjugate gradient method!


<figure style="width: 40%">
    <img src="/assets/images/cgm_vs_sd.svg">
    <p><i>Note</i>: Purely illustrative. The actual convergence path of conjugate gradient method does not look like this.</p>
</figure>


### Steepest descent is already "steepest", so how to be even faster?

One annoying thing about the steepest descent is that it's like north-east, south-east, north-east *again*, south-east *again*, etc. Can we move north-east far enough, so we'd never need any north-east? That is, we exhaust everything in the north-east direction, and *all* later steps would no longer need to move along the same north-east direction again. This "*all*" contrasts with steepest descent, which only requires two *consecutive* steps have no overlap in direction. That is, steepest descent has move $i$ and move $i + 1$ orthogonal to each other, but we can go further by making move $i + 1$ orthogonal to *all* previous move $j\leq i$.

### The intuition and not rigourous math

Let's fix the notations first. Step $i$ moves from $\vec\beta_{i - 1}$ to $\vec\beta_i$, its direction is $\vec d_i$, its step size is $\eta_i$, same as above, and finally the error vector after the move is $\vec e_i = \vec\beta - \vec\beta_i$.[^error] Assume we are in the simple 2D case and start from $\vec\beta_0$. As we just get started and know nothing, a sensible and natural first step would head towards the gradient direction, so

[^error]: "Error" here refers to the difference between the current point and the solution. It has nothing to do with the errors in econometrics, which is $y - X\beta$. For simplicity, the direction of error vector points to the solution $\vec\beta$. It may be more natural to define it as $\vec\beta_i - \vec\beta$, but just a matter of plus or minus sign.

$$
\begin{equation}
\vec d_1 = \nabla\ell(\vec\beta_0)\label{eq:d_1}
\end{equation}
$$

. After step $1$, we arrive at $\vec\beta_1 = \vec\beta_0 + \eta_1\vec d_1$. Because we are in 2D case and want step $2$ to finish everything, so step $2$ will move exactly the error $\vec e_1$, which is

$$
\begin{equation}
\vec e_1 \overset{\text{def}}{=} \vec\beta - \vec\beta_1 = {\color{red}(X^TX)^{-1}}{\color{blue}X^Ty} - {\color{blue}\vec\beta_1}\label{eq:e_1_def}
\end{equation}
$$

. This is utterly useless, as it needs to compute $\beta$ using the brute force normal equation. Nevertheless, compare this with the old steepest descent direction, which is

$$
\begin{equation}
\nabla\ell(\vec\beta_1) = {\color{blue}X^Ty} - {\color{red}X^TX}{\color{blue}\vec\beta_1}\label{eq:nabla_1}
\end{equation}
$$

. The difference between ($\ref{eq:e_1_def}$) and ($\ref{eq:nabla_1}$) is $X^TX$! That is,

$$
\vec e_1 = {\color{red}(X^TX)^{-1}}\nabla\ell(\vec\beta_1)
$$

. So instead of moving towards the gradient direction $\nabla\ell(\vec\beta_1)$ like the steepest descent, step $2$ now needs to move along ${\color{red}(X^TX)^{-1}}\nabla\ell(\vec\beta_1)$ in order to be faster:

$$
\begin{equation}
\vec d_2 = \text{some constant/scalar }{\color{red}(X^TX)^{-1}}\nabla\ell(\vec\beta_1)\label{eq:d_2}
\end{equation}
$$

. To further understand what $\color{red}(X^TX)^{-1}$ means, let's compare the new directions with the gradient directions in steepest descent. Because we take step $1$ the same as step $1$ in steepest descent (i.e., the step $1$’s step size minimise the loss along the gradient direction), the gradient at $\vec\beta_0$ and $\vec\beta_1$ must be perpendicular, which means

$$
\begin{align}
\nabla\ell(\vec\beta_0)\cdot\nabla\ell(\vec\beta_1) &= 0 & [\text{from }(\ref{eq:sd_orthogonal})] \nonumber \\
\vec d_1 \cdot \text{some scalar }{\color{red}X^TX}\vec d_2 &= 0 & [\text{plug in }(\ref{eq:d_1})\text{ and }(\ref{eq:d_2})] \nonumber \\
\vec d_1^T{\color{red}X^TX}\vec d_2 &= 0 & [\text{divides the scalar on both sides}]\label{eq:A_orthogonal}
\end{align}
$$

. Here is the key difference between conjugate gradient descent and steepest descent: the directions are no longer orthogonal ($\vec d_1^T\vec d_2 = 0$), but are $\color{red}X^TX$-orthogonal ($\vec d_1^T{\color{red}X^TX}\vec d_2 = 0$)! This means now $\vec d_1$ and $\vec d_2$ are no longer perpendicular, but $\vec d_1$ is perpendicular to ${\color{red}X^TX}\vec d_2$, which is $\vec d_2$ after the linear transformation $\color{red}X^TX$. When a matrix left multiply a vector, it "stretches" the vector. To get a sense of $\color{red}X^TX$-orthogonal, we start with a simple circle in (A), and a pair of vectors along $x$- and $y$-axis. After some linear transformation, the circle is "pressed" and now becomes an ellipse in (B). At the same time, the vectors are also pressed. So they are no longer perpendicular but form an acute angle. (C) is another example of transformation. So when we face an ellipse contour, the orthogonality may not mean 90 degrees, but could mean perpendicular in some "stretched" way. This "stretched" orthogonality is what conjugate gradient method wants. See [Wikipedia](https://en.wikipedia.org/wiki/Derivation_of_the_conjugate_gradient_method#Conjugate_directions) for a better explanation.

<figure>
    <figure style="display: inline-block; width: 32.4%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/ols_big/cgm/figures/conjugate_vec_1.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) circle</p>
    </figure>
    <figure style="display: inline-block; width: 32.4%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/ols_big/cgm/figures/conjugate_vec_0.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) ellipse</p>
    </figure>
    <figure style="display: inline-block; width: 32.4%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/main/ols_big/cgm/figures/conjugate_vec_2.svg">
        <p style="text-align: center; margin-bottom: 0;">(C) ellipse</p>
    </figure>
</figure>

Given $\vec d_2$, what's the step size? Again, the step size should minimise the loss along this direction. The proof is trivial: because after step $2$ we reach the solution, so $\vec\beta_2 = \vec\beta$ can be reached with some step size along $\vec d_2$. And the loss is quadratic, so there is only one minimum, which is $\vec\beta$. So if step $2$ reaches $\vec\beta$, its step size minimise the loss. $\blacksquare$


To sum up, step $1$ is the same as the steepest descent: $\vec d_1 = \nabla\ell (\vec\beta_0)$, and $\eta_1$ is the step size that minimise the loss along $\vec d_1$. Step 2 has $\vec d_2$ $\color{red}X^TX$-orthogonal to $\vec d_1$, and its step size also minimises the loss along $\vec d_2$.

The remaining problem is, given this $\color{red}X^TX$-orthogonal condition, how to compute $\vec d_2$? Or to frame it differently, given a vector $\vec d_1$, how to construct a vector $\vec d_2$ s.t. $\vec d_1^T{\color{red}X^TX}\vec d_2 = 0$? Let's take a step back and look at the simple perpendicular case. Given $\vec d_1 = (1, 2)^T$, how can I find a vector $\vec d_2$ that is perpendicular to it? High school math often assume $\vec d_2 = (x, y)^T$, and then solve for $\vec d_2 \cdot \vec d_1 = x\cdot 1 + y\cdot 2 = 0$, so $\vec d_2 = (4, -2)^T$ or $(-2, 1)^T$, etc. are what we want. The key takeaway is that we often assume $\vec d_2$ as something like $x\vec i + y\vec j$, where $\vec i$ and $\vec j$ is a *basis* of the vector space, and do the dot product and solve for $x$ and $y$. In the simple *perpendicular* case, we take the unit vector of $x$- and $y$-axis as the *basis*. Now what's the equivalent in the $\color{red}X^TX$-*orthogonal* case? It's natural to take two *perpendicular* vectors for the *basis* as well, and there are such ready-to-use vectors: the last step's direction $\vec d_1$ and the current gradient $\nabla\ell (\vec\beta_1)$, because the step size minimise the loss along its direction so two consecutive gradients are orthogonal. So let's take $\vec i = \vec d_1$ and $\vec j = \nabla\ell (\vec\beta_1)$, and WLOG, let $y = 1$. Then

$$
\begin{equation}
\vec d_2 = \lambda_2\vec d_1 + \nabla\ell (\vec\beta_1) \label{eq:d_2_lambda}
\end{equation}
$$

. Now plugging ($\ref{eq:d_2_lambda}$) into ($\ref{eq:A_orthogonal}$) gives

$$
\begin{align}
\vec d_1^T {\color{red}X^TX} \big (\lambda_2\vec d_1 + \nabla\ell (\vec\beta_1)\big ) &= 0 \nonumber \\
\lambda_2 &= -\frac{\vec d_1^T {\color{red}X^TX} \nabla\ell (\vec\beta_1)}{\vec d_1^T {\color{red}X^TX} \vec d_1} \label{eq:lambda}
\end{align}
$$

. Everything in ($\ref{eq:lambda}$) is known: $\vec d_1$ is the direction of step $1$, which is the gradient we can calculate; $X^TX$ needs some heavy computation but is known; $\nabla\ell (\vec\beta_1)$ is a gradient we can compute as well. Now plug ($\ref{eq:lambda}$) and ($\ref{eq:d_2_lambda}$) into ($\ref{eq:eta}$), we have the step size

$$
\begin{equation}
\eta_2 = -\frac{\vec d_2^T\nabla\ell (\vec\beta_1)}{2\vec d_2^TX^TX\vec d_2}
\end{equation}
$$

. By induction, for step $i$, its direction and step size are

$$
\begin{align}
\vec d_1 &= \nabla\ell (\vec\beta_0) \\
\vec\beta_i &= \vec\beta_{i - 1} + {\color{red}\eta_i} {\color{blue}\vec{d_i}} \\
{\color{blue}\vec d_i} &= {\color{green}\lambda_i} \vec d_{i - 1} + \nabla\ell(\vec \beta_{i - 1}) \\
{\color{green}\lambda_i} &= -\frac{\vec d_{i - 1}^T X^TX \nabla\ell (\vec\beta_{i - 1})}{\vec d_{i - 1}^T X^TX \vec d_{i - 1}} \\
{\color{red}\eta_i} &= -\frac{\vec d_i^T\nabla\ell (\vec\beta_{i - 1})}{2\vec d_i^TX^TX\vec d_i}
\end{align}
$$

. This is all we need!


### Implementation

```python
d = [None for _ in range(n_iter)]
d[1] = jac(X, Y, b)  # First step is just the gradient at the initial guess

for i in range(1, n_iter):
    grad = jac(X, Y, bs[i - 1])
    if d[i] is None:  # Only need to compute from step 2 onwards
        lambdaa = -d[i - 1].T @ X.T @ X @ grad / (d[i - 1].T @ X.T @ X @ d[i - 1])
        d[i] = lambdaa * d[i - 1] + grad
    eta = -(d[i].T @ grad) / (2 * d[i].T @ X.T @ X @ d[i])
    b += eta * d[i]
```

As the figures show, we find the solution in precisely two steps, as there are two unknowns $\beta_0$ and $\beta_1$. This is expected and what we wish for: we move along one direction, exhaust everything in that direction, so the later steps would never need to revisit this direction any more. Since we have two unknowns, we are on a 2D plane. There are two perpendicular directions, i.e. the basis has two components, so one step kills one direction, and the second step finish the other perpendicular direction.

<figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/cgm_iter_3d.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) 3D surface</p>
    </figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/cgm_iter_contour.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) Contour</p>
    </figure>
    <p><i>Notes</i>: The dark red dot and dashed lines denote location of the solution. The blue line with arrows shows the $\beta_i$ in the iteration.</p>
</figure>

And if we compare steepest descent and conjugate gradient method side by side, the difference is clear. Steepest descent steps themselves are orthogonal, but conjugate gradient method has $\color{red}X^TX$-orthogonal steps. These new steps are "orthogonal" in the ellipse sense, i.e. they move along the major and/or minor axes of the contour.

<figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/sd_iter_contour.svg">
        <p style="text-align: center; margin-bottom: 0;">(A) Steepest descent</p>
    </figure>
    <figure style="display: inline-block; width: 49%; margin-bottom: 0;">
        <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/ols_big/cgm/figures/cgm_iter_contour.svg">
        <p style="text-align: center; margin-bottom: 0;">(B) Conjugate gradient method</p>
    </figure>
    <p><i>Notes</i>: The dark red dot and dashed lines denote location of the solution. The blue line with arrows shows the $\beta_i$ in the iteration.</p>
</figure>


## Some final notes

### Gram-Schmidt orthogonalisation

To decide the direction of the next step, ($\ref{eq:d_2_lambda}$) solves for a $X^TX$-orthogonal vector. And we do this iteratively: direction 2 is $X^TX$-orthogonal to direction 1, direction 3 is then going to be $X^TX$-orthogonal to direction 2, etc. This is essentially [Gram-Schmidt orthogonalisation](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process). This method takes a set of $k$ linearly independent vectors, and spits out a set of $k$ vectors that are orthogonal to each other, and span the same subspace as the original $k$ input vectors. Suppose we have $k$ linearly independent vectors $\vec v_1, \vec v_2, \cdots, \vec v_k$, and the output vectors are $\vec u_1, \cdots, \vec u_k$. $Without proof, the Gram-Schmidt process is as follows:

1. keep the first vector $\vec k_1$ untouched: $\vec u_1 = \vec v_1$
1. we project the second vector $\vec v_2$ onto the direction of $\vec v_1$, and subtract the projection from $\vec v_2$: $\vec u_2 = \vec v_2 - \vec v_2\cdot\dfrac{\vec v_1}{\|\|\vec v_1\|\|}$. The intuition is straightforward: we can think this a decompose a force $\vec F$ into two directions, or we can think this as some sort of the residuals in Frisch–Waugh–Lovell theorem
1. the third vector $\vec v_3$ is projected to the direction of $\vec v_1$ and $\vec v_2$, and we get the remainder: $\vec u_3 = \vec v_3 - \vec v_3\cdot\dfrac{\vec v_2}{\|\|\vec v_2\|\|} - \vec v_3\cdot\dfrac{\vec v_1}{\|\|\vec v_1\|\|}$
1. $\cdots$

The $\vec u$’s are what we want. We can apply this method to the search of $\vec d$’s in ($\ref{eq:d_2_lambda}$).


### Is it really fast?

Short answer is no. We are almost never able to beat things like `np.linalg.solve` or `scipy.optimize.least_squares`. And theoretically, although conjugate gradient method converges in fewer steps, each step may require more computation, so the overall speed may not be much faster than steepest descent. My suggestion is, for small datasets that can fit in the RAM or VRAM, use `np.linalg.solve` or `cupy.linalg.solve` on GPU. If the dataset is really big, go straight to PyTorch to optimise the loss using SGD or whatever. It's a quadratic problem with no constraint, so any sensible optimiser should reach to the unique optimal solution.
