---
layout:       post
title:        "pandas: inplace, view, and copy"
date:         2025-01-27
tags:         [python, stats]
---

In recent pandas versions, I sometimes get <code class="highlighter-rouge">FutureWarning: Downcasting behavior in `replace` is deprecated</code> warnings when I use `inplace=True`, e.g. `df.x.replace({1: True, 0: False}, inplace=True)`. And I never understand the `SettingWithCopyWarning` either. I thought inplace operations should be preferred, as they are faster and they save memory, because they do things inplace rather than copying the object. After some readings, I realise I was wrong. The takeaways are

* copy is generally safer,
* a copy is almost always created, no matter whether I do it inplace or copy it, and
* the speed difference between inplace and copy is minimal.


## View vs copy: the notorious `SettingWithCopyWarning` and how to be safe

Inplace, copy, view, reference, ... What are they exactly? Look at the code below.

```python
>>> a = [0, 1, 2]  # Create a brand new object `a` in memory
>>> b = a          # `b` gets `a`, but it "points" to `a`, instead of copy
>>> print(a)
[0, 1, 2]
>>> print(b)
[0, 1, 2]
>>> print(id(a) == id(b))  # `a` and `b` has the same memory address
True
>>> b[1] = 'xxx'           # Therefore, change `b` will also change `a`
>>> print(b)
[0, 'xxx', 2]
>>> print(a)
[0, 'xxx', 2]
```

It's usually very clear in native Python when we get a copy and when we get a reference. In C terminology, `b` kind of works like a pointer which points to `a` and automatically dereferences itself when you use it.[^names-in-python] However, the same thing is a bit confusing in pandas. Again, check the code below.

[^names-in-python]: I found the [blog post](https://nedbatchelder.com/text/names.html) and the PyCon 2015 [presentation](https://nedbatchelder.com/text/names1.html) by Ned Batchelder, who is a [CPython core developer](https://discuss.python.org/t/vote-to-promote-ned-batchelder/57668), extremely helpful in explaining how these things work in Python.


```python
>>> import pandas as pd
>>> df = pd.DataFrame({'col1': [0, 1, 2, 3],
                       'col2': [False, False, True, True]})
>>> df2 = df[df.col2 == True]  # A subset of `df`. A copy? A view/reference?
>>> print(df)
   col1   col2
0     0  False
1     1  False
2     2   True
3     3   True
>>> print(df2)
   col1  col2
2     2  True
3     3  True
>>> df2.col1 = 999              # Get a warning here
<python-input-71>:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df2.col1 = 999
>>> df2  # `df2` changed as expected
   col1  col2
2   999  True
3   999  True
>>> df  # `df` untouched. Is it expected or not?
   col1   col2
0     0  False
1     1  False
2     2   True
3     3   True
>>> df[df.col2 == True].col1 = 999  # I want to change `df` inplace
<python-input-75>:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df[df.col2 == True].col1 = 999
>>> df                              # But why it is not changed?
   col1   col2
0     0  False
1     1  False
2     2   True
3     3   True
```

The above is a very typical workflow for a data project: we have a raw input `df`, and we work on a subset of it, which is `df2`. The problem is, is the subset `df2` a "pointer"/view pointing to `df`, or is it a copy, an entirely new object? 
"View" is a pandas terminology. It is more or less the same as a "pointer". Basically a view views the memory of the original object. In the above example, `df2` is a copy, but why do we still get the `SettingWithCopyWarning`? Then if we want to change certain values in `df` *inplace*, why nothing happens with the same warning? 

The short answer is, we don't really know when it's a copy or a view. I know, it's a deterministic code so must return a deterministic thing, but it's just so confusing. Therefore, whenever returning a copy and view is confusing, pandas gives a warning, when things work as expected, and when things fail.[^further-reading-on-copy]

[^further-reading-on-copy]: If you don't have much time, read the [blog post](https://jorisvandenbossche.github.io/blog/2022/04/07/pandas-copy-views/) by [Joris Van den Bossche](https://jorisvandenbossche.github.io/pages/about.html), who is a pandas core developer. His blog post is so clear and concise. If you have a lot of time, read [PDEP-7](https://pandas.pydata.org/pdeps/0007-copy-on-write.html) and pandas doc on [returning a view versus a copy](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy) and on [Copy-on-Write (CoW)](https://pandas.pydata.org/docs/user_guide/copy_on_write.html). They are much longer but cover more details. The long discussion at [pandas-dev/pandas#57734](https://github.com/pandas-dev/pandas/issues/57734) is also worth reading.

> pandas' current behavior on whether indexing returns a view or copy is confusing. Even for experienced users, it's hard to tell whether a view or copy will be returned[.]
> 
> <div style="text-align: right">—<a href="https://pandas.pydata.org/pdeps/0007-copy-on-write.html">PDEP-7: Consistent copy/view semantics in pandas with Copy-on-Write</a></div>

Currently, if we want to get rid of `SettingWithCopyWarning` and be safe, using `df2 = xxxxx.copy()` explicitly will do the job. That is, explicitly copy the dataframe when you don't want to change the original.


## No sorry, inplace does not save memory

Now we know what a copy is and what a view is, and we know it's confusing so we frequently get `SettingWithCopyWarning` whenever there is ambiguity. But how is that related to inplace operations? I (wrongly) learnt from many many places and people that inplace saves memory, because it does not create a copy. It "sounds" like inplace directly works on the original object (or its view), without copying it. This is generally wrong! Regardless of inplace or not, we *always* need additional memory. It's very clear in the [source code](https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/core/generic.py#L7885-L8153) that `inplace=True` or `inplace=False` only affects whether we return `None` or return a copy, but during the calculation, the copy is *always* created!

To see the actual memory usage, consider the following example: we create a dataframe with one single column and 100 million rows, filling with all one's (`int64`). Then we replace all one's with 100's, and then sort the dataframe. The dataframe roughly takes $\frac{10^8 * 64\text{ bit}}{1024\times 1024} = 762$ MB. [pythonprofilers/memory_profiler](https://github.com/pythonprofilers/memory_profiler) confirms that `df` does take 763MB RAM. Then the inplace replace uses additional **191.062** MB memory, and the sort uses **763.141** more memory. Note that the 763.141 MB is basically the same size of the original dataframe. So inplace actually *copies* the dataframe!

```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     5   91.312 MiB   91.312 MiB           1   @profile(precision=3)
     6                                         def foo():
     7  854.438 MiB  763.125 MiB           1       df = pd.DataFrame(1, index=range(100_000_000), columns=['a'])
     8 1045.500 MiB  191.062 MiB           1       df.a.replace({1: 100}, inplace=True)
     9 1808.641 MiB  763.141 MiB           1       df.sort_values('a', inplace=True)
    10                                         
    11 1808.641 MiB    0.000 MiB           1       return df
```

The not inplace version uses almost identical **191.047** MB for replace and **763.141** MB for sort.

```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     5   90.078 MiB   90.078 MiB           1   @profile(precision=3)
     6                                         def foo():
     7  853.203 MiB  763.125 MiB           1       df = pd.DataFrame(1, index=range(100_000_000), columns=['a'])
     8 1044.250 MiB  191.047 MiB           1       df.a = df.a.replace({1: 100}, inplace=False)
     9 1807.391 MiB  763.141 MiB           1       df = df.sort_values('a', inplace=False)
    10                                         
    11 1807.391 MiB    0.000 MiB           1       return df
```

Inplace does *not* generally save memory![^community-answers]

[^community-answers]: Many people have asked whether inplace actually saves memory. And the answer is mixed, which is why I got the wrong impression that inplace saves memory. For example, people say no: [https://stackoverflow.com/a/47253732](https://stackoverflow.com/a/47253732/12867291), and people say yes: [https://stackoverflow.com/a/74751146](https://stackoverflow.com/a/74751146/12867291).

> no inplace creates a new copy then assigns the pointer
>
> these are equivalent from a memory perspective
> 
> <div style="text-align: right">—<a href="https://github.com/pandas-dev/pandas/issues/20639#issuecomment-379597417">Jeff Reback (Two, Sigma, pandas core developer) at pandas-dev/pandas#20639</a></div>

> inplace does not generally do anything inplace
> 
> <div style="text-align: right">—<a href="https://github.com/pandas-dev/pandas/issues/16529#issuecomment-323890422">Jeff Reback (Two Sigma, pandas core developer) at pandas-dev/pandas#16529</a></div>


## Speedwise, all are also very close

I run a simple benchmark test between inplace and not inplace operations, i.e. `df.sort_values(xx, inplace=True)` vs `df = df.sort_values(xx)`. The test dataset has three integer columns and ten million rows. 20% of the cells are missing. The figure below shows there is almost *no* speed difference between inplace and not inplace operations!

<figure>
    <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/inplace/benchmark_simple.svg">
    <p><i>Notes</i>: I normalise the time for inplace operations to unit variance, so all bars can fit into the same $y$ scale. The error bars show 95% confidence interval.</p>
</figure>

Many other people have done similar benchmark before, but the results are very mixed. E.g. [this blog plost](https://blog.dailydoseofds.com/p/a-misconception-about-pandas-inplace) finds inplace operations are *slower* than not inplace ones across the board. But [this post](https://medium.com/@akashsingh9303/optimizing-pandas-understanding-the-impact-of-inplace-true-vs-inplace-false-69965ad127ad) shares similar results with me that there are no significant speed difference. But I believe the big picture is the time complexity of both is very close,

Another popular argument for not using `inplace=True` is that it kills method chaining. Compare

```python
df = df.sort_values('a')
df = df.fillna(0)
df = df.replace(3, 5)
```

vs

```python
df = df.sort_values('a').fillna(0).replace(3, 5)
```

If you use inplace, because inplace always returns `None`, there is not way to chain the methods, and you have to write three methods in three lines. But does method chaining really improve efficiency? Not really, as shown below.

<figure style="width: 50%">
    <img src="https://github.com/harningle/useful-scripts/raw/refs/heads/main/inplace/benchmark_chaining.svg">
    <p><i>Notes</i>: The error bars show 95% confidence interval.</p>
</figure>

Unlike Spark or Polars, pandas does not do lazy execution (I believe). So chaining or not chaining doesn't matter much. Since inplace doesn't matter for every single operation, the chained and unchained operation should also perform similarly in terms of speed. This is expected.


## Conclusion

Don't use `inplace=True` any more. There is no memory gain, no speed gain, and involves some risks. Just copy the dataframe, as you are doing it anyways already.
