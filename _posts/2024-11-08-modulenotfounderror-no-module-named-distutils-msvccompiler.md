---
layout:       post
title:        "ModuleNotFoundError: No module named 'distutils.msvccompiler'"
date:         2024-11-08
tags:         [tips]
---

When re-running some old Python script in a new env., I sometimes get `ModuleNotFoundError: No module named 'distutils.msvccompiler'`, even if I've installed all dependencies already.[^not-just-me] It's probably due to the fact that some old packages we use do not take care of the removal of old MSVC modules from setuptools ([pypa/setuptools#3505](https://github.com/pypa/setuptools/pull/3505)). Usually, `pip install setuptools` should fix it. If not, try something like `pip install "setuptools<68"` or `pip install "setuptools<65"` and so on.

[^not-just-me]: And it's not just me. Many other people are getting the same error: [numpy/numpy#27405](https://github.com/numpy/numpy/issues/27405), [pyodide/pyodide#2971](https://github.com/pyodide/pyodide/issues/2971), [https://askubuntu.com/q/1239829](https://askubuntu.com/q/1239829), [https://stackoverflow.com/q/77233855](https://stackoverflow.com/q/77233855), [https://stackoverflow.com/q/79063140](https://stackoverflow.com/q/79063140).
