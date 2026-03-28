---
layout:       post
title:        "Excel hell: Lessons learnt the hard way"
date:         2026-03-29
tags:         tips
---


<figure style="width: 50%"><img src="https://i.redd.it/fshqfne5os291.jpg"></figure>

I save all the sh\*t about Excel here...


## Precision loss for large integers

Let's first create a csv like this:

```text
whatever,col
9,123456789
10,1234567890
11,12345678901
12,123456789012
13,1234567890123
14,12345678901234
15,123456789012345
16,1234567890123456
17,12345678901234567
18,123456789012345678
19,1234567890123456789
```

. This should be super easy and have exactly no ambiguity at all. However, when we open it in Excel, it interprets `col` data type wrongly:

<figure style="width: 40%"><img src="/assets/images/excel_shit_1.png"></figure>

. First, when the integer has 16 digits or more, Excel takes it as a string, not an integer... Second, it use the scientific notation for large numbers, which itself is fine and no precision loss happens yet. Then we save it as a new `.csv` file, without no touch in any cell. We should expect an identical `.csv` file, but no... We get:

```text
whatever,col
9,123456789
10,1234567890
11,12345678901
12,1.23457E+11
13,1.23457E+12
14,1.23457E+13
15,1.23457E+14
16,1234567890123456
17,12345678901234567
18,123456789012345678
19,1234567890123456789
```

. We permanently lose everything after the fifth significant digit. `.csv` is a plain text file, so no way to recover... Excel claims to have 15 significant digits, but NO!!! It's only five digits in this example! This is well known problem, and I believe it is a "feature" not a "bug", so won't be fixed... See [Numbers in csv-file longer then 15 digit are lost after importing CSV to Excel. - Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/5035135/numbers-in-csv-file-longer-then-15-digit-are-lost) and [Numeric precision in Microsoft Excel - Wikipedia](https://en.wikipedia.org/wiki/Numeric_precision_in_Microsoft_Excel).


## Style is data, and data is style

In almost all other proper data software, data is data, and display format is an entirely independent thing. When we change the number display format, the underlying data remains untouched. This is not true in Excel. Assume we have:

```text
col
1234.56
1234.56
```

Let's format cell A3 into "Accounting", i.e. add thousands separator, and save it.

<figure style="width: 80%"><img src="/assets/images/excel_shit_2.png"></figure>

Now it becomes

```text
col
1234.56
" 1,234.56 "
```

. Yes, Excel funnily converts a float into a string with padding and trailing whitespaces, for a *style* change...
