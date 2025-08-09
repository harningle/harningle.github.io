---
layout:       post
title:        "Tesseract: Fine-tuning and reproducibility"
date:         2025-07-26
last_updated: 2025-08-09
tags:         [ml]
---

As part of our effort to compile a [Formula 1 database](https://github.com/harningle/fia-doc), we need to read [PDF documents](https://www.fia.com/documents), [some](https://www.fia.com/system/files/decision-document/2025_austrian_grand_prix_-_final_race_classification.pdf) of which are essentially images rather than texts. So OCR is a must. Nowadays, OCR seems to be trivial with Gemini or ChatGPT. However, it may not be [free](https://ai.google.dev/gemini-api/docs/pricing), not very [replicable](https://ai.google.dev/gemini-api/docs/prompting-strategies#under-the-hood),[^top-p] not consistent in style,[^consistency] and, most importantly, not able to handle the page layout or tables (see [the previous post on PDF parsers]({% link _posts/2025-06-11-pdf-parsers-a-benchmark.md %})). So more "traditional" OCR tools are preferred. Among them [Tesseract](https://github.com/tesseract-ocr/tesseract) seems to be the best.[^why-tesseract] Yet, Tesseract has its own problems: the accuracy is not even close to the above large models, and it is not fully reproducible either. We can improve OCR accuracy by fine tuning, and pin some package versions to help with reproducibility. This post walks through all these things. (But we still can't achieve fully deterministic output).

[^top-p]: E.g. due to [nucleus sampling](https://en.wikipedia.org/wiki/Top-p_sampling).
[^consistency]: Sometimes the LLM response can begin with useless sentences like "sure I'm happy to help you". Such random sentences can break some if not all later processing. 
[^why-tesseract]: There are a few requirements. First, I'd like to avoid any vision language models, as the reproducibility is often not guaranteed. Second, it should preserve the PDF layout. That is, I don't expect it gives me a ready-to-use `.txt` or markdown output, but it should at least tell me which texts are in which locations on the page, so I can handle them myself. If you search "OCR" in GitHub, the repo. satisfying both criteria and with the most stars is [Tesseract](https://github.com/tesseract-ocr/tesseract).


## Why Tesseract is good, and why it's not perfect

The goal is to get data from a PDF like below:

<div style="display: flex; justify-content: center; height: 50vh; margin-bottom: 3vh;">
    <object data="https://www.fia.com/sites/default/files/2025_11_aut_f1_r0_timing_raceprovisionalclassification_v01.pdf" width="100%" type="application/pdf"></object>
</div>

I've tried Gemini, ChatGPT, and Mistral OCR. None of them gives consistently correct answer, even after a lot of prompt engineering. If we move away from those "modern" large models, the best canonical OCR engine seems to be Tesseract. The most important feature we care is that it preserves spatial relationship. That is, it tells us the text "Lando Norris" is inside the rectangle of coordinates e.g. $(100, 200, 300, 400)$. These coordinates enable us to easily parse the tables. There are other OCR models/packages with superior performance, but almost none of them returns the positioning of texts like Tesseract, which means table structure is lost.

As said before, Tesseract has two problems: (1) the accuracy is not good, and (2) it's not reproducible.

**Performance.** Tesseract is built on LSTM, so of course we shouldn't expect it to have the same performance as Gemini. But since all our input PDFs are written in the same font. It's fairly easy to fine-tune Tesseract on our PDFs to get better accuracy. (In the end we effectively get 100% whoa!)

**Reproducibility.** This is more tricky. Many users report differences in OCR output across OS.[^users-report] I [also experienced the same](https://github.com/tesseract-ocr/tesseract/issues/3812#issuecomment-3082038158). To make it more mysterious, even if using the same docker image, the difference persists among different OS.[^more-users-report] At this point, it may seem to be hardware/driver related, but no. On the same physical machine and using the same Tesseract version, my Windows and Windows Subsystem for Linux give [different results](https://github.com/tesseract-ocr/tesseract/issues/3812#issuecomment-3082038158). I give up here. But for our case, it isn't a big problem. E.g., on Windows it says the text is 99.92% "Lewis", and on Mac it's 99.96% "Lewis". It really doesn't matter, as long as "Lewis" is the most probable text. Therefore, if we do have a very accurate model, then the tiny difference won't affect the final output. So *improving performance helps reproducibility*.

[^users-report]: E.g., [tesseract-ocr/tesseract#3812](https://github.com/tesseract-ocr/tesseract/issues/3812), [tesseract-ocr/tesseract#1254](https://github.com/tesseract-ocr/tesseract/issues/1254), and [https://groups.google.com/g/tesseract-ocr/c/Ey9K-v4gJwk](https://groups.google.com/g/tesseract-ocr/c/Ey9K-v4gJwk).
[^more-users-report]: E.g. both of [https://groups.google.com/g/tesseract-ocr/c/Ph2PT5WWXQU/m/_9mZh4oaBAAJ](https://groups.google.com/g/tesseract-ocr/c/Ph2PT5WWXQU/m/_9mZh4oaBAAJ) and [https://groups.google.com/g/tesseract-ocr/c/PKRDrGa755U/m/9nqriZLuBwAJ](https://groups.google.com/g/tesseract-ocr/c/PKRDrGa755U/m/9nqriZLuBwAJ) use the same docker image on different OS platforms, and get different results. I also tried the same docker image on Mac and Windows, and got difference in output.


## Fine-tuning

To get better accuracy, we decided to fine tune Tesseract model. In some sense, it's not a very typical fine tuning from today's perspective. It's now 2025, and you would likely expect things like `import torch`, and loading all images and labels onto the GPU, etc. This is not how Tessearct works. It's, both the training and inference, written in C++. So the easiest way to train the model is simply using their training script. However, I found their doc a bit outdated and confusing. Here is how I ran it (as of July 2025).


### Build training tools

<p><font color="#828282">(If you have time, read all the docs at <a href="https://github.com/tesseract-ocr/tesstrain">tesseract-ocr/tesstrain</a>, <a href="https://tesseract-ocr.github.io/tessdoc/tess5/TrainingTesseract-5.html">How to train LSTM/neural net Tesseract</a>, and <a href="https://tesseract-ocr.github.io/tessdoc/Compiling-%E2%80%93-GitInstallation">Installing Tesseract from Git</a>)</font></p>

First of all, start with a fresh Ubuntu environment, ideally in a docker container. Then install all dependencies for (compiling) Tesseract itself and its training tools.

```bash
sudo apt-get update
sudo apt-get install -y automake ca-certificates g++ git libtool \
                        libleptonica-dev make pkg-config libpango1.0-dev \
                        libicu-dev libcairo2-dev wget python3.12-venv \
                        python3-pip
```

The training requires compilation from the source code, so let's pull it from the GitHub repo. and compile it.

```bash
git clone --depth 1 https://github.com/tesseract-ocr/tesseract.git
cd tesseract
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
make training
sudo make training-install
cd ..
git clone --depth 1  https://github.com/tesseract-ocr/tesstrain.git
cd tesstrain
```

Several things to note here:

* to enforce reproducibility in the training stage, it may be good to clone a specific commit, rather than the lastest main branch
* we may or may not want to have `-O2` optimisation and/or other features in compilation. I believe (but no evidence) this also matters for reproducibility
* if you have a powerful CPU with many cores, you can run multiple processes during compilation.

Below is what I did:

```bash
# Get source code at tesseract-ocr/tesseract@d3e50cf
git init tesseract
cd tesseract
git remote add origin https://github.com/tesseract-ocr/tesseract.git
git fetch --depth 1 origin d3e50cfb0674574dad8a8b0ac2117fd019b78b5c
git checkout FETCH_HEAD

# Compile, with -O2 and six processes in parallel
./autogen.sh
./configure 'CXXFLAGS=-O2'
make -j 6
sudo make install
sudo ldconfig
make training -j 6
sudo make training-install

# Get the training scripts as well, also at a specific commit
cd ..
git init tesstrain
cd tesstrain
git remote add origin https://github.com/tesseract-ocr/tesstrain.git
git fetch --depth 1 origin 405346a3a67d8e4e049341d1da6a4b752e0b8351
git checkout FETCH_HEAD
```


### Prepare model and training data

When we say "training", we actually mean fine tuning. We are not starting from scratch (of course we can but not necessary); we build on the existing very good Tesseract English OCR model. So let's download this English model as well.

```bash
mkdir -p usr/share/tessdata
wget -O usr/share/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/refs/heads/main/eng.traineddata
```

Note that `usr/share/tessdata` is NOT a typo. It's `usr/` rather than <code class="language-plaintext highlighter-rouge"><b>/</b>usr/</code>. See the doc [here](https://github.com/tesseract-ocr/tesstrain?tab=readme-ov-file#train). The default model folder is the *relative* path `./usr/share/tessdata`, rather than the architecture-independent data folder `/usr/share/tessdata` at root.

The training data need to put in the `data` folder. If we want to have a model name of `foo`, then training samples should be in `./data/foo-ground-truth`. Each training sample consists of two files: the image and the text. The file naming should be `abc.png` and `abc.gt.txt`. The training script accepts `.png` and `.tif`.

One last note is that the ground truth can't be an empty string. It's very possible that the ground truth should be empty, but tesseract simply does not accept that. In my case, I use "placeholder" char. for empty string. That is, when the ground truth should be an empty string, I put something like `".\n"` in `abc.gt.txt`. I then manually process these "placeholder" char. after OCR. See [tesseract-ocr/tesstrain#304](https://github.com/tesseract-ocr/tesstrain/issues/304) for details.


### Install Python dependencies

The last bit is the Python environment. It's the same as usual Python project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Now we have everything for training. The folder structure should look like:

```plaintext
tesstrain/                   <-- We are currently inside this folder
├─ .venv/                    <-- Python virtual env. created above
│  ├─ .../
├─ data/
│  └─ foo-ground-truth/      <-- <MODEL_NAME>-ground-truth
│     ├─ 1.png
│     ├─ 1.gt.txt
│     ├─ 2.png               <-- Must be xyz.png
│     ├─ 2.gt.txt            <-- Must be xyz.gt.txt
│     ├─ ...
├─ usr/                      <-- Pretrained model
│  └─ share/
│     └─ tessdata/
│        └─ eng.traineddata
├─ Makefile                  <-- Other files and folders in tesstrain repo.
├─ shuffle.py
├─ ...
```


### Training

Finally we can run the training!

```bash
make training MODEL_NAME=foo START_MODEL=eng PSM=7 MAX_ITERATIONS=5000
```

* `training`: train/fine-tune a model
* `MODEL_NAME`: name of the model to be trained. It must match the training data folder name. That is, if `MODEL_NAME=bar`, then the images and texts for training should be inside `data/bar-ground-truth`
* `START_MODEL`: from what model to start fine-tuning. We usually want English `eng`. The default is nothing, i.e. train a brand new model from scratch
* `PSM`: page segmentation mode. Check [TESSERACT(1) Manual Page](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#options). Here I used `PSM=7`, which is a single line text. I found PSM matters *a lot* in both training and inference. Check [https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/) for detailed explanations and examples of different PSMs. `PSM=13` may deserve some extra attention
* `MAX_ITERATIONS`: max iterations. You can leave it blank. I found my training converges quite quickly

A more detailed explanation can be found [here](https://github.com/tesseract-ocr/tesstrain?tab=readme-ov-file#train). Please note that it's not very documented. You'd better play around with them a bit and see how they affect the training. E.g. `FINETUNE_TYPE` is not explained anywhere and I found this parameter had zero effect on my training.[^finetune-type]

[^finetune-type]: Maybe this parameter is not used any more. See [tesseract-ocr/tesstrain#316](https://github.com/tesseract-ocr/tesstrain/issues/316).

You would probably see some downloads, some image box creation, etc. After those one-off upfront things, the training log looks like:

```plaintext
[...]
At iteration 215/1200/1200, mean rms=0.776%, delta=1.771%, BCER train=7.510%, BWER train=13.463%, skip ratio=0.000%, New best BCER = 7.5
10 wrote best model:data/foo/checkpoints/foo_7.510_215_1200.checkpoint wrote checkpoint.
2 Percent improvement time=22, best error was 10.217 @ 201
At iteration 223/1300/1300, mean rms=0.740%, delta=1.624%, BCER train=6.659%, BWER train=12.055%, skip ratio=0.000%, New best BCER = 6.6
59 wrote best model:data/foo/checkpoints/foo_6.659_223_1300.checkpoint wrote checkpoint.
2 Percent improvement time=20, best error was 8.377 @ 207
[...]
Finished! Selected model with minimal training error rate (BCER) = 0.709

lstmtraining \
--stop_training \
--continue_from data/foo/checkpoints/foo_checkpoint \
--traineddata data/foo/foo.traineddata \
--model_output data/foo.traineddata
Loaded file data/foo/checkpoints/foo_checkpoint, unpacking...
```

The trained model, at each checkpoint, is saved in `data/foo/tessdata_best`.


### Evaluation

The training stops when hit `MAX_ITERATIONS`. To understand the performance, we can plot the loss in training and the holdout sample:

```bash
make plot MODEL_NAME=foo
```

It produces several figures, but the most useful one is `data/foo/foo.plot_cer.png`. You can interpret the two loss lines as a usual training.

<figure>
    <img src="/assets/images/tesstrain.svg">
</figure>

The figure says at iteration 234/1,500, the loss in the validation set hits minimum. If we want the model at this step, we can find it as `data/foo/tessdata_best/foo_4.729_234_1500.traineddata`. The "234" and "1500" is simply the iteration No., and "4.729" is the loss of the training set at that time. It doesn't matter. The iteration itself can uniquely identify the model file. If you want the model at iteration ~2,200, where both training and validation loss are small, the model is available at `data/foo/tessdata_best/foo_1.936_275_2200.traineddata`.


## Use the fine-tuned model

Now we have the model `data/foo/tessdata_best/foo_1.936_275_2200.traineddata`. To use it, the easiest way is simply copying it to `TESSDATA_PREFIX/tessdata` folder, which is `/usr/share/tessdata` by default. To do this:

```bash
cp data/foo/tessdata_best/foo_1.936_275_2200.traineddata \
   /usr/share/tessdata/foo.traineddata
```

And you should now have:

```plaintext
/usr/
├─ share/
│  └─ tessdata/
│     ├─ foo.traineddata
│     ├─ ...
├─ ...
```

Then we can use our own "foo" model by:

```bash
tesseract image.png output_text -l foo
```

The OCR results will be stored in `output_text.txt`. Done!
