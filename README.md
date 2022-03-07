# Code repository for Diploma Thesis

- Thesis Title: **Generative Adversarial Networks for pose and style selection in fashion design applications**
- Author: **Athanasios Charisoudis achariso@ece.auth.gr**
- Supervisors: **Prof. Pericles Mitkas, Dr. Antonios Chrysopoulos**
- Department: **Electrical and Computer Engineering, Aristotle University of Thessaloniki, Greece**

## Thesis Files

### Abstract

_Generative Modelling, a branch of Machine Learning that focuses on generating realistic-looking samples, has
traditionally constituted the upper bound of what Machine and Deep Learning models can achieve. This regime has
completely changed the past years, especially after 2014, when I. Goodfellow presented his idea for a generative model
comprising two competing neural networks: the Generative Adversarial Network of GAN for short. Subsequently, a plethora
of models based on GAN have been proposed with impressive results, some of which, principally in the context of image
generation, surprise even an experienced human vision system._

_Concurrently, more and more research is devoted during the last decades around the development of techniques for
demystifying the notion of fashion and fashion trends. Among its purposes, is creating artificial intelligence systems
that provide help in the process of designing new garments as well as in the process of conducting better and more
well-targeted purchases. In an endeavour to apply modern machine learning techniques to automate generation and editing
of fashion images, in this project we employ Generative Adversarial Networks. In particular, we design and utilize a
multi-tool for automatic editing of fashion images, equipped with four (4) fundamental operations: pose change, cloth
extraction, style matching and on-demand realistic fashion images generation._

_In order to achieve our goals, we train four models based on the Generative Adversarial Network in fashion image (i.e.
images of garments as well as human models advertising them) datasets, giving the corresponding outcomes at the end. It
is our firm belief that further developments of such models will play a central role in fashion design and especially in
clothes distribution through e-commerce systems in the near future, which has made us focus zealously on implementing an
effective intelligent tool for fashion image editing in this work._

### Report \& Presentation

- Final report PDF may be found here: https://thanasis.charisoudis.gr/pdf/gans-thesis-report.pdf
- Presentation PDF (handout) may be found here: https://thanasis.charisoudis.gr/pdf/gans-thesis-presentation.pdf
- Corresponding publication: _not-yet finalized_

## Regarding the code

### How to run

Open any of the provided notebooks (path: <code>notebooks/\*/\*.ipynb</code>) on the corresponding platform. You may
contact the author (achariso@ece.auth.gr) for GDrive and gh keys in order for the notebooks to be plug-n-play.

Alternatively, you may run the <code>.py</code> files inside <code>src</code> directory, using the extensive comments as
guidance.

## Code Stats

| Total Lines (.py) | Source Code Lines (%) | Comment Lines (%) | Blank Lines (%) | Notebook Lines (.ipynb) | 
| :---------------: | :-------------------: | :-------------: | :-------------: | :---------------------: |
| 15612 | 8403 (54%) | 5529 (35%) | 1680 (11%) | 6945 |

## Future Extensions - TODOs

- [x] Add ability to attach personal Google Drive for experiments continuation
- [x] Add Perceptual Path Length (PPL) as a regularizer/loss/metric in StyleGAN's Generator
- [x] Re-implement StyleGAN to mimic the exact architectures presented in Karras et al.
- [x] Fix Progressing Growing bugs (that lead to major visual artifacts)
- [x] Re-train StyleGAN for at least 320 epochs on DeepFashion's Image Synthesis Benchmark dataset
- [ ] Train an Inception model on fashion image dataset(s) and re-evaluate generative metrics based on embeddings of
  that classifier (instead of the one used now which was trained on ImageNET)
