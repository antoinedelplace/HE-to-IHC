---
title: H&E-to-IHC Stain Translation
emoji: 🪄🧬🌈
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
---

# H&E-to-IHC Stain Translation
Gradio App based on Adaptive Supervised PatchNCE Loss for Learning H&E-to-IHC Stain Translation with Inconsistent Groundtruth Image Pairs (MICCAI 2023)

Online demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AntoineDelplace/HE-to-IHC)

Original folder: [lifangda01/AdaptiveSupervisedPatchNCE](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE)
Original paper: [![arXiv](https://img.shields.io/badge/arXiv-2303.06193-00ff00.svg)](https://arxiv.org/pdf/2303.06193)

## 🎯 Overview
This repository features a Gradio-based application built on the methods introduced in the MICCAI 2023 paper, "Adaptive Supervised PatchNCE Loss for Learning H&E-to-IHC Stain Translation with Inconsistent Groundtruth Image Pairs." The application facilitates automatic virtual staining, transforming H&E (Hematoxylin and Eosin) images into corresponding IHC (ImmunoHistoChemistry) images.

Users can generate virtual IHC stains for four key biomarkers critical to breast cancer diagnostics:
- HER2: Human Epidermal Growth Factor Receptor 2
- ER: Estrogen Receptor
- Ki67: Antigen KI-67 (cell proliferation marker)
- PR: Progesterone Receptor

This tool simplifies and accelerates the analysis of histopathological samples, making advanced diagnostic insights more accessible through virtual staining technology.