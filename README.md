# Token Inflation Project Experiments

This repository contains the code used for the experiments in our NeurIPS submission on token-count inflation in LLM auditing frameworks.

## Overview

The code is organized by experiment. Each folder contains the scripts, configuration files, and helper utilities needed to reproduce the corresponding results reported in the paper.

## Repository Structure

Each experiment folder includes:

- data-processing scripts,
- experiment-running scripts,
- model or verifier utilities,
- plotting or result-generation scripts, where applicable,

## Reproducibility

To reproduce the results, enter the folder for the desired experiment and follow the instructions provided there. Some experiments require local model checkpoints, processed datasets, or GPU access. Paths to datasets and model checkpoints may need to be updated depending on the local environment.

## Notes

This repository is intended for research reproducibility and code review. The code has been anonymized and documented for submission.


## CoIn Framework

In order to run the experiments for the CoIn auditing framework, you need to first collect the library and class files from the official github repo for the CoIn framework, which can be found in: https://github.com/CASE-Lab-UMD/LLM-Auditing-CoIn.

We specially need the following files from the code repo:
heads.py from the path "3_Tokens2Block/train"
You will also need to download the model weights, specially for the 256 token version of the CoIn framework.
The dataset can be found here: https://huggingface.co/datasets/glaiveai/reasoning-v1-20m


## PALACE Framework

Unfortunately, the code and model weights for the PALACE framework are not publicly available. Please contact the authors to get the code and model weights before evaluating the experiments inside the PALACE folder.

## Statistical Auditor

The statistical auditor model has been proposed by the "Auditing Pay-Per-Token in Large Language Models" paper and the github repo for the project can be found here: https://github.com/Human-Centric-Machine-Learning/token-audit

In our experiments, we primarily used the following dataset: https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data


