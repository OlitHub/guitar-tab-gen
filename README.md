# Conditional Generation of Bass Guitar Tablature for Guitar Accompaniment in Western Popular Music


## Abstract

*The field of symbolic music generation has seen great advancements with the rise of transformer-based architectures.
Addressing a specific need identified through a user study, we focus on developing AI tools to generate bass guitar tablatures conditioned on scores of other instruments in Western popular music.
The bass guitar, a vital component of the rhythmic and harmonic sections in music, presents a unique challenge due to its dual role in providing structure and groove.
Building upon the tablature notation, which simplifies the interpretation of music for stringed instruments, this work adopts modern encoding schemes to integrate tablature representation into transformer-based models.
To this end, the project involves preprocessing a large dataset of music scores, and fine-tuning state-of-the-art transformer architectures.
The generated tablatures will then be evaluated both numerically and qualitatively, with feedback from musicians.*

## Repository Structure

**models/.** Contains the transformer-based models which we took inspiration from and modified for our task. The models and code in this folder are untouched and all credit goes to the authors of the respective repositories.

-------------------------------------

**reports-presentations/.** Contains the final report, mid-term report, final presentation slides, all the figures, tables, audios and videos used and bibliography.    

-------------------------------------

**src/.** Root folder for all the code. Necessary notebooks are numbered in the order they should be run. Contains the following subfolders:
- **1. preprocessing:** Contains the code to generate the dataset from DadaGP dataset and the code to process the dataset to generate the input and target sequences for the model.
- **2. training:** Adapted from Makris et al. model_training script, training can be performed using notebook 1.model_training.ipynb or the script model_training_script.py.
- **3. inference:** Adapted from Makris et al. gen_drums script, inference is performed using notebook 1.gen_bass_tokens.ipynb. The code imports checkpoints generated during the training phase and outputs generated tokens in tokens_out folder. Also contains the code to convert the generated tokens to Guitar Pro 5 files. Conversion is done in notebook 2.gen_gp5_from_tokens.ipynb using an algorithm built by Sarmento et al. The code imports tokens from tokens_out folder and outputs Guitar Pro 5 files in generated_gp5 folder.

-------------------------------------

**diary.txt** Contains the daily progress of the project.

**requirements_dadagp.txt** Contains the necessary libraries to run the data processing notebooks.

**requirements_cp.txt** Contains the necessary libraries to run the training and inference scripts.


