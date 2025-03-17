1. Introduction

* Definition of the need (user study)
* Role of the bass guitar in Western popular music
* 


Numéroter slides
Audio Harmonic and rhythmic
Quand citation footnote précise

Vidéo tablature/plus longue tablature

Simplifier pipeline pas directement parler de gen conditionnelle etc


Data processing

Présenter dadagp avant tokenization, caractéristiques format nombre etc

Avant slide rhythm guitar, montrer tablature guitare rhythm et guitare lead avec son.

Slide annexe distribution sequence length

Préciser citation,
préciser que c'est prompt qui controle la future génération
instead of dissonant, no possibility of adaptation: free generation

slide training
couleurs visibles avec handicap?

results
faire écouter au moins rhythm guitar et puis les bass (originale et générées)
avec vidéo (capture écran)



Découper slide 1 (d'abord bass guitar role puis user study)

Si manque de temps compound word en annexe

slide 9 simplifier
Slide 10 poser texte
slide 13 préciser checkpoint
slide bullet point conclusion






Slide 1: Presentation slide

Slide 2: Table of contents

Slide 3: Title slide

Slide 4: Role of bass guitar
- Typical rock band instrumentation: lead guitar, rhythm guitar, bass guitar, drums, vocals
- Bass guitar: bridge between rhythm and harmony
- Harmonic role, reinforcing the harmony at a lower pitch, giving the impression of a full sound
- Melodic role, doubling the lead guitar
- Rhythmic role, doubling the drum pattern

Slide 5: Our vision
- Some guitarist don't have enough bass guitar knowledge to write bass lines
- The tool we propose would help them compose bass lines adapted to their guitar parts

Slide 6: Title slide

Slide 7: DadaGP dataset
- More than 26 000 scores of Western popular music
- GP and tokens and algorithm to convert from one to the other

Slide 8: DadaGP tokenization
- Metadata tokens
- Succession of new measure tokens then tokens for each note of each instrument in the measure and wait tokens for the length

Slide 9: Token extraction
- Extraction of the bass tokens 
- If wait tokens become successive, they are summed

Slide 10: Rhythm guitar identification
- Drums: good rhythmic information but no harmonic information
- Lead guitar: good harmonic information but no rhythmic information
- Rhythm guitar: good rhythmic and harmonic information
- Probability between 0 and 1 for each bar and each guitar
- Close to 0: rhythm bar
- Rhythm guitar: guitar with the most rhythm bars
- Once we had this: token extraction on the rhythm guitar

Slide 11: Fitting data to the model
- Sequences of 16 measures with stride 8 (1 - 16, 9 - 24, 17 - 32, etc.)
- 'sos', 'eos', padding, conversion to indices

Slide 12: Title slide

Slide 13: Baseline model
- DadaGP Transformer generative model trained on the dataset
- Example prompt --> no adaptation, free generation
- Emergence of other instruments --> if logits of other instruments are set to 0 --> repetitive patterns

Slide 14: Our model
- Adapted version of Makris et al. model, they used it for conditional drums generation: different because no pitch.
- BiLSTM to encode the rhythm guitar, so that the embeddings of the rhythm guitar sequence take into account future and past information
- Transformer to generate the bass guitar sequence

Slide 15: Training
- We reach 0.14 sparse categorical crossentropy on the validation set

Slide 16: Title slide
- 3 generated examples from songs that were not in Dadagp

Slide 17: Ricard Peinard
- Typical working basic example
- Bass guitar harmonically and rhythmically close and adapted to the rhythm guitar

Slide 18: Do I wanna know?
- Correct bass guitar part
- Time signature issue that causes a shift in the bass guitar part, desynchronization with the rhythm guitar

Slide 19: Saturday night
- Good accompaniment bass guitar
- But far from what is usually expected in this style
--> Model is great for accompaniment parts, but not for more complex bass lines

Slide 20: Title slide

Slide 21: Perspectives
- Refinement in the model's layers. Tuning of the hyperparameters.
- Implementation of other tokenization: the initial tokenization used with this model was not the one we used
- Quantitative metrics and user study
- Fine-tuning of the model on specific styles of music
