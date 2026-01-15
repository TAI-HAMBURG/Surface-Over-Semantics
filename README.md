![Logo SOS](/img/sos_header.png)


<h1 align="center">
<span>SoS: Analysis of Surface over Semantics in Multilingual Text-To-Image Generation</span>
</h1>

[![ACL Anthology](https://img.shields.io/badge/ACL-Anthology-blue)](https://xxx)

## Paper Abstract

Text-to-image (T2I) models are increasingly employed by users worldwide. However, prior research has pointed to the high sensitivity of T2I towards particular input languages -- when faced with languages other than English (i.e., different surface forms of the same prompt), T2I models often produce culturally stereotypical depictions, prioritizing the surface over the prompt's semantics. Yet a comprehensive analysis of this behavior, which we dub Surface-over-Semantics (SoS), is missing. We present the first analysis of T2I models' SoS tendencies. To this end, we create a set of prompts covering 171 cultural identities, translated into 14 languages, and use it to prompt seven T2I models. To quantify SoS tendencies across models, languages, and cultures, we introduce a novel evaluation measure and analyze how the tendencies we identify manifest visually. We show that all tested models exhibit strong surface tendencies in at least three languages, and that this effect intensifies throughout the layers of T2Is' text encoders. Furthermore, strong surface tendencies often directly relate to stereotypical depictions and are reflected in distinct color profiles.

------------------------
## Getting Started

We conducted all our experiments with Python 3.10. Before getting started, make sure you install the requirements listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 📂 Directory/File Structure Overview

This repository contains all the code and data needed to reproduce the experiments and results reported in our paper.

### Data 

A brief description of the files in *data* is:

- **full_dataset.csv**
    - Contains all translated prompts.

- **human_annotations/annotator_x.csv**
    - Contains the annotations of each annotator on the validation set of the SoS scores



### Code

Includes all python files and notebooks subject to this paper.

A brief description of the files in *code* is:

- **creation_of_paper_plots.ipynb**
    - This notebook can be used to recreate all plots present in the paper, based on the experimental results.


------------------------
## References

Please use the following bibtex entry to cite us (TBD):
 
```bib
@inproceedings{}
```


---
*Author contact information: carolin.holtermann@uni-hamburg.de*


## License

All source code is made available under a 
