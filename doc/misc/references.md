---
tocdepth: '2'
---

(references)=

# References

In the following, we provide an overview of scientific publications that are concerned with the BOOMER algorithm. In particular, this includes papers where different aspects of the algorithm's methodology have originally been proposed. They are the best source of information for those who are interested in how BOOMER works from a conceptual and mathematical point of view. In addition, we also refer to studies that have made use of the algorithm.

(references-first-party)=

## Our Publications

If you use the BOOMER algorithm in a scientific publication, we would appreciate citations to one of the following papers that discuss different aspects of its underlying ideas and concepts.

### Learning Gradient Boosted Multi-label Classification Rules

The algorithm was first published in the following [paper](https://doi.org/10.1007/978-3-030-67664-3_8). A preprint version is publicly available on [arxiv.org](https://arxiv.org/pdf/2006.13346.pdf).

*Michael Rapp, Eneldo Loza Mencía, Johannes Fürnkranz, Vu-Linh Nguyen and Eyke Hüllermeier. Learning Gradient Boosted Multi-label Classification Rules. In: Proceedings of the European Conference on Machine Learning and Knowledge Discovery (ECML-PKDD), pages 124-140, 2020, Springer.*

```bibtex
@inproceedings{rapp2020boomer,
    title={Learning Gradient Boosted Multi-label Classification Rules},
    author={Rapp, Michael and Loza Menc{\'i}a, Eneldo and F{\"u}rnkranz, Johannes and Nguyen, Vu-Linh and H{\"u}llermeier, Eyke},
    booktitle={Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
    pages={124--140},
    year={2020},
    publisher={Springer}
}
```

### Gradient-based Label Binning in Multi-label Classification

Gradient-based label binning (GBLB), which is an extension to the original algorithm, was proposed in the following [paper](https://doi.org/10.1007/978-3-030-86523-8_28). A preprint version is available on [arxiv.org](https://arxiv.org/pdf/2106.11690.pdf).

*Michael Rapp, Eneldo Loza Mencía, Johannes Fürnkranz and Eyke Hüllermeier. Gradient-based Label Binning in Multi-label Classification. In: Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), pages 462-477, 2021, Springer.*

```bibtex
@inproceedings{rapp2021gblb,
    title={Gradient-based Label Binning in Multi-label Classification},
    author={Rapp, Michael and Loza Menc{\'i}a, Eneldo and F{\"u}rnkranz, Johannes and H{\"u}llermeier, Eyke},
    booktitle={Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
    pages={462--477},
    year={2021},
    publisher={Springer}
}
```

### BOOMER – An Algorithm for Learning Gradient Boosted Multi-label Classification Rules

A technical report on the implementation of the BOOMER algorithm was published for open access in the following [article](https://doi.org/10.1016/j.simpa.2021.100137).

*Michael Rapp. BOOMER – An Algorithm for Learning Gradient Boosted Multi-label Classification Rules. In: Software Impacts (10), page 100137, 2021, Elsevier.*

```bibtex
@article{rapp2021boomer,
    title={{BOOMER} -- An Algorithm for Learning Gradient Boosted Multi-label Classification Rules},
    author={Rapp, Michael},
    journal={Software Impacts},
    volume={10},
    pages={100137},
    year={2021},
    publisher={Elsevier}
}
```

### Multi-label Rule Learning

An extensive discussion of the BOOMER algorithm, including a presentation of the conceptual framework it is based on, as well as a detailed explanation of several implementation details, can be found in the following [Ph.D. thesis](https://tuprints.ulb.tu-darmstadt.de/id/eprint/22099).

*Michael Rapp. Multi-label Rule Learning, 2022, Technische Universität Darmstadt.*

```bibtex
@phdthesis{rapp2022phdthesis,
    title={Multi-label Rule Learning},
    author={Rapp, Michael},
    year={2022},
    school={Technische Universit{\"a}t Darmstadt}
}
```

### On the Efficient Implementation of Classification Rule Learning

A detailed discussion of algorithmic concepts and approximation techniques used by the BOOMER algorithm can be found in the following [paper](https://doi.org/10.1007/s11634-023-00553-7).

*Michael Rapp, Johannes Fürnkranz and Eyke Hüllermeier. On the efficient implementation of classification rule learning. In: Advances in Data Analysis and Classification, 2023, pages 1-42, Springer.*

```bibtex
@article{rapp2023,
    title={On the efficient implementation of classification rule learning},
    author={Rapp, Michael and F{\"u}rnkranz, Johannes and H{\"u}llermeier, Eyke},
    journal={Advances in Data Analysis and Classification},
    pages={1--42},
    year={2023},
    publisher={Springer}
```

(references-third-party)=

## Citations of BOOMER

In the following, we provide a selection of interesting publications that have made use of the BOOMER algorithm in experimental studies or by building upon its code for the implementation of novel machine learning approaches.

```{note}
If you are the author of a paper that you would like to be presented in this section, feel free to reach out to us via the project's [issue tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues).
```

### pRSL: Interpretable Multi-label Stacking by Learning Probabilistic Rules

The BOOMER algorithm was used as a baseline in the experimental study that is included in the following paper, where a probabilistic rule stacking learner, which uses probabilistic propositional logic rules and belief propagation to combine the predictions of several classifiers, is proposed. A preprint of the paper is available at [arxiv.org](https://arxiv.org/pdf/2105.13850.pdf).

*Michael Kirchhof, Lena Schmid, Christopher Reining, Michael ten Hompel and Markus Pauly. pRSL: Interpretable Multi-label Stacking by Learning Probabilistic Rules. In: Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI), pages 461-470, 2021, PMLR.*

### Correlation-based Discovery of Disease Patterns for Syndromic Surveillance

In the following [paper](https://www.frontiersin.org/article/10.3389/fdata.2021.784159), a novel rule learning approach for discovering syndrome definitions for the early detection of infectious diseases is presented. The implementation of the proposed method, which is available at [GitHub](https://github.com/mrapp-ke/SyndromeLearner), is based on this project's source code. A preprint of the paper is available at [arxiv.org](https://arxiv.org/pdf/2110.09208.pdf).

*Michael Rapp, Moritz Kulessa, Eneldo Loza Mencía and Johannes Fürnkranz. Correlation-based Discovery of Disease Patterns for Syndromic Surveillance. In: Frontiers in Big Data (4), 2021, Frontiers Media SA.*

### A Flexible Class of Dependence-aware Multi-label Loss Functions

In the following [paper](https://link.springer.com/article/10.1007/s10994-021-06107-2), the predictive performance of the BOOMER algorithm in terms of a family of novel multi-label evaluation measures is compared experimentally to several competitors. A preprint of the paper is available at [arxiv.org](https://arxiv.org/pdf/2011.00792.pdf).

*Eyke Hüllermeier, Marcel Wever, Eneldo Loza Mencía, Johannes Fürnkranz and Michael Rapp. A Flexible Class of Dependence-aware Multi-label Loss Functions. In: Machine Learning (111), pages 713-737, 2022, Springer.*
