(user-guide-optimizations)=

# Algorithmic Optimizations

```{seealso}
Most of the content in this chapter has been taken from the publication {ref}`‘On the efficient implementation of classification rule learning’, Michael Rapp, Johannes Fürnkranz and Eyke Hüllermeier (2023) <references-rapp2023>`.
```

Rule learning methods have a long history of active research in the machine learning community. They are not only a common choice in applications that demand human-interpretable classification models but have also been shown to achieve state-of-the-art performance when used in ensemble methods. Unfortunately, only little information can be found in the literature about the various implementation details that are crucial for the efficient induction of rule-based models. In this documentation, we provide a detailed discussion of algorithmic concepts and approximations that enable applying our rule learning algorithms to large amounts of data. The techniques discussed in this chapter are used by both, the {ref}`BOOMER algorithm <user-guide-boomer>` and the {ref}`SeCo algorithm <user-guide-seco>`, which is made possible by their common {ref}`framework <user-guide-framework>`.
