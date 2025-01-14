# saliency_metrics

Metrics for comparing two saliency maps or two fixation maps. These metrics are implemented in Matlab in the original repository [here](https://github.com/cvzoya/saliency/tree/master/code_forMetrics). This is just a reimplementation of the metrics in Python.

The saliency benchmark (in Matlab) and this code produce the same measures for the same set of inputs.

Metrics implemented are:

1. AUC Judd
2. AUC Borji
3. AUC shuffled
4. NSS
5. Info Gain
6. SIM
7. CC
8. KL divergence

The two maps to be compared are `fixation_map` and `saliency_map`. Each method also discretizes (if needed) and normalizes the maps. Details about each of the methods can be found in the MIT Saliency Benchmark paper "What do different evaluation metrics tell us about saliency models?".

## Citations

```
@misc{mit-saliency-benchmark,
  author       = {Zoya Bylinskii and Tilke Judd and Fr{\'e}do Durand and Aude Oliva and Antonio Torralba},
  title        = {MIT Saliency Benchmark},
  howpublished = {http://saliency.mit.edu/}
}

@article{salMetrics_Bylinskii,
    title    = {What do different evaluation metrics tell us about saliency models?},
    author   = {Zoya Bylinskii and Tilke Judd and Aude Oliva and Antonio Torralba and Fr{\'e}do Durand},
    journal  = {arXiv preprint arXiv:1604.03605},
    year     = {2016}
}
```
