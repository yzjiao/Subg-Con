# Subg-Con
Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning (Jiao *et al.*, ICDM 2020): [https://arxiv.org/abs/2009.10273](https://arxiv.org/abs/2009.10273)


## Overview
Here we provide an implementation of Subg-Con in PyTorch and Torch Geometric. The repository is organised as follows:
- `subgcon.py` is the implementation of the Subg-Con pipeline;
- `subgraph.py` is the implementation of subgraph extractor;
- `model.py` is the implementation of components for Subg-Con, including a GNN layer, a pooling layer, and a scoring function;
- `utils_mp.py` is the necessary processing subroutines;
- `dataset/` will contain the automatically downloaded datasets;
- `subgraph/` will contain the processed subgraphs.

Finally, `train.py` puts all of the above together and may be used to execute a full training. 


## Dependencies
- Python 3.7.3
- [PyTorch](https://github.com/pytorch/pytorch) 1.5.1
- [torch_geometric](https://github.com/rusty1s/pytorch_geometric) 1.4.3
- scikit-learn 0.23.2
- scipy 1.5.2
- cytoolz 0.10.0


## Reference
If you make advantage of Subg-Con in your research, please cite the following in your manuscript:

```
@article{jiao2020sub,
  title={Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning},
  author={Jiao, Yizhu and Xiong, Yun and Zhang, Jiawei and Zhang, Yao and Zhang, Tianqi and Zhu, Yangyong},
  journal={arXiv preprint arXiv:2009.10273},
  year={2020}
}
```


