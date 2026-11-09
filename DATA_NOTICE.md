# Data sources and stored artifacts

The MNIST files in `data/MNIST/raw/` are an external benchmark, not original research data created by this project.

[Keras's MNIST documentation](https://keras.io/api/datasets/mnist/) credits Yann LeCun and Corinna Cortes and identifies the dataset's [Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/). The [MNIST source page](https://yann.lecun.com/exdb/mnist/) describes the benchmark. Retain attribution and the applicable license when redistributing MNIST material.

The repository's `uncp/data/colored_mnist.py` loader uses `torchvision.datasets.MNIST`. Generated color variants remain derived benchmark material; this notice does not remove upstream obligations.

Stored checkpoints, pickled results, tables, and figures are experiment artifacts. Their presence does not establish independent validation, a completed reproduction, or rights to unrelated third-party data. Load serialized artifacts only from sources you trust.

This notice documents upstream dataset terms. It does not assign a code license, license the project's checkpoints, or grant rights to any other benchmark.
