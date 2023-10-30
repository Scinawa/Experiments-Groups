# Experiments-Groups


- `dataset-random-graphs`: The k-reduced-skew-spectra of Erdős–Rényi graphs.
- `dataset-simple-graphs`: some simple graphs downloaded from the link below
- `dataset-all-graphs`: all directed and undirected graph for small (very small :) ) n

  
- `generate-random-dataset.py`: generate the datasets in  `dataset-random-graphs` directory and the dataset for `EXP-Atlas-undirected
- `generate_all_graphs`: folder with HPC code to generate in parallel(!) all directed graphs, eventually skipping some.
- `fromsparse.py:` convert g6 graphs into netnworkx->numpy and gerate k-reduced-skew-spectra


- `EXP-on-graphs.ipynb`: the experiments with all undirected graphs of n=7 nodes
- `EXP-discriminating-important-graphs`: our 3-reduced-skew-spectra can distringuish two important graphs :)
- `EXP-new-QM7-regression.ipynb`: Some examples on how the k-reduced-skew spectra improves the regression task on QM7-datasets


==== Useful links ====
- http://users.cecs.anu.edu.au/%7Ebdm/data/graphs.html
