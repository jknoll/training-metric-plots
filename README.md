Script to parse a set of cluster training files (`rank_0.txt`) from Strong Compute cluster training runs and generate a .csv and a set of matplotlib graphs of the training metrics.

How to use:
Look at your control plane "Experiments" view and determine which experiment ID(s) correspond to the training run(s) you wish to visualize.

Copy the `rank_0.txt` files into `rank-files` and run plot.py.

![Training Metrics](/img/training_metrics.png)