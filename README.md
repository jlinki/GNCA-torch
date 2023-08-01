# GNCA-torch
PyTorch Implementation of "Learning Graph Cellular Automata" (Grattarola et al. 2021) using PyTorch Geometric.
Currently only supports fixed target experiments.

### Dependencies:
```
torch torch_geometric matplotlib numpy scipy pygsp 
```

### Usage:
Simply run e.g. for 2d grid graph (see argparse for further options):
```
python run_fixed_target.py --graphs Grid2d
```

![test](images/evolution.pdf)

### Known issues:
- Gradient clipping is not working properly with parameters from paper. This why it's disabled by default. 
- Open issues for any questions or bugs.