import torch as to

optimizers = {
    "Adadelta": to.optim.Adadelta,
    "Adagrad": to.optim.Adagrad,
    "Adam": to.optim.Adam,
    "AdamW": to.optim.AdamW,
    "SparseAdam": to.optim.SparseAdam,
    "Adamax": to.optim.Adamax,
    "ASGD": to.optim.ASGD,
    "LBFGS": to.optim.LBFGS,
    "RMSprop": to.optim.RMSprop,
    "Rprop": to.optim.Rprop,
    "SGD": to.optim.SGD,
}
