from torch import nn

loss_functions = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "CTC": nn.CTCLoss,
    "NLL": nn.NLLLoss,
    "PoissonNLL": nn.PoissonNLLLoss,
    "KLDiv": nn.KLDivLoss,
    "BCE": nn.BCELoss,
    "BCEWithLogits": nn.BCEWithLogitsLoss,
    "MarginRanking": nn.MarginRankingLoss,
    "HingeEmbedding": nn.HingeEmbeddingLoss,
    "MultiLabelMargin": nn.MultiLabelMarginLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "SoftMargin": nn.SoftMarginLoss,
    "MultiLabelSoftMargin": nn.MultiLabelSoftMarginLoss,
    "CosineEmbedding": nn.CosineEmbeddingLoss,
    "MultiMargin": nn.MultiMarginLoss,
    "TripletMargin": nn.TripletMarginLoss
}
