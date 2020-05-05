from message_passing_nn.model import graph_rnn_encoder, graph_gru_encoder

models = {
    "RNN": graph_rnn_encoder.GraphRNNEncoder,
    "GRU": graph_gru_encoder.GraphGRUEncoder,
}
