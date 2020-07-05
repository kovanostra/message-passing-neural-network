from message_passing_nn.graph import rnn_encoder, gru_encoder

models = {
    "RNN": rnn_encoder.RNNEncoder,
    "GRU": gru_encoder.GRUEncoder,
}
