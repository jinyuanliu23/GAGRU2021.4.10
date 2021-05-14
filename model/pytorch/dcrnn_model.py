import numpy as np
import torch
import torch.nn as nn
# str.encode('utf-8')
# bytes.decode('utf-8')
import logging
from model.pytorch.dcrnn_cell import GAGRUCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        # self.multi_head_nums = int(model_kwargs.get('multi_head_nums',2))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.num_rnn_encode_layers = int(model_kwargs.get('num_rnn_encode_layers', 1))
        self.num_rnn_decode_layers = int(model_kwargs.get('num_rnn_decode_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = (self.num_nodes , self.rnn_units)


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.linearlayerencoder = nn.Linear(self.rnn_units,self.rnn_units)
        self.lstm = nn.LSTM(self.rnn_units, (self.num_rnn_encode_layers * self.rnn_units) // 2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((self.num_rnn_encode_layers * self.rnn_units) // 2), 1)
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.gagru_layers = nn.ModuleList(
            [GAGRUCell(self.rnn_units, adj_mx, self.num_nodes, self.input_dim
                       )] +
            [GAGRUCell(self.rnn_units, adj_mx, self.num_nodes,self.rnn_units
     ) for _ in range(self.num_rnn_encode_layers - 1)])



    def forward(self, inputs, hidden_state = None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, *_ = inputs.size()
        # print('model-54',inputs.size())
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_encode_layers, batch_size, *self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        # print('output.size',output.shape)
        # return None
        for layer_num, gagru_layer in enumerate(self.gagru_layers):
            # print('layer_num',layer_num)
            # print('output.size',output.shape)
            # logging.warning('enlayer_num {}'.format(layer_num))
            # logging.warning('enoutput size {}'.format(output.shape))
            next_hidden_state = gagru_layer(output , hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
            # output = self.linearlayerencoder(output)
            # print(next_hidden_state.shape())
        # next_hidden_states = torch.cat(output, dim=0 )
        # alpha ,_ =self.lstm(next_hidden_states)
        # alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        # alpha = torch.softmax(alpha, dim=-1)
        #
        # output = (next_hidden_states* alpha.unsqueeze(-1)).sum(dim=1)
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        # self.linearlayerdecoder = nn.Linear(self.rnn_units, self.output_dim)
        # self.gagru_layers = nn.ModuleList(
        #     [GAGRUCell(self.rnn_units, adj_mx,  self.num_nodes , self.rnn_units) for _ in range(self.num_rnn_decode_layers)])
        self.gagru_layers = nn.ModuleList(
            [GAGRUCell(self.rnn_units, adj_mx, self.num_nodes, self.output_dim
                       )] +
            [GAGRUCell(self.rnn_units, adj_mx, self.num_nodes, self.rnn_units
                       ) for _ in range(self.num_rnn_decode_layers - 1)])
        # self.gagru_layers = nn.ModuleList([GAGRUCell(self.rnn_units, adj_mx, self.num_nodes, self.rnn_units
        #                ) for _ in range(self.num_rnn_decode_layers - 1)] +
        #     [GAGRUCell(self.rnn_units, adj_mx, self.num_nodes, self.output_dim
        #                )])
    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        # logging.warning('output size {}'.format(output.shape))
        for layer_num, gagru_layer in enumerate(self.gagru_layers):
            # logging.warning('delayer_num {}'.format(layer_num))
            # logging.warning('deoutput size {}'.format(output.shape))
            next_hidden_state = gagru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # logging.warning('llllllllllllllllll output size {}'.format(output.shape))

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes ,self.output_dim)

        return output, torch.stack(hidden_states)


class GARNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        # self.linearlayeren = nn.Linear(self.rnn_units,self.rnn_units)
        # self.linearlayerde = nn.Linear(self.rnn_units, self.rnn_units)
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        # logging.warning('encoder-input{}'.format(inputs.shape))
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        # logging.warning('encoder-input{}'.format(encoder_hidden_state.shape))
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes ,self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)

        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        # encoder_hidden_state = self.linearlayeren(encoder_hidden_state)
        # print('encoder_hidden_state',encoder_hidden_state.shape)

        # loss = encoder_hidden_state.sum()
        # loss.backward()
        # assert False
        self._logger.debug("Encoder complete, starting decoder")
        # logging.warning('Encoder complete, starting decoder {}'.format(inputs.shape))

        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # logging.warning('decode output size {}'.format(outputs.shape))
        self._logger.debug("Decoder complete")
        # logging.warning('Decoder complete{}'.format(outputs.shape))

        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs

# if __name__ == "__main__":
#  # test here
#      batch_size = 64
#      n_vertex =4
#      n_step = 12
#      n_output = 12
#      n_channel =2
#      test_input = torch.randn(batch_size, n_vertex, n_step, n_channel)
#      test_adj = torch.randn(n_vertex, n_vertex)
#      test_network = DCRNNModel(n_vertex, n_channel)
#      test_output = test_network( test_adj ,test_input)




class JumpingKnowledge(torch.nn.Module):

    def __init__(self, mode, channels, num_layers):
        super(JumpingKnowledge, self).__init__()


        if mode == 'lstm':
            self.lstm = LSTM(
                channels, (num_layers * channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):

        x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (x * alpha.unsqueeze(-1)).sum(dim=1)

