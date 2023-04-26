import numpy as np
import gym
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork

tf1, tf, tfv = try_import_tf()


class LSTMModel(RecurrentNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, ):
        super(LSTMModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        input_dim = int(np.product(self.obs_space.shape))
        sensor_cell_size = 256

        img_dim = (32, 32, 1)
        img_elem = np.product(img_dim)
        vision_num_filters = 4
        vision_output_size = 8
        vision_state_dim = (vision_output_size, vision_output_size, vision_num_filters)
        vision_cell_size = np.product(vision_state_dim)
        self.cell_size_all = sensor_cell_size + vision_cell_size

        inputs = tf.keras.layers.Input(shape=(None, input_dim), name="inputs")

        all_state_in_h = tf.keras.layers.Input(shape=(self.cell_size_all,), name="h")
        all_state_in_c = tf.keras.layers.Input(shape=(self.cell_size_all,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        vision_input = inputs[:, :, :img_elem]
        vision_input = tf.keras.layers.Reshape((-1,) + img_dim)(vision_input)
        vision_state_in_h = all_state_in_h[:, :vision_cell_size]
        vision_state_in_c = all_state_in_c[:, :vision_cell_size]
        vision_state_in_h = tf.reshape(vision_state_in_h, (-1,) + vision_state_dim)
        vision_state_in_c = tf.reshape(vision_state_in_c, (-1,) + vision_state_dim)

        state_input = inputs[:, :, img_elem:]
        cell_state_in_h = all_state_in_h[:, vision_cell_size:]
        cell_state_in_c = all_state_in_c[:, vision_cell_size:]

        vision_lstm_out, vision_state_h, vision_state_c = tf.keras.layers.ConvLSTM2D(
            vision_num_filters, (4, 4), strides=(4, 4), return_sequences=True, return_state=True)(
            inputs=vision_input,
            mask=tf.sequence_mask(seq_in),
            initial_state=[vision_state_in_h, vision_state_in_c],
        )
        vision_lstm_out = tf.keras.layers.Reshape((-1, vision_cell_size))(vision_lstm_out)
        vision_state_h = tf.keras.layers.Flatten()(vision_state_h)
        vision_state_c = tf.keras.layers.Flatten()(vision_state_c)

        sensor_lstm_out, sensor_state_h, sensor_state_c = tf.keras.layers.LSTM(
            sensor_cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=state_input,
            mask=tf.sequence_mask(seq_in),
            initial_state=[cell_state_in_h, cell_state_in_c],
        )

        features = tf.keras.layers.Concatenate()([vision_lstm_out, sensor_lstm_out])

        state_out_h = tf.concat([vision_state_h, sensor_state_h], axis=-1)
        state_out_c = tf.concat([vision_state_c, sensor_state_c], axis=-1)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(features)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(features)

        self._value_out = None

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[inputs, seq_in, all_state_in_h, all_state_in_c],
            outputs=[logits, values, state_out_h, state_out_c])
        # self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size_all, np.float32),
            np.zeros(self.cell_size_all, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
