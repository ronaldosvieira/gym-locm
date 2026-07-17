import numpy as np
import pickle


def model_import(weights_path, weights_name):
    with open(weights_path, "rb") as f:
        weights_value = pickle.load(f)
    weights = {name: value for name, value in zip(weights_name, weights_value)}
    return weights


def relu(x):
    return x * (x > 0)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def gelu(x):
    cdf = 0.5 * (1.0 + tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.pow(x, 3)))))
    return x * cdf


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def identity(x):
    return x


def _activation(activation_name):
    if activation_name == "relu":
        return relu
    elif activation_name == "tanh":
        return tanh
    elif activation_name == "gelu":
        return gelu
    else:
        return identity


class fully_connected_layers:
    def __init__(self, weights, biases, activation="relu") -> None:
        self.weights = weights
        self.biases = biases
        self.activation = _activation(activation)

    def __call__(self, input):
        output = np.matmul(input, self.weights) + self.biases
        return self.activation(output) if self.activation else output


class LSTM_layers:
    def __init__(
        self, kernel_weight, bias, projection_weight, forget_bias=1.0, activation="tanh"
    ) -> None:
        self.kernel_weight = kernel_weight
        self.bias = bias
        self.projection_weight = projection_weight
        self.forget_bias = forget_bias
        self.num_nuits = self.kernel_weight.shape[1] // 4
        self.num_proj = self.projection_weight.shape[1]
        self.activation = _activation(activation)

    def __call__(self, inputs, state=None):
        inputs = inputs.reshape(1, -1)
        if state is None:
            c_prev, m_prev = np.zeros([inputs.shape[0], self.num_nuits]), np.zeros(
                [inputs.shape[0], self.num_proj]
            )
        else:
            c_prev, m_prev = state[:, : self.num_nuits], state[:, self.num_nuits :]
        lstm_matrix = (
            np.matmul(np.concatenate([inputs, m_prev], axis=1), self.kernel_weight)
            + self.bias
        )
        i, j, f, o = np.split(lstm_matrix, 4, axis=1)
        c = sigmoid(f + self.forget_bias) * c_prev + sigmoid(i) * self.activation(j)
        m = sigmoid(o) * self.activation(c)
        if self.projection_weight is not None:
            m = np.matmul(m, self.projection_weight)
        return m, np.concatenate([c, m], 1)


class Conv1d:
    def __init__(
        self, kernel_weight, bias, strides=1, padding="same", activation="relu"
    ) -> None:
        self.kernel_weight = kernel_weight
        self.kernel_size, self.n_features, self.out_channels = kernel_weight.shape
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = _activation(activation)

    def __call__(self, inputs):
        assert len(inputs.shape) == 3  # batch_size = 1
        B = inputs.shape[0]
        inputs_len = inputs.shape[1]
        if self.kernel_size != 1:
            if self.padding == "same":
                zeros = np.zeros((B, self.kernel_size - 1, self.n_features))
                ext_inputs = np.concatenate((zeros, inputs, zeros), axis=1)
                w = inputs.shape[1]
            else:
                ext_inputs = inputs
                w = (inputs_len - self.kernel_size) // self.strides + 1
            output = np.zeros((B, w, self.out_channels))
            for j in range(self.out_channels):
                for i in range(w):
                    output[:, i, j] = self.activation(
                        np.sum(
                            ext_inputs[
                                :,
                                i * self.strides : i * self.strides + self.kernel_size,
                                :,
                            ]
                            * self.kernel_weight[:, :, j]
                            + self.bias[j]
                        )
                    )
        else:
            output = self.activation(
                np.matmul(inputs, self.kernel_weight.squeeze(0)) + self.bias
            )
        return output


class discrete_action_head:
    def __init__(self, logits_weight, logits_bias) -> None:
        self.logits_weight = logits_weight
        self.logits_bias = logits_bias

    def __call__(self, inputs, action_mask):
        head_logits = np.matmul(inputs.squeeze(), self.logits_weight) + self.logits_bias
        neginf = np.zeros_like(head_logits) - np.inf
        head_logits = np.where(action_mask, head_logits, neginf)
        return head_logits


class cb_discrete_action_head:
    def __init__(self, logits_weight, logits_bias) -> None:
        self.conv1d = Conv1d(logits_weight, logits_bias, activation=None)

    def __call__(self, inputs, action_mask, inputs_cards_can_select_mask):
        head_logits = self.conv1d(inputs)
        head_logits = np.squeeze(head_logits)
        head_logits = np.concatenate(
            [np.zeros(shape=(1,), dtype=np.float32), head_logits], axis=-1
        )
        neginf = np.zeros_like(head_logits) - np.inf
        head_logits = np.where(action_mask, head_logits, neginf)
        return head_logits
