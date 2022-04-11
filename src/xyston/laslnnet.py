import tensorflow as tf

from keras.layers import Conv
from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers


class Conv4D(Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1, 1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(Conv4D, self).__init__(
            rank=4,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs,
        )

    def convolution_op(self, inputs, kernel):
        b, c_i, l_i, d_i, h_i, w_i = tuple(inputs.get_shape().as_list())
        l_k, d_k, h_k, w_k = kernel
        if self.padding == "VALID":
            l_o = l_i - l_k + 1
        else:
            l_o = l_i

        res = l_o * [None]
        for i in range(l_k):
            for j in range(l_i):
                n = j - (i - l_k // 2) - (l_i - l_o) // 2
                if n < 0 or n >= l_o:
                    continue
                c = tf.nn.conv3d(
                    inputs,
                    kernel,
                    strides=list(self.strides),
                    padding=self.padding.upper(),
                    dilations=list(self.dilation_rate),
                    data_format=self._tf_data_format,
                    name=self.__class__.__name__ + f"_3dchan{i}",
                )
                if res[n] is None:
                    res[n] = c
                else:
                    res[n] += c
        output = tf.stack(res, axis=2)
        if self.activation:
            self.activation(output)
        return output
