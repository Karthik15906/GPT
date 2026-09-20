import numpy as np


class LayerNorm:

    def __init__(self, d_model=256, eps=1e-5):

        self.d_model = d_model
        self.eps = eps

        self.gamma = np.ones((1, d_model))
        self.beta = np.zeros((1, d_model))

    def forward(self, inputs):

        self.inputs = inputs

        self.mean = np.mean(inputs,axis=-1,keepdims=True)

        self.variance = np.var(inputs,axis=-1,keepdims=True)

        self.x_hat = ((inputs - self.mean)/ np.sqrt(self.variance + self.eps))

        self.outputs = self.gamma * self.x_hat + self.beta

        return self.outputs