import numpy as np
print('normalizer running..........')

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

    def backward(self,dvalues):

        self.dgamma = np.sum(dvalues * self.x_hat,axis=0,keepdims=True)
        
        self.dbeta = np.sum(dvalues,axis=0,keepdims=True)

        self.dinputs = (self.gamma/np.sqrt(self.variance + self.eps)*(dvalues - np.mean(dvalues,axis=-1,keepdims=True)-self.x_hat*np.mean(dvalues * self.x_hat,axis=-1,keepdims=True)))

        return self.dinputs