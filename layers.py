import numpy as np
from math import erf
class Add_layer:

    def __init__(self,inputs,neurons):
        self.weights = np.random.randn(inputs,neurons) * np.sqrt(2/(inputs))
        self.bias = np.zeros((1,neurons))

    def forward(self,inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs,self.weights) + self.bias

    def backward(self,dvalues):

        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbias = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class Activation_ReLu:

    def forward(self,inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0]=0

class Activation_Leaky_ReLu:
    def __init__(self,negative_slope=0.01):
        self.negative_slope = negative_slope
    def forward(self,inputs):
        self.inputs = inputs
        self.outputs = (np.maximum(0,inputs) + self.negative_slope * np.minimum(0,inputs))

    def backward(self,dvalues):

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0] *= self.negative_slope

class Activation_Softmax:

    def forward(self,inputs):
        exp_value = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_value / np.sum(exp_value,axis=1,keepdims=True)
        self.outputs = probabilities

    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index,(single_output,single_dvalues) in enumerate(zip(self.outputs,dvalues)):

            single_output = single_output.reshape(-1,1)
            jacobian_matrix = (np.diagflat(single_output) - np.dot(single_output,single_output.T))
            self.dinputs[index] = (jacobian_matrix @ single_dvalues)


class Activation_GeLu:

    def forward(self,inputs):
        self.inputs = inputs

        phi_cdf = 0.5 * (1+np.vectorize(erf)(self.inputs/np.sqrt(2)))
        self.outputs = self.inputs * phi_cdf

    def backward(self,dvalues):

        phi_cdf = 0.5 * (1+np.vectorize(erf)(self.inputs/np.sqrt(2)))
        phi_pdf = (1 / np.sqrt(2*np.pi))*np.exp(-(self.inputs**2)/2)

        gelu_derivative = phi_cdf + self.inputs * phi_pdf

        self.dinputs = dvalues * gelu_derivative


class Loss:
    def calculate(self,output,y):

        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CCE(Loss):

    def forward(self,y_pred,y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape)==1:
            correct_confidence = y_pred_clipped[range(sample),y_true]
        elif len(y_true.shape)==2:
            correct_confidence = np.sum(y_pred_clipped * y_true,axis=1)

        self.negative_log_likelihood = -np.log(correct_confidence)
        return self.negative_log_likelihood

    def backward(self,dvalues,y_true):

        sample = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape)==1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/sample


class Softmax_CCE:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = CCE()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs

        return self.loss.calculate(self.outputs,y_true)

    def backward(self,dvalues,y_true):
        samples = len(dvalues)

        if len(y_true.shape)==2:
            y_true = np.argmax(y_true,axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples),y_true]-=1

        self.dinputs /= samples    





class Optimizer_SGD:

    def __init__(self,lr=0.01,decay=0.0):
        self.learning_rate = lr
        self.curr_lr = self.learning_rate
        self.decay = decay
        self.iter = 0
    def pre_update_params(self):
        if self.decay:
            self.curr_lr = (self.learning_rate/(1+self.decay * self.iter))
    def update_params(self,layer):
        layer.weights -= (self.curr_lr * layer.dweights)
        layer.bias -= (self.curr_lr * layer.dbias)

    def post_update_params(self):
        self.iter +=1