from attention import MultiHeadAttention
from normalization import LayerNorm
from feedforward import FeedForward
from embedding import encode,decode,Positional_encoding,embedding
import numpy as np
from math import erf
print('transformer running............')
import json
class TransformerBlock:

    def __init__(self,d_model=256,num_heads=4,d_ff=1024):
        self.attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.normalization1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model,d_ff=d_ff)
        self.normalization2 = LayerNorm(d_model)

    def forward(self,inputs):

        attention_outputs,attention_weights = self.attention.forward(inputs)
        x = inputs + attention_outputs
        x = self.normalization1.forward(x)

        self.ffn_input = x
        ffn_output = self.ffn.forward(x)
        x = ffn_output + x
        x = self.normalization2.forward(x)

        return x,attention_weights

    def backward(self, dvalues):

        self.normalization2.backward(dvalues)
        dx = self.normalization2.dinputs

        self.ffn.backward(dx)
        dx = self.ffn.dinputs + dx

        self.normalization1.backward(dx)
        dx = self.normalization1.dinputs

        self.attention.backward(dx)
        dx = self.attention.dinputs + dx

        self.dinputs = dx


class Transformer:

    def __init__(self,num_layers=4,d_model=256,num_heads=4,d_ff=1024):
        self.blocks=[]
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff))

    def forward(self,inputs):
        x = inputs
        attention_weights = []
        for block in self.blocks:

            x,weights = block.forward(x)
            attention_weights.append(weights)

        return x,attention_weights

    def backward(self, dvalues):

        dx = dvalues

        for block in reversed(self.blocks):
            block.backward(dx)
            dx = block.dinputs

        self.dinputs = dx



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


class LanguageModelHead:
    def __init__(self,d_model=256,vocab_size=1256):
        self.output_layer = Add_layer(d_model,vocab_size)

    def forward(self,inputs):
        self.output_layer.forward(inputs)

        return self.output_layer.outputs
    
    def backward(self, dvalues):
        self.output_layer.backward(dvalues)
        self.dinputs = self.output_layer.dinputs

class Activation_Softmax:

    def forward(self,inputs):
        exp_value = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_value / np.sum(exp_value,axis=1,keepdims=True)
        self.outputs = probabilities

        return self.outputs

    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index,(single_output,single_dvalues) in enumerate(zip(self.outputs,dvalues)):

            single_output = single_output.reshape(-1,1)
            jacobian_matrix = (np.diagflat(single_output) - np.dot(single_output,single_output.T))
            self.dinputs[index] = (jacobian_matrix @ single_dvalues)


with open("merges.json", "r") as f:
    data = json.load(f)

merges = {}

for pair, new_token in data.items():
    a, b = map(int, pair.split(","))
    merges[(a, b)] = new_token

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

token_ids = encode(text, merges)


if len(token_ids) < 2:
    raise ValueError("Enter at least 2 tokens for training.")


# ============================================================
# Training settings
# ============================================================

epochs = 10
learning_rate = 0.01
sequence_length = 128


# ============================================================
# Create model
# ============================================================

transformer = Transformer(num_layers=4,d_model=256,num_heads=4,d_ff=1024)

lm_head = LanguageModelHead(d_model=256,vocab_size=756)

softmax = Activation_Softmax()


# ============================================================
# Training
# ============================================================

for epoch in range(epochs):

    total_loss = 0.0
    steps = 0

    for start in range(0,len(token_ids) - sequence_length,sequence_length):

        # Input / target

        input_ids = token_ids[start:start + sequence_length]

        target_ids = token_ids[start + 1:start + sequence_length + 1]

        # Embedding

        x = embedding.forward(input_ids)

        # Positional encoding

        pe = Positional_encoding(len(input_ids),256)

        x = x + pe.sinusoidal_positional_encoding()

        # Transformer

        output, attention_weights = transformer.forward(x)

        # Language model head

        logits = lm_head.forward(output)

        # Softmax

        probabilities = softmax.forward(logits)

        # Cross entropy loss

        N = len(target_ids)

        correct_probabilities = probabilities[np.arange(N),target_ids]

        loss = -np.mean(np.log(correct_probabilities + 1e-9))

        total_loss += loss
        steps += 1

        # Backward

        dlogits = probabilities.copy()

        dlogits[np.arange(N),target_ids] -= 1

        dlogits /= N

        # Language model head
        lm_head.backward(dlogits)

        # Transformer
        transformer.backward(lm_head.dinputs)

        # Embedding
        embedding.backward(transformer.dinputs,input_ids)

        # Update embedding

        embedding.weight -= (learning_rate *embedding.dweights)

        # Update Transformer

        for block in transformer.blocks:

            # Attention
            block.attention.w_q -= (learning_rate *block.attention.dw_q)

            block.attention.w_k -= (learning_rate *block.attention.dw_k)

            block.attention.w_v -= (learning_rate *block.attention.dw_v)

            block.attention.w_o -= (learning_rate *block.attention.dw_o)

            # LayerNorm 1
            block.normalization1.gamma -= (learning_rate *block.normalization1.dgamma)

            block.normalization1.beta -= (learning_rate *block.normalization1.dbeta)

            # FFN layer 1
            block.ffn.layer1.weights -= (learning_rate *block.ffn.layer1.dweights)

            block.ffn.layer1.bias -= (learning_rate *block.ffn.layer1.dbias)

            # FFN layer 2
            block.ffn.layer2.weights -= (learning_rate *block.ffn.layer2.dweights)

            block.ffn.layer2.bias -= (learning_rate *block.ffn.layer2.dbias)

            # LayerNorm 2
            block.normalization2.gamma -= (learning_rate *block.normalization2.dgamma)

            block.normalization2.beta -= (learning_rate *block.normalization2.dbeta)

        # Update Language Model Head

        lm_head.output_layer.weights -= (learning_rate *lm_head.output_layer.dweights)

        lm_head.output_layer.bias -= (learning_rate *lm_head.output_layer.dbias)

    # Epoch loss

    average_loss = total_loss / steps

    print(f"Epoch {epoch + 1}/{epochs} "f"| Loss: {average_loss:.6f}")


# ============================================================
# Test after training
# ============================================================
print("\n===== TEST =====")

test_ids = token_ids[:16]

print("token_ids length:", len(token_ids))
print("test_ids length:", len(test_ids))

x = embedding.forward(test_ids)

print("embedding shape:", x.shape)

pe = Positional_encoding(len(test_ids), 256)
x = x + pe.sinusoidal_positional_encoding()

print("x shape:", x.shape)

output, attention_weights = transformer.forward(x)

print("transformer output shape:", output.shape)

logits = lm_head.forward(output)
probabilities = softmax.forward(logits)

predicted_ids = np.argmax(probabilities, axis=1)

predicted_text = decode(predicted_ids, merges)

print("Predicted:", predicted_text)








# transformer = Transformer(
#     num_layers=4,
#     d_model=256,
#     num_heads=4,
#     d_ff=1024
# )

# text = input("enter input: ")

# with open("merges.json", "r") as f:
#     data = json.load(f)

# merges = {}

# for pair, new_token in data.items():
#     a, b = map(int, pair.split(","))
#     merges[(a, b)] = new_token


# # Tokenize
# token_ids = encode(text, merges)

# # print("Token IDs:")
# # print(token_ids)


# # Embedding
# x = embedding.forward(token_ids)

# # print("Embedding shape:")
# # print(x.shape)


# # Positional encoding
# pe = Positional_encoding(len(token_ids), 256)

# pos_encoding = pe.sinusoidal_positional_encoding()

# x = x + pos_encoding

# # print("After positional encoding:")
# # print(x.shape)

# output, attention_weights = transformer.forward(x)

# print("Transformer output shape:", output.shape)
# print("Number of layers:", len(attention_weights))

# for i, weights in enumerate(attention_weights):
#     print(f"Layer {i+1} attention shape:", weights.shape)




# lm_head = LanguageModelHead(d_model=256,vocab_size=756)
# logits = lm_head.forward(output)

# print("Logits shape:", logits.shape)

# softmax = Activation_Softmax()
# softmax.forward(logits)
# probabilities = softmax.outputs

# print("Probabilities shape:", probabilities.shape)
# print("\nProbability sums:")

# print(np.sum(probabilities, axis=1))
# predicted_ids = np.argmax(probabilities, axis=1)

# print("\nPredicted token IDs:")
# print(predicted_ids)
# predicted_text = decode(predicted_ids, merges)

# print("\nPredicted text:")
# print(predicted_text)


















# # Transformer Block
# block = TransformerBlock(
#     d_model=256,
#     num_heads=4,
#     d_ff=1024
# )

# output, attention_weights = block.forward(x)


# print("\nTransformer block output:")
# print(output)

# print("Output shape:")
# print(output.shape)


# print("\nAttention weights shape:")
# print(attention_weights.shape)