# Mini Language Models

A significant portion of the code is from the tutorial [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2064s&ab_channel=AndrejKarpathy) by Andrej Karpathy. This code also is in PyTorch instead of TensorFlow unlike some of my other projects.

## How to Run

Just run the notebook corresponding to the model you want to run. The data should be automatically downloaded and extracted. You will need PyTorch and other common ML libraries already installed.

## Models

### Bigram Language Model ([bigram.ipynb](/bigram.ipynb))

This is a model that predicts the next token based on the previous token. It uses an embedding with the vocab size as the embedding dim, making it a square. To generate the next token, it gets the embedding and calls `softmax` on it, essentially using the embedding as the logits. This works because it each character is a token, so with only 65 tokens there is $65^2$ parameters. This is a very simple model, but it is a good baseline for more complex models. Also, the performance is not that bad for the parameter count. Currently, the model is trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, but it can be trained on any text file.

#### Sample Output

Sample output with input `"LUCENT"`, temperature $1.0$

```
LUCENTER: und howiste ty dyotrd,
Theal lerno, y va f m my mulde ben s, r bet!
AMAs sod ke alved.
Thup sthe
```

### N-gram Language Model ([ngram.ipynb](/ngram.ipynb))

This is my own implementation based on the bigram model, but configurable to use n previous tokens. The model has a second hyperparameter $n \ge 2$ which is the number of tokens the model will look at. It uses a token embedding and a positional embedding added together for each of the $n$ tokens. Then, it uses a weight for each of the $n$ embeddings and adds them together. Finally, it is then passed through a 2 linear fully-connected layers to get the logits. Just like the bigram model, this model is also trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset.

#### Implementation Pseudocode:

```python
n, embed_size = 10, 160
token_embedding = nn.Embedding(vocab_size, embed_size)
pos_embedding = ... # parameter, shape (n, embed_size)
adding_weight = ... # parameter, shape (n,)
fc = nn.Linear(embed_size, 200)
relu = nn.ReLU()
final = nn.Linear(200, vocab_size)

input_tokens = ... # shape (n,)
x = token_embedding(input_tokens) + pos_embedding
x = softmax(adding_weight) @ x
x = fc(x)
x = relu(x)
output = final(x) # shape (vocab_size,)
```

#### Sample Output

Sample output with input `"LUCENT"`, $n = 10$, and temperature $0.9$

```
LUCENTIO:
Ther to spared alle slandy woustry, to thins me! O puntempays,
But his a darce, be now on wil by tain estrains, rave have my, fave:
No, ad prove!

KING RICHARD II:
Shall:
I ders in's thus my red
A
```

### RNN Language Model (not implemented yet)

This model will probably use a LSTM network but is not implemented yet.
