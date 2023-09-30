# Mini Language Models

A significant portion of the code is from the tutorial [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2064s&ab_channel=AndrejKarpathy) by Andrej Karpathy.

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

### GPT (not implemented yet)

This model is not implemented yet.
