import torch
import torch.nn as nn
import re
import nltk
import itertools
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable

vocabulary_size = 2500
unknown_token = "UNKNOWN_TOKEN"
sentence_start = "START"
sentence_end = "END"


def preprocessing():
    data = open('test2.txt', 'r', encoding="utf-8-sig")
    lines = data.read()
    data.close()
    all_sentences = [sent for sent in nltk.sent_tokenize(lines)]
    all_filtered = []
    for sent in all_sentences:
        filtered = re.sub('[A-Z]+[A-Z]+[A-Z]*', '', sent)
        filtered = re.sub('Mr\s*\.|Mrs\s*\.|Dr\s*\.', '', filtered)
        filtered = re.sub('\s+[A-Z]+[a-z]+', '', filtered)
        filtered = re.sub(" \d+", '', filtered)
        filtered = re.sub("[:,;“”]", ' ', filtered)
        filtered = filtered.lower()
        filtered = re.sub("\.", ' END', filtered)
        all_filtered.append("%s %s" % (sentence_start, filtered))
    all_tokens = []
    for sent in all_filtered:
        all_tokens.append(nltk.word_tokenize(sent))
    word_frequencies = nltk.FreqDist(itertools.chain(*all_tokens))  # распределение слов
    vocabulary = word_frequencies.most_common(vocabulary_size - 1)  # выбираем vocabulary_size самых часто употребляемых
    index_to_word = [tup[0] for tup in vocabulary]  # просто список слов
    index_to_word.append(unknown_token)  # unknown_token - любое слово, не принадлежащее словарю
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    for i, sentence in enumerate(all_tokens):
        # заменяем слова не из словаря на unknown_token
        all_tokens[i] = [word if word in word_to_index else unknown_token for word in sentence]

    X_train = np.asarray([[word_to_index[word] for word in sent[:-1]] for sent in all_tokens])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in all_tokens])

    f = open('X_train.txt', 'w')
    for sent in X_train:
        f.write(str(sent) + '\n')

    return X_train, y_train, index_to_word  # возвращаем вместе с метрикой(так надо)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.l_in = nn.Linear(input_size, hidden_size)
        self.l_hidden = nn.Linear(hidden_size, hidden_size)
        self.l_out = nn.Linear(hidden_size, output_size)
        # self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.tan = nn.Tanh()

    def forward(self, input, hidden):
        inp = self.relu(self.l_in(input))
        hidden = self.tan(self.l_hidden(hidden + inp))
        output = self.l_out(hidden)
        # output = self.dropout(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size), requires_grad=True)


def sentence_to_tensor(sentence):
    tensor = torch.zeros(len(sentence), 1, vocabulary_size)
    for wi, word in enumerate(sentence):
        tensor[wi][0][word] = 1
    return tensor


def pick_random(l):
    random_index = random.randint(0, len(l) - 1)
    return l[random_index], random_index


def random_training_ex(X_train, y_train):
    random_X_train_example, random_index = pick_random(X_train)
    random_y_train_example = y_train[random_index]

    x_tensor = sentence_to_tensor(random_X_train_example)
    y_tensor = torch.LongTensor(random_y_train_example)

    return x_tensor, y_tensor


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


epochs = 10000
print_every = 500
plot_every = 50
hidden_size = 100
learning_rate = 0.005

criterion = nn.CrossEntropyLoss()


# criterion = nn.NLLLoss()


def main():
    X_train, y_train, index_to_word = preprocessing()

    model = RNN(vocabulary_size, hidden_size, vocabulary_size)

    start = time.time()
    all_losses = []
    current_loss = 0

    optimizer = torch.optim.Adam(model.parameters())

    for iter in range(1, epochs + 1):
        x_tensor, y_tensor = random_training_ex(X_train, y_train)
        loss = train(model, optimizer, y_tensor, x_tensor)
        current_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.10f' % (time_since(start), iter, iter / epochs * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)

    for i in range(30):
        print(evaluate(model, index_to_word, temperature=0.8))


def train(rnn, optimizer, y_tensor, x_tensor):
    y_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0

    # complete bptt
    T = x_tensor.size()[0]
    for t in range(T):
        output, hidden = rnn(x_tensor[t], hidden)
        loss += criterion(output, y_tensor[t])

    # truncated bptt
    """T = x_tensor.size()[0]
    k = 15
    if T > k:
        for p in rnn.parameters():
            p.requires_grad = False

    for t in range(T):
        output, hidden = rnn(x_tensor[t], hidden)
        if T - t == k:
            for p in rnn.parameters():
                p.requires_grad = True
        if T - t < k:
            loss += criterion(output, y_tensor[t])"""

    loss.backward()
    optimizer.step()

    # updating weights explicitly
    # for p in rnn.parameters():
    #    p.data.add_(-learning_rate, p.grad.data)

    return loss.item() / x_tensor.size(0)


def evaluate(rnn, metric, temperature=0.8):
    length = random.randint(10, 25)
    result = []

    input = sentence_to_tensor(np.array([1]))
    hidden = rnn.initHidden()

    len_interval = np.array([i for i in range(10, 25)])
    for i in range(length):
        output, hidden = rnn(input[0], hidden)

        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        pred_word = metric[top_i]
        if sentence_end == pred_word and i in len_interval:
            break
        elif unknown_token == pred_word:
            continue
        elif sentence_end != pred_word:
            result.append(pred_word)

        input = sentence_to_tensor(np.array([top_i]))

    return " ".join(result)


main()
