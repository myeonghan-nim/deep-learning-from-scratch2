from commons.optimizer import SGD
from datasets import ptb
from simple_rnnlm import SimpleRnnlm
import matplotlib.pyplot as plt
import numpy as np

# set hyperparameters
batch_size = 10
wordvec_size = 100
hidden_size = 100  # number of hidden vectors in RNN
time_size = 5  # spread time area at Truncated BPTT once
lr = 0.1
max_epoch = 100

# read datasets(first 1000)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # input
ts = corpus[1:]  # output(answer)
data_size = len(xs)
print(f'Corpus Size: {corpus_size}, Number of Vocab: {vocab_size}')

# variables for learning
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# init model
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# calculate start reading point of each sample of mini batch
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # get mini batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # renewal params with grads
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # evalueate perplexity of each epoch
    ppl = np.exp(total_loss / loss_count)
    print(f'| Epoch {epoch + 1} | Perplexity {ppl:.2f}')
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# draw graph
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
