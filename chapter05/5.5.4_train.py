from commons.trainer import RnnlmTrainer
from commons.optimizer import SGD
from datasets import ptb
from simple_rnnlm import SimpleRnnlm

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

# init model
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()
