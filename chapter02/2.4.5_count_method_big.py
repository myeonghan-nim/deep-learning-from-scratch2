from commons.util import most_similar, create_co_matrix, ppmi
from datasets import ptb
import numpy as np

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('Calculating coincide number ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('Calculating PPMI ...')
W = ppmi(C, verbose=True)

print('Calculating SVD ...')
try:
    # truncated SVD(fast)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(
        W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    # SVD(slow)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
