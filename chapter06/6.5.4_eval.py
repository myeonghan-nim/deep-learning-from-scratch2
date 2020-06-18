from rnnlm import RNNLM
from better_rnnlm import BetterRNNLM
from datasets import ptb
from commons.util import eval_perplexity

if __name__ == '__main__':
    # select model for evaluation
    model = RNNLM()
    # model = BetterRNNLM()

    # read tunned params
    model.load_params()
    corpus, _, _ = ptb.load_data('test')

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus)
    print('Test Perplexity:', ppl_test)
