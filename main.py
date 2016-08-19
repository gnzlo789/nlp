import utils
import model

#import tensorflow as tf

WORD_VECTOR_SIZE = 50
NUMBER_OF_EPOCHS = 500

babi_train_raw, babi_test_raw = utils.get_babi_raw("1", "1") # First argument is babi_id: babi task id, second is babi_id of test set

word2vec = utils.load_glove(WORD_VECTOR_SIZE)

dmn = model.DMNModel(babi_train_raw, babi_test_raw, word2vec)

