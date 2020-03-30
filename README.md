# Usage example

```python
from seq2vec import Seq2VecHash

transformer = Seq2VecHash(vector_length=100)
train_seqs = [  # corpus for pretain
    ['AGU', 'C', 'CG', ..., 'UUCGA'],
    ['U', 'GAU',..., 'CGAU'],
    ...,
]
test_seqs =[
    ['UAG','CCU',...,'AUA'],
    ['UAG','GCCA',...,'GACU'],
    ...,
]
transformer.fit(train_seqs)
result = transformer.transform(test_seqs)
print(result)
'''
array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
'''
```

### Training with generator on file

We provide an example with LSTM to LSTM auto-encoder (word embedding).

Use the following training method while lack of memory is an issue for you.

The file should be a tokenized txt file splitted by whitespace with a sequence
per line.

```python
from seq2vec.word2vec import GensimWord2vec

from seq2vec.model import Seq2VecR2RWord
from seq2vec.transformer import WordEmbeddingTransformer
from seq2vec.util import DataGenterator

word2vec = GensimWord2vec(word2vec_model_path)
max_length = 20

transformer = Seq2VecR2RWord(
    word2vec_model=word2vec,
    max_length=max_length,
    latent_size=200,
    encoding_size=300,
    learning_rate=0.05
)

train_data = DataGenterator(
    corpus_for_training_path, 
    transformer.input_transformer,
    transformer.output_transformer, 
    batch_size=128
)
test_data = DataGenterator(
    corpus_for_validation_path, 
    transformer.input_transformer,
    transformer.output_transformer, 
    batch_size=128
)

transformer.fit_generator(
    train_data,
    test_data,
    epochs=10,
    batch_number=1250 # The number of batch per epoch
)

transformer.save_model(model_path) # save your model

# You can reload your model and retrain it.
transformer.load_model(model_path)
transformer.fit_generator(
    train_data,
    test_data,
    epochs=10,
    batch_number=1250 # The number of batch per epoch
)
```


