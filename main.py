from seq2vec import Seq2VecR2RHash

transformer = Seq2VecR2RHash(
    max_index=1000,
    max_length=10,
    latent_size=20,
    embedding_size=200,
    encoding_size=30,
    learning_rate=0.05
)

train_seq = [
    ['我', '有', '一個', '蘋果'],
    ['我', '有', '筆'],
    ['一個', '鳳梨'],
    ['你好', '吃西瓜']
]
test_seq = [
    ['我', '愛', '吃', '鳳梨'],
]
transformer.fit(train_seq)
result = transformer.transform(test_seq)
print(result)