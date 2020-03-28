"""Sequence-to-Sequence Auto Encoder."""
import keras.models
from keras.models import Input
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, LSTM, Permute, Reshape,Lambda, RepeatVector, Multiply
from keras.models import Model
from keras.layers import Flatten

from yklz import BidirectionalRNNEncoder, RNNDecoder
from yklz import RNNCell, Pick

from seq2vec.transformer import HashIndexTransformer
from seq2vec.transformer import OneHotEncodedTransformer
from seq2vec.model import TrainableSeq2VecBase


class Seq2VecR2RHashA(TrainableSeq2VecBase):
    """Hash words and feed to seq2seq auto-encoder.

    Attributes
    ----------
    max_index: int
        The length of input vector.

    max_length: int
        The length of longest sequence.

    latent_size: int
        The returned latent vector size after encoding.

    """

    def __init__(
            self,
            max_index=300,
            max_length=10,
            encoding_size=100,
            learning_rate=0.0001,
            word_embedding_size=64,
            latent_size=20,
            **kwargs
        ):
        self.max_index = max_index
        self.word_embedding_size = word_embedding_size
        self.encoding_size = encoding_size

        self.input_transformer = HashIndexTransformer(
            max_index, max_length
        )
        self.output_transformer = OneHotEncodedTransformer(
            max_index, max_length
        )

        super(Seq2VecR2RHashA, self).__init__(
            max_length,
            latent_size,
            learning_rate
        )
        self.custom_objects['BidirectionalRNNEncoder']= BidirectionalRNNEncoder
        self.custom_objects['RNNDecoder'] = RNNDecoder
        self.custom_objects['RNNCell'] = RNNCell
        self.custom_objects['Pick'] = Pick

    def attention_3d_block(self,inputs):
        #print(inputs)
        INPUT_DIM = 2
        TIME_STEPS = self.max_length
        # if True, the attention vector is shared across the input_dimensions where the attention is applied.
        SINGLE_ATTENTION_VECTOR = False
        APPLY_ATTENTION_BEFORE_LSTM = False
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        x = Lambda(lambda x: x, output_shape=lambda s: s)(inputs)
        a = Permute((2, 1))(x)
        print(a.get_shape())
        print(input_dim)
        print(TIME_STEPS)
        #a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
        a = Dense(TIME_STEPS, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        print(output_attention_mul)
        return output_attention_mul

    def create_model(
            self,
            rho=0.9,
            decay=0.0,
        ):

        inputs = Input(
            shape=(
                self.max_length,
            )
        )
        char_embedding = Embedding(
            input_dim=self.max_index,
            output_dim=self.word_embedding_size,
            input_length=self.max_length,
            mask_zero=True,
        )(inputs)

        encoded = BidirectionalRNNEncoder(
            RNNCell(
                LSTM(
                    units=self.latent_size,
                    dropout=0.,
                    recurrent_dropout=0.,
                    return_sequences=True
                ),
                Dense(
                    units=self.encoding_size // 2,
                    activation='tanh'
                ),
                dense_dropout=0.
            )
        )(char_embedding)
        attention_mul = self.attention_3d_block(encoded)
        #attention_mul = Flatten()(attention_mul)
        decoded = RNNDecoder(
            RNNCell(
                LSTM(
                    units=self.latent_size,
                    dropout=0.,
                    recurrent_dropout=0.
                ),
                Dense(
                    units=self.encoding_size,
                    activation='tanh'
                ),
                dense_dropout=0.
            )
        )(attention_mul)
        outputs = TimeDistributed(
            Dense(
                units=self.max_index,
                activation='softmax'
            )
        )(decoded)

        model = Model(inputs, outputs)


        picked = Pick()(encoded)
        print(picked)
        encoder = Model(inputs, picked)
        #print(decoded)

        optimizer = RMSprop(
            lr=self.learning_rate,
            rho=rho,
            decay=decay,
        )
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        print(model.summary())

        return model, encoder

    def load_model(self, file_path):
        self.model = self.load_customed_model(file_path)
        picked = Pick()(self.model.get_layer(index=2).output)
        self.encoder = Model(
            self.model.input,
            picked
        )
        self.max_index = self.model.get_layer(index=1).input_dim
        self.max_length = self.model.input_shape[1]
        self.word_embedding_size = self.model.get_layer(index=1).output_dim
        self.latent_size = self.model.get_layer(index=2).layer.recurrent_layer.units
        self.encoding_size = self.model.get_layer(index=2).layer.dense_layer.units * 2

        self.input_transformer = HashIndexTransformer(
            self.max_index,
            self.max_length
        )
        self.output_transformer = OneHotEncodedTransformer(
            self.max_index,
            self.max_length
        )

