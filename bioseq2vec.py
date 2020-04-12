from numpy import shape
import numpy as np
from seq2vec import Seq2VecR2RHash, Seq2VecR2RWord
from seq2vec.util import DataGenterator
import argparse


def read_fasta_file(fasta_file):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        # let's discard the newline at the end (if any)
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:]  # discarding the initial >
            seq_dict[name] = ''
        else:
            # it is sequence
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    return seq_dict


def get_words(k, seq):
    # seq_len = len(seq)
    words = []

    # tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1 - k):
        kmer = seq[x:x + k]
        words.append(str(kmer))
    # tri_feature = [float(val)/seq_len for val in tmp_fea]
    # pdb.set_trace()
    return words


def data_convert(filenames):
    results = []
    for filename in filenames:
        result = []
        fr = open(filename, "r")
        seq = []
        for line in fr:
            if line[0] != '>':
                for i in line:
                    if i != '\n':
                        seq.append(i)
            else:
                result.append(list(seq))
                seq = []
        results.append(result)

    return results[0], results[1]


def data_convert_1(filename):
    result = []
    fr = open(filename, "r")
    seq = []
    for line in fr:
        if line[0] != '>':
            for i in line:
                if i != '\n':
                    seq.append(i)
        else:
            result.append(list(seq))

    return result


def pretrain(data, transformer, type):
    print("pretrain starts!")
    # data = np.array(data).tolist()
    transformer.fit(data)
    print("pretrain ends!")
    transformer.save_model("pretrained models/seq2vec_" + str(type) + ".model")  # save pretrained model

    return


if __name__ == "__main__":
    # input_file = sys.argv[1]
    transformer = Seq2VecR2RHash(
        max_index=10,
        max_length=300,
        latent_size=20,
        embedding_size=200,
        encoding_size=100,
        learning_rate=0.1
    )

    # transformer_wv = Seq2VecR2RWord()

    file_path = "data/corpus/gencode.v33.pc_translations.fa"
    seq_dict = read_fasta_file(file_path)
    sequences = []
    for seq in seq_dict.values():
        words = get_words(3, seq)
        sequences.append(words)  # list('AAAU') -> 'A','A','A','U'
    pretrain(sequences, transformer, "protein_word")

    ###################   test code ###################
    # transformer.load_customed_model(file_path='pretrained models/seq2vec_rna_word.model')
    # fea = transformer.transform(['AUC'])
    # print(fea)
    # # print(fea.reshape(-1))
