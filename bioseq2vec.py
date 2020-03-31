from seq2vec import Seq2VecR2RHash
import argparse


def parse_args(train_data):
    '''
    Parses the BioSeq2vec arguments.
    '''
    parser = argparse.ArgumentParser(
        description="BioSeq2vec: a widely applicable biological sequence feature extraction method.")

    parser.add_argument('--input', nargs='?', default=str(train_data) + '.fa', type=str,
                        help='Input sequence data in fsata format.')

    parser.add_argument('--output', nargs='?', default=r'' + str(train_data) + '_embedding.txt',
                        help='Output file name.')

    parser.add_argument('--model', default="Seq2VecR2RHash", type=str)

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of embedding dimensions. Default is 128.')

    parser.add_argument('--sequence type', type=str, default=" RNA ",
                        help='the type of biological sequences.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    return parser.parse_args()


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


def get_k_nucleotide_composition(k, seq):
    seq_len = len(seq)
    words = []

    # tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1 - k):
        kmer = seq[x:x + k]
        words.append(kmer)
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


def prepare_seq2vec_feature():
    transformer_rna = Seq2VecR2RHash(
        max_index=10,
        max_length=10000,
        latent_size=200,
        embedding_size=200,
        encoding_size=100,
        learning_rate=0.05
    )
    transformer_protein = Seq2VecR2RHash(
        max_index=10,
        max_length=10000,
        latent_size=200,
        embedding_size=200,
        encoding_size=100,
        learning_rate=0.05
    )
    filename_rna = 'data/corpus/gencode.v29.lncRNA_transcripts.fa'
    filename_protein = 'data/corpus/gencode.v29.pc_translations.fa'
    filename = [filename_rna, filename_protein]
    # convert data
    rna, protein = data_convert(filename)
    # rna = read_fasta_file(filename_rna)
    # protein = read_fasta_file(filename_protein)

    # pretrain seq2vec
    pretrain(rna, transformer_rna, rna)
    pretrain(protein, transformer_protein, protein)

    return


if __name__ == "__main__":
    # input_file = sys.argv[1]
    transformer = Seq2VecR2RHash(
        max_index=10,
        max_length=100,
        latent_size=20,
        embedding_size=200,
        encoding_size=100,
        learning_rate=0.15
    )

    file_path = "data/corpus/gencode.v29.pc_translations.fa"
    seq_dict = read_fasta_file(file_path)
    sequences = []
    for seq in seq_dict.values():
        sequences.append(list(seq))  # list('AAAU') -> 'A','A','A','U'
    # print(sequences)
    pretrain(sequences, transformer, "protein")
