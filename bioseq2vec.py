from seq2vec import Seq2VecHash, Seq2VecC2RChar, Seq2VecC2RWord, Seq2VecR2RHash, Seq2VecR2RWord
import argparse

def parse_args(train_data):
	'''
	Parses the BioSeq2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="BioSeq2vec: a widely applicable biological sequence feature extraction method.")

	parser.add_argument('--input', nargs='?', default=str(train_data)+'.fa', type= str,
						help='Input sequence data in fsata format.')

	parser.add_argument('--output', nargs='?', default=r''+str(train_data)+'_embedding.txt',
						help='Output file name.')

	parser.add_argument('--model', default="Seq2VecBase", type=str,
						help='Optional training models: Seq2VecHash, Seq2VecR2RHash, Seq2VecC2RChar, Seq2VecC2RWord, Seq2VecR2RWord, TrainableSeq2VecBase')

	parser.add_argument('--dimensions', type=int, default = 128,
						help='Number of embedding dimensions. Default is 128.')

	parser.add_argument('--sequence type', type= str, default= " RNA ",
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

    #tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        words.append(kmer)
    #tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()
    return words

if __name__ == "__main__":
	# input_file = sys.argv[1]
	sentences = []
	args = parse_args(train_data="RNA.fa")
	transformer = args.model
	sequence_dict = read_fasta_file(args.input)
	words = get_k_nucleotide_composition(4,sequence_dict)
	sentences.append(s)


