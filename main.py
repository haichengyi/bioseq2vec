from random import randint

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from seq2vec import Seq2VecR2RHash
import numpy as np
import pandas as pd
from numpy import *
import matplotlib

matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import NuSVC, SVC

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


def prepare_seq2vec_feature():
    filename_rna = 'data/corpus/gencode.v29.lncRNA_transcripts.fa'
    filename_protein = 'data/corpus/gencode.v29.pc_translations.fa'
    filename = [filename_rna, filename_protein]
    # convert data
    rna, protein = data_convert(filename)
    # rna = read_fasta_file(filename_rna)
    # protein = read_fasta_file(filename_protein)

    # pretrain seq2vec
    pretrain(rna, transformer_rna)
    pretrain(protein, transformer_protein)

    return


def get_4_trids():
    '''
    Returns: List of all 4-mer nucleic acid combinations of RNA, e.g. [AAAA,AAAC,AAAG，......UUUG, UUUU]
    -------
    '''

    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[int(n % base)]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        n = n / base
        ch3 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return nucle_com


def translate_sequence(seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    # TranslationDict = TransDict_from_list()
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    # TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq


def TransDict_from_list(groups):
    # groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)  # Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
        index = index + 1
    return result


def get_3_protein_trids():
    '''
    Returns: List of all amino acid combinations of protein in 7 groups, e.g. [000,001,...006，......665, 666]
    -------
    '''
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com


def get_k_nucleotide_composition(tris, seq):
    '''
    Parameters
    ----------
    tris: List of all possible mers
    seq: input single sequence

    Returns: kmer feature of single sequence
    -------
    '''

    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1 - k):
        kmer = seq[x:x + k]
        if kmer in tris:
            ind = tris.index(kmer)
            tmp_fea[ind] = tmp_fea[ind] + 1
    tri_feature = [float(val) / seq_len for val in tmp_fea]
    # pdb.set_trace()
    return tri_feature


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


def read_orf_seq(fasta_file, RNA=False):
    protein_seq_dict = {}
    with open(fasta_file, 'r') as fp:
        for line in fp:
            line = line.rstrip()
            if line[0] == '>':
                name1 = line.split()
                name = name1[0][1:].strip()
                protein_seq_dict[name] = ''
            else:
                if RNA:
                    line = line.replace('T', 'U')
                protein_seq_dict[name] = protein_seq_dict[name] + line

    return protein_seq_dict


def pretrain(data, transformer):
    print("pretrain starts!")
    # data = np.array(data).tolist()
    transformer.fit(data)
    print("pretrain ends!")

    '''
    if type == "rna":
        transformer.save_model("attention_rna.h5")
    else:
        transformer.save_model("attention_pc.h5")'''

    return transformer


def generate_dic(filename):
    dic = {}
    fr = open(filename, "r")
    seq = ""
    for line in fr:
        if line[0] != '>':
            seq = line.strip('\n').upper()
        else:
            name = line[1:].strip('\n').upper()
            dic[name] = seq
            seq = ""

    return dic


def read_fasta_file(fasta_file):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    # pdb.set_trace()
    for line in fp:
        # let's discard the newline at the end (if any)
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:].upper()  # discarding the initial >
            seq_dict[name] = ''
        else:
            # it is sequence
            seq_dict[name] = seq_dict[name] + line
    fp.close()

    return seq_dict


def read_orf_interaction(interaction_file):
    interacton_pair = []
    with open(interaction_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue
            values = line.rstrip().split()
            protein, RNA = values[0].split('_')
            interacton_pair.append((protein, RNA))
    return interacton_pair


def prepare_RPI13254_feature(seperate=False, extract_only_posi=False, indep_test=False):
    protein_seq_dict = read_orf_seq('data/ncRNA-protein/RPI13254_RNA_seq.fa')
    RNA_seq_dict = read_orf_seq('data/ncRNA-protein/RPI13254_protein_seq.fa', RNA=True)
    positive_pairs = read_orf_interaction('data/ncRNA-protein/RPI13254_positive.txt')
    negative_pairs = read_orf_interaction('data/ncRNA-protein/RPI13254_negative.txt')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    train = []
    label = []
    if not indep_test:
        random.shuffle(positive_pairs)
        nega_num = len(negative_pairs)
        positive_pairs = positive_pairs[:nega_num]
    # pdb.set_trace()
    for val in positive_pairs:
        protein, RNA = val
        if RNA in RNA_seq_dict and protein in protein_seq_dict:
            label.append(1)
            # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
            RNA_seq = RNA_seq_dict[RNA]
            protein_seq = translate_sequence(protein_seq_dict[protein], group_dict)
            RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
            protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
            if seperate:
                tmp_fea = (protein_tri_fea, RNA_tri_fea)

            else:
                tmp_fea = protein_tri_fea + RNA_tri_fea

            train.append(tmp_fea)

        else:
            print(RNA, protein)
    if not extract_only_posi:
        for val in negative_pairs:
            protein, RNA = val
            if RNA in RNA_seq_dict and protein in protein_seq_dict:
                label.append(0)
                # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                RNA_seq = RNA_seq_dict[RNA]
                protein_seq = translate_sequence(protein_seq_dict[protein], group_dict)
                RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
                protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)

                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea

                train.append(tmp_fea)

            else:
                print(RNA, protein)

    return np.array(train), label


def read_name_from_fasta(fasta_file):
    name_list = []
    fp = open(fasta_file, 'r')
    for line in fp:
        if line[0] == '>':
            name = line.rstrip('\r\n')[1:]
            name_list.append(name.upper())
    fp.close()
    return name_list


def prepare_NPinter_feature(extract_only_posi=False, graph=False, deepmind=False, seperate=False, chem_fea=True):
    print('NPinter data')
    name_list = read_name_from_fasta('data/ncRNA-protein/NPinter_RNA_seq.fa')
    seq_dict = read_fasta_file('data/ncRNA-protein/NPinter_RNA_seq.fa')
    protein_seq_dict = read_fasta_file('data/ncRNA-protein/NPinter_protein_seq.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    # pdb.set_trace()
    train = []
    label = []
    chem_fea = []
    posi_set = set()
    pro_set = set()
    tris = get_4_trids()
    with open('data/ncRNA-protein/NPInter10412_dataset.txt', 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue
            RNA, RNA_len, protein, protein_len, org = line.rstrip().split('\t')
            RNA = RNA.upper()
            protein = protein.upper()
            posi_set.add((RNA, protein))
            pro_set.add(protein)
            if RNA in seq_dict and protein in protein_seq_dict:  # has_keys() in python 2
                label.append(1)
                # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                RNA_seq = seq_dict[RNA]
                protein_seq = translate_sequence(protein_seq_dict[protein], group_dict)

                RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
                protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)

                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea

                train.append(tmp_fea)

            else:
                print(RNA, protein)

    if not extract_only_posi:
        pro_list = list(pro_set)
        total_pro_len = len(pro_list)
        # get negative data
        with open('data/ncRNA-protein/NPInter10412_dataset.txt', 'r') as fp:
            head = True
            for line in fp:
                if head:
                    head = False
                    continue
                RNA, RNA_len, protein, protein_len, org = line.rstrip().split('\t')
                RNA = RNA.upper()
                protein = protein.upper()
                for val in range(50):
                    random_choice = randint(0, total_pro_len - 1)
                    select_pro = pro_list[random_choice]
                    selec_nega = (RNA, select_pro)
                    if selec_nega not in posi_set:
                        posi_set.add(selec_nega)
                        # print selec_nega
                        break
                if RNA in seq_dict and select_pro in protein_seq_dict:
                    # and RNA_fea_dict.has_key(RNA) and protein_fea_dict.has_key(select_pro) :
                    label.append(0)
                    # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                    RNA_seq = seq_dict[RNA]
                    protein_seq = translate_sequence(protein_seq_dict[select_pro], group_dict)

                    RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
                    protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
                    if seperate:
                        tmp_fea = (protein_tri_fea, RNA_tri_fea)
                    else:
                        tmp_fea = protein_tri_fea + RNA_tri_fea
                    train.append(tmp_fea)
                    # chem_fea.append(chem_tmp_fea)
                else:
                    print(RNA, protein)  # for key, val in RNA_fea_dict.iteritems():

    return np.array(train), label


def prepare_RPI1807_feature(graph=False, seperate=False, chem_fea=True):
    print('RPI1807 data')
    # name_list = read_name_from_fasta('ncRNA-protein/RNA_seq.fa')
    seq_dict = read_fasta_file('data/ncRNA-protein/RPI1807_RNA_seq.fa')
    protein_seq_dict = read_fasta_file('data/ncRNA-protein/RPI1807_protein_seq.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    # pdb.set_trace()
    train = []
    label = []
    chem_fea = []
    # pdb.set_trace()
    with open('data/ncRNA-protein/RPI1807_PositivePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            pro1 = pro1.upper()
            pro2 = pro2.upper()
            if pro2 in seq_dict and pro1 in protein_seq_dict:  # and protein_fea_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(1)
                RNA_seq = seq_dict[pro2]
                protein_seq = translate_sequence(protein_seq_dict[pro1], group_dict)

                RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
                protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
                # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                # tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)
                    # chem_tmp_fea = (protein_fea_dict[pro1], RNA_fea_dict[pro2])
                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea
                    # chem_tmp_fea = protein_fea_dict[pro1] + RNA_fea_dict[pro2]
                train.append(tmp_fea)
                # chem_fea.append(chem_tmp_fea)
            else:
                print(pro1, pro2)
    with open('data/ncRNA-protein/RPI1807_NegativePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            pro1 = pro1.upper()
            pro2 = pro2.upper()
            if pro2 in seq_dict and pro1 in protein_seq_dict:  # and protein_fea_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(0)
                RNA_seq = seq_dict[pro2]
                protein_seq = translate_sequence(protein_seq_dict[pro1], group_dict)

                RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
                protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
                # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                # tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)
                    # chem_tmp_fea = (protein_fea_dict[pro1], RNA_fea_dict[pro2])
                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea
                    # chem_tmp_fea = protein_fea_dict[pro1] + RNA_fea_dict[pro2]
                train.append(tmp_fea)
                # chem_fea.append(chem_tmp_fea)
            else:
                print(pro1, pro2)
    return np.array(train), label


def prepare_RPI2241_369_feature(rna_fasta_file, data_file, protein_fasta_file, seperate=False):
    seq_dict = read_fasta_file(rna_fasta_file)
    protein_seq_dict = read_fasta_file(protein_fasta_file)
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    train = []
    label = []
    chem_fea = []
    # posi_set = set()
    # pro_set = set()
    with open(data_file, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, RNA, tmplabel = line.rstrip('\r\n').split('\t')
            if RNA in seq_dict and protein in protein_seq_dict:
                label.append(int(tmplabel))
                RNA_seq = seq_dict[RNA]
                protein_seq = translate_sequence(protein_seq_dict[protein], group_dict)

                RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
                protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)
                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea
                train.append(tmp_fea)
            else:
                print(RNA, protein)

    return np.array(train), label


def prepare_RPI488_feature(seperate=False, chem_fea=True):
    print('RPI488 dataset')

    seq2vec_rna = Seq2VecR2RHash()
    seq2vec_pro = Seq2VecR2RHash()
    seq2vec_rna.load_customed_model("pretrained models/seq2vec_rna.model")  # maybe this way is wrong
    seq2vec_pro.load_customed_model("pretrained models/seq2vec_protein.model")

    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    with open('data/ncRNA-protein/lncRNA-protein-488.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label = values[1]
                name = values[0].split('_')
                protein = name[0] + '-' + name[1]
                RNA = name[0] + '-' + name[2]
                if label == 'interactive':
                    interaction_pair[(protein, RNA)] = 1
                else:
                    interaction_pair[(protein, RNA)] = 0
                index = 0
            else:
                seq = line[:-1]
                if index == 0:
                    protein_seq_dict[protein] = seq
                else:
                    RNA_seq_dict[RNA] = seq
                index = index + 1
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()

    train = {}  # Changed
    train[0] = []  # for kmer feature
    train[1] = []  # for seq2vec feature

    label = []
    for key, val in interaction_pair.items():  # iteritems() removed in python 3
        protein, RNA = key[0], key[1]
        if RNA in RNA_seq_dict and protein in protein_seq_dict:
            # python2: protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
            label.append(val)
            RNA_seq = RNA_seq_dict[RNA]
            protein_seq = protein_seq_dict[protein]

            # seq2vec feature
            RNA_seq2vec_fea = seq2vec_rna.transform([list(RNA_seq)])  # list(RNA_seq)
            RNA_seq2vec_fea = RNA_seq2vec_fea.reshape(-1)  # [[]] -> []  2D array to 1D array
            protein_seq2vec_fea = seq2vec_pro.transform([list(protein_seq)])  # list(protein_seq)
            protein_seq2vec_fea = protein_seq2vec_fea.reshape(-1)  # [[]] -> []

            protein_seq = translate_sequence(protein_seq_dict[protein], group_dict)  # reduced Alphabet
            RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
            protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)
            if seperate:
                tmp_fea = (protein_tri_fea, RNA_tri_fea)
                tmp_2vec = (protein_seq2vec_fea, RNA_seq2vec_fea)
            else:
                tmp_fea = protein_tri_fea + RNA_tri_fea
                tmp_2vec = protein_seq2vec_fea + RNA_seq2vec_fea
            train[0].append(tmp_fea)
            train[1].append(tmp_2vec)
        else:
            print(RNA, protein)

    return train, label


def prepare_RPIntDB_feature(seperate=False, chem_fea=True):
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    # tris3 = get_3_trids()
    train = []
    label = []
    chem_fea = []
    head = True
    with open('data/ncRNA-protein/RPIntDB_interactions_new.txt', 'r') as fp:
        for line in fp:
            if head:
                head = False
                continue
            values = line.rstrip('\r\n').split('\t')
            protein = values[0]
            protein_seq = values[1]
            RNA = values[2]
            RNA_seq = values[3]
            label.append(1)
            RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
            protein_tri_fea = get_k_nucleotide_composition(protein_tris, protein_seq)

            if seperate:
                tmp_fea = (protein_tri_fea, RNA_tri_fea)
                # chem_tmp_fea = (protein_fea_dict[protein], RNA_fea_dict[RNA])
            else:
                tmp_fea = protein_tri_fea + RNA_tri_fea
                # chem_tmp_fea = protein_fea_dict[protein] + RNA_fea_dict[RNA]
            train.append(tmp_fea)

    return np.array(train), label


def get_data(dataset, seperate=False):
    if dataset == 'RPI1807':
        X, labels = prepare_RPI1807_feature(graph=False, seperate=seperate)
    elif dataset == 'NPInter':
        X, labels = prepare_NPinter_feature(graph=False, seperate=seperate)
    elif dataset == 'RPI2241':
        X, labels = prepare_RPI2241_369_feature('data/ncRNA-protein/RPI2241_rna.fa',
                                                'data/ncRNA-protein/RPI2241_all.txt',
                                                'data/ncRNA-protein/RPI2241_protein.fa',
                                                seperate=seperate)
    elif dataset == 'RPI369':
        X, labels = prepare_RPI2241_369_feature('data/ncRNA-protein/RPI369_rna.fa', 'data/ncRNA-protein/RPI369_all.txt',
                                                'data/ncRNA-protein/RPI369_protein.fa',
                                                seperate=seperate)
    elif dataset == 'RPI488':
        X, labels = prepare_RPI488_feature(seperate=seperate)
    elif dataset == 'RPIntDB':
        X, labels = prepare_RPIntDB_feature(seperate=seperate)
    elif dataset == 'RPI13254':
        X, labels = prepare_RPI13254_feature(seperate=seperate)

    return X, labels


# def preprocess(dataset):
#     rna_test = []
#     pc_test = []
#     label = []
#
#     if dataset == "RPI2241":
#         path_pc = "data/ncRNA-protein/" + dataset + "_protein.fa"
#         path_rna = "data/ncRNA-protein/" + dataset + "_rna.fa"
#         path = "data/ncRNA-protein/" + dataset + "_all.txt"
#
#         # generate dictionary of rna and protein
#         dic_pc = generate_dic(path_pc)
#         dic_rna = generate_dic(path_rna)
#         data = np.array(pd.read_csv(path, sep="\t")).tolist()
#
#         for i in data:
#
#             if i[0] in dic_pc.keys() and i[1] in dic_rna.keys():
#                 pc_test.append([dic_pc[i[0]]])
#                 rna_test.append([dic_rna[i[1]]])
#                 label.append(i[2])
#             elif i[1] in dic_pc.keys() and i[0] in dic_rna.keys():
#                 pc_test.append([dic_pc[i[1]]])
#                 rna_test.append([dic_rna[i[0]]])
#                 label.append(i[2])
#             else:
#                 print(i)
#                 continue
#     elif dataset == "RPI488":
#         path = "data/ncRNA-protein/lncRNA-protein-488.txt"
#         fr = open(path, "r")
#         count = 0
#         for line in fr:
#             count += 1
#             if line[0] == '>':
#                 s = str(line).strip('\n').split('|', 2)
#                 # print(s)
#                 if s[1] == "non":
#                     label.append(int(0))
#                 else:
#                     label.append(int(1))
#             elif count == 3:
#                 rna_test.append(list(line.strip('\n')))
#                 count = 0
#             else:
#                 pc_test.append(list(line.strip('\n')))
#     # print(len(pc_test))
#     # print(len(rna_test))
#     result_pc = transformer_protein.transform(pc_test)
#     result_rna = transformer_rna.transform(rna_test)
#
#     pd.DataFrame(result_pc).to_csv("transformedPC.csv")
#     pd.DataFrame(result_rna).to_csv("transformedRNA.csv")
#
#     return np.array(result_rna), np.array(result_pc), label


def plot_roc_curve(labels, probality, legend_text, auc_tag=True):
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)


def calculate_performance(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC


def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])
    return np.array(formated_matrix1), np.array(formated_matrix2)


def preprocess_data(X, scaler=None, stand=True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X


def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


def main(X_data, y):
    print(shape(X_data))
    num_cross_val = 5  # 5-fold
    all_performance_svm = []
    all_performance_rf = []
    all_performance_ada = []
    all_labels = []
    all_prob = {}
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []

    for fold in range(num_cross_val):
        print("fold ", fold)
        train = np.array([x for i, x in enumerate(X_data) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X_data) if i % num_cross_val == fold])

        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        real_labels = []

        for val in test_label:
            # generate test data
            if val == 1:
                real_labels.append(1)
            else:
                real_labels.append(0)

        train_label_new = []
        for val in train_label:
            # generate train data
            if val == 1:
                train_label_new.append(1)
            else:
                train_label_new.append(0)

        all_labels = all_labels + real_labels

        print('SVM')
        svm1 = SVC(probability=True)
        svm1.fit(train, train_label)
        svm_proba = svm1.predict_proba(test)[:, 1]
        all_prob[0] = all_prob[0] + [val for val in svm_proba]
        y_pred_svm = transfer_label_from_prob(svm_proba)
        # print proba
        acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_svm, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_svm.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

        print('AdaBoost')
        Ada = AdaBoostClassifier()
        Ada.fit(train, train_label)
        proba = Ada.predict_proba(test)[:, 1]
        all_prob[1] = all_prob[1] + [val for val in proba]
        y_pred_ada = transfer_label_from_prob(proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_ada, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_ada.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

        print('Random forest')
        rd = RandomForestClassifier(n_estimators=51)
        rd.fit(train, train_label)
        rd_proba = rd.predict_proba(test)[:, 1]
        all_prob[2] = all_prob[2] + [val for val in rd_proba]
        y_pred_rf = transfer_label_from_prob(rd_proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_rf, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_rf.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

        # print('XGB')
        # xgb1 = XGBClassifier(max_depth=6, booster='gbtree')  # learning_rate=0.1,max_depth=6, booster='gbtree'
        # xgb1.fit(train_kmer, train_label)
        # xgb_proba = xgb1.predict_proba(test_kmer)[:, 1]
        # all_prob[2] = all_prob[2] + [val for val in xgb_proba]
        # y_pred_xgb = transfer_label_from_prob(xgb_proba)
        # acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_xgb, real_labels)
        # print(acc, precision, sensitivity, specificity, MCC)
        # all_performance_xgb.append([acc, precision, sensitivity, specificity, MCC])
        # print('---' * 50)

    return all_performance_svm, all_performance_ada, all_performance_rf, all_labels, all_prob


if __name__ == "__main__":
    # prepare data
    dataset = "RPI488"
    # get_data()方法同时回传kmer, seq2vec特征
    X, labels = get_data(dataset)
    X1, X2 = X[0], X[1]
    X1 = preprocess_data(X1)  # 特征StandardScaler()归一化
    X2 = preprocess_data(X2)
    y = np.array(labels, dtype=int)

    # 以 kmer特征的 ‘X1’ 调用main()方法
    all_performance_svm1, all_performance_ada1, all_performance_rf1, all_labels, all_prob = main(X1, y)
    # 以seq2vec特征的 ‘X2’ 再次调用main()
    all_performance_svm2, all_performance_ada2, all_performance_rf2, all_labels2, all_prob2 = main(X2, y)

    print('mean performance of svm using kmer feature')
    print(np.mean(np.array(all_performance_svm1), axis=0))
    print('---' * 50)
    print('mean performance of AdaBoost using kmer feature')
    print(np, mean(np.array(all_performance_ada1), axis=0))
    print('---' * 50)
    print('mean performance of Random forest using kmer feature')
    print(np.mean(np.array(all_performance_rf1), axis=0))
    print('---' * 50)

    print('mean performance of svm using bioseq2vec feature')
    print(np.mean(np.array(all_performance_svm2), axis=0))
    print('---' * 50)
    print('mean performance of AdaBoost using bioseq2vec feature')
    print(np, mean(np.array(all_performance_ada2), axis=0))
    print('---' * 50)
    print('mean performance of Random forest using bioseq2vec feature')
    print(np.mean(np.array(all_performance_rf2), axis=0))
    print('---' * 50)

    Figure = plt.figure()
    plot_roc_curve(all_labels, all_prob[0], 'kmer_SVM')
    plot_roc_curve(all_labels, all_prob[1], 'kmer_AdaBoost')
    plot_roc_curve(all_labels, all_prob[2], 'kmer_Random Forest')
    plot_roc_curve(all_labels2, all_prob2[0], 'bioseq2vec_SVM')
    plot_roc_curve(all_labels2, all_prob2[1], 'bioseq2vec_AdaBoost')
    plot_roc_curve(all_labels2, all_prob2[2], 'bioseq2vec_Random Forest')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('result/' + dataset + '.tif', dpi=300)  # .svg
    plt.show()
