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
import sklearn.svm as svm

transformer_rna = Seq2VecR2RHash(
    max_index=10,
    max_length=100,
    latent_size=200,
    embedding_size=200,
    encoding_size=100,
    learning_rate=0.05
)
transformer_protein = Seq2VecR2RHash(
    max_index=10,
    max_length=100,
    latent_size=200,
    embedding_size=200,
    encoding_size=100,
    learning_rate=0.05
)


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
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[n % base]
        n = n / base
        ch2 = chars[n % base]
        n = n / base
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return nucle_com


def get_3_trids():
    '''
    Returns: List of all 4-mer nucleic acid combinations of RNA, e.g. [AAA,AAC,AAG，......UUG, UUU]
    -------
    '''

    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[n % base]
        n = n / base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com


def translate_sequence(seq):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    TranslationDict = TransDict_from_list()
    import string
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(string.maketrans(str(from_list), str(to_list)))
    # TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq


def TransDict_from_list():
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
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
        ch1 = chars[n % base]
        n = n / base
        ch2 = chars[n % base]
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


def read_fasta_file(fasta_file):
    seq_dict = [ ]
    fp = open(fasta_file, 'r')
    # name = ''
    for line in fp:
        # let's discard the newline at the end (if any)
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] != '>':  # or line.startswith('>')
            seq_dict.append(list(line.upper()))
    fp.close()

    return seq_dict

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

    return 0


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


def preprocess(dataset):
    rna_test = []
    pc_test = []
    label = []

    if dataset == "RPI2241":
        path_pc = "data/ncRNA-protein/" + dataset + "_protein.fa"
        path_rna = "data/ncRNA-protein/" + dataset + "_rna.fa"
        path = "data/ncRNA-protein/" + dataset + "_all.txt"

        # generate dictionary of rna and protein
        dic_pc = generate_dic(path_pc)
        dic_rna = generate_dic(path_rna)
        data = np.array(pd.read_csv(path, sep="\t")).tolist()

        for i in data:

            if i[0] in dic_pc.keys() and i[1] in dic_rna.keys():
                pc_test.append([dic_pc[i[0]]])
                rna_test.append([dic_rna[i[1]]])
                label.append(i[2])
            elif i[1] in dic_pc.keys() and i[0] in dic_rna.keys():
                pc_test.append([dic_pc[i[1]]])
                rna_test.append([dic_rna[i[0]]])
                label.append(i[2])
            else:
                print(i)
                continue
    elif dataset == "RPI488":
        path = "data/ncRNA-protein/lncRNA-protein-488.txt"
        fr = open(path, "r")
        count = 0
        for line in fr:
            count += 1
            if line[0] == '>':
                s = str(line).strip('\n').split('|', 2)
                # print(s)
                if s[1] == "non":
                    label.append(int(0))
                else:
                    label.append(int(1))
            elif count == 3:
                rna_test.append(list(line.strip('\n')))
                count = 0
            else:
                pc_test.append(list(line.strip('\n')))
    # print(len(pc_test))
    # print(len(rna_test))
    result_pc = transformer_protein.transform(pc_test)
    result_rna = transformer_rna.transform(rna_test)

    pd.DataFrame(result_pc).to_csv("transformedPC.csv")
    pd.DataFrame(result_rna).to_csv("transformedRNA.csv")

    return np.array(result_rna), np.array(result_pc), label


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


def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


def main(X_data, y):
    num_cross_val = 5  # 5-fold
    all_performance_svm = []
    all_performance_rf = []
    all_performance_ada = []
    all_performance_xgb = []

    all_labels = []
    all_prob = {}
    num_classifier = 4
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_prob[4] = []
    all_prob[5] = []
    all_average = []
    print(X_data.shape, X_data.shape[0], X_data.shape[1])

    for fold in range(num_cross_val):
        print("fold ", fold)
        train = np.array([x for i, x in enumerate(X_data) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X_data) if i % num_cross_val == fold])

        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        real_labels = []

        # print("train label")
        # print(train_label.shape)

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
        class_index = 0

        print('SVM')
        svm1 = svm.SVC(probability=True)
        svm1.fit(train, train_label)
        svm_proba = svm1.predict_proba(test)[:, 1]
        all_prob[0] = all_prob[0] + [val for val in svm_proba]
        y_pred_svm = transfer_label_from_prob(svm_proba)
        # print proba
        acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_svm, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_svm.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

        print('Random forest')
        rd = RandomForestClassifier(n_estimators=50, oob_score=True)
        rd.fit(train, train_label)
        rd_proba = rd.predict_proba(test)[:, 1]
        all_prob[1] = all_prob[1] + [val for val in rd_proba]
        y_pred_rf = transfer_label_from_prob(rd_proba)
        # print proba
        acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_rf, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_rf.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

        print('XGB')
        class_index = class_index + 1
        xgb1 = XGBClassifier(max_depth=6, booster='gbtree')  # learning_rate=0.1,max_depth=6, booster='gbtree'
        xgb1.fit(train, train_label)
        xgb_proba = xgb1.predict_proba(test)[:, 1]
        all_prob[2] = all_prob[2] + [val for val in xgb_proba]
        y_pred_xgb = transfer_label_from_prob(xgb_proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_xgb, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_xgb.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

        print('AdaBoost')
        class_index = class_index + 1
        Ada = AdaBoostClassifier()
        Ada.fit(train, train_label)
        proba = Ada.predict_proba(test)[:, 1]
        all_prob[3] = all_prob[3] + [val for val in proba]
        y_pred_ada = transfer_label_from_prob(proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performance(len(real_labels), y_pred_ada, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_ada.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

    print('mean performance of svm')
    print(np.mean(np.array(all_performance_svm), axis=0))
    print('---' * 50)
    print('mean performance of Random forest')
    print(np.mean(np.array(all_performance_rf), axis=0))
    print('---' * 50)
    print('mean performance of AdaBoost')
    print(np, mean(np.array(all_performance_ada), axis=0))
    print('---' * 50)
    print('mean performance of XGBoost')
    print(np, mean(np.array(all_performance_xgb), axis=0))
    print('---' * 50)

    Figure = plt.figure()
    plot_roc_curve(all_labels, all_prob[0], 'SVM')
    plot_roc_curve(all_labels, all_prob[1], 'Random Forest')
    plot_roc_curve(all_labels, all_prob[3], 'AdaBoost')
    plot_roc_curve(all_labels, all_prob[2], 'XGBoost')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.savefig(save_fig_dir + selected + '_' + class_type + '.png')
    plt.show()

    return


if __name__ == "__main__":
    # load corpus data
    filename_rna = "data/corpus/gencode.v29.lncRNA_transcripts.fa"
    filename_protein = "data/corpus/gencode.v29.pc_translations.fa"
    filename = [filename_rna, filename_protein]
    # convert data
    rna, protein = data_convert(filename)
    # rna = read_fasta_file(filename_rna)
    # protein = read_fasta_file(filename_protein)

    # pretrain seq2vec
    pretrain(rna, transformer_rna)
    pretrain(protein, transformer_protein)

    # prepare data
    dataset = "RPI488"
    rnaT, pcT, labels = preprocess(dataset)
    # rnaT = np.array(pd.read_csv("transformedPC.csv"))
    # pcT = np.array(pd.read_csv("transformedRNA.csv"))
    # print(labels)

    # predict
    X_data = np.concatenate((rnaT, pcT), axis=1)
    y = np.array(labels, dtype=int)
    # print(y)
    main(X_data, y)
