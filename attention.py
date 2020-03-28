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
# from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import sklearn.svm as svm

transformer_rna = Seq2VecR2RHash(
    Seq2VecR2RHashA(attention),
    max_index=1000,
    max_length=100,
    latent_size=20,
    embedding_size=100,
    encoding_size=100,
    learning_rate=0.05
)
transformer_pc = Seq2VecR2RHash(
    max_index=1000,
    max_length=100,
    latent_size=20,
    embedding_size=100,
    encoding_size=100,
    learning_rate=0.05
)

filename_rna = "data/train/gencode.v29.lncRNA_transcripts.fa"
filename_pc = "data/train/gencode.v29.pc_translations.fa"


def data_convert(filenames):
    results = []
    for filename in filenames:
        result = []
        fr = open(filename, "r")
        seq = ""
        for line in fr:
            if line[0] != '>':
                for i in line:
                    if i != '\n':
                        seq = seq + str(i)
            else:
                result.append([seq])
                seq = ""
        results.append(result)

    return results[0], results[1]


def trainingModel(data, transformer, type):
    print("training starts!")
    data = np.array(data).tolist()
    transformer.fit(data[1:])
    print("training ends!")

    '''
    if type == "rna":
        transformer.save_model("attention_rna.h5")
    else:
        transformer.save_model("attention_pc.h5")'''

    return 0


def generate_dic(filename):
    dic = {}
    fr = open(filename, "r")
    seq = " "
    for line in fr:
        if line[0] != '>':
            seq = line.strip('\n').upper()
        else:
            name = line[1:].strip('\n').upper()
            dic[name] = seq
            seq = " "

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


    elif dataset == "RPI448":

        path = "data/ncRNA-protein/lncRNA-protein-488.txt"

        # data = np.array(pd.read_csv(path, sep="\t")).tolist()
        # print(data)

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
                rna_test.append([line.strip('\n')])
                count = 0
            else:
                pc_test.append([line.strip('\n')])

    # print(len(pc_test))
    # print(len(rna_test))
    result_pc = transformer_pc.transform(pc_test)
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


def calculate_performace(test_num, pred_y, labels):
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


def Lattention(X_data, y):
    num_cross_val = 5  # 5-fold
    all_performance_xgb = []
    all_performance_rf = []
    all_performance_ada = []
    all_performance_knn = []
    all_performance_stack = []
    all_performance_lstm = []
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
    # print(X_data)

    for fold in range(num_cross_val):
        print("fold ", fold)
        # print()
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

        # print('SVM')
        # svm1 = svm.SVC(probability=True)
        # svm1.fit(train, train_label)
        # svm_proba = svm1.predict_proba(test)[:, 1]
        # all_prob[0] = all_prob[0] + [val for val in svm_proba]
        # y_pred_lgbm = transfer_label_from_prob(svm_proba)
        # # print proba
        # acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_lgbm, real_labels)
        # print(acc, precision, sensitivity, specificity, MCC)
        # all_performance_lgbm.append([acc, precision, sensitivity, specificity, MCC])
        # print('---' * 50)

        print('Random forest')
        rd = RandomForestClassifier(n_estimators=50, oob_score=True)
        rd.fit(train, train_label)
        rd_proba = rd.predict_proba(test)[:, 1]
        all_prob[1] = all_prob[1] + [val for val in rd_proba]
        y_pred_lgbm = transfer_label_from_prob(rd_proba)
        # print proba
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_lgbm, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_rf.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

        '''print ('XGB')
        class_index = class_index + 1
        xgb1 = XGBClassifier(max_depth=6,booster='gbtree')#learning_rate=0.1,max_depth=6, booster='gbtree'
        xgb1.fit(train, train_label)
        xgb_proba = xgb1.predict_proba(test)[:, 1]
        all_prob[1] = all_prob[1] + [val for val in xgb_proba]
        tmp_aver = [val1 + val2 / 4 for val1, val2 in zip(xgb_proba, tmp_aver)]
        y_pred_xgb = transfer_label_from_prob(xgb_proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_xgb, real_labels)
        print (acc, precision, sensitivity, specificity, MCC)
        all_performance_xgb.append([acc, precision, sensitivity, specificity, MCC])
        get_blend_data(class_index, xgb1, skf, test, train, np.array(train_label_new), blend_train, blend_test)
        print ('---' * 50)'''

        print('AdaBoost')
        class_index = class_index + 1
        Ada = AdaBoostClassifier()
        Ada.fit(train, train_label)
        proba = Ada.predict_proba(test)[:, 1]
        all_prob[3] = all_prob[3] + [val for val in proba]
        # tmp_aver = [val1 + val2 / 4 for val1, val2 in zip(proba, tmp_aver)]
        y_pred_gnb = transfer_label_from_prob(proba)
        # y_pred_stack = blcf.predict(test)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_gnb, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_ada.append([acc, precision, sensitivity, specificity, MCC])
        # get_blend_data(class_index, Ada, skf, test, train, np.array(train_label_new), blend_train, blend_test)
        print('---' * 50)

    # print('mean performance of XGB')
    # print(np.mean(np.array(all_performance_xgb), axis=0))
    # print('---' * 50)
    print('mean performance of Random forest')
    print(np.mean(np.array(all_performance_rf), axis=0))
    print('---' * 50)
    print('mean performance of AdaBoost')
    print(np, mean(np.array(all_performance_ada), axis=0))
    print('---' * 50)

    Figure = plt.figure()
    # plot_roc_curve(all_labels, all_prob[0], 'SVM')
    plot_roc_curve(all_labels, all_prob[1], 'Random Forest')
    plot_roc_curve(all_labels, all_prob[3], 'AdaBoost')

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
    # load data
    filename = [filename_rna, filename_pc]
    # convert data
    rna, pc = data_convert(filename)

    # training model
    # trainingModel(rna,transformer_rna,"rna")
    # trainingModel(pc,transformer_pc,"pc")

    # prepare data
    dataset = "RPI448"
    rnaT, pcT, labels = preprocess(dataset)
    # rnaT = np.array(pd.read_csv("transformedPC.csv"))
    # pcT = np.array(pd.read_csv("transformedRNA.csv"))
    # print(labels)

    # predict
    X_data = np.concatenate((rnaT, pcT), axis=1)
    y = np.array(labels, dtype=int)
    # print(y)
    Lattention(X_data, y)
