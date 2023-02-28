import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import random
from ecdf import ECDF
import json


test_name = "test"
number_test = 10


def metric_DMD(known, novel):
    """ Based on https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py
        mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    """

    #tp, fp
    known = np.asarray(known)
    novel = np.asarray(novel)
    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    # TNR
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    TNR = 1. - fp[tpr95_pos] / num_n

    # AUROC
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    AUROC = -np.trapz(1.-fpr, tpr)

    # DTACC
    DTACC = .5 * (tp/tp[0] + 1.-fp/fp[0]).max()

    # AUIN
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    AUIN = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    AUOUT = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    result = [AUROC, (AUIN+AUOUT)/2, DTACC, TNR]
    return result


def load_raw(fe, model, ood, data):
    path = "../features/"+fe+"/"+model+"/scores/"+ood+"_"+data+".npy"
    x = np.load(path)
    return x


def get_indexes(n, rate=0.2):
    indexes = np.asarray(range(n))
    np.random.shuffle(indexes)
    train_ind = indexes[:int(n*rate)]
    test_ind = indexes[int(n*rate):]
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    return test_ind, train_ind


def get_data_one(data, indexes):
    return np.transpose(np.asarray([x[indexes] for x in data]))


def get_data(ind, out, indexes):

    x_in = get_data_one(ind, indexes)
    x_out = get_data_one(out, indexes)

    X = np.concatenate((x_in, x_out))
    y = np.concatenate((np.ones(x_out.shape[0]), np.zeros(x_in.shape[0])))
    #y = y.reshape((X.shape[0], 1))
    return X, y


def test_nn(model="cifar10_densenet", name="noise", fes=["gap", "crow", "gap_all", "scda", "gmp"], oods=["Mahalanobis", "lof_euclidean", "MaxSoftmax"], rate=0.2):
    X_in = []
    X_out = []
    for ood in oods:
        for fe in fes:
            X_in.append(load_raw(fe, model, ood, test_name))
            X_out.append(load_raw(fe, model, ood, name))

    n = len(X_in[0])
    test_ind, train_ind = get_indexes(n, rate)

    X_train, Y_train = get_data(X_in, X_out, train_ind)
    scalers = [ECDF(x[train_ind]) for x in X_in]

    X_train = np.transpose(np.asarray(
        [scalers[i](x) for i, x in enumerate(np.transpose(X_train))]))
    regr = MLPRegressor(random_state=np.random.randint(
        10000), alpha=0.2).fit(X_train, Y_train)

    x_in = get_data_one(X_in, test_ind)
    x_in = np.asarray([scalers[i](x)
                      for i, x in enumerate(np.transpose(x_in))])
    #y_in = np.sum(x_in,axis=0)
    y_in_mlp = regr.predict(np.transpose(x_in))

    x_out = get_data_one(X_out, test_ind)
    x_out = np.asarray([scalers[i](x)
                       for i, x in enumerate(np.transpose(x_out))])
    y_out_mlp = regr.predict(np.transpose(x_out))
    result = metric_DMD(y_in_mlp, y_out_mlp)

    return result





def size_of_validation():
    model = "WideResNet"
    ood = "cifar100"
    test_name = "test"
    fes = ["gap", "crow", "gap_all", "scda", "gmp"]
    oods = ["Mahalanobis", "lof_euclidean", "MaxSoftmax"]
    results = {}
    metric = 0
    number_test = 50
    for r in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        _results = []
        for i in range(number_test):
            _results.append(test_nn(model, ood, fes, oods, rate=r))
        _results = np.asarray(_results)
        resp = [np.mean(_results, axis=0).tolist(),
                np.std(_results, axis=0).tolist()]
        results[r] = [resp[0][metric], resp[1][metric]]
        print(results)

    path = "size_of_validation_ood"
    with open("../results/"+path+".json", "wt") as f:
        json.dump(results, f)

    path = "size_of_validation_ood"
    with open("../results/"+path+".tex", "wt") as f:
        for r in results:
            f.write("(")
            f.write(str(r))
            f.write(",")
            f.write(str(results[r][0]))
            f.write(")")
        f.write("\r\n")
        for r in results:
            f.write("(")
            f.write(str(r))
            f.write(",")
            f.write(str(results[r][0]+3*results[r][1]))
            f.write(")")
        f.write("\r\n")
        for r in results:
            f.write("(")
            f.write(str(r))
            f.write(",")
            f.write(str(results[r][0]-3*results[r][1]))
            f.write(")")
        f.write("\r\n")


def model_print(model):
    if model == "resnet":
        return "ResNet"
    if model == "wrn":
        return "\\begin{tabular}{l}Wide\\\\ResNet\\end{tabular}"
    if model == "cifar10_densenet":
        return "DenseNet"

    return model


def ood_print(ood):
    if ood == "svhn":
        return "SVHN"
    if ood == "cifar100":
        return "CIFAR-100"
    return ood


def method_influence():
    test_name = "test"
    path = "ensamble_influence"
    fes = ["gap", "crow", "gap_all", "scda", "gmp"]
    datas = ["svhn", "cifar100"]
    oods = ["Mahalanobis", "lof_euclidean", "MaxSoftmax"]
    models = ["cifar10_densenet",'ResNet','AlexNet','ShuffleNetV2','WideResNet','MobileNetV2','VGG16']
    types = [["Mahalanobis"], ["Mahalanobis", "MaxSoftmax"],
             ["Mahalanobis", "lof_euclidean", "MaxSoftmax"]]

    metric = 0

    with open("../results/"+path+".tex", "wt") as f:

        for model in models:
            first_ood = True
            for d in datas:
                if first_ood:
                    f.write(
                        "\\multirow{2}{*}{"+model_print(model.replace("cifar10_", ""))+"}")
                    first_ood = False
                f.write("&"+ood_print(d))
                for t in types:
                    f.write("&")
                    _results = []
                    for i in range(number_test):
                        _results.append(
                            test_nn(model, d, ["gap", "crow", "gap_all", "scda", "gmp"], t, rate=0.2))

                    _results = np.asarray(_results)
                    resp = [np.mean(_results, axis=0).tolist(),
                            np.std(_results, axis=0).tolist()]
                    results = [resp[0][metric], resp[1][metric]]
                    f.write("%3.1f" % (int(results[0]*10000)/100))
                    f.write("$\\pm$%3.2f" % (int(results[1]*10000)/100))
                    print(model, d, t, results)
                f.write("\\\\\n")
                print("__________")
            f.write("\midrule\n")


np.random.seed(0)
random.seed(0)
#Figure 1
size_of_validation()

np.random.seed(0)
random.seed(0)
#Table 5
method_influence()
