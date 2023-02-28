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
models = [
    'ResNet',
    'AlexNet',
    'ShuffleNetV2',
    'WideResNet',
    'MobileNet2',
    'VGG2']


def metric_DMD(known, novel):
    """ Based on https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py
        mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    """

    # tp, fp
    known = np.asarray(known)
    novel = np.asarray(novel)
    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[l + 1:] = tp[l]
            fp[l + 1:] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[l + 1:] = np.arange(tp[l] - 1, -1, -1)
            fp[l + 1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l + 1] = tp[l]
                fp[l + 1] = fp[l] - 1
            else:
                k += 1
                tp[l + 1] = tp[l] - 1
                fp[l + 1] = fp[l]

    # TNR
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    TNR = 1. - fp[tpr95_pos] / num_n

    # AUROC
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    AUROC = -np.trapz(1. - fpr, tpr)

    # DTACC
    DTACC = .5 * (tp / tp[0] + 1. - fp / fp[0]).max()

    # AUIN
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp / denom, [0.]])
    AUIN = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    AUOUT = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])

    result = [AUROC, (AUIN + AUOUT) / 2, DTACC, TNR]
    return result


def load_raw(fe, model, ood, data):
    path = "../../generete_features_new/features/" + fe + \
        "/" + model + "/scores/" + ood + "_" + data + ".npy"
    if ood == "KNN" and model == "cifar10_densenet":
        path = "../../generete_features_new/features/" + fe + \
            "/" + model + "/scores_knn/" + ood + "_" + data + ".npy"
    x = np.load(path)
    return x


def get_indexes(n, rate=0.2):
    indexes = np.asarray(range(n))
    np.random.shuffle(indexes)
    train_ind = indexes[:int(n * rate)]
    test_ind = indexes[int(n * rate):]
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    return test_ind, train_ind


def get_data_one(data, indexes):
    return np.transpose(np.asarray([x[indexes] for x in data]))


def get_data(ind, out, in_indexes, out_indexes):

    x_in = get_data_one(ind, in_indexes)
    x_out = get_data_one(out, out_indexes)

    X = np.concatenate((x_in, x_out))
    y = np.concatenate((np.ones(x_out.shape[0]), np.zeros(x_in.shape[0])))
    return X, y


def test_one(fe, model, ood, ind, outd, rate):

    x_out = load_raw(fe, model, ood, outd)

    n = len(x_out)
    indexes_out, _ = get_indexes(n, rate)

    n_in = len(x_in)
    _, indexes_in = get_indexes(n_in, len(indexes_out) / n_in)
    y_in = x_in[indexes_in]
    y_out = x_out[indexes_out]

    return metric_DMD(y_in, y_out)


def test_nn(model="cifar10_densenet", name="noise", fes=["gap", "crow", "gap_all", "scda", "gmp"], oods=[
            "Mahalanobis", "lof_euclidean", "MaxSoftmax"], rate=0.2):
    X_in = []
    X_out = []
    for ood in oods:
        for fe in fes:
            X_in.append(load_raw(fe, model, ood, test_name))
            X_out.append(load_raw(fe, model, ood, name))

    n = len(X_out[0])
    test_ind_out, train_ind_out = get_indexes(n, rate)

    indexes = np.asarray(range(len(X_in[0])))
    np.random.shuffle(indexes)
    test_ind_in = indexes[test_ind_out]
    train_ind_in = indexes[train_ind_out]

    X_train, Y_train = get_data(X_in, X_out, train_ind_in, train_ind_out)
    scalers = [ECDF(x[train_ind_in]) for x in X_in]

    X_train = np.transpose(np.asarray(
        [scalers[i](x) for i, x in enumerate(np.transpose(X_train))]))
    regr = MLPRegressor(random_state=np.random.randint(
        10000), alpha=0.2).fit(X_train, Y_train)

    x_in = get_data_one(X_in, test_ind_in)
    x_in = np.asarray([scalers[i](x)
                      for i, x in enumerate(np.transpose(x_in))])
    y_in_mlp = regr.predict(np.transpose(x_in))

    x_out = get_data_one(X_out, test_ind_out)
    x_out = np.asarray([scalers[i](x)
                       for i, x in enumerate(np.transpose(x_out))])
    y_out_mlp = regr.predict(np.transpose(x_out))

    result = metric_DMD(y_in_mlp, y_out_mlp)

    return result


def read_data(model, name):
    X_in = []
    X_out = []
    oods = ["Mahalanobis", "lof_euclidean", "MaxSoftmax"]
    fes = ["gap", "crow", "gap_all", "scda", "gmp"]

    for ood in oods:
        for fe in fes:
            X_in.append(load_raw(fe, model, ood, test_name))
            X_out.append(load_raw(fe, model, ood, name))
    return X_in, X_out


def multi_test(model, name, rate):
    metrics = {}
    results = []

    for i in range(number_test):
        results.append(test_nn(model, name, ["gap", "crow", "gap_all", "scda", "gmp"], oods=[
                       "Mahalanobis", "lof_euclidean", "MaxSoftmax"], rate=rate))
    results = np.asarray(results)
    resp = [np.mean(results, axis=0).tolist(),
            np.std(results, axis=0).tolist()]
    print("  wnn_all", resp)
    metrics["wnn_all"] = resp

    results = []
    for i in range(number_test):
        results.append(test_nn(model, name, ["gap", "crow", "gap_all", "scda", "gmp"], oods=[
                       "Mahalanobis", "MaxSoftmax"], rate=rate))
    results = np.asarray(results)
    resp = [np.mean(results, axis=0).tolist(),
            np.std(results, axis=0).tolist()]
    print("  wnn_mm", resp)
    metrics["wnn_mm"] = resp

    results = []
    for i in range(number_test):
        results.append(test_nn(model, name, ["gap"], oods=[
                       "Mahalanobis", "lof_euclidean", "MaxSoftmax"], rate=rate))
    results = np.asarray(results)
    resp = [np.mean(results, axis=0).tolist(),
            np.std(results, axis=0).tolist()]
    print("  wnn_gap", resp)
    metrics["wnn_gap"] = resp

    for ood in ["Mahalanobis", "lof_euclidean", "MaxSoftmax"]:
        results = []
        for i in range(number_test):
            results.append(test_one("gap", model, ood, test_name, name, rate))
        results = np.asarray(results)
        resp = [np.mean(results, axis=0).tolist(),
                np.std(results, axis=0).tolist()]
        print("  " + ood, resp)
        metrics[ood] = resp

    return metrics


def attacks_cw(results, datas, rate):
    for model in models:
        X_in, X_out = read_data(model, "CW_hard")
        X_train, Y_train = get_data(X_in, X_out, list(range(len(X_in[0]))))
        scalers = [ECDF(x) for x in X_in]
        _results = {d: []for d in datas}
        X_train = np.transpose(np.asarray(
            [scalers[i](x) for i, x in enumerate(np.transpose(X_train))]))
        for i in range(number_test):
            regr = MLPRegressor(random_state=np.random.randint(
                10000), alpha=0.2).fit(X_train, Y_train)
            for d in datas:
                try:
                    X_in_d, X_out_d = read_data(model, d)
                except BaseException:
                    continue
                x_in = np.asarray([scalers[i](x)
                                  for i, x in enumerate(X_in_d)])
                y_in_mlp = regr.predict(np.transpose(x_in))
                x_out = np.asarray([scalers[i](x)
                                   for i, x in enumerate(X_out_d)])
                y_out_mlp = regr.predict(np.transpose(x_out))
                result = metric_DMD(y_in_mlp, y_out_mlp)
                _results[d].append(result)
        for d in datas:
            resp = [np.mean(_results[d], axis=0).tolist(),
                    np.std(_results[d], axis=0).tolist()]
            print(model, d, "  wnn_fgsm", resp)
            if not d in results:
                results[d] = {}
            if not model in results[d]:
                results[d][model] = {}
            results[d][model]["wnn_cw"] = resp


def test_ood():
    menu = "OoD"
    test_name = "test"
    r = 0.2
    datas = ["svhn", "cifar100"]
    path = "ood"

    results = {}
    for name in datas:
        results[name] = {}
        print(r, menu, ": ", name)
        for model in models:
            print("  Model: ", model)
            try:
                resp = multi_test(model, name, rate=r)
                results[name][model] = resp
            except FileNotFoundError as e:
                print("results not avaliable " + str(e))
    with open("../results/" + path + ".json", "wt") as f:
        json.dump(results, f)


def test_attacks():
    menu = "Attacks"
    test_name = "test"
    datas = [
        "CW_hard",
        "DeepFool_hard",
        "FGSM_hard",
        "OnePixel_hard",
        "PGD_hard",
        "Square_hard"]
    a_models = ["cifar10_resnet"]
    path = "attacks"
    r = 0.2

    results = {}
    for name in datas:
        results[name] = {}
        print(r, menu, ": ", name)
        for model in a_models:
            print("  Model: ", model)
            try:
                resp = multi_test(model, name, rate=r)
                results[name][model] = resp
            except FileNotFoundError as e:
                print("results no avaliable for: " + e.filename)
    with open("../results/" + path + ".json", "wt") as f:
        json.dump(results, f)


def test_attacks_knn():
    menu = "Attacks"
    test_name = "test"
    rate = 0.2
    datas = [
        "CW_hard",
        "DeepFool_hard",
        "FGSM_hard",
        "OnePixel_hard",
        "PGD_hard",
        "Square_hard"]
    a_models = ["cifar10_resnet"]

    path = "attacks_knn"

    fresults = {}
    for name in datas:
        fresults[name] = {}
        print(rate, menu, ": ", name)
        for model in a_models:
            print("  Model: ", model)
            results = []
            try:
                for i in range(number_test):
                    results.append(
                        test_one(
                            "gap",
                            model,
                            "KNN",
                            test_name,
                            name,
                            rate))
                results = np.asarray(results)
                resp = [np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist()]
                print("  KNN", resp)
                fresults[name][model] = {}
                fresults[name][model]["knn"] = resp
            except FileNotFoundError as e:
                print("results not avaliable " + str(e))
    with open("../results/" + path + ".json", "wt") as f:
        json.dump(fresults, f)


def test_knn():
    menu = "KNN"
    test_name = "test"
    rate = 0.2
    datas = ["svhn", "cifar100"]
    path = "knn"

    fresults = {}
    for name in datas:
        fresults[name] = {}
        print(rate, menu, ": ", name)
        for model in models:
            print("  Model: ", model)
            results = []
            try:
                for i in range(number_test):
                    results.append(
                        test_one(
                            "gap",
                            model,
                            "KNN",
                            test_name,
                            name,
                            rate))
                results = np.asarray(results)
                resp = [np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist()]
                print("  KNN", resp)
                fresults[name][model] = {}
                fresults[name][model]["knn"] = resp
            except FileNotFoundError as e:
                print("results not avaliable " + str(e))
    with open("../results/" + path + ".json", "wt") as f:
        json.dump(fresults, f)


np.random.seed(0)
random.seed(0)
test_knn()


np.random.seed(0)
random.seed(0)
test_ood()

np.random.seed(0)
random.seed(0)
test_attacks()

np.random.seed(0)
random.seed(0)
test_attacks_knn()
