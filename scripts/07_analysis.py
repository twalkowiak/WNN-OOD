import json


def analyse(path, metric=0):
    success = 0
    failer = 0
    with open(path, "rt") as f:
        data = json.load(f)
    for key1 in data:
        for key2 in data[key1]:
            if key1 == key2:
                continue
            for model in data[key1][key2]:
                results = data[key1][key2][model]
                w = results["wnn_all"][0][metric]
                ok = True
                for method in results:
                    if method == "wnn_all":
                        continue
                    m = results[method][0][metric]
                    if m > w:
                        print(key1, key2, model, method, (m - w) / w * 100)
                        ok = False
                        break
                if ok:
                    success += 1
                else:
                    failer += 1
    print(success, failer, success + failer)


def analyse2(path, metric=0):
    success = 0
    failer = 0
    with open(path, "rt") as f:
        data = json.load(f)
    for key1 in data:
        for key2 in data[key1]:
            if key1 != key2:
                continue
            for model in data[key1][key2]:
                results = data[key1][key2][model]
                w = results["wnn_all"][0][metric]
                ok = True
                for method in results:
                    if method == "wnn_all":
                        continue
                    m = results[method][0][metric]
                    if m > w:
                        print(key1, key2, model, method, (m - w) / w * 100)
                        print(results)
                        ok = False
                        break
                if ok:
                    success += 1
                else:
                    failer += 1
    print(success, failer, success + failer)


def analyse3(path, path2, metric=0):
    print("KNN")
    success = 0
    failer = 0
    with open(path, "rt") as f:
        data = json.load(f)
    with open(path2, "rt") as f:
        data2 = json.load(f)
    for key1 in data:
        for key2 in data[key1]:
            if key1 != key2:
                continue
            for model in data[key1][key2]:
                results = data[key1][key2][model]
                w = results["wnn_all"][0][metric]
                k = data2[key1][model]['knn'][0][metric]
                if w > k:
                    success += 1
                else:
                    failer += 1
                    print(key1, model, (k - w) / w * 100)
    print(success, failer, success + failer)


analyse2("../results/trans2_ood.json")
analyse3("../results/trans2_ood.json", "../results/knn2.json")
