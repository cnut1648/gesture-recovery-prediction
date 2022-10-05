import pickle



for k in range(4):
    with open(f"/home/jiashu/seq/artifact/transformer/k{k}.pkl", "rb") as f:
        attns = pickle.load(f)
    
    print("++++++++++")
    print(k)
    print("++++++++++")

    for layer, layer_attns in attns.items():
        for case, d in layer_attns.items():
            L, R = d["occl_scores"]
            if (L > 0).sum() > 0:
                print(case, "L")
            if (R > 0).sum() > 0:
                print(case, "R")
        del attns[layer][case]["orig_logits"]

    with open(f"/home/jiashu/seq/artifact/transformer/{k}.pkl", "wb") as f:
        pickle.dump(attns, f)
