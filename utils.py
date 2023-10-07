import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_text_distribution(len_dict):
    """Plot the length distribution for each column"""
    fig = plt.figure(figsize=(15, 4))
    rows, cols = 1, 3
    bins = [5, 10, 20]
    fontdict = {"fontsize": 10, "color": "blue"}
    for i, column in enumerate(len_dict.keys()):
        fig.add_subplot(rows, cols, i+1)
        plt.hist(len_dict[column], bins=30, color="aqua", edgecolor="blue")
        plt.title(f"Length of {column} distribution", fontdict=fontdict)
        plt.xlabel("Length (number of words)", fontdict=fontdict)
    
def plot_class_distribution(df):
    """Return the distribution of verdict and domain."""
    columns = ["verdict", "domain"]
    fig = plt.figure(figsize=(12, 4))
    rows, cols = 1, 2
    for i, column in enumerate(columns):
        dist = df[column].value_counts()
        fontdict = {"fontsize": 10, "color": "blue"}
        fig.add_subplot(rows, cols, i+1)
        plt.bar(dist.index, dist, color="aqua", edgecolor="blue")
        plt.title(f"{column} distribution", fontdict=fontdict)
        plt.ylim(0, np.max(dist) + 0.2*np.max(dist))
        ax = plt.gca()
        for container in ax.containers:
            plt.bar_label(container)
        
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted", labels=np.unique(preds))
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", labels=np.unique(preds))
    recall = recall_score(labels, preds, average="weighted", labels=np.unique(preds))

    return {"accuracy": acc, "precision" : prec, "recall" : recall, "f1": f1}

def preprocess_text(text: str) -> str:
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text

def strict_accuracy(gt: dict, pred: dict) -> dict:
    gt_verdict = gt["verdict"]
    pred_verdict = pred["verdict"]
    gt_evidence = gt["evidence"]
    pred_evidence = pred["evidence"]

    gt_evidence = preprocess_text(gt_evidence)
    pred_evidence = preprocess_text(pred_evidence)

    acc = int(gt_verdict == pred_verdict)
    acc_1 = int(gt_evidence == pred_evidence)
    strict_acc = acc * acc_1

    return {
        "strict_acc": strict_acc,
        "acc": acc,
        "acc@1": acc_1,
    }