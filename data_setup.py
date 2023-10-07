import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import re

def get_dataframe(file_path):
    """Get dataset from json file."""
    df = pd.read_json(file_path).T
    df.index.name = "id"
    return df

def count_words(text):
    txt = re.sub(r"['\",\.\?:\-!\n\(\)]", " ", str(text))
    txt_list = [word.strip() for word in txt.split()]
    len_text = len(txt_list)
    return len_text

def find_length(data):
    columns = ["context", "claim", "evidence"]
    len_dict = {}
    for column in columns:
        length = [count_words(text) for text in df[column]]
        print(f"Column '{column}':")
        print(f"\tLongest {column}: {max(length)} words | id: {df.index[np.argmax(length)]} | index: {np.argmax(length)}")
        print(f"\tShortest {column}: {min(length)} words | id: {df.index[np.argmin(length)]} | index: {np.argmin(length)}\n")
        len_dict.update({column: length})
    return len_dict