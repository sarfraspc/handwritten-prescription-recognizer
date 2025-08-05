def encode_labels(texts, char_to_idx):
    encoded = []
    for text in texts:
        encoded.append([char_to_idx[char] for char in text])
    return encoded


def add_encoded_labels(df, label_col, char_to_idx):
    df['encoded_label'] = encode_labels(df[label_col].astype(str).tolist(), char_to_idx)
    return df
