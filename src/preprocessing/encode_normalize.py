from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def encode_categorical(data, label_cols):
    """
    Encode categorical columns using LabelEncoder.
    Returns encoded data and dictionary of encoders.
    """
    encoders = {}
    for col in label_cols:
        enc = LabelEncoder()
        data[col + "_encoded"] = enc.fit_transform(data[col])
        encoders[col] = enc
    data = data.drop(columns=label_cols)
    return data, encoders

def normalize_numerical(data, num_cols):
    """
    Normalize numerical columns using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data, scaler
