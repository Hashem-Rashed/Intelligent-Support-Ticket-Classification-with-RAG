def drop_unnecessary_columns(data, cols_to_drop):
    """
    Drop columns that are not required for preprocessing.
    """
    return data.drop(columns=cols_to_drop)
