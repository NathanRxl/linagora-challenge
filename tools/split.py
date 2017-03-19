def split_xy(df):
    return df.drop("recipients", axis=1), df[["recipients"]]
