import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    df = pd.read_csv('github-combined-file-no-symbols-rare-clean.csv')[['snippet', 'language']]
    df = df[df['snippet'].notna()]
    df['snippet'] = df['snippet'].apply(lambda x: x[:4096])
    le = LabelEncoder()
    df['language'] = le.fit_transform(df.language)
    mapping_dict = {
        0:2,
        1:3,
        2:4,
        3:5,
        4:6,
        5:7,
        6:8,
        7:9,
        8:10,
        9:11,
        10:100,
        11:12,
        12:13,
        13:14,
        14:15,
        15:21,
        16:17,
        17:26,
        18:16,
        19:18,
        20:19,
        21:22,
        22:26,
        23:100,
        24:28,
        25:39,
        26:44,
        27:50,
        28:48,
        29:49,
        30:100,
        31:59,
        32:0,
        33:70,
        34:73,
        35:78,
        36:79,
        37:83,
        38:100,
        39:89,
        40:99,
    }
    df['language'] = df['language'].replace(mapping_dict)
    df.index = df['language']
    df = df.drop(columns=['language'])
    df_cp = df.groupby('language')['snippet'].apply(''.join).reset_index()[['snippet', 'language']].T
    df_cp.to_csv('data/train_data_labeled.csv', index=False)
    df_5 = df.groupby(by=['language']).sample(5)
    df_5 = df_5.groupby('language')['snippet'].apply(''.join).reset_index()[['snippet', 'language']].T
    df_5.to_csv('data/train_data_5exp_labeled.csv', index=False)
    labels = pd.DataFrame(le.classes_).rename(columns={0: 'language'})
    labels.T.to_csv('data/labels.csv', index=False)
