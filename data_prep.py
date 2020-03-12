def process_data(file_in, save_name):
    import pandas as pd
    import numpy as np
    import librosa


    from librosa import load as ld
    from librosa import amplitude_to_db as amp
    from librosa import stft

    df = pd.read_csv(file_in)

    X = []

    for i in range(len(df)):
        y, sr = ld(df['File'][i], sr=None)
        D = amp(stft(y))
        X.append(D.T)

    df['X'] = X
    df.to_csv(f'/home/houston/Desktop/General_Assembly/CapStone/Phase 2/X_path/{save_name}.csv')
    
    del df, D, y, sr