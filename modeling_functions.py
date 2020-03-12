### GENERATOR FUNCTIONS


def X_gen(i):
    """
    Reads in a soundfile and returns a transposed STFT matrix
    """
    from librosa import load as ld
    from librosa import amplitude_to_db as amp
    from librosa import stft  
    
    
    y, sr = ld(i, sr=None)
    D = amp(stft(y))
    return D.T

def pad_X(i):
    """
    Reads in a transposed sound matrix and pads the length with 0's until 800 units long
    """
    import numpy as np
    a = np.zeros((800, 1025))

    a[:i.shape[0], :i.shape[1]] = i

    return a

def gen_y_feed(i):
    import pandas as pd
    temp_df = pd.DataFrame(i)

    temp_df = temp_df.shift().fillna(0)

    array = temp_df.to_numpy()

    return array

def reshape_to_vect(i):
    """
    Converts list of strings into a 1D vector
    """
    import numpy as np
    return np.array(i).reshape(-1, 1)

def densify(i):
    import pandas as pd
    i = pd.DataFrame(i.todense())
    return i.to_numpy()
    
def generator(sample_path):
    """
    Reads in Soundfile and converts to STFT and pads length with 0's until (800 or ~16 seconds)
    Reads in character lists converts list string into array and onehot_encodes it. 
    
    Returns Pandas DataFrame with Sound Array as X and one_hot character array as y
    
    sample_paths * Required: List of file paths for model to read data in

    
    """
    import numpy as np
    import pandas as pd
    from librosa import load as ld
    from librosa import amplitude_to_db as amp
    from librosa import stft  
    from sklearn.preprocessing import OneHotEncoder
    from ast import literal_eval

    
    one_hot = OneHotEncoder()
    
    one_hot.fit(np.array([i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ\'"^ ']).reshape(-1, 1))
    


    df = pd.read_csv(sample_path)
    df['X'] = df['File'].apply(X_gen).apply(pad_X)
    df['y'] = df['Character_Labels'].apply(literal_eval).apply(reshape_to_vect).apply(one_hot.transform).apply(densify)
    df['y_in'] = df['y'].apply(gen_y_feed)
    
    return df[['X', 'y_in', 'y']]




### MODEL FUNCTIONS


def build_model(input_dim, model_path):
    import tensorflow.keras as keras
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dropout, Input, BatchNormalization, TimeDistributed, Dense, Activation
    from tensorflow.keras.models import Model, save_model
    import tensorflow as tf

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(input_dim))
    encoder = LSTM(512, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, 30))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(30, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    save_model(model, model_path, save_format='h5')
    
    return model


def model_train_one_it(df, i, model_path):
    import numpy as np
    import time
    from tensorflow.keras.models import Model, load_model, save_model
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    
    X_inpre = df['X'][i]
    y_inpre = df['y_in'][i]
    y_outpre = df['y'][i]

    characters = y_outpre.shape[0]
    letters = y_outpre.shape[1]

    X_in = np.stack([X_inpre for x in range(characters)])
    y_in = np.array(y_inpre).reshape(characters, 1, letters)
    y_out= np.array(y_outpre).reshape(characters, 1, letters)

    model = load_model(model_path)
    
    history = model.fit([X_in, y_in], 
                        y_out, batch_size=32, max_queue_size=256, use_multiprocessing=True)
    save_model(model, model_path, save_format='h5')
    
    return history
    
    
def model_train_one_df(gen_path, model_path):    
    from tensorflow.keras.models import Model, load_model, save_model

    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    
    df = generator(gen_path)
    model = load_model(model_path)
    
    for i in range(len(df)):
        model_train_one_it(df, i, model)
        
    save_model(model, model_path, save_format='h5')
    time.sleep(0.25)
    
    
def eval_model():
    return loss



