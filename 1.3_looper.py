import librosa
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa.display as ld
import os
import yaml

df = pd.read_csv('./data/file_dictionary.csv')

with open('./data/looper.yaml', 'r') as yams:
    storps = yaml.full_load(yams)
    
start = storps['start']
end = storps['end']

df =  df.loc[start:end, :]

dic = {}
dic['start'] = (start + 5)
dic['end'] = (end + 5)

with open('./data/looper.yaml', 'w') as yams:
    yaml.dump(dic, yams)
    


def make_jpg(data, path, title, new_file_name):
    fig = plt.figure(figsize=[1,1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    ld.specshow(D, y_axis='linear')
    try:
        os.makedirs(f'data/audio_images/{path}/{title}/')
    except:
        pass
    plt.savefig(f'data/audio_images/{path}/{title}/{new_file_name}', dpi=500, bbox_inches='tight',pad_inches=0)
    plt.close()
    del D, fig    
def generate_data(language, title, file):
    if language == 1:
        path = 'English/english'
    else:
        path = 'German/german'
    try:
        filename = f'./data/{path}/{title}/{file}'
        data, samplerate = sf.read(filename, dtype='float32')
        sf.close()
    except:
        pass
    try:
        data = data[:, 1]
    except:
        pass
    return data, samplerate, path
def generate_time_slice(sec, data1, samplerate):
    data2 = data1[(sec*15)*samplerate : ((sec+1)*15)*samplerate]
    data2 = np.asfortranarray(data2)
    return data2
def generate_path(file, sec):
    if file.endswith('o.ogg'):
        marker = 0
    elif file.endswith('1.ogg'):
        marker = 1
    else:
        marker = 2
    new_file_name = f'{marker}_{sec*15}-{sec*15 + 15}_audio_to_img.jpg'
    return new_file_name

for j, row in df.iterrows():

    file = df['Filename'][j]
    language = df['Language'][j]
    title = df['Title'][j]

    try:
        data, samplerate, path = generate_data(language, title, file)

        try:
            count = 0
            for i in range(20):
                data2 = generate_time_slice(i, data, samplerate)
                new_file_name = generate_path(file, i)

                if len(data2/samplerate) > 10:
                    make_jpg(data2, path, title, new_file_name)
                    count += 1
                else:
                    pass
                del data2
            print(f'Count={count}, File: {title}, new Index: df.loc[{j+1}:, :]')
            
        except:
            print(f'File: {title}, No usable data')
        del data, samplerate, path
    except:
        print(f'Data for {title} corrupted or unusuable')
        
