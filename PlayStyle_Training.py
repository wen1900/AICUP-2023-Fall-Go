import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Softmax, BatchNormalization, Dropout, Add, MaxPool2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers

import numpy as np
from sklearn.model_selection import train_test_split

tf.__version__

seed = 10
np.random.seed(seed)

df = open('./CSVs/play_style_train.csv').read().splitlines()
games = [i.split(',',2)[-1] for i in df]
game_styles = [int(i.split(',',2)[-2]) for i in df]

chars = 'abcdefghijklmnopqrs'
coordinates = {k:v for v,k in enumerate(chars)}
coordinates

def prepare_input(moves):
    x = np.zeros((19,19,8))
    for i, move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        x[row,column,0] = 1
        if i == len(moves) - 2:
            x[row,column,2] = 1
            x[row,column,6] = 1
        elif i == len(moves) - 3:
            x[row,column,3] = 1
            x[row,column,6] = 1
        elif i == len(moves) - 4:
            x[row,column,4] = 1
            x[row,column,6] = 1
        elif i < len(moves) - 4:
            x[row,column,5] = 1  
    if moves:
        lase_move_color = color
        x[row,column,1] = 1
        x[row,column,6] = 1
    for move in moves:
        if move[0] == lase_move_color:
            column = coordinates[move[2]]
            row = coordinates[move[3]]
            x[row,column,7] = 1         
    
    return x

# Check how many samples can be obtained
n_games = 0
for game in games:
    n_games += 1
print(f"Total Games: {n_games}")

x = []
for game in games:
    moves_list = game.split(',')
    x.append(prepare_input(moves_list))
x = np.array(x)
y = np.array(game_styles)-1

np.bincount(y)
y_hot = tf.one_hot(y, depth=3)

x_train, x_val, y_train, y_val = train_test_split(x, y_hot.numpy(), test_size=0.30)

def create_model():
    inputs = Input(shape=(19, 19, 8))
    outputs = Conv2D(kernel_size=3, filters=64, padding='same', activation='relu')(inputs)
    outputs = Conv2D(kernel_size=3, filters=32, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=16, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=8, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=4, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=2, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=4, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=8, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=16, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=32, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=64, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=1, padding='same', activation='relu')(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(128, activation='relu')(outputs)
    outputs = Dense(3, activation='softmax', )(outputs)
    model = Model(inputs, outputs)
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
#model.load_weights('./weight/model_playstyle.h5')
model.summary()

history = model.fit(
    x = x_train, 
    y = y_train,
    batch_size = 1024,
    epochs = 100,
    validation_data=(x_val, y_val),
)

model.save('./weight/model_playstyle.h5')