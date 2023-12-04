import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Softmax, MaxPool2D, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet101V2
import numpy as np
from sklearn.model_selection import train_test_split
import os

tf.__version__

seed = 10
np.random.seed(seed)

checkpoint_path = "./checkpoints/model_dan_tutorial-{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

df = open('./CSVs/dan_train.csv').read().splitlines()
games = [i.split(',',2)[-1] for i in df] 

chars = 'abcdefghijklmnopqrs'
coordinates = {k:v for v,k in enumerate(chars)}
chartonumbers = {k:v for k,v in enumerate(chars)}
coordinates

def prepare_input(moves):
    x = np.zeros((19,19,7))
    for i, move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column,0] = 1
            x[row,column,2] = 1
        if color == 'W':
            x[row,column,1] = 1
            x[row,column,2] = 1
        if i == len(moves) - 2:
            x[row,column,4] = 1
        elif i == len(moves) - 3:
            x[row,column,5] = 1
        elif i == len(moves) - 4:
            x[row,column,6] = 1
    if moves:
        last_move_column = coordinates[moves[-1][2]]
        last_move_row = coordinates[moves[-1][3]]
        x[row,column,3] = 1
    x[:,:,2] = np.where(x[:,:,2] == 0, 1, 0)
    return x

def prepare_label(move):
    column = coordinates[move[2]]
    row = coordinates[move[3]]
    return column*19+row

# Check how many samples can be obtained
n_games = 0
n_moves = 0
for game in games:
    n_games += 1
    moves_list = game.split(',')
    for move in moves_list:
        n_moves += 1
print(f"Total Games: {n_games}, Total Moves: {n_moves}")

# BEGIN: 5d4f3g7hj8kl
def data_generator(games):
    for game in games:
        game_str = game.decode('utf-8') # decode bytes to string
        moves_list = game_str.split(',')
        for count, move in enumerate(moves_list):
            x = prepare_input(moves_list[:count])
            y = prepare_label(moves_list[count])
            yield np.array(x), tf.one_hot(np.array(y), depth=19*19)

games_train, games_val = train_test_split(games, test_size=0.1, random_state=seed)

batch_size = 1024
train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    args=[games_train],
    output_types=(tf.float32, tf.int32),
    output_shapes=((19, 19, 7), tf.TensorShape((19*19,))))
train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    data_generator,
    args=[games_val],
    output_types=(tf.float32, tf.int32),
    output_shapes=((19, 19, 7), tf.TensorShape((19*19,))))
val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# Create the model
def residual_block(x, filters, kernel_size):
    y = Conv2D(kernel_size=kernel_size,
               filters=filters,
               padding='same')(x)
    y = ReLU()(y)
    y = Conv2D(kernel_size=kernel_size,
               filters=filters,
               padding='same')(y)
    output = Add()([x,y])
    output = ReLU()(output)
    return output

def go_res():
    inputs = Input(shape=(19, 19, 7))
    conv5x5 = Conv2D(kernel_size=5,
                     filters=256,
                     padding="same",
                     name='conv5x5')(inputs)
    conv1x1 = Conv2D(kernel_size=1,
                     filters=256,
                     padding="same",
                     name='conv1x1')(inputs)
    outputs = Add()([conv5x5, conv1x1])
    outputs = ReLU()(outputs)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = Conv2D(kernel_size=3,
                     filters=1,
                     padding="same")(outputs)
    outputs = ReLU()(outputs)
    outputs = Flatten()(outputs)
    outputs = Softmax()(outputs)
    model = Model(inputs, outputs)
    
    opt = Adam(learning_rate=0.00005)
    model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_path, 
                                verbose=0, 
                                save_weights_only=True,
                                save_freq=1)
#create model
model = go_res()
#load weights
#model.load_weights('./weight/model_dan_tutorial.h5')

model.summary()
#save weights
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(
    train_dataset, 
    epochs = 3,
    validation_data=val_dataset,
    verbose=1,
    callbacks=[cp_callback],
    #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
)

model.save('./weight/model_dan_tutorial.h5')

