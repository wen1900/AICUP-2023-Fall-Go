import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

chars = 'abcdefghijklmnopqrs'
coordinates = {k:v for v,k in enumerate(chars)}
chartonumbers = {k:v for k,v in enumerate(chars)}

#create submission file-----------------------------------------------------------------------------------------

def prepare_input_for_dan_kyu_models(moves):
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

def prepare_input_for_playstyle_model(moves):
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
        last_move_column = column #coordinates[moves[-1][2]]
        last_move_row = row #coordinates[moves[-1][3]]
        lase_move_color = color
        x[row,column,1] = 1
        x[row,column,6] = 1
    for move in moves:
        if move[0] == lase_move_color:
            column = coordinates[move[2]]
            row = coordinates[move[3]]
            x[row,column,7] = 1
    return x

def number_to_char(number):
    number_1, number_2 = divmod(number, 19)
    return chartonumbers[number_1] + chartonumbers[number_2]

def top_5_preds_with_chars(predictions):
    resulting_preds_numbers_sort = [(np.argsort(prediction)[-5:][::-1]) for prediction in predictions]
    resulting_preds_chars = np.vectorize(number_to_char)(resulting_preds_numbers_sort)
    return resulting_preds_chars
#---------------------------------------------------------- public_submission_template ----------------------------------------------------------

# save predictions for Kyu-----------------------------------------------------------------------------------------
# Load your own model. Here we use the baseline model
model = load_model('./weight/model_kyu_tutorial.h5')

# Load the corresponding dataset
df = open('./CSVs/kyu_test_public.csv').read().splitlines()
games_id = [i.split(',',2)[0] for i in df]
games = [i.split(',',2)[-1] for i in df]

x_testing = []

for game in games:
    moves_list = game.split(',')
    x_testing.append(prepare_input_for_dan_kyu_models(moves_list))

x_testing = np.array(x_testing)
predictions = model.predict(x_testing)
prediction_chars = top_5_preds_with_chars(predictions)

# Save results to public_submission_template.csv
with open('./submit/pri_sub_.csv','a') as f:
    for index in range(len(prediction_chars)):
        answer_row = games_id[index] + ',' + ','.join(prediction_chars[index]) + '\n'
        f.write(answer_row)

# save predictions for Dan-----------------------------------------------------------------------------------------
# Load your own model. Here we use the baseline model
model = load_model('./weight/model_dan_tutorial.h5')

# Load the corresponding dataset
df = open('./CSVs/dan_test_public.csv').read().splitlines()
games_id = [i.split(',',2)[0] for i in df]
games = [i.split(',',2)[-1] for i in df]

x_testing = []

for game in games:
    moves_list = game.split(',')
    x_testing.append(prepare_input_for_dan_kyu_models(moves_list))

x_testing = np.array(x_testing)
predictions = model.predict(x_testing)
prediction_chars = top_5_preds_with_chars(predictions)

# Save results to public_submission_template.csv
with open('./submit/pri_sub_.csv','a') as f:
    for index in range(len(prediction_chars)):
        answer_row = games_id[index] + ',' + ','.join(prediction_chars[index]) + '\n'
        f.write(answer_row)

# save predictions for Playstyle-----------------------------------------------------------------------------------------
# Load your own model. Here we use the baseline model
model = load_model('./weight/model_playstyle.h5')

# Load the corresponding dataset
df = open('./CSVs/play_style_test_public.csv').read().splitlines()
games_id = [i.split(',',2)[0] for i in df]
games = [i.split(',',2)[-1] for i in df]

x_testing = []

for game in games:
    moves_list = game.split(',')
    x_testing.append(prepare_input_for_playstyle_model(moves_list))

x_testing = np.array(x_testing)
predictions = model.predict(x_testing)
prediction_numbers = np.argmax(predictions, axis=1)

with open('./submit/pri_sub_.csv','a') as f:
    for index, number in enumerate(prediction_numbers):
        answer_row = games_id[index] + ',' + str(number+1) + '\n'
        f.write(answer_row)

#---------------------------------------------------------- private_submission_template ----------------------------------------------------------

# save predictions for Kyu-----------------------------------------------------------------------------------------
# Load your own model. Here we use the baseline model
model = load_model('./weight/model_kyu_tutorial.h5')

# Load the corresponding dataset
df = open('./CSVs/kyu_test_private.csv').read().splitlines()
games_id = [i.split(',',2)[0] for i in df]
games = [i.split(',',2)[-1] for i in df]

x_testing = []

for game in games:
    moves_list = game.split(',')
    x_testing.append(prepare_input_for_dan_kyu_models(moves_list))

x_testing = np.array(x_testing)
predictions = model.predict(x_testing)
prediction_chars = top_5_preds_with_chars(predictions)

# Save results to public_submission_template.csv
with open('./submit/pri_sub_.csv','a') as f:
    for index in range(len(prediction_chars)):
        answer_row = games_id[index] + ',' + ','.join(prediction_chars[index]) + '\n'
        f.write(answer_row)
# save predictions for Dan-----------------------------------------------------------------------------------------
# Load your own model. Here we use the baseline model
model = load_model('./weight/model_dan_tutorial.h5')

# Load the corresponding dataset
df = open('./CSVs/dan_test_private.csv').read().splitlines()
games_id = [i.split(',',2)[0] for i in df]
games = [i.split(',',2)[-1] for i in df]

x_testing = []

for game in games:
    moves_list = game.split(',')
    x_testing.append(prepare_input_for_dan_kyu_models(moves_list))

x_testing = np.array(x_testing)
predictions = model.predict(x_testing)
prediction_chars = top_5_preds_with_chars(predictions)

# Save results to public_submission_template.csv
with open('./submit/pri_sub_.csv','a') as f:
    for index in range(len(prediction_chars)):
        answer_row = games_id[index] + ',' + ','.join(prediction_chars[index]) + '\n'
        f.write(answer_row)

# save predictions for Playstyle-----------------------------------------------------------------------------------------
# Load your own model. Here we use the baseline model
model = load_model('./weight/model_playstyle.h5')

# Load the corresponding dataset
df = open('./CSVs/play_style_test_private.csv').read().splitlines()
games_id = [i.split(',',2)[0] for i in df]
games = [i.split(',',2)[-1] for i in df]

x_testing = []

for game in games:
    moves_list = game.split(',')
    x_testing.append(prepare_input_for_playstyle_model(moves_list))

x_testing = np.array(x_testing)
predictions = model.predict(x_testing)
prediction_numbers = np.argmax(predictions, axis=1)

with open('./submit/pri_sub_.csv','a') as f:
    for index, number in enumerate(prediction_numbers):
        answer_row = games_id[index] + ',' + str(number+1) + '\n'
        f.write(answer_row)