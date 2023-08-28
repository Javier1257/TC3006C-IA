import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

errors = []

# HYPOTHESIS CALCULATION FUNCTION
def hypothesis(params, x) :
    h = params[0]

    for i in range(len(x)) :
        h = h + (params[i + 1] * x[i])
    h = 1 / (1 + np.exp((-1 * h)))      # <--- SIGMOID (ACTIVATION FUNCTION)
    return h

# GRADIENT DESCENT CALCULATION FUNCTION
def gradient_descent(params, LR, x, y) :
    new_params = []
    sum = 0

    for i in range(len(x)) :
        sum = sum + (hypothesis(params, x.iloc[i]) - y[i])
    
    new_theta = params[0] - ((LR / len(x)) * sum)
    new_params.append(new_theta)
    
    sum = 0

    for i in range(1, (len(params))) :
        for j in range(len(x)) :
            sum = sum + ((hypothesis(params, x.iloc[j]) - y[j]) * x.iloc[j][i - 1])
        new_theta = params[i] - ((LR / len(x)) * sum)
        new_params.append(new_theta)
        sum = 0

    return new_params

# ERROR CALCULATION FUNCTION (CROSS-ENTROPY LOSS)
def cross_entropy_loss(params, x, y) :
    global errors

    error_i = 0
    error_sum = 0
    mean_error = 0

    for i in range(len(x)) :
        y_hat = hypothesis(params, x.iloc[i])

        if y_hat == 1 :
            y_hat = 0.99999
        
        if y_hat == 0 :
            y_hat = 0.00001

        if y[i] == 1 :
            error_i = -1 * np.log(y_hat)
        else :
            error_i = -1 * np.log(1 - y_hat)

        error_sum = error_sum + error_i
    
    mean_error = error_sum / len(x)
    errors.append(mean_error)

    return mean_error

# RANDOM PARAMS GENERATOR FUNCTION
def params_generator(n) :
    random_params = []

    for i in range(n + 1) :
        random_params.append(random.random())
    
    return random_params

# LOADING DATASET
columns = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = pd.read_csv('C:/Users/paco1/Downloads/titanic/train.csv', names = columns)
print('ORIGINAL DATASET')
print(df.head())
print('')

# PRINTING IN THE TERMINLA THE GENERAL INFORMATION OF THE DATASET
print(df.info())
print('')

# CLEANING THE DATASET FOR THE MODEL TRAINING
df['Age'] = df['Age'].replace(np.nan, 30)       # REPLACING THE NAN VALUES FOR THE AGE MEAN GOT WITH df['Age'].mean()

CLEANED_DF = pd.get_dummies(df, columns = ['Sex'])        # USING THE PANDAS FUNCTION get_dummies TO MAKE ONE-HOT ENCODING
CLEANED_DF.drop(['PassengerID', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)   # DROPPING THE COLUMNS THAT WON'T BE USE

CLEANED_DF.rename(columns = {'Sex_female' : 'Female', 'Sex_male' : 'Male'}, inplace = True)   # RENAMING COLUMNS

print('CLEANED DATASET')
print(CLEANED_DF.head())
print('')

# DIVIDING THE COLUMNS BETWEEN THE INPUTS AND OUTPUT
X = CLEANED_DF[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Female', 'Male']]
Y = CLEANED_DF['Survived']

# DIVIDING THE DATASET BETWEEN THE TEST AND TRAIN DATAFRAMES
train_x = X[:713]           # THE DATASET IS DIVIDED IN 80% TRAIN, 20% TEST
train_y = Y[:713]
test_x = X[713:]
test_y = Y[713:]

# GENERATING RANDOM PARAMS
params = params_generator(len(train_x.columns))

# LEARNING RATE
ALFA = 0.001

epochs = 0

# TRAINING LOOP
while True :
    print('#################### EPOCH ', epochs, ' ####################')
    old_params = params
    print('OLD PARAMS: ')
    print(old_params)
    params = gradient_descent(params, ALFA, train_x, train_y)
    print('NEW PARAMS: ')
    print(params)
    print('')
    loss = cross_entropy_loss(params, train_x, train_y)
    print('LOSS: ', loss)
    print('')
    epochs += 1
    if (epochs == 1000) or (loss == 0.001) or (old_params == params):
        print('#################### FINAL PARAMS ####################')
        print(params)
        print('')
        break

# TESTING LOOP
preds = []
i = 0

for i in range(len(test_x)) :
    y_hat = hypothesis(params, test_x.iloc[i])
    if y_hat >= 0.6 :
        y_hat = 1
    else :
        y_hat = 0
    preds.append(y_hat)

# CONFUSION MATRIX
TP = 0      # TRUE POSITIVE PREDICTION
TN = 0      # TRUE NEGATIVE PREDICTION
FP = 0      # FALSE POSITIVE PREDICTION
FN = 0      # FALSE NEGATIVE PREDICTION

i = 0

for i in range(len(preds)) :
    if (preds[i] == 1) and (test_y.iloc[i] == 1) :
        TP += 1
    elif (preds[i] == 0) and (test_y.iloc[i] == 0) :
        TN += 1
    elif (preds[i] == 1) and (test_y.iloc[i] == 0) :
        FP += 1
    elif (preds[i] == 0) and (test_y.iloc[i] == 1) :
        FN += 0

accuracy = (TP + TN) / len(preds)
precision = TP / (TP + FP)

print('ACCURACY: ', accuracy)
print('PRECISION: ', precision)

plt.plot(errors)
plt.show()




