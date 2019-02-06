# Import data
import pandas as pd
import numpy as np 
import pandas as pd
import collections, ast
from sklearn.model_selection import train_test_split
from functools import reduce
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
data1 = pd.read_csv("battle_data1.csv")
data2 = pd.read_csv("battle_data2.csv")
data3 = pd.read_csv("battle_data3.csv")
data4 = pd.read_csv("battle_data4.csv")
data5 = pd.read_csv("battle_data5.csv")
data6 = pd.read_csv("battle_data6.csv")
data7 = pd.read_csv("battle_data7.csv")
pokemon_data = pd.read_csv("pokemon_data.csv")
battle_data = pd.concat([data1, data2, data3, data4, data5, 
                        data6, data7]).reset_index(drop=True)

# Preprocess
types = [ast.literal_eval(x) for x in pokemon_data['Types']]
pokemon_data['Types'] = types
abilities = [ast.literal_eval(x) for x in pokemon_data['Abilities']]
pokemon_data['Abilities'] = abilities
# Encode types and abilities
types = pokemon_data['Types'].apply(collections.Counter)
types = pd.DataFrame.from_records(types).fillna(value=0)
abilities = pokemon_data['Abilities'].apply(collections.Counter)
abilities = pd.DataFrame.from_records(abilities).fillna(value=0)
# Drop abilities that have no effect in battle
abilities = abilities.drop(['Honey Gather', 'Illuminate',
                              'Run Away'], axis=1)
# Join the encoded types and abilities to the dataset
new_pokemon_data = pd.concat([pokemon_data, types, abilities], axis=1)\
                    .reset_index(drop=True).drop(['Types', 'Abilities'], axis=1)
# Remove pairs where two Pokemons of the same species face against each other
rows = battle_data['Pokemon 1'] == battle_data['Pokemon 2']
battle_data = battle_data[-rows]

# Get half of the data randomly and switches the two columns
n = len(battle_data.index)
idx = np.random.choice(n, int(n/2.), replace=False)
battle_data.iloc[idx,:] = battle_data.iloc[idx,:].rename( 
            columns={'Pokemon 1':'Pokemon 2','Pokemon 2':'Pokemon 1'})

# Label the data
con1 = battle_data['Winner'] == battle_data['Pokemon 1']
con2 = battle_data['Winner'] == battle_data['Pokemon 2']
battle_data.loc[:,'Winner'][con1] = 0
battle_data.loc[:,'Winner'][con2] = 1

X = battle_data.loc[:,('Pokemon 1', 'Pokemon 2')]
Y = battle_data.loc[:,('Winner')]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, 
                                                Y, test_size=0.35)

def isin_row(a, b, cols=None):
    """
    Function that checks if a Series is in a data frame
    """
    cols = cols or a.columns
    return reduce(lambda x, y:x&y, [a[f].isin([b[f]]) for f in cols])
X_test = X_test.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)
drops = []
for index, row in X_test.iterrows():
    rename = {"Pokemon 1": "Pokemon 2", "Pokemon 2": "Pokemon 1"}
    win = Y_test.iloc[index,]
    reverse = row.rename(index=rename)
    if any(isin_row(X_train, row)) or any(isin_row(X_train, reverse)):
        X_train = X_train.append(row)
        Y_train = Y_train.append(pd.Series(win))
        drops.append(index)
X_test = X_test.drop(drops, axis=0)
Y_test = Y_test.drop(drops, axis=0)
for _ in [X_train, X_test, Y_train, Y_test]:
    _ = _.reset_index(drop=True)

def look_up(name):
    index = new_pokemon_data['Name'] == name
    try:
        pokemon = new_pokemon_data[index].iloc[0,2:]
    except:
        # Some pokemon have different forms that only have minor 
        # differences such as appearance and hence is not recorded
        # in the pokemon data. Search using the original form instead.
        name = re.search('(.*?)-', name).group(1)
        index = new_pokemon_data['Name'] == name
        pokemon = new_pokemon_data[index].iloc[0,2:]
    return pokemon

def get_data(X):    
    result = []
    pokemon_1 = look_up(X.iloc[0])
    pokemon_2 = look_up(X.iloc[1])
    return pd.concat([pokemon_1, pokemon_2], axis=0)


train_x = []
test_x = []
for index, row in X_train.iterrows():
    train_x.append(get_data(row).values)c
for index, row in X_test.iterrows():
    test_x.append(get_data(row).values)
Y_train = np.array(Y_train.astype('int'))
Y_test = np.array(Y_test.astype('int'))
train_x = np.array(train_x)
test_x = np.array(test_x)

# Random forest
clf = RandomForestClassifier(n_estimators=100, oob_score=True)
clf = clf.fit(train_x, Y_train)
pred = model.predict(test_x)
train_score = clf.oob_score_
test_score = accuracy_score(Y_test,pred)

# Deep learning
model = Sequential()
model.add(Dense(36, input_dim=508, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, Y_train, validation_split=0.2, epochs=100, verbose=0)

train_score = model.evaluate(train_x, Y_train)[1]
test_score = model.evaluate(test_x, Y_test)[1]
pred = model.predict(test_x)