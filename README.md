# Pokemon battle winner predictor

In this project, I implement a model that takes in two Pokemons as input (each Pokemon has n features, hence the input would have 2n features) and classifies which Pokemon will win the match. The battle data is parsed from publicly available Pokemon battles on https://replay.pokemonshowdown.com , a Pokemon battle simulator maintained by Smogon, one of the biggest Pokemon communities. The data for each Pokemon is parsed from the Pokedex for Generation VII from Serebii.net (https://www.serebii.net/pokedex-sm/). See appendix attached at the end for the code used to parse the data sets.

## Variables

For each Pokemon, the variables used in the classifiers are:
- Types: A Pokemon can have one or two types, which would affect the damage it takes from attacks, as well as the damage of its own attacks to its opponent.
- Abilities: A species of Pokemon can have from one to three abilities, each of which have different effects in battle such as decreasing on opponent's stat, or increase the damage if it uses a certain attack, etc.
- Base hit point (HP): Decide how much damage the Pokemon can take
- Base attack: Decide how much damage the Pokemon's physical attack does
- Base defense: Decide how much damage the Pokemon take from physical attack
- Base special attack: Decide how much damage the Pokemon's special attack does
- Base defense: Decide how much damage the Pokemon take from special attack
- Base speed: Decide the speed of the pokemon, i.e. which Pokemon attacks first

## Models and parameters

For this problem, I used two models: a random forest and a feedforward neural network. Originally, I planned to use an support vector machine (SVM) model as well, but even after using principal components analysis (PCA) to project the part of the data (the encoded abilities) down to a lower dimension, the SVM model takes way too long to fit even with a linear kernel (did not terminate after 30 minutes). For other models, using the PCA projection slightly improved the time to fit but not the accuracy of the classifier. Hence, the SVM model was removed and I revert back to using the original data without PCA projection.

Explanation for the choice of model parameters:
- Breiman and Cutler (n.d.) remarked that random forest does not overfit. However, Oshiro, Perez, and Baranauskas (2012) found that beyond a certain threshold there is no significant gain in performance but only an increase in the computational cost and recommend that a random forest should have a number of trees between 64 - 128 trees. Here, I choose to build a random forest with 100 trees.

- For the neural network model, there is no reliable way to tell how many hidden layers and neurons for each layer are necessary. Heaton (2008) remarked that two hidden layers can represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions. Hence I chose to go with two hidden layers. Different sources use different variables (size of input and output layer, size of training set, etc.) to estimate the number of neurons needed (Heaton, 2008; Kaufmann, 1993). If there are many redundant nodes, the network would overfit. After experimenting with different values, I went with 36 neurons for the first hidden layer and 12 neurons for the second layer.

- The SVM model took longer to run partly because to avoid overfitting, we need to perform a grid search with cross validation to find the best parameters, which significantly increase computational cost and runtime. Random forest and deep learning have the advantage that they do not need cross validation over various set of parameters to find the optimal one (though deep learning does require some trial and error).

## Preprocessing data

For the Pokemon data set, I will encode types and abilities, which are multi-valued categorical variables. I will also drop three abilities that have no effect in battle.

For the battle data set, I will label the data based on the winner. The label is 0 if the winner is the first Pokemon and 1 if the winner is the second Pokemon. However, because of the way the data is parsed, in the majority of cases, the first Pokemon is the attacker that knocks out the second pokemon. This would lead to an imbalanced data set, which is not optimal since decision trees are sensitive to class imbalance and I am using random forest. While there are multiple methods to solve this problem such as signing weights to each class or using different sampling methods. However, in this case, we can solve this by simply switching the first Pokemon with the second Pokemon before labeling. Hence, I will randomly switch the position of the two Pokemons in half of the data.

Because there are multiple data of the same pair of Pokemons, I will make sure that pairs that are in the training set would not appear in the test set. Splitting 65% the dataset into training and the other 35% into test, I then move all the pairs in the test set that appears in the training set into the training set. I end up with roughly 90% of the original dataset (~37000 data points) in the training set.

## To be improved

- Our data did not take into account the moves of the Pokemon. In competitive playing, Pokemon needs to carry moves that are super effective against their threads, and popular offensive Pokemons have good type coverage with their moves. On the other hand, defensive Pokemons need effective healing move to keep themselves up longer. Moves play a big part in battling, but because each Pokemon has a huge movepool, can only have four moves, and utilizes different moves based on their stats and types, it is hard to record this information for each Pokemon.

- Similarly, there is no data on what items the Pokemon were holding. It is hard to incorporate held items into the model because any Pokemon can hold any items, and the items are not necessarily recorded in the battle log. Held items also have a significant role in the battle, with effects ranging from healing, doing extra damage, to increasing the stats or change the environment of the battle.

- I recorded data throughout the battle, which means a Pokemon that is worn down after taking down three other Pokemons is treated the same as a healthy Pokemon newly sent into battle. Hence, a Pokemon that normally would have won a match up would be recorded as losing in our record. We would expect the accuracy to improve if we only include matches where both Pokemons are healthy (or of equal health condition).

- Most importantly, the outcome of a battle is not deterministic, hence the same match up may have different winners in re-matches. Because our data record each match as one data point, the same input can have different outputs, so we cannot achieve perfect accuracy on the test set. To see how well our models actually performed, we need to examine how much of the input have different outputs and derive the best possible accuracy we can achieve with this test set.
