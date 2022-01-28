## IA - Les méthodes d'apprentissage supervisees



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import time
from functools import reduce
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
import warnings
warnings.filterwarnings('ignore')
```


### Cette partie du Tp a pour but de mettre en oeuvre un ***Anti-Spam*** en réalisant des modéles de classifieurs binaires construits à l'aide d'un ***DataSet*** basé sur les activités normales et anormales d'une messagerie électronique

### Prétraitement de données


```python
# ====== Préparation des données =======
# chargement des données
creation = pd.read_csv("Uses_Cases/Spam/Spamcreation.csv")
pred = pd.read_csv("Uses_Cases/Spam/Spamprediction.csv",skipinitialspace=True)
heads = list(pred.keys())
pred = pd.read_csv("Uses_Cases/Spam/Spamprediction.csv",skipinitialspace=True,usecols=heads)

# On peut utiliser les fonctions de pandas pour veridier les données
#pred.describe() #récapitulative sur les données
#pred.tail()  # la dernière ligne
pred.head()  # la première ligne

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_freq_make</th>
      <th>word_freq_address</th>
      <th>word_freq_all</th>
      <th>word_freq_3d</th>
      <th>word_freq_our</th>
      <th>word_freq_over</th>
      <th>word_freq_remove</th>
      <th>word_freq_internet</th>
      <th>word_freq_order</th>
      <th>word_freq_mail</th>
      <th>...</th>
      <th>char_freq_%3B</th>
      <th>char_freq_%28</th>
      <th>char_freq_%5B</th>
      <th>char_freq_%21</th>
      <th>char_freq_%24</th>
      <th>char_freq_%23</th>
      <th>capital_run_length_average</th>
      <th>capital_run_length_longest</th>
      <th>capital_run_length_total</th>
      <th>Spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.70</td>
      <td>0.00</td>
      <td>0.70</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.105</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2.342</td>
      <td>47</td>
      <td>89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.45</td>
      <td>0.91</td>
      <td>0.45</td>
      <td>0.91</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.254</td>
      <td>0.0</td>
      <td>0.063</td>
      <td>0.127</td>
      <td>0.000</td>
      <td>4.735</td>
      <td>46</td>
      <td>161</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.168</td>
      <td>0.0</td>
      <td>0.112</td>
      <td>0.000</td>
      <td>0.056</td>
      <td>2.933</td>
      <td>23</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.0</td>
      <td>0.14</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.056</td>
      <td>0.094</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.028</td>
      <td>0.000</td>
      <td>2.394</td>
      <td>24</td>
      <td>881</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
freqEmailYes=[]
freqEmailNo=[]
freqYes=[]
freqNo=[]

for i in range(len(pred.Spam)):
    if pred.Spam[i]==0:
        freqEmailYes.append(pred.word_freq_mail[i])
        freqYes.append(pred.word_freq_internet[i])
    else:
        freqEmailNo.append(pred.word_freq_mail[i])
        freqNo.append(pred.word_freq_internet[i])

```


```python
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(freqEmailYes, freqYes, color="red")
ax[1].scatter(freqEmailNo, freqNo, color="blue", edgecolors="black", lw=0.5)

ax[0].set_title("Spam yes")
ax[1].set_title("Spam not")
ax[0].set_xlabel(" Email")
ax[1].set_xlabel(" Email ")
ax[0].set_ylabel(" Internet ")
ax[1].set_ylabel(" Internet ")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("L'analyse des frequences des spam des emails provenant de l'internet ")

plt.show()
```



![png](/IA/output_20_0.png)




```python
fig, ax = plt.subplots(1, 2)

ax[0].hist(freqYes, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(freqNo, 10, facecolor='blue', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Spam yes ")
ax[0].set_ylabel("Frequency des emails ")
ax[1].set_xlabel("Spam not ")
ax[1].set_ylabel("Frequence des emails ")
fig.suptitle("L'analyse de la frequence des emails provenat de l'internet")

plt.show()
```



![png](/IA/output_21_0.png)



### Conclusion du pretraitement  des données

D'après l'analyse des données on voit qu'un ***spam*** est define selon une frequence des mots des variables du Dataset.

Donc notre réseau de neurones va analyser ces frequences pour predire si on est dans un cas de spam ou non.

### 1 - Construire un modèle M1 basé sur une architecture à base de neurones sans couches cachées


```python


# coder les labels dans le format one-hot
y=to_categorical(pred.Spam)
#print(np.sum(y,axis=0)) verifier si on a la bonne distribution des classes

#isoler les descripteurs
X= pred.iloc[:,0:pred.shape[1]-1]



# on subdivise 274 en test et 1000 en apprentissage

Xtrain,Xtest,Ytrain,Ytest = model_selection.train_test_split(X,y,test_size=274,random_state=100)



# standardisation des descripteurs
cr = StandardScaler(with_mean=True,with_std=True)

# calcul des paramètres + centrage réduction du train set

XtrainSd = cr.fit_transform(Xtrain)


# architecture du réseau: sans couche cachée de 512 neuronnes et sans dropout.

network = Sequential()
# Ajouter une couche d'entrée
network.add(layers.Dense(12,activation='relu',input_shape=(57,)))
# Ajouter une couche de sortie
network.add(layers.Dense(2,activation='softmax'))





# compilation - algorithme d'optimisation
network.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])



# Lancement de l'apprentissage
network.fit(XtrainSd, Ytrain, epochs=10, batch_size=10, verbose = 1)


#Xtrain.shape,Ytrain.shape,Xtest.shape,Ytest.shape
# Evaluation du modèle
test_loss, test_accM1 = network.evaluate(Xtest, Ytest)

# Affichage du résultat
print ('Acccuracy = ', test_accM1)
```

    Epoch 1/10
    100/100 [==============================] - 0s 482us/step - loss: 0.6837 - accuracy: 0.6530
    Epoch 2/10
    100/100 [==============================] - 0s 494us/step - loss: 0.4920 - accuracy: 0.8590
    Epoch 3/10
    100/100 [==============================] - 0s 588us/step - loss: 0.3768 - accuracy: 0.8960
    Epoch 4/10
    100/100 [==============================] - 0s 573us/step - loss: 0.3049 - accuracy: 0.9100
    Epoch 5/10
    100/100 [==============================] - 0s 592us/step - loss: 0.2603 - accuracy: 0.9210
    Epoch 6/10
    100/100 [==============================] - 0s 574us/step - loss: 0.2299 - accuracy: 0.9230
    Epoch 7/10
    100/100 [==============================] - 0s 589us/step - loss: 0.2071 - accuracy: 0.9350
    Epoch 8/10
    100/100 [==============================] - 0s 599us/step - loss: 0.1895 - accuracy: 0.9380
    Epoch 9/10
    100/100 [==============================] - 0s 589us/step - loss: 0.1750 - accuracy: 0.9420
    Epoch 10/10
    100/100 [==============================] - 0s 604us/step - loss: 0.1631 - accuracy: 0.9430
    9/9 [==============================] - 0s 692us/step - loss: 33.3502 - accuracy: 0.4599
    Acccuracy =  0.45985400676727295


### 2 - Construire un modèle M2 basé sur une architecture à base de neurones avec couches cachées


```python


# coder les labels dans le format one-hot
y=to_categorical(pred.Spam)
#y=list(pred.Spam)
#print(np.sum(y,axis=0)) verifier si on a la bonne distribution des classes

#isoler les descripteurs
X= pred.iloc[:,0:pred.shape[1]-1]

# on subdivise 274 en test et 1000 en apprentissage

Xtrain,Xtest,Ytrain,Ytest = model_selection.train_test_split(X,y,test_size=274,random_state=100)



# standardisation des descripteurs
cr = StandardScaler(with_mean=True,with_std=True)

# calcul des paramètres + centrage réduction du train set

XtrainSd = cr.fit_transform(Xtrain)


# architecture du réseau: sans couche cachée de 512 neuronnes et sans dropout.

network = Sequential()
# Ajouter une couche d'entrée
network.add(layers.Dense(12,activation='relu',input_shape=(57,)))
# Ajouter une couche cachée
network.add(layers.Dense(8, activation='relu'))
# Ajouter une couche de sortie
network.add(layers.Dense(2,activation='softmax'))



# compilation - algorithme d'optimisation
network.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])



# Lancement de l'apprentissage
network.fit(XtrainSd, Ytrain, epochs=10, batch_size=10, verbose = 1)


#Xtrain.shape,Ytrain.shape,Xtest.shape,Ytest.shape
# Evaluation du modèle
test_loss, test_accM2 = network.evaluate(Xtest, Ytest)

# Affichage du résultat
print ('Acccuracy = ', test_accM2)
```

    Epoch 1/10
    100/100 [==============================] - 0s 495us/step - loss: 0.6275 - accuracy: 0.6560
    Epoch 2/10
    100/100 [==============================] - 0s 477us/step - loss: 0.4788 - accuracy: 0.8330
    Epoch 3/10
    100/100 [==============================] - 0s 477us/step - loss: 0.3453 - accuracy: 0.8940
    Epoch 4/10
    100/100 [==============================] - 0s 487us/step - loss: 0.2639 - accuracy: 0.9140
    Epoch 5/10
    100/100 [==============================] - 0s 485us/step - loss: 0.2182 - accuracy: 0.9290
    Epoch 6/10
    100/100 [==============================] - 0s 483us/step - loss: 0.1906 - accuracy: 0.9400
    Epoch 7/10
    100/100 [==============================] - 0s 479us/step - loss: 0.1709 - accuracy: 0.9420
    Epoch 8/10
    100/100 [==============================] - 0s 491us/step - loss: 0.1557 - accuracy: 0.9470
    Epoch 9/10
    100/100 [==============================] - 0s 486us/step - loss: 0.1430 - accuracy: 0.9540
    Epoch 10/10
    100/100 [==============================] - 0s 479us/step - loss: 0.1328 - accuracy: 0.9570
    9/9 [==============================] - 0s 544us/step - loss: 28.2859 - accuracy: 0.5219
    Acccuracy =  0.5218977928161621
