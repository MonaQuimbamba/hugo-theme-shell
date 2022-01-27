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

### L'algorithme KppV

Ce programme applique la méthode de discrimination de kppv sur un ensemble d'individu élément de R^2.



#### Les paramètres :

- La variable test :  
    Un tableau qui doit contenir les différents individus à classer rangés par colonne. Le nombre de ligne est 2 et le nombre de colonne est n.


- La variable apprentissage :
    Un tableau qui doit contenir les différents individus de l'ensemble d'apprentissage rangés par colonne. Le nombre de ligne est 2 et le nombre de colonne est m.



- La variable oracle :
     Vecteur qui indique la classification de l'ensemble d'apprentissage oracle[i] indique le nombre de la classe de l'individu apprentissage[:i].


- La variable K :
   Indique le nombre de voisins utilisés dans l'algorithme.

#### Le résultat :

- La variable clas :
    Vecteur qui indique le résultat de l'algorithme de la discrimination clas[i] indique le numéro de la classe de l'individu x[:,i]





### L'exemple de l'exécution de l'algorithme

test = [ [3,2],[4,7],[8,9] ]

learn =[
     [ 0  2]
     [ 2  0]
     [ 2  6]
     [ 4  4]
     [ 2  4]
     [ 6  7]
     [10  4]
     [ 4  0]
     [ 4  2]
     [ 6  0]
     [ 6  2]
    ]

oracle= [0,0,1,1,0,1,0,0,0,1,1]

k=3

### on calcule distance-euclidean chaque élément de test avec l'ensemble des éléments learn

### Etape 1 :

#### 1 - calcul test[0]= [3,2]
    distance-euclidean [3 2]  et  [0 2]  est  3.0
    distance-euclidean [3 2]  et  [2 0]  est  2.23606797749979
    distance-euclidean [3 2]  et  [2 6]  est  4.123105625617661
    distance-euclidean [3 2]  et  [4 4]  est  2.23606797749979
    distance-euclidean [3 2]  et  [2 4]  est  2.23606797749979
    distance-euclidean [3 2]  et  [6 7]  est  5.830951894845301
    distance-euclidean [3 2]  et  [10  4]  est  7.280109889280518
    distance-euclidean [3 2]  et  [4 0]  est  2.23606797749979
    distance-euclidean [3 2]  et  [4 2]  est  1.0
    distance-euclidean [3 2]  et  [6 0]  est  3.605551275463989
    distance-euclidean [3 2]  et  [6 2]  est  3.0

#### 2 - calcul test[1]= [4,7]

    distance-euclidean [4 7]  et  [0 2]  est  6.4031242374328485
    distance-euclidean [4 7]  et  [2 0]  est  7.280109889280518
    distance-euclidean [4 7]  et  [2 6]  est  2.23606797749979
    distance-euclidean [4 7]  et  [4 4]  est  3.0
    distance-euclidean [4 7]  et  [2 4]  est  3.605551275463989
    distance-euclidean [4 7]  et  [6 7]  est  2.0
    distance-euclidean [4 7]  et  [10  4]  est  6.708203932499369
    distance-euclidean [4 7]  et  [4 0]  est  7.0
    distance-euclidean [4 7]  et  [4 2]  est  5.0
    distance-euclidean [4 7]  et  [6 0]  est  7.280109889280518
    distance-euclidean [4 7]  et  [6 2]  est  5.385164807134504

#### 1 - calcul test[2]= [8,9]

    distance-euclidean [8 9]  et  [0 2]  est  10.63014581273465
    distance-euclidean [8 9]  et  [2 0]  est  10.816653826391969
    distance-euclidean [8 9]  et  [2 6]  est  6.708203932499369
    distance-euclidean [8 9]  et  [4 4]  est  6.4031242374328485
    distance-euclidean [8 9]  et  [2 4]  est  7.810249675906654
    distance-euclidean [8 9]  et  [6 7]  est  2.8284271247461903
    distance-euclidean [8 9]  et  [10  4]  est  5.385164807134504
    distance-euclidean [8 9]  et  [4 0]  est  9.848857801796104
    distance-euclidean [8 9]  et  [4 2]  est  8.06225774829855
    distance-euclidean [8 9]  et  [6 0]  est  9.219544457292887
    distance-euclidean [8 9]  et  [6 2]  est  7.280109889280518

### Etape 2 :
#### On regarde les k distance minimales

- pour test[0]=[3,2] sont :

        *    1.O  avec [4 2] indice dans learn 8
        *    2.23 avec [2 0] indice dans learn 1
        *    2.23 avec [4 4] indice dans learn 3

- pour test[1]=[4,7] sont :


         * 2    avec  [6 7] indice dans learn 5
         * 2.23 avec  [2 6] indice dans learn 2
         * 3    avec  [4 4] indice dans learn 3

- pour test[2]=[8 9] sont :

        * 2.83 avec [6 7] indice dans learn  5
        * 5.38 avec [10 4] indice dans learn 6
        * 6.4 avec  [4 4] indice dans learn  3

### Etape 3 :
#### On regarder les classes/groupe dans des k distances minimales dans oracle

- pour ***test[0]=[3,2]***   

    * oracle[8] = 0
    * oracle[1] = 0
    * oracle[3] = 1



- pour ***test[1]=[4,7]***  

    * oracle[5]= 1
    * oracle[2]= 1
    * oracle[3]= 1




- pour ****test[2]=[8,9]***  

    * oracle[5] = 1
    * oracle[6] = 0
    * oracle[3] = 1

### Etape 4 :
#### on vote , on regarde parmis les classes/groupes ceux qui sont les plus nombreux


- pour ***test[0]=[3,2]*** on voit qu'il est de la classe 0



- pour ***test[1]=[4,7]*** on voit qu'il est de la classe 1



- pour ***test[2]=[8,9]*** on voit qu'il est de la classe 1





## L'implementation de l'algorithme KNNV en python


```python
def dist(x,y,dname='euclidean'):

    if (dname == 'manhattan') or (dname == 'cityblock'):
        d = np.sum(np.abs(x-y))
    elif dname == 'euclidean':
        d = np.sqrt(np.sum(np.abs(x-y)**2))
    elif (dname == 'chebychev') or (dname == 'chebyshev'):
        d = np.max(np.abs(x-y))
    elif dname == 'cosine':
        d = 1 - x.dot(y) / np.sqrt(x.dot(x)*y.dot(y))
    else:
        d = np.sqrt(np.sum(np.abs(x-y)**2))       
    return d
```


```python
def KNNV(test, learn, ylearn, K=1, dname='euclidean'):
    test_nb = test.shape[0]
    learn_nb = learn.shape[0]
    labels_id = np.unique(ylearn)
    labels_nb = len(labels_id)
    dist_to_learn = np.zeros((test_nb, learn_nb))
    votes = np.zeros((test_nb,labels_nb))
    for i in range(test_nb):
        for j in range(learn_nb):
            dist_to_learn[i,j] = dist(test[i,:],learn[j,:],dname)  # Etape 1
        KNN_index = np.argsort(dist_to_learn[i,:])[:K]             # Etape 2  argSort envoit les indices K des element d'un tableau
        KNN_y = ylearn[[KNN_index]]                                # Etape 3
        for j in range(labels_nb):
            votes[i,j] = len(np.argwhere(KNN_y==labels_id[j]))     # Etape 4 argWhere Trouvez les indices des éléments du tableau non nuls, regroupés par élément.
    ypred = np.argmax(votes,axis=1) # Returns the indices of the maximum values along an axis.
    return ypred

```


```python
def affiche_classeKKpV(x,clas,K):
    for k in range(0,K):
        ind=(clas==k)
        plt.plot(x[0,ind],x[1,ind],"o")
    plt.show()
```

## Jeu d'essaie KNNV


```python
# Données de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]  

data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))

mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]]  

data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))
data=np.concatenate((data1, data2), axis=1)

oracle=np.concatenate((np.zeros(128),np.ones(128)))

test1=np.transpose(np.random.multivariate_normal(mean1, cov1, 64))
test2=np.transpose(np.random.multivariate_normal(mean2, cov2,64))

test=np.concatenate((test1,test2), axis=1)
K=3
clas=KNNV(test.T,data.T,oracle,K)
affiche_classeKKpV(test,clas,2)
```



![png](/IA/output_9_0.png)



## Le Perceptron

## -  1 Mise en place d'un perceptron simple

Ce programme doit évaluer la sortie d'un perceptron simple ( 1 neurone ) pour une entrée élément de R^2.

### Les Paramètres :

    - La variable w contient les poids synaptiques du  neurones. C'est un vecteur à 3 lignes. La première ligne correspond au seuil.
    - La variable x contient l'éntrée du réseau de neurones. C'est un vecteur à 2 lignes.
    - La variable active indique la fonction d'activation utilisée.

    * Si active == 0 :
      σ(x) = sign(x)
    * Si active == 1 :
      σ(x) = tanh(x)

### Le Résultat :

    - La variable y est un scalaire correspondant à la sortie du neurone

## - 2 Etude de l'apprentissage


Ce programme retourne le poids de w obtenu par apprentissage selon la règle d'apprentissage utilisant la descente du gradient .


### Les paramètres :

 - La variable x contient l'ensemble d'apprentissage. C'est une matrice à 2 lignes et n colonnes.

 - La variable yd(i) indique la réponse désirée pour chaque élément x(:,i).

 yd est un vecteur de 1 ligne et n colonnes de valeurs +1 ou -1 (classification à 2 classes).

 - On suggère d'utiliser 100 itérations


### Le Résultat :

 - La variable w contient les poids synaptiques du neurone après apprentissage. C'est un vecteur à 3 lignes . La première ligne correspond au seuil.


 - La variable erreur contient l'erreur cumulée calculée pour le passage complet de l'ensemble d'apprentissage à savoir :

 ![image.png](/IA/attachment:image.png)


La variable erreur sera un vecteur de taille fixée par le nombre d'itération. Cela permettra de représenter l'évolution de l'erreur au cours des itérations de l'apprentissage.



### L'exemple de l'exécution de l'algorithme

|     Xt      | Yt           |
|:------------|:------------:|
|   [2,0]     |   1          |
|   [0,3]     |   0          |
|   [3,0]     |   0          |
|   [1,1]     |   1          |


* Simulation avec biais α = 0.1



* Initialisation w <- [0,0], b=0.5

### Pour chaque paire (Xt,Yt):

#### Etape 1 ***le perceptron***

    h(xt)=Threshold(z)

    où z= w*xt+b

    et Threshold(z)=1 si z >=0 et 0 sinon


    # pour l'algo en dessous on utilise la focntion activation
    # si active == 0 Threshold(z)=

Pour x1=[2,0]

    h(x1)= [0,0]*[2 0] + 0.5 = 0.5

    donc

    Threshold(0.5)= 1

#### Etape 2  mettre à jour le ***w*** et le ***b***

    - Si h(xt) = yt , on ne fait pas de mise à jour
    - Si h(xt) != yt , on fait la mise à jour
        *  w = w + α(y2 - h(x2)) x2
        *  b = b + α(y2 - h(x2)

puisque h(x1)=y1 , on ne fait pas de mise à jour

On continue la boucle

####  Pour x2=[0,3]

#### Etape 1 :

    h(x2)=[0,0]*[0,3]+0.5 = 1

#### Etape 2 :
     h(x2)!= y2

     - mise à jour :
     * w = [0,0] + 0.1 *(0-1) * [0,3] = [0,-0.3]
     * b = [0,0]+0.1*(0-1) = 0.4


On continue la boucle





## L'implementation de l'algorithme le Perceptron en python


```python
# Le coefficient d'apprentissage α sera égal à 0.1
def perceptron(x,w,active):
    z=np.dot(x,[w[0][1],w[0][2]]) + w[0][0]
    if active ==0:
        y= np.sign(z)
    else:
        y=np.tanh(z)
    return y

def apprentissage(x,yd,active):
    erreur=[]
    w=np.array([[0.5,0,0]])
    for _ in range(100):
        sum_erreur=0
        for i, x_i in enumerate(x):
            y=perceptron([ x_i[0],x_i[1] ],w,active) # Etape 1
            active=np.tanh(y)
            if(y!=yd[i]):  # Etape 2
                # mise à jour w
                tmp=[x_i[0],x_i[1]]
                val=0.1*(yd[i]-y)
                wtmp = np.add([w[0][1],w[0][2]],np.dot(val,tmp))
                w[0][1]=wtmp[0]
                w[0][2]=wtmp[1]

                # mise à jour b

                w[0][0]=w[0][0] + 0.1*( yd[i] -y)

                # calcul erreur

                sum_erreur+=(yd[i]-y)**2

        erreur.append(sum_erreur)


    return w,erreur

def affiche_classe(x,clas,K,w):
    t=[np.min(x[0,:]),np.max(x[0,:])]
    z=[(-w[0,0]-w[0,1]*np.min(x[0,:]))/w[0,2],(-w[0,0]-w[0,1]*np.max(x[0,:]))/w[0,2]]
    plt.plot(t,z);
    ind=(clas==-1)
    plt.plot(x[0,ind],x[1,ind],"o")
    ind=(clas==1)
    plt.plot(x[0,ind],x[1,ind],"o")
    plt.show()

```


```python
# Données de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]  #
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))
mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]]  #
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))
data=np.concatenate((data1, data2), axis=1)
oracle=np.concatenate((np.zeros(128)-1,np.ones(128)))
w,mdiff=apprentissage(data.T,oracle,1)

plt.plot(mdiff)
plt.show()
affiche_classe(data,oracle,2,w)
```



![png](/IA/output_15_0.png)





![png](/IA/output_15_1.png)



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


### 3 - Construire un modèle M3 basé sur un classifieur de type réseau de Bayes naif

###  L'exemple de l'exécution de l'algorithme



|Sexe         | Taille(cm)  | Poids(kg)      |  Pointure(cm) |
|:------------|:-----------:|:--------------:|:-------------:|
|masculin     | 182         |    81,6        | 30            |
|masculin     | 180         |    86,2        | 28            |
|masculin     | 170         |    77,1        | 30            |
|masculin     | 180         |    74,8        | 25            |
|féminin      | 152         |    45,4        | 15            |
|féminin      | 168         |    68          | 20            |
|féminin      | 165         |    59          | 18            |
|féminin      | 175         |    68          | 23            |


On souhaite classer la personne avec les caractéristiques suivantes en tant que féminin ou
masculin ?
Sexe= inconnu, taille=183, poids= 59, pointure= 20

### Etape 1 : Le calcul de la probalilité à priori de deux classe
On prend comme classe le sexe donc :
- masculin 4/8 = 0.5
- féminin 4/8  = 0.5

### Etape 2 :  On regarde la vraisemblance de l'individu à évaluer

d'après notre exemple l'individu à évaluer a ces caracteristiques :

Sexe= inconnu, taille=183, poids= 59, pointure= 20

 - On regarde si on connaît la taille de l'individu dans notre tableau d'entrée



|Sexe|   Taille(cm)|
|:--:|:-----------:|
| M  | 182         |  
| M  |180          |    
| M  |170          |    
| M  |180          |    
| F  |152          |    
| F  |168          |    
| F  |165          |    
| F  |175          |



On voit qu'il n'existe pas la taille 183

 - On regarde si on connaît le poids de l'individu dans notre tableau d'entrée



|Sexe| Poids(kg)      |
|:--:|:--------------:|
| M  |   81,6         |
| M  |   86,2         |
| M  |   77,1         |
| M  |   74,8         |
| F  |   45,4         |
| F  |   68           |
| F  |   59           |
| F  |   68           |  


on Voit qu'il existe un poids de 59 pour le sexe feminin


 - On regarde si on connaît la pointure de l'individu dans notre tableau d'entrée



|Sexe|  Pointure(cm) |
|:--:|:-------------:|
| M  | 30            |
| M  | 28            |
| M  | 30            |
| M  | 25            |
| F  | 15            |
| F  | 20            |
| F  | 18            |
| F  | 23            |


On voit qu'il existe une pointure de 20 pour le sexe feminin


On conclut la vraisemblance est :

        -  pour le sexe masculin 1/4 * 1/4 * 1/4 =1/16
        -  pour le sexe feminin  2/4 *1/4 *1/4 = 2/16


### Etape 3 : on calcul la  probabilité postérieure

    - pour le sexe masculin 1/16*0.5 = 0.03125
    - pour le sexe feminin  2/16*0.5 = 0.0625

### Conclusion
  La probalilité être du sexe feminin est plus éléve donc elle est une femme





```python
dataSet = pd.read_csv("Uses_Cases/Spam/Spamprediction.csv", sep=',', header =(0))

"""
  Cette fonction fait l'etape 1
"""
def proba_priori(dataSet):
    class_values = list(set(dataSet.Spam))
    class_data =  list(dataSet.Spam)
    priori = {}
    for i in class_values:
        priori[i]  = class_data.count(i)/float(len(class_data))
    return priori


"""
    Cette fonction fait l'étape 2
"""
def proba_vraisemblance(dataSet,champ, type_champ, prob):

        data_attr = list(dataSet[champ])
        class_data = list(dataSet.Spam)
        match =1
        for i in range(0, len(data_attr)):
            if class_data[i] == prob and data_attr[i] == type_champ:
                match+=1
        return match/float(class_data.count(prob))


"""
       Cette fonction fait l'étape 2
"""
def proba_conditional(dataSet,proba_priori,individu):
    res={}
    for prob in proba_priori:
        res[prob] = {}
        for champ in individu:
            #proba_vraisemblance(dataSet,champ, individu[champ], prob)
            #proba_vraisemblance[prob].update({individu[champ]:1})
            res[prob].update({ individu[champ]: proba_vraisemblance(dataSet,champ, individu[champ], prob)})
    return res

"""
    Cette fonction fait l'étape 3
"""
def classification(proba_vraisemblance,proba_priori):
    pred=[]
    for i in proba_vraisemblance:
        pred.append(reduce(lambda x, y: x*y, proba_vraisemblance[i].values())*proba_priori[i])
    return (0 if pred[0] > pred[1] else 1) # si c'est un Spam on renvoit 1 sinon 0

"""
    Faire l'accurancy
"""
def accuracy(y_true, y_pred):
    score=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            score+=1
    return (score/float(len(y_true)))


# L'entraitement
proba_priori = proba_priori(dataSet)

# On  test 100 individus  
#########################################
predition=[]

for i in range(100):
    individu={}
    for cle,item in dataSet.T[i].drop(labels=['Spam']).items():
        individu[cle]=item

    vraisemblance = proba_conditional(dataSet,proba_priori,individu)
    predition.append(classification(vraisemblance,proba_priori))

test_accM3 = accuracy(list(dataSet.Spam)[:100],predition)
# Affichage du résultat
print ('Acccuracy = ', test_accM3)
```

    Acccuracy =  0.97


4 - Evaluer la performance des trois modéles M1,M2,M3 à l'aide de la base de test fournie.


```python
x = [0,100]
y1 = [0,test_accM1]
y2 = [0,test_accM2]
y3 = [0,test_accM3]
plt.plot(x,y1,x,y2,x,y3)
plt.legend([' Modele M1 ', ' Modele M2 ',' Modele M3 '])
plt.show()
```



![png](/IA/output_31_0.png)



D'après le graphique on remarque que le modele m3 est le plus efficace, au niveau de l'apprentissage
