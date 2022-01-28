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
