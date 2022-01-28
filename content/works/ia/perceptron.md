
![logo](/gitlog.jpeg)



## Le Perceptron

[Go back](https://claudio-a.netlify.app/works/go/)

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
