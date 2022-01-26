



![logo](/gitlog.jpeg)
# Brute-Force

[Go back](https://claudio-a.netlify.app/works/go/)



In this paper I'm going to share with you a tool that I made to do brute force attac.

 Brute force attack is a method used in cryptanalysis to find a password or key. It is a question of testing, one by one, all the possible combinations.
 [source wkipedia](https://fr.wikipedia.org/wiki/Attaque_par_force_brute)


The idea is we have a zip archive and a file where are all possible password of this archive , the tool will trying each password to find out the good.


This is the function that does it.

```python
def bruteForceFromFile(listeMdP):
    res=""
    for motdepasse in listeMdP:
        cmd = subprocess.Popen("unzip -P %s -o archive.zip 2> /dev/stdout"%motdepasse, shell=True,stdout=subprocess.PIPE)
        (resultat, ignorer) = cmd.communicate()
        resultat=resultat.decode("utf-8")
        if resultat.find("incorrect")==-1 and resultat.find("error")==-1:
            res=motdepasse
            break
    print("Thread",threading.get_ident()," Trouvé ? ",res)

```

there can take long time if the file has a quite a lot of words then we us threads to parallelize the operations





 [Code source on my github](https://github.com/MonaQuimbamba/TIC/blob/master/1/brute-force/bruteForceFromFile.py)
