# Recalage à l'aide de bathymétrie pour un AUV
***
## Description du projet

Recalage sur une carte MNT d'un AUV à l'aide d'un filtre à particules

## Fichiers
### AUV_localisation
Ce fichier regroupe les algorithmes utilisés dans le cadre du sujet

### References
Documents utilisés pour ce projet :
* Documentation du DVL
* Documentation du filtre particulaire (PF)
* Sujet du projet

### Liens utiles
Manuel des différents capteurs utilisés
[SOURCE](https://moodle.ensta-bretagne.fr/course/view.php?id=863)

Lien YT d'explication de base pour le filtre à particules
[SOURCE](https://www.youtube.com/watch?v=NrzmH_yerBU&ab_channel=MATLAB)

### TODO
* Voir comment calibrer le DVL:
* Syncroniser de vitesse INS/DVL
* Voir les facteurs de réalignement du DVL lors de la 1ere phase
* Vérifier le paramétrage -> txt/py

***
### Capteurs mis à disposition
* DVL : Vitesse de fond/distance obliques (pas de Vitesse de courant)
* INS : gyroscope/Acceleromètre -> Vitesse et position (voir comment faire)
* GNSS : RTK (à traiter)
* Multifaisceau MBES : bathymétrie (équi-angle ?)
* SVP : Célérimètre de coque
