ensae-prog27

# Projet Python pour la data science : Titre du projet 


Auteurs : Camille BOURGET, Gautier COSTE, Valentin FAZEMBAI TREMOUILLES
---

## I : Présentation du projet et méthode 

### A : Problématique

### B : Présentation du dataset

Les données utilisées pour ce projet proviennent de la base de donnée Lichess : https://database.lichess.org/
Les données sont des parties au format PGN, le format standart pour enregistrer des parties d'échec. Pour chaque partie, le PGN contient : le tournoi ou contexte de la partie, le lieu, la date, la ronde, les noms de joueurs, le resultat le classement des joueurs et la cadence, ainsi que l'ensemble de la partie avec les coups en notation algébrique standart (SAN). Ce format permet de stocker un maximum de données sur une partie tout en étant très léger, ce qui nous a permis de travailler sur un grand nombre de parties (environ 100 millions de parties par mois).

### C : L'esthétisme : comment quantifier un concept humain ?

Notre objectif étant de trouver les plus belles parties parmis celles jouées dans le mois, la première question était de définir ce qui fait une "belle" partie d'échecs. En effet, on part d'une notion subjective et humaine, qui ne peut être saisie par une machine. Il faut donc l'adapter avec des critères quantifiables et mesurables pour s'en approcher un maximum, et permettre de filtrer les nombreuses parties pour n'en conserver que quelques unes. Ainsi, nous avons selectionnés les éléments qui nous parraissaient les plus pertinents pour juger de l'esthétisme d'une partie d'échecs : les retournements d'avantages, les sacrifices de pièces, le dynamisme de la partie ... Nous avons ainsi dû selectionner nos critères de tri des parties, ainsi que la façon de les mesurer, afin de filtrer les parties au fur et à mesure, en commençant par les critères les plus évidents et facilement quantifiables pour ensuite étudier plus précisemment les parties restantes (voir partie II). 

## II

### A : premier filtrage : filtrer_parties

La première étape de notre projet consiste à réduire grandement l'échantillon de données pour ne conserver qu'un nombre restreint de parties intéressantes, que nous analyserons ensuite de manière plus poussée. 
Pour cela, nous regroupons les parties par paquets de 500 000, et nous effectuons le filtrage suivant : conserver seulement les parties avec un vainqueur, dont les deux joueurs ont plus de 1500 elo (ce qui correspond au top 50% des joueurs) et avec un faible écart de niveau entre les joueurs. Nous avons supprimons ensuite les parties jouées par correspondance, et celles dont l'évaluation n'est pas disponible : l'évaluation nous permet d'avoir plus d'informations sur la partie, et le fait qu'un des joueurs ait cherché à l'évaluer renvoie déjà un signal sur sa qualité. 
A la fin de cette étape, il nous reste un total de *3 124 123* parties, soit *3,3 %* du nombre du départ. Cette première étape nous permet donc de réduire grandement la quantité de données à traiter de manière rapide et efficace (on a peu de chance d'éliminer une des plus "belles" parties de cette façon).
Enfin, en plus de fluidifier l'exécution du code, le fait de regrouper les parties par groupe de 500 000 nous a permis d'évaluer combien de parties ce filtrage conserve en moyenne afin dimensionner au mieux l'étape suivante. En effet, nous avons constaté lors des tests que cette étape conserve en moyenne *158 000* parties par groupe, avec une faible variance empirique (*3%* de la moyenne). Sur cette base, nous avons donc calibré l'algorithme de l'étape suivante pour traiter 3 millions de parties, le risque de dépasser significativement ce nombre étant très faible. 