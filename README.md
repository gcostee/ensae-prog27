# ensae-prog27
I : Présentation du projet et méthode 

**a : Problématique** 

**b : Présentation du dataset**

Les données utilisées pour ce projet proviennent de la base de donnée Lichess : https://database.lichess.org/
Les données sont des parties au format PGN, le format standart pour enregistrer des parties d'échec. Pour chaque partie, le PGN contient : le tournoi ou contexte de la partie, le lieu, la date, la ronde, les noms de joueurs, le resultat le classement des joueurs et la cadence, ainsi que l'ensemble de la partie avec les coups en notation algébrique standart (SAN). Ce format permet de stocker un maximum de données sur une partie tout en étant très léger, ce qui nous a permis de travailler sur un grand nombre de parties (100 millions de parties par mois).

**c : L'esthétisme : comment quantifier un concept humain ?**

Notre objectif étant de trouver les plus belles parties parmis celles jouées dans la mois, la première question était de définir ce qui fait une "belle" partie d'échecs. En effet, on part d'une notion subjective et humaine, qui ne peut être saisie par une machine. Il faut donc l'adapter avec des critères quantifiables et mesurables pour s'en approcher un maximum, et permettre de filtrer les nombreuses parties pour n'en conserver que quelques unes. Ainsi, nous avons selectionnés les éléments qui nous parraissaient les plus pertinents pour juger de l'esthétisme d'une partie d'échecs : les retournements d'avantages, les sacrifices de pièces, le dynamisme de la partie ... Nous avons ainsi dû selectionner nos critères de tri des parties, ainsi que la façon de les mesurer, afin de filtrer les parties au fur et à mesure, en commençant par les critères les plus évidents et facilement quantifiables pour ensuite étudier plus précisemment les parties restantes. 