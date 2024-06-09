
# Prédiction de l'Admissibilité aux Prêts Bancaires

## Introduction

Dans le paysage financier actuel, l'accès au crédit est un élément crucial pour de nombreux individus et entreprises. Les prêts bancaires permettent de réaliser des projets importants tels que l'achat d'une maison, le financement d'une éducation ou le démarrage d'une entreprise. Cependant, l'octroi de prêts n'est pas une décision à prendre à la légère pour les institutions financières. Elles doivent évaluer avec soin la solvabilité de chaque demandeur afin de minimiser les risques de défaut de paiement et de maximiser leurs bénéfices.

Dans ce contexte, la Data Science offre des outils puissants pour analyser les données et construire des modèles prédictifs capables d'estimer la probabilité qu'un individu soit en mesure de rembourser un prêt. Ce projet vise à utiliser ces techniques pour répondre à une question commerciale cruciale : devons-nous accorder un prêt bancaire au client X ?

## Objectifs

Ce projet vise à atteindre les objectifs suivants :

1. **Prédiction de l'admissibilité aux prêts bancaires** : Développer un modèle de machine learning capable d'estimer la probabilité qu'un client donné rembourse un prêt bancaire. Cette prédiction sera basée sur les caractéristiques du client et son historique financier.
2. **Optimisation des décisions de crédit** : Fournir des recommandations précieuses aux décideurs quant à l'octroi de prêts pour le client X. Ces recommandations devront être fondées sur une analyse approfondie des prédictions du modèle et des facteurs influençant la décision de crédit.
3. **Robustesse et interprétabilité du modèle** : Concevoir un modèle de machine learning robuste et interprétable, capable de générer des prédictions fiables tout en permettant une compréhension claire des mécanismes sous-jacents.
4. **Amélioration de la rentabilité et de la gestion des risques** : Contribuer à l'amélioration de la rentabilité de l'institution financière en minimisant les risques de défaut de paiement tout en maximisant les opportunités de prêts rentables et sûrs.

## Setup

Pour mener à bien ce projet de Data Science, nous utiliserons les bibliothèques suivantes :

- Pandas
- Numpy
- Scikit-Learn
- Matplotlib
- Seaborn

## Détail de la base de données

Imaginez que vous êtes invité en tant que Data Scientist à travailler sur un projet visant à évaluer l'admissibilité aux prêts bancaires pour divers clients. Pour mener à bien cette tâche, vous disposez d'un ensemble de données détaillant les caractéristiques des clients ainsi que leur historique financier. Cette base de données est essentielle pour comprendre le contexte et construire des modèles prédictifs efficaces.

### Description de la Base de Données

| Variable            | Description                                                      |
|---------------------|------------------------------------------------------------------|
| ApplicantIncome     | Revenu du demandeur exprimé en dollar                            |
| Gender              | Sexe du demandeur (male: Homme, female: Femme)                   |
| Married             | Statut matrimonial du client (Célibataire, Marié, Divorcé, Veuf) |
| Education           | Niveau d'éducation du demandeur (Graduate: Diplômé, Not-Graduate: Non diplômé) |
| Loan_Amount_Term    | Durée du prêt en jours                                           |
| Self_Employed       | Entrepreneur (yes: Oui, no: Non)                                 |
| Credit_History      | Historique de crédit du client (1: Oui, 0: Non)                  |
| LoanAmount          | Montant demandé pour le prêt en devise locale                    |
| Loan_Status         | Statut des prêts (Y: Oui, N: Non)                                |
| Property_Area       | Zone d'habitation du demandeur (urbain, semi-urbain, rural)      |
| CoapplicantIncome   | Revenu du conjoint                                               |

### Objectif de la Base de Données

L'objectif principal de cette base de données est de fournir des informations pertinentes pour la prise de décision en matière de crédit. En analysant ces données, nous chercherons à identifier les facteurs qui influent sur la capacité d'un individu à rembourser un prêt et à construire un modèle prédictif capable d'estimer cette probabilité.

## Démarches du Travail

### 1. Analyse Exploratoire de Données (EDA)

**Objectif :** Comprendre au maximum les données dont nous disposons pour définir une stratégie.

- **Analyse de la forme**
  - Identification de la cible : Loan_Status
  - Nombre de lignes et de colonnes : (614 lignes, 13 colonnes)
  - Types de variables :
    - Variables qualitatives : 8
    - Variables quantitatives : 5
  - Identification des valeurs manquantes et procédure d'imputation.

- **Analyse du fond**
  - Visualisation de la cible (Histogramme/Boxplot)
    - Pourcentage d'accord de prêt : 68.73%
    - Pourcentage des prêts non-accordés : 31.27%
  - Compréhension des différentes variables (Internet)
  - Visualisation des relations features-cible (Histogramme/Boxplot)
    - Target/Credit_History : Les demandeurs ayant un historique de prêt obtiennent un nombre significativement plus élevé d'accords de prêt que ceux sans antécédents de prêt.
  - Identification des valeurs aberrantes.

### 2. Prétraitement des Données

**Objectif :** Transformer les données pour les mettre en format propice au Machine Learning.

**Démarches :**
  - Création du Train set / Test set
  - Élimination des valeurs manquantes
  - Encodage
  - Suppression des outliers néfastes au modèle (Optionnel)
  - Sélection des features
  - Ingénierie des features
  - Mise à l'échelle des features

### 3. Construction du Modèle (Modeling)

**Objectif :** Développer un modèle ML qui répond à l'objectif final.

**Démarches :**
  - Définir une fonction d'évaluation
  - Entraînement des différents modèles
  - Optimisation avec GridSearchCV
  - Analyse des erreurs et retour au pre-processing/EDA
  - Courbe d'apprentissage et prise de décision

### 4. Interprétation des Résultats

**Analyse des Prédictions :**
  - Interpréter les prédictions du modèle pour comprendre les décisions qu'il prend.
  - Examiner les exemples où le modèle se trompe pour identifier les erreurs de prédiction et comprendre les raisons possibles.

**Identification des Caractéristiques Importantes :**
  - Analyser les caractéristiques les plus importantes pour la prédiction en utilisant des techniques telles que l'importance des variables ou les coefficients du modèle.
  - Identifier les facteurs qui ont le plus d'impact sur la décision finale.

**Rétroaction et Réajustement :**
  - Réexaminer les étapes précédentes en fonction des résultats obtenus et ajuster le processus si nécessaire.
  - Réentraîner le modèle avec de nouvelles données ou des paramètres différents si cela est jugé nécessaire.

### 5. Communication des Résultats

**Rapports et Visualisations :**
  - Présenter les résultats sous forme de rapports écrits ou de présentations visuelles.
  - Utiliser des graphiques, des diagrammes et des tableaux pour illustrer les résultats de manière claire et compréhensible.

**Recommandations :**
  - Formuler des recommandations basées sur les résultats de l'analyse pour soutenir la prise de décision.
  - Expliquer les implications des résultats et proposer des actions concrètes à entreprendre en fonction des conclusions tirées.

**Discussion et Partage :**
  - Discuter des conclusions avec les parties prenantes concernées pour obtenir leur rétroaction.
  - Partager les résultats et les méthodes utilisées avec d'autres membres de l'équipe ou de l'organisation.

