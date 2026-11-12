# Pretraitement du Dataset de Transactions Bancaires

## Apercu

Ce projet presente un pipeline de pretraitement de donnees applique a un jeu de transactions bancaires pour la detection de fraude.

Les defis traites dans ce projet :
- Nettoyage des donnees : suppression des doublons et imputation des valeurs manquantes.
- Prevention du Data Leakage : exclusion des identifiants et de la variable post-incident (`resultat_enquete`).
- Gestion des valeurs extremes (outliers) : analyse par l'ecart interquartile (IQR) sur les montants.
- Comparaison des scalers : evaluation de StandardScaler, MinMaxScaler et RobustScaler.
- Desequilibre des classes : re-echantillonnage cible sur le groupe d'entrainement.
- Pipeline de production : ColumnTransformer scikit-learn avec gestion des categories inconnues (`handle_unknown='ignore'`) et export Joblib.

---

## Description du Dataset

Le fichier `transactions-dataset.csv` contient 3 000 operations bancaires decrites par les attributs suivants :

| Variable | Type | Description |
| :--- | :--- | :--- |
| `id_operation` | Identifiant | Cle technique arbitraire (exclue du modele). |
| `id_client` | Identifiant | Identifiant client (exclu du modele). |
| `montant_dh` | Numerique continu | Montant de la transaction en Dirhams marocains (DH). |
| `anciennete_mois` | Numerique continu | Anciennete du compte client en mois. |
| `nb_ops_24h` | Numerique discret | Nombre d'operations effectuees au cours des 24 dernieres heures. |
| `heure` | Numerique discret | Heure de la transaction (de 0 a 23). |
| `canal` | Categoriel | Canal utilise : application, site_web, agence, telephone. |
| `appareil` | Categoriel | Terminal : mobile, ordinateur, terminal. |
| `region` | Categoriel | Region : Casablanca-Settat, Marrakech-Safi, Souss-Massa, Draa-Tafilalet, Autre. |
| `international` | Categoriel binaire | Indicateur d'operation internationale (oui / non). |
| `resultat_enquete`| Categoriel | Resultat post-incident (exclu pour eviter le Data Leakage). |
| `fraude` | Binaire (0 ou 1) | Variable cible (1 = Fraude, 0 = Legitime). |

---

## Etapes de Pretraitement

### 1. Diagnostic Initial et Prevention du Data Leakage

- **Suppression des doublons** : Elimination des transactions dupliquees basees sur `id_operation`.
- **Valeurs manquantes** : Identification des colonnes a completer (imputation par mediane pour les nombres, par mode pour les categories).
- **Prevention du Data Leakage** :
  - `id_operation` et `id_client` sont retires car ils ne contiennent pas de signal generalisable.
  - `resultat_enquete` est retire car cette donnee n'est disponible qu'apres l'investigation humaine. L'inclure fausserait totalement l'apprentissage en donnant la reponse au modele.

---

### 2. Distribution des Montants et Traitement des Outliers

La variable `montant_dh` est asymetrique avec une longue queue vers les valeurs elevees :

| Distribution du Montant | Boxplot des Outliers |
| :---: | :---: |
| ![Distribution Montant](images/distribution_montant.png) | ![Boxplot Montant](images/boxplot_montant.png) |

#### Reperes statistiques IQR :
- Q1 (25%) : 384.27 DH
- Mediane (50%) : 706.80 DH
- Q3 (75%) : 1244.90 DH
- IQR : 860.63 DH
- Seuil superieur L2 (Q3 + 1.5 * IQR) : 2535.85 DH

#### Comparaison : Option A vs Option B

| Option A : Conservation des montants reels | Option B : Ecretage au seuil L2 (2535.85 DH) |
| :---: | :---: |
| ![Option A vs Option B](images/gestion_outliers_option_a_vs_b.png) |

- **Choix retenu** : Dans la detection de fraude, les montants atypiques portent un signal essentiel. Nous conservons les montants reels (Option A) et utilisons un scaler insensible aux valeurs extremes.

---

### 3. Comparaison des Methodes de Mise a l'Echelle

Trois transformateurs ont ete compares pour les variables numeriques :

![Comparaison des Scalers](images/comparaison_scalers.png)

1. **StandardScaler** : Sensible aux valeurs extremes, la moyenne et l'ecart-type sont tires vers le haut, ce qui ecrase les valeurs courantes pres de zero.
2. **MinMaxScaler** : Sensible aux montants maximums (> 30 000 DH), comprimant la majorite des observations dans un intervalle tres reduit.
3. **RobustScaler (Choix retenu)** : Utilise la mediane et l'ecart interquartile (IQR). Les valeurs ordinaires restent centrees autour de 0 sans deformation due aux montants eleves.

---

### 4. Gestion du Desequilibre des Classes

Les fraudes representent environ 10 % des transactions :

| Classes Initiales | Echantillon Train | Re-echantillonnage (OverSampling) |
| :---: | :---: | :---: |
| ![Classes Initiales](images/equilibre_classes_initial.png) | ![Classes Train](images/equilibre_classes_train.png) | ![OverSampling](images/oversampling_equilibre.png) |

Le sur-echantillonnage (RandomOverSampler / SMOTE-NC) est applique exclusivement sur `X_train` afin de preserver l'integrite du jeu de test.

---

### 5. Pipeline Final et Exportation

Le pipeline complet est construit avec `ColumnTransformer` :
- **Numerique** : `SimpleImputer(strategy='median')` -> `RobustScaler()`
- **Categoriel** : `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(handle_unknown='ignore')`

Le parametre `handle_unknown='ignore'` permet au modele de traiter de nouvelles categories sans lever d'erreur en production.

Le preprocesseur est sauvegarde avec `joblib` :

```python
import joblib
joblib.dump(full_pipeline, 'preprocessor_transactions.joblib')
```

---

## Structure du Projet

```
transactions-dataset-preprocessing/
|-- images/
|   |-- boxplot_montant.png
|   |-- comparaison_scalers.png
|   |-- distribution_montant.png
|   |-- equilibre_classes_initial.png
|   |-- equilibre_classes_train.png
|   |-- gestion_outliers_option_a_vs_b.png
|   `-- oversampling_equilibre.png
|-- .gitignore
|-- pretraitement-dataset-transactions.ipynb
|-- README.md
`-- transactions-dataset.csv
```

---

## Utilisation

Pour executer le notebook :

```bash
# Cloner le depot
git clone https://github.com/abdelhamid-elmghari/regression-lab.git

# Se deplacer dans le dossier
cd regression-lab/"practical projects"/transactions-dataset-preprocessing

# Lancer Jupyter
jupyter notebook pretraitement-dataset-transactions.ipynb
```
