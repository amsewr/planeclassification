# planeclassification

Le but de ce projet est de construire un modèle de classification fonctionnant sur des images d'avions. Lien vers la démo :

# Setup 

Le fichier du repository requirements.txt regroupe l'ensemble des prérequis pour créer notre modèle. 

# Usage 

1) Le modèle de classification est créé à partir du notebook d'entrainement : train_classification_model.ipynb.
   On note que plusieurs modèles peuvent-être entrainés en fonction de l'information qu'on souhaite prédire (manufacturer, family ou variant). 
   La modification de la variable à prédire se fait directement à partir du fichier launch.yaml.
2) Les modèles sont enregistrés dans le répertoire MODEL_DIR sous le format suivant : MODEL_DIR/TARGET_NAME.h5
   Le fichier cat.yaml qui regroupe les listes des noms de catégories de nos variables est aussi enregistré dans ce même répertoire. 
3) L'application est dévelopée dans le fichier app.py qui fonctionne à partir des fichiers cat.yaml précedemment cité ainsi que du fichier app.yaml regroupant les        paramètres d'initialisation parmis lesquels on retrouve les liens vers les modèles entrainés.
4) L'application peut-être lancée à partir du terminal anaconda prompt en utilisant notamment la commande suivante : 
   
