# projet-INF8225
Video colorization with machine learning


Le script video_to_gray.py vous permet de transformer une vidéo en couleur en une vidéo en noir et blanc.
Vous pouvez utilisez les options suivantes : 
-i ou --input pour préciser la vidéo à passer en entrée
-o ou --output pour préciser le fichier de sortie

Le script video.py vous permet d'utiliser le deep learning pour colorier la vidéo.
-i ou --input pour préciser la vidéo à passer en entrée
-o ou --output pour préciser le fichier de sortie.
-m ou --model pour préciser le model qui vous souhaitez utiliser. Par défaut, on utilise la seconde architecture présenté dans le rapport, inspiré de cet article de recherche : https://arxiv.org/pdf/1603.08511.pdf

Pour que l'architecture 2 fonctionne, veuillez télécharger le fichier suivant et le mettre dans le repertoire models : https://www.dropbox.com/s/kyo9b78aojljqj2/google_net_colorize.caffemodel?dl=0