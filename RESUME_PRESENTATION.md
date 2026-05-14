# Resume de Presentation - Controle Qualite des Impellers par Deep Learning

## Titre du projet

**Systeme Intelligent de Controle Qualite des Produits de Fonderie (Impellers)**
*Transfer Learning et Fine-Tuning avec EfficientNetB0*

---

## Guide Slide par Slide

### Slide 1 - Page de titre

**A dire :** Presenter le titre, votre nom, la date. Mentionner que ce projet porte sur l'application du deep learning au controle qualite industriel.

---

### Slide 2 - Contexte et problematique

**A dire :**
- Le controle qualite des pieces de fonderie est traditionnellement manuel (operateur humain)
- Problemes : fatigue, subjectivite, lenteur, cout
- Objectif : automatiser la detection des defauts par vision par ordinateur et deep learning
- Question cle : peut-on utiliser un reseau pre-entraine pour classer des images industrielles ?

---

### Slide 3 - Dataset

**A dire :**
- Dataset : "Casting Product Image Data for Quality Inspection"
- ~7364 images au total (train : 3758 defectueux + 2891 ok ; test : 453 defectueux + 262 ok)
- Images en niveaux de gris, vue frontale des impellers
- Dataset deja augmente a la source
- Classification binaire : defectueux vs ok

---

### Slide 4 - Architecture du modele

**A dire :**
- Base : EfficientNetB0 pre-entraine sur ImageNet (5.3M parametres)
- Pourquoi EfficientNetB0 ? Meilleur ratio precision/taille, ideal pour Colab
- Architecture : EfficientNetB0 (features) + GlobalAveragePooling + Dropout(0.3) + Dense(1, sigmoid)
- Entree : images 224x224x3

---

### Slide 5 - Strategie d'entrainement en 2 phases

**A dire :**
- **Phase 1** : Base gelee, on entraine uniquement la tete (10 epochs, lr=1e-3)
  - Le reseau apprend a utiliser les features ImageNet pour notre tache
- **Phase 2** : On degele les 40 dernieres couches du backbone (10 epochs, lr=1e-5)
  - LR tres faible pour ne pas detruire les features apprises
  - BatchNorm reste gele (statistiques ImageNet preservees)
- Cette approche evite le catastrophic forgetting

---

### Slide 6 - Resultats Phase 1 (Transfer Learning)

**A dire :**
- Meme avec le backbone gele, on atteint deja une excellente accuracy (~95-97%)
- Cela montre que les features ImageNet sont transferables aux images industrielles
- Convergence rapide (quelques epochs suffisent)

---

### Slide 7 - Resultats Phase 2 (Fine-Tuning)

**A dire :**
- Le fine-tuning ameliore encore les performances
- Accuracy finale sur le test set : ~99%+
- AUC-ROC : ~0.99+
- Le gain est visible sur les courbes d'entrainement

---

### Slide 8 - Evaluation detaillee

**A dire :**
- Matrice de confusion : tres peu d'erreurs
- Courbe ROC proche du coin superieur gauche (AUC quasi-parfait)
- Courbe Precision-Recall : precision et recall eleves simultanement
- Classification report : precision/recall/f1 > 0.97 pour les deux classes

---

### Slide 9 - Analyse des erreurs

**A dire :**
- En controle qualite, les Faux Negatifs sont plus graves que les Faux Positifs
- FN = piece defectueuse livree au client (risque securite/qualite)
- FP = bonne piece rejetee (perte economique, moins critique)
- On peut ajuster le seuil de decision pour privilegier la detection des defauts
- Compromis configurable selon le contexte industriel

---

### Slide 10 - Comparaison avec baseline

**A dire :**
- CNN simple from scratch : performances correctes mais inferieures
- EfficientNetB0 gele (Phase 1 seule) : nette amelioration grace au transfer learning
- EfficientNetB0 fine-tune (modele final) : meilleures performances
- Conclusion : le transfer learning + fine-tuning apporte un gain significatif, surtout avec un dataset de taille moderee

---

### Slide 11 - Limites et ameliorations

**A dire :**
- Limites : dataset limite (~7k), binaire seulement, angle unique, pas de localisation du defaut
- Ameliorations proposees :
  - Grad-CAM pour visualiser ou le modele regarde
  - Classification multi-classe des types de defauts
  - Compression (TFLite) pour deploiement embarque
  - Detection/segmentation pour localiser les defauts

---

### Slide 12 - Conclusion

**A dire :**
- Le transfer learning avec EfficientNetB0 atteint ~99% d'accuracy sur la detection de defauts
- L'approche en 2 phases est efficace et rapide a entrainer (~10 min sur GPU Colab)
- Applicable a d'autres problemes d'inspection visuelle industrielle
- Prochaines etapes : Grad-CAM, multi-classe, deploiement production

---

## Points cles a retenir

1. **Transfer Learning** = reutiliser un modele pre-entraine (ImageNet) pour une tache specifique
2. **Fine-Tuning** = ajuster les couches superieures du modele pre-entraine a nos donnees
3. **2 phases** = d'abord la tete, puis le backbone (avec un LR 100x plus petit)
4. **EfficientNetB0** = architecture compacte et performante, ideale pour les petits datasets
5. **Resultats** : ~99% accuracy, AUC ~0.99 sur le test set
6. **Application industrielle** : reduction des couts, amelioration de la qualite, elimination de la subjectivite humaine

---

## Questions possibles du prof (avec reponses courtes)

### Q : Pourquoi EfficientNetB0 et pas ResNet ou VGG ?

**R :** EfficientNetB0 offre le meilleur compromis accuracy/taille. Il est plus leger que VGG16 (5.3M vs 138M parametres) et plus performant. Sur un petit dataset avec un GPU Colab, c'est le choix optimal.

### Q : Pourquoi geler le backbone en Phase 1 ?

**R :** Pour eviter le "catastrophic forgetting". Si on entraine tout le reseau d'un coup avec un grand learning rate, on detruit les features utiles apprises sur ImageNet. On entraine d'abord la tete, puis on ajuste finement le backbone.

### Q : Pourquoi un learning rate plus petit en Phase 2 ?

**R :** Les poids du backbone sont deja proches de l'optimum (pre-entraines sur 1.2M images ImageNet). On veut les ajuster legerement, pas les reinitialiser. Un LR de 1e-5 (vs 1e-3 en Phase 1) permet un ajustement fin sans perturbation majeure.

### Q : Pourquoi garder les BatchNorm gelees en Phase 2 ?

**R :** Les couches BatchNorm ont des statistiques (moyenne, variance) calculees sur ImageNet. Avec notre petit batch size (32), les recalculer donnerait des estimations bruitees et degraderait les performances.

### Q : 99% d'accuracy, n'est-ce pas du surapprentissage ?

**R :** Non, car : (1) c'est mesure sur le test set (donnees jamais vues), (2) le test set est separe du train, (3) on utilise EarlyStopping et un validation set, (4) les images de casting ont des patterns de defauts assez distincts, ce qui facilite la tache.

### Q : Quelle est la difference entre transfer learning et fine-tuning ?

**R :** Transfer learning = utiliser un modele pre-entraine comme extracteur de features (backbone gele). Fine-tuning = deverrouiller certaines couches du backbone pour les adapter a la tache specifique. Le fine-tuning est une etape supplementaire du transfer learning.

### Q : Comment deployer ce modele en production ?

**R :** Plusieurs options : (1) TensorFlow Serving (API REST), (2) Conversion en TFLite pour embarque/mobile, (3) Export ONNX pour integration multi-framework, (4) Edge device avec accelerateur (Jetson Nano, Coral). Il faudrait aussi un systeme de monitoring pour detecter la degradation des performances.

### Q : Pourquoi les FN sont plus graves que les FP dans ce contexte ?

**R :** Un Faux Negatif = une piece defectueuse envoyee au client. Consequence : panne, rappel produit, perte de confiance, voire accident. Un Faux Positif = une bonne piece rejetee. Consequence : perte economique limitee (la piece peut etre re-inspectee manuellement).

---

## Suggestions de timing (~10 minutes)

| Slide | Duree | Contenu |
|-------|-------|---------|
| 1 | 30s | Titre et introduction |
| 2 | 1 min | Contexte et problematique |
| 3 | 1 min | Presentation du dataset |
| 4 | 1 min | Architecture du modele |
| 5 | 1 min 30s | Strategie 2 phases (point cle) |
| 6-7 | 1 min 30s | Resultats Phase 1 + Phase 2 |
| 8 | 1 min | Evaluation detaillee |
| 9 | 1 min | Analyse des erreurs |
| 10 | 1 min | Comparaison baseline |
| 11 | 30s | Limites et ameliorations |
| 12 | 30s | Conclusion |

**Total : ~10 minutes**

---

## Conseils pour la presentation

- Insister sur le **pourquoi** (probleme industriel reel) avant le **comment** (technique)
- Montrer les **graphiques** (courbes, matrice de confusion) pour appuyer vos propos
- Preparer une **demo rapide** si possible (montrer le notebook qui tourne)
- Rester synthetique : ne pas lire le code, expliquer les concepts
- Anticiper les questions sur les choix techniques (EfficientNet, 2 phases, seuil)
