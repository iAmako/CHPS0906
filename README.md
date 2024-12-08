# CHPS0906 - Kilyan Slowinski

---

## Fichiers

├── [Premier modèle (Chapitre 1)](premier_modele.md)       # Chapitre 1
├── [Deuxième modèle](chapitre_2_modele.py)                # Chapitre 2
├── [Bottleneck](bottleneck.py)                            # Mise en place de l'architecture bottleneck
├── [Inverted Bottleneck](inverted_bottleneck.py)          # Mise en place de l'architecture inverted bottleneck
├── [README](README.md)                                    # Documentation

## Résultats

Les résultats ci-dessous sont obtenus après 10 epochs.
En résumé :

- Le chapitre 1 donne des résultats satisfaisant à son échelle.
- Le chapitre 2 donne de bon résultats quand on pousse le nombre d'epochs.
- Le bottleneck est largement moins bon que le modèle du chapitre 2.
- L'inverted bottleneck est visiblement défectueux.

| Modèle  | Loss          | Accuracy |
| :------------------- |:-----:| -----:|
| Chapitre 1 | 1.8184 | 0.3600 |
| Chapitre 2  | 0.8193 | 0.7154 |
| Bottleneck  | 2.1106 | 0.3103 |
| Inverted Bottleneck | 5.5866 | 0.1000 |
