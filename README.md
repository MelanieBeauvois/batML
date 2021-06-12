# Automated Detection of Bat Species in Belgium
Python code to detect and identify the bat species in audio recordings by applying multi-class and multi-label Machine Learning techniques. This work was achieved by Lucile Dierckx and Mélanie Beauvois as part of their Master's thesis at the Université Catholique de Louvain.

### Our Architectures
During our thesis we designed several architectures, some more performant than others. They are referenced in the code with the following names and are listed from the most performant to the least performant:
- *batmen*: A single CNN receives spectrogram features and performs both detection and classification.
- *cnn2*: A first CNN receives spectrogram features and performs detection. A second CNN receives spectrogram features and performs classification.
- *hybrid_cnn_xgboost*: An XGBoost model takes as input features the output of the second to last layer of the batmen CNN. The XGBoost model performs both detection and classification.
- *hybrid_call_xgboost*: A first CNN receives spectrogram features and performs detection. An XGBoost model receives features related to bat calls and performs classification.
- *hybrid_call_svm*: A first CNN receives spectrogram features and performs detection. An SVM model receives features related to bat calls and performs classification.
- *hybrid_cnn_svm*: An SVM model takes as input features the output of the second to last layer of the batmen CNN. The SVM model performs both detection and classification.


### Reference
In case you want to use our work as part of your research please consider citing us:
```
Beauvois, Mélanie and Dierckx, Lucile (2021). Automated Detection of Bat Species in Belgium.
Université Catholique de Louvain, Louvain-la-Neuve, Belgium.
```


### Acknowledgements
We would first like to thank our Master's thesis supervisors Mr. Siegfried Nijssen and Mr. Olivier Bonaventure for their guidance and valuable advice as well as their time.

This thesis could not have been completed without the work and datasets of [Batdetective](http://visual.cs.ucl.ac.uk/pubs/batDetective), which served as a starting point for our automated detection of bat species.

We would also like to express our gratitude to [Natagora](https://www.natagora.be/) and more specifically [Plecotus](https://plecotus.natagora.be/index.php?id=707) for the large amount of labelled bat call recordings they shared with us.

Our sincere thanks go to [Maxime Franco and Corrado Lipani](https://github.com/c-lipani/batmen) for the work they accomplished last year. We are especially thankful for their data preprocessing and their prompt responses to our questions regarding their work.

A special thanks to Maxime Piraux for his help on technical aspects as well as for his advice and time.

Last, but not least, many thanks to our families and friends for their support and encouragements.


### License
Everything that is made available on this GitHub repository can be used solely for non-commercial research purposes. Otherwise, the authors need to be contacted.
