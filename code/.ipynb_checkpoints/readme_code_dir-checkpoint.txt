readme, code dir

2024-03-12
List of dicts to store as rows

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(),
    LinearSVC(dual="auto", random_state=0, tol=1e-5))

2024-03-11
Tybalt in start of dstnc.ipynb
    - build into stand-alone nb - done

# Front-end crossfolds
# RFE <- transfer learning
# Cellinger

# SVM
    - fifth option (?) SGD?