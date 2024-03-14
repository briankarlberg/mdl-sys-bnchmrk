readme, code dir

2024-03-13
clssfr.ipynb and dstnc.ipynb to archive
    using synched cross validation in transfer_ipynb now
    and mmd.ipynb now for distance

rf.ipynb to archive
    this has the biomrk vars conversion
    and the itertools pairwise woven cross fold random forest

job_scheduler.sh to archive
    Exacloud bash loop

12 notebooks remain, 2 groups

torch.ipynb to archive

11 notebooks remain, deal with umaps next

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
# Cellinger <- dependency issues, output left printe

# SVM
    - fifth option (?) SGD? <-- address this