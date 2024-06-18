# import sklearn
from sklearn import datasets
from pathlib import Path

skl_data_home = Path("work", "data", "sklearn")

datasets.fetch_olivetti_faces(
    data_home=skl_data_home,
    shuffle=True,
)
