import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv(filename, sep="\t")
titanic.head()

titanic.columns
titanic.shape

titanic.isnull().sum()
plt.figure(figsize=(10, 6))

sns.heatmap(titanic.isnull(), cmap="spring")
plt.show()

titanic.drop("Cabin", axis=1, inplace=True)
titanic.head()

titanic.dropna(inplace=True)
titanic.isnull().sum()

plt.figure(figsize=(10, 6))
sns.heatmap(titanic.isnull(), cbar=False)

plt.show()
pd.get_dummies(titanic["Sex"]).head()


sex = pd.get_dummies(titanic["Sex"], drop_first=True)
sex.head(4)

pd.get_dummies(titanic["Embarked"]).head(4)


embarked = pd.get_dummies(
    titanic["Embarked"],
    drop_first=True
)

embarked.head(4)

pclass = pd.get_dummies(
    titanic["Pclass"],
    drop_first=True
)

pclass.head(4)

titanic = pd.concat(
    [titanic, sex, embarked, pclass],
    axis=1
)

titanic.head()

