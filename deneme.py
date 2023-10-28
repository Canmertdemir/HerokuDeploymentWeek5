import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as mt
import pickle

data = pd.read_excel("CANSU DEMİR TEZ İSTATİSTİK - SON.xlsx")

hemipleji = data.loc[:, ["HEMİPLEJİ SÜRESİ"]].values.reshape(-1, 1)

tinetindenge_son = data["TİNETTİ DENGE 2"].values.reshape(-1, 1)

reg = lm.LinearRegression()

xtrain, xtest, ytrain, ytest = ms.train_test_split(hemipleji, tinetindenge_son, test_size=0.3, random_state=50)

# Modelin fitlenmesi aşaması

reg.fit(xtrain, ytrain)

# Modelin tahmin kısmı

ypredict = reg.predict(xtest)

print({"Vücut Kitle Endeksi:": xtest, "Tahmin edilen Düzelme Süresi": ypredict})

pickle.dump(reg, open("denemes.pkl", "wb"))