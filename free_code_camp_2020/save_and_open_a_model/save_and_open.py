from sklearn.externals import joblib


# Salva o model para uso posterior
#saving
filename = 'model.sav'
joblib.dump(model, filename)

# Abrindo o modelo posteriormente
# open
model = joblib.load(filename)