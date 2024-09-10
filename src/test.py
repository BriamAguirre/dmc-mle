# Código de Evaluación
############################################################################

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os


# Cargar la tabla transformada
def eval_model(filename, target):
    df = pd.read_csv(os.path.join('../data/data_process', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    x_test = df.drop([str(target)],axis=1)
    y_test = df[[str(target)]]
    y_pred_test=model.predict(x_test)
    resid = y_test - y_pred_test
    # Generamos métricas de diagnóstico
    r2 = model.score(x_test, y_test)
    print("Bondad de ajuste del modelo: ", np.round(r2,3))
    rmse = np.sqrt(pow(y_pred_test-y_test,2).sum()/len(resid))
    print("RMSE: ", np.round(rmse[str(target)]),3)

# Validación desde el inicio
def main():
    df = eval_model('df_test_vf.csv', "charges")
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()