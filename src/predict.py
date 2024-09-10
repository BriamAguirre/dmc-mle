# Código para las predicciones
############################################################################

from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
import os


# Cargar la tabla transformada
def predict_model(filename, predict, target):
    df = pd.read_csv(os.path.join('../data/data_process', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de test    
    res = model.predict(df.drop(columns = [str(target)])).reshape(-1,1)
    pred = pd.DataFrame(res, columns=[str(target)])
    pred.to_csv(os.path.join('../data/', predict))
    print(predict, 'exportado correctamente en la carpeta data')


# Scoring desde el inicio
def main():
    df = predict_model('df_test_vf.csv','prediciones.csv',"charges")
    print('Finalizó las predicciones del Modelo')


if __name__ == "__main__":
    main()