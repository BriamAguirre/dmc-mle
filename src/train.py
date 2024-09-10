# Código de Entrenamiento
############################################################################

from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/data_process', filename))
    x_train = df.drop(['charges'],axis=1)
    y_train = df[['charges']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    model_lr = LinearRegression()
    model_lr.fit(x_train, y_train)
    #xgb_mod=xgb.XGBClassifier(max_depth=2, n_estimators=50, objective='binary:logistic', seed=0, silent=True, subsample=.8)
    #xgb_mod.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(model_lr, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('df_train_vf.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()