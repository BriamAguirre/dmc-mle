#################################################################################
######################## Código para el tratamiento ###########################
#################################################################################
# 1. Cargamos las librerias necesarias
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
# 2. Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/data_origin', filename))
    print(filename, "cargado correctamente")
    return df
# 3. Realizamos las transformaciones correspondientes
def data_preparation(df, target):
    # Tratamiento de Dummies
    df_dum = pd.get_dummies(df, dtype = "int", drop_first = True)
    # Tratamiento de escalamiento
    scaler = MinMaxScaler()
    df_esc = pd.DataFrame(scaler.fit_transform(df_dum[df_dum.columns.drop(str(target))]), columns = df_dum.columns.drop(str(target)))
    df_vf = pd.concat([df_esc, df[[str(target)]]], axis = 1)
    return df_vf
# 4. Exportamos la matriz de datos
def data_exporting(df, filename):
    df.to_csv(os.path.join("../data/data_process/",filename))
    print(filename, 'exportado correctamente en la carpeta data_process')
# 5. Generamos las matrices necesarias para la implementación
def main():
    # Matriz de Entrenamiento
    df_train = read_file_csv("insurance_train.csv")
    df_train_vf = data_preparation(df_train, "charges")
    data_exporting(df_train_vf, "df_train_vf.csv")
    # Matriz de Validación
    df_test = read_file_csv("insurance_test.csv")
    df_test_vf = data_preparation(df_test, "charges")
    data_exporting(df_test_vf, "df_test_vf.csv")
    
if __name__ == "__main__":
    main()