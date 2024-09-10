#################################################################################
######################## Código para el tratamiento ###########################
#################################################################################
# 1. Cargamos las librerias necesarias
import pandas as pd
import os
# 2. Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join("https://raw.githubusercontent.com/BriamAguirre/dmc-mle/main/data/data_origin/",filename), index_col = 0)
    print(filename, "cargado correctamente")
    return df
# 3. Realizamos las transformaciones correspondientes
def data_preparation(df, target):
    # Tratamiento de Dummies
    df = pd.get_dummies(df_train, dtype = "int", drop_first = True)
    # Tratamiento de escalamiento
    df2 = pd.DataFrame(scaler.fit_transform(df_train_dm[df_train_dm.columns.drop(str(target))]), columns = df_train_dm.columns.drop(str(target)))
    return df2
# 4. Exportamos la matriz de datos
def data_exporting(df, filename):
    df.to_csv(os.path.join("https://github.com/BriamAguirre/dmc-mle/tree/main/data/data_process/",filename))
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