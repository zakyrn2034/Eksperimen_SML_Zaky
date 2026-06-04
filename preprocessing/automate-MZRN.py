import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import kagglehub
import joblib

#Variabel-variabel
#Format : [nama dataset,file csv]
ds_link = ["fedesoriano/stellar-classification-dataset-sdss17","star_classification.csv"]

target_col = "class"
chosen_features = []

path = "./"
ds_path = path + "StellarClassification-raw.csv"
file_path = path + "preprocessing/StellarClassification_preprocessing/headers.csv"
save_path = path + "pipeline.joblib"

#Membaca dataset
def load_ds(ds_link = ds_link):
    path = kagglehub.dataset_download(ds_link[0])

    csv_path = os.path.join(path, "star_classification.csv")

    df = pd.read_csv(csv_path)
    return df

def load_ds_file(file_path = ds_path):
    df = pd.read_csv(ds_path)
    return df

#Mendefinisikan fungsi:
def del_missing(df):
  df_missing = df.dropna() #Delete empty rows
  df_missing = df_missing.dropna(how='all', axis=1) #Delete empty columns
  df_missing = df_missing.drop_duplicates() #Delete duplicates
  return df_missing

def cap_outlier(df):
  df_cap = df.copy()
  df_cap.info()

  for col in df_cap.columns:
    upper = np.percentile(df_cap[col], 95)
    lower = np.percentile(df_cap[col], 5)
    df_cap[col] = np.clip(df_cap[col], lower, upper)
  return df_cap

#Sebagian kode di bawah diambil dari modul Dicoding
def data_preprocessing(df,target_col,save_path,file_path,chosen_features=None):

  if chosen_features is not None:
    valid_features = []
    for i in chosen_features:
      if i in df.columns:
        valid_features.push(i)
    data = df[chosen_features]
  else:
    data = df.copy()

  data = del_missing(data)

  numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
  categorical_features = data.select_dtypes(include=['object']).columns.tolist()
  column_names = data.columns

  column_names = data.columns.drop(target_col) #Drop target column

  df_header = pd.DataFrame(columns=column_names)          #
  df_header.to_csv(file_path, index=False)                # Menyimpan nama-nama kolom ke file CSV
  print(f"Nama kolom berhasil disimpan ke: {file_path}")  #

  #Menghapus target col dari daftar kolom
  if target_col in numeric_features:
        numeric_features.remove(target_col)
  if target_col in categorical_features:
      categorical_features.remove(target_col)

  numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('outlier', FunctionTransformer(cap_outlier)),
        ('scaler', StandardScaler())
  ])

  categorical_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
      ('encoder', OneHotEncoder(handle_unknown='ignore'))
  ])

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', numeric_transformer, numeric_features),
          ('cat', categorical_transformer, categorical_features)
      ]
  )

  preprocessor.set_output(transform="pandas")

  X = data.drop(columns=[target_col])
  y = data[target_col]

  #Split ds and transform
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  X_train = preprocessor.fit_transform(X_train)
  X_test = preprocessor.transform(X_test)

  #joblib.dump(preprocessor, save_path) #We don't need the dump right now.

  return X_train, X_test, y_train, y_test

os.makedirs("preprocessing/StellarClassification_preprocessing", exist_ok=True)

df = load_ds_file(file_path)
res = data_preprocessing(df=df,target_col=target_col,save_path=save_path,file_path=file_path)

file_names = ["X_train", "X_test", "y_train", "y_test"]
for i in range(len(res)):
    if i >= 2:
        res[i].to_csv(path + "preprocessing/StellarClassification_preprocessing/" + file_names[i] + ".csv",index=False)
    else:
        res[i].to_csv(path + "preprocessing/StellarClassification_preprocessing/" + file_names[i] + ".csv")