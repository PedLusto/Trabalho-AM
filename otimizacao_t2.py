import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error # Importar métrica

# Ignorar avisos futuros do XGBoost e Scikit-learn
warnings.filterwarnings('ignore', category=FutureWarning)

print("Iniciando script de otimização (Trabalho 2)...")
print("Carregando e processando os dados (mesmo pipeline do T1)...")

# --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO (IDÊNTICO AO T1) ---

try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'train.csv' não encontrado.")
    print("Por favor, baixe o 'train.csv' do Ames Housing (Kaggle) e coloque na mesma pasta.")
    exit()

# Remover espaços nos nomes das colunas (correção que fizemos no T1)
df.columns = df.columns.str.strip()

# 1.1 Transformação Logarítmica na Variável Alvo
# Usamos log1p para tratar valores 0 (embora não existam em SalePrice, é boa prática)
df['SalePrice'] = np.log1p(df['SalePrice'])

# 1.2 Tratamento de Outliers (Identificados na EDA do T1)
df = df.drop(df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 12.5)].index)

# 1.3 Definição dos Atributos
# Usamos os mesmos atributos que provaram ser importantes no T1
# (Vamos usar o dataset completo para a otimização, não apenas a seleção inicial)

# Identificar colunas numéricas e categóricas
numerical_features = df.select_dtypes(include=np.number).columns.drop(['PID', 'Order', 'SalePrice'])
categorical_features = df.select_dtypes(include='object').columns

# 1.4 Pipelines de Pré-processamento

# Pipeline para atributos numéricos:
# 1. Imputer: Preenche dados faltantes (NaN) com a mediana
# 2. Scaler: Normaliza os dados (StandardScaler)
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para atributos categóricos:
# 1. Imputer: Preenche dados faltantes (NaN) com a string 'Nenhum'
# 2. OneHotEncoder: Transforma categorias em colunas binárias (0 ou 1)
#    handle_unknown='ignore' evita erros se uma categoria rara não aparecer em um fold
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Nenhum')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 1.5 ColumnTransformer
# Junta os dois pipelines, aplicando cada um ao seu tipo de coluna
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough'
)

# 1.6 Preparando Dados de Treino
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Aplicar o pré-processamento
# (Usamos X_full e y_full para o GridSearchCV, ele faz a divisão interna)
print("Pré-processamento concluído. Ajustando dados...")
X_full = preprocessor.fit_transform(X)
y_full = y

print(f"Dimensões dos dados processados: {X_full.shape}")
print("-" * 40)

# --- 1.5. CÁLCULO DO BASELINE (T1) ---
# Precisamos dos scores "Antes" da otimização para comparar.

print("\nIniciando cálculo dos Baselines (T1)...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Baseline (T1) para Random Forest (parâmetros padrão)
rf_baseline = RandomForestRegressor(random_state=42)
rf_baseline_scores = cross_val_score(
    rf_baseline, 
    X_full, 
    y_full, 
    cv=kfold, 
    scoring='neg_root_mean_squared_error'
)
rf_baseline_rmse = -np.mean(rf_baseline_scores)
print(f"Baseline RMSE (RF - T1) (em log): {rf_baseline_rmse:.5f}")

# Baseline (T1) para XGBoost (parâmetros "manuais" do T1)
# (Usando os parâmetros do script gerar_graficos.py)
xgb_baseline = XGBRegressor(
    random_state=42, 
    objective='reg:squarederror', 
    eval_metric='rmse',
    n_estimators=500, 
    learning_rate=0.05,
    n_jobs=-1
)
xgb_baseline_scores = cross_val_score(
    xgb_baseline,
    X_full,
    y_full,
    cv=kfold,
    scoring='neg_root_mean_squared_error'
)
xgb_baseline_rmse = -np.mean(xgb_baseline_scores)
print(f"Baseline RMSE (XGB - T1) (em log): {xgb_baseline_rmse:.5f}")
print("-" * 40)


# --- 2. OTIMIZAÇÃO DE HIPERPARÂMETROS (TRABALHO 2) ---

# 2.1 Otimização do Random Forest
print("\nIniciando otimização (GridSearchCV) para Random Forest...")
print("Isso pode demorar vários minutos...")

# Grade de parâmetros para testar no RF
# (Pode ser maior, mas mantemos pequena para o tempo de execução do trabalho)
rf_grid = {
    'n_estimators': [150, 250],         # Número de árvores
    'max_depth': [None, 10, 20],      # Profundidade máxima
    'min_samples_leaf': [1, 2, 4]       # Mínimo de amostras por folha
}

rf = RandomForestRegressor(random_state=42)

# Configurando o GridSearchCV
# cv=5: Validação cruzada de 5 folds (como no T1)
# scoring='neg_root_mean_squared_error': Otimiza pelo RMSE (negativo, pois o grid search maximiza)
# n_jobs=-1: Usa todos os processadores do seu computador
rf_grid_search = GridSearchCV(
    estimator=rf,
    param_grid=rf_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2  # Mostra o progresso
)

# Rodando a busca
rf_grid_search.fit(X_full, y_full)

print("\nOtimização do Random Forest concluída.")
print(f"Melhores Parâmetros (RF - T2): {rf_grid_search.best_params_}")
# Multiplicamos por -1 para obter o RMSE (que é positivo)
print(f"Melhor RMSE (RF - T2) (em log): {-rf_grid_search.best_score_:.5f}")
print("-" * 40)


# 2.2 Otimização do XGBoost
print("\nIniciando otimização (GridSearchCV) para XGBoost...")
print("Isso também pode demorar bastante...")

# Grade de parâmetros para testar no XGBoost
xgb_grid = {
    'n_estimators': [200, 500],           # Número de árvores
    'learning_rate': [0.05, 0.1],         # Taxa de aprendizado
    'max_depth': [3, 5],                  # Profundidade máxima
    'subsample': [0.7, 1.0]               # Amostra de dados por árvore
}

xgb = XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='rmse')

xgb_grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Rodando a busca
xgb_grid_search.fit(X_full, y_full)

print("\nOtimização do XGBoost concluída.")
print(f"Melhores Parâmetros (XGB - T2): {xgb_grid_search.best_params_}")
print(f"Melhor RMSE (XGB - T2) (em log): {-xgb_grid_search.best_score_:.5f}")
print("-" * 40)

print("\nScript finalizado. Copie os resultados 'Baseline RMSE (T1)' e 'Melhor RMSE (T2)' de cada modelo.")